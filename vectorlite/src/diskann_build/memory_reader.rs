//! Checked in-memory binary view used only by the public native medoid helper.
//! No serialized corpus copy, path lookup, filesystem provider or scratch file exists.

use std::io::{self, Read, Seek, SeekFrom};
use std::sync::{Arc, Mutex};

use diskann_providers::storage::StorageReadProvider;

use super::{add, mul, BuildError, BuildResult, Cancellation};

#[derive(Debug, Default, Clone, Copy)]
struct ReaderReport {
    bytes: u64,
    position: usize,
    seeks: usize,
    closed: bool,
}
#[derive(Debug, Default)]
struct Audit {
    opened: usize,
    reports: [ReaderReport; 3],
    first_error: Option<String>,
    cancelled: bool,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ReadStats {
    pub bytes_read: u64,
    pub full_scan_vectors: usize,
    pub readers_closed: usize,
}

#[derive(Debug)]
pub(super) struct MemoryStorage {
    input: Arc<Vec<f32>>,
    header: [u8; 8],
    length: usize,
    population: u32,
    dimension: usize,
    audit: Arc<Mutex<Audit>>,
    cancellation: Cancellation,
    #[cfg(test)]
    fault: Option<ReaderFault>,
}
impl MemoryStorage {
    pub const KEY: &'static str = "owned-snapshot";

    pub fn new(
        input: Arc<Vec<f32>>,
        population: u32,
        dimension: usize,
        cancellation: Cancellation,
    ) -> BuildResult<Self> {
        if dimension == 0
            || dimension > u32::MAX as usize
            || input.len() != mul(population as usize, dimension)?
        {
            return Err(BuildError::InvalidInput(
                "invalid native medoid memory-reader shape".into(),
            ));
        }
        let length = add(8, mul(input.len(), 4)?)?;
        if length > isize::MAX as usize {
            return Err(BuildError::Limit(
                "native medoid stream length overflow".into(),
            ));
        }
        let mut header = [0; 8];
        header[..4].copy_from_slice(&population.to_le_bytes());
        header[4..].copy_from_slice(&(dimension as u32).to_le_bytes());
        Ok(Self {
            input,
            header,
            length,
            population,
            dimension,
            audit: Arc::new(Mutex::new(Audit::default())),
            cancellation,
            #[cfg(test)]
            fault: None,
        })
    }

    /// The helper must have dropped all readers. On success, require its exact
    /// metadata/sample/full-scan pattern for pinned 0.60.0. In particular, validate
    /// a complete N-vector final scan rather than trusting error-as-EOF iteration.
    pub fn finish(&self, success: bool) -> BuildResult<ReadStats> {
        let audit = self
            .audit
            .lock()
            .map_err(|_| BuildError::InputRead("native medoid reader audit poisoned".into()))?;
        if let Some(error) = &audit.first_error {
            return if audit.cancelled {
                Err(BuildError::Cancelled)
            } else {
                Err(BuildError::InputRead(error.clone()))
            };
        }
        let closed = audit.reports[..audit.opened]
            .iter()
            .filter(|r| r.closed)
            .count();
        if closed != audit.opened {
            return Err(BuildError::InputRead(
                "native medoid reader escaped its operation".into(),
            ));
        }
        if success {
            let expected = self.length as u64;
            let metadata = audit.reports[0];
            let sample = audit.reports[1];
            let full = audit.reports[2];
            if audit.opened != 3
                || metadata.bytes != 8
                || metadata.position != 8
                || metadata.seeks != 0
                || full.bytes != expected
                || full.position != self.length
                || full.seeks != 0
                || sample.bytes < (8 + self.dimension * 4) as u64
            {
                return Err(BuildError::InputRead(
                    "native medoid reader cardinality/consumption mismatch".into(),
                ));
            }
            if self.population as usize <= diskann_providers::utils::MAX_MEDOID_SAMPLE_SIZE
                && (sample.bytes != expected || sample.position != self.length || sample.seeks != 0)
            {
                return Err(BuildError::InputRead(
                    "native medoid dense sample did not consume the full declared population"
                        .into(),
                ));
            }
        }
        let bytes_read = audit
            .reports
            .iter()
            .try_fold(0_u64, |a, r| a.checked_add(r.bytes))
            .ok_or_else(|| BuildError::InputRead("native medoid read counter overflow".into()))?;
        Ok(ReadStats {
            bytes_read,
            full_scan_vectors: if success { self.population as usize } else { 0 },
            readers_closed: closed,
        })
    }
    fn fail(&self, kind: io::ErrorKind, message: &str) -> io::Error {
        latch(&self.audit, false, message);
        io::Error::new(kind, message)
    }
    #[cfg(test)]
    pub fn set_fault(&mut self, fault: ReaderFault) {
        self.fault = Some(fault);
    }
}

impl StorageReadProvider for MemoryStorage {
    type Reader = MemoryReader;
    fn open_reader(&self, key: &str) -> io::Result<Self::Reader> {
        if key != Self::KEY {
            return Err(self.fail(
                io::ErrorKind::NotFound,
                "unknown native medoid memory object",
            ));
        }
        let mut audit = self
            .audit
            .lock()
            .map_err(|_| io::Error::other("native medoid reader audit poisoned"))?;
        let ordinal = audit.opened;
        if ordinal >= audit.reports.len() {
            drop(audit);
            return Err(self.fail(
                io::ErrorKind::InvalidData,
                "native medoid opened too many readers",
            ));
        }
        #[cfg(test)]
        if self.fault == Some(ReaderFault::Open(ordinal)) {
            drop(audit);
            return Err(self.fail(io::ErrorKind::Other, "injected medoid open failure"));
        }
        audit.opened += 1;
        drop(audit);
        Ok(MemoryReader {
            input: self.input.clone(),
            header: self.header,
            length: self.length,
            ordinal,
            report: ReaderReport::default(),
            audit: self.audit.clone(),
            cancellation: self.cancellation.clone(),
            #[cfg(test)]
            fault: self.fault,
        })
    }
    fn get_length(&self, key: &str) -> io::Result<u64> {
        if key != Self::KEY {
            return Err(self.fail(
                io::ErrorKind::NotFound,
                "unknown native medoid memory object",
            ));
        }
        #[cfg(test)]
        if self.fault == Some(ReaderFault::Length) {
            return Ok(self.length as u64 + 1);
        }
        u64::try_from(self.length).map_err(|_| {
            self.fail(
                io::ErrorKind::InvalidData,
                "native medoid memory length overflow",
            )
        })
    }
    fn exists(&self, key: &str) -> bool {
        key == Self::KEY
    }
}

fn latch(audit: &Mutex<Audit>, cancelled: bool, message: &str) {
    let mut audit = audit.lock().unwrap_or_else(|poison| poison.into_inner());
    if audit.first_error.is_none() {
        audit.first_error = Some(message.into());
        audit.cancelled = cancelled;
    }
}

#[derive(Debug)]
pub(super) struct MemoryReader {
    input: Arc<Vec<f32>>,
    header: [u8; 8],
    length: usize,
    ordinal: usize,
    report: ReaderReport,
    audit: Arc<Mutex<Audit>>,
    cancellation: Cancellation,
    #[cfg(test)]
    fault: Option<ReaderFault>,
}
impl MemoryReader {
    fn fail(&self, kind: io::ErrorKind, message: &str) -> io::Error {
        latch(&self.audit, false, message);
        io::Error::new(kind, message)
    }
}
impl Read for MemoryReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if buffer.is_empty() {
            return Ok(0);
        }
        if self.cancellation.is_cancelled() {
            latch(&self.audit, true, "native medoid memory read cancelled");
            // read_exact retries Interrupted forever. Cancellation is terminal
            // for this owned operation, with its distinct BuildError in the latch.
            return Err(io::Error::other("native medoid memory read cancelled"));
        }
        let take = buffer.len().min(self.length - self.report.position);
        #[cfg(test)]
        let take = if let Some(ReaderFault::Read { reader, at, eof }) = self.fault {
            if reader == self.ordinal {
                if self.report.position >= at {
                    let error =
                        self.fail(io::ErrorKind::UnexpectedEof, "injected medoid read failure");
                    return if eof { Ok(0) } else { Err(error) };
                }
                take.min(at - self.report.position)
            } else {
                take
            }
        } else {
            take
        };
        let mut copied = 0;
        let header_take = take.min(8_usize.saturating_sub(self.report.position));
        if header_take > 0 {
            buffer[..header_take].copy_from_slice(
                &self.header[self.report.position..self.report.position + header_take],
            );
            copied += header_take;
        }
        while copied < take {
            let byte = self.report.position + copied - 8;
            let offset = byte % 4;
            let count = (4 - offset).min(take - copied);
            let encoded = self.input[byte / 4].to_le_bytes();
            buffer[copied..copied + count].copy_from_slice(&encoded[offset..offset + count]);
            copied += count;
        }
        self.report.position = self.report.position.checked_add(take).ok_or_else(|| {
            self.fail(
                io::ErrorKind::InvalidData,
                "native medoid reader position overflow",
            )
        })?;
        self.report.bytes = self.report.bytes.checked_add(take as u64).ok_or_else(|| {
            self.fail(
                io::ErrorKind::InvalidData,
                "native medoid read counter overflow",
            )
        })?;
        Ok(take)
    }
}
impl Seek for MemoryReader {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        #[cfg(test)]
        if self.fault == Some(ReaderFault::Seek(self.ordinal)) {
            return Err(self.fail(io::ErrorKind::Other, "injected medoid seek failure"));
        }
        let position = match from {
            SeekFrom::Start(position) => i128::from(position),
            SeekFrom::End(delta) => self.length as i128 + i128::from(delta),
            SeekFrom::Current(delta) => self.report.position as i128 + i128::from(delta),
        };
        let position = usize::try_from(position)
            .ok()
            .filter(|&position| position <= self.length)
            .ok_or_else(|| {
                self.fail(
                    io::ErrorKind::InvalidInput,
                    "native medoid seek outside checked memory object",
                )
            })?;
        self.report.position = position;
        self.report.seeks = self.report.seeks.checked_add(1).ok_or_else(|| {
            self.fail(
                io::ErrorKind::InvalidData,
                "native medoid seek counter overflow",
            )
        })?;
        Ok(position as u64)
    }
}
impl Drop for MemoryReader {
    fn drop(&mut self) {
        self.report.closed = true;
        let mut audit = self
            .audit
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        audit.reports[self.ordinal] = self.report;
    }
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ReaderFault {
    Open(usize),
    Read { reader: usize, at: usize, eof: bool },
    Seek(usize),
    Length,
}
