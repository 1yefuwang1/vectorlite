//! The stateful index used by the virtual table. All *policy* lives here in
//! Rust: choosing the distance callback, quantizing/normalizing vectors,
//! orchestrating the per-query `ef`, applying the rowid filter, the load
//! data-size check and save/load orchestration. It calls hnswlib and `ops`
//! only through their FFI wrappers (`hnsw.rs`, `ops.rs`).

#![deny(unsafe_op_in_unsafe_fn)]

use std::borrow::Cow;
use std::cell::RefCell;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use crate::half::{Bf16Bits, F16Bits};
use crate::hnsw::{Hnsw, Space};
use crate::ops;
use crate::vector_space::{DistanceType, VectorType};

pub use crate::hnsw::{RowidFilter as SearchFilter, SearchResult};

/// A vector encoded into the index's stored element type. Unmodified f32
/// input is borrowed, so L2/IP insertions and queries do not copy it again.
enum Stored<'a> {
    F32(Cow<'a, [f32]>),
    F16(Vec<F16Bits>),
    Bf16(Vec<Bf16Bits>),
}

impl Stored<'_> {
    fn bytes(&self) -> &[u8] {
        match self {
            Stored::F32(v) => bytemuck::cast_slice(v),
            Stored::F16(v) => bytemuck::cast_slice(v),
            Stored::Bf16(v) => bytemuck::cast_slice(v),
        }
    }
}

// Version 1 envelope: magic, little-endian version and dimension, element,
// metric, normalization, native-layout identifier, and raw payload byte length.
// Legacy untyped HNSW files are rejected: their element/metric cannot be inferred.
const FILE_MAGIC: &[u8; 8] = b"VLTIDX01";
const HEADER_LEN: usize = 32;

pub struct Index {
    // Interior mutability so `load` can swap the underlying hnswlib index behind
    // a shared reference (the registry hands out `&IndexEntry`). SQLite
    // serialises access per connection, so no locking is required. Each Hnsw
    // retains its own space owner, including during an index replacement.
    index: RefCell<Hnsw>,
    space: Space,
    dim: usize,
    vector_type: VectorType,
    distance_type: DistanceType,
    normalize: bool,
    max_elements: usize,
    allow_replace_deleted: bool,
}

impl Index {
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        dim: usize,
        distance_type: DistanceType,
        vector_type: VectorType,
        max_elements: usize,
        m: usize,
        ef_construction: usize,
        random_seed: usize,
        allow_replace_deleted: bool,
    ) -> Result<Index, String> {
        if dim == 0 {
            return Err("Dimension must be greater than 0".to_string());
        }
        let space = Space::new(distance_type, vector_type, dim)?;
        let index = Hnsw::create(
            &space,
            max_elements,
            m,
            ef_construction,
            random_seed,
            allow_replace_deleted,
        )?;
        Ok(Index {
            index: RefCell::new(index),
            space,
            dim,
            vector_type,
            distance_type,
            normalize: distance_type == DistanceType::Cosine,
            max_elements,
            allow_replace_deleted,
        })
    }

    /// Quantizes and/or normalizes an f32 vector into the stored element type.
    fn encode<'a>(&self, v: &'a [f32]) -> Result<Stored<'a>, String> {
        if v.len() != self.dim {
            return Err(format!(
                "dimension mismatch: expected {}, got {}",
                self.dim,
                v.len()
            ));
        }
        Ok(match self.vector_type {
            VectorType::Float32 => {
                let mut buf = Cow::Borrowed(v);
                if self.normalize {
                    ops::normalize_f32(buf.to_mut());
                }
                Stored::F32(buf)
            }
            VectorType::BFloat16 => {
                let mut buf = vec![Bf16Bits::default(); self.dim];
                ops::quantize_bf16(v, &mut buf);
                if self.normalize {
                    ops::normalize_bf16(&mut buf);
                }
                Stored::Bf16(buf)
            }
            VectorType::Float16 => {
                let mut buf = vec![F16Bits::default(); self.dim];
                ops::quantize_f16(v, &mut buf);
                if self.normalize {
                    ops::normalize_f16(&mut buf);
                }
                Stored::F16(buf)
            }
        })
    }

    pub fn add(&self, v: &[f32], rowid: u64) -> Result<(), String> {
        let stored = self.encode(v)?;
        self.index
            .borrow()
            .add_point(stored.bytes(), rowid, self.allow_replace_deleted)
    }

    pub fn mark_delete(&self, rowid: u64) -> Result<(), String> {
        self.index.borrow().mark_delete(rowid)
    }

    pub fn contains(&self, rowid: u64) -> bool {
        self.index.borrow().contains(rowid)
    }

    /// Reads a stored vector back as f32 (dequantizing as needed), or `None` if
    /// the rowid is absent.
    pub fn get_vector(&self, rowid: u64) -> Option<Vec<f32>> {
        let index = self.index.borrow();
        match self.vector_type {
            VectorType::Float32 => {
                let mut buf = vec![0f32; self.dim];
                if !index.get_data(rowid, bytemuck::cast_slice_mut(&mut buf)) {
                    return None;
                }
                Some(buf)
            }
            VectorType::BFloat16 => {
                let mut raw = vec![Bf16Bits::default(); self.dim];
                if !index.get_data(rowid, bytemuck::cast_slice_mut(&mut raw)) {
                    return None;
                }
                let mut out = vec![0f32; self.dim];
                ops::bf16_to_f32(&raw, &mut out);
                Some(out)
            }
            VectorType::Float16 => {
                let mut raw = vec![F16Bits::default(); self.dim];
                if !index.get_data(rowid, bytemuck::cast_slice_mut(&mut raw)) {
                    return None;
                }
                let mut out = vec![0f32; self.dim];
                ops::f16_to_f32(&raw, &mut out);
                Some(out)
            }
        }
    }

    /// k-NN search. Applies the per-query `ef` override (restoring the prior
    /// value afterwards so it does not leak into later queries) and the rowid
    /// filter.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_override: Option<usize>,
        filter: SearchFilter,
    ) -> Result<Vec<SearchResult>, String> {
        let stored = self.encode(query)?;
        let index = self.index.borrow();
        let saved_ef = index.get_ef();
        if let Some(ef) = ef_override {
            index.set_ef(ef);
        }
        let result = index.search(stored.bytes(), k, &filter);
        index.set_ef(saved_ef);
        result
    }

    fn descriptor(&self) -> [u8; 24] {
        let mut descriptor = [0u8; 24];
        descriptor[..8].copy_from_slice(FILE_MAGIC);
        descriptor[8..12].copy_from_slice(&1u32.to_le_bytes());
        descriptor[12..20].copy_from_slice(&(self.dim as u64).to_le_bytes());
        descriptor[20] = self.vector_type as u8;
        descriptor[21] = self.distance_type as u8;
        descriptor[22] = u8::from(self.normalize);
        // HNSW payloads retain their native layout; reject incompatible hosts.
        descriptor[23] =
            (std::mem::size_of::<usize>() as u8) * 2 + u8::from(cfg!(target_endian = "big"));
        descriptor
    }

    /// Writes a versioned, typed index to a sibling temporary file, checks and
    /// syncs every write, and atomically replaces the destination on success.
    pub fn save(&self, path: &str) -> Result<(), String> {
        if path.is_empty() {
            return Err("path must not be empty".to_string());
        }
        self.save_to(Path::new(path))
            .map_err(|e| format!("failed to save index: {e}"))
    }

    fn save_to(&self, path: &Path) -> Result<(), String> {
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        #[cfg(unix)]
        let destination_permissions = match std::fs::metadata(path) {
            Ok(metadata) if metadata.is_file() => Some(metadata.permissions()),
            Ok(_) => None,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(error.to_string()),
        };
        // Closing the named payload file before native code opens it also works
        // on Windows. TempPath removes it on every success/error return.
        let payload = tempfile::NamedTempFile::new_in(parent)
            .map_err(|e| e.to_string())?
            .into_temp_path();
        let payload_path = payload
            .to_str()
            .ok_or_else(|| "temporary path is not UTF-8".to_string())?;
        self.index.borrow().save(payload_path)?;
        let mut input = File::open(&payload).map_err(|e| e.to_string())?;
        let payload_len = input.metadata().map_err(|e| e.to_string())?.len();
        let mut output = tempfile::NamedTempFile::new_in(parent).map_err(|e| e.to_string())?;
        output
            .write_all(&self.descriptor())
            .map_err(|e| e.to_string())?;
        output
            .write_all(&payload_len.to_le_bytes())
            .map_err(|e| e.to_string())?;
        let written = std::io::copy(&mut input, &mut output).map_err(|e| e.to_string())?;
        if written != payload_len {
            return Err("native payload length changed during save".to_string());
        }
        #[cfg(unix)]
        if let Some(permissions) = destination_permissions {
            output
                .as_file()
                .set_permissions(permissions)
                .map_err(|e| e.to_string())?;
        }
        output.flush().map_err(|e| e.to_string())?;
        output.as_file().sync_all().map_err(|e| e.to_string())?;
        output.persist(path).map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Replaces the in-memory index only after validating the versioned vector
    /// descriptor and the native payload. To migrate legacy untyped files,
    /// export vectors and rowids with the old extension, then reinsert them with
    /// this version and save again; their format cannot be inferred safely.
    pub fn load(&self, path: &str) -> Result<(), String> {
        if path.is_empty() {
            return Err("path must not be empty".to_string());
        }
        let mut input = File::open(path).map_err(|e| format!("cannot open index file: {e}"))?;
        let mut header = [0u8; HEADER_LEN];
        input.read_exact(&mut header).map_err(|_| {
            "unsupported or truncated index; for legacy untyped HNSW files, export vectors and rowids with the old extension and reinsert them with this version"
                .to_string()
        })?;
        if &header[..8] != FILE_MAGIC {
            return Err(
                "unsupported index format; for legacy untyped HNSW files, export vectors and rowids with the old extension and reinsert them with this version"
                    .to_string(),
            );
        }
        if header[..24] != self.descriptor() {
            return Err("index descriptor mismatch: version, dimension, element type, metric, normalization, and native layout must match the table".to_string());
        }
        let payload_len =
            u64::from_le_bytes(header[24..32].try_into().expect("fixed header width"));
        if payload_len.checked_add(HEADER_LEN as u64)
            != Some(input.metadata().map_err(|e| e.to_string())?.len())
        {
            return Err("index payload length does not match the envelope".to_string());
        }
        let new_index = Hnsw::load(
            &self.space,
            input.take(payload_len),
            payload_len,
            self.max_elements,
            self.allow_replace_deleted,
        )?;
        if new_index.per_vector_data_size() != self.dim * self.vector_type.element_size() {
            return Err("native index data size does not match the table".to_string());
        }
        *self.index.borrow_mut() = new_index;
        Ok(())
    }
}

/// Computes the distance between two equal-length f32 vectors via `ops`.
/// Cosine normalizes both inputs first, matching the C++ implementation.
pub fn distance(a: &[f32], b: &[f32], distance_type: DistanceType) -> Option<f32> {
    if a.len() != b.len() {
        return None;
    }
    match distance_type {
        DistanceType::L2 => Some(ops::l2_sq_f32(a, b)),
        DistanceType::InnerProduct => Some(ops::ip_dist_f32(a, b)),
        DistanceType::Cosine => {
            let mut na = a.to_vec();
            let mut nb = b.to_vec();
            ops::normalize_f32(&mut na);
            ops::normalize_f32(&mut nb);
            Some(ops::ip_dist_f32(&na, &nb))
        }
    }
}

/// Returns the best SIMD target chosen by Highway at runtime.
pub fn best_target() -> String {
    ops::best_target()
}

#[cfg(test)]
#[path = "core_tests.rs"]
mod tests;
