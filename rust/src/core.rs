//! The stateful index used by the virtual table. All *policy* lives here in
//! Rust: choosing the distance callback, quantizing/normalizing vectors,
//! orchestrating the per-query `ef`, applying the rowid filter, the load
//! data-size check and save/load orchestration. It calls hnswlib and `ops`
//! only through their FFI wrappers (`hnsw.rs`, `ops.rs`).

use std::cell::RefCell;
use std::path::Path;

use crate::hnsw::{Hnsw, Space};
use crate::ops;
use crate::vector_space::{DistanceType, VectorType};

pub use crate::hnsw::RowidFilter as SearchFilter;

/// Reinterprets a typed slice as its raw bytes (native layout) for the index,
/// which stores opaque per-vector byte blobs.
fn as_bytes<T>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
fn as_bytes_mut<T>(v: &mut [T]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, std::mem::size_of_val(v)) }
}

/// A vector encoded into the index's stored element type.
enum Stored {
    F32(Vec<f32>),
    U16(Vec<u16>),
}

impl Stored {
    fn bytes(&self) -> &[u8] {
        match self {
            Stored::F32(v) => as_bytes(v),
            Stored::U16(v) => as_bytes(v),
        }
    }
}

pub struct Index {
    // Interior mutability so `load` can swap the underlying hnswlib index behind
    // a shared reference (the registry hands out `&IndexEntry`). SQLite
    // serialises access per connection, so no locking is required. Declared
    // before `space` so it is dropped first (hnswlib caches the space pointer).
    index: RefCell<Hnsw>,
    space: Space,
    dim: usize,
    vector_type: VectorType,
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
        let data_size = dim * vector_type.element_size();
        let dist_func = ops::dist_func_for(distance_type, vector_type);
        let space = Space::new(dist_func, dim, data_size);
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
            normalize: distance_type == DistanceType::Cosine,
            max_elements,
            allow_replace_deleted,
        })
    }

    /// Quantizes and/or normalizes an f32 vector into the stored element type.
    fn encode(&self, v: &[f32]) -> Stored {
        match self.vector_type {
            VectorType::Float32 => {
                let mut buf = v.to_vec();
                if self.normalize {
                    ops::normalize_f32(&mut buf);
                }
                Stored::F32(buf)
            }
            VectorType::BFloat16 => {
                let mut buf = vec![0u16; self.dim];
                ops::quantize_bf16(v, &mut buf);
                if self.normalize {
                    ops::normalize_bf16(&mut buf);
                }
                Stored::U16(buf)
            }
            VectorType::Float16 => {
                let mut buf = vec![0u16; self.dim];
                ops::quantize_f16(v, &mut buf);
                if self.normalize {
                    ops::normalize_f16(&mut buf);
                }
                Stored::U16(buf)
            }
        }
    }

    pub fn add(&self, v: &[f32], rowid: u64) -> Result<(), String> {
        let stored = self.encode(v);
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
                if !index.get_data(rowid, as_bytes_mut(&mut buf)) {
                    return None;
                }
                Some(buf)
            }
            VectorType::BFloat16 => {
                let mut raw = vec![0u16; self.dim];
                if !index.get_data(rowid, as_bytes_mut(&mut raw)) {
                    return None;
                }
                let mut out = vec![0f32; self.dim];
                ops::bf16_to_f32(&raw, &mut out);
                Some(out)
            }
            VectorType::Float16 => {
                let mut raw = vec![0u16; self.dim];
                if !index.get_data(rowid, as_bytes_mut(&mut raw)) {
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
    ) -> Result<Vec<(f32, u64)>, String> {
        let stored = self.encode(query);
        let index = self.index.borrow();
        let saved_ef = index.get_ef();
        if let Some(ef) = ef_override {
            index.set_ef(ef);
        }
        let result = index.search(stored.bytes(), k, &filter);
        index.set_ef(saved_ef);
        result
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        if path.is_empty() {
            return Err("path must not be empty".to_string());
        }
        self.index.borrow().save(path)
    }

    /// Replaces the in-memory index with one loaded from `path`. The table's
    /// configured max_elements and allow_replace_deleted are preserved. On any
    /// error the current index is left unchanged; a per-vector data-size
    /// mismatch (wrong dimension or element type) is rejected.
    pub fn load(&self, path: &str) -> Result<(), String> {
        if path.is_empty() {
            return Err("path must not be empty".to_string());
        }
        if !Path::new(path).exists() {
            return Err(format!("index file does not exist: {}", path));
        }
        let new_index = Hnsw::load(
            &self.space,
            path,
            self.max_elements,
            self.allow_replace_deleted,
        )?;

        let expected = self.dim * self.vector_type.element_size();
        let file_size = new_index.per_vector_data_size();
        if file_size != expected {
            return Err(format!(
                "index data size mismatch: file has {} bytes per vector, table expects {}",
                file_size, expected
            ));
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
