//! Safe Rust wrapper over the C ABI exposed by `cpp/core_shim.{h,cpp}`.
//!
//! The C++ core owns hnswlib, the vectorlite SIMD spaces and quantization. This
//! module confines the `unsafe` FFI calls and exposes an idiomatic interface.

use std::ffi::CString;
use std::os::raw::{c_char, c_float, c_int};

use crate::vector_space::{DistanceType, VectorType};

#[repr(C)]
struct VlIndex {
    _private: [u8; 0],
}

// Filter kinds, must match VlFilterKind in core_shim.h.
const VL_FILTER_NONE: c_int = 0;
const VL_FILTER_IN: c_int = 1;
const VL_FILTER_EQ: c_int = 2;

extern "C" {
    fn vl_index_create(
        dim: usize,
        distance_type: c_int,
        vector_type: c_int,
        max_elements: usize,
        m: usize,
        ef_construction: usize,
        random_seed: usize,
        allow_replace_deleted: c_int,
        err: *mut *mut c_char,
    ) -> *mut VlIndex;
    fn vl_index_free(index: *mut VlIndex);
    fn vl_index_dim(index: *const VlIndex) -> usize;
    #[allow(dead_code)]
    fn vl_index_data_size(index: *const VlIndex) -> usize;
    fn vl_index_add(
        index: *mut VlIndex,
        data: *const c_float,
        len: usize,
        rowid: u64,
        err: *mut *mut c_char,
    ) -> c_int;
    fn vl_index_mark_delete(index: *mut VlIndex, rowid: u64, err: *mut *mut c_char) -> c_int;
    fn vl_index_contains(index: *const VlIndex, rowid: u64) -> c_int;
    fn vl_index_get_vector(index: *const VlIndex, rowid: u64, out: *mut c_float) -> c_int;
    #[allow(clippy::too_many_arguments)]
    fn vl_index_search(
        index: *mut VlIndex,
        query: *const c_float,
        len: usize,
        k: usize,
        ef_override: usize,
        filter_kind: c_int,
        filter_rowids: *const u64,
        filter_count: usize,
        out_distances: *mut c_float,
        out_rowids: *mut u64,
        err: *mut *mut c_char,
    ) -> c_int;
    fn vl_index_save(index: *mut VlIndex, path: *const c_char, err: *mut *mut c_char) -> c_int;
    fn vl_index_load(index: *mut VlIndex, path: *const c_char, err: *mut *mut c_char) -> c_int;
    fn vl_distance(
        a: *const c_float,
        b: *const c_float,
        dim: usize,
        distance_type: c_int,
        out: *mut c_float,
    ) -> c_int;
    fn vl_best_target() -> *const c_char;
    fn vl_free_err(err: *mut c_char);
}

/// Takes ownership of a C-allocated error string and returns it as a `String`.
unsafe fn take_err(err: *mut c_char) -> String {
    if err.is_null() {
        return "unknown error".to_string();
    }
    let s = std::ffi::CStr::from_ptr(err).to_string_lossy().into_owned();
    vl_free_err(err);
    s
}

/// Rowid filter applied during a k-NN search.
pub enum SearchFilter<'a> {
    None,
    In(&'a [u64]),
    Equals(u64),
}

/// An owned HNSW index plus its vector space, living in the C++ core.
pub struct Index {
    ptr: *mut VlIndex,
}

// The index is only ever touched while SQLite holds the per-connection lock, so
// it is effectively single-threaded from our perspective.
unsafe impl Send for Index {}

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
        let mut err: *mut c_char = std::ptr::null_mut();
        let ptr = unsafe {
            vl_index_create(
                dim,
                distance_type as c_int,
                vector_type as c_int,
                max_elements,
                m,
                ef_construction,
                random_seed,
                allow_replace_deleted as c_int,
                &mut err,
            )
        };
        if ptr.is_null() {
            return Err(unsafe { take_err(err) });
        }
        Ok(Index { ptr })
    }

    pub fn dim(&self) -> usize {
        unsafe { vl_index_dim(self.ptr) }
    }

    #[allow(dead_code)]
    pub fn data_size(&self) -> usize {
        unsafe { vl_index_data_size(self.ptr) }
    }

    pub fn add(&self, data: &[f32], rowid: u64) -> Result<(), String> {
        let mut err: *mut c_char = std::ptr::null_mut();
        let rc = unsafe {
            vl_index_add(self.ptr, data.as_ptr(), data.len(), rowid, &mut err)
        };
        if rc != 0 {
            return Err(unsafe { take_err(err) });
        }
        Ok(())
    }

    pub fn mark_delete(&self, rowid: u64) -> Result<(), String> {
        let mut err: *mut c_char = std::ptr::null_mut();
        let rc = unsafe { vl_index_mark_delete(self.ptr, rowid, &mut err) };
        if rc != 0 {
            return Err(unsafe { take_err(err) });
        }
        Ok(())
    }

    pub fn contains(&self, rowid: u64) -> bool {
        unsafe { vl_index_contains(self.ptr, rowid) != 0 }
    }

    /// Reads a stored vector back as f32, or `None` if the rowid is absent.
    pub fn get_vector(&self, rowid: u64) -> Option<Vec<f32>> {
        let mut out = vec![0f32; self.dim()];
        let rc = unsafe { vl_index_get_vector(self.ptr, rowid, out.as_mut_ptr()) };
        if rc != 0 {
            return None;
        }
        Some(out)
    }

    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_override: Option<usize>,
        filter: SearchFilter,
    ) -> Result<Vec<(f32, u64)>, String> {
        let (kind, rowids): (c_int, Vec<u64>) = match filter {
            SearchFilter::None => (VL_FILTER_NONE, Vec::new()),
            SearchFilter::In(ids) => (VL_FILTER_IN, ids.to_vec()),
            SearchFilter::Equals(id) => (VL_FILTER_EQ, vec![id]),
        };
        let mut distances = vec![0f32; k];
        let mut out_rowids = vec![0u64; k];
        let mut err: *mut c_char = std::ptr::null_mut();
        let count = unsafe {
            vl_index_search(
                self.ptr,
                query.as_ptr(),
                query.len(),
                k,
                ef_override.unwrap_or(0),
                kind,
                rowids.as_ptr(),
                rowids.len(),
                distances.as_mut_ptr(),
                out_rowids.as_mut_ptr(),
                &mut err,
            )
        };
        if count < 0 {
            return Err(unsafe { take_err(err) });
        }
        let count = count as usize;
        Ok((0..count).map(|i| (distances[i], out_rowids[i])).collect())
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        let c = CString::new(path).map_err(|_| "invalid path".to_string())?;
        let mut err: *mut c_char = std::ptr::null_mut();
        let rc = unsafe { vl_index_save(self.ptr, c.as_ptr(), &mut err) };
        if rc != 0 {
            return Err(unsafe { take_err(err) });
        }
        Ok(())
    }

    pub fn load(&self, path: &str) -> Result<(), String> {
        let c = CString::new(path).map_err(|_| "invalid path".to_string())?;
        let mut err: *mut c_char = std::ptr::null_mut();
        let rc = unsafe { vl_index_load(self.ptr, c.as_ptr(), &mut err) };
        if rc != 0 {
            return Err(unsafe { take_err(err) });
        }
        Ok(())
    }
}

impl Drop for Index {
    fn drop(&mut self) {
        unsafe { vl_index_free(self.ptr) }
    }
}

/// Computes the distance between two equal-length f32 vectors.
pub fn distance(a: &[f32], b: &[f32], distance_type: DistanceType) -> Option<f32> {
    if a.len() != b.len() {
        return None;
    }
    let mut out: f32 = 0.0;
    let rc = unsafe {
        vl_distance(
            a.as_ptr(),
            b.as_ptr(),
            a.len(),
            distance_type as c_int,
            &mut out,
        )
    };
    if rc != 0 {
        return None;
    }
    Some(out)
}

/// Returns the best SIMD target chosen by Highway at runtime.
pub fn best_target() -> String {
    unsafe {
        let p = vl_best_target();
        if p.is_null() {
            return "unknown".to_string();
        }
        std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}
