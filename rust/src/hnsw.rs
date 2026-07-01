//! Safe Rust wrappers over the hnswlib C ABI (`cpp/core_shim.cpp`). Confines the
//! `unsafe` FFI calls to hnswlib. The index stores raw, already-encoded bytes;
//! all quantization/normalization/filtering *policy* lives in `core.rs`.

use std::collections::HashSet;
use std::ffi::CString;
use std::os::raw::{c_char, c_void};

use crate::ops::DistFunc;

#[repr(C)]
struct VlSpace {
    _private: [u8; 0],
}
#[repr(C)]
struct VlHnsw {
    _private: [u8; 0],
}

type VlFilterFunc = unsafe extern "C" fn(*mut c_void, u64) -> std::os::raw::c_int;

extern "C" {
    fn vl_hnsw_space_create(distfunc: DistFunc, dim: usize, data_size: usize) -> *mut VlSpace;
    fn vl_hnsw_space_free(space: *mut VlSpace);
    fn vl_hnsw_create(
        space: *mut VlSpace,
        max_elements: usize,
        m: usize,
        ef_construction: usize,
        random_seed: usize,
        allow_replace_deleted: std::os::raw::c_int,
        err: *mut *mut c_char,
    ) -> *mut VlHnsw;
    fn vl_hnsw_load(
        space: *mut VlSpace,
        path: *const c_char,
        max_elements: usize,
        allow_replace_deleted: std::os::raw::c_int,
        err: *mut *mut c_char,
    ) -> *mut VlHnsw;
    fn vl_hnsw_free(index: *mut VlHnsw);
    fn vl_hnsw_add_point(
        index: *mut VlHnsw,
        data: *const c_void,
        label: u64,
        replace_deleted: std::os::raw::c_int,
        err: *mut *mut c_char,
    ) -> std::os::raw::c_int;
    fn vl_hnsw_mark_delete(
        index: *mut VlHnsw,
        label: u64,
        err: *mut *mut c_char,
    ) -> std::os::raw::c_int;
    fn vl_hnsw_contains(index: *mut VlHnsw, label: u64) -> std::os::raw::c_int;
    fn vl_hnsw_get_data(
        index: *mut VlHnsw,
        label: u64,
        out: *mut c_void,
        nbytes: usize,
    ) -> std::os::raw::c_int;
    fn vl_hnsw_search(
        index: *mut VlHnsw,
        query: *const c_void,
        k: usize,
        filter: Option<VlFilterFunc>,
        filter_ctx: *mut c_void,
        out_dist: *mut f32,
        out_label: *mut u64,
        err: *mut *mut c_char,
    ) -> std::os::raw::c_int;
    fn vl_hnsw_save(index: *mut VlHnsw, path: *const c_char, err: *mut *mut c_char)
        -> std::os::raw::c_int;
    fn vl_hnsw_get_ef(index: *mut VlHnsw) -> usize;
    fn vl_hnsw_set_ef(index: *mut VlHnsw, ef: usize);
    fn vl_hnsw_per_vector_data_size(index: *mut VlHnsw) -> usize;
    fn vl_free_err(err: *mut c_char);
}

unsafe fn take_err(err: *mut c_char) -> String {
    if err.is_null() {
        return "unknown error".to_string();
    }
    let s = std::ffi::CStr::from_ptr(err).to_string_lossy().into_owned();
    vl_free_err(err);
    s
}

/// An hnswlib SpaceInterface built from a Rust distance callback. Must outlive
/// every `Hnsw` created from it (hnswlib caches its param pointer).
pub struct Space {
    ptr: *mut VlSpace,
}

unsafe impl Send for Space {}

impl Space {
    pub fn new(dist_func: DistFunc, dim: usize, data_size: usize) -> Space {
        let ptr = unsafe { vl_hnsw_space_create(dist_func, dim, data_size) };
        assert!(!ptr.is_null(), "space allocation failed");
        Space { ptr }
    }
}

impl Drop for Space {
    fn drop(&mut self) {
        unsafe { vl_hnsw_space_free(self.ptr) }
    }
}

/// Rowid filter applied during a k-NN search. Evaluated in Rust via a trampoline.
pub enum RowidFilter<'a> {
    None,
    In(&'a HashSet<u64>),
    Equals(u64),
}

unsafe extern "C" fn filter_trampoline(ctx: *mut c_void, label: u64) -> std::os::raw::c_int {
    let filter = &*(ctx as *const RowidFilter);
    let keep = match filter {
        RowidFilter::None => true,
        RowidFilter::In(set) => set.contains(&label),
        RowidFilter::Equals(id) => *id == label,
    };
    keep as std::os::raw::c_int
}

/// A raw hnswlib index. Stores already-encoded bytes; knows nothing about the
/// vector's logical type.
pub struct Hnsw {
    ptr: *mut VlHnsw,
}

unsafe impl Send for Hnsw {}

impl Hnsw {
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        space: &Space,
        max_elements: usize,
        m: usize,
        ef_construction: usize,
        random_seed: usize,
        allow_replace_deleted: bool,
    ) -> Result<Hnsw, String> {
        let mut err: *mut c_char = std::ptr::null_mut();
        let ptr = unsafe {
            vl_hnsw_create(
                space.ptr,
                max_elements,
                m,
                ef_construction,
                random_seed,
                allow_replace_deleted as std::os::raw::c_int,
                &mut err,
            )
        };
        if ptr.is_null() {
            return Err(unsafe { take_err(err) });
        }
        Ok(Hnsw { ptr })
    }

    pub fn load(
        space: &Space,
        path: &str,
        max_elements: usize,
        allow_replace_deleted: bool,
    ) -> Result<Hnsw, String> {
        let c = CString::new(path).map_err(|_| "invalid path".to_string())?;
        let mut err: *mut c_char = std::ptr::null_mut();
        let ptr = unsafe {
            vl_hnsw_load(
                space.ptr,
                c.as_ptr(),
                max_elements,
                allow_replace_deleted as std::os::raw::c_int,
                &mut err,
            )
        };
        if ptr.is_null() {
            return Err(unsafe { take_err(err) });
        }
        Ok(Hnsw { ptr })
    }

    /// Adds a point from already-encoded stored bytes.
    pub fn add_point(&self, data: &[u8], label: u64, replace_deleted: bool) -> Result<(), String> {
        let mut err: *mut c_char = std::ptr::null_mut();
        let rc = unsafe {
            vl_hnsw_add_point(
                self.ptr,
                data.as_ptr() as *const c_void,
                label,
                replace_deleted as std::os::raw::c_int,
                &mut err,
            )
        };
        if rc != 0 {
            return Err(unsafe { take_err(err) });
        }
        Ok(())
    }

    pub fn mark_delete(&self, label: u64) -> Result<(), String> {
        let mut err: *mut c_char = std::ptr::null_mut();
        let rc = unsafe { vl_hnsw_mark_delete(self.ptr, label, &mut err) };
        if rc != 0 {
            return Err(unsafe { take_err(err) });
        }
        Ok(())
    }

    pub fn contains(&self, label: u64) -> bool {
        unsafe { vl_hnsw_contains(self.ptr, label) != 0 }
    }

    /// Reads the raw stored bytes for `label` into `out`, or returns false if
    /// the label is absent.
    pub fn get_data(&self, label: u64, out: &mut [u8]) -> bool {
        let rc = unsafe {
            vl_hnsw_get_data(self.ptr, label, out.as_mut_ptr() as *mut c_void, out.len())
        };
        rc == 0
    }

    /// k-NN search over already-encoded query bytes, returning (distance, rowid)
    /// closer-first. Uses the index's current ef.
    pub fn search(
        &self,
        query: &[u8],
        k: usize,
        filter: &RowidFilter,
    ) -> Result<Vec<(f32, u64)>, String> {
        let mut distances = vec![0f32; k];
        let mut labels = vec![0u64; k];
        let mut err: *mut c_char = std::ptr::null_mut();

        let (cb, ctx): (Option<VlFilterFunc>, *mut c_void) = match filter {
            RowidFilter::None => (None, std::ptr::null_mut()),
            _ => (
                Some(filter_trampoline),
                filter as *const RowidFilter as *mut c_void,
            ),
        };

        let count = unsafe {
            vl_hnsw_search(
                self.ptr,
                query.as_ptr() as *const c_void,
                k,
                cb,
                ctx,
                distances.as_mut_ptr(),
                labels.as_mut_ptr(),
                &mut err,
            )
        };
        if count < 0 {
            return Err(unsafe { take_err(err) });
        }
        let count = count as usize;
        Ok((0..count).map(|i| (distances[i], labels[i])).collect())
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        let c = CString::new(path).map_err(|_| "invalid path".to_string())?;
        let mut err: *mut c_char = std::ptr::null_mut();
        let rc = unsafe { vl_hnsw_save(self.ptr, c.as_ptr(), &mut err) };
        if rc != 0 {
            return Err(unsafe { take_err(err) });
        }
        Ok(())
    }

    pub fn get_ef(&self) -> usize {
        unsafe { vl_hnsw_get_ef(self.ptr) }
    }
    pub fn set_ef(&self, ef: usize) {
        unsafe { vl_hnsw_set_ef(self.ptr, ef) }
    }
    pub fn per_vector_data_size(&self) -> usize {
        unsafe { vl_hnsw_per_vector_data_size(self.ptr) }
    }
}

impl Drop for Hnsw {
    fn drop(&mut self) {
        unsafe { vl_hnsw_free(self.ptr) }
    }
}
