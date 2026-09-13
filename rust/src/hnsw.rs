//! Checked Rust wrappers over the hnswlib C ABI. Native indices retain their
//! distance space, and every buffer is validated before crossing the boundary.
#![deny(unsafe_op_in_unsafe_fn)]

use std::collections::HashSet;
use std::ffi::{CStr, CString};
use std::io::{Read, Write};
use std::os::raw::{c_char, c_int, c_void};
use std::ptr::NonNull;
use std::rc::Rc;

use crate::ops::{self, DistFunc};
use crate::vector_space::{DistanceType, VectorType};

#[repr(C)]
struct VlSpace {
    _private: [u8; 0],
}
#[repr(C)]
struct VlHnsw {
    _private: [u8; 0],
}

/// The C-compatible result buffer is filled directly by the native search.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct SearchResult {
    pub distance: f32,
    pub rowid: u64,
}

impl SearchResult {
    pub fn new(distance: f32, rowid: u64) -> Self {
        Self { distance, rowid }
    }
}

type VlFilterFunc = unsafe extern "C" fn(*mut c_void, u64) -> c_int;

extern "C" {
    fn vl_hnsw_space_create(distfunc: DistFunc, dim: usize, data_size: usize) -> *mut VlSpace;
    fn vl_hnsw_space_free(space: *mut VlSpace);
    fn vl_hnsw_create(
        space: *mut VlSpace,
        max_elements: usize,
        m: usize,
        ef_construction: usize,
        random_seed: usize,
        allow_replace_deleted: c_int,
        err: *mut *mut c_char,
    ) -> *mut VlHnsw;
    fn vl_hnsw_load(
        space: *mut VlSpace,
        path: *const c_char,
        max_elements: usize,
        allow_replace_deleted: c_int,
        err: *mut *mut c_char,
    ) -> *mut VlHnsw;
    fn vl_hnsw_free(index: *mut VlHnsw);
    fn vl_hnsw_add_point(
        index: *mut VlHnsw,
        data: *const c_void,
        label: u64,
        replace_deleted: c_int,
        err: *mut *mut c_char,
    ) -> c_int;
    fn vl_hnsw_mark_delete(index: *mut VlHnsw, label: u64, err: *mut *mut c_char) -> c_int;
    fn vl_hnsw_contains(index: *mut VlHnsw, label: u64) -> c_int;
    fn vl_hnsw_get_data(index: *mut VlHnsw, label: u64, out: *mut c_void, nbytes: usize) -> c_int;
    fn vl_hnsw_search(
        index: *mut VlHnsw,
        query: *const c_void,
        k: usize,
        filter: Option<VlFilterFunc>,
        filter_ctx: *mut c_void,
        out: *mut SearchResult,
        count: *mut usize,
        err: *mut *mut c_char,
    ) -> c_int;
    fn vl_hnsw_save(index: *mut VlHnsw, path: *const c_char, err: *mut *mut c_char) -> c_int;
    fn vl_hnsw_get_ef(index: *mut VlHnsw) -> usize;
    fn vl_hnsw_set_ef(index: *mut VlHnsw, ef: usize);
    fn vl_hnsw_per_vector_data_size(index: *mut VlHnsw) -> usize;
    fn vl_hnsw_current_count(index: *mut VlHnsw) -> usize;
    fn vl_free_err(err: *mut c_char);
}

/// # Safety
/// `err` must be null or an owned, NUL-terminated allocation from this shim.
unsafe fn take_err(err: *mut c_char) -> String {
    if err.is_null() {
        return "native operation failed".to_string();
    }
    // SAFETY: the shim returns owned, NUL-terminated error strings allocated by
    // malloc; vl_free_err uses the matching native allocator exactly once.
    unsafe {
        let message = CStr::from_ptr(err).to_string_lossy().into_owned();
        vl_free_err(err);
        message
    }
}

struct SpaceOwner(NonNull<VlSpace>);

impl Drop for SpaceOwner {
    fn drop(&mut self) {
        // SAFETY: this is the sole owning allocation, retained by every index.
        unsafe { vl_hnsw_space_free(self.0.as_ptr()) }
    }
}

/// A supported distance space. Rc retains the native dimension parameter until
/// the final index has been destroyed and intentionally prevents Send/Sync.
#[derive(Clone)]
pub struct Space {
    owner: Rc<SpaceOwner>,
    data_size: usize,
    alignment: usize,
}

impl Space {
    pub fn new(distance: DistanceType, element: VectorType, dim: usize) -> Result<Self, String> {
        let data_size = dim
            .checked_mul(element.element_size())
            .filter(|&n| dim > 0 && n <= isize::MAX as usize)
            .ok_or_else(|| "dimension is zero or vector byte size is too large".to_string())?;
        // SAFETY: only the built-in callback matching this dimension and
        // element width can be selected; native allocation failures return null.
        let ptr =
            unsafe { vl_hnsw_space_create(ops::dist_func_for(distance, element), dim, data_size) };
        let ptr = NonNull::new(ptr).ok_or_else(|| "failed to allocate hnsw space".to_string())?;
        Ok(Self {
            owner: Rc::new(SpaceOwner(ptr)),
            data_size,
            alignment: element.element_size(),
        })
    }

    fn check_input(&self, data: &[u8]) -> Result<(), String> {
        if data.len() != self.data_size {
            return Err(format!(
                "stored vector has {} bytes; expected {}",
                data.len(),
                self.data_size
            ));
        }
        if !(data.as_ptr() as usize).is_multiple_of(self.alignment) {
            return Err("stored vector is not aligned for its element type".to_string());
        }
        Ok(())
    }
}

pub enum RowidFilter<'a> {
    None,
    In(&'a HashSet<u64>),
    Equals(u64),
}

unsafe extern "C" fn filter_trampoline(ctx: *mut c_void, label: u64) -> c_int {
    // SAFETY: search passes a live shared RowidFilter and the synchronous native
    // search neither retains it nor invokes the callback on another thread.
    let filter = unsafe { &*(ctx as *const RowidFilter) };
    (match filter {
        RowidFilter::None => true,
        RowidFilter::In(set) => set.contains(&label),
        RowidFilter::Equals(id) => *id == label,
    }) as c_int
}

/// A native index retaining the space used by its distance callbacks.
pub struct Hnsw {
    ptr: NonNull<VlHnsw>,
    space: Space,
}

impl Hnsw {
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        space: &Space,
        max_elements: usize,
        m: usize,
        ef_construction: usize,
        random_seed: usize,
        allow_replace_deleted: bool,
    ) -> Result<Self, String> {
        if !(2..=10_000).contains(&m) || ef_construction == 0 {
            return Err(
                "M must be between 2 and 10000, and ef_construction must be positive".to_string(),
            );
        }
        // HNSW uses u32 node identifiers and native-size labels. Check every
        // term in its level-zero allocation before native arithmetic occurs.
        let stride = m
            .checked_mul(8)
            .and_then(|n| n.checked_add(4))
            .and_then(|n| n.checked_add(space.data_size))
            .and_then(|n| n.checked_add(std::mem::size_of::<usize>()));
        let allocation = stride.and_then(|n| n.checked_mul(max_elements));
        if max_elements == 0
            || max_elements > u32::MAX as usize
            || allocation.filter(|&n| n <= isize::MAX as usize).is_none()
        {
            return Err("HNSW capacity or allocation size is out of range".to_string());
        }
        let mut err = std::ptr::null_mut();
        // SAFETY: the owned space and checked layout remain valid throughout
        // construction; the shim contains exceptions and reports allocation errors.
        let ptr = unsafe {
            vl_hnsw_create(
                space.owner.0.as_ptr(),
                max_elements,
                m,
                ef_construction,
                random_seed,
                allow_replace_deleted as c_int,
                &mut err,
            )
        };
        Self::from_native(ptr, space, err)
    }

    /// Copies a payload stream into a private snapshot before validating and
    /// loading it. The native parser never reopens the caller's mutable path.
    pub fn load(
        space: &Space,
        mut input: impl Read,
        expected_len: u64,
        max_elements: usize,
        allow_replace_deleted: bool,
    ) -> Result<Self, String> {
        let mut snapshot = tempfile::NamedTempFile::new().map_err(|e| e.to_string())?;
        let copied = std::io::copy(&mut input, &mut snapshot).map_err(|e| e.to_string())?;
        if copied != expected_len {
            return Err("index payload length changed while reading".to_string());
        }
        snapshot.flush().map_err(|e| e.to_string())?;
        // Close the file before C++ opens it (including on Windows). The private
        // path stays owned here until both validation and loading have finished.
        let snapshot = snapshot.into_temp_path();
        let path = snapshot
            .to_str()
            .ok_or_else(|| "temporary path is not UTF-8".to_string())?;
        let path = CString::new(path).map_err(|_| "invalid temporary path".to_string())?;
        let mut err = std::ptr::null_mut();
        // SAFETY: the native wrapper validates offsets/counts against the owned
        // space. It only reopens our completed private snapshot, never a source
        // path that another caller might change between validation and loading.
        let ptr = unsafe {
            vl_hnsw_load(
                space.owner.0.as_ptr(),
                path.as_ptr(),
                max_elements,
                allow_replace_deleted as c_int,
                &mut err,
            )
        };
        Self::from_native(ptr, space, err)
    }

    fn from_native(ptr: *mut VlHnsw, space: &Space, err: *mut c_char) -> Result<Self, String> {
        match NonNull::new(ptr) {
            Some(ptr) => Ok(Self {
                ptr,
                space: space.clone(),
            }),
            None => {
                // SAFETY: only shim constructor error pointers reach this path.
                Err(unsafe { take_err(err) })
            }
        }
    }

    pub fn add_point(&self, data: &[u8], label: u64, replace_deleted: bool) -> Result<(), String> {
        self.space.check_input(data)?;
        if usize::try_from(label).is_err() {
            return Err("rowid is out of range for this platform".to_string());
        }
        let mut err = std::ptr::null_mut();
        // SAFETY: the checked buffer has the space's complete aligned vector;
        // native code copies it before returning and does not retain the slice.
        let rc = unsafe {
            vl_hnsw_add_point(
                self.ptr.as_ptr(),
                data.as_ptr().cast(),
                label,
                replace_deleted as c_int,
                &mut err,
            )
        };
        Self::check_result(rc, err)
    }

    fn check_result(rc: c_int, err: *mut c_char) -> Result<(), String> {
        if rc == 0 {
            Ok(())
        } else {
            // SAFETY: error pointers come only from the native shim.
            Err(unsafe { take_err(err) })
        }
    }

    pub fn mark_delete(&self, label: u64) -> Result<(), String> {
        if usize::try_from(label).is_err() {
            return Err("rowid is out of range".to_string());
        }
        let mut err = std::ptr::null_mut();
        // SAFETY: self owns a live index; no borrowed native data escapes.
        let rc = unsafe { vl_hnsw_mark_delete(self.ptr.as_ptr(), label, &mut err) };
        Self::check_result(rc, err)
    }

    pub fn contains(&self, label: u64) -> bool {
        if usize::try_from(label).is_err() {
            return false;
        }
        // SAFETY: self owns a live index; the native lookup contains exceptions.
        unsafe { vl_hnsw_contains(self.ptr.as_ptr(), label) != 0 }
    }

    pub fn get_data(&self, label: u64, out: &mut [u8]) -> bool {
        if out.len() != self.space.data_size || usize::try_from(label).is_err() {
            return false;
        }
        // SAFETY: output has exactly the native vector width. memcpy permits
        // unaligned byte destinations and writes only within this unique slice.
        unsafe {
            vl_hnsw_get_data(self.ptr.as_ptr(), label, out.as_mut_ptr().cast(), out.len()) == 0
        }
    }

    pub fn search(
        &self,
        query: &[u8],
        k: usize,
        filter: &RowidFilter,
    ) -> Result<Vec<SearchResult>, String> {
        self.space.check_input(query)?;
        let capacity = k.min(self.current_count());
        if capacity == 0 {
            return Ok(Vec::new());
        }
        let mut result = Vec::new();
        result
            .try_reserve_exact(capacity)
            .map_err(|e| e.to_string())?;
        result.resize(capacity, SearchResult::default());
        let (callback, context): (Option<VlFilterFunc>, *mut c_void) = match filter {
            RowidFilter::None => (None, std::ptr::null_mut()),
            _ => (
                Some(filter_trampoline),
                filter as *const RowidFilter as *mut c_void,
            ),
        };
        let mut count = 0;
        let mut err = std::ptr::null_mut();
        // SAFETY: query is complete/aligned, the output has capacity initialized
        // repr(C) entries, and the filter lives until synchronous search returns.
        let rc = unsafe {
            vl_hnsw_search(
                self.ptr.as_ptr(),
                query.as_ptr().cast(),
                capacity,
                callback,
                context,
                result.as_mut_ptr(),
                &mut count,
                &mut err,
            )
        };
        Self::check_result(rc, err)?;
        if count > capacity {
            return Err("native search returned an invalid result count".to_string());
        }
        result.truncate(count);
        Ok(result)
    }

    /// Writes a checked raw HNSW payload. Atomic versioned persistence lives in core.
    pub fn save(&self, path: &str) -> Result<(), String> {
        let path = CString::new(path).map_err(|_| "invalid path".to_string())?;
        let mut err = std::ptr::null_mut();
        // SAFETY: path is NUL-terminated and the index is live for this call.
        let rc = unsafe { vl_hnsw_save(self.ptr.as_ptr(), path.as_ptr(), &mut err) };
        Self::check_result(rc, err)
    }

    pub fn get_ef(&self) -> usize {
        // SAFETY: reads a scalar from the live index, with no concurrent access.
        unsafe { vl_hnsw_get_ef(self.ptr.as_ptr()) }
    }
    pub fn set_ef(&self, ef: usize) {
        // SAFETY: writes a scalar in the live index; this type is not Sync.
        unsafe { vl_hnsw_set_ef(self.ptr.as_ptr(), ef) }
    }
    pub fn per_vector_data_size(&self) -> usize {
        // SAFETY: creation/loading validated the native offsets.
        unsafe { vl_hnsw_per_vector_data_size(self.ptr.as_ptr()) }
    }
    pub fn current_count(&self) -> usize {
        // SAFETY: reads the count of the live index.
        unsafe { vl_hnsw_current_count(self.ptr.as_ptr()) }
    }
}

impl Drop for Hnsw {
    fn drop(&mut self) {
        // SAFETY: the unique index allocation is destroyed before the retained
        // space field is dropped, so its cached dimension pointer stays valid.
        unsafe { vl_hnsw_free(self.ptr.as_ptr()) }
    }
}
