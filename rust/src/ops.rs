//! Safe Rust bindings to vectorlite's SIMD `ops`, plus the hnswlib distance
//! callbacks. `ops` is called only through FFI (it is not reimplemented here);
//! all *decisions* about which op to apply live in Rust (`core.rs`).

use std::os::raw::{c_char, c_void};

use crate::vector_space::{DistanceType, VectorType};

extern "C" {
    fn vl_ops_l2_sq_f32(a: *const f32, b: *const f32, n: usize) -> f32;
    fn vl_ops_l2_sq_bf16(a: *const u16, b: *const u16, n: usize) -> f32;
    fn vl_ops_l2_sq_f16(a: *const u16, b: *const u16, n: usize) -> f32;
    fn vl_ops_ip_dist_f32(a: *const f32, b: *const f32, n: usize) -> f32;
    fn vl_ops_ip_dist_bf16(a: *const u16, b: *const u16, n: usize) -> f32;
    fn vl_ops_ip_dist_f16(a: *const u16, b: *const u16, n: usize) -> f32;

    fn vl_ops_normalize_f32(inout: *mut f32, n: usize);
    fn vl_ops_normalize_bf16(inout: *mut u16, n: usize);
    fn vl_ops_normalize_f16(inout: *mut u16, n: usize);

    fn vl_ops_quantize_f32_to_bf16(input: *const f32, out: *mut u16, n: usize);
    fn vl_ops_quantize_f32_to_f16(input: *const f32, out: *mut u16, n: usize);
    fn vl_ops_bf16_to_f32(input: *const u16, out: *mut f32, n: usize);
    fn vl_ops_f16_to_f32(input: *const u16, out: *mut f32, n: usize);

    fn vl_ops_best_target() -> *const c_char;
}

// --- safe wrappers over the raw ops ---

pub fn l2_sq_f32(a: &[f32], b: &[f32]) -> f32 {
    unsafe { vl_ops_l2_sq_f32(a.as_ptr(), b.as_ptr(), a.len()) }
}
pub fn ip_dist_f32(a: &[f32], b: &[f32]) -> f32 {
    unsafe { vl_ops_ip_dist_f32(a.as_ptr(), b.as_ptr(), a.len()) }
}

pub fn normalize_f32(v: &mut [f32]) {
    unsafe { vl_ops_normalize_f32(v.as_mut_ptr(), v.len()) }
}
pub fn normalize_bf16(v: &mut [u16]) {
    unsafe { vl_ops_normalize_bf16(v.as_mut_ptr(), v.len()) }
}
pub fn normalize_f16(v: &mut [u16]) {
    unsafe { vl_ops_normalize_f16(v.as_mut_ptr(), v.len()) }
}

pub fn quantize_bf16(input: &[f32], out: &mut [u16]) {
    unsafe { vl_ops_quantize_f32_to_bf16(input.as_ptr(), out.as_mut_ptr(), input.len()) }
}
pub fn quantize_f16(input: &[f32], out: &mut [u16]) {
    unsafe { vl_ops_quantize_f32_to_f16(input.as_ptr(), out.as_mut_ptr(), input.len()) }
}
pub fn bf16_to_f32(input: &[u16], out: &mut [f32]) {
    unsafe { vl_ops_bf16_to_f32(input.as_ptr(), out.as_mut_ptr(), input.len()) }
}
pub fn f16_to_f32(input: &[u16], out: &mut [f32]) {
    unsafe { vl_ops_f16_to_f32(input.as_ptr(), out.as_mut_ptr(), input.len()) }
}

pub fn best_target() -> String {
    unsafe {
        let p = vl_ops_best_target();
        if p.is_null() {
            return "unknown".to_string();
        }
        std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}

// --- hnswlib distance callbacks ---
//
// hnswlib invokes these as `f(a, b, param)`, where `param` is the pointer the
// space adapter returns from get_dist_func_param(); the shim makes it point at
// the dimension (a `usize`). Each callback reads the dimension and forwards to
// the matching `ops` distance function on the stored element type.

unsafe fn dim_of(param: *const c_void) -> usize {
    *(param as *const usize)
}

pub unsafe extern "C" fn dist_l2_f32(
    a: *const c_void,
    b: *const c_void,
    param: *const c_void,
) -> f32 {
    vl_ops_l2_sq_f32(a as *const f32, b as *const f32, dim_of(param))
}
pub unsafe extern "C" fn dist_l2_bf16(
    a: *const c_void,
    b: *const c_void,
    param: *const c_void,
) -> f32 {
    vl_ops_l2_sq_bf16(a as *const u16, b as *const u16, dim_of(param))
}
pub unsafe extern "C" fn dist_l2_f16(
    a: *const c_void,
    b: *const c_void,
    param: *const c_void,
) -> f32 {
    vl_ops_l2_sq_f16(a as *const u16, b as *const u16, dim_of(param))
}
pub unsafe extern "C" fn dist_ip_f32(
    a: *const c_void,
    b: *const c_void,
    param: *const c_void,
) -> f32 {
    vl_ops_ip_dist_f32(a as *const f32, b as *const f32, dim_of(param))
}
pub unsafe extern "C" fn dist_ip_bf16(
    a: *const c_void,
    b: *const c_void,
    param: *const c_void,
) -> f32 {
    vl_ops_ip_dist_bf16(a as *const u16, b as *const u16, dim_of(param))
}
pub unsafe extern "C" fn dist_ip_f16(
    a: *const c_void,
    b: *const c_void,
    param: *const c_void,
) -> f32 {
    vl_ops_ip_dist_f16(a as *const u16, b as *const u16, dim_of(param))
}

/// The hnswlib distance-function pointer type.
pub type DistFunc = unsafe extern "C" fn(*const c_void, *const c_void, *const c_void) -> f32;

/// Selects the distance callback for a (metric, element-type) pair. Cosine uses
/// the inner-product function (vectors are normalized separately at insert and
/// query time), matching the C++ implementation.
pub fn dist_func_for(distance_type: DistanceType, vector_type: VectorType) -> DistFunc {
    use DistanceType::*;
    use VectorType::*;
    match (distance_type, vector_type) {
        (L2, Float32) => dist_l2_f32,
        (L2, BFloat16) => dist_l2_bf16,
        (L2, Float16) => dist_l2_f16,
        (InnerProduct | Cosine, Float32) => dist_ip_f32,
        (InnerProduct | Cosine, BFloat16) => dist_ip_bf16,
        (InnerProduct | Cosine, Float16) => dist_ip_f16,
    }
}
