//! Safe Rust bindings to vectorlite's SIMD `ops`, plus the hnswlib distance
//! callbacks. `ops` is called only through FFI (it is not reimplemented here);
//! all *decisions* about which op to apply live in Rust (`core.rs`).

#![deny(unsafe_op_in_unsafe_fn)]

use std::os::raw::{c_char, c_void};

use crate::half::{Bf16Bits, F16Bits};
use crate::vector_space::{DistanceType, VectorType};

// The transparent half types have the layout of the C ABI's uint16_t.
// Keeping the distinct pointee types here prevents mixing formats in Rust.
extern "C" {
    fn vl_ops_l2_sq_f32(a: *const f32, b: *const f32, n: usize) -> f32;
    fn vl_ops_l2_sq_bf16(a: *const Bf16Bits, b: *const Bf16Bits, n: usize) -> f32;
    fn vl_ops_l2_sq_f16(a: *const F16Bits, b: *const F16Bits, n: usize) -> f32;
    fn vl_ops_ip_dist_f32(a: *const f32, b: *const f32, n: usize) -> f32;
    fn vl_ops_ip_dist_bf16(a: *const Bf16Bits, b: *const Bf16Bits, n: usize) -> f32;
    fn vl_ops_ip_dist_f16(a: *const F16Bits, b: *const F16Bits, n: usize) -> f32;

    fn vl_ops_normalize_f32(inout: *mut f32, n: usize);
    fn vl_ops_normalize_bf16(inout: *mut Bf16Bits, n: usize);
    fn vl_ops_normalize_f16(inout: *mut F16Bits, n: usize);

    fn vl_ops_quantize_f32_to_bf16(input: *const f32, out: *mut Bf16Bits, n: usize);
    fn vl_ops_quantize_f32_to_f16(input: *const f32, out: *mut F16Bits, n: usize);
    fn vl_ops_bf16_to_f32(input: *const Bf16Bits, out: *mut f32, n: usize);
    fn vl_ops_f16_to_f32(input: *const F16Bits, out: *mut f32, n: usize);

    fn vl_ops_best_target() -> *const c_char;
}

// --- safe wrappers over the raw ops ---

pub fn l2_sq_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "distance vectors must have equal lengths");
    // SAFETY: typed slices are aligned and the wrapper has checked all lengths.
    unsafe { vl_ops_l2_sq_f32(a.as_ptr(), b.as_ptr(), a.len()) }
}
pub fn ip_dist_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "distance vectors must have equal lengths");
    // SAFETY: typed slices are aligned and the wrapper has checked all lengths.
    unsafe { vl_ops_ip_dist_f32(a.as_ptr(), b.as_ptr(), a.len()) }
}

pub fn normalize_f32(v: &mut [f32]) {
    // SAFETY: typed slices are aligned and the wrapper has checked all lengths.
    unsafe { vl_ops_normalize_f32(v.as_mut_ptr(), v.len()) }
}
pub fn normalize_bf16(v: &mut [Bf16Bits]) {
    // SAFETY: typed slices are aligned and the wrapper has checked all lengths.
    unsafe { vl_ops_normalize_bf16(v.as_mut_ptr(), v.len()) }
}
pub fn normalize_f16(v: &mut [F16Bits]) {
    // SAFETY: typed slices are aligned and the wrapper has checked all lengths.
    unsafe { vl_ops_normalize_f16(v.as_mut_ptr(), v.len()) }
}

pub fn quantize_bf16(input: &[f32], out: &mut [Bf16Bits]) {
    assert_eq!(
        input.len(),
        out.len(),
        "conversion buffers must have equal lengths"
    );
    // SAFETY: typed slices are aligned and the wrapper has checked all lengths.
    unsafe { vl_ops_quantize_f32_to_bf16(input.as_ptr(), out.as_mut_ptr(), input.len()) }
}
pub fn quantize_f16(input: &[f32], out: &mut [F16Bits]) {
    assert_eq!(
        input.len(),
        out.len(),
        "conversion buffers must have equal lengths"
    );
    // SAFETY: typed slices are aligned and the wrapper has checked all lengths.
    unsafe { vl_ops_quantize_f32_to_f16(input.as_ptr(), out.as_mut_ptr(), input.len()) }
}
pub fn bf16_to_f32(input: &[Bf16Bits], out: &mut [f32]) {
    assert_eq!(
        input.len(),
        out.len(),
        "conversion buffers must have equal lengths"
    );
    // SAFETY: typed slices are aligned and the wrapper has checked all lengths.
    unsafe { vl_ops_bf16_to_f32(input.as_ptr(), out.as_mut_ptr(), input.len()) }
}
pub fn f16_to_f32(input: &[F16Bits], out: &mut [f32]) {
    assert_eq!(
        input.len(),
        out.len(),
        "conversion buffers must have equal lengths"
    );
    // SAFETY: typed slices are aligned and the wrapper has checked all lengths.
    unsafe { vl_ops_f16_to_f32(input.as_ptr(), out.as_mut_ptr(), input.len()) }
}

pub fn best_target() -> String {
    // SAFETY: Highway returns a static NUL-terminated target name.
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
    // SAFETY: the owning Space keeps its aligned dimension parameter live.
    unsafe { *(param as *const usize) }
}

unsafe extern "C" fn dist_l2_f32(a: *const c_void, b: *const c_void, param: *const c_void) -> f32 {
    // SAFETY: HNSW supplies complete vectors in the configured element type;
    // its retained Space owns the dimension parameter throughout the call.
    unsafe { vl_ops_l2_sq_f32(a as *const f32, b as *const f32, dim_of(param)) }
}
unsafe extern "C" fn dist_l2_bf16(a: *const c_void, b: *const c_void, param: *const c_void) -> f32 {
    // SAFETY: HNSW supplies complete vectors in the configured element type;
    // its retained Space owns the dimension parameter throughout the call.
    unsafe { vl_ops_l2_sq_bf16(a as *const Bf16Bits, b as *const Bf16Bits, dim_of(param)) }
}
unsafe extern "C" fn dist_l2_f16(a: *const c_void, b: *const c_void, param: *const c_void) -> f32 {
    // SAFETY: HNSW supplies complete vectors in the configured element type;
    // its retained Space owns the dimension parameter throughout the call.
    unsafe { vl_ops_l2_sq_f16(a as *const F16Bits, b as *const F16Bits, dim_of(param)) }
}
unsafe extern "C" fn dist_ip_f32(a: *const c_void, b: *const c_void, param: *const c_void) -> f32 {
    // SAFETY: HNSW supplies complete vectors in the configured element type;
    // its retained Space owns the dimension parameter throughout the call.
    unsafe { vl_ops_ip_dist_f32(a as *const f32, b as *const f32, dim_of(param)) }
}
unsafe extern "C" fn dist_ip_bf16(a: *const c_void, b: *const c_void, param: *const c_void) -> f32 {
    // SAFETY: HNSW supplies complete vectors in the configured element type;
    // its retained Space owns the dimension parameter throughout the call.
    unsafe { vl_ops_ip_dist_bf16(a as *const Bf16Bits, b as *const Bf16Bits, dim_of(param)) }
}
unsafe extern "C" fn dist_ip_f16(a: *const c_void, b: *const c_void, param: *const c_void) -> f32 {
    // SAFETY: HNSW supplies complete vectors in the configured element type;
    // its retained Space owns the dimension parameter throughout the call.
    unsafe { vl_ops_ip_dist_f16(a as *const F16Bits, b as *const F16Bits, dim_of(param)) }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_conversion_uses_binary16_encoding() {
        let input = [0.0, 1.0, -2.0, 1.5];
        let mut encoded = [F16Bits::default(); 4];
        quantize_f16(&input, &mut encoded);
        assert_eq!(
            bytemuck::cast_slice::<F16Bits, u16>(&encoded),
            &[0x0000, 0x3c00, 0xc000, 0x3e00]
        );
        let mut decoded = [0.0; 4];
        f16_to_f32(&encoded, &mut decoded);
        assert_eq!(decoded, input);
    }

    #[test]
    fn bf16_conversion_uses_bfloat16_encoding() {
        let input = [0.0, 1.0, -2.0, 1.5];
        let mut encoded = [Bf16Bits::default(); 4];
        quantize_bf16(&input, &mut encoded);
        assert_eq!(
            bytemuck::cast_slice::<Bf16Bits, u16>(&encoded),
            &[0x0000, 0x3f80, 0xc000, 0x3fc0]
        );
        let mut decoded = [0.0; 4];
        bf16_to_f32(&encoded, &mut decoded);
        assert_eq!(decoded, input);
    }

    #[test]
    #[should_panic(expected = "equal lengths")]
    fn unequal_distance_slices_are_rejected_before_ffi() {
        l2_sq_f32(&[1., 2.], &[1.]);
    }

    #[test]
    #[should_panic(expected = "equal lengths")]
    fn undersized_quantization_output_is_rejected_before_ffi() {
        quantize_bf16(&[1., 2.], &mut [Bf16Bits::default()]);
    }

    #[test]
    #[should_panic(expected = "equal lengths")]
    fn undersized_decode_output_is_rejected_before_ffi() {
        f16_to_f32(&[F16Bits::default(); 2], &mut [0.]);
    }
}
