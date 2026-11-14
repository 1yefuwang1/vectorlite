//! Callback-local receiver for the native `vectorlite.batch.f32.v1` ABI.
//!
//! Native producers bind a pointer to `BatchF32V1` using sqlite3_bind_pointer.
//! They must own a readable, aligned, fully initialized descriptor and readable,
//! aligned native f32/i64 arrays in single allocations of the declared lengths.
//! The descriptor and arrays must remain immutable and alive until the binding
//! is cleared, replaced or finalized (reset alone does not release a binding).
//! A tag match does not authenticate an allocation: runtime checks cannot prove
//! address validity, allocation lengths or freedom from concurrent mutation.
//! Violating this native caller contract can cause undefined behavior. SQL
//! INTEGERs and BLOBs are never reinterpreted as native addresses.
//!
//! The view borrows the protected Value only for this SQLite callback. Owners
//! must copy/normalize bounded chunks before insertion; neither this view nor
//! its borrowed slices may be retained in spawned tasks or after the callback.

#![deny(unsafe_op_in_unsafe_fn)]
#![deny(clippy::undocumented_unsafe_blocks, clippy::missing_safety_doc)]

use std::marker::PhantomData;
use std::mem::{align_of, size_of};
use std::os::raw::c_int;
use std::rc::Rc;

use crate::ffi::{self, Value};
use crate::index_error::IndexError;

const POINTER_TAG: &[u8] = b"vectorlite.batch.f32.v1\0";
const ABI_VERSION: u32 = 1;

/// Exact counterpart of the public C descriptor, with native float32 payloads.
/// The descriptor itself, not either array, is bound using POINTER_TAG.
#[repr(C)]
#[derive(Clone, Copy)]
struct BatchF32V1 {
    abi_version: u32,
    struct_size: u32,
    count: u64,
    dimension: u64,
    vectors: *const f32,
    rowids: *const i64,
}

/// Borrowed input for one callback; intentionally neither Send nor Sync.
/// Its owner must copy bounded chunks instead of moving the view into tasks.
pub(crate) struct BatchView<'a> {
    vectors: &'a [f32],
    rowids: &'a [i64],
    dimension: usize,
    _callback: PhantomData<&'a Value<'a>>,
    _local: PhantomData<Rc<()>>,
}

impl<'a> BatchView<'a> {
    /// Receives only the exact tagged native descriptor, never SQL addresses.
    ///
    /// This safe internal entry point relies on the native binding contract
    /// documented above and in the public header. SQLite guarantees only tag
    /// matching and binding lifetime, NOT validity of an arbitrary bound pointer.
    pub(crate) fn from_value(
        value: &'a Value<'_>,
        expected_dimension: usize,
    ) -> Result<BatchView<'a>, IndexError> {
        // SAFETY: POINTER_TAG is static and NUL-terminated. Value owns a
        // protected callback argument; retrieval does not convert its SQL type.
        // The returned pointer is not dereferenced until address checks below.
        let pointer = unsafe { value.pointer(POINTER_TAG.as_ptr().cast()) }.cast::<BatchF32V1>();
        validate_address(pointer, 1, "descriptor")?;
        // SAFETY: null/alignment/span checks precede this dereference. The native
        // caller additionally promises a readable initialized BatchF32V1; a tag
        // alone cannot prove that promise. The descriptor stays immutable/alive
        // through the binding and hence through this callback's Value borrow.
        let descriptor = unsafe { &*pointer };
        let layout = validate_descriptor(descriptor, expected_dimension)?;
        // SAFETY: all metadata, array alignment and representable address spans
        // are checked before either payload is borrowed. The native caller
        // promises both complete allocations readable/immutable for 'a, which
        // is bounded by the protected Value's callback lifetime.
        Ok(unsafe { borrow_payload(descriptor, layout) })
    }

    pub(crate) fn len(&self) -> usize {
        self.rowids.len()
    }

    pub(crate) fn dimension(&self) -> usize {
        self.dimension
    }

    /// Converts an explicit native signed rowid without wrapping negative IDs.
    pub(crate) fn rowid(&self, index: usize) -> Result<u64, IndexError> {
        let rowid = self
            .rowids
            .get(index)
            .ok_or_else(|| range("batch row index is out of range"))?;
        u64::try_from(*rowid).map_err(|_| range("batch rowid must be nonnegative"))
    }

    /// Borrows one contiguous vector. The caller must supply index < len().
    pub(crate) fn vector(&self, index: usize) -> &[f32] {
        assert!(index < self.len(), "batch vector index is out of range");
        // The complete count * dimension product was checked before borrowing.
        let start = index * self.dimension;
        &self.vectors[start..start + self.dimension]
    }
}

fn misuse(message: impl Into<String>) -> IndexError {
    IndexError::with_code(ffi::SQLITE_MISUSE as c_int, message)
}

fn mismatch(message: impl Into<String>) -> IndexError {
    IndexError::with_code(ffi::SQLITE_MISMATCH as c_int, message)
}

fn range(message: impl Into<String>) -> IndexError {
    IndexError::with_code(ffi::SQLITE_RANGE as c_int, message)
}

fn too_big(message: impl Into<String>) -> IndexError {
    IndexError::with_code(ffi::SQLITE_TOOBIG as c_int, message)
}

#[derive(Clone, Copy, Debug)]
struct Layout {
    count: usize,
    dimension: usize,
    vector_elements: usize,
}

/// Pure metadata validation: it never dereferences either array pointer.
/// Tests can pass a known-valid local descriptor without initializing SQLite.
fn validate_descriptor(
    descriptor: &BatchF32V1,
    expected_dimension: usize,
) -> Result<Layout, IndexError> {
    if descriptor.abi_version != ABI_VERSION {
        return Err(mismatch("unsupported native batch ABI version"));
    }
    if u64::from(descriptor.struct_size) != size_of::<BatchF32V1>() as u64 {
        return Err(mismatch("native batch descriptor size mismatch"));
    }
    if descriptor.dimension == 0 {
        return Err(mismatch("native batch dimension must be greater than zero"));
    }
    let dimension = usize::try_from(descriptor.dimension)
        .map_err(|_| too_big("native batch dimension exceeds usize"))?;
    if dimension != expected_dimension {
        return Err(mismatch(format!(
            "native batch dimension mismatch: expected {expected_dimension}, got {dimension}"
        )));
    }
    if descriptor.count > i64::MAX as u64 {
        return Err(range("native batch count exceeds i64::MAX"));
    }
    let count = usize::try_from(descriptor.count)
        .map_err(|_| too_big("native batch count exceeds usize"))?;
    let vector_elements = count
        .checked_mul(dimension)
        .ok_or_else(|| too_big("native batch count * dimension overflows usize"))?;
    // These checks precede all payload access, including rowid reads. Empty
    // arrays permit null/unaligned pointers because they are never borrowed.
    validate_address(descriptor.vectors, vector_elements, "vectors")?;
    validate_address(descriptor.rowids, count, "rowids")?;
    Ok(Layout {
        count,
        dimension,
        vector_elements,
    })
}

/// Checks representable arithmetic, nullness and alignment, not allocation
/// validity. Merely inspecting a raw address is safe and does not dereference it.
fn validate_address<T>(pointer: *const T, elements: usize, name: &str) -> Result<(), IndexError> {
    if elements == 0 {
        return Ok(());
    }
    let bytes = elements
        .checked_mul(size_of::<T>())
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or_else(|| too_big(format!("native batch {name} byte span exceeds isize::MAX")))?;
    if pointer.is_null() {
        return Err(misuse(format!("native batch {name} pointer is null")));
    }
    if !pointer.addr().is_multiple_of(align_of::<T>()) {
        return Err(misuse(format!("native batch {name} pointer is misaligned")));
    }
    if pointer.addr().checked_add(bytes).is_none() {
        return Err(too_big(format!(
            "native batch {name} address span wraps usize"
        )));
    }
    Ok(())
}

/// Borrows payloads only after both complete spans have been validated.
///
/// # Safety
/// `layout` must come from validate_descriptor on this unchanged descriptor.
/// Each nonempty array must additionally occupy one initialized, readable,
/// aligned allocation of the declared length, immutable and alive for 'a. The
/// caller must bound 'a by the callback Value borrow (or test buffer borrows).
unsafe fn borrow_payload<'a>(descriptor: &BatchF32V1, layout: Layout) -> BatchView<'a> {
    let (vectors, rowids) = if layout.count == 0 {
        // Rust still requires aligned nonnull pointers for empty from_raw_parts;
        // use safe empty slices instead of either caller-provided pointer.
        (&[][..], &[][..])
    } else {
        // SAFETY: the caller guarantees allocation validity and immutable
        // lifetime 'a; validation checked both lengths, alignment and spans.
        unsafe {
            (
                std::slice::from_raw_parts(descriptor.vectors, layout.vector_elements),
                std::slice::from_raw_parts(descriptor.rowids, layout.count),
            )
        }
    };
    BatchView {
        vectors,
        rowids,
        dimension: layout.dimension,
        _callback: PhantomData,
        _local: PhantomData,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::offset_of;

    fn descriptor(count: u64, dimension: u64) -> BatchF32V1 {
        BatchF32V1 {
            abi_version: ABI_VERSION,
            struct_size: size_of::<BatchF32V1>() as u32,
            count,
            dimension,
            // Deliberately non-dereferenceable but aligned. Metadata tests
            // prove rejection without ever constructing payload slices.
            vectors: std::ptr::without_provenance(align_of::<f32>()),
            rowids: std::ptr::without_provenance(align_of::<i64>()),
        }
    }

    /// Safe local-buffer parser for payload tests: derives exact pointers and
    /// lengths from immutable slices before calling the private unsafe borrower.
    fn local_view<'a>(vectors: &'a [f32], rowids: &'a [i64], dimension: usize) -> BatchView<'a> {
        assert_eq!(vectors.len(), rowids.len() * dimension);
        let descriptor = BatchF32V1 {
            vectors: vectors.as_ptr(),
            rowids: rowids.as_ptr(),
            ..descriptor(rowids.len() as u64, dimension as u64)
        };
        let layout = validate_descriptor(&descriptor, dimension).unwrap();
        // SAFETY: these pointers and exact lengths were derived from the two
        // immutable live slices, so allocations and initialization are proven
        // locally. Returned 'a is bounded by those buffer borrows, not by the
        // temporary descriptor, whose fields the view copies/borrows through.
        unsafe { borrow_payload(&descriptor, layout) }
    }

    fn error_code(descriptor: &BatchF32V1, expected: usize) -> c_int {
        validate_descriptor(descriptor, expected).unwrap_err().code
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn repr_c_layout_matches_public_64_bit_header() {
        assert_eq!(size_of::<BatchF32V1>(), 40);
        assert_eq!(align_of::<BatchF32V1>(), 8);
        assert_eq!(offset_of!(BatchF32V1, abi_version), 0);
        assert_eq!(offset_of!(BatchF32V1, struct_size), 4);
        assert_eq!(offset_of!(BatchF32V1, count), 8);
        assert_eq!(offset_of!(BatchF32V1, dimension), 16);
        assert_eq!(offset_of!(BatchF32V1, vectors), 24);
        assert_eq!(offset_of!(BatchF32V1, rowids), 32);
        let header = include_str!("../include/vectorlite_batch.h");
        assert!(header.contains("#define VECTORLITE_BATCH_F32_V1_STRUCT_SIZE 40u"));
        assert!(header.contains("#define VECTORLITE_BATCH_F32_V1_ABI_VERSION 1u"));
        assert!(header.contains("#define VECTORLITE_BATCH_F32_V1_TAG \"vectorlite.batch.f32.v1\""));
        assert_eq!(POINTER_TAG, b"vectorlite.batch.f32.v1\0");
    }

    #[test]
    fn descriptor_null_and_misalignment_are_rejected_without_dereference() {
        assert_eq!(
            validate_address(std::ptr::null::<BatchF32V1>(), 1, "descriptor")
                .unwrap_err()
                .code,
            ffi::SQLITE_MISUSE as c_int
        );
        assert_eq!(
            validate_address(
                std::ptr::without_provenance::<BatchF32V1>(1),
                1,
                "descriptor"
            )
            .unwrap_err()
            .code,
            ffi::SQLITE_MISUSE as c_int
        );
    }

    #[test]
    fn incompatible_abi_and_size_are_rejected_before_payload() {
        let mut bad = descriptor(1, 3);
        bad.vectors = std::ptr::null();
        bad.rowids = std::ptr::null();
        bad.abi_version = ABI_VERSION + 1;
        assert_eq!(error_code(&bad, 3), ffi::SQLITE_MISMATCH as c_int);
        bad.abi_version = ABI_VERSION;
        for size in [0, 8, 39, 41, u32::MAX] {
            bad.struct_size = size;
            assert_eq!(error_code(&bad, 3), ffi::SQLITE_MISMATCH as c_int);
        }
    }

    #[test]
    fn dimensions_are_validated_even_for_empty_batches() {
        for count in [0, 1] {
            let mut bad = descriptor(count, 0);
            bad.vectors = std::ptr::null();
            bad.rowids = std::ptr::null();
            assert_eq!(error_code(&bad, 3), ffi::SQLITE_MISMATCH as c_int);
            bad.dimension = 4;
            assert_eq!(error_code(&bad, 3), ffi::SQLITE_MISMATCH as c_int);
        }
    }

    #[test]
    fn count_must_fit_public_signed_64_bit_domain() {
        let bad = descriptor(i64::MAX as u64 + 1, 1);
        assert_eq!(error_code(&bad, 1), ffi::SQLITE_RANGE as c_int);
        let bad = descriptor(u64::MAX, 1);
        assert_eq!(error_code(&bad, 1), ffi::SQLITE_RANGE as c_int);
    }

    #[test]
    fn element_product_and_both_byte_spans_are_checked_before_payload() {
        let bad = descriptor(2, usize::MAX as u64);
        assert_eq!(error_code(&bad, usize::MAX), ffi::SQLITE_TOOBIG as c_int);
        let bad = descriptor((isize::MAX as u64 / 4) + 1, 1);
        assert_eq!(error_code(&bad, 1), ffi::SQLITE_TOOBIG as c_int);
        // The f32 span fits, but the explicit i64 rowid span does not.
        let bad = descriptor((isize::MAX as u64 / 8) + 1, 1);
        assert_eq!(error_code(&bad, 1), ffi::SQLITE_TOOBIG as c_int);
        let bad = descriptor(1, usize::MAX as u64);
        assert_eq!(error_code(&bad, usize::MAX), ffi::SQLITE_TOOBIG as c_int);
    }

    #[test]
    fn nonempty_buffer_nullness_and_alignment_are_checked_before_payload() {
        let mut bad = descriptor(1, 3);
        bad.vectors = std::ptr::null();
        assert_eq!(error_code(&bad, 3), ffi::SQLITE_MISUSE as c_int);
        bad.vectors = std::ptr::without_provenance(1);
        assert_eq!(error_code(&bad, 3), ffi::SQLITE_MISUSE as c_int);
        bad.vectors = std::ptr::without_provenance(align_of::<f32>());
        bad.rowids = std::ptr::null();
        assert_eq!(error_code(&bad, 3), ffi::SQLITE_MISUSE as c_int);
        bad.rowids = std::ptr::without_provenance(1);
        assert_eq!(error_code(&bad, 3), ffi::SQLITE_MISUSE as c_int);
    }

    #[test]
    fn numeric_address_wrap_is_rejected_before_payload() {
        let mut bad = descriptor(1, 1);
        bad.vectors = std::ptr::without_provenance(usize::MAX - 3);
        assert_eq!(error_code(&bad, 1), ffi::SQLITE_TOOBIG as c_int);
        bad.vectors = std::ptr::without_provenance(align_of::<f32>());
        bad.rowids = std::ptr::without_provenance(usize::MAX - 7);
        assert_eq!(error_code(&bad, 1), ffi::SQLITE_TOOBIG as c_int);
        assert_eq!(
            validate_address(
                std::ptr::without_provenance::<BatchF32V1>(usize::MAX - 7),
                1,
                "descriptor"
            )
            .unwrap_err()
            .code,
            ffi::SQLITE_TOOBIG as c_int
        );
    }

    #[test]
    fn contiguous_vectors_and_explicit_rowids_are_borrowed_without_copying() {
        let vectors = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let rowids = [0, i64::MAX];
        let view = local_view(&vectors, &rowids, 3);
        assert_eq!(view.len(), 2);
        assert_eq!(view.dimension(), 3);
        assert_eq!(view.vector(0), &vectors[..3]);
        assert_eq!(view.vector(1), &vectors[3..]);
        assert_eq!(view.vector(0).as_ptr(), vectors.as_ptr());
        assert_eq!(view.vector(1).as_ptr(), vectors[3..].as_ptr());
        assert_eq!(view.rowid(0).unwrap(), 0);
        assert_eq!(view.rowid(1).unwrap(), i64::MAX as u64);
    }

    #[test]
    fn negative_rowids_and_out_of_range_indices_do_not_wrap() {
        let vectors = [1.0, 2.0, 3.0];
        let rowids = [-1, i64::MIN, 42];
        let view = local_view(&vectors, &rowids, 1);
        assert_eq!(view.rowid(0).unwrap_err().code, ffi::SQLITE_RANGE as c_int);
        assert_eq!(view.rowid(1).unwrap_err().code, ffi::SQLITE_RANGE as c_int);
        assert_eq!(view.rowid(2).unwrap(), 42);
        assert_eq!(view.rowid(3).unwrap_err().code, ffi::SQLITE_RANGE as c_int);
        assert_eq!(
            view.rowid(usize::MAX).unwrap_err().code,
            ffi::SQLITE_RANGE as c_int
        );
    }

    #[test]
    fn empty_batches_allow_null_or_unaligned_arrays_and_keep_dimension() {
        let mut empty = descriptor(0, 3);
        empty.vectors = std::ptr::null();
        empty.rowids = std::ptr::null();
        let layout = validate_descriptor(&empty, 3).unwrap();
        // SAFETY: a validated zero count borrows neither raw array and uses
        // safe empty slices, so no pointer lifetime or allocation is needed.
        let view = unsafe { borrow_payload(&empty, layout) };
        assert_eq!(view.len(), 0);
        assert_eq!(view.dimension(), 3);
        assert_eq!(view.rowid(0).unwrap_err().code, ffi::SQLITE_RANGE as c_int);
        empty.vectors = std::ptr::without_provenance(1);
        empty.rowids = std::ptr::without_provenance(1);
        assert_eq!(validate_descriptor(&empty, 3).unwrap().count, 0);
    }
}
