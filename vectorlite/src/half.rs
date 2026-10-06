//! Opaque bit representations of the two supported 16-bit vector formats.
//! Numerical operations live in `ops`; these types carry no integer arithmetic.

/// IEEE 754 binary16 storage, with the same layout and alignment as `u16`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(transparent)]
pub struct F16Bits(u16);

/// BFloat16 storage, distinct from binary16 despite sharing its two-byte width.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(transparent)]
pub struct Bf16Bits(u16);

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    #[test]
    fn half_storage_preserves_native_u16_layout() {
        assert_eq!(size_of::<F16Bits>(), size_of::<u16>());
        assert_eq!(align_of::<F16Bits>(), align_of::<u16>());
        assert_eq!(size_of::<Bf16Bits>(), size_of::<u16>());
        assert_eq!(align_of::<Bf16Bits>(), align_of::<u16>());
        assert_eq!(bytemuck::bytes_of(&F16Bits::default()), &[0, 0]);
        assert_eq!(bytemuck::bytes_of(&Bf16Bits::default()), &[0, 0]);
    }
}
