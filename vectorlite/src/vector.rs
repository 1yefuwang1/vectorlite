//! Conversions between SQLite blobs / JSON and f32 vectors used by the scalar
//! functions and virtual table.

use std::borrow::Cow;

/// Borrows native aligned float32 bytes, decoding otherwise.
pub fn view_from_blob(blob: &[u8]) -> Result<Cow<'_, [f32]>, String> {
    if blob.is_empty() {
        return Ok(Cow::Borrowed(&[]));
    }
    if cfg!(target_endian = "little") {
        // bytemuck checks the actual pointer alignment and byte length. f32 is
        // plain data with no invalid bit patterns; no unsafe cast is needed.
        if let Ok(values) = bytemuck::try_cast_slice(blob) {
            return Ok(Cow::Borrowed(values));
        }
    }
    blob_to_f32(blob).map(Cow::Owned)
}

/// Parses a little-endian f32 blob. Errors if the length is not a multiple of 4.
pub fn blob_to_f32(blob: &[u8]) -> Result<Vec<f32>, String> {
    if !blob.len().is_multiple_of(std::mem::size_of::<f32>()) {
        return Err("Blob size is not a multiple of float".to_string());
    }
    let mut out = Vec::with_capacity(blob.len() / 4);
    for chunk in blob.as_chunks::<4>().0 {
        out.push(f32::from_le_bytes(*chunk));
    }
    Ok(out)
}

/// Borrows little-endian native float data for SQLite's copying result API.
/// Big-endian hosts serialize explicitly into the wire format.
pub fn blob_from_f32(values: &[f32]) -> Cow<'_, [u8]> {
    if cfg!(target_endian = "little") {
        Cow::Borrowed(bytemuck::cast_slice(values))
    } else {
        Cow::Owned(f32_to_blob(values))
    }
}

/// Serialises an f32 vector to a little-endian blob.
pub fn f32_to_blob(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Parses a JSON array of numbers into an f32 vector.
pub fn from_json(json: &str) -> Result<Vec<f32>, String> {
    let values: Vec<f64> = serde_json::from_str(json).map_err(|e| e.to_string())?;
    Ok(values.into_iter().map(|number| number as f32).collect())
}

/// Serialises an f32 vector to a JSON array. Values are widened to f64 so the
/// output has the conventional `1.0` formatting and round-trips exactly. Returns
/// an error if any value is non-finite (NaN/Infinity), which JSON cannot
/// represent — silently emitting `null` there would produce a blob that no
/// longer round-trips through `vector_from_json`.
pub fn to_json(v: &[f32]) -> Result<String, String> {
    if let Some(value) = v.iter().find(|value| !value.is_finite()) {
        return Err(format!(
            "vector contains a non-finite value ({value}) that cannot be represented in JSON"
        ));
    }
    let numbers: Vec<f64> = v.iter().map(|&value| value as f64).collect();
    serde_json::to_string(&numbers).map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_little_endian_view_borrows_input() {
        let values = [1.0f32, -2.5, 3.25];
        let bytes: &[u8] = bytemuck::cast_slice(&values);
        let view = view_from_blob(bytes).unwrap();
        if cfg!(target_endian = "little") {
            assert!(matches!(view, Cow::Borrowed(_)));
            assert_eq!(view.as_ptr(), values.as_ptr());
        }
    }

    #[test]
    fn unaligned_view_decodes_little_endian_values() {
        let mut storage = [0u32; 3];
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut storage);
        bytes[1..5].copy_from_slice(&1.25f32.to_le_bytes());
        bytes[5..9].copy_from_slice(&(-2.5f32).to_le_bytes());
        let view = view_from_blob(&bytes[1..9]).unwrap();
        assert!(matches!(view, Cow::Owned(_)));
        assert_eq!(&*view, &[1.25, -2.5]);
    }

    #[test]
    fn view_rejects_partial_float_and_handles_empty() {
        assert!(view_from_blob(&[0; 3]).is_err());
        assert!(view_from_blob(&[]).unwrap().is_empty());
    }

    #[test]
    fn blob_roundtrip_preserves_values() {
        let v = vec![1.0f32, -2.5, 3.25, 0.0];
        let blob = f32_to_blob(&v);
        assert_eq!(blob.len(), 16);
        assert_eq!(blob_to_f32(&blob).unwrap(), v);
    }

    #[test]
    fn blob_to_f32_rejects_non_multiple_of_four() {
        assert!(blob_to_f32(&[0u8, 1, 2]).is_err());
        assert!(blob_to_f32(&[0u8; 5]).is_err());
    }

    #[test]
    fn blob_to_f32_accepts_empty() {
        assert_eq!(blob_to_f32(&[]).unwrap(), Vec::<f32>::new());
    }

    #[test]
    fn from_json_parses_number_array() {
        assert_eq!(from_json("[1, 2.5, -3]").unwrap(), vec![1.0f32, 2.5, -3.0]);
    }

    #[test]
    fn from_json_rejects_non_array_and_non_numeric() {
        assert!(from_json("not json").is_err());
        assert!(from_json("{}").is_err());
        assert!(from_json("[1, \"x\"]").is_err());
    }

    #[test]
    fn to_json_formats_as_float_array() {
        assert_eq!(to_json(&[1.0, 2.0, 3.0]).unwrap(), "[1.0,2.0,3.0]");
    }

    #[test]
    fn to_json_rejects_non_finite() {
        assert!(to_json(&[1.0, f32::NAN]).is_err());
        assert!(to_json(&[f32::INFINITY]).is_err());
        assert!(to_json(&[f32::NEG_INFINITY]).is_err());
    }

    #[test]
    fn json_round_trip() {
        let v = vec![0.5f32, -1.25, 100.0];
        let json = to_json(&v).unwrap();
        assert_eq!(from_json(&json).unwrap(), v);
    }
}
