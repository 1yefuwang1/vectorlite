//! Conversions between SQLite blobs / JSON and f32 vectors. Mirrors the float
//! specialisation of `vector.h` / `vector_view.h` used by the scalar functions.

/// Parses a little-endian f32 blob. Errors if the length is not a multiple of 4.
pub fn blob_to_f32(blob: &[u8]) -> Result<Vec<f32>, String> {
    if blob.len() % std::mem::size_of::<f32>() != 0 {
        return Err("Blob size is not a multiple of float".to_string());
    }
    let mut out = Vec::with_capacity(blob.len() / 4);
    for chunk in blob.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
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
    let value: serde_json::Value = serde_json::from_str(json).map_err(|e| e.to_string())?;
    let arr = value
        .as_array()
        .ok_or_else(|| "Input JSON is not an array.".to_string())?;
    let mut out = Vec::with_capacity(arr.len());
    for v in arr {
        let num = v
            .as_f64()
            .ok_or_else(|| "JSON array contains non-numeric value.".to_string())?;
        out.push(num as f32);
    }
    Ok(out)
}

/// Serialises an f32 vector to a JSON array. Values are widened to f64 so the
/// output has the conventional `1.0` formatting and round-trips exactly. Returns
/// an error if any value is non-finite (NaN/Infinity), which JSON cannot
/// represent — silently emitting `null` there would produce a blob that no
/// longer round-trips through `vector_from_json`.
pub fn to_json(v: &[f32]) -> Result<String, String> {
    let mut nums: Vec<serde_json::Value> = Vec::with_capacity(v.len());
    for &x in v {
        match serde_json::Number::from_f64(x as f64) {
            Some(n) => nums.push(serde_json::Value::Number(n)),
            None => {
                return Err(format!(
                    "vector contains a non-finite value ({x}) that cannot be represented in JSON"
                ))
            }
        }
    }
    Ok(serde_json::Value::Array(nums).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

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
