//! Conversions between SQLite blobs / JSON and f32 vectors. Mirrors the float
//! specialisation of `vector.h` / `vector_view.h` used by the scalar functions.

/// Parses a little-endian f32 blob. Errors if the length is not a multiple of 4.
pub fn blob_to_f32(blob: &[u8]) -> Result<Vec<f32>, String> {
    if !blob.len().is_multiple_of(std::mem::size_of::<f32>()) {
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
/// output has the conventional `1.0` formatting and round-trips exactly.
pub fn to_json(v: &[f32]) -> String {
    let nums: Vec<serde_json::Value> = v
        .iter()
        .map(|&x| {
            serde_json::Number::from_f64(x as f64)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null)
        })
        .collect();
    serde_json::Value::Array(nums).to_string()
}
