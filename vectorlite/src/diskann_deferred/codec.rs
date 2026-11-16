//! Checked Format4 little-endian payload and numeric policies.
use super::*;
pub(crate) fn corrupt(message: impl Into<String>) -> IndexError {
    IndexError::with_code(ffi::SQLITE_CORRUPT as i32, message)
}

pub(crate) fn too_big(message: impl Into<String>) -> IndexError {
    IndexError::with_code(ffi::SQLITE_TOOBIG as i32, message)
}

pub(crate) fn no_memory() -> IndexError {
    IndexError::with_code(
        ffi::SQLITE_NOMEM as i32,
        "cannot allocate DiskANN workspace",
    )
}

pub(crate) fn reserve<T>(values: &mut Vec<T>, count: usize) -> Result<()> {
    values.try_reserve_exact(count).map_err(|_| no_memory())
}

pub(crate) fn checked_id(id: u64) -> Result<i64> {
    i64::try_from(id).map_err(|_| {
        IndexError::with_code(
            ffi::SQLITE_RANGE as i32,
            "DiskANN row/node ID exceeds i64::MAX",
        )
    })
}

pub(crate) fn nonnegative(statement: &Statement<'_>, column: usize, name: &str) -> Result<u64> {
    // Type errors in an on-disk format are corruption, not user coercions.
    if statement.column_type(column)? != ffi::SQLITE_INTEGER as i32 {
        return Err(corrupt(format!("DiskANN {name} is not an INTEGER")));
    }
    u64::try_from(statement.column_i64(column)?)
        .map_err(|_| corrupt(format!("DiskANN {name} is negative")))
}

pub(crate) fn checked_increment(value: u64, name: &str) -> Result<u64> {
    value
        .checked_add(1)
        .filter(|&value| value <= MAX_ID)
        .ok_or_else(|| too_big(format!("DiskANN {name} exhausted")))
}

pub(crate) fn checked_product(a: usize, b: usize, name: &str) -> Result<usize> {
    a.checked_mul(b)
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or_else(|| too_big(format!("DiskANN {name} size overflow")))
}

pub(crate) fn blob_projection(column: &str, maximum: usize, exact: bool) -> String {
    // Identifiers here are fixed implementation column names, never user input.
    // SQLite's length(BLOB)/typeof opcodes read only record header metadata;
    // CASE prevents its payload Column opcode from copying oversized corrupt
    // BLOBs or TEXT before the Rust wrapper can inspect the length field.
    let comparison = if exact { "=" } else { "<=" };
    format!(
        "CASE WHEN typeof({column})='blob' THEN length({column}) ELSE -1 END,CASE WHEN typeof({column})='blob' AND length({column}){comparison}{maximum} THEN {column} ELSE NULL END"
    )
}

pub(crate) fn bounded_blob(
    statement: &Statement<'_>,
    length_column: usize,
    blob_column: usize,
    maximum: usize,
    exact: bool,
    name: &str,
) -> Result<Vec<u8>> {
    let length = nonnegative(statement, length_column, name)?;
    if length > maximum as u64 || (exact && length != maximum as u64) {
        return Err(corrupt(format!(
            "DiskANN {name} has an invalid byte length"
        )));
    }
    if statement.column_type(blob_column)? != ffi::SQLITE_BLOB as i32 {
        return Err(corrupt(format!("DiskANN {name} is not a BLOB")));
    }
    // SQL length() is checked BEFORE the wrapper copies any BLOB bytes.
    let bytes = statement.column_blob_owned(blob_column)?;
    if bytes.len() as u64 != length {
        return Err(corrupt(format!("DiskANN {name} byte length changed")));
    }
    Ok(bytes)
}

pub(crate) fn encode_ids(ids: &[u64], maximum: usize, owner: Option<u64>) -> Result<Vec<u8>> {
    validate_ids(ids, maximum, owner)?;
    let bytes = checked_product(ids.len(), 8, "adjacency")?;
    let mut output = Vec::new();
    reserve(&mut output, bytes)?;
    for &id in ids {
        output.extend_from_slice(&id.to_le_bytes());
    }
    Ok(output)
}

pub(crate) fn validate_ids(ids: &[u64], maximum: usize, owner: Option<u64>) -> Result<()> {
    if ids.len() > maximum {
        return Err(corrupt("DiskANN adjacency exceeds physical degree"));
    }
    // Degree is bounded by the options; no graph-sized duplicate set is needed.
    for (index, &id) in ids.iter().enumerate() {
        if id > MAX_ID || owner == Some(id) || ids[..index].contains(&id) {
            return Err(corrupt(
                "DiskANN adjacency contains an invalid, duplicate, or self ID",
            ));
        }
    }
    Ok(())
}

pub(crate) fn decode_ids(bytes: &[u8], maximum: usize, owner: Option<u64>) -> Result<Vec<u64>> {
    if !bytes.len().is_multiple_of(8) || bytes.len() / 8 > maximum {
        return Err(corrupt("DiskANN adjacency has an invalid byte length"));
    }
    let mut output = Vec::new();
    reserve(&mut output, bytes.len() / 8)?;
    for chunk in bytes.as_chunks::<8>().0 {
        output.push(u64::from_le_bytes(*chunk));
    }
    validate_ids(&output, maximum, owner)?;
    Ok(output)
}

pub(crate) fn encode_vector(vector: &[f32]) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    reserve(&mut bytes, checked_product(vector.len(), 4, "vector")?)?;
    for &value in vector {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(bytes)
}

#[cfg(test)]
pub(crate) fn decode_vector(bytes: &[u8], dimension: usize) -> Result<Vec<f32>> {
    let mut vector = Vec::new();
    decode_vector_into(bytes, dimension, &mut vector)?;
    Ok(vector)
}

pub(crate) fn decode_vector_into(
    bytes: &[u8],
    dimension: usize,
    vector: &mut Vec<f32>,
) -> Result<()> {
    if bytes.len() != checked_product(dimension, 4, "vector")? {
        return Err(corrupt("DiskANN vector has an invalid byte length"));
    }
    vector.clear();
    reserve(vector, dimension)?;
    for chunk in bytes.as_chunks::<4>().0 {
        let value = f32::from_le_bytes(*chunk);
        if !value.is_finite() {
            return Err(corrupt("DiskANN stored vector contains a non-finite value"));
        }
        vector.push(value);
    }
    Ok(())
}

pub(crate) fn with_decoded_vector(
    bytes: &[u8],
    dimension: usize,
    decode_buffer: &mut Vec<f32>,
    action: &mut dyn FnMut(StoredVector<'_>) -> Result<()>,
) -> Result<()> {
    // Preserve the checked dimension arithmetic even on a borrowed projection.
    checked_product(dimension, 4, "vector")?;
    if cfg!(target_endian = "little") {
        if let Ok(vector) = bytemuck::try_cast_slice::<u8, f32>(bytes) {
            // The sole authoritative length/finite check for this exact borrowed
            // slice. Core still matches its own GraphConfig dimension, but does
            // not rescan the proof before scoring or copying.
            let vector = ValidatedVector::new(vector, dimension).map_err(|error| match error {
                VectorValidationError::Dimension => {
                    corrupt("DiskANN vector has an invalid byte length")
                }
                VectorValidationError::NonFinite => {
                    corrupt("DiskANN stored vector contains a non-finite value")
                }
            })?;
            return action(StoredVector::Validated(vector));
        }
    }
    // Misaligned/big-endian values keep the checked LE decoder and its errors.
    // It does not manufacture a proof; core may recheck this uncommon copy path.
    decode_vector_into(bytes, dimension, decode_buffer)?;
    action(StoredVector::Unvalidated(decode_buffer))
}

pub(crate) fn validate_numeric_range(vector: &[f32], metric: DistanceType) -> Result<f64> {
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(IndexError::new(
            "DiskANN vector contains a non-finite value",
        ));
    }
    let squared_norm: f64 = vector.iter().map(|&value| f64::from(value).powi(2)).sum();
    let (maximum, message) = match metric {
        // The triangle inequality gives ||a-b||² <= 4 * max(||a||²,||b||²).
        // This DiskANN-only restriction makes *all* accepted L2 pairs safe before
        // graph writes, rather than discovering overflow after allocation.
        DistanceType::L2 => (
            f64::from(f32::MAX) / 4.0,
            "DiskANN squared L2 requires squared norm <= f32::MAX/4 to avoid distance overflow",
        ),
        DistanceType::Cosine => (
            f64::from(f32::MAX),
            "DiskANN cosine norm would overflow float32",
        ),
        DistanceType::InnerProduct => {
            return Err(IndexError::new("DiskANN does not support inner product"))
        }
    };
    if !squared_norm.is_finite() || squared_norm > maximum {
        return Err(IndexError::new(message));
    }
    Ok(squared_norm)
}

pub(crate) fn normalize_input(
    vector: &mut [f32],
    dimension: usize,
    metric: DistanceType,
) -> Result<()> {
    if vector.len() != dimension {
        return Err(IndexError::new(format!(
            "dimension mismatch: expected {dimension}, got {}",
            vector.len()
        )));
    }
    let squared_norm = validate_numeric_range(vector, metric)?;
    if metric == DistanceType::Cosine {
        if !ops::ip_dist_f32(vector, vector).is_finite() {
            return Err(IndexError::new(
                "DiskANN cosine norm would overflow float32",
            ));
        }
        ops::normalize_f32(vector);
        if vector.iter().any(|value| !value.is_finite())
            || (squared_norm != 0.0 && vector.iter().all(|&value| value == 0.0))
        {
            return Err(IndexError::new(
                "DiskANN cosine normalization overflowed float32",
            ));
        }
    }
    Ok(())
}

pub(crate) fn validate_journal_mode(mode: &str) -> Result<()> {
    match mode {
        "delete" | "truncate" | "persist" | "memory" | "wal" => Ok(()),
        "off" => Err(IndexError::new(
            "DiskANN mutations require rollback journaling; journal_mode=OFF is not supported",
        )),
        _ => Err(corrupt(
            "DiskANN encountered an unsupported SQLite journal_mode",
        )),
    }
}

pub(crate) fn descriptor(space: &NamedVectorSpace, options: &DiskAnnOptions) -> serde_json::Value {
    serde_json::json!({
        "format_version": FORMAT_VERSION,
        "backend": "diskann",
        "dimension": space.dim,
        "vector_type": "float32",
        "metric": if space.distance_type == DistanceType::Cosine { "cosine" } else { "l2_squared" },
        "normalize": space.distance_type == DistanceType::Cosine,
        "adapter_version": "0.60",
        "atomic_write_protocol": "guard-two-row-v1",
        "deletion_policy": "immutable-versions-v1",
        "maintenance": "deferred-oneshot-v1",
        "options": {
            "degree": options.degree,
            "build_list_size": options.build_list_size,
            "search_list_size": options.search_list_size,
            "alpha": options.alpha,
            "cache_bytes": options.cache_bytes,
            "max_visits": options.max_visits,
        }
    })
}

pub(crate) fn validate_descriptor(bytes: &[u8], expected: &serde_json::Value) -> Result<()> {
    if bytes.len() > MAX_DESCRIPTOR_BYTES {
        return Err(corrupt("DiskANN descriptor is too large"));
    }
    let actual: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|_| corrupt("DiskANN descriptor is invalid JSON"))?;
    if &actual != expected {
        return Err(corrupt(
            "DiskANN descriptor does not match the virtual-table declaration",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn space(metric: DistanceType) -> NamedVectorSpace {
        NamedVectorSpace {
            vector_name: "embedding".into(),
            dim: 3,
            distance_type: metric,
            vector_type: VectorType::Float32,
        }
    }

    #[test]
    fn payload_projection_rejects_wrong_types_and_bounds_before_copying() {
        let vector = blob_projection("vector", 12, true);
        assert!(vector.contains("THEN length(vector) ELSE -1 END"));
        assert!(vector.contains("typeof(vector)='blob' AND length(vector)=12"));
        assert!(vector.ends_with("THEN vector ELSE NULL END"));
        let neighbors = blob_projection("neighbors", 328, false);
        assert!(neighbors.contains("length(neighbors)<=328"));
        assert!(neighbors.ends_with("THEN neighbors ELSE NULL END"));
    }

    #[test]
    fn vector_codec_is_little_endian_and_finite() {
        let vector = [1.25, -2.5, 0.0];
        let bytes = encode_vector(&vector).unwrap();
        assert_eq!(&bytes[..4], &1.25f32.to_le_bytes());
        assert_eq!(decode_vector(&bytes, 3).unwrap(), vector);
        assert!(decode_vector(&bytes[..11], 3).is_err());
        assert!(decode_vector(&f32::NAN.to_le_bytes(), 1).is_err());
        assert!(decode_vector(&f32::INFINITY.to_le_bytes(), 1).is_err());
    }

    #[test]
    fn borrowed_vector_decoder_lends_a_checked_proof_for_the_exact_slice() {
        if !cfg!(target_endian = "little") {
            return;
        }
        let values = [1.25f32, -2.5, -0.0];
        let bytes = bytemuck::cast_slice::<f32, u8>(&values);
        let mut buffer = Vec::new();
        let mut calls = 0;
        with_decoded_vector(bytes, 3, &mut buffer, &mut |vector| {
            calls += 1;
            assert!(matches!(vector, StoredVector::Validated(_)));
            let slice = vector.validate(3).unwrap();
            assert_eq!(slice.as_ptr(), values.as_ptr());
            assert_eq!(slice[2].to_bits(), (-0.0f32).to_bits());
            Ok(())
        })
        .unwrap();
        assert_eq!(calls, 1);
        assert_eq!(
            buffer.capacity(),
            0,
            "borrowed proof must not allocate a decode buffer"
        );
        let error = with_decoded_vector(bytes, 2, &mut buffer, &mut |_| {
            panic!("wrong-length value must not reach the callback");
        })
        .unwrap_err();
        assert_eq!(error.code, ffi::SQLITE_CORRUPT as i32);
        assert!(error.message.contains("invalid byte length"));
    }

    #[test]
    fn decoded_vector_copy_fallback_preserves_le_values_and_callback_errors() {
        let values = [1.25, -2.5, -0.0];
        let encoded = encode_vector(&values).unwrap();
        // u32 storage guarantees a four-byte-aligned base; offset one forces
        // the checked copy fallback even on a little-endian test host.
        let mut storage = [0u32; 4];
        let bytes = bytemuck::cast_slice_mut::<u32, u8>(&mut storage);
        bytes[1..13].copy_from_slice(&encoded);
        let mut buffer = Vec::new();
        with_decoded_vector(&bytes[1..13], 3, &mut buffer, &mut |vector| {
            assert!(matches!(vector, StoredVector::Unvalidated(_)));
            let slice = vector.validate(3).unwrap();
            assert_eq!(slice, values);
            assert_eq!(slice[2].to_bits(), (-0.0f32).to_bits());
            Ok(())
        })
        .unwrap();
        assert_eq!(buffer.len(), 3);
        let callback_error = IndexError::with_code(779, "original callback error");
        for offset in [0, 1] {
            bytes[offset..offset + 12].copy_from_slice(&encoded);
            let error =
                with_decoded_vector(&bytes[offset..offset + 12], 3, &mut buffer, &mut |_| {
                    Err(callback_error.clone())
                })
                .unwrap_err();
            assert_eq!(error, callback_error);
        }
    }

    #[test]
    fn decoded_vector_rejects_nonfinite_and_bad_lengths_before_callbacks() {
        let mut storage = [0u32; 3];
        let bytes = bytemuck::cast_slice_mut::<u32, u8>(&mut storage);
        let mut buffer = Vec::new();
        for offset in [0, 1] {
            for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let encoded = encode_vector(&[0.0, value]).unwrap();
                bytes[offset..offset + 8].copy_from_slice(&encoded);
                let error =
                    with_decoded_vector(&bytes[offset..offset + 8], 2, &mut buffer, &mut |_| {
                        panic!("non-finite value must not reach the callback")
                    })
                    .unwrap_err();
                assert_eq!(error.code, ffi::SQLITE_CORRUPT as i32);
                assert!(error
                    .message
                    .contains("stored vector contains a non-finite"));
            }
            for length in [0, 1, 4, 7, 9] {
                let error = with_decoded_vector(
                    &bytes[offset..offset + length],
                    2,
                    &mut buffer,
                    &mut |_| panic!("invalid length must not reach the callback"),
                )
                .unwrap_err();
                assert_eq!(error.code, ffi::SQLITE_CORRUPT as i32);
                assert!(error.message.contains("invalid byte length"));
            }
        }
    }

    #[test]
    fn adjacency_codec_checks_degree_duplicates_self_and_signed_range() {
        let bytes = encode_ids(&[0, 17, MAX_ID], 3, Some(2)).unwrap();
        assert_eq!(decode_ids(&bytes, 3, Some(2)).unwrap(), [0, 17, MAX_ID]);
        assert!(decode_ids(&bytes, 2, None).is_err());
        assert!(decode_ids(&bytes[..23], 3, None).is_err());
        assert!(encode_ids(&[1, 1], 3, None).is_err());
        assert!(encode_ids(&[3], 3, Some(3)).is_err());
        assert!(encode_ids(&[MAX_ID + 1], 3, None).is_err());
        assert!(decode_ids(&encode_ids(&[0, 1], 3, None).unwrap(), 3, Some(1)).is_err());
    }

    #[test]
    fn descriptor_is_owned_versioned_and_options_sensitive() {
        let options = DiskAnnOptions::default();
        let expected = descriptor(&space(DistanceType::Cosine), &options);
        let bytes = serde_json::to_vec(&expected).unwrap();
        validate_descriptor(&bytes, &expected).unwrap();
        assert!(
            validate_descriptor(&bytes, &descriptor(&space(DistanceType::L2), &options)).is_err()
        );
        let mut changed = options.clone();
        changed.cache_bytes *= 2;
        assert!(
            validate_descriptor(&bytes, &descriptor(&space(DistanceType::Cosine), &changed))
                .is_err()
        );
        assert!(validate_descriptor(b"not-json", &expected).is_err());
        assert!(validate_descriptor(&vec![b' '; MAX_DESCRIPTOR_BYTES + 1], &expected).is_err());
        assert_eq!(expected["adapter_version"], "0.60");
        assert_eq!(expected["format_version"], 4);
        assert_eq!(expected["atomic_write_protocol"], "guard-two-row-v1");
        assert_eq!(expected["deletion_policy"], "immutable-versions-v1");
        assert_eq!(expected["maintenance"], "deferred-oneshot-v1");
        assert_eq!(expected["normalize"], true);
        let mut obsolete = expected.clone();
        obsolete["format_version"] = serde_json::json!(1);
        assert!(validate_descriptor(&serde_json::to_vec(&obsolete).unwrap(), &expected).is_err());
        obsolete = expected.clone();
        obsolete["format_version"] = serde_json::json!(2);
        assert!(validate_descriptor(&serde_json::to_vec(&obsolete).unwrap(), &expected).is_err());
        obsolete = expected.clone();
        obsolete["maintenance"] = serde_json::json!("drop-deleted-neighbors");
        assert!(validate_descriptor(&serde_json::to_vec(&obsolete).unwrap(), &expected).is_err());
        obsolete = expected.clone();
        obsolete["atomic_write_protocol"] = serde_json::json!("guard-v1");
        assert!(validate_descriptor(&serde_json::to_vec(&obsolete).unwrap(), &expected).is_err());
    }

    #[test]
    fn prewrite_numeric_ranges_are_finite_and_metric_specific() {
        assert_eq!(
            validate_numeric_range(&[0.0, 0.0], DistanceType::Cosine).unwrap(),
            0.0
        );
        assert!(validate_numeric_range(&[f32::NAN], DistanceType::L2).is_err());
        assert!(validate_numeric_range(&[f32::INFINITY], DistanceType::Cosine).is_err());
        assert!(validate_numeric_range(&[f32::MAX], DistanceType::Cosine).is_err());
        let large = [1.0e19f32];
        assert!(validate_numeric_range(&large, DistanceType::Cosine).is_ok());
        let error = validate_numeric_range(&large, DistanceType::L2).unwrap_err();
        assert!(error.message.contains("squared norm <= f32::MAX/4"));
        assert!(validate_numeric_range(&[8.0e18f32], DistanceType::L2).is_ok());
        assert!(validate_numeric_range(&[1.0], DistanceType::InnerProduct).is_err());
    }

    #[test]
    fn journal_policy_requires_rollback_without_changing_settings() {
        for mode in ["delete", "truncate", "persist", "memory", "wal"] {
            validate_journal_mode(mode).unwrap();
        }
        let error = validate_journal_mode("off").unwrap_err();
        assert!(error.message.contains("journal_mode=OFF"));
        assert!(validate_journal_mode("unknown").is_err());
    }
    #[test]
    fn input_normalization_matches_single_and_keeps_zero_finite() {
        let mut l2 = [1.0, 2.0, 3.0];
        normalize_input(&mut l2, 3, DistanceType::L2).unwrap();
        assert_eq!(l2, [1.0, 2.0, 3.0]);
        let mut cosine = [3.0, 4.0, 0.0];
        let mut expected = cosine;
        ops::normalize_f32(&mut expected);
        normalize_input(&mut cosine, 3, DistanceType::Cosine).unwrap();
        assert_eq!(cosine, expected);
        let mut zero = [0.0; 3];
        normalize_input(&mut zero, 3, DistanceType::Cosine).unwrap();
        assert!(zero.iter().all(|value| value.is_finite()));
        assert!(normalize_input(&mut l2, 2, DistanceType::L2).is_err());
        assert!(normalize_input(&mut [f32::NAN], 1, DistanceType::L2).is_err());
        assert!(normalize_input(&mut [f32::MAX], 1, DistanceType::Cosine).is_err());
    }

    #[test]
    fn checked_size_and_id_boundaries() {
        assert_eq!(checked_id(MAX_ID).unwrap(), i64::MAX);
        assert!(checked_id(MAX_ID + 1).is_err());
        assert!(checked_increment(MAX_ID, "ID").is_err());
        assert!(checked_product(usize::MAX, 4, "vector").is_err());
        assert_eq!(checked_increment(0, "revision").unwrap(), 1);
    }
}
