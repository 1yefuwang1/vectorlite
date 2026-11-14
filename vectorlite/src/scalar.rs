//! SQL scalar callbacks. Raw SQLite arguments are wrapped once at entry; all
//! validation, vector conversion and numerical policy below use safe Rust.

use std::os::raw::{c_char, c_int, c_void};

use crate::core;
use crate::ffi::{self, sqlite3_context, sqlite3_value, Context, Value};
use crate::vector;
use crate::vector_space::parse_distance_type;

/// Shared producer/consumer tag for SQLite's owned KNN parameter pointer.
pub const KNN_PARAM_TYPE: &[u8] = b"vectorlite_knn_param\0";

pub struct KnnParam {
    pub query_vector: Vec<f32>,
    pub k: u64,
    pub ef: Option<u64>,
    pub diskann_search_list_size: Option<u64>,
}

unsafe extern "C" fn knn_param_destroy(ptr: *mut c_void) {
    if !ptr.is_null() {
        // SAFETY: SQLite calls this destructor exactly once on the Box pointer
        // transferred by knn_param_impl through sqlite3_result_pointer.
        unsafe { drop(Box::from_raw(ptr.cast::<KnnParam>())) };
    }
}

/// Generates only the FFI adapter; implementations receive scoped safe values.
macro_rules! scalar {
    ($name:ident, $implementation:ident) => {
        pub unsafe extern "C" fn $name(
            ctx: *mut sqlite3_context,
            argc: c_int,
            argv: *mut *mut sqlite3_value,
        ) {
            // SAFETY: SQLite invokes this registered callback with a live
            // context and protected arguments valid for this invocation.
            unsafe { ffi::scalar_callback(ctx, argc, argv, $implementation) }
        }
    };
}

// knn_search is a planner marker. SQLite evaluates it through xFilter.
pub unsafe extern "C" fn knn_search(
    _ctx: *mut sqlite3_context,
    _argc: c_int,
    _argv: *mut *mut sqlite3_value,
) {
}

scalar!(knn_param, knn_param_impl);
scalar!(vector_distance, vector_distance_impl);
scalar!(vector_from_json, vector_from_json_impl);
scalar!(vector_to_json, vector_to_json_impl);
scalar!(vectorlite_info, vectorlite_info_impl);

fn knn_param_impl(ctx: &Context, args: &mut [Value<'_>]) -> Result<(), String> {
    if args.len() != 2 && args.len() != 3 {
        return Err("invalid number of parameters to knn_param(). 2 or 3 is expected".into());
    }
    if args[0].kind() != ffi::SQLITE_BLOB as c_int {
        return Err("vector(1st param of knn_param) should be of type Blob".into());
    }
    if args[1].kind() != ffi::SQLITE_INTEGER as c_int {
        return Err("k(2nd param of knn_param) should be of type INTEGER".into());
    }
    let k = args[1].int64();
    if k <= 0 {
        return Err("k should be greater than 0".into());
    }
    let (ef, diskann_search_list_size) = if args.len() == 3 {
        if args[2].kind() == ffi::SQLITE_TEXT as c_int {
            let options: serde_json::Value = serde_json::from_str(args[2].text()?)
                .map_err(|error| format!("Invalid DiskANN search options: {error}"))?;
            let object = options.as_object().ok_or_else(|| {
                "DiskANN search options must be a JSON object with search_list_size".to_owned()
            })?;
            if object.len() != 1 || !object.contains_key("search_list_size") {
                return Err("DiskANN search options support only search_list_size".into());
            }
            let search_list_size = object["search_list_size"]
                .as_u64()
                .filter(|&size| size > 0 && size <= i64::MAX as u64)
                .ok_or_else(|| "DiskANN search_list_size must be a positive integer".to_owned())?;
            (None, Some(search_list_size))
        } else {
            if args[2].kind() != ffi::SQLITE_INTEGER as c_int {
                return Err("ef(3rd param of knn_param) should be of type INTEGER".into());
            }
            let ef = args[2].int64();
            if ef <= 0 {
                return Err("ef should be greater than 0".into());
            }
            (Some(ef as u64), None)
        }
    } else {
        (None, None)
    };
    // KNN parameters outlive this callback, so retain exactly one owned vector.
    let query_vector = vector::view_from_blob(args[0].blob()?)
        .map_err(|error| format!("Failed to parse vector due to: {error}"))?
        .into_owned();
    let parameter = Box::new(KnnParam {
        query_vector,
        k: k as u64,
        ef,
        diskann_search_list_size,
    });
    // SAFETY: the static tag is NUL-terminated and shared with xFilter. SQLite
    // owns this Box until its value is released, using the matching destructor.
    unsafe {
        ctx.pointer(
            Box::into_raw(parameter).cast::<c_void>(),
            KNN_PARAM_TYPE.as_ptr().cast::<c_char>(),
            Some(knn_param_destroy),
        );
    }
    Ok(())
}

fn vector_distance_impl(ctx: &Context, args: &mut [Value<'_>]) -> Result<(), String> {
    let count = args.len();
    let [first, second, metric] = args else {
        return Err(format!(
            "vector_distance expects 3 arguments but {count} provided"
        ));
    };
    if first.kind() != ffi::SQLITE_BLOB as c_int || second.kind() != ffi::SQLITE_BLOB as c_int {
        return Err(format!(
            "vector_distance expects vectors of type blob but found {} and {}",
            first.kind(),
            second.kind()
        ));
    }
    if metric.kind() != ffi::SQLITE_TEXT as c_int {
        return Err("vector_distance expects space type of type text".into());
    }
    let metric = metric.text()?;
    let distance_type = parse_distance_type(metric)
        .ok_or_else(|| format!("Failed to parse space type: {metric}"))?;
    let first = vector::view_from_blob(first.blob()?)
        .map_err(|error| format!("Failed to parse 1st vector due to: {error}"))?;
    let second = vector::view_from_blob(second.blob()?)
        .map_err(|error| format!("Failed to parse 2nd vector due to: {error}"))?;
    let distance = core::distance(&first, &second, distance_type)
        .ok_or_else(|| format!("Dimension mismatch: {} != {}", first.len(), second.len()))?;
    ctx.double(distance as f64);
    Ok(())
}

fn vector_from_json_impl(ctx: &Context, args: &mut [Value<'_>]) -> Result<(), String> {
    let count = args.len();
    let [json] = args else {
        return Err(format!(
            "vector_from_json expects 1 argument but {count} provided"
        ));
    };
    if json.kind() != ffi::SQLITE_TEXT as c_int {
        return Err("vector_from_json expects a JSON string".into());
    }
    let values = vector::from_json(json.text()?)
        .map_err(|error| format!("Failed to parse vector due to: {error}"))?;
    ctx.blob(&vector::blob_from_f32(&values));
    Ok(())
}

fn vector_to_json_impl(ctx: &Context, args: &mut [Value<'_>]) -> Result<(), String> {
    let count = args.len();
    let [blob] = args else {
        return Err(format!(
            "vector_to_json expects 1 argument but {count} provided"
        ));
    };
    if blob.kind() != ffi::SQLITE_BLOB as c_int {
        return Err("vector_to_json expects vector of type blob".into());
    }
    let values = vector::view_from_blob(blob.blob()?)
        .map_err(|error| format!("Failed to parse vector due to: {error}"))?;
    ctx.text(&vector::to_json(&values)?);
    Ok(())
}

fn vectorlite_info_impl(ctx: &Context, _args: &mut [Value<'_>]) -> Result<(), String> {
    ctx.text(&format!(
        "vectorlite extension version {}. Best SIMD target in use: {}",
        env!("CARGO_PKG_VERSION"),
        core::best_target()
    ));
    Ok(())
}
