//! Scalar SQL functions: vector_distance, vector_from_json, vector_to_json,
//! knn_search (a BestIndex marker), knn_param and vectorlite_info.
//! Mirrors `sqlite_functions.cpp` and the knn helpers in `virtual_table.cpp`.

use std::os::raw::{c_char, c_int, c_void};

use crate::core;
use crate::ffi::{self, sqlite3_context, sqlite3_value};
use crate::vector;
use crate::vector_space::parse_distance_type;

/// Pointer-type tag shared by knn_param (producer) and the knn_search
/// constraint (consumer). Must be a stable, NUL-terminated string.
pub const KNN_PARAM_TYPE: &[u8] = b"vectorlite_knn_param\0";

/// Parameters carried from `knn_param(...)` to the BestIndex/Filter machinery.
pub struct KnnParam {
    pub query_vector: Vec<f32>,
    pub k: u32,
    pub ef: Option<u32>,
}

unsafe extern "C" fn knn_param_destroy(p: *mut c_void) {
    if !p.is_null() {
        drop(Box::from_raw(p as *mut KnnParam));
    }
}

unsafe fn arg(argv: *mut *mut sqlite3_value, i: usize) -> *mut sqlite3_value {
    *argv.add(i)
}

/// `knn_search(vector, knn_param(...))` — a marker consulted by xFindFunction /
/// xBestIndex; it is never actually evaluated as a row function.
pub unsafe extern "C" fn knn_search(
    _ctx: *mut sqlite3_context,
    _argc: c_int,
    _argv: *mut *mut sqlite3_value,
) {
}

pub unsafe extern "C" fn knn_param(
    ctx: *mut sqlite3_context,
    argc: c_int,
    argv: *mut *mut sqlite3_value,
) {
    if argc != 2 && argc != 3 {
        ffi::result_error(
            ctx,
            "invalid number of paramters to knn_param(). 2 or 3 is expected",
        );
        return;
    }
    if ffi::value_type(arg(argv, 0)) != ffi::SQLITE_BLOB as c_int {
        ffi::result_error(ctx, "vector(1st param of knn_param) should be of type Blob");
        return;
    }
    if ffi::value_type(arg(argv, 1)) != ffi::SQLITE_INTEGER as c_int {
        ffi::result_error(ctx, "k(2nd param of knn_param) should be of type INTEGER");
        return;
    }
    if argc == 3 && ffi::value_type(arg(argv, 2)) != ffi::SQLITE_INTEGER as c_int {
        ffi::result_error(ctx, "ef(3rd param of knn_param) should be of type INTEGER");
        return;
    }

    let blob = ffi::value_blob_slice(arg(argv, 0));
    let query_vector = match vector::blob_to_f32(&blob) {
        Ok(v) => v,
        Err(e) => {
            ffi::result_error(ctx, &format!("Failed to parse vector due to: {}", e));
            return;
        }
    };

    let k = ffi::value_int(arg(argv, 1));
    if k <= 0 {
        ffi::result_error(ctx, "k should be greater than 0");
        return;
    }

    let mut ef: Option<u32> = None;
    if argc == 3 {
        let e = ffi::value_int(arg(argv, 2));
        if e <= 0 {
            ffi::result_error(ctx, "ef should be greater than 0");
            return;
        }
        ef = Some(e as u32);
    }

    let param = Box::new(KnnParam {
        query_vector,
        k: k as u32,
        ef,
    });
    ffi::result_pointer(
        ctx,
        Box::into_raw(param) as *mut c_void,
        KNN_PARAM_TYPE.as_ptr() as *const c_char,
        Some(knn_param_destroy),
    );
}

pub unsafe extern "C" fn vector_distance(
    ctx: *mut sqlite3_context,
    argc: c_int,
    argv: *mut *mut sqlite3_value,
) {
    if argc != 3 {
        ffi::result_error(
            ctx,
            &format!("vector_distance expects 3 arguments but {} provided", argc),
        );
        return;
    }
    let t0 = ffi::value_type(arg(argv, 0));
    let t1 = ffi::value_type(arg(argv, 1));
    if t0 != ffi::SQLITE_BLOB as c_int || t1 != ffi::SQLITE_BLOB as c_int {
        ffi::result_error(
            ctx,
            &format!(
                "vector_distance expects vectors of type blob but found {} and {}",
                t0, t1
            ),
        );
        return;
    }
    if ffi::value_type(arg(argv, 2)) != ffi::SQLITE_TEXT as c_int {
        ffi::result_error(ctx, "vectors_distance expects space type of type text");
        return;
    }

    let space_str = ffi::value_text_string(arg(argv, 2));
    let distance_type = match parse_distance_type(&space_str) {
        Some(d) => d,
        None => {
            ffi::result_error(ctx, &format!("Failed to parse space type: {}", space_str));
            return;
        }
    };

    let b0 = ffi::value_blob_slice(arg(argv, 0));
    let v0 = match vector::blob_to_f32(&b0) {
        Ok(v) => v,
        Err(e) => {
            ffi::result_error(ctx, &format!("Failed to parse 1st vector due to: {}", e));
            return;
        }
    };
    let b1 = ffi::value_blob_slice(arg(argv, 1));
    let v1 = match vector::blob_to_f32(&b1) {
        Ok(v) => v,
        Err(e) => {
            ffi::result_error(ctx, &format!("Failed to parse 2nd vector due to: {}", e));
            return;
        }
    };

    if v0.len() != v1.len() {
        ffi::result_error(
            ctx,
            &format!("Dimension mismatch: {} != {}", v0.len(), v1.len()),
        );
        return;
    }

    match core::distance(&v0, &v1, distance_type) {
        Some(d) => ffi::result_double(ctx, d as f64),
        None => ffi::result_error(ctx, "Invalid distance type"),
    }
}

pub unsafe extern "C" fn vector_from_json(
    ctx: *mut sqlite3_context,
    argc: c_int,
    argv: *mut *mut sqlite3_value,
) {
    if argc != 1 {
        ffi::result_error(
            ctx,
            &format!("vector_from_json expects 1 argument but {} provided", argc),
        );
        return;
    }
    if ffi::value_type(arg(argv, 0)) != ffi::SQLITE_TEXT as c_int {
        ffi::result_error(ctx, "vector_from_json expects a JSON string");
        return;
    }
    let json = ffi::value_text_string(arg(argv, 0));
    match vector::from_json(&json) {
        Ok(v) => ffi::result_blob(ctx, &vector::f32_to_blob(&v)),
        Err(e) => ffi::result_error(ctx, &format!("Failed to parse vector due to: {}", e)),
    }
}

pub unsafe extern "C" fn vector_to_json(
    ctx: *mut sqlite3_context,
    argc: c_int,
    argv: *mut *mut sqlite3_value,
) {
    if argc != 1 {
        ffi::result_error(
            ctx,
            &format!("vector_to_json expects 1 argument but {} provided", argc),
        );
        return;
    }
    if ffi::value_type(arg(argv, 0)) != ffi::SQLITE_BLOB as c_int {
        ffi::result_error(ctx, "vector_to_json expects vector of type blob");
        return;
    }
    let blob = ffi::value_blob_slice(arg(argv, 0));
    match vector::blob_to_f32(&blob) {
        Ok(v) => ffi::result_text(ctx, &vector::to_json(&v)),
        Err(e) => ffi::result_error(ctx, &format!("Failed to parse vector due to: {}", e)),
    }
}

pub unsafe extern "C" fn vectorlite_info(
    ctx: *mut sqlite3_context,
    _argc: c_int,
    _argv: *mut *mut sqlite3_value,
) {
    let info = format!(
        "vectorlite extension version {}. Best SIMD target in use: {}",
        env!("CARGO_PKG_VERSION"),
        core::best_target()
    );
    ffi::result_text(ctx, &info);
}
