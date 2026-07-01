//! Rust port of the vectorlite SQLite extension (virtual table + scalar
//! functions). The numeric core (hnswlib + SIMD ops + quantization) stays in C++
//! and is linked as a static library; this crate owns the SQLite glue.

#![allow(non_upper_case_globals, non_camel_case_types, non_snake_case)]

mod core;
mod ffi;
mod hnsw;
mod index_options;
mod ops;
mod registry;
mod scalar;
mod vector;
mod vector_space;
mod virtual_table;

use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};

use ffi::{sqlite3, sqlite3_api_routines};
use registry::Registry;

unsafe extern "C" fn registry_destroy(p: *mut c_void) {
    if !p.is_null() {
        drop(Box::from_raw(p as *mut Registry));
    }
}

unsafe fn register_function(
    db: *mut sqlite3,
    pz_err_msg: *mut *mut c_char,
    name: &str,
    n_arg: c_int,
    flags: c_int,
    func: unsafe extern "C" fn(
        *mut ffi::sqlite3_context,
        c_int,
        *mut *mut ffi::sqlite3_value,
    ),
) -> c_int {
    let cname = CString::new(name).unwrap();
    let rc = ffi::create_function(
        db,
        cname.as_ptr(),
        n_arg,
        flags,
        std::ptr::null_mut(),
        Some(func),
    );
    if rc != ffi::SQLITE_OK as c_int {
        ffi::set_err(pz_err_msg, &format!("Failed to create function {}", name));
    }
    rc
}

/// Loadable-extension entry point. SQLite resolves this symbol when the
/// `vectorlite` shared library is loaded.
///
/// # Safety
/// Called by SQLite with a valid database handle and API routine table.
#[no_mangle]
pub unsafe extern "C" fn sqlite3_extension_init(
    db: *mut sqlite3,
    pz_err_msg: *mut *mut c_char,
    p_api: *const sqlite3_api_routines,
) -> c_int {
    ffi::set_api(p_api);

    let utf8 = ffi::SQLITE_UTF8 as c_int;
    let deterministic_flags =
        (ffi::SQLITE_UTF8 | ffi::SQLITE_INNOCUOUS | ffi::SQLITE_DETERMINISTIC) as c_int;

    let rc = register_function(
        db,
        pz_err_msg,
        "vector_distance",
        3,
        deterministic_flags,
        scalar::vector_distance,
    );
    if rc != ffi::SQLITE_OK as c_int {
        return rc;
    }

    let rc = register_function(
        db,
        pz_err_msg,
        "vector_from_json",
        1,
        deterministic_flags,
        scalar::vector_from_json,
    );
    if rc != ffi::SQLITE_OK as c_int {
        return rc;
    }

    let rc = register_function(
        db,
        pz_err_msg,
        "vector_to_json",
        1,
        deterministic_flags,
        scalar::vector_to_json,
    );
    if rc != ffi::SQLITE_OK as c_int {
        return rc;
    }

    let rc = register_function(db, pz_err_msg, "knn_search", 2, utf8, scalar::knn_search);
    if rc != ffi::SQLITE_OK as c_int {
        return rc;
    }

    let rc = register_function(db, pz_err_msg, "knn_param", -1, utf8, scalar::knn_param);
    if rc != ffi::SQLITE_OK as c_int {
        return rc;
    }

    let rc = register_function(
        db,
        pz_err_msg,
        "vectorlite_info",
        0,
        utf8,
        scalar::vectorlite_info,
    );
    if rc != ffi::SQLITE_OK as c_int {
        return rc;
    }

    let registry = Box::into_raw(Box::new(Registry::new())) as *mut c_void;
    let module_name = CString::new("vectorlite").unwrap();
    let rc = ffi::create_module_v2(
        db,
        module_name.as_ptr(),
        virtual_table::module_ptr(),
        registry,
        Some(registry_destroy),
    );
    if rc != ffi::SQLITE_OK as c_int {
        ffi::set_err(pz_err_msg, "Failed to create module vectorlite");
        return rc;
    }

    ffi::SQLITE_OK as c_int
}

/// Filename-derived entry point alias (`sqlite3_<name>_init`), in case SQLite
/// looks it up instead of the generic `sqlite3_extension_init`.
///
/// # Safety
/// Same contract as `sqlite3_extension_init`.
#[no_mangle]
pub unsafe extern "C" fn sqlite3_vectorlite_init(
    db: *mut sqlite3,
    pz_err_msg: *mut *mut c_char,
    p_api: *const sqlite3_api_routines,
) -> c_int {
    sqlite3_extension_init(db, pz_err_msg, p_api)
}
