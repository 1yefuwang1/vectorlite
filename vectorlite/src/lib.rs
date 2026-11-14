//! Vectorlite SQLite extension: virtual table, scalar functions, and index policy.
//! The native hnswlib and Highway kernels are reached through a narrow C ABI;
//! this crate is the main implementation and owns all SQLite integration.

#![deny(unsafe_op_in_unsafe_fn)]
#![deny(clippy::undocumented_unsafe_blocks, clippy::missing_safety_doc)]

mod atomic_callback;
mod batch_input;
mod core;
mod diskann_core;
mod diskann_store;
mod ffi;
mod half;
mod hnsw;
mod index_error;
mod index_options;
mod ops;
mod registry;
mod scalar;
mod sqlite;
mod vector;
mod vector_space;
mod virtual_table;

use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};

use ffi::{sqlite3, sqlite3_api_routines};
use registry::Registry;

unsafe extern "C" fn registry_destroy(p: *mut c_void) {
    if !p.is_null() {
        // SAFETY: SQLite releases the unique Box transferred as module pAux.
        unsafe { drop(Box::from_raw(p as *mut Registry)) };
    }
}

unsafe fn register_function(
    db: *mut sqlite3,
    pz_err_msg: *mut *mut c_char,
    name: &str,
    n_arg: c_int,
    flags: c_int,
    func: unsafe extern "C" fn(*mut ffi::sqlite3_context, c_int, *mut *mut ffi::sqlite3_value),
) -> c_int {
    let cname = CString::new(name).unwrap();
    // SAFETY: SQLite supplies the live database and the callback pointers
    // have the declared C ABI; the name CString lives through registration.
    let rc = unsafe {
        ffi::create_function(
            db,
            cname.as_ptr(),
            n_arg,
            flags,
            std::ptr::null_mut(),
            Some(func),
        )
    };
    if rc != ffi::SQLITE_OK as c_int {
        // SAFETY: SQLite supplies this output slot for an allocated error.
        unsafe { ffi::set_err(pz_err_msg, &format!("Failed to create function {name}")) };
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
    // SAFETY: SQLite supplies a live extension table; set_api reads only the
    // host-supported prefix and checks required callbacks before registration.
    if let Err(error) = unsafe { ffi::set_api(p_api) } {
        // SAFETY: use this incoming host's allocator even when initialization
        // was rejected before any global API table could be installed.
        unsafe { ffi::initialization_error(p_api, pz_err_msg, error) };
        return ffi::SQLITE_ERROR as c_int;
    }

    let utf8 = ffi::SQLITE_UTF8 as c_int;
    let deterministic =
        (ffi::SQLITE_UTF8 | ffi::SQLITE_INNOCUOUS | ffi::SQLITE_DETERMINISTIC) as c_int;

    type ScalarFn =
        unsafe extern "C" fn(*mut ffi::sqlite3_context, c_int, *mut *mut ffi::sqlite3_value);
    let functions: &[(&str, c_int, c_int, ScalarFn)] = &[
        ("vector_distance", 3, deterministic, scalar::vector_distance),
        (
            "vector_from_json",
            1,
            deterministic,
            scalar::vector_from_json,
        ),
        ("vector_to_json", 1, deterministic, scalar::vector_to_json),
        ("knn_search", 2, utf8, scalar::knn_search),
        ("knn_param", -1, utf8, scalar::knn_param),
        ("vectorlite_info", 0, utf8, scalar::vectorlite_info),
        (
            atomic_callback::FUNCTION_NAME,
            2,
            utf8 | ffi::SQLITE_DIRECTONLY as c_int,
            atomic_callback::invoke,
        ),
    ];
    for &(name, n_arg, flags, func) in functions {
        // SAFETY: inherited from the extension entry point contract.
        let rc = unsafe { register_function(db, pz_err_msg, name, n_arg, flags, func) };
        if rc != ffi::SQLITE_OK as c_int {
            return rc;
        }
    }

    let registry = Box::into_raw(Box::new(Registry::new())) as *mut c_void;
    let module_name = CString::new("vectorlite").unwrap();
    // SAFETY: SQLite retains the static module and owns registry from this
    // call onward, invoking registry_destroy even on registration failure.
    let rc = unsafe {
        ffi::create_module_v2(
            db,
            module_name.as_ptr(),
            virtual_table::module_ptr(),
            registry,
            Some(registry_destroy),
        )
    };
    if rc != ffi::SQLITE_OK as c_int {
        // SAFETY: SQLite supplies the error output slot.
        unsafe { ffi::set_err(pz_err_msg, "Failed to create module vectorlite") };
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
    // SAFETY: the alias has exactly the same SQLite entry-point contract.
    unsafe { sqlite3_extension_init(db, pz_err_msg, p_api) }
}
