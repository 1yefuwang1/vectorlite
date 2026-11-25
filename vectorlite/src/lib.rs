//! Vectorlite SQLite extension: virtual table, scalar functions, and index policy.
//! The native hnswlib and Highway kernels are reached through a narrow C ABI;
//! this crate is the main implementation and owns all SQLite integration.

#![deny(unsafe_op_in_unsafe_fn)]
#![deny(clippy::undocumented_unsafe_blocks, clippy::missing_safety_doc)]

mod atomic_callback;
mod batch_input;
mod core;
mod diskann_build;
mod diskann_build_guard;
mod diskann_control;
mod diskann_core;
#[path = "diskann_deferred/query.rs"]
mod diskann_query;
mod diskann_registration;
mod diskann_store;
mod diskann_target;
mod diskann_worker;
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

/// Callback-local initialization rollback. On a later registration failure,
/// remove only our successful prefix before SQLite may unload this library.
/// Deletion invokes each owned scalar's matching xDestroy exactly once.
struct RegistrationCleanup {
    db: *mut sqlite3,
    functions: [Option<(&'static str, c_int)>; 9],
    count: usize,
    armed: bool,
}
impl RegistrationCleanup {
    fn record(&mut self, name: &'static str, argc: c_int) {
        self.functions[self.count] = Some((name, argc));
        self.count += 1;
    }
}
impl Drop for RegistrationCleanup {
    fn drop(&mut self) {
        if self.armed {
            for (name, argc) in self.functions[..self.count].iter().flatten().rev() {
                let name = CString::new(*name).expect("static function name");
                // SAFETY: this guard never escapes the live, serialized extension
                // initialization callback. These names/counts were registered by
                // this attempt; NULL callbacks delete them and run owned xDestroy.
                let _ = unsafe {
                    ffi::create_function(
                        self.db,
                        name.as_ptr(),
                        *argc,
                        ffi::SQLITE_UTF8 as c_int,
                        std::ptr::null_mut(),
                        None,
                    )
                };
            }
        }
    }
}

/// All reserved scalar slots must be free before any callback/module overwrite.
/// Preparing and immediately finalizing invokes no scalar and performs no DML.
fn preflight_functions(
    connection: &crate::sqlite::Connection<'_>,
) -> Result<(), crate::index_error::IndexError> {
    for (name, argc) in [
        ("vector_distance", 3),
        ("vector_from_json", 1),
        ("vector_to_json", 1),
        ("knn_search", 2),
        ("knn_param", 2),
        ("knn_param", 3),
        ("vectorlite_info", 0),
        (atomic_callback::FUNCTION_NAME, 2),
        ("vectorlite_index_status", 2),
        ("vectorlite_index_build", 3),
    ] {
        let args = vec!["NULL"; argc].join(",");
        match connection.prepare(&format!("SELECT {name}({args})")) {
            Ok(statement) => {
                statement.finish()?;
                return Err(crate::index_error::IndexError::with_code(ffi::SQLITE_ERROR as c_int,
                    format!("vectorlite function {name} is already registered on this connection; load vectorlite once on a fresh connection")));
            }
            Err(error)
                if error.code & 0xff == ffi::SQLITE_ERROR as c_int
                    && error.message.contains(&format!("no such function: {name}")) => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
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

    // SAFETY: the entry point keeps db live and serialized for this local borrow;
    // all unstepped probe statements are finalized before registration starts.
    let preflight = unsafe { crate::sqlite::Connection::borrow(db) }
        .and_then(|connection| preflight_functions(&connection));
    if let Err(error) = preflight {
        // SAFETY: SQLite supplies this writable initialization error output.
        unsafe { ffi::set_err(pz_err_msg, &error.message) };
        return error.code;
    }
    let utf8 = ffi::SQLITE_UTF8 as c_int;
    let deterministic =
        (ffi::SQLITE_UTF8 | ffi::SQLITE_INNOCUOUS | ffi::SQLITE_DETERMINISTIC) as c_int;

    type ScalarFn =
        unsafe extern "C" fn(*mut ffi::sqlite3_context, c_int, *mut *mut ffi::sqlite3_value);
    let mut cleanup = RegistrationCleanup {
        db,
        functions: [None; 9],
        count: 0,
        armed: true,
    };
    let functions: &[(&'static str, c_int, c_int, ScalarFn)] = &[
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
        cleanup.record(name, n_arg);
    }

    let registration = match diskann_registration::Registration::new() {
        Ok(state) => state,
        Err(error) => {
            // SAFETY: the entry point supplies SQLite's writable error slot.
            unsafe { ffi::set_err(pz_err_msg, &error.message) };
            return error.code;
        }
    };
    for (function_name, argc, callback) in [
        (
            "vectorlite_index_status",
            2,
            diskann_control::index_status as ScalarFn,
        ),
        (
            "vectorlite_index_build",
            3,
            diskann_control::index_build as ScalarFn,
        ),
    ] {
        let name = CString::new(function_name).unwrap();
        let userdata = Box::into_raw(Box::new(std::sync::Arc::clone(&registration))).cast();
        // SAFETY: SQLite takes the unique boxed Arc and always invokes its matching
        // destructor, including registration failure. No Arc points back to a VTab.
        let rc = unsafe {
            ffi::create_owned_function(
                db,
                name.as_ptr(),
                argc,
                utf8 | ffi::SQLITE_DIRECTONLY as c_int,
                userdata,
                Some(callback),
                Some(diskann_control::registration_destroy),
            )
        };
        if rc != ffi::SQLITE_OK as c_int {
            return rc;
        }
        cleanup.record(function_name, argc);
    }
    let registry =
        Box::into_raw(Box::new(Registry::with_registration(registration))) as *mut c_void;
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

    cleanup.armed = false;
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
