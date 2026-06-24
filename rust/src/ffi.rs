//! Routing layer for the SQLite loadable-extension C API.
//!
//! A loadable extension must not link against libsqlite3 directly; instead the
//! host passes a `sqlite3_api_routines` table at load time and every SQLite call
//! is dispatched through it (this is what the `SQLITE_EXTENSION_INIT2` macro does
//! in C). We store that pointer once and expose thin typed wrappers. All of the
//! `unsafe` needed to talk to SQLite is concentrated here.

#![allow(non_upper_case_globals, non_camel_case_types, non_snake_case, dead_code)]

use std::os::raw::{c_char, c_int, c_void};

pub mod sys {
    pub use vectorlite_sqlite_sys::*;
}

pub use sys::*;

/// The host-provided API routine table. Set exactly once from the extension
/// entry point before any other SQLite call is made.
static mut API: *const sqlite3_api_routines = std::ptr::null();

/// Stores the API routine table. Called from `sqlite3_extension_init`.
pub unsafe fn set_api(api: *const sqlite3_api_routines) {
    API = api;
}

#[inline]
fn api() -> &'static sqlite3_api_routines {
    // Safe in practice: set_api runs before any wrapper is used.
    unsafe { &*API }
}

/// The `SQLITE_TRANSIENT` sentinel destructor: tells SQLite to copy the buffer.
#[inline]
pub fn transient() -> Option<unsafe extern "C" fn(*mut c_void)> {
    unsafe { std::mem::transmute(-1isize) }
}

// --- value accessors ---

pub unsafe fn value_type(v: *mut sqlite3_value) -> c_int {
    (api().value_type.unwrap())(v)
}
pub unsafe fn value_bytes(v: *mut sqlite3_value) -> c_int {
    (api().value_bytes.unwrap())(v)
}
pub unsafe fn value_blob(v: *mut sqlite3_value) -> *const c_void {
    (api().value_blob.unwrap())(v)
}
pub unsafe fn value_text(v: *mut sqlite3_value) -> *const u8 {
    (api().value_text.unwrap())(v)
}
pub unsafe fn value_int(v: *mut sqlite3_value) -> c_int {
    (api().value_int.unwrap())(v)
}
pub unsafe fn value_int64(v: *mut sqlite3_value) -> sqlite3_int64 {
    (api().value_int64.unwrap())(v)
}
pub unsafe fn value_pointer(v: *mut sqlite3_value, t: *const c_char) -> *mut c_void {
    (api().value_pointer.unwrap())(v, t)
}

/// Returns the bytes of a blob/text value as a slice (empty if NULL/zero-length).
pub unsafe fn value_blob_slice(v: *mut sqlite3_value) -> Vec<u8> {
    let n = value_bytes(v);
    if n <= 0 {
        return Vec::new();
    }
    let p = value_blob(v) as *const u8;
    if p.is_null() {
        return Vec::new();
    }
    std::slice::from_raw_parts(p, n as usize).to_vec()
}

pub unsafe fn value_text_string(v: *mut sqlite3_value) -> String {
    let n = value_bytes(v);
    if n <= 0 {
        return String::new();
    }
    let p = value_text(v);
    if p.is_null() {
        return String::new();
    }
    let bytes = std::slice::from_raw_parts(p, n as usize);
    String::from_utf8_lossy(bytes).into_owned()
}

// --- result setters ---

pub unsafe fn result_double(ctx: *mut sqlite3_context, d: f64) {
    (api().result_double.unwrap())(ctx, d)
}
pub unsafe fn result_null(ctx: *mut sqlite3_context) {
    (api().result_null.unwrap())(ctx)
}
pub unsafe fn result_blob(ctx: *mut sqlite3_context, data: &[u8]) {
    (api().result_blob.unwrap())(
        ctx,
        data.as_ptr() as *const c_void,
        data.len() as c_int,
        transient(),
    )
}
pub unsafe fn result_text(ctx: *mut sqlite3_context, s: &str) {
    (api().result_text.unwrap())(
        ctx,
        s.as_ptr() as *const c_char,
        s.len() as c_int,
        transient(),
    )
}
pub unsafe fn result_error(ctx: *mut sqlite3_context, msg: &str) {
    (api().result_error.unwrap())(ctx, msg.as_ptr() as *const c_char, msg.len() as c_int)
}
pub unsafe fn result_pointer(
    ctx: *mut sqlite3_context,
    p: *mut c_void,
    t: *const c_char,
    destructor: Option<unsafe extern "C" fn(*mut c_void)>,
) {
    (api().result_pointer.unwrap())(ctx, p, t, destructor)
}

// --- memory ---

pub unsafe fn sqlite_malloc(n: usize) -> *mut c_void {
    (api().malloc.unwrap())(n as c_int)
}
pub unsafe fn sqlite_free(p: *mut c_void) {
    (api().free.unwrap())(p)
}

/// Allocates a SQLite-owned NUL-terminated copy of `s` (freeable by SQLite).
pub unsafe fn sqlite_strdup(s: &str) -> *mut c_char {
    let bytes = s.as_bytes();
    let p = sqlite_malloc(bytes.len() + 1) as *mut u8;
    if p.is_null() {
        return std::ptr::null_mut();
    }
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), p, bytes.len());
    *p.add(bytes.len()) = 0;
    p as *mut c_char
}

/// Sets `*pp` to a SQLite-owned copy of `msg`, freeing any previous value.
pub unsafe fn set_err(pp: *mut *mut c_char, msg: &str) {
    if pp.is_null() {
        return;
    }
    if !(*pp).is_null() {
        sqlite_free(*pp as *mut c_void);
    }
    *pp = sqlite_strdup(msg);
}

// --- vtab in (IN-operator support) ---

pub unsafe fn vtab_in(info: *mut sqlite3_index_info, i: c_int, handle: c_int) -> c_int {
    (api().vtab_in.unwrap())(info, i, handle)
}
pub unsafe fn vtab_in_first(
    v: *mut sqlite3_value,
    out: *mut *mut sqlite3_value,
) -> c_int {
    (api().vtab_in_first.unwrap())(v, out)
}
pub unsafe fn vtab_in_next(
    v: *mut sqlite3_value,
    out: *mut *mut sqlite3_value,
) -> c_int {
    (api().vtab_in_next.unwrap())(v, out)
}

// --- schema / module / functions ---

pub unsafe fn declare_vtab(db: *mut sqlite3, sql: &str) -> c_int {
    let c = std::ffi::CString::new(sql).unwrap();
    (api().declare_vtab.unwrap())(db, c.as_ptr())
}

pub unsafe fn vtab_config_constraint_support(db: *mut sqlite3) -> c_int {
    (api().vtab_config.unwrap())(db, SQLITE_VTAB_CONSTRAINT_SUPPORT as c_int, 1 as c_int)
}

pub unsafe fn libversion_number() -> c_int {
    (api().libversion_number.unwrap())()
}

pub unsafe fn create_module_v2(
    db: *mut sqlite3,
    name: *const c_char,
    module: *const sqlite3_module,
    p_aux: *mut c_void,
    destroy: Option<unsafe extern "C" fn(*mut c_void)>,
) -> c_int {
    (api().create_module_v2.unwrap())(db, name, module, p_aux, destroy)
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn create_function(
    db: *mut sqlite3,
    name: *const c_char,
    n_arg: c_int,
    flags: c_int,
    p_app: *mut c_void,
    x_func: Option<
        unsafe extern "C" fn(*mut sqlite3_context, c_int, *mut *mut sqlite3_value),
    >,
) -> c_int {
    (api().create_function_v2.unwrap())(
        db, name, n_arg, flags, p_app, x_func, None, None, None,
    )
}
