//! Routing layer for the SQLite loadable-extension C API.
//!
//! A loadable extension must not link against libsqlite3 directly; instead the
//! host passes a `sqlite3_api_routines` table at load time and every SQLite call
//! is dispatched through it (this is what the `SQLITE_EXTENSION_INIT2` macro does
//! in C). We store that pointer once and expose thin typed wrappers. All of the
//! `unsafe` needed to talk to SQLite is concentrated here.

use std::marker::PhantomData;
use std::os::raw::{c_char, c_int, c_void};
use std::sync::atomic::{AtomicPtr, Ordering};

pub mod sys {
    pub use vectorlite_sqlite_sys::*;
}

pub use sys::*;

/// Host-owned table. Read individual fields through raw pointers: an older
/// SQLite host owns only the prefix appropriate to its version, not an entire
/// instance of the latest generated sqlite3_api_routines type.
static API: AtomicPtr<sqlite3_api_routines> = AtomicPtr::new(std::ptr::null_mut());

macro_rules! api_field {
    ($field:ident) => {{
        let table = API.load(Ordering::Acquire);
        // SAFETY: set_api validates the minimum version and every used entry
        // before registration. Optional IN APIs are guarded by the host version.
        // addr_of! projects only this field, never a reference to the whole table.
        std::ptr::addr_of!((*table).$field)
            .read()
            .expect("validated SQLite API")
    }};
}

/// Validates the host ABI before registering any callbacks.
///
/// # Safety
/// `table` must be the host's live SQLite extension table. Only the prefix
/// provided by that host version needs to exist. It must outlive all callbacks.
pub unsafe fn set_api(table: *const sqlite3_api_routines) -> Result<(), &'static str> {
    if table.is_null() {
        return Err("SQLite API table is null");
    }
    // SAFETY: libversion_number belongs to the original extension-table prefix.
    let version = unsafe { std::ptr::addr_of!((*table).libversion_number).read() }
        .ok_or("SQLite version API is unavailable")?;
    // SAFETY: the host supplied this valid function pointer.
    let version = unsafe { version() };
    if version < 3020000 {
        return Err("SQLite 3.20.0 or newer is required");
    }
    macro_rules! require {
        ($($field:ident),+ $(,)?) => {$(
            // SAFETY: all entries listed below exist in the checked host prefix.
            if unsafe { std::ptr::addr_of!((*table).$field).read() }.is_none() {
                return Err(concat!("SQLite API is unavailable: ", stringify!($field)));
            }
        )+};
    }
    require!(
        value_type,
        value_bytes,
        value_blob,
        value_text,
        value_int64,
        value_pointer,
        result_double,
        result_null,
        result_blob,
        result_text,
        result_error,
        result_pointer,
        malloc,
        free,
        declare_vtab,
        vtab_config,
        create_module_v2,
        create_function_v2,
        prepare_v2,
        finalize,
        bind_int64,
        bind_blob,
        bind_text,
        bind_pointer,
        context_db_handle,
        result_error_code,
        step,
        column_count,
        column_type,
        column_int64,
        column_blob,
        column_bytes,
        column_text,
        errmsg,
        extended_errcode,
        randomness,
        changes
    );
    if version >= 3038000 {
        require!(vtab_in, vtab_in_first, vtab_in_next);
    }
    match API.compare_exchange(
        std::ptr::null_mut(),
        table.cast_mut(),
        Ordering::AcqRel,
        Ordering::Acquire,
    ) {
        Ok(_) => Ok(()),
        Err(existing) if std::ptr::eq(existing, table) => Ok(()),
        Err(_) => Err("loading the same extension into different SQLite runtimes is unsupported"),
    }
}

/// Reports an initialization error through the incoming host's allocator,
/// including when validation failed before the global table was installed.
///
/// # Safety
/// `table` and `output` come from SQLite's extension entry point. The original
/// API prefix is valid even when the host is older than this extension supports.
pub unsafe fn initialization_error(
    table: *const sqlite3_api_routines,
    output: *mut *mut c_char,
    message: &str,
) {
    if table.is_null() || output.is_null() {
        return;
    }
    let Some(length) = message
        .len()
        .checked_add(1)
        .and_then(|n| c_int::try_from(n).ok())
    else {
        return;
    };
    // SAFETY: malloc is in the original API prefix; output is SQLite's writable
    // error slot. The byte copy and terminator fit the checked allocation.
    unsafe {
        let Some(allocate) = std::ptr::addr_of!((*table).malloc).read() else {
            return;
        };
        let buffer = allocate(length).cast::<u8>();
        if !buffer.is_null() {
            std::ptr::copy_nonoverlapping(message.as_ptr(), buffer, message.len());
            *buffer.add(message.len()) = 0;
            *output = buffer.cast::<c_char>();
        }
    }
}

/// The `SQLITE_TRANSIENT` sentinel destructor: tells SQLite to copy the buffer.
#[inline]
pub fn transient() -> Option<unsafe extern "C" fn(*mut c_void)> {
    // SAFETY: SQLite specifies -1 cast to its destructor callback type as
    // SQLITE_TRANSIENT. SQLite treats it as a sentinel and never calls it.
    unsafe { std::mem::transmute::<isize, Option<unsafe extern "C" fn(*mut c_void)>>(-1) }
}

// --- value accessors ---

pub unsafe fn value_type(v: *mut sqlite3_value) -> c_int {
    // SAFETY: the caller supplies valid SQLite arguments with the documented
    // callback lifetime; set_api validated the dispatched function pointers.
    unsafe { (api_field!(value_type))(v) }
}
pub unsafe fn value_bytes(v: *mut sqlite3_value) -> c_int {
    // SAFETY: the caller supplies valid SQLite arguments with the documented
    // callback lifetime; set_api validated the dispatched function pointers.
    unsafe { (api_field!(value_bytes))(v) }
}
pub unsafe fn value_blob(v: *mut sqlite3_value) -> *const c_void {
    // SAFETY: the caller supplies valid SQLite arguments with the documented
    // callback lifetime; set_api validated the dispatched function pointers.
    unsafe { (api_field!(value_blob))(v) }
}
pub unsafe fn value_text(v: *mut sqlite3_value) -> *const u8 {
    // SAFETY: the caller supplies valid SQLite arguments with the documented
    // callback lifetime; set_api validated the dispatched function pointers.
    unsafe { (api_field!(value_text))(v) }
}
pub unsafe fn value_int64(v: *mut sqlite3_value) -> sqlite3_int64 {
    // SAFETY: the caller supplies valid SQLite arguments with the documented
    // callback lifetime; set_api validated the dispatched function pointers.
    unsafe { (api_field!(value_int64))(v) }
}
pub unsafe fn value_pointer(v: *mut sqlite3_value, t: *const c_char) -> *mut c_void {
    // SAFETY: the caller supplies valid SQLite arguments with the documented
    // callback lifetime; set_api validated the dispatched function pointers.
    unsafe { (api_field!(value_pointer))(v, t) }
}

// --- result setters ---

pub unsafe fn result_double(ctx: *mut sqlite3_context, d: f64) {
    // SAFETY: the caller supplies valid SQLite arguments with the documented
    // callback lifetime; set_api validated the dispatched function pointers.
    unsafe { (api_field!(result_double))(ctx, d) }
}
pub unsafe fn result_null(ctx: *mut sqlite3_context) {
    // SAFETY: the caller supplies valid SQLite arguments with the documented
    // callback lifetime; set_api validated the dispatched function pointers.
    unsafe { (api_field!(result_null))(ctx) }
}
pub unsafe fn result_blob(ctx: *mut sqlite3_context, data: &[u8]) {
    let Ok(length) = c_int::try_from(data.len()) else {
        // SAFETY: ctx is the same valid callback context supplied by the caller.
        unsafe { result_error(ctx, "result exceeds SQLite's maximum byte length") };
        return;
    };
    // SAFETY: the caller supplies valid SQLite arguments with the documented
    // callback lifetime; set_api validated the dispatched function pointers.
    unsafe { (api_field!(result_blob))(ctx, data.as_ptr() as *const c_void, length, transient()) }
}
pub unsafe fn result_text(ctx: *mut sqlite3_context, s: &str) {
    let Ok(length) = c_int::try_from(s.len()) else {
        // SAFETY: ctx is the same valid callback context supplied by the caller.
        unsafe { result_error(ctx, "result exceeds SQLite's maximum byte length") };
        return;
    };
    // SAFETY: the caller supplies valid SQLite arguments with the documented
    // callback lifetime; set_api validated the dispatched function pointers.
    unsafe { (api_field!(result_text))(ctx, s.as_ptr() as *const c_char, length, transient()) }
}
pub unsafe fn result_error(ctx: *mut sqlite3_context, msg: &str) {
    // SAFETY: the caller supplies valid SQLite arguments with the documented
    // callback lifetime; set_api validated the dispatched function pointers.
    unsafe {
        (api_field!(result_error))(
            ctx,
            msg.as_ptr() as *const c_char,
            c_int::try_from(msg.len()).unwrap_or(c_int::MAX),
        )
    }
}
pub unsafe fn result_pointer(
    ctx: *mut sqlite3_context,
    p: *mut c_void,
    t: *const c_char,
    destructor: Option<unsafe extern "C" fn(*mut c_void)>,
) {
    // SAFETY: the caller supplies valid SQLite arguments with the documented
    // callback lifetime; set_api validated the dispatched function pointers.
    unsafe { (api_field!(result_pointer))(ctx, p, t, destructor) }
}

// --- memory ---

pub unsafe fn sqlite_malloc(n: usize) -> *mut c_void {
    // SAFETY: the caller supplies valid SQLite arguments with the documented
    // callback lifetime; set_api validated the dispatched function pointers.
    unsafe {
        match c_int::try_from(n) {
            Ok(n) => (api_field!(malloc))(n),
            Err(_) => std::ptr::null_mut(),
        }
    }
}
pub unsafe fn sqlite_free(p: *mut c_void) {
    // SAFETY: the caller supplies valid SQLite arguments with the documented
    // callback lifetime; set_api validated the dispatched function pointers.
    unsafe { (api_field!(free))(p) }
}

/// Allocates a SQLite-owned NUL-terminated copy of `s` (freeable by SQLite).
pub unsafe fn sqlite_strdup(s: &str) -> *mut c_char {
    // SAFETY: the caller supplies valid SQLite arguments with the documented
    // callback lifetime; set_api validated the dispatched function pointers.
    unsafe {
        let bytes = s.as_bytes();
        let Some(length) = bytes.len().checked_add(1) else {
            return std::ptr::null_mut();
        };
        let p = sqlite_malloc(length) as *mut u8;
        if p.is_null() {
            return std::ptr::null_mut();
        }
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), p, bytes.len());
        *p.add(bytes.len()) = 0;
        p as *mut c_char
    }
}

/// Sets `*pp` to a SQLite-owned copy of `msg`, freeing any previous value.
pub unsafe fn set_err(pp: *mut *mut c_char, msg: &str) {
    // SAFETY: the caller supplies valid SQLite arguments with the documented
    // callback lifetime; set_api validated the dispatched function pointers.
    unsafe {
        if pp.is_null() {
            return;
        }
        if !(*pp).is_null() {
            sqlite_free(*pp as *mut c_void);
        }
        *pp = sqlite_strdup(msg);
    }
}

// --- vtab in (IN-operator support) ---

pub unsafe fn vtab_in(info: *mut sqlite3_index_info, i: c_int, handle: c_int) -> c_int {
    // SAFETY: the caller supplies valid SQLite arguments with the documented
    // callback lifetime; set_api validated the dispatched function pointers.
    unsafe { (api_field!(vtab_in))(info, i, handle) }
}
pub unsafe fn vtab_in_first(v: *mut sqlite3_value, out: *mut *mut sqlite3_value) -> c_int {
    // SAFETY: the caller supplies valid SQLite arguments with the documented
    // callback lifetime; set_api validated the dispatched function pointers.
    unsafe { (api_field!(vtab_in_first))(v, out) }
}
pub unsafe fn vtab_in_next(v: *mut sqlite3_value, out: *mut *mut sqlite3_value) -> c_int {
    // SAFETY: the caller supplies valid SQLite arguments with the documented
    // callback lifetime; set_api validated the dispatched function pointers.
    unsafe { (api_field!(vtab_in_next))(v, out) }
}

// --- prepared statements on the borrowed host connection ---

// These entries all precede the SQLite 3.20 API prefix validated by set_api.
// Keep raw routing here; sqlite.rs owns statement state and buffer lifetimes.
macro_rules! statement_api {
    ($name:ident($($argument:ident: $kind:ty),* $(,)?) -> $result:ty) => {
        /// Routes through the initialized host's loadable-extension API.
        ///
        /// # Safety
        /// Arguments must satisfy this SQLite routine's connection, statement,
        /// buffer, and output-pointer contracts. The API must be initialized.
        pub unsafe fn $name($($argument: $kind),*) -> $result {
            // SAFETY: the caller supplies this routine's SQLite argument
            // contract; set_api validated this entry in the supported prefix.
            unsafe { (api_field!($name))($($argument),*) }
        }
    };
}

statement_api!(prepare_v2(
    db: *mut sqlite3,
    sql: *const c_char,
    length: c_int,
    statement: *mut *mut sqlite3_stmt,
    tail: *mut *const c_char,
) -> c_int);
statement_api!(finalize(statement: *mut sqlite3_stmt) -> c_int);
statement_api!(bind_int64(statement: *mut sqlite3_stmt, index: c_int, value: i64) -> c_int);
statement_api!(bind_blob(
    statement: *mut sqlite3_stmt,
    index: c_int,
    value: *const c_void,
    length: c_int,
    destructor: sqlite3_destructor_type,
) -> c_int);
statement_api!(bind_text(
    statement: *mut sqlite3_stmt,
    index: c_int,
    value: *const c_char,
    length: c_int,
    destructor: sqlite3_destructor_type,
) -> c_int);
statement_api!(bind_pointer(
    statement: *mut sqlite3_stmt,
    index: c_int,
    value: *mut c_void,
    tag: *const c_char,
    destructor: sqlite3_destructor_type,
) -> c_int);
statement_api!(context_db_handle(context: *mut sqlite3_context) -> *mut sqlite3);
statement_api!(result_error_code(context: *mut sqlite3_context, code: c_int) -> ());
statement_api!(step(statement: *mut sqlite3_stmt) -> c_int);
statement_api!(column_count(statement: *mut sqlite3_stmt) -> c_int);
statement_api!(column_type(statement: *mut sqlite3_stmt, column: c_int) -> c_int);
statement_api!(column_int64(statement: *mut sqlite3_stmt, column: c_int) -> i64);
statement_api!(column_blob(statement: *mut sqlite3_stmt, column: c_int) -> *const c_void);
statement_api!(column_bytes(statement: *mut sqlite3_stmt, column: c_int) -> c_int);
statement_api!(column_text(statement: *mut sqlite3_stmt, column: c_int) -> *const u8);
statement_api!(errmsg(db: *mut sqlite3) -> *const c_char);
statement_api!(extended_errcode(db: *mut sqlite3) -> c_int);
statement_api!(randomness(length: c_int, output: *mut c_void) -> ());
statement_api!(changes(db: *mut sqlite3) -> c_int);

// --- schema / module / functions ---

pub unsafe fn declare_vtab(db: *mut sqlite3, sql: &str) -> c_int {
    // SAFETY: the caller supplies valid SQLite arguments with the documented
    // callback lifetime; set_api validated the dispatched function pointers.
    unsafe {
        let Ok(c) = std::ffi::CString::new(sql) else {
            return SQLITE_ERROR as c_int;
        };
        (api_field!(declare_vtab))(db, c.as_ptr())
    }
}

pub unsafe fn vtab_config_constraint_support(db: *mut sqlite3) -> c_int {
    // SAFETY: the caller supplies valid SQLite arguments with the documented
    // callback lifetime; set_api validated the dispatched function pointers.
    unsafe { (api_field!(vtab_config))(db, SQLITE_VTAB_CONSTRAINT_SUPPORT as c_int, 1 as c_int) }
}

/// Disables constraint-support promises for a backend with multi-row writes.
///
/// # Safety
/// Called only from xCreate/xConnect with their live host connection.
pub unsafe fn vtab_config_constraint_support_disabled(db: *mut sqlite3) -> c_int {
    // SAFETY: the caller supplies the constructor's live connection; this
    // configuration takes one variadic int and the API entry is validated.
    unsafe { (api_field!(vtab_config))(db, SQLITE_VTAB_CONSTRAINT_SUPPORT as c_int, 0 as c_int) }
}

pub unsafe fn vtab_config_directonly(db: *mut sqlite3) -> c_int {
    // SAFETY: the caller supplies a live database during xCreate/xConnect;
    // SQLITE_VTAB_DIRECTONLY takes no variadic argument.
    unsafe { (api_field!(vtab_config))(db, SQLITE_VTAB_DIRECTONLY as c_int) }
}

pub unsafe fn libversion_number() -> c_int {
    // SAFETY: the caller supplies valid SQLite arguments with the documented
    // callback lifetime; set_api validated the dispatched function pointers.
    unsafe { (api_field!(libversion_number))() }
}

pub unsafe fn create_module_v2(
    db: *mut sqlite3,
    name: *const c_char,
    module: *const sqlite3_module,
    p_aux: *mut c_void,
    destroy: Option<unsafe extern "C" fn(*mut c_void)>,
) -> c_int {
    // SAFETY: the caller supplies valid SQLite arguments with the documented
    // callback lifetime; set_api validated the dispatched function pointers.
    unsafe { (api_field!(create_module_v2))(db, name, module, p_aux, destroy) }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn create_function(
    db: *mut sqlite3,
    name: *const c_char,
    n_arg: c_int,
    flags: c_int,
    p_app: *mut c_void,
    x_func: Option<unsafe extern "C" fn(*mut sqlite3_context, c_int, *mut *mut sqlite3_value)>,
) -> c_int {
    // SAFETY: the caller supplies valid SQLite arguments with the documented
    // callback lifetime; set_api validated the dispatched function pointers.
    unsafe {
        (api_field!(create_function_v2))(db, name, n_arg, flags, p_app, x_func, None, None, None)
    }
}

/// A protected SQLite value valid for one callback. Conversion methods require
/// a mutable borrow so an outstanding byte/text view cannot be invalidated.
#[repr(transparent)]
pub struct Value<'a> {
    ptr: *mut sqlite3_value,
    lifetime: PhantomData<&'a mut sqlite3_value>,
}

impl<'a> Value<'a> {
    /// # Safety
    /// `ptr` must be a non-null protected value for `'a`. While views borrowed
    /// from this wrapper exist, no alias may convert or release that value.
    pub unsafe fn from_raw(ptr: *mut sqlite3_value) -> Self {
        Self {
            ptr,
            lifetime: PhantomData,
        }
    }

    pub fn as_ptr(&self) -> *mut sqlite3_value {
        self.ptr
    }

    pub fn kind(&self) -> c_int {
        // SAFETY: construction guarantees a protected value for this lifetime.
        unsafe { value_type(self.ptr) }
    }

    /// Returns the integer payload without coercing a different representation.
    pub fn int64(&self) -> i64 {
        if self.kind() != SQLITE_INTEGER as c_int {
            return 0;
        }
        // SAFETY: the value is already INTEGER, so no borrowed representation
        // is converted or invalidated, including aliases in SQLite's argv.
        unsafe { value_int64(self.ptr) }
    }

    pub fn blob(&mut self) -> Result<&[u8], &'static str> {
        if self.kind() != SQLITE_BLOB as c_int {
            return Err("value must be of type Blob");
        }
        // SAFETY: the value is already BLOB. Read the pointer before its byte
        // length, and tie the view to this mutable borrow. No conversion occurs.
        unsafe {
            let ptr = value_blob(self.ptr).cast::<u8>();
            let len = value_bytes(self.ptr);
            if len == 0 {
                return Ok(&[]);
            }
            if ptr.is_null() || len < 0 {
                return Err("unable to read SQLite blob");
            }
            Ok(std::slice::from_raw_parts(ptr, len as usize))
        }
    }

    pub fn text(&mut self) -> Result<&str, &'static str> {
        if self.kind() != SQLITE_TEXT as c_int {
            return Err("value must be of type TEXT");
        }
        // SAFETY: SQLite keeps the converted UTF-8 storage alive throughout
        // this callback and until another conversion; the mutable borrow
        // prevents conversion through this wrapper while the string is used.
        unsafe {
            let ptr = value_text(self.ptr);
            let len = value_bytes(self.ptr);
            if ptr.is_null() || len < 0 {
                return Err("unable to read SQLite text");
            }
            std::str::from_utf8(std::slice::from_raw_parts(ptr, len as usize))
                .map_err(|_| "SQLite text is not valid UTF-8")
        }
    }

    /// # Safety
    /// `tag` is a valid static NUL-terminated SQLite pointer tag. The returned
    /// pointer may only be dereferenced using the producer's type and lifetime.
    pub unsafe fn pointer(&self, tag: *const c_char) -> *mut c_void {
        // SAFETY: construction and the caller's tag contract satisfy SQLite.
        unsafe { value_pointer(self.ptr, tag) }
    }
}

/// Wraps callback arguments without allocation.
///
/// # Safety
/// SQLite supplies `argc` non-null protected values in a valid `argv` array,
/// exclusively available to the current callback for the returned lifetime.
/// Conversion through external aliases must not invalidate borrowed views.
pub unsafe fn arguments<'a>(argc: c_int, argv: *mut *mut sqlite3_value) -> &'a mut [Value<'a>] {
    if argc <= 0 {
        return &mut [];
    }
    // SAFETY: Value is transparent over its pointer; PhantomData has no layout
    // effect. The caller guarantees the argument-array length and lifetime.
    unsafe { std::slice::from_raw_parts_mut(argv.cast::<Value<'a>>(), argc as usize) }
}

/// A scalar callback's output channel. SQLite copies borrowed result data.
pub struct Context {
    ptr: *mut sqlite3_context,
}

impl Context {
    pub fn double(&self, value: f64) {
        // SAFETY: constructed only for a live callback context.
        unsafe { result_double(self.ptr, value) }
    }
    pub fn blob(&self, value: &[u8]) {
        // SAFETY: SQLite_TRANSIENT copies the live byte slice before returning.
        unsafe { result_blob(self.ptr, value) }
    }
    pub fn text(&self, value: &str) {
        // SAFETY: SQLite_TRANSIENT copies the live UTF-8 string before returning.
        unsafe { result_text(self.ptr, value) }
    }
    pub fn error(&self, value: &str) {
        // SAFETY: SQLite copies the live error string before returning.
        unsafe { result_error(self.ptr, value) }
    }
    /// # Safety
    /// The pointer, type tag and destructor must share the producer/consumer
    /// ownership contract required by sqlite3_result_pointer.
    pub unsafe fn pointer(
        &self,
        value: *mut c_void,
        tag: *const c_char,
        destructor: Option<unsafe extern "C" fn(*mut c_void)>,
    ) {
        // SAFETY: the caller supplies the pointer ownership and tag contract.
        unsafe { result_pointer(self.ptr, value, tag, destructor) }
    }
}

/// Dispatches one SQL scalar callback into safe Rust code.
///
/// # Safety
/// All pointers and argument lifetimes must satisfy SQLite's scalar callback
/// contract. The callback must not allow argument views or the context to escape.
pub unsafe fn scalar_callback(
    ctx: *mut sqlite3_context,
    argc: c_int,
    argv: *mut *mut sqlite3_value,
    callback: impl FnOnce(&Context, &mut [Value<'_>]) -> Result<(), String>,
) {
    let ctx = Context { ptr: ctx };
    // SAFETY: inherited directly from the SQLite callback contract above.
    let args = unsafe { arguments(argc, argv) };
    if let Err(error) = callback(&ctx, args) {
        ctx.error(&error);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    const CASE_ENV: &str = "VECTORLITE_HOST_API_PREFIX_CASE";

    unsafe extern "C" fn version() -> c_int {
        if std::env::var(CASE_ENV)
            .unwrap_or_default()
            .starts_with("338")
        {
            3_038_000
        } else {
            3_020_000
        }
    }

    unsafe extern "C" fn presence_only() {
        // Presence-only placeholders must never be invoked by initialization.
        std::process::abort();
    }

    #[allow(clippy::missing_transmute_annotations)]
    fn fixture(case: &str) -> sqlite3_api_routines {
        // SAFETY: every field of sqlite3_api_routines is an optional function
        // pointer; zero is the valid None representation for all fields.
        let mut table: sqlite3_api_routines = unsafe { std::mem::zeroed() };
        table.libversion_number = Some(version);
        macro_rules! present {
            ($($field:ident),+ $(,)?) => {$(
                // SAFETY: initialization only tests these function pointers for
                // presence, never calls them. The executable non-null address
                // is converted to the field's inferred function-pointer type.
                // Only libversion_number is dispatched, with its true signature.
                table.$field = Some(unsafe {
                    std::mem::transmute::<unsafe extern "C" fn(), _>(presence_only as unsafe extern "C" fn())
                });
            )+};
        }
        present!(
            value_type,
            value_bytes,
            value_blob,
            value_text,
            value_int64,
            value_pointer,
            result_double,
            result_null,
            result_blob,
            result_text,
            result_error,
            result_pointer,
            malloc,
            free,
            declare_vtab,
            vtab_config,
            create_module_v2,
            create_function_v2,
            prepare_v2,
            finalize,
            bind_int64,
            bind_blob,
            bind_text,
            bind_pointer,
            context_db_handle,
            result_error_code,
            step,
            column_count,
            column_type,
            column_int64,
            column_blob,
            column_bytes,
            column_text,
            errmsg,
            extended_errcode,
            randomness,
            changes,
            vtab_in,
            vtab_in_first,
            vtab_in_next,
        );
        match case {
            "320-missing-prepare" => table.prepare_v2 = None,
            "320-missing-extended" => table.extended_errcode = None,
            "320-missing-randomness" => table.randomness = None,
            "320-missing-column" => table.column_blob = None,
            "320-missing-bind-pointer" => table.bind_pointer = None,
            "320-missing-context-db" => table.context_db_handle = None,
            "320-missing-result-code" => table.result_error_code = None,
            "338-missing-in" => table.vtab_in_next = None,
            _ => {}
        }
        table
    }

    // Put the usable prefix immediately before an unreadable page. This catches
    // whole-table reference/copy regressions and accidental reads of newer fields.
    struct GuardPrefix {
        allocation: *mut c_void,
        length: usize,
        table: *const sqlite3_api_routines,
    }

    impl GuardPrefix {
        fn new(source: &sqlite3_api_routines, prefix: usize) -> Self {
            let (allocation, page) = pages::allocate();
            assert!(prefix <= page);
            assert_eq!(prefix % std::mem::align_of::<sqlite3_api_routines>(), 0);
            // SAFETY: allocate returns two pages, with the second inaccessible.
            // The prefix fits in the readable first page and is pointer-aligned.
            // Copy only the initialized source bytes that the fixture host owns.
            let table = unsafe {
                let destination = allocation.cast::<u8>().add(page - prefix);
                std::ptr::copy_nonoverlapping(
                    (source as *const sqlite3_api_routines).cast::<u8>(),
                    destination,
                    prefix,
                );
                destination.cast::<sqlite3_api_routines>()
            };
            Self {
                allocation,
                length: page * 2,
                table,
            }
        }
    }

    impl Drop for GuardPrefix {
        fn drop(&mut self) {
            // SAFETY: this owns the exact two-page allocation returned by pages;
            // invalid initialization never retained the table in the global API.
            unsafe { pages::release(self.allocation, self.length) };
        }
    }

    #[cfg(unix)]
    mod pages {
        use super::*;
        extern "C" {
            fn getpagesize() -> c_int;
            fn mmap(
                address: *mut c_void,
                length: usize,
                protection: c_int,
                flags: c_int,
                fd: c_int,
                offset: i64,
            ) -> *mut c_void;
            fn mprotect(address: *mut c_void, length: usize, protection: c_int) -> c_int;
            fn munmap(address: *mut c_void, length: usize) -> c_int;
        }

        pub(super) fn allocate() -> (*mut c_void, usize) {
            // SAFETY: getpagesize takes no arguments; mmap requests two private
            // anonymous writable pages without using a file or existing mapping.
            let (address, page) = unsafe {
                let page = usize::try_from(getpagesize()).unwrap();
                let anonymous = if cfg!(target_os = "macos") {
                    0x1000
                } else {
                    0x20
                };
                (
                    mmap(std::ptr::null_mut(), page * 2, 3, 2 | anonymous, -1, 0),
                    page,
                )
            };
            assert_ne!(address as isize, -1, "mmap failed");
            // SAFETY: mmap returned two pages; the second page's protection is
            // removed, leaving the prefix's backing page writable and readable.
            let status = unsafe { mprotect(address.cast::<u8>().add(page).cast(), page, 0) };
            assert_eq!(status, 0, "mprotect failed");
            (address, page)
        }

        pub(super) unsafe fn release(address: *mut c_void, length: usize) {
            // SAFETY: caller supplies this module's live allocation and exact size.
            unsafe { munmap(address, length) };
        }
    }

    #[cfg(windows)]
    mod pages {
        use super::*;
        #[link(name = "kernel32")]
        extern "system" {
            #[link_name = "VirtualAlloc"]
            fn virtual_alloc(
                address: *mut c_void,
                length: usize,
                allocation: u32,
                protection: u32,
            ) -> *mut c_void;
            #[link_name = "VirtualProtect"]
            fn virtual_protect(
                address: *mut c_void,
                length: usize,
                protection: u32,
                old: *mut u32,
            ) -> c_int;
            #[link_name = "VirtualFree"]
            fn virtual_free(address: *mut c_void, length: usize, operation: u32) -> c_int;
        }

        pub(super) fn allocate() -> (*mut c_void, usize) {
            // Windows x64's 64-KiB allocation granularity is also page-aligned.
            let page = 65_536;
            // SAFETY: request fresh committed/reserved writable memory; no
            // existing address or foreign allocation is modified.
            let address = unsafe { virtual_alloc(std::ptr::null_mut(), page * 2, 0x3000, 0x04) };
            assert!(!address.is_null(), "VirtualAlloc failed");
            let mut old = 0;
            // SAFETY: protect the allocation's second aligned 64-KiB region;
            // old is a writable output and the first region stays accessible.
            let status = unsafe {
                virtual_protect(address.cast::<u8>().add(page).cast(), page, 0x01, &mut old)
            };
            assert_ne!(status, 0, "VirtualProtect failed");
            (address, page)
        }

        pub(super) unsafe fn release(address: *mut c_void, _length: usize) {
            // SAFETY: caller supplies the owned base; MEM_RELEASE requires size0.
            unsafe { virtual_free(address, 0, 0x8000) };
        }
    }

    #[test]
    fn host_api_prefix_subprocess() {
        if let Ok(case) = std::env::var(CASE_ENV) {
            let source = fixture(&case);
            let prefix = if case.starts_with("338") {
                std::mem::offset_of!(sqlite3_api_routines, deserialize)
            } else {
                std::mem::offset_of!(sqlite3_api_routines, vtab_nochange)
            };
            let memory = GuardPrefix::new(&source, prefix);
            // SAFETY: the fixture owns this host-version prefix, with a genuine
            // libversion_number entry. It is never borrowed as a complete table.
            // A successful installation below retains the mapping until exit.
            let result = unsafe { set_api(memory.table) };
            let expected = match case.as_str() {
                "320-missing-prepare" => Some("prepare_v2"),
                "320-missing-extended" => Some("extended_errcode"),
                "320-missing-randomness" => Some("randomness"),
                "320-missing-column" => Some("column_blob"),
                "320-missing-bind-pointer" => Some("bind_pointer"),
                "320-missing-context-db" => Some("context_db_handle"),
                "320-missing-result-code" => Some("result_error_code"),
                "338-missing-in" => Some("vtab_in_next"),
                _ => None,
            };
            if let Some(expected) = expected {
                assert!(result.unwrap_err().contains(expected));
            } else {
                assert_eq!(result, Ok(()));
                // The global API keeps this borrowed prefix; preserve it for
                // this isolated subprocess's lifetime, without invoking stubs.
                std::mem::forget(memory);
            }
            return;
        }
        for case in [
            "320",
            "338",
            "320-missing-prepare",
            "320-missing-extended",
            "320-missing-randomness",
            "320-missing-column",
            "320-missing-bind-pointer",
            "320-missing-context-db",
            "320-missing-result-code",
            "338-missing-in",
        ] {
            let output = Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "ffi::tests::host_api_prefix_subprocess",
                    "--nocapture",
                ])
                .env(CASE_ENV, case)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "host prefix case {case} failed: {}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
        }
    }
}
