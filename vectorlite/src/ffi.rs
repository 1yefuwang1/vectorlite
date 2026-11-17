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
        reset,
        clear_bindings,
        bind_int64,
        bind_blob,
        bind_text,
        bind_pointer,
        context_db_handle,
        user_data,
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
        stmt_status
    );
    if version >= 3038000 {
        require!(total_changes64, vtab_in, vtab_in_first, vtab_in_next);
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

/// Routes `sqlite3_prepare_v2` and counts each dispatch attempt and successful
/// preparation separately in Debug builds.
///
/// # Safety
/// Arguments must satisfy `sqlite3_prepare_v2`'s connection, SQL, output
/// statement and output-tail contracts. The API must be initialized.
pub unsafe fn prepare_v2(
    db: *mut sqlite3,
    sql: *const c_char,
    length: c_int,
    statement: *mut *mut sqlite3_stmt,
    tail: *mut *const c_char,
) -> c_int {
    #[cfg(debug_assertions)]
    instrumentation::note_prepare_attempt();
    // SAFETY: the caller supplies this routine's SQLite argument contract;
    // set_api validated this entry in the supported prefix.
    let status = unsafe { (api_field!(prepare_v2))(db, sql, length, statement, tail) };
    #[cfg(debug_assertions)]
    instrumentation::note_prepare(status);
    status
}
/// Routes `sqlite3_finalize` and counts every statement-boundary finalization in Debug builds.
///
/// # Safety
/// `statement` must be null or the unique statement being finalized on its live
/// connection's callback thread; no column view may outlive this call.
pub unsafe fn finalize(statement: *mut sqlite3_stmt) -> c_int {
    // SAFETY: the caller transfers the unique statement and its finalizing lease;
    // set_api validated this entry in the supported prefix.
    let status = unsafe { (api_field!(finalize))(statement) };
    #[cfg(debug_assertions)]
    instrumentation::note_finalize();
    status
}
statement_api!(reset(statement: *mut sqlite3_stmt) -> c_int);
statement_api!(clear_bindings(statement: *mut sqlite3_stmt) -> c_int);
statement_api!(bind_int64(statement: *mut sqlite3_stmt, index: c_int, value: i64) -> c_int);
statement_api!(
    bind_blob(
        statement: *mut sqlite3_stmt,
        index: c_int,
        value: *const c_void,
        length: c_int,
        destructor: sqlite3_destructor_type,
    ) -> c_int
);
statement_api!(
    bind_text(
        statement: *mut sqlite3_stmt,
        index: c_int,
        value: *const c_char,
        length: c_int,
        destructor: sqlite3_destructor_type,
    ) -> c_int
);
statement_api!(
    bind_pointer(
        statement: *mut sqlite3_stmt,
        index: c_int,
        value: *mut c_void,
        tag: *const c_char,
        destructor: sqlite3_destructor_type,
    ) -> c_int
);
statement_api!(context_db_handle(context: *mut sqlite3_context) -> *mut sqlite3);
statement_api!(user_data(context: *mut sqlite3_context) -> *mut c_void);
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
statement_api!(stmt_status(statement: *mut sqlite3_stmt, op: c_int, reset: c_int) -> c_int);

// --- debug-only statement lifecycle instrumentation ---

/// Statement prepare/finalize counters for Debug-only native test hosts.
///
/// The counters are incremented at the two `Statement` lifecycle boundaries:
/// once for every `sqlite3_prepare_v2` dispatch that reported success, and once
/// for every `sqlite3_finalize` dispatch (explicit `finish` or `Drop`). A query
/// that retains readers must therefore report `prepares == finalizes` once every
/// statement is finalized. A separate attempt counter also counts failed prepare
/// dispatches, so forced-failure paths can be told apart from successful work.
/// Release builds compile this module out entirely, so no `vectorlite_test_`
/// symbol is present in release artifacts.
#[cfg(debug_assertions)]
pub(crate) mod instrumentation {
    use std::os::raw::c_int;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Successful `sqlite3_prepare_v2` dispatches since the last reset.
    static PREPARES: AtomicU64 = AtomicU64::new(0);
    /// Total `sqlite3_prepare_v2` dispatch attempts since the last reset,
    /// successful or failed.
    static PREPARE_ATTEMPTS: AtomicU64 = AtomicU64::new(0);
    /// `sqlite3_finalize` dispatches since the last reset.
    static FINALIZES: AtomicU64 = AtomicU64::new(0);

    /// Counts one prepare dispatch attempt before SQLite reports its status, so a
    /// forced or failing preparation appears here but not in `PREPARES`.
    pub(crate) fn note_prepare_attempt() {
        PREPARE_ATTEMPTS.fetch_add(1, Ordering::Relaxed);
    }

    /// Counts one preparation when SQLite reported success for it.
    pub(crate) fn note_prepare(status: c_int) {
        if status == super::SQLITE_OK as c_int {
            PREPARES.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Counts one statement finalization.
    pub(crate) fn note_finalize() {
        FINALIZES.fetch_add(1, Ordering::Relaxed);
    }

    /// Resets all Debug-only statement counters to zero.
    ///
    /// Exported only by Debug builds so a native test host can measure one query.
    #[no_mangle]
    pub extern "C" fn vectorlite_test_reset_counters() {
        PREPARES.store(0, Ordering::Relaxed);
        PREPARE_ATTEMPTS.store(0, Ordering::Relaxed);
        FINALIZES.store(0, Ordering::Relaxed);
    }

    /// Writes `[prepares, finalizes]` into `out` for a Debug-only native test host.
    ///
    /// # Safety
    /// `out` must be null or point to two writable, properly aligned `u64`
    /// slots that remain valid for this call and are not concurrently written
    /// through another alias.
    #[no_mangle]
    pub unsafe extern "C" fn vectorlite_test_read_counters(out: *mut u64) {
        if out.is_null() {
            return;
        }
        // SAFETY: the caller guarantees two writable slots at `out`; both are
        // written synchronously and the pointer is not retained.
        unsafe {
            out.write(PREPARES.load(Ordering::Relaxed));
            out.add(1).write(FINALIZES.load(Ordering::Relaxed));
        }
    }

    /// Reports the monotonic count of `sqlite3_prepare_v2` dispatch attempts.
    ///
    /// Unlike the successful-preparation count reported by
    /// `vectorlite_test_read_counters`, this includes attempts that SQLite (or a
    /// Debug-only forced reader-prepare failure) rejected, so a host can separate
    /// attempted work from prepared work. Debug builds only; release artifacts
    /// export no instrumentation.
    #[no_mangle]
    pub extern "C" fn vectorlite_test_prepare_attempts() -> u64 {
        PREPARE_ATTEMPTS.load(Ordering::Relaxed)
    }
}

// --- optional transaction-free DiskANN build capabilities ---

/// A validated capability set, not a borrow of the host's possibly short table.
/// Keep this separate from set_api: missing build APIs must not raise the HNSW
/// minimum or prevent HNSW from loading on SQLite 3.20.
#[derive(Clone, Copy)]
pub(crate) struct BuildStateApi {
    get_autocommit: unsafe extern "C" fn(*mut sqlite3) -> c_int,
    txn_state: unsafe extern "C" fn(*mut sqlite3, *const c_char) -> c_int,
    next_stmt: unsafe extern "C" fn(*mut sqlite3, *mut sqlite3_stmt) -> *mut sqlite3_stmt,
    stmt_busy: unsafe extern "C" fn(*mut sqlite3_stmt) -> c_int,
}

impl BuildStateApi {
    /// Validates only fields present in a DiskANN-supported host prefix.
    ///
    /// # Safety
    /// table must be readable during validation, with the prefix appropriate
    /// to its reported version. The function pointers and originating runtime
    /// must remain live through all calls; the returned value does not borrow table.
    pub(crate) unsafe fn from_table(
        table: *const sqlite3_api_routines,
    ) -> Result<Self, &'static str> {
        if table.is_null() {
            return Err("SQLite API table is null");
        }
        // SAFETY: libversion_number is in the original loadable-extension prefix;
        // reading this field does not borrow or copy the newer whole table.
        let version = unsafe { std::ptr::addr_of!((*table).libversion_number).read() }
            .ok_or("SQLite version API is unavailable")?;
        // SAFETY: this is the host's correctly typed, live version function.
        if unsafe { version() } < 3_038_000 {
            return Err("DiskANN build requires SQLite 3.38.0 or newer");
        }
        macro_rules! required {
            ($field:ident) => {
                // SAFETY: each field below belongs to the checked 3.38 prefix.
                // In particular txn_state was added in 3.34; it is NEVER read
                // from an older table, even if other older fields are present.
                unsafe { std::ptr::addr_of!((*table).$field).read() }.ok_or(concat!(
                    "SQLite build API is unavailable: ",
                    stringify!($field)
                ))?
            };
        }
        Ok(Self {
            get_autocommit: required!(get_autocommit),
            txn_state: required!(txn_state),
            next_stmt: required!(next_stmt),
            stmt_busy: required!(stmt_busy),
        })
    }

    /// # Safety
    /// db must be live and accessed only in its serialized callback on this thread.
    pub(crate) unsafe fn get_autocommit(self, db: *mut sqlite3) -> c_int {
        // SAFETY: caller supplies the live callback-confined connection.
        unsafe { (self.get_autocommit)(db) }
    }

    /// Returns the maximum transaction state across ALL schemas, including TEMP
    /// and attached databases. Unknown/negative results must not be treated as NONE.
    ///
    /// # Safety
    /// db must be live and accessed only in its serialized callback on this thread.
    pub(crate) unsafe fn txn_state(self, db: *mut sqlite3) -> c_int {
        // SAFETY: NULL is SQLite's documented all-schemas selector.
        unsafe { (self.txn_state)(db, std::ptr::null()) }
    }

    /// # Safety
    /// db must be a live callback-confined connection. previous is NULL or a
    /// live statement of db; no statement may be finalized during enumeration.
    pub(crate) unsafe fn next_stmt(
        self,
        db: *mut sqlite3,
        previous: *mut sqlite3_stmt,
    ) -> *mut sqlite3_stmt {
        // SAFETY: the caller maintains the enumeration's connection/statement lease.
        unsafe { (self.next_stmt)(db, previous) }
    }

    /// # Safety
    /// statement must be live on its connection's serialized callback thread.
    pub(crate) unsafe fn stmt_busy(self, statement: *mut sqlite3_stmt) -> c_int {
        // SAFETY: the caller maintains the statement lease through this call.
        unsafe { (self.stmt_busy)(statement) }
    }
}

pub(crate) fn build_state_api() -> Result<BuildStateApi, &'static str> {
    let table = API.load(Ordering::Acquire);
    // SAFETY: set_api stores only a live host table whose lifetime covers every
    // callback; from_table also rejects NULL before inspecting any field.
    unsafe { BuildStateApi::from_table(table) }
}

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

/// Registers owned scalar userdata. SQLite invokes destroy on replacement, close,
/// and registration failure; callers transfer one unique allocation even on error.
///
/// # Safety
/// The live database/name/callback/pApp and destructor must satisfy SQLite's
/// create_function_v2 lifetime and single-destruction contract.
#[allow(clippy::too_many_arguments)]
pub unsafe fn create_owned_function(
    db: *mut sqlite3,
    name: *const c_char,
    n_arg: c_int,
    flags: c_int,
    p_app: *mut c_void,
    x_func: Option<unsafe extern "C" fn(*mut sqlite3_context, c_int, *mut *mut sqlite3_value)>,
    destroy: Option<unsafe extern "C" fn(*mut c_void)>,
) -> c_int {
    // SAFETY: caller transfers owned pApp with its matching destructor and valid callback.
    unsafe {
        (api_field!(create_function_v2))(db, name, n_arg, flags, p_app, x_func, None, None, destroy)
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
        } else if std::env::var(CASE_ENV)
            .unwrap_or_default()
            .starts_with("334")
        {
            3_034_000
        } else if std::env::var(CASE_ENV)
            .unwrap_or_default()
            .starts_with("337")
        {
            3_037_000
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
            reset,
            clear_bindings,
            bind_int64,
            bind_blob,
            bind_text,
            bind_pointer,
            context_db_handle,
            user_data,
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
            stmt_status,
            total_changes64,
            get_autocommit,
            txn_state,
            next_stmt,
            stmt_busy,
            vtab_in,
            vtab_in_first,
            vtab_in_next,
        );
        match case {
            "320-missing-prepare" => table.prepare_v2 = None,
            "320-missing-reset" => table.reset = None,
            "320-missing-clear-bindings" => table.clear_bindings = None,
            "320-missing-extended" => table.extended_errcode = None,
            "320-missing-randomness" => table.randomness = None,
            "320-missing-column" => table.column_blob = None,
            "320-missing-bind-pointer" => table.bind_pointer = None,
            "320-missing-context-db" => table.context_db_handle = None,
            "320-missing-result-code" => table.result_error_code = None,
            "320-missing-stmt-status" => table.stmt_status = None,
            "338-missing-total-changes64" => table.total_changes64 = None,
            "338-missing-in" => table.vtab_in_next = None,
            "338-missing-build-autocommit" => table.get_autocommit = None,
            "338-missing-build-txn-state" => table.txn_state = None,
            "338-missing-build-next-stmt" => table.next_stmt = None,
            "338-missing-build-stmt-busy" => table.stmt_busy = None,
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
    fn build_capability_null_and_missing_version_fail_closed() {
        // SAFETY: the validator rejects NULL before any field access.
        let null = unsafe { BuildStateApi::from_table(std::ptr::null()) };
        assert!(null.err().unwrap().contains("table is null"));
        // SAFETY: all fields are optional function pointers, so zero means None.
        let missing: sqlite3_api_routines = unsafe { std::mem::zeroed() };
        // SAFETY: this fully initialized table is readable during validation.
        let version = unsafe { BuildStateApi::from_table(&missing) };
        assert!(version.err().unwrap().contains("version API"));
    }

    #[test]
    fn host_api_prefix_subprocess() {
        if let Ok(case) = std::env::var(CASE_ENV) {
            let source = fixture(&case);
            let prefix = if case.starts_with("338") {
                std::mem::offset_of!(sqlite3_api_routines, deserialize)
            } else if case.starts_with("334") {
                std::mem::offset_of!(sqlite3_api_routines, changes64)
            } else if case.starts_with("337") {
                std::mem::offset_of!(sqlite3_api_routines, error_offset)
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
                "320-missing-reset" => Some("reset"),
                "320-missing-clear-bindings" => Some("clear_bindings"),
                "320-missing-extended" => Some("extended_errcode"),
                "320-missing-randomness" => Some("randomness"),
                "320-missing-column" => Some("column_blob"),
                "320-missing-bind-pointer" => Some("bind_pointer"),
                "320-missing-context-db" => Some("context_db_handle"),
                "320-missing-result-code" => Some("result_error_code"),
                "320-missing-stmt-status" => Some("stmt_status"),
                "338-missing-total-changes64" => Some("total_changes64"),
                "338-missing-in" => Some("vtab_in_next"),
                _ => None,
            };
            if let Some(expected) = expected {
                assert!(result.unwrap_err().contains(expected));
            } else {
                assert_eq!(result, Ok(()));
                // SAFETY: the mapping owns exactly the reported host prefix. The
                // validator must reject older hosts BEFORE reading txn_state.
                let build = unsafe { BuildStateApi::from_table(memory.table) };
                if !case.starts_with("338") {
                    assert!(build.err().unwrap().contains("3.38.0"));
                } else {
                    let missing = match case.as_str() {
                        "338-missing-build-autocommit" => Some("get_autocommit"),
                        "338-missing-build-txn-state" => Some("txn_state"),
                        "338-missing-build-next-stmt" => Some("next_stmt"),
                        "338-missing-build-stmt-busy" => Some("stmt_busy"),
                        _ => None,
                    };
                    if let Some(missing) = missing {
                        assert!(build.err().unwrap().contains(missing));
                        assert!(build_state_api().err().unwrap().contains(missing));
                    } else {
                        assert!(build.is_ok());
                        assert!(build_state_api().is_ok());
                    }
                }
                // The global API keeps this borrowed prefix; preserve it for
                // this isolated subprocess's lifetime, without invoking stubs.
                std::mem::forget(memory);
            }
            return;
        }
        for case in [
            "320",
            "334",
            "337",
            "338",
            "338-missing-build-autocommit",
            "338-missing-build-txn-state",
            "338-missing-build-next-stmt",
            "338-missing-build-stmt-busy",
            "320-missing-prepare",
            "320-missing-reset",
            "320-missing-clear-bindings",
            "320-missing-extended",
            "320-missing-randomness",
            "320-missing-column",
            "320-missing-bind-pointer",
            "320-missing-context-db",
            "320-missing-result-code",
            "320-missing-stmt-status",
            "338-missing-total-changes64",
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
