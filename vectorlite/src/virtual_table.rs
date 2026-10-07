//! SQLite callbacks for vectorlite. Callbacks validate the host's arguments and
//! delegate policy to safe helpers; numeric work lives in `crate::core`.

#![deny(unsafe_op_in_unsafe_fn)]

use std::cell::{Cell, UnsafeCell};
use std::collections::HashSet;
use std::ffi::CStr;
use std::marker::PhantomData;
use std::os::raw::{c_char, c_int, c_void};
use std::rc::Rc;

use crate::core::{Index, SearchFilter, SearchResult};
use crate::diskann_store::{DiskAnnTable, QueryRows};
use crate::ffi::{
    self, sqlite3, sqlite3_context, sqlite3_index_info, sqlite3_module, sqlite3_value,
    sqlite3_vtab, sqlite3_vtab_cursor, Value,
};
use crate::index_error::IndexError;
use crate::index_options::BackendOptions;
use crate::registry::{IndexEntry, Registry, RegistryKey};
use crate::scalar::{self, KnnParam, KNN_PARAM_TYPE};
use crate::sqlite::{Connection, Statement};
use crate::vector;
use crate::vector_space::{parse_named_vector_space, NamedVectorSpace};

const COL_VECTOR: c_int = 0;
const COL_DISTANCE: c_int = 1;
const COL_OPERATION: c_int = 2;
const COL_PATH: c_int = 3;
const FUNC_KNN: c_int = ffi::SQLITE_INDEX_CONSTRAINT_FUNCTION as c_int;
const SQLITE_VTAB_DIRECTONLY_MIN_VERSION: c_int = 3_031_000;
// The low bits remain SQLite's argument count, preserving the checked plan.
const PLAN_VECTOR_OUTPUT: c_int = 1 << 30;

/// SQLite owns the header and can write it while Rust holds shared references
/// to the table state. `UnsafeCell` explicitly permits those writes. `repr(C)`
/// and the first-field layout make a VTab pointer a valid sqlite3_vtab pointer.
/// Callbacks are serialized by the connection; the registry outlives every VTab.
#[repr(C)]
pub struct VTab {
    base: UnsafeCell<sqlite3_vtab>,
    registry: *mut Registry,
    schema: String,
    table: String,
    // Retaining the entry avoids registry lookups and key allocations in row
    // callbacks, while the registry retains it across xDisconnect/xConnect.
    backend: Backend,
}

/// HNSW retains its registry-owned graph. DiskANN owns only host-connection
/// handles; its authoritative data survives independently in shadow tables.
enum Backend {
    Hnsw(Rc<IndexEntry>),
    Diskann(Box<DiskAnnTable>),
}

thread_local! {
    static DISKANN_CALLBACK_ACTIVE: Cell<bool> = const { Cell::new(false) };
}

/// Keeps exact-scan and graph paths under the same callback reentrancy rule.
/// The marker prevents moving cleanup to a different thread's TLS instance.
struct DiskannCallbackGuard(PhantomData<Rc<()>>);

impl DiskannCallbackGuard {
    fn enter() -> Result<Self, VTabError> {
        DISKANN_CALLBACK_ACTIVE.with(|active| {
            if active.replace(true) {
                return Err(VTabError::with_code(
                    ffi::SQLITE_LOCKED as c_int,
                    "recursive DiskANN callbacks are not supported",
                ));
            }
            Ok(Self(PhantomData))
        })
    }

    fn for_backend(backend: &Backend) -> Result<Option<Self>, VTabError> {
        if matches!(backend, Backend::Diskann(_)) {
            Self::enter().map(Some)
        } else {
            Ok(None)
        }
    }
}

impl Drop for DiskannCallbackGuard {
    fn drop(&mut self) {
        DISKANN_CALLBACK_ACTIVE.with(|active| active.set(false));
    }
}

impl Backend {
    fn space(&self) -> &NamedVectorSpace {
        match self {
            Self::Hnsw(entry) => &entry.space,
            Self::Diskann(table) => table.space(),
        }
    }

    fn contains(&self, rowid: u64) -> Result<bool, VTabError> {
        match self {
            Self::Hnsw(entry) => Ok(entry.index.contains(rowid)),
            Self::Diskann(table) => table.contains(rowid).map_err(Into::into),
        }
    }

    fn mark_delete(&self, rowid: u64) -> Result<(), VTabError> {
        match self {
            Self::Hnsw(entry) => entry.index.mark_delete(rowid).map_err(VTabError::new),
            Self::Diskann(table) => table.mark_delete(rowid).map_err(Into::into),
        }
    }
}

#[repr(C)]
pub struct Cursor {
    base: sqlite3_vtab_cursor,
    result: Vec<SearchResult>,
    current: usize,
    disk_vectors: Option<Vec<Vec<u8>>>,
    snapshot_anchor: Option<Statement<'static>>,
}

#[derive(Debug)]
struct VTabError {
    code: c_int,
    message: String,
}

impl VTabError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            code: ffi::SQLITE_ERROR as c_int,
            message: message.into(),
        }
    }

    fn with_code(code: c_int, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl From<IndexError> for VTabError {
    fn from(error: IndexError) -> Self {
        Self::with_code(error.code, error.message)
    }
}

impl From<&str> for VTabError {
    fn from(message: &str) -> Self {
        Self::new(message)
    }
}

impl VTab {
    fn key(&self) -> RegistryKey {
        (self.schema.clone(), self.table.clone())
    }

    fn report(&self, error: VTabError) -> c_int {
        // SAFETY: only the externally mutable header is accessed. SQLite
        // serializes callbacks, owns the existing error allocation, and accepts
        // a replacement allocated by its own allocator. No header reference is
        // kept across this call, and the Rust state is outside the UnsafeCell.
        unsafe {
            ffi::set_err(
                std::ptr::addr_of_mut!((*self.base.get()).zErrMsg),
                &error.message,
            );
        }
        error.code
    }
}

impl Drop for VTab {
    fn drop(&mut self) {
        let header = self.base.get_mut();
        // SAFETY: disconnect/destroy return ownership of this table to us;
        // zErrMsg is either null or a live allocation from SQLite's allocator.
        unsafe { ffi::sqlite_free(header.zErrMsg.cast()) };
        header.zErrMsg = std::ptr::null_mut();
    }
}

/// # Safety
/// `p` is a NUL-terminated string supplied for the current SQLite callback and
/// remains valid for the returned borrow.
unsafe fn cstr<'a>(p: *const c_char) -> &'a str {
    // SAFETY: guaranteed by this helper's caller.
    unsafe { CStr::from_ptr(p) }.to_str().unwrap_or("")
}

// ---- create / connect ----

struct TableDefinition {
    space: NamedVectorSpace,
    options: BackendOptions,
    vector_space_str: String,
    index_options_str: String,
}

impl TableDefinition {
    fn parse(vector_space_str: String, index_options_str: String) -> Result<Self, VTabError> {
        let space = parse_named_vector_space(&vector_space_str).map_err(|error| {
            VTabError::new(format!(
                "Invalid vector space: {vector_space_str}. Reason: {error}"
            ))
        })?;
        let options = BackendOptions::parse(&index_options_str).map_err(|error| {
            VTabError::new(format!(
                "Invalid index_options {index_options_str}. Reason: {error}"
            ))
        })?;
        Ok(Self {
            space,
            options,
            vector_space_str,
            index_options_str,
        })
    }
}

fn find_or_create_entry(
    registry: &mut Registry,
    is_create: bool,
    key: RegistryKey,
    definition: TableDefinition,
) -> Result<Rc<IndexEntry>, VTabError> {
    let TableDefinition {
        space,
        options,
        vector_space_str,
        index_options_str,
    } = definition;
    let BackendOptions::Hnsw(options) = options else {
        return Err(VTabError::new("HNSW registry cannot own a DiskANN table"));
    };
    if !is_create {
        if let Some(existing) = registry.find(&key) {
            if existing.vector_space_str == vector_space_str
                && existing.index_options_str == index_options_str
            {
                return Ok(existing);
            }
        }
    }
    let index = Index::create(
        space.dim,
        space.distance_type,
        space.vector_type,
        options.max_elements,
        options.m,
        options.ef_construction,
        options.random_seed,
        options.allow_replace_deleted,
    )
    .map_err(|error| VTabError::new(format!("Failed to create virtual table: {error}")))?;
    Ok(registry.insert(
        key,
        IndexEntry {
            index,
            space,
            vector_space_str,
            index_options_str,
        },
    ))
}

#[allow(clippy::too_many_arguments)]
unsafe fn init_vtab(
    is_create: bool,
    db: *mut sqlite3,
    p_aux: *mut c_void,
    argc: c_int,
    argv: *const *const c_char,
    pp_vtab: *mut *mut sqlite3_vtab,
    pz_err: *mut *mut c_char,
) -> c_int {
    // DIRECTONLY prevents a database schema from invoking the save/load command
    // channel through a trigger or view. Older SQLite hosts do not know this op.
    // SAFETY: extension initialization validated this host API entry.
    if unsafe { ffi::libversion_number() } >= SQLITE_VTAB_DIRECTONLY_MIN_VERSION {
        // SAFETY: called during xCreate/xConnect with SQLite's live connection.
        let rc = unsafe { ffi::vtab_config_directonly(db) };
        if rc != ffi::SQLITE_OK as c_int {
            return rc;
        }
    }
    const MODULE_PARAM_OFFSET: c_int = 3;
    if argc != 2 + MODULE_PARAM_OFFSET {
        let message = format!(
            "vectorlite expects 2 arguments (a vector space and index options), got {}. \
             The index file path argument has been removed; use INSERT INTO \
             <table>(operation, path) VALUES('save', <path>) to persist an index and \
             INSERT INTO <table>(operation, path) VALUES('load', <path>) to restore one.",
            argc.saturating_sub(MODULE_PARAM_OFFSET)
        );
        // SAFETY: SQLite supplies the writable constructor error output.
        unsafe { ffi::set_err(pz_err, &message) };
        return ffi::SQLITE_ERROR as c_int;
    }
    // SAFETY: argc was checked; SQLite supplies argc valid C string pointers.
    let (schema, table, vector_space_str, index_options_str) = unsafe {
        (
            cstr(*argv.add(1)).to_owned(),
            cstr(*argv.add(2)).to_owned(),
            cstr(*argv.add(3)).to_owned(),
            cstr(*argv.add(4)).to_owned(),
        )
    };
    let definition = match TableDefinition::parse(vector_space_str, index_options_str) {
        Ok(definition) => definition,
        Err(error) => {
            // SAFETY: pz_err is SQLite's writable error output.
            unsafe { ffi::set_err(pz_err, &error.message) };
            return error.code;
        }
    };
    let is_diskann = matches!(definition.options, BackendOptions::Diskann(_));
    // SAFETY: constructor callbacks supply a live connection. DiskANN's many
    // shadow writes need ABORT handling, not a promise of pre-write constraints.
    let rc = unsafe {
        if is_diskann {
            ffi::vtab_config_constraint_support_disabled(db)
        } else {
            ffi::vtab_config_constraint_support(db)
        }
    };
    if rc != ffi::SQLITE_OK as c_int {
        return rc;
    }
    // SAFETY: the initialized host API provides its actual version.
    if is_diskann && unsafe { ffi::libversion_number() } < 3038000 {
        // SAFETY: SQLite supplies the constructor's writable error output.
        unsafe { ffi::set_err(pz_err, "DiskANN requires SQLite 3.38.0 or newer") };
        return ffi::SQLITE_ERROR as c_int;
    }
    let declare_sql = format!(
        "CREATE TABLE X({}, distance REAL hidden, operation TEXT hidden, path TEXT hidden)",
        definition.space.vector_name
    );
    // SAFETY: called from the constructor with its live connection handle.
    // Declare before changing the registry so a failed schema leaves it intact.
    let rc = unsafe { ffi::declare_vtab(db, &declare_sql) };
    if rc != ffi::SQLITE_OK as c_int {
        return rc;
    }
    let registry = p_aux.cast::<Registry>();
    let backend = match &definition.options {
        BackendOptions::Hnsw(_) => {
            // SAFETY: pAux outlives the tables. This HNSW factory performs no
            // SQLite calls while borrowing the connection-local registry.
            find_or_create_entry(
                unsafe { &mut *registry },
                is_create,
                (schema.clone(), table.clone()),
                definition,
            )
            .map(Backend::Hnsw)
        }
        BackendOptions::Diskann(options) => {
            // SAFETY: SQLite owns this VTab and all derived cursors, closes them
            // before closing db, and serializes connection callbacks. The erased
            // lifetime stays private in these non-Send host-borrowing handles.
            let connection: Result<Connection<'static>, IndexError> =
                unsafe { Connection::borrow(db) };
            connection
                .and_then(|connection| {
                    DiskAnnTable::open(
                        connection,
                        &schema,
                        &table,
                        definition.space,
                        options.clone(),
                        is_create,
                    )
                })
                .map(|table| Backend::Diskann(Box::new(table)))
                .map_err(Into::into)
        }
    };
    let backend = match backend {
        Ok(backend) => backend,
        Err(error) => {
            // SAFETY: pz_err is SQLite's writable constructor error output.
            unsafe { ffi::set_err(pz_err, &error.message) };
            return error.code;
        }
    };
    let vtab = Box::new(VTab {
        base: UnsafeCell::new(sqlite3_vtab {
            pModule: std::ptr::null(),
            nRef: 0,
            zErrMsg: std::ptr::null_mut(),
        }),
        registry,
        schema,
        table,
        backend,
    });
    // SAFETY: SQLite supplies a writable output slot and takes ownership until
    // xDisconnect/xDestroy. VTab's first field has sqlite3_vtab's layout.
    unsafe { *pp_vtab = Box::into_raw(vtab).cast() };
    ffi::SQLITE_OK as c_int
}

unsafe extern "C" fn x_create(
    db: *mut sqlite3,
    p_aux: *mut c_void,
    argc: c_int,
    argv: *const *const c_char,
    pp_vtab: *mut *mut sqlite3_vtab,
    pz_err: *mut *mut c_char,
) -> c_int {
    // SAFETY: forwards SQLite's constructor callback contract unchanged.
    unsafe { init_vtab(true, db, p_aux, argc, argv, pp_vtab, pz_err) }
}

unsafe extern "C" fn x_connect(
    db: *mut sqlite3,
    p_aux: *mut c_void,
    argc: c_int,
    argv: *const *const c_char,
    pp_vtab: *mut *mut sqlite3_vtab,
    pz_err: *mut *mut c_char,
) -> c_int {
    // SAFETY: forwards SQLite's constructor callback contract unchanged.
    unsafe { init_vtab(false, db, p_aux, argc, argv, pp_vtab, pz_err) }
}

unsafe extern "C" fn x_disconnect(p_vtab: *mut sqlite3_vtab) -> c_int {
    // SAFETY: SQLite returns the live Box allocated by init_vtab exactly once,
    // after its cursors close; the registry retains the index for reconnect.
    drop(unsafe { Box::from_raw(p_vtab.cast::<VTab>()) });
    ffi::SQLITE_OK as c_int
}

unsafe extern "C" fn x_destroy(p_vtab: *mut sqlite3_vtab) -> c_int {
    // SAFETY: SQLite retains this live allocation if destruction fails.
    let vtab = unsafe { &*p_vtab.cast::<VTab>() };
    if let Backend::Diskann(table) = &vtab.backend {
        if let Err(error) = table.destroy() {
            return vtab.report(error.into());
        }
    }
    // SAFETY: successful destruction now transfers the allocation exactly once.
    let vtab = unsafe { Box::from_raw(p_vtab.cast::<VTab>()) };
    if matches!(vtab.backend, Backend::Hnsw(_)) {
        // SAFETY: pAux outlives the table. No SQLite call or registry borrow is
        // active; live HNSW handles retain independently owned entries.
        unsafe { &mut *vtab.registry }.erase(&vtab.key());
    }
    drop(vtab);
    ffi::SQLITE_OK as c_int
}

unsafe extern "C" fn x_rename(p_vtab: *mut sqlite3_vtab, z_new: *const c_char) -> c_int {
    // SAFETY: SQLite keeps this wrapper live through the rename callback.
    let vtab = unsafe { &*p_vtab.cast::<VTab>() };
    // SAFETY: z_new is a valid callback-scoped C string.
    let new_table = unsafe { cstr(z_new) }.to_owned();
    if let Backend::Diskann(table) = &vtab.backend {
        return match table.rename(&new_table) {
            Ok(()) => ffi::SQLITE_OK as c_int,
            Err(error) => vtab.report(error.into()),
        };
    }
    // SAFETY: only HNSW remains; its rename performs no SQLite calls, so no
    // reentrant callback can alias this exclusive registry/name mutation.
    let vtab = unsafe { &mut *p_vtab.cast::<VTab>() };
    let old_key = vtab.key();
    let new_key = (vtab.schema.clone(), new_table.clone());
    // SAFETY: the module registry remains live and no registry borrow is kept
    // in VTab (only its independently owned Rc entry).
    unsafe { &mut *vtab.registry }.rename(&old_key, new_key);
    vtab.table = new_table;
    ffi::SQLITE_OK as c_int
}

// ---- open / close / cursor stepping ----

unsafe extern "C" fn x_open(
    p_vtab: *mut sqlite3_vtab,
    pp_cursor: *mut *mut sqlite3_vtab_cursor,
) -> c_int {
    let cursor = Box::new(Cursor {
        base: sqlite3_vtab_cursor { pVtab: p_vtab },
        result: Vec::new(),
        current: 0,
        disk_vectors: None,
        snapshot_anchor: None,
    });
    // SAFETY: SQLite supplies the output slot and retains ownership until
    // xClose. Cursor is repr(C) with sqlite3_vtab_cursor as its first field.
    unsafe { *pp_cursor = Box::into_raw(cursor).cast() };
    ffi::SQLITE_OK as c_int
}

unsafe extern "C" fn x_close(p_cur: *mut sqlite3_vtab_cursor) -> c_int {
    // SAFETY: SQLite closes the allocation from xOpen exactly once.
    drop(unsafe { Box::from_raw(p_cur.cast::<Cursor>()) });
    ffi::SQLITE_OK as c_int
}

unsafe extern "C" fn x_eof(p_cur: *mut sqlite3_vtab_cursor) -> c_int {
    // SAFETY: this callback receives a live cursor allocated by xOpen.
    let cursor = unsafe { &*p_cur.cast::<Cursor>() };
    (cursor.current >= cursor.result.len()) as c_int
}

unsafe extern "C" fn x_next(p_cur: *mut sqlite3_vtab_cursor) -> c_int {
    // SAFETY: SQLite serializes access to this live cursor for the callback.
    let cursor = unsafe { &mut *p_cur.cast::<Cursor>() };
    if cursor.current < cursor.result.len() {
        cursor.current += 1;
    }
    ffi::SQLITE_OK as c_int
}

unsafe extern "C" fn x_rowid(p_cur: *mut sqlite3_vtab_cursor, p_rowid: *mut i64) -> c_int {
    // SAFETY: SQLite provides a live cursor and writable rowid output.
    let cursor = unsafe { &*p_cur.cast::<Cursor>() };
    if let Some(row) = cursor.result.get(cursor.current) {
        // SAFETY: p_rowid is writable for this callback.
        unsafe { *p_rowid = row.rowid as i64 };
        ffi::SQLITE_OK as c_int
    } else {
        ffi::SQLITE_ERROR as c_int
    }
}

unsafe extern "C" fn x_column(
    p_cur: *mut sqlite3_vtab_cursor,
    ctx: *mut sqlite3_context,
    n: c_int,
) -> c_int {
    // SAFETY: p_cur and its owning table remain live throughout the callback.
    let cursor = unsafe { &*p_cur.cast::<Cursor>() };
    let Some(row) = cursor.result.get(cursor.current) else {
        return ffi::SQLITE_ERROR as c_int;
    };
    match n {
        COL_DISTANCE => {
            // SAFETY: ctx is the callback's live result context.
            unsafe { ffi::result_double(ctx, row.distance as f64) };
        }
        COL_VECTOR => {
            // SAFETY: SQLite keeps the parent table alive until xClose.
            let vtab = unsafe { &*cursor.base.pVtab.cast::<VTab>() };
            if matches!(vtab.backend, Backend::Diskann(_)) {
                if let Some(bytes) = cursor
                    .disk_vectors
                    .as_ref()
                    .and_then(|vectors| vectors.get(cursor.current))
                {
                    // SAFETY: ctx is live and SQLite copies cursor-owned bytes;
                    // these bytes belong to the same snapshot as its distance.
                    unsafe { ffi::result_blob(ctx, bytes) };
                    return ffi::SQLITE_OK as c_int;
                }
                return vtab.report(VTabError::with_code(
                    ffi::SQLITE_MISUSE as c_int,
                    "DiskANN vector column was not included in its snapshot projection",
                ));
            }
            let Backend::Hnsw(entry) = &vtab.backend else {
                return ffi::SQLITE_MISUSE as c_int;
            };
            match entry.index.get_vector(row.rowid) {
                Some(v) => {
                    // SAFETY: SQLite copies these bytes before return.
                    unsafe { ffi::result_blob(ctx, &vector::blob_from_f32(&v)) };
                }
                None => {
                    // SAFETY: ctx is the callback's live result context.
                    unsafe {
                        ffi::result_error(
                            ctx,
                            &format!("Can't find vector with rowid {}", row.rowid),
                        )
                    };
                    return ffi::SQLITE_ERROR as c_int;
                }
            }
        }
        COL_OPERATION | COL_PATH => {
            // SAFETY: ctx is the callback's live result context.
            unsafe { ffi::result_null(ctx) };
        }
        _ => {
            // SAFETY: ctx is the callback's live result context.
            unsafe { ffi::result_error(ctx, &format!("Invalid column index: {n}")) };
            return ffi::SQLITE_ERROR as c_int;
        }
    }
    ffi::SQLITE_OK as c_int
}

// ---- best index ----

unsafe extern "C" fn x_best_index(
    p_vtab: *mut sqlite3_vtab,
    info: *mut sqlite3_index_info,
) -> c_int {
    // SAFETY: SQLite supplies a live table and exclusive planning structure.
    let vtab = unsafe { &*p_vtab.cast::<VTab>() };
    // SAFETY: info is exclusively accessible in this planning callback.
    let info = unsafe { &mut *info };
    let mut argv_index = 0;
    let mut short_names = Vec::new();
    for i in 0..info.nConstraint as usize {
        // SAFETY: SQLite supplies nConstraint initialized constraint entries.
        // Copy the fields so no element reference crosses the vtab_in call.
        let (usable, column, op) = unsafe {
            let constraint = &*info.aConstraint.add(i);
            (
                constraint.usable,
                constraint.iColumn,
                constraint.op as c_int,
            )
        };
        if usable == 0 {
            continue;
        }
        let short_name = if op == FUNC_KNN && column == COL_VECTOR {
            info.estimatedCost = 100.0;
            Some("ks")
        } else if column == -1 {
            // SAFETY: the extension API is initialized before any callback.
            if unsafe { ffi::libversion_number() } < 3038000 {
                return vtab.report(VTabError::new(
                    "SQLite version is too old: sqlite version 3.38.0 or higher is required.",
                ));
            }
            if op == ffi::SQLITE_INDEX_CONSTRAINT_EQ as c_int {
                // SAFETY: called only in xBestIndex, after checking the API's
                // minimum version, using an in-bounds constraint index.
                if unsafe { ffi::vtab_in(info, i as c_int, 1) } != 0 {
                    info.estimatedCost = 200.0;
                    Some("in")
                } else {
                    info.estimatedCost = 100.0;
                    Some("eq")
                }
            } else {
                None
            }
        } else {
            None
        };
        if let Some(short_name) = short_name {
            argv_index += 1;
            // SAFETY: aConstraintUsage has nConstraint writable entries.
            let usage = unsafe { &mut *info.aConstraintUsage.add(i) };
            usage.argvIndex = argv_index;
            usage.omit = 1;
            short_names.push(short_name);
        }
    }
    if short_names.is_empty() {
        return vtab.report(VTabError::with_code(
            ffi::SQLITE_CONSTRAINT as c_int,
            "No valid constraint found in where clause",
        ));
    }
    // SAFETY: initialized SQLite allocator; SQLite frees this plan string.
    let plan = unsafe { ffi::sqlite_strdup(&short_names.concat()) };
    if plan.is_null() {
        return vtab.report(VTabError::with_code(
            ffi::SQLITE_NOMEM as c_int,
            "Failed to allocate memory for idxStr",
        ));
    }
    info.idxStr = plan;
    info.needToFreeIdxStr = 1;
    info.idxNum = argv_index;
    if matches!(vtab.backend, Backend::Diskann(_)) && info.colUsed & 1 != 0 {
        info.idxNum |= PLAN_VECTOR_OUTPUT;
    }
    ffi::SQLITE_OK as c_int
}

// ---- filter ----

#[derive(Debug, PartialEq)]
enum Constraint {
    Knn,
    In,
    Eq,
}

fn parse_plan(idx_num: c_int, codes: &[u8], argc: c_int) -> Result<Vec<Constraint>, VTabError> {
    if argc <= 0
        || idx_num & !PLAN_VECTOR_OUTPUT != argc
        || !codes.len().is_multiple_of(2)
        || codes.len() / 2 != argc as usize
    {
        return Err(VTabError::new(
            "invalid query constraint plan or argument count",
        ));
    }
    codes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|code| match code {
            b"ks" => Ok(Constraint::Knn),
            b"in" => Ok(Constraint::In),
            b"eq" => Ok(Constraint::Eq),
            _ => Err(VTabError::new("unknown constraint short name")),
        })
        .collect()
}

fn in_step_status(code: c_int, has_value: bool) -> Result<bool, VTabError> {
    match code {
        code if code == ffi::SQLITE_DONE as c_int => Ok(false),
        code if code == ffi::SQLITE_OK as c_int && has_value => Ok(true),
        code if code == ffi::SQLITE_OK as c_int => Err(VTabError::new(
            "IN iterator returned no value without SQLITE_DONE",
        )),
        code => Err(VTabError::with_code(
            code,
            format!("Failed to iterate rowid IN constraint (SQLite error {code})"),
        )),
    }
}

/// The token is constructed only for an all-at-once IN argument inside xFilter.
/// Items copy the rowid before advancing, so SQLite's short-lived value pointer
/// cannot escape the iterator. Any terminal error is delivered as an Err item.
struct InRowids<'value, 'callback> {
    value: &'value mut Value<'callback>,
    started: bool,
    done: bool,
}

impl<'value, 'callback> InRowids<'value, 'callback> {
    /// # Safety
    /// `value` is the current xFilter argument selected by xBestIndex for
    /// all-at-once IN processing; this iterator must stay in that callback.
    unsafe fn new(value: &'value mut Value<'callback>) -> Self {
        Self {
            value,
            started: false,
            done: false,
        }
    }
}

impl Iterator for InRowids<'_, '_> {
    type Item = Result<u64, VTabError>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let mut rowid_value = std::ptr::null_mut();
        // SAFETY: construction records the all-at-once IN callback contract.
        // No returned value survives this method or the next iterator call.
        let code = unsafe {
            if self.started {
                ffi::vtab_in_next(self.value.as_ptr(), &mut rowid_value)
            } else {
                ffi::vtab_in_first(self.value.as_ptr(), &mut rowid_value)
            }
        };
        self.started = true;
        match in_step_status(code, !rowid_value.is_null()) {
            Ok(false) => {
                self.done = true;
                None
            }
            Err(error) => {
                self.done = true;
                Some(Err(error))
            }
            Ok(true) => {
                // SAFETY: SQLITE_OK provided a protected non-null value valid
                // until the next step, and only integer access occurs here.
                let value = unsafe { Value::from_raw(rowid_value) };
                if value.kind() != ffi::SQLITE_INTEGER as c_int {
                    self.done = true;
                    Some(Err(VTabError::new("rowid must be of type INTEGER")))
                } else {
                    Some(Ok(value.int64() as u64))
                }
            }
        }
    }
}

#[derive(Default)]
struct Constraints<'a> {
    knn: Option<&'a KnnParam>,
    rowid_in: Option<HashSet<u64>>,
    rowid_eq: Option<u64>,
}

/// # Safety
/// `plan` must describe the values supplied by SQLite to the current xFilter,
/// as selected by this module's xBestIndex. KnnParam pointers must remain live
/// until the returned constraints have been consumed within this callback.
unsafe fn materialize_constraints<'a>(
    plan: &[Constraint],
    values: &mut [Value<'a>],
    diskann_in_limit: Option<usize>,
) -> Result<Constraints<'a>, VTabError> {
    let mut constraints = Constraints::default();
    for (code, value) in plan.iter().zip(values.iter_mut()) {
        match code {
            Constraint::Knn => {
                if constraints.knn.is_some() {
                    return Err(VTabError::new("only one knn_search constraint is allowed"));
                }
                // SAFETY: this stable tag belongs to scalar::knn_param, whose
                // destructor keeps the pointed-to KnnParam alive in this value.
                let pointer =
                    unsafe { value.pointer(KNN_PARAM_TYPE.as_ptr().cast()) }.cast::<KnnParam>();
                if pointer.is_null() {
                    return Err(VTabError::new("Failed to materialize constraint: knn_param() should be used for the 2nd param of knn_search()"));
                }
                // SAFETY: the producer/tag pair guarantees type and alignment;
                // the borrowed object is consumed before xFilter returns.
                constraints.knn = Some(unsafe { &*pointer });
            }
            Constraint::In | Constraint::Eq => {
                if constraints.rowid_in.is_some() || constraints.rowid_eq.is_some() {
                    return Err(VTabError::new("only one rowid constraint is allowed"));
                }
                if *code == Constraint::In {
                    // SAFETY: this checked plan code corresponds to xBestIndex's
                    // successful all-at-once IN selection for this argument.
                    // SAFETY: the selected IN value remains scoped to this
                    // callback. DiskANN limits even the caller's filter set.
                    let rowids = unsafe { InRowids::new(value) };
                    if let Some(limit) = diskann_in_limit {
                        let mut ids = HashSet::new();
                        for rowid in rowids {
                            let rowid = rowid?;
                            if ids.contains(&rowid) {
                                continue;
                            }
                            if ids.len() >= limit {
                                return Err(VTabError::with_code(
                                    ffi::SQLITE_TOOBIG as c_int,
                                    "DiskANN rowid filter exceeds its memory budget",
                                ));
                            }
                            ids.try_reserve(1).map_err(|_| {
                                VTabError::with_code(
                                    ffi::SQLITE_NOMEM as c_int,
                                    "cannot allocate DiskANN rowid filter",
                                )
                            })?;
                            ids.insert(rowid);
                        }
                        constraints.rowid_in = Some(ids);
                    } else {
                        constraints.rowid_in = Some(rowids.collect::<Result<_, _>>()?);
                    }
                } else {
                    if value.kind() != ffi::SQLITE_INTEGER as c_int {
                        return Err(VTabError::new("rowid must be of type INTEGER"));
                    }
                    constraints.rowid_eq = Some(value.int64() as u64);
                }
            }
        }
    }
    Ok(constraints)
}

fn query_rows(
    entry: &IndexEntry,
    constraints: Constraints<'_>,
) -> Result<Vec<SearchResult>, VTabError> {
    if let Some(knn) = constraints.knn {
        if knn.diskann_search_list_size.is_some() {
            return Err(VTabError::new(
                "search_list_size is a DiskANN option; HNSW uses integer ef",
            ));
        }
        if knn.query_vector.len() != entry.space.dim {
            return Err(VTabError::new(format!(
                "query vector's dimension({}) doesn't match {}'s dimension: {}",
                knn.query_vector.len(),
                entry.space.vector_name,
                entry.space.dim
            )));
        }
        let filter = if let Some(ref ids) = constraints.rowid_in {
            SearchFilter::In(ids)
        } else if let Some(eq) = constraints.rowid_eq {
            SearchFilter::Equals(eq)
        } else {
            SearchFilter::None
        };
        let k =
            usize::try_from(knn.k).map_err(|_| VTabError::new("k exceeds the supported range"))?;
        let ef = knn
            .ef
            .map(usize::try_from)
            .transpose()
            .map_err(|_| VTabError::new("ef exceeds the supported range"))?;
        entry
            .index
            .search(&knn.query_vector, k, ef, filter)
            .map_err(|error| VTabError::new(format!("Failed to execute query due to: {error}")))
    } else {
        let mut out = Vec::new();
        if let Some(ids) = constraints.rowid_in {
            for id in ids {
                if entry.index.contains(id) {
                    out.push(SearchResult::new(0.0, id));
                }
            }
        } else if let Some(eq) = constraints.rowid_eq {
            if entry.index.contains(eq) {
                out.push(SearchResult::new(0.0, eq));
            }
        }
        Ok(out)
    }
}

enum QueryBatch {
    Hnsw(Vec<SearchResult>),
    Diskann(QueryRows),
}

fn query_backend(
    backend: &Backend,
    constraints: Constraints<'_>,
    project_vector: bool,
) -> Result<QueryBatch, VTabError> {
    let Backend::Diskann(table) = backend else {
        let Backend::Hnsw(entry) = backend else {
            return Err(VTabError::new("unknown vector backend"));
        };
        return query_rows(entry, constraints).map(QueryBatch::Hnsw);
    };
    let filter = if let Some(ref ids) = constraints.rowid_in {
        SearchFilter::In(ids)
    } else if let Some(rowid) = constraints.rowid_eq {
        SearchFilter::Equals(rowid)
    } else {
        SearchFilter::None
    };
    let rows = if let Some(knn) = constraints.knn {
        if knn.ef.is_some() {
            return Err(VTabError::new(
                "integer ef is an HNSW option; DiskANN uses JSON search_list_size",
            ));
        }
        let k =
            usize::try_from(knn.k).map_err(|_| VTabError::new("k exceeds the supported range"))?;
        let search_l = knn
            .diskann_search_list_size
            .map(usize::try_from)
            .transpose()
            .map_err(|_| VTabError::new("search_list_size exceeds the supported range"))?;
        table.knn(&knn.query_vector, k, search_l, filter, project_vector)
    } else {
        table.select_rowids(filter, project_vector)
    }?;
    Ok(QueryBatch::Diskann(rows))
}

unsafe extern "C" fn x_filter(
    p_cur: *mut sqlite3_vtab_cursor,
    idx_num: c_int,
    idx_str: *const c_char,
    argc: c_int,
    argv: *mut *mut sqlite3_value,
) -> c_int {
    // SAFETY: SQLite provides exclusive access to the live cursor.
    let cursor = unsafe { &mut *p_cur.cast::<Cursor>() };
    // SAFETY: SQLite keeps the cursor's parent table alive during the callback.
    let vtab = unsafe { &*cursor.base.pVtab.cast::<VTab>() };
    let _operation = match DiskannCallbackGuard::for_backend(&vtab.backend) {
        Ok(guard) => guard,
        Err(error) => return vtab.report(error),
    };
    if matches!(vtab.backend, Backend::Diskann(_)) {
        // Release old capacity before admitting a new operation's workspace.
        cursor.result = Vec::new();
    } else {
        cursor.result.clear();
    }
    cursor.disk_vectors = None;
    cursor.snapshot_anchor = None;
    cursor.current = 0;
    let codes = if idx_str.is_null() {
        &[][..]
    } else {
        // SAFETY: SQLite supplies the NUL-terminated plan created by xBestIndex.
        unsafe { CStr::from_ptr(idx_str) }.to_bytes()
    };
    let plan = match parse_plan(idx_num, codes, argc) {
        Ok(plan) => plan,
        Err(error) => return vtab.report(error),
    };
    // SAFETY: positive argc was checked against the plan. SQLite supplies this
    // many protected values, used exclusively within the current callback.
    let values = unsafe { ffi::arguments(argc, argv) };
    // SAFETY: plan and values come from xBestIndex and this xFilter invocation;
    // both IN iteration and the KnnParam borrow end before this callback exits.
    let in_limit = match &vtab.backend {
        Backend::Diskann(table) => Some(table.options().cache_bytes / 256),
        Backend::Hnsw(_) => None,
    };
    // SAFETY: the validated xBestIndex plan describes these protected xFilter
    // values; IN iteration and KnnParam borrows remain within this callback.
    let result =
        unsafe { materialize_constraints(&plan, values, in_limit) }.and_then(|constraints| {
            query_backend(
                &vtab.backend,
                constraints,
                idx_num & PLAN_VECTOR_OUTPUT != 0,
            )
        });
    match result {
        Ok(QueryBatch::Hnsw(rows)) => {
            cursor.result = rows;
            ffi::SQLITE_OK as c_int
        }
        Ok(QueryBatch::Diskann(rows)) => {
            cursor.result = rows.results;
            cursor.disk_vectors = rows.vectors;
            cursor.snapshot_anchor = Some(rows.anchor);
            ffi::SQLITE_OK as c_int
        }
        Err(error) => vtab.report(error),
    }
}

// ---- find function ----

unsafe extern "C" fn x_find_function(
    _p_vtab: *mut sqlite3_vtab,
    _n_arg: c_int,
    z_name: *const c_char,
    px_func: *mut Option<
        unsafe extern "C" fn(*mut sqlite3_context, c_int, *mut *mut sqlite3_value),
    >,
    pp_arg: *mut *mut c_void,
) -> c_int {
    // SAFETY: SQLite supplies a valid NUL-terminated function name.
    if unsafe { cstr(z_name) } == "knn_search" {
        // SAFETY: both output pointers are writable for this callback.
        unsafe {
            *px_func = Some(scalar::knn_search);
            *pp_arg = std::ptr::null_mut();
        }
        FUNC_KNN
    } else {
        0
    }
}

// ---- update (insert / delete / update / persistence) ----

fn execute_persistence(entry: &IndexEntry, values: &mut [Value<'_>]) -> Result<i64, VTabError> {
    let operation = values[(2 + COL_OPERATION) as usize].text()?.to_owned();
    let path_value = &mut values[(2 + COL_PATH) as usize];
    if path_value.kind() != ffi::SQLITE_TEXT as c_int {
        return Err(VTabError::new(format!(
            "path must be provided as TEXT for '{operation}' operation"
        )));
    }
    let path = path_value.text()?;
    match operation.as_str() {
        "save" => entry.index.save(path),
        "load" => entry.index.load(path),
        _ => {
            return Err(VTabError::new(format!(
                "unknown operation '{operation}'; expected 'save' or 'load'"
            )))
        }
    }
    .map_err(|error| VTabError::new(format!("{operation} failed: {error}")))?;
    Ok(0)
}

fn insert_or_update_vector(
    backend: &Backend,
    value: &mut Value<'_>,
    rowid: u64,
    is_update: bool,
) -> Result<(), VTabError> {
    if value.kind() != ffi::SQLITE_BLOB as c_int {
        return Err(VTabError::new("vector must be of type Blob"));
    }
    let vec = vector::view_from_blob(value.blob()?)
        .map_err(|error| VTabError::new(format!("Failed to perform insertion due to: {error}")))?;
    if vec.len() != backend.space().dim {
        return Err(VTabError::new(format!(
            "Dimension mismatch: vector's dimension {}, table's dimension {}",
            vec.len(),
            backend.space().dim
        )));
    }
    match backend {
        Backend::Hnsw(entry) => entry.index.add(&vec, rowid).map_err(|error| {
            VTabError::new(format!("Failed to insert row {rowid} due to: {error}"))
        }),
        Backend::Diskann(table) => if is_update {
            table.update(rowid, rowid, &vec)
        } else {
            table.insert(rowid, &vec)
        }
        .map_err(Into::into),
    }
}

fn update(
    backend: &Backend,
    values: &mut [Value<'_>],
    insert_rowid: Option<i64>,
) -> Result<i64, VTabError> {
    if values.len() != 1 && values.len() != 6 {
        return Err(VTabError::new("invalid update argument count"));
    }
    let argv0_type = values[0].kind();
    let null = ffi::SQLITE_NULL as c_int;
    let integer = ffi::SQLITE_INTEGER as c_int;
    if values.len() > 1 && argv0_type == null {
        if values[(2 + COL_OPERATION) as usize].kind() == ffi::SQLITE_TEXT as c_int {
            return match backend {
                Backend::Hnsw(entry) => execute_persistence(entry, values),
                Backend::Diskann(table) => {
                    let operation = values[(2 + COL_OPERATION) as usize].text()?;
                    if operation == "consolidate" {
                        table.consolidate().map_err(VTabError::from)?;
                        Ok(0)
                    } else {
                        Err(VTabError::new(
                            "DiskANN data persists inside SQLite; use SQLite backup instead of save/load. Supported operation: consolidate",
                        ))
                    }
                }
            };
        }
        if values[1].kind() == null {
            return Err(VTabError::new("rowid must be specified during insertion"));
        }
        let raw_rowid = insert_rowid.ok_or_else(|| VTabError::new("missing insertion rowid"))?;
        if raw_rowid < 0 {
            return Err(VTabError::new(format!("rowid {raw_rowid} out of range")));
        }
        let rowid = raw_rowid as u64;
        if backend.contains(rowid)? {
            let error = format!("row {rowid} already exists");
            return Err(if matches!(backend, Backend::Diskann(_)) {
                VTabError::with_code(ffi::SQLITE_CONSTRAINT as c_int, error)
            } else {
                VTabError::new(error)
            });
        }
        insert_or_update_vector(backend, &mut values[2], rowid, false)?;
        Ok(raw_rowid)
    } else if values.len() == 1 && argv0_type != null {
        let raw_rowid = values[0].int64();
        if raw_rowid < 0 {
            return Err(VTabError::new(format!("rowid {raw_rowid} out of range")));
        }
        backend.mark_delete(raw_rowid as u64)?;
        Ok(raw_rowid)
    } else if values.len() > 1 && argv0_type != null {
        if argv0_type != integer {
            return Err(VTabError::new("rowid must be of type INTEGER"));
        }
        if values[1].kind() != integer {
            return Err(VTabError::new("target rowid must be of type INTEGER"));
        }
        let source_rowid = values[0].int64();
        if source_rowid != values[1].int64() {
            return Err(VTabError::new("rowid cannot be changed"));
        }
        if source_rowid < 0 {
            return Err(VTabError::new(format!("rowid {source_rowid} out of range")));
        }
        let rowid = source_rowid as u64;
        if !backend.contains(rowid)? {
            return Err(VTabError::new(format!("rowid {source_rowid} not found")));
        }
        insert_or_update_vector(backend, &mut values[2], rowid, true)?;
        Ok(source_rowid)
    } else {
        Err(VTabError::new("Operation not supported for now"))
    }
}

unsafe extern "C" fn x_update(
    p_vtab: *mut sqlite3_vtab,
    argc: c_int,
    argv: *mut *mut sqlite3_value,
    p_rowid: *mut i64,
) -> c_int {
    // SAFETY: SQLite provides this module's live table for the callback.
    let vtab = unsafe { &*p_vtab.cast::<VTab>() };
    let _operation = match DiskannCallbackGuard::for_backend(&vtab.backend) {
        Ok(guard) => guard,
        Err(error) => return vtab.report(error),
    };
    if argc != 1 && argc != 6 {
        return vtab.report(VTabError::new("invalid update argument count"));
    }
    // SAFETY: SQLite supplies the checked number of protected callback values.
    let values = unsafe { ffi::arguments(argc, argv) };
    let insert_rowid = if argc == 6
        && values[0].kind() == ffi::SQLITE_NULL as c_int
        && values[(2 + COL_OPERATION) as usize].kind() != ffi::SQLITE_TEXT as c_int
    {
        // SAFETY: preserve SQLite's existing insertion-rowid coercion before
        // borrowing any argument views, including potentially aliased values.
        Some(unsafe { ffi::value_int64(values[1].as_ptr()) })
    } else {
        None
    };
    let result = match &vtab.backend {
        Backend::Hnsw(_) => update(&vtab.backend, values, insert_rowid),
        Backend::Diskann(table) => table
            .atomic_write(|| {
                update(&vtab.backend, values, insert_rowid)
                    .map_err(|error| IndexError::with_code(error.code, error.message))
            })
            .map_err(Into::into),
    };
    match result {
        Ok(rowid) => {
            // SQLite only requires p_rowid for inserts; do not assume it is
            // writable for DELETE callbacks (argc == 1).
            if argc > 1 && !p_rowid.is_null() {
                // SAFETY: SQLite supplies a writable rowid output for inserts.
                unsafe { *p_rowid = rowid };
            }
            ffi::SQLITE_OK as c_int
        }
        Err(error) => vtab.report(error),
    }
}

// ---- transaction / shadow-table callbacks ----

// DiskANN writes through inside its ordinary, multirow journal-carrier UPDATE.
// SQLite owns semantic undo, including statement savepoints: a single VUpdate
// callback alone is not a sufficient journal boundary for several shadow writes.
// No graph/workspace is buffered between callbacks, and xCommit must never do
// fallible I/O. HNSW remains intentionally nontransactional despite these hooks.
unsafe extern "C" fn x_transaction(_p_vtab: *mut sqlite3_vtab) -> c_int {
    ffi::SQLITE_OK as c_int
}

unsafe extern "C" fn x_savepoint(_p_vtab: *mut sqlite3_vtab, _savepoint: c_int) -> c_int {
    // Includes SQLite's statement rollback-to(-1) and late enrollment. Every
    // DiskANN operation discarded its transient workspace before this callback.
    ffi::SQLITE_OK as c_int
}

unsafe extern "C" fn x_shadow_name(suffix: *const c_char) -> c_int {
    if suffix.is_null() {
        return 0;
    }
    // SAFETY: SQLite supplies the NUL-terminated suffix synchronously.
    i32::from(matches!(
        unsafe { cstr(suffix) },
        "diskann_meta" | "diskann_nodes" | "diskann_txn" | "diskann_rebuild"
    ))
}

// ---- module definition ----

struct ModuleWrap(sqlite3_module);
// SAFETY: this static module contains immutable function pointers and null
// optional hooks. SQLite only reads it; per-connection state is stored in pAux.
unsafe impl Sync for ModuleWrap {}

static MODULE: ModuleWrap = ModuleWrap(sqlite3_module {
    iVersion: 3,
    xCreate: Some(x_create),
    xConnect: Some(x_connect),
    xBestIndex: Some(x_best_index),
    xDisconnect: Some(x_disconnect),
    xDestroy: Some(x_destroy),
    xOpen: Some(x_open),
    xClose: Some(x_close),
    xFilter: Some(x_filter),
    xNext: Some(x_next),
    xEof: Some(x_eof),
    xColumn: Some(x_column),
    xRowid: Some(x_rowid),
    xUpdate: Some(x_update),
    xBegin: Some(x_transaction),
    xSync: Some(x_transaction),
    xCommit: Some(x_transaction),
    xRollback: Some(x_transaction),
    xFindFunction: Some(x_find_function),
    xRename: Some(x_rename),
    xSavepoint: Some(x_savepoint),
    xRelease: Some(x_savepoint),
    xRollbackTo: Some(x_savepoint),
    xShadowName: Some(x_shadow_name),
    xIntegrity: None,
});

pub fn module_ptr() -> *const sqlite3_module {
    &MODULE.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_requires_matching_counts_and_known_codes() {
        assert_eq!(
            parse_plan(2, b"ksin", 2).unwrap(),
            vec![Constraint::Knn, Constraint::In]
        );
        assert_eq!(parse_plan(1, b"eq", 1).unwrap(), vec![Constraint::Eq]);
        for (number, codes, count) in [
            (0, &b""[..], 0),
            (1, &b"ks"[..], 0),
            (1, &b"ksin"[..], 1),
            (2, &b"ks"[..], 2),
            (1, &b"k"[..], 1),
            (1, &b"xx"[..], 1),
            (-1, &b"eq"[..], -1),
        ] {
            assert!(parse_plan(number, codes, count).is_err());
        }
    }

    #[test]
    fn in_iteration_distinguishes_completion_from_errors() {
        assert!(in_step_status(ffi::SQLITE_OK as c_int, true).unwrap());
        assert!(!in_step_status(ffi::SQLITE_DONE as c_int, false).unwrap());
        assert!(in_step_status(ffi::SQLITE_OK as c_int, false).is_err());
        for code in [ffi::SQLITE_NOMEM, ffi::SQLITE_ERROR, ffi::SQLITE_INTERRUPT] {
            let error = in_step_status(code as c_int, false).unwrap_err();
            assert_eq!(error.code, code as c_int);
        }
    }
}
