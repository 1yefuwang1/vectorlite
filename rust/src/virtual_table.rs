//! SQLite callbacks for vectorlite. Callbacks validate the host's arguments and
//! delegate policy to safe helpers; numeric work lives in `crate::core`.

#![deny(unsafe_op_in_unsafe_fn)]

use std::cell::UnsafeCell;
use std::collections::HashSet;
use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_void};
use std::rc::Rc;

use crate::core::{Index, SearchFilter, SearchResult};
use crate::ffi::{
    self, sqlite3, sqlite3_context, sqlite3_index_info, sqlite3_module, sqlite3_value,
    sqlite3_vtab, sqlite3_vtab_cursor, Value,
};
use crate::index_options::IndexOptions;
use crate::registry::{IndexEntry, Registry, RegistryKey};
use crate::scalar::{self, KnnParam, KNN_PARAM_TYPE};
use crate::vector;
use crate::vector_space::{parse_named_vector_space, NamedVectorSpace};

const COL_VECTOR: c_int = 0;
const COL_DISTANCE: c_int = 1;
const COL_OPERATION: c_int = 2;
const COL_PATH: c_int = 3;
const FUNC_KNN: c_int = ffi::SQLITE_INDEX_CONSTRAINT_FUNCTION as c_int;

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
    entry: Rc<IndexEntry>,
}

#[repr(C)]
pub struct Cursor {
    base: sqlite3_vtab_cursor,
    result: Vec<SearchResult>,
    current: usize,
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
    options: IndexOptions,
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
        let options = IndexOptions::parse(&index_options_str).map_err(|error| {
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
    // SAFETY: this helper is called only from xCreate/xConnect with SQLite's db.
    let rc = unsafe { ffi::vtab_config_constraint_support(db) };
    if rc != ffi::SQLITE_OK as c_int {
        return rc;
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
    // SAFETY: the module owns a Registry in pAux until every table disconnects.
    // No SQLite calls occur while this mutable registry borrow is held.
    let entry = match find_or_create_entry(
        unsafe { &mut *registry },
        is_create,
        (schema.clone(), table.clone()),
        definition,
    ) {
        Ok(entry) => entry,
        Err(error) => {
            // SAFETY: pz_err is SQLite's writable error output.
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
        entry,
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
    // SAFETY: SQLite transfers the table allocation back exactly once.
    let vtab = unsafe { Box::from_raw(p_vtab.cast::<VTab>()) };
    // SAFETY: pAux outlives this table; serialized callbacks and the retained
    // entry ensure no registry reference is live during this mutation.
    unsafe { &mut *vtab.registry }.erase(&vtab.key());
    drop(vtab);
    ffi::SQLITE_OK as c_int
}

unsafe extern "C" fn x_rename(p_vtab: *mut sqlite3_vtab, z_new: *const c_char) -> c_int {
    // SAFETY: SQLite provides exclusive callback access to this live table.
    let vtab = unsafe { &mut *p_vtab.cast::<VTab>() };
    // SAFETY: z_new is a valid callback-scoped C string.
    let new_table = unsafe { cstr(z_new) }.to_owned();
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
            match vtab.entry.index.get_vector(row.rowid) {
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
        || idx_num != argc
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
                    constraints.rowid_in =
                        Some(unsafe { InRowids::new(value) }.collect::<Result<_, _>>()?);
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
    cursor.result.clear();
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
    let result = unsafe { materialize_constraints(&plan, values) }
        .and_then(|constraints| query_rows(&vtab.entry, constraints));
    match result {
        Ok(rows) => {
            cursor.result = rows;
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
    entry: &IndexEntry,
    value: &mut Value<'_>,
    rowid: u64,
) -> Result<(), VTabError> {
    if value.kind() != ffi::SQLITE_BLOB as c_int {
        return Err(VTabError::new("vector must be of type Blob"));
    }
    let vec = vector::view_from_blob(value.blob()?)
        .map_err(|error| VTabError::new(format!("Failed to perform insertion due to: {error}")))?;
    if vec.len() != entry.space.dim {
        return Err(VTabError::new(format!(
            "Dimension mismatch: vector's dimension {}, table's dimension {}",
            vec.len(),
            entry.space.dim
        )));
    }
    entry
        .index
        .add(&vec, rowid)
        .map_err(|error| VTabError::new(format!("Failed to insert row {rowid} due to: {error}")))
}

fn update(
    entry: &IndexEntry,
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
            return execute_persistence(entry, values);
        }
        if values[1].kind() == null {
            return Err(VTabError::new("rowid must be specified during insertion"));
        }
        let raw_rowid = insert_rowid.ok_or_else(|| VTabError::new("missing insertion rowid"))?;
        if raw_rowid < 0 {
            return Err(VTabError::new(format!("rowid {raw_rowid} out of range")));
        }
        let rowid = raw_rowid as u64;
        if entry.index.contains(rowid) {
            return Err(VTabError::new(format!("row {rowid} already exists")));
        }
        insert_or_update_vector(entry, &mut values[2], rowid)?;
        Ok(raw_rowid)
    } else if values.len() == 1 && argv0_type != null {
        let raw_rowid = values[0].int64();
        if raw_rowid < 0 {
            return Err(VTabError::new(format!("rowid {raw_rowid} out of range")));
        }
        entry.index.mark_delete(raw_rowid as u64).map_err(|error| {
            VTabError::new(format!("Delete failed with rowid {raw_rowid}: {error}"))
        })?;
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
        if !entry.index.contains(rowid) {
            return Err(VTabError::new(format!("rowid {source_rowid} not found")));
        }
        insert_or_update_vector(entry, &mut values[2], rowid)?;
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
    match update(&vtab.entry, values, insert_rowid) {
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
    xBegin: None,
    xSync: None,
    xCommit: None,
    xRollback: None,
    xFindFunction: Some(x_find_function),
    xRename: Some(x_rename),
    xSavepoint: None,
    xRelease: None,
    xRollbackTo: None,
    xShadowName: None,
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
