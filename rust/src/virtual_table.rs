//! The vectorlite virtual-table module. Mirrors `virtual_table.cpp`: it owns the
//! SQLite glue (xCreate/xConnect/xBestIndex/xFilter/xUpdate/xColumn/...) and
//! delegates all numeric work to the C++ core via `crate::core`.

use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_void};

use crate::core::{Index, SearchFilter};
use crate::ffi::{
    self, sqlite3, sqlite3_context, sqlite3_index_info, sqlite3_module, sqlite3_value,
    sqlite3_vtab, sqlite3_vtab_cursor,
};
use crate::index_options::IndexOptions;
use crate::registry::{IndexEntry, Registry, RegistryKey};
use crate::scalar::{self, KnnParam, KNN_PARAM_TYPE};
use crate::vector;
use crate::vector_space::parse_named_vector_space;

const COL_VECTOR: c_int = 0;
const COL_DISTANCE: c_int = 1;
const COL_OPERATION: c_int = 2;
const COL_PATH: c_int = 3;

// xFindFunction return code identifying the knn_search constraint.
const FUNC_KNN: c_int = ffi::SQLITE_INDEX_CONSTRAINT_FUNCTION as c_int;

// idxStr short names, matching constraint.h.
const SN_KNN: &str = "ks";
const SN_IN: &str = "in";
const SN_EQ: &str = "eq";

#[repr(C)]
pub struct VTab {
    base: sqlite3_vtab,
    registry: *mut Registry,
    schema: String,
    table: String,
}

#[repr(C)]
pub struct Cursor {
    base: sqlite3_vtab_cursor,
    result: Vec<(f32, u64)>,
    current: usize,
}

impl VTab {
    fn key(&self) -> RegistryKey {
        (self.schema.clone(), self.table.clone())
    }
    unsafe fn registry(&self) -> &Registry {
        &*self.registry
    }
    unsafe fn registry_mut(&self) -> &mut Registry {
        &mut *self.registry
    }
    unsafe fn entry(&self) -> &IndexEntry {
        self.registry().find(&self.key()).expect("registry entry missing")
    }
}

unsafe fn set_vtab_err(vtab: *mut sqlite3_vtab, msg: &str) {
    let base = &mut *vtab;
    set_vtab_err_field(&mut base.zErrMsg, msg);
}

unsafe fn set_vtab_err_field(field: &mut *mut c_char, msg: &str) {
    if !field.is_null() {
        ffi::sqlite_free(*field as *mut c_void);
    }
    *field = ffi::sqlite_strdup(msg);
}

unsafe fn cstr<'a>(p: *const c_char) -> &'a str {
    CStr::from_ptr(p).to_str().unwrap_or("")
}

unsafe fn uarg(argv: *mut *mut sqlite3_value, i: usize) -> *mut sqlite3_value {
    *argv.add(i)
}

// ---- create / connect ----

unsafe fn init_vtab(
    is_create: bool,
    db: *mut sqlite3,
    p_aux: *mut c_void,
    argc: c_int,
    argv: *const *const c_char,
    pp_vtab: *mut *mut sqlite3_vtab,
    pz_err: *mut *mut c_char,
) -> c_int {
    let rc = ffi::vtab_config_constraint_support(db);
    if rc != ffi::SQLITE_OK as c_int {
        return rc;
    }

    const MODULE_PARAM_OFFSET: usize = 3;
    if argc as usize != 2 + MODULE_PARAM_OFFSET {
        ffi::set_err(
            pz_err,
            &format!(
                "vectorlite expects 2 arguments (a vector space and index options), got {}. \
                 The index file path argument has been removed; use INSERT INTO \
                 <table>(operation, path) VALUES('save', <path>) to persist an index and \
                 INSERT INTO <table>(operation, path) VALUES('load', <path>) to restore one.",
                argc as usize - MODULE_PARAM_OFFSET
            ),
        );
        return ffi::SQLITE_ERROR as c_int;
    }

    let schema = cstr(*argv.add(1)).to_string();
    let table = cstr(*argv.add(2)).to_string();
    let vector_space_str = cstr(*argv.add(MODULE_PARAM_OFFSET)).to_string();
    let index_options_str = cstr(*argv.add(1 + MODULE_PARAM_OFFSET)).to_string();

    let space = match parse_named_vector_space(&vector_space_str) {
        Ok(s) => s,
        Err(e) => {
            ffi::set_err(
                pz_err,
                &format!("Invalid vector space: {}. Reason: {}", vector_space_str, e),
            );
            return ffi::SQLITE_ERROR as c_int;
        }
    };

    let options = match IndexOptions::parse(&index_options_str) {
        Ok(o) => o,
        Err(e) => {
            ffi::set_err(
                pz_err,
                &format!("Invalid index_options {}. Reason: {}", index_options_str, e),
            );
            return ffi::SQLITE_ERROR as c_int;
        }
    };

    let declare_sql = format!(
        "CREATE TABLE X({}, distance REAL hidden, operation TEXT hidden, path TEXT hidden)",
        space.vector_name
    );
    let rc = ffi::declare_vtab(db, &declare_sql);
    if rc != ffi::SQLITE_OK as c_int {
        return rc;
    }

    let registry = p_aux as *mut Registry;
    let key: RegistryKey = (schema.clone(), table.clone());

    // On connect/reparse, reuse an existing matching index; otherwise build fresh.
    let mut need_build = true;
    if !is_create {
        if let Some(existing) = (*registry).find(&key) {
            if existing.vector_space_str == vector_space_str
                && existing.index_options_str == index_options_str
            {
                need_build = false;
            }
        }
    }

    if need_build {
        let index = match Index::create(
            space.dim,
            space.distance_type,
            space.vector_type,
            options.max_elements,
            options.m,
            options.ef_construction,
            options.random_seed,
            options.allow_replace_deleted,
        ) {
            Ok(i) => i,
            Err(e) => {
                ffi::set_err(pz_err, &format!("Failed to create virtual table: {}", e));
                return ffi::SQLITE_ERROR as c_int;
            }
        };
        (*registry).insert(
            key,
            IndexEntry {
                index,
                space,
                allow_replace_deleted: options.allow_replace_deleted,
                vector_space_str,
                index_options_str,
            },
        );
    }

    let vtab = Box::new(VTab {
        base: std::mem::zeroed(),
        registry,
        schema,
        table,
    });
    *pp_vtab = Box::into_raw(vtab) as *mut sqlite3_vtab;
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
    init_vtab(true, db, p_aux, argc, argv, pp_vtab, pz_err)
}

unsafe extern "C" fn x_connect(
    db: *mut sqlite3,
    p_aux: *mut c_void,
    argc: c_int,
    argv: *const *const c_char,
    pp_vtab: *mut *mut sqlite3_vtab,
    pz_err: *mut *mut c_char,
) -> c_int {
    init_vtab(false, db, p_aux, argc, argv, pp_vtab, pz_err)
}

unsafe extern "C" fn x_disconnect(p_vtab: *mut sqlite3_vtab) -> c_int {
    let vtab = Box::from_raw(p_vtab as *mut VTab);
    if !vtab.base.zErrMsg.is_null() {
        ffi::sqlite_free(vtab.base.zErrMsg as *mut c_void);
    }
    drop(vtab);
    ffi::SQLITE_OK as c_int
}

unsafe extern "C" fn x_destroy(p_vtab: *mut sqlite3_vtab) -> c_int {
    let vtab = Box::from_raw(p_vtab as *mut VTab);
    let key = vtab.key();
    (&mut *vtab.registry).erase(&key);
    if !vtab.base.zErrMsg.is_null() {
        ffi::sqlite_free(vtab.base.zErrMsg as *mut c_void);
    }
    drop(vtab);
    ffi::SQLITE_OK as c_int
}

unsafe extern "C" fn x_rename(p_vtab: *mut sqlite3_vtab, z_new: *const c_char) -> c_int {
    let vtab = &mut *(p_vtab as *mut VTab);
    let new_table = cstr(z_new).to_string();
    let old_key = vtab.key();
    let new_key = (vtab.schema.clone(), new_table.clone());
    vtab.registry_mut().rename(&old_key, new_key);
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
    *pp_cursor = Box::into_raw(cursor) as *mut sqlite3_vtab_cursor;
    ffi::SQLITE_OK as c_int
}

unsafe extern "C" fn x_close(p_cur: *mut sqlite3_vtab_cursor) -> c_int {
    drop(Box::from_raw(p_cur as *mut Cursor));
    ffi::SQLITE_OK as c_int
}

unsafe extern "C" fn x_eof(p_cur: *mut sqlite3_vtab_cursor) -> c_int {
    let cursor = &*(p_cur as *mut Cursor);
    (cursor.current >= cursor.result.len()) as c_int
}

unsafe extern "C" fn x_next(p_cur: *mut sqlite3_vtab_cursor) -> c_int {
    let cursor = &mut *(p_cur as *mut Cursor);
    if cursor.current < cursor.result.len() {
        cursor.current += 1;
    }
    ffi::SQLITE_OK as c_int
}

unsafe extern "C" fn x_rowid(p_cur: *mut sqlite3_vtab_cursor, p_rowid: *mut i64) -> c_int {
    let cursor = &*(p_cur as *mut Cursor);
    if cursor.current < cursor.result.len() {
        *p_rowid = cursor.result[cursor.current].1 as i64;
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
    let cursor = &*(p_cur as *mut Cursor);
    if cursor.current >= cursor.result.len() {
        return ffi::SQLITE_ERROR as c_int;
    }
    let (dist, rowid) = cursor.result[cursor.current];

    if n == COL_DISTANCE {
        ffi::result_double(ctx, dist as f64);
        return ffi::SQLITE_OK as c_int;
    }
    if n == COL_VECTOR {
        let vtab = &*(cursor.base.pVtab as *mut VTab);
        match vtab.entry().index.get_vector(rowid) {
            Some(v) => {
                ffi::result_blob(ctx, &vector::f32_to_blob(&v));
                ffi::SQLITE_OK as c_int
            }
            None => {
                ffi::result_error(ctx, &format!("Can't find vector with rowid {}", rowid));
                ffi::SQLITE_ERROR as c_int
            }
        }
    } else if n == COL_OPERATION || n == COL_PATH {
        ffi::result_null(ctx);
        ffi::SQLITE_OK as c_int
    } else {
        ffi::result_error(ctx, &format!("Invalid column index: {}", n));
        ffi::SQLITE_ERROR as c_int
    }
}

// ---- best index ----

unsafe extern "C" fn x_best_index(
    p_vtab: *mut sqlite3_vtab,
    info: *mut sqlite3_index_info,
) -> c_int {
    let info = &mut *info;
    let mut argv_index = 0;
    let mut short_names: Vec<&str> = Vec::new();

    for i in 0..info.nConstraint as isize {
        let constraint = &*info.aConstraint.offset(i);
        if constraint.usable == 0 {
            continue;
        }
        let column = constraint.iColumn;
        let usage = &mut *info.aConstraintUsage.offset(i);

        if constraint.op as c_int == FUNC_KNN && column == COL_VECTOR {
            argv_index += 1;
            usage.argvIndex = argv_index;
            usage.omit = 1;
            short_names.push(SN_KNN);
            info.estimatedCost = 100.0;
        } else if column == -1 {
            if ffi::libversion_number() < 3038000 {
                set_vtab_err(p_vtab, "SQLite version is too old: sqlite version 3.38.0 or higher is required.");
                return ffi::SQLITE_ERROR as c_int;
            }
            if constraint.op as c_int == ffi::SQLITE_INDEX_CONSTRAINT_EQ as c_int {
                let can_vtab_in = ffi::vtab_in(info, i as c_int, 1) != 0;
                argv_index += 1;
                usage.argvIndex = argv_index;
                usage.omit = 1;
                if can_vtab_in {
                    short_names.push(SN_IN);
                    info.estimatedCost = 200.0;
                } else {
                    short_names.push(SN_EQ);
                    info.estimatedCost = 100.0;
                }
            }
        }
    }

    if short_names.is_empty() {
        set_vtab_err(p_vtab, "No valid constraint found in where clause");
        return ffi::SQLITE_CONSTRAINT as c_int;
    }

    let idx_str: String = short_names.concat();
    let p = ffi::sqlite_strdup(&idx_str);
    if p.is_null() {
        set_vtab_err(p_vtab, "Failed to allocate memory for idxStr");
        return ffi::SQLITE_NOMEM as c_int;
    }
    info.idxStr = p;
    info.needToFreeIdxStr = 1;
    info.idxNum = (short_names.len() * 2) as c_int;
    ffi::SQLITE_OK as c_int
}

// ---- filter ----

unsafe extern "C" fn x_filter(
    p_cur: *mut sqlite3_vtab_cursor,
    idx_num: c_int,
    idx_str: *const c_char,
    _argc: c_int,
    argv: *mut *mut sqlite3_value,
) -> c_int {
    let cursor = &mut *(p_cur as *mut Cursor);
    let p_vtab = cursor.base.pVtab;
    let vtab = &*(p_vtab as *mut VTab);
    let entry = vtab.entry();

    let codes: Vec<u8> = if idx_str.is_null() || idx_num <= 0 {
        Vec::new()
    } else {
        std::slice::from_raw_parts(idx_str as *const u8, idx_num as usize).to_vec()
    };

    let mut knn: Option<&KnnParam> = None;
    let mut rowid_in: Option<Vec<u64>> = None;
    let mut rowid_eq: Option<u64> = None;

    let mut i = 0usize;
    let mut arg_index = 0usize;
    while i + 1 < codes.len() {
        let code = [codes[i], codes[i + 1]];
        let value = uarg(argv, arg_index);
        match &code {
            b"ks" => {
                let p = ffi::value_pointer(value, KNN_PARAM_TYPE.as_ptr() as *const c_char)
                    as *const KnnParam;
                if p.is_null() {
                    set_vtab_err(
                        p_vtab,
                        "Failed to materialize constraint: knn_param() should be used for the 2nd param of knn_search()",
                    );
                    return ffi::SQLITE_ERROR as c_int;
                }
                if knn.is_some() {
                    set_vtab_err(p_vtab, "only one knn_search constraint is allowed");
                    return ffi::SQLITE_ERROR as c_int;
                }
                knn = Some(&*p);
            }
            b"in" => {
                if rowid_in.is_some() || rowid_eq.is_some() {
                    set_vtab_err(p_vtab, "only one rowid constraint is allowed");
                    return ffi::SQLITE_ERROR as c_int;
                }
                let mut ids = Vec::new();
                let mut rowid_value: *mut sqlite3_value = std::ptr::null_mut();
                let mut rc = ffi::vtab_in_first(value, &mut rowid_value);
                while rc == ffi::SQLITE_OK as c_int && !rowid_value.is_null() {
                    if ffi::value_type(rowid_value) != ffi::SQLITE_INTEGER as c_int {
                        set_vtab_err(p_vtab, "rowid must be of type INTEGER");
                        return ffi::SQLITE_ERROR as c_int;
                    }
                    ids.push(ffi::value_int64(rowid_value) as u64);
                    rc = ffi::vtab_in_next(value, &mut rowid_value);
                }
                rowid_in = Some(ids);
            }
            b"eq" => {
                if rowid_in.is_some() || rowid_eq.is_some() {
                    set_vtab_err(p_vtab, "only one rowid constraint is allowed");
                    return ffi::SQLITE_ERROR as c_int;
                }
                if ffi::value_type(value) != ffi::SQLITE_INTEGER as c_int {
                    set_vtab_err(p_vtab, "rowid must be of type INTEGER");
                    return ffi::SQLITE_ERROR as c_int;
                }
                rowid_eq = Some(ffi::value_int64(value) as u64);
            }
            _ => {
                set_vtab_err(p_vtab, "unknown constraint short name");
                return ffi::SQLITE_ERROR as c_int;
            }
        }
        i += 2;
        arg_index += 1;
    }

    let result: Result<Vec<(f32, u64)>, String> = if let Some(knn) = knn {
        if knn.query_vector.len() != entry.space.dim {
            set_vtab_err(
                p_vtab,
                &format!(
                    "query vector's dimension({}) doesn't match {}'s dimension: {}",
                    knn.query_vector.len(),
                    entry.space.vector_name,
                    entry.space.dim
                ),
            );
            return ffi::SQLITE_ERROR as c_int;
        }
        let filter = if let Some(ref ids) = rowid_in {
            SearchFilter::In(ids)
        } else if let Some(eq) = rowid_eq {
            SearchFilter::Equals(eq)
        } else {
            SearchFilter::None
        };
        entry
            .index
            .search(&knn.query_vector, knn.k as usize, knn.ef.map(|e| e as usize), filter)
    } else {
        let mut out = Vec::new();
        if let Some(ids) = rowid_in {
            for id in ids {
                if entry.index.contains(id) {
                    out.push((0.0f32, id));
                }
            }
        } else if let Some(eq) = rowid_eq {
            if entry.index.contains(eq) {
                out.push((0.0f32, eq));
            }
        }
        Ok(out)
    };

    match result {
        Ok(rows) => {
            cursor.result = rows;
            cursor.current = 0;
            ffi::SQLITE_OK as c_int
        }
        Err(e) => {
            set_vtab_err(p_vtab, &format!("Failed to execute query due to: {}", e));
            ffi::SQLITE_ERROR as c_int
        }
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
    if cstr(z_name) == "knn_search" {
        *px_func = Some(scalar::knn_search);
        *pp_arg = std::ptr::null_mut();
        return FUNC_KNN;
    }
    0
}

// ---- update (insert / delete / update / persistence) ----

unsafe fn execute_persistence(vtab: &VTab, p_vtab: *mut sqlite3_vtab, argv: *mut *mut sqlite3_value) -> c_int {
    let op_value = uarg(argv, (2 + COL_OPERATION) as usize);
    let operation = ffi::value_text_string(op_value);

    let path_value = uarg(argv, (2 + COL_PATH) as usize);
    if ffi::value_type(path_value) != ffi::SQLITE_TEXT as c_int {
        set_vtab_err(
            p_vtab,
            &format!("path must be provided as TEXT for '{}' operation", operation),
        );
        return ffi::SQLITE_ERROR as c_int;
    }
    let path = ffi::value_text_string(path_value);

    let entry = vtab.entry();
    let result = match operation.as_str() {
        "save" => entry.index.save(&path),
        "load" => entry.index.load(&path),
        _ => {
            set_vtab_err(
                p_vtab,
                &format!("unknown operation '{}'; expected 'save' or 'load'", operation),
            );
            return ffi::SQLITE_ERROR as c_int;
        }
    };
    match result {
        Ok(()) => ffi::SQLITE_OK as c_int,
        Err(e) => {
            set_vtab_err(p_vtab, &format!("{} failed: {}", operation, e));
            ffi::SQLITE_ERROR as c_int
        }
    }
}

unsafe fn insert_or_update_vector(
    entry: &IndexEntry,
    p_vtab: *mut sqlite3_vtab,
    value: *mut sqlite3_value,
    rowid: u64,
) -> c_int {
    if ffi::value_type(value) != ffi::SQLITE_BLOB as c_int {
        set_vtab_err(p_vtab, "vector must be of type Blob");
        return ffi::SQLITE_ERROR as c_int;
    }
    let blob = ffi::value_blob_slice(value);
    let vec = match vector::blob_to_f32(&blob) {
        Ok(v) => v,
        Err(e) => {
            set_vtab_err(p_vtab, &format!("Failed to perform insertion due to: {}", e));
            return ffi::SQLITE_ERROR as c_int;
        }
    };
    if vec.len() != entry.space.dim {
        set_vtab_err(
            p_vtab,
            &format!(
                "Dimension mismatch: vector's dimension {}, table's dimension {}",
                vec.len(),
                entry.space.dim
            ),
        );
        return ffi::SQLITE_ERROR as c_int;
    }
    match entry.index.add(&vec, rowid) {
        Ok(()) => ffi::SQLITE_OK as c_int,
        Err(e) => {
            set_vtab_err(p_vtab, &format!("Failed to insert row {} due to: {}", rowid, e));
            ffi::SQLITE_ERROR as c_int
        }
    }
}

unsafe extern "C" fn x_update(
    p_vtab: *mut sqlite3_vtab,
    argc: c_int,
    argv: *mut *mut sqlite3_value,
    p_rowid: *mut i64,
) -> c_int {
    let vtab = &*(p_vtab as *mut VTab);
    let argv0_type = ffi::value_type(uarg(argv, 0));
    let null = ffi::SQLITE_NULL as c_int;
    let integer = ffi::SQLITE_INTEGER as c_int;

    if argc > 1 && argv0_type == null {
        // INSERT.
        // A non-NULL operation column means a save/load command, not a vector.
        if ffi::value_type(uarg(argv, (2 + COL_OPERATION) as usize)) == ffi::SQLITE_TEXT as c_int {
            *p_rowid = 0;
            return execute_persistence(vtab, p_vtab, argv);
        }
        if ffi::value_type(uarg(argv, 1)) == null {
            set_vtab_err(p_vtab, "rowid must be specified during insertion");
            return ffi::SQLITE_ERROR as c_int;
        }
        let raw_rowid = ffi::value_int64(uarg(argv, 1));
        if raw_rowid < 0 {
            set_vtab_err(p_vtab, &format!("rowid {} out of range", raw_rowid));
            return ffi::SQLITE_ERROR as c_int;
        }
        let rowid = raw_rowid as u64;
        *p_rowid = raw_rowid;

        let entry = vtab.entry();
        if entry.index.contains(rowid) {
            set_vtab_err(p_vtab, &format!("row {} already exists", rowid));
            return ffi::SQLITE_ERROR as c_int;
        }
        insert_or_update_vector(entry, p_vtab, uarg(argv, 2), rowid)
    } else if argc == 1 && argv0_type != null {
        // DELETE.
        let raw_rowid = ffi::value_int64(uarg(argv, 0));
        if raw_rowid < 0 {
            set_vtab_err(p_vtab, &format!("rowid {} out of range", raw_rowid));
            return ffi::SQLITE_ERROR as c_int;
        }
        let entry = vtab.entry();
        match entry.index.mark_delete(raw_rowid as u64) {
            Ok(()) => ffi::SQLITE_OK as c_int,
            Err(e) => {
                set_vtab_err(p_vtab, &format!("Delete failed with rowid {}: {}", raw_rowid, e));
                ffi::SQLITE_ERROR as c_int
            }
        }
    } else if argc > 1 && argv0_type != null {
        // UPDATE.
        if argv0_type != integer {
            set_vtab_err(p_vtab, "rowid must be of type INTEGER");
            return ffi::SQLITE_ERROR as c_int;
        }
        if ffi::value_type(uarg(argv, 1)) != integer {
            set_vtab_err(p_vtab, "target rowid must be of type INTEGER");
            return ffi::SQLITE_ERROR as c_int;
        }
        let source_rowid = ffi::value_int64(uarg(argv, 0));
        let target_rowid = ffi::value_int64(uarg(argv, 1));
        if source_rowid != target_rowid {
            set_vtab_err(p_vtab, "rowid cannot be changed");
            return ffi::SQLITE_ERROR as c_int;
        }
        if source_rowid < 0 {
            set_vtab_err(p_vtab, &format!("rowid {} out of range", source_rowid));
            return ffi::SQLITE_ERROR as c_int;
        }
        let rowid = source_rowid as u64;
        let entry = vtab.entry();
        if !entry.index.contains(rowid) {
            set_vtab_err(p_vtab, &format!("rowid {} not found", source_rowid));
            return ffi::SQLITE_ERROR as c_int;
        }
        insert_or_update_vector(entry, p_vtab, uarg(argv, 2), rowid)
    } else {
        set_vtab_err(p_vtab, "Operation not supported for now");
        ffi::SQLITE_ERROR as c_int
    }
}

// ---- module definition ----

struct ModuleWrap(sqlite3_module);
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
    &MODULE.0 as *const sqlite3_module
}
