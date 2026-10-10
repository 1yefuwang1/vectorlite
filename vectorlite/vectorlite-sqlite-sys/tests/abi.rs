#![cfg(feature = "abi-check")]
use vectorlite_sqlite_sys::*;
extern "C" {
    fn vectorlite_sqlite_abi_layout(index: usize) -> usize;
}
macro_rules! offset {
    ($ty:ty, $field:ident) => {{
        let value = std::mem::MaybeUninit::<$ty>::uninit();
        let base = value.as_ptr();
        // SAFETY: addr_of! forms a field address without reading uninitialized data.
        unsafe { std::ptr::addr_of!((*base).$field) as usize - base as usize }
    }};
}
#[test]
fn used_bindings_match_native_sqlite_headers() {
    let rust_layout = [
        ("sizeof(sqlite3_vtab)", std::mem::size_of::<sqlite3_vtab>()),
        ("sqlite3_vtab.pModule", offset!(sqlite3_vtab, pModule)),
        ("sqlite3_vtab.nRef", offset!(sqlite3_vtab, nRef)),
        ("sqlite3_vtab.zErrMsg", offset!(sqlite3_vtab, zErrMsg)),
        (
            "sizeof(sqlite3_vtab_cursor)",
            std::mem::size_of::<sqlite3_vtab_cursor>(),
        ),
        (
            "sqlite3_vtab_cursor.pVtab",
            offset!(sqlite3_vtab_cursor, pVtab),
        ),
        (
            "sizeof(sqlite3_module)",
            std::mem::size_of::<sqlite3_module>(),
        ),
        ("sqlite3_module.iVersion", offset!(sqlite3_module, iVersion)),
        ("sqlite3_module.xCreate", offset!(sqlite3_module, xCreate)),
        ("sqlite3_module.xConnect", offset!(sqlite3_module, xConnect)),
        (
            "sqlite3_module.xBestIndex",
            offset!(sqlite3_module, xBestIndex),
        ),
        (
            "sqlite3_module.xDisconnect",
            offset!(sqlite3_module, xDisconnect),
        ),
        ("sqlite3_module.xDestroy", offset!(sqlite3_module, xDestroy)),
        ("sqlite3_module.xOpen", offset!(sqlite3_module, xOpen)),
        ("sqlite3_module.xClose", offset!(sqlite3_module, xClose)),
        ("sqlite3_module.xFilter", offset!(sqlite3_module, xFilter)),
        ("sqlite3_module.xNext", offset!(sqlite3_module, xNext)),
        ("sqlite3_module.xEof", offset!(sqlite3_module, xEof)),
        ("sqlite3_module.xColumn", offset!(sqlite3_module, xColumn)),
        ("sqlite3_module.xRowid", offset!(sqlite3_module, xRowid)),
        ("sqlite3_module.xUpdate", offset!(sqlite3_module, xUpdate)),
        (
            "sqlite3_module.xFindFunction",
            offset!(sqlite3_module, xFindFunction),
        ),
        ("sqlite3_module.xRename", offset!(sqlite3_module, xRename)),
        ("sqlite3_module.xBegin", offset!(sqlite3_module, xBegin)),
        ("sqlite3_module.xSync", offset!(sqlite3_module, xSync)),
        ("sqlite3_module.xCommit", offset!(sqlite3_module, xCommit)),
        (
            "sqlite3_module.xRollback",
            offset!(sqlite3_module, xRollback),
        ),
        (
            "sqlite3_module.xSavepoint",
            offset!(sqlite3_module, xSavepoint),
        ),
        ("sqlite3_module.xRelease", offset!(sqlite3_module, xRelease)),
        (
            "sqlite3_module.xRollbackTo",
            offset!(sqlite3_module, xRollbackTo),
        ),
        (
            "sqlite3_module.xShadowName",
            offset!(sqlite3_module, xShadowName),
        ),
        (
            "sizeof(sqlite3_index_info)",
            std::mem::size_of::<sqlite3_index_info>(),
        ),
        (
            "sqlite3_index_info.nConstraint",
            offset!(sqlite3_index_info, nConstraint),
        ),
        (
            "sqlite3_index_info.aConstraint",
            offset!(sqlite3_index_info, aConstraint),
        ),
        (
            "sqlite3_index_info.nOrderBy",
            offset!(sqlite3_index_info, nOrderBy),
        ),
        (
            "sqlite3_index_info.aOrderBy",
            offset!(sqlite3_index_info, aOrderBy),
        ),
        (
            "sqlite3_index_info.aConstraintUsage",
            offset!(sqlite3_index_info, aConstraintUsage),
        ),
        (
            "sqlite3_index_info.idxNum",
            offset!(sqlite3_index_info, idxNum),
        ),
        (
            "sqlite3_index_info.idxStr",
            offset!(sqlite3_index_info, idxStr),
        ),
        (
            "sqlite3_index_info.needToFreeIdxStr",
            offset!(sqlite3_index_info, needToFreeIdxStr),
        ),
        (
            "sqlite3_index_info.orderByConsumed",
            offset!(sqlite3_index_info, orderByConsumed),
        ),
        (
            "sqlite3_index_info.estimatedCost",
            offset!(sqlite3_index_info, estimatedCost),
        ),
        (
            "sqlite3_index_info.estimatedRows",
            offset!(sqlite3_index_info, estimatedRows),
        ),
        (
            "sqlite3_index_info.idxFlags",
            offset!(sqlite3_index_info, idxFlags),
        ),
        (
            "sqlite3_index_info.colUsed",
            offset!(sqlite3_index_info, colUsed),
        ),
        (
            "sizeof(sqlite3_api_routines)",
            std::mem::size_of::<sqlite3_api_routines>(),
        ),
        (
            "sqlite3_api_routines.libversion_number",
            offset!(sqlite3_api_routines, libversion_number),
        ),
        (
            "sqlite3_api_routines.value_type",
            offset!(sqlite3_api_routines, value_type),
        ),
        (
            "sqlite3_api_routines.value_bytes",
            offset!(sqlite3_api_routines, value_bytes),
        ),
        (
            "sqlite3_api_routines.value_blob",
            offset!(sqlite3_api_routines, value_blob),
        ),
        (
            "sqlite3_api_routines.value_text",
            offset!(sqlite3_api_routines, value_text),
        ),
        (
            "sqlite3_api_routines.value_int",
            offset!(sqlite3_api_routines, value_int),
        ),
        (
            "sqlite3_api_routines.value_int64",
            offset!(sqlite3_api_routines, value_int64),
        ),
        (
            "sqlite3_api_routines.value_pointer",
            offset!(sqlite3_api_routines, value_pointer),
        ),
        (
            "sqlite3_api_routines.result_double",
            offset!(sqlite3_api_routines, result_double),
        ),
        (
            "sqlite3_api_routines.result_null",
            offset!(sqlite3_api_routines, result_null),
        ),
        (
            "sqlite3_api_routines.result_blob",
            offset!(sqlite3_api_routines, result_blob),
        ),
        (
            "sqlite3_api_routines.result_text",
            offset!(sqlite3_api_routines, result_text),
        ),
        (
            "sqlite3_api_routines.result_error",
            offset!(sqlite3_api_routines, result_error),
        ),
        (
            "sqlite3_api_routines.result_pointer",
            offset!(sqlite3_api_routines, result_pointer),
        ),
        (
            "sqlite3_api_routines.malloc",
            offset!(sqlite3_api_routines, malloc),
        ),
        (
            "sqlite3_api_routines.free",
            offset!(sqlite3_api_routines, free),
        ),
        (
            "sqlite3_api_routines.declare_vtab",
            offset!(sqlite3_api_routines, declare_vtab),
        ),
        (
            "sqlite3_api_routines.vtab_config",
            offset!(sqlite3_api_routines, vtab_config),
        ),
        (
            "sqlite3_api_routines.create_module_v2",
            offset!(sqlite3_api_routines, create_module_v2),
        ),
        (
            "sqlite3_api_routines.create_function_v2",
            offset!(sqlite3_api_routines, create_function_v2),
        ),
        (
            "sqlite3_api_routines.vtab_in",
            offset!(sqlite3_api_routines, vtab_in),
        ),
        (
            "sqlite3_api_routines.vtab_in_first",
            offset!(sqlite3_api_routines, vtab_in_first),
        ),
        (
            "sqlite3_api_routines.vtab_in_next",
            offset!(sqlite3_api_routines, vtab_in_next),
        ),
        (
            "sqlite3_api_routines.prepare_v2",
            offset!(sqlite3_api_routines, prepare_v2),
        ),
        (
            "sqlite3_api_routines.finalize",
            offset!(sqlite3_api_routines, finalize),
        ),
        (
            "sqlite3_api_routines.bind_int64",
            offset!(sqlite3_api_routines, bind_int64),
        ),
        (
            "sqlite3_api_routines.bind_blob",
            offset!(sqlite3_api_routines, bind_blob),
        ),
        (
            "sqlite3_api_routines.bind_text",
            offset!(sqlite3_api_routines, bind_text),
        ),
        (
            "sqlite3_api_routines.step",
            offset!(sqlite3_api_routines, step),
        ),
        (
            "sqlite3_api_routines.column_count",
            offset!(sqlite3_api_routines, column_count),
        ),
        (
            "sqlite3_api_routines.column_type",
            offset!(sqlite3_api_routines, column_type),
        ),
        (
            "sqlite3_api_routines.column_int64",
            offset!(sqlite3_api_routines, column_int64),
        ),
        (
            "sqlite3_api_routines.column_blob",
            offset!(sqlite3_api_routines, column_blob),
        ),
        (
            "sqlite3_api_routines.column_bytes",
            offset!(sqlite3_api_routines, column_bytes),
        ),
        (
            "sqlite3_api_routines.column_text",
            offset!(sqlite3_api_routines, column_text),
        ),
        (
            "sqlite3_api_routines.errmsg",
            offset!(sqlite3_api_routines, errmsg),
        ),
        (
            "sqlite3_api_routines.extended_errcode",
            offset!(sqlite3_api_routines, extended_errcode),
        ),
        (
            "sqlite3_api_routines.randomness",
            offset!(sqlite3_api_routines, randomness),
        ),
        (
            "sqlite3_api_routines.changes",
            offset!(sqlite3_api_routines, changes),
        ),
        (
            "sqlite3_api_routines.bind_pointer",
            offset!(sqlite3_api_routines, bind_pointer),
        ),
        (
            "sqlite3_api_routines.context_db_handle",
            offset!(sqlite3_api_routines, context_db_handle),
        ),
        (
            "sqlite3_api_routines.result_error_code",
            offset!(sqlite3_api_routines, result_error_code),
        ),
        (
            "sqlite3_api_routines.reset",
            offset!(sqlite3_api_routines, reset),
        ),
        (
            "sqlite3_api_routines.clear_bindings",
            offset!(sqlite3_api_routines, clear_bindings),
        ),
    ];
    for (index, (name, value)) in rust_layout.iter().enumerate() {
        // SAFETY: the helper reads only its static layout array and returns a size.
        let native_value = unsafe { vectorlite_sqlite_abi_layout(index) };
        assert_eq!(*value, native_value, "{name}");
    }
}
