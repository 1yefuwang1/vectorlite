#include "sqlite3ext.h"

SQLITE_EXTENSION_INIT1;
/* Hello World function */
static void HelloWorld(sqlite3_context *context, int argc, sqlite3_value **argv) {
    sqlite3_result_text(context, "Hello, World!", -1, SQLITE_STATIC);
}

#ifdef __cplusplus
extern "C" {
#endif

int sqlite3_extension_init(
    sqlite3 *db,
    char **pzErrMsg,
    const sqlite3_api_routines *pApi
) {
    int rc = SQLITE_OK;
    SQLITE_EXTENSION_INIT2(pApi);
    
    rc = sqlite3_create_function(
        db,
        "hello_world",
        0,
        SQLITE_UTF8,
        0,
        HelloWorld,
        0,
        0
    );

    if (rc != SQLITE_OK) {
        *pzErrMsg = sqlite3_mprintf("Failed to create hello_world function: %s", sqlite3_errstr(rc));
    }

    return rc;
}

#ifdef __cplusplus
}
#endif