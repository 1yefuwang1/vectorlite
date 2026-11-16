//! Native SQLite host probe for batch INSERT under four ambient Tokio contexts.
//!
//! Run with the path to the already built loadable extension. This independent
//! executable links host SQLite, not the extension crate. The dynamically loaded
//! Canonical ingestion must succeed in all contexts. Deferred native construction
//! uses a second connection and must also succeed; no SQLite handle is sent to a task.
//!
//! Every connection and statement stays on its original thread. Runtime::block_on
//! polls its top-level future there, even with the multi-thread runtime flavor.
//! No SQLite handle, descriptor, buffer or extension pointer is sent to a task.

#![deny(unsafe_op_in_unsafe_fn)]
#![deny(clippy::undocumented_unsafe_blocks, clippy::missing_safety_doc)]

use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::mem::{offset_of, size_of};
use std::os::raw::{c_char, c_int, c_void};
use std::path::Path;
use std::ptr;
use std::rc::Rc;
use std::thread::{self, ThreadId};

const SQLITE_OK: c_int = 0;
const SQLITE_ROW: c_int = 100;
const SQLITE_DONE: c_int = 101;
const TAG: &[u8] = b"vectorlite.batch.f32.v1\0";
const COUNT: usize = 32;
const DIMENSION: usize = 2;
type Result<T> = std::result::Result<T, String>;

// Opaque handles are used solely by the linked native SQLite implementation.
#[repr(C)]
struct Sqlite {
    _opaque: [u8; 0],
}
#[repr(C)]
struct SqliteStatement {
    _opaque: [u8; 0],
}

#[link(name = "sqlite3", kind = "static")]
extern "C" {
    fn sqlite3_libversion_number() -> c_int;
    fn sqlite3_open(filename: *const c_char, output: *mut *mut Sqlite) -> c_int;
    fn sqlite3_close(db: *mut Sqlite) -> c_int;
    fn sqlite3_extended_result_codes(db: *mut Sqlite, enabled: c_int) -> c_int;
    fn sqlite3_enable_load_extension(db: *mut Sqlite, enabled: c_int) -> c_int;
    fn sqlite3_load_extension(
        db: *mut Sqlite,
        filename: *const c_char,
        entry: *const c_char,
        error: *mut *mut c_char,
    ) -> c_int;
    fn sqlite3_errmsg(db: *mut Sqlite) -> *const c_char;
    fn sqlite3_free(pointer: *mut c_void);
    fn sqlite3_prepare_v2(
        db: *mut Sqlite,
        sql: *const c_char,
        length: c_int,
        output: *mut *mut SqliteStatement,
        tail: *mut *const c_char,
    ) -> c_int;
    fn sqlite3_step(statement: *mut SqliteStatement) -> c_int;
    fn sqlite3_finalize(statement: *mut SqliteStatement) -> c_int;
    fn sqlite3_bind_pointer(
        statement: *mut SqliteStatement,
        index: c_int,
        pointer: *mut c_void,
        tag: *const c_char,
        destructor: Option<unsafe extern "C" fn(*mut c_void)>,
    ) -> c_int;
    fn sqlite3_column_int64(statement: *mut SqliteStatement, column: c_int) -> i64;
    fn sqlite3_column_text(statement: *mut SqliteStatement, column: c_int) -> *const u8;
}

#[repr(C)]
struct BatchF32V1 {
    abi_version: u32,
    struct_size: u32,
    count: u64,
    dimension: u64,
    vectors: *const f32,
    rowids: *const i64,
}

struct Connection {
    pointer: *mut Sqlite,
    owner: ThreadId,
    _local: PhantomData<Rc<()>>,
}

impl Connection {
    fn open(extension: &Path, database: &Path) -> Result<Self> {
        // SAFETY: this no-argument routine reads the linked host's SQLite version.
        let version = unsafe { sqlite3_libversion_number() };
        if version < 3_038_000 {
            return Err(format!("native host requires SQLite >=3.38, got {version}"));
        }
        let filename = CString::new(database.to_str().ok_or("database path must be UTF-8")?)
            .map_err(|e| e.to_string())?;
        let mut pointer = ptr::null_mut();
        // SAFETY: filename is NUL-terminated; output is a live writable slot.
        let status = unsafe { sqlite3_open(filename.as_ptr(), &mut pointer) };
        if pointer.is_null() {
            return Err(format!("sqlite3_open returned null ({status})"));
        }
        let connection = Self {
            pointer,
            owner: thread::current().id(),
            _local: PhantomData,
        };
        connection.check(status)?;
        // SAFETY: this live uniquely owned connection stays on its owner thread.
        connection.check(unsafe { sqlite3_extended_result_codes(pointer, 1) })?;
        // SAFETY: same live connection, with extension loading explicitly enabled.
        connection.check(unsafe { sqlite3_enable_load_extension(pointer, 1) })?;
        let extension = extension
            .to_str()
            .ok_or_else(|| "extension path must be UTF-8 for this host probe".to_string())?;
        let extension = CString::new(extension).map_err(|error| error.to_string())?;
        let mut error = ptr::null_mut();
        // SAFETY: filename and writable error slot are live; null entry lets
        // SQLite resolve the normal extension entry point. Connection is owned.
        let status =
            unsafe { sqlite3_load_extension(pointer, extension.as_ptr(), ptr::null(), &mut error) };
        if !error.is_null() {
            // SAFETY: SQLite returned an allocated NUL-terminated error message;
            // copy it before releasing it with the matching host allocator.
            let message = unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned();
            // SAFETY: this allocation belongs to this linked SQLite allocator.
            unsafe { sqlite3_free(error.cast()) };
            if status != SQLITE_OK {
                return Err(format!("extension load failed ({status}): {message}"));
            }
        }
        connection.check(status)?;
        // SAFETY: extension loading is disabled again on this same live handle.
        connection.check(unsafe { sqlite3_enable_load_extension(pointer, 0) })?;
        Ok(connection)
    }

    fn assert_owner(&self) {
        assert_eq!(
            self.owner,
            thread::current().id(),
            "SQLite owner thread changed"
        );
    }

    fn message(&self) -> String {
        self.assert_owner();
        // SAFETY: live connection owns the NUL-terminated error string; copy it
        // synchronously before any further SQLite call can invalidate it.
        unsafe { CStr::from_ptr(sqlite3_errmsg(self.pointer)) }
            .to_string_lossy()
            .into_owned()
    }

    fn check(&self, status: c_int) -> Result<()> {
        self.assert_owner();
        if status == SQLITE_OK {
            Ok(())
        } else {
            Err(format!("SQLite error {status}: {}", self.message()))
        }
    }

    fn prepare(&self, sql: &str) -> Result<Statement<'_>> {
        self.assert_owner();
        let sql = CString::new(sql).map_err(|error| error.to_string())?;
        let mut pointer = ptr::null_mut();
        // SAFETY: SQLite copies/prepares the live NUL-terminated SQL during this
        // call. Output is writable; statement cannot outlive the owned connection.
        let status = unsafe {
            sqlite3_prepare_v2(
                self.pointer,
                sql.as_ptr(),
                -1,
                &mut pointer,
                ptr::null_mut(),
            )
        };
        self.check(status)?;
        if pointer.is_null() {
            return Err("expected a nonempty SQL statement".to_string());
        }
        Ok(Statement {
            pointer,
            connection: self,
        })
    }

    fn execute(&self, sql: &str) -> Result<()> {
        let statement = self.prepare(sql)?;
        match statement.step() {
            SQLITE_DONE => statement.finish(),
            status => Err(format!("SQL failed ({status}): {}: {sql}", self.message())),
        }
    }

    fn integer(&self, sql: &str) -> Result<i64> {
        let statement = self.prepare(sql)?;
        let status = statement.step();
        if status != SQLITE_ROW {
            return Err(format!(
                "expected scalar row ({status}): {}",
                self.message()
            ));
        }
        // SAFETY: step returned ROW; this query has its requested integer column.
        let value = unsafe { sqlite3_column_int64(statement.pointer, 0) };
        let status = statement.step();
        if status != SQLITE_DONE {
            return Err(format!("expected one scalar row, got status {status}"));
        }
        statement.finish()?;
        Ok(value)
    }
    fn text(&self, sql: &str) -> Result<String> {
        let statement = self.prepare(sql)?;
        let status = statement.step();
        if status != SQLITE_ROW {
            return Err(format!(
                "expected text row ({status}): {}: {sql}",
                self.message()
            ));
        }
        // SAFETY: a live ROW exposes the SQLite-owned text until the next step;
        // copy it while the statement and connection remain on their owner thread.
        let value = unsafe {
            let pointer = sqlite3_column_text(statement.pointer, 0);
            if pointer.is_null() {
                return Err("unexpected NULL text".to_string());
            }
            CStr::from_ptr(pointer.cast())
                .to_string_lossy()
                .into_owned()
        };
        if statement.step() != SQLITE_DONE {
            return Err("expected one text row".to_string());
        }
        statement.finish()?;
        Ok(value)
    }
}

impl Drop for Connection {
    fn drop(&mut self) {
        self.assert_owner();
        // SAFETY: every statement borrows this owner and has already finalized;
        // close releases exactly this host connection on its original thread.
        let status = unsafe { sqlite3_close(self.pointer) };
        assert_eq!(status, SQLITE_OK, "SQLite connection did not close cleanly");
    }
}

struct Statement<'a> {
    pointer: *mut SqliteStatement,
    connection: &'a Connection,
}

impl Statement<'_> {
    fn step(&self) -> c_int {
        self.connection.assert_owner();
        // SAFETY: statement is live, prepared by this connection on its owner
        // thread. All bound batch objects, when present, remain alive and immutable.
        unsafe { sqlite3_step(self.pointer) }
    }

    fn finish(mut self) -> Result<()> {
        self.connection.assert_owner();
        // SAFETY: consume this uniquely owned statement exactly once; caller
        // retains any bound descriptor and buffers until this finalize returns.
        let status = unsafe { sqlite3_finalize(self.pointer) };
        self.pointer = ptr::null_mut();
        self.connection.check(status)
    }
}

impl Drop for Statement<'_> {
    fn drop(&mut self) {
        if !self.pointer.is_null() {
            self.connection.assert_owner();
            // SAFETY: failure paths still own this live statement. Finalization
            // occurs while its caller's earlier-declared bound buffers are alive.
            unsafe { sqlite3_finalize(self.pointer) };
        }
    }
}

fn batch_insert(connection: &Connection) -> Result<Option<(c_int, String)>> {
    let mut vectors = [0.0f32; COUNT * DIMENSION];
    let mut rowids = [0i64; COUNT];
    for index in 0..COUNT {
        vectors[index * DIMENSION] = index as f32 + 1.0;
        vectors[index * DIMENSION + 1] = ((index * 7) % 11) as f32 * 0.1;
        rowids[index] = index as i64 + 1;
    }
    // From here through statement finalize these initialized native arrays and
    // the aligned descriptor stay readable and immutable on this owner thread.
    let mut descriptor = BatchF32V1 {
        abi_version: 1,
        struct_size: size_of::<BatchF32V1>() as u32,
        count: COUNT as u64,
        dimension: DIMENSION as u64,
        vectors: vectors.as_ptr(),
        rowids: rowids.as_ptr(),
    };
    // Declare statement after its buffers so every early-return Drop finalizes
    // the binding before descriptor or arrays can leave scope. Reset is not used.
    let statement = connection.prepare(
        "INSERT INTO v(operation,embedding,path) VALUES('insert_batch',?1,'{\"batch_size\":8}')",
    )?;
    // SAFETY: bind the DESCRIPTOR address, with static NUL-terminated exact tag.
    // Null destructor leaves all ownership here. Both arrays are aligned single
    // allocations of count*dimension f32 and count i64, alive/immutable until
    // this statement finalizes. No integer-to-pointer or BLOB backdoor is used.
    connection.check(unsafe {
        sqlite3_bind_pointer(
            statement.pointer,
            1,
            ptr::from_mut(&mut descriptor).cast(),
            TAG.as_ptr().cast(),
            None,
        )
    })?;
    let status = statement.step();
    if status == SQLITE_DONE {
        statement.finish()?;
        Ok(None)
    } else {
        let message = connection.message();
        // SQLite finalization repeats the prior step error. It must agree with
        // step, and the guard must finalize before any native buffer dies.
        let finalize = statement.finish();
        if finalize.is_ok() {
            return Err(format!(
                "step {status} failed but finalize unexpectedly succeeded"
            ));
        }
        Ok(Some((status, message)))
    }
}

fn run_case(extension: &Path, directory: &Path, mode: &str) -> Result<()> {
    println!(
        "START {mode}; host_tokio_entered={}",
        tokio::runtime::Handle::try_current().is_ok()
    );
    let database = directory.join(format!("tokio-{mode}.db"));
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&database)
        .map_err(|error| format!("reserve {}: {error}", database.display()))?;
    let connection = Connection::open(extension, &database)?;
    connection.execute("CREATE VIRTUAL TABLE v USING vectorlite(embedding float32[2] l2,diskann(degree=8,build_list_size=32,search_list_size=32,max_visits=65536))")?;
    connection.execute("INSERT INTO v(rowid,embedding) VALUES(10000,vector_from_json('[0,0]'))")?;
    if let Some((status, message)) = batch_insert(&connection)? {
        return Err(format!(
            "{mode}: canonical batch rejected ({status}): {message}"
        ));
    }
    let canonical = connection.integer("SELECT canonical_count FROM v_diskann_meta")?;
    let entries = connection.integer("SELECT count(*) FROM v_diskann_entries")?;
    let current = connection.integer("SELECT count(*) FROM v_diskann_current")?;
    if (canonical, entries, current) != (33, 33, 33) {
        return Err(format!(
            "{mode}: wrong counts {canonical}/{entries}/{current}"
        ));
    }
    let search = "SELECT count(*) FROM v WHERE knn_search(embedding,knn_param(vector_from_json('[0,0]'),33))";
    if connection.integer(search)? != 0 {
        return Err("pending rows visible before publication".into());
    }
    // The second connection owns the entire fenced maintenance session. It and all
    // statements stay on this thread; the extension's native worker gets no SQLite.
    let worker = Connection::open(extension, &database)?;
    worker.execute("INSERT INTO v(operation,path) VALUES('index_prepare','{}')")?;
    let status = |key: &str| {
        format!("SELECT json_extract(vectorlite_index_status('main','v'),'$.current_job.{key}')")
    };
    let id = worker.integer(&status("job_id"))?;
    let fence = worker.integer(&status("fence"))?;
    let ids = format!("\"job_id\":{id},\"fence\":{fence}");
    let control = |op: &str, options: &str| {
        worker.execute(&format!(
            "INSERT INTO v(operation,path) VALUES('{op}','{{{options}}}')"
        ))
    };
    for _ in 0..100 {
        if worker.integer(&status("phase"))? != 0 {
            break;
        }
        control("index_snapshot_step", &format!("{ids},\"max_records\":8"))?;
    }
    if worker.integer(&status("phase"))? != 1 {
        return Err("snapshot did not finish".into());
    }
    control("index_claim_build", &ids)?;
    let token = worker.text(&format!("SELECT json_extract(vectorlite_index_build('main','v','{{{ids},\"memory_budget_bytes\":268435456}}'),'$.candidate_token')"))?;
    if !token.starts_with("cand:")
        || !token
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b":._-".contains(&c))
    {
        return Err("invalid candidate token".into());
    }
    let options = format!("{ids},\"candidate_token\":\"{token}\",\"max_records\":8");
    for (phase, op) in [(2, "index_import_step"), (3, "index_validate_step")] {
        for _ in 0..100 {
            if worker.integer(&status("phase"))? != phase {
                break;
            }
            control(op, &options)?;
        }
        if worker.integer(&status("phase"))? == phase {
            return Err(format!("{op} did not finish"));
        }
    }
    if worker.integer(&status("phase"))? != 4 {
        return Err("candidate not ready to publish".into());
    }
    control(
        "index_publish",
        &format!("{ids},\"candidate_token\":\"{token}\""),
    )?;
    control("index_cleanup_step", "")?;
    if connection.integer(search)? != 33 {
        return Err("published population mismatch after cleanup".into());
    }
    println!(
        "PASS {mode}: 33 canonical and published rows; owner_thread={:?}",
        connection.owner
    );
    Ok(())
}

fn verify_layout() -> Result<()> {
    if size_of::<usize>() != 8
        || size_of::<BatchF32V1>() != 40
        || offset_of!(BatchF32V1, abi_version) != 0
        || offset_of!(BatchF32V1, struct_size) != 4
        || offset_of!(BatchF32V1, count) != 8
        || offset_of!(BatchF32V1, dimension) != 16
        || offset_of!(BatchF32V1, vectors) != 24
        || offset_of!(BatchF32V1, rowids) != 32
    {
        return Err("native fixture requires the official 40-byte 64-bit batch ABI".to_string());
    }
    Ok(())
}

fn run() -> Result<()> {
    verify_layout()?;
    let mut arguments = std::env::args_os();
    let executable = arguments.next().unwrap_or_default();
    let extension = arguments.next().ok_or_else(|| {
        format!(
            "usage: {} /absolute/path/to/vectorlite.extension artifact-directory",
            Path::new(&executable).display()
        )
    })?;
    let directory = arguments
        .next()
        .ok_or("expected artifact directory argument")?;
    if arguments.next().is_some() {
        return Err("expected extension and artifact directory".to_string());
    }
    let directory = Path::new(&directory);
    let extension = Path::new(&extension)
        .canonicalize()
        .map_err(|error| error.to_string())?;
    println!("Native host Tokio=1.53.2; shared-extension TLS visibility is not assumed.");
    run_case(&extension, directory, "outside_runtime")?;
    {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .map_err(|error| error.to_string())?;
        let _entered = runtime.enter();
        run_case(&extension, directory, "entered_current_thread")?;
    }
    {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .map_err(|error| error.to_string())?;
        runtime.block_on(async { run_case(&extension, directory, "block_on_current_thread") })?;
    }
    {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .build()
            .map_err(|error| error.to_string())?;
        // Top-level future polling remains on this caller, not on worker tasks.
        runtime.block_on(async { run_case(&extension, directory, "block_on_multi_thread") })?;
    }
    println!("PASS: all four native host contexts returned without abort or partial mutation");
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("FAIL: {error}");
        std::process::exit(1);
    }
}
