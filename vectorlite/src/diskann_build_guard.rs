//! Callback-thread admission and owned-worker boundary for deferred DiskANN builds.
//!
//! A standalone compute SELECT is itself busy. We permit exactly that one busy
//! statement, not zero statements and not an arbitrary extra reader. Autocommit
//! alone cannot prove this. Materialization may read briefly, but every tracked
//! reader is explicitly finalized before the all-schema NONE check and worker
//! launch. Only owned Send + 'static data/closures cross the worker boundary.
//!
//! T1 provides this seam without registering a production build function; T4/T7
//! must qualify the final schema/attempt wrapper that calls it.
#![allow(dead_code)] // T1 seam: production registration is deliberately deferred.

use std::cell::RefCell;
use std::marker::PhantomData;
use std::os::raw::c_int;
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread::{self, ThreadId};
use std::time::Duration;

use crate::ffi::{self, BuildStateApi};
use crate::index_error::IndexError;
use crate::sqlite::{Connection, Statement};

type Result<T> = std::result::Result<T, IndexError>;

/// Include this requested stack in the caller's build-memory reservation. OS
/// rounding/guard pages and allocator/RSS effects are not hard-capped by it.
pub(crate) const OWNED_WORKER_STACK_BYTES: usize = 2 * 1024 * 1024;

// Ephemeral callback reservations only, never a candidate cache or authority.
// No SQLite pointer in this list is dereferenced, and Drop removes each entry
// before its callback/borrowed connection lease ends. Connections may otherwise
// be used on different threads in different serialized callbacks.
thread_local! {
    static ACTIVE_BUILDS: RefCell<Vec<NonNull<ffi::sqlite3>>> = const { RefCell::new(Vec::new()) };
}

struct Reservation {
    db: NonNull<ffi::sqlite3>,
    _local: PhantomData<Rc<()>>,
}

impl Reservation {
    fn acquire(db: NonNull<ffi::sqlite3>) -> Result<Self> {
        ACTIVE_BUILDS.with(|active| {
            let mut active = active
                .try_borrow_mut()
                .map_err(|_| misuse("recursive DiskANN build admission"))?;
            if active.contains(&db) {
                return Err(misuse("recursive DiskANN build on this connection"));
            }
            active.try_reserve(1).map_err(|_| {
                IndexError::with_code(ffi::SQLITE_NOMEM as c_int, "cannot reserve build guard")
            })?;
            active.push(db);
            Ok(Self {
                db,
                _local: PhantomData,
            })
        })
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        ACTIVE_BUILDS.with(|active| {
            // No SQLite calls or user callbacks occur while this list is borrowed.
            let mut active = active.borrow_mut();
            if let Some(index) = active.iter().rposition(|db| *db == self.db) {
                active.remove(index);
            }
        });
    }
}

fn misuse(message: &str) -> IndexError {
    IndexError::with_code(ffi::SQLITE_MISUSE as c_int, message)
}

/// The one live build callback. Neither this guard nor its connection/readers
/// can be sent/shared with the owned native worker. It owns no SQLite connection.
pub(crate) struct BuildGuard<'db> {
    connection: Connection<'db>,
    api: BuildStateApi,
    compute_statement: NonNull<ffi::sqlite3_stmt>,
    thread: ThreadId,
    _reservation: Reservation,
}

impl<'db> BuildGuard<'db> {
    /// Call in the scalar callback before materializing any large input. All
    /// other driver cursors/transactions must already have been finished.
    pub(crate) fn enter(connection: &Connection<'db>) -> Result<Self> {
        let api = ffi::build_state_api().map_err(misuse)?;
        Self::enter_with_api(connection, api)
    }

    fn enter_with_api(connection: &Connection<'db>, api: BuildStateApi) -> Result<Self> {
        let db =
            NonNull::new(connection.raw_handle()).ok_or_else(|| misuse("null build connection"))?;
        let reservation = Reservation::acquire(db)?;
        let compute_statement = transaction_free_statement(connection, api)?;
        Ok(Self {
            connection: connection.clone(),
            api,
            compute_statement,
            thread: thread::current().id(),
            _reservation: reservation,
        })
    }

    /// Uses state APIs only on the original callback thread. A changed/unknown
    /// transaction or statement state fails closed, including attached schemas.
    pub(crate) fn recheck(&self) -> Result<()> {
        if thread::current().id() != self.thread {
            return Err(misuse(
                "DiskANN build guard used on a different callback thread",
            ));
        }
        let statement = transaction_free_statement(&self.connection, self.api)?;
        if statement != self.compute_statement {
            return Err(misuse("DiskANN compute statement changed during build"));
        }
        Ok(())
    }

    /// Materializes immutable rows, ALWAYS finishes all tracked readers, then
    /// proves NONE again before CPU. The higher-ranked borrow and Send + 'static
    /// result prevent a SQLite statement/value/byte view escaping in safe code.
    pub(crate) fn materialize<T: Send + 'static>(
        &self,
        read: impl FnOnce(&mut SnapshotReaders<'_, 'db>) -> Result<T>,
    ) -> Result<T> {
        self.recheck()?;
        let mut readers = SnapshotReaders {
            connection: &self.connection,
            statements: Vec::new(),
        };
        let result = read(&mut readers);
        let cleanup = readers.finish();
        // Capture primary errors before cleanup. A cleanup error takes precedence
        // over success, but must NEVER replace the original SQL/read/validation error.
        let owned = original_error_first(result, cleanup)?;
        self.recheck()?;
        Ok(owned)
    }

    /// Runs one owned native worker, outside any ambient host Tokio runtime.
    /// The callback retains ALL SQLite leases and joins the worker before return.
    /// Every polling interval verifies NONE using only callback-thread state APIs;
    /// there is no SQL preparation/step or error-report DML in this compute phase.
    /// A violation is latched, but tasks are still joined: this is NOT a promise
    /// to preempt arbitrary native CPU or infallible allocation/panic-abort.
    pub(crate) fn run_owned<T, U, F>(&self, input: T, work: F) -> Result<U>
    where
        T: Send + 'static,
        U: Send + 'static,
        F: FnOnce(T) -> Result<U> + Send + 'static,
    {
        self.recheck()?;
        let (finished, completion) = mpsc::channel();
        let worker = thread::Builder::new()
            .name("vectorlite-diskann-build".to_owned())
            .stack_size(OWNED_WORKER_STACK_BYTES)
            .spawn(move || {
                let result = work(input);
                let _ = finished.send(());
                result
            })
            .map_err(|error| {
                IndexError::new(format!("cannot start DiskANN build worker: {error}"))
            })?;
        let mut violation = None;
        loop {
            if let Err(error) = self.recheck() {
                violation.get_or_insert(error);
            }
            match completion.recv_timeout(Duration::from_millis(25)) {
                Ok(()) | Err(RecvTimeoutError::Disconnected) => break,
                Err(RecvTimeoutError::Timeout) => {}
            }
        }
        let result = worker
            .join()
            .unwrap_or_else(|_| Err(IndexError::new("DiskANN build worker panicked")));
        let final_state = self.recheck();
        original_error_first(result, violation.map_or(final_state, Err))
    }
}

fn transaction_free_statement(
    connection: &Connection<'_>,
    api: BuildStateApi,
) -> Result<NonNull<ffi::sqlite3_stmt>> {
    let db = connection.raw_handle();
    // SAFETY: Connection's borrow contract confines this live connection to its
    // serialized callback. Both API entries were version/presence validated.
    let autocommit = unsafe { api.get_autocommit(db) };
    if autocommit == 0 {
        return Err(misuse(
            "DiskANN build requires autocommit; finish BEGIN/SAVEPOINT first",
        ));
    }
    // SAFETY: the live callback connection uses NULL to check ALL schemas.
    let transaction = unsafe { api.txn_state(db) };
    if transaction != ffi::SQLITE_TXN_NONE as c_int {
        return Err(misuse(
            "DiskANN build requires transaction state NONE in every schema",
        ));
    }
    let mut previous = std::ptr::null_mut();
    let mut compute_statement = None;
    loop {
        // SAFETY: enumeration is synchronous on the callback thread; no statement
        // is finalized/reset while enumerating. previous is NULL or the prior result.
        let statement = unsafe { api.next_stmt(db, previous) };
        let Some(statement) = NonNull::new(statement) else {
            break;
        };
        // SAFETY: this statement is a live result of the connection's enumeration.
        match unsafe { api.stmt_busy(statement.as_ptr()) } {
            0 => {}
            1 if compute_statement.is_none() => compute_statement = Some(statement),
            1 => {
                return Err(misuse(
                    "DiskANN build rejects another busy statement or open cursor",
                ))
            }
            _ => return Err(misuse("SQLite returned unknown statement busy state")),
        }
        previous = statement.as_ptr();
    }
    // sqlite3_context has no public current-statement accessor. Inside a scalar
    // callback the compute SELECT MUST be busy, so exactly one busy statement
    // identifies it. Zero is not proof, and autocommit never exempts other cursors.
    compute_statement.ok_or_else(|| misuse("DiskANN build requires one active compute SELECT"))
}

/// Owns every snapshot statement until explicit finish, including on read errors.
/// Input parsing/admission belongs to the caller; this type adds no schema policy.
pub(crate) struct SnapshotReaders<'guard, 'db> {
    connection: &'guard Connection<'db>,
    statements: Vec<Statement<'db>>,
}

impl<'db> SnapshotReaders<'_, 'db> {
    pub(crate) fn prepare(&mut self, sql: &str) -> Result<&mut Statement<'db>> {
        self.statements.try_reserve(1).map_err(|_| {
            IndexError::with_code(ffi::SQLITE_NOMEM as c_int, "cannot track snapshot reader")
        })?;
        self.statements.push(self.connection.prepare(sql)?);
        Ok(self
            .statements
            .last_mut()
            .expect("just inserted snapshot reader"))
    }

    fn finish(mut self) -> Result<()> {
        let mut first_error = None;
        // Finish every reader even when an earlier finalization fails. Drop must
        // not be the success path, because it intentionally discards SQLite errors.
        for statement in self.statements.drain(..) {
            if let Err(error) = statement.finish() {
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}

fn original_error_first<T>(result: Result<T>, cleanup: Result<()>) -> Result<T> {
    match result {
        Err(error) => Err(error),
        Ok(value) => cleanup.map(|()| value),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    struct State {
        autocommit: c_int,
        transaction: c_int,
        busy: [c_int; 3],
    }
    const CLEAR: State = State {
        autocommit: 1,
        transaction: 0,
        busy: [1, 0, 0],
    };
    thread_local! { static STATE: std::cell::Cell<State> = const { std::cell::Cell::new(CLEAR) }; }

    unsafe extern "C" fn version() -> c_int {
        3_038_000
    }
    unsafe extern "C" fn autocommit(_: *mut ffi::sqlite3) -> c_int {
        STATE.with(|state| state.get().autocommit)
    }
    unsafe extern "C" fn transaction(
        _: *mut ffi::sqlite3,
        schema: *const std::os::raw::c_char,
    ) -> c_int {
        assert!(schema.is_null(), "must inspect all schemas, not just main");
        STATE.with(|state| state.get().transaction)
    }
    unsafe extern "C" fn next(
        _: *mut ffi::sqlite3,
        previous: *mut ffi::sqlite3_stmt,
    ) -> *mut ffi::sqlite3_stmt {
        match previous as usize {
            0 => std::ptr::dangling_mut::<ffi::sqlite3_stmt>(),
            1 => 2usize as *mut ffi::sqlite3_stmt,
            2 => 3usize as *mut ffi::sqlite3_stmt,
            _ => std::ptr::null_mut(),
        }
    }
    unsafe extern "C" fn busy(statement: *mut ffi::sqlite3_stmt) -> c_int {
        STATE.with(|state| state.get().busy[statement as usize - 1])
    }
    fn api() -> BuildStateApi {
        // SAFETY: the generated table contains only nullable function pointers.
        let mut table: ffi::sqlite3_api_routines = unsafe { std::mem::zeroed() };
        table.libversion_number = Some(version);
        table.get_autocommit = Some(autocommit);
        table.txn_state = Some(transaction);
        table.next_stmt = Some(next);
        table.stmt_busy = Some(busy);
        // SAFETY: the fixture table is live through validation and every returned
        // function pointer is a correctly typed static mock, never dereferencing db.
        unsafe { BuildStateApi::from_table(&table).unwrap() }
    }
    fn connection() -> Connection<'static> {
        // SAFETY: mock state functions only compare this opaque non-null identity;
        // no mock dereferences a SQLite connection or accesses a native runtime.
        unsafe { Connection::borrow(NonNull::<ffi::sqlite3>::dangling().as_ptr()).unwrap() }
    }

    #[test]
    fn explicit_transaction_without_pager_state_rejects() {
        STATE.with(|state| {
            state.set(State {
                autocommit: 0,
                ..CLEAR
            })
        });
        let error = BuildGuard::enter_with_api(&connection(), api())
            .err()
            .unwrap();
        assert!(error.message.contains("autocommit"));
        STATE.with(|state| state.set(CLEAR));
        assert!(
            BuildGuard::enter_with_api(&connection(), api()).is_ok(),
            "failed admission releases reservation"
        );
    }

    #[test]
    fn non_none_and_unknown_transaction_states_reject() {
        for transaction in [1, 2, -1, 3, c_int::MAX] {
            STATE.with(|state| {
                state.set(State {
                    transaction,
                    ..CLEAR
                })
            });
            let error = BuildGuard::enter_with_api(&connection(), api())
                .err()
                .unwrap();
            assert!(error.message.contains("state NONE"));
        }
    }

    #[test]
    fn current_select_count_does_not_hide_busy_cursors() {
        for busy in [[1, 1, 0], [1, 0, 1], [0, 1, 1]] {
            STATE.with(|state| state.set(State { busy, ..CLEAR }));
            let error = BuildGuard::enter_with_api(&connection(), api())
                .err()
                .unwrap();
            assert!(error.message.contains("another busy"));
        }
        for busy in [[0, 0, 0], [1, -1, 0], [1, 2, 0]] {
            STATE.with(|state| state.set(State { busy, ..CLEAR }));
            assert!(BuildGuard::enter_with_api(&connection(), api()).is_err());
        }
        STATE.with(|state| state.set(CLEAR));
        assert!(BuildGuard::enter_with_api(&connection(), api()).is_ok());
    }

    #[test]
    fn reentry_and_statement_change_fail_closed() {
        STATE.with(|state| state.set(CLEAR));
        let connection = connection();
        let guard = BuildGuard::enter_with_api(&connection, api()).unwrap();
        let nested = BuildGuard::enter_with_api(&connection, api())
            .err()
            .unwrap();
        assert!(nested.message.contains("recursive"));
        STATE.with(|state| {
            state.set(State {
                busy: [0, 1, 0],
                ..CLEAR
            })
        });
        assert!(guard
            .recheck()
            .unwrap_err()
            .message
            .contains("statement changed"));
        STATE.with(|state| state.set(CLEAR));
        drop(guard);
        assert!(BuildGuard::enter_with_api(&connection, api()).is_ok());
    }

    #[test]
    fn post_read_state_violation_rejects_before_worker() {
        STATE.with(|state| state.set(CLEAR));
        let guard = BuildGuard::enter_with_api(&connection(), api()).unwrap();
        let result = guard.materialize(|_| {
            STATE.with(|state| {
                state.set(State {
                    transaction: 1,
                    ..CLEAR
                })
            });
            Ok(vec![1u8])
        });
        assert!(result.unwrap_err().message.contains("state NONE"));
    }

    #[test]
    fn cleanup_errors_block_success_but_preserve_primary_error() {
        let primary = IndexError::with_code(ffi::SQLITE_CONSTRAINT as c_int, "original read error");
        let cleanup = IndexError::with_code(ffi::SQLITE_IOERR_CLOSE as c_int, "finalize error");
        assert_eq!(
            original_error_first::<()>(Err(primary.clone()), Err(cleanup.clone())),
            Err(primary)
        );
        assert_eq!(
            original_error_first(Ok(vec![1u8]), Err(cleanup.clone())),
            Err(cleanup)
        );
    }

    #[test]
    fn owned_worker_is_joined_on_a_different_thread() {
        STATE.with(|state| state.set(CLEAR));
        let guard = BuildGuard::enter_with_api(&connection(), api()).unwrap();
        let callback_thread = thread::current().id();
        let output = guard
            .run_owned(vec![1u8, 2, 3], move |input| {
                assert_ne!(thread::current().id(), callback_thread);
                Ok(input.iter().map(|value| usize::from(*value)).sum::<usize>())
            })
            .unwrap();
        assert_eq!(output, 6);
        assert!(guard.recheck().is_ok());
    }
}
