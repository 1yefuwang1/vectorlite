//! Private connection-local invocation inside a SQLite journaled statement.
#![deny(unsafe_op_in_unsafe_fn)]

use std::cell::Cell;
use std::marker::PhantomData;
use std::os::raw::{c_char, c_int, c_void};
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::{self, ThreadId};

use crate::ffi;
use crate::index_error::IndexError;

type Result<T> = std::result::Result<T, IndexError>;

pub(crate) const FUNCTION_NAME: &str = "vectorlite_atomic_write";
pub(crate) const POINTER_TAG: &[u8] = b"vectorlite_atomic_action\0";

/// Borrowed SQLite binding. It must not escape the execute closure or remain
/// bound after its statement is finalized. The nonce prevents address-reuse ABA.
pub(crate) struct Invocation {
    pub(crate) pointer: *mut c_void,
    pub(crate) token: i64,
    _local: PhantomData<Rc<()>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Violation {
    Pointer,
    Token,
    Connection,
    Thread,
    Busy,
    Duplicate,
    Closed,
    Arguments,
}
impl Violation {
    fn error(self) -> IndexError {
        let message = match self {
            Self::Pointer => "foreign atomic action pointer",
            Self::Token => "stale or invalid atomic action nonce",
            Self::Connection => "atomic action used by a different SQLite connection",
            Self::Thread => "atomic action used on the wrong callback thread",
            Self::Busy => "recursive atomic action invocation",
            Self::Duplicate => "atomic action invoked more than once",
            Self::Closed => "atomic action scope is closing",
            Self::Arguments => "atomic action requires its private pointer and integer nonce",
        };
        IndexError::with_code(ffi::SQLITE_MISUSE as c_int, message)
    }
}

type Dispatch = unsafe fn(NonNull<()>) -> Result<()>;
#[derive(Clone, Copy)]
struct Frame {
    pointer: NonNull<()>,
    token: i64,
    db: *mut ffi::sqlite3,
    thread: ThreadId,
    dispatch: Dispatch,
    called: bool,
    busy: bool,
    closed: bool,
    violation: Option<Violation>,
}
thread_local! { static ACTIVE: Cell<Option<Frame>> = const { Cell::new(None) }; }
static NEXT_TOKEN: AtomicU64 = AtomicU64::new(1);

struct Slot<F, T> {
    action: Option<F>,
    result: Option<Result<T>>,
}

fn normalize_error(error: IndexError) -> IndexError {
    let primary = error.code & 0xff;
    if error.code < 0
        || primary == ffi::SQLITE_OK as c_int
        || primary == ffi::SQLITE_ROW as c_int
        || primary == ffi::SQLITE_DONE as c_int
    {
        IndexError::with_code(ffi::SQLITE_ERROR as c_int, error.message)
    } else {
        error
    }
}

fn no_scope() -> IndexError {
    IndexError::with_code(ffi::SQLITE_MISUSE as c_int, "no active atomic SQL action")
}
fn latch_violation(violation: Violation) -> IndexError {
    ACTIVE.with(|active| {
        if let Some(mut frame) = active.get() {
            let first = *frame.violation.get_or_insert(violation);
            active.set(Some(frame));
            first.error()
        } else {
            violation.error()
        }
    })
}
thread_local! { static ENTERED: Cell<bool> = const { Cell::new(false) }; }
struct EntryLease(PhantomData<Rc<()>>);
impl EntryLease {
    fn enter() -> Result<Self> {
        ENTERED.with(|entered| {
            if entered.replace(true) {
                return Err(latch_violation(Violation::Busy));
            }
            Ok(Self(PhantomData))
        })
    }
}
impl Drop for EntryLease {
    fn drop(&mut self) {
        ENTERED.with(|entered| entered.set(false));
    }
}

struct Scope<F, T> {
    // Own the Box allocation through its raw pointer while callbacks are live.
    // Moving/reborrowing a Box would otherwise retag the published alias.
    slot: NonNull<Slot<F, T>>,
    token: i64,
    _owned: PhantomData<Box<Slot<F, T>>>,
    _local: PhantomData<Rc<()>>,
}
impl<F, T> Scope<F, T>
where
    F: FnOnce() -> Result<T>,
{
    fn new(db: *mut ffi::sqlite3, action: F) -> Result<Self> {
        if ACTIVE.with(|active| active.get().is_some()) {
            return Err(latch_violation(Violation::Busy));
        }
        let nonce = NEXT_TOKEN
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                (n <= i64::MAX as u64).then(|| n + 1)
            })
            .map_err(|_| IndexError::new("atomic action nonce space exhausted"))?;
        let token = nonce as i64;
        let allocation = Box::new(Slot {
            action: Some(action),
            result: None,
        });
        // SAFETY: Box::into_raw returns a non-null aligned allocation. Scope
        // exclusively owns it until Drop reconstructs exactly one Box.
        let slot = unsafe { NonNull::new_unchecked(Box::into_raw(allocation)) };
        let pointer = slot.cast();
        ACTIVE.with(|active| {
            active.set(Some(Frame {
                pointer,
                token,
                db,
                thread: thread::current().id(),
                dispatch: dispatch_slot::<F, T>,
                called: false,
                busy: false,
                closed: false,
                violation: None,
            }))
        });
        Ok(Self {
            slot,
            token,
            _owned: PhantomData,
            _local: PhantomData,
        })
    }
    fn invocation(&self) -> Invocation {
        Invocation {
            pointer: self.slot.as_ptr().cast(),
            token: self.token,
            _local: PhantomData,
        }
    }
    fn finish(&mut self, sql_result: Result<()>) -> Result<T> {
        close_frame(self.token);
        let frame = ACTIVE.with(|active| active.get().filter(|frame| frame.token == self.token));
        // SAFETY: execute finalized its statement, the frame is now closed,
        // and Scope still exclusively owns this live typed allocation.
        let stored = unsafe { self.slot.as_mut() }.result.take();
        if let Err(sql_error) = sql_result {
            let sql_error = normalize_error(sql_error);
            // SQLite may report generic SQLITE_ERROR after a scalar error. Keep
            // the original code in that case, or when the exact codes match.
            // An unrelated execution/finalize error (especially I/O) wins.
            let action_error = stored.as_ref().and_then(|result| result.as_ref().err());
            return Err(match action_error {
                Some(error)
                    if sql_error.code == error.code
                        || sql_error.code == ffi::SQLITE_ERROR as c_int =>
                {
                    error.clone()
                }
                _ => sql_error,
            });
        }
        let frame = frame.ok_or_else(no_scope)?;
        if let Some(violation) = frame.violation {
            return Err(violation.error());
        }
        if !frame.called {
            return Err(IndexError::with_code(
                ffi::SQLITE_CORRUPT as c_int,
                "journaled statement did not invoke its atomic action",
            ));
        }
        stored.ok_or_else(|| IndexError::new("atomic action produced no result"))?
    }
}
fn close_frame(token: i64) {
    ACTIVE.with(|active| {
        if let Some(mut frame) = active.get().filter(|frame| frame.token == token) {
            frame.closed = true;
            frame.busy = true;
            active.set(Some(frame));
        }
    });
}
fn clear_frame(token: i64) {
    ACTIVE.with(|active| {
        if active.get().is_some_and(|frame| frame.token == token) {
            active.set(None);
        }
    });
}
struct ClearLease(i64);
impl Drop for ClearLease {
    fn drop(&mut self) {
        clear_frame(self.0);
    }
}
impl<F, T> Drop for Scope<F, T> {
    fn drop(&mut self) {
        close_frame(self.token);
        // Also clear on an unexpected destructor unwind. The closed frame can
        // never dereference its pointer during capture/result destruction.
        let _clear = ClearLease(self.token);
        // SAFETY: Scope exclusively owns this allocation from Box::into_raw;
        // the closed frame prevents any further callback references. This is
        // the one and only reconstruction/deallocation of the original Box.
        let mut slot = unsafe { Box::from_raw(self.slot.as_ptr()) };
        drop(slot.action.take());
        drop(slot.result.take());
        clear_frame(self.token);
        // The allocation is freed after captures/results and pointer removal.
        drop(slot);
    }
}

/// Runs synchronous connection-local work inside the caller's proven statement
/// journal. verify_guard must validate the canonical ordinary guard table and
/// full-table multi-row statement shape before execute can publish writes.
/// execute must bind BOTH Invocation fields and finalize its statement before
/// returning, even on failure. This adapter never starts a SQL transaction.
pub(crate) fn with_action<T, F, V, E>(
    db: *mut ffi::sqlite3,
    action: F,
    verify_guard: V,
    execute: E,
) -> Result<T>
where
    F: FnOnce() -> Result<T>,
    V: FnOnce() -> Result<()>,
    E: FnOnce(Invocation) -> Result<()>,
{
    if db.is_null() {
        return Err(IndexError::with_code(
            ffi::SQLITE_MISUSE as c_int,
            "atomic action connection is null",
        ));
    }
    let _entry = EntryLease::enter()?;
    verify_guard()?;
    let mut scope = Scope::new(db, action)?;
    let execution = execute(scope.invocation());
    let result = scope.finish(execution);
    drop(scope);
    result
}
unsafe fn dispatch_slot<F, T>(pointer: NonNull<()>) -> Result<()>
where
    F: FnOnce() -> Result<T>,
{
    // SAFETY: only invoke_bound calls this after checking the exact live Box
    // pointer, nonce, connection, owner thread, and unique nonrecursive call.
    // The monomorphized function was installed with this same Slot<F,T>.
    let slot = unsafe { &mut *pointer.cast::<Slot<F, T>>().as_ptr() };
    let action = slot
        .action
        .take()
        .ok_or_else(|| Violation::Duplicate.error())?;
    let result = action().map_err(normalize_error);
    let report = result.as_ref().map(|_| ()).map_err(Clone::clone);
    // Keep the original typed result/error before SQLite can format or narrow it.
    slot.result = Some(result);
    report
}

fn invoke_bound(db: *mut ffi::sqlite3, pointer: *mut c_void, token: i64) -> Result<()> {
    let frame = ACTIVE.with(|active| active.get()).ok_or_else(no_scope)?;
    if let Some(violation) = frame.violation {
        return Err(violation.error());
    }
    let violation = if frame.thread != thread::current().id() {
        Some(Violation::Thread)
    } else if frame.db != db {
        Some(Violation::Connection)
    } else if frame.pointer.as_ptr().cast::<c_void>() != pointer {
        Some(Violation::Pointer)
    } else if token <= 0 || frame.token != token {
        Some(Violation::Token)
    } else if frame.closed {
        Some(Violation::Closed)
    } else if frame.busy {
        Some(Violation::Busy)
    } else if frame.called {
        Some(Violation::Duplicate)
    } else {
        None
    };
    if let Some(violation) = violation {
        return Err(latch_violation(violation));
    }
    let mut running = frame;
    running.called = true;
    running.busy = true;
    ACTIVE.with(|active| active.set(Some(running)));
    // SAFETY: all identity/lifetime/thread/one-call checks completed above,
    // before any Slot reference was constructed. Busy prevents recursive aliasing
    // while the action itself makes nested host SQLite calls.
    let result = unsafe { (frame.dispatch)(frame.pointer) };
    ACTIVE.with(|active| {
        if let Some(mut current) = active.get().filter(|current| current.token == frame.token) {
            current.busy = false;
            active.set(Some(current));
        }
    });
    if let Some(violation) = ACTIVE.with(|active| active.get().and_then(|frame| frame.violation)) {
        return Err(violation.error());
    }
    result
}
/// Private DIRECTONLY scalar registered with exactly two arguments.
///
/// # Safety
/// SQLite supplies a live result context and two protected values whenever the
/// argument count is valid. The extension API is initialized for that host.
pub(crate) unsafe extern "C" fn invoke(
    context: *mut ffi::sqlite3_context,
    argc: c_int,
    argv: *mut *mut ffi::sqlite3_value,
) {
    if context.is_null() {
        let _ = latch_violation(Violation::Arguments);
        return;
    }
    let result = if argc != 2 || argv.is_null() {
        Err(latch_violation(Violation::Arguments))
    } else {
        // SAFETY: the checked callback count gives two readable pointer entries.
        // Copy them before invoking work, so no argv borrow spans nested SQLite.
        let (first, second) = unsafe { (argv.read(), argv.add(1).read()) };
        if first.is_null() || second.is_null() {
            Err(latch_violation(Violation::Arguments))
        } else {
            // SAFETY: protected values/context are live for this callback. Only
            // SQLite-owned metadata is inspected; no supplied pointer is dereferenced.
            let kind = unsafe { ffi::value_type(second) };
            if kind != ffi::SQLITE_INTEGER as c_int {
                Err(latch_violation(Violation::Arguments))
            } else {
                // SAFETY: the private tag is static NUL-terminated; both
                // protected values and context are live for this callback.
                let (pointer, token, db) = unsafe {
                    (
                        ffi::value_pointer(first, POINTER_TAG.as_ptr().cast::<c_char>()),
                        ffi::value_int64(second),
                        ffi::context_db_handle(context),
                    )
                };
                invoke_bound(db, pointer, token)
            }
        }
    };
    match result {
        Ok(()) => {
            // SAFETY: SQLite's result context remains live until callback return.
            unsafe { ffi::result_double(context, 0.0) };
        }
        Err(error) => {
            // Both message and exact code are set DURING the journaled SQL
            // statement, before SQLite can release its statement savepoint.
            // SAFETY: the live context copies the message; setting its code
            // afterwards preserves extended constraints/I/O codes.
            unsafe {
                ffi::result_error(context, &error.message);
                ffi::result_error_code(context, error.code);
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    fn db() -> *mut ffi::sqlite3 {
        NonNull::<ffi::sqlite3>::dangling().as_ptr()
    }
    fn clear() {
        assert!(ACTIVE.with(|active| active.get().is_none()));
        assert!(!ENTERED.with(Cell::get));
    }

    #[test]
    fn typed_borrowed_non_send_action_executes_exactly_once() {
        let calls = Rc::new(Cell::new(0));
        let captured = calls.clone();
        let mut value = 40;
        let result = with_action(
            db(),
            || {
                captured.set(captured.get() + 1);
                value += 2;
                Ok(value)
            },
            || Ok(()),
            |invocation| invoke_bound(db(), invocation.pointer, invocation.token),
        )
        .unwrap();
        assert_eq!(result, 42);
        assert_eq!(value, 42);
        assert_eq!(calls.get(), 1);
        clear();
    }

    #[test]
    fn missing_or_invalid_guard_prevents_all_action_access() {
        let calls = Cell::new(0);
        let error = with_action(
            db(),
            || {
                calls.set(1);
                Ok(())
            },
            || {
                Err(IndexError::with_code(
                    ffi::SQLITE_CORRUPT as c_int,
                    "missing guard row/shape",
                ))
            },
            |_| panic!("executor must not run before guard verification succeeds"),
        )
        .unwrap_err();
        assert!(error.message.contains("guard"));
        assert_eq!(calls.get(), 0);
        clear();
    }

    #[test]
    fn foreign_pointer_nonce_and_connection_fail_before_dereference() {
        for wrong in [Violation::Pointer, Violation::Token, Violation::Connection] {
            let calls = Cell::new(0);
            let error = with_action(
                db(),
                || {
                    calls.set(1);
                    Ok(())
                },
                || Ok(()),
                |invocation| {
                    let pointer = if wrong == Violation::Pointer {
                        NonNull::<u8>::dangling().as_ptr().cast()
                    } else {
                        invocation.pointer
                    };
                    let nonce = if wrong == Violation::Token {
                        invocation.token + 1
                    } else {
                        invocation.token
                    };
                    let connection = if wrong == Violation::Connection {
                        std::ptr::without_provenance_mut(2)
                    } else {
                        db()
                    };
                    invoke_bound(connection, pointer, nonce)
                },
            )
            .unwrap_err();
            assert_eq!(error.code, ffi::SQLITE_MISUSE as c_int);
            assert_eq!(calls.get(), 0);
            clear();
        }
    }

    #[test]
    fn stale_scope_and_aba_nonce_cannot_reach_a_new_action() {
        let mut old_nonce = 0;
        let mut old_pointer = std::ptr::null_mut();
        with_action(
            db(),
            || Ok(()),
            || Ok(()),
            |invocation| {
                old_nonce = invocation.token;
                old_pointer = invocation.pointer;
                invoke_bound(db(), invocation.pointer, invocation.token)
            },
        )
        .unwrap();
        assert!(invoke_bound(db(), old_pointer, old_nonce).is_err());
        let called = Cell::new(false);
        with_action(
            db(),
            || {
                called.set(true);
                Ok(())
            },
            || Ok(()),
            |invocation| {
                // Simulate address reuse explicitly: current real pointer + old token.
                invoke_bound(db(), invocation.pointer, old_nonce)
            },
        )
        .unwrap_err();
        assert!(!called.get());
        clear();
    }

    #[test]
    fn duplicate_and_recursive_calls_are_sticky_even_if_executor_ignores_them() {
        let calls = Cell::new(0);
        with_action(
            db(),
            || {
                calls.set(calls.get() + 1);
                Ok(())
            },
            || Ok(()),
            |invocation| {
                invoke_bound(db(), invocation.pointer, invocation.token)?;
                assert!(invoke_bound(db(), invocation.pointer, invocation.token).is_err());
                Ok(())
            },
        )
        .unwrap_err();
        assert_eq!(calls.get(), 1);
        clear();
        let invocation_copy = Cell::new((std::ptr::null_mut(), 0));
        with_action(
            db(),
            || {
                let (pointer, token) = invocation_copy.get();
                assert!(invoke_bound(db(), pointer, token).is_err());
                Ok(())
            },
            || Ok(()),
            |invocation| {
                invocation_copy.set((invocation.pointer, invocation.token));
                invoke_bound(db(), invocation.pointer, invocation.token)
            },
        )
        .unwrap_err();
        clear();
    }

    #[test]
    fn nested_scopes_including_guard_verification_do_not_run_nested_actions() {
        let nested = Cell::new(false);
        with_action(
            db(),
            || Ok(()),
            || {
                assert!(with_action(
                    db(),
                    || {
                        nested.set(true);
                        Ok(())
                    },
                    || Ok(()),
                    |_| Ok(())
                )
                .is_err());
                Ok(())
            },
            |invocation| invoke_bound(db(), invocation.pointer, invocation.token),
        )
        .unwrap();
        assert!(!nested.get());
        clear();
        with_action(
            db(),
            || {
                assert!(with_action(
                    db(),
                    || {
                        nested.set(true);
                        Ok(())
                    },
                    || Ok(()),
                    |_| Ok(())
                )
                .is_err());
                Ok(())
            },
            || Ok(()),
            |invocation| invoke_bound(db(), invocation.pointer, invocation.token),
        )
        .unwrap_err();
        assert!(!nested.get());
        clear();
    }

    #[test]
    fn wrong_thread_has_no_permission_to_access_the_callback_slot() {
        let calls = Cell::new(0);
        with_action(
            db(),
            || {
                calls.set(1);
                Ok(())
            },
            || Ok(()),
            |invocation| {
                // Only an opaque address/nonce is transported, not a Send SQL owner.
                let address = invocation.pointer as usize;
                let nonce = invocation.token;
                std::thread::spawn(move || {
                    invoke_bound(db(), std::ptr::without_provenance_mut(address), nonce)
                })
                .join()
                .unwrap()
            },
        )
        .unwrap_err();
        assert_eq!(calls.get(), 0);
        clear();
    }

    #[test]
    fn original_action_error_survives_sqlite_generic_error_but_unrelated_io_wins() {
        let error = with_action::<(), _, _, _>(
            db(),
            || Err(IndexError::with_code(1811, "late node constraint")),
            || Ok(()),
            |invocation| {
                let callback =
                    invoke_bound(db(), invocation.pointer, invocation.token).unwrap_err();
                assert_eq!(callback.code, 1811);
                Err(IndexError::with_code(
                    ffi::SQLITE_ERROR as c_int,
                    "SQLite scalar failed",
                ))
            },
        )
        .unwrap_err();
        assert_eq!(error.code, 1811);
        assert_eq!(error.message, "late node constraint");
        clear();
        let error = with_action::<(), _, _, _>(
            db(),
            || Err(IndexError::with_code(1811, "constraint")),
            || Ok(()),
            |invocation| {
                invoke_bound(db(), invocation.pointer, invocation.token).unwrap_err();
                Err(IndexError::with_code(778, "finalize I/O failure"))
            },
        )
        .unwrap_err();
        assert_eq!(error.code, 778);
        clear();
    }

    #[test]
    fn action_success_is_never_accepted_after_failed_sql_or_finalize() {
        let error = with_action(
            db(),
            || Ok(42),
            || Ok(()),
            |invocation| {
                invoke_bound(db(), invocation.pointer, invocation.token)?;
                Err(IndexError::with_code(778, "journaled statement failed"))
            },
        )
        .unwrap_err();
        assert_eq!(error.code, 778);
        clear();
        let error = with_action(db(), || Ok(42), || Ok(()), |_| Ok(())).unwrap_err();
        assert!(error.message.contains("did not invoke"));
        clear();
    }

    struct DropProbe(Rc<Cell<bool>>);
    impl Drop for DropProbe {
        fn drop(&mut self) {
            let frame = ACTIVE
                .with(Cell::get)
                .expect("capture/result drops before TLS pointer lease clears");
            assert!(frame.closed && frame.busy);
            self.0.set(true);
        }
    }
    #[test]
    fn uncalled_capture_and_unsuccessful_result_drop_before_tls_removal() {
        let dropped = Rc::new(Cell::new(false));
        let probe = DropProbe(dropped.clone());
        with_action(
            db(),
            move || {
                let _hold = &probe;
                Ok(42)
            },
            || Ok(()),
            |_| Ok(()),
        )
        .unwrap_err();
        assert!(dropped.get());
        clear();
        let dropped = Rc::new(Cell::new(false));
        let probe = DropProbe(dropped.clone());
        let result = with_action(
            db(),
            move || Ok(probe),
            || Ok(()),
            |invocation| {
                invoke_bound(db(), invocation.pointer, invocation.token)?;
                Err(IndexError::with_code(778, "execution failed"))
            },
        );
        assert!(result.is_err());
        assert!(dropped.get());
        clear();
    }

    #[test]
    fn invalid_error_status_cannot_signal_sql_success() {
        for code in [0, ffi::SQLITE_ROW as c_int, ffi::SQLITE_DONE as c_int, -1] {
            let error = with_action::<(), _, _, _>(
                db(),
                || Err(IndexError::with_code(code, "invalid error status")),
                || Ok(()),
                |invocation| invoke_bound(db(), invocation.pointer, invocation.token),
            )
            .unwrap_err();
            assert_eq!(error.code, ffi::SQLITE_ERROR as c_int);
            clear();
        }
    }

    #[test]
    fn executor_failure_before_callback_drops_pending_action_and_clears_scope() {
        let dropped = Rc::new(Cell::new(false));
        let probe = DropProbe(dropped.clone());
        let error = with_action(
            db(),
            move || {
                let _hold = &probe;
                Ok(())
            },
            || Ok(()),
            |_| {
                Err(IndexError::with_code(
                    ffi::SQLITE_BUSY as c_int,
                    "cannot prepare carrier",
                ))
            },
        )
        .unwrap_err();
        assert_eq!(error.code, ffi::SQLITE_BUSY as c_int);
        assert!(dropped.get());
        clear();
    }

    #[test]
    fn destructor_unwind_clears_closed_pointer_lease() {
        struct PanickingCapture;
        impl Drop for PanickingCapture {
            fn drop(&mut self) {
                let frame = ACTIVE
                    .with(Cell::get)
                    .expect("cleanup pointer lease remains live");
                assert!(frame.closed && frame.busy);
                panic!("capture destructor panic");
            }
        }
        let result = std::panic::catch_unwind(|| {
            let capture = PanickingCapture;
            with_action(
                db(),
                move || {
                    let _hold = &capture;
                    Ok(())
                },
                || Ok(()),
                |_| Ok(()),
            )
        });
        assert!(result.is_err());
        clear();
        with_action(
            db(),
            || Ok(()),
            || Ok(()),
            |invocation| invoke_bound(db(), invocation.pointer, invocation.token),
        )
        .unwrap();
    }

    #[test]
    fn unwind_cleanup_closes_scope_without_leaving_an_authorized_pointer() {
        let result = std::panic::catch_unwind(|| {
            with_action(
                db(),
                || Ok(()),
                || Ok(()),
                |_| -> Result<()> {
                    panic!("executor panic");
                },
            )
        });
        assert!(result.is_err());
        clear();
    }
}
