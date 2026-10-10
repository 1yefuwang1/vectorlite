//! Non-owning SQLite connection and owned statements for shadow-table storage.
//!
//! Every call uses the host's extension API. No connection is opened or closed
//! here, and no handle is Send/Sync. An anchor statement may remain at Row to
//! keep the host connection's read transaction alive for a virtual-table cursor.

use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::os::raw::c_int;
use std::ptr::NonNull;
use std::rc::Rc;

use crate::ffi;
use crate::index_error::IndexError;

type Result<T> = std::result::Result<T, IndexError>;

#[derive(Clone)]
pub(crate) struct Connection<'db> {
    db: NonNull<ffi::sqlite3>,
    _lifetime: PhantomData<&'db ffi::sqlite3>,
    _connection_local: PhantomData<Rc<()>>,
}

impl<'db> Connection<'db> {
    /// Borrows, but never assumes ownership of, SQLite's host connection.
    ///
    /// # Safety
    /// The extension API must be initialized. `db` must be a live host connection
    /// that outlives `'db` and every derived statement. All operations must stay
    /// within SQLite's serialized callbacks for this connection. The caller must
    /// not close the connection while any borrowed handle or statement exists.
    pub(crate) unsafe fn borrow(db: *mut ffi::sqlite3) -> Result<Self> {
        let db = NonNull::new(db).ok_or_else(|| {
            IndexError::with_code(ffi::SQLITE_MISUSE as c_int, "SQLite connection is null")
        })?;
        Ok(Self {
            db,
            _lifetime: PhantomData,
            _connection_local: PhantomData,
        })
    }

    /// Prepares exactly one nonempty statement. Values must be bound separately.
    /// Only whitespace may follow the statement; SQL scripts are not accepted.
    pub(crate) fn prepare(&self, sql: &str) -> Result<Statement<'db>> {
        let length = checked_length(sql.len().checked_add(1))?;
        let sql = CString::new(sql).map_err(|_| {
            IndexError::with_code(ffi::SQLITE_MISUSE as c_int, "SQL contains a NUL byte")
        })?;
        let mut pointer = std::ptr::null_mut();
        let mut tail = std::ptr::null();
        // SAFETY: the borrowed host connection is live; the checked CString and
        // writable output slots remain valid throughout preparation.
        let status = unsafe {
            ffi::prepare_v2(
                self.db.as_ptr(),
                sql.as_ptr(),
                length,
                &mut pointer,
                &mut tail,
            )
        };
        if status != ffi::SQLITE_OK as c_int {
            let error = self.error(status, "prepare statement");
            if !pointer.is_null() {
                // SAFETY: a non-null prepare output is an owned statement. Its
                // error was copied above, before cleanup can change host errors.
                unsafe { ffi::finalize(pointer) };
            }
            return Err(error);
        }
        let pointer = NonNull::new(pointer).ok_or_else(|| {
            IndexError::with_code(ffi::SQLITE_MISUSE as c_int, "SQL contains no statement")
        })?;
        let statement = Statement {
            pointer: Some(pointer),
            connection: self.clone(),
            state: State::Ready,
        };
        let start = sql.as_ptr() as usize;
        let end = start + sql.as_bytes().len();
        let tail_address = tail as usize;
        if tail.is_null() || tail_address < start || tail_address > end {
            return Err(IndexError::with_code(
                ffi::SQLITE_MISUSE as c_int,
                "SQLite returned an invalid statement tail",
            ));
        }
        if !sql.as_bytes()[tail_address - start..]
            .iter()
            .all(u8::is_ascii_whitespace)
        {
            return Err(IndexError::with_code(
                ffi::SQLITE_MISUSE as c_int,
                "expected one SQL statement with no trailing SQL",
            ));
        }
        Ok(statement)
    }

    /// Runs one statement to completion; any result rows are discarded.
    pub(crate) fn execute(&self, sql: &str) -> Result<()> {
        let mut statement = self.prepare(sql)?;
        while statement.step()? == Step::Row {}
        statement.finish()
    }

    pub(crate) fn random_instance_id(&self) -> [u8; 16] {
        let mut identity = [0u8; 16];
        // SAFETY: the initialized host API accepts this writable 16-byte buffer
        // and fills it synchronously without retaining its address.
        unsafe { ffi::randomness(16, identity.as_mut_ptr().cast()) };
        identity
    }

    /// Exposes only an opaque identity for scoped callback matching. Accessing
    /// the host connection through this pointer still requires an unsafe API.
    pub(crate) fn raw_handle(&self) -> *mut ffi::sqlite3 {
        self.db.as_ptr()
    }

    pub(crate) fn changes(&self) -> c_int {
        // SAFETY: the constructor's contract keeps the connection live and
        // restricts calls to its serialized callback access.
        unsafe { ffi::changes(self.db.as_ptr()) }
    }

    fn error(&self, status: c_int, operation: &str) -> IndexError {
        // SAFETY: both accessors use the live borrowed connection. Copy the
        // connection-owned error string before any further SQLite operation.
        let (extended, detail) = unsafe {
            let extended = ffi::extended_errcode(self.db.as_ptr());
            let message = ffi::errmsg(self.db.as_ptr());
            let detail = if message.is_null() {
                String::new()
            } else {
                CStr::from_ptr(message).to_string_lossy().into_owned()
            };
            (extended, detail)
        };
        // Some routines return an error without updating the connection's error
        // slot. Never replace it with a stale, unrelated extended error code.
        let code = if status & !0xff != 0 {
            status
        } else if extended & 0xff == status & 0xff {
            extended
        } else {
            status
        };
        let message = if extended & 0xff == status & 0xff && !detail.is_empty() {
            format!("{operation}: {detail}")
        } else {
            format!("{operation}: SQLite error {code}")
        };
        IndexError::with_code(code, message)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Step {
    Row,
    Done,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Ready,
    Row,
    Done,
    Failed,
}

pub(crate) struct Statement<'db> {
    pointer: Option<NonNull<ffi::sqlite3_stmt>>,
    connection: Connection<'db>,
    state: State,
}

impl Statement<'_> {
    // The pointer is taken only by consuming finish(), so every public method
    // that still has access to self retains the live SQLite statement.
    fn raw(&self) -> *mut ffi::sqlite3_stmt {
        self.pointer.map_or(std::ptr::null_mut(), NonNull::as_ptr)
    }

    fn bind_index(&self, index: usize) -> Result<c_int> {
        if self.state != State::Ready {
            return Err(IndexError::with_code(
                ffi::SQLITE_MISUSE as c_int,
                "bind values before stepping the statement",
            ));
        }
        c_int::try_from(index)
            .ok()
            .filter(|&index| index > 0)
            .ok_or_else(|| {
                IndexError::with_code(ffi::SQLITE_RANGE as c_int, "binding index is out of range")
            })
    }

    fn check(&self, status: c_int, operation: &str) -> Result<()> {
        if status == ffi::SQLITE_OK as c_int {
            Ok(())
        } else {
            Err(self.connection.error(status, operation))
        }
    }

    pub(crate) fn bind_i64(&mut self, index: usize, value: i64) -> Result<()> {
        let index = self.bind_index(index)?;
        // SAFETY: the owned statement has not been stepped since preparation.
        // SQLite checks the positive parameter index and copies the value.
        let status = unsafe { ffi::bind_int64(self.raw(), index, value) };
        self.check(status, "bind integer")
    }

    pub(crate) fn bind_blob(&mut self, index: usize, value: &[u8]) -> Result<()> {
        let index = self.bind_index(index)?;
        let length = checked_length(Some(value.len()))?;
        // SAFETY: the ready statement is live, and SQLITE_TRANSIENT copies the
        // full checked buffer before returning. Empty slices still have a
        // non-null address, preserving empty BLOB rather than SQL NULL.
        let status = unsafe {
            ffi::bind_blob(
                self.raw(),
                index,
                value.as_ptr().cast(),
                length,
                ffi::transient(),
            )
        };
        self.check(status, "bind blob")
    }

    pub(crate) fn bind_text(&mut self, index: usize, value: &str) -> Result<()> {
        let index = self.bind_index(index)?;
        let length = checked_length(Some(value.len()))?;
        // SAFETY: this UTF-8 string is readable for the checked byte length;
        // SQLITE_TRANSIENT copies it before return, including interior NULs.
        let status = unsafe {
            ffi::bind_text(
                self.raw(),
                index,
                value.as_ptr().cast(),
                length,
                ffi::transient(),
            )
        };
        self.check(status, "bind text")
    }

    /// Binds a scoped, non-owning pointer that SQL cannot construct or inspect.
    ///
    /// # Safety
    /// The pointer and NUL-terminated tag must follow their private producer/
    /// consumer contract. Its owner must remain pinned and live until this
    /// statement is finalized, including every step/error path. SQLite receives
    /// no destructor and does not own the pointed-to allocation.
    pub(crate) unsafe fn bind_pointer(
        &mut self,
        index: usize,
        value: *mut std::os::raw::c_void,
        tag: *const std::os::raw::c_char,
    ) -> Result<()> {
        let index = self.bind_index(index)?;
        // SAFETY: the caller guarantees the private pointer/tag lifetime; this
        // ready, owned statement remains valid and SQLite retains no ownership.
        let status = unsafe { ffi::bind_pointer(self.raw(), index, value, tag, None) };
        self.check(status, "bind scoped pointer")
    }

    /// Rewinds a completed statement without discarding its bindings. A Row
    /// must be drained through DONE first: resetting unfinished DML can hide a
    /// late error or commit successful effects. Failed steps are never reused.
    pub(crate) fn reset(&mut self) -> Result<()> {
        if !matches!(self.state, State::Ready | State::Done) {
            return Err(IndexError::with_code(
                ffi::SQLITE_MISUSE as c_int,
                "drain a successful statement through DONE before resetting",
            ));
        }
        // SAFETY: this owns a live, completed (or unstepped) statement; no
        // SQLite column views escape the copying accessors.
        let status = unsafe { ffi::reset(self.raw()) };
        if let Err(error) = self.check(status, "reset statement") {
            self.state = State::Failed;
            return Err(error);
        }
        self.state = State::Ready;
        Ok(())
    }

    /// Clears copied buffers and private pointer bindings before reuse.
    pub(crate) fn clear_bindings(&mut self) -> Result<()> {
        if self.state != State::Ready {
            return Err(IndexError::with_code(
                ffi::SQLITE_MISUSE as c_int,
                "reset a completed statement before clearing bindings",
            ));
        }
        // SAFETY: the uniquely owned ready statement is live. SQLite releases
        // only its own binding storage; non-owning pointer bindings have no dtor.
        let status = unsafe { ffi::clear_bindings(self.raw()) };
        self.check(status, "clear statement bindings")
    }

    pub(crate) fn step(&mut self) -> Result<Step> {
        if self.state == State::Done {
            return Ok(Step::Done);
        }
        if self.state == State::Failed {
            return Err(IndexError::with_code(
                ffi::SQLITE_MISUSE as c_int,
                "prepare a new statement after a failed step",
            ));
        }
        // SAFETY: this uniquely borrowed statement is live and ready or positioned
        // on a row. No SQLite column views escape the owned accessor methods.
        let status = unsafe { ffi::step(self.raw()) };
        match status {
            status if status == ffi::SQLITE_ROW as c_int => {
                self.state = State::Row;
                Ok(Step::Row)
            }
            status if status == ffi::SQLITE_DONE as c_int => {
                self.state = State::Done;
                Ok(Step::Done)
            }
            status => {
                self.state = State::Failed;
                Err(self.connection.error(status, "step statement"))
            }
        }
    }

    fn column_index(&self, column: usize) -> Result<c_int> {
        if self.state != State::Row {
            return Err(IndexError::with_code(
                ffi::SQLITE_MISUSE as c_int,
                "statement is not positioned on a row",
            ));
        }
        // SAFETY: the statement remains live and positioned on the current row.
        let count = unsafe { ffi::column_count(self.raw()) };
        c_int::try_from(column)
            .ok()
            .filter(|&column| column < count)
            .ok_or_else(|| {
                IndexError::with_code(ffi::SQLITE_RANGE as c_int, "column index is out of range")
            })
    }

    pub(crate) fn column_type(&self, column: usize) -> Result<c_int> {
        let column = self.column_index(column)?;
        // SAFETY: column_index checked the live row and in-bounds column.
        Ok(unsafe { ffi::column_type(self.raw(), column) })
    }

    pub(crate) fn column_is_null(&self, column: usize) -> Result<bool> {
        Ok(self.column_type(column)? == ffi::SQLITE_NULL as c_int)
    }

    fn typed_column(&self, column: usize, kind: c_int) -> Result<c_int> {
        let index = self.column_index(column)?;
        // SAFETY: the statement is on a row, with a checked column index. This
        // type check does not convert or invalidate any SQLite column value.
        if unsafe { ffi::column_type(self.raw(), index) } != kind {
            return Err(IndexError::with_code(
                ffi::SQLITE_MISMATCH as c_int,
                format!("column {column} has an unexpected SQLite storage type"),
            ));
        }
        Ok(index)
    }

    pub(crate) fn column_i64(&self, column: usize) -> Result<i64> {
        let column = self.typed_column(column, ffi::SQLITE_INTEGER as c_int)?;
        // SAFETY: the checked live column holds an INTEGER; no coercion is needed.
        Ok(unsafe { ffi::column_int64(self.raw(), column) })
    }

    pub(crate) fn column_blob_owned(&self, column: usize) -> Result<Vec<u8>> {
        let column = self.typed_column(column, ffi::SQLITE_BLOB as c_int)?;
        // SAFETY: this is a live BLOB column. Requesting its byte count after its
        // BLOB pointer preserves the pointer; neither API steps/resets the row.
        let (pointer, length) = unsafe {
            let pointer = ffi::column_blob(self.raw(), column).cast::<u8>();
            let length = ffi::column_bytes(self.raw(), column);
            (pointer, length)
        };
        self.copy_column(pointer, length, true)
    }

    pub(crate) fn column_text(&self, column: usize) -> Result<String> {
        let column = self.typed_column(column, ffi::SQLITE_TEXT as c_int)?;
        // SAFETY: obtain the UTF-8 representation before its UTF-8 byte count,
        // and copy before any future statement step or finalization can invalidate it.
        let (pointer, length) = unsafe {
            let pointer = ffi::column_text(self.raw(), column);
            let length = ffi::column_bytes(self.raw(), column);
            (pointer, length)
        };
        String::from_utf8(self.copy_column(pointer, length, false)?).map_err(|_| {
            IndexError::with_code(
                ffi::SQLITE_MISMATCH as c_int,
                "column contains invalid UTF-8",
            )
        })
    }

    fn copy_column(&self, pointer: *const u8, length: c_int, empty_blob: bool) -> Result<Vec<u8>> {
        let length = usize::try_from(length).map_err(|_| {
            IndexError::with_code(
                ffi::SQLITE_CORRUPT as c_int,
                "negative SQLite column byte length",
            )
        })?;
        if pointer.is_null() {
            // SAFETY: the connection outlives the current row. A NULL BLOB
            // pointer may mean either an empty BLOB or an allocation failure;
            // inspect the error before treating zero bytes as a valid value.
            let status = unsafe { ffi::extended_errcode(self.connection.db.as_ptr()) };
            if !(empty_blob && length == 0) || status & 0xff == ffi::SQLITE_NOMEM as c_int {
                return Err(self
                    .connection
                    .error(ffi::SQLITE_NOMEM as c_int, "read column bytes"));
            }
        }
        let mut output = Vec::new();
        output.try_reserve_exact(length).map_err(|_| {
            IndexError::with_code(
                ffi::SQLITE_NOMEM as c_int,
                "cannot allocate SQLite column copy",
            )
        })?;
        if length != 0 {
            // SAFETY: SQLite's current BLOB/text pointer covers this checked byte
            // count until the next step/conversion/finalization. No such call occurs
            // before this synchronous copy; reserve provided full destination.
            output.extend_from_slice(unsafe { std::slice::from_raw_parts(pointer, length) });
        }
        Ok(output)
    }

    /// Finalizes explicitly so a deferred SQLite error can be reported.
    pub(crate) fn finish(mut self) -> Result<()> {
        if let Some(pointer) = self.pointer.take() {
            // SAFETY: take transfers this statement's unique allocation for
            // exactly one finalization; Drop now sees no pointer to finalize.
            let status = unsafe { ffi::finalize(pointer.as_ptr()) };
            self.check(status, "finalize statement")
        } else {
            Ok(())
        }
    }
}

impl Drop for Statement<'_> {
    fn drop(&mut self) {
        if let Some(pointer) = self.pointer.take() {
            // SAFETY: Drop uniquely owns this statement and finalizes it once.
            // The borrowed host connection is not closed. Explicit finish is
            // available when the caller needs to report finalization errors.
            unsafe { ffi::finalize(pointer.as_ptr()) };
        }
    }
}

fn checked_length(length: Option<usize>) -> Result<c_int> {
    length
        .and_then(|length| c_int::try_from(length).ok())
        .ok_or_else(|| {
            IndexError::with_code(
                ffi::SQLITE_TOOBIG as c_int,
                "SQLite byte length is out of range",
            )
        })
}

/// Quotes a SQLite identifier, independently of any surrounding SQL syntax.
pub(crate) fn quote_identifier(identifier: &str) -> Result<String> {
    if identifier.contains('\0') {
        return Err(IndexError::with_code(
            ffi::SQLITE_MISUSE as c_int,
            "SQLite identifier contains a NUL byte",
        ));
    }
    Ok(format!("\"{}\"", identifier.replace('"', "\"\"")))
}

pub(crate) fn qualified_name(schema: &str, table: &str) -> Result<String> {
    Ok(format!(
        "{}.{}",
        quote_identifier(schema)?,
        quote_identifier(table)?
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifier_quoting_keeps_schema_and_table_separate() {
        assert_eq!(quote_identifier("a\"b").unwrap(), "\"a\"\"b\"");
        assert_eq!(
            qualified_name("attached.db", "a\"b").unwrap(),
            "\"attached.db\".\"a\"\"b\""
        );
        assert_eq!(
            quote_identifier("a\0b").unwrap_err().code,
            ffi::SQLITE_MISUSE as c_int
        );
    }

    #[test]
    fn byte_lengths_fail_without_truncation_or_panicking() {
        assert_eq!(checked_length(Some(0)).unwrap(), 0);
        assert_eq!(
            checked_length(Some(c_int::MAX as usize)).unwrap(),
            c_int::MAX
        );
        assert_eq!(
            checked_length(None).unwrap_err().code,
            ffi::SQLITE_TOOBIG as c_int
        );
        assert_eq!(
            checked_length(Some(c_int::MAX as usize + 1))
                .unwrap_err()
                .code,
            ffi::SQLITE_TOOBIG as c_int
        );
    }

    #[test]
    fn reset_and_clear_reject_unfinished_or_failed_states_before_host_access() {
        let connection = Connection {
            db: NonNull::dangling(),
            _lifetime: PhantomData,
            _connection_local: PhantomData,
        };
        for state in [State::Row, State::Failed] {
            // No raw statement/connection is ever accessed: both guards reject
            // these states, and Drop sees no owned statement allocation.
            let mut statement = Statement {
                pointer: None,
                connection: connection.clone(),
                state,
            };
            assert_eq!(
                statement.reset().unwrap_err().code,
                ffi::SQLITE_MISUSE as c_int
            );
            assert_eq!(
                statement.clear_bindings().unwrap_err().code,
                ffi::SQLITE_MISUSE as c_int
            );
        }
        let mut statement = Statement {
            pointer: None,
            connection,
            state: State::Done,
        };
        assert_eq!(
            statement.clear_bindings().unwrap_err().code,
            ffi::SQLITE_MISUSE as c_int
        );
    }

    #[test]
    fn null_connection_is_rejected_before_host_access() {
        // SAFETY: null is checked before any host API access; no handle is made.
        let error = unsafe { Connection::borrow(std::ptr::null_mut()) }
            .err()
            .unwrap();
        assert_eq!(error.code, ffi::SQLITE_MISUSE as c_int);
    }
}
