//! Backend errors retain the host SQLite status, including extended I/O codes.

use std::fmt;
use std::os::raw::c_int;

use crate::ffi;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct IndexError {
    pub(crate) code: c_int,
    pub(crate) message: String,
}

impl IndexError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self::with_code(ffi::SQLITE_ERROR as c_int, message)
    }

    pub(crate) fn with_code(code: c_int, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl fmt::Display for IndexError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for IndexError {}

impl From<String> for IndexError {
    fn from(message: String) -> Self {
        Self::new(message)
    }
}

impl From<&str> for IndexError {
    fn from(message: &str) -> Self {
        Self::new(message)
    }
}
