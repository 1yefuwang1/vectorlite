//! Pure verification of one SQLite CREATE VIRTUAL TABLE declaration target.
//!
//! The SQL owner reads exactly one bounded sqlite_schema.sql row for a table in
//! an exact schema and asks this module whether that declaration authoritatively
//! names the vectorlite module with exactly two supported DiskANN arguments for
//! the expected schema/table. This module executes no SQL, holds no SQLite/FFI
//! pointer, performs no database lookup, and never derives authority from a
//! substring such as "USING vectorlite". Grammar positions and SQLite identifier
//! rules decide the module and target; a quoted literal, comment, other module,
//! ordinary table/view or malformed statement is rejected.
//!
//! The lexer handles SQL whitespace, line/block comments, double/backtick/bracket
//! quoted identifiers (with doubled-quote escapes), single-quoted string spans,
//! nested parentheses and top-level comma splitting. Module arguments are
//! returned trimmed of outer whitespace and otherwise verbatim, matching SQLite's
//! xCreate argv text (SQLite does not dequote or strip comments from virtual
//! table argument text). Argument parsing is delegated unchanged to the existing
//! vector-space and index-options parsers; this module does not broaden syntax.
//!
//! This is a pure T4 target-verification seam. The root SQL owner remains
//! responsible for reading the exact row, UUID/layout/lifetime/job identity and
//! binding the verified declaration to candidate ownership.
#![allow(dead_code)] // T4 seam: root resolver registration follows.

use std::fmt;

use crate::index_options::{BackendOptions, DiskAnnOptions};
use crate::vector_space::{parse_named_vector_space, DistanceType, NamedVectorSpace, VectorType};

/// Absolute upper bound for one sqlite_schema declaration text.
pub const MAX_DECLARATION_BYTES: usize = 16 * 1024;
/// The only virtual-table module this parser authorizes.
pub const MODULE_NAME: &str = "vectorlite";

#[derive(Clone, Debug)]
pub struct DeclarationTarget {
    /// Schema associated with the declaration: the parsed qualifier, or the
    /// caller-supplied actual row schema for the unqualified form SQLite stores.
    pub schema: String,
    /// Parsed/dequoted target table name, verified against the caller's row value.
    pub table: String,
    /// Parsed/dequoted module name (always vectorlite after verification).
    pub module: String,
    pub vector_space: NamedVectorSpace,
    pub diskann: DiskAnnOptions,
    /// Exact module argument text after outer SQL-whitespace trimming.
    pub vector_space_argument: String,
    pub index_options_argument: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetErrorKind {
    Limit,
    Empty,
    Malformed,
    NotCreateVirtualTable,
    NotVirtualTable,
    MissingUsing,
    InvalidIdentifier,
    InvalidExpectedIdentity,
    TargetMismatch,
    ModuleMismatch,
    ArgumentShape,
    InvalidVectorSpace,
    InvalidOptions,
    NotDiskann,
    UnsupportedVectorType,
    UnsupportedMetric,
    TrailingInput,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TargetError {
    Limit { max_bytes: usize },
    Empty,
    Malformed { detail: &'static str },
    NotCreateVirtualTable,
    NotVirtualTable,
    MissingUsing,
    InvalidIdentifier { detail: &'static str },
    InvalidExpectedIdentity,
    TargetMismatch,
    ModuleMismatch,
    ArgumentShape,
    InvalidVectorSpace { message: String },
    InvalidOptions { message: String },
    NotDiskann,
    UnsupportedVectorType,
    UnsupportedMetric,
    TrailingInput,
}

impl TargetError {
    pub fn kind(&self) -> TargetErrorKind {
        match self {
            Self::Limit { .. } => TargetErrorKind::Limit,
            Self::Empty => TargetErrorKind::Empty,
            Self::Malformed { .. } => TargetErrorKind::Malformed,
            Self::NotCreateVirtualTable => TargetErrorKind::NotCreateVirtualTable,
            Self::NotVirtualTable => TargetErrorKind::NotVirtualTable,
            Self::MissingUsing => TargetErrorKind::MissingUsing,
            Self::InvalidIdentifier { .. } => TargetErrorKind::InvalidIdentifier,
            Self::InvalidExpectedIdentity => TargetErrorKind::InvalidExpectedIdentity,
            Self::TargetMismatch => TargetErrorKind::TargetMismatch,
            Self::ModuleMismatch => TargetErrorKind::ModuleMismatch,
            Self::ArgumentShape => TargetErrorKind::ArgumentShape,
            Self::InvalidVectorSpace { .. } => TargetErrorKind::InvalidVectorSpace,
            Self::InvalidOptions { .. } => TargetErrorKind::InvalidOptions,
            Self::NotDiskann => TargetErrorKind::NotDiskann,
            Self::UnsupportedVectorType => TargetErrorKind::UnsupportedVectorType,
            Self::UnsupportedMetric => TargetErrorKind::UnsupportedMetric,
            Self::TrailingInput => TargetErrorKind::TrailingInput,
        }
    }
}

impl fmt::Display for TargetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Limit { max_bytes } => write!(
                formatter,
                "sqlite_schema declaration exceeds the {max_bytes}-byte parsing bound"
            ),
            Self::Empty => formatter.write_str("sqlite_schema declaration is empty"),
            Self::Malformed { detail } => write!(formatter, "malformed declaration: {detail}"),
            Self::NotCreateVirtualTable => {
                formatter.write_str("declaration is not a CREATE statement")
            }
            Self::NotVirtualTable => formatter.write_str("declaration is not CREATE VIRTUAL TABLE"),
            Self::MissingUsing => {
                formatter.write_str("declaration is missing the USING module clause")
            }
            Self::InvalidIdentifier { detail } => {
                write!(formatter, "invalid SQLite identifier: {detail}")
            }
            Self::InvalidExpectedIdentity => formatter
                .write_str("expected schema/table identity must be non-empty and contain no NUL"),
            Self::TargetMismatch => formatter.write_str(
                "declared schema/table does not match the exact sqlite_schema row identity",
            ),
            Self::ModuleMismatch => {
                formatter.write_str("declaration does not name the vectorlite module")
            }
            Self::ArgumentShape => formatter.write_str(
                "declaration requires exactly two non-empty vectorlite module arguments",
            ),
            Self::InvalidVectorSpace { message } => {
                write!(formatter, "invalid vector space argument: {message}")
            }
            Self::InvalidOptions { message } => {
                write!(formatter, "invalid index options argument: {message}")
            }
            Self::NotDiskann => {
                formatter.write_str("declaration is not a DiskANN table (HNSW options rejected)")
            }
            Self::UnsupportedVectorType => {
                formatter.write_str("DiskANN requires the float32 vector type")
            }
            Self::UnsupportedMetric => {
                formatter.write_str("DiskANN requires the l2 or cosine distance type")
            }
            Self::TrailingInput => {
                formatter.write_str("declaration has trailing tokens after the module clause")
            }
        }
    }
}

impl std::error::Error for TargetError {}

/// SQLite identifier equality: ASCII case-insensitive, otherwise byte-exact.
///
/// Non-ASCII bytes compare exactly because SQLite's built-in NOCASE folds only
/// ASCII; SQLite's own name lookup has the same practical limitation.
fn sqlite_identifier_eq(left: &str, right: &str) -> bool {
    left.len() == right.len()
        && left
            .bytes()
            .zip(right.bytes())
            .all(|(left, right)| left.eq_ignore_ascii_case(&right))
}

fn is_sql_space(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

struct Lexer<'a> {
    text: &'a str,
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Lexer<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            text,
            bytes: text.as_bytes(),
            position: 0,
        }
    }

    fn eof(&self) -> bool {
        self.position >= self.bytes.len()
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.position).copied()
    }

    fn peek_next(&self) -> Option<u8> {
        self.bytes.get(self.position + 1).copied()
    }

    /// Skips SQL whitespace, line comments and block comments.
    fn skip_trivia(&mut self) -> Result<(), TargetError> {
        loop {
            while let Some(byte) = self.peek() {
                if is_sql_space(byte) {
                    self.position += 1;
                } else {
                    break;
                }
            }
            if self.peek() == Some(b'-') && self.peek_next() == Some(b'-') {
                self.position += 2;
                while let Some(byte) = self.peek() {
                    self.position += 1;
                    if byte == b'\n' {
                        break;
                    }
                }
                continue;
            }
            if self.peek() == Some(b'/') && self.peek_next() == Some(b'*') {
                self.position += 2;
                let mut closed = false;
                while let Some(byte) = self.peek() {
                    self.position += 1;
                    if byte == b'*' && self.peek() == Some(b'/') {
                        self.position += 1;
                        closed = true;
                        break;
                    }
                }
                if !closed {
                    return Err(TargetError::Malformed {
                        detail: "unterminated block comment",
                    });
                }
                continue;
            }
            return Ok(());
        }
    }

    fn scan_bare_identifier(&mut self) -> Option<&'a str> {
        let start = self.position;
        let first = *self.bytes.get(start)?;
        if !(first.is_ascii_alphabetic() || first == b'_' || first >= 0x80) {
            return None;
        }
        self.position += 1;
        while let Some(byte) = self.peek() {
            if byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'$') || byte >= 0x80 {
                self.position += 1;
            } else {
                break;
            }
        }
        Some(&self.text[start..self.position])
    }

    fn take_keyword(&mut self, keyword: &str) -> Result<bool, TargetError> {
        let saved = self.position;
        self.skip_trivia()?;
        match self.scan_bare_identifier() {
            Some(word) if word.eq_ignore_ascii_case(keyword) => Ok(true),
            _ => {
                self.position = saved;
                Ok(false)
            }
        }
    }

    fn expect_keyword(&mut self, keyword: &str, detail: &'static str) -> Result<(), TargetError> {
        if self.take_keyword(keyword)? {
            Ok(())
        } else {
            Err(TargetError::Malformed { detail })
        }
    }

    fn consume_doubled_quoted(&mut self, quote: u8) -> Result<String, TargetError> {
        self.position += 1;
        let mut segment_start = self.position;
        let mut output = String::new();
        loop {
            let Some(byte) = self.peek() else {
                return Err(TargetError::Malformed {
                    detail: "unterminated quoted identifier",
                });
            };
            if byte == quote {
                output.push_str(&self.text[segment_start..self.position]);
                if self.peek_next() == Some(quote) {
                    output.push(quote as char);
                    self.position += 2;
                    segment_start = self.position;
                } else {
                    self.position += 1;
                    return Ok(output);
                }
            } else {
                self.position += 1;
            }
        }
    }

    fn consume_bracket_quoted(&mut self) -> Result<String, TargetError> {
        self.position += 1;
        let start = self.position;
        loop {
            let Some(byte) = self.peek() else {
                return Err(TargetError::Malformed {
                    detail: "unterminated bracket identifier",
                });
            };
            if byte == b']' {
                let value = self.text[start..self.position].to_owned();
                self.position += 1;
                return Ok(value);
            }
            self.position += 1;
        }
    }

    fn consume_identifier(&mut self, detail: &'static str) -> Result<String, TargetError> {
        self.skip_trivia()?;
        match self.peek() {
            Some(b'"') => self.consume_doubled_quoted(b'"'),
            Some(0x60) => self.consume_doubled_quoted(0x60),
            Some(b'[') => self.consume_bracket_quoted(),
            Some(b'\'') => Err(TargetError::InvalidIdentifier {
                detail: "single-quoted string is not a target identifier",
            }),
            Some(_) => match self.scan_bare_identifier() {
                Some(word) if !word.is_empty() => Ok(word.to_owned()),
                _ => Err(TargetError::InvalidIdentifier { detail }),
            },
            None => Err(TargetError::InvalidIdentifier { detail }),
        }
    }

    fn take_dot(&mut self) -> Result<bool, TargetError> {
        self.skip_trivia()?;
        if self.peek() == Some(b'.') {
            self.position += 1;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn skip_single_quoted(&mut self) -> Result<(), TargetError> {
        self.position += 1;
        loop {
            let Some(byte) = self.peek() else {
                return Err(TargetError::Malformed {
                    detail: "unterminated string literal",
                });
            };
            if byte == b'\'' {
                if self.peek_next() == Some(b'\'') {
                    self.position += 2;
                } else {
                    self.position += 1;
                    return Ok(());
                }
            } else {
                self.position += 1;
            }
        }
    }

    fn skip_bracket_span(&mut self) -> Result<(), TargetError> {
        self.position += 1;
        loop {
            let Some(byte) = self.peek() else {
                return Err(TargetError::Malformed {
                    detail: "unterminated bracket identifier",
                });
            };
            self.position += 1;
            if byte == b']' {
                return Ok(());
            }
        }
    }

    fn skip_quoted_span(&mut self, quote: u8) -> Result<(), TargetError> {
        self.position += 1;
        loop {
            let Some(byte) = self.peek() else {
                return Err(TargetError::Malformed {
                    detail: "unterminated quoted identifier",
                });
            };
            if byte == quote {
                if self.peek_next() == Some(quote) {
                    self.position += 2;
                } else {
                    self.position += 1;
                    return Ok(());
                }
            } else {
                self.position += 1;
            }
        }
    }

    /// Consumes the module argument list starting immediately after the opening
    /// parenthesis. Returns exactly two trimmed argument texts and stops after
    /// the matching closing parenthesis.
    fn take_module_arguments(&mut self) -> Result<(String, String), TargetError> {
        let mut spans = [None, None];
        let mut argument_count = 0usize;
        let mut argument_start = self.position;
        let mut depth = 0usize;
        loop {
            if self.eof() {
                return Err(TargetError::Malformed {
                    detail: "unterminated module argument list",
                });
            }
            let byte = self.peek().expect("checked eof");
            match byte {
                b'\'' => self.skip_single_quoted()?,
                b'"' => self.skip_quoted_span(b'"')?,
                0x60 => self.skip_quoted_span(0x60)?,
                b'[' => self.skip_bracket_span()?,
                b'-' if self.peek_next() == Some(b'-') => {
                    self.position += 2;
                    while let Some(next) = self.peek() {
                        self.position += 1;
                        if next == b'\n' {
                            break;
                        }
                    }
                }
                b'/' if self.peek_next() == Some(b'*') => {
                    self.position += 2;
                    let mut closed = false;
                    while let Some(next) = self.peek() {
                        self.position += 1;
                        if next == b'*' && self.peek() == Some(b'/') {
                            self.position += 1;
                            closed = true;
                            break;
                        }
                    }
                    if !closed {
                        return Err(TargetError::Malformed {
                            detail: "unterminated block comment",
                        });
                    }
                }
                b'(' => {
                    depth += 1;
                    self.position += 1;
                }
                b')' if depth == 0 => {
                    self.finish_argument(
                        argument_start,
                        self.position,
                        &mut spans,
                        argument_count,
                    )?;
                    argument_count += 1;
                    self.position += 1;
                    break;
                }
                b')' => {
                    depth -= 1;
                    self.position += 1;
                }
                b',' if depth == 0 => {
                    self.finish_argument(
                        argument_start,
                        self.position,
                        &mut spans,
                        argument_count,
                    )?;
                    argument_count += 1;
                    if argument_count > 2 {
                        return Err(TargetError::ArgumentShape);
                    }
                    self.position += 1;
                    argument_start = self.position;
                }
                _ => self.position += 1,
            }
        }
        if argument_count != 2 {
            return Err(TargetError::ArgumentShape);
        }
        match spans {
            [Some(first), Some(second)] => Ok((first, second)),
            _ => Err(TargetError::ArgumentShape),
        }
    }

    fn finish_argument(
        &self,
        start: usize,
        end: usize,
        spans: &mut [Option<String>; 2],
        index: usize,
    ) -> Result<(), TargetError> {
        if index >= spans.len() {
            return Err(TargetError::ArgumentShape);
        }
        let mut trimmed = &self.text[start..end];
        while let Some(first) = trimmed.as_bytes().first() {
            if is_sql_space(*first) {
                trimmed = &trimmed[1..];
            } else {
                break;
            }
        }
        while let Some(last) = trimmed.as_bytes().last() {
            if is_sql_space(*last) {
                trimmed = &trimmed[..trimmed.len() - 1];
            } else {
                break;
            }
        }
        if trimmed.is_empty() {
            return Err(TargetError::ArgumentShape);
        }
        spans[index] = Some(trimmed.to_owned());
        Ok(())
    }
}

fn valid_expected_identity(value: &str) -> bool {
    !value.is_empty() && !value.as_bytes().contains(&0)
}

/// Parses and verifies one CREATE VIRTUAL TABLE declaration for an exact row.
///
/// expected_schema/expected_table are the caller's actual sqlite_schema schema
/// and name values. max_bytes must be 1..=MAX_DECLARATION_BYTES and the input must
/// fit it. No SQL is executed and no database lookup is performed.
pub fn parse_declaration(
    sql: &str,
    expected_schema: &str,
    expected_table: &str,
    max_bytes: usize,
) -> Result<DeclarationTarget, TargetError> {
    if max_bytes == 0 || max_bytes > MAX_DECLARATION_BYTES {
        return Err(TargetError::Limit { max_bytes });
    }
    if sql.len() > max_bytes {
        return Err(TargetError::Limit { max_bytes });
    }
    if !valid_expected_identity(expected_schema) || !valid_expected_identity(expected_table) {
        return Err(TargetError::InvalidExpectedIdentity);
    }
    if sql.trim().is_empty() {
        return Err(TargetError::Empty);
    }
    let mut lexer = Lexer::new(sql);
    if !lexer.take_keyword("CREATE")? {
        return Err(TargetError::NotCreateVirtualTable);
    }
    let temporary = if lexer.take_keyword("TEMP")? {
        true
    } else {
        lexer.take_keyword("TEMPORARY")?
    };
    if !lexer.take_keyword("VIRTUAL")? {
        if lexer.take_keyword("TABLE")? || lexer.take_keyword("VIEW")? {
            return Err(TargetError::NotVirtualTable);
        }
        return Err(TargetError::Malformed {
            detail: "expected VIRTUAL after CREATE",
        });
    }
    lexer.expect_keyword("TABLE", "expected TABLE after VIRTUAL")?;
    if lexer.take_keyword("IF")? {
        lexer.expect_keyword("NOT", "expected NOT after IF")?;
        lexer.expect_keyword("EXISTS", "expected EXISTS after IF NOT")?;
    }
    let first = lexer.consume_identifier("expected target table identifier")?;
    let (schema_qualifier, table) = if lexer.take_dot()? {
        let table = lexer.consume_identifier("expected table identifier after schema")?;
        if lexer.take_dot()? {
            return Err(TargetError::TargetMismatch);
        }
        (Some(first), table)
    } else {
        (None, first)
    };
    if !lexer.take_keyword("USING")? {
        return Err(TargetError::MissingUsing);
    }
    let module = lexer.consume_identifier("expected module identifier after USING")?;
    lexer.skip_trivia()?;
    if lexer.peek() != Some(b'(') {
        return Err(TargetError::ArgumentShape);
    }
    lexer.position += 1;
    let (vector_space_argument, index_options_argument) = lexer.take_module_arguments()?;
    lexer.skip_trivia()?;
    if !lexer.eof() {
        return Err(TargetError::TrailingInput);
    }
    if !sqlite_identifier_eq(&module, MODULE_NAME) {
        return Err(TargetError::ModuleMismatch);
    }
    if temporary {
        if let Some(qualifier) = &schema_qualifier {
            if !sqlite_identifier_eq(qualifier, "temp") {
                return Err(TargetError::TargetMismatch);
            }
        }
        if !sqlite_identifier_eq(expected_schema, "temp") {
            return Err(TargetError::TargetMismatch);
        }
    } else if let Some(qualifier) = &schema_qualifier {
        if !sqlite_identifier_eq(qualifier, expected_schema) {
            return Err(TargetError::TargetMismatch);
        }
    }
    if !sqlite_identifier_eq(&table, expected_table) {
        return Err(TargetError::TargetMismatch);
    }
    let space = parse_named_vector_space(&vector_space_argument)
        .map_err(|message| TargetError::InvalidVectorSpace { message })?;
    let options = BackendOptions::parse(&index_options_argument)
        .map_err(|message| TargetError::InvalidOptions { message })?;
    let BackendOptions::Diskann(diskann) = options else {
        return Err(TargetError::NotDiskann);
    };
    if space.vector_type != VectorType::Float32 {
        return Err(TargetError::UnsupportedVectorType);
    }
    match space.distance_type {
        DistanceType::L2 | DistanceType::Cosine => {}
        DistanceType::InnerProduct => return Err(TargetError::UnsupportedMetric),
    }
    let schema = schema_qualifier.unwrap_or_else(|| expected_schema.to_owned());
    Ok(DeclarationTarget {
        schema,
        table,
        module,
        vector_space: space,
        diskann,
        vector_space_argument,
        index_options_argument,
    })
}

#[cfg(test)]
#[path = "diskann_target/tests.rs"]
mod tests;
