//! Pure T4 target-declaration parser qualification tests.

use super::*;

const SPACE: &str = "embedding float32[2]";
const OPTIONS: &str = "diskann()";

fn ddl(space: &str, options: &str) -> String {
    format!("CREATE VIRTUAL TABLE t USING vectorlite({space}, {options})")
}

fn parse_main(sql: &str) -> Result<DeclarationTarget, TargetError> {
    parse_declaration(sql, "main", "t", MAX_DECLARATION_BYTES)
}

fn expect_ok(sql: &str, schema: &str, table: &str) -> DeclarationTarget {
    parse_declaration(sql, schema, table, MAX_DECLARATION_BYTES).expect("valid declaration")
}

fn expect_kind(sql: &str, kind: TargetErrorKind) {
    let error = parse_main(sql).expect_err("expected rejection");
    assert_eq!(error.kind(), kind, "{error}");
}

#[test]
fn minimal_declaration_returns_owned_parsed_fields() {
    let target = expect_ok(&ddl(SPACE, OPTIONS), "main", "t");
    assert_eq!(target.schema, "main");
    assert_eq!(target.table, "t");
    assert_eq!(target.module, "vectorlite");
    assert_eq!(target.vector_space_argument, SPACE);
    assert_eq!(target.index_options_argument, OPTIONS);
    assert_eq!(target.vector_space.vector_name, "embedding");
    assert_eq!(target.vector_space.dim, 2);
    assert_eq!(target.vector_space.vector_type, VectorType::Float32);
    assert_eq!(target.vector_space.distance_type, DistanceType::L2);
    assert_eq!(target.diskann, DiskAnnOptions::default());
}

#[test]
fn qualified_quoted_module_and_custom_options() {
    let sql = concat!(
        "CREATE VIRTUAL TABLE \"main\".\"My T\" USING \"vectorlite\"(",
        "embedding float32[2] cosine, ",
        "diskann(degree=48, build_list_size=120, search_list_size=80, max_visits=200000)",
        ")"
    );
    let target = expect_ok(sql, "main", "My T");
    assert_eq!(target.schema, "main");
    assert_eq!(target.table, "My T");
    assert_eq!(target.module, "vectorlite");
    assert_eq!(target.vector_space.distance_type, DistanceType::Cosine);
    assert_eq!(target.diskann.degree, 48);
    assert_eq!(target.diskann.build_list_size, 120);
    assert_eq!(target.diskann.search_list_size, 80);
    assert_eq!(target.diskann.max_visits, 200_000);
}

#[test]
fn keywords_are_case_insensitive_but_argument_text_is_verbatim() {
    let sql = concat!(
        "cReAtE vIrTuAl TaBlE \"MAIN\".\"T\" uSiNg ",
        "VeCtOrLiTe(embedding  float32[2]  , diskann( degree = 32 ))"
    );
    let target = expect_ok(sql, "main", "t");
    assert_eq!(target.table, "T");
    assert_eq!(target.vector_space_argument, "embedding  float32[2]");
    assert_eq!(target.index_options_argument, "diskann( degree = 32 )");
    assert_eq!(target.diskann.degree, 32);
}

#[test]
fn comments_and_whitespace_are_ignored_outside_arguments() {
    let sql = concat!(
        "/* lead */ CREATE -- comment\n",
        "VIRTUAL /*x*/ TABLE /*y*/ \"t\" /*z*/ USING ",
        "-- module\n vectorlite(embedding float32[2], diskann()) -- tail comment"
    );
    let target = expect_ok(sql, "main", "t");
    assert_eq!(target.table, "t");
    assert_eq!(target.vector_space_argument, SPACE);
}

#[test]
fn if_not_exists_and_attached_schema_qualifier() {
    let sql = "CREATE VIRTUAL TABLE IF NOT EXISTS \"aux\".\"t\" USING vectorlite(embedding float32[2] cosine, diskann())";
    let target = expect_ok(sql, "aux", "t");
    assert_eq!(target.schema, "aux");
    assert_eq!(target.table, "t");
}

#[test]
fn temp_keyword_and_temp_qualified_targets() {
    let sql = "CREATE TEMP VIRTUAL TABLE t USING vectorlite(embedding float32[2], diskann())";
    let target = expect_ok(sql, "temp", "t");
    assert_eq!(target.schema, "temp");
    let sql = "CREATE TEMPORARY VIRTUAL TABLE temp.\"t\" USING vectorlite(embedding float32[2], diskann())";
    let target = expect_ok(sql, "temp", "t");
    assert_eq!(target.schema, "temp");
    // SQLite stores temp virtual tables unqualified with no TEMP keyword.
    let stored = "CREATE VIRTUAL TABLE t USING vectorlite(embedding float32[2], diskann())";
    assert_eq!(expect_ok(stored, "temp", "t").schema, "temp");
    assert_eq!(
        parse_declaration(sql, "main", "t", MAX_DECLARATION_BYTES)
            .unwrap_err()
            .kind(),
        TargetErrorKind::TargetMismatch
    );
}

#[test]
fn quoted_table_name_escapes_and_dots() {
    let sql = "CREATE VIRTUAL TABLE \"a\"\"b\" USING vectorlite(embedding float32[2], diskann())";
    assert_eq!(expect_ok(sql, "main", "a\"b").table, "a\"b");

    let bt = "\u{60}";
    let sql = format!(
        "CREATE VIRTUAL TABLE {bt}a{bt}{bt}b{bt} USING vectorlite(embedding float32[2], diskann())"
    );
    assert_eq!(expect_ok(&sql, "main", "a\u{60}b").table, "a\u{60}b");

    let sql = "CREATE VIRTUAL TABLE [a b] USING vectorlite(embedding float32[2], diskann())";
    assert_eq!(expect_ok(sql, "main", "a b").table, "a b");

    let sql = "CREATE VIRTUAL TABLE \"a.b\" USING vectorlite(embedding float32[2], diskann())";
    assert_eq!(expect_ok(sql, "main", "a.b").table, "a.b");
}

#[test]
fn module_identifier_quote_forms_are_accepted() {
    let bt = "\u{60}";
    for sql in [
        "CREATE VIRTUAL TABLE t USING [vectorlite](embedding float32[2], diskann())".to_owned(),
        "CREATE VIRTUAL TABLE t USING \"vectorlite\"(embedding float32[2], diskann())".to_owned(),
        format!("CREATE VIRTUAL TABLE t USING {bt}vectorlite{bt}(embedding float32[2], diskann())"),
    ] {
        assert_eq!(expect_ok(&sql, "main", "t").module, "vectorlite");
    }
}

#[test]
fn nested_parentheses_and_commas_keep_two_arguments() {
    let sql = ddl(
        SPACE,
        "diskann(degree=48, build_list_size=120, alpha=1.2, max_visits=100000)",
    );
    let target = expect_ok(&sql, "main", "t");
    assert_eq!(target.diskann.degree, 48);
    assert_eq!(target.diskann.build_list_size, 120);
}

#[test]
fn commas_and_parentheses_inside_string_or_comment_spans_do_not_split_arguments() {
    let sql = "CREATE VIRTUAL TABLE t USING vectorlite('a,b)', diskann())";
    let error = parse_main(sql).expect_err("quoted vector argument is not valid text");
    assert_eq!(error.kind(), TargetErrorKind::InvalidVectorSpace);

    let sql = "CREATE VIRTUAL TABLE t USING vectorlite(embedding float32[2] /* ), */, diskann())";
    let error = parse_main(sql).expect_err("comment inside argument is preserved");
    assert_eq!(error.kind(), TargetErrorKind::InvalidVectorSpace);

    let sql = "CREATE VIRTUAL TABLE t USING vectorlite(embedding float32[2], 'diskann(degree=32, max_visits=100000)')";
    let error = parse_main(sql).expect_err("quoted options argument is not valid text");
    assert_eq!(error.kind(), TargetErrorKind::InvalidOptions);
}

#[test]
fn ordinary_table_and_view_are_rejected() {
    expect_kind(
        "CREATE TABLE t(x INTEGER)",
        TargetErrorKind::NotVirtualTable,
    );
    expect_kind(
        "CREATE VIEW t AS SELECT 1",
        TargetErrorKind::NotVirtualTable,
    );
    expect_kind("DROP TABLE t", TargetErrorKind::NotCreateVirtualTable);
    expect_kind("SELECT 1", TargetErrorKind::NotCreateVirtualTable);
    expect_kind("CREATE INDEX t ON x(y)", TargetErrorKind::Malformed);
}

#[test]
fn other_module_names_and_keyword_spoofing_are_rejected() {
    expect_kind(
        "CREATE VIRTUAL TABLE t USING fts5(embedding float32[2], diskann())",
        TargetErrorKind::ModuleMismatch,
    );
    expect_kind(
        "CREATE VIRTUAL TABLE t USING vectorlite_extra(embedding float32[2], diskann())",
        TargetErrorKind::ModuleMismatch,
    );
    expect_kind(
        "CREATE VIRTUAL TABLE t USING other(embedding float32[2], diskann()) /* USING vectorlite */",
        TargetErrorKind::ModuleMismatch,
    );
    expect_kind(
        "CREATE TABLE t(x); -- USING vectorlite(embedding float32[2], diskann())",
        TargetErrorKind::NotVirtualTable,
    );
    expect_kind(
        "CREATE VIRTUAL TABLE t /* USING vectorlite(embedding float32[2], diskann()) */ USING other(embedding float32[2], diskann())",
        TargetErrorKind::ModuleMismatch,
    );
}

#[test]
fn target_identity_must_match_the_exact_row() {
    expect_kind(
        "CREATE VIRTUAL TABLE other USING vectorlite(embedding float32[2], diskann())",
        TargetErrorKind::TargetMismatch,
    );
    expect_kind(
        "CREATE VIRTUAL TABLE aux.t USING vectorlite(embedding float32[2], diskann())",
        TargetErrorKind::TargetMismatch,
    );
    let qualified = expect_ok(
        "CREATE VIRTUAL TABLE main.t USING vectorlite(embedding float32[2], diskann())",
        "main",
        "t",
    );
    assert_eq!(qualified.schema, "main");
    assert_eq!(qualified.table, "t");
    expect_kind(
        "CREATE TEMP VIRTUAL TABLE main.t USING vectorlite(embedding float32[2], diskann())",
        TargetErrorKind::TargetMismatch,
    );
    expect_kind(
        "CREATE VIRTUAL TABLE a.b.c USING vectorlite(embedding float32[2], diskann())",
        TargetErrorKind::TargetMismatch,
    );
    assert_eq!(
        parse_declaration(
            "CREATE VIRTUAL TABLE t USING vectorlite(embedding float32[2], diskann())",
            "",
            "t",
            MAX_DECLARATION_BYTES
        )
        .unwrap_err()
        .kind(),
        TargetErrorKind::InvalidExpectedIdentity
    );
}

#[test]
fn malformed_grammar_missing_using_and_argument_shapes_are_rejected() {
    expect_kind(
        "CREATE VIRTUAL TABLE t vectorlite(embedding float32[2], diskann())",
        TargetErrorKind::MissingUsing,
    );
    expect_kind(
        "CREATE VIRTUAL TABLE t USING vectorlite",
        TargetErrorKind::ArgumentShape,
    );
    expect_kind(
        "CREATE VIRTUAL TABLE t USING vectorlite()",
        TargetErrorKind::ArgumentShape,
    );
    expect_kind(
        "CREATE VIRTUAL TABLE t USING vectorlite(embedding float32[2])",
        TargetErrorKind::ArgumentShape,
    );
    expect_kind(
        "CREATE VIRTUAL TABLE t USING vectorlite(embedding float32[2], diskann(), extra)",
        TargetErrorKind::ArgumentShape,
    );
    expect_kind(
        "CREATE VIRTUAL TABLE t USING vectorlite(, diskann())",
        TargetErrorKind::ArgumentShape,
    );
    expect_kind(
        "CREATE VIRTUAL TABLE t USING vectorlite(embedding float32[2],)",
        TargetErrorKind::ArgumentShape,
    );
    expect_kind(
        "CREATE VIRTUAL TABLE t USING vectorlite(embedding float32[2], diskann(",
        TargetErrorKind::Malformed,
    );
    expect_kind(
        "CREATE VIRTUAL TABLE t USING vectorlite(embedding float32[2], 'diskann())",
        TargetErrorKind::Malformed,
    );
    expect_kind(
        "CREATE VIRTUAL TABLE t USING vectorlite(embedding float32[2], diskann()) /* unterminated",
        TargetErrorKind::Malformed,
    );
}

#[test]
fn trailing_input_and_trailing_semicolon_are_rejected() {
    expect_kind(
        &format!("{} extra", ddl(SPACE, OPTIONS)),
        TargetErrorKind::TrailingInput,
    );
    expect_kind(
        &format!("{};", ddl(SPACE, OPTIONS)),
        TargetErrorKind::TrailingInput,
    );
    expect_kind(
        &format!("{} SELECT 1", ddl(SPACE, OPTIONS)),
        TargetErrorKind::TrailingInput,
    );
}

#[test]
fn unsupported_backends_space_types_and_metrics_are_rejected() {
    let hnsw = ddl("embedding float32[2]", "hnsw(max_elements=8)");
    expect_kind(&hnsw, TargetErrorKind::NotDiskann);

    let ip = ddl("embedding float32[2] ip", OPTIONS);
    expect_kind(&ip, TargetErrorKind::UnsupportedMetric);

    let invalid_dtype = ddl("embedding notatype[2]", OPTIONS);
    let error = parse_main(&invalid_dtype).expect_err("invalid dtype");
    assert_eq!(error.kind(), TargetErrorKind::InvalidVectorSpace);
    assert!(error.to_string().contains("Invalid vector type"));

    let float16 = ddl("embedding float16[2] l2", OPTIONS);
    expect_kind(&float16, TargetErrorKind::UnsupportedVectorType);
    let bfloat16 = ddl("embedding bfloat16[2] cosine", OPTIONS);
    expect_kind(&bfloat16, TargetErrorKind::UnsupportedVectorType);

    let bad_options = ddl(SPACE, "diskann(unknown=1)");
    expect_kind(&bad_options, TargetErrorKind::InvalidOptions);

    let quoted_space = ddl("'embedding float32[2]'", OPTIONS);
    let error = parse_main(&quoted_space).expect_err("quoted space");
    assert_eq!(error.kind(), TargetErrorKind::InvalidVectorSpace);
}

#[test]
fn bounded_input_and_limits() {
    let sql = ddl(SPACE, OPTIONS);
    assert_eq!(
        parse_declaration(&sql, "main", "t", 0).unwrap_err().kind(),
        TargetErrorKind::Limit
    );
    assert_eq!(
        parse_declaration(&sql, "main", "t", MAX_DECLARATION_BYTES + 1)
            .unwrap_err()
            .kind(),
        TargetErrorKind::Limit
    );
    assert_eq!(
        parse_declaration(&sql, "main", "t", sql.len() - 1)
            .unwrap_err()
            .kind(),
        TargetErrorKind::Limit
    );
    let oversized = format!(
        "{}{}",
        ddl(SPACE, OPTIONS),
        " ".repeat(MAX_DECLARATION_BYTES)
    );
    assert_eq!(
        parse_declaration(&oversized, "main", "t", MAX_DECLARATION_BYTES)
            .unwrap_err()
            .kind(),
        TargetErrorKind::Limit
    );
    expect_kind("   ", TargetErrorKind::Empty);
}

#[test]
fn line_comments_and_crlf_are_lexed() {
    let sql = "CREATE -- one\r\nVIRTUAL\tTABLE t USING vectorlite(embedding float32[2], diskann())";
    let target = expect_ok(sql, "main", "t");
    assert_eq!(target.table, "t");
}

#[test]
fn quoted_schema_forms_and_case_are_verified() {
    for (sql, schema) in [
        (
            "CREATE VIRTUAL TABLE \"MAIN\".t USING vectorlite(embedding float32[2], diskann())",
            "main",
        ),
        (
            "CREATE VIRTUAL TABLE [main].[t] USING vectorlite(embedding float32[2], diskann())",
            "main",
        ),
        (
            "CREATE VIRTUAL TABLE temp.[t] USING vectorlite(embedding float32[2], diskann())",
            "temp",
        ),
        (
            "CREATE VIRTUAL TABLE [temp].\"t\" USING vectorlite(embedding float32[2], diskann())",
            "temp",
        ),
    ] {
        let target = expect_ok(sql, schema, "t");
        assert_eq!(target.table, "t");
    }
    let bt = "\u{60}";
    let sql = format!(
        "CREATE VIRTUAL TABLE {bt}main{bt}.{bt}t{bt} USING vectorlite(embedding float32[2], diskann())"
    );
    assert_eq!(expect_ok(&sql, "main", "t").schema, "main");
}

#[test]
fn line_comment_inside_argument_is_preserved() {
    let sql =
        "CREATE VIRTUAL TABLE t USING vectorlite(embedding float32[2] -- comment\n, diskann())";
    let error = parse_main(sql).expect_err("comment is part of argument text");
    assert_eq!(error.kind(), TargetErrorKind::InvalidVectorSpace);
}

#[test]
fn string_literals_cannot_be_target_or_module_identifiers() {
    expect_kind(
        "CREATE VIRTUAL TABLE 't' USING vectorlite(embedding float32[2], diskann())",
        TargetErrorKind::InvalidIdentifier,
    );
    expect_kind(
        "CREATE VIRTUAL TABLE t USING 'vectorlite'(embedding float32[2], diskann())",
        TargetErrorKind::InvalidIdentifier,
    );
}

#[test]
fn schema_qualified_module_name_is_rejected() {
    expect_kind(
        "CREATE VIRTUAL TABLE t USING main.vectorlite(embedding float32[2], diskann())",
        TargetErrorKind::ArgumentShape,
    );
}

#[test]
fn trailing_whitespace_is_accepted() {
    let sql = format!("{}\n\t  ", ddl(SPACE, OPTIONS));
    assert_eq!(expect_ok(&sql, "main", "t").table, "t");
}
