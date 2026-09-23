//! Tests for `fhy_core::provenance`: positions, spans, the provenance
//! variants, `Provenance::fuse`, rendering, and the dict wire form.
//!
//! Public API only. Nothing here touches process-global state, so the tests
//! run in parallel freely.

#[path = "common/hashing.rs"]
pub mod hashing_support;
#[path = "common/stack.rs"]
pub mod stack_support;

use fhy_core::provenance::{
    CallSiteProvenance, FileProvenance, FusedProvenance, HasProvenance, NamedProvenance, Position,
    Provenance, ProvenanceError, Span,
};
use hashing_support::hash_of;
use rstest::rstest;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use stack_support::{SMALL_STACK_DEPTH, run_on_small_stack};

// =============================================================================
// Helpers
// =============================================================================

/// Build the position at `line` and `column`, which must both be non-zero.
fn build_position(line: u64, column: u64) -> Position {
    Position::try_new(line, column).expect("line and column are non-zero")
}

/// Build a span with only offsets set.
fn build_offset_span(start_offset: Option<u64>, end_offset: Option<u64>) -> Span {
    Span::try_new(start_offset, end_offset, None, None).expect("offsets are ordered")
}

/// Build a span with only positions set, each given as `(line, column)`.
fn build_position_span(start: Option<(u64, u64)>, end: Option<(u64, u64)>) -> Span {
    Span::try_new(
        None,
        None,
        start.map(|(line, column)| build_position(line, column)),
        end.map(|(line, column)| build_position(line, column)),
    )
    .expect("positions are ordered")
}

/// Build the file provenance for `path` with no span.
fn build_file(path: &str) -> Provenance {
    Provenance::File(FileProvenance::new(path, None))
}

/// Build the file provenance for `path` over `span`.
fn build_file_with_span(path: &str, span: Span) -> Provenance {
    Provenance::File(FileProvenance::new(path, Some(span)))
}

/// Build the named provenance `name` over `child`.
fn build_named(name: &str, child: Provenance) -> Provenance {
    Provenance::Named(NamedProvenance::try_new(name, child).expect("name is non-empty"))
}

/// Build the call-site provenance of `callee` at `caller`.
fn build_call_site(called: Provenance, call_site: Provenance) -> Provenance {
    Provenance::CallSite(CallSiteProvenance::new(called, call_site))
}

/// Build the fusion of `sources` labelled with `metadata`, as given.
fn build_fused(sources: Vec<Provenance>, metadata: Option<&str>) -> Provenance {
    Provenance::Fused(FusedProvenance::new(sources, metadata.map(str::to_owned)))
}

/// Assert decoding `payload` as a `T` fails with the message `expected`.
fn assert_decode_rejected<T: DeserializeOwned + std::fmt::Debug>(payload: Value, expected: &str) {
    let rendered = payload.to_string();
    let error = serde_json::from_value::<T>(payload)
        .expect_err(&format!("{rendered} is malformed and must be rejected"));
    assert_eq!(error.to_string(), expected, "the error for {rendered}");
}

// =============================================================================
// Position
// =============================================================================

/// Test a position stores the line and column it was built with.
#[test]
fn position_try_new_stores_line_and_column() {
    let position = Position::try_new(2, 8).expect("2:8 is valid");

    assert_eq!(position.line().get(), 2);
    assert_eq!(position.column().get(), 8);
}

/// Test a zero line or column is rejected with the variant naming it.
#[rstest]
#[case::zero_line(0, 1, ProvenanceError::ZeroLine)]
#[case::zero_column(1, 0, ProvenanceError::ZeroColumn)]
#[case::both_zero_reports_the_line_first(0, 0, ProvenanceError::ZeroLine)]
fn position_try_new_rejects_zero_components(
    #[case] line: u64,
    #[case] column: u64,
    #[case] expected: ProvenanceError,
) {
    let result = Position::try_new(line, column);

    assert_eq!(result, Err(expected));
}

/// Test the largest representable line and column are accepted.
#[test]
fn position_try_new_accepts_the_largest_u64_values() {
    let position = Position::try_new(u64::MAX, u64::MAX).expect("u64::MAX is valid");

    assert_eq!(position.line().get(), u64::MAX);
    assert_eq!(position.column().get(), u64::MAX);
    assert_eq!(
        position.to_string(),
        "18446744073709551615:18446744073709551615"
    );
}

/// Test positions compare equal exactly when line and column both match.
#[test]
fn position_equality_is_by_value() {
    assert_eq!(build_position(1, 1), build_position(1, 1));
    assert_ne!(build_position(1, 1), build_position(1, 2));
    assert_ne!(build_position(1, 1), build_position(2, 1));
}

/// Test positions order by line first, then by column.
#[test]
fn position_orders_lexicographically_by_line_then_column() {
    assert!(build_position(1, 1) < build_position(1, 2));
    assert!(build_position(1, 99) < build_position(2, 1));
    assert!(build_position(3, 1) > build_position(2, 50));
}

/// Test sorting positions yields line-then-column order.
#[test]
fn position_sorting_follows_line_then_column() {
    let mut positions = vec![
        build_position(2, 1),
        build_position(1, 9),
        build_position(1, 2),
        build_position(3, 3),
    ];

    positions.sort();

    assert_eq!(
        positions,
        vec![
            build_position(1, 2),
            build_position(1, 9),
            build_position(2, 1),
            build_position(3, 3),
        ]
    );
}

/// Test a position renders as `line:column`.
#[test]
fn position_display_renders_line_colon_column() {
    assert_eq!(build_position(2, 8).to_string(), "2:8");
}

// =============================================================================
// Span
// =============================================================================

/// Test the unknown span reports itself as unknown and has no bounds.
#[test]
fn span_unknown_has_no_bounds() {
    let span = Span::unknown();

    assert!(span.is_unknown());
    assert_eq!(span.start_offset(), None);
    assert_eq!(span.end_offset(), None);
    assert_eq!(span.start_position(), None);
    assert_eq!(span.end_position(), None);
}

/// Test a span built with no bounds equals the unknown span.
#[test]
fn span_try_new_with_no_bounds_equals_unknown() {
    let span = Span::try_new(None, None, None, None).expect("no bounds is valid");

    assert_eq!(span, Span::unknown());
    assert!(span.is_unknown());
}

/// Test setting any single bound makes a span known.
#[rstest]
#[case::start_offset(Some(0), None, None, None)]
#[case::end_offset(None, Some(3), None, None)]
#[case::start_position(None, None, Some((1, 1)), None)]
#[case::end_position(None, None, None, Some((1, 2)))]
fn span_with_any_bound_is_not_unknown(
    #[case] start_offset: Option<u64>,
    #[case] end_offset: Option<u64>,
    #[case] start_position: Option<(u64, u64)>,
    #[case] end_position: Option<(u64, u64)>,
) {
    let span = Span::try_new(
        start_offset,
        end_offset,
        start_position.map(|(line, column)| build_position(line, column)),
        end_position.map(|(line, column)| build_position(line, column)),
    )
    .expect("a single bound is valid");

    assert!(!span.is_unknown());
}

/// Test a span stores all four bounds it was built with.
#[test]
fn span_try_new_stores_every_bound() {
    let span = Span::try_new(
        Some(0),
        Some(3),
        Some(build_position(1, 1)),
        Some(build_position(1, 4)),
    )
    .expect("ordered bounds are valid");

    assert_eq!(span.start_offset(), Some(0));
    assert_eq!(span.end_offset(), Some(3));
    assert_eq!(span.start_position(), Some(build_position(1, 1)));
    assert_eq!(span.end_position(), Some(build_position(1, 4)));
}

/// Test an end offset before the start offset is rejected with both offsets.
#[test]
fn span_try_new_rejects_end_offset_before_start_offset() {
    let result = Span::try_new(Some(5), Some(3), None, None);

    assert_eq!(
        result,
        Err(ProvenanceError::EndOffsetBeforeStartOffset {
            start_offset: 5,
            end_offset: 3,
        })
    );
}

/// Test an end position before the start position is rejected with both
/// positions.
#[test]
fn span_try_new_rejects_end_position_before_start_position() {
    let result = Span::try_new(
        None,
        None,
        Some(build_position(2, 1)),
        Some(build_position(1, 1)),
    );

    assert_eq!(
        result,
        Err(ProvenanceError::EndPositionBeforeStartPosition {
            start_position: build_position(2, 1),
            end_position: build_position(1, 1),
        })
    );
}

/// Test the offset check runs before the position check when both fail.
#[test]
fn span_try_new_reports_the_offset_error_first() {
    let result = Span::try_new(
        Some(5),
        Some(3),
        Some(build_position(2, 1)),
        Some(build_position(1, 1)),
    );

    assert_eq!(
        result,
        Err(ProvenanceError::EndOffsetBeforeStartOffset {
            start_offset: 5,
            end_offset: 3,
        })
    );
}

/// Test a zero-width offset range is accepted.
#[test]
fn span_try_new_allows_equal_start_and_end_offset() {
    let span = build_offset_span(Some(4), Some(4));

    assert_eq!(span.start_offset(), Some(4));
    assert_eq!(span.end_offset(), Some(4));
}

/// Test a zero-width position range is accepted.
#[test]
fn span_try_new_allows_equal_start_and_end_position() {
    let span = build_position_span(Some((1, 1)), Some((1, 1)));

    assert_eq!(span.start_position(), Some(build_position(1, 1)));
    assert_eq!(span.end_position(), Some(build_position(1, 1)));
}

/// Test a lone bound is never compared against its missing partner, and the
/// offsets are never checked against the positions.
#[rstest]
#[case::start_offset_only(Some(9), None, None, None)]
#[case::end_offset_zero_only(None, Some(0), None, None)]
#[case::start_position_only(None, None, Some((9, 9)), None)]
#[case::offsets_disagree_with_positions(Some(0), Some(100), Some((50, 1)), Some((50, 2)))]
fn span_try_new_accepts_bounds_it_cannot_order(
    #[case] start_offset: Option<u64>,
    #[case] end_offset: Option<u64>,
    #[case] start_position: Option<(u64, u64)>,
    #[case] end_position: Option<(u64, u64)>,
) {
    let result = Span::try_new(
        start_offset,
        end_offset,
        start_position.map(|(line, column)| build_position(line, column)),
        end_position.map(|(line, column)| build_position(line, column)),
    );

    let span = result.expect("unorderable bounds are accepted");
    assert_eq!(span.start_offset(), start_offset);
    assert_eq!(span.end_offset(), end_offset);
}

/// Test every rendering rule of a span, including that positions win over
/// offsets and that a missing bound renders as `?`.
#[rstest]
#[case::unknown(None, None, None, None, "<unknown>")]
#[case::both_offsets(Some(0), Some(3), None, None, "@0-3")]
#[case::start_offset_only(Some(5), None, None, None, "@5-?")]
#[case::end_offset_only(None, Some(3), None, None, "@?-3")]
#[case::both_positions(None, None, Some((1, 1)), Some((1, 4)), "1:1-1:4")]
#[case::start_position_only(None, None, Some((1, 1)), None, "1:1-?")]
#[case::end_position_only(None, None, None, Some((1, 4)), "?-1:4")]
#[case::positions_win_over_offsets(Some(0), Some(3), Some((1, 1)), Some((1, 4)), "1:1-1:4")]
#[case::one_position_hides_both_offsets(Some(0), Some(3), Some((1, 1)), None, "1:1-?")]
#[case::end_position_hides_start_offset(Some(7), None, None, Some((4, 2)), "?-4:2")]
fn span_display_renders_positions_or_offsets(
    #[case] start_offset: Option<u64>,
    #[case] end_offset: Option<u64>,
    #[case] start_position: Option<(u64, u64)>,
    #[case] end_position: Option<(u64, u64)>,
    #[case] expected: &str,
) {
    let span = Span::try_new(
        start_offset,
        end_offset,
        start_position.map(|(line, column)| build_position(line, column)),
        end_position.map(|(line, column)| build_position(line, column)),
    )
    .expect("ordered bounds are valid");

    assert_eq!(span.to_string(), expected);
}

// =============================================================================
// Provenance variants
// =============================================================================

/// Test a file provenance built without a span stores the path and no span.
#[test]
fn file_provenance_new_without_span_stores_the_path() {
    let provenance = FileProvenance::new("a.fhy", None);

    assert_eq!(provenance.file_path(), "a.fhy");
    assert_eq!(provenance.span(), None);
}

/// Test a file provenance stores the span it was given.
#[test]
fn file_provenance_new_stores_the_span() {
    let span = build_offset_span(Some(0), Some(3));

    let provenance = FileProvenance::new("a.fhy", Some(span));

    assert_eq!(provenance.span(), Some(&span));
}

/// Test file paths are normalized as Python's `PurePosixPath` normalizes
/// them, on every platform: `/` is the only separator, repeated separators
/// and `.` components go, a trailing separator goes, the empty path becomes
/// `.`, and `..`, `~`, a leading `//`, backslashes and drive letters stay.
/// Each expected path is what `str(PurePosixPath(path))` returns.
#[rstest]
#[case::plain("a.fhy", "a.fhy")]
#[case::leading_current_directory("./a", "a")]
#[case::trailing_separator("a/", "a")]
#[case::repeated_separator("a//b", "a/b")]
#[case::inner_current_directory("a/./b", "a/b")]
#[case::three_leading_separators("///a", "/a")]
#[case::two_leading_separators_kept("//a", "//a")]
#[case::empty("", ".")]
#[case::current_directory(".", ".")]
#[case::current_directory_with_separator("./", ".")]
#[case::trailing_current_directory("a/.", "a")]
#[case::root_current_directory("/./a", "/a")]
#[case::parent_directory_kept("a/../b", "a/../b")]
#[case::leading_parent_directory("../a", "../a")]
#[case::current_then_parent("./../a", "../a")]
#[case::trailing_parent_directory("a/..", "a/..")]
#[case::tilde_kept("~/x", "~/x")]
#[case::hidden_file(".a", ".a")]
#[case::root("/", "/")]
#[case::double_root("//", "//")]
#[case::messy_absolute("////a//b/", "/a/b")]
#[case::non_ascii("h\u{e9}llo/w\u{f6}rld.fhy", "h\u{e9}llo/w\u{f6}rld.fhy")]
#[case::spaces("a b/c d.fhy", "a b/c d.fhy")]
#[case::windows_drive_path("C:\\src\\a.fhy", "C:\\src\\a.fhy")]
#[case::backslash_is_not_a_separator("a\\b/c", "a\\b/c")]
#[case::backslash_then_current_directory("a\\.\\b", "a\\.\\b")]
#[case::unc_path("\\\\server\\share", "\\\\server\\share")]
#[case::drive_letter_with_slash("C:/x", "C:/x")]
#[case::bare_drive_letter("C:", "C:")]
#[case::drive_letter_then_current_directory("C:/./x/", "C:/x")]
#[case::two_leading_separators_normalized("//a/./b/", "//a/b")]
#[case::two_leading_separators_then_current_directory("//./a", "//a")]
#[case::current_directory_then_repeated_separator(".//a", "a")]
#[case::three_separators("///", "/")]
#[case::four_leading_separators("////a", "/a")]
fn file_provenance_new_normalizes_the_path(#[case] path: &str, #[case] expected: &str) {
    let provenance = FileProvenance::new(path, None);

    assert_eq!(provenance.file_path(), expected);
}

/// Test two spellings of one normalized path give equal provenances with
/// equal hashes.
#[test]
fn file_provenance_equality_follows_the_normalized_path() {
    let written = FileProvenance::new("./dir//a.fhy/", None);
    let canonical = FileProvenance::new("dir/a.fhy", None);

    assert_eq!(written, canonical);
    assert_eq!(hash_of(&written), hash_of(&canonical));
}

/// Test normalization never resolves `..` or merges a leading `//` into `/`.
#[test]
fn file_provenance_keeps_paths_that_normalize_differently_apart() {
    assert_ne!(
        FileProvenance::new("a/../b", None),
        FileProvenance::new("b", None)
    );
    assert_ne!(
        FileProvenance::new("//a", None),
        FileProvenance::new("/a", None)
    );
}

/// Test a file provenance without a span differs from one with the unknown
/// span.
#[test]
fn file_provenance_without_span_differs_from_unknown_span() {
    assert_ne!(
        FileProvenance::new("a", None),
        FileProvenance::new("a", Some(Span::unknown()))
    );
}

/// Test a named provenance stores its name and child, for a builtin over the
/// unknown provenance and for a library symbol over its file.
#[rstest]
#[case::builtin("fhy.add", Provenance::Unknown)]
#[case::library_symbol("mylib::matmul", build_file("mylib.fhyobj"))]
fn named_provenance_try_new_stores_name_and_child(#[case] name: &str, #[case] child: Provenance) {
    let provenance = NamedProvenance::try_new(name, child.clone()).expect("name is non-empty");

    assert_eq!(provenance.name(), name);
    assert_eq!(provenance.child(), &child);
}

/// Test an empty name is rejected.
#[test]
fn named_provenance_try_new_rejects_an_empty_name() {
    let result = NamedProvenance::try_new("", Provenance::Unknown);

    assert_eq!(result, Err(ProvenanceError::EmptyName));
}

/// Test a name made only of whitespace is not empty and is accepted.
#[test]
fn named_provenance_try_new_accepts_a_whitespace_name() {
    let provenance = NamedProvenance::try_new(" ", Provenance::Unknown).expect("not empty");

    assert_eq!(provenance.name(), " ");
}

/// Test a call-site provenance stores both arms.
#[test]
fn call_site_provenance_new_stores_callee_and_caller() {
    let called = build_file("callee.fhy");
    let call_site = build_file("caller.fhy");

    let provenance = CallSiteProvenance::new(called.clone(), call_site.clone());

    assert_eq!(provenance.callee(), &called);
    assert_eq!(provenance.caller(), &call_site);
}

/// Test a fused provenance keeps its sources in order and no label.
#[test]
fn fused_provenance_new_keeps_sources_in_order() {
    let a = build_file("a.fhy");
    let b = build_file("b.fhy");

    let provenance = FusedProvenance::new(vec![a.clone(), b.clone()], None);

    assert_eq!(provenance.sources(), &[a, b]);
    assert_eq!(provenance.metadata(), None);
}

/// Test a fused provenance stores its label.
#[test]
fn fused_provenance_new_stores_the_metadata() {
    let provenance =
        FusedProvenance::new(vec![build_file("a.fhy")], Some("loop-fusion".to_owned()));

    assert_eq!(provenance.metadata(), Some("loop-fusion"));
}

/// Test direct construction keeps a non-canonical fusion as given.
#[test]
fn fused_provenance_new_keeps_non_canonical_sources() {
    let sources = vec![Provenance::Unknown, build_fused(vec![], None)];

    let provenance = FusedProvenance::new(sources.clone(), None);

    assert_eq!(provenance.sources(), sources.as_slice());
}

/// Test an empty fusion is a fusion, not the unknown provenance.
#[test]
fn fused_provenance_with_no_sources_differs_from_unknown() {
    assert_ne!(build_fused(vec![], None), Provenance::Unknown);
}

/// Test a single-source fusion differs from its source.
#[test]
fn fused_provenance_with_one_source_differs_from_the_source() {
    let a = build_file("a.fhy");

    assert_ne!(build_fused(vec![a.clone()], None), a);
}

/// Test provenances of different variants are never equal.
#[test]
fn provenances_of_different_variants_are_unequal() {
    let file = build_file("a.fhy");
    let named = build_named("a.fhy", Provenance::Unknown);
    let call_site = build_call_site(Provenance::Unknown, Provenance::Unknown);

    assert_ne!(file, named);
    assert_ne!(named, call_site);
    assert_ne!(call_site, Provenance::Unknown);
}

/// Test equal provenances of every variant hash equally.
#[rstest]
#[case::file(|| build_file_with_span("a.fhy", build_offset_span(Some(0), Some(3))))]
#[case::named(|| build_named("fhy.add", Provenance::Unknown))]
#[case::call_site(|| build_call_site(build_file("a.fhy"), build_file("b.fhy")))]
#[case::fused(|| build_fused(vec![build_file("a.fhy")], Some("cse")))]
fn equal_provenances_hash_equally(#[case] build: fn() -> Provenance) {
    let first = build();
    let second = build();

    assert_eq!(first, second);
    assert_eq!(hash_of(&first), hash_of(&second));
}

// =============================================================================
// Provenance::fuse
// =============================================================================

/// Test fusing nothing gives the unknown provenance.
#[test]
fn fuse_with_no_inputs_returns_unknown() {
    let fused = Provenance::fuse([], None);

    assert_eq!(fused, Provenance::Unknown);
}

/// Test fusing only unknown provenances gives the unknown provenance.
#[test]
fn fuse_with_only_unknown_inputs_returns_unknown() {
    let fused = Provenance::fuse([Provenance::Unknown, Provenance::Unknown], None);

    assert_eq!(fused, Provenance::Unknown);
}

/// Test metadata is discarded when nothing survives the flattening.
#[rstest]
#[case::no_inputs(vec![])]
#[case::only_unknown(vec![Provenance::Unknown])]
#[case::empty_fusion(vec![build_fused(vec![Provenance::Unknown], None)])]
fn fuse_with_metadata_and_no_survivors_returns_unknown(#[case] inputs: Vec<Provenance>) {
    let fused = Provenance::fuse(inputs, Some("m"));

    assert_eq!(fused, Provenance::Unknown);
}

/// Test fusing one provenance without metadata returns it unchanged.
#[test]
fn fuse_single_input_without_metadata_returns_input_unchanged() {
    let a = build_file("a.fhy");

    let fused = Provenance::fuse([a.clone()], None);

    assert_eq!(fused, a);
}

/// Test fusing one provenance with metadata wraps it.
#[test]
fn fuse_single_input_with_metadata_wraps_in_fused() {
    let a = build_file("a.fhy");

    let fused = Provenance::fuse([a.clone()], Some("cse"));

    assert_eq!(fused, build_fused(vec![a], Some("cse")));
}

/// Test unknown inputs are dropped from a mixed input.
#[test]
fn fuse_drops_unknown_inputs_from_mixed_input() {
    let a = build_file("a.fhy");
    let b = build_file("b.fhy");

    let fused = Provenance::fuse([a.clone(), Provenance::Unknown, b.clone()], None);

    assert_eq!(fused, build_fused(vec![a, b], None));
}

/// Test an unlabelled nested fusion is spliced into the result.
#[test]
fn fuse_flattens_nested_fused_when_inner_metadata_is_none() {
    let a = build_file("a.fhy");
    let b = build_file("b.fhy");
    let c = build_file("c.fhy");
    let inner = build_fused(vec![a.clone(), b.clone()], None);

    let fused = Provenance::fuse([inner, c.clone()], None);

    assert_eq!(fused, build_fused(vec![a, b, c], None));
}

/// Test a labelled nested fusion is kept whole.
#[test]
fn fuse_preserves_nested_fused_when_inner_metadata_is_set() {
    let a = build_file("a.fhy");
    let b = build_file("b.fhy");
    let c = build_file("c.fhy");
    let inner = build_fused(vec![a, b], Some("cse"));

    let fused = Provenance::fuse([inner.clone(), c.clone()], None);

    assert_eq!(fused, build_fused(vec![inner, c], None));
}

/// Test a fusion labelled with the empty string is labelled, so it stays
/// whole.
#[test]
fn fuse_treats_an_empty_label_as_a_label() {
    let inner = build_fused(vec![build_file("a.fhy"), build_file("b.fhy")], Some(""));

    let fused = Provenance::fuse([inner.clone()], None);

    assert_eq!(fused, inner);
}

/// Test the result keeps the input order.
#[test]
fn fuse_preserves_input_order() {
    let a = build_file("a.fhy");
    let b = build_file("b.fhy");
    let c = build_file("c.fhy");

    let fused = Provenance::fuse([a.clone(), b.clone(), c.clone()], None);

    assert_eq!(fused, build_fused(vec![a, b, c], None));
}

/// Test fusing the same inputs in another order gives another result.
#[test]
fn fuse_is_not_commutative() {
    let a = build_file("a.fhy");
    let b = build_file("b.fhy");

    let forward = Provenance::fuse([a.clone(), b.clone()], None);
    let backward = Provenance::fuse([b, a], None);

    assert_ne!(forward, backward);
}

/// Test equal sources are not deduplicated.
#[test]
fn fuse_preserves_duplicate_sources() {
    let a = build_file("a.fhy");

    let fused = Provenance::fuse([a.clone(), a.clone()], None);

    assert_eq!(fused, build_fused(vec![a.clone(), a], None));
}

/// Test the metadata labels the result.
#[test]
fn fuse_attaches_metadata_to_result() {
    let a = build_file("a.fhy");
    let b = build_file("b.fhy");

    let fused = Provenance::fuse([a.clone(), b.clone()], Some("loop-fusion"));

    assert_eq!(fused, build_fused(vec![a, b], Some("loop-fusion")));
}

/// Test unknown provenances revealed by splicing are dropped too.
#[test]
fn fuse_drops_unknown_inside_directly_constructed_nested_fused() {
    let a = build_file("a.fhy");
    let b = build_file("b.fhy");
    let non_canonical_inner = build_fused(vec![Provenance::Unknown, a.clone()], None);

    let fused = Provenance::fuse([non_canonical_inner, b.clone()], None);

    assert_eq!(fused, build_fused(vec![a, b], None));
}

/// Test unlabelled fusions are spliced at any depth.
#[test]
fn fuse_flattens_nested_metadata_less_fused_transitively() {
    let a = build_file("a.fhy");
    let b = build_file("b.fhy");
    let c = build_file("c.fhy");
    let deeply_nested = build_fused(
        vec![a.clone(), build_fused(vec![b.clone(), c.clone()], None)],
        None,
    );

    let fused = Provenance::fuse([deeply_nested], None);

    assert_eq!(fused, build_fused(vec![a, b, c], None));
}

/// Test splicing stops at a labelled fusion, even deep inside.
#[test]
fn fuse_does_not_flatten_through_metadata_bearing_fused() {
    let a = build_file("a.fhy");
    let labeled_inner = build_fused(vec![build_file("b.fhy"), build_file("c.fhy")], Some("cse"));
    let outer_metadata_less = build_fused(vec![a.clone(), labeled_inner.clone()], None);

    let fused = Provenance::fuse([outer_metadata_less], None);

    assert_eq!(fused, build_fused(vec![a, labeled_inner], None));
}

/// Test dropping, splicing and ordering compose.
#[test]
fn fuse_combines_all_reduction_rules() {
    let a = build_file("a.fhy");
    let b = build_file("b.fhy");
    let c = build_file("c.fhy");
    let inner_no_meta = build_fused(vec![a.clone(), b.clone()], None);
    let inner_with_meta = build_fused(vec![b.clone(), c.clone()], Some("x"));

    let fused = Provenance::fuse(
        [
            Provenance::Unknown,
            inner_no_meta,
            Provenance::Unknown,
            inner_with_meta.clone(),
            c.clone(),
        ],
        None,
    );

    assert_eq!(fused, build_fused(vec![a, b, inner_with_meta, c], None));
}

/// Test a single surviving source reached through splicing is returned
/// bare.
#[test]
fn fuse_unwraps_a_single_source_found_by_splicing() {
    let a = build_file("a.fhy");
    let nested = build_fused(vec![build_fused(vec![a.clone()], None)], None);

    let fused = Provenance::fuse([nested], None);

    assert_eq!(fused, a);
}

/// Test fusions nested inside named and call-site provenances are not
/// inspected.
#[test]
fn fuse_does_not_look_inside_named_or_call_site_children() {
    let named = build_named(
        "n",
        build_fused(vec![Provenance::Unknown, build_file("a.fhy")], None),
    );
    let call_site = build_call_site(
        build_fused(vec![build_file("a.fhy")], None),
        Provenance::Unknown,
    );

    assert_eq!(Provenance::fuse([named.clone()], None), named);
    assert_eq!(Provenance::fuse([call_site.clone()], None), call_site);
}

/// Test unlabelled fusions nested [`SMALL_STACK_DEPTH`] levels deep, each
/// holding a file then the next fusion, are spliced into one flat fusion of
/// every file in order, on a small thread stack.
#[test]
fn fuse_flattens_deeply_nested_unlabelled_fusions_on_a_small_stack() {
    run_on_small_stack(|| {
        let files: Vec<Provenance> = (0..=SMALL_STACK_DEPTH)
            .map(|index| build_file(&format!("{index}.fhy")))
            .collect();
        let mut nested = files[SMALL_STACK_DEPTH].clone();
        for file in files[..SMALL_STACK_DEPTH].iter().rev() {
            nested = build_fused(vec![file.clone(), nested], None);
        }

        let fused = Provenance::fuse([nested], None);

        let Provenance::Fused(flat) = &fused else {
            panic!("the splice leaves a fusion");
        };
        assert_eq!(flat.metadata(), None);
        assert!(flat.sources() == files, "the spliced sources differ");
    });
}

/// Test fusion is associative without metadata.
#[test]
fn fuse_is_associative_without_metadata() {
    let a = build_file("a.fhy");
    let b = build_file("b.fhy");
    let c = build_file("c.fhy");

    let left = Provenance::fuse(
        [Provenance::fuse([a.clone(), b.clone()], None), c.clone()],
        None,
    );
    let right = Provenance::fuse(
        [a.clone(), Provenance::fuse([b.clone(), c.clone()], None)],
        None,
    );
    let flat = Provenance::fuse([a, b, c], None);

    assert_eq!(left, flat);
    assert_eq!(right, flat);
}

/// Test a labelled inner fusion breaks associativity, since it stays whole.
#[test]
fn fuse_with_inner_metadata_is_not_associative() {
    let a = build_file("a.fhy");
    let b = build_file("b.fhy");
    let c = build_file("c.fhy");

    let grouped = Provenance::fuse(
        [
            Provenance::fuse([a.clone(), b.clone()], Some("m")),
            c.clone(),
        ],
        None,
    );
    let flat = Provenance::fuse([a, b, c], Some("m"));

    assert_ne!(grouped, flat);
}

/// Test a deeply nested chain of unlabelled fusions flattens without
/// exhausting the stack.
#[test]
fn fuse_flattens_a_deeply_nested_chain() {
    const DEPTH: usize = 3000;
    let a = build_file("a.fhy");
    let b = build_file("b.fhy");
    let mut nested = build_fused(vec![a.clone()], None);
    for _ in 0..DEPTH {
        nested = build_fused(vec![nested], None);
    }

    let fused = Provenance::fuse([nested, b.clone()], None);

    assert_eq!(fused, build_fused(vec![a, b], None));
}

/// Test a loop-fusion pass records both source regions under its label.
#[test]
fn fuse_loop_fusion_pass_combines_two_provenances() {
    let op_a = build_file_with_span("a.fhy", build_offset_span(Some(0), Some(3)));
    let op_b = build_file_with_span("b.fhy", build_offset_span(Some(10), Some(13)));

    let fused = Provenance::fuse([op_a.clone(), op_b.clone()], Some("loop-fusion"));

    let Provenance::Fused(fused) = &fused else {
        panic!("expected a fused provenance, got {fused:?}");
    };
    assert_eq!(fused.sources(), &[op_a, op_b]);
    assert_eq!(fused.metadata(), Some("loop-fusion"));
}

// =============================================================================
// Display
// =============================================================================

/// Test every rendering rule of the provenance variants.
#[rstest]
#[case::unknown(Provenance::Unknown, "<unknown>")]
#[case::file_without_span(build_file("a.fhy"), "a.fhy")]
#[case::file_with_unknown_span(build_file_with_span("a.fhy", Span::unknown()), "a.fhy")]
#[case::file_with_offset_span(
    build_file_with_span("a.fhy", build_offset_span(Some(0), Some(3))),
    "a.fhy:@0-3"
)]
#[case::file_with_position_span(
    build_file_with_span("a.fhy", build_position_span(Some((1, 1)), Some((1, 4)))),
    "a.fhy:1:1-1:4"
)]
#[case::file_renders_the_normalized_path(
    build_file_with_span("./x//y.fhy", build_offset_span(Some(0), Some(3))),
    "x/y.fhy:@0-3"
)]
#[case::named_with_unknown_child(build_named("fhy.add", Provenance::Unknown), "fhy.add")]
#[case::named_with_known_child(
    build_named("mylib::matmul", build_file("mylib.fhyobj")),
    "mylib::matmul (mylib.fhyobj)"
)]
#[case::named_with_empty_fusion_child(
    build_named("empty-fusion", build_fused(vec![], None)),
    "empty-fusion (fused[])"
)]
#[case::call_site(
    build_call_site(build_file("callee.fhy"), build_file("caller.fhy")),
    "callee.fhy at caller.fhy"
)]
#[case::call_site_chain(
    build_call_site(
        build_call_site(build_file("a.fhy"), build_file("b.fhy")),
        build_file("c.fhy")
    ),
    "a.fhy at b.fhy at c.fhy"
)]
#[case::fused_without_metadata(
    build_fused(vec![build_file("a.fhy"), build_file("b.fhy")], None),
    "fused[a.fhy, b.fhy]"
)]
#[case::fused_with_metadata(build_fused(vec![build_file("a.fhy")], Some("loop-fusion")), "loop-fusion[a.fhy]")]
#[case::fused_recurses_into_sources(
    build_fused(
        vec![
            build_fused(vec![build_file("a.fhy"), build_file("b.fhy")], Some("cse")),
            build_file("c.fhy"),
        ],
        None,
    ),
    "fused[cse[a.fhy, b.fhy], c.fhy]"
)]
#[case::fused_with_no_sources(build_fused(vec![], None), "fused[]")]
#[case::fused_with_empty_label(
    build_fused(vec![build_file("a.fhy"), build_file("b.fhy")], Some("")),
    "[a.fhy, b.fhy]"
)]
fn provenance_display_renders_each_variant(#[case] provenance: Provenance, #[case] expected: &str) {
    assert_eq!(provenance.to_string(), expected);
}

// =============================================================================
// HasProvenance
// =============================================================================

/// A stand-in IR node carrying a provenance.
struct StoryNode {
    provenance: Provenance,
}

impl HasProvenance for StoryNode {
    fn provenance(&self) -> &Provenance {
        &self.provenance
    }
}

/// Test a pass merging two nodes reads their provenances through
/// `HasProvenance` and gives the merged node their fusion.
#[test]
fn merging_two_nodes_fuses_their_provenances() {
    let left = StoryNode {
        provenance: build_file_with_span("kernel.fhy", build_offset_span(Some(0), Some(4))),
    };
    let right = StoryNode {
        provenance: build_named("fhy.add", Provenance::Unknown),
    };

    let merged = StoryNode {
        provenance: Provenance::fuse(
            [left.provenance().clone(), right.provenance().clone()],
            Some("cse"),
        ),
    };

    assert_eq!(
        merged.provenance().to_string(),
        "cse[kernel.fhy:@0-4, fhy.add]"
    );
}

/// Test an inliner's chain of call sites exposes every arm.
#[test]
fn inliner_call_site_chain_is_walkable() {
    let inner = build_file("inner.fhy");
    let middle = build_file("middle.fhy");
    let outer = build_file("outer.fhy");

    let chain = CallSiteProvenance::new(
        build_call_site(inner.clone(), middle.clone()),
        outer.clone(),
    );

    assert_eq!(chain.caller(), &outer);
    let Provenance::CallSite(callee) = chain.callee() else {
        panic!("expected a call-site callee, got {:?}", chain.callee());
    };
    assert_eq!(callee.callee(), &inner);
    assert_eq!(callee.caller(), &middle);
}

// =============================================================================
// Wire form
// =============================================================================

/// Test a position encodes as its two fields.
#[test]
fn position_encodes_as_line_and_column() {
    let encoded = serde_json::to_value(build_position(2, 8)).expect("positions encode");

    assert_eq!(encoded, json!({"line": 2, "column": 8}));
}

/// Test a span encodes all four keys, with `null` for absent bounds.
#[rstest]
#[case::full(
    Span::try_new(Some(0), Some(3), Some(build_position(1, 1)), Some(build_position(1, 4))).unwrap(),
    json!({
        "start_offset": 0,
        "end_offset": 3,
        "start_position": {"line": 1, "column": 1},
        "end_position": {"line": 1, "column": 4},
    })
)]
#[case::unknown(
    Span::unknown(),
    json!({"start_offset": null, "end_offset": null, "start_position": null, "end_position": null})
)]
fn span_encodes_every_key(#[case] span: Span, #[case] expected: Value) {
    let encoded = serde_json::to_value(span).expect("spans encode");

    assert_eq!(encoded, expected);
}

/// Test each provenance variant encodes in the wrapped form, with an empty
/// data dict for the unknown provenance and the normalized path for a file.
#[rstest]
#[case::unknown(Provenance::Unknown, json!({"__type__": "provenance.unknown", "__data__": {}}))]
#[case::file(
    build_file_with_span("./a//b.fhy", build_offset_span(Some(0), Some(3))),
    json!({
        "__type__": "provenance.file",
        "__data__": {
            "file_path": "a/b.fhy",
            "span": {"start_offset": 0, "end_offset": 3, "start_position": null, "end_position": null},
        },
    })
)]
#[case::file_without_span(
    build_file("c.fhy"),
    json!({"__type__": "provenance.file", "__data__": {"file_path": "c.fhy", "span": null}})
)]
#[case::named(
    build_named("lib", Provenance::Unknown),
    json!({
        "__type__": "provenance.named",
        "__data__": {"name": "lib", "child": {"__type__": "provenance.unknown", "__data__": {}}},
    })
)]
#[case::call_site(
    build_call_site(build_file("a.fhy"), Provenance::Unknown),
    json!({
        "__type__": "provenance.call_site",
        "__data__": {
            "callee": {"__type__": "provenance.file", "__data__": {"file_path": "a.fhy", "span": null}},
            "caller": {"__type__": "provenance.unknown", "__data__": {}},
        },
    })
)]
#[case::fused_without_metadata(
    build_fused(vec![Provenance::Unknown], None),
    json!({
        "__type__": "provenance.fused",
        "__data__": {
            "sources": [{"__type__": "provenance.unknown", "__data__": {}}],
            "metadata": null,
        },
    })
)]
#[case::fused_with_metadata(
    build_fused(vec![], Some("fuse")),
    json!({"__type__": "provenance.fused", "__data__": {"sources": [], "metadata": "fuse"}})
)]
fn provenance_encodes_in_the_wrapped_form(#[case] provenance: Provenance, #[case] expected: Value) {
    let encoded = serde_json::to_value(&provenance).expect("provenances encode");

    assert_eq!(encoded, expected);
}

/// Test every variant round-trips through JSON text.
#[rstest]
#[case::unknown(Provenance::Unknown)]
#[case::file(build_file("c.fhy"))]
#[case::file_with_full_span(build_file_with_span(
    "a.fhy",
    Span::try_new(Some(0), Some(3), Some(build_position(1, 1)), Some(build_position(1, 4))).unwrap(),
))]
#[case::builtin(build_named("fhy.add", Provenance::Unknown))]
#[case::library_symbol(build_named("mylib::matmul", build_file("mylib.fhyobj")))]
#[case::call_site(build_call_site(
    build_file_with_span("a.fhy", build_offset_span(Some(0), Some(10))),
    build_file("b.fhy"),
))]
#[case::inlined_call_site(build_call_site(
    build_named("inlined", build_file("a.fhy")),
    build_file("b.fhy"),
))]
#[case::fused(build_fused(vec![build_file("a.fhy"), build_file("b.fhy")], None))]
#[case::fused_with_metadata(build_fused(vec![build_file("a.fhy")], Some("loop-fusion")))]
fn provenance_round_trips_through_json(#[case] provenance: Provenance) {
    let json = serde_json::to_string(&provenance).expect("provenances encode");

    let restored: Provenance = serde_json::from_str(&json).expect("encoded provenances decode");

    assert_eq!(restored, provenance);
}

/// Test position and span values round-trip through JSON text.
#[test]
fn position_and_span_round_trip_through_json() {
    let position = build_position(7, 3);
    let span = Span::try_new(Some(1), None, Some(position), None).expect("valid");

    let restored_position: Position =
        serde_json::from_str(&serde_json::to_string(&position).unwrap()).unwrap();
    let restored_span: Span = serde_json::from_str(&serde_json::to_string(&span).unwrap()).unwrap();

    assert_eq!(restored_position, position);
    assert_eq!(restored_span, span);
}

/// Test decoding two unknown payloads gives equal provenances.
#[test]
fn decoded_unknown_provenances_compare_equal() {
    let payload = json!({"__type__": "provenance.unknown", "__data__": {}});

    let first: Provenance = serde_json::from_value(payload.clone()).expect("valid payload");
    let second: Provenance = serde_json::from_value(payload).expect("valid payload");

    assert_eq!(first, Provenance::Unknown);
    assert_eq!(first, second);
}

/// Test decoding a file payload normalizes its path.
#[test]
fn provenance_decode_normalizes_the_file_path() {
    let payload = json!({"__type__": "provenance.file", "__data__": {"file_path": "./x//y.fhy/", "span": null}});

    let decoded: Provenance = serde_json::from_value(payload).expect("valid payload");

    assert_eq!(decoded, build_file("x/y.fhy"));
}

/// Test malformed position payloads are rejected with a message naming the
/// problem.
#[rstest]
#[case::missing_column(
    json!({"line": 1}),
    "missing field `column`"
)]
#[case::bool_line(
    json!({"line": true, "column": 1}),
    "invalid type: boolean `true`, expected u64"
)]
#[case::extra_key(
    json!({"line": 1, "column": 2, "z": 3}),
    "unknown field `z`, expected `line` or `column`"
)]
#[case::zero_line(
    json!({"line": 0, "column": 1}),
    "a position's line must be at least 1"
)]
#[case::zero_column(
    json!({"line": 1, "column": 0}),
    "a position's column must be at least 1"
)]
#[case::float_line(
    json!({"line": 1.0, "column": 1}),
    "invalid number"
)]
#[case::string_line(
    json!({"line": "1", "column": 1}),
    r#"invalid type: string "1", expected u64"#
)]
#[case::negative_line(
    json!({"line": -1, "column": 1}),
    "invalid number"
)]
#[case::null_line(
    json!({"line": null, "column": 1}),
    "invalid type: null, expected u64"
)]
#[case::not_a_map(
    json!([1, 1]),
    "invalid type: sequence, expected struct PositionPayload"
)]
fn position_decode_rejects_malformed_payloads(#[case] payload: Value, #[case] expected: &str) {
    assert_decode_rejected::<Position>(payload, expected);
}

/// Test malformed span payloads, including one missing each key, are
/// rejected with a message naming the problem.
#[rstest]
#[case::missing_start_offset(
    json!({"end_offset": null, "start_position": null, "end_position": null}),
    "missing field `start_offset`"
)]
#[case::missing_end_offset(
    json!({"start_offset": null, "start_position": null, "end_position": null}),
    "missing field `end_offset`"
)]
#[case::missing_start_position(
    json!({"start_offset": null, "end_offset": null, "end_position": null}),
    "missing field `start_position`"
)]
#[case::missing_end_position(
    json!({"start_offset": null, "end_offset": null, "start_position": null}),
    "missing field `end_position`"
)]
#[case::extra_key(
    json!({"start_offset": 0, "end_offset": 3, "start_position": null, "end_position": null, "extra": 1}),
    "unknown field `extra`, expected one of `start_offset`, `end_offset`, `start_position`, `end_position`"
)]
#[case::end_offset_before_start(
    json!({"start_offset": 5, "end_offset": 3, "start_position": null, "end_position": null}),
    "a span's end offset 3 precedes its start offset 5"
)]
#[case::end_position_before_start(
    json!({
    "start_offset": null,
    "end_offset": null,
    "start_position": {"line": 2, "column": 1},
    "end_position": {"line": 1, "column": 1},
}),
    "a span's end position 1:1 precedes its start position 2:1"
)]
#[case::negative_offset(
    json!({"start_offset": -1, "end_offset": null, "start_position": null, "end_position": null}),
    "invalid number"
)]
#[case::bool_offset(
    json!({"start_offset": true, "end_offset": null, "start_position": null, "end_position": null}),
    "invalid type: boolean `true`, expected u64"
)]
#[case::invalid_position(
    json!({
    "start_offset": null,
    "end_offset": null,
    "start_position": {"line": 0, "column": 1},
    "end_position": null,
}),
    "a position's line must be at least 1"
)]
#[case::list_position(
    json!({"start_offset": null, "end_offset": null, "start_position": [1, 1], "end_position": null}),
    "invalid type: sequence, expected struct PositionPayload"
)]
fn span_decode_rejects_malformed_payloads(#[case] payload: Value, #[case] expected: &str) {
    assert_decode_rejected::<Span>(payload, expected);
}

/// Test malformed provenance payloads are rejected with a message naming the
/// problem, including an unknown type id and an unknown provenance without
/// its empty data dict.
#[rstest]
#[case::unknown_type_id(
    json!({"__type__": "provenance.does_not_exist", "__data__": {}}),
    "unknown variant `provenance.does_not_exist`, expected one of `provenance.unknown`, `provenance.file`, `provenance.named`, `provenance.call_site`, `provenance.fused`"
)]
#[case::non_provenance_type_id(
    json!({"__type__": "position", "__data__": {"line": 1, "column": 1}}),
    "unknown variant `position`, expected one of `provenance.unknown`, `provenance.file`, `provenance.named`, `provenance.call_site`, `provenance.fused`"
)]
#[case::unknown_without_data(
    json!({"__type__": "provenance.unknown"}),
    "missing field `__data__`"
)]
#[case::unknown_with_null_data(
    json!({"__type__": "provenance.unknown", "__data__": null}),
    "invalid type: null, expected struct UnknownFields"
)]
#[case::unknown_with_a_field(
    json!({"__type__": "provenance.unknown", "__data__": {"x": 1}}),
    "unknown field `x`, there are no fields"
)]
#[case::extra_envelope_key(
    json!({"__type__": "provenance.unknown", "__data__": {}, "z": 1}),
    r#"invalid value: string "z", expected "__type__" or "__data__""#
)]
#[case::missing_type(
    json!({"__data__": {}}),
    "missing field `__type__`"
)]
#[case::file_missing_span(
    json!({"__type__": "provenance.file", "__data__": {"file_path": "a"}}),
    "missing field `span`"
)]
#[case::file_path_not_a_string(
    json!({"__type__": "provenance.file", "__data__": {"file_path": 3, "span": null}}),
    "invalid type: integer `3`, expected a string"
)]
#[case::empty_name(
    json!({
    "__type__": "provenance.named",
    "__data__": {"name": "", "child": {"__type__": "provenance.unknown", "__data__": {}}},
}),
    "a named provenance's name must be non-empty"
)]
#[case::unwrapped_child(
    json!({"__type__": "provenance.named", "__data__": {"name": "n", "child": {"line": 1, "column": 1}}}),
    r#"invalid value: string "column", expected "__type__" or "__data__""#
)]
#[case::null_caller(
    json!({
    "__type__": "provenance.call_site",
    "__data__": {"callee": {"__type__": "provenance.unknown", "__data__": {}}, "caller": null},
}),
    "invalid type: null, expected adjacently tagged enum ProvenancePayload"
)]
#[case::integer_metadata(
    json!({"__type__": "provenance.fused", "__data__": {"sources": [], "metadata": 5}}),
    "invalid type: integer `5`, expected a string"
)]
#[case::null_sources(
    json!({"__type__": "provenance.fused", "__data__": {"sources": null, "metadata": null}}),
    "invalid type: null, expected a sequence"
)]
#[case::missing_metadata(
    json!({"__type__": "provenance.fused", "__data__": {"sources": []}}),
    "missing field `metadata`"
)]
#[case::invalid_nested_source(
    json!({
    "__type__": "provenance.fused",
    "__data__": {
        "sources": [{"__type__": "provenance.named", "__data__": {"name": "", "child": {"__type__": "provenance.unknown", "__data__": {}}}}],
        "metadata": null,
    },
}),
    "a named provenance's name must be non-empty"
)]
#[case::not_a_map(
    json!([{"__type__": "provenance.unknown", "__data__": {}}]),
    "invalid type: sequence, expected adjacently tagged enum ProvenancePayload"
)]
fn provenance_decode_rejects_malformed_payloads(#[case] payload: Value, #[case] expected: &str) {
    assert_decode_rejected::<Provenance>(payload, expected);
}

/// Test a line, column or offset beyond `u64` is rejected on decode, where
/// an unbounded integer would be accepted.
#[rstest]
#[case::huge_line(json!({"line": 1_180_591_620_717_411_303_424_u128, "column": 1}))]
#[case::line_just_past_u64(json!({"line": 18_446_744_073_709_551_616_u128, "column": 1}))]
fn position_decode_rejects_values_beyond_u64(#[case] payload: Value) {
    assert_decode_rejected::<Position>(payload, "invalid number");
}

/// Test a span offset beyond `u64` is rejected on decode.
#[test]
fn span_decode_rejects_an_offset_beyond_u64() {
    let payload = json!({
        "start_offset": 18_446_744_073_709_551_616_u128,
        "end_offset": null,
        "start_position": null,
        "end_position": null,
    });

    assert_decode_rejected::<Span>(payload, "invalid number");
}

/// Return `depth` named provenances nested over the unknown provenance.
fn build_nested_named(depth: usize) -> Provenance {
    let mut provenance = Provenance::Unknown;
    for _ in 0..depth {
        provenance = build_named("n", provenance);
    }
    provenance
}

/// Test JSON text decodes a provenance nested as deep as `serde_json`'s
/// nesting limit allows, two JSON levels per provenance level, and refuses
/// one level more with an error rather than a crash; a JSON value parsed
/// from that text meets the same limit.
#[test]
fn provenance_json_text_decodes_up_to_the_serde_json_nesting_limit() {
    let deepest = build_nested_named(62);
    let too_deep = build_nested_named(63);
    let deepest_text = serde_json::to_string(&deepest).expect("the provenance serializes");
    let too_deep_text = serde_json::to_string(&too_deep).expect("the provenance serializes");

    let decoded: Provenance =
        serde_json::from_str(&deepest_text).expect("62 levels over a leaf decode");
    let refusal = serde_json::from_str::<Provenance>(&too_deep_text);
    let value_refusal = serde_json::from_str::<Value>(&too_deep_text);

    assert_eq!(decoded, deepest);
    let error = refusal.expect_err("63 levels over a leaf exceed the nesting limit");
    assert!(
        error.to_string().starts_with("recursion limit exceeded"),
        "{error}"
    );
    let error = value_refusal.expect_err("63 levels over a leaf exceed the nesting limit");
    assert!(
        error.to_string().starts_with("recursion limit exceeded"),
        "{error}"
    );
}

// =============================================================================
// Errors and thread safety
// =============================================================================

/// Test every error variant renders its full message, with the offsets and
/// positions of an unordered span each in its own place.
#[rstest]
#[case::zero_line(ProvenanceError::ZeroLine, "a position's line must be at least 1")]
#[case::zero_column(ProvenanceError::ZeroColumn, "a position's column must be at least 1")]
#[case::end_offset(
    ProvenanceError::EndOffsetBeforeStartOffset { start_offset: 5, end_offset: 3 },
    "a span's end offset 3 precedes its start offset 5"
)]
#[case::end_position(
    ProvenanceError::EndPositionBeforeStartPosition {
        start_position: build_position(2, 1),
        end_position: build_position(1, 1),
    },
    "a span's end position 1:1 precedes its start position 2:1"
)]
#[case::empty_name(
    ProvenanceError::EmptyName,
    "a named provenance's name must be non-empty"
)]
fn provenance_error_display_writes_the_full_message(
    #[case] error: ProvenanceError,
    #[case] expected: &str,
) {
    let message = error.to_string();

    assert_eq!(message, expected);
}

/// Compile-time check that the provenance types can cross threads.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Position>();
    assert_send_sync::<Span>();
    assert_send_sync::<Provenance>();
    assert_send_sync::<ProvenanceError>();
};
