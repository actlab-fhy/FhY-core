//! Tests for `fhy_core::provenance`: positions, spans, the provenance
//! variants, `Provenance::fuse`, rendering, and the wire form.
//!
//! Nothing here touches process-global state, so the tests run in parallel
//! freely.

use crate::support::hashing as hashing_support;
use crate::support::provenance as provenance_support;
use crate::support::stack as stack_support;

use std::ops::Range;

use fhy_core::provenance::{
    CallSiteProvenance, FileProvenance, FusedProvenance, HasProvenance, NamedProvenance,
    NamedProvenanceError, Position, PositionError, Provenance, Span, SpanError,
};
use hashing_support::hash_of;
use provenance_support::{build_file, build_named};
use rstest::rstest;
use serde::de::DeserializeOwned;
use serde_json::error::Category;
use serde_json::{Value, json};
use stack_support::{SMALL_STACK_DEPTH, run_on_small_stack};

// =============================================================================
// Helpers
// =============================================================================

/// Build the position at `line` and `column`, which must both be non-zero.
fn build_position(line: u64, column: u64) -> Position {
    Position::try_new(line, column).expect("line and column are non-zero")
}

/// Build the span with the given bounds, each position given as `(line,
/// column)`, setting each present bound with its single-bound builder.
fn try_build_span(
    start_offset: Option<u64>,
    end_offset: Option<u64>,
    start_position: Option<(u64, u64)>,
    end_position: Option<(u64, u64)>,
) -> Result<Span, SpanError> {
    let mut span = Span::unknown();
    if let Some(offset) = start_offset {
        span = span.with_start_offset(offset)?;
    }
    if let Some(offset) = end_offset {
        span = span.with_end_offset(offset)?;
    }
    if let Some((line, column)) = start_position {
        span = span.with_start_position(build_position(line, column))?;
    }
    if let Some((line, column)) = end_position {
        span = span.with_end_position(build_position(line, column))?;
    }
    Ok(span)
}

/// Build the span with all four bounds set, each position given as `(line,
/// column)`.
fn build_full_span(offsets: Range<u64>, positions: Range<(u64, u64)>) -> Span {
    try_build_span(
        Some(offsets.start),
        Some(offsets.end),
        Some(positions.start),
        Some(positions.end),
    )
    .expect("ordered bounds are valid")
}

/// Build a span with only offsets set.
fn build_offset_span(start_offset: Option<u64>, end_offset: Option<u64>) -> Span {
    try_build_span(start_offset, end_offset, None, None).expect("offsets are ordered")
}

/// Build a span with only positions set, each given as `(line, column)`.
fn build_position_span(start: Option<(u64, u64)>, end: Option<(u64, u64)>) -> Span {
    try_build_span(None, None, start, end).expect("positions are ordered")
}

/// Build the call-site provenance of `called` at `call_site`.
fn build_call_site(called: Provenance, call_site: Provenance) -> Provenance {
    Provenance::CallSite(CallSiteProvenance::new(called, call_site))
}

/// Build the fusion of `sources`, labelled with `label` when one is given,
/// with the sources kept as given.
fn build_fused(sources: Vec<Provenance>, label: Option<&str>) -> Provenance {
    Provenance::Fused(match label {
        Some(label) => FusedProvenance::labelled(sources, label),
        None => FusedProvenance::new(sources),
    })
}

/// Assert decoding the JSON text `payload_text` as a `T` fails with an
/// error of `category` and, when `crate_error` is given, with a message
/// containing that crate error's text.
///
/// The rest of the message is serde's, which this crate does not own.
fn assert_decode_rejected<T: DeserializeOwned + std::fmt::Debug>(
    payload_text: &str,
    category: Category,
    crate_error: Option<&str>,
) {
    let error = serde_json::from_str::<T>(payload_text)
        .expect_err(&format!("{payload_text} is malformed and must be rejected"));
    assert_eq!(
        error.classify(),
        category,
        "the error for {payload_text}: {error}"
    );
    if let Some(crate_error) = crate_error {
        assert!(
            error.to_string().contains(crate_error),
            "the error for {payload_text}: {error}"
        );
    }
}

// =============================================================================
// Position
// =============================================================================

#[test]
fn position_try_new_stores_line_and_column() {
    let position = Position::try_new(2, 8).expect("2:8 is valid");

    assert_eq!(position.line().get(), 2);
    assert_eq!(position.column().get(), 8);
}

/// Test a zero line or column is rejected with the variant naming it.
#[rstest]
#[case::zero_line(0, 1, PositionError::ZeroLine)]
#[case::zero_column(1, 0, PositionError::ZeroColumn)]
#[case::both_zero_reports_the_line_first(0, 0, PositionError::ZeroLine)]
fn position_try_new_rejects_zero_components(
    #[case] line: u64,
    #[case] column: u64,
    #[case] expected: PositionError,
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

#[test]
fn position_equality_is_by_value() {
    assert_eq!(build_position(1, 1), build_position(1, 1));
    assert_ne!(build_position(1, 1), build_position(1, 2));
    assert_ne!(build_position(1, 1), build_position(2, 1));
}

#[test]
fn position_orders_lexicographically_by_line_then_column() {
    assert!(build_position(1, 1) < build_position(1, 2));
    assert!(build_position(1, 99) < build_position(2, 1));
    assert!(build_position(3, 1) > build_position(2, 50));
}

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

#[test]
fn position_display_renders_line_colon_column() {
    assert_eq!(build_position(2, 8).to_string(), "2:8");
}

// =============================================================================
// Span
// =============================================================================

#[test]
fn span_unknown_has_no_bounds() {
    let span = Span::unknown();

    assert!(span.is_unknown());
    assert_eq!(span.start_offset(), None);
    assert_eq!(span.end_offset(), None);
    assert_eq!(span.start_position(), None);
    assert_eq!(span.end_position(), None);
}

#[test]
fn span_unknown_is_a_constant() {
    const UNKNOWN: Span = Span::unknown();

    assert_eq!(UNKNOWN, Span::unknown());
    assert!(UNKNOWN.is_unknown());
}

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
    let span = try_build_span(start_offset, end_offset, start_position, end_position)
        .expect("a single bound is valid");

    assert!(!span.is_unknown());
}

/// Test a span built from an offset range and a position range stores all
/// four bounds.
#[test]
fn span_builders_store_every_bound() {
    let span = Span::from_offsets(0..3)
        .and_then(|span| span.with_positions(build_position(1, 1)..build_position(1, 4)))
        .expect("ordered bounds are valid");

    assert_eq!(span.start_offset(), Some(0));
    assert_eq!(span.end_offset(), Some(3));
    assert_eq!(span.start_position(), Some(build_position(1, 1)));
    assert_eq!(span.end_position(), Some(build_position(1, 4)));
    assert_eq!(span.to_string(), "1:1-1:4");
}

/// Test `from_offsets` sets only the offsets and `from_positions` only the
/// positions.
#[test]
fn span_from_a_range_sets_only_that_pair() {
    let offsets = Span::from_offsets(2..5).expect("ordered offsets are valid");
    let positions = Span::from_positions(build_position(1, 1)..build_position(2, 1))
        .expect("ordered positions are valid");

    assert_eq!(
        (offsets.start_offset(), offsets.end_offset()),
        (Some(2), Some(5))
    );
    assert_eq!(
        (offsets.start_position(), offsets.end_position()),
        (None, None)
    );
    assert_eq!(
        (positions.start_position(), positions.end_position()),
        (Some(build_position(1, 1)), Some(build_position(2, 1)))
    );
    assert_eq!(
        (positions.start_offset(), positions.end_offset()),
        (None, None)
    );
}

/// Test an offset range ending before its start is rejected with both
/// offsets.
#[test]
fn span_from_offsets_rejects_a_reversed_range() {
    let (start, end) = (5, 3);

    let result = Span::from_offsets(start..end);

    assert_eq!(
        result,
        Err(SpanError::EndOffsetBeforeStart { start: 5, end: 3 })
    );
}

/// Test a position range ending before its start is rejected with both
/// positions.
#[test]
fn span_from_positions_rejects_a_reversed_range() {
    let result = Span::from_positions(build_position(2, 1)..build_position(1, 1));

    assert_eq!(
        result,
        Err(SpanError::EndPositionBeforeStart {
            start: build_position(2, 1),
            end: build_position(1, 1),
        })
    );
}

/// Test an end offset set after the start offset is checked against it.
#[test]
fn span_with_end_offset_is_checked_against_the_start_offset() {
    let span = Span::unknown()
        .with_start_offset(5)
        .expect("a lone start offset is valid");

    let result = span.with_end_offset(3);

    assert_eq!(
        result,
        Err(SpanError::EndOffsetBeforeStart { start: 5, end: 3 })
    );
}

/// Test a start bound set after the end bound is checked against it, for
/// offsets and for positions.
#[test]
fn span_single_bound_builders_check_their_counterpart() {
    let offsets = Span::unknown()
        .with_end_offset(3)
        .expect("a lone end offset is valid");
    let positions = Span::unknown()
        .with_end_position(build_position(1, 1))
        .expect("a lone end position is valid");

    let offset_result = offsets.with_start_offset(5);
    let position_result = positions.with_start_position(build_position(2, 1));

    assert_eq!(
        offset_result,
        Err(SpanError::EndOffsetBeforeStart { start: 5, end: 3 })
    );
    assert_eq!(
        position_result,
        Err(SpanError::EndPositionBeforeStart {
            start: build_position(2, 1),
            end: build_position(1, 1),
        })
    );
}

/// Test the pair builders replace both bounds of their pair and keep the
/// other pair.
#[test]
fn span_with_a_range_replaces_both_bounds_of_its_pair() {
    let span = build_full_span(10..20, (1, 1)..(1, 2));

    let offsets_replaced = span.with_offsets(0..3).expect("ordered offsets are valid");
    let positions_replaced = span
        .with_positions(build_position(4, 1)..build_position(5, 1))
        .expect("ordered positions are valid");

    assert_eq!(offsets_replaced, build_full_span(0..3, (1, 1)..(1, 2)));
    assert_eq!(positions_replaced, build_full_span(10..20, (4, 1)..(5, 1)));
}

/// Test a single-bound builder replaces a bound that is already set, and
/// the replacement is checked against the other bound of its pair.
#[test]
fn span_single_bound_builders_replace_a_set_bound() {
    let span = Span::from_offsets(2..8).expect("ordered offsets are valid");

    let moved = span.with_start_offset(8).expect("equal bounds are valid");
    let refused = span.with_end_offset(1);

    assert_eq!(
        (moved.start_offset(), moved.end_offset()),
        (Some(8), Some(8))
    );
    assert_eq!(
        refused,
        Err(SpanError::EndOffsetBeforeStart { start: 2, end: 1 })
    );
}

#[test]
fn span_from_offsets_allows_equal_start_and_end() {
    let span = Span::from_offsets(4..4).expect("an empty range is valid");

    assert_eq!(span.start_offset(), Some(4));
    assert_eq!(span.end_offset(), Some(4));
}

#[test]
fn span_from_positions_allows_equal_start_and_end() {
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
fn span_builders_accept_bounds_they_cannot_order(
    #[case] start_offset: Option<u64>,
    #[case] end_offset: Option<u64>,
    #[case] start_position: Option<(u64, u64)>,
    #[case] end_position: Option<(u64, u64)>,
) {
    let result = try_build_span(start_offset, end_offset, start_position, end_position);

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
    let span = try_build_span(start_offset, end_offset, start_position, end_position)
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

#[test]
fn file_provenance_new_stores_the_span() {
    let span = build_offset_span(Some(0), Some(3));

    let provenance = FileProvenance::new("a.fhy", Some(span));

    assert_eq!(provenance.span(), Some(&span));
}

/// Test file paths are normalized lexically, the same on every platform:
/// `/` is the only separator, repeated separators and `.` components go, a
/// trailing separator goes, three or more leading separators become `/`,
/// the empty path becomes `.`, and `..`, `~`, a root of exactly `//`,
/// backslashes and drive letters stay.
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
#[case::library_symbol("mylib::matmul", build_file("mylib.fhyobj", None))]
fn named_provenance_try_new_stores_name_and_child(#[case] name: &str, #[case] child: Provenance) {
    let provenance = NamedProvenance::try_new(name, child.clone()).expect("name is non-empty");

    assert_eq!(provenance.name(), name);
    assert_eq!(provenance.child(), &child);
}

#[test]
fn named_provenance_try_new_rejects_an_empty_name() {
    let result = NamedProvenance::try_new("", Provenance::Unknown);

    assert_eq!(result, Err(NamedProvenanceError::EmptyName));
}

/// Test a name made only of whitespace is not empty and is accepted.
#[test]
fn named_provenance_try_new_accepts_a_whitespace_name() {
    let provenance = NamedProvenance::try_new(" ", Provenance::Unknown).expect("not empty");

    assert_eq!(provenance.name(), " ");
}

#[test]
fn call_site_provenance_new_stores_callee_and_caller() {
    let called = build_file("callee.fhy", None);
    let call_site = build_file("caller.fhy", None);

    let provenance = CallSiteProvenance::new(called.clone(), call_site.clone());

    assert_eq!(provenance.callee(), &called);
    assert_eq!(provenance.caller(), &call_site);
}

/// Test a fused provenance keeps its sources in order and no label.
#[test]
fn fused_provenance_new_keeps_sources_in_order() {
    let [a, b] = ["a.fhy", "b.fhy"].map(|path| build_file(path, None));

    let provenance = FusedProvenance::new(vec![a.clone(), b.clone()]);

    assert_eq!(provenance.sources(), &[a, b]);
    assert_eq!(provenance.label(), None);
}

/// Test a labelled fused provenance stores its label and its sources as
/// given.
#[test]
fn fused_provenance_labelled_stores_the_label() {
    let sources = vec![Provenance::Unknown, build_file("a.fhy", None)];

    let provenance = FusedProvenance::labelled(sources.clone(), "loop-fusion");

    assert_eq!(provenance.label(), Some("loop-fusion"));
    assert_eq!(provenance.sources(), sources.as_slice());
}

/// Test direct construction keeps a non-canonical fusion as given.
#[test]
fn fused_provenance_new_keeps_non_canonical_sources() {
    let sources = vec![Provenance::Unknown, build_fused(vec![], None)];

    let provenance = FusedProvenance::new(sources.clone());

    assert_eq!(provenance.sources(), sources.as_slice());
}

#[test]
fn fused_provenance_with_no_sources_differs_from_unknown() {
    assert_ne!(build_fused(vec![], None), Provenance::Unknown);
}

#[test]
fn fused_provenance_with_one_source_differs_from_the_source() {
    let a = build_file("a.fhy", None);

    assert_ne!(build_fused(vec![a.clone()], None), a);
}

#[test]
fn provenances_of_different_variants_are_unequal() {
    let file = build_file("a.fhy", None);
    let named = build_named("a.fhy", Provenance::Unknown);
    let call_site = build_call_site(Provenance::Unknown, Provenance::Unknown);

    assert_ne!(file, named);
    assert_ne!(named, call_site);
    assert_ne!(call_site, Provenance::Unknown);
}

/// Test equal provenances of every variant hash equally.
#[rstest]
#[case::file(|| build_file("a.fhy", Some(build_offset_span(Some(0), Some(3)))))]
#[case::named(|| build_named("fhy.add", Provenance::Unknown))]
#[case::call_site(|| build_call_site(build_file("a.fhy", None), build_file("b.fhy", None)))]
#[case::fused(|| build_fused(vec![build_file("a.fhy", None)], Some("cse")))]
fn equal_provenances_hash_equally(#[case] build: fn() -> Provenance) {
    let first = build();
    let second = build();

    assert_eq!(first, second);
    assert_eq!(hash_of(&first), hash_of(&second));
}

// =============================================================================
// Provenance::fuse
// =============================================================================

#[test]
fn fuse_with_no_inputs_returns_unknown() {
    let fused = Provenance::fuse([]);

    assert_eq!(fused, Provenance::Unknown);
}

#[test]
fn fuse_with_only_unknown_inputs_returns_unknown() {
    let fused = Provenance::fuse([Provenance::Unknown, Provenance::Unknown]);

    assert_eq!(fused, Provenance::Unknown);
}

/// Test the label is discarded when nothing survives the flattening.
#[rstest]
#[case::no_inputs(vec![])]
#[case::only_unknown(vec![Provenance::Unknown])]
#[case::empty_fusion(vec![build_fused(vec![Provenance::Unknown], None)])]
fn fuse_labelled_with_no_survivors_returns_unknown(#[case] inputs: Vec<Provenance>) {
    let fused = Provenance::fuse_labelled(inputs, "m");

    assert_eq!(fused, Provenance::Unknown);
}

#[test]
fn fuse_without_a_label_returns_a_single_input_unchanged() {
    let a = build_file("a.fhy", None);

    let fused = Provenance::fuse([a.clone()]);

    assert_eq!(fused, a);
}

#[test]
fn fuse_labelled_wraps_a_single_input() {
    let a = build_file("a.fhy", None);

    let fused = Provenance::fuse_labelled([a.clone()], "cse");

    assert_eq!(fused, build_fused(vec![a], Some("cse")));
}

#[test]
fn fuse_drops_unknown_inputs_from_mixed_input() {
    let [a, b] = ["a.fhy", "b.fhy"].map(|path| build_file(path, None));

    let fused = Provenance::fuse([a.clone(), Provenance::Unknown, b.clone()]);

    assert_eq!(fused, build_fused(vec![a, b], None));
}

#[test]
fn fuse_flattens_a_nested_unlabelled_fusion() {
    let [a, b, c] = ["a.fhy", "b.fhy", "c.fhy"].map(|path| build_file(path, None));
    let inner = build_fused(vec![a.clone(), b.clone()], None);

    let fused = Provenance::fuse([inner, c.clone()]);

    assert_eq!(fused, build_fused(vec![a, b, c], None));
}

#[test]
fn fuse_preserves_a_nested_labelled_fusion() {
    let [a, b, c] = ["a.fhy", "b.fhy", "c.fhy"].map(|path| build_file(path, None));
    let inner = build_fused(vec![a, b], Some("cse"));

    let fused = Provenance::fuse([inner.clone(), c.clone()]);

    assert_eq!(fused, build_fused(vec![inner, c], None));
}

/// Test a fusion labelled with the empty string is labelled, so it stays
/// whole.
#[test]
fn fuse_treats_an_empty_label_as_a_label() {
    let inner = build_fused(
        vec![build_file("a.fhy", None), build_file("b.fhy", None)],
        Some(""),
    );

    let fused = Provenance::fuse([inner.clone()]);

    assert_eq!(fused, inner);
}

#[test]
fn fuse_preserves_input_order() {
    let [a, b, c] = ["a.fhy", "b.fhy", "c.fhy"].map(|path| build_file(path, None));

    let fused = Provenance::fuse([a.clone(), b.clone(), c.clone()]);

    assert_eq!(fused, build_fused(vec![a, b, c], None));
}

#[test]
fn fuse_is_not_commutative() {
    let [a, b] = ["a.fhy", "b.fhy"].map(|path| build_file(path, None));

    let forward = Provenance::fuse([a.clone(), b.clone()]);
    let backward = Provenance::fuse([b, a]);

    assert_ne!(forward, backward);
}

#[test]
fn fuse_preserves_duplicate_sources() {
    let a = build_file("a.fhy", None);

    let fused = Provenance::fuse([a.clone(), a.clone()]);

    assert_eq!(fused, build_fused(vec![a.clone(), a], None));
}

#[test]
fn fuse_labelled_attaches_the_label_to_the_result() {
    let [a, b] = ["a.fhy", "b.fhy"].map(|path| build_file(path, None));

    let fused = Provenance::fuse_labelled([a.clone(), b.clone()], "loop-fusion");

    assert_eq!(fused, build_fused(vec![a, b], Some("loop-fusion")));
}

/// Test unknown provenances revealed by splicing are dropped too.
#[test]
fn fuse_drops_unknown_inside_directly_constructed_nested_fused() {
    let [a, b] = ["a.fhy", "b.fhy"].map(|path| build_file(path, None));
    let non_canonical_inner = build_fused(vec![Provenance::Unknown, a.clone()], None);

    let fused = Provenance::fuse([non_canonical_inner, b.clone()]);

    assert_eq!(fused, build_fused(vec![a, b], None));
}

/// Test unlabelled fusions are spliced at any depth.
#[test]
fn fuse_flattens_nested_unlabelled_fusions_transitively() {
    let [a, b, c] = ["a.fhy", "b.fhy", "c.fhy"].map(|path| build_file(path, None));
    let deeply_nested = build_fused(
        vec![a.clone(), build_fused(vec![b.clone(), c.clone()], None)],
        None,
    );

    let fused = Provenance::fuse([deeply_nested]);

    assert_eq!(fused, build_fused(vec![a, b, c], None));
}

/// Test splicing stops at a labelled fusion, even deep inside.
#[test]
fn fuse_does_not_flatten_through_a_labelled_fusion() {
    let a = build_file("a.fhy", None);
    let labeled_inner = build_fused(
        vec![build_file("b.fhy", None), build_file("c.fhy", None)],
        Some("cse"),
    );
    let outer_unlabelled = build_fused(vec![a.clone(), labeled_inner.clone()], None);

    let fused = Provenance::fuse([outer_unlabelled]);

    assert_eq!(fused, build_fused(vec![a, labeled_inner], None));
}

/// Test dropping, splicing and ordering compose.
#[test]
fn fuse_combines_all_reduction_rules() {
    let [a, b, c] = ["a.fhy", "b.fhy", "c.fhy"].map(|path| build_file(path, None));
    let inner_unlabelled = build_fused(vec![a.clone(), b.clone()], None);
    let inner_labelled = build_fused(vec![b.clone(), c.clone()], Some("x"));

    let fused = Provenance::fuse([
        Provenance::Unknown,
        inner_unlabelled,
        Provenance::Unknown,
        inner_labelled.clone(),
        c.clone(),
    ]);

    assert_eq!(fused, build_fused(vec![a, b, inner_labelled, c], None));
}

/// Test a single surviving source reached through splicing is returned
/// bare.
#[test]
fn fuse_unwraps_a_single_source_found_by_splicing() {
    let a = build_file("a.fhy", None);
    let nested = build_fused(vec![build_fused(vec![a.clone()], None)], None);

    let fused = Provenance::fuse([nested]);

    assert_eq!(fused, a);
}

/// Test fusions nested inside named and call-site provenances are not
/// inspected.
#[test]
fn fuse_does_not_look_inside_named_or_call_site_children() {
    let named = build_named(
        "n",
        build_fused(vec![Provenance::Unknown, build_file("a.fhy", None)], None),
    );
    let call_site = build_call_site(
        build_fused(vec![build_file("a.fhy", None)], None),
        Provenance::Unknown,
    );

    assert_eq!(Provenance::fuse([named.clone()]), named);
    assert_eq!(Provenance::fuse([call_site.clone()]), call_site);
}

/// Test unlabelled fusions nested [`SMALL_STACK_DEPTH`] levels deep, each
/// holding a file then the next fusion, are spliced into one flat fusion of
/// every file in order, on a small thread stack.
#[test]
fn fuse_flattens_deeply_nested_unlabelled_fusions_on_a_small_stack() {
    run_on_small_stack(|| {
        let files: Vec<Provenance> = (0..=SMALL_STACK_DEPTH)
            .map(|index| build_file(&format!("{index}.fhy"), None))
            .collect();
        let nested = files[..SMALL_STACK_DEPTH]
            .iter()
            .rev()
            .fold(files[SMALL_STACK_DEPTH].clone(), |nested, file| {
                build_fused(vec![file.clone(), nested], None)
            });

        let fused = Provenance::fuse([nested]);

        let Provenance::Fused(flat) = &fused else {
            panic!("the splice leaves a fusion");
        };
        assert_eq!(flat.label(), None);
        assert!(flat.sources() == files, "the spliced sources differ");
    });
}

#[test]
fn fuse_is_associative() {
    let [a, b, c] = ["a.fhy", "b.fhy", "c.fhy"].map(|path| build_file(path, None));

    let left = Provenance::fuse([Provenance::fuse([a.clone(), b.clone()]), c.clone()]);
    let right = Provenance::fuse([a.clone(), Provenance::fuse([b.clone(), c.clone()])]);
    let flat = Provenance::fuse([a, b, c]);

    assert_eq!(left, flat);
    assert_eq!(right, flat);
}

/// Test a labelled inner fusion breaks associativity, since it stays whole.
#[test]
fn fuse_labelled_is_not_associative() {
    let [a, b, c] = ["a.fhy", "b.fhy", "c.fhy"].map(|path| build_file(path, None));

    let grouped = Provenance::fuse([
        Provenance::fuse_labelled([a.clone(), b.clone()], "m"),
        c.clone(),
    ]);
    let flat = Provenance::fuse_labelled([a, b, c], "m");

    assert_ne!(grouped, flat);
}

/// Test a deeply nested chain of unlabelled fusions flattens without
/// exhausting the stack.
#[test]
fn fuse_flattens_a_deeply_nested_chain() {
    const DEPTH: usize = 3000;
    let [a, b] = ["a.fhy", "b.fhy"].map(|path| build_file(path, None));
    let nested = (0..DEPTH).fold(build_fused(vec![a.clone()], None), |nested, _| {
        build_fused(vec![nested], None)
    });

    let fused = Provenance::fuse([nested, b.clone()]);

    assert_eq!(fused, build_fused(vec![a, b], None));
}

// =============================================================================
// Display
// =============================================================================

/// Test every rendering rule of the provenance variants.
#[rstest]
#[case::unknown(Provenance::Unknown, "<unknown>")]
#[case::file_without_span(build_file("a.fhy", None), "a.fhy")]
#[case::file_with_unknown_span(build_file("a.fhy", Some(Span::unknown())), "a.fhy")]
#[case::file_with_offset_span(
    build_file("a.fhy", Some(build_offset_span(Some(0), Some(3)))),
    "a.fhy:@0-3"
)]
#[case::file_with_position_span(
    build_file("a.fhy", Some(build_position_span(Some((1, 1)), Some((1, 4))))),
    "a.fhy:1:1-1:4"
)]
#[case::file_renders_the_normalized_path(
    build_file("./x//y.fhy", Some(build_offset_span(Some(0), Some(3)))),
    "x/y.fhy:@0-3"
)]
#[case::named_with_unknown_child(build_named("fhy.add", Provenance::Unknown), "fhy.add")]
#[case::named_with_known_child(
    build_named("mylib::matmul", build_file("mylib.fhyobj", None)),
    "mylib::matmul (mylib.fhyobj)"
)]
#[case::named_with_empty_fusion_child(
    build_named("empty-fusion", build_fused(vec![], None)),
    "empty-fusion (fused[])"
)]
#[case::call_site(
    build_call_site(build_file("callee.fhy", None), build_file("caller.fhy", None)),
    "callee.fhy at caller.fhy"
)]
#[case::call_site_chain(
    build_call_site(
        build_call_site(build_file("a.fhy", None), build_file("b.fhy", None)),
        build_file("c.fhy", None)
    ),
    "a.fhy at b.fhy at c.fhy"
)]
#[case::fused_without_label(
    build_fused(vec![build_file("a.fhy", None), build_file("b.fhy", None)], None),
    "fused[a.fhy, b.fhy]"
)]
#[case::fused_with_label(build_fused(vec![build_file("a.fhy", None)], Some("loop-fusion")), "loop-fusion[a.fhy]")]
#[case::fused_recurses_into_sources(
    build_fused(
        vec![
            build_fused(vec![build_file("a.fhy", None), build_file("b.fhy", None)], Some("cse")),
            build_file("c.fhy", None),
        ],
        None,
    ),
    "fused[cse[a.fhy, b.fhy], c.fhy]"
)]
#[case::fused_with_no_sources(build_fused(vec![], None), "fused[]")]
#[case::fused_with_empty_label(
    build_fused(vec![build_file("a.fhy", None), build_file("b.fhy", None)], Some("")),
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
        provenance: build_file("kernel.fhy", Some(build_offset_span(Some(0), Some(4)))),
    };
    let right = StoryNode {
        provenance: build_named("fhy.add", Provenance::Unknown),
    };

    let merged = StoryNode {
        provenance: Provenance::fuse_labelled(
            [left.provenance().clone(), right.provenance().clone()],
            "cse",
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
    let inner = build_file("inner.fhy", None);
    let middle = build_file("middle.fhy", None);
    let outer = build_file("outer.fhy", None);

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

#[test]
fn position_encodes_as_line_and_column() {
    let encoded = serde_json::to_value(build_position(2, 8)).expect("positions encode");

    assert_eq!(encoded, json!({"line": 2, "column": 8}));
}

/// Test a span encodes all four keys, with `null` for absent bounds.
#[rstest]
#[case::full(
    build_full_span(0..3, (1, 1)..(1, 4)),
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

/// Test each provenance variant encodes externally tagged: the unknown
/// provenance as the string `"unknown"` and every other variant as a
/// one-key map from its snake-case name to its fields, with the normalized
/// path for a file.
#[rstest]
#[case::unknown(Provenance::Unknown, json!("unknown"))]
#[case::file(
    build_file("./a//b.fhy", Some(build_offset_span(Some(0), Some(3)))),
    json!({
        "file": {
            "file_path": "a/b.fhy",
            "span": {"start_offset": 0, "end_offset": 3, "start_position": null, "end_position": null},
        },
    })
)]
#[case::file_without_span(
    build_file("c.fhy", None),
    json!({"file": {"file_path": "c.fhy", "span": null}})
)]
#[case::named(
    build_named("lib", Provenance::Unknown),
    json!({"named": {"name": "lib", "child": "unknown"}})
)]
#[case::call_site(
    build_call_site(build_file("a.fhy", None), Provenance::Unknown),
    json!({
        "call_site": {
            "callee": {"file": {"file_path": "a.fhy", "span": null}},
            "caller": "unknown",
        },
    })
)]
#[case::fused_without_label(
    build_fused(vec![Provenance::Unknown], None),
    json!({"fused": {"sources": ["unknown"], "label": null}})
)]
#[case::fused_with_label(
    build_fused(vec![], Some("fuse")),
    json!({"fused": {"sources": [], "label": "fuse"}})
)]
fn provenance_encodes_externally_tagged(#[case] provenance: Provenance, #[case] expected: Value) {
    let encoded = serde_json::to_value(&provenance).expect("provenances encode");

    assert_eq!(encoded, expected);
}

/// Test every variant round-trips through JSON text.
#[rstest]
#[case::unknown(Provenance::Unknown)]
#[case::file(build_file("c.fhy", None))]
#[case::file_with_full_span(build_file("a.fhy", Some(build_full_span(0..3, (1, 1)..(1, 4)))))]
#[case::builtin(build_named("fhy.add", Provenance::Unknown))]
#[case::library_symbol(build_named("mylib::matmul", build_file("mylib.fhyobj", None)))]
#[case::call_site(build_call_site(
    build_file("a.fhy", Some(build_offset_span(Some(0), Some(10)))),
    build_file("b.fhy", None),
))]
#[case::inlined_call_site(build_call_site(
    build_named("inlined", build_file("a.fhy", None)),
    build_file("b.fhy", None),
))]
#[case::fused(build_fused(vec![build_file("a.fhy", None), build_file("b.fhy", None)], None))]
#[case::fused_with_label(build_fused(vec![build_file("a.fhy", None)], Some("loop-fusion")))]
fn provenance_round_trips_through_json(#[case] provenance: Provenance) {
    let json = serde_json::to_string(&provenance).expect("provenances encode");

    let restored: Provenance = serde_json::from_str(&json).expect("encoded provenances decode");

    assert_eq!(restored, provenance);
}

/// Test position and span values round-trip through JSON text.
#[test]
fn position_and_span_round_trip_through_json() {
    let position = build_position(7, 3);
    let span = Span::unknown()
        .with_start_offset(1)
        .and_then(|span| span.with_start_position(position))
        .expect("lone bounds are valid");

    let restored_position: Position =
        serde_json::from_str(&serde_json::to_string(&position).unwrap()).unwrap();
    let restored_span: Span = serde_json::from_str(&serde_json::to_string(&span).unwrap()).unwrap();

    assert_eq!(restored_position, position);
    assert_eq!(restored_span, span);
}

#[test]
fn decoded_unknown_provenances_compare_equal() {
    let payload = json!("unknown");

    let first: Provenance = serde_json::from_value(payload.clone()).expect("valid payload");
    let second: Provenance = serde_json::from_value(payload).expect("valid payload");

    assert_eq!(first, Provenance::Unknown);
    assert_eq!(first, second);
}

#[test]
fn provenance_decode_normalizes_the_file_path() {
    let payload = json!({"file": {"file_path": "./x//y.fhy/", "span": null}});

    let decoded: Provenance = serde_json::from_value(payload).expect("valid payload");

    assert_eq!(decoded, build_file("x/y.fhy", None));
}

/// Test a file provenance written with an unnormalized path, in either
/// format, decodes to the normalized path.
#[test]
fn file_provenance_decode_normalizes_the_path_in_either_format() {
    /// The fields of a file provenance, with the path written as given.
    #[derive(serde::Serialize)]
    struct RawFile<'a> {
        file_path: &'a str,
        span: Option<Span>,
    }
    let raw = RawFile {
        file_path: "./src//a.fhy/",
        span: None,
    };
    let json = serde_json::to_string(&raw).expect("the raw fields encode");
    let bytes = postcard::to_allocvec(&raw).expect("the raw fields encode");

    let from_json: FileProvenance = serde_json::from_str(&json).expect("the JSON decodes");
    let from_postcard: FileProvenance = postcard::from_bytes(&bytes).expect("the bytes decode");

    assert_eq!(from_json.file_path(), "src/a.fhy");
    assert_eq!(from_postcard.file_path(), "src/a.fhy");
}

/// Test a span payload missing a bound's key decodes with that bound
/// absent.
#[rstest]
#[case::no_keys(json!({}), Span::unknown())]
#[case::only_offsets(
    json!({"start_offset": 2, "end_offset": 5}),
    Span::from_offsets(2..5).unwrap()
)]
#[case::only_an_end_position(
    json!({"end_position": {"line": 3, "column": 1}}),
    Span::unknown().with_end_position(build_position(3, 1)).unwrap()
)]
fn span_decode_reads_a_missing_bound_as_absent(#[case] payload: Value, #[case] expected: Span) {
    let decoded: Span = serde_json::from_str(&payload.to_string()).expect("the payload decodes");

    assert_eq!(decoded, expected);
}

/// Test a file payload without its span and a fused payload without its
/// label decode with those fields absent.
#[test]
fn file_and_fused_decode_read_a_missing_option_as_absent() {
    let file: Provenance =
        serde_json::from_str(r#"{"file": {"file_path": "a.fhy"}}"#).expect("the file decodes");
    let fused: Provenance =
        serde_json::from_str(r#"{"fused": {"sources": []}}"#).expect("the fusion decodes");

    assert_eq!(file, build_file("a.fhy", None));
    assert_eq!(fused, build_fused(vec![], None));
}

/// Test a span payload with reversed offsets is refused with the span
/// error's text.
#[test]
fn span_decode_rejects_reversed_offsets_with_the_span_error_text() {
    let payload = r#"{"start_offset": 5, "end_offset": 3}"#;

    let error = serde_json::from_str::<Span>(payload).expect_err("the offsets are reversed");

    let expected = SpanError::EndOffsetBeforeStart { start: 5, end: 3 }.to_string();
    assert!(error.to_string().contains(&expected), "{error}");
}

/// Test malformed position payloads are rejected with a data error.
#[rstest]
#[case::missing_column(json!({"line": 1}))]
#[case::bool_line(json!({"line": true, "column": 1}))]
#[case::extra_key(json!({"line": 1, "column": 2, "z": 3}))]
#[case::zero_line(json!({"line": 0, "column": 1}))]
#[case::zero_column(json!({"line": 1, "column": 0}))]
#[case::float_line(json!({"line": 1.0, "column": 1}))]
#[case::string_line(json!({"line": "1", "column": 1}))]
#[case::negative_line(json!({"line": -1, "column": 1}))]
#[case::null_line(json!({"line": null, "column": 1}))]
fn position_decode_rejects_malformed_payloads(#[case] payload: Value) {
    assert_decode_rejected::<Position>(&payload.to_string(), Category::Data, None);
}

/// Test malformed span payloads are rejected with a data error, whose
/// message names the crate's own error where one applies.
#[rstest]
#[case::extra_key(
    json!({"start_offset": 0, "end_offset": 3, "start_position": null, "end_position": null, "extra": 1}),
    None
)]
#[case::end_offset_before_start(
    json!({"start_offset": 5, "end_offset": 3, "start_position": null, "end_position": null}),
    Some(SpanError::EndOffsetBeforeStart { start: 5, end: 3 }.to_string())
)]
#[case::end_position_before_start(
    json!({
    "start_offset": null,
    "end_offset": null,
    "start_position": {"line": 2, "column": 1},
    "end_position": {"line": 1, "column": 1},
}),
    Some(SpanError::EndPositionBeforeStart { start: build_position(2, 1), end: build_position(1, 1) }.to_string())
)]
#[case::negative_offset(
    json!({"start_offset": -1, "end_offset": null, "start_position": null, "end_position": null}),
    None
)]
#[case::bool_offset(
    json!({"start_offset": true, "end_offset": null, "start_position": null, "end_position": null}),
    None
)]
#[case::invalid_position(
    json!({
    "start_offset": null,
    "end_offset": null,
    "start_position": {"line": 0, "column": 1},
    "end_position": null,
}),
    None
)]
fn span_decode_rejects_malformed_payloads(
    #[case] payload: Value,
    #[case] crate_error: Option<String>,
) {
    assert_decode_rejected::<Span>(&payload.to_string(), Category::Data, crate_error.as_deref());
}

/// Test malformed provenance payloads, including an unknown variant name,
/// a variant body of the wrong shape and two variants at once, are rejected
/// with an error of the expected category, whose message names the crate's
/// own error where one applies. Where a provenance is due, `serde_json`
/// reports a JSON value that is neither a string nor a map, or a second
/// variant key, as a syntax error rather than a data error.
#[rstest]
#[case::unknown_variant_name(json!({"does_not_exist": {}}), Category::Data, None)]
#[case::unknown_variant_string(json!("does_not_exist"), Category::Data, None)]
#[case::variant_with_a_non_map_body(json!({"file": 5}), Category::Data, None)]
#[case::unit_variant_given_a_body(json!({"unknown": {"x": 1}}), Category::Data, None)]
#[case::newtype_variant_without_a_body(json!("file"), Category::Data, None)]
#[case::two_top_level_keys(
    json!({"file": {"file_path": "a", "span": null}, "named": {"name": "n", "child": "unknown"}}),
    Category::Syntax,
    None
)]
#[case::file_path_not_a_string(json!({"file": {"file_path": 3, "span": null}}), Category::Data, None)]
#[case::file_with_an_unknown_field(
    json!({"file": {"file_path": "a", "span": null, "extra": 1}}),
    Category::Data,
    None
)]
#[case::empty_name(
    json!({"named": {"name": "", "child": "unknown"}}),
    Category::Data,
    Some(NamedProvenanceError::EmptyName.to_string())
)]
#[case::named_child_not_a_provenance(
    json!({"named": {"name": "n", "child": {"line": 1, "column": 1}}}),
    Category::Data,
    None
)]
#[case::null_caller(json!({"call_site": {"callee": "unknown", "caller": null}}), Category::Syntax, None)]
#[case::integer_label(json!({"fused": {"sources": [], "label": 5}}), Category::Data, None)]
#[case::null_sources(json!({"fused": {"sources": null, "label": null}}), Category::Data, None)]
#[case::invalid_nested_source(
    json!({"fused": {"sources": [{"named": {"name": "", "child": "unknown"}}], "label": null}}),
    Category::Data,
    Some(NamedProvenanceError::EmptyName.to_string())
)]
#[case::not_a_map(json!(["unknown"]), Category::Syntax, None)]
fn provenance_decode_rejects_malformed_payloads(
    #[case] payload: Value,
    #[case] category: Category,
    #[case] crate_error: Option<String>,
) {
    assert_decode_rejected::<Provenance>(&payload.to_string(), category, crate_error.as_deref());
}

/// Test a line beyond `u64::MAX` is rejected on decode: lines, columns and
/// offsets decode as `u64`.
#[rstest]
#[case::huge_line(r#"{"line": 1180591620717411303424, "column": 1}"#)]
#[case::line_just_past_u64(r#"{"line": 18446744073709551616, "column": 1}"#)]
fn position_decode_rejects_values_beyond_u64(#[case] payload_text: &str) {
    assert_decode_rejected::<Position>(payload_text, Category::Data, None);
}

/// Test a span offset beyond `u64` is rejected on decode.
#[test]
fn span_decode_rejects_an_offset_beyond_u64() {
    let payload_text = r#"{"start_offset": 18446744073709551616, "end_offset": null,
        "start_position": null, "end_position": null}"#;

    assert_decode_rejected::<Span>(payload_text, Category::Data, None);
}

/// Return `depth` named provenances nested over the unknown provenance.
fn build_nested_named(depth: usize) -> Provenance {
    (0..depth).fold(Provenance::Unknown, |provenance, _| {
        build_named("n", provenance)
    })
}

/// Test JSON text nested far past `serde_json`'s nesting limit is refused
/// with an error rather than a crash, while a moderately deep provenance
/// round-trips.
///
/// The exact limit and its message are `serde_json`'s, not this crate's.
#[test]
fn provenance_nested_past_the_json_limit_is_refused_not_crashed() {
    let moderate = build_nested_named(40);
    let too_deep = build_nested_named(200);
    let moderate_text = serde_json::to_string(&moderate).expect("the provenance serializes");
    let too_deep_text = serde_json::to_string(&too_deep).expect("the provenance serializes");

    let decoded: Provenance =
        serde_json::from_str(&moderate_text).expect("40 levels over a leaf decode");
    let refusal = serde_json::from_str::<Provenance>(&too_deep_text);

    assert_eq!(decoded, moderate);
    assert!(refusal.is_err(), "200 levels over a leaf are refused");
}

// =============================================================================
// Errors and thread safety
// =============================================================================

/// Test every position error renders its full message.
#[rstest]
#[case::zero_line(PositionError::ZeroLine, "a position's line must be at least 1")]
#[case::zero_column(PositionError::ZeroColumn, "a position's column must be at least 1")]
fn position_error_display_writes_the_full_message(
    #[case] error: PositionError,
    #[case] expected: &str,
) {
    let message = error.to_string();

    assert_eq!(message, expected);
}

/// Test every span error renders its full message, with the offsets and
/// positions of an unordered span each in its own place.
#[rstest]
#[case::end_offset(
    SpanError::EndOffsetBeforeStart { start: 5, end: 3 },
    "a span's end offset 3 precedes its start offset 5"
)]
#[case::end_position(
    SpanError::EndPositionBeforeStart { start: build_position(2, 1), end: build_position(1, 1) },
    "a span's end position 1:1 precedes its start position 2:1"
)]
fn span_error_display_writes_the_full_message(#[case] error: SpanError, #[case] expected: &str) {
    let message = error.to_string();

    assert_eq!(message, expected);
}

#[test]
fn named_provenance_error_display_writes_the_full_message() {
    let message = NamedProvenanceError::EmptyName.to_string();

    assert_eq!(message, "a named provenance's name must be non-empty");
}

/// Test every error renders as one lowercase line without a trailing
/// period, and has no underlying cause.
#[rstest]
#[case::zero_line(&PositionError::ZeroLine)]
#[case::zero_column(&PositionError::ZeroColumn)]
#[case::end_offset(&SpanError::EndOffsetBeforeStart { start: 5, end: 3 })]
#[case::end_position(&SpanError::EndPositionBeforeStart {
    start: build_position(2, 1),
    end: build_position(1, 1),
})]
#[case::empty_name(&NamedProvenanceError::EmptyName)]
fn provenance_error_display_is_one_lowercase_line(#[case] error: &dyn std::error::Error) {
    let message = error.to_string();

    assert!(!message.contains('\n'), "{message:?}");
    assert!(!message.ends_with('.'), "{message:?}");
    assert!(
        message.starts_with(|first: char| first.is_lowercase()),
        "{message:?}"
    );
    assert!(error.source().is_none());
}

/// Compile-time check that the provenance types can cross threads.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Position>();
    assert_send_sync::<Span>();
    assert_send_sync::<Provenance>();
    assert_send_sync::<PositionError>();
    assert_send_sync::<SpanError>();
    assert_send_sync::<NamedProvenanceError>();
};
