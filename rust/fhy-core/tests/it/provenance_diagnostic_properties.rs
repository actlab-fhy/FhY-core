//! Property tests for `Provenance::fuse`, file path normalization, the
//! provenance wire form, position ordering, and `ValidationReport`.
//!
//! `flatten_sources` is a test-side reference for the reduction `fuse`
//! documents (drop `Provenance::Unknown`, splice the sources of every
//! unlabelled fusion in order, keep everything else whole), written
//! independently of `fuse`'s own loop.

use std::fmt::Write as _;

use fhy_core::diagnostic::{Diagnostic, DiagnosticLevel, Note, ValidationReport};
use fhy_core::provenance::{
    CallSiteProvenance, FileProvenance, FusedProvenance, NamedProvenance, Position, Provenance,
    Span,
};
use proptest::prelude::*;
use proptest::sample::select;

/// File paths the strategies draw from: spellings that normalize to one
/// another, absolute paths, the POSIX `//` root, the empty path and the
/// root alone, backslashes, non-ASCII text, and characters JSON escapes.
const FILE_PATHS: &[&str] = &[
    "a.fhy",
    "b.fhy",
    "c.fhy",
    "./a.fhy",
    "dir//b.fhy",
    "dir/../c.fhy",
    "/abs/a.fhy",
    "/abs//./b.fhy/",
    "//a",
    "//a/./b",
    "",
    "/",
    "C:\\src\\a.fhy",
    "dir\\b.fhy",
    "ünïcødé/файл.fhy",
    "quote\"and\\slash.fhy",
    "tab\there/new\nline\u{1}.fhy",
];

/// Labels the strategies draw from, including the empty label.
const LABELS: &[&str] = &["cse", "loop-fusion", ""];

/// Names the strategies draw from for named provenances.
const NAMES: &[&str] = &["n", "fhy.add", " "];

/// Largest fusion depth the tree strategy builds.
const MAXIMUM_TREE_DEPTH: u32 = 3;

/// Build the file provenance for `path` over `span`.
fn build_file(path: &str, span: Option<Span>) -> Provenance {
    Provenance::File(FileProvenance::new(path, span))
}

/// Build the named provenance `name` over `child`.
fn build_named(name: &str, child: Provenance) -> Provenance {
    Provenance::Named(NamedProvenance::try_new(name, child).expect("names are non-empty"))
}

/// Return the flattened source list `Provenance::fuse` documents for
/// `inputs`: unknown provenances dropped and unlabelled fusions spliced in
/// order at any depth.
fn flatten_sources(inputs: &[Provenance]) -> Vec<Provenance> {
    let mut flat = Vec::new();
    for input in inputs {
        match input {
            Provenance::Unknown => {}
            Provenance::Fused(fused) if fused.metadata().is_none() => {
                flat.extend(flatten_sources(fused.sources()));
            }
            other => flat.push(other.clone()),
        }
    }
    flat
}

/// Return whether `provenance` holds, at its top level, an unknown
/// provenance or an unlabelled fusion among its fused sources.
fn has_reducible_sources(provenance: &Provenance) -> bool {
    match provenance {
        Provenance::Fused(fused) => fused.sources().iter().any(|source| {
            matches!(source, Provenance::Unknown)
                || matches!(source, Provenance::Fused(inner) if inner.metadata().is_none())
        }),
        _ => false,
    }
}

/// Return a strategy for offsets and line or column numbers, small or
/// near `u64::MAX`.
fn arbitrary_bound_value() -> BoxedStrategy<u64> {
    prop_oneof![3 => 0_u64..1000, 1 => (u64::MAX - 1000)..=u64::MAX].boxed()
}

/// Return a strategy for an optional start and end drawn from `values`,
/// either one alone, or both with the end at or after the start by
/// `distance`, where `u64::MAX` caps it.
fn arbitrary_ordered_bounds<T: std::fmt::Debug + Clone>(
    values: impl Strategy<Value = T> + Clone,
    advance: fn(&T, u64) -> T,
) -> impl Strategy<Value = (Option<T>, Option<T>)> {
    prop_oneof![
        Just((None, None)),
        values.clone().prop_map(|start| (Some(start), None)),
        values.clone().prop_map(|end| (None, Some(end))),
        (values, prop_oneof![0_u64..3, any::<u64>()]).prop_map(move |(start, distance)| {
            let end = advance(&start, distance);
            (Some(start), Some(end))
        }),
    ]
}

/// Return a strategy for positions with small or huge lines and columns.
fn arbitrary_position() -> BoxedStrategy<Position> {
    (arbitrary_bound_value(), arbitrary_bound_value())
        .prop_map(|(line, column)| Position::try_new(line.max(1), column.max(1)).expect("non-zero"))
        .boxed()
}

/// Return the position `distance` lines after `start`, at the same column
/// when `distance` is zero and at column 1 otherwise, with `u64::MAX`
/// capping the line.
fn advance_position(start: &Position, distance: u64) -> Position {
    let line = start.line().get().saturating_add(distance);
    let column = if line == start.line().get() {
        start.column().get()
    } else {
        1
    };
    Position::try_new(line, column).expect("non-zero")
}

/// Return a strategy for spans mixing unknown, one-sided and two-sided
/// offset bounds with unknown, one-sided and two-sided position bounds,
/// with small and huge values.
fn arbitrary_span() -> impl Strategy<Value = Span> {
    (
        arbitrary_ordered_bounds(arbitrary_bound_value(), |start, distance| {
            start.saturating_add(distance)
        }),
        arbitrary_ordered_bounds(arbitrary_position(), advance_position),
    )
        .prop_map(
            |((start_offset, end_offset), (start_position, end_position))| {
                Span::try_new(start_offset, end_offset, start_position, end_position)
                    .expect("bounds are ordered")
            },
        )
}

/// Return a strategy for a leaf provenance: unknown, a file, or a name over
/// a file.
fn arbitrary_leaf() -> impl Strategy<Value = Provenance> {
    prop_oneof![
        Just(Provenance::Unknown),
        (select(FILE_PATHS), proptest::option::of(arbitrary_span()))
            .prop_map(|(path, span)| build_file(path, span)),
        (select(NAMES), select(FILE_PATHS))
            .prop_map(|(name, path)| build_named(name, build_file(path, None))),
    ]
}

/// Return a strategy for provenance trees up to [`MAXIMUM_TREE_DEPTH`]
/// levels deep, whose inner nodes are fusions (labelled or not), names and
/// call sites.
fn arbitrary_tree() -> impl Strategy<Value = Provenance> {
    arbitrary_leaf().prop_recursive(MAXIMUM_TREE_DEPTH, 32, 3, |inner| {
        prop_oneof![
            3 => (
                proptest::collection::vec(inner.clone(), 0..=3),
                proptest::option::of(select(LABELS)),
            )
                .prop_map(|(sources, label)| {
                    Provenance::Fused(FusedProvenance::new(sources, label.map(str::to_owned)))
                }),
            1 => (select(NAMES), inner.clone()).prop_map(|(name, child)| build_named(name, child)),
            1 => (inner.clone(), inner).prop_map(|(callee, caller)| {
                Provenance::CallSite(CallSiteProvenance::new(callee, caller))
            }),
        ]
    })
}

/// Return a strategy for 0 to 4 trees, the inputs of one `fuse` call.
fn arbitrary_inputs() -> impl Strategy<Value = Vec<Provenance>> {
    proptest::collection::vec(arbitrary_tree(), 0..=4)
}

/// Return a strategy for an optional label.
fn arbitrary_metadata() -> impl Strategy<Value = Option<&'static str>> {
    proptest::option::of(select(LABELS))
}

proptest! {
    /// Test normalizing a file path twice changes nothing more than
    /// normalizing it once, and the result has no empty or `.` component
    /// after its root, whatever mix of separators, dots, backslashes and
    /// drive letters the path holds.
    #[test]
    fn file_path_normalization_is_idempotent(path in "[a/.\\\\:C]{0,12}") {
        let normalized = FileProvenance::new(&path, None);

        let renormalized = FileProvenance::new(normalized.file_path(), None);

        prop_assert_eq!(renormalized.file_path(), normalized.file_path());
        let unrooted = normalized.file_path().trim_start_matches('/');
        prop_assert!(
            unrooted == "."
                || unrooted.is_empty()
                || unrooted.split('/').all(|component| !component.is_empty() && component != "."),
            "{:?} normalized to {:?}",
            path,
            normalized.file_path()
        );
    }
}

proptest! {
    /// Test `fuse` equals the reference flattening, collapsing to the unknown
    /// provenance for no survivors and to the bare survivor for one survivor
    /// without metadata.
    #[test]
    fn fuse_result_equals_the_flattened_input(
        inputs in arbitrary_inputs(),
        metadata in arbitrary_metadata(),
    ) {
        let flat = flatten_sources(&inputs);

        let result = Provenance::fuse(inputs, metadata);

        match (flat.as_slice(), metadata) {
            ([], _) => prop_assert_eq!(result, Provenance::Unknown),
            ([single], None) => prop_assert_eq!(&result, single),
            _ => {
                let Provenance::Fused(fused) = &result else {
                    return Err(TestCaseError::fail(format!("expected a fusion, got {result:?}")));
                };
                prop_assert_eq!(fused.sources(), flat.as_slice());
                prop_assert_eq!(fused.metadata(), metadata);
                prop_assert!(!has_reducible_sources(&result));
            }
        }
    }

    /// Test fusing `fuse`'s own output alone, without metadata, changes
    /// nothing.
    #[test]
    fn fuse_is_idempotent_on_its_own_output(
        inputs in arbitrary_inputs(),
        metadata in arbitrary_metadata(),
    ) {
        let result = Provenance::fuse(inputs, metadata);

        let refused = Provenance::fuse([result.clone()], None);

        prop_assert_eq!(refused, result);
    }

    /// Test fusing already-fused groups without metadata equals fusing all
    /// of their inputs at once.
    #[test]
    fn fuse_is_associative_without_metadata(
        first in arbitrary_inputs(),
        second in arbitrary_inputs(),
        third in arbitrary_inputs(),
    ) {
        let all: Vec<Provenance> =
            first.iter().chain(&second).chain(&third).cloned().collect();

        let grouped = Provenance::fuse(
            [
                Provenance::fuse(first, None),
                Provenance::fuse(second, None),
                Provenance::fuse(third, None),
            ],
            None,
        );

        prop_assert_eq!(grouped, Provenance::fuse(all, None));
    }

    /// Test inserting the unknown provenance anywhere among the inputs does
    /// not change the result.
    #[test]
    fn fuse_treats_unknown_as_an_identity(
        inputs in arbitrary_inputs(),
        index in any::<prop::sample::Index>(),
        metadata in arbitrary_metadata(),
    ) {
        let mut with_unknown = inputs.clone();
        with_unknown.insert(index.index(inputs.len() + 1), Provenance::Unknown);

        let expected = Provenance::fuse(inputs, metadata);

        prop_assert_eq!(Provenance::fuse(with_unknown, metadata), expected);
    }

    /// Test every provenance tree round-trips through JSON text.
    #[test]
    fn provenance_round_trips_through_json(provenance in arbitrary_tree()) {
        let json = serde_json::to_string(&provenance).expect("provenances encode");

        let restored: Provenance = serde_json::from_str(&json).expect("encoded provenances decode");

        prop_assert_eq!(restored, provenance);
    }

    /// Test re-encoding a decoded provenance reproduces the same JSON text.
    #[test]
    fn provenance_json_text_is_stable_across_a_round_trip(provenance in arbitrary_tree()) {
        let json = serde_json::to_string(&provenance).expect("provenances encode");
        let restored: Provenance = serde_json::from_str(&json).expect("encoded provenances decode");

        let re_encoded = serde_json::to_string(&restored).expect("provenances encode");

        prop_assert_eq!(re_encoded, json);
    }

    /// Test positions order exactly as their `(line, column)` pairs do.
    #[test]
    fn position_order_matches_line_column_pair_order(
        left in (1_u64..=u64::MAX, 1_u64..=u64::MAX),
        right in (1_u64..=u64::MAX, 1_u64..=u64::MAX),
    ) {
        let left_position = Position::try_new(left.0, left.1).expect("non-zero");
        let right_position = Position::try_new(right.0, right.1).expect("non-zero");

        prop_assert_eq!(left_position.cmp(&right_position), left.cmp(&right));
    }
}

/// Return a strategy for text of any characters, newlines included, whose
/// length in characters is in `lengths`.
fn arbitrary_text(lengths: std::ops::RangeInclusive<usize>) -> impl Strategy<Value = String> {
    proptest::collection::vec(any::<char>(), lengths).prop_map(String::from_iter)
}

/// Return a strategy for one diagnostic with a random level, message,
/// source and detail.
fn arbitrary_diagnostic() -> impl Strategy<Value = Diagnostic> {
    (
        select(
            &[
                DiagnosticLevel::Error,
                DiagnosticLevel::Warning,
                DiagnosticLevel::Info,
            ][..],
        ),
        arbitrary_text(0..=20),
        arbitrary_text(1..=10),
        proptest::option::of(arbitrary_text(0..=20)),
    )
        .prop_map(|(level, message, source, detail)| {
            Diagnostic::new(level, Note::with_other_kind(message), source, detail)
        })
}

/// Return a strategy for a report with 0 to 8 diagnostics and 0 to 4
/// records.
fn arbitrary_report() -> impl Strategy<Value = ValidationReport<u32>> {
    (
        proptest::collection::vec(arbitrary_diagnostic(), 0..=8),
        proptest::collection::vec(any::<u32>(), 0..=4),
    )
        .prop_map(|(diagnostics, records)| ValidationReport::new(diagnostics, records))
}

/// Return the rendering `ValidationReport::format` documents for
/// `diagnostics`: a placeholder for none, and otherwise one `[LEVEL]
/// source: message` line per diagnostic, followed by an indented detail line
/// when the detail is present and non-empty, joined by newlines.
fn render_report(diagnostics: &[Diagnostic]) -> String {
    if diagnostics.is_empty() {
        return "No validation diagnostics.".to_owned();
    }
    let mut text = String::new();
    for (index, diagnostic) in diagnostics.iter().enumerate() {
        if index > 0 {
            text.push('\n');
        }
        let level = match diagnostic.level() {
            DiagnosticLevel::Error => "ERROR",
            DiagnosticLevel::Warning => "WARNING",
            DiagnosticLevel::Info => "INFO",
        };
        write!(
            text,
            "[{level}] {}: {}",
            diagnostic.source(),
            diagnostic.message().message()
        )
        .expect("writing to a string succeeds");
        match diagnostic.detail() {
            Some(detail) if !detail.is_empty() => {
                write!(text, "\n    detail: {detail}").expect("writing to a string succeeds");
            }
            _ => {}
        }
    }
    text
}

/// Return the diagnostics of `diagnostics` at `level`, in order.
fn select_level(diagnostics: &[Diagnostic], level: DiagnosticLevel) -> Vec<&Diagnostic> {
    diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.level() == level)
        .collect()
}

proptest! {
    /// Test `has_errors` holds exactly when some diagnostic is an error.
    #[test]
    fn report_has_errors_matches_any_error_level_diagnostic(report in arbitrary_report()) {
        let expected = report
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.level() == DiagnosticLevel::Error);

        prop_assert_eq!(report.has_errors(), expected);
    }

    /// Test the level filters partition the diagnostics, each keeping
    /// emission order.
    #[test]
    fn report_level_filters_partition_the_diagnostics_in_order(report in arbitrary_report()) {
        let errors: Vec<&Diagnostic> = report.errors().collect();
        let warnings: Vec<&Diagnostic> = report.warnings().collect();
        let infos: Vec<&Diagnostic> = report.infos().collect();

        prop_assert_eq!(&errors, &select_level(report.diagnostics(), DiagnosticLevel::Error));
        prop_assert_eq!(&warnings, &select_level(report.diagnostics(), DiagnosticLevel::Warning));
        prop_assert_eq!(&infos, &select_level(report.diagnostics(), DiagnosticLevel::Info));
        prop_assert_eq!(
            errors.len() + warnings.len() + infos.len(),
            report.diagnostics().len()
        );
    }

    /// Test a report renders as its documented text, built here diagnostic
    /// by diagnostic.
    #[test]
    fn report_format_writes_the_documented_text(report in arbitrary_report()) {
        let text = report.format();

        prop_assert_eq!(text, render_report(report.diagnostics()));
    }

    /// Test `into_result` fails exactly when the report has errors, and the
    /// failure owns an equal report, records included, whose text it
    /// renders.
    #[test]
    fn report_into_result_fails_iff_it_has_errors(report in arbitrary_report()) {
        let has_errors = report.has_errors();
        let expected = report.clone();

        let result = report.into_result();

        match result {
            Ok(returned) => {
                prop_assert!(!has_errors);
                prop_assert_eq!(returned, expected);
            }
            Err(error) => {
                prop_assert!(has_errors);
                prop_assert_eq!(error.to_string(), expected.format());
                prop_assert_eq!(error.into_report(), expected);
            }
        }
    }
}
