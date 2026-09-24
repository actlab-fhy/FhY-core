//! Tests that every public `Serialize` type round-trips through JSON text
//! and through postcard, a format that is not self-describing.
//!
//! A decode that fails partway may leave the identifiers
//! and canonical values it already read behind; the stories at the end pin
//! that documented partial effect.

use std::fmt::Debug;

use crate::support::expression as expression_support;
use crate::support::provenance as provenance_support;
use crate::support::stack as stack_support;

use expression_support::{
    build_callee, build_decimal_literal, build_deep_sum, build_doubling_dag, build_identifier,
    build_literal,
};
use fhy_core::diagnostic::{Note, NoteKind};
use fhy_core::expr::builtins::{BuiltinConstant, BuiltinFunction};
use fhy_core::expr::{
    BigInt, BinaryOperation, Callee, Decimal, Expression, ExpressionKind, FunctionName,
    FunctionSort, LiteralValue, LogicalOperation, SymbolType, UnaryOperation,
};
use fhy_core::identifier::Identifier;
use fhy_core::interned::{Canonical, Interned};
use fhy_core::op_attribute::OpAttribute;
use fhy_core::provenance::{CallSiteProvenance, FusedProvenance, Position, Provenance, Span};
use fhy_core::value_domain::ValueDomain;
use provenance_support::{build_file, build_named};
use rstest::rstest;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use stack_support::run_on_stack;

// =============================================================================
// Helpers
// =============================================================================

/// Assert `value` round-trips through JSON text and through postcard, each
/// decode equal to `value`.
fn assert_round_trips<T: Serialize + DeserializeOwned + PartialEq + Debug>(value: &T) {
    let json = serde_json::to_string(value).expect("the value encodes as JSON");
    let from_json: T = serde_json::from_str(&json).expect("the JSON text decodes");
    assert_eq!(&from_json, value, "the JSON round trip of {json}");

    let bytes = postcard::to_allocvec(value).expect("the value encodes as postcard");
    let from_postcard: T = postcard::from_bytes(&bytes).expect("the postcard bytes decode");
    assert_eq!(
        &from_postcard, value,
        "the postcard round trip of {value:?}"
    );
}

fn assert_each_round_trips<T: Serialize + DeserializeOwned + PartialEq + Debug>(
    values: impl IntoIterator<Item = T>,
) {
    for value in values {
        assert_round_trips(&value);
    }
}

/// Assert every strict prefix of `bytes` fails to decode as a `T`.
fn assert_every_prefix_fails<T: DeserializeOwned + Debug>(bytes: &[u8]) {
    for length in 0..bytes.len() {
        let result = postcard::from_bytes::<T>(&bytes[..length]);
        assert!(
            result.is_err(),
            "a {length}-byte prefix decoded: {result:?}"
        );
    }
}

/// Build the position at `line` and `column`, which must both be non-zero.
fn build_position(line: u64, column: u64) -> Position {
    Position::try_new(line, column).expect("line and column are non-zero")
}

/// Build a provenance using every variant: a labelled fusion of a named
/// call-site chain, an unlabelled fusion and an unknown provenance.
fn build_nested_provenance() -> Provenance {
    let span = Span::from_offsets(0..3).expect("ordered offsets are valid");
    let call_site = Provenance::CallSite(CallSiteProvenance::new(
        build_named("inlined", build_file("callee.fhy", Some(span))),
        build_file("caller.fhy", None),
    ));
    Provenance::Fused(FusedProvenance::labelled(
        vec![
            build_named("chain", call_site),
            Provenance::Fused(FusedProvenance::new(vec![build_file("a.fhy", None)])),
            Provenance::Unknown,
        ],
        "cse",
    ))
}

// =============================================================================
// Diagnostic and provenance types
// =============================================================================

type ShippedKind = fn() -> &'static Canonical<NoteKind>;

/// Test a note of each shipped kind round-trips through both formats and
/// decodes to the shipped handle.
#[rstest]
#[case::rationale(NoteKind::rationale)]
#[case::suggestion(NoteKind::suggestion)]
#[case::remark(NoteKind::remark)]
#[case::other(NoteKind::other)]
fn note_round_trips_through_postcard(#[case] get_kind: ShippedKind) {
    let note = Note::new("a message", get_kind().clone());

    assert_round_trips(&note);

    let bytes = postcard::to_allocvec(&note).expect("the note encodes");
    let restored: Note = postcard::from_bytes(&bytes).expect("the note decodes");
    assert!(Canonical::ptr_eq(restored.kind(), get_kind()));
}

/// Test a note tagged with a caller's kind round-trips through both formats.
#[test]
fn note_with_a_custom_kind_round_trips_through_postcard() {
    let kind = NoteKind::register(Identifier::new("postcard-kind"), "A custom kind.");

    assert_round_trips(&Note::new("custom", kind));
}

/// Test a position round-trips through both formats, at the smallest and
/// largest lines and columns.
#[rstest]
#[case::smallest(1, 1)]
#[case::typical(7, 3)]
#[case::largest(u64::MAX, u64::MAX)]
fn position_round_trips_through_postcard(#[case] line: u64, #[case] column: u64) {
    assert_round_trips(&build_position(line, column));
}

/// Test spans with every mix of bounds round-trip through both formats.
#[rstest]
#[case::unknown(Span::unknown())]
#[case::offsets_only(Span::from_offsets(0..3).unwrap())]
#[case::positions_only(Span::from_positions(build_position(1, 1)..build_position(2, 4)).unwrap())]
#[case::start_offset_only(Span::unknown().with_start_offset(5).unwrap())]
#[case::end_offset_only(Span::unknown().with_end_offset(5).unwrap())]
#[case::start_position_only(Span::unknown().with_start_position(build_position(3, 1)).unwrap())]
#[case::end_position_only(Span::unknown().with_end_position(build_position(3, 1)).unwrap())]
#[case::both_pairs(
    Span::from_offsets(0..3)
        .and_then(|span| span.with_positions(build_position(1, 1)..build_position(1, 4)))
        .unwrap()
)]
fn span_round_trips_through_postcard(#[case] span: Span) {
    assert_round_trips(&span);
}

/// Test each provenance variant, and a tree nesting all of them,
/// round-trips through both formats.
#[rstest]
#[case::unknown(Provenance::Unknown)]
#[case::file_without_span(build_file("a.fhy", None))]
#[case::file_with_span(build_file("./src//a.fhy", Some(Span::from_offsets(2..9).unwrap())))]
#[case::named(build_named("fhy.add", Provenance::Unknown))]
#[case::call_site(Provenance::CallSite(CallSiteProvenance::new(
    build_file("callee.fhy", None),
    build_file("caller.fhy", None),
)))]
#[case::unlabelled_fusion(Provenance::Fused(FusedProvenance::new(vec![
    build_file("a.fhy", None),
    build_file("b.fhy", None),
])))]
#[case::labelled_fusion(Provenance::Fused(FusedProvenance::labelled(
    vec![build_file("a.fhy", None)],
    "loop-fusion",
)))]
#[case::empty_labelled_fusion(Provenance::Fused(FusedProvenance::labelled(vec![], "")))]
#[case::nested(build_nested_provenance())]
fn provenance_round_trips_through_postcard(#[case] provenance: Provenance) {
    assert_round_trips(&provenance);
}

// =============================================================================
// Identity and tag types
// =============================================================================

/// Test an identifier round-trips through both formats with its id.
#[test]
fn identifier_round_trips_through_postcard() {
    assert_round_trips(&Identifier::new("postcard-identifier"));
}

/// Test shipped and registered op attributes round-trip through both
/// formats.
#[test]
fn op_attribute_round_trips_through_postcard() {
    let registered = OpAttribute::register(Identifier::new("postcard-attribute"), "Custom.");

    assert_round_trips(OpAttribute::commutative());
    assert_round_trips(&registered);
}

/// Test shipped note kinds and a registered one round-trip through both
/// formats.
#[test]
fn note_kind_round_trips_through_postcard() {
    let registered = NoteKind::register(Identifier::new("postcard-note-kind"), "Custom.");

    assert_round_trips(NoteKind::remark());
    assert_round_trips(&registered);
}

/// Test a value domain with a parent chain round-trips through both
/// formats.
#[test]
fn value_domain_with_a_parent_chain_round_trips_through_postcard() {
    let tile = ValueDomain::register_child(
        Identifier::new("postcard-tile"),
        "A tile of data.",
        ValueDomain::data(),
    )
    .expect("the name is fresh");
    let subtile = ValueDomain::register_child(Identifier::new("postcard-subtile"), "Part.", &tile)
        .expect("the name is fresh");

    assert_round_trips(ValueDomain::address());
    assert_round_trips(&subtile);
}

// =============================================================================
// Expression types
// =============================================================================

/// Build an expression using every node kind, every literal kind, a
/// built-in and a named call, and a subtree shared by two parents.
fn build_every_node_kind() -> Expression {
    let (_, x) = build_identifier("postcard-x");
    let big: BigInt = "-123456789012345678901234567890".parse().expect("digits");
    let shared = &x + big;
    let call = Expression::call(
        BuiltinFunction::Max,
        [
            shared.clone(),
            Expression::call(build_callee("softplus"), [build_literal(f64::NAN)]),
        ],
    );
    Expression::piecewise(
        [
            (x.less(build_decimal_literal("1.50")), -&shared),
            (Expression::any([!x.equals(0), build_literal(true)]), call),
        ],
        build_literal(u64::MAX).floor_mod(2.5),
    )
    .expect("a valid piecewise")
}

/// Test an expression using every node kind round-trips through both
/// formats.
#[test]
fn expression_of_every_node_kind_round_trips_through_postcard() {
    assert_round_trips(&build_every_node_kind());
}

/// Test every literal kind, big integers beyond both `u64` and `i64::MIN`
/// and the non-finite floats included, round-trips through both formats.
#[rstest]
#[case::boolean(LiteralValue::from(true))]
#[case::integer(LiteralValue::from(-7))]
#[case::above_u64(LiteralValue::from("18446744073709551616".parse::<BigInt>().unwrap()))]
#[case::below_i64(LiteralValue::from("-9223372036854775809".parse::<BigInt>().unwrap()))]
#[case::float(LiteralValue::from(0.1))]
#[case::negative_zero(LiteralValue::from(-0.0))]
#[case::infinity(LiteralValue::from(f64::INFINITY))]
#[case::nan(LiteralValue::from(f64::NAN))]
#[case::decimal(LiteralValue::parse_text("0.001").unwrap())]
fn literal_value_round_trips_through_postcard(#[case] literal: LiteralValue) {
    assert_round_trips(&literal);
    assert_round_trips(&Expression::from(literal));
}

/// Test a decimal, a callee of each kind and a function name round-trip
/// through both formats.
#[test]
fn decimal_callee_and_function_name_round_trip_through_postcard() {
    let decimal: Decimal = "12.3400".parse().expect("a decimal text");
    let name = FunctionName::try_new("softplus").expect("a user function name");

    assert_round_trips(&decimal);
    assert_round_trips(&name);
    assert_round_trips(&Callee::Named(name));
    assert_round_trips(&Callee::Builtin(BuiltinFunction::ClampSymmetric));
}

/// Test every variant of every vocabulary enum round-trips through both
/// formats.
#[test]
fn vocabulary_enums_round_trip_through_postcard() {
    assert_each_round_trips([SymbolType::Real, SymbolType::Int, SymbolType::Bool]);
    assert_each_round_trips([
        FunctionSort::Bool,
        FunctionSort::Nat,
        FunctionSort::Int,
        FunctionSort::Real,
    ]);
    assert_each_round_trips([
        UnaryOperation::Negate,
        UnaryOperation::Positive,
        UnaryOperation::LogicalNot,
    ]);
    assert_each_round_trips([
        BinaryOperation::Add,
        BinaryOperation::FloorMod,
        BinaryOperation::GreaterEqual,
    ]);
    assert_each_round_trips([LogicalOperation::And, LogicalOperation::Or]);
    assert_each_round_trips(BuiltinFunction::iter());
    assert_each_round_trips(BuiltinConstant::iter());
}

/// Test a DAG keeps its sharing through both formats: the subtree shared
/// by two parents decodes as one node.
#[test]
fn expression_dag_keeps_its_sharing_through_postcard() {
    let (_, x) = build_identifier("postcard-shared");
    let shared = &x * 2;
    let dag = Expression::all([shared.less(1), shared.greater(0)]);

    let text = serde_json::to_string(&dag).expect("the DAG encodes as JSON");
    let from_json: Expression = serde_json::from_str(&text).expect("the JSON text decodes");
    let bytes = postcard::to_allocvec(&dag).expect("the DAG encodes as postcard");
    let from_postcard: Expression = postcard::from_bytes(&bytes).expect("the bytes decode");

    for restored in [&from_json, &from_postcard] {
        let ExpressionKind::Logical(conjunction) = restored.kind() else {
            panic!("the root is a conjunction");
        };
        let [
            ExpressionKind::Binary(less),
            ExpressionKind::Binary(greater),
        ] = [
            conjunction.operands()[0].kind(),
            conjunction.operands()[1].kind(),
        ]
        else {
            panic!("both operands are comparisons");
        };
        assert!(Expression::ptr_eq(less.left(), greater.left()));
        assert_eq!(restored, &dag);
    }
}

/// Test a sum 100,000 levels deep round-trips through JSON text and
/// postcard on a 256 KiB stack: neither direction recurses per level.
#[test]
fn expression_deep_sum_round_trips_through_json_text_on_a_small_stack() {
    run_on_stack(256 << 10, || {
        let (_, x) = build_identifier("postcard-deep");
        let tree = build_deep_sum(&x, 100_000);

        assert_round_trips(&tree);
    });
}

/// Test a conjunction of ten thousand comparisons, one logical node,
/// round-trips through JSON text and postcard.
#[test]
fn expression_conjunction_of_ten_thousand_comparisons_round_trips_through_json_text() {
    let (_, x) = build_identifier("postcard-bounded");
    let conjunction = Expression::all((0..10_000).map(|bound| x.less(bound)));

    assert_round_trips(&conjunction);
}

/// Test a doubling DAG 64 levels deep, with more than `2^64` occurrences,
/// encodes as its 65 distinct nodes and decodes sharing both operands of
/// every level, in JSON and postcard.
#[test]
fn expression_wire_encodes_a_doubling_dag_once_per_distinct_node() {
    const LEVELS: usize = 64;
    let (_, x) = build_identifier("postcard-doubling");
    let dag = build_doubling_dag(&x, LEVELS);

    let wire = serde_json::to_value(&dag).expect("the DAG encodes as JSON");
    let from_json: Expression = serde_json::from_value(wire.clone()).expect("the table decodes");
    let bytes = postcard::to_allocvec(&dag).expect("the DAG encodes as postcard");
    let from_postcard: Expression = postcard::from_bytes(&bytes).expect("the bytes decode");

    assert_eq!(wire["nodes"].as_array().map(Vec::len), Some(LEVELS + 1));
    for restored in [&from_json, &from_postcard] {
        let mut node = restored;
        for level in 0..LEVELS {
            let ExpressionKind::Binary(sum) = node.kind() else {
                panic!("level {level} is not a sum");
            };
            assert!(
                Expression::ptr_eq(sum.left(), sum.right()),
                "level {level} does not share its operands"
            );
            node = sum.left();
        }
        assert_eq!(node, &x);
        assert_eq!(restored, &dag);
    }
}

/// Test a big integer literal serializes in JSON as the decimal string of
/// its digits.
#[test]
fn a_big_integer_literal_serializes_as_a_decimal_string_in_json() {
    let big: BigInt = "1000000000000000000000000000000".parse().expect("digits");

    let text = serde_json::to_string(&LiteralValue::from(big)).expect("the literal encodes");

    assert_eq!(text, r#"{"int":"1000000000000000000000000000000"}"#);
}

// =============================================================================
// Builds that include fhy-core
// =============================================================================

/// Test JSON numbers compare by value in a build that includes this crate:
/// no `serde_json` feature leaks from it that keeps a number's text.
#[test]
fn serde_json_numbers_compare_by_value_in_a_build_with_fhy_core() {
    let short: Value = serde_json::from_str("1.0").expect("a JSON number");
    let long: Value = serde_json::from_str("1.00").expect("a JSON number");

    assert_eq!(short, long);
}

/// A dependent crate's untagged number, which a buffering decoder reads.
#[derive(Debug, PartialEq, Deserialize)]
#[serde(untagged)]
enum UntaggedNumber {
    Float(f64),
}

/// Test a dependent crate's untagged float decodes in a build that
/// includes this crate: no `serde_json` feature leaks from it that turns a
/// buffered number into a map.
#[test]
fn a_dependent_untagged_float_decodes_in_a_build_with_fhy_core() {
    let decoded: UntaggedNumber = serde_json::from_str("1.5").expect("an untagged float");

    assert_eq!(decoded, UntaggedNumber::Float(1.5));
}

// =============================================================================
// Adversarial input
// =============================================================================

/// Test every strict prefix of an encoded nested provenance, of an encoded
/// note and of an encoded expression decodes to an error rather than a
/// panic.
#[test]
fn truncated_postcard_bytes_are_an_error_not_a_panic() {
    let provenance = postcard::to_allocvec(&build_nested_provenance()).expect("encodes");
    let note = postcard::to_allocvec(&Note::with_other_kind("truncated")).expect("encodes");
    let expression = postcard::to_allocvec(&build_every_node_kind()).expect("encodes");

    assert_every_prefix_fails::<Provenance>(&provenance);
    assert_every_prefix_fails::<Note>(&note);
    assert_every_prefix_fails::<Expression>(&expression);
}

/// Test a note decode that fails after reading its kind leaves that kind
/// registered, and a retry with a valid message returns the same canonical
/// kind.
#[test]
fn a_note_decode_that_fails_after_its_kind_leaves_the_kind_registered() {
    let name = Identifier::new("partial-note-kind");
    let kind = format!(
        r#"{{"name": {{"id": {}, "name_hint": "partial-note-kind"}}, "description": "d"}}"#,
        name.id()
    );

    let failure = serde_json::from_str::<Note>(&format!(r#"{{"kind": {kind}, "message": 5}}"#));
    let registered = NoteKind::intern_registry().get(&name);
    let retry: Note = serde_json::from_str(&format!(r#"{{"kind": {kind}, "message": "m"}}"#))
        .expect("the retry decodes");

    assert!(failure.is_err(), "a numeric message is refused");
    let registered = registered.expect("the failed decode registered the kind");
    assert!(Canonical::ptr_eq(retry.kind(), &registered));
}
