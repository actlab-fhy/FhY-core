//! Tests for captures, patterns and match bindings: every pattern shape, the
//! order bindings are threaded and recorded in, repeated captures, literal
//! equality, predicates and their failures, alternatives and the binding
//! trail, the matching methods, callback errors, and patterns thousands of
//! levels deep.
//!
//! Public API only (`fhy_core::expr::pattern`).

use crate::support::expression as expression_support;
use crate::support::hashing as hashing_support;
use crate::support::pattern as pattern_support;
use crate::support::stack as stack_support;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use expression_support::{
    build_call_or_panic, build_callee, build_decimal_literal, build_deep_sum, build_identifier,
    build_literal,
};
use fhy_core::expr::builtins::BuiltinFunction;
use fhy_core::expr::pattern::{CallbackError, Capture, MatchBindings, Pattern};
use fhy_core::expr::{
    BigInt, BinaryOperation, Expression, ExpressionKind, FunctionName, FunctionNameError,
    LiteralTextError, LiteralValue, LogicalOperation, PiecewiseError, RebuildError, UnaryOperation,
};
use hashing_support::hash_of;
use pattern_support::{ProbeError, expect_probe_error};
use rstest::rstest;
use stack_support::{SMALL_STACK_DEPTH, run_on_small_stack, run_on_stack};

/// Depth of the deep trees and patterns the operations documented as
/// recursive are run over, on a stack sized for that recursion.
const DEEP_TREE_DEPTH: usize = 4000;

/// Stack size for matching a pattern [`DEEP_TREE_DEPTH`] levels deep, which
/// recurses once per pattern level.
const PATTERN_MATCH_STACK_BYTES: usize = 16 << 20;

/// Match `pattern` against `expression` and return the result, failing the
/// test if a predicate fails.
///
/// # Panics
///
/// Panics if a predicate in `pattern` fails.
#[must_use]
fn match_infallibly(pattern: &Pattern, expression: &Expression) -> Option<MatchBindings> {
    pattern.matches(expression).expect("no predicate fails")
}

/// Match `pattern` against `expression` and return the bindings, failing the
/// test if it does not match.
///
/// # Panics
///
/// Panics if a predicate fails or `pattern` does not match.
#[must_use]
fn expect_match(pattern: &Pattern, expression: &Expression) -> MatchBindings {
    match_infallibly(pattern, expression)
        .unwrap_or_else(|| panic!("{pattern:?} does not match {expression:?}"))
}

/// Return the literal expression `LiteralValue::parse_text` reads from
/// `text`.
fn build_parsed_literal(text: &str) -> Expression {
    build_literal(LiteralValue::parse_text(text).expect("the text is a literal text"))
}

/// Return the captures `bindings` binds, in binding order.
fn collect_captures(bindings: &MatchBindings) -> Vec<&Capture> {
    bindings.iter().map(|(capture, _)| capture).collect()
}

/// Return `1 op 2`.
fn build_simple_binary(operation: BinaryOperation) -> Expression {
    Expression::new_binary(operation, build_literal(1), build_literal(2))
}

/// Return the logical node `op` over the comparisons `x < 1`, `x < 2`, ...,
/// `x < count`, of `count` operands.
fn build_simple_logical(operation: LogicalOperation, count: i64) -> Expression {
    let (_, x) = build_identifier("x");
    Expression::new_logical(operation, (1..=count).map(|bound| x.less(bound)))
}

/// Return a one-case piecewise `true -> 1, otherwise 2`.
fn build_one_case_piecewise() -> Expression {
    Expression::piecewise([(build_literal(true), build_literal(1))], build_literal(2))
        .expect("a valid piecewise")
}

/// Return a two-case piecewise `true -> 1, false -> 2, otherwise 3`.
fn build_two_case_piecewise() -> Expression {
    Expression::piecewise(
        [
            (build_literal(true), build_literal(1)),
            (build_literal(false), build_literal(2)),
        ],
        build_literal(3),
    )
    .expect("a valid piecewise")
}

/// A kind of expression node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeKind {
    Literal,
    Identifier,
    Unary,
    Binary,
    Logical,
    Piecewise,
    Call,
}

impl NodeKind {
    /// Return an expression whose root is a node of this kind.
    fn build(self) -> Expression {
        let (_, x) = build_identifier("x");
        match self {
            Self::Literal => build_literal(5),
            Self::Identifier => x,
            Self::Unary => -&x,
            Self::Binary => build_simple_binary(BinaryOperation::Add),
            Self::Logical => build_simple_logical(LogicalOperation::And, 2),
            Self::Piecewise => build_one_case_piecewise(),
            Self::Call => build_call_or_panic("f", vec![x]),
        }
    }
}

/// Return a predicate pattern counting its calls in `calls` and answering
/// `verdict`.
fn build_counting_predicate(calls: &Arc<AtomicUsize>, verdict: bool) -> Pattern {
    let calls = Arc::clone(calls);
    Pattern::predicate(move |_| {
        calls.fetch_add(1, Ordering::SeqCst);
        verdict
    })
}

/// Return a predicate pattern failing with the [`ProbeError`] `message`.
fn build_failing_predicate(message: &'static str) -> Pattern {
    Pattern::try_predicate(move |_| Err(CallbackError::from(ProbeError(message))))
}

/// Return the pattern matching `build_deep_sum(leaf, depth)` exactly, with
/// the leaf matched by `leaf_pattern`.
fn build_deep_sum_pattern(leaf_pattern: Pattern, depth: usize) -> Pattern {
    let mut pattern = leaf_pattern;
    for _ in 0..depth {
        pattern = Pattern::binary(BinaryOperation::Add, pattern, Pattern::literal(1));
    }
    pattern
}

// =============================================================================
// Capture
// =============================================================================

/// Test a capture equals its clones and nothing else, and hashes like its
/// clones.
#[test]
fn capture_equals_exactly_its_clones() {
    let capture = Capture::new("x");
    let clone = capture.clone();
    let same_name = Capture::new("x");

    assert_eq!(capture, clone);
    assert_eq!(hash_of(&capture), hash_of(&clone));
    assert_ne!(capture, same_name);
}

/// Test a capture keeps the name it was built with, and displays as it.
#[rstest]
#[case::plain("x")]
#[case::empty("")]
fn capture_name_is_the_given_name(#[case] name: &str) {
    let capture = Capture::new(name);

    assert_eq!(capture.name(), name);
    assert_eq!(capture.to_string(), name);
}

// =============================================================================
// MatchBindings
// =============================================================================

/// Test new bindings bind no capture, like the default ones.
#[test]
fn match_bindings_new_binds_no_capture() {
    let x = Capture::new("x");

    let bindings = MatchBindings::new();

    assert!(bindings.is_empty());
    assert_eq!(bindings.len(), 0);
    assert_eq!(bindings.get(&x), None);
    assert_eq!(collect_captures(&bindings), Vec::<&Capture>::new());
    assert_eq!(MatchBindings::default(), bindings);
}

/// Test `get` returns `None` for a capture the matched pattern lacks.
#[test]
fn match_bindings_get_returns_none_for_a_capture_the_pattern_lacks() {
    let (x, y) = (Capture::new("x"), Capture::new("y"));
    let bindings = expect_match(&Pattern::capture(&y), &build_literal(0));

    let bound = bindings.get(&x);

    assert!(bound.is_none(), "got {bound:?}");
}

/// Test `contains` reports exactly the bound captures.
#[test]
fn match_bindings_contains_reports_only_bound_captures() {
    let (x, y) = (Capture::new("x"), Capture::new("y"));

    let bindings = expect_match(&Pattern::capture(&x), &build_literal(0));

    assert!(bindings.contains(&x));
    assert!(!bindings.contains(&y));
    assert!(!MatchBindings::new().contains(&x));
    assert_eq!(bindings.len(), 1);
}

/// Test `iter` lists every bound capture once, in binding order, with the
/// expression it binds.
#[test]
fn match_bindings_iter_lists_captures_in_binding_order() {
    let (x, y) = (Capture::new("x"), Capture::new("y"));
    let pattern = Pattern::binary_any_operation(
        Pattern::binary_any_operation(Pattern::capture(&y), Pattern::capture(&x)),
        Pattern::capture(&y),
    );
    let expression = build_simple_binary(BinaryOperation::Add) + 1;

    let bindings = expect_match(&pattern, &expression);

    let entries: Vec<(&Capture, &Expression)> = bindings.iter().collect();
    assert_eq!(entries, [(&y, &build_literal(1)), (&x, &build_literal(2))]);
}

/// Test indexing by a capture gives the expression bound to it.
#[test]
fn match_bindings_index_returns_the_bound_expression() {
    let x = Capture::new("x");
    let expression = build_literal(5);

    let bindings = expect_match(&Pattern::capture(&x), &expression);

    assert!(Expression::ptr_eq(&bindings[&x], &expression));
}

/// Test indexing by a capture the bindings do not bind panics.
#[test]
#[should_panic(expected = "capture `x` is not bound")]
fn match_bindings_index_panics_for_an_unbound_capture() {
    let x = Capture::new("x");
    let bindings = MatchBindings::new();

    let _ = &bindings[&x];
}

/// Test bindings of one capture to equal expressions are equal and hash
/// equally.
#[test]
fn match_bindings_with_equal_content_are_equal_and_hash_equally() {
    let x = Capture::new("x");

    let left = expect_match(&Pattern::capture(&x), &build_literal(7));
    let right = expect_match(&Pattern::capture(&x), &build_literal(7));

    assert_eq!(left, right);
    assert_eq!(hash_of(&left), hash_of(&right));
}

/// Test equality compares bound expressions structurally, with literal
/// equality.
#[test]
fn match_bindings_equality_compares_expressions_structurally() {
    let x = Capture::new("x");

    let left = expect_match(&Pattern::capture(&x), &build_literal(5));
    let right = expect_match(&Pattern::capture(&x), &build_parsed_literal("05"));

    assert_eq!(left, right);
    assert_eq!(hash_of(&left), hash_of(&right));
}

/// Test bindings of a capture to different expressions are unequal.
#[test]
fn match_bindings_with_different_expressions_are_unequal() {
    let x = Capture::new("x");

    let left = expect_match(&Pattern::capture(&x), &build_literal(1));
    let right = expect_match(&Pattern::capture(&x), &build_literal(2));

    assert_ne!(left, right);
}

/// Test equality ignores binding order, and so does hashing: `x` and `y`
/// bound to `1` and `2` in either order.
#[test]
fn match_bindings_equality_ignores_binding_order() {
    let (x, y) = (Capture::new("x"), Capture::new("y"));
    let x_first = Pattern::binary(
        BinaryOperation::Add,
        Pattern::capture(&x),
        Pattern::capture(&y),
    );
    let y_first = Pattern::binary(
        BinaryOperation::Subtract,
        Pattern::capture(&y),
        Pattern::capture(&x),
    );

    let x_first = expect_match(&x_first, &build_simple_binary(BinaryOperation::Add));
    let y_first = expect_match(
        &y_first,
        &Expression::new_binary(
            BinaryOperation::Subtract,
            build_literal(2),
            build_literal(1),
        ),
    );

    assert_ne!(collect_captures(&x_first), collect_captures(&y_first));
    assert_eq!(x_first, y_first);
    assert_eq!(hash_of(&x_first), hash_of(&y_first));
}

/// Return bindings of `x` and of `y`, each to `1`.
fn bind_disjoint_captures(x: &Capture, y: &Capture) -> (MatchBindings, MatchBindings) {
    (
        expect_match(&Pattern::capture(x), &build_literal(1)),
        expect_match(&Pattern::capture(y), &build_literal(1)),
    )
}

/// Return bindings of `x` to `1`, and of `x` to `1` and `y` to `2`.
fn bind_one_capture_more(x: &Capture, y: &Capture) -> (MatchBindings, MatchBindings) {
    let both = Pattern::binary_any_operation(Pattern::capture(x), Pattern::capture(y));
    (
        expect_match(&Pattern::capture(x), &build_literal(1)),
        expect_match(&both, &build_simple_binary(BinaryOperation::Add)),
    )
}

/// Return empty bindings, and bindings of `x` to `1`.
fn bind_no_capture_and_one(x: &Capture, _: &Capture) -> (MatchBindings, MatchBindings) {
    (
        expect_match(&Pattern::wildcard(), &build_literal(1)),
        expect_match(&Pattern::capture(x), &build_literal(1)),
    )
}

/// Return bindings of `x` to `1`, and of another capture named `x` to `1`.
fn bind_captures_of_one_name(x: &Capture, _: &Capture) -> (MatchBindings, MatchBindings) {
    let other_x = Capture::new(x.name());
    (
        expect_match(&Pattern::capture(x), &build_literal(1)),
        expect_match(&Pattern::capture(&other_x), &build_literal(1)),
    )
}

/// Test bindings of different capture sets are unequal, even when two
/// captures share a name.
#[rstest]
#[case::disjoint_captures(bind_disjoint_captures)]
#[case::one_capture_more(bind_one_capture_more)]
#[case::empty_and_bound(bind_no_capture_and_one)]
#[case::captures_of_one_name(bind_captures_of_one_name)]
fn match_bindings_with_different_captures_are_unequal(
    #[case] bind: fn(&Capture, &Capture) -> (MatchBindings, MatchBindings),
) {
    let (x, y) = (Capture::new("x"), Capture::new("y"));

    let (left, right) = bind(&x, &y);

    assert_ne!(left, right);
    assert_ne!(right, left);
}

// =============================================================================
// Wildcard and nothing
// =============================================================================

/// Test the wildcard matches a node of every kind and captures nothing.
#[rstest]
fn pattern_wildcard_matches_every_node_kind(
    #[values(
        NodeKind::Literal,
        NodeKind::Identifier,
        NodeKind::Unary,
        NodeKind::Binary,
        NodeKind::Logical,
        NodeKind::Piecewise,
        NodeKind::Call
    )]
    kind: NodeKind,
) {
    let result = match_infallibly(&Pattern::wildcard(), &kind.build());

    assert_eq!(result, Some(MatchBindings::new()));
}

/// Test `nothing` matches no node of any kind.
#[rstest]
fn pattern_nothing_matches_no_expression(
    #[values(
        NodeKind::Literal,
        NodeKind::Identifier,
        NodeKind::Unary,
        NodeKind::Binary,
        NodeKind::Logical,
        NodeKind::Piecewise,
        NodeKind::Call
    )]
    kind: NodeKind,
) {
    let result = match_infallibly(&Pattern::nothing(), &kind.build());

    assert!(result.is_none(), "got {result:?}");
}

// =============================================================================
// Capture patterns
// =============================================================================

/// Test a capture binds a handle to the matched node.
#[test]
fn pattern_capture_binds_the_matched_node() {
    let x = Capture::new("x");
    let expression = build_simple_binary(BinaryOperation::Add);

    let bindings = expect_match(&Pattern::capture(&x), &expression);

    assert!(Expression::ptr_eq(&bindings[&x], &expression));
}

/// Test a capture binds exactly its own handle.
#[test]
fn pattern_capture_binds_exactly_its_handle() {
    let x = Capture::new("x");

    let bindings = expect_match(&Pattern::capture(&x), &build_literal(5));

    assert_eq!(collect_captures(&bindings), [&x]);
}

/// Test a capture with an empty name builds and binds.
#[test]
fn pattern_capture_accepts_an_empty_name() {
    let unnamed = Capture::new("");
    let expression = build_literal(5);

    let bindings = expect_match(&Pattern::capture(&unnamed), &expression);

    assert!(Expression::ptr_eq(&bindings[&unnamed], &expression));
}

/// Test two captures built with the same name are independent: they bind
/// different operands.
#[test]
fn pattern_captures_with_the_same_name_are_independent() {
    let (first, second) = (Capture::new("x"), Capture::new("x"));
    let pattern = Pattern::binary(
        BinaryOperation::Subtract,
        Pattern::capture(&first),
        Pattern::capture(&second),
    );

    let bindings = expect_match(&pattern, &build_simple_binary(BinaryOperation::Subtract));

    assert_eq!(bindings[&first], build_literal(1));
    assert_eq!(bindings[&second], build_literal(2));
}

/// Test one capture in two sibling positions binds once.
#[test]
fn pattern_capture_shared_by_siblings_binds_once() {
    let x = Capture::new("x");
    let pattern = Pattern::binary(
        BinaryOperation::Subtract,
        Pattern::capture(&x),
        Pattern::capture(&x),
    );
    let left = build_literal(5);
    let expression = Expression::new_binary(BinaryOperation::Subtract, &left, build_literal(5));

    let bindings = expect_match(&pattern, &expression);

    assert_eq!(collect_captures(&bindings), [&x]);
    assert!(Expression::ptr_eq(&bindings[&x], &left));
}

/// Test `captured_as` fails when its pattern fails.
#[test]
fn pattern_captured_as_fails_when_the_pattern_fails() {
    let x = Capture::new("x");
    let pattern = Pattern::literal(5).captured_as(&x);

    let result = match_infallibly(&pattern, &build_literal(6));

    assert!(result.is_none(), "got {result:?}");
}

/// Test `captured_as` binds when its pattern matches.
#[test]
fn pattern_captured_as_binds_when_the_pattern_matches() {
    let x = Capture::new("x");
    let pattern = Pattern::literal(5).captured_as(&x);
    let expression = build_literal(5);

    let bindings = expect_match(&pattern, &expression);

    assert!(Expression::ptr_eq(&bindings[&x], &expression));
}

/// Test a capture repeated over equal operands matches.
#[test]
fn pattern_capture_repeated_over_equal_operands_matches() {
    let x = Capture::new("x");
    let pattern = Pattern::binary(
        BinaryOperation::Subtract,
        Pattern::capture(&x),
        Pattern::capture(&x),
    );
    let expression = Expression::new_binary(
        BinaryOperation::Subtract,
        build_literal(5),
        build_literal(5),
    );

    let bindings = expect_match(&pattern, &expression);

    assert_eq!(bindings[&x], build_literal(5));
}

/// Test a capture repeated over different operands fails.
#[test]
fn pattern_capture_repeated_over_different_operands_fails() {
    let x = Capture::new("x");
    let pattern = Pattern::binary(
        BinaryOperation::Subtract,
        Pattern::capture(&x),
        Pattern::capture(&x),
    );
    let expression = Expression::new_binary(
        BinaryOperation::Subtract,
        build_literal(5),
        build_literal(6),
    );

    let result = match_infallibly(&pattern, &expression);

    assert!(result.is_none(), "got {result:?}");
}

/// Test a capture repeated over a structurally equal compound matches and
/// keeps the first operand.
#[test]
fn pattern_capture_repeated_over_equal_compounds_keeps_the_first() {
    let x = Capture::new("x");
    let pattern = Pattern::binary_any_operation(Pattern::capture(&x), Pattern::capture(&x));
    let left = build_simple_binary(BinaryOperation::Add);
    let expression = Expression::new_binary(
        BinaryOperation::Subtract,
        &left,
        build_simple_binary(BinaryOperation::Add),
    );

    let bindings = expect_match(&pattern, &expression);

    assert!(Expression::ptr_eq(&bindings[&x], &left));
}

/// Test a repeated capture over equal literals matches and keeps the first
/// operand: `5` minus the integer parsed from `"05"`, `NaN - NaN`, and
/// `0.0 - -0.0`.
#[rstest]
#[case::integer_and_integer_text(build_literal(5), build_parsed_literal("05"))]
#[case::nan_and_nan(build_literal(f64::NAN), build_literal(f64::NAN))]
#[case::zero_and_negative_zero(build_literal(0.0), build_literal(-0.0))]
fn pattern_capture_repeated_over_equal_literals_keeps_the_first(
    #[case] left: Expression,
    #[case] right: Expression,
) {
    let x = Capture::new("x");
    let pattern = Pattern::binary_any_operation(Pattern::capture(&x), Pattern::capture(&x));
    let expression = Expression::new_binary(BinaryOperation::Subtract, &left, &right);

    let bindings = expect_match(&pattern, &expression);

    assert!(Expression::ptr_eq(&bindings[&x], &left));
}

/// Test nested captures bind the inner capture before the outer one, both
/// to the matched node.
#[test]
fn pattern_capture_nested_binds_inner_before_outer() {
    let (outer, inner) = (Capture::new("outer"), Capture::new("inner"));
    let pattern = Pattern::capture(&inner).captured_as(&outer);
    let expression = build_literal(5);

    let bindings = expect_match(&pattern, &expression);

    assert_eq!(collect_captures(&bindings), [&inner, &outer]);
    assert!(Expression::ptr_eq(&bindings[&outer], &expression));
    assert!(Expression::ptr_eq(&bindings[&inner], &expression));
}

/// Test bindings are recorded in completion order: a compound's captures in
/// matching order, then the capture around it.
#[test]
fn pattern_capture_records_bindings_in_completion_order() {
    let (outer, a, b) = (Capture::new("outer"), Capture::new("a"), Capture::new("b"));
    let pattern = Pattern::binary_any_operation(Pattern::capture(&b), Pattern::capture(&a))
        .captured_as(&outer);

    let bindings = expect_match(&pattern, &build_simple_binary(BinaryOperation::Add));

    assert_eq!(collect_captures(&bindings), [&b, &a, &outer]);
}

// =============================================================================
// Literal
// =============================================================================

/// Test the any-literal pattern matches a literal of every variant.
#[rstest]
#[case::integer(build_literal(5))]
#[case::float(build_literal(2.75))]
#[case::boolean(build_literal(true))]
#[case::integer_text(build_parsed_literal("05"))]
#[case::decimal(build_decimal_literal("1.50"))]
#[case::nan(build_literal(f64::NAN))]
fn pattern_any_literal_matches_every_literal(#[case] expression: Expression) {
    let result = match_infallibly(&Pattern::any_literal(), &expression);

    assert_eq!(result, Some(MatchBindings::new()));
}

/// Test the any-literal pattern rejects every node that is not a literal.
#[rstest]
fn pattern_literal_rejects_non_literal_nodes(
    #[values(
        NodeKind::Literal,
        NodeKind::Identifier,
        NodeKind::Unary,
        NodeKind::Binary,
        NodeKind::Logical,
        NodeKind::Piecewise,
        NodeKind::Call
    )]
    kind: NodeKind,
) {
    let result = match_infallibly(&Pattern::any_literal(), &kind.build());

    assert_eq!(
        result.is_some(),
        kind == NodeKind::Literal,
        "got {result:?}"
    );
}

/// Test a literal pattern matches every literal equal to its value, as
/// `LiteralValue` compares them.
#[rstest]
#[case::integer(LiteralValue::from(5), build_literal(5))]
#[case::big_integer(
    LiteralValue::from("1000000000000000000000000000000".parse::<BigInt>().expect("digits")),
    build_literal("1000000000000000000000000000000".parse::<BigInt>().expect("digits"))
)]
#[case::float(LiteralValue::from(2.5), build_literal(2.5))]
#[case::zero_and_negative_zero(LiteralValue::from(0.0), build_literal(-0.0))]
#[case::negative_zero_and_zero(LiteralValue::from(-0.0), build_literal(0.0))]
#[case::boolean(LiteralValue::from(false), build_literal(false))]
#[case::integer_text(
    LiteralValue::parse_text("05").expect("a text"),
    build_parsed_literal("05")
)]
#[case::decimal_text(
    LiteralValue::parse_text("1.50").expect("a text"),
    build_decimal_literal("1.50")
)]
#[case::integer_and_integer_text(LiteralValue::from(5), build_parsed_literal("5"))]
#[case::integer_text_and_integer(
    LiteralValue::parse_text("5").expect("a text"),
    build_literal(5)
)]
#[case::integer_texts_spelled_differently(
    LiteralValue::parse_text("5").expect("a text"),
    build_parsed_literal("05")
)]
#[case::decimal_texts_spelled_differently(
    LiteralValue::parse_text("1.5").expect("a text"),
    build_decimal_literal("1.50")
)]
#[case::nan_and_nan(LiteralValue::from(f64::NAN), build_literal(f64::NAN))]
fn pattern_literal_matches_an_equal_literal(
    #[case] value: LiteralValue,
    #[case] expression: Expression,
) {
    let result = match_infallibly(&Pattern::literal(value), &expression);

    assert_eq!(result, Some(MatchBindings::new()));
}

/// Test a literal pattern rejects a literal of another variant or with
/// another value.
#[rstest]
#[case::other_integer(LiteralValue::from(5), build_literal(6))]
#[case::integer_and_float(LiteralValue::from(5), build_literal(5.0))]
#[case::integer_and_bool(LiteralValue::from(1), build_literal(true))]
#[case::bool_and_integer(LiteralValue::from(true), build_literal(1))]
#[case::integer_and_decimal(LiteralValue::from(5), build_decimal_literal("5"))]
#[case::float_and_decimal(LiteralValue::from(1.5), build_decimal_literal("1.5"))]
#[case::nan_and_number(LiteralValue::from(f64::NAN), build_literal(1.0))]
fn pattern_literal_rejects_an_unequal_literal(
    #[case] value: LiteralValue,
    #[case] expression: Expression,
) {
    let result = match_infallibly(&Pattern::literal(value), &expression);

    assert!(result.is_none(), "got {result:?}");
}

/// Test a literal pattern and a repeated capture agree on which literals
/// are equal: `literal(a)` matches `b` exactly when `x - x` matches `a - b`.
#[rstest]
#[case::integer_and_integer_text(LiteralValue::from(5), build_parsed_literal("05"))]
#[case::nan_and_nan(LiteralValue::from(f64::NAN), build_literal(f64::NAN))]
#[case::zero_and_negative_zero(LiteralValue::from(0.0), build_literal(-0.0))]
#[case::decimal_texts(
    LiteralValue::parse_text("1.5").expect("a text"),
    build_decimal_literal("1.50")
)]
#[case::integer_and_float(LiteralValue::from(1), build_literal(1.0))]
#[case::integer_and_bool(LiteralValue::from(1), build_literal(true))]
#[case::float_and_decimal(LiteralValue::from(1.5), build_decimal_literal("1.5"))]
#[case::integer_and_decimal(LiteralValue::from(1), build_decimal_literal("1"))]
fn pattern_literal_and_repeated_capture_agree_on_literal_equality(
    #[case] value: LiteralValue,
    #[case] other: Expression,
) {
    let x = Capture::new("x");
    let repeated = Pattern::binary_any_operation(Pattern::capture(&x), Pattern::capture(&x));
    let difference = Expression::new_binary(
        BinaryOperation::Subtract,
        build_literal(value.clone()),
        &other,
    );

    let by_literal = Pattern::literal(value)
        .is_match(&other)
        .expect("no predicate");
    let by_capture = repeated.is_match(&difference).expect("no predicate");

    assert_eq!(by_literal, by_capture);
}

// =============================================================================
// Identifier
// =============================================================================

/// Test the any-identifier pattern matches any reference.
#[test]
fn pattern_any_identifier_matches_any_reference() {
    let (_, x) = build_identifier("x");
    let (_, y) = build_identifier("y");

    let x_result = match_infallibly(&Pattern::any_identifier(), &x);
    let y_result = match_infallibly(&Pattern::any_identifier(), &y);

    assert_eq!(x_result, Some(MatchBindings::new()));
    assert_eq!(y_result, Some(MatchBindings::new()));
}

/// Test the any-identifier pattern rejects every node that is not a
/// reference.
#[rstest]
fn pattern_identifier_rejects_non_reference_nodes(
    #[values(
        NodeKind::Literal,
        NodeKind::Identifier,
        NodeKind::Unary,
        NodeKind::Binary,
        NodeKind::Logical,
        NodeKind::Piecewise,
        NodeKind::Call
    )]
    kind: NodeKind,
) {
    let result = match_infallibly(&Pattern::any_identifier(), &kind.build());

    assert_eq!(
        result.is_some(),
        kind == NodeKind::Identifier,
        "got {result:?}"
    );
}

/// Test an identifier pattern matches a reference to the same identifier,
/// including a reference built from a clone of it.
#[test]
fn pattern_identifier_matches_a_reference_to_the_same_identifier() {
    let (identifier, reference) = build_identifier("x");
    let pattern = Pattern::identifier(identifier.clone());

    let direct = match_infallibly(&pattern, &reference);
    let through_clone = match_infallibly(&pattern, &Expression::from(identifier));

    assert_eq!(direct, Some(MatchBindings::new()));
    assert_eq!(through_clone, Some(MatchBindings::new()));
}

/// Test an identifier pattern rejects a different identifier with the same
/// name hint.
#[test]
fn pattern_identifier_rejects_a_different_identifier_with_the_same_hint() {
    let (first, _) = build_identifier("x");
    let (_, second_reference) = build_identifier("x");

    let result = match_infallibly(&Pattern::identifier(first), &second_reference);

    assert!(result.is_none(), "got {result:?}");
}

// =============================================================================
// Unary
// =============================================================================

/// Test a unary pattern matches its operation.
#[test]
fn pattern_unary_matches_its_operation() {
    let pattern = Pattern::unary(UnaryOperation::Negate, Pattern::wildcard());

    let result = match_infallibly(&pattern, &-build_literal(5));

    assert_eq!(result, Some(MatchBindings::new()));
}

/// Test a unary pattern rejects another operation.
#[test]
fn pattern_unary_rejects_another_operation() {
    let pattern = Pattern::unary(UnaryOperation::Negate, Pattern::wildcard());
    let expression = Expression::new_unary(UnaryOperation::LogicalNot, build_literal(5));

    let result = match_infallibly(&pattern, &expression);

    assert!(result.is_none(), "got {result:?}");
}

/// Test a unary pattern of any operation matches every operation.
#[rstest]
#[case::negate(UnaryOperation::Negate)]
#[case::positive(UnaryOperation::Positive)]
#[case::logical_not(UnaryOperation::LogicalNot)]
fn pattern_unary_any_operation_matches_every_operation(#[case] operation: UnaryOperation) {
    let pattern = Pattern::unary_any_operation(Pattern::wildcard());

    let result = match_infallibly(
        &pattern,
        &Expression::new_unary(operation, build_literal(5)),
    );

    assert_eq!(result, Some(MatchBindings::new()));
}

/// Test a unary pattern rejects every node that is not unary.
#[rstest]
fn pattern_unary_rejects_other_node_kinds(
    #[values(
        NodeKind::Literal,
        NodeKind::Identifier,
        NodeKind::Unary,
        NodeKind::Binary,
        NodeKind::Logical,
        NodeKind::Piecewise,
        NodeKind::Call
    )]
    kind: NodeKind,
) {
    let pattern = Pattern::unary_any_operation(Pattern::wildcard());

    let result = match_infallibly(&pattern, &kind.build());

    assert_eq!(result.is_some(), kind == NodeKind::Unary, "got {result:?}");
}

/// Test a unary pattern binds the captures of its operand pattern.
#[test]
fn pattern_unary_binds_captures_in_its_operand() {
    let x = Capture::new("x");
    let pattern = Pattern::unary(UnaryOperation::Negate, Pattern::capture(&x));
    let operand = build_literal(5);

    let bindings = expect_match(&pattern, &-&operand);

    assert!(Expression::ptr_eq(&bindings[&x], &operand));
}

/// Test a unary pattern fails when its operand pattern fails.
#[test]
fn pattern_unary_fails_when_its_operand_fails() {
    let pattern = Pattern::unary(UnaryOperation::Negate, Pattern::literal(5));

    let result = match_infallibly(&pattern, &-build_literal(6));

    assert!(result.is_none(), "got {result:?}");
}

// =============================================================================
// Binary
// =============================================================================

/// Test a binary pattern matches its operation.
#[test]
fn pattern_binary_matches_its_operation() {
    let pattern = Pattern::binary(
        BinaryOperation::Add,
        Pattern::wildcard(),
        Pattern::wildcard(),
    );

    let result = match_infallibly(&pattern, &build_simple_binary(BinaryOperation::Add));

    assert_eq!(result, Some(MatchBindings::new()));
}

/// Test a binary pattern rejects another operation.
#[test]
fn pattern_binary_rejects_another_operation() {
    let pattern = Pattern::binary(
        BinaryOperation::Add,
        Pattern::wildcard(),
        Pattern::wildcard(),
    );

    let result = match_infallibly(&pattern, &build_simple_binary(BinaryOperation::Subtract));

    assert!(result.is_none(), "got {result:?}");
}

/// Test a binary pattern of any operation matches every operation.
#[rstest]
#[case::add(BinaryOperation::Add)]
#[case::multiply(BinaryOperation::Multiply)]
#[case::power(BinaryOperation::Power)]
#[case::floor_mod(BinaryOperation::FloorMod)]
#[case::greater_equal(BinaryOperation::GreaterEqual)]
fn pattern_binary_any_operation_matches_every_operation(#[case] operation: BinaryOperation) {
    let pattern = Pattern::binary_any_operation(Pattern::wildcard(), Pattern::wildcard());

    let result = match_infallibly(&pattern, &build_simple_binary(operation));

    assert_eq!(result, Some(MatchBindings::new()));
}

/// Test a binary pattern rejects every node that is not binary.
#[rstest]
fn pattern_binary_rejects_other_node_kinds(
    #[values(
        NodeKind::Literal,
        NodeKind::Identifier,
        NodeKind::Unary,
        NodeKind::Binary,
        NodeKind::Logical,
        NodeKind::Piecewise,
        NodeKind::Call
    )]
    kind: NodeKind,
) {
    let pattern = Pattern::binary_any_operation(Pattern::wildcard(), Pattern::wildcard());

    let result = match_infallibly(&pattern, &kind.build());

    assert_eq!(result.is_some(), kind == NodeKind::Binary, "got {result:?}");
}

/// Test a binary pattern binds the captures of both operand patterns, left
/// first.
#[test]
fn pattern_binary_binds_captures_in_both_operands() {
    let (a, b) = (Capture::new("a"), Capture::new("b"));
    let pattern = Pattern::binary(
        BinaryOperation::Add,
        Pattern::capture(&a),
        Pattern::capture(&b),
    );
    let left = build_literal(1);
    let right = build_literal(2);

    let bindings = expect_match(
        &pattern,
        &Expression::new_binary(BinaryOperation::Add, &left, &right),
    );

    assert_eq!(collect_captures(&bindings), [&a, &b]);
    assert!(Expression::ptr_eq(&bindings[&a], &left));
    assert!(Expression::ptr_eq(&bindings[&b], &right));
}

/// Test a binary pattern fails when its left operand pattern fails.
#[test]
fn pattern_binary_fails_when_its_left_operand_fails() {
    let pattern = Pattern::binary(
        BinaryOperation::Add,
        Pattern::literal(99),
        Pattern::wildcard(),
    );

    let result = match_infallibly(&pattern, &build_simple_binary(BinaryOperation::Add));

    assert!(result.is_none(), "got {result:?}");
}

/// Test a binary pattern fails when its right operand pattern fails.
#[test]
fn pattern_binary_fails_when_its_right_operand_fails() {
    let pattern = Pattern::binary(
        BinaryOperation::Add,
        Pattern::wildcard(),
        Pattern::literal(99),
    );

    let result = match_infallibly(&pattern, &build_simple_binary(BinaryOperation::Add));

    assert!(result.is_none(), "got {result:?}");
}

/// Test a binary pattern whose left operand fails does not try its right
/// operand.
#[test]
fn pattern_binary_skips_the_right_operand_after_a_left_failure() {
    let calls = Arc::new(AtomicUsize::new(0));
    let pattern =
        Pattern::binary_any_operation(Pattern::literal(99), build_counting_predicate(&calls, true));

    let result = match_infallibly(&pattern, &build_simple_binary(BinaryOperation::Add));

    assert!(result.is_none(), "got {result:?}");
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

// =============================================================================
// Logical
// =============================================================================

/// Test a logical pattern matches its operation and operand count, binding
/// the captures of its operand patterns in order.
#[test]
fn pattern_logical_binds_captures_in_its_operands() {
    let (a, b) = (Capture::new("a"), Capture::new("b"));
    let pattern = Pattern::logical(
        LogicalOperation::Or,
        [Pattern::capture(&a), Pattern::capture(&b)],
    );
    let (_, x) = build_identifier("x");
    let (first, second) = (x.less(1), x.greater(2));

    let bindings = expect_match(&pattern, &Expression::any([&first, &second]));

    assert_eq!(collect_captures(&bindings), [&a, &b]);
    assert!(Expression::ptr_eq(&bindings[&a], &first));
    assert!(Expression::ptr_eq(&bindings[&b], &second));
}

/// Test a logical pattern rejects the other operation.
#[rstest]
#[case::and_pattern_on_or(LogicalOperation::And, LogicalOperation::Or)]
#[case::or_pattern_on_and(LogicalOperation::Or, LogicalOperation::And)]
fn pattern_logical_rejects_another_operation(
    #[case] wanted: LogicalOperation,
    #[case] actual: LogicalOperation,
) {
    let pattern = Pattern::logical(wanted, [Pattern::wildcard(), Pattern::wildcard()]);

    let result = match_infallibly(&pattern, &build_simple_logical(actual, 2));

    assert!(result.is_none(), "got {result:?}");
}

/// Test a logical pattern with operand patterns rejects another operand
/// count.
#[rstest]
#[case::fewer_operands(2)]
#[case::more_operands(4)]
fn pattern_logical_rejects_another_operand_count(#[case] count: i64) {
    let pattern = Pattern::logical(
        LogicalOperation::And,
        [
            Pattern::wildcard(),
            Pattern::wildcard(),
            Pattern::wildcard(),
        ],
    );

    let result = match_infallibly(
        &pattern,
        &build_simple_logical(LogicalOperation::And, count),
    );

    assert!(result.is_none(), "got {result:?}");
}

/// Test a logical pattern fails when one of its operand patterns fails.
#[test]
fn pattern_logical_fails_when_an_operand_fails() {
    let pattern = Pattern::logical(
        LogicalOperation::And,
        [Pattern::wildcard(), Pattern::any_literal()],
    );

    let result = match_infallibly(&pattern, &build_simple_logical(LogicalOperation::And, 2));

    assert!(result.is_none(), "got {result:?}");
}

/// Test a logical pattern of any operation matches both operations.
#[rstest]
#[case::and(LogicalOperation::And)]
#[case::or(LogicalOperation::Or)]
fn pattern_logical_any_operation_matches_both_operations(#[case] operation: LogicalOperation) {
    let pattern = Pattern::logical_any_operation([Pattern::wildcard(), Pattern::wildcard()]);

    let result = match_infallibly(&pattern, &build_simple_logical(operation, 2));

    assert_eq!(result, Some(MatchBindings::new()));
}

/// Test a logical pattern of any operands matches any operand count of its
/// operation, and not the other operation.
#[rstest]
#[case::two_operands(2)]
#[case::five_operands(5)]
fn pattern_logical_any_operands_matches_any_operand_count(#[case] count: i64) {
    let pattern = Pattern::logical_any_operands(LogicalOperation::And);

    let conjunction = match_infallibly(
        &pattern,
        &build_simple_logical(LogicalOperation::And, count),
    );
    let disjunction =
        match_infallibly(&pattern, &build_simple_logical(LogicalOperation::Or, count));

    assert_eq!(conjunction, Some(MatchBindings::new()));
    assert!(disjunction.is_none(), "got {disjunction:?}");
}

/// Test the any-logical pattern matches logical nodes only.
#[rstest]
fn pattern_any_logical_rejects_other_node_kinds(
    #[values(
        NodeKind::Literal,
        NodeKind::Identifier,
        NodeKind::Unary,
        NodeKind::Binary,
        NodeKind::Logical,
        NodeKind::Piecewise,
        NodeKind::Call
    )]
    kind: NodeKind,
) {
    let result = match_infallibly(&Pattern::any_logical(), &kind.build());

    assert_eq!(
        result.is_some(),
        kind == NodeKind::Logical,
        "got {result:?}"
    );
}

// =============================================================================
// Piecewise
// =============================================================================

/// Test a piecewise pattern with no case pattern matches nothing, since
/// every piecewise has a case.
#[rstest]
#[case::one_case(build_one_case_piecewise())]
#[case::two_cases(build_two_case_piecewise())]
fn pattern_piecewise_with_no_cases_matches_nothing(#[case] expression: Expression) {
    let pattern = Pattern::piecewise(Vec::<(Pattern, Pattern)>::new(), Pattern::wildcard());

    let result = match_infallibly(&pattern, &expression);

    assert!(result.is_none(), "got {result:?}");
}

/// Test a piecewise pattern with one case pattern matches a one-case
/// piecewise.
#[test]
fn pattern_piecewise_matches_a_one_case_piecewise() {
    let pattern = Pattern::piecewise(
        [(Pattern::wildcard(), Pattern::wildcard())],
        Pattern::wildcard(),
    );

    let result = match_infallibly(&pattern, &build_one_case_piecewise());

    assert_eq!(result, Some(MatchBindings::new()));
}

/// Test a piecewise pattern rejects every node that is not a piecewise.
#[rstest]
fn pattern_piecewise_rejects_other_node_kinds(
    #[values(
        NodeKind::Literal,
        NodeKind::Identifier,
        NodeKind::Unary,
        NodeKind::Binary,
        NodeKind::Logical,
        NodeKind::Piecewise,
        NodeKind::Call
    )]
    kind: NodeKind,
) {
    let pattern = Pattern::piecewise_any_cases(Pattern::wildcard());

    let result = match_infallibly(&pattern, &kind.build());

    assert_eq!(
        result.is_some(),
        kind == NodeKind::Piecewise,
        "got {result:?}"
    );
}

/// Test a piecewise pattern of any cases matches any case count and
/// matches only the otherwise branch.
#[test]
fn pattern_piecewise_any_cases_matches_any_case_count() {
    let o = Capture::new("o");
    let pattern = Pattern::piecewise_any_cases(Pattern::capture(&o));

    let one_case_bindings = expect_match(&pattern, &build_one_case_piecewise());
    let two_case_bindings = expect_match(&pattern, &build_two_case_piecewise());

    assert_eq!(collect_captures(&one_case_bindings), [&o]);
    assert_eq!(one_case_bindings[&o], build_literal(2));
    assert_eq!(collect_captures(&two_case_bindings), [&o]);
    assert_eq!(two_case_bindings[&o], build_literal(3));
}

/// Test a piecewise pattern with case patterns rejects a different case
/// count.
#[test]
fn pattern_piecewise_rejects_a_different_case_count() {
    let pattern = Pattern::piecewise(
        [(Pattern::wildcard(), Pattern::wildcard())],
        Pattern::wildcard(),
    );

    let result = match_infallibly(&pattern, &build_two_case_piecewise());

    assert!(result.is_none(), "got {result:?}");
}

/// Test a piecewise pattern binds the condition, the value and the
/// otherwise branch.
#[test]
fn pattern_piecewise_binds_condition_value_and_otherwise() {
    let (c, v, o) = (Capture::new("c"), Capture::new("v"), Capture::new("o"));
    let pattern = Pattern::piecewise(
        [(Pattern::capture(&c), Pattern::capture(&v))],
        Pattern::capture(&o),
    );
    let (condition, value, otherwise) = (build_literal(true), build_literal(1), build_literal(2));
    let expression =
        Expression::piecewise([(&condition, &value)], &otherwise).expect("a valid piecewise");

    let bindings = expect_match(&pattern, &expression);

    assert_eq!(collect_captures(&bindings), [&c, &v, &o]);
    assert!(Expression::ptr_eq(&bindings[&c], &condition));
    assert!(Expression::ptr_eq(&bindings[&v], &value));
    assert!(Expression::ptr_eq(&bindings[&o], &otherwise));
}

/// Test a piecewise pattern binds every case in evaluation order,
/// condition before value.
#[test]
fn pattern_piecewise_binds_cases_in_evaluation_order() {
    let (c1, v1) = (Capture::new("c1"), Capture::new("v1"));
    let (c2, v2) = (Capture::new("c2"), Capture::new("v2"));
    let pattern = Pattern::piecewise(
        [
            (Pattern::capture(&c1), Pattern::capture(&v1)),
            (Pattern::capture(&c2), Pattern::capture(&v2)),
        ],
        Pattern::wildcard(),
    );
    let (first_condition, second_condition) = (build_literal(true), build_literal(false));
    let (first_value, second_value) = (build_literal(1), build_literal(2));
    let expression = Expression::piecewise(
        [
            (&first_condition, &first_value),
            (&second_condition, &second_value),
        ],
        build_literal(0),
    )
    .expect("a valid piecewise");

    let bindings = expect_match(&pattern, &expression);

    assert_eq!(collect_captures(&bindings), [&c1, &v1, &c2, &v2]);
    assert!(Expression::ptr_eq(&bindings[&c1], &first_condition));
    assert!(Expression::ptr_eq(&bindings[&v1], &first_value));
    assert!(Expression::ptr_eq(&bindings[&c2], &second_condition));
    assert!(Expression::ptr_eq(&bindings[&v2], &second_value));
}

/// Test a piecewise pattern fails when a condition pattern fails.
#[test]
fn pattern_piecewise_fails_when_a_condition_fails() {
    let pattern = Pattern::piecewise(
        [(Pattern::literal(99), Pattern::wildcard())],
        Pattern::wildcard(),
    );

    let result = match_infallibly(&pattern, &build_one_case_piecewise());

    assert!(result.is_none(), "got {result:?}");
}

/// Test a piecewise pattern fails when a value pattern fails.
#[test]
fn pattern_piecewise_fails_when_a_value_fails() {
    let pattern = Pattern::piecewise(
        [(Pattern::wildcard(), Pattern::literal(99))],
        Pattern::wildcard(),
    );

    let result = match_infallibly(&pattern, &build_one_case_piecewise());

    assert!(result.is_none(), "got {result:?}");
}

/// Test a piecewise pattern fails when its otherwise pattern fails, although
/// every case matches.
#[test]
fn pattern_piecewise_fails_when_the_otherwise_branch_fails() {
    let pattern = Pattern::piecewise(
        [(Pattern::wildcard(), Pattern::wildcard())],
        Pattern::literal(99),
    );

    let result = match_infallibly(&pattern, &build_one_case_piecewise());

    assert!(result.is_none(), "got {result:?}");
}

/// Test a capture repeated across a case value and the otherwise branch
/// requires them to be equal.
#[rstest]
#[case::equal_branches(build_literal(1), true)]
#[case::different_branches(build_literal(2), false)]
fn pattern_piecewise_repeated_capture_spans_cases_and_otherwise(
    #[case] otherwise: Expression,
    #[case] expected_match: bool,
) {
    let v = Capture::new("v");
    let pattern = Pattern::piecewise(
        [(Pattern::wildcard(), Pattern::capture(&v))],
        Pattern::capture(&v),
    );
    let expression = Expression::piecewise([(build_literal(true), build_literal(1))], otherwise)
        .expect("a piecewise");

    let result = match_infallibly(&pattern, &expression);

    assert_eq!(result.is_some(), expected_match, "got {result:?}");
}

// =============================================================================
// Call
// =============================================================================

/// Test a call pattern matches its callee.
#[test]
fn pattern_call_matches_its_callee() {
    let pattern = Pattern::call(build_callee("f"), [Pattern::wildcard()]);

    let result = match_infallibly(&pattern, &build_call_or_panic("f", vec![build_literal(1)]));

    assert_eq!(result, Some(MatchBindings::new()));
}

/// Test a call pattern rejects another callee: another user function, or a
/// built-in function.
#[rstest]
#[case::another_name("g")]
#[case::builtin("sqrt")]
fn pattern_call_rejects_another_callee(#[case] function_name: &str) {
    let pattern = Pattern::call(build_callee("f"), [Pattern::wildcard()]);

    let result = match_infallibly(
        &pattern,
        &build_call_or_panic(function_name, vec![build_literal(1)]),
    );

    assert!(result.is_none(), "got {result:?}");
}

/// Test a call pattern of a built-in function matches calls of it and not
/// of another built-in.
#[test]
fn pattern_call_of_a_builtin_matches_only_that_builtin() {
    let pattern = Pattern::call(BuiltinFunction::Sqrt, [Pattern::wildcard()]);

    let sqrt = match_infallibly(&pattern, &Expression::call(BuiltinFunction::Sqrt, [1]));
    let exp = match_infallibly(&pattern, &Expression::call(BuiltinFunction::Exp, [1]));

    assert_eq!(sqrt, Some(MatchBindings::new()));
    assert!(exp.is_none(), "got {exp:?}");
}

/// Test a call pattern of any callee matches any callee.
#[rstest]
#[case::f("f")]
#[case::g("g")]
#[case::long_name("a_function_name")]
#[case::builtin("sqrt")]
fn pattern_call_any_callee_matches_any_callee(#[case] function_name: &str) {
    let pattern = Pattern::call_any_callee([Pattern::wildcard()]);

    let result = match_infallibly(
        &pattern,
        &build_call_or_panic(function_name, vec![build_literal(1)]),
    );

    assert_eq!(result, Some(MatchBindings::new()));
}

/// Test a call pattern with argument patterns rejects another arity.
#[rstest]
#[case::fewer_arguments(vec![build_literal(1)])]
#[case::more_arguments(vec![build_literal(1), build_literal(2), build_literal(3)])]
fn pattern_call_rejects_another_arity(#[case] arguments: Vec<Expression>) {
    let pattern = Pattern::call(
        build_callee("f"),
        [Pattern::wildcard(), Pattern::wildcard()],
    );

    let result = match_infallibly(&pattern, &build_call_or_panic("f", arguments));

    assert!(result.is_none(), "got {result:?}");
}

/// Test a call pattern of any arguments matches any arity and matches no
/// argument.
#[rstest]
#[case::no_arguments(Vec::new())]
#[case::one_argument(vec![build_literal(1)])]
#[case::two_arguments(vec![build_literal(1), build_literal(2)])]
fn pattern_call_any_arguments_matches_any_arity(#[case] arguments: Vec<Expression>) {
    let pattern = Pattern::call_any_arguments(build_callee("f"));

    let result = match_infallibly(&pattern, &build_call_or_panic("f", arguments));

    assert_eq!(result, Some(MatchBindings::new()));
}

/// Test a call pattern of any arguments still requires its callee.
#[test]
fn pattern_call_any_arguments_rejects_another_callee() {
    let pattern = Pattern::call_any_arguments(build_callee("f"));

    let result = match_infallibly(&pattern, &build_call_or_panic("g", vec![build_literal(1)]));

    assert!(result.is_none(), "got {result:?}");
}

/// Test a call pattern binds the captures of its argument patterns in
/// order.
#[test]
fn pattern_call_binds_captures_in_its_arguments() {
    let (a, b) = (Capture::new("a"), Capture::new("b"));
    let pattern = Pattern::call(
        build_callee("f"),
        [Pattern::capture(&a), Pattern::capture(&b)],
    );
    let (first, second) = (build_literal(1), build_literal(2));

    let bindings = expect_match(
        &pattern,
        &build_call_or_panic("f", vec![first.clone(), second.clone()]),
    );

    assert_eq!(collect_captures(&bindings), [&a, &b]);
    assert!(Expression::ptr_eq(&bindings[&a], &first));
    assert!(Expression::ptr_eq(&bindings[&b], &second));
}

/// Test the any-call pattern rejects every node that is not a call.
#[rstest]
fn pattern_call_rejects_other_node_kinds(
    #[values(
        NodeKind::Literal,
        NodeKind::Identifier,
        NodeKind::Unary,
        NodeKind::Binary,
        NodeKind::Logical,
        NodeKind::Piecewise,
        NodeKind::Call
    )]
    kind: NodeKind,
) {
    let result = match_infallibly(&Pattern::any_call(), &kind.build());

    assert_eq!(result.is_some(), kind == NodeKind::Call, "got {result:?}");
}

/// Test a call pattern with no argument pattern matches only calls without
/// arguments.
#[test]
fn pattern_call_with_empty_arguments_matches_only_calls_without_arguments() {
    let pattern = Pattern::call(build_callee("f"), Vec::<Pattern>::new());

    let without_arguments = match_infallibly(
        &pattern,
        &build_call_or_panic("f", Vec::<Expression>::new()),
    );
    let with_argument =
        match_infallibly(&pattern, &build_call_or_panic("f", vec![build_literal(1)]));

    assert_eq!(without_arguments, Some(MatchBindings::new()));
    assert!(with_argument.is_none(), "got {with_argument:?}");
}

/// Test a capture repeated across call arguments requires equal
/// arguments.
#[rstest]
#[case::equal_arguments(build_literal(1), true)]
#[case::different_arguments(build_literal(2), false)]
fn pattern_call_repeated_capture_requires_equal_arguments(
    #[case] second_argument: Expression,
    #[case] expected_match: bool,
) {
    let x = Capture::new("x");
    let pattern = Pattern::call(
        build_callee("f"),
        [Pattern::capture(&x), Pattern::capture(&x)],
    );
    let expression = build_call_or_panic("f", vec![build_literal(1), second_argument]);

    let result = match_infallibly(&pattern, &expression);

    assert_eq!(result.is_some(), expected_match, "got {result:?}");
}

// =============================================================================
// Predicate
// =============================================================================

/// Test a predicate pattern matches when the predicate holds.
#[test]
fn pattern_predicate_matches_when_the_predicate_holds() {
    let pattern = Pattern::predicate(|_| true);

    let result = match_infallibly(&pattern, &build_literal(5));

    assert_eq!(result, Some(MatchBindings::new()));
}

/// Test a predicate pattern does not match when the predicate does not
/// hold.
#[test]
fn pattern_predicate_rejects_when_the_predicate_does_not_hold() {
    let pattern = Pattern::predicate(|_| false);

    let result = match_infallibly(&pattern, &build_literal(5));

    assert!(result.is_none(), "got {result:?}");
}

/// Test a fallible predicate matches exactly when it returns `Ok(true)`.
#[rstest]
#[case::holds(true)]
#[case::does_not_hold(false)]
fn pattern_try_predicate_matches_when_it_answers_true(#[case] verdict: bool) {
    let pattern = Pattern::try_predicate(move |_| Ok(verdict));

    let result = match_infallibly(&pattern, &build_literal(5));

    assert_eq!(result.is_some(), verdict, "got {result:?}");
}

/// Test the predicate is called once, with a handle to the candidate.
#[test]
fn pattern_predicate_receives_the_candidate() {
    let seen = Arc::new(std::sync::Mutex::new(Vec::<Expression>::new()));
    let recorder = Arc::clone(&seen);
    let pattern = Pattern::predicate(move |expression| {
        recorder
            .lock()
            .expect("an unpoisoned lock")
            .push(expression.clone());
        true
    });
    let expression = build_literal(5);

    let result = match_infallibly(&pattern, &expression);

    assert_eq!(result, Some(MatchBindings::new()));
    let seen = seen.lock().expect("an unpoisoned lock");
    assert_eq!(seen.len(), 1);
    assert!(Expression::ptr_eq(&seen[0], &expression));
}

/// Test a failing predicate's error is returned from the match unchanged.
#[test]
fn pattern_predicate_error_is_returned_from_the_match() {
    let pattern = build_failing_predicate("predicate failed");

    let result = pattern.matches(&build_literal(5));

    let error = result.expect_err("the predicate fails");
    assert_eq!(expect_probe_error(&error), &ProbeError("predicate failed"));
}

/// Test a failing predicate ends the match: later sub-patterns are not
/// tried, not even in another alternative.
#[test]
fn pattern_predicate_error_stops_the_match() {
    let calls = Arc::new(AtomicUsize::new(0));
    let pattern = Pattern::alternatives([
        Pattern::binary_any_operation(
            build_failing_predicate("stop"),
            build_counting_predicate(&calls, true),
        ),
        build_counting_predicate(&calls, true),
    ]);

    let result = pattern.is_match(&build_simple_binary(BinaryOperation::Add));

    let error = result.expect_err("the predicate fails");
    assert_eq!(expect_probe_error(&error), &ProbeError("stop"));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

/// Test a predicate pattern binds nothing: beside a capture, only the
/// capture is bound.
#[test]
fn pattern_predicate_captures_nothing() {
    let x = Capture::new("x");
    let pattern = Pattern::binary_any_operation(Pattern::capture(&x), Pattern::predicate(|_| true));

    let bindings = expect_match(&pattern, &build_simple_binary(BinaryOperation::Add));

    assert_eq!(collect_captures(&bindings), [&x]);
}

/// Test a predicate filtering node kinds matches literals only.
#[rstest]
fn pattern_predicate_filters_node_kinds(
    #[values(
        NodeKind::Literal,
        NodeKind::Identifier,
        NodeKind::Unary,
        NodeKind::Binary,
        NodeKind::Logical,
        NodeKind::Piecewise,
        NodeKind::Call
    )]
    kind: NodeKind,
) {
    let pattern =
        Pattern::predicate(|expression| matches!(expression.kind(), ExpressionKind::Literal(_)));

    let result = match_infallibly(&pattern, &kind.build());

    assert_eq!(
        result.is_some(),
        kind == NodeKind::Literal,
        "got {result:?}"
    );
}

/// Test a clone of a predicate pattern shares the predicate.
#[test]
fn pattern_clone_shares_the_predicate() {
    let calls = Arc::new(AtomicUsize::new(0));
    let pattern = build_counting_predicate(&calls, true);
    let clone = pattern.clone();

    let original_result = match_infallibly(&pattern, &build_literal(1));
    let clone_result = match_infallibly(&clone, &build_literal(1));

    assert_eq!(original_result, Some(MatchBindings::new()));
    assert_eq!(clone_result, Some(MatchBindings::new()));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

// =============================================================================
// Alternatives
// =============================================================================

/// Test an alternatives pattern of no alternative matches nothing.
#[rstest]
fn pattern_alternatives_of_none_matches_nothing(
    #[values(
        NodeKind::Literal,
        NodeKind::Identifier,
        NodeKind::Unary,
        NodeKind::Binary,
        NodeKind::Logical,
        NodeKind::Piecewise,
        NodeKind::Call
    )]
    kind: NodeKind,
) {
    let pattern = Pattern::alternatives(Vec::<Pattern>::new());

    let result = match_infallibly(&pattern, &kind.build());

    assert!(result.is_none(), "got {result:?}");
}

/// Test the first matching alternative decides the bindings.
#[test]
fn pattern_alternatives_returns_the_first_match() {
    let (first, second) = (Capture::new("first"), Capture::new("second"));
    let pattern = Pattern::alternatives([
        Pattern::any_literal().captured_as(&first),
        Pattern::wildcard().captured_as(&second),
    ]);
    let expression = build_literal(5);

    let bindings = expect_match(&pattern, &expression);

    assert_eq!(collect_captures(&bindings), [&first]);
    assert!(Expression::ptr_eq(&bindings[&first], &expression));
}

/// Test a later alternative matches when the earlier ones fail.
#[test]
fn pattern_alternatives_falls_through_to_a_later_alternative() {
    let x = Capture::new("x");
    let pattern = Pattern::alternatives([
        Pattern::any_literal().captured_as(&x),
        Pattern::any_identifier().captured_as(&x),
    ]);
    let (_, expression) = build_identifier("x");

    let bindings = expect_match(&pattern, &expression);

    assert!(Expression::ptr_eq(&bindings[&x], &expression));
}

/// Test an alternatives pattern fails when every alternative fails.
#[test]
fn pattern_alternatives_fails_when_every_alternative_fails() {
    let pattern = Pattern::alternatives([Pattern::literal(5), Pattern::literal(6)]);

    let result = match_infallibly(&pattern, &build_literal(7));

    assert!(result.is_none(), "got {result:?}");
}

/// Test a capture bound by an alternative that then fails does not
/// constrain the next alternative: in `1 + 2`, the first alternative binds
/// `x` to the left operand before its right operand fails to match, and the
/// second binds `x` to the right operand alone.
#[test]
fn pattern_alternatives_isolates_failed_attempts() {
    let x = Capture::new("x");
    let pattern = Pattern::alternatives([
        Pattern::binary(
            BinaryOperation::Add,
            Pattern::capture(&x),
            Pattern::literal(99),
        ),
        Pattern::binary(
            BinaryOperation::Add,
            Pattern::wildcard(),
            Pattern::capture(&x),
        ),
    ]);

    let bindings = expect_match(&pattern, &build_simple_binary(BinaryOperation::Add));

    assert_eq!(collect_captures(&bindings), [&x]);
    assert_eq!(bindings[&x], build_literal(2));
}

/// Test a capture made inside a failed alternative is discarded.
#[test]
fn pattern_alternatives_discards_captures_of_a_failed_alternative() {
    let (in_failed, in_successful) = (Capture::new("in_failed"), Capture::new("in_successful"));
    let pattern = Pattern::alternatives([
        Pattern::binary(
            BinaryOperation::Add,
            Pattern::literal(1).captured_as(&in_failed),
            Pattern::literal(99),
        ),
        Pattern::capture(&in_successful),
    ]);

    let bindings = expect_match(&pattern, &build_simple_binary(BinaryOperation::Add));

    assert_eq!(collect_captures(&bindings), [&in_successful]);
}

/// Test a match that fails after an alternative bound a capture leaves no
/// binding behind: in `5 + 1`, the chosen alternative binds `x` before the
/// right operand fails, and the enclosing alternative binds only `z`.
#[test]
fn pattern_alternatives_restore_bindings_after_a_later_sibling_fails() {
    let (x, y, z) = (Capture::new("x"), Capture::new("y"), Capture::new("z"));
    let failing = Pattern::binary_any_operation(
        Pattern::alternatives([Pattern::any_literal().captured_as(&x), Pattern::capture(&y)]),
        Pattern::literal(0),
    );
    let pattern = Pattern::alternatives([failing.clone(), Pattern::capture(&z)]);
    let expression =
        Expression::new_binary(BinaryOperation::Add, build_literal(5), build_literal(1));

    let failed = match_infallibly(&failing, &expression);
    let bindings = expect_match(&pattern, &expression);

    assert!(failed.is_none(), "got {failed:?}");
    assert_eq!(collect_captures(&bindings), [&z]);
}

/// Test an alternative sees the bindings made before the alternatives
/// pattern, and its captures are visible to later siblings.
#[test]
fn pattern_alternatives_threads_bindings_through_the_chosen_alternative() {
    let (x, y) = (Capture::new("x"), Capture::new("y"));
    let pattern = Pattern::binary_any_operation(
        Pattern::capture(&x),
        Pattern::alternatives([
            Pattern::literal(2).captured_as(&x),
            Pattern::wildcard().captured_as(&y),
        ]),
    );

    let bindings = expect_match(&pattern, &build_simple_binary(BinaryOperation::Add));

    assert_eq!(collect_captures(&bindings), [&x, &y]);
    assert_eq!(bindings[&y], build_literal(2));
}

/// Test the choice of an alternative is final: when a later sibling fails,
/// the remaining alternatives are not tried.
#[test]
fn pattern_alternatives_commits_to_the_first_match() {
    let x = Capture::new("x");
    let pattern = Pattern::binary_any_operation(
        Pattern::alternatives([Pattern::any_literal().captured_as(&x), Pattern::wildcard()]),
        Pattern::capture(&x),
    );

    let result = match_infallibly(&pattern, &build_simple_binary(BinaryOperation::Add));

    assert!(result.is_none(), "got {result:?}");
}

/// Test an alternatives pattern of 10,000 failing branches before a
/// matching one finds the match and binds only its captures.
#[test]
fn pattern_alternatives_of_many_failing_branches_matches_in_linear_time() {
    let x = Capture::new("x");
    let failing = (1..=10_000).map(|value| {
        Pattern::binary(
            BinaryOperation::Add,
            Pattern::capture(&x),
            Pattern::literal(value),
        )
    });
    let matching = Pattern::binary(
        BinaryOperation::Add,
        Pattern::capture(&x),
        Pattern::literal(0),
    );
    let pattern = Pattern::alternatives(failing.chain([matching]));
    let (_, a) = build_identifier("a");

    let bindings = expect_match(&pattern, &(&a + 0));

    assert_eq!(collect_captures(&bindings), [&x]);
    assert!(Expression::ptr_eq(&bindings[&x], &a));
}

// =============================================================================
// matches and is_match
// =============================================================================

/// Test `matches` returns `None` for a mismatch.
#[test]
fn pattern_matches_returns_none_on_a_mismatch() {
    let result = match_infallibly(&Pattern::literal(5), &build_literal(6));

    assert!(result.is_none(), "got {result:?}");
}

/// Test `is_match` answers true for a match.
#[test]
fn pattern_is_match_is_true_on_a_match() {
    let result = Pattern::literal(5).is_match(&build_literal(5));

    assert!(result.expect("no callback"));
}

/// Test `is_match` answers false for a mismatch.
#[test]
fn pattern_is_match_is_false_on_a_mismatch() {
    let result = Pattern::literal(5).is_match(&build_literal(6));

    assert!(!result.expect("no callback"));
}

/// Test matching tests the root only and never searches subexpressions.
#[test]
fn pattern_matches_does_not_search_subexpressions() {
    let result = match_infallibly(
        &Pattern::literal(1),
        &build_simple_binary(BinaryOperation::Add),
    );

    assert!(result.is_none(), "got {result:?}");
}

/// Return the pattern `5`.
fn build_five_pattern(_: &Capture, _: &Capture) -> Pattern {
    Pattern::literal(5)
}

/// Return the pattern `a + b`.
fn build_sum_pattern(a: &Capture, b: &Capture) -> Pattern {
    Pattern::binary(
        BinaryOperation::Add,
        Pattern::capture(a),
        Pattern::capture(b),
    )
}

/// Return the pattern matching `5` or any identifier.
fn build_five_or_identifier_pattern(_: &Capture, _: &Capture) -> Pattern {
    Pattern::alternatives([Pattern::literal(5), Pattern::any_identifier()])
}

/// Test repeated matches of one pattern and expression give the same,
/// expected result: the values bound to `a` and `b`, in binding order.
#[rstest]
#[case::literal(build_five_pattern, None)]
#[case::binary_capture(build_sum_pattern, Some(vec![(0, 1), (1, 2)]))]
#[case::alternatives(build_five_or_identifier_pattern, None)]
fn pattern_matches_is_deterministic(
    #[case] build_pattern: fn(&Capture, &Capture) -> Pattern,
    #[case] expected: Option<Vec<(usize, i64)>>,
) {
    let captures = [Capture::new("a"), Capture::new("b")];
    let pattern = build_pattern(&captures[0], &captures[1]);
    let expression = build_simple_binary(BinaryOperation::Add);

    let first = match_infallibly(&pattern, &expression);
    let second = match_infallibly(&pattern, &expression);

    let describe = |bindings: &Option<MatchBindings>| {
        bindings.as_ref().map(|bindings| {
            bindings
                .iter()
                .map(|(capture, expression)| (capture.clone(), expression.clone()))
                .collect::<Vec<_>>()
        })
    };
    let expected = expected.map(|entries| {
        entries
            .into_iter()
            .map(|(index, value)| (captures[index].clone(), build_literal(value)))
            .collect::<Vec<_>>()
    });
    assert_eq!(describe(&first), expected);
    assert_eq!(describe(&second), expected);
}

// =============================================================================
// Callback errors
// =============================================================================

/// Test a callback error built from text displays the text and has no
/// source.
#[rstest]
#[case::from_str(CallbackError::from("no verdict"))]
#[case::from_string(CallbackError::from(String::from("no verdict")))]
fn callback_error_from_text_displays_the_text(#[case] error: CallbackError) {
    let message = error.to_string();

    assert_eq!(message, "no verdict");
    assert!(error.source().is_none());
}

/// Test a callback error is the wrapped error itself: its message and a
/// downcast to it.
#[test]
fn callback_error_downcasts_to_the_wrapped_error() {
    let error = CallbackError::from(ProbeError("probe"));

    let message = error.to_string();
    let downcast = error.downcast::<ProbeError>();

    assert_eq!(message, "probe");
    assert_eq!(downcast.ok().as_deref(), Some(&ProbeError("probe")));
}

/// Fail with a [`ProbeError`] through `?`.
fn fail_with_a_probe_error() -> Result<(), CallbackError> {
    Err(ProbeError("probe"))?
}

/// Fail with a refused piecewise through `?`.
fn fail_with_a_piecewise_error() -> Result<(), CallbackError> {
    Expression::piecewise(Vec::<(Expression, Expression)>::new(), build_literal(0))?;
    Ok(())
}

/// Fail with a refused rebuild through `?`.
fn fail_with_a_rebuild_error() -> Result<(), CallbackError> {
    build_simple_binary(BinaryOperation::Add).rebuild_with_children(Vec::new())?;
    Ok(())
}

/// Fail with a refused function name through `?`.
fn fail_with_a_function_name_error() -> Result<(), CallbackError> {
    FunctionName::try_new("")?;
    Ok(())
}

/// Fail with a refused literal text through `?`.
fn fail_with_a_literal_text_error() -> Result<(), CallbackError> {
    LiteralValue::parse_text("abc")?;
    Ok(())
}

/// Test `?` converts any error into a callback error holding it.
#[rstest]
#[case::probe_error(fail_with_a_probe_error, |error: &CallbackError| {
    error.downcast_ref::<ProbeError>() == Some(&ProbeError("probe"))
})]
#[case::piecewise_error(fail_with_a_piecewise_error, |error: &CallbackError| {
    error.downcast_ref::<PiecewiseError>() == Some(&PiecewiseError::NoCases)
})]
#[case::rebuild_error(fail_with_a_rebuild_error, |error: &CallbackError| {
    error.downcast_ref::<RebuildError>()
        == Some(&RebuildError::ChildCount { expected: 2, actual: 0 })
})]
#[case::function_name_error(fail_with_a_function_name_error, |error: &CallbackError| {
    error.downcast_ref::<FunctionNameError>() == Some(&FunctionNameError::Empty)
})]
#[case::literal_text_error(fail_with_a_literal_text_error, |error: &CallbackError| {
    error
        .downcast_ref::<LiteralTextError>()
        .is_some_and(|wrapped| wrapped.text() == "abc")
})]
fn callback_error_converts_from_any_error_with_question_mark(
    #[case] fail: fn() -> Result<(), CallbackError>,
    #[case] holds_the_error: fn(&CallbackError) -> bool,
) {
    let error = fail().expect_err("the callback fails");

    assert!(holds_the_error(&error), "got {error:?}");
}

/// Test `?` passes a callback error on into a function returning a boxed
/// error, unchanged.
#[test]
fn callback_error_converts_into_a_boxed_error() {
    fn propagate() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        fail_with_a_probe_error()?;
        Ok(())
    }

    let error = propagate().expect_err("the callback fails");

    assert_eq!(
        error.downcast_ref::<ProbeError>(),
        Some(&ProbeError("probe"))
    );
}

// =============================================================================
// Deep trees
// =============================================================================

/// Test a pattern that does not descend matches a tree
/// [`SMALL_STACK_DEPTH`] levels deep on a small thread stack, binding
/// the root's left operand.
#[test]
fn pattern_of_a_shallow_shape_matches_a_deep_tree_on_a_small_stack() {
    run_on_small_stack(|| {
        let rest = Capture::new("rest");
        let (_, x) = build_identifier("x");
        let tree = build_deep_sum(&x, SMALL_STACK_DEPTH);
        let pattern = Pattern::binary(
            BinaryOperation::Add,
            Pattern::capture(&rest),
            Pattern::literal(1),
        );

        let bindings = match_infallibly(&pattern, &tree).expect("the deep sum matches");

        let ExpressionKind::Binary(root) = tree.kind() else {
            panic!("a sum at the root");
        };
        assert!(Expression::ptr_eq(&bindings[&rest], root.left()));
    });
}

/// Test a capture repeated across both operands compares two separately
/// built trees [`SMALL_STACK_DEPTH`] levels deep on a small thread stack:
/// equal operands match, binding the capture to the left one, and operands
/// differing only at the bottom do not.
#[test]
fn pattern_with_a_repeated_capture_compares_deep_operands_on_a_small_stack() {
    run_on_small_stack(|| {
        let operand = Capture::new("operand");
        let (_, x) = build_identifier("x");
        let (_, y) = build_identifier("y");
        let left = build_deep_sum(&x, SMALL_STACK_DEPTH);
        let equal = build_deep_sum(&x, SMALL_STACK_DEPTH);
        let unequal = build_deep_sum(&y, SMALL_STACK_DEPTH);
        let pattern = Pattern::binary(
            BinaryOperation::Subtract,
            Pattern::capture(&operand),
            Pattern::capture(&operand),
        );

        let matched = match_infallibly(&pattern, &(&left - &equal));
        let mismatched = match_infallibly(&pattern, &(&left - &unequal));

        let bindings = matched.expect("equal operands match");
        assert!(Expression::ptr_eq(&bindings[&operand], &left));
        assert!(
            mismatched.is_none(),
            "operands differing at the bottom match"
        );
    });
}

/// Test a pattern mirroring a 50-level chain matches it.
#[test]
fn pattern_matches_a_deeply_nested_chain() {
    let leaf = Capture::new("leaf");
    let (_, x) = build_identifier("x");
    let tree = build_deep_sum(&x, 50);
    let pattern = build_deep_sum_pattern(Pattern::capture(&leaf), 50);

    let bindings = expect_match(&pattern, &tree);

    assert!(Expression::ptr_eq(&bindings[&leaf], &x));
}

/// Test a pattern [`DEEP_TREE_DEPTH`] levels deep matches a tree as deep on a
/// [`PATTERN_MATCH_STACK_BYTES`] thread stack, and a mismatch at the bottom
/// is found.
#[test]
fn pattern_matches_a_pattern_thousands_of_levels_deep() {
    let (matched, mismatched) = run_on_stack(PATTERN_MATCH_STACK_BYTES, || {
        let (identifier, x) = build_identifier("x");
        let (_, y) = build_identifier("y");
        let pattern = build_deep_sum_pattern(Pattern::identifier(identifier), DEEP_TREE_DEPTH);
        let matched = pattern.is_match(&build_deep_sum(&x, DEEP_TREE_DEPTH));
        let mismatched = pattern.is_match(&build_deep_sum(&y, DEEP_TREE_DEPTH));
        (
            matched.expect("no callback"),
            mismatched.expect("no callback"),
        )
    });

    assert!(matched);
    assert!(!mismatched);
}
