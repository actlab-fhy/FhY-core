//! Tests for patterns and match bindings: every pattern shape, the order
//! bindings are threaded and recorded in, repeated captures, exact literal
//! matching, predicates and their failures, alternatives, the free matching
//! functions, and patterns thousands of levels deep.
//!
//! Public API only (`fhy_core::expr::pattern`).

use crate::support::expression as expression_support;
use crate::support::hashing as hashing_support;
use crate::support::pattern as pattern_support;
use crate::support::stack as stack_support;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use expression_support::{
    DEEP_TREE_DEPTH, PATTERN_MATCH_STACK_BYTES, build_call_or_panic, build_deep_sum,
    build_identifier, build_literal, build_text_literal,
};
use fhy_core::expr::pattern::{
    CallbackError, MatchBindings, Pattern, PatternError, does_pattern_match, match_pattern,
};
use fhy_core::expr::{
    BigInt, BinaryOperation, Expression, ExpressionBuildError, ExpressionKind, LiteralValue,
    UnaryOperation, build_piecewise,
};
use hashing_support::hash_of;
use pattern_support::{
    ProbeError, build_alternatives, build_capture, build_capture_of, build_literal_pattern,
    build_piecewise_pattern, expect_bound, expect_match, expect_probe_error, match_infallibly,
};
use rstest::rstest;
use stack_support::{SMALL_STACK_DEPTH, run_on_small_stack, run_on_stack};

/// Return the bound names of `bindings` in binding order.
fn collect_names(bindings: &MatchBindings) -> Vec<&str> {
    bindings.names().collect()
}

/// Return `bindings` with `name` bound to `expression`, failing the test if
/// the binding is refused.
fn bind(bindings: &MatchBindings, name: &str, expression: &Expression) -> MatchBindings {
    bindings
        .try_bind(name, expression)
        .unwrap_or_else(|| panic!("binding {name} to {expression:?} is refused"))
}

/// Return `1 op 2`.
fn build_simple_binary(operation: BinaryOperation) -> Expression {
    Expression::new_binary(operation, build_literal(1), build_literal(2))
}

/// Return a one-case piecewise `true -> 1, otherwise 2`.
fn build_one_case_piecewise() -> Expression {
    build_piecewise([(build_literal(true), build_literal(1))], build_literal(2))
        .expect("a valid piecewise")
}

/// Return a two-case piecewise `true -> 1, false -> 2, otherwise 3`.
fn build_two_case_piecewise() -> Expression {
    build_piecewise(
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
        Ok(verdict)
    })
}

/// Return the pattern matching `build_deep_sum(leaf, depth)` exactly, with
/// the leaf matched by `leaf_pattern`.
fn build_deep_sum_pattern(leaf_pattern: Pattern, depth: usize) -> Pattern {
    let mut pattern = leaf_pattern;
    for _ in 0..depth {
        pattern = Pattern::binary(
            Some(BinaryOperation::Add),
            pattern,
            build_literal_pattern(1),
        );
    }
    pattern
}

// =============================================================================
// MatchBindings
// =============================================================================

/// Test empty bindings bind no name.
#[test]
fn match_bindings_empty_binds_no_name() {
    let bindings = MatchBindings::empty();

    assert!(bindings.is_empty());
    assert_eq!(collect_names(&bindings), Vec::<&str>::new());
    assert!(bindings.get("x").is_none());
}

/// Test binding an unbound name records a handle to the given node.
#[test]
fn match_bindings_try_bind_records_an_unbound_name() {
    let expression = build_literal(5);

    let bound = bind(&MatchBindings::empty(), "x", &expression);

    assert!(!bound.is_empty());
    assert!(bound.has("x"));
    assert!(Expression::ptr_eq(expect_bound(&bound, "x"), &expression));
}

/// Test binding leaves the receiver unchanged.
#[test]
fn match_bindings_try_bind_leaves_the_receiver_unchanged() {
    let bindings = MatchBindings::empty();

    let bound = bind(&bindings, "x", &build_literal(1));

    assert!(bound.has("x"));
    assert!(!bindings.has("x"));
    assert!(bindings.is_empty());
}

/// Test confirming a name with a structurally equal expression returns the
/// same bindings.
#[test]
fn match_bindings_try_bind_confirming_an_equal_expression_returns_the_same_bindings() {
    let bindings = bind(&MatchBindings::empty(), "x", &build_literal(5));

    let rebound = bindings.try_bind("x", &build_literal(5));

    let rebound = rebound.expect("an equal expression confirms the binding");
    assert_eq!(rebound, bindings);
    assert_eq!(collect_names(&rebound), vec!["x"]);
}

/// Test confirming a binding keeps the first-bound handle.
#[test]
fn match_bindings_try_bind_confirming_keeps_the_first_bound_handle() {
    let original = build_literal(5);
    let later = build_literal(5);
    let bindings = bind(&MatchBindings::empty(), "x", &original);

    let rebound = bind(&bindings, "x", &later);

    assert!(Expression::ptr_eq(expect_bound(&rebound, "x"), &original));
    assert!(!Expression::ptr_eq(expect_bound(&rebound, "x"), &later));
}

/// Test binding a bound name to a different expression is refused.
#[test]
fn match_bindings_try_bind_refuses_a_different_expression() {
    let bindings = bind(&MatchBindings::empty(), "x", &build_literal(5));

    let rebound = bindings.try_bind("x", &build_literal(6));

    assert!(rebound.is_none(), "got {rebound:?}");
}

/// Test a structurally equal compound confirms a binding.
#[test]
fn match_bindings_try_bind_confirms_a_structurally_equal_compound() {
    let expression = build_simple_binary(BinaryOperation::Add);
    let equal = build_simple_binary(BinaryOperation::Add);
    let bindings = bind(&MatchBindings::empty(), "x", &expression);

    let rebound = bindings.try_bind("x", &equal);

    let rebound = rebound.expect("an equal compound confirms the binding");
    assert!(Expression::ptr_eq(expect_bound(&rebound, "x"), &expression));
}

/// Test confirmation uses literal equality: `5` confirms a binding to the
/// integer text `"05"`, and a NaN confirms a binding to a NaN.
#[rstest]
#[case::integer_and_integer_text(build_text_literal("05"), build_literal(5))]
#[case::nan_and_nan(build_literal(f64::NAN), build_literal(f64::NAN))]
#[case::zero_and_negative_zero(build_literal(0.0), build_literal(-0.0))]
fn match_bindings_try_bind_confirms_an_equal_literal_in_another_form(
    #[case] first: Expression,
    #[case] second: Expression,
) {
    let bindings = bind(&MatchBindings::empty(), "x", &first);

    let rebound = bindings.try_bind("x", &second);

    let rebound = rebound.expect("equal literals confirm the binding");
    assert!(Expression::ptr_eq(expect_bound(&rebound, "x"), &first));
}

/// Test a literal of another bucket does not confirm a binding.
#[rstest]
#[case::integer_and_float(build_literal(1), build_literal(1.0))]
#[case::integer_and_bool(build_literal(1), build_literal(true))]
#[case::float_and_decimal_text(build_literal(1.5), build_text_literal("1.5"))]
fn match_bindings_try_bind_refuses_a_literal_of_another_bucket(
    #[case] first: Expression,
    #[case] second: Expression,
) {
    let bindings = bind(&MatchBindings::empty(), "x", &first);

    let rebound = bindings.try_bind("x", &second);

    assert!(rebound.is_none(), "got {rebound:?}");
}

/// Test `get` returns `None` for an unbound name.
#[test]
fn match_bindings_get_returns_none_for_an_unbound_name() {
    let bindings = bind(&MatchBindings::empty(), "y", &build_literal(0));

    let bound = bindings.get("x");

    assert!(bound.is_none(), "got {bound:?}");
}

/// Test `has` reports a name only once it is bound.
#[test]
fn match_bindings_has_reports_only_bound_names() {
    let bindings = MatchBindings::empty();

    let bound = bind(&bindings, "x", &build_literal(0));

    assert!(!bindings.has("x"));
    assert!(bound.has("x"));
    assert!(!bound.has("y"));
}

/// Test `names` lists every bound name once, in binding order.
#[test]
fn match_bindings_names_lists_names_in_binding_order() {
    let step_one = bind(&MatchBindings::empty(), "y", &build_literal(1));
    let step_two = bind(&step_one, "x", &build_literal(2));

    let step_three = bind(&step_two, "y", &build_literal(1));

    assert_eq!(collect_names(&step_three), vec!["y", "x"]);
}

/// Test bindings of the same names to equal expressions are equal and hash
/// equally.
#[test]
fn match_bindings_with_equal_content_are_equal_and_hash_equally() {
    let left = bind(&MatchBindings::empty(), "x", &build_literal(7));
    let right = bind(&MatchBindings::empty(), "x", &build_literal(7));

    assert_eq!(left, right);
    assert_eq!(hash_of(&left), hash_of(&right));
}

/// Test equality compares bound expressions structurally, with literal
/// equality.
#[test]
fn match_bindings_equality_compares_expressions_structurally() {
    let left = bind(&MatchBindings::empty(), "x", &build_literal(5));
    let right = bind(&MatchBindings::empty(), "x", &build_text_literal("05"));

    assert_eq!(left, right);
    assert_eq!(hash_of(&left), hash_of(&right));
}

/// Test bindings of a name to different expressions are unequal.
#[test]
fn match_bindings_with_different_expressions_are_unequal() {
    let left = bind(&MatchBindings::empty(), "x", &build_literal(1));
    let right = bind(&MatchBindings::empty(), "x", &build_literal(2));

    assert_ne!(left, right);
}

/// Test equality ignores binding order, and so does hashing.
#[test]
fn match_bindings_equality_ignores_binding_order() {
    let x_first = bind(
        &bind(&MatchBindings::empty(), "x", &build_literal(1)),
        "y",
        &build_literal(2),
    );
    let y_first = bind(
        &bind(&MatchBindings::empty(), "y", &build_literal(2)),
        "x",
        &build_literal(1),
    );

    assert_eq!(x_first, y_first);
    assert_eq!(hash_of(&x_first), hash_of(&y_first));
}

/// Test bindings of different name sets are unequal.
#[rstest]
#[case::disjoint_names(&[("x", 1)], &[("y", 1)])]
#[case::one_name_more(&[("x", 1)], &[("x", 1), ("y", 2)])]
#[case::empty_and_bound(&[], &[("x", 1)])]
fn match_bindings_with_different_names_are_unequal(
    #[case] left_entries: &[(&str, i64)],
    #[case] right_entries: &[(&str, i64)],
) {
    let build = |entries: &[(&str, i64)]| {
        entries
            .iter()
            .fold(MatchBindings::empty(), |bindings, (name, value)| {
                bind(&bindings, name, &build_literal(*value))
            })
    };

    let (left, right) = (build(left_entries), build(right_entries));

    assert_ne!(left, right);
    assert_ne!(right, left);
}

// =============================================================================
// Wildcard
// =============================================================================

/// Test the wildcard matches a node of every kind and captures nothing.
#[rstest]
fn pattern_wildcard_matches_every_node_kind(
    #[values(
        NodeKind::Literal,
        NodeKind::Identifier,
        NodeKind::Unary,
        NodeKind::Binary,
        NodeKind::Piecewise,
        NodeKind::Call
    )]
    kind: NodeKind,
) {
    let pattern = Pattern::wildcard();

    let bindings = match_infallibly(&pattern, &kind.build());

    let bindings = bindings.expect("the wildcard matches");
    assert!(bindings.is_empty(), "bound {bindings:?}");
}

/// Test the wildcard returns the bindings it was given.
#[test]
fn pattern_wildcard_match_under_returns_the_given_bindings() {
    let bound = build_literal(0);
    let starting = bind(&MatchBindings::empty(), "a", &bound);

    let result = Pattern::wildcard().match_under(&build_literal(1), &starting);

    let result = result.expect("no callback").expect("the wildcard matches");
    assert_eq!(result, starting);
    assert_eq!(collect_names(&result), vec!["a"]);
    assert!(Expression::ptr_eq(expect_bound(&result, "a"), &bound));
}

// =============================================================================
// Capture
// =============================================================================

/// Test a capture of the wildcard binds a handle to the matched node.
#[test]
fn pattern_capture_binds_the_matched_node() {
    let expression = build_simple_binary(BinaryOperation::Add);

    let bindings = expect_match(&build_capture("x"), &expression);

    assert!(Expression::ptr_eq(
        expect_bound(&bindings, "x"),
        &expression
    ));
}

/// Test a capture binds under exactly the given name.
#[test]
fn pattern_capture_binds_under_the_given_name() {
    let bindings = expect_match(&build_capture("x"), &build_literal(5));

    assert_eq!(collect_names(&bindings), vec!["x"]);
}

/// Test an empty capture name is refused.
#[test]
fn pattern_capture_rejects_an_empty_name() {
    let result = Pattern::capture("", Pattern::wildcard());

    assert!(
        matches!(result, Err(PatternError::EmptyCaptureName)),
        "got {result:?}"
    );
}

/// Test two sibling captures of one name share one binding.
#[test]
fn pattern_capture_shared_by_siblings_binds_one_name() {
    let pattern = Pattern::binary(
        Some(BinaryOperation::Subtract),
        build_capture("x"),
        build_capture("x"),
    );
    let left = build_literal(5);
    let expression = Expression::new_binary(BinaryOperation::Subtract, &left, build_literal(5));

    let bindings = expect_match(&pattern, &expression);

    assert_eq!(collect_names(&bindings), vec!["x"]);
    assert!(Expression::ptr_eq(expect_bound(&bindings, "x"), &left));
}

/// Test a capture fails when its sub-pattern fails.
#[test]
fn pattern_capture_fails_when_the_sub_pattern_fails() {
    let pattern = build_capture_of("x", build_literal_pattern(5));

    let result = match_infallibly(&pattern, &build_literal(6));

    assert!(result.is_none(), "got {result:?}");
}

/// Test a capture binds when its sub-pattern matches.
#[test]
fn pattern_capture_binds_when_the_sub_pattern_matches() {
    let pattern = build_capture_of("x", build_literal_pattern(5));
    let expression = build_literal(5);

    let bindings = expect_match(&pattern, &expression);

    assert!(Expression::ptr_eq(
        expect_bound(&bindings, "x"),
        &expression
    ));
}

/// Test a name captured over equal operands matches.
#[test]
fn pattern_capture_repeated_over_equal_operands_matches() {
    let pattern = Pattern::binary(
        Some(BinaryOperation::Subtract),
        build_capture("x"),
        build_capture("x"),
    );
    let expression = Expression::new_binary(
        BinaryOperation::Subtract,
        build_literal(5),
        build_literal(5),
    );

    let bindings = expect_match(&pattern, &expression);

    assert_eq!(expect_bound(&bindings, "x"), &build_literal(5));
}

/// Test a name captured over different operands fails.
#[test]
fn pattern_capture_repeated_over_different_operands_fails() {
    let pattern = Pattern::binary(
        Some(BinaryOperation::Subtract),
        build_capture("x"),
        build_capture("x"),
    );
    let expression = Expression::new_binary(
        BinaryOperation::Subtract,
        build_literal(5),
        build_literal(6),
    );

    let result = match_infallibly(&pattern, &expression);

    assert!(result.is_none(), "got {result:?}");
}

/// Test a repeated capture over equal literals in different forms matches
/// and keeps the first operand: `5 - "5"` and `NaN - NaN`.
#[rstest]
#[case::integer_and_integer_text(build_literal(5), build_text_literal("5"))]
#[case::nan_and_nan(build_literal(f64::NAN), build_literal(f64::NAN))]
fn pattern_capture_repeated_over_equal_literals_keeps_the_first(
    #[case] left: Expression,
    #[case] right: Expression,
) {
    let pattern = Pattern::binary(None, build_capture("x"), build_capture("x"));
    let expression = Expression::new_binary(BinaryOperation::Subtract, &left, &right);

    let bindings = expect_match(&pattern, &expression);

    assert!(Expression::ptr_eq(expect_bound(&bindings, "x"), &left));
}

/// Test nested captures bind the inner name before the outer one, both to
/// the matched node.
#[test]
fn pattern_capture_nested_binds_inner_before_outer() {
    let pattern = build_capture_of("outer", build_capture("inner"));
    let expression = build_literal(5);

    let bindings = expect_match(&pattern, &expression);

    assert_eq!(collect_names(&bindings), vec!["inner", "outer"]);
    assert!(Expression::ptr_eq(
        expect_bound(&bindings, "outer"),
        &expression
    ));
    assert!(Expression::ptr_eq(
        expect_bound(&bindings, "inner"),
        &expression
    ));
}

/// Test bindings are recorded in completion order: a compound's captures in
/// matching order, then the capture around it.
#[test]
fn pattern_capture_records_bindings_in_completion_order() {
    let pattern = build_capture_of(
        "outer",
        Pattern::binary(None, build_capture("b"), build_capture("a")),
    );

    let bindings = expect_match(&pattern, &build_simple_binary(BinaryOperation::Add));

    assert_eq!(collect_names(&bindings), vec!["b", "a", "outer"]);
}

// =============================================================================
// Literal
// =============================================================================

/// Test a literal pattern without a value matches a literal of every stored
/// form.
#[rstest]
#[case::integer(build_literal(5))]
#[case::float(build_literal(2.75))]
#[case::boolean(build_literal(true))]
#[case::integer_text(build_text_literal("05"))]
#[case::decimal_text(build_text_literal("1.50"))]
#[case::nan(build_literal(f64::NAN))]
fn pattern_literal_without_value_matches_every_literal(#[case] expression: Expression) {
    let bindings = match_infallibly(&Pattern::literal(None), &expression);

    let bindings = bindings.expect("any literal matches");
    assert!(bindings.is_empty());
}

/// Test a literal pattern rejects every node that is not a literal.
#[rstest]
fn pattern_literal_rejects_non_literal_nodes(
    #[values(
        NodeKind::Literal,
        NodeKind::Identifier,
        NodeKind::Unary,
        NodeKind::Binary,
        NodeKind::Piecewise,
        NodeKind::Call
    )]
    kind: NodeKind,
) {
    let pattern = Pattern::literal(None);

    let result = match_infallibly(&pattern, &kind.build());

    assert_eq!(
        result.is_some(),
        kind == NodeKind::Literal,
        "got {result:?}"
    );
}

/// Test a literal pattern with a value matches a literal stored exactly so.
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
    build_text_literal("05")
)]
#[case::decimal_text(
    LiteralValue::parse_text("1.50").expect("a text"),
    build_text_literal("1.50")
)]
fn pattern_literal_matches_an_exactly_stored_value(
    #[case] value: LiteralValue,
    #[case] expression: Expression,
) {
    let result = match_infallibly(&Pattern::literal(Some(value)), &expression);

    let bindings = result.expect("the stored value matches");
    assert!(bindings.is_empty());
}

/// Test a literal pattern with a value rejects a literal stored in another
/// form or with another value, even an equal literal.
#[rstest]
#[case::other_integer(LiteralValue::from(5), build_literal(6))]
#[case::integer_and_float(LiteralValue::from(5), build_literal(5.0))]
#[case::integer_and_bool(LiteralValue::from(1), build_literal(true))]
#[case::bool_and_integer(LiteralValue::from(true), build_literal(1))]
#[case::integer_and_integer_text(LiteralValue::from(5), build_text_literal("5"))]
#[case::integer_text_and_integer(
    LiteralValue::parse_text("5").expect("a text"),
    build_literal(5)
)]
#[case::integer_texts_spelled_differently(
    LiteralValue::parse_text("5").expect("a text"),
    build_text_literal("05")
)]
#[case::decimal_texts_spelled_differently(
    LiteralValue::parse_text("1.5").expect("a text"),
    build_text_literal("1.50")
)]
#[case::float_and_decimal_text(LiteralValue::from(1.5), build_text_literal("1.5"))]
#[case::nan_and_nan(LiteralValue::from(f64::NAN), build_literal(f64::NAN))]
fn pattern_literal_rejects_another_stored_form(
    #[case] value: LiteralValue,
    #[case] expression: Expression,
) {
    let result = match_infallibly(&Pattern::literal(Some(value)), &expression);

    assert!(result.is_none(), "got {result:?}");
}

// =============================================================================
// Identifier
// =============================================================================

/// Test an identifier pattern without an identifier matches any reference.
#[test]
fn pattern_identifier_without_identifier_matches_any_reference() {
    let (_, x) = build_identifier("x");
    let (_, y) = build_identifier("y");

    let x_result = match_infallibly(&Pattern::identifier(None), &x);
    let y_result = match_infallibly(&Pattern::identifier(None), &y);

    assert!(x_result.is_some_and(|bindings| bindings.is_empty()));
    assert!(y_result.is_some_and(|bindings| bindings.is_empty()));
}

/// Test an identifier pattern rejects every node that is not a reference.
#[rstest]
fn pattern_identifier_rejects_non_reference_nodes(
    #[values(
        NodeKind::Literal,
        NodeKind::Identifier,
        NodeKind::Unary,
        NodeKind::Binary,
        NodeKind::Piecewise,
        NodeKind::Call
    )]
    kind: NodeKind,
) {
    let pattern = Pattern::identifier(None);

    let result = match_infallibly(&pattern, &kind.build());

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
    let pattern = Pattern::identifier(Some(identifier.clone()));

    let direct = match_infallibly(&pattern, &reference);
    let through_clone = match_infallibly(&pattern, &Expression::from(identifier));

    assert!(direct.is_some());
    assert!(through_clone.is_some());
}

/// Test an identifier pattern rejects a different identifier with the same
/// name hint.
#[test]
fn pattern_identifier_rejects_a_different_identifier_with_the_same_hint() {
    let (first, _) = build_identifier("x");
    let (_, second_reference) = build_identifier("x");
    let pattern = Pattern::identifier(Some(first));

    let result = match_infallibly(&pattern, &second_reference);

    assert!(result.is_none(), "got {result:?}");
}

// =============================================================================
// Unary
// =============================================================================

/// Test a unary pattern matches its operation.
#[test]
fn pattern_unary_matches_its_operation() {
    let pattern = Pattern::unary(Some(UnaryOperation::Negate), Pattern::wildcard());

    let result = match_infallibly(&pattern, &-build_literal(5));

    assert!(result.is_some());
}

/// Test a unary pattern rejects another operation.
#[test]
fn pattern_unary_rejects_another_operation() {
    let pattern = Pattern::unary(Some(UnaryOperation::Negate), Pattern::wildcard());
    let expression = Expression::new_unary(UnaryOperation::LogicalNot, build_literal(5));

    let result = match_infallibly(&pattern, &expression);

    assert!(result.is_none(), "got {result:?}");
}

/// Test a unary pattern without an operation matches every operation.
#[rstest]
#[case::negate(UnaryOperation::Negate)]
#[case::positive(UnaryOperation::Positive)]
#[case::logical_not(UnaryOperation::LogicalNot)]
fn pattern_unary_without_operation_matches_every_operation(#[case] operation: UnaryOperation) {
    let pattern = Pattern::unary(None, Pattern::wildcard());

    let result = match_infallibly(
        &pattern,
        &Expression::new_unary(operation, build_literal(5)),
    );

    assert!(result.is_some());
}

/// Test a unary pattern rejects every node that is not unary.
#[rstest]
fn pattern_unary_rejects_other_node_kinds(
    #[values(
        NodeKind::Literal,
        NodeKind::Identifier,
        NodeKind::Unary,
        NodeKind::Binary,
        NodeKind::Piecewise,
        NodeKind::Call
    )]
    kind: NodeKind,
) {
    let pattern = Pattern::unary(None, Pattern::wildcard());

    let result = match_infallibly(&pattern, &kind.build());

    assert_eq!(result.is_some(), kind == NodeKind::Unary, "got {result:?}");
}

/// Test a unary pattern binds the captures of its operand pattern.
#[test]
fn pattern_unary_binds_captures_in_its_operand() {
    let pattern = Pattern::unary(Some(UnaryOperation::Negate), build_capture("x"));
    let operand = build_literal(5);

    let bindings = expect_match(&pattern, &-&operand);

    assert!(Expression::ptr_eq(expect_bound(&bindings, "x"), &operand));
}

/// Test a unary pattern fails when its operand pattern fails.
#[test]
fn pattern_unary_fails_when_its_operand_fails() {
    let pattern = Pattern::unary(Some(UnaryOperation::Negate), build_literal_pattern(5));

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
        Some(BinaryOperation::Add),
        Pattern::wildcard(),
        Pattern::wildcard(),
    );

    let result = match_infallibly(&pattern, &build_simple_binary(BinaryOperation::Add));

    assert!(result.is_some());
}

/// Test a binary pattern rejects another operation.
#[test]
fn pattern_binary_rejects_another_operation() {
    let pattern = Pattern::binary(
        Some(BinaryOperation::Add),
        Pattern::wildcard(),
        Pattern::wildcard(),
    );

    let result = match_infallibly(&pattern, &build_simple_binary(BinaryOperation::Subtract));

    assert!(result.is_none(), "got {result:?}");
}

/// Test a binary pattern without an operation matches every operation.
#[rstest]
#[case::add(BinaryOperation::Add)]
#[case::multiply(BinaryOperation::Multiply)]
#[case::power(BinaryOperation::Power)]
#[case::logical_or(BinaryOperation::LogicalOr)]
#[case::greater_equal(BinaryOperation::GreaterEqual)]
fn pattern_binary_without_operation_matches_every_operation(#[case] operation: BinaryOperation) {
    let pattern = Pattern::binary(None, Pattern::wildcard(), Pattern::wildcard());

    let result = match_infallibly(&pattern, &build_simple_binary(operation));

    assert!(result.is_some());
}

/// Test a binary pattern rejects every node that is not binary.
#[rstest]
fn pattern_binary_rejects_other_node_kinds(
    #[values(
        NodeKind::Literal,
        NodeKind::Identifier,
        NodeKind::Unary,
        NodeKind::Binary,
        NodeKind::Piecewise,
        NodeKind::Call
    )]
    kind: NodeKind,
) {
    let pattern = Pattern::binary(None, Pattern::wildcard(), Pattern::wildcard());

    let result = match_infallibly(&pattern, &kind.build());

    assert_eq!(result.is_some(), kind == NodeKind::Binary, "got {result:?}");
}

/// Test a binary pattern binds the captures of both operand patterns, left
/// first.
#[test]
fn pattern_binary_binds_captures_in_both_operands() {
    let pattern = Pattern::binary(
        Some(BinaryOperation::Add),
        build_capture("a"),
        build_capture("b"),
    );
    let left = build_literal(1);
    let right = build_literal(2);

    let bindings = expect_match(
        &pattern,
        &Expression::new_binary(BinaryOperation::Add, &left, &right),
    );

    assert_eq!(collect_names(&bindings), vec!["a", "b"]);
    assert!(Expression::ptr_eq(expect_bound(&bindings, "a"), &left));
    assert!(Expression::ptr_eq(expect_bound(&bindings, "b"), &right));
}

/// Test a binary pattern fails when its left operand pattern fails.
#[test]
fn pattern_binary_fails_when_its_left_operand_fails() {
    let pattern = Pattern::binary(
        Some(BinaryOperation::Add),
        build_literal_pattern(99),
        Pattern::wildcard(),
    );

    let result = match_infallibly(&pattern, &build_simple_binary(BinaryOperation::Add));

    assert!(result.is_none(), "got {result:?}");
}

/// Test a binary pattern fails when its right operand pattern fails.
#[test]
fn pattern_binary_fails_when_its_right_operand_fails() {
    let pattern = Pattern::binary(
        Some(BinaryOperation::Add),
        Pattern::wildcard(),
        build_literal_pattern(99),
    );

    let result = match_infallibly(&pattern, &build_simple_binary(BinaryOperation::Add));

    assert!(result.is_none(), "got {result:?}");
}

/// Test a binary pattern whose left operand fails does not try its right
/// operand.
#[test]
fn pattern_binary_skips_the_right_operand_after_a_left_failure() {
    let calls = Arc::new(AtomicUsize::new(0));
    let pattern = Pattern::binary(
        None,
        build_literal_pattern(99),
        build_counting_predicate(&calls, true),
    );

    let result = match_infallibly(&pattern, &build_simple_binary(BinaryOperation::Add));

    assert!(result.is_none(), "got {result:?}");
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

// =============================================================================
// Piecewise
// =============================================================================

/// Test a piecewise pattern with an empty case list is refused.
#[test]
fn pattern_piecewise_rejects_an_empty_case_list() {
    let result = Pattern::piecewise(Some(Vec::new()), Pattern::wildcard());

    assert!(
        matches!(result, Err(PatternError::EmptyPiecewiseCases)),
        "got {result:?}"
    );
}

/// Test a piecewise pattern with one case pattern matches a one-case
/// piecewise.
#[test]
fn pattern_piecewise_matches_a_one_case_piecewise() {
    let pattern = build_piecewise_pattern(
        Some(vec![(Pattern::wildcard(), Pattern::wildcard())]),
        Pattern::wildcard(),
    );

    let result = match_infallibly(&pattern, &build_one_case_piecewise());

    assert!(result.is_some());
}

/// Test a piecewise pattern rejects every node that is not a piecewise.
#[rstest]
fn pattern_piecewise_rejects_other_node_kinds(
    #[values(
        NodeKind::Literal,
        NodeKind::Identifier,
        NodeKind::Unary,
        NodeKind::Binary,
        NodeKind::Piecewise,
        NodeKind::Call
    )]
    kind: NodeKind,
) {
    let pattern = build_piecewise_pattern(None, Pattern::wildcard());

    let result = match_infallibly(&pattern, &kind.build());

    assert_eq!(
        result.is_some(),
        kind == NodeKind::Piecewise,
        "got {result:?}"
    );
}

/// Test a piecewise pattern without cases matches any case count and
/// matches only the otherwise branch.
#[test]
fn pattern_piecewise_without_cases_matches_any_case_count() {
    let pattern = build_piecewise_pattern(None, build_capture("o"));
    let one_case = build_one_case_piecewise();
    let two_cases = build_two_case_piecewise();

    let one_case_bindings = expect_match(&pattern, &one_case);
    let two_case_bindings = expect_match(&pattern, &two_cases);

    assert_eq!(collect_names(&one_case_bindings), vec!["o"]);
    assert_eq!(expect_bound(&one_case_bindings, "o"), &build_literal(2));
    assert_eq!(collect_names(&two_case_bindings), vec!["o"]);
    assert_eq!(expect_bound(&two_case_bindings, "o"), &build_literal(3));
}

/// Test a piecewise pattern with case patterns rejects a different case
/// count.
#[test]
fn pattern_piecewise_rejects_a_different_case_count() {
    let pattern = build_piecewise_pattern(
        Some(vec![(Pattern::wildcard(), Pattern::wildcard())]),
        Pattern::wildcard(),
    );

    let result = match_infallibly(&pattern, &build_two_case_piecewise());

    assert!(result.is_none(), "got {result:?}");
}

/// Test a piecewise pattern binds the condition, the value and the
/// otherwise branch.
#[test]
fn pattern_piecewise_binds_condition_value_and_otherwise() {
    let pattern = build_piecewise_pattern(
        Some(vec![(build_capture("c"), build_capture("v"))]),
        build_capture("o"),
    );
    let (condition, value, otherwise) = (build_literal(true), build_literal(1), build_literal(2));
    let expression =
        build_piecewise([(&condition, &value)], &otherwise).expect("a valid piecewise");

    let bindings = expect_match(&pattern, &expression);

    assert_eq!(collect_names(&bindings), vec!["c", "v", "o"]);
    assert!(Expression::ptr_eq(expect_bound(&bindings, "c"), &condition));
    assert!(Expression::ptr_eq(expect_bound(&bindings, "v"), &value));
    assert!(Expression::ptr_eq(expect_bound(&bindings, "o"), &otherwise));
}

/// Test a piecewise pattern binds every case in evaluation order,
/// condition before value.
#[test]
fn pattern_piecewise_binds_cases_in_evaluation_order() {
    let pattern = build_piecewise_pattern(
        Some(vec![
            (build_capture("c1"), build_capture("v1")),
            (build_capture("c2"), build_capture("v2")),
        ]),
        Pattern::wildcard(),
    );
    let (c1, c2) = (build_literal(true), build_literal(false));
    let (v1, v2) = (build_literal(1), build_literal(2));
    let expression =
        build_piecewise([(&c1, &v1), (&c2, &v2)], build_literal(0)).expect("a valid piecewise");

    let bindings = expect_match(&pattern, &expression);

    assert_eq!(collect_names(&bindings), vec!["c1", "v1", "c2", "v2"]);
    assert!(Expression::ptr_eq(expect_bound(&bindings, "c1"), &c1));
    assert!(Expression::ptr_eq(expect_bound(&bindings, "v1"), &v1));
    assert!(Expression::ptr_eq(expect_bound(&bindings, "c2"), &c2));
    assert!(Expression::ptr_eq(expect_bound(&bindings, "v2"), &v2));
}

/// Test a piecewise pattern fails when a condition pattern fails.
#[test]
fn pattern_piecewise_fails_when_a_condition_fails() {
    let pattern = build_piecewise_pattern(
        Some(vec![(build_literal_pattern(99), Pattern::wildcard())]),
        Pattern::wildcard(),
    );

    let result = match_infallibly(&pattern, &build_one_case_piecewise());

    assert!(result.is_none(), "got {result:?}");
}

/// Test a piecewise pattern fails when a value pattern fails.
#[test]
fn pattern_piecewise_fails_when_a_value_fails() {
    let pattern = build_piecewise_pattern(
        Some(vec![(Pattern::wildcard(), build_literal_pattern(99))]),
        Pattern::wildcard(),
    );

    let result = match_infallibly(&pattern, &build_one_case_piecewise());

    assert!(result.is_none(), "got {result:?}");
}

/// Test a piecewise pattern fails when its otherwise pattern fails, although
/// every case matches.
#[test]
fn pattern_piecewise_fails_when_the_otherwise_branch_fails() {
    let pattern = build_piecewise_pattern(
        Some(vec![(Pattern::wildcard(), Pattern::wildcard())]),
        build_literal_pattern(99),
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
    let pattern = build_piecewise_pattern(
        Some(vec![(Pattern::wildcard(), build_capture("v"))]),
        build_capture("v"),
    );
    let expression =
        build_piecewise([(build_literal(true), build_literal(1))], otherwise).expect("a piecewise");

    let result = match_infallibly(&pattern, &expression);

    assert_eq!(result.is_some(), expected_match, "got {result:?}");
}

// =============================================================================
// Call
// =============================================================================

/// Test a call pattern matches its function name.
#[test]
fn pattern_call_matches_its_function_name() {
    let pattern = Pattern::call(Some("f"), Some(vec![Pattern::wildcard()]));

    let result = match_infallibly(&pattern, &build_call_or_panic("f", vec![build_literal(1)]));

    assert!(result.is_some());
}

/// Test a call pattern rejects another function name.
#[test]
fn pattern_call_rejects_another_function_name() {
    let pattern = Pattern::call(Some("f"), Some(vec![Pattern::wildcard()]));

    let result = match_infallibly(&pattern, &build_call_or_panic("g", vec![build_literal(1)]));

    assert!(result.is_none(), "got {result:?}");
}

/// Test a call pattern without a function name matches any name.
#[rstest]
#[case::f("f")]
#[case::g("g")]
#[case::long_name("a_function_name")]
fn pattern_call_without_function_name_matches_any_name(#[case] function_name: &str) {
    let pattern = Pattern::call(None, Some(vec![Pattern::wildcard()]));

    let result = match_infallibly(
        &pattern,
        &build_call_or_panic(function_name, vec![build_literal(1)]),
    );

    assert!(result.is_some());
}

/// Test a call pattern with argument patterns rejects another arity.
#[rstest]
#[case::fewer_arguments(vec![build_literal(1)])]
#[case::more_arguments(vec![build_literal(1), build_literal(2), build_literal(3)])]
fn pattern_call_rejects_another_arity(#[case] arguments: Vec<Expression>) {
    let pattern = Pattern::call(
        Some("f"),
        Some(vec![Pattern::wildcard(), Pattern::wildcard()]),
    );

    let result = match_infallibly(&pattern, &build_call_or_panic("f", arguments));

    assert!(result.is_none(), "got {result:?}");
}

/// Test a call pattern without argument patterns matches any arity and
/// matches no argument.
#[rstest]
#[case::no_arguments(Vec::new())]
#[case::one_argument(vec![build_literal(1)])]
#[case::two_arguments(vec![build_literal(1), build_literal(2)])]
fn pattern_call_without_arguments_matches_any_arity(#[case] arguments: Vec<Expression>) {
    let pattern = Pattern::call(Some("f"), None);

    let result = match_infallibly(&pattern, &build_call_or_panic("f", arguments));

    assert!(result.is_some_and(|bindings| bindings.is_empty()));
}

/// Test a call pattern binds the captures of its argument patterns in
/// order.
#[test]
fn pattern_call_binds_captures_in_its_arguments() {
    let pattern = Pattern::call(
        Some("f"),
        Some(vec![build_capture("a"), build_capture("b")]),
    );
    let (first, second) = (build_literal(1), build_literal(2));

    let bindings = expect_match(
        &pattern,
        &build_call_or_panic("f", vec![first.clone(), second.clone()]),
    );

    assert_eq!(collect_names(&bindings), vec!["a", "b"]);
    assert!(Expression::ptr_eq(expect_bound(&bindings, "a"), &first));
    assert!(Expression::ptr_eq(expect_bound(&bindings, "b"), &second));
}

/// Test a call pattern rejects every node that is not a call.
#[rstest]
fn pattern_call_rejects_other_node_kinds(
    #[values(
        NodeKind::Literal,
        NodeKind::Identifier,
        NodeKind::Unary,
        NodeKind::Binary,
        NodeKind::Piecewise,
        NodeKind::Call
    )]
    kind: NodeKind,
) {
    let pattern = Pattern::call(None, None);

    let result = match_infallibly(&pattern, &kind.build());

    assert_eq!(result.is_some(), kind == NodeKind::Call, "got {result:?}");
}

/// Test a call pattern with an empty argument list matches only calls
/// without arguments.
#[test]
fn pattern_call_with_empty_arguments_matches_only_calls_without_arguments() {
    let pattern = Pattern::call(Some("f"), Some(Vec::new()));

    let without_arguments = match_infallibly(
        &pattern,
        &build_call_or_panic("f", Vec::<Expression>::new()),
    );
    let with_argument =
        match_infallibly(&pattern, &build_call_or_panic("f", vec![build_literal(1)]));

    assert!(without_arguments.is_some());
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
    let pattern = Pattern::call(
        Some("f"),
        Some(vec![build_capture("x"), build_capture("x")]),
    );
    let expression = build_call_or_panic("f", vec![build_literal(1), second_argument]);

    let result = match_infallibly(&pattern, &expression);

    assert_eq!(result.is_some(), expected_match, "got {result:?}");
}

// =============================================================================
// Predicate
// =============================================================================

/// Test a predicate pattern matches when the predicate answers true.
#[test]
fn pattern_predicate_matches_when_the_predicate_holds() {
    let pattern = Pattern::predicate(|_| Ok(true));

    let result = match_infallibly(&pattern, &build_literal(5));

    assert!(result.is_some());
}

/// Test a predicate pattern does not match when the predicate answers
/// false.
#[test]
fn pattern_predicate_rejects_when_the_predicate_does_not_hold() {
    let pattern = Pattern::predicate(|_| Ok(false));

    let result = match_infallibly(&pattern, &build_literal(5));

    assert!(result.is_none(), "got {result:?}");
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
        Ok(true)
    });
    let expression = build_literal(5);

    let result = match_infallibly(&pattern, &expression);

    assert!(result.is_some());
    let seen = seen.lock().expect("an unpoisoned lock");
    assert_eq!(seen.len(), 1);
    assert!(Expression::ptr_eq(&seen[0], &expression));
}

/// Test a failing predicate's error is returned from the match unchanged.
#[test]
fn pattern_predicate_error_is_returned_from_the_match() {
    let pattern = Pattern::predicate(|_| Err(CallbackError::new(ProbeError("predicate failed"))));

    let result = match_pattern(&pattern, &build_literal(5));

    let error = result.expect_err("the predicate fails");
    assert_eq!(expect_probe_error(&error), &ProbeError("predicate failed"));
}

/// Test a failing predicate ends the match: later sub-patterns are not
/// tried, not even in another alternative.
#[test]
fn pattern_predicate_error_stops_the_match() {
    let calls = Arc::new(AtomicUsize::new(0));
    let failing = Pattern::predicate(|_| Err(CallbackError::new(ProbeError("stop"))));
    let pattern = build_alternatives(vec![
        Pattern::binary(None, failing, build_counting_predicate(&calls, true)),
        build_counting_predicate(&calls, true),
    ]);

    let result = does_pattern_match(&pattern, &build_simple_binary(BinaryOperation::Add));

    let error = result.expect_err("the predicate fails");
    assert_eq!(expect_probe_error(&error), &ProbeError("stop"));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

/// Test a predicate pattern binds nothing.
#[test]
fn pattern_predicate_captures_nothing() {
    let starting = bind(&MatchBindings::empty(), "x", &build_literal(0));
    let pattern = Pattern::predicate(|_| Ok(true));

    let result = pattern.match_under(&build_literal(5), &starting);

    let result = result
        .expect("no callback fails")
        .expect("the predicate holds");
    assert_eq!(result, starting);
    assert_eq!(collect_names(&result), vec!["x"]);
}

/// Test a predicate filtering node kinds matches literals only.
#[rstest]
fn pattern_predicate_filters_node_kinds(
    #[values(
        NodeKind::Literal,
        NodeKind::Identifier,
        NodeKind::Unary,
        NodeKind::Binary,
        NodeKind::Piecewise,
        NodeKind::Call
    )]
    kind: NodeKind,
) {
    let pattern = Pattern::predicate(|expression| {
        Ok(matches!(expression.kind(), ExpressionKind::Literal(_)))
    });

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

    assert!(original_result.is_some());
    assert!(clone_result.is_some());
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

// =============================================================================
// Alternatives
// =============================================================================

/// Test an alternatives pattern without alternatives is refused.
#[test]
fn pattern_alternatives_rejects_an_empty_list() {
    let result = Pattern::alternatives(Vec::new());

    assert!(
        matches!(result, Err(PatternError::EmptyAlternatives)),
        "got {result:?}"
    );
}

/// Test the first matching alternative decides the bindings.
#[test]
fn pattern_alternatives_returns_the_first_match() {
    let pattern = build_alternatives(vec![
        build_capture_of("first", Pattern::literal(None)),
        build_capture_of("second", Pattern::wildcard()),
    ]);
    let expression = build_literal(5);

    let bindings = expect_match(&pattern, &expression);

    assert_eq!(collect_names(&bindings), vec!["first"]);
    assert!(Expression::ptr_eq(
        expect_bound(&bindings, "first"),
        &expression
    ));
}

/// Test a later alternative matches when the earlier ones fail.
#[test]
fn pattern_alternatives_falls_through_to_a_later_alternative() {
    let pattern = build_alternatives(vec![
        build_capture_of("x", Pattern::literal(None)),
        build_capture_of("x", Pattern::identifier(None)),
    ]);
    let (_, expression) = build_identifier("x");

    let bindings = expect_match(&pattern, &expression);

    assert!(Expression::ptr_eq(
        expect_bound(&bindings, "x"),
        &expression
    ));
}

/// Test an alternatives pattern fails when every alternative fails.
#[test]
fn pattern_alternatives_fails_when_every_alternative_fails() {
    let pattern = build_alternatives(vec![build_literal_pattern(5), build_literal_pattern(6)]);

    let result = match_infallibly(&pattern, &build_literal(7));

    assert!(result.is_none(), "got {result:?}");
}

/// Test a capture bound by an alternative that then fails does not
/// constrain the next alternative: in `1 + 2`, the first alternative binds
/// `x` to the left operand before its right operand fails to match, and the
/// second binds `x` to the right operand alone.
#[test]
fn pattern_alternatives_isolates_failed_attempts() {
    let pattern = build_alternatives(vec![
        Pattern::binary(
            Some(BinaryOperation::Add),
            build_capture("x"),
            build_literal_pattern(99),
        ),
        Pattern::binary(
            Some(BinaryOperation::Add),
            Pattern::wildcard(),
            build_capture("x"),
        ),
    ]);

    let bindings = expect_match(&pattern, &build_simple_binary(BinaryOperation::Add));

    assert_eq!(collect_names(&bindings), vec!["x"]);
    assert_eq!(expect_bound(&bindings, "x"), &build_literal(2));
}

/// Test a capture made inside a failed alternative is discarded.
#[test]
fn pattern_alternatives_discards_captures_of_a_failed_alternative() {
    let pattern = build_alternatives(vec![
        Pattern::binary(
            Some(BinaryOperation::Add),
            build_capture_of("captured_in_failed", build_literal_pattern(1)),
            build_literal_pattern(99),
        ),
        build_capture("captured_in_successful"),
    ]);

    let bindings = expect_match(&pattern, &build_simple_binary(BinaryOperation::Add));

    assert_eq!(collect_names(&bindings), vec!["captured_in_successful"]);
}

/// Test an alternative sees the bindings made before the alternatives
/// pattern, and its captures are visible to later siblings.
#[test]
fn pattern_alternatives_threads_bindings_through_the_chosen_alternative() {
    let pattern = Pattern::binary(
        None,
        build_capture("x"),
        build_alternatives(vec![
            build_capture_of("x", build_literal_pattern(2)),
            build_capture_of("y", Pattern::wildcard()),
        ]),
    );

    let bindings = expect_match(&pattern, &build_simple_binary(BinaryOperation::Add));

    assert_eq!(collect_names(&bindings), vec!["x", "y"]);
    assert_eq!(expect_bound(&bindings, "y"), &build_literal(2));
}

/// Test the choice of an alternative is final: when a later sibling fails,
/// the remaining alternatives are not tried.
#[test]
fn pattern_alternatives_commits_to_the_first_match() {
    let pattern = Pattern::binary(
        None,
        build_alternatives(vec![
            build_capture_of("x", Pattern::literal(None)),
            Pattern::wildcard(),
        ]),
        build_capture("x"),
    );

    let result = match_infallibly(&pattern, &build_simple_binary(BinaryOperation::Add));

    assert!(result.is_none(), "got {result:?}");
}

// =============================================================================
// Free functions
// =============================================================================

/// Test `match_pattern` gives the result of matching from empty bindings.
#[rstest]
#[case::matching(build_literal(5))]
#[case::not_matching(build_literal(6))]
fn match_pattern_equals_match_under_from_empty_bindings(#[case] expression: Expression) {
    let pattern = build_capture_of("x", build_literal_pattern(5));

    let from_function = match_pattern(&pattern, &expression).expect("no callback");
    let from_method = pattern
        .match_under(&expression, &MatchBindings::empty())
        .expect("no callback");

    assert_eq!(from_function, from_method);
}

/// Test `match_pattern` returns `None` for a mismatch.
#[test]
fn match_pattern_returns_none_on_a_mismatch() {
    let result = match_infallibly(&build_literal_pattern(5), &build_literal(6));

    assert!(result.is_none(), "got {result:?}");
}

/// Test `does_pattern_match` answers true for a match.
#[test]
fn does_pattern_match_is_true_on_a_match() {
    let result = does_pattern_match(&build_literal_pattern(5), &build_literal(5));

    assert!(result.expect("no callback"));
}

/// Test `does_pattern_match` answers false for a mismatch.
#[test]
fn does_pattern_match_is_false_on_a_mismatch() {
    let result = does_pattern_match(&build_literal_pattern(5), &build_literal(6));

    assert!(!result.expect("no callback"));
}

/// Test matching tests the root only and never searches subexpressions.
#[test]
fn match_pattern_does_not_search_subexpressions() {
    let pattern = build_literal_pattern(1);

    let result = match_infallibly(&pattern, &build_simple_binary(BinaryOperation::Add));

    assert!(result.is_none(), "got {result:?}");
}

/// Test repeated matches of one pattern and expression give the same,
/// expected result.
#[rstest]
#[case::literal(build_literal_pattern(5), None)]
#[case::binary_capture(
    Pattern::binary(Some(BinaryOperation::Add), build_capture("a"), build_capture("b")),
    Some(vec![("a", 1), ("b", 2)])
)]
#[case::alternatives(
    build_alternatives(vec![build_literal_pattern(5), Pattern::identifier(None)]),
    None
)]
fn match_pattern_is_deterministic(
    #[case] pattern: Pattern,
    #[case] expected: Option<Vec<(&str, i64)>>,
) {
    let expression = build_simple_binary(BinaryOperation::Add);

    let first = match_infallibly(&pattern, &expression);
    let second = match_infallibly(&pattern, &expression);

    let describe = |bindings: &Option<MatchBindings>| {
        bindings.as_ref().map(|bindings| {
            bindings
                .names()
                .map(|name| (name.to_owned(), expect_bound(bindings, name).clone()))
                .collect::<Vec<_>>()
        })
    };
    let expected = expected.map(|entries| {
        entries
            .into_iter()
            .map(|(name, value)| (name.to_owned(), build_literal(value)))
            .collect::<Vec<_>>()
    });
    assert_eq!(describe(&first), expected);
    assert_eq!(describe(&second), expected);
}

// =============================================================================
// Errors
// =============================================================================

/// Test each pattern error displays its documented message.
#[rstest]
#[case::empty_capture_name(
    PatternError::EmptyCaptureName,
    "a capture pattern needs a non-empty name"
)]
#[case::empty_piecewise_cases(
    PatternError::EmptyPiecewiseCases,
    "a piecewise pattern with cases needs at least one case"
)]
#[case::empty_alternatives(
    PatternError::EmptyAlternatives,
    "an alternatives pattern needs at least one alternative"
)]
fn pattern_error_display_describes_the_refusal(
    #[case] error: PatternError,
    #[case] expected: &str,
) {
    let message = error.to_string();

    assert_eq!(message, expected);
}

/// Test a callback error built from text displays the text and has no
/// source.
#[rstest]
#[case::from_str(CallbackError::new("no verdict"))]
#[case::from_string(CallbackError::new(String::from("no verdict")))]
fn callback_error_new_from_text_displays_the_text(#[case] error: CallbackError) {
    let message = error.to_string();

    assert_eq!(message, "no verdict");
    assert!(std::error::Error::source(&error).is_none());
}

/// Test a callback error wraps an error transparently: its message, its
/// source, and the wrapped error itself.
#[test]
fn callback_error_wraps_an_error_transparently() {
    let error = CallbackError::new(ProbeError("probe"));

    let message = error.to_string();
    let inner = expect_probe_error(&error).clone();
    let unwrapped = error.into_inner();

    assert_eq!(message, "probe");
    assert_eq!(inner, ProbeError("probe"));
    assert_eq!(
        unwrapped.downcast_ref::<ProbeError>(),
        Some(&ProbeError("probe"))
    );
}

/// Test a callback error forwards the wrapped error's source.
#[test]
fn callback_error_source_is_the_wrapped_errors_source() {
    /// An error caused by a [`ProbeError`].
    #[derive(Debug)]
    struct CausedError(ProbeError);

    impl std::fmt::Display for CausedError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("caused")
        }
    }

    impl std::error::Error for CausedError {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            Some(&self.0)
        }
    }

    let error = CallbackError::new(CausedError(ProbeError("cause")));

    let source = std::error::Error::source(&error);

    assert_eq!(error.to_string(), "caused");
    let source = source.expect("the wrapped error's source");
    assert_eq!(
        source.downcast_ref::<ProbeError>(),
        Some(&ProbeError("cause"))
    );
}

/// Test a refused node build converts into a callback error wrapping it.
#[test]
fn callback_error_from_expression_build_error_wraps_it() {
    let build_error = ExpressionBuildError::EmptyFunctionName;

    let error = CallbackError::from(build_error.clone());

    assert_eq!(
        error.inner().downcast_ref::<ExpressionBuildError>(),
        Some(&build_error)
    );
    assert_eq!(error.to_string(), build_error.to_string());
}

/// Test a refused literal text converts into a callback error wrapping it.
#[test]
fn callback_error_from_literal_text_error_wraps_it() {
    let text_error = LiteralValue::parse_text("abc").expect_err("not a literal text");
    let expected_message = text_error.to_string();

    let error = CallbackError::from(text_error);

    let wrapped = error
        .inner()
        .downcast_ref::<fhy_core::expr::LiteralTextError>()
        .expect("a wrapped literal text error");
    assert_eq!(wrapped.text(), "abc");
    assert_eq!(error.to_string(), expected_message);
}

// =============================================================================
// Deep trees
// =============================================================================

/// Test a pattern that does not descend matches a tree
/// [`SMALL_STACK_DEPTH`] levels deep on a small thread stack, binding
/// the root's left operand.
#[test]
fn match_pattern_with_a_shallow_pattern_matches_a_deep_tree_on_a_small_stack() {
    run_on_small_stack(|| {
        let (_, x) = build_identifier("x");
        let tree = build_deep_sum(&x, SMALL_STACK_DEPTH);
        let pattern = Pattern::binary(
            Some(BinaryOperation::Add),
            build_capture("rest"),
            build_literal_pattern(1),
        );

        let bindings = match_infallibly(&pattern, &tree).expect("the deep sum matches");

        let ExpressionKind::Binary(root) = tree.kind() else {
            panic!("a sum at the root");
        };
        let rest = bindings.get("rest").expect("rest is bound");
        assert!(Expression::ptr_eq(rest, root.left()));
    });
}

/// Test a capture repeated across both operands compares two separately
/// built trees [`SMALL_STACK_DEPTH`] levels deep on a small thread stack:
/// equal operands match, binding the capture to the left one, and operands
/// differing only at the bottom do not.
#[test]
fn match_pattern_with_a_repeated_capture_compares_deep_operands_on_a_small_stack() {
    run_on_small_stack(|| {
        let (_, x) = build_identifier("x");
        let (_, y) = build_identifier("y");
        let left = build_deep_sum(&x, SMALL_STACK_DEPTH);
        let equal = build_deep_sum(&x, SMALL_STACK_DEPTH);
        let unequal = build_deep_sum(&y, SMALL_STACK_DEPTH);
        let pattern = Pattern::binary(
            Some(BinaryOperation::Subtract),
            build_capture("operand"),
            build_capture("operand"),
        );

        let matched = match_infallibly(&pattern, &(&left - &equal));
        let mismatched = match_infallibly(&pattern, &(&left - &unequal));

        let bindings = matched.expect("equal operands match");
        let bound = bindings.get("operand").expect("operand is bound");
        assert!(Expression::ptr_eq(bound, &left));
        assert!(
            mismatched.is_none(),
            "operands differing at the bottom match"
        );
    });
}

/// Test a pattern mirroring a 50-level chain matches it.
#[test]
fn match_pattern_matches_a_deeply_nested_chain() {
    let (_, x) = build_identifier("x");
    let tree = build_deep_sum(&x, 50);
    let pattern = build_deep_sum_pattern(build_capture("leaf"), 50);

    let bindings = expect_match(&pattern, &tree);

    assert!(Expression::ptr_eq(expect_bound(&bindings, "leaf"), &x));
}

/// Test a pattern [`DEEP_TREE_DEPTH`] levels deep matches a tree as deep on a
/// [`PATTERN_MATCH_STACK_BYTES`] thread stack, and a mismatch at the bottom
/// is found.
#[test]
fn match_pattern_matches_a_pattern_thousands_of_levels_deep() {
    let (matched, mismatched) = run_on_stack(PATTERN_MATCH_STACK_BYTES, || {
        let (identifier, x) = build_identifier("x");
        let (_, y) = build_identifier("y");
        let pattern =
            build_deep_sum_pattern(Pattern::identifier(Some(identifier)), DEEP_TREE_DEPTH);
        let matched = does_pattern_match(&pattern, &build_deep_sum(&x, DEEP_TREE_DEPTH));
        let mismatched = does_pattern_match(&pattern, &build_deep_sum(&y, DEEP_TREE_DEPTH));
        (
            matched.expect("no callback"),
            mismatched.expect("no callback"),
        )
    });

    assert!(matched);
    assert!(!mismatched);
}
