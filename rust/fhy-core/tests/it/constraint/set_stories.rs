//! Tests for `SetConstraint`: the order of its evaluation, membership of
//! bound values and literal expressions, the refused values, the native
//! constant refusal, its events, and its expression.
//!
//! The cases are ported from `test_set_constraints.py`,
//! `test_bindings_evaluation.py` and `test_convert_to_expression.py`.

use fhy_core::constraint::{
    Binding, ConstraintContext, ConstraintError, Outcome, Polarity, SetConstraint,
    UnusableBindingReason, Value,
};
use fhy_core::expression::builtins::BuiltinConstant;
use fhy_core::expression::registry::{FunctionRegistry, NativeConstant};
use fhy_core::expression::{
    Decimal, Expression, FunctionName, FunctionSort, LiteralValue, LogicalOperation,
};
use fhy_core::identifier::Identifier;
use fhy_core::solver::Solver;
use rstest::rstest;

use crate::support::constraint::{
    RecordedEvent, RecordingObserver, TestOpaque, bind, int, int_set, member_set, text,
};
use crate::support::expression::{expect_binary, expect_literal, expect_logical};

fn build_set(polarity: Polarity, members: &[i64]) -> (Identifier, SetConstraint) {
    let x = Identifier::new("x");
    let constraint = SetConstraint::new(x.clone(), int_set(members.iter().copied()), polarity);
    (x, constraint)
}

fn evaluate(
    constraint: &SetConstraint,
    bindings: &fhy_core::constraint::Bindings,
) -> (Result<Outcome, ConstraintError>, Vec<RecordedEvent>) {
    let solver = Solver::new();
    let observer = RecordingObserver::default();
    let context = ConstraintContext::new(&solver).with_observer(&observer);
    let outcome = constraint.evaluate(bindings, &context);
    (outcome, observer.events())
}

#[rstest]
#[case::in_set(Polarity::In)]
#[case::not_in_set(Polarity::NotIn)]
fn unbound_variable_is_undecided_and_reported(#[case] polarity: Polarity) {
    let (x, constraint) = build_set(polarity, &[1, 2]);
    let y = Identifier::new("y");

    let (outcome, events) = evaluate(&constraint, &bind([(y, Binding::Value(int(1)))]));

    assert_eq!(
        outcome.expect("an unbound variable decides nothing"),
        Outcome::Undecided
    );
    assert_eq!(events, [RecordedEvent::Unbound(x)]);
}

#[rstest]
#[case::a_member_in(Polarity::In, 2, Outcome::Satisfied)]
#[case::a_non_member_in(Polarity::In, 3, Outcome::Violated)]
#[case::a_member_not_in(Polarity::NotIn, 2, Outcome::Violated)]
#[case::a_non_member_not_in(Polarity::NotIn, 3, Outcome::Satisfied)]
fn bound_value_is_decided_by_membership_and_polarity(
    #[case] polarity: Polarity,
    #[case] value: i64,
    #[case] expected: Outcome,
) {
    let (x, constraint) = build_set(polarity, &[1, 2]);

    let (outcome, events) = evaluate(&constraint, &bind([(x, Binding::Value(int(value)))]));

    assert_eq!(outcome.expect("a member-shaped value is decided"), expected);
    assert!(events.is_empty(), "{events:?}");
}

#[rstest]
#[case::an_integer_literal(LiteralValue::Int(2.into()), Outcome::Satisfied)]
#[case::a_float_literal_of_a_member_value(LiteralValue::Float(2.0), Outcome::Violated)]
#[case::a_boolean_literal(LiteralValue::Bool(true), Outcome::Violated)]
#[case::a_decimal_literal(LiteralValue::Decimal("2".parse::<Decimal>().expect("a decimal")), Outcome::Violated)]
#[case::a_nan_literal(LiteralValue::Float(f64::NAN), Outcome::Violated)]
fn literal_expression_binding_is_decided_by_its_value(
    #[case] literal: LiteralValue,
    #[case] expected: Outcome,
) {
    let (x, constraint) = build_set(Polarity::In, &[1, 2]);

    let (outcome, _) = evaluate(
        &constraint,
        &bind([(x, Binding::Expression(Expression::literal(literal)))]),
    );

    assert_eq!(outcome.expect("a literal is decided"), expected);
}

#[test]
fn symbolic_binding_is_undecided_and_reported() {
    let (x, constraint) = build_set(Polarity::In, &[1]);
    let y = Identifier::new("y");
    let symbolic = &Expression::from(y) + 1;

    let (outcome, events) = evaluate(
        &constraint,
        &bind([(x.clone(), Binding::Expression(symbolic.clone()))]),
    );

    assert_eq!(
        outcome.expect("a symbolic binding decides nothing"),
        Outcome::Undecided
    );
    assert_eq!(events, [RecordedEvent::SymbolicBinding(x, symbolic)]);
}

#[rstest]
#[case::a_decimal(Value::Decimal("1".parse::<Decimal>().expect("a decimal")))]
#[case::an_opaque_non_member(TestOpaque { is_member_shaped: false, ..TestOpaque::token(1) }.into_value())]
#[case::a_tuple_holding_a_decimal(Value::Tuple(vec![Value::Decimal("1".parse::<Decimal>().expect("a decimal"))]))]
fn value_that_could_never_be_a_member_is_refused(#[case] value: Value) {
    let (x, constraint) = build_set(Polarity::NotIn, &[1, 2, 3]);

    let (outcome, _) = evaluate(&constraint, &bind([(x.clone(), Binding::Value(value))]));

    let error = outcome.expect_err("the value is not member-shaped");
    assert!(
        matches!(
            &error,
            ConstraintError::UnusableBinding {
                identifier,
                reason: UnusableBindingReason::NotMemberShaped,
            } if *identifier == x
        ),
        "{error:?}"
    );
    assert_eq!(
        error.to_string(),
        format!("the binding of {x:?} is neither an expression nor a value that could be a member")
    );
}

#[test]
fn value_that_cannot_be_looked_up_is_refused_with_its_cause() {
    let (x, constraint) = build_set(Polarity::In, &[1]);
    let unhashable = TestOpaque {
        is_hashable: false,
        ..TestOpaque::token(1)
    };

    let (outcome, _) = evaluate(
        &constraint,
        &bind([(x, Binding::Value(unhashable.into_value()))]),
    );

    let error = outcome.expect_err("the value is unhashable");
    assert!(
        matches!(
            &error,
            ConstraintError::UnusableBinding {
                reason: UnusableBindingReason::Unhashable(_),
                ..
            }
        ),
        "{error:?}"
    );
    assert_eq!(
        std::error::Error::source(&error).map(ToString::to_string),
        Some("Token is unhashable".to_owned())
    );
}

#[rstest]
#[case::a_tuple(Value::Tuple(vec![int(1), int(2)]))]
#[case::a_frozen_set(Value::FrozenSet(vec![int(2), int(1)]))]
#[case::an_opaque_member(TestOpaque::token(4).into_value())]
fn container_and_opaque_values_are_decided(#[case] value: Value) {
    let x = Identifier::new("x");
    let members = member_set([value.clone()]);
    let in_set = SetConstraint::new(x.clone(), members.clone(), Polarity::In);
    let not_in_set = SetConstraint::new(x.clone(), members, Polarity::NotIn);
    let bindings = bind([(x, Binding::Value(value))]);

    assert_eq!(
        evaluate(&in_set, &bindings).0.expect("decided"),
        Outcome::Satisfied
    );
    assert_eq!(
        evaluate(&not_in_set, &bindings).0.expect("decided"),
        Outcome::Violated
    );
}

#[test]
fn nan_binding_is_decided_as_no_member() {
    let x = Identifier::new("x");
    let constraint = SetConstraint::new(x.clone(), member_set([Value::Float(1.0)]), Polarity::In);

    let (outcome, _) = evaluate(
        &constraint,
        &bind([(x, Binding::Value(Value::Float(f64::NAN)))]),
    );

    assert_eq!(outcome.expect("a NaN is decided"), Outcome::Violated);
}

#[test]
fn negative_zero_binding_matches_a_zero_member() {
    let x = Identifier::new("x");
    let constraint = SetConstraint::new(x.clone(), member_set([Value::Float(0.0)]), Polarity::In);

    let (outcome, _) = evaluate(
        &constraint,
        &bind([(x, Binding::Value(Value::Float(-0.0)))]),
    );

    assert_eq!(outcome.expect("a zero is decided"), Outcome::Satisfied);
}

#[test]
fn other_bindings_are_never_read() {
    let (x, constraint) = build_set(Polarity::In, &[1]);
    let unusable = Value::Decimal("1".parse::<Decimal>().expect("a decimal"));

    let (outcome, _) = evaluate(
        &constraint,
        &bind([
            (Identifier::new("y"), Binding::Value(unusable)),
            (x, Binding::Value(int(1))),
        ]),
    );

    assert_eq!(outcome.expect("only x is read"), Outcome::Satisfied);
}

#[test]
fn built_in_constant_variable_is_refused_after_the_value_checks() {
    let pi = BuiltinConstant::Pi.identifier().clone();
    let constraint = SetConstraint::new(pi.clone(), int_set([1]), Polarity::In);
    let solver = Solver::new();
    let observer = RecordingObserver::default();
    let context = ConstraintContext::new(&solver).with_observer(&observer);

    let refused = constraint.evaluate(&bind([(pi.clone(), Binding::Value(int(1)))]), &context);
    let unusable = constraint.evaluate(
        &bind([(
            pi.clone(),
            Binding::Value(Value::Decimal("1".parse().expect("a decimal"))),
        )]),
        &context,
    );

    assert_eq!(
        refused.expect("the constant is refused"),
        Outcome::Undecided
    );
    assert!(unusable.is_err(), "the unusable value is refused first");
    assert_eq!(
        observer.events(),
        [RecordedEvent::BoundNativeConstants(vec![pi])]
    );
}

#[test]
fn registered_constant_variable_is_refused_through_the_registry() {
    let mut registry = FunctionRegistry::new();
    let answer = registry
        .register_constant(
            NativeConstant::new(
                FunctionName::new("answer").expect("a name"),
                FunctionSort::Int,
                42,
            )
            .expect("an integer constant"),
        )
        .expect("the name is free");
    let constraint = SetConstraint::new(answer.clone(), int_set([42]), Polarity::In);
    let bindings = bind([(answer, Binding::Value(int(42)))]);
    let solver = Solver::new();

    let with_registry = constraint.evaluate(
        &bindings,
        &ConstraintContext::new(&solver).with_registry(&registry),
    );
    let without_registry = constraint.evaluate(&bindings, &ConstraintContext::new(&solver));

    assert_eq!(with_registry.expect("refused"), Outcome::Undecided);
    assert_eq!(without_registry.expect("decided"), Outcome::Satisfied);
}

#[rstest]
#[case::in_set(Polarity::In, false)]
#[case::not_in_set(Polarity::NotIn, true)]
fn empty_set_converts_to_a_boolean(#[case] polarity: Polarity, #[case] expected: bool) {
    let (_, constraint) = build_set(polarity, &[]);

    let expression = constraint.to_expression().expect("an empty set converts");

    assert_eq!(expect_literal(&expression), &LiteralValue::Bool(expected));
}

#[test]
fn one_member_converts_to_one_comparison() {
    let (x, constraint) = build_set(Polarity::In, &[7]);

    let expression = constraint.to_expression().expect("an integer lifts");

    assert_eq!(expression, Expression::from(x).equals(7));
}

#[rstest]
#[case::in_set(Polarity::In, LogicalOperation::Or)]
#[case::not_in_set(Polarity::NotIn, LogicalOperation::And)]
fn members_convert_to_one_connective_in_canonical_order(
    #[case] polarity: Polarity,
    #[case] operation: LogicalOperation,
) {
    let (x, constraint) = build_set(polarity, &[12, 3, 7, 9]);
    let reference = Expression::from(x);

    let expression = constraint.to_expression().expect("integers lift");

    let logical = expect_logical(&expression);
    assert_eq!(logical.operation(), operation);
    let expected: Vec<Expression> = [3, 7, 9, 12]
        .into_iter()
        .map(|value| match polarity {
            Polarity::In => reference.equals(value),
            _ => reference.not_equals(value),
        })
        .collect();
    assert_eq!(logical.operands(), expected.as_slice());
    assert!(
        expect_binary(&logical.operands()[0])
            .right()
            .free_identifiers()
            .is_empty()
    );
}

#[test]
fn string_member_does_not_convert_since_membership_is_type_strict() {
    let x = Identifier::new("x");
    let constraint = SetConstraint::new(x, member_set([int(1), text("5")]), Polarity::In);

    let error = constraint
        .to_expression()
        .expect_err("a string does not lift");

    assert!(
        matches!(error, ConstraintError::UnliftableMember(_)),
        "{error:?}"
    );
    assert!(error.to_string().contains("type-strict"), "{error}");
}

#[rstest]
#[case::a_tuple(Value::Tuple(vec![int(1)]), "tuple")]
#[case::a_frozen_set(Value::FrozenSet(vec![int(1)]), "frozenset")]
#[case::an_opaque_member(TestOpaque::token(1).into_value(), "Token")]
fn container_or_opaque_member_does_not_convert(#[case] value: Value, #[case] kind: &str) {
    let x = Identifier::new("x");
    let constraint = SetConstraint::new(x, member_set([value]), Polarity::NotIn);

    let error = constraint.to_expression().expect_err("it does not lift");

    assert_eq!(
        error.to_string(),
        format!("conversion of type {kind} to an expression is not supported")
    );
}

#[test]
fn scope_is_the_variable() {
    let (x, constraint) = build_set(Polarity::In, &[1]);

    assert_eq!(constraint.free_identifiers(), [x].into_iter().collect());
}
