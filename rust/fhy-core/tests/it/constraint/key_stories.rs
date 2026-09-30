//! Tests for the canonical ordering keys: equal for structurally equivalent
//! constraints, distinct otherwise, and computed on a small stack for a
//! deep expression.
//!
//! The cases are ported from `test_ordering_key.py`.

use fhy_core::constraint::{
    Constraint, ConstraintSystem, EquationConstraint, Polarity, SetConstraint, Value,
};
use fhy_core::expression::{Decimal, Expression, LiteralValue};
use fhy_core::identifier::Identifier;
use rstest::rstest;

use crate::support::constraint::ConstraintKey;
use crate::support::constraint::{TestOpaque, int, int_set, member_set, text};
use crate::support::expression::{build_deep_sum, build_parsed_literal};
use crate::support::stack::{SMALL_STACK_DEPTH, run_on_small_stack};

fn equation(expression: Expression) -> Constraint {
    Constraint::from(EquationConstraint::new(expression))
}

fn decimal(text: &str) -> Expression {
    Expression::literal(text.parse::<Decimal>().expect("a decimal text"))
}

#[test]
fn keys_start_with_the_kind() {
    let x = Identifier::new("x");

    assert!(
        equation(Expression::from(x.clone()).less(1))
            .key()
            .starts_with("equation|")
    );
    assert!(
        Constraint::from(SetConstraint::new(x.clone(), int_set([1]), Polarity::In))
            .key()
            .starts_with("in_set|")
    );
    assert!(
        Constraint::from(SetConstraint::new(x, int_set([1]), Polarity::NotIn))
            .key()
            .starts_with("not_in_set|")
    );
}

#[rstest]
#[case::an_integer_and_its_padded_text(Expression::literal(5), build_parsed_literal("05"))]
#[case::a_decimal_and_its_padded_text(decimal("1.5"), build_parsed_literal("1.50"))]
#[case::two_nans(Expression::literal(f64::NAN), Expression::literal(f64::NAN))]
#[case::two_zeros(Expression::literal(-0.0), Expression::literal(0.0))]
fn equal_literals_key_alike(#[case] left: Expression, #[case] right: Expression) {
    let x = Identifier::new("x");
    let reference = Expression::from(x);

    assert_eq!(
        equation(reference.equals(left)).key(),
        equation(reference.equals(right)).key()
    );
}

#[test]
fn equations_built_apart_over_one_identifier_key_alike() {
    let x = Identifier::new("x");
    let build = || {
        equation(Expression::all([
            Expression::from(x.clone()).greater(0),
            Expression::from(x.clone()).less(10),
        ]))
    };

    assert_eq!(build().key(), build().key());
}

#[rstest]
#[case::another_comparison(Expression::from(Identifier::new("x")).less(1), Expression::from(Identifier::new("x")).less_equal(1))]
#[case::another_identifier_with_the_same_name(Expression::from(Identifier::new("x")).less(1), Expression::from(Identifier::new("x")).less(1))]
#[case::a_float_and_a_decimal(Expression::literal(true).equals(Expression::literal(1.5)), Expression::literal(true).equals(decimal("1.5")))]
#[case::an_integer_and_a_boolean(Expression::literal(true).equals(Expression::literal(1)), Expression::literal(true).equals(Expression::literal(true)))]
#[case::and_and_or(Expression::literal(true).and(Expression::literal(false)), Expression::literal(true).or(Expression::literal(false)))]
#[case::flat_and_nested(
    Expression::all([Expression::literal(true), Expression::literal(true), Expression::literal(true)]),
    Expression::all([Expression::literal(true), Expression::all([Expression::literal(true), Expression::literal(true)])])
)]
#[case::decimals_apart_in_the_thirtieth_digit(
    Expression::literal(true).equals(decimal("0.100000000000000000000000000001")),
    Expression::literal(true).equals(decimal("0.100000000000000000000000000002"))
)]
fn different_equations_key_apart(#[case] left: Expression, #[case] right: Expression) {
    assert_ne!(equation(left).key(), equation(right).key());
}

#[test]
fn set_keys_hold_the_polarity_the_variable_and_the_members() {
    let (x, y) = (Identifier::new("x"), Identifier::new("y"));
    let key = |variable: &Identifier, members: &[i64], polarity| {
        SetConstraint::new(variable.clone(), int_set(members.iter().copied()), polarity).key()
    };

    assert_eq!(
        key(&x, &[1, 2], Polarity::In),
        key(&x, &[2, 1, 2], Polarity::In)
    );
    assert_ne!(
        key(&x, &[1, 2], Polarity::In),
        key(&x, &[1, 2], Polarity::NotIn)
    );
    assert_ne!(
        key(&x, &[1, 2], Polarity::In),
        key(&y, &[1, 2], Polarity::In)
    );
    assert_ne!(
        key(&x, &[1, 2], Polarity::In),
        key(&x, &[1, 3], Polarity::In)
    );
}

#[rstest]
#[case::an_integer_and_a_boolean(int(1), Value::Bool(true))]
#[case::an_integer_and_a_float(int(1), Value::Float(1.0))]
#[case::an_integer_and_its_text(int(1), text("1"))]
#[case::a_tuple_and_a_frozen_set(Value::Tuple(vec![int(1)]), Value::FrozenSet(vec![int(1)]))]
#[case::a_string_holding_a_separator(text("a,b"), Value::Tuple(vec![text("a"), text("b")]))]
fn different_members_key_apart(#[case] left: Value, #[case] right: Value) {
    let x = Identifier::new("x");
    let key = |value: Value| SetConstraint::new(x.clone(), member_set([value]), Polarity::In).key();

    assert_ne!(key(left), key(right));
}

#[test]
fn colliding_opaque_members_key_alike_in_either_order() {
    let x = Identifier::new("x");
    let key = |first: i64, second: i64| {
        SetConstraint::new(
            x.clone(),
            member_set([
                TestOpaque::colliding(first).into_value(),
                TestOpaque::colliding(second).into_value(),
            ]),
            Polarity::In,
        )
        .key()
    };

    assert_eq!(key(1, 2), key(2, 1));
}

#[test]
fn equation_key_is_computed_on_a_small_stack() {
    let key = run_on_small_stack(|| {
        let x = Expression::from(Identifier::new("x"));
        let deep = build_deep_sum(&x, SMALL_STACK_DEPTH).less(0);
        let key = equation(deep).key();
        key.len()
    });

    assert!(key > SMALL_STACK_DEPTH);
}

#[test]
fn literal_values_the_key_distinguishes_are_unequal_literals() {
    let pairs = [
        (LiteralValue::Int(1.into()), LiteralValue::Float(1.0)),
        (
            LiteralValue::Float(1.0),
            LiteralValue::Decimal("1".parse().expect("a decimal")),
        ),
    ];
    for (left, right) in pairs {
        assert_ne!(left, right);
        assert_ne!(
            equation(Expression::literal(true).equals(Expression::literal(left))).key(),
            equation(Expression::literal(true).equals(Expression::literal(right))).key()
        );
    }
}

// =============================================================================
// The table key
// =============================================================================

#[test]
fn a_doubling_dag_of_depth_64_keys_in_linear_size() {
    let x = Expression::from(Identifier::new("x"));
    let dag = crate::support::expression::build_doubling_dag(&x, 64);

    let constraint = equation(dag.equals(0));
    let key = constraint.key();

    assert!(key.len() < 64 * 48, "{} bytes", key.len());
    let system = ConstraintSystem::new([constraint]).expect("the key is linear");
    assert_eq!(system.constraints().len(), 1);
}

#[test]
fn keys_are_equal_however_the_expression_is_shared() {
    let x = Identifier::new("x");
    let leaf = Expression::from(x.clone());
    let shared = equation((&leaf + &leaf).equals(0));
    let repeated = equation((Expression::from(x.clone()) + Expression::from(x.clone())).equals(0));

    assert_eq!(shared.key(), repeated.key());
    assert_eq!(
        repeated.key(),
        format!(
            "equation|identifier[{}]();binary[add](0,0);literal[int:0]();binary[equal](1,2)",
            x.id()
        )
    );
}

#[test]
fn a_callee_key_is_its_stable_text() {
    let x = Identifier::try_restore(7, "x").expect("below the cap");
    let reference = Expression::from(x);
    let named = equation(
        Expression::call(
            fhy_core::expression::FunctionName::new("f").expect("a name"),
            [reference.clone()],
        )
        .greater(0),
    );
    let builtin = equation(
        Expression::call(
            fhy_core::expression::builtins::BuiltinFunction::Max,
            [reference, Expression::literal(1)],
        )
        .greater(0),
    );

    assert_eq!(
        named.key(),
        r#"equation|identifier[7]();call[named:"f"](0);literal[int:0]();binary[greater](1,2)"#
    );
    assert_eq!(
        builtin.key(),
        "equation|identifier[7]();literal[int:1]();call[builtin:max](0,1);literal[int:0]();\
         binary[greater](2,3)"
    );
}

#[test]
fn a_float_key_is_its_canonical_text_with_one_zero_and_one_nan() {
    let x = Identifier::try_restore(7, "x").expect("below the cap");
    let reference = Expression::from(x);
    let key = |value: f64| equation(reference.equals(Expression::literal(value))).key();

    assert_eq!(
        key(1e300),
        "equation|identifier[7]();literal[float:1e300]();binary[equal](0,1)"
    );
    assert_eq!(key(-0.0), key(0.0));
    assert!(key(0.0).contains("literal[float:0]()"));
    assert!(key(-f64::NAN).contains("literal[float:NaN]()"));
}
