//! Properties of the ground simplifier, with the core's evaluator as the
//! oracle: a ground integer or Boolean tree folds to the literal it
//! evaluates to, a fold is a fixed point, and a tree with a free identifier
//! is declined.

use std::collections::HashMap;

use fhy_core::expression::evaluate::{Evaluator, Scalar};
use fhy_core::expression::registry::FunctionRegistry;
use fhy_core::expression::{Expression, LogicalOperation};
use fhy_core::identifier::Identifier;
use fhy_core::solver::{GroundSimplifier, SimplifyContext};
use proptest::prelude::*;

/// Return a nonzero divisor.
fn divisor() -> impl Strategy<Value = i64> {
    prop_oneof![-4_i64..=-1, 1_i64..=4]
}

/// Return an integer tree over small integers: sums, differences,
/// products, floor divisions and moduli by nonzero integers, negations, and small powers
/// of small integers, so that no value leaves the evaluator's `i64`.
fn integer_tree() -> BoxedStrategy<Expression> {
    (-6_i64..=6)
        .prop_map(Expression::from)
        .prop_recursive(4, 24, 2, |inner| {
            prop_oneof![
                (inner.clone(), inner.clone()).prop_map(|(a, b)| a + b),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| a - b),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| a * b),
                (inner.clone(), divisor()).prop_map(|(a, k)| a.floor_divide(k)),
                (inner.clone(), divisor()).prop_map(|(a, k)| a.floor_mod(k)),
                (-2_i64..=2, 0_i64..=3).prop_map(|(a, k)| Expression::from(a).power(k)),
                inner.prop_map(|a| -a),
            ]
        })
        .boxed()
}

/// Return a Boolean tree of comparisons of integer trees.
fn boolean_tree() -> BoxedStrategy<Expression> {
    let comparison =
        (integer_tree(), integer_tree(), 0..6_u8).prop_map(|(a, b, which)| match which {
            0 => a.less(b),
            1 => a.less_equal(b),
            2 => a.greater(b),
            3 => a.greater_equal(b),
            4 => a.equals(b),
            _ => a.not_equals(b),
        });
    comparison
        .prop_recursive(2, 8, 2, |inner| {
            prop_oneof![
                (inner.clone(), inner.clone())
                    .prop_map(|(a, b)| Expression::new_logical(LogicalOperation::And, [a, b])),
                (inner.clone(), inner.clone())
                    .prop_map(|(a, b)| Expression::new_logical(LogicalOperation::Or, [a, b])),
                inner.prop_map(|a| !a),
            ]
        })
        .boxed()
}

/// Return the ground simplifier's fold of `expression`.
fn folded(expression: &Expression) -> Option<Expression> {
    GroundSimplifier::new().try_simplify(expression, &SimplifyContext::default())
}

/// Return the literal of the scalar `value`.
fn literal_of(value: Scalar) -> Expression {
    match value {
        Scalar::Bool(value) => Expression::literal(value),
        Scalar::Int(value) => Expression::literal(value),
        Scalar::Real(value) => Expression::literal(value),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn a_ground_tree_folds_to_the_literal_it_evaluates_to(
        tree in prop_oneof![integer_tree(), boolean_tree()],
    ) {
        let registry = FunctionRegistry::new();
        let evaluated = Evaluator::new(&registry).evaluate(&tree, &HashMap::new());

        prop_assert_eq!(folded(&tree), Some(literal_of(evaluated.expect("evaluated"))));
    }

    #[test]
    fn a_fold_is_a_fixed_point(tree in prop_oneof![integer_tree(), boolean_tree()]) {
        if let Some(literal) = folded(&tree) {
            prop_assert_eq!(folded(&literal), Some(literal));
        }
    }

    #[test]
    fn a_tree_with_a_free_identifier_is_declined(tree in integer_tree()) {
        let free = Expression::from(Identifier::new("free"));

        prop_assert_eq!(folded(&(tree + free)), None);
    }
}
