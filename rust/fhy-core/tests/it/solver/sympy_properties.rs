//! Properties of the SymPy backend, under the `sympy` feature, with the
//! core's evaluator as the oracle: a ground integer or Boolean tree
//! simplifies to the literal it evaluates to, and lifting the lowering of a
//! tree over identifiers gives an expression that evaluates the same at any
//! binding.

use std::collections::HashMap;

use fhy_core::expression::evaluate::{Evaluator, Scalar};
use fhy_core::expression::registry::FunctionRegistry;
use fhy_core::expression::{Expression, LogicalOperation};
use fhy_core::identifier::Identifier;
use fhy_core::solver::{SimplifyContext, Solver, SympySimplifier};
use proptest::prelude::*;

use crate::support::sympy::{attached, backend};

/// Return a small integer tree over `leaves`.
fn integer_tree(leaves: BoxedStrategy<Expression>) -> BoxedStrategy<Expression> {
    leaves
        .prop_recursive(4, 24, 2, |inner| {
            prop_oneof![
                (inner.clone(), inner.clone()).prop_map(|(a, b)| a + b),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| a - b),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| a * b),
                inner.prop_map(|a| -a),
            ]
        })
        .boxed()
}

/// Return a Boolean tree of comparisons of integer trees over `leaves`.
fn boolean_tree(leaves: BoxedStrategy<Expression>) -> BoxedStrategy<Expression> {
    let comparison =
        (integer_tree(leaves.clone()), integer_tree(leaves), 0..6_u8).prop_map(|(a, b, which)| {
            match which {
                0 => a.less(b),
                1 => a.less_equal(b),
                2 => a.greater(b),
                3 => a.greater_equal(b),
                4 => a.equals(b),
                _ => a.not_equals(b),
            }
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

/// Return the literal of the scalar `value`.
fn literal_of(value: Scalar) -> Expression {
    match value {
        Scalar::Bool(value) => Expression::literal(value),
        Scalar::Int(value) => Expression::literal(value),
        Scalar::Real(value) => Expression::literal(value),
    }
}

/// Return the value of `expression` over `environment`.
fn evaluated(expression: &Expression, environment: &HashMap<Identifier, Scalar>) -> Scalar {
    Evaluator::new(&FunctionRegistry::new())
        .evaluate(expression, environment)
        .expect("evaluated")
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn ground_tree_simplifies_to_the_literal_it_evaluates_to(
        tree in prop_oneof![
            integer_tree((-5_i64..=5).prop_map(Expression::from).boxed()),
            boolean_tree((-5_i64..=5).prop_map(Expression::from).boxed()),
        ],
    ) {
        backend();
        let registry = FunctionRegistry::new();
        let solver = Solver::new().with_simplifier(SympySimplifier::new());

        let simplified = solver
            .simplify(&tree, &HashMap::new(), &SimplifyContext::from_registry(&registry))
            .expect("simplified");

        prop_assert_eq!(simplified, literal_of(evaluated(&tree, &HashMap::new())));
    }

    #[test]
    fn lifting_the_lowering_keeps_the_value_at_every_binding(
        tree in {
            let x = Identifier::new("x");
            let y = Identifier::new("y");
            let leaves = prop_oneof![
                (-3_i64..=3).prop_map(Expression::from),
                Just(Expression::from(x.clone())),
                Just(Expression::from(y.clone())),
            ]
            .boxed();
            (integer_tree(leaves), Just(x), Just(y))
        },
        x_value in -4_i64..=4,
        y_value in -4_i64..=4,
    ) {
        let (tree, x, y) = tree;
        let registry = FunctionRegistry::new();
        let context = SimplifyContext::from_registry(&registry);
        let round_trip = attached(|py| {
            let lowered = backend().lower(py, &tree, &context).expect("lowered");
            backend().lift(&lowered).expect("lifted")
        });
        let environment =
            HashMap::from([(x, Scalar::Int(x_value)), (y, Scalar::Int(y_value))]);

        prop_assert_eq!(evaluated(&round_trip, &environment), evaluated(&tree, &environment));
    }
}
