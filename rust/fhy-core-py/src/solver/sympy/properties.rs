//! Properties of the SymPy backend, with the core's evaluator as the
//! oracle: a ground integer or Boolean tree
//! simplifies to the literal it evaluates to, and lifting the lowering of a
//! tree over identifiers gives an expression that evaluates the same at any
//! binding.

use std::collections::HashMap;

use fhy_core::expression::evaluate::{Evaluator, Scalar};
use fhy_core::expression::registry::FunctionRegistry;
use fhy_core::expression::{BinaryOperation, Expression, ExpressionKind, LogicalOperation};
use fhy_core::identifier::Identifier;
use fhy_core::solver::{SimplifyContext, Solver};
use proptest::prelude::*;

use super::SympySimplifier;
use super::test_support::{attached, backend};

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

    #[test]
    fn simplify_then_evaluate_equals_evaluate_on_integer_grids(
        tree in {
            let x = Identifier::new("x");
            let y = Identifier::new("y");
            let leaves = prop_oneof![
                (-3_i64..=3).prop_map(Expression::from),
                Just(Expression::from(x.clone())),
                Just(Expression::from(y.clone())),
            ]
            .boxed();
            (grid_tree(leaves), Just(x), Just(y))
        },
    ) {
        let (tree, x, y) = tree;
        let registry = FunctionRegistry::new();
        backend();
        // A tree SymPy folds to a value no expression denotes, such as the
        // complex infinity of `1 / 0`, is refused, as it should be.
        let simplified = Solver::new()
            .with_simplifier(SympySimplifier::new())
            .simplify(&tree, &HashMap::new(), &SimplifyContext::from_registry(&registry));
        prop_assume!(simplified.is_ok());
        let simplified = simplified.expect("simplified");

        for x_value in -3_i64..=3 {
            for y_value in -3_i64..=3 {
                let environment =
                    HashMap::from([(x.clone(), Scalar::Int(x_value)), (y.clone(), Scalar::Int(y_value))]);
                let evaluator = Evaluator::new(&registry);
                let Ok(expected) = evaluator.evaluate(&tree, &environment) else {
                    continue;
                };
                if matches!(expected, Scalar::Real(value) if !value.is_finite())
                    || !is_every_divisor_nonzero(evaluator, &tree, &environment)
                {
                    continue;
                }
                let actual = evaluator.evaluate(&simplified, &environment);
                prop_assert!(
                    actual.as_ref().is_ok_and(|actual| is_numerically_equal(*actual, expected)),
                    "{} simplified to {}, which at x = {}, y = {} gives {:?}, not {:?}",
                    tree, simplified, x_value, y_value, actual, expected
                );
            }
        }
    }
}

/// Return a grid tree over `leaves`: sums, differences, products, floor
/// divisions by `1`, `2` or `4`, floor moduli by `2` or `3`, powers by `0`
/// to `3`, negations, piecewise nodes, and quotients by `2` or by another
/// tree. Powers of two keep the quotients' binary floats exact where
/// `sympy.simplify` distributes a divisor.
fn grid_tree(leaves: BoxedStrategy<Expression>) -> BoxedStrategy<Expression> {
    leaves
        .prop_recursive(3, 16, 2, |inner| {
            prop_oneof![
                (inner.clone(), inner.clone()).prop_map(|(a, b)| a + b),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| a - b),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| a * b),
                (inner.clone(), prop_oneof![Just(1_i64), Just(2), Just(4)])
                    .prop_map(|(a, k)| a.floor_divide(k)),
                (inner.clone(), 2_i64..=3).prop_map(|(a, k)| a.floor_mod(k)),
                (inner.clone(), 0_i64..=3).prop_map(|(a, k)| a.power(k)),
                inner.clone().prop_map(|a| -a),
                (inner.clone(), inner.clone(), inner.clone(), inner.clone()).prop_map(
                    |(left, right, value, otherwise)| {
                        Expression::piecewise([(left.less(right), value)], otherwise)
                            .expect("a piecewise")
                    }
                ),
                inner.clone().prop_map(|a| a / 2),
                (inner.clone(), inner).prop_map(|(a, b)| a / b),
            ]
        })
        .boxed()
}

/// Return whether every quotient of `tree` divides by a finite nonzero
/// number at `environment`. Where one does not, the tree passes through an
/// infinity or a NaN, which `sympy.simplify`'s cancellations assume away
/// (`y / y` is `1`), so the grid point says nothing about the lifting.
fn is_every_divisor_nonzero(
    evaluator: Evaluator<'_>,
    tree: &Expression,
    environment: &HashMap<Identifier, Scalar>,
) -> bool {
    let mut pending = vec![tree];
    while let Some(node) = pending.pop() {
        if let ExpressionKind::Binary(binary) = node.kind() {
            if binary.operation() == BinaryOperation::Divide {
                let is_nonzero = match evaluator.evaluate(binary.right(), environment) {
                    Ok(Scalar::Int(value)) => value != 0,
                    Ok(Scalar::Real(value)) => value.is_finite() && value != 0.0,
                    _ => false,
                };
                if !is_nonzero {
                    return false;
                }
            }
        }
        pending.extend(node.children());
    }
    true
}

/// Return whether `actual` and `expected` are the same Boolean, or numbers
/// within a relative `1e-9` of each other, integers and reals alike.
fn is_numerically_equal(actual: Scalar, expected: Scalar) -> bool {
    let number = |value: Scalar| match value {
        Scalar::Bool(_) => None,
        #[expect(clippy::cast_precision_loss, reason = "the grid's integers are small")]
        Scalar::Int(value) => Some(value as f64),
        Scalar::Real(value) => Some(value),
    };
    match (actual, expected) {
        (Scalar::Bool(actual), Scalar::Bool(expected)) => actual == expected,
        _ => match (number(actual), number(expected)) {
            (Some(actual), Some(expected)) => {
                (actual - expected).abs() <= 1e-9 * expected.abs().max(1.0)
            }
            _ => false,
        },
    }
}
