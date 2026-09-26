//! Properties of evaluation: folding does not change a value, composed
//! built-ins evaluate as their definitions say, and, with the `ndarray`
//! feature, every lane of an array evaluation is the scalar evaluation of
//! that lane.

use std::collections::HashMap;
use std::sync::LazyLock;

use fhy_core::expression::builtins::BuiltinFunction;
use fhy_core::expression::evaluate::{EvaluationError, Evaluator, NoNativeCalls, Scalar};
use fhy_core::expression::registry::FunctionRegistry;
use fhy_core::expression::{BinaryOperation, Callee, Expression, LiteralValue, UnaryOperation};
use fhy_core::identifier::Identifier;
use proptest::prelude::*;
use proptest::sample::select;

/// The identifiers of the generated trees: an integer, a real and a
/// Boolean.
static IDENTIFIERS: LazyLock<[Identifier; 3]> = LazyLock::new(|| {
    [
        Identifier::new("n"),
        Identifier::new("r"),
        Identifier::new("p"),
    ]
});

const ARITHMETIC: [BinaryOperation; 7] = [
    BinaryOperation::Add,
    BinaryOperation::Subtract,
    BinaryOperation::Multiply,
    BinaryOperation::Divide,
    BinaryOperation::FloorDivide,
    BinaryOperation::FloorMod,
    BinaryOperation::Power,
];

const COMPARISONS: [BinaryOperation; 6] = [
    BinaryOperation::Equal,
    BinaryOperation::NotEqual,
    BinaryOperation::Less,
    BinaryOperation::LessEqual,
    BinaryOperation::Greater,
    BinaryOperation::GreaterEqual,
];

const NATIVES: [BuiltinFunction; 6] = [
    BuiltinFunction::Floor,
    BuiltinFunction::Round,
    BuiltinFunction::Sqrt,
    BuiltinFunction::Exp,
    BuiltinFunction::Log,
    BuiltinFunction::Tanh,
];

fn numeric_leaf() -> BoxedStrategy<Expression> {
    prop_oneof![
        Just(Expression::from(IDENTIFIERS[0].clone())),
        Just(Expression::from(IDENTIFIERS[1].clone())),
        (-4_i64..5).prop_map(|value| Expression::from(LiteralValue::from(value))),
        select(vec![0.0, -0.5, 1.5, 2.0, f64::NAN, f64::INFINITY])
            .prop_map(|value| Expression::from(LiteralValue::from(value))),
    ]
    .boxed()
}

/// Return a numeric tree and a Boolean tree strategy of `depth` levels.
fn tree_strategies(depth: u32) -> (BoxedStrategy<Expression>, BoxedStrategy<Expression>) {
    let numeric = numeric_leaf();
    let boolean = prop_oneof![
        Just(Expression::from(IDENTIFIERS[2].clone())),
        any::<bool>().prop_map(|value| Expression::from(LiteralValue::from(value))),
    ]
    .boxed();
    (0..depth).fold((numeric, boolean), |(numeric, boolean), _| {
        let next_numeric = prop_oneof![
            numeric.clone(),
            (
                select(ARITHMETIC.to_vec()),
                numeric.clone(),
                numeric.clone()
            )
                .prop_map(|(operation, left, right)| Expression::new_binary(
                    operation, left, right
                )),
            numeric
                .clone()
                .prop_map(|operand| Expression::new_unary(UnaryOperation::Negate, operand)),
            (select(NATIVES.to_vec()), numeric.clone())
                .prop_map(|(function, argument)| Expression::call(function, [argument])),
            (boolean.clone(), numeric.clone(), numeric.clone()).prop_map(
                |(condition, value, otherwise)| {
                    Expression::piecewise([(condition, value)], otherwise)
                        .expect("a valid piecewise")
                }
            ),
        ]
        .boxed();
        let next_boolean = prop_oneof![
            boolean.clone(),
            (
                select(COMPARISONS.to_vec()),
                numeric.clone(),
                numeric.clone()
            )
                .prop_map(|(operation, left, right)| Expression::new_binary(
                    operation, left, right
                )),
            (boolean.clone(), boolean.clone())
                .prop_map(|(left, right)| Expression::all([left, right])),
            (boolean.clone(), boolean.clone())
                .prop_map(|(left, right)| Expression::any([left, right])),
            boolean.clone().prop_map(|operand| !operand),
        ]
        .boxed();
        (next_numeric, next_boolean)
    })
}

fn any_tree() -> BoxedStrategy<Expression> {
    let (numeric, boolean) = tree_strategies(4);
    prop_oneof![numeric, boolean].boxed()
}

fn lane_values() -> impl Strategy<Value = (i64, f64, bool)> {
    (
        prop_oneof![-3_i64..4, select(vec![0_i64, i64::MAX, i64::MIN, 1 << 40])],
        prop_oneof![
            -3.0_f64..3.0,
            select(vec![
                0.0,
                -0.0,
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
                1e300
            ])
        ],
        any::<bool>(),
    )
}

fn scalar_environment(lane: (i64, f64, bool)) -> HashMap<Identifier, Scalar> {
    HashMap::from([
        (IDENTIFIERS[0].clone(), Scalar::Int(lane.0)),
        (IDENTIFIERS[1].clone(), Scalar::Real(lane.1)),
        (IDENTIFIERS[2].clone(), Scalar::Bool(lane.2)),
    ])
}

/// Return whether two scalars are the same value, every NaN equal.
fn is_same_scalar(left: Scalar, right: Scalar) -> bool {
    match (left, right) {
        (Scalar::Real(a), Scalar::Real(b)) => {
            a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
        }
        _ => left == right,
    }
}

/// Return a comparable summary of an evaluation's outcome.
fn summarize(result: &Result<Scalar, EvaluationError>) -> String {
    match result {
        Ok(Scalar::Real(value)) if value.is_nan() => "NaN".to_owned(),
        Ok(value) => format!("{value:?}"),
        Err(EvaluationError::Lane { failure, .. }) => format!("lane failure: {failure}"),
        Err(error) => {
            let debug = format!("{error:?}");
            let kind = debug
                .split(['(', ' ', '{'])
                .next()
                .unwrap_or_default()
                .to_owned();
            format!("error: {kind}")
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn folding_does_not_change_the_value(tree in any_tree(), lane in lane_values()) {
        let registry = FunctionRegistry::new();
        let evaluator = Evaluator::new(&registry);
        let environment = scalar_environment(lane);
        // A literal call the fold refuses, such as `round(nan)`, is refused
        // wherever it sits, while the evaluation may discard its lane.
        let Ok(folding) = evaluator.fold(&tree, &NoNativeCalls) else {
            return Ok(());
        };

        let original = evaluator.evaluate(&tree, &environment);
        let folded = evaluator.evaluate(folding.output(), &environment);

        prop_assert_eq!(summarize(&original), summarize(&folded));
    }

    #[test]
    fn composed_builtins_evaluate_as_their_definitions(
        a in -5.0_f64..5.0,
        b in -5.0_f64..5.0,
        c in -5.0_f64..5.0,
    ) {
        let registry = FunctionRegistry::new();
        let evaluator = Evaluator::new(&registry);
        let literal = |value: f64| Expression::from(LiteralValue::from(value));
        let value = |function: BuiltinFunction, arguments: &[f64]| {
            let tree = Expression::call(Callee::Builtin(function), arguments.iter().map(|&x| literal(x)));
            evaluator.evaluate(&tree, &HashMap::new()).expect("a real call")
        };

        prop_assert_eq!(value(BuiltinFunction::Max, &[a, b]), Scalar::Real(a.max(b)));
        prop_assert_eq!(value(BuiltinFunction::Min, &[a, b]), Scalar::Real(a.min(b)));
        prop_assert_eq!(value(BuiltinFunction::Abs, &[a]), Scalar::Real(a.abs()));
        prop_assert_eq!(value(BuiltinFunction::Relu, &[a]), Scalar::Real(a.max(0.0)));
        let (low, high) = (b.min(c), b.max(c));
        prop_assert_eq!(value(BuiltinFunction::Clamp, &[a, low, high]), Scalar::Real(a.max(low).min(high)));
        let sign = if a > 0.0 { 1 } else if a < 0.0 { -1 } else { 0 };
        prop_assert!(is_same_scalar(value(BuiltinFunction::Sign, &[a]), Scalar::Int(sign)));
        prop_assert_eq!(value(BuiltinFunction::Sigmoid, &[a]), Scalar::Real(1.0 / (1.0 + (-a).exp())));
    }
}

#[cfg(feature = "ndarray")]
mod lanes {
    use super::*;

    use fhy_core::expression::evaluate::{ArrayBinding, ArrayValue, CoreKernels};
    use ndarray::{Array1, IxDyn};

    fn lane_scalar(value: &ArrayValue, index: usize) -> Scalar {
        let position = if value.shape().is_empty() {
            IxDyn(&[])
        } else {
            IxDyn(&[index])
        };
        match value {
            ArrayValue::Bool(array) => Scalar::Bool(array[position]),
            ArrayValue::Int(array) => Scalar::Int(array[position]),
            ArrayValue::Real(array) => Scalar::Real(array[position]),
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn each_lane_of_an_array_evaluation_is_the_scalar_evaluation_of_that_lane(
            tree in any_tree(),
            lanes in proptest::collection::vec(lane_values(), 1..8),
        ) {
            let registry = FunctionRegistry::new();
            let evaluator = Evaluator::new(&registry);
            let integers: Array1<i64> = lanes.iter().map(|lane| lane.0).collect();
            let reals: Array1<f64> = lanes.iter().map(|lane| lane.1).collect();
            let booleans: Array1<bool> = lanes.iter().map(|lane| lane.2).collect();
            let environment = HashMap::from([
                (IDENTIFIERS[0].clone(), ArrayBinding::Int(integers.view().into_dyn())),
                (IDENTIFIERS[1].clone(), ArrayBinding::Real(reals.view().into_dyn())),
                (IDENTIFIERS[2].clone(), ArrayBinding::Bool(booleans.view().into_dyn())),
            ]);
            let scalars: Vec<Result<Scalar, EvaluationError>> = lanes
                .iter()
                .map(|lane| evaluator.evaluate(&tree, &scalar_environment(*lane)))
                .collect();

            let prepared = evaluator.prepare(&tree).expect("no call to inline fails");
            match prepared.evaluate_array(&environment, &CoreKernels) {
                Ok(result) => {
                    for (index, scalar) in scalars.iter().enumerate() {
                        prop_assert!(scalar.is_ok(), "lane {} fails alone: {:?}", index, scalar);
                        let lane = lane_scalar(&result, index);
                        let scalar = *scalar.as_ref().expect("checked");
                        prop_assert!(is_same_scalar(lane, scalar), "lane {}: {:?} != {:?}", index, lane, scalar);
                    }
                }
                Err(EvaluationError::Lane { failure, node }) => {
                    let first = scalars.iter().position(Result::is_err).expect("some lane fails alone");
                    let Err(EvaluationError::Lane { failure: alone, node: alone_node }) = &scalars[first] else {
                        panic!("lane {first} fails differently: {:?}", scalars[first]);
                    };
                    prop_assert_eq!(failure, *alone);
                    prop_assert!(Expression::ptr_eq(&node, alone_node) || node == *alone_node);
                }
                Err(error) => {
                    let expected = summarize(&Err(error));
                    for scalar in &scalars {
                        prop_assert_eq!(summarize(scalar), expected.clone());
                    }
                }
            }
        }
    }
}
