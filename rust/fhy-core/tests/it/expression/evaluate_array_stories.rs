//! Tests for `Prepared::evaluate_array`: broadcasting, result domains,
//! borrowed bindings, lane failures in selected and unselected lanes, the
//! first failed lane, plugged-in kernels, and a million lanes.

use crate::support::expression as expression_support;

use std::cell::RefCell;
use std::collections::HashMap;

use fhy_core::expression::builtins::BuiltinFunction;
use fhy_core::expression::evaluate::{
    ArrayBinding, ArrayKernels, ArrayValue, CoreKernels, EvaluationError, Evaluator, LaneFailure,
    Scalar,
};
use fhy_core::expression::registry::FunctionRegistry;
use fhy_core::expression::{Expression, SymbolType};
use fhy_core::foreign::BoxError;
use fhy_core::identifier::Identifier;
use ndarray::{ArrayD, CowArray, IxDyn, arr0, array};

use expression_support::{build_identifier, build_literal, call};

fn piecewise(cases: Vec<(Expression, Expression)>, otherwise: impl Into<Expression>) -> Expression {
    Expression::piecewise(cases, otherwise).expect("a valid piecewise")
}

fn evaluate_with(
    expression: &Expression,
    bindings: Vec<(&Identifier, ArrayBinding<'_>)>,
    kernels: &dyn ArrayKernels,
) -> Result<ArrayValue, EvaluationError> {
    let registry = FunctionRegistry::new();
    let environment: HashMap<Identifier, ArrayBinding<'_>> = bindings
        .into_iter()
        .map(|(identifier, binding)| (identifier.clone(), binding))
        .collect();
    Evaluator::new(&registry)
        .prepare(expression)?
        .evaluate_array(&environment, kernels)
}

fn evaluate(
    expression: &Expression,
    bindings: Vec<(&Identifier, ArrayBinding<'_>)>,
) -> Result<ArrayValue, EvaluationError> {
    evaluate_with(expression, bindings, &CoreKernels)
}

fn expect_real(value: ArrayValue) -> ArrayD<f64> {
    match value {
        ArrayValue::Real(array) => array,
        other => panic!("expected reals, got {other:?}"),
    }
}

fn expect_int(value: ArrayValue) -> ArrayD<i64> {
    match value {
        ArrayValue::Int(array) => array,
        other => panic!("expected integers, got {other:?}"),
    }
}

fn expect_bool(value: ArrayValue) -> ArrayD<bool> {
    match value {
        ArrayValue::Bool(array) => array,
        other => panic!("expected Booleans, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Broadcasting and domains
// ---------------------------------------------------------------------------

#[test]
fn evaluate_array_computes_each_lane() {
    let (x, reference) = build_identifier("x");
    let values = array![1.0, 2.0, 3.0].into_dyn();

    let result = evaluate(
        &(&reference * &reference + 1),
        vec![(&x, ArrayBinding::Real(values.view()))],
    );

    assert_eq!(
        expect_real(result.unwrap()),
        array![2.0, 5.0, 10.0].into_dyn()
    );
}

#[test]
fn evaluate_array_broadcasts_as_numpy_does() {
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");
    let column = array![[1], [2]].into_dyn();
    let row = array![10, 20, 30].into_dyn();

    let result = evaluate(
        &(x_reference + y_reference),
        vec![
            (&x, ArrayBinding::Int(column.view())),
            (&y, ArrayBinding::Int(row.view())),
        ],
    );

    assert_eq!(
        expect_int(result.unwrap()),
        array![[11, 21, 31], [12, 22, 32]].into_dyn()
    );
}

#[test]
fn evaluate_array_refuses_shapes_that_do_not_broadcast() {
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");
    let two = array![1.0, 2.0].into_dyn();
    let three = array![1.0, 2.0, 3.0].into_dyn();

    let error = evaluate(
        &(x_reference + y_reference),
        vec![
            (&x, ArrayBinding::Real(two.view())),
            (&y, ArrayBinding::Real(three.view())),
        ],
    )
    .expect_err("2 and 3 do not broadcast");

    assert!(
        matches!(&error, EvaluationError::Shape { left, right } if left == &[2] && right == &[3])
    );
    assert_eq!(error.to_string(), "shapes [2] and [3] do not broadcast");
}

/// Return a column of shape `(rows, 1)` and a row of shape `(1, columns)`,
/// each a zero-stride view of one integer, as `numpy.broadcast_to` gives.
fn build_broadcast_pair(
    one: &ArrayD<i64>,
    rows: usize,
    columns: usize,
) -> (ndarray::ArrayViewD<'_, i64>, ndarray::ArrayViewD<'_, i64>) {
    let column = one.broadcast(IxDyn(&[rows, 1])).expect("a broadcast");
    let row = one.broadcast(IxDyn(&[1, columns])).expect("a broadcast");
    (column, row)
}

#[test]
fn a_broadcast_whose_lane_count_overflows_is_an_error() {
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");
    let one = arr0(1_i64).into_dyn();
    let (column, row) = build_broadcast_pair(&one, 1 << 33, 1 << 33);

    let error = evaluate(
        &(x_reference + y_reference),
        vec![
            (&x, ArrayBinding::Int(column)),
            (&y, ArrayBinding::Int(row)),
        ],
    )
    .expect_err("2^66 lanes do not fit");

    assert!(
        matches!(&error, EvaluationError::BroadcastTooLarge { shape } if shape == &[1 << 33, 1 << 33]),
        "got {error:?}"
    );
    assert_eq!(
        error.to_string(),
        "the broadcast shape [8589934592, 8589934592] has more lanes than an array can hold"
    );
}

#[test]
fn a_broadcast_too_large_to_reserve_is_an_error() {
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");
    let one = arr0(1_i64).into_dyn();
    let (column, row) = build_broadcast_pair(&one, 1 << 30, 1 << 29);

    let error = evaluate(
        &(x_reference + y_reference),
        vec![
            (&x, ArrayBinding::Int(column)),
            (&y, ArrayBinding::Int(row)),
        ],
    )
    .expect_err("2^59 integer lanes, 4 EiB, cannot be reserved on any platform");

    assert!(
        matches!(error, EvaluationError::OutOfMemory { lanes } if lanes == 1 << 59),
        "got {error:?}"
    );
    assert_eq!(
        error.to_string(),
        "cannot allocate the 576460752303423488 lanes of the result"
    );
}

#[test]
fn evaluate_array_keeps_zero_dimensional_and_empty_shapes() {
    let (x, reference) = build_identifier("x");
    let scalar = arr0(2.0).into_dyn();
    let empty = ArrayD::<f64>::zeros(IxDyn(&[0, 3]));

    let zero_dimensional = evaluate(
        &(&reference * 2),
        vec![(&x, ArrayBinding::Real(scalar.view()))],
    );
    let nothing = evaluate(
        &(&reference * 2),
        vec![(&x, ArrayBinding::Real(empty.view()))],
    );
    let literal = evaluate(&build_literal(1), vec![]);

    assert_eq!(expect_real(zero_dimensional.unwrap()), arr0(4.0).into_dyn());
    assert_eq!(nothing.unwrap().shape(), [0, 3]);
    assert_eq!(expect_int(literal.unwrap()), arr0(1).into_dyn());
}

#[test]
fn evaluate_array_yields_the_domain_of_each_operation() {
    let (x, reference) = build_identifier("x");
    let values = array![1, 3, 5].into_dyn();
    let binding = || vec![(&x, ArrayBinding::Int(values.view()))];

    let quotient = evaluate(&(&reference / 2), binding()).unwrap();
    let comparison = evaluate(&reference.greater(2), binding()).unwrap();
    let rounded = evaluate(&call(BuiltinFunction::Round, [&reference / 2]), binding()).unwrap();

    assert_eq!(expect_real(quotient), array![0.5, 1.5, 2.5].into_dyn());
    assert_eq!(
        expect_bool(comparison),
        array![false, true, true].into_dyn()
    );
    assert_eq!(rounded.symbol_type(), SymbolType::Int);
    assert_eq!(expect_int(rounded), array![0, 2, 2].into_dyn());
}

#[test]
fn evaluate_array_reads_strided_bindings_without_copying_them_into_the_result() {
    let (x, reference) = build_identifier("x");
    let values = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
    let transposed = values.t();

    let result =
        expect_real(evaluate(&reference, vec![(&x, ArrayBinding::Real(transposed))]).unwrap());

    assert_eq!(result, array![[1.0, 3.0], [2.0, 4.0]].into_dyn());
    assert_ne!(result.as_ptr(), values.as_ptr());
    assert!(result.is_standard_layout());
}

// ---------------------------------------------------------------------------
// Lane failures
// ---------------------------------------------------------------------------

#[test]
fn evaluate_array_discards_the_failures_of_unselected_lanes() {
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");
    let dividends = array![7, 7, 7].into_dyn();
    let divisors = array![2, 0, -2].into_dyn();
    let tree = piecewise(
        vec![(
            y_reference.not_equals(0),
            x_reference.floor_divide(&y_reference),
        )],
        0,
    );

    let result = evaluate(
        &tree,
        vec![
            (&x, ArrayBinding::Int(dividends.view())),
            (&y, ArrayBinding::Int(divisors.view())),
        ],
    );

    assert_eq!(expect_int(result.unwrap()), array![3, 0, -4].into_dyn());
}

#[test]
fn evaluate_array_guards_a_non_finite_cast_per_lane() {
    let (x, reference) = build_identifier("x");
    let values = array![4.0, -1.0, 10.0].into_dyn();
    let tree = piecewise(
        vec![(
            reference.greater_equal(0),
            call(
                BuiltinFunction::Floor,
                [call(BuiltinFunction::Sqrt, [reference.clone()])],
            ),
        )],
        -1,
    );

    let result = evaluate(&tree, vec![(&x, ArrayBinding::Real(values.view()))]);

    assert_eq!(expect_int(result.unwrap()), array![2, -1, 3].into_dyn());
}

#[test]
fn evaluate_array_raises_the_first_failed_lane_in_c_order() {
    let (x, reference) = build_identifier("x");
    let values = array![[1, 0], [i64::MIN, 0]].into_dyn();
    let tree = reference.floor_divide(&reference) + reference.floor_divide(-1);

    let error = evaluate(&tree, vec![(&x, ArrayBinding::Int(values.view()))]).expect_err("fails");

    assert!(matches!(
        &error,
        EvaluationError::Lane { failure: LaneFailure::DivisionByZero, node, lane: Some(1) } if node.to_string() == "(x // x)"
    ));
}

#[test]
fn evaluate_array_keeps_the_failure_of_the_operand_computed_first() {
    let (x, reference) = build_identifier("x");
    let values = array![i64::MIN].into_dyn();
    let tree = (-&reference).floor_divide(0);

    let error = evaluate(&tree, vec![(&x, ArrayBinding::Int(values.view()))]).expect_err("fails");

    assert!(matches!(
        error,
        EvaluationError::Lane {
            failure: LaneFailure::IntegerOverflow,
            ..
        }
    ));
}

#[test]
fn evaluate_array_discards_failures_decided_by_a_connective() {
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");
    let dividends = array![7, 7, 7].into_dyn();
    let divisors = array![0, 2, 7].into_dyn();
    let tree = Expression::all([
        y_reference.not_equals(0),
        x_reference.floor_divide(&y_reference).greater(1),
    ]);

    let result = evaluate(
        &tree,
        vec![
            (&x, ArrayBinding::Int(dividends.view())),
            (&y, ArrayBinding::Int(divisors.view())),
        ],
    );

    assert_eq!(
        expect_bool(result.unwrap()),
        array![false, true, false].into_dyn()
    );
}

#[test]
fn a_nested_piecewise_is_guarded_per_lane_by_its_outer_condition() {
    let (x, reference) = build_identifier("x");
    let values = array![-1, 3].into_dyn();
    let inner = piecewise(vec![(reference.less(0), reference.floor_divide(0))], 1);
    let tree = piecewise(vec![(reference.greater(0), inner)], 2);

    let result = evaluate(&tree, vec![(&x, ArrayBinding::Int(values.view()))]);

    assert_eq!(expect_int(result.unwrap()), array![2, 1].into_dyn());
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/// Kernels that compute `exp` as a constant and record their arguments.
#[derive(Default)]
struct RecordingKernels {
    arguments: RefCell<Vec<(BuiltinFunction, Vec<f64>, bool)>>,
}

impl ArrayKernels for RecordingKernels {
    fn handles(&self, function: BuiltinFunction) -> bool {
        matches!(function, BuiltinFunction::Exp | BuiltinFunction::Log)
    }

    fn native(
        &self,
        function: BuiltinFunction,
        argument: CowArray<'_, f64, IxDyn>,
    ) -> Result<ArrayD<f64>, BoxError> {
        self.arguments.borrow_mut().push((
            function,
            argument.iter().copied().collect(),
            argument.is_view(),
        ));
        if function == BuiltinFunction::Log {
            return Err("the kernel failed".into());
        }
        Ok(argument.mapv(|_| 7.0))
    }
}

#[test]
fn evaluate_array_computes_the_natives_a_kernel_handles_with_it() {
    let (x, reference) = build_identifier("x");
    let values = array![1.0, 2.0].into_dyn();
    let kernels = RecordingKernels::default();
    let tree = call(BuiltinFunction::Exp, [reference.clone()])
        + call(BuiltinFunction::Exp, [&reference + 1])
        + call(BuiltinFunction::Sqrt, [reference.clone()]);

    let result = evaluate_with(
        &tree,
        vec![(&x, ArrayBinding::Real(values.view()))],
        &kernels,
    );

    assert_eq!(
        expect_real(result.unwrap()),
        array![14.0 + 1.0, 14.0 + 2.0_f64.sqrt()].into_dyn()
    );
    let arguments = kernels.arguments.borrow();
    assert_eq!(arguments.len(), 2);
    assert_eq!(arguments[0], (BuiltinFunction::Exp, vec![1.0, 2.0], true));
    assert_eq!(arguments[1], (BuiltinFunction::Exp, vec![2.0, 3.0], false));
}

#[test]
fn evaluate_array_reports_a_failing_kernel() {
    let (x, reference) = build_identifier("x");
    let values = array![1.0].into_dyn();

    let error = evaluate_with(
        &call(BuiltinFunction::Log, [reference]),
        vec![(&x, ArrayBinding::Real(values.view()))],
        &RecordingKernels::default(),
    )
    .expect_err("the kernel fails");

    assert!(matches!(
        error,
        EvaluationError::Kernel {
            function: BuiltinFunction::Log,
            ..
        }
    ));
    assert_eq!(error.to_string(), "the array kernel of log failed");
    assert_eq!(
        std::error::Error::source(&error).map(ToString::to_string),
        Some("the kernel failed".to_owned())
    );
}

#[test]
fn evaluate_array_runs_a_million_lanes() {
    let (x, reference) = build_identifier("x");
    let values = ArrayD::from_shape_fn(IxDyn(&[1_000_000]), |index| {
        f64::from(u32::try_from(index[0]).expect("small"))
    });

    let result = expect_real(
        evaluate(
            &(&reference * 2 + 1),
            vec![(&x, ArrayBinding::Real(values.view()))],
        )
        .unwrap(),
    );

    assert_eq!(result.len(), 1_000_000);
    assert_eq!(result[[999_999]].to_bits(), 1_999_999.0_f64.to_bits());
}

// ---------------------------------------------------------------------------
// Chunks
// ---------------------------------------------------------------------------

/// More lanes than one chunk holds.
const MANY_LANES: usize = 200_003;

fn build_reals(count: usize, lane: impl Fn(usize) -> f64) -> ArrayD<f64> {
    ArrayD::from_shape_fn(IxDyn(&[count]), |index| lane(index[0]))
}

#[test]
fn evaluate_array_computes_every_chunk_as_one_evaluation_would() {
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");
    let values = build_reals(MANY_LANES, |index| {
        f64::from(u32::try_from(index).unwrap()) - 1e5
    });
    let tree = piecewise(
        vec![(x_reference.greater(0), &x_reference * &y_reference)],
        call(BuiltinFunction::Exp, [x_reference.clone() / 1e5]),
    );

    let result = expect_real(
        evaluate(
            &tree,
            vec![
                (&x, ArrayBinding::Real(values.view())),
                (&y, ArrayBinding::Real(arr0(3.0).into_dyn().view())),
            ],
        )
        .unwrap(),
    );

    assert_eq!(result.shape(), [MANY_LANES]);
    for (index, (&lane, &x)) in result.iter().zip(values.iter()).enumerate() {
        let expected = if x > 0.0 { x * 3.0 } else { (x / 1e5).exp() };
        assert_eq!(lane.to_bits(), expected.to_bits(), "lane {index}");
    }
}

#[test]
fn evaluate_array_broadcasts_bindings_across_chunks() {
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");
    let column = ArrayD::from_shape_fn(IxDyn(&[500, 1]), |index| i64::try_from(index[0]).unwrap());
    let row = ArrayD::from_shape_fn(IxDyn(&[1_000]), |index| {
        i64::try_from(index[0]).unwrap() * 1_000
    });
    let transposed = ArrayD::from_shape_fn(IxDyn(&[1_000, 500]), |index| {
        i64::try_from(index[0] + index[1]).unwrap()
    });
    let (z, z_reference) = build_identifier("z");

    let result = expect_int(
        evaluate(
            &(x_reference + y_reference + z_reference),
            vec![
                (&x, ArrayBinding::Int(column.view())),
                (&y, ArrayBinding::Int(row.view())),
                (&z, ArrayBinding::Int(transposed.t().into_dyn())),
            ],
        )
        .unwrap(),
    );

    assert_eq!(result.shape(), [500, 1_000]);
    assert!(result.is_standard_layout());
    assert_eq!(result[[499, 999]], 499 + 999_000 + 999 + 499);
    assert_eq!(result[[3, 7]], 3 + 7_000 + 7 + 3);
}

#[test]
fn evaluate_array_raises_the_first_failed_lane_of_a_later_chunk() {
    let (x, reference) = build_identifier("x");
    let values = ArrayD::from_shape_fn(IxDyn(&[MANY_LANES]), |index| match index[0] {
        150_000 => 0_i64,
        199_999 => i64::MIN,
        _ => 1,
    });
    let tree = (-&reference).floor_divide(&reference);

    let error = evaluate(&tree, vec![(&x, ArrayBinding::Int(values.view()))]).expect_err("fails");

    assert!(matches!(
        error,
        EvaluationError::Lane {
            failure: LaneFailure::DivisionByZero,
            lane: Some(150_000),
            ..
        }
    ));
}

#[test]
fn evaluate_array_hands_a_kernel_each_chunk() {
    let (x, reference) = build_identifier("x");
    let values = build_reals(MANY_LANES, |_| 1.0);
    let kernels = RecordingKernels::default();

    let result = expect_real(
        evaluate_with(
            &call(BuiltinFunction::Exp, [reference]),
            vec![(&x, ArrayBinding::Real(values.view()))],
            &kernels,
        )
        .unwrap(),
    );

    assert_eq!(result.len(), MANY_LANES);
    assert!(
        result
            .iter()
            .all(|&lane| lane.to_bits() == 7.0_f64.to_bits())
    );
    let arguments = kernels.arguments.borrow();
    assert_eq!(
        arguments
            .iter()
            .map(|(_, lanes, _)| lanes.len())
            .sum::<usize>(),
        MANY_LANES
    );
    assert!(arguments.len() > 1);
}

/// Kernels computing `exp` that record the address of each argument's
/// lanes and of each result's.
#[derive(Default)]
struct AddressKernels {
    addresses: RefCell<Vec<(usize, usize)>>,
}

impl ArrayKernels for AddressKernels {
    fn handles(&self, function: BuiltinFunction) -> bool {
        function == BuiltinFunction::Exp
    }

    fn native(
        &self,
        _function: BuiltinFunction,
        argument: CowArray<'_, f64, IxDyn>,
    ) -> Result<ArrayD<f64>, BoxError> {
        let result = argument.mapv(f64::exp);
        self.addresses
            .borrow_mut()
            .push((argument.as_ptr().addr(), result.as_ptr().addr()));
        Ok(result)
    }
}

#[test]
fn unary_plus_passes_its_operands_lanes_through() {
    let (x, reference) = build_identifier("x");
    let values = array![0.0, 1.0].into_dyn();
    let kernels = AddressKernels::default();
    let tree = call(
        BuiltinFunction::Exp,
        [call(BuiltinFunction::Exp, [reference]).positive()],
    );

    let result = expect_real(
        evaluate_with(
            &tree,
            vec![(&x, ArrayBinding::Real(values.view()))],
            &kernels,
        )
        .unwrap(),
    );

    assert_eq!(result, values.mapv(|lane| lane.exp().exp()));
    let addresses = kernels.addresses.borrow();
    assert_eq!(addresses.len(), 2);
    assert_eq!(
        addresses[1].0, addresses[0].1,
        "the outer exp reads the inner exp's lanes, not a copy made by +"
    );
}

#[test]
fn a_lane_error_names_its_lane() {
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");
    let numerators = array![[1_i64, 2, 3], [4, 5, 6]].into_dyn();
    let divisors = array![1_i64, 1, 0].into_dyn();

    let error = evaluate(
        &x_reference.floor_divide(&y_reference),
        vec![
            (&x, ArrayBinding::Int(numerators.view())),
            (&y, ArrayBinding::Int(divisors.view())),
        ],
    )
    .expect_err("a division by zero");

    assert!(
        matches!(
            error,
            EvaluationError::Lane {
                failure: LaneFailure::DivisionByZero,
                lane: Some(2),
                ..
            }
        ),
        "{error:?}"
    );
    assert_eq!(
        error.to_string(),
        "integer division by zero at lane 2 in (x // y)"
    );
}

#[test]
fn a_lane_error_names_its_lane_in_the_whole_result_when_its_value_broadcasts() {
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");
    let rows = array![[2_i64], [1]].into_dyn();
    let row = array![5_i64, 6, 7].into_dyn();

    let error = evaluate(
        &(x_reference.floor_divide(&x_reference - 1) + y_reference),
        vec![
            (&x, ArrayBinding::Int(rows.view())),
            (&y, ArrayBinding::Int(row.view())),
        ],
    )
    .expect_err("a division by zero");

    assert!(
        matches!(error, EvaluationError::Lane { lane: Some(3), .. }),
        "{error:?}"
    );
}

// ---------------------------------------------------------------------------
// Unary plus, Boolean piecewise, Boolean and transposed bindings
// ---------------------------------------------------------------------------

#[test]
fn unary_plus_of_integers_and_reals_is_the_operand() {
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");
    let integers = array![3_i64, -4].into_dyn();
    let reals = array![2.5, -0.0].into_dyn();

    let plus_integers = evaluate(
        &x_reference.positive(),
        vec![(&x, ArrayBinding::Int(integers.view()))],
    );
    let plus_reals = expect_real(
        evaluate(
            &y_reference.positive(),
            vec![(&y, ArrayBinding::Real(reals.view()))],
        )
        .unwrap(),
    );

    assert_eq!(expect_int(plus_integers.unwrap()), integers);
    assert_eq!(
        plus_reals
            .iter()
            .map(|lane| lane.to_bits())
            .collect::<Vec<_>>(),
        [2.5_f64.to_bits(), (-0.0_f64).to_bits()]
    );
}

#[test]
fn a_boolean_piecewise_selects_booleans_per_lane() {
    let (p, p_reference) = build_identifier("p");
    let (q, q_reference) = build_identifier("q");
    let conditions = array![true, false, true].into_dyn();
    let values = array![false, true, true].into_dyn();
    let tree = piecewise(vec![(p_reference, q_reference.clone())], !&q_reference);

    let result = evaluate(
        &tree,
        vec![
            (&p, ArrayBinding::Bool(conditions.view())),
            (&q, ArrayBinding::Bool(values.view())),
        ],
    );

    assert_eq!(
        expect_bool(result.unwrap()),
        array![false, false, true].into_dyn()
    );
}

/// Test Boolean and transposed real bindings, of a lane count below the
/// chunk size and of one above it, evaluate as their lanes say.
#[rstest::rstest]
#[case::below_the_chunk_size(3, 5)]
#[case::above_the_chunk_size(300, 401)]
fn boolean_and_transposed_bindings_evaluate_lane_by_lane(
    #[case] rows: usize,
    #[case] columns: usize,
) {
    let (p, p_reference) = build_identifier("p");
    let (r, r_reference) = build_identifier("r");
    let booleans = ArrayD::from_shape_fn(IxDyn(&[rows, columns]), |index| {
        (index[0] + index[1]) % 3 == 0
    });
    let storage = ArrayD::from_shape_fn(IxDyn(&[columns, rows]), |index| {
        f64::from(u32::try_from(index[0] * 10 + index[1]).unwrap())
    });
    let transposed = storage.t();
    let tree = piecewise(vec![(p_reference, r_reference.clone())], -&r_reference);

    let result = expect_real(
        evaluate(
            &tree,
            vec![
                (&p, ArrayBinding::Bool(booleans.view())),
                (&r, ArrayBinding::Real(transposed.view().into_dyn())),
            ],
        )
        .unwrap(),
    );

    assert_eq!(result.shape(), [rows, columns]);
    assert!(result.is_standard_layout());
    for row in 0..rows {
        for column in 0..columns {
            let lane = transposed[[row, column]];
            let expected = if booleans[[row, column]] { lane } else { -lane };
            assert_eq!(result[[row, column]].to_bits(), expected.to_bits());
        }
    }
}

/// Kernels computing `exp` that return an array of the wrong shape.
struct MisshapenKernels;

impl ArrayKernels for MisshapenKernels {
    fn handles(&self, function: BuiltinFunction) -> bool {
        function == BuiltinFunction::Exp
    }

    fn native(
        &self,
        _function: BuiltinFunction,
        argument: CowArray<'_, f64, IxDyn>,
    ) -> Result<ArrayD<f64>, BoxError> {
        Ok(ArrayD::zeros(IxDyn(&[argument.len() + 1])))
    }
}

#[test]
fn a_kernel_returning_the_wrong_shape_is_an_error() {
    let (x, reference) = build_identifier("x");
    let values = array![1.0, 2.0].into_dyn();

    let error = evaluate_with(
        &call(BuiltinFunction::Exp, [reference]),
        vec![(&x, ArrayBinding::Real(values.view()))],
        &MisshapenKernels,
    )
    .expect_err("the kernel's shape is refused");

    assert!(
        matches!(
            &error,
            EvaluationError::Kernel {
                function: BuiltinFunction::Exp,
                ..
            }
        ),
        "{error:?}"
    );
    assert_eq!(error.to_string(), "the array kernel of exp failed");
    assert_eq!(
        std::error::Error::source(&error).map(ToString::to_string),
        Some("the kernel returned shape [3] for an argument of shape [2]".to_owned())
    );
}

/// Test a chunked evaluation of 300,003 lanes, over a column binding, a
/// row binding and a transposed one, with lane failures that a connective
/// or a piecewise discards in some lanes only, equals the scalar evaluation
/// of every lane.
#[test]
fn a_chunked_evaluation_equals_the_scalar_evaluation_of_every_lane() {
    let (x, ex) = build_identifier("x");
    let (y, ey) = build_identifier("y");
    let (r, er) = build_identifier("r");
    let guard = ex.not_equals(0_i64).and(Expression::any([
        Expression::from(10_i64).floor_divide(&ex).greater(&ey),
        Expression::from(1_i64)
            .floor_divide(&ex - 1_i64)
            .equals(0_i64),
    ]));
    let tree = piecewise(
        vec![
            (guard, ex.floor_mod(7_i64) + &ey),
            (er.greater(0.5), call(BuiltinFunction::Floor, [&er * 100.0])),
        ],
        (&ex + 5_i64).floor_divide(&ey - 1_i64),
    );
    let (rows, columns) = (3, 100_001);
    let xs = ArrayD::from_shape_fn(IxDyn(&[columns]), |index| {
        i64::try_from(index[0] % 11).unwrap() - 5
    });
    let ys = ArrayD::from_shape_fn(IxDyn(&[rows, 1]), |index| {
        i64::try_from(index[0]).unwrap() - 1
    });
    let storage = ArrayD::from_shape_fn(IxDyn(&[columns, rows]), |index| {
        f64::from(u32::try_from((index[0] * 7 + index[1] * 3) % 10).unwrap()) / 10.0
    });
    let rs = storage.t();
    let registry = FunctionRegistry::new();
    let prepared = Evaluator::new(&registry).prepare(&tree).unwrap();
    let environment = HashMap::from([
        (x.clone(), ArrayBinding::Int(xs.view())),
        (y.clone(), ArrayBinding::Int(ys.view())),
        (r.clone(), ArrayBinding::Real(rs.view())),
    ]);

    let array = prepared.evaluate_array(&environment, &CoreKernels);

    let mut first_failure = None;
    let mut expected = Vec::with_capacity(rows * columns);
    let mut scalars = HashMap::new();
    for row in 0..rows {
        for column in 0..columns {
            scalars.insert(x.clone(), Scalar::Int(xs[[column]]));
            scalars.insert(y.clone(), Scalar::Int(ys[[row, 0]]));
            scalars.insert(r.clone(), Scalar::Real(rs[[row, column]]));
            match prepared.evaluate(&scalars) {
                Ok(Scalar::Int(value)) => expected.push(value),
                Ok(other) => panic!("an integer lane, got {other:?}"),
                Err(error) => {
                    first_failure.get_or_insert((row * columns + column, error.to_string()));
                    expected.push(0);
                }
            }
        }
    }
    match (array, first_failure) {
        (Ok(ArrayValue::Int(values)), None) => {
            assert_eq!(values.shape(), [rows, columns]);
            assert!(values.iter().copied().eq(expected));
        }
        (Err(error), Some((lane, text))) => {
            let EvaluationError::Lane {
                lane: Some(failed), ..
            } = &error
            else {
                panic!("a lane failure, got {error:?}");
            };
            assert_eq!(*failed, lane);
            assert_eq!(
                error.to_string(),
                text.replacen(" in ", &format!(" at lane {lane} in "), 1)
            );
        }
        (array, first_failure) => {
            panic!("array {array:?} against the first scalar failure {first_failure:?}")
        }
    }
}
