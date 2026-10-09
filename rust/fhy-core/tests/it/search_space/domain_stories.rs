//! Tests for the step domains: `ChoiceDomain`, `OrderDomain`, `StridedRun`,
//! `StridedDomain` and `StepDomain`, their refusals, coordinates and
//! admission, `DecisionKind`, and the `DomainSignature` a replay compares.

use fhy_core::constraint::Value;
use fhy_core::expression::BigInt;
use fhy_core::identifier::Identifier;
use fhy_core::search_space::{
    ChoiceDomain, Coordinate, DecisionKind, EmptyKind, OrderDomain, StepDomain, StepDomainError,
    StridedDomain, StridedRun,
};
use num_bigint::BigUint;
use rstest::rstest;

use crate::support::constraint::{TestOpaque, int, text};
use crate::support::search::{index, int_choices, order, order_of, run, strided};

/// Return the order domain over three fresh identifiers, and them.
fn build_order_of_three() -> (OrderDomain, [Identifier; 3]) {
    let elements = ["i", "j", "k"].map(Identifier::new);
    let domain = OrderDomain::new(elements.iter().cloned().map(Value::Identifier).collect())
        .expect("three distinct elements");
    (domain, elements)
}

/// Return the tuple of the identifiers `elements`, in order.
fn ordering(elements: &[&Identifier]) -> Value {
    Value::Tuple(
        elements
            .iter()
            .map(|&element| Value::Identifier(element.clone()))
            .collect(),
    )
}

// ---------------------------------------------------------------------------
// DecisionKind
// ---------------------------------------------------------------------------

/// Test a kind keeps its name, displays it, and equals a kind of that name.
#[test]
fn decision_kind_is_its_name() {
    let kind = DecisionKind::new("moga.cir.address").expect("a named kind");

    assert_eq!(kind.as_str(), "moga.cir.address");
    assert_eq!(kind.to_string(), "moga.cir.address");
    assert_eq!(
        kind,
        DecisionKind::new("moga.cir.address").expect("a named kind")
    );
    assert_ne!(
        kind,
        DecisionKind::new("moga.cir.tile").expect("a named kind")
    );
}

/// Test a kind needs a name.
#[test]
fn decision_kind_refuses_an_empty_name() {
    let result = DecisionKind::new("");

    assert_eq!(result, Err(EmptyKind));
}

/// Test the choice kind is `search_space.choice`.
#[test]
fn decision_kind_of_a_choice_is_search_space_choice() {
    let kind = DecisionKind::choice();

    assert_eq!(kind.as_str(), "search_space.choice");
    assert_eq!(kind.as_str(), DecisionKind::CHOICE);
}

/// Test a kind writes its name as a string and reading refuses an empty
/// one.
#[test]
fn decision_kind_serializes_as_its_name() {
    let kind = DecisionKind::new("moga.cir.tile").expect("a named kind");

    let text = serde_json::to_string(&kind).expect("a kind serializes");

    assert_eq!(text, r#""moga.cir.tile""#);
    assert_eq!(
        serde_json::from_str::<DecisionKind>(&text).expect("the name decodes"),
        kind
    );
    let error = serde_json::from_str::<DecisionKind>(r#""""#).expect_err("an empty name");
    assert!(error.to_string().contains("needs a name"), "{error}");
}

// ---------------------------------------------------------------------------
// ChoiceDomain
// ---------------------------------------------------------------------------

/// Test a choice domain refuses no value.
#[test]
fn choice_domain_refuses_no_values() {
    let result = ChoiceDomain::new(Vec::new());

    assert_eq!(result.expect_err("no value"), StepDomainError::EmptyChoice);
}

/// Test a choice domain refuses a NaN, at the top or inside a tuple, before
/// it looks for repeats.
#[rstest]
#[case::top(vec![int(1), Value::Float(f64::NAN), int(1)], 1)]
#[case::nested(vec![Value::Tuple(vec![int(1), Value::Float(f64::NAN)]), int(2)], 0)]
fn choice_domain_refuses_a_nan(#[case] values: Vec<Value>, #[case] at: usize) {
    let result = ChoiceDomain::new(values);

    assert_eq!(
        result.expect_err("a NaN"),
        StepDomainError::NanValue { index: at }
    );
}

/// Test a choice domain refuses two equal values, naming both positions.
#[test]
fn choice_domain_refuses_a_repeated_value() {
    let result = ChoiceDomain::new(vec![text("x"), int(2), text("x")]);

    assert_eq!(
        result.expect_err("a repeat"),
        StepDomainError::RepeatedValue {
            first: 0,
            second: 2
        }
    );
}

/// Test values are distinct type-strictly: `1`, `true` and `1.0` are three
/// values, each at its own coordinate.
#[test]
fn choice_domain_tells_one_true_and_one_point_zero_apart() {
    let domain =
        ChoiceDomain::new(vec![int(1), Value::Bool(true), Value::Float(1.0)]).expect("distinct");

    assert_eq!(domain.cardinality(), 3);
    assert_eq!(domain.coordinate_of(&int(1)), Some(0));
    assert_eq!(domain.coordinate_of(&Value::Bool(true)), Some(1));
    assert_eq!(domain.coordinate_of(&Value::Float(1.0)), Some(2));
}

/// Test a choice domain keeps its values in the order given and maps
/// coordinates to values and back.
#[test]
fn choice_domain_coordinates_round_trip() {
    let domain = ChoiceDomain::new(vec![text("a"), text("b"), text("c")]).expect("distinct");

    let values: Vec<Value> = (0..domain.cardinality())
        .map(|index| domain.value_at(index).expect("in range").clone())
        .collect();

    assert_eq!(values, [text("a"), text("b"), text("c")]);
    assert_eq!(domain.values(), [text("a"), text("b"), text("c")]);
    for (position, value) in values.iter().enumerate() {
        assert_eq!(
            domain.coordinate_of(value),
            Some(u64::try_from(position).expect("small"))
        );
    }
}

/// Test a choice domain answers nothing past its last value and for a
/// value it does not hold.
#[test]
fn choice_domain_answers_none_outside_itself() {
    let domain = int_choices(&[10, 20, 30]);
    let StepDomain::Choice(domain) = domain else {
        panic!("a choice domain");
    };

    assert_eq!(domain.value_at(3), None);
    assert_eq!(domain.coordinate_of(&int(99)), None);
    assert!(!domain.admits(&int(99)));
    assert!(domain.admits(&int(20)));
}

/// Test opaque values match by their own equality: an equal opaque value
/// is admitted, an unequal one is not.
#[test]
fn choice_domain_admits_opaque_values_by_their_own_equality() {
    let domain = ChoiceDomain::new(vec![
        TestOpaque::token(1).into_value(),
        TestOpaque::token(2).into_value(),
    ])
    .expect("distinct tokens");

    assert_eq!(
        domain.coordinate_of(&TestOpaque::token(2).into_value()),
        Some(1)
    );
    assert!(!domain.admits(&TestOpaque::token(3).into_value()));
}

// ---------------------------------------------------------------------------
// OrderDomain
// ---------------------------------------------------------------------------

/// Test an order domain counts `n!` orderings and admits only tuples
/// holding each element once.
#[test]
fn order_domain_counts_and_admits_only_permutations() {
    let (domain, [i, j, k]) = build_order_of_three();

    assert_eq!(domain.cardinality(), BigUint::from(6_u8));
    assert!(domain.admits(&ordering(&[&k, &i, &j])));
    assert!(domain.admits(&ordering(&[&i, &j, &k])));
    assert!(!domain.admits(&ordering(&[&i, &j])));
    assert!(!domain.admits(&ordering(&[&i, &j, &j])));
    assert!(!domain.admits(&Value::FrozenSet(vec![
        Value::Identifier(i.clone()),
        Value::Identifier(j.clone()),
        Value::Identifier(k.clone()),
    ])));
}

/// Test an order domain's cardinality grows as the factorial, past what a
/// machine word holds.
#[test]
fn order_domain_cardinality_is_the_factorial() {
    let domain = OrderDomain::new((0..25).map(int).collect()).expect("distinct");

    let expected: BigUint = (1_u32..=25).map(BigUint::from).product();
    assert_eq!(domain.cardinality(), expected);
}

/// Test an order domain refuses no element and a repeated one.
#[rstest]
#[case::empty(Vec::new(), StepDomainError::EmptyOrder)]
#[case::repeated(vec![int(1), int(1)], StepDomainError::RepeatedValue { first: 0, second: 1 })]
fn order_domain_refuses_bad_elements(#[case] elements: Vec<Value>, #[case] error: StepDomainError) {
    let result = OrderDomain::new(elements);

    assert_eq!(result.expect_err("refused"), error);
}

/// Test a coordinate names positions: `(2, 0, 1)` is third, first, second,
/// and the same coordinate replays onto another domain's own elements.
#[test]
fn order_coordinates_round_trip_onto_fresh_elements() {
    let (first, [i, j, k]) = build_order_of_three();
    let (second, [i2, j2, k2]) = build_order_of_three();

    let value = first.value_at(&[2, 0, 1]).expect("a permutation");
    let coordinate = first.coordinate_of(&value).expect("an ordering");
    let replayed = second.value_at(&coordinate).expect("a permutation");

    assert_eq!(value, ordering(&[&k, &i, &j]));
    assert_eq!(&*coordinate, [2, 0, 1]);
    assert_eq!(replayed, ordering(&[&k2, &i2, &j2]));
}

/// Test an order domain names no ordering for positions that are not a
/// permutation.
#[rstest]
#[case::repeat(vec![0, 0, 1])]
#[case::short(vec![0, 1])]
#[case::out_of_range(vec![0, 1, 3])]
fn order_value_at_refuses_a_non_permutation(#[case] positions: Vec<u32>) {
    let (domain, _) = build_order_of_three();

    let value = domain.value_at(&positions);

    assert_eq!(value, None);
}

// ---------------------------------------------------------------------------
// StridedRun and StridedDomain
// ---------------------------------------------------------------------------

/// Test a run refuses a stop not above its start, and a zero stride.
#[rstest]
#[case::empty(64, 64, 1, StepDomainError::EmptyRun)]
#[case::backwards(10, 5, 1, StepDomainError::EmptyRun)]
#[case::zero_stride(0, 10, 0, StepDomainError::ZeroStride)]
fn strided_run_refuses_a_bad_run(
    #[case] start: i64,
    #[case] stop: i64,
    #[case] stride: u64,
    #[case] error: StepDomainError,
) {
    let result = StridedRun::new(
        BigInt::from(start),
        BigInt::from(stop),
        BigUint::from(stride),
    );

    assert_eq!(result.expect_err("refused"), error);
}

/// Test a strided run admits only the integers on its stride, below its
/// stop.
#[test]
fn strided_run_admits_only_its_stride() {
    let banked = run(64, 200, 64);

    assert_eq!(banked.width(), BigUint::from(3_u8));
    assert_eq!(banked.start(), &BigInt::from(64));
    assert_eq!(banked.stop(), &BigInt::from(200));
    assert_eq!(banked.stride(), &BigUint::from(64_u8));
    assert!(banked.admits(&BigInt::from(64)));
    assert!(banked.admits(&BigInt::from(192)));
    assert!(!banked.admits(&BigInt::from(65)));
    assert!(!banked.admits(&BigInt::from(200)));
    assert!(!banked.admits(&BigInt::from(0)));
}

/// Test a strided domain numbers its runs' integers in order, so every
/// index names a distinct integer and each run counts by its width.
#[test]
fn strided_domain_flattens_its_runs() {
    let StepDomain::Strided(domain) = strided(&[(0, 3), (100, 105)]) else {
        panic!("a strided domain");
    };

    let values: Vec<BigInt> = (0..domain.cardinality())
        .map(|index| domain.value_at(index).expect("in range"))
        .collect();

    assert_eq!(domain.cardinality(), 8);
    assert_eq!(values, [0, 1, 2, 100, 101, 102, 103, 104].map(BigInt::from));
}

/// Test a strided domain indexes only its strides, across runs.
#[test]
fn strided_domain_indexes_only_its_strides() {
    let domain = StridedDomain::new(vec![run(0, 100, 32), run(256, 300, 32)]).expect("valid");

    let values: Vec<BigInt> = (0..domain.cardinality())
        .map(|index| domain.value_at(index).expect("in range"))
        .collect();

    assert_eq!(values, [0, 32, 64, 96, 256, 288].map(BigInt::from));
    assert!(!domain.admits(&int(16)));
}

/// Test a strided domain admits nothing between its runs, and no Boolean.
#[test]
fn strided_domain_admits_nothing_between_runs_nor_a_boolean() {
    let StepDomain::Strided(domain) = strided(&[(0, 3), (100, 105)]) else {
        panic!("a strided domain");
    };

    assert!(!domain.admits(&int(3)));
    assert!(!domain.admits(&int(99)));
    assert!(!domain.admits(&int(105)));
    assert!(!domain.admits(&int(-1)));
    assert!(!domain.admits(&Value::Bool(true)));
    assert!(!domain.admits(&Value::Float(1.0)));
    assert!(domain.admits(&int(1)));
}

/// Test a strided domain names nothing past its last integer and finds no
/// coordinate for an integer between its runs.
#[test]
fn strided_value_at_past_the_end_is_none() {
    let StepDomain::Strided(domain) = strided(&[(0, 4)]) else {
        panic!("a strided domain");
    };

    assert_eq!(domain.value_at(4), None);
    assert_eq!(domain.coordinate_of(&BigInt::from(7)), None);
}

/// Test a strided domain's coordinates round-trip over every integer.
#[test]
fn strided_coordinates_round_trip() {
    let domain = StridedDomain::new(vec![run(0, 8, 4), run(100, 112, 4)]).expect("valid");

    let coordinates: Vec<Option<u64>> = (0..domain.cardinality())
        .map(|index| domain.coordinate_of(&domain.value_at(index).expect("in range")))
        .collect();

    assert_eq!(
        coordinates,
        (0..domain.cardinality()).map(Some).collect::<Vec<_>>()
    );
    assert_eq!(domain.coordinate_of(&BigInt::from(3)), None);
}

/// Test a strided domain refuses no run, and runs out of order or
/// overlapping, naming the run.
#[rstest]
#[case::none(Vec::new(), StepDomainError::EmptyRuns)]
#[case::overlapping(vec![(0, 10), (5, 20)], StepDomainError::UnorderedRuns { index: 1 })]
#[case::unsorted(vec![(100, 110), (0, 10)], StepDomainError::UnorderedRuns { index: 1 })]
fn strided_domain_refuses_bad_runs(#[case] runs: Vec<(i64, i64)>, #[case] error: StepDomainError) {
    let result = StridedDomain::new(
        runs.into_iter()
            .map(|(start, stop)| run(start, stop, 1))
            .collect(),
    );

    assert_eq!(result.expect_err("refused"), error);
}

/// Test adjacent runs, one starting where the previous stops, are allowed.
#[test]
fn strided_domain_allows_adjacent_runs() {
    let domain = StridedDomain::new(vec![run(0, 4, 1), run(4, 8, 1)]).expect("adjacent runs");

    assert_eq!(domain.cardinality(), 8);
}

/// Test a strided domain of more integers than a coordinate numbers is
/// refused.
#[test]
fn strided_domain_refuses_more_integers_than_coordinates_number() {
    let huge = StridedRun::new(
        BigInt::from(0),
        BigInt::from(1_u8) << 70_u32,
        BigUint::from(1_u8),
    )
    .expect("a valid run");

    let result = StridedDomain::new(vec![huge]);

    assert_eq!(result.expect_err("too large"), StepDomainError::TooLarge);
}

// ---------------------------------------------------------------------------
// StepDomain
// ---------------------------------------------------------------------------

/// Test a step domain answers for each shape: cardinality, which
/// coordinates it contains, the value one names and the coordinate of a
/// value.
#[rstest]
#[case::choice(int_choices(&[5, 6]), index(1), Some(int(6)), 2_u32)]
#[case::strided(strided(&[(10, 13)]), index(2), Some(int(12)), 3_u32)]
#[case::choice_past_the_end(int_choices(&[5, 6]), index(2), None, 2_u32)]
#[case::wrong_shape(int_choices(&[5, 6]), order(&[0, 1]), None, 2_u32)]
fn step_domain_answers_per_shape(
    #[case] domain: StepDomain,
    #[case] coordinate: Coordinate,
    #[case] value: Option<Value>,
    #[case] cardinality: u32,
) {
    let found = domain.value_at(&coordinate);

    assert_eq!(found, value);
    assert_eq!(domain.contains(&coordinate), value.is_some());
    assert_eq!(domain.cardinality(), BigUint::from(cardinality));
    if let Some(value) = &value {
        assert_eq!(domain.coordinate_of(value), Some(coordinate.clone()));
        assert!(domain.admits(value));
    }
}

/// Test an order step domain names orderings by order coordinates only.
#[test]
fn step_domain_of_an_order_takes_order_coordinates() {
    let [i, j] = ["i", "j"].map(Identifier::new);
    let domain = order_of(&[&i, &j]);

    assert_eq!(domain.value_at(&order(&[1, 0])), Some(ordering(&[&j, &i])));
    assert!(domain.contains(&order(&[1, 0])));
    assert!(!domain.contains(&index(0)));
    assert_eq!(
        domain.coordinate_of(&ordering(&[&j, &i])),
        Some(order(&[1, 0]))
    );
    assert_eq!(domain.cardinality(), BigUint::from(2_u8));
}

// ---------------------------------------------------------------------------
// DomainSignature
// ---------------------------------------------------------------------------

/// Test signatures keep shape and size.
#[test]
fn domain_signature_keeps_shape_and_cardinality() {
    let signature = strided(&[(0, 3), (100, 105)]).signature();

    assert_eq!(signature.shape(), "strided");
    assert_eq!(signature.cardinality(), BigUint::from(8_u8));
    assert!(signature.contains(&index(7)));
    assert!(!signature.contains(&index(8)));
    assert_eq!(int_choices(&[1]).signature().shape(), "choice");
    assert_eq!(
        order_of(&[&Identifier::new("i")]).signature().shape(),
        "order"
    );
}

/// Test plain values are part of a signature: reordered or moved plain
/// domains of one size have different signatures.
#[rstest]
#[case::reordered_choices(int_choices(&[1, 2, 3]), int_choices(&[3, 2, 1]))]
#[case::moved_run(strided(&[(0, 64)]), strided(&[(64, 128)]))]
#[case::other_values(int_choices(&[1, 2]), int_choices(&[1, 4]))]
fn domain_signature_tells_plain_values_apart(#[case] left: StepDomain, #[case] right: StepDomain) {
    let (left, right) = (left.signature(), right.signature());

    assert_ne!(left, right);
    assert_eq!(left.cardinality(), right.cardinality());
}

/// Test identifiers and opaque values count only by position: domains of
/// fresh identifiers, or of other opaque values, of one size have equal
/// signatures.
#[test]
fn domain_signature_counts_identifiers_and_opaque_values_by_position() {
    let fresh = |count: usize| {
        StepDomain::from(
            ChoiceDomain::new(
                (0..count)
                    .map(|_| Value::Identifier(Identifier::new("option")))
                    .collect(),
            )
            .expect("fresh identifiers are distinct"),
        )
    };
    let opaque = |payloads: &[i64]| {
        StepDomain::from(
            ChoiceDomain::new(
                payloads
                    .iter()
                    .map(|&payload| TestOpaque::token(payload).into_value())
                    .collect(),
            )
            .expect("distinct tokens"),
        )
    };

    assert_eq!(fresh(3).signature(), fresh(3).signature());
    assert_ne!(fresh(3).signature(), fresh(2).signature());
    assert_eq!(opaque(&[1, 2]).signature(), opaque(&[7, 8]).signature());
    assert_ne!(fresh(2).signature(), opaque(&[1, 2]).signature());
}
