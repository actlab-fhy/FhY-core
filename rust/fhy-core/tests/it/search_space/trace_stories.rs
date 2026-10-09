//! Tests for `TraceStep` and `Trace`: building a dynamic step, the
//! accessors, the traversed cardinality, filtering by kind, the bare
//! coordinates, `Display`, and `==` and `Hash`.

use fhy_core::identifier::Identifier;
use fhy_core::search_space::{Coordinate, StepDomain, Trace, TraceError, TraceStep};
use num_bigint::BigUint;
use rstest::rstest;

use crate::support::constraint::int;
use crate::support::hashing::hash_of;
use crate::support::search::{index, int_choices, kind, order, order_of, strided};

/// Return the dynamic step of `kind_name` over a choice of `cardinality`
/// integers, answered with `answer`.
fn build_step(kind_name: &str, cardinality: i64, answer: u64) -> TraceStep {
    let values: Vec<i64> = (0..cardinality).collect();
    TraceStep::dynamic(
        kind(kind_name),
        Identifier::new("s"),
        &int_choices(&values),
        index(answer),
    )
    .expect("the answer is in the domain")
}

/// Test a dynamic step keeps its kind, subject, signature, answer and the
/// value the answer names, and has no decision.
#[test]
fn trace_step_dynamic_keeps_what_was_asked_and_answered() {
    let subject = Identifier::new("operand");
    let domain = int_choices(&[10, 20, 30]);

    let step = TraceStep::dynamic(kind("moga.cir.option"), subject.clone(), &domain, index(2))
        .expect("in the domain");

    assert_eq!(step.kind().as_str(), "moga.cir.option");
    assert_eq!(step.subject(), &subject);
    assert_eq!(step.decision(), None);
    assert_eq!(step.signature(), &domain.signature());
    assert_eq!(step.cardinality(), BigUint::from(3_u8));
    assert_eq!(step.coordinate(), &index(2));
    assert_eq!(step.value(), Some(&int(30)));
}

/// Test a dynamic step over an order domain keeps the ordering it names.
#[test]
fn trace_step_dynamic_over_an_order_keeps_the_ordering() {
    let [i, j] = ["i", "j"].map(Identifier::new);
    let domain = order_of(&[&i, &j]);

    let step = TraceStep::dynamic(
        kind("moga.cir.walk_order"),
        i.clone(),
        &domain,
        order(&[1, 0]),
    )
    .expect("a permutation");

    assert_eq!(step.value(), domain.value_at(&order(&[1, 0])).as_ref());
    assert_eq!(step.cardinality(), BigUint::from(2_u8));
}

/// Test a dynamic step refuses an answer its domain does not contain.
#[rstest]
#[case::past_the_end(int_choices(&[1, 2]), index(2))]
#[case::wrong_shape(int_choices(&[1, 2]), order(&[0, 1]))]
#[case::not_a_permutation(order_of(&[&Identifier::new("i"), &Identifier::new("j")]), order(&[0, 0]))]
fn trace_step_dynamic_refuses_an_answer_outside_its_domain(
    #[case] domain: StepDomain,
    #[case] coordinate: Coordinate,
) {
    let result = TraceStep::dynamic(kind("k"), Identifier::new("s"), &domain, coordinate);

    assert!(
        matches!(
            result,
            Err(TraceError::CoordinateOutOfDomain { position: 0 })
        ),
        "{result:?}"
    );
}

/// Test the traversed cardinality multiplies the steps' cardinalities.
#[test]
fn trace_traversed_cardinality_multiplies_the_steps() {
    let trace = Trace::new(vec![
        build_step("option", 3, 0),
        build_step("address", 4, 1),
        build_step("address", 5, 2),
    ]);

    let cardinality = trace.traversed_cardinality();

    assert_eq!(trace.len(), 3);
    assert_eq!(cardinality, BigUint::from(60_u8));
}

/// Test an empty trace counts one path, has no step, and displays as zero
/// steps.
#[test]
fn empty_trace_counts_one_and_displays_zero_steps() {
    let trace = Trace::default();

    assert!(trace.is_empty());
    assert_eq!(trace.len(), 0);
    assert_eq!(trace.traversed_cardinality(), BigUint::from(1_u8));
    assert_eq!(trace.to_string(), "0 steps");
    assert_eq!(trace, Trace::new(Vec::new()));
}

/// Test the traversed cardinality of an order step is the factorial.
#[test]
fn trace_traversed_cardinality_counts_an_order_as_its_factorial() {
    let elements = ["a", "b", "c", "d"].map(Identifier::new);
    let domain = order_of(&elements.iter().collect::<Vec<_>>());
    let step = TraceStep::dynamic(
        kind("walk"),
        elements[0].clone(),
        &domain,
        order(&[3, 2, 1, 0]),
    )
    .expect("a permutation");

    let trace = Trace::new(vec![step, build_step("tile", 2, 1)]);

    assert_eq!(trace.traversed_cardinality(), BigUint::from(48_u8));
}

/// Test `of_kind` keeps ask order and drops other kinds.
#[test]
fn trace_of_kind_keeps_ask_order() {
    let first_address = build_step("address", 4, 3);
    let option = build_step("option", 3, 1);
    let second_address = build_step("address", 5, 0);
    let trace = Trace::new(vec![
        first_address.clone(),
        option.clone(),
        second_address.clone(),
    ]);

    let addresses: Vec<&TraceStep> = trace.of_kind(&kind("address")).collect();
    let options: Vec<&TraceStep> = trace.of_kind(&kind("option")).collect();
    let tiles: Vec<&TraceStep> = trace.of_kind(&kind("tile")).collect();

    assert_eq!(addresses, [&first_address, &second_address]);
    assert_eq!(options, [&option]);
    assert_eq!(tiles, Vec::<&TraceStep>::new());
}

/// Test the coordinates are the bare answers, in ask order.
#[test]
fn trace_coordinates_are_the_bare_vector() {
    let trace = Trace::new(vec![
        build_step("option", 3, 2),
        build_step("address", 4, 0),
    ]);

    let coordinates: Vec<&Coordinate> = trace.coordinates().collect();

    assert_eq!(coordinates, [&index(2), &index(0)]);
    assert_eq!(trace.coordinates().len(), 2);
}

/// Test a trace displays its step count and each kind's, in the order the
/// kinds first occur.
#[test]
fn trace_displays_its_counts_by_kind_in_first_seen_order() {
    let trace = Trace::new(vec![
        build_step("search_space.choice", 2, 0),
        build_step("moga.cir.address", 4, 1),
        build_step("search_space.choice", 3, 2),
        build_step("moga.cir.address", 4, 3),
        build_step("moga.cir.address", 4, 0),
    ]);

    let text = trace.to_string();

    assert_eq!(text, "5 steps (2 search_space.choice, 3 moga.cir.address)");
}

/// Test traces of equal steps are equal and hash alike, and a different
/// answer makes them unequal.
#[test]
fn trace_equality_compares_the_steps() {
    let subject = Identifier::new("s");
    let domain = int_choices(&[1, 2, 3]);
    let step = |answer| {
        TraceStep::dynamic(kind("k"), subject.clone(), &domain, index(answer)).expect("in range")
    };

    let left = Trace::new(vec![step(0), step(2)]);
    let right = Trace::new(vec![step(0), step(2)]);
    let other = Trace::new(vec![step(0), step(1)]);

    assert_eq!(left, right);
    assert_eq!(hash_of(&left), hash_of(&right));
    assert_ne!(left, other);
}

/// Test steps over domains of one size but other plain values differ: the
/// signature is part of a step.
#[test]
fn trace_steps_over_other_plain_domains_differ() {
    let subject = Identifier::new("s");

    let low = TraceStep::dynamic(kind("k"), subject.clone(), &strided(&[(0, 64)]), index(5))
        .expect("in range");
    let high =
        TraceStep::dynamic(kind("k"), subject, &strided(&[(64, 128)]), index(5)).expect("in range");

    assert_ne!(low, high);
    assert_eq!(low.value(), Some(&int(5)));
    assert_eq!(high.value(), Some(&int(69)));
}
