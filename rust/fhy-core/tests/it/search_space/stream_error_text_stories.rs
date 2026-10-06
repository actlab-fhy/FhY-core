//! The text and the source of every variant of the search-stream errors:
//! `EmptyKind`, `StepDomainError`, `TraceError` and `ReplayError`.

use std::error::Error;

use fhy_core::identifier::Identifier;
use fhy_core::search_space::{
    Activity, Coordinate, EmptyKind, ReplayError, StepDomainError, TraceError,
};
use rstest::rstest;

use crate::support::error_text::test_error;
use crate::support::serde::restored;

/// Return the identifier `name` at the fixed id `id`, so a table's text can
/// name it as `name::id`.
fn build_identifier(id: u64, name: &str) -> Identifier {
    restored(63_600 + id, name)
}

/// Assert that `error` writes `text` and that its `source` writes
/// `source`, if it has one.
fn assert_error_text(error: &(dyn Error + 'static), text: &str, source: Option<&str>) {
    assert_eq!(error.to_string(), text, "{error:?}");
    assert_eq!(
        error.source().map(ToString::to_string).as_deref(),
        source,
        "{error:?}"
    );
}

/// Test an empty kind's text.
#[test]
fn empty_kind_text() {
    assert_error_text(&EmptyKind, "a decision kind needs a name", None);
}

/// Test each domain refusal's text.
#[rstest]
#[case::empty_choice(
    StepDomainError::EmptyChoice,
    "a choice domain needs at least one value"
)]
#[case::empty_order(
    StepDomainError::EmptyOrder,
    "an order domain needs at least one element"
)]
#[case::empty_runs(StepDomainError::EmptyRuns, "a strided domain needs at least one run")]
#[case::nan(StepDomainError::NanValue { index: 2 }, "the value at 2 is or holds a NaN")]
#[case::repeated(
    StepDomainError::RepeatedValue { first: 0, second: 3 },
    "the values at 0 and 3 are equal"
)]
#[case::empty_run(
    StepDomainError::EmptyRun,
    "a strided run's stop must be above its start"
)]
#[case::zero_stride(
    StepDomainError::ZeroStride,
    "a strided run's stride must be at least 1"
)]
#[case::unordered(
    StepDomainError::UnorderedRuns { index: 1 },
    "the run at 1 starts below the previous run's stop"
)]
#[case::too_large(
    StepDomainError::TooLarge,
    "the domain holds more values than coordinates number"
)]
fn step_domain_error_text(#[case] error: StepDomainError, #[case] text: &str) {
    assert_error_text(&error, text, None);
}

/// Test each run refusal's text and source.
#[rstest]
#[case::domain(
    TraceError::Domain(StepDomainError::EmptyChoice),
    "the step's domain is invalid: a choice domain needs at least one value",
    Some("a choice domain needs at least one value")
)]
#[case::unknown_decision(
    TraceError::UnknownDecision { name: build_identifier(1, "d") },
    "d::63601 is not a decision of the space",
    None
)]
#[case::already_decided(
    TraceError::AlreadyDecided { name: build_identifier(2, "d") },
    "the decision d::63602 was decided already in this run",
    None
)]
#[case::not_active(
    TraceError::NotActive { name: build_identifier(3, "d"), activity: Activity::Pending },
    "the decision d::63603 is Pending, not active",
    None
)]
#[case::not_enumerable(
    TraceError::NotEnumerable { decision: build_identifier(4, "n") },
    "the variable n::63604 has no finite domain to search",
    None
)]
#[case::no_space(
    TraceError::NoSpace,
    "a static step needs a recorder over a space",
    None
)]
#[case::out_of_domain(
    TraceError::CoordinateOutOfDomain { position: 3 },
    "the answer to step 3 names no value of its domain",
    None
)]
#[case::inadmissible(
    TraceError::Inadmissible { position: 1, coordinate: Coordinate::Index(4) },
    "the answer to step 1 is not admissible",
    None
)]
#[case::dead_end(
    TraceError::DeadEnd { decision: build_identifier(5, "k") },
    "the step for k::63605 has no admissible value",
    None
)]
#[case::oracle(
    TraceError::Oracle { position: 2, source: test_error() },
    "the oracle failed at step 2",
    Some("no")
)]
#[case::hook(
    TraceError::Hook { decision: build_identifier(6, "v"), source: test_error() },
    "the search domain of the variable v::63606 failed",
    Some("no")
)]
#[case::unasked(
    TraceError::Unasked { decisions: vec![build_identifier(7, "a"), build_identifier(8, "b")] },
    "the run never asked the assigned decisions a::63607, b::63608",
    None
)]
#[case::incomplete(
    TraceError::Incomplete,
    "the configuration to mutate is not complete",
    None
)]
#[case::other_space(TraceError::OtherSpace, "the configuration is of another space", None)]
#[case::nothing_to_mutate(
    TraceError::NothingToMutate,
    "no decision of the configuration has another admissible value",
    None
)]
#[case::attempts_exhausted(
    TraceError::AttemptsExhausted { attempts: 16 },
    "every one of the 16 attempts was refused",
    None
)]
fn trace_error_text(#[case] error: TraceError, #[case] text: &str, #[case] source: Option<&str>) {
    assert_error_text(&error, text, source);
}

/// Test each replay mismatch's text and source.
#[rstest]
#[case::exhausted(
    ReplayError::Exhausted { position: 4 },
    "the trace has no step 4 for the run to replay",
    None
)]
#[case::kind(
    ReplayError::KindMismatch { position: 0 },
    "step 0 is of another kind than the one recorded",
    None
)]
#[case::decision(
    ReplayError::DecisionMismatch { position: 1 },
    "step 1 asks another decision than the one recorded",
    None
)]
#[case::domain(
    ReplayError::DomainMismatch { position: 2 },
    "step 2 is offered another domain than the one recorded",
    None
)]
#[case::out_of_domain(
    ReplayError::CoordinateOutOfDomain { position: 3 },
    "the recorded answer to step 3 names no value of the domain offered",
    None
)]
#[case::inadmissible(
    ReplayError::Inadmissible { position: 5 },
    "the recorded answer to step 5 is not admissible in this run",
    None
)]
#[case::unconsumed(
    ReplayError::Unconsumed { position: 6 },
    "the run ended before asking the recorded step 6",
    None
)]
#[case::missing(
    ReplayError::MissingStep { decision: build_identifier(9, "x") },
    "the trace holds no step for the decision x::63609",
    None
)]
#[case::repeated(
    ReplayError::RepeatedStep { decision: build_identifier(10, "y") },
    "the trace holds two steps for the decision y::63610",
    None
)]
#[case::trace(
    ReplayError::Trace(Box::new(TraceError::NoSpace)),
    "a static step needs a recorder over a space",
    Some("a static step needs a recorder over a space")
)]
fn replay_error_text(#[case] error: ReplayError, #[case] text: &str, #[case] source: Option<&str>) {
    assert_error_text(&error, text, source);
}
