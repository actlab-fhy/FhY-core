//! The text and the source of every variant of the `param` errors.

use fhy_core::constraint::ConstraintError;
use fhy_core::param::{
    AssignmentError, BoundSide, DomainError, DomainKind, IntervalError, ParamBuildError,
    ParamError, SetOperation,
};
use rstest::rstest;

use crate::support::error_text::{Source, assert_error_text, fixed, test_error};
use crate::support::param::at_least;

fn custom_constraint_error() -> ConstraintError {
    ConstraintError::Custom(test_error())
}

#[rstest]
#[case::empty(
    DomainError::EmptyValues(DomainKind::Ordinal),
    "the values of an ordinal domain must be non-empty",
    Source::None
)]
#[case::not_a_leaf(
    DomainError::NotALeafValue { kind: DomainKind::Categorical, index: 2 },
    "value 2 is no value a categorical domain admits",
    Source::None
)]
#[case::nan(
    DomainError::NanValue(DomainKind::Permutation),
    "the values of a permutation domain must not include NaN: NaN is unequal to itself, so a \
     NaN value could never be admitted",
    Source::None
)]
#[case::incomparable(
    DomainError::IncomparableValues,
    "ordinal values must be mutually comparable for sorting",
    Source::None
)]
#[case::duplicate(
    DomainError::DuplicateValues(DomainKind::Categorical),
    "the values of a categorical domain must be unique",
    Source::None
)]
#[case::custom(
    DomainError::Custom(test_error()),
    "an opaque value failed",
    Source::TestValue
)]
fn domain_error_text(#[case] error: DomainError, #[case] text: &str, #[case] source: Source) {
    assert_error_text(&error, text, source);
}

#[rstest]
#[case::interval_kind(
    ParamBuildError::ForbiddenConstraintKind(DomainKind::IntervalInteger),
    "interval integer parameters only support equation constraints".to_owned(),
    Source::None
)]
#[case::finite_kind(
    ParamBuildError::ForbiddenConstraintKind(DomainKind::Ordinal),
    "only in-set and not-in-set constraints are allowed for ordinal parameters".to_owned(),
    Source::None
)]
#[case::not_a_bound(
    ParamBuildError::NotABound,
    "interval integer parameters only support bound expressions of the form \"x >= k\", \
     \"x > k\", \"x <= k\", or \"x < k\" where k is an integer"
        .to_owned(),
    Source::None
)]
#[case::native_constant(
    ParamBuildError::NativeConstantVariable(fixed(7, "pi")),
    "the variable pi::7 is a native constant's canonical identifier, which names a value rather \
     than a variable"
        .to_owned(),
    Source::None
)]
#[case::out_of_scope(
    ParamBuildError::OutOfScope {
        constraint: at_least(&fixed(8, "y"), 0),
        variable: fixed(7, "x"),
    },
    "a constraint's scope must include the param's variable x::7".to_owned(),
    Source::None
)]
#[case::natural_bound(
    ParamBuildError::NaturalBound {
        side: BoundSide::Lower,
        zero_included: true,
        is_inclusive: true,
        is_negative: true,
    },
    "lower bound must be non-negative".to_owned(),
    Source::None
)]
#[case::unordered(
    ParamBuildError::UnorderedBounds,
    "lower bound must be less than or equal to upper bound".to_owned(),
    Source::None
)]
#[case::empty_interval(
    ParamBuildError::EmptyInterval(fixed(7, "x")),
    "empty integer interval represented by the constraints of x::7".to_owned(),
    Source::None
)]
#[case::constraint(
    ParamBuildError::Constraint(custom_constraint_error()),
    "a constraint failed".to_owned(),
    Source::Constraint
)]
#[case::custom(
    ParamBuildError::Custom(test_error()),
    "a custom domain failed".to_owned(),
    Source::TestValue
)]
fn build_error_text(#[case] error: ParamBuildError, #[case] text: String, #[case] source: Source) {
    assert_error_text(&error, &text, source);
}

#[rstest]
#[case::lower_zero_negative(BoundSide::Lower, true, true, true, "lower bound must be non-negative")]
#[case::lower_zero_exclusive(
    BoundSide::Lower,
    true,
    false,
    false,
    "lower bound must be at least 1 if zero is included and bound is exclusive"
)]
#[case::lower_positive_inclusive(
    BoundSide::Lower,
    false,
    true,
    false,
    "lower bound must be at least 1 when zero is not included"
)]
#[case::lower_positive_exclusive(
    BoundSide::Lower,
    false,
    false,
    true,
    "lower bound must be non-negative when zero is not included and bound is exclusive"
)]
#[case::upper_zero_inclusive(
    BoundSide::Upper,
    true,
    true,
    true,
    "upper bound must be non-negative when zero is included"
)]
#[case::upper_zero_exclusive(
    BoundSide::Upper,
    true,
    false,
    false,
    "upper bound must be at least 1 if zero is included and bound is exclusive"
)]
#[case::upper_positive_inclusive(
    BoundSide::Upper,
    false,
    true,
    false,
    "upper bound must be at least 1 when zero is not included"
)]
#[case::upper_positive_exclusive(
    BoundSide::Upper,
    false,
    false,
    false,
    "upper bound must be at least 2 when zero is not included and bound is exclusive"
)]
fn natural_bound_text_covers_every_combination(
    #[case] side: BoundSide,
    #[case] zero_included: bool,
    #[case] is_inclusive: bool,
    #[case] is_negative: bool,
    #[case] text: &str,
) {
    let error = ParamBuildError::NaturalBound {
        side,
        zero_included,
        is_inclusive,
        is_negative,
    };

    assert_error_text(&error, text, Source::None);
}

#[test]
fn a_zero_included_lower_bound_is_refused_as_negative_only_when_it_is() {
    let refused = |is_negative| {
        ParamBuildError::NaturalBound {
            side: BoundSide::Lower,
            zero_included: true,
            is_inclusive: false,
            is_negative,
        }
        .to_string()
    };

    assert_eq!(refused(true), "lower bound must be non-negative");
    assert_eq!(
        refused(false),
        "lower bound must be at least 1 if zero is included and bound is exclusive"
    );
}

#[rstest]
#[case::inadmissible(
    AssignmentError::Inadmissible,
    "the value is not admissible".to_owned(),
    Source::None
)]
#[case::violated(
    AssignmentError::ViolatedConstraint { member: 2 },
    "the value violates the param's constraint 2".to_owned(),
    Source::None
)]
#[case::unverified(
    AssignmentError::UnverifiedConstraint { member: 1 },
    "the value could not be verified against the param's constraint 1".to_owned(),
    Source::None
)]
#[case::binds_variable(
    AssignmentError::BindingsBindVariable(fixed(7, "x")),
    "the bindings must not bind the param's own variable x::7, whose value is the value checked"
        .to_owned(),
    Source::None
)]
#[case::constraint(
    AssignmentError::Constraint(custom_constraint_error()),
    "a constraint failed".to_owned(),
    Source::Constraint
)]
#[case::custom(
    AssignmentError::Custom(test_error()),
    "a custom domain failed".to_owned(),
    Source::TestValue
)]
fn assignment_error_text(
    #[case] error: AssignmentError,
    #[case] text: String,
    #[case] source: Source,
) {
    assert_error_text(&error, &text, source);
}

#[rstest]
#[case::not_an_operand(
    IntervalError::NotAnIntervalOperand,
    "arithmetic is only supported on interval-integer parameters",
    Source::None
)]
#[case::unsupported(
    IntervalError::UnsupportedOperand,
    "unsupported operand of interval arithmetic",
    Source::None
)]
#[case::non_bound(
    IntervalError::NonBoundOperand(None),
    "cannot coerce an integer parameter with non-bound constraints to an interval parameter",
    Source::None
)]
#[case::non_bound_failing(
    IntervalError::NonBoundOperand(Some(custom_constraint_error())),
    "cannot coerce an integer parameter with non-bound constraints to an interval parameter",
    Source::Constraint
)]
#[case::malformed(
    IntervalError::MalformedBound,
    "an interval parameter holds a constraint that is not a bound",
    Source::None
)]
#[case::build(
    IntervalError::Build(ParamBuildError::UnorderedBounds),
    "lower bound must be less than or equal to upper bound",
    Source::None
)]
#[case::build_failing(
    IntervalError::Build(ParamBuildError::Custom(test_error())),
    "a custom domain failed",
    Source::TestValue
)]
#[case::custom(
    IntervalError::Custom(test_error()),
    "a custom domain failed",
    Source::TestValue
)]
fn interval_error_text(#[case] error: IntervalError, #[case] text: &str, #[case] source: Source) {
    assert_error_text(&error, text, source);
}

#[rstest]
#[case::kind_mismatch(
    ParamError::KindMismatch {
        operation: SetOperation::Union,
        own: DomainKind::Integer,
        other: DomainKind::Real,
    },
    "cannot union an integer domain with a real domain".to_owned(),
    Source::None
)]
#[case::intersect_mismatch(
    ParamError::KindMismatch {
        operation: SetOperation::Intersection,
        own: DomainKind::Categorical,
        other: DomainKind::IntervalInteger,
    },
    "cannot intersect a categorical domain with an interval integer domain".to_owned(),
    Source::None
)]
#[case::empty_union(
    ParamError::EmptyUnion(DomainKind::Categorical),
    "the union of the categorical value sets is empty".to_owned(),
    Source::None
)]
#[case::empty_intersection(
    ParamError::EmptyIntersection(DomainKind::Ordinal),
    "the intersection of the ordinal value sets is empty".to_owned(),
    Source::None
)]
#[case::permutation_members(
    ParamError::DifferentPermutationMembers,
    "the intersection of permutation domains with different member sets is empty".to_owned(),
    Source::None
)]
#[case::unsupported_union(
    ParamError::UnsupportedUnion(DomainKind::Real),
    "union is not supported for a real domain".to_owned(),
    Source::None
)]
#[case::empty_params(
    ParamError::EmptyParamIntersection,
    "the intersection of the parameters is empty".to_owned(),
    Source::None
)]
#[case::rescope(
    ParamError::Rescope { from: fixed(7, "x"), to: fixed(9, "z"), variable: fixed(8, "y") },
    "cannot rescope a set constraint from x::7 to z::9: it is scoped to y::8, not x::7".to_owned(),
    Source::None
)]
#[case::unexpected_kind(
    ParamError::UnexpectedConstraintKind,
    "cannot rescope a constraint of an unexpected kind".to_owned(),
    Source::None
)]
#[case::domain(
    ParamError::Domain(DomainError::IncomparableValues),
    "ordinal values must be mutually comparable for sorting".to_owned(),
    Source::None
)]
#[case::domain_failing(
    ParamError::Domain(DomainError::Custom(test_error())),
    "an opaque value failed".to_owned(),
    Source::TestValue
)]
#[case::build(
    ParamError::Build(ParamBuildError::Constraint(custom_constraint_error())),
    "a constraint failed".to_owned(),
    Source::Constraint
)]
#[case::interval(
    ParamError::Interval(IntervalError::MalformedBound),
    "an interval parameter holds a constraint that is not a bound".to_owned(),
    Source::None
)]
#[case::interval_failing(
    ParamError::Interval(IntervalError::Custom(test_error())),
    "a custom domain failed".to_owned(),
    Source::TestValue
)]
#[case::constraint(
    ParamError::Constraint(custom_constraint_error()),
    "a constraint failed".to_owned(),
    Source::Constraint
)]
#[case::custom(
    ParamError::Custom(test_error()),
    "a custom domain failed".to_owned(),
    Source::TestValue
)]
fn param_error_text(#[case] error: ParamError, #[case] text: String, #[case] source: Source) {
    assert_error_text(&error, &text, source);
}

#[test]
fn the_wrapped_constraint_is_the_source_itself() {
    let error =
        ParamBuildError::Constraint(ConstraintError::MissingSymbolTypes(vec![fixed(7, "x")]));

    let source = std::error::Error::source(&error)
        .and_then(|source| source.downcast_ref::<ConstraintError>())
        .expect("the constraint error");

    assert!(
        matches!(source, ConstraintError::MissingSymbolTypes(identifiers)
            if *identifiers == vec![fixed(7, "x")])
    );
}
