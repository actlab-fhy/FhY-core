//! The text and the source of every variant of the `search_space` errors.

use std::error::Error;

use fhy_core::constraint::{ConstraintError, Value};
use fhy_core::identifier::Identifier;
use fhy_core::param::AssignmentError;
use fhy_core::search_space::{ConfigurationError, EquivalenceError, SpaceError};
use rstest::rstest;

use crate::support::constraint::{TestValueError, int};
use crate::support::error_text::test_error;
use crate::support::search_space::{int_variable, space_of, try_configure};
use crate::support::serde::restored;

// -- helpers ----------------------------------------------------------------

/// Return the identifier `name` at the fixed id `id`, so a table's text can
/// name it as `name::id`.
fn build_identifier(id: u64, name: &str) -> Identifier {
    restored(63_400 + id, name)
}

/// Return the error a custom constraint reports: `a custom constraint
/// failed`.
fn build_constraint_error() -> ConstraintError {
    ConstraintError::Custom(test_error())
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

// -- SpaceError -------------------------------------------------------------

#[rstest]
#[case::duplicate_name(
    SpaceError::DuplicateName { name: build_identifier(1, "x") },
    "the name x::63401 is used more than once",
    None
)]
#[case::empty_choice(
    SpaceError::EmptyChoice { choice: build_identifier(2, "c") },
    "the choice c::63402 has no alternative",
    None
)]
#[case::unknown_condition_target(
    SpaceError::UnknownConditionTarget { target: build_identifier(3, "t") },
    "the condition's target t::63403 is not a decision of the space",
    None
)]
#[case::unknown_reference(
    SpaceError::UnknownReference { name: build_identifier(4, "r") },
    "r::63404 is not a decision of the space",
    None
)]
#[case::equation_over_choice(
    SpaceError::EquationOverChoice { choice: build_identifier(5, "c") },
    "an equation names the choice c::63405, which conditions and forbidden clauses name only \
     in set constraints",
    None
)]
#[case::unknown_alternative(
    SpaceError::UnknownAlternative { choice: build_identifier(5, "c"), value: int(3) },
    "the choice c::63405 has no alternative 3",
    None
)]
#[case::condition_references_subtree(
    SpaceError::ConditionReferencesSubtree { target: build_identifier(6, "t"), name: build_identifier(7, "u") },
    "the condition on t::63406 names u::63407, which is the target or under it",
    None
)]
#[case::choice_too_deep(
    SpaceError::ChoiceTooDeep { choice: build_identifier(5, "c") },
    "the choice c::63405 nests choices more than 16 levels deep",
    None
)]
#[case::empty_condition(
    SpaceError::EmptyCondition { target: build_identifier(6, "t") },
    "the condition on t::63406 names no decision",
    None
)]
#[case::empty_forbidden(
    SpaceError::EmptyForbidden { index: 2 },
    "the forbidden clause 2 names no decision",
    None
)]
#[case::cyclic_dependency_of_one(
    SpaceError::CyclicDependency { cycle: vec![build_identifier(8, "a")] },
    "the decisions a::63408 depend on each other in a cycle",
    None
)]
#[case::cyclic_dependency_of_three(
    SpaceError::CyclicDependency { cycle: vec![build_identifier(8, "a"), build_identifier(9, "b"), build_identifier(10, "c")] },
    "the decisions a::63408, b::63409, c::63410 depend on each other in a cycle",
    None
)]
#[case::hook(
    SpaceError::Hook { alternative: build_identifier(11, "alt"), source: test_error() },
    "the bound identifiers of the alternative alt::63411 failed",
    Some("no")
)]
#[case::constraint(
    SpaceError::Constraint(build_constraint_error()),
    "a custom constraint failed",
    Some("a custom constraint failed")
)]
fn space_error_text(#[case] error: SpaceError, #[case] text: &str, #[case] source: Option<&str>) {
    assert_error_text(&error, text, source);
}

#[test]
fn space_error_hook_source_is_the_implementations_error() {
    let error = SpaceError::Hook {
        alternative: build_identifier(11, "alt"),
        source: test_error(),
    };

    let source = error.source().expect("a hook error has a source");

    assert_eq!(
        source.downcast_ref::<TestValueError>(),
        Some(&TestValueError("no".to_owned()))
    );
}

#[test]
fn space_error_constraint_source_is_the_constraint_error() {
    let error = SpaceError::Constraint(build_constraint_error());

    let source = error.source().expect("a constraint error has a source");

    assert!(
        matches!(
            source.downcast_ref::<ConstraintError>(),
            Some(ConstraintError::Custom(_))
        ),
        "{source:?}"
    );
}

// -- ConfigurationError -----------------------------------------------------

#[rstest]
#[case::unknown_decision(
    ConfigurationError::UnknownDecision { name: build_identifier(20, "x") },
    "x::63420 is not a decision of the space",
    None
)]
#[case::duplicate_entry(
    ConfigurationError::DuplicateEntry { name: build_identifier(21, "x") },
    "the decision x::63421 is given more than one value",
    None
)]
#[case::inactive_decision(
    ConfigurationError::InactiveDecision { name: build_identifier(22, "x") },
    "the decision x::63422 is given a value but is not active",
    None
)]
#[case::unknown_alternative(
    ConfigurationError::UnknownAlternative {
        choice: build_identifier(23, "c"),
        value: Value::Identifier(build_identifier(24, "a")),
    },
    "the choice c::63423 has no alternative a",
    None
)]
#[case::unknown_alternative_of_a_number(
    ConfigurationError::UnknownAlternative {
        choice: build_identifier(23, "c"),
        value: int(3),
    },
    "the choice c::63423 has no alternative 3",
    None
)]
#[case::assignment(
    ConfigurationError::Assignment {
        variable: build_identifier(25, "v"),
        error: AssignmentError::Inadmissible,
    },
    "the value of the variable v::63425 cannot be assigned to its param",
    Some("the value is not admissible")
)]
#[case::forbidden(
    ConfigurationError::Forbidden { index: 1 },
    "the configuration takes the forbidden combination 1",
    None
)]
#[case::undecided_condition(
    ConfigurationError::UndecidedCondition { target: build_identifier(26, "t") },
    "the condition on t::63426 could not be decided",
    None
)]
#[case::undecided_forbidden(
    ConfigurationError::UndecidedForbidden { index: 3 },
    "the forbidden clause 3 could not be decided",
    None
)]
#[case::failed_condition(
    ConfigurationError::FailedCondition { target: build_identifier(27, "t"), error: build_constraint_error() },
    "the condition on t::63427 failed to evaluate",
    Some("a custom constraint failed")
)]
#[case::failed_forbidden(
    ConfigurationError::FailedForbidden { index: 4, error: build_constraint_error() },
    "the forbidden clause 4 failed to evaluate",
    Some("a custom constraint failed")
)]
fn configuration_error_text(
    #[case] error: ConfigurationError,
    #[case] text: &str,
    #[case] source: Option<&str>,
) {
    assert_error_text(&error, text, source);
}

#[test]
fn configuration_error_assignment_source_is_the_assignment_error() {
    let error = ConfigurationError::Assignment {
        variable: build_identifier(25, "v"),
        error: AssignmentError::UnverifiedConstraint { member: 0 },
    };

    let source = error.source().expect("an assignment problem has a source");

    assert!(
        matches!(
            source.downcast_ref::<AssignmentError>(),
            Some(AssignmentError::UnverifiedConstraint { member: 0 })
        ),
        "{source:?}"
    );
}

#[rstest]
#[case::failed_condition(ConfigurationError::FailedCondition {
    target: build_identifier(27, "t"),
    error: build_constraint_error(),
})]
#[case::failed_forbidden(ConfigurationError::FailedForbidden {
    index: 4,
    error: build_constraint_error(),
})]
fn configuration_error_failed_evaluation_source_is_the_constraint_error(
    #[case] error: ConfigurationError,
) {
    let source = error.source().expect("a failed evaluation has a source");

    assert!(
        matches!(
            source.downcast_ref::<ConstraintError>(),
            Some(ConstraintError::Custom(_))
        ),
        "{source:?}"
    );
}

// -- ConfigurationErrors ----------------------------------------------------

#[test]
fn configuration_errors_display_every_problem_in_the_order_found() {
    let known = build_identifier(30, "a");
    let unknown = build_identifier(31, "ghost");
    let space = space_of(
        &build_identifier(32, "space"),
        vec![int_variable(&known, &[1, 2])],
        Vec::new(),
    );

    let Err(errors) = try_configure(
        &space,
        [
            (unknown, int(1)),
            (known, int(1)),
            (build_identifier(30, "a"), int(2)),
        ],
    ) else {
        panic!("an unknown decision and a duplicate entry are refused");
    };

    assert_eq!(
        errors.to_string(),
        "the configuration is invalid: ghost::63431 is not a decision of the space; the \
         decision a::63430 is given more than one value"
    );
    let [
        ConfigurationError::UnknownDecision { name: first },
        ConfigurationError::DuplicateEntry { name: second },
    ] = errors.errors()
    else {
        panic!("two problems in the order found: {errors:?}");
    };
    assert_eq!(
        (first, second),
        (&build_identifier(31, "ghost"), &build_identifier(30, "a"))
    );
}

#[test]
fn configuration_errors_display_a_single_problem_without_a_separator() {
    let space = space_of(
        &build_identifier(32, "space"),
        vec![int_variable(&build_identifier(30, "a"), &[1, 2])],
        Vec::new(),
    );

    let Err(errors) = try_configure(&space, [(build_identifier(31, "ghost"), int(1))]) else {
        panic!("an unknown decision is refused");
    };

    assert_eq!(
        errors.to_string(),
        "the configuration is invalid: ghost::63431 is not a decision of the space"
    );
    assert!(errors.source().is_none(), "{errors:?}");
}

// -- EquivalenceError -------------------------------------------------------

#[rstest]
#[case::constraint(
    EquivalenceError::Constraint(build_constraint_error()),
    "a custom constraint failed during the comparison",
    Some("a custom constraint failed")
)]
#[case::extension(
    EquivalenceError::Extension(test_error()),
    "an implementation's hook failed during the comparison",
    Some("no")
)]
fn equivalence_error_text(
    #[case] error: EquivalenceError,
    #[case] text: &str,
    #[case] source: Option<&str>,
) {
    assert_error_text(&error, text, source);
}

#[test]
fn equivalence_error_constraint_source_is_the_constraint_error() {
    let error = EquivalenceError::Constraint(build_constraint_error());

    let source = error.source().expect("a constraint error has a source");

    assert!(
        matches!(
            source.downcast_ref::<ConstraintError>(),
            Some(ConstraintError::Custom(_))
        ),
        "{source:?}"
    );
}

#[test]
fn equivalence_error_extension_source_is_the_hooks_error() {
    let error = EquivalenceError::Extension(test_error());

    let source = error.source().expect("an extension error has a source");

    assert_eq!(
        source.downcast_ref::<TestValueError>(),
        Some(&TestValueError("no".to_owned()))
    );
}
