//! Tables of the solver's error texts and sources: every variant of
//! [`SolveError`] and [`LoweringError`], its one-line `Display`, and whether
//! its `source()` is set, and to what.

use std::error::Error;
use std::io;

use fhy_core::expression::builtins::BuiltinConstant;
use fhy_core::expression::{
    BooleanScreen, Expression, NonBooleanLogicalOperandError, PiecewiseError,
};
use fhy_core::identifier::Identifier;
use fhy_core::solver::{LoweringError, QueryKind, SolveError};
use rstest::rstest;

use crate::support::expression::{build_identifier, build_literal};

/// What an error's `source()` is expected to be.
#[derive(Debug, Clone, Copy)]
enum Source {
    None,
    IllTyped,
    Piecewise,
    Lowering,
    Io,
}

/// Check that `error`'s `source()` is what `expected` names, by downcast.
fn assert_source(error: &dyn Error, expected: Source) {
    let source = error.source();
    let is_expected = match expected {
        Source::None => source.is_none(),
        Source::IllTyped => source.is_some_and(<dyn Error>::is::<NonBooleanLogicalOperandError>),
        Source::Piecewise => source.is_some_and(<dyn Error>::is::<PiecewiseError>),
        Source::Lowering => source.is_some_and(<dyn Error>::is::<LoweringError>),
        Source::Io => source.is_some_and(<dyn Error>::is::<io::Error>),
    };
    assert!(
        is_expected,
        "{error:?} has the source {source:?}, not {expected:?}"
    );
}

/// Return the refusal of a Boolean position holding the number `1`.
fn ill_typed() -> NonBooleanLogicalOperandError {
    BooleanScreen::new()
        .check_logical_operands(&build_literal(1).and(build_literal(true)))
        .expect_err("a number in a boolean position")
}

/// Return two identifiers named `x` and `y`, ordered by id.
fn two_identifiers() -> (Identifier, Identifier) {
    (build_identifier("x").0, build_identifier("y").0)
}

#[rstest]
#[case::no_capable_backend(
    |_: &(Identifier, Identifier)| SolveError::NoCapableBackend(QueryKind::UniversalValidity),
    |_: &(Identifier, Identifier)| "the backends of this solver cannot answer universal validity queries".to_owned(),
    Source::None
)]
#[case::missing_symbol_types(
    |(x, y): &(Identifier, Identifier)| SolveError::MissingSymbolTypes(vec![x.clone(), y.clone()]),
    |(x, y): &(Identifier, Identifier)| format!("symbol_types is missing entries for identifiers: x::{}, y::{}", x.id(), y.id()),
    Source::None
)]
#[case::ill_typed(
    |_: &(Identifier, Identifier)| SolveError::IllTyped(ill_typed()),
    |_: &(Identifier, Identifier)| "the expression is ill-typed".to_owned(),
    Source::IllTyped
)]
#[case::bound_native_constant(
    |_: &(Identifier, Identifier)| SolveError::BoundNativeConstant(vec![BuiltinConstant::Pi.identifier().clone(), BuiltinConstant::E.identifier().clone()]),
    |_: &(Identifier, Identifier)| "the environment binds native constants the expression refers to, whose values are fixed: pi::48, e::49".to_owned(),
    Source::None
)]
#[case::substitution(
    |_: &(Identifier, Identifier)| SolveError::Substitution(PiecewiseError::NoCases),
    |_: &(Identifier, Identifier)| "substituting the environment built an invalid piecewise".to_owned(),
    Source::Piecewise
)]
#[case::lowering(
    |(x, _): &(Identifier, Identifier)| SolveError::Lowering(LoweringError::MissingSymbolTypes(vec![x.clone()])),
    |_: &(Identifier, Identifier)| "the expression has no smt-lib2 lowering".to_owned(),
    Source::Lowering
)]
#[case::backend(
    |_: &(Identifier, Identifier)| SolveError::Backend { backend: "z3".to_owned(), source: Box::new(io::Error::other("broken pipe")) },
    |_: &(Identifier, Identifier)| r#"the backend "z3" failed"#.to_owned(),
    Source::Io
)]
fn solve_error_displays_one_line_and_its_source(
    #[case] build: fn(&(Identifier, Identifier)) -> SolveError,
    #[case] text: fn(&(Identifier, Identifier)) -> String,
    #[case] source: Source,
) {
    let identifiers = two_identifiers();
    let error = build(&identifiers);

    assert_eq!(error.to_string(), text(&identifiers));
    assert_source(&error, source);
}

#[rstest]
#[case::missing_symbol_types(
    |(x, _): &(Identifier, Expression)| LoweringError::MissingSymbolTypes(vec![x.clone()]),
    |(x, _): &(Identifier, Expression)| format!("symbol_types is missing entries for identifiers: x::{}", x.id()),
    Source::None
)]
#[case::ill_typed(
    |_: &(Identifier, Expression)| LoweringError::IllTyped(ill_typed()),
    |_: &(Identifier, Expression)| "the expression is ill-typed".to_owned(),
    Source::IllTyped
)]
#[case::native_constants(
    |_: &(Identifier, Expression)| LoweringError::NativeConstants(vec![BuiltinConstant::Inf.identifier().clone()]),
    |_: &(Identifier, Expression)| "smt-lib2 has no term for the native constants inf::50".to_owned(),
    Source::None
)]
#[case::non_finite_literal(
    |_: &(Identifier, Expression)| LoweringError::NonFiniteLiteral(build_literal(f64::NEG_INFINITY)),
    |_: &(Identifier, Expression)| "smt-lib2 has no term for the non-finite float -inf".to_owned(),
    Source::None
)]
#[case::call_of_a_native_builtin(
    |(_, x): &(Identifier, Expression)| LoweringError::Call(Expression::call(fhy_core::expression::builtins::BuiltinFunction::Sin, [x.clone()])),
    |_: &(Identifier, Expression)| r#"smt-lib2 has no term for a call of the native built-in "sin""#.to_owned(),
    Source::None
)]
#[case::call_of_a_composed_builtin(
    |(_, x): &(Identifier, Expression)| LoweringError::Call(Expression::call(fhy_core::expression::builtins::BuiltinFunction::Relu, [x.clone()])),
    |_: &(Identifier, Expression)| r#"smt-lib2 has no term for a call of the built-in "relu"; inline it first with inline_functions"#.to_owned(),
    Source::None
)]
#[case::call_of_a_user_function(
    |(_, x): &(Identifier, Expression)| LoweringError::Call(Expression::call(fhy_core::expression::FunctionName::try_new("f").expect("a name"), [x.clone()])),
    |_: &(Identifier, Expression)| r#"smt-lib2 has no term for a call of "f"; a user function must be inlined first with inline_functions"#.to_owned(),
    Source::None
)]
#[case::call_of_a_non_call(
    |(_, x): &(Identifier, Expression)| LoweringError::Call(x.clone() + 1),
    |_: &(Identifier, Expression)| "smt-lib2 has no term for the call (x + 1)".to_owned(),
    Source::None
)]
#[case::sort_mismatch(
    |(_, x): &(Identifier, Expression)| LoweringError::SortMismatch(x.clone().less(build_literal(true))),
    |_: &(Identifier, Expression)| "a boolean and a number meet in the node (x < true)".to_owned(),
    Source::None
)]
#[case::unsupported_power(
    |(_, x): &(Identifier, Expression)| LoweringError::UnsupportedPower(x.clone().power(-1)),
    |_: &(Identifier, Expression)| "smt-lib2 has no term for the power (x ** -1), whose exponent is not an integer literal of at least one".to_owned(),
    Source::None
)]
fn lowering_error_displays_one_line_and_its_source(
    #[case] build: fn(&(Identifier, Expression)) -> LoweringError,
    #[case] text: fn(&(Identifier, Expression)) -> String,
    #[case] source: Source,
) {
    let x = build_identifier("x");
    let error = build(&x);

    assert_eq!(error.to_string(), text(&x));
    assert_source(&error, source);
}
