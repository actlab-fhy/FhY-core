//! Tables of the SymPy backend's error texts and sources: every kind of
//! [`SympyError`], its one-line `Display`, and whether its `source()` is
//! set, and to what; every [`SympyUnavailableError`]; and the phases'
//! names.

use std::error::Error;

use fhy_core::expression::builtins::BuiltinConstant;
use fhy_core::expression::{BooleanScreen, Callee, FunctionName, NonBooleanLogicalOperandError};
use fhy_core::identifier::Identifier;
use pyo3::PyErr;
use pyo3::exceptions::PyValueError;
use rstest::rstest;

use super::test_support::{build_identifier, build_literal};
use super::{SympyError, SympyErrorKind, SympyPhase, SympyUnavailableError};

/// What an error's `source()` is expected to be.
#[derive(Debug, Clone, Copy)]
enum Source {
    None,
    Unavailable,
    IllTyped,
    Python,
}

/// Check that `error`'s `source()` is what `expected` names, by downcast.
fn assert_source(error: &dyn Error, expected: Source) {
    let source = error.source();
    let is_expected = match expected {
        Source::None => source.is_none(),
        Source::Unavailable => source.is_some_and(<dyn Error>::is::<SympyUnavailableError>),
        Source::IllTyped => source.is_some_and(<dyn Error>::is::<NonBooleanLogicalOperandError>),
        Source::Python => source.is_some_and(<dyn Error>::is::<PyErr>),
    };
    assert!(
        is_expected,
        "{error:?} has the source {source:?}, not {expected:?}"
    );
}

/// Return a Python exception, which needs no interpreter until raised.
fn python_error() -> PyErr {
    PyValueError::new_err("raised")
}

/// Return the name `name`.
fn name(name: &str) -> FunctionName {
    FunctionName::try_new(name).expect("a name")
}

#[rstest]
#[case::unavailable(
    |_: &Identifier| SympyErrorKind::Unavailable(SympyUnavailableError::MissingSympy(python_error())),
    |_: &Identifier| "the sympy backend cannot run".to_owned(),
    Source::Unavailable
)]
#[case::ill_typed(
    |_: &Identifier| SympyErrorKind::IllTyped(
        BooleanScreen::new()
            .check_logical_operands(&build_literal(1).and(build_literal(true)))
            .expect_err("a number in a boolean position"),
    ),
    |_: &Identifier| "the expression cannot be lowered to sympy: it is ill-typed".to_owned(),
    Source::IllTyped
)]
#[case::call_needs_inlining(
    |_: &Identifier| SympyErrorKind::CallNeedsInlining(Callee::Named(name("f"))),
    |_: &Identifier| r#"cannot lower a call of the expression-bodied function "f" to sympy; call inline_functions first to expand "f""#.to_owned(),
    Source::None
)]
#[case::constant_called(
    |_: &Identifier| SympyErrorKind::ConstantCalled(name("c")),
    |_: &Identifier| r#"call to "c" is to a registered native constant, which is not callable"#.to_owned(),
    Source::None
)]
#[case::no_sympy_lowering(
    |_: &Identifier| SympyErrorKind::NoSympyLowering(name("g")),
    |_: &Identifier| r#"native function "g" has no sympy lowering"#.to_owned(),
    Source::None
)]
#[case::unknown_function(
    |_: &Identifier| SympyErrorKind::UnknownFunction(name("h")),
    |_: &Identifier| r#"cannot lower call to unknown function "h" to sympy"#.to_owned(),
    Source::None
)]
#[case::constant_value_unknown(
    |c: &Identifier| SympyErrorKind::ConstantValueUnknown(c.clone()),
    |c: &Identifier| format!("the value of the native constant c::{} is unknown without a registry", c.id()),
    Source::None
)]
#[case::unsupported_constant(
    |_: &Identifier| SympyErrorKind::UnsupportedConstant(BuiltinConstant::Pi.identifier().clone()),
    |_: &Identifier| "the built-in constant pi::48 has no sympy value".to_owned(),
    Source::None
)]
#[case::bound_native_constant(
    |_: &Identifier| SympyErrorKind::BoundNativeConstant(vec![BuiltinConstant::Pi.identifier().clone(), BuiltinConstant::E.identifier().clone()]),
    |_: &Identifier| "cannot bind the native constants the expression refers to, whose values are fixed by the registry: pi::48, e::49".to_owned(),
    Source::None
)]
#[case::complex_infinity(
    |_: &Identifier| SympyErrorKind::ComplexInfinity,
    |_: &Identifier| "cannot lift zoo to an expression: sympy folds a quotient by zero to its directionless complex infinity, and no expression denotes that value".to_owned(),
    Source::None
)]
#[case::partial_piecewise(
    |_: &Identifier| SympyErrorKind::PartialPiecewise("Piecewise((x, x > 0))".to_owned()),
    |_: &Identifier| "cannot represent the partial sympy.Piecewise Piecewise((x, x > 0)) as an expression: its final branch's condition is not sympy.true, so it has no value where every condition fails".to_owned(),
    Source::None
)]
#[case::unsupported_node(
    |_: &Identifier| SympyErrorKind::UnsupportedNode("<class 'int'>".to_owned()),
    |_: &Identifier| "unsupported node type: <class 'int'>".to_owned(),
    Source::None
)]
#[case::unsupported_expression(
    |_: &Identifier| SympyErrorKind::UnsupportedExpression("Integral".to_owned()),
    |_: &Identifier| "unsupported expression type: Integral".to_owned(),
    Source::None
)]
#[case::unsupported_boolean(
    |_: &Identifier| SympyErrorKind::UnsupportedBoolean("Exclusive".to_owned()),
    |_: &Identifier| "unsupported boolean expression type: Exclusive".to_owned(),
    Source::None
)]
#[case::unsupported_relational(
    |_: &Identifier| SympyErrorKind::UnsupportedRelational("Rel".to_owned()),
    |_: &Identifier| "unsupported relational type: Rel".to_owned(),
    Source::None
)]
#[case::arity(
    |_: &Identifier| SympyErrorKind::Arity("an ITE to have exactly three arguments".to_owned()),
    |_: &Identifier| "expected an ITE to have exactly three arguments".to_owned(),
    Source::None
)]
#[case::implies(
    |_: &Identifier| SympyErrorKind::Implies("Implies(a, b)".to_owned()),
    |_: &Identifier| "implies is not supported: Implies(a, b)".to_owned(),
    Source::None
)]
#[case::unreadable_symbol(
    |_: &Identifier| SympyErrorKind::UnreadableSymbol("q".to_owned()),
    |_: &Identifier| r#"cannot read an identifier from the symbol "q": the lowering names identifiers <name_hint>_<id>"#.to_owned(),
    Source::None
)]
#[case::python(
    |_: &Identifier| SympyErrorKind::Python(python_error()),
    |_: &Identifier| "python raised an exception during simplification".to_owned(),
    Source::Python
)]
fn sympy_error_displays_one_line_and_its_source(
    #[case] build: fn(&Identifier) -> SympyErrorKind,
    #[case] text: fn(&Identifier) -> String,
    #[case] source: Source,
) {
    let (c, _) = build_identifier("c");
    let error = SympyError::new(SympyPhase::Simplification, build(&c));

    assert_eq!(error.to_string(), text(&c));
    assert_eq!(error.phase(), SympyPhase::Simplification);
    assert_source(&error, source);
}

#[rstest]
#[case::missing_sympy(
    SympyUnavailableError::MissingSympy(python_error()),
    "the sympy package cannot be imported"
)]
#[case::incompatible(
    SympyUnavailableError::Incompatible(python_error()),
    "the sympy package does not provide what the backend needs"
)]
fn unavailable_error_displays_what_is_missing_with_python_s_error(
    #[case] error: SympyUnavailableError,
    #[case] text: &str,
) {
    assert_eq!(error.to_string(), text);
    assert_source(&error, Source::Python);
}

#[rstest]
#[case::lowering(SympyPhase::Lowering, "lowering")]
#[case::simplification(SympyPhase::Simplification, "simplification")]
#[case::substitution(SympyPhase::Substitution, "substitution")]
#[case::lifting(SympyPhase::Lifting, "lifting")]
fn phase_is_named_in_words(#[case] phase: SympyPhase, #[case] text: &str) {
    assert_eq!(phase.as_str(), text);
    assert_eq!(phase.to_string(), text);
}
