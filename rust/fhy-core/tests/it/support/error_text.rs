//! What the error-text tables check of an error: its text, and the type of
//! its `source`.

use std::error::Error;

use fhy_core::constraint::ConstraintError;
use fhy_core::expression::{LiteralTextError, NonBooleanLogicalOperandError, PiecewiseError};
use fhy_core::foreign::BoxError;
use fhy_core::identifier::Identifier;
use fhy_core::solver::SolveError;

use super::constraint::TestValueError;

/// The type of an error's `source`, as the tables name it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Source {
    /// No source.
    None,
    /// A [`TestValueError`], as a test's custom part reports.
    TestValue,
    /// A [`ConstraintError`].
    Constraint,
    /// A [`LiteralTextError`].
    LiteralText,
    /// A [`NonBooleanLogicalOperandError`].
    NonBooleanOperand,
    /// A [`SolveError`].
    Solve,
    /// A [`PiecewiseError`].
    Piecewise,
}

/// Return the kind of `error`'s source.
///
/// # Panics
///
/// Panics for a source of a type the tables do not name.
pub(crate) fn source_of(error: &(dyn Error + 'static)) -> Source {
    let Some(source) = error.source() else {
        return Source::None;
    };
    if source.is::<TestValueError>() {
        Source::TestValue
    } else if source.is::<ConstraintError>() {
        Source::Constraint
    } else if source.is::<LiteralTextError>() {
        Source::LiteralText
    } else if source.is::<NonBooleanLogicalOperandError>() {
        Source::NonBooleanOperand
    } else if source.is::<SolveError>() {
        Source::Solve
    } else if source.is::<PiecewiseError>() {
        Source::Piecewise
    } else {
        panic!("a source of an unnamed type: {source:?}")
    }
}

/// Assert that `error` writes `text` and has a source of the kind `source`.
pub(crate) fn assert_error_text(error: &(dyn Error + 'static), text: &str, source: Source) {
    assert_eq!(error.to_string(), text, "{error:?}");
    assert_eq!(source_of(error), source, "{error:?}");
}

/// Return the error a test's custom part reports: `no`.
pub(crate) fn test_error() -> BoxError {
    Box::new(TestValueError("no".to_owned()))
}

/// Return the identifier `name` with the fixed id `id`, so a table's text
/// can name it.
pub(crate) fn fixed(id: u64, name: &str) -> Identifier {
    Identifier::try_restore(id, name).expect("the id is below the cap")
}
