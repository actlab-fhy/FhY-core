//! The Python exceptions of the param core's errors.
//!
//! An error with a core counterpart raises the class the Python
//! implementation raised, with the core's text; one whose Python text names
//! Python values, such as a class, keeps Python's words.

use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;

use fhy_core::constraint::ConstraintError;
use fhy_core::foreign::BoxError;
use fhy_core::param::{
    AssignmentError, DomainError, DomainKind, IntervalError, ParamBuildError, ParamError,
    SetOperation,
};

use crate::constraint::{constraint_error, constraint_error_to_py, type_name};
use crate::error::IntoPyErr;

use super::value::value_kind_message;

/// Return the `ParamError` with `message`.
pub(super) fn param_error(py: Python<'_>, message: impl Into<String>) -> PyErr {
    crate::exceptions::PARAM_ERROR.err(py, (message.into(),))
}

/// Return the name of the Python class of a domain of `kind`, after its
/// indefinite article.
const fn class_with_article(kind: DomainKind) -> &'static str {
    match kind {
        DomainKind::Integer => "an IntegerDomain",
        DomainKind::IntervalInteger => "an IntervalIntegerDomain",
        DomainKind::Real => "a RealDomain",
        DomainKind::Ordinal => "an OrdinalDomain",
        DomainKind::Categorical => "a CategoricalDomain",
        DomainKind::Permutation => "a PermutationDomain",
        _ => "a ParamDomain",
    }
}

/// Any error of the param core: the families of `fhy_core::param`, so one
/// mapping raises each variant as the Python implementation did.
#[derive(Debug)]
pub(crate) enum ParamFailure {
    /// A finite domain that cannot be built.
    Domain(DomainError),
    /// A param that cannot be built.
    Build(ParamBuildError),
    /// A value that cannot be assigned.
    Assignment(AssignmentError),
    /// Interval arithmetic that cannot be done.
    Interval(IntervalError),
    /// A question that cannot be decided.
    Question(ParamError),
    /// A constraint that failed to evaluate.
    Constraint(ConstraintError),
}

impl From<DomainError> for ParamFailure {
    fn from(error: DomainError) -> Self {
        Self::Domain(error)
    }
}

impl From<ParamBuildError> for ParamFailure {
    fn from(error: ParamBuildError) -> Self {
        Self::Build(error)
    }
}

impl From<AssignmentError> for ParamFailure {
    fn from(error: AssignmentError) -> Self {
        Self::Assignment(error)
    }
}

impl From<IntervalError> for ParamFailure {
    fn from(error: IntervalError) -> Self {
        Self::Interval(error)
    }
}

impl From<ParamError> for ParamFailure {
    fn from(error: ParamError) -> Self {
        match error {
            ParamError::Domain(error) => Self::Domain(error),
            ParamError::Build(error) => Self::Build(error),
            ParamError::Interval(error) => Self::Interval(error),
            other => Self::Question(other),
        }
    }
}

impl From<ConstraintError> for ParamFailure {
    fn from(error: ConstraintError) -> Self {
        Self::Constraint(error)
    }
}

/// Return the Python exception a custom domain or value raised, boxed as
/// `error`, or a `RuntimeError` naming it and `text`.
fn custom_error_to_py(text: &str, error: BoxError) -> PyErr {
    crate::exceptions::unbox_py_err(error)
        .unwrap_or_else(|error| PyRuntimeError::new_err(format!("{text}: {error}")))
}

/// Return the exception of `error`, where `other` is the other domain of a
/// set operation, if any.
pub(crate) fn param_error_to_py(
    py: Python<'_>,
    error: impl Into<ParamFailure>,
    other: Option<&Bound<'_, PyAny>>,
) -> PyErr {
    let error = error.into();
    let text = match &error {
        ParamFailure::Domain(error) => error.to_string(),
        ParamFailure::Build(error) => error.to_string(),
        ParamFailure::Assignment(error) => error.to_string(),
        ParamFailure::Interval(error) => error.to_string(),
        ParamFailure::Question(error) => error.to_string(),
        ParamFailure::Constraint(error) => error.to_string(),
    };
    match error {
        ParamFailure::Domain(error) => match error {
            DomainError::NotALeafValue { kind, .. } => {
                PyTypeError::new_err(value_kind_message(kind))
            }
            DomainError::IncomparableValues => PyTypeError::new_err(text),
            DomainError::Custom(error) => custom_error_to_py(&text, error),
            _ => param_error(py, text),
        },
        ParamFailure::Build(error) => match error {
            ParamBuildError::ForbiddenConstraintKind(DomainKind::IntervalInteger) => {
                PyTypeError::new_err(text)
            }
            ParamBuildError::ForbiddenConstraintKind(_)
            | ParamBuildError::NotABound
            | ParamBuildError::EmptyInterval(_)
            | ParamBuildError::NaturalBound { .. }
            | ParamBuildError::UnorderedBounds => param_error(py, text),
            ParamBuildError::Constraint(error) => constraint_error_to_py(py, error, None),
            ParamBuildError::Custom(error) => custom_error_to_py(&text, error),
            _ => PyRuntimeError::new_err(text),
        },
        ParamFailure::Assignment(error) => match error {
            AssignmentError::Constraint(error) => constraint_error_to_py(py, error, None),
            AssignmentError::Custom(error) => custom_error_to_py(&text, error),
            _ => PyRuntimeError::new_err(text),
        },
        ParamFailure::Interval(error) => match error {
            IntervalError::NotAnIntervalOperand => PyTypeError::new_err(text),
            IntervalError::Build(error) => param_error_to_py(py, error, other),
            IntervalError::Custom(error) => custom_error_to_py(&text, error),
            _ => PyRuntimeError::new_err(text),
        },
        ParamFailure::Question(error) => match error {
            ParamError::KindMismatch { operation, own, .. } => {
                let verb = match operation {
                    SetOperation::Union => "union",
                    _ => "intersect",
                };
                let other = other.map_or_else(|| "?".to_owned(), type_name);
                PyTypeError::new_err(format!(
                    "Cannot {verb} {} with a domain of type {other}.",
                    class_with_article(own)
                ))
            }
            ParamError::Rescope { .. } | ParamError::UnexpectedConstraintKind => {
                constraint_error(py, text)
            }
            ParamError::Constraint(error) => constraint_error_to_py(py, error, None),
            ParamError::Custom(error) => custom_error_to_py(&text, error),
            ParamError::EmptyUnion(_)
            | ParamError::EmptyIntersection(_)
            | ParamError::DifferentPermutationMembers
            | ParamError::EmptyParamIntersection => param_error(py, text),
            _ => PyRuntimeError::new_err(text),
        },
        ParamFailure::Constraint(error) => constraint_error_to_py(py, error, None),
    }
}

/// Implements [`IntoPyErr`] for a param core error family by converting it
/// through [`param_error_to_py`], with no other domain to name.
macro_rules! impl_into_py_err {
    ($($error:ty),+ $(,)?) => {
        $(
            impl IntoPyErr for $error {
                fn into_py_err(self) -> PyErr {
                    Python::attach(|py| param_error_to_py(py, self, None))
                }
            }
        )+
    };
}

impl_into_py_err!(
    DomainError,
    ParamBuildError,
    AssignmentError,
    IntervalError,
    ParamError,
);

/// Return the exception of a failed ordinal construction: a raising `<`'s
/// `TypeError` chained under the core's `TypeError`, another exception it
/// raised as itself, and otherwise the error's own exception.
pub(super) fn ordinal_error_to_py(
    py: Python<'_>,
    error: DomainError,
    raised: Option<PyErr>,
) -> PyErr {
    let chain_under_incomparable = |raised: PyErr| {
        let refused = PyTypeError::new_err(DomainError::IncomparableValues.to_string());
        refused.set_cause(py, Some(raised));
        refused
    };
    match (error, raised) {
        (DomainError::IncomparableValues, Some(raised))
            if raised.is_instance_of::<PyTypeError>(py) =>
        {
            chain_under_incomparable(raised)
        }
        (_, Some(raised)) => raised,
        // A value's `<` that raised: a `TypeError` means the values do not
        // order, as the order error says.
        (DomainError::Custom(source), None) => match crate::exceptions::unbox_py_err(source) {
            Ok(raised) if raised.is_instance_of::<PyTypeError>(py) => {
                chain_under_incomparable(raised)
            }
            Ok(raised) => raised,
            Err(other) => param_error_to_py(py, DomainError::Custom(other), None),
        },
        (error, None) => param_error_to_py(py, error, None),
    }
}
