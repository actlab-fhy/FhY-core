//! The SymPy backend, and `fhy_core._rs.SympySimplifier`, the backend as a
//! native `Simplifier`, with the Python exceptions of its errors.
//!
//! [`SympySimplifier`] lowers an expression to SymPy, simplifies it, and
//! lifts the result, in the interpreter the extension runs in; its
//! submodules are the lowering, the lifting, the substitution, the
//! simplification's workarounds, the Boolean positions they share, the
//! loading of SymPy and of the prelude (`prelude.py`, the Python module of
//! the classes and hooks only Python code can define), and the errors. A
//! `Solver` holding the pyclass simplifies in Rust, from the facade into
//! SymPy, with no Python backend between. Its methods expose the backend's
//! SymPy-level operations, over which
//! `fhy_core.symbolic.expression.passes.sympy` defines the bridge's
//! functions and passes.
//!
//! The backend's stories are this module's `#[cfg(test)]` submodules, run
//! by `cargo test -p fhy-core-py` in an interpreter the test binary
//! embeds; they need Python with SymPy.

mod address_hash;
mod boolean;
mod error;
mod lift;
mod load;
mod lower;
mod simplifier;
mod simplify;
mod substitute;

#[cfg(test)]
mod error_stories;
#[cfg(test)]
mod lifting_stories;
#[cfg(test)]
mod lowering_stories;
#[cfg(test)]
mod properties;
#[cfg(test)]
mod simplify_stories;
#[cfg(test)]
mod test_support;

pub(crate) use error::{SympyError, SympyErrorKind, SympyPhase, SympyUnavailableError};
pub(crate) use simplifier::SympySimplifier;

use std::collections::HashMap;
use std::sync::Arc;

use pyo3::exceptions::{
    PyException, PyNotImplementedError, PyRuntimeError, PyTypeError, PyValueError,
};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyMapping, PyTuple};

use fhy_core::expression::Expression;
use fhy_core::identifier::Identifier;
use fhy_core::solver::SimplifyContext;

use crate::error::IntoPyErr;
use crate::expression::{
    PyExpression, materialize_expression, materialize_substituted, registry_snapshot,
};
use crate::identifier::{read_identifier_id, restore_identifier};

use super::backends::{PySimplifierBase, type_name};

/// The registered name of the bridge's pass for each phase that runs as a
/// pass.
const fn pass_name(phase: SympyPhase) -> Option<&'static str> {
    match phase {
        SympyPhase::Lowering => Some("fhy_core.symbolic.expression.to_sympy"),
        SympyPhase::Substitution => Some("fhy_core.symbolic.expression.substitute_sympy_variables"),
        SympyPhase::Lifting => Some("fhy_core.symbolic.expression.from_sympy"),
        SympyPhase::Simplification => None,
    }
}

/// The message of `SolverBackendUnavailableError` for a missing SymPy.
const UNAVAILABLE_MESSAGE: &str = "The sympy solver backend needs the sympy package, which is \
     not installed; install it with `pip install fhy_core[sympy]`, or `pip install \
     fhy_core[solvers]` for every solver backend.";

/// Return the exception of `error` raised in `phase`, unwrapped.
fn exception_of(py: Python<'_>, error: SympyError) -> PyErr {
    let text = error.to_string();
    match error.into_kind() {
        SympyErrorKind::Unavailable(unavailable) => {
            let error =
                crate::exceptions::SOLVER_BACKEND_UNAVAILABLE_ERROR.err(py, (UNAVAILABLE_MESSAGE,));
            let (SympyUnavailableError::MissingSympy(cause)
            | SympyUnavailableError::Incompatible(cause)) = unavailable;
            error.set_cause(py, Some(cause));
            error
        }
        SympyErrorKind::IllTyped(error) => error.into_py_err(),
        SympyErrorKind::BoundNativeConstant(_) => {
            crate::exceptions::NATIVE_CONSTANT_BINDING_ERROR.err(py, (text,))
        }
        SympyErrorKind::ComplexInfinity => {
            crate::exceptions::COMPLEX_INFINITY_LIFT_ERROR.err(py, (text,))
        }
        SympyErrorKind::PartialPiecewise(_) => {
            crate::exceptions::PARTIAL_PIECEWISE_ERROR.err(py, (text,))
        }
        SympyErrorKind::Arity(_) => PyValueError::new_err(text),
        SympyErrorKind::Implies(node) => {
            warn_implies(py, &node);
            PyNotImplementedError::new_err(text)
        }
        SympyErrorKind::UnreadableSymbol(_) => PyRuntimeError::new_err(text),
        SympyErrorKind::Python(error) => error,
        _ => PyTypeError::new_err(text),
    }
}

/// Log the refusal of a SymPy `Implies` at WARNING on the bridge's logger.
fn warn_implies(py: Python<'_>, node: &str) {
    let logged = py
        .import(intern!(py, "fhy_core.logger"))
        .and_then(|logger| {
            logger.call_method1(
                intern!(py, "get_logger"),
                ("fhy_core.symbolic.expression.passes.sympy",),
            )
        })
        .and_then(|logger| {
            logger.call_method1(
                intern!(py, "warning"),
                ("encountered unsupported Implies node %s", node),
            )
        });
    drop(logged);
}

/// Return whether [`sympy_error_to_py`], wrapping, raises `error` as a
/// `PassExecutionError`.
pub(super) fn is_raised_as_pass_error(py: Python<'_>, error: &SympyError) -> bool {
    if pass_name(error.phase()).is_none() || is_unwrapped(error) {
        return false;
    }
    match error.kind() {
        SympyErrorKind::Python(error) => error.is_instance_of::<PyException>(py),
        _ => true,
    }
}

/// Return whether `error` keeps its own class whatever phase it arose in:
/// the screen's refusal, a bound constant, and a missing backend, which the
/// Python bridge raised before any pass ran.
const fn is_unwrapped(error: &SympyError) -> bool {
    matches!(
        error.kind(),
        SympyErrorKind::IllTyped(_)
            | SympyErrorKind::BoundNativeConstant(_)
            | SympyErrorKind::Unavailable(_)
    )
}

/// Return the Python exception of `error`.
///
/// With `wrap`, a failure of a phase the bridge runs as a pass is raised
/// as the `PassExecutionError` that pass raised, naming it, with the
/// exception as its `__cause__`, as the Python bridge's passes did; a
/// `BaseException` that is not an `Exception` passes through.
pub(super) fn sympy_error_to_py(py: Python<'_>, error: SympyError, wrap: bool) -> PyErr {
    let name = pass_name(error.phase()).filter(|_| wrap && !is_unwrapped(&error));
    let exception = exception_of(py, error);
    let Some(name) = name else {
        return exception;
    };
    if !exception.is_instance_of::<PyException>(py) {
        return exception;
    }
    let keywords = PyDict::new(py);
    if let Err(error) = keywords
        .set_item(intern!(py, "pass_name"), name)
        .and_then(|()| keywords.set_item(intern!(py, "hook"), "run_pass"))
    {
        return error;
    }
    match crate::exceptions::PASS_EXECUTION_ERROR.build(
        py,
        (format!("pass {name:?} failed in run_pass"),),
        Some(&keywords),
    ) {
        Ok(wrapped) => {
            wrapped.set_cause(py, Some(exception));
            wrapped
        }
        Err(error) => error,
    }
}

/// The SymPy backend: a `Simplifier` that lowers an expression to
/// SymPy, simplifies it best-effort, and lifts the result.
///
/// Constructing it imports nothing; its first operation imports SymPy, and
/// raises `SolverBackendUnavailableError` when SymPy is not installed. A
/// `Solver` holding it simplifies in Rust. It does not enforce a
/// simplification's `timeout_milliseconds`: SymPy has no cancellation.
#[pyclass(extends = PySimplifierBase, frozen, module = "fhy_core._rs", name = "SympySimplifier")]
pub(crate) struct PySympySimplifier {
    backend: Arc<SympySimplifier>,
}

impl PySympySimplifier {
    /// Return the core backend, shared.
    pub(super) fn backend(&self) -> Arc<SympySimplifier> {
        Arc::clone(&self.backend)
    }
}

/// Return the Rust expression of `value`, an `Expression`.
fn read_expression<'py>(
    value: &Bound<'py, PyAny>,
    owner: &str,
) -> PyResult<Bound<'py, PyExpression>> {
    value
        .cast::<PyExpression>()
        .cloned()
        .map_err(|_not_an_expression| {
            PyTypeError::new_err(format!(
                "{owner} expression must be an Expression, got {}.",
                type_name(value)
            ))
        })
}

#[pymethods]
impl PySympySimplifier {
    /// Create the backend, loading nothing.
    #[new]
    fn new() -> PyClassInitializer<Self> {
        PyClassInitializer::from(PySimplifierBase).add_subclass(Self {
            backend: Arc::new(SympySimplifier::new()),
        })
    }

    /// The backend's name, `"sympy"`.
    #[getter]
    fn name(&self) -> String {
        fhy_core::solver::Simplifier::name(self.backend.as_ref()).into_owned()
    }

    /// Import SymPy and load the backend's prelude now.
    ///
    /// Raises `SolverBackendUnavailableError` when SymPy is not installed.
    fn load(&self, py: Python<'_>) -> PyResult<()> {
        self.backend.load(py).map_err(|error| {
            sympy_error_to_py(
                py,
                SympyError::new(SympyPhase::Lowering, SympyErrorKind::Unavailable(error)),
                false,
            )
        })
    }

    /// Return the simplification of `expression`: lowered, simplified
    /// best-effort, and lifted, with the registry's constants.
    ///
    /// Raises the bridge's exceptions: `NonBooleanLogicalOperandError`,
    /// `PassExecutionError` from the lowering or lifting pass with the
    /// failure as its cause, or `sympy.simplify`'s own exception.
    fn simplify<'py>(&self, expression: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = expression.py();
        let input = read_expression(expression, "SympySimplifier.simplify")?;
        let registry = registry_snapshot();
        let context = SimplifyContext::from_registry(registry.registry());
        let result = fhy_core::solver::Simplifier::simplify(
            self.backend.as_ref(),
            input.get().expression(),
            &context,
        )
        .map_err(|error| match error.downcast::<SympyError>() {
            Ok(error) => sympy_error_to_py(py, *error, true),
            Err(error) => PyRuntimeError::new_err(error.to_string()),
        })?;
        materialize_substituted(
            &input,
            &result,
            &mut crate::object_table::ObjectTable::new(),
        )
    }

    /// Return the SymPy object of `expression`, with the registry's
    /// constants.
    ///
    /// Raises the lowering's exception itself: `TypeError` for a call SymPy
    /// has no function for, and `NonBooleanLogicalOperandError`.
    fn lower<'py>(&self, expression: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = expression.py();
        let input = read_expression(expression, "SympySimplifier.lower")?;
        let registry = registry_snapshot();
        self.backend
            .lower(
                py,
                input.get().expression(),
                &SimplifyContext::from_registry(registry.registry()),
            )
            .map_err(|error| sympy_error_to_py(py, error, false))
    }

    /// Return the expression the SymPy object `sympy_expression` denotes.
    ///
    /// Raises the lifting's exception itself: `ComplexInfinityLiftError`,
    /// `PartialPiecewiseError`, `TypeError` for an unsupported node,
    /// `NotImplementedError` for an `Implies`.
    fn lift<'py>(&self, sympy_expression: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = sympy_expression.py();
        let expression = self
            .backend
            .lift(sympy_expression)
            .map_err(|error| sympy_error_to_py(py, error, false))?;
        materialize_expression(py, &expression)
    }

    /// Return the best-effort simplification of the SymPy object
    /// `sympy_expression`, or the object itself where SymPy gives up.
    fn simplify_object<'py>(
        &self,
        sympy_expression: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = sympy_expression.py();
        self.backend
            .simplify_object(sympy_expression)
            .map_err(|error| sympy_error_to_py(py, error, false))
    }

    /// Return `sympy_expression` with `environment`, a mapping of
    /// identifiers to expressions, substituted simultaneously.
    ///
    /// Raises `NativeConstantBindingError` when `environment` binds a
    /// native constant the object refers to, and a failure of the
    /// substitution or of lowering a value as the `PassExecutionError` of
    /// its pass.
    fn substitute<'py>(
        &self,
        sympy_expression: &Bound<'py, PyAny>,
        environment: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = sympy_expression.py();
        let mut bindings: HashMap<Identifier, Expression> = HashMap::new();
        for item in environment.cast::<PyMapping>()?.items()?.iter() {
            let (key, value) = item.extract::<(Bound<'py, PyAny>, Bound<'py, PyAny>)>()?;
            if read_identifier_id(&key)?.is_none() {
                continue;
            }
            let value = read_expression(&value, "SympySimplifier.substitute")?;
            bindings.insert(
                restore_identifier(&key, "environment", "key")?,
                value.get().expression().clone(),
            );
        }
        let registry = registry_snapshot();
        self.backend
            .substitute(
                sympy_expression,
                &bindings,
                &SimplifyContext::from_registry(registry.registry()),
            )
            .map_err(|error| sympy_error_to_py(py, error, true))
    }

    /// Return `sympy_expression` with `replacements`, a mapping of SymPy
    /// symbols to SymPy objects, applied simultaneously, keeping every
    /// Boolean position Boolean.
    ///
    /// Raises the exception a rebuilt node raises, itself.
    fn substitute_symbols<'py>(
        &self,
        sympy_expression: &Bound<'py, PyAny>,
        replacements: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = sympy_expression.py();
        self.backend
            .substitute_symbols(sympy_expression, replacements)
            .map_err(|error| sympy_error_to_py(py, error, false))
    }

    /// Pickle as a call of the class: the backend holds no state.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        PyTuple::new(
            py,
            [slf.get_type().into_any(), PyTuple::empty(py).into_any()],
        )
    }

    #[expect(clippy::unused_self, reason = "a Python method receives the object")]
    const fn __repr__(&self) -> &'static str {
        "SympySimplifier()"
    }
}
