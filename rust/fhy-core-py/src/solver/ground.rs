//! `fhy_core._rs.GroundSimplifier`: the core's ground simplifier as a
//! native `Simplifier`, alone or in front of another one.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::types::PyTuple;

use fhy_core::expression::Expression;
use fhy_core::solver::{GroundSimplifier, GroundWithFallback, Simplifier, SimplifyContext};

use crate::expression::{PyExpression, materialize_substituted, registry_snapshot};
use crate::object_table::ObjectTable;
use crate::util::gc::{Slots, collect_slots};

use super::backends::{PySimplifierBase, build_simplifier, read_expression, run_simplification};
use super::error::backend_error_to_py;

/// The ground simplifier: a `Simplifier` that folds an expression with no
/// free identifier in exact arithmetic, with no SymPy and no Python.
///
/// It is the core's `GroundSimplifier` with its default strategies: the
/// registered native constants, decimal-literal normalization, exact
/// integer and rational arithmetic, comparisons, logical operators, a
/// decided piecewise and the exact built-ins SymPy folds itself. Rust
/// callers extend it with their own strategies; the composed built-ins,
/// which SymPy refuses until they are inlined, are an opt-in one, not
/// included. Where it folds, its result is exactly what the SymPy backend
/// returns for the same input. Where it cannot match SymPy exactly (a free
/// identifier, a float, a user function, an irrational or undefined value,
/// a power beyond a million bits, an integer result beyond a million bits,
/// a fraction with a part beyond 4096 bits, an expression nested more than
/// 256 deep) it returns the expression unchanged, and never approximates.
/// It honors the simplification's timeout: when the time is up it returns
/// the expression unchanged, never a partial result.
///
/// With `fallback`, a `Simplifier`, it is the chain: it tries the ground
/// simplifier first and asks `fallback` for what it declines, so with a
/// `SympySimplifier` the answer is SymPy's, faster where the expression is
/// ground. The timeout bounds the ground part, and the fallback is asked
/// under what is left of it (a fallback such as SymPy that cannot be
/// cancelled still runs to its end). A ground simplifier given as the
/// fallback is unwrapped: its own fallback, if it has one, is the fallback.
///
/// A `Solver` holding it simplifies in Rust with the interpreter detached;
/// a fallback that is a Python backend attaches again for its own call.
///
/// A direct `simplify` does not screen the expression as a `Solver` does:
/// an ill-formed piecewise or logical operand is returned unchanged where
/// `SympySimplifier.simplify` raises.
#[pyclass(extends = PySimplifierBase, frozen, module = "fhy_core._rs", name = "GroundSimplifier")]
pub(crate) struct PyGroundSimplifier {
    backend: Arc<dyn Simplifier>,
    fallback: Option<Py<PyAny>>,
    /// The slots a Python fallback's adapter reads its object from, which
    /// this object owns.
    slots: Slots,
}

impl PyGroundSimplifier {
    /// Return the core backend, shared.
    pub(super) fn backend(&self) -> Arc<dyn Simplifier> {
        Arc::clone(&self.backend)
    }
}

#[pymethods]
impl PyGroundSimplifier {
    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(self.fallback.as_ref())?;
        self.slots.traverse(&visit)
    }

    /// Create the simplifier, asking `fallback`, a `Simplifier`, for what
    /// it declines when one is given.
    ///
    /// Raises `TypeError` for a fallback that is not a `Simplifier`.
    #[new]
    #[pyo3(signature = (fallback = None))]
    fn new(
        py: Python<'_>,
        fallback: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let fallback = match fallback.filter(|backend| !backend.is_none()) {
            Some(object) => match object.cast::<Self>() {
                Ok(ground) => ground
                    .get()
                    .fallback
                    .as_ref()
                    .map(|inner| inner.bind(py).clone()),
                Err(_not_ground) => Some(object.clone()),
            },
            None => None,
        };
        let (backend, slots) = collect_slots(|| -> PyResult<Arc<dyn Simplifier>> {
            Ok(match &fallback {
                Some(object) => {
                    Arc::new(GroundWithFallback::from_shared(build_simplifier(object)?))
                }
                None => Arc::new(GroundSimplifier::new()),
            })
        });
        Ok(
            PyClassInitializer::from(PySimplifierBase).add_subclass(Self {
                backend: backend?,
                fallback: fallback.map(Bound::unbind),
                slots,
            }),
        )
    }

    /// The backend's name: `"ground"`, or `"ground+"` and the fallback's.
    #[getter]
    fn name(&self) -> String {
        self.backend.name().into_owned()
    }

    /// The simplifier asked for what the ground fold declines, or `None`.
    #[getter]
    fn fallback(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.fallback.as_ref().map(|object| object.clone_ref(py))
    }

    /// Return the simplification of `expression`: the literal it folds to,
    /// or what the fallback returns, or `expression` itself when there is
    /// no fallback and the fold declines.
    ///
    /// Raises `TypeError` for an `expression` that is not an `Expression`,
    /// and the fallback's exception, if it fails.
    fn simplify<'py>(&self, expression: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = expression.py();
        let input = read_expression(expression, "GroundSimplifier.simplify")?;
        let rust_input = input.get().expression().clone();
        let registry = registry_snapshot();
        let backend = &self.backend;
        let (result, returned, mut known) =
            run_simplification(input.clone().unbind(), ObjectTable::new(), || {
                py.detach(|| {
                    backend.simplify(
                        &rust_input,
                        &SimplifyContext::from_registry(registry.registry()),
                    )
                })
            });
        let result = result.map_err(|source| backend_error_to_py(py, backend.name(), source))?;
        if let Some(returned) = returned {
            let returned = returned.into_bound(py);
            if let Ok(object) = returned.cast::<PyExpression>() {
                if Expression::ptr_eq(object.get().expression(), &result) {
                    return Ok(returned);
                }
            }
        }
        materialize_substituted(&input, &result, &mut known)
    }

    /// Pickle as a call of the class with the fallback.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let arguments = match &slf.get().fallback {
            Some(fallback) => PyTuple::new(py, [fallback.bind(py).clone()])?,
            None => PyTuple::empty(py),
        };
        PyTuple::new(py, [slf.get_type().into_any(), arguments.into_any()])
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        match &self.fallback {
            Some(fallback) => Ok(format!("GroundSimplifier({})", fallback.bind(py).repr()?)),
            None => Ok("GroundSimplifier()".to_owned()),
        }
    }
}
