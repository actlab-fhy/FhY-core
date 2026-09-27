//! The backends: `fhy_core._rs.SmtSolverBase` and `SimplifierBase`, the
//! bases of the Python `SmtSolver` and `Simplifier` ABCs (P3; D-S8-11), the
//! adapters that drive a Python backend from Rust, and
//! `SmtLib2ProcessSolver`, the core's process backend.
//!
//! An adapter calls its Python backend once per question, never per node.
//! A simplifier receives the substituted expression as a Python object: the
//! binding keeps, per thread, a stack of the simplifications in progress,
//! with the input's object and the environment's objects, so the object
//! handed to the hook reuses them, and the object the hook returns is the
//! object the caller gets. No frame is borrowed across a call into Python.

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use pyo3::exceptions::PyTypeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString, PyTuple, PyType};

use fhy_core::expression::Expression;
use fhy_core::foreign::BoxError;
use fhy_core::solver::{
    CheckLimits, SatResult, Simplifier, SimplifyContext, SmtLib2Process, SmtScript, SmtSolver,
};
use fhy_core::tree::{NodeHandle, NodeIdentity};

use crate::expression::{PyExpression, materialize_expression, materialize_substituted};
use crate::pass::refuse_unused_arguments;

use super::values::{PySatResult, PySmtScript};

/// Return the name of the type of `value`, for messages.
pub(super) fn type_name(value: &Bound<'_, PyAny>) -> String {
    value
        .get_type()
        .name()
        .map_or_else(|_| "?".to_owned(), |name| name.to_string())
}

/// Return the `name` of the Python backend `object`, or its class name if
/// reading it fails or gives no `str`.
fn read_name(object: &Bound<'_, PyAny>) -> String {
    object
        .getattr(intern!(object.py(), "name"))
        .ok()
        .and_then(|name| name.extract::<String>().ok())
        .unwrap_or_else(|| type_name(object))
}

/// Return `limits`' timeout in whole milliseconds.
fn timeout_milliseconds(limits: &CheckLimits) -> Option<u64> {
    limits
        .timeout()
        .map(|timeout| u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX))
}

// ---------------------------------------------------------------------------
// The bases
// ---------------------------------------------------------------------------

/// The base of the Python `SmtSolver` ABC: a backend that decides SMT-LIB2
/// scripts, implemented in Python or natively.
///
/// It takes any constructor arguments, so a Python subclass defines its own
/// `__init__`. A solver calls the subclass's `check` from Rust, once per
/// question.
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "SmtSolverBase")]
pub(crate) struct PySmtSolverBase;

#[pymethods]
impl PySmtSolverBase {
    /// Accept any arguments, so a subclass's `__init__` takes its own.
    ///
    /// Raises `TypeError` for arguments a subclass without an `__init__`
    /// of its own was given.
    #[new]
    #[classmethod]
    #[pyo3(signature = (*args, **kwargs))]
    fn new(
        cls: &Bound<'_, PyType>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        refuse_unused_arguments(cls, args, kwargs)?;
        Ok(Self)
    }
}

/// The base of the Python `Simplifier` ABC: a backend that simplifies
/// expressions, implemented in Python.
///
/// It takes any constructor arguments, so a Python subclass defines its own
/// `__init__`. A solver calls the subclass's `simplify` from Rust, once per
/// question.
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "SimplifierBase")]
pub(crate) struct PySimplifierBase;

#[pymethods]
impl PySimplifierBase {
    /// Accept any arguments, so a subclass's `__init__` takes its own.
    ///
    /// Raises `TypeError` for arguments a subclass without an `__init__`
    /// of its own was given.
    #[new]
    #[classmethod]
    #[pyo3(signature = (*args, **kwargs))]
    fn new(
        cls: &Bound<'_, PyType>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        refuse_unused_arguments(cls, args, kwargs)?;
        Ok(Self)
    }
}

// ---------------------------------------------------------------------------
// The adapters
// ---------------------------------------------------------------------------

/// A Python `SmtSolver`, as a core backend: its `check` called with the
/// script and the timeout.
pub(super) struct PythonSmtSolver {
    object: Py<PyAny>,
}

impl fmt::Debug for PythonSmtSolver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PythonSmtSolver").finish_non_exhaustive()
    }
}

impl SmtSolver for PythonSmtSolver {
    fn name(&self) -> Cow<'_, str> {
        Cow::Owned(Python::attach(|py| read_name(self.object.bind(py))))
    }

    /// Call the Python `check(script, *, timeout_milliseconds=...)`.
    ///
    /// Fails with the exception it raises, unchanged, and with a
    /// `TypeError` for a result that is not a `SatResult`.
    fn check(&self, script: &SmtScript, limits: &CheckLimits) -> Result<SatResult, BoxError> {
        Python::attach(|py| -> PyResult<SatResult> {
            let object = self.object.bind(py);
            let script = Bound::new(py, PySmtScript::new(script.clone()))?;
            let keywords = PyDict::new(py);
            keywords.set_item(
                intern!(py, "timeout_milliseconds"),
                timeout_milliseconds(limits),
            )?;
            let result = object.call_method(intern!(py, "check"), (script,), Some(&keywords))?;
            match result.cast::<PySatResult>() {
                Ok(result) => Ok(result.get().result().clone()),
                Err(_not_a_result) => Err(PyTypeError::new_err(format!(
                    "{}.check must return a SatResult, got {}.",
                    type_name(object),
                    type_name(&result)
                ))),
            }
        })
        .map_err(|error| Box::new(error) as BoxError)
    }
}

/// One simplification in progress on this thread: the input's object, and
/// the objects of the environment's values by the identity of their Rust
/// handles, which it holds.
struct SimplifyFrame {
    input: Py<PyExpression>,
    known: Vec<(Expression, Py<PyAny>)>,
    result: Option<Py<PyAny>>,
}

thread_local! {
    /// The simplifications in progress on this thread, innermost last.
    static FRAMES: RefCell<Vec<SimplifyFrame>> = const { RefCell::new(Vec::new()) };
}

/// Run `simplify` as the simplification of `input`, whose environment's
/// values are `known`, and return its result and the object the Python
/// simplifier returned, if one ran.
pub(super) fn run_simplification<R>(
    input: Py<PyExpression>,
    known: Vec<(Expression, Py<PyAny>)>,
    simplify: impl FnOnce() -> R,
) -> (R, Option<Py<PyAny>>) {
    FRAMES.with(|frames| {
        frames.borrow_mut().push(SimplifyFrame {
            input,
            known,
            result: None,
        });
    });
    let result = simplify();
    let frame = FRAMES.with(|frames| frames.borrow_mut().pop());
    (result, frame.and_then(|frame| frame.result))
}

/// Return the Python object of `expression`, the substituted input of the
/// innermost simplification, reusing its input's and environment's
/// objects.
fn current_input_object<'py>(
    py: Python<'py>,
    expression: &Expression,
) -> PyResult<Bound<'py, PyAny>> {
    let frame = FRAMES.with(|frames| {
        frames.borrow().last().map(|frame| {
            (
                frame.input.clone_ref(py),
                frame
                    .known
                    .iter()
                    .map(|(handle, object)| (handle.identity(), object.clone_ref(py)))
                    .collect::<Vec<(NodeIdentity, Py<PyAny>)>>(),
            )
        })
    });
    match frame {
        Some((input, known)) => {
            let known: HashMap<NodeIdentity, Bound<'py, PyAny>> = known
                .into_iter()
                .map(|(identity, object)| (identity, object.into_bound(py)))
                .collect();
            materialize_substituted(input.bind(py), expression, known)
        }
        None => materialize_expression(py, expression),
    }
}

/// Record `object` as what the innermost simplification's Python
/// simplifier returned.
fn record_result(object: Py<PyAny>) {
    FRAMES.with(|frames| {
        if let Some(frame) = frames.borrow_mut().last_mut() {
            frame.result = Some(object);
        }
    });
}

/// A Python `Simplifier`, as a core backend: its `simplify` called with the
/// substituted expression's object.
pub(super) struct PythonSimplifier {
    object: Py<PyAny>,
}

impl fmt::Debug for PythonSimplifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PythonSimplifier").finish_non_exhaustive()
    }
}

impl Simplifier for PythonSimplifier {
    fn name(&self) -> Cow<'_, str> {
        Cow::Owned(Python::attach(|py| read_name(self.object.bind(py))))
    }

    /// Call the Python `simplify(expression)`.
    ///
    /// Fails with the exception it raises, unchanged, and with a
    /// `TypeError` for a result that is not an `Expression`.
    fn simplify(
        &self,
        expression: &Expression,
        _context: &SimplifyContext<'_>,
    ) -> Result<Expression, BoxError> {
        Python::attach(|py| -> PyResult<Expression> {
            let object = self.object.bind(py);
            let input = current_input_object(py, expression)?;
            let result = object.call_method1(intern!(py, "simplify"), (input,))?;
            let handle = match result.cast::<PyExpression>() {
                Ok(expression) => expression.get().expression().clone(),
                Err(_not_an_expression) => {
                    return Err(PyTypeError::new_err(format!(
                        "{}.simplify must return an Expression, got {}.",
                        type_name(object),
                        type_name(&result)
                    )));
                }
            };
            record_result(result.unbind());
            Ok(handle)
        })
        .map_err(|error| Box::new(error) as BoxError)
    }
}

/// Return the core backend of the Python `SmtSolver` `object`: the native
/// backend of an `SmtLib2ProcessSolver`, or an adapter calling `object`.
///
/// # Errors
///
/// Raises `TypeError` for an object that is not an `SmtSolverBase`.
pub(super) fn build_smt_solver(object: &Bound<'_, PyAny>) -> PyResult<Arc<dyn SmtSolver>> {
    if let Ok(process) = object.cast::<PySmtLib2ProcessSolver>() {
        return Ok(Arc::new(process.get().backend.clone()));
    }
    if object.is_instance_of::<PySmtSolverBase>() {
        return Ok(Arc::new(PythonSmtSolver {
            object: object.clone().unbind(),
        }));
    }
    Err(PyTypeError::new_err(format!(
        "Solver smt_solver must be an SmtSolver, got {}.",
        type_name(object)
    )))
}

/// Return the core backend of the Python `Simplifier` `object`: the native
/// backend of a `SympySimplifier`, or an adapter calling `object`.
///
/// # Errors
///
/// Raises `TypeError` for an object that is not a `SimplifierBase`.
pub(super) fn build_simplifier(object: &Bound<'_, PyAny>) -> PyResult<Arc<dyn Simplifier>> {
    if let Ok(native) = object.cast::<super::sympy::PySympySimplifier>() {
        return Ok(native.get().backend());
    }
    if object.is_instance_of::<PySimplifierBase>() {
        return Ok(Arc::new(PythonSimplifier {
            object: object.clone().unbind(),
        }));
    }
    Err(PyTypeError::new_err(format!(
        "Solver simplifier must be a Simplifier, got {}.",
        type_name(object)
    )))
}

// ---------------------------------------------------------------------------
// SmtLib2ProcessSolver
// ---------------------------------------------------------------------------

/// The core's process backend: it runs an SMT-LIB2 executable for each
/// check, such as `z3 -in` or `cvc5 --lang=smt2`, and kills it at the
/// timeout.
///
/// It is a native `SmtSolver`: a solver holding it checks in Rust, with the
/// interpreter detached.
#[pyclass(extends = PySmtSolverBase, frozen, module = "fhy_core._rs", name = "SmtLib2ProcessSolver")]
pub(crate) struct PySmtLib2ProcessSolver {
    backend: SmtLib2Process,
    program: Py<PyString>,
    args: Py<PyTuple>,
}

#[pymethods]
impl PySmtLib2ProcessSolver {
    /// Create the backend running `program`, a path, with the `str`
    /// arguments `args`.
    ///
    /// Raises `TypeError` for a program that is not a path or an argument
    /// that is not a `str`.
    #[new]
    #[pyo3(signature = (program, args = None))]
    fn new(
        program: &Bound<'_, PyAny>,
        args: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let py = program.py();
        let path = py
            .import(intern!(py, "os"))?
            .call_method1(intern!(py, "fspath"), (program,))
            .and_then(|path| Ok(path.cast_into::<PyString>()?))
            .map_err(|_not_a_path| {
                PyTypeError::new_err(format!(
                    "SmtLib2ProcessSolver program must be a str or a path, got {}.",
                    type_name(program)
                ))
            })?;
        let mut arguments: Vec<Bound<'_, PyString>> = Vec::new();
        if let Some(args) = args.filter(|args| !args.is_none()) {
            for argument in args.try_iter()? {
                let argument = argument?;
                arguments.push(argument.cast_into::<PyString>().map_err(|error| {
                    PyTypeError::new_err(format!(
                        "SmtLib2ProcessSolver args must be strs, got {}.",
                        type_name(&error.into_inner())
                    ))
                })?);
            }
        }
        let backend = SmtLib2Process::new(path.to_str()?).with_args(
            arguments
                .iter()
                .map(|argument| argument.to_str().map(str::to_owned))
                .collect::<PyResult<Vec<String>>>()?,
        );
        Ok(
            PyClassInitializer::from(PySmtSolverBase).add_subclass(Self {
                backend,
                program: path.unbind(),
                args: PyTuple::new(py, arguments)?.unbind(),
            }),
        )
    }

    /// The backend's name: the program's file name.
    #[getter]
    fn name(&self) -> String {
        self.backend.name().into_owned()
    }

    /// The program the backend runs.
    #[getter]
    fn program(&self, py: Python<'_>) -> Py<PyString> {
        self.program.clone_ref(py)
    }

    /// The arguments the program runs with.
    #[getter]
    fn args(&self, py: Python<'_>) -> Py<PyTuple> {
        self.args.clone_ref(py)
    }

    /// Decide `script`, bounded by `timeout_milliseconds`, as a solver
    /// does, with the interpreter detached.
    ///
    /// Raises `SolverBackendError` when the program fails.
    #[pyo3(signature = (script, *, timeout_milliseconds = None))]
    fn check(
        &self,
        py: Python<'_>,
        script: &Bound<'_, PySmtScript>,
        timeout_milliseconds: Option<u64>,
    ) -> PyResult<PySatResult> {
        let limits = timeout_milliseconds.map_or_else(CheckLimits::new, |milliseconds| {
            CheckLimits::new().with_timeout(Duration::from_millis(milliseconds))
        });
        let script = script.get();
        let result = py.detach(|| self.backend.check(script.script(), &limits));
        result.map(PySatResult::from).map_err(|source| {
            super::error::solve_error_to_py(
                py,
                fhy_core::solver::SolveError::Backend {
                    backend: self.backend.name().into_owned(),
                    source,
                },
            )
        })
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "SmtLib2ProcessSolver({}, {})",
            self.program.bind(py).repr()?,
            self.args.bind(py).repr()?
        ))
    }
}
