//! The backends: `fhy_core._rs.SmtSolverBase` and `SimplifierBase`, the
//! bases of the Python `SmtSolver` and `Simplifier` ABCs, the adapters that
//! drive a Python backend from Rust, and `SmtLib2ProcessSolver`, the
//! core's process backend.
//!
//! An adapter calls its Python backend once per question, never per node.
//! A simplifier receives the substituted expression as a Python object: the
//! binding keeps, per thread, a stack of the simplifications in progress,
//! with the input's object and the environment's objects, so the object
//! handed to the hook reuses them, and the object the hook returns is the
//! object the caller gets. No frame is borrowed across a call into Python.

use std::borrow::Cow;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use pyo3::exceptions::PyTypeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::types::{PyDict, PyString, PyTuple, PyType};

use fhy_core::expression::Expression;
use fhy_core::foreign::BoxError;
use fhy_core::solver::{
    CheckLimits, SatResult, Simplifier, SimplifyContext, SimplifyLimits, SmtLib2Process, SmtScript,
    SmtSolver,
};

use crate::expression::{PyExpression, materialize_expression, materialize_substituted};
use crate::kit::gc::Slot;
pub(super) use crate::kit::python::type_name;
use crate::kit::scoped::ScopedStack;
use crate::object_table::ObjectTable;
use crate::pass::refuse_unused_arguments;

use super::values::{PySatResult, PySmtScript};

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

    /// Return the context of the simplification in progress on this
    /// thread, which `simplify` reads its limits from: unbounded outside a
    /// simplification, such as when `simplify` is called directly.
    #[getter]
    #[expect(clippy::unused_self, reason = "a Python property receives the object")]
    fn context(&self) -> PySimplifyContext {
        PySimplifyContext {
            limits: current_limits(),
        }
    }
}

/// What a Python `Simplifier`'s `simplify` is told about its
/// simplification besides the expression: its limits.
///
/// The limits bound the simplification a solver asked for; a simplifier
/// honors them if it can.
#[pyclass(frozen, module = "fhy_core._rs", name = "SimplifyContext")]
pub(crate) struct PySimplifyContext {
    limits: SimplifyLimits,
}

#[pymethods]
impl PySimplifyContext {
    /// Return how long the simplification may run, in seconds, or `None`
    /// when it is unbounded.
    #[getter]
    fn timeout(&self) -> Option<f64> {
        self.limits.timeout().map(|timeout| timeout.as_secs_f64())
    }

    /// Return how long the simplification may run, in whole milliseconds,
    /// or `None` when it is unbounded.
    #[getter]
    fn timeout_milliseconds(&self) -> Option<u64> {
        self.limits
            .timeout()
            .map(|timeout| u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX))
    }

    fn __repr__(&self) -> String {
        match self.timeout_milliseconds() {
            Some(milliseconds) => format!("SimplifyContext(timeout_milliseconds={milliseconds})"),
            None => "SimplifyContext(timeout_milliseconds=None)".to_owned(),
        }
    }
}

// ---------------------------------------------------------------------------
// The adapters
// ---------------------------------------------------------------------------

/// A Python `SmtSolver`, as a core backend: its `check` called with the
/// script and the timeout.
///
/// The object is kept in a [`Slot`], which the solver owns.
pub(super) struct PythonSmtSolver {
    object: Slot,
}

impl fmt::Debug for PythonSmtSolver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PythonSmtSolver").finish_non_exhaustive()
    }
}

impl SmtSolver for PythonSmtSolver {
    fn name(&self) -> Cow<'_, str> {
        Cow::Owned(Python::attach(|py| read_name(&self.object.get(py))))
    }

    /// Call the Python `check(script, *, timeout_milliseconds=...)`.
    ///
    /// Fails with the exception it raises, unchanged, and with a
    /// `TypeError` for a result that is not a `SatResult`.
    fn check(&self, script: &SmtScript, limits: &CheckLimits) -> Result<SatResult, BoxError> {
        Python::attach(|py| -> PyResult<SatResult> {
            let object = &self.object.get(py);
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

/// One simplification in progress on this thread: the input's object, the
/// objects of the environment's values by the identity of their Rust
/// handles, which it holds, and the limits the core handed the simplifier.
struct SimplifyFrame {
    input: Py<PyExpression>,
    known: ObjectTable,
    limits: SimplifyLimits,
    result: Option<Py<PyAny>>,
}

thread_local! {
    /// The simplifications in progress on this thread, innermost last.
    static FRAMES: ScopedStack<SimplifyFrame> = const { ScopedStack::new() };
}

/// Run `simplify` as the simplification of `input`, whose environment's
/// values are `known`, and return its result, the object the Python
/// simplifier returned, if one ran, and `known`.
pub(super) fn run_simplification<R>(
    input: Py<PyExpression>,
    known: ObjectTable,
    simplify: impl FnOnce() -> R,
) -> (R, Option<Py<PyAny>>, ObjectTable) {
    // Popped when the guard drops, on unwind included.
    let scope = ScopedStack::push(
        &FRAMES,
        SimplifyFrame {
            input,
            known,
            limits: SimplifyLimits::new(),
            result: None,
        },
    );
    let result = simplify();
    let frame = scope.pop();
    (result, frame.result, frame.known)
}

/// Return the Python object of `expression`, the substituted input of the
/// innermost simplification, reusing its input's and environment's
/// objects.
fn current_input_object<'py>(
    py: Python<'py>,
    expression: &Expression,
) -> PyResult<Bound<'py, PyAny>> {
    let frame = ScopedStack::with_top(&FRAMES, |frame| {
        frame.map(|frame| (frame.input.clone_ref(py), frame.known.clone_ref(py)))
    });
    match frame {
        Some((input, mut known)) => materialize_substituted(input.bind(py), expression, &mut known),
        None => materialize_expression(py, expression),
    }
}

/// Record `limits` as the limits of the innermost simplification.
fn record_limits(limits: SimplifyLimits) {
    ScopedStack::with_top_mut(&FRAMES, |frame| {
        if let Some(frame) = frame {
            frame.limits = limits;
        }
    });
}

/// Return the limits of the innermost simplification, unbounded outside
/// one.
fn current_limits() -> SimplifyLimits {
    ScopedStack::with_top(&FRAMES, |frame| {
        frame.map_or_else(SimplifyLimits::new, |frame| frame.limits)
    })
}

/// Record `object` as what the innermost simplification's Python
/// simplifier returned.
fn record_result(object: Py<PyAny>) {
    let mut object = Some(object);
    let replaced = ScopedStack::with_top_mut(&FRAMES, |frame| {
        frame.and_then(|frame| std::mem::replace(&mut frame.result, object.take()))
    });
    // Dropped outside the stack's borrow: a finalizer may run Python.
    drop((replaced, object));
}

/// A Python `Simplifier`, as a core backend: its `simplify` called with the
/// substituted expression's object.
///
/// The object is kept in a [`Slot`], which the solver owns.
pub(super) struct PythonSimplifier {
    object: Slot,
}

impl fmt::Debug for PythonSimplifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PythonSimplifier").finish_non_exhaustive()
    }
}

impl Simplifier for PythonSimplifier {
    fn name(&self) -> Cow<'_, str> {
        Cow::Owned(Python::attach(|py| read_name(&self.object.get(py))))
    }

    /// Call the Python `simplify(expression)`, with `context`'s limits
    /// readable from the simplifier's `context`.
    ///
    /// Fails with the exception it raises, unchanged, and with a
    /// `TypeError` for a result that is not an `Expression`.
    fn simplify(
        &self,
        expression: &Expression,
        context: &SimplifyContext<'_>,
    ) -> Result<Expression, BoxError> {
        record_limits(context.limits());
        Python::attach(|py| -> PyResult<Expression> {
            let object = &self.object.get(py);
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
            object: Slot::new(object.clone().unbind()),
        }));
    }
    Err(PyTypeError::new_err(format!(
        "Solver smt_solver must be an SmtSolver, got {}.",
        type_name(object)
    )))
}

/// Return the core backend of the Python `Simplifier` `object`: the native
/// backend of a `SympySimplifier` or a `GroundSimplifier`, or an adapter calling `object`.
///
/// # Errors
///
/// Raises `TypeError` for an object that is not a `SimplifierBase`.
pub(super) fn build_simplifier(object: &Bound<'_, PyAny>) -> PyResult<Arc<dyn Simplifier>> {
    if let Ok(native) = object.cast::<super::sympy::PySympySimplifier>() {
        return Ok(native.get().backend());
    }
    if let Ok(native) = object.cast::<super::ground::PyGroundSimplifier>() {
        return Ok(native.get().backend());
    }
    if object.is_instance_of::<PySimplifierBase>() {
        return Ok(Arc::new(PythonSimplifier {
            object: Slot::new(object.clone().unbind()),
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
/// check, such as `z3 -in` or `cvc5 --lang=smt2`. The timeout bounds the
/// whole check: a program that has not answered, or has not exited after
/// answering, by the deadline is killed, together with its process group
/// on Unix. A wrapper script should `exec` its solver.
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
    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.program)?;
        visit.call(&self.args)?;
        Ok(())
    }

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

#[cfg(test)]
mod scoped_stack_tests {
    use super::*;

    /// Test a panic inside a simplification leaves no frame behind, so the
    /// next one does not reuse a dead input.
    #[test]
    fn a_panic_inside_a_simplification_leaves_the_stack_empty() {
        Python::initialize();
        Python::attach(|py| {
            let input = PyExpression::bare_for_tests(py, Expression::from(1));
            let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_simplification(input, ObjectTable::new(), || -> () {
                    panic!("inside a simplification")
                })
            }));

            let _panic = unwound.unwrap_err();
            assert_eq!(ScopedStack::depth(&FRAMES), 0);
        });
    }
}
