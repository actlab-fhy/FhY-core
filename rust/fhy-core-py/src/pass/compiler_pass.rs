//! `fhy_core._rs.CompilerPassBase`, the base of the Python `CompilerPass`
//! ABC, and the adapter that drives a Python pass through the core's
//! lifecycle.
//!
//! The adapter implements the core's [`CompilerPass`] over [`PyIr`] by
//! calling the Python hooks: `validate_input`, `should_run` then
//! `get_noop_output` as `skip`, `run_pass` as `run`, `validate_output`,
//! `did_change`, and `get_preserved_analyses` as `preserved_analyses`. The
//! class records, once, which of the hooks with a default it overrides
//! (`CompilerPass._python_hooks`); the adapter runs the defaults of the
//! others in Rust, without calling Python.

use std::borrow::Cow;
use std::error::Error;
use std::fmt;
use std::mem;
use std::sync::{Mutex, MutexGuard, PoisonError};

use pyo3::exceptions::{PyException, PyRuntimeError, PyTypeError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::types::{PyDict, PyString, PyTuple, PyType};

use fhy_core::diagnostic::{Diagnostic, Note};
use fhy_core::foreign::BoxError;
use fhy_core::pass::{CompilerPass, ExecutePass, PassContext, PreservedAnalyses};

use crate::diagnostic::{borrow_python_diagnostic, diagnostic_to_python, level_to_python};

use super::analysis::{PyPreservedAnalyses, preserved_to_python};
use super::context::{self, FrameGuard, Recording};
use super::convert::diagnostics_to_python;
use super::error::{chain_of_exception, error_to_python, take_nested};
use super::ir::PyIr;
use super::records::PyPassResult;
use super::scope::{self, NestedNote, ScopeGuard};

/// The Python module that defines the public pass classes and the logging
/// helpers the binding calls.
pub(super) const CORE_MODULE: &str = "fhy_core.pass_infrastructure.core";

/// The bit of each hook with a default in `CompilerPass._python_hooks`, set
/// when a class overrides the hook. Matches the Python implementation:
/// `fhy_core.pass_infrastructure.core._HOOKS_WITH_RUST_DEFAULTS`.
mod hook_bit {
    pub(super) const VALIDATE_INPUT: u32 = 1 << 0;
    pub(super) const SHOULD_RUN: u32 = 1 << 1;
    pub(super) const VALIDATE_OUTPUT: u32 = 1 << 2;
    pub(super) const DID_CHANGE: u32 = 1 << 3;
    pub(super) const GET_PRESERVED_ANALYSES: u32 = 1 << 4;
    pub(super) const GET_PASS_NAME: u32 = 1 << 5;
}

/// Refuse constructor arguments for a subclass of `cls` whose `__init__` is
/// `object.__init__`, as `object` does: a base's `__new__` accepts any
/// arguments so a subclass's own `__init__` can take them.
///
/// # Errors
///
/// Raises `TypeError` with Python's message, `X() takes no arguments`.
pub(crate) fn refuse_unused_arguments(
    cls: &Bound<'_, PyType>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    if args.is_empty() && kwargs.is_none_or(PyDictMethods::is_empty) {
        return Ok(());
    }
    let py = cls.py();
    let object_init = py.get_type::<PyAny>().getattr(intern!(py, "__init__"))?;
    if cls.getattr(intern!(py, "__init__"))?.is(&object_init) {
        return Err(PyTypeError::new_err(format!(
            "{}() takes no arguments",
            cls.name()?
        )));
    }
    Ok(())
}

/// Log a diagnostic on its source's pass logger, as `CompilerPass.report`
/// does: `fhy_core.pass_infrastructure.core._log_diagnostic`.
pub(super) fn log_diagnostic(
    py: Python<'_>,
    diagnostic: &Diagnostic,
    exc_info: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    crate::python::cached_attr!(py, CORE_MODULE, "_log_diagnostic" => PyAny)?.call1((
        diagnostic.source(),
        level_to_python(py, diagnostic.level())?,
        diagnostic.message_text(),
        diagnostic.detail(),
        exc_info,
    ))?;
    Ok(())
}

/// Return the logger of the pass `name` if it logs `DEBUG` lines, and
/// otherwise `None`: `fhy_core.pass_infrastructure.core._lifecycle_logger`.
fn lifecycle_logger(py: Python<'_>, name: &str) -> PyResult<Option<Py<PyAny>>> {
    let logger = crate::python::cached_attr!(py, CORE_MODULE, "_lifecycle_logger" => PyAny)?
        .call1((name,))?;
    Ok((!logger.is_none()).then(|| logger.unbind()))
}

/// The error a Python hook raised, as the hook's [`BoxError`].
///
/// It renders as the exception does, `ValueError: boom`, or as the chain
/// of a pass error the binding raised, which names Python hooks.
#[derive(Debug)]
pub(super) struct HookFailure {
    /// The Python hook that raised.
    pub(super) python_hook: &'static str,
    /// The exception.
    pub(super) error: PyErr,
    /// The rendering.
    pub(super) chain: String,
}

impl fmt::Display for HookFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.chain)
    }
}

impl Error for HookFailure {}

/// The last failure of a hook in a run: the Python hook, the rendering of
/// what it raised, and the exception.
pub(super) struct FailureNote {
    pub(super) python_hook: &'static str,
    pub(super) chain: String,
    pub(super) exception: Py<PyAny>,
}

/// Move the diagnostics a hook reported, `reported`, into `cx`, and record
/// them in the run's scope so they return to Python as themselves.
pub(super) fn report_into(py: Python<'_>, cx: &mut PassContext<'_>, reported: Vec<Py<PyAny>>) {
    if reported.is_empty() {
        return;
    }
    for object in &reported {
        if let Some(diagnostic) = borrow_python_diagnostic(object.bind(py)) {
            cx.report(diagnostic.clone());
        }
    }
    scope::record_reported(py, reported);
}

/// Return the error of a hook that did not run because an exception that
/// is not an `Exception` interrupted the run; the run's boundary raises
/// that exception instead.
pub(super) fn build_interrupted_failure(python_hook: &'static str) -> BoxError {
    Box::new(HookFailure {
        python_hook,
        error: PyRuntimeError::new_err("the run was interrupted"),
        chain: "the run was interrupted".to_owned(),
    })
}

/// The adapter that runs a Python `CompilerPass` as the core's
/// [`CompilerPass`].
pub(super) struct PythonPass {
    object: Py<PyCompilerPassBase>,
    /// The hooks with a default that the class overrides.
    hooks: u32,
    /// The pass name, read once when the adapter is built.
    name: String,
    /// The pass logger, when it logs `DEBUG` lines.
    logger: Option<Py<PyAny>>,
    /// Whether a pass error a hook raises nests in the run's error: in the
    /// lifecycle, not in a validation.
    nests: bool,
    /// Whether the current run skipped.
    is_skipped: bool,
    /// The last failure of a hook.
    failure: Option<FailureNote>,
}

impl PythonPass {
    /// Return the adapter of the Python pass `pass`.
    ///
    /// # Errors
    ///
    /// Raises what reading the pass's name raises, and `TypeError` for a
    /// name that is not a `str`.
    pub(super) fn new(pass: &Bound<'_, PyCompilerPassBase>, nests: bool) -> PyResult<Self> {
        let py = pass.py();
        let class = pass.get_type();
        let hooks = match class.getattr(intern!(py, "_python_hooks")) {
            Ok(hooks) => hooks.extract::<u32>()?,
            Err(_no_table) => u32::MAX,
        };
        let name = read_pass_name(&class, hooks)?;
        let logger = lifecycle_logger(py, &name)?;
        Ok(Self {
            object: pass.clone().unbind(),
            hooks,
            name,
            logger,
            nests,
            is_skipped: false,
            failure: None,
        })
    }

    /// Return whether the class overrides the hook of `bit`.
    fn overrides(&self, bit: u32) -> bool {
        self.hooks & bit != 0
    }

    /// Return the pass name.
    pub(super) fn pass_name(&self) -> &str {
        &self.name
    }

    /// Return the pass object.
    pub(super) fn object<'py>(&self, py: Python<'py>) -> &Bound<'py, PyCompilerPassBase> {
        self.object.bind(py)
    }

    /// Return the last failure of a hook, removing it.
    pub(super) fn take_failure(&mut self) -> Option<FailureNote> {
        self.failure.take()
    }

    /// Log `message` with `arguments` at `DEBUG` on the pass logger, if it
    /// logs them.
    fn log_debug<'py>(
        &self,
        py: Python<'py>,
        message: &str,
        arguments: impl IntoIterator<Item = Bound<'py, PyAny>>,
    ) -> PyResult<()> {
        let Some(logger) = &self.logger else {
            return Ok(());
        };
        let mut items = vec![PyString::new(py, message).into_any()];
        items.extend(arguments);
        logger
            .bind(py)
            .call_method1(intern!(py, "debug"), PyTuple::new(py, items)?)?;
        Ok(())
    }

    /// Begin a run over `ir`: clear the pass's diagnostics and log the
    /// entering line.
    fn start_run(&mut self, py: Python<'_>, ir: &PyIr) -> PyResult<()> {
        self.is_skipped = false;
        self.failure = None;
        self.object(py).get().clear_diagnostics();
        if self.logger.is_some() {
            let input_type = ir.bind(py).get_type().name()?.into_any();
            self.log_debug(py, "entering (input type=%s)", [input_type])?;
        }
        Ok(())
    }

    /// Return the failure of `hook` raising `error`, recording it in the
    /// run's scope.
    fn fail(&mut self, py: Python<'_>, hook: &'static str, error: PyErr) -> BoxError {
        let exception = error.value(py).clone().into_any().unbind();
        if !error.is_instance_of::<PyException>(py) {
            scope::record_interrupt(error.clone_ref(py));
            let chain = error.to_string();
            self.failure = Some(FailureNote {
                python_hook: hook,
                chain: chain.clone(),
                exception,
            });
            return Box::new(HookFailure {
                python_hook: hook,
                error,
                chain,
            });
        }
        scope::record_failed_pass(self.object.clone_ref(py).into_any());
        if self.nests {
            if let Some((inner, chain)) = take_nested(py, &error) {
                self.failure = Some(FailureNote {
                    python_hook: hook,
                    chain: chain.clone(),
                    exception: exception.clone_ref(py),
                });
                scope::record_nested(NestedNote {
                    python_hook: hook,
                    exception,
                    chain,
                });
                return Box::new(inner);
            }
        }
        let chain = chain_of_exception(py, &error);
        self.failure = Some(FailureNote {
            python_hook: hook,
            chain: chain.clone(),
            exception,
        });
        Box::new(HookFailure {
            python_hook: hook,
            error,
            chain,
        })
    }

    /// Call the Python hook `hook` with `call`, in a frame with `cx`'s
    /// analyses when the core gives the hook a context, and without one
    /// otherwise.
    fn call_hook<'py, R>(
        &mut self,
        py: Python<'py>,
        hook: &'static str,
        cx: Option<&mut PassContext<'_>>,
        call: impl FnOnce(&Bound<'py, PyAny>) -> PyResult<R>,
    ) -> Result<R, BoxError> {
        if scope::is_interrupted() {
            return Err(build_interrupted_failure(hook));
        }
        let object = self.object.bind(py).clone().into_any();
        let result = if let Some(cx) = cx {
            let (result, reported) = cx.with_detached_analyses(|analyses| {
                let frame = FrameGuard::push(&object, hook, Some(analyses.clone()), true);
                let result = call(&object);
                (result, frame.finish())
            });
            report_into(py, cx, reported);
            result
        } else {
            let frame = FrameGuard::push(&object, hook, None, false);
            let result = call(&object);
            drop(frame.finish());
            result
        };
        result.map_err(|error| self.fail(py, hook, error))
    }
}

/// Return the name of a pass of `class`: its `get_pass_name()` if the class
/// overrides it, and otherwise its registered name or its `__name__`.
fn read_pass_name(class: &Bound<'_, PyType>, hooks: u32) -> PyResult<String> {
    let py = class.py();
    let name = if hooks & hook_bit::GET_PASS_NAME != 0 {
        class.call_method0(intern!(py, "get_pass_name"))?
    } else {
        let registered = class.getattr(intern!(py, "_pass_name"))?;
        if registered.is_truthy()? {
            registered
        } else {
            class.getattr(intern!(py, "__name__"))?
        }
    };
    match name.cast::<PyString>() {
        Ok(name) => Ok(name.to_str()?.to_owned()),
        Err(_not_a_str) => Err(PyTypeError::new_err(format!(
            "{}.get_pass_name must return a str, got {}.",
            class.qualname()?,
            name.get_type().name()?
        ))),
    }
}

/// Return the core's set of the hook result `result`, which must be a
/// `PreservedAnalyses`.
fn read_preserved(
    result: &Bound<'_, PyAny>,
    pass: &Bound<'_, PyAny>,
) -> PyResult<PreservedAnalyses> {
    match result.cast::<PyPreservedAnalyses>() {
        Ok(preserved) => Ok(preserved.get().preserved().clone()),
        Err(_not_a_set) => Err(PyTypeError::new_err(format!(
            "{}.get_preserved_analyses must return a PreservedAnalyses, got {}.",
            pass.get_type().qualname()?,
            result.get_type().name()?
        ))),
    }
}

impl CompilerPass<PyIr> for PythonPass {
    fn name(&self) -> Cow<'static, str> {
        Cow::Owned(self.name.clone())
    }

    fn validate_input(&mut self, ir: &PyIr, cx: &mut PassContext<'_>) -> Result<(), BoxError> {
        Python::attach(|py| {
            if let Err(error) = self.start_run(py, ir) {
                return Err(self.fail(py, "validate_input", error));
            }
            if !self.overrides(hook_bit::VALIDATE_INPUT) {
                return Ok(());
            }
            self.call_hook(py, "validate_input", Some(cx), |pass| {
                pass.call_method1(intern!(py, "validate_input"), (ir.bind(py),))
                    .map(drop)
            })
        })
    }

    fn skip(&mut self, ir: &PyIr, cx: &mut PassContext<'_>) -> Result<Option<PyIr>, BoxError> {
        if !self.overrides(hook_bit::SHOULD_RUN) {
            return Ok(None);
        }
        Python::attach(|py| {
            let should_run = self.call_hook(py, "should_run", Some(&mut *cx), |pass| {
                pass.call_method1(intern!(py, "should_run"), (ir.bind(py),))?
                    .is_truthy()
            })?;
            if should_run {
                return Ok(None);
            }
            if let Err(error) = self.log_debug(py, "skipped: should_run returned False", []) {
                return Err(self.fail(py, "should_run", error));
            }
            let output = self.call_hook(py, "get_noop_output", Some(cx), |pass| {
                pass.call_method1(intern!(py, "get_noop_output"), (ir.bind(py),))
            })?;
            self.is_skipped = true;
            Ok(Some(PyIr::new(&output)))
        })
    }

    fn run(&mut self, ir: &PyIr, cx: &mut PassContext<'_>) -> Result<PyIr, BoxError> {
        Python::attach(|py| {
            let output = self.call_hook(py, "run_pass", Some(cx), |pass| {
                pass.call_method1(intern!(py, "run_pass"), (ir.bind(py),))
            })?;
            Ok(PyIr::new(&output))
        })
    }

    fn validate_output(
        &mut self,
        input: &PyIr,
        output: &PyIr,
        cx: &mut PassContext<'_>,
    ) -> Result<(), BoxError> {
        if !self.overrides(hook_bit::VALIDATE_OUTPUT) {
            return Ok(());
        }
        Python::attach(|py| {
            self.call_hook(py, "validate_output", Some(cx), |pass| {
                pass.call_method1(
                    intern!(py, "validate_output"),
                    (input.bind(py), output.bind(py)),
                )
                .map(drop)
            })
        })
    }

    fn did_change(&mut self, input: &PyIr, output: &PyIr) -> Result<bool, BoxError> {
        Python::attach(|py| {
            if self.overrides(hook_bit::DID_CHANGE) {
                return self.call_hook(py, "did_change", None, |pass| {
                    pass.call_method1(intern!(py, "did_change"), (input.bind(py), output.bind(py)))?
                        .is_truthy()
                });
            }
            let (input, output) = (input.bind(py), output.bind(py));
            Ok(input.ne(output).unwrap_or_else(|_| !input.is(output)))
        })
    }

    fn preserved_analyses(
        &mut self,
        input: &PyIr,
        output: &PyIr,
        changed: bool,
    ) -> Result<PreservedAnalyses, BoxError> {
        Python::attach(|py| {
            let preserved = if self.overrides(hook_bit::GET_PRESERVED_ANALYSES) {
                self.call_hook(py, "get_preserved_analyses", None, |pass| {
                    let keywords = PyDict::new(py);
                    keywords.set_item(intern!(py, "changed"), changed)?;
                    let result = pass.call_method(
                        intern!(py, "get_preserved_analyses"),
                        (input.bind(py), output.bind(py)),
                        Some(&keywords),
                    )?;
                    read_preserved(&result, pass)
                })?
            } else if changed {
                PreservedAnalyses::none()
            } else {
                PreservedAnalyses::all()
            };
            if !self.is_skipped && self.logger.is_some() {
                let logged = (|| -> PyResult<()> {
                    let output_type = output.bind(py).get_type().name()?.into_any();
                    let count = self.object(py).get().diagnostic_count();
                    self.log_debug(
                        py,
                        "finished (changed=%s, output type=%s, diagnostics=%d)",
                        [
                            pyo3::types::PyBool::new(py, changed).to_owned().into_any(),
                            output_type,
                            count.into_pyobject(py)?.into_any(),
                        ],
                    )
                })();
                if let Err(error) = logged {
                    return Err(self.fail(py, "get_preserved_analyses", error));
                }
            }
            Ok(preserved)
        })
    }
}

/// The failure diagnostic the binding reports for `pass`, whose hook
/// failed as `failure` says, in a validation: the core's text, naming the
/// Python hook.
pub(super) fn report_check_failure(
    py: Python<'_>,
    pass: &PythonPass,
    failure: &FailureNote,
    cx: &mut PassContext<'_>,
) -> PyResult<()> {
    let text = format!(
        "pass {:?} failed in {}: {}",
        pass.pass_name(),
        failure.python_hook,
        failure.chain
    );
    let diagnostic = Diagnostic::error(Note::with_other_kind(text), pass.pass_name().to_owned());
    let object = diagnostic_to_python(py, &diagnostic)?;
    log_diagnostic(py, &diagnostic, Some(failure.exception.bind(py)))?;
    pass.object(py)
        .get()
        .push_diagnostic(object.clone().unbind());
    cx.report(diagnostic);
    scope::record_reported(py, vec![object.unbind()]);
    Ok(())
}

/// The base of the Python `CompilerPass` ABC.
///
/// It keeps the diagnostics of the pass's current or last run, and of
/// `report` calls outside a run, and drives the pass through the core's
/// lifecycle in `execute`.
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "CompilerPassBase")]
pub(crate) struct PyCompilerPassBase {
    diagnostics: Mutex<Vec<Py<PyAny>>>,
}

impl PyCompilerPassBase {
    fn lock(&self) -> MutexGuard<'_, Vec<Py<PyAny>>> {
        self.diagnostics
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// Forget the diagnostics, as a new run does.
    pub(super) fn clear_diagnostics(&self) {
        let forgotten = mem::take(&mut *self.lock());
        drop(forgotten);
    }

    /// Append `diagnostic`, a `Diagnostic` object.
    pub(super) fn push_diagnostic(&self, diagnostic: Py<PyAny>) {
        self.lock().push(diagnostic);
    }

    /// Replace the diagnostics with the objects of `diagnostics`.
    pub(super) fn set_diagnostics(&self, diagnostics: &Bound<'_, PyTuple>) {
        let replacement: Vec<Py<PyAny>> = diagnostics.iter().map(Bound::unbind).collect();
        let replaced = mem::replace(&mut *self.lock(), replacement);
        drop(replaced);
    }

    /// Return the number of diagnostics.
    fn diagnostic_count(&self) -> usize {
        self.lock().len()
    }

    /// Run the pass over `ir` through the core's lifecycle, standalone, and
    /// return the outcome's parts with the run's scope, or the Python error
    /// of a failed run.
    fn run_standalone<'py>(
        slf: &Bound<'py, Self>,
        ir: &Bound<'py, PyAny>,
    ) -> PyResult<(fhy_core::pass::PassOutcome<PyIr>, super::scope::RunScope)> {
        let py = slf.py();
        let guard = ScopeGuard::enter();
        let mut adapter = PythonPass::new(slf, true)?;
        let result = ExecutePass::execute(&mut adapter, &PyIr::new(ir));
        drop(adapter);
        let mut scope = guard.finish();
        if let Some(interrupt) = scope.take_interrupt() {
            return Err(interrupt);
        }
        match result {
            Ok(outcome) => Ok((outcome, scope)),
            Err(error) => Err(error_to_python(py, error, &mut scope, &[])?),
        }
    }
}

#[pymethods]
impl PyCompilerPassBase {
    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        crate::gc::traverse_locked(&self.diagnostics, |diagnostics| {
            crate::gc::traverse_all(&visit, diagnostics)
        })
    }

    /// Drop what only this object holds, for the cycle collector.
    fn __clear__(&self) {
        crate::gc::clear_locked(&self.diagnostics);
    }

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
        Ok(Self {
            diagnostics: Mutex::new(Vec::new()),
        })
    }

    /// The diagnostics of the current run during a hook; afterwards those of
    /// the last run, a failed one included; and those reported outside a
    /// run since.
    #[getter]
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let objects: Vec<Bound<'py, PyAny>> = self
            .lock()
            .iter()
            .map(|object| object.bind(py).clone())
            .collect();
        PyTuple::new(py, objects)
    }

    /// Record `diagnostic`, which `report` built: into the running hook's
    /// context during a hook, and on the pass.
    ///
    /// Raises `TypeError` if `diagnostic` is not a `Diagnostic`, and
    /// `RuntimeError` in `did_change` or `get_preserved_analyses`, which run
    /// without a context.
    fn _record_diagnostic(slf: &Bound<'_, Self>, diagnostic: &Bound<'_, PyAny>) -> PyResult<()> {
        match context::record_diagnostic(slf.as_any(), diagnostic)? {
            Recording::InHook | Recording::Standalone => {
                slf.get().push_diagnostic(diagnostic.clone().unbind());
            }
        }
        Ok(())
    }

    /// Run the pass over `ir` through its lifecycle and return the
    /// `PassResult`.
    ///
    /// Raises `PassValidationError` or `PassExecutionError` if a hook
    /// fails, with the core's message naming the Python hook, and the
    /// hook's exception as the `__cause__`; an exception that is not an
    /// `Exception` propagates unchanged.
    fn execute<'py>(slf: &Bound<'py, Self>, ir: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let (outcome, mut scope) = Self::run_standalone(slf, ir)?;
        let diagnostics = diagnostics_to_python(py, outcome.diagnostics(), &mut scope)?;
        let preserved = preserved_to_python(py, outcome.preserved_analyses())?;
        let (changed, skipped) = (outcome.is_changed(), outcome.is_skipped());
        let output = outcome.into_output().into_inner().into_bound(py);
        PyPassResult::build(py, output, changed, diagnostics, preserved, skipped)
    }

    /// Run the pass over `ir` and return only its output.
    fn __call__<'py>(
        slf: &Bound<'py, Self>,
        ir: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (outcome, _scope) = Self::run_standalone(slf, ir)?;
        Ok(outcome.into_output().into_inner().into_bound(slf.py()))
    }

    /// Return the result of the analysis `analysis_type` for `ir`: through
    /// the run's cache during a hook in a pipeline, and computed afresh
    /// otherwise.
    ///
    /// Raises `TypeError` for an `analysis_type` that is not an `Analysis`
    /// subclass, and `RuntimeError` in `did_change` or
    /// `get_preserved_analyses`.
    fn get_analysis<'py>(
        slf: &Bound<'py, Self>,
        analysis_type: &Bound<'py, PyAny>,
        ir: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        context::get_analysis(slf.as_any(), analysis_type, ir)
    }

    /// Return the `AnalysisManager` view of the run's analyses during a
    /// hook, or `None` outside one.
    ///
    /// Raises `RuntimeError` in `did_change` or `get_preserved_analyses`.
    fn get_analysis_manager<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<Option<Bound<'py, super::context::PyAnalysisManager>>> {
        context::get_analysis_manager(slf.as_any())
    }

    /// Return no constructor arguments, so a pass pickles through its
    /// `__dict__`.
    fn __getnewargs__<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyTuple> {
        PyTuple::empty(slf.py())
    }
}
