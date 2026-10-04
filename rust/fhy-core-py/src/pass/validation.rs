//! `fhy_core._rs.ValidatorBase`, the base of the Python `Validator` ABC,
//! `fhy_core._rs.ValidationManager`, and the validators the binding runs:
//! Python validators and Python passes as checks. The registry verifier of
//! a pipeline is in `verification.rs`.
//!
//! A pass runs as a check as the core's `PassValidator` runs one:
//! `validate_input`, `should_run` then `get_noop_output`, `run_pass`, and
//! `validate_output`. A hook that fails fails the check after the error
//! diagnostic recording it, `pass "X" failed in run_pass: ValueError:
//! boom`, which the binding writes with the Python hook's name.

use std::borrow::Cow;
use std::sync::{Mutex, MutexGuard, PoisonError};

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::types::{PyDict, PyString, PyTuple, PyType};

use fhy_core::diagnostic::{DiagnosticLevel, Note};
use fhy_core::foreign::BoxError;
use fhy_core::identifier::Identifier;
use fhy_core::pass::{CompilerPass, PassContext, ValidationManager, Validator};

use crate::identifier::{new_python_identifier, read_identifier_id, restore_identifier};
use crate::kit::dataclass::build_argument_type_error;

use super::compiler_pass::{
    HookFailure, PyCompilerPassBase, PythonPass, build_interrupted_failure, log_diagnostic,
    refuse_unused_arguments, report_check_failure, report_into,
};
use super::context::{self, FrameGuard, PyAnalysisManager};
use super::convert::validation_report_to_python;
use super::error::chain_of_exception;
use super::ir::PyIr;
use super::scope::{self, ScopeGuard};

/// Run `pass` over `ir` as a check, reporting into `cx`, and report the
/// failure diagnostic of a hook that fails.
pub(super) fn run_check(
    py: Python<'_>,
    pass: &mut PythonPass,
    ir: &PyIr,
    cx: &mut PassContext<'_>,
) -> Result<(), BoxError> {
    let result = (|| -> Result<(), BoxError> {
        pass.validate_input(ir, cx)?;
        if pass.skip(ir, cx)?.is_some() {
            return Ok(());
        }
        let output = pass.run(ir, cx)?;
        pass.validate_output(ir, &output, cx)
    })();
    if result.is_err() {
        if let Some(failure) = pass.take_failure() {
            report_check_failure(py, pass, &failure, cx)
                .map_err(|error| Box::new(error) as BoxError)?;
        }
    }
    result
}

/// A Python pass run as a check.
struct PassCheck {
    pass: PythonPass,
}

impl Validator<PyIr> for PassCheck {
    fn name(&self) -> Cow<'static, str> {
        CompilerPass::name(&self.pass)
    }

    fn validate(&mut self, ir: &PyIr, cx: &mut PassContext<'_>) -> Result<(), BoxError> {
        Python::attach(|py| run_check(py, &mut self.pass, ir, cx))
    }
}

/// The adapter that runs a Python `Validator` as the core's [`Validator`].
struct PythonValidator {
    object: Py<PyAny>,
    /// The validator's name, read once when the adapter is built.
    name: String,
}

impl PythonValidator {
    /// Return the adapter of the Python validator `validator`.
    ///
    /// # Errors
    ///
    /// Raises what reading its `name` raises, and `TypeError` for a name
    /// that is not a `str`.
    fn new(validator: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = validator.py();
        let name = validator.getattr(intern!(py, "name"))?;
        let Ok(name) = name.cast::<PyString>() else {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "{}.name must be a str, got {}.",
                validator.get_type().qualname()?,
                name.get_type().name()?
            )));
        };
        Ok(Self {
            object: validator.clone().unbind(),
            name: name.to_str()?.to_owned(),
        })
    }
}

/// Return the failure of the validator `name` that raised `error`, logging
/// the error the core synthesizes when the validator reported none into
/// `cx`, and recording an exception that is not an `Exception` so the run's
/// boundary raises it.
pub(super) fn fail_validator(
    py: Python<'_>,
    name: &str,
    error: PyErr,
    cx: &PassContext<'_>,
) -> BoxError {
    if !error.is_instance_of::<pyo3::exceptions::PyException>(py) {
        scope::record_interrupt(error.clone_ref(py));
    }
    let chain = chain_of_exception(py, &error);
    let reported_error = cx
        .diagnostics()
        .iter()
        .any(|diagnostic| diagnostic.level() == DiagnosticLevel::Error);
    if !reported_error {
        let text = format!("validator {name:?} failed without reporting an error: {chain}");
        let diagnostic =
            fhy_core::diagnostic::Diagnostic::error(Note::with_other_kind(text), name.to_owned());
        let exception = error.value(py).clone().into_any();
        if let Err(logging_error) = log_diagnostic(py, &diagnostic, Some(&exception)) {
            logging_error.write_unraisable(py, None);
        }
    }
    Box::new(HookFailure {
        python_hook: "validate",
        error,
        chain,
    })
}

impl Validator<PyIr> for PythonValidator {
    fn name(&self) -> Cow<'static, str> {
        Cow::Owned(self.name.clone())
    }

    fn validate(&mut self, ir: &PyIr, cx: &mut PassContext<'_>) -> Result<(), BoxError> {
        Python::attach(|py| {
            if scope::is_interrupted() {
                return Err(build_interrupted_failure("validate"));
            }
            let object = self.object.bind(py).clone();
            let (result, reported) = cx.with_detached_analyses(|analyses| {
                let frame = FrameGuard::push(&object, "validate", Some(analyses.clone()), true);
                let result = object.call_method1(intern!(py, "validate"), (ir.bind(py),));
                (result, frame.finish())
            });
            report_into(py, cx, reported);
            match result {
                Ok(_) => Ok(()),
                Err(error) => Err(fail_validator(py, &self.name, error, cx)),
            }
        })
    }
}

/// Return the core's validation pipeline named `name` over `validators`,
/// Python `Validator`s and passes.
fn build_manager(
    py: Python<'_>,
    name: Identifier,
    validators: &[Py<PyAny>],
) -> PyResult<ValidationManager<'static, PyIr>> {
    let mut manager = ValidationManager::new(name);
    for validator in validators {
        let validator = validator.bind(py);
        if let Ok(pass) = validator.cast::<PyCompilerPassBase>() {
            manager.add(PassCheck {
                pass: PythonPass::new(pass, false)?,
            });
        } else {
            manager.add(PythonValidator::new(validator)?);
        }
    }
    Ok(manager)
}

/// The base of the Python `Validator` ABC.
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "ValidatorBase")]
pub(crate) struct PyValidatorBase;

#[pymethods]
impl PyValidatorBase {
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

    /// Record `diagnostic`, which `report` built, into the running check's
    /// context; outside a check it is only logged.
    ///
    /// Raises `TypeError` if `diagnostic` is not a `Diagnostic`.
    fn _record_diagnostic(slf: &Bound<'_, Self>, diagnostic: &Bound<'_, PyAny>) -> PyResult<()> {
        context::record_diagnostic(slf.as_any(), diagnostic).map(drop)
    }

    /// Return the result of the analysis `analysis_type` for `ir`, as a
    /// pass's `get_analysis` does.
    fn get_analysis<'py>(
        slf: &Bound<'py, Self>,
        analysis_type: &Bound<'py, PyAny>,
        ir: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        context::get_analysis(slf.as_any(), analysis_type, ir)
    }

    /// Return the `AnalysisManager` view of the run's analyses during a
    /// check, or `None` outside one.
    fn get_analysis_manager<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<Option<Bound<'py, PyAnalysisManager>>> {
        context::get_analysis_manager(slf.as_any())
    }

    /// Return no constructor arguments, so a validator pickles through its
    /// `__dict__`.
    fn __getnewargs__<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyTuple> {
        PyTuple::empty(slf.py())
    }
}

/// Return `name` as a pipeline name: an `Identifier`, or a new one named
/// `default` for `None`.
pub(super) fn read_pipeline_name<'py>(
    py: Python<'py>,
    name: Option<&Bound<'py, PyAny>>,
    owner: &str,
    default: &str,
) -> PyResult<Py<PyAny>> {
    match name.filter(|name| !name.is_none()) {
        Some(name) if read_identifier_id(name)?.is_some() => Ok(name.clone().unbind()),
        Some(name) => Err(build_argument_type_error(
            owner,
            "name",
            "an Identifier",
            name,
        )?),
        None => Ok(new_python_identifier(py, default)?.unbind()),
    }
}

/// A sequence of validators whose diagnostics aggregate into one report.
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "ValidationManager")]
pub(crate) struct PyValidationManager {
    name: Py<PyAny>,
    validators: Mutex<Vec<Py<PyAny>>>,
}

impl PyValidationManager {
    fn lock(&self) -> MutexGuard<'_, Vec<Py<PyAny>>> {
        self.validators
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// Return the core's pipeline of the current validators.
    pub(super) fn build(&self, py: Python<'_>) -> PyResult<ValidationManager<'static, PyIr>> {
        let validators: Vec<Py<PyAny>> = self.lock().iter().map(|v| v.clone_ref(py)).collect();
        let name = restore_identifier(self.name.bind(py), "ValidationManager", "name")?;
        build_manager(py, name, &validators)
    }
}

#[pymethods]
impl PyValidationManager {
    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.name)?;
        crate::kit::gc::traverse_locked(&self.validators, |validators| {
            crate::kit::gc::traverse_all(&visit, validators)
        })
    }

    /// Drop what only this object holds, for the cycle collector.
    fn __clear__(&self) {
        crate::kit::gc::clear_locked(&self.validators);
    }

    /// Create the empty validation pipeline `name`, by default an
    /// identifier named `validation-pipeline`.
    ///
    /// Raises `TypeError` if `name` is not an `Identifier`.
    #[new]
    #[pyo3(signature = (name = None))]
    fn new(py: Python<'_>, name: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        Ok(Self {
            name: read_pipeline_name(py, name, "ValidationManager", "validation-pipeline")?,
            validators: Mutex::new(Vec::new()),
        })
    }

    /// The pipeline's name.
    #[getter]
    fn name(&self, py: Python<'_>) -> Py<PyAny> {
        self.name.clone_ref(py)
    }

    /// Return the pipeline's name.
    fn get_identifier(&self, py: Python<'_>) -> Py<PyAny> {
        self.name.clone_ref(py)
    }

    /// The validators, in pipeline order.
    #[getter]
    fn validators<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let validators: Vec<Bound<'py, PyAny>> =
            self.lock().iter().map(|v| v.bind(py).clone()).collect();
        PyTuple::new(py, validators)
    }

    /// Append `validator`, a `Validator` or a `CompilerPass`.
    ///
    /// Raises `TypeError` for anything else.
    fn add(&self, validator: &Bound<'_, PyAny>) -> PyResult<()> {
        if !validator.is_instance_of::<PyCompilerPassBase>()
            && !validator.is_instance_of::<PyValidatorBase>()
        {
            return Err(build_argument_type_error(
                "ValidationManager.add",
                "validator",
                "a Validator or a CompilerPass",
                validator,
            )?);
        }
        self.lock().push(validator.clone().unbind());
        Ok(())
    }

    /// Run every validator over `ir` and return the aggregated
    /// `ValidationReport`, with one `ValidatorRecord` per validator.
    ///
    /// Every validator runs, whatever the earlier ones reported. A
    /// validator that fails keeps its diagnostics; a failing pass adds the
    /// error diagnostic of its hook's failure, and a failing `Validator`
    /// that reported no error gains `validator "X" failed without reporting
    /// an error: <cause>`. An exception that is not an `Exception`
    /// propagates unchanged.
    fn validate<'py>(
        &self,
        py: Python<'py>,
        ir: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let guard = ScopeGuard::enter();
        let mut manager = self.build(py)?;
        let report = manager.validate(&PyIr::new(ir));
        drop(manager);
        let mut scope = guard.finish();
        if let Some(interrupt) = scope.take_interrupt() {
            return Err(interrupt);
        }
        validation_report_to_python(py, &report, &mut scope)
    }

    fn __reduce__(_slf: &Bound<'_, Self>) -> PyResult<()> {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "a ValidationManager cannot be pickled",
        ))
    }
}
