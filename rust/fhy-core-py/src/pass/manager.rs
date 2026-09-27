//! `fhy_core._rs.PassManager` and `FixpointPassGroup` (P2; D-S6-11,
//! D-S6-12): pipelines over Python item lists.
//!
//! A pipeline keeps the Python passes and groups it was given. Each run
//! builds the core's `PassManager` over [`PyIr`] from the current items,
//! with an adapter per pass, and runs it with its own analysis cache, so a
//! group changed after it was added behaves as it is now, and two runs of
//! one pipeline are independent. By default the run verifies its input and
//! every changed output with the passes the verification registry holds for
//! the IR's type; `set_verifier` replaces that verifier or turns it off.

use std::num::NonZeroUsize;
use std::sync::{Mutex, MutexGuard, PoisonError};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::types::{PyBool, PyInt, PyTuple};

use fhy_core::pass::{FixpointPassGroup, PassManager};

use crate::dataclass::build_argument_type_error;
use crate::identifier::{read_identifier_id, restore_identifier};

use super::compiler_pass::{PyCompilerPassBase, PythonPass};
use super::convert::records_to_python;
use super::error::error_to_python;
use super::ir::PyIr;
use super::records::PyPassManagerResult;
use super::scope::ScopeGuard;
use super::validation::{PyValidationManager, read_pipeline_name};
use super::verification::build_registry_verifier;

/// Lock `mutex`, ignoring poisoning: a panic cannot leave a list of Python
/// objects inconsistent.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Return `pass` as a pass, or raise the `TypeError` naming `owner`.
fn read_pass<'a, 'py>(
    pass: &'a Bound<'py, PyAny>,
    owner: &str,
) -> PyResult<&'a Bound<'py, PyCompilerPassBase>> {
    match pass.cast::<PyCompilerPassBase>() {
        Ok(pass) => Ok(pass),
        Err(_not_a_pass) => Err(build_argument_type_error(
            owner,
            "compiler_pass",
            "a CompilerPass",
            pass,
        )?),
    }
}

/// A pass sequence a pipeline repeats until no pass changes the IR.
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "FixpointPassGroup")]
pub(crate) struct PyFixpointPassGroup {
    name: Py<PyAny>,
    max_iterations: NonZeroUsize,
    fail_on_non_convergence: bool,
    passes: Mutex<Vec<Py<PyCompilerPassBase>>>,
}

#[pymethods]
impl PyFixpointPassGroup {
    /// Visit the Python objects the object holds, for the cycle collector (R2-003).
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.name)?;
        crate::gc::traverse_locked(&self.passes, |passes| {
            crate::gc::traverse_all(&visit, passes)
        })
    }

    /// Drop what only this object holds, for the cycle collector.
    fn __clear__(&self) {
        crate::gc::clear_locked(&self.passes);
    }

    /// Create the empty group `name`, an `Identifier`, with a budget of
    /// `max_iterations` iterations that fails a pipeline when it does not
    /// converge if `fail_on_non_convergence` holds.
    ///
    /// Raises `ValueError` if `max_iterations` is below 1, and `TypeError`
    /// for an argument of the wrong type.
    #[new]
    #[pyo3(signature = (name, *, max_iterations = None, fail_on_non_convergence = None))]
    fn new(
        name: &Bound<'_, PyAny>,
        max_iterations: Option<&Bound<'_, PyAny>>,
        fail_on_non_convergence: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if read_identifier_id(name)?.is_none() {
            return Err(build_argument_type_error(
                "FixpointPassGroup",
                "name",
                "an Identifier",
                name,
            )?);
        }
        let max_iterations = match max_iterations {
            None => NonZeroUsize::new(10).unwrap_or(NonZeroUsize::MIN),
            Some(value) => {
                if !value.is_instance_of::<PyInt>() || value.is_instance_of::<PyBool>() {
                    return Err(build_argument_type_error(
                        "FixpointPassGroup",
                        "max_iterations",
                        "an int",
                        value,
                    )?);
                }
                let value: i128 = value.extract()?;
                usize::try_from(value)
                    .ok()
                    .and_then(NonZeroUsize::new)
                    .ok_or_else(|| PyValueError::new_err("\"max_iterations\" must be >= 1."))?
            }
        };
        let fail_on_non_convergence = match fail_on_non_convergence {
            None => true,
            Some(value) => match value.cast::<PyBool>() {
                Ok(value) => value.is_true(),
                Err(_not_a_bool) => {
                    return Err(build_argument_type_error(
                        "FixpointPassGroup",
                        "fail_on_non_convergence",
                        "a bool",
                        value,
                    )?);
                }
            },
        };
        Ok(Self {
            name: name.clone().unbind(),
            max_iterations,
            fail_on_non_convergence,
            passes: Mutex::new(Vec::new()),
        })
    }

    /// The group's name.
    #[getter]
    fn name(&self, py: Python<'_>) -> Py<PyAny> {
        self.name.clone_ref(py)
    }

    /// Return the group's name.
    fn get_identifier(&self, py: Python<'_>) -> Py<PyAny> {
        self.name.clone_ref(py)
    }

    /// The iteration budget.
    #[getter]
    fn max_iterations(&self) -> usize {
        self.max_iterations.get()
    }

    /// Whether the group fails a pipeline when it does not converge.
    #[getter]
    fn fail_on_non_convergence(&self) -> bool {
        self.fail_on_non_convergence
    }

    /// The group's passes, in order.
    #[getter]
    fn passes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let passes: Vec<Bound<'py, PyAny>> = lock(&self.passes)
            .iter()
            .map(|pass| pass.bind(py).clone().into_any())
            .collect();
        PyTuple::new(py, passes)
    }

    /// Append `compiler_pass`, a `CompilerPass`, to the group.
    ///
    /// Raises `TypeError` for anything else.
    fn add_pass(&self, compiler_pass: &Bound<'_, PyAny>) -> PyResult<()> {
        let compiler_pass = read_pass(compiler_pass, "FixpointPassGroup.add_pass")?;
        lock(&self.passes).push(compiler_pass.clone().unbind());
        Ok(())
    }

    fn __reduce__(_slf: &Bound<'_, Self>) -> PyResult<()> {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "a FixpointPassGroup cannot be pickled",
        ))
    }
}

/// The core's pipeline of a run, and the Python group name of each item,
/// `None` for a pass.
type BuiltPipeline = (PassManager<'static, PyIr>, Vec<Option<Py<PyAny>>>);

/// One item of a Python pipeline.
enum Item {
    Pass(Py<PyCompilerPassBase>),
    FixpointGroup(Py<PyFixpointPassGroup>),
}

impl Item {
    fn clone_ref(&self, py: Python<'_>) -> Self {
        match self {
            Self::Pass(pass) => Self::Pass(pass.clone_ref(py)),
            Self::FixpointGroup(group) => Self::FixpointGroup(group.clone_ref(py)),
        }
    }
}

/// How a pipeline verifies its IR.
enum Verifier {
    /// With the verification passes registered for the IR's type.
    Registry,
    /// Not at all.
    Off,
    /// With this validation pipeline.
    Manager(Py<PyValidationManager>),
}

impl Verifier {
    fn clone_ref(&self, py: Python<'_>) -> Self {
        match self {
            Self::Registry => Self::Registry,
            Self::Off => Self::Off,
            Self::Manager(manager) => Self::Manager(manager.clone_ref(py)),
        }
    }
}

/// An ordered pipeline of passes and fixpoint groups.
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "PassManager")]
pub(crate) struct PyPassManager {
    name: Py<PyAny>,
    items: Mutex<Vec<Item>>,
    verifier: Mutex<Verifier>,
}

impl PyPassManager {
    /// Return the core's pipeline of the current items, with an adapter per
    /// pass, and the Python name of each item's group, `None` for a pass.
    fn build(&self, py: Python<'_>) -> PyResult<BuiltPipeline> {
        let items: Vec<Item> = lock(&self.items)
            .iter()
            .map(|item| item.clone_ref(py))
            .collect();
        let verifier = lock(&self.verifier).clone_ref(py);
        let mut pipeline = PassManager::new(restore_identifier(
            self.name.bind(py),
            "PassManager",
            "name",
        )?);
        let mut group_names = Vec::with_capacity(items.len());
        for item in &items {
            match item {
                Item::Pass(pass) => {
                    pipeline.add_pass(PythonPass::new(pass.bind(py), true)?);
                    group_names.push(None);
                }
                Item::FixpointGroup(group) => {
                    let group = group.get();
                    let passes: Vec<Py<PyCompilerPassBase>> = lock(&group.passes)
                        .iter()
                        .map(|p| p.clone_ref(py))
                        .collect();
                    let mut core_group = FixpointPassGroup::new(restore_identifier(
                        group.name.bind(py),
                        "FixpointPassGroup",
                        "name",
                    )?)
                    .with_max_iterations(group.max_iterations)
                    .with_fail_on_non_convergence(group.fail_on_non_convergence);
                    for pass in &passes {
                        core_group.add_pass(PythonPass::new(pass.bind(py), true)?);
                    }
                    pipeline.add_fixpoint_group(core_group);
                    group_names.push(Some(group.name.clone_ref(py)));
                }
            }
        }
        match verifier {
            Verifier::Registry => pipeline.set_verifier(build_registry_verifier()),
            Verifier::Off => {}
            Verifier::Manager(manager) => pipeline.set_verifier(manager.get().build(py)?),
        }
        Ok((pipeline, group_names))
    }
}

#[pymethods]
impl PyPassManager {
    /// Visit the Python objects the object holds, for the cycle collector (R2-003).
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.name)?;
        crate::gc::traverse_locked(&self.items, |items| {
            items.iter().try_for_each(|item| match item {
                Item::Pass(compiler_pass) => visit.call(compiler_pass),
                Item::FixpointGroup(group) => visit.call(group),
            })
        })?;
        crate::gc::traverse_locked(&self.verifier, |verifier| match verifier {
            Verifier::Manager(manager) => visit.call(manager),
            Verifier::Registry | Verifier::Off => Ok(()),
        })
    }

    /// Drop what only this object holds, for the cycle collector.
    fn __clear__(&self) {
        crate::gc::clear_locked(&self.items);
        let verifier = std::mem::replace(&mut *lock(&self.verifier), Verifier::Off);
        drop(verifier);
    }

    /// Create the empty pipeline `name`, by default an identifier named
    /// `pipeline`, which verifies with the verification registry.
    ///
    /// Raises `TypeError` if `name` is not an `Identifier`.
    #[new]
    #[pyo3(signature = (name = None))]
    fn new(py: Python<'_>, name: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        Ok(Self {
            name: read_pipeline_name(py, name, "PassManager", "pipeline")?,
            items: Mutex::new(Vec::new()),
            verifier: Mutex::new(Verifier::Registry),
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

    /// Append `compiler_pass`, a `CompilerPass`.
    ///
    /// Raises `TypeError` for anything else.
    fn add_pass(&self, compiler_pass: &Bound<'_, PyAny>) -> PyResult<()> {
        let compiler_pass = read_pass(compiler_pass, "PassManager.add_pass")?;
        lock(&self.items).push(Item::Pass(compiler_pass.clone().unbind()));
        Ok(())
    }

    /// Append `group`, a `FixpointPassGroup`.
    ///
    /// Raises `TypeError` for anything else.
    fn add_fixpoint_group(&self, group: &Bound<'_, PyAny>) -> PyResult<()> {
        let Ok(group) = group.cast::<PyFixpointPassGroup>() else {
            return Err(build_argument_type_error(
                "PassManager.add_fixpoint_group",
                "group",
                "a FixpointPassGroup",
                group,
            )?);
        };
        lock(&self.items).push(Item::FixpointGroup(group.clone().unbind()));
        Ok(())
    }

    /// Verify the IR of every run with `verifier`, a `ValidationManager`, or
    /// verify nothing for `None`.
    ///
    /// A run then validates its input before the first pass, blaming that
    /// pass, and the output of every pass that reports a change, blaming the
    /// pass that produced it. Raises `TypeError` for anything else.
    fn set_verifier(&self, verifier: &Bound<'_, PyAny>) -> PyResult<()> {
        let replacement = if verifier.is_none() {
            Verifier::Off
        } else if let Ok(manager) = verifier.cast::<PyValidationManager>() {
            Verifier::Manager(manager.clone().unbind())
        } else {
            return Err(build_argument_type_error(
                "PassManager.set_verifier",
                "verifier",
                "a ValidationManager or None",
                verifier,
            )?);
        };
        let replaced = std::mem::replace(&mut *lock(&self.verifier), replacement);
        drop(replaced);
        Ok(())
    }

    /// Return the number of items, for the pipeline's log lines.
    fn _item_count(&self) -> usize {
        lock(&self.items).len()
    }

    /// Run the pipeline over `ir` and return the `PassManagerResult`.
    ///
    /// Raises the `PassValidationError` or `PassExecutionError` of the first
    /// failure, a pass's, the verifier's rejection or a fixpoint group's
    /// non-convergence, with the records of the work completed before it.
    fn run<'py>(&self, py: Python<'py>, ir: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let guard = ScopeGuard::enter();
        let (mut pipeline, group_names) = self.build(py)?;
        let result = pipeline.run(&PyIr::new(ir));
        drop(pipeline);
        let mut scope = guard.finish();
        if let Some(interrupt) = scope.take_interrupt() {
            return Err(interrupt);
        }
        match result {
            Ok(result) => {
                let records = records_to_python(py, result.records(), &group_names, &mut scope)?;
                let output = result.into_output().into_inner().into_bound(py);
                PyPassManagerResult::build(py, output, records)
            }
            Err(error) => Err(error_to_python(py, error, &mut scope, &group_names)?),
        }
    }

    fn __reduce__(_slf: &Bound<'_, Self>) -> PyResult<()> {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "a PassManager cannot be pickled",
        ))
    }
}
