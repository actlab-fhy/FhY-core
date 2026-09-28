//! The verification registry behind `fhy_core.pass_infrastructure.verification`
//! (S14 of `docs/design/python-switch.md`): the one registry of the Python
//! API, held in the extension's module state, the three functions over it,
//! and the registry verifier a pipeline runs by default (D-S6-12).
//!
//! The core's [`VerificationRegistry`] keys its registrations by the
//! address of the IR type they were registered for, and names each by the
//! address of its pass class ([`VerifierId::of_ptr`]); the state keeps both
//! objects alive for as long as it exists. A lookup's lineage is the
//! reversed `__mro__` of the IR's type, the Python reflection the core
//! leaves to its caller. A registration swaps in a new state, and a lookup
//! takes the current one and releases the lock before it builds a check,
//! so the lock is never held across a call into Python.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyTuple, PyType};

use fhy_core::foreign::BoxError;
use fhy_core::identifier::Identifier;
use fhy_core::pass::{PassContext, ValidationManager, Validator, VerificationRegistry, VerifierId};

use crate::dataclass::build_argument_type_error;

use super::compiler_pass::{
    HookFailure, PyCompilerPassBase, PythonPass, build_interrupted_failure,
};
use super::convert::validation_report_to_python;
use super::ir::PyIr;
use super::scope::{self, ScopeGuard};
use super::validation::{fail_validator, run_check};

/// The name of the validator a pipeline verifies its IR with by default.
const REGISTRY_VERIFIER_NAME: &str = "verification";

/// The attribute of `fhy_core._rs` that holds the registry.
pub(crate) const REGISTRY_ATTRIBUTE: &str = "_verification_registry";

/// A Python type, as a kind of the core registry: its address.
type TypeKey = usize;

/// Return the kind of the type `ty`.
fn type_key(ty: &Bound<'_, PyAny>) -> TypeKey {
    ty.as_ptr().addr()
}

/// One version of the registry: the core's registrations, and the Python
/// objects their kinds and ids stand for.
#[derive(Clone, Default)]
struct RegistryState {
    registry: VerificationRegistry<TypeKey, PyIr>,
    /// The registered pass classes, by their ids.
    classes: HashMap<VerifierId, Arc<Py<PyType>>>,
    /// The IR types registered for, by their kinds.
    kinds: HashMap<TypeKey, Arc<Py<PyType>>>,
}

/// The verification registry of the Python API, held in the extension's
/// module state as `fhy_core._rs._verification_registry`.
#[pyclass(frozen, module = "fhy_core._rs", name = "VerificationRegistryState")]
pub(crate) struct PyVerificationRegistry {
    state: Mutex<Arc<RegistryState>>,
}

impl PyVerificationRegistry {
    /// Create the empty registry.
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(Arc::new(RegistryState::default())),
        }
    }

    /// Return the current version of the registry.
    fn snapshot(&self) -> Arc<RegistryState> {
        Arc::clone(&self.state.lock().unwrap_or_else(PoisonError::into_inner))
    }
}

/// Return the registry of the extension's module state.
fn registry(py: Python<'_>) -> PyResult<&Bound<'_, PyVerificationRegistry>> {
    static REGISTRY: PyOnceLock<Py<PyVerificationRegistry>> = PyOnceLock::new();
    REGISTRY.import(py, "fhy_core._rs", REGISTRY_ATTRIBUTE)
}

/// Return `ir_type` as a type, or the `TypeError` naming `owner`.
fn read_ir_type<'a, 'py>(
    ir_type: &'a Bound<'py, PyAny>,
    owner: &str,
) -> PyResult<&'a Bound<'py, PyType>> {
    match ir_type.cast::<PyType>() {
        Ok(ir_type) => Ok(ir_type),
        Err(_not_a_type) => Err(build_argument_type_error(
            owner, "ir_type", "a type", ir_type,
        )?),
    }
}

/// Return the lineage of `ir_type`: its `__mro__`, base first.
fn lineage_of(ir_type: &Bound<'_, PyType>) -> Vec<TypeKey> {
    let mro = ir_type.mro();
    let mut lineage: Vec<TypeKey> = mro.iter().map(|ty| type_key(&ty)).collect();
    lineage.reverse();
    lineage
}

/// A verification pass, built when a lookup runs, as a check.
enum VerificationCheck {
    /// The pass object, built.
    Built(PythonPass),
    /// The pass class raised when it was called, or built something other
    /// than a pass.
    Unbuilt { name: String, error: PyErr },
}

/// Return the name of the pass class `class` for a check it could not
/// build: its `get_pass_name()`, or its `__name__`.
fn read_class_pass_name(class: &Bound<'_, PyType>) -> String {
    let py = class.py();
    class
        .call_method0(intern!(py, "get_pass_name"))
        .and_then(|name| name.extract::<String>())
        .or_else(|_no_pass_name| class.name().map(|name| name.to_string()))
        .unwrap_or_else(|_no_name| "verification pass".to_owned())
}

/// Return the check of a new instance of the pass class `class`.
fn build_check(class: &Bound<'_, PyType>) -> VerificationCheck {
    let built = class.call0().and_then(|pass| {
        let pass = pass
            .cast_into::<PyCompilerPassBase>()
            .map_err(PyErr::from)?;
        PythonPass::new(&pass, false)
    });
    match built {
        Ok(pass) => VerificationCheck::Built(pass),
        Err(error) => VerificationCheck::Unbuilt {
            name: read_class_pass_name(class),
            error,
        },
    }
}

impl Validator<PyIr> for VerificationCheck {
    fn name(&self) -> Cow<'static, str> {
        match self {
            Self::Built(pass) => Cow::Owned(pass.pass_name().to_owned()),
            Self::Unbuilt { name, .. } => Cow::Owned(name.clone()),
        }
    }

    fn validate(&mut self, ir: &PyIr, cx: &mut PassContext<'_>) -> Result<(), BoxError> {
        Python::attach(|py| match self {
            Self::Built(pass) => run_check(py, pass, ir, cx),
            Self::Unbuilt { name, error } => {
                let error = error.clone_ref(py);
                Err(fail_validator(py, name, error, cx))
            }
        })
    }
}

/// Return the failure of the registry verifier that `error` stopped.
fn verifier_failure(error: PyErr) -> BoxError {
    let chain = error.to_string();
    Box::new(HookFailure {
        python_hook: "validate",
        error,
        chain,
    })
}

/// The verifier a pipeline runs by default: it checks the IR with a new
/// instance of each verification pass registered for the IR's type, as
/// `run_verification` would, into one record.
struct RegistryVerifier;

impl Validator<PyIr> for RegistryVerifier {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed(REGISTRY_VERIFIER_NAME)
    }

    fn validate(&mut self, ir: &PyIr, cx: &mut PassContext<'_>) -> Result<(), BoxError> {
        Python::attach(|py| {
            let state = registry(py).map_err(verifier_failure)?.get().snapshot();
            let lineage = lineage_of(&ir.bind(py).get_type());
            let mut failure = None;
            for mut check in state.registry.validators_for(&lineage) {
                if scope::is_interrupted() {
                    return Err(build_interrupted_failure("validate"));
                }
                if let Err(error) = check.validate(ir, cx) {
                    failure = Some(error);
                }
            }
            failure.map_or(Ok(()), Err)
        })
    }
}

/// Return the core's validation pipeline of the registry verifier.
pub(super) fn build_registry_verifier() -> ValidationManager<'static, PyIr> {
    let mut manager = ValidationManager::new(Identifier::new(REGISTRY_VERIFIER_NAME));
    manager.add(RegistryVerifier);
    manager
}

/// Return the `PassRegistrationError` for `pass_class`, which is not a
/// `CompilerPass` subclass.
fn build_not_a_pass_error(pass_class: &Bound<'_, PyAny>) -> PyResult<PyErr> {
    let py = pass_class.py();
    let qualname = match pass_class.getattr(intern!(py, "__qualname__")) {
        Ok(qualname) => qualname.str()?.to_string(),
        Err(_no_qualname) => pass_class.repr()?.to_string(),
    };
    Ok(crate::exceptions::PASS_REGISTRATION_ERROR.err(
        py,
        (format!(
            "Cannot register non-CompilerPass type as a verification pass: {qualname}."
        ),),
    ))
}

/// Register the verification pass class `pass_class` for the IR type
/// `ir_type`, and return whether the registration is new.
///
/// Registering the same pair again changes nothing. Raises `TypeError` if
/// `ir_type` is not a type, and `PassRegistrationError` if `pass_class` is
/// not a `CompilerPass` subclass; nothing is registered then.
#[pyfunction]
pub(crate) fn register_verification_pass(
    ir_type: &Bound<'_, PyAny>,
    pass_class: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let py = ir_type.py();
    let ir_type = read_ir_type(ir_type, "register_verification_pass")?;
    let pass_class = match pass_class.cast::<PyType>() {
        Ok(class) if class.is_subclass_of::<PyCompilerPassBase>()? => class,
        _not_a_pass => return Err(build_not_a_pass_error(pass_class)?),
    };
    let kind = type_key(ir_type.as_any());
    let id = VerifierId::of_ptr(pass_class.as_ptr());
    let class = Arc::new(pass_class.clone().unbind());
    let ir_type = Arc::new(ir_type.clone().unbind());
    let registry = registry(py)?.get();
    let mut state = registry
        .state
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    if state.registry.ids_for([&kind]).contains(&id) {
        return Ok(false);
    }
    let next = Arc::make_mut(&mut state);
    let factory_class = Arc::clone(&class);
    next.registry.register_with_id(kind, id, move || {
        Python::attach(|py| build_check(factory_class.bind(py)))
    });
    next.classes.entry(id).or_insert(class);
    next.kinds.entry(kind).or_insert(ir_type);
    Ok(true)
}

/// Return the verification pass classes registered for `ir_type` and its
/// bases: each type's in `__mro__` order, base first, in registration
/// order, and each class once, at its first position.
///
/// Raises `TypeError` if `ir_type` is not a type.
#[pyfunction]
pub(crate) fn get_verification_passes_for<'py>(
    ir_type: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyTuple>> {
    let py = ir_type.py();
    let ir_type = read_ir_type(ir_type, "get_verification_passes_for")?;
    let state = registry(py)?.get().snapshot();
    let classes = state
        .registry
        .ids_for(&lineage_of(ir_type))
        .into_iter()
        .filter_map(|id| state.classes.get(&id))
        .map(|class| class.bind(py).clone())
        .collect::<Vec<_>>();
    PyTuple::new(py, classes)
}

/// Verify `ir` with a new instance of each verification pass registered for
/// its type, and return the aggregated `ValidationReport`, with one
/// `ValidatorRecord` per pass.
///
/// Every pass runs, as a `ValidationManager` runs it; a pass class whose
/// construction raises fails its check. An exception that is not an
/// `Exception` propagates unchanged.
#[pyfunction]
pub(crate) fn run_verification<'py>(ir: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = ir.py();
    let state = registry(py)?.get().snapshot();
    let lineage = lineage_of(&ir.get_type());
    let guard = ScopeGuard::enter();
    let report = state.registry.verify(&lineage, &PyIr::new(ir));
    drop(state);
    let mut scope = guard.finish();
    if let Some(interrupt) = scope.take_interrupt() {
        return Err(interrupt);
    }
    validation_report_to_python(py, &report, &mut scope)
}
