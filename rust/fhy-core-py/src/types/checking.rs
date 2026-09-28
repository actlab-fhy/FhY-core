//! The functions `fhy_core.types.checking` runs: the type checker, the body
//! checks, the sweep and the sort tables (S11b of
//! `docs/design/python-switch.md`, D-S11-20 to D-S11-22).
//!
//! The checker's two lookups are Python callables, called once per
//! identifier occurrence and once per call node the walk meets; the walk
//! checks a shared sub-expression other than a leaf at most twice, so the
//! lookups inside it run a bounded number of times however often it is
//! reached. A lookup's exception
//! propagates as the same object, and a result of the wrong shape raises
//! `TypeError`. When the call-target resolver is the registry's
//! `get_registered_entry`, calls resolve through the registry snapshot
//! without calling Python.

use std::cell::OnceCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyString, PyTuple, PyType};

use fhy_core::expression::{Callee, ExpressionKind, FunctionName, FunctionSort, LiteralValue};
use fhy_core::foreign::BoxError;
use fhy_core::identifier::Identifier;
use fhy_core::tree::NodeHandle;
use fhy_core::types::checking::{
    BodyCheck, BodyCheckError, CallTarget, CallTargetError, CallTargets, FunctionSignature,
    IdentifierTypes, TypeCheckError, TypeChecker, check_all_function_bodies, check_function_body,
};
use fhy_core::types::{CoreDataType, Type, TypeQualifier};

use crate::dataclass::build_argument_type_error;
use crate::error::IntoPyResult;
use crate::expression::{
    PyExpression, PyIdentifierExpression, RegistryState, get_registered_entry, read_big_int,
    read_call_target, read_sort, registry_snapshot,
};
use crate::identifier::restore_identifier;

use super::adapter::{Context, run_in_context};
use super::convert::{identifier_to_python, read_type, type_to_python};
use super::enums::{
    core_data_type_to_python, read_core_data_type, read_type_qualifier, type_qualifier_to_python,
};
use super::error::core_type_error_class;

/// The source of the diagnostics the registry sweep reports.
const SWEEP_SOURCE: &str = "fhy_core.types.checking.check_all_registered_function_bodies";

// ---------------------------------------------------------------------------
// Python classes
// ---------------------------------------------------------------------------

/// Return `EntryLookupError`.
fn entry_lookup_error_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    crate::python::cached_attr!(py, "fhy_core.symbolic.expression.errors", "EntryLookupError" => PyType)
}

/// Return `EntryRegistrationError`.
fn entry_registration_error_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    crate::python::cached_attr!(py, "fhy_core.symbolic.expression.errors", "EntryRegistrationError" => PyType)
}

/// Return whether `resolver` is the registry's `get_registered_entry`.
fn is_registry_resolver(resolver: &Bound<'_, PyAny>) -> PyResult<bool> {
    static RESOLVER: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    let registry_resolver = RESOLVER.import(
        resolver.py(),
        "fhy_core.symbolic.expression.registry",
        "get_registered_entry",
    )?;
    Ok(resolver.is(registry_resolver))
}

/// Return the class `name` of `fhy_core.diagnostic`.
fn diagnostic_class<'py>(py: Python<'py>, name: &'static str) -> PyResult<Bound<'py, PyAny>> {
    static MODULE: PyOnceLock<Py<PyModule>> = PyOnceLock::new();
    MODULE
        .get_or_try_init(py, || py.import("fhy_core.diagnostic").map(Bound::unbind))?
        .bind(py)
        .getattr(name)
}

/// Return the exception of `class` with `message`, or the error building
/// it.
fn build_error(class: PyResult<&Bound<'_, PyType>>, message: String) -> PyErr {
    match class.and_then(|class| class.call1((message,))) {
        Ok(error) => PyErr::from_value(error),
        Err(error) => error,
    }
}

/// Return the lookup's own exception boxed in `error`, or a `RuntimeError`
/// with its text when it is no Python exception.
fn callback_to_python(error: BoxError) -> PyErr {
    match error.downcast::<PyErr>() {
        Ok(error) => *error,
        Err(other) => PyRuntimeError::new_err(other.to_string()),
    }
}

/// Return the text of the `EntryLookupError` `error`: its message, not the
/// quoted `repr` a `KeyError` prints.
fn lookup_error_message(py: Python<'_>, error: &PyErr) -> String {
    let value = error.value(py);
    if let Ok(args) = value.getattr(intern!(py, "args")) {
        if let Ok(args) = args.cast::<PyTuple>() {
            if args.len() == 1 {
                if let Ok(message) = args.get_item(0) {
                    if let Ok(message) = message.cast::<PyString>() {
                        return message.to_string();
                    }
                }
            }
        }
    }
    value.to_string()
}

// ---------------------------------------------------------------------------
// The lookups
// ---------------------------------------------------------------------------

/// The identifier types of a Python `get_identifier_type(identifier)`,
/// which raises `KeyError` for an identifier it does not bind.
struct PythonIdentifierTypes<'py, 'a> {
    lookup: &'a Bound<'py, PyAny>,
    context: &'a Context,
    /// The root expression's object, whose identifier references give the
    /// lookup the identifier objects the caller built.
    root: &'a Bound<'py, PyAny>,
    /// The identifier objects of the root, by id, collected on the first
    /// lookup.
    objects: OnceCell<HashMap<u64, Py<PyAny>>>,
}

impl<'py, 'a> PythonIdentifierTypes<'py, 'a> {
    fn new(
        lookup: &'a Bound<'py, PyAny>,
        context: &'a Context,
        root: &'a Bound<'py, PyAny>,
    ) -> Self {
        Self {
            lookup,
            context,
            root,
            objects: OnceCell::new(),
        }
    }

    /// Return the identifier objects of the expression tree of `root`, by
    /// id, visiting each shared node once.
    fn collect_objects(root: &Bound<'py, PyAny>) -> HashMap<u64, Py<PyAny>> {
        let py = root.py();
        let mut objects = HashMap::new();
        let mut seen = HashSet::new();
        let mut pending = vec![root.clone()];
        while let Some(object) = pending.pop() {
            let Ok(node) = object.cast::<PyExpression>() else {
                continue;
            };
            if !seen.insert(node.get().expression().identity()) {
                continue;
            }
            if let Ok(reference) = object.cast::<PyIdentifierExpression>() {
                if let ExpressionKind::Identifier(identifier) = node.get().expression().kind() {
                    objects
                        .entry(identifier.id())
                        .or_insert_with(|| reference.get().identifier_object().clone_ref(py));
                }
                continue;
            }
            pending.extend(node.get().children(py).iter());
        }
        objects
    }

    /// Return the Python object of `identifier`: the caller's, or a new one.
    fn identifier_object(&self, identifier: &Identifier) -> PyResult<Bound<'py, PyAny>> {
        let py = self.lookup.py();
        let objects = self
            .objects
            .get_or_init(|| Self::collect_objects(self.root));
        match objects.get(&identifier.id()) {
            Some(object) => Ok(object.bind(py).clone()),
            None => identifier_to_python(py, self.context, identifier),
        }
    }

    /// Return the type and qualifier the lookup gives `identifier`, or
    /// `None` when it raises `KeyError`.
    fn look_up(&self, identifier: &Identifier) -> PyResult<Option<(Type, TypeQualifier)>> {
        let py = self.lookup.py();
        let object = self.identifier_object(identifier)?;
        let result = match self.lookup.call1((object,)) {
            Ok(result) => result,
            Err(error) if error.is_instance_of::<pyo3::exceptions::PyKeyError>(py) => {
                return Ok(None);
            }
            Err(error) => return Err(error),
        };
        let pair = match result.cast::<PyTuple>() {
            Ok(pair) if pair.len() == 2 => pair.clone(),
            _ => {
                return Err(PyTypeError::new_err(format!(
                    "get_identifier_type must return a (Type, TypeQualifier) pair, got {}.",
                    result.get_type().name()?
                )));
            }
        };
        let type_object = pair.get_item(0)?;
        let Some(value) = read_type(self.context, &type_object)? else {
            return Err(build_argument_type_error(
                "get_identifier_type",
                "result type",
                "a Type",
                &type_object,
            )?);
        };
        let qualifier = read_type_qualifier(
            &pair.get_item(1)?,
            "get_identifier_type",
            "result qualifier",
        )?;
        Ok(Some((value, qualifier)))
    }
}

impl IdentifierTypes for PythonIdentifierTypes<'_, '_> {
    fn identifier_type(
        &self,
        identifier: &Identifier,
    ) -> Result<Option<(Type, TypeQualifier)>, BoxError> {
        self.look_up(identifier)
            .map_err(|error| -> BoxError { Box::new(error) })
    }
}

/// The call targets of a Python `resolve_call_target(name)`, which raises
/// `EntryLookupError` for a name it does not know; the registry's own
/// `get_registered_entry` resolves through the snapshot.
enum PythonCallTargets<'py, 'a> {
    /// The registry snapshot.
    Registry(Arc<RegistryState>, Python<'py>),
    /// Any other resolver.
    Resolver(&'a Bound<'py, PyAny>),
}

impl<'py, 'a> PythonCallTargets<'py, 'a> {
    fn new(resolver: &'a Bound<'py, PyAny>) -> PyResult<Self> {
        Ok(if is_registry_resolver(resolver)? {
            Self::Registry(registry_snapshot(), resolver.py())
        } else {
            Self::Resolver(resolver)
        })
    }

    /// Return the unknown-call error of `name`, carrying `error`.
    fn unknown(py: Python<'_>, name: &str, error: PyErr) -> CallTargetError {
        CallTargetError::Unknown {
            name: name.to_owned(),
            message: lookup_error_message(py, &error),
            source: Some(Box::new(error)),
        }
    }
}

impl CallTargets for PythonCallTargets<'_, '_> {
    fn call_target(&self, callee: &Callee) -> Result<CallTarget, CallTargetError> {
        match self {
            Self::Registry(snapshot, py) => match snapshot.registry().call_target(callee) {
                Err(CallTargetError::Unknown { name, .. }) => {
                    // The lookup's own error, so its text and class are
                    // those `get_registered_entry` raises.
                    let error = match get_registered_entry(&PyString::new(*py, &name)) {
                        Ok(_) => PyRuntimeError::new_err(format!(
                            "the registry resolved '{name}' after the snapshot did not"
                        )),
                        Err(error) => error,
                    };
                    Err(Self::unknown(*py, &name, error))
                }
                other => other,
            },
            Self::Resolver(resolver) => {
                let py = resolver.py();
                let name = callee.name();
                let entry = match resolver.call1((name,)) {
                    Ok(entry) => entry,
                    Err(error) => {
                        let is_lookup_error = entry_lookup_error_class(py)
                            .is_ok_and(|class| error.is_instance(py, class));
                        return Err(if is_lookup_error {
                            Self::unknown(py, name, error)
                        } else {
                            CallTargetError::Callback(Box::new(error))
                        });
                    }
                };
                match read_call_target(&entry) {
                    Ok(Some(target)) => Ok(target),
                    Ok(None) => Err(CallTargetError::Callback(Box::new(
                        match build_argument_type_error(
                            "resolve_call_target",
                            "result",
                            "a RegisteredFunction, NativeFunction or NativeConstant",
                            &entry,
                        ) {
                            Ok(error) | Err(error) => error,
                        },
                    ))),
                    Err(error) => Err(CallTargetError::Callback(Box::new(error))),
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Return the Python exception of the checker's error `error`:
/// `NotImplementedError` for an unsupported construct, `FhYCoreTypeError`
/// for another broken rule, the lookup's `EntryLookupError` for a deferred
/// unknown call, and a lookup's own exception.
fn type_check_error_to_python(py: Python<'_>, error: TypeCheckError) -> PyErr {
    match error {
        TypeCheckError::Rule { ref rule, .. } if rule.is_unsupported() => {
            PyNotImplementedError::new_err(error.to_string())
        }
        error @ TypeCheckError::Rule { .. } => {
            build_error(core_type_error_class(py), error.to_string())
        }
        TypeCheckError::UnknownCall(error) => call_target_error_to_python(py, error),
        TypeCheckError::Callback(source) => callback_to_python(source),
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

/// Return the Python exception of the call-target error `error`.
fn call_target_error_to_python(py: Python<'_>, error: CallTargetError) -> PyErr {
    match error {
        CallTargetError::Unknown {
            source: Some(source),
            ..
        }
        | CallTargetError::Callback(source) => callback_to_python(source),
        CallTargetError::Unknown { message, .. } => {
            build_error(entry_lookup_error_class(py), message)
        }
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

/// Return the `EntryRegistrationError` of the body-check error `error`,
/// whose `__cause__` is the checker's or the lookup's exception, or a
/// lookup's own exception when a lookup failed.
fn body_check_error_to_python(py: Python<'_>, error: BodyCheckError) -> PyErr {
    let message = error.to_string();
    let cause = match error {
        BodyCheckError::Callback(source) => return callback_to_python(source),
        BodyCheckError::UnknownCall { error, .. } => Some(call_target_error_to_python(py, error)),
        BodyCheckError::Unsupported { error, .. } | BodyCheckError::IllTyped { error, .. } => {
            Some(type_check_error_to_python(py, error))
        }
        _ => None,
    };
    let registration_error = build_error(entry_registration_error_class(py), message);
    if cause.is_some() {
        registration_error.set_cause(py, cause);
    }
    registration_error
}

// ---------------------------------------------------------------------------
// The checker
// ---------------------------------------------------------------------------

/// Return the type and qualifier of `expression`: synthesized, or checked
/// against `expected_type` when it is not `None`. Identifiers are typed by
/// `get_identifier_type`, calls resolved by `resolve_call_target`, and a
/// call of an unknown function raises the resolver's `EntryLookupError`
/// unframed when `defer_on_unknown_call` holds.
///
/// Raises `FhYCoreTypeError` for a broken rule, `NotImplementedError` for
/// an unsupported construct, a lookup's own exception, and `TypeError` for
/// an argument or a lookup result of the wrong kind.
#[pyfunction]
pub(crate) fn types_check_expression<'py>(
    expression: &Bound<'py, PyAny>,
    expected_type: &Bound<'py, PyAny>,
    get_identifier_type: &Bound<'py, PyAny>,
    resolve_call_target: &Bound<'py, PyAny>,
    defer_on_unknown_call: bool,
) -> PyResult<Bound<'py, PyTuple>> {
    let py = expression.py();
    let Ok(node) = expression.cast::<PyExpression>() else {
        return Err(build_argument_type_error(
            "ExpressionTypeChecker",
            "expression",
            "an Expression",
            expression,
        )?);
    };
    let targets = PythonCallTargets::new(resolve_call_target)?;
    run_in_context(py, None, |context| {
        let expected = if expected_type.is_none() {
            None
        } else {
            match read_type(context, expected_type)? {
                Some(expected) => Some(expected),
                None => {
                    return Err(build_argument_type_error(
                        "ExpressionTypeChecker",
                        "expected_type",
                        "a Type",
                        expected_type,
                    )?);
                }
            }
        };
        let identifiers = PythonIdentifierTypes::new(get_identifier_type, context, expression);
        let snapshot = registry_snapshot();
        let mut checker = TypeChecker::new(&identifiers, &targets, snapshot.registry());
        if defer_on_unknown_call {
            checker = checker.with_deferred_unknown_calls();
        }
        let rust_expression = node.get().expression();
        let result = match &expected {
            Some(expected) => checker.check(rust_expression, expected),
            None => checker.synthesize(rust_expression),
        };
        let (value, qualifier) = result.map_err(|error| type_check_error_to_python(py, error))?;
        PyTuple::new(
            py,
            [
                type_to_python(py, context, &value)?,
                type_qualifier_to_python(py, qualifier)?,
            ],
        )
    })
}

/// Hold the body `body` of the function `name` of `parameters`, each of the
/// sort at its position in `parameter_sorts`, to `result_sort`, resolving
/// calls through `resolve_call_target`; a call of an unknown function
/// abandons the check when `defer_unresolved_calls` holds.
///
/// Raises `EntryRegistrationError` with the core's text when the body does
/// not satisfy its signature, a lookup's own exception, and `TypeError` or
/// `ValueError` for arguments of the wrong kind.
#[pyfunction]
pub(crate) fn types_check_function_body(
    name: &Bound<'_, PyAny>,
    parameters: &Bound<'_, PyAny>,
    parameter_sorts: &Bound<'_, PyAny>,
    result_sort: &Bound<'_, PyAny>,
    body: &Bound<'_, PyAny>,
    resolve_call_target: &Bound<'_, PyAny>,
    defer_unresolved_calls: bool,
) -> PyResult<()> {
    const OWNER: &str = "RegisteredFunctionBodyTypeChecker";
    let py = name.py();
    let Ok(text) = name.cast::<PyString>() else {
        return Err(build_argument_type_error(OWNER, "name", "a str", name)?);
    };
    let function_name = FunctionName::new(text.to_str()?).into_py_result()?;
    let rust_parameters = parameters
        .try_iter()?
        .map(|parameter| restore_identifier(&parameter?, OWNER, "parameters"))
        .collect::<PyResult<Vec<Identifier>>>()?;
    let rust_sorts = parameter_sorts
        .try_iter()?
        .map(|sort| read_sort(&sort?, OWNER, "parameter_sorts"))
        .collect::<PyResult<Vec<FunctionSort>>>()?;
    if rust_parameters.len() != rust_sorts.len() {
        return Err(PyValueError::new_err(format!(
            "{OWNER} has {} parameters but {} parameter sorts.",
            rust_parameters.len(),
            rust_sorts.len()
        )));
    }
    let rust_result_sort = read_sort(result_sort, OWNER, "result_sort")?;
    let Ok(body) = body.cast::<PyExpression>() else {
        return Err(build_argument_type_error(
            OWNER,
            "body",
            "an Expression",
            body,
        )?);
    };
    let targets = PythonCallTargets::new(resolve_call_target)?;
    run_in_context(py, None, |_context| {
        let signature = FunctionSignature::new(
            &function_name,
            &rust_parameters,
            &rust_sorts,
            rust_result_sort,
        )
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let snapshot = registry_snapshot();
        match check_function_body(
            &signature,
            body.get().expression(),
            &targets,
            snapshot.registry(),
            defer_unresolved_calls,
        ) {
            Ok(BodyCheck::Checked | BodyCheck::Deferred) => Ok(()),
            Ok(other) => Err(PyRuntimeError::new_err(format!(
                "unexpected body check outcome {other:?}"
            ))),
            Err(error) => Err(body_check_error_to_python(py, error)),
        }
    })
}

/// Return the `ValidationReport` of every function body held to its
/// declared result sort, the composed built-ins in catalogue order and then
/// the user functions in registration order, one ERROR diagnostic per
/// failing function, resolving calls through the registry snapshot.
///
/// `on_checked`, when given, is called with the name of each function the
/// sweep checked, in that order, once the sweep is done.
#[pyfunction]
#[pyo3(signature = (on_checked = None))]
pub(crate) fn types_check_all_function_bodies<'py>(
    py: Python<'py>,
    on_checked: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let snapshot = registry_snapshot();
    let sweep = run_in_context(py, None, |_context| {
        Ok(check_all_function_bodies(snapshot.registry()))
    })?;
    if let Some(on_checked) = on_checked {
        for label in sweep.checked() {
            on_checked.call1((label.to_string(),))?;
        }
    }
    let failures = sweep.into_failures();
    let error_level = diagnostic_class(py, "DiagnosticLevel")?.getattr("ERROR")?;
    let note = diagnostic_class(py, "Note")?;
    let diagnostic = diagnostic_class(py, "Diagnostic")?;
    let mut diagnostics = Vec::with_capacity(failures.len());
    for (_, error) in failures {
        let kwargs = PyDict::new(py);
        kwargs.set_item("level", &error_level)?;
        kwargs.set_item("message", note.call1((error.to_string(),))?)?;
        kwargs.set_item("source", SWEEP_SOURCE)?;
        diagnostics.push(diagnostic.call((), Some(&kwargs))?);
    }
    let kwargs = PyDict::new(py);
    kwargs.set_item("diagnostics", PyTuple::new(py, diagnostics)?)?;
    diagnostic_class(py, "ValidationReport")?.call((), Some(&kwargs))
}

// ---------------------------------------------------------------------------
// The sort tables and literals
// ---------------------------------------------------------------------------

/// Return whether a value of `core_data_type` satisfies `sort`.
///
/// Raises `TypeError` if either is not a member of its enum.
#[pyfunction]
pub(crate) fn is_core_data_type_compatible_with_sort(
    core_data_type: &Bound<'_, PyAny>,
    sort: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    const OWNER: &str = "is_core_data_type_compatible_with_sort";
    let core_data_type = read_core_data_type(core_data_type, OWNER, "core_data_type")?;
    let sort = read_sort(sort, OWNER, "sort")?;
    Ok(core_data_type.is_compatible_with_sort(sort))
}

/// Return the core data type a value of `sort` takes.
///
/// Raises `TypeError` if `sort` is no `FunctionSort`.
#[pyfunction]
pub(crate) fn get_result_core_data_type_for_sort<'py>(
    sort: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let rust_sort = read_sort(sort, "get_result_core_data_type_for_sort", "sort")?;
    core_data_type_to_python(sort.py(), CoreDataType::of_sort(rust_sort))
}

/// Return the weak core data type of the literal value `literal`.
///
/// Raises `NotImplementedError` for a `Decimal` or a `str`, and
/// `ValueError` for any other value that is no `bool`, `int` or `float`.
#[pyfunction]
pub(crate) fn get_core_data_type_from_literal_type<'py>(
    literal: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    use pyo3::types::{PyBool, PyFloat, PyInt};
    let py = literal.py();
    let value = if let Ok(boolean) = literal.cast::<PyBool>() {
        LiteralValue::Bool(boolean.is_true())
    } else if literal.is_instance_of::<PyInt>() {
        LiteralValue::Int(read_big_int(literal)?)
    } else if let Ok(float) = literal.cast::<PyFloat>() {
        LiteralValue::Float(float.value())
    } else if literal
        .is_instance(crate::python::cached_attr!(py, "decimal", "Decimal" => PyType)?)?
    {
        return Err(PyNotImplementedError::new_err(
            fhy_core::types::LiteralTypeError::UnsupportedDecimal.to_string(),
        ));
    } else if literal.is_instance_of::<PyString>() {
        return Err(PyNotImplementedError::new_err(
            "string literals are not yet supported",
        ));
    } else {
        return Err(PyValueError::new_err(format!(
            "unsupported literal type: {}",
            literal.get_type().repr()?
        )));
    };
    let core_data_type = CoreDataType::of_literal(&value).into_py_result()?;
    core_data_type_to_python(py, core_data_type)
}
