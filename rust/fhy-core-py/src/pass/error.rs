//! A failed run's [`PassError`] as the Python exception it raises (D-S6-6,
//! as the S6 resolutions amend it; N-S6-2).
//!
//! The class follows [`PassError::class`]: `PassValidationError` or
//! `PassExecutionError`. The message is the core's, naming the Python hook,
//! for example `pass "X" failed in run_pass`, and so is the error
//! diagnostic that records a hook's failure, `pass "X" failed in run_pass:
//! ValueError: boom`. The exception carries `pass_name`, `hook`,
//! `diagnostics`, `records`, and for a verification failure `report`; its
//! `__cause__` is the hook's exception, or, for a nested run's error, the
//! exception that nested run raised.
//!
//! The exception also keeps its Rust error. When it propagates through a
//! hook of another run, the adapter hands that error back to the core,
//! which nests it, so a nested run's error is the core's `Nested`.

use std::error::Error;
use std::sync::{Mutex, PoisonError};

use pyo3::exceptions::PyException;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};

use fhy_core::diagnostic::{Diagnostic, Note};
use fhy_core::pass::{FailureClass, PassError, PassErrorKind, PassHook};

use super::compiler_pass::{CORE_MODULE, HookFailure, PyCompilerPassBase, log_diagnostic};
use super::convert::{diagnostics_to_python, records_to_python, validation_report_to_python};
use super::scope::RunScope;

/// The attribute of an exception the binding raised that keeps its Rust
/// error.
const RUST_ERROR_ATTRIBUTE: &str = "_rust_error";

/// The Rust error of an exception the binding raised, and its rendered
/// chain, kept on the exception. Not exported.
#[pyclass(frozen, module = "fhy_core._rs", name = "_PassErrorCell")]
pub(super) struct PassErrorCell {
    /// The error, until a hook hands it back to the core.
    error: Mutex<Option<PassError>>,
    /// The error's message followed by its causes, as a failure diagnostic
    /// of an outer run ends with it.
    chain: String,
}

/// Return the cell of the exception `error`, if the binding raised it.
fn find_cell<'py>(py: Python<'py>, error: &PyErr) -> Option<Bound<'py, PassErrorCell>> {
    let cell = error
        .value(py)
        .getattr(intern!(py, RUST_ERROR_ATTRIBUTE))
        .ok()?;
    cell.cast_into::<PassErrorCell>().ok()
}

/// Return the rendering of `error` as a failure diagnostic ends with it:
/// the chain of a pass error the binding raised, and otherwise the
/// exception's class and text, `ValueError: boom`.
pub(super) fn chain_of_exception(py: Python<'_>, error: &PyErr) -> String {
    match find_cell(py, error) {
        Some(cell) => cell.get().chain.clone(),
        None => error.to_string(),
    }
}

/// Return the Rust error and chain of `error`, if the binding raised it and
/// no hook took its Rust error yet, taking the error.
pub(super) fn take_nested(py: Python<'_>, error: &PyErr) -> Option<(PassError, String)> {
    let cell = find_cell(py, error)?;
    let cell = cell.get();
    let inner = cell
        .error
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .take()?;
    Some((inner, cell.chain.clone()))
}

/// Return the Python name of the core's `hook`, for a failure no Python
/// hook recorded: `skip` is `should_run`, `run` is `run_pass`, and
/// `preserved_analyses` is `get_preserved_analyses`.
fn python_hook_name(hook: PassHook) -> &'static str {
    match hook {
        PassHook::Skip => "should_run",
        PassHook::Run => "run_pass",
        PassHook::PreservedAnalyses => "get_preserved_analyses",
        other => other.as_str(),
    }
}

/// Return the message of `error` followed by the message of each of its
/// sources, joined by `: `, as the core renders a cause chain.
fn render_chain(error: &(dyn Error + 'static)) -> String {
    let mut text = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        text.push_str(": ");
        text.push_str(&cause.to_string());
        source = cause.source();
    }
    text
}

/// Return the Python error class of `class`.
fn error_class(py: Python<'_>, class: FailureClass) -> PyResult<&Bound<'_, PyType>> {
    if matches!(class, FailureClass::Validation) {
        crate::python::cached_attr!(py, CORE_MODULE, "PassValidationError" => PyType)
    } else {
        crate::python::cached_attr!(py, CORE_MODULE, "PassExecutionError" => PyType)
    }
}

/// What a failure renders as.
struct Rendering {
    message: String,
    chain: String,
    python_hook: Option<&'static str>,
    cause: Option<PyErr>,
    /// The core's text of a hook's failure diagnostic, which the binding
    /// replaces with `chain`.
    core_prefix: Option<String>,
}

/// Return the rendering of `error`, or the exception to raise unchanged,
/// one that is not an `Exception`.
fn render(py: Python<'_>, error: &PassError, scope: &mut RunScope) -> Result<Rendering, PyErr> {
    Ok(match error.kind() {
        PassErrorKind::Hook {
            pass_name,
            hook,
            source,
            ..
        } => {
            let failure = source.downcast_ref::<HookFailure>();
            if let Some(failure) = failure {
                if !failure.error.is_instance_of::<PyException>(py) {
                    return Err(failure.error.clone_ref(py));
                }
            }
            let python_hook = failure.map_or_else(|| python_hook_name(hook), |f| f.python_hook);
            let message = format!("pass {pass_name:?} failed in {python_hook}");
            let tail = failure.map_or_else(|| render_chain(source), |f| f.chain.clone());
            Rendering {
                chain: format!("{message}: {tail}"),
                message,
                python_hook: Some(python_hook),
                cause: failure.map(|f| f.error.clone_ref(py)),
                core_prefix: Some(format!("pass {pass_name:?} failed in {hook}: ")),
            }
        }
        PassErrorKind::Nested {
            pass_name,
            hook,
            inner,
            ..
        } => {
            let note = scope.take_nested();
            let python_hook = note
                .as_ref()
                .map_or_else(|| python_hook_name(hook), |note| note.python_hook);
            let message = format!("pass {pass_name:?} failed in {python_hook}");
            let tail = note
                .as_ref()
                .map_or_else(|| render_chain(inner), |note| note.chain.clone());
            Rendering {
                chain: format!("{message}: {tail}"),
                message,
                python_hook: Some(python_hook),
                cause: note.map(|note| PyErr::from_value(note.exception.into_bound(py))),
                core_prefix: Some(format!("pass {pass_name:?} failed in {hook}: ")),
            }
        }
        _ => {
            let message = error.to_string();
            Rendering {
                chain: message.clone(),
                message,
                python_hook: None,
                cause: None,
                core_prefix: None,
            }
        }
    })
}

/// Return `diagnostics` with the last one, the core's record of a hook's
/// failure starting with `core_prefix`, rewritten to `chain`.
fn rewrite_failure_diagnostic(
    diagnostics: &[Diagnostic],
    core_prefix: &str,
    chain: &str,
) -> Vec<Diagnostic> {
    let mut rewritten = diagnostics.to_vec();
    if let Some(last) = rewritten.last_mut() {
        if last.message_text().starts_with(core_prefix) {
            let mut replacement = Diagnostic::new(
                last.level(),
                Note::new(chain, last.message().kind().clone()),
                last.source().to_owned(),
            );
            if let Some(detail) = last.detail() {
                replacement = replacement.with_detail(detail);
            }
            *last = replacement;
        }
    }
    rewritten
}

/// Return the Python exception of `error`, the failure of a run whose scope
/// is `scope` and whose pipeline items have the Python group names
/// `group_names`.
///
/// An exception that is not an `Exception`, which a hook raised, is
/// returned unchanged. The failing pass's `diagnostics` become the error's.
///
/// # Errors
///
/// Returns what building the exception raises.
pub(super) fn error_to_python(
    py: Python<'_>,
    error: PassError,
    scope: &mut RunScope,
    group_names: &[Option<Py<PyAny>>],
) -> PyResult<PyErr> {
    let rendering = match render(py, &error, scope) {
        Ok(rendering) => rendering,
        Err(unwrapped) => return Ok(unwrapped),
    };
    let diagnostics = match &rendering.core_prefix {
        Some(prefix) => rewrite_failure_diagnostic(error.diagnostics(), prefix, &rendering.chain),
        None => error.diagnostics().to_vec(),
    };
    let diagnostic_objects = diagnostics_to_python(py, &diagnostics, scope)?;
    if let Some(last) = diagnostics.last() {
        let exc_info = rendering
            .cause
            .as_ref()
            .map(|cause| cause.value(py).clone().into_any());
        if rendering.core_prefix.is_some()
            || matches!(error.kind(), PassErrorKind::Verification { .. })
        {
            log_diagnostic(py, last, exc_info.as_ref())?;
        }
    }
    if rendering.core_prefix.is_some() {
        if let Some(pass) = scope.take_failed_pass() {
            if let Ok(pass) = pass.bind(py).cast::<PyCompilerPassBase>() {
                pass.get().set_diagnostics(&diagnostic_objects);
            }
        }
    }
    let records = records_to_python(py, error.records(), group_names, scope)?;
    let class = error.class();
    let keywords = PyDict::new(py);
    keywords.set_item(intern!(py, "pass_name"), error.pass_name())?;
    keywords.set_item(intern!(py, "hook"), rendering.python_hook)?;
    keywords.set_item(intern!(py, "diagnostics"), diagnostic_objects)?;
    keywords.set_item(intern!(py, "records"), records)?;
    if let PassErrorKind::Verification { report, .. } = error.kind() {
        if class == FailureClass::Validation {
            keywords.set_item(
                intern!(py, "report"),
                validation_report_to_python(py, report, scope)?,
            )?;
        }
    }
    let exception = error_class(py, class)?.call((rendering.message,), Some(&keywords))?;
    exception.setattr(
        intern!(py, RUST_ERROR_ATTRIBUTE),
        Bound::new(
            py,
            PassErrorCell {
                error: Mutex::new(Some(error)),
                chain: rendering.chain,
            },
        )?,
    )?;
    let python_error = PyErr::from_value(exception);
    if let Some(cause) = rendering.cause {
        python_error.set_cause(py, Some(cause));
    }
    Ok(python_error)
}
