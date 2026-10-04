//! The log records of undecided outcomes: the core reports each
//! [`ConstraintEvent`], and the observer here logs it on
//! `fhy_core.symbolic.constraint.core` with its level and its text, which
//! names values by their Python `repr`s.

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyString;

use fhy_core::constraint::{ConstraintEvent, ConstraintObserver};

use crate::expression::render_expression_repr;

use super::value::repr_text;
use crate::util::pending::record_pending_error;

/// `logging.DEBUG`.
pub(crate) const DEBUG: u8 = 10;
/// `logging.WARNING`.
pub(crate) const WARNING: u8 = 30;

/// Return `fhy_core.symbolic.constraint.core`'s logger.
pub(crate) fn core_logger(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    static LOGGER: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    LOGGER
        .get_or_try_init(py, || {
            py.import(intern!(py, "fhy_core.logger"))?
                .call_method1(
                    intern!(py, "get_logger"),
                    ("fhy_core.symbolic.constraint.core",),
                )
                .map(Bound::unbind)
        })
        .map(|logger| logger.bind(py).clone())
}

/// Log `message` at `level` on `logger`, if the logger handles that level.
///
/// The message is passed as the argument of `"%s"`, so it is not formatted
/// again.
pub(crate) fn log(
    logger: &Bound<'_, PyAny>,
    level: u8,
    message: impl FnOnce() -> String,
) -> PyResult<()> {
    let py = logger.py();
    if !logger
        .call_method1(intern!(py, "isEnabledFor"), (level,))?
        .is_truthy()?
    {
        return Ok(());
    }
    logger.call_method1(intern!(py, "log"), (level, "%s", message()))?;
    Ok(())
}

/// Return the text of the WARNING refusing the bound native constants
/// `identifiers` at the entry point `context`.
pub(crate) fn native_constant_refusal(context: &str, identifiers: &str) -> String {
    format!(
        "{context}: identifier(s) {identifiers} are the canonical identifier(s) of registered \
         native constant(s), which name a value rather than a variable, so the supplied \
         binding cannot be honored; reporting UNDECIDED rather than a decision the binding did \
         not take part in"
    )
}

/// Return the items of a Python collection as `format_comma_separated_list`
/// joins them: a `str` as it is, anything else by its `repr`.
pub(crate) fn join_items(items: &Bound<'_, PyAny>) -> PyResult<String> {
    let mut texts = Vec::new();
    for item in items.try_iter()? {
        let item = item?;
        texts.push(match item.cast::<PyString>() {
            Ok(text) => text.to_str()?.to_owned(),
            Err(_not_a_string) => repr_text(&item),
        });
    }
    Ok(texts.join(", "))
}

/// The observer of one evaluation of one constraint.
pub(crate) struct LoggingObserver {
    /// The name of the constraint's class.
    kind: String,
    /// The set constraint's variable, or `None` for an equation.
    variable: Option<Py<PyAny>>,
    /// The bindings the constraint was evaluated under.
    bindings: Py<PyAny>,
    /// The set constraint's bound value, if any.
    bound_value: Option<Py<PyAny>>,
}

impl LoggingObserver {
    /// Return the observer of an evaluation of the constraint of class
    /// `kind` under `bindings`, whose set variable, if any, is `variable`,
    /// bound to `bound_value`.
    pub(crate) const fn new(
        kind: String,
        variable: Option<Py<PyAny>>,
        bindings: Py<PyAny>,
        bound_value: Option<Py<PyAny>>,
    ) -> Self {
        Self {
            kind,
            variable,
            bindings,
            bound_value,
        }
    }

    /// Return the `repr` of the set constraint's variable.
    fn variable_repr(&self, py: Python<'_>) -> String {
        self.variable
            .as_ref()
            .map_or_else(String::new, |variable| repr_text(variable.bind(py)))
    }

    /// Log `event`.
    pub(crate) fn log_event(&self, py: Python<'_>, event: &ConstraintEvent<'_>) -> PyResult<()> {
        let logger = core_logger(py)?;
        let kind = &self.kind;
        match *event {
            ConstraintEvent::Unbound { .. } => log(&logger, DEBUG, || {
                let supplied = join_items(self.bindings.bind(py))
                    .ok()
                    .filter(|text| !text.is_empty())
                    .unwrap_or_else(|| "no identifiers".to_owned());
                format!(
                    "{kind}.evaluate_with_bindings: no binding for variable {}; the bindings \
                     supplied {supplied}; reporting UNDECIDED",
                    self.variable_repr(py)
                )
            }),
            ConstraintEvent::SymbolicBinding { binding, .. } => log(&logger, DEBUG, || {
                let bound = self.bound_value.as_ref().map_or_else(
                    || render_expression_repr(binding),
                    |value| repr_text(value.bind(py)),
                );
                format!(
                    "{kind}.evaluate_with_bindings: the binding for {} is the non-literal \
                     expression {bound}; this leaf decides against a concrete value and cannot \
                     consume a symbolic one; reporting UNDECIDED",
                    self.variable_repr(py)
                )
            }),
            ConstraintEvent::BoundNativeConstants { identifiers } => log(&logger, WARNING, || {
                let names = identifiers
                    .iter()
                    .map(|identifier| format!("{identifier:?}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                native_constant_refusal(&format!("{kind}.evaluate_with_bindings"), &names)
            }),
            ConstraintEvent::Residual {
                residual,
                has_free_identifiers,
            } => {
                let (level, detail) = if has_free_identifiers {
                    (DEBUG, "free identifiers remain unbound")
                } else {
                    (WARNING, "though every free identifier was bound")
                };
                log(&logger, level, || {
                    let connector = if has_free_identifiers { "; " } else { " " };
                    format!(
                        "{kind}.evaluate_with_bindings: substituted expression {} did not reduce \
                         to a literal{connector}{detail}; reporting UNDECIDED",
                        render_expression_repr(residual)
                    )
                })
            }
            _ => Ok(()),
        }
    }
}

impl ConstraintObserver for LoggingObserver {
    fn notify(&self, event: &ConstraintEvent<'_>) {
        Python::attach(|py| {
            if let Err(error) = self.log_event(py, event) {
                record_pending_error(error);
            }
        });
    }
}
