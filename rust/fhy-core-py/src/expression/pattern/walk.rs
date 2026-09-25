//! The rewrite walk: `apply_rewrite_rules` over the core's
//! [`apply_rewrite_rules`](fhy_core::expression::pattern::apply_rewrite_rules)
//! (D-S5-10), and its errors (D-S5-11).

use std::cell::RefCell;

use pyo3::exceptions::{PyException, PyRuntimeError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyList, PyType};

use fhy_core::expression::Expression;
use fhy_core::expression::pattern::{
    CallbackError, RewriteError, RewriteRule, Rule, apply_rewrite_rules as apply_core_rules,
};

use crate::error::IntoPyErr;

use super::super::node::read_expression;
use super::kinds::{
    argument_type_error, callback_error_to_py, ensure_depth_within_recursion_limit,
    read_optional_str,
};
use super::objects::ActiveTable;
use super::rules::{PyFiredRule, PyRewriteRule, PyRuleBase, PythonRule};

/// The Python module that defines the walk's error classes.
const MODULE: &str = "fhy_core.symbolic.expression.pattern.rewrite";

/// Return the error class named `name` of the rewrite module.
fn error_class<'py>(py: Python<'py>, name: &str) -> PyResult<&'py Bound<'py, PyType>> {
    static CALLBACK: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    static REBUILD: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    let cell = if name == "RewriteCallbackError" {
        &CALLBACK
    } else {
        &REBUILD
    };
    cell.import(py, MODULE, name)
}

/// Raises the Python class of the variant, `RewriteCallbackError` or
/// `RewriteRebuildError`, with the core's text, the rule's index and name,
/// and the callback's exception or the rebuild's `ValueError` as its
/// `__cause__`. An exception that is not an `Exception`, such as
/// `KeyboardInterrupt`, is raised unchanged instead (D-S5-7).
impl IntoPyErr for RewriteError {
    fn into_py_err(self) -> PyErr {
        let message = self.to_string();
        let rule_index = self.rule_index();
        let rule_name = self.rule_name().map(str::to_owned);
        Python::attach(|py| {
            let (class_name, cause) = match self {
                RewriteError::Callback { source, .. } => {
                    let cause = callback_error_to_py(source);
                    if !cause.is_instance_of::<PyException>(py) {
                        return cause;
                    }
                    ("RewriteCallbackError", cause)
                }
                RewriteError::Rebuild { source, .. } => {
                    ("RewriteRebuildError", source.into_py_err())
                }
                _ => return PyRuntimeError::new_err(message),
            };
            let error = error_class(py, class_name)
                .and_then(|class| class.call1((message, rule_index, rule_name)));
            match error {
                Ok(error) => {
                    let error = PyErr::from_value(error);
                    error.set_cause(py, Some(cause));
                    error
                }
                Err(error) => error,
            }
        })
    }
}

/// A rule of a walk: a rewrite rule, run natively, or a Python `Rule`.
enum WalkRuleKind {
    Native(RewriteRule),
    Python(PythonRule),
}

/// A rule of a walk, which records its own firings: a rule fires when it
/// returns a replacement other than the node it was tried on, which is how
/// the core's walk decides.
struct WalkRule<'f> {
    rule_index: usize,
    kind: WalkRuleKind,
    /// The rule's name, read once per walk.
    name: Option<String>,
    fired: &'f RefCell<Vec<(usize, Option<String>)>>,
}

impl Rule for WalkRule<'_> {
    fn apply(&self, node: &Expression) -> Result<Option<Expression>, CallbackError> {
        let replacement = match &self.kind {
            WalkRuleKind::Native(rule) => rule.apply(node)?,
            WalkRuleKind::Python(rule) => rule.apply(node)?,
        };
        if replacement
            .as_ref()
            .is_some_and(|replacement| !Expression::ptr_eq(replacement, node))
        {
            self.fired
                .borrow_mut()
                .push((self.rule_index, self.name.clone()));
        }
        Ok(replacement)
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

/// Return the walk rules of the iterable `rules`.
///
/// Raises `TypeError` for an item that is not a `Rule` or a Python rule
/// whose `name` is not a `str` or `None`, and `RecursionError` for a rewrite
/// rule whose pattern is deeper than the recursion limit.
fn read_rules<'f>(
    rules: &Bound<'_, PyAny>,
    fired: &'f RefCell<Vec<(usize, Option<String>)>>,
) -> PyResult<Vec<WalkRule<'f>>> {
    let py = rules.py();
    let mut walk_rules = Vec::new();
    for (rule_index, rule) in rules.try_iter()?.enumerate() {
        let rule = rule?;
        let (kind, name) = if let Ok(native) = rule.cast::<PyRewriteRule>() {
            let native = native.get();
            ensure_depth_within_recursion_limit(py, native.depth())?;
            let name = native.rule().name().map(str::to_owned);
            (WalkRuleKind::Native(native.rule().clone()), name)
        } else if rule.is_instance_of::<PyRuleBase>() {
            let name = rule.getattr(intern!(py, "name"))?;
            let name = read_optional_str(&name, "Rule", "name")?
                .map(|name| name.to_str().map(str::to_owned))
                .transpose()?;
            (WalkRuleKind::Python(PythonRule::new(&rule)), name)
        } else {
            return Err(argument_type_error(
                "apply_rewrite_rules",
                "rules",
                "Rules",
                &rule,
            ));
        };
        walk_rules.push(WalkRule {
            rule_index,
            kind,
            name,
            fired,
        });
    }
    Ok(walk_rules)
}

/// Rewrite `expression` bottom-up once with `rules`, appending each firing
/// to the list `fired`, those before the failure after a failed walk.
fn run<'py>(
    expression: &Bound<'py, PyAny>,
    rules: &Bound<'py, PyAny>,
    fired: Option<&Bound<'py, PyList>>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = expression.py();
    let expression = read_expression(expression, "apply_rewrite_rules", "expression")?;
    let firings = RefCell::new(Vec::new());
    let walk_rules = read_rules(rules, &firings)?;
    let table = ActiveTable::enter(expression);
    let outcome = apply_core_rules(expression.get().expression(), &walk_rules);
    drop(walk_rules);
    if let Some(fired) = fired {
        for (rule_index, name) in firings.into_inner() {
            fired.append(PyFiredRule::build(py, rule_index, name.as_deref())?)?;
        }
    }
    let outcome = outcome.map_err(IntoPyErr::into_py_err)?;
    let output = if outcome.is_changed() {
        table.object_of(py, outcome.output())?
    } else {
        expression.clone().into_any()
    };
    drop(table);
    Ok(output)
}

/// Rewrite `expression` bottom-up in one pass, trying `rules` in order at
/// every node, and return the rewritten expression; append each firing, as
/// a `FiredRule`, to the list `fired` if one is given, including after a
/// failed walk the firings before the failure.
///
/// Every node's children are rewritten first, then the first rule that
/// fires replaces the node: a rule fires when it returns a replacement
/// other than the node itself. A replacement is not rewritten again, and a
/// node that occurs in several places is rewritten once. When no rule
/// fires, the result is `expression` itself; otherwise it shares every
/// node object of `expression` the walk kept.
///
/// Raises `TypeError` for an expression that is not an `Expression` or a
/// rule that is not a `Rule`, `RecursionError` for a rewrite rule whose
/// pattern is deeper than the recursion limit, `RewriteCallbackError` when
/// a callback raises an `Exception`, which is its `__cause__`, the
/// exception itself when it is not an `Exception`, and
/// `RewriteRebuildError` when a node cannot be rebuilt around its
/// rewritten children.
#[pyfunction]
#[pyo3(signature = (expression, rules, fired = None))]
pub(crate) fn apply_rewrite_rules<'py>(
    expression: &Bound<'py, PyAny>,
    rules: &Bound<'py, PyAny>,
    fired: Option<&Bound<'py, PyList>>,
) -> PyResult<Bound<'py, PyAny>> {
    run(expression, rules, fired)
}
