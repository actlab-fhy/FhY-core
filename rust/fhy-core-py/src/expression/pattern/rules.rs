//! The rules: `fhy_core._rs.RewriteRule`, backed by the Rust
//! [`RewriteRule`]; `fhy_core._rs.RuleBase`, the base of the Python `Rule`
//! ABC; and `fhy_core._rs.FiredRule`, one firing of a walk.

use std::sync::Arc;

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::types::{PyDict, PyString, PyTuple, PyType};

use fhy_core::expression::Expression;
use fhy_core::expression::pattern::RewriteRule;
use fhy_core::foreign::BoxError;

use crate::dataclass::{compare_as_dataclass, format_dataclass_repr, hash_value};
use crate::frozen::build_frozen_mutation_error;
use crate::public_class::PublicClass;

use super::super::node::{PyExpression, read_expression};
use super::bindings::PyMatchBindings;
use super::kinds::{
    PyPattern, argument_type_error, ensure_depth_within_recursion_limit, into_callback_error,
    read_optional_str, type_name,
};
use super::objects::{ActiveTable, current_adopt, current_object_of};
use crate::gc::{Slot, Slots, collect_slots};
use crate::python::Seed;

// ---------------------------------------------------------------------------
// RewriteRule
// ---------------------------------------------------------------------------

/// The fields of a rewrite rule, from which its Rust rule is built.
struct RuleFields<'py> {
    pattern: Bound<'py, PyPattern>,
    rewrite: Bound<'py, PyAny>,
    guards: Bound<'py, PyTuple>,
    name: Option<Bound<'py, PyString>>,
    is_partial: bool,
}

impl<'py> RuleFields<'py> {
    /// Return the fields of the arguments of a constructor, checked.
    ///
    /// Raises `TypeError` for a pattern that is not a `Pattern`, a rewrite
    /// or a guard that is not callable, or a name that is not a `str`.
    fn read(
        pattern: &Bound<'py, PyAny>,
        rewrite: &Bound<'py, PyAny>,
        guards: &Bound<'py, PyTuple>,
        name: &Bound<'py, PyAny>,
        is_partial: bool,
    ) -> PyResult<Self> {
        let Ok(pattern) = pattern.cast::<PyPattern>() else {
            return Err(argument_type_error(
                "RewriteRule",
                "pattern",
                "a Pattern",
                pattern,
            ));
        };
        if !rewrite.is_callable() {
            return Err(argument_type_error(
                "RewriteRule",
                "rewrite",
                "callable",
                rewrite,
            ));
        }
        for guard in guards {
            if !guard.is_callable() {
                return Err(argument_type_error(
                    "RewriteRule",
                    "guard",
                    "callable or None",
                    &guard,
                ));
            }
        }
        Ok(Self {
            pattern: pattern.clone(),
            rewrite: rewrite.clone(),
            guards: guards.clone(),
            name: read_optional_str(name, "RewriteRule", "name")?,
            is_partial,
        })
    }

    /// Return the rule's label in messages: `RewriteRule 'name'`, or
    /// `RewriteRule '<unnamed>'` for an unnamed rule.
    fn label(&self) -> PyResult<String> {
        Ok(match &self.name {
            Some(name) => format!("RewriteRule {}", name.repr()?),
            None => "RewriteRule '<unnamed>'".to_owned(),
        })
    }

    /// Return the Rust rule of these fields: the pattern, each guard in
    /// order, the rewrite and the name, with the callbacks calling the
    /// Python callables with the match's bindings object.
    ///
    /// Each callback reads its callable from a [`Slot`], which the rule
    /// object built from these fields owns.
    fn build_rule(&self) -> PyResult<RewriteRule> {
        let py = self.pattern.py();
        let captures = Arc::clone(self.pattern.get().captures(py)?);
        let rewrite = Slot::new(self.rewrite.clone().unbind());
        let is_partial = self.is_partial;
        let label = self.label()?;
        let rewrite_captures = Arc::clone(&captures);
        let mut rule =
            RewriteRule::new_partial(self.pattern.get().pattern().clone(), move |bindings| {
                Python::attach(|py| -> PyResult<Option<Expression>> {
                    let object = PyMatchBindings::build(py, bindings, &rewrite_captures)?;
                    let result = rewrite.get(py).call1((object,))?;
                    read_rewrite_result(&result, is_partial, &label)
                })
                .map_err(into_callback_error)
            });
        for guard in &self.guards {
            let guard = Slot::new(guard.unbind());
            let guard_captures = Arc::clone(&captures);
            rule = rule.with_guard(move |bindings| {
                Python::attach(|py| -> PyResult<bool> {
                    let object = PyMatchBindings::build(py, bindings, &guard_captures)?;
                    guard.get(py).call1((object,))?.is_truthy()
                })
                .map_err(into_callback_error)
            });
        }
        if let Some(name) = &self.name {
            rule = rule.with_name(name.to_str()?);
        }
        Ok(rule)
    }

    /// Return the initializer of the rule of these fields.
    fn into_rule(self) -> PyResult<PyRewriteRule> {
        let (rule, slots) = collect_slots(|| self.build_rule());
        let rule = rule?;
        let depth = self.pattern.get().depth();
        Ok(PyRewriteRule {
            rule,
            slots,
            depth,
            pattern: self.pattern.into_any().unbind(),
            rewrite: self.rewrite.unbind(),
            guards: self.guards.unbind(),
            name: self.name.map(Bound::unbind),
            is_partial: self.is_partial,
        })
    }
}

/// Return the Rust node of a rewrite's `result`, or `None` for a partial
/// rule's `None`.
///
/// Raises `TypeError` naming the rule `label` for any other result.
fn read_rewrite_result(
    result: &Bound<'_, PyAny>,
    is_partial: bool,
    label: &str,
) -> PyResult<Option<Expression>> {
    if is_partial && result.is_none() {
        return Ok(None);
    }
    match result.cast::<PyExpression>() {
        Ok(expression) => Ok(Some(current_adopt(expression)?)),
        Err(_not_an_expression) => Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "{label} rewrite must return an Expression{}, got {}.",
            if is_partial { " or None" } else { "" },
            type_name(result)
        ))),
    }
}

/// The contents of a rule the binding builds, handed to the public class's
/// constructor. Not exported.
#[pyclass(frozen, module = "fhy_core._rs", name = "_RewriteRuleSeed")]
struct RewriteRuleSeed(Seed<PyRewriteRule>);

/// A pattern paired with a rewrite of what it matches, optionally guarded
/// and named, backed by the Rust [`RewriteRule`].
///
/// A rule fires on an expression when its pattern matches it at the root,
/// every guard returns a true value for the match's bindings, in order, and
/// the rewrite returns a replacement other than the expression itself.
/// Rules compare and hash by identity, since callables cannot be compared.
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "RewriteRule")]
pub(crate) struct PyRewriteRule {
    rule: RewriteRule,
    /// The slots of the rule's callbacks, which this object owns.
    slots: Slots,
    /// The depth of the pattern.
    depth: usize,
    /// The pattern.
    #[pyo3(get)]
    pattern: Py<PyAny>,
    /// The rewrite callable.
    #[pyo3(get)]
    rewrite: Py<PyAny>,
    /// The guard callables, in order.
    #[pyo3(get)]
    guards: Py<PyTuple>,
    /// The name, or `None`.
    #[pyo3(get)]
    name: Option<Py<PyString>>,
    /// Whether the rewrite may return `None` to decline.
    is_partial: bool,
}

impl PyRewriteRule {
    /// Return the public Python class registered for this class.
    fn public_class() -> &'static PublicClass {
        static PUBLIC_CLASS: PublicClass = PublicClass::new("RewriteRule");
        &PUBLIC_CLASS
    }

    /// Return the Rust rule.
    pub(super) fn rule(&self) -> &RewriteRule {
        &self.rule
    }

    /// Return the depth of the pattern.
    pub(super) fn depth(&self) -> usize {
        self.depth
    }

    /// Return the tuple of `guard`, or the empty tuple for `None`.
    fn guard_tuple<'py>(guard: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyTuple>> {
        let py = guard.py();
        if guard.is_none() {
            Ok(PyTuple::empty(py))
        } else {
            PyTuple::new(py, [guard])
        }
    }

    /// Return a rule of the public class `cls` holding `rule`.
    fn build_object<'py>(
        cls: &Bound<'py, PyType>,
        rule: PyRewriteRule,
    ) -> PyResult<Bound<'py, PyAny>> {
        cls.call1((RewriteRuleSeed(Seed::new(rule)),))
    }

    /// Return the fields of this rule.
    fn fields<'py>(&self, py: Python<'py>) -> PyResult<RuleFields<'py>> {
        Ok(RuleFields {
            pattern: self.pattern.bind(py).cast::<PyPattern>()?.clone(),
            rewrite: self.rewrite.bind(py).clone(),
            guards: self.guards.bind(py).clone(),
            name: self.name.as_ref().map(|name| name.bind(py).clone()),
            is_partial: self.is_partial,
        })
    }
}

#[pymethods]
impl PyRewriteRule {
    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.pattern)?;
        visit.call(&self.rewrite)?;
        visit.call(&self.guards)?;
        visit.call(self.name.as_ref())?;
        self.slots.traverse(&visit)
    }

    /// Create the rule rewriting what `pattern` matches with the callable
    /// `rewrite`, which must return an `Expression`, when the callable
    /// `guard`, if given, allows it, named `name`.
    ///
    /// Raises `TypeError` for a pattern that is not a `Pattern`, a rewrite
    /// or guard that is not callable, or a name that is not a `str`.
    #[new]
    #[pyo3(signature = (pattern, rewrite = None, guard = None, name = None))]
    fn new(
        pattern: &Bound<'_, PyAny>,
        rewrite: Option<&Bound<'_, PyAny>>,
        guard: Option<&Bound<'_, PyAny>>,
        name: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let py = pattern.py();
        if let Ok(seed) = pattern.cast::<RewriteRuleSeed>() {
            return seed.get().0.take("a rewrite rule");
        }
        let Some(rewrite) = rewrite else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "RewriteRule() missing required argument: 'rewrite'",
            ));
        };
        let none = py.None().into_bound(py);
        let guards = Self::guard_tuple(guard.unwrap_or(&none))?;
        RuleFields::read(pattern, rewrite, &guards, name.unwrap_or(&none), false)?.into_rule()
    }

    /// Return the rule of `cls` whose `rewrite` may return `None` to
    /// decline, and otherwise as the constructor builds it.
    #[classmethod]
    #[pyo3(signature = (pattern, rewrite, guard = None, name = None))]
    fn new_partial<'py>(
        cls: &Bound<'py, PyType>,
        pattern: &Bound<'py, PyAny>,
        rewrite: &Bound<'py, PyAny>,
        guard: Option<&Bound<'py, PyAny>>,
        name: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = cls.py();
        let none = py.None().into_bound(py);
        let guards = Self::guard_tuple(guard.unwrap_or(&none))?;
        let rule = RuleFields::read(pattern, rewrite, &guards, name.unwrap_or(&none), true)?
            .into_rule()?;
        Self::build_object(cls, rule)
    }

    /// Return the rule of `cls` of the fields `pattern`, `rewrite`,
    /// `guards`, `name` and `is_partial`, the pickle of a rule.
    #[classmethod]
    fn _from_fields<'py>(
        cls: &Bound<'py, PyType>,
        pattern: &Bound<'py, PyAny>,
        rewrite: &Bound<'py, PyAny>,
        guards: &Bound<'py, PyAny>,
        name: &Bound<'py, PyAny>,
        is_partial: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let guards = crate::dataclass::collect_tuple(guards)?;
        let rule = RuleFields::read(pattern, rewrite, &guards, name, is_partial)?.into_rule()?;
        Self::build_object(cls, rule)
    }

    /// Return this rule with `guard` added after its other guards.
    ///
    /// Raises `TypeError` for a guard that is not callable.
    fn with_guard<'py>(
        slf: &Bound<'py, Self>,
        guard: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let this = slf.get();
        if !guard.is_callable() {
            return Err(argument_type_error(
                "RewriteRule",
                "guard",
                "callable",
                guard,
            ));
        }
        let mut guards: Vec<Bound<'py, PyAny>> = this.guards.bind(py).iter().collect();
        guards.push(guard.clone());
        let mut fields = this.fields(py)?;
        fields.guards = PyTuple::new(py, guards)?;
        Self::build_object(&slf.get_type(), fields.into_rule()?)
    }

    /// Return this rule named `name`, replacing any earlier name.
    ///
    /// Raises `TypeError` for a name that is not a `str`.
    fn with_name<'py>(
        slf: &Bound<'py, Self>,
        name: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let Ok(name) = name.cast::<PyString>() else {
            return Err(argument_type_error("RewriteRule", "name", "a str", name));
        };
        let mut fields = slf.get().fields(py)?;
        fields.name = Some(name.clone());
        Self::build_object(&slf.get_type(), fields.into_rule()?)
    }

    /// Try the rule once at the root of `expression`, and return the
    /// replacement, or `None` if the pattern does not match, a guard
    /// refuses, or the rewrite declines or returns `expression` itself.
    ///
    /// Raises `TypeError` for a value that is not an expression or a
    /// rewrite result that is not one, `RecursionError` for a pattern
    /// deeper than the recursion limit, and the exception a predicate, a
    /// guard or the rewrite raises, unchanged.
    fn apply<'py>(&self, expression: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let py = expression.py();
        let expression = read_expression(expression, "RewriteRule.apply", "expression")?;
        ensure_depth_within_recursion_limit(py, self.depth)?;
        let table = ActiveTable::enter(expression);
        let replacement = self
            .rule
            .apply(expression.get().expression())
            .map_err(crate::exceptions::boxed_error_to_py)?;
        let result = match replacement {
            Some(replacement) => Some(table.object_of(py, &replacement)?),
            None => None,
        };
        drop(table);
        Ok(result)
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let this = slf.get();
        let name = match &this.name {
            Some(name) => name.bind(py).clone().into_any(),
            None => py.None().into_bound(py),
        };
        format_dataclass_repr(
            &slf.get_type(),
            &[("pattern", this.pattern.bind(py)), ("name", &name)],
        )
    }

    /// Pickle as `_from_fields` of the rule's class, which pickles only if
    /// its callables do.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        let this = slf.get();
        let constructor = slf.get_type().getattr(intern!(py, "_from_fields"))?;
        let name = match &this.name {
            Some(name) => name.bind(py).clone().into_any(),
            None => py.None().into_bound(py),
        };
        let arguments = (
            this.pattern.bind(py),
            this.rewrite.bind(py),
            this.guards.bind(py),
            name,
            this.is_partial,
        )
            .into_pyobject(py)?;
        Ok((constructor, arguments))
    }

    /// Always true: rules are immutable.
    #[getter]
    fn is_frozen(_slf: &Bound<'_, Self>) -> bool {
        true
    }

    /// Do nothing: rules are always frozen.
    fn freeze(_slf: &Bound<'_, Self>) {}

    /// Do nothing: rules are always frozen, and mutating one raises.
    fn assert_frozen(_slf: &Bound<'_, Self>) {}

    fn __setattr__(slf: &Bound<'_, Self>, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let _ = value;
        Err(build_frozen_mutation_error(slf, "modify", name)?)
    }

    fn __delattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
        Err(build_frozen_mutation_error(slf, "delete", name)?)
    }

    /// Register `cls` as the public class.
    ///
    /// Raises `RuntimeError` if another public class is registered.
    #[classmethod]
    fn _register_public_class(cls: &Bound<'_, PyType>) -> PyResult<()> {
        Self::public_class().register(cls)
    }
}

// ---------------------------------------------------------------------------
// RuleBase and Python rules
// ---------------------------------------------------------------------------

/// The base of the Python `Rule` ABC: a rewrite tried at the root of one
/// expression, implemented in Python.
///
/// It takes any constructor arguments, so a Python subclass defines its own
/// `__init__`. A walk calls the subclass's `apply` from Rust.
#[pyclass(subclass, module = "fhy_core._rs", name = "RuleBase")]
pub(crate) struct PyRuleBase;

#[pymethods]
impl PyRuleBase {
    #[new]
    #[pyo3(signature = (*args, **kwargs))]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        let _ = (args, kwargs);
        Self
    }
}

/// A Python `Rule` object, as a Rust rule's callback: its `apply` called
/// with a node's object.
pub(super) struct PythonRule {
    object: Py<PyAny>,
}

impl PythonRule {
    /// Return the rule of the Python `Rule` object `object`.
    pub(super) fn new(object: &Bound<'_, PyAny>) -> Self {
        Self {
            object: object.clone().unbind(),
        }
    }

    /// Return the replacement the rule's `apply` returns for `node`, or
    /// `None` when it declines.
    ///
    /// # Errors
    ///
    /// Returns the exception `apply` raises, and a `TypeError` for a result
    /// that is neither an `Expression` nor `None`.
    pub(super) fn apply(&self, node: &Expression) -> Result<Option<Expression>, BoxError> {
        Python::attach(|py| -> PyResult<Option<Expression>> {
            let object = current_object_of(py, node)?;
            let rule = self.object.bind(py);
            let result = rule.call_method1(intern!(py, "apply"), (object,))?;
            if result.is_none() {
                return Ok(None);
            }
            match result.cast::<PyExpression>() {
                Ok(expression) => Ok(Some(current_adopt(expression)?)),
                Err(_not_an_expression) => Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "{}.apply must return an Expression or None, got {}.",
                    rule.get_type().qualname()?,
                    type_name(&result)
                ))),
            }
        })
        .map_err(into_callback_error)
    }
}

// ---------------------------------------------------------------------------
// FiredRule
// ---------------------------------------------------------------------------

/// One firing of a rule during a rewrite walk: the position of the rule in
/// the rule list, and its name.
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "FiredRule")]
pub(crate) struct PyFiredRule {
    /// The position of the rule in the rule list.
    #[pyo3(get)]
    rule_index: usize,
    /// The rule's name, or `None`.
    #[pyo3(get)]
    name: Option<Py<PyString>>,
    /// The name, for equality and hashing.
    text: Option<String>,
}

impl PyFiredRule {
    /// Return the public Python class registered for this class.
    fn public_class() -> &'static PublicClass {
        static PUBLIC_CLASS: PublicClass = PublicClass::new("FiredRule");
        &PUBLIC_CLASS
    }

    /// Return the firing of the rule at `rule_index` named `name`, of the
    /// public class.
    pub(super) fn build<'py>(
        py: Python<'py>,
        rule_index: usize,
        name: Option<&str>,
    ) -> PyResult<Bound<'py, PyAny>> {
        Self::public_class().get(py)?.call1((rule_index, name))
    }
}

#[pymethods]
impl PyFiredRule {
    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(self.name.as_ref())?;
        Ok(())
    }

    /// Create the firing of the rule at `rule_index` named `name`.
    ///
    /// Raises `TypeError` for a name that is not a `str` or `None`, and
    /// `OverflowError` or `TypeError` for an index that is not a
    /// non-negative `int`.
    #[new]
    fn new(rule_index: usize, name: &Bound<'_, PyAny>) -> PyResult<Self> {
        let name = read_optional_str(name, "FiredRule", "name")?;
        let text = name
            .as_ref()
            .map(|name| name.to_str().map(str::to_owned))
            .transpose()?;
        Ok(Self {
            rule_index,
            name: name.map(Bound::unbind),
            text,
        })
    }

    /// Return the text of the diagnostic the rewrite pass reports for the
    /// firing, `applied rewrite rule "name"`, the name escaped as Rust's
    /// `Debug` writes a string, or `None` for an unnamed rule.
    fn _diagnostic_message(&self) -> Option<String> {
        self.text
            .as_ref()
            .map(|name| format!("applied rewrite rule {name:?}"))
    }

    fn __eq__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        compare_as_dataclass(slf, other, |this, other| {
            Ok(this.rule_index == other.rule_index && this.text == other.text)
        })
    }

    fn __hash__(&self) -> u64 {
        hash_value(&(self.rule_index, &self.text))
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let this = slf.get();
        let index = this.rule_index.into_pyobject(py)?.into_any();
        let name = match &this.name {
            Some(name) => name.bind(py).clone().into_any(),
            None => py.None().into_bound(py),
        };
        format_dataclass_repr(&slf.get_type(), &[("rule_index", &index), ("name", &name)])
    }

    /// Pickle as a call of the class with the fields.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyType>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        let this = slf.get();
        let arguments = (
            this.rule_index,
            this.name.as_ref().map(|name| name.bind(py)),
        )
            .into_pyobject(py)?;
        Ok((slf.get_type(), arguments))
    }

    /// Always true: firings are immutable.
    #[getter]
    fn is_frozen(_slf: &Bound<'_, Self>) -> bool {
        true
    }

    /// Do nothing: firings are always frozen.
    fn freeze(_slf: &Bound<'_, Self>) {}

    /// Do nothing: firings are always frozen, and mutating one raises.
    fn assert_frozen(_slf: &Bound<'_, Self>) {}

    fn __setattr__(slf: &Bound<'_, Self>, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let _ = value;
        Err(build_frozen_mutation_error(slf, "modify", name)?)
    }

    fn __delattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
        Err(build_frozen_mutation_error(slf, "delete", name)?)
    }

    /// Register `cls` as the public class.
    ///
    /// Raises `RuntimeError` if another public class is registered.
    #[classmethod]
    fn _register_public_class(cls: &Bound<'_, PyType>) -> PyResult<()> {
        Self::public_class().register(cls)
    }
}
