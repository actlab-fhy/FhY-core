//! Keeping SymPy's Boolean positions Boolean: the piecewise expansion, the
//! comparison of Booleans, and case conditions.
//!
//! A `sympy.Piecewise` is not a SymPy `Boolean` even when every branch is
//! Boolean, so SymPy's `And`, `Or` and comparisons mishandle one. These
//! helpers rewrite such a piecewise wherever SymPy needs a Boolean.

use std::sync::{Arc, Mutex, PoisonError};

use pyo3::exceptions::PyRecursionError;
use pyo3::prelude::*;
use pyo3::types::{PyCFunction, PyDict, PyTuple};

use super::error::SympyErrorKind;
use super::load::Handles;

/// How deeply the recursive helpers may nest, as Python's recursion limit
/// bounds a Python walk: past it they raise `RecursionError` rather than
/// exhaust the Rust stack.
pub(super) const MAX_NESTING: usize = 1_000;

/// A failure of a helper: an exception, or a failure the backend reports
/// as its own kind.
pub(super) type Fallible<T> = Result<T, SympyErrorKind>;

impl From<PyErr> for SympyErrorKind {
    fn from(error: PyErr) -> Self {
        Self::Python(error)
    }
}

/// Return the `RecursionError` of a helper nested past [`MAX_NESTING`].
pub(super) fn too_deep() -> SympyErrorKind {
    SympyErrorKind::Python(PyRecursionError::new_err(
        "maximum recursion depth exceeded in the sympy backend",
    ))
}

/// Return whether `value` is a SymPy Boolean other than a bare symbol: a
/// literal, a relational or a connective.
pub(super) fn is_boolean_node(handles: &Handles, value: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = value.py();
    Ok(value.is_instance(handles.boolean.bind(py))? && !is_symbol(value)?)
}

/// Return whether `value` is a symbol.
pub(super) fn is_symbol(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    match value.getattr("is_Symbol") {
        Ok(flag) => flag.is_truthy(),
        Err(_) => Ok(false),
    }
}

/// Return whether `value` is a piecewise one branch of which shows it is
/// Boolean.
fn is_boolean_valued_piecewise(
    handles: &Handles,
    value: &Bound<'_, PyAny>,
    depth: usize,
) -> Fallible<bool> {
    let py = value.py();
    if !value.is_instance(handles.piecewise.bind(py))? {
        return Ok(false);
    }
    if depth > MAX_NESTING {
        return Err(too_deep());
    }
    for branch in value.getattr("args")?.try_iter()? {
        let branch_value = branch?.get_item(0)?;
        if is_boolean_node(handles, &branch_value)?
            || is_boolean_valued_piecewise(handles, &branch_value, depth + 1)?
        {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Return whether an `Eq` or `Ne` of `left` and `right` compares Booleans.
pub(super) fn is_boolean_comparison(
    handles: &Handles,
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
) -> Fallible<bool> {
    for operand in [left, right] {
        if is_boolean_node(handles, operand)? || is_boolean_valued_piecewise(handles, operand, 0)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Return `operand` in a form SymPy's Boolean operators handle.
///
/// A piecewise becomes its first-match expansion, `(condition & value) |
/// (~condition & rest)` over its branches in order, ending at the value of
/// the first branch whose condition is `True`; a branch value is rewritten
/// the same way. Any other operand is returned as it stands. A piecewise
/// with no `True` condition has no value where every condition fails, and
/// is refused.
pub(super) fn to_boolean<'py>(
    handles: &Handles,
    operand: &Bound<'py, PyAny>,
) -> Fallible<Bound<'py, PyAny>> {
    to_boolean_at(handles, operand, 0)
}

fn to_boolean_at<'py>(
    handles: &Handles,
    operand: &Bound<'py, PyAny>,
    depth: usize,
) -> Fallible<Bound<'py, PyAny>> {
    let py = operand.py();
    if !operand.is_instance(handles.piecewise.bind(py))? {
        return Ok(operand.clone());
    }
    if depth > MAX_NESTING {
        return Err(too_deep());
    }
    let true_value = handles.true_value.bind(py);
    let mut reachable: Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)> = Vec::new();
    let mut is_total = false;
    for branch in operand.getattr("args")?.try_iter()? {
        let branch = branch?;
        let value = to_boolean_at(handles, &branch.get_item(0)?, depth + 1)?;
        let value = handles.as_boolean.bind(py).call1((value,))?;
        let condition = branch.get_item(1)?;
        let is_last = condition.is(true_value);
        reachable.push((value, condition));
        if is_last {
            is_total = true;
            break;
        }
    }
    if !is_total {
        return Err(SympyErrorKind::PartialPiecewise(
            operand.repr()?.to_string(),
        ));
    }
    let (mut result, _) = reachable.pop().expect("a total piecewise has a branch");
    let and = handles.and.bind(py);
    let or = handles.or.bind(py);
    let not = handles.not.bind(py);
    while let Some((value, condition)) = reachable.pop() {
        let taken = and.call1((&condition, value))?;
        let skipped = and.call1((not.call1((condition,))?, result))?;
        result = or.call1((taken, skipped))?;
    }
    Ok(result)
}

/// Return the operands of an `Eq` or `Ne` with a Boolean piecewise made a
/// Boolean, when the comparison is between Booleans; a numeric comparison's
/// operands are returned as they stand.
pub(super) fn comparison_operands<'py>(
    handles: &Handles,
    left: &Bound<'py, PyAny>,
    right: &Bound<'py, PyAny>,
) -> Fallible<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    if !is_boolean_comparison(handles, left, right)? {
        return Ok((left.clone(), right.clone()));
    }
    Ok((to_boolean(handles, left)?, to_boolean(handles, right)?))
}

/// The failure a hook of a SymPy `replace` walk left behind, which it
/// reports by raising the prelude's `AbortWalk`.
struct HookFailure<E>(Mutex<Option<E>>);

impl<E: From<PyErr>> HookFailure<E> {
    /// Keep `failure`, and return the exception that stops the walk.
    fn abort(&self, handles: &Handles, py: Python<'_>, failure: E) -> PyErr {
        *self.0.lock().unwrap_or_else(PoisonError::into_inner) = Some(failure);
        match handles.abort_walk.bind(py).call0() {
            Ok(error) => PyErr::from_value(error),
            Err(error) => error,
        }
    }

    /// Return the failure of a walk that ended with `error`: the kept one
    /// if a hook stopped it, and `error` otherwise.
    fn take_failure(&self, error: PyErr) -> E {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
            .unwrap_or_else(|| E::from(error))
    }
}

/// Return `expression.replace(query, value, simultaneous=False)` with Rust
/// hooks: SymPy's own bottom-up walk, so its traversal rules hold.
///
/// A hook failing stops the walk, and the walk fails with the hook's
/// failure.
pub(super) fn replace<'py, E, Q, V>(
    handles: &Arc<Handles>,
    expression: &Bound<'py, PyAny>,
    query: Q,
    value: V,
) -> Result<Bound<'py, PyAny>, E>
where
    E: From<PyErr> + Send + 'static,
    Q: Fn(&Handles, &Bound<'_, PyAny>) -> Result<bool, E> + Send + Sync + 'static,
    V: for<'a> Fn(&Handles, &Bound<'a, PyAny>) -> Result<Bound<'a, PyAny>, E>
        + Send
        + Sync
        + 'static,
{
    let py = expression.py();
    let failure: Arc<HookFailure<E>> = Arc::new(HookFailure(Mutex::new(None)));
    let query = {
        let handles = Arc::clone(handles);
        let failure = Arc::clone(&failure);
        PyCFunction::new_closure(
            py,
            None,
            None,
            move |args: &Bound<'_, PyTuple>,
                  _kwargs: Option<&Bound<'_, PyDict>>|
                  -> PyResult<bool> {
                let node = args.get_item(0)?;
                query(&handles, &node).map_err(|kind| failure.abort(&handles, args.py(), kind))
            },
        )?
    };
    let value = {
        let handles = Arc::clone(handles);
        let failure = Arc::clone(&failure);
        PyCFunction::new_closure(
            py,
            None,
            None,
            move |args: &Bound<'_, PyTuple>,
                  _kwargs: Option<&Bound<'_, PyDict>>|
                  -> PyResult<Py<PyAny>> {
                let node = args.get_item(0)?;
                value(&handles, &node)
                    .map(Bound::unbind)
                    .map_err(|kind| failure.abort(&handles, args.py(), kind))
            },
        )?
    };
    let keywords = PyDict::new(py);
    keywords.set_item("simultaneous", false)?;
    expression
        .call_method("replace", (query, value), Some(&keywords))
        .map_err(|error| failure.take_failure(error))
}

/// Return whether `node` is an `Eq` or `Ne` of a symbol and a relational.
fn is_symbol_compared_with_relational(
    handles: &Handles,
    node: &Bound<'_, PyAny>,
) -> Fallible<bool> {
    let py = node.py();
    if !(node.is_instance(handles.equality.bind(py))?
        || node.is_instance(handles.unequality.bind(py))?)
    {
        return Ok(false);
    }
    let mut has_symbol = false;
    let mut has_relational = false;
    for operand in node.getattr("args")?.try_iter()? {
        let operand = operand?;
        has_symbol |= is_symbol(&operand)?;
        has_relational |= operand.is_instance(handles.relational.bind(py))?;
    }
    Ok(has_symbol && has_relational)
}

/// Return `comparison` with its symbol negated and its relation inverted:
/// `b == (x < 1)` holds exactly when `~b != (x < 1)` does.
fn negate_symbol_side<'a>(
    handles: &Handles,
    comparison: &Bound<'a, PyAny>,
) -> Fallible<Bound<'a, PyAny>> {
    let py = comparison.py();
    let inverted = if comparison.is_instance(handles.equality.bind(py))? {
        handles.unequality.bind(py)
    } else {
        handles.equality.bind(py)
    };
    let mut operands = Vec::new();
    for operand in comparison.getattr("args")?.try_iter()? {
        let operand = operand?;
        operands.push(if is_symbol(&operand)? {
            handles.not.bind(py).call1((operand,))?
        } else {
            operand
        });
    }
    Ok(inverted.call1(PyTuple::new(py, operands)?)?)
}

/// Return a piecewise branch condition SymPy can evaluate a piecewise
/// over.
///
/// A comparison of a Boolean symbol with a relational, such as `Eq(b, x <
/// 1)`, is rewritten to `Ne(~b, x < 1)`, which has no side SymPy subtracts
/// from the other. A condition holding a piecewise is folded, and then
/// rewritten as a Boolean, which leaves SymPy nothing to rewrite with its
/// `ITE` routes, which drop branches.
pub(super) fn condition<'py>(
    handles: &Arc<Handles>,
    condition: &Bound<'py, PyAny>,
) -> Fallible<Bound<'py, PyAny>> {
    let py = condition.py();
    let basic = handles.basic.bind(py);
    let mut condition = condition.clone();
    if condition.is_instance(basic)? {
        condition = replace::<SympyErrorKind, _, _>(
            handles,
            &condition,
            is_symbol_compared_with_relational,
            negate_symbol_side,
        )?;
    }
    if condition.is_instance(basic)?
        && condition
            .call_method1("has", (handles.piecewise.bind(py),))?
            .is_truthy()?
    {
        condition = handles
            .sympy
            .bind(py)
            .getattr("piecewise_fold")?
            .call1((condition,))?;
    }
    to_boolean(handles, &condition)
}

/// Return the new arguments `arguments` of `node` with each Boolean
/// position made Boolean: every argument of a Boolean connective, a
/// piecewise branch's condition, and both operands of an `Eq` or `Ne` that
/// compares Booleans.
pub(super) fn boolean_positions<'py>(
    handles: &Arc<Handles>,
    node: &Bound<'py, PyAny>,
    arguments: Vec<Bound<'py, PyAny>>,
) -> Fallible<Vec<Bound<'py, PyAny>>> {
    let py = node.py();
    if (node.is_instance(handles.equality.bind(py))?
        || node.is_instance(handles.unequality.bind(py))?)
        && arguments.len() == 2
    {
        let (left, right) = comparison_operands(handles, &arguments[0], &arguments[1])?;
        return Ok(vec![left, right]);
    }
    if node.is_instance(handles.boolean_function.bind(py))? {
        return arguments
            .iter()
            .map(|argument| to_boolean(handles, argument))
            .collect();
    }
    if node.is_instance(handles.expr_cond_pair.bind(py))? && arguments.len() == 2 {
        let converted = condition(handles, &arguments[1])?;
        return Ok(vec![arguments[0].clone(), converted]);
    }
    Ok(arguments)
}
