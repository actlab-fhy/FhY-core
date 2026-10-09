//! Simplifying a SymPy object with `sympy.simplify`, around the cases
//! SymPy mishandles, best-effort.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use super::boolean::{Fallible, MAX_NESTING, is_boolean_comparison, replace, to_boolean, too_deep};
use super::error::SympyErrorKind;
use super::load::Handles;
use super::substitute::{Visit, rebuild_bottom_up, substitute_symbols};

/// Why a simplification gives up and keeps its input.
enum GiveUp {
    /// A relational holding a piecewise that must be folded out of it
    /// compares a value that is not an extended real.
    Unfoldable,
}

/// Return `expression` simplified, or `None` to keep it as it stands.
///
/// SymPy can fail to produce a form to lift while its input stays correct:
/// `sympy.simplify` raises `PrecisionExhausted` when a `floor`'s argument
/// is exactly an integer at a point it checks numerically; it drops a
/// piecewise's final `True` branch when the earlier conditions cover every
/// real, which leaves a partial piecewise; and a comparison of a piecewise
/// that must be split per branch cannot be simplified when a branch is one
/// no comparison takes. Each case returns `None`; every other failure is
/// returned.
pub(super) fn try_simplify<'py>(
    handles: &Arc<Handles>,
    expression: &Bound<'py, PyAny>,
) -> Fallible<Option<Bound<'py, PyAny>>> {
    let py = expression.py();
    let simplified = match transform_masking(handles, expression, 0) {
        Ok(simplified) => simplified,
        Err(Failure::GiveUp(GiveUp::Unfoldable)) => return Ok(None),
        Err(Failure::Kind(SympyErrorKind::Python(error)))
            if error.is_instance(py, handles.precision_exhausted.bind(py)) =>
        {
            return Ok(None);
        }
        Err(Failure::Kind(kind)) => return Err(kind),
    };
    if handles
        .holds_partial_piecewise
        .bind(py)
        .call1((&simplified,))?
        .is_truthy()?
    {
        return Ok(None);
    }
    Ok(Some(simplified))
}

/// A failure inside the simplification: an error, or a reason to give up.
enum Failure {
    Kind(SympyErrorKind),
    GiveUp(GiveUp),
}

impl From<SympyErrorKind> for Failure {
    fn from(kind: SympyErrorKind) -> Self {
        Self::Kind(kind)
    }
}

impl From<PyErr> for Failure {
    fn from(error: PyErr) -> Self {
        Self::Kind(SympyErrorKind::Python(error))
    }
}

/// Return [`fold_and_simplify`] of `expression` with each comparison
/// between Booleans held opaque.
///
/// `sympy.simplify`'s rules for a conjunction subtract or negate a
/// comparison's sides, which raises for a comparison between Booleans with
/// a side such as `True` or `x < 1`. Each such comparison is simplified on
/// its own operands and held as an opaque dummy symbol while the rest is
/// simplified, then put back.
fn transform_masking<'py>(
    handles: &Arc<Handles>,
    expression: &Bound<'py, PyAny>,
    depth: usize,
) -> Result<Bound<'py, PyAny>, Failure> {
    let py = expression.py();
    if depth > MAX_NESTING {
        return Err(too_deep().into());
    }
    if !expression.is_instance(handles.basic.bind(py))? {
        return fold_and_simplify(handles, expression);
    }
    let comparisons = PyDict::new(py);
    let (masked, _) = rebuild_bottom_up::<Failure>(
        handles,
        expression,
        |node| {
            let is_comparison = node.is_instance(handles.equality.bind(py))?
                || node.is_instance(handles.unequality.bind(py))?;
            if !is_comparison {
                return Ok(Visit::Descend);
            }
            let arguments: Vec<Bound<'py, PyAny>> =
                node.getattr("args")?.try_iter()?.collect::<PyResult<_>>()?;
            if arguments.len() != 2
                || !is_boolean_comparison(handles, &arguments[0], &arguments[1])?
            {
                return Ok(Visit::Descend);
            }
            let mut operands = Vec::with_capacity(2);
            for argument in &arguments {
                operands.push(transform_masking(handles, argument, depth + 1)?);
            }
            let comparison = node.getattr("func")?.call1(PyTuple::new(py, operands)?)?;
            let still_a_comparison = comparison.is_instance(handles.equality.bind(py))?
                || comparison.is_instance(handles.unequality.bind(py))?;
            if !still_a_comparison {
                let changed = !comparison.is(node);
                return Ok(Visit::Replace(comparison, changed));
            }
            let placeholder = handles.dummy.bind(py).call1(("boolean_comparison",))?;
            comparisons.set_item(&placeholder, comparison)?;
            Ok(Visit::Replace(placeholder, true))
        },
        |node, arguments, changed| {
            if !changed {
                return Ok((node.clone(), false));
            }
            let rebuilt = node.getattr("func")?.call1(PyTuple::new(py, arguments)?)?;
            Ok((rebuilt, true))
        },
    )?;
    let simplified = fold_and_simplify(handles, &masked)?;
    if comparisons.is_empty() {
        return Ok(simplified);
    }
    Ok(substitute_symbols(
        handles,
        &simplified,
        comparisons.as_any(),
    )?)
}

/// Return `sympy.simplify` of `expression`, with the piecewise nodes it
/// mishandles folded first, and each plain `Piecewise` of the result made
/// parity-opaque.
///
/// `sympy.simplify` is read from the module at each call.
fn fold_and_simplify<'py>(
    handles: &Arc<Handles>,
    expression: &Bound<'py, PyAny>,
) -> Result<Bound<'py, PyAny>, Failure> {
    let py = expression.py();
    let piecewise = handles.piecewise.bind(py);
    let has_piecewise = |object: &Bound<'py, PyAny>| -> PyResult<bool> {
        Ok(object.is_instance(handles.basic.bind(py))?
            && object.call_method1("has", (piecewise,))?.is_truthy()?)
    };
    let mut folded = expression.clone();
    // Without a piecewise, neither fold matches a node, and SymPy's
    // `replace` would return the expression itself.
    if has_piecewise(&folded)? {
        folded = fold_integer_parts(handles, &folded)?;
        folded = fold_relationals(handles, &folded)?;
    }
    let simplified = handles
        .sympy
        .bind(py)
        .getattr("simplify")?
        .call1((folded,))?;
    if has_piecewise(&simplified)? {
        return Ok(handles
            .hide_piecewise_parity
            .bind(py)
            .call1((simplified,))?);
    }
    Ok(simplified)
}

/// Return `expression` with each `Mod`, `floor` and `ceiling` holding a
/// piecewise folded into the piecewise's branches.
///
/// `sympy.simplify` rebuilds an integer part around its simplified
/// argument, and simplifying an argument that holds a piecewise folds it
/// into a plain `Piecewise`, whose all-even branches make SymPy misjudge
/// the quotient's integrality. Folding first applies the integer part to
/// each branch value alone.
fn fold_integer_parts<'py>(
    handles: &Arc<Handles>,
    expression: &Bound<'py, PyAny>,
) -> Result<Bound<'py, PyAny>, Failure> {
    replace::<Failure, _, _>(
        handles,
        expression,
        |handles, node| {
            let py = node.py();
            let is_integer_part = node.is_instance(handles.modulo.bind(py))?
                || node.is_instance(handles.floor.bind(py))?
                || node.is_instance(handles.ceiling.bind(py))?;
            Ok(is_integer_part
                && node
                    .call_method1("has", (handles.piecewise.bind(py),))?
                    .is_truthy()?)
        },
        |handles, node| {
            let py = node.py();
            let folded = handles
                .sympy
                .bind(py)
                .getattr("piecewise_fold")?
                .call1((node,))?;
            Ok(handles.hide_piecewise_parity.bind(py).call1((folded,))?)
        },
    )
}

/// Return `expression` with each relational `sympy.simplify` cannot take
/// folded.
///
/// `sympy.simplify` checks whether a relational's sides are equal by
/// substituting a product of dummy symbols for each free symbol, so a
/// piecewise inside the relational with a bare symbol in a Boolean position
/// of a condition receives a product there and raises. Folding the
/// piecewise out of such a relational leaves no condition inside it.
fn fold_relationals<'py>(
    handles: &Arc<Handles>,
    expression: &Bound<'py, PyAny>,
) -> Result<Bound<'py, PyAny>, Failure> {
    replace::<Failure, _, _>(
        handles,
        expression,
        |handles, relational| Ok(must_fold_piecewise_out_of(handles, relational)?),
        |handles, relational| {
            let py = relational.py();
            let folded = handles
                .sympy
                .bind(py)
                .getattr("piecewise_fold")?
                .call1((relational,));
            match folded {
                Ok(folded) => Ok(to_boolean(handles, &folded)?),
                Err(error) if error.is_instance_of::<pyo3::exceptions::PyTypeError>(py) => {
                    Err(Failure::GiveUp(GiveUp::Unfoldable))
                }
                Err(error) => Err(error.into()),
            }
        },
    )
}

/// Return whether `relational` holds a piecewise with a bare symbol in a
/// Boolean position of a condition, which `sympy.simplify` cannot compare.
fn must_fold_piecewise_out_of(handles: &Handles, relational: &Bound<'_, PyAny>) -> Fallible<bool> {
    let py = relational.py();
    if !relational.is_instance(handles.relational.bind(py))? {
        return Ok(false);
    }
    let piecewises = relational.call_method1("atoms", (handles.piecewise.bind(py),))?;
    for piecewise in piecewises.try_iter()? {
        for branch in piecewise?.getattr("args")?.try_iter()? {
            if holds_boolean_symbol_position(handles, &branch?.get_item(1)?)? {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// Return whether a bare symbol stands where `condition` needs a Boolean:
/// `condition` itself, or an argument of a Boolean connective inside it.
fn holds_boolean_symbol_position(
    handles: &Handles,
    condition: &Bound<'_, PyAny>,
) -> Fallible<bool> {
    let py = condition.py();
    if !condition.is_instance(handles.basic.bind(py))? {
        return Ok(false);
    }
    if condition.getattr("is_Symbol")?.is_truthy()? {
        return Ok(true);
    }
    let connectives = condition.call_method1("atoms", (handles.boolean_function.bind(py),))?;
    for connective in connectives.try_iter()? {
        for argument in connective?.getattr("args")?.try_iter()? {
            if argument?.getattr("is_Symbol")?.is_truthy()? {
                return Ok(true);
            }
        }
    }
    Ok(false)
}
