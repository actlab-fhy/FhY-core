//! Rebuilding SymPy trees bottom-up: the substitution of symbols, and the
//! walk the masking of Boolean comparisons shares with it.

use std::collections::HashMap;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyTuple;

use super::address_hash::BuildAddressHasher;
use super::boolean::{Fallible, boolean_positions};
use super::error::SympyErrorKind;
use super::load::Handles;

/// What a walk's hook decides for a node before its arguments are walked.
pub(super) enum Visit<'py> {
    /// Walk the node's arguments, and rebuild it if one changes.
    Descend,
    /// Use this object for the node, without walking it, and say whether
    /// it counts as a change.
    Replace(Bound<'py, PyAny>, bool),
}

/// One node of a walk in progress: its arguments, and what they became.
struct Frame<'py> {
    node: Bound<'py, PyAny>,
    arguments: Vec<Bound<'py, PyAny>>,
    done: Vec<(Bound<'py, PyAny>, bool)>,
}

/// Rebuild `root` bottom-up on a work list, as a recursive walk over
/// SymPy's `args` would, without the recursion.
///
/// A node met again, the same Python object (`id`), takes the result it
/// had, so a SymPy DAG costs its distinct nodes, and a shared node's result
/// is shared too. The memo holds every node it keys by, so no id is reused
/// while the walk runs.
///
/// `visit` decides for each SymPy node (`Basic`) whether to replace it or
/// to walk its arguments; an argument that is not a SymPy node is kept.
/// Once a node's arguments are walked, `rebuild` receives the node, the
/// arguments' results, and whether any changed, and returns the node's
/// result and whether it changed. Returns the root's result and whether it
/// changed.
pub(super) fn rebuild_bottom_up<'py, E: From<PyErr>>(
    handles: &Handles,
    root: &Bound<'py, PyAny>,
    mut visit: impl FnMut(&Bound<'py, PyAny>) -> Result<Visit<'py>, E>,
    mut rebuild: impl FnMut(
        &Bound<'py, PyAny>,
        Vec<Bound<'py, PyAny>>,
        bool,
    ) -> Result<(Bound<'py, PyAny>, bool), E>,
) -> Result<(Bound<'py, PyAny>, bool), E> {
    let py = root.py();
    let basic = handles.basic.bind(py);
    let mut memo: Memo<'py> = HashMap::default();
    let mut open =
        |node: &Bound<'py, PyAny>| -> Result<Result<Frame<'py>, (Bound<'py, PyAny>, bool)>, E> {
            match visit(node)? {
                Visit::Replace(object, changed) => Ok(Err((object, changed))),
                Visit::Descend => {
                    let arguments = node
                        .getattr("args")
                        .and_then(|arguments| arguments.try_iter()?.collect::<PyResult<Vec<_>>>())
                        .map_err(E::from)?;
                    Ok(Ok(Frame {
                        node: node.clone(),
                        done: Vec::with_capacity(arguments.len()),
                        arguments,
                    }))
                }
            }
        };
    let mut stack: Vec<Frame<'py>> = Vec::new();
    match open(root)? {
        Err(finished) => return Ok(finished),
        Ok(frame) => stack.push(frame),
    }
    loop {
        let top = stack.last_mut().expect("a walk in progress");
        if top.done.len() < top.arguments.len() {
            let argument = top.arguments[top.done.len()].clone();
            if !argument.is_instance(basic).map_err(E::from)? {
                top.done.push((argument, false));
                continue;
            }
            if let Some((_, finished)) = memo.get(&argument.as_ptr().addr()) {
                top.done.push(finished.clone());
                continue;
            }
            match open(&argument)? {
                Err(finished) => {
                    memo.insert(argument.as_ptr().addr(), (argument, finished.clone()));
                    stack.last_mut().expect("the parent").done.push(finished);
                }
                Ok(frame) => stack.push(frame),
            }
            continue;
        }
        let frame = stack.pop().expect("the finished node");
        let changed = frame.done.iter().any(|(_, changed)| *changed);
        let results = frame.done.into_iter().map(|(result, _)| result).collect();
        let finished = rebuild(&frame.node, results, changed)?;
        match stack.last_mut() {
            Some(parent) => {
                parent.done.push(finished.clone());
                memo.insert(frame.node.as_ptr().addr(), (frame.node, finished));
            }
            None => return Ok(finished),
        }
    }
}

/// The results of a walk's finished nodes, keyed by each node's address,
/// with the node, which keeps the address its own.
type Memo<'py> = HashMap<usize, (Bound<'py, PyAny>, (Bound<'py, PyAny>, bool)), BuildAddressHasher>;

/// Return `node` with `replacements`, a mapping from SymPy symbols to SymPy
/// objects, applied.
///
/// Substitutes as `xreplace` does, in one simultaneous bottom-up pass that
/// rebuilds only a node whose arguments changed, but makes the rebuilt
/// node's Boolean positions Boolean first: a rebuild evaluates the node, and
/// SymPy decides `Eq(Piecewise, True)` as `False` on sight and refuses a
/// piecewise under `And`. A node no replacement reaches is kept as it
/// stands.
pub(super) fn substitute_symbols<'py>(
    handles: &Arc<Handles>,
    node: &Bound<'py, PyAny>,
    replacements: &Bound<'py, PyAny>,
) -> Fallible<Bound<'py, PyAny>> {
    let py = node.py();
    let (result, _) = rebuild_bottom_up::<SympyErrorKind>(
        handles,
        node,
        |current| {
            if replacements.contains(current)? {
                Ok(Visit::Replace(replacements.get_item(current)?, true))
            } else {
                Ok(Visit::Descend)
            }
        },
        |current, arguments, changed| {
            if !changed {
                return Ok((current.clone(), false));
            }
            let arguments = boolean_positions(handles, current, arguments)?;
            let rebuilt = current
                .getattr("func")?
                .call1(PyTuple::new(py, arguments)?)?;
            Ok((rebuilt, true))
        },
    )?;
    Ok(result)
}
