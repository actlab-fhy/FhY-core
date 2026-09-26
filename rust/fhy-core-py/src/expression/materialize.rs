//! Turning a tree the core built into Python node objects, reusing the
//! objects of every subtree the core kept.
//!
//! The core's `substitute` returns a handle to each subtree it left
//! unchanged, and to the replacement node at every replaced reference. The
//! materializer walks the result top-down beside the input's Python
//! objects: a result node that is the handle of the Python object at the
//! same place, or of a replacement, is that object; any other node is built
//! from its children's objects through the public class of its kind. A node
//! the core shares is built once. The walk keeps its pending nodes on the
//! heap, so a tree of any depth materializes.

use std::collections::HashMap;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyMapping;

use fhy_core::expression::Expression;
use fhy_core::identifier::Identifier;
use fhy_core::tree::{NodeHandle, NodeIdentity, Tree};

use crate::error::IntoPyResult;
use crate::identifier::{read_identifier_id, restore_identifier};

use super::node::{PyExpression, build_node};

/// One step of the materializing walk.
enum Step<'a, 'py> {
    /// Materialize `node`, whose Python object at the same place of the
    /// input tree is `hint`, if the input has a node there.
    Enter {
        node: &'a Expression,
        hint: Option<Bound<'py, PyExpression>>,
    },
    /// Build the object of `node` from its children's objects, the last
    /// `child_count` results.
    Build {
        node: &'a Expression,
        child_count: usize,
    },
}

/// The materializer of one result tree.
struct Materializer<'py> {
    py: Python<'py>,
    /// The objects already known for core nodes, by identity: the
    /// replacements, and every shared node built so far.
    known: HashMap<NodeIdentity, Bound<'py, PyAny>>,
}

impl<'py> Materializer<'py> {
    /// Return the Python object of `root`, whose input object at the same
    /// place is `hint`.
    fn materialize(
        &mut self,
        root: &Expression,
        hint: Option<Bound<'py, PyExpression>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut results: Vec<Bound<'py, PyAny>> = Vec::new();
        let mut pending = vec![Step::Enter { node: root, hint }];
        while let Some(step) = pending.pop() {
            match step {
                Step::Enter { node, hint } => {
                    if let Some(hint) = &hint {
                        if Expression::ptr_eq(hint.get().expression(), node) {
                            results.push(hint.clone().into_any());
                            continue;
                        }
                    }
                    if let Some(object) = self.known.get(&node.identity()) {
                        results.push(object.clone());
                        continue;
                    }
                    let children: Vec<&Expression> = node.children().collect();
                    let hint_children = hint.as_ref().and_then(|hint| {
                        let objects = hint.get().children(self.py);
                        (objects.len() == children.len()).then(|| objects.clone())
                    });
                    pending.push(Step::Build {
                        node,
                        child_count: children.len(),
                    });
                    for (index, child) in children.into_iter().enumerate().rev() {
                        let child_hint = match &hint_children {
                            Some(objects) => {
                                Some(objects.get_item(index)?.cast_into::<PyExpression>()?)
                            }
                            None => None,
                        };
                        pending.push(Step::Enter {
                            node: child,
                            hint: child_hint,
                        });
                    }
                }
                Step::Build { node, child_count } => {
                    let first = results.len() - child_count;
                    let children = results.split_off(first);
                    let object = build_node(self.py, node, children)?;
                    if node.is_shared() {
                        self.known.insert(node.identity(), object.clone());
                    }
                    results.push(object);
                }
            }
        }
        Ok(results
            .pop()
            .unwrap_or_else(|| unreachable!("the walk yields the root's object")))
    }
}

/// Return the expression `slf` with the replacements of the mapping
/// `replacements` substituted, as `Expression.substitute` does.
///
/// Entries whose key is not an `Identifier` never match a reference and are
/// ignored. A value that is not an expression is refused only when its key
/// occurs in the expression.
///
/// # Errors
///
/// Raises `TypeError` for a referenced identifier mapped to a value that is
/// not an expression, and `ValueError` if a replacement puts a literal other
/// than a Boolean in a piecewise case condition.
pub(super) fn substitute<'py>(
    slf: &Bound<'py, PyExpression>,
    replacements: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = slf.py();
    let mapping = replacements.cast::<PyMapping>()?;
    let mut map: HashMap<Identifier, Expression> = HashMap::new();
    let mut known = HashMap::new();
    let mut refused: Vec<(Identifier, Bound<'py, PyAny>, Bound<'py, PyAny>)> = Vec::new();
    for item in mapping.items()?.iter() {
        let (key, value) = item.extract::<(Bound<'py, PyAny>, Bound<'py, PyAny>)>()?;
        if read_identifier_id(&key)?.is_none() {
            continue;
        }
        let identifier = restore_identifier(&key, "substitute", "key")?;
        match value.cast::<PyExpression>() {
            Ok(expression) => {
                let handle = expression.get().expression().clone();
                known.insert(handle.identity(), value.clone());
                map.insert(identifier, handle);
            }
            Err(_not_an_expression) => refused.push((identifier, key, value)),
        }
    }
    let this = slf.get();
    if !refused.is_empty() {
        let free = this.expression().free_identifiers();
        if let Some((_, key, value)) = refused
            .iter()
            .find(|(identifier, _, _)| free.contains(identifier))
        {
            return Err(PyTypeError::new_err(format!(
                "Cannot substitute {} with a non-Expression term of type {}.",
                key.repr()?,
                value.get_type().name()?
            )));
        }
    }
    if map.is_empty() {
        return Ok(slf.clone().into_any());
    }
    let result = this.expression().substitute(&map).into_py_result()?;
    if Expression::ptr_eq(&result, this.expression()) {
        return Ok(slf.clone().into_any());
    }
    let mut materializer = Materializer { py, known };
    materializer.materialize(&result, Some(slf.clone()))
}

/// Return the Python object of the tree `result` the core built from the
/// expression `input`, reusing the objects of every subtree of `input` the
/// core kept in place.
///
/// # Errors
///
/// Raises whatever building a node through its public class raises.
pub(super) fn materialize_beside<'py>(
    input: &Bound<'py, PyExpression>,
    result: &Expression,
) -> PyResult<Bound<'py, PyAny>> {
    if Expression::ptr_eq(result, input.get().expression()) {
        return Ok(input.clone().into_any());
    }
    let mut materializer = Materializer {
        py: input.py(),
        known: HashMap::new(),
    };
    materializer.materialize(result, Some(input.clone()))
}

/// Return a new Python object of the core tree `expression`, every node
/// built through the public class of its kind, and a node the core shares
/// built once.
///
/// # Errors
///
/// Raises whatever building a node through its public class raises.
pub(super) fn materialize_expression<'py>(
    py: Python<'py>,
    expression: &Expression,
) -> PyResult<Bound<'py, PyAny>> {
    let mut materializer = Materializer {
        py,
        known: HashMap::new(),
    };
    materializer.materialize(expression, None)
}
