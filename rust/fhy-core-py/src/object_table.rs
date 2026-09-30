//! The Python objects one call has seen for the core's expression nodes and
//! identifiers.
//!
//! The core hands the binding back handles to nodes that Python objects
//! already stand for: the input a substitution kept in place, a
//! replacement, a pattern's capture. Returning the same Python object for
//! such a node keeps Python identity, and costs no new object. Every call
//! that turns core values back into Python objects records what it has seen
//! in an [`ObjectTable`]: nodes by identity, with the node, which the table
//! holds so its identity stays unique while recorded, and `Identifier`
//! objects by id. A table a call reaches through a thread-local, the
//! pattern tables and the type-system contexts, is pushed on a
//! [`ScopedStack`](crate::scoped::ScopedStack).

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use pyo3::prelude::*;

use fhy_core::expression::Expression;
use fhy_core::tree::{NodeHandle, NodeIdentity};

/// The Python objects of the expression nodes and identifiers one call has
/// seen.
#[derive(Default)]
pub(crate) struct ObjectTable {
    /// Each node's object by the node's identity, with the node.
    nodes: HashMap<NodeIdentity, (Expression, Py<PyAny>)>,
    /// Each `Identifier` object by its id.
    identifiers: HashMap<u64, Py<PyAny>>,
}

impl std::fmt::Debug for ObjectTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObjectTable")
            .field("nodes", &self.nodes.len())
            .field("identifiers", &self.identifiers.len())
            .finish()
    }
}

impl ObjectTable {
    /// Return an empty table.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Return the object of `node`, if recorded.
    pub(crate) fn node<'py>(
        &self,
        py: Python<'py>,
        node: &Expression,
    ) -> Option<Bound<'py, PyAny>> {
        self.nodes
            .get(&node.identity())
            .map(|(_, object)| object.bind(py).clone())
    }

    /// Record `object` as the object of `node`, in place of any recorded.
    pub(crate) fn insert_node(&mut self, node: &Expression, object: &Bound<'_, PyAny>) {
        self.nodes
            .insert(node.identity(), (node.clone(), object.clone().unbind()));
    }

    /// Record `object` as the object of `node` unless one is recorded;
    /// return whether it was recorded.
    pub(crate) fn insert_node_if_absent(
        &mut self,
        node: &Expression,
        object: &Bound<'_, PyAny>,
    ) -> bool {
        match self.nodes.entry(node.identity()) {
            Entry::Vacant(entry) => {
                entry.insert((node.clone(), object.clone().unbind()));
                true
            }
            Entry::Occupied(_) => false,
        }
    }

    /// Return the `Identifier` object of `id`, if recorded.
    pub(crate) fn identifier<'py>(&self, py: Python<'py>, id: u64) -> Option<Bound<'py, PyAny>> {
        self.identifiers
            .get(&id)
            .map(|object| object.bind(py).clone())
    }

    /// Record `object` as the `Identifier` object of `id`, in place of any
    /// recorded.
    pub(crate) fn insert_identifier(&mut self, id: u64, object: &Bound<'_, PyAny>) {
        self.identifiers.insert(id, object.clone().unbind());
    }

    /// Record `object` as the `Identifier` object of `id` unless one is
    /// recorded.
    pub(crate) fn insert_identifier_if_absent(&mut self, id: u64, object: &Bound<'_, PyAny>) {
        self.identifiers
            .entry(id)
            .or_insert_with(|| object.clone().unbind());
    }

    /// Return a copy of the table, sharing its objects.
    pub(crate) fn clone_ref(&self, py: Python<'_>) -> Self {
        Self {
            nodes: self
                .nodes
                .iter()
                .map(|(identity, (node, object))| (*identity, (node.clone(), object.clone_ref(py))))
                .collect(),
            identifiers: self
                .identifiers
                .iter()
                .map(|(id, object)| (*id, object.clone_ref(py)))
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use pyo3::types::PyString;

    use super::*;

    #[test]
    fn a_node_keeps_its_first_object_unless_replaced() {
        Python::initialize();
        Python::attach(|py| {
            let node = Expression::from(1);
            let first = PyString::new(py, "first").into_any();
            let second = PyString::new(py, "second").into_any();
            let mut table = ObjectTable::new();
            assert!(table.node(py, &node).is_none());
            assert!(table.insert_node_if_absent(&node, &first));
            assert!(!table.insert_node_if_absent(&node, &second));
            assert!(
                table
                    .node(py, &node)
                    .is_some_and(|object| object.is(&first))
            );
            table.insert_node(&node, &second);
            assert!(
                table
                    .node(py, &node)
                    .is_some_and(|object| object.is(&second))
            );
            assert!(table.node(py, &Expression::from(1)).is_none());
        });
    }

    #[test]
    fn an_identifier_keeps_its_first_object_unless_replaced_and_copies_share_objects() {
        Python::initialize();
        Python::attach(|py| {
            let first = PyString::new(py, "first").into_any();
            let second = PyString::new(py, "second").into_any();
            let mut table = ObjectTable::new();
            table.insert_identifier_if_absent(7, &first);
            table.insert_identifier_if_absent(7, &second);
            assert!(
                table
                    .identifier(py, 7)
                    .is_some_and(|object| object.is(&first))
            );
            let copy = table.clone_ref(py);
            table.insert_identifier(7, &second);
            assert!(
                table
                    .identifier(py, 7)
                    .is_some_and(|object| object.is(&second))
            );
            assert!(
                copy.identifier(py, 7)
                    .is_some_and(|object| object.is(&first))
            );
            assert!(copy.identifier(py, 8).is_none());
        });
    }
}
