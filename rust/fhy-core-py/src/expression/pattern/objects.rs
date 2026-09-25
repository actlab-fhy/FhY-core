//! The Python objects of the nodes a match or a rewrite walk reaches.
//!
//! The core matches and rewrites Rust [`Expression`] handles, but every
//! callback receives, and every result returns, Python node objects. While a
//! match or a walk runs, an [`ObjectTable`] maps the identity of each Rust
//! node it reaches to that node's Python object:
//!
//! - an object of the input tree, or of a replacement a callback returned,
//!   is found by walking that tree's objects beside their Rust handles, on
//!   first need, and every object passed on the way is recorded, so the
//!   whole walk costs time linear in the trees;
//! - a node the walk rebuilt around rewritten children, which has no object,
//!   gets one built once from its children's objects through the public
//!   class of its kind, as the materializer of `substitute` does. That
//!   object holds its own Rust handle, equal to the rebuilt node's; the
//!   table maps the object back to the node it stands for, so a callback
//!   returning it returns that node.
//!
//! A match or a walk makes its table current with [`ActiveTable::enter`]
//! for as long as it runs. The tables form a per-thread stack, so a callback
//! that matches or rewrites another tree gets its own table and the outer
//! one is current again when it returns. The interpreter is attached
//! throughout, and no table is borrowed across a call into Python code.

use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::rc::Rc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use fhy_core::expression::Expression;
use fhy_core::expression::pattern::{Capture, MatchBindings};
use fhy_core::tree::{NodeHandle, NodeIdentity};

use super::super::node::{PyExpression, build_node};

/// The last bindings object a table built: the identities of its captures'
/// nodes, in binding order, and the object.
struct CachedBindings {
    entries: Vec<(Capture, NodeIdentity)>,
    object: Py<PyAny>,
}

/// The Python objects of the Rust nodes one match or walk reaches.
pub(super) struct ObjectTable {
    /// Each known object by the identity of the Rust node it stands for,
    /// with that node, which is held so its identity stays unique.
    known: HashMap<NodeIdentity, (Expression, Py<PyAny>)>,
    /// Each object built here for a rebuilt node, by the object's address,
    /// with the node it stands for.
    stand_ins: HashMap<usize, Expression>,
    /// The known objects whose children are not recorded yet.
    frontier: Vec<Py<PyAny>>,
    /// The bindings object built last, reused for the same bindings, which
    /// a rule's guards and rewrite all receive.
    last_bindings: Option<CachedBindings>,
}

impl ObjectTable {
    fn new() -> Self {
        Self {
            known: HashMap::new(),
            stand_ins: HashMap::new(),
            frontier: Vec::new(),
            last_bindings: None,
        }
    }

    /// Record `object`, a Python expression, and queue its children.
    fn discover(&mut self, object: &Bound<'_, PyExpression>) {
        let handle = object.get().expression();
        if let Entry::Vacant(entry) = self.known.entry(handle.identity()) {
            entry.insert((handle.clone(), object.clone().into_any().unbind()));
            self.frontier.push(object.clone().into_any().unbind());
        }
    }

    /// Record the children of the next queued object; return whether one
    /// was queued.
    fn expand_next(&mut self, py: Python<'_>) -> PyResult<bool> {
        let Some(object) = self.frontier.pop() else {
            return Ok(false);
        };
        let object = object.bind(py).cast::<PyExpression>()?.clone();
        for child in object.get().children(py) {
            self.discover(child.cast::<PyExpression>()?);
        }
        Ok(true)
    }

    /// Return the Python object of `node`, finding it in the recorded trees
    /// or building it.
    fn object_of<'py>(
        &mut self,
        py: Python<'py>,
        node: &Expression,
    ) -> PyResult<Bound<'py, PyAny>> {
        loop {
            if let Some((_, object)) = self.known.get(&node.identity()) {
                return Ok(object.bind(py).clone());
            }
            if !self.expand_next(py)? {
                return self.build(py, node);
            }
        }
    }

    /// Return the object of `root`, a node no recorded tree holds, built
    /// from its children's objects; build each child without one first.
    fn build<'py>(&mut self, py: Python<'py>, root: &Expression) -> PyResult<Bound<'py, PyAny>> {
        enum Step<'a> {
            Enter(&'a Expression),
            Build(&'a Expression, usize),
        }
        let mut results: Vec<Bound<'py, PyAny>> = Vec::new();
        let mut pending = vec![Step::Enter(root)];
        while let Some(step) = pending.pop() {
            match step {
                Step::Enter(node) => {
                    if let Some((_, object)) = self.known.get(&node.identity()) {
                        results.push(object.bind(py).clone());
                        continue;
                    }
                    let children: Vec<&Expression> = node.children().collect();
                    pending.push(Step::Build(node, children.len()));
                    pending.extend(children.into_iter().rev().map(Step::Enter));
                }
                Step::Build(node, child_count) => {
                    let children = results.split_off(results.len() - child_count);
                    let object = build_node(py, node, children)?;
                    let own = object.cast::<PyExpression>()?.get().expression().clone();
                    self.known
                        .entry(own.identity())
                        .or_insert_with(|| (own.clone(), object.clone().unbind()));
                    self.known
                        .insert(node.identity(), (node.clone(), object.clone().unbind()));
                    self.stand_ins
                        .insert(object.as_ptr() as usize, node.clone());
                    results.push(object);
                }
            }
        }
        Ok(results
            .pop()
            .unwrap_or_else(|| unreachable!("the walk yields the root's object")))
    }

    /// Return the Rust node the Python expression `object` stands for, and
    /// record its tree: the rebuilt node for an object built here, and its
    /// own handle otherwise.
    fn adopt(&mut self, object: &Bound<'_, PyExpression>) -> Expression {
        if let Some(node) = self.stand_ins.get(&(object.as_ptr() as usize)) {
            return node.clone();
        }
        self.discover(object);
        object.get().expression().clone()
    }
}

thread_local! {
    /// The tables of the matches and walks running on this thread,
    /// innermost last.
    static TABLES: RefCell<Vec<Rc<RefCell<ObjectTable>>>> = const { RefCell::new(Vec::new()) };
}

/// The current table of a match or walk, for as long as the value lives.
pub(super) struct ActiveTable {
    table: Rc<RefCell<ObjectTable>>,
}

impl ActiveTable {
    /// Make a new table current, holding the tree of `root`.
    pub(super) fn enter(root: &Bound<'_, PyExpression>) -> Self {
        let mut table = ObjectTable::new();
        table.discover(root);
        let table = Rc::new(RefCell::new(table));
        TABLES.with(|tables| tables.borrow_mut().push(Rc::clone(&table)));
        Self { table }
    }

    /// Return the Python object of `node`.
    ///
    /// # Errors
    ///
    /// Raises what building an object raises.
    pub(super) fn object_of<'py>(
        &self,
        py: Python<'py>,
        node: &Expression,
    ) -> PyResult<Bound<'py, PyAny>> {
        borrow_table(&self.table)?.object_of(py, node)
    }
}

impl Drop for ActiveTable {
    fn drop(&mut self) {
        TABLES.with(|tables| {
            let popped = tables.borrow_mut().pop();
            debug_assert!(
                popped.is_some_and(|popped| Rc::ptr_eq(&popped, &self.table)),
                "the tables are left in the order they were entered"
            );
        });
    }
}

/// Borrow `table` mutably.
///
/// # Errors
///
/// Raises `RuntimeError` if it is borrowed already, which would mean a
/// Python call re-entered the table while it was in use.
fn borrow_table(table: &RefCell<ObjectTable>) -> PyResult<std::cell::RefMut<'_, ObjectTable>> {
    table.try_borrow_mut().map_err(|_already_borrowed| {
        PyRuntimeError::new_err("the pattern object table was re-entered while in use")
    })
}

/// Return the current table.
///
/// # Errors
///
/// Raises `RuntimeError` if no match or walk is running, which would mean
/// the core called a callback outside one.
fn current_table() -> PyResult<Rc<RefCell<ObjectTable>>> {
    TABLES
        .with(|tables| tables.borrow().last().cloned())
        .ok_or_else(|| {
            PyRuntimeError::new_err("a pattern callback ran outside a match or a rewrite walk")
        })
}

/// Return the Python object of `node` in the current table.
///
/// # Errors
///
/// Raises `RuntimeError` outside a match or walk, and what building an
/// object raises.
pub(super) fn current_object_of<'py>(
    py: Python<'py>,
    node: &Expression,
) -> PyResult<Bound<'py, PyAny>> {
    let table = current_table()?;
    borrow_table(&table)?.object_of(py, node)
}

/// Return the Rust node `object`, a callback's result, stands for, and
/// record its tree in the current table.
///
/// # Errors
///
/// Raises `RuntimeError` outside a match or walk.
pub(super) fn current_adopt(object: &Bound<'_, PyExpression>) -> PyResult<Expression> {
    let table = current_table()?;
    let node = borrow_table(&table)?.adopt(object);
    Ok(node)
}

/// Return the bindings object of `bindings`, built by `build` from the
/// objects of its nodes, or the last one the current table built if it
/// bound the same captures to the same nodes.
///
/// # Errors
///
/// Raises `RuntimeError` outside a match or walk, and what `build` or
/// building an object raises.
pub(super) fn current_bindings_object<'py>(
    py: Python<'py>,
    bindings: &MatchBindings,
    build: impl FnOnce(Vec<(&Capture, Bound<'py, PyAny>)>) -> PyResult<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let table = current_table()?;
    let mut table = borrow_table(&table)?;
    if let Some(cached) = &table.last_bindings {
        let is_same = cached.entries.len() == bindings.len()
            && cached.entries.iter().zip(bindings.iter()).all(
                |((capture, identity), (bound, node))| {
                    capture == bound && *identity == node.identity()
                },
            );
        if is_same {
            return Ok(cached.object.bind(py).clone());
        }
    }
    let entries = bindings
        .iter()
        .map(|(capture, node)| Ok((capture, table.object_of(py, node)?)))
        .collect::<PyResult<Vec<_>>>()?;
    // Building the object runs no user code, but releasing the table first
    // keeps every borrow short.
    drop(table);
    let object = build(entries)?;
    let table = current_table()?;
    borrow_table(&table)?.last_bindings = Some(CachedBindings {
        entries: bindings
            .iter()
            .map(|(capture, node)| (capture.clone(), node.identity()))
            .collect(),
        object: object.clone().unbind(),
    });
    Ok(object)
}
