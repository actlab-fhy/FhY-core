//! A toy tree IR and recording visitors and rewriters over it, for the tree
//! traversal tests.
//!
//! The visitor and the rewriter work under any traversal context, so one
//! value serves a direct walk with `&mut ()` and a walk or rewrite pass.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use fhy_core::tree::{NodeHandle, NodeIdentity, Rewriter, Tree, TreeVisitor};

// =============================================================================
// The toy tree
// =============================================================================

/// One immutable node of the toy tree.
#[derive(Debug, Default)]
struct ToyNode {
    name: String,
    value: i64,
    children: Vec<ToyTree>,
    is_frozen: bool,
    hides_sharing: bool,
    hash_conses: bool,
}

/// A handle to a toy tree node: a name, an integer, and ordered children.
///
/// Clones share the node. Equality is structural and ignores whether a node
/// is frozen. Dropping a deep tree does not recurse.
#[derive(Debug, Clone)]
pub(crate) struct ToyTree(Arc<ToyNode>);

/// The error rebuilding a toy tree node returns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ToyRebuildError {
    /// The node was given a child list of the wrong length.
    ChildCountMismatch {
        /// The number of children the node has.
        expected: usize,
        /// The number of children given.
        actual: usize,
    },
    /// The node is frozen and refuses every rebuild.
    Frozen {
        /// The name of the frozen node.
        name: String,
    },
}

impl fmt::Display for ToyRebuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChildCountMismatch { expected, actual } => {
                write!(f, "expected {expected} children, got {actual}")
            }
            Self::Frozen { name } => write!(f, "node {name} is frozen"),
        }
    }
}

impl Error for ToyRebuildError {}

/// Build the node `name` holding `value` over `children`.
fn build_toy_node(name: &str, value: i64, children: Vec<ToyTree>, is_frozen: bool) -> ToyTree {
    ToyTree(Arc::new(ToyNode {
        name: name.to_owned(),
        value,
        children,
        is_frozen,
        hides_sharing: false,
        hash_conses: false,
    }))
}

/// Build the leaf `name` holding `value`, which reports itself unshared
/// however many handles to it exist.
#[must_use]
pub(crate) fn build_leaf_hiding_sharing(name: &str, value: i64) -> ToyTree {
    ToyTree(Arc::new(ToyNode {
        name: name.to_owned(),
        value,
        hides_sharing: true,
        ..ToyNode::default()
    }))
}

/// Build the inner node `name`, holding zero, over `children`, whose rebuild
/// around children equal to its own returns the node itself, as a
/// hash-consing IR does.
#[must_use]
pub(crate) fn build_hash_consing_node(name: &str, children: &[&ToyTree]) -> ToyTree {
    ToyTree(Arc::new(ToyNode {
        name: name.to_owned(),
        children: children.iter().map(|&child| child.clone()).collect(),
        hash_conses: true,
        ..ToyNode::default()
    }))
}

/// Build the leaf `name` holding `value`.
#[must_use]
pub(crate) fn build_leaf(name: &str, value: i64) -> ToyTree {
    build_toy_node(name, value, Vec::new(), false)
}

/// Build the inner node `name`, holding zero, over `children`.
#[must_use]
pub(crate) fn build_node(name: &str, children: &[&ToyTree]) -> ToyTree {
    let children = children.iter().map(|&child| child.clone()).collect();
    build_toy_node(name, 0, children, false)
}

/// Build the inner node `name`, holding zero, over `children`, which refuses
/// every rebuild.
#[must_use]
pub(crate) fn build_frozen_node(name: &str, children: &[&ToyTree]) -> ToyTree {
    let children = children.iter().map(|&child| child.clone()).collect();
    build_toy_node(name, 0, children, true)
}

impl ToyTree {
    /// Return the node's name.
    #[must_use]
    pub(crate) fn name(&self) -> &str {
        &self.0.name
    }

    /// Return the node's integer.
    #[must_use]
    pub(crate) fn value(&self) -> i64 {
        self.0.value
    }

    /// Return the node's children, in order.
    #[must_use]
    pub(crate) fn child_nodes(&self) -> &[Self] {
        &self.0.children
    }

    /// Return the child at `index`.
    ///
    /// # Panics
    ///
    /// Panics if the node has no child at `index`.
    #[must_use]
    pub(crate) fn child(&self, index: usize) -> &Self {
        &self.0.children[index]
    }

    /// Return whether the node refuses every rebuild.
    #[must_use]
    pub(crate) fn is_frozen(&self) -> bool {
        self.0.is_frozen
    }

    /// Return this node with `value` in place of its integer, sharing its
    /// children.
    #[must_use]
    pub(crate) fn with_value(&self, value: i64) -> Self {
        build_toy_node(
            &self.0.name,
            value,
            self.0.children.clone(),
            self.0.is_frozen,
        )
    }

    /// Return whether both handles point to the same node.
    #[must_use]
    pub(crate) fn is_same_node(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    /// Return the number of distinct nodes in the tree.
    #[must_use]
    pub(crate) fn count_distinct_nodes(&self) -> usize {
        let mut seen = HashSet::new();
        let mut pending = vec![self];
        while let Some(node) = pending.pop() {
            if seen.insert(node.identity()) {
                pending.extend(node.child_nodes());
            }
        }
        seen.len()
    }

    /// Return the number of node occurrences in the tree, counting a shared
    /// node once per occurrence.
    #[must_use]
    pub(crate) fn count_occurrences(&self) -> usize {
        let mut count = 0;
        let mut pending = vec![self];
        while let Some(node) = pending.pop() {
            count += 1;
            pending.extend(node.child_nodes());
        }
        count
    }

    /// Return a copy of the tree sharing no node, with one node per
    /// occurrence.
    #[must_use]
    pub(crate) fn copy_unshared(&self) -> Self {
        let mut results: Vec<Self> = Vec::new();
        let mut steps = vec![(self, false)];
        while let Some((node, is_exit)) = steps.pop() {
            if is_exit {
                let first_child = results.len() - node.child_nodes().len();
                let children = results.split_off(first_child);
                results.push(build_toy_node(
                    node.name(),
                    node.value(),
                    children,
                    node.is_frozen(),
                ));
            } else {
                steps.push((node, true));
                steps.extend(node.child_nodes().iter().rev().map(|child| (child, false)));
            }
        }
        results
            .pop()
            .expect("a copy leaves exactly the copied root")
    }

    /// Return the names of the tree's nodes in pre-order, one per
    /// occurrence.
    #[must_use]
    pub(crate) fn list_names_in_pre_order(&self) -> Vec<String> {
        let mut names = Vec::new();
        let mut pending = vec![self];
        while let Some(node) = pending.pop() {
            names.push(node.name().to_owned());
            pending.extend(node.child_nodes().iter().rev());
        }
        names
    }
}

impl PartialEq for ToyTree {
    fn eq(&self, other: &Self) -> bool {
        let mut pending = vec![(self, other)];
        while let Some((left, right)) = pending.pop() {
            if left.name() != right.name()
                || left.value() != right.value()
                || left.child_nodes().len() != right.child_nodes().len()
            {
                return false;
            }
            pending.extend(left.child_nodes().iter().zip(right.child_nodes()));
        }
        true
    }
}

impl Eq for ToyTree {}

impl Drop for ToyTree {
    fn drop(&mut self) {
        let Some(node) = Arc::get_mut(&mut self.0) else {
            return;
        };
        let mut pending = std::mem::take(&mut node.children);
        while let Some(mut child) = pending.pop() {
            if let Some(node) = Arc::get_mut(&mut child.0) {
                pending.append(&mut node.children);
            }
        }
    }
}

impl NodeHandle for ToyTree {
    fn identity(&self) -> NodeIdentity {
        NodeIdentity::of_arc(&self.0)
    }
}

impl Tree for ToyTree {
    type RebuildError = ToyRebuildError;

    fn children(&self) -> impl Iterator<Item = &Self> {
        self.0.children.iter()
    }

    fn is_shared(&self) -> bool {
        !self.0.hides_sharing
    }

    fn rebuild_with_children(&self, children: Vec<Self>) -> Result<Self, ToyRebuildError> {
        if self.is_frozen() {
            return Err(ToyRebuildError::Frozen {
                name: self.name().to_owned(),
            });
        }
        if children.len() != self.child_nodes().len() {
            return Err(ToyRebuildError::ChildCountMismatch {
                expected: self.child_nodes().len(),
                actual: children.len(),
            });
        }
        if self.0.hash_conses && children == self.child_nodes() {
            return Ok(self.clone());
        }
        Ok(build_toy_node(self.name(), self.value(), children, false))
    }
}

// =============================================================================
// Hook errors
// =============================================================================

/// The error the test visitors and rewriters return.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HookError(pub(crate) String);

impl fmt::Display for HookError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for HookError {}

// =============================================================================
// A recording visitor
// =============================================================================

/// A walk hook of [`TreeVisitor`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WalkHook {
    /// [`TreeVisitor::before_visit`].
    Before,
    /// [`TreeVisitor::visit`].
    Visit,
    /// [`TreeVisitor::after_visit`].
    After,
}

impl WalkHook {
    /// Return the label of the hook in the recorded events.
    #[must_use]
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Before => "before",
            Self::Visit => "visit",
            Self::After => "after",
        }
    }
}

/// A visitor recording `<hook>:<name>` for every hook call, which prunes the
/// children of the nodes named in `pruned` and fails in one hook on one node
/// name if asked to.
#[derive(Debug, Default)]
pub(crate) struct RecordingVisitor {
    events: Vec<String>,
    pruned: HashSet<String>,
    failure: Option<(WalkHook, String)>,
}

impl RecordingVisitor {
    /// Create a visitor recording every hook call.
    #[must_use]
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Return this visitor pruning the children of the nodes named `name`.
    #[must_use]
    pub(crate) fn with_pruned(mut self, name: &str) -> Self {
        self.pruned.insert(name.to_owned());
        self
    }

    /// Return this visitor failing in `hook` on the node named `name`, after
    /// recording the call.
    #[must_use]
    pub(crate) fn with_failure(mut self, hook: WalkHook, name: &str) -> Self {
        self.failure = Some((hook, name.to_owned()));
        self
    }

    /// Return the recorded events, in call order.
    #[must_use]
    pub(crate) fn events(&self) -> &[String] {
        &self.events
    }

    /// Forget the recorded events.
    pub(crate) fn clear_events(&mut self) {
        self.events.clear();
    }

    /// Return the node names `hook` was called on, in call order.
    #[must_use]
    pub(crate) fn list_names(&self, hook: WalkHook) -> Vec<String> {
        let prefix = format!("{}:", hook.label());
        self.events
            .iter()
            .filter_map(|event| event.strip_prefix(&prefix).map(str::to_owned))
            .collect()
    }

    /// Record a call of `hook` on `node`, failing if asked to.
    fn record(&mut self, hook: WalkHook, node: &ToyTree) -> Result<(), HookError> {
        let event = format!("{}:{}", hook.label(), node.name());
        self.events.push(event.clone());
        match &self.failure {
            Some((failing_hook, name)) if *failing_hook == hook && name == node.name() => {
                Err(HookError(format!("{event} failed")))
            }
            _ => Ok(()),
        }
    }
}

impl<C: ?Sized> TreeVisitor<ToyTree, C> for RecordingVisitor {
    type Error = HookError;

    fn before_visit(&mut self, node: &ToyTree, _cx: &mut C) -> Result<(), HookError> {
        self.record(WalkHook::Before, node)
    }

    fn visit(&mut self, node: &ToyTree, _cx: &mut C) -> Result<(), HookError> {
        self.record(WalkHook::Visit, node)
    }

    fn after_visit(&mut self, node: &ToyTree, _cx: &mut C) -> Result<(), HookError> {
        self.record(WalkHook::After, node)
    }

    fn walks_children(&mut self, node: &ToyTree) -> bool {
        !self.pruned.contains(node.name())
    }
}

// =============================================================================
// Rewriters
// =============================================================================

/// The rewrite of a [`ClosureRewriter`].
type RewriteHook = Box<dyn FnMut(&ToyTree) -> Result<Option<ToyTree>, HookError> + Send>;

/// A rewriter whose rewrite is a closure, recording every node it is asked
/// to rewrite.
pub(crate) struct ClosureRewriter {
    rewrite: RewriteHook,
    seen: Vec<ToyTree>,
}

impl ClosureRewriter {
    /// Create the rewriter rewriting with `rewrite`.
    pub(crate) fn new(
        rewrite: impl FnMut(&ToyTree) -> Result<Option<ToyTree>, HookError> + Send + 'static,
    ) -> Self {
        Self {
            rewrite: Box::new(rewrite),
            seen: Vec::new(),
        }
    }

    /// Return the nodes the rewriter was asked to rewrite, in call order.
    #[must_use]
    pub(crate) fn seen(&self) -> &[ToyTree] {
        &self.seen
    }

    /// Return the names of the nodes the rewriter was asked to rewrite, in
    /// call order.
    #[must_use]
    pub(crate) fn list_seen_names(&self) -> Vec<String> {
        self.seen
            .iter()
            .map(|node| node.name().to_owned())
            .collect()
    }
}

impl fmt::Debug for ClosureRewriter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClosureRewriter")
            .field("seen", &self.seen.len())
            .finish_non_exhaustive()
    }
}

impl<C: ?Sized> Rewriter<ToyTree, C> for ClosureRewriter {
    type Error = HookError;

    fn rewrite(&mut self, node: &ToyTree, _cx: &mut C) -> Result<Option<ToyTree>, HookError> {
        self.seen.push(node.clone());
        (self.rewrite)(node)
    }
}

/// Return a rewriter that keeps every node.
#[must_use]
pub(crate) fn build_keeping_rewriter() -> ClosureRewriter {
    ClosureRewriter::new(|_| Ok(None))
}
