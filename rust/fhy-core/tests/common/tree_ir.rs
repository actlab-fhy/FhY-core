//! A toy tree IR, recording visitors and rewriters over it, and a helper
//! that lends a pass context to a test, for the tree traversal tests.
//!
//! Included by the tree test targets with
//! `#[path = "common/tree_ir.rs"] pub mod tree_ir;`, so an item one target
//! does not use is not reported as dead code there.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use fhy_core::pass::{
    CompilerPass, ExecutePass, NodeHandle, NodeIdentity, PassContext, PassFailure, Rewriter, Tree,
    TreeVisitor,
};

// =============================================================================
// The toy tree
// =============================================================================

/// One immutable node of the toy tree.
#[derive(Debug)]
pub struct ToyNode {
    name: String,
    value: i64,
    children: Vec<ToyTree>,
    is_frozen: bool,
    hides_sharing: bool,
}

/// A handle to a toy tree node: a name, an integer, and ordered children.
///
/// Clones share the node. Equality is structural and ignores whether a node
/// is frozen. Dropping a deep tree does not recurse.
#[derive(Debug, Clone)]
pub struct ToyTree(Arc<ToyNode>);

/// The error rebuilding a toy tree node returns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToyRebuildError {
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
    }))
}

/// Build the leaf `name` holding `value`, which reports itself unshared
/// however many handles to it exist.
#[must_use]
pub fn build_leaf_hiding_sharing(name: &str, value: i64) -> ToyTree {
    ToyTree(Arc::new(ToyNode {
        name: name.to_owned(),
        value,
        children: Vec::new(),
        is_frozen: false,
        hides_sharing: true,
    }))
}

/// Build the leaf `name` holding `value`.
#[must_use]
pub fn build_leaf(name: &str, value: i64) -> ToyTree {
    build_toy_node(name, value, Vec::new(), false)
}

/// Build the inner node `name`, holding zero, over `children`.
#[must_use]
pub fn build_node(name: &str, children: &[&ToyTree]) -> ToyTree {
    let children = children.iter().map(|&child| child.clone()).collect();
    build_toy_node(name, 0, children, false)
}

/// Build the inner node `name`, holding zero, over `children`, which refuses
/// every rebuild.
#[must_use]
pub fn build_frozen_node(name: &str, children: &[&ToyTree]) -> ToyTree {
    let children = children.iter().map(|&child| child.clone()).collect();
    build_toy_node(name, 0, children, true)
}

/// Build a chain `depth` nodes above `leaf`, each named `name` with `leaf`
/// at the bottom.
#[must_use]
pub fn build_chain(name: &str, leaf: &ToyTree, depth: usize) -> ToyTree {
    let mut chain = leaf.clone();
    for _ in 0..depth {
        chain = build_node(name, &[&chain]);
    }
    chain
}

/// Build the DAG `x_0 = leaf`, `x_{k+1} = name(x_k, x_k)`, and return
/// `x_levels`, which has `levels + 1` distinct nodes and `2^(levels + 1) - 1`
/// node occurrences.
#[must_use]
pub fn build_doubling_dag(name: &str, leaf: &ToyTree, levels: usize) -> ToyTree {
    let mut dag = leaf.clone();
    for _ in 0..levels {
        dag = build_node(name, &[&dag, &dag]);
    }
    dag
}

impl ToyTree {
    /// Return the node's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.0.name
    }

    /// Return the node's integer.
    #[must_use]
    pub fn value(&self) -> i64 {
        self.0.value
    }

    /// Return the node's children, in order.
    #[must_use]
    pub fn child_nodes(&self) -> &[ToyTree] {
        &self.0.children
    }

    /// Return the child at `index`.
    ///
    /// # Panics
    ///
    /// Panics if the node has no child at `index`.
    #[must_use]
    pub fn child(&self, index: usize) -> &ToyTree {
        &self.0.children[index]
    }

    /// Return whether the node refuses every rebuild.
    #[must_use]
    pub fn is_frozen(&self) -> bool {
        self.0.is_frozen
    }

    /// Return this node with `value` in place of its integer, sharing its
    /// children.
    #[must_use]
    pub fn with_value(&self, value: i64) -> ToyTree {
        build_toy_node(
            &self.0.name,
            value,
            self.0.children.clone(),
            self.0.is_frozen,
        )
    }

    /// Return whether both handles point to the same node.
    #[must_use]
    pub fn is_same_node(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    /// Return the number of distinct nodes in the tree.
    #[must_use]
    pub fn count_distinct_nodes(&self) -> usize {
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
    pub fn count_occurrences(&self) -> usize {
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
    ///
    /// # Panics
    ///
    /// Panics if the copy walk does not end with exactly the copied root,
    /// which it always does.
    #[must_use]
    pub fn copy_unshared(&self) -> ToyTree {
        let mut results: Vec<ToyTree> = Vec::new();
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
    pub fn list_names_in_pre_order(&self) -> Vec<String> {
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
        Ok(build_toy_node(self.name(), self.value(), children, false))
    }
}

// =============================================================================
// Hook errors
// =============================================================================

/// The error the test visitors and rewriters return.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HookError(pub String);

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
pub enum WalkHook {
    /// [`TreeVisitor::before_visit`].
    Before,
    /// [`TreeVisitor::visit`].
    Visit,
    /// [`TreeVisitor::after_visit`].
    After,
}

impl WalkHook {
    /// Return the event label of the hook.
    fn label(self) -> &'static str {
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
pub struct RecordingVisitor {
    events: Vec<String>,
    pruned: HashSet<String>,
    failure: Option<(WalkHook, String)>,
}

impl RecordingVisitor {
    /// Create a visitor recording every hook call.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Return this visitor pruning the children of the nodes named `name`.
    #[must_use]
    pub fn with_pruned(mut self, name: &str) -> Self {
        self.pruned.insert(name.to_owned());
        self
    }

    /// Return this visitor failing in `hook` on the node named `name`, after
    /// recording the call.
    #[must_use]
    pub fn with_failure(mut self, hook: WalkHook, name: &str) -> Self {
        self.failure = Some((hook, name.to_owned()));
        self
    }

    /// Return the recorded events, in call order.
    #[must_use]
    pub fn events(&self) -> &[String] {
        &self.events
    }

    /// Forget the recorded events.
    pub fn clear_events(&mut self) {
        self.events.clear();
    }

    /// Return the node names `hook` was called on, in call order.
    #[must_use]
    pub fn list_names(&self, hook: WalkHook) -> Vec<String> {
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

impl TreeVisitor<ToyTree> for RecordingVisitor {
    type Error = HookError;

    fn before_visit(&mut self, node: &ToyTree, _cx: &mut PassContext<'_>) -> Result<(), HookError> {
        self.record(WalkHook::Before, node)
    }

    fn visit(&mut self, node: &ToyTree, _cx: &mut PassContext<'_>) -> Result<(), HookError> {
        self.record(WalkHook::Visit, node)
    }

    fn after_visit(&mut self, node: &ToyTree, _cx: &mut PassContext<'_>) -> Result<(), HookError> {
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
pub type RewriteHook = Box<dyn FnMut(&ToyTree) -> Result<Option<ToyTree>, HookError> + Send>;

/// A rewriter whose rewrite is a closure, recording every node it is asked
/// to rewrite.
pub struct ClosureRewriter {
    rewrite: RewriteHook,
    seen: Vec<ToyTree>,
}

impl ClosureRewriter {
    /// Create the rewriter rewriting with `rewrite`.
    pub fn new(
        rewrite: impl FnMut(&ToyTree) -> Result<Option<ToyTree>, HookError> + Send + 'static,
    ) -> Self {
        Self {
            rewrite: Box::new(rewrite),
            seen: Vec::new(),
        }
    }

    /// Return the nodes the rewriter was asked to rewrite, in call order.
    #[must_use]
    pub fn seen(&self) -> &[ToyTree] {
        &self.seen
    }

    /// Return the names of the nodes the rewriter was asked to rewrite, in
    /// call order.
    #[must_use]
    pub fn list_seen_names(&self) -> Vec<String> {
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

impl Rewriter<ToyTree> for ClosureRewriter {
    type Error = HookError;

    fn rewrite(
        &mut self,
        node: &ToyTree,
        _cx: &mut PassContext<'_>,
    ) -> Result<Option<ToyTree>, HookError> {
        self.seen.push(node.clone());
        (self.rewrite)(node)
    }
}

/// Return a rewriter that doubles the integer of every leaf.
#[must_use]
pub fn build_leaf_doubler() -> ClosureRewriter {
    ClosureRewriter::new(|node| {
        Ok(node
            .child_nodes()
            .is_empty()
            .then(|| node.with_value(node.value() * 2)))
    })
}

/// Return a rewriter that replaces every leaf holding `target` by a leaf of
/// the same name holding `replacement`.
#[must_use]
pub fn build_leaf_replacer(target: i64, replacement: i64) -> ClosureRewriter {
    ClosureRewriter::new(move |node| {
        Ok((node.child_nodes().is_empty() && node.value() == target)
            .then(|| node.with_value(replacement)))
    })
}

/// Return a rewriter that keeps every node.
#[must_use]
pub fn build_keeping_rewriter() -> ClosureRewriter {
    ClosureRewriter::new(|_| Ok(None))
}

// =============================================================================
// A pass context for direct traversal calls
// =============================================================================

/// The body a [`ContextLender`] runs.
type LentBody<'a, T> = Box<dyn FnOnce(&mut PassContext<'_>) -> T + 'a>;

/// A pass whose run hands its context to a closure and keeps the result.
struct ContextLender<'a, T> {
    body: Option<LentBody<'a, T>>,
    result: Option<T>,
}

impl<T> CompilerPass<()> for ContextLender<'_, T> {
    fn name(&self) -> String {
        "context-lender".to_owned()
    }

    fn run(&mut self, _ir: &(), cx: &mut PassContext<'_>) -> Result<(), PassFailure> {
        let body = self.body.take().ok_or("the body runs once")?;
        self.result = Some(body(cx));
        Ok(())
    }

    fn did_change(&mut self, _input: &(), _output: &()) -> Result<bool, PassFailure> {
        Ok(false)
    }
}

/// Run `body` with the context of a standalone pass run and return its
/// result.
///
/// # Panics
///
/// Panics if the lending pass fails, which it does not.
pub fn run_with_pass_context<'a, T>(body: impl FnOnce(&mut PassContext<'_>) -> T + 'a) -> T {
    let mut lender = ContextLender {
        body: Some(Box::new(body)),
        result: None,
    };
    lender.execute(&()).expect("the lending pass runs");
    lender.result.expect("the lending pass ran the body")
}
