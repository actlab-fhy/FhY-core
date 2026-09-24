//! Tests for the tree traversals of `fhy_core::tree`: the walk hooks and their
//! order, pruning, failing hooks, the traversal context, the bottom-up
//! memoized rewrite and the handles it keeps, what counts as a change,
//! failing rewrites and rebuilds, the walk and rewrite passes of
//! `fhy_core::pass`, and trees too deep or too shared for a recursive
//! traversal.
//!
//! Public API only, over the toy tree of `support/tree_ir.rs`.

use crate::support::stack as stack_support;
use crate::support::tree_ir;

use std::borrow::Cow;
use std::error::Error;
use std::num::NonZeroUsize;

use fhy_core::diagnostic::Diagnostic;
use fhy_core::identifier::Identifier;
use fhy_core::pass::{
    Analysis, CompilerPass, ExecutePass, FailureClass, FixpointPassGroup, PassContext, PassError,
    PassErrorKind, PassHook, PassManager, PassValidator, PipelineRecord, PreservedAnalyses,
    RewritePass, ValidationManager, WalkPass,
};
use fhy_core::tree::{
    RewriteTreeError, Rewriter, TraversalOrder, TreeVisitor, rewrite_tree, walk_tree,
};
use rstest::rstest;
use stack_support::{SMALL_STACK_DEPTH, run_on_small_stack};
use tree_ir::{
    ClosureRewriter, HookError, RecordingVisitor, ToyRebuildError, ToyTree, WalkHook, build_chain,
    build_doubling_dag, build_frozen_node, build_hash_consing_node, build_keeping_rewriter,
    build_leaf, build_leaf_doubler, build_leaf_hiding_sharing, build_leaf_replacer, build_node,
};

// =============================================================================
// Helpers
// =============================================================================

/// Build `root(left(left_leaf), right)`.
fn build_small_tree() -> ToyTree {
    let left_leaf = build_leaf("left_leaf", 1);
    let left = build_node("left", &[&left_leaf]);
    let right = build_leaf("right", 2);
    build_node("root", &[&left, &right])
}

/// Walk `root` with `visitor` in `order`, returning the walk's result.
fn walk(
    visitor: &mut RecordingVisitor,
    root: &ToyTree,
    order: TraversalOrder,
) -> Result<(), HookError> {
    walk_tree(visitor, root, order, &mut ())
}

/// Rewrite `root` with `rewriter`, returning the rewrite's result.
fn rewrite(
    rewriter: &mut ClosureRewriter,
    root: &ToyTree,
) -> Result<ToyTree, RewriteTreeError<ToyTree, HookError>> {
    rewrite_tree(rewriter, root, &mut ())
}

/// Rewrite `root` with `rewriter`, failing the test if the rewrite fails.
fn rewrite_or_panic(rewriter: &mut ClosureRewriter, root: &ToyTree) -> ToyTree {
    rewrite(rewriter, root).expect("the rewrite succeeds")
}

/// Return the events of a walk calling every hook of the named nodes in
/// order, as `before:<name>`, `visit:<name>`, `after:<name>`.
fn list_events(hooks_and_names: &[(WalkHook, &str)]) -> Vec<String> {
    hooks_and_names
        .iter()
        .map(|(hook, name)| {
            let label = match hook {
                WalkHook::Before => "before",
                WalkHook::Visit => "visit",
                WalkHook::After => "after",
            };
            format!("{label}:{name}")
        })
        .collect()
}

/// A visitor overriding only `before_visit` and `after_visit`.
#[derive(Debug, Default)]
struct BracketVisitor {
    events: Vec<String>,
}

impl TreeVisitor<ToyTree> for BracketVisitor {
    type Error = HookError;

    fn before_visit(&mut self, node: &ToyTree, _cx: &mut ()) -> Result<(), HookError> {
        self.events.push(format!("before:{}", node.name()));
        Ok(())
    }

    fn after_visit(&mut self, node: &ToyTree, _cx: &mut ()) -> Result<(), HookError> {
        self.events.push(format!("after:{}", node.name()));
        Ok(())
    }
}

/// A visitor overriding no hook.
#[derive(Debug, Default)]
struct SilentVisitor;

impl<C: ?Sized> TreeVisitor<ToyTree, C> for SilentVisitor {
    type Error = HookError;
}

/// A visitor reporting a diagnostic for each node it visits.
#[derive(Debug, Default)]
struct ReportingVisitor;

impl TreeVisitor<ToyTree, PassContext<'_>> for ReportingVisitor {
    type Error = HookError;

    fn visit(&mut self, node: &ToyTree, cx: &mut PassContext<'_>) -> Result<(), HookError> {
        cx.report_text(
            fhy_core::diagnostic::DiagnosticLevel::Info,
            format!("visited {}", node.name()),
            None,
        );
        Ok(())
    }
}

/// A visitor named after its type, for the default pass name.
#[derive(Debug, Default)]
struct NameProbeVisitor;

impl<C: ?Sized> TreeVisitor<ToyTree, C> for NameProbeVisitor {
    type Error = HookError;
}

/// Records the occurrence count of every node it visits.
#[derive(Debug, Default)]
struct CountingVisitor {
    counts: Vec<usize>,
}

impl TreeVisitor<ToyTree, PassContext<'_>> for CountingVisitor {
    type Error = HookError;

    fn visit(&mut self, node: &ToyTree, cx: &mut PassContext<'_>) -> Result<(), HookError> {
        self.counts
            .push(*cx.analysis::<OccurrenceCountAnalysis>(node));
        Ok(())
    }
}

/// A rewriter named after its type, for the default pass name.
#[derive(Debug, Default)]
struct NameProbeRewriter;

impl<C: ?Sized> Rewriter<ToyTree, C> for NameProbeRewriter {
    type Error = HookError;

    fn rewrite(&mut self, _node: &ToyTree, _cx: &mut C) -> Result<Option<ToyTree>, HookError> {
        Ok(None)
    }
}

/// Counts the node occurrences of a toy tree.
#[derive(Debug, Default)]
struct OccurrenceCountAnalysis;

impl Analysis for OccurrenceCountAnalysis {
    type Ir = ToyTree;
    type Output = usize;

    fn run(&self, ir: &ToyTree) -> usize {
        ir.count_occurrences()
    }
}

/// Return the rewrite error's rebuild parts, failing the test for another
/// failure.
fn expect_rebuild_failure(
    error: &RewriteTreeError<ToyTree, HookError>,
) -> (&ToyTree, &[ToyTree], &ToyRebuildError) {
    let RewriteTreeError::Rebuild {
        node,
        children,
        source,
        ..
    } = error
    else {
        panic!("expected a rebuild failure, got {error:?}");
    };
    (node, children, source)
}

// =============================================================================
// walk_tree: hook order
// =============================================================================

/// Test a pre-order walk visits each node before its children.
#[test]
fn walk_tree_visits_nodes_in_pre_order() {
    let mut visitor = RecordingVisitor::new();

    walk(&mut visitor, &build_small_tree(), TraversalOrder::Pre).expect("no hook fails");

    assert_eq!(
        visitor.list_names(WalkHook::Visit),
        ["root", "left", "left_leaf", "right"]
    );
}

/// Test a post-order walk visits each node after its children.
#[test]
fn walk_tree_visits_nodes_in_post_order() {
    let mut visitor = RecordingVisitor::new();

    walk(&mut visitor, &build_small_tree(), TraversalOrder::Post).expect("no hook fails");

    assert_eq!(
        visitor.list_names(WalkHook::Visit),
        ["left_leaf", "left", "right", "root"]
    );
}

/// Test a pre-order walk calls before, visit, the children, then after, for
/// every node.
#[test]
fn walk_tree_brackets_every_node_in_pre_order() {
    let tree = build_node("root", &[&build_leaf("left", 1), &build_leaf("right", 2)]);
    let mut visitor = RecordingVisitor::new();

    walk(&mut visitor, &tree, TraversalOrder::Pre).expect("no hook fails");

    assert_eq!(
        visitor.events(),
        list_events(&[
            (WalkHook::Before, "root"),
            (WalkHook::Visit, "root"),
            (WalkHook::Before, "left"),
            (WalkHook::Visit, "left"),
            (WalkHook::After, "left"),
            (WalkHook::Before, "right"),
            (WalkHook::Visit, "right"),
            (WalkHook::After, "right"),
            (WalkHook::After, "root"),
        ])
    );
}

/// Test a post-order walk calls before, the children, visit, then after,
/// for every node.
#[test]
fn walk_tree_brackets_every_node_in_post_order() {
    let tree = build_node("root", &[&build_leaf("left", 1), &build_leaf("right", 2)]);
    let mut visitor = RecordingVisitor::new();

    walk(&mut visitor, &tree, TraversalOrder::Post).expect("no hook fails");

    assert_eq!(
        visitor.events(),
        list_events(&[
            (WalkHook::Before, "root"),
            (WalkHook::Before, "left"),
            (WalkHook::Visit, "left"),
            (WalkHook::After, "left"),
            (WalkHook::Before, "right"),
            (WalkHook::Visit, "right"),
            (WalkHook::After, "right"),
            (WalkHook::Visit, "root"),
            (WalkHook::After, "root"),
        ])
    );
}

/// Test the traversal order defaults to pre-order.
#[test]
fn traversal_order_defaults_to_pre_order() {
    let order = TraversalOrder::default();

    assert_eq!(order, TraversalOrder::Pre);
}

/// Test a visitor overriding only the bracketing hooks walks every node with
/// the default visit doing nothing.
#[test]
fn walk_tree_runs_the_bracketing_hooks_around_a_default_visit() {
    let tree = build_node("root", &[&build_leaf("child", 1)]);
    let mut visitor = BracketVisitor::default();

    walk_tree(&mut visitor, &tree, TraversalOrder::Pre, &mut ()).expect("no hook fails");

    assert_eq!(
        visitor.events,
        ["before:root", "before:child", "after:child", "after:root"]
    );
}

/// Test a shared node is walked at each of its occurrences.
#[test]
fn walk_tree_walks_a_shared_node_at_each_occurrence() {
    let shared = build_node("shared", &[&build_leaf("leaf", 1)]);
    let tree = build_node("root", &[&shared, &shared]);
    let mut visitor = RecordingVisitor::new();

    walk(&mut visitor, &tree, TraversalOrder::Pre).expect("no hook fails");

    assert_eq!(
        visitor.list_names(WalkHook::Visit),
        ["root", "shared", "leaf", "shared", "leaf"]
    );
}

/// Test a walk of a single leaf calls each hook once.
#[rstest]
#[case::pre(TraversalOrder::Pre)]
#[case::post(TraversalOrder::Post)]
fn walk_tree_calls_each_hook_once_on_a_leaf(#[case] order: TraversalOrder) {
    let mut visitor = RecordingVisitor::new();

    walk(&mut visitor, &build_leaf("leaf", 1), order).expect("no hook fails");

    assert_eq!(
        visitor.events(),
        ["before:leaf", "visit:leaf", "after:leaf"]
    );
}

// =============================================================================
// walk_tree: pruning
// =============================================================================

/// Test a node whose children the visitor does not walk still has its own
/// hooks called, and its children are skipped.
#[rstest]
#[case::pre(
    TraversalOrder::Pre,
    &[
        (WalkHook::Before, "root"),
        (WalkHook::Visit, "root"),
        (WalkHook::Before, "left"),
        (WalkHook::Visit, "left"),
        (WalkHook::After, "left"),
        (WalkHook::Before, "right"),
        (WalkHook::Visit, "right"),
        (WalkHook::After, "right"),
        (WalkHook::After, "root"),
    ],
)]
#[case::post(
    TraversalOrder::Post,
    &[
        (WalkHook::Before, "root"),
        (WalkHook::Before, "left"),
        (WalkHook::Visit, "left"),
        (WalkHook::After, "left"),
        (WalkHook::Before, "right"),
        (WalkHook::Visit, "right"),
        (WalkHook::After, "right"),
        (WalkHook::Visit, "root"),
        (WalkHook::After, "root"),
    ],
)]
fn walk_tree_prunes_only_the_children_of_a_pruned_node(
    #[case] order: TraversalOrder,
    #[case] expected: &[(WalkHook, &str)],
) {
    let mut visitor = RecordingVisitor::new().with_pruned("left");

    walk(&mut visitor, &build_small_tree(), order).expect("no hook fails");

    assert_eq!(visitor.events(), list_events(expected));
}

/// Test pruning the root walks the root alone.
#[test]
fn walk_tree_pruning_the_root_walks_the_root_alone() {
    let mut visitor = RecordingVisitor::new().with_pruned("root");

    walk(&mut visitor, &build_small_tree(), TraversalOrder::Pre).expect("no hook fails");

    assert_eq!(
        visitor.events(),
        ["before:root", "visit:root", "after:root"]
    );
}

// =============================================================================
// walk_tree: failures
// =============================================================================

/// Test the first failing hook ends the walk with its error and no hook
/// runs after it, whatever the hook and the order.
#[rstest]
#[case::pre_before(
    TraversalOrder::Pre,
    WalkHook::Before,
    &[
        (WalkHook::Before, "root"),
        (WalkHook::Visit, "root"),
        (WalkHook::Before, "boom"),
    ],
)]
#[case::pre_visit(
    TraversalOrder::Pre,
    WalkHook::Visit,
    &[
        (WalkHook::Before, "root"),
        (WalkHook::Visit, "root"),
        (WalkHook::Before, "boom"),
        (WalkHook::Visit, "boom"),
    ],
)]
#[case::pre_after(
    TraversalOrder::Pre,
    WalkHook::After,
    &[
        (WalkHook::Before, "root"),
        (WalkHook::Visit, "root"),
        (WalkHook::Before, "boom"),
        (WalkHook::Visit, "boom"),
        (WalkHook::Before, "leaf"),
        (WalkHook::Visit, "leaf"),
        (WalkHook::After, "leaf"),
        (WalkHook::After, "boom"),
    ],
)]
#[case::post_before(
    TraversalOrder::Post,
    WalkHook::Before,
    &[(WalkHook::Before, "root"), (WalkHook::Before, "boom")],
)]
#[case::post_visit(
    TraversalOrder::Post,
    WalkHook::Visit,
    &[
        (WalkHook::Before, "root"),
        (WalkHook::Before, "boom"),
        (WalkHook::Before, "leaf"),
        (WalkHook::Visit, "leaf"),
        (WalkHook::After, "leaf"),
        (WalkHook::Visit, "boom"),
    ],
)]
#[case::post_after(
    TraversalOrder::Post,
    WalkHook::After,
    &[
        (WalkHook::Before, "root"),
        (WalkHook::Before, "boom"),
        (WalkHook::Before, "leaf"),
        (WalkHook::Visit, "leaf"),
        (WalkHook::After, "leaf"),
        (WalkHook::Visit, "boom"),
        (WalkHook::After, "boom"),
    ],
)]
fn walk_tree_stops_at_the_first_failing_hook(
    #[case] order: TraversalOrder,
    #[case] failing_hook: WalkHook,
    #[case] expected: &[(WalkHook, &str)],
) {
    let boom = build_node("boom", &[&build_leaf("leaf", 1)]);
    let tree = build_node("root", &[&boom, &build_leaf("sibling", 2)]);
    let mut visitor = RecordingVisitor::new().with_failure(failing_hook, "boom");

    let result = walk(&mut visitor, &tree, order);

    let failure = result.expect_err("the hook fails");
    assert_eq!(visitor.events(), list_events(expected));
    assert_eq!(
        failure,
        HookError(format!(
            "{} failed",
            visitor.events().last().expect("a call")
        ))
    );
}

// =============================================================================
// rewrite_tree: identity
// =============================================================================

/// Test a rewriter that keeps every node returns the root itself.
#[rstest]
#[case::leaf(build_leaf("leaf", 5))]
#[case::unary(build_node("unary", &[&build_leaf("leaf", 3)]))]
#[case::pair(build_node("pair", &[&build_leaf("a", 1), &build_leaf("b", 2)]))]
#[case::list(build_node(
    "list",
    &[&build_leaf("a", 1), &build_leaf("b", 2), &build_leaf("c", 3)],
))]
#[case::empty_list(build_node("list", &[]))]
#[case::nested(build_node(
    "pair",
    &[
        &build_node("unary", &[&build_leaf("a", 7)]),
        &build_node(
            "list",
            &[
                &build_leaf("b", 1),
                &build_node("pair", &[&build_leaf("c", 2), &build_leaf("d", 3)]),
            ],
        ),
    ],
))]
fn rewrite_tree_keeping_every_node_returns_the_root_itself(#[case] tree: ToyTree) {
    let mut rewriter = build_keeping_rewriter();

    let output = rewrite_or_panic(&mut rewriter, &tree);

    assert!(output.is_same_node(&tree));
}

/// Test a node with no children that no rewrite touches is returned
/// itself, never rebuilt.
#[test]
fn rewrite_tree_returns_an_untouched_childless_node_itself() {
    let tree = build_frozen_node("list", &[]);
    let mut rewriter = build_leaf_replacer(9, 0);

    let output = rewrite_or_panic(&mut rewriter, &tree);

    assert!(output.is_same_node(&tree));
}

/// Test a rewriter returning `None` for the root returns the root itself.
#[test]
fn rewrite_tree_returns_the_root_when_the_rewriter_keeps_it() {
    let tree = build_leaf("leaf", 3);
    let mut rewriter = build_leaf_replacer(4, 0);

    let output = rewrite_or_panic(&mut rewriter, &tree);

    assert!(output.is_same_node(&tree));
}

/// Test a rewriter's replacement of the root is the output.
#[test]
fn rewrite_tree_returns_the_replacement_of_the_root() {
    let tree = build_leaf("leaf", 3);
    let mut rewriter = build_leaf_doubler();

    let output = rewrite_or_panic(&mut rewriter, &tree);

    assert!(!output.is_same_node(&tree));
    assert_eq!(output, build_leaf("leaf", 6));
}

// =============================================================================
// rewrite_tree: rewriting below the root
// =============================================================================

/// Test the leaves at every position are rewritten and their ancestors
/// rebuilt around them.
#[rstest]
#[case::unary(
    build_node("unary", &[&build_leaf("a", 5)]),
    build_node("unary", &[&build_leaf("a", 10)]),
)]
#[case::pair(
    build_node("pair", &[&build_leaf("a", 1), &build_leaf("b", 2)]),
    build_node("pair", &[&build_leaf("a", 2), &build_leaf("b", 4)]),
)]
#[case::list(
    build_node("list", &[&build_leaf("a", 1), &build_leaf("b", 2), &build_leaf("c", 3)]),
    build_node("list", &[&build_leaf("a", 2), &build_leaf("b", 4), &build_leaf("c", 6)]),
)]
fn rewrite_tree_rewrites_the_leaves_under_a_node(#[case] tree: ToyTree, #[case] expected: ToyTree) {
    let mut rewriter = build_leaf_doubler();

    let output = rewrite_or_panic(&mut rewriter, &tree);

    assert!(!output.is_same_node(&tree));
    assert_eq!(output, expected);
}

/// Test a rewriter may replace a node by its own child.
#[test]
fn rewrite_tree_replaces_a_node_by_its_child() {
    let operand = build_leaf("operand", 7);
    let tree = build_node("unary", &[&operand]);
    let mut rewriter =
        ClosureRewriter::new(|node| Ok((node.name() == "unary").then(|| node.child(0).clone())));

    let output = rewrite_or_panic(&mut rewriter, &tree);

    assert!(output.is_same_node(&operand));
}

/// Test a rewriter may replace an inner node by a new leaf.
#[test]
fn rewrite_tree_replaces_an_inner_node_by_a_leaf() {
    let tree = build_node("pair", &[&build_leaf("a", 1), &build_leaf("b", 2)]);
    let mut rewriter =
        ClosureRewriter::new(
            |node| Ok((node.name() == "pair").then(|| build_leaf("sentinel", 99))),
        );

    let output = rewrite_or_panic(&mut rewriter, &tree);

    assert_eq!(output, build_leaf("sentinel", 99));
}

/// Test the children a rewrite leaves alone keep their handles in the
/// rebuilt parent.
#[test]
fn rewrite_tree_keeps_the_handles_of_untouched_siblings() {
    let second = build_leaf("b", 2);
    let third = build_node("c", &[&build_leaf("d", 3)]);
    let tree = build_node("list", &[&build_leaf("a", 1), &second, &third]);
    let mut rewriter = build_leaf_replacer(1, 100);

    let output = rewrite_or_panic(&mut rewriter, &tree);

    assert!(!output.is_same_node(&tree));
    assert_eq!(output.child(0), &build_leaf("a", 100));
    assert!(output.child(1).is_same_node(&second));
    assert!(output.child(2).is_same_node(&third));
}

/// Test a deep untouched subtree keeps its handle, and so every handle
/// within it, beside a rewritten sibling.
#[test]
fn rewrite_tree_keeps_a_deep_untouched_subtree() {
    let deep = build_chain("unary", &build_leaf("deep", 99), 3);
    let tree = build_node("pair", &[&build_leaf("a", 7), &deep]);
    let mut rewriter = build_leaf_replacer(7, 0);

    let output = rewrite_or_panic(&mut rewriter, &tree);

    assert_eq!(output.child(0), &build_leaf("a", 0));
    assert!(output.child(1).is_same_node(&deep));
}

/// Test a rewriter returning every node itself changes nothing: no parent
/// is rebuilt, and the output is the input itself.
#[test]
fn rewrite_tree_counts_a_node_returned_as_itself_as_unchanged() {
    let leaf = build_leaf("leaf", 1);
    let tree = build_node("unary", &[&leaf]);
    let mut rewriter = ClosureRewriter::new(|node| Ok(Some(node.clone())));

    let output = rewrite_or_panic(&mut rewriter, &tree);

    assert!(output.is_same_node(&tree));
    assert_eq!(rewriter.list_seen_names(), ["leaf", "unary"]);
    assert!(rewriter.seen()[1].is_same_node(&tree));
}

/// Test a rewriter that returns the original parent after its children
/// changed reverts the change: the output is the input itself.
#[test]
fn rewrite_tree_counts_a_reverted_node_as_unchanged() {
    let tree = build_node("unary", &[&build_leaf("leaf", 1)]);
    let original = tree.clone();
    let mut rewriter = ClosureRewriter::new(move |node| {
        Ok(Some(if node.name() == "unary" {
            original.clone()
        } else {
            node.with_value(node.value() * 2)
        }))
    });

    let output = rewrite_or_panic(&mut rewriter, &tree);

    assert!(output.is_same_node(&tree));
    let rebuilt = &rewriter.seen()[1];
    assert!(!rebuilt.is_same_node(&tree));
    assert_eq!(rebuilt.child(0), &build_leaf("leaf", 2));
}

/// Test a reverted child is no change for its parent: the parent is not
/// rebuilt, and the output is the input itself.
#[test]
fn rewrite_tree_does_not_rebuild_the_parent_of_a_reverted_child() {
    let leaf = build_leaf("leaf", 1);
    let tree = build_frozen_node("frozen", &[&build_node("unary", &[&leaf])]);
    let reverted = tree.child(0).clone();
    let mut rewriter = ClosureRewriter::new(move |node| {
        Ok(match node.name() {
            "unary" => Some(reverted.clone()),
            "leaf" => Some(node.with_value(5)),
            _ => None,
        })
    });

    let output = rewrite_or_panic(&mut rewriter, &tree);

    assert!(output.is_same_node(&tree));
}

/// Test a rebuild that returns the node itself, as a hash-consing IR does
/// for children equal to its own, counts as no change.
#[test]
fn rewrite_tree_counts_a_rebuild_returning_the_node_itself_as_unchanged() {
    let tree = build_hash_consing_node("interned", &[&build_leaf("a", 1), &build_leaf("b", 2)]);
    let mut rewriter = ClosureRewriter::new(|node| {
        Ok(node
            .child_nodes()
            .is_empty()
            .then(|| node.with_value(node.value())))
    });

    let output = rewrite_or_panic(&mut rewriter, &tree);

    assert!(output.is_same_node(&tree));
    assert!(rewriter.seen()[2].is_same_node(&tree));
}

// =============================================================================
// rewrite_tree: bottom-up order
// =============================================================================

/// Test the rewriter sees every child before its parent, children in order.
#[rstest]
#[case::pair(
    build_node("pair", &[&build_leaf("a", 1), &build_leaf("b", 2)]),
    &["a", "b", "pair"],
)]
#[case::list(
    build_node("list", &[&build_leaf("a", 1), &build_leaf("b", 2), &build_leaf("c", 3)]),
    &["a", "b", "c", "list"],
)]
#[case::unary_over_pair(
    build_node("unary", &[&build_node("pair", &[&build_leaf("a", 1), &build_leaf("b", 2)])]),
    &["a", "b", "pair", "unary"],
)]
fn rewrite_tree_rewrites_children_before_their_parent(
    #[case] tree: ToyTree,
    #[case] expected: &[&str],
) {
    let mut rewriter = build_keeping_rewriter();

    rewrite_or_panic(&mut rewriter, &tree);

    assert_eq!(rewriter.list_seen_names(), expected);
}

/// Test the rewriter sees a parent rebuilt around its rewritten children.
#[rstest]
#[case::pair(
    build_node("pair", &[&build_leaf("a", 5), &build_leaf("b", 7)]),
    build_node("pair", &[&build_leaf("a", 10), &build_leaf("b", 14)]),
)]
#[case::list(
    build_node("list", &[&build_leaf("a", 3), &build_leaf("b", 4)]),
    build_node("list", &[&build_leaf("a", 6), &build_leaf("b", 8)]),
)]
fn rewrite_tree_shows_the_parent_its_rewritten_children(
    #[case] tree: ToyTree,
    #[case] expected_parent: ToyTree,
) {
    let mut rewriter = build_leaf_doubler();

    rewrite_or_panic(&mut rewriter, &tree);

    let parent = rewriter.seen().last().expect("the parent was seen");
    assert_eq!(parent, &expected_parent);
    assert!(!parent.is_same_node(&tree));
}

/// Test the rewriter sees an unchanged parent as the input node itself.
#[test]
fn rewrite_tree_shows_an_unchanged_parent_as_itself() {
    let tree = build_node("pair", &[&build_leaf("a", 1), &build_leaf("b", 2)]);
    let mut rewriter = build_leaf_replacer(9, 0);

    rewrite_or_panic(&mut rewriter, &tree);

    let parent = rewriter.seen().last().expect("the parent was seen");
    assert!(parent.is_same_node(&tree));
}

/// Test a replacement is not rewritten again: the pair's replacement leaf
/// is not doubled.
#[test]
fn rewrite_tree_does_not_rewrite_a_replacement_again() {
    let tree = build_node("pair", &[&build_leaf("a", 1), &build_leaf("b", 2)]);
    let mut rewriter = ClosureRewriter::new(|node| {
        Ok(if node.name() == "pair" {
            Some(build_leaf("five", 5))
        } else {
            Some(node.with_value(node.value() * 2))
        })
    });

    let output = rewrite_or_panic(&mut rewriter, &tree);

    assert_eq!(output, build_leaf("five", 5));
    assert_eq!(rewriter.list_seen_names(), ["a", "b", "pair"]);
}

// =============================================================================
// rewrite_tree: shared nodes
// =============================================================================

/// Test a shared subtree is rewritten once and its result reused at every
/// occurrence.
#[test]
fn rewrite_tree_rewrites_a_shared_subtree_once() {
    let shared = build_node("shared", &[&build_leaf("leaf", 1)]);
    let tree = build_node("root", &[&shared, &shared]);
    let mut rewriter = build_leaf_doubler();

    let output = rewrite_or_panic(&mut rewriter, &tree);

    assert_eq!(rewriter.list_seen_names(), ["leaf", "shared", "root"]);
    assert!(output.child(0).is_same_node(output.child(1)));
    assert_eq!(
        output.child(0),
        &build_node("shared", &[&build_leaf("leaf", 2)])
    );
}

/// Test an untouched shared subtree keeps its handle at every occurrence.
#[test]
fn rewrite_tree_keeps_an_untouched_shared_subtree_at_every_occurrence() {
    let shared = build_node("shared", &[&build_leaf("leaf", 1)]);
    let tree = build_node("root", &[&shared, &build_leaf("other", 2), &shared]);
    let mut rewriter = build_leaf_replacer(2, 20);

    let output = rewrite_or_panic(&mut rewriter, &tree);

    assert!(output.child(0).is_same_node(&shared));
    assert!(output.child(2).is_same_node(&shared));
    assert_eq!(output.child(1), &build_leaf("other", 20));
}

/// Test a node reporting itself unshared is rewritten at each of its
/// occurrences.
#[test]
fn rewrite_tree_rewrites_a_node_reporting_itself_unshared_at_each_occurrence() {
    let hidden = build_leaf_hiding_sharing("hidden", 1);
    let tree = build_node("root", &[&hidden, &hidden]);
    let mut rewriter = build_leaf_doubler();

    let output = rewrite_or_panic(&mut rewriter, &tree);

    assert_eq!(rewriter.list_seen_names(), ["hidden", "hidden", "root"]);
    assert_eq!(
        output,
        build_node(
            "root",
            &[&build_leaf("hidden", 2), &build_leaf("hidden", 2)]
        )
    );
    assert!(!output.child(0).is_same_node(output.child(1)));
}

/// Test a DAG of 65 distinct nodes and 2^65 - 1 occurrences is rewritten
/// with one rewriter call per distinct node, and the output shares its
/// nodes the same way.
#[test]
fn rewrite_tree_rewrites_a_doubling_dag_once_per_distinct_node() {
    let levels = 64;
    let dag = build_doubling_dag("sum", &build_leaf("x", 1), levels);
    let mut rewriter = build_leaf_doubler();

    let output = rewrite_or_panic(&mut rewriter, &dag);

    assert_eq!(rewriter.seen().len(), levels + 1);
    assert_eq!(output.count_distinct_nodes(), levels + 1);
    let mut node = &output;
    for _ in 0..levels {
        assert!(node.child(0).is_same_node(node.child(1)));
        node = node.child(0);
    }
    assert_eq!(node, &build_leaf("x", 2));
}

// =============================================================================
// rewrite_tree: failures
// =============================================================================

/// Test a failing rewrite ends the rewrite with its error, and no node after
/// it is rewritten.
#[test]
fn rewrite_tree_stops_at_the_first_failing_rewrite() {
    let tree = build_node("pair", &[&build_leaf("a", 1), &build_leaf("b", 2)]);
    let mut rewriter = ClosureRewriter::new(|node| {
        if node.name() == "a" {
            Err(HookError("a failed".to_owned()))
        } else {
            Ok(None)
        }
    });

    let result = rewrite(&mut rewriter, &tree);

    let Err(RewriteTreeError::Rewrite(failure)) = &result else {
        panic!("expected a rewrite failure, got {result:?}");
    };
    assert_eq!(failure, &HookError("a failed".to_owned()));
    assert_eq!(rewriter.list_seen_names(), ["a"]);
}

/// Test a node refusing to be rebuilt around a rewritten child ends the
/// rewrite with the refusal, naming the node and the children it was given.
#[test]
fn rewrite_tree_reports_a_refused_rebuild() {
    let kept = build_leaf("kept", 2);
    let tree = build_frozen_node("frozen", &[&build_leaf("a", 1), &kept]);
    let mut rewriter = build_leaf_replacer(1, 10);

    let result = rewrite(&mut rewriter, &tree);

    let error = result.expect_err("the rebuild is refused");
    let (node, children, source) = expect_rebuild_failure(&error);
    assert!(node.is_same_node(&tree));
    assert_eq!(children, [build_leaf("a", 10), kept.clone()]);
    assert!(children[1].is_same_node(&kept));
    assert_eq!(
        source,
        &ToyRebuildError::Frozen {
            name: "frozen".to_owned()
        }
    );
}

/// Test a refused rebuild stops the rewrite before the refusing node or any
/// later node is rewritten.
#[test]
fn rewrite_tree_stops_at_a_refused_rebuild() {
    let frozen = build_frozen_node("frozen", &[&build_leaf("a", 1)]);
    let tree = build_node("root", &[&frozen, &build_leaf("later", 1)]);
    let mut rewriter = build_leaf_replacer(1, 10);

    let result = rewrite(&mut rewriter, &tree);

    assert!(
        matches!(result, Err(RewriteTreeError::Rebuild { .. })),
        "got {result:?}"
    );
    assert_eq!(rewriter.list_seen_names(), ["a"]);
}

/// Test a rewrite failure is transparent: it displays the rewriter's error
/// and passes on that error's source, here none.
#[test]
fn rewrite_tree_error_describes_a_failing_rewrite() {
    let error: RewriteTreeError<ToyTree, HookError> =
        RewriteTreeError::Rewrite(HookError("inner".to_owned()));

    let message = error.to_string();

    assert_eq!(message, "inner");
    assert!(error.source().is_none());
}

/// Test a rewrite failure passes on the source of the rewriter's error.
#[test]
fn rewrite_tree_error_passes_on_the_source_of_the_rewriters_error() {
    let error: RewriteTreeError<ToyTree, PassError> = RewriteTreeError::Rewrite(
        WalkPass::new(
            RecordingVisitor::new().with_failure(WalkHook::Visit, "leaf"),
            TraversalOrder::Pre,
        )
        .execute(&build_leaf("leaf", 1))
        .expect_err("the visit fails"),
    );

    let source = error.source().expect("the pass error has a source");

    assert_eq!(
        source.downcast_ref::<HookError>(),
        Some(&HookError("visit:leaf failed".to_owned()))
    );
}

/// Test a rebuild failure displays its message and exposes the rebuild error
/// as the source.
#[test]
fn rewrite_tree_error_describes_a_refused_rebuild() {
    let refusal = ToyRebuildError::Frozen {
        name: "frozen".to_owned(),
    };
    let mut rewriter = build_leaf_doubler();
    let error = rewrite(
        &mut rewriter,
        &build_frozen_node("frozen", &[&build_leaf("a", 1)]),
    )
    .expect_err("the rebuild is refused");

    let message = error.to_string();

    assert_eq!(
        message,
        "rebuilding a node around its rewritten children failed"
    );
    let source = error.source().expect("a rebuild failure has a source");
    assert_eq!(source.downcast_ref::<ToyRebuildError>(), Some(&refusal));
}

// =============================================================================
// The traversal context
// =============================================================================

/// Counts the leaves it visits, needing no context.
#[derive(Debug, Default)]
struct LeafCounter {
    leaves: usize,
}

impl TreeVisitor<ToyTree> for LeafCounter {
    type Error = HookError;

    fn visit(&mut self, node: &ToyTree, _cx: &mut ()) -> Result<(), HookError> {
        if node.child_nodes().is_empty() {
            self.leaves += 1;
        }
        Ok(())
    }
}

/// Negates every leaf, needing no context.
#[derive(Debug, Default)]
struct LeafNegator;

impl Rewriter<ToyTree> for LeafNegator {
    type Error = HookError;

    fn rewrite(&mut self, node: &ToyTree, _cx: &mut ()) -> Result<Option<ToyTree>, HookError> {
        Ok(node
            .child_nodes()
            .is_empty()
            .then(|| node.with_value(-node.value())))
    }
}

/// Logs the name of every node it visits or rewrites into its context, and
/// doubles every leaf.
#[derive(Debug, Default)]
struct LoggingTraversal;

impl TreeVisitor<ToyTree, Vec<String>> for LoggingTraversal {
    type Error = HookError;

    fn visit(&mut self, node: &ToyTree, log: &mut Vec<String>) -> Result<(), HookError> {
        log.push(format!("visit:{}", node.name()));
        Ok(())
    }
}

impl Rewriter<ToyTree, Vec<String>> for LoggingTraversal {
    type Error = HookError;

    fn rewrite(
        &mut self,
        node: &ToyTree,
        log: &mut Vec<String>,
    ) -> Result<Option<ToyTree>, HookError> {
        log.push(format!("rewrite:{}", node.name()));
        Ok(node
            .child_nodes()
            .is_empty()
            .then(|| node.with_value(node.value() * 2)))
    }
}

/// Test a visitor and a rewriter written without a context run with the
/// unit context, outside any pass.
#[test]
fn walk_tree_and_rewrite_tree_run_without_a_pass_context() {
    let tree = build_small_tree();
    let mut counter = LeafCounter::default();
    let mut negator = LeafNegator;

    walk_tree(&mut counter, &tree, TraversalOrder::Pre, &mut ()).expect("no hook fails");
    let output = rewrite_tree(&mut negator, &tree, &mut ()).expect("no rewrite fails");

    assert_eq!(counter.leaves, 2);
    assert_eq!(
        output,
        build_node(
            "root",
            &[
                &build_node("left", &[&build_leaf("left_leaf", -1)]),
                &build_leaf("right", -2)
            ]
        )
    );
}

/// Test the walk and the rewrite hand the caller's context to every hook,
/// in hook order, and leave it otherwise untouched.
#[test]
fn rewrite_tree_threads_a_caller_chosen_context() {
    let tree = build_small_tree();
    let mut traversal = LoggingTraversal;
    let mut log = vec!["start".to_owned()];

    walk_tree(&mut traversal, &tree, TraversalOrder::Post, &mut log).expect("no hook fails");
    let output = rewrite_tree(&mut traversal, &tree, &mut log).expect("no rewrite fails");

    assert_eq!(
        log,
        [
            "start",
            "visit:left_leaf",
            "visit:left",
            "visit:right",
            "visit:root",
            "rewrite:left_leaf",
            "rewrite:left",
            "rewrite:right",
            "rewrite:root",
        ]
    );
    assert_eq!(output.child(1), &build_leaf("right", 4));
}

// =============================================================================
// Deep trees
// =============================================================================

/// Test walking a chain [`SMALL_STACK_DEPTH`] nodes deep on a small stack
/// calls every hook once per node, in both orders.
#[rstest]
#[case::pre(TraversalOrder::Pre)]
#[case::post(TraversalOrder::Post)]
fn walk_tree_walks_a_deep_chain_on_a_small_stack(#[case] order: TraversalOrder) {
    let event_count = run_on_small_stack(move || {
        let chain = build_chain("link", &build_leaf("bottom", 0), SMALL_STACK_DEPTH);
        let mut visitor = RecordingVisitor::new();

        walk(&mut visitor, &chain, order).expect("no hook fails");

        visitor.events().len()
    });

    assert_eq!(event_count, 3 * (SMALL_STACK_DEPTH + 1));
}

/// Test rewriting the bottom of a chain [`SMALL_STACK_DEPTH`] nodes deep on
/// a small stack rebuilds every node above it.
#[test]
fn rewrite_tree_rebuilds_a_deep_chain_on_a_small_stack() {
    let (is_rebuilt, is_equal_to_expected) = run_on_small_stack(|| {
        let chain = build_chain("link", &build_leaf("bottom", 1), SMALL_STACK_DEPTH);
        let expected = build_chain("link", &build_leaf("bottom", 2), SMALL_STACK_DEPTH);
        let mut rewriter = build_leaf_doubler();

        let output = rewrite_or_panic(&mut rewriter, &chain);

        (!output.is_same_node(&chain), output == expected)
    });

    assert!(is_rebuilt);
    assert!(is_equal_to_expected);
}

/// Test a rewrite of a chain [`SMALL_STACK_DEPTH`] nodes deep that changes
/// nothing returns the chain itself on a small stack.
#[test]
fn rewrite_tree_keeps_a_deep_chain_on_a_small_stack() {
    let is_same = run_on_small_stack(|| {
        let chain = build_chain("link", &build_leaf("bottom", 1), SMALL_STACK_DEPTH);
        let mut rewriter = build_keeping_rewriter();

        rewrite_or_panic(&mut rewriter, &chain).is_same_node(&chain)
    });

    assert!(is_same);
}

// =============================================================================
// WalkPass
// =============================================================================

/// Test executing a walk pass walks its input in the pass's order and
/// outputs unit, unchanged.
#[rstest]
#[case::pre(TraversalOrder::Pre, &["root", "left", "left_leaf", "right"])]
#[case::post(TraversalOrder::Post, &["left_leaf", "left", "right", "root"])]
fn walk_pass_execute_walks_the_input_unchanged(
    #[case] order: TraversalOrder,
    #[case] expected: &[&str],
) {
    let mut pass = WalkPass::new(RecordingVisitor::new(), order);

    let outcome = pass.execute(&build_small_tree()).expect("no hook fails");

    assert_eq!(outcome.output(), &());
    assert!(!outcome.is_changed());
    assert!(outcome.preserved_analyses().preserves_all());
    assert_eq!(pass.visitor().list_names(WalkHook::Visit), expected);
}

/// Test a walk pass over a visitor overriding no hook outputs unit.
#[test]
fn walk_pass_with_default_hooks_outputs_unit() {
    let mut pass = WalkPass::new(SilentVisitor, TraversalOrder::Pre);

    let outcome = pass.execute(&build_small_tree()).expect("no hook fails");

    assert_eq!(outcome.output(), &());
    assert!(!outcome.is_changed());
}

/// Test a failing hook fails the walk pass's run, with the hook's error as
/// the source, after the walk stopped.
#[test]
fn walk_pass_execute_fails_with_the_hook_error() {
    let tree = build_node("root", &[&build_leaf("boom", 1), &build_leaf("later", 2)]);
    let mut pass = WalkPass::new(
        RecordingVisitor::new().with_failure(WalkHook::Visit, "boom"),
        TraversalOrder::Pre,
    );

    let result = pass.execute(&tree);

    let error = result.expect_err("the visit fails");
    assert!(
        matches!(
            error.kind(),
            PassErrorKind::Hook {
                hook: PassHook::Run,
                ..
            }
        ),
        "{error:?}"
    );
    assert_eq!(error.class(), FailureClass::Execution);
    let source = error.source().expect("the hook error is the source");
    assert_eq!(
        source.downcast_ref::<HookError>(),
        Some(&HookError("visit:boom failed".to_owned()))
    );
    assert_eq!(
        pass.visitor().events(),
        ["before:root", "visit:root", "before:boom", "visit:boom"]
    );
}

/// Test a walk pass's hooks report into the run's diagnostics.
#[test]
fn walk_pass_hooks_report_into_the_run() {
    let mut pass = WalkPass::new(ReportingVisitor, TraversalOrder::Pre);

    let outcome = pass.execute(&build_small_tree()).expect("no hook fails");

    let messages: Vec<String> = outcome
        .diagnostics()
        .iter()
        .map(|diagnostic| diagnostic.message_text().to_owned())
        .collect();
    assert_eq!(
        messages,
        [
            "visited root",
            "visited left",
            "visited left_leaf",
            "visited right"
        ]
    );
}

/// Test a walk pass never reports a change.
#[test]
fn walk_pass_did_change_is_false() {
    let mut pass = WalkPass::new(SilentVisitor, TraversalOrder::Pre);

    let changed = CompilerPass::<ToyTree, ()>::did_change(&mut pass, &build_small_tree(), &());

    assert!(!changed.expect("the comparison succeeds"));
}

/// Test the walk pass hands out its visitor by reference, for mutation, and
/// by value.
#[test]
fn walk_pass_exposes_its_visitor() {
    let mut pass = WalkPass::new(RecordingVisitor::new(), TraversalOrder::Pre);
    pass.execute(&build_leaf("first", 1))
        .expect("no hook fails");
    assert_eq!(pass.visitor().list_names(WalkHook::Visit), ["first"]);

    pass.visitor_mut().clear_events();
    pass.execute(&build_leaf("second", 2))
        .expect("no hook fails");
    let visitor = pass.into_visitor();

    assert_eq!(
        visitor.events(),
        ["before:second", "visit:second", "after:second"]
    );
}

/// Test a walk pass is named after its visitor's type, borrowing the name.
#[test]
fn walk_pass_is_named_after_its_visitor() {
    let pass = WalkPass::new(NameProbeVisitor, TraversalOrder::Pre);

    let name = CompilerPass::<ToyTree, ()>::name(&pass);

    assert!(matches!(name, Cow::Borrowed("NameProbeVisitor")));
    assert_eq!(
        CompilerPass::<ToyTree, ()>::description(&pass),
        "NameProbeVisitor"
    );
}

// =============================================================================
// RewritePass
// =============================================================================

/// Test executing a rewrite pass outputs the rewritten tree, changed, with
/// no analysis preserved.
#[test]
fn rewrite_pass_execute_reports_a_rewrite() {
    let mut pass = RewritePass::new(build_leaf_doubler());

    let outcome = pass
        .execute(&build_leaf("leaf", 5))
        .expect("no rewrite fails");

    assert_eq!(outcome.output(), &build_leaf("leaf", 10));
    assert!(outcome.is_changed());
    assert!(!outcome.preserved_analyses().preserves_all());
    assert_eq!(outcome.preserved_analyses(), &PreservedAnalyses::none());
}

/// Test executing a rewrite pass that changes nothing outputs its input
/// itself, unchanged, with every analysis preserved.
#[test]
fn rewrite_pass_execute_reports_no_change() {
    let tree = build_leaf("leaf", 5);
    let mut pass = RewritePass::new(build_keeping_rewriter());

    let outcome = pass.execute(&tree).expect("no rewrite fails");

    assert!(outcome.output().is_same_node(&tree));
    assert!(!outcome.is_changed());
    assert!(outcome.preserved_analyses().preserves_all());
}

/// Test a rewrite pass returning its input's root for the root reports no
/// change.
#[test]
fn rewrite_pass_reports_no_change_for_a_root_returned_as_itself() {
    let tree = build_node("unary", &[&build_leaf("leaf", 1)]);
    let mut pass = RewritePass::new(ClosureRewriter::new(|node| {
        Ok((node.name() == "unary").then(|| node.clone()))
    }));

    let outcome = pass.execute(&tree).expect("no rewrite fails");

    assert!(outcome.output().is_same_node(&tree));
    assert!(!outcome.is_changed());
}

/// Test a rewrite pass decides a change by identity: an equal but distinct
/// node is a change, the input itself is not.
#[test]
fn rewrite_pass_did_change_compares_identity() {
    let mut pass = RewritePass::new(build_keeping_rewriter());
    let tree = build_leaf("leaf", 3);
    let equal = build_leaf("leaf", 3);

    let changed_for_equal = pass.did_change(&tree, &equal).expect("comparable");
    let changed_for_itself = pass.did_change(&tree, &tree.clone()).expect("comparable");

    assert!(changed_for_equal);
    assert!(!changed_for_itself);
}

/// Test a failing rewrite fails the rewrite pass's run with the rewrite
/// error, whose source is the rewriter's error.
#[test]
fn rewrite_pass_execute_fails_with_the_rewrite_error() {
    let mut pass = RewritePass::new(ClosureRewriter::new(|_| {
        Err(HookError("refused".to_owned()))
    }));

    let result = pass.execute(&build_leaf("leaf", 1));

    let error: PassError = result.expect_err("the rewrite fails");
    assert!(
        matches!(
            error.kind(),
            PassErrorKind::Hook {
                hook: PassHook::Run,
                ..
            }
        ),
        "{error:?}"
    );
    let source = error.source().expect("the rewrite error is the source");
    let tree_error = source
        .downcast_ref::<RewriteTreeError<ToyTree, HookError>>()
        .expect("the source is the rewrite error");
    assert!(
        matches!(tree_error, RewriteTreeError::Rewrite(HookError(message)) if message == "refused"),
        "got {tree_error:?}"
    );
}

/// Test a failing rewrite pass records a diagnostic naming the rewriter's
/// own error, which the transparent rewrite failure passes on.
#[test]
fn rewrite_pass_diagnostic_names_the_rewriters_error() {
    let mut pass = RewritePass::new(ClosureRewriter::new(|_| {
        Err(HookError("node leaf is refused".to_owned()))
    }));

    let error = pass
        .execute(&build_leaf("leaf", 1))
        .expect_err("the rewrite fails");

    assert_eq!(
        error.diagnostics().last().map(Diagnostic::message_text),
        Some("pass \"ClosureRewriter\" failed in run: node leaf is refused")
    );
}

/// Test a refused rebuild fails the rewrite pass's run with the rebuild
/// failure.
#[test]
fn rewrite_pass_execute_fails_with_a_refused_rebuild() {
    let tree = build_frozen_node("frozen", &[&build_leaf("a", 1)]);
    let mut pass = RewritePass::new(build_leaf_doubler());

    let result = pass.execute(&tree);

    let error = result.expect_err("the rebuild is refused");
    let source = error.source().expect("the rewrite error is the source");
    let tree_error = source
        .downcast_ref::<RewriteTreeError<ToyTree, HookError>>()
        .expect("the source is the rewrite error");
    let (node, _, refusal) = expect_rebuild_failure(tree_error);
    assert!(node.is_same_node(&tree));
    assert_eq!(
        refusal,
        &ToyRebuildError::Frozen {
            name: "frozen".to_owned()
        }
    );
}

/// Test the rewrite pass hands out its rewriter by reference, for mutation,
/// and by value.
#[test]
fn rewrite_pass_exposes_its_rewriter() {
    let mut pass = RewritePass::new(build_keeping_rewriter());
    pass.execute(&build_leaf("leaf", 1))
        .expect("no rewrite fails");
    assert_eq!(pass.rewriter().list_seen_names(), ["leaf"]);

    let seen_before = pass.rewriter_mut().seen().len();
    let rewriter = pass.into_rewriter();

    assert_eq!(seen_before, 1);
    assert_eq!(rewriter.list_seen_names(), ["leaf"]);
}

/// Test a rewrite pass is named after its rewriter's type, borrowing the
/// name.
#[test]
fn rewrite_pass_is_named_after_its_rewriter() {
    let pass = RewritePass::new(NameProbeRewriter);

    let name = CompilerPass::<ToyTree>::name(&pass);

    assert!(matches!(name, Cow::Borrowed("NameProbeRewriter")));
}

// =============================================================================
// Passes in a pipeline
// =============================================================================

/// Test a pipeline rewrites with a rewrite pass and verifies with a walk
/// pass, which walks the input and then the rewritten output.
#[test]
fn pass_manager_verifies_a_rewrite_with_a_walk_pass() {
    let tree = build_node("root", &[&build_leaf("a", 1), &build_leaf("b", 2)]);
    let mut walk_pass = WalkPass::new(RecordingVisitor::new(), TraversalOrder::Post);
    let mut verifier = ValidationManager::new(Identifier::new("tree-verifier"));
    verifier.add(PassValidator::new(&mut walk_pass));
    let mut manager = PassManager::new(Identifier::new("tree-pipeline"));
    manager.add_pass(RewritePass::new(build_leaf_replacer(1, 10)));
    manager.set_verifier(verifier);

    let result = manager.run(&tree).expect("no pass fails");

    assert_eq!(
        result.output(),
        &build_node("root", &[&build_leaf("a", 10), &build_leaf("b", 2)])
    );
    let changes: Vec<bool> = result
        .records()
        .iter()
        .map(|record| match record {
            PipelineRecord::Pass(record) => record.is_changed(),
            other => panic!("unexpected record {other:?}"),
        })
        .collect();
    assert_eq!(changes, [true]);
    drop(manager);
    assert_eq!(
        walk_pass.visitor().list_names(WalkHook::Visit),
        ["a", "b", "root", "a", "b", "root"]
    );
}

/// Test a fixpoint group of a rewrite pass that returns every node itself
/// converges in its first iteration: the pass changes nothing.
#[test]
fn fixpoint_group_of_an_identity_rewrite_pass_converges() {
    let tree = build_small_tree();
    let mut group = FixpointPassGroup::new(Identifier::new("identity-group"))
        .with_max_iterations(NonZeroUsize::new(3).expect("positive"));
    group.add_pass(RewritePass::new(ClosureRewriter::new(|node| {
        Ok(Some(node.clone()))
    })));
    let mut manager = PassManager::new(Identifier::new("tree-pipeline"));
    manager.add_fixpoint_group(group);

    let result = manager.run(&tree).expect("the group converges");

    assert!(result.output().is_same_node(&tree));
    let Some(PipelineRecord::FixpointGroup(record)) = result.records().first() else {
        panic!("expected a group record, got {:?}", result.records());
    };
    assert!(record.is_converged());
    assert_eq!(record.iterations(), 1);
}

/// Test a walk pass reads analyses of the node it walks through its
/// context.
#[test]
fn walk_pass_hooks_read_analyses() {
    let mut pass = WalkPass::new(CountingVisitor::default(), TraversalOrder::Pre);

    pass.execute(&build_small_tree()).expect("no hook fails");

    assert_eq!(pass.visitor().counts, [4, 2, 1, 1]);
}
