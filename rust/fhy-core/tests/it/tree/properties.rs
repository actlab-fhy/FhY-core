//! Property tests for the tree traversals of `fhy_core::tree` over toy DAGs,
//! which share nodes at random: a rewrite that keeps every node, or returns
//! every node itself, returns the root, the output is the root exactly when
//! nothing changed, the memoized rewrite of a DAG equals the rewrite of its
//! unshared copy for a pure rewriter, and a walk brackets every occurrence
//! with balanced hooks in pre- or post-order.

use crate::support::tree_ir;

use std::collections::HashSet;

use fhy_core::tree::{NodeHandle, TraversalOrder, rewrite_tree, walk_tree};
use proptest::prelude::*;
use proptest::sample::{Index, select};
use tree_ir::{
    ClosureRewriter, RecordingVisitor, ToyTree, WalkHook, build_keeping_rewriter, build_leaf,
    build_node,
};

/// The most nodes a generated DAG has.
const MAX_NODES: usize = 10;

/// The most children a generated node has.
const MAX_CHILDREN: usize = 3;

/// Return a strategy for DAGs of up to [`MAX_NODES`] distinct nodes, named
/// `n0`, `n1`, ..., each of whose children is any earlier node, so a node
/// may occur several times.
fn arbitrary_dag() -> impl Strategy<Value = ToyTree> {
    prop::collection::vec(
        (
            0_i64..6,
            prop::collection::vec(any::<Index>(), 0..=MAX_CHILDREN),
        ),
        1..=MAX_NODES,
    )
    .prop_map(|specifications| {
        let mut nodes: Vec<ToyTree> = Vec::new();
        for (position, (value, child_picks)) in specifications.into_iter().enumerate() {
            let name = format!("n{position}");
            let node = if position == 0 || child_picks.is_empty() {
                build_leaf(&name, value)
            } else {
                let children: Vec<&ToyTree> = child_picks
                    .iter()
                    .map(|pick| &nodes[pick.index(position)])
                    .collect();
                build_node(&name, &children).with_value(value)
            };
            nodes.push(node);
        }
        nodes.pop().expect("at least one node")
    })
}

/// Return a pure rewriter: an odd leaf holds three times its integer, and an
/// inner node holding zero whose children are all leaves folds into a leaf
/// named `folded` holding their sum.
fn build_pure_rewriter() -> ClosureRewriter {
    ClosureRewriter::new(|node| {
        let children = node.child_nodes();
        Ok(if children.is_empty() {
            (node.value() % 2 == 1).then(|| node.with_value(node.value() * 3))
        } else if node.value() == 0 && children.iter().all(|child| child.child_nodes().is_empty()) {
            Some(build_leaf(
                "folded",
                children.iter().map(ToyTree::value).sum(),
            ))
        } else {
            None
        })
    })
}

/// Return the distinct leaves of `tree`.
fn collect_distinct_leaves(tree: &ToyTree) -> Vec<ToyTree> {
    let mut seen = HashSet::new();
    let mut leaves = Vec::new();
    let mut pending = vec![tree];
    while let Some(node) = pending.pop() {
        if !seen.insert(node.identity()) {
            continue;
        }
        if node.child_nodes().is_empty() {
            leaves.push(node.clone());
        }
        pending.extend(node.child_nodes());
    }
    leaves
}

/// Return the names of `tree`'s nodes in post-order, one per occurrence.
fn list_names_in_post_order(tree: &ToyTree) -> Vec<String> {
    let mut names = Vec::new();
    let mut steps = vec![(tree, false)];
    while let Some((node, is_exit)) = steps.pop() {
        if is_exit {
            names.push(node.name().to_owned());
        } else {
            steps.push((node, true));
            steps.extend(node.child_nodes().iter().rev().map(|child| (child, false)));
        }
    }
    names
}

proptest! {
    /// Test a rewrite that keeps every node returns the root itself and
    /// sees each distinct node once.
    #[test]
    fn rewrite_tree_keeping_every_node_returns_the_root(dag in arbitrary_dag()) {
        let mut rewriter = build_keeping_rewriter();

        let output = rewrite_tree(&mut rewriter, &dag, &mut ()).expect("no rewrite fails");

        prop_assert!(output.is_same_node(&dag));
        prop_assert_eq!(rewriter.seen().len(), dag.count_distinct_nodes());
    }

    /// Test a rewrite that returns every node itself returns the root
    /// itself and sees each distinct node once, as itself.
    #[test]
    fn rewrite_tree_with_an_identity_rewriter_returns_the_root(dag in arbitrary_dag()) {
        let mut rewriter = ClosureRewriter::new(|node| Ok(Some(node.clone())));

        let output = rewrite_tree(&mut rewriter, &dag, &mut ()).expect("no rewrite fails");

        prop_assert!(output.is_same_node(&dag));
        prop_assert_eq!(rewriter.seen().len(), dag.count_distinct_nodes());
        let distinct: HashSet<_> = rewriter.seen().iter().map(NodeHandle::identity).collect();
        prop_assert_eq!(distinct.len(), dag.count_distinct_nodes());
    }

    /// Test doubling a random subset of the leaves, where doubling a zero
    /// returns the leaf itself, returns the root itself exactly when no
    /// doubled leaf holds a non-zero value.
    #[test]
    fn rewrite_tree_output_is_the_root_iff_nothing_changed(
        dag in arbitrary_dag(),
        doubled in prop::collection::hash_set(0..MAX_NODES, 0..=MAX_NODES),
    ) {
        let doubled_names: HashSet<String> =
            doubled.iter().map(|position| format!("n{position}")).collect();
        let expected_unchanged = collect_distinct_leaves(&dag)
            .iter()
            .all(|leaf| !doubled_names.contains(leaf.name()) || leaf.value() == 0);
        let mut rewriter = ClosureRewriter::new(move |node| {
            Ok((node.child_nodes().is_empty() && doubled_names.contains(node.name())).then(|| {
                if node.value() == 0 {
                    node.clone()
                } else {
                    node.with_value(node.value() * 2)
                }
            }))
        });

        let output = rewrite_tree(&mut rewriter, &dag, &mut ()).expect("no rewrite fails");

        prop_assert_eq!(output.is_same_node(&dag), expected_unchanged);
    }

    /// Test the memoized rewrite of a DAG by a pure rewriter equals the
    /// rewrite of its unshared copy, seeing each distinct node once where the
    /// copy's rewrite sees each occurrence.
    #[test]
    fn rewrite_tree_of_a_dag_equals_the_rewrite_of_its_unshared_copy(
        dag in arbitrary_dag()
    ) {
        let copy = dag.copy_unshared();
        let mut dag_rewriter = build_pure_rewriter();
        let mut copy_rewriter = build_pure_rewriter();

        let dag_output =
            rewrite_tree(&mut dag_rewriter, &dag, &mut ()).expect("no rewrite fails");
        let copy_output =
            rewrite_tree(&mut copy_rewriter, &copy, &mut ()).expect("no rewrite fails");

        prop_assert_eq!(&dag_output, &copy_output);
        prop_assert_eq!(dag_rewriter.seen().len(), dag.count_distinct_nodes());
        prop_assert_eq!(copy_rewriter.seen().len(), dag.count_occurrences());
    }

    /// Test a walk calls every hook once per occurrence, brackets each
    /// node's visit and its descendants between its before and after hooks,
    /// and visits in the requested order.
    #[test]
    fn walk_tree_brackets_every_occurrence_in_order(
        dag in arbitrary_dag(),
        order in select(vec![TraversalOrder::Pre, TraversalOrder::Post]),
    ) {
        let mut visitor = RecordingVisitor::new();

        walk_tree(&mut visitor, &dag, order, &mut ()).expect("no hook fails");

        let mut open: Vec<&str> = Vec::new();
        for event in visitor.events() {
            let (hook, name) = event.split_once(':').expect("a hook and a name");
            match hook {
                "before" => open.push(name),
                "visit" => prop_assert_eq!(open.last(), Some(&name)),
                _ => prop_assert_eq!(open.pop(), Some(name)),
            }
        }
        prop_assert!(open.is_empty());
        prop_assert_eq!(visitor.events().len(), 3 * dag.count_occurrences());
        prop_assert_eq!(visitor.list_names(WalkHook::Before), dag.list_names_in_pre_order());
        let expected_visits = match order {
            TraversalOrder::Pre => dag.list_names_in_pre_order(),
            TraversalOrder::Post => list_names_in_post_order(&dag),
        };
        prop_assert_eq!(visitor.list_names(WalkHook::Visit), expected_visits);
    }
}
