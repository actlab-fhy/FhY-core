//! The walk over a tree and the hooks it calls.

use super::node::Tree;

/// Where [`walk_tree`] visits a node relative to its children.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TraversalOrder {
    /// Visit the node before its children.
    #[default]
    Pre,
    /// Visit the node after its children.
    Post,
}

/// The hooks [`walk_tree`] calls for each node it walks, each given the
/// walk's context `C`.
///
/// For each node the walk calls [`before_visit`](Self::before_visit); then
/// [`visit`](Self::visit) and the walk of the children, visit first under
/// [`TraversalOrder::Pre`] and children first under
/// [`TraversalOrder::Post`]; then [`after_visit`](Self::after_visit). The
/// walk stops at the first hook that returns an error.
///
/// A visitor that needs no context implements `TreeVisitor<N>` and is
/// walked with `&mut ()`. One that works under any context implements
/// `TreeVisitor<N, C>` for every `C: ?Sized`, so it serves a direct walk and
/// the walk pass of `fhy_core::pass` alike.
pub trait TreeVisitor<N: Tree, C: ?Sized = ()> {
    /// The error the hooks return.
    type Error;

    /// Handle `node` before it and its children are visited.
    ///
    /// # Errors
    ///
    /// Returns an error to stop the walk. By default, does nothing.
    fn before_visit(&mut self, node: &N, cx: &mut C) -> Result<(), Self::Error> {
        let _ = (node, cx);
        Ok(())
    }

    /// Visit `node`.
    ///
    /// # Errors
    ///
    /// Returns an error to stop the walk. By default, does nothing.
    fn visit(&mut self, node: &N, cx: &mut C) -> Result<(), Self::Error> {
        let _ = (node, cx);
        Ok(())
    }

    /// Handle `node` after it and its children were visited.
    ///
    /// # Errors
    ///
    /// Returns an error to stop the walk. By default, does nothing.
    fn after_visit(&mut self, node: &N, cx: &mut C) -> Result<(), Self::Error> {
        let _ = (node, cx);
        Ok(())
    }

    /// Return whether the walk descends into `node`'s children. By default,
    /// it always does.
    ///
    /// The walk asks once per occurrence of `node`, when it would descend. A
    /// node whose children are skipped still gets all three of its own
    /// hooks.
    fn walks_children(&mut self, node: &N) -> bool {
        let _ = node;
        true
    }
}

/// One step of a walk: start walking a node, or finish it once its children
/// are walked.
enum WalkStep<'t, N> {
    Enter(&'t N),
    Leave(&'t N),
}

/// Walk the tree under `root` with `visitor`, visiting every occurrence of
/// every node in `order` and handing `cx` to every hook.
///
/// The walk itself never reads `cx`.
///
/// # Errors
///
/// Returns the first error a hook returns; no hook runs after it.
pub fn walk_tree<N, C, V>(
    visitor: &mut V,
    root: &N,
    order: TraversalOrder,
    cx: &mut C,
) -> Result<(), V::Error>
where
    N: Tree,
    C: ?Sized,
    V: TreeVisitor<N, C> + ?Sized,
{
    let mut steps = vec![WalkStep::Enter(root)];
    while let Some(step) = steps.pop() {
        match step {
            WalkStep::Enter(node) => {
                visitor.before_visit(node, cx)?;
                if order == TraversalOrder::Pre {
                    visitor.visit(node, cx)?;
                }
                steps.push(WalkStep::Leave(node));
                if visitor.walks_children(node) {
                    let first_child = steps.len();
                    steps.extend(node.children().map(WalkStep::Enter));
                    steps[first_child..].reverse();
                }
            }
            WalkStep::Leave(node) => {
                if order == TraversalOrder::Post {
                    visitor.visit(node, cx)?;
                }
                visitor.after_visit(node, cx)?;
            }
        }
    }
    Ok(())
}
