//! Tree-shaped IR, the walk and rewrite traversals over it, and the adapters
//! that turn traversal hooks into passes.
//!
//! [`walk_tree`] visits every occurrence of every node, calling a
//! [`TreeVisitor`]'s hooks around each. [`rewrite_tree`] rebuilds a tree
//! bottom-up through a [`Rewriter`], rewriting each distinct node once even
//! when the tree shares it. Both keep their own work stack, so the depth of
//! a tree is bounded by memory, not by the call stack.

#![expect(
    dead_code,
    unused_variables,
    clippy::needless_pass_by_value,
    clippy::todo,
    reason = "interface stub; the tree traversals are not implemented yet"
)]

use std::error::Error;
use std::fmt;

use super::analysis::NodeHandle;
use super::context::PassContext;
use super::pass::{CompilerPass, PassFailure};

/// An IR node handle whose node has an ordered list of children of the same
/// type.
pub trait Tree: NodeHandle {
    /// The error [`rebuild_with_children`](Self::rebuild_with_children)
    /// returns.
    type RebuildError: Error + Send + Sync + 'static;

    /// Return the node's children, in visiting order.
    fn children(&self) -> impl Iterator<Item = &Self>;

    /// Return a node like this one with `children` in place of its children,
    /// in the order [`children`](Self::children) lists them.
    ///
    /// # Errors
    ///
    /// Returns an error if `children` does not fit the node, for example
    /// because it has the wrong length.
    fn rebuild_with_children(&self, children: Vec<Self>) -> Result<Self, Self::RebuildError>;
}

/// Where [`walk_tree`] visits a node relative to its children.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TraversalOrder {
    /// Visit the node before its children.
    #[default]
    Pre,
    /// Visit the node after its children.
    Post,
}

/// The hooks [`walk_tree`] calls for each node it walks.
///
/// For each node the walk calls [`before_visit`](Self::before_visit); then
/// [`visit`](Self::visit) and the walk of the children, visit first under
/// [`TraversalOrder::Pre`] and children first under
/// [`TraversalOrder::Post`]; then [`after_visit`](Self::after_visit). The
/// walk stops at the first hook that returns an error.
pub trait TreeVisitor<N: Tree> {
    /// The error the hooks return.
    type Error;

    /// Handle `node` before it and its children are visited.
    ///
    /// # Errors
    ///
    /// Returns an error to stop the walk. By default, does nothing.
    fn before_visit(&mut self, node: &N, cx: &mut PassContext<'_>) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Visit `node`.
    ///
    /// # Errors
    ///
    /// Returns an error to stop the walk. By default, does nothing.
    fn visit(&mut self, node: &N, cx: &mut PassContext<'_>) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Handle `node` after it and its children were visited.
    ///
    /// # Errors
    ///
    /// Returns an error to stop the walk. By default, does nothing.
    fn after_visit(&mut self, node: &N, cx: &mut PassContext<'_>) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Return whether the walk descends into `node`'s children. By default,
    /// it always does.
    fn walks_children(&mut self, node: &N) -> bool {
        true
    }
}

/// Walk the tree under `root` with `visitor`, visiting every occurrence of
/// every node in `order`.
///
/// # Errors
///
/// Returns the first error a hook returns; no hook runs after it.
pub fn walk_tree<N, V>(
    visitor: &mut V,
    root: &N,
    order: TraversalOrder,
    cx: &mut PassContext<'_>,
) -> Result<(), V::Error>
where
    N: Tree,
    V: TreeVisitor<N> + ?Sized,
{
    todo!()
}

/// The hook [`rewrite_tree`] calls for each distinct node.
pub trait Rewriter<N: Tree> {
    /// The error the hook returns.
    type Error;

    /// Return the replacement for `node`, or `None` to keep it.
    ///
    /// `node` is seen after its children were rewritten: when any child
    /// changed, `node` is the original rebuilt around the rewritten
    /// children.
    ///
    /// # Errors
    ///
    /// Returns an error to stop the rewrite.
    fn rewrite(&mut self, node: &N, cx: &mut PassContext<'_>) -> Result<Option<N>, Self::Error>;
}

/// Rewrite the tree under `root` bottom-up with `rewriter`.
///
/// A node the tree shares is rewritten once and its result reused at every
/// occurrence. When nothing changes, the result is a handle to `root`
/// itself.
///
/// # Errors
///
/// Returns the first error the rewriter returns, or the first error
/// rebuilding a node around rewritten children returns.
pub fn rewrite_tree<N, R>(
    rewriter: &mut R,
    root: &N,
    cx: &mut PassContext<'_>,
) -> Result<N, RewriteTreeError<R::Error, N::RebuildError>>
where
    N: Tree,
    R: Rewriter<N> + ?Sized,
{
    todo!()
}

/// A failed [`rewrite_tree`].
#[derive(Debug)]
pub enum RewriteTreeError<E, B> {
    /// The rewriter returned an error.
    Rewrite(E),
    /// Rebuilding a node around rewritten children failed.
    Rebuild(B),
}

impl<E: fmt::Display, B: fmt::Display> fmt::Display for RewriteTreeError<E, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!()
    }
}

impl<E, B> Error for RewriteTreeError<E, B>
where
    E: Error + 'static,
    B: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        todo!()
    }
}

/// A pass that walks its input with a [`TreeVisitor`] and never changes it.
#[derive(Debug)]
pub struct WalkPass<V> {
    visitor: V,
    order: TraversalOrder,
}

impl<V> WalkPass<V> {
    /// Create the pass that walks with `visitor` in `order`.
    #[must_use]
    pub fn new(visitor: V, order: TraversalOrder) -> Self {
        todo!()
    }

    /// Return the visitor.
    #[must_use]
    pub fn visitor(&self) -> &V {
        todo!()
    }

    /// Return the visitor for mutation.
    #[must_use]
    pub fn visitor_mut(&mut self) -> &mut V {
        todo!()
    }

    /// Return the visitor, consuming the pass.
    #[must_use]
    pub fn into_visitor(self) -> V {
        todo!()
    }
}

impl<N, V> CompilerPass<N, ()> for WalkPass<V>
where
    N: Tree,
    V: TreeVisitor<N>,
    V::Error: Into<PassFailure>,
{
    fn noop_output(&mut self, ir: &N, cx: &mut PassContext<'_>) -> Result<(), PassFailure> {
        todo!()
    }

    fn run(&mut self, ir: &N, cx: &mut PassContext<'_>) -> Result<(), PassFailure> {
        todo!()
    }

    fn did_change(&mut self, input: &N, output: &()) -> Result<bool, PassFailure> {
        todo!()
    }
}

/// A pass that rewrites its input with a [`Rewriter`].
///
/// The run changed the IR exactly when its output is a different node from
/// its input, and a skipped run outputs its input.
#[derive(Debug)]
pub struct RewritePass<R> {
    rewriter: R,
}

impl<R> RewritePass<R> {
    /// Create the pass that rewrites with `rewriter`.
    #[must_use]
    pub fn new(rewriter: R) -> Self {
        todo!()
    }

    /// Return the rewriter.
    #[must_use]
    pub fn rewriter(&self) -> &R {
        todo!()
    }

    /// Return the rewriter for mutation.
    #[must_use]
    pub fn rewriter_mut(&mut self) -> &mut R {
        todo!()
    }

    /// Return the rewriter, consuming the pass.
    #[must_use]
    pub fn into_rewriter(self) -> R {
        todo!()
    }
}

impl<N, R> CompilerPass<N, N> for RewritePass<R>
where
    N: Tree,
    R: Rewriter<N>,
    R::Error: Error + Send + Sync + 'static,
{
    fn noop_output(&mut self, ir: &N, cx: &mut PassContext<'_>) -> Result<N, PassFailure> {
        todo!()
    }

    fn run(&mut self, ir: &N, cx: &mut PassContext<'_>) -> Result<N, PassFailure> {
        todo!()
    }

    fn did_change(&mut self, input: &N, output: &N) -> Result<bool, PassFailure> {
        todo!()
    }
}
