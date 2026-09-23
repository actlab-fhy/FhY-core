//! Tree-shaped IR, the walk and rewrite traversals over it, and the adapters
//! that turn traversal hooks into passes.
//!
//! [`walk_tree`] visits every occurrence of every node, calling a
//! [`TreeVisitor`]'s hooks around each. [`rewrite_tree`] rebuilds a tree
//! bottom-up through a [`Rewriter`], rewriting each distinct node once even
//! when the tree shares it. Both keep their own work stack, so the depth of
//! a tree is bounded by memory, not by the call stack.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;

use super::analysis::{NodeHandle, NodeIdentity};
use super::context::PassContext;
use super::pass::{CompilerPass, PassFailure};
use super::registry;

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
        let _ = (node, cx);
        Ok(())
    }

    /// Visit `node`.
    ///
    /// # Errors
    ///
    /// Returns an error to stop the walk. By default, does nothing.
    fn visit(&mut self, node: &N, cx: &mut PassContext<'_>) -> Result<(), Self::Error> {
        let _ = (node, cx);
        Ok(())
    }

    /// Handle `node` after it and its children were visited.
    ///
    /// # Errors
    ///
    /// Returns an error to stop the walk. By default, does nothing.
    fn after_visit(&mut self, node: &N, cx: &mut PassContext<'_>) -> Result<(), Self::Error> {
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

/// The hook [`rewrite_tree`] calls for each distinct node.
pub trait Rewriter<N: Tree> {
    /// The error the hook returns.
    type Error;

    /// Return the replacement for `node`, or `None` to keep it.
    ///
    /// `node` is seen after its children were rewritten: when any child
    /// changed, `node` is the original rebuilt around the rewritten
    /// children. A replacement counts as a change even when it is `node`
    /// itself, so the parent is rebuilt around it.
    ///
    /// # Errors
    ///
    /// Returns an error to stop the rewrite.
    fn rewrite(&mut self, node: &N, cx: &mut PassContext<'_>) -> Result<Option<N>, Self::Error>;
}

/// A node of a rewrite whose children are being rewritten: the node, its
/// children, and the results of the children rewritten so far (`None` for a
/// child that stays as it was).
struct RewriteFrame<'t, N> {
    node: &'t N,
    children: Vec<&'t N>,
    results: Vec<Option<N>>,
}

impl<'t, N: Tree> RewriteFrame<'t, N> {
    /// Start rewriting `node`.
    fn new(node: &'t N) -> Self {
        let children: Vec<&'t N> = node.children().collect();
        Self {
            node,
            results: Vec::with_capacity(children.len()),
            children,
        }
    }

    /// Return the next child to rewrite, or `None` once every child is
    /// rewritten.
    fn next_child(&self) -> Option<&'t N> {
        self.children.get(self.results.len()).copied()
    }

    /// Return the node's children with each rewritten child in place of the
    /// original.
    fn merge_children(&self) -> Vec<N> {
        self.children
            .iter()
            .zip(&self.results)
            .map(|(&original, result)| result.as_ref().unwrap_or(original).clone())
            .collect()
    }
}

/// Finish a node whose children are all rewritten: rebuild it if a child
/// changed, then ask `rewriter` for its replacement. Return the result, or
/// `None` when the node stays as it was.
fn finish_node<N, R>(
    frame: &RewriteFrame<'_, N>,
    rewriter: &mut R,
    cx: &mut PassContext<'_>,
) -> Result<Option<N>, RewriteTreeError<N, R::Error>>
where
    N: Tree,
    R: Rewriter<N> + ?Sized,
{
    let rebuilt = if frame.results.iter().any(Option::is_some) {
        let rebuilt = frame
            .node
            .rebuild_with_children(frame.merge_children())
            .map_err(|source| RewriteTreeError::Rebuild {
                node: frame.node.clone(),
                children: frame.merge_children(),
                source,
            })?;
        Some(rebuilt)
    } else {
        None
    };
    let visited = rebuilt.as_ref().unwrap_or(frame.node);
    let replacement = rewriter
        .rewrite(visited, cx)
        .map_err(RewriteTreeError::Rewrite)?;
    Ok(replacement.or(rebuilt))
}

/// Rewrite the tree under `root` bottom-up with `rewriter`.
///
/// Every node's children are rewritten first, in
/// [`Tree::children`] order. A node with a rewritten child is rebuilt from
/// the rewritten children, keeping the handles of the children no rewrite
/// touched, and [`Rewriter::rewrite`] then sees the rebuilt node (or the
/// node itself when no child changed). A replacement is not rewritten
/// again.
///
/// A node the tree shares is rewritten once and its result reused at every
/// occurrence, so the rewriter sees each distinct node once and a DAG costs
/// time linear in its distinct nodes. When nothing changes, the result is a
/// handle to `root` itself, and every subtree nothing changed in keeps its
/// handle.
///
/// # Errors
///
/// Returns the first error the rewriter returns, or the first error
/// rebuilding a node around rewritten children returns.
pub fn rewrite_tree<N, R>(
    rewriter: &mut R,
    root: &N,
    cx: &mut PassContext<'_>,
) -> Result<N, RewriteTreeError<N, R::Error>>
where
    N: Tree,
    R: Rewriter<N> + ?Sized,
{
    // Every original node stays alive through `root` while the rewrite runs,
    // so its identity keys its result unambiguously.
    let mut results: HashMap<NodeIdentity, Option<N>> = HashMap::new();
    let mut ancestors: Vec<RewriteFrame<'_, N>> = Vec::new();
    let mut current = RewriteFrame::new(root);
    loop {
        if let Some(child) = current.next_child() {
            if let Some(result) = results.get(&child.identity()) {
                current.results.push(result.clone());
            } else {
                ancestors.push(std::mem::replace(&mut current, RewriteFrame::new(child)));
            }
            continue;
        }
        let result = finish_node(&current, rewriter, cx)?;
        let Some(mut parent) = ancestors.pop() else {
            return Ok(result.unwrap_or_else(|| root.clone()));
        };
        results.insert(current.node.identity(), result.clone());
        parent.results.push(result);
        current = parent;
    }
}

/// A failed [`rewrite_tree`].
pub enum RewriteTreeError<N: Tree, E> {
    /// The rewriter returned an error.
    ///
    /// Displays as `rewriting a node failed`; the rewriter's error is the
    /// [`source`](Error::source).
    Rewrite(E),
    /// Rebuilding a node around its rewritten children failed.
    ///
    /// Displays as `rebuilding a node around its rewritten children failed`;
    /// the rebuild error is the [`source`](Error::source).
    Rebuild {
        /// The node that could not be rebuilt.
        node: N,
        /// The children it was given: the rewritten children, with the
        /// children no rewrite touched in their places.
        children: Vec<N>,
        /// The rebuild error.
        source: N::RebuildError,
    },
}

impl<N: Tree, E: fmt::Debug> fmt::Debug for RewriteTreeError<N, E> {
    /// Show the rewriter's or the rebuild error; the nodes are opaque.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rewrite(error) => f.debug_tuple("Rewrite").field(error).finish(),
            Self::Rebuild { source, .. } => f
                .debug_struct("Rebuild")
                .field("source", source)
                .finish_non_exhaustive(),
        }
    }
}

impl<N: Tree, E> fmt::Display for RewriteTreeError<N, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rewrite(_) => f.write_str("rewriting a node failed"),
            Self::Rebuild { .. } => {
                f.write_str("rebuilding a node around its rewritten children failed")
            }
        }
    }
}

impl<N: Tree, E: Error + 'static> Error for RewriteTreeError<N, E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Rewrite(error) => Some(error),
            Self::Rebuild { source, .. } => Some(source),
        }
    }
}

/// A pass that walks its input with a [`TreeVisitor`] and never changes it.
///
/// The pass outputs `()`, so it fits a
/// [`ValidationManager`](super::ValidationManager) or a standalone
/// [`execute`](super::ExecutePass::execute) rather than a pipeline step. Its
/// default name is the name its own type is registered under, else the
/// visitor's default name: the name the visitor's type is registered under,
/// else the visitor's type name without its module path and generic
/// arguments. An unregistered pass's runs therefore count under
/// [`run_count::<V>()`](super::run_count).
///
/// # Examples
///
/// ```
/// use std::convert::Infallible;
///
/// use fhy_core::identifier::Identifier;
/// use fhy_core::pass_infrastructure::{
///     CompilerPass, ExecutePass, PassContext, TraversalOrder, TreeVisitor, WalkPass,
/// };
/// use fhy_core::symbolic::expression::Expression;
///
/// #[derive(Default)]
/// struct NodeCounter(usize);
///
/// impl TreeVisitor<Expression> for NodeCounter {
///     type Error = Infallible;
///
///     fn visit(&mut self, _node: &Expression, _cx: &mut PassContext<'_>) -> Result<(), Infallible> {
///         self.0 += 1;
///         Ok(())
///     }
/// }
///
/// let x = Expression::from(Identifier::new("x"));
/// let mut pass = WalkPass::new(NodeCounter::default(), TraversalOrder::Pre);
///
/// pass.execute(&(-&x + 1))?;
///
/// assert_eq!(pass.visitor().0, 4);
/// assert_eq!(CompilerPass::<Expression, ()>::name(&pass), "NodeCounter");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct WalkPass<V> {
    visitor: V,
    order: TraversalOrder,
}

impl<V> WalkPass<V> {
    /// Create the pass that walks with `visitor` in `order`.
    #[must_use]
    pub fn new(visitor: V, order: TraversalOrder) -> Self {
        Self { visitor, order }
    }

    /// Return the visitor.
    #[must_use]
    pub fn visitor(&self) -> &V {
        &self.visitor
    }

    /// Return the visitor for mutation.
    #[must_use]
    pub fn visitor_mut(&mut self) -> &mut V {
        &mut self.visitor
    }

    /// Return the visitor, consuming the pass.
    #[must_use]
    pub fn into_visitor(self) -> V {
        self.visitor
    }
}

impl<N, V> CompilerPass<N, ()> for WalkPass<V>
where
    N: Tree,
    V: TreeVisitor<N>,
    V::Error: Into<PassFailure>,
{
    fn name(&self) -> String {
        registry::find_registered_pass_name::<Self>()
            .unwrap_or_else(registry::find_default_pass_name::<V>)
    }

    fn noop_output(&mut self, ir: &N, cx: &mut PassContext<'_>) -> Result<(), PassFailure> {
        let _ = (ir, cx);
        Ok(())
    }

    fn run(&mut self, ir: &N, cx: &mut PassContext<'_>) -> Result<(), PassFailure> {
        walk_tree(&mut self.visitor, ir, self.order, cx).map_err(Into::into)
    }

    fn did_change(&mut self, input: &N, output: &()) -> Result<bool, PassFailure> {
        let _ = (input, output);
        Ok(false)
    }
}

/// A pass that rewrites its input with a [`Rewriter`].
///
/// The run changed the IR exactly when its output is a different node from
/// its input, and a skipped run outputs its input. Its default name is the
/// name its own type is registered under, else the rewriter's default name,
/// as for [`WalkPass`].
///
/// # Examples
///
/// ```
/// use std::convert::Infallible;
///
/// use fhy_core::identifier::Identifier;
/// use fhy_core::pass_infrastructure::{ExecutePass, PassContext, RewritePass, Rewriter};
/// use fhy_core::symbolic::expression::{Expression, ExpressionKind};
///
/// /// Replaces `x` by `y`.
/// struct Rename {
///     x: Identifier,
///     y: Expression,
/// }
///
/// impl Rewriter<Expression> for Rename {
///     type Error = Infallible;
///
///     fn rewrite(
///         &mut self,
///         node: &Expression,
///         _cx: &mut PassContext<'_>,
///     ) -> Result<Option<Expression>, Infallible> {
///         Ok(match node.kind() {
///             ExpressionKind::Identifier(identifier) if identifier == &self.x => {
///                 Some(self.y.clone())
///             }
///             _ => None,
///         })
///     }
/// }
///
/// let x = Identifier::new("x");
/// let y = Expression::from(Identifier::new("y"));
/// let mut pass = RewritePass::new(Rename { x: x.clone(), y: y.clone() });
///
/// let outcome = pass.execute(&(Expression::from(x) + 1))?;
///
/// assert!(outcome.is_changed());
/// assert_eq!(outcome.output(), &(y + 1));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct RewritePass<R> {
    rewriter: R,
}

impl<R> RewritePass<R> {
    /// Create the pass that rewrites with `rewriter`.
    #[must_use]
    pub fn new(rewriter: R) -> Self {
        Self { rewriter }
    }

    /// Return the rewriter.
    #[must_use]
    pub fn rewriter(&self) -> &R {
        &self.rewriter
    }

    /// Return the rewriter for mutation.
    #[must_use]
    pub fn rewriter_mut(&mut self) -> &mut R {
        &mut self.rewriter
    }

    /// Return the rewriter, consuming the pass.
    #[must_use]
    pub fn into_rewriter(self) -> R {
        self.rewriter
    }
}

impl<N, R> CompilerPass<N, N> for RewritePass<R>
where
    N: Tree,
    R: Rewriter<N>,
    R::Error: Error + Send + Sync + 'static,
{
    fn name(&self) -> String {
        registry::find_registered_pass_name::<Self>()
            .unwrap_or_else(registry::find_default_pass_name::<R>)
    }

    fn noop_output(&mut self, ir: &N, cx: &mut PassContext<'_>) -> Result<N, PassFailure> {
        let _ = cx;
        Ok(ir.clone())
    }

    fn run(&mut self, ir: &N, cx: &mut PassContext<'_>) -> Result<N, PassFailure> {
        rewrite_tree(&mut self.rewriter, ir, cx).map_err(|error| Box::new(error) as PassFailure)
    }

    fn did_change(&mut self, input: &N, output: &N) -> Result<bool, PassFailure> {
        Ok(input.identity() != output.identity())
    }
}
