//! The adapters that turn tree traversal hooks into passes.
//!
//! [`WalkPass`] runs a [`TreeVisitor`] and [`RewritePass`] runs a
//! [`Rewriter`], each with the pass's [`PassContext`] as the traversal
//! context, so their hooks report diagnostics and read analyses.

use std::borrow::Cow;
use std::error::Error;

use super::compiler_pass::{CompilerPass, PassFailure, short_type_name};
use super::context::PassContext;
use crate::tree::{Rewriter, TraversalOrder, Tree, TreeVisitor, rewrite_tree, walk_tree};

/// A pass that walks its input with a [`TreeVisitor`] and never changes it.
///
/// The visitor is walked with the run's [`PassContext`] as its context, so
/// it implements `TreeVisitor<N, PassContext<'_>>`, or `TreeVisitor<N, C>`
/// for every `C: ?Sized`.
///
/// The pass outputs `()`, so it fits a
/// [`ValidationManager`](super::ValidationManager) or a standalone
/// [`execute`](super::ExecutePass::execute) rather than a pipeline step. It
/// is named [`short_type_name::<V>()`](short_type_name), after its visitor.
///
/// # Examples
///
/// ```
/// use std::convert::Infallible;
///
/// use fhy_core::expr::Expression;
/// use fhy_core::identifier::Identifier;
/// use fhy_core::pass::{CompilerPass, ExecutePass, WalkPass};
/// use fhy_core::tree::{TraversalOrder, TreeVisitor};
///
/// #[derive(Default)]
/// struct NodeCounter(usize);
///
/// impl<C: ?Sized> TreeVisitor<Expression, C> for NodeCounter {
///     type Error = Infallible;
///
///     fn visit(&mut self, _node: &Expression, _cx: &mut C) -> Result<(), Infallible> {
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
    V: for<'a> TreeVisitor<N, PassContext<'a>>,
    for<'a> <V as TreeVisitor<N, PassContext<'a>>>::Error: Into<PassFailure>,
{
    fn name(&self) -> Cow<'static, str> {
        short_type_name::<V>()
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
/// The rewriter is called with the run's [`PassContext`] as its context, so
/// it implements `Rewriter<N, PassContext<'_>>`, or `Rewriter<N, C>` for
/// every `C: ?Sized`.
///
/// The run changed the IR exactly when its output is a different node from
/// its input, and a skipped run outputs its input. It is named
/// [`short_type_name::<R>()`](short_type_name), after its rewriter.
///
/// # Examples
///
/// ```
/// use std::convert::Infallible;
///
/// use fhy_core::expr::{Expression, ExpressionKind};
/// use fhy_core::identifier::Identifier;
/// use fhy_core::pass::{ExecutePass, RewritePass};
/// use fhy_core::tree::Rewriter;
///
/// /// Replaces `x` by `y`.
/// struct Rename {
///     x: Identifier,
///     y: Expression,
/// }
///
/// impl<C: ?Sized> Rewriter<Expression, C> for Rename {
///     type Error = Infallible;
///
///     fn rewrite(&mut self, node: &Expression, _cx: &mut C) -> Result<Option<Expression>, Infallible> {
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
    R: for<'a> Rewriter<N, PassContext<'a>>,
    for<'a> <R as Rewriter<N, PassContext<'a>>>::Error: Error + Send + Sync + 'static,
{
    fn name(&self) -> Cow<'static, str> {
        short_type_name::<R>()
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
