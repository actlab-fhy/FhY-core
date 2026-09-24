//! The bottom-up rewrite of a tree and the hook it calls.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;

use super::hash::BuildIdentityHasher;
use super::node::{NodeIdentity, Tree};

/// The hook [`rewrite_tree`] calls for each distinct node, given the
/// rewrite's context `C`.
///
/// A rewriter that needs no context implements `Rewriter<N>` and is called
/// with `&mut ()`. One that works under any context implements
/// `Rewriter<N, C>` for every `C: ?Sized`, so it serves a direct rewrite and
/// the rewrite pass of `fhy_core::pass` alike.
pub trait Rewriter<N: Tree, C: ?Sized = ()> {
    /// The error the hook returns.
    type Error;

    /// Return the replacement for `node`, or `None` to keep it.
    ///
    /// `node` is seen after its children were rewritten: when any child
    /// changed, `node` is the original rebuilt around the rewritten
    /// children. A replacement that is the original node itself, the one
    /// the tree held before its children were rewritten, counts as no
    /// change, so returning it reverts the node and its parent is not
    /// rebuilt on its account.
    ///
    /// # Errors
    ///
    /// Returns an error to stop the rewrite.
    fn rewrite(&mut self, node: &N, cx: &mut C) -> Result<Option<N>, Self::Error>;
}

/// A node of a rewrite whose children are being rewritten, and where its
/// children and their results start on the rewrite's shared stacks.
struct RewriteFrame<'t, N> {
    node: &'t N,
    first_child: usize,
    first_result: usize,
}

/// Finish `node`, whose `children` are rewritten to `results` (`None` for
/// an unchanged child): rebuild it if a child changed, then ask `rewriter`
/// for its replacement. Return `None` when the result is `node` itself.
fn finish_node<N, C, R>(
    node: &N,
    children: &[&N],
    results: &[Option<N>],
    rewriter: &mut R,
    cx: &mut C,
) -> Result<Option<N>, RewriteTreeError<N, R::Error>>
where
    N: Tree,
    C: ?Sized,
    R: Rewriter<N, C> + ?Sized,
{
    let merge_children = || -> Vec<N> {
        children
            .iter()
            .zip(results)
            .map(|(&original, result)| result.as_ref().unwrap_or(original).clone())
            .collect()
    };
    let rebuilt = if results.iter().any(Option::is_some) {
        let rebuilt = node
            .rebuild_with_children(merge_children())
            .map_err(|source| RewriteTreeError::Rebuild {
                node: node.clone(),
                children: merge_children(),
                source,
            })?;
        Some(rebuilt)
    } else {
        None
    };
    let visited = rebuilt.as_ref().unwrap_or(node);
    let replacement = rewriter
        .rewrite(visited, cx)
        .map_err(RewriteTreeError::Rewrite)?;
    Ok(replacement
        .or(rebuilt)
        .filter(|result| result.identity() != node.identity()))
}

/// Rewrite the tree under `root` bottom-up with `rewriter`, handing `cx` to
/// every call of [`Rewriter::rewrite`].
///
/// Every node's children are rewritten first, in [`Tree::children`] order.
/// A node with a changed child is rebuilt around the rewritten children,
/// and [`Rewriter::rewrite`] then sees the rebuilt node, or the node itself
/// when no child changed. A replacement is not rewritten again. A result
/// that is the original node itself, from the rewriter or the rebuild,
/// counts as unchanged.
///
/// A node the tree shares is rewritten once and its result reused at every
/// occurrence, provided [`Tree::is_shared`] answers `true` for it, so a DAG
/// costs time linear in its distinct nodes. Every subtree in which nothing
/// changed keeps its handle, so the result is `root` itself exactly when
/// nothing changed. The rewrite itself never reads `cx`.
///
/// # Errors
///
/// Returns [`RewriteTreeError::Rewrite`] with the first error the rewriter
/// returns, or [`RewriteTreeError::Rebuild`] for the first node that refuses
/// to be rebuilt around its rewritten children.
///
/// # Examples
///
/// ```
/// use std::convert::Infallible;
/// use std::sync::Arc;
///
/// use fhy_core::tree::{NodeHandle, NodeIdentity, Rewriter, Tree, rewrite_tree};
///
/// /// A node holding an integer over ordered children.
/// #[derive(Clone, Debug)]
/// struct Node(Arc<(i64, Vec<Node>)>);
///
/// impl NodeHandle for Node {
///     fn identity(&self) -> NodeIdentity {
///         NodeIdentity::of_arc(&self.0)
///     }
/// }
///
/// impl Tree for Node {
///     type RebuildError = Infallible;
///
///     fn children(&self) -> impl Iterator<Item = &Self> {
///         self.0.1.iter()
///     }
///
///     fn rebuild_with_children(&self, children: Vec<Self>) -> Result<Self, Infallible> {
///         Ok(Node(Arc::new((self.0.0, children))))
///     }
/// }
///
/// /// Negates every leaf.
/// struct NegateLeaves;
///
/// impl Rewriter<Node> for NegateLeaves {
///     type Error = Infallible;
///
///     fn rewrite(&mut self, node: &Node, _cx: &mut ()) -> Result<Option<Node>, Infallible> {
///         Ok(node.0.1.is_empty().then(|| Node(Arc::new((-node.0.0, Vec::new())))))
///     }
/// }
///
/// let root = Node(Arc::new((1, vec![Node(Arc::new((2, Vec::new())))])));
///
/// let output = rewrite_tree(&mut NegateLeaves, &root, &mut ())?;
///
/// assert_eq!(output.0.1[0].0.0, -2);
/// assert_ne!(output.identity(), root.identity());
/// # Ok::<(), fhy_core::tree::RewriteTreeError<Node, Infallible>>(())
/// ```
pub fn rewrite_tree<N, C, R>(
    rewriter: &mut R,
    root: &N,
    cx: &mut C,
) -> Result<N, RewriteTreeError<N, R::Error>>
where
    N: Tree,
    C: ?Sized,
    R: Rewriter<N, C> + ?Sized,
{
    // The results of the shared nodes rewritten so far. Every original node
    // stays alive through `root` while the rewrite runs, so its identity
    // keys its result unambiguously.
    let mut shared_results: HashMap<NodeIdentity, Option<N>, BuildIdentityHasher> =
        HashMap::default();
    // The children of every node being rewritten, and the results of those
    // rewritten so far, each node's in one contiguous run on top of its
    // ancestors', so the walk allocates no list per node.
    let mut child_stack: Vec<&N> = root.children().collect();
    let mut result_stack: Vec<Option<N>> = Vec::new();
    let mut ancestors: Vec<RewriteFrame<'_, N>> = Vec::new();
    let mut current = RewriteFrame {
        node: root,
        first_child: 0,
        first_result: 0,
    };
    loop {
        let next_child = current.first_child + (result_stack.len() - current.first_result);
        if let Some(&child) = child_stack.get(next_child) {
            let known = child
                .is_shared()
                .then(|| shared_results.get(&child.identity()))
                .flatten();
            if let Some(result) = known {
                result_stack.push(result.clone());
            } else {
                let frame = RewriteFrame {
                    node: child,
                    first_child: child_stack.len(),
                    first_result: result_stack.len(),
                };
                child_stack.extend(child.children());
                ancestors.push(std::mem::replace(&mut current, frame));
            }
            continue;
        }
        let result = finish_node(
            current.node,
            &child_stack[current.first_child..],
            &result_stack[current.first_result..],
            rewriter,
            cx,
        )?;
        child_stack.truncate(current.first_child);
        result_stack.truncate(current.first_result);
        let Some(parent) = ancestors.pop() else {
            return Ok(result.unwrap_or_else(|| root.clone()));
        };
        if current.node.is_shared() {
            shared_results.insert(current.node.identity(), result.clone());
        }
        result_stack.push(result);
        current = parent;
    }
}

/// A failed [`rewrite_tree`].
///
/// More variants may be added, so a `match` on one needs a wildcard arm.
#[non_exhaustive]
pub enum RewriteTreeError<N: Tree, E> {
    /// The rewriter returned an error.
    ///
    /// Transparent: displays as the rewriter's error, and its
    /// [`source`](Error::source) is that error's source.
    Rewrite(E),
    /// Rebuilding a node around its rewritten children failed.
    ///
    /// Displays as `rebuilding a node around its rewritten children failed`;
    /// the rebuild error is the [`source`](Error::source).
    #[non_exhaustive]
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

impl<N: Tree, E: fmt::Display> fmt::Display for RewriteTreeError<N, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rewrite(error) => fmt::Display::fmt(error, f),
            Self::Rebuild { .. } => {
                f.write_str("rebuilding a node around its rewritten children failed")
            }
        }
    }
}

impl<N: Tree, E: Error + 'static> Error for RewriteTreeError<N, E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Rewrite(error) => error.source(),
            Self::Rebuild { source, .. } => Some(source),
        }
    }
}
