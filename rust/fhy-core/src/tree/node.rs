//! Node identity, node handles, and tree-shaped handles.

use std::error::Error;
use std::sync::Arc;

/// The opaque identity of a live IR node.
///
/// Two identities are equal only when they come from handles to the same
/// node while that node is alive. Once every handle to a node is dropped, a
/// later node may receive the same identity; a holder that needs an identity
/// to stay unique keeps a handle alive alongside it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NodeIdentity(usize);

impl NodeIdentity {
    /// Return the identity of the allocation `node` points to.
    ///
    /// Clones of one [`Arc`] share an identity; two separately allocated
    /// `Arc`s have different identities while both are alive.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use fhy_core::tree::NodeIdentity;
    ///
    /// let node = Arc::new(5);
    /// let alias = Arc::clone(&node);
    /// let other = Arc::new(5);
    ///
    /// assert_eq!(NodeIdentity::of_arc(&node), NodeIdentity::of_arc(&alias));
    /// assert_ne!(NodeIdentity::of_arc(&node), NodeIdentity::of_arc(&other));
    /// ```
    #[must_use]
    pub fn of_arc<T: ?Sized>(node: &Arc<T>) -> Self {
        Self(Arc::as_ptr(node).cast::<()>().addr())
    }
}

/// A cheap-to-clone handle to an immutable IR node with a stable identity.
///
/// Pipeline IR and every IR an analysis is cached for is a node handle.
/// Cloning a handle must not copy the node: a clone reports the same
/// [`identity`](Self::identity), and the node cannot change while any
/// handle to it is alive, so a result computed for one handle holds for
/// every clone.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
///
/// use fhy_core::tree::{NodeHandle, NodeIdentity};
///
/// #[derive(Clone)]
/// struct Module(Arc<Vec<String>>);
///
/// impl NodeHandle for Module {
///     fn identity(&self) -> NodeIdentity {
///         NodeIdentity::of_arc(&self.0)
///     }
/// }
///
/// let module = Module(Arc::new(vec!["main".to_owned()]));
/// assert_eq!(module.identity(), module.clone().identity());
/// ```
pub trait NodeHandle: Clone + Send + Sync + 'static {
    /// Return the identity of the node this handle points to.
    #[must_use]
    fn identity(&self) -> NodeIdentity;
}

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
    /// A handle type that shares structurally equal nodes may return a
    /// handle to an existing node, this node included.
    ///
    /// # Errors
    ///
    /// Returns an error if `children` does not fit the node, for example
    /// because it has the wrong length.
    fn rebuild_with_children(&self, children: Vec<Self>) -> Result<Self, Self::RebuildError>;

    /// Return whether handles to this node other than this one may exist.
    /// By default, `true`.
    ///
    /// [`rewrite_tree`](super::rewrite_tree) remembers the result of a
    /// shared node for its later occurrences and skips that bookkeeping for
    /// a node that is not shared. A handle type that can tell a node has a
    /// single handle, such as by its reference count, may answer `false`
    /// for it. A node that answers `false` although it occurs more than
    /// once is rewritten at each occurrence.
    fn is_shared(&self) -> bool {
        true
    }
}
