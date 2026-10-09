//! Tree-shaped IR and the two traversals over it.
//!
//! A [`NodeHandle`] is a cheap handle to an immutable IR node with a stable
//! [`NodeIdentity`], and a [`Tree`] is a node handle whose node has ordered
//! children of the same type. [`walk_tree`] visits every occurrence of every
//! node, calling a [`TreeVisitor`]'s hooks around each. [`rewrite_tree`]
//! rebuilds a tree bottom-up through a [`Rewriter`], rewriting each distinct
//! node once even when the tree shares it. Both keep their own work stack,
//! so the depth of a tree is bounded by memory, not by the call stack.
//!
//! Both traversals hand every hook a context `&mut C` of the caller's
//! choosing, which they never read themselves.
//!
//! # Examples
//!
//! ```
//! use std::convert::Infallible;
//! use std::sync::Arc;
//!
//! use fhy_core::tree::{NodeHandle, NodeIdentity, TraversalOrder, Tree, TreeVisitor, walk_tree};
//!
//! /// A node holding an integer over ordered children.
//! #[derive(Clone)]
//! struct Node(Arc<(i64, Vec<Node>)>);
//!
//! impl NodeHandle for Node {
//!     fn identity(&self) -> NodeIdentity {
//!         NodeIdentity::of_arc(&self.0)
//!     }
//! }
//!
//! impl Tree for Node {
//!     type RebuildError = Infallible;
//!
//!     fn children(&self) -> impl Iterator<Item = &Self> {
//!         self.0.1.iter()
//!     }
//!
//!     fn rebuild_with_children(&self, children: Vec<Self>) -> Result<Self, Infallible> {
//!         Ok(Node(Arc::new((self.0.0, children))))
//!     }
//! }
//!
//! /// Sums the integers of every occurrence.
//! struct Sum(i64);
//!
//! impl TreeVisitor<Node> for Sum {
//!     type Error = Infallible;
//!
//!     fn visit(&mut self, node: &Node, _cx: &mut ()) -> Result<(), Infallible> {
//!         self.0 += node.0.0;
//!         Ok(())
//!     }
//! }
//!
//! let leaf = Node(Arc::new((2, Vec::new())));
//! let root = Node(Arc::new((1, vec![leaf.clone(), leaf])));
//! let mut sum = Sum(0);
//!
//! walk_tree(&mut sum, &root, TraversalOrder::Pre, &mut ())?;
//!
//! assert_eq!(sum.0, 5);
//! # Ok::<(), Infallible>(())
//! ```

mod hash;
mod node;
mod rewrite;
mod walk;

pub(crate) use hash::BuildIdentityHasher;
pub use node::{NodeHandle, NodeIdentity, Tree};
pub use rewrite::{RewriteTreeError, Rewriter, rewrite_tree};
pub use walk::{TraversalOrder, TreeVisitor, walk_tree};
