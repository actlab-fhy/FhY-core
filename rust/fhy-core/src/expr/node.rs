//! The expression handle, its node kinds, and structural equality.
//!
//! An [`Expression`] is a cheap, clonable handle to an immutable node; a
//! clone shares the node, so one subtree may appear in several places of
//! one tree or of several trees. [`Expression::kind`] exposes the node as an
//! [`ExpressionKind`] to match on. Equality and hashing are structural:
//! two expressions are equal when they have the same shape, the same
//! operations and call function names, identifiers with the same ids, and
//! equal literals (see [`LiteralValue`]). [`Expression::ptr_eq`] tells
//! whether two handles share one node.
//!
//! Dropping, equality, hashing, alpha-equivalence, substitution and free
//! identifier collection keep their pending nodes in a work list on the heap
//! rather than on the call stack, so they handle a tree of any depth. All but
//! dropping handle a subtree that occurs in several places once, so they
//! take time linear in the distinct nodes of a DAG, not in its occurrences.
//! `Debug` is iterative too, and bounded.

use std::collections::{HashMap, HashSet};
use std::convert::Infallible;
use std::hash::{BuildHasher, DefaultHasher, Hash, Hasher};
use std::sync::Arc;

use crate::identifier::Identifier;
use crate::tree::{
    BuildIdentityHasher, NodeHandle, NodeIdentity, RewriteTreeError, Rewriter, TraversalOrder,
    Tree, TreeVisitor, rewrite_tree, walk_tree,
};

use super::alpha::AlphaRenaming;
use super::error::{FunctionNameError, PiecewiseError, RebuildError};
use super::literal::LiteralValue;
use super::operation::{BinaryOperation, LogicalOperation, UnaryOperation};

/// The children of a node, in visiting order, from either end.
enum Children<'a> {
    /// The children of a node that stores them contiguously.
    Contiguous(std::slice::Iter<'a, Expression>),
    /// The children of a piecewise: each case's condition then value, then
    /// the otherwise branch.
    Piecewise {
        cases: std::slice::Iter<'a, (Expression, Expression)>,
        front_value: Option<&'a Expression>,
        back_condition: Option<&'a Expression>,
        otherwise: Option<&'a Expression>,
    },
}

impl<'a> Iterator for Children<'a> {
    type Item = &'a Expression;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Contiguous(children) => children.next(),
            Self::Piecewise {
                cases,
                front_value,
                back_condition,
                otherwise,
            } => {
                if let Some(value) = front_value.take() {
                    return Some(value);
                }
                if let Some((condition, value)) = cases.next() {
                    *front_value = Some(value);
                    return Some(condition);
                }
                back_condition.take().or_else(|| otherwise.take())
            }
        }
    }
}

impl DoubleEndedIterator for Children<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        match self {
            Self::Contiguous(children) => children.next_back(),
            Self::Piecewise {
                cases,
                front_value,
                back_condition,
                otherwise,
            } => {
                if let Some(otherwise) = otherwise.take() {
                    return Some(otherwise);
                }
                if let Some(condition) = back_condition.take() {
                    return Some(condition);
                }
                if let Some((condition, value)) = cases.next_back() {
                    *back_condition = Some(condition);
                    return Some(value);
                }
                front_value.take()
            }
        }
    }
}

/// Return the error for rebuilding a node of `expected` children from
/// `actual` children.
fn build_child_count_mismatch(expected: usize, actual: usize) -> RebuildError {
    RebuildError::ChildCount { expected, actual }
}

/// Return the placeholder a node's kind is replaced with while its
/// children are moved out to be dropped.
fn build_drop_placeholder() -> ExpressionKind {
    ExpressionKind::Literal(LiteralValue::Bool(false))
}

/// Move the children out of `kind` onto `pending`.
fn move_children(kind: ExpressionKind, pending: &mut Vec<Expression>) {
    match kind {
        ExpressionKind::Unary(node) => pending.push(node.operand),
        ExpressionKind::Binary(node) => pending.extend(node.operands),
        ExpressionKind::Logical(node) => pending.extend(node.operands),
        ExpressionKind::Piecewise(node) => {
            for (condition, value) in node.cases {
                pending.push(condition);
                pending.push(value);
            }
            pending.push(node.otherwise);
        }
        ExpressionKind::Call(node) => pending.extend(node.arguments),
        ExpressionKind::Identifier(_) | ExpressionKind::Literal(_) => {}
    }
}

/// Compare the data of two nodes, excluding their children, and return
/// whether it matches; `identifiers_match` compares two identifier leaves.
fn is_node_data_equal(
    left: &Expression,
    right: &Expression,
    identifiers_match: &impl Fn(&Identifier, &Identifier) -> bool,
) -> bool {
    match (left.kind(), right.kind()) {
        (ExpressionKind::Unary(left), ExpressionKind::Unary(right)) => {
            left.operation == right.operation
        }
        (ExpressionKind::Binary(left), ExpressionKind::Binary(right)) => {
            left.operation == right.operation
        }
        (ExpressionKind::Logical(left), ExpressionKind::Logical(right)) => {
            left.operation == right.operation && left.operands.len() == right.operands.len()
        }
        (ExpressionKind::Identifier(left), ExpressionKind::Identifier(right)) => {
            identifiers_match(left, right)
        }
        (ExpressionKind::Literal(left), ExpressionKind::Literal(right)) => left == right,
        (ExpressionKind::Piecewise(left), ExpressionKind::Piecewise(right)) => {
            left.cases.len() == right.cases.len()
        }
        (ExpressionKind::Call(left), ExpressionKind::Call(right)) => {
            left.function_name == right.function_name
                && left.arguments.len() == right.arguments.len()
        }
        _ => false,
    }
}

/// Compare two trees node by node; identical handles are skipped when
/// `skip_shared_nodes` is set, and `identifiers_match` compares two
/// identifier leaves.
///
/// A pair of nodes both of which may be shared is compared once: the walk
/// is depth-first, so by the time a pair comes up again every pair below
/// its first occurrence has been compared, and a difference would have
/// ended the walk. Two DAGs therefore compare in time linear in their
/// distinct pairs of nodes.
fn is_tree_equal(
    left: &Expression,
    right: &Expression,
    skip_shared_nodes: bool,
    identifiers_match: &impl Fn(&Identifier, &Identifier) -> bool,
) -> bool {
    let mut compared: HashSet<(NodeIdentity, NodeIdentity), BuildIdentityHasher> =
        HashSet::default();
    let mut pending = vec![(left, right)];
    while let Some((left, right)) = pending.pop() {
        if skip_shared_nodes && Expression::ptr_eq(left, right) {
            continue;
        }
        if left.is_shared()
            && right.is_shared()
            && !compared.insert((left.identity(), right.identity()))
        {
            continue;
        }
        if !is_node_data_equal(left, right, identifiers_match) {
            return false;
        }
        pending.extend(left.children().zip(right.children()));
    }
    true
}

/// Feed the data of `expression`'s node, excluding its children, to
/// `hasher`: a tag for its kind, then its operation (and operand count for a
/// logical node), identifier, literal, case count, or function name and
/// argument count.
fn hash_node_data(expression: &Expression, hasher: &mut impl Hasher) {
    match expression.kind() {
        ExpressionKind::Unary(node) => {
            hasher.write_u8(0);
            node.operation.hash(hasher);
        }
        ExpressionKind::Binary(node) => {
            hasher.write_u8(1);
            node.operation.hash(hasher);
        }
        ExpressionKind::Identifier(identifier) => {
            hasher.write_u8(2);
            identifier.hash(hasher);
        }
        ExpressionKind::Literal(value) => {
            hasher.write_u8(3);
            value.hash(hasher);
        }
        ExpressionKind::Piecewise(node) => {
            hasher.write_u8(4);
            hasher.write_usize(node.cases.len());
        }
        ExpressionKind::Call(node) => {
            hasher.write_u8(5);
            node.function_name.hash(hasher);
            hasher.write_usize(node.arguments.len());
        }
        ExpressionKind::Logical(node) => {
            hasher.write_u8(6);
            node.operation.hash(hasher);
            hasher.write_usize(node.operands.len());
        }
    }
}

/// A node whose structural digest is being computed: the node, its
/// children still to digest, and where its children's digests start on the
/// walk's digest stack.
struct DigestFrame<'a> {
    node: &'a Expression,
    children: Children<'a>,
    first_digest: usize,
}

impl<'a> DigestFrame<'a> {
    /// Start digesting `node`, whose children's digests will start at
    /// `first_digest`.
    fn new(node: &'a Expression, first_digest: usize) -> Self {
        Self {
            node,
            children: node.iterate_children(),
            first_digest,
        }
    }
}

/// Return the structural digest of `root`: the hash, under the fixed-key
/// [`DefaultHasher`], of the node's data (see [`hash_node_data`]) followed
/// by the digests of its children in order.
///
/// The digest depends only on the structure, so equal expressions have
/// equal digests however their subtrees are shared. It is computed
/// bottom-up, and the digest of a node that may be shared is remembered by
/// identity, so each distinct shared node is digested once.
fn compute_structural_digest(root: &Expression) -> u64 {
    let mut shared_digests: HashMap<NodeIdentity, u64, BuildIdentityHasher> = HashMap::default();
    let mut digests: Vec<u64> = Vec::new();
    let mut ancestors: Vec<DigestFrame<'_>> = Vec::new();
    let mut current = DigestFrame::new(root, 0);
    loop {
        if let Some(child) = current.children.next() {
            let known = child
                .is_shared()
                .then(|| shared_digests.get(&child.identity()))
                .flatten();
            if let Some(&digest) = known {
                digests.push(digest);
            } else {
                let frame = DigestFrame::new(child, digests.len());
                ancestors.push(std::mem::replace(&mut current, frame));
            }
            continue;
        }
        let mut hasher = DefaultHasher::new();
        hash_node_data(current.node, &mut hasher);
        for child_digest in digests.drain(current.first_digest..) {
            hasher.write_u64(child_digest);
        }
        let digest = hasher.finish();
        let Some(parent) = ancestors.pop() else {
            return digest;
        };
        if current.node.is_shared() {
            shared_digests.insert(current.node.identity(), digest);
        }
        digests.push(digest);
        current = parent;
    }
}

/// The visitor behind [`Expression::free_identifiers`]: records every
/// identifier reference, walking below a node that may be shared only at
/// its first occurrence.
#[derive(Default)]
struct FreeIdentifierCollector {
    free: HashSet<Identifier>,
    visited_shared_nodes: HashSet<NodeIdentity, BuildIdentityHasher>,
    /// Whether the node just visited is seen for the first time.
    is_first_visit: bool,
}

impl TreeVisitor<Expression> for FreeIdentifierCollector {
    type Error = Infallible;

    fn visit(&mut self, node: &Expression, _cx: &mut ()) -> Result<(), Infallible> {
        self.is_first_visit =
            !node.is_shared() || self.visited_shared_nodes.insert(node.identity());
        if self.is_first_visit {
            if let ExpressionKind::Identifier(identifier) = node.kind() {
                self.free.insert(identifier.clone());
            }
        }
        Ok(())
    }

    fn walks_children(&mut self, _node: &Expression) -> bool {
        self.is_first_visit
    }
}

/// The rewriter behind [`Expression::substitute`]: replaces each reference
/// to a mapped identifier with a handle to its replacement.
struct Substitution<'m, S> {
    replacements: &'m HashMap<Identifier, Expression, S>,
}

impl<S: BuildHasher> Rewriter<Expression> for Substitution<'_, S> {
    type Error = Infallible;

    fn rewrite(
        &mut self,
        node: &Expression,
        _cx: &mut (),
    ) -> Result<Option<Expression>, Infallible> {
        let ExpressionKind::Identifier(identifier) = node.kind() else {
            return Ok(None);
        };
        Ok(self.replacements.get(identifier).cloned())
    }
}

/// A symbolic expression: a shared handle to an immutable node.
///
/// Cloning bumps a reference count and shares the node. `==` and `Hash`
/// compare trees structurally; [`ptr_eq`](Self::ptr_eq) compares handles.
/// `Display` writes the text [`display`](Self::display) writes under the
/// default options, a DAG's shared subtrees at every occurrence. `Debug`
/// writes a bounded text for diagnostics: the functional notation with
/// identifier ids, eliding everything after 1,000 nodes, so a failing
/// `assert_eq!` on any expression prints in bounded time.
///
/// Dropping, comparing, hashing, substituting into, collecting the free
/// identifiers of, displaying, and screening a tree keep their pending
/// nodes on the heap, so they handle a tree of any depth on any thread. All
/// but dropping and displaying handle a subtree occurring in several places
/// once, so a DAG such as `x(k+1) = xk + xk` costs time linear in its
/// distinct nodes. Serialization and deserialization recurse once per tree
/// level: a
/// serialization round trip of a tree 4000 levels deep through `serde_json`
/// values needs up to 32 MiB of stack in an unoptimized build and up to
/// 8 MiB in an optimized one.
///
/// Decoding JSON text is also capped by `serde_json`'s nesting limit: its
/// text deserializer refuses input nested more than 127 JSON levels deep
/// with a `recursion limit exceeded` error. A tree level takes two JSON
/// levels, or three below a piecewise case or a call argument, which sit in
/// lists, so `serde_json::from_str` decodes a chain of at most 62 unary or
/// binary nodes over a leaf. A `serde_json::Value` parsed from text meets the
/// same limit. A caller needing deeper trees can enable `serde_json`'s
/// `unbounded_depth` feature and decode through a `serde_json::Deserializer`
/// after calling its `disable_recursion_limit`, on a thread with a stack
/// large enough for the recursion above.
///
/// # Examples
///
/// ```
/// use fhy_core::identifier::Identifier;
/// use fhy_core::expr::{Expression, ExpressionKind};
///
/// let x = Expression::from(Identifier::new("x"));
/// let sum = &x + 1;
/// let same_sum = &x + 1;
/// assert_eq!(sum, same_sum);
/// assert!(!Expression::ptr_eq(&sum, &same_sum));
/// assert!(matches!(sum.kind(), ExpressionKind::Binary(_)));
/// ```
///
/// `/` builds true division, whatever the operand types: `x / 4` over
/// integers is the exact real quotient, and
/// [`floor_divide`](Self::floor_divide) rounds down. There is no `%`
/// operator; the remainder of floor division is
/// [`floor_mod`](Self::floor_mod):
///
/// ```compile_fail,E0369
/// use fhy_core::identifier::Identifier;
/// use fhy_core::expr::Expression;
///
/// let remainder = Expression::from(Identifier::new("x")) % 3;
/// ```
///
/// Expressions have no order: `<` does not compare them, and a comparison
/// node is built with [`less`](Self::less) and its siblings.
///
/// ```compile_fail,E0369
/// use fhy_core::identifier::Identifier;
/// use fhy_core::expr::Expression;
///
/// let x = Expression::from(Identifier::new("x"));
/// let y = Expression::from(Identifier::new("y"));
/// let ordered = x < y;
/// ```
#[derive(Clone)]
pub struct Expression(Arc<ExpressionKind>);

/// The node an [`Expression`] refers to, one variant per node kind.
#[derive(Debug, Clone)]
pub enum ExpressionKind {
    /// A unary operation applied to one operand.
    Unary(UnaryExpression),
    /// A binary operation applied to two operands.
    Binary(BinaryExpression),
    /// A conjunction or disjunction of two or more operands.
    Logical(LogicalExpression),
    /// A reference to an identifier.
    Identifier(Identifier),
    /// A constant.
    Literal(LiteralValue),
    /// A first-match-wins choice among cases, with a fallback.
    Piecewise(PiecewiseExpression),
    /// A named function applied to arguments.
    Call(CallExpression),
}

/// A unary operation applied to one operand.
#[derive(Debug, Clone)]
pub struct UnaryExpression {
    operation: UnaryOperation,
    operand: Expression,
}

/// A binary operation applied to a left and a right operand.
#[derive(Debug, Clone)]
pub struct BinaryExpression {
    operation: BinaryOperation,
    operands: [Expression; 2],
}

/// A conjunction or disjunction of two or more operands, in order.
///
/// A logical node has at least two operands. It is built by
/// [`Expression::all`], [`Expression::any`], [`Expression::new_logical`],
/// [`Expression::and`] and [`Expression::or`], none of which splices the
/// operands of a nested logical node into it.
#[derive(Debug, Clone)]
pub struct LogicalExpression {
    operation: LogicalOperation,
    operands: Box<[Expression]>,
}

/// A first-match-wins choice: the value of the first case whose condition
/// holds, or the otherwise branch when none does.
///
/// A piecewise has at least one case, and no case condition is a literal
/// other than a Boolean.
#[derive(Debug, Clone)]
pub struct PiecewiseExpression {
    cases: Box<[(Expression, Expression)]>,
    otherwise: Expression,
}

/// A named function applied to argument expressions.
///
/// The function name is not empty. Neither the name nor the argument count is
/// checked against any function catalogue.
#[derive(Debug, Clone)]
pub struct CallExpression {
    function_name: Arc<str>,
    arguments: Box<[Expression]>,
}

impl Expression {
    /// Return the node this expression refers to.
    #[must_use]
    pub fn kind(&self) -> &ExpressionKind {
        &self.0
    }

    /// Return whether `this` and `other` are handles to the same node.
    ///
    /// Structurally equal trees built separately are not the same node.
    #[must_use]
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        Arc::ptr_eq(&this.0, &other.0)
    }

    /// Return the direct children in visiting order.
    ///
    /// A unary node yields its operand; a binary node its left then its
    /// right operand; a logical node its operands in order; a piecewise node
    /// each case's condition then value, in case order, then its otherwise
    /// branch (`c0, v0, c1, v1, ..., otherwise`); a call its arguments in
    /// order. An identifier or a literal has no children.
    #[must_use]
    pub fn children(&self) -> impl DoubleEndedIterator<Item = &Expression> {
        self.iterate_children()
    }

    /// Build a node of the same kind and operation from new children, given
    /// in [`children`](Self::children) order.
    ///
    /// An identifier or a literal takes no children and returns a handle to
    /// itself. A piecewise rebuilt from `2n + 1` children has `n` cases; a
    /// call keeps its function name. A logical node takes exactly its own
    /// operand count, so it keeps at least two operands, and keeps the
    /// children as given, never flattening a nested logical node into
    /// itself.
    ///
    /// # Errors
    ///
    /// Returns [`RebuildError::ChildCount`] if the number of children
    /// differs from the node's own child count, and
    /// [`RebuildError::Piecewise`] holding
    /// [`PiecewiseError::NonBooleanConditionLiteral`] if a new piecewise
    /// case condition is a literal other than a Boolean.
    pub fn rebuild_with_children(
        &self,
        children: Vec<Expression>,
    ) -> Result<Expression, RebuildError> {
        let expected = self.count_children();
        let actual = children.len();
        if actual != expected {
            return Err(build_child_count_mismatch(expected, actual));
        }
        let rebuilt = match self.kind() {
            ExpressionKind::Identifier(_) | ExpressionKind::Literal(_) => self.clone(),
            ExpressionKind::Unary(node) => {
                let [operand]: [Expression; 1] =
                    children.try_into().map_err(|rejected: Vec<Expression>| {
                        build_child_count_mismatch(1, rejected.len())
                    })?;
                Self::from(UnaryExpression::new(node.operation, operand))
            }
            ExpressionKind::Binary(node) => {
                let [left, right]: [Expression; 2] =
                    children.try_into().map_err(|rejected: Vec<Expression>| {
                        build_child_count_mismatch(2, rejected.len())
                    })?;
                Self::from(BinaryExpression::new(node.operation, left, right))
            }
            ExpressionKind::Logical(node) => Self::from_kind(ExpressionKind::Logical(
                LogicalExpression::new(node.operation, children.into_boxed_slice()),
            )),
            ExpressionKind::Piecewise(_) => {
                let mut children = children.into_iter();
                let otherwise = children
                    .next_back()
                    .ok_or_else(|| build_child_count_mismatch(expected, actual))?;
                let mut cases = Vec::with_capacity(expected / 2);
                while let (Some(condition), Some(value)) = (children.next(), children.next()) {
                    cases.push((condition, value));
                }
                Self::from(
                    PiecewiseExpression::try_new(cases, otherwise)
                        .map_err(RebuildError::Piecewise)?,
                )
            }
            ExpressionKind::Call(node) => Self::from(CallExpression {
                function_name: Arc::clone(&node.function_name),
                arguments: children.into_boxed_slice(),
            }),
        };
        Ok(rebuilt)
    }

    /// Return the identifiers the expression refers to.
    ///
    /// Expressions bind no identifiers, so every identifier referenced is
    /// free. A subtree occurring in several places is walked once.
    #[must_use]
    pub fn free_identifiers(&self) -> HashSet<Identifier> {
        let mut collector = FreeIdentifierCollector::default();
        let Ok(()) = walk_tree(&mut collector, self, TraversalOrder::Pre, &mut ());
        collector.free
    }

    /// Replace every reference to a mapped identifier with its replacement,
    /// simultaneously.
    ///
    /// Replacements are not substituted into in turn, and each occurrence of
    /// a mapped identifier becomes a handle to the same replacement node
    /// ([`ptr_eq`](Self::ptr_eq) with it). Only the nodes above a replaced
    /// reference are rebuilt: every subtree without one is returned as a
    /// handle to itself, so substituting a map that replaces nothing, or
    /// maps identifiers only to handles of their own references, returns a
    /// handle to this expression. A subtree occurring in several places is
    /// substituted into once and its result reused at every occurrence, so
    /// the result shares its subtrees as this expression does, and the work
    /// is linear in the distinct nodes.
    ///
    /// # Errors
    ///
    /// Returns [`PiecewiseError::NonBooleanConditionLiteral`] if a
    /// replacement puts a literal other than a Boolean in a piecewise case
    /// condition.
    pub fn substitute<S: BuildHasher>(
        &self,
        replacements: &HashMap<Identifier, Expression, S>,
    ) -> Result<Expression, PiecewiseError> {
        let mut substitution = Substitution { replacements };
        rewrite_tree(&mut substitution, self, &mut ()).map_err(|error| match error {
            RewriteTreeError::Rewrite(never) => match never {},
            RewriteTreeError::Rebuild {
                source: RebuildError::Piecewise(error),
                ..
            } => error,
            RewriteTreeError::Rebuild {
                source: RebuildError::ChildCount { .. },
                ..
            } => unreachable!("the tree walk rebuilds a node from exactly its own children"),
        })
    }

    /// Return whether `other` is this expression with its free identifiers
    /// renamed by `renaming`.
    ///
    /// The trees must have the same structure. Where this expression refers
    /// to an identifier `renaming` maps, `other` must refer to its image;
    /// where it refers to an unmapped identifier, `other` must refer to the
    /// same identifier, and that identifier must not be an image of
    /// `renaming`. Under the empty renaming this is structural equality.
    ///
    /// A pair of subtrees met again at another place of the two trees is
    /// compared once, so two DAGs compare in time linear in their distinct
    /// nodes.
    #[must_use]
    pub fn is_alpha_equivalent_under(&self, other: &Expression, renaming: &AlphaRenaming) -> bool {
        is_tree_equal(self, other, renaming.is_empty(), &|left, right| {
            renaming.are_identifiers_alpha_equivalent(left, right)
        })
    }

    /// Return the direct children in visiting order, as
    /// [`children`](Self::children) does, in an iterator type a walk can
    /// keep in its frames.
    fn iterate_children(&self) -> Children<'_> {
        match self.kind() {
            ExpressionKind::Unary(node) => {
                Children::Contiguous(std::slice::from_ref(&node.operand).iter())
            }
            ExpressionKind::Binary(node) => Children::Contiguous(node.operands.iter()),
            ExpressionKind::Logical(node) => Children::Contiguous(node.operands.iter()),
            ExpressionKind::Piecewise(node) => Children::Piecewise {
                cases: node.cases.iter(),
                front_value: None,
                back_condition: None,
                otherwise: Some(&node.otherwise),
            },
            ExpressionKind::Call(node) => Children::Contiguous(node.arguments.iter()),
            ExpressionKind::Identifier(_) | ExpressionKind::Literal(_) => {
                Children::Contiguous(std::slice::Iter::default())
            }
        }
    }

    /// Return the number of children, the length of
    /// [`children`](Self::children).
    fn count_children(&self) -> usize {
        match self.kind() {
            ExpressionKind::Identifier(_) | ExpressionKind::Literal(_) => 0,
            ExpressionKind::Unary(_) => 1,
            ExpressionKind::Binary(_) => 2,
            ExpressionKind::Logical(node) => node.operands.len(),
            ExpressionKind::Piecewise(node) => 2 * node.cases.len() + 1,
            ExpressionKind::Call(node) => node.arguments.len(),
        }
    }

    /// Wrap `kind` in a new handle.
    pub(super) fn from_kind(kind: ExpressionKind) -> Self {
        Self(Arc::new(kind))
    }
}

impl From<Identifier> for Expression {
    /// Wrap an identifier in an identifier reference.
    fn from(identifier: Identifier) -> Self {
        Self(Arc::new(ExpressionKind::Identifier(identifier)))
    }
}

impl From<LiteralValue> for Expression {
    /// Wrap a constant in a literal expression.
    fn from(value: LiteralValue) -> Self {
        Self(Arc::new(ExpressionKind::Literal(value)))
    }
}

impl From<UnaryExpression> for Expression {
    fn from(node: UnaryExpression) -> Self {
        Self(Arc::new(ExpressionKind::Unary(node)))
    }
}

impl From<BinaryExpression> for Expression {
    fn from(node: BinaryExpression) -> Self {
        Self(Arc::new(ExpressionKind::Binary(node)))
    }
}

impl From<PiecewiseExpression> for Expression {
    fn from(node: PiecewiseExpression) -> Self {
        Self(Arc::new(ExpressionKind::Piecewise(node)))
    }
}

impl From<CallExpression> for Expression {
    fn from(node: CallExpression) -> Self {
        Self(Arc::new(ExpressionKind::Call(node)))
    }
}

impl PartialEq for Expression {
    /// Compare structurally: same node kinds, operations and call function
    /// names, identifiers with the same ids, equal literals, children equal
    /// in order.
    fn eq(&self, other: &Self) -> bool {
        is_tree_equal(self, other, true, &|left, right| left == right)
    }
}

impl Eq for Expression {}

impl Hash for Expression {
    /// Feed the structural digest of the tree to `state`.
    ///
    /// The digest is computed bottom-up from each node's data and its
    /// children's digests under a fixed-key hasher, so it depends on the
    /// structure alone and agrees with `==`: equal trees digest alike
    /// whatever they share, and a subtree occurring in several places is
    /// digested once.
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(compute_structural_digest(self));
    }
}

/// The identity of the node the handle shares: equal for clones, distinct
/// for separately built nodes, even structurally equal ones.
impl NodeHandle for Expression {
    fn identity(&self) -> NodeIdentity {
        NodeIdentity::of_arc(&self.0)
    }
}

/// The tree view of an expression: [`Expression::children`] and
/// [`Expression::rebuild_with_children`]; a node is shared while it has more
/// than one handle.
impl Tree for Expression {
    type RebuildError = RebuildError;

    fn children(&self) -> impl Iterator<Item = &Self> {
        Expression::children(self)
    }

    fn rebuild_with_children(&self, children: Vec<Self>) -> Result<Self, RebuildError> {
        Expression::rebuild_with_children(self, children)
    }

    fn is_shared(&self) -> bool {
        Arc::strong_count(&self.0) > 1
    }
}

impl Drop for Expression {
    /// Drop the node if this is its last handle, moving the children of
    /// every node dropped with it onto a work list, so a deep tree drops
    /// without deep recursion.
    fn drop(&mut self) {
        let Some(kind) = Arc::get_mut(&mut self.0) else {
            return;
        };
        let mut pending = Vec::new();
        move_children(
            std::mem::replace(kind, build_drop_placeholder()),
            &mut pending,
        );
        while let Some(mut expression) = pending.pop() {
            if let Some(kind) = Arc::get_mut(&mut expression.0) {
                move_children(
                    std::mem::replace(kind, build_drop_placeholder()),
                    &mut pending,
                );
            }
        }
    }
}

impl UnaryExpression {
    /// Construct a unary node.
    pub(super) fn new(operation: UnaryOperation, operand: Expression) -> Self {
        Self { operation, operand }
    }

    /// Return the operation.
    #[must_use]
    pub fn operation(&self) -> UnaryOperation {
        self.operation
    }

    /// Return the operand.
    #[must_use]
    pub fn operand(&self) -> &Expression {
        &self.operand
    }
}

impl BinaryExpression {
    /// Construct a binary node.
    pub(super) fn new(operation: BinaryOperation, left: Expression, right: Expression) -> Self {
        Self {
            operation,
            operands: [left, right],
        }
    }

    /// Return the operation.
    #[must_use]
    pub fn operation(&self) -> BinaryOperation {
        self.operation
    }

    /// Return the left operand.
    #[must_use]
    pub fn left(&self) -> &Expression {
        &self.operands[0]
    }

    /// Return the right operand.
    #[must_use]
    pub fn right(&self) -> &Expression {
        &self.operands[1]
    }
}

impl LogicalExpression {
    /// Construct the logical node of `operation` over `operands`, which
    /// number at least two: the builders, rebuilding and the wire decoder
    /// ensure the count before calling.
    pub(super) fn new(operation: LogicalOperation, operands: Box<[Expression]>) -> Self {
        Self {
            operation,
            operands,
        }
    }

    /// Return the operation.
    #[must_use]
    pub fn operation(&self) -> LogicalOperation {
        self.operation
    }

    /// Return the operands in order; there are always at least two.
    #[must_use]
    pub fn operands(&self) -> &[Expression] {
        &self.operands
    }
}

/// Check a piecewise with `case_count` cases has at least one.
///
/// The constructor and the wire decoder share this check, so a payload is
/// refused for the same reason before any of its identifiers is restored.
///
/// # Errors
///
/// Returns [`PiecewiseError::NoCases`] if `case_count` is zero.
pub(super) fn validate_case_count(case_count: usize) -> Result<(), PiecewiseError> {
    if case_count == 0 {
        return Err(PiecewiseError::NoCases);
    }
    Ok(())
}

/// Check the literal condition of the piecewise case at `case_index` is a
/// Boolean.
///
/// The constructor and the wire decoder share this check, so a payload is
/// refused for the same reason before any of its identifiers is restored.
///
/// # Errors
///
/// Returns [`PiecewiseError::NonBooleanConditionLiteral`] naming
/// `case_index` if `condition` is not a Boolean.
pub(super) fn validate_condition_literal(
    case_index: usize,
    condition: &LiteralValue,
) -> Result<(), PiecewiseError> {
    if !matches!(condition, LiteralValue::Bool(_)) {
        return Err(PiecewiseError::NonBooleanConditionLiteral { case_index });
    }
    Ok(())
}

/// Check a call's function name is not empty.
///
/// The constructor and the wire decoder share this check, so a payload is
/// refused for the same reason before any of its identifiers is restored.
///
/// # Errors
///
/// Returns [`FunctionNameError::Empty`] if `function_name` is empty.
pub(super) fn validate_function_name(function_name: &str) -> Result<(), FunctionNameError> {
    if function_name.is_empty() {
        return Err(FunctionNameError::Empty);
    }
    Ok(())
}

impl PiecewiseExpression {
    /// Construct a piecewise from its `(condition, value)` cases, in
    /// evaluation order, and its otherwise branch.
    ///
    /// # Errors
    ///
    /// Returns [`PiecewiseError::NoCases`] if `cases` is empty, and
    /// [`PiecewiseError::NonBooleanConditionLiteral`] naming the
    /// first case whose condition is a literal other than a Boolean.
    pub fn try_new(
        cases: Vec<(Expression, Expression)>,
        otherwise: Expression,
    ) -> Result<Self, PiecewiseError> {
        validate_case_count(cases.len())?;
        for (case_index, (condition, _)) in cases.iter().enumerate() {
            if let ExpressionKind::Literal(literal) = condition.kind() {
                validate_condition_literal(case_index, literal)?;
            }
        }
        Ok(Self {
            cases: cases.into_boxed_slice(),
            otherwise,
        })
    }

    /// Return the `(condition, value)` cases in evaluation order.
    #[must_use]
    pub fn cases(&self) -> &[(Expression, Expression)] {
        &self.cases
    }

    /// Return the otherwise branch.
    #[must_use]
    pub fn otherwise(&self) -> &Expression {
        &self.otherwise
    }
}

impl CallExpression {
    /// Construct a call of `function_name` with `arguments` in order.
    ///
    /// # Errors
    ///
    /// Returns [`FunctionNameError::Empty`] if `function_name` is empty.
    pub fn try_new(
        function_name: &str,
        arguments: Vec<Expression>,
    ) -> Result<Self, FunctionNameError> {
        validate_function_name(function_name)?;
        Ok(Self {
            function_name: Arc::from(function_name),
            arguments: arguments.into_boxed_slice(),
        })
    }

    /// Return the function name.
    #[must_use]
    pub fn function_name(&self) -> &str {
        &self.function_name
    }

    /// Return the arguments in order.
    #[must_use]
    pub fn arguments(&self) -> &[Expression] {
        &self.arguments
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Expression>();
    assert_send_sync::<ExpressionKind>();
};
