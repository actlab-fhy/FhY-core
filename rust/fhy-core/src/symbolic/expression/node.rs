//! The expression handle, its node kinds, and structural equality.
//!
//! An [`Expression`] is a cheap, clonable handle to an immutable node; a
//! clone shares the node, so one subtree may appear in several places of
//! one tree or of several trees. [`Expression::kind`] exposes the node as an
//! [`ExpressionKind`] to match on. Equality and hashing are structural:
//! two expressions are equal when they have the same shape, the same
//! operations, identifiers with the same ids, and equal literals (see
//! [`LiteralValue`]). [`Expression::ptr_eq`] tells whether two handles share
//! one node.
//!
//! Every walk here keeps its pending nodes in a work list on the heap rather
//! than on the call stack, so it handles a tree of any depth.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::Arc;

use crate::identifier::Identifier;

use super::error::ExpressionBuildError;
use super::literal::{LiteralKind, LiteralValue};
use super::operation::{BinaryOperation, UnaryOperation};

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

/// One step of a bottom-up rebuild: visit a node's children, or rebuild
/// the node from their results.
enum RebuildStep<'a> {
    Enter(&'a Expression),
    Exit(&'a Expression),
}

/// Return the error for rebuilding a node of `expected` children from
/// `actual` children.
fn build_child_count_mismatch(expected: usize, actual: usize) -> ExpressionBuildError {
    ExpressionBuildError::ChildCountMismatch { expected, actual }
}

/// Return the placeholder a node's kind is replaced with while its
/// children are moved out to be dropped.
fn build_drop_placeholder() -> ExpressionKind {
    ExpressionKind::Literal(LiteralValue::from_bool(false))
}

/// Move the children out of `kind` onto `pending`.
fn move_children(kind: ExpressionKind, pending: &mut Vec<Expression>) {
    match kind {
        ExpressionKind::Unary(node) => pending.push(node.operand),
        ExpressionKind::Binary(node) => pending.extend(node.operands),
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

/// Compare the identifiers of two leaves under a free-identifier renaming
/// whose images are `images`.
fn is_identifier_renamed<S: BuildHasher>(
    left: &Identifier,
    right: &Identifier,
    renaming: &HashMap<Identifier, Identifier, S>,
    images: &HashSet<&Identifier>,
) -> bool {
    match renaming.get(left) {
        Some(image) => image == right,
        None => !images.contains(right) && left == right,
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
fn is_tree_equal(
    left: &Expression,
    right: &Expression,
    skip_shared_nodes: bool,
    identifiers_match: &impl Fn(&Identifier, &Identifier) -> bool,
) -> bool {
    let mut pending = vec![(left, right)];
    while let Some((left, right)) = pending.pop() {
        if skip_shared_nodes && Expression::ptr_eq(left, right) {
            continue;
        }
        if !is_node_data_equal(left, right, identifiers_match) {
            return false;
        }
        pending.extend(left.children().zip(right.children()));
    }
    true
}

/// A symbolic expression: a shared handle to an immutable node.
///
/// Cloning bumps a reference count and shares the node. `==` and `Hash`
/// compare trees structurally; [`ptr_eq`](Self::ptr_eq) compares handles.
/// There is no `Display`: the printer formats expressions, and `Debug`
/// shows the node structure.
///
/// Dropping, comparing, hashing, substituting into, collecting the free
/// identifiers of, and screening a tree keep their pending nodes on the
/// heap, so they handle a tree of any depth on any thread. `Debug`,
/// serialization, and deserialization recurse once per tree level: a
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
/// use fhy_core::symbolic::expression::{Expression, ExpressionKind};
///
/// let x = Expression::from(Identifier::new("x"));
/// let sum = &x + 1;
/// let same_sum = &x + 1;
/// assert_eq!(sum, same_sum);
/// assert!(!Expression::ptr_eq(&sum, &same_sum));
/// assert!(matches!(sum.kind(), ExpressionKind::Binary(_)));
/// ```
///
/// Expressions have no order: `<` does not compare them, and a comparison
/// node is built with [`less`](Self::less) and its siblings.
///
/// ```compile_fail,E0369
/// use fhy_core::identifier::Identifier;
/// use fhy_core::symbolic::expression::Expression;
///
/// let x = Expression::from(Identifier::new("x"));
/// let y = Expression::from(Identifier::new("y"));
/// let ordered = x < y;
/// ```
#[derive(Debug, Clone)]
pub struct Expression(Arc<ExpressionKind>);

/// The node an [`Expression`] refers to, one variant per node kind.
#[derive(Debug, Clone)]
pub enum ExpressionKind {
    /// A unary operation applied to one operand.
    Unary(UnaryExpression),
    /// A binary operation applied to two operands.
    Binary(BinaryExpression),
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
    /// right operand; a piecewise node each case's condition then value, in
    /// case order, then its otherwise branch (`c0, v0, c1, v1, ...,
    /// otherwise`); a call its arguments in order. An identifier or a
    /// literal has no children.
    #[must_use]
    pub fn children(&self) -> impl DoubleEndedIterator<Item = &Expression> {
        match self.kind() {
            ExpressionKind::Unary(node) => {
                Children::Contiguous(std::slice::from_ref(&node.operand).iter())
            }
            ExpressionKind::Binary(node) => Children::Contiguous(node.operands.iter()),
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

    /// Build a node of the same kind and operation from new children, given
    /// in [`children`](Self::children) order.
    ///
    /// An identifier or a literal takes no children and returns a handle to
    /// itself. A piecewise rebuilt from `2n + 1` children has `n` cases; a
    /// call keeps its function name.
    ///
    /// # Errors
    ///
    /// Returns [`ExpressionBuildError::ChildCountMismatch`] if the number of
    /// children differs from the node's own child count, and
    /// [`ExpressionBuildError::NonBooleanConditionLiteral`] if a new
    /// piecewise case condition is a literal other than a Boolean.
    pub fn rebuild_with_children(
        &self,
        children: Vec<Expression>,
    ) -> Result<Expression, ExpressionBuildError> {
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
            ExpressionKind::Piecewise(_) => {
                let mut children = children.into_iter();
                let otherwise = children
                    .next_back()
                    .ok_or_else(|| build_child_count_mismatch(expected, actual))?;
                let mut cases = Vec::with_capacity(expected / 2);
                while let (Some(condition), Some(value)) = (children.next(), children.next()) {
                    cases.push((condition, value));
                }
                Self::from(PiecewiseExpression::try_new(cases, otherwise)?)
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
    /// free.
    #[must_use]
    pub fn free_identifiers(&self) -> HashSet<Identifier> {
        let mut free = HashSet::new();
        let mut pending = vec![self];
        while let Some(expression) = pending.pop() {
            if let ExpressionKind::Identifier(identifier) = expression.kind() {
                free.insert(identifier.clone());
            }
            pending.extend(expression.children());
        }
        free
    }

    /// Replace every reference to a mapped identifier with its replacement,
    /// simultaneously.
    ///
    /// Replacements are not substituted into in turn, and each occurrence of
    /// a mapped identifier becomes a handle to the same replacement node
    /// ([`ptr_eq`](Self::ptr_eq) with it). An unmapped identifier and a
    /// literal are returned as handles to themselves; every other node is
    /// rebuilt from its substituted children, so a leaf that nothing replaces
    /// keeps its node.
    ///
    /// # Errors
    ///
    /// Returns [`ExpressionBuildError::NonBooleanConditionLiteral`] if a
    /// replacement puts a literal other than a Boolean in a piecewise case
    /// condition.
    #[expect(
        clippy::missing_panics_doc,
        reason = "the one expect guards an invariant of the rebuild walk, never caller input"
    )]
    pub fn substitute<S: BuildHasher>(
        &self,
        replacements: &HashMap<Identifier, Expression, S>,
    ) -> Result<Expression, ExpressionBuildError> {
        let mut steps = vec![RebuildStep::Enter(self)];
        let mut results: Vec<Expression> = Vec::new();
        while let Some(step) = steps.pop() {
            match step {
                RebuildStep::Enter(expression) => match expression.kind() {
                    ExpressionKind::Identifier(identifier) => {
                        results.push(replacements.get(identifier).unwrap_or(expression).clone());
                    }
                    ExpressionKind::Literal(_) => results.push(expression.clone()),
                    _ => {
                        steps.push(RebuildStep::Exit(expression));
                        steps.extend(expression.children().rev().map(RebuildStep::Enter));
                    }
                },
                RebuildStep::Exit(expression) => {
                    let first_child = results.len() - expression.count_children();
                    let children = results.split_off(first_child);
                    results.push(expression.rebuild_with_children(children)?);
                }
            }
        }
        Ok(results
            .pop()
            .expect("a rebuild walk leaves exactly the rebuilt root"))
    }

    /// Return whether `other` is this expression with its free identifiers
    /// renamed by `renaming`.
    ///
    /// The trees must have the same structure. Where this expression refers
    /// to an identifier `renaming` maps, `other` must refer to its image;
    /// where it refers to an unmapped identifier, `other` must refer to the
    /// same identifier, and that identifier must not be an image of
    /// `renaming`. With an empty `renaming` this is structural equality.
    #[must_use]
    pub fn is_alpha_equivalent_under<S: BuildHasher>(
        &self,
        other: &Expression,
        renaming: &HashMap<Identifier, Identifier, S>,
    ) -> bool {
        let images: HashSet<&Identifier> = renaming.values().collect();
        is_tree_equal(self, other, renaming.is_empty(), &|left, right| {
            is_identifier_renamed(left, right, renaming, &images)
        })
    }

    /// Return the number of children, the length of
    /// [`children`](Self::children).
    fn count_children(&self) -> usize {
        match self.kind() {
            ExpressionKind::Identifier(_) | ExpressionKind::Literal(_) => 0,
            ExpressionKind::Unary(_) => 1,
            ExpressionKind::Binary(_) => 2,
            ExpressionKind::Piecewise(node) => 2 * node.cases.len() + 1,
            ExpressionKind::Call(node) => node.arguments.len(),
        }
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
    /// Compare structurally: same node kinds and operations, identifiers
    /// with the same ids, equal literals, children equal in order.
    fn eq(&self, other: &Self) -> bool {
        is_tree_equal(self, other, true, &|left, right| left == right)
    }
}

impl Eq for Expression {}

impl Hash for Expression {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let mut pending = vec![self];
        while let Some(expression) = pending.pop() {
            match expression.kind() {
                ExpressionKind::Unary(node) => {
                    state.write_u8(0);
                    node.operation.hash(state);
                }
                ExpressionKind::Binary(node) => {
                    state.write_u8(1);
                    node.operation.hash(state);
                }
                ExpressionKind::Identifier(identifier) => {
                    state.write_u8(2);
                    identifier.hash(state);
                }
                ExpressionKind::Literal(value) => {
                    state.write_u8(3);
                    value.hash(state);
                }
                ExpressionKind::Piecewise(node) => {
                    state.write_u8(4);
                    state.write_usize(node.cases.len());
                }
                ExpressionKind::Call(node) => {
                    state.write_u8(5);
                    node.function_name.hash(state);
                    state.write_usize(node.arguments.len());
                }
            }
            pending.extend(expression.children().rev());
        }
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

/// Check a piecewise with `case_count` cases has at least one.
///
/// The constructor and the wire decoder share this check, so a payload is
/// refused for the same reason before any of its identifiers is restored.
///
/// # Errors
///
/// Returns [`ExpressionBuildError::EmptyPiecewise`] if `case_count` is zero.
pub(super) fn validate_case_count(case_count: usize) -> Result<(), ExpressionBuildError> {
    if case_count == 0 {
        return Err(ExpressionBuildError::EmptyPiecewise);
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
/// Returns [`ExpressionBuildError::NonBooleanConditionLiteral`] naming
/// `case_index` if `condition` is not a Boolean.
pub(super) fn validate_condition_literal(
    case_index: usize,
    condition: &LiteralValue,
) -> Result<(), ExpressionBuildError> {
    if !matches!(condition.kind(), LiteralKind::Bool(_)) {
        return Err(ExpressionBuildError::NonBooleanConditionLiteral { case_index });
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
/// Returns [`ExpressionBuildError::EmptyFunctionName`] if `function_name` is
/// empty.
pub(super) fn validate_function_name(function_name: &str) -> Result<(), ExpressionBuildError> {
    if function_name.is_empty() {
        return Err(ExpressionBuildError::EmptyFunctionName);
    }
    Ok(())
}

impl PiecewiseExpression {
    /// Construct a piecewise from its `(condition, value)` cases, in
    /// evaluation order, and its otherwise branch.
    ///
    /// # Errors
    ///
    /// Returns [`ExpressionBuildError::EmptyPiecewise`] if `cases` is empty,
    /// and [`ExpressionBuildError::NonBooleanConditionLiteral`] naming the
    /// first case whose condition is a literal other than a Boolean.
    pub fn try_new(
        cases: Vec<(Expression, Expression)>,
        otherwise: Expression,
    ) -> Result<Self, ExpressionBuildError> {
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
    /// Returns [`ExpressionBuildError::EmptyFunctionName`] if
    /// `function_name` is empty.
    pub fn try_new(
        function_name: &str,
        arguments: Vec<Expression>,
    ) -> Result<Self, ExpressionBuildError> {
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
