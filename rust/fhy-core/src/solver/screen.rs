//! The hazard screen: the shapes a question's expression is refused for
//! before it is lowered, because SMT-LIB2 arithmetic cannot state them in
//! this crate's semantics.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt;

use num_bigint::Sign;

use crate::expression::builtins::BuiltinConstant;
use crate::expression::{
    BinaryOperation, Callee, Expression, ExpressionKind, FunctionSort, LiteralValue, SortLookup,
    SymbolType, SymbolTypes, UnaryOperation,
};
use crate::identifier::Identifier;
use crate::tree::{BuildIdentityHasher, NodeHandle, NodeIdentity, Tree};

use super::error::write_identifiers;

/// A shape the hazard screen refuses an expression for, with what it
/// refused.
///
/// [`Hazard::find`] reports the first one it finds, checking the five
/// kinds in the order of the variants.
///
/// # Examples
///
/// ```
/// use fhy_core::expression::{Expression, NoRegisteredSorts, SymbolType};
/// use fhy_core::identifier::Identifier;
/// use fhy_core::solver::Hazard;
///
/// let x = Identifier::new("x");
/// let halved = Expression::from(x) / 0;
/// let symbol_types = |_: &Identifier| Some(SymbolType::Real);
/// let hazard = Hazard::find(&halved.equals(1.0), &symbol_types, &NoRegisteredSorts);
///
/// assert!(matches!(hazard, Some(Hazard::PartialOperation(node)) if node == halved));
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Hazard {
    /// The expression refers to native constants, ordered by id: a
    /// built-in constant's identifier, or one the sort lookup reports as a
    /// native constant's. SMT-LIB2 has no term for their values, and a
    /// variable in their place would let a solver choose them.
    NativeConstant(Vec<Identifier>),
    /// A float literal that is infinite or NaN, which has no rational value.
    NonFiniteLiteral(Expression),
    /// A node putting a Boolean where a number is required: arithmetic with
    /// a Boolean operand, a comparison of a Boolean with a number, or a
    /// piecewise whose branches mix the two.
    BooleanCoercion(Expression),
    /// A partial operation off the domain its lowering is sound on: a
    /// division whose divisor is not a finite nonzero integer or float
    /// literal, or whose operands are not provably real; a floor division
    /// or floor modulo whose divisor is not a finite positive integer or
    /// float literal; or a power whose exponent is not an integer literal
    /// of at least one.
    PartialOperation(Expression),
    /// An equality or inequality of a numeric literal with an operand whose
    /// integer or real kind differs from the literal's or is unknown, which
    /// a solver's numeric comparison would equate where this crate keeps
    /// `1` and `1.0` apart.
    MixedIntRealEquality(Expression),
}

impl Hazard {
    /// Return the first hazard in `expression`, or `None` when it lowers
    /// soundly.
    ///
    /// The five kinds are checked in the order of the variants, and the
    /// first kind found is reported: for the native constants, every one
    /// the expression refers to; for the other kinds, the first node in
    /// depth-first pre-order, children in [`Expression::children`] order.
    ///
    /// The screen reads the value kind of an identifier from
    /// `symbol_types`, the result sort of a call of a named function and
    /// the native constants from `sorts`, and the built-in functions and
    /// constants from their catalogue. It classifies each node twice: the
    /// sort it lowers to (Boolean, numeric, or undetermined), and the
    /// integer or real kind it evaluates to, if that can be determined. A
    /// run keeps its pending nodes on the heap and remembers each shared
    /// node's classifications, so it handles a tree of any depth in time
    /// linear in its distinct nodes.
    #[must_use]
    pub fn find(
        expression: &Expression,
        symbol_types: &dyn SymbolTypes,
        sorts: &dyn SortLookup,
    ) -> Option<Hazard> {
        let constants = find_native_constants(expression, sorts);
        if !constants.is_empty() {
            return Some(Self::NativeConstant(constants));
        }
        let mut screen = Screen::new(symbol_types, sorts);
        if let Some(node) = find_first(expression, is_non_finite_literal) {
            return Some(Self::NonFiniteLiteral(node.clone()));
        }
        if let Some(node) = find_first(expression, |node| screen.does_coerce_a_boolean(node)) {
            return Some(Self::BooleanCoercion(node.clone()));
        }
        if let Some(node) = find_first(expression, |node| screen.is_unsafe_partial_operation(node))
        {
            return Some(Self::PartialOperation(node.clone()));
        }
        find_first(expression, |node| {
            screen.does_mix_int_and_real_equality(node)
        })
        .map(|node| Self::MixedIntRealEquality(node.clone()))
    }

    /// Return the node the hazard refuses, or `None` for native constants,
    /// which are refused for the whole expression.
    #[must_use]
    pub fn node(&self) -> Option<&Expression> {
        match self {
            Self::NativeConstant(_) => None,
            Self::NonFiniteLiteral(node)
            | Self::BooleanCoercion(node)
            | Self::PartialOperation(node)
            | Self::MixedIntRealEquality(node) => Some(node),
        }
    }
}

impl fmt::Display for Hazard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NativeConstant(identifiers) => {
                f.write_str(
                    "the expression refers to native constants that smt-lib2 has no term for: ",
                )?;
                write_identifiers(f, identifiers)
            }
            Self::NonFiniteLiteral(_) => {
                f.write_str("the expression holds a non-finite float, which has no rational value")
            }
            Self::BooleanCoercion(_) => {
                f.write_str("the expression lowers a boolean operand into a numeric context")
            }
            Self::PartialOperation(_) => f.write_str(
                "the expression applies a partial operation off the domain its lowering is \
                 sound on",
            ),
            Self::MixedIntRealEquality(_) => f.write_str(
                "the expression compares a numeric literal for equality with an operand of \
                 another or an unknown numeric kind",
            ),
        }
    }
}

/// Return whether `identifier` is a native constant's: a built-in
/// constant's, or one `sorts` reports.
pub(super) fn is_native_constant(identifier: &Identifier, sorts: &dyn SortLookup) -> bool {
    BuiltinConstant::of_identifier(identifier).is_some()
        || sorts.native_constant_sort(identifier).is_some()
}

/// Return the native constants `expression` refers to, ordered by id.
pub(super) fn find_native_constants(
    expression: &Expression,
    sorts: &dyn SortLookup,
) -> Vec<Identifier> {
    let mut constants: Vec<Identifier> = expression
        .free_identifiers()
        .into_iter()
        .filter(|identifier| is_native_constant(identifier, sorts))
        .collect();
    constants.sort_by_key(Identifier::id);
    constants
}

/// Return the first node of `root`, in depth-first pre-order with children
/// in [`Expression::children`] order, that `is_hazard` holds for.
///
/// A shared node met again is skipped: the walk is depth-first, so its
/// subtree has been walked without a hazard by then.
fn find_first<'e>(
    root: &'e Expression,
    mut is_hazard: impl FnMut(&'e Expression) -> bool,
) -> Option<&'e Expression> {
    let mut visited: HashSet<NodeIdentity, BuildIdentityHasher> = HashSet::default();
    let mut pending = vec![root];
    while let Some(node) = pending.pop() {
        if node.is_shared() && !visited.insert(node.identity()) {
            continue;
        }
        if is_hazard(node) {
            return Some(node);
        }
        pending.extend(node.children().rev());
    }
    None
}

/// Return whether `node` is a float literal that is infinite or NaN.
fn is_non_finite_literal(node: &Expression) -> bool {
    matches!(node.kind(), ExpressionKind::Literal(LiteralValue::Float(value)) if !value.is_finite())
}

/// Return the sign of `node` against zero if it is a finite integer or
/// float literal, the literals a divisor is safe as; a Boolean, a decimal
/// and a non-finite float are not.
fn finite_divisor_sign(node: &Expression) -> Option<Ordering> {
    match node.kind() {
        ExpressionKind::Literal(LiteralValue::Int(value)) => Some(match value.sign() {
            Sign::Minus => Ordering::Less,
            Sign::NoSign => Ordering::Equal,
            Sign::Plus => Ordering::Greater,
        }),
        ExpressionKind::Literal(LiteralValue::Float(value)) if value.is_finite() => {
            value.partial_cmp(&0.0)
        }
        _ => None,
    }
}

/// Return whether `node` is an integer literal of at least one, the only
/// exponent a power lowers soundly with.
fn is_safe_exponent(node: &Expression) -> bool {
    matches!(node.kind(), ExpressionKind::Literal(LiteralValue::Int(value)) if value.sign() == Sign::Plus)
}

/// The sort a node lowers to, as far as the tree tells.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LoweredSort {
    Boolean,
    Numeric,
    Undetermined,
}

/// The kind of number a node evaluates to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NumericKind {
    Int,
    Real,
}

/// The classifications of one screen run, remembered per node.
///
/// The nodes are borrowed from the screened expression for the whole run,
/// so their identities stay distinct.
struct Screen<'a> {
    symbol_types: &'a dyn SymbolTypes,
    sorts: &'a dyn SortLookup,
    lowered_sorts: HashMap<NodeIdentity, LoweredSort, BuildIdentityHasher>,
    numeric_kinds: HashMap<NodeIdentity, Option<NumericKind>, BuildIdentityHasher>,
    reals: HashMap<NodeIdentity, bool, BuildIdentityHasher>,
}

/// Return the value of `node` a classification with the dependencies
/// `dependencies` and the rule `combine` gives it, computing it bottom-up
/// on a work list and remembering it in `memo`.
fn classify<T: Copy>(
    node: &Expression,
    memo: &mut HashMap<NodeIdentity, T, BuildIdentityHasher>,
    dependencies: impl Fn(&Expression) -> Vec<&Expression>,
    mut combine: impl FnMut(&Expression, &[T]) -> T,
) -> T {
    let mut pending = vec![(node, false)];
    while let Some((current, is_expanded)) = pending.pop() {
        if memo.contains_key(&current.identity()) {
            continue;
        }
        let needed = dependencies(current);
        if !is_expanded && !needed.is_empty() {
            pending.push((current, true));
            pending.extend(
                needed
                    .into_iter()
                    .filter(|dependency| !memo.contains_key(&dependency.identity()))
                    .map(|dependency| (dependency, false)),
            );
            continue;
        }
        let values: Vec<T> = needed
            .into_iter()
            .map(|dependency| memo[&dependency.identity()])
            .collect();
        let value = combine(current, &values);
        memo.insert(current.identity(), value);
    }
    memo[&node.identity()]
}

/// Return the branches of a piecewise, its case values then its otherwise
/// branch, or nothing for any other node.
fn branches(node: &Expression) -> Vec<&Expression> {
    match node.kind() {
        ExpressionKind::Piecewise(piecewise) => piecewise
            .cases()
            .iter()
            .map(|(_, value)| value)
            .chain(std::iter::once(piecewise.otherwise()))
            .collect(),
        _ => Vec::new(),
    }
}

impl<'a> Screen<'a> {
    fn new(symbol_types: &'a dyn SymbolTypes, sorts: &'a dyn SortLookup) -> Self {
        Self {
            symbol_types,
            sorts,
            lowered_sorts: HashMap::default(),
            numeric_kinds: HashMap::default(),
            reals: HashMap::default(),
        }
    }

    /// Return the sort `node` lowers to: a Boolean literal, identifier,
    /// comparison, connective or negation is Boolean; another literal,
    /// identifier or operation numeric; a piecewise the sort its branches
    /// agree on; a call, an undeclared identifier, or a piecewise whose
    /// branches disagree undetermined.
    fn lowered_sort(&mut self, node: &Expression) -> LoweredSort {
        let symbol_types = self.symbol_types;
        classify(
            node,
            &mut self.lowered_sorts,
            branches,
            |current, values| match current.kind() {
                ExpressionKind::Literal(LiteralValue::Bool(_)) | ExpressionKind::Logical(_) => {
                    LoweredSort::Boolean
                }
                ExpressionKind::Identifier(identifier) => {
                    match symbol_types.symbol_type(identifier) {
                        Some(SymbolType::Bool) => LoweredSort::Boolean,
                        Some(SymbolType::Int | SymbolType::Real) => LoweredSort::Numeric,
                        None => LoweredSort::Undetermined,
                    }
                }
                ExpressionKind::Binary(binary) if binary.operation().is_arithmetic() => {
                    LoweredSort::Numeric
                }
                ExpressionKind::Binary(_) => LoweredSort::Boolean,
                ExpressionKind::Unary(unary) if unary.operation() == UnaryOperation::LogicalNot => {
                    LoweredSort::Boolean
                }
                ExpressionKind::Literal(_) | ExpressionKind::Unary(_) => LoweredSort::Numeric,
                ExpressionKind::Piecewise(_) => match values.split_first() {
                    Some((first, rest)) if rest.iter().all(|value| value == first) => *first,
                    _ => LoweredSort::Undetermined,
                },
                ExpressionKind::Call(_) => LoweredSort::Undetermined,
            },
        )
    }

    /// Return whether `node` itself puts a Boolean where a number is
    /// required: arithmetic with a Boolean operand, a comparison of a
    /// Boolean with a number, or a piecewise whose branches mix them.
    fn does_coerce_a_boolean(&mut self, node: &Expression) -> bool {
        let sorts: Vec<LoweredSort> = match node.kind() {
            ExpressionKind::Binary(binary) => {
                let sorts = vec![
                    self.lowered_sort(binary.left()),
                    self.lowered_sort(binary.right()),
                ];
                if binary.operation().is_arithmetic() {
                    return sorts.contains(&LoweredSort::Boolean);
                }
                sorts
            }
            ExpressionKind::Piecewise(_) => branches(node)
                .into_iter()
                .map(|branch| self.lowered_sort(branch))
                .collect(),
            _ => return false,
        };
        sorts.contains(&LoweredSort::Boolean) && sorts.contains(&LoweredSort::Numeric)
    }

    /// Return whether `node` provably lowers to a real: a real identifier,
    /// a float or decimal literal, arithmetic with a real operand, or the
    /// negation of a real.
    fn is_real(&mut self, node: &Expression) -> bool {
        let symbol_types = self.symbol_types;
        classify(
            node,
            &mut self.reals,
            |current| match current.kind() {
                ExpressionKind::Binary(binary) if binary.operation().is_arithmetic() => {
                    vec![binary.left(), binary.right()]
                }
                ExpressionKind::Unary(unary) if unary.operation() == UnaryOperation::Negate => {
                    vec![unary.operand()]
                }
                _ => Vec::new(),
            },
            |current, values| match current.kind() {
                ExpressionKind::Identifier(identifier) => {
                    symbol_types.symbol_type(identifier) == Some(SymbolType::Real)
                }
                ExpressionKind::Literal(literal) => {
                    matches!(literal, LiteralValue::Float(_) | LiteralValue::Decimal(_))
                }
                _ => values.iter().any(|&value| value),
            },
        )
    }

    /// Return whether `node` applies a partial operation off the domain its
    /// lowering is sound on.
    fn is_unsafe_partial_operation(&mut self, node: &Expression) -> bool {
        let ExpressionKind::Binary(binary) = node.kind() else {
            return false;
        };
        match binary.operation() {
            BinaryOperation::Power => !is_safe_exponent(binary.right()),
            BinaryOperation::FloorDivide | BinaryOperation::FloorMod => {
                finite_divisor_sign(binary.right()) != Some(Ordering::Greater)
            }
            BinaryOperation::Divide => {
                let is_nonzero = matches!(
                    finite_divisor_sign(binary.right()),
                    Some(Ordering::Less | Ordering::Greater)
                );
                !(is_nonzero && (self.is_real(binary.left()) || self.is_real(binary.right())))
            }
            _ => false,
        }
    }

    /// Return the kind of number `node` evaluates to, or `None` when it is
    /// no number or its kind cannot be determined.
    fn numeric_kind(&mut self, node: &Expression) -> Option<NumericKind> {
        let (symbol_types, sorts) = (self.symbol_types, self.sorts);
        classify(
            node,
            &mut self.numeric_kinds,
            |current| match current.kind() {
                ExpressionKind::Unary(unary) if unary.operation() != UnaryOperation::LogicalNot => {
                    vec![unary.operand()]
                }
                ExpressionKind::Binary(binary) if binary.operation().is_arithmetic() => {
                    vec![binary.left(), binary.right()]
                }
                ExpressionKind::Piecewise(_) => branches(current),
                _ => Vec::new(),
            },
            |current, values| match current.kind() {
                ExpressionKind::Literal(LiteralValue::Bool(_)) => None,
                ExpressionKind::Literal(LiteralValue::Int(_)) => Some(NumericKind::Int),
                ExpressionKind::Literal(_) => Some(NumericKind::Real),
                ExpressionKind::Identifier(identifier) => {
                    match symbol_types.symbol_type(identifier) {
                        Some(SymbolType::Int) => Some(NumericKind::Int),
                        Some(SymbolType::Real) => Some(NumericKind::Real),
                        Some(SymbolType::Bool) | None => None,
                    }
                }
                ExpressionKind::Unary(unary) if unary.operation() != UnaryOperation::LogicalNot => {
                    values[0]
                }
                ExpressionKind::Binary(binary) if binary.operation().is_arithmetic() => {
                    combine_binary_kinds(binary.operation(), values[0], values[1], binary.right())
                }
                ExpressionKind::Piecewise(_) => match values.split_first() {
                    Some((first, rest))
                        if first.is_some() && rest.iter().all(|value| value == first) =>
                    {
                        *first
                    }
                    _ => None,
                },
                ExpressionKind::Call(call) => {
                    let result_sort = match call.callee() {
                        Callee::Builtin(function) => Some(function.result_sort()),
                        Callee::Named(name) => sorts.call_result_sort(name),
                    };
                    match result_sort {
                        Some(FunctionSort::Int | FunctionSort::Nat) => Some(NumericKind::Int),
                        Some(FunctionSort::Real) => Some(NumericKind::Real),
                        Some(FunctionSort::Bool) | None => None,
                    }
                }
                ExpressionKind::Unary(_)
                | ExpressionKind::Binary(_)
                | ExpressionKind::Logical(_) => None,
            },
        )
    }

    /// Return whether `node` is an equality or inequality of a numeric
    /// literal with an operand of another or an unknown numeric kind.
    fn does_mix_int_and_real_equality(&mut self, node: &Expression) -> bool {
        let ExpressionKind::Binary(binary) = node.kind() else {
            return false;
        };
        if !matches!(
            binary.operation(),
            BinaryOperation::Equal | BinaryOperation::NotEqual
        ) {
            return false;
        }
        self.is_literal_of_another_kind(binary.left(), binary.right())
            || self.is_literal_of_another_kind(binary.right(), binary.left())
    }

    /// Return whether `candidate` is a numeric literal whose kind `other`
    /// does not provably share.
    fn is_literal_of_another_kind(&mut self, candidate: &Expression, other: &Expression) -> bool {
        if !matches!(candidate.kind(), ExpressionKind::Literal(_)) {
            return false;
        }
        let Some(literal_kind) = self.numeric_kind(candidate) else {
            return false;
        };
        self.numeric_kind(other) != Some(literal_kind)
    }
}

/// Return the kind of number an arithmetic `operation` of operands of the
/// kinds `left` and `right` evaluates to, with `exponent` the right operand
/// of a power.
fn combine_binary_kinds(
    operation: BinaryOperation,
    left: Option<NumericKind>,
    right: Option<NumericKind>,
    exponent: &Expression,
) -> Option<NumericKind> {
    match operation {
        BinaryOperation::Power if left == Some(NumericKind::Int) && is_safe_exponent(exponent) => {
            Some(NumericKind::Int)
        }
        BinaryOperation::Power | BinaryOperation::Divide => match (left?, right?) {
            (NumericKind::Int, NumericKind::Int) => None,
            _ => Some(NumericKind::Real),
        },
        _ => match (left?, right?) {
            (NumericKind::Int, NumericKind::Int) => Some(NumericKind::Int),
            _ => Some(NumericKind::Real),
        },
    }
}
