//! Screens that refuse a number in a Boolean position.

use std::collections::{HashMap, HashSet};
use std::hash::BuildHasher;

use crate::expr::SymbolType;
use crate::identifier::Identifier;
use crate::tree::{BuildIdentityHasher, NodeHandle, NodeIdentity, Tree};

use super::error::{BooleanPosition, NonBooleanLogicalOperandError};
use super::literal::LiteralValue;
use super::node::{Expression, ExpressionKind};
use super::sort::FunctionSort;

/// What a screen reads beside the tree: the bindings, the declared value
/// kinds, and the registered sorts.
struct ScreenContext<'a, E, T, L: ?Sized> {
    environment: &'a HashMap<Identifier, Expression, E>,
    symbol_types: &'a HashMap<Identifier, SymbolType, T>,
    sorts: &'a L,
}

/// A node waiting to be screened: whether it sits in a Boolean position,
/// and whether the bindings apply inside it (they do not inside a bound
/// value).
struct PendingNode<'a> {
    expression: &'a Expression,
    is_in_boolean_position: bool,
    is_bound_here: bool,
}

/// Return whether a screen may reach `expression` more than once with the
/// same flags: when the tree may share it, or when it sits inside a bound
/// value, which every reference to the bound identifier reaches.
fn may_recur(expression: &Expression, is_bound_here: bool) -> bool {
    !is_bound_here || expression.is_shared()
}

impl<E: BuildHasher, T: BuildHasher, L: SortLookup + ?Sized> ScreenContext<'_, E, T, L> {
    /// Return the value `identifier` is bound to, unless it is a native
    /// constant's canonical identifier or the bindings do not apply.
    fn find_binding(&self, identifier: &Identifier, is_bound_here: bool) -> Option<&Expression> {
        if !is_bound_here || self.sorts.native_constant_sort(identifier).is_some() {
            return None;
        }
        self.environment.get(identifier)
    }

    /// Return whether `expression` provably denotes a number.
    ///
    /// A piecewise does when all its branches do, so the branches are
    /// checked from a work list rather than by recursion. A node reached
    /// again with the same flags is checked once: the answer is the
    /// conjunction over every node reached, so checking it again adds
    /// nothing.
    fn is_provably_numeric(&self, expression: &Expression, is_bound_here: bool) -> bool {
        let mut checked: HashSet<(NodeIdentity, bool), BuildIdentityHasher> = HashSet::default();
        let mut pending = vec![(expression, is_bound_here)];
        while let Some((expression, is_bound_here)) = pending.pop() {
            if may_recur(expression, is_bound_here)
                && !checked.insert((expression.identity(), is_bound_here))
            {
                continue;
            }
            let is_numeric = match expression.kind() {
                ExpressionKind::Literal(literal) => !matches!(literal, LiteralValue::Bool(_)),
                ExpressionKind::Unary(node) => node.operation().is_arithmetic(),
                ExpressionKind::Binary(node) => node.operation().is_arithmetic(),
                ExpressionKind::Logical(_) => false,
                ExpressionKind::Call(node) => self
                    .sorts
                    .call_result_sort(node.function_name())
                    .is_some_and(|sort| sort != FunctionSort::Bool),
                ExpressionKind::Piecewise(node) => {
                    pending.extend(node.cases().iter().map(|(_, value)| (value, is_bound_here)));
                    pending.push((node.otherwise(), is_bound_here));
                    continue;
                }
                ExpressionKind::Identifier(identifier) => {
                    if let Some(sort) = self.sorts.native_constant_sort(identifier) {
                        sort != FunctionSort::Bool
                    } else if let Some(bound) = self.find_binding(identifier, is_bound_here) {
                        pending.push((bound, false));
                        continue;
                    } else {
                        matches!(
                            self.symbol_types.get(identifier),
                            Some(SymbolType::Int | SymbolType::Real)
                        )
                    }
                }
            };
            if !is_numeric {
                return false;
            }
        }
        true
    }

    /// Report the first Boolean-position operand under `root` that provably
    /// denotes a number, in depth-first pre-order, as an error.
    ///
    /// A node reached again with the same flags is screened once: the walk
    /// is depth-first, so by then everything below its first occurrence
    /// has been screened without an error.
    fn find_numeric_operand(
        &self,
        root: &Expression,
        is_root_in_boolean_position: bool,
    ) -> Result<(), NonBooleanLogicalOperandError> {
        let mut screened: HashSet<(NodeIdentity, bool, bool), BuildIdentityHasher> =
            HashSet::default();
        let mut pending = vec![PendingNode {
            expression: root,
            is_in_boolean_position: is_root_in_boolean_position,
            is_bound_here: true,
        }];
        while let Some(node) = pending.pop() {
            if may_recur(node.expression, node.is_bound_here)
                && !screened.insert((
                    node.expression.identity(),
                    node.is_in_boolean_position,
                    node.is_bound_here,
                ))
            {
                continue;
            }
            if let ExpressionKind::Identifier(identifier) = node.expression.kind() {
                if let Some(bound) = self.find_binding(identifier, node.is_bound_here) {
                    pending.push(PendingNode {
                        expression: bound,
                        is_in_boolean_position: node.is_in_boolean_position,
                        is_bound_here: false,
                    });
                }
                continue;
            }
            let positions = find_boolean_positions(node.expression, node.is_in_boolean_position);
            let mut operands: Vec<(&Expression, BooleanPosition)> = node
                .expression
                .children()
                .zip(&positions)
                .filter_map(|(operand, position)| position.map(|position| (operand, position)))
                .collect();
            operands.sort_by_key(|(_, position)| rank_checking_order(*position));
            if let Some((operand, position)) = operands
                .into_iter()
                .find(|(operand, _)| self.is_provably_numeric(operand, node.is_bound_here))
            {
                return Err(NonBooleanLogicalOperandError::new(
                    operand.clone(),
                    Some(node.expression.clone()),
                    position,
                ));
            }
            pending.extend(
                node.expression
                    .children()
                    .rev()
                    .zip(positions.into_iter().rev())
                    .map(|(child, position)| PendingNode {
                        expression: child,
                        is_in_boolean_position: position.is_some(),
                        is_bound_here: node.is_bound_here,
                    }),
            );
        }
        Ok(())
    }
}

/// Return the rank of `position` in the order a node's Boolean-position
/// operands are checked: every case condition, then every case value, then
/// the otherwise branch; the ranks are stable within a kind of position.
fn rank_checking_order(position: BooleanPosition) -> u8 {
    match position {
        BooleanPosition::NegatedOperand
        | BooleanPosition::LogicalOperand { .. }
        | BooleanPosition::CaseCondition { .. }
        | BooleanPosition::PredicateRoot => 0,
        BooleanPosition::CaseValue { .. } => 1,
        BooleanPosition::Otherwise => 2,
    }
}

/// Return, for each child of `expression` in order, the Boolean position it
/// sits in, if any.
///
/// The operands of a logical negation, conjunction, or disjunction and the
/// conditions of a piecewise are Boolean positions; a piecewise's case
/// values and otherwise branch are too when the piecewise itself sits in a
/// Boolean position.
fn find_boolean_positions(
    expression: &Expression,
    is_in_boolean_position: bool,
) -> Vec<Option<BooleanPosition>> {
    match expression.kind() {
        ExpressionKind::Unary(node) if node.operation().is_logical_connective() => {
            vec![Some(BooleanPosition::NegatedOperand)]
        }
        ExpressionKind::Logical(node) => (0..node.operands().len())
            .map(|operand_index| {
                Some(BooleanPosition::LogicalOperand {
                    operation: node.operation(),
                    operand_index,
                })
            })
            .collect(),
        ExpressionKind::Piecewise(node) => {
            let mut positions = Vec::with_capacity(2 * node.cases().len() + 1);
            for case_index in 0..node.cases().len() {
                positions.push(Some(BooleanPosition::CaseCondition { case_index }));
                positions.push(
                    is_in_boolean_position.then_some(BooleanPosition::CaseValue { case_index }),
                );
            }
            positions.push(is_in_boolean_position.then_some(BooleanPosition::Otherwise));
            positions
        }
        _ => vec![None; expression.children().count()],
    }
}

/// The sorts of registered native constants and function results.
///
/// The screens ask it for the sort of a native constant's canonical
/// identifier and for the result sort of a called function.
pub trait SortLookup {
    /// Return the sort of the native constant `identifier` is the canonical
    /// identifier of, or `None` if it is no native constant's.
    fn native_constant_sort(&self, identifier: &Identifier) -> Option<FunctionSort>;

    /// Return the result sort of the function registered as
    /// `function_name`, or `None` if no function is registered under it.
    fn call_result_sort(&self, function_name: &str) -> Option<FunctionSort>;
}

/// A [`SortLookup`] that knows no native constant and no function.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct NoRegisteredSorts;

impl SortLookup for NoRegisteredSorts {
    /// Return `None`.
    fn native_constant_sort(&self, _identifier: &Identifier) -> Option<FunctionSort> {
        None
    }

    /// Return `None`.
    fn call_result_sort(&self, _function_name: &str) -> Option<FunctionSort> {
        None
    }
}

/// Check that no Boolean position in `expression` holds an operand that
/// provably denotes a number.
///
/// A Boolean position is an operand of a logical negation, conjunction, or
/// disjunction, or a piecewise case condition; a piecewise that itself sits
/// in a Boolean position puts its case values and its otherwise branch in
/// one too.
///
/// `environment` binds identifiers to the values that will be substituted
/// for them, and `symbol_types` declares the value kinds of the identifiers
/// left free after that substitution; `sorts` reports native constants and
/// call result sorts. Pass empty maps when nothing is bound or declared.
///
/// An operand provably denotes a number when it is:
///
/// - a literal other than a Boolean;
/// - a unary node other than a logical negation, or an arithmetic binary
///   node (`+ - * / // % **`);
/// - a call whose result sort, as `sorts` reports it, is not
///   [`FunctionSort::Bool`];
/// - a piecewise whose every case value and otherwise branch provably
///   denotes a number;
/// - an identifier `sorts` reports as a native constant of a sort other than
///   [`FunctionSort::Bool`], whatever `environment` binds to it;
/// - any other identifier `environment` binds to a value that provably
///   denotes a number, judged with no binding applied to the value in turn;
/// - an unbound identifier `symbol_types` declares [`SymbolType::Int`] or
///   [`SymbolType::Real`].
///
/// Anything else, such as an undeclared unbound identifier or a call `sorts`
/// does not know, passes: the screen refuses what it can prove ill-typed,
/// not everything it cannot prove well-typed.
///
/// The walk is depth-first and pre-order: at each node the Boolean-position
/// operands are checked (a connective's operands left to right; a
/// piecewise's conditions, then its values, then its otherwise branch) and
/// the first offending one is reported, before the walk descends into the
/// children in [`Expression::children`] order. The walk keeps its pending
/// nodes on the heap, so it handles a tree of any depth, and screens a
/// subtree it meets again in the same position once, so a DAG costs time
/// linear in its distinct nodes. An identifier `environment` binds (and
/// `sorts` does not report as a native constant) is screened by walking its
/// bound value in the identifier's own position, with no binding applied
/// inside it.
///
/// # Errors
///
/// Returns [`NonBooleanLogicalOperandError`] for the first Boolean position,
/// in walk order, whose operand provably denotes a number.
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
///
/// use fhy_core::expr::{
///     BooleanPosition, Expression, NoRegisteredSorts, validate_logical_operands,
/// };
///
/// let ill_typed = Expression::all([2, 4]);
/// let error = validate_logical_operands(
///     &ill_typed,
///     &HashMap::new(),
///     &HashMap::new(),
///     &NoRegisteredSorts,
/// )
/// .expect_err("a number under a conjunction is refused");
/// assert!(matches!(error.position(), BooleanPosition::LogicalOperand { .. }));
/// ```
pub fn validate_logical_operands<E, T, L>(
    expression: &Expression,
    environment: &HashMap<Identifier, Expression, E>,
    symbol_types: &HashMap<Identifier, SymbolType, T>,
    sorts: &L,
) -> Result<(), NonBooleanLogicalOperandError>
where
    E: BuildHasher,
    T: BuildHasher,
    L: SortLookup + ?Sized,
{
    let context = ScreenContext {
        environment,
        symbol_types,
        sorts,
    };
    context.find_numeric_operand(expression, false)
}

/// Check that `expression` can be used as a predicate.
///
/// Performs the check of [`validate_logical_operands`] with the root
/// itself in a Boolean position: the root must not provably denote a
/// number, a root piecewise has its case values and otherwise branch
/// screened too, and a root identifier the environment binds has its bound
/// value walked as a Boolean position.
///
/// # Errors
///
/// Returns [`NonBooleanLogicalOperandError`] with
/// [`BooleanPosition::PredicateRoot`](super::BooleanPosition::PredicateRoot)
/// if the root provably denotes a number, and otherwise for the first
/// Boolean position whose operand provably denotes a number.
pub fn validate_predicate<E, T, L>(
    expression: &Expression,
    environment: &HashMap<Identifier, Expression, E>,
    symbol_types: &HashMap<Identifier, SymbolType, T>,
    sorts: &L,
) -> Result<(), NonBooleanLogicalOperandError>
where
    E: BuildHasher,
    T: BuildHasher,
    L: SortLookup + ?Sized,
{
    let context = ScreenContext {
        environment,
        symbol_types,
        sorts,
    };
    if context.is_provably_numeric(expression, true) {
        return Err(NonBooleanLogicalOperandError::new(
            expression.clone(),
            None,
            BooleanPosition::PredicateRoot,
        ));
    }
    context.find_numeric_operand(expression, true)
}
