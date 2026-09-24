//! The screen that refuses a number in a Boolean position.
//!
//! A [`BooleanScreen`] judges an expression against what it is told about
//! the identifiers and functions in it: an [`Environment`] of bindings,
//! [`SymbolTypes`] declaring the value kinds of free identifiers, and a
//! [`SortLookup`] of native constants and named functions. Each defaults to
//! knowing nothing.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::BuildHasher;

use crate::identifier::Identifier;
use crate::tree::{BuildIdentityHasher, NodeHandle, NodeIdentity, Tree};

use super::callee::{Callee, FunctionName};
use super::error::{BooleanPosition, NonBooleanLogicalOperandError};
use super::literal::LiteralValue;
use super::node::{Expression, ExpressionKind};
use super::sort::FunctionSort;
use super::symbol_type::SymbolType;

/// The bindings a screen applies before judging an identifier: the values
/// that will be substituted for identifiers.
pub trait Environment {
    /// Return the value `identifier` is bound to, or `None` if it is
    /// unbound.
    fn binding(&self, identifier: &Identifier) -> Option<&Expression>;
}

impl<S: BuildHasher> Environment for HashMap<Identifier, Expression, S> {
    fn binding(&self, identifier: &Identifier) -> Option<&Expression> {
        self.get(identifier)
    }
}

/// The declared value kinds of the identifiers left free after the
/// bindings are applied.
///
/// Implemented by maps and by closures:
///
/// ```
/// use fhy_core::expr::{BooleanScreen, Expression, SymbolType};
/// use fhy_core::identifier::Identifier;
///
/// let n = Identifier::new("n");
/// let all_integers = |_: &Identifier| Some(SymbolType::Int);
/// let screen = BooleanScreen::new().with_symbol_types(&all_integers);
///
/// assert!(screen.check_predicate(&Expression::from(n)).is_err());
/// ```
pub trait SymbolTypes {
    /// Return the declared value kind of `identifier`, or `None` if it has
    /// none.
    fn symbol_type(&self, identifier: &Identifier) -> Option<SymbolType>;
}

impl<S: BuildHasher> SymbolTypes for HashMap<Identifier, SymbolType, S> {
    fn symbol_type(&self, identifier: &Identifier) -> Option<SymbolType> {
        self.get(identifier).copied()
    }
}

impl<F: Fn(&Identifier) -> Option<SymbolType>> SymbolTypes for F {
    fn symbol_type(&self, identifier: &Identifier) -> Option<SymbolType> {
        self(identifier)
    }
}

/// The environment binding nothing, a bare screen's default.
struct NoBindings;

impl Environment for NoBindings {
    fn binding(&self, _identifier: &Identifier) -> Option<&Expression> {
        None
    }
}

/// The symbol types declaring nothing, a bare screen's default.
struct NoSymbolTypes;

impl SymbolTypes for NoSymbolTypes {
    fn symbol_type(&self, _identifier: &Identifier) -> Option<SymbolType> {
        None
    }
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

/// The answers "provably denotes a number" of the nodes a screen run has
/// walked to judge, by node and by whether the bindings apply at it.
type NumericMemo = HashMap<(NodeIdentity, bool), bool, BuildIdentityHasher>;

/// One step of judging whether a node provably denotes a number.
enum NumericStep<'a> {
    /// Judge the node, with or without the bindings applied.
    Judge(&'a Expression, bool),
    /// Combine the answers of the node's `branch_count` branches, the last
    /// ones on the answer stack, and remember the result.
    Combine(&'a Expression, bool, usize),
}

/// A screen refusing an operand that provably denotes a number in a Boolean
/// position.
///
/// A Boolean position is an operand of a logical negation, conjunction, or
/// disjunction, or a piecewise case condition; a piecewise that itself sits
/// in a Boolean position puts its case values and its otherwise branch in
/// one too.
///
/// The screen judges identifiers and calls by what it is told: the
/// [`Environment`] binds identifiers to the values that will be substituted
/// for them, the [`SymbolTypes`] declare the value kinds of the identifiers
/// left free after that substitution, and the [`SortLookup`] reports native
/// constants and the result sorts of named functions. A bare
/// [`BooleanScreen::new`] knows none of these.
///
/// An operand provably denotes a number when it is:
///
/// - a literal other than a Boolean;
/// - a unary node other than a logical negation, or an arithmetic binary
///   node (`+ - * / // % **`);
/// - a call of a built-in function whose catalogue result sort is not
///   [`FunctionSort::Bool`], or of a named function whose result sort, as
///   the sort lookup reports it, is not;
/// - a piecewise whose every case value and otherwise branch provably
///   denotes a number;
/// - an identifier the sort lookup reports as a native constant of a sort
///   other than [`FunctionSort::Bool`], whatever the environment binds to
///   it;
/// - any other identifier the environment binds to a value that provably
///   denotes a number, judged with no binding applied to the value in turn;
/// - an unbound identifier the symbol types declare [`SymbolType::Int`] or
///   [`SymbolType::Real`].
///
/// Anything else, such as an undeclared unbound identifier or a call of a
/// named function the lookup does not know, passes: the screen refuses what
/// it can prove ill-typed, not everything it cannot prove well-typed.
///
/// The walk is depth-first and pre-order: at each node the Boolean-position
/// operands are checked (a connective's operands left to right; a
/// piecewise's conditions, then its values, then its otherwise branch) and
/// the first offending one is reported, before the walk descends into the
/// children in [`Expression::children`] order. An identifier the
/// environment binds (and the lookup does not report as a native constant)
/// is screened by walking its bound value in the identifier's own position,
/// with no binding applied inside it.
///
/// A run keeps its pending nodes on the heap, so it handles a tree of any
/// depth. It screens a subtree it meets again in the same position once, and
/// judges each piecewise and bound identifier numeric or not at most once
/// per binding state, so a run costs time linear in the distinct nodes, for
/// a DAG and for nested piecewise nodes alike.
///
/// # Examples
///
/// ```
/// use fhy_core::expr::{BooleanPosition, BooleanScreen, Expression, NoRegisteredSorts};
///
/// let ill_typed = Expression::all([2, 4]);
/// let error = BooleanScreen::new()
///     .with_sorts(&NoRegisteredSorts)
///     .check_logical_operands(&ill_typed)
///     .expect_err("a number under a conjunction is refused");
/// assert!(matches!(error.parent(), Some((_, BooleanPosition::LogicalOperand { .. }))));
/// ```
#[derive(Clone, Copy)]
pub struct BooleanScreen<'a> {
    sorts: &'a dyn SortLookup,
    environment: &'a dyn Environment,
    symbol_types: &'a dyn SymbolTypes,
}

impl<'a> BooleanScreen<'a> {
    /// Create the screen that knows no sorts, no bindings and no symbol
    /// types.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sorts: &NoRegisteredSorts,
            environment: &NoBindings,
            symbol_types: &NoSymbolTypes,
        }
    }

    /// Return the screen reading native constants and named functions'
    /// result sorts from `sorts`.
    #[must_use]
    pub fn with_sorts(self, sorts: &'a dyn SortLookup) -> Self {
        Self { sorts, ..self }
    }

    /// Return the screen applying the bindings of `environment`.
    #[must_use]
    pub fn with_environment(self, environment: &'a dyn Environment) -> Self {
        Self {
            environment,
            ..self
        }
    }

    /// Return the screen reading the declared value kinds of free
    /// identifiers from `symbol_types`.
    #[must_use]
    pub fn with_symbol_types(self, symbol_types: &'a dyn SymbolTypes) -> Self {
        Self {
            symbol_types,
            ..self
        }
    }

    /// Check that no Boolean position in `expression` holds an operand that
    /// provably denotes a number.
    ///
    /// # Errors
    ///
    /// Returns [`NonBooleanLogicalOperandError`] for the first Boolean
    /// position, in walk order, whose operand provably denotes a number.
    pub fn check_logical_operands(
        &self,
        expression: &Expression,
    ) -> Result<(), NonBooleanLogicalOperandError> {
        let mut memo = NumericMemo::default();
        self.find_numeric_operand(expression, false, &mut memo)
    }

    /// Check that `expression` can be used as a predicate.
    ///
    /// Performs the check of
    /// [`check_logical_operands`](Self::check_logical_operands) with the
    /// root itself in a Boolean position: the root must not provably denote
    /// a number, a root piecewise has its case values and otherwise branch
    /// screened too, and a root identifier the environment binds has its
    /// bound value walked as a Boolean position.
    ///
    /// # Errors
    ///
    /// Returns [`NonBooleanLogicalOperandError`] with no parent if the root
    /// provably denotes a number, and otherwise for the first Boolean
    /// position whose operand provably denotes a number.
    pub fn check_predicate(
        &self,
        expression: &Expression,
    ) -> Result<(), NonBooleanLogicalOperandError> {
        let mut memo = NumericMemo::default();
        if self.is_provably_numeric(expression, true, &mut memo) {
            return Err(NonBooleanLogicalOperandError::new_predicate_root(
                expression.clone(),
            ));
        }
        self.find_numeric_operand(expression, true, &mut memo)
    }

    /// Return the value `identifier` is bound to, unless it is a native
    /// constant's canonical identifier or the bindings do not apply.
    fn find_binding(&self, identifier: &Identifier, is_bound_here: bool) -> Option<&'a Expression> {
        if !is_bound_here || self.sorts.native_constant_sort(identifier).is_some() {
            return None;
        }
        self.environment.binding(identifier)
    }

    /// Return whether `expression` provably denotes a number.
    ///
    /// A piecewise does when all its branches do, and a bound identifier
    /// when its bound value does, so both are judged from their branches,
    /// bottom-up from a work list rather than by recursion, and their
    /// answers are kept in `memo` for the rest of the run. Every other node
    /// is judged from its own data.
    fn is_provably_numeric<'e>(
        &self,
        expression: &'e Expression,
        is_bound_here: bool,
        memo: &mut NumericMemo,
    ) -> bool
    where
        'a: 'e,
    {
        let mut answers: Vec<bool> = Vec::new();
        let mut pending = vec![NumericStep::Judge(expression, is_bound_here)];
        while let Some(step) = pending.pop() {
            match step {
                NumericStep::Judge(node, is_bound_here) => {
                    if let Some(&answer) = memo.get(&(node.identity(), is_bound_here)) {
                        answers.push(answer);
                        continue;
                    }
                    match self.find_numeric_branches(node, is_bound_here) {
                        Ok(answer) => answers.push(answer),
                        Err(branches) => {
                            pending.push(NumericStep::Combine(node, is_bound_here, branches.len()));
                            pending.extend(
                                branches
                                    .into_iter()
                                    .map(|(branch, bound)| NumericStep::Judge(branch, bound)),
                            );
                        }
                    }
                }
                NumericStep::Combine(node, is_bound_here, branch_count) => {
                    let first = answers.len() - branch_count;
                    let answer = answers.drain(first..).all(|answer| answer);
                    memo.insert((node.identity(), is_bound_here), answer);
                    answers.push(answer);
                }
            }
        }
        answers.pop().unwrap_or(false)
    }

    /// Return whether `node` provably denotes a number, judged from its own
    /// data, or, for a piecewise or a bound identifier, the branches whose
    /// answers decide it: it provably denotes a number when they all do.
    fn find_numeric_branches<'e>(
        &self,
        node: &'e Expression,
        is_bound_here: bool,
    ) -> Result<bool, Vec<(&'e Expression, bool)>>
    where
        'a: 'e,
    {
        let answer = match node.kind() {
            ExpressionKind::Literal(literal) => !matches!(literal, LiteralValue::Bool(_)),
            ExpressionKind::Unary(unary) => unary.operation().is_arithmetic(),
            ExpressionKind::Binary(binary) => binary.operation().is_arithmetic(),
            ExpressionKind::Logical(_) => false,
            ExpressionKind::Call(call) => match call.callee() {
                Callee::Builtin(function) => function.result_sort() != FunctionSort::Bool,
                Callee::Named(name) => self
                    .sorts
                    .call_result_sort(name)
                    .is_some_and(|sort| sort != FunctionSort::Bool),
            },
            ExpressionKind::Piecewise(piecewise) => {
                let mut branches: Vec<(&Expression, bool)> = piecewise
                    .cases()
                    .iter()
                    .map(|(_, value)| (value, is_bound_here))
                    .collect();
                branches.push((piecewise.otherwise(), is_bound_here));
                return Err(branches);
            }
            ExpressionKind::Identifier(identifier) => {
                if let Some(sort) = self.sorts.native_constant_sort(identifier) {
                    sort != FunctionSort::Bool
                } else if let Some(bound) = self.find_binding(identifier, is_bound_here) {
                    return Err(vec![(bound, false)]);
                } else {
                    matches!(
                        self.symbol_types.symbol_type(identifier),
                        Some(SymbolType::Int | SymbolType::Real)
                    )
                }
            }
        };
        Ok(answer)
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
        memo: &mut NumericMemo,
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
                .find(|(operand, _)| self.is_provably_numeric(operand, node.is_bound_here, memo))
            {
                return Err(NonBooleanLogicalOperandError::new(
                    operand.clone(),
                    node.expression.clone(),
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

impl Default for BooleanScreen<'_> {
    /// Return [`BooleanScreen::new`].
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for BooleanScreen<'_> {
    /// Write the type name only: the lookups need not implement `Debug`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BooleanScreen").finish_non_exhaustive()
    }
}

/// Return the rank of `position` in the order a node's Boolean-position
/// operands are checked: every case condition, then every case value, then
/// the otherwise branch; the ranks are stable within a kind of position.
fn rank_checking_order(position: BooleanPosition) -> u8 {
    match position {
        BooleanPosition::NegatedOperand
        | BooleanPosition::LogicalOperand { .. }
        | BooleanPosition::CaseCondition { .. } => 0,
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

/// The sorts of registered native constants and of the results of named
/// functions.
///
/// The screens ask it for the sort of a native constant's canonical
/// identifier and for the result sort of a called [`Callee::Named`]
/// function; a built-in function's result sort comes from the catalogue
/// ([`BuiltinFunction::result_sort`](super::builtins::BuiltinFunction::result_sort)).
/// Both methods default to knowing nothing.
pub trait SortLookup {
    /// Return the sort of the native constant `identifier` is the canonical
    /// identifier of, or `None` if it is no native constant's.
    fn native_constant_sort(&self, _identifier: &Identifier) -> Option<FunctionSort> {
        None
    }

    /// Return the result sort of the user function `name`, or `None` if no
    /// function is registered under it.
    fn call_result_sort(&self, _name: &FunctionName) -> Option<FunctionSort> {
        None
    }
}

/// A [`SortLookup`] that knows no native constant and no named function.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct NoRegisteredSorts;

impl SortLookup for NoRegisteredSorts {}
