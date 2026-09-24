//! Captures, patterns, match bindings, and matching.
//!
//! A [`Pattern`] describes a shape an [`Expression`] may have: any
//! expression or none, a literal or an identifier (any, or a specific one),
//! a node of a given kind over sub-patterns, a Boolean test supplied by the
//! caller, or a choice among alternatives. A pattern captured as a
//! [`Capture`] records the expression it matched in [`MatchBindings`]; a
//! capture used twice must match structurally equal expressions both times.
//!
//! Matching is one-shot and anchored at the root: [`Pattern::matches`]
//! tests the given expression only and never searches its subexpressions.
//! Structural mismatches are not errors; only a predicate supplied by the
//! caller can fail, with a [`CallbackError`].

use std::collections::hash_map::DefaultHasher;
use std::error::Error;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Index;
use std::sync::Arc;

use crate::identifier::Identifier;

use super::super::callee::Callee;
use super::super::literal::LiteralValue;
use super::super::node::{Expression, ExpressionKind};
use super::super::operation::{BinaryOperation, LogicalOperation, UnaryOperation};

/// The failure of a caller-supplied callback: a pattern predicate, a rewrite
/// rule's guard, or a rewrite rule's rewrite.
///
/// It is the callback's own error, boxed: `?` converts any error type, a
/// `String` or a `&str` into it, and `downcast_ref` recovers the error the
/// callback returned. A match or a rewrite walk returns it unchanged.
///
/// # Examples
///
/// ```
/// use fhy_core::expr::{Expression, LiteralValue};
/// use fhy_core::expr::pattern::{CallbackError, Pattern};
///
/// let refusing = Pattern::try_predicate(|_| Err(CallbackError::from("no verdict")));
///
/// let result = refusing.matches(&Expression::from(LiteralValue::from(1)));
///
/// let error = result.expect_err("the predicate fails");
/// assert_eq!(error.to_string(), "no verdict");
/// ```
pub type CallbackError = Box<dyn Error + Send + Sync + 'static>;

/// A handle a pattern binds a matched expression to.
///
/// Identity, not name, decides equality: two captures are equal exactly
/// when one is a clone of the other, so `Capture::new("x")` called twice
/// gives two independent captures. The name serves only `Debug`, `Display`
/// and panic messages, and it may be empty. Hashing agrees with equality.
///
/// # Examples
///
/// ```
/// use fhy_core::expr::pattern::Capture;
///
/// let x = Capture::new("x");
///
/// assert_eq!(x, x.clone());
/// assert_ne!(x, Capture::new("x"));
/// assert_eq!(x.name(), "x");
/// ```
#[derive(Clone)]
pub struct Capture(Arc<str>);

impl Capture {
    /// Create a capture named `name`, distinct from every other capture.
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self(Arc::from(name))
    }

    /// Return the name the capture was created with.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.0
    }
}

impl PartialEq for Capture {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for Capture {}

impl Hash for Capture {
    /// Feed the address of the capture's shared name.
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(Arc::as_ptr(&self.0).cast::<()>().addr());
    }
}

impl fmt::Debug for Capture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Capture").field(&self.name()).finish()
    }
}

impl fmt::Display for Capture {
    /// Write the [`name`](Self::name).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A Boolean test over a candidate expression, supplied by the caller.
type PredicateFn = Arc<dyn Fn(&Expression) -> Result<bool, CallbackError> + Send + Sync>;

/// A caller-supplied predicate, opaque to `Debug`.
#[derive(Clone)]
struct Predicate(PredicateFn);

impl fmt::Debug for Predicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Predicate").finish_non_exhaustive()
    }
}

/// The shape a [`Pattern`] describes. `None` leaves a part of a node
/// unconstrained.
#[derive(Debug)]
enum PatternKind {
    Wildcard,
    Nothing,
    Captured {
        pattern: Pattern,
        capture: Capture,
    },
    Literal(Option<LiteralValue>),
    Identifier(Option<Identifier>),
    Unary {
        operation: Option<UnaryOperation>,
        operand: Pattern,
    },
    Binary {
        operation: Option<BinaryOperation>,
        left: Pattern,
        right: Pattern,
    },
    Logical {
        operation: Option<LogicalOperation>,
        operands: Option<Box<[Pattern]>>,
    },
    Piecewise {
        cases: Option<Box<[(Pattern, Pattern)]>>,
        otherwise: Pattern,
    },
    Call {
        callee: Option<Callee>,
        arguments: Option<Box<[Pattern]>>,
    },
    Predicate(Predicate),
    Alternatives(Box<[Pattern]>),
}

/// Match each `(pattern, expression)` pair in order into `bindings`,
/// stopping at the first that fails.
fn match_each<'a>(
    pairs: impl IntoIterator<Item = (&'a Pattern, &'a Expression)>,
    bindings: &mut MatchBindings,
) -> Result<bool, CallbackError> {
    for (pattern, expression) in pairs {
        if !pattern.match_into(expression, bindings)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Match `expressions` against `patterns`, one to one and in order, or
/// accept any list for `None`.
fn match_list(
    patterns: Option<&[Pattern]>,
    expressions: &[Expression],
    bindings: &mut MatchBindings,
) -> Result<bool, CallbackError> {
    let Some(patterns) = patterns else {
        return Ok(true);
    };
    if patterns.len() != expressions.len() {
        return Ok(false);
    }
    match_each(patterns.iter().zip(expressions), bindings)
}

/// A shape an expression may have, with captures.
///
/// A pattern is built with one constructor per shape and matched with
/// [`matches`](Self::matches) or [`is_match`](Self::is_match). A
/// constructor named `<kind>_any_<part>` leaves that part of the node
/// unconstrained, and one named `any_<kind>` leaves every part of it
/// unconstrained. Construction never fails; a pattern that could never
/// match, such as [`nothing`](Self::nothing) or an alternatives pattern of
/// no alternative, simply matches nothing.
///
/// Compound patterns match their sub-patterns in a fixed order, threading
/// one set of bindings through them: a unary pattern its operand; a binary
/// pattern its left then its right operand; a logical pattern its operands
/// in order; a piecewise pattern each case's condition then value, in case
/// order, then the otherwise branch; a call pattern its arguments in order.
/// Matching stops at the first sub-pattern that fails. Operand order is
/// literal: nothing is matched modulo commutativity or associativity.
///
/// Matching recurses once per level of the pattern and not at all below its
/// leaves, so matching a shallow pattern against a deep expression costs
/// little stack; a pattern nested 4000 levels deep matches within a 16 MiB
/// thread stack. Cloning is cheap and shares the pattern, predicates
/// included.
///
/// # Examples
///
/// ```
/// use fhy_core::identifier::Identifier;
/// use fhy_core::expr::{BinaryOperation, Expression};
/// use fhy_core::expr::pattern::{Capture, Pattern};
///
/// // `x + 0`, capturing `x`.
/// let x = Capture::new("x");
/// let pattern = Pattern::binary(BinaryOperation::Add, Pattern::capture(&x), Pattern::literal(0));
/// let a = Expression::from(Identifier::new("a"));
///
/// let bindings = pattern.matches(&(&a + 0))?.expect("`a + 0` matches");
/// assert!(Expression::ptr_eq(&bindings[&x], &a));
/// assert!(!pattern.is_match(&(&a + 1))?);
/// # Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
/// ```
#[derive(Debug, Clone)]
pub struct Pattern(Arc<PatternKind>);

impl Pattern {
    fn from_kind(kind: PatternKind) -> Self {
        Self(Arc::new(kind))
    }

    /// Build the pattern every expression matches, capturing nothing.
    #[must_use]
    pub fn wildcard() -> Self {
        Self::from_kind(PatternKind::Wildcard)
    }

    /// Build the pattern no expression matches.
    #[must_use]
    pub fn nothing() -> Self {
        Self::from_kind(PatternKind::Nothing)
    }

    /// Build the pattern matching any expression and binding `capture` to
    /// it; the same as `Pattern::wildcard().captured_as(capture)`.
    #[must_use]
    pub fn capture(capture: &Capture) -> Self {
        Self::wildcard().captured_as(capture)
    }

    /// Return a pattern matching what this one matches and binding
    /// `capture` to the matched expression.
    ///
    /// The capture binds after this pattern has matched, so the captures
    /// inside it are bound first. When `capture` is already bound, the match
    /// succeeds only if the expression equals the bound one structurally
    /// (with [`LiteralValue`]'s equality for literals), and the binding keeps
    /// the first-bound expression.
    #[must_use]
    pub fn captured_as(self, capture: &Capture) -> Self {
        Self::from_kind(PatternKind::Captured {
            pattern: self,
            capture: capture.clone(),
        })
    }

    /// Build the pattern matching every literal.
    #[must_use]
    pub fn any_literal() -> Self {
        Self::from_kind(PatternKind::Literal(None))
    }

    /// Build the pattern matching every literal equal to `value` under
    /// [`LiteralValue`]'s equality.
    #[must_use]
    pub fn literal(value: impl Into<LiteralValue>) -> Self {
        Self::from_kind(PatternKind::Literal(Some(value.into())))
    }

    /// Build the pattern matching every identifier reference.
    #[must_use]
    pub fn any_identifier() -> Self {
        Self::from_kind(PatternKind::Identifier(None))
    }

    /// Build the pattern matching a reference to an identifier with the
    /// same id as `identifier`.
    #[must_use]
    pub fn identifier(identifier: Identifier) -> Self {
        Self::from_kind(PatternKind::Identifier(Some(identifier)))
    }

    /// Build the pattern matching a unary node of `operation` whose operand
    /// matches `operand`.
    #[must_use]
    pub fn unary(operation: UnaryOperation, operand: Pattern) -> Self {
        Self::from_kind(PatternKind::Unary {
            operation: Some(operation),
            operand,
        })
    }

    /// Build the pattern matching a unary node of any operation whose
    /// operand matches `operand`.
    #[must_use]
    pub fn unary_any_operation(operand: Pattern) -> Self {
        Self::from_kind(PatternKind::Unary {
            operation: None,
            operand,
        })
    }

    /// Build the pattern matching a binary node of `operation` whose left
    /// operand matches `left` and whose right operand matches `right`.
    #[must_use]
    pub fn binary(operation: BinaryOperation, left: Pattern, right: Pattern) -> Self {
        Self::from_kind(PatternKind::Binary {
            operation: Some(operation),
            left,
            right,
        })
    }

    /// Build the pattern matching a binary node of any operation whose left
    /// operand matches `left` and whose right operand matches `right`.
    #[must_use]
    pub fn binary_any_operation(left: Pattern, right: Pattern) -> Self {
        Self::from_kind(PatternKind::Binary {
            operation: None,
            left,
            right,
        })
    }

    /// Build the pattern matching a logical node of `operation` with exactly
    /// as many operands as `operands`, each matching the pattern at its
    /// position.
    ///
    /// A logical node has at least two operands, so fewer than two operand
    /// patterns match nothing.
    #[must_use]
    pub fn logical(
        operation: LogicalOperation,
        operands: impl IntoIterator<Item = Pattern>,
    ) -> Self {
        Self::from_kind(PatternKind::Logical {
            operation: Some(operation),
            operands: Some(operands.into_iter().collect()),
        })
    }

    /// Build the pattern matching a logical node of either operation with
    /// exactly as many operands as `operands`, each matching the pattern at
    /// its position.
    #[must_use]
    pub fn logical_any_operation(operands: impl IntoIterator<Item = Pattern>) -> Self {
        Self::from_kind(PatternKind::Logical {
            operation: None,
            operands: Some(operands.into_iter().collect()),
        })
    }

    /// Build the pattern matching a logical node of `operation` with any
    /// operands, which are not matched.
    #[must_use]
    pub fn logical_any_operands(operation: LogicalOperation) -> Self {
        Self::from_kind(PatternKind::Logical {
            operation: Some(operation),
            operands: None,
        })
    }

    /// Build the pattern matching every logical node.
    #[must_use]
    pub fn any_logical() -> Self {
        Self::from_kind(PatternKind::Logical {
            operation: None,
            operands: None,
        })
    }

    /// Build the pattern matching a piecewise node with exactly as many
    /// cases as `cases`, each case's condition and value matching the
    /// `(condition, value)` pattern pair at its position, and an otherwise
    /// branch matching `otherwise`.
    ///
    /// A piecewise node has at least one case, so an empty `cases` matches
    /// nothing.
    #[must_use]
    pub fn piecewise(
        cases: impl IntoIterator<Item = (Pattern, Pattern)>,
        otherwise: Pattern,
    ) -> Self {
        Self::from_kind(PatternKind::Piecewise {
            cases: Some(cases.into_iter().collect()),
            otherwise,
        })
    }

    /// Build the pattern matching a piecewise node with any cases, which are
    /// not matched, and an otherwise branch matching `otherwise`.
    #[must_use]
    pub fn piecewise_any_cases(otherwise: Pattern) -> Self {
        Self::from_kind(PatternKind::Piecewise {
            cases: None,
            otherwise,
        })
    }

    /// Build the pattern matching a call of `callee` with exactly as many
    /// arguments as `arguments`, each matching the pattern at its position;
    /// no argument pattern matches only calls without arguments.
    #[must_use]
    pub fn call(callee: impl Into<Callee>, arguments: impl IntoIterator<Item = Pattern>) -> Self {
        Self::from_kind(PatternKind::Call {
            callee: Some(callee.into()),
            arguments: Some(arguments.into_iter().collect()),
        })
    }

    /// Build the pattern matching a call of `callee` with any arguments,
    /// which are not matched.
    #[must_use]
    pub fn call_any_arguments(callee: impl Into<Callee>) -> Self {
        Self::from_kind(PatternKind::Call {
            callee: Some(callee.into()),
            arguments: None,
        })
    }

    /// Build the pattern matching a call of any callee with exactly as many
    /// arguments as `arguments`, each matching the pattern at its position.
    #[must_use]
    pub fn call_any_callee(arguments: impl IntoIterator<Item = Pattern>) -> Self {
        Self::from_kind(PatternKind::Call {
            callee: None,
            arguments: Some(arguments.into_iter().collect()),
        })
    }

    /// Build the pattern matching every call.
    #[must_use]
    pub fn any_call() -> Self {
        Self::from_kind(PatternKind::Call {
            callee: None,
            arguments: None,
        })
    }

    /// Build the pattern matching the expressions for which `predicate`
    /// returns `true`, capturing nothing.
    ///
    /// The predicate runs each time the pattern is matched.
    #[must_use]
    pub fn predicate<F>(predicate: F) -> Self
    where
        F: Fn(&Expression) -> bool + Send + Sync + 'static,
    {
        Self::try_predicate(move |expression| Ok(predicate(expression)))
    }

    /// Build the pattern matching the expressions for which the fallible
    /// `predicate` returns `Ok(true)`, capturing nothing.
    ///
    /// The predicate runs each time the pattern is matched. An error it
    /// returns ends the match and is returned from it unchanged.
    #[must_use]
    pub fn try_predicate<F>(predicate: F) -> Self
    where
        F: Fn(&Expression) -> Result<bool, CallbackError> + Send + Sync + 'static,
    {
        Self::from_kind(PatternKind::Predicate(Predicate(Arc::new(predicate))))
    }

    /// Build the pattern trying `alternatives` in order, left to right.
    ///
    /// The first alternative that matches decides the result, and a failed
    /// alternative leaves no captures behind. The choice is final: when an
    /// enclosing pattern later fails, the remaining alternatives are not
    /// tried. No alternative at all matches nothing, like
    /// [`nothing`](Self::nothing).
    #[must_use]
    pub fn alternatives(alternatives: impl IntoIterator<Item = Pattern>) -> Self {
        Self::from_kind(PatternKind::Alternatives(
            alternatives.into_iter().collect(),
        ))
    }

    /// Match `expression` against this pattern, at the root only, and
    /// return the captures, or `None` if it does not match.
    ///
    /// # Errors
    ///
    /// Returns the [`CallbackError`] of the first predicate that fails;
    /// matching stops there.
    pub fn matches(&self, expression: &Expression) -> Result<Option<MatchBindings>, CallbackError> {
        let mut bindings = MatchBindings::new();
        Ok(self
            .match_into(expression, &mut bindings)?
            .then_some(bindings))
    }

    /// Return whether this pattern matches `expression` at the root: the
    /// same as `self.matches(expression)?.is_some()`.
    ///
    /// # Errors
    ///
    /// Returns the [`CallbackError`] of the first predicate that fails;
    /// matching stops there.
    pub fn is_match(&self, expression: &Expression) -> Result<bool, CallbackError> {
        Ok(self.matches(expression)?.is_some())
    }

    /// Match `expression` against this pattern, appending its captures to
    /// `bindings`.
    ///
    /// Returns `Ok(true)` with this pattern's captures appended, or
    /// `Ok(false)` with `bindings` as it was on entry. After an error,
    /// `bindings` holds unspecified captures.
    pub(super) fn match_into(
        &self,
        expression: &Expression,
        bindings: &mut MatchBindings,
    ) -> Result<bool, CallbackError> {
        let mark = bindings.len();
        let is_match = self.match_node(expression, bindings)?;
        if !is_match {
            bindings.truncate(mark);
        }
        Ok(is_match)
    }

    /// Match `expression` against this pattern's own shape, appending
    /// captures to `bindings`, which a failure may leave partly appended.
    fn match_node(
        &self,
        expression: &Expression,
        bindings: &mut MatchBindings,
    ) -> Result<bool, CallbackError> {
        match (&*self.0, expression.kind()) {
            (PatternKind::Wildcard, _) => Ok(true),
            (PatternKind::Captured { pattern, capture }, _) => {
                Ok(pattern.match_into(expression, bindings)? && bindings.bind(capture, expression))
            }
            (PatternKind::Literal(wanted), ExpressionKind::Literal(candidate)) => {
                Ok(wanted.as_ref().is_none_or(|wanted| wanted == candidate))
            }
            (PatternKind::Identifier(wanted), ExpressionKind::Identifier(candidate)) => {
                Ok(wanted.as_ref().is_none_or(|wanted| wanted == candidate))
            }
            (PatternKind::Unary { operation, operand }, ExpressionKind::Unary(node)) => {
                if operation.is_some_and(|wanted| wanted != node.operation()) {
                    return Ok(false);
                }
                operand.match_into(node.operand(), bindings)
            }
            (
                PatternKind::Binary {
                    operation,
                    left,
                    right,
                },
                ExpressionKind::Binary(node),
            ) => {
                if operation.is_some_and(|wanted| wanted != node.operation()) {
                    return Ok(false);
                }
                match_each([(left, node.left()), (right, node.right())], bindings)
            }
            (
                PatternKind::Logical {
                    operation,
                    operands,
                },
                ExpressionKind::Logical(node),
            ) => {
                if operation.is_some_and(|wanted| wanted != node.operation()) {
                    return Ok(false);
                }
                match_list(operands.as_deref(), node.operands(), bindings)
            }
            (PatternKind::Piecewise { cases, otherwise }, ExpressionKind::Piecewise(node)) => {
                let Some(case_patterns) = cases else {
                    return otherwise.match_into(node.otherwise(), bindings);
                };
                if case_patterns.len() != node.cases().len() {
                    return Ok(false);
                }
                let pairs = case_patterns
                    .iter()
                    .zip(node.cases())
                    .flat_map(|((condition_pattern, value_pattern), (condition, value))| {
                        [(condition_pattern, condition), (value_pattern, value)]
                    })
                    .chain([(otherwise, node.otherwise())]);
                match_each(pairs, bindings)
            }
            (PatternKind::Call { callee, arguments }, ExpressionKind::Call(node)) => {
                if callee
                    .as_ref()
                    .is_some_and(|wanted| wanted != node.callee())
                {
                    return Ok(false);
                }
                match_list(arguments.as_deref(), node.arguments(), bindings)
            }
            (PatternKind::Predicate(Predicate(predicate)), _) => predicate(expression),
            (PatternKind::Alternatives(alternatives), _) => {
                for alternative in alternatives {
                    if alternative.match_into(expression, bindings)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            (
                PatternKind::Nothing
                | PatternKind::Literal(_)
                | PatternKind::Identifier(_)
                | PatternKind::Unary { .. }
                | PatternKind::Binary { .. }
                | PatternKind::Logical { .. }
                | PatternKind::Piecewise { .. }
                | PatternKind::Call { .. },
                _,
            ) => Ok(false),
        }
    }
}

/// The captures of a successful match: each bound [`Capture`] with the
/// expression it matched.
///
/// Captures are kept in binding order. A capture binds after its pattern
/// has matched, so the captures inside that pattern precede it, and
/// captures in sibling positions follow the matching order of the enclosing
/// pattern. A capture used again keeps its first position.
///
/// Only a successful match binds captures; [`new`](Self::new) and
/// [`default`](Default::default) give bindings that bind none. Two bindings
/// are equal when they bind the same set of captures, whatever the order,
/// and each capture to structurally equal expressions. Hashing agrees with
/// equality.
///
/// # Examples
///
/// ```
/// use fhy_core::expr::{BinaryOperation, Expression, LiteralValue};
/// use fhy_core::expr::pattern::{Capture, Pattern};
///
/// let (x, y) = (Capture::new("x"), Capture::new("y"));
/// let pattern = Pattern::binary(BinaryOperation::Add, Pattern::capture(&x), Pattern::capture(&y));
/// let one = Expression::from(LiteralValue::from(1));
///
/// let bindings = pattern.matches(&(&one + 2))?.expect("a sum matches");
///
/// assert_eq!(bindings.len(), 2);
/// assert!(Expression::ptr_eq(&bindings[&x], &one));
/// assert_eq!(bindings.get(&y), Some(&Expression::from(LiteralValue::from(2))));
/// # Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
/// ```
#[derive(Debug, Clone, Default)]
pub struct MatchBindings {
    entries: Vec<(Capture, Expression)>,
}

impl MatchBindings {
    /// Create bindings that bind no capture.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Return whether no capture is bound.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return the number of bound captures.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return whether `capture` is bound.
    #[must_use]
    pub fn contains(&self, capture: &Capture) -> bool {
        self.get(capture).is_some()
    }

    /// Return the expression bound to `capture`, or `None` if it is not
    /// bound.
    #[must_use]
    pub fn get(&self, capture: &Capture) -> Option<&Expression> {
        self.entries
            .iter()
            .find(|(bound, _)| bound == capture)
            .map(|(_, expression)| expression)
    }

    /// Return the bound captures with their expressions, in binding order.
    pub fn iter(&self) -> impl Iterator<Item = (&Capture, &Expression)> + '_ {
        self.entries
            .iter()
            .map(|(capture, expression)| (capture, expression))
    }

    /// Bind `capture` to `expression`, or confirm the existing binding:
    /// return `false`, binding nothing, when `capture` is bound to an
    /// unequal expression.
    fn bind(&mut self, capture: &Capture, expression: &Expression) -> bool {
        if let Some(bound) = self.get(capture) {
            return bound == expression;
        }
        self.entries.push((capture.clone(), expression.clone()));
        true
    }

    fn truncate(&mut self, len: usize) {
        self.entries.truncate(len);
    }
}

impl Index<&Capture> for MatchBindings {
    type Output = Expression;

    /// Return the expression bound to `capture`.
    ///
    /// # Panics
    ///
    /// Panics with ``capture `{name}` is not bound`` if `capture` is not
    /// bound: it is not in the matched pattern, or only under an
    /// alternative that did not match.
    fn index(&self, capture: &Capture) -> &Expression {
        self.get(capture)
            .unwrap_or_else(|| panic!("capture `{capture}` is not bound"))
    }
}

impl PartialEq for MatchBindings {
    /// Compare the sets of bound captures, then each capture's expressions
    /// structurally.
    fn eq(&self, other: &Self) -> bool {
        self.entries.len() == other.entries.len()
            && self
                .entries
                .iter()
                .all(|(capture, expression)| other.get(capture) == Some(expression))
    }
}

impl Eq for MatchBindings {}

impl Hash for MatchBindings {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Bound captures are distinct, so equal bindings hold equal entries
        // in some order; a commutative sum of per-entry hashes ignores it.
        let entries_hash = self
            .entries
            .iter()
            .map(|entry| {
                let mut hasher = DefaultHasher::new();
                entry.hash(&mut hasher);
                hasher.finish()
            })
            .fold(0_u64, u64::wrapping_add);
        state.write_usize(self.entries.len());
        state.write_u64(entries_hash);
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Capture>();
    assert_send_sync::<Pattern>();
    assert_send_sync::<MatchBindings>();
};
