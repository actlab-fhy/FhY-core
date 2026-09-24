//! Patterns, match bindings, and matching.
//!
//! A [`Pattern`] describes a shape an [`Expression`] may have: any
//! expression, a literal or an identifier (optionally a specific one), a
//! node of a given kind and operation over sub-patterns, a Boolean test
//! supplied by the caller, or a choice among alternatives. A capture
//! pattern records the expression it matched under a name in
//! [`MatchBindings`]; a name captured twice must match structurally equal
//! expressions both times.
//!
//! Matching is one-shot and anchored at the root: [`match_pattern`] tests
//! the given expression only and never searches its subexpressions.
//! Structural mismatches are not errors; only a predicate supplied by the
//! caller can fail, with a [`CallbackError`].

use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use std::error::Error;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::identifier::Identifier;

use super::super::literal::LiteralValue;
use super::super::node::{Expression, ExpressionKind};
use super::super::operation::{BinaryOperation, UnaryOperation};

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

/// The shape a [`Pattern`] describes.
#[derive(Debug, Clone)]
enum PatternKind {
    Wildcard,
    Capture {
        name: Arc<str>,
        sub_pattern: Box<Pattern>,
    },
    Literal(Option<LiteralValue>),
    Identifier(Option<Identifier>),
    Unary {
        operation: Option<UnaryOperation>,
        operand: Box<Pattern>,
    },
    Binary {
        operation: Option<BinaryOperation>,
        left: Box<Pattern>,
        right: Box<Pattern>,
    },
    Piecewise {
        cases: Option<Vec<(Pattern, Pattern)>>,
        otherwise: Box<Pattern>,
    },
    Call {
        function_name: Option<Arc<str>>,
        arguments: Option<Vec<Pattern>>,
    },
    Predicate(Predicate),
    Alternatives(Vec<Pattern>),
}

/// Return whether `expression` is a literal stored exactly as `value`, or
/// any literal for `None`.
///
/// Literals compare variant by variant with the raw values' `==`, so `0.0`
/// equals `-0.0` and a NaN equals nothing.
fn match_literal(value: Option<&LiteralValue>, expression: &Expression) -> bool {
    let ExpressionKind::Literal(candidate) = expression.kind() else {
        return false;
    };
    value.is_none_or(|wanted| is_stored_alike(wanted, candidate))
}

/// Return whether `left` and `right` are the same variant with raw values
/// equal under the value type's own `==`.
fn is_stored_alike(left: &LiteralValue, right: &LiteralValue) -> bool {
    match (left, right) {
        (LiteralValue::Bool(left), LiteralValue::Bool(right)) => left == right,
        (LiteralValue::Int(left), LiteralValue::Int(right)) => left == right,
        (LiteralValue::Float(left), LiteralValue::Float(right)) => {
            left.partial_cmp(right) == Some(Ordering::Equal)
        }
        (LiteralValue::Decimal(left), LiteralValue::Decimal(right)) => left == right,
        _ => false,
    }
}

/// Match each `(pattern, expression)` pair in order, threading `bindings`
/// through them and stopping at the first failure.
fn match_sequence<'a>(
    pairs: impl IntoIterator<Item = (&'a Pattern, &'a Expression)>,
    bindings: MatchBindings,
) -> Result<Option<MatchBindings>, CallbackError> {
    let mut accumulator = bindings;
    for (pattern, expression) in pairs {
        match pattern.match_from(expression, accumulator)? {
            Some(next) => accumulator = next,
            None => return Ok(None),
        }
    }
    Ok(Some(accumulator))
}

/// A shape an expression may have, with named captures.
///
/// A pattern is built with one constructor per shape and matched with
/// [`match_under`](Self::match_under), [`match_pattern`], or
/// [`does_pattern_match`]. Compound patterns match their sub-patterns in a
/// fixed order, threading one set of bindings through them: a unary
/// pattern its operand; a binary pattern its left then its right operand; a
/// piecewise pattern each case's condition then value, in case order, then
/// the otherwise branch; a call pattern its arguments in order. Matching
/// stops at the first sub-pattern that fails. Operand order is literal:
/// nothing is matched modulo commutativity or associativity.
///
/// Matching recurses once per level of the pattern and not at all below its
/// leaves, so matching a shallow pattern against a deep expression costs
/// little stack; a pattern nested 4000 levels deep matches within a 16 MiB
/// thread stack. Cloning shares the predicates a pattern holds.
///
/// # Examples
///
/// ```
/// use fhy_core::identifier::Identifier;
/// use fhy_core::expr::{BinaryOperation, Expression, LiteralValue};
/// use fhy_core::expr::pattern::{Pattern, match_pattern};
///
/// // `x + 0`, capturing `x`.
/// let pattern = Pattern::binary(
///     Some(BinaryOperation::Add),
///     Pattern::capture("x", Pattern::wildcard())?,
///     Pattern::literal(Some(LiteralValue::from(0))),
/// );
/// let a = Expression::from(Identifier::new("a"));
///
/// let bindings = match_pattern(&pattern, &(&a + 0))?.expect("`a + 0` matches");
/// assert!(Expression::ptr_eq(bindings.get("x").expect("`x` is bound"), &a));
/// assert!(match_pattern(&pattern, &(&a + 1))?.is_none());
/// # Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
/// ```
#[derive(Debug, Clone)]
pub struct Pattern(PatternKind);

impl Pattern {
    /// Build the pattern every expression matches, capturing nothing.
    #[must_use]
    pub fn wildcard() -> Self {
        Self(PatternKind::Wildcard)
    }

    /// Build a pattern that matches what `sub_pattern` matches and binds the
    /// matched expression to `name`.
    ///
    /// The capture binds after `sub_pattern` has matched, so captures inside
    /// `sub_pattern` are bound first. When `name` is already bound, the match
    /// succeeds only if the expression equals the bound one structurally,
    /// and the binding keeps the first-bound expression (see
    /// [`MatchBindings::try_bind`]).
    ///
    /// # Errors
    ///
    /// Returns [`PatternError::EmptyCaptureName`] if `name` is empty.
    pub fn capture(name: &str, sub_pattern: Pattern) -> Result<Self, PatternError> {
        if name.is_empty() {
            return Err(PatternError::EmptyCaptureName);
        }
        Ok(Self(PatternKind::Capture {
            name: Arc::from(name),
            sub_pattern: Box::new(sub_pattern),
        }))
    }

    /// Build a pattern matching a literal: any literal for `None`, or a
    /// literal with exactly the stored form of `value`.
    ///
    /// Exactly the stored form means the same [`LiteralValue`] variant and
    /// an equal raw value: the integer `5` matches the integer parsed from
    /// `"05"` but not the float `5.0` or the decimal `5`, the integer `1`
    /// does not match the Boolean `true`, the decimal parsed from `"1.5"`
    /// matches the one parsed from `"1.50"`, the float `0.0` matches `-0.0`,
    /// and a NaN value matches no literal at all.
    #[must_use]
    pub fn literal(value: Option<LiteralValue>) -> Self {
        Self(PatternKind::Literal(value))
    }

    /// Build a pattern matching an identifier reference: any identifier for
    /// `None`, or a reference to an identifier with the same id as
    /// `identifier`.
    #[must_use]
    pub fn identifier(identifier: Option<Identifier>) -> Self {
        Self(PatternKind::Identifier(identifier))
    }

    /// Build a pattern matching a unary node whose operation is `operation`
    /// (any operation for `None`) and whose operand matches `operand`.
    #[must_use]
    pub fn unary(operation: Option<UnaryOperation>, operand: Pattern) -> Self {
        Self(PatternKind::Unary {
            operation,
            operand: Box::new(operand),
        })
    }

    /// Build a pattern matching a binary node whose operation is `operation`
    /// (any operation for `None`), whose left operand matches `left`, and
    /// whose right operand matches `right`.
    #[must_use]
    pub fn binary(operation: Option<BinaryOperation>, left: Pattern, right: Pattern) -> Self {
        Self(PatternKind::Binary {
            operation,
            left: Box::new(left),
            right: Box::new(right),
        })
    }

    /// Build a pattern matching a piecewise node.
    ///
    /// With `Some(cases)`, the node must have exactly as many cases, and each
    /// case's condition and value must match the corresponding
    /// `(condition, value)` pattern pair. With `None`, the node may have any
    /// number of cases, and they are not matched. The otherwise branch must
    /// match `otherwise` either way.
    ///
    /// # Errors
    ///
    /// Returns [`PatternError::EmptyPiecewiseCases`] if `cases` is
    /// `Some` of an empty list, which no piecewise node could match.
    pub fn piecewise(
        cases: Option<Vec<(Pattern, Pattern)>>,
        otherwise: Pattern,
    ) -> Result<Self, PatternError> {
        if cases.as_ref().is_some_and(Vec::is_empty) {
            return Err(PatternError::EmptyPiecewiseCases);
        }
        Ok(Self(PatternKind::Piecewise {
            cases,
            otherwise: Box::new(otherwise),
        }))
    }

    /// Build a pattern matching a call.
    ///
    /// The call must name `function_name`, or any function for `None`. With
    /// `Some(arguments)`, the call must have exactly as many arguments, each
    /// matching the pattern at its position; `Some` of an empty list matches
    /// only calls without arguments. With `None`, the call may have any
    /// number of arguments, and they are not matched.
    #[must_use]
    pub fn call(function_name: Option<&str>, arguments: Option<Vec<Pattern>>) -> Self {
        Self(PatternKind::Call {
            function_name: function_name.map(Arc::from),
            arguments,
        })
    }

    /// Build a pattern matching the expressions for which `predicate`
    /// returns `Ok(true)`, capturing nothing.
    ///
    /// The predicate runs each time the pattern is matched. An error it
    /// returns ends the match and is returned from it unchanged.
    #[must_use]
    pub fn predicate<F>(predicate: F) -> Self
    where
        F: Fn(&Expression) -> Result<bool, CallbackError> + Send + Sync + 'static,
    {
        Self(PatternKind::Predicate(Predicate(Arc::new(predicate))))
    }

    /// Build a pattern trying `alternatives` in order, left to right.
    ///
    /// Each alternative is matched against the bindings given to this
    /// pattern, so a failed alternative leaves no captures behind, and the
    /// first alternative that matches decides the result. The choice is
    /// final: when an enclosing pattern later fails, the remaining
    /// alternatives are not tried.
    ///
    /// # Errors
    ///
    /// Returns [`PatternError::EmptyAlternatives`] if `alternatives` is
    /// empty.
    pub fn alternatives(alternatives: Vec<Pattern>) -> Result<Self, PatternError> {
        if alternatives.is_empty() {
            return Err(PatternError::EmptyAlternatives);
        }
        Ok(Self(PatternKind::Alternatives(alternatives)))
    }

    /// Match `expression` against this pattern, starting from `bindings`,
    /// and return the bindings after the match, or `None` if it does not
    /// match.
    ///
    /// A pattern that captures nothing returns `bindings` unchanged on a
    /// match. This is the step compound patterns take for each
    /// sub-pattern; [`match_pattern`] starts it from empty bindings.
    ///
    /// # Errors
    ///
    /// Returns the [`CallbackError`] of the first predicate that fails;
    /// matching stops there.
    pub fn match_under(
        &self,
        expression: &Expression,
        bindings: &MatchBindings,
    ) -> Result<Option<MatchBindings>, CallbackError> {
        self.match_from(expression, bindings.clone())
    }
}

impl Pattern {
    /// Match `expression` starting from the owned `bindings`, which a
    /// pattern capturing nothing hands back unchanged.
    fn match_from(
        &self,
        expression: &Expression,
        bindings: MatchBindings,
    ) -> Result<Option<MatchBindings>, CallbackError> {
        match &self.0 {
            PatternKind::Wildcard => Ok(Some(bindings)),
            PatternKind::Capture { name, sub_pattern } => Ok(sub_pattern
                .match_from(expression, bindings)?
                .and_then(|inner| inner.bind(name, expression))),
            PatternKind::Literal(value) => {
                Ok(match_literal(value.as_ref(), expression).then_some(bindings))
            }
            PatternKind::Identifier(identifier) => {
                let ExpressionKind::Identifier(candidate) = expression.kind() else {
                    return Ok(None);
                };
                let is_match = identifier.as_ref().is_none_or(|wanted| wanted == candidate);
                Ok(is_match.then_some(bindings))
            }
            PatternKind::Unary { operation, operand } => {
                let ExpressionKind::Unary(node) = expression.kind() else {
                    return Ok(None);
                };
                if operation.is_some_and(|wanted| wanted != node.operation()) {
                    return Ok(None);
                }
                operand.match_from(node.operand(), bindings)
            }
            PatternKind::Binary {
                operation,
                left,
                right,
            } => {
                let ExpressionKind::Binary(node) = expression.kind() else {
                    return Ok(None);
                };
                if operation.is_some_and(|wanted| wanted != node.operation()) {
                    return Ok(None);
                }
                match_sequence([(&**left, node.left()), (&**right, node.right())], bindings)
            }
            PatternKind::Piecewise { cases, otherwise } => {
                let ExpressionKind::Piecewise(node) = expression.kind() else {
                    return Ok(None);
                };
                let Some(case_patterns) = cases else {
                    return otherwise.match_from(node.otherwise(), bindings);
                };
                if case_patterns.len() != node.cases().len() {
                    return Ok(None);
                }
                let pairs = case_patterns
                    .iter()
                    .zip(node.cases())
                    .flat_map(|((condition_pattern, value_pattern), (condition, value))| {
                        [(condition_pattern, condition), (value_pattern, value)]
                    })
                    .chain([(&**otherwise, node.otherwise())]);
                match_sequence(pairs, bindings)
            }
            PatternKind::Call {
                function_name,
                arguments,
            } => {
                let ExpressionKind::Call(node) = expression.kind() else {
                    return Ok(None);
                };
                if function_name
                    .as_ref()
                    .is_some_and(|wanted| **wanted != *node.callee().name())
                {
                    return Ok(None);
                }
                let Some(argument_patterns) = arguments else {
                    return Ok(Some(bindings));
                };
                if argument_patterns.len() != node.arguments().len() {
                    return Ok(None);
                }
                match_sequence(argument_patterns.iter().zip(node.arguments()), bindings)
            }
            PatternKind::Predicate(Predicate(predicate)) => {
                Ok(predicate(expression)?.then_some(bindings))
            }
            PatternKind::Alternatives(alternatives) => {
                for alternative in alternatives {
                    let result = alternative.match_from(expression, bindings.clone())?;
                    if result.is_some() {
                        return Ok(result);
                    }
                }
                Ok(None)
            }
        }
    }
}

/// The captures of a successful match: capture names bound to the
/// expressions they matched.
///
/// Names are kept in binding order. A capture binds after its sub-pattern
/// has matched, so a capture's inner captures precede it, and captures in
/// sibling positions follow the matching order of the enclosing pattern.
/// Rebinding a name keeps its position.
///
/// Two bindings are equal when they bind the same set of names, whatever
/// the order, and each name to structurally equal expressions. Hashing
/// agrees with equality.
///
/// # Examples
///
/// ```
/// use fhy_core::expr::{Expression, LiteralValue};
/// use fhy_core::expr::pattern::MatchBindings;
///
/// let five = Expression::from(LiteralValue::from(5));
/// let bindings = MatchBindings::empty()
///     .try_bind("x", &five)
///     .expect("an unbound name binds");
/// assert!(bindings.try_bind("x", &Expression::from(LiteralValue::from(5))).is_some());
/// assert!(bindings.try_bind("x", &Expression::from(LiteralValue::from(6))).is_none());
/// ```
#[derive(Debug, Clone)]
pub struct MatchBindings {
    entries: Vec<(Arc<str>, Expression)>,
}

impl MatchBindings {
    /// Build bindings that bind no name.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Return whether no name is bound.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return the bound names in binding order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|(name, _)| &**name)
    }

    /// Return the expression bound to `name`, or `None` if `name` is not
    /// bound.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Expression> {
        self.entries
            .iter()
            .find(|(bound_name, _)| &**bound_name == name)
            .map(|(_, expression)| expression)
    }

    /// Return whether `name` is bound.
    #[must_use]
    pub fn has(&self, name: &str) -> bool {
        self.get(name).is_some()
    }

    /// Bind `name` to `expression`, or confirm an existing binding.
    ///
    /// When `name` is not bound, return these bindings with `name` bound to
    /// a handle to `expression`, placed last. When `name` is bound to an
    /// expression structurally equal to `expression`, return these bindings
    /// unchanged: the first-bound expression is kept. Otherwise return
    /// `None`, and the match in progress fails. The receiver is never
    /// changed.
    #[must_use]
    pub fn try_bind(&self, name: &str, expression: &Expression) -> Option<MatchBindings> {
        self.clone().bind(&Arc::from(name), expression)
    }
}

impl MatchBindings {
    /// Bind `name` to `expression` in these bindings, or confirm the
    /// existing binding; `None` when `name` is bound to an unequal
    /// expression. A new binding shares `name`.
    fn bind(self, name: &Arc<str>, expression: &Expression) -> Option<Self> {
        match self.get(name) {
            Some(bound) => (bound == expression).then_some(self),
            None => Some(self.push(Arc::clone(name), expression)),
        }
    }

    /// Return these bindings with `name` bound to `expression`, placed last.
    fn push(mut self, name: Arc<str>, expression: &Expression) -> Self {
        self.entries.push((name, expression.clone()));
        self
    }
}

impl PartialEq for MatchBindings {
    /// Compare the sets of bound names, then each name's expressions
    /// structurally.
    fn eq(&self, other: &Self) -> bool {
        self.entries.len() == other.entries.len()
            && self
                .entries
                .iter()
                .all(|(name, expression)| other.get(name) == Some(expression))
    }
}

impl Eq for MatchBindings {}

impl Hash for MatchBindings {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Bound names are distinct, so equal bindings hold equal entries in
        // some order; a commutative sum of per-entry hashes ignores it.
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

/// A pattern that could not be built because it could never match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PatternError {
    /// A capture pattern was given an empty name.
    ///
    /// Displays as `a capture pattern needs a non-empty name`.
    EmptyCaptureName,
    /// A piecewise pattern was given an empty list of cases.
    ///
    /// Displays as `a piecewise pattern with cases needs at least one case`.
    EmptyPiecewiseCases,
    /// An alternatives pattern was given no alternative.
    ///
    /// Displays as `an alternatives pattern needs at least one alternative`.
    EmptyAlternatives,
}

impl fmt::Display for PatternError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::EmptyCaptureName => "a capture pattern needs a non-empty name",
            Self::EmptyPiecewiseCases => "a piecewise pattern with cases needs at least one case",
            Self::EmptyAlternatives => "an alternatives pattern needs at least one alternative",
        };
        f.write_str(message)
    }
}

impl Error for PatternError {}

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
/// use fhy_core::expr::pattern::{CallbackError, Pattern, match_pattern};
///
/// let refusing = Pattern::predicate(|_| Err(CallbackError::from("no verdict")));
///
/// let result = match_pattern(&refusing, &Expression::from(LiteralValue::from(1)));
///
/// let error = result.expect_err("the predicate fails");
/// assert_eq!(error.to_string(), "no verdict");
/// ```
pub type CallbackError = Box<dyn Error + Send + Sync + 'static>;

/// Match `expression` against `pattern` from empty bindings, at the root
/// only, and return the captures, or `None` if it does not match.
///
/// # Errors
///
/// Returns the [`CallbackError`] of the first predicate in `pattern` that
/// fails.
pub fn match_pattern(
    pattern: &Pattern,
    expression: &Expression,
) -> Result<Option<MatchBindings>, CallbackError> {
    pattern.match_from(expression, MatchBindings::empty())
}

/// Return whether `pattern` matches `expression` at the root.
///
/// # Errors
///
/// Returns the [`CallbackError`] of the first predicate in `pattern` that
/// fails.
pub fn does_pattern_match(
    pattern: &Pattern,
    expression: &Expression,
) -> Result<bool, CallbackError> {
    Ok(match_pattern(pattern, expression)?.is_some())
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Pattern>();
    assert_send_sync::<MatchBindings>();
};
