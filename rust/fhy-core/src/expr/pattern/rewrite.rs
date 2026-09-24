//! Rewrite rules and the bottom-up rewrite walk.
//!
//! A [`Rule`] is a rewrite tried at the root of one expression. A
//! [`RewriteRule`] is the rule built from a [`Pattern`] and a rewrite that
//! builds a replacement from the match's [`MatchBindings`], optionally behind
//! guards and under a name; any other type can implement [`Rule`] itself.
//! [`RewriteRule::apply`] tries one rule at the root of an expression;
//! [`apply_rewrite_rules`] walks a whole tree bottom-up once, trying a list
//! of rules at every node, and reports the rewritten tree, whether it
//! differs from the input, and which rules fired.
//! [`RewriteRuleApplier`](crate::expr::passes::RewriteRuleApplier) is the
//! same walk as a compiler pass.

use std::cell::OnceCell;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use super::super::error::RebuildError;
use super::super::node::Expression;
use super::matching::{CallbackError, MatchBindings, Pattern};
use crate::tree::{
    BuildIdentityHasher, NodeHandle, NodeIdentity, RewriteTreeError, Rewriter, rewrite_tree,
};

/// A rewrite: the replacement built from a match's bindings, or `None` to
/// decline.
type RewriteFn =
    Arc<dyn Fn(&MatchBindings) -> Result<Option<Expression>, CallbackError> + Send + Sync>;

/// A guard: whether a rule may fire on a match's bindings.
type GuardFn = Arc<dyn Fn(&MatchBindings) -> Result<bool, CallbackError> + Send + Sync>;

/// A rewrite tried at the root of one expression.
///
/// [`apply_rewrite_rules`] and
/// [`RewriteRuleApplier`](crate::expr::passes::RewriteRuleApplier) take a
/// list of any one rule type; a list of `Box<dyn Rule>` mixes rule types.
/// [`RewriteRule`] is the rule built from a pattern; a rule that borrows
/// context, or that needs no pattern, implements the trait on its own type.
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
///
/// use fhy_core::identifier::Identifier;
/// use fhy_core::expr::{Expression, ExpressionKind, LiteralValue};
/// use fhy_core::expr::pattern::{CallbackError, Rule, apply_rewrite_rules};
///
/// /// Replaces each known identifier with its value.
/// struct Substitute<'v>(&'v HashMap<Identifier, i64>);
///
/// impl Rule for Substitute<'_> {
///     fn apply(&self, expression: &Expression) -> Result<Option<Expression>, CallbackError> {
///         let ExpressionKind::Identifier(identifier) = expression.kind() else {
///             return Ok(None);
///         };
///         Ok(self.0.get(identifier).map(|value| Expression::from(LiteralValue::from(*value))))
///     }
/// }
///
/// let a = Identifier::new("a");
/// let values = HashMap::from([(a.clone(), 2)]);
///
/// let outcome = apply_rewrite_rules(&(Expression::from(a) + 1), &[Substitute(&values)])?;
///
/// assert_eq!(outcome.output(), &(Expression::from(LiteralValue::from(2)) + 1));
/// # Ok::<(), fhy_core::expr::pattern::RewriteError>(())
/// ```
pub trait Rule {
    /// Return the replacement for `expression`, or `Ok(None)` to decline.
    ///
    /// Returning a handle to `expression` itself ([`Expression::ptr_eq`])
    /// means the same as declining: the walk records no firing and tries
    /// the next rule.
    ///
    /// # Errors
    ///
    /// Returns an error to stop the walk trying the rule;
    /// [`apply_rewrite_rules`] reports it as [`RewriteError::Callback`].
    fn apply(&self, expression: &Expression) -> Result<Option<Expression>, CallbackError>;

    /// Return the rule's name, used in firings and errors, or `None` for an
    /// unnamed rule. By default, `None`.
    fn name(&self) -> Option<&str> {
        None
    }
}

impl<R: Rule + ?Sized> Rule for &R {
    fn apply(&self, expression: &Expression) -> Result<Option<Expression>, CallbackError> {
        (**self).apply(expression)
    }

    fn name(&self) -> Option<&str> {
        (**self).name()
    }
}

impl<R: Rule + ?Sized> Rule for Box<R> {
    fn apply(&self, expression: &Expression) -> Result<Option<Expression>, CallbackError> {
        (**self).apply(expression)
    }

    fn name(&self) -> Option<&str> {
        (**self).name()
    }
}

impl<R: Rule + ?Sized> Rule for Arc<R> {
    fn apply(&self, expression: &Expression) -> Result<Option<Expression>, CallbackError> {
        (**self).apply(expression)
    }

    fn name(&self) -> Option<&str> {
        (**self).name()
    }
}

/// Return the last child in `rewritten` that is not the node in the same
/// position of `originals`, paired with that original.
fn find_last_replaced_child<'e>(
    originals: impl Iterator<Item = &'e Expression>,
    rewritten: impl Iterator<Item = &'e Expression>,
) -> Option<(&'e Expression, &'e Expression)> {
    originals
        .zip(rewritten)
        .filter(|(original, rewritten)| !Expression::ptr_eq(original, rewritten))
        .last()
}

/// The rewriter behind [`apply_rewrite_rules`]: tries the rules in order at
/// each node and records every firing.
struct RuleApplier<'r, R> {
    rules: &'r [R],
    /// The name of each rule, converted on its first firing or failure.
    names: Vec<OnceCell<Option<Arc<str>>>>,
    fired: Vec<FiredRule>,
    /// Each replacement a rule returned, by its identity, with the position
    /// of that rule; read only to blame a refused rebuild. Holding the
    /// replacement keeps its identity unique for the whole walk.
    replacements: HashMap<NodeIdentity, (Expression, usize), BuildIdentityHasher>,
}

impl<'r, R: Rule> RuleApplier<'r, R> {
    /// Create the applier of `rules`.
    fn new(rules: &'r [R]) -> Self {
        Self {
            rules,
            names: rules.iter().map(|_| OnceCell::new()).collect(),
            fired: Vec::new(),
            replacements: HashMap::default(),
        }
    }

    /// Return the name of the rule at `rule_index`, shared by every firing
    /// and error of the rule.
    fn rule_name(&self, rule_index: usize) -> Option<Arc<str>> {
        let rule = &self.rules[rule_index];
        self.names[rule_index]
            .get_or_init(|| rule.name().map(Arc::from))
            .clone()
    }

    /// Return the position of the rule responsible for `rewritten`, which
    /// took the place of `original`: the rule that returned it, or, for a
    /// node rebuilt around rewritten children, the rule responsible for its
    /// last rewritten child.
    fn find_responsible_rule<'e>(
        &self,
        mut original: &'e Expression,
        mut rewritten: &'e Expression,
    ) -> Option<usize> {
        loop {
            if let Some((_, rule_index)) = self.replacements.get(&rewritten.identity()) {
                return Some(*rule_index);
            }
            (original, rewritten) =
                find_last_replaced_child(original.children(), rewritten.children())?;
        }
    }

    /// Return the position of the rule to blame for `node` refusing to be
    /// rebuilt from `children` with `error`: the rule responsible for the
    /// refused child, else for the last rewritten child, else the rule that
    /// fired last.
    ///
    /// # Panics
    ///
    /// Panics if no rule fired, which a refused rebuild rules out: a node is
    /// rebuilt only around a child some rule rewrote.
    fn find_blamed_rule(
        &self,
        node: &Expression,
        children: &[Expression],
        error: &RebuildError,
    ) -> usize {
        let refused_child = Expression::refused_child_index(error)
            .and_then(|index| Some((node.children().nth(index)?, children.get(index)?)))
            .filter(|(original, rewritten)| !Expression::ptr_eq(original, rewritten));
        refused_child
            .or_else(|| find_last_replaced_child(node.children(), children.iter()))
            .and_then(|(original, rewritten)| self.find_responsible_rule(original, rewritten))
            .or_else(|| self.fired.last().map(FiredRule::rule_index))
            .expect("a node is rebuilt only after a rule fired")
    }
}

impl<R: Rule> Rewriter<Expression> for RuleApplier<'_, R> {
    type Error = RewriteError;

    fn rewrite(
        &mut self,
        node: &Expression,
        _cx: &mut (),
    ) -> Result<Option<Expression>, RewriteError> {
        for (rule_index, rule) in self.rules.iter().enumerate() {
            let replacement = rule.apply(node).map_err(|source| RewriteError::Callback {
                rule_index,
                rule_name: self.rule_name(rule_index),
                source,
            })?;
            if let Some(expression) =
                replacement.filter(|expression| !Expression::ptr_eq(expression, node))
            {
                self.fired.push(FiredRule {
                    rule_index,
                    name: self.rule_name(rule_index),
                });
                self.replacements
                    .insert(expression.identity(), (expression.clone(), rule_index));
                return Ok(Some(expression));
            }
        }
        Ok(None)
    }
}

/// A finished rewrite walk: the rewritten tree or the failure that ended
/// the walk, and the firings up to its end.
pub(in crate::expr) struct RuleRun {
    pub(in crate::expr) output: Result<Expression, RewriteError>,
    pub(in crate::expr) fired: Vec<FiredRule>,
}

/// Rewrite `expression` bottom-up once with `rules`.
pub(in crate::expr) fn run_rewrite_rules<R: Rule>(expression: &Expression, rules: &[R]) -> RuleRun {
    let mut applier = RuleApplier::new(rules);
    let output = rewrite_tree(&mut applier, expression, &mut ()).map_err(|error| match error {
        RewriteTreeError::Rewrite(error) => error,
        RewriteTreeError::Rebuild {
            node,
            children,
            source,
        } => {
            let rule_index = applier.find_blamed_rule(&node, &children, &source);
            RewriteError::Rebuild {
                rule_index,
                rule_name: applier.rule_name(rule_index),
                source,
            }
        }
    });
    RuleRun {
        output,
        fired: applier.fired,
    }
}

/// A pattern paired with a rewrite of what it matches, optionally guarded
/// and named.
///
/// A rule fires on an expression when its pattern matches the expression at
/// the root, every guard returns `Ok(true)` for the match's bindings, and
/// the rewrite returns a replacement. A rule built with
/// [`new`](Self::new) always returns one; a rule built with
/// [`new_partial`](Self::new_partial) may decline. Cloning shares the
/// callbacks. Rules have no equality: callbacks cannot be compared.
///
/// # Examples
///
/// ```
/// use fhy_core::identifier::Identifier;
/// use fhy_core::expr::{BinaryOperation, Expression};
/// use fhy_core::expr::pattern::{Capture, Pattern, RewriteRule};
///
/// // `x + 0 -> x`
/// let x = Capture::new("x");
/// let pattern = Pattern::binary(BinaryOperation::Add, Pattern::capture(&x), Pattern::literal(0));
/// let rule = RewriteRule::new(pattern, move |bindings| Ok(bindings[&x].clone()))
///     .with_name("x + 0 -> x");
/// let a = Expression::from(Identifier::new("a"));
///
/// let rewritten = rule.apply(&(&a + 0))?;
///
/// assert!(rewritten.is_some_and(|result| Expression::ptr_eq(&result, &a)));
/// # Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
/// ```
#[derive(Clone)]
pub struct RewriteRule {
    pattern: Pattern,
    guards: Vec<GuardFn>,
    rewrite: RewriteFn,
    name: Option<Arc<str>>,
}

impl RewriteRule {
    /// Build an unguarded, unnamed rule rewriting what `pattern` matches
    /// with `rewrite`, which always returns a replacement.
    #[must_use]
    pub fn new<F>(pattern: Pattern, rewrite: F) -> Self
    where
        F: Fn(&MatchBindings) -> Result<Expression, CallbackError> + Send + Sync + 'static,
    {
        Self::new_partial(pattern, move |bindings| rewrite(bindings).map(Some))
    }

    /// Build an unguarded, unnamed rule rewriting what `pattern` matches
    /// with `rewrite`, which returns `Ok(None)` to decline.
    #[must_use]
    pub fn new_partial<F>(pattern: Pattern, rewrite: F) -> Self
    where
        F: Fn(&MatchBindings) -> Result<Option<Expression>, CallbackError> + Send + Sync + 'static,
    {
        Self {
            pattern,
            guards: Vec::new(),
            rewrite: Arc::new(rewrite),
            name: None,
        }
    }

    /// Return this rule with `guard` added after its other guards.
    ///
    /// The guards run after the pattern has matched, on the match's
    /// bindings, in the order they were added; the rule fires only when
    /// every guard returns `Ok(true)`, and the first that does not stops
    /// the rest.
    #[must_use]
    pub fn with_guard<G>(mut self, guard: G) -> Self
    where
        G: Fn(&MatchBindings) -> Result<bool, CallbackError> + Send + Sync + 'static,
    {
        self.guards.push(Arc::new(guard));
        self
    }

    /// Return this rule named `name`, replacing any earlier name.
    #[must_use]
    pub fn with_name(self, name: impl Into<Arc<str>>) -> Self {
        Self {
            name: Some(name.into()),
            ..self
        }
    }

    /// Return the rule's name, or `None` for an unnamed rule.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Try this rule once at the root of `expression` and return the
    /// replacement, or `None` if the pattern does not match, a guard
    /// refuses, or the rewrite declines or returns `expression` itself
    /// ([`Expression::ptr_eq`]).
    ///
    /// The pattern is matched first, the guards run in order only on a
    /// match, and the rewrite only when every guard allows it.
    /// Subexpressions are never tried.
    ///
    /// # Errors
    ///
    /// Returns the [`CallbackError`] of a failing predicate in the pattern,
    /// of a guard, or of the rewrite, unchanged.
    pub fn apply(&self, expression: &Expression) -> Result<Option<Expression>, CallbackError> {
        let Some(bindings) = self.pattern.matches(expression)? else {
            return Ok(None);
        };
        for guard in &self.guards {
            if !guard(&bindings)? {
                return Ok(None);
            }
        }
        let replacement = (self.rewrite)(&bindings)?;
        Ok(replacement.filter(|replacement| !Expression::ptr_eq(replacement, expression)))
    }
}

impl Rule for RewriteRule {
    fn apply(&self, expression: &Expression) -> Result<Option<Expression>, CallbackError> {
        RewriteRule::apply(self, expression)
    }

    fn name(&self) -> Option<&str> {
        RewriteRule::name(self)
    }
}

impl fmt::Debug for RewriteRule {
    /// Show the pattern and the name; the callbacks are opaque.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RewriteRule")
            .field("pattern", &self.pattern)
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

/// One firing of a rule during [`apply_rewrite_rules`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FiredRule {
    rule_index: usize,
    name: Option<Arc<str>>,
}

impl FiredRule {
    /// Return the position of the rule that fired in the rule list.
    #[must_use]
    pub fn rule_index(&self) -> usize {
        self.rule_index
    }

    /// Return the name of the rule that fired, or `None` for an unnamed
    /// rule.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

/// The result of [`apply_rewrite_rules`]: the rewritten tree, whether it
/// differs from the input, and the rules that fired.
#[must_use = "the outcome holds the rewritten tree"]
#[derive(Debug, Clone)]
pub struct RewriteOutcome {
    output: Expression,
    changed: bool,
    fired: Vec<FiredRule>,
}

impl RewriteOutcome {
    /// Return the rewritten tree.
    #[must_use]
    pub fn output(&self) -> &Expression {
        &self.output
    }

    /// Return the rewritten tree, consuming the outcome.
    #[must_use]
    pub fn into_output(self) -> Expression {
        self.output
    }

    /// Return whether the rewritten tree is a different node from the input:
    /// exactly when [`output`](Self::output) and the input are not
    /// [`Expression::ptr_eq`].
    ///
    /// A tree in which no rule fired is unchanged, and a rule that returns
    /// the node it was tried on does not fire, so no firing implies no
    /// change. A rule that builds a fresh node equal to the one it matched
    /// fires and counts as a change.
    #[must_use]
    pub fn is_changed(&self) -> bool {
        self.changed
    }

    /// Return every firing in walk order: children before their parent, and
    /// children in [`Expression::children`] order. A node that occurs in
    /// several places is rewritten once, so a firing on it is recorded once,
    /// at its first occurrence.
    #[must_use]
    pub fn fired(&self) -> &[FiredRule] {
        &self.fired
    }
}

/// A rewrite walk that failed.
///
/// Both variants carry the position of the rule responsible in the rule
/// list and its name, read with [`rule_index`](Self::rule_index) and
/// [`rule_name`](Self::rule_name).
#[derive(Debug)]
#[non_exhaustive]
pub enum RewriteError {
    /// A predicate in a rule's pattern, the rule's guard, or its rewrite
    /// failed.
    ///
    /// Displays as `rewrite rule {rule_index} failed`, or as
    /// `rewrite rule {rule_index} ({rule_name}) failed` for a named rule;
    /// the error the callback returned is the [`source`](Error::source).
    #[non_exhaustive]
    Callback {
        /// The position of the failing rule in the rule list.
        rule_index: usize,
        /// The name of the failing rule, or `None` for an unnamed rule.
        rule_name: Option<Arc<str>>,
        /// The callback's error.
        source: CallbackError,
    },
    /// A node could not be rebuilt from its rewritten children, such as a
    /// piecewise whose case condition a rule rewrote to a literal other than
    /// a Boolean.
    ///
    /// The rule named is the one that rewrote the refused child, or, when
    /// the rebuild error names no single child, the rule responsible for the
    /// last rewritten child. Displays as `rebuilding a node after rewrite
    /// rule {rule_index} failed`, or as `rebuilding a node after rewrite
    /// rule {rule_index} ({rule_name}) failed` for a named rule; the rebuild
    /// error is the [`source`](Error::source).
    #[non_exhaustive]
    Rebuild {
        /// The position of the responsible rule in the rule list.
        rule_index: usize,
        /// The name of the responsible rule, or `None` for an unnamed rule.
        rule_name: Option<Arc<str>>,
        /// The rebuild error.
        source: RebuildError,
    },
}

impl RewriteError {
    /// Return the position of the responsible rule in the rule list.
    #[must_use]
    pub fn rule_index(&self) -> usize {
        match self {
            Self::Callback { rule_index, .. } | Self::Rebuild { rule_index, .. } => *rule_index,
        }
    }

    /// Return the name of the responsible rule, or `None` for an unnamed
    /// rule.
    #[must_use]
    pub fn rule_name(&self) -> Option<&str> {
        match self {
            Self::Callback { rule_name, .. } | Self::Rebuild { rule_name, .. } => {
                rule_name.as_deref()
            }
        }
    }
}

impl fmt::Display for RewriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rule_index = self.rule_index();
        match (self, self.rule_name()) {
            (Self::Callback { .. }, None) => write!(f, "rewrite rule {rule_index} failed"),
            (Self::Callback { .. }, Some(name)) => {
                write!(f, "rewrite rule {rule_index} ({name}) failed")
            }
            (Self::Rebuild { .. }, None) => write!(
                f,
                "rebuilding a node after rewrite rule {rule_index} failed"
            ),
            (Self::Rebuild { .. }, Some(name)) => write!(
                f,
                "rebuilding a node after rewrite rule {rule_index} ({name}) failed"
            ),
        }
    }
}

impl Error for RewriteError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Callback { source, .. } => Some(&**source),
            Self::Rebuild { source, .. } => Some(source),
        }
    }
}

/// Rewrite `expression` bottom-up in one pass, trying `rules` in order at
/// every node.
///
/// Every node's children are rewritten first, in [`Expression::children`]
/// order. A node with a rewritten child is rebuilt from the rewritten
/// children, keeping the handles of the children no rule touched. The rules
/// are then tried, in order, on the rebuilt node (or on the node itself when
/// no child changed), and the first that fires replaces it. A rule fires
/// when it returns a replacement other than the node it was tried on
/// itself; a rule that declines or returns that node is not recorded, and
/// the next rule is tried. A replacement is not rewritten again in the same
/// pass. A node that occurs in several places is rewritten once and its
/// result reused at every occurrence, so a tree sharing its subtrees costs
/// time linear in its distinct nodes.
///
/// When no rule fires anywhere, the output is a handle to `expression`
/// itself; see [`RewriteOutcome::is_changed`] for the exact meaning of a
/// change. A caller wanting a fixpoint repeats the call until the outcome
/// is unchanged. That loop terminates for rules that return their input or
/// a subterm of it, but not for a rule that keeps building a fresh node
/// equal to the one it matched, which counts as a change every time.
///
/// The walk keeps its own work stack, so a tree of any depth rewrites within
/// the default 2 MiB thread stack when the rules' patterns are shallow.
///
/// # Errors
///
/// Returns [`RewriteError::Callback`] for the first callback that fails,
/// naming its rule, and [`RewriteError::Rebuild`] if a node cannot be
/// rebuilt from its rewritten children, naming the rule whose rewrite it
/// refuses. The walk stops at the first error.
pub fn apply_rewrite_rules<R: Rule>(
    expression: &Expression,
    rules: &[R],
) -> Result<RewriteOutcome, RewriteError> {
    let RuleRun { output, fired } = run_rewrite_rules(expression, rules);
    let output = output?;
    let changed = !Expression::ptr_eq(&output, expression);
    Ok(RewriteOutcome {
        output,
        changed,
        fired,
    })
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<RewriteRule>();
    assert_send_sync::<RewriteOutcome>();
    assert_send_sync::<RewriteError>();
};
