//! Rewrite rules and the bottom-up rewrite walk.
//!
//! A [`RewriteRule`] pairs a [`Pattern`] with a rewrite that builds a
//! replacement from the match's [`MatchBindings`], optionally behind a guard
//! and under a name. [`apply_rewrite_rule`] tries one rule at the root of an
//! expression; [`apply_rewrite_rules`] walks a whole tree bottom-up once,
//! trying a list of rules at every node, and reports the rewritten tree,
//! whether it differs from the input, and which rules fired.
//! [`RewriteRuleApplier`](crate::expr::passes::RewriteRuleApplier) is the
//! same walk as a compiler pass.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use super::super::error::{PiecewiseError, RebuildError};
use super::super::node::Expression;
use super::matching::{CallbackError, MatchBindings, Pattern, match_pattern};
use crate::tree::{NodeHandle, NodeIdentity, RewriteTreeError, Rewriter, rewrite_tree};

/// A rewrite: the replacement built from a match's bindings.
type RewriteFn = Arc<dyn Fn(&MatchBindings) -> Result<Expression, CallbackError> + Send + Sync>;

/// A guard: whether a rule may fire on a match's bindings.
type GuardFn = Arc<dyn Fn(&MatchBindings) -> Result<bool, CallbackError> + Send + Sync>;

/// Return the position of the child that `error` refuses, or `None` when
/// the error names no single child.
fn find_refused_child_index(error: &RebuildError) -> Option<usize> {
    let RebuildError::Piecewise(PiecewiseError::NonBooleanConditionLiteral { case_index }) = error
    else {
        return None;
    };
    case_index.checked_mul(2)
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
#[derive(Debug)]
struct RuleApplier<'r> {
    rules: &'r [RewriteRule],
    fired: Vec<FiredRule>,
    /// Each replacement a rule returned, by its identity, with the position
    /// of that rule. Holding the replacement keeps its identity unique.
    replacements: HashMap<NodeIdentity, (Expression, usize)>,
}

impl<'r> RuleApplier<'r> {
    /// Create the applier of `rules`.
    fn new(rules: &'r [RewriteRule]) -> Self {
        Self {
            rules,
            fired: Vec::new(),
            replacements: HashMap::new(),
        }
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
        let refused_child = find_refused_child_index(error)
            .and_then(|index| Some((node.children().nth(index)?, children.get(index)?)))
            .filter(|(original, rewritten)| !Expression::ptr_eq(original, rewritten));
        refused_child
            .or_else(|| find_last_replaced_child(node.children(), children.iter()))
            .and_then(|(original, rewritten)| self.find_responsible_rule(original, rewritten))
            .or_else(|| self.fired.last().map(FiredRule::rule_index))
            .expect("a node is rebuilt only after a rule fired")
    }
}

impl Rewriter<Expression> for RuleApplier<'_> {
    type Error = RewriteError;

    fn rewrite(
        &mut self,
        node: &Expression,
        _cx: &mut (),
    ) -> Result<Option<Expression>, RewriteError> {
        for (rule_index, rule) in self.rules.iter().enumerate() {
            let replacement =
                apply_rewrite_rule(rule, node).map_err(|source| RewriteError::Callback {
                    rule_index,
                    rule_name: rule.name().map(str::to_owned),
                    source,
                })?;
            if let Some(expression) = replacement {
                self.fired.push(FiredRule {
                    rule_index,
                    name: rule.name.clone(),
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
pub(in crate::expr) fn run_rewrite_rules(
    expression: &Expression,
    rules: &[RewriteRule],
) -> RuleRun {
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
                rule_name: rules[rule_index].name().map(str::to_owned),
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
/// the root and its guard, if any, returns `Ok(true)` for the match's
/// bindings; the rewrite then builds the replacement from those bindings.
/// Cloning shares the callbacks. Rules have no equality: callbacks cannot
/// be compared.
///
/// # Examples
///
/// ```
/// use fhy_core::identifier::Identifier;
/// use fhy_core::expr::{BinaryOperation, Expression, LiteralValue};
/// use fhy_core::expr::pattern::{
///     CallbackError, Pattern, RewriteRule, apply_rewrite_rule,
/// };
///
/// // `x + 0 -> x`
/// let rule = RewriteRule::new(
///     Pattern::binary(
///         Some(BinaryOperation::Add),
///         Pattern::capture("x", Pattern::wildcard())?,
///         Pattern::literal(Some(LiteralValue::from(0))),
///     ),
///     |bindings| {
///         bindings
///             .get("x")
///             .cloned()
///             .ok_or_else(|| CallbackError::new("`x` is unbound"))
///     },
/// )
/// .with_name("x + 0 -> x");
/// let a = Expression::from(Identifier::new("a"));
///
/// let rewritten = apply_rewrite_rule(&rule, &(&a + 0))?;
///
/// assert!(rewritten.is_some_and(|result| Expression::ptr_eq(&result, &a)));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone)]
pub struct RewriteRule {
    pattern: Pattern,
    rewrite: RewriteFn,
    guard: Option<GuardFn>,
    name: Option<Arc<str>>,
}

impl RewriteRule {
    /// Build an unguarded, unnamed rule rewriting what `pattern` matches
    /// with `rewrite`.
    #[must_use]
    pub fn new<F>(pattern: Pattern, rewrite: F) -> Self
    where
        F: Fn(&MatchBindings) -> Result<Expression, CallbackError> + Send + Sync + 'static,
    {
        Self {
            pattern,
            rewrite: Arc::new(rewrite),
            guard: None,
            name: None,
        }
    }

    /// Return this rule guarded by `guard`, replacing any earlier guard.
    ///
    /// The guard runs after the pattern has matched, on the match's
    /// bindings; the rule fires only when it returns `Ok(true)`.
    #[must_use]
    pub fn with_guard<G>(self, guard: G) -> Self
    where
        G: Fn(&MatchBindings) -> Result<bool, CallbackError> + Send + Sync + 'static,
    {
        Self {
            guard: Some(Arc::new(guard)),
            ..self
        }
    }

    /// Return this rule named `name`, replacing any earlier name.
    #[must_use]
    pub fn with_name(self, name: &str) -> Self {
        Self {
            name: Some(Arc::from(name)),
            ..self
        }
    }

    /// Return the rule's name, or `None` for an unnamed rule.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

impl fmt::Debug for RewriteRule {
    /// Show the pattern, the name, and whether a guard is set; the
    /// callbacks themselves are opaque.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RewriteRule")
            .field("pattern", &self.pattern)
            .field("name", &self.name)
            .field("has_guard", &self.guard.is_some())
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
    /// A tree in which no rule fired is unchanged. A rule that fires and
    /// returns the node it matched, at the root or below it, leaves that
    /// node unchanged too, so no ancestor is rebuilt on its account.
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
#[derive(Debug)]
#[non_exhaustive]
pub enum RewriteError {
    /// A predicate in a rule's pattern, the rule's guard, or its rewrite
    /// failed.
    ///
    /// Displays as `rewrite rule {rule_index} failed`, or as
    /// `rewrite rule {rule_index} ({rule_name}) failed` for a named rule;
    /// the callback's error is the [`source`](Error::source).
    Callback {
        /// The position of the failing rule in the rule list.
        rule_index: usize,
        /// The name of the failing rule, or `None` for an unnamed rule.
        rule_name: Option<String>,
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
    Rebuild {
        /// The position of the responsible rule in the rule list.
        rule_index: usize,
        /// The name of the responsible rule, or `None` for an unnamed rule.
        rule_name: Option<String>,
        /// The rebuild error.
        source: RebuildError,
    },
}

impl fmt::Display for RewriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Callback {
                rule_index,
                rule_name: None,
                ..
            } => write!(f, "rewrite rule {rule_index} failed"),
            Self::Callback {
                rule_index,
                rule_name: Some(name),
                ..
            } => write!(f, "rewrite rule {rule_index} ({name}) failed"),
            Self::Rebuild {
                rule_index,
                rule_name: None,
                ..
            } => write!(
                f,
                "rebuilding a node after rewrite rule {rule_index} failed"
            ),
            Self::Rebuild {
                rule_index,
                rule_name: Some(name),
                ..
            } => write!(
                f,
                "rebuilding a node after rewrite rule {rule_index} ({name}) failed"
            ),
        }
    }
}

impl Error for RewriteError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Callback { source, .. } => Some(source),
            Self::Rebuild { source, .. } => Some(source),
        }
    }
}

/// Try `rule` once at the root of `expression` and return the rewrite, or
/// `None` if the pattern does not match or the guard returns `Ok(false)`.
///
/// The pattern is matched first, the guard runs only on a match, and the
/// rewrite only when the guard allows it. Subexpressions are never tried.
///
/// # Errors
///
/// Returns the [`CallbackError`] of a failing predicate in the pattern, of
/// the guard, or of the rewrite, unchanged.
pub fn apply_rewrite_rule(
    rule: &RewriteRule,
    expression: &Expression,
) -> Result<Option<Expression>, CallbackError> {
    let Some(bindings) = match_pattern(&rule.pattern, expression)? else {
        return Ok(None);
    };
    if let Some(guard) = &rule.guard {
        if !guard(&bindings)? {
            return Ok(None);
        }
    }
    (rule.rewrite)(&bindings).map(Some)
}

/// Rewrite `expression` bottom-up in one pass, trying `rules` in order at
/// every node.
///
/// Every node's children are rewritten first, in [`Expression::children`]
/// order. A node with a rewritten child is rebuilt from the rewritten
/// children, keeping the handles of the children no rule touched. The rules
/// are then tried, in order, on the rebuilt node (or on the node itself when
/// no child changed), and the first that fires replaces it. A replacement is
/// not rewritten again in the same pass; a caller wanting a fixpoint repeats
/// the call until the outcome is unchanged. A node that occurs in several
/// places is rewritten once and its result reused at every occurrence, so a
/// tree sharing its subtrees costs time linear in its distinct nodes.
///
/// When no rule fires anywhere, or every rule that fires returns the node it
/// matched, the output is a handle to `expression` itself. See [`RewriteOutcome::is_changed`] for the exact meaning of a
/// change.
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
pub fn apply_rewrite_rules(
    expression: &Expression,
    rules: &[RewriteRule],
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
