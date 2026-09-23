//! Rewrite rules and the bottom-up rewrite walk.
//!
//! A [`RewriteRule`] pairs a [`Pattern`] with a rewrite that builds a
//! replacement from the match's [`MatchBindings`], optionally behind a guard
//! and under a name. [`apply_rewrite_rule`] tries one rule at the root of an
//! expression; [`apply_rewrite_rules`] walks a whole tree bottom-up once,
//! trying a list of rules at every node, and reports the rewritten tree,
//! whether it differs from the input, and which rules fired.

use std::error::Error;
use std::fmt;
use std::sync::Arc;

use super::super::error::ExpressionBuildError;
use super::super::node::Expression;
use super::core::{CallbackError, MatchBindings, Pattern, match_pattern};

/// A rewrite: the replacement built from a match's bindings.
type RewriteFn = Arc<dyn Fn(&MatchBindings) -> Result<Expression, CallbackError> + Send + Sync>;

/// A guard: whether a rule may fire on a match's bindings.
type GuardFn = Arc<dyn Fn(&MatchBindings) -> Result<bool, CallbackError> + Send + Sync>;

/// One node of the rewrite walk: the node, its children, and the rewrites
/// of the children visited so far (`None` for a child left as it was).
struct WalkFrame {
    node: Expression,
    children: Vec<Expression>,
    rewritten_children: Vec<Option<Expression>>,
}

impl WalkFrame {
    /// Start visiting `node`.
    fn new(node: &Expression) -> Self {
        let children: Vec<Expression> = node.children().cloned().collect();
        Self {
            node: node.clone(),
            rewritten_children: Vec::with_capacity(children.len()),
            children,
        }
    }

    /// Return the next child to visit, or `None` once every child is
    /// rewritten.
    fn next_child(&self) -> Option<Expression> {
        self.children.get(self.rewritten_children.len()).cloned()
    }
}

/// Finish a node whose children are all rewritten: rebuild it if a child
/// changed, then try `rules` on it in order, recording a firing in
/// `fired`. Return the replacement, or `None` when the node stays as it
/// was.
fn finish_node(
    frame: WalkFrame,
    rules: &[RewriteRule],
    fired: &mut Vec<FiredRule>,
) -> Result<Option<Expression>, RewriteError> {
    let WalkFrame {
        node,
        children,
        rewritten_children,
    } = frame;
    let rebuilt = if rewritten_children.iter().any(Option::is_some) {
        let merged = rewritten_children
            .into_iter()
            .zip(children)
            .map(|(rewritten, original)| rewritten.unwrap_or(original))
            .collect();
        Some(
            node.rebuild_with_children(merged)
                .map_err(RewriteError::Rebuild)?,
        )
    } else {
        None
    };
    let visited = rebuilt.as_ref().unwrap_or(&node);
    for (rule_index, rule) in rules.iter().enumerate() {
        let replacement =
            apply_rewrite_rule(rule, visited).map_err(|source| RewriteError::Callback {
                rule_index,
                rule_name: rule.name().map(str::to_owned),
                source,
            })?;
        if let Some(replacement) = replacement {
            fired.push(FiredRule {
                rule_index,
                name: rule.name.clone(),
            });
            return Ok(Some(replacement));
        }
    }
    Ok(rebuilt)
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
/// use fhy_core::symbolic::expression::{BinaryOperation, Expression, LiteralValue};
/// use fhy_core::symbolic::expression::pattern::{
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
    /// A tree in which no rule fired is unchanged. A rule firing at the root
    /// and returning the root itself leaves the tree unchanged too; a rule
    /// firing below the root always rebuilds the root, even when it returns
    /// the node it matched, so the tree is changed although it is
    /// structurally equal to the input.
    #[must_use]
    pub fn is_changed(&self) -> bool {
        self.changed
    }

    /// Return every firing in walk order: children before their parent, and
    /// children in [`Expression::children`] order.
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
    /// piecewise whose rewritten case condition became a literal other than
    /// a Boolean.
    ///
    /// Displays as `rebuilding a node from its rewritten children failed`;
    /// the build error is the [`source`](Error::source).
    Rebuild(ExpressionBuildError),
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
            Self::Rebuild(_) => f.write_str("rebuilding a node from its rewritten children failed"),
        }
    }
}

impl Error for RewriteError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Callback { source, .. } => Some(source),
            Self::Rebuild(error) => Some(error),
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
/// the call until the outcome is unchanged. A subtree that occurs in several
/// places is rewritten at each occurrence.
///
/// When no rule fires anywhere, the output is a handle to `expression`
/// itself. See [`RewriteOutcome::is_changed`] for the exact meaning of a
/// change.
///
/// The walk keeps its own work stack, so a tree of any depth rewrites within
/// the default 2 MiB thread stack when the rules' patterns are shallow.
///
/// # Errors
///
/// Returns [`RewriteError::Callback`] for the first callback that fails,
/// naming its rule, and [`RewriteError::Rebuild`] if a node cannot be
/// rebuilt from its rewritten children. The walk stops at the first error.
pub fn apply_rewrite_rules(
    expression: &Expression,
    rules: &[RewriteRule],
) -> Result<RewriteOutcome, RewriteError> {
    let mut fired = Vec::new();
    let mut ancestors: Vec<WalkFrame> = Vec::new();
    let mut current = WalkFrame::new(expression);
    loop {
        if let Some(child) = current.next_child() {
            ancestors.push(std::mem::replace(&mut current, WalkFrame::new(&child)));
            continue;
        }
        let rewritten = finish_node(current, rules, &mut fired)?;
        let Some(mut parent) = ancestors.pop() else {
            let output = rewritten.unwrap_or_else(|| expression.clone());
            let changed = !Expression::ptr_eq(&output, expression);
            return Ok(RewriteOutcome {
                output,
                changed,
                fired,
            });
        };
        parent.rewritten_children.push(rewritten);
        current = parent;
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<RewriteRule>();
    assert_send_sync::<RewriteOutcome>();
    assert_send_sync::<RewriteError>();
};
