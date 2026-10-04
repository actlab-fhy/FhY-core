//! The ground simplifier: a [`Simplifier`] that needs neither Python nor
//! `SymPy`, built from [strategies](super::strategy), and the composition
//! that falls back to another simplifier where it declines.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::expression::Expression;
use crate::foreign::BoxError;
use crate::tree::{BuildIdentityHasher, NodeHandle, NodeIdentity, Tree};

use super::backend::{Simplifier, SimplifyContext};
use super::strategy::{SimplificationStrategy, default_strategies, is_decided};

/// How deeply the driver nests before it declines, which keeps a deep tree
/// from exhausting the stack.
const MAX_DEPTH: usize = 256;

/// The rewrites a run may make by default: far more than the nodes of an
/// expression a caller means to fold, and few enough that strategies
/// undoing each other's rewrites stop quickly.
const DEFAULT_MAX_REWRITES: usize = 100_000;

/// A [`Simplifier`] that rewrites an expression with an ordered list of
/// [strategies](super::strategy), with no Python and no `SymPy`.
///
/// # What it answers
///
/// The default strategies fold an expression with no free identifier in
/// exact arithmetic over [`BigInt`](crate::expression::BigInt)s and
/// rationals. Wherever it folds, the result is **exactly** what the `SymPy`
/// simplifier of the Python binding returns for the same input, in the same
/// form: an integer is an `Int` literal, a Boolean a `Bool` literal, and a
/// rational that is not an integer the decimal literal of its value when a
/// binary float equals it (negated by a unary minus when it is negative) and
/// otherwise the quotient of its numerator and denominator.
///
/// Wherever it cannot match `SymPy` exactly it **declines**: it returns the
/// expression unchanged, as the best-effort contract of [`Simplifier`]
/// allows, and never approximates. It declines an expression with a free
/// identifier, the identifier of a built-in constant, a float, a call of a
/// user function, an operation without an exact rational result, a value
/// that is not of the sort an operation takes, and a power of more than a
/// million bits. [`strategy`](super::strategy) lists what each default
/// strategy rewrites.
///
/// # How it works
///
/// The driver rewrites bottom-up: it simplifies the children of a node
/// first, then tries the strategies in order on the node, the first that
/// rewrites it winning, and repeats on the rewritten node until no strategy
/// rewrites it, a fixed point. A node the expression shares is simplified
/// once.
///
/// The run is bounded: it makes at most
/// [`max_rewrites`](Self::max_rewrites) rewrites (100 000 by default), and
/// when it reaches the bound it stops where it is, cleanly, so strategies
/// that undo each other's rewrites end the run instead of looping. A tree
/// nested more than 256 deep is declined.
///
/// # What it returns
///
/// Until a strategy reproduces `SymPy`'s canonical form of an expression
/// that is not decided, a partly folded expression would differ from
/// `SymPy`'s (`x + 3` against `3 + x`). So the driver returns the rewritten
/// expression only when it is **decided**, a literal in the form `SymPy`
/// answers, and the expression unchanged otherwise.
/// [`with_partial_rewrites`](Self::with_partial_rewrites) keeps the
/// rewritten expression of a run whatever it is, for strategies whose
/// rewrites match `SymPy`'s form of a larger expression.
///
/// # Extending it
///
/// [`new`](Self::new) holds the [default strategies](default_strategies).
/// [`with_strategy`](Self::with_strategy) adds one,
/// [`with_strategy_first`](Self::with_strategy_first) adds one that is tried
/// before the others, [`without`](Self::without) removes one by name, and
/// [`empty`](Self::empty) starts from none. The [`strategy`](super::strategy)
/// module states the contract a strategy keeps and how to add one.
///
/// The simplifier ignores [`SimplifyLimits`](super::SimplifyLimits): its
/// run is bounded by rewrites, not time.
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
///
/// use fhy_core::expression::Expression;
/// use fhy_core::identifier::Identifier;
/// use fhy_core::solver::{GroundSimplifier, SimplifyContext, Solver};
///
/// let x = Identifier::new("x");
/// let bound = Expression::from(x.clone()).greater_equal(0);
/// let solver = Solver::new().with_simplifier(GroundSimplifier::new());
/// let context = SimplifyContext::default();
///
/// let environment = HashMap::from([(x, Expression::from(3))]);
/// let decided = solver.simplify(&bound, &environment, &context)?;
/// assert_eq!(decided, Expression::literal(true));
///
/// // With `x` free there is nothing to decide, and the expression is kept.
/// let kept = solver.simplify(&bound, &HashMap::new(), &context)?;
/// assert_eq!(kept, bound);
///
/// // Without the comparison strategy, the comparison is not decided.
/// let without = GroundSimplifier::new().without("comparisons");
/// let comparison = Expression::from(1).less(2);
/// assert_eq!(without.try_simplify(&comparison, &context), None);
/// # Ok::<(), fhy_core::solver::SolveError>(())
/// ```
#[derive(Clone)]
pub struct GroundSimplifier {
    strategies: Vec<Arc<dyn SimplificationStrategy>>,
    max_rewrites: usize,
    keeps_partial_rewrites: bool,
}

impl GroundSimplifier {
    /// Return the simplifier holding the [default
    /// strategies](default_strategies).
    #[must_use]
    pub fn new() -> Self {
        Self::empty().with_default_strategies()
    }

    /// Return the simplifier holding no strategy, which rewrites nothing.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            strategies: Vec::new(),
            max_rewrites: DEFAULT_MAX_REWRITES,
            keeps_partial_rewrites: false,
        }
    }

    /// Return this simplifier with the [default
    /// strategies](default_strategies) added after the ones it holds.
    #[must_use]
    pub fn with_default_strategies(mut self) -> Self {
        self.strategies.extend(default_strategies());
        self
    }

    /// Return this simplifier with `strategy` tried after the ones it
    /// holds.
    #[must_use]
    pub fn with_strategy(self, strategy: impl SimplificationStrategy + 'static) -> Self {
        self.with_shared_strategy(Arc::new(strategy))
    }

    /// Return this simplifier with the shared `strategy` tried after the
    /// ones it holds.
    #[must_use]
    pub fn with_shared_strategy(mut self, strategy: Arc<dyn SimplificationStrategy>) -> Self {
        self.strategies.push(strategy);
        self
    }

    /// Return this simplifier with `strategy` tried before the ones it
    /// holds.
    #[must_use]
    pub fn with_strategy_first(mut self, strategy: impl SimplificationStrategy + 'static) -> Self {
        self.strategies.insert(0, Arc::new(strategy));
        self
    }

    /// Return this simplifier without the strategies named `name`.
    #[must_use]
    pub fn without(mut self, name: &str) -> Self {
        self.strategies.retain(|strategy| strategy.name() != name);
        self
    }

    /// Return this simplifier bounded to `max_rewrites` rewrites per run.
    #[must_use]
    pub const fn with_max_rewrites(mut self, max_rewrites: usize) -> Self {
        self.max_rewrites = max_rewrites;
        self
    }

    /// Return this simplifier keeping the rewritten expression of a run
    /// whether or not it is decided.
    ///
    /// Only for strategies whose rewrites match `SymPy`'s form of an
    /// expression that is not a literal; the default strategies do not,
    /// so with them `x + (1 + 2)` would come back as `x + 3`, where `SymPy`
    /// answers `3 + x`.
    #[must_use]
    pub const fn with_partial_rewrites(mut self) -> Self {
        self.keeps_partial_rewrites = true;
        self
    }

    /// Return the names of the strategies, in the order they are tried.
    #[must_use]
    pub fn strategy_names(&self) -> Vec<String> {
        self.strategies
            .iter()
            .map(|strategy| strategy.name().into_owned())
            .collect()
    }

    /// Return the most rewrites one run makes before it stops.
    #[must_use]
    pub const fn max_rewrites(&self) -> usize {
        self.max_rewrites
    }

    /// Return the expression `expression` is rewritten to, or `None` where
    /// the simplifier declines.
    ///
    /// The result is decided, a literal in the form `SymPy` answers, unless
    /// [`with_partial_rewrites`](Self::with_partial_rewrites) keeps a rewrite
    /// that is not; then it is any expression other than `expression`.
    /// [`simplify`](Simplifier::simplify) is this, with the expression itself
    /// for the declined case; [`GroundWithFallback`] asks this to know whether
    /// to fall back.
    ///
    /// # Examples
    ///
    /// ```
    /// use fhy_core::expression::Expression;
    /// use fhy_core::identifier::Identifier;
    /// use fhy_core::solver::{GroundSimplifier, SimplifyContext};
    ///
    /// let context = SimplifyContext::default();
    /// let ground = GroundSimplifier::new();
    ///
    /// let sum = Expression::from(2) * Expression::from(3) + Expression::from(1);
    /// assert_eq!(ground.try_simplify(&sum, &context), Some(Expression::from(7)));
    ///
    /// let free = Expression::from(Identifier::new("x")) + 1;
    /// assert_eq!(ground.try_simplify(&free, &context), None);
    /// ```
    #[must_use]
    pub fn try_simplify(
        &self,
        expression: &Expression,
        context: &SimplifyContext<'_>,
    ) -> Option<Expression> {
        let mut run = Run {
            simplifier: self,
            context,
            memo: HashMap::default(),
            rewrites: 0,
        };
        let result = run.simplify(expression, 0)?;
        if is_decided(&result) || (self.keeps_partial_rewrites && result != *expression) {
            Some(result)
        } else {
            None
        }
    }
}

impl Default for GroundSimplifier {
    /// Return [`GroundSimplifier::new`].
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for GroundSimplifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GroundSimplifier")
            .field("strategies", &self.strategy_names())
            .field("max_rewrites", &self.max_rewrites)
            .field("keeps_partial_rewrites", &self.keeps_partial_rewrites)
            .finish()
    }
}

impl Simplifier for GroundSimplifier {
    /// Return `ground`.
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed("ground")
    }

    /// Return what [`try_simplify`](GroundSimplifier::try_simplify) returns,
    /// or `expression` itself where the simplifier declines. It never fails.
    fn simplify(
        &self,
        expression: &Expression,
        context: &SimplifyContext<'_>,
    ) -> Result<Expression, BoxError> {
        Ok(self
            .try_simplify(expression, context)
            .unwrap_or_else(|| expression.clone()))
    }
}

/// One run of the driver: the context, the simplified form of each node it
/// has met by the node's identity, so a node the expression shares is
/// simplified once, and the rewrites made so far.
struct Run<'s, 'c, 'a> {
    simplifier: &'s GroundSimplifier,
    context: &'c SimplifyContext<'a>,
    memo: HashMap<NodeIdentity, Expression, BuildIdentityHasher>,
    rewrites: usize,
}

impl Run<'_, '_, '_> {
    /// Return `node` simplified, or `None` to decline the whole run.
    fn simplify(&mut self, node: &Expression, depth: usize) -> Option<Expression> {
        if depth > MAX_DEPTH {
            return None;
        }
        let has_children = node.children().len() > 0;
        // Only a node held in several places is worth remembering.
        let is_shared = has_children && node.is_shared();
        if is_shared {
            if let Some(simplified) = self.memo.get(&node.identity()) {
                return Some(simplified.clone());
            }
        }
        let mut current = if has_children {
            self.with_simplified_children(node, depth)?
        } else {
            node.clone()
        };
        while let Some(rewritten) = self.rewrite_once(&current) {
            if rewritten.children().len() == 0 {
                current = rewritten;
            } else {
                // A rewrite may build new children, which are simplified
                // before the node is tried again.
                current = self.simplify(&rewritten, depth + 1)?;
                break;
            }
        }
        if is_shared {
            self.memo.insert(node.identity(), current.clone());
        }
        Some(current)
    }

    /// Return `node` with each child simplified.
    fn with_simplified_children(&mut self, node: &Expression, depth: usize) -> Option<Expression> {
        let mut children = Vec::with_capacity(node.children().len());
        let mut is_changed = false;
        for child in node.children() {
            let simplified = self.simplify(child, depth + 1)?;
            is_changed |= !Expression::ptr_eq(&simplified, child);
            children.push(simplified);
        }
        if is_changed {
            node.rebuild_with_children(children).ok()
        } else {
            Some(node.clone())
        }
    }

    /// Return what the first strategy that rewrites `node` rewrites it to,
    /// or `None` where none does, or the run has reached its bound.
    fn rewrite_once(&mut self, node: &Expression) -> Option<Expression> {
        if self.rewrites >= self.simplifier.max_rewrites {
            return None;
        }
        let rewritten = self.simplifier.strategies.iter().find_map(|strategy| {
            strategy
                .rewrite(node, self.context)
                .filter(|rewritten| rewritten != node)
        })?;
        self.rewrites += 1;
        Some(rewritten)
    }
}

/// A [`Simplifier`] that tries the [`GroundSimplifier`] first and asks
/// another simplifier where it declines.
///
/// Every answer is the ground simplifier's, which equals what `SymPy`
/// would answer, or the fallback's, so with `SymPy` as the fallback the
/// composition answers what `SymPy` alone does, faster where the
/// expression is ground. A failure is the fallback's, unchanged. Where the
/// ground simplifier keeps a rewrite that is not decided (see
/// [`with_partial_rewrites`](GroundSimplifier::with_partial_rewrites)), the
/// fallback simplifies the rewritten expression.
///
/// # Examples
///
/// ```
/// use std::borrow::Cow;
///
/// use fhy_core::expression::Expression;
/// use fhy_core::foreign::BoxError;
/// use fhy_core::identifier::Identifier;
/// use fhy_core::solver::{GroundWithFallback, Simplifier, SimplifyContext};
///
/// /// A fallback that rewrites everything it is asked to `0`.
/// #[derive(Debug)]
/// struct Zeroing;
///
/// impl Simplifier for Zeroing {
///     fn name(&self) -> Cow<'_, str> {
///         Cow::Borrowed("zeroing")
///     }
///
///     fn simplify(
///         &self,
///         _expression: &Expression,
///         _context: &SimplifyContext<'_>,
///     ) -> Result<Expression, BoxError> {
///         Ok(Expression::from(0))
///     }
/// }
///
/// let chain = GroundWithFallback::new(Zeroing);
/// let context = SimplifyContext::default();
///
/// let ground = Expression::from(6) / Expression::from(3);
/// assert_eq!(chain.simplify(&ground, &context)?, Expression::from(2));
///
/// let free = Expression::from(Identifier::new("x")) + 1;
/// assert_eq!(chain.simplify(&free, &context)?, Expression::from(0));
/// # Ok::<(), BoxError>(())
/// ```
#[derive(Clone)]
pub struct GroundWithFallback {
    ground: GroundSimplifier,
    fallback: Arc<dyn Simplifier>,
}

impl GroundWithFallback {
    /// Return the composition of a default [`GroundSimplifier`] and
    /// `fallback`, which is asked where the ground simplifier declines.
    #[must_use]
    pub fn new(fallback: impl Simplifier + 'static) -> Self {
        Self::from_shared(Arc::new(fallback))
    }

    /// Return the composition of a default [`GroundSimplifier`] and the
    /// shared `fallback`.
    #[must_use]
    pub fn from_shared(fallback: Arc<dyn Simplifier>) -> Self {
        Self {
            ground: GroundSimplifier::new(),
            fallback,
        }
    }

    /// Return this composition with `ground`, a simplifier of the caller's
    /// strategies, in place of the default one.
    #[must_use]
    pub fn with_ground(mut self, ground: GroundSimplifier) -> Self {
        self.ground = ground;
        self
    }

    /// Return the ground simplifier asked first.
    #[must_use]
    pub const fn ground(&self) -> &GroundSimplifier {
        &self.ground
    }

    /// Return the simplifier asked where the ground simplifier declines.
    #[must_use]
    pub const fn fallback(&self) -> &Arc<dyn Simplifier> {
        &self.fallback
    }
}

impl fmt::Debug for GroundWithFallback {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GroundWithFallback")
            .field("ground", &self.ground)
            .field("fallback", &self.fallback)
            .finish()
    }
}

impl Simplifier for GroundWithFallback {
    /// Return `ground+` and the fallback's name, such as `ground+sympy`.
    fn name(&self) -> Cow<'_, str> {
        Cow::Owned(format!("ground+{}", self.fallback.name()))
    }

    /// Return the ground simplifier's answer when it decides `expression`,
    /// and the fallback's for what is left otherwise.
    ///
    /// # Errors
    ///
    /// Returns the fallback's failure, when the ground simplifier does not
    /// decide `expression` and the fallback fails.
    fn simplify(
        &self,
        expression: &Expression,
        context: &SimplifyContext<'_>,
    ) -> Result<Expression, BoxError> {
        match self.ground.try_simplify(expression, context) {
            Some(rewritten) if is_decided(&rewritten) => Ok(rewritten),
            Some(rewritten) => self.fallback.simplify(&rewritten, context),
            None => self.fallback.simplify(expression, context),
        }
    }
}
