//! The strategies of the [`GroundSimplifier`](super::GroundSimplifier): the
//! local rewrites it applies, one concern each, and the trait that adds
//! more.
//!
//! A [`SimplificationStrategy`] looks at one node whose children are
//! already simplified, and rewrites it or declines. The driver applies the
//! strategies bottom-up and in order; see [`GroundSimplifier`](super::GroundSimplifier).
//!
//! # The default strategies
//!
//! [`default_strategies`] lists them, in the order the driver tries them
//! on a node:
//!
//! | Strategy | Name | Rewrites |
//! |---|---|---|
//! | [`RegisteredConstants`] | `registered_constants` | a native constant of the context's registry, to its value |
//! | [`NormalizeLiterals`] | `normalize_literals` | a decimal literal, to the form `SymPy` writes it in |
//! | [`ExactArithmetic`] | `exact_arithmetic` | `+`, `-`, `*`, `/`, `//`, `%`, `**` and the unary `-` and `+` of exact numbers |
//! | [`Comparisons`] | `comparisons` | a comparison of numbers, or of Booleans for `==` and `!=` |
//! | [`LogicalOperators`] | `logical_operators` | `!`, `&&` and `\|\|` of Booleans |
//! | [`PiecewiseDecision`] | `piecewise` | a piecewise whose conditions and values are decided, to the value of its first true case |
//! | [`ExactBuiltins`] | `exact_builtins` | a call of a built-in with an exact result that `SymPy` folds itself |
//!
//! # The opt-in strategies
//!
//! A strategy outside the default list is an extension: a caller adds it with
//! [`GroundSimplifier::with_strategy`](super::GroundSimplifier::with_strategy).
//!
//! | Strategy | Name | Rewrites |
//! |---|---|---|
//! | [`ComposedBuiltins`] | `composed_builtins` | a call of a composed built-in (`max`, `min`, `abs`, `sign`, `clamp`, `clamp_symmetric`, `relu`, `leaky_relu`, `xor`, `nand`, `nor`, `implies`, `iff`), by its definition |
//!
//! `SymPy` refuses these calls until they are inlined, so the default
//! simplifier declines them and, with it, the default pipeline is exactly
//! `SymPy`'s answer or a decline. An opt-in strategy gives the answer `SymPy`
//! gives for the inlined form instead.
//!
//! # The contract
//!
//! Every strategy keeps one invariant, which is what makes the simplifier
//! safe to put in front of `SymPy`: **a rewrite is exactly the result of
//! the `SymPy` simplifier for the node it rewrites, or the strategy
//! declines.** Exactly means the same expression: the literal kind (`Int`,
//! `Bool`, a decimal or a quotient for a rational), and the structure of
//! anything larger. A strategy never approximates, and it declines whatever
//! it is not sure `SymPy` answers alike: a float, an irrational value, an
//! operation `SymPy` leaves unevaluated or refuses, a free identifier.
//!
//! The rest of the contract is what makes strategies composable:
//!
//! - A strategy is **local**: it reads the node and its children, and the
//!   [`SimplifyContext`], and nothing else.
//! - It makes **no assumption about the other strategies**: it may run
//!   alone, after any other, or never. It reads a child as a value only
//!   where it is written in a form it recognises, and declines otherwise.
//! - It is **deterministic**, and returns `None` when it has nothing to
//!   rewrite, rather than the node itself.
//! - It **terminates**: the driver bounds the rewrites of a run, but a
//!   strategy that always rewrites to something new is a strategy that
//!   never lets the run reach its fixed point.
//!
//! # Adding a strategy
//!
//! 1. Implement [`SimplificationStrategy`] and meet the contract above.
//! 2. Test it alone in
//!    `rust/fhy-core/tests/it/solver/ground_strategy_stories.rs`
//!    (`ground_stories.rs` tests the whole pipeline): what it rewrites, and
//!    what it declines, through
//!    [`rewrite`](SimplificationStrategy::rewrite) and through a
//!    [`GroundSimplifier`](super::GroundSimplifier) holding only it.
//! 3. Add its cases to the differential tests of the binding
//!    (`rust/fhy-core-py/src/solver/sympy/ground_differential.rs`), which
//!    compare each rewrite with `SymPy`'s result for the same node:
//!    `cargo test -p fhy-core-py ground_differential`, which needs Python
//!    with `SymPy` (`CONTRIBUTING.md`, "Rust test layout").
//! 4. Add it to [`default_strategies`] if it should run by default, and to
//!    the table above. It belongs there only if `SymPy` answers the node
//!    itself, without inlining. Any other strategy is opt-in: list it in the
//!    second table, and a caller adds it with
//!    [`GroundSimplifier::with_strategy`](super::GroundSimplifier::with_strategy).
//!
//! # Examples
//!
//! A strategy that rewrites `min(a, a)` to `a`, which `SymPy` does too:
//!
//! ```
//! use std::borrow::Cow;
//!
//! use fhy_core::expression::builtins::BuiltinFunction;
//! use fhy_core::expression::{Callee, Expression, ExpressionKind};
//! use fhy_core::solver::strategy::SimplificationStrategy;
//! use fhy_core::solver::{GroundSimplifier, SimplifyContext};
//!
//! #[derive(Debug)]
//! struct MinOfEquals;
//!
//! impl SimplificationStrategy for MinOfEquals {
//!     fn name(&self) -> Cow<'_, str> {
//!         Cow::Borrowed("min_of_equals")
//!     }
//!
//!     fn rewrite(&self, node: &Expression, _context: &SimplifyContext<'_>) -> Option<Expression> {
//!         let ExpressionKind::Call(call) = node.kind() else {
//!             return None;
//!         };
//!         let is_min = matches!(call.callee(), Callee::Builtin(BuiltinFunction::Min));
//!         match call.arguments() {
//!             [a, b] if is_min && a == b => Some(a.clone()),
//!             _ => None,
//!         }
//!     }
//! }
//!
//! let simplifier = GroundSimplifier::empty().with_strategy(MinOfEquals);
//! let seven = Expression::from(7);
//! let call = Expression::call(BuiltinFunction::Min, [&seven, &seven]);
//!
//! assert_eq!(simplifier.try_simplify(&call, &SimplifyContext::default()), Some(seven));
//! ```

mod arithmetic;
mod builtins;
mod comparison;
mod composed;
mod constants;
mod exact;
mod literals;
mod logic;
mod piecewise;

use std::borrow::Cow;
use std::fmt;
use std::sync::Arc;

use crate::expression::Expression;

use super::backend::SimplifyContext;

pub use arithmetic::ExactArithmetic;
pub use builtins::ExactBuiltins;
pub use comparison::Comparisons;
pub use composed::ComposedBuiltins;
pub use constants::RegisteredConstants;
pub use literals::NormalizeLiterals;
pub use logic::LogicalOperators;
pub use piecewise::PiecewiseDecision;

pub(super) use exact::is_decided;

/// A local rewrite of the [`GroundSimplifier`](super::GroundSimplifier):
/// one concern, applied to one node at a time.
///
/// The module documentation states the contract every strategy keeps:
/// a rewrite is exactly `SymPy`'s result for the node, or the strategy
/// declines.
pub trait SimplificationStrategy: Send + Sync + fmt::Debug {
    /// Return the strategy's name, which
    /// [`GroundSimplifier::without`](super::GroundSimplifier::without)
    /// removes it by and
    /// [`GroundSimplifier::holds`](super::GroundSimplifier::holds) looks it
    /// up by.
    ///
    /// Each strategy the crate ships exposes its name as the constant
    /// `Type::NAME` (`Comparisons::NAME`, for one), which a caller uses in
    /// place of the text so a misspelling does not compile.
    fn name(&self) -> Cow<'_, str>;

    /// Return the expression `node` rewrites to, or `None` where the
    /// strategy has nothing to rewrite.
    ///
    /// The children of `node` are already simplified by the driver. The
    /// strategy returns `None`, not `node` itself, when it declines.
    fn rewrite(&self, node: &Expression, context: &SimplifyContext<'_>) -> Option<Expression>;
}

/// Return the default strategies, in the order the driver tries them.
///
/// # Examples
///
/// ```
/// use fhy_core::solver::strategy::default_strategies;
///
/// let names: Vec<_> = default_strategies().iter().map(|s| s.name().into_owned()).collect();
///
/// assert_eq!(names[0], "registered_constants");
/// assert!(names.contains(&"exact_arithmetic".to_owned()));
/// ```
#[must_use]
pub fn default_strategies() -> Vec<Arc<dyn SimplificationStrategy>> {
    vec![
        Arc::new(RegisteredConstants),
        Arc::new(NormalizeLiterals),
        Arc::new(ExactArithmetic),
        Arc::new(Comparisons),
        Arc::new(LogicalOperators),
        Arc::new(PiecewiseDecision),
        Arc::new(ExactBuiltins),
    ]
}
