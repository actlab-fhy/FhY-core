//! The composed built-in functions, by their definitions.

use std::borrow::Cow;

use crate::expression::builtins::BuiltinFunction;
use crate::expression::{BinaryOperation, Callee, Expression, ExpressionKind, Rational};
use crate::solver::SimplifyContext;

use super::SimplificationStrategy;
use super::exact::{
    absolute, arithmetic, boolean, larger, negated, number, number_expression, sign, smaller, zero,
};

/// Rewrites a call of a composed built-in function of exact numbers or
/// Booleans to its value, by the function's definition:
///
/// - the numeric functions `max`, `min`, `abs`, `sign`, `clamp`,
///   `clamp_symmetric`, `relu` and `leaky_relu`;
/// - the Boolean functions `xor`, `nand`, `nor`, `implies` and `iff`.
///
/// This strategy is an **extension, and not one of the
/// [default strategies](super::default_strategies)**. The `SymPy` backend
/// refuses a call of a composed function until it is inlined, so there is no
/// `SymPy` answer for the call itself, and the contract of the default
/// strategies (`SymPy`'s answer, or a decline) has nothing to hold the
/// rewrite to. The value this strategy gives is the one `SymPy` gives for
/// the function's inlined form, which the differential tests check. A
/// caller that wants a composed call folded adds the strategy with
/// [`GroundSimplifier::with_strategy`](crate::solver::GroundSimplifier::with_strategy).
///
/// An operand that is not a decided number (or Boolean, for the Boolean
/// functions) makes the strategy decline.
///
/// # Examples
///
/// ```
/// use fhy_core::expression::Expression;
/// use fhy_core::expression::builtins::BuiltinFunction;
/// use fhy_core::solver::strategy::ComposedBuiltins;
/// use fhy_core::solver::{GroundSimplifier, SimplifyContext};
///
/// let context = SimplifyContext::default();
/// let larger = Expression::call(BuiltinFunction::Max, [2, 5]);
///
/// // The default simplifier declines it.
/// assert_eq!(GroundSimplifier::new().try_simplify(&larger, &context), None);
///
/// // With the extension it folds.
/// let simplifier = GroundSimplifier::new().with_strategy(ComposedBuiltins::new());
/// assert_eq!(simplifier.try_simplify(&larger, &context), Some(Expression::from(5)));
/// ```
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub struct ComposedBuiltins;

impl ComposedBuiltins {
    /// Return the strategy.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl SimplificationStrategy for ComposedBuiltins {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed("composed_builtins")
    }

    fn rewrite(&self, node: &Expression, _context: &SimplifyContext<'_>) -> Option<Expression> {
        let ExpressionKind::Call(call) = node.kind() else {
            return None;
        };
        let Callee::Builtin(function) = call.callee() else {
            return None;
        };
        if let [left, right] = call.arguments() {
            if let (Some(left), Some(right)) = (boolean(left), boolean(right)) {
                return boolean_value(*function, left, right).map(Expression::literal);
            }
        }
        let numbers = call
            .arguments()
            .iter()
            .map(number)
            .collect::<Option<Vec<Rational>>>()?;
        Some(number_expression(numeric_value(*function, &numbers)?))
    }
}

/// Return the value of the Boolean built-in `function` of two Booleans.
fn boolean_value(function: BuiltinFunction, left: bool, right: bool) -> Option<bool> {
    Some(match function {
        BuiltinFunction::Xor => left != right,
        BuiltinFunction::Nand => !(left && right),
        BuiltinFunction::Nor => !(left || right),
        BuiltinFunction::Implies => !left || right,
        BuiltinFunction::Iff => left == right,
        _ => return None,
    })
}

/// Return the value of the numeric built-in `function` of `arguments`.
fn numeric_value(function: BuiltinFunction, arguments: &[Rational]) -> Option<Rational> {
    Some(match (function, arguments) {
        (BuiltinFunction::Max, [a, b]) => larger(a, b).clone(),
        (BuiltinFunction::Min, [a, b]) => smaller(a, b).clone(),
        (BuiltinFunction::Abs, [x]) => absolute(x),
        (BuiltinFunction::Sign, [x]) => sign(x),
        (BuiltinFunction::Clamp, [x, low, high]) => clamp(x, low, high),
        (BuiltinFunction::ClampSymmetric, [x, bound]) => clamp(x, &negated(bound.clone()), bound),
        (BuiltinFunction::Relu, [x]) => larger(x, &zero()).clone(),
        (BuiltinFunction::LeakyRelu, [x, slope]) => {
            if *x > zero() {
                x.clone()
            } else {
                arithmetic(BinaryOperation::Multiply, x, slope)?
            }
        }
        _ => return None,
    })
}

/// Return `min(max(x, low), high)`.
fn clamp(x: &Rational, low: &Rational, high: &Rational) -> Rational {
    smaller(larger(x, low), high).clone()
}
