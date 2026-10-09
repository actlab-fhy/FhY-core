//! The composed built-in functions, by their definitions.

use std::borrow::Cow;

use crate::expression::builtins::BuiltinFunction;
use num_traits::Signed;

use crate::expression::{BinaryOperation, Expression, Rational, build_zero, compute_arithmetic};
use crate::solver::SimplifyContext;

use super::SimplificationStrategy;
use super::exact::{
    build_number_expression, pick_larger, pick_smaller, read_boolean, read_builtin_call,
    read_numbers, take_absolute, take_sign,
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
    /// The strategy's name, which [`name`](SimplificationStrategy::name)
    /// returns.
    pub const NAME: &str = "composed_builtins";

    /// Return the strategy.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl SimplificationStrategy for ComposedBuiltins {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(Self::NAME)
    }

    fn rewrite(&self, node: &Expression, _context: &SimplifyContext<'_>) -> Option<Expression> {
        let (function, arguments) = read_builtin_call(node)?;
        if let [left, right] = arguments {
            if let (Some(left), Some(right)) = (read_boolean(left), read_boolean(right)) {
                return evaluate_boolean(function, left, right).map(Expression::literal);
            }
        }
        let numbers = read_numbers(arguments)?;
        Some(build_number_expression(evaluate_numeric(
            function, &numbers,
        )?))
    }
}

/// Return the value of the Boolean built-in `function` of two Booleans.
fn evaluate_boolean(function: BuiltinFunction, left: bool, right: bool) -> Option<bool> {
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
fn evaluate_numeric(function: BuiltinFunction, arguments: &[Rational]) -> Option<Rational> {
    Some(match (function, arguments) {
        (BuiltinFunction::Max, [a, b]) => pick_larger(a, b).clone(),
        (BuiltinFunction::Min, [a, b]) => pick_smaller(a, b).clone(),
        (BuiltinFunction::Abs, [x]) => take_absolute(x),
        (BuiltinFunction::Sign, [x]) => take_sign(x),
        (BuiltinFunction::Clamp, [x, low, high]) => clamp(x, low, high),
        (BuiltinFunction::ClampSymmetric, [x, bound]) => clamp(x, &-bound.clone(), bound),
        (BuiltinFunction::Relu, [x]) => {
            if x.numerator().is_positive() {
                x.clone()
            } else {
                build_zero()
            }
        }
        (BuiltinFunction::LeakyRelu, [x, slope]) => {
            if x.numerator().is_positive() {
                x.clone()
            } else {
                compute_arithmetic(BinaryOperation::Multiply, x, slope)?
            }
        }
        _ => return None,
    })
}

/// Return `min(max(x, low), high)`.
fn clamp(x: &Rational, low: &Rational, high: &Rational) -> Rational {
    pick_smaller(pick_larger(x, low), high).clone()
}
