//! The built-in functions with an exact result.

use std::borrow::Cow;

use num_traits::One;

use crate::expression::builtins::BuiltinFunction;
use crate::expression::{Callee, Expression, ExpressionKind, Rational};
use crate::solver::SimplifyContext;

use super::SimplificationStrategy;
use super::exact::{
    ceiling, floor, integer, integer_logarithm, number, number_expression, power, zero,
};

/// Rewrites a call of a built-in function of exact numbers to its value,
/// where the value is exact:
///
/// - `floor`, `ceil`, and `round` of an integer;
/// - `sqrt` and `exp2` where the result is rational, `log2` and `log10`
///   where it is an integer;
/// - `exp`, `log`, the trigonometric, hyperbolic and `erf` functions at the
///   points where `SymPy` knows their value (`exp(0)`, `log(1)`, `sin(0)`,
///   `cos(0)`, `acos(1)`, ...).
///
/// Each of these is a function the `SymPy` backend folds itself. The
/// composed functions (`max`, `min`, `abs`, ...), which it refuses until
/// they are inlined, are [`ComposedBuiltins`](super::ComposedBuiltins)', a
/// strategy that is not a default. `sigmoid`, `silu`, `gelu`, `round` of a
/// non-integer and every other call are declined.
///
/// # Examples
///
/// ```
/// use fhy_core::expression::Expression;
/// use fhy_core::expression::builtins::BuiltinFunction;
/// use fhy_core::solver::strategy::{ExactBuiltins, SimplificationStrategy};
/// use fhy_core::solver::SimplifyContext;
///
/// let context = SimplifyContext::default();
///
/// let root = Expression::call(BuiltinFunction::Sqrt, [16]);
/// assert_eq!(ExactBuiltins::new().rewrite(&root, &context), Some(Expression::from(4)));
///
/// let irrational = Expression::call(BuiltinFunction::Sqrt, [2]);
/// assert_eq!(ExactBuiltins::new().rewrite(&irrational, &context), None);
/// ```
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub struct ExactBuiltins;

impl ExactBuiltins {
    /// Return the strategy.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl SimplificationStrategy for ExactBuiltins {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed("exact_builtins")
    }

    fn rewrite(&self, node: &Expression, _context: &SimplifyContext<'_>) -> Option<Expression> {
        let ExpressionKind::Call(call) = node.kind() else {
            return None;
        };
        let Callee::Builtin(function) = call.callee() else {
            return None;
        };
        let numbers = call
            .arguments()
            .iter()
            .map(number)
            .collect::<Option<Vec<Rational>>>()?;
        Some(number_expression(call_value(*function, &numbers)?))
    }
}

/// Return the exact value of the built-in `function` of `arguments`.
fn call_value(function: BuiltinFunction, arguments: &[Rational]) -> Option<Rational> {
    Some(match (function, arguments) {
        (BuiltinFunction::Floor, [x]) => integer(floor(x)),
        (BuiltinFunction::Ceil, [x]) => integer(ceiling(x)),
        (BuiltinFunction::Round, [x]) if x.denominator().is_one() => x.clone(),
        (BuiltinFunction::Sqrt, [x]) => power(x, &Rational::new(1.into(), 2.into())?)?,
        (BuiltinFunction::Exp2, [x]) => power(&integer(2.into()), x)?,
        (BuiltinFunction::Log2, [x]) => integer_logarithm(x, 2)?,
        (BuiltinFunction::Log10, [x]) => integer_logarithm(x, 10)?,
        (BuiltinFunction::Exp | BuiltinFunction::Cos | BuiltinFunction::Cosh, [x])
            if is_zero(x) =>
        {
            integer(1.into())
        }
        (
            BuiltinFunction::Sin
            | BuiltinFunction::Tan
            | BuiltinFunction::Arcsin
            | BuiltinFunction::Arctan
            | BuiltinFunction::Sinh
            | BuiltinFunction::Tanh
            | BuiltinFunction::Erf,
            [x],
        ) if is_zero(x) => zero(),
        (BuiltinFunction::Log | BuiltinFunction::Arccos, [x]) if *x == integer(1.into()) => zero(),
        _ => return None,
    })
}

fn is_zero(number: &Rational) -> bool {
    *number == zero()
}
