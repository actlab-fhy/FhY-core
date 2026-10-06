//! The built-in functions with an exact result.

use std::borrow::Cow;

use num_traits::{One, Zero};

use crate::expression::builtins::BuiltinFunction;
use crate::expression::{Expression, Rational};
use crate::solver::SimplifyContext;

use super::SimplificationStrategy;
use super::exact::{
    build_integer, build_number_expression, build_zero, ceil, floor, is_one, raise_to_power,
    read_builtin_call, read_numbers, take_integer_logarithm, take_root,
};

/// Rewrites a call of a built-in function of exact numbers to its value,
/// where the value is exact:
///
/// - `floor` and `ceil` of any exact number, and `round` of an integer;
/// - `sqrt` and `exp2` where the result is rational, `log2` and `log10`
///   where it is an integer and the argument has at most 4096 bits;
/// - `exp`, `cos` and `cosh` at `0`, which are `1`; `sin`, `tan`, `arcsin`,
///   `arctan`, `sinh`, `tanh` and `erf` at `0`, which are `0`; and `log`
///   and `arccos` at `1`, which are `0`.
///
/// Each of these is a function the `SymPy` backend folds itself. The
/// composed functions (`max`, `min`, `abs`, ...), which it refuses until
/// they are inlined, are folded by
/// [`ComposedBuiltins`](super::ComposedBuiltins), an opt-in strategy.
/// `sigmoid`, `silu`, `gelu`, `round` of a non-integer and every other call
/// are declined.
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
    /// The strategy's name, which [`name`](SimplificationStrategy::name)
    /// returns.
    pub const NAME: &str = "exact_builtins";

    /// Return the strategy.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl SimplificationStrategy for ExactBuiltins {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(Self::NAME)
    }

    fn rewrite(&self, node: &Expression, _context: &SimplifyContext<'_>) -> Option<Expression> {
        let (function, arguments) = read_builtin_call(node)?;
        let numbers = read_numbers(arguments)?;
        Some(build_number_expression(evaluate_call(function, &numbers)?))
    }
}

/// Return the exact value of the built-in `function` of `arguments`.
fn evaluate_call(function: BuiltinFunction, arguments: &[Rational]) -> Option<Rational> {
    Some(match (function, arguments) {
        (BuiltinFunction::Floor, [x]) => build_integer(floor(x)),
        (BuiltinFunction::Ceil, [x]) => build_integer(ceil(x)),
        (BuiltinFunction::Round, [x]) if x.denominator().is_one() => x.clone(),
        (BuiltinFunction::Sqrt, [x]) => take_root(x, 2)?,
        (BuiltinFunction::Exp2, [x]) => raise_to_power(&build_integer(2.into()), x)?,
        (BuiltinFunction::Log2, [x]) => take_integer_logarithm(x, 2)?,
        (BuiltinFunction::Log10, [x]) => take_integer_logarithm(x, 10)?,
        (BuiltinFunction::Exp | BuiltinFunction::Cos | BuiltinFunction::Cosh, [x])
            if is_zero(x) =>
        {
            build_integer(1.into())
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
        ) if is_zero(x) => build_zero(),
        (BuiltinFunction::Log | BuiltinFunction::Arccos, [x]) if is_one(x) => build_zero(),
        _ => return None,
    })
}

fn is_zero(number: &Rational) -> bool {
    number.numerator().is_zero()
}
