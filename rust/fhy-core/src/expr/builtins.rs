//! The shipped catalogue of built-in functions and constants.
//!
//! [`BuiltinFunction`] names the 35 built-in functions, in catalogue order:
//!
//! - the 16 composed functions, whose meaning is an expression over their
//!   parameters ([`BuiltinFunction::composed`]): `max`, `min`, `abs`,
//!   `sign`, `clamp`, `clamp_symmetric`, `relu`, `leaky_relu`, `xor`,
//!   `nand`, `nor`, `implies`, `iff`, `sigmoid`, `silu`, `gelu`;
//! - the 19 functions computed natively, each taking one real argument:
//!   `exp`, `exp2`, `log`, `log2`, `log10`, `sqrt`, `sin`, `cos`, `tan`,
//!   `arcsin`, `arccos`, `arctan`, `sinh`, `cosh`, `tanh`, `erf`, `round`,
//!   `floor`, `ceil`.
//!
//! [`BuiltinConstant`] names the four real constants `pi`, `e`, `inf` and
//! `nan`. No name is shared by two functions or two constants. A composed
//! body calls other built-in functions through [`Callee::Builtin`](super::Callee::Builtin), and
//! refers to nothing but its own parameters by identifier.
//!
//! The catalogue is data only. It registers nothing, and it does not
//! compute native functions. Look a function up by name with
//! [`FromStr`](std::str::FromStr), and list the composed functions with
//! `BuiltinFunction::iter().filter_map(BuiltinFunction::composed)`.
//!
//! # Examples
//!
//! ```
//! use fhy_core::expr::FunctionSort;
//! use fhy_core::expr::builtins::BuiltinFunction;
//!
//! let relu: BuiltinFunction = "relu".parse()?;
//! assert_eq!(relu.parameter_sorts(), [FunctionSort::Real]);
//! assert_eq!(relu.result_sort(), FunctionSort::Real);
//! assert!(relu.composed().is_some());
//! assert!(BuiltinFunction::Exp.composed().is_none());
//! # Ok::<(), fhy_core::expr::UnknownNameError>(())
//! ```

use std::sync::LazyLock;

use serde::{Deserialize, Serialize};

use crate::identifier::Identifier;

use super::node::Expression;
use super::operation::impl_name_text;
use super::sort::FunctionSort;

/// Sorts of a function taking one real argument.
const REAL_1: &[FunctionSort; 1] = &[FunctionSort::Real];

/// Sorts of a function taking two real arguments.
const REAL_2: &[FunctionSort; 2] = &[FunctionSort::Real, FunctionSort::Real];

/// Sorts of a function taking three real arguments.
const REAL_3: &[FunctionSort; 3] = &[FunctionSort::Real, FunctionSort::Real, FunctionSort::Real];

/// Sorts of a function taking two Boolean arguments.
const BOOL_2: &[FunctionSort; 2] = &[FunctionSort::Bool, FunctionSort::Bool];

/// A built-in function.
///
/// Serializes as its [`name`](Self::name), and deserializes, like
/// [`FromStr`](std::str::FromStr) parses, only from exactly that text.
///
/// # Examples
///
/// ```
/// use fhy_core::expr::builtins::BuiltinFunction;
///
/// assert_eq!(BuiltinFunction::ClampSymmetric.name(), "clamp_symmetric");
/// assert_eq!("log10".parse::<BuiltinFunction>()?, BuiltinFunction::Log10);
/// assert_eq!(BuiltinFunction::iter().len(), 35);
/// # Ok::<(), fhy_core::expr::UnknownNameError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum BuiltinFunction {
    /// `max(a, b) = {a if a > b; b otherwise}`.
    Max,
    /// `min(a, b) = {a if a < b; b otherwise}`.
    Min,
    /// `abs(x) = {x if x >= 0.0; -x otherwise}`.
    Abs,
    /// `sign(x) = {1 if x > 0.0; -1 if x < 0.0; 0 otherwise}`.
    Sign,
    /// `clamp(x, lo, hi) = min(max(x, lo), hi)`.
    Clamp,
    /// `clamp_symmetric(x, bound) = clamp(x, -bound, bound)`.
    ClampSymmetric,
    /// `relu(x) = max(x, 0)`.
    Relu,
    /// `leaky_relu(x, slope) = {x if x > 0.0; x * slope otherwise}`.
    LeakyRelu,
    /// `xor(a, b) = (a || b) && !(a && b)`.
    Xor,
    /// `nand(a, b) = !(a && b)`.
    Nand,
    /// `nor(a, b) = !(a || b)`.
    Nor,
    /// `implies(a, b) = !a || b`.
    Implies,
    /// `iff(a, b) = a == b`.
    Iff,
    /// `sigmoid(x) = 1.0 / (1.0 + exp(-x))`.
    Sigmoid,
    /// `silu(x) = x * sigmoid(x)`.
    Silu,
    /// `gelu(x) = (0.5 * x) * (1.0 + erf(x / sqrt(2.0)))`.
    Gelu,
    /// The natural exponential.
    Exp,
    /// The base-2 exponential.
    Exp2,
    /// The natural logarithm.
    Log,
    /// The base-2 logarithm.
    Log2,
    /// The base-10 logarithm.
    Log10,
    /// The square root.
    Sqrt,
    /// The sine.
    Sin,
    /// The cosine.
    Cos,
    /// The tangent.
    Tan,
    /// The inverse sine.
    Arcsin,
    /// The inverse cosine.
    Arccos,
    /// The inverse tangent.
    Arctan,
    /// The hyperbolic sine.
    Sinh,
    /// The hyperbolic cosine.
    Cosh,
    /// The hyperbolic tangent.
    Tanh,
    /// The error function.
    Erf,
    /// Rounding to the nearest integer.
    Round,
    /// Rounding toward negative infinity.
    Floor,
    /// Rounding toward positive infinity.
    Ceil,
}

/// Every built-in function in catalogue order: the composed functions, then
/// the native ones.
const FUNCTIONS: [BuiltinFunction; 35] = [
    BuiltinFunction::Max,
    BuiltinFunction::Min,
    BuiltinFunction::Abs,
    BuiltinFunction::Sign,
    BuiltinFunction::Clamp,
    BuiltinFunction::ClampSymmetric,
    BuiltinFunction::Relu,
    BuiltinFunction::LeakyRelu,
    BuiltinFunction::Xor,
    BuiltinFunction::Nand,
    BuiltinFunction::Nor,
    BuiltinFunction::Implies,
    BuiltinFunction::Iff,
    BuiltinFunction::Sigmoid,
    BuiltinFunction::Silu,
    BuiltinFunction::Gelu,
    BuiltinFunction::Exp,
    BuiltinFunction::Exp2,
    BuiltinFunction::Log,
    BuiltinFunction::Log2,
    BuiltinFunction::Log10,
    BuiltinFunction::Sqrt,
    BuiltinFunction::Sin,
    BuiltinFunction::Cos,
    BuiltinFunction::Tan,
    BuiltinFunction::Arcsin,
    BuiltinFunction::Arccos,
    BuiltinFunction::Arctan,
    BuiltinFunction::Sinh,
    BuiltinFunction::Cosh,
    BuiltinFunction::Tanh,
    BuiltinFunction::Erf,
    BuiltinFunction::Round,
    BuiltinFunction::Floor,
    BuiltinFunction::Ceil,
];

/// The number of composed functions, which lead [`FUNCTIONS`].
const COMPOSED_FUNCTION_COUNT: usize = 16;

impl BuiltinFunction {
    /// Return every built-in function in catalogue order: the 16 composed
    /// functions, then the 19 native ones.
    #[must_use]
    pub fn iter() -> impl ExactSizeIterator<Item = BuiltinFunction> + Clone {
        FUNCTIONS.into_iter()
    }

    /// Return the name a call refers to the function by, the variant name in
    /// lowercase words joined by underscores: `"max"`, `"clamp_symmetric"`,
    /// `"exp2"`, `"log10"`, and so on.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Max => "max",
            Self::Min => "min",
            Self::Abs => "abs",
            Self::Sign => "sign",
            Self::Clamp => "clamp",
            Self::ClampSymmetric => "clamp_symmetric",
            Self::Relu => "relu",
            Self::LeakyRelu => "leaky_relu",
            Self::Xor => "xor",
            Self::Nand => "nand",
            Self::Nor => "nor",
            Self::Implies => "implies",
            Self::Iff => "iff",
            Self::Sigmoid => "sigmoid",
            Self::Silu => "silu",
            Self::Gelu => "gelu",
            Self::Exp => "exp",
            Self::Exp2 => "exp2",
            Self::Log => "log",
            Self::Log2 => "log2",
            Self::Log10 => "log10",
            Self::Sqrt => "sqrt",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Tan => "tan",
            Self::Arcsin => "arcsin",
            Self::Arccos => "arccos",
            Self::Arctan => "arctan",
            Self::Sinh => "sinh",
            Self::Cosh => "cosh",
            Self::Tanh => "tanh",
            Self::Erf => "erf",
            Self::Round => "round",
            Self::Floor => "floor",
            Self::Ceil => "ceil",
        }
    }

    /// Return the sort of each parameter, in positional order.
    ///
    /// The Boolean connectives `xor`, `nand`, `nor`, `implies` and `iff`
    /// take two Booleans; every other function takes one to three reals.
    #[must_use]
    pub fn parameter_sorts(self) -> &'static [FunctionSort] {
        match self {
            Self::Max | Self::Min | Self::ClampSymmetric | Self::LeakyRelu => REAL_2,
            Self::Clamp => REAL_3,
            Self::Xor | Self::Nand | Self::Nor | Self::Implies | Self::Iff => BOOL_2,
            Self::Abs
            | Self::Sign
            | Self::Relu
            | Self::Sigmoid
            | Self::Silu
            | Self::Gelu
            | Self::Exp
            | Self::Exp2
            | Self::Log
            | Self::Log2
            | Self::Log10
            | Self::Sqrt
            | Self::Sin
            | Self::Cos
            | Self::Tan
            | Self::Arcsin
            | Self::Arccos
            | Self::Arctan
            | Self::Sinh
            | Self::Cosh
            | Self::Tanh
            | Self::Erf
            | Self::Round
            | Self::Floor
            | Self::Ceil => REAL_1,
        }
    }

    /// Return the sort of the function's result: [`FunctionSort::Bool`] for
    /// the Boolean connectives, [`FunctionSort::Int`] for `sign`, `round`,
    /// `floor` and `ceil`, and [`FunctionSort::Real`] for the rest.
    #[must_use]
    pub fn result_sort(self) -> FunctionSort {
        match self {
            Self::Xor | Self::Nand | Self::Nor | Self::Implies | Self::Iff => FunctionSort::Bool,
            Self::Sign | Self::Round | Self::Floor | Self::Ceil => FunctionSort::Int,
            Self::Max
            | Self::Min
            | Self::Abs
            | Self::Clamp
            | Self::ClampSymmetric
            | Self::Relu
            | Self::LeakyRelu
            | Self::Sigmoid
            | Self::Silu
            | Self::Gelu
            | Self::Exp
            | Self::Exp2
            | Self::Log
            | Self::Log2
            | Self::Log10
            | Self::Sqrt
            | Self::Sin
            | Self::Cos
            | Self::Tan
            | Self::Arcsin
            | Self::Arccos
            | Self::Arctan
            | Self::Sinh
            | Self::Cosh
            | Self::Tanh
            | Self::Erf => FunctionSort::Real,
        }
    }

    /// Return the definition of a composed built-in, or `None` for a native
    /// one.
    ///
    /// The composed functions are built on the first call in the process
    /// that asks for one, and every later call returns the same functions
    /// with the same parameters.
    #[must_use]
    pub fn composed(self) -> Option<&'static ComposedFunction> {
        COMPOSED_FUNCTIONS.get(self.catalogue_index())
    }

    /// Return the function's position in catalogue order, the order of
    /// [`iter`](Self::iter).
    fn catalogue_index(self) -> usize {
        match self {
            Self::Max => 0,
            Self::Min => 1,
            Self::Abs => 2,
            Self::Sign => 3,
            Self::Clamp => 4,
            Self::ClampSymmetric => 5,
            Self::Relu => 6,
            Self::LeakyRelu => 7,
            Self::Xor => 8,
            Self::Nand => 9,
            Self::Nor => 10,
            Self::Implies => 11,
            Self::Iff => 12,
            Self::Sigmoid => 13,
            Self::Silu => 14,
            Self::Gelu => 15,
            Self::Exp => 16,
            Self::Exp2 => 17,
            Self::Log => 18,
            Self::Log2 => 19,
            Self::Log10 => 20,
            Self::Sqrt => 21,
            Self::Sin => 22,
            Self::Cos => 23,
            Self::Tan => 24,
            Self::Arcsin => 25,
            Self::Arccos => 26,
            Self::Arctan => 27,
            Self::Sinh => 28,
            Self::Cosh => 29,
            Self::Tanh => 30,
            Self::Erf => 31,
            Self::Round => 32,
            Self::Floor => 33,
            Self::Ceil => 34,
        }
    }
}

impl_name_text!(BuiltinFunction, name, "built-in function");

/// A built-in real constant.
///
/// Serializes as its [`name`](Self::name), and deserializes, like
/// [`FromStr`](std::str::FromStr) parses, only from exactly that text.
///
/// # Examples
///
/// ```
/// use fhy_core::expr::builtins::BuiltinConstant;
///
/// assert_eq!(BuiltinConstant::Pi.value(), std::f64::consts::PI);
/// assert!("nan".parse::<BuiltinConstant>()?.value().is_nan());
/// # Ok::<(), fhy_core::expr::UnknownNameError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum BuiltinConstant {
    /// The `f64` nearest to pi.
    Pi,
    /// The `f64` nearest to e.
    E,
    /// Positive infinity.
    Inf,
    /// The positive quiet NaN with no payload.
    Nan,
}

/// Every built-in constant in catalogue order.
const CONSTANTS: [BuiltinConstant; 4] = [
    BuiltinConstant::Pi,
    BuiltinConstant::E,
    BuiltinConstant::Inf,
    BuiltinConstant::Nan,
];

impl BuiltinConstant {
    /// Return every built-in constant in catalogue order: `pi`, `e`, `inf`,
    /// `nan`.
    #[must_use]
    pub fn iter() -> impl ExactSizeIterator<Item = BuiltinConstant> + Clone {
        CONSTANTS.into_iter()
    }

    /// Return the constant's name: `"pi"`, `"e"`, `"inf"`, or `"nan"`.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Pi => "pi",
            Self::E => "e",
            Self::Inf => "inf",
            Self::Nan => "nan",
        }
    }

    /// Return the constant's sort, [`FunctionSort::Real`] for all four.
    #[must_use]
    pub fn sort(self) -> FunctionSort {
        match self {
            Self::Pi | Self::E | Self::Inf | Self::Nan => FunctionSort::Real,
        }
    }

    /// Return the constant's value: `pi` and `e` are the `f64` values nearest
    /// to pi and e, `inf` is positive infinity, and `nan` is the positive
    /// quiet NaN with no payload, the bits `0x7ff8_0000_0000_0000`.
    #[must_use]
    pub fn value(self) -> f64 {
        match self {
            Self::Pi => std::f64::consts::PI,
            Self::E => std::f64::consts::E,
            Self::Inf => f64::INFINITY,
            Self::Nan => f64::from_bits(0x7ff8_0000_0000_0000),
        }
    }
}

impl_name_text!(BuiltinConstant, name, "built-in constant");

/// Build a piecewise from cases whose conditions are comparisons and whose
/// list is non-empty, so the construction cannot be refused.
fn create_piecewise<C, V, O>(cases: impl IntoIterator<Item = (C, V)>, otherwise: O) -> Expression
where
    C: Into<Expression>,
    V: Into<Expression>,
    O: Into<Expression>,
{
    Expression::piecewise(cases, otherwise)
        .expect("a catalogue piecewise has at least one case and comparison conditions")
}

fn build_max_body([a, b]: &[Expression; 2]) -> Expression {
    create_piecewise([(a.greater(b), a)], b)
}

fn build_min_body([a, b]: &[Expression; 2]) -> Expression {
    create_piecewise([(a.less(b), a)], b)
}

fn build_abs_body([x]: &[Expression; 1]) -> Expression {
    create_piecewise([(x.greater_equal(0.0), x)], -x)
}

fn build_sign_body([x]: &[Expression; 1]) -> Expression {
    create_piecewise([(x.greater(0.0), 1_i64), (x.less(0.0), -1_i64)], 0_i64)
}

fn build_clamp_body([x, lo, hi]: &[Expression; 3]) -> Expression {
    Expression::call(
        BuiltinFunction::Min,
        [&Expression::call(BuiltinFunction::Max, [x, lo]), hi],
    )
}

fn build_clamp_symmetric_body([x, bound]: &[Expression; 2]) -> Expression {
    Expression::call(BuiltinFunction::Clamp, [x, &-bound, bound])
}

fn build_relu_body([x]: &[Expression; 1]) -> Expression {
    Expression::call(BuiltinFunction::Max, [x, &Expression::from(0_i64)])
}

fn build_leaky_relu_body([x, slope]: &[Expression; 2]) -> Expression {
    create_piecewise([(x.greater(0.0), x)], x * slope)
}

fn build_xor_body([a, b]: &[Expression; 2]) -> Expression {
    a.or(b).and(!a.and(b))
}

fn build_nand_body([a, b]: &[Expression; 2]) -> Expression {
    !a.and(b)
}

fn build_nor_body([a, b]: &[Expression; 2]) -> Expression {
    !a.or(b)
}

fn build_implies_body([a, b]: &[Expression; 2]) -> Expression {
    (!a).or(b)
}

fn build_iff_body([a, b]: &[Expression; 2]) -> Expression {
    a.equals(b)
}

fn build_sigmoid_body([x]: &[Expression; 1]) -> Expression {
    1.0 / (1.0 + Expression::call(BuiltinFunction::Exp, [-x]))
}

fn build_silu_body([x]: &[Expression; 1]) -> Expression {
    x * Expression::call(BuiltinFunction::Sigmoid, [x])
}

fn build_gelu_body([x]: &[Expression; 1]) -> Expression {
    0.5 * x
        * (1.0
            + Expression::call(
                BuiltinFunction::Erf,
                [x / Expression::call(BuiltinFunction::Sqrt, [2.0])],
            ))
}

/// Build the composed `function`: mint one parameter per name hint in
/// `parameter_names`, then build the body over references to them.
fn create_composed_function<const N: usize>(
    function: BuiltinFunction,
    parameter_names: [&str; N],
    build_body: fn(&[Expression; N]) -> Expression,
) -> ComposedFunction {
    let parameters = parameter_names.map(Identifier::new);
    // A clone keeps the identifier's id, so the body's references equal the
    // parameters the function keeps.
    let references = parameters
        .each_ref()
        .map(|parameter| Expression::from(parameter.clone()));
    ComposedFunction {
        function,
        parameters: Box::new(parameters),
        body: build_body(&references),
    }
}

/// Build the composed functions in catalogue order.
fn create_composed_functions() -> [ComposedFunction; COMPOSED_FUNCTION_COUNT] {
    [
        create_composed_function(BuiltinFunction::Max, ["a", "b"], build_max_body),
        create_composed_function(BuiltinFunction::Min, ["a", "b"], build_min_body),
        create_composed_function(BuiltinFunction::Abs, ["x"], build_abs_body),
        create_composed_function(BuiltinFunction::Sign, ["x"], build_sign_body),
        create_composed_function(BuiltinFunction::Clamp, ["x", "lo", "hi"], build_clamp_body),
        create_composed_function(
            BuiltinFunction::ClampSymmetric,
            ["x", "bound"],
            build_clamp_symmetric_body,
        ),
        create_composed_function(BuiltinFunction::Relu, ["x"], build_relu_body),
        create_composed_function(
            BuiltinFunction::LeakyRelu,
            ["x", "slope"],
            build_leaky_relu_body,
        ),
        create_composed_function(BuiltinFunction::Xor, ["a", "b"], build_xor_body),
        create_composed_function(BuiltinFunction::Nand, ["a", "b"], build_nand_body),
        create_composed_function(BuiltinFunction::Nor, ["a", "b"], build_nor_body),
        create_composed_function(BuiltinFunction::Implies, ["a", "b"], build_implies_body),
        create_composed_function(BuiltinFunction::Iff, ["a", "b"], build_iff_body),
        create_composed_function(BuiltinFunction::Sigmoid, ["x"], build_sigmoid_body),
        create_composed_function(BuiltinFunction::Silu, ["x"], build_silu_body),
        create_composed_function(BuiltinFunction::Gelu, ["x"], build_gelu_body),
    ]
}

/// The composed functions, built on first use.
static COMPOSED_FUNCTIONS: LazyLock<[ComposedFunction; COMPOSED_FUNCTION_COUNT]> =
    LazyLock::new(create_composed_functions);

/// A built-in function defined by an expression over its parameters.
///
/// The parameters are identifiers owned by the catalogue: they are created
/// once per process, never equal an identifier created elsewhere, and no two
/// functions share one. The body refers to no identifier other than the
/// parameters.
#[derive(Debug)]
pub struct ComposedFunction {
    function: BuiltinFunction,
    parameters: Box<[Identifier]>,
    body: Expression,
}

impl ComposedFunction {
    /// Return the built-in function this defines; its
    /// [`parameter_sorts`](BuiltinFunction::parameter_sorts) and
    /// [`result_sort`](BuiltinFunction::result_sort) are the signature.
    #[must_use]
    pub fn function(&self) -> BuiltinFunction {
        self.function
    }

    /// Return the parameters in positional order.
    ///
    /// Each parameter's name hint is its conventional name (`a`, `x`, `lo`,
    /// ...). The slice has one parameter per entry of the function's
    /// [`parameter_sorts`](BuiltinFunction::parameter_sorts).
    #[must_use]
    pub fn parameters(&self) -> &[Identifier] {
        &self.parameters
    }

    /// Return the body, an expression over [`parameters`](Self::parameters).
    #[must_use]
    pub fn body(&self) -> &Expression {
        &self.body
    }
}

#[cfg(test)]
mod tests {
    use super::super::callee::Callee;
    use super::super::node::ExpressionKind;
    use super::*;

    /// Test that the catalogue-order arrays list every variant exactly once,
    /// at the index an exhaustive `match` gives it, so a variant cannot be
    /// added without listing it.
    #[test]
    fn catalogue_order_array_lists_every_variant() {
        fn index_constant(constant: BuiltinConstant) -> usize {
            match constant {
                BuiltinConstant::Pi => 0,
                BuiltinConstant::E => 1,
                BuiltinConstant::Inf => 2,
                BuiltinConstant::Nan => 3,
            }
        }

        for (index, function) in FUNCTIONS.into_iter().enumerate() {
            assert_eq!(function.catalogue_index(), index, "{function:?}");
        }
        for (index, constant) in CONSTANTS.into_iter().enumerate() {
            assert_eq!(index_constant(constant), index, "{constant:?}");
        }
    }

    /// Test that the composed functions lead the catalogue, each built for
    /// its own variant, and that no native function has a definition.
    #[test]
    fn composed_functions_lead_the_catalogue_in_order() {
        for (index, function) in FUNCTIONS.into_iter().enumerate() {
            let composed = function.composed();

            assert_eq!(
                composed.is_some(),
                index < COMPOSED_FUNCTION_COUNT,
                "{function:?}"
            );
            if let Some(composed) = composed {
                assert_eq!(composed.function(), function);
                assert_eq!(
                    composed.parameters().len(),
                    function.parameter_sorts().len()
                );
            }
        }
    }

    #[test]
    fn composed_bodies_call_only_builtin_functions() {
        for composed in BuiltinFunction::iter().filter_map(BuiltinFunction::composed) {
            let mut pending = vec![composed.body()];
            while let Some(node) = pending.pop() {
                if let ExpressionKind::Call(call) = node.kind() {
                    assert!(matches!(call.callee(), Callee::Builtin(_)), "{node:?}");
                }
                pending.extend(node.children());
            }
        }
    }
}
