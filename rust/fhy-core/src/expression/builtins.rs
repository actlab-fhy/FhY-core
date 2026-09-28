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
//! The catalogue registers nothing. It computes the native functions,
//! through [`BuiltinFunction::native_value`], for the evaluator of
//! [`evaluate`](super::evaluate). Look a function up by name with
//! [`FromStr`](std::str::FromStr), and list the composed functions with
//! `BuiltinFunction::iter().filter_map(BuiltinFunction::composed)`.
//!
//! # Examples
//!
//! ```
//! use fhy_core::expression::FunctionSort;
//! use fhy_core::expression::builtins::BuiltinFunction;
//!
//! let relu: BuiltinFunction = "relu".parse()?;
//! assert_eq!(relu.parameter_sorts(), [FunctionSort::Real]);
//! assert_eq!(relu.result_sort(), FunctionSort::Real);
//! assert!(relu.composed().is_some());
//! assert!(BuiltinFunction::Exp.composed().is_none());
//! # Ok::<(), fhy_core::error::UnknownNameError>(())
//! ```

use std::sync::LazyLock;

use serde::{Deserialize, Serialize};

use crate::identifier::Identifier;
use crate::identifier::reserved::{self, ReservedIdentifier};

use super::node::Expression;
use super::sort::FunctionSort;
use crate::error::impl_name_text;

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
/// use fhy_core::expression::builtins::BuiltinFunction;
///
/// assert_eq!(BuiltinFunction::ClampSymmetric.name(), "clamp_symmetric");
/// assert_eq!("log10".parse::<BuiltinFunction>()?, BuiltinFunction::Log10);
/// assert_eq!(BuiltinFunction::iter().len(), 35);
/// # Ok::<(), fhy_core::error::UnknownNameError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum BuiltinFunction {
    /// `max(a, b) = {a if a > b || a != a; b otherwise}`: a NaN operand, on
    /// either side, is the result, as in `NumPy`'s `maximum`.
    Max,
    /// `min(a, b) = {a if a < b || a != a; b otherwise}`: a NaN operand, on
    /// either side, is the result, as in `NumPy`'s `minimum`.
    Min,
    /// `abs(x) = {x if x > 0.0; 0 - x otherwise}`: both zeros give `0.0`,
    /// and a NaN gives a NaN.
    Abs,
    /// `sign(x) = {1 if x > 0.0; -1 if x < 0.0; 0 otherwise}`, so
    /// `sign(nan)` is `0`.
    Sign,
    /// `clamp(x, lo, hi) = min(max(x, lo), hi)`, so a NaN argument
    /// propagates.
    Clamp,
    /// `clamp_symmetric(x, bound) = clamp(x, -bound, bound)`.
    ClampSymmetric,
    /// `relu(x) = max(x, 0)`, so `relu(nan)` is NaN.
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

/// One function's catalogue entry: its name and sorts, and either how to
/// compute it natively or how to build its composed definition.
struct FunctionEntry {
    function: BuiltinFunction,
    name: &'static str,
    parameter_sorts: &'static [FunctionSort],
    result_sort: FunctionSort,
    /// `Some` for a native function, computing its value at one real
    /// argument.
    native: Option<fn(f64) -> f64>,
    /// `Some` for a composed function.
    composed: Option<ComposedSpec>,
}

/// How to build a composed function's definition: its parameters' name
/// hints, and how to build its body over references to them.
#[derive(Clone, Copy)]
struct ComposedSpec {
    parameter_names: &'static [&'static str],
    build_body: fn(&[Expression]) -> Expression,
}

/// The catalogue in catalogue order: the 16 composed functions, then the 19
/// native ones. [`BuiltinFunction::catalogue_index`] indexes it.
const CATALOGUE: [FunctionEntry; 35] = [
    FunctionEntry {
        function: BuiltinFunction::Max,
        name: "max",
        parameter_sorts: REAL_2,
        result_sort: FunctionSort::Real,
        native: None,
        composed: Some(ComposedSpec {
            parameter_names: &["a", "b"],
            build_body: build_max_body,
        }),
    },
    FunctionEntry {
        function: BuiltinFunction::Min,
        name: "min",
        parameter_sorts: REAL_2,
        result_sort: FunctionSort::Real,
        native: None,
        composed: Some(ComposedSpec {
            parameter_names: &["a", "b"],
            build_body: build_min_body,
        }),
    },
    FunctionEntry {
        function: BuiltinFunction::Abs,
        name: "abs",
        parameter_sorts: REAL_1,
        result_sort: FunctionSort::Real,
        native: None,
        composed: Some(ComposedSpec {
            parameter_names: &["x"],
            build_body: build_abs_body,
        }),
    },
    FunctionEntry {
        function: BuiltinFunction::Sign,
        name: "sign",
        parameter_sorts: REAL_1,
        result_sort: FunctionSort::Int,
        native: None,
        composed: Some(ComposedSpec {
            parameter_names: &["x"],
            build_body: build_sign_body,
        }),
    },
    FunctionEntry {
        function: BuiltinFunction::Clamp,
        name: "clamp",
        parameter_sorts: REAL_3,
        result_sort: FunctionSort::Real,
        native: None,
        composed: Some(ComposedSpec {
            parameter_names: &["x", "lo", "hi"],
            build_body: build_clamp_body,
        }),
    },
    FunctionEntry {
        function: BuiltinFunction::ClampSymmetric,
        name: "clamp_symmetric",
        parameter_sorts: REAL_2,
        result_sort: FunctionSort::Real,
        native: None,
        composed: Some(ComposedSpec {
            parameter_names: &["x", "bound"],
            build_body: build_clamp_symmetric_body,
        }),
    },
    FunctionEntry {
        function: BuiltinFunction::Relu,
        name: "relu",
        parameter_sorts: REAL_1,
        result_sort: FunctionSort::Real,
        native: None,
        composed: Some(ComposedSpec {
            parameter_names: &["x"],
            build_body: build_relu_body,
        }),
    },
    FunctionEntry {
        function: BuiltinFunction::LeakyRelu,
        name: "leaky_relu",
        parameter_sorts: REAL_2,
        result_sort: FunctionSort::Real,
        native: None,
        composed: Some(ComposedSpec {
            parameter_names: &["x", "slope"],
            build_body: build_leaky_relu_body,
        }),
    },
    FunctionEntry {
        function: BuiltinFunction::Xor,
        name: "xor",
        parameter_sorts: BOOL_2,
        result_sort: FunctionSort::Bool,
        native: None,
        composed: Some(ComposedSpec {
            parameter_names: &["a", "b"],
            build_body: build_xor_body,
        }),
    },
    FunctionEntry {
        function: BuiltinFunction::Nand,
        name: "nand",
        parameter_sorts: BOOL_2,
        result_sort: FunctionSort::Bool,
        native: None,
        composed: Some(ComposedSpec {
            parameter_names: &["a", "b"],
            build_body: build_nand_body,
        }),
    },
    FunctionEntry {
        function: BuiltinFunction::Nor,
        name: "nor",
        parameter_sorts: BOOL_2,
        result_sort: FunctionSort::Bool,
        native: None,
        composed: Some(ComposedSpec {
            parameter_names: &["a", "b"],
            build_body: build_nor_body,
        }),
    },
    FunctionEntry {
        function: BuiltinFunction::Implies,
        name: "implies",
        parameter_sorts: BOOL_2,
        result_sort: FunctionSort::Bool,
        native: None,
        composed: Some(ComposedSpec {
            parameter_names: &["a", "b"],
            build_body: build_implies_body,
        }),
    },
    FunctionEntry {
        function: BuiltinFunction::Iff,
        name: "iff",
        parameter_sorts: BOOL_2,
        result_sort: FunctionSort::Bool,
        native: None,
        composed: Some(ComposedSpec {
            parameter_names: &["a", "b"],
            build_body: build_iff_body,
        }),
    },
    FunctionEntry {
        function: BuiltinFunction::Sigmoid,
        name: "sigmoid",
        parameter_sorts: REAL_1,
        result_sort: FunctionSort::Real,
        native: None,
        composed: Some(ComposedSpec {
            parameter_names: &["x"],
            build_body: build_sigmoid_body,
        }),
    },
    FunctionEntry {
        function: BuiltinFunction::Silu,
        name: "silu",
        parameter_sorts: REAL_1,
        result_sort: FunctionSort::Real,
        native: None,
        composed: Some(ComposedSpec {
            parameter_names: &["x"],
            build_body: build_silu_body,
        }),
    },
    FunctionEntry {
        function: BuiltinFunction::Gelu,
        name: "gelu",
        parameter_sorts: REAL_1,
        result_sort: FunctionSort::Real,
        native: None,
        composed: Some(ComposedSpec {
            parameter_names: &["x"],
            build_body: build_gelu_body,
        }),
    },
    FunctionEntry {
        function: BuiltinFunction::Exp,
        name: "exp",
        parameter_sorts: REAL_1,
        result_sort: FunctionSort::Real,
        native: Some(f64::exp),
        composed: None,
    },
    FunctionEntry {
        function: BuiltinFunction::Exp2,
        name: "exp2",
        parameter_sorts: REAL_1,
        result_sort: FunctionSort::Real,
        native: Some(f64::exp2),
        composed: None,
    },
    FunctionEntry {
        function: BuiltinFunction::Log,
        name: "log",
        parameter_sorts: REAL_1,
        result_sort: FunctionSort::Real,
        native: Some(f64::ln),
        composed: None,
    },
    FunctionEntry {
        function: BuiltinFunction::Log2,
        name: "log2",
        parameter_sorts: REAL_1,
        result_sort: FunctionSort::Real,
        native: Some(f64::log2),
        composed: None,
    },
    FunctionEntry {
        function: BuiltinFunction::Log10,
        name: "log10",
        parameter_sorts: REAL_1,
        result_sort: FunctionSort::Real,
        native: Some(f64::log10),
        composed: None,
    },
    FunctionEntry {
        function: BuiltinFunction::Sqrt,
        name: "sqrt",
        parameter_sorts: REAL_1,
        result_sort: FunctionSort::Real,
        native: Some(f64::sqrt),
        composed: None,
    },
    FunctionEntry {
        function: BuiltinFunction::Sin,
        name: "sin",
        parameter_sorts: REAL_1,
        result_sort: FunctionSort::Real,
        native: Some(f64::sin),
        composed: None,
    },
    FunctionEntry {
        function: BuiltinFunction::Cos,
        name: "cos",
        parameter_sorts: REAL_1,
        result_sort: FunctionSort::Real,
        native: Some(f64::cos),
        composed: None,
    },
    FunctionEntry {
        function: BuiltinFunction::Tan,
        name: "tan",
        parameter_sorts: REAL_1,
        result_sort: FunctionSort::Real,
        native: Some(f64::tan),
        composed: None,
    },
    FunctionEntry {
        function: BuiltinFunction::Arcsin,
        name: "arcsin",
        parameter_sorts: REAL_1,
        result_sort: FunctionSort::Real,
        native: Some(f64::asin),
        composed: None,
    },
    FunctionEntry {
        function: BuiltinFunction::Arccos,
        name: "arccos",
        parameter_sorts: REAL_1,
        result_sort: FunctionSort::Real,
        native: Some(f64::acos),
        composed: None,
    },
    FunctionEntry {
        function: BuiltinFunction::Arctan,
        name: "arctan",
        parameter_sorts: REAL_1,
        result_sort: FunctionSort::Real,
        native: Some(f64::atan),
        composed: None,
    },
    FunctionEntry {
        function: BuiltinFunction::Sinh,
        name: "sinh",
        parameter_sorts: REAL_1,
        result_sort: FunctionSort::Real,
        native: Some(f64::sinh),
        composed: None,
    },
    FunctionEntry {
        function: BuiltinFunction::Cosh,
        name: "cosh",
        parameter_sorts: REAL_1,
        result_sort: FunctionSort::Real,
        native: Some(f64::cosh),
        composed: None,
    },
    FunctionEntry {
        function: BuiltinFunction::Tanh,
        name: "tanh",
        parameter_sorts: REAL_1,
        result_sort: FunctionSort::Real,
        native: Some(f64::tanh),
        composed: None,
    },
    FunctionEntry {
        function: BuiltinFunction::Erf,
        name: "erf",
        parameter_sorts: REAL_1,
        result_sort: FunctionSort::Real,
        native: Some(libm::erf),
        composed: None,
    },
    FunctionEntry {
        function: BuiltinFunction::Round,
        name: "round",
        parameter_sorts: REAL_1,
        result_sort: FunctionSort::Int,
        native: Some(f64::round_ties_even),
        composed: None,
    },
    FunctionEntry {
        function: BuiltinFunction::Floor,
        name: "floor",
        parameter_sorts: REAL_1,
        result_sort: FunctionSort::Int,
        native: Some(f64::floor),
        composed: None,
    },
    FunctionEntry {
        function: BuiltinFunction::Ceil,
        name: "ceil",
        parameter_sorts: REAL_1,
        result_sort: FunctionSort::Int,
        native: Some(f64::ceil),
        composed: None,
    },
];

impl BuiltinFunction {
    /// Return every built-in function in catalogue order: the 16 composed
    /// functions, then the 19 native ones.
    #[must_use]
    pub fn iter() -> impl ExactSizeIterator<Item = Self> + Clone {
        CATALOGUE.iter().map(|entry| entry.function)
    }

    /// Return the name a call refers to the function by, the variant name in
    /// lowercase words joined by underscores: `"max"`, `"clamp_symmetric"`,
    /// `"exp2"`, `"log10"`, and so on.
    #[must_use]
    pub fn name(self) -> &'static str {
        CATALOGUE[self.catalogue_index()].name
    }

    /// Return the sort of each parameter, in positional order.
    ///
    /// The Boolean connectives `xor`, `nand`, `nor`, `implies` and `iff`
    /// take two Booleans; every other function takes one to three reals.
    #[must_use]
    pub fn parameter_sorts(self) -> &'static [FunctionSort] {
        CATALOGUE[self.catalogue_index()].parameter_sorts
    }

    /// Return the sort of the function's result: [`FunctionSort::Bool`] for
    /// the Boolean connectives, [`FunctionSort::Int`] for `sign`, `round`,
    /// `floor` and `ceil`, and [`FunctionSort::Real`] for the rest.
    #[must_use]
    pub fn result_sort(self) -> FunctionSort {
        CATALOGUE[self.catalogue_index()].result_sort
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

    /// Return the value of a native built-in at `argument`, before its
    /// result sort's cast, or `None` for a composed built-in.
    ///
    /// The results are IEEE's: `sqrt` of a negative number and `log` of a
    /// negative number are NaN, `log(0)` is negative infinity, and `exp` of
    /// a large number is infinity. `round` rounds half to even. `erf` is
    /// `libm`'s, and every other function is the standard library's `f64`
    /// method, which calls the platform's math library, so the last bits
    /// may differ between platforms.
    ///
    /// # Examples
    ///
    /// ```
    /// use fhy_core::expression::builtins::BuiltinFunction;
    ///
    /// assert_eq!(BuiltinFunction::Round.native_value(2.5), Some(2.0));
    /// assert!(BuiltinFunction::Sqrt.native_value(-1.0).is_some_and(f64::is_nan));
    /// assert_eq!(BuiltinFunction::Relu.native_value(1.0), None);
    /// ```
    #[must_use]
    pub fn native_value(self, argument: f64) -> Option<f64> {
        let native = CATALOGUE[self.catalogue_index()].native?;
        Some(native(argument))
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
/// use fhy_core::expression::builtins::BuiltinConstant;
///
/// assert_eq!(BuiltinConstant::Pi.value(), std::f64::consts::PI);
/// assert!("nan".parse::<BuiltinConstant>()?.value().is_nan());
/// # Ok::<(), fhy_core::error::UnknownNameError>(())
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
    pub fn iter() -> impl ExactSizeIterator<Item = Self> + Clone {
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

    /// Return the identifier an expression refers to the constant by.
    ///
    /// Each constant's identifier has a fixed id from the reserved block
    /// (`pi` 48, `e` 49, `inf` 50, `nan` 51) and the constant's name as its
    /// name hint, so it is the same in every process and in every payload,
    /// and the counter never issues it to another identifier.
    #[must_use]
    pub fn identifier(self) -> &'static Identifier {
        &CONSTANT_IDENTIFIERS[self.catalogue_index()]
    }

    /// Return the constant `identifier` refers to, or `None` if it is no
    /// built-in constant's [`identifier`](Self::identifier).
    ///
    /// The id decides: an identifier merely named `pi` is no constant.
    #[must_use]
    pub fn of_identifier(identifier: &Identifier) -> Option<Self> {
        CONSTANTS
            .into_iter()
            .find(|constant| constant.reserved_entry().id() == identifier.id())
    }

    /// Return the constant's position in catalogue order.
    fn catalogue_index(self) -> usize {
        match self {
            Self::Pi => 0,
            Self::E => 1,
            Self::Inf => 2,
            Self::Nan => 3,
        }
    }

    /// Return the constant's entry of the reserved-id table.
    fn reserved_entry(self) -> ReservedIdentifier {
        match self {
            Self::Pi => reserved::PI_CONSTANT,
            Self::E => reserved::E_CONSTANT,
            Self::Inf => reserved::INF_CONSTANT,
            Self::Nan => reserved::NAN_CONSTANT,
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

/// The constants' identifiers in catalogue order, built from the reserved
/// table on first use; building them draws nothing from the counter.
static CONSTANT_IDENTIFIERS: LazyLock<[Identifier; 4]> =
    LazyLock::new(|| CONSTANTS.map(|constant| Identifier::reserved(constant.reserved_entry())));

/// Build a piecewise from cases whose conditions are comparisons, or
/// connectives of comparisons, and whose list is non-empty, so the
/// construction cannot be refused.
fn create_piecewise<C, V, O>(cases: impl IntoIterator<Item = (C, V)>, otherwise: O) -> Expression
where
    C: Into<Expression>,
    V: Into<Expression>,
    O: Into<Expression>,
{
    Expression::piecewise(cases, otherwise)
        .expect("a catalogue piecewise has at least one case and comparison conditions")
}

/// `a if (a > b || a != a) else b`: a NaN operand, on either side, is the
/// result, as `NumPy`'s `maximum` has it.
fn build_max_body(args: &[Expression]) -> Expression {
    let [a, b] = args else {
        unreachable!("max takes 2 arguments")
    };
    create_piecewise([(a.greater(b).or(a.not_equals(a)), a)], b)
}

/// `a if (a < b || a != a) else b`: a NaN operand, on either side, is the
/// result, as `NumPy`'s `minimum` has it.
fn build_min_body(args: &[Expression]) -> Expression {
    let [a, b] = args else {
        unreachable!("min takes 2 arguments")
    };
    create_piecewise([(a.less(b).or(a.not_equals(a)), a)], b)
}

/// `x if x > 0.0 else 0 - x`: both zeros give the positive zero, and a NaN
/// gives a NaN. The subtraction from an integer zero keeps an integer
/// operand's kind.
fn build_abs_body(args: &[Expression]) -> Expression {
    let [x] = args else {
        unreachable!("abs takes 1 argument")
    };
    create_piecewise([(x.greater(0.0), x)], Expression::from(0_i64) - x)
}

fn build_sign_body(args: &[Expression]) -> Expression {
    let [x] = args else {
        unreachable!("sign takes 1 argument")
    };
    create_piecewise([(x.greater(0.0), 1_i64), (x.less(0.0), -1_i64)], 0_i64)
}

fn build_clamp_body(args: &[Expression]) -> Expression {
    let [x, lo, hi] = args else {
        unreachable!("clamp takes 3 arguments")
    };
    Expression::call(
        BuiltinFunction::Min,
        [&Expression::call(BuiltinFunction::Max, [x, lo]), hi],
    )
}

fn build_clamp_symmetric_body(args: &[Expression]) -> Expression {
    let [x, bound] = args else {
        unreachable!("clamp_symmetric takes 2 arguments")
    };
    Expression::call(BuiltinFunction::Clamp, [x, &-bound, bound])
}

fn build_relu_body(args: &[Expression]) -> Expression {
    let [x] = args else {
        unreachable!("relu takes 1 argument")
    };
    Expression::call(BuiltinFunction::Max, [x, &Expression::from(0_i64)])
}

fn build_leaky_relu_body(args: &[Expression]) -> Expression {
    let [x, slope] = args else {
        unreachable!("leaky_relu takes 2 arguments")
    };
    create_piecewise([(x.greater(0.0), x)], x * slope)
}

fn build_xor_body(args: &[Expression]) -> Expression {
    let [a, b] = args else {
        unreachable!("xor takes 2 arguments")
    };
    a.or(b).and(!a.and(b))
}

fn build_nand_body(args: &[Expression]) -> Expression {
    let [a, b] = args else {
        unreachable!("nand takes 2 arguments")
    };
    !a.and(b)
}

fn build_nor_body(args: &[Expression]) -> Expression {
    let [a, b] = args else {
        unreachable!("nor takes 2 arguments")
    };
    !a.or(b)
}

fn build_implies_body(args: &[Expression]) -> Expression {
    let [a, b] = args else {
        unreachable!("implies takes 2 arguments")
    };
    (!a).or(b)
}

fn build_iff_body(args: &[Expression]) -> Expression {
    let [a, b] = args else {
        unreachable!("iff takes 2 arguments")
    };
    a.equals(b)
}

fn build_sigmoid_body(args: &[Expression]) -> Expression {
    let [x] = args else {
        unreachable!("sigmoid takes 1 argument")
    };
    1.0 / (1.0 + Expression::call(BuiltinFunction::Exp, [-x]))
}

fn build_silu_body(args: &[Expression]) -> Expression {
    let [x] = args else {
        unreachable!("silu takes 1 argument")
    };
    x * Expression::call(BuiltinFunction::Sigmoid, [x])
}

fn build_gelu_body(args: &[Expression]) -> Expression {
    let [x] = args else {
        unreachable!("gelu takes 1 argument")
    };
    0.5 * x
        * (1.0
            + Expression::call(
                BuiltinFunction::Erf,
                [x / Expression::call(BuiltinFunction::Sqrt, [2.0])],
            ))
}

/// Build the composed `function`: mint one parameter per name hint in
/// `parameter_names`, then build the body over references to them.
fn create_composed_function(
    function: BuiltinFunction,
    parameter_names: &'static [&'static str],
    build_body: fn(&[Expression]) -> Expression,
) -> ComposedFunction {
    let parameters: Box<[Identifier]> = parameter_names
        .iter()
        .copied()
        .map(Identifier::new)
        .collect();
    // A clone keeps the identifier's id, so the body's references equal the
    // parameters the function keeps.
    let references: Box<[Expression]> = parameters
        .iter()
        .map(|parameter| Expression::from(parameter.clone()))
        .collect();
    ComposedFunction {
        function,
        parameters,
        body: build_body(&references),
    }
}

/// The composed functions, in catalogue order, built from [`CATALOGUE`] on
/// first use.
static COMPOSED_FUNCTIONS: LazyLock<Vec<ComposedFunction>> = LazyLock::new(|| {
    CATALOGUE
        .iter()
        .filter_map(|entry| {
            let spec = entry.composed?;
            Some(create_composed_function(
                entry.function,
                spec.parameter_names,
                spec.build_body,
            ))
        })
        .collect()
});

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

    /// Test that the catalogue lists every variant exactly once, at the
    /// index an exhaustive `match` gives it, so a variant cannot be added
    /// without listing it.
    #[test]
    fn catalogue_order_array_lists_every_variant() {
        for (index, entry) in CATALOGUE.iter().enumerate() {
            assert_eq!(
                entry.function.catalogue_index(),
                index,
                "{:?}",
                entry.function
            );
        }
        for (index, constant) in CONSTANTS.into_iter().enumerate() {
            assert_eq!(constant.catalogue_index(), index, "{constant:?}");
        }
    }

    /// Test that the composed functions lead the catalogue, each built for
    /// its own variant, and that no native function has a definition.
    #[test]
    fn composed_functions_lead_the_catalogue_in_order() {
        let composed_count = CATALOGUE
            .iter()
            .filter(|entry| entry.composed.is_some())
            .count();
        for (index, function) in BuiltinFunction::iter().enumerate() {
            let composed = function.composed();

            assert_eq!(composed.is_some(), index < composed_count, "{function:?}");
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
