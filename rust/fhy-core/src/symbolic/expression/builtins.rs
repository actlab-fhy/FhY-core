//! The shipped catalogue of built-in functions and constants.
//!
//! The catalogue has three tables, each in a fixed order:
//!
//! - [`list_composed_functions`]: the 16 functions whose meaning is an
//!   expression over their parameters (`max`, `min`, `abs`, `sign`,
//!   `clamp`, `clamp_symmetric`, `relu`, `leaky_relu`, `xor`, `nand`, `nor`,
//!   `implies`, `iff`, `sigmoid`, `silu`, `gelu`);
//! - [`list_native_functions`]: the signatures of the 19 functions computed
//!   natively (`exp`, `exp2`, `log`, `log2`, `log10`, `sqrt`, `sin`, `cos`,
//!   `tan`, `arcsin`, `arccos`, `arctan`, `sinh`, `cosh`, `tanh`, `erf`,
//!   `round`, `floor`, `ceil`), each taking one real argument;
//! - [`list_native_constants`]: the four real constants `pi`, `e`, `inf`, `nan`.
//!
//! No name appears twice across the three tables. A composed body refers to
//! other built-in functions by name through call nodes, and to nothing but
//! its own parameters by identifier.
//!
//! The catalogue is data only. It registers nothing, and it does not
//! compute native functions.
//!
//! # Examples
//!
//! ```
//! use fhy_core::symbolic::expression::FunctionSort;
//! use fhy_core::symbolic::expression::builtins::find_composed_function;
//!
//! let relu = find_composed_function("relu").expect("relu is a composed built-in");
//! assert_eq!(relu.parameter_sorts(), [FunctionSort::Real]);
//! assert_eq!(relu.result_sort(), FunctionSort::Real);
//! ```

use std::sync::LazyLock;

use crate::identifier::Identifier;

use super::build::{IntoOperand, build_call, build_piecewise};
use super::literal::LiteralValue;
use super::node::Expression;
use super::operation::BinaryOperation;
use super::sort::FunctionSort;

/// Sorts of a function taking one real argument.
const REAL_1: &[FunctionSort; 1] = &[FunctionSort::Real];

/// Sorts of a function taking two real arguments.
const REAL_2: &[FunctionSort; 2] = &[FunctionSort::Real, FunctionSort::Real];

/// Sorts of a function taking three real arguments.
const REAL_3: &[FunctionSort; 3] = &[FunctionSort::Real, FunctionSort::Real, FunctionSort::Real];

/// Sorts of a function taking two Boolean arguments.
const BOOL_2: &[FunctionSort; 2] = &[FunctionSort::Bool, FunctionSort::Bool];

/// Build a piecewise from cases whose conditions are comparisons and whose
/// list is non-empty, so the construction cannot be refused.
fn create_piecewise<C, V, O>(cases: impl IntoIterator<Item = (C, V)>, otherwise: O) -> Expression
where
    C: IntoOperand,
    V: IntoOperand,
    O: IntoOperand,
{
    build_piecewise(cases, otherwise)
        .expect("a catalogue piecewise has at least one case and comparison conditions")
}

/// Build a call of the non-empty `function_name`, so the construction cannot
/// be refused.
fn create_call<I>(function_name: &str, arguments: I) -> Expression
where
    I: IntoIterator,
    I::Item: IntoOperand,
{
    build_call(function_name, arguments).expect("a catalogue function name is not empty")
}

/// `max(a, b) = {a if a > b; b otherwise}`.
fn build_max_body([a, b]: &[Expression; 2]) -> Expression {
    create_piecewise([(a.greater(b), a)], b)
}

/// `min(a, b) = {a if a < b; b otherwise}`.
fn build_min_body([a, b]: &[Expression; 2]) -> Expression {
    create_piecewise([(a.less(b), a)], b)
}

/// `abs(x) = {x if x >= 0.0; -x otherwise}`.
fn build_abs_body([x]: &[Expression; 1]) -> Expression {
    create_piecewise([(x.greater_equal(0.0), x)], -x)
}

/// `sign(x) = {1 if x > 0.0; -1 if x < 0.0; 0 otherwise}`, with integer
/// results.
fn build_sign_body([x]: &[Expression; 1]) -> Expression {
    create_piecewise([(x.greater(0.0), 1_i64), (x.less(0.0), -1_i64)], 0_i64)
}

/// `clamp(x, lo, hi) = min(max(x, lo), hi)`.
fn build_clamp_body([x, lo, hi]: &[Expression; 3]) -> Expression {
    create_call("min", [&create_call("max", [x, lo]), hi])
}

/// `clamp_symmetric(x, bound) = clamp(x, -bound, bound)`.
fn build_clamp_symmetric_body([x, bound]: &[Expression; 2]) -> Expression {
    create_call("clamp", [x, &-bound, bound])
}

/// `relu(x) = max(x, 0)`, with an integer zero.
fn build_relu_body([x]: &[Expression; 1]) -> Expression {
    create_call("max", [x, &Expression::from(LiteralValue::from(0_i64))])
}

/// `leaky_relu(x, slope) = {x if x > 0.0; x * slope otherwise}`.
fn build_leaky_relu_body([x, slope]: &[Expression; 2]) -> Expression {
    create_piecewise([(x.greater(0.0), x)], x * slope)
}

/// `xor(a, b) = (a || b) && !(a && b)`.
fn build_xor_body([a, b]: &[Expression; 2]) -> Expression {
    Expression::new_binary(
        BinaryOperation::LogicalAnd,
        Expression::new_binary(BinaryOperation::LogicalOr, a, b),
        Expression::new_binary(BinaryOperation::LogicalAnd, a, b).logical_not(),
    )
}

/// `nand(a, b) = !(a && b)`.
fn build_nand_body([a, b]: &[Expression; 2]) -> Expression {
    Expression::new_binary(BinaryOperation::LogicalAnd, a, b).logical_not()
}

/// `nor(a, b) = !(a || b)`.
fn build_nor_body([a, b]: &[Expression; 2]) -> Expression {
    Expression::new_binary(BinaryOperation::LogicalOr, a, b).logical_not()
}

/// `implies(a, b) = !a || b`.
fn build_implies_body([a, b]: &[Expression; 2]) -> Expression {
    Expression::new_binary(BinaryOperation::LogicalOr, a.logical_not(), b)
}

/// `iff(a, b) = a == b`.
fn build_iff_body([a, b]: &[Expression; 2]) -> Expression {
    a.equals(b)
}

/// `sigmoid(x) = 1.0 / (1.0 + exp(-x))`.
fn build_sigmoid_body([x]: &[Expression; 1]) -> Expression {
    1.0 / (1.0 + create_call("exp", [-x]))
}

/// `silu(x) = x * sigmoid(x)`.
fn build_silu_body([x]: &[Expression; 1]) -> Expression {
    x * create_call("sigmoid", [x])
}

/// `gelu(x) = (0.5 * x) * (1.0 + erf(x / sqrt(2.0)))`.
fn build_gelu_body([x]: &[Expression; 1]) -> Expression {
    0.5 * x * (1.0 + create_call("erf", [x / create_call("sqrt", [2.0])]))
}

/// Build the composed function `name`: mint one parameter per name hint in
/// `parameter_names`, then build the body over references to them.
fn create_composed_function<const N: usize>(
    name: &'static str,
    parameter_names: [&str; N],
    parameter_sorts: &'static [FunctionSort; N],
    result_sort: FunctionSort,
    build_body: fn(&[Expression; N]) -> Expression,
) -> ComposedFunction {
    let parameters = parameter_names.map(Identifier::new_unscoped);
    // A clone keeps the identifier's id, so the body's references equal the
    // parameters the function keeps.
    let references = parameters
        .each_ref()
        .map(|parameter| Expression::from(parameter.clone()));
    ComposedFunction {
        name,
        parameters: Box::new(parameters),
        parameter_sorts,
        result_sort,
        body: build_body(&references),
    }
}

/// Build the composed functions in catalogue order.
fn create_composed_functions() -> [ComposedFunction; 16] {
    [
        create_composed_function(
            "max",
            ["a", "b"],
            REAL_2,
            FunctionSort::Real,
            build_max_body,
        ),
        create_composed_function(
            "min",
            ["a", "b"],
            REAL_2,
            FunctionSort::Real,
            build_min_body,
        ),
        create_composed_function("abs", ["x"], REAL_1, FunctionSort::Real, build_abs_body),
        create_composed_function("sign", ["x"], REAL_1, FunctionSort::Int, build_sign_body),
        create_composed_function(
            "clamp",
            ["x", "lo", "hi"],
            REAL_3,
            FunctionSort::Real,
            build_clamp_body,
        ),
        create_composed_function(
            "clamp_symmetric",
            ["x", "bound"],
            REAL_2,
            FunctionSort::Real,
            build_clamp_symmetric_body,
        ),
        create_composed_function("relu", ["x"], REAL_1, FunctionSort::Real, build_relu_body),
        create_composed_function(
            "leaky_relu",
            ["x", "slope"],
            REAL_2,
            FunctionSort::Real,
            build_leaky_relu_body,
        ),
        create_composed_function(
            "xor",
            ["a", "b"],
            BOOL_2,
            FunctionSort::Bool,
            build_xor_body,
        ),
        create_composed_function(
            "nand",
            ["a", "b"],
            BOOL_2,
            FunctionSort::Bool,
            build_nand_body,
        ),
        create_composed_function(
            "nor",
            ["a", "b"],
            BOOL_2,
            FunctionSort::Bool,
            build_nor_body,
        ),
        create_composed_function(
            "implies",
            ["a", "b"],
            BOOL_2,
            FunctionSort::Bool,
            build_implies_body,
        ),
        create_composed_function(
            "iff",
            ["a", "b"],
            BOOL_2,
            FunctionSort::Bool,
            build_iff_body,
        ),
        create_composed_function(
            "sigmoid",
            ["x"],
            REAL_1,
            FunctionSort::Real,
            build_sigmoid_body,
        ),
        create_composed_function("silu", ["x"], REAL_1, FunctionSort::Real, build_silu_body),
        create_composed_function("gelu", ["x"], REAL_1, FunctionSort::Real, build_gelu_body),
    ]
}

/// The composed functions, built on first use.
static COMPOSED_FUNCTIONS: LazyLock<[ComposedFunction; 16]> =
    LazyLock::new(create_composed_functions);

/// Build the composed functions, and their parameters, if this is their first
/// use.
pub(crate) fn initialize_composed_functions() {
    LazyLock::force(&COMPOSED_FUNCTIONS);
}

/// The native function signatures, in catalogue order.
static NATIVE_FUNCTIONS: [NativeFunctionSignature; 19] = [
    NativeFunctionSignature::create("exp", FunctionSort::Real),
    NativeFunctionSignature::create("exp2", FunctionSort::Real),
    NativeFunctionSignature::create("log", FunctionSort::Real),
    NativeFunctionSignature::create("log2", FunctionSort::Real),
    NativeFunctionSignature::create("log10", FunctionSort::Real),
    NativeFunctionSignature::create("sqrt", FunctionSort::Real),
    NativeFunctionSignature::create("sin", FunctionSort::Real),
    NativeFunctionSignature::create("cos", FunctionSort::Real),
    NativeFunctionSignature::create("tan", FunctionSort::Real),
    NativeFunctionSignature::create("arcsin", FunctionSort::Real),
    NativeFunctionSignature::create("arccos", FunctionSort::Real),
    NativeFunctionSignature::create("arctan", FunctionSort::Real),
    NativeFunctionSignature::create("sinh", FunctionSort::Real),
    NativeFunctionSignature::create("cosh", FunctionSort::Real),
    NativeFunctionSignature::create("tanh", FunctionSort::Real),
    NativeFunctionSignature::create("erf", FunctionSort::Real),
    NativeFunctionSignature::create("round", FunctionSort::Int),
    NativeFunctionSignature::create("floor", FunctionSort::Int),
    NativeFunctionSignature::create("ceil", FunctionSort::Int),
];

/// The constants, in catalogue order.
static NATIVE_CONSTANTS: [NativeConstantSpec; 4] = [
    NativeConstantSpec::create("pi", std::f64::consts::PI),
    NativeConstantSpec::create("e", std::f64::consts::E),
    NativeConstantSpec::create("inf", f64::INFINITY),
    NativeConstantSpec::create("nan", f64::NAN),
];

/// A built-in function defined by an expression over its parameters.
///
/// The parameters are identifiers owned by the catalogue: they are created
/// once per process, never equal an identifier created elsewhere, and no two
/// functions share one. The body refers to no identifier other than the
/// parameters.
#[derive(Debug)]
pub struct ComposedFunction {
    name: &'static str,
    parameters: Box<[Identifier]>,
    parameter_sorts: &'static [FunctionSort],
    result_sort: FunctionSort,
    body: Expression,
}

impl ComposedFunction {
    /// Return the function's name, the name a call node uses to refer to it.
    #[must_use]
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Return the parameters in positional order.
    ///
    /// Each parameter's name hint is its conventional name (`a`, `x`, `lo`,
    /// ...). The slice has one parameter per entry of
    /// [`parameter_sorts`](Self::parameter_sorts).
    #[must_use]
    pub fn parameters(&self) -> &[Identifier] {
        &self.parameters
    }

    /// Return the sort of each parameter, in positional order.
    #[must_use]
    pub fn parameter_sorts(&self) -> &'static [FunctionSort] {
        self.parameter_sorts
    }

    /// Return the sort of the function's result.
    #[must_use]
    pub fn result_sort(&self) -> FunctionSort {
        self.result_sort
    }

    /// Return the body, an expression over [`parameters`](Self::parameters).
    #[must_use]
    pub fn body(&self) -> &Expression {
        &self.body
    }
}

/// The signature of a built-in function computed natively.
#[derive(Debug)]
pub struct NativeFunctionSignature {
    name: &'static str,
    parameter_sorts: &'static [FunctionSort],
    result_sort: FunctionSort,
}

impl NativeFunctionSignature {
    /// Construct the signature of `name`, taking one real argument and
    /// returning `result_sort`.
    const fn create(name: &'static str, result_sort: FunctionSort) -> Self {
        Self {
            name,
            parameter_sorts: REAL_1,
            result_sort,
        }
    }

    /// Return the function's name, the name a call node uses to refer to it.
    #[must_use]
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Return the sort of each parameter, in positional order.
    #[must_use]
    pub fn parameter_sorts(&self) -> &'static [FunctionSort] {
        self.parameter_sorts
    }

    /// Return the sort of the function's result: [`FunctionSort::Int`] for
    /// `round`, `floor` and `ceil`, and [`FunctionSort::Real`] for the rest.
    #[must_use]
    pub fn result_sort(&self) -> FunctionSort {
        self.result_sort
    }
}

/// A built-in named constant and its value.
#[derive(Debug)]
pub struct NativeConstantSpec {
    name: &'static str,
    sort: FunctionSort,
    value: f64,
}

impl NativeConstantSpec {
    /// Construct the real constant `name` holding `value`.
    const fn create(name: &'static str, value: f64) -> Self {
        Self {
            name,
            sort: FunctionSort::Real,
            value,
        }
    }

    /// Return the constant's name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Return the constant's sort.
    #[must_use]
    pub fn sort(&self) -> FunctionSort {
        self.sort
    }

    /// Return the constant's value: `pi` and `e` are the `f64` values nearest
    /// to pi and e, `inf` is positive infinity, and `nan` is a quiet NaN.
    #[must_use]
    pub fn value(&self) -> f64 {
        self.value
    }
}

/// Return the composed built-in functions in catalogue order: `max`, `min`,
/// `abs`, `sign`, `clamp`, `clamp_symmetric`, `relu`, `leaky_relu`, `xor`,
/// `nand`, `nor`, `implies`, `iff`, `sigmoid`, `silu`, `gelu`.
///
/// The functions are built on the first call to this function or to
/// [`find_composed_function`] in the process, and every later call returns
/// the same functions with the same parameters.
#[must_use]
pub fn list_composed_functions() -> &'static [ComposedFunction] {
    &*COMPOSED_FUNCTIONS
}

/// Return the native built-in function signatures in catalogue order:
/// `exp`, `exp2`, `log`, `log2`, `log10`, `sqrt`, `sin`, `cos`, `tan`,
/// `arcsin`, `arccos`, `arctan`, `sinh`, `cosh`, `tanh`, `erf`, `round`,
/// `floor`, `ceil`.
#[must_use]
pub fn list_native_functions() -> &'static [NativeFunctionSignature] {
    &NATIVE_FUNCTIONS
}

/// Return the built-in constants in catalogue order: `pi`, `e`, `inf`,
/// `nan`.
#[must_use]
pub fn list_native_constants() -> &'static [NativeConstantSpec] {
    &NATIVE_CONSTANTS
}

/// Return the composed built-in function named exactly `name`, or `None` if
/// no composed built-in has that name.
///
/// # Examples
///
/// ```
/// use fhy_core::symbolic::expression::builtins::find_composed_function;
///
/// let max = find_composed_function("max").expect("max is a composed built-in");
/// assert_eq!(max.name(), "max");
/// assert!(find_composed_function("exp").is_none());
/// ```
#[must_use]
pub fn find_composed_function(name: &str) -> Option<&'static ComposedFunction> {
    list_composed_functions()
        .iter()
        .find(|function| function.name == name)
}

/// Return the signature of the native built-in function named exactly
/// `name`, or `None` if no native built-in has that name.
#[must_use]
pub fn find_native_function(name: &str) -> Option<&'static NativeFunctionSignature> {
    list_native_functions()
        .iter()
        .find(|function| function.name == name)
}

/// Return the built-in constant named exactly `name`, or `None` if no
/// built-in constant has that name.
#[must_use]
pub fn find_native_constant(name: &str) -> Option<&'static NativeConstantSpec> {
    list_native_constants()
        .iter()
        .find(|constant| constant.name == name)
}
