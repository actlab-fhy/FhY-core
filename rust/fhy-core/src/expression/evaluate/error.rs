//! The errors of evaluation and of folding.

use std::error::Error;
use std::fmt;

use crate::expression::BigInt;
use crate::expression::builtins::BuiltinFunction;
use crate::expression::callee::{Callee, FunctionName};
use crate::expression::error::{NonBooleanLogicalOperandError, PiecewiseError};
use crate::expression::literal::{Decimal, LiteralValue};
use crate::expression::node::Expression;
use crate::expression::registry::InlineError;
use crate::expression::sort::FunctionSort;
use crate::foreign::BoxError;
use crate::identifier::Identifier;

/// Why an unbound identifier is probably not the variable it looks like.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NearMiss {
    /// Its name hint is the name of a built-in or registered function, so a
    /// call was probably dropped.
    NamesFunction,
    /// Its name hint is the name of a built-in or registered constant, but
    /// it is not the constant's identifier.
    SharesConstantName,
}

/// Why one lane of an evaluation has no value.
///
/// A lane failure is kept with its lane, and raised only if the lane
/// reaches the result: a piecewise selecting another branch in that lane,
/// or a connective whose other operand decides that lane, discards it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum LaneFailure {
    /// An integer operation overflowed the 64-bit range.
    ///
    /// Displays as `integer overflow`.
    IntegerOverflow,
    /// An integer was floor-divided by zero, or its remainder taken.
    ///
    /// Displays as `integer division by zero`.
    DivisionByZero,
    /// An integer was raised to a negative integer power.
    ///
    /// Displays as `an integer raised to a negative integer power`.
    NegativeIntegerExponent,
    /// A NaN or an infinity reached an integer-sorted result.
    ///
    /// Displays as `a non-finite value cast to an integer`.
    NonFiniteCast,
    /// A finite value outside the 64-bit range reached an integer-sorted
    /// result.
    ///
    /// Displays as `a value outside the 64-bit range cast to an integer`.
    OutOfRangeCast,
}

impl LaneFailure {
    /// Every lane failure, in the order of their codes.
    pub(super) const ALL: [Self; 5] = [
        Self::IntegerOverflow,
        Self::DivisionByZero,
        Self::NegativeIntegerExponent,
        Self::NonFiniteCast,
        Self::OutOfRangeCast,
    ];

    /// Return the failure's code, its index in [`ALL`](Self::ALL).
    pub(super) fn code(self) -> u32 {
        match self {
            Self::IntegerOverflow => 0,
            Self::DivisionByZero => 1,
            Self::NegativeIntegerExponent => 2,
            Self::NonFiniteCast => 3,
            Self::OutOfRangeCast => 4,
        }
    }
}

impl fmt::Display for LaneFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::IntegerOverflow => "integer overflow",
            Self::DivisionByZero => "integer division by zero",
            Self::NegativeIntegerExponent => "an integer raised to a negative integer power",
            Self::NonFiniteCast => "a non-finite value cast to an integer",
            Self::OutOfRangeCast => "a value outside the 64-bit range cast to an integer",
        })
    }
}

/// An expression that could not be evaluated.
#[derive(Debug)]
#[non_exhaustive]
pub enum EvaluationError {
    /// Inlining the expression failed.
    ///
    /// Displays as the inliner's error.
    Inline(InlineError),
    /// A Boolean position provably holds a number.
    ///
    /// Displays as the screen's error.
    IllTyped(NonBooleanLogicalOperandError),
    /// The environment binds the identifiers of constants the expression
    /// refers to, sorted by id.
    ///
    /// Displays as `cannot bind the constants pi::48: a constant's value is
    /// fixed`.
    BoundNativeConstant(Vec<Identifier>),
    /// An identifier is neither bound nor a constant.
    ///
    /// Displays as `identifier "x" is not bound`, followed by the near miss.
    Unbound {
        /// The identifier.
        identifier: Identifier,
        /// Why it is probably not the variable it looks like, if it is not.
        near_miss: Option<NearMiss>,
    },
    /// A decimal literal has no binary float equal to it.
    ///
    /// Displays as `decimal 0.1 has no exact binary float`.
    InexactDecimal(Decimal),
    /// An integer literal, or an integer constant's value, lies outside the
    /// 64-bit range.
    ///
    /// Displays as `integer 9223372036854775808 is outside the 64-bit
    /// range`.
    IntegerOutOfRange(BigInt),
    /// A call of a function the evaluator cannot compute: a native user
    /// function, which has no implementation here.
    ///
    /// Displays as `function "f" has no implementation the evaluator can
    /// run`.
    Unsupported(Callee),
    /// A Boolean is used as a number: in arithmetic, an ordering, a sign, a
    /// numeric argument, or an equality with a number.
    ///
    /// Displays as `a boolean is used as a number in (x + true)`.
    BooleanArithmetic(Expression),
    /// A number is used as a Boolean: an operand of a connective or of a
    /// negation, or a piecewise condition, which the screen did not refuse.
    ///
    /// Displays as `a number is used as a boolean in (x && 1)`.
    NumberAsBoolean(Expression),
    /// A piecewise's branches mix Booleans and numbers.
    ///
    /// Displays as `the branches of {...} mix booleans and numbers`.
    MixedBranches(Expression),
    /// Two operands' shapes do not broadcast together.
    ///
    /// Displays as `shapes [2] and [3] do not broadcast`.
    Shape {
        /// The first operand's shape.
        left: Vec<usize>,
        /// The second operand's shape.
        right: Vec<usize>,
    },
    /// A lane of the result failed.
    ///
    /// Displays as `integer overflow in (x * y)`, the failure and the node
    /// where it occurred, for the first failed lane in C order.
    Lane {
        /// Why the lane failed.
        failure: LaneFailure,
        /// The node whose operation failed.
        node: Expression,
    },
    /// An array kernel plugged into the evaluator failed.
    ///
    /// Displays as `the array kernel of exp failed`; the kernel's error is
    /// the [`source`](Error::source).
    Kernel {
        /// The built-in function the kernel computes.
        function: BuiltinFunction,
        /// The kernel's error.
        source: BoxError,
    },
}

impl fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Inline(error) => write!(f, "{error}"),
            Self::IllTyped(error) => write!(f, "{error}"),
            Self::BoundNativeConstant(identifiers) => {
                f.write_str("cannot bind the constants ")?;
                write_identifiers(f, identifiers)?;
                f.write_str(": a constant's value is fixed")
            }
            Self::Unbound {
                identifier,
                near_miss,
            } => {
                let name = identifier.name_hint();
                write!(f, "identifier {name:?} is not bound")?;
                match near_miss {
                    None => Ok(()),
                    Some(NearMiss::NamesFunction) => write!(
                        f,
                        "; it names a function, so call it as {name}(...) or bind it"
                    ),
                    Some(NearMiss::SharesConstantName) => write!(
                        f,
                        "; it shares its name with the constant {name:?} but is not \
                         its identifier"
                    ),
                }
            }
            Self::InexactDecimal(decimal) => {
                write!(f, "decimal {decimal} has no exact binary float")
            }
            Self::IntegerOutOfRange(integer) => {
                write!(f, "integer {integer} is outside the 64-bit range")
            }
            Self::Unsupported(callee) => write!(
                f,
                "function {:?} has no implementation the evaluator can run",
                callee.name()
            ),
            Self::BooleanArithmetic(node) => write!(f, "a boolean is used as a number in {node}"),
            Self::NumberAsBoolean(node) => write!(f, "a number is used as a boolean in {node}"),
            Self::MixedBranches(node) => {
                write!(f, "the branches of {node} mix booleans and numbers")
            }
            Self::Shape { left, right } => {
                write!(f, "shapes {left:?} and {right:?} do not broadcast")
            }
            Self::Lane { failure, node } => write!(f, "{failure} in {node}"),
            Self::Kernel { function, .. } => {
                write!(f, "the array kernel of {} failed", function.name())
            }
        }
    }
}

impl Error for EvaluationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Inline(error) => error.source(),
            Self::IllTyped(error) => error.source(),
            Self::Kernel { source, .. } => Some(source.as_ref()),
            Self::BoundNativeConstant(_)
            | Self::Unbound { .. }
            | Self::InexactDecimal(_)
            | Self::IntegerOutOfRange(_)
            | Self::Unsupported(_)
            | Self::BooleanArithmetic(_)
            | Self::NumberAsBoolean(_)
            | Self::MixedBranches(_)
            | Self::Shape { .. }
            | Self::Lane { .. } => None,
        }
    }
}

/// A call that folding could not fold, or a reference it could not resolve.
#[derive(Debug)]
#[non_exhaustive]
pub enum FoldError {
    /// A call names a function that is neither built in nor registered.
    ///
    /// Displays as `no function is registered under "f"`.
    UnknownFunction(FunctionName),
    /// A call names a constant, registered or built in.
    ///
    /// Displays as `"c" is a constant, not a function`.
    NotCallable(FunctionName),
    /// A native call with literal arguments passes the wrong number of them.
    ///
    /// Displays as `"f" takes 2 arguments but the call passes 1`.
    Arity {
        /// The function called.
        callee: Callee,
        /// How many parameters the function has.
        expected: usize,
        /// How many arguments the call passes.
        actual: usize,
    },
    /// A native call's literal argument does not have its parameter's sort.
    ///
    /// Displays as `argument 0 of "f" must be real, got true`.
    ArgumentSort {
        /// The function called.
        callee: Callee,
        /// The zero-based position of the argument.
        position: usize,
        /// The parameter's sort.
        sort: FunctionSort,
        /// The argument.
        argument: LiteralValue,
    },
    /// A native user function returned a value its result sort does not
    /// accept.
    ///
    /// Displays as `native function "f" returned 1.5, which is not of its
    /// result sort int`.
    ResultSort {
        /// The function.
        function: FunctionName,
        /// Its declared result sort.
        sort: FunctionSort,
        /// The value it returned.
        value: LiteralValue,
    },
    /// A decimal argument has no binary float equal to it.
    ///
    /// Displays as `decimal 0.1 has no exact binary float`.
    InexactDecimal(Decimal),
    /// A native built-in's integer-sorted result is a NaN or an infinity.
    ///
    /// Displays as `floor(inf) has no integer value`.
    NonFiniteCast {
        /// The function.
        function: BuiltinFunction,
        /// Its value before the cast.
        value: f64,
    },
    /// Folding put a literal other than a Boolean in a piecewise case
    /// condition, where a native call with literal arguments was the
    /// condition.
    ///
    /// Displays as `folding built an invalid piecewise`; the source is the
    /// piecewise's refusal.
    Piecewise(PiecewiseError),
    /// A native user function's implementation failed.
    ///
    /// Displays as `native function "f" failed`; the implementation's error
    /// is the [`source`](Error::source).
    Native {
        /// The function.
        function: FunctionName,
        /// The implementation's error.
        source: BoxError,
    },
}

impl fmt::Display for FoldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownFunction(name) => {
                write!(f, "no function is registered under {:?}", name.as_str())
            }
            Self::NotCallable(name) => {
                write!(f, "{:?} is a constant, not a function", name.as_str())
            }
            Self::Arity {
                callee,
                expected,
                actual,
            } => write!(
                f,
                "{:?} takes {expected} argument{} but the call passes {actual}",
                callee.name(),
                if *expected == 1 { "" } else { "s" }
            ),
            Self::ArgumentSort {
                callee,
                position,
                sort,
                argument,
            } => write!(
                f,
                "argument {position} of {:?} must be {sort}, got {argument}",
                callee.name()
            ),
            Self::ResultSort {
                function,
                sort,
                value,
            } => write!(
                f,
                "native function {:?} returned {value}, which is not of its result sort {sort}",
                function.as_str()
            ),
            Self::InexactDecimal(decimal) => {
                write!(f, "decimal {decimal} has no exact binary float")
            }
            Self::NonFiniteCast { function, value } => {
                write!(f, "{}({value}) has no integer value", function.name())
            }
            Self::Piecewise(_) => f.write_str("folding built an invalid piecewise"),
            Self::Native { function, .. } => {
                write!(f, "native function {:?} failed", function.as_str())
            }
        }
    }
}

impl Error for FoldError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Native { source, .. } => Some(source.as_ref()),
            Self::Piecewise(error) => Some(error),
            Self::UnknownFunction(_)
            | Self::NotCallable(_)
            | Self::Arity { .. }
            | Self::ArgumentSort { .. }
            | Self::ResultSort { .. }
            | Self::InexactDecimal(_)
            | Self::NonFiniteCast { .. } => None,
        }
    }
}

/// Write `identifiers` as `name::id`, separated by commas.
fn write_identifiers(f: &mut fmt::Formatter<'_>, identifiers: &[Identifier]) -> fmt::Result {
    for (index, identifier) in identifiers.iter().enumerate() {
        if index > 0 {
            f.write_str(", ")?;
        }
        write!(f, "{}::{}", identifier.name_hint(), identifier.id())?;
    }
    Ok(())
}
