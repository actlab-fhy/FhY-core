//! The errors of building registry entries, registering them, and inlining
//! through a registry.

use std::error::Error;
use std::fmt;

use crate::expression::builtins::BuiltinConstant;
use crate::expression::callee::{Callee, FunctionName};
use crate::expression::error::PiecewiseError;
use crate::expression::literal::LiteralValue;
use crate::expression::sort::FunctionSort;
use crate::identifier::Identifier;

/// Write `count` followed by `noun`, pluralized with an `s` unless `count`
/// is one.
fn write_count(f: &mut fmt::Formatter<'_>, count: usize, noun: &str) -> fmt::Result {
    let plural = if count == 1 { "" } else { "s" };
    write!(f, "{count} {noun}{plural}")
}

/// A [`FunctionDefinition`](super::FunctionDefinition) that could not be
/// built.
///
/// # Examples
///
/// ```
/// use fhy_core::expression::registry::{FunctionDefinition, FunctionDefinitionError};
/// use fhy_core::expression::{Expression, FunctionName, FunctionSort};
/// use fhy_core::identifier::Identifier;
///
/// let x = Identifier::new("x");
/// let error = FunctionDefinition::try_new(
///     FunctionName::try_new("f")?,
///     [x.clone()],
///     [],
///     FunctionSort::Real,
///     Expression::from(x),
/// )
/// .expect_err("one parameter needs one sort");
/// assert_eq!(error.to_string(), r#"function "f" has 1 parameter but 0 parameter sorts"#);
/// # Ok::<(), fhy_core::expression::FunctionNameError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum FunctionDefinitionError {
    /// The parameters and their sorts differ in number.
    ///
    /// Displays as `function "f" has 1 parameter but 2 parameter sorts`.
    SortCountMismatch {
        /// The function being defined.
        function: FunctionName,
        /// How many parameters it has.
        parameters: usize,
        /// How many parameter sorts it has.
        sorts: usize,
    },
    /// A parameter occurs twice in the parameter list.
    ///
    /// Displays as `function "f" repeats the parameter "x"`, naming the
    /// parameter by its name hint.
    RepeatedParameter {
        /// The function being defined.
        function: FunctionName,
        /// The first parameter that occurs twice.
        parameter: Identifier,
    },
}

impl fmt::Display for FunctionDefinitionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SortCountMismatch {
                function,
                parameters,
                sorts,
            } => {
                write!(f, "function {:?} has ", function.as_str())?;
                write_count(f, *parameters, "parameter")?;
                f.write_str(" but ")?;
                write_count(f, *sorts, "parameter sort")
            }
            Self::RepeatedParameter {
                function,
                parameter,
            } => write!(
                f,
                "function {:?} repeats the parameter {:?}",
                function.as_str(),
                parameter.name_hint()
            ),
        }
    }
}

impl Error for FunctionDefinitionError {}

/// A [`NativeConstant`](super::NativeConstant) whose sort does not accept
/// its value.
///
/// Displays as `constant "c" of sort nat cannot hold -1`, with the value's
/// literal text.
///
/// # Examples
///
/// ```
/// use fhy_core::expression::registry::NativeConstant;
/// use fhy_core::expression::{FunctionName, FunctionSort, LiteralValue};
///
/// let error = NativeConstant::try_new(FunctionName::try_new("flag")?, FunctionSort::Int, true)
///     .expect_err("a Boolean is no integer");
/// assert_eq!(error.sort(), FunctionSort::Int);
/// assert_eq!(error.value(), &LiteralValue::from(true));
/// assert_eq!(error.to_string(), r#"constant "flag" of sort int cannot hold true"#);
/// # Ok::<(), fhy_core::expression::FunctionNameError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ConstantValueError {
    name: FunctionName,
    sort: FunctionSort,
    value: LiteralValue,
}

impl ConstantValueError {
    /// Build the error of the constant `name` of `sort` refusing `value`.
    pub(super) fn new(name: FunctionName, sort: FunctionSort, value: LiteralValue) -> Self {
        Self { name, sort, value }
    }

    /// Return the name of the constant being declared.
    #[must_use]
    pub fn name(&self) -> &FunctionName {
        &self.name
    }

    /// Return the sort the constant was declared with.
    #[must_use]
    pub fn sort(&self) -> FunctionSort {
        self.sort
    }

    /// Return the value the sort does not accept.
    #[must_use]
    pub fn value(&self) -> &LiteralValue {
        &self.value
    }
}

impl fmt::Display for ConstantValueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "constant {:?} of sort {} cannot hold {}",
            self.name.as_str(),
            self.sort,
            self.value
        )
    }
}

impl Error for ConstantValueError {}

/// An entry a [`FunctionRegistry`](super::FunctionRegistry) refused to
/// register.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RegistrationError {
    /// The name is already registered, for a function or a constant.
    ///
    /// Displays as `"f" is already registered`.
    NameTaken(FunctionName),
    /// The name is a built-in constant's, which names only that constant.
    ///
    /// Displays as `"pi" is the name of a built-in constant`.
    BuiltinConstantName(BuiltinConstant),
    /// The function's body refers to identifiers that are neither its
    /// parameters, nor constants of the registry, nor built-in constants.
    ///
    /// Displays as `function "f" captures identifiers that are not its
    /// parameters: x, y`, listing the identifiers by name hint.
    CapturedIdentifiers {
        /// The function being registered.
        function: FunctionName,
        /// The captured identifiers, ordered by name hint and then by id.
        identifiers: Vec<Identifier>,
    },
}

impl fmt::Display for RegistrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NameTaken(name) => write!(f, "{:?} is already registered", name.as_str()),
            Self::BuiltinConstantName(constant) => {
                write!(
                    f,
                    "{:?} is the name of a built-in constant",
                    constant.name()
                )
            }
            Self::CapturedIdentifiers {
                function,
                identifiers,
            } => {
                write!(
                    f,
                    "function {:?} captures identifiers that are not its parameters: ",
                    function.as_str()
                )?;
                for (index, identifier) in identifiers.iter().enumerate() {
                    if index > 0 {
                        f.write_str(", ")?;
                    }
                    f.write_str(identifier.name_hint())?;
                }
                Ok(())
            }
        }
    }
}

impl Error for RegistrationError {}

/// A call [`FunctionRegistry::inline`](super::FunctionRegistry::inline)
/// could not inline, or keep.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum InlineError {
    /// A call names a function that is neither built in nor registered.
    ///
    /// Displays as `no function is registered under "f"`.
    UnknownFunction(FunctionName),
    /// A call passes a number of arguments other than the function's
    /// parameter count.
    ///
    /// Displays as `"f" takes 2 arguments but the call passes 1`.
    ArityMismatch {
        /// The function called.
        callee: Callee,
        /// How many parameters the function has.
        expected: usize,
        /// How many arguments the call passes.
        actual: usize,
    },
    /// A call names a registered constant, which cannot be called.
    ///
    /// Displays as `"c" is a constant, not a function`.
    NotCallable(FunctionName),
    /// A function is called again while its own body is being inlined,
    /// directly or through other functions.
    ///
    /// Displays as `function "f" is recursive and cannot be inlined`.
    Recursive(FunctionName),
    /// Inlining put a literal other than a Boolean in a piecewise case
    /// condition, where a body used a parameter as a condition.
    ///
    /// Displays as `inlining built an invalid piecewise`; the source is the
    /// piecewise's refusal.
    Piecewise(PiecewiseError),
}

impl fmt::Display for InlineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownFunction(name) => {
                write!(f, "no function is registered under {:?}", name.as_str())
            }
            Self::ArityMismatch {
                callee,
                expected,
                actual,
            } => {
                write!(f, "{:?} takes ", callee.name())?;
                write_count(f, *expected, "argument")?;
                write!(f, " but the call passes {actual}")
            }
            Self::NotCallable(name) => {
                write!(f, "{:?} is a constant, not a function", name.as_str())
            }
            Self::Recursive(name) => write!(
                f,
                "function {:?} is recursive and cannot be inlined",
                name.as_str()
            ),
            Self::Piecewise(_) => f.write_str("inlining built an invalid piecewise"),
        }
    }
}

impl Error for InlineError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Piecewise(error) => Some(error),
            Self::UnknownFunction(_)
            | Self::ArityMismatch { .. }
            | Self::NotCallable(_)
            | Self::Recursive(_) => None,
        }
    }
}
