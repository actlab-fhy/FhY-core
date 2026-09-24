//! What a call calls: a built-in function or a function named by the user.
//!
//! A [`Callee`] is either [`Callee::Builtin`], one of the catalogue's
//! [`BuiltinFunction`]s, or [`Callee::Named`], a [`FunctionName`] no
//! built-in function has. Built-in names are reserved, so every call of a
//! built-in has one representation.

use std::error::Error;
use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use serde::de::{self, Deserializer, Visitor};
use serde::{Deserialize, Serialize, Serializer};

use super::builtins::BuiltinFunction;

/// The function a call applies.
///
/// Serializes externally tagged, as `{"builtin": "max"}` or
/// `{"named": "f"}`. [`FromStr`] routes a built-in function's name to
/// [`Builtin`](Self::Builtin) and any other non-empty name to
/// [`Named`](Self::Named).
///
/// # Examples
///
/// ```
/// use fhy_core::expr::builtins::BuiltinFunction;
/// use fhy_core::expr::{Callee, FunctionNameError};
///
/// assert_eq!("max".parse::<Callee>()?, Callee::Builtin(BuiltinFunction::Max));
/// assert!(matches!("f".parse::<Callee>()?, Callee::Named(name) if name.as_str() == "f"));
/// assert_eq!("".parse::<Callee>(), Err(FunctionNameError::Empty));
/// # Ok::<(), FunctionNameError>(())
/// ```
#[expect(
    clippy::exhaustive_enums,
    reason = "a call names either a built-in or a user function, and passes match both"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Callee {
    /// A built-in function of the catalogue.
    Builtin(BuiltinFunction),
    /// A function the catalogue does not know, by name.
    Named(FunctionName),
}

impl Callee {
    /// Return the name the call refers to the function by.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Builtin(function) => function.name(),
            Self::Named(name) => name.as_str(),
        }
    }
}

impl From<BuiltinFunction> for Callee {
    fn from(function: BuiltinFunction) -> Self {
        Self::Builtin(function)
    }
}

impl From<FunctionName> for Callee {
    fn from(name: FunctionName) -> Self {
        Self::Named(name)
    }
}

impl FromStr for Callee {
    type Err = FunctionNameError;

    /// Parse the built-in function named `name`, or else the user function
    /// named `name`.
    ///
    /// # Errors
    ///
    /// Returns [`FunctionNameError::Empty`] if `name` is empty.
    fn from_str(name: &str) -> Result<Self, FunctionNameError> {
        match name.parse::<BuiltinFunction>() {
            Ok(function) => Ok(Self::Builtin(function)),
            Err(_not_builtin) => FunctionName::try_new(name).map(Self::Named),
        }
    }
}

impl fmt::Display for Callee {
    /// Write the [`name`](Self::name).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The name of a function the built-in catalogue does not know: not empty,
/// and not the name of any [`BuiltinFunction`].
///
/// Serializes as a string, and deserializes from a string that
/// [`try_new`](Self::try_new) accepts.
///
/// # Examples
///
/// ```
/// use fhy_core::expr::builtins::BuiltinFunction;
/// use fhy_core::expr::{FunctionName, FunctionNameError};
///
/// assert_eq!(FunctionName::try_new("softplus")?.as_str(), "softplus");
/// assert_eq!(
///     FunctionName::try_new("max"),
///     Err(FunctionNameError::Builtin(BuiltinFunction::Max))
/// );
/// # Ok::<(), FunctionNameError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionName(Arc<str>);

impl FunctionName {
    /// Accept `name` as the name of a user function.
    ///
    /// # Errors
    ///
    /// Returns [`FunctionNameError::Empty`] if `name` is empty, and
    /// [`FunctionNameError::Builtin`] naming the built-in function if
    /// `name` is a built-in function's name.
    pub fn try_new(name: &str) -> Result<Self, FunctionNameError> {
        if name.is_empty() {
            return Err(FunctionNameError::Empty);
        }
        if let Ok(function) = name.parse::<BuiltinFunction>() {
            return Err(FunctionNameError::Builtin(function));
        }
        Ok(Self(Arc::from(name)))
    }

    /// Return the name.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for FunctionName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Serialize for FunctionName {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0)
    }
}

struct FunctionNameVisitor;

impl Visitor<'_> for FunctionNameVisitor {
    type Value = FunctionName;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a function name")
    }

    fn visit_str<E: de::Error>(self, name: &str) -> Result<FunctionName, E> {
        FunctionName::try_new(name).map_err(E::custom)
    }
}

impl<'de> Deserialize<'de> for FunctionName {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_str(FunctionNameVisitor)
    }
}

/// A function name that could not be accepted.
///
/// # Examples
///
/// ```
/// use fhy_core::expr::builtins::BuiltinFunction;
/// use fhy_core::expr::FunctionNameError;
///
/// assert_eq!(FunctionNameError::Empty.to_string(), "function name is empty");
/// assert_eq!(
///     FunctionNameError::Builtin(BuiltinFunction::Max).to_string(),
///     "function name `max` is a built-in function"
/// );
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FunctionNameError {
    /// The name is empty.
    ///
    /// Displays as `function name is empty`.
    Empty,
    /// The name is the name of a built-in function, which a call names
    /// through [`Callee::Builtin`] instead.
    ///
    /// Displays as ``function name `{name}` is a built-in function``.
    Builtin(BuiltinFunction),
}

impl fmt::Display for FunctionNameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("function name is empty"),
            Self::Builtin(function) => {
                write!(f, "function name `{function}` is a built-in function")
            }
        }
    }
}

impl Error for FunctionNameError {}
