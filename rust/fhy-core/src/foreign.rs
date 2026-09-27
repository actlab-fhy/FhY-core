//! Parts of a serialized value that another implementation defines.
//!
//! The open variants of this crate's types hold parts defined outside it:
//! a [`Type::Extension`](crate::types::Type::Extension), a
//! [`DataType::Extension`](crate::types::DataType::Extension), a
//! [`Constraint::Custom`](crate::constraint::Constraint::Custom), a
//! [`ParamDomain::Custom`](crate::param::ParamDomain::Custom), and an
//! opaque [`Value`](crate::constraint::Value) or member. Such a part
//! serializes as a [`Foreign`]: the type id its implementation registered
//! under, and its own payload as text. Serializing asks the part for it
//! through its trait's `to_foreign`; a part whose implementation answers
//! [`ForeignError::NoWireForm`], the default, cannot be serialized.
//!
//! Deserializing a type that can hold a foreign part goes through its wire
//! form, a plain data type with the part left as a [`Foreign`], whose
//! `build` method takes a [`Resolve`]r that turns each part back into its
//! implementation. The types' own `Deserialize` impls build with
//! [`NoForeign`], which refuses every part by its type id, so a program
//! without the implementations still reads every value this crate defines.
//!
//! # Examples
//!
//! ```
//! use fhy_core::foreign::{Foreign, ForeignError, NoForeign, Resolve};
//!
//! let part = Foreign::new("pkg.even", r#"{"modulus":2}"#);
//! let text = serde_json::to_string(&part)?;
//! assert_eq!(text, r#"{"type_id":"pkg.even","data":"{\"modulus\":2}"}"#);
//!
//! let refused: Result<u8, ForeignError> = NoForeign.resolve(&part);
//! assert_eq!(
//!     refused.unwrap_err().to_string(),
//!     "no implementation for the foreign part `pkg.even`"
//! );
//! # Ok::<(), serde_json::Error>(())
//! ```

use std::error::Error;
use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

/// A serialized part another implementation defines: the type id it is
/// registered under and its payload, as that implementation's own text.
///
/// Serializes as `{"type_id": .., "data": ..}`, both strings. The crate
/// never reads `data`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Foreign {
    type_id: Arc<str>,
    data: Arc<str>,
}

impl Foreign {
    /// Return the part registered as `type_id` whose payload is `data`.
    #[must_use]
    pub fn new(type_id: impl Into<Arc<str>>, data: impl Into<Arc<str>>) -> Self {
        Self {
            type_id: type_id.into(),
            data: data.into(),
        }
    }

    /// Return the type id the part's implementation is registered under.
    #[must_use]
    pub fn type_id(&self) -> &str {
        &self.type_id
    }

    /// Return the part's payload, its implementation's own text.
    #[must_use]
    pub fn data(&self) -> &str {
        &self.data
    }
}

/// Turns a [`Foreign`] part back into its implementation, a `T`.
pub trait Resolve<T> {
    /// Return the implementation of `foreign`.
    ///
    /// # Errors
    ///
    /// Returns [`ForeignError::Unresolved`] when no implementation is
    /// registered under the part's type id, and
    /// [`ForeignError::Failed`] when the implementation refuses its
    /// payload.
    fn resolve(&self, foreign: &Foreign) -> Result<T, ForeignError>;
}

/// The resolver that knows no implementation, and refuses every part.
#[expect(
    clippy::exhaustive_structs,
    reason = "a stateless unit type that callers name as a value"
)]
#[derive(Debug, Clone, Copy, Default)]
pub struct NoForeign;

impl<T> Resolve<T> for NoForeign {
    fn resolve(&self, foreign: &Foreign) -> Result<T, ForeignError> {
        Err(ForeignError::Unresolved {
            type_id: foreign.type_id().to_owned(),
        })
    }
}

/// Why a foreign part could not be serialized or resolved.
#[derive(Debug)]
#[non_exhaustive]
pub enum ForeignError {
    /// No implementation is registered under the part's type id.
    Unresolved {
        /// The part's type id.
        type_id: String,
    },
    /// The part's implementation has no wire form.
    NoWireForm {
        /// The name of the part's type.
        type_name: String,
    },
    /// The part's implementation failed to write or read its payload.
    Failed {
        /// The part's type id, or its type's name when it has no id.
        type_id: String,
        /// The implementation's error.
        source: Box<dyn Error + Send + Sync + 'static>,
    },
}

impl fmt::Display for ForeignError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unresolved { type_id } => {
                write!(f, "no implementation for the foreign part `{type_id}`")
            }
            Self::NoWireForm { type_name } => write!(f, "`{type_name}` has no wire form"),
            Self::Failed { type_id, .. } => write!(f, "the foreign part `{type_id}` failed"),
        }
    }
}

impl Error for ForeignError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Failed { source, .. } => Some(source.as_ref()),
            _ => None,
        }
    }
}

/// Why a wire form could not be built into its value.
///
/// `Display` and `source` are those of the underlying error.
#[derive(Debug)]
#[non_exhaustive]
pub enum BuildError {
    /// A foreign part could not be resolved.
    Foreign(ForeignError),
    /// The data breaks an invariant of the value; the error is the one its
    /// constructor returns.
    Invalid(Box<dyn Error + Send + Sync + 'static>),
}

impl BuildError {
    /// Return the error of data a constructor refuses with `error`.
    pub fn invalid(error: impl Error + Send + Sync + 'static) -> Self {
        Self::Invalid(Box::new(error))
    }
}

impl From<ForeignError> for BuildError {
    fn from(error: ForeignError) -> Self {
        Self::Foreign(error)
    }
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Foreign(error) => error.fmt(f),
            Self::Invalid(error) => error.fmt(f),
        }
    }
}

impl Error for BuildError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Foreign(error) => error.source(),
            Self::Invalid(error) => error.source(),
        }
    }
}
