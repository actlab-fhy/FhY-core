//! Parts of a serialized value that another implementation defines.
//!
//! The open variants of this crate's types hold parts defined outside it:
//! a [`Type::Extension`](crate::types::Type::Extension), a
//! [`DataType::Extension`](crate::types::DataType::Extension), a
//! [`Constraint::Custom`](crate::constraint::Constraint::Custom), a
//! [`ParamDomain::Custom`](crate::param::ParamDomain::Custom), an
//! opaque [`Value`](crate::constraint::Value) or member, and a search
//! space's [`Variable`](crate::search_space::Variable) or
//! [`Alternative`](crate::search_space::Alternative). Such a part
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

use std::any::Any;
use std::borrow::Cow;
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

/// Any error, boxed: the error a hook or a callback that another
/// implementation defines reports.
///
/// It is the implementor's own error: `?` converts any error type, a
/// `String` or a `&str` into it, and `downcast_ref` recovers the error the
/// implementation returned. Each module wraps it in its own error type,
/// such as [`ConstraintError::Custom`](crate::constraint::ConstraintError::Custom),
/// [`SolveError::Backend`](crate::solver::SolveError::Backend) or a
/// [`PassError`](crate::pass::PassError), which returns it as its source.
///
/// # Examples
///
/// ```
/// use fhy_core::expression::{Expression, LiteralValue};
/// use fhy_core::expression::pattern::Pattern;
/// use fhy_core::foreign::BoxError;
///
/// let refusing = Pattern::try_predicate(|_| Err(BoxError::from("no verdict")));
///
/// let result = refusing.matches(&Expression::from(LiteralValue::from(1)));
///
/// let error = result.expect_err("the predicate fails");
/// assert_eq!(error.to_string(), "no verdict");
/// ```
pub type BoxError = Box<dyn Error + Send + Sync + 'static>;

/// A value as [`Any`], so the implementation of an extension point can
/// recognize its own parts by downcasting.
///
/// Every `'static` type implements it, through the blanket impl, so an
/// implementor never writes `as_any`. Call it through the trait object,
/// `part.get().as_any()` or `ForeignPart::as_any(&*handle)`, never on an
/// `Arc` or a [`Part`] itself: the blanket impl applies to those too, and
/// would hand back the handle's own `Any`.
///
/// It stands in for trait upcasting to `dyn Any`, which needs Rust 1.86,
/// above the crate's minimum.
pub trait AsAny: Any {
    /// Return `self` as [`Any`].
    fn as_any(&self) -> &dyn Any;
}

impl<T: Any> AsAny for T {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// What every extension point's implementation is: a part another
/// implementation defines, which this crate holds in a [`Part`], names in
/// messages, and serializes as a [`Foreign`].
///
/// It is the supertrait of
/// [`OpaqueValue`](crate::constraint::OpaqueValue),
/// [`CustomConstraint`](crate::constraint::CustomConstraint),
/// [`CustomDomain`](crate::param::CustomDomain),
/// [`TypeExtension`](crate::types::TypeExtension),
/// [`DataTypeExtension`](crate::types::DataTypeExtension),
/// [`Variable`](crate::search_space::Variable) and
/// [`Alternative`](crate::search_space::Alternative). A part is shared
/// across threads, so it is `Send` and `Sync`:
///
/// ```compile_fail
/// use std::borrow::Cow;
/// use std::cell::Cell;
///
/// use fhy_core::foreign::ForeignPart;
///
/// #[derive(Debug)]
/// struct Counter(Cell<u32>);
///
/// impl ForeignPart for Counter {
///     fn type_name(&self) -> Cow<'_, str> {
///         Cow::Borrowed("Counter")
///     }
/// }
/// ```
pub trait ForeignPart: AsAny + fmt::Debug + Send + Sync {
    /// Return the name of the part's type, as messages and
    /// [`ForeignError::NoWireForm`] name it.
    fn type_name(&self) -> Cow<'_, str>;

    /// Return the part as a [`Foreign`], for serialization.
    ///
    /// # Errors
    ///
    /// The default returns [`ForeignError::NoWireForm`] with the
    /// [`type_name`](Self::type_name): the part cannot be serialized.
    fn to_foreign(&self) -> Result<Foreign, ForeignError> {
        Err(ForeignError::NoWireForm {
            type_name: self.type_name().into_owned(),
        })
    }
}

/// A shared handle to a part another implementation defines, such as a
/// `Part<dyn OpaqueValue>`.
///
/// Cloning shares the part. For each extension trait, `==` holds for the
/// same part, and otherwise asks the trait's `eq_part`, and `Hash` feeds the
/// trait's `hash_part`. `Debug` writes the part's own `Debug`.
pub struct Part<T: ?Sized>(Arc<T>);

impl<T: ?Sized> Part<T> {
    /// Wrap `part`, an implementation of the extension trait `T` names,
    /// such as a value of a type implementing
    /// [`OpaqueValue`](crate::constraint::OpaqueValue) for a
    /// `Part<dyn OpaqueValue>`.
    #[must_use]
    pub fn new(part: impl IntoPart<T>) -> Self {
        Self(part.into_arc())
    }

    /// Wrap the shared `part`.
    #[must_use]
    pub fn from_arc(part: Arc<T>) -> Self {
        Self(part)
    }

    /// Return the part.
    #[must_use]
    pub fn get(&self) -> &T {
        &self.0
    }

    /// Return whether `this` and `other` hold the same part.
    #[must_use]
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        Arc::ptr_eq(&this.0, &other.0)
    }
}

impl<T: ?Sized> Clone for Part<T> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl<T: ?Sized + fmt::Debug> fmt::Debug for Part<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// A value a [`Part<T>`] can hold: every implementation of an extension
/// trait is one for the trait object `T` of its trait, so
/// [`Part::new`] infers the part from the handle it builds.
pub trait IntoPart<T: ?Sized> {
    /// Return `self`, shared as the trait object `T`.
    fn into_arc(self) -> Arc<T>;
}

/// Implement [`IntoPart<dyn $Trait>`](IntoPart) for every implementor of
/// `$Trait`, and `PartialEq`, `Eq` and `Hash` for `Part<dyn $Trait>`
/// through the trait's `eq_part` and `hash_part`.
macro_rules! impl_part {
    ($Trait:ident) => {
        impl<U: $Trait + 'static> $crate::foreign::IntoPart<dyn $Trait> for U {
            fn into_arc(self) -> ::std::sync::Arc<dyn $Trait> {
                ::std::sync::Arc::new(self)
            }
        }

        impl ::std::cmp::PartialEq for $crate::foreign::Part<dyn $Trait> {
            /// Compare the same part equal, and otherwise ask
            #[doc = concat!("[`", stringify!($Trait), "::eq_part`].")]
            fn eq(&self, other: &Self) -> bool {
                Self::ptr_eq(self, other) || self.get().eq_part(other.get())
            }
        }

        impl ::std::cmp::Eq for $crate::foreign::Part<dyn $Trait> {}

        impl ::std::hash::Hash for $crate::foreign::Part<dyn $Trait> {
            /// Feed
            #[doc = concat!("[`", stringify!($Trait), "::hash_part`].")]
            fn hash<H: ::std::hash::Hasher>(&self, state: &mut H) {
                self.get().hash_part(state);
            }
        }
    };
}

pub(crate) use impl_part;

/// Return whether `part` and `other` are the same object, the default of
/// an extension trait's `eq_part`.
pub(crate) fn is_same_part<T: ?Sized, U: ?Sized>(part: &T, other: &U) -> bool {
    std::ptr::addr_eq(std::ptr::from_ref(part), std::ptr::from_ref(other))
}

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
        source: BoxError,
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
    Invalid(BoxError),
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
