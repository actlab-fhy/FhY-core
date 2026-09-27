//! Types and data types defined outside this crate.
//!
//! A [`Type::Extension`](super::Type::Extension) holds a
//! [`TypeExtension`], and a [`DataType::Extension`](super::DataType::Extension)
//! a [`DataTypeExtension`]: a Rust IR's own kinds of type, or, in the Python
//! binding, the `Type` and `DataType` subclasses Python code defines. The
//! core compares, binds, substitutes and unifies them through these hooks.
//! A hook that returns `None` asks for the core's default rule, which is
//! what an extension without a rule of its own gets.

use std::any::Any;
use std::borrow::Cow;
use std::fmt;
use std::hash::Hasher;

use crate::foreign::{Foreign, ForeignError};

use super::data_type::DataType;
use super::environment::TypeUnificationEnvironment;
use super::error::UnificationError;
use super::ty::Type;

/// A kind of [`Type`] defined outside this crate.
///
/// `Display` writes the type as messages show it.
pub trait TypeExtension: fmt::Debug + fmt::Display + Send + Sync {
    /// Return the name of the kind of type, as a refusal names it.
    fn type_name(&self) -> Cow<'_, str>;

    /// Return `self` as [`Any`], so its implementer can recover its own
    /// type.
    fn as_any(&self) -> &dyn Any;

    /// Return the type as a [`Foreign`] part, for serialization.
    ///
    /// # Errors
    ///
    /// The default returns [`ForeignError::NoWireForm`]: the type
    /// cannot be serialized.
    fn to_foreign(&self) -> Result<Foreign, ForeignError> {
        Err(ForeignError::NoWireForm {
            type_name: self.type_name().into_owned(),
        })
    }

    /// Return whether `other` is structurally equivalent to this type. The
    /// default answers `false`.
    fn is_structurally_equivalent(&self, other: &Type) -> bool {
        let _ = other;
        false
    }

    /// Return whether `other` equals this type, for `==` on [`Type`]. The
    /// default is identity: the same extension object.
    fn eq_extension(&self, other: &dyn TypeExtension) -> bool {
        std::ptr::addr_eq(std::ptr::from_ref(self), std::ptr::from_ref(other))
    }

    /// Feed this type's hash to `state`, consistently with
    /// [`eq_extension`](Self::eq_extension). The default feeds nothing.
    fn hash_extension(&self, state: &mut dyn Hasher) {
        let _ = state;
    }

    /// Bind this type, as a pattern, against `actual`, or return `None` for
    /// the default rule: `actual` must be structurally equivalent, and
    /// nothing is bound.
    fn bind_template(
        &self,
        actual: &Type,
        environment: &TypeUnificationEnvironment,
    ) -> Option<Result<TypeUnificationEnvironment, UnificationError>> {
        let _ = (actual, environment);
        None
    }

    /// Return this type with the placeholders `environment` binds
    /// substituted, or `None` for the default rule: the type unchanged.
    fn substitute_template(
        &self,
        environment: &TypeUnificationEnvironment,
    ) -> Option<Result<Type, UnificationError>> {
        let _ = environment;
        None
    }

    /// Unify this type, as the expected one, with `actual`, or return `None`
    /// for the default rule: `actual` must be structurally equivalent, and
    /// the result is this type, with nothing bound.
    fn unify(
        &self,
        actual: &Type,
        environment: &TypeUnificationEnvironment,
    ) -> Option<Result<(Type, TypeUnificationEnvironment), UnificationError>> {
        let _ = (actual, environment);
        None
    }
}

/// A kind of [`DataType`] defined outside this crate.
///
/// `Display` writes the data type as messages show it.
pub trait DataTypeExtension: fmt::Debug + fmt::Display + Send + Sync {
    /// Return the name of the kind of data type, as a refusal names it.
    fn type_name(&self) -> Cow<'_, str>;

    /// Return `self` as [`Any`], so its implementer can recover its own
    /// type.
    fn as_any(&self) -> &dyn Any;

    /// Return the data type as a [`Foreign`] part, for serialization.
    ///
    /// # Errors
    ///
    /// The default returns [`ForeignError::NoWireForm`]: the data type
    /// cannot be serialized.
    fn to_foreign(&self) -> Result<Foreign, ForeignError> {
        Err(ForeignError::NoWireForm {
            type_name: self.type_name().into_owned(),
        })
    }

    /// Return whether `other` is structurally equivalent to this data type.
    /// The default answers `false`.
    fn is_structurally_equivalent(&self, other: &DataType) -> bool {
        let _ = other;
        false
    }

    /// Return whether `other` equals this data type, for `==` on
    /// [`DataType`]. The default is identity: the same extension object.
    fn eq_extension(&self, other: &dyn DataTypeExtension) -> bool {
        std::ptr::addr_eq(std::ptr::from_ref(self), std::ptr::from_ref(other))
    }

    /// Feed this data type's hash to `state`, consistently with
    /// [`eq_extension`](Self::eq_extension). The default feeds nothing.
    fn hash_extension(&self, state: &mut dyn Hasher) {
        let _ = state;
    }

    /// Bind this data type, as a pattern, against `actual`, or return
    /// `None` for the default rule: `actual` must be structurally
    /// equivalent, and nothing is bound.
    fn bind_template(
        &self,
        actual: &DataType,
        environment: &TypeUnificationEnvironment,
    ) -> Option<Result<TypeUnificationEnvironment, UnificationError>> {
        let _ = (actual, environment);
        None
    }

    /// Return this data type with the placeholders `environment` binds
    /// substituted, or `None` for the default rule: the data type
    /// unchanged.
    fn substitute_template(
        &self,
        environment: &TypeUnificationEnvironment,
    ) -> Option<Result<DataType, UnificationError>> {
        let _ = environment;
        None
    }
}
