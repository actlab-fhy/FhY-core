//! Types and data types defined outside this crate.
//!
//! A [`Type::Extension`](super::Type::Extension) holds a
//! [`TypeExtension`], and a [`DataType::Extension`](super::DataType::Extension)
//! a [`DataTypeExtension`], each behind a [`Part`](crate::foreign::Part): a Rust IR's own kinds of
//! type, or, in the Python binding, the `Type` and `DataType` subclasses
//! Python code defines. The core compares, binds, substitutes and unifies
//! them through these hooks. A hook an implementation does not override
//! runs the core's default rule, a public function of this module's parent
//! ([`default_bind_template`](super::default_bind_template),
//! [`default_substitute_template`](super::default_substitute_template),
//! [`default_unify`](super::default_unify) and their data-type
//! counterparts), which an override can call too.
//!
//! The binding hooks receive `this`, the type or data type the extension is
//! the part of, so the defaults can answer with the very handle.

use std::fmt;
use std::hash::Hasher;

use crate::foreign::{BoxError, ForeignPart, impl_part, is_same_part};

use super::data_type::DataType;
use super::environment::TypeUnificationEnvironment;
use super::error::UnificationError;
use super::ty::Type;

/// A kind of [`Type`] defined outside this crate.
///
/// `Display` writes the type as messages show it.
pub trait TypeExtension: ForeignPart + fmt::Display {
    /// Return whether `other` is structurally equivalent to this type.
    ///
    /// The core asks the extension whichever side it is on, so the answer
    /// must be symmetric: an extension equivalent to a built-in type or to
    /// another extension must answer so when asked about it. The default is
    /// [`eq_part`](Self::eq_part) against an extension, and `false` against
    /// a built-in type, so an extension is equivalent to itself.
    ///
    /// # Errors
    ///
    /// Returns the implementation's error, which the core raises as
    /// [`UnificationError::Extension`] from a binding or a unification.
    fn is_structurally_equivalent(&self, other: &Type) -> Result<bool, BoxError> {
        Ok(match other {
            Type::Extension(other) => self.eq_part(other.get()),
            _ => false,
        })
    }

    /// Return whether `other` equals this type, for `==` on [`Type`]. The
    /// default is identity: the same extension object.
    ///
    /// It must be an equivalence relation, symmetric included, as `==` is,
    /// and agree with [`hash_part`](Self::hash_part).
    fn eq_part(&self, other: &dyn TypeExtension) -> bool {
        is_same_part(self, other)
    }

    /// Feed this type's hash to `state`, consistently with
    /// [`eq_part`](Self::eq_part). The default feeds nothing.
    fn hash_part(&self, state: &mut dyn Hasher) {
        let _ = state;
    }

    /// Bind `this`, the type this extension is the part of, as a pattern,
    /// against `actual`.
    ///
    /// The default is [`default_bind_template`](super::default_bind_template):
    /// `actual` must be structurally equivalent, and nothing is bound.
    ///
    /// # Errors
    ///
    /// Returns the [`UnificationError`] of the rule that fails.
    fn bind_template(
        &self,
        this: &Type,
        actual: &Type,
        environment: &TypeUnificationEnvironment,
    ) -> Result<TypeUnificationEnvironment, UnificationError> {
        super::default_bind_template(this, actual, environment)
    }

    /// Return `this`, the type this extension is the part of, with the
    /// placeholders `environment` binds substituted.
    ///
    /// The default is
    /// [`default_substitute_template`](super::default_substitute_template):
    /// `this` unchanged, the same handle.
    ///
    /// # Errors
    ///
    /// Returns the implementation's [`UnificationError`].
    fn substitute_template(
        &self,
        this: &Type,
        environment: &TypeUnificationEnvironment,
    ) -> Result<Type, UnificationError> {
        super::default_substitute_template(this, environment)
    }

    /// Unify `this`, the type this extension is the part of, as the
    /// expected one, with `actual`.
    ///
    /// The default is [`default_unify`](super::default_unify): `actual` must
    /// be structurally equivalent, and the result is `this`, with nothing
    /// bound.
    ///
    /// # Errors
    ///
    /// Returns the [`UnificationError`] of the rule that fails.
    fn unify(
        &self,
        this: &Type,
        actual: &Type,
        environment: &TypeUnificationEnvironment,
    ) -> Result<(Type, TypeUnificationEnvironment), UnificationError> {
        super::default_unify(this, actual, environment)
    }
}

impl_part!(TypeExtension);

/// A kind of [`DataType`] defined outside this crate.
///
/// `Display` writes the data type as messages show it.
pub trait DataTypeExtension: ForeignPart + fmt::Display {
    /// Return whether `other` is structurally equivalent to this data type.
    ///
    /// The core asks the extension whichever side it is on, so the answer
    /// must be symmetric. The default is [`eq_part`](Self::eq_part) against
    /// an extension, and `false` against a built-in data type, so an
    /// extension is equivalent to itself.
    ///
    /// # Errors
    ///
    /// Returns the implementation's error, which the core raises as
    /// [`UnificationError::Extension`] from a binding or a unification.
    fn is_structurally_equivalent(&self, other: &DataType) -> Result<bool, BoxError> {
        Ok(match other {
            DataType::Extension(other) => self.eq_part(other.get()),
            _ => false,
        })
    }

    /// Return whether `other` equals this data type, for `==` on
    /// [`DataType`]. The default is identity: the same extension object.
    ///
    /// It must be an equivalence relation, symmetric included, as `==` is,
    /// and agree with [`hash_part`](Self::hash_part).
    fn eq_part(&self, other: &dyn DataTypeExtension) -> bool {
        is_same_part(self, other)
    }

    /// Feed this data type's hash to `state`, consistently with
    /// [`eq_part`](Self::eq_part). The default feeds nothing.
    fn hash_part(&self, state: &mut dyn Hasher) {
        let _ = state;
    }

    /// Bind `this`, the data type this extension is the part of, as a
    /// pattern, against `actual`.
    ///
    /// The default is
    /// [`default_bind_data_template`](super::default_bind_data_template):
    /// `actual` must be structurally equivalent, and nothing is bound.
    ///
    /// # Errors
    ///
    /// Returns the [`UnificationError`] of the rule that fails.
    fn bind_template(
        &self,
        this: &DataType,
        actual: &DataType,
        environment: &TypeUnificationEnvironment,
    ) -> Result<TypeUnificationEnvironment, UnificationError> {
        super::default_bind_data_template(this, actual, environment)
    }

    /// Return `this`, the data type this extension is the part of, with the
    /// placeholders `environment` binds substituted.
    ///
    /// The default is
    /// [`default_substitute_data_template`](super::default_substitute_data_template):
    /// `this` unchanged.
    ///
    /// # Errors
    ///
    /// Returns the implementation's [`UnificationError`].
    fn substitute_template(
        &self,
        this: &DataType,
        environment: &TypeUnificationEnvironment,
    ) -> Result<DataType, UnificationError> {
        super::default_substitute_data_template(this, environment)
    }
}

impl_part!(DataTypeExtension);
