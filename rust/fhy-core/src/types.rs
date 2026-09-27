//! The type system of the IR: data types, numerical and index types, their
//! promotion, and template binding, substitution and unification.
//!
//! - [`CoreDataType`] is the primitive element type, with its promotion
//!   orders ([`CoreDataType::promote`]) and the type a literal resolves to
//!   ([`CoreDataType::resolve_literal`]). [`TypeQualifier`] says how a value
//!   may be used.
//! - [`DataType`] is a core data type, a [`TemplateDataType`] placeholder,
//!   or an extension; [`Type`] is a [`NumericalType`] array over a shape of
//!   [`Dimension`]s, an [`IndexType`] range, or an extension.
//! - [`Type::bind_template`], [`Type::substitute_template`] and
//!   [`Type::unify`], with their [`DataType`] counterparts and
//!   [`unify_expressions`], bind placeholders into a
//!   [`TypeUnificationEnvironment`] and read them back.
//! - [`TypeExtension`] and [`DataTypeExtension`] let types defined outside
//!   this crate take part in every operation.
//! - [`checking`] type-checks expressions against the type system, and holds
//!   function bodies to their declared result sorts.
//!
//! # Examples
//!
//! ```
//! use fhy_core::expression::Expression;
//! use fhy_core::identifier::Identifier;
//! use fhy_core::types::{
//!     CoreDataType, Dimension, NumericalType, TemplateDataType, Type, TypeUnificationEnvironment,
//! };
//!
//! let (t, n) = (Identifier::new("T"), Identifier::new("N"));
//! let pattern = Type::from(NumericalType::new(
//!     TemplateDataType::new(t.clone()),
//!     [Dimension::from(Expression::from(n.clone()))],
//! ));
//! let actual = Type::from(NumericalType::new(CoreDataType::Int32, [Dimension::from(Expression::from(8))]));
//!
//! let environment = pattern.bind_template(&actual, &TypeUnificationEnvironment::new())?;
//!
//! assert_eq!(environment.expression_binding(&n), Some(&Expression::from(8)));
//! assert_eq!(pattern.substitute_template(&environment)?, actual);
//! # Ok::<(), fhy_core::types::UnificationError>(())
//! ```

pub mod checking;

mod core_data_type;
mod data_type;
mod environment;
mod error;
mod extension;
mod qualifier;
mod ty;
mod unify;
pub mod wire;

pub use core_data_type::CoreDataType;
pub use data_type::{DataType, TemplateDataType};
pub use environment::TypeUnificationEnvironment;
pub use error::{
    IntegerFamily, LiteralTypeError, PromotionError, TemplateWidthError, TypeOperation,
    UnificationError,
};
pub use extension::{DataTypeExtension, TypeExtension};
pub use qualifier::TypeQualifier;
pub use ty::{Dimension, IndexType, NumericalType, Type};
pub use unify::unify_expressions;
