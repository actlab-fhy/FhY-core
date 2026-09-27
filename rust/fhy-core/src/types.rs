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
//!   this crate take part in every operation, each held in a
//!   [`Part`](crate::foreign::Part). Their hooks are fallible, except the
//!   `eq_part` and `hash_part` behind `==` and `Hash`, so a failing hook is
//!   an [`UnificationError::Extension`], which structural equivalence
//!   returns too; a hook left to its default runs the public default rule
//!   ([`default_bind_template`] and the others).
//! - [`checking`] type-checks expressions against the type system, and holds
//!   function bodies to their declared result sorts.
//!
//! # Two equalities
//!
//! [`Type`], [`DataType`] and
//! [`SymbolFrame`](crate::symbol_table::SymbolFrame) each have two:
//!
//! - **`==`** (with `Hash`, agreeing with it) compares every field by its own
//!   `==`, and an extension through its `eq_part`, which is identity unless
//!   the extension overrides it. It never fails and never asks a fallible
//!   hook, so types can key a `HashMap`.
//! - **`is_structurally_equivalent`** is the relation binding and
//!   unification use. It asks an extension's own
//!   `is_structurally_equivalent` about the other side, whichever side the
//!   extension is on, and compares a numerical type's data type through it,
//!   so it can fail with [`UnificationError::Extension`].
//!
//! For built-in types without extensions the two agree. They differ only
//! where an extension answers differently through its two hooks: one
//! structurally equivalent to a built-in type is still not `==` to it. A
//! frame's `==` compares its types by `==`, and its
//! `is_structurally_equivalent` by theirs.
//!
//! A [`TemplateDataType`]'s widths are a set, kept sorted and without
//! repeats, so both relations compare them as sets.
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
pub use unify::{
    default_bind_data_template, default_bind_template, default_substitute_data_template,
    default_substitute_template, default_unify, unify_expressions,
};
