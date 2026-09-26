//! Core data structures for the `FhY` compiler: identifiers, interned
//! vocabularies, diagnostics and provenance, symbolic expressions with
//! patterns and rewrite rules, tree traversals, and a compiler-pass
//! framework.
//!
//! # Modules
//!
//! Each module depends only on the modules listed before it, except that
//! [`tree`], [`term`] and [`lattice`] are independent of each other,
//! [`expression`] and [`pass`] are independent of each other,
//! [`expression::passes`] joins them, and [`solver`] and [`types`] depend on
//! [`expression`] and not on [`pass`].
//!
//! | Module | Contents |
//! |---|---|
//! | [`identifier`] | [`Identifier`](identifier::Identifier): a name hint and a process-unique id |
//! | [`interned`] | [`InternRegistry`](interned::InternRegistry) and [`Canonical`](interned::Canonical): one canonical value per key |
//! | [`described_tag`] | [`DescribedTag`](described_tag::DescribedTag): open vocabulary entries, a name and a description |
//! | [`diagnostic`] | diagnostics, notes and their [`NoteKind`](diagnostic::NoteKind)s, and validation reports |
//! | [`op_attribute`] | [`OpAttribute`](op_attribute::OpAttribute): open semantic tags on operations |
//! | [`value_domain`] | [`ValueDomain`](value_domain::ValueDomain): the hierarchy of value classifications |
//! | [`provenance`] | source positions, spans and where a value came from |
//! | [`tree`] | the [`Tree`](tree::Tree) trait and iterative walks and rewrites over any tree-shaped IR |
//! | [`term`] | [`AlphaRenaming`](term::AlphaRenaming) and the traits of terms: alpha equivalence, free identifiers, substitution, and [`Binder`](term::Binder)s |
//! | [`lattice`] | [`PartiallyOrderedSet`](lattice::PartiallyOrderedSet) and [`Lattice`](lattice::Lattice): partial orders over hashable elements, their meets and joins |
//! | [`expression`] | symbolic expressions, their builders and analyses; the built-in catalogue in [`expression::builtins`], the owned registry of user functions and constants, and inlining, in [`expression::registry`], folding and evaluation in [`expression::evaluate`], patterns and rewrite rules in [`expression::pattern`], and the passes over expressions in [`expression::passes`] |
//! | [`pass`] | compiler passes, pipelines, fixpoint groups, analyses, validators and the pass registry |
//! | [`solver`] | questions about expressions answered by pluggable backends: the hazard screen, the SMT-LIB2 lowering, and a backend that drives an SMT-LIB2 executable |
//! | [`types`] | the IR type system: core data types and their promotion, data types, numerical and index types, and template binding, substitution and unification, with extensions; expression type checking in [`types::checking`] |
//!
//! Each public item has exactly one public path.
//!
//! # Example
//!
//! ```
//! use std::collections::HashMap;
//!
//! use fhy_core::expression::Expression;
//! use fhy_core::identifier::Identifier;
//!
//! let x = Identifier::new("x");
//! let y = Identifier::new("y");
//! let sum = &Expression::from(x.clone()) + 1;
//!
//! let replaced = sum.substitute(&HashMap::from([(x, Expression::from(y))]))?;
//!
//! assert_eq!(replaced.to_string(), "(y + 1)");
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Process-global state
//!
//! The identifier id counter and each [`Interned`](interned::Interned)
//! type's [`InternRegistry`](interned::InternRegistry) are process-global
//! and append-only: an id is never reissued, and a canonical value is never
//! replaced or removed. The tags this crate ships hold fixed ids below
//! [`RESERVED_ID_COUNT`](identifier::RESERVED_ID_COUNT), the same in every
//! process, and so do the built-in constants' identifiers. Everything else,
//! including the pass registry and the function registry, is an owned
//! value.
//!
//! A process must therefore hold exactly one compiled copy of this crate. A
//! second copy would issue ids that collide with the first copy's, and keep
//! registries whose canonical instances never equal the first copy's, so
//! link the crate into one Python extension module and compile Rust code
//! from other `FhY` packages into that same module.
//!
//! # Serialization
//!
//! Every public type that implements `Serialize` also implements
//! `Deserialize`, with a plain serde shape documented on the type. The
//! impls work with self-describing formats such as JSON and with
//! non-self-describing ones such as postcard. The integers and floats of an
//! [`Expression`](expression::Expression)'s literals, which a format may not hold
//! exactly, serialize as strings.
//!
//! Deserializing an [`Identifier`](identifier::Identifier) advances the id
//! counter past its id, and deserializing a
//! [`Canonical<T>`](interned::Canonical) interns the value. A decode that
//! fails partway may leave both effects behind for the parts it already
//! decoded; they only ever add ids and canonical values, so they never
//! invalidate an existing one. The `__type__`/`__data__` envelope of
//! Python's serialization framework belongs to the Python binding. This
//! crate does not depend on `serde_json`, so depending on it changes nothing
//! about how another crate's JSON numbers parse or compare.

pub mod described_tag;
pub mod diagnostic;
pub mod expression;
pub mod identifier;
pub mod interned;
pub mod lattice;
pub mod op_attribute;
pub mod pass;
pub mod provenance;
pub mod solver;
pub mod term;
pub mod tree;
pub mod types;
pub mod value_domain;

#[cfg(test)]
mod test_support;
