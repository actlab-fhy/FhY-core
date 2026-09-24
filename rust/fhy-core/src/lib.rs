//! Core data structures for the `FhY` compiler: identifiers, interned
//! vocabularies, diagnostics and provenance, symbolic expressions with
//! patterns and rewrite rules, tree traversals, and a compiler-pass
//! framework.
//!
//! # Modules
//!
//! Each module depends only on the modules listed before it, except that
//! [`expr`] and [`pass`] are independent of each other and
//! [`expr::passes`] joins them.
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
//! | [`expr`] | symbolic expressions, their builders and analyses; the built-in catalogue in [`expr::builtins`], patterns and rewrite rules in [`expr::pattern`], and the passes over expressions in [`expr::passes`] |
//! | [`pass`] | compiler passes, pipelines, fixpoint groups, analyses, validators and the pass registry |
//!
//! Each public item has exactly one public path.
//!
//! # Example
//!
//! ```
//! use std::collections::HashMap;
//!
//! use fhy_core::expr::Expression;
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
//! `static`s and append-only: an id is never reissued, and a canonical value
//! is never replaced or removed. The tags this crate ships hold fixed ids
//! below [`RESERVED_ID_COUNT`](identifier::RESERVED_ID_COUNT), the same in
//! every process. A process must therefore hold exactly one compiled copy of
//! this crate: link it into one Python extension module, and compile Rust
//! code from other `FhY` packages into that same module rather than into a
//! second one. A second copy would issue ids that collide with the first
//! copy's, and keep registries whose canonical instances never equal the
//! first copy's. Everything else, including the pass registry, is an owned
//! value.
//!
//! # Serialization
//!
//! Every public type that implements `Serialize` also implements
//! `Deserialize`, with a plain serde shape that this crate defines and
//! documents on the type. The impls are format-agnostic: they work with
//! self-describing formats such as JSON and with non-self-describing formats
//! such as postcard, and the tests round-trip every such type through both.
//! Numbers a format may not hold exactly, the integers and floats of an
//! [`Expression`](expr::Expression)'s literals, serialize as strings.
//! Deserializing an [`Identifier`](identifier::Identifier) advances the
//! process-global id counter past its id, and deserializing a
//! [`Canonical<T>`](interned::Canonical) interns the value. A decode that
//! fails partway may leave both effects behind for the parts it already
//! decoded. Both effects only ever add ids and canonical values, so they
//! never invalidate an existing identifier or canonical value. The
//! `__type__`/`__data__` envelope that Python's serialization framework uses
//! belongs to the Python binding, not to these shapes. This crate does not
//! depend on `serde_json`, so depending on it changes nothing about how
//! another crate's JSON numbers parse or compare.

pub mod described_tag;
pub mod diagnostic;
pub mod expr;
pub mod identifier;
pub mod interned;
pub mod op_attribute;
pub mod pass;
pub mod provenance;
pub mod tree;
pub mod value_domain;

#[cfg(test)]
mod test_support;
