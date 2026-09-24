//! Described tags: open, registry-backed vocabulary entries.
//!
//! A [`DescribedTag`] is an open vocabulary entry: an [`Identifier`] name
//! that is its identity, and a human-readable description that takes no part
//! in equality, hashing or interning. Each vocabulary is a [`TagKind`] with
//! its own process-wide registry, so the same identifier registered in two
//! vocabularies names two independent tags. This crate ships two
//! vocabularies, [`OpAttribute`](crate::op_attribute::OpAttribute) and
//! [`NoteKind`](crate::diagnostic::NoteKind); the set is closed.
//!
//! A tag encodes as `{"name": <identifier>, "description": ..}`. Decoding a
//! [`Canonical`] of it interns it.

use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use serde::{Deserialize, Deserializer, Serialize};

use crate::identifier::reserved::ReservedIdentifier;
use crate::identifier::{HasIdentifier, Identifier};
use crate::interned::{Canonical, InternOutcome, InternRegistry, Interned};

/// A vocabulary of described tags.
///
/// Sealed: only this crate's vocabularies exist, so a tag of one of them can
/// only be registered through its own registry.
///
/// ```compile_fail
/// use fhy_core::described_tag::TagKind;
///
/// enum MyVocabulary {}
///
/// impl TagKind for MyVocabulary {}
/// ```
pub trait TagKind: sealed::Sealed + Send + Sync + 'static {}

/// The sealing supertrait of [`TagKind`], which carries a vocabulary's name
/// and registry.
pub(crate) mod sealed {
    use super::{DescribedTag, InternRegistry, TagKind};

    #[expect(
        unnameable_types,
        reason = "the sealing supertrait is nameable only inside this crate by design"
    )]
    pub trait Sealed: Sized {
        /// The vocabulary's name, which `Debug` renders a tag under.
        const TYPE_NAME: &'static str;

        /// Return the vocabulary's process-wide registry.
        fn registry() -> &'static InternRegistry<DescribedTag<Self>>
        where
            Self: TagKind;
    }
}

/// An open vocabulary entry of the vocabulary `K`.
///
/// Two tags are equal when they carry the same [`Identifier`], whatever their
/// descriptions say.
#[derive(Serialize)]
pub struct DescribedTag<K: TagKind> {
    name: Identifier,
    description: String,
    #[serde(skip)]
    kind: PhantomData<fn() -> K>,
}

impl<K: TagKind> DescribedTag<K> {
    /// Register the tag named `name`, unless one is registered already, and
    /// return the canonical handle for that name.
    ///
    /// The first registration wins: when `name` is already registered, the
    /// registered tag is returned and `description` is dropped.
    ///
    /// # Examples
    ///
    /// ```
    /// use fhy_core::identifier::Identifier;
    /// use fhy_core::interned::Interned;
    /// use fhy_core::op_attribute::OpAttribute;
    ///
    /// let name = Identifier::new("idempotent");
    /// let attribute = OpAttribute::register(name.clone(), "Applying the op twice changes nothing.");
    /// let again = OpAttribute::register(name.clone(), "Another description.");
    ///
    /// assert_eq!(attribute.name(), &name);
    /// assert_eq!(again.description(), "Applying the op twice changes nothing.");
    /// assert_eq!(OpAttribute::intern_registry().get(&name), Some(attribute));
    /// ```
    pub fn register(name: Identifier, description: impl Into<String>) -> Canonical<Self> {
        match K::registry().intern(Self::create(name, description)) {
            InternOutcome::Registered(canonical) => canonical,
            InternOutcome::AlreadyCanonical { canonical, .. } => {
                // The first registration wins, so a different description
                // given here is dropped.
                // TODO: warn here once the log dependency is added
                canonical
            }
        }
    }

    /// Build the shipped tag `entry` names, for a vocabulary's defaults.
    pub(crate) fn create_shipped(entry: ReservedIdentifier, description: &str) -> Self {
        Self::create(Identifier::reserved(entry), description)
    }

    /// Build the tag without registering it.
    fn create(name: Identifier, description: impl Into<String>) -> Self {
        Self {
            name,
            description: description.into(),
            kind: PhantomData,
        }
    }

    /// Return the tag's name.
    #[must_use]
    pub fn name(&self) -> &Identifier {
        &self.name
    }

    /// Return the tag's human-readable description.
    #[must_use]
    pub fn description(&self) -> &str {
        &self.description
    }
}

impl<K: TagKind> HasIdentifier for DescribedTag<K> {
    fn identifier(&self) -> &Identifier {
        &self.name
    }
}

impl<K: TagKind> Interned for DescribedTag<K> {
    type Key = Identifier;

    fn intern_key(&self) -> &Identifier {
        &self.name
    }

    fn intern_registry() -> &'static InternRegistry<Self> {
        K::registry()
    }
}

impl<K: TagKind> PartialEq for DescribedTag<K> {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl<K: TagKind> Eq for DescribedTag<K> {}

impl<K: TagKind> Hash for DescribedTag<K> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

/// Render the vocabulary's name with the tag's name and description.
impl<K: TagKind> fmt::Debug for DescribedTag<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct(K::TYPE_NAME)
            .field("name", &self.name)
            .field("description", &self.description)
            .finish()
    }
}

/// Render the name's hint, for example `commutative`.
impl<K: TagKind> fmt::Display for DescribedTag<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.name, f)
    }
}

/// Decodes a tag's payload and registers it: a payload naming a registered
/// tag yields that tag, keeping its description.
///
/// Only the handle decodes, so a decoded tag is always the registered one:
///
/// ```compile_fail
/// use fhy_core::op_attribute::OpAttribute;
///
/// let json = r#"{"name":{"id":16,"name_hint":"commutative"},"description":"d"}"#;
/// let _detached: OpAttribute = serde_json::from_str(json).unwrap();
/// ```
///
/// ```
/// use fhy_core::interned::Canonical;
/// use fhy_core::op_attribute::OpAttribute;
///
/// let json = r#"{"name":{"id":16,"name_hint":"commutative"},"description":"d"}"#;
/// let decoded: Canonical<OpAttribute> = serde_json::from_str(json).unwrap();
///
/// assert!(Canonical::ptr_eq(&decoded, OpAttribute::commutative()));
/// ```
impl<'de, K: TagKind> Deserialize<'de> for Canonical<DescribedTag<K>> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = DescribedTagWire::deserialize(deserializer)?;
        Ok(DescribedTag::register(wire.name, wire.description))
    }
}

/// The wire form of a [`DescribedTag`], decoded.
#[derive(Deserialize)]
#[serde(
    rename = "DescribedTag",
    expecting = "a described tag",
    deny_unknown_fields
)]
struct DescribedTagWire {
    name: Identifier,
    description: String,
}

/// Return the canonical shipped tag `entry` names in the vocabulary `K`.
///
/// # Panics
///
/// Panics if `entry` does not name one of the defaults `K`'s registry was
/// created with.
pub(crate) fn require_shipped<K: TagKind>(entry: ReservedIdentifier) -> Canonical<DescribedTag<K>> {
    crate::interned::require_default(&Identifier::reserved(entry))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::diagnostic::NoteKind;
    use crate::op_attribute::OpAttribute;
    use crate::test_support::compute_hash;

    /// Test one identifier registered as an attribute and as a note kind
    /// names two independent tags, each seen only by its own registry.
    #[test]
    fn attribute_and_note_kind_registries_are_independent() {
        let name = Identifier::new("in-both-vocabularies");

        let attribute = OpAttribute::register(name.clone(), "an attribute");
        let kind = NoteKind::register(name.clone(), "a note kind");

        assert_eq!(attribute.description(), "an attribute");
        assert_eq!(kind.description(), "a note kind");
        assert_eq!(OpAttribute::intern_registry().get(&name), Some(attribute));
        assert_eq!(NoteKind::intern_registry().get(&name), Some(kind));
    }

    /// Test an identifier registered in one vocabulary is unknown to the
    /// other.
    #[test]
    fn a_tag_registered_in_one_vocabulary_is_unknown_to_the_other() {
        let name = Identifier::new("in-one-vocabulary");

        let _attribute = OpAttribute::register(name.clone(), "an attribute");

        assert_eq!(NoteKind::intern_registry().get(&name), None);
    }

    /// Test two tags with one name are equal whatever their descriptions.
    #[test]
    fn tags_with_the_same_name_are_equal_whatever_the_description() {
        let name = Identifier::new("equality-ignores-description");
        let first = OpAttribute::create(name.clone(), "first description");
        let second = OpAttribute::create(name, "second description");

        assert_eq!(first, second);
        assert_eq!(second, first);
    }

    /// Test two tags with one name hash equally whatever their descriptions.
    #[test]
    fn equal_tags_hash_equally() {
        let name = Identifier::new("hash-ignores-description");
        let first = OpAttribute::create(name.clone(), "first description");
        let second = OpAttribute::create(name, "second description");

        assert_eq!(compute_hash(&first), compute_hash(&second));
    }

    /// Test a tag of either vocabulary displays as its name hint.
    #[test]
    fn display_renders_the_name_hint() {
        assert_eq!(OpAttribute::commutative().to_string(), "commutative");
        assert_eq!(NoteKind::rationale().to_string(), "rationale");
    }

    /// Test `Debug` renders the vocabulary's name, the name and the
    /// description.
    #[test]
    fn debug_names_the_vocabulary() {
        let rendered = format!("{:?}", **OpAttribute::pure());

        assert!(
            rendered.starts_with("OpAttribute { name: pure::18"),
            "{rendered}"
        );
        assert!(
            rendered.contains(OpAttribute::pure().description()),
            "{rendered}"
        );
    }
}
