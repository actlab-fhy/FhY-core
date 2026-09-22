//! Open, registry-backed semantic tags for compiler operations.
//!
//! Every layer of a compiler stack carries semantic attributes on its
//! operations: algebraic properties such as commutativity and associativity,
//! purity, elementwise application, and family-specific tags contributed by
//! particular IRs. [`OpAttribute`] keeps that classification open, so a layer
//! shares the four generic attributes shipped here and contributes its own
//! without changing this crate.
//!
//! An attribute is a free-standing tag: it depends on no operation type and is
//! specialized for no layer. Each one is canonicalized by its [`Identifier`]
//! through [`crate::interned`], so import the constants below rather than
//! building a fresh attribute with the same name hint. Identifiers compare by
//! id, and a second `Identifier::new("commutative")` is a different key.
//!
//! A `description` is human-readable metadata. It takes no part in equality,
//! hashing or interning: the first attribute registered under an identifier
//! stays canonical, and a later one is handed back to its caller in
//! [`InternOutcome::AlreadyCanonical`] instead of replacing it.

use std::hash::{Hash, Hasher};
use std::sync::LazyLock;

use serde::{Deserialize, Serialize};

use crate::identifier::{HasIdentifier, Identifier};
use crate::interned::{Canonical, InternOutcome, InternRegistry, Interned};

/// Open semantic tag attached to a compiler operation.
///
/// Two attributes are equal when they carry the same [`Identifier`], whatever
/// their descriptions say.
///
/// Decoding an attribute canonicalizes it only through the handle, so
/// deserialize a [`Canonical<OpAttribute>`]. Deserializing a bare
/// `OpAttribute` yields a value that no registry knows about.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpAttribute {
    name: Identifier,
    description: String,
}

impl OpAttribute {
    /// Build the attribute named `name` and register it as the canonical one
    /// for that name.
    ///
    /// The outcome carries the canonical handle either way. When `name` is
    /// already taken the earlier attribute stays canonical, and the one built
    /// here comes back as the outcome's `discarded` value, so a caller that
    /// cares about the dropped description can see it.
    ///
    /// # Examples
    ///
    /// ```
    /// use fhy_core::identifier::Identifier;
    /// use fhy_core::interned::Interned;
    /// use fhy_core::op_attribute::OpAttribute;
    ///
    /// let name = Identifier::new("idempotent");
    /// let attribute =
    ///     OpAttribute::new(name.clone(), "Applying the op twice changes nothing.")
    ///         .into_canonical();
    ///
    /// assert_eq!(attribute.name(), &name);
    /// assert_eq!(OpAttribute::intern_registry().get(&name), Some(attribute));
    /// ```
    pub fn new(name: Identifier, description: impl Into<String>) -> InternOutcome<Self> {
        Self::intern_registry().intern(Self::create(name, description))
    }

    /// Build the attribute without registering it.
    fn create(name: Identifier, description: impl Into<String>) -> Self {
        Self {
            name,
            description: description.into(),
        }
    }

    /// Return the attribute's name.
    #[must_use]
    pub fn name(&self) -> &Identifier {
        &self.name
    }

    /// Return the attribute's human-readable description.
    #[must_use]
    pub fn description(&self) -> &str {
        &self.description
    }
}

impl HasIdentifier for OpAttribute {
    fn identifier(&self) -> &Identifier {
        &self.name
    }
}

impl Interned for OpAttribute {
    type Key = Identifier;

    fn intern_key(&self) -> &Identifier {
        &self.name
    }

    fn intern_registry() -> &'static InternRegistry<Self> {
        static REGISTRY: InternRegistry<OpAttribute> =
            InternRegistry::with_defaults(create_default_attributes);
        &REGISTRY
    }
}

impl PartialEq for OpAttribute {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for OpAttribute {}

impl Hash for OpAttribute {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

/// Name of the attribute shipped as [`COMMUTATIVE`].
static COMMUTATIVE_NAME: LazyLock<Identifier> =
    LazyLock::new(|| Identifier::new_unscoped("commutative"));

/// Name of the attribute shipped as [`ASSOCIATIVE`].
static ASSOCIATIVE_NAME: LazyLock<Identifier> =
    LazyLock::new(|| Identifier::new_unscoped("associative"));

/// Name of the attribute shipped as [`PURE`].
static PURE_NAME: LazyLock<Identifier> = LazyLock::new(|| Identifier::new_unscoped("pure"));

/// Name of the attribute shipped as [`ELEMENTWISE`].
static ELEMENTWISE_NAME: LazyLock<Identifier> =
    LazyLock::new(|| Identifier::new_unscoped("elementwise"));

/// Build the attributes this module ships, in registration order.
///
/// The registry calls this on its first use and again whenever it is cleared,
/// so the constants below stay canonical for the life of the process.
fn create_default_attributes() -> Vec<OpAttribute> {
    vec![
        OpAttribute::create(
            COMMUTATIVE_NAME.clone(),
            "Op output is invariant under operand swap.",
        ),
        OpAttribute::create(
            ASSOCIATIVE_NAME.clone(),
            "Op composes associatively across applications.",
        ),
        OpAttribute::create(
            PURE_NAME.clone(),
            "Op has no side effects and produces deterministic outputs.",
        ),
        OpAttribute::create(
            ELEMENTWISE_NAME.clone(),
            "Op acts independently on each element of its operands.",
        ),
    ]
}

/// Return the canonical attribute registered under a shipped default's name.
///
/// # Panics
///
/// Panics if `name` is not one of the names [`create_default_attributes`]
/// builds, since the registry registers every default on its first use.
fn require_default(name: &Identifier) -> Canonical<OpAttribute> {
    OpAttribute::intern_registry()
        .require(name)
        .expect("the registry registers every default on its first use")
}

/// Op output is invariant under operand swap.
pub static COMMUTATIVE: LazyLock<Canonical<OpAttribute>> =
    LazyLock::new(|| require_default(&COMMUTATIVE_NAME));

/// Op composes associatively across applications.
pub static ASSOCIATIVE: LazyLock<Canonical<OpAttribute>> =
    LazyLock::new(|| require_default(&ASSOCIATIVE_NAME));

/// Op has no side effects and produces deterministic outputs.
pub static PURE: LazyLock<Canonical<OpAttribute>> = LazyLock::new(|| require_default(&PURE_NAME));

/// Op acts independently on each element of its operands.
pub static ELEMENTWISE: LazyLock<Canonical<OpAttribute>> =
    LazyLock::new(|| require_default(&ELEMENTWISE_NAME));

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::{PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

    use crate::test_support::{compute_hash, reserve_pinned_id};

    /// Serializes the test that clears the process-wide registry against the
    /// tests that need their own entries to survive. Registering tests hold
    /// the read side; the clearing test holds the write side.
    static REGISTRY_GUARD: RwLock<()> = RwLock::new(());

    fn hold_registry() -> RwLockReadGuard<'static, ()> {
        REGISTRY_GUARD
            .read()
            .unwrap_or_else(PoisonError::into_inner)
    }

    fn hold_registry_exclusively() -> RwLockWriteGuard<'static, ()> {
        REGISTRY_GUARD
            .write()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// Intern `description` under `name`, expecting the key to be taken, and
    /// return the value that lost the race.
    fn intern_and_take_discarded(name: Identifier, description: &str) -> OpAttribute {
        match OpAttribute::new(name, description) {
            InternOutcome::AlreadyCanonical { discarded, .. } => discarded,
            InternOutcome::Registered(_) => panic!("expected {description:?} to be discarded"),
        }
    }

    #[test]
    fn new_stores_the_name_and_description() {
        let _guard = hold_registry();
        let name = Identifier::new("stores-name-and-description");
        let attribute = OpAttribute::new(name.clone(), "an attribute").into_canonical();

        assert_eq!(attribute.name(), &name);
        assert_eq!(attribute.description(), "an attribute");
    }

    #[test]
    fn has_identifier_returns_the_name() {
        let _guard = hold_registry();
        let name = Identifier::new("has-identifier");
        let attribute = OpAttribute::new(name.clone(), "desc").into_canonical();

        assert_eq!(attribute.identifier(), &name);
    }

    #[test]
    fn intern_key_is_the_name() {
        let _guard = hold_registry();
        let name = Identifier::new("intern-key");
        let attribute = OpAttribute::new(name.clone(), "desc").into_canonical();

        assert_eq!(attribute.intern_key(), &name);
    }

    #[test]
    fn new_keeps_the_first_attribute_canonical_for_a_repeated_name() {
        let _guard = hold_registry();
        let name = Identifier::new("repeated-name");
        let first = OpAttribute::new(name.clone(), "first").into_canonical();

        let outcome = OpAttribute::new(name.clone(), "second");

        assert!(!outcome.is_registered());
        let InternOutcome::AlreadyCanonical {
            canonical,
            discarded,
        } = &outcome
        else {
            panic!("expected the second attribute to lose the registration");
        };
        assert_eq!(canonical, &first);
        assert_eq!(discarded.description(), "second");
        assert_eq!(OpAttribute::intern_registry().get(&name), Some(first));
    }

    #[test]
    fn identifiers_sharing_a_name_hint_intern_separately() {
        let _guard = hold_registry();
        let first_name = Identifier::new("dup");
        let second_name = Identifier::new("dup");
        let first = OpAttribute::new(first_name.clone(), "a").into_canonical();
        let second = OpAttribute::new(second_name.clone(), "b").into_canonical();

        assert_ne!(first, second);
        assert_eq!(OpAttribute::intern_registry().get(&first_name), Some(first));
        assert_eq!(
            OpAttribute::intern_registry().get(&second_name),
            Some(second)
        );
    }

    #[test]
    fn attributes_with_the_same_name_are_equal_whatever_the_description() {
        let _guard = hold_registry();
        let name = Identifier::new("equality-ignores-description");
        let canonical = OpAttribute::new(name.clone(), "first description").into_canonical();

        let other = intern_and_take_discarded(name, "second description");

        assert_eq!(*canonical, other);
        assert_eq!(other, *canonical);
    }

    #[test]
    fn equal_attributes_hash_equally() {
        let _guard = hold_registry();
        let name = Identifier::new("hash-ignores-description");
        let canonical = OpAttribute::new(name.clone(), "first description").into_canonical();

        let other = intern_and_take_discarded(name, "second description");

        assert_eq!(compute_hash(&*canonical), compute_hash(&other));
    }

    #[test]
    fn attributes_with_different_names_are_unequal() {
        let _guard = hold_registry();
        let left = OpAttribute::new(Identifier::new("a"), "desc").into_canonical();
        let right = OpAttribute::new(Identifier::new("b"), "desc").into_canonical();

        assert_ne!(*left, *right);
    }

    #[test]
    fn canonical_attributes_can_be_collected_into_a_set() {
        let _guard = hold_registry();
        let tags: HashSet<Canonical<OpAttribute>> =
            [COMMUTATIVE.clone(), PURE.clone()].into_iter().collect();

        assert!(tags.contains(&*COMMUTATIVE));
        assert!(tags.contains(&*PURE));
        assert!(!tags.contains(&*ASSOCIATIVE));
        assert_eq!(tags.len(), 2);
    }

    #[test]
    fn a_repeated_canonical_attribute_collapses_to_one_set_entry() {
        let _guard = hold_registry();
        let tags: HashSet<Canonical<OpAttribute>> =
            [COMMUTATIVE.clone(), COMMUTATIVE.clone(), PURE.clone()]
                .into_iter()
                .collect();

        let expected: HashSet<Canonical<OpAttribute>> =
            [COMMUTATIVE.clone(), PURE.clone()].into_iter().collect();
        assert_eq!(tags, expected);
    }

    #[test]
    fn commutative_is_registered_under_its_name() {
        let _guard = hold_registry();
        assert_eq!(
            OpAttribute::intern_registry().get(COMMUTATIVE.name()),
            Some(COMMUTATIVE.clone())
        );
    }

    #[test]
    fn associative_is_registered_under_its_name() {
        let _guard = hold_registry();
        assert_eq!(
            OpAttribute::intern_registry().get(ASSOCIATIVE.name()),
            Some(ASSOCIATIVE.clone())
        );
    }

    #[test]
    fn pure_is_registered_under_its_name() {
        let _guard = hold_registry();
        assert_eq!(
            OpAttribute::intern_registry().get(PURE.name()),
            Some(PURE.clone())
        );
    }

    #[test]
    fn elementwise_is_registered_under_its_name() {
        let _guard = hold_registry();
        assert_eq!(
            OpAttribute::intern_registry().get(ELEMENTWISE.name()),
            Some(ELEMENTWISE.clone())
        );
    }

    #[test]
    fn the_default_attributes_are_pairwise_distinct() {
        let _guard = hold_registry();
        let defaults = [
            COMMUTATIVE.clone(),
            ASSOCIATIVE.clone(),
            PURE.clone(),
            ELEMENTWISE.clone(),
        ];

        for (index, left) in defaults.iter().enumerate() {
            for right in &defaults[index + 1..] {
                assert_ne!(left, right);
                assert_ne!(left.name(), right.name());
            }
        }
    }

    #[test]
    fn the_default_attributes_carry_non_empty_descriptions() {
        let _guard = hold_registry();
        for default in [&*COMMUTATIVE, &*ASSOCIATIVE, &*PURE, &*ELEMENTWISE] {
            assert!(!default.description().trim().is_empty());
        }
    }

    #[test]
    fn clearing_the_registry_keeps_the_default_attributes_canonical() {
        let _guard = hold_registry_exclusively();
        let name = Identifier::new("dropped-by-clear");
        let dropped = OpAttribute::new(name.clone(), "dropped").into_canonical();
        assert_eq!(OpAttribute::intern_registry().get(&name), Some(dropped));

        OpAttribute::intern_registry().clear();

        assert_eq!(OpAttribute::intern_registry().get(&name), None);
        for default in [&COMMUTATIVE, &ASSOCIATIVE, &PURE, &ELEMENTWISE] {
            assert_eq!(
                OpAttribute::intern_registry().get(default.name()),
                Some((*default).clone())
            );
        }
    }

    #[test]
    fn an_attribute_encodes_as_its_name_and_description() {
        let _guard = hold_registry();
        let id = reserve_pinned_id("encode-anchor");
        let name = Identifier::deserialize(id, "encoded".to_string());
        let attribute = OpAttribute::new(name, "a description").into_canonical();

        let json = serde_json::to_string(&*attribute).unwrap();

        assert_eq!(
            json,
            format!(
                "{{\"name\":{{\"id\":{id},\"name_hint\":\"encoded\"}},\
                 \"description\":\"a description\"}}"
            )
        );
    }

    #[test]
    fn an_attribute_round_trips_through_json() {
        let _guard = hold_registry();
        let attribute = OpAttribute::new(Identifier::new("round-trip"), "desc").into_canonical();

        let json = serde_json::to_string(&*attribute).unwrap();
        let restored: Canonical<OpAttribute> = serde_json::from_str(&json).unwrap();

        assert_eq!(restored, attribute);
    }

    #[test]
    fn decoding_a_registered_name_returns_the_canonical_attribute() {
        let _guard = hold_registry();
        let json = serde_json::to_string(&*PURE).unwrap();

        let restored: Canonical<OpAttribute> = serde_json::from_str(&json).unwrap();

        assert_eq!(restored, *PURE);
    }

    #[test]
    fn decoding_an_unregistered_name_registers_the_decoded_attribute() {
        let _guard = hold_registry();
        let id = reserve_pinned_id("unregistered-decode-anchor");
        let json = format!(
            "{{\"name\":{{\"id\":{id},\"name_hint\":\"never-registered\"}},\
             \"description\":\"fresh from decode\"}}"
        );

        let restored: Canonical<OpAttribute> = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.description(), "fresh from decode");
        assert_eq!(
            OpAttribute::intern_registry().get(restored.name()),
            Some(restored)
        );
    }

    #[test]
    fn decoding_a_divergent_description_keeps_the_canonical_one() {
        let _guard = hold_registry();
        let name = Identifier::new("divergent-description");
        let canonical = OpAttribute::new(name.clone(), "original description").into_canonical();
        let json = format!(
            "{{\"name\":{{\"id\":{},\"name_hint\":\"{}\"}},\
             \"description\":\"divergent description\"}}",
            name.id(),
            name.name_hint()
        );

        let restored: Canonical<OpAttribute> = serde_json::from_str(&json).unwrap();

        assert_eq!(restored, canonical);
        assert_eq!(restored.description(), "original description");
    }

    #[test]
    fn new_reports_a_matching_duplicate_as_already_canonical() {
        let _guard = hold_registry();
        let name = Identifier::new("matching-description");
        let canonical = OpAttribute::new(name.clone(), "matching").into_canonical();

        let discarded = intern_and_take_discarded(name, "matching");

        assert_eq!(discarded.description(), canonical.description());
        assert_eq!(discarded, *canonical);
    }

    #[test]
    fn decoding_a_payload_with_an_unknown_field_is_rejected() {
        let _guard = hold_registry();
        let id = reserve_pinned_id("unknown-field-anchor");
        let json = format!(
            "{{\"name\":{{\"id\":{id},\"name_hint\":\"extra\"}},\
             \"description\":\"desc\",\"surprise\":1}}"
        );

        let error = serde_json::from_str::<Canonical<OpAttribute>>(&json).unwrap_err();

        assert!(error.to_string().contains("surprise"), "{error}");
    }

    #[test]
    fn decoding_a_payload_missing_the_description_is_rejected() {
        let _guard = hold_registry();
        let id = reserve_pinned_id("missing-field-anchor");
        let json = format!("{{\"name\":{{\"id\":{id},\"name_hint\":\"partial\"}}}}");

        let error = serde_json::from_str::<Canonical<OpAttribute>>(&json).unwrap_err();

        assert!(error.to_string().contains("description"), "{error}");
    }

    #[test]
    fn debug_mentions_the_name_hint_and_description() {
        let _guard = hold_registry();
        let attribute =
            OpAttribute::new(Identifier::new("debug-attribute"), "debug desc").into_canonical();

        let rendered = format!("{attribute:?}");

        assert!(rendered.contains("debug-attribute"), "{rendered}");
        assert!(rendered.contains("debug desc"), "{rendered}");
    }
}
