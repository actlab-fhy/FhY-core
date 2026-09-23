//! Open, registry-backed semantic tags for compiler operations.
//!
//! Every layer of a compiler stack carries semantic attributes on its
//! operations: algebraic properties such as commutativity and associativity,
//! purity, elementwise application, and family-specific tags contributed by
//! particular IRs. [`OpAttribute`] keeps that classification open, so a layer
//! shares the generic attributes shipped here and contributes its own
//! without changing this crate.
//!
//! An attribute is a free-standing tag: it depends on no operation type and is
//! specialized for no layer. Each one is canonicalized by its [`Identifier`]
//! through [`crate::interned`], so call the accessor functions below rather
//! than building a fresh attribute with the same name hint. Identifiers
//! compare by id, and a second `Identifier::new("commutative")` is a
//! different key.
//!
//! A `description` is human-readable metadata. It takes no part in equality,
//! hashing or interning: the first attribute registered under an identifier
//! stays canonical, and a later one is handed back to its caller in
//! [`InternOutcome::AlreadyCanonical`] instead of replacing it.

use std::hash::{Hash, Hasher};
use std::sync::LazyLock;

use serde::{Deserialize, Deserializer, Serialize, de};

use crate::decode::{self, Decode};

use crate::identifier::{HasIdentifier, Identifier, IdentifierPayload};
use crate::interned::{Canonical, InternOutcome, InternRegistry, Interned, require_default};

/// Open semantic tag attached to a compiler operation.
///
/// Two attributes are equal when they carry the same [`Identifier`], whatever
/// their descriptions say.
///
/// Decoding an attribute canonicalizes it only through the handle, so
/// deserialize a [`Canonical<OpAttribute>`]. Deserializing a bare
/// `OpAttribute` yields a value that no registry knows about.
#[derive(Debug, Serialize)]
pub struct OpAttribute {
    name: Identifier,
    description: String,
}

impl Decode for OpAttribute {
    type Payload = OpAttributePayload;

    fn build_from_payload<E: de::Error>(payload: Self::Payload) -> Result<Self, E> {
        Self::intern_registry().initialize();
        Ok(Self::create(payload.name.restore(), payload.description))
    }
}

/// Decoding checks every field of the payload before it restores the name, so
/// a rejected payload leaves the id counter untouched. An accepted payload
/// registers the shipped defaults first if this is the registry's first use,
/// so their names draw ids before the payload's name can exhaust the counter.
impl<'de> Deserialize<'de> for OpAttribute {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        decode::deserialize_via_payload(deserializer)
    }
}

/// An attribute payload, checked but with its name not yet restored.
#[derive(Deserialize)]
#[serde(rename = "OpAttribute", deny_unknown_fields)]
pub(crate) struct OpAttributePayload {
    name: IdentifierPayload,
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

/// Name of the attribute returned by [`get_commutative`].
static COMMUTATIVE_NAME: LazyLock<Identifier> =
    LazyLock::new(|| Identifier::new_unscoped("commutative"));

/// Name of the attribute returned by [`get_associative`].
static ASSOCIATIVE_NAME: LazyLock<Identifier> =
    LazyLock::new(|| Identifier::new_unscoped("associative"));

/// Name of the attribute returned by [`get_pure`].
static PURE_NAME: LazyLock<Identifier> = LazyLock::new(|| Identifier::new_unscoped("pure"));

/// Name of the attribute returned by [`get_elementwise`].
static ELEMENTWISE_NAME: LazyLock<Identifier> =
    LazyLock::new(|| Identifier::new_unscoped("elementwise"));

/// Build the attributes this module ships, in registration order.
///
/// The registry calls this once, on its first use, and keeps the instances it
/// builds. A clear registers those same instances again rather than building
/// new ones, so the shipped constants stay canonical for the life of the
/// process.
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

static COMMUTATIVE: LazyLock<Canonical<OpAttribute>> =
    LazyLock::new(|| require_default(&*COMMUTATIVE_NAME));

static ASSOCIATIVE: LazyLock<Canonical<OpAttribute>> =
    LazyLock::new(|| require_default(&*ASSOCIATIVE_NAME));

static PURE: LazyLock<Canonical<OpAttribute>> = LazyLock::new(|| require_default(&*PURE_NAME));

static ELEMENTWISE: LazyLock<Canonical<OpAttribute>> =
    LazyLock::new(|| require_default(&*ELEMENTWISE_NAME));

/// Return the attribute for ops whose output is invariant under operand swap.
#[must_use]
pub fn get_commutative() -> &'static Canonical<OpAttribute> {
    &COMMUTATIVE
}

/// Return the attribute for ops that compose associatively across
/// applications.
#[must_use]
pub fn get_associative() -> &'static Canonical<OpAttribute> {
    &ASSOCIATIVE
}

/// Return the attribute for ops that have no side effects and produce
/// deterministic outputs.
#[must_use]
pub fn get_pure() -> &'static Canonical<OpAttribute> {
    &PURE
}

/// Return the attribute for ops that act independently on each element of
/// their operands.
#[must_use]
pub fn get_elementwise() -> &'static Canonical<OpAttribute> {
    &ELEMENTWISE
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    use rstest::rstest;

    use crate::identifier::{IdSpaceExhausted, try_allocate_id};
    use crate::test_support::{
        RegistryGuard, assert_isolated_test_passes, compute_hash, has_counter_passed,
        hold_id_counter, is_isolated_run, reserve_far_ahead_ids, reserve_pinned_id, take_discarded,
    };

    /// Serializes the test that clears the process-wide registry against the
    /// tests that need their own entries to survive.
    static REGISTRY_GUARD: RegistryGuard = RegistryGuard::new();

    /// Return every attribute this module ships as a default.
    fn list_default_attributes() -> [&'static Canonical<OpAttribute>; 4] {
        [
            get_commutative(),
            get_associative(),
            get_pure(),
            get_elementwise(),
        ]
    }

    #[test]
    fn new_stores_the_name_and_description() {
        let _guard = REGISTRY_GUARD.hold();
        let name = Identifier::new("stores-name-and-description");
        let attribute = OpAttribute::new(name.clone(), "an attribute").into_canonical();

        assert_eq!(attribute.name(), &name);
        assert_eq!(attribute.description(), "an attribute");
    }

    #[test]
    fn has_identifier_returns_the_name() {
        let _guard = REGISTRY_GUARD.hold();
        let name = Identifier::new("has-identifier");
        let attribute = OpAttribute::new(name.clone(), "desc").into_canonical();

        assert_eq!(attribute.identifier(), &name);
    }

    #[test]
    fn intern_key_is_the_name() {
        let _guard = REGISTRY_GUARD.hold();
        let name = Identifier::new("intern-key");
        let attribute = OpAttribute::new(name.clone(), "desc").into_canonical();

        assert_eq!(attribute.intern_key(), &name);
    }

    #[test]
    fn new_keeps_the_first_attribute_canonical_for_a_repeated_name() {
        let _guard = REGISTRY_GUARD.hold();
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
        let _guard = REGISTRY_GUARD.hold();
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
        let _guard = REGISTRY_GUARD.hold();
        let name = Identifier::new("equality-ignores-description");
        let canonical = OpAttribute::new(name.clone(), "first description").into_canonical();

        let other = take_discarded(OpAttribute::new(name, "second description"));

        assert_eq!(*canonical, other);
        assert_eq!(other, *canonical);
    }

    #[test]
    fn equal_attributes_hash_equally() {
        let _guard = REGISTRY_GUARD.hold();
        let name = Identifier::new("hash-ignores-description");
        let canonical = OpAttribute::new(name.clone(), "first description").into_canonical();

        let other = take_discarded(OpAttribute::new(name, "second description"));

        assert_eq!(compute_hash(&*canonical), compute_hash(&other));
    }

    #[test]
    fn attributes_with_different_names_are_unequal() {
        let _guard = REGISTRY_GUARD.hold();
        let left = OpAttribute::new(Identifier::new("a"), "desc").into_canonical();
        let right = OpAttribute::new(Identifier::new("b"), "desc").into_canonical();

        assert_ne!(*left, *right);
    }

    #[test]
    fn canonical_attributes_can_be_collected_into_a_set() {
        let _guard = REGISTRY_GUARD.hold();
        let tags: HashSet<Canonical<OpAttribute>> = [get_commutative().clone(), get_pure().clone()]
            .into_iter()
            .collect();

        assert!(tags.contains(get_commutative()));
        assert!(tags.contains(get_pure()));
        assert!(!tags.contains(get_associative()));
        assert_eq!(tags.len(), 2);
    }

    #[test]
    fn a_repeated_canonical_attribute_collapses_to_one_set_entry() {
        let _guard = REGISTRY_GUARD.hold();
        let tags: HashSet<Canonical<OpAttribute>> = [
            get_commutative().clone(),
            get_commutative().clone(),
            get_pure().clone(),
        ]
        .into_iter()
        .collect();

        let expected: HashSet<Canonical<OpAttribute>> =
            [get_commutative().clone(), get_pure().clone()]
                .into_iter()
                .collect();
        assert_eq!(tags, expected);
    }

    /// Test each shipped default attribute is the canonical entry for its
    /// name.
    #[rstest]
    #[case::commutative(get_commutative)]
    #[case::associative(get_associative)]
    #[case::pure(get_pure)]
    #[case::elementwise(get_elementwise)]
    fn a_default_attribute_is_registered_under_its_name(
        #[case] get_default: fn() -> &'static Canonical<OpAttribute>,
    ) {
        let _guard = REGISTRY_GUARD.hold();
        let default = get_default();

        assert_eq!(
            OpAttribute::intern_registry().get(default.name()),
            Some(default.clone())
        );
    }

    #[test]
    fn the_default_attributes_are_pairwise_distinct() {
        let _guard = REGISTRY_GUARD.hold();
        let defaults = list_default_attributes();

        for (index, left) in defaults.iter().enumerate() {
            for right in &defaults[index + 1..] {
                assert_ne!(left, right);
                assert_ne!(left.name(), right.name());
            }
        }
    }

    /// Test each shipped default attribute carries a non-empty description.
    #[rstest]
    #[case::commutative(get_commutative)]
    #[case::associative(get_associative)]
    #[case::pure(get_pure)]
    #[case::elementwise(get_elementwise)]
    fn a_default_attribute_carries_a_non_empty_description(
        #[case] get_default: fn() -> &'static Canonical<OpAttribute>,
    ) {
        let _guard = REGISTRY_GUARD.hold();

        let description = get_default().description();

        assert!(!description.trim().is_empty(), "{description:?}");
    }

    #[test]
    fn clearing_the_registry_keeps_the_default_attributes_canonical() {
        let _guard = REGISTRY_GUARD.hold_exclusively();
        let name = Identifier::new("dropped-by-clear");
        let dropped = OpAttribute::new(name.clone(), "dropped").into_canonical();
        assert_eq!(OpAttribute::intern_registry().get(&name), Some(dropped));

        OpAttribute::intern_registry().clear();

        assert_eq!(OpAttribute::intern_registry().get(&name), None);
        for default in list_default_attributes() {
            assert_eq!(
                OpAttribute::intern_registry().get(default.name()),
                Some(default.clone())
            );
        }
    }

    #[test]
    fn an_attribute_encodes_as_its_name_and_description() {
        let _guard = REGISTRY_GUARD.hold();
        let id = reserve_pinned_id("encode-anchor");
        let name = Identifier::restore(id, "encoded".to_string());
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
        let _guard = REGISTRY_GUARD.hold();
        let attribute = OpAttribute::new(Identifier::new("round-trip"), "desc").into_canonical();

        let json = serde_json::to_string(&*attribute).unwrap();
        let restored: Canonical<OpAttribute> = serde_json::from_str(&json).unwrap();

        assert_eq!(restored, attribute);
    }

    #[test]
    fn decoding_a_registered_name_returns_the_canonical_attribute() {
        let _guard = REGISTRY_GUARD.hold();
        let json = serde_json::to_string(get_pure()).unwrap();

        let restored: Canonical<OpAttribute> = serde_json::from_str(&json).unwrap();

        assert_eq!(restored, *get_pure());
    }

    #[test]
    fn decoding_an_unregistered_name_registers_the_decoded_attribute() {
        let _guard = REGISTRY_GUARD.hold();
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
        let _guard = REGISTRY_GUARD.hold();
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
        let _guard = REGISTRY_GUARD.hold();
        let name = Identifier::new("matching-description");
        let canonical = OpAttribute::new(name.clone(), "matching").into_canonical();

        let discarded = take_discarded(OpAttribute::new(name, "matching"));

        assert_eq!(discarded.description(), canonical.description());
        assert_eq!(discarded, *canonical);
    }

    /// Test a payload with an unknown field or without its description is
    /// rejected, naming the offending field.
    #[rstest]
    #[case::unknown_field(",\"description\":\"desc\",\"surprise\":1", "surprise")]
    #[case::missing_description("", "description")]
    fn decoding_a_malformed_payload_is_rejected(
        #[case] fields_after_the_name: &str,
        #[case] expected_message: &str,
    ) {
        let _guard = REGISTRY_GUARD.hold();
        let id = reserve_pinned_id("malformed-payload-anchor");
        let json =
            format!("{{\"name\":{{\"id\":{id},\"name_hint\":\"x\"}}{fields_after_the_name}}}");

        let error = serde_json::from_str::<Canonical<OpAttribute>>(&json).unwrap_err();

        assert!(error.to_string().contains(expected_message), "{error}");
    }

    /// Test a payload rejected for a field after its name leaves the name's
    /// id unrestored.
    #[rstest]
    #[case::trailing_unknown_field(",\"description\":\"desc\",\"zzz\":1", "zzz")]
    #[case::mistyped_trailing_description(",\"description\":3", "invalid type")]
    fn a_payload_rejected_after_its_name_restores_no_name(
        #[case] fields_after_the_name: &str,
        #[case] expected_message: &str,
    ) {
        let _guard = REGISTRY_GUARD.hold();
        let _counter = hold_id_counter();
        let [id] = reserve_far_ahead_ids("rejected-after-name-anchor");
        let json =
            format!("{{\"name\":{{\"id\":{id},\"name_hint\":\"a\"}}{fields_after_the_name}}}");

        let error = serde_json::from_str::<Canonical<OpAttribute>>(&json).unwrap_err();

        assert!(error.to_string().contains(expected_message), "{error}");
        assert!(!has_counter_passed(id));
    }

    #[test]
    fn debug_mentions_the_name_hint_and_description() {
        let _guard = REGISTRY_GUARD.hold();
        let attribute =
            OpAttribute::new(Identifier::new("debug-attribute"), "debug desc").into_canonical();

        let rendered = format!("{attribute:?}");

        assert!(rendered.contains("debug-attribute"), "{rendered}");
        assert!(rendered.contains("debug desc"), "{rendered}");
    }

    /// Return the JSON payload of an attribute named with the largest
    /// issuable id, which exhausts the id counter when restored.
    fn encode_largest_id_payload() -> String {
        format!(
            "{{\"name\":{{\"id\":{},\"name_hint\":\"largest\"}},\"description\":\"desc\"}}",
            u64::MAX - 1
        )
    }

    /// Check that the shipped defaults survive a decode that exhausts the id
    /// counter before the registry's first use. Only meaningful in the child
    /// process that
    /// [`decoding_the_largest_id_as_the_first_use_keeps_the_defaults`]
    /// starts.
    #[test]
    #[ignore = "exhausts the process-global counter; run through assert_isolated_test_passes"]
    fn decoding_the_largest_id_as_the_first_use_keeps_the_defaults_in_isolation() {
        if !is_isolated_run() {
            return;
        }

        let restored: Canonical<OpAttribute> =
            serde_json::from_str(&encode_largest_id_payload()).expect("u64::MAX - 1 is valid");

        assert_eq!(restored.name().id(), u64::MAX - 1);
        assert_eq!(try_allocate_id(), Err(IdSpaceExhausted));
        assert_eq!(get_commutative().name().name_hint(), "commutative");
        assert_eq!(get_elementwise().name().name_hint(), "elementwise");
    }

    #[test]
    fn decoding_the_largest_id_as_the_first_use_keeps_the_defaults() {
        assert_isolated_test_passes(
            "op_attribute::tests::\
             decoding_the_largest_id_as_the_first_use_keeps_the_defaults_in_isolation",
        );
    }

    /// Check that decoding a bare attribute, which interns nothing, still
    /// leaves the shipped defaults buildable after it exhausts the id
    /// counter. Only meaningful in the child process that
    /// [`decoding_a_bare_largest_id_as_the_first_use_keeps_the_defaults`]
    /// starts.
    #[test]
    #[ignore = "exhausts the process-global counter; run through assert_isolated_test_passes"]
    fn decoding_a_bare_largest_id_as_the_first_use_keeps_the_defaults_in_isolation() {
        if !is_isolated_run() {
            return;
        }

        let restored: OpAttribute =
            serde_json::from_str(&encode_largest_id_payload()).expect("u64::MAX - 1 is valid");

        assert_eq!(restored.name().id(), u64::MAX - 1);
        assert_eq!(try_allocate_id(), Err(IdSpaceExhausted));
        assert_eq!(get_pure().name().name_hint(), "pure");
    }

    #[test]
    fn decoding_a_bare_largest_id_as_the_first_use_keeps_the_defaults() {
        assert_isolated_test_passes(
            "op_attribute::tests::\
             decoding_a_bare_largest_id_as_the_first_use_keeps_the_defaults_in_isolation",
        );
    }
}
