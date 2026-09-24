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
//!
//! [`Identifier`]: crate::identifier::Identifier
//! [`InternOutcome::AlreadyCanonical`]: crate::interned::InternOutcome::AlreadyCanonical

use crate::described_tag::define_described_tag;

define_described_tag! {
    /// Open semantic tag attached to a compiler operation.
    ///
    /// Two attributes are equal when they carry the same [`Identifier`], whatever
    /// their descriptions say.
    ///
    /// Decoding an attribute canonicalizes it only through the handle, so
    /// deserialize a [`Canonical<OpAttribute>`]. Deserializing a bare
    /// `OpAttribute` yields a value that no registry knows about.
    ///
    /// [`Identifier`]: crate::identifier::Identifier
    /// [`Canonical<OpAttribute>`]: crate::interned::Canonical
    pub struct OpAttribute;
    payload OpAttributePayload as "OpAttribute";
    noun "attribute";

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
    fn new;

    shipped by create_default_attributes {
        /// Return the attribute for ops whose output is invariant under operand swap.
        fn get_commutative => COMMUTATIVE, COMMUTATIVE_NAME =
            (COMMUTATIVE, "Op output is invariant under operand swap.");

        /// Return the attribute for ops that compose associatively across
        /// applications.
        fn get_associative => ASSOCIATIVE, ASSOCIATIVE_NAME =
            (ASSOCIATIVE, "Op composes associatively across applications.");

        /// Return the attribute for ops that have no side effects and produce
        /// deterministic outputs.
        fn get_pure => PURE, PURE_NAME =
            (PURE, "Op has no side effects and produces deterministic outputs.");

        /// Return the attribute for ops that act independently on each element of
        /// their operands.
        fn get_elementwise => ELEMENTWISE, ELEMENTWISE_NAME =
            (ELEMENTWISE, "Op acts independently on each element of its operands.");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    use rstest::rstest;

    use crate::identifier::{HasIdentifier, Identifier};
    use crate::interned::{Canonical, InternOutcome, Interned};
    use crate::test_support::{
        RegistryGuard, compute_hash, has_counter_passed, hold_id_counter, reserve_far_ahead_ids,
        reserve_pinned_id, take_discarded,
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

    /// Test each shipped attribute holds its fixed reserved id and name hint.
    #[rstest]
    #[case::commutative(get_commutative, 16, "commutative")]
    #[case::associative(get_associative, 17, "associative")]
    #[case::pure(get_pure, 18, "pure")]
    #[case::elementwise(get_elementwise, 19, "elementwise")]
    fn a_shipped_attribute_holds_its_reserved_id(
        #[case] get_default: fn() -> &'static Canonical<OpAttribute>,
        #[case] id: u64,
        #[case] name_hint: &str,
    ) {
        let name = get_default().name();

        assert_eq!((name.id(), name.name_hint()), (id, name_hint));
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
        let name = Identifier::try_restore(id, "encoded").expect("the id is below the cap");
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
}
