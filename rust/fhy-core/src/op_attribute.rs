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
//! through [`crate::interned`], so call the shipped accessors such as
//! [`OpAttribute::commutative`] rather than building a fresh attribute with
//! the same name hint. Identifiers
//! compare by id, and a second `Identifier::new("commutative")` is a
//! different key.
//!
//! A `description` is human-readable metadata. It takes no part in equality,
//! hashing or interning: the first attribute registered under an identifier
//! stays canonical, and a later registration of that name returns it,
//! dropping its own description.
//!
//! [`Identifier`]: crate::identifier::Identifier

use std::sync::LazyLock;

use crate::described_tag::{DescribedTag, TagKind, require_shipped, sealed};
use crate::identifier::reserved::{self, ReservedIdentifier};
use crate::interned::{Canonical, InternRegistry};

/// The vocabulary of [`OpAttribute`]s.
#[expect(
    clippy::exhaustive_enums,
    reason = "an uninhabited marker type, with no variants to add"
)]
#[derive(Debug)]
pub enum OpAttributeVocabulary {}

impl TagKind for OpAttributeVocabulary {}

impl sealed::Sealed for OpAttributeVocabulary {
    const TYPE_NAME: &'static str = "OpAttribute";

    fn registry() -> &'static InternRegistry<OpAttribute> {
        static REGISTRY: InternRegistry<OpAttribute> =
            InternRegistry::with_defaults(create_default_attributes);
        &REGISTRY
    }
}

/// Open semantic tag attached to a compiler operation.
///
/// Two attributes are equal when they carry the same
/// [`Identifier`](crate::identifier::Identifier), whatever their
/// descriptions say.
///
/// Only a [`Canonical<OpAttribute>`] decodes, registering the attribute
/// unless its name is registered already.
pub type OpAttribute = DescribedTag<OpAttributeVocabulary>;

/// The shipped attributes, in registration order, with their descriptions.
const SHIPPED_ATTRIBUTES: [(ReservedIdentifier, &str); 4] = [
    (
        reserved::COMMUTATIVE,
        "Op output is invariant under operand swap.",
    ),
    (
        reserved::ASSOCIATIVE,
        "Op composes associatively across applications.",
    ),
    (
        reserved::PURE,
        "Op has no side effects and produces deterministic outputs.",
    ),
    (
        reserved::ELEMENTWISE,
        "Op acts independently on each element of its operands.",
    ),
];

/// Build the attributes this module ships, in registration order.
///
/// The registry calls this once, on its first use, and keeps the instances
/// it builds, so the shipped attributes stay canonical for the life of the
/// process.
fn create_default_attributes() -> Vec<OpAttribute> {
    SHIPPED_ATTRIBUTES
        .iter()
        .map(|&(entry, description)| OpAttribute::create_shipped(entry, description))
        .collect()
}

static COMMUTATIVE: LazyLock<Canonical<OpAttribute>> =
    LazyLock::new(|| require_shipped(reserved::COMMUTATIVE));

static ASSOCIATIVE: LazyLock<Canonical<OpAttribute>> =
    LazyLock::new(|| require_shipped(reserved::ASSOCIATIVE));

static PURE: LazyLock<Canonical<OpAttribute>> = LazyLock::new(|| require_shipped(reserved::PURE));

static ELEMENTWISE: LazyLock<Canonical<OpAttribute>> =
    LazyLock::new(|| require_shipped(reserved::ELEMENTWISE));

impl DescribedTag<OpAttributeVocabulary> {
    /// Return the attribute for ops whose output is invariant under operand
    /// swap.
    #[must_use]
    pub fn commutative() -> &'static Canonical<OpAttribute> {
        &COMMUTATIVE
    }

    /// Return the attribute for ops that compose associatively across
    /// applications.
    #[must_use]
    pub fn associative() -> &'static Canonical<OpAttribute> {
        &ASSOCIATIVE
    }

    /// Return the attribute for ops that have no side effects and produce
    /// deterministic outputs.
    #[must_use]
    pub fn pure() -> &'static Canonical<OpAttribute> {
        &PURE
    }

    /// Return the attribute for ops that act independently on each element
    /// of their operands.
    #[must_use]
    pub fn elementwise() -> &'static Canonical<OpAttribute> {
        &ELEMENTWISE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    use rstest::rstest;

    use crate::identifier::{HasIdentifier, Identifier};
    use crate::interned::{Canonical, Interned};

    /// Return every attribute this module ships as a default.
    fn list_default_attributes() -> [&'static Canonical<OpAttribute>; 4] {
        [
            OpAttribute::commutative(),
            OpAttribute::associative(),
            OpAttribute::pure(),
            OpAttribute::elementwise(),
        ]
    }

    #[test]
    fn new_stores_the_name_and_description() {
        let name = Identifier::new("stores-name-and-description");
        let attribute = OpAttribute::register(name.clone(), "an attribute");

        assert_eq!(attribute.name(), &name);
        assert_eq!(attribute.description(), "an attribute");
    }

    #[test]
    fn has_identifier_returns_the_name() {
        let name = Identifier::new("has-identifier");
        let attribute = OpAttribute::register(name.clone(), "desc");

        assert_eq!(attribute.identifier(), &name);
    }

    #[test]
    fn intern_key_is_the_name() {
        let name = Identifier::new("intern-key");
        let attribute = OpAttribute::register(name.clone(), "desc");

        assert_eq!(attribute.intern_key(), &name);
    }

    #[test]
    fn register_keeps_the_first_attribute_for_a_repeated_name() {
        let name = Identifier::new("repeated-name");
        let first = OpAttribute::register(name.clone(), "first");

        let second = OpAttribute::register(name.clone(), "second");

        assert_eq!(second, first);
        assert_eq!(second.description(), "first");
        assert_eq!(OpAttribute::intern_registry().get(&name), Some(first));
    }

    #[test]
    fn register_returns_the_first_attribute_for_a_known_name() {
        let name = Identifier::new("known-name");
        let first = OpAttribute::register(name.clone(), "first");

        let again = OpAttribute::register(name, "first");

        assert_eq!(again, first);
    }

    #[test]
    fn identifiers_sharing_a_name_hint_intern_separately() {
        let first_name = Identifier::new("dup");
        let second_name = Identifier::new("dup");
        let first = OpAttribute::register(first_name.clone(), "a");
        let second = OpAttribute::register(second_name.clone(), "b");

        assert_ne!(first, second);
        assert_eq!(OpAttribute::intern_registry().get(&first_name), Some(first));
        assert_eq!(
            OpAttribute::intern_registry().get(&second_name),
            Some(second)
        );
    }

    #[test]
    fn attributes_with_different_names_are_unequal() {
        let left = OpAttribute::register(Identifier::new("a"), "desc");
        let right = OpAttribute::register(Identifier::new("b"), "desc");

        assert_ne!(*left, *right);
    }

    #[test]
    fn canonical_attributes_can_be_collected_into_a_set() {
        let tags: HashSet<Canonical<OpAttribute>> = [
            OpAttribute::commutative().clone(),
            OpAttribute::pure().clone(),
        ]
        .into_iter()
        .collect();

        assert!(tags.contains(OpAttribute::commutative()));
        assert!(tags.contains(OpAttribute::pure()));
        assert!(!tags.contains(OpAttribute::associative()));
        assert_eq!(tags.len(), 2);
    }

    #[test]
    fn a_repeated_canonical_attribute_collapses_to_one_set_entry() {
        let tags: HashSet<Canonical<OpAttribute>> = [
            OpAttribute::commutative().clone(),
            OpAttribute::commutative().clone(),
            OpAttribute::pure().clone(),
        ]
        .into_iter()
        .collect();

        let expected: HashSet<Canonical<OpAttribute>> = [
            OpAttribute::commutative().clone(),
            OpAttribute::pure().clone(),
        ]
        .into_iter()
        .collect();
        assert_eq!(tags, expected);
    }

    /// Test each shipped default attribute is the canonical entry for its
    /// name.
    #[rstest]
    #[case::commutative(OpAttribute::commutative)]
    #[case::associative(OpAttribute::associative)]
    #[case::pure(OpAttribute::pure)]
    #[case::elementwise(OpAttribute::elementwise)]
    fn a_default_attribute_is_registered_under_its_name(
        #[case] get_default: fn() -> &'static Canonical<OpAttribute>,
    ) {
        let default = get_default();

        assert_eq!(
            OpAttribute::intern_registry().get(default.name()),
            Some(default.clone())
        );
    }

    /// Test each shipped attribute holds its fixed reserved id and name hint.
    #[rstest]
    #[case::commutative(OpAttribute::commutative, 16, "commutative")]
    #[case::associative(OpAttribute::associative, 17, "associative")]
    #[case::pure(OpAttribute::pure, 18, "pure")]
    #[case::elementwise(OpAttribute::elementwise, 19, "elementwise")]
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
    #[case::commutative(OpAttribute::commutative)]
    #[case::associative(OpAttribute::associative)]
    #[case::pure(OpAttribute::pure)]
    #[case::elementwise(OpAttribute::elementwise)]
    fn a_default_attribute_carries_a_non_empty_description(
        #[case] get_default: fn() -> &'static Canonical<OpAttribute>,
    ) {
        let description = get_default().description();

        assert!(!description.trim().is_empty(), "{description:?}");
    }

    #[test]
    fn an_attribute_encodes_as_its_name_and_description() {
        let name = Identifier::new("encoded");
        let id = name.id();
        let attribute = OpAttribute::register(name, "a description");

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
        let attribute = OpAttribute::register(Identifier::new("round-trip"), "desc");

        let json = serde_json::to_string(&*attribute).unwrap();
        let restored: Canonical<OpAttribute> = serde_json::from_str(&json).unwrap();

        assert!(Canonical::ptr_eq(&restored, &attribute));
    }

    #[test]
    fn decoding_a_registered_name_returns_the_canonical_attribute() {
        let json = serde_json::to_string(OpAttribute::pure()).unwrap();

        let restored: Canonical<OpAttribute> = serde_json::from_str(&json).unwrap();

        assert!(Canonical::ptr_eq(&restored, OpAttribute::pure()));
    }

    #[test]
    fn decoding_an_unregistered_name_registers_the_decoded_attribute() {
        let id = Identifier::new("unregistered-decode").id();
        let json = format!(
            "{{\"name\":{{\"id\":{id},\"name_hint\":\"never-registered\"}},\
             \"description\":\"fresh from decode\"}}"
        );

        let restored: Canonical<OpAttribute> = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.description(), "fresh from decode");
        let registered = OpAttribute::intern_registry().get(restored.name());
        assert!(registered.is_some_and(|registered| Canonical::ptr_eq(&registered, &restored)));
    }

    #[test]
    fn decoding_a_divergent_description_keeps_the_canonical_one() {
        let name = Identifier::new("divergent-description");
        let canonical = OpAttribute::register(name.clone(), "original description");
        let json = format!(
            "{{\"name\":{{\"id\":{},\"name_hint\":\"{}\"}},\
             \"description\":\"divergent description\"}}",
            name.id(),
            name.name_hint()
        );

        let restored: Canonical<OpAttribute> = serde_json::from_str(&json).unwrap();

        assert!(Canonical::ptr_eq(&restored, &canonical));
        assert_eq!(restored.description(), "original description");
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
        let id = Identifier::new("malformed-payload").id();
        let json =
            format!("{{\"name\":{{\"id\":{id},\"name_hint\":\"x\"}}{fields_after_the_name}}}");

        let error = serde_json::from_str::<Canonical<OpAttribute>>(&json).unwrap_err();

        assert!(error.to_string().contains(expected_message), "{error}");
    }

    /// Test an attribute round-trips through postcard, a format that is not
    /// self-describing, back to its canonical handle.
    #[test]
    fn an_attribute_round_trips_through_postcard() {
        let attribute = OpAttribute::register(Identifier::new("postcard-round-trip"), "desc");

        let bytes = postcard::to_allocvec(&*attribute).expect("the attribute encodes");
        let restored: Canonical<OpAttribute> =
            postcard::from_bytes(&bytes).expect("the attribute decodes");

        assert!(Canonical::ptr_eq(&restored, &attribute));
    }

    /// Test a payload with a reserved id and any description decodes to the
    /// shipped attribute, keeping its canonical description.
    #[test]
    fn a_reserved_id_with_a_conflicting_description_decodes_to_the_shipped_attribute() {
        let json = r#"{"name":{"id":17,"name_hint":"renamed"},"description":"conflicting"}"#;

        let restored: Canonical<OpAttribute> = serde_json::from_str(json).unwrap();

        assert!(Canonical::ptr_eq(&restored, OpAttribute::associative()));
        assert_eq!(
            restored.description(),
            OpAttribute::associative().description()
        );
    }

    /// Test a payload with an id in the reserved block that no shipped tag
    /// holds decodes as an ordinary attribute.
    #[test]
    fn an_unassigned_reserved_id_decodes_as_an_ordinary_attribute() {
        let json = r#"{"name":{"id":500,"name_hint":"unassigned"},"description":"ordinary"}"#;

        let restored: Canonical<OpAttribute> = serde_json::from_str(json).unwrap();

        assert_eq!(restored.name().id(), 500);
        assert_eq!(restored.description(), "ordinary");
    }

    /// Test a payload naming its id twice is rejected.
    #[test]
    fn a_payload_with_a_duplicate_id_is_rejected() {
        let id = Identifier::new("duplicate-id").id();
        let json = format!(
            "{{\"name\":{{\"id\":{id},\"id\":{id},\"name_hint\":\"x\"}},\"description\":\"d\"}}"
        );

        let error = serde_json::from_str::<Canonical<OpAttribute>>(&json).unwrap_err();

        assert!(
            error.to_string().contains("duplicate field `id`"),
            "{error}"
        );
    }

    #[test]
    fn debug_mentions_the_name_hint_and_description() {
        let attribute = OpAttribute::register(Identifier::new("debug-attribute"), "debug desc");

        let rendered = format!("{attribute:?}");

        assert!(rendered.contains("debug-attribute"), "{rendered}");
        assert!(rendered.contains("debug desc"), "{rendered}");
    }
}
