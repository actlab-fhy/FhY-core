//! User-story tests for `fhy_core::op_attribute` and `fhy_core::value_domain`.
//!
//! Public API only. These tests share the process-wide `OpAttribute` and
//! `ValueDomain` registries, which are append-only, and each story registers
//! identifiers of its own.

use crate::support::stack as stack_support;

use std::collections::HashSet;
use std::sync::LazyLock;

use fhy_core::identifier::Identifier;
use fhy_core::interned::{Canonical, Interned};
use fhy_core::op_attribute::OpAttribute;
use fhy_core::value_domain::ValueDomain;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use stack_support::{SMALL_STACK_DEPTH, run_on_small_stack};

/// A stand-in for a compiler op, carrying the semantic tags attached to it.
struct StoryOp {
    tags: HashSet<Canonical<OpAttribute>>,
}

impl StoryOp {
    /// Build an op tagged with `tags`, collapsing repeated canonical tags
    /// into one set entry.
    fn create(tags: impl IntoIterator<Item = Canonical<OpAttribute>>) -> Self {
        Self {
            tags: tags.into_iter().collect(),
        }
    }

    /// Return whether the op carries `tag`.
    fn has_tag(&self, tag: &Canonical<OpAttribute>) -> bool {
        self.tags.contains(tag)
    }

    /// Return how many distinct tags the op carries.
    fn count_tags(&self) -> usize {
        self.tags.len()
    }
}

/// Name of the layer-specific attribute this story registers for itself.
static IDEMPOTENT_NAME: LazyLock<Identifier> =
    LazyLock::new(|| Identifier::new("tagging-story-idempotent"));

/// Test an op's tag set holds shipped and layer-registered attributes, and a
/// same-name `OpAttribute` collapses into the existing entry.
#[test]
fn tagging_an_operation_with_shipped_and_layer_specific_attributes() {
    let idempotent = OpAttribute::register(
        IDEMPOTENT_NAME.clone(),
        "Applying the op twice changes nothing.",
    );

    let op = StoryOp::create([
        OpAttribute::commutative().clone(),
        OpAttribute::pure().clone(),
        idempotent.clone(),
    ]);

    assert_eq!(op.count_tags(), 3);
    assert!(op.has_tag(OpAttribute::commutative()));
    assert!(op.has_tag(OpAttribute::pure()));
    assert!(op.has_tag(&idempotent));
    assert!(!op.has_tag(OpAttribute::associative()));

    let same_name_again = OpAttribute::register(IDEMPOTENT_NAME.clone(), "a different description");
    let op_with_repeat = StoryOp::create([
        OpAttribute::commutative().clone(),
        OpAttribute::pure().clone(),
        idempotent.clone(),
        same_name_again,
    ]);

    assert_eq!(op_with_repeat.count_tags(), 3);
}

/// Name of this story's middle-tier domain, a child of `ValueDomain::data()`.
static TENSOR_NAME: LazyLock<Identifier> = LazyLock::new(|| Identifier::new("domain-story-tensor"));

/// Name of this story's leaf domain, a child of the tensor domain.
static TILE_NAME: LazyLock<Identifier> = LazyLock::new(|| Identifier::new("domain-story-tile"));

/// Name of this story's domain on an unrelated branch, a child of
/// `ValueDomain::address()`.
static TOKEN_NAME: LazyLock<Identifier> = LazyLock::new(|| Identifier::new("domain-story-token"));

/// Test a three-level domain hierarchy registered under `ValueDomain::data()`
/// relates each level to its ancestors via `is_subdomain_of`, and relates
/// none of them to a domain on an unrelated branch.
#[test]
fn a_three_level_domain_hierarchy_relates_its_levels() {
    let tensor = ValueDomain::register_child(
        TENSOR_NAME.clone(),
        "A tensor of concrete data.",
        ValueDomain::data(),
    )
    .expect("the domain registers");
    let tile =
        ValueDomain::register_child(TILE_NAME.clone(), "A tile carved from a tensor.", &tensor)
            .expect("the domain registers");
    let token = ValueDomain::register_child(
        TOKEN_NAME.clone(),
        "A control token, unrelated to the data branch.",
        ValueDomain::address(),
    )
    .expect("the domain registers");

    assert!(tile.is_subdomain_of(&tile));
    assert!(tile.is_subdomain_of(&tensor));
    assert!(tile.is_subdomain_of(ValueDomain::data()));
    assert!(!tile.is_subdomain_of(&token));
    assert!(!ValueDomain::data().is_subdomain_of(&tile));
    assert!(!token.is_subdomain_of(ValueDomain::data()));
}

/// Name of the attribute this story persists.
static PERSISTED_ATTRIBUTE_NAME: LazyLock<Identifier> =
    LazyLock::new(|| Identifier::new("persistence-story-attribute"));

/// Name of the domain this story persists, a child of `ValueDomain::address()`.
static PERSISTED_DOMAIN_NAME: LazyLock<Identifier> =
    LazyLock::new(|| Identifier::new("persistence-story-domain"));

/// A tagged operation as it would be written to and read back from storage.
#[derive(Serialize, Deserialize)]
struct PersistedOp {
    attribute: Canonical<OpAttribute>,
    domain: Canonical<ValueDomain>,
}

/// Test a tagged operation serializes to JSON and, decoded back, carries the
/// very same canonical `OpAttribute` and `ValueDomain` instances it was built
/// with.
#[test]
fn persisting_and_restoring_a_tagged_operation() {
    let attribute =
        OpAttribute::register(PERSISTED_ATTRIBUTE_NAME.clone(), "a persisted attribute");
    let domain = ValueDomain::register_child(
        PERSISTED_DOMAIN_NAME.clone(),
        "a persisted domain",
        ValueDomain::address(),
    )
    .expect("the domain registers");
    let persisted = PersistedOp {
        attribute: attribute.clone(),
        domain: domain.clone(),
    };

    let json = serde_json::to_string(&persisted).expect("op serializes");
    let restored: PersistedOp = serde_json::from_str(&json).expect("op deserializes");

    assert_eq!(restored.attribute, attribute);
    assert_eq!(restored.domain, domain);
    assert!(Canonical::ptr_eq(&restored.attribute, &attribute));
    assert!(Canonical::ptr_eq(&restored.domain, &domain));
}

/// Test registering an attribute under a name already registered keeps the
/// first attribute and its description.
#[test]
fn registering_a_known_attribute_keeps_the_first_description() {
    let name = Identifier::new("known-attribute-story");
    let first = OpAttribute::register(name.clone(), "the first description");

    let again = OpAttribute::register(name, "a later description");

    assert_eq!(again, first);
    assert_eq!(again.description(), "the first description");
}

/// One level of a domain chain's payload.
#[derive(Serialize)]
struct Level<'a> {
    name: &'a Identifier,
    description: &'a str,
}

/// Return the payload of the chain `levels`, root first.
fn encode_chain(levels: &[(&Identifier, &str)]) -> Value {
    let levels: Vec<Level<'_>> = levels
        .iter()
        .map(|&(name, description)| Level { name, description })
        .collect();
    serde_json::to_value(levels).expect("the chain encodes")
}

/// Test decoding a chain of three domains no registry knows registers every
/// level, each under the level above it.
#[test]
fn a_decoded_domain_chain_registers_every_level() {
    let names = ["chain-story-root", "chain-story-middle", "chain-story-leaf"].map(Identifier::new);
    let payload = encode_chain(&[
        (&names[0], "root"),
        (&names[1], "middle"),
        (&names[2], "leaf"),
    ]);

    let decoded: Canonical<ValueDomain> =
        serde_json::from_value(payload).expect("the chain decodes");

    let registered = names.each_ref().map(|name| {
        ValueDomain::intern_registry()
            .get(name)
            .expect("every level is registered")
    });
    assert!(Canonical::ptr_eq(&decoded, &registered[2]));
    assert_eq!(registered[0].parent(), None);
    assert_eq!(registered[1].parent(), Some(&registered[0]));
    assert_eq!(registered[2].parent(), Some(&registered[1]));
}

/// Test decoding a registered domain under a different parent is rejected,
/// and the registered domain keeps its parent and description.
#[test]
fn a_decoded_domain_under_another_parent_is_rejected_and_the_canonical_domain_is_unchanged() {
    let name = Identifier::new("reparented-story-domain");
    let canonical = ValueDomain::register_child(name.clone(), "under data", ValueDomain::data())
        .expect("the domain registers");
    let payload = encode_chain(&[
        (ValueDomain::address().name(), "address"),
        (&name, "under address"),
    ]);

    let result = serde_json::from_value::<Canonical<ValueDomain>>(payload);

    assert!(result.is_err(), "the reparented domain decoded");
    let registered = ValueDomain::intern_registry()
        .get(&name)
        .expect("the domain stays registered");
    assert_eq!(registered, canonical);
    assert_eq!(registered.parent(), Some(ValueDomain::data()));
    assert_eq!(registered.description(), "under data");
}

/// Test a payload rejected for its structure leaves the registered domain of
/// its name as it was.
#[test]
fn a_rejected_payload_leaves_the_canonical_domain_unchanged() {
    let name = Identifier::new("rejected-story-domain");
    let canonical =
        ValueDomain::register_root(name.clone(), "registered").expect("the domain registers");
    let mut payload = encode_chain(&[(&name, "rejected")]);
    payload[0]["unexpected"] = json!(1);

    let result = serde_json::from_value::<Canonical<ValueDomain>>(payload);

    assert!(result.is_err(), "the malformed payload decoded");
    let registered = ValueDomain::intern_registry()
        .get(&name)
        .expect("the domain stays registered");
    assert_eq!(registered, canonical);
    assert_eq!(registered.description(), "registered");
    assert_eq!(registered.parent(), None);
}

/// Test a chain as deep as a stack of a few hundred kilobytes allows no
/// recursion over decodes from JSON and from postcard, registering every
/// level: the chain is encoded flat and decoded one level at a time.
#[test]
fn value_domain_decodes_a_deep_chain_on_a_small_stack() {
    let json_names: Vec<Identifier> = (0..SMALL_STACK_DEPTH)
        .map(|_| Identifier::new("deep-json-level"))
        .collect();
    let postcard_names: Vec<Identifier> = (0..SMALL_STACK_DEPTH)
        .map(|_| Identifier::new("deep-postcard-level"))
        .collect();
    let json_levels: Vec<Level<'_>> = json_names
        .iter()
        .map(|name| Level {
            name,
            description: "deep",
        })
        .collect();
    let postcard_levels: Vec<Level<'_>> = postcard_names
        .iter()
        .map(|name| Level {
            name,
            description: "deep",
        })
        .collect();
    let json = serde_json::to_string(&json_levels).expect("the chain encodes");
    let bytes = postcard::to_allocvec(&postcard_levels).expect("the chain encodes");

    let (json_leaf, postcard_leaf) = run_on_small_stack(move || {
        let from_json: Canonical<ValueDomain> =
            serde_json::from_str(&json).expect("the deep chain decodes from JSON");
        let from_postcard: Canonical<ValueDomain> =
            postcard::from_bytes(&bytes).expect("the deep chain decodes from postcard");
        (from_json.name().id(), from_postcard.name().id())
    });

    assert_eq!(json_leaf, json_names[SMALL_STACK_DEPTH - 1].id());
    assert_eq!(postcard_leaf, postcard_names[SMALL_STACK_DEPTH - 1].id());
    for names in [&json_names, &postcard_names] {
        let leaf = ValueDomain::intern_registry()
            .get(&names[SMALL_STACK_DEPTH - 1])
            .expect("the leaf is registered");
        let parent = leaf.parent().expect("the leaf has a parent");
        assert_eq!(parent.name(), &names[SMALL_STACK_DEPTH - 2]);
        assert!(
            ValueDomain::intern_registry().get(&names[0]).is_some(),
            "the root is registered"
        );
    }
}
