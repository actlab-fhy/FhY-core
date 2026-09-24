//! User-story tests for `fhy_core::op_attribute` and `fhy_core::value_domain`.
//!
//! Public API only. These tests share the process-wide `OpAttribute` and
//! `ValueDomain` registries and run in parallel, so none of them clears a
//! registry, and each story creates its identifiers under a name hint unique
//! to that story.

use std::collections::HashSet;
use std::sync::LazyLock;

use fhy_core::identifier::Identifier;
use fhy_core::interned::{Canonical, Interned};
use fhy_core::op_attribute::{OpAttribute, get_associative, get_commutative, get_pure};
use fhy_core::value_domain::{ValueDomain, get_address_domain, get_data_domain};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

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
    let idempotent = OpAttribute::new(
        IDEMPOTENT_NAME.clone(),
        "Applying the op twice changes nothing.",
    )
    .into_canonical();

    let op = StoryOp::create([
        get_commutative().clone(),
        get_pure().clone(),
        idempotent.clone(),
    ]);

    assert_eq!(op.count_tags(), 3);
    assert!(op.has_tag(get_commutative()));
    assert!(op.has_tag(get_pure()));
    assert!(op.has_tag(&idempotent));
    assert!(!op.has_tag(get_associative()));

    let same_name_again =
        OpAttribute::new(IDEMPOTENT_NAME.clone(), "a different description").into_canonical();
    let op_with_repeat = StoryOp::create([
        get_commutative().clone(),
        get_pure().clone(),
        idempotent.clone(),
        same_name_again,
    ]);

    assert_eq!(op_with_repeat.count_tags(), 3);
}

/// Name of this story's middle-tier domain, a child of `get_data_domain()`.
static TENSOR_NAME: LazyLock<Identifier> = LazyLock::new(|| Identifier::new("domain-story-tensor"));

/// Name of this story's leaf domain, a child of the tensor domain.
static TILE_NAME: LazyLock<Identifier> = LazyLock::new(|| Identifier::new("domain-story-tile"));

/// Name of this story's domain on an unrelated branch, a child of
/// `get_address_domain()`.
static TOKEN_NAME: LazyLock<Identifier> = LazyLock::new(|| Identifier::new("domain-story-token"));

/// Test a three-level domain hierarchy registered under `get_data_domain()`
/// relates each level to its ancestors via `is_subdomain_of`, and relates
/// none of them to a domain on an unrelated branch.
#[test]
fn a_three_level_domain_hierarchy_relates_its_levels() {
    let tensor = ValueDomain::new(
        TENSOR_NAME.clone(),
        "A tensor of concrete data.",
        Some(get_data_domain().clone()),
    )
    .into_canonical();
    let tile = ValueDomain::new(
        TILE_NAME.clone(),
        "A tile carved from a tensor.",
        Some(tensor.clone()),
    )
    .into_canonical();
    let token = ValueDomain::new(
        TOKEN_NAME.clone(),
        "A control token, unrelated to the data branch.",
        Some(get_address_domain().clone()),
    )
    .into_canonical();

    assert!(tile.is_subdomain_of(&tile));
    assert!(tile.is_subdomain_of(&tensor));
    assert!(tile.is_subdomain_of(get_data_domain()));
    assert!(!tile.is_subdomain_of(&token));
    assert!(!get_data_domain().is_subdomain_of(&tile));
    assert!(!token.is_subdomain_of(get_data_domain()));
}

/// Name of the attribute this story persists.
static PERSISTED_ATTRIBUTE_NAME: LazyLock<Identifier> =
    LazyLock::new(|| Identifier::new("persistence-story-attribute"));

/// Name of the domain this story persists, a child of `get_address_domain()`.
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
/// with, since `==` on `Canonical` is identity.
#[test]
fn persisting_and_restoring_a_tagged_operation() {
    let attribute = OpAttribute::new(PERSISTED_ATTRIBUTE_NAME.clone(), "a persisted attribute")
        .into_canonical();
    let domain = ValueDomain::new(
        PERSISTED_DOMAIN_NAME.clone(),
        "a persisted domain",
        Some(get_address_domain().clone()),
    )
    .into_canonical();
    let persisted = PersistedOp {
        attribute: attribute.clone(),
        domain: domain.clone(),
    };

    let json = serde_json::to_string(&persisted).expect("op serializes");
    let restored: PersistedOp = serde_json::from_str(&json).expect("op deserializes");

    assert_eq!(restored.attribute, attribute);
    assert_eq!(restored.domain, domain);
}

/// Test registering an attribute under a name already registered keeps the
/// first attribute and its description.
#[test]
fn registering_a_known_attribute_keeps_the_first_description() {
    let name = Identifier::new("known-attribute-story");
    let first = OpAttribute::new(name.clone(), "the first description").into_canonical();

    let again = OpAttribute::new(name, "a later description").into_canonical();

    assert_eq!(again, first);
    assert_eq!(again.description(), "the first description");
}

/// Return the payload of the domain `name` under the payload `parent`.
fn encode_domain(name: &Identifier, description: &str, parent: &Value) -> Value {
    json!({
        "name": {"id": name.id(), "name_hint": name.name_hint()},
        "description": description,
        "parent": parent,
    })
}

/// Test decoding a chain of three domains no registry knows registers every
/// level, each under the level above it.
#[test]
fn a_decoded_domain_chain_registers_every_level() {
    let names = ["chain-story-root", "chain-story-middle", "chain-story-leaf"].map(Identifier::new);
    let root = encode_domain(&names[0], "root", &Value::Null);
    let middle = encode_domain(&names[1], "middle", &root);
    let leaf = encode_domain(&names[2], "leaf", &middle);

    let decoded: Canonical<ValueDomain> = serde_json::from_value(leaf).expect("the chain decodes");

    let registered = names.each_ref().map(|name| {
        ValueDomain::intern_registry()
            .get(name)
            .expect("every level is registered")
    });
    assert_eq!(decoded, registered[2]);
    assert_eq!(registered[0].parent(), None);
    assert_eq!(registered[1].parent(), Some(&registered[0]));
    assert_eq!(registered[2].parent(), Some(&registered[1]));
}

/// Test decoding a registered domain under a different parent is rejected,
/// and the registered domain keeps its parent and description.
#[test]
fn a_decoded_domain_under_another_parent_is_rejected_and_the_canonical_domain_is_unchanged() {
    let name = Identifier::new("reparented-story-domain");
    let canonical = ValueDomain::new(name.clone(), "under data", Some(get_data_domain().clone()))
        .into_canonical();
    let payload = encode_domain(
        &name,
        "under address",
        &serde_json::to_value(get_address_domain()).expect("the domain encodes"),
    );

    let result = serde_json::from_value::<Canonical<ValueDomain>>(payload);

    assert!(result.is_err(), "the reparented domain decoded");
    let registered = ValueDomain::intern_registry()
        .get(&name)
        .expect("the domain stays registered");
    assert_eq!(registered, canonical);
    assert_eq!(registered.parent(), Some(get_data_domain()));
    assert_eq!(registered.description(), "under data");
}

/// Test a payload rejected for its structure leaves the registered domain of
/// its name as it was.
#[test]
fn a_rejected_payload_leaves_the_canonical_domain_unchanged() {
    let name = Identifier::new("rejected-story-domain");
    let canonical = ValueDomain::new(name.clone(), "registered", None).into_canonical();
    let mut payload = encode_domain(&name, "rejected", &Value::Null);
    payload["unexpected"] = json!(1);

    let result = serde_json::from_value::<Canonical<ValueDomain>>(payload);

    assert!(result.is_err(), "the malformed payload decoded");
    let registered = ValueDomain::intern_registry()
        .get(&name)
        .expect("the domain stays registered");
    assert_eq!(registered, canonical);
    assert_eq!(registered.description(), "registered");
    assert_eq!(registered.parent(), None);
}
