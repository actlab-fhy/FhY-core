//! User-story tests for `fhy_core::op_attribute` and `fhy_core::value_domain`.
//!
//! Public API only. These tests share the process-wide `OpAttribute` and
//! `ValueDomain` registries and run in parallel, so none of them clears a
//! registry, and each story creates its identifiers under a name hint unique
//! to that story.

use std::collections::HashSet;
use std::sync::LazyLock;

use fhy_core::identifier::Identifier;
use fhy_core::interned::Canonical;
use fhy_core::op_attribute::{ASSOCIATIVE, COMMUTATIVE, OpAttribute, PURE};
use fhy_core::value_domain::{ADDRESS_DOMAIN, DATA_DOMAIN, ValueDomain};
use serde::{Deserialize, Serialize};

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

/// Test tagging an op with shipped attributes plus a layer-registered one,
/// querying membership with `has_tag`, and showing a second `OpAttribute`
/// built with the same `Identifier` collapses into the existing set entry
/// instead of adding a new one.
#[test]
fn tagging_an_operation_with_shipped_and_layer_specific_attributes() {
    let idempotent = OpAttribute::new(
        IDEMPOTENT_NAME.clone(),
        "Applying the op twice changes nothing.",
    )
    .into_canonical();

    let op = StoryOp::create([COMMUTATIVE.clone(), PURE.clone(), idempotent.clone()]);

    assert_eq!(op.count_tags(), 3);
    assert!(op.has_tag(&COMMUTATIVE));
    assert!(op.has_tag(&PURE));
    assert!(op.has_tag(&idempotent));
    assert!(!op.has_tag(&ASSOCIATIVE));

    let same_name_again =
        OpAttribute::new(IDEMPOTENT_NAME.clone(), "a different description").into_canonical();
    let op_with_repeat = StoryOp::create([
        COMMUTATIVE.clone(),
        PURE.clone(),
        idempotent.clone(),
        same_name_again,
    ]);

    assert_eq!(op_with_repeat.count_tags(), 3);
}

/// Name of this story's middle-tier domain, a child of `DATA_DOMAIN`.
static TENSOR_NAME: LazyLock<Identifier> = LazyLock::new(|| Identifier::new("domain-story-tensor"));

/// Name of this story's leaf domain, a child of the tensor domain.
static TILE_NAME: LazyLock<Identifier> = LazyLock::new(|| Identifier::new("domain-story-tile"));

/// Name of this story's domain on an unrelated branch, a child of
/// `ADDRESS_DOMAIN`.
static TOKEN_NAME: LazyLock<Identifier> = LazyLock::new(|| Identifier::new("domain-story-token"));

/// Test a three-level domain hierarchy registered under `DATA_DOMAIN` relates
/// each level to its ancestors via `is_subdomain_of`, and relates none of
/// them to a domain on an unrelated branch.
#[test]
fn a_three_level_domain_hierarchy_relates_its_levels() {
    let tensor = ValueDomain::new(
        TENSOR_NAME.clone(),
        "A tensor of concrete data.",
        Some(DATA_DOMAIN.clone()),
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
        Some(ADDRESS_DOMAIN.clone()),
    )
    .into_canonical();

    assert!(tile.is_subdomain_of(&tile));
    assert!(tile.is_subdomain_of(&tensor));
    assert!(tile.is_subdomain_of(&DATA_DOMAIN));
    assert!(!tile.is_subdomain_of(&token));
    assert!(!DATA_DOMAIN.is_subdomain_of(&tile));
    assert!(!token.is_subdomain_of(&DATA_DOMAIN));
}

/// Name of the attribute this story persists.
static PERSISTED_ATTRIBUTE_NAME: LazyLock<Identifier> =
    LazyLock::new(|| Identifier::new("persistence-story-attribute"));

/// Name of the domain this story persists, a child of `ADDRESS_DOMAIN`.
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
        Some(ADDRESS_DOMAIN.clone()),
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
