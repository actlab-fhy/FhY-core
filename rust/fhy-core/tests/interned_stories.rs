//! User-story tests for `fhy_core::interned`, modeling a MOGA-style
//! `ResourceKind` tag with builtin canonical constants.
//!
//! Public API only. These tests share `ResourceKind`'s process-wide registry
//! and run in parallel, so none of them clears it, and a test that needs an
//! unseen key uses one named after the test.

use std::sync::LazyLock;

use fhy_core::interned::{Canonical, InternRegistry, Interned};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ResourceKind {
    name: String,
    description: String,
}

impl Interned for ResourceKind {
    type Key = String;

    fn intern_key(&self) -> &String {
        &self.name
    }

    fn intern_registry() -> &'static InternRegistry<ResourceKind> {
        static REGISTRY: InternRegistry<ResourceKind> =
            InternRegistry::with_defaults(create_builtin_kinds);
        &REGISTRY
    }
}

fn create_builtin_kinds() -> Vec<ResourceKind> {
    vec![
        ResourceKind {
            name: "memory".to_string(),
            description: "addressable storage".to_string(),
        },
        ResourceKind {
            name: "compute".to_string(),
            description: "an executing processor".to_string(),
        },
    ]
}

static MEMORY: LazyLock<Canonical<ResourceKind>> = LazyLock::new(|| {
    ResourceKind::intern_registry()
        .require("memory")
        .expect("builtin kind")
});

fn create_resource_kind(name: &str, description: &str) -> Canonical<ResourceKind> {
    ResourceKind::intern_registry()
        .intern(ResourceKind {
            name: name.to_string(),
            description: description.to_string(),
        })
        .into_canonical()
}

/// Test the builtin `MEMORY` constant is the canonical `ResourceKind`
/// registered under "memory".
#[test]
fn resource_kind_builtin_constant_is_the_canonical_instance() {
    let looked_up = ResourceKind::intern_registry()
        .get("memory")
        .expect("memory is a builtin kind");

    assert_eq!(*MEMORY, looked_up);
}

/// Test reconstructing a "memory" kind returns the builtin constant, keeping
/// its description rather than the reconstruction's payload.
#[test]
fn resource_kind_reconstruction_returns_the_builtin_constant() {
    let reconstructed = create_resource_kind("memory", "a different description");

    assert_eq!(reconstructed, *MEMORY);
    assert_eq!(reconstructed.description, MEMORY.description);
}

/// Test interning a brand-new kind makes it the canonical instance for its
/// key.
#[test]
fn resource_kind_new_kind_becomes_canonical() {
    let key = "resource_kind_new_kind_becomes_canonical";

    let created = create_resource_kind(key, "a scratch kind for this test");

    let looked_up = ResourceKind::intern_registry()
        .get(key)
        .expect("the new kind is registered");
    assert_eq!(created, looked_up);
    assert_eq!(created.description, "a scratch kind for this test");
}

/// Test a `ResourceKind` deserialized as a field inside a larger document is
/// the canonical instance.
#[test]
fn resource_kind_deserialized_inside_a_document_is_canonical() {
    #[derive(Debug, Serialize, Deserialize)]
    struct Allocation {
        kind: Canonical<ResourceKind>,
        bytes: u64,
    }

    let allocation = Allocation {
        kind: MEMORY.clone(),
        bytes: 4096,
    };
    let json = serde_json::to_string(&allocation).expect("allocation serializes");

    let restored: Allocation = serde_json::from_str(&json).expect("allocation deserializes");

    assert_eq!(restored.kind, *MEMORY);
    assert_eq!(restored.bytes, 4096);
}
