//! Open, registry-backed classification of value kinds in a compiler IR.
//!
//! A compiler intermediate representation often needs to say what kind of
//! value an operation produces or consumes -- concrete data, an address, a
//! control token -- without a foundational crate committing to a closed set.
//! [`ValueDomain`] keeps that classification open: a layer registers the
//! domains it needs without changing this crate.
//!
//! [`DATA_DOMAIN`] and [`ADDRESS_DOMAIN`] are the two domains shipped here.
//! Import them rather than building a fresh domain with the same name hint.
//! Identifiers compare by id, and a second `Identifier::new("data")` is a
//! different key.
//!
//! Domains form a hierarchy through their parents, and
//! [`ValueDomain::is_subdomain_of`] walks that chain, so a layer can relate
//! its domains without the relationships living in this crate.
//!
//! A `description` is human-readable metadata. It takes no part in equality,
//! hashing or interning: the first domain registered under an identifier stays
//! canonical, and a later one is handed back to its caller in
//! [`InternOutcome::AlreadyCanonical`] instead of replacing it.

use std::hash::{Hash, Hasher};
use std::sync::LazyLock;

use serde::{Deserialize, Serialize};

use crate::identifier::{HasIdentifier, Identifier};
use crate::interned::{Canonical, InternOutcome, InternRegistry, Interned};

/// Open classification of the kind of value an IR operation handles.
///
/// Two domains are equal when they carry the same [`Identifier`] and the same
/// parent, whatever their descriptions say.
///
/// Decoding a domain canonicalizes it only through the handle, so deserialize
/// a [`Canonical<ValueDomain>`]. Deserializing a bare `ValueDomain` yields a
/// value that no registry knows about.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValueDomain {
    name: Identifier,
    description: String,
    // `serde` lets an `Option` field be missing and decode as `None`, but the
    // Python payload always carries `parent`. Naming a `deserialize_with`
    // turns off that special case, so a payload without the key is rejected.
    #[serde(deserialize_with = "Option::deserialize")]
    parent: Option<Canonical<ValueDomain>>,
}

impl ValueDomain {
    /// Build the domain named `name` and register it as the canonical one for
    /// that name.
    ///
    /// `parent` is the domain's super-domain, or `None` for a root domain.
    ///
    /// The outcome carries the canonical handle either way. When `name` is
    /// already taken the earlier domain stays canonical, and the one built
    /// here comes back as the outcome's `discarded` value, so a caller that
    /// cares about the dropped description or parent can see it.
    ///
    /// # Examples
    ///
    /// ```
    /// use fhy_core::identifier::Identifier;
    /// use fhy_core::value_domain::{ValueDomain, DATA_DOMAIN};
    ///
    /// let tile = ValueDomain::new(
    ///     Identifier::new("tile"),
    ///     "A tile of concrete data.",
    ///     Some(DATA_DOMAIN.clone()),
    /// )
    /// .into_canonical();
    ///
    /// assert!(tile.is_subdomain_of(&DATA_DOMAIN));
    /// assert!(!DATA_DOMAIN.is_subdomain_of(&tile));
    /// ```
    pub fn new(
        name: Identifier,
        description: impl Into<String>,
        parent: Option<Canonical<ValueDomain>>,
    ) -> InternOutcome<Self> {
        Self::intern_registry().intern(Self::create(name, description, parent))
    }

    /// Build the domain without registering it.
    fn create(
        name: Identifier,
        description: impl Into<String>,
        parent: Option<Canonical<ValueDomain>>,
    ) -> Self {
        Self {
            name,
            description: description.into(),
            parent,
        }
    }

    /// Return the domain's name.
    #[must_use]
    pub fn name(&self) -> &Identifier {
        &self.name
    }

    /// Return the domain's human-readable description.
    #[must_use]
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Return the domain's super-domain, or `None` for a root domain.
    #[must_use]
    pub fn parent(&self) -> Option<&Canonical<ValueDomain>> {
        self.parent.as_ref()
    }

    /// Return whether `other` is this domain or one of its ancestors.
    ///
    /// The walk follows parents, so the relation is reflexive and one-way: a
    /// domain is a subdomain of its ancestors and of itself, never of its
    /// descendants or of an unrelated domain. A parent is always registered
    /// before its children, so the chain cannot cycle.
    #[must_use]
    pub fn is_subdomain_of(&self, other: &ValueDomain) -> bool {
        let mut current = self;
        loop {
            if current == other {
                return true;
            }
            match current.parent.as_deref() {
                Some(parent) => current = parent,
                None => return false,
            }
        }
    }
}

impl HasIdentifier for ValueDomain {
    fn identifier(&self) -> &Identifier {
        &self.name
    }
}

impl Interned for ValueDomain {
    type Key = Identifier;

    fn intern_key(&self) -> &Identifier {
        &self.name
    }

    fn intern_registry() -> &'static InternRegistry<Self> {
        static REGISTRY: InternRegistry<ValueDomain> =
            InternRegistry::with_defaults(create_default_domains);
        &REGISTRY
    }
}

impl PartialEq for ValueDomain {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.parent == other.parent
    }
}

impl Eq for ValueDomain {}

impl Hash for ValueDomain {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.parent.hash(state);
    }
}

/// Name of the domain shipped as [`DATA_DOMAIN`].
static DATA_DOMAIN_NAME: LazyLock<Identifier> = LazyLock::new(|| Identifier::new_unscoped("data"));

/// Name of the domain shipped as [`ADDRESS_DOMAIN`].
static ADDRESS_DOMAIN_NAME: LazyLock<Identifier> =
    LazyLock::new(|| Identifier::new_unscoped("address"));

/// Build the domains this module ships, in registration order.
///
/// The registry calls this on its first use and again whenever it is cleared,
/// so the constants below stay canonical for the life of the process.
fn create_default_domains() -> Vec<ValueDomain> {
    vec![
        ValueDomain::create(
            DATA_DOMAIN_NAME.clone(),
            "Concrete data values flowing through the IR.",
            None,
        ),
        ValueDomain::create(
            ADDRESS_DOMAIN_NAME.clone(),
            "Index, offset, or address values used to access data.",
            None,
        ),
    ]
}

/// Return the canonical domain registered under a shipped default's name.
///
/// # Panics
///
/// Panics if `name` is not one of the names [`create_default_domains`] builds,
/// since the registry registers every default on its first use.
fn require_default(name: &Identifier) -> Canonical<ValueDomain> {
    ValueDomain::intern_registry()
        .require(name)
        .expect("the registry registers every default on its first use")
}

/// Concrete data values flowing through the IR.
pub static DATA_DOMAIN: LazyLock<Canonical<ValueDomain>> =
    LazyLock::new(|| require_default(&DATA_DOMAIN_NAME));

/// Index, offset, or address values used to access data.
pub static ADDRESS_DOMAIN: LazyLock<Canonical<ValueDomain>> =
    LazyLock::new(|| require_default(&ADDRESS_DOMAIN_NAME));

#[cfg(test)]
mod tests {
    use super::*;
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

    /// Intern a root domain under a fresh name and return its handle.
    fn intern_root(name_hint: &str) -> Canonical<ValueDomain> {
        ValueDomain::new(Identifier::new(name_hint), "root", None).into_canonical()
    }

    /// Intern a child of `parent` under a fresh name and return its handle.
    fn intern_child(name_hint: &str, parent: &Canonical<ValueDomain>) -> Canonical<ValueDomain> {
        ValueDomain::new(Identifier::new(name_hint), "child", Some(parent.clone())).into_canonical()
    }

    /// Intern `description` under `name`, expecting the key to be taken, and
    /// return the value that lost the race.
    fn intern_and_take_discarded(
        name: Identifier,
        description: &str,
        parent: Option<Canonical<ValueDomain>>,
    ) -> ValueDomain {
        match ValueDomain::new(name, description, parent) {
            InternOutcome::AlreadyCanonical { discarded, .. } => discarded,
            InternOutcome::Registered(_) => panic!("expected {description:?} to be discarded"),
        }
    }

    #[test]
    fn new_stores_the_name_description_and_absent_parent() {
        let _guard = hold_registry();
        let name = Identifier::new("stores-name-and-description");
        let domain = ValueDomain::new(name.clone(), "a domain", None).into_canonical();

        assert_eq!(domain.name(), &name);
        assert_eq!(domain.description(), "a domain");
        assert_eq!(domain.parent(), None);
    }

    #[test]
    fn new_stores_the_parent_it_is_given() {
        let _guard = hold_registry();
        let parent = intern_root("stored-parent");
        let child = intern_child("stores-parent", &parent);

        assert_eq!(child.parent(), Some(&parent));
    }

    #[test]
    fn has_identifier_returns_the_name() {
        let _guard = hold_registry();
        let name = Identifier::new("has-identifier");
        let domain = ValueDomain::new(name.clone(), "desc", None).into_canonical();

        assert_eq!(domain.identifier(), &name);
    }

    #[test]
    fn intern_key_is_the_name() {
        let _guard = hold_registry();
        let name = Identifier::new("intern-key");
        let domain = ValueDomain::new(name.clone(), "desc", None).into_canonical();

        assert_eq!(domain.intern_key(), &name);
    }

    #[test]
    fn new_keeps_the_first_domain_canonical_for_a_repeated_name() {
        let _guard = hold_registry();
        let name = Identifier::new("repeated-name");
        let first = ValueDomain::new(name.clone(), "first", None).into_canonical();

        let outcome = ValueDomain::new(name.clone(), "second", None);

        assert!(!outcome.is_registered());
        let InternOutcome::AlreadyCanonical {
            canonical,
            discarded,
        } = &outcome
        else {
            panic!("expected the second domain to lose the registration");
        };
        assert_eq!(canonical, &first);
        assert_eq!(discarded.description(), "second");
        assert_eq!(ValueDomain::intern_registry().get(&name), Some(first));
    }

    #[test]
    fn identifiers_sharing_a_name_hint_intern_separately() {
        let _guard = hold_registry();
        let first_name = Identifier::new("dup");
        let second_name = Identifier::new("dup");
        let first = ValueDomain::new(first_name.clone(), "a", None).into_canonical();
        let second = ValueDomain::new(second_name.clone(), "b", None).into_canonical();

        assert_ne!(first, second);
        assert_eq!(ValueDomain::intern_registry().get(&first_name), Some(first));
        assert_eq!(
            ValueDomain::intern_registry().get(&second_name),
            Some(second)
        );
    }

    #[test]
    fn domains_with_the_same_name_and_parent_are_equal() {
        let _guard = hold_registry();
        let name = Identifier::new("equality-ignores-description");
        let canonical = ValueDomain::new(name.clone(), "first description", None).into_canonical();

        let other = intern_and_take_discarded(name, "second description", None);

        assert_eq!(*canonical, other);
        assert_eq!(other, *canonical);
    }

    #[test]
    fn equal_domains_hash_equally() {
        let _guard = hold_registry();
        let name = Identifier::new("hash-ignores-description");
        let canonical = ValueDomain::new(name.clone(), "first description", None).into_canonical();

        let other = intern_and_take_discarded(name, "second description", None);

        assert_eq!(compute_hash(&*canonical), compute_hash(&other));
    }

    #[test]
    fn domains_with_different_names_are_unequal() {
        let _guard = hold_registry();
        let left = ValueDomain::new(Identifier::new("a"), "desc", None).into_canonical();
        let right = ValueDomain::new(Identifier::new("b"), "desc", None).into_canonical();

        assert_ne!(*left, *right);
    }

    #[test]
    fn domains_with_different_parents_are_unequal() {
        let _guard = hold_registry();
        let name = Identifier::new("parent-differs");
        let parented =
            ValueDomain::new(name.clone(), "desc", Some(DATA_DOMAIN.clone())).into_canonical();

        let orphan = intern_and_take_discarded(name, "desc", None);

        assert_ne!(*parented, orphan);
    }

    #[test]
    fn the_data_domain_is_registered_under_its_name() {
        let _guard = hold_registry();
        assert_eq!(
            ValueDomain::intern_registry().get(DATA_DOMAIN.name()),
            Some(DATA_DOMAIN.clone())
        );
    }

    #[test]
    fn the_address_domain_is_registered_under_its_name() {
        let _guard = hold_registry();
        assert_eq!(
            ValueDomain::intern_registry().get(ADDRESS_DOMAIN.name()),
            Some(ADDRESS_DOMAIN.clone())
        );
    }

    #[test]
    fn the_default_domains_are_distinct() {
        let _guard = hold_registry();
        assert_ne!(*DATA_DOMAIN, *ADDRESS_DOMAIN);
        assert_ne!(DATA_DOMAIN.name(), ADDRESS_DOMAIN.name());
    }

    #[test]
    fn the_default_domains_have_no_parent() {
        let _guard = hold_registry();
        assert_eq!(DATA_DOMAIN.parent(), None);
        assert_eq!(ADDRESS_DOMAIN.parent(), None);
    }

    #[test]
    fn the_default_domains_carry_non_empty_descriptions() {
        let _guard = hold_registry();
        for default in [&*DATA_DOMAIN, &*ADDRESS_DOMAIN] {
            assert!(!default.description().trim().is_empty());
        }
    }

    #[test]
    fn clearing_the_registry_keeps_the_default_domains_canonical() {
        let _guard = hold_registry_exclusively();
        let name = Identifier::new("dropped-by-clear");
        let dropped = ValueDomain::new(name.clone(), "dropped", None).into_canonical();
        assert_eq!(ValueDomain::intern_registry().get(&name), Some(dropped));

        ValueDomain::intern_registry().clear();

        assert_eq!(ValueDomain::intern_registry().get(&name), None);
        for default in [&DATA_DOMAIN, &ADDRESS_DOMAIN] {
            assert_eq!(
                ValueDomain::intern_registry().get(default.name()),
                Some((*default).clone())
            );
        }
    }

    #[test]
    fn a_domain_is_a_subdomain_of_itself() {
        let _guard = hold_registry();
        let child = intern_child("subdomain-of-itself", &DATA_DOMAIN);

        assert!(child.is_subdomain_of(&child));
    }

    #[test]
    fn a_domain_is_a_subdomain_of_its_parent() {
        let _guard = hold_registry();
        let child = intern_child("subdomain-of-parent", &DATA_DOMAIN);

        assert!(child.is_subdomain_of(&DATA_DOMAIN));
    }

    #[test]
    fn a_domain_is_a_subdomain_of_a_distant_ancestor() {
        let _guard = hold_registry();
        let middle = intern_child("subdomain-middle", &DATA_DOMAIN);
        let leaf = intern_child("subdomain-leaf", &middle);

        assert!(leaf.is_subdomain_of(&DATA_DOMAIN));
    }

    #[test]
    fn a_domain_is_not_a_subdomain_of_a_sibling() {
        let _guard = hold_registry();
        let child = intern_child("subdomain-sibling", &DATA_DOMAIN);

        assert!(!child.is_subdomain_of(&ADDRESS_DOMAIN));
    }

    #[test]
    fn a_parent_is_not_a_subdomain_of_its_child() {
        let _guard = hold_registry();
        let child = intern_child("subdomain-one-way", &DATA_DOMAIN);

        assert!(!DATA_DOMAIN.is_subdomain_of(&child));
    }

    #[test]
    fn a_root_domain_encodes_with_a_null_parent() {
        let _guard = hold_registry();
        let id = reserve_pinned_id("encode-anchor");
        let name = Identifier::deserialize(id, "encoded".to_string());
        let domain = ValueDomain::new(name, "a description", None).into_canonical();

        let json = serde_json::to_string(&*domain).unwrap();

        assert_eq!(
            json,
            format!(
                "{{\"name\":{{\"id\":{id},\"name_hint\":\"encoded\"}},\
                 \"description\":\"a description\",\"parent\":null}}"
            )
        );
    }

    #[test]
    fn a_child_domain_encodes_its_parent_inline() {
        let _guard = hold_registry();
        let parent_id = reserve_pinned_id("encode-parent-anchor");
        let parent_name = Identifier::deserialize(parent_id, "parent".to_string());
        let parent = ValueDomain::new(parent_name, "the parent", None).into_canonical();
        let child_id = reserve_pinned_id("encode-child-anchor");
        let child_name = Identifier::deserialize(child_id, "child".to_string());
        let child = ValueDomain::new(child_name, "the child", Some(parent)).into_canonical();

        let json = serde_json::to_string(&*child).unwrap();

        assert_eq!(
            json,
            format!(
                "{{\"name\":{{\"id\":{child_id},\"name_hint\":\"child\"}},\
                 \"description\":\"the child\",\
                 \"parent\":{{\"name\":{{\"id\":{parent_id},\"name_hint\":\"parent\"}},\
                 \"description\":\"the parent\",\"parent\":null}}}}"
            )
        );
    }

    #[test]
    fn a_domain_round_trips_through_json() {
        let _guard = hold_registry();
        let domain = intern_child("round-trip", &DATA_DOMAIN);

        let json = serde_json::to_string(&*domain).unwrap();
        let restored: Canonical<ValueDomain> = serde_json::from_str(&json).unwrap();

        assert_eq!(restored, domain);
    }

    #[test]
    fn decoding_a_registered_name_returns_the_canonical_domain() {
        let _guard = hold_registry();
        let json = serde_json::to_string(&*DATA_DOMAIN).unwrap();

        let restored: Canonical<ValueDomain> = serde_json::from_str(&json).unwrap();

        assert_eq!(restored, *DATA_DOMAIN);
    }

    #[test]
    fn decoding_an_unregistered_name_registers_the_decoded_domain() {
        let _guard = hold_registry();
        let id = reserve_pinned_id("unregistered-decode-anchor");
        let json = format!(
            "{{\"name\":{{\"id\":{id},\"name_hint\":\"never-registered\"}},\
             \"description\":\"fresh from decode\",\"parent\":null}}"
        );

        let restored: Canonical<ValueDomain> = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.description(), "fresh from decode");
        assert_eq!(restored.parent(), None);
        assert_eq!(
            ValueDomain::intern_registry().get(restored.name()),
            Some(restored)
        );
    }

    #[test]
    fn decoding_a_nested_parent_canonicalizes_it() {
        let _guard = hold_registry();
        let child = intern_child("nested-parent-child", &DATA_DOMAIN);
        let json = serde_json::to_string(&*child).unwrap();

        let restored: Canonical<ValueDomain> = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.parent(), Some(&*DATA_DOMAIN));
    }

    #[test]
    fn decoding_a_divergent_description_keeps_the_canonical_one() {
        let _guard = hold_registry();
        let name = Identifier::new("divergent-description");
        let canonical =
            ValueDomain::new(name.clone(), "original description", None).into_canonical();
        let json = format!(
            "{{\"name\":{{\"id\":{},\"name_hint\":\"{}\"}},\
             \"description\":\"divergent description\",\"parent\":null}}",
            name.id(),
            name.name_hint()
        );

        let restored: Canonical<ValueDomain> = serde_json::from_str(&json).unwrap();

        assert_eq!(restored, canonical);
        assert_eq!(restored.description(), "original description");
    }

    #[test]
    fn new_reports_a_matching_duplicate_as_already_canonical() {
        let _guard = hold_registry();
        let name = Identifier::new("matching-description");
        let canonical = ValueDomain::new(name.clone(), "matching", None).into_canonical();

        let discarded = intern_and_take_discarded(name, "matching", None);

        assert_eq!(discarded.description(), canonical.description());
        assert_eq!(discarded, *canonical);
    }

    #[test]
    fn decoding_a_payload_with_an_unknown_field_is_rejected() {
        let _guard = hold_registry();
        let id = reserve_pinned_id("unknown-field-anchor");
        let json = format!(
            "{{\"name\":{{\"id\":{id},\"name_hint\":\"extra\"}},\
             \"description\":\"desc\",\"parent\":null,\"surprise\":1}}"
        );

        let error = serde_json::from_str::<Canonical<ValueDomain>>(&json).unwrap_err();

        assert!(error.to_string().contains("surprise"), "{error}");
    }

    #[test]
    fn decoding_a_payload_missing_the_parent_is_rejected() {
        let _guard = hold_registry();
        let id = reserve_pinned_id("missing-parent-anchor");
        let json = format!(
            "{{\"name\":{{\"id\":{id},\"name_hint\":\"partial\"}},\"description\":\"desc\"}}"
        );

        let error = serde_json::from_str::<Canonical<ValueDomain>>(&json).unwrap_err();

        assert!(error.to_string().contains("parent"), "{error}");
    }

    #[test]
    fn debug_mentions_the_name_hint_and_description() {
        let _guard = hold_registry();
        let domain =
            ValueDomain::new(Identifier::new("debug-domain"), "debug desc", None).into_canonical();

        let rendered = format!("{domain:?}");

        assert!(rendered.contains("debug-domain"), "{rendered}");
        assert!(rendered.contains("debug desc"), "{rendered}");
    }
}
