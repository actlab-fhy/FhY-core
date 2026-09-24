//! Open, registry-backed classification of value kinds in a compiler IR.
//!
//! A compiler intermediate representation often needs to say what kind of
//! value an operation produces or consumes -- concrete data, an address, a
//! control token -- without a foundational crate committing to a closed set.
//! [`ValueDomain`] keeps that classification open: a layer registers the
//! domains it needs without changing this crate.
//!
//! [`ValueDomain::data`] and [`ValueDomain::address`] return the two
//! domains shipped here. Call them rather than building a fresh domain with the same
//! name hint. Identifiers compare by id, and a second `Identifier::new("data")`
//! is a different key.
//!
//! Domains form a hierarchy through their parents, and
//! [`ValueDomain::is_subdomain_of`] walks that chain, so a layer can relate
//! its domains without the relationships living in this crate.
//!
//! A `description` is human-readable metadata. It takes no part in equality,
//! hashing or interning: the first domain registered under an identifier stays
//! canonical, and a later registration of that name returns it, dropping its
//! own description. A name has one parent, so registering it again under a
//! parent of another name is a [`ValueDomainConflict`].

use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::LazyLock;

use serde::{Deserialize, Deserializer, Serialize, de};

use crate::decode::{self, Decode, DeferredPayload};
use crate::identifier::{HasIdentifier, Identifier, IdentifierWire, reserved};
use crate::interned::{
    Canonical, InternOutcome, InternRegistry, Interned, intern_decoded, require_default,
};

/// Open classification of the kind of value an IR operation handles.
///
/// Two domains are equal when they carry the same [`Identifier`] and equal
/// parents, whatever their descriptions say. Parents compare by value up the
/// whole chain, not by handle, so a chain rebuilt after the registry is
/// cleared equals the chain built before it.
///
/// Decoding a domain canonicalizes it only through the handle, so deserialize
/// a [`Canonical<ValueDomain>`]. Deserializing a bare `ValueDomain` yields a
/// value that no registry knows about. Decoding a handle for a name that is
/// already canonical fails when the payload names a different parent.
#[derive(Debug, Serialize)]
pub struct ValueDomain {
    name: Identifier,
    description: String,
    parent: Option<Canonical<ValueDomain>>,
}

impl ValueDomain {
    /// Register the root domain named `name`, unless a domain of that name
    /// is registered already, and return the canonical handle for that name.
    ///
    /// The first registration wins: when `name` is already registered as a
    /// root, the registered domain is returned and `description` is dropped.
    ///
    /// # Errors
    ///
    /// Returns [`ValueDomainConflict`] if `name` is registered with a
    /// parent.
    ///
    /// # Examples
    ///
    /// ```
    /// use fhy_core::identifier::Identifier;
    /// use fhy_core::value_domain::ValueDomain;
    ///
    /// let token = ValueDomain::register_root(Identifier::new("token"), "A control token.")
    ///     .expect("the name is fresh");
    ///
    /// assert_eq!(token.parent(), None);
    /// assert!(!token.is_subdomain_of(ValueDomain::data()));
    /// ```
    pub fn register_root(
        name: Identifier,
        description: impl Into<String>,
    ) -> Result<Canonical<ValueDomain>, ValueDomainConflict> {
        Self::register(Self::create(name, description, None))
    }

    /// Register the domain named `name` as a child of `parent`, unless a
    /// domain of that name is registered already, and return the canonical
    /// handle for that name.
    ///
    /// The first registration wins: when `name` is already registered under
    /// a parent of the same name, the registered domain is returned and
    /// `description` is dropped.
    ///
    /// # Errors
    ///
    /// Returns [`ValueDomainConflict`] if `name` is registered as a root or
    /// under a parent of another name.
    ///
    /// # Examples
    ///
    /// ```
    /// use fhy_core::identifier::Identifier;
    /// use fhy_core::value_domain::ValueDomain;
    ///
    /// let tile = ValueDomain::register_child(
    ///     Identifier::new("tile"),
    ///     "A tile of concrete data.",
    ///     ValueDomain::data(),
    /// )
    /// .expect("the name is fresh");
    ///
    /// assert!(tile.is_subdomain_of(ValueDomain::data()));
    /// assert!(!ValueDomain::data().is_subdomain_of(&tile));
    /// ```
    pub fn register_child(
        name: Identifier,
        description: impl Into<String>,
        parent: &Canonical<ValueDomain>,
    ) -> Result<Canonical<ValueDomain>, ValueDomainConflict> {
        Self::register(Self::create(name, description, Some(parent.clone())))
    }

    /// Register `domain` unless its name is registered already, and return
    /// the canonical handle for the name.
    ///
    /// # Errors
    ///
    /// Returns [`ValueDomainConflict`] if the name is registered under a
    /// parent of another name, or as a root when `domain` has a parent, or
    /// the other way around.
    fn register(domain: ValueDomain) -> Result<Canonical<ValueDomain>, ValueDomainConflict> {
        match Self::intern_registry().intern(domain) {
            InternOutcome::Registered(canonical) => Ok(canonical),
            InternOutcome::AlreadyCanonical {
                canonical,
                discarded,
            } => {
                let registered_parent = canonical.parent().map(|parent| parent.name().clone());
                let requested_parent = discarded.parent().map(|parent| parent.name().clone());
                if registered_parent != requested_parent {
                    return Err(ValueDomainConflict {
                        name: discarded.name,
                        registered_parent,
                        requested_parent,
                    });
                }
                // The first registration wins, so a different description
                // given here is dropped.
                // TODO: warn here once the log dependency is added
                Ok(canonical)
            }
        }
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
    /// descendants or of an unrelated domain. A domain's parent exists before
    /// the domain is built and never changes, so the chain cannot cycle.
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

/// A value domain's name is already registered under another parent.
///
/// Registration keeps one parent per name, so registering a known name as a
/// root, as a child of a parent of another name, or registering a known
/// root as a child, is refused.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ValueDomainConflict {
    name: Identifier,
    registered_parent: Option<Identifier>,
    requested_parent: Option<Identifier>,
}

impl ValueDomainConflict {
    /// Return the name of the domain that could not be registered.
    #[must_use]
    pub fn name(&self) -> &Identifier {
        &self.name
    }

    /// Return the name of the registered domain's parent, or `None` when it
    /// is a root.
    #[must_use]
    pub fn registered_parent(&self) -> Option<&Identifier> {
        self.registered_parent.as_ref()
    }

    /// Return the name of the parent the registration asked for, or `None`
    /// when it asked for a root.
    #[must_use]
    pub fn requested_parent(&self) -> Option<&Identifier> {
        self.requested_parent.as_ref()
    }
}

/// Render the conflict on one line, for example ``value domain `tile` is
/// already registered with parent `data`, not `address` ``.
impl fmt::Display for ValueDomainConflict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "value domain `{}` is already registered with ",
            self.name
        )?;
        match &self.registered_parent {
            Some(parent) => write!(f, "parent `{parent}`")?,
            None => f.write_str("no parent")?,
        }
        match &self.requested_parent {
            Some(parent) => write!(f, ", not `{parent}`"),
            None => f.write_str(", not as a root"),
        }
    }
}

impl std::error::Error for ValueDomainConflict {}

impl Decode for ValueDomain {
    type Payload = ValueDomainPayload;

    /// Build the domain this level describes, leaving it unregistered.
    ///
    /// Restore the name, then check, build and intern the parent level.
    /// Python likewise restores the name before decoding the parent.
    ///
    /// # Errors
    ///
    /// Returns an error if a nested level is malformed or conflicts with the
    /// canonical instance for its name.
    fn build_from_payload<E: de::Error>(payload: Self::Payload) -> Result<Self, E> {
        let name = Identifier::try_from(payload.name).map_err(E::custom)?;
        let parent = match payload.parent {
            None => None,
            Some(parent) => Some(intern_decoded(parent.decode("parent")?)?),
        };
        Ok(ValueDomain::create(name, payload.description, parent))
    }
}

/// Decoding checks one level of the payload at a time, outermost first. A
/// level whose fields are all present, known and well typed restores its name
/// before the level nested in its `parent` is checked, and each parent is
/// interned, and so registered, before the domain that holds it is built.
/// A payload rejected for its structure therefore registers nothing beyond
/// the shipped defaults, though the names of the levels above the defect stay
/// restored. A payload rejected because a level conflicts with its canonical
/// instance leaves the fresh parents below that level registered.
///
/// The nested levels are read before they are decoded, which needs a
/// self-describing format such as JSON.
impl<'de> Deserialize<'de> for ValueDomain {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        decode::deserialize_via_payload(deserializer)
    }
}

/// One level of a value-domain payload, checked but not yet built.
///
/// Decoding a level touches neither the id counter nor the registry: its name
/// is held unrestored and its parent is held unread.
#[derive(Deserialize)]
#[serde(rename = "ValueDomain", deny_unknown_fields)]
pub(crate) struct ValueDomainPayload {
    name: IdentifierWire,
    description: String,
    // `serde` lets an `Option` field be missing and decode as `None`, but the
    // Python payload always carries `parent`. Naming a `deserialize_with`
    // turns off that special case, so a payload without the key is rejected.
    #[serde(deserialize_with = "Option::deserialize")]
    parent: Option<DeferredPayload<ValueDomain>>,
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
        self.name == other.name && self.parent.as_deref() == other.parent.as_deref()
    }
}

impl Eq for ValueDomain {}

impl Hash for ValueDomain {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.parent.as_deref().hash(state);
    }
}

/// Name of the domain returned by [`ValueDomain::data`].
static DATA_DOMAIN_NAME: LazyLock<Identifier> =
    LazyLock::new(|| Identifier::reserved(reserved::DATA_DOMAIN));

/// Name of the domain returned by [`ValueDomain::address`].
static ADDRESS_DOMAIN_NAME: LazyLock<Identifier> =
    LazyLock::new(|| Identifier::reserved(reserved::ADDRESS_DOMAIN));

/// Build the domains this module ships, in registration order.
///
/// The registry calls this once, on its first use, and keeps the instances it
/// builds. A clear registers those same instances again rather than building
/// new ones, so the shipped constants stay canonical for the life of the
/// process.
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

static DATA_DOMAIN: LazyLock<Canonical<ValueDomain>> =
    LazyLock::new(|| require_default(&*DATA_DOMAIN_NAME));

static ADDRESS_DOMAIN: LazyLock<Canonical<ValueDomain>> =
    LazyLock::new(|| require_default(&*ADDRESS_DOMAIN_NAME));

impl ValueDomain {
    /// Return the domain for concrete data values flowing through the IR.
    #[must_use]
    pub fn data() -> &'static Canonical<ValueDomain> {
        &DATA_DOMAIN
    }

    /// Return the domain for index, offset, or address values used to
    /// access data.
    #[must_use]
    pub fn address() -> &'static Canonical<ValueDomain> {
        &ADDRESS_DOMAIN
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use proptest::prelude::*;
    use rstest::rstest;

    use crate::test_support::{
        compute_hash, has_counter_passed, hold_id_counter, reserve_far_ahead_ids, reserve_pinned_id,
    };

    /// Intern a root domain under a fresh name and return its handle.
    fn intern_root(name_hint: &str) -> Canonical<ValueDomain> {
        ValueDomain::register_root(Identifier::new(name_hint), "root")
            .expect("the domain registers")
    }

    /// Intern a child of `parent` under a fresh name and return its handle.
    fn intern_child(name_hint: &str, parent: &Canonical<ValueDomain>) -> Canonical<ValueDomain> {
        ValueDomain::register_child(Identifier::new(name_hint), "child", parent)
            .expect("the domain registers")
    }

    #[test]
    fn new_stores_the_name_description_and_absent_parent() {
        let name = Identifier::new("stores-name-and-description");
        let domain =
            ValueDomain::register_root(name.clone(), "a domain").expect("the domain registers");

        assert_eq!(domain.name(), &name);
        assert_eq!(domain.description(), "a domain");
        assert_eq!(domain.parent(), None);
    }

    #[test]
    fn new_stores_the_parent_it_is_given() {
        let parent = intern_root("stored-parent");
        let child = intern_child("stores-parent", &parent);

        assert_eq!(child.parent(), Some(&parent));
    }

    #[test]
    fn has_identifier_returns_the_name() {
        let name = Identifier::new("has-identifier");
        let domain =
            ValueDomain::register_root(name.clone(), "desc").expect("the domain registers");

        assert_eq!(domain.identifier(), &name);
    }

    #[test]
    fn intern_key_is_the_name() {
        let name = Identifier::new("intern-key");
        let domain =
            ValueDomain::register_root(name.clone(), "desc").expect("the domain registers");

        assert_eq!(domain.intern_key(), &name);
    }

    #[test]
    fn register_root_of_a_known_root_returns_the_first_domain() {
        let name = Identifier::new("repeated-name");
        let first = ValueDomain::register_root(name.clone(), "first").expect("the name is fresh");

        let second = ValueDomain::register_root(name.clone(), "second")
            .expect("the name is registered as a root");

        assert_eq!(second, first);
        assert_eq!(second.description(), "first");
        assert_eq!(ValueDomain::intern_registry().get(&name), Some(first));
    }

    #[test]
    fn register_child_of_a_known_child_under_the_same_parent_returns_the_first_domain() {
        let name = Identifier::new("repeated-child");
        let first = ValueDomain::register_child(name.clone(), "first", ValueDomain::data())
            .expect("the name is fresh");

        let second = ValueDomain::register_child(name, "second", ValueDomain::data())
            .expect("the name is registered under the same parent");

        assert_eq!(second, first);
        assert_eq!(second.description(), "first");
    }

    #[test]
    fn registering_a_known_name_under_another_parent_is_a_conflict() {
        let name = Identifier::new("conflicting-child");
        let canonical = ValueDomain::register_child(name.clone(), "desc", ValueDomain::data())
            .expect("the name is fresh");

        let conflict = ValueDomain::register_child(name.clone(), "desc", ValueDomain::address())
            .expect_err("the name is registered under another parent");

        assert_eq!(conflict.name(), &name);
        assert_eq!(
            conflict.registered_parent(),
            Some(ValueDomain::data().name())
        );
        assert_eq!(
            conflict.requested_parent(),
            Some(ValueDomain::address().name())
        );
        assert_eq!(ValueDomain::intern_registry().get(&name), Some(canonical));
    }

    #[test]
    fn registering_a_known_child_as_a_root_is_a_conflict() {
        let name = Identifier::new("child-as-root");
        let _canonical = ValueDomain::register_child(name.clone(), "desc", ValueDomain::data())
            .expect("the name is fresh");

        let conflict = ValueDomain::register_root(name, "desc").expect_err("the name has a parent");

        assert_eq!(
            conflict.registered_parent(),
            Some(ValueDomain::data().name())
        );
        assert_eq!(conflict.requested_parent(), None);
    }

    #[test]
    fn registering_a_known_root_as_a_child_is_a_conflict() {
        let name = Identifier::new("root-as-child");
        let _canonical =
            ValueDomain::register_root(name.clone(), "desc").expect("the name is fresh");

        let conflict = ValueDomain::register_child(name, "desc", ValueDomain::data())
            .expect_err("the name is a root");

        assert_eq!(conflict.registered_parent(), None);
        assert_eq!(
            conflict.requested_parent(),
            Some(ValueDomain::data().name())
        );
    }

    /// Test each form of a conflict displays on one line, naming the domain
    /// and both parents.
    #[rstest]
    #[case::two_parents(
        Some("data"),
        Some("address"),
        "value domain `tile` is already registered with parent `data`, not `address`"
    )]
    #[case::registered_root(
        None,
        Some("data"),
        "value domain `tile` is already registered with no parent, not `data`"
    )]
    #[case::requested_root(
        Some("data"),
        None,
        "value domain `tile` is already registered with parent `data`, not as a root"
    )]
    fn conflict_error_names_the_domain_and_both_parents(
        #[case] registered_parent: Option<&str>,
        #[case] requested_parent: Option<&str>,
        #[case] expected: &str,
    ) {
        let conflict = ValueDomainConflict {
            name: Identifier::new("tile"),
            registered_parent: registered_parent.map(Identifier::new),
            requested_parent: requested_parent.map(Identifier::new),
        };

        assert_eq!(conflict.to_string(), expected);
    }

    #[test]
    fn conflict_error_is_a_std_error_without_a_source() {
        let conflict = ValueDomainConflict {
            name: Identifier::new("tile"),
            registered_parent: None,
            requested_parent: Some(Identifier::new("data")),
        };
        let error: &dyn std::error::Error = &conflict;

        assert!(error.source().is_none());
    }

    #[test]
    fn identifiers_sharing_a_name_hint_intern_separately() {
        let first_name = Identifier::new("dup");
        let second_name = Identifier::new("dup");
        let first =
            ValueDomain::register_root(first_name.clone(), "a").expect("the domain registers");
        let second =
            ValueDomain::register_root(second_name.clone(), "b").expect("the domain registers");

        assert_ne!(first, second);
        assert_eq!(ValueDomain::intern_registry().get(&first_name), Some(first));
        assert_eq!(
            ValueDomain::intern_registry().get(&second_name),
            Some(second)
        );
    }

    #[test]
    fn domains_with_the_same_name_and_parent_are_equal() {
        let name = Identifier::new("equality-ignores-description");
        let first = ValueDomain::create(name.clone(), "first description", None);
        let second = ValueDomain::create(name, "second description", None);

        assert_eq!(first, second);
        assert_eq!(second, first);
    }

    #[test]
    fn equal_domains_hash_equally() {
        let name = Identifier::new("hash-ignores-description");
        let first = ValueDomain::create(name.clone(), "first description", None);
        let second = ValueDomain::create(name, "second description", None);

        assert_eq!(compute_hash(&first), compute_hash(&second));
    }

    #[test]
    fn domains_with_different_names_are_unequal() {
        let left =
            ValueDomain::register_root(Identifier::new("a"), "desc").expect("the domain registers");
        let right =
            ValueDomain::register_root(Identifier::new("b"), "desc").expect("the domain registers");

        assert_ne!(*left, *right);
    }

    #[test]
    fn domains_with_different_parents_are_unequal() {
        let name = Identifier::new("parent-differs");
        let parented = ValueDomain::create(name.clone(), "desc", Some(ValueDomain::data().clone()));
        let orphan = ValueDomain::create(name, "desc", None);

        assert_ne!(parented, orphan);
    }

    /// Test each shipped default domain is the canonical entry for its name.
    #[rstest]
    #[case::data(ValueDomain::data)]
    #[case::address(ValueDomain::address)]
    fn a_default_domain_is_registered_under_its_name(
        #[case] get_default: fn() -> &'static Canonical<ValueDomain>,
    ) {
        let default = get_default();

        assert_eq!(
            ValueDomain::intern_registry().get(default.name()),
            Some(default.clone())
        );
    }

    /// Test each shipped domain holds its fixed reserved id and name hint.
    #[rstest]
    #[case::data(ValueDomain::data, 32, "data")]
    #[case::address(ValueDomain::address, 33, "address")]
    fn a_shipped_domain_holds_its_reserved_id(
        #[case] get_default: fn() -> &'static Canonical<ValueDomain>,
        #[case] id: u64,
        #[case] name_hint: &str,
    ) {
        let name = get_default().name();

        assert_eq!((name.id(), name.name_hint()), (id, name_hint));
    }

    #[test]
    fn the_default_domains_are_distinct() {
        assert_ne!(*ValueDomain::data(), *ValueDomain::address());
        assert_ne!(ValueDomain::data().name(), ValueDomain::address().name());
    }

    /// Test each shipped default domain is a root domain.
    #[rstest]
    #[case::data(ValueDomain::data)]
    #[case::address(ValueDomain::address)]
    fn a_default_domain_has_no_parent(
        #[case] get_default: fn() -> &'static Canonical<ValueDomain>,
    ) {
        assert_eq!(get_default().parent(), None);
    }

    /// Test each shipped default domain carries a non-empty description.
    #[rstest]
    #[case::data(ValueDomain::data)]
    #[case::address(ValueDomain::address)]
    fn a_default_domain_carries_a_non_empty_description(
        #[case] get_default: fn() -> &'static Canonical<ValueDomain>,
    ) {
        let description = get_default().description();

        assert!(!description.trim().is_empty(), "{description:?}");
    }

    #[test]
    fn a_domain_is_a_subdomain_of_itself() {
        let child = intern_child("subdomain-of-itself", ValueDomain::data());

        assert!(child.is_subdomain_of(&child));
    }

    #[test]
    fn a_domain_is_a_subdomain_of_its_parent() {
        let child = intern_child("subdomain-of-parent", ValueDomain::data());

        assert!(child.is_subdomain_of(ValueDomain::data()));
    }

    #[test]
    fn a_domain_is_a_subdomain_of_a_distant_ancestor() {
        let middle = intern_child("subdomain-middle", ValueDomain::data());
        let leaf = intern_child("subdomain-leaf", &middle);

        assert!(leaf.is_subdomain_of(ValueDomain::data()));
    }

    #[test]
    fn a_domain_is_not_a_subdomain_of_a_sibling() {
        let child = intern_child("subdomain-sibling", ValueDomain::data());

        assert!(!child.is_subdomain_of(ValueDomain::address()));
    }

    #[test]
    fn a_parent_is_not_a_subdomain_of_its_child() {
        let child = intern_child("subdomain-one-way", ValueDomain::data());

        assert!(!ValueDomain::data().is_subdomain_of(&child));
    }

    /// Largest number of domains one generated hierarchy holds.
    const MAXIMUM_HIERARCHY_SIZE: usize = 8;

    /// Where a generated domain hangs: at the top, under a shipped default, or
    /// under a domain generated before it. The first domain has none before
    /// it, so `Earlier` makes it a root.
    #[derive(Debug, Clone, Copy)]
    enum ParentChoice {
        Root,
        Data,
        Address,
        Earlier(prop::sample::Index),
    }

    /// A domain of a generated hierarchy as the reference model sees it.
    #[derive(Debug, Clone, Copy, PartialEq)]
    enum HierarchyNode {
        Data,
        Address,
        Built(usize),
    }

    /// Build a strategy for the parent choices of a hierarchy, favoring
    /// domains nested under earlier ones so chains grow deep.
    fn build_hierarchy_strategy() -> impl Strategy<Value = Vec<ParentChoice>> {
        let choice = prop_oneof![
            1 => Just(ParentChoice::Root),
            1 => Just(ParentChoice::Data),
            1 => Just(ParentChoice::Address),
            3 => any::<prop::sample::Index>().prop_map(ParentChoice::Earlier),
        ];
        prop::collection::vec(choice, 1..=MAXIMUM_HIERARCHY_SIZE)
    }

    /// Return the domain `node` stands for.
    fn resolve_node(
        node: HierarchyNode,
        domains: &[Canonical<ValueDomain>],
    ) -> &Canonical<ValueDomain> {
        match node {
            HierarchyNode::Data => ValueDomain::data(),
            HierarchyNode::Address => ValueDomain::address(),
            HierarchyNode::Built(index) => &domains[index],
        }
    }

    /// Intern one domain under a fresh name per choice, and return the
    /// domains with the parent node of each.
    fn intern_hierarchy(
        choices: &[ParentChoice],
    ) -> (Vec<Canonical<ValueDomain>>, Vec<Option<HierarchyNode>>) {
        let mut domains = Vec::with_capacity(choices.len());
        let mut parents = Vec::with_capacity(choices.len());
        for (index, choice) in choices.iter().enumerate() {
            let parent_node = match *choice {
                ParentChoice::Root => None,
                ParentChoice::Data => Some(HierarchyNode::Data),
                ParentChoice::Address => Some(HierarchyNode::Address),
                ParentChoice::Earlier(_) if index == 0 => None,
                ParentChoice::Earlier(earlier) => Some(HierarchyNode::Built(earlier.index(index))),
            };
            let parent = parent_node.map(|node| resolve_node(node, &domains).clone());
            let name = Identifier::new(&format!("hierarchy-{index}"));
            let domain = match parent {
                None => ValueDomain::register_root(name, "generated"),
                Some(parent) => ValueDomain::register_child(name, "generated", &parent),
            };
            domains.push(domain.expect("the name is fresh"));
            parents.push(parent_node);
        }
        (domains, parents)
    }

    /// Return `node` and every node above it, following `parents`.
    fn list_model_ancestors(
        node: HierarchyNode,
        parents: &[Option<HierarchyNode>],
    ) -> Vec<HierarchyNode> {
        let mut ancestors = vec![node];
        let mut current = node;
        while let HierarchyNode::Built(index) = current {
            let Some(parent) = parents[index] else {
                break;
            };
            ancestors.push(parent);
            current = parent;
        }
        ancestors
    }

    proptest! {
        /// Test `is_subdomain_of` holds exactly when the other domain is the
        /// domain itself or one of its ancestors, for every pair of domains in
        /// any hierarchy built under the shipped defaults.
        #[test]
        fn is_subdomain_of_matches_the_ancestor_relation_for_any_hierarchy(
            choices in build_hierarchy_strategy(),
        ) {
            let (domains, parents) = intern_hierarchy(&choices);
            let nodes: Vec<HierarchyNode> = [HierarchyNode::Data, HierarchyNode::Address]
                .into_iter()
                .chain((0..domains.len()).map(HierarchyNode::Built))
                .collect();

            for &subject in &nodes {
                let ancestors = list_model_ancestors(subject, &parents);
                for &other in &nodes {
                    prop_assert_eq!(
                        resolve_node(subject, &domains)
                            .is_subdomain_of(resolve_node(other, &domains)),
                        ancestors.contains(&other),
                        "{:?} is_subdomain_of {:?}",
                        subject,
                        other
                    );
                }
            }
        }

        /// Test every domain of any hierarchy decodes from its JSON back to
        /// its own canonical handle.
        #[test]
        fn a_domain_in_any_hierarchy_round_trips_through_json(
            choices in build_hierarchy_strategy(),
        ) {
            let (domains, _parents) = intern_hierarchy(&choices);

            for domain in &domains {
                let json = serde_json::to_string(&**domain).unwrap();
                let restored: Canonical<ValueDomain> = serde_json::from_str(&json).unwrap();

                prop_assert_eq!(&restored, domain);
            }
        }
    }

    #[test]
    fn a_root_domain_encodes_with_a_null_parent() {
        let id = reserve_pinned_id("encode-anchor");
        let name = Identifier::try_restore(id, "encoded").expect("the id is below the cap");
        let domain =
            ValueDomain::register_root(name, "a description").expect("the domain registers");

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
        let parent_id = reserve_pinned_id("encode-parent-anchor");
        let parent_name =
            Identifier::try_restore(parent_id, "parent").expect("the id is below the cap");
        let parent =
            ValueDomain::register_root(parent_name, "the parent").expect("the domain registers");
        let child_id = reserve_pinned_id("encode-child-anchor");
        let child_name =
            Identifier::try_restore(child_id, "child").expect("the id is below the cap");
        let child = ValueDomain::register_child(child_name, "the child", &parent)
            .expect("the domain registers");

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
        let domain = intern_child("round-trip", ValueDomain::data());

        let json = serde_json::to_string(&*domain).unwrap();
        let restored: Canonical<ValueDomain> = serde_json::from_str(&json).unwrap();

        assert_eq!(restored, domain);
    }

    #[test]
    fn decoding_a_registered_name_returns_the_canonical_domain() {
        let json = serde_json::to_string(ValueDomain::data()).unwrap();

        let restored: Canonical<ValueDomain> = serde_json::from_str(&json).unwrap();

        assert_eq!(restored, *ValueDomain::data());
    }

    #[test]
    fn decoding_an_unregistered_name_registers_the_decoded_domain() {
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
        let child = intern_child("nested-parent-child", ValueDomain::data());
        let json = serde_json::to_string(&*child).unwrap();

        let restored: Canonical<ValueDomain> = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.parent(), Some(ValueDomain::data()));
    }

    #[test]
    fn decoding_a_divergent_description_keeps_the_canonical_one() {
        let name = Identifier::new("divergent-description");
        let canonical = ValueDomain::register_root(name.clone(), "original description")
            .expect("the domain registers");
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
    fn decoding_a_conflicting_parent_is_rejected() {
        let name = Identifier::new("conflicting-parent");
        let canonical = ValueDomain::register_child(name.clone(), "desc", ValueDomain::data())
            .expect("the domain registers");
        let conflicting =
            ValueDomain::create(name.clone(), "desc", Some(ValueDomain::address().clone()));
        let json = serde_json::to_string(&conflicting).unwrap();

        let error = serde_json::from_str::<Canonical<ValueDomain>>(&json).unwrap_err();

        assert!(error.to_string().contains("conflicts"), "{error}");
        assert_eq!(ValueDomain::intern_registry().get(&name), Some(canonical));
    }

    #[test]
    fn decoding_a_dropped_parent_is_rejected() {
        let name = Identifier::new("dropped-parent");
        let _canonical = ValueDomain::register_child(name.clone(), "desc", ValueDomain::data())
            .expect("the domain registers");
        let json = serde_json::to_string(&ValueDomain::create(name, "desc", None)).unwrap();

        let error = serde_json::from_str::<Canonical<ValueDomain>>(&json).unwrap_err();

        assert!(error.to_string().contains("conflicts"), "{error}");
    }

    #[test]
    fn decoding_a_divergent_description_under_a_matching_parent_keeps_the_canonical() {
        let name = Identifier::new("divergent-description-parented");
        let canonical = ValueDomain::register_child(name.clone(), "original", ValueDomain::data())
            .expect("the domain registers");
        let divergent = ValueDomain::create(name, "divergent", Some(ValueDomain::data().clone()));
        let json = serde_json::to_string(&divergent).unwrap();

        let restored: Canonical<ValueDomain> = serde_json::from_str(&json).unwrap();

        assert_eq!(restored, canonical);
        assert_eq!(restored.description(), "original");
    }

    /// Test a payload with an unknown field or without its parent is
    /// rejected, naming the offending field.
    #[rstest]
    #[case::unknown_field(",\"parent\":null,\"surprise\":1", "surprise")]
    #[case::missing_parent("", "missing field `parent`")]
    fn decoding_a_malformed_payload_is_rejected(
        #[case] fields_after_the_description: &str,
        #[case] expected_message: &str,
    ) {
        let id = reserve_pinned_id("malformed-payload-anchor");
        let json = format!(
            "{{\"name\":{{\"id\":{id},\"name_hint\":\"x\"}},\
             \"description\":\"desc\"{fields_after_the_description}}}"
        );

        let error = serde_json::from_str::<Canonical<ValueDomain>>(&json).unwrap_err();

        assert!(error.to_string().contains(expected_message), "{error}");
    }

    /// Return the JSON payload of a domain named `id` whose `parent` field
    /// holds the JSON `parent` and whose trailing fields are `trailing`.
    fn encode_domain_payload(id: u64, parent: &str, trailing: &str) -> String {
        format!(
            "{{\"name\":{{\"id\":{id},\"name_hint\":\"n{id}\"}},\
             \"description\":\"d{id}\",\"parent\":{parent}{trailing}}}"
        )
    }

    /// Return the domain registered under the id `id`, restoring `id` to
    /// look it up, so call it only after checking the counter.
    fn find_registered(id: u64) -> Option<Canonical<ValueDomain>> {
        ValueDomain::intern_registry()
            .get(&Identifier::try_restore(id, "").expect("the id is below the cap"))
    }

    #[test]
    fn a_payload_rejected_for_a_trailing_unknown_field_registers_its_fresh_parent_nowhere() {
        let _counter = hold_id_counter();
        let [outer, parent] = reserve_far_ahead_ids("trailing-unknown-anchor");
        let json = encode_domain_payload(
            outer,
            &encode_domain_payload(parent, "null", ""),
            ",\"zzz\":1",
        );

        let error = serde_json::from_str::<Canonical<ValueDomain>>(&json).unwrap_err();

        assert!(error.to_string().contains("zzz"), "{error}");
        assert!(!has_counter_passed(outer));
        assert_eq!(find_registered(parent), None);
    }

    #[test]
    fn a_payload_rejected_for_a_trailing_unknown_field_registers_no_fresh_ancestor() {
        let _counter = hold_id_counter();
        let [outer, parent, grandparent] = reserve_far_ahead_ids("trailing-unknown-deep-anchor");
        let json = encode_domain_payload(
            outer,
            &encode_domain_payload(parent, &encode_domain_payload(grandparent, "null", ""), ""),
            ",\"zzz\":1",
        );

        let error = serde_json::from_str::<Canonical<ValueDomain>>(&json).unwrap_err();

        assert!(error.to_string().contains("zzz"), "{error}");
        assert!(!has_counter_passed(outer));
        assert_eq!(find_registered(grandparent), None);
        assert_eq!(find_registered(parent), None);
    }

    #[test]
    fn a_payload_rejected_inside_its_parent_restores_only_the_outer_name() {
        let _counter = hold_id_counter();
        let [outer, parent, grandparent] = reserve_far_ahead_ids("rejected-parent-anchor");
        let json = encode_domain_payload(
            outer,
            &encode_domain_payload(
                parent,
                &encode_domain_payload(grandparent, "null", ""),
                ",\"zzz\":1",
            ),
            "",
        );

        let error = serde_json::from_str::<Canonical<ValueDomain>>(&json).unwrap_err();

        assert!(error.to_string().contains("zzz"), "{error}");
        assert!(has_counter_passed(outer));
        assert!(!has_counter_passed(parent));
        assert_eq!(find_registered(grandparent), None);
        assert_eq!(find_registered(parent), None);
    }

    #[test]
    fn a_payload_rejected_inside_its_grandparent_restores_the_names_above_it() {
        let _counter = hold_id_counter();
        let [outer, parent, grandparent] = reserve_far_ahead_ids("rejected-grandparent-anchor");
        let json = encode_domain_payload(
            outer,
            &encode_domain_payload(
                parent,
                &encode_domain_payload(grandparent, "null", ",\"zzz\":1"),
                "",
            ),
            "",
        );

        let error = serde_json::from_str::<Canonical<ValueDomain>>(&json).unwrap_err();

        assert!(error.to_string().contains("zzz"), "{error}");
        assert!(has_counter_passed(parent));
        assert!(!has_counter_passed(grandparent));
        assert_eq!(find_registered(grandparent), None);
        assert_eq!(find_registered(parent), None);
    }

    #[test]
    fn a_payload_whose_parent_key_comes_first_restores_the_outer_name_first() {
        let _counter = hold_id_counter();
        let [outer, parent] = reserve_far_ahead_ids("parent-first-anchor");
        let json = format!(
            "{{\"parent\":{},\"name\":{{\"id\":{outer},\"name_hint\":\"outer\"}},\
             \"description\":\"desc\"}}",
            encode_domain_payload(parent, "null", ",\"zzz\":1")
        );

        let error = serde_json::from_str::<Canonical<ValueDomain>>(&json).unwrap_err();

        assert!(error.to_string().contains("zzz"), "{error}");
        assert!(has_counter_passed(outer));
        assert!(!has_counter_passed(parent));
    }

    #[test]
    fn a_payload_whose_parent_holds_an_out_of_range_id_restores_only_the_outer_name() {
        let _counter = hold_id_counter();
        let [outer] = reserve_far_ahead_ids("out-of-range-parent-anchor");
        let json = encode_domain_payload(
            outer,
            &format!(
                "{{\"name\":{{\"id\":{},\"name_hint\":\"max\"}},\
                 \"description\":\"desc\",\"parent\":null}}",
                u64::MAX
            ),
            "",
        );

        let error = serde_json::from_str::<Canonical<ValueDomain>>(&json).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("an id from 0 to 9223372036854775807"),
            "{error}"
        );
        assert!(has_counter_passed(outer));
    }

    #[test]
    fn a_payload_whose_parent_is_not_a_map_restores_nothing() {
        let _counter = hold_id_counter();
        let [outer] = reserve_far_ahead_ids("non-map-parent-anchor");
        let json = encode_domain_payload(outer, "\"data\"", "");

        let error = serde_json::from_str::<Canonical<ValueDomain>>(&json).unwrap_err();

        assert!(error.to_string().contains("invalid type"), "{error}");
        assert!(!has_counter_passed(outer));
    }

    #[test]
    fn a_nested_payload_missing_its_parent_is_rejected() {
        let _counter = hold_id_counter();
        let [outer, parent] = reserve_far_ahead_ids("nested-missing-parent-anchor");
        let nested = format!(
            "{{\"name\":{{\"id\":{parent},\"name_hint\":\"p\"}},\"description\":\"desc\"}}"
        );
        let json = encode_domain_payload(outer, &nested, "");

        let error = serde_json::from_str::<Canonical<ValueDomain>>(&json).unwrap_err();

        assert!(
            error.to_string().contains("missing field `parent`"),
            "{error}"
        );
        assert!(!has_counter_passed(parent));
    }

    #[test]
    fn a_conflicting_payload_restores_every_name_and_registers_its_fresh_parent() {
        let _counter = hold_id_counter();
        let name = Identifier::new("conflict-after-fresh-parent");
        let canonical = ValueDomain::register_child(name.clone(), "desc", ValueDomain::data())
            .expect("the domain registers");
        let [parent] = reserve_far_ahead_ids("conflict-after-fresh-parent-anchor");
        let json = encode_domain_payload(name.id(), &encode_domain_payload(parent, "null", ""), "");

        let error = serde_json::from_str::<Canonical<ValueDomain>>(&json).unwrap_err();

        assert!(error.to_string().contains("conflicts"), "{error}");
        assert!(has_counter_passed(parent));
        let registered_parent = find_registered(parent).expect("the fresh parent registers");
        assert_eq!(registered_parent.description(), format!("d{parent}"));
        assert_eq!(ValueDomain::intern_registry().get(&name), Some(canonical));
    }

    #[test]
    fn a_payload_whose_parent_conflicts_registers_only_the_fresh_grandparent() {
        let _counter = hold_id_counter();
        let parent_name = Identifier::new("conflicting-middle");
        let _canonical_parent =
            ValueDomain::register_child(parent_name.clone(), "desc", ValueDomain::data())
                .expect("the name is fresh");
        let [outer, grandparent] = reserve_far_ahead_ids("conflicting-middle-anchor");
        let json = encode_domain_payload(
            outer,
            &encode_domain_payload(
                parent_name.id(),
                &encode_domain_payload(grandparent, "null", ""),
                "",
            ),
            "",
        );

        let error = serde_json::from_str::<Canonical<ValueDomain>>(&json).unwrap_err();

        assert!(error.to_string().contains("conflicts"), "{error}");
        assert!(has_counter_passed(grandparent));
        assert!(find_registered(grandparent).is_some());
        assert_eq!(find_registered(outer), None);
    }

    #[test]
    fn debug_mentions_the_name_hint_and_description() {
        let domain = ValueDomain::register_root(Identifier::new("debug-domain"), "debug desc")
            .expect("the domain registers");

        let rendered = format!("{domain:?}");

        assert!(rendered.contains("debug-domain"), "{rendered}");
        assert!(rendered.contains("debug desc"), "{rendered}");
    }
}
