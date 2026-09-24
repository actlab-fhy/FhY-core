//! Open, registry-backed classification of value kinds in a compiler IR.
//!
//! A [`ValueDomain`] says what kind of value an operation produces or
//! consumes, such as concrete data or an address. The set is open: a layer
//! registers the domains it needs without changing this crate, and
//! [`ValueDomain::data`] and [`ValueDomain::address`] return the two shipped
//! here. Use them rather than a fresh domain with the same name hint:
//! identifiers compare by id, so a second `Identifier::new("data")` is a
//! different key.
//!
//! Domains form a hierarchy through their parents, which
//! [`ValueDomain::is_subdomain_of`] walks.
//!
//! A description takes no part in equality, hashing or interning: the first
//! domain registered under a name stays canonical, and a later registration
//! of that name returns it, dropping its own description. A name has one
//! parent, so registering it again under a parent of another name is a
//! [`ValueDomainConflict`].

use std::fmt;
use std::hash::{Hash, Hasher};
use std::iter;
use std::sync::LazyLock;

use serde::de::{self, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::identifier::{HasIdentifier, Identifier, reserved};
use crate::interned::{Canonical, InternOutcome, InternRegistry, Interned, require_default};

/// Open classification of the kind of value an IR operation handles.
///
/// Two domains are equal when they carry the same [`Identifier`], whatever
/// their descriptions say. Registration keeps one parent per name, so the
/// name decides the parent too.
///
/// A domain encodes as the flat list of its chain, root first, each level
/// written as `{"name": <identifier>, "description": ..}`, so encoding and
/// decoding take no stack per level. Only a [`Canonical<ValueDomain>`]
/// decodes: it registers each level under the one before it, root first,
/// and fails with the [`ValueDomainConflict`] of the first level whose name
/// is registered under another parent, leaving the levels before it
/// registered.
#[derive(Debug)]
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
    /// The walk follows parents and compares names, so it takes time linear
    /// in the depth of this domain. The relation is reflexive and one-way: a
    /// domain is a subdomain of its ancestors and of itself, never of its
    /// descendants or of an unrelated domain.
    #[must_use]
    pub fn is_subdomain_of(&self, other: &ValueDomain) -> bool {
        self.chain().any(|domain| domain == other)
    }

    /// Yield this domain, then each ancestor up to the root.
    ///
    /// A parent exists before its child is built and never changes, so the
    /// chain cannot cycle.
    fn chain(&self) -> impl Iterator<Item = &Self> {
        iter::successors(Some(self), |domain| domain.parent.as_deref())
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

impl Serialize for ValueDomain {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let chain: Vec<&Self> = self.chain().collect();
        serializer.collect_seq(chain.into_iter().rev().map(|domain| LevelRef {
            name: &domain.name,
            description: &domain.description,
        }))
    }
}

/// One level of a domain's encoded chain, borrowed for encoding.
#[derive(Serialize)]
#[serde(rename = "ValueDomainLevel")]
struct LevelRef<'a> {
    name: &'a Identifier,
    description: &'a str,
}

/// One level of a domain's encoded chain, decoded.
#[derive(Deserialize)]
#[serde(
    rename = "ValueDomainLevel",
    expecting = "a value domain level",
    deny_unknown_fields
)]
struct Level {
    name: Identifier,
    description: String,
}

/// Decodes the flat chain a [`ValueDomain`] encodes as, one level at a time,
/// registering each level under the one before it.
impl<'de> Deserialize<'de> for Canonical<ValueDomain> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_seq(ChainVisitor)
    }
}

struct ChainVisitor;

impl<'de> Visitor<'de> for ChainVisitor {
    type Value = Canonical<ValueDomain>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a non-empty list of value domain levels, root first")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut levels: A) -> Result<Self::Value, A::Error> {
        let mut domain: Option<Canonical<ValueDomain>> = None;
        while let Some(level) = levels.next_element::<Level>()? {
            let registered = match &domain {
                None => ValueDomain::register_root(level.name, level.description),
                Some(parent) => ValueDomain::register_child(level.name, level.description, parent),
            };
            domain = Some(registered.map_err(de::Error::custom)?);
        }
        domain.ok_or_else(|| de::Error::invalid_length(0, &self))
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
        self.name == other.name
    }
}

impl Eq for ValueDomain {}

impl Hash for ValueDomain {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

fn create_default_domains() -> Vec<ValueDomain> {
    vec![
        ValueDomain::create(
            Identifier::reserved(reserved::DATA_DOMAIN),
            "Concrete data values flowing through the IR.",
            None,
        ),
        ValueDomain::create(
            Identifier::reserved(reserved::ADDRESS_DOMAIN),
            "Index, offset, or address values used to access data.",
            None,
        ),
    ]
}

static DATA_DOMAIN: LazyLock<Canonical<ValueDomain>> =
    LazyLock::new(|| require_default(&Identifier::reserved(reserved::DATA_DOMAIN)));

static ADDRESS_DOMAIN: LazyLock<Canonical<ValueDomain>> =
    LazyLock::new(|| require_default(&Identifier::reserved(reserved::ADDRESS_DOMAIN)));

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

    use serde_json::{Value, json};

    use crate::test_support::compute_hash;

    fn intern_root(name_hint: &str) -> Canonical<ValueDomain> {
        ValueDomain::register_root(Identifier::new(name_hint), "root")
            .expect("the domain registers")
    }

    fn intern_child(name_hint: &str, parent: &Canonical<ValueDomain>) -> Canonical<ValueDomain> {
        ValueDomain::register_child(Identifier::new(name_hint), "child", parent)
            .expect("the domain registers")
    }

    #[test]
    fn register_root_stores_the_name_description_and_absent_parent() {
        let name = Identifier::new("stores-name-and-description");
        let domain =
            ValueDomain::register_root(name.clone(), "a domain").expect("the domain registers");

        assert_eq!(domain.name(), &name);
        assert_eq!(domain.description(), "a domain");
        assert_eq!(domain.parent(), None);
    }

    #[test]
    fn register_child_stores_the_parent_it_is_given() {
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
    fn domains_with_the_same_name_are_equal_whatever_the_description() {
        let name = Identifier::new("equality-ignores-description");
        let first = ValueDomain::create(name.clone(), "first description", None);
        let second = ValueDomain::create(name, "second description", None);

        assert_eq!(first, second);
        assert_eq!(second, first);
    }

    /// Test equality and hashing depend on the name alone: registration keeps
    /// one parent per name, so two values under one name are the same domain
    /// whatever parent or description they carry.
    #[test]
    fn equality_and_hash_depend_only_on_the_name() {
        let name = Identifier::new("name-only");
        let under_data = ValueDomain::create(name.clone(), "a", Some(ValueDomain::data().clone()));
        let root = ValueDomain::create(name, "b", None);
        let other = ValueDomain::create(Identifier::new("name-only"), "a", None);

        assert_eq!(under_data, root);
        assert_eq!(compute_hash(&under_data), compute_hash(&root));
        assert_ne!(root, other);
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
    fn equality_ignores_the_parent() {
        let name = Identifier::new("parent-differs");
        let parented = ValueDomain::create(name.clone(), "desc", Some(ValueDomain::data().clone()));
        let orphan = ValueDomain::create(name, "desc", None);

        assert_eq!(parented, orphan);
    }

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

    #[rstest]
    #[case::data(ValueDomain::data)]
    #[case::address(ValueDomain::address)]
    fn a_default_domain_has_no_parent(
        #[case] get_default: fn() -> &'static Canonical<ValueDomain>,
    ) {
        assert_eq!(get_default().parent(), None);
    }

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

        /// Test that every domain of any hierarchy decodes from its JSON back
        /// to its own canonical handle.
        #[test]
        fn a_domain_in_any_hierarchy_round_trips_through_json(
            choices in build_hierarchy_strategy(),
        ) {
            let (domains, _parents) = intern_hierarchy(&choices);

            for domain in &domains {
                let json = serde_json::to_string(&**domain).unwrap();
                let restored: Canonical<ValueDomain> = serde_json::from_str(&json).unwrap();

                prop_assert!(Canonical::ptr_eq(&restored, domain));
            }
        }

        /// Test that every domain of any hierarchy decodes from its postcard
        /// bytes back to its own canonical handle.
        #[test]
        fn a_domain_in_any_hierarchy_round_trips_through_postcard(
            choices in build_hierarchy_strategy(),
        ) {
            let (domains, _parents) = intern_hierarchy(&choices);

            for domain in &domains {
                let bytes = postcard::to_allocvec(&**domain).unwrap();
                let restored: Canonical<ValueDomain> = postcard::from_bytes(&bytes).unwrap();

                prop_assert!(Canonical::ptr_eq(&restored, domain));
            }
        }
    }

    fn encode_level(name: &Identifier, description: &str) -> Value {
        json!({
            "name": {"id": name.id(), "name_hint": name.name_hint()},
            "description": description,
        })
    }

    #[test]
    fn a_root_domain_encodes_as_a_chain_of_one_level() {
        let name = Identifier::new("encoded");
        let domain =
            ValueDomain::register_root(name.clone(), "a description").expect("the name is fresh");

        let encoded = serde_json::to_value(&*domain).unwrap();

        assert_eq!(encoded, json!([encode_level(&name, "a description")]));
    }

    #[test]
    fn a_child_domain_encodes_its_chain_root_first() {
        let parent_name = Identifier::new("encoded-parent");
        let parent = ValueDomain::register_root(parent_name.clone(), "the parent")
            .expect("the name is fresh");
        let child_name = Identifier::new("encoded-child");
        let child = ValueDomain::register_child(child_name.clone(), "the child", &parent)
            .expect("the name is fresh");

        let encoded = serde_json::to_value(&*child).unwrap();

        assert_eq!(
            encoded,
            json!([
                encode_level(&parent_name, "the parent"),
                encode_level(&child_name, "the child"),
            ])
        );
    }

    #[test]
    fn a_domain_round_trips_through_json() {
        let domain = intern_child("round-trip", ValueDomain::data());

        let json = serde_json::to_string(&*domain).unwrap();
        let restored: Canonical<ValueDomain> = serde_json::from_str(&json).unwrap();

        assert!(Canonical::ptr_eq(&restored, &domain));
    }

    #[test]
    fn decoding_a_registered_name_returns_the_canonical_domain() {
        let json = serde_json::to_string(ValueDomain::data()).unwrap();

        let restored: Canonical<ValueDomain> = serde_json::from_str(&json).unwrap();

        assert!(Canonical::ptr_eq(&restored, ValueDomain::data()));
    }

    #[test]
    fn decoding_an_unregistered_name_registers_the_decoded_domain() {
        let name = Identifier::new("never-registered");
        let payload = json!([encode_level(&name, "fresh from decode")]);

        let restored: Canonical<ValueDomain> = serde_json::from_value(payload).unwrap();

        assert_eq!(restored.description(), "fresh from decode");
        assert_eq!(restored.parent(), None);
        let registered = ValueDomain::intern_registry()
            .get(&name)
            .expect("the decoded domain registers");
        assert!(Canonical::ptr_eq(&registered, &restored));
    }

    #[test]
    fn decoding_a_chain_canonicalizes_its_parent() {
        let child = intern_child("nested-parent-child", ValueDomain::data());
        let json = serde_json::to_string(&*child).unwrap();

        let restored: Canonical<ValueDomain> = serde_json::from_str(&json).unwrap();

        let parent = restored.parent().expect("the chain names a parent");
        assert!(Canonical::ptr_eq(parent, ValueDomain::data()));
    }

    #[test]
    fn decoding_a_divergent_description_keeps_the_canonical_one() {
        let name = Identifier::new("divergent-description");
        let canonical =
            ValueDomain::register_root(name.clone(), "original description").expect("fresh");
        let payload = json!([encode_level(&name, "divergent description")]);

        let restored: Canonical<ValueDomain> = serde_json::from_value(payload).unwrap();

        assert!(Canonical::ptr_eq(&restored, &canonical));
        assert_eq!(restored.description(), "original description");
    }

    #[test]
    fn decoding_a_divergent_description_under_a_matching_parent_keeps_the_canonical() {
        let name = Identifier::new("divergent-description-parented");
        let canonical = ValueDomain::register_child(name.clone(), "original", ValueDomain::data())
            .expect("the name is fresh");
        let payload = json!([
            encode_level(ValueDomain::data().name(), "data"),
            encode_level(&name, "divergent"),
        ]);

        let restored: Canonical<ValueDomain> = serde_json::from_value(payload).unwrap();

        assert!(Canonical::ptr_eq(&restored, &canonical));
        assert_eq!(restored.description(), "original");
    }

    #[test]
    fn decoding_a_conflicting_parent_is_rejected() {
        let name = Identifier::new("conflicting-parent");
        let canonical = ValueDomain::register_child(name.clone(), "desc", ValueDomain::data())
            .expect("the name is fresh");
        let payload = json!([
            encode_level(ValueDomain::address().name(), "address"),
            encode_level(&name, "desc"),
        ]);

        let error = serde_json::from_value::<Canonical<ValueDomain>>(payload).unwrap_err();

        assert!(
            error.to_string().contains("already registered with parent"),
            "{error}"
        );
        let registered = ValueDomain::intern_registry().get(&name);
        assert!(registered.is_some_and(|registered| Canonical::ptr_eq(&registered, &canonical)));
    }

    #[test]
    fn decoding_a_dropped_parent_is_rejected() {
        let name = Identifier::new("dropped-parent");
        let _canonical = ValueDomain::register_child(name.clone(), "desc", ValueDomain::data())
            .expect("the name is fresh");
        let payload = json!([encode_level(&name, "desc")]);

        let error = serde_json::from_value::<Canonical<ValueDomain>>(payload).unwrap_err();

        assert!(
            error.to_string().contains("already registered with parent"),
            "{error}"
        );
    }

    #[test]
    fn a_decoded_payload_with_a_conflicting_parent_reports_the_conflict() {
        let name = Identifier::new("reported-conflict");
        let _canonical = ValueDomain::register_child(name.clone(), "desc", ValueDomain::data())
            .expect("the name is fresh");
        let payload = json!([
            encode_level(ValueDomain::address().name(), "address"),
            encode_level(&name, "desc"),
        ]);

        let error = serde_json::from_value::<Canonical<ValueDomain>>(payload).unwrap_err();

        let conflict = ValueDomain::register_child(name, "desc", ValueDomain::address())
            .expect_err("the name is registered under data");
        assert!(
            error.to_string().starts_with(&conflict.to_string()),
            "{error}"
        );
    }

    /// Test that a chain rejected at one level leaves the levels before it
    /// registered and nothing after it.
    #[test]
    fn a_chain_rejected_at_one_level_registers_the_levels_before_it() {
        let conflicting = Identifier::new("conflicting-level");
        let _canonical =
            ValueDomain::register_child(conflicting.clone(), "desc", ValueDomain::data())
                .expect("the name is fresh");
        let [root, leaf] = ["rejected-chain-root", "rejected-chain-leaf"].map(Identifier::new);
        let payload = json!([
            encode_level(&root, "root"),
            encode_level(&conflicting, "desc"),
            encode_level(&leaf, "leaf"),
        ]);

        let error = serde_json::from_value::<Canonical<ValueDomain>>(payload);

        assert!(error.is_err(), "the conflicting level decoded");
        assert!(ValueDomain::intern_registry().get(&root).is_some());
        assert!(ValueDomain::intern_registry().get(&leaf).is_none());
    }

    #[rstest]
    #[case::unknown_field(json!({"surprise": 1}), "unknown field `surprise`")]
    #[case::missing_description(json!({"description": null}), "missing field `description`")]
    #[case::not_a_level(json!("data"), "invalid type: string")]
    fn decoding_a_malformed_level_is_rejected(
        #[case] damage: Value,
        #[case] expected_message: &str,
    ) {
        let mut level = encode_level(&Identifier::new("malformed-level"), "desc");
        match damage {
            Value::Object(fields) => {
                for (key, value) in fields {
                    if value.is_null() {
                        level
                            .as_object_mut()
                            .expect("a level is a map")
                            .remove(&key);
                    } else {
                        level[key] = value;
                    }
                }
            }
            replacement => level = replacement,
        }
        let payload = json!([level]);

        let error = serde_json::from_value::<Canonical<ValueDomain>>(payload).unwrap_err();

        assert!(error.to_string().contains(expected_message), "{error}");
    }

    #[rstest]
    #[case::empty(json!([]), "invalid length 0")]
    #[case::a_map(json!({"name": 1}), "invalid type: map")]
    #[case::null(json!(null), "invalid type: null")]
    fn decoding_a_payload_that_is_not_a_chain_is_rejected(
        #[case] payload: Value,
        #[case] expected_message: &str,
    ) {
        let error = serde_json::from_value::<Canonical<ValueDomain>>(payload).unwrap_err();

        assert!(error.to_string().contains(expected_message), "{error}");
        assert!(
            error
                .to_string()
                .contains("a non-empty list of value domain levels, root first"),
            "{error}"
        );
    }

    #[test]
    fn a_parent_with_an_out_of_range_id_is_rejected() {
        let payload = json!([
            {"name": {"id": u64::MAX, "name_hint": "max"}, "description": "desc"},
            encode_level(&Identifier::new("out-of-range-parent"), "desc"),
        ]);

        let error = serde_json::from_value::<Canonical<ValueDomain>>(payload).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("an id from 0 to 9223372036854775807"),
            "{error}"
        );
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
