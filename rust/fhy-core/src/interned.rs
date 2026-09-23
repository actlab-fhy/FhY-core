//! Canonical instances of values identified by a key.
//!
//! A type implements [`Interned`] when each of its values carries a key and
//! at most one value per key is shared as *the* canonical instance. Every
//! such type owns one process-wide [`InternRegistry`]. Interning a value
//! either registers it as the canonical instance for its key or, when a
//! canonical instance already exists, keeps that one and hands the new value
//! back. Either way the caller receives a [`Canonical`] handle, so a
//! non-canonical duplicate is never shared.
//!
//! Canonical handles compare and hash by identity. Two handles are equal iff
//! they point at the same registered instance, which for handles taken from
//! one registry with no clear between them means iff their keys are equal.
//!
//! A registry may be created with default instances, which it registers the
//! first time it is used and restores whenever it is cleared. A default
//! keeps its identity across clears, so a handle to a default taken before a
//! clear still equals the handle taken after it.
//!
//! Registries are safe to use from many threads at once.

use std::borrow::Borrow;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::{Arc, OnceLock, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A type whose values are canonicalized by key.
///
/// Implementations must uphold three rules:
///
/// - [`intern_key`](Self::intern_key) returns the same key for a value on
///   every call.
/// - [`intern_registry`](Self::intern_registry) returns the same registry on
///   every call, normally a `static` declared inside the method.
/// - Neither `intern_key` nor the key's `Hash` and `Eq` implementations
///   access the type's registry. The registry calls them while it holds its
///   lock.
///
/// # Examples
///
/// ```
/// use fhy_core::interned::{InternRegistry, Interned};
///
/// #[derive(Debug)]
/// struct Tag {
///     name: String,
/// }
///
/// impl Interned for Tag {
///     type Key = String;
///
///     fn intern_key(&self) -> &String {
///         &self.name
///     }
///
///     fn intern_registry() -> &'static InternRegistry<Self> {
///         static REGISTRY: InternRegistry<Tag> = InternRegistry::new();
///         &REGISTRY
///     }
/// }
///
/// let registry = Tag::intern_registry();
/// let first = registry.intern(Tag { name: "x".to_string() }).into_canonical();
/// let again = registry.intern(Tag { name: "x".to_string() }).into_canonical();
///
/// assert_eq!(first, again);
/// assert_eq!(registry.get("x"), Some(first));
/// ```
pub trait Interned: Sized + Send + Sync + 'static {
    /// Key under which the canonical instance is registered.
    type Key: Eq + Hash + Clone + fmt::Debug + Send + Sync + 'static;

    /// Return the key this value is interned under.
    fn intern_key(&self) -> &Self::Key;

    /// Return the process-wide registry holding this type's canonical
    /// instances.
    fn intern_registry() -> &'static InternRegistry<Self>;
}

/// Thread-safe map from keys to the canonical instances registered under
/// them.
///
/// A registry is normally a `static` returned by
/// [`Interned::intern_registry`], but a local registry is independent of it
/// and of every other registry.
///
/// A registry suits small, long-lived vocabularies such as attributes and
/// value domains. Every [`intern`](Self::intern) takes the write lock, even
/// when the key is already registered, and an instance stays registered
/// until a [`clear`](Self::clear), so a registry is not a hash-consing table
/// for IR nodes.
pub struct InternRegistry<T: Interned> {
    create_defaults: fn() -> Vec<T>,
    state: OnceLock<RwLock<RegistryState<T>>>,
}

/// Build a map from each instance's key to the instance, assuming
/// `instances` holds at most one instance per key.
fn index_by_key<T: Interned>(instances: &[Arc<T>]) -> HashMap<T::Key, Arc<T>> {
    let mut entries = HashMap::with_capacity(instances.len());
    for instance in instances {
        entries.insert(instance.intern_key().clone(), Arc::clone(instance));
    }
    entries
}

/// Registered instances, and the defaults a clear restores.
///
/// For keys whose `Hash` and `Eq` behave the same on every call, every
/// method that mutates this state either completes or panics before changing
/// anything (for example, a key's `Hash` panicking mid-lookup), so a panic
/// inside a critical section never leaves the state half-updated. That is
/// what makes recovering a poisoned lock over this state safe.
struct RegistryState<T: Interned> {
    defaults: Vec<Arc<T>>,
    entries: HashMap<T::Key, Arc<T>>,
}

impl<T: Interned> RegistryState<T> {
    /// Create the state from freshly constructed default values, keeping
    /// only the first value registered under each key.
    fn from_defaults(values: Vec<T>) -> Self {
        let mut defaults: Vec<Arc<T>> = Vec::with_capacity(values.len());
        let mut entries: HashMap<T::Key, Arc<T>> = HashMap::with_capacity(values.len());
        for value in values {
            if let Entry::Vacant(slot) = entries.entry(value.intern_key().clone()) {
                let instance = Arc::new(value);
                slot.insert(Arc::clone(&instance));
                defaults.push(instance);
            }
        }
        Self { defaults, entries }
    }

    /// Unregister every instance except the defaults, restoring each
    /// default's original identity.
    fn restore_defaults(&mut self) {
        self.entries = index_by_key(&self.defaults);
    }
}

impl<T: Interned> InternRegistry<T> {
    /// Create an empty registry.
    #[must_use]
    pub const fn new() -> Self {
        Self::with_defaults(Vec::new)
    }

    /// Create a registry whose defaults are the values `create_defaults`
    /// returns.
    ///
    /// `create_defaults` runs on the registry's first use, and its values are
    /// registered in order, so the first of several defaults sharing a key
    /// becomes canonical. It must build plain values and must not access
    /// this registry.
    #[must_use]
    pub const fn with_defaults(create_defaults: fn() -> Vec<T>) -> Self {
        Self {
            create_defaults,
            state: OnceLock::new(),
        }
    }

    /// Register `value` as the canonical instance for its key, unless one is
    /// already registered.
    ///
    /// # Panics
    ///
    /// Panics if this is the registry's first use and `create_defaults`
    /// panics.
    pub fn intern(&self, value: T) -> InternOutcome<T> {
        let mut state = self.write_state();
        match state.entries.get(value.intern_key()) {
            None => {
                let key = value.intern_key().clone();
                let instance = Arc::new(value);
                state.entries.insert(key, Arc::clone(&instance));
                InternOutcome::Registered(Canonical(instance))
            }
            Some(existing) => InternOutcome::AlreadyCanonical {
                canonical: Canonical(Arc::clone(existing)),
                discarded: value,
            },
        }
    }

    /// Return the canonical instance registered under `key`, if any.
    ///
    /// # Panics
    ///
    /// Panics if this is the registry's first use and `create_defaults`
    /// panics.
    #[must_use]
    pub fn get<Q>(&self, key: &Q) -> Option<Canonical<T>>
    where
        T::Key: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.read_state()
            .entries
            .get(key)
            .map(|instance| Canonical(Arc::clone(instance)))
    }

    /// Return the canonical instance registered under `key`.
    ///
    /// # Errors
    ///
    /// Returns [`NotInternedError`] if no instance is registered under
    /// `key`.
    ///
    /// # Panics
    ///
    /// Panics if this is the registry's first use and `create_defaults`
    /// panics.
    pub fn require<Q>(&self, key: &Q) -> Result<Canonical<T>, NotInternedError<T::Key>>
    where
        T::Key: Borrow<Q>,
        Q: Hash + Eq + ToOwned<Owned = T::Key> + ?Sized,
    {
        self.get(key).ok_or_else(|| NotInternedError {
            type_name: std::any::type_name::<T>(),
            key: key.to_owned(),
        })
    }

    /// Unregister every instance except the defaults.
    ///
    /// Handles to unregistered instances stay valid but are no longer
    /// canonical: interning an equal value afterwards registers a new
    /// instance that compares unequal to them. Defaults are registered again
    /// with the same identity they had before.
    ///
    /// # Panics
    ///
    /// Panics if this is the registry's first use and `create_defaults`
    /// panics.
    pub fn clear(&self) {
        self.write_state().restore_defaults();
    }

    /// Register the defaults now if this is the registry's first use.
    ///
    /// A decode calls this before it restores any identifier, so the
    /// defaults' names draw their ids from the counter before a payload's id
    /// can exhaust it.
    ///
    /// # Panics
    ///
    /// Panics if this is the registry's first use and `create_defaults`
    /// panics.
    pub(crate) fn initialize(&self) {
        self.state();
    }

    /// Return the registry's state, initializing it from `create_defaults`
    /// on first use.
    fn state(&self) -> &RwLock<RegistryState<T>> {
        self.state.get_or_init(|| {
            let defaults = (self.create_defaults)();
            RwLock::new(RegistryState::from_defaults(defaults))
        })
    }

    /// Take the state's read lock, recovering from poisoning.
    fn read_state(&self) -> RwLockReadGuard<'_, RegistryState<T>> {
        self.state().read().unwrap_or_else(PoisonError::into_inner)
    }

    /// Take the state's write lock, recovering from poisoning.
    fn write_state(&self) -> RwLockWriteGuard<'_, RegistryState<T>> {
        self.state().write().unwrap_or_else(PoisonError::into_inner)
    }
}

impl<T: Interned> Default for InternRegistry<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Interned> fmt::Debug for InternRegistry<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InternRegistry")
            .field("type", &std::any::type_name::<T>())
            .finish_non_exhaustive()
    }
}

/// Result of [`InternRegistry::intern`].
#[must_use = "the outcome holds the canonical handle"]
#[derive(Debug)]
pub enum InternOutcome<T> {
    /// The value became the canonical instance for its key.
    Registered(Canonical<T>),
    /// A canonical instance already existed for the key, so the value was
    /// not registered.
    AlreadyCanonical {
        /// The instance registered under the key.
        canonical: Canonical<T>,
        /// The value that was passed in and not registered.
        discarded: T,
    },
}

impl<T> InternOutcome<T> {
    /// Return the canonical handle.
    #[must_use]
    pub fn canonical(&self) -> &Canonical<T> {
        match self {
            Self::Registered(canonical) | Self::AlreadyCanonical { canonical, .. } => canonical,
        }
    }

    /// Return the canonical handle, dropping any discarded value.
    #[must_use]
    pub fn into_canonical(self) -> Canonical<T> {
        match self {
            Self::Registered(canonical) | Self::AlreadyCanonical { canonical, .. } => canonical,
        }
    }

    /// Return whether the value became the canonical instance.
    #[must_use]
    pub fn is_registered(&self) -> bool {
        matches!(self, Self::Registered(_))
    }
}

/// Shared handle to a canonical instance.
///
/// Handles compare and hash by identity: two handles are equal iff they
/// point at the same registered instance. A handle dereferences to the
/// instance.
///
/// A handle serializes as its instance. Deserializing a handle interns the
/// decoded value in its type's registry and yields the canonical instance
/// for its key, or fails when the decoded value is unequal to a canonical
/// instance already registered under that key.
pub struct Canonical<T>(Arc<T>);

impl<T> Deref for Canonical<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> Clone for Canonical<T> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl<T> PartialEq for Canonical<T> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl<T> Eq for Canonical<T> {}

impl<T> Hash for Canonical<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::ptr::hash(Arc::as_ptr(&self.0), state);
    }
}

impl<T: fmt::Debug> fmt::Debug for Canonical<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&*self.0, f)
    }
}

impl<T: fmt::Display> fmt::Display for Canonical<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&*self.0, f)
    }
}

impl<T: Serialize> Serialize for Canonical<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        T::serialize(&self.0, serializer)
    }
}

/// Decoding interns the decoded value. When its key is already canonical, the
/// decoded value must equal the canonical instance under `T`'s `Eq`, and an
/// unequal one is rejected with an error naming the type and key. A value
/// that differs only in fields `Eq` ignores, such as a description, decodes to
/// the canonical instance without any report: this crate has no logger, so
/// the ignored metadata is dropped silently.
///
/// `T`'s own decode runs first and decides what the payload's nested handles
/// register before the value itself is interned. Rejecting the value as a
/// conflict leaves registered whatever that decode registered.
impl<'de, T: Interned + Eq + Deserialize<'de>> Deserialize<'de> for Canonical<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        intern_decoded(T::deserialize(deserializer)?)
    }
}

/// Intern a decoded value and return the canonical handle for its key.
///
/// # Errors
///
/// Returns an error naming `T` and the key when the key is already canonical
/// and `value` is unequal to the canonical instance, which stays registered.
pub(crate) fn intern_decoded<T: Interned + Eq, E: serde::de::Error>(
    value: T,
) -> Result<Canonical<T>, E> {
    match T::intern_registry().intern(value) {
        InternOutcome::Registered(canonical) => Ok(canonical),
        InternOutcome::AlreadyCanonical {
            canonical,
            discarded,
        } => {
            if discarded == *canonical {
                Ok(canonical)
            } else {
                Err(E::custom(format_args!(
                    "payload for {} under key {:?} conflicts with the canonical instance",
                    std::any::type_name::<T>(),
                    canonical.intern_key()
                )))
            }
        }
    }
}

/// Return the canonical instance of `T` registered under a default's key.
///
/// # Panics
///
/// Panics if `key` is not the key of one of the defaults `T`'s registry was
/// created with, since the registry registers every default on its first
/// use.
pub(crate) fn require_default<T: Interned>(key: &T::Key) -> Canonical<T> {
    T::intern_registry()
        .require(key)
        .expect("the registry registers every default on its first use")
}

/// No canonical instance is registered under a key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NotInternedError<K> {
    type_name: &'static str,
    key: K,
}

impl<K: fmt::Debug> fmt::Display for NotInternedError<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "no canonical {} is interned under key {:?}",
            self.type_name, self.key
        )
    }
}

impl<K: fmt::Debug> std::error::Error for NotInternedError<K> {}

impl<K> NotInternedError<K> {
    /// Return the key that was looked up.
    #[must_use]
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Return the name of the interned type that was searched.
    #[must_use]
    pub fn type_name(&self) -> &'static str {
        self.type_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::panic::{self, AssertUnwindSafe};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    use crate::test_support::assert_send_sync;

    /// Interned fixture type: a name-keyed tag carrying a `note` that
    /// identifies the exact instance under test.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Tag {
        name: String,
        note: String,
    }

    impl fmt::Display for Tag {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.name)
        }
    }

    impl Interned for Tag {
        type Key = String;

        fn intern_key(&self) -> &String {
            &self.name
        }

        fn intern_registry() -> &'static InternRegistry<Tag> {
            static REGISTRY: InternRegistry<Tag> = InternRegistry::new();
            &REGISTRY
        }
    }

    /// Interned fixture type whose `label` is metadata that `Eq` ignores.
    #[derive(Debug, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct LabeledTag {
        name: String,
        label: String,
    }

    impl PartialEq for LabeledTag {
        fn eq(&self, other: &Self) -> bool {
            self.name == other.name
        }
    }

    impl Eq for LabeledTag {}

    impl Interned for LabeledTag {
        type Key = String;

        fn intern_key(&self) -> &String {
            &self.name
        }

        fn intern_registry() -> &'static InternRegistry<LabeledTag> {
            static REGISTRY: InternRegistry<LabeledTag> = InternRegistry::new();
            &REGISTRY
        }
    }

    fn build_tag(name: &str, note: &str) -> Tag {
        Tag {
            name: name.to_string(),
            note: note.to_string(),
        }
    }

    fn create_tag_defaults() -> Vec<Tag> {
        vec![
            build_tag("alpha", "default-alpha"),
            build_tag("beta", "default-beta"),
        ]
    }

    fn create_duplicate_defaults() -> Vec<Tag> {
        vec![build_tag("alpha", "first"), build_tag("alpha", "second")]
    }

    /// Call counter for [`create_counted_defaults`]. Only
    /// `with_defaults_runs_the_default_constructor_once` constructs a
    /// registry with that function, so no other test perturbs this counter.
    static DEFAULT_CALLS: AtomicUsize = AtomicUsize::new(0);

    fn create_counted_defaults() -> Vec<Tag> {
        DEFAULT_CALLS.fetch_add(1, Ordering::SeqCst);
        vec![build_tag("alpha", "default-alpha")]
    }

    /// Key whose `Hash` panics on a specific value, used to poison a
    /// registry's lock on purpose.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TripwireKey(String);

    impl Hash for TripwireKey {
        fn hash<H: Hasher>(&self, state: &mut H) {
            assert!(self.0 != "tripwire", "tripwire key hashed");
            self.0.hash(state);
        }
    }

    struct TripwireTag {
        key: TripwireKey,
    }

    impl Interned for TripwireTag {
        type Key = TripwireKey;

        fn intern_key(&self) -> &TripwireKey {
            &self.key
        }

        fn intern_registry() -> &'static InternRegistry<TripwireTag> {
            static REGISTRY: InternRegistry<TripwireTag> = InternRegistry::new();
            &REGISTRY
        }
    }

    /// Test the first value interned for a key becomes canonical.
    #[test]
    fn intern_registers_the_first_value_for_a_key() {
        let registry = InternRegistry::<Tag>::new();
        let tag = build_tag("alpha", "note");

        let outcome = registry.intern(tag);

        assert!(outcome.is_registered());
        let InternOutcome::Registered(canonical) = outcome else {
            panic!("expected a Registered outcome");
        };
        assert_eq!(canonical.name, "alpha");
        assert_eq!(canonical.note, "note");
    }

    /// Test a repeated key keeps the first interned value canonical.
    #[test]
    fn intern_keeps_the_first_value_canonical_for_a_repeated_key() {
        let registry = InternRegistry::<Tag>::new();
        let first = registry
            .intern(build_tag("alpha", "first-note"))
            .into_canonical();

        let second_outcome = registry.intern(build_tag("alpha", "second-note"));

        let InternOutcome::AlreadyCanonical { canonical, .. } = &second_outcome else {
            panic!("expected an AlreadyCanonical outcome");
        };
        assert_eq!(*canonical, first);
        assert_eq!(registry.get("alpha"), Some(first.clone()));
        let required = registry.require("alpha").expect("alpha is registered");
        assert_eq!(required, first);
        assert_eq!(canonical.note, "first-note");
    }

    /// Test a repeated key hands back the discarded value alongside the
    /// canonical one.
    #[test]
    fn intern_hands_back_the_discarded_value_for_a_repeated_key() {
        let registry = InternRegistry::<Tag>::new();
        let _first = registry.intern(build_tag("alpha", "first-note"));

        let outcome = registry.intern(build_tag("alpha", "second-note"));

        let InternOutcome::AlreadyCanonical {
            canonical,
            discarded,
        } = outcome
        else {
            panic!("expected an AlreadyCanonical outcome");
        };
        assert_eq!(discarded.note, "second-note");
        assert_eq!(canonical.note, "first-note");
    }

    /// Test a repeated key with matching metadata hands back a discarded
    /// value that is equal to the canonical one.
    #[test]
    fn intern_hands_back_an_equal_discarded_value_when_metadata_agrees() {
        let registry = InternRegistry::<Tag>::new();
        let _first = registry.intern(build_tag("alpha", "same-note"));

        let outcome = registry.intern(build_tag("alpha", "same-note"));

        let InternOutcome::AlreadyCanonical {
            canonical,
            discarded,
        } = outcome
        else {
            panic!("expected an AlreadyCanonical outcome");
        };
        assert_eq!(discarded, *canonical);
    }

    /// Test an interned value is retrievable afterward by its key.
    #[test]
    fn intern_makes_the_value_retrievable_by_key() {
        let registry = InternRegistry::<Tag>::new();

        let outcome = registry.intern(build_tag("alpha", "note"));

        let canonical = outcome.into_canonical();
        assert_eq!(registry.get("alpha"), Some(canonical));
    }

    /// Test `get` returns `None` for a key with no canonical instance.
    #[test]
    fn get_returns_none_for_a_missing_key() {
        let registry = InternRegistry::<Tag>::new();

        assert_eq!(registry.get("missing"), None);
    }

    /// Test `get` accepts a borrowed `&str` for a `String` key.
    #[test]
    fn get_accepts_a_borrowed_key() {
        let registry = InternRegistry::<Tag>::new();
        let canonical = registry.intern(build_tag("alpha", "note")).into_canonical();

        let looked_up: Option<Canonical<Tag>> = registry.get("alpha");

        assert_eq!(looked_up, Some(canonical));
    }

    /// Test `require` returns the canonical instance for a registered key.
    #[test]
    fn require_returns_the_canonical_instance_for_a_registered_key() {
        let registry = InternRegistry::<Tag>::new();
        let canonical = registry.intern(build_tag("alpha", "note")).into_canonical();

        let required = registry.require("alpha").expect("alpha is registered");

        assert_eq!(required, canonical);
    }

    /// Test `require` reports the missing key and the interned type's name.
    #[test]
    fn require_reports_the_missing_key_and_type() {
        let registry = InternRegistry::<Tag>::new();

        let result = registry.require("missing");

        let Err(error) = result else {
            panic!("expected an error for a missing key");
        };
        assert_eq!(error.key(), "missing");
        assert!(
            error.type_name().contains("Tag"),
            "got {}",
            error.type_name()
        );
        let message = error.to_string();
        assert!(message.contains("\"missing\""), "got {message}");
    }

    /// Test a missing-key error's message names the interned type and the
    /// key's `Debug` form.
    #[test]
    fn require_error_display_names_the_type_and_key() {
        let registry = InternRegistry::<Tag>::new();

        let Err(error) = registry.require("missing") else {
            panic!("expected an error for a missing key");
        };

        assert_eq!(
            error.to_string(),
            format!(
                "no canonical {} is interned under key \"missing\"",
                std::any::type_name::<Tag>()
            )
        );
    }

    /// Test a missing-key error is a `std::error::Error` with no source.
    #[test]
    fn require_error_is_a_std_error_without_a_source() {
        let registry = InternRegistry::<Tag>::new();

        let Err(error) = registry.require("missing") else {
            panic!("expected an error for a missing key");
        };
        let error: &dyn std::error::Error = &error;

        assert!(error.source().is_none());
    }

    /// Test `require` accepts a borrowed `&str` for a `String` key.
    #[test]
    fn require_accepts_a_borrowed_key() {
        let registry = InternRegistry::<Tag>::new();
        let canonical = registry.intern(build_tag("beta", "note")).into_canonical();

        let required: Canonical<Tag> = registry.require("beta").expect("beta is registered");

        assert_eq!(required, canonical);
    }

    /// Test defaults are registered before any explicit intern.
    #[test]
    fn with_defaults_registers_defaults_before_any_intern() {
        let registry = InternRegistry::<Tag>::with_defaults(create_tag_defaults);

        let alpha = registry.get("alpha").expect("alpha default is registered");
        let beta = registry.get("beta").expect("beta default is registered");

        assert_eq!(alpha.note, "default-alpha");
        assert_eq!(beta.note, "default-beta");
    }

    /// Test duplicate-keyed defaults keep the first one canonical.
    #[test]
    fn with_defaults_keeps_the_first_of_duplicate_default_keys() {
        let registry = InternRegistry::<Tag>::with_defaults(create_duplicate_defaults);

        let canonical = registry.get("alpha").expect("alpha default is registered");

        assert_eq!(canonical.note, "first");
    }

    /// Test the default constructor runs exactly once, on the registry's
    /// first use, regardless of how many operations follow.
    #[test]
    fn with_defaults_runs_the_default_constructor_once() {
        let calls_before = DEFAULT_CALLS.load(Ordering::SeqCst);

        let registry = InternRegistry::<Tag>::with_defaults(create_counted_defaults);
        assert_eq!(DEFAULT_CALLS.load(Ordering::SeqCst), calls_before);

        let _first_get = registry.get("alpha");
        assert_eq!(DEFAULT_CALLS.load(Ordering::SeqCst), calls_before + 1);

        let _second_get = registry.get("alpha");
        let _setup_intern = registry.intern(build_tag("gamma", "note"));
        registry.clear();
        assert_eq!(DEFAULT_CALLS.load(Ordering::SeqCst), calls_before + 1);
    }

    /// Test interning a value under a default's key returns the default as
    /// canonical.
    #[test]
    fn intern_of_a_default_key_returns_the_default() {
        let registry = InternRegistry::<Tag>::with_defaults(create_tag_defaults);
        let default_alpha = registry.get("alpha").expect("alpha default is registered");

        let outcome = registry.intern(build_tag("alpha", "override-note"));

        let InternOutcome::AlreadyCanonical { canonical, .. } = outcome else {
            panic!("expected an AlreadyCanonical outcome");
        };
        assert_eq!(canonical, default_alpha);
    }

    /// Test interning a default's key as a registry's first operation keeps
    /// the default canonical.
    #[test]
    fn intern_as_the_first_operation_defers_to_a_default() {
        let registry = InternRegistry::<Tag>::with_defaults(create_tag_defaults);

        let outcome = registry.intern(build_tag("alpha", "override-note"));

        let InternOutcome::AlreadyCanonical { canonical, .. } = outcome else {
            panic!("expected an AlreadyCanonical outcome");
        };
        assert_eq!(canonical.note, "default-alpha");
    }

    /// Test clearing as a registry's first operation leaves its defaults
    /// registered.
    #[test]
    fn clear_as_the_first_operation_keeps_defaults() {
        let registry = InternRegistry::<Tag>::with_defaults(create_tag_defaults);

        registry.clear();

        let alpha = registry.get("alpha").expect("alpha default is registered");
        assert_eq!(alpha.note, "default-alpha");
    }

    /// Test `clear` unregisters instances that are not defaults.
    #[test]
    fn clear_unregisters_non_default_instances() {
        let registry = InternRegistry::<Tag>::with_defaults(create_tag_defaults);
        let _setup = registry.intern(build_tag("gamma", "note"));

        registry.clear();

        assert_eq!(registry.get("gamma"), None);
    }

    /// Test `clear` keeps the identity of default instances.
    #[test]
    fn clear_keeps_the_identity_of_defaults() {
        let registry = InternRegistry::<Tag>::with_defaults(create_tag_defaults);
        let before = registry.get("alpha").expect("alpha default is registered");

        registry.clear();

        let after = registry.get("alpha").expect("alpha default is restored");
        assert_eq!(before, after);
    }

    /// Test interning after `clear` registers a fresh canonical instance.
    #[test]
    fn intern_after_clear_registers_a_new_canonical_instance() {
        let registry = InternRegistry::<Tag>::new();
        let before = registry
            .intern(build_tag("alpha", "before"))
            .into_canonical();

        registry.clear();
        let outcome = registry.intern(build_tag("alpha", "after"));

        assert!(outcome.is_registered());
        let after = outcome.into_canonical();
        assert_ne!(after, before);
    }

    /// Test `clear` empties a registry that has no defaults.
    #[test]
    fn clear_empties_a_registry_without_defaults() {
        let registry = InternRegistry::<Tag>::new();
        let _setup = registry.intern(build_tag("alpha", "note"));

        registry.clear();

        assert_eq!(registry.get("alpha"), None);
    }

    /// Test a `Default`-constructed registry starts empty.
    #[test]
    fn default_registry_is_empty() {
        let registry = InternRegistry::<Tag>::default();

        assert_eq!(registry.get("alpha"), None);
    }

    /// Test two handles for the same key from one registry are equal.
    #[test]
    fn canonical_handles_from_one_registry_are_equal_for_one_key() {
        let registry = InternRegistry::<Tag>::new();
        let _setup = registry.intern(build_tag("alpha", "note"));

        let first = registry.get("alpha").expect("alpha is registered");
        let second = registry.get("alpha").expect("alpha is registered");

        assert_eq!(first, second);
    }

    /// Test handles for the same key and value from separate registries are
    /// unequal by identity, though their underlying values are equal.
    #[test]
    fn canonical_handles_from_separate_registries_are_unequal() {
        let registry_a = InternRegistry::<Tag>::new();
        let registry_b = InternRegistry::<Tag>::new();

        let a = registry_a
            .intern(build_tag("alpha", "note"))
            .into_canonical();
        let b = registry_b
            .intern(build_tag("alpha", "note"))
            .into_canonical();

        assert_ne!(a, b);
        assert_eq!(*a, *b);
    }

    /// Test canonical handles hash by identity, matching their equality.
    #[test]
    fn canonical_handles_hash_by_identity() {
        let registry = InternRegistry::<Tag>::new();
        let _setup = registry.intern(build_tag("alpha", "note"));
        let first = registry.get("alpha").expect("alpha is registered");
        let second = registry.get("alpha").expect("alpha is registered");

        let mut handles: HashSet<Canonical<Tag>> = HashSet::new();
        handles.insert(first);
        handles.insert(second);
        assert_eq!(handles.len(), 1);

        let other_registry = InternRegistry::<Tag>::new();
        let from_other_registry = other_registry
            .intern(build_tag("alpha", "note"))
            .into_canonical();
        handles.insert(from_other_registry);
        assert_eq!(handles.len(), 2);
    }

    /// Test a canonical handle dereferences to its interned instance.
    #[test]
    fn canonical_dereferences_to_the_instance() {
        let registry = InternRegistry::<Tag>::new();

        let canonical = registry.intern(build_tag("alpha", "note")).into_canonical();

        assert_eq!(canonical.name, "alpha");
        assert_eq!(canonical.note, "note");
        assert_eq!(&*canonical, &build_tag("alpha", "note"));
    }

    /// Test a canonical handle's `Display` matches its instance's `Display`.
    #[test]
    fn canonical_display_matches_the_instance() {
        let registry = InternRegistry::<Tag>::new();
        let tag = build_tag("alpha", "note");
        let expected = tag.to_string();

        let canonical = registry.intern(tag).into_canonical();

        assert_eq!(canonical.to_string(), expected);
    }

    /// Test a canonical handle's `Debug` matches its instance's `Debug`.
    #[test]
    fn canonical_debug_matches_the_instance() {
        let registry = InternRegistry::<Tag>::new();

        let canonical = registry.intern(build_tag("alpha", "note")).into_canonical();

        assert_eq!(format!("{canonical:?}"), format!("{:?}", *canonical));
    }

    /// Test cloning a canonical handle yields a handle equal to the
    /// original.
    #[test]
    fn canonical_clone_equals_the_original() {
        let registry = InternRegistry::<Tag>::new();
        let canonical = registry.intern(build_tag("alpha", "note")).into_canonical();

        let cloned = canonical.clone();

        assert_eq!(cloned, canonical);
    }

    /// Test `InternOutcome::canonical` matches `into_canonical` for both
    /// variants.
    #[test]
    fn intern_outcome_canonical_matches_into_canonical() {
        let registry = InternRegistry::<Tag>::new();

        let registered_outcome = registry.intern(build_tag("alpha", "note"));
        let registered_via_canonical = registered_outcome.canonical().clone();
        assert_eq!(
            registered_via_canonical,
            registered_outcome.into_canonical()
        );

        let already_canonical_outcome = registry.intern(build_tag("alpha", "other-note"));
        let already_canonical_via_canonical = already_canonical_outcome.canonical().clone();
        assert_eq!(
            already_canonical_via_canonical,
            already_canonical_outcome.into_canonical()
        );
    }

    /// Test `is_registered` is false when a canonical instance already
    /// existed for the key.
    #[test]
    fn intern_outcome_is_registered_is_false_for_an_existing_key() {
        let registry = InternRegistry::<Tag>::new();
        let _first = registry.intern(build_tag("alpha", "note"));

        let outcome = registry.intern(build_tag("alpha", "other-note"));

        assert!(!outcome.is_registered());
    }

    /// Test a canonical handle serializes the same as its instance.
    #[test]
    fn canonical_serializes_as_its_instance() {
        let registry = InternRegistry::<Tag>::new();
        let canonical = registry.intern(build_tag("alpha", "note")).into_canonical();

        let handle_json = serde_json::to_value(&canonical).expect("Tag serializes");
        let instance_json = serde_json::to_value(&*canonical).expect("Tag serializes");

        assert_eq!(handle_json, instance_json);
        assert_eq!(
            handle_json,
            serde_json::json!({"name": "alpha", "note": "note"})
        );
    }

    /// Test deserializing an unregistered key registers the decoded value.
    #[test]
    fn canonical_deserialization_registers_an_unregistered_key() {
        let key = "canonical_deserialization_registers_an_unregistered_key";
        let payload = serde_json::json!({"name": key, "note": "decoded"});

        let deserialized: Canonical<Tag> =
            serde_json::from_value(payload).expect("payload deserializes");

        let expected = Tag::intern_registry()
            .get(key)
            .expect("key is registered after deserialization");
        assert_eq!(deserialized, expected);
    }

    /// Test deserializing a registered key with a payload equal to the
    /// canonical instance returns that instance.
    #[test]
    fn canonical_deserialization_returns_the_existing_canonical_instance() {
        let key = "canonical_deserialization_returns_the_existing_canonical_instance";
        let original = Tag::intern_registry()
            .intern(build_tag(key, "original"))
            .into_canonical();
        let payload = serde_json::json!({"name": key, "note": "original"});

        let deserialized: Canonical<Tag> =
            serde_json::from_value(payload).expect("payload deserializes");

        assert_eq!(deserialized, original);
    }

    /// Test deserializing a registered key with a payload unequal to the
    /// canonical instance fails, naming the type and key, and leaves the
    /// canonical instance registered.
    #[test]
    fn canonical_deserialization_rejects_a_payload_unequal_to_the_canonical_instance() {
        let key = "canonical_deserialization_rejects_a_payload_unequal_to_the_canonical_instance";
        let original = Tag::intern_registry()
            .intern(build_tag(key, "original"))
            .into_canonical();
        let payload = serde_json::json!({"name": key, "note": "conflicting"});

        let result: Result<Canonical<Tag>, _> = serde_json::from_value(payload);

        let Err(error) = result else {
            panic!("expected a deserialization error for a conflicting payload");
        };
        let message = error.to_string();
        assert!(message.contains("conflicts"), "got {message}");
        assert!(
            message.contains(std::any::type_name::<Tag>()),
            "got {message}"
        );
        assert!(message.contains(&format!("{key:?}")), "got {message}");
        assert_eq!(Tag::intern_registry().get(key), Some(original));
    }

    /// Test deserializing a registered key with a payload that differs only in
    /// a field `Eq` ignores returns the canonical instance unchanged.
    #[test]
    fn canonical_deserialization_accepts_a_payload_differing_only_in_ignored_metadata() {
        let key = "canonical_deserialization_accepts_a_payload_differing_only_in_ignored_metadata";
        let original = LabeledTag::intern_registry()
            .intern(LabeledTag {
                name: key.to_string(),
                label: "original".to_string(),
            })
            .into_canonical();
        let payload = serde_json::json!({"name": key, "label": "payload"});

        let deserialized: Canonical<LabeledTag> =
            serde_json::from_value(payload).expect("payload deserializes");

        assert_eq!(deserialized, original);
        assert_eq!(deserialized.label, "original");
    }

    /// Test deserializing a payload missing a required field fails without
    /// registering anything.
    #[test]
    fn canonical_deserialization_rejects_a_payload_missing_a_field() {
        let missing_field_key = "canonical_deserialization_rejects_a_payload_missing_a_field";
        let missing_field_payload = serde_json::json!({"name": missing_field_key});

        let missing_field_result: Result<Canonical<Tag>, _> =
            serde_json::from_value(missing_field_payload);
        let Err(missing_field_error) = missing_field_result else {
            panic!("expected a deserialization error for a missing field");
        };
        assert!(
            missing_field_error.to_string().contains("missing field"),
            "got {missing_field_error}"
        );
        assert_eq!(Tag::intern_registry().get(missing_field_key), None);
    }

    /// Test deserializing a payload with an unknown field fails without
    /// registering anything.
    #[test]
    fn canonical_deserialization_rejects_a_payload_with_an_unknown_field() {
        let unknown_field_key = "canonical_deserialization_rejects_a_payload_with_an_unknown_field";
        let unknown_field_payload =
            serde_json::json!({"name": unknown_field_key, "note": "note", "extra": "field"});

        let unknown_field_result: Result<Canonical<Tag>, _> =
            serde_json::from_value(unknown_field_payload);
        let Err(unknown_field_error) = unknown_field_result else {
            panic!("expected a deserialization error for an unknown field");
        };
        assert!(
            unknown_field_error.to_string().contains("unknown field"),
            "got {unknown_field_error}"
        );
        assert_eq!(Tag::intern_registry().get(unknown_field_key), None);
    }

    /// Test errors for one missing key compare equal and errors for
    /// different missing keys do not.
    #[test]
    fn require_errors_compare_by_missing_key() {
        let registry = InternRegistry::<Tag>::new();

        let Err(first) = registry.require("missing") else {
            panic!("expected an error for a missing key");
        };
        let Err(second) = registry.require("missing") else {
            panic!("expected an error for a missing key");
        };
        let Err(other) = registry.require("absent") else {
            panic!("expected an error for a missing key");
        };

        assert_eq!(first, second);
        assert_ne!(first, other);
    }

    /// Test a registry's `Debug` output names the registry type.
    #[test]
    fn registry_debug_names_the_registry_type() {
        let registry = InternRegistry::<Tag>::new();

        let debug_text = format!("{registry:?}");

        assert!(debug_text.contains("InternRegistry"), "got {debug_text}");
    }

    /// The public types stay usable from multiple threads.
    const _: () = {
        assert_send_sync::<Canonical<Tag>>();
        assert_send_sync::<InternRegistry<Tag>>();
        assert_send_sync::<InternOutcome<Tag>>();
        assert_send_sync::<NotInternedError<String>>();
    };

    /// Test concurrent interning of one key yields exactly one canonical
    /// instance, shared by every caller.
    #[test]
    fn intern_yields_one_canonical_instance_under_concurrent_interning() {
        let registry = InternRegistry::<Tag>::new();
        let registry_ref = &registry;

        let outcomes: Vec<InternOutcome<Tag>> = thread::scope(|scope| {
            let handles: Vec<_> = (0..16)
                .map(|index| {
                    scope.spawn(move || {
                        registry_ref.intern(build_tag("shared", &format!("note-{index}")))
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .unwrap_or_else(|payload| panic::resume_unwind(payload))
                })
                .collect()
        });

        let registered_count = outcomes
            .iter()
            .filter(|outcome| outcome.is_registered())
            .count();
        assert_eq!(
            registered_count, 1,
            "expected exactly one Registered outcome"
        );

        let canonical_handles: Vec<Canonical<Tag>> = outcomes
            .into_iter()
            .map(InternOutcome::into_canonical)
            .collect();
        let first = &canonical_handles[0];
        assert!(
            canonical_handles.iter().all(|handle| handle == first),
            "not all canonical handles were equal"
        );
    }

    /// Test `get` never returns a different handle for a key it has already
    /// observed, while one writer interns new keys concurrently with
    /// several readers.
    #[test]
    fn get_never_loses_a_registered_key_under_concurrent_interning() {
        const KEY_COUNT: usize = 200;
        const READER_ITERATIONS: usize = 5000;

        let registry = InternRegistry::<Tag>::new();
        let registry_ref = &registry;
        let progress = AtomicUsize::new(0);
        let progress_ref = &progress;

        thread::scope(|scope| {
            let writer_handle = scope.spawn(move || {
                for index in 0..KEY_COUNT {
                    let _outcome = registry_ref.intern(build_tag(&format!("k{index}"), "writer"));
                    progress_ref.store(index + 1, Ordering::Release);
                }
            });

            let reader_handles: Vec<_> = (0..4)
                .map(|reader_index| {
                    scope.spawn(move || {
                        let mut observed: HashMap<usize, Canonical<Tag>> = HashMap::new();
                        for step in 0..READER_ITERATIONS {
                            let seen = progress_ref.load(Ordering::Acquire);
                            if seen == 0 {
                                continue;
                            }
                            let key_index = (step + reader_index) % seen;
                            let key = format!("k{key_index}");
                            let handle = registry_ref.get(key.as_str()).unwrap_or_else(|| {
                                panic!("key {key} must be registered once progress observed it")
                            });
                            if let Some(previous) = observed.get(&key_index) {
                                assert_eq!(
                                    *previous, handle,
                                    "reader {reader_index} saw a different handle for {key} \
                                     on step {step}"
                                );
                            } else {
                                observed.insert(key_index, handle);
                            }
                        }
                    })
                })
                .collect();

            writer_handle
                .join()
                .unwrap_or_else(|payload| panic::resume_unwind(payload));
            for handle in reader_handles {
                handle
                    .join()
                    .unwrap_or_else(|payload| panic::resume_unwind(payload));
            }
        });
    }

    /// Test the registry keeps working after a key's `Hash` implementation
    /// panics while the registry's lock is held.
    #[test]
    fn registry_keeps_working_after_a_key_hash_panics() {
        let registry = InternRegistry::<TripwireTag>::new();
        let tripwire = TripwireTag {
            key: TripwireKey("tripwire".to_string()),
        };

        let result = panic::catch_unwind(AssertUnwindSafe(|| registry.intern(tripwire)));

        let Err(panic_payload) = result else {
            panic!("expected intern to panic while hashing the tripwire key");
        };
        let panic_message = panic_payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic_payload.downcast_ref::<&str>().copied())
            .unwrap_or_default();
        assert_eq!(panic_message, "tripwire key hashed");

        let outcome = registry.intern(TripwireTag {
            key: TripwireKey("ok".to_string()),
        });
        assert!(outcome.is_registered());
        let canonical = registry
            .get(&TripwireKey("ok".to_string()))
            .expect("the registry must still serve lookups after recovering from the poison");
        assert_eq!(canonical.key, TripwireKey("ok".to_string()));
    }
}
