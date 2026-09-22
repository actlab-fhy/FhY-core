//! Process-globally unique, named compiler symbol.
//!
//! Two [`Identifier`] instances are equal iff they share the same `id`;
//! `name_hint` is a debugging aid and is not consulted by equality or
//! hashing. Ids are drawn from a single process-global, monotonically
//! increasing counter and are never reused.
//!
//! Construction and deserialization share the same counter: a deserialized
//! id can never collide with a subsequently constructed id, regardless of
//! interleaving across threads. Deserializing an id greater than or equal to
//! the next-to-be-issued value advances the counter past it.
//!
//! Ids are `u64`s, so the largest id this module ever issues is
//! `u64::MAX - 1`. Deserializing `u64::MAX`, or constructing an identifier
//! once the counter has reached `u64::MAX`, panics instead of wrapping and
//! reissuing a live id.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use serde::de::{Deserializer, MapAccess, Visitor};
use serde::ser::{SerializeStruct, Serializer};
use serde::{Deserialize, Serialize};

/// The process-global, monotonically-increasing id counter.
static NEXT_ID: AtomicU64 = AtomicU64::new(0);

/// Panic message for an id counter that cannot advance without wrapping.
const ID_SPACE_EXHAUSTED: &str = "identifier id space exhausted";

/// Process-globally unique, named compiler symbol.
///
/// Cloning an `Identifier` is cheap: `name_hint` is stored behind an
/// [`Arc<str>`] so clones share the underlying string.
///
/// Ids are `u64`s; the largest id an `Identifier` ever holds is
/// `u64::MAX - 1`. Reaching `u64::MAX`, whether by construction or by
/// deserializing it directly, panics instead of wrapping.
#[derive(Clone)]
pub struct Identifier {
    id: u64,
    name_hint: Arc<str>,
}

impl Identifier {
    /// Construct a new identifier, drawing the next id from the global
    /// counter.
    ///
    /// # Panics
    ///
    /// Panics if the counter has reached `u64::MAX`, which only
    /// deserializing an id near `u64::MAX` can cause.
    #[must_use]
    pub fn new(name_hint: &str) -> Self {
        let id = Self::next_id(name_hint);
        Self {
            id,
            name_hint: Arc::from(name_hint),
        }
    }

    /// Reconstruct an identifier with a specific id and name hint, advancing
    /// the global counter so `id` is never re-issued to a later
    /// construction.
    ///
    /// This is the deserialization path: it ignores any
    /// deterministic-identifier scope and always consults the real global
    /// counter.
    ///
    /// # Panics
    ///
    /// Panics if `id` is `u64::MAX`, since the counter cannot advance past it.
    #[must_use]
    pub fn deserialize(id: u64, name_hint: String) -> Self {
        advance_counter_past(id);
        Self {
            id,
            name_hint: Arc::from(name_hint),
        }
    }

    /// Construct an identifier whose id always comes from the global
    /// counter, even inside a deterministic-identifier scope.
    ///
    /// For identifiers held by this crate's shipped statics, which are created
    /// on first use: a scope must never hand a test identifier the id of a
    /// shipped constant.
    #[must_use]
    pub(crate) fn new_unscoped(name_hint: &str) -> Self {
        Self {
            id: allocate_id(),
            name_hint: Arc::from(name_hint),
        }
    }

    /// Return the identifier's unique id.
    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Return the identifier's name hint.
    #[must_use]
    pub fn name_hint(&self) -> &str {
        &self.name_hint
    }

    /// Return the id `name_hint` receives inside the current thread's
    /// deterministic-identifier scope, or the next id from the global counter
    /// outside one.
    #[cfg(any(test, feature = "testing"))]
    fn next_id(name_hint: &str) -> u64 {
        crate::testing::find_scoped_id(name_hint).unwrap_or_else(allocate_id)
    }

    /// Return the next id from the global counter.
    #[cfg(not(any(test, feature = "testing")))]
    fn next_id(_name_hint: &str) -> u64 {
        allocate_id()
    }
}

/// Draw the next id from the process-global counter.
///
/// Every id an [`Identifier`] is constructed with comes from here.
///
/// # Panics
///
/// Panics if the counter has reached `u64::MAX`.
#[must_use]
pub(crate) fn allocate_id() -> u64 {
    take_next_id(&NEXT_ID)
}

/// Advance the process-global counter so `id` is never issued, leaving it
/// unchanged when it is already past `id`.
///
/// # Panics
///
/// Panics if `id` is `u64::MAX`, since the counter cannot advance past it.
pub(crate) fn advance_counter_past(id: u64) {
    advance_past(&NEXT_ID, id);
}

/// Return `counter`'s current value and advance it by one.
///
/// # Panics
///
/// Panics instead of wrapping when `counter` is at `u64::MAX`, since a
/// wrapped counter would re-issue live ids.
fn take_next_id(counter: &AtomicU64) -> u64 {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next_id| {
            next_id.checked_add(1)
        })
        .unwrap_or_else(|_| panic!("{ID_SPACE_EXHAUSTED}"))
}

/// Advance `counter` to at least `id + 1`, leaving it unchanged when it is
/// already past `id`.
///
/// # Panics
///
/// Panics if `id` is `u64::MAX`, since `counter` cannot advance past it.
fn advance_past(counter: &AtomicU64, id: u64) {
    let floor = id.checked_add(1).expect(ID_SPACE_EXHAUSTED);
    counter.fetch_max(floor, Ordering::Relaxed);
}

impl PartialEq for Identifier {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Identifier {}

impl std::hash::Hash for Identifier {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl fmt::Display for Identifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name_hint)
    }
}

impl fmt::Debug for Identifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}::{}", self.name_hint, self.id)
    }
}

impl Serialize for Identifier {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("Identifier", 2)?;
        state.serialize_field("id", &self.id)?;
        state.serialize_field("name_hint", &*self.name_hint)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for Identifier {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field {
            Id,
            NameHint,
        }

        struct IdentifierVisitor;

        impl<'de> Visitor<'de> for IdentifierVisitor {
            type Value = Identifier;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a map with an `id` and a `name_hint`")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut id: Option<u64> = None;
                let mut name_hint: Option<String> = None;
                while let Some(key) = map.next_key::<Field>()? {
                    match key {
                        Field::Id => {
                            if id.is_some() {
                                return Err(serde::de::Error::duplicate_field("id"));
                            }
                            id = Some(map.next_value()?);
                        }
                        Field::NameHint => {
                            if name_hint.is_some() {
                                return Err(serde::de::Error::duplicate_field("name_hint"));
                            }
                            name_hint = Some(map.next_value()?);
                        }
                    }
                }
                let id = id.ok_or_else(|| serde::de::Error::missing_field("id"))?;
                let name_hint =
                    name_hint.ok_or_else(|| serde::de::Error::missing_field("name_hint"))?;
                // Deserialization is a correctness-critical seam: it must
                // always advance the global counter past `id`, regardless of
                // the active testing strategy, so a deserialized id can
                // never collide with a subsequently constructed one.
                Ok(Identifier::deserialize(id, name_hint))
            }
        }

        const FIELDS: &[&str] = &["id", "name_hint"];
        deserializer.deserialize_struct("Identifier", FIELDS, IdentifierVisitor)
    }
}

/// Types that expose a stable [`Identifier`].
pub trait HasIdentifier {
    /// Return the type's stable identifier.
    #[must_use]
    fn identifier(&self) -> &Identifier;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::thread;

    #[test]
    fn new_identifiers_get_increasing_ids() {
        let a = Identifier::new("a");
        let b = Identifier::new("b");
        assert!(b.id() > a.id());
    }

    #[test]
    fn equality_and_hash_ignore_name_hint() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let id_value = Identifier::new("equality-and-hash-anchor").id();
        let a = Identifier::deserialize(id_value, "a".to_string());
        let b = Identifier::deserialize(id_value, "b".to_string());
        assert_eq!(a, b);

        let mut hasher_a = DefaultHasher::new();
        a.hash(&mut hasher_a);
        let mut hasher_b = DefaultHasher::new();
        b.hash(&mut hasher_b);
        assert_eq!(hasher_a.finish(), hasher_b.finish());
    }

    #[test]
    fn display_returns_name_hint() {
        let id_value = Identifier::new("display-anchor").id();
        let id = Identifier::deserialize(id_value, "my_name".to_string());
        assert_eq!(format!("{id}"), "my_name");
    }

    #[test]
    fn debug_returns_name_hint_and_id() {
        let id_value = Identifier::new("debug-anchor").id();
        let id = Identifier::deserialize(id_value, "my_name".to_string());
        assert_eq!(format!("{id:?}"), format!("my_name::{id_value}"));
    }

    /// Test that Display and Debug handle an empty name hint.
    #[test]
    fn display_and_debug_handle_an_empty_name_hint() {
        let id_value = Identifier::new("empty-name-hint-anchor").id();
        let identifier = Identifier::deserialize(id_value, String::new());

        assert_eq!(format!("{identifier}"), "");
        assert_eq!(format!("{identifier:?}"), format!("::{id_value}"));
    }

    /// Test that Display and Debug handle a name hint containing "::".
    #[test]
    fn display_and_debug_handle_a_name_hint_containing_a_double_colon() {
        let id_value = Identifier::new("double-colon-anchor").id();
        let identifier = Identifier::deserialize(id_value, "foo::bar".to_string());

        assert_eq!(format!("{identifier}"), "foo::bar");
        assert_eq!(format!("{identifier:?}"), format!("foo::bar::{id_value}"));
    }

    #[test]
    fn deserialize_advances_counter_past_a_future_id() {
        let far_future_id = Identifier::new("deserialize-advances-counter-anchor").id() + 1_000_000;
        let restored = Identifier::deserialize(far_future_id, "restored".to_string());
        assert_eq!(restored.id(), far_future_id);

        let next = Identifier::new("next");
        assert!(next.id() > far_future_id);
    }

    #[test]
    fn deserialize_is_a_no_op_when_counter_already_ahead() {
        let first = Identifier::new("first");
        let stale_id = first.id();
        // Re-deserializing an already-issued id must not rewind the
        // counter.
        let _stale = Identifier::deserialize(stale_id, "stale".to_string());
        let next = Identifier::new("next");
        assert!(next.id() > stale_id);
    }

    #[test]
    fn serde_round_trip_preserves_id_and_name_hint() {
        let id_value = Identifier::new("serde-roundtrip-anchor").id();
        let original = Identifier::deserialize(id_value, "roundtrip".to_string());
        let json = serde_json::to_string(&original).unwrap();
        assert!(json.contains(&format!("\"id\":{id_value}")));
        assert!(json.contains("\"name_hint\":\"roundtrip\""));

        let restored: Identifier = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.id(), id_value);
        assert_eq!(restored.name_hint(), "roundtrip");
    }

    /// Test that serde preserves an empty name hint across a round trip.
    #[test]
    fn serde_round_trip_preserves_an_empty_name_hint() {
        let id_value = Identifier::new("serde-empty-name-hint-anchor").id();
        let original = Identifier::deserialize(id_value, String::new());
        let json = serde_json::to_string(&original).unwrap();

        let restored: Identifier = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.id(), id_value);
        assert_eq!(restored.name_hint(), "");
    }

    /// Test that serde preserves a name hint containing "::" across a round
    /// trip.
    #[test]
    fn serde_round_trip_preserves_a_name_hint_containing_a_double_colon() {
        let id_value = Identifier::new("serde-double-colon-anchor").id();
        let original = Identifier::deserialize(id_value, "foo::bar".to_string());
        let json = serde_json::to_string(&original).unwrap();

        let restored: Identifier = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.id(), id_value);
        assert_eq!(restored.name_hint(), "foo::bar");
    }

    #[test]
    fn take_next_id_issues_the_last_id_below_u64_max() {
        let counter = AtomicU64::new(u64::MAX - 1);
        assert_eq!(take_next_id(&counter), u64::MAX - 1);
        assert_eq!(counter.load(Ordering::Relaxed), u64::MAX);
    }

    #[test]
    #[should_panic(expected = "identifier id space exhausted")]
    fn take_next_id_panics_instead_of_wrapping_at_u64_max() {
        let counter = AtomicU64::new(u64::MAX);
        take_next_id(&counter);
    }

    #[test]
    fn advance_past_raises_the_counter_to_one_past_the_id() {
        let counter = AtomicU64::new(0);
        advance_past(&counter, u64::MAX - 1);
        assert_eq!(counter.load(Ordering::Relaxed), u64::MAX);
    }

    #[test]
    #[should_panic(expected = "identifier id space exhausted")]
    fn advance_past_panics_instead_of_wrapping_at_u64_max() {
        let counter = AtomicU64::new(0);
        advance_past(&counter, u64::MAX);
    }

    #[test]
    #[should_panic(expected = "identifier id space exhausted")]
    fn deserializing_u64_max_panics() {
        let _max = Identifier::deserialize(u64::MAX, "max".to_string());
    }

    #[test]
    fn allocate_id_shares_the_counter_with_construction() {
        let allocated = allocate_id();
        let constructed = Identifier::new("after-allocate");
        assert!(constructed.id() > allocated);
        assert!(allocate_id() > constructed.id());
    }

    #[test]
    fn advance_counter_past_keeps_later_ids_beyond_the_advanced_id() {
        let far_future_id = allocate_id() + 1_000_000;
        advance_counter_past(far_future_id);
        assert!(allocate_id() > far_future_id);
        assert!(Identifier::new("after-advance").id() > far_future_id);
    }

    #[test]
    fn advance_counter_past_is_a_no_op_for_an_issued_id() {
        let issued = allocate_id();
        let latest = allocate_id();
        advance_counter_past(issued);
        assert!(allocate_id() > latest);
    }

    #[test]
    #[should_panic(expected = "identifier id space exhausted")]
    fn advance_counter_past_u64_max_panics() {
        advance_counter_past(u64::MAX);
    }

    #[test]
    fn serde_deserialize_advances_the_counter() {
        let far_future_id = Identifier::new("serde-deserialize-advances-anchor").id() + 1_000_000;
        let json = format!("{{\"id\":{far_future_id},\"name_hint\":\"far\"}}");
        let _restored: Identifier = serde_json::from_str(&json).unwrap();

        let next = Identifier::new("next-after-serde");
        assert!(next.id() > far_future_id);
    }

    /// Test that many threads constructing identifiers concurrently always
    /// receive pairwise-distinct ids.
    #[test]
    fn concurrent_construction_yields_pairwise_distinct_ids() {
        let ids: Vec<u64> = thread::scope(|scope| {
            let handles: Vec<_> = (0..16)
                .map(|_| {
                    scope.spawn(|| {
                        (0..200)
                            .map(|_| Identifier::new("concurrent-construction").id())
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|handle| handle.join().unwrap())
                .collect()
        });

        let unique: HashSet<u64> = ids.iter().copied().collect();
        assert_eq!(unique.len(), ids.len(), "duplicate id among: {ids:?}");
    }

    /// Test that constructing identifiers concurrently with deserializing
    /// ids far ahead of the counter never yields a constructed id equal to
    /// a deserialized one.
    ///
    /// Deserialize targets are spaced `CONSTRUCTION_BUDGET` apart, a bound
    /// deliberately far larger than the number of ids this test's
    /// construction threads (and any other test racing the same global
    /// counter) could plausibly consume while it runs. That keeps every
    /// target unreachable by ordinary sequential construction until its own
    /// deserialize call has already fired, regardless of how the threads
    /// below are scheduled.
    #[test]
    fn concurrent_construction_never_collides_with_ids_deserialized_ahead_of_it() {
        const CONSTRUCTION_BUDGET: u64 = 1_000_000;
        const CONSTRUCTION_THREADS: u64 = 8;
        const CONSTRUCTIONS_PER_THREAD: u64 = 200;
        const DESERIALIZE_THREADS: u64 = 8;

        let baseline = Identifier::new("construct-vs-deserialize-anchor").id();

        let (constructed_ids, deserialized_ids): (Vec<u64>, Vec<u64>) = thread::scope(|scope| {
            let construct_handles: Vec<_> = (0..CONSTRUCTION_THREADS)
                .map(|_| {
                    scope.spawn(|| {
                        (0..CONSTRUCTIONS_PER_THREAD)
                            .map(|_| Identifier::new("concurrent-construction").id())
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            let deserialize_handles: Vec<_> = (0..DESERIALIZE_THREADS)
                .map(|thread_index| {
                    scope.spawn(move || {
                        let id = baseline + CONSTRUCTION_BUDGET * (thread_index + 1);
                        Identifier::deserialize(id, "concurrent-deserialize".to_string()).id()
                    })
                })
                .collect();

            let constructed = construct_handles
                .into_iter()
                .flat_map(|handle| handle.join().unwrap())
                .collect();
            let deserialized = deserialize_handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect();
            (constructed, deserialized)
        });

        let deserialized_ids: HashSet<u64> = deserialized_ids.into_iter().collect();
        assert!(
            constructed_ids
                .iter()
                .all(|id| !deserialized_ids.contains(id)),
            "a constructed id collided with a deserialized one"
        );
    }

    /// Test that concurrent `advance_past` calls never move a counter
    /// backwards, as observed by a single reader thread.
    #[test]
    fn concurrent_advance_past_never_lowers_a_concurrently_observed_counter() {
        let counter = AtomicU64::new(0);
        let counter_ref = &counter;

        let observed = thread::scope(|scope| {
            for thread_index in 0..8u64 {
                scope.spawn(move || {
                    for step in 0..500u64 {
                        advance_past(counter_ref, thread_index * 500 + step);
                    }
                });
            }
            let observer = scope.spawn(move || {
                (0..5000)
                    .map(|_| counter_ref.load(Ordering::Relaxed))
                    .collect::<Vec<_>>()
            });
            observer.join().unwrap()
        });

        assert!(
            observed.windows(2).all(|pair| pair[0] <= pair[1]),
            "counter observed to decrease: {observed:?}"
        );
    }

    /// Minimal type exercising the [`HasIdentifier`] trait.
    struct NamedThing {
        identifier: Identifier,
    }

    impl HasIdentifier for NamedThing {
        fn identifier(&self) -> &Identifier {
            &self.identifier
        }
    }

    #[test]
    fn has_identifier_returns_the_stored_identifier() {
        let identifier = Identifier::new("named-thing");
        let expected_id = identifier.id();
        let thing = NamedThing { identifier };

        assert_eq!(thing.identifier().id(), expected_id);
    }
}
