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
//! `u64::MAX - 1`, and the counter never wraps to reissue a live id. Once
//! that id is issued or restored, the counter holds `u64::MAX` and
//! constructing another identifier panics. No identifier ever holds
//! `u64::MAX`, so a serialized payload carrying it is rejected as invalid.

use std::error::Error;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::de::{self, Deserializer, MapAccess, Unexpected, Visitor};
use serde::ser::{SerializeStruct, Serializer};
use serde::{Deserialize, Serialize};

use crate::decode::{self, Decode};

/// The process-global, monotonically-increasing id counter.
static NEXT_ID: AtomicU64 = AtomicU64::new(0);

/// Error for an id counter that cannot advance without wrapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct IdSpaceExhausted;

impl fmt::Display for IdSpaceExhausted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("identifier id space exhausted")
    }
}

impl Error for IdSpaceExhausted {}

/// Process-globally unique, named compiler symbol.
///
/// Cloning an `Identifier` is cheap: `name_hint` is stored behind an
/// [`Arc<str>`] so clones share the underlying string.
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
    /// Panics if the counter has reached `u64::MAX`, which only restoring
    /// or deserializing the id `u64::MAX - 1`, or advancing the counter past
    /// it, can cause.
    #[must_use]
    pub fn new(name_hint: &str) -> Self {
        let id = Self::next_id(name_hint);
        Self {
            id,
            name_hint: Arc::from(name_hint),
        }
    }

    /// Restore an identifier with a specific id and name hint, advancing the
    /// global counter so `id` is never re-issued to a later construction.
    ///
    /// This is the deserialization path: it ignores any
    /// deterministic-identifier scope and always consults the real global
    /// counter.
    ///
    /// # Panics
    ///
    /// Panics if `id` is `u64::MAX`, which no identifier ever holds and the
    /// counter cannot advance past. Deserializing through serde rejects that
    /// id with an error instead.
    #[must_use]
    pub fn restore(id: u64, name_hint: String) -> Self {
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
/// Every fresh id comes from here; a restored id is taken from its payload
/// instead.
///
/// # Panics
///
/// Panics if the counter has reached `u64::MAX`.
#[must_use]
pub(crate) fn allocate_id() -> u64 {
    try_allocate_id().unwrap_or_else(|exhausted| panic!("{exhausted}"))
}

/// Draw the next id from the process-global counter, leaving the counter
/// unchanged when it cannot advance.
///
/// Serves callers that store ids themselves, such as language bindings: it
/// draws from the same counter as [`Identifier::new`], but ignores any
/// deterministic-identifier scope.
///
/// # Errors
///
/// Returns [`IdSpaceExhausted`] if the counter has reached `u64::MAX`.
pub fn try_allocate_id() -> Result<u64, IdSpaceExhausted> {
    take_next_id(&NEXT_ID)
}

/// Advance the process-global counter so `id` is never issued, leaving it
/// unchanged when it is already past `id`.
///
/// # Panics
///
/// Panics if `id` is `u64::MAX`, since the counter cannot advance past it.
pub(crate) fn advance_counter_past(id: u64) {
    try_advance_counter_past(id).unwrap_or_else(|exhausted| panic!("{exhausted}"));
}

/// Advance the process-global counter so `id` is never issued, leaving it
/// unchanged when it is already past `id`.
///
/// Serves callers that store ids themselves, such as language bindings: it
/// advances the same counter [`Identifier::new`] draws from.
///
/// # Errors
///
/// Returns [`IdSpaceExhausted`], leaving the counter unchanged, if `id` is
/// `u64::MAX`, since the counter cannot advance past it.
pub fn try_advance_counter_past(id: u64) -> Result<(), IdSpaceExhausted> {
    advance_past(&NEXT_ID, id)
}

/// Return `counter`'s current value and advance it by one.
///
/// # Errors
///
/// Returns [`IdSpaceExhausted`], leaving `counter` unchanged, instead of
/// wrapping when `counter` is at `u64::MAX`, since a wrapped counter would
/// re-issue live ids.
fn take_next_id(counter: &AtomicU64) -> Result<u64, IdSpaceExhausted> {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next_id| {
            next_id.checked_add(1)
        })
        .map_err(|_exhausted_value| IdSpaceExhausted)
}

/// Advance `counter` to at least `id + 1`, leaving it unchanged when it is
/// already past `id`.
///
/// # Errors
///
/// Returns [`IdSpaceExhausted`], leaving `counter` unchanged, if `id` is
/// `u64::MAX`, since `counter` cannot advance past it.
fn advance_past(counter: &AtomicU64, id: u64) -> Result<(), IdSpaceExhausted> {
    let floor = id.checked_add(1).ok_or(IdSpaceExhausted)?;
    counter.fetch_max(floor, Ordering::Relaxed);
    Ok(())
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

/// A decoded identifier payload whose id has not been restored yet.
///
/// Decoding one checks the payload's fields and rejects the id `u64::MAX`
/// without touching the global counter, so a caller can check a whole
/// payload before any id in it advances the counter.
pub(crate) struct IdentifierPayload {
    id: u64,
    name_hint: String,
}

impl IdentifierPayload {
    /// Restore the identifier, advancing the global counter past its id.
    pub(crate) fn restore(self) -> Identifier {
        Identifier::restore(self.id, self.name_hint)
    }
}

impl<'de> Deserialize<'de> for IdentifierPayload {
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
            type Value = IdentifierPayload;

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
                if id == u64::MAX {
                    return Err(serde::de::Error::invalid_value(
                        Unexpected::Unsigned(id),
                        &"an id below u64::MAX",
                    ));
                }
                Ok(IdentifierPayload { id, name_hint })
            }
        }

        const FIELDS: &[&str] = &["id", "name_hint"];
        deserializer.deserialize_struct("Identifier", FIELDS, IdentifierVisitor)
    }
}

impl Decode for Identifier {
    type Payload = IdentifierPayload;

    fn build_from_payload<E: de::Error>(payload: Self::Payload) -> Result<Self, E> {
        Ok(payload.restore())
    }
}

impl<'de> Deserialize<'de> for Identifier {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        decode::deserialize_via_payload(deserializer)
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
    use std::cell::RefCell;
    use std::collections::HashSet;
    use std::thread;

    use proptest::prelude::*;
    use rstest::rstest;

    use crate::test_support::{
        assert_isolated_test_passes, assert_send_sync, compute_hash, is_isolated_run,
    };

    #[test]
    fn new_identifiers_get_increasing_ids() {
        let a = Identifier::new("a");
        let b = Identifier::new("b");
        assert!(b.id() > a.id());
    }

    #[test]
    fn equality_and_hash_ignore_name_hint() {
        let id_value = Identifier::new("equality-and-hash-anchor").id();
        let a = Identifier::restore(id_value, "a".to_string());
        let b = Identifier::restore(id_value, "b".to_string());
        assert_eq!(a, b);

        assert_eq!(compute_hash(&a), compute_hash(&b));
    }

    /// Test that a new identifier keeps its name hint as given, ASCII or not.
    #[rstest]
    #[case::ascii("test_name")]
    #[case::latin("é")]
    #[case::astral("\u{1d465}")]
    #[case::cjk("名前")]
    fn new_keeps_the_name_hint_as_given(#[case] name_hint: &str) {
        let identifier = Identifier::new(name_hint);

        assert_eq!(identifier.name_hint(), name_hint);
    }

    /// Test that Display writes the name hint verbatim.
    #[rstest]
    #[case::plain("my_name")]
    #[case::empty("")]
    #[case::double_colon("foo::bar")]
    fn display_returns_name_hint(#[case] name_hint: &str) {
        let id_value = Identifier::new("display-anchor").id();
        let identifier = Identifier::restore(id_value, name_hint.to_string());

        assert_eq!(format!("{identifier}"), name_hint);
    }

    /// Test that Debug writes the name hint and the id joined by "::".
    #[rstest]
    #[case::plain("my_name")]
    #[case::empty("")]
    #[case::double_colon("foo::bar")]
    fn debug_returns_name_hint_and_id(#[case] name_hint: &str) {
        let id_value = Identifier::new("debug-anchor").id();
        let identifier = Identifier::restore(id_value, name_hint.to_string());

        assert_eq!(
            format!("{identifier:?}"),
            format!("{name_hint}::{id_value}")
        );
    }

    #[test]
    fn restore_advances_counter_past_a_future_id() {
        let far_future_id = Identifier::new("deserialize-advances-counter-anchor").id() + 1_000_000;
        let restored = Identifier::restore(far_future_id, "restored".to_string());
        assert_eq!(restored.id(), far_future_id);

        let next = Identifier::new("next");
        assert!(next.id() > far_future_id);
    }

    #[test]
    fn restore_is_a_no_op_when_counter_already_ahead() {
        let first = Identifier::new("first");
        let stale_id = first.id();
        let _stale = Identifier::restore(stale_id, "stale".to_string());
        let next = Identifier::new("next");
        assert!(next.id() > stale_id);
    }

    /// Test that serde writes the id and name hint and reads both back.
    #[rstest]
    #[case::plain("roundtrip")]
    #[case::empty("")]
    #[case::double_colon("foo::bar")]
    #[case::non_ascii("名前")]
    fn serde_round_trip_preserves_id_and_name_hint(#[case] name_hint: &str) {
        let id_value = Identifier::new("serde-roundtrip-anchor").id();
        let original = Identifier::restore(id_value, name_hint.to_string());
        let json = serde_json::to_string(&original).unwrap();
        assert!(json.contains(&format!("\"id\":{id_value}")), "{json}");
        assert!(
            json.contains(&format!("\"name_hint\":\"{name_hint}\"")),
            "{json}"
        );

        let restored: Identifier = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.id(), id_value);
        assert_eq!(restored.name_hint(), name_hint);
    }

    /// Test that serde accepts the smallest id, zero.
    #[test]
    fn serde_accepts_the_zero_id() {
        let restored: Identifier =
            serde_json::from_str("{\"id\":0,\"name_hint\":\"x\"}").expect("0 is a valid id");

        assert_eq!(restored.id(), 0);
        assert_eq!(restored.name_hint(), "x");
    }

    /// Test that serde rejects a payload whose fields are missing, unknown or
    /// of the wrong type.
    #[rstest]
    #[case::missing_id("{\"name_hint\":\"x\"}", "missing field `id`")]
    #[case::missing_name_hint("{\"id\":0}", "missing field `name_hint`")]
    #[case::string_id("{\"id\":\"not_an_int\",\"name_hint\":\"x\"}", "invalid type: string")]
    #[case::true_id("{\"id\":true,\"name_hint\":\"x\"}", "invalid type: boolean `true`")]
    #[case::false_id("{\"id\":false,\"name_hint\":\"x\"}", "invalid type: boolean `false`")]
    #[case::negative_id("{\"id\":-1,\"name_hint\":\"x\"}", "invalid value: integer `-1`")]
    #[case::integer_name_hint("{\"id\":0,\"name_hint\":123}", "invalid type: integer `123`")]
    #[case::extra_key("{\"id\":0,\"name_hint\":\"x\",\"extra\":1}", "unknown field `extra`")]
    #[case::typo_key(
        "{\"id\":0,\"name_hint\":\"x\",\"name_hit\":\"typo\"}",
        "unknown field `name_hit`"
    )]
    fn serde_rejects_a_malformed_payload(#[case] json: &str, #[case] expected_message: &str) {
        let error = serde_json::from_str::<Identifier>(json).expect_err("the payload is malformed");

        assert!(
            error.to_string().contains(expected_message),
            "unexpected error: {error}"
        );
    }

    proptest! {
        /// Test that serde round-trips an identifier whatever its name hint.
        #[test]
        fn serde_round_trip_preserves_any_name_hint(name_hint in any::<String>()) {
            let original = Identifier::new(&name_hint);

            let json = serde_json::to_string(&original).unwrap();
            let restored: Identifier = serde_json::from_str(&json).unwrap();

            prop_assert_eq!(restored.id(), original.id());
            prop_assert_eq!(restored.name_hint(), name_hint.as_str());
        }
    }

    #[test]
    fn id_space_exhausted_displays_the_exhaustion_message() {
        assert_eq!(
            IdSpaceExhausted.to_string(),
            "identifier id space exhausted"
        );
    }

    #[test]
    fn take_next_id_issues_the_last_id_below_u64_max() {
        let counter = AtomicU64::new(u64::MAX - 1);
        assert_eq!(take_next_id(&counter), Ok(u64::MAX - 1));
        assert_eq!(counter.load(Ordering::Relaxed), u64::MAX);
    }

    #[test]
    fn take_next_id_refuses_to_wrap_at_u64_max() {
        let counter = AtomicU64::new(u64::MAX);
        assert_eq!(take_next_id(&counter), Err(IdSpaceExhausted));
        assert_eq!(counter.load(Ordering::Relaxed), u64::MAX);
    }

    #[test]
    fn advance_past_raises_the_counter_to_one_past_the_id() {
        let counter = AtomicU64::new(0);
        assert_eq!(advance_past(&counter, u64::MAX - 1), Ok(()));
        assert_eq!(counter.load(Ordering::Relaxed), u64::MAX);
    }

    #[test]
    fn advance_past_refuses_to_wrap_at_u64_max() {
        let counter = AtomicU64::new(7);
        assert_eq!(advance_past(&counter, u64::MAX), Err(IdSpaceExhausted));
        assert_eq!(counter.load(Ordering::Relaxed), 7);
    }

    /// Reference id counter: a plain next id that allocation hands out and
    /// increments, and that an advance raises to one past the given id.
    struct ReferenceCounter {
        next_id: u64,
    }

    impl ReferenceCounter {
        fn allocate(&mut self) -> u64 {
            let id = self.next_id;
            self.next_id += 1;
            id
        }

        fn advance_past(&mut self, id: u64) {
            self.next_id = self.next_id.max(id + 1);
        }
    }

    /// Run `operations` on a counter after allocating an anchor id: `None`
    /// allocates, `Some(offset)` advances past the id `offset` past the
    /// anchor. Allocate once more at the end, and return every id allocated
    /// after the anchor, minus the anchor.
    fn run_counter_operations(
        operations: &[Option<u64>],
        mut allocate: impl FnMut() -> u64,
        mut advance_past: impl FnMut(u64),
    ) -> Vec<u64> {
        let base = allocate();
        let mut relative_ids = Vec::new();
        for operation in operations {
            match operation {
                None => relative_ids.push(allocate() - base),
                Some(offset) => advance_past(base + offset),
            }
        }
        relative_ids.push(allocate() - base);
        relative_ids
    }

    proptest! {
        /// Test that the id counter, from any starting value, issues the same
        /// ids relative to an anchor as the reference counter started at zero,
        /// for any sequence of allocations and advances.
        #[test]
        fn counter_issues_the_reference_counters_relative_ids_for_any_operation_sequence(
            start in 0..u64::MAX - 1024,
            operations in prop::collection::vec(prop::option::of(0..=500_u64), 0..=20),
        ) {
            let counter = AtomicU64::new(start);
            let reference = RefCell::new(ReferenceCounter { next_id: 0 });

            let counter_ids = run_counter_operations(
                &operations,
                || take_next_id(&counter).expect("the counter stays below u64::MAX"),
                |id| advance_past(&counter, id).expect("the id stays below u64::MAX"),
            );
            let reference_ids = run_counter_operations(
                &operations,
                || reference.borrow_mut().allocate(),
                |id| reference.borrow_mut().advance_past(id),
            );

            prop_assert_eq!(counter_ids, reference_ids);
        }
    }

    #[test]
    #[should_panic(expected = "identifier id space exhausted")]
    fn restoring_u64_max_panics() {
        let _max = Identifier::restore(u64::MAX, "max".to_string());
    }

    #[test]
    fn try_advance_counter_past_u64_max_returns_an_error() {
        assert_eq!(try_advance_counter_past(u64::MAX), Err(IdSpaceExhausted));
    }

    #[test]
    fn try_allocate_id_shares_the_counter_with_construction() {
        let allocated = try_allocate_id();
        let constructed = Identifier::new("after-try-allocate");
        assert!(allocated.is_ok_and(|id| id < constructed.id()));
    }

    /// Test that serde rejects an id of `u64::MAX`, which no identifier ever
    /// holds, and any id too large for a `u64`.
    #[rstest]
    #[case::two_pow_64_minus_one("18446744073709551615", "an id below u64::MAX")]
    #[case::two_pow_64("18446744073709551616", "expected u64")]
    #[case::just_above_two_pow_64("18446744073709551617", "expected u64")]
    #[case::two_pow_200(
        "1606938044258990275541962092341162602522202993782792835301376",
        "expected u64"
    )]
    fn serde_rejects_an_id_of_u64_max_or_more(#[case] id: &str, #[case] expected_message: &str) {
        let json = format!("{{\"id\":{id},\"name_hint\":\"x\"}}");

        let error = serde_json::from_str::<Identifier>(&json)
            .expect_err("no identifier ever holds u64::MAX or more");

        assert!(
            error.to_string().contains(expected_message),
            "unexpected error: {error}"
        );
    }

    /// Deserialize the largest issuable id, then check the counter is
    /// exhausted. Only meaningful in the child process that
    /// [`serde_accepts_the_largest_issuable_id`] starts.
    #[test]
    #[ignore = "exhausts the process-global counter; run through assert_isolated_test_passes"]
    fn serde_accepts_the_largest_issuable_id_in_isolation() {
        if !is_isolated_run() {
            return;
        }
        let json = format!("{{\"id\":{},\"name_hint\":\"largest\"}}", u64::MAX - 1);

        let restored: Identifier = serde_json::from_str(&json).expect("u64::MAX - 1 is valid");

        assert_eq!(restored.id(), u64::MAX - 1);
        assert_eq!(try_allocate_id(), Err(IdSpaceExhausted));
    }

    #[test]
    fn serde_accepts_the_largest_issuable_id() {
        assert_isolated_test_passes(
            "identifier::tests::serde_accepts_the_largest_issuable_id_in_isolation",
        );
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
    /// Deserialize targets are spaced `CONSTRUCTION_BUDGET` apart, far more
    /// ids than this test or any concurrent test could consume while it
    /// runs, so construction cannot reach a target before its deserialize
    /// fires.
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
                        Identifier::restore(id, "concurrent-deserialize".to_string()).id()
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
                        assert_eq!(advance_past(counter_ref, thread_index * 500 + step), Ok(()));
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

    /// The public types stay usable from multiple threads. The tag types are
    /// covered by `Interned`'s own `Send + Sync` bounds.
    const _: () = {
        assert_send_sync::<Identifier>();
        assert_send_sync::<IdSpaceExhausted>();
    };
}
