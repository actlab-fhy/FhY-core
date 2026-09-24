//! Process-globally unique, named compiler symbol.
//!
//! Two [`Identifier`] instances are equal iff they share the same `id`;
//! `name_hint` is a debugging aid and is not consulted by equality or
//! hashing. Ids are drawn from a single process-global, monotonically
//! increasing counter and are never reused.
//!
//! The ids `0..RESERVED_ID_COUNT` are reserved: the identifiers this crate
//! ships (its note kinds, op attributes and value domains) hold fixed ids
//! from that block, the same in every process, and the counter issues fresh
//! ids from [`RESERVED_ID_COUNT`] upward.
//!
//! Construction and deserialization share the same counter: a deserialized
//! id can never collide with a subsequently constructed id, regardless of
//! interleaving across threads. Deserializing an id greater than or equal to
//! the next-to-be-issued value advances the counter past it.
//!
//! Ids are `u64`s. A payload id, one read from a serialized identifier or
//! handed to [`try_advance_counter_past`], must lie below [`ID_CAP`]
//! (`2^63`), so no payload can raise the counter above `ID_CAP` and leave
//! it too few ids: exhausting the counter takes `2^63` fresh identifiers.
//! Fresh ids may exceed the cap. A payload id outside `0..ID_CAP`, however
//! large or negative, is rejected with the range it must lie in, wherever
//! the identifier is nested, and leaves the counter unchanged.

use std::error::Error;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::de::{self, Deserializer, Unexpected, Visitor};
use serde::ser::{SerializeStruct, Serializer};
use serde::{Deserialize, Serialize};

pub(crate) mod reserved;

use reserved::ReservedIdentifier;

/// Number of ids reserved for the identifiers this crate ships.
///
/// Ids `0..RESERVED_ID_COUNT` are the fixed ids of the shipped identifiers,
/// and the counter issues fresh ids from `RESERVED_ID_COUNT` upward.
///
/// Matches the Python implementation: `fhy_core.identifier._RESERVED_ID_COUNT`.
pub const RESERVED_ID_COUNT: u64 = 65_536;

/// Exclusive upper bound of a payload id.
///
/// A deserialized or restored id must be below it, so a payload can raise
/// the counter to at most `ID_CAP`. Fresh ids may exceed it.
///
/// Matches the Python implementation: `fhy_core.identifier._ID_CAP`.
pub const ID_CAP: u64 = 1 << 63;

/// The process-global, monotonically-increasing id counter.
static NEXT_ID: AtomicU64 = AtomicU64::new(RESERVED_ID_COUNT);

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

/// Error for a payload id at or above [`ID_CAP`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct IdOutOfRange {
    id: u64,
}

impl IdOutOfRange {
    /// Return the rejected id.
    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }
}

/// Matches the Python implementation: the `OverflowError` message of
/// `fhy_core.identifier`'s pure-Python counter.
impl fmt::Display for IdOutOfRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "identifier id {} is at or above the cap {ID_CAP}",
            self.id
        )
    }
}

impl Error for IdOutOfRange {}

/// An id checked to lie below [`ID_CAP`], so one past it never overflows.
#[derive(Debug, Clone, Copy)]
struct PayloadId(u64);

impl PayloadId {
    /// Check `id` against the cap.
    ///
    /// # Errors
    ///
    /// Returns [`IdOutOfRange`] if `id` is at or above [`ID_CAP`].
    fn new(id: u64) -> Result<Self, IdOutOfRange> {
        if id < ID_CAP {
            Ok(Self(id))
        } else {
            Err(IdOutOfRange { id })
        }
    }

    /// Return the smallest counter value that never issues this id.
    fn successor(self) -> u64 {
        self.0 + 1
    }
}

/// Process-globally unique, named compiler symbol.
///
/// Cloning an `Identifier` is cheap: `name_hint` is stored behind an
/// [`Arc<str>`] so clones share the underlying string.
///
/// An identifier serializes as `{"id": .., "name_hint": ..}`. Deserializing
/// one reads that shape, rejects an id at or above [`ID_CAP`], and advances
/// the global counter past the id it restores.
#[derive(Clone, Deserialize)]
#[serde(try_from = "IdentifierWire")]
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
    /// Panics if the counter has reached `u64::MAX`. A payload raises the
    /// counter to at most [`ID_CAP`], so that takes `2^63` fresh
    /// identifiers, and no input can cause it. [`try_new`](Self::try_new)
    /// returns an error instead.
    #[must_use]
    pub fn new(name_hint: &str) -> Self {
        Self::try_new(name_hint).unwrap_or_else(|exhausted| panic!("{exhausted}"))
    }

    /// Construct a new identifier, drawing the next id from the global
    /// counter, or fail without drawing one.
    ///
    /// # Errors
    ///
    /// Returns [`IdSpaceExhausted`], leaving the counter unchanged, if the
    /// counter has reached `u64::MAX`.
    pub fn try_new(name_hint: &str) -> Result<Self, IdSpaceExhausted> {
        Ok(Self {
            id: try_allocate_id()?,
            name_hint: Arc::from(name_hint),
        })
    }

    /// Restore an identifier with a payload id and name hint, advancing the
    /// global counter so `id` is never re-issued to a later construction.
    ///
    /// # Errors
    ///
    /// Returns [`IdOutOfRange`], leaving the counter unchanged, if `id` is
    /// at or above [`ID_CAP`].
    pub(crate) fn try_restore(id: u64, name_hint: &str) -> Result<Self, IdOutOfRange> {
        try_advance_counter_past(id)?;
        Ok(Self {
            id,
            name_hint: Arc::from(name_hint),
        })
    }

    /// Return the shipped identifier `entry` names, with its fixed id.
    ///
    /// Draws nothing from the counter, which never issues a reserved id.
    #[must_use]
    pub(crate) fn reserved(entry: ReservedIdentifier) -> Self {
        Self {
            id: entry.id(),
            name_hint: Arc::from(entry.name_hint()),
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
}

/// Draw the next id from the process-global counter, leaving the counter
/// unchanged when it cannot advance.
///
/// For language bindings, which store ids in their own objects: it draws
/// from the same counter as [`Identifier::new`].
///
/// # Errors
///
/// Returns [`IdSpaceExhausted`] if the counter has reached `u64::MAX`.
pub fn try_allocate_id() -> Result<u64, IdSpaceExhausted> {
    take_next_id(&NEXT_ID)
}

/// Advance the process-global counter so the payload id `id` is never
/// issued, leaving it unchanged when it is already past `id`, as it always
/// is for a reserved id.
///
/// For language bindings, which store ids in their own objects: it advances
/// the same counter [`Identifier::new`] draws from.
///
/// # Errors
///
/// Returns [`IdOutOfRange`], leaving the counter unchanged, if `id` is at or
/// above [`ID_CAP`].
pub fn try_advance_counter_past(id: u64) -> Result<(), IdOutOfRange> {
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
/// Returns [`IdOutOfRange`], leaving `counter` unchanged, if `id` is at or
/// above [`ID_CAP`].
fn advance_past(counter: &AtomicU64, id: u64) -> Result<(), IdOutOfRange> {
    let id = PayloadId::new(id)?;
    counter.fetch_max(id.successor(), Ordering::Relaxed);
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

/// Decodes a payload id, checked against [`ID_CAP`].
///
/// It asks the format for a `u64`, so it also reads formats that are not
/// self-describing. An integer outside `0..ID_CAP` is reported as out of
/// range with the range it must lie in, and anything else as the wrong type.
impl<'de> Deserialize<'de> for PayloadId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_u64(PayloadIdVisitor)
    }
}

/// Visitor accepting the integers `0..ID_CAP` as a [`PayloadId`].
struct PayloadIdVisitor;

impl Visitor<'_> for PayloadIdVisitor {
    type Value = PayloadId;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "an id from 0 to {}", ID_CAP - 1)
    }

    fn visit_u64<E: de::Error>(self, value: u64) -> Result<PayloadId, E> {
        PayloadId::new(value).map_err(|out_of_range| {
            E::invalid_value(Unexpected::Unsigned(out_of_range.id()), &self)
        })
    }

    fn visit_i64<E: de::Error>(self, value: i64) -> Result<PayloadId, E> {
        u64::try_from(value)
            .ok()
            .and_then(|id| PayloadId::new(id).ok())
            .ok_or_else(|| E::invalid_value(Unexpected::Signed(value), &self))
    }
}

/// The wire form of an [`Identifier`], decoded but not yet restored.
///
/// Decoding one checks its fields and rejects an id at or above [`ID_CAP`]
/// without touching the global counter; converting it into an
/// [`Identifier`] restores the id.
#[derive(Deserialize)]
#[serde(
    rename = "Identifier",
    expecting = "an identifier",
    deny_unknown_fields
)]
pub(crate) struct IdentifierWire {
    id: PayloadId,
    name_hint: String,
}

impl TryFrom<IdentifierWire> for Identifier {
    type Error = IdOutOfRange;

    /// Restore the identifier, advancing the global counter past its id.
    ///
    /// # Errors
    ///
    /// Never fails in practice: decoding the wire form checked its id
    /// against the cap.
    fn try_from(wire: IdentifierWire) -> Result<Self, IdOutOfRange> {
        Self::try_restore(wire.id.0, &wire.name_hint)
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

    use crate::test_support::{assert_send_sync, compute_hash};

    #[test]
    fn new_identifiers_get_increasing_ids() {
        let a = Identifier::new("a");
        let b = Identifier::new("b");
        assert!(b.id() > a.id());
    }

    #[test]
    fn equality_and_hash_ignore_name_hint() {
        let id_value = Identifier::new("equality-and-hash-anchor").id();
        let a = Identifier::try_restore(id_value, "a").expect("the id is below the cap");
        let b = Identifier::try_restore(id_value, "b").expect("the id is below the cap");
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
        let identifier =
            Identifier::try_restore(id_value, name_hint).expect("the id is below the cap");

        assert_eq!(format!("{identifier}"), name_hint);
    }

    /// Test that Debug writes the name hint and the id joined by "::".
    #[rstest]
    #[case::plain("my_name")]
    #[case::empty("")]
    #[case::double_colon("foo::bar")]
    fn debug_returns_name_hint_and_id(#[case] name_hint: &str) {
        let id_value = Identifier::new("debug-anchor").id();
        let identifier =
            Identifier::try_restore(id_value, name_hint).expect("the id is below the cap");

        assert_eq!(
            format!("{identifier:?}"),
            format!("{name_hint}::{id_value}")
        );
    }

    #[test]
    fn restore_advances_counter_past_a_future_id() {
        let far_future_id = Identifier::new("deserialize-advances-counter-anchor").id() + 1_000_000;
        let restored =
            Identifier::try_restore(far_future_id, "restored").expect("the id is below the cap");
        assert_eq!(restored.id(), far_future_id);

        let next = Identifier::new("next");
        assert!(next.id() > far_future_id);
    }

    #[test]
    fn restore_is_a_no_op_when_counter_already_ahead() {
        let first = Identifier::new("first");
        let stale_id = first.id();
        let _stale = Identifier::try_restore(stale_id, "stale").expect("the id is below the cap");
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
        let original =
            Identifier::try_restore(id_value, name_hint).expect("the id is below the cap");
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
    #[case::negative_id(
        "{\"id\":-1,\"name_hint\":\"x\"}",
        "invalid value: integer `-1`, expected an id from 0 to 9223372036854775807"
    )]
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

    proptest! {
        /// Test that an identifier with any already-issued id and any name
        /// hint round-trips through postcard, a format that is not
        /// self-describing. Only issued ids are decoded, so the counter never
        /// moves.
        #[test]
        fn serde_round_trips_through_postcard_for_any_payload_id(
            raw_id in any::<u64>(),
            name_hint in any::<String>(),
        ) {
            let issued = Identifier::new("postcard-anchor").id();
            let id = raw_id % (issued + 1);
            let original = Identifier::try_restore(id, &name_hint).expect("the id is issued");

            let bytes = postcard::to_allocvec(&original).expect("the identifier encodes");
            let restored: Identifier = postcard::from_bytes(&bytes).expect("the identifier decodes");

            prop_assert_eq!(restored.id(), id);
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
        assert_eq!(advance_past(&counter, ID_CAP - 1), Ok(()));
        assert_eq!(counter.load(Ordering::Relaxed), ID_CAP);
    }

    /// Test advancing past an id at or above the cap fails, naming the id,
    /// and leaves the counter unchanged.
    #[rstest]
    #[case::the_cap(ID_CAP)]
    #[case::u64_max(u64::MAX)]
    fn advance_past_rejects_an_id_at_or_above_the_cap(#[case] id: u64) {
        let counter = AtomicU64::new(7);
        assert_eq!(advance_past(&counter, id), Err(IdOutOfRange { id }));
        assert_eq!(counter.load(Ordering::Relaxed), 7);
    }

    #[test]
    fn id_out_of_range_displays_the_id_and_the_cap() {
        assert_eq!(
            IdOutOfRange { id: ID_CAP }.to_string(),
            "identifier id 9223372036854775808 is at or above the cap 9223372036854775808"
        );
    }

    #[test]
    fn id_out_of_range_reports_the_rejected_id() {
        assert_eq!(IdOutOfRange { id: u64::MAX }.id(), u64::MAX);
    }

    #[test]
    fn the_cap_is_two_to_the_sixty_third() {
        assert_eq!(ID_CAP, 9_223_372_036_854_775_808);
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
            start in 0..ID_CAP - 1024,
            operations in prop::collection::vec(prop::option::of(0..=500_u64), 0..=20),
        ) {
            let counter = AtomicU64::new(start);
            let reference = RefCell::new(ReferenceCounter { next_id: 0 });

            let counter_ids = run_counter_operations(
                &operations,
                || take_next_id(&counter).expect("the counter stays below u64::MAX"),
                |id| advance_past(&counter, id).expect("the id stays below the cap"),
            );
            let reference_ids = run_counter_operations(
                &operations,
                || reference.borrow_mut().allocate(),
                |id| reference.borrow_mut().advance_past(id),
            );

            prop_assert_eq!(counter_ids, reference_ids);
        }
    }

    /// Test advancing the global counter past an id at or above the cap
    /// fails, naming the id, and leaves the counter below the cap.
    #[rstest]
    #[case::the_cap(ID_CAP)]
    #[case::u64_max(u64::MAX)]
    fn try_advance_counter_past_rejects_an_id_at_or_above_the_cap(#[case] id: u64) {
        assert_eq!(try_advance_counter_past(id), Err(IdOutOfRange { id }));
        assert!(Identifier::new("after-a-rejected-advance").id() < ID_CAP);
    }

    /// Test restoring an id at or above the cap fails, naming the id, and
    /// leaves the counter below the cap.
    #[rstest]
    #[case::the_cap(ID_CAP)]
    #[case::u64_max(u64::MAX)]
    fn try_restore_rejects_an_id_at_or_above_the_cap(#[case] id: u64) {
        let error = Identifier::try_restore(id, "capped").map(drop);

        assert_eq!(error, Err(IdOutOfRange { id }));
        assert!(Identifier::new("after-a-rejected-restore").id() < ID_CAP);
    }

    #[test]
    fn try_new_draws_a_fresh_id_from_the_counter() {
        let before = Identifier::new("before-try-new").id();
        let drawn = Identifier::try_new("drawn").expect("the counter is far from exhausted");
        let after = Identifier::new("after-try-new").id();

        assert!(before < drawn.id() && drawn.id() < after);
        assert_eq!(drawn.name_hint(), "drawn");
    }

    #[test]
    fn try_allocate_id_shares_the_counter_with_construction() {
        let allocated = try_allocate_id();
        let constructed = Identifier::new("after-try-allocate");
        assert!(allocated.is_ok_and(|id| id < constructed.id()));
    }

    /// The paths along which an identifier payload is decoded.
    #[derive(Debug, Clone, Copy)]
    enum IdentifierPath {
        /// A bare identifier, from JSON text.
        Text,
        /// A bare identifier, from a JSON value.
        Value,
        /// The name of a note's kind.
        NoteKind,
        /// The name of a canonical op attribute.
        OpAttribute,
        /// The name of a value domain's parent, the first level of its
        /// chain.
        ValueDomainParent,
        /// The identifier of an identifier reference expression.
        Expression,
    }

    impl IdentifierPath {
        /// Decode an identifier payload with the id token `id` along this
        /// path, returning the error text.
        fn decode_error(self, id: &str) -> String {
            use crate::diagnostic::Note;
            use crate::expr::Expression;
            use crate::interned::Canonical;
            use crate::op_attribute::OpAttribute;
            use crate::value_domain::ValueDomain;

            let identifier = format!("{{\"id\":{id},\"name_hint\":\"x\"}}");
            let error = match self {
                Self::Text => serde_json::from_str::<Identifier>(&identifier).map(drop),
                Self::Value => serde_json::from_str::<serde_json::Value>(&identifier)
                    .and_then(serde_json::from_value::<Identifier>)
                    .map(drop),
                Self::NoteKind => serde_json::from_str::<Note>(&format!(
                    "{{\"message\":\"m\",\"kind\":{{\"name\":{identifier},\"description\":\"d\"}}}}"
                ))
                .map(drop),
                Self::OpAttribute => serde_json::from_str::<Canonical<OpAttribute>>(&format!(
                    "{{\"name\":{identifier},\"description\":\"d\"}}"
                ))
                .map(drop),
                Self::ValueDomainParent => {
                    let valid = format!(
                        "{{\"id\":{},\"name_hint\":\"child\"}}",
                        Identifier::new("out-of-range-child").id()
                    );
                    serde_json::from_str::<Canonical<ValueDomain>>(&format!(
                        "[{{\"name\":{identifier},\"description\":\"d\"}},\
                         {{\"name\":{valid},\"description\":\"d\"}}]"
                    ))
                    .map(drop)
                }
                Self::Expression => serde_json::from_str::<Expression>(&format!(
                    "{{\"__type__\":\"identifier_expression\",\"__data__\":{{\"identifier\":{identifier}}}}}"
                ))
                .map(drop),
            }
            .expect_err("the id is out of range");
            error.to_string()
        }

        /// Return the text the error along this path starts with, before the
        /// identifier's own message.
        fn describe_prefix(self) -> &'static str {
            match self {
                Self::Text
                | Self::Value
                | Self::NoteKind
                | Self::OpAttribute
                | Self::ValueDomainParent => "",
                Self::Expression => "in `identifier`: ",
            }
        }

        /// Return whether a number reaches the identifier's decode as the
        /// JSON text wrote it.
        ///
        /// Along the other paths it passes through a `serde_json::Value`
        /// first. With `serde_json`'s `arbitrary_precision` feature on, such
        /// a number reaches the decode as its digits, so a negative,
        /// fractional or oversized id is still rejected there, but with
        /// `serde_json`'s own message.
        fn reads_numbers_exactly(self) -> bool {
            matches!(
                self,
                Self::Text | Self::NoteKind | Self::OpAttribute | Self::ValueDomainParent
            )
        }

        /// Assert the error along this path for the id token `id` is
        /// `expected`, or, where [`reads_numbers_exactly`] does not hold, is
        /// at least reported from the identifier's position.
        ///
        /// [`reads_numbers_exactly`]: Self::reads_numbers_exactly
        fn assert_rejected_with(self, id: &str, expected: &str) {
            let message = self.decode_error(id);

            let expected = format!("{}{expected}", self.describe_prefix());
            if self.reads_numbers_exactly() {
                assert!(message.starts_with(&expected), "{self:?}: {message}");
            } else {
                assert!(
                    message.starts_with(self.describe_prefix()),
                    "{self:?}: {message}"
                );
            }
        }
    }

    /// Test that serde rejects an id at or above the cap that still fits a
    /// `u64`, saying the id is out of range and what the range is, whichever
    /// payload the identifier is nested in.
    #[rstest]
    fn serde_rejects_an_id_at_or_above_the_cap(
        #[values(
            IdentifierPath::Text,
            IdentifierPath::Value,
            IdentifierPath::NoteKind,
            IdentifierPath::OpAttribute,
            IdentifierPath::ValueDomainParent,
            IdentifierPath::Expression
        )]
        path: IdentifierPath,
        #[values("9223372036854775808", "9223372036854775809", "18446744073709551615")] id: &str,
    ) {
        let message = path.decode_error(id);

        let expected = format!(
            "{}invalid value: integer `{id}`, expected an id from 0 to 9223372036854775807",
            path.describe_prefix()
        );
        assert!(message.starts_with(&expected), "{path:?}: {message}");
    }

    /// Test that serde rejects an integer beyond `u64`, which a format reads
    /// as a float, as the wrong type, whichever payload the identifier is
    /// nested in.
    #[rstest]
    fn serde_rejects_an_id_beyond_u64_as_the_wrong_type(
        #[values(
            IdentifierPath::Text,
            IdentifierPath::Value,
            IdentifierPath::NoteKind,
            IdentifierPath::OpAttribute,
            IdentifierPath::ValueDomainParent,
            IdentifierPath::Expression
        )]
        path: IdentifierPath,
        #[values(
            "18446744073709551616",
            "1606938044258990275541962092341162602522202993782792835301376",
            "-18446744073709551616"
        )]
        id: &str,
    ) {
        path.assert_rejected_with(id, "invalid type: floating point");
    }

    /// Test that serde rejects a negative, fractional or string id, whichever
    /// payload the identifier is nested in.
    #[rstest]
    fn serde_rejects_a_negative_or_fractional_id(
        #[values(
            IdentifierPath::Text,
            IdentifierPath::Value,
            IdentifierPath::NoteKind,
            IdentifierPath::OpAttribute,
            IdentifierPath::ValueDomainParent,
            IdentifierPath::Expression
        )]
        path: IdentifierPath,
        #[values(
            ("-1", "invalid value: integer `-1`, expected an id from 0 to 9223372036854775807"),
            ("1.5", "invalid type: floating point `1.5`, expected an id from 0 to \
                     9223372036854775807"),
            ("1e3", "invalid type: floating point `1000.0`, expected an id from 0 to \
                     9223372036854775807"),
            ("\"7\"", "invalid type: string \"7\", expected an id from 0 to \
                       9223372036854775807")
        )]
        case: (&str, &str),
    ) {
        let (id, expected_message) = case;

        path.assert_rejected_with(id, expected_message);
    }

    #[test]
    fn advance_counter_past_keeps_later_ids_beyond_the_advanced_id() {
        let far_future_id = Identifier::new("advance-anchor").id() + 1_000_000;
        try_advance_counter_past(far_future_id).expect("the id is below the cap");
        assert!(try_allocate_id().is_ok_and(|id| id > far_future_id));
        assert!(Identifier::new("after-advance").id() > far_future_id);
    }

    #[test]
    fn advance_counter_past_is_a_no_op_for_an_issued_id() {
        let issued = Identifier::new("issued").id();
        let latest = Identifier::new("latest").id();
        try_advance_counter_past(issued).expect("the id is below the cap");
        assert!(Identifier::new("after-no-op").id() > latest);
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
                        Identifier::try_restore(id, "concurrent-deserialize")
                            .expect("the id is below the cap")
                            .id()
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

    /// Test building a reserved identifier draws no id from the counter:
    /// between two fresh anchors, two reserved identifiers leave the anchors
    /// one id apart. A test running in parallel can draw an id between the
    /// anchors, so the check is retried; a reserved identifier that drew an
    /// id would leave the anchors at least two ids apart on every try.
    #[test]
    fn reserved_identifiers_take_no_id_from_the_counter() {
        let mut deltas = Vec::new();
        for _ in 0..1_000 {
            let before = Identifier::new("reserved-anchor").id();
            let first = Identifier::reserved(reserved::COMMUTATIVE);
            let second = Identifier::reserved(reserved::COMMUTATIVE);
            let after = Identifier::new("reserved-anchor").id();
            assert_eq!((first.id(), second.id()), (16, 16));
            deltas.push(after - before);
            if after - before == 1 {
                return;
            }
        }
        panic!("the anchors were never one id apart: {deltas:?}");
    }

    /// Test a reserved identifier holds its table entry's id and name hint.
    #[test]
    fn a_reserved_identifier_holds_its_entry() {
        let identifier = Identifier::reserved(reserved::ADDRESS_DOMAIN);

        assert_eq!(identifier.id(), 33);
        assert_eq!(identifier.name_hint(), "address");
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
        assert_send_sync::<IdOutOfRange>();
    };
}
