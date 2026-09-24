//! The fixed ids of the identifiers this crate ships.
//!
//! Every shipped tag names itself with an entry of this table, through
//! [`Identifier::reserved`](super::Identifier::reserved), so its id is the
//! same in every process and in every payload, whatever order the shipped
//! values are first used in. Building a reserved identifier draws nothing
//! from the id counter, which issues fresh ids from
//! [`RESERVED_ID_COUNT`] upward.
//!
//! The table is append-only: an assigned id never changes, a retired entry
//! leaves a hole, and a new entry takes the next free id in its family's
//! block. The blocks are `0..16` for note kinds, `16..32` for op attributes
//! and `32..48` for value domains.

use super::RESERVED_ID_COUNT;

/// One entry of the table: a fixed id and the name hint shipped with it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ReservedIdentifier {
    id: u64,
    name_hint: &'static str,
}

impl ReservedIdentifier {
    /// Private, so only this table mints entries.
    const fn new(id: u64, name_hint: &'static str) -> Self {
        Self { id, name_hint }
    }

    pub(crate) const fn id(self) -> u64 {
        self.id
    }

    pub(crate) const fn name_hint(self) -> &'static str {
        self.name_hint
    }
}

/// The note kind for notes that explain a decision.
pub(crate) const RATIONALE_NOTE_KIND: ReservedIdentifier = ReservedIdentifier::new(0, "rationale");

/// The note kind for notes that suggest a fix.
pub(crate) const SUGGESTION_NOTE_KIND: ReservedIdentifier =
    ReservedIdentifier::new(1, "suggestion");

/// The note kind for neutral notes.
pub(crate) const REMARK_NOTE_KIND: ReservedIdentifier = ReservedIdentifier::new(2, "remark");

/// The note kind for uncategorized notes.
pub(crate) const OTHER_NOTE_KIND: ReservedIdentifier = ReservedIdentifier::new(3, "other");

/// The op attribute for commutative ops.
pub(crate) const COMMUTATIVE: ReservedIdentifier = ReservedIdentifier::new(16, "commutative");

/// The op attribute for associative ops.
pub(crate) const ASSOCIATIVE: ReservedIdentifier = ReservedIdentifier::new(17, "associative");

/// The op attribute for pure ops.
pub(crate) const PURE: ReservedIdentifier = ReservedIdentifier::new(18, "pure");

/// The op attribute for elementwise ops.
pub(crate) const ELEMENTWISE: ReservedIdentifier = ReservedIdentifier::new(19, "elementwise");

/// The value domain of concrete data.
pub(crate) const DATA_DOMAIN: ReservedIdentifier = ReservedIdentifier::new(32, "data");

/// The value domain of addresses.
pub(crate) const ADDRESS_DOMAIN: ReservedIdentifier = ReservedIdentifier::new(33, "address");

// Fails the build unless every id lies below `RESERVED_ID_COUNT` and no two
// entries share an id.
const _: () = {
    let table = [
        RATIONALE_NOTE_KIND,
        SUGGESTION_NOTE_KIND,
        REMARK_NOTE_KIND,
        OTHER_NOTE_KIND,
        COMMUTATIVE,
        ASSOCIATIVE,
        PURE,
        ELEMENTWISE,
        DATA_DOMAIN,
        ADDRESS_DOMAIN,
    ];
    let mut index = 0;
    while index < table.len() {
        assert!(
            table[index].id < RESERVED_ID_COUNT,
            "a reserved id lies outside the reserved block"
        );
        let mut other = index + 1;
        while other < table.len() {
            assert!(
                table[index].id != table[other].id,
                "two reserved entries share an id"
            );
            other += 1;
        }
        index += 1;
    }
};
