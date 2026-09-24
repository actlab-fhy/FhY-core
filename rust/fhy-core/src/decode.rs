//! Side-effecting decode split into a checked payload and a build step.
//!
//! Decoding an [`crate::identifier::Identifier`] advances the process-global
//! id counter, and decoding a [`crate::interned::Canonical`] interns into a
//! process-global registry. To match the Python deserializer's side effects,
//! a type whose decode carries either of those effects must check its
//! payload -- fields present, known and well typed, identifier ids held
//! unrestored, nested levels held unread -- before it performs the effects,
//! and it must perform them in a set order.
//!
//! A type whose decode restores identifiers or interns values implements
//! [`Decode`]; its `Deserialize` impl is a one-line call to
//! [`deserialize_via_payload`].
//!
//! Every payload decodes from its map form only. A derived struct decoder
//! would also accept the struct's fields as a sequence, without their keys;
//! [`deserialize_via_payload`] and [`deserialize_map_only`] refuse that
//! form.

use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde::de::{self, Deserializer};

/// A type decoded in two phases, so its decode's side effects happen in a set
/// order.
pub(crate) trait Decode: Sized {
    /// One checked level of the type's payload. Decoding it has no side
    /// effects: identifier ids stay unrestored and nested levels stay
    /// unread.
    type Payload: DeserializeOwned;

    /// Build the value from its checked payload, performing the decode's
    /// side effects (restoring identifiers, decoding and interning nested
    /// levels).
    ///
    /// The built value itself is left unregistered; decoding a
    /// [`Canonical`](crate::interned::Canonical) is what interns it.
    ///
    /// # Errors
    ///
    /// Returns an error if a nested level is malformed or conflicts with a
    /// canonical instance.
    fn build_from_payload<E: de::Error>(payload: Self::Payload) -> Result<Self, E>;
}

/// Decode `T` by checking its payload, in its map form only, first and then
/// building it.
pub(crate) fn deserialize_via_payload<'de, T: Decode, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<T, D::Error> {
    T::build_from_payload(deserialize_map_only::<D, T::Payload>(deserializer)?)
}

/// Decode `T` from `deserializer`, accepting only the map form.
///
/// # Errors
///
/// Returns an error if the input is not a map or does not decode as `T`.
pub(crate) fn deserialize_map_only<'de, D: Deserializer<'de>, T: Deserialize<'de>>(
    deserializer: D,
) -> Result<T, D::Error> {
    T::deserialize(MapOnly(deserializer))
}

/// Deserializer adapter that reads the value it wraps as a map.
///
/// A derived struct decoder also accepts the struct's fields as a sequence;
/// routing it through this adapter refuses that form. Only the wrapped value
/// is affected: its fields are read by the wrapped deserializer, so a nested
/// value is map-only when its own `Deserialize` impl routes through
/// [`deserialize_map_only`].
struct MapOnly<D>(D);

impl<'de, D: Deserializer<'de>> Deserializer<'de> for MapOnly<D> {
    type Error = D::Error;

    fn deserialize_any<V: de::Visitor<'de>>(self, visitor: V) -> Result<V::Value, D::Error> {
        self.0.deserialize_map(visitor)
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
        bytes byte_buf option unit unit_struct newtype_struct seq tuple
        tuple_struct map struct enum identifier ignored_any
    }
}
