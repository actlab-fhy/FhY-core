//! Decoding a payload from its map form only.
//!
//! A derived struct decoder also accepts the struct's fields as a sequence,
//! without their keys. The expression wire format reads its payload into a
//! JSON value and checks it as a map, so it refuses that form through
//! [`deserialize_map_only`]. Every other type decodes through plain serde
//! derives and accepts both forms.

use serde::Deserialize;
use serde::de::{self, Deserializer};

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
/// Only the wrapped value is affected: its fields are read by the wrapped
/// deserializer.
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
