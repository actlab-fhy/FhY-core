//! Payload subtrees read now and decoded later.
//!
//! [`super::DeferredPayload`] holds its nested level as a [`BufferedMap`], read
//! now and decoded from once the holder's build reaches it. A streaming
//! format hands over fields in whatever order the payload lists them, so the
//! nested level must be read into a value before it can wait for its turn.
//!
//! Buffering reads the subtree through `deserialize_any`, so it needs a
//! self-describing format such as JSON.

use std::fmt;

use serde::de::value::{Error, MapDeserializer, SeqDeserializer};
use serde::de::{Deserializer, IntoDeserializer, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, forward_to_deserialize_any};

/// Any value of a self-describing payload, held until something decodes it.
#[derive(Debug)]
pub(super) enum BufferedValue {
    Unit,
    Bool(bool),
    Unsigned(u64),
    Signed(i64),
    Float(f64),
    Text(String),
    Seq(Vec<BufferedValue>),
    Map(BufferedMap),
}

/// A map-valued payload subtree, held until something decodes it.
///
/// Reading one rejects any value that is not a map, so a caller learns that
/// the subtree has the wrong shape before it decodes anything else.
#[derive(Debug)]
pub(super) struct BufferedMap(Vec<(BufferedValue, BufferedValue)>);

/// Collect a map's entries without interpreting them.
fn collect_map_entries<'de, A: MapAccess<'de>>(
    mut map: A,
) -> Result<Vec<(BufferedValue, BufferedValue)>, A::Error> {
    let mut entries = Vec::with_capacity(map.size_hint().unwrap_or(0));
    while let Some(entry) = map.next_entry()? {
        entries.push(entry);
    }
    Ok(entries)
}

impl<'de> Deserialize<'de> for BufferedValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct BufferedValueVisitor;

        impl<'de> Visitor<'de> for BufferedValueVisitor {
            type Value = BufferedValue;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("any payload value")
            }

            fn visit_unit<E>(self) -> Result<BufferedValue, E> {
                Ok(BufferedValue::Unit)
            }

            fn visit_none<E>(self) -> Result<BufferedValue, E> {
                Ok(BufferedValue::Unit)
            }

            fn visit_some<D: Deserializer<'de>>(
                self,
                deserializer: D,
            ) -> Result<BufferedValue, D::Error> {
                BufferedValue::deserialize(deserializer)
            }

            fn visit_bool<E>(self, value: bool) -> Result<BufferedValue, E> {
                Ok(BufferedValue::Bool(value))
            }

            fn visit_u64<E>(self, value: u64) -> Result<BufferedValue, E> {
                Ok(BufferedValue::Unsigned(value))
            }

            fn visit_i64<E>(self, value: i64) -> Result<BufferedValue, E> {
                Ok(BufferedValue::Signed(value))
            }

            fn visit_f64<E>(self, value: f64) -> Result<BufferedValue, E> {
                Ok(BufferedValue::Float(value))
            }

            fn visit_str<E>(self, value: &str) -> Result<BufferedValue, E> {
                Ok(BufferedValue::Text(value.to_owned()))
            }

            fn visit_string<E>(self, value: String) -> Result<BufferedValue, E> {
                Ok(BufferedValue::Text(value))
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<BufferedValue, A::Error> {
                let mut items = Vec::with_capacity(seq.size_hint().unwrap_or(0));
                while let Some(item) = seq.next_element()? {
                    items.push(item);
                }
                Ok(BufferedValue::Seq(items))
            }

            fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<BufferedValue, A::Error> {
                collect_map_entries(map).map(|entries| BufferedValue::Map(BufferedMap(entries)))
            }
        }

        deserializer.deserialize_any(BufferedValueVisitor)
    }
}

impl<'de> Deserialize<'de> for BufferedMap {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct BufferedMapVisitor;

        impl<'de> Visitor<'de> for BufferedMapVisitor {
            type Value = BufferedMap;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a map")
            }

            fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<BufferedMap, A::Error> {
                collect_map_entries(map).map(BufferedMap)
            }
        }

        deserializer.deserialize_map(BufferedMapVisitor)
    }
}

impl From<BufferedMap> for BufferedValue {
    fn from(map: BufferedMap) -> Self {
        Self::Map(map)
    }
}

impl<'de> Deserializer<'de> for BufferedValue {
    type Error = Error;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        match self {
            Self::Unit => visitor.visit_unit(),
            Self::Bool(value) => visitor.visit_bool(value),
            Self::Unsigned(value) => visitor.visit_u64(value),
            Self::Signed(value) => visitor.visit_i64(value),
            Self::Float(value) => visitor.visit_f64(value),
            Self::Text(value) => visitor.visit_string(value),
            Self::Seq(items) => {
                let mut seq = SeqDeserializer::new(items.into_iter());
                let value = visitor.visit_seq(&mut seq)?;
                seq.end()?;
                Ok(value)
            }
            Self::Map(BufferedMap(entries)) => {
                let mut map = MapDeserializer::new(entries.into_iter());
                let value = visitor.visit_map(&mut map)?;
                map.end()?;
                Ok(value)
            }
        }
    }

    fn deserialize_option<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        match self {
            Self::Unit => visitor.visit_none(),
            other => visitor.visit_some(other),
        }
    }

    forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
        bytes byte_buf unit unit_struct newtype_struct seq tuple tuple_struct
        map struct enum identifier ignored_any
    }
}

impl IntoDeserializer<'_, Error> for BufferedValue {
    type Deserializer = Self;

    fn into_deserializer(self) -> Self {
        self
    }
}
