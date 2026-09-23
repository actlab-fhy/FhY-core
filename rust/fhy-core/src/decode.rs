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
//! [`deserialize_via_payload`]. A nested level that must stay unchecked until
//! the outer level's side effects have run is a [`DeferredPayload`], decoded
//! with [`DeferredPayload::decode`] once the holder's build reaches it; a
//! nested canonical value is then interned with
//! `interned::intern_decoded(deferred.decode("field")?)`.
//!
//! A deferred level is buffered through `deserialize_any` before it is
//! decoded, so it needs a self-describing format such as JSON.
//!
//! Every payload decodes from its map form only. A derived struct decoder
//! would also accept the struct's fields as a sequence, without their keys;
//! [`deserialize_via_payload`] and [`deserialize_map_only`] refuse that
//! form.

mod buffered;

use std::marker::PhantomData;

use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde::de::{self, Deserializer};

use buffered::{BufferedMap, BufferedValue};

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

/// A nested level of a payload, read now and checked and built only when its
/// holder's build reaches it.
pub(crate) struct DeferredPayload<T> {
    map: BufferedMap,
    decoded_as: PhantomData<fn() -> T>,
}

impl<'de, T> Deserialize<'de> for DeferredPayload<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        BufferedMap::deserialize(deserializer).map(|map| Self {
            map,
            decoded_as: PhantomData,
        })
    }
}

impl<T: Decode> DeferredPayload<T> {
    /// Decode the deferred level held in the holder's `field`, performing its
    /// build's side effects.
    ///
    /// # Errors
    ///
    /// Returns an error if the level is malformed or conflicts with a
    /// canonical instance, prefixed with ``in `field`: `` so an error from a
    /// deep chain shows the path to the level it came from.
    pub(crate) fn decode<E: de::Error>(self, field: &str) -> Result<T, E> {
        let add_field =
            |error: &dyn std::fmt::Display| E::custom(format_args!("in `{field}`: {error}"));
        let payload = T::Payload::deserialize(BufferedValue::from(self.map))
            .map_err(|error| add_field(&error))?;
        T::build_from_payload::<E>(payload).map_err(|error| add_field(&error))
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;

    thread_local! {
        /// Names `Probe::build_from_payload` has built so far, in build
        /// order. Thread-local because tests run in parallel.
        static BUILD_LOG: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    }

    /// Record that `name` was built.
    fn record_build(name: &str) {
        BUILD_LOG.with(|log| log.borrow_mut().push(name.to_owned()));
    }

    /// Drain and return the names built so far, in build order.
    fn take_build_log() -> Vec<String> {
        BUILD_LOG.with(|log| std::mem::take(&mut *log.borrow_mut()))
    }

    /// Minimal type exercising [`Decode`] through a nested
    /// [`DeferredPayload`].
    #[derive(Debug)]
    struct Probe {
        name: String,
        child: Option<Box<Probe>>,
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct ProbePayload {
        name: String,
        // Naming a `deserialize_with` turns off `Option`'s missing-key
        // special case, matching how the value-domain payload's `parent`
        // field is read.
        #[serde(deserialize_with = "Option::deserialize")]
        child: Option<DeferredPayload<Probe>>,
    }

    impl Decode for Probe {
        type Payload = ProbePayload;

        fn build_from_payload<E: de::Error>(payload: Self::Payload) -> Result<Self, E> {
            record_build(&payload.name);
            let child = match payload.child {
                None => None,
                Some(child) => Some(Box::new(child.decode("child")?)),
            };
            Ok(Self {
                name: payload.name,
                child,
            })
        }
    }

    impl<'de> Deserialize<'de> for Probe {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            deserialize_via_payload(deserializer)
        }
    }

    /// Test that a payload rejected for an unknown trailing field builds
    /// nothing.
    #[test]
    fn a_payload_rejected_for_an_unknown_field_builds_nothing() {
        let json = r#"{"name":"outer","child":null,"surprise":1}"#;

        let error = serde_json::from_str::<Probe>(json).unwrap_err();

        assert!(error.to_string().contains("surprise"), "{error}");
        assert!(take_build_log().is_empty());
    }

    /// Test that a malformed child level is not checked until the outer
    /// build reaches it.
    #[test]
    fn a_malformed_child_is_not_checked_until_the_outer_build_reaches_it() {
        let json = r#"{"name":"outer","child":{"name":"inner","child":null,"surprise":1}}"#;

        let error = serde_json::from_str::<Probe>(json).unwrap_err();

        assert!(error.to_string().contains("surprise"), "{error}");
        assert_eq!(take_build_log(), vec!["outer".to_owned()]);
    }

    /// Test that an error from a level two deep names the path to it.
    #[test]
    fn an_error_two_levels_deep_names_the_path_to_its_level() {
        let json = r#"{"name":"outer","child":{"name":"inner","child":
            {"name":"leaf","child":null,"surprise":1}}}"#;

        let error = serde_json::from_str::<Probe>(json).unwrap_err();

        let message = error.to_string();
        assert!(
            message.starts_with("in `child`: in `child`: unknown field `surprise`"),
            "{message}"
        );
    }

    /// Test that a child which is not a map is rejected while the outer
    /// payload is read, before anything is built.
    #[test]
    fn a_non_map_child_is_rejected_before_anything_is_built() {
        let json = r#"{"name":"outer","child":"not a map"}"#;

        let error = serde_json::from_str::<Probe>(json).unwrap_err();

        assert!(error.to_string().contains("invalid type"), "{error}");
        assert!(take_build_log().is_empty());
    }

    /// Test that a well-formed two-level payload builds the outer level
    /// first, then the child, and returns the nested value.
    #[test]
    fn a_well_formed_payload_builds_outer_first_then_child() {
        let json = r#"{"name":"outer","child":{"name":"inner","child":null}}"#;

        let probe = serde_json::from_str::<Probe>(json).unwrap();

        assert_eq!(
            take_build_log(),
            vec!["outer".to_owned(), "inner".to_owned()]
        );
        assert_eq!(probe.name, "outer");
        let child = probe.child.expect("the child decodes");
        assert_eq!(child.name, "inner");
        assert!(child.child.is_none());
    }
}
