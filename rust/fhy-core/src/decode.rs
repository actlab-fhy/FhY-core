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
//! `interned::intern_decoded(deferred.decode()?)`.
//!
//! A deferred level is buffered through `deserialize_any` before it is
//! decoded, so it needs a self-describing format such as JSON.

mod buffered;

use std::marker::PhantomData;

use serde::de::DeserializeOwned;
use serde::de::{self, Deserializer};
use serde::Deserialize;

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

/// Decode `T` by checking its payload first and then building it.
pub(crate) fn deserialize_via_payload<'de, T: Decode, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<T, D::Error> {
    T::build_from_payload(T::Payload::deserialize(deserializer)?)
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
    /// Decode the deferred level, performing its build's side effects.
    ///
    /// # Errors
    ///
    /// Returns an error if the level is malformed or conflicts with a
    /// canonical instance.
    pub(crate) fn decode<E: de::Error>(self) -> Result<T, E> {
        let payload = T::Payload::deserialize(BufferedValue::from(self.map)).map_err(E::custom)?;
        T::build_from_payload(payload)
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

    /// Return the names built so far, in build order.
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
                Some(child) => Some(Box::new(child.decode()?)),
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
