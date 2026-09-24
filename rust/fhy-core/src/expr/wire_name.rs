//! The text and serde forms of an enum named on the wire by its `as_str`
//! text.
//!
//! [`impl_wire_name_traits`] gives such an enum a `Display` and a
//! `Serialize` that write its `as_str` text, and a `Deserialize` that
//! accepts exactly one of those texts through [`deserialize_wire_name`].

use serde::de::{self, Deserialize, Deserializer};

/// Decode the member of `members` whose `as_str` text is the string
/// `deserializer` holds.
///
/// # Errors
///
/// Returns an error if the input is not a string, or is a string that no
/// member's text matches exactly; the latter names the string and
/// `expecting`.
pub(crate) fn deserialize_wire_name<'de, D: Deserializer<'de>, T: Copy>(
    deserializer: D,
    members: &[T],
    as_str: fn(T) -> &'static str,
    expecting: &'static str,
) -> Result<T, D::Error> {
    let text = String::deserialize(deserializer)?;
    members
        .iter()
        .copied()
        .find(|candidate| as_str(*candidate) == text)
        .ok_or_else(|| de::Error::invalid_value(de::Unexpected::Str(&text), &expecting))
}

/// Implement `Display`, `Serialize` and `Deserialize` for an enum through
/// its `as_str` text, given the array listing every member and the
/// description a rejected string's error expects.
macro_rules! impl_wire_name_traits {
    ($Type:ty, $members:expr, $expecting:literal) => {
        impl ::std::fmt::Display for $Type {
            /// Write the [`as_str`](Self::as_str) text.
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                f.write_str(self.as_str())
            }
        }

        impl ::serde::Serialize for $Type {
            fn serialize<S: ::serde::Serializer>(
                &self,
                serializer: S,
            ) -> ::std::result::Result<S::Ok, S::Error> {
                serializer.serialize_str(self.as_str())
            }
        }

        impl<'de> ::serde::Deserialize<'de> for $Type {
            fn deserialize<D: ::serde::Deserializer<'de>>(
                deserializer: D,
            ) -> ::std::result::Result<Self, D::Error> {
                $crate::expr::wire_name::deserialize_wire_name(
                    deserializer,
                    &$members,
                    Self::as_str,
                    $expecting,
                )
            }
        }
    };
}

pub(crate) use impl_wire_name_traits;
