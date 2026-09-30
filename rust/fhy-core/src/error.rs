//! Errors shared by the whole crate.
//!
//! [`UnknownNameError`] is the error of every enum whose variants have a
//! stable name text: its [`FromStr`](std::str::FromStr) parses exactly that
//! text, and refuses any other with this error. The enum's own module
//! defines the text; this layer-1 module holds only the error, so a module
//! of any layer returns it without depending on another.

use std::error::Error;
use std::fmt;

use serde::Deserialize;
use serde::de::IntoDeserializer;
use serde::de::value::StrDeserializer;

/// A name that no variant of an enum has, refused by the enum's
/// [`FromStr`](std::str::FromStr).
///
/// `Display` writes the enum's description and the name in backticks, such
/// as ``unknown binary operation `plus` ``.
///
/// # Examples
///
/// ```
/// use fhy_core::error::UnknownNameError;
/// use fhy_core::expression::BinaryOperation;
///
/// let error: UnknownNameError = "plus".parse::<BinaryOperation>().unwrap_err();
/// assert_eq!(error.name(), "plus");
/// assert_eq!(error.to_string(), "unknown binary operation `plus`");
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownNameError {
    name: Box<str>,
    expected: &'static str,
}

impl UnknownNameError {
    /// Return the error of `name`, which no variant of the enum
    /// `expected` describes has.
    pub(crate) fn new(name: &str, expected: &'static str) -> Self {
        Self {
            name: name.into(),
            expected,
        }
    }

    /// Return the refused name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl fmt::Display for UnknownNameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unknown {} `{}`", self.expected, self.name)
    }
}

impl Error for UnknownNameError {}

/// Parse `text` as the variant of `T` whose serialized name it is, exactly,
/// through `T`'s derived `Deserialize`; `expected` names the enum in the
/// error.
///
/// # Errors
///
/// Returns [`UnknownNameError`] if no variant of `T` has the name `text`.
pub(crate) fn parse_variant_name<'a, T: Deserialize<'a>>(
    text: &'a str,
    expected: &'static str,
) -> Result<T, UnknownNameError> {
    let deserializer: StrDeserializer<'a, NoVariant> = text.into_deserializer();
    T::deserialize(deserializer)
        .map_err(|_no_variant: NoVariant| UnknownNameError::new(text, expected))
}

/// The deserializer error of [`parse_variant_name`].
///
/// It discards serde's message: a derived enum's unknown-variant error
/// otherwise formats a list of every variant name, which a caller that only
/// asks "is this a variant" never reads, and which made parsing a
/// non-built-in function name slow.
#[derive(Debug)]
struct NoVariant;

impl fmt::Display for NoVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("no variant has this name")
    }
}

impl Error for NoVariant {}

impl serde::de::Error for NoVariant {
    fn custom<M: fmt::Display>(_message: M) -> Self {
        Self
    }
}

/// Implement `Display`, writing the text the `$text` method returns
/// (`as_str` by default), and `FromStr`, parsing exactly that text through
/// the derived `Deserialize`, for an enum; `$expected` names the enum in a
/// refusal.
macro_rules! impl_name_text {
    ($Type:ty, $expected:literal) => {
        $crate::error::impl_name_text!($Type, as_str, $expected);
    };
    ($Type:ty, $text:ident, $expected:literal) => {
        impl ::std::fmt::Display for $Type {
            /// Write the
            #[doc = concat!("[`", stringify!($text), "`](Self::", stringify!($text), ")")]
            /// text.
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                f.write_str(self.$text())
            }
        }

        impl ::std::str::FromStr for $Type {
            type Err = $crate::error::UnknownNameError;

            /// Parse the variant whose
            #[doc = concat!("[`", stringify!($text), "`](Self::", stringify!($text), ")")]
            /// text is exactly `text`.
            ///
            /// # Errors
            ///
            /// Returns [`UnknownNameError`](crate::error::UnknownNameError)
            /// if no variant has that text.
            fn from_str(text: &str) -> Result<Self, Self::Err> {
                $crate::error::parse_variant_name(text, $expected)
            }
        }
    };
}

/// Implement `FromStr` for an enum that writes its own `Display`: parse
/// exactly the text the `$text` method returns for one of the listed
/// `$variant`s, which are all of them; `$expected` names the enum in a
/// refusal.
macro_rules! impl_from_name {
    ($Type:ty, $text:ident, $expected:literal, [$($variant:ident),+ $(,)?]) => {
        impl ::std::str::FromStr for $Type {
            type Err = $crate::error::UnknownNameError;

            /// Parse the variant whose
            #[doc = concat!("[`", stringify!($text), "`](Self::", stringify!($text), ")")]
            /// text is exactly `text`.
            ///
            /// # Errors
            ///
            /// Returns [`UnknownNameError`](crate::error::UnknownNameError)
            /// if no variant has that text.
            fn from_str(text: &str) -> Result<Self, Self::Err> {
                [$(Self::$variant),+]
                    .into_iter()
                    .find(|variant| variant.$text() == text)
                    .ok_or_else(|| $crate::error::UnknownNameError::new(text, $expected))
            }
        }
    };
}

pub(crate) use {impl_from_name, impl_name_text};
