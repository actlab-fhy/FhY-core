//! The shared shape of a described, registry-backed tag.
//!
//! A described tag is an open vocabulary entry: an [`Identifier`] name that
//! is its identity, and a human-readable description that takes no part in
//! equality, hashing or interning. [`define_described_tag`] generates such a
//! type together with its shipped defaults, so every tag of this shape
//! encodes, decodes, interns and compares the same way.
//!
//! [`Identifier`]: crate::identifier::Identifier

/// Define a described tag type, its payload and its shipped defaults.
///
/// The invocation names the type and its documentation, the payload type and
/// the struct name its decode errors report, the noun the generated
/// documentation calls a value, the documentation of `new`, and each shipped
/// default: its getter and the documentation of that getter, the statics that
/// hold it and its name, and the reserved-table entry that names it and its
/// description. Defaults are registered in the order they are listed.
///
/// The type encodes as `{"name": <identifier>, "description": ..}`. Its
/// decode checks every field before it restores the name, then builds the
/// value without registering it; decoding a
/// [`Canonical`](crate::interned::Canonical) of it interns it.
macro_rules! define_described_tag {
    (
        $(#[$type_meta:meta])*
        pub struct $Type:ident;
        payload $Payload:ident as $wire_name:tt;
        noun $noun:literal;
        $(#[$new_meta:meta])*
        fn new;
        shipped by $create_defaults:ident {
            $(
                $(#[$getter_meta:meta])*
                fn $getter:ident => $STATIC:ident, $NAME:ident =
                    ($reserved:ident, $description:literal);
            )+
        }
    ) => {
        $(#[$type_meta])*
        #[derive(Debug, ::serde::Serialize)]
        pub struct $Type {
            name: $crate::identifier::Identifier,
            description: ::std::string::String,
        }

        impl $Type {
            $(#[$new_meta])*
            pub fn new(
                name: $crate::identifier::Identifier,
                description: impl ::std::convert::Into<::std::string::String>,
            ) -> $crate::interned::InternOutcome<Self> {
                <Self as $crate::interned::Interned>::intern_registry()
                    .intern(Self::create(name, description))
            }

            #[doc = concat!("Build the ", $noun, " without registering it.")]
            fn create(
                name: $crate::identifier::Identifier,
                description: impl ::std::convert::Into<::std::string::String>,
            ) -> Self {
                Self {
                    name,
                    description: description.into(),
                }
            }

            #[doc = concat!("Return the ", $noun, "'s name.")]
            #[must_use]
            pub fn name(&self) -> &$crate::identifier::Identifier {
                &self.name
            }

            #[doc = concat!("Return the ", $noun, "'s human-readable description.")]
            #[must_use]
            pub fn description(&self) -> &str {
                &self.description
            }
        }

        impl $crate::identifier::HasIdentifier for $Type {
            fn identifier(&self) -> &$crate::identifier::Identifier {
                &self.name
            }
        }

        impl $crate::interned::Interned for $Type {
            type Key = $crate::identifier::Identifier;

            fn intern_key(&self) -> &$crate::identifier::Identifier {
                &self.name
            }

            fn intern_registry() -> &'static $crate::interned::InternRegistry<Self> {
                static REGISTRY: $crate::interned::InternRegistry<$Type> =
                    $crate::interned::InternRegistry::with_defaults($create_defaults);
                &REGISTRY
            }
        }

        impl ::std::cmp::PartialEq for $Type {
            fn eq(&self, other: &Self) -> bool {
                self.name == other.name
            }
        }

        impl ::std::cmp::Eq for $Type {}

        impl ::std::hash::Hash for $Type {
            fn hash<H: ::std::hash::Hasher>(&self, state: &mut H) {
                self.name.hash(state);
            }
        }

        impl $crate::decode::Decode for $Type {
            type Payload = $Payload;

            fn build_from_payload<E: ::serde::de::Error>(
                payload: Self::Payload,
            ) -> ::std::result::Result<Self, E> {
                let name = $crate::identifier::Identifier::try_from(payload.name)
                    .map_err(E::custom)?;
                Ok(Self::create(name, payload.description))
            }
        }

        /// Decoding checks every field of the payload before it restores the
        /// name, so a rejected payload leaves the id counter untouched.
        impl<'de> ::serde::Deserialize<'de> for $Type {
            fn deserialize<D: ::serde::Deserializer<'de>>(
                deserializer: D,
            ) -> ::std::result::Result<Self, D::Error> {
                $crate::decode::deserialize_via_payload(deserializer)
            }
        }

        #[doc = concat!(
            "The checked payload of a [`", stringify!($Type),
            "`], with its name not yet restored."
        )]
        #[derive(::serde::Deserialize)]
        #[serde(rename = $wire_name, deny_unknown_fields)]
        pub(crate) struct $Payload {
            name: $crate::identifier::IdentifierWire,
            description: ::std::string::String,
        }

        $(
            #[doc = concat!(
                "Name of the ", $noun, " returned by [`", stringify!($getter), "`]."
            )]
            static $NAME: ::std::sync::LazyLock<$crate::identifier::Identifier> =
                ::std::sync::LazyLock::new(|| {
                    $crate::identifier::Identifier::reserved(
                        $crate::identifier::reserved::$reserved,
                    )
                });

            static $STATIC: ::std::sync::LazyLock<$crate::interned::Canonical<$Type>> =
                ::std::sync::LazyLock::new(|| $crate::interned::require_default(&*$NAME));

            $(#[$getter_meta])*
            #[must_use]
            pub fn $getter() -> &'static $crate::interned::Canonical<$Type> {
                &$STATIC
            }
        )+

        /// Build the defaults this module ships, in registration order.
        ///
        /// The registry calls this once, on its first use, and keeps the
        /// instances it builds. A clear registers those same instances again
        /// rather than building new ones, so the shipped defaults stay
        /// canonical for the life of the process.
        fn $create_defaults() -> ::std::vec::Vec<$Type> {
            ::std::vec![$($Type::create($NAME.clone(), $description)),+]
        }

    };
}

pub(crate) use define_described_tag;
