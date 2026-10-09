//! Shared builders for the provenance and diagnostic tests, and test
//! custom provenances with a resolver that reads them back.

use std::borrow::Cow;
use std::fmt;
use std::hash::Hasher;

use fhy_core::foreign::{Foreign, ForeignError, ForeignPart, Part, Resolve};
use fhy_core::provenance::{CustomProvenance, FileProvenance, NamedProvenance, Provenance, Span};

/// The type id of [`EdgePropagation`].
pub(crate) const EDGE_PROPAGATION: &str = "test.edge_propagation";

/// Build the file provenance for `path` over `span`.
#[must_use]
pub(crate) fn build_file(path: &str, span: Option<Span>) -> Provenance {
    Provenance::File(FileProvenance::new(path, span))
}

/// Build the named provenance `name` over `child`.
///
/// # Panics
///
/// Panics if `name` is empty.
#[must_use]
pub(crate) fn build_named(name: &str, child: Provenance) -> Provenance {
    Provenance::Named(NamedProvenance::new(name, child).expect("name is non-empty"))
}

/// A custom provenance holding an edge's text, equal to another of the same
/// text, displayed as `edge<text>`, with a wire form under
/// [`EDGE_PROPAGATION`].
#[derive(Debug, PartialEq)]
pub(crate) struct EdgePropagation(pub(crate) String);

impl EdgePropagation {
    /// Return the custom provenance of the edge `text`.
    #[must_use]
    pub(crate) fn build(text: &str) -> Provenance {
        Provenance::Custom(Part::new(Self(text.to_owned())))
    }
}

impl fmt::Display for EdgePropagation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "edge<{}>", self.0)
    }
}

impl ForeignPart for EdgePropagation {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed("EdgePropagation")
    }

    fn to_foreign(&self) -> Result<Foreign, ForeignError> {
        Ok(Foreign::new(EDGE_PROPAGATION, self.0.as_str()))
    }
}

impl CustomProvenance for EdgePropagation {
    fn eq_part(&self, other: &dyn CustomProvenance) -> bool {
        other.as_any().downcast_ref::<Self>() == Some(self)
    }

    fn hash_part(&self, state: &mut dyn Hasher) {
        state.write(self.0.as_bytes());
    }
}

/// A custom provenance that keeps every default of its trait: equal only to
/// itself, hashing nothing, and with no wire form. It displays as
/// `unwired<text>`.
#[derive(Debug)]
pub(crate) struct UnwiredProvenance(pub(crate) String);

impl UnwiredProvenance {
    /// Return the custom provenance of `text`.
    #[must_use]
    pub(crate) fn build(text: &str) -> Provenance {
        Provenance::Custom(Part::new(Self(text.to_owned())))
    }
}

impl fmt::Display for UnwiredProvenance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unwired<{}>", self.0)
    }
}

impl ForeignPart for UnwiredProvenance {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed("UnwiredProvenance")
    }
}

impl CustomProvenance for UnwiredProvenance {}

/// The resolver of [`EdgePropagation`]: it reads a part of type id
/// [`EDGE_PROPAGATION`] and refuses any other.
#[derive(Debug, Default)]
pub(crate) struct EdgeResolver;

impl Resolve<Part<dyn CustomProvenance>> for EdgeResolver {
    fn resolve(&self, foreign: &Foreign) -> Result<Part<dyn CustomProvenance>, ForeignError> {
        if foreign.type_id() != EDGE_PROPAGATION {
            return Err(ForeignError::Unresolved {
                type_id: foreign.type_id().to_owned(),
            });
        }
        Ok(Part::new(EdgePropagation(foreign.data().to_owned())))
    }
}
