//! The wire form of a [`Provenance`] that may hold a custom provenance at
//! any depth, as a [`Foreign`] part until a resolver builds it.
//!
//! The shape is [`Provenance`]'s own: `{"unknown": {}}`, `{"file": ..}`,
//! `{"named": {"name", "child"}}`, `{"call_site": {"callee", "caller"}}`,
//! `{"fused": {"sources", "label"}}` and `{"custom": {"type_id", "data"}}`.

#![expect(
    unused_variables,
    clippy::todo,
    reason = "interface stub; bodies are todo!() until implementation"
)]

use serde::{Deserialize, Serialize};

use crate::foreign::{BuildError, Foreign, ForeignError, Part, Resolve};

use super::{CustomProvenance, FileProvenance, Provenance};

/// The wire form of a [`Provenance`], its custom provenances unresolved.
///
/// Decoding normalizes a file path and refuses an empty name, as
/// [`Provenance`]'s own decoding does, and keeps every custom part as a
/// [`Foreign`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ProvenanceData(ProvenanceRepr);

/// The parts of a [`ProvenanceData`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "Provenance", rename_all = "snake_case")]
enum ProvenanceRepr {
    Unknown(UnknownRepr),
    File(FileProvenance),
    Named(NamedRepr),
    CallSite(CallSiteRepr),
    Fused(FusedRepr),
    Custom(Foreign),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "Unknown", deny_unknown_fields)]
struct UnknownRepr {}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "NamedProvenance", deny_unknown_fields)]
struct NamedRepr {
    name: String,
    child: Box<ProvenanceData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "CallSiteProvenance", deny_unknown_fields)]
struct CallSiteRepr {
    callee: Box<ProvenanceData>,
    caller: Box<ProvenanceData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "FusedProvenance", deny_unknown_fields)]
struct FusedRepr {
    sources: Vec<ProvenanceData>,
    label: Option<String>,
}

impl ProvenanceData {
    /// Return the wire form of `provenance`.
    ///
    /// # Errors
    ///
    /// Returns the error of a custom provenance, at any depth, that cannot
    /// give its foreign form.
    pub fn of(provenance: &Provenance) -> Result<Self, ForeignError> {
        todo!()
    }

    /// Return the provenance, each custom part resolved by `resolver`.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::Foreign`] for a custom part `resolver`
    /// refuses, and [`BuildError::Invalid`] with the
    /// [`NamedProvenanceError`](super::NamedProvenanceError) of an empty
    /// name.
    pub fn build<R: Resolve<Part<dyn CustomProvenance>> + ?Sized>(
        self,
        resolver: &R,
    ) -> Result<Provenance, BuildError> {
        todo!()
    }
}
