//! The wire form of a [`Provenance`] that may hold a custom provenance at
//! any depth, as a [`Foreign`] part until a resolver builds it.
//!
//! The shape is [`Provenance`]'s own: `{"unknown": {}}`, `{"file": ..}`,
//! `{"named": {"name", "child"}}`, `{"call_site": {"callee", "caller"}}`,
//! `{"fused": {"sources", "label"}}` and `{"custom": {"type_id", "data"}}`.

use serde::{Deserialize, Serialize};

use crate::foreign::{BuildError, Foreign, ForeignError, Part, Resolve};

use super::{
    CallSiteProvenance, CustomProvenance, FileProvenance, FusedProvenance, NamedProvenance,
    Provenance,
};

/// The wire form of a [`Provenance`], its custom provenances unresolved.
///
/// Decoding normalizes a file path, keeps every custom part as a
/// [`Foreign`], and accepts an empty name: only [`ProvenanceData::build`]
/// refuses an empty name.
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
        Ok(Self(match provenance {
            Provenance::Unknown => ProvenanceRepr::Unknown(UnknownRepr {}),
            Provenance::File(file) => ProvenanceRepr::File(file.clone()),
            Provenance::Named(named) => ProvenanceRepr::Named(NamedRepr {
                name: named.name().to_owned(),
                child: Box::new(Self::of(named.child())?),
            }),
            Provenance::CallSite(call_site) => ProvenanceRepr::CallSite(CallSiteRepr {
                callee: Box::new(Self::of(call_site.callee())?),
                caller: Box::new(Self::of(call_site.caller())?),
            }),
            Provenance::Fused(fused) => ProvenanceRepr::Fused(FusedRepr {
                sources: fused
                    .sources()
                    .iter()
                    .map(Self::of)
                    .collect::<Result<_, _>>()?,
                label: fused.label().map(str::to_owned),
            }),
            Provenance::Custom(custom) => ProvenanceRepr::Custom(custom.get().to_foreign()?),
        }))
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
        Ok(match self.0 {
            ProvenanceRepr::Unknown(UnknownRepr {}) => Provenance::Unknown,
            ProvenanceRepr::File(file) => Provenance::File(file),
            ProvenanceRepr::Named(NamedRepr { name, child }) => Provenance::Named(
                NamedProvenance::new(name, child.build(resolver)?).map_err(BuildError::invalid)?,
            ),
            ProvenanceRepr::CallSite(CallSiteRepr { callee, caller }) => Provenance::CallSite(
                CallSiteProvenance::new(callee.build(resolver)?, caller.build(resolver)?),
            ),
            ProvenanceRepr::Fused(FusedRepr { sources, label }) => {
                let sources = sources
                    .into_iter()
                    .map(|source| source.build(resolver))
                    .collect::<Result<Vec<_>, _>>()?;
                Provenance::Fused(match label {
                    Some(label) => FusedProvenance::labelled(sources, label),
                    None => FusedProvenance::new(sources),
                })
            }
            ProvenanceRepr::Custom(foreign) => Provenance::Custom(resolver.resolve(&foreign)?),
        })
    }
}
