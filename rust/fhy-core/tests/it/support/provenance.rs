//! Shared builders for the provenance and diagnostic tests.

use fhy_core::provenance::{FileProvenance, NamedProvenance, Provenance, Span};

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
    Provenance::Named(NamedProvenance::try_new(name, child).expect("name is non-empty"))
}
