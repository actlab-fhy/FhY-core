//! `PyO3` class for the note kinds of [`fhy_core::diagnostic`]:
//! `fhy_core._rs.NoteKind`, the base of `fhy_core.diagnostic.NoteKind` on
//! the Rust backend.

use pyo3::prelude::*;

use fhy_core::diagnostic::NoteKind;

use crate::described_tag::define_described_tag_class;

define_described_tag_class! {
    /// Open, registry-backed classification of an explanatory note's role,
    /// backed by the canonical Rust [`NoteKind`].
    class PyNoteKind as "NoteKind";
    seed NoteKindSeed;
    tag NoteKind;
    extra_methods {
        /// Return the name hint of the kind's name, as `Note` renders it.
        fn __str__(&self) -> String {
            self.tag.name().name_hint().to_owned()
        }
    }
}
