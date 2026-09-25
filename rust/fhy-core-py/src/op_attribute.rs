//! `PyO3` class for [`fhy_core::op_attribute`]: `fhy_core._rs.OpAttribute`,
//! the base of `fhy_core.op_attribute.OpAttribute` on the Rust backend.

use pyo3::prelude::*;

use fhy_core::op_attribute::OpAttribute;

use crate::described_tag::define_described_tag_class;

define_described_tag_class! {
    /// Open semantic tag attached to a compiler operation, backed by the
    /// canonical Rust [`OpAttribute`].
    class PyOpAttribute as "OpAttribute";
    seed OpAttributeSeed;
    tag OpAttribute;
    extra_methods {}
}
