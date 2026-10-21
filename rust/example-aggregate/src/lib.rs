//! A test-only aggregate extension: the proof that a downstream binding
//! composes with `fhy_core`, and the template for one.
//!
//! The module registers all of `fhy_core`'s binding
//! ([`fhy_core_py::register`]) and two downstream *layers* on top of it, each
//! of which sets its own `__<layer>_version__` attribute:
//!
//! - `fhy_example` holds one extra class, `Tagger`, that takes and returns
//!   `fhy-core` values through [`fhy_core_py::convert`]: an `Identifier`,
//!   whose ids come from the one counter, and an interned `OpAttribute`,
//!   which lives in the one registry;
//! - `fhy_example_search` holds two downstream kinds of
//!   `fhy_core.search_space` (the `search_space` module) and a downstream
//!   oracle kind (the `oracle` module).
//!
//! The two layers stand for a chain of products, each a Python package with
//! its own Rust crate, built into one module. The distribution advertises both
//! in the entry-point group `fhy_core.native`, one entry point per layer, and
//! `fhy_core` checks the module reports each. It is a workspace member
//! so the gate builds it, `publish = false`, and no wheel ships it. The
//! Python test `tests/test_composed_extension.py` builds it, installs it as
//! the extension of a fresh interpreter through the `fhy_core.native` entry
//! point, and checks that the two parts share one copy of `fhy-core`.
//!
//! What a real downstream aggregate does the same way:
//!
//! 1. its `#[pymodule]` calls [`fhy_core_py::register`] first, then the
//!    registration function of each of its own crates (here the two
//!    per-layer functions `register_example` and `register_example_search`,
//!    which `register` calls in turn), each of which sets its layer's
//!    `__<layer>_version__`;
//! 2. its own classes name their module (`module = "..."`) as the package
//!    they belong to, so their qualified names do not depend on the native
//!    module's name;
//! 3. it is a top-level module that imports no `fhy_core` Python code when it
//!    is imported, because `fhy_core` imports it while `fhy_core` itself is
//!    being imported.

mod oracle;
mod search_space;

use fhy_core::identifier::Identifier;
use fhy_core::op_attribute::OpAttribute;
use fhy_core_py::convert;
use pyo3::prelude::*;

/// A class that takes and returns `fhy-core` values.
#[pyclass(frozen, module = "fhy_example_aggregate", name = "Tagger")]
struct PyTagger;

#[pymethods]
impl PyTagger {
    #[new]
    const fn new() -> Self {
        Self
    }

    /// Return the identifier with `identifier`'s id and name hint,
    /// converted to Rust and back.
    #[staticmethod]
    fn echo<'py>(py: Python<'py>, identifier: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        convert::identifier_to_python(py, &convert::identifier_from_python(identifier)?)
    }

    /// Return a new identifier named `identifier`'s name hint and `suffix`,
    /// with an id the Rust side draws from the process-global counter.
    #[staticmethod]
    fn derive<'py>(
        py: Python<'py>,
        identifier: &Bound<'py, PyAny>,
        suffix: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let source = convert::identifier_from_python(identifier)?;
        let derived = Identifier::new(&format!("{}{suffix}", source.name_hint()));
        convert::identifier_to_python(py, &derived)
    }

    /// Register the attribute `name` names, if it is not registered, and
    /// return its one Python object.
    #[staticmethod]
    fn tag<'py>(
        py: Python<'py>,
        name: &Bound<'py, PyAny>,
        description: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let attribute = OpAttribute::register(convert::identifier_from_python(name)?, description);
        convert::op_attribute_to_python(py, attribute)
    }

    /// Return whether `attribute` is the canonical `commutative` attribute,
    /// compared as the Rust registry knows it.
    #[staticmethod]
    fn is_commutative(attribute: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(convert::op_attribute_from_python(attribute)? == *OpAttribute::commutative())
    }
}

/// The attribute that marks the layer `fhy_example` in a module, set to
/// this crate's version.
const EXAMPLE_VERSION_ATTRIBUTE: &str = "__fhy_example_version__";

/// The attribute that marks the layer `fhy_example_search` in a module.
const EXAMPLE_SEARCH_VERSION_ATTRIBUTE: &str = "__fhy_example_search_version__";

/// Add the layer `fhy_example` to `module`: the `Tagger` class, and the
/// layer's version.
///
/// # Errors
///
/// Raises whatever adding a class or the version raises.
fn register_example(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyTagger>()?;
    module.add(EXAMPLE_VERSION_ATTRIBUTE, env!("CARGO_PKG_VERSION"))
}

/// Add the layer `fhy_example_search` to `module`: the kinds of
/// `fhy_core.search_space` (a search-space kind and an oracle kind), and the
/// layer's version.
///
/// # Errors
///
/// Raises whatever adding a class, registering a kind or adding the version
/// raises.
fn register_example_search(module: &Bound<'_, PyModule>) -> PyResult<()> {
    search_space::register(module)?;
    oracle::register(module)?;
    module.add(EXAMPLE_SEARCH_VERSION_ATTRIBUTE, env!("CARGO_PKG_VERSION"))
}

/// Add this crate's two layers to `module`, one after the other, each setting
/// its own `__<layer>_version__`.
///
/// # Errors
///
/// Raises whatever registering a layer raises.
fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    register_example(module)?;
    register_example_search(module)
}

/// The aggregate: `fhy_core`'s binding and this crate's, in one module.
#[pymodule(name = "_fhy_example_aggregate", gil_used = true)]
fn aggregate(module: &Bound<'_, PyModule>) -> PyResult<()> {
    fhy_core_py::register(module.py(), module)?;
    register(module)
}
