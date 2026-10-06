//! The conversions of `fhy_core.search_space`'s objects, and the
//! registration of the `Variable` and `Alternative` kinds a downstream
//! Rust crate defines.
//!
//! # Reading and writing parts
//!
//! [`variable_from_python`] reads, in order: the public `Variable` itself,
//! as a [`PlainVariable`](fhy_core::search_space::PlainVariable); an
//! instance of a registered kind's class, through its `from_python`; and an
//! instance of a Python subclass of `Variable`, as an adapter that calls
//! its `extension_*` hooks. [`variable_to_python`] writes a plain variable
//! as a new public `Variable`, an adapter as its own instance, and a
//! registered kind's part through its `to_python`. The alternative
//! functions do the same for `Alternative`, and the choice functions read
//! and build a `Choice`. A downstream kind's own `from_python` and
//! `to_python` call these for the variables and choices it holds.
//!
//! # Registering a kind
//!
//! A downstream binding crate registers each kind once, from its
//! aggregate's `#[pymodule]`, after [`register`](crate::register):
//!
//! - `kind` is the type id the kind's parts write
//!   ([`Variable::kind`](fhy_core::search_space::Variable::kind)), and
//!   the type id a foreign part of the kind is resolved by;
//! - `class` is the kind's `#[pyclass]`, which becomes a virtual subclass
//!   of the public `Variable` (or `Alternative`) as soon as both exist;
//! - `from_python` reads an instance of `class`, `to_python` builds the
//!   object of a part of the kind, and `resolve` builds a part from its
//!   foreign part, the inverse of its `to_foreign`.
//!
//! The registry is append-only module state of the extension: a kind, or
//! a class, is registered once and never replaced.
//!
//! # Examples
//!
//! ```ignore
//! #[pymodule(name = "_product", gil_used = true)]
//! fn aggregate(module: &Bound<'_, PyModule>) -> PyResult<()> {
//!     fhy_core_py::register(module.py(), module)?;
//!     fhy_core_py::convert::search_space::register_variable_kind(
//!         module,
//!         "product.tiled_variable",
//!         &module.py().get_type::<PyTiledVariable>(),
//!         read_tiled_variable,
//!         tiled_variable_to_python,
//!         resolve_tiled_variable,
//!     )
//! }
//! ```

use pyo3::prelude::*;
use pyo3::types::PyType;

use fhy_core::foreign::{Foreign, ForeignError, Part};
use fhy_core::search_space::{Alternative, Choice, Variable};

/// Read an object of a registered `Variable` kind's class as its part.
pub type VariableFromPython = fn(&Bound<'_, PyAny>) -> PyResult<Part<dyn Variable>>;

/// Build the Python object of a part of a registered `Variable` kind.
pub type VariableToPython =
    for<'py> fn(Python<'py>, &Part<dyn Variable>) -> PyResult<Bound<'py, PyAny>>;

/// Resolve a foreign part of a registered `Variable` kind.
pub type VariableResolver = fn(&Foreign) -> Result<Part<dyn Variable>, ForeignError>;

/// Read an object of a registered `Alternative` kind's class as its part.
pub type AlternativeFromPython = fn(&Bound<'_, PyAny>) -> PyResult<Part<dyn Alternative>>;

/// Build the Python object of a part of a registered `Alternative` kind.
pub type AlternativeToPython =
    for<'py> fn(Python<'py>, &Part<dyn Alternative>) -> PyResult<Bound<'py, PyAny>>;

/// Resolve a foreign part of a registered `Alternative` kind.
pub type AlternativeResolver = fn(&Foreign) -> Result<Part<dyn Alternative>, ForeignError>;

/// Return the core part of the `Variable` object `object`.
///
/// # Errors
///
/// Raises `TypeError` for an object that is no `Variable`, and
/// `RuntimeError` for an instance of a Python subclass whose `__init__`
/// never called `Variable.__init__`.
pub fn variable_from_python(object: &Bound<'_, PyAny>) -> PyResult<Part<dyn Variable>> {
    crate::search_space::variable_from_python(object)
}

/// Return the Python object of `variable`.
///
/// # Errors
///
/// Raises `TypeError` for a part of an unregistered kind, and what
/// building the object raises.
pub fn variable_to_python<'py>(
    py: Python<'py>,
    variable: &Part<dyn Variable>,
) -> PyResult<Bound<'py, PyAny>> {
    crate::search_space::variable_to_python(py, variable)
}

/// Return the core part of the `Alternative` object `object`.
///
/// # Errors
///
/// Raises `TypeError` for an object that is no `Alternative`, and
/// `RuntimeError` for an instance of a Python subclass whose `__init__`
/// never called `Alternative.__init__`.
pub fn alternative_from_python(object: &Bound<'_, PyAny>) -> PyResult<Part<dyn Alternative>> {
    crate::search_space::alternative_from_python(object)
}

/// Return the Python object of `alternative`.
///
/// # Errors
///
/// Raises `TypeError` for a part of an unregistered kind, and what
/// building the object raises.
pub fn alternative_to_python<'py>(
    py: Python<'py>,
    alternative: &Part<dyn Alternative>,
) -> PyResult<Bound<'py, PyAny>> {
    crate::search_space::alternative_to_python(py, alternative)
}

/// Return the core choice of the `Choice` object `object`, sharing it.
///
/// # Errors
///
/// Raises `TypeError` for an object that is no `Choice`.
pub fn choice_from_python(object: &Bound<'_, PyAny>) -> PyResult<Choice> {
    crate::search_space::choice_from_python(object)
}

/// Return a new Python `Choice` of `choice`, over the objects of its
/// alternatives.
///
/// # Errors
///
/// Raises what building the object of a part raises.
pub fn choice_to_python<'py>(py: Python<'py>, choice: &Choice) -> PyResult<Bound<'py, PyAny>> {
    crate::search_space::choice_to_python(py, choice)
}

/// Register the `Variable` kind `kind` of the class `class` in the kind
/// registry of `module`, the module [`register`](crate::register) built.
///
/// # Errors
///
/// Raises `ValueError` if `kind` is the plain variable's kind or registered
/// already, or `class` is registered for a kind already; `RuntimeError` if
/// `module` holds no `fhy_core` binding; and what registering `class` as a
/// virtual subclass of the public `Variable` raises.
pub fn register_variable_kind(
    module: &Bound<'_, PyModule>,
    kind: &str,
    class: &Bound<'_, PyType>,
    from_python: VariableFromPython,
    to_python: VariableToPython,
    resolve: VariableResolver,
) -> PyResult<()> {
    crate::search_space::register_variable_kind(
        module,
        kind,
        class,
        from_python,
        to_python,
        resolve,
    )
}

/// Register the `Alternative` kind `kind` of the class `class` in the kind
/// registry of `module`, the module [`register`](crate::register) built.
///
/// # Errors
///
/// As [`register_variable_kind`], for the plain alternative's kind and the
/// public `Alternative`.
pub fn register_alternative_kind(
    module: &Bound<'_, PyModule>,
    kind: &str,
    class: &Bound<'_, PyType>,
    from_python: AlternativeFromPython,
    to_python: AlternativeToPython,
    resolve: AlternativeResolver,
) -> PyResult<()> {
    crate::search_space::register_alternative_kind(
        module,
        kind,
        class,
        from_python,
        to_python,
        resolve,
    )
}
