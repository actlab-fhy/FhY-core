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
//!   ([`Variable::kind`]), and
//!   the type id a foreign part of the kind is resolved by;
//! - `class` is the kind's `#[pyclass]`, which becomes a virtual subclass
//!   of the public `Variable` (or `Alternative`) as soon as both exist;
//! - `from_python` reads an instance of `class`, `to_python` builds the
//!   object of a part of the kind, and `resolve` builds a part from its
//!   foreign part, the inverse of its `to_foreign`.
//!
//! `resolve` is handed the binding's resolver of the payload being decoded,
//! as a [`SearchSpaceResolver`], and the [`ParamContext`] it is decoded
//! under. A kind whose parts hold other parts (a variable, a choice, a
//! param, whose values or domain may be Python-defined) builds their wire
//! forms with these two, so each nested part is decoded as the binding
//! decodes its own: a Python subclass's variable, an opaque value or a
//! registered kind alike, under the same context.
//!
//! The registry is append-only module state of the extension: a kind, or
//! a class, is registered once and never replaced.
//!
//! A kind's type id is never also the type id of a class of Python's
//! serialization framework (`register_serializable`), since decoding asks
//! the kinds first and the class could not read its own payloads back.
//! Whichever registration comes second is refused: registering a kind
//! under a type id a Python class holds raises `ValueError`, and
//! `register_serializable` under a kind's type id raises
//! `SerializationError`. An aggregate registers its kinds before any
//! Python class is registered, so it is the Python class that is refused.
//!
//! # Oracles and step domains
//!
//! A downstream Rust oracle registers its `#[pyclass]` with
//! [`register_oracle_kind`] and a lease, which borrows the oracle out of an
//! instance for one call, so a run asks it with no Python call per step.
//! An oracle argument is read as one of `fhy_core`'s oracle classes, as an
//! instance of a registered oracle class, or as any object with a callable
//! `decide`. [`step_domain_from_python`] and [`step_domain_to_python`]
//! convert the domain objects (`ChoiceDomain`, `OrderDomain`,
//! `StridedDomain`), for a kind whose
//! [`search_domain`](Variable::search_domain) is written in Python.
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
//!
//! fn resolve_tiled_variable(
//!     foreign: &Foreign,
//!     parts: &dyn SearchSpaceResolver,
//!     context: &ParamContext<'_>,
//! ) -> Result<Part<dyn Variable>, ForeignError> {
//!     let data: TiledVariableData = serde_json::from_str(foreign.data())
//!         .map_err(|error| failed(foreign, error))?;
//!     let param = data
//!         .param
//!         .build(parts, context)
//!         .map_err(|error| failed(foreign, error))?;
//!     Ok(Part::new(TiledVariable::new(data.identifier, param)))
//! }
//! ```

use pyo3::prelude::*;
use pyo3::types::PyType;

use fhy_core::foreign::{Foreign, ForeignError, Part};
use fhy_core::param::ParamContext;
use fhy_core::search_space::wire::SearchSpaceResolver;
use fhy_core::search_space::{Alternative, Choice, SearchOracle, StepDomain, Variable};

/// Read an object of a registered `Variable` kind's class as its part.
pub type VariableFromPython = fn(&Bound<'_, PyAny>) -> PyResult<Part<dyn Variable>>;

/// Build the Python object of a part of a registered `Variable` kind.
pub type VariableToPython =
    for<'py> fn(Python<'py>, &Part<dyn Variable>) -> PyResult<Bound<'py, PyAny>>;

/// Resolve a foreign part of a registered `Variable` kind, the foreign
/// parts it holds resolved by the binding's resolver of the payload, under
/// the context the payload is decoded in.
pub type VariableResolver = fn(
    &Foreign,
    &dyn SearchSpaceResolver,
    &ParamContext<'_>,
) -> Result<Part<dyn Variable>, ForeignError>;

/// Read an object of a registered `Alternative` kind's class as its part.
pub type AlternativeFromPython = fn(&Bound<'_, PyAny>) -> PyResult<Part<dyn Alternative>>;

/// Build the Python object of a part of a registered `Alternative` kind.
pub type AlternativeToPython =
    for<'py> fn(Python<'py>, &Part<dyn Alternative>) -> PyResult<Bound<'py, PyAny>>;

/// Resolve a foreign part of a registered `Alternative` kind, the foreign
/// parts it holds resolved by the binding's resolver of the payload, under
/// the context the payload is decoded in.
pub type AlternativeResolver = fn(
    &Foreign,
    &dyn SearchSpaceResolver,
    &ParamContext<'_>,
) -> Result<Part<dyn Alternative>, ForeignError>;

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
    crate::search_space::choice_to_python(py, choice, crate::util::gc::Slots::default())
}

/// Register the `Variable` kind `kind` of the class `class` in the kind
/// registry of `module`, the module [`register`](crate::register) built.
///
/// # Errors
///
/// Raises `ValueError` if `kind` is the plain variable's kind, registered
/// already, or the type id a class of Python's serialization framework is
/// registered under (`register_serializable`), or `class` is registered for
/// a kind already; `RuntimeError` if `module` holds no `fhy_core` binding;
/// and what reading the framework's registry or registering `class` as a
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

/// Borrow the Rust oracle out of an instance of a registered oracle class,
/// for one call: typically a `PyRefMut` of the class in a type that
/// forwards [`SearchOracle::decide`] to it.
pub type OracleLease =
    for<'a, 'py> fn(&'a Bound<'py, PyAny>) -> PyResult<Box<dyn SearchOracle + 'a>>;

/// Register the oracle kind `kind` of the class `class`, whose instances
/// `lease` borrows the oracle out of, in the kind registry of `module`, the
/// module [`register`](crate::register) built.
///
/// # Errors
///
/// Raises `ValueError` if `kind` is registered already or `class` is
/// registered for a kind already, and `RuntimeError` if `module` holds no
/// `fhy_core` binding.
pub fn register_oracle_kind(
    module: &Bound<'_, PyModule>,
    kind: &str,
    class: &Bound<'_, PyType>,
    lease: OracleLease,
) -> PyResult<()> {
    crate::search_space::register_oracle_kind(module, kind, class, lease)
}

/// Return the core domain of the domain object `object`: a `ChoiceDomain`,
/// an `OrderDomain` or a `StridedDomain`.
///
/// # Errors
///
/// Raises `TypeError` for any other object.
pub fn step_domain_from_python(object: &Bound<'_, PyAny>) -> PyResult<StepDomain> {
    crate::search_space::step_domain_from_python(object)
}

/// Return a new domain object of the core domain `domain`.
///
/// # Errors
///
/// Raises what writing a value of the domain raises.
pub fn step_domain_to_python<'py>(
    py: Python<'py>,
    domain: &StepDomain,
) -> PyResult<Bound<'py, PyAny>> {
    crate::search_space::step_domain_to_python(py, domain)
}
