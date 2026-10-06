//! Two downstream kinds of `fhy_core.search_space`, registered with the
//! binding's kind registry: the template of a product's own `Variable` and
//! `Alternative` kinds.
//!
//! - `TiledVariable` (kind `example.tiled_variable`) is a variable with
//!   index symbols, references compared through its hooks, shaped like
//!   MOGA-VM's `ArrayTileKnob`.
//! - `AxisAlternative` (kind `example.axis_alternative`) is an alternative
//!   that binds its axes, shaped like MOGA-VM's `RealizationOption`.
//!
//! Each writes its foreign part as JSON text of its own shape and resolves
//! it back. `KindRegistrar` registers kinds again on purpose, so the tests
//! see the registry's refusals.

use std::borrow::Cow;

use pyo3::prelude::*;
use pyo3::types::{PyTuple, PyType};
use serde::{Deserialize, Serialize};

use fhy_core::constraint::{CustomConstraint, OpaqueValue};
use fhy_core::foreign::{BoxError, Foreign, ForeignError, ForeignPart, NoForeign, Part, Resolve};
use fhy_core::identifier::Identifier;
use fhy_core::param::wire::ParamData;
use fhy_core::param::{CustomDomain, Param, ParamContext};
use fhy_core::search_space::wire::VariableData;
use fhy_core::search_space::{Alternative, Variable};
use fhy_core::solver::Solver;
use fhy_core::term::AlphaRenaming;
use fhy_core_py::convert;

/// The kind of [`TiledVariable`].
const TILED_VARIABLE: &str = "example.tiled_variable";

/// The kind of [`AxisAlternative`].
const AXIS_ALTERNATIVE: &str = "example.axis_alternative";

/// A variable over a param, with index symbols its hooks compare.
#[derive(Debug)]
struct TiledVariable {
    name: Identifier,
    param: Param,
    index_symbols: Vec<Identifier>,
}

/// The foreign data of a [`TiledVariable`].
#[derive(Serialize, Deserialize)]
struct TiledVariableData {
    identifier: Identifier,
    param: ParamData,
    index_symbols: Vec<Identifier>,
}

/// Return the foreign error of a part of `type_id` that failed.
fn failed(type_id: &str, error: impl std::error::Error + Send + Sync + 'static) -> ForeignError {
    ForeignError::Failed {
        type_id: type_id.to_owned(),
        source: Box::new(error),
    }
}

impl ForeignPart for TiledVariable {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed("TiledVariable")
    }

    fn to_foreign(&self) -> Result<Foreign, ForeignError> {
        let data = TiledVariableData {
            identifier: self.name.clone(),
            param: ParamData::of(&self.param)?,
            index_symbols: self.index_symbols.clone(),
        };
        let text = serde_json::to_string(&data).map_err(|error| failed(TILED_VARIABLE, error))?;
        Ok(Foreign::new(TILED_VARIABLE, text))
    }
}

impl Variable for TiledVariable {
    fn kind(&self) -> Cow<'_, str> {
        Cow::Borrowed(TILED_VARIABLE)
    }

    fn name(&self) -> &Identifier {
        &self.name
    }

    fn param(&self) -> &Param {
        &self.param
    }

    fn is_extension_structurally_equivalent(&self, other: &dyn Variable) -> Result<bool, BoxError> {
        let Some(other) = other.as_any().downcast_ref::<Self>() else {
            return Ok(false);
        };
        Ok(self.index_symbols == other.index_symbols)
    }

    fn is_extension_alpha_equivalent_under(
        &self,
        other: &dyn Variable,
        renaming: &AlphaRenaming,
    ) -> Result<bool, BoxError> {
        let Some(other) = other.as_any().downcast_ref::<Self>() else {
            return Ok(false);
        };
        Ok(self.index_symbols.len() == other.index_symbols.len()
            && self
                .index_symbols
                .iter()
                .zip(&other.index_symbols)
                .all(|(left, right)| renaming.is_corresponding(left, right)))
    }
}

/// An alternative that binds its axes.
#[derive(Debug)]
struct AxisAlternative {
    name: Identifier,
    variables: Vec<Part<dyn Variable>>,
    axes: Vec<Identifier>,
}

/// The foreign data of an [`AxisAlternative`].
#[derive(Serialize, Deserialize)]
struct AxisAlternativeData {
    identifier: Identifier,
    variables: Vec<VariableData>,
    axes: Vec<Identifier>,
}

impl ForeignPart for AxisAlternative {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed("AxisAlternative")
    }

    fn to_foreign(&self) -> Result<Foreign, ForeignError> {
        let data = AxisAlternativeData {
            identifier: self.name.clone(),
            variables: self
                .variables
                .iter()
                .map(VariableData::of)
                .collect::<Result<_, _>>()?,
            axes: self.axes.clone(),
        };
        let text = serde_json::to_string(&data).map_err(|error| failed(AXIS_ALTERNATIVE, error))?;
        Ok(Foreign::new(AXIS_ALTERNATIVE, text))
    }
}

impl Alternative for AxisAlternative {
    fn kind(&self) -> Cow<'_, str> {
        Cow::Borrowed(AXIS_ALTERNATIVE)
    }

    fn name(&self) -> &Identifier {
        &self.name
    }

    fn variables(&self) -> &[Part<dyn Variable>] {
        &self.variables
    }

    fn bound_identifiers(&self) -> Result<Vec<Identifier>, BoxError> {
        Ok(self.axes.clone())
    }
}

/// The resolver of the foreign parts the two kinds' data holds: their own
/// kinds, and nothing else.
struct ExampleResolver;

impl Resolve<Part<dyn Variable>> for ExampleResolver {
    fn resolve(&self, foreign: &Foreign) -> Result<Part<dyn Variable>, ForeignError> {
        if foreign.type_id() == TILED_VARIABLE {
            resolve_tiled_variable(foreign)
        } else {
            NoForeign.resolve(foreign)
        }
    }
}

impl Resolve<Part<dyn Alternative>> for ExampleResolver {
    fn resolve(&self, foreign: &Foreign) -> Result<Part<dyn Alternative>, ForeignError> {
        if foreign.type_id() == AXIS_ALTERNATIVE {
            resolve_axis_alternative(foreign)
        } else {
            NoForeign.resolve(foreign)
        }
    }
}

impl Resolve<Part<dyn CustomDomain>> for ExampleResolver {
    fn resolve(&self, foreign: &Foreign) -> Result<Part<dyn CustomDomain>, ForeignError> {
        NoForeign.resolve(foreign)
    }
}

impl Resolve<Part<dyn CustomConstraint>> for ExampleResolver {
    fn resolve(&self, foreign: &Foreign) -> Result<Part<dyn CustomConstraint>, ForeignError> {
        NoForeign.resolve(foreign)
    }
}

impl Resolve<Part<dyn OpaqueValue>> for ExampleResolver {
    fn resolve(&self, foreign: &Foreign) -> Result<Part<dyn OpaqueValue>, ForeignError> {
        NoForeign.resolve(foreign)
    }
}

/// Resolve a foreign `TiledVariable`.
fn resolve_tiled_variable(foreign: &Foreign) -> Result<Part<dyn Variable>, ForeignError> {
    let data: TiledVariableData =
        serde_json::from_str(foreign.data()).map_err(|error| failed(TILED_VARIABLE, error))?;
    let solver = Solver::new();
    let param = data
        .param
        .build(&ExampleResolver, &ParamContext::new(&solver))
        .map_err(|error| failed(TILED_VARIABLE, error))?;
    Ok(Part::new(TiledVariable {
        name: data.identifier,
        param,
        index_symbols: data.index_symbols,
    }))
}

/// Resolve a foreign `AxisAlternative`.
fn resolve_axis_alternative(foreign: &Foreign) -> Result<Part<dyn Alternative>, ForeignError> {
    let data: AxisAlternativeData =
        serde_json::from_str(foreign.data()).map_err(|error| failed(AXIS_ALTERNATIVE, error))?;
    let solver = Solver::new();
    let context = ParamContext::new(&solver);
    let variables = data
        .variables
        .into_iter()
        .map(|variable| variable.build(&ExampleResolver, &context))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| failed(AXIS_ALTERNATIVE, error))?;
    Ok(Part::new(AxisAlternative {
        name: data.identifier,
        variables,
        axes: data.axes,
    }))
}

/// Return the identifiers of the Python iterable `objects`.
fn read_identifiers(objects: &Bound<'_, PyAny>) -> PyResult<Vec<Identifier>> {
    objects
        .try_iter()?
        .map(|object| convert::identifier_from_python(&object?))
        .collect()
}

/// Return the tuple of the Python objects of `identifiers`.
fn identifiers_to_python<'py>(
    py: Python<'py>,
    identifiers: &[Identifier],
) -> PyResult<Bound<'py, PyTuple>> {
    let objects = identifiers
        .iter()
        .map(|identifier| convert::identifier_to_python(py, identifier))
        .collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(py, objects)
}

/// The Python class of [`TiledVariable`].
#[pyclass(frozen, module = "fhy_example_aggregate", name = "TiledVariable")]
struct PyTiledVariable {
    variable: Part<dyn Variable>,
}

impl PyTiledVariable {
    /// Return the variable.
    fn tiled(&self) -> &TiledVariable {
        self.variable
            .get()
            .as_any()
            .downcast_ref::<TiledVariable>()
            .expect("a PyTiledVariable holds a TiledVariable")
    }
}

#[pymethods]
impl PyTiledVariable {
    #[new]
    fn new(
        param: &Bound<'_, PyAny>,
        index_symbols: &Bound<'_, PyAny>,
        name: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        Ok(Self {
            variable: Part::new(TiledVariable {
                name: convert::identifier_from_python(name)?,
                param: convert::param_from_python(param)?,
                index_symbols: read_identifiers(index_symbols)?,
            }),
        })
    }

    #[getter]
    fn name<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        convert::identifier_to_python(py, &self.tiled().name)
    }

    #[getter]
    fn index_symbols<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        identifiers_to_python(py, &self.tiled().index_symbols)
    }

    #[getter]
    fn kind(&self) -> String {
        self.variable.get().kind().into_owned()
    }
}

/// Read a `TiledVariable` object.
fn tiled_variable_from_python(object: &Bound<'_, PyAny>) -> PyResult<Part<dyn Variable>> {
    Ok(object.cast::<PyTiledVariable>()?.get().variable.clone())
}

/// Build the object of a `TiledVariable` part.
fn tiled_variable_to_python<'py>(
    py: Python<'py>,
    variable: &Part<dyn Variable>,
) -> PyResult<Bound<'py, PyAny>> {
    Ok(Bound::new(
        py,
        PyTiledVariable {
            variable: variable.clone(),
        },
    )?
    .into_any())
}

/// The Python class of [`AxisAlternative`].
#[pyclass(frozen, module = "fhy_example_aggregate", name = "AxisAlternative")]
struct PyAxisAlternative {
    alternative: Part<dyn Alternative>,
}

impl PyAxisAlternative {
    /// Return the alternative.
    fn axis(&self) -> &AxisAlternative {
        self.alternative
            .get()
            .as_any()
            .downcast_ref::<AxisAlternative>()
            .expect("a PyAxisAlternative holds an AxisAlternative")
    }
}

#[pymethods]
impl PyAxisAlternative {
    #[new]
    fn new(
        axes: &Bound<'_, PyAny>,
        variables: &Bound<'_, PyAny>,
        name: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let variables = variables
            .try_iter()?
            .map(|variable| convert::search_space::variable_from_python(&variable?))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            alternative: Part::new(AxisAlternative {
                name: convert::identifier_from_python(name)?,
                variables,
                axes: read_identifiers(axes)?,
            }),
        })
    }

    #[getter]
    fn name<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        convert::identifier_to_python(py, &self.axis().name)
    }

    #[getter]
    fn axes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        identifiers_to_python(py, &self.axis().axes)
    }

    #[getter]
    fn variables<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let objects = self
            .axis()
            .variables
            .iter()
            .map(|variable| convert::search_space::variable_to_python(py, variable))
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, objects)
    }
}

/// Read an `AxisAlternative` object.
fn axis_alternative_from_python(object: &Bound<'_, PyAny>) -> PyResult<Part<dyn Alternative>> {
    Ok(object
        .cast::<PyAxisAlternative>()?
        .get()
        .alternative
        .clone())
}

/// Build the object of an `AxisAlternative` part.
fn axis_alternative_to_python<'py>(
    py: Python<'py>,
    alternative: &Part<dyn Alternative>,
) -> PyResult<Bound<'py, PyAny>> {
    Ok(Bound::new(
        py,
        PyAxisAlternative {
            alternative: alternative.clone(),
        },
    )?
    .into_any())
}

/// Registers the two kinds' functions under other kinds and classes, for
/// the tests of the registry's refusals.
#[pyclass(frozen, module = "fhy_example_aggregate", name = "KindRegistrar")]
struct PyKindRegistrar;

#[pymethods]
impl PyKindRegistrar {
    /// Register `cls` as the `Variable` kind `kind` of `module`, with
    /// `TiledVariable`'s functions.
    #[staticmethod]
    fn register_variable(
        module: &Bound<'_, PyModule>,
        kind: &str,
        cls: &Bound<'_, PyType>,
    ) -> PyResult<()> {
        convert::search_space::register_variable_kind(
            module,
            kind,
            cls,
            tiled_variable_from_python,
            tiled_variable_to_python,
            resolve_tiled_variable,
        )
    }

    /// Register `cls` as the `Alternative` kind `kind` of `module`, with
    /// `AxisAlternative`'s functions.
    #[staticmethod]
    fn register_alternative(
        module: &Bound<'_, PyModule>,
        kind: &str,
        cls: &Bound<'_, PyType>,
    ) -> PyResult<()> {
        convert::search_space::register_alternative_kind(
            module,
            kind,
            cls,
            axis_alternative_from_python,
            axis_alternative_to_python,
            resolve_axis_alternative,
        )
    }
}

/// Add the two kinds' classes to `module` and register the kinds.
///
/// # Errors
///
/// Raises whatever adding a class or registering a kind raises.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add_class::<PyTiledVariable>()?;
    module.add_class::<PyAxisAlternative>()?;
    module.add_class::<PyKindRegistrar>()?;
    PyKindRegistrar::register_variable(module, TILED_VARIABLE, &py.get_type::<PyTiledVariable>())?;
    PyKindRegistrar::register_alternative(
        module,
        AXIS_ALTERNATIVE,
        &py.get_type::<PyAxisAlternative>(),
    )
}
