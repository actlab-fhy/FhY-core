//! `fhy_core._rs.AnalysisBase`, the base of the Python `Analysis` ABC, and
//! `fhy_core._rs.PreservedAnalyses`, backed by the core's
//! [`PreservedAnalyses`].
//!
//! A Python analysis is named by the `Identifier` its class's
//! `get_analysis_name()` returns, and the core names it by
//! [`AnalysisId::of_identifier`] of that identifier, so a preserved set
//! built in Python preserves the results the run's cache holds under it.

use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBool, PyDict, PyFrozenSet, PyTuple, PyType};

use fhy_core::pass::{AnalysisId, PreservedAnalyses};

use crate::dataclass::{
    OptionalArgument, build_argument_type_error, compare_as_dataclass, format_dataclass_repr,
    hash_value,
};
use crate::frozen::build_frozen_mutation_error;
use crate::identifier::{identifier_to_python, restore_identifier};
use crate::public_class::PublicClass;
use crate::python::Seed;

use super::compiler_pass::refuse_unused_arguments;

/// The base of the Python `Analysis` ABC.
///
/// It holds nothing: a Python analysis runs in Python, and the binding
/// caches its result per node under the analysis's identifier. The class
/// marks what the binding accepts as an analysis type.
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "AnalysisBase")]
pub(crate) struct PyAnalysisBase;

#[pymethods]
impl PyAnalysisBase {
    /// Accept any arguments, so a subclass's `__init__` takes its own.
    ///
    /// Raises `TypeError` for arguments a subclass without an `__init__`
    /// of its own was given.
    #[new]
    #[classmethod]
    #[pyo3(signature = (*args, **kwargs))]
    fn new(
        cls: &Bound<'_, PyType>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        refuse_unused_arguments(cls, args, kwargs)?;
        Ok(Self)
    }

    /// Return no constructor arguments, so a subclass instance pickles
    /// through its `__dict__`.
    fn __getnewargs__<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyTuple> {
        PyTuple::empty(slf.py())
    }
}

/// Return the analysis id of the Python `Identifier` `name`.
///
/// # Errors
///
/// Raises `TypeError` naming `owner` and `field` if `name` is not an
/// `Identifier`.
pub(super) fn analysis_id_of(
    name: &Bound<'_, PyAny>,
    owner: &str,
    field: &str,
) -> PyResult<AnalysisId> {
    Ok(AnalysisId::of_identifier(&restore_identifier(
        name, owner, field,
    )?))
}

/// The set of analyses a pass run preserves, keyed by analysis name.
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "PreservedAnalyses")]
pub(crate) struct PyPreservedAnalyses {
    preserved: PreservedAnalyses,
    /// The names, as the frozenset of Python identifiers the set was built
    /// from, or built on first read for a set the core returned.
    names: PyOnceLock<Py<PyFrozenSet>>,
}

impl PyPreservedAnalyses {
    /// Return the public Python class registered for this class.
    fn public_class() -> &'static PublicClass {
        static PUBLIC_CLASS: PublicClass = PublicClass::new("PreservedAnalyses");
        &PUBLIC_CLASS
    }

    /// Return the core's set.
    pub(super) fn preserved(&self) -> &PreservedAnalyses {
        &self.preserved
    }

    /// Return a new object of `cls` holding `preserved` and `names`.
    fn build<'py>(
        cls: &Bound<'py, PyType>,
        preserved: PreservedAnalyses,
        names: Option<Bound<'py, PyFrozenSet>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = cls.py();
        let seed = Bound::new(
            py,
            PreservedAnalysesSeed(Seed::new((preserved, names.map(Bound::unbind)))),
        )?;
        py.get_type::<Self>()
            .call_method1(intern!(py, "__new__"), (cls, seed))
    }

    /// Return the frozenset of the Python identifiers the set names.
    fn names<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyFrozenSet>> {
        let names = self.names.get_or_try_init(py, || -> PyResult<_> {
            let identifiers = self
                .preserved
                .preserved_ids()
                .filter_map(AnalysisId::identifier)
                .map(|identifier| identifier_to_python(py, identifier))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyFrozenSet::new(py, identifiers)?.unbind())
        })?;
        Ok(names.bind(py).clone())
    }
}

/// Return the public `PreservedAnalyses` object of the core's `preserved`.
pub(super) fn preserved_to_python<'py>(
    py: Python<'py>,
    preserved: &PreservedAnalyses,
) -> PyResult<Bound<'py, PyAny>> {
    let cls = PyPreservedAnalyses::public_class().get(py)?;
    PyPreservedAnalyses::build(cls, preserved.clone(), None)
}

/// The contents of a set the binding builds, handed to `__new__`. Not
/// exported.
#[pyclass(frozen, module = "fhy_core._rs", name = "_PreservedAnalysesSeed")]
struct PreservedAnalysesSeed(Seed<(PreservedAnalyses, Option<Py<PyFrozenSet>>)>);

#[pymethods]
impl PyPreservedAnalyses {
    /// Create the set preserving every analysis if `preserve_all` holds,
    /// and otherwise the analyses named by the identifiers
    /// `analysis_names`.
    ///
    /// Raises `ValueError` if both are given, and `TypeError` if
    /// `preserve_all` is not a `bool` or a name is not an `Identifier`.
    #[new]
    #[pyo3(signature = (
        preserve_all = OptionalArgument::Omitted,
        analysis_names = OptionalArgument::Omitted,
    ))]
    fn new(
        py: Python<'_>,
        preserve_all: OptionalArgument<'_>,
        analysis_names: OptionalArgument<'_>,
    ) -> PyResult<Self> {
        let preserve_all = match preserve_all {
            OptionalArgument::Omitted => PyBool::new(py, false).to_owned().into_any(),
            OptionalArgument::Given(value) => value,
        };
        if let Ok(seed) = preserve_all.cast::<PreservedAnalysesSeed>() {
            let (preserved, seed_names) = seed.get().0.take("a preserved-analyses")?;
            let names = PyOnceLock::new();
            if let Some(seed_names) = seed_names {
                names.get_or_init(py, || seed_names);
            }
            return Ok(Self { preserved, names });
        }
        let Ok(preserve_all) = preserve_all.cast::<PyBool>() else {
            return Err(build_argument_type_error(
                "PreservedAnalyses",
                "preserve_all",
                "a bool",
                &preserve_all,
            )?);
        };
        let names = match analysis_names {
            OptionalArgument::Given(names) => {
                PyFrozenSet::new(py, names.try_iter()?.collect::<PyResult<Vec<_>>>()?)?
            }
            OptionalArgument::Omitted => PyFrozenSet::empty(py)?,
        };
        let preserved = if preserve_all.is_true() {
            if !names.is_empty() {
                return Err(PyValueError::new_err(
                    "PreservedAnalyses: analysis_names must be empty when preserve_all=True.",
                ));
            }
            PreservedAnalyses::all()
        } else {
            let mut preserved = PreservedAnalyses::none();
            for name in &names {
                preserved = preserved.preserve_id(analysis_id_of(
                    &name,
                    "PreservedAnalyses",
                    "analysis_names",
                )?);
            }
            preserved
        };
        let stored = PyOnceLock::new();
        stored.get_or_init(py, || names.unbind());
        Ok(Self {
            preserved,
            names: stored,
        })
    }

    /// Whether every analysis is preserved.
    #[getter]
    fn preserve_all(&self) -> bool {
        self.preserved.preserves_all()
    }

    /// The names of the preserved analyses: empty when every analysis is
    /// preserved.
    #[getter]
    fn analysis_names<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyFrozenSet>> {
        self.names(py)
    }

    /// Return the set preserving every analysis.
    #[classmethod]
    fn all<'py>(cls: &Bound<'py, PyType>) -> PyResult<Bound<'py, PyAny>> {
        Self::build(cls, PreservedAnalyses::all(), None)
    }

    /// Return the set preserving no analysis.
    #[classmethod]
    fn none<'py>(cls: &Bound<'py, PyType>) -> PyResult<Bound<'py, PyAny>> {
        Self::build(cls, PreservedAnalyses::none(), None)
    }

    /// Return the set that also preserves the analysis `analysis_name`: this
    /// set itself when it preserves it already.
    ///
    /// Raises `TypeError` if `analysis_name` is not an `Identifier`.
    fn preserve<'py>(
        slf: &Bound<'py, Self>,
        analysis_name: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let this = slf.get();
        let id = analysis_id_of(analysis_name, "PreservedAnalyses.preserve", "analysis_name")?;
        if this.preserved.is_id_preserved(&id) {
            return Ok(slf.clone().into_any());
        }
        let names = this.names(py)?;
        let mut items: Vec<Bound<'py, PyAny>> = names.iter().collect();
        items.push(analysis_name.clone());
        let names = PyFrozenSet::new(py, items)?;
        Self::build(
            &slf.get_type(),
            this.preserved.clone().preserve_id(id),
            Some(names),
        )
    }

    /// Return whether the analysis `analysis_name` is preserved.
    ///
    /// Raises `TypeError` if `analysis_name` is not an `Identifier`.
    fn is_preserved(&self, analysis_name: &Bound<'_, PyAny>) -> PyResult<bool> {
        if self.preserved.preserves_all() {
            return Ok(true);
        }
        let id = analysis_id_of(
            analysis_name,
            "PreservedAnalyses.is_preserved",
            "analysis_name",
        )?;
        Ok(self.preserved.is_id_preserved(&id))
    }

    /// Always true: preserved sets are immutable.
    #[getter]
    fn is_frozen(_slf: &Bound<'_, Self>) -> bool {
        true
    }

    /// Do nothing: preserved sets are always frozen.
    fn freeze(_slf: &Bound<'_, Self>) {}

    /// Do nothing: preserved sets are always frozen, and mutating one
    /// raises.
    fn assert_frozen(_slf: &Bound<'_, Self>) {}

    fn __eq__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        compare_as_dataclass(slf, other, |this, other| {
            Ok(this.preserved == other.preserved)
        })
    }

    fn __hash__(&self) -> u64 {
        hash_value(&self.preserved)
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let this = slf.get();
        let preserve_all = PyBool::new(py, this.preserved.preserves_all())
            .to_owned()
            .into_any();
        let names = this.names(py)?.into_any();
        format_dataclass_repr(
            &slf.get_type(),
            &[("preserve_all", &preserve_all), ("analysis_names", &names)],
        )
    }

    fn __setattr__(slf: &Bound<'_, Self>, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let _ = value;
        Err(build_frozen_mutation_error(slf, "modify", name)?)
    }

    fn __delattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
        Err(build_frozen_mutation_error(slf, "delete", name)?)
    }

    /// Pickle as a constructor call of the set's class with its fields.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyType>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        let this = slf.get();
        let preserve_all = PyBool::new(py, this.preserved.preserves_all())
            .to_owned()
            .into_any();
        let arguments = PyTuple::new(py, [preserve_all, this.names(py)?.into_any()])?;
        Ok((slf.get_type(), arguments))
    }

    /// Register `cls` as the public class.
    ///
    /// Raises `RuntimeError` if another public class is registered.
    #[classmethod]
    fn _register_public_class(cls: &Bound<'_, PyType>) -> PyResult<()> {
        Self::public_class().register(cls)
    }
}
