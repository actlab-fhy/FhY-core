//! The results and records of pass runs:
//! `PassResult`, `PassRunRecord`, `FixpointIterationRecord`,
//! `FixpointGroupRecord`, `PassManagerResult` and `ValidatorRecord`.
//!
//! Each holds the Python objects of its fields, checked when it is built,
//! and compares, hashes, prints and pickles as a frozen dataclass does:
//! equality and the hash follow the tuple of its fields, the repr is the
//! dataclass one, and a pickle is a call of its class with its fields. The
//! binding builds them through their public classes.

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::types::{PyBool, PyInt, PyString, PyTuple, PyType};

use crate::diagnostic::borrow_python_diagnostic;
use crate::identifier::read_identifier_id;
use crate::util::dataclass::{
    OptionalArgument, build_argument_type_error, collect_tuple, compare_as_dataclass,
    format_dataclass_repr,
};
use crate::util::frozen::{refuse_attribute_assignment, refuse_attribute_deletion};
use crate::util::public_class::PublicClass;

use super::analysis::{PyPreservedAnalyses, preserved_to_python};

/// Define a record class whose fields are Python objects, with the
/// dataclass behavior every record shares, and the constructor and extra
/// methods given in `methods`.
macro_rules! define_record_class {
    (
        $(#[$meta:meta])*
        class $class:ident as $py_name:literal {
            $( $(#[$field_meta:meta])* $field:ident ),+ $(,)?
        }
        methods { $($methods:tt)* }
    ) => {
        $(#[$meta])*
        #[pyclass(subclass, frozen, module = "fhy_core._rs", name = $py_name)]
        pub(crate) struct $class {
            $( $(#[$field_meta])* #[pyo3(get)] $field: Py<PyAny>, )+
        }

        impl $class {
            /// Return the public Python class registered for this class.
            pub(super) fn public_class() -> &'static PublicClass {
                static PUBLIC_CLASS: PublicClass = PublicClass::new($py_name);
                &PUBLIC_CLASS
            }

            /// Return the names and objects of the fields, in constructor
            /// order.
            fn field_values<'py>(&self, py: Python<'py>) -> Vec<(&'static str, Bound<'py, PyAny>)> {
                vec![$( (stringify!($field), self.$field.bind(py).clone()) ),+]
            }

            /// Return the tuple of the fields.
            fn field_tuple<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
                PyTuple::new(py, self.field_values(py).into_iter().map(|(_, value)| value))
            }
        }

        #[pymethods]
        impl $class {
            $($methods)*

            /// Visit the fields, for the cycle collector.
            fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
                $( visit.call(&self.$field)?; )+
                Ok(())
            }

            /// Always true: records are immutable.
            #[getter]
            const fn is_frozen(_slf: &Bound<'_, Self>) -> bool {
                true
            }

            /// Do nothing: records are always frozen.
            const fn freeze(_slf: &Bound<'_, Self>) {}

            /// Do nothing: records are always frozen, and mutating one
            /// raises.
            const fn assert_frozen(_slf: &Bound<'_, Self>) {}

            /// Compare the tuples of the fields, for an object of exactly
            /// the same class.
            fn __eq__<'py>(
                slf: &Bound<'py, Self>,
                other: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyAny>> {
                let py = slf.py();
                compare_as_dataclass(slf, other, |this, other| {
                    this.field_tuple(py)?.eq(other.field_tuple(py)?)
                })
            }

            /// Hash the tuple of the fields; raises `TypeError` for an
            /// unhashable field.
            fn __hash__(&self, py: Python<'_>) -> PyResult<isize> {
                self.field_tuple(py)?.hash()
            }

            fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
                let fields = slf.get().field_values(slf.py());
                let fields: Vec<(&str, &Bound<'_, PyAny>)> =
                    fields.iter().map(|(name, value)| (*name, value)).collect();
                format_dataclass_repr(&slf.get_type(), &fields)
            }

            fn __setattr__(
                slf: &Bound<'_, Self>,
                name: &str,
                _value: &Bound<'_, PyAny>,
            ) -> PyResult<()> {
                refuse_attribute_assignment(slf, name)
            }

            fn __delattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
                refuse_attribute_deletion(slf, name)
            }

            /// Pickle as a constructor call of the record's class with its
            /// fields.
            fn __reduce__<'py>(
                slf: &Bound<'py, Self>,
            ) -> PyResult<(Bound<'py, PyType>, Bound<'py, PyTuple>)> {
                Ok((slf.get_type(), slf.get().field_tuple(slf.py())?))
            }

            /// Register `cls` as the public class.
            ///
            /// Raises `RuntimeError` if another public class is registered.
            #[classmethod]
            fn _register_public_class(cls: &Bound<'_, PyType>) -> PyResult<()> {
                Self::public_class().register(cls)
            }
        }
    };
}

/// Return `value`, which must be a `bool`.
///
/// # Errors
///
/// Raises `TypeError` naming `owner` and `field` otherwise.
fn read_bool(value: &Bound<'_, PyAny>, owner: &str, field: &str) -> PyResult<Py<PyAny>> {
    if value.is_instance_of::<PyBool>() {
        Ok(value.clone().unbind())
    } else {
        Err(build_argument_type_error(owner, field, "a bool", value)?)
    }
}

/// Return `value`, a `bool` defaulting to `False` when omitted.
fn read_optional_bool(
    py: Python<'_>,
    value: OptionalArgument<'_>,
    owner: &str,
    field: &str,
) -> PyResult<Py<PyAny>> {
    match value {
        OptionalArgument::Omitted => Ok(PyBool::new(py, false).to_owned().into_any().unbind()),
        OptionalArgument::Given(value) => read_bool(&value, owner, field),
    }
}

/// Return the items of the iterable `values` as a tuple, each of which
/// `is_valid` accepts.
///
/// # Errors
///
/// Raises `TypeError` naming `owner`, `field` and `expected` for an item it
/// refuses, and whatever iterating `values` raises.
fn read_tuple_of(
    values: &Bound<'_, PyAny>,
    owner: &str,
    field: &str,
    expected: &str,
    is_valid: impl Fn(&Bound<'_, PyAny>) -> bool,
) -> PyResult<Py<PyAny>> {
    let values = collect_tuple(values)?;
    for value in values.iter() {
        if !is_valid(&value) {
            return Err(build_argument_type_error(owner, field, expected, &value)?);
        }
    }
    Ok(values.into_any().unbind())
}

/// Return the iterable `values` of `Diagnostic`s as a tuple.
fn read_diagnostics(values: &Bound<'_, PyAny>, owner: &str) -> PyResult<Py<PyAny>> {
    read_tuple_of(
        values,
        owner,
        "diagnostics",
        "Diagnostic instances",
        |value| borrow_python_diagnostic(value).is_some(),
    )
}

/// Return `value`, which must be a `PreservedAnalyses`.
fn read_preserved(value: &Bound<'_, PyAny>, owner: &str) -> PyResult<Py<PyAny>> {
    if value.is_instance_of::<PyPreservedAnalyses>() {
        Ok(value.clone().unbind())
    } else {
        Err(build_argument_type_error(
            owner,
            "preserved_analyses",
            "a PreservedAnalyses",
            value,
        )?)
    }
}

/// Return `value`, which must be a `str`.
fn read_name(value: &Bound<'_, PyAny>, owner: &str, field: &str) -> PyResult<Py<PyAny>> {
    if value.is_instance_of::<PyString>() {
        Ok(value.clone().unbind())
    } else {
        Err(build_argument_type_error(owner, field, "a str", value)?)
    }
}

define_record_class! {
    /// The result of a pass execution.
    class PyPassResult as "PassResult" {
        output,
        changed,
        diagnostics,
        preserved_analyses,
        skipped,
    }
    methods {
        /// Create the result of a run that produced `output`, changed the
        /// IR if `changed` holds, emitted `diagnostics` and preserved
        /// `preserved_analyses`, by default none; `skipped` holds for a run
        /// the pass skipped.
        ///
        /// Raises `TypeError` for a `changed` or `skipped` that is not a
        /// `bool`, a diagnostic that is not a `Diagnostic`, or a
        /// `preserved_analyses` that is not a `PreservedAnalyses`.
        #[new]
        #[pyo3(signature = (
            output,
            changed,
            diagnostics = OptionalArgument::Omitted,
            preserved_analyses = OptionalArgument::Omitted,
            skipped = OptionalArgument::Omitted,
        ))]
        fn new(
            py: Python<'_>,
            output: Bound<'_, PyAny>,
            changed: &Bound<'_, PyAny>,
            diagnostics: OptionalArgument<'_>,
            preserved_analyses: OptionalArgument<'_>,
            skipped: OptionalArgument<'_>,
        ) -> PyResult<Self> {
            let diagnostics = match diagnostics {
                OptionalArgument::Omitted => PyTuple::empty(py).into_any().unbind(),
                OptionalArgument::Given(values) => read_diagnostics(&values, "PassResult")?,
            };
            let preserved_analyses = match preserved_analyses {
                OptionalArgument::Omitted => {
                    preserved_to_python(py, &fhy_core::pass::PreservedAnalyses::none())?.unbind()
                }
                OptionalArgument::Given(value) => read_preserved(&value, "PassResult")?,
            };
            Ok(Self {
                output: output.unbind(),
                changed: read_bool(changed, "PassResult", "changed")?,
                diagnostics,
                preserved_analyses,
                skipped: read_optional_bool(py, skipped, "PassResult", "skipped")?,
            })
        }
    }
}

impl PyPassResult {
    /// Return a new public `PassResult` of these fields.
    pub(super) fn build<'py>(
        py: Python<'py>,
        output: Bound<'py, PyAny>,
        changed: bool,
        diagnostics: Bound<'py, PyTuple>,
        preserved_analyses: Bound<'py, PyAny>,
        skipped: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        Self::public_class().get(py)?.call1((
            output,
            changed,
            diagnostics,
            preserved_analyses,
            skipped,
        ))
    }
}

define_record_class! {
    /// The record of one pass run in a pipeline.
    class PyPassRunRecord as "PassRunRecord" {
        pass_name,
        changed,
        diagnostics,
        preserved_analyses,
        skipped,
    }
    methods {
        /// Create the record of a run of the pass `pass_name` that changed
        /// the IR if `changed` holds, emitted `diagnostics` and preserved
        /// `preserved_analyses`; `skipped` holds for a run the pass
        /// skipped.
        ///
        /// Raises `TypeError` for a field of the wrong type.
        #[new]
        #[pyo3(signature = (
            pass_name,
            changed,
            diagnostics,
            preserved_analyses,
            skipped = OptionalArgument::Omitted,
        ))]
        fn new(
            py: Python<'_>,
            pass_name: &Bound<'_, PyAny>,
            changed: &Bound<'_, PyAny>,
            diagnostics: &Bound<'_, PyAny>,
            preserved_analyses: &Bound<'_, PyAny>,
            skipped: OptionalArgument<'_>,
        ) -> PyResult<Self> {
            Ok(Self {
                pass_name: read_name(pass_name, "PassRunRecord", "pass_name")?,
                changed: read_bool(changed, "PassRunRecord", "changed")?,
                diagnostics: read_diagnostics(diagnostics, "PassRunRecord")?,
                preserved_analyses: read_preserved(preserved_analyses, "PassRunRecord")?,
                skipped: read_optional_bool(py, skipped, "PassRunRecord", "skipped")?,
            })
        }
    }
}

impl PyPassRunRecord {
    /// Return a new public `PassRunRecord` of these fields.
    pub(super) fn build<'py>(
        py: Python<'py>,
        pass_name: &str,
        changed: bool,
        diagnostics: Bound<'py, PyTuple>,
        preserved_analyses: Bound<'py, PyAny>,
        skipped: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        Self::public_class().get(py)?.call1((
            pass_name,
            changed,
            diagnostics,
            preserved_analyses,
            skipped,
        ))
    }
}

define_record_class! {
    /// The record of one iteration of a fixpoint group.
    class PyFixpointIterationRecord as "FixpointIterationRecord" {
        iteration,
        changed,
        pass_runs,
    }
    methods {
        /// Create the record of the 1-based iteration `iteration`, in which
        /// a pass changed the IR if `changed` holds, with the records
        /// `pass_runs` of its pass runs.
        ///
        /// Raises `TypeError` for a field of the wrong type.
        #[new]
        fn new(
            iteration: &Bound<'_, PyAny>,
            changed: &Bound<'_, PyAny>,
            pass_runs: &Bound<'_, PyAny>,
        ) -> PyResult<Self> {
            if !iteration.is_instance_of::<PyInt>() || iteration.is_instance_of::<PyBool>() {
                return Err(build_argument_type_error(
                    "FixpointIterationRecord",
                    "iteration",
                    "an int",
                    iteration,
                )?);
            }
            Ok(Self {
                iteration: iteration.clone().unbind(),
                changed: read_bool(changed, "FixpointIterationRecord", "changed")?,
                pass_runs: read_tuple_of(
                    pass_runs,
                    "FixpointIterationRecord",
                    "pass_runs",
                    "PassRunRecord instances",
                    |value| value.is_instance_of::<PyPassRunRecord>(),
                )?,
            })
        }
    }
}

impl PyFixpointIterationRecord {
    /// Return a new public `FixpointIterationRecord` of these fields.
    pub(super) fn build<'py>(
        py: Python<'py>,
        iteration: usize,
        changed: bool,
        pass_runs: Bound<'py, PyTuple>,
    ) -> PyResult<Bound<'py, PyAny>> {
        Self::public_class()
            .get(py)?
            .call1((iteration, changed, pass_runs))
    }
}

define_record_class! {
    /// The record of a fixpoint group's run.
    class PyFixpointGroupRecord as "FixpointGroupRecord" {
        group_name,
        iteration_records,
        converged,
    }
    methods {
        /// Create the record of the group `group_name`, an `Identifier`,
        /// with the records `iteration_records` of its iterations; it
        /// converged if `converged` holds.
        ///
        /// Raises `TypeError` for a field of the wrong type.
        #[new]
        fn new(
            group_name: &Bound<'_, PyAny>,
            iteration_records: &Bound<'_, PyAny>,
            converged: &Bound<'_, PyAny>,
        ) -> PyResult<Self> {
            if read_identifier_id(group_name)?.is_none() {
                return Err(build_argument_type_error(
                    "FixpointGroupRecord",
                    "group_name",
                    "an Identifier",
                    group_name,
                )?);
            }
            Ok(Self {
                group_name: group_name.clone().unbind(),
                iteration_records: read_tuple_of(
                    iteration_records,
                    "FixpointGroupRecord",
                    "iteration_records",
                    "FixpointIterationRecord instances",
                    |value| value.is_instance_of::<PyFixpointIterationRecord>(),
                )?,
                converged: read_bool(converged, "FixpointGroupRecord", "converged")?,
            })
        }

        /// The number of iterations run.
        #[getter]
        fn iterations(&self, py: Python<'_>) -> PyResult<usize> {
            self.iteration_records.bind(py).len()
        }
    }
}

impl PyFixpointGroupRecord {
    /// Return a new public `FixpointGroupRecord` of these fields.
    pub(super) fn build<'py>(
        py: Python<'py>,
        group_name: Bound<'py, PyAny>,
        iteration_records: Bound<'py, PyTuple>,
        converged: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        Self::public_class()
            .get(py)?
            .call1((group_name, iteration_records, converged))
    }
}

/// Append the records of every pass run in `records`, pipeline records, to
/// `runs`, with each group's in place of the group.
fn collect_pass_runs<'py>(
    records: &Bound<'py, PyAny>,
    runs: &mut Vec<Bound<'py, PyAny>>,
) -> PyResult<()> {
    let py = records.py();
    for record in records.try_iter()? {
        let record = record?;
        if let Ok(group) = record.cast::<PyFixpointGroupRecord>() {
            for iteration in group.get().iteration_records.bind(py).try_iter()? {
                let iteration = iteration?;
                let iteration = iteration.cast::<PyFixpointIterationRecord>()?;
                runs.extend(
                    iteration
                        .get()
                        .pass_runs
                        .bind(py)
                        .try_iter()?
                        .collect::<PyResult<Vec<_>>>()?,
                );
            }
        } else {
            runs.push(record);
        }
    }
    Ok(())
}

define_record_class! {
    /// The result of a pipeline run: the final IR and one record per item.
    class PyPassManagerResult as "PassManagerResult" {
        output,
        records,
    }
    methods {
        /// Create the result of a run that produced `output`, with one
        /// record in `records`, a `PassRunRecord` or a
        /// `FixpointGroupRecord`, per item of the pipeline.
        ///
        /// Raises `TypeError` for a record of another type.
        #[new]
        fn new(output: Bound<'_, PyAny>, records: &Bound<'_, PyAny>) -> PyResult<Self> {
            Ok(Self {
                output: output.unbind(),
                records: read_tuple_of(
                    records,
                    "PassManagerResult",
                    "records",
                    "PassRunRecord or FixpointGroupRecord instances",
                    |value| {
                        value.is_instance_of::<PyPassRunRecord>()
                            || value.is_instance_of::<PyFixpointGroupRecord>()
                    },
                )?,
            })
        }

        /// Return the record of every pass run, in run order, with the runs
        /// of each fixpoint group's iterations in place of the group.
        fn pass_runs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
            let mut runs = Vec::new();
            collect_pass_runs(self.records.bind(py), &mut runs)?;
            PyTuple::new(py, runs)
        }

        /// Return the number of pass runs the pipeline made, not counting
        /// the runs a pass skipped.
        fn run_count(&self, py: Python<'_>) -> PyResult<usize> {
            let mut count = 0;
            for run in self.pass_runs(py)? {
                if !run.getattr(intern!(py, "skipped"))?.is_truthy()? {
                    count += 1;
                }
            }
            Ok(count)
        }
    }
}

impl PyPassManagerResult {
    /// Return a new public `PassManagerResult` of these fields.
    pub(super) fn build<'py>(
        py: Python<'py>,
        output: Bound<'py, PyAny>,
        records: Bound<'py, PyTuple>,
    ) -> PyResult<Bound<'py, PyAny>> {
        Self::public_class().get(py)?.call1((output, records))
    }
}

define_record_class! {
    /// The record of one validator in a `ValidationReport`.
    class PyValidatorRecord as "ValidatorRecord" {
        validator_name,
        failed,
        diagnostics,
    }
    methods {
        /// Create the record of the validator `validator_name`, which
        /// failed to finish its check if `failed` holds, with its
        /// `diagnostics`.
        ///
        /// Raises `TypeError` for a field of the wrong type.
        #[new]
        fn new(
            validator_name: &Bound<'_, PyAny>,
            failed: &Bound<'_, PyAny>,
            diagnostics: &Bound<'_, PyAny>,
        ) -> PyResult<Self> {
            Ok(Self {
                validator_name: read_name(validator_name, "ValidatorRecord", "validator_name")?,
                failed: read_bool(failed, "ValidatorRecord", "failed")?,
                diagnostics: read_diagnostics(diagnostics, "ValidatorRecord")?,
            })
        }
    }
}

impl PyValidatorRecord {
    /// Return a new public `ValidatorRecord` of these fields.
    pub(super) fn build<'py>(
        py: Python<'py>,
        validator_name: &str,
        failed: bool,
        diagnostics: Bound<'py, PyTuple>,
    ) -> PyResult<Bound<'py, PyAny>> {
        Self::public_class()
            .get(py)?
            .call1((validator_name, failed, diagnostics))
    }
}
