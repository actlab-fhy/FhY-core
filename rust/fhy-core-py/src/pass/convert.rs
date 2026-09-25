//! Converting what a run returns to Python (D-S6-18): diagnostics, the
//! records of a pipeline and validation reports.
//!
//! Every value converts once, where it leaves Rust. A diagnostic a Python
//! hook reported comes back as the object it reported, found in the run's
//! scope; one the core made is built through the public classes.

use pyo3::prelude::*;
use pyo3::types::PyTuple;

use fhy_core::diagnostic::{Diagnostic, ValidationReport};
use fhy_core::pass::{FixpointGroupRecord, PassRunRecord, PipelineRecord, ValidatorRecord};

use crate::diagnostic::{diagnostic_to_python, report_to_python};
use crate::identifier::identifier_to_python;

use super::analysis::preserved_to_python;
use super::records::{
    PyFixpointGroupRecord, PyFixpointIterationRecord, PyPassRunRecord, PyValidatorRecord,
};
use super::scope::RunScope;

/// Return the Python object of `diagnostic`: the one a hook reported, or a
/// new one.
pub(super) fn diagnostic_object<'py>(
    py: Python<'py>,
    diagnostic: &Diagnostic,
    scope: &mut RunScope,
) -> PyResult<Bound<'py, PyAny>> {
    match scope.take_reported(py, diagnostic) {
        Some(object) => Ok(object),
        None => diagnostic_to_python(py, diagnostic),
    }
}

/// Return the tuple of the Python objects of `diagnostics`.
pub(super) fn diagnostics_to_python<'py>(
    py: Python<'py>,
    diagnostics: &[Diagnostic],
    scope: &mut RunScope,
) -> PyResult<Bound<'py, PyTuple>> {
    let objects = diagnostics
        .iter()
        .map(|diagnostic| diagnostic_object(py, diagnostic, scope))
        .collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(py, objects)
}

/// Return the public `PassRunRecord` of `record`.
fn pass_run_to_python<'py>(
    py: Python<'py>,
    record: &PassRunRecord,
    scope: &mut RunScope,
) -> PyResult<Bound<'py, PyAny>> {
    PyPassRunRecord::build(
        py,
        record.pass_name(),
        record.is_changed(),
        diagnostics_to_python(py, record.diagnostics(), scope)?,
        preserved_to_python(py, record.preserved_analyses())?,
        record.is_skipped(),
    )
}

/// Return the public `FixpointGroupRecord` of `record`, whose group's
/// Python name is `name` if known.
fn group_to_python<'py>(
    py: Python<'py>,
    record: &FixpointGroupRecord,
    name: Option<&Py<PyAny>>,
    scope: &mut RunScope,
) -> PyResult<Bound<'py, PyAny>> {
    let mut iterations = Vec::with_capacity(record.iteration_records().len());
    for iteration in record.iteration_records() {
        let runs = iteration
            .pass_runs()
            .iter()
            .map(|run| pass_run_to_python(py, run, scope))
            .collect::<PyResult<Vec<_>>>()?;
        iterations.push(PyFixpointIterationRecord::build(
            py,
            iteration.iteration(),
            iteration.is_changed(),
            PyTuple::new(py, runs)?,
        )?);
    }
    let name = match name {
        Some(name) => name.bind(py).clone(),
        None => identifier_to_python(py, record.group_name())?,
    };
    PyFixpointGroupRecord::build(
        py,
        name,
        PyTuple::new(py, iterations)?,
        record.is_converged(),
    )
}

/// Return the tuple of the public records of `records`, one per pipeline
/// item, where `group_names` holds each item's Python group name, `None`
/// for a pass.
pub(super) fn records_to_python<'py>(
    py: Python<'py>,
    records: &[PipelineRecord],
    group_names: &[Option<Py<PyAny>>],
    scope: &mut RunScope,
) -> PyResult<Bound<'py, PyTuple>> {
    let mut objects = Vec::with_capacity(records.len());
    for (index, record) in records.iter().enumerate() {
        let name = group_names.get(index).and_then(Option::as_ref);
        objects.push(match record {
            PipelineRecord::Pass(record) => pass_run_to_python(py, record, scope)?,
            PipelineRecord::FixpointGroup(record) => group_to_python(py, record, name, scope)?,
            _ => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "the pipeline returned a record of an unknown kind",
                ));
            }
        });
    }
    PyTuple::new(py, objects)
}

/// Return the public `ValidationReport` of `report`, whose records are
/// `ValidatorRecord`s holding their slices of the report's diagnostic
/// objects.
pub(super) fn validation_report_to_python<'py>(
    py: Python<'py>,
    report: &ValidationReport<ValidatorRecord>,
    scope: &mut RunScope,
) -> PyResult<Bound<'py, PyAny>> {
    let diagnostics = diagnostics_to_python(py, report.diagnostics(), scope)?;
    let mut records = Vec::with_capacity(report.records().len());
    let mut start = 0;
    for record in report.records() {
        let end = start + record.diagnostics_in(report).len();
        records.push(PyValidatorRecord::build(
            py,
            record.validator_name(),
            record.is_failed(),
            diagnostics.get_slice(start, end),
        )?);
        start = end;
    }
    report_to_python(py, diagnostics, PyTuple::new(py, records)?)
}
