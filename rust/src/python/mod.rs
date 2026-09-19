mod identifier;

use pyo3::prelude::*;

use identifier::{advance_identifier_counter_past, allocate_identifier_id};

#[pymodule]
fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.2.0")?;
    m.add_function(wrap_pyfunction!(rust_available, m)?)?;
    m.add_function(wrap_pyfunction!(allocate_identifier_id, m)?)?;
    m.add_function(wrap_pyfunction!(advance_identifier_counter_past, m)?)?;
    Ok(())
}

#[pyfunction]
fn rust_available() -> bool {
    crate::rust_available()
}
