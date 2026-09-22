mod identifier;

use pyo3::prelude::*;

use identifier::{advance_identifier_counter_past, allocate_identifier_id};

#[pymodule]
fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(allocate_identifier_id, m)?)?;
    m.add_function(wrap_pyfunction!(advance_identifier_counter_past, m)?)?;
    Ok(())
}
