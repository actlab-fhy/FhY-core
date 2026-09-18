use pyo3::prelude::*;

#[pymodule]
fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.2.0")?;
    m.add_function(wrap_pyfunction!(rust_available, m)?)?;
    Ok(())
}

#[pyfunction]
fn rust_available() -> bool {
    crate::identifier::rust_available()
}
