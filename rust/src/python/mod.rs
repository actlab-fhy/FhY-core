mod identifier;

use pyo3::prelude::*;

use identifier::Identifier;

#[pymodule]
fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.2.0")?;
    m.add_function(wrap_pyfunction!(rust_available, m)?)?;
    m.add_class::<Identifier>()?;
    Ok(())
}

#[pyfunction]
fn rust_available() -> bool {
    crate::rust_available()
}
