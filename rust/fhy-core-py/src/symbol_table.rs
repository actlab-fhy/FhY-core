//! `PyO3` bindings for [`fhy_core::symbol_table`].
//!
//! - `frames.rs`: the three built-in frames and `FunctionKeyword`, which
//!   stays a Python enum.
//! - `table.rs`: `SymbolTable` over the core table.
//!
//! The abstract `SymbolTableFrame` stays a Python class; a frame Python
//! defines is held by the table as an opaque entry.

mod frames;
mod table;

pub(crate) use frames::{
    PyFunctionSymbolTableFrame, PyImportSymbolTableFrame, PyVariableSymbolTableFrame,
    frame_from_wire, frame_wire_data,
};
pub(crate) use table::PySymbolTable;
