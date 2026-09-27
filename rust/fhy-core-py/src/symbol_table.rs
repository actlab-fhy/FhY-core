//! `PyO3` bindings for [`fhy_core::symbol_table`] (S15 of
//! `docs/design/python-switch.md`).
//!
//! - `frames.rs`: the three built-in frames (D-S15-7, D-S15-8) and
//!   `FunctionKeyword`, which stays a Python enum.
//! - `table.rs`: `SymbolTable` over the core table (D-S15-10 to D-S15-13).
//!
//! The abstract `SymbolTableFrame` stays a Python class; a frame Python
//! defines is held by the table as an opaque entry (D-S15-9).

mod frames;
mod table;

pub(crate) use frames::{
    PyFunctionSymbolTableFrame, PyImportSymbolTableFrame, PyVariableSymbolTableFrame,
    frame_from_wire, frame_wire_data,
};
pub(crate) use table::PySymbolTable;
