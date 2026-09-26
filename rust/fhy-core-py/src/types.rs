//! `PyO3` bindings for [`fhy_core::types`] (S11a of
//! `docs/design/python-switch.md`).
//!
//! - `classes.rs`: the bases `Type` and `DataType`, and the four built-in
//!   classes (D-S11-11).
//! - `environment.rs`: `TypeUnificationEnvironment` (D-S11-13).
//! - `adapter.rs`: the Python-defined types as core extensions, driven
//!   through the dispatchers' registered handlers, and the context each
//!   call runs in (D-S11-9).
//! - `convert.rs`: the conversions between Python objects and core values.
//! - `dispatch.rs`: the dispatchers' functions and the promotion helpers.
//! - `enums.rs`: `CoreDataType` and `TypeQualifier`, which stay Python
//!   enums (D-S11-12).
//! - `error.rs`: the Python exceptions of the core's errors (D-S11-14).
//! - `checking.rs`: the type checker, the body checks and the sort tables
//!   (S11b, D-S11-20 to D-S11-22).

mod adapter;
mod checking;
mod classes;
mod convert;
mod dispatch;
mod enums;
mod environment;
mod error;

pub(crate) use checking::{
    get_core_data_type_from_literal_type, get_result_core_data_type_for_sort,
    is_core_data_type_compatible_with_sort, types_check_all_function_bodies,
    types_check_expression, types_check_function_body,
};
pub(crate) use classes::{
    PyDataTypeBase, PyIndexType, PyNumericalType, PyPrimitiveDataType, PyTemplateDataType,
    PyTypeBase,
};
pub(crate) use dispatch::{
    get_core_data_type_bit_width, is_weak_core_data_type, promote_core_data_types,
    promote_primitive_data_types, promote_type_qualifiers, resolve_literal_core_data_type,
    types_bind_data_template, types_bind_template, types_is_structurally_equivalent,
    types_substitute_data_template, types_substitute_template, types_unify, types_unify_expression,
};
pub(crate) use environment::PyTypeUnificationEnvironment;
