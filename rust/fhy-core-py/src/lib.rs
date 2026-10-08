//! `PyO3` bindings for `fhy-core`'s Rust implementation, as a library that a
//! Python extension module is composed from.
//!
//! # Composition
//!
//! Each `PyO3` extension module links its own copy of every Rust crate it
//! uses, so two extension modules that both link `fhy-core` in one Python
//! process hold two identifier id counters, two registries per interned
//! type and two sets of Python classes. This crate therefore builds no
//! extension module itself. It is an `rlib` that adds its classes and
//! functions to a module it is given, so that **one** `cdylib` per product
//! links `fhy-core`, this crate, and the binding crates of the packages that
//! build on `fhy_core`, and registers all of them into **one** module:
//!
//! - `fhy_core`'s own extension, `fhy_core._rs`, is the `fhy-core-ext`
//!   crate, whose one `#[pymodule]` calls [`register`];
//! - an aggregate extension for a downstream product is a `cdylib` whose
//!   `#[pymodule]` calls [`register`] and then its own crates' registration
//!   functions, and takes and returns `fhy-core` values through [`convert`].
//!   Its classes are written over the building blocks of [`util`], as `fhy_core`'s are.
//!
//! The split into a library and a thin `cdylib` is deliberate. A `#[pymodule]`
//! exports a `PyInit_<name>` symbol, which an `rlib` that is linked into an
//! aggregate would export from it too, so a crate that is both the module and
//! the library cannot be linked into an aggregate whose module has the same
//! name, and every aggregate would export a second module's initializer.
//!
//! # Class identity
//!
//! Every `#[pyclass]` here names its module explicitly, `module =
//! "fhy_core._rs"`, and so keeps the qualified name users see, such as
//! `fhy_core._rs.Param`, whichever native module holds it: `__module__`,
//! `repr`, and `pickle`, which imports the class by that name, all follow
//! it. The binding also finds its own module state by that name, through
//! `sys.modules["fhy_core._rs"]`. An aggregate module is therefore
//! installed as `fhy_core._rs` as well as under its own name, which
//! `fhy_core._extension` does when it loads the module: the loader documents
//! the protocol.
//!
//! # Module state and version
//!
//! [`register`] creates the verification registry of the Python API and
//! the kind registry of `fhy_core.search_space` in the module's state, and
//! sets [`VERSION_ATTRIBUTE`], the version of this crate,
//! which the loader compares with the installed package's. It refuses a
//! module it has registered into already.

mod constraint;
mod described_tag;
mod diagnostic;
mod error;
mod expression;
mod identifier;
mod lattice;
mod object_table;
mod op_attribute;
mod param;
mod pass;
mod provenance;
mod search_space;
mod solver;
mod symbol_table;
mod term;
mod types;
mod value_domain;
mod wire;

pub mod convert;
pub mod util;

use pyo3::prelude::*;

/// The attribute of a module [`register`] was called on that holds this
/// crate's version, which is `fhy_core`'s.
///
/// `fhy_core._extension` compares it with the installed `fhy_core`'s
/// version, so a stale aggregate is refused as a stale `fhy_core._rs` is.
pub const VERSION_ATTRIBUTE: &str = "__fhy_core_version__";

/// Add every class, function and piece of module state of `fhy_core`'s
/// binding to `module`.
///
/// The classes keep their `fhy_core._rs` qualified names whichever module
/// holds them. The module must be reachable as `fhy_core._rs` for the
/// binding to work (see the [crate documentation](crate)); the loader of
/// the Python package arranges that.
///
/// # Errors
///
/// Raises `RuntimeError` if `module` has been registered into already, and
/// whatever adding an item to the module raises.
pub fn register(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    if module.hasattr(VERSION_ATTRIBUTE)? {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "the module {} holds fhy_core's classes already",
            module.name()?
        )));
    }
    register_part_1(module)?;
    register_part_2(module)?;
    register_part_3(module)?;
    register_part_4(module)?;
    module.add(VERSION_ATTRIBUTE, env!("CARGO_PKG_VERSION"))?;
    module.add(
        pass::REGISTRY_ATTRIBUTE,
        Py::new(py, pass::PyVerificationRegistry::new())?,
    )?;
    module.add(
        search_space::KIND_REGISTRY_ATTRIBUTE,
        Py::new(py, search_space::PyKindRegistry::new())?,
    )
}

/// Add the classes and functions of the `constraint`, `diagnostic`, `expression`, `identifier`, `lattice`, `op_attribute` modules.
fn register_part_1(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<constraint::PyConstraintSystem>()?;
    module.add_class::<constraint::PyEquationConstraint>()?;
    module.add_class::<constraint::PyInSetConstraint>()?;
    module.add_class::<constraint::PyNotInSetConstraint>()?;
    module.add_function(wrap_pyfunction!(
        constraint::does_member_lift_to_expression,
        module
    )?)?;
    module.add_class::<diagnostic::PyDiagnostic>()?;
    module.add_class::<diagnostic::PyNote>()?;
    module.add_class::<diagnostic::PyNoteKind>()?;
    module.add_class::<diagnostic::PyValidationReport>()?;
    module.add_class::<expression::PyAlternativesPattern>()?;
    module.add_class::<expression::PyBinaryExpressionPattern>()?;
    module.add_class::<expression::PyCallExpressionPattern>()?;
    module.add_class::<expression::PyCapture>()?;
    module.add_class::<expression::PyCapturePattern>()?;
    module.add_class::<expression::PyFiredRule>()?;
    module.add_class::<expression::PyIdentifierPattern>()?;
    module.add_class::<expression::PyLiteralPattern>()?;
    module.add_class::<expression::PyLogicalExpressionPattern>()?;
    module.add_class::<expression::PyMatchBindings>()?;
    module.add_class::<expression::PyPattern>()?;
    module.add_class::<expression::PyPiecewiseExpressionPattern>()?;
    module.add_class::<expression::PyPredicatePattern>()?;
    module.add_class::<expression::PyRewriteRule>()?;
    module.add_class::<expression::PyRuleBase>()?;
    module.add_class::<expression::PyUnaryExpressionPattern>()?;
    module.add_class::<expression::PyWildcardPattern>()?;
    module.add_function(wrap_pyfunction!(expression::apply_rewrite_rules, module)?)?;
    module.add_class::<expression::PyBinaryExpression>()?;
    module.add_class::<expression::PyCallExpression>()?;
    module.add_class::<expression::PyExpression>()?;
    module.add_class::<expression::PyIdentifierExpression>()?;
    module.add_class::<expression::PyLiteralExpression>()?;
    module.add_class::<expression::PyLogicalExpression>()?;
    module.add_class::<expression::PyPiecewiseExpression>()?;
    module.add_class::<expression::PyUnaryExpression>()?;
    module.add_function(wrap_pyfunction!(
        expression::validate_logical_operands,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(expression::validate_predicate, module)?)?;
    module.add_class::<expression::PyBuiltinNativeImplementation>()?;
    module.add_function(wrap_pyfunction!(expression::coerce_literal_value, module)?)?;
    module.add_function(wrap_pyfunction!(
        expression::evaluate_expression_with_numpy,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(expression::fold_expression, module)?)?;
    module.add_function(wrap_pyfunction!(
        expression::is_decimal_text_exactly_binary,
        module
    )?)?;
    module.add_class::<expression::PyNativeConstant>()?;
    module.add_class::<expression::PyNativeFunction>()?;
    module.add_class::<expression::PyRegisteredFunction>()?;
    module.add_function(wrap_pyfunction!(
        expression::get_native_constant_identifier,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        expression::get_registered_entries,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(expression::get_registered_entry, module)?)?;
    module.add_function(wrap_pyfunction!(expression::inline_functions, module)?)?;
    module.add_function(wrap_pyfunction!(expression::is_entry_registered, module)?)?;
    module.add_function(wrap_pyfunction!(expression::register_function, module)?)?;
    module.add_function(wrap_pyfunction!(
        expression::register_native_constant,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        expression::register_native_function,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(expression::set_registry_state, module)?)?;
    module.add_function(wrap_pyfunction!(
        expression::try_get_native_constant_for_identifier,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        expression::try_get_registered_result_sort,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        identifier::advance_identifier_counter_past,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        identifier::allocate_identifier_id,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(identifier::next_identifier_id, module)?)?;
    module.add_class::<lattice::PyLattice>()?;
    module.add_class::<lattice::PyPartiallyOrderedSet>()?;
    module.add_class::<op_attribute::PyOpAttribute>()?;
    Ok(())
}

/// Add the classes and functions of the `param`, `pass`, `provenance`, `solver`, `symbol_table`, `term` modules.
fn register_part_2(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<param::PyCategoricalDomain>()?;
    module.add_class::<param::PyIntegerDomain>()?;
    module.add_class::<param::PyIntervalIntegerDomain>()?;
    module.add_class::<param::PyOrdinalDomain>()?;
    module.add_class::<param::PyParam>()?;
    module.add_class::<param::PyParamAssignment>()?;
    module.add_class::<param::PyPermutationDomain>()?;
    module.add_class::<param::PyRealDomain>()?;
    module.add_function(wrap_pyfunction!(
        param::are_all_constraints_satisfied,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        param::check_param_bounds_are_ordered,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        param::compute_constraint_implication_subset,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(param::evaluate_system_outcome, module)?)?;
    module.add_function(wrap_pyfunction!(param::is_bound_expression, module)?)?;
    module.add_class::<pass::PyAnalysisBase>()?;
    module.add_class::<pass::PyAnalysisManager>()?;
    module.add_class::<pass::PyCompilerPassBase>()?;
    module.add_class::<pass::PyFixpointGroupRecord>()?;
    module.add_class::<pass::PyFixpointIterationRecord>()?;
    module.add_class::<pass::PyFixpointPassGroup>()?;
    module.add_class::<pass::PyPassManager>()?;
    module.add_class::<pass::PyPassManagerResult>()?;
    module.add_class::<pass::PyPassResult>()?;
    module.add_class::<pass::PyPassRunRecord>()?;
    module.add_class::<pass::PyPreservedAnalyses>()?;
    module.add_class::<pass::PyValidationManager>()?;
    module.add_class::<pass::PyValidatorBase>()?;
    module.add_class::<pass::PyValidatorRecord>()?;
    module.add_function(wrap_pyfunction!(pass::get_verification_passes_for, module)?)?;
    module.add_function(wrap_pyfunction!(pass::register_verification_pass, module)?)?;
    module.add_function(wrap_pyfunction!(pass::run_verification, module)?)?;
    module.add_class::<provenance::PyCallSiteProvenance>()?;
    module.add_class::<provenance::PyFileProvenance>()?;
    module.add_class::<provenance::PyFusedProvenance>()?;
    module.add_class::<provenance::PyNamedProvenance>()?;
    module.add_class::<provenance::PyPosition>()?;
    module.add_class::<provenance::PyProvenance>()?;
    module.add_class::<provenance::PySpan>()?;
    module.add_class::<provenance::PyUnknownProvenance>()?;
    module.add_class::<solver::PyGroundSimplifier>()?;
    module.add_class::<solver::PySatResult>()?;
    module.add_class::<solver::PySimplifierBase>()?;
    module.add_class::<solver::PySimplifyContext>()?;
    module.add_class::<solver::PySmtLib2ProcessSolver>()?;
    module.add_class::<solver::PySmtScript>()?;
    module.add_class::<solver::PySmtSolverBase>()?;
    module.add_class::<solver::PySolver>()?;
    module.add_class::<solver::PySympySimplifier>()?;
    module.add_function(wrap_pyfunction!(solver::get_default_solver, module)?)?;
    module.add_function(wrap_pyfunction!(solver::set_default_solver, module)?)?;
    module.add_class::<symbol_table::PyFunctionSymbolTableFrame>()?;
    module.add_class::<symbol_table::PyImportSymbolTableFrame>()?;
    module.add_class::<symbol_table::PySymbolTable>()?;
    module.add_class::<symbol_table::PyVariableSymbolTableFrame>()?;
    module.add_class::<term::PyAlphaRenaming>()?;
    module.add_class::<term::PyEquivalenceRole>()?;
    module.add_function(wrap_pyfunction!(term::binder_get_free_identifiers, module)?)?;
    module.add_function(wrap_pyfunction!(
        term::binder_is_alpha_equivalent_under,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(term::binder_substitute, module)?)?;
    module.add_function(wrap_pyfunction!(
        term::derived_is_alpha_equivalent_under,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        term::derived_is_structurally_equivalent,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        term::is_identifier_mapping_alpha_equivalent_under,
        module
    )?)?;
    Ok(())
}

/// Add the classes and functions of the `types`, `value_domain`, `wire` modules.
fn register_part_3(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<types::PyDataTypeBase>()?;
    module.add_class::<types::PyIndexType>()?;
    module.add_class::<types::PyNumericalType>()?;
    module.add_class::<types::PyPrimitiveDataType>()?;
    module.add_class::<types::PyTemplateDataType>()?;
    module.add_class::<types::PyTypeBase>()?;
    module.add_class::<types::PyTypeUnificationEnvironment>()?;
    module.add_function(wrap_pyfunction!(
        types::get_core_data_type_bit_width,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(types::is_weak_core_data_type, module)?)?;
    module.add_function(wrap_pyfunction!(types::promote_core_data_types, module)?)?;
    module.add_function(wrap_pyfunction!(
        types::promote_primitive_data_types,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(types::promote_type_qualifiers, module)?)?;
    module.add_function(wrap_pyfunction!(
        types::resolve_literal_core_data_type,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(types::types_bind_data_template, module)?)?;
    module.add_function(wrap_pyfunction!(types::types_bind_template, module)?)?;
    module.add_function(wrap_pyfunction!(
        types::types_is_structurally_equivalent,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        types::types_substitute_data_template,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(types::types_substitute_template, module)?)?;
    module.add_function(wrap_pyfunction!(types::types_unify, module)?)?;
    module.add_function(wrap_pyfunction!(types::types_unify_expression, module)?)?;
    module.add_function(wrap_pyfunction!(
        types::get_core_data_type_from_literal_type,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        types::get_result_core_data_type_for_sort,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        types::is_core_data_type_compatible_with_sort,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        types::types_check_all_function_bodies,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(types::types_check_expression, module)?)?;
    module.add_function(wrap_pyfunction!(types::types_check_function_body, module)?)?;
    module.add_class::<value_domain::PyValueDomain>()?;
    module.add_function(wrap_pyfunction!(wire::decode_wire_family, module)?)?;
    module.add_function(wrap_pyfunction!(wire::decode_wire_family_json, module)?)?;
    module.add_function(wrap_pyfunction!(wire::deserialize_wire_value, module)?)?;
    module.add_function(wrap_pyfunction!(wire::encode_wire_dict, module)?)?;
    module.add_function(wrap_pyfunction!(wire::encode_wire_json, module)?)?;
    module.add_function(wrap_pyfunction!(wire::serialize_wire_value, module)?)?;
    Ok(())
}

/// Add the classes and functions of the `search_space` module.
fn register_part_4(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<search_space::PyAlternativeBase>()?;
    module.add_class::<search_space::PyChoice>()?;
    module.add_class::<search_space::PyCondition>()?;
    module.add_class::<search_space::PyConfiguration>()?;
    module.add_class::<search_space::PyConfigurationKey>()?;
    module.add_class::<search_space::PyForbidden>()?;
    module.add_class::<search_space::PySpace>()?;
    module.add_class::<search_space::PyVariableBase>()?;
    module.add_class::<search_space::PyRng>()?;
    module.add_class::<search_space::PyChoiceDomain>()?;
    module.add_class::<search_space::PyOrderDomain>()?;
    module.add_class::<search_space::PyStridedRun>()?;
    module.add_class::<search_space::PyStridedDomain>()?;
    module.add_class::<search_space::PyPendingStep>()?;
    module.add_class::<search_space::PyRandomOracle>()?;
    module.add_class::<search_space::PyReplayOracle>()?;
    module.add_class::<search_space::PyExhaustiveOracle>()?;
    module.add_class::<search_space::PyRecorder>()?;
    module.add_class::<search_space::PyTraceStep>()?;
    module.add_class::<search_space::PyTrace>()?;
    module.add_class::<search_space::PyObjective>()?;
    module.add_class::<search_space::PyMeasurement>()?;
    module.add_function(wrap_pyfunction!(
        search_space::get_search_space_kind_class,
        module
    )?)?;
    Ok(())
}
