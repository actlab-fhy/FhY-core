//! `PyO3` extension module exposing `fhy-core`'s Rust implementation to
//! Python as `fhy_core._rs`.
//!
//! One declarative module holds the whole extension. Each core module's
//! bindings live in a file of the same name, and `lib.rs` exports them into
//! one flat Python namespace: `PyO3` submodules are attributes, not
//! importable packages.

mod constraint;
mod dataclass;
mod described_tag;
mod diagnostic;
mod error;
mod expression;
mod frozen;
mod gc;
mod identifier;
mod interned;
mod lattice;
mod op_attribute;
mod param;
mod pass;
mod provenance;
mod public_class;
mod scoped;
mod serialization;
mod solver;
mod symbol_table;
mod term;
mod types;
mod value_domain;
mod wire;

/// `fhy_core`'s Rust implementation.
///
/// The module declares that it uses the GIL (`gil_used = true`, R2-043), so a
/// free-threaded interpreter re-enables the GIL when importing it: the
/// binding's invariants were argued for the GIL build only, and no CI job
/// runs a free-threaded one. CONTRIBUTING "One extension module per process"
/// records the decision.
#[pyo3::pymodule(name = "_rs", gil_used = true)]
mod rs_module {
    use pyo3::prelude::*;

    #[pymodule_export]
    use super::constraint::{
        PyConstraintSystem, PyEquationConstraint, PyInSetConstraint, PyNotInSetConstraint,
        does_member_lift_to_expression,
    };
    #[pymodule_export]
    use super::diagnostic::{PyDiagnostic, PyNote, PyNoteKind, PyValidationReport};
    #[pymodule_export]
    use super::expression::{
        PyAlternativesPattern, PyBinaryExpressionPattern, PyCallExpressionPattern, PyCapture,
        PyCapturePattern, PyFiredRule, PyIdentifierPattern, PyLiteralPattern,
        PyLogicalExpressionPattern, PyMatchBindings, PyPattern, PyPiecewiseExpressionPattern,
        PyPredicatePattern, PyRewriteRule, PyRuleBase, PyUnaryExpressionPattern, PyWildcardPattern,
        apply_rewrite_rules,
    };
    #[pymodule_export]
    use super::expression::{
        PyBinaryExpression, PyCallExpression, PyExpression, PyIdentifierExpression,
        PyLiteralExpression, PyLogicalExpression, PyPiecewiseExpression, PyUnaryExpression,
        validate_logical_operands, validate_predicate,
    };
    #[pymodule_export]
    use super::expression::{
        PyBuiltinNativeImplementation, coerce_literal_value, evaluate_expression_with_numpy,
        fold_expression, is_decimal_text_exactly_binary,
    };
    #[pymodule_export]
    use super::expression::{
        PyNativeConstant, PyNativeFunction, PyRegisteredFunction, get_native_constant_identifier,
        get_registered_entries, get_registered_entry, inline_functions, is_entry_registered,
        register_function, register_native_constant, register_native_function,
        set_registry_state_for_tests, try_get_native_constant_for_identifier,
        try_get_registered_result_sort,
    };
    #[pymodule_export]
    use super::identifier::{
        advance_identifier_counter_past, allocate_identifier_id, next_identifier_id,
    };
    #[pymodule_export]
    use super::lattice::{PyLattice, PyPartiallyOrderedSet};
    #[pymodule_export]
    use super::op_attribute::PyOpAttribute;
    #[pymodule_export]
    use super::param::{
        PyCategoricalDomain, PyIntegerDomain, PyIntervalIntegerDomain, PyOrdinalDomain, PyParam,
        PyParamAssignment, PyPermutationDomain, PyRealDomain, are_all_constraints_satisfied,
        check_param_bounds_are_ordered, compute_constraint_implication_subset,
        evaluate_system_outcome, is_bound_expression,
    };
    #[pymodule_export]
    use super::pass::{
        PyAnalysisBase, PyAnalysisManager, PyCompilerPassBase, PyFixpointGroupRecord,
        PyFixpointIterationRecord, PyFixpointPassGroup, PyPassManager, PyPassManagerResult,
        PyPassResult, PyPassRunRecord, PyPreservedAnalyses, PyValidationManager, PyValidatorBase,
        PyValidatorRecord, get_verification_passes_for, register_verification_pass,
        run_verification,
    };
    #[pymodule_export]
    use super::provenance::{
        PyCallSiteProvenance, PyFileProvenance, PyFusedProvenance, PyNamedProvenance, PyPosition,
        PyProvenance, PySpan, PyUnknownProvenance,
    };
    #[pymodule_export]
    use super::solver::{
        PySatResult, PySimplifierBase, PySimplifyContext, PySmtLib2ProcessSolver, PySmtScript,
        PySmtSolverBase, PySolver, PySympySimplifier, get_default_solver, set_default_solver,
    };
    #[pymodule_export]
    use super::symbol_table::{
        PyFunctionSymbolTableFrame, PyImportSymbolTableFrame, PySymbolTable,
        PyVariableSymbolTableFrame,
    };
    #[pymodule_export]
    use super::term::{
        PyAlphaRenaming, PyEquivalenceRole, binder_get_free_identifiers,
        binder_is_alpha_equivalent_under, binder_substitute, derived_is_alpha_equivalent_under,
        derived_is_structurally_equivalent, is_identifier_mapping_alpha_equivalent_under,
    };
    #[pymodule_export]
    use super::types::{
        PyDataTypeBase, PyIndexType, PyNumericalType, PyPrimitiveDataType, PyTemplateDataType,
        PyTypeBase, PyTypeUnificationEnvironment, get_core_data_type_bit_width,
        is_weak_core_data_type, promote_core_data_types, promote_primitive_data_types,
        promote_type_qualifiers, resolve_literal_core_data_type, types_bind_data_template,
        types_bind_template, types_is_structurally_equivalent, types_substitute_data_template,
        types_substitute_template, types_unify, types_unify_expression,
    };
    #[pymodule_export]
    use super::types::{
        get_core_data_type_from_literal_type, get_result_core_data_type_for_sort,
        is_core_data_type_compatible_with_sort, types_check_all_function_bodies,
        types_check_expression, types_check_function_body,
    };
    #[pymodule_export]
    use super::value_domain::PyValueDomain;
    #[pymodule_export]
    use super::wire::{
        decode_wire_family, decode_wire_family_json, deserialize_wire_value, encode_wire_dict,
        encode_wire_json, serialize_wire_value,
    };

    /// Set the extension's `__version__` to the crate version, which the
    /// package compares with its own when it is imported, and create the
    /// verification registry of the Python API in the module's state.
    #[pymodule_init]
    fn init(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add("__version__", env!("CARGO_PKG_VERSION"))?;
        module.add(
            super::pass::REGISTRY_ATTRIBUTE,
            Py::new(module.py(), super::pass::PyVerificationRegistry::new())?,
        )
    }
}
