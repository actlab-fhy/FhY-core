//! `PyO3` extension module exposing `fhy-core`'s Rust implementation to
//! Python as `fhy_core._rs`.
//!
//! One declarative module holds the whole extension. Each core module's
//! bindings live in a file of the same name, and `lib.rs` exports them into
//! one flat Python namespace: `PyO3` submodules are attributes, not
//! importable packages.

mod dataclass;
mod described_tag;
mod diagnostic;
mod error;
mod expression;
mod frozen;
mod identifier;
mod interned;
mod op_attribute;
mod pass;
mod provenance;
mod public_class;
mod serialization;
mod solver;
mod term;
mod value_domain;

/// `fhy_core`'s Rust implementation.
#[pyo3::pymodule(name = "_rs")]
mod rs_module {
    use pyo3::prelude::*;

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
    use super::identifier::{advance_identifier_counter_past, allocate_identifier_id};
    #[pymodule_export]
    use super::op_attribute::PyOpAttribute;
    #[pymodule_export]
    use super::pass::{
        PyAnalysisBase, PyAnalysisManager, PyCompilerPassBase, PyFixpointGroupRecord,
        PyFixpointIterationRecord, PyFixpointPassGroup, PyPassManager, PyPassManagerResult,
        PyPassResult, PyPassRunRecord, PyPreservedAnalyses, PyValidationManager, PyValidatorBase,
        PyValidatorRecord,
    };
    #[pymodule_export]
    use super::provenance::{
        PyCallSiteProvenance, PyFileProvenance, PyFusedProvenance, PyNamedProvenance, PyPosition,
        PyProvenance, PySpan, PyUnknownProvenance,
    };
    #[pymodule_export]
    use super::solver::{
        PySatResult, PySimplifierBase, PySmtLib2ProcessSolver, PySmtScript, PySmtSolverBase,
        PySolver, PySympySimplifier, get_default_solver, set_default_solver,
    };
    #[pymodule_export]
    use super::term::{
        PyAlphaRenaming, PyEquivalenceRole, binder_get_free_identifiers,
        binder_is_alpha_equivalent_under, binder_substitute, derived_is_alpha_equivalent_under,
        derived_is_structurally_equivalent, is_identifier_mapping_alpha_equivalent_under,
    };
    #[pymodule_export]
    use super::value_domain::PyValueDomain;

    /// Set the extension's `__version__` to the crate version, which the
    /// package compares with its own when it is imported.
    #[pymodule_init]
    fn init(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add("__version__", env!("CARGO_PKG_VERSION"))
    }
}
