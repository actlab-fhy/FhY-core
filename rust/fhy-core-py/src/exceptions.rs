//! The Python exception classes the binding raises, one constructor each
//! (R2-033).
//!
//! Each class of the Python package the binding raises is one
//! [`ExceptionClass`] static here, imported on first use. A binding module
//! builds the exception with [`ExceptionClass::err`], or reads the class
//! with [`ExceptionClass::class`] to test an exception against it.
//! [`unbox_py_err`] returns the Python exception a core error boxed.

use fhy_core::foreign::BoxError;
use pyo3::call::PyCallArgs;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};

use crate::python::ImportedAttr;

/// A Python exception class, imported on first use.
pub(crate) struct ExceptionClass(ImportedAttr<PyType>);

impl ExceptionClass {
    /// Return the class `name` of `module`, not yet imported.
    const fn new(module: &'static str, name: &'static str) -> Self {
        Self(ImportedAttr::new(module, name))
    }

    /// Return the class.
    ///
    /// # Errors
    ///
    /// Raises what importing it raises.
    pub(crate) fn class<'py>(&'static self, py: Python<'py>) -> PyResult<&'py Bound<'py, PyType>> {
        self.0.get(py)
    }

    /// Return the exception the class builds from `args` and `keywords`.
    ///
    /// # Errors
    ///
    /// Raises what importing the class or building the exception raises.
    pub(crate) fn build<'py>(
        &'static self,
        py: Python<'py>,
        args: impl PyCallArgs<'py>,
        keywords: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<PyErr> {
        self.class(py)?.call(args, keywords).map(PyErr::from_value)
    }

    /// Return the exception the class builds from `args`, or the exception
    /// importing the class or building it raised.
    pub(crate) fn err<'py>(&'static self, py: Python<'py>, args: impl PyCallArgs<'py>) -> PyErr {
        self.build(py, args, None).unwrap_or_else(|error| error)
    }

    /// Return whether `error` is an instance of the class, false if the
    /// class does not import.
    pub(crate) fn is_instance_of(&'static self, py: Python<'_>, error: &PyErr) -> bool {
        self.class(py)
            .is_ok_and(|class| error.is_instance(py, class))
    }
}

/// Return the Python exception boxed in `error`, or `error` back if it holds
/// none.
///
/// # Errors
///
/// Returns `error` when it is no Python exception.
pub(crate) fn unbox_py_err(error: BoxError) -> Result<PyErr, BoxError> {
    error.downcast::<PyErr>().map(|error| *error)
}

/// Return the Python exception boxed in `error`, unchanged, or a
/// `RuntimeError` with its text when it is no Python exception.
pub(crate) fn boxed_error_to_py(error: BoxError) -> PyErr {
    unbox_py_err(error).unwrap_or_else(|other| PyRuntimeError::new_err(other.to_string()))
}

/// The module of the serialization framework.
const SERIALIZATION: &str = "fhy_core.serialization";
/// The module of the expression errors.
const EXPRESSION_ERRORS: &str = "fhy_core.symbolic.expression.errors";
/// The module of the constraint errors.
const CONSTRAINT_ERRORS: &str = "fhy_core.symbolic.constraint.errors";
/// The module of the solver.
const SOLVER: &str = "fhy_core.symbolic.solver";
/// The module of the pass infrastructure's classes.
const PASS: &str = "fhy_core.pass_infrastructure.core";

/// Declare one [`ExceptionClass`] static per class.
macro_rules! exception_classes {
    ($($static:ident = $module:expr, $name:literal;)*) => {
        $(
            #[doc = concat!("`", $name, "`.")]
            pub(crate) static $static: ExceptionClass = ExceptionClass::new($module, $name);
        )*
    };
}

exception_classes! {
    SERIALIZATION_ERROR = SERIALIZATION, "SerializationError";
    DESERIALIZATION_VALUE_ERROR = SERIALIZATION, "DeserializationValueError";
    DESERIALIZATION_DICT_STRUCTURE_ERROR = SERIALIZATION, "DeserializationDictStructureError";
    MALFORMED_PAYLOAD_ERROR = SERIALIZATION, "MalformedPayloadError";

    ENTRY_LOOKUP_ERROR = EXPRESSION_ERRORS, "EntryLookupError";
    ENTRY_REGISTRATION_ERROR = EXPRESSION_ERRORS, "EntryRegistrationError";
    NATIVE_CONSTANT_BINDING_ERROR = EXPRESSION_ERRORS, "NativeConstantBindingError";
    NATIVE_CONSTANT_LOWERING_ERROR = EXPRESSION_ERRORS, "NativeConstantLoweringError";
    NATIVE_RESULT_SORT_ERROR = EXPRESSION_ERRORS, "NativeResultSortError";
    NON_FINITE_CAST_ERROR = EXPRESSION_ERRORS, "NonFiniteCastError";
    STRING_LITERAL_PRECISION_ERROR = EXPRESSION_ERRORS, "StringLiteralPrecisionError";
    UNBOUND_VARIABLE_ERROR = EXPRESSION_ERRORS, "UnboundVariableError";
    UNSUPPORTED_NUMPY_LOWERING_ERROR = EXPRESSION_ERRORS, "UnsupportedNumpyLoweringError";
    NON_BOOLEAN_LOGICAL_OPERAND_ERROR = EXPRESSION_ERRORS, "NonBooleanLogicalOperandError";
    UNDECIDABLE_ERROR = EXPRESSION_ERRORS, "UndecidableError";
    COMPLEX_INFINITY_LIFT_ERROR = EXPRESSION_ERRORS, "ComplexInfinityLiftError";
    PARTIAL_PIECEWISE_ERROR = EXPRESSION_ERRORS, "PartialPiecewiseError";
    FUNCTION_ARITY_ERROR = "fhy_core.symbolic.expression.passes.inline", "FunctionArityError";
    REWRITE_CALLBACK_ERROR = "fhy_core.symbolic.expression.pattern.rewrite", "RewriteCallbackError";
    REWRITE_REBUILD_ERROR = "fhy_core.symbolic.expression.pattern.rewrite", "RewriteRebuildError";

    CONSTRAINT_ERROR = CONSTRAINT_ERRORS, "ConstraintError";
    MISSING_SYMBOL_TYPE_ERROR = CONSTRAINT_ERRORS, "MissingSymbolTypeError";
    PARAM_ERROR = "fhy_core.symbolic.param.values", "ParamError";

    SOLVER_CAPABILITY_ERROR = SOLVER, "SolverCapabilityError";
    SOLVER_BACKEND_ERROR = SOLVER, "SolverBackendError";
    SOLVER_BACKEND_UNAVAILABLE_ERROR = SOLVER, "SolverBackendUnavailableError";

    PASS_EXECUTION_ERROR = PASS, "PassExecutionError";
    PASS_VALIDATION_ERROR = PASS, "PassValidationError";
    PASS_REGISTRATION_ERROR = PASS, "PassRegistrationError";
    VALIDATION_FAILED_ERROR = "fhy_core.diagnostic", "ValidationFailedError";

    FROZEN_MUTATION_ERROR = "fhy_core.traits.frozen", "FrozenMutationError";
    VERIFICATION_ERROR = "fhy_core.traits.verifiable", "VerificationError";
    CORE_TYPE_ERROR = "fhy_core.types.core", "FhYCoreTypeError";
    SYMBOL_TABLE_ERROR = "fhy_core.symbol_table", "SymbolTableError";
    EQUIVALENCE_DERIVATION_ERROR = "fhy_core.term.derived_equivalence", "EquivalenceDerivationError";
}

#[cfg(test)]
mod tests {
    use pyo3::exceptions::{PyModuleNotFoundError, PyValueError};

    use super::*;

    static VALUE_ERROR: ExceptionClass = ExceptionClass::new("builtins", "ValueError");
    static MISSING: ExceptionClass = ExceptionClass::new("fhy_core_no_such_module", "Missing");

    #[test]
    fn a_class_builds_its_exception_or_returns_the_import_error() {
        Python::initialize();
        Python::attach(|py| {
            let error = VALUE_ERROR.err(py, ("bad",));
            assert!(error.is_instance_of::<PyValueError>(py));
            assert_eq!(error.value(py).to_string(), "bad");
            assert!(VALUE_ERROR.is_instance_of(py, &error));
            assert!(
                MISSING
                    .err(py, ("bad",))
                    .is_instance_of::<PyModuleNotFoundError>(py)
            );
            assert!(!MISSING.is_instance_of(py, &error));
        });
    }

    #[test]
    fn a_boxed_python_exception_is_unboxed_unchanged() {
        Python::initialize();
        Python::attach(|py| {
            let raised = PyValueError::new_err("raised");
            let value = raised.value(py).clone();
            let unboxed = unbox_py_err(Box::new(raised)).expect("a Python exception");
            assert!(unboxed.value(py).is(&value));
            let other = unbox_py_err("not Python".into()).expect_err("no Python exception");
            assert_eq!(other.to_string(), "not Python");
            let error = boxed_error_to_py("not Python".into());
            assert!(error.is_instance_of::<PyRuntimeError>(py));
            assert_eq!(error.value(py).to_string(), "not Python");
        });
    }
}
