//! The Python exception classes the binding raises, one constructor each.
//!
//! Each class of the Python package the binding raises is one
//! [`ExceptionClass`] static, imported on first use. A binding module
//! builds the exception with [`ExceptionClass::err`], or reads the class
//! with [`ExceptionClass::class`] to test an exception against it.
//! [`unbox_py_err`] returns the Python exception a core error boxed.
//!
//! A downstream binding crate declares its own classes the same way,
//! `static CLASS: ExceptionClass = ExceptionClass::new("pkg.errors", "Name")`,
//! and raises `fhy_core`'s serialization and frozen-mutation exceptions
//! through the statics here: [`SERIALIZATION_ERROR`],
//! [`DESERIALIZATION_VALUE_ERROR`], [`DESERIALIZATION_DICT_STRUCTURE_ERROR`],
//! [`MALFORMED_PAYLOAD_ERROR`], [`FROZEN_MUTATION_ERROR`] and
//! [`EQUIVALENCE_DERIVATION_ERROR`]. The binding's other classes stay
//! private to it.

use fhy_core::foreign::BoxError;
use pyo3::call::PyCallArgs;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};

use crate::util::python::ImportedAttr;

/// A Python exception class, imported on first use.
#[derive(Debug)]
pub struct ExceptionClass(ImportedAttr<PyType>);

impl ExceptionClass {
    /// Return the class `name` of `module`, not yet imported.
    #[must_use]
    pub const fn new(module: &'static str, name: &'static str) -> Self {
        Self(ImportedAttr::new(module, name))
    }

    /// Return the class.
    ///
    /// # Errors
    ///
    /// Raises what importing it raises.
    pub fn class<'py>(&'static self, py: Python<'py>) -> PyResult<&'py Bound<'py, PyType>> {
        self.0.get(py)
    }

    /// Return the exception the class builds from `args` and `keywords`.
    ///
    /// # Errors
    ///
    /// Raises what importing the class or building the exception raises.
    pub fn build<'py>(
        &'static self,
        py: Python<'py>,
        args: impl PyCallArgs<'py>,
        keywords: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<PyErr> {
        self.class(py)?.call(args, keywords).map(PyErr::from_value)
    }

    /// Return the exception the class builds from `args`, or the exception
    /// importing the class or building it raised.
    pub fn err<'py>(&'static self, py: Python<'py>, args: impl PyCallArgs<'py>) -> PyErr {
        self.build(py, args, None).unwrap_or_else(|error| error)
    }

    /// Return whether `error` is an instance of the class, false if the
    /// class does not import.
    pub fn is_instance_of(&'static self, py: Python<'_>, error: &PyErr) -> bool {
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
pub fn unbox_py_err(error: BoxError) -> Result<PyErr, BoxError> {
    error.downcast::<PyErr>().map(|error| *error)
}

/// Return the Python exception boxed in `error`, unchanged, or a
/// `RuntimeError` with its text when it is no Python exception.
#[must_use]
pub fn boxed_error_to_py(error: BoxError) -> PyErr {
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
    ($($vis:vis $static:ident = $module:expr, $name:literal;)*) => {
        $(
            #[doc = concat!("`", $name, "`.")]
            $vis static $static: ExceptionClass = ExceptionClass::new($module, $name);
        )*
    };
}

exception_classes! {
    pub SERIALIZATION_ERROR = SERIALIZATION, "SerializationError";
    pub DESERIALIZATION_VALUE_ERROR = SERIALIZATION, "DeserializationValueError";
    pub DESERIALIZATION_DICT_STRUCTURE_ERROR = SERIALIZATION, "DeserializationDictStructureError";
    pub MALFORMED_PAYLOAD_ERROR = SERIALIZATION, "MalformedPayloadError";

    pub(crate) ENTRY_LOOKUP_ERROR = EXPRESSION_ERRORS, "EntryLookupError";
    pub(crate) ENTRY_REGISTRATION_ERROR = EXPRESSION_ERRORS, "EntryRegistrationError";
    pub(crate) NATIVE_CONSTANT_BINDING_ERROR = EXPRESSION_ERRORS, "NativeConstantBindingError";
    pub(crate) NATIVE_CONSTANT_LOWERING_ERROR = EXPRESSION_ERRORS, "NativeConstantLoweringError";
    pub(crate) NATIVE_RESULT_SORT_ERROR = EXPRESSION_ERRORS, "NativeResultSortError";
    pub(crate) NON_FINITE_CAST_ERROR = EXPRESSION_ERRORS, "NonFiniteCastError";
    pub(crate) STRING_LITERAL_PRECISION_ERROR = EXPRESSION_ERRORS, "StringLiteralPrecisionError";
    pub(crate) UNBOUND_VARIABLE_ERROR = EXPRESSION_ERRORS, "UnboundVariableError";
    pub(crate) UNSUPPORTED_NUMPY_LOWERING_ERROR = EXPRESSION_ERRORS, "UnsupportedNumpyLoweringError";
    pub(crate) NON_BOOLEAN_LOGICAL_OPERAND_ERROR = EXPRESSION_ERRORS, "NonBooleanLogicalOperandError";
    pub(crate) UNDECIDABLE_ERROR = EXPRESSION_ERRORS, "UndecidableError";
    pub(crate) COMPLEX_INFINITY_LIFT_ERROR = EXPRESSION_ERRORS, "ComplexInfinityLiftError";
    pub(crate) PARTIAL_PIECEWISE_ERROR = EXPRESSION_ERRORS, "PartialPiecewiseError";
    pub(crate) FUNCTION_ARITY_ERROR = "fhy_core.symbolic.expression.passes.inline", "FunctionArityError";
    pub(crate) REWRITE_CALLBACK_ERROR = "fhy_core.symbolic.expression.pattern.rewrite", "RewriteCallbackError";
    pub(crate) REWRITE_REBUILD_ERROR = "fhy_core.symbolic.expression.pattern.rewrite", "RewriteRebuildError";

    pub(crate) CONSTRAINT_ERROR = CONSTRAINT_ERRORS, "ConstraintError";
    pub(crate) MISSING_SYMBOL_TYPE_ERROR = CONSTRAINT_ERRORS, "MissingSymbolTypeError";
    pub(crate) PARAM_ERROR = "fhy_core.symbolic.param.values", "ParamError";

    pub(crate) SOLVER_CAPABILITY_ERROR = SOLVER, "SolverCapabilityError";
    pub(crate) SOLVER_BACKEND_ERROR = SOLVER, "SolverBackendError";
    pub(crate) SOLVER_BACKEND_UNAVAILABLE_ERROR = SOLVER, "SolverBackendUnavailableError";

    pub(crate) PASS_EXECUTION_ERROR = PASS, "PassExecutionError";
    pub(crate) PASS_VALIDATION_ERROR = PASS, "PassValidationError";
    pub(crate) PASS_REGISTRATION_ERROR = PASS, "PassRegistrationError";
    pub(crate) VALIDATION_FAILED_ERROR = "fhy_core.diagnostic", "ValidationFailedError";

    pub FROZEN_MUTATION_ERROR = "fhy_core.traits.frozen", "FrozenMutationError";
    pub(crate) FROZEN_VALIDATION_ERROR = "fhy_core.traits.frozen", "FrozenValidationError";
    pub(crate) VERIFICATION_ERROR = "fhy_core.traits.verifiable", "VerificationError";
    pub(crate) CORE_TYPE_ERROR = "fhy_core.types.core", "FhYCoreTypeError";
    pub(crate) SYMBOL_TABLE_ERROR = "fhy_core.symbol_table", "SymbolTableError";
    pub EQUIVALENCE_DERIVATION_ERROR = "fhy_core.term.derived_equivalence", "EquivalenceDerivationError";
}

#[cfg(test)]
mod tests {
    use pyo3::exceptions::{PyModuleNotFoundError, PyTypeError, PyValueError};

    use crate::util::testing::{evaluate, with_stand_ins};

    use super::*;

    static VALUE_ERROR: ExceptionClass = ExceptionClass::new("builtins", "ValueError");
    static MISSING: ExceptionClass = ExceptionClass::new("fhy_core_no_such_module", "Missing");
    static KEYWORD_ERROR: ExceptionClass =
        ExceptionClass::new("fhy_core.serialization", "KeywordError");
    static NOT_A_CLASS: ExceptionClass = ExceptionClass::new("builtins", "len");

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
    fn a_class_hands_back_the_one_class_it_imported() {
        with_stand_ins(|py| {
            let class = VALUE_ERROR.class(py).expect("a class");

            assert!(class.is(evaluate(py, "ValueError")));
            assert!(class.is(VALUE_ERROR.class(py).expect("a class")));
            MISSING.class(py).expect_err("no such module");
            let error = NOT_A_CLASS.class(py).expect_err("len is no class");
            assert!(error.is_instance_of::<PyTypeError>(py));
        });
    }

    #[test]
    fn build_passes_the_arguments_and_keywords_to_the_class() {
        with_stand_ins(|py| {
            let keywords = pyo3::types::PyDict::new(py);
            keywords.set_item("field", 3).expect("set");

            let error = KEYWORD_ERROR
                .build(py, ("a", 1), Some(&keywords))
                .expect("builds");

            assert!(KEYWORD_ERROR.is_instance_of(py, &error));
            let value = error.value(py);
            assert_eq!(value.getattr("args").unwrap().to_string(), "('a', 1)");
            assert_eq!(
                value.getattr("keywords").unwrap().to_string(),
                "{'field': 3}"
            );
            let failed = VALUE_ERROR
                .build(py, ("a",), Some(&keywords))
                .expect_err("ValueError takes no keywords");
            assert!(failed.is_instance_of::<PyTypeError>(py));
            MISSING.build(py, (), None).expect_err("no such module");
        });
    }

    #[test]
    fn the_public_classes_are_the_frameworks() {
        with_stand_ins(|py| {
            let serialization = SERIALIZATION_ERROR.class(py).expect("class");
            for class in [
                &DESERIALIZATION_VALUE_ERROR,
                &DESERIALIZATION_DICT_STRUCTURE_ERROR,
                &MALFORMED_PAYLOAD_ERROR,
            ] {
                assert!(
                    class
                        .class(py)
                        .expect("class")
                        .is_subclass(serialization)
                        .unwrap()
                );
            }
            let expected: [(&ExceptionClass, &str); 6] = [
                (&SERIALIZATION_ERROR, "SerializationError"),
                (&DESERIALIZATION_VALUE_ERROR, "DeserializationValueError"),
                (
                    &DESERIALIZATION_DICT_STRUCTURE_ERROR,
                    "DeserializationDictStructureError",
                ),
                (&MALFORMED_PAYLOAD_ERROR, "MalformedPayloadError"),
                (&FROZEN_MUTATION_ERROR, "FrozenMutationError"),
                (&EQUIVALENCE_DERIVATION_ERROR, "EquivalenceDerivationError"),
            ];
            for (class, name) in expected {
                let error = if name == "DeserializationDictStructureError" {
                    class.err(py, ("Cls", "expected", "data"))
                } else {
                    class.err(py, ("message",))
                };
                assert_eq!(error.get_type(py).name().unwrap().to_string(), name);
                assert!(class.is_instance_of(py, &error));
                if name != "DeserializationDictStructureError" {
                    assert_eq!(error.value(py).to_string(), "message");
                }
            }
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
            let kept = boxed_error_to_py(Box::new(PyValueError::new_err("kept")));
            assert!(kept.is_instance_of::<PyValueError>(py));
        });
    }
}
