//! Holding function bodies to their declared result sorts.

use std::collections::HashMap;
use std::fmt;

use crate::expression::builtins::BuiltinFunction;
use crate::expression::registry::{FunctionRegistry, RegistryEntry};
use crate::expression::{Expression, FunctionName, FunctionSort, SortLookup};
use crate::identifier::Identifier;

use super::super::core_data_type::CoreDataType;
use super::super::data_type::DataType;
use super::super::qualifier::TypeQualifier;
use super::super::ty::{NumericalType, Type};
use super::checker::{CallTargets, TypeChecker};
use super::error::{BodyCheckError, SignatureError, TypeCheckError};

/// The function a body check names: a composed built-in function, whose
/// name no [`FunctionName`] holds, or a user function.
///
/// Displays the function's name.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[expect(
    clippy::exhaustive_enums,
    reason = "a checked body is a built-in's or a user function's; callers match both"
)]
pub enum FunctionLabel {
    /// A composed built-in function.
    Builtin(BuiltinFunction),
    /// A user function.
    User(FunctionName),
}

impl fmt::Display for FunctionLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Builtin(function) => f.write_str(function.name()),
            Self::User(name) => write!(f, "{name}"),
        }
    }
}

impl From<BuiltinFunction> for FunctionLabel {
    fn from(function: BuiltinFunction) -> Self {
        Self::Builtin(function)
    }
}

impl From<FunctionName> for FunctionLabel {
    fn from(name: FunctionName) -> Self {
        Self::User(name)
    }
}

impl From<&FunctionName> for FunctionLabel {
    fn from(name: &FunctionName) -> Self {
        Self::User(name.clone())
    }
}

/// The declared signature of a function whose body is checked.
#[derive(Debug, Clone)]
pub struct FunctionSignature<'s> {
    label: FunctionLabel,
    parameters: &'s [Identifier],
    parameter_sorts: &'s [FunctionSort],
    result_sort: FunctionSort,
}

impl<'s> FunctionSignature<'s> {
    /// Return the signature of the function `label` of `parameters`, each of
    /// the sort at its position in `parameter_sorts`, returning
    /// `result_sort`.
    ///
    /// # Errors
    ///
    /// Returns [`SignatureError::LengthMismatch`] when `parameters` and
    /// `parameter_sorts` differ in length.
    pub fn new(
        label: impl Into<FunctionLabel>,
        parameters: &'s [Identifier],
        parameter_sorts: &'s [FunctionSort],
        result_sort: FunctionSort,
    ) -> Result<Self, SignatureError> {
        let label = label.into();
        if parameters.len() != parameter_sorts.len() {
            return Err(SignatureError::LengthMismatch {
                function: label,
                parameters: parameters.len(),
                parameter_sorts: parameter_sorts.len(),
            });
        }
        Ok(Self {
            label,
            parameters,
            parameter_sorts,
            result_sort,
        })
    }

    /// Return the label of the function.
    #[must_use]
    pub fn label(&self) -> &FunctionLabel {
        &self.label
    }
}

/// The outcome of a body check that raised no error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BodyCheck {
    /// The body satisfies its declared result sort.
    Checked,
    /// The body calls a function no target resolves yet, and the check was
    /// asked to defer such calls, so it was abandoned.
    Deferred,
}

/// Hold the body `body` of the function of `signature` to its declared
/// result sort.
///
/// The body is synthesized with each parameter a `Param` scalar of the
/// concrete type of its sort ([`CoreDataType::of_sort`]), calls resolved
/// through `targets` and native constants through `sorts`; its type must
/// be a scalar numerical type of a primitive data type compatible with the
/// result sort. A call no target resolves abandons the check when
/// `defer_unresolved_calls` holds, and fails it otherwise.
///
/// # Errors
///
/// Returns the [`BodyCheckError`] of the first failure.
pub fn check_function_body(
    signature: &FunctionSignature<'_>,
    body: &Expression,
    targets: &dyn CallTargets,
    sorts: &dyn SortLookup,
    defer_unresolved_calls: bool,
) -> Result<BodyCheck, BodyCheckError> {
    let function = &signature.label;
    let parameters: HashMap<Identifier, (Type, TypeQualifier)> = signature
        .parameters
        .iter()
        .zip(signature.parameter_sorts)
        .map(|(parameter, sort)| {
            (
                parameter.clone(),
                (
                    Type::Numerical(NumericalType::scalar(CoreDataType::of_sort(*sort))),
                    TypeQualifier::Param,
                ),
            )
        })
        .collect();
    let checker = TypeChecker::new(&parameters, targets, sorts).with_deferred_unknown_calls();
    let body_type = match checker.synthesize(body) {
        Ok((body_type, _)) => body_type,
        Err(TypeCheckError::UnknownCall(_)) if defer_unresolved_calls => {
            return Ok(BodyCheck::Deferred);
        }
        Err(TypeCheckError::UnknownCall(error)) => {
            return Err(BodyCheckError::UnknownCall {
                function: function.clone(),
                error,
            });
        }
        Err(TypeCheckError::Callback(source)) => return Err(BodyCheckError::Callback(source)),
        Err(error @ TypeCheckError::Rule { .. }) => {
            let is_unsupported =
                matches!(&error, TypeCheckError::Rule { rule, .. } if rule.is_unsupported());
            let function = function.clone();
            return Err(if is_unsupported {
                BodyCheckError::Unsupported { function, error }
            } else {
                BodyCheckError::IllTyped { function, error }
            });
        }
    };
    let core = match &body_type {
        Type::Numerical(numerical) => match numerical.data_type() {
            DataType::Primitive(core) => Some(*core),
            _ => None,
        },
        _ => None,
    };
    let Some(core) = core else {
        return Err(BodyCheckError::NotScalar {
            function: function.clone(),
            body_type,
        });
    };
    if !core.is_compatible_with_sort(signature.result_sort) {
        return Err(BodyCheckError::IncompatibleResult {
            function: function.clone(),
            body: core,
            sort: signature.result_sort,
        });
    }
    Ok(BodyCheck::Checked)
}

/// What a sweep of function bodies checked, and what failed.
#[derive(Debug)]
pub struct BodySweep {
    checked: Vec<FunctionLabel>,
    failures: Vec<(FunctionLabel, BodyCheckError)>,
}

impl BodySweep {
    /// Return the label of every function whose body was checked, in the
    /// order checked: the composed built-ins in catalogue order, then the
    /// user functions in registration order.
    #[must_use]
    pub fn checked(&self) -> &[FunctionLabel] {
        &self.checked
    }

    /// Return each failing function's label and error, in the order
    /// checked.
    #[must_use]
    pub fn failures(&self) -> &[(FunctionLabel, BodyCheckError)] {
        &self.failures
    }

    /// Return each failing function's label and error, in the order
    /// checked.
    #[must_use]
    pub fn into_failures(self) -> Vec<(FunctionLabel, BodyCheckError)> {
        self.failures
    }
}

/// Hold every function body to its declared result sort: the composed
/// built-in functions, in catalogue order, then the functions of
/// `registry`, in registration order, resolving calls through `registry`
/// and not deferring any.
///
/// Returns the labels checked and each failing function's label and error;
/// no failure means every body checks out.
#[must_use]
pub fn check_all_function_bodies(registry: &FunctionRegistry) -> BodySweep {
    sweep(
        registry,
        BuiltinFunction::iter().filter_map(|function| {
            let composed = function.composed()?;
            Some((function, composed.parameters(), composed.body()))
        }),
    )
}

/// Sweep the built-in bodies `builtins`, each a function with its
/// parameters and body, then the user functions of `registry`: the seam
/// through which a test hands the sweep a broken built-in body.
fn sweep<'b>(
    registry: &FunctionRegistry,
    builtins: impl IntoIterator<Item = (BuiltinFunction, &'b [Identifier], &'b Expression)>,
) -> BodySweep {
    let mut sweep = BodySweep {
        checked: Vec::new(),
        failures: Vec::new(),
    };
    let mut check = |signature: Result<FunctionSignature<'_>, SignatureError>,
                     body: &Expression| {
        let (label, result) = match signature {
            Ok(signature) => {
                let result = check_function_body(&signature, body, registry, registry, false);
                (signature.label, result)
            }
            Err(error) => {
                let SignatureError::LengthMismatch { function, .. } = &error;
                (function.clone(), Err(BodyCheckError::Signature(error)))
            }
        };
        if let Err(error) = result {
            sweep.failures.push((label.clone(), error));
        }
        sweep.checked.push(label);
    };
    for (function, parameters, body) in builtins {
        check(
            FunctionSignature::new(
                function,
                parameters,
                function.parameter_sorts(),
                function.result_sort(),
            ),
            body,
        );
    }
    for entry in registry.iter() {
        let RegistryEntry::Function(definition) = entry else {
            continue;
        };
        check(
            FunctionSignature::new(
                definition.name(),
                definition.parameters(),
                definition.parameter_sorts(),
                definition.result_sort(),
            ),
            definition.body(),
        );
    }
    sweep
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::LiteralValue;

    #[test]
    fn a_broken_builtin_body_is_reported_by_its_label() {
        let x = Identifier::new("x");
        let y = Identifier::new("y");
        let parameters = [x.clone(), y];
        let broken = Expression::from(x) + LiteralValue::Bool(true);

        let result = sweep(
            &FunctionRegistry::new(),
            [(BuiltinFunction::Max, &parameters[..], &broken)],
        );

        assert_eq!(
            result.checked(),
            [FunctionLabel::Builtin(BuiltinFunction::Max)]
        );
        let [(label, error)] = result.failures() else {
            panic!("one failure, got {:?}", result.failures());
        };
        assert_eq!(*label, FunctionLabel::Builtin(BuiltinFunction::Max));
        assert!(
            matches!(error, BodyCheckError::IllTyped { function, .. } if *function == *label),
            "{error:?}"
        );
        assert!(
            error
                .to_string()
                .starts_with("function 'max' body failed to type-check"),
            "{error}"
        );
    }
}
