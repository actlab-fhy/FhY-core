//! Holding function bodies to their declared result sorts.

use std::collections::HashMap;

use crate::expression::builtins::BuiltinFunction;
use crate::expression::registry::{FunctionRegistry, RegistryEntry};
use crate::expression::{Expression, FunctionName, FunctionSort, SortLookup};
use crate::identifier::Identifier;

use super::super::core_data_type::CoreDataType;
use super::super::data_type::DataType;
use super::super::qualifier::TypeQualifier;
use super::super::ty::{NumericalType, Type};
use super::checker::{CallTargets, TypeChecker};
use super::error::{BodyCheckError, TypeCheckError};

/// The declared signature of a function whose body is checked.
#[derive(Debug, Clone, Copy)]
pub struct FunctionSignature<'s> {
    name: &'s FunctionName,
    parameters: &'s [Identifier],
    parameter_sorts: &'s [FunctionSort],
    result_sort: FunctionSort,
}

impl<'s> FunctionSignature<'s> {
    /// Return the signature of the function `name` of `parameters`, each of
    /// the sort at its position in `parameter_sorts`, returning
    /// `result_sort`.
    #[must_use]
    pub fn new(
        name: &'s FunctionName,
        parameters: &'s [Identifier],
        parameter_sorts: &'s [FunctionSort],
        result_sort: FunctionSort,
    ) -> Self {
        Self {
            name,
            parameters,
            parameter_sorts,
            result_sort,
        }
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
    let function = signature.name;
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

/// Hold every function body to its declared result sort: the composed
/// built-in functions, in catalogue order, then the functions of
/// `registry`, in registration order, resolving calls through `registry`
/// and not deferring any.
///
/// Returns each failing function's name and error; the list is empty when
/// every body checks out.
#[must_use]
pub fn check_all_function_bodies(
    registry: &FunctionRegistry,
) -> Vec<(FunctionName, BodyCheckError)> {
    let mut failures = Vec::new();
    for function in BuiltinFunction::iter() {
        let Some(composed) = function.composed() else {
            continue;
        };
        let Ok(name) = FunctionName::new(function.name()) else {
            continue;
        };
        let signature = FunctionSignature::new(
            &name,
            composed.parameters(),
            function.parameter_sorts(),
            function.result_sort(),
        );
        if let Err(error) =
            check_function_body(&signature, composed.body(), registry, registry, false)
        {
            failures.push((name.clone(), error));
        }
    }
    for entry in registry.iter() {
        let RegistryEntry::Function(definition) = entry else {
            continue;
        };
        let signature = FunctionSignature::new(
            definition.name(),
            definition.parameters(),
            definition.parameter_sorts(),
            definition.result_sort(),
        );
        if let Err(error) =
            check_function_body(&signature, definition.body(), registry, registry, false)
        {
            failures.push((definition.name().clone(), error));
        }
    }
    failures
}
