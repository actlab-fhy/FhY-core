//! Type checking of expressions against the type system.
//!
//! - [`TypeChecker`] synthesizes the type and qualifier of an expression
//!   bottom-up, or checks it against an expected type, handing the expected
//!   type to literals so a weak literal adopts its context. The types of
//!   identifiers come from an [`IdentifierTypes`], the signatures of calls
//!   from a [`CallTargets`], and the sorts of native constants from a
//!   [`SortLookup`](crate::expression::SortLookup).
//! - [`CoreDataType::is_compatible_with_sort`](super::CoreDataType::is_compatible_with_sort)
//!   and [`CoreDataType::of_sort`](super::CoreDataType::of_sort) map
//!   function sorts to core data types.
//! - [`check_function_body`] holds a function's body to its declared result
//!   sort, and [`check_all_function_bodies`] does so for every function a
//!   [`FunctionRegistry`](crate::expression::registry::FunctionRegistry) and
//!   the built-in catalogue define, naming each by its [`FunctionLabel`]
//!   and returning a [`BodySweep`] of the labels checked and the failures.
//!
//! The checker walks the expression on its own work list, so an expression
//! of any depth checks on a small stack.
//!
//! # Examples
//!
//! ```
//! use std::collections::HashMap;
//!
//! use fhy_core::expression::Expression;
//! use fhy_core::expression::registry::FunctionRegistry;
//! use fhy_core::identifier::Identifier;
//! use fhy_core::types::checking::TypeChecker;
//! use fhy_core::types::{CoreDataType, NumericalType, Type, TypeQualifier};
//!
//! let x = Identifier::new("x");
//! let int32 = Type::from(NumericalType::scalar(CoreDataType::Int32));
//! let identifiers = HashMap::from([(x.clone(), (int32.clone(), TypeQualifier::State))]);
//! let registry = FunctionRegistry::new();
//! let checker = TypeChecker::new(&identifiers, &registry, &registry);
//!
//! // The literal adopts the concrete type of the other operand.
//! let (synthesized, qualifier) = checker.synthesize(&(Expression::from(x) + 1))?;
//!
//! assert_eq!(synthesized, int32);
//! assert_eq!(qualifier, TypeQualifier::Temp);
//! # Ok::<(), fhy_core::types::checking::TypeCheckError>(())
//! ```

mod body;
mod checker;
mod error;
mod sort;

pub use body::{
    BodyCheck, BodySweep, FunctionLabel, FunctionSignature, check_all_function_bodies,
    check_function_body,
};
pub use checker::{CallTarget, CallTargets, IdentifierTypes, TypeChecker};
pub use error::{
    BodyCheckError, CallTargetError, SignatureError, TypeCheckError, TypeRule, TypeRuleKind,
};
