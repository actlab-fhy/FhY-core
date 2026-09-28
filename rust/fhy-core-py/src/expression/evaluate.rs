//! The expression evaluators: the fold of `evaluate_expression`, the
//! `NumPy` evaluator, the built-ins' native implementations, and the literal
//! helpers of `native_lowering.py`, over the core's
//! [`fhy_core::expression::evaluate`].

mod builtins;
mod error;
mod fold;
mod literal;
mod numpy;

pub(crate) use builtins::PyBuiltinNativeImplementation;
pub(in crate::expression) use builtins::builtin_implementation;
pub(crate) use fold::fold_expression;
pub(crate) use literal::{coerce_literal_value, is_decimal_text_exactly_binary};
pub(crate) use numpy::evaluate_expression_with_numpy;
