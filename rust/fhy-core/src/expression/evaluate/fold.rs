//! The walk of [`Evaluator::fold`](super::Evaluator::fold).
//!
//! The walk is post-order on an explicit stack. Every node's result is
//! remembered by the node's identity, with the node, so a node the tree
//! shares is folded once.

use std::collections::{HashMap, HashSet};

use num_bigint::BigInt;
use num_traits::{FromPrimitive, Signed, ToPrimitive};

use crate::expression::builtins::{BuiltinConstant, BuiltinFunction};
use crate::expression::callee::Callee;
use crate::expression::error::RebuildError;
use crate::expression::literal::LiteralValue;
use crate::expression::node::{CallExpression, Expression, ExpressionKind};
use crate::expression::registry::{FunctionRegistry, NativeFunction, RegistryEntry};
use crate::expression::sort::FunctionSort;
use crate::identifier::Identifier;
use crate::tree::{BuildIdentityHasher, NodeHandle, NodeIdentity};

use super::error::FoldError;
use super::{Folding, NativeCalls};

/// One step of the walk.
enum Step {
    /// Fold the node, its children first.
    Enter(Expression),
    /// Combine the results of the node's children, the last ones on the
    /// result stack.
    Exit(Expression),
}

/// What a folded call computes.
enum Target<'r> {
    /// A native built-in.
    Builtin(BuiltinFunction),
    /// A native user function.
    User(&'r NativeFunction),
}

/// The state of one fold.
struct Folder<'r, 'n> {
    registry: &'r FunctionRegistry,
    natives: &'n dyn NativeCalls,
    /// Each folded node's result, by the node's identity, with the node, so
    /// that its identity stays unique while the walk runs.
    results: HashMap<NodeIdentity, (Expression, Expression), BuildIdentityHasher>,
    not_inlined: Vec<Callee>,
    listed: HashSet<Callee>,
}

impl Folder<'_, '_> {
    /// Return the fold of `expression`.
    fn run(mut self, expression: &Expression) -> Result<Folding, FoldError> {
        let mut outputs: Vec<Expression> = Vec::new();
        let mut pending = vec![Step::Enter(expression.clone())];
        while let Some(step) = pending.pop() {
            match step {
                Step::Enter(node) => {
                    if let Some((_, result)) = self.results.get(&node.identity()) {
                        outputs.push(result.clone());
                        continue;
                    }
                    match node.kind() {
                        ExpressionKind::Identifier(identifier) => {
                            let result = self
                                .fold_identifier(identifier)
                                .unwrap_or_else(|| node.clone());
                            self.results.insert(node.identity(), (node, result.clone()));
                            outputs.push(result);
                        }
                        ExpressionKind::Literal(_) => outputs.push(node),
                        _ => {
                            let children: Vec<Expression> = node.children().cloned().collect();
                            pending.push(Step::Exit(node));
                            pending.extend(children.into_iter().rev().map(Step::Enter));
                        }
                    }
                }
                Step::Exit(node) => {
                    let child_count = node.children().count();
                    let children = outputs.split_off(outputs.len() - child_count);
                    let folded = match node.kind() {
                        ExpressionKind::Call(call) => self.fold_call(call, &children)?,
                        _ => None,
                    };
                    let result = match folded {
                        Some(literal) => Expression::from(literal),
                        None => rebuild(&node, children)?,
                    };
                    self.results.insert(node.identity(), (node, result.clone()));
                    outputs.push(result);
                }
            }
        }
        let output = outputs
            .pop()
            .unwrap_or_else(|| unreachable!("the walk yields the root's result"));
        Ok(Folding {
            output,
            not_inlined: self.not_inlined,
        })
    }

    /// Return the value of the constant `identifier` refers to, if any.
    fn fold_identifier(&self, identifier: &Identifier) -> Option<Expression> {
        if let Some(constant) = BuiltinConstant::of_identifier(identifier) {
            return Some(Expression::from(LiteralValue::Float(constant.value())));
        }
        self.registry
            .constant(identifier)
            .map(|constant| Expression::from(constant.value().clone()))
    }

    /// Record that a call of `callee`, a function with a body, was kept.
    fn list_not_inlined(&mut self, callee: &Callee) {
        if self.listed.insert(callee.clone()) {
            self.not_inlined.push(callee.clone());
        }
    }

    /// Return the literal a call folds to, or `None` when it is kept.
    fn fold_call(
        &mut self,
        call: &CallExpression,
        arguments: &[Expression],
    ) -> Result<Option<LiteralValue>, FoldError> {
        let target = match call.callee() {
            Callee::Builtin(function) => {
                if function.composed().is_some() {
                    self.list_not_inlined(call.callee());
                    return Ok(None);
                }
                Target::Builtin(*function)
            }
            Callee::Named(name) => {
                if name.as_str().parse::<BuiltinConstant>().is_ok() {
                    return Err(FoldError::NotCallable(name.clone()));
                }
                match self.registry.entry(name.as_str()) {
                    None => return Err(FoldError::UnknownFunction(name.clone())),
                    Some(RegistryEntry::Constant(..)) => {
                        return Err(FoldError::NotCallable(name.clone()));
                    }
                    Some(RegistryEntry::Function(_)) => {
                        self.list_not_inlined(call.callee());
                        return Ok(None);
                    }
                    Some(RegistryEntry::Native(function)) => Target::User(function),
                }
            }
        };
        let mut literals = Vec::with_capacity(arguments.len());
        for argument in arguments {
            let ExpressionKind::Literal(literal) = argument.kind() else {
                return Ok(None);
            };
            literals.push(literal.clone());
        }
        let parameter_sorts = match &target {
            Target::Builtin(function) => function.parameter_sorts(),
            Target::User(function) => function.parameter_sorts(),
        };
        if literals.len() != parameter_sorts.len() {
            return Err(FoldError::Arity {
                callee: call.callee().clone(),
                expected: parameter_sorts.len(),
                actual: literals.len(),
            });
        }
        for (position, (literal, sort)) in literals.iter().zip(parameter_sorts).enumerate() {
            if !sort.accepts_literal(literal) {
                return Err(FoldError::ArgumentSort {
                    callee: call.callee().clone(),
                    position,
                    sort: *sort,
                    argument: literal.clone(),
                });
            }
        }
        for literal in &mut literals {
            if let LiteralValue::Decimal(decimal) = literal {
                let value = decimal
                    .to_f64_exact()
                    .ok_or_else(|| FoldError::InexactDecimal(decimal.clone()))?;
                *literal = LiteralValue::Float(value);
            }
        }
        match target {
            Target::Builtin(function) => fold_builtin(function, &literals[0]).map(Some),
            Target::User(function) => {
                let value =
                    self.natives
                        .call(function, &literals)
                        .map_err(|source| FoldError::Native {
                            function: function.name().clone(),
                            source,
                        })?;
                if !function.result_sort().accepts_literal(&value) {
                    return Err(FoldError::ResultSort {
                        function: function.name().clone(),
                        sort: function.result_sort(),
                        value,
                    });
                }
                Ok(Some(value))
            }
        }
    }
}

/// Return the literal of the native built-in `function` at `argument`, a
/// Boolean-free, decimal-free literal of its parameter's sort.
fn fold_builtin(
    function: BuiltinFunction,
    argument: &LiteralValue,
) -> Result<LiteralValue, FoldError> {
    let real = match argument {
        LiteralValue::Float(value) => *value,
        LiteralValue::Int(integer) => integer.to_f64().unwrap_or(if integer.is_negative() {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        }),
        LiteralValue::Bool(_) | LiteralValue::Decimal(_) => {
            unreachable!("the argument is checked and converted")
        }
    };
    let value = function
        .native_value(real)
        .unwrap_or_else(|| unreachable!("the function is native"));
    match function.result_sort() {
        FunctionSort::Real => Ok(LiteralValue::Float(value)),
        FunctionSort::Int | FunctionSort::Nat | FunctionSort::Bool => BigInt::from_f64(value)
            .filter(|_| value.is_finite())
            .map(LiteralValue::Int)
            .ok_or(FoldError::NonFiniteCast { function, value }),
    }
}

/// Return `node` with its children replaced by `children`, or `node` itself
/// when every child is the one it has.
fn rebuild(node: &Expression, children: Vec<Expression>) -> Result<Expression, FoldError> {
    if node
        .children()
        .zip(&children)
        .all(|(child, result)| Expression::ptr_eq(child, result))
    {
        return Ok(node.clone());
    }
    node.rebuild_with_children(children)
        .map_err(|error| match error {
            RebuildError::Piecewise(error) => FoldError::Piecewise(error),
            _ => unreachable!("a node is rebuilt from exactly its own children"),
        })
}

/// Fold `expression` over `registry` and `natives`, as
/// [`Evaluator::fold`](super::Evaluator::fold) documents.
pub(super) fn fold(
    registry: &FunctionRegistry,
    natives: &dyn NativeCalls,
    expression: &Expression,
) -> Result<Folding, FoldError> {
    Folder {
        registry,
        natives,
        results: HashMap::default(),
        not_inlined: Vec::new(),
        listed: HashSet::new(),
    }
    .run(expression)
}
