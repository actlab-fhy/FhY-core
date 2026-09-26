//! The walk of [`FunctionRegistry::inline`](super::FunctionRegistry::inline).
//!
//! The walk is post-order on an explicit stack. A node's children are
//! inlined first; a call of a function with a body then has the body
//! substituted with the inlined arguments, and the substituted body is
//! walked in turn, with the function marked as in progress until its body
//! is done. Every node's result is remembered by the node's identity, and
//! so is every result's own (a result has nothing left to inline), so a
//! node met again, at another place of the input or inside a substituted
//! body, is not walked again. A substituted body holds each argument's
//! result, which the walk then meets as a remembered result: a body using a
//! parameter twice costs no more than one using it once.

use std::collections::{HashMap, HashSet};

use crate::expression::builtins::BuiltinFunction;
use crate::expression::callee::{Callee, FunctionName};
use crate::expression::error::RebuildError;
use crate::expression::node::{CallExpression, Expression, ExpressionKind};
use crate::identifier::Identifier;
use crate::tree::{BuildIdentityHasher, NodeHandle, NodeIdentity};

use super::{FunctionRegistry, InlineError, StoredEntry};

/// One step of the walk.
enum Step {
    /// Inline `node`: push its result, walking its children first.
    Enter(Expression),
    /// Combine the results of `node`'s children, the last ones on the result
    /// stack, into its result.
    Exit(Expression),
    /// The last result is the inlined body of the call `node`, of the
    /// function `function` when it is a user function, which is no longer in
    /// progress.
    FinishCall {
        node: Expression,
        function: Option<FunctionName>,
    },
}

/// What a call's callee resolves to.
enum Resolution<'r> {
    /// A function with a body: its parameters and its body.
    Body {
        parameters: &'r [Identifier],
        body: &'r Expression,
    },
    /// A native function of this many parameters, whose call is kept.
    Native(usize),
}

/// The state of one inlining walk.
struct Inliner<'r> {
    registry: &'r FunctionRegistry,
    /// Each walked node's result, by the node's identity, with the node, so
    /// that its identity stays unique while the walk runs.
    results: HashMap<NodeIdentity, (Expression, Expression), BuildIdentityHasher>,
    /// The user functions whose bodies are being walked.
    in_progress: HashSet<FunctionName>,
}

impl<'r> Inliner<'r> {
    /// Return what `call`'s callee resolves to.
    fn resolve(&self, call: &CallExpression) -> Result<Resolution<'r>, InlineError> {
        match call.callee() {
            Callee::Builtin(function) => Ok(resolve_builtin(*function)),
            Callee::Named(name) => {
                if self.in_progress.contains(name) {
                    return Err(InlineError::Recursive(name.clone()));
                }
                match self.registry.find(name.as_str()) {
                    None => Err(InlineError::UnknownFunction(name.clone())),
                    Some(StoredEntry::Constant(..)) => Err(InlineError::NotCallable(name.clone())),
                    Some(StoredEntry::Native(function)) => {
                        Ok(Resolution::Native(function.parameter_sorts().len()))
                    }
                    Some(StoredEntry::Function(function)) => Ok(Resolution::Body {
                        parameters: function.parameters(),
                        body: function.body(),
                    }),
                }
            }
        }
    }

    /// Remember `result` as the result of `node`, and as its own.
    fn remember(&mut self, node: Expression, result: &Expression) {
        if !Expression::ptr_eq(&node, result) {
            self.results
                .insert(result.identity(), (result.clone(), result.clone()));
        }
        self.results.insert(node.identity(), (node, result.clone()));
    }

    /// Return the result of `expression`.
    fn run(mut self, expression: &Expression) -> Result<Expression, InlineError> {
        let mut outputs: Vec<Expression> = Vec::new();
        let mut pending = vec![Step::Enter(expression.clone())];
        while let Some(step) = pending.pop() {
            match step {
                Step::Enter(node) => {
                    if let Some((_, result)) = self.results.get(&node.identity()) {
                        outputs.push(result.clone());
                        continue;
                    }
                    if matches!(
                        node.kind(),
                        ExpressionKind::Identifier(_) | ExpressionKind::Literal(_)
                    ) {
                        outputs.push(node);
                        continue;
                    }
                    let children: Vec<Expression> = node.children().cloned().collect();
                    pending.push(Step::Exit(node));
                    pending.extend(children.into_iter().rev().map(Step::Enter));
                }
                Step::Exit(node) => {
                    let child_count = node.children().count();
                    let children = outputs.split_off(outputs.len() - child_count);
                    if let ExpressionKind::Call(call) = node.kind() {
                        if let Some(body) = self.expand_call(call, &children)? {
                            let function = match call.callee() {
                                Callee::Named(name) => {
                                    self.in_progress.insert(name.clone());
                                    Some(name.clone())
                                }
                                Callee::Builtin(_) => None,
                            };
                            pending.push(Step::FinishCall {
                                node: node.clone(),
                                function,
                            });
                            pending.push(Step::Enter(body));
                            continue;
                        }
                    }
                    let result = rebuild(&node, children)?;
                    self.remember(node, &result);
                    outputs.push(result);
                }
                Step::FinishCall { node, function } => {
                    if let Some(function) = function {
                        self.in_progress.remove(&function);
                    }
                    let result = outputs
                        .last()
                        .cloned()
                        .unwrap_or_else(|| unreachable!("the body's walk yields its result"));
                    self.remember(node, &result);
                }
            }
        }
        Ok(outputs
            .pop()
            .unwrap_or_else(|| unreachable!("the walk yields the root's result")))
    }

    /// Return the body of `call` over the inlined `arguments` when its
    /// callee has a body, or `None` when the call is kept, once its
    /// argument count is checked.
    fn expand_call(
        &self,
        call: &CallExpression,
        arguments: &[Expression],
    ) -> Result<Option<Expression>, InlineError> {
        let resolution = self.resolve(call)?;
        let expected = match &resolution {
            Resolution::Body { parameters, .. } => parameters.len(),
            Resolution::Native(parameter_count) => *parameter_count,
        };
        if arguments.len() != expected {
            return Err(InlineError::ArityMismatch {
                callee: call.callee().clone(),
                expected,
                actual: arguments.len(),
            });
        }
        let Resolution::Body { parameters, body } = resolution else {
            return Ok(None);
        };
        let replacements: HashMap<Identifier, Expression> = parameters
            .iter()
            .cloned()
            .zip(arguments.iter().cloned())
            .collect();
        body.substitute(&replacements)
            .map(Some)
            .map_err(InlineError::Piecewise)
    }
}

/// Return what a call of the built-in `function` resolves to.
fn resolve_builtin(function: BuiltinFunction) -> Resolution<'static> {
    match function.composed() {
        Some(composed) => Resolution::Body {
            parameters: composed.parameters(),
            body: composed.body(),
        },
        None => Resolution::Native(function.parameter_sorts().len()),
    }
}

/// Return `node` with its children replaced by `children`, or `node` itself
/// when every child is the one it has.
fn rebuild(node: &Expression, children: Vec<Expression>) -> Result<Expression, InlineError> {
    if node
        .children()
        .zip(&children)
        .all(|(child, result)| Expression::ptr_eq(child, result))
    {
        return Ok(node.clone());
    }
    node.rebuild_with_children(children)
        .map_err(|error| match error {
            RebuildError::Piecewise(error) => InlineError::Piecewise(error),
            RebuildError::ChildCount { .. } => {
                unreachable!("a node is rebuilt from exactly its own children")
            }
        })
}

/// Inline `expression` through `registry`, as
/// [`FunctionRegistry::inline`] documents.
pub(super) fn inline(
    registry: &FunctionRegistry,
    expression: &Expression,
) -> Result<Expression, InlineError> {
    Inliner {
        registry,
        results: HashMap::default(),
        in_progress: HashSet::new(),
    }
    .run(expression)
}
