//! [`EquationConstraint`]: a Boolean expression that must hold.

use std::collections::{HashMap, HashSet};

use crate::expression::{BooleanScreen, Expression, ExpressionKind, LiteralValue};
use crate::identifier::Identifier;
use crate::term::AlphaRenaming;

use super::Outcome;
use super::binding::{Binding, Bindings};
use super::context::{ConstraintContext, Event};
use super::error::{ConstraintError, UnusableBindingReason};
use super::key;
use super::value::Value;

/// A Boolean expression that must hold under an assignment of its free
/// identifiers, which are the constraint's scope.
///
/// Cloning one shares its expression.
#[derive(Debug, Clone)]
pub struct EquationConstraint {
    expression: Expression,
}

impl EquationConstraint {
    /// Return the constraint that `expression` holds.
    #[must_use]
    pub fn new(expression: Expression) -> Self {
        Self { expression }
    }

    /// Return the expression.
    #[must_use]
    pub fn expression(&self) -> &Expression {
        &self.expression
    }

    /// Return the scope: the expression's free identifiers.
    #[must_use]
    pub fn free_identifiers(&self) -> HashSet<Identifier> {
        self.expression.free_identifiers()
    }

    /// Decide the constraint under `bindings`.
    ///
    /// In order, it:
    ///
    /// 1. keeps the bindings of identifiers in its scope, in the order they
    ///    were made, and never reads the others; a literal value becomes a
    ///    literal expression;
    /// 2. screens the expression as a predicate, with those bindings, as
    ///    [`BooleanScreen::check_predicate`] judges it;
    /// 3. answers [`Outcome::Undecided`], reporting
    ///    [`Event::BoundNativeConstants`], if the bindings bind a native
    ///    constant;
    /// 4. simplifies the expression under the bindings with the context's
    ///    solver, and reads a Boolean literal as
    ///    [`Satisfied`](Outcome::Satisfied) or
    ///    [`Violated`](Outcome::Violated), and anything that is not a
    ///    literal as [`Undecided`](Outcome::Undecided), reporting
    ///    [`Event::Residual`].
    ///
    /// # Errors
    ///
    /// Returns [`ConstraintError::UnusableBinding`] for a value that is not
    /// a literal, or a string outside the literal grammar, in the scope;
    /// [`ConstraintError::IllTyped`] if the expression cannot be a
    /// predicate; [`ConstraintError::NonBooleanResult`] if it simplifies to
    /// a literal that is not a Boolean; and [`ConstraintError::Solve`] if
    /// the solver refuses or fails the simplification.
    pub fn evaluate(
        &self,
        bindings: &Bindings,
        context: &ConstraintContext<'_>,
    ) -> Result<Outcome, ConstraintError> {
        let scope = self.free_identifiers();
        let mut environment: HashMap<Identifier, Expression> = HashMap::new();
        for (identifier, binding) in bindings.iter() {
            if scope.contains(identifier) {
                environment.insert(identifier.clone(), lift_binding(identifier, binding)?);
            }
        }
        BooleanScreen::new()
            .with_sorts(context.sorts())
            .with_environment(&environment)
            .check_predicate(&self.expression)
            .map_err(ConstraintError::IllTyped)?;
        let mut constants: Vec<Identifier> = environment
            .keys()
            .filter(|identifier| context.is_native_constant(identifier))
            .cloned()
            .collect();
        if !constants.is_empty() {
            constants.sort_by_key(Identifier::id);
            context.notify(&Event::BoundNativeConstants {
                identifiers: &constants,
            });
            return Ok(Outcome::Undecided);
        }
        let result = context
            .solver()
            .simplify(&self.expression, &environment, &context.simplify_context())
            .map_err(ConstraintError::Solve)?;
        match result.kind() {
            ExpressionKind::Literal(LiteralValue::Bool(true)) => Ok(Outcome::Satisfied),
            ExpressionKind::Literal(LiteralValue::Bool(false)) => Ok(Outcome::Violated),
            ExpressionKind::Literal(_) => Err(ConstraintError::NonBooleanResult {
                predicate: self.expression.clone(),
                result,
            }),
            _ => {
                context.notify(&Event::Residual {
                    residual: &result,
                    has_free_identifiers: !result.free_identifiers().is_empty(),
                });
                Ok(Outcome::Undecided)
            }
        }
    }

    /// Return the canonical ordering key: `equation|` and the tree's key.
    #[must_use]
    pub fn ordering_key(&self) -> String {
        key::equation_key(&self.expression)
    }

    /// Return whether `other` holds a structurally equal expression.
    #[must_use]
    pub fn is_structurally_equivalent(&self, other: &Self) -> bool {
        self.expression == other.expression
    }

    /// Return whether `other` holds an alpha-equivalent expression under
    /// `renaming`.
    #[must_use]
    pub fn is_alpha_equivalent_under(&self, other: &Self, renaming: &AlphaRenaming) -> bool {
        self.expression
            .is_alpha_equivalent_under(&other.expression, renaming)
    }
}

/// Return the expression `binding` puts in an equation's environment: an
/// expression itself, or the literal a literal value denotes.
fn lift_binding(identifier: &Identifier, binding: &Binding) -> Result<Expression, ConstraintError> {
    let refuse = |reason| ConstraintError::UnusableBinding {
        identifier: identifier.clone(),
        reason,
    };
    let literal = match binding {
        Binding::Expression(expression) => return Ok(expression.clone()),
        Binding::Value(value) => match value {
            Value::Bool(value) => LiteralValue::Bool(*value),
            Value::Int(value) => LiteralValue::Int(value.clone()),
            Value::Float(value) => LiteralValue::Float(*value),
            Value::Decimal(value) => LiteralValue::Decimal(value.clone()),
            Value::Str(text) => LiteralValue::parse_text(text)
                .map_err(|error| refuse(UnusableBindingReason::UnparsableText(error)))?,
            Value::Tuple(_) | Value::FrozenSet(_) | Value::Opaque(_) => {
                return Err(refuse(UnusableBindingReason::NotALiteral));
            }
        },
    };
    Ok(Expression::literal(literal))
}
