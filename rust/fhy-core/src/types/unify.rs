//! Structural equivalence, template binding, substitution and unification.
//!
//! The expression walks, the substitution of shape variables and the
//! occurs check, go through the core's iterative [`Expression::substitute`]
//! and [`Expression::free_identifiers`], so they reach every node kind and
//! any depth. Following the expression bindings keeps its chain on the heap
//! and substitutes each binding once per call, so a chain of any length,
//! and bindings shared along many paths, substitute in linear time.

use std::collections::{HashMap, HashSet};

use crate::expression::{Expression, ExpressionKind, PiecewiseError};
use crate::identifier::Identifier;

use super::data_type::{DataType, TemplateDataType};
use super::environment::TypeUnificationEnvironment;
use super::error::{TypeOperation, UnificationError};
use super::ty::{Dimension, IndexType, NumericalType, Type};

type Result<T> = std::result::Result<T, UnificationError>;

impl DataType {
    /// Return whether `other` is structurally equivalent: the same core data
    /// type, the same template (identifier and widths), or, when either side
    /// is an extension, what its [`is_structurally_equivalent`](super::DataTypeExtension::is_structurally_equivalent)
    /// answers about the other side, the left one's when both are.
    ///
    /// # Errors
    ///
    /// Returns [`UnificationError::Extension`] for an extension that fails.
    pub fn is_structurally_equivalent(&self, other: &DataType) -> Result<bool> {
        match (self, other) {
            (Self::Primitive(left), Self::Primitive(right)) => Ok(left == right),
            (Self::Template(left), Self::Template(right)) => Ok(left == right),
            (Self::Extension(extension), _) => extension
                .get()
                .is_structurally_equivalent(other)
                .map_err(UnificationError::Extension),
            (_, Self::Extension(extension)) => extension
                .get()
                .is_structurally_equivalent(self)
                .map_err(UnificationError::Extension),
            _ => Ok(false),
        }
    }

    /// Bind this data type, as a pattern, against `actual`.
    ///
    /// - A template binds `actual`, after checking its width constraint, or
    ///   checks its existing binding against it; against a template it must
    ///   be the same template, and binds nothing.
    /// - A primitive type must meet the same primitive type.
    /// - An extension binds through its hook, or by the default rule:
    ///   `actual` must be structurally equivalent.
    ///
    /// # Errors
    ///
    /// Returns the [`UnificationError`] of the rule that fails.
    pub fn bind_template(
        &self,
        actual: &DataType,
        environment: &TypeUnificationEnvironment,
    ) -> Result<TypeUnificationEnvironment> {
        match self {
            Self::Template(template) => bind_template_data_type(template, actual, environment),
            Self::Primitive(expected) => match actual {
                Self::Primitive(actual) if actual == expected => Ok(environment.clone()),
                Self::Primitive(actual) => Err(UnificationError::CoreDataTypeMismatch {
                    expected: *expected,
                    actual: *actual,
                }),
                _ => Err(UnificationError::KindMismatch {
                    operation: TypeOperation::Bind,
                    expected: self.kind_name(),
                    actual: actual.kind_name(),
                }),
            },
            Self::Extension(extension) => extension.get().bind_template(self, actual, environment),
        }
    }

    /// Return this data type with its placeholders substituted: a template
    /// bound in `environment` becomes its binding, and an extension
    /// substitutes through its hook. Anything else is returned unchanged.
    ///
    /// # Errors
    ///
    /// Returns an extension's error.
    pub fn substitute_template(
        &self,
        environment: &TypeUnificationEnvironment,
    ) -> Result<DataType> {
        match self {
            Self::Template(template) => Ok(environment
                .data_type_binding(template.identifier())
                .cloned()
                .unwrap_or_else(|| self.clone())),
            Self::Primitive(_) => Ok(self.clone()),
            Self::Extension(extension) => extension.get().substitute_template(self, environment),
        }
    }
}

/// Bind the data type `this`, as a pattern, against `actual` by the default
/// rule of an extension without a rule of its own: `actual` must be
/// structurally equivalent, and nothing is bound.
///
/// # Errors
///
/// Returns [`UnificationError::DataTypeMismatch`] if `actual` is not
/// equivalent, and an extension's error.
pub fn default_bind_data_template(
    this: &DataType,
    actual: &DataType,
    environment: &TypeUnificationEnvironment,
) -> Result<TypeUnificationEnvironment> {
    if this.is_structurally_equivalent(actual)? {
        Ok(environment.clone())
    } else {
        Err(UnificationError::DataTypeMismatch {
            operation: TypeOperation::Bind,
            expected: this.clone(),
            actual: actual.clone(),
        })
    }
}

/// Return the data type `this` with its placeholders substituted by the
/// default rule of an extension without a rule of its own: `this`
/// unchanged.
///
/// # Errors
///
/// Never fails; the signature is the hook's.
pub fn default_substitute_data_template(
    this: &DataType,
    environment: &TypeUnificationEnvironment,
) -> Result<DataType> {
    let _ = environment;
    Ok(this.clone())
}

/// Bind the template `template` against `actual`.
fn bind_template_data_type(
    template: &TemplateDataType,
    actual: &DataType,
    environment: &TypeUnificationEnvironment,
) -> Result<TypeUnificationEnvironment> {
    if let DataType::Template(other) = actual {
        if other != template {
            return Err(UnificationError::DistinctTemplates {
                operation: TypeOperation::Bind,
                expected: template.clone(),
                actual: other.clone(),
            });
        }
        return Ok(environment.clone());
    }
    check_width_constraint(template, actual)?;
    let identifier = template.identifier();
    match environment.data_type_binding(identifier) {
        None => Ok(environment.with_data_type_binding(identifier.clone(), actual.clone())),
        Some(bound) if !bound.is_structurally_equivalent(actual)? => {
            Err(UnificationError::ConflictingDataTypeBinding {
                identifier: identifier.clone(),
                bound: bound.clone(),
                actual: actual.clone(),
            })
        }
        Some(_) => Ok(environment.clone()),
    }
}

/// Check that `actual` satisfies the width constraint of `template`: a
/// primitive type whose bit width is one of the widths. A weak type has no
/// width, so it never does.
fn check_width_constraint(template: &TemplateDataType, actual: &DataType) -> Result<()> {
    let Some(widths) = template.widths() else {
        return Ok(());
    };
    let DataType::Primitive(core_data_type) = actual else {
        return Err(UnificationError::WidthOnNonPrimitive {
            template: template.clone(),
            actual: actual.clone(),
        });
    };
    match core_data_type.bit_width() {
        Some(width) if widths.contains(&width) => Ok(()),
        _ => Err(UnificationError::WidthMismatch {
            template: template.clone(),
            actual: *core_data_type,
        }),
    }
}

impl Type {
    /// Return whether `other` is structurally equivalent.
    ///
    /// - Numerical types need equivalent data types and shapes of one rank
    ///   whose dimensions are equal expressions, or both the wildcard.
    /// - Index types need equal bounds and strides.
    /// - When either side is an extension, it answers about the other side
    ///   through its
    ///   [`is_structurally_equivalent`](super::TypeExtension::is_structurally_equivalent),
    ///   the left one when both are.
    ///
    /// # Errors
    ///
    /// Returns [`UnificationError::Extension`] for an extension that fails.
    pub fn is_structurally_equivalent(&self, other: &Type) -> Result<bool> {
        match (self, other) {
            (Self::Numerical(left), Self::Numerical(right)) => {
                Ok(NumericalType::ptr_eq(left, right)
                    || (left.shape() == right.shape()
                        && left
                            .data_type()
                            .is_structurally_equivalent(right.data_type())?))
            }
            (Self::Index(left), Self::Index(right)) => Ok(left == right),
            (Self::Extension(extension), _) => extension
                .get()
                .is_structurally_equivalent(other)
                .map_err(UnificationError::Extension),
            (_, Self::Extension(extension)) => extension
                .get()
                .is_structurally_equivalent(self)
                .map_err(UnificationError::Extension),
            _ => Ok(false),
        }
    }

    /// Bind this type, as a pattern, against `actual`.
    ///
    /// - A numerical pattern needs a numerical actual. Its data type binds
    ///   first. A template data type over the full-shape wildcard then binds
    ///   the whole actual type; the full-shape wildcard alone binds nothing
    ///   more; otherwise the ranks must agree, and each dimension binds: a
    ///   wildcard in the pattern matches any dimension, one in the actual is
    ///   refused, a shape variable binds or checks its binding, and any other
    ///   dimension must equal the actual one.
    /// - An index pattern needs an index actual, and binds its bounds and
    ///   stride as dimensions.
    /// - An extension binds through its hook, or by the default rule:
    ///   `actual` must be structurally equivalent.
    ///
    /// # Errors
    ///
    /// Returns the [`UnificationError`] of the first rule that fails.
    pub fn bind_template(
        &self,
        actual: &Type,
        environment: &TypeUnificationEnvironment,
    ) -> Result<TypeUnificationEnvironment> {
        match self {
            Self::Numerical(pattern) => {
                let Self::Numerical(actual_numerical) = actual else {
                    return Err(self.kind_mismatch(TypeOperation::Bind, actual));
                };
                bind_numerical(pattern, actual_numerical, actual, environment)
            }
            Self::Index(pattern) => {
                let Self::Index(actual) = actual else {
                    return Err(self.kind_mismatch(TypeOperation::Bind, actual));
                };
                let environment =
                    bind_dimension(pattern.lower_bound(), actual.lower_bound(), environment)?;
                let environment =
                    bind_dimension(pattern.upper_bound(), actual.upper_bound(), &environment)?;
                bind_dimension(pattern.stride(), actual.stride(), &environment)
            }
            Self::Extension(extension) => extension.get().bind_template(self, actual, environment),
        }
    }

    /// Return this type with the placeholders `environment` binds
    /// substituted.
    ///
    /// - A numerical type whose data type is a template over the full-shape
    ///   wildcard, bound as a full type, becomes that type. Otherwise its
    ///   data type and its dimensions are substituted, and a wildcard stays.
    /// - An index type's bounds and stride are substituted.
    /// - An extension substitutes through its hook, or is returned
    ///   unchanged.
    ///
    /// A shape variable is replaced by its binding substituted in turn, so a
    /// chain of bindings is followed to its end, stopping at an identifier
    /// already on the chain. What is unchanged is returned as the same
    /// handle.
    ///
    /// # Errors
    ///
    /// Returns [`UnificationError::Substitution`] when substituting a shape
    /// expression is refused, and an extension's error.
    pub fn substitute_template(&self, environment: &TypeUnificationEnvironment) -> Result<Type> {
        match self {
            Self::Numerical(numerical) => {
                if let DataType::Template(template) = numerical.data_type() {
                    if numerical.is_wildcard_shape() {
                        if let Some(bound) = environment.type_binding(template.identifier()) {
                            return Ok(bound.clone());
                        }
                    }
                }
                let data_type = numerical.data_type().substitute_template(environment)?;
                let shape: Vec<Dimension> = numerical
                    .shape()
                    .iter()
                    .map(|dimension| match dimension {
                        Dimension::Expression(expression) => {
                            substitute_expression(expression, environment)
                                .map(Dimension::Expression)
                        }
                        Dimension::Wildcard => Ok(Dimension::Wildcard),
                    })
                    .collect::<Result<_>>()?;
                if is_same_data_type(&data_type, numerical.data_type())
                    && shape.iter().zip(numerical.shape()).all(is_same_dimension)
                {
                    return Ok(self.clone());
                }
                Ok(Self::Numerical(NumericalType::new(data_type, shape)))
            }
            Self::Index(index) => {
                let lower = substitute_expression(index.lower_bound(), environment)?;
                let upper = substitute_expression(index.upper_bound(), environment)?;
                let stride = substitute_expression(index.stride(), environment)?;
                if Expression::ptr_eq(&lower, index.lower_bound())
                    && Expression::ptr_eq(&upper, index.upper_bound())
                    && Expression::ptr_eq(&stride, index.stride())
                {
                    return Ok(self.clone());
                }
                Ok(Self::Index(IndexType::new(lower, upper, stride)))
            }
            Self::Extension(extension) => extension.get().substitute_template(self, environment),
        }
    }

    /// Unify this type, as the expected one, with `actual`, binding
    /// placeholders on either side, and return the unified type with the
    /// environment of every binding learned.
    ///
    /// - Numerical types need equal ranks; their data types unify (two
    ///   templates must be the same template, one template binds the other
    ///   data type, and otherwise the two must be structurally equivalent),
    ///   and each pair of dimensions unifies as expressions. A wildcard is
    ///   refused.
    /// - Index types unify their bounds and strides.
    /// - An extension unifies through its hook, or by the default rule:
    ///   `actual` must be structurally equivalent, and the result is this
    ///   type.
    ///
    /// # Errors
    ///
    /// Returns the [`UnificationError`] of the first rule that fails.
    pub fn unify(
        &self,
        actual: &Type,
        environment: &TypeUnificationEnvironment,
    ) -> Result<(Type, TypeUnificationEnvironment)> {
        match self {
            Self::Numerical(expected) => {
                let Self::Numerical(actual) = actual else {
                    return Err(self.kind_mismatch(TypeOperation::Unify, actual));
                };
                unify_numerical(expected, actual, environment)
            }
            Self::Index(expected) => {
                let Self::Index(actual) = actual else {
                    return Err(self.kind_mismatch(TypeOperation::Unify, actual));
                };
                let (lower, environment) =
                    unify_expressions(expected.lower_bound(), actual.lower_bound(), environment)?;
                let (upper, environment) =
                    unify_expressions(expected.upper_bound(), actual.upper_bound(), &environment)?;
                let (stride, environment) =
                    unify_expressions(expected.stride(), actual.stride(), &environment)?;
                Ok((
                    Self::Index(IndexType::new(lower, upper, stride)),
                    environment,
                ))
            }
            Self::Extension(extension) => extension.get().unify(self, actual, environment),
        }
    }

    /// Return the kind mismatch of this type's rule meeting `actual`.
    fn kind_mismatch(&self, operation: TypeOperation, actual: &Type) -> UnificationError {
        UnificationError::KindMismatch {
            operation,
            expected: self.kind_name(),
            actual: actual.kind_name(),
        }
    }

    /// Apply the default rule of a type without one of its own: `actual`
    /// must be structurally equivalent, and nothing is bound.
    fn check_default_equivalence(
        &self,
        operation: TypeOperation,
        actual: &Type,
        environment: &TypeUnificationEnvironment,
    ) -> Result<TypeUnificationEnvironment> {
        if self.is_structurally_equivalent(actual)? {
            Ok(environment.clone())
        } else {
            Err(UnificationError::TypeMismatch {
                operation,
                expected: self.clone(),
                actual: actual.clone(),
            })
        }
    }
}

/// Bind the type `this`, as a pattern, against `actual` by the default rule
/// of an extension without a rule of its own: `actual` must be structurally
/// equivalent, and nothing is bound.
///
/// # Errors
///
/// Returns [`UnificationError::TypeMismatch`] if `actual` is not
/// equivalent, and an extension's error.
pub fn default_bind_template(
    this: &Type,
    actual: &Type,
    environment: &TypeUnificationEnvironment,
) -> Result<TypeUnificationEnvironment> {
    this.check_default_equivalence(TypeOperation::Bind, actual, environment)
}

/// Return the type `this` with its placeholders substituted by the default
/// rule of an extension without a rule of its own: `this` unchanged, the
/// same handle.
///
/// # Errors
///
/// Never fails; the signature is the hook's.
pub fn default_substitute_template(
    this: &Type,
    environment: &TypeUnificationEnvironment,
) -> Result<Type> {
    let _ = environment;
    Ok(this.clone())
}

/// Unify the type `this`, as the expected one, with `actual` by the default
/// rule of an extension without a rule of its own: `actual` must be
/// structurally equivalent, and the result is `this`, with nothing bound.
///
/// # Errors
///
/// Returns [`UnificationError::TypeMismatch`] if `actual` is not
/// equivalent, and an extension's error.
pub fn default_unify(
    this: &Type,
    actual: &Type,
    environment: &TypeUnificationEnvironment,
) -> Result<(Type, TypeUnificationEnvironment)> {
    this.check_default_equivalence(TypeOperation::Unify, actual, environment)
        .map(|environment| (this.clone(), environment))
}

/// Return whether two data types are the same handle, or equal built-in
/// values, so a substitution that changed nothing keeps its input.
fn is_same_data_type(left: &DataType, right: &DataType) -> bool {
    match (left, right) {
        (DataType::Extension(left), DataType::Extension(right)) => {
            crate::foreign::Part::ptr_eq(left, right)
        }
        _ => left == right,
    }
}

/// Return whether a substituted dimension is the same as the input one.
fn is_same_dimension((left, right): (&Dimension, &Dimension)) -> bool {
    match (left, right) {
        (Dimension::Expression(left), Dimension::Expression(right)) => {
            Expression::ptr_eq(left, right)
        }
        (Dimension::Wildcard, Dimension::Wildcard) => true,
        _ => false,
    }
}

/// Bind the numerical pattern `pattern` against `actual_numerical`, the
/// numerical type `actual`.
fn bind_numerical(
    pattern: &NumericalType,
    actual_numerical: &NumericalType,
    actual: &Type,
    environment: &TypeUnificationEnvironment,
) -> Result<TypeUnificationEnvironment> {
    let environment = pattern
        .data_type()
        .bind_template(actual_numerical.data_type(), environment)?;
    if pattern.is_wildcard_shape() {
        if let DataType::Template(template) = pattern.data_type() {
            let identifier = template.identifier();
            if let Some(bound) = environment.type_binding(identifier) {
                if !bound.is_structurally_equivalent(actual)? {
                    return Err(UnificationError::ConflictingTypeBinding {
                        identifier: identifier.clone(),
                        bound: bound.clone(),
                        actual: actual.clone(),
                    });
                }
            }
            return Ok(environment.with_type_binding(identifier.clone(), actual.clone()));
        }
        return Ok(environment);
    }
    if pattern.shape().len() != actual_numerical.shape().len() {
        return Err(UnificationError::RankMismatch {
            operation: TypeOperation::Bind,
            expected: pattern.shape().len(),
            actual: actual_numerical.shape().len(),
        });
    }
    let mut environment = environment;
    for (pattern_dimension, actual_dimension) in
        pattern.shape().iter().zip(actual_numerical.shape())
    {
        let Dimension::Expression(pattern_expression) = pattern_dimension else {
            continue;
        };
        let Dimension::Expression(actual_expression) = actual_dimension else {
            return Err(UnificationError::WildcardInActual);
        };
        environment = bind_dimension(pattern_expression, actual_expression, &environment)?;
    }
    Ok(environment)
}

/// Bind the dimension `pattern` against `actual`: a shape variable binds
/// `actual` or checks its binding, and any other dimension must equal it.
fn bind_dimension(
    pattern: &Expression,
    actual: &Expression,
    environment: &TypeUnificationEnvironment,
) -> Result<TypeUnificationEnvironment> {
    if let ExpressionKind::Identifier(identifier) = pattern.kind() {
        return match environment.expression_binding(identifier) {
            None => Ok(environment.with_expression_binding(identifier.clone(), actual.clone())),
            Some(bound) if bound != actual => Err(UnificationError::ConflictingExpressionBinding {
                identifier: identifier.clone(),
                bound: bound.clone(),
                actual: actual.clone(),
            }),
            Some(_) => Ok(environment.clone()),
        };
    }
    if pattern == actual {
        Ok(environment.clone())
    } else {
        Err(UnificationError::DimensionMismatch {
            expected: pattern.clone(),
            actual: actual.clone(),
        })
    }
}

/// Unify two numerical types.
fn unify_numerical(
    expected: &NumericalType,
    actual: &NumericalType,
    environment: &TypeUnificationEnvironment,
) -> Result<(Type, TypeUnificationEnvironment)> {
    if expected.shape().len() != actual.shape().len() {
        return Err(UnificationError::RankMismatch {
            operation: TypeOperation::Unify,
            expected: expected.shape().len(),
            actual: actual.shape().len(),
        });
    }
    let (data_type, mut environment) =
        unify_data_types(expected.data_type(), actual.data_type(), environment)?;
    let mut shape = Vec::with_capacity(expected.shape().len());
    for (expected_dimension, actual_dimension) in expected.shape().iter().zip(actual.shape()) {
        let (Dimension::Expression(left), Dimension::Expression(right)) =
            (expected_dimension, actual_dimension)
        else {
            return Err(UnificationError::WildcardInUnification);
        };
        let (unified, next) = unify_expressions(left, right, &environment)?;
        shape.push(Dimension::Expression(unified));
        environment = next;
    }
    Ok((
        Type::Numerical(NumericalType::new(data_type, shape)),
        environment,
    ))
}

/// Unify two data types.
fn unify_data_types(
    left: &DataType,
    right: &DataType,
    environment: &TypeUnificationEnvironment,
) -> Result<(DataType, TypeUnificationEnvironment)> {
    match (left, right) {
        (DataType::Template(left_template), DataType::Template(right_template)) => {
            if left_template == right_template {
                Ok((left.clone(), environment.clone()))
            } else {
                Err(UnificationError::DistinctTemplates {
                    operation: TypeOperation::Unify,
                    expected: left_template.clone(),
                    actual: right_template.clone(),
                })
            }
        }
        (DataType::Template(_), _) => Ok((right.clone(), left.bind_template(right, environment)?)),
        (_, DataType::Template(_)) => Ok((left.clone(), right.bind_template(left, environment)?)),
        _ if left.is_structurally_equivalent(right)? => Ok((left.clone(), environment.clone())),
        _ => Err(UnificationError::DataTypeMismatch {
            operation: TypeOperation::Unify,
            expected: left.clone(),
            actual: right.clone(),
        }),
    }
}

/// Return `expression` with each bound shape variable replaced by its
/// binding substituted in turn, stopping at an identifier already on the
/// chain.
///
/// # Errors
///
/// Returns [`UnificationError::Substitution`] when a substitution is
/// refused.
pub(super) fn substitute_expression(
    expression: &Expression,
    environment: &TypeUnificationEnvironment,
) -> Result<Expression> {
    substitute_avoiding(expression, environment).map_err(UnificationError::Substitution)
}

/// One pending substitution of [`substitute_avoiding`]: an expression, the
/// bound shape variables of it still to substitute, and the forms of those
/// done.
struct Frame {
    /// The shape variable whose binding this is, or `None` for the
    /// expression substituted.
    identifier: Option<Identifier>,
    expression: Expression,
    pending: Vec<Identifier>,
    replacements: HashMap<Identifier, Expression>,
    /// Whether the form depends on the chain it was reached along: it kept
    /// an identifier on the chain other than its own, or used such a form.
    depends_on_chain: bool,
}

impl Frame {
    fn new(
        identifier: Option<Identifier>,
        expression: Expression,
        environment: &TypeUnificationEnvironment,
    ) -> Self {
        let pending = expression
            .free_identifiers()
            .into_iter()
            .filter(|free| environment.expression_binding(free).is_some())
            .collect();
        Self {
            identifier,
            expression,
            pending,
            replacements: HashMap::new(),
            depends_on_chain: false,
        }
    }
}

/// Substitute `expression`: each bound shape variable free in it becomes
/// the form of its binding, substituted in turn, and a variable already on
/// the chain of bindings being substituted stays as it is.
///
/// The walk keeps the chain on the heap, marking each variable white (not
/// met), grey (on the chain) or black (its form known). A form that met no
/// grey variable but itself is the same along any chain, so it is
/// remembered and reused, and an acyclic environment substitutes in time
/// linear in its bindings; a form that met a cycle is recomputed wherever
/// it is reached.
fn substitute_avoiding(
    expression: &Expression,
    environment: &TypeUnificationEnvironment,
) -> std::result::Result<Expression, PiecewiseError> {
    let root = Frame::new(None, expression.clone(), environment);
    if root.pending.is_empty() {
        return Ok(root.expression);
    }
    let mut black: HashMap<Identifier, Expression> = HashMap::new();
    let mut grey: HashSet<Identifier> = HashSet::new();
    let mut frames = vec![root];
    loop {
        let top = frames.last_mut().unwrap_or_else(|| unreachable!("a frame"));
        if let Some(next) = top.pending.pop() {
            if let Some(form) = black.get(&next) {
                top.replacements.insert(next, form.clone());
            } else if grey.contains(&next) {
                if top.identifier.as_ref() != Some(&next) {
                    top.depends_on_chain = true;
                }
            } else {
                let bound = environment
                    .expression_binding(&next)
                    .unwrap_or_else(|| unreachable!("a pending variable is bound"))
                    .clone();
                grey.insert(next.clone());
                frames.push(Frame::new(Some(next), bound, environment));
            }
            continue;
        }
        let frame = frames.pop().unwrap_or_else(|| unreachable!("a frame"));
        let form = if frame.replacements.is_empty() {
            frame.expression
        } else {
            frame.expression.substitute(&frame.replacements)?
        };
        let Some(identifier) = frame.identifier else {
            return Ok(form);
        };
        grey.remove(&identifier);
        let parent = frames
            .last_mut()
            .unwrap_or_else(|| unreachable!("a parent frame"));
        if frame.depends_on_chain {
            parent.depends_on_chain = true;
        } else {
            black.insert(identifier.clone(), form.clone());
        }
        parent.replacements.insert(identifier, form);
    }
}

/// Return whether `identifier` is reachable from `expression` through the
/// expression bindings of `environment`: free in it, or free in the binding
/// of a shape variable reachable so far.
///
/// The walk keeps its pending identifiers on the heap and visits each once,
/// so it needs no substituted form and ends on a cyclic environment.
fn occurs_through_bindings(
    identifier: &Identifier,
    expression: &Expression,
    environment: &TypeUnificationEnvironment,
) -> bool {
    let mut pending: Vec<Identifier> = expression.free_identifiers().into_iter().collect();
    let mut visited = HashSet::new();
    while let Some(next) = pending.pop() {
        if next == *identifier {
            return true;
        }
        if !visited.insert(next.id()) {
            continue;
        }
        if let Some(bound) = environment.expression_binding(&next) {
            pending.extend(bound.free_identifiers());
        }
    }
    false
}

/// Return the end of the chain of expression bindings that starts at
/// `expression`: the first expression that is no bound identifier, or an
/// identifier met again.
fn resolve_chain<'a>(
    expression: &'a Expression,
    environment: &'a TypeUnificationEnvironment,
) -> &'a Expression {
    let mut visited = HashSet::new();
    let mut current = expression;
    while let ExpressionKind::Identifier(identifier) = current.kind() {
        if !visited.insert(identifier.id()) {
            return current;
        }
        match environment.expression_binding(identifier) {
            Some(bound) => current = bound,
            None => return current,
        }
    }
    current
}

/// Unify two expressions, binding placeholders on either side, and return
/// the unified expression with the environment of every binding learned.
///
/// Each side is first resolved through the chain of its expression
/// bindings. The same identifier on both sides unifies with itself. An
/// identifier on one side binds the other side, after an occurs check
/// against the other side with the existing bindings substituted. Two other
/// expressions must be equal.
///
/// # Errors
///
/// Returns [`UnificationError::OccursCheck`],
/// [`UnificationError::ExpressionMismatch`], or
/// [`UnificationError::Substitution`] when substituting the existing
/// bindings for the occurs check is refused.
///
/// # Examples
///
/// ```
/// use fhy_core::expression::Expression;
/// use fhy_core::identifier::Identifier;
/// use fhy_core::types::{TypeUnificationEnvironment, unify_expressions};
///
/// let n = Identifier::new("N");
/// let (unified, environment) = unify_expressions(
///     &Expression::from(n.clone()),
///     &Expression::from(4),
///     &TypeUnificationEnvironment::new(),
/// )?;
///
/// assert_eq!(unified, Expression::from(4));
/// assert_eq!(environment.expression_binding(&n), Some(&Expression::from(4)));
/// # Ok::<(), fhy_core::types::UnificationError>(())
/// ```
pub fn unify_expressions(
    left: &Expression,
    right: &Expression,
    environment: &TypeUnificationEnvironment,
) -> Result<(Expression, TypeUnificationEnvironment)> {
    let left = resolve_chain(left, environment);
    let right = resolve_chain(right, environment);
    match (left.kind(), right.kind()) {
        (
            ExpressionKind::Identifier(left_identifier),
            ExpressionKind::Identifier(right_identifier),
        ) if left_identifier == right_identifier => Ok((left.clone(), environment.clone())),
        (ExpressionKind::Identifier(identifier), _) => {
            bind_placeholder(identifier, right, environment).map(|next| (right.clone(), next))
        }
        (_, ExpressionKind::Identifier(identifier)) => {
            bind_placeholder(identifier, left, environment).map(|next| (left.clone(), next))
        }
        _ if left == right => Ok((left.clone(), environment.clone())),
        _ => Err(UnificationError::ExpressionMismatch {
            left: left.clone(),
            right: right.clone(),
        }),
    }
}

/// Bind the placeholder `identifier` to `expression`, after the occurs
/// check: `identifier` must be neither free in `expression` substituted
/// through the existing bindings nor reachable from it through them.
fn bind_placeholder(
    identifier: &Identifier,
    expression: &Expression,
    environment: &TypeUnificationEnvironment,
) -> Result<TypeUnificationEnvironment> {
    let substituted = substitute_expression(expression, environment)?;
    if substituted.free_identifiers().contains(identifier)
        || occurs_through_bindings(identifier, expression, environment)
    {
        return Err(UnificationError::OccursCheck {
            identifier: identifier.clone(),
            expression: expression.clone(),
            substituted,
        });
    }
    Ok(environment.with_expression_binding(identifier.clone(), expression.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_occurs_check_follows_bound_variables() {
        let (c, x, y, z) = (
            Identifier::new("C"),
            Identifier::new("X"),
            Identifier::new("Y"),
            Identifier::new("Z"),
        );
        let environment = TypeUnificationEnvironment::new()
            .with_expression_binding(c.clone(), Expression::from(5))
            .with_expression_binding(y.clone(), Expression::from(x.clone()) + 1)
            .with_expression_binding(z.clone(), Expression::from(c.clone()));
        let unsubstituted = Expression::piecewise(
            [(Expression::from(c.clone()), Expression::from(y.clone()))],
            0,
        )
        .expect("an identifier condition");

        assert!(occurs_through_bindings(&x, &unsubstituted, &environment));
        assert!(occurs_through_bindings(&y, &unsubstituted, &environment));
        assert!(!occurs_through_bindings(&z, &unsubstituted, &environment));
        assert!(!occurs_through_bindings(
            &x,
            &Expression::from(z),
            &environment
        ));
    }

    #[test]
    fn the_occurs_check_ends_on_a_cyclic_environment() {
        let (m, n, x) = (
            Identifier::new("M"),
            Identifier::new("N"),
            Identifier::new("X"),
        );
        let environment = TypeUnificationEnvironment::new()
            .with_expression_binding(m.clone(), Expression::from(n.clone()))
            .with_expression_binding(n.clone(), Expression::from(m.clone()));

        assert!(!occurs_through_bindings(
            &x,
            &Expression::from(m.clone()),
            &environment
        ));
        assert!(occurs_through_bindings(
            &n,
            &Expression::from(m),
            &environment
        ));
    }
}
