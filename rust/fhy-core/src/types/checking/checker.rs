//! The bidirectional type checker.

use std::collections::HashMap;
use std::hash::BuildHasher;

use crate::expression::builtins::BuiltinConstant;
use crate::expression::registry::{FunctionRegistry, RegistryEntry};
use crate::expression::{
    BigInt, BinaryOperation, Callee, Expression, ExpressionKind, FunctionSort, LiteralValue,
    LogicalOperation, SortLookup, UnaryOperation,
};
use crate::foreign::BoxError;
use crate::identifier::Identifier;

use super::super::core_data_type::CoreDataType;
use super::super::data_type::DataType;
use super::super::error::LiteralTypeError;
use super::super::qualifier::TypeQualifier;
use super::super::ty::{IndexType, NumericalType, Type};
use super::error::{CallTargetError, TypeCheckError, TypeRule, TypeRuleKind, format_expression};

/// A type and a qualifier, the result of checking an expression.
type Typed = (Type, TypeQualifier);

type Result<T> = std::result::Result<T, TypeCheckError>;

/// The types of identifiers, as a checker asks for them.
///
/// It need not be `Sync`, unlike the expression lookups: an implementation
/// may hold state bound to one thread, as the Python binding's hold Python
/// objects. So a [`TypeChecker`] is `Send` or `Sync` only as its lookups
/// are.
pub trait IdentifierTypes {
    /// Return the type and qualifier of `identifier`, or `None` if it is not
    /// bound here.
    ///
    /// # Errors
    ///
    /// Returns the lookup's own failure, which the checker passes on.
    fn identifier_type(
        &self,
        identifier: &Identifier,
    ) -> std::result::Result<Option<Typed>, BoxError>;
}

impl<S: BuildHasher> IdentifierTypes for HashMap<Identifier, Typed, S> {
    fn identifier_type(
        &self,
        identifier: &Identifier,
    ) -> std::result::Result<Option<Typed>, BoxError> {
        Ok(self.get(identifier).cloned())
    }
}

/// What a call's callee resolves to.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CallTarget {
    /// A function with its signature.
    Function {
        /// The function's name.
        name: String,
        /// The sort of each parameter.
        parameter_sorts: Vec<FunctionSort>,
        /// The sort of the result.
        result_sort: FunctionSort,
    },
    /// A constant, which is referenced, never called.
    Constant {
        /// The constant's name.
        name: String,
    },
}

/// The signatures of the functions calls resolve to, as a checker asks for
/// them.
///
/// It need not be `Sync`, as [`IdentifierTypes`] need not.
pub trait CallTargets {
    /// Return what `callee` resolves to.
    ///
    /// # Errors
    ///
    /// Returns [`CallTargetError::Unknown`] when nothing is known under the
    /// callee's name, and [`CallTargetError::Callback`] when the lookup
    /// fails.
    fn call_target(&self, callee: &Callee) -> std::result::Result<CallTarget, CallTargetError>;
}

/// Resolves a built-in function through the catalogue, a built-in constant
/// by name, and any other name through the registry's entries.
impl CallTargets for FunctionRegistry {
    fn call_target(&self, callee: &Callee) -> std::result::Result<CallTarget, CallTargetError> {
        match callee {
            Callee::Builtin(function) => Ok(CallTarget::Function {
                name: function.name().to_owned(),
                parameter_sorts: function.parameter_sorts().to_vec(),
                result_sort: function.result_sort(),
            }),
            Callee::Named(name) => {
                if let Ok(constant) = name.as_str().parse::<BuiltinConstant>() {
                    return Ok(CallTarget::Constant {
                        name: constant.name().to_owned(),
                    });
                }
                match self.entry(name.as_str()) {
                    Some(RegistryEntry::Function(function)) => Ok(CallTarget::Function {
                        name: function.name().to_string(),
                        parameter_sorts: function.parameter_sorts().to_vec(),
                        result_sort: function.result_sort(),
                    }),
                    Some(RegistryEntry::Native(function)) => Ok(CallTarget::Function {
                        name: function.name().to_string(),
                        parameter_sorts: function.parameter_sorts().to_vec(),
                        result_sort: function.result_sort(),
                    }),
                    Some(RegistryEntry::Constant(constant, _)) => Ok(CallTarget::Constant {
                        name: constant.name().to_string(),
                    }),
                    _ => Err(CallTargetError::Unknown {
                        name: name.to_string(),
                        message: format!("no entry is registered under the name '{name}'"),
                        source: None,
                    }),
                }
            }
        }
    }
}

/// A bidirectional type checker of expressions.
///
/// [`synthesize`](Self::synthesize) infers a type from the leaves up;
/// [`check`](Self::check) hands an expected type to the literals the rules
/// pass it to, so a weak literal adopts a concrete type, and then requires
/// the synthesized type to be assignable to the expected one. Both return
/// the type and the qualifier.
///
/// - An identifier's type comes from the [`IdentifierTypes`]; one it does
///   not bind may be a native constant, whose type follows its sort. A type
///   supplied for a constant's identifier, and reading an `output`
///   identifier, are refused.
/// - A literal takes its weak type: a Boolean `bool`, a non-negative integer
///   `uint`, a negative one `int`, a float `float`. The negation of an
///   integer or float literal is that one negated literal, so `-(128)`
///   checks against `int8` and `-(5)` does not against `uint8`.
/// - Arithmetic promotes the operands' primitive types; true division of
///   integers is a real float as wide as the wider operand, and floor
///   division refuses complex operands. Comparisons are Boolean. Index
///   types shift by an integral offset and scale by a positive integer
///   literal.
/// - A weak literal operand of a binary operation whose other operand is
///   concrete is checked against it, so its range is checked.
/// - A call's arguments must satisfy its target's parameter sorts, and its
///   type follows the result sort.
///
/// The walk keeps its pending steps on the heap.
#[expect(
    missing_debug_implementations,
    reason = "the lookups are trait objects without Debug"
)]
pub struct TypeChecker<'a> {
    identifiers: &'a dyn IdentifierTypes,
    targets: &'a dyn CallTargets,
    sorts: &'a dyn SortLookup,
    defer_unknown_calls: bool,
}

impl<'a> TypeChecker<'a> {
    /// Return the checker over the identifier types `identifiers`, the call
    /// targets `targets`, and the native constants' sorts `sorts`, besides
    /// the built-in constants.
    #[must_use]
    pub fn new(
        identifiers: &'a dyn IdentifierTypes,
        targets: &'a dyn CallTargets,
        sorts: &'a dyn SortLookup,
    ) -> Self {
        Self {
            identifiers,
            targets,
            sorts,
            defer_unknown_calls: false,
        }
    }

    /// Return this checker reporting a call of an unknown function as
    /// [`TypeCheckError::UnknownCall`], unframed, instead of a broken rule.
    #[must_use]
    pub fn with_deferred_unknown_calls(self) -> Self {
        Self {
            defer_unknown_calls: true,
            ..self
        }
    }

    /// Return the type and qualifier synthesized for `expression`.
    ///
    /// # Errors
    ///
    /// Returns the [`TypeCheckError`] of the first rule `expression` breaks,
    /// or of a lookup that fails.
    pub fn synthesize(&self, expression: &Expression) -> Result<Typed> {
        Walk::new(self, expression).run(vec![Step::Infer(expression, None)])
    }

    /// Return the type and qualifier of `expression` checked against
    /// `expected`.
    ///
    /// # Errors
    ///
    /// Returns the [`TypeCheckError`] of the first rule `expression` breaks,
    /// of a synthesized type that does not fit `expected`, or of a lookup
    /// that fails.
    pub fn check(&self, expression: &Expression, expected: &Type) -> Result<Typed> {
        Walk::new(self, expression).run(vec![
            Step::CheckExpected(expression, expected.clone()),
            Step::Infer(expression, Some(expected.clone())),
        ])
    }
}

/// One step of the walk.
enum Step<'e> {
    /// Infer the type of the node, with the expected type the rules hand it.
    Infer(&'e Expression, Option<Type>),
    /// Check the type just inferred for the node against the expected type.
    CheckExpected(&'e Expression, Type),
    /// Apply the unary rule to the operand's type.
    Unary(&'e Expression),
    /// Apply the binary rules to the operands' types.
    Binary(&'e Expression),
    /// Read the type of the logical node's operand at the index as a value
    /// type.
    LogicalOperand(&'e Expression, usize),
    /// Apply the logical rule to the operands' value types.
    Logical(&'e Expression),
    /// Check the piecewise node's condition at the index.
    PiecewiseCondition(&'e Expression, usize),
    /// Check the piecewise node's value at the index; `None` is the
    /// otherwise branch.
    PiecewiseBranch(&'e Expression, Option<usize>),
    /// Promote the piecewise node's branches.
    Piecewise(&'e Expression),
    /// Check the call's arguments against the target's parameter sorts.
    Call(&'e Expression, String, Vec<FunctionSort>, FunctionSort),
}

/// One run of a checker over an expression.
struct Walk<'c, 'a, 'e> {
    checker: &'c TypeChecker<'a>,
    root: &'e Expression,
}

/// The Boolean scalar type.
fn boolean() -> Type {
    Type::Numerical(NumericalType::scalar(CoreDataType::Bool))
}

/// Return the scalar type of `core_data_type`.
fn scalar(core_data_type: CoreDataType) -> Type {
    Type::Numerical(NumericalType::scalar(core_data_type))
}

/// Return whether `value` is the Boolean scalar type.
fn is_boolean(value: &Type) -> bool {
    primitive(value) == Some(CoreDataType::Bool)
}

/// Return the core data type of a numerical type of a primitive data type.
fn primitive(value: &Type) -> Option<CoreDataType> {
    match value {
        Type::Numerical(numerical) => match numerical.data_type() {
            DataType::Primitive(core_data_type) => Some(*core_data_type),
            _ => None,
        },
        _ => None,
    }
}

/// Return whether `value` is the type of a weak literal.
fn is_weak(value: &Type) -> bool {
    primitive(value).is_some_and(CoreDataType::is_weak)
}

/// Return the name of an operation in messages.
fn binary_name(operation: BinaryOperation) -> &'static str {
    operation.as_str()
}

/// Return whether `operation` is arithmetic.
fn is_arithmetic(operation: BinaryOperation) -> bool {
    matches!(
        operation,
        BinaryOperation::Add
            | BinaryOperation::Subtract
            | BinaryOperation::Multiply
            | BinaryOperation::Divide
            | BinaryOperation::FloorDivide
            | BinaryOperation::FloorMod
            | BinaryOperation::Power
    )
}

/// Return whether `operation` is an equality.
fn is_equality(operation: BinaryOperation) -> bool {
    matches!(
        operation,
        BinaryOperation::Equal | BinaryOperation::NotEqual
    )
}

/// Return whether `operation` is an ordering.
fn is_ordering(operation: BinaryOperation) -> bool {
    matches!(
        operation,
        BinaryOperation::Less
            | BinaryOperation::LessEqual
            | BinaryOperation::Greater
            | BinaryOperation::GreaterEqual
    )
}

/// Return the literal of `expression`, if it is one.
fn literal(expression: &Expression) -> Option<&LiteralValue> {
    match expression.kind() {
        ExpressionKind::Literal(value) => Some(value),
        _ => None,
    }
}

/// Return the literal `-v` that `expression` denotes when it is the
/// negation of an integer or float literal `v`, which the checker checks as
/// that one literal against the expected type.
fn negated_literal(expression: &Expression) -> Option<LiteralValue> {
    let ExpressionKind::Unary(unary) = expression.kind() else {
        return None;
    };
    if unary.operation() != UnaryOperation::Negate {
        return None;
    }
    match literal(unary.operand())? {
        LiteralValue::Int(integer) => Some(LiteralValue::Int(-integer)),
        LiteralValue::Float(float) => Some(LiteralValue::Float(-float)),
        LiteralValue::Bool(_) | LiteralValue::Decimal(_) => None,
    }
}

/// Return the smallest real float at least `bit_width` wide, or the weak
/// `Float` for no width.
fn real_float_of_width(bit_width: Option<u32>) -> Option<CoreDataType> {
    let Some(bit_width) = bit_width else {
        return Some(CoreDataType::Float);
    };
    [
        CoreDataType::Float16,
        CoreDataType::Float32,
        CoreDataType::Float64,
    ]
    .into_iter()
    .find(|candidate| {
        candidate
            .bit_width()
            .is_some_and(|width| width >= bit_width)
    })
}

/// Return the qualifier of a value computed from values of `qualifiers`,
/// starting from `Param`.
fn promote_qualifiers(qualifiers: impl IntoIterator<Item = TypeQualifier>) -> TypeQualifier {
    qualifiers
        .into_iter()
        .fold(TypeQualifier::Param, TypeQualifier::promote)
}

impl CoreDataType {
    /// Return the weak type a literal takes on its own: a Boolean `Bool`, a
    /// non-negative integer `Uint`, a negative one `Int`, and a float
    /// `Float`.
    ///
    /// # Errors
    ///
    /// Returns [`LiteralTypeError::UnsupportedDecimal`] for a decimal, which
    /// has no core data type yet.
    pub fn of_literal(literal: &LiteralValue) -> std::result::Result<Self, LiteralTypeError> {
        match literal {
            LiteralValue::Bool(_) => Ok(Self::Bool),
            LiteralValue::Int(value) if *value >= BigInt::ZERO => Ok(Self::Uint),
            LiteralValue::Int(_) => Ok(Self::Int),
            LiteralValue::Float(_) => Ok(Self::Float),
            LiteralValue::Decimal(_) => Err(LiteralTypeError::UnsupportedDecimal),
        }
    }
}

impl<'c, 'a, 'e> Walk<'c, 'a, 'e> {
    fn new(checker: &'c TypeChecker<'a>, root: &'e Expression) -> Self {
        Self { checker, root }
    }

    /// Return the error of `rule` broken at `at`.
    fn error(
        &self,
        at: &Expression,
        kind: TypeRuleKind,
        reason: impl Into<String>,
    ) -> TypeCheckError {
        TypeCheckError::Rule {
            root: self.root.clone(),
            at: at.clone(),
            rule: TypeRule::new(kind, reason),
        }
    }

    /// Return the error of the literal error `error` at `at`.
    fn literal_error(&self, at: &Expression, error: &LiteralTypeError) -> TypeCheckError {
        let kind = if error.is_unsupported() {
            TypeRuleKind::Unsupported
        } else {
            TypeRuleKind::Literal
        };
        self.error(at, kind, error.to_string())
    }

    /// Return the join of two core data types, or the promotion error at
    /// `at`.
    fn promote(
        &self,
        at: &Expression,
        left: CoreDataType,
        right: CoreDataType,
    ) -> Result<CoreDataType> {
        left.promote(right)
            .map_err(|error| self.error(at, TypeRuleKind::Promotion, error.to_string()))
    }

    /// Return the core data type of the value type `value`, which the
    /// checker only calls with numerical types of primitive data types.
    fn primitive_of(&self, at: &Expression, value: &Type) -> Result<CoreDataType> {
        primitive(value).ok_or_else(|| {
            self.error(
                at,
                TypeRuleKind::NotAValueType,
                format!("expected a primitive data type, got {value}"),
            )
        })
    }

    /// Return `value` as the type of the value of `expression`: an index
    /// type with a stride that is not the literal `0`, or a scalar
    /// numerical type of a primitive data type. The failure is reported at
    /// `at`.
    fn as_value(&self, at: &Expression, expression: &Expression, value: Type) -> Result<Type> {
        match &value {
            Type::Index(index) => {
                if matches!(literal(index.stride()), Some(LiteralValue::Int(stride)) if *stride == BigInt::ZERO)
                {
                    return Err(self.error(
                        at,
                        TypeRuleKind::ZeroStride,
                        "index type with stride `0` is not allowed; stride must be a non-zero \
                         integer or a non-literal expression",
                    ));
                }
                Ok(value)
            }
            Type::Numerical(numerical) => {
                if !matches!(numerical.data_type(), DataType::Primitive(_)) {
                    return Err(self.error(
                        at,
                        TypeRuleKind::NotAValueType,
                        format!(
                            "sub-expression `{}` must resolve to a primitive numerical type, \
                             but got {value}",
                            format_expression(expression)
                        ),
                    ));
                }
                if !numerical.is_scalar() {
                    return Err(self.error(
                        at,
                        TypeRuleKind::Unsupported,
                        format!(
                            "sub-expression `{}` resolves to tensor type {value}, but only \
                             scalar numerical and index types are allowed in expressions",
                            format_expression(expression)
                        ),
                    ));
                }
                Ok(value)
            }
            _ => Err(self.error(
                at,
                TypeRuleKind::NotAValueType,
                format!(
                    "sub-expression `{}` must resolve to a scalar numerical type or index type, \
                     but got {value}",
                    format_expression(expression)
                ),
            )),
        }
    }

    /// Run the steps and return the one result they leave.
    #[expect(clippy::too_many_lines, reason = "one arm per step of the walk")]
    fn run(&self, mut steps: Vec<Step<'e>>) -> Result<Typed> {
        let mut results: Vec<Typed> = Vec::new();
        while let Some(step) = steps.pop() {
            match step {
                Step::Infer(node, expected) => {
                    self.infer(node, expected, &mut steps, &mut results)?;
                }
                Step::CheckExpected(node, expected) => {
                    let actual = results.pop().unwrap_or_else(|| unreachable!("a result"));
                    self.check_expected(node, node, &actual.0, &expected)?;
                    results.push(actual);
                }
                Step::Unary(node) => {
                    let operand = results.pop().unwrap_or_else(|| unreachable!("a result"));
                    results.push(self.unary(node, operand)?);
                }
                Step::Binary(node) => {
                    let right = results.pop().unwrap_or_else(|| unreachable!("a result"));
                    let left = results.pop().unwrap_or_else(|| unreachable!("a result"));
                    results.push(self.binary(node, left, right)?);
                }
                Step::LogicalOperand(node, index) => {
                    let ExpressionKind::Logical(logical) = node.kind() else {
                        unreachable!("a logical node")
                    };
                    let (value, qualifier) =
                        results.pop().unwrap_or_else(|| unreachable!("a result"));
                    let value = self.as_value(node, &logical.operands()[index], value)?;
                    results.push((value, qualifier));
                }
                Step::Logical(node) => {
                    let ExpressionKind::Logical(logical) = node.kind() else {
                        unreachable!("a logical node")
                    };
                    let operands = results.split_off(results.len() - logical.operands().len());
                    results.push(self.logical(node, logical.operation(), &operands)?);
                }
                Step::PiecewiseCondition(node, index) => {
                    let ExpressionKind::Piecewise(piecewise) = node.kind() else {
                        unreachable!("a piecewise node")
                    };
                    let (value, qualifier) =
                        results.pop().unwrap_or_else(|| unreachable!("a result"));
                    let value = self.as_value(node, &piecewise.cases()[index].0, value)?;
                    if !is_boolean(&value) {
                        return Err(self.error(
                            node,
                            TypeRuleKind::Piecewise,
                            format!(
                                "piecewise case {index} condition must be boolean, but got {value}"
                            ),
                        ));
                    }
                    results.push((value, qualifier));
                }
                Step::PiecewiseBranch(node, index) => {
                    let ExpressionKind::Piecewise(piecewise) = node.kind() else {
                        unreachable!("a piecewise node")
                    };
                    let (value, qualifier) =
                        results.pop().unwrap_or_else(|| unreachable!("a result"));
                    let branch = match index {
                        Some(index) => &piecewise.cases()[index].1,
                        None => piecewise.otherwise(),
                    };
                    let value = self.as_value(node, branch, value)?;
                    if !matches!(value, Type::Numerical(_)) {
                        let which = match index {
                            Some(index) => format!("case {index}"),
                            None => "otherwise".to_owned(),
                        };
                        return Err(self.error(
                            node,
                            TypeRuleKind::Piecewise,
                            format!(
                                "piecewise case values and otherwise must all be scalar numerical \
                                 types, but got {value} for {which}"
                            ),
                        ));
                    }
                    results.push((value, qualifier));
                }
                Step::Piecewise(node) => {
                    let ExpressionKind::Piecewise(piecewise) = node.kind() else {
                        unreachable!("a piecewise node")
                    };
                    let cases = piecewise.cases().len();
                    let parts = results.split_off(results.len() - (2 * cases + 1));
                    results.push(self.piecewise(node, cases, &parts)?);
                }
                Step::Call(node, name, parameter_sorts, result_sort) => {
                    let arguments = results.split_off(results.len() - parameter_sorts.len());
                    results.push(self.call(
                        node,
                        &name,
                        &parameter_sorts,
                        result_sort,
                        &arguments,
                    )?);
                }
            }
        }
        Ok(results
            .pop()
            .unwrap_or_else(|| unreachable!("the walk leaves one result")))
    }

    /// Infer the type of `node` with `expected`: a leaf at once, and any
    /// other node by pushing its steps.
    #[expect(clippy::too_many_lines, reason = "one arm per node kind")]
    fn infer(
        &self,
        node: &'e Expression,
        expected: Option<Type>,
        steps: &mut Vec<Step<'e>>,
        results: &mut Vec<Typed>,
    ) -> Result<()> {
        match node.kind() {
            ExpressionKind::Identifier(identifier) => {
                results.push(self.identifier(node, identifier)?);
            }
            ExpressionKind::Literal(value) => {
                results.push(self.literal(node, value, expected.as_ref())?);
            }
            ExpressionKind::Unary(unary) => {
                if let Some(negated) = negated_literal(node) {
                    results.push(self.literal(node, &negated, expected.as_ref())?);
                } else {
                    steps.push(Step::Unary(node));
                    steps.push(Step::Infer(unary.operand(), expected));
                }
            }
            ExpressionKind::Binary(binary) => {
                let expected_for_literals = expected.filter(|expected| {
                    is_arithmetic(binary.operation())
                        && matches!(expected, Type::Numerical(numerical) if numerical.is_scalar())
                });
                let for_operand = |operand: &Expression| {
                    expected_for_literals
                        .clone()
                        .filter(|_| literal(operand).is_some())
                };
                steps.push(Step::Binary(node));
                steps.push(Step::Infer(binary.right(), for_operand(binary.right())));
                steps.push(Step::Infer(binary.left(), for_operand(binary.left())));
            }
            ExpressionKind::Logical(logical) => {
                steps.push(Step::Logical(node));
                for (index, operand) in logical.operands().iter().enumerate().rev() {
                    steps.push(Step::LogicalOperand(node, index));
                    steps.push(Step::Infer(operand, None));
                }
            }
            ExpressionKind::Piecewise(piecewise) => {
                let branch_expected = expected
                    .filter(|expected| matches!(expected, Type::Numerical(numerical) if numerical.is_scalar()));
                steps.push(Step::Piecewise(node));
                steps.push(Step::PiecewiseBranch(node, None));
                steps.push(Step::Infer(piecewise.otherwise(), branch_expected.clone()));
                for (index, (_, value)) in piecewise.cases().iter().enumerate().rev() {
                    steps.push(Step::PiecewiseBranch(node, Some(index)));
                    steps.push(Step::Infer(value, branch_expected.clone()));
                }
                for (index, (condition, _)) in piecewise.cases().iter().enumerate().rev() {
                    steps.push(Step::PiecewiseCondition(node, index));
                    steps.push(Step::Infer(condition, Some(boolean())));
                }
            }
            ExpressionKind::Call(call) => {
                let target = match self.checker.targets.call_target(call.callee()) {
                    Ok(target) => target,
                    Err(CallTargetError::Callback(source)) => {
                        return Err(TypeCheckError::Callback(source));
                    }
                    Err(error) if self.checker.defer_unknown_calls => {
                        return Err(TypeCheckError::UnknownCall(error));
                    }
                    Err(error) => {
                        return Err(self.error(
                            node,
                            TypeRuleKind::UnknownCall,
                            format!(
                                "call to unknown function '{}': {error}",
                                call.callee().name()
                            ),
                        ));
                    }
                };
                let (name, parameter_sorts, result_sort) = match target {
                    CallTarget::Function {
                        name,
                        parameter_sorts,
                        result_sort,
                    } => (name, parameter_sorts, result_sort),
                    CallTarget::Constant { .. } => {
                        return Err(self.error(
                            node,
                            TypeRuleKind::Call,
                            format!(
                                "'{}' is a registered constant, not a function; reference it as \
                                 an identifier instead of calling it",
                                call.callee().name()
                            ),
                        ));
                    }
                };
                if parameter_sorts.len() != call.arguments().len() {
                    return Err(self.error(
                        node,
                        TypeRuleKind::Call,
                        format!(
                            "function '{name}' expects {} argument(s), but got {}",
                            parameter_sorts.len(),
                            call.arguments().len()
                        ),
                    ));
                }
                steps.push(Step::Call(node, name, parameter_sorts, result_sort));
                for argument in call.arguments().iter().rev() {
                    steps.push(Step::Infer(argument, None));
                }
            }
        }
        Ok(())
    }

    /// Return the sort of the native constant `identifier` is the canonical
    /// identifier of, if any.
    fn constant_sort(&self, identifier: &Identifier) -> Option<FunctionSort> {
        BuiltinConstant::of_identifier(identifier)
            .map(BuiltinConstant::sort)
            .or_else(|| self.checker.sorts.native_constant_sort(identifier))
    }

    /// Return the type of the identifier reference `node`.
    fn identifier(&self, node: &Expression, identifier: &Identifier) -> Result<Typed> {
        let found = self
            .checker
            .identifiers
            .identifier_type(identifier)
            .map_err(TypeCheckError::Callback)?;
        let Some((value, qualifier)) = found else {
            if let Some(sort) = self.constant_sort(identifier) {
                return Ok((scalar(CoreDataType::of_sort(sort)), TypeQualifier::Param));
            }
            return Err(self.error(
                node,
                TypeRuleKind::UnboundIdentifier,
                format!("identifier `{}` is not bound", format_expression(node)),
            ));
        };
        if self.constant_sort(identifier).is_some() {
            return Err(self.error(
                node,
                TypeRuleKind::SuppliedConstantType,
                format!(
                    "identifier `{}` names a native constant, whose type is fixed by its sort and \
                     cannot be supplied by the environment",
                    format_expression(node)
                ),
            ));
        }
        if qualifier == TypeQualifier::Output {
            return Err(self.error(
                node,
                TypeRuleKind::OutputRead,
                format!(
                    "identifier `{}` has type qualifier \"output\" and cannot be read from",
                    format_expression(node)
                ),
            ));
        }
        Ok((self.as_value(node, node, value)?, qualifier))
    }

    /// Return the type of the literal `node`, given the expected type.
    fn literal(
        &self,
        node: &Expression,
        value: &LiteralValue,
        expected: Option<&Type>,
    ) -> Result<Typed> {
        let own = || -> Result<Typed> {
            let core_data_type = CoreDataType::of_literal(value)
                .map_err(|error| self.literal_error(node, &error))?;
            Ok((scalar(core_data_type), TypeQualifier::Param))
        };
        let Some(expected) = expected else {
            return own();
        };
        let expected = self.as_value(node, node, expected.clone())?;
        if matches!(expected, Type::Index(_)) {
            return Err(self.error(
                node,
                TypeRuleKind::LiteralAgainstIndex,
                format!("a literal value cannot be checked against index type {expected}"),
            ));
        }
        let context = self.primitive_of(node, &expected)?;
        if context.is_weak() {
            return own();
        }
        if matches!(value, LiteralValue::Decimal(_)) {
            return Err(self.error(
                node,
                TypeRuleKind::Literal,
                format!(
                    "expected a numeric literal value, got {}",
                    format_expression(node)
                ),
            ));
        }
        let resolved = CoreDataType::resolve_literal(value, context)
            .map_err(|error| self.literal_error(node, &error))?;
        Ok((scalar(resolved), TypeQualifier::Param))
    }

    /// Check the synthesized type `actual` of `expression` against
    /// `expected`, reporting at `at`.
    fn check_expected(
        &self,
        at: &Expression,
        expression: &Expression,
        actual: &Type,
        expected: &Type,
    ) -> Result<()> {
        if let (Type::Index(_), Type::Index(_)) = (actual, expected) {
            // Index types are equivalent exactly when they are equal.
            if actual != expected {
                return Err(self.error(
                    at,
                    TypeRuleKind::ExpectedType,
                    format!(
                        "synthesized index type {actual} is not structurally equivalent to the \
                         expected index type {expected}"
                    ),
                ));
            }
            return Ok(());
        }
        let actual_value = self.as_value(at, expression, actual.clone())?;
        let expected_value = self.as_value(at, expression, expected.clone())?;
        if matches!(actual_value, Type::Index(_)) || matches!(expected_value, Type::Index(_)) {
            return Err(self.error(
                at,
                TypeRuleKind::ExpectedType,
                format!(
                    "synthesized type {actual} is incompatible with the expected type {expected}: \
                     one is an index type and the other is a numerical type"
                ),
            ));
        }
        let expected_core = self.primitive_of(at, &expected_value)?;
        let promoted = self.promote(at, self.primitive_of(at, &actual_value)?, expected_core)?;
        if promoted != expected_core {
            return Err(self.error(
                at,
                TypeRuleKind::ExpectedType,
                format!(
                    "synthesized type {actual} is wider than the expected type {expected}; \
                     promoting them yields {promoted}, which does not match the expected type"
                ),
            ));
        }
        Ok(())
    }

    /// Return the type of the literal `node` checked against `expected`, as
    /// a nested check does.
    fn check_literal(
        &self,
        node: &Expression,
        value: &LiteralValue,
        expected: &Type,
    ) -> Result<Typed> {
        let (actual, qualifier) = self.literal(node, value, Some(expected))?;
        self.check_expected(node, node, &actual, expected)?;
        Ok((actual, qualifier))
    }

    /// Apply the unary rule of `node` to its operand's type.
    fn unary(&self, node: &Expression, (operand_type, qualifier): Typed) -> Result<Typed> {
        let ExpressionKind::Unary(unary) = node.kind() else {
            unreachable!("a unary node")
        };
        let operand = self.as_value(node, node, operand_type)?;
        match unary.operation() {
            UnaryOperation::Positive => {
                if is_boolean(&operand) {
                    return Err(self.error(
                        node,
                        TypeRuleKind::Boolean,
                        "unary positive is not defined for boolean operands",
                    ));
                }
                Ok((operand, qualifier))
            }
            UnaryOperation::Negate => {
                if matches!(operand, Type::Index(_)) {
                    return Err(self.error(
                        node,
                        TypeRuleKind::Index,
                        "unary negation is not defined for index types; the resulting bounds and \
                         stride cannot be inferred safely",
                    ));
                }
                if is_boolean(&operand) {
                    return Err(self.error(
                        node,
                        TypeRuleKind::Boolean,
                        "unary negation is not defined for boolean operands",
                    ));
                }
                // A negated integer or float literal never reaches this
                // rule: `infer` checks it as the one literal it denotes.
                Ok((operand, qualifier))
            }
            UnaryOperation::LogicalNot => {
                if !is_boolean(&operand) {
                    return Err(self.error(
                        node,
                        TypeRuleKind::Boolean,
                        format!("logical NOT requires a boolean operand, but got {operand}"),
                    ));
                }
                Ok((boolean(), qualifier))
            }
        }
    }

    /// Check the weak literal operand `operand` of `node` against the other
    /// operand's concrete type, when the rescue applies.
    fn rescue(
        &self,
        node: &Expression,
        operand: &Expression,
        value: Typed,
        other: &Type,
    ) -> Result<Typed> {
        let Some(literal_value) = literal(operand) else {
            return Ok(value);
        };
        if !is_weak(&value.0)
            || !matches!(other, Type::Numerical(_))
            || is_weak(other)
            || is_boolean(other)
        {
            return Ok(value);
        }
        let (checked, qualifier) = self.check_literal(operand, literal_value, other)?;
        Ok((self.as_value(node, operand, checked)?, qualifier))
    }

    /// Apply the binary rules of `node` to its operands' types.
    fn binary(&self, node: &Expression, left: Typed, right: Typed) -> Result<Typed> {
        let ExpressionKind::Binary(binary) = node.kind() else {
            unreachable!("a binary node")
        };
        let operation = binary.operation();
        let left = (self.as_value(node, binary.left(), left.0)?, left.1);
        let right = (self.as_value(node, binary.right(), right.0)?, right.1);
        let left = self.rescue(node, binary.left(), left, &right.0)?;
        let right = self.rescue(node, binary.right(), right, &left.0)?;
        if is_equality(operation) || is_ordering(operation) {
            return self.comparison(node, operation, &left, &right);
        }
        if is_boolean(&left.0) || is_boolean(&right.0) {
            return Err(self.error(
                node,
                TypeRuleKind::Boolean,
                format!(
                    "the {} operation is not defined for boolean operands",
                    binary_name(operation)
                ),
            ));
        }
        let qualifier = left.1.promote(right.1);
        let value = self.arithmetic(
            node,
            operation,
            binary.left(),
            binary.right(),
            &left.0,
            &right.0,
        )?;
        Ok((value, qualifier))
    }

    /// Apply the comparison rules.
    fn comparison(
        &self,
        node: &Expression,
        operation: BinaryOperation,
        left: &Typed,
        right: &Typed,
    ) -> Result<Typed> {
        let qualifier = left.1.promote(right.1);
        let (left, right) = (&left.0, &right.0);
        let (left_is_boolean, right_is_boolean) = (is_boolean(left), is_boolean(right));
        let name = binary_name(operation);
        if is_ordering(operation) && (left_is_boolean || right_is_boolean) {
            return Err(self.error(
                node,
                TypeRuleKind::Boolean,
                format!("ordering operation {name} is not defined for boolean operands"),
            ));
        }
        if is_equality(operation) && (left_is_boolean || right_is_boolean) {
            if !(left_is_boolean && right_is_boolean) {
                return Err(self.error(
                    node,
                    TypeRuleKind::Boolean,
                    format!(
                        "equality operation {name} is not defined between boolean and \
                         non-boolean operands ({left}, {right})"
                    ),
                ));
            }
            return Ok((boolean(), qualifier));
        }
        match (left, right) {
            (Type::Index(_), Type::Index(_)) => {
                if is_equality(operation) {
                    // Index types are equivalent exactly when they are equal.
                    if left != right {
                        return Err(self.error(
                            node,
                            TypeRuleKind::Index,
                            format!(
                                "equality on index types requires structural equivalence, but \
                                 {left} and {right} differ"
                            ),
                        ));
                    }
                    return Ok((boolean(), qualifier));
                }
                Err(self.error(
                    node,
                    TypeRuleKind::Index,
                    format!("ordering operation {name} is not defined between two index types"),
                ))
            }
            (Type::Index(_), other) | (other, Type::Index(_)) => Err(self.error(
                node,
                TypeRuleKind::Index,
                format!("comparison {name} is not defined between index type and {other}"),
            )),
            _ => {
                self.promote(
                    node,
                    self.primitive_of(node, left)?,
                    self.primitive_of(node, right)?,
                )?;
                Ok((boolean(), qualifier))
            }
        }
    }

    /// Apply the arithmetic rules to the operands' value types.
    fn arithmetic(
        &self,
        node: &Expression,
        operation: BinaryOperation,
        left_expression: &Expression,
        right_expression: &Expression,
        left: &Type,
        right: &Type,
    ) -> Result<Type> {
        let name = binary_name(operation);
        match (left, right) {
            (Type::Numerical(_), Type::Numerical(_)) => {
                let (left_core, right_core) = (
                    self.primitive_of(node, left)?,
                    self.primitive_of(node, right)?,
                );
                let core = match operation {
                    BinaryOperation::Divide => self.division(node, left_core, right_core)?,
                    BinaryOperation::FloorDivide => {
                        self.floor_division(node, left, right, left_core, right_core)?
                    }
                    _ => self.promote(node, left_core, right_core)?,
                };
                Ok(scalar(core))
            }
            _ if matches!(
                operation,
                BinaryOperation::Divide | BinaryOperation::FloorDivide
            ) =>
            {
                Err(self.error(
                    node,
                    TypeRuleKind::Index,
                    "division is not defined for operands of index type",
                ))
            }
            (Type::Index(_), Type::Index(_)) => Err(self.error(
                node,
                TypeRuleKind::Index,
                format!(
                    "the {name} operation between two index types is not supported; arithmetic \
                     on two strided ranges does not generally produce a result whose index \
                     bounds and stride can be inferred safely. Index types are valid as operands \
                     of shift (``index +/- int``) or scale (``index * positive int literal``)."
                ),
            )),
            (Type::Index(index), Type::Numerical(_)) if operation == BinaryOperation::Add => {
                self.shift(node, index, right, right_expression, "right", false)
            }
            (Type::Numerical(_), Type::Index(index)) if operation == BinaryOperation::Add => {
                self.shift(node, index, left, left_expression, "left", false)
            }
            (Type::Index(index), Type::Numerical(_)) if operation == BinaryOperation::Subtract => {
                self.shift(node, index, right, right_expression, "right", true)
            }
            (Type::Index(index), Type::Numerical(_)) if operation == BinaryOperation::Multiply => {
                self.scale(node, index, right, right_expression, "right")
            }
            (Type::Numerical(_), Type::Index(index)) if operation == BinaryOperation::Multiply => {
                self.scale(node, index, left, left_expression, "left")
            }
            _ => Err(self.error(
                node,
                TypeRuleKind::Index,
                format!(
                    "the {name} operation is not defined for operands of types {left} and \
                     {right}; the resulting index bounds and stride cannot be inferred safely"
                ),
            )),
        }
    }

    /// Return a real float of the width of an integral `core`, the weak
    /// `Float` for a weak integer.
    fn lift(&self, node: &Expression, core: CoreDataType) -> Result<CoreDataType> {
        let width = core.bit_width();
        real_float_of_width(width).ok_or_else(|| {
            self.error(
                node,
                TypeRuleKind::Promotion,
                format!(
                    "no real float core data type found for bit width {}",
                    width.unwrap_or_default()
                ),
            )
        })
    }

    /// Return the pair with an integral side lifted to a real float of its
    /// width when the other side is a float or complex type.
    fn lift_pair(
        &self,
        node: &Expression,
        left: CoreDataType,
        right: CoreDataType,
    ) -> Result<(CoreDataType, CoreDataType)> {
        if left.is_integral() && right.is_float_like() {
            return Ok((self.lift(node, left)?, right));
        }
        if left.is_float_like() && right.is_integral() {
            return Ok((left, self.lift(node, right)?));
        }
        Ok((left, right))
    }

    /// Return the type of a true division.
    fn division(
        &self,
        node: &Expression,
        left: CoreDataType,
        right: CoreDataType,
    ) -> Result<CoreDataType> {
        if left.is_integral() && right.is_integral() {
            let width = [left.bit_width(), right.bit_width()]
                .into_iter()
                .flatten()
                .max();
            return real_float_of_width(width).ok_or_else(|| {
                self.error(
                    node,
                    TypeRuleKind::Promotion,
                    format!(
                        "no real float core data type found for bit width {}",
                        width.unwrap_or_default()
                    ),
                )
            });
        }
        let (left, right) = self.lift_pair(node, left, right)?;
        self.promote(node, left, right)
    }

    /// Return the type of a floor division.
    fn floor_division(
        &self,
        node: &Expression,
        left_type: &Type,
        right_type: &Type,
        left: CoreDataType,
        right: CoreDataType,
    ) -> Result<CoreDataType> {
        if left.is_integral() && right.is_integral() {
            return self.promote(node, left, right);
        }
        if left.is_complex() || right.is_complex() {
            return Err(self.error(
                node,
                TypeRuleKind::Promotion,
                format!(
                    "floor division is not defined for complex numerical types (operand types \
                     were {left_type} and {right_type})"
                ),
            ));
        }
        let (left, right) = self.lift_pair(node, left, right)?;
        self.promote(node, left, right)
    }

    /// Return `index` shifted by the offset `offset_expression`, whose type
    /// is `offset`, on the `side`.
    fn shift(
        &self,
        node: &Expression,
        index: &IndexType,
        offset: &Type,
        offset_expression: &Expression,
        side: &str,
        subtract: bool,
    ) -> Result<Type> {
        if !self.primitive_of(node, offset)?.is_integral() {
            return Err(self.error(
                node,
                TypeRuleKind::Index,
                format!(
                    "index shift requires an integral scalar offset, but the {side} operand has \
                     type {offset}"
                ),
            ));
        }
        let (lower, upper) = if subtract {
            (
                index.lower_bound() - offset_expression,
                index.upper_bound() - offset_expression,
            )
        } else {
            (
                index.lower_bound() + offset_expression,
                index.upper_bound() + offset_expression,
            )
        };
        Ok(Type::Index(IndexType::new(
            lower,
            upper,
            index.stride().clone(),
        )))
    }

    /// Return `index` scaled by the literal `scalar_expression`, whose type
    /// is `scalar_type`, on the `side`.
    fn scale(
        &self,
        node: &Expression,
        index: &IndexType,
        scalar_type: &Type,
        scalar_expression: &Expression,
        side: &str,
    ) -> Result<Type> {
        let Some(value) = literal(scalar_expression) else {
            return Err(self.error(
                node,
                TypeRuleKind::Index,
                format!(
                    "index scaling requires a positive integer literal scalar, but the {side} \
                     operand `{}` is not a literal expression",
                    format_expression(scalar_expression)
                ),
            ));
        };
        if !self.primitive_of(node, scalar_type)?.is_integral() {
            return Err(self.error(
                node,
                TypeRuleKind::Index,
                format!(
                    "index scaling requires a positive integer literal scalar, but the {side} \
                     operand has non-integral type {scalar_type}"
                ),
            ));
        }
        let LiteralValue::Int(factor) = value else {
            return Err(self.error(
                node,
                TypeRuleKind::Index,
                format!(
                    "index scaling requires an integer literal scalar, but got scalar value {}",
                    format_expression(scalar_expression)
                ),
            ));
        };
        if *factor <= BigInt::ZERO {
            return Err(self.error(
                node,
                TypeRuleKind::Index,
                format!("index scaling requires a positive integer literal scalar, but got scalar value {factor}"),
            ));
        }
        let stride = match literal(index.stride()) {
            Some(LiteralValue::Int(stride)) => Expression::from(factor * stride),
            _ => scalar_expression * index.stride(),
        };
        Ok(Type::Index(IndexType::new(
            scalar_expression * index.lower_bound(),
            scalar_expression * index.upper_bound(),
            stride,
        )))
    }

    /// Apply the logical rule to the operands' value types.
    fn logical(
        &self,
        node: &Expression,
        operation: LogicalOperation,
        operands: &[Typed],
    ) -> Result<Typed> {
        if !operands.iter().all(|(value, _)| is_boolean(value)) {
            let (last, rest) = operands
                .split_last()
                .unwrap_or_else(|| unreachable!("two operands"));
            let rendered: Vec<String> = rest.iter().map(|(value, _)| value.to_string()).collect();
            return Err(self.error(
                node,
                TypeRuleKind::Boolean,
                format!(
                    "logical {} requires boolean operands, but got {} and {}",
                    operation.as_str(),
                    rendered.join(", "),
                    last.0
                ),
            ));
        }
        let qualifier = operands
            .iter()
            .map(|(_, qualifier)| *qualifier)
            .reduce(TypeQualifier::promote)
            .unwrap_or(TypeQualifier::Param);
        Ok((boolean(), qualifier))
    }

    /// Promote the branches of the piecewise `node` and its qualifiers:
    /// `parts` holds the conditions, the values and the otherwise branch.
    fn piecewise(&self, node: &Expression, cases: usize, parts: &[Typed]) -> Result<Typed> {
        let (conditions, branches) = parts.split_at(cases);
        let mut core = self.primitive_of(node, &branches[0].0)?;
        for (value, _) in &branches[1..] {
            core = self.promote(node, core, self.primitive_of(node, value)?)?;
        }
        let qualifier = conditions
            .iter()
            .chain(branches)
            .map(|(_, qualifier)| *qualifier)
            .reduce(TypeQualifier::promote)
            .unwrap_or(TypeQualifier::Param);
        Ok((scalar(core), qualifier))
    }

    /// Check the call's arguments against the parameter sorts, and return
    /// the type of its result.
    fn call(
        &self,
        node: &Expression,
        name: &str,
        parameter_sorts: &[FunctionSort],
        result_sort: FunctionSort,
        arguments: &[Typed],
    ) -> Result<Typed> {
        for (index, (sort, (argument, _))) in parameter_sorts.iter().zip(arguments).enumerate() {
            let compatible =
                primitive(argument).is_some_and(|core| core.is_compatible_with_sort(*sort));
            if !compatible {
                return Err(self.error(
                    node,
                    TypeRuleKind::Call,
                    format!("argument {index} of function '{name}' expects sort {sort}, but got {argument}"),
                ));
            }
        }
        Ok((
            scalar(CoreDataType::of_sort(result_sort)),
            promote_qualifiers(arguments.iter().map(|(_, qualifier)| *qualifier)),
        ))
    }
}
