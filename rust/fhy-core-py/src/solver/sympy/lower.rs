//! Lowering an expression to SymPy objects.

use std::collections::HashMap;
use std::sync::Arc;

use pyo3::basic::CompareOp;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyTuple};

use fhy_core::expression::builtins::{BuiltinConstant, BuiltinFunction};
use fhy_core::expression::registry::RegistryEntry;
use fhy_core::expression::{
    BigInt, BinaryOperation, Callee, Decimal, Expression, ExpressionKind, LiteralValue,
    LogicalOperation, UnaryOperation,
};
use fhy_core::identifier::Identifier;
use fhy_core::solver::SimplifyContext;
use fhy_core::tree::{NodeHandle, NodeIdentity};

use super::boolean::{Fallible, comparison_operands, condition, to_boolean};
use super::error::SympyErrorKind;
use super::load::Handles;

/// Return the Python `int` of `value`.
pub(super) fn python_int<'py>(py: Python<'py>, value: &BigInt) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(small) = i64::try_from(value) {
        return Ok(small.into_pyobject(py)?.into_any());
    }
    py.get_type::<PyInt>().call1((value.to_string(),))
}

/// Return the name of the SymPy symbol of `identifier`: `<name_hint>_<id>`.
pub(super) fn symbol_name(identifier: &Identifier) -> String {
    format!("{}_{}", identifier.name_hint(), identifier.id())
}

/// Return the SymPy number of `value`, exactly: a Boolean as `true` or
/// `false`, an integer as an `Integer`, a binary float as the `Float` of
/// its value, and a decimal as the `Rational` it denotes.
pub(super) fn literal<'py>(
    py: Python<'py>,
    handles: &Handles,
    value: &LiteralValue,
) -> PyResult<Bound<'py, PyAny>> {
    match value {
        LiteralValue::Bool(true) => Ok(handles.true_value.bind(py).clone()),
        LiteralValue::Bool(false) => Ok(handles.false_value.bind(py).clone()),
        LiteralValue::Int(value) => handles.integer.bind(py).call1((python_int(py, value)?,)),
        LiteralValue::Float(value) => handles.float.bind(py).call1((PyFloat::new(py, *value),)),
        LiteralValue::Decimal(value) => rational(py, handles, value),
    }
}

/// Return the `Rational` a decimal denotes.
fn rational<'py>(
    py: Python<'py>,
    handles: &Handles,
    value: &Decimal,
) -> PyResult<Bound<'py, PyAny>> {
    let ten = BigInt::from(10);
    let exponent = value.exponent();
    let magnitude = exponent.unsigned_abs();
    let power = num_traits::pow(ten, usize::try_from(magnitude).unwrap_or(usize::MAX));
    let (numerator, denominator) = if exponent >= 0 {
        (value.coefficient() * power, BigInt::from(1))
    } else {
        (value.coefficient().clone(), power)
    };
    handles
        .rational
        .bind(py)
        .call1((python_int(py, &numerator)?, python_int(py, &denominator)?))
}

/// A lowering in progress, remembering the SymPy object of each node it
/// lowered, so a node an expression shares is lowered once.
pub(super) struct Lowerer<'h, 'c> {
    handles: &'h Arc<Handles>,
    context: &'c SimplifyContext<'c>,
}

impl<'h, 'c> Lowerer<'h, 'c> {
    pub(super) fn new(handles: &'h Arc<Handles>, context: &'c SimplifyContext<'c>) -> Self {
        Self { handles, context }
    }

    /// Return the SymPy object of `root`, lowering on a work list.
    pub(super) fn lower<'py>(
        &self,
        py: Python<'py>,
        root: &Expression,
    ) -> Fallible<Bound<'py, PyAny>> {
        let mut memo: HashMap<NodeIdentity, Bound<'py, PyAny>> = HashMap::new();
        let mut pending: Vec<(&Expression, bool)> = vec![(root, false)];
        while let Some((node, is_expanded)) = pending.pop() {
            if memo.contains_key(&node.identity()) {
                continue;
            }
            if !is_expanded {
                self.refuse_call(node)?;
                let children = operands_of(node);
                if !children.is_empty() {
                    pending.push((node, true));
                    pending.extend(
                        children
                            .into_iter()
                            .rev()
                            .filter(|child| !memo.contains_key(&child.identity()))
                            .map(|child| (child, false)),
                    );
                    continue;
                }
            }
            let operands: Vec<Bound<'py, PyAny>> = operands_of(node)
                .into_iter()
                .map(|child| memo[&child.identity()].clone())
                .collect();
            let object = self.build(py, node, operands)?;
            memo.insert(node.identity(), object);
        }
        Ok(memo
            .remove(&root.identity())
            .expect("the root is lowered last"))
    }

    /// Refuse a call SymPy has no function for, before its arguments are
    /// lowered.
    fn refuse_call(&self, node: &Expression) -> Fallible<()> {
        let ExpressionKind::Call(call) = node.kind() else {
            return Ok(());
        };
        match call.callee() {
            Callee::Builtin(function) if function.composed().is_some() => {
                Err(SympyErrorKind::CallNeedsInlining(call.callee().clone()))
            }
            Callee::Builtin(_) => Ok(()),
            Callee::Named(name) => Err(
                match self.context.registry().and_then(|r| r.entry(name.as_str())) {
                    Some(RegistryEntry::Function(_)) => {
                        SympyErrorKind::CallNeedsInlining(call.callee().clone())
                    }
                    Some(RegistryEntry::Native(_)) => SympyErrorKind::NoSympyLowering(name.clone()),
                    Some(RegistryEntry::Constant(..)) => {
                        SympyErrorKind::ConstantCalled(name.clone())
                    }
                    _ => SympyErrorKind::UnknownFunction(name.clone()),
                },
            ),
        }
    }

    /// Return the SymPy object of `node` over its lowered `operands`.
    fn build<'py>(
        &self,
        py: Python<'py>,
        node: &Expression,
        operands: Vec<Bound<'py, PyAny>>,
    ) -> Fallible<Bound<'py, PyAny>> {
        let handles = self.handles;
        Ok(match node.kind() {
            ExpressionKind::Literal(value) => literal(py, handles, value)?,
            ExpressionKind::Identifier(identifier) => self.identifier(py, identifier)?,
            ExpressionKind::Unary(unary) => {
                let operand = &operands[0];
                match unary.operation() {
                    UnaryOperation::Negate => operand.neg()?,
                    UnaryOperation::Positive => operand.pos()?,
                    UnaryOperation::LogicalNot => handles
                        .not
                        .bind(py)
                        .call1((to_boolean(handles, operand)?,))?,
                }
            }
            ExpressionKind::Binary(binary) => {
                let (left, right) = (&operands[0], &operands[1]);
                match binary.operation() {
                    BinaryOperation::Add => left.add(right)?,
                    BinaryOperation::Subtract => left.sub(right)?,
                    BinaryOperation::Multiply => left.mul(right)?,
                    BinaryOperation::Divide => left.div(right)?,
                    BinaryOperation::FloorDivide => {
                        handles.floor.bind(py).call1((left.div(right)?,))?
                    }
                    BinaryOperation::FloorMod => left.rem(right)?,
                    BinaryOperation::Power => left.pow(right, py.None())?,
                    BinaryOperation::Equal => {
                        let (left, right) = comparison_operands(handles, left, right)?;
                        handles.equality.bind(py).call1((left, right))?
                    }
                    BinaryOperation::NotEqual => {
                        let (left, right) = comparison_operands(handles, left, right)?;
                        handles.unequality.bind(py).call1((left, right))?
                    }
                    BinaryOperation::Less => left.rich_compare(right, CompareOp::Lt)?,
                    BinaryOperation::LessEqual => left.rich_compare(right, CompareOp::Le)?,
                    BinaryOperation::Greater => left.rich_compare(right, CompareOp::Gt)?,
                    BinaryOperation::GreaterEqual => left.rich_compare(right, CompareOp::Ge)?,
                }
            }
            ExpressionKind::Logical(logical) => {
                let operands = operands
                    .iter()
                    .map(|operand| to_boolean(handles, operand))
                    .collect::<Fallible<Vec<_>>>()?;
                let connective = match logical.operation() {
                    LogicalOperation::And => handles.and.bind(py),
                    LogicalOperation::Or => handles.or.bind(py),
                };
                connective.call1(PyTuple::new(py, operands)?)?
            }
            ExpressionKind::Piecewise(piecewise) => {
                let mut branches = Vec::with_capacity(piecewise.cases().len() + 1);
                let mut iterator = operands.into_iter();
                for _ in piecewise.cases() {
                    let value = iterator.next().expect("a lowered case value");
                    let case_condition = iterator.next().expect("a lowered case condition");
                    let case_condition = condition(handles, &case_condition)?;
                    branches.push(PyTuple::new(py, [value, case_condition])?.into_any());
                }
                let otherwise = iterator.next().expect("a lowered otherwise branch");
                branches.push(
                    PyTuple::new(py, [otherwise, PyBool::new(py, true).to_owned().into_any()])?
                        .into_any(),
                );
                let keywords = PyDict::new(py);
                keywords.set_item("evaluate", false)?;
                handles
                    .parity_opaque_piecewise
                    .bind(py)
                    .call(PyTuple::new(py, branches)?, Some(&keywords))?
            }
            ExpressionKind::Call(call) => {
                let Callee::Builtin(function) = call.callee() else {
                    unreachable!("a named call is refused before it is lowered")
                };
                self.native(py, *function, operands)?
            }
        })
    }

    /// Return the SymPy object of a reference to `identifier`: a native
    /// constant's value, or a symbol.
    fn identifier<'py>(
        &self,
        py: Python<'py>,
        identifier: &Identifier,
    ) -> Fallible<Bound<'py, PyAny>> {
        let handles = self.handles;
        if let Some(constant) = BuiltinConstant::of_identifier(identifier) {
            let value = match constant {
                BuiltinConstant::Pi => &handles.pi,
                BuiltinConstant::E => &handles.e,
                BuiltinConstant::Inf => &handles.infinity,
                BuiltinConstant::Nan => &handles.nan,
                _ => return Err(SympyErrorKind::UnsupportedConstant(identifier.clone())),
            };
            return Ok(value.bind(py).clone());
        }
        if let Some(constant) = self
            .context
            .registry()
            .and_then(|registry| registry.constant(identifier))
        {
            return Ok(literal(py, handles, constant.value())?);
        }
        if self
            .context
            .sorts()
            .native_constant_sort(identifier)
            .is_some()
        {
            return Err(SympyErrorKind::ConstantValueUnknown(identifier.clone()));
        }
        Ok(handles.symbol.bind(py).call1((symbol_name(identifier),))?)
    }

    /// Return the SymPy application of the native built-in `function`.
    fn native<'py>(
        &self,
        py: Python<'py>,
        function: BuiltinFunction,
        arguments: Vec<Bound<'py, PyAny>>,
    ) -> Fallible<Bound<'py, PyAny>> {
        let handles = self.handles;
        let arguments = PyTuple::new(py, arguments)?;
        Ok(match function {
            BuiltinFunction::Exp2 => {
                let mut with_base = vec![PyInt::new(py, 2_i64).into_any()];
                with_base.extend(arguments.iter());
                handles.pow.bind(py).call1(PyTuple::new(py, with_base)?)?
            }
            BuiltinFunction::Log2 | BuiltinFunction::Log10 => {
                let base: i64 = if function == BuiltinFunction::Log2 {
                    2
                } else {
                    10
                };
                let mut with_base: Vec<Bound<'py, PyAny>> = arguments.iter().collect();
                with_base.push(PyInt::new(py, base).into_any());
                handles.log.bind(py).call1(PyTuple::new(py, with_base)?)?
            }
            other => {
                let lowering = handles
                    .natives
                    .iter()
                    .find(|(native, _)| *native == other)
                    .map(|(_, lowering)| lowering.bind(py))
                    .expect("every native built-in has a sympy function");
                lowering.call1(arguments)?
            }
        })
    }
}

/// Return the operands of `node` in the order the lowering visits them: a
/// piecewise's case values and conditions in turn, then its otherwise
/// branch.
fn operands_of(node: &Expression) -> Vec<&Expression> {
    match node.kind() {
        ExpressionKind::Unary(unary) => vec![unary.operand()],
        ExpressionKind::Binary(binary) => vec![binary.left(), binary.right()],
        ExpressionKind::Logical(logical) => logical.operands().iter().collect(),
        ExpressionKind::Piecewise(piecewise) => piecewise
            .cases()
            .iter()
            .flat_map(|(condition, value)| [value, condition])
            .chain([piecewise.otherwise()])
            .collect(),
        ExpressionKind::Call(call) => call.arguments().iter().collect(),
        ExpressionKind::Identifier(_) | ExpressionKind::Literal(_) => Vec::new(),
    }
}
