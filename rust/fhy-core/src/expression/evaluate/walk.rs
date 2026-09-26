//! The walk of an evaluation, generic over its backend's [`Lanes`].
//!
//! The walk is post-order on an explicit stack over the borrowed tree, and
//! a node the tree shares is evaluated once and its value reused. A value
//! carries its lanes' failures, if any, as a container of failure ids: zero
//! for a lane that did not fail, and otherwise an index into the walk's
//! table of failures, which records the node and the [`LaneFailure`]. A
//! node passes its operands' failures on, the first operand's first, and
//! adds its own for the lanes its kernel fails; a piecewise and a
//! connective drop the failures of lanes they do not need.

use std::collections::HashMap;
use std::rc::Rc;

use crate::expression::builtins::{BuiltinConstant, BuiltinFunction};
use crate::expression::callee::Callee;
use crate::expression::node::{Expression, ExpressionKind};
use crate::expression::operation::{BinaryOperation, LogicalOperation, UnaryOperation};
use crate::expression::registry::{FunctionRegistry, RegistryEntry};
use crate::expression::sort::FunctionSort;
use crate::identifier::Identifier;
use crate::tree::{BuildIdentityHasher, NodeHandle, NodeIdentity, Tree};

use super::error::{EvaluationError, LaneFailure, NearMiss};
use super::kernel;
use super::lanes::{Lane, Lanes};
use super::value::{Scalar, literal_scalar};

/// The lanes of a value, in its domain.
pub(super) enum Data<L: Lanes> {
    /// Booleans.
    Bool(L::Of<bool>),
    /// 64-bit integers.
    Int(L::Of<i64>),
    /// Reals.
    Real(L::Of<f64>),
}

impl<L: Lanes> Clone for Data<L>
where
    L::Of<bool>: Clone,
    L::Of<i64>: Clone,
    L::Of<f64>: Clone,
{
    fn clone(&self) -> Self {
        match self {
            Self::Bool(lanes) => Self::Bool(lanes.clone()),
            Self::Int(lanes) => Self::Int(lanes.clone()),
            Self::Real(lanes) => Self::Real(lanes.clone()),
        }
    }
}

/// A value and the failure ids of its lanes, if any lane failed.
struct Value<L: Lanes> {
    data: Data<L>,
    failures: Option<L::Of<u32>>,
}

/// The lanes of a result and the failure ids its own kernel added, if any.
type WithFailures<L, T> = (T, Option<<L as Lanes>::Of<u32>>);

/// A piecewise condition's lanes and failure ids.
type Condition<'v, L> = (
    &'v <L as Lanes>::Of<bool>,
    Option<&'v <L as Lanes>::Of<u32>>,
);

/// The reals of a value: its own lanes, or its integers converted.
enum Reals<'v, L: Lanes> {
    Borrowed(&'v L::Of<f64>),
    Converted(L::Of<f64>),
}

impl<L: Lanes> Reals<'_, L> {
    fn get(&self) -> &L::Of<f64> {
        match self {
            Self::Borrowed(lanes) => lanes,
            Self::Converted(lanes) => lanes,
        }
    }
}

/// One step of the walk.
enum Step<'e> {
    /// Evaluate the node, its children first.
    Enter(&'e Expression),
    /// Combine the values of the node's children, the last ones on the
    /// value stack.
    Exit(&'e Expression),
}

/// The state of one evaluation walk.
pub(super) struct Walk<'a, L: Lanes, B> {
    lanes: &'a L,
    registry: &'a FunctionRegistry,
    bindings: B,
    /// The failures lanes refer to by id: id `i` is entry `i - 1`.
    failures: Vec<(LaneFailure, Expression)>,
}

impl<'a, L, B> Walk<'a, L, B>
where
    L: Lanes,
    L::Of<bool>: Clone,
    L::Of<i64>: Clone,
    L::Of<f64>: Clone,
    L::Of<u32>: Clone,
    B: Fn(&Identifier) -> Option<Data<L>>,
{
    /// Create the walk over `lanes`, reading constants from `registry` and
    /// identifiers' values from `bindings`.
    pub(super) fn new(lanes: &'a L, registry: &'a FunctionRegistry, bindings: B) -> Self {
        Self {
            lanes,
            registry,
            bindings,
            failures: Vec::new(),
        }
    }

    /// Return the value of `root`.
    ///
    /// # Errors
    ///
    /// Returns the first refusal the walk meets, and
    /// [`EvaluationError::Lane`] for the first failed lane of the result.
    pub(super) fn run(mut self, root: &Expression) -> Result<Data<L>, EvaluationError> {
        let mut values: Vec<Rc<Value<L>>> = Vec::new();
        let mut shared: HashMap<NodeIdentity, Rc<Value<L>>, BuildIdentityHasher> =
            HashMap::default();
        let mut pending = vec![Step::Enter(root)];
        while let Some(step) = pending.pop() {
            match step {
                Step::Enter(node) => {
                    if let Some(value) = shared.get(&node.identity()) {
                        values.push(Rc::clone(value));
                        continue;
                    }
                    let value = match node.kind() {
                        ExpressionKind::Identifier(identifier) => self.identifier(identifier)?,
                        ExpressionKind::Literal(literal) => self.scalar(literal_scalar(literal)?),
                        _ => {
                            pending.push(Step::Exit(node));
                            pending.extend(node.children().rev().map(Step::Enter));
                            continue;
                        }
                    };
                    let value = Rc::new(value);
                    if node.is_shared() {
                        shared.insert(node.identity(), Rc::clone(&value));
                    }
                    values.push(value);
                }
                Step::Exit(node) => {
                    let arguments = values.split_off(values.len() - node.children().count());
                    let value = Rc::new(self.combine(node, arguments)?);
                    if node.is_shared() {
                        shared.insert(node.identity(), Rc::clone(&value));
                    }
                    values.push(value);
                }
            }
        }
        let root_value = values
            .pop()
            .unwrap_or_else(|| unreachable!("the walk yields the root's value"));
        drop(shared);
        let Value { data, failures } =
            Rc::try_unwrap(root_value).unwrap_or_else(|shared| clone_value(&shared));
        if let Some(failures) = failures {
            let aligned = self.align(&failures, &data)?;
            if let Some(id) = self.lanes.first_nonzero(&aligned) {
                let index = usize::try_from(id - 1).expect("a failure id indexes the table");
                let (failure, node) = self.failures.swap_remove(index);
                return Err(EvaluationError::Lane { failure, node });
            }
        }
        Ok(data)
    }

    /// Return `failures` broadcast to the shape of `data`.
    fn align(&self, failures: &L::Of<u32>, data: &Data<L>) -> Result<L::Of<u32>, EvaluationError> {
        let lanes = self.lanes;
        match data {
            Data::Bool(value) => lanes.map2(failures, value, |id, _| id),
            Data::Int(value) => lanes.map2(failures, value, |id, _| id),
            Data::Real(value) => lanes.map2(failures, value, |id, _| id),
        }
    }

    /// Return the value holding `scalar` in every lane.
    fn scalar(&self, scalar: Scalar) -> Value<L> {
        let lanes = self.lanes;
        let data = match scalar {
            Scalar::Bool(value) => Data::Bool(lanes.splat(value)),
            Scalar::Int(value) => Data::Int(lanes.splat(value)),
            Scalar::Real(value) => Data::Real(lanes.splat(value)),
        };
        Value {
            data,
            failures: None,
        }
    }

    /// Return the value of `identifier`: its binding, or a constant's value.
    fn identifier(&self, identifier: &Identifier) -> Result<Value<L>, EvaluationError> {
        if let Some(data) = (self.bindings)(identifier) {
            return Ok(Value {
                data,
                failures: None,
            });
        }
        if let Some(constant) = BuiltinConstant::of_identifier(identifier) {
            return Ok(self.scalar(Scalar::Real(constant.value())));
        }
        if let Some(constant) = self.registry.constant(identifier) {
            return Ok(self.scalar(literal_scalar(constant.value())?));
        }
        Err(EvaluationError::Unbound {
            identifier: identifier.clone(),
            near_miss: self.find_near_miss(identifier.name_hint()),
        })
    }

    /// Return why an unbound identifier named `hint` is probably not the
    /// variable it looks like, if it is not.
    fn find_near_miss(&self, hint: &str) -> Option<NearMiss> {
        if hint.parse::<BuiltinFunction>().is_ok() {
            return Some(NearMiss::NamesFunction);
        }
        if hint.parse::<BuiltinConstant>().is_ok() {
            return Some(NearMiss::SharesConstantName);
        }
        match self.registry.entry(hint)? {
            RegistryEntry::Constant(..) => Some(NearMiss::SharesConstantName),
            _ => Some(NearMiss::NamesFunction),
        }
    }

    /// Return the value of `node` from the values of its children.
    fn combine(
        &mut self,
        node: &Expression,
        mut arguments: Vec<Rc<Value<L>>>,
    ) -> Result<Value<L>, EvaluationError> {
        match node.kind() {
            ExpressionKind::Unary(unary) => self.unary(node, unary.operation(), &arguments[0]),
            ExpressionKind::Binary(binary) => {
                self.binary(node, binary.operation(), &arguments[0], &arguments[1])
            }
            ExpressionKind::Logical(logical) => self.logical(node, logical.operation(), &arguments),
            ExpressionKind::Piecewise(_) => self.piecewise(node, &arguments),
            ExpressionKind::Call(call) => {
                let Callee::Builtin(function) = call.callee() else {
                    return Err(EvaluationError::Unsupported(call.callee().clone()));
                };
                let argument = arguments.pop();
                match (argument, arguments.is_empty()) {
                    (Some(argument), true) if function.composed().is_none() => {
                        self.native(node, *function, argument)
                    }
                    _ => Err(EvaluationError::Unsupported(call.callee().clone())),
                }
            }
            ExpressionKind::Identifier(_) | ExpressionKind::Literal(_) => {
                unreachable!("leaves are evaluated when entered")
            }
        }
    }

    /// Record the failures of `node` and return the id of the first, the
    /// ids of the others following in [`LaneFailure::ALL`]'s order.
    fn reserve_failures(&mut self, node: &Expression) -> u32 {
        let base = u32::try_from(self.failures.len() + 1).expect("fewer than 2^32 failures");
        self.failures.extend(
            LaneFailure::ALL
                .iter()
                .map(|failure| (*failure, node.clone())),
        );
        base
    }

    /// Apply the fallible `kernel` to each lane of `a`, recording the
    /// failures of `node`.
    fn try_map1<A: Lane, C: Lane>(
        &mut self,
        node: &Expression,
        a: &L::Of<A>,
        kernel: impl Fn(A) -> Result<C, LaneFailure>,
    ) -> (L::Of<C>, Option<L::Of<u32>>) {
        let (lanes, failed) = self.lanes.try_map1(a, &kernel);
        if !failed {
            return (lanes, None);
        }
        let base = self.reserve_failures(node);
        let ids = self.lanes.map1(a, |x| {
            kernel(x).err().map_or(0, |failure| base + failure.code())
        });
        (lanes, Some(ids))
    }

    /// Apply the fallible `kernel` to each pair of lanes of `a` and `b`,
    /// recording the failures of `node`.
    fn try_map2<A: Lane, B2: Lane, C: Lane>(
        &mut self,
        node: &Expression,
        a: &L::Of<A>,
        b: &L::Of<B2>,
        kernel: impl Fn(A, B2) -> Result<C, LaneFailure>,
    ) -> Result<WithFailures<L, L::Of<C>>, EvaluationError> {
        let (lanes, failed) = self.lanes.try_map2(a, b, &kernel)?;
        if !failed {
            return Ok((lanes, None));
        }
        let base = self.reserve_failures(node);
        let ids = self.lanes.map2(a, b, |x, y| {
            kernel(x, y)
                .err()
                .map_or(0, |failure| base + failure.code())
        })?;
        Ok((lanes, Some(ids)))
    }

    /// Return the failures of `masks` combined, the first one's first.
    fn merge<'m>(
        &self,
        masks: impl IntoIterator<Item = Option<&'m L::Of<u32>>>,
    ) -> Result<Option<L::Of<u32>>, EvaluationError>
    where
        L::Of<u32>: 'm,
    {
        let mut merged: Option<L::Of<u32>> = None;
        for mask in masks.into_iter().flatten() {
            merged = Some(match merged {
                None => mask.clone(),
                Some(earlier) => {
                    self.lanes.map2(
                        &earlier,
                        mask,
                        |first, second| {
                            if first != 0 { first } else { second }
                        },
                    )?
                }
            });
        }
        Ok(merged)
    }

    /// Return the reals of `data`, or `None` for Booleans.
    fn reals<'v>(&self, data: &'v Data<L>) -> Option<Reals<'v, L>> {
        match data {
            Data::Bool(_) => None,
            Data::Int(lanes) => Some(Reals::Converted(
                self.lanes.map1(lanes, kernel::int_to_real),
            )),
            Data::Real(lanes) => Some(Reals::Borrowed(lanes)),
        }
    }

    /// Evaluate a unary node.
    fn unary(
        &mut self,
        node: &Expression,
        operation: UnaryOperation,
        operand: &Value<L>,
    ) -> Result<Value<L>, EvaluationError> {
        let lanes = self.lanes;
        let (data, own) = match (operation, &operand.data) {
            (UnaryOperation::LogicalNot, Data::Bool(value)) => {
                (Data::Bool(lanes.map1(value, |x| !x)), None)
            }
            (UnaryOperation::LogicalNot, _) => {
                return Err(EvaluationError::NumberAsBoolean(node.clone()));
            }
            (_, Data::Bool(_)) => return Err(EvaluationError::BooleanArithmetic(node.clone())),
            (UnaryOperation::Positive, Data::Int(value)) => {
                (Data::Int(lanes.map1(value, |x| x)), None)
            }
            (UnaryOperation::Positive, Data::Real(value)) => {
                (Data::Real(lanes.map1(value, |x| x)), None)
            }
            (UnaryOperation::Negate, Data::Int(value)) => {
                let (lanes, own) = self.try_map1(node, value, |x: i64| {
                    x.checked_neg().ok_or(LaneFailure::IntegerOverflow)
                });
                (Data::Int(lanes), own)
            }
            (UnaryOperation::Negate, Data::Real(value)) => {
                (Data::Real(lanes.map1(value, |x: f64| -x)), None)
            }
        };
        let failures = self.merge([operand.failures.as_ref(), own.as_ref()])?;
        Ok(Value { data, failures })
    }

    /// Evaluate a binary node.
    fn binary(
        &mut self,
        node: &Expression,
        operation: BinaryOperation,
        left: &Value<L>,
        right: &Value<L>,
    ) -> Result<Value<L>, EvaluationError> {
        let (data, own) = if is_comparison(operation) {
            (
                self.compare(node, operation, &left.data, &right.data)?,
                None,
            )
        } else {
            self.arithmetic(node, operation, &left.data, &right.data)?
        };
        let failures = self.merge([
            left.failures.as_ref(),
            right.failures.as_ref(),
            own.as_ref(),
        ])?;
        Ok(Value { data, failures })
    }

    /// Evaluate a comparison.
    fn compare(
        &self,
        node: &Expression,
        operation: BinaryOperation,
        left: &Data<L>,
        right: &Data<L>,
    ) -> Result<Data<L>, EvaluationError> {
        let lanes = self.lanes;
        let result = match (left, right) {
            (Data::Bool(a), Data::Bool(b)) => match operation {
                BinaryOperation::Equal => lanes.map2(a, b, |x, y| x == y)?,
                BinaryOperation::NotEqual => lanes.map2(a, b, |x, y| x != y)?,
                _ => return Err(EvaluationError::BooleanArithmetic(node.clone())),
            },
            (Data::Bool(_), _) | (_, Data::Bool(_)) => {
                return Err(EvaluationError::BooleanArithmetic(node.clone()));
            }
            (Data::Int(a), Data::Int(b)) => compare_lanes(lanes, operation, a, b)?,
            _ => {
                let (Some(a), Some(b)) = (self.reals(left), self.reals(right)) else {
                    unreachable!("neither side is Boolean");
                };
                compare_lanes(lanes, operation, a.get(), b.get())?
            }
        };
        Ok(Data::Bool(result))
    }

    /// Evaluate an arithmetic operation.
    fn arithmetic(
        &mut self,
        node: &Expression,
        operation: BinaryOperation,
        left: &Data<L>,
        right: &Data<L>,
    ) -> Result<WithFailures<L, Data<L>>, EvaluationError> {
        let lanes = self.lanes;
        match (left, right) {
            (Data::Bool(_), _) | (_, Data::Bool(_)) => {
                Err(EvaluationError::BooleanArithmetic(node.clone()))
            }
            (Data::Int(a), Data::Int(b)) => {
                let (result, own) = match operation {
                    BinaryOperation::Divide => {
                        let quotient = lanes
                            .map2(a, b, |x, y| kernel::int_to_real(x) / kernel::int_to_real(y))?;
                        return Ok((Data::Real(quotient), None));
                    }
                    BinaryOperation::Add => self.try_map2(node, a, b, |x: i64, y: i64| {
                        x.checked_add(y).ok_or(LaneFailure::IntegerOverflow)
                    })?,
                    BinaryOperation::Subtract => self.try_map2(node, a, b, |x: i64, y: i64| {
                        x.checked_sub(y).ok_or(LaneFailure::IntegerOverflow)
                    })?,
                    BinaryOperation::Multiply => self.try_map2(node, a, b, |x: i64, y: i64| {
                        x.checked_mul(y).ok_or(LaneFailure::IntegerOverflow)
                    })?,
                    BinaryOperation::FloorDivide => {
                        self.try_map2(node, a, b, kernel::int_floor_divide)?
                    }
                    BinaryOperation::FloorMod => {
                        self.try_map2(node, a, b, kernel::int_floor_mod)?
                    }
                    BinaryOperation::Power => self.try_map2(node, a, b, kernel::int_power)?,
                    _ => unreachable!("comparisons are not arithmetic"),
                };
                Ok((Data::Int(result), own))
            }
            _ => {
                let (Some(a), Some(b)) = (self.reals(left), self.reals(right)) else {
                    unreachable!("neither side is Boolean");
                };
                let (a, b) = (a.get(), b.get());
                let result = match operation {
                    BinaryOperation::Add => lanes.map2(a, b, |x: f64, y: f64| x + y)?,
                    BinaryOperation::Subtract => lanes.map2(a, b, |x: f64, y: f64| x - y)?,
                    BinaryOperation::Multiply => lanes.map2(a, b, |x: f64, y: f64| x * y)?,
                    BinaryOperation::Divide => lanes.map2(a, b, |x: f64, y: f64| x / y)?,
                    BinaryOperation::FloorDivide => lanes.map2(a, b, kernel::real_floor_divide)?,
                    BinaryOperation::FloorMod => lanes.map2(a, b, kernel::real_floor_mod)?,
                    BinaryOperation::Power => lanes.map2(a, b, f64::powf)?,
                    _ => unreachable!("comparisons are not arithmetic"),
                };
                Ok((Data::Real(result), None))
            }
        }
    }

    /// Evaluate a conjunction or a disjunction.
    fn logical(
        &mut self,
        node: &Expression,
        operation: LogicalOperation,
        operands: &[Rc<Value<L>>],
    ) -> Result<Value<L>, EvaluationError> {
        let lanes = self.lanes;
        let mut booleans = Vec::with_capacity(operands.len());
        for operand in operands {
            let Data::Bool(value) = &operand.data else {
                return Err(EvaluationError::NumberAsBoolean(node.clone()));
            };
            booleans.push(value);
        }
        let is_conjunction = matches!(operation, LogicalOperation::And);
        let combine = |x: bool, y: bool| if is_conjunction { x && y } else { x || y };
        let mut data = booleans[0].clone();
        for value in &booleans[1..] {
            data = lanes.map2(&data, value, combine)?;
        }
        let failures =
            if operands.iter().any(|operand| operand.failures.is_some()) {
                // A lane is decided by an operand that did not fail there and
                // holds the connective's absorbing value.
                let decisive = !is_conjunction;
                let mut decided = lanes.splat(false);
                let mut first_failure = lanes.splat(0_u32);
                for (operand, value) in operands.iter().zip(&booleans) {
                    match &operand.failures {
                        None => {
                            decided = lanes.map2(&decided, value, |d, x| d || x == decisive)?;
                        }
                        Some(ids) => {
                            decided = lanes.map3(&decided, value, ids, |d, x, id| {
                                d || (id == 0 && x == decisive)
                            })?;
                            first_failure = lanes.map2(&first_failure, ids, |first, id| {
                                if first != 0 { first } else { id }
                            })?;
                        }
                    }
                }
                Some(lanes.map2(&decided, &first_failure, |d, id| if d { 0 } else { id })?)
            } else {
                None
            };
        Ok(Value {
            data: Data::Bool(data),
            failures,
        })
    }

    /// Evaluate a piecewise from its children: each case's condition and
    /// value, then the otherwise branch.
    fn piecewise(
        &mut self,
        node: &Expression,
        children: &[Rc<Value<L>>],
    ) -> Result<Value<L>, EvaluationError> {
        let lanes = self.lanes;
        let (cases, otherwise) = children.split_at(children.len() - 1);
        let otherwise = &otherwise[0];
        let mut conditions = Vec::with_capacity(cases.len() / 2);
        let mut branches: Vec<&Value<L>> = Vec::with_capacity(cases.len() / 2 + 1);
        for pair in cases.chunks(2) {
            let Data::Bool(condition) = &pair[0].data else {
                return Err(EvaluationError::NumberAsBoolean(node.clone()));
            };
            conditions.push((condition, pair[0].failures.as_ref()));
            branches.push(&pair[1]);
        }
        branches.push(otherwise);
        let data = self.select(node, &conditions, &branches)?;
        let failures = if children.iter().any(|child| child.failures.is_some()) {
            let zero = lanes.splat(0_u32);
            let mut ids = otherwise.failures.clone().unwrap_or_else(|| zero.clone());
            for ((condition, condition_ids), branch) in conditions.iter().zip(&branches).rev() {
                let branch_ids = branch.failures.as_ref().unwrap_or(&zero);
                ids = lanes.map3(
                    condition,
                    branch_ids,
                    &ids,
                    |c, taken, other| {
                        if c { taken } else { other }
                    },
                )?;
                if let Some(condition_ids) = condition_ids {
                    ids = lanes.map2(
                        condition_ids,
                        &ids,
                        |own, other| {
                            if own != 0 { own } else { other }
                        },
                    )?;
                }
            }
            Some(ids)
        } else {
            None
        };
        Ok(Value { data, failures })
    }

    /// Return the lanes of the branch each lane's first holding condition
    /// selects, `branches` holding the cases' values then the otherwise
    /// branch, in the branches' common domain.
    fn select(
        &self,
        node: &Expression,
        conditions: &[Condition<'_, L>],
        branches: &[&Value<L>],
    ) -> Result<Data<L>, EvaluationError> {
        let lanes = self.lanes;
        let is_boolean = |value: &&Value<L>| matches!(value.data, Data::Bool(_));
        if branches.iter().all(is_boolean) {
            let values: Vec<&L::Of<bool>> = branches
                .iter()
                .map(|branch| match &branch.data {
                    Data::Bool(lanes) => lanes,
                    _ => unreachable!("every branch is Boolean"),
                })
                .collect();
            return Ok(Data::Bool(fold_selection(lanes, conditions, &values)?));
        }
        if branches.iter().any(is_boolean) {
            return Err(EvaluationError::MixedBranches(node.clone()));
        }
        if branches
            .iter()
            .all(|branch| matches!(branch.data, Data::Int(_)))
        {
            let values: Vec<&L::Of<i64>> = branches
                .iter()
                .map(|branch| match &branch.data {
                    Data::Int(lanes) => lanes,
                    _ => unreachable!("every branch is an integer"),
                })
                .collect();
            return Ok(Data::Int(fold_selection(lanes, conditions, &values)?));
        }
        let reals: Vec<Reals<'_, L>> = branches
            .iter()
            .map(|branch| {
                self.reals(&branch.data)
                    .unwrap_or_else(|| unreachable!("no branch is Boolean"))
            })
            .collect();
        let values: Vec<&L::Of<f64>> = reals.iter().map(Reals::get).collect();
        Ok(Data::Real(fold_selection(lanes, conditions, &values)?))
    }

    /// Evaluate a call of the native built-in `function`.
    fn native(
        &mut self,
        node: &Expression,
        function: BuiltinFunction,
        argument: Rc<Value<L>>,
    ) -> Result<Value<L>, EvaluationError> {
        let Value { data, failures } =
            Rc::try_unwrap(argument).unwrap_or_else(|shared| clone_value(&shared));
        let reals = match data {
            Data::Bool(_) => return Err(EvaluationError::BooleanArithmetic(node.clone())),
            Data::Int(lanes) => self.lanes.map1(&lanes, kernel::int_to_real),
            Data::Real(lanes) => lanes,
        };
        let values = self.lanes.native(function, reals)?;
        let (data, own) = match function.result_sort() {
            FunctionSort::Real => (Data::Real(values), None),
            FunctionSort::Int | FunctionSort::Nat => {
                let (integers, own) = self.try_map1(node, &values, kernel::real_to_int);
                (Data::Int(integers), own)
            }
            FunctionSort::Bool => unreachable!("no native built-in is Boolean"),
        };
        let failures = self.merge([failures.as_ref(), own.as_ref()])?;
        Ok(Value { data, failures })
    }
}

/// Return whether `operation` is a comparison.
fn is_comparison(operation: BinaryOperation) -> bool {
    matches!(
        operation,
        BinaryOperation::Equal
            | BinaryOperation::NotEqual
            | BinaryOperation::Less
            | BinaryOperation::LessEqual
            | BinaryOperation::Greater
            | BinaryOperation::GreaterEqual
    )
}

/// Compare each pair of lanes of `a` and `b` with `operation`.
fn compare_lanes<L: Lanes, T: Lane + PartialOrd>(
    lanes: &L,
    operation: BinaryOperation,
    a: &L::Of<T>,
    b: &L::Of<T>,
) -> Result<L::Of<bool>, EvaluationError> {
    match operation {
        BinaryOperation::Equal => lanes.map2(a, b, |x, y| x == y),
        BinaryOperation::NotEqual => lanes.map2(a, b, |x, y| x != y),
        BinaryOperation::Less => lanes.map2(a, b, |x, y| x < y),
        BinaryOperation::LessEqual => lanes.map2(a, b, |x, y| x <= y),
        BinaryOperation::Greater => lanes.map2(a, b, |x, y| x > y),
        BinaryOperation::GreaterEqual => lanes.map2(a, b, |x, y| x >= y),
        _ => unreachable!("the operation is a comparison"),
    }
}

/// Return the lanes of the first branch whose condition holds, the last of
/// `values` being the otherwise branch, by a right fold of selections.
fn fold_selection<L: Lanes, T: Lane>(
    lanes: &L,
    conditions: &[Condition<'_, L>],
    values: &[&L::Of<T>],
) -> Result<L::Of<T>, EvaluationError>
where
    L::Of<T>: Clone,
{
    let (otherwise, cases) = values
        .split_last()
        .unwrap_or_else(|| unreachable!("a piecewise has an otherwise branch"));
    let mut result = (*otherwise).clone();
    for ((condition, _), value) in conditions.iter().zip(cases).rev() {
        result = lanes.map3(
            condition,
            value,
            &result,
            |c, taken, other| {
                if c { taken } else { other }
            },
        )?;
    }
    Ok(result)
}

/// Return a copy of `value`.
fn clone_value<L: Lanes>(value: &Value<L>) -> Value<L>
where
    L::Of<bool>: Clone,
    L::Of<i64>: Clone,
    L::Of<f64>: Clone,
    L::Of<u32>: Clone,
{
    Value {
        data: value.data.clone(),
        failures: value.failures.clone(),
    }
}
