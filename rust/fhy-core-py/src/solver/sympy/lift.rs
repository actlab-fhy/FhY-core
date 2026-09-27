//! Lifting SymPy objects to expressions.

use std::collections::HashMap;

use num_traits::{One, Signed, Zero};
use pyo3::basic::CompareOp;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString, PyTuple};

use fhy_core::expression::builtins::{BuiltinConstant, BuiltinFunction};
use fhy_core::expression::{
    BigInt, BinaryOperation, Decimal, Expression, LiteralValue, LogicalOperation, UnaryOperation,
};
use fhy_core::identifier::Identifier;

use super::address_hash::BuildAddressHasher;
use super::boolean::Fallible;
use super::error::SympyErrorKind;
use super::load::Handles;

/// Return the Rust integer of the Python `int` `value`, of any size.
pub(super) fn read_int(value: &Bound<'_, PyAny>) -> PyResult<BigInt> {
    if let Ok(small) = value.extract::<i64>() {
        return Ok(BigInt::from(small));
    }
    let text = value.str()?;
    text.to_str()?
        .parse()
        .map_err(|_unparsable| pyo3::exceptions::PyValueError::new_err("an int has decimal digits"))
}

/// Return the text of the type of `value`, as Python's `str(type(value))`
/// writes it: `<class 'int'>`.
fn type_text(value: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(value.get_type().str()?.to_string())
}

/// How to build an expression from the lifted expressions of a SymPy
/// node's parts.
#[derive(Clone, Copy)]
enum Build {
    /// A unary node over one part.
    Unary(UnaryOperation),
    /// A binary node over two parts.
    Binary(BinaryOperation),
    /// A right fold of a binary operation over the parts, as SymPy's n-ary
    /// `Add` and `Mul` lift.
    RightFold(BinaryOperation, usize),
    /// A division of the right-folded product of the first count of parts
    /// (`1` when there are none) by the right-folded product of the second
    /// count, as a SymPy product with negative integer powers lifts.
    Quotient(usize, usize),
    /// A connective over the parts, or the one part itself.
    Connective(LogicalOperation, usize),
    /// The negation of a connective over the parts.
    NegatedConnective(LogicalOperation, usize),
    /// `Xor` of two parts: `(a || b) && !(a && b)`.
    Xor,
    /// An `ITE` of three parts.
    Ite,
    /// A call of a built-in over the parts.
    Call(BuiltinFunction, usize),
    /// A piecewise over its conditions, then its values, then its
    /// otherwise branch.
    Piecewise(usize),
}

/// A step of the lifting's work list.
enum Task<'py> {
    Visit(Bound<'py, PyAny>),
    Build(Build),
    /// Remember the last result as the lifting of this node.
    Remember(Bound<'py, PyAny>),
}

/// What visiting one SymPy node gives.
enum Step<'py> {
    /// The node's expression, lifted at once.
    Done(Expression),
    /// The node's parts, to lift first, and how to assemble them.
    Expand(Build, Vec<Bound<'py, PyAny>>),
    /// Another node whose expression the node's is.
    Forward(Bound<'py, PyAny>),
}

/// The lifting of one SymPy object.
pub(super) struct Lifter<'h> {
    handles: &'h Handles,
}

impl<'h> Lifter<'h> {
    pub(super) fn new(handles: &'h Handles) -> Self {
        Self { handles }
    }

    /// Return the expression `root` denotes, lifting on a work list.
    ///
    /// A node met again, the same Python object (`id`), takes the
    /// expression it lifted to, so a SymPy DAG costs its distinct nodes and
    /// lifts to an expression that shares as it does. The memo holds every
    /// node it keys by, so no id is reused while the lifting runs.
    pub(super) fn lift<'py>(&self, root: &Bound<'py, PyAny>) -> Fallible<Expression> {
        let mut tasks = vec![Task::Visit(root.clone())];
        let mut results: Vec<Expression> = Vec::new();
        let mut memo: HashMap<usize, (Bound<'py, PyAny>, Expression), BuildAddressHasher> =
            HashMap::default();
        while let Some(task) = tasks.pop() {
            match task {
                Task::Visit(node) => {
                    if let Some((_, lifted)) = memo.get(&node.as_ptr().addr()) {
                        results.push(lifted.clone());
                        continue;
                    }
                    match self.visit(&node)? {
                        Step::Done(expression) => {
                            results.push(expression.clone());
                            memo.insert(node.as_ptr().addr(), (node, expression));
                        }
                        Step::Forward(next) => {
                            tasks.push(Task::Remember(node));
                            tasks.push(Task::Visit(next));
                        }
                        Step::Expand(build, parts) => {
                            tasks.push(Task::Remember(node));
                            tasks.push(Task::Build(build));
                            tasks.extend(parts.into_iter().rev().map(Task::Visit));
                        }
                    }
                }
                Task::Build(build) => {
                    let lifted = assemble(build, &mut results)?;
                    results.push(lifted);
                }
                Task::Remember(node) => {
                    let lifted = results.last().expect("the node is lifted").clone();
                    memo.insert(node.as_ptr().addr(), (node, lifted));
                }
            }
        }
        Ok(results.pop().expect("the root is lifted last"))
    }

    /// Return what visiting `node` gives, in the order SymPy's kinds are
    /// tried: an `Expr` as a constant, a native function, a square root,
    /// then by kind; any other `Boolean` by kind.
    fn visit<'py>(&self, node: &Bound<'py, PyAny>) -> Fallible<Step<'py>> {
        let py = node.py();
        let handles = self.handles;
        if !node.is_instance(handles.expr.bind(py))? {
            if !node.is_instance(handles.boolean.bind(py))? {
                return Err(SympyErrorKind::UnsupportedNode(type_text(node)?));
            }
            return self.visit_boolean(node);
        }
        self.visit_expression(node)
    }

    /// Return what visiting the SymPy `Expr` `node` gives.
    fn visit_expression<'py>(&self, node: &Bound<'py, PyAny>) -> Fallible<Step<'py>> {
        let py = node.py();
        let handles = self.handles;
        let is = |class: &Py<PyAny>| node.is_instance(class.bind(py));
        let arguments =
            || -> PyResult<Vec<Bound<'py, PyAny>>> { node.getattr("args")?.try_iter()?.collect() };
        {
            if let Some(constant) = self.constant(node) {
                return Ok(Step::Done(constant));
            }
            for (class, function) in &handles.native_lifts {
                if node.is_instance(class.bind(py))? {
                    let parts = arguments()?;
                    return Ok(Step::Expand(Build::Call(*function, parts.len()), parts));
                }
            }
            if is(&handles.pow)? {
                let parts = arguments()?;
                if parts.len() > 1
                    && parts[1]
                        .rich_compare(handles.half.bind(py), CompareOp::Eq)?
                        .is_truthy()?
                {
                    return Ok(Step::Expand(
                        Build::Call(BuiltinFunction::Sqrt, 1),
                        vec![parts[0].clone()],
                    ));
                }
                if let Some(denominator) = self.reciprocal_denominator(node)? {
                    return Ok(Step::Expand(Build::Quotient(0, 1), vec![denominator]));
                }
            }
            if is(&handles.mul)? {
                let mut numerator = Vec::new();
                let mut denominator = Vec::new();
                for part in arguments()? {
                    match self.reciprocal_denominator(&part)? {
                        Some(factor) => denominator.push(factor),
                        None => numerator.push(part),
                    }
                }
                if !denominator.is_empty() {
                    let counts = (numerator.len(), denominator.len());
                    numerator.extend(denominator);
                    return Ok(Step::Expand(Build::Quotient(counts.0, counts.1), numerator));
                }
            }
            if is(&handles.piecewise)? {
                return self.visit_piecewise(node);
            }
            for (class, operation, identity) in [
                (&handles.add, BinaryOperation::Add, 0),
                (&handles.mul, BinaryOperation::Multiply, 1),
            ] {
                if is(class)? {
                    let mut parts = arguments()?;
                    return Ok(match parts.len() {
                        0 => Step::Done(Expression::literal(identity)),
                        1 => Step::Forward(parts.remove(0)),
                        count => Step::Expand(Build::RightFold(operation, count), parts),
                    });
                }
            }
            for (class, operation) in [
                (&handles.modulo, BinaryOperation::FloorMod),
                (&handles.pow, BinaryOperation::Power),
            ] {
                if is(class)? {
                    return Ok(Step::Expand(
                        Build::Binary(operation),
                        binary_parts(arguments()?)?,
                    ));
                }
            }
            if is(&handles.symbol)? {
                return Ok(Step::Done(Expression::from(read_symbol(node)?)));
            }
            if is(&handles.integer)? {
                return Ok(Step::Done(Expression::literal(read_int(
                    &node.getattr("p")?,
                )?)));
            }
            if is(&handles.float)? {
                let value: f64 = node.call_method0("__float__")?.extract()?;
                return Ok(Step::Done(Expression::literal(value)));
            }
            if is(&handles.rational)? {
                let numerator = read_int(&node.getattr("p")?)?;
                let denominator = read_int(&node.getattr("q")?)?;
                return Ok(Step::Done(lift_rational(&numerator, &denominator)));
            }
            if is(&handles.complex_infinity)? {
                return Err(SympyErrorKind::ComplexInfinity);
            }
            Err(SympyErrorKind::UnsupportedExpression(type_text(node)?))
        }
    }

    /// Return what visiting the SymPy Boolean `node`, which is no `Expr`,
    /// gives.
    fn visit_boolean<'py>(&self, node: &Bound<'py, PyAny>) -> Fallible<Step<'py>> {
        let py = node.py();
        let handles = self.handles;
        let is = |class: &Py<PyAny>| node.is_instance(class.bind(py));
        let arguments =
            || -> PyResult<Vec<Bound<'py, PyAny>>> { node.getattr("args")?.try_iter()?.collect() };
        if is(&handles.not)? {
            let first = arguments()?.into_iter().next().ok_or_else(|| {
                SympyErrorKind::Arity("a negation to have an argument".to_owned())
            })?;
            return Ok(Step::Expand(
                Build::Unary(UnaryOperation::LogicalNot),
                vec![first],
            ));
        }
        for (class, operation) in [
            (&handles.and, LogicalOperation::And),
            (&handles.or, LogicalOperation::Or),
        ] {
            if is(class)? {
                let parts = arguments()?;
                return Ok(Step::Expand(
                    Build::Connective(operation, parts.len()),
                    parts,
                ));
            }
        }
        if is(&handles.xor)? {
            let parts = arguments()?;
            if parts.is_empty() {
                return Err(SympyErrorKind::Arity(
                    "a xor to have an argument".to_owned(),
                ));
            }
            let keywords = PyDict::new(py);
            keywords.set_item("evaluate", false)?;
            let rest = handles
                .xor
                .bind(py)
                .call(PyTuple::new(py, &parts[1..])?, Some(&keywords))?;
            return Ok(Step::Expand(Build::Xor, vec![parts[0].clone(), rest]));
        }
        for (class, operation) in [
            (&handles.nor, LogicalOperation::Or),
            (&handles.nand, LogicalOperation::And),
        ] {
            if is(class)? {
                let parts = arguments()?;
                return Ok(Step::Expand(
                    Build::NegatedConnective(operation, parts.len()),
                    parts,
                ));
            }
        }
        if is(&handles.ite)? {
            let parts = arguments()?;
            if parts.len() != 3 {
                return Err(SympyErrorKind::Arity(
                    "an ITE to have exactly three arguments".to_owned(),
                ));
            }
            return Ok(Step::Expand(Build::Ite, parts));
        }
        if is(&handles.relational)? {
            let operation = if is(&handles.equality)? {
                BinaryOperation::Equal
            } else if is(&handles.unequality)? {
                BinaryOperation::NotEqual
            } else if is(&handles.strict_less_than)? {
                BinaryOperation::Less
            } else if is(&handles.less_than)? {
                BinaryOperation::LessEqual
            } else if is(&handles.strict_greater_than)? {
                BinaryOperation::Greater
            } else if is(&handles.greater_than)? {
                BinaryOperation::GreaterEqual
            } else {
                return Err(SympyErrorKind::UnsupportedRelational(type_text(node)?));
            };
            return Ok(Step::Expand(
                Build::Binary(operation),
                binary_parts(arguments()?)?,
            ));
        }
        if is(&handles.implies)? {
            return Err(SympyErrorKind::Implies(node.repr()?.to_string()));
        }
        if is(&handles.boolean_true)? {
            return Ok(Step::Done(Expression::literal(true)));
        }
        if is(&handles.boolean_false)? {
            return Ok(Step::Done(Expression::literal(false)));
        }
        Err(SympyErrorKind::UnsupportedBoolean(type_text(node)?))
    }

    /// Return the denominator `node` stands for when it is a power by a
    /// negative integer: `b` for `b ** -1`, and `b ** k`, unevaluated, for
    /// `b ** -k`. The evaluators refuse an integer raised to a negative
    /// integer power, so such a power lifts as a division.
    fn reciprocal_denominator<'py>(
        &self,
        node: &Bound<'py, PyAny>,
    ) -> Fallible<Option<Bound<'py, PyAny>>> {
        let py = node.py();
        let handles = self.handles;
        if !node.is_instance(handles.pow.bind(py))? {
            return Ok(None);
        }
        let parts: Vec<Bound<'py, PyAny>> =
            node.getattr("args")?.try_iter()?.collect::<PyResult<_>>()?;
        let [base, exponent] = parts.as_slice() else {
            return Ok(None);
        };
        if !exponent.is_instance(handles.integer.bind(py))? {
            return Ok(None);
        }
        let power = read_int(&exponent.getattr("p")?)?;
        if !power.is_negative() {
            return Ok(None);
        }
        if power == -BigInt::one() {
            return Ok(Some(base.clone()));
        }
        let keywords = PyDict::new(py);
        keywords.set_item("evaluate", false)?;
        let positive = handles
            .pow
            .bind(py)
            .call((base, exponent.neg()?), Some(&keywords))?;
        Ok(Some(positive))
    }

    /// Return the step of a piecewise: its conditions, its values, then its
    /// otherwise branch, or only its otherwise value when it has no case.
    /// A piecewise whose final condition is not `True` has no value where
    /// every condition fails, and is refused.
    fn visit_piecewise<'py>(&self, node: &Bound<'py, PyAny>) -> Fallible<Step<'py>> {
        let py = node.py();
        let branches: Vec<Bound<'py, PyAny>> =
            node.getattr("args")?.try_iter()?.collect::<PyResult<_>>()?;
        let Some(last) = branches.last() else {
            return Err(SympyErrorKind::Arity(
                "a piecewise to have a branch".to_owned(),
            ));
        };
        if !last.get_item(1)?.is(self.handles.true_value.bind(py)) {
            return Err(SympyErrorKind::PartialPiecewise(node.repr()?.to_string()));
        }
        let otherwise = last.get_item(0)?;
        let cases = &branches[..branches.len() - 1];
        if cases.is_empty() {
            return Ok(Step::Forward(otherwise));
        }
        let mut parts = Vec::with_capacity(branches.len() * 2);
        for branch in cases {
            parts.push(branch.get_item(1)?);
        }
        for branch in cases {
            parts.push(branch.get_item(0)?);
        }
        parts.push(otherwise);
        Ok(Step::Expand(Build::Piecewise(cases.len()), parts))
    }

    /// Return the expression of a SymPy constant atom: the canonical
    /// identifier of a built-in constant, and `-oo` as the negation of
    /// `inf`'s. SymPy's constants are singletons, so identity finds them.
    fn constant(&self, node: &Bound<'_, PyAny>) -> Option<Expression> {
        let py = node.py();
        let handles = self.handles;
        let reference = |constant: BuiltinConstant| Expression::from(constant.identifier().clone());
        if node.is(handles.negative_infinity.bind(py)) {
            return Some(Expression::new_unary(
                UnaryOperation::Negate,
                reference(BuiltinConstant::Inf),
            ));
        }
        [
            (&handles.pi, BuiltinConstant::Pi),
            (&handles.e, BuiltinConstant::E),
            (&handles.infinity, BuiltinConstant::Inf),
            (&handles.nan, BuiltinConstant::Nan),
        ]
        .into_iter()
        .find(|(value, _)| node.is(value.bind(py)))
        .map(|(_, constant)| reference(constant))
    }
}

/// Return the two arguments of a binary node, or refuse another count.
fn binary_parts(parts: Vec<Bound<'_, PyAny>>) -> Fallible<Vec<Bound<'_, PyAny>>> {
    if parts.len() == 2 {
        Ok(parts)
    } else {
        Err(SympyErrorKind::Arity(
            "a binary operation to have exactly two arguments".to_owned(),
        ))
    }
}

/// Return the identifier the symbol `node` names, `<name_hint>_<id>`.
fn read_symbol(node: &Bound<'_, PyAny>) -> Fallible<Identifier> {
    let name = node.getattr("name")?;
    let name = name
        .cast::<PyString>()
        .map_err(PyErr::from)?
        .to_str()?
        .to_owned();
    let unreadable = || SympyErrorKind::UnreadableSymbol(name.clone());
    let (name_hint, id) = name.rsplit_once('_').ok_or_else(unreadable)?;
    let id: u64 = id.parse().map_err(|_not_an_id| unreadable())?;
    Identifier::try_restore(id, name_hint).map_err(|_out_of_range| unreadable())
}

/// Return the expression of the rational `numerator / denominator`, in
/// lowest terms with a positive denominator other than one.
///
/// A rational some binary float equals becomes the decimal literal of its
/// value, negated when it is negative, which the evaluators read; every
/// other one becomes the exact quotient of its numerator and denominator.
pub(super) fn lift_rational(numerator: &BigInt, denominator: &BigInt) -> Expression {
    if let Some(decimal) = exact_decimal(numerator.abs(), denominator) {
        if decimal.to_f64_exact().is_some() {
            let magnitude = Expression::literal(LiteralValue::Decimal(decimal));
            return if numerator.is_negative() {
                Expression::new_unary(UnaryOperation::Negate, magnitude)
            } else {
                magnitude
            };
        }
    }
    Expression::new_binary(
        BinaryOperation::Divide,
        Expression::literal(numerator.clone()),
        Expression::literal(denominator.clone()),
    )
}

/// Return the multiplicity of `prime` in `value`, and `value` without it.
fn split_off(mut value: BigInt, prime: u32) -> (u32, BigInt) {
    let prime = BigInt::from(prime);
    let mut exponent = 0;
    while (&value % &prime).is_zero() {
        value /= &prime;
        exponent += 1;
    }
    (exponent, value)
}

/// Return the decimal equal to the non-negative `magnitude / denominator`,
/// or `None` when its expansion does not end: exactly when the
/// denominator's only prime factors are 2 and 5.
fn exact_decimal(magnitude: BigInt, denominator: &BigInt) -> Option<Decimal> {
    if !denominator.is_positive() {
        return None;
    }
    let (twos, rest) = split_off(denominator.clone(), 2);
    let (fives, rest) = split_off(rest, 5);
    if !rest.is_one() {
        return None;
    }
    let digits_after_point = twos.max(fives);
    let scaled = magnitude
        * num_traits::pow(BigInt::from(2), (digits_after_point - twos) as usize)
        * num_traits::pow(BigInt::from(5), (digits_after_point - fives) as usize);
    let mut text = scaled.to_string();
    let fraction_digits = digits_after_point as usize;
    if text.len() <= fraction_digits {
        text = format!("{}{text}", "0".repeat(fraction_digits + 1 - text.len()));
    }
    text.insert(text.len() - fraction_digits, '.');
    text.parse().ok()
}

/// Return the expression `build` assembles from the last parts of
/// `results`.
fn assemble(build: Build, results: &mut Vec<Expression>) -> Fallible<Expression> {
    let mut take = |count: usize| results.split_off(results.len() - count);
    Ok(match build {
        Build::Unary(operation) => {
            let [operand] = <[Expression; 1]>::try_from(take(1)).expect("one part");
            Expression::new_unary(operation, operand)
        }
        Build::Binary(operation) => {
            let [left, right] = <[Expression; 2]>::try_from(take(2)).expect("two parts");
            Expression::new_binary(operation, left, right)
        }
        Build::RightFold(operation, count) => fold_right(operation, take(count)),
        Build::Quotient(numerator, denominator) => {
            let mut parts = take(numerator + denominator);
            let denominator = fold_right(BinaryOperation::Multiply, parts.split_off(numerator));
            let numerator = if parts.is_empty() {
                Expression::literal(1)
            } else {
                fold_right(BinaryOperation::Multiply, parts)
            };
            Expression::new_binary(BinaryOperation::Divide, numerator, denominator)
        }
        Build::Connective(operation, count) => {
            let parts = take(count);
            Expression::new_logical(operation, parts)
        }
        Build::NegatedConnective(operation, count) => {
            let parts = take(count);
            Expression::new_unary(
                UnaryOperation::LogicalNot,
                Expression::new_logical(operation, parts),
            )
        }
        Build::Xor => {
            let [left, right] = <[Expression; 2]>::try_from(take(2)).expect("two parts");
            let either = Expression::new_logical(LogicalOperation::Or, [&left, &right]);
            let both = Expression::new_logical(LogicalOperation::And, [left, right]);
            Expression::new_logical(
                LogicalOperation::And,
                [
                    either,
                    Expression::new_unary(UnaryOperation::LogicalNot, both),
                ],
            )
        }
        Build::Ite => {
            let [condition, consequent, alternative] =
                <[Expression; 3]>::try_from(take(3)).expect("three parts");
            piecewise(vec![(condition, consequent)], alternative)?
        }
        Build::Call(function, count) => Expression::call(function, take(count)),
        Build::Piecewise(count) => {
            let mut parts = take(2 * count + 1);
            let otherwise = parts.pop().expect("an otherwise branch");
            let values = parts.split_off(count);
            piecewise(parts.into_iter().zip(values).collect(), otherwise)?
        }
    })
}

/// Return the right fold of `operation` over `parts`, which are not empty.
fn fold_right(operation: BinaryOperation, mut parts: Vec<Expression>) -> Expression {
    let mut folded = parts.pop().expect("a fold has parts");
    while let Some(left) = parts.pop() {
        folded = Expression::new_binary(operation, left, folded);
    }
    folded
}

/// Return the piecewise of `cases` and `otherwise`, or the error of one no
/// expression can be.
fn piecewise(cases: Vec<(Expression, Expression)>, otherwise: Expression) -> Fallible<Expression> {
    Expression::piecewise(cases, otherwise).map_err(|error| {
        SympyErrorKind::Python(pyo3::exceptions::PyValueError::new_err(error.to_string()))
    })
}
