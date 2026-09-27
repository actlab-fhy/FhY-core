//! The lowering of expressions to the typed terms of a script.

use std::collections::HashMap;

use num_bigint::{BigInt, Sign};
use num_traits::{One, Signed, Zero};

use crate::expression::{
    BinaryOperation, Decimal, Expression, ExpressionKind, LiteralValue, LogicalOperation,
    SymbolType, SymbolTypes, UnaryOperation,
};
use crate::identifier::Identifier;
use crate::tree::{BuildIdentityHasher, NodeHandle, NodeIdentity};

use super::super::error::LoweringError;
use super::term::{Assertion, Operator, Symbol, Term, TermId, TermNode};
use super::{Declaration, Logic, SmtScript};

/// The name of the constant a named expression's value is declared as. No
/// identifier's symbol can equal it, since those end in `_` and digits.
const VALUE_SYMBOL: &str = "value";

/// Return `name_hint` with every character a quoted symbol cannot hold
/// replaced by `_`: `|`, `\`, and every control character
/// ([`char::is_control`]), since SMT-LIB2 allows only printable text there
/// and a NUL ends the C string some solvers read the script as.
///
/// The `_<id>` suffix of the symbol keeps two sanitized hints apart.
fn sanitize(name_hint: &str) -> String {
    name_hint
        .chars()
        .map(|character| {
            if matches!(character, '|' | '\\') || character.is_control() {
                '_'
            } else {
                character
            }
        })
        .collect()
}

/// Return the greatest common divisor of two non-negative integers.
fn gcd(mut left: BigInt, mut right: BigInt) -> BigInt {
    while !right.is_zero() {
        let remainder = &left % &right;
        left = right;
        right = remainder;
    }
    left
}

/// Return the exact rational a finite `f64` denotes, as a numerator of any
/// sign and a positive denominator in lowest terms.
fn rationalize_float(value: f64) -> (BigInt, BigInt) {
    if value == 0.0 {
        return (BigInt::zero(), BigInt::one());
    }
    let bits = value.to_bits();
    let is_negative = bits >> 63 == 1;
    let biased_exponent = i64::try_from((bits >> 52) & 0x7ff).expect("an 11-bit field");
    let fraction = bits & ((1_u64 << 52) - 1);
    let (mut mantissa, mut exponent) = if biased_exponent == 0 {
        (fraction, -1074_i64)
    } else {
        (fraction | (1_u64 << 52), biased_exponent - 1075)
    };
    let trailing_zeros = mantissa.trailing_zeros();
    mantissa >>= trailing_zeros;
    exponent += i64::from(trailing_zeros);
    let magnitude = BigInt::from(mantissa);
    let (numerator, denominator) = if exponent >= 0 {
        let shift = u64::try_from(exponent).expect("a non-negative exponent");
        (magnitude << shift, BigInt::one())
    } else {
        let shift = u64::try_from(-exponent).expect("a positive shift");
        (magnitude, BigInt::one() << shift)
    };
    if is_negative {
        (-numerator, denominator)
    } else {
        (numerator, denominator)
    }
}

/// Return the exact rational a decimal denotes, in lowest terms.
fn rationalize_decimal(decimal: &Decimal) -> (BigInt, BigInt) {
    let ten = BigInt::from(10);
    let exponent = decimal.exponent();
    if exponent >= 0 {
        let power = u32::try_from(exponent).map_or_else(
            |_| unreachable!("a decimal's exponent is bounded by its digit count"),
            |exponent| ten.pow(exponent),
        );
        return (decimal.coefficient() * power, BigInt::one());
    }
    let power = u32::try_from(-exponent).map_or_else(
        |_| unreachable!("a decimal's exponent is bounded by its digit count"),
        |exponent| ten.pow(exponent),
    );
    let divisor = gcd(decimal.coefficient().clone(), power.clone());
    (decimal.coefficient() / &divisor, power / divisor)
}

/// Lowers expressions into the terms of one script.
///
/// It remembers the term of each node it lowered by the node's identity,
/// so a subtree shared by several places, or by several expressions of one
/// question, is one term. It keeps a handle to every expression it lowered,
/// so no other node can take a remembered node's address.
pub(crate) struct Lowerer<'a> {
    symbol_types: &'a dyn SymbolTypes,
    terms: Vec<TermNode>,
    symbols: Vec<Symbol>,
    symbol_indices: HashMap<u64, usize>,
    memo: HashMap<NodeIdentity, TermId, BuildIdentityHasher>,
    lowered: Vec<Expression>,
}

impl<'a> Lowerer<'a> {
    /// Return the lowerer reading the sorts of identifiers from
    /// `symbol_types`.
    pub(crate) fn new(symbol_types: &'a dyn SymbolTypes) -> Self {
        Self {
            symbol_types,
            terms: Vec::new(),
            symbols: Vec::new(),
            symbol_indices: HashMap::new(),
            memo: HashMap::default(),
            lowered: Vec::new(),
        }
    }

    /// Return the sort of the term `id`.
    pub(crate) fn sort(&self, id: TermId) -> SymbolType {
        self.terms[id.index()].sort
    }

    /// Return the index of the symbol of `identifier`, if a lowered
    /// expression referred to it.
    pub(crate) fn symbol_of(&self, identifier: &Identifier) -> Option<usize> {
        self.symbol_indices.get(&identifier.id()).copied()
    }

    /// Lower `expression`, returning the id of its term.
    ///
    /// The nodes are lowered in post-order, so the refusal is that of the
    /// first node, in post-order, that has no term.
    pub(crate) fn lower(&mut self, expression: &Expression) -> Result<TermId, LoweringError> {
        self.lowered.push(expression.clone());
        let mut pending = vec![(expression, false)];
        while let Some((node, is_expanded)) = pending.pop() {
            if self.memo.contains_key(&node.identity()) {
                continue;
            }
            if !is_expanded {
                pending.push((node, true));
                pending.extend(
                    node.children()
                        .rev()
                        .filter(|child| !self.memo.contains_key(&child.identity()))
                        .map(|child| (child, false)),
                );
                continue;
            }
            let arguments: Vec<TermId> = node
                .children()
                .map(|child| self.memo[&child.identity()])
                .collect();
            let term = self.lower_node(node, &arguments)?;
            self.memo.insert(node.identity(), term);
        }
        Ok(self.memo[&expression.identity()])
    }

    /// Return the term of `node`, whose children lowered to `arguments`.
    fn lower_node(
        &mut self,
        node: &Expression,
        arguments: &[TermId],
    ) -> Result<TermId, LoweringError> {
        let mismatch = || LoweringError::SortMismatch(node.clone());
        match node.kind() {
            ExpressionKind::Identifier(identifier) => self.lower_identifier(identifier),
            ExpressionKind::Literal(literal) => self.lower_literal(node, literal),
            ExpressionKind::Unary(unary) => {
                let operand = arguments[0];
                match unary.operation() {
                    UnaryOperation::LogicalNot => {
                        self.require(operand, SymbolType::Bool)
                            .ok_or_else(mismatch)?;
                        Ok(self.apply(Operator::Not, vec![operand], SymbolType::Bool))
                    }
                    UnaryOperation::Negate => {
                        self.require_number(operand).ok_or_else(mismatch)?;
                        Ok(self.negate(operand))
                    }
                    UnaryOperation::Positive => {
                        self.require_number(operand).ok_or_else(mismatch)?;
                        Ok(operand)
                    }
                }
            }
            ExpressionKind::Binary(binary) => {
                let (left, right) = (arguments[0], arguments[1]);
                self.lower_binary(node, binary.operation(), left, right, binary.right())
            }
            ExpressionKind::Logical(logical) => {
                for &operand in arguments {
                    self.require(operand, SymbolType::Bool)
                        .ok_or_else(mismatch)?;
                }
                let operator = match logical.operation() {
                    LogicalOperation::And => Operator::And,
                    LogicalOperation::Or => Operator::Or,
                };
                Ok(match arguments {
                    [] => self.push(
                        Term::Bool(operator == Operator::And),
                        SymbolType::Bool,
                        true,
                    ),
                    [only] => *only,
                    _ => self.apply(operator, arguments.to_vec(), SymbolType::Bool),
                })
            }
            ExpressionKind::Piecewise(_) => self.lower_piecewise(node, arguments),
            ExpressionKind::Call(_) => Err(LoweringError::Call(node.clone())),
        }
    }

    fn lower_identifier(&mut self, identifier: &Identifier) -> Result<TermId, LoweringError> {
        let index = if let Some(index) = self.symbol_of(identifier) {
            index
        } else {
            let sort = self
                .symbol_types
                .symbol_type(identifier)
                .ok_or_else(|| LoweringError::MissingSymbolTypes(vec![identifier.clone()]))?;
            self.symbols.push(Symbol {
                identifier: Some(identifier.clone()),
                name: format!("{}_{}", sanitize(identifier.name_hint()), identifier.id()),
                sort,
            });
            self.symbol_indices
                .insert(identifier.id(), self.symbols.len() - 1);
            self.symbols.len() - 1
        };
        let sort = self.symbols[index].sort;
        Ok(self.push(Term::Symbol(index), sort, false))
    }

    fn lower_literal(
        &mut self,
        node: &Expression,
        literal: &LiteralValue,
    ) -> Result<TermId, LoweringError> {
        Ok(match literal {
            LiteralValue::Bool(value) => self.push(Term::Bool(*value), SymbolType::Bool, true),
            LiteralValue::Int(value) => self.integer(value.clone()),
            LiteralValue::Float(value) if value.is_finite() => {
                let (numerator, denominator) = rationalize_float(*value);
                self.rational(numerator, denominator)
            }
            LiteralValue::Float(_) => return Err(LoweringError::NonFiniteLiteral(node.clone())),
            LiteralValue::Decimal(decimal) => {
                let (numerator, denominator) = rationalize_decimal(decimal);
                self.rational(numerator, denominator)
            }
        })
    }

    fn lower_binary(
        &mut self,
        node: &Expression,
        operation: BinaryOperation,
        left: TermId,
        right: TermId,
        right_node: &Expression,
    ) -> Result<TermId, LoweringError> {
        let mismatch = || LoweringError::SortMismatch(node.clone());
        match operation {
            BinaryOperation::Equal | BinaryOperation::NotEqual => {
                let operator = if operation == BinaryOperation::Equal {
                    Operator::Equal
                } else {
                    Operator::Distinct
                };
                let (left, right) = match (self.sort(left), self.sort(right)) {
                    (SymbolType::Bool, SymbolType::Bool) => (left, right),
                    (SymbolType::Bool, _) | (_, SymbolType::Bool) => return Err(mismatch()),
                    _ => self.unify(left, right),
                };
                Ok(self.apply(operator, vec![left, right], SymbolType::Bool))
            }
            BinaryOperation::Less
            | BinaryOperation::LessEqual
            | BinaryOperation::Greater
            | BinaryOperation::GreaterEqual => {
                self.require_number(left).ok_or_else(mismatch)?;
                self.require_number(right).ok_or_else(mismatch)?;
                let (left, right) = self.unify(left, right);
                let operator = match operation {
                    BinaryOperation::Less => Operator::Less,
                    BinaryOperation::LessEqual => Operator::LessEqual,
                    BinaryOperation::Greater => Operator::Greater,
                    _ => Operator::GreaterEqual,
                };
                Ok(self.apply(operator, vec![left, right], SymbolType::Bool))
            }
            _ => {
                self.require_number(left).ok_or_else(mismatch)?;
                self.require_number(right).ok_or_else(mismatch)?;
                self.lower_arithmetic(node, operation, left, right, right_node)
            }
        }
    }

    fn lower_arithmetic(
        &mut self,
        node: &Expression,
        operation: BinaryOperation,
        left: TermId,
        right: TermId,
        right_node: &Expression,
    ) -> Result<TermId, LoweringError> {
        let are_integers =
            self.sort(left) == SymbolType::Int && self.sort(right) == SymbolType::Int;
        Ok(match operation {
            BinaryOperation::Add | BinaryOperation::Subtract | BinaryOperation::Multiply => {
                let operator = match operation {
                    BinaryOperation::Add => Operator::Add,
                    BinaryOperation::Subtract => Operator::Subtract,
                    _ => Operator::Multiply,
                };
                let (left, right) = self.unify(left, right);
                let sort = self.sort(left);
                self.apply(operator, vec![left, right], sort)
            }
            BinaryOperation::Divide => {
                let (left, right) = (self.convert_to_real(left), self.convert_to_real(right));
                self.apply(Operator::Divide, vec![left, right], SymbolType::Real)
            }
            BinaryOperation::FloorDivide if are_integers => {
                self.floor_integers(Operator::IntDivide, left, right)
            }
            BinaryOperation::FloorMod if are_integers => {
                self.floor_integers(Operator::Modulo, left, right)
            }
            BinaryOperation::FloorDivide => {
                let floor = self.floor_real_quotient(left, right);
                self.convert_to_real(floor)
            }
            BinaryOperation::FloorMod => {
                let (dividend, divisor) = (self.convert_to_real(left), self.convert_to_real(right));
                let floor = self.floor_real_quotient(left, right);
                let quotient = self.convert_to_real(floor);
                let product = self.apply(
                    Operator::Multiply,
                    vec![divisor, quotient],
                    SymbolType::Real,
                );
                self.apply(
                    Operator::Subtract,
                    vec![dividend, product],
                    SymbolType::Real,
                )
            }
            BinaryOperation::Power => match right_node.kind() {
                ExpressionKind::Literal(LiteralValue::Int(exponent))
                    if exponent.sign() == Sign::Plus =>
                {
                    self.power(left, exponent)
                }
                _ => return Err(LoweringError::UnsupportedPower(node.clone())),
            },
            _ => unreachable!("{operation:?} is no arithmetic operation"),
        })
    }

    /// Return `(div a b)` or `(mod a b)` with floor semantics: as it is for
    /// a positive divisor, through the negated operands for a negative one,
    /// and an `ite` of both for a divisor of unknown sign.
    fn floor_integers(&mut self, operator: Operator, dividend: TermId, divisor: TermId) -> TermId {
        let is_modulo = operator == Operator::Modulo;
        if let Term::Integer(value) = &self.terms[divisor.index()].term {
            if value.is_negative() {
                return self.floor_by_a_negative(operator, dividend, divisor, is_modulo);
            }
            return self.apply(operator, vec![dividend, divisor], SymbolType::Int);
        }
        let zero = self.integer(BigInt::zero());
        let is_positive = self.apply(Operator::Greater, vec![divisor, zero], SymbolType::Bool);
        let positive = self.apply(operator, vec![dividend, divisor], SymbolType::Int);
        let negative = self.floor_by_a_negative(operator, dividend, divisor, is_modulo);
        self.apply(
            Operator::Ite,
            vec![is_positive, positive, negative],
            SymbolType::Int,
        )
    }

    /// Return the floor division or modulo of `dividend` by the negative
    /// `divisor`, through the negated operands.
    fn floor_by_a_negative(
        &mut self,
        operator: Operator,
        dividend: TermId,
        divisor: TermId,
        is_modulo: bool,
    ) -> TermId {
        let (dividend, divisor) = (self.negate(dividend), self.negate(divisor));
        let applied = self.apply(operator, vec![dividend, divisor], SymbolType::Int);
        if is_modulo {
            self.negate(applied)
        } else {
            applied
        }
    }

    /// Return `(to_int (/ a b))` of the operands converted to reals: the
    /// floor of their quotient, an integer.
    fn floor_real_quotient(&mut self, dividend: TermId, divisor: TermId) -> TermId {
        let (dividend, divisor) = (
            self.convert_to_real(dividend),
            self.convert_to_real(divisor),
        );
        let quotient = self.apply(Operator::Divide, vec![dividend, divisor], SymbolType::Real);
        self.apply(Operator::ToInt, vec![quotient], SymbolType::Int)
    }

    /// Return `base` to the positive `exponent`, a product built by
    /// squaring.
    fn power(&mut self, base: TermId, exponent: &BigInt) -> TermId {
        let sort = self.sort(base);
        let mut remaining = exponent.clone();
        let mut square = base;
        let mut result: Option<TermId> = None;
        loop {
            if remaining.bit(0) {
                result = Some(match result {
                    None => square,
                    Some(product) => self.apply(Operator::Multiply, vec![product, square], sort),
                });
            }
            remaining >>= 1_u32;
            if remaining.is_zero() {
                break;
            }
            square = self.apply(Operator::Multiply, vec![square, square], sort);
        }
        result.unwrap_or(base)
    }

    fn lower_piecewise(
        &mut self,
        node: &Expression,
        arguments: &[TermId],
    ) -> Result<TermId, LoweringError> {
        let mismatch = || LoweringError::SortMismatch(node.clone());
        let (cases, otherwise) = arguments.split_at(arguments.len() - 1);
        let mut values: Vec<TermId> = cases.iter().skip(1).step_by(2).copied().collect();
        values.push(otherwise[0]);
        for &condition in cases.iter().step_by(2) {
            self.require(condition, SymbolType::Bool)
                .ok_or_else(mismatch)?;
        }
        let sorts: Vec<SymbolType> = values.iter().map(|&value| self.sort(value)).collect();
        let sort = if sorts.iter().all(|&sort| sort == SymbolType::Bool) {
            SymbolType::Bool
        } else if sorts.contains(&SymbolType::Bool) {
            return Err(mismatch());
        } else if sorts.contains(&SymbolType::Real) {
            SymbolType::Real
        } else {
            SymbolType::Int
        };
        let values: Vec<TermId> = values
            .into_iter()
            .map(|value| {
                if sort == SymbolType::Real {
                    self.convert_to_real(value)
                } else {
                    value
                }
            })
            .collect();
        let mut result = values[values.len() - 1];
        for (index, &condition) in cases.iter().step_by(2).enumerate().rev() {
            result = self.apply(Operator::Ite, vec![condition, values[index], result], sort);
        }
        Ok(result)
    }

    /// Return `Some(())` if the term `id` has `sort`.
    fn require(&self, id: TermId, sort: SymbolType) -> Option<()> {
        (self.sort(id) == sort).then_some(())
    }

    /// Return `Some(())` if the term `id` is a number.
    fn require_number(&self, id: TermId) -> Option<()> {
        (self.sort(id) != SymbolType::Bool).then_some(())
    }

    /// Return the two numeric terms with the integer one converted to a
    /// real when the other is a real.
    fn unify(&mut self, left: TermId, right: TermId) -> (TermId, TermId) {
        if self.sort(left) == SymbolType::Real || self.sort(right) == SymbolType::Real {
            (self.convert_to_real(left), self.convert_to_real(right))
        } else {
            (left, right)
        }
    }

    /// Return the numeric term `id` as a real: itself if it is one, the
    /// real numeral of an integer constant, or `(to_real id)`.
    pub(crate) fn convert_to_real(&mut self, id: TermId) -> TermId {
        if self.sort(id) == SymbolType::Real {
            return id;
        }
        if let Term::Integer(value) = &self.terms[id.index()].term {
            let value = value.clone();
            return self.rational(value, BigInt::one());
        }
        self.apply(Operator::ToReal, vec![id], SymbolType::Real)
    }

    /// Return the negation of the numeric term `id`, folded for a constant.
    fn negate(&mut self, id: TermId) -> TermId {
        match &self.terms[id.index()].term {
            Term::Integer(value) => {
                let value = -value;
                self.integer(value)
            }
            Term::Rational {
                numerator,
                denominator,
            } => {
                let (numerator, denominator) = (-numerator, denominator.clone());
                self.rational(numerator, denominator)
            }
            _ => {
                let sort = self.sort(id);
                self.apply(Operator::Negate, vec![id], sort)
            }
        }
    }

    /// Return the integer constant `value`.
    fn integer(&mut self, value: BigInt) -> TermId {
        self.push(Term::Integer(value), SymbolType::Int, true)
    }

    /// Return the real constant `numerator / denominator`, already in
    /// lowest terms with a positive denominator.
    fn rational(&mut self, numerator: BigInt, denominator: BigInt) -> TermId {
        self.push(
            Term::Rational {
                numerator,
                denominator,
            },
            SymbolType::Real,
            true,
        )
    }

    /// Return the application of `operator` to `arguments`, of `sort`.
    pub(crate) fn apply(
        &mut self,
        operator: Operator,
        arguments: Vec<TermId>,
        sort: SymbolType,
    ) -> TermId {
        let is_ground = arguments
            .iter()
            .all(|argument| self.terms[argument.index()].is_ground);
        self.push(Term::Apply(operator, arguments.into()), sort, is_ground)
    }

    fn push(&mut self, term: Term, sort: SymbolType, is_ground: bool) -> TermId {
        self.terms.push(TermNode {
            term,
            sort,
            is_ground,
        });
        TermId::at(self.terms.len() - 1)
    }

    /// Return the script of `assertions`, declaring every symbol they do
    /// not quantify, and naming `value` as the constant `value` when given.
    pub(crate) fn finish(
        mut self,
        mut assertions: Vec<Assertion>,
        value: Option<TermId>,
    ) -> SmtScript {
        let value_sort = value.map(|value| {
            let sort = self.sort(value);
            self.symbols.push(Symbol {
                identifier: None,
                name: VALUE_SYMBOL.to_owned(),
                sort,
            });
            let symbol = self.push(Term::Symbol(self.symbols.len() - 1), sort, false);
            let body = self.apply(Operator::Equal, vec![symbol, value], SymbolType::Bool);
            assertions.push(Assertion {
                quantified: Vec::new(),
                body,
            });
            sort
        });
        let logic = self.select_logic(&assertions);
        let mut declarations: Vec<Declaration> = self
            .symbols
            .iter()
            .enumerate()
            .filter(|(index, _)| {
                !assertions
                    .iter()
                    .any(|assertion| assertion.quantified.contains(index))
            })
            .filter_map(|(_, symbol)| {
                symbol.identifier.as_ref().map(|identifier| Declaration {
                    identifier: identifier.clone(),
                    symbol: symbol.name.clone(),
                    sort: symbol.sort,
                })
            })
            .collect();
        declarations.sort_by_key(|declaration| declaration.identifier.id());
        SmtScript {
            logic,
            symbols: self.symbols,
            declarations,
            terms: self.terms,
            assertions,
            value_sort,
        }
    }

    /// Return the narrowest logic of the terms `assertions` reach.
    ///
    /// Solvers read linearity from the syntax, so a product is linear only
    /// when all its factors but one are numerals, and a quotient, `div` or
    /// `mod` only when its divisor is one.
    fn select_logic(&self, assertions: &[Assertion]) -> Logic {
        let mut has_int = false;
        let mut has_real = false;
        let mut is_nonlinear = false;
        let mut visited = vec![false; self.terms.len()];
        let mut pending: Vec<TermId> = assertions.iter().map(|assertion| assertion.body).collect();
        while let Some(id) = pending.pop() {
            if std::mem::replace(&mut visited[id.index()], true) {
                continue;
            }
            let node = &self.terms[id.index()];
            has_int |= node.sort == SymbolType::Int;
            has_real |= node.sort == SymbolType::Real;
            if let Term::Apply(operator, arguments) = &node.term {
                let is_numeral = |argument: &TermId| {
                    matches!(
                        self.terms[argument.index()].term,
                        Term::Integer(_) | Term::Rational { .. }
                    )
                };
                is_nonlinear |= match operator {
                    Operator::Multiply => {
                        arguments
                            .iter()
                            .filter(|argument| !is_numeral(argument))
                            .count()
                            > 1
                    }
                    Operator::Divide | Operator::IntDivide | Operator::Modulo => {
                        !is_numeral(&arguments[1])
                    }
                    _ => false,
                };
                pending.extend(arguments.iter().copied());
            }
        }
        let is_quantified = assertions
            .iter()
            .any(|assertion| !assertion.quantified.is_empty());
        match (has_int, has_real, is_nonlinear, is_quantified) {
            (true, true, _, _) | (false, false, _, _) => Logic::All,
            (true, false, false, false) => Logic::QfLia,
            (true, false, true, false) => Logic::QfNia,
            (false, true, false, false) => Logic::QfLra,
            (false, true, true, false) => Logic::QfNra,
            (true, false, false, true) => Logic::Lia,
            (true, false, true, true) => Logic::Nia,
            (false, true, false, true) => Logic::Lra,
            (false, true, true, true) => Logic::Nra,
        }
    }
}
