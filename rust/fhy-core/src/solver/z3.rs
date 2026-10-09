//! A backend that decides scripts with the z3 library, behind the `z3`
//! feature.

use std::borrow::Cow;
use std::error::Error;
use std::fmt;

use ::z3::ast::{Ast, Bool, Int, Real};
use ::z3::{Config, Solver as Z3SolverSession, with_z3_config};
use num_bigint::BigInt;
use num_traits::Signed;

use crate::expression::SymbolType;

use super::backend::{CheckLimits, SatResult, SmtSolver};
use super::smt::{Operator, SmtScript, Term, TermId};
use crate::foreign::BoxError;

/// An [`SmtSolver`] that decides a script with the z3 library, linked
/// through the `z3` crate.
///
/// Each check runs in a fresh z3 context, with the timeout of its
/// [`CheckLimits`] in the context's configuration, so no z3 state outlives
/// a check and checks on several threads do not share one. The backend
/// builds z3 terms from the script's typed terms, with no text in between.
/// It holds no state, so it is `Send + Sync` and cheap to share.
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
///
/// use fhy_core::expression::{Expression, SymbolType};
/// use fhy_core::identifier::Identifier;
/// use fhy_core::solver::{Answer, QueryContext, Question, Solver, Z3Solver};
///
/// let x = Identifier::new("x");
/// let positive = Expression::from(x.clone()).greater(0);
/// let symbol_types = HashMap::from([(x, SymbolType::Int)]);
/// let solver = Solver::new().with_smt_solver(Z3Solver::new());
///
/// let answer = solver.ask(&Question::Satisfiability(&positive), &QueryContext::new(&symbol_types))?;
///
/// assert_eq!(answer, Answer::Yes);
/// # Ok::<(), fhy_core::solver::SolveError>(())
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(docsrs, doc(cfg(feature = "z3")))]
pub struct Z3Solver;

impl Z3Solver {
    /// Return the backend.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl SmtSolver for Z3Solver {
    /// Return `z3`.
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed("z3")
    }

    fn check(&self, script: &SmtScript, limits: &CheckLimits) -> Result<SatResult, BoxError> {
        let mut config = Config::new();
        if let Some(timeout) = limits.timeout() {
            let milliseconds = u64::try_from(timeout.as_millis())
                .unwrap_or(u64::MAX)
                .max(1);
            config.set_timeout_msec(milliseconds);
        }
        with_z3_config(&config, || check_in_context(script))
            .map_err(|error| Box::new(error) as BoxError)
    }
}

/// A term the z3 library cannot build, which only a malformed script holds.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(docsrs, doc(cfg(feature = "z3")))]
pub struct Z3TermError {
    text: String,
}

impl fmt::Display for Z3TermError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "z3 cannot build the term {}", self.text)
    }
}

impl Error for Z3TermError {}

/// A z3 term of one of the script's sorts.
#[derive(Clone)]
enum Z3Term {
    Bool(Bool),
    Int(Int),
    Real(Real),
}

impl Z3Term {
    fn as_dynamic(&self) -> &dyn Ast {
        match self {
            Self::Bool(term) => term,
            Self::Int(term) => term,
            Self::Real(term) => term,
        }
    }
}

/// Decide `script` in the current thread's z3 context.
fn check_in_context(script: &SmtScript) -> Result<SatResult, Z3TermError> {
    let symbols: Vec<Z3Term> = script
        .symbols()
        .iter()
        .map(|symbol| match symbol.sort {
            SymbolType::Bool => Z3Term::Bool(Bool::new_const(symbol.name.as_str())),
            SymbolType::Int => Z3Term::Int(Int::new_const(symbol.name.as_str())),
            SymbolType::Real => Z3Term::Real(Real::new_const(symbol.name.as_str())),
        })
        .collect();
    let mut terms: Vec<Z3Term> = Vec::with_capacity(script.term_count());
    for index in 0..script.term_count() {
        let term = build_term(script, TermId::at(index), &symbols, &terms)?;
        terms.push(term);
    }
    let solver = Z3SolverSession::new();
    for assertion in script.assertions() {
        let Z3Term::Bool(body) = &terms[assertion.body.index()] else {
            return Err(term_error(script, assertion.body));
        };
        if assertion.quantified.is_empty() {
            solver.assert(body);
        } else {
            let bound: Vec<&dyn Ast> = assertion
                .quantified
                .iter()
                .map(|&index| symbols[index].as_dynamic())
                .collect();
            solver.assert(::z3::ast::forall_const(&bound, &[], body));
        }
    }
    Ok(match solver.check() {
        ::z3::SatResult::Sat => SatResult::Sat,
        ::z3::SatResult::Unsat => SatResult::Unsat,
        ::z3::SatResult::Unknown => SatResult::Unknown {
            reason: solver.get_reason_unknown().unwrap_or_default(),
        },
    })
}

fn term_error(script: &SmtScript, id: TermId) -> Z3TermError {
    Z3TermError {
        text: format!("{:?}", script.term(id).term),
    }
}

/// Return the z3 integer `value`.
fn build_integer(value: &BigInt) -> Option<Int> {
    let magnitude: Int = value.abs().to_string().parse().ok()?;
    Some(if value.is_negative() {
        magnitude.unary_minus()
    } else {
        magnitude
    })
}

/// Return the z3 real `numerator / denominator`.
fn build_rational(numerator: &BigInt, denominator: &BigInt) -> Option<Real> {
    let magnitude =
        Real::from_rational_str(&numerator.abs().to_string(), &denominator.to_string())?;
    Some(if numerator.is_negative() {
        magnitude.unary_minus()
    } else {
        magnitude
    })
}

/// Build the z3 term of the term `id`, whose arguments are among the
/// already built `terms`.
fn build_term(
    script: &SmtScript,
    id: TermId,
    symbols: &[Z3Term],
    terms: &[Z3Term],
) -> Result<Z3Term, Z3TermError> {
    let error = || term_error(script, id);
    let node = script.term(id);
    let term = match &node.term {
        Term::Symbol(index) => symbols[*index].clone(),
        Term::Bool(value) => Z3Term::Bool(Bool::from_bool(*value)),
        Term::Integer(value) => Z3Term::Int(build_integer(value).ok_or_else(error)?),
        Term::Rational {
            numerator,
            denominator,
        } => Z3Term::Real(build_rational(numerator, denominator).ok_or_else(error)?),
        Term::Apply(operator, arguments) => {
            let arguments: Vec<&Z3Term> = arguments
                .iter()
                .map(|argument| &terms[argument.index()])
                .collect();
            apply(*operator, &arguments).ok_or_else(error)?
        }
    };
    Ok(term)
}

/// Return the z3 term applying `operator` to `arguments`, or `None` when
/// their sorts do not fit it.
fn apply(operator: Operator, arguments: &[&Z3Term]) -> Option<Z3Term> {
    use Z3Term::{Bool as B, Int as I, Real as R};
    Some(match (operator, arguments) {
        (Operator::Add | Operator::Subtract | Operator::Multiply, [I(_), ..]) => {
            let operands = integers(arguments)?;
            I(match operator {
                Operator::Add => Int::add(&operands),
                Operator::Subtract => Int::sub(&operands),
                _ => Int::mul(&operands),
            })
        }
        (Operator::Add | Operator::Subtract | Operator::Multiply, [R(_), ..]) => {
            let operands = reals(arguments)?;
            R(match operator {
                Operator::Add => Real::add(&operands),
                Operator::Subtract => Real::sub(&operands),
                _ => Real::mul(&operands),
            })
        }
        (Operator::Negate, [I(operand)]) => I(operand.unary_minus()),
        (Operator::Negate, [R(operand)]) => R(operand.unary_minus()),
        (Operator::Divide, [R(left), R(right)]) => R(left.div(right)),
        (Operator::IntDivide, [I(left), I(right)]) => I(left.div(right)),
        (Operator::Modulo, [I(left), I(right)]) => I(left.modulo(right)),
        (Operator::ToReal, [I(operand)]) => R(operand.to_real()),
        (Operator::ToInt, [R(operand)]) => I(operand.to_int()),
        (Operator::Equal | Operator::Distinct, [left, right]) => {
            let equal = match (left, right) {
                (B(left), B(right)) => left.eq(right),
                (I(left), I(right)) => left.eq(right),
                (R(left), R(right)) => left.eq(right),
                _ => return None,
            };
            B(if operator == Operator::Equal {
                equal
            } else {
                equal.not()
            })
        }
        (
            Operator::Less | Operator::LessEqual | Operator::Greater | Operator::GreaterEqual,
            [I(left), I(right)],
        ) => B(match operator {
            Operator::Less => left.lt(right),
            Operator::LessEqual => left.le(right),
            Operator::Greater => left.gt(right),
            _ => left.ge(right),
        }),
        (
            Operator::Less | Operator::LessEqual | Operator::Greater | Operator::GreaterEqual,
            [R(left), R(right)],
        ) => B(match operator {
            Operator::Less => left.lt(right),
            Operator::LessEqual => left.le(right),
            Operator::Greater => left.gt(right),
            _ => left.ge(right),
        }),
        (Operator::And | Operator::Or, _) => {
            let operands = booleans(arguments)?;
            B(if operator == Operator::And {
                Bool::and(&operands)
            } else {
                Bool::or(&operands)
            })
        }
        (Operator::Not, [B(operand)]) => B(operand.not()),
        (Operator::Ite, [B(condition), then, otherwise]) => match (then, otherwise) {
            (B(then), B(otherwise)) => B(condition.ite(then, otherwise)),
            (I(then), I(otherwise)) => I(condition.ite(then, otherwise)),
            (R(then), R(otherwise)) => R(condition.ite(then, otherwise)),
            _ => return None,
        },
        _ => return None,
    })
}

fn integers(arguments: &[&Z3Term]) -> Option<Vec<Int>> {
    arguments
        .iter()
        .map(|argument| match argument {
            Z3Term::Int(term) => Some(term.clone()),
            _ => None,
        })
        .collect()
}

fn reals(arguments: &[&Z3Term]) -> Option<Vec<Real>> {
    arguments
        .iter()
        .map(|argument| match argument {
            Z3Term::Real(term) => Some(term.clone()),
            _ => None,
        })
        .collect()
}

fn booleans(arguments: &[&Z3Term]) -> Option<Vec<Bool>> {
    arguments
        .iter()
        .map(|argument| match argument {
            Z3Term::Bool(term) => Some(term.clone()),
            _ => None,
        })
        .collect()
}
