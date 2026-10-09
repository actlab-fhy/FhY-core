//! The typed terms of a script, kept in one arena: a term refers to its
//! arguments by id, so a shared subterm is one term, and deep terms drop
//! without recursion.

use crate::expression::{BigInt, SymbolType};
use crate::identifier::Identifier;

/// The id of a term in its script's arena. An argument's id is always
/// smaller than the id of the term applying it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct TermId(usize);

impl TermId {
    /// Return the id of the term at `index` of the arena.
    pub(crate) const fn at(index: usize) -> Self {
        Self(index)
    }

    /// Return the index of the term in the arena.
    pub(crate) const fn index(self) -> usize {
        self.0
    }
}

/// The function a term applies to its arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Operator {
    /// `+`, of two or more numbers.
    Add,
    /// `-`, of two numbers.
    Subtract,
    /// `*`, of two numbers.
    Multiply,
    /// `-`, of one number.
    Negate,
    /// `/`, of two reals.
    Divide,
    /// `div`, of two integers.
    IntDivide,
    /// `mod`, of two integers.
    Modulo,
    /// `to_real`, of an integer.
    ToReal,
    /// `to_int`, of a real: its floor.
    ToInt,
    /// `=`, of two terms of one sort.
    Equal,
    /// `distinct`, of two terms of one sort.
    Distinct,
    /// `<`, of two numbers.
    Less,
    /// `<=`, of two numbers.
    LessEqual,
    /// `>`, of two numbers.
    Greater,
    /// `>=`, of two numbers.
    GreaterEqual,
    /// `and`, of two or more Booleans.
    And,
    /// `or`, of two or more Booleans.
    Or,
    /// `not`, of a Boolean.
    Not,
    /// `ite`, of a Boolean condition and two terms of one sort.
    Ite,
}

impl Operator {
    /// Return the operator's SMT-LIB2 symbol.
    pub(crate) const fn symbol(self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Subtract | Self::Negate => "-",
            Self::Multiply => "*",
            Self::Divide => "/",
            Self::IntDivide => "div",
            Self::Modulo => "mod",
            Self::ToReal => "to_real",
            Self::ToInt => "to_int",
            Self::Equal => "=",
            Self::Distinct => "distinct",
            Self::Less => "<",
            Self::LessEqual => "<=",
            Self::Greater => ">",
            Self::GreaterEqual => ">=",
            Self::And => "and",
            Self::Or => "or",
            Self::Not => "not",
            Self::Ite => "ite",
        }
    }
}

/// A term, without its sort.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Term {
    /// The symbol at this index of the script's symbols.
    Symbol(usize),
    /// A Boolean constant.
    Bool(bool),
    /// An integer constant, of any sign.
    Integer(BigInt),
    /// A real constant: a numerator of any sign over a positive
    /// denominator, in lowest terms.
    Rational {
        /// The numerator.
        numerator: BigInt,
        /// The denominator, at least one.
        denominator: BigInt,
    },
    /// An operator applied to its arguments.
    Apply(Operator, Box<[TermId]>),
}

/// A term and its sort.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TermNode {
    /// The term.
    pub(crate) term: Term,
    /// Its sort.
    pub(crate) sort: SymbolType,
    /// Whether no symbol occurs in it.
    pub(crate) is_ground: bool,
}

/// A symbol a term may refer to: an identifier's constant, or the `value`
/// constant of a named expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Symbol {
    /// The identifier, or `None` for the `value` constant.
    pub(crate) identifier: Option<Identifier>,
    /// The symbol's name, written quoted. It holds no control character,
    /// `|` or `\`, so it is printable text and a valid C string, which the
    /// z3 backend names its constants by.
    pub(crate) name: String,
    /// The symbol's sort.
    pub(crate) sort: SymbolType,
}

/// An assertion: a Boolean term, universally quantified over some symbols.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Assertion {
    /// The symbols the assertion quantifies universally, which the script
    /// does not declare; empty for a quantifier-free assertion.
    pub(crate) quantified: Vec<usize>,
    /// The Boolean term asserted.
    pub(crate) body: TermId,
}
