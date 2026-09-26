//! SMT-LIB2 scripts: the lowering of expressions to typed terms, and the
//! text a solver reads.

mod lower;
mod print;
mod term;

use std::fmt;

use crate::expression::{BooleanScreen, Expression, SortLookup, SymbolType, SymbolTypes};
use crate::identifier::Identifier;

use super::error::LoweringError;
use super::screen::{find_native_constants, is_native_constant};

pub(crate) use lower::Lowerer;
#[cfg(not(feature = "z3"))]
use term::TermId;
pub(crate) use term::{Assertion, Operator};
use term::{Symbol, TermNode};
#[cfg(feature = "z3")]
pub(crate) use term::{Term, TermId};

/// An SMT-LIB2 logic: the theories and quantifiers a script uses.
///
/// Displays as its SMT-LIB2 name, such as `QF_LIA`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Logic {
    /// Quantifier-free linear integer arithmetic.
    QfLia,
    /// Quantifier-free linear real arithmetic.
    QfLra,
    /// Quantifier-free nonlinear integer arithmetic.
    QfNia,
    /// Quantifier-free nonlinear real arithmetic.
    QfNra,
    /// Linear integer arithmetic with quantifiers.
    Lia,
    /// Linear real arithmetic with quantifiers.
    Lra,
    /// Nonlinear integer arithmetic with quantifiers.
    Nia,
    /// Nonlinear real arithmetic with quantifiers.
    Nra,
    /// Every logic a solver supports: for terms of both numeric sorts, or of
    /// none.
    All,
}

impl Logic {
    /// Return the SMT-LIB2 name of the logic, such as `"QF_LIA"`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::QfLia => "QF_LIA",
            Self::QfLra => "QF_LRA",
            Self::QfNia => "QF_NIA",
            Self::QfNra => "QF_NRA",
            Self::Lia => "LIA",
            Self::Lra => "LRA",
            Self::Nia => "NIA",
            Self::Nra => "NRA",
            Self::All => "ALL",
        }
    }
}

impl fmt::Display for Logic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A constant a script declares for an identifier: its symbol and its
/// sort.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Declaration {
    identifier: Identifier,
    symbol: String,
    sort: SymbolType,
}

impl Declaration {
    /// Return the identifier the constant stands for.
    #[must_use]
    pub fn identifier(&self) -> &Identifier {
        &self.identifier
    }

    /// Return the constant's symbol, `<name hint>_<id>` with every `|` and
    /// `\` of the name hint replaced by `_`. The script writes it quoted,
    /// as `|x_7|`.
    #[must_use]
    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    /// Return the constant's sort.
    #[must_use]
    pub fn sort(&self) -> SymbolType {
        self.sort
    }
}

/// An SMT-LIB2 script: a logic, the declarations of constants, and
/// assertions over them, as typed terms.
///
/// `Display` writes the script's text, one command per line: `(set-logic
/// ...)`, a `declare-const` for each declared constant, ordered by
/// identifier id, and an `assert` for each assertion. It writes no
/// `check-sat`; that is the backend's command. A term shared by several
/// places of an assertion is written once, bound by a `let` to a symbol
/// `t!1`, `t!2`, ..., which no identifier's symbol can equal, since those
/// end in `_` and digits.
///
/// [`lower`](Self::lower) builds the script of one expression, and a
/// [`Solver`](super::Solver) builds the script of each question it asks.
#[derive(Debug, Clone)]
pub struct SmtScript {
    logic: Logic,
    symbols: Vec<Symbol>,
    declarations: Vec<Declaration>,
    terms: Vec<TermNode>,
    assertions: Vec<Assertion>,
    value_sort: Option<SymbolType>,
}

impl SmtScript {
    /// Lower `expression` to a script.
    ///
    /// A Boolean expression is the script's one assertion. Any other
    /// expression is named: the script declares the constant `value` of
    /// the expression's sort and asserts `(= value e)`, and
    /// [`value_sort`](Self::value_sort) reports that sort.
    ///
    /// The lowering keeps this crate's semantics: division is exact, floor
    /// division and floor modulo round toward negative infinity, and an
    /// integer operand meeting a real one is converted with `to_real`. A
    /// float or decimal literal is its exact rational, written with
    /// decimal numerals, and an integer literal in a real position is a
    /// real numeral. A power with an integer literal exponent of at least
    /// one is a product built by squaring.
    ///
    /// # Errors
    ///
    /// Returns, checked in this order:
    ///
    /// - [`LoweringError::MissingSymbolTypes`] naming every identifier
    ///   other than a native constant's that `symbol_types` declares no
    ///   kind for;
    /// - [`LoweringError::IllTyped`] if a Boolean position holds an
    ///   operand that provably denotes a number, as
    ///   [`BooleanScreen::check_logical_operands`](crate::expression::BooleanScreen::check_logical_operands)
    ///   judges it;
    /// - [`LoweringError::NativeConstants`] naming every native constant
    ///   the expression refers to;
    /// - the refusal of the first node, in post-order, that has no term: a
    ///   non-finite float, a call, a Boolean meeting a number, or a power
    ///   with another exponent.
    pub fn lower(
        expression: &Expression,
        symbol_types: &dyn SymbolTypes,
        sorts: &dyn SortLookup,
    ) -> Result<Self, LoweringError> {
        let mut missing: Vec<Identifier> = expression
            .free_identifiers()
            .into_iter()
            .filter(|identifier| {
                symbol_types.symbol_type(identifier).is_none()
                    && !is_native_constant(identifier, sorts)
            })
            .collect();
        if !missing.is_empty() {
            missing.sort_by_key(Identifier::id);
            return Err(LoweringError::MissingSymbolTypes(missing));
        }
        BooleanScreen::new()
            .with_sorts(sorts)
            .with_symbol_types(symbol_types)
            .check_logical_operands(expression)
            .map_err(LoweringError::IllTyped)?;
        let constants = find_native_constants(expression, sorts);
        if !constants.is_empty() {
            return Err(LoweringError::NativeConstants(constants));
        }
        let mut lowerer = Lowerer::new(symbol_types);
        let term = lowerer.lower(expression)?;
        Ok(if lowerer.sort(term) == SymbolType::Bool {
            lowerer.finish(
                vec![Assertion {
                    quantified: Vec::new(),
                    body: term,
                }],
                None,
            )
        } else {
            lowerer.finish(Vec::new(), Some(term))
        })
    }

    /// Return the logic of the script.
    #[must_use]
    pub fn logic(&self) -> Logic {
        self.logic
    }

    /// Return the constants the script declares for identifiers, ordered by
    /// id. The `value` constant of a named expression is not among them.
    #[must_use]
    pub fn declarations(&self) -> &[Declaration] {
        &self.declarations
    }

    /// Return the sort of the named expression when the script names one,
    /// and `None` when its assertions are predicates.
    #[must_use]
    pub fn value_sort(&self) -> Option<SymbolType> {
        self.value_sort
    }

    /// Return the term with id `id`.
    pub(crate) fn term(&self, id: TermId) -> &TermNode {
        &self.terms[id.index()]
    }

    /// Return the symbols the terms refer to.
    #[cfg(feature = "z3")]
    pub(crate) fn symbols(&self) -> &[Symbol] {
        &self.symbols
    }

    /// Return how many terms the script holds; their ids are
    /// `0..term_count()`, each argument's before the term applying it.
    #[cfg(feature = "z3")]
    pub(crate) fn term_count(&self) -> usize {
        self.terms.len()
    }

    /// Return the assertions.
    #[cfg(feature = "z3")]
    pub(crate) fn assertions(&self) -> &[Assertion] {
        &self.assertions
    }
}

impl fmt::Display for SmtScript {
    /// Write the script's text.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.write(f)
    }
}
