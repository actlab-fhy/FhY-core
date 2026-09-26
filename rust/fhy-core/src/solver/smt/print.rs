//! The text of a script.
//!
//! Terms are written from an explicit work list, so a deep term writes
//! without deep recursion, and a term an assertion reaches from several
//! places is written once, under a `let`.

use std::collections::{HashMap, HashSet};
use std::fmt;

use num_traits::{One, Signed};

use crate::expression::SymbolType;

use super::SmtScript;
use super::term::{Symbol, Term, TermId};

/// Return the SMT-LIB2 name of `sort`.
fn sort_name(sort: SymbolType) -> &'static str {
    match sort {
        SymbolType::Bool => "Bool",
        SymbolType::Int => "Int",
        SymbolType::Real => "Real",
    }
}

/// Write `symbol` as a script writes it: an identifier's symbol quoted, the
/// `value` constant bare.
fn write_symbol(f: &mut fmt::Formatter<'_>, symbol: &Symbol) -> fmt::Result {
    if symbol.identifier.is_some() {
        write!(f, "|{}|", symbol.name)
    } else {
        f.write_str(&symbol.name)
    }
}

/// One step of writing a term.
enum Step<'s> {
    Term(TermId),
    Text(&'s str),
}

impl SmtScript {
    /// Write the script's text.
    pub(super) fn write(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "(set-logic {})", self.logic)?;
        let quantified: HashSet<usize> = self
            .assertions
            .iter()
            .flat_map(|assertion| assertion.quantified.iter().copied())
            .collect();
        let mut declared: Vec<(usize, &Symbol)> = self
            .symbols
            .iter()
            .enumerate()
            .filter(|(index, _)| !quantified.contains(index))
            .collect();
        declared.sort_by_key(|(_, symbol)| {
            symbol
                .identifier
                .as_ref()
                .map_or(u64::MAX, crate::identifier::Identifier::id)
        });
        for (_, symbol) in declared {
            f.write_str("(declare-const ")?;
            write_symbol(f, symbol)?;
            writeln!(f, " {})", sort_name(symbol.sort))?;
        }
        for assertion in &self.assertions {
            f.write_str("(assert ")?;
            if assertion.quantified.is_empty() {
                self.write_body(f, assertion.body)?;
            } else {
                let mut bound: Vec<&Symbol> = assertion
                    .quantified
                    .iter()
                    .map(|&index| &self.symbols[index])
                    .collect();
                bound.sort_by_key(|symbol| {
                    symbol
                        .identifier
                        .as_ref()
                        .map(crate::identifier::Identifier::id)
                });
                f.write_str("(forall (")?;
                for (position, symbol) in bound.into_iter().enumerate() {
                    if position > 0 {
                        f.write_str(" ")?;
                    }
                    f.write_str("(")?;
                    write_symbol(f, symbol)?;
                    write!(f, " {})", sort_name(symbol.sort))?;
                }
                f.write_str(") ")?;
                self.write_body(f, assertion.body)?;
                f.write_str(")")?;
            }
            f.write_str(")\n")?;
        }
        Ok(())
    }

    /// Write the term `root`, binding each application it reaches from
    /// several places to a `let` symbol, in the order of their ids.
    fn write_body(&self, f: &mut fmt::Formatter<'_>, root: TermId) -> fmt::Result {
        let mut references: HashMap<TermId, usize> = HashMap::new();
        let mut visited: HashSet<TermId> = HashSet::new();
        let mut pending = vec![root];
        while let Some(id) = pending.pop() {
            if !visited.insert(id) {
                continue;
            }
            if let Term::Apply(_, arguments) = &self.term(id).term {
                for &argument in arguments {
                    *references.entry(argument).or_default() += 1;
                    pending.push(argument);
                }
            }
        }
        let mut shared: Vec<TermId> = references
            .into_iter()
            .filter(|&(id, count)| count > 1 && matches!(self.term(id).term, Term::Apply(..)))
            .map(|(id, _)| id)
            .collect();
        shared.sort_unstable();
        let names: HashMap<TermId, String> = shared
            .iter()
            .enumerate()
            .map(|(index, &id)| (id, format!("t!{}", index + 1)))
            .collect();
        for &id in &shared {
            write!(f, "(let (({} ", names[&id])?;
            self.write_term(f, id, &names)?;
            f.write_str(")) ")?;
        }
        self.write_term(f, root, &names)?;
        for _ in &shared {
            f.write_str(")")?;
        }
        Ok(())
    }

    /// Write the term `root`, writing each term `names` binds, other than
    /// `root` itself, as its name.
    fn write_term(
        &self,
        f: &mut fmt::Formatter<'_>,
        root: TermId,
        names: &HashMap<TermId, String>,
    ) -> fmt::Result {
        let mut pending = vec![Step::Term(root)];
        while let Some(step) = pending.pop() {
            let id = match step {
                Step::Text(text) => {
                    f.write_str(text)?;
                    continue;
                }
                Step::Term(id) => id,
            };
            if id != root {
                if let Some(name) = names.get(&id) {
                    f.write_str(name)?;
                    continue;
                }
            }
            match &self.term(id).term {
                Term::Symbol(index) => write_symbol(f, &self.symbols[*index])?,
                Term::Bool(value) => write!(f, "{value}")?,
                Term::Integer(value) if value.is_negative() => write!(f, "(- {})", value.abs())?,
                Term::Integer(value) => write!(f, "{value}")?,
                Term::Rational {
                    numerator,
                    denominator,
                } => {
                    let magnitude = if denominator.is_one() {
                        format!("{}.0", numerator.abs())
                    } else {
                        format!("(/ {}.0 {denominator}.0)", numerator.abs())
                    };
                    if numerator.is_negative() {
                        write!(f, "(- {magnitude})")?;
                    } else {
                        f.write_str(&magnitude)?;
                    }
                }
                Term::Apply(operator, arguments) => {
                    write!(f, "({}", operator.symbol())?;
                    pending.push(Step::Text(")"));
                    for &argument in arguments.iter().rev() {
                        pending.push(Step::Term(argument));
                        pending.push(Step::Text(" "));
                    }
                }
            }
        }
        Ok(())
    }
}
