//! Helpers of the mutation and one-entry-building stories: a variable that
//! counts the search-domain hook, a table space shaped like MOGA-VM's
//! candidate table, and a description of a configuration that survives
//! rebuilding the space.

use std::borrow::Cow;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use fhy_core::foreign::{BoxError, ForeignPart, Part};
use fhy_core::identifier::Identifier;
use fhy_core::param::Param;
use fhy_core::search_space::{Configuration, Space, StepDomain, Variable};

use super::search::bounded_param;
use super::search_space::{
    bare_alternative, choice_of, int_param, int_variable, plain_alternative, plain_variable,
    space_of,
};

/// A variable that counts each ask of its search-domain hook and offers no
/// domain, so the step's domain derives from its param.
#[derive(Debug)]
pub(crate) struct CountingVariable {
    name: Identifier,
    param: Param,
    asks: Arc<AtomicUsize>,
}

impl CountingVariable {
    /// Return the variable `name` over the integer categories `values`,
    /// counting its asks in `asks`.
    pub(crate) fn part(
        name: &Identifier,
        values: &[i64],
        asks: &Arc<AtomicUsize>,
    ) -> Part<dyn Variable> {
        Part::new(Self {
            name: name.clone(),
            param: int_param(values),
            asks: Arc::clone(asks),
        })
    }
}

impl ForeignPart for CountingVariable {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed("CountingVariable")
    }
}

impl Variable for CountingVariable {
    fn kind(&self) -> Cow<'_, str> {
        Cow::Borrowed("test.counting_variable")
    }

    fn name(&self) -> &Identifier {
        &self.name
    }

    fn param(&self) -> &Param {
        &self.param
    }

    fn search_domain(&self) -> Result<Option<StepDomain>, BoxError> {
        self.asks.fetch_add(1, Ordering::SeqCst);
        Ok(None)
    }
}

/// Return the space of `choices` top-level choices, each between two
/// alternatives holding one counting variable over `{1, 2}`.
pub(crate) fn build_counting_space(choices: usize, asks: &Arc<AtomicUsize>) -> Space {
    let choices = (0..choices)
        .map(|_| {
            let alternatives = (0..2)
                .map(|_| {
                    plain_alternative(
                        &Identifier::new("option"),
                        vec![CountingVariable::part(
                            &Identifier::new("knob"),
                            &[1, 2],
                            asks,
                        )],
                        Vec::new(),
                    )
                })
                .collect();
            choice_of(&Identifier::new("entry"), alternatives)
        })
        .collect();
    space_of(&Identifier::new("counting"), Vec::new(), choices)
}

/// Return the space of `entries` top-level choices, each among four
/// alternatives holding three variables: an integer in `[1, 8]` and two
/// categoricals, as MOGA-VM's candidate table holds a realization's knobs.
pub(crate) fn build_table_space(entries: usize) -> Space {
    let choices = (0..entries)
        .map(|entry| {
            let alternatives = (0..4)
                .map(|option| {
                    plain_alternative(
                        &Identifier::new(&format!("option{option}")),
                        vec![
                            plain_variable(&Identifier::new("size"), bounded_param(1, 8)),
                            int_variable(&Identifier::new("order"), &[1, 2, 3]),
                            int_variable(&Identifier::new("flag"), &[1, 2]),
                        ],
                        Vec::new(),
                    )
                })
                .collect();
            choice_of(&Identifier::new(&format!("entry{entry}")), alternatives)
        })
        .collect();
    space_of(&Identifier::new("table"), Vec::new(), choices)
}

/// Return the space of `count` top-level variables over `{1, 2}`.
pub(crate) fn build_flat_space(count: usize) -> Space {
    let variables = (0..count)
        .map(|_| int_variable(&Identifier::new("flag"), &[1, 2]))
        .collect();
    space_of(&Identifier::new("flat"), variables, Vec::new())
}

/// Return the space of a choice between `empty`, holding nothing, and
/// `eleven`, holding eleven two-valued variables and then one whose bounds
/// enclose no integer: the choice has two alternatives and only the first
/// has a completion.
pub(crate) fn build_unreachable_alternative_space() -> (Space, Identifier, Identifier) {
    let [choice, empty, eleven] = ["choice", "empty", "eleven"].map(Identifier::new);
    let mut variables: Vec<Part<dyn Variable>> = (0..11)
        .map(|_| int_variable(&Identifier::new("flag"), &[1, 2]))
        .collect();
    variables.push(plain_variable(
        &Identifier::new("none"),
        bounded_param(5, 3),
    ));
    let space = space_of(
        &Identifier::new("unreachable"),
        Vec::new(),
        vec![choice_of(
            &choice,
            vec![
                bare_alternative(&empty),
                plain_alternative(&eleven, variables, Vec::new()),
            ],
        )],
    );
    (space, choice, empty)
}

/// Return `configuration`'s entries as `name=value` texts, in canonical
/// order, which do not depend on the identifiers' ids.
pub(crate) fn describe_entries(configuration: &Configuration) -> Vec<String> {
    configuration
        .entries()
        .map(|(name, value)| format!("{}={value}", name.name_hint()))
        .collect()
}
