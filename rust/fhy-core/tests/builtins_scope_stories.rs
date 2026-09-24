//! A deterministic-identifier scope never aliases a built-in parameter.
//!
//! The built-in catalogue creates its parameters on first use. This binary
//! holds a single test so that the scope below is what first creates them.
//! Do not add a second test: one that touched the catalogue first would
//! create the parameters outside any scope, and this check would pass
//! without exercising anything.

use std::collections::HashSet;

use fhy_core::expr::builtins::{ComposedFunction, list_composed_functions};
use fhy_core::identifier::Identifier;
use fhy_core::testing::DeterministicIdentifierScope;

/// Test a scope that first creates the catalogue neither merges same-named
/// parameters of different functions nor hands their ids to same-named
/// identifiers created in the scope.
#[test]
fn composed_functions_first_used_in_a_scope_keeps_its_parameters_unique() {
    let _scope = DeterministicIdentifierScope::enter();
    let parameters: Vec<Identifier> = list_composed_functions()
        .iter()
        .flat_map(ComposedFunction::parameters)
        .cloned()
        .collect();

    let lookalikes: Vec<Identifier> = ["a", "b", "x", "lo", "hi", "bound", "slope"]
        .into_iter()
        .map(Identifier::new)
        .collect();

    let distinct_ids: HashSet<u64> = parameters.iter().map(Identifier::id).collect();
    assert_eq!(
        distinct_ids.len(),
        parameters.len(),
        "a parameter id repeats"
    );
    for lookalike in &lookalikes {
        assert!(
            !parameters.contains(lookalike),
            "{lookalike:?} aliases a built-in parameter"
        );
    }
}
