//! Property tests for `fhy_core::pass::VerificationRegistry`: lookups along
//! a lineage agree with a reference model of the walk, and verification
//! runs one validator per id found.
//!
//! Nothing here reads process-global state.

use crate::support::pass_ir;

use std::borrow::Cow;

use fhy_core::foreign::BoxError;
use fhy_core::pass::{PassContext, Validator, VerificationRegistry, VerifierId};
use pass_ir::BoxIr;
use proptest::prelude::*;

/// The number of distinct kinds the registrations draw from.
const KIND_COUNT: u8 = 4;

/// The number of distinct validators the registrations draw from.
const VALIDATOR_COUNT: usize = 6;

/// A validator named after its index into the validator objects.
struct Indexed(usize);

impl Validator<BoxIr> for Indexed {
    fn name(&self) -> Cow<'static, str> {
        Cow::Owned(self.0.to_string())
    }

    fn validate(&mut self, _ir: &BoxIr, _cx: &mut PassContext<'_>) -> Result<(), BoxError> {
        Ok(())
    }
}

/// Return what the walk finds for `lineage` over `registrations`, each a
/// `(kind, validator)` pair in registration order: each kind's validators
/// in lineage order and then registration order, each validator once, at
/// its first position, with repeated registrations of a pair ignored.
fn model_lookup(registrations: &[(u8, usize)], lineage: &[u8]) -> Vec<usize> {
    let mut found = Vec::new();
    for kind in lineage {
        for (registered_kind, validator) in registrations {
            if registered_kind == kind && !found.contains(validator) {
                found.push(*validator);
            }
        }
    }
    found
}

proptest! {
    #[test]
    fn lookups_agree_with_the_reference_walk(
        registrations in prop::collection::vec((0..KIND_COUNT, 0..VALIDATOR_COUNT), 0..16),
        lineage in prop::collection::vec(0..KIND_COUNT, 0..6),
    ) {
        // One object per validator stands for it, as a binding's class does.
        let objects: Vec<Box<u8>> = (0..VALIDATOR_COUNT).map(|_| Box::new(0)).collect();
        let id_of = |validator: usize| VerifierId::of_ptr(&raw const *objects[validator]);
        let mut registry = VerificationRegistry::new();
        let mut new_pairs = Vec::new();
        for &(kind, validator) in &registrations {
            let is_new = !new_pairs.contains(&(kind, validator));
            let added = registry.register_with_id(kind, id_of(validator), move || Indexed(validator));
            prop_assert_eq!(added, is_new);
            if is_new {
                new_pairs.push((kind, validator));
            }
        }
        let expected = model_lookup(&registrations, &lineage);

        let ids = registry.ids_for(&lineage);
        let report = registry.verify(&lineage, &BoxIr::new(0));

        prop_assert_eq!(registry.len(), new_pairs.len());
        prop_assert_eq!(ids, expected.iter().map(|&validator| id_of(validator)).collect::<Vec<_>>());
        let names: Vec<String> = report
            .records()
            .iter()
            .map(|record| record.validator_name().to_owned())
            .collect();
        prop_assert_eq!(names, expected.iter().map(ToString::to_string).collect::<Vec<_>>());
    }
}
