//! The stories of the frozen refusals, run in an interpreter this test binary
//! embeds, against the stand-in of `FrozenMutationError`.

use crate::kit::exceptions::FROZEN_MUTATION_ERROR;
use crate::kit::testing::{evaluate, with_framework};

use super::*;

#[test]
fn an_assignment_is_refused_with_the_mixins_message() {
    with_framework(|py| {
        let object = evaluate(py, "[]");

        let error = refuse_attribute_assignment(&object, "size").expect_err("always refuses");

        assert!(FROZEN_MUTATION_ERROR.is_instance_of(py, &error));
        assert_eq!(
            error.value(py).to_string(),
            "Cannot modify \"size\" on frozen list."
        );
    });
}

#[test]
fn a_deletion_is_refused_with_the_mixins_message() {
    with_framework(|py| {
        let object = evaluate(py, "type('Point', (), {})()");

        let error = refuse_attribute_deletion(&object, "x").expect_err("always refuses");

        assert!(FROZEN_MUTATION_ERROR.is_instance_of(py, &error));
        assert_eq!(
            error.value(py).to_string(),
            "Cannot delete \"x\" on frozen Point."
        );
    });
}
