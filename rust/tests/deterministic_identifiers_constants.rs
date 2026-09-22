//! A deterministic-identifier scope never aliases a shipped constant.
//!
//! The shipped constants are created on first use. This binary holds a single
//! test so that the scope below is what first creates them. Do not add a
//! second test: one that touched a constant first would create it outside any
//! scope, and this check would pass without exercising anything.

use fhy_core::identifier::Identifier;
use fhy_core::interned::Interned;
use fhy_core::op_attribute::{OpAttribute, COMMUTATIVE};
use fhy_core::testing::DeterministicIdentifierScope;
use fhy_core::value_domain::{ValueDomain, DATA_DOMAIN};

/// Test a scope that first creates the shipped constants does not give a
/// same-hint identifier their ids, so no registry mistakes it for a constant.
#[test]
fn a_scope_that_first_creates_the_shipped_constants_does_not_alias_them() {
    let _scope = DeterministicIdentifierScope::enter();
    let data = DATA_DOMAIN.clone();
    let commutative = COMMUTATIVE.clone();

    let data_lookalike = Identifier::new("data");
    let commutative_lookalike = Identifier::new("commutative");

    assert_ne!(&data_lookalike, data.name());
    assert_ne!(&commutative_lookalike, commutative.name());
    assert_eq!(ValueDomain::intern_registry().get(&data_lookalike), None);
    assert_eq!(
        OpAttribute::intern_registry().get(&commutative_lookalike),
        None
    );
}
