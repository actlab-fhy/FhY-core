//! Tests for `is_mapping_alpha_equivalent_under`: sizes, keys resolved
//! through the renaming, collisions, capture, and the values compared in
//! the left map's order.
//!
//! The cases are ported from the mapping-helper tests of
//! `tests/test_alpha_equivalence.py`.

use std::cell::RefCell;
use std::collections::HashMap;

use fhy_core::identifier::Identifier;
use fhy_core::term::{AlphaEquivalence, AlphaRenaming, is_mapping_alpha_equivalent_under};

fn build_identifiers<const N: usize>(names: [&str; N]) -> [Identifier; N] {
    names.map(Identifier::new)
}

/// A value compared by payload that records each comparison it makes, and
/// optionally checks that `reference` corresponds to its counterpart.
#[derive(Debug)]
struct Leaf<'a> {
    payload: i64,
    reference: Option<Identifier>,
    log: Option<&'a RefCell<Vec<i64>>>,
}

fn leaf(payload: i64) -> Leaf<'static> {
    Leaf {
        payload,
        reference: None,
        log: None,
    }
}

impl AlphaEquivalence for Leaf<'_> {
    fn is_alpha_equivalent_under(&self, other: &Self, renaming: &AlphaRenaming) -> bool {
        if let Some(log) = self.log {
            log.borrow_mut().push(self.payload);
        }
        self.payload == other.payload
            && match (&self.reference, &other.reference) {
                (Some(left), Some(right)) => renaming.is_corresponding(left, right),
                (None, None) => true,
                _ => false,
            }
    }
}

fn compare<V: AlphaEquivalence>(
    left: &[(Identifier, V)],
    right: &HashMap<Identifier, V>,
    renaming: &AlphaRenaming,
) -> bool {
    is_mapping_alpha_equivalent_under(
        left.iter().map(|(key, value)| (key, value)),
        right,
        renaming,
    )
}

#[test]
fn empty_maps_are_alpha_equivalent() {
    let right: HashMap<Identifier, Leaf<'_>> = HashMap::new();

    assert!(compare(&[], &right, &AlphaRenaming::default()));
}

#[test]
fn a_map_is_alpha_equivalent_to_its_copy() {
    let [x, y] = build_identifiers(["x", "y"]);

    let left = [(x.clone(), leaf(1)), (y.clone(), leaf(2))];
    let right = HashMap::from([(x, leaf(1)), (y, leaf(2))]);

    assert!(compare(&left, &right, &AlphaRenaming::default()));
}

#[test]
fn maps_of_different_sizes_are_not_alpha_equivalent() {
    let [x, y] = build_identifiers(["x", "y"]);

    let left = [(x.clone(), leaf(1))];
    let right = HashMap::from([(x, leaf(1)), (y, leaf(2))]);

    assert!(!compare(&left, &right, &AlphaRenaming::default()));
}

#[test]
fn maps_whose_values_differ_are_not_alpha_equivalent() {
    let [x] = build_identifiers(["x"]);

    let left = [(x.clone(), leaf(1))];
    let right = HashMap::from([(x, leaf(2))]);

    assert!(!compare(&left, &right, &AlphaRenaming::default()));
}

#[test]
fn keys_resolve_through_a_binder_frame_and_the_free_renaming() {
    let [x, x_prime, a, b] = build_identifiers(["x", "x_prime", "a", "b"]);
    let mut framed = AlphaRenaming::default();
    framed
        .enter_binders(std::slice::from_ref(&x), std::slice::from_ref(&x_prime))
        .expect("one pair");
    let free = AlphaRenaming::try_new(HashMap::from([(a.clone(), b.clone())])).expect("one pair");

    assert!(compare(
        &[(x.clone(), leaf(1))],
        &HashMap::from([(x_prime.clone(), leaf(1))]),
        &framed
    ));
    assert!(compare(
        &[(a.clone(), leaf(1))],
        &HashMap::from([(b.clone(), leaf(1))]),
        &free
    ));
    assert!(!compare(
        &[(x, leaf(1))],
        &HashMap::from([(x_prime, leaf(1))]),
        &AlphaRenaming::default()
    ));
}

#[test]
fn maps_whose_resolved_keys_differ_are_not_alpha_equivalent() {
    let [x, y, z, w] = build_identifiers(["x", "y", "z", "w"]);

    let left = [(x, leaf(1)), (y, leaf(2))];
    let right = HashMap::from([(w, leaf(1)), (z, leaf(2))]);

    assert!(!compare(&left, &right, &AlphaRenaming::default()));
}

#[test]
fn two_keys_resolving_to_one_are_not_alpha_equivalent() {
    let [x, y, target] = build_identifiers(["x", "y", "target"]);
    let renaming =
        AlphaRenaming::try_new(HashMap::from([(x.clone(), target.clone())])).expect("one pair");

    // `x` resolves to `target`, and `target` is itself a key on the left.
    let left = [(x, leaf(1)), (target.clone(), leaf(2))];
    let right = HashMap::from([(target, leaf(1)), (y, leaf(2))]);

    assert!(!compare(&left, &right, &renaming));
}

#[test]
fn a_key_captured_by_a_frame_on_the_other_side_is_not_alpha_equivalent() {
    let [x, y] = build_identifiers(["x", "y"]);
    let mut renaming = AlphaRenaming::default();
    renaming
        .enter_binders(std::slice::from_ref(&x), std::slice::from_ref(&y))
        .expect("one pair");

    // The free `y` on the left resolves to itself, but `y` is bound on the
    // right, so the keys do not correspond.
    assert!(!compare(
        &[(y.clone(), leaf(1))],
        &HashMap::from([(y, leaf(1))]),
        &renaming
    ));
}

#[test]
fn the_values_are_compared_under_the_renaming() {
    let [k, x, y] = build_identifiers(["k", "x", "y"]);
    let mut renaming = AlphaRenaming::default();
    renaming
        .enter_binders(std::slice::from_ref(&x), std::slice::from_ref(&y))
        .expect("one pair");

    let left = [(
        k.clone(),
        Leaf {
            payload: 1,
            reference: Some(x.clone()),
            log: None,
        },
    )];
    let right = HashMap::from([(
        k.clone(),
        Leaf {
            payload: 1,
            reference: Some(y),
            log: None,
        },
    )]);
    let unrenamed = HashMap::from([(
        k,
        Leaf {
            payload: 1,
            reference: Some(x),
            log: None,
        },
    )]);

    assert!(compare(&left, &right, &renaming));
    assert!(!compare(&left, &unrenamed, &renaming));
}

#[test]
fn the_values_are_compared_in_the_left_order_stopping_at_the_first_difference() {
    let [a, b, c] = build_identifiers(["a", "b", "c"]);
    let log = RefCell::new(Vec::new());
    let logged = |payload| Leaf {
        payload,
        reference: None,
        log: Some(&log),
    };

    let left = [
        (c.clone(), logged(3)),
        (a.clone(), logged(1)),
        (b.clone(), logged(2)),
    ];
    let right = HashMap::from([(a, leaf(1)), (b, leaf(99)), (c, leaf(3))]);

    assert!(!compare(&left, &right, &AlphaRenaming::default()));
    assert_eq!(*log.borrow(), vec![3, 1, 2]);
}

#[test]
fn no_value_is_compared_when_the_keys_differ() {
    let [a, b] = build_identifiers(["a", "b"]);
    let log = RefCell::new(Vec::new());

    let left = [(
        a,
        Leaf {
            payload: 1,
            reference: None,
            log: Some(&log),
        },
    )];
    let right = HashMap::from([(b, leaf(1))]);

    assert!(!compare(&left, &right, &AlphaRenaming::default()));
    assert!(log.borrow().is_empty());
}
