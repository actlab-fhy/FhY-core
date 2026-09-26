//! Tests for `PartiallyOrderedSet`: membership, orders and their
//! refusals, reachability, and the two iteration orders.
//!
//! Ported from `tests/test_poset.py`; the traceability table is in
//! `docs/design/python-switch.md`, "S11a.2 implementation notes".

use crate::support::stack::{SMALL_STACK_DEPTH, run_on_small_stack};

use fhy_core::lattice::{OrderError, PartiallyOrderedSet};

/// Return the poset of `elements` with the orders `orders`, added in order.
fn build_poset<T: Eq + std::hash::Hash + Clone + std::fmt::Debug>(
    elements: &[T],
    orders: &[(T, T)],
) -> PartiallyOrderedSet<T> {
    let mut poset = PartiallyOrderedSet::new();
    for element in elements {
        poset
            .add_element(element.clone())
            .expect("each element is new");
    }
    for (lower, upper) in orders {
        poset
            .add_order(lower, upper)
            .expect("each order is acyclic");
    }
    poset
}

#[test]
fn empty_poset_has_no_elements() {
    let poset: PartiallyOrderedSet<i32> = PartiallyOrderedSet::new();

    assert_eq!(poset.len(), 0);
    assert!(poset.is_empty());
    assert!(!poset.contains(&1));
    assert_eq!(poset.iter().count(), 0);
}

#[test]
fn added_elements_are_members() {
    let poset = build_poset(&[1, 2], &[(1, 2)]);

    assert_eq!(poset.len(), 2);
    assert!(poset.contains(&1));
    assert!(poset.contains(&2));
    assert!(!poset.contains(&3));
}

#[test]
fn adding_a_member_again_is_refused_and_leaves_the_poset_unchanged() {
    let mut poset = build_poset(&[1], &[]);

    assert_eq!(poset.add_element(1), Err(OrderError::AlreadyAMember(1)));
    assert_eq!(poset.len(), 1);
}

#[test]
fn order_holds_directly_transitively_and_reflexively() {
    let poset = build_poset(&[1, 2, 3], &[(1, 2), (2, 3)]);

    assert_eq!(poset.is_at_most(&1, &2), Ok(true));
    assert_eq!(poset.is_at_most(&1, &3), Ok(true));
    assert_eq!(poset.is_at_most(&2, &2), Ok(true));
    assert_eq!(poset.is_at_most(&3, &1), Ok(false));
}

#[test]
fn unrelated_elements_are_not_ordered_either_way() {
    let poset = build_poset(&[1, 2], &[]);

    assert_eq!(poset.is_at_most(&1, &2), Ok(false));
    assert_eq!(poset.is_at_most(&2, &1), Ok(false));
}

#[test]
fn asking_about_a_non_member_is_refused_lower_first() {
    let poset = build_poset(&[1, 2], &[(1, 2)]);

    assert_eq!(poset.is_at_most(&3, &2), Err(OrderError::NotAMember(3)));
    assert_eq!(poset.is_at_most(&1, &3), Err(OrderError::NotAMember(3)));
    assert_eq!(poset.is_at_most(&4, &3), Err(OrderError::NotAMember(4)));
}

#[test]
fn ordering_a_non_member_is_refused_lower_first() {
    let mut poset = build_poset(&[1, 2], &[(1, 2)]);

    assert_eq!(poset.add_order(&3, &2), Err(OrderError::NotAMember(3)));
    assert_eq!(poset.add_order(&1, &3), Err(OrderError::NotAMember(3)));
}

#[test]
fn reversing_an_order_is_refused_as_a_cycle_and_leaves_the_poset_unchanged() {
    let mut poset = build_poset(&[1, 2, 3], &[(1, 2), (2, 3)]);

    assert_eq!(
        poset.add_order(&3, &1),
        Err(OrderError::WouldCycle { lower: 3, upper: 1 })
    );
    assert_eq!(poset.is_at_most(&3, &1), Ok(false));
    assert_eq!(poset.is_at_most(&1, &3), Ok(true));
}

#[test]
fn ordering_an_element_below_itself_is_refused_as_a_cycle() {
    let mut poset = build_poset(&[1], &[]);

    assert_eq!(
        poset.add_order(&1, &1),
        Err(OrderError::WouldCycle { lower: 1, upper: 1 })
    );
}

#[test]
fn adding_an_order_that_already_holds_is_accepted() {
    let mut poset = build_poset(&[1, 2, 3], &[(1, 2), (2, 3)]);

    assert_eq!(poset.add_order(&1, &2), Ok(()));
    assert_eq!(poset.add_order(&1, &3), Ok(()));
    assert_eq!(poset.iter().copied().collect::<Vec<_>>(), [1, 2, 3]);
}

#[test]
fn a_later_order_extends_the_up_sets_of_everything_below() {
    let poset = build_poset(&[1, 2, 3, 4], &[(1, 2), (3, 4), (2, 3)]);

    assert_eq!(poset.is_at_most(&1, &4), Ok(true));
}

#[test]
fn iteration_is_topological_with_insertion_order_breaking_ties() {
    let poset = build_poset(&["a", "b", "c", "d"], &[("c", "a"), ("d", "b")]);

    assert_eq!(
        poset.iter().copied().collect::<Vec<_>>(),
        ["c", "a", "d", "b"]
    );
}

#[test]
fn iteration_repeats_the_same_order() {
    let poset = build_poset(&[1, 2], &[(1, 2)]);

    let first: Vec<i32> = poset.iter().copied().collect();
    let second: Vec<i32> = poset.iter().copied().collect();

    assert_eq!(first, [1, 2]);
    assert_eq!(first, second);
}

#[test]
fn iteration_by_key_breaks_ties_by_the_least_key() {
    let poset = build_poset(&["b", "a", "c"], &[]);

    assert_eq!(
        poset
            .iter_by_key(|element| *element)
            .copied()
            .collect::<Vec<_>>(),
        ["a", "b", "c"]
    );
}

#[test]
fn iteration_by_key_stays_topological() {
    let poset = build_poset(&["a", "b", "c"], &[("c", "a")]);

    let order: Vec<&str> = poset.iter_by_key(|element| *element).copied().collect();

    assert_eq!(order, ["b", "c", "a"]);
}

#[test]
fn iteration_by_key_takes_any_ordered_key() {
    let poset = build_poset(&["apple", "banana", "cherry"], &[]);

    let order: Vec<&str> = poset
        .iter_by_key(|element| std::cmp::Reverse(element.as_bytes()[0]))
        .copied()
        .collect();

    assert_eq!(order, ["cherry", "banana", "apple"]);
}

#[test]
fn iteration_by_key_breaks_equal_keys_by_insertion_order() {
    let poset = build_poset(&[3, 1, 2], &[]);

    assert_eq!(
        poset.iter_by_key(|_| 0).copied().collect::<Vec<_>>(),
        [3, 1, 2]
    );
}

#[test]
fn iteration_reports_its_exact_length() {
    let poset = build_poset(&[1, 2, 3], &[(1, 3)]);

    assert_eq!(poset.iter().len(), 3);
    assert_eq!(poset.iter_by_key(|element| *element).len(), 3);
}

#[test]
fn errors_display_one_lowercase_line() {
    assert_eq!(
        OrderError::AlreadyAMember(1).to_string(),
        "1 is already a member of the partially ordered set"
    );
    assert_eq!(
        OrderError::NotAMember(3).to_string(),
        "3 is not a member of the partially ordered set"
    );
    assert_eq!(
        OrderError::WouldCycle { lower: 2, upper: 1 }.to_string(),
        "ordering 2 below 1 would close a cycle"
    );
}

#[test]
fn long_chain_orders_and_iterates_on_a_small_stack() {
    run_on_small_stack(|| {
        let length = SMALL_STACK_DEPTH / 50;
        let mut poset = PartiallyOrderedSet::new();
        for element in 0..length {
            poset.add_element(element).expect("each element is new");
        }
        for element in 1..length {
            poset
                .add_order(&(element - 1), &element)
                .expect("the chain is acyclic");
        }
        assert_eq!(poset.is_at_most(&0, &(length - 1)), Ok(true));
        assert!(poset.iter().copied().eq(0..length));
    });
}
