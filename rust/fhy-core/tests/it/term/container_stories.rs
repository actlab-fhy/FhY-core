//! Tests for the `AlphaEquivalence` impls of `Option`, slices, `Vec`, arrays,
//! `Box`, `Rc`, `Arc` and tuples, and for a downstream-style type that
//! implements the trait next to them.
//!
//! The impls compare their elements in order under the one renaming they
//! are given. A container binds nothing, so no element extends the
//! renaming for the next: what an element must respect is the renaming the
//! caller passes in.

use std::cell::RefCell;
use std::collections::HashMap;
use std::convert::Infallible;
use std::fmt;
use std::rc::Rc;
use std::sync::{Arc, LazyLock};

use fhy_core::identifier::Identifier;
use fhy_core::term::{AlphaEquivalence, AlphaRenaming};
use proptest::prelude::*;

use crate::support::lambda::{Alpha, build_identifiers, lam, var};

/// A reference to an identifier: a term that corresponds to another when
/// their identifiers do.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Reference(Identifier);

impl AlphaEquivalence for Reference {
    type Error = Infallible;

    fn is_alpha_equivalent_under(
        &self,
        other: &Self,
        renaming: &AlphaRenaming,
    ) -> Result<bool, Infallible> {
        Ok(renaming.is_corresponding(&self.0, &other.0))
    }
}

/// A downstream type with several term fields, compared through the
/// container impls with no hand-written chain of its own.
#[derive(Debug)]
struct Access {
    base: Reference,
    index: Option<Reference>,
    strides: Vec<Reference>,
}

impl AlphaEquivalence for Access {
    type Error = Infallible;

    fn is_alpha_equivalent_under(
        &self,
        other: &Self,
        renaming: &AlphaRenaming,
    ) -> Result<bool, Infallible> {
        Ok(self.base.is_alpha_equivalent_under(&other.base, renaming)?
            && self
                .index
                .is_alpha_equivalent_under(&other.index, renaming)?
            && self
                .strides
                .is_alpha_equivalent_under(&other.strides, renaming)?)
    }
}

/// An error a failing comparison reports.
#[derive(Debug, PartialEq, Eq)]
struct Failure(&'static str);

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "comparison failed: {}", self.0)
    }
}

impl std::error::Error for Failure {}

/// A term that is equal by payload, fails when its payload is negative, and
/// records the payloads it compares.
struct Probe<'a> {
    payload: i64,
    log: &'a RefCell<Vec<i64>>,
}

impl AlphaEquivalence for Probe<'_> {
    type Error = Failure;

    fn is_alpha_equivalent_under(
        &self,
        other: &Self,
        _renaming: &AlphaRenaming,
    ) -> Result<bool, Failure> {
        self.log.borrow_mut().push(self.payload);
        if self.payload < 0 {
            return Err(Failure("negative payload"));
        }
        Ok(self.payload == other.payload)
    }
}

fn reference(identifier: &Identifier) -> Reference {
    Reference(identifier.clone())
}

fn references<const N: usize>(identifiers: [&Identifier; N]) -> Vec<Reference> {
    identifiers.into_iter().map(reference).collect()
}

fn renaming_of(pairs: &[(&Identifier, &Identifier)]) -> AlphaRenaming {
    AlphaRenaming::new(
        pairs
            .iter()
            .map(|(left, right)| ((*left).clone(), (*right).clone()))
            .collect::<HashMap<_, _>>(),
    )
    .expect("the renaming is injective")
}

#[test]
fn an_option_matches_none_with_none_and_some_with_equivalent_contents() {
    let [a, b, c] = build_identifiers(["a", "b", "c"]);
    let renaming = renaming_of(&[(&a, &b)]);

    assert!(None::<Reference>.alpha_equivalent(&None));
    assert!(Some(reference(&a)).alpha_equivalent(&Some(reference(&a))));
    assert!(Some(reference(&a)).alpha_equivalent_under(&Some(reference(&b)), &renaming));
    assert!(!Some(reference(&a)).alpha_equivalent(&Some(reference(&c))));
    assert!(!Some(reference(&a)).alpha_equivalent(&None));
    assert!(!None.alpha_equivalent(&Some(reference(&a))));
}

#[test]
fn a_vec_matches_elements_in_order_and_by_length() {
    let [a, b, c] = build_identifiers(["a", "b", "c"]);

    assert!(Vec::<Reference>::new().alpha_equivalent(&Vec::new()));
    assert!(references([&a, &b]).alpha_equivalent(&references([&a, &b])));
    assert!(!references([&a, &b]).alpha_equivalent(&references([&b, &a])));
    assert!(!references([&a, &b]).alpha_equivalent(&references([&a, &c])));
    assert!(!references([&a, &b]).alpha_equivalent(&references([&a])));
    assert!(!references([&a]).alpha_equivalent(&references([&a, &b])));
    assert!(!references([&a]).alpha_equivalent(&Vec::new()));
}

#[test]
fn a_slice_and_an_array_compare_as_a_vec_does() {
    let [a, b, c] = build_identifiers(["a", "b", "c"]);
    let left = references([&a, &b]);

    assert!(left[..].alpha_equivalent(&references([&a, &b])[..]));
    assert!(!left[..].alpha_equivalent(&references([&a, &c])[..]));
    assert!(!left[..].alpha_equivalent(&references([&a])[..]));
    assert!(!left[..1].alpha_equivalent(&left[..]));

    assert!([reference(&a), reference(&b)].alpha_equivalent(&[reference(&a), reference(&b)]));
    assert!(![reference(&a), reference(&b)].alpha_equivalent(&[reference(&a), reference(&c)]));
    let empty: [Reference; 0] = [];
    assert!(empty.alpha_equivalent(&[]));
}

#[test]
fn a_renaming_given_to_a_container_binds_every_element() {
    let [a, b, c, input, output] = build_identifiers(["a", "b", "c", "input", "output"]);
    let renaming = renaming_of(&[(&a, &b), (&input, &output)]);
    let left = references([&a, &input, &a]);

    assert!(left.alpha_equivalent_under(&references([&b, &output, &b]), &renaming));
    // The last element must respect what the first established: `a` is `b`.
    assert!(!left.alpha_equivalent_under(&references([&b, &output, &c]), &renaming));
    assert!(
        !left.alpha_equivalent_under(&references([&b, &output, &b]), &AlphaRenaming::default())
    );
    assert!(Some(left).alpha_equivalent_under(&Some(references([&b, &output, &b])), &renaming));
}

#[test]
fn a_container_of_binders_compares_each_under_the_renaming_it_is_given() {
    let [input, output, free] = build_identifiers(["input", "output", "free"]);
    let renaming = renaming_of(&[(&free, &free)]);
    let left = vec![lam([&input], var(&input)), lam([&input], var(&free))];

    assert!(left.alpha_equivalent_under(
        &vec![lam([&output], var(&output)), lam([&output], var(&free))],
        &renaming
    ));
    assert!(!left.alpha_equivalent_under(
        &vec![lam([&output], var(&output)), lam([&output], var(&output))],
        &renaming
    ));
}

#[test]
fn a_pointer_compares_the_pointee_and_never_the_pointer() {
    let [a, b] = build_identifiers(["a", "b"]);
    let renaming = renaming_of(&[(&a, &b)]);

    assert!(Box::new(reference(&a)).alpha_equivalent_under(&Box::new(reference(&b)), &renaming));
    assert!(!Box::new(reference(&a)).alpha_equivalent(&Box::new(reference(&b))));
    assert!(Rc::new(reference(&a)).alpha_equivalent_under(&Rc::new(reference(&b)), &renaming));
    assert!(!Rc::new(reference(&a)).alpha_equivalent(&Rc::new(reference(&b))));
    assert!(Arc::new(reference(&a)).alpha_equivalent_under(&Arc::new(reference(&b)), &renaming));
    assert!(!Arc::new(reference(&a)).alpha_equivalent(&Arc::new(reference(&b))));

    let shared = Arc::new(reference(&a));
    assert!(shared.alpha_equivalent(&Arc::clone(&shared)));
    // A shared allocation is compared under the renaming as any other: `a`
    // does not correspond to itself once the renaming sends it to `b`.
    assert!(!shared.alpha_equivalent_under(&Arc::clone(&shared), &renaming));

    let left: Box<[Reference]> = references([&a, &b]).into();
    let right: Box<[Reference]> = references([&a, &b]).into();
    assert!(left.alpha_equivalent(&right));
    assert!(!left.alpha_equivalent(&references([&a]).into()));
}

#[test]
fn a_tuple_matches_every_element_in_order() {
    let [a, b, c] = build_identifiers(["a", "b", "c"]);
    let renaming = renaming_of(&[(&a, &b)]);

    assert!((reference(&a),).alpha_equivalent(&(reference(&a),)));
    assert!(!(reference(&a),).alpha_equivalent(&(reference(&b),)));
    assert!(
        (reference(&a), reference(&c))
            .alpha_equivalent_under(&(reference(&b), reference(&c)), &renaming)
    );
    // The second element must respect the renaming too.
    assert!(
        !(reference(&a), reference(&a))
            .alpha_equivalent_under(&(reference(&b), reference(&c)), &renaming)
    );
    assert!(!(reference(&a), reference(&c)).alpha_equivalent(&(reference(&a), reference(&b))));

    let eight = |identifier: &Identifier| {
        let reference = reference(identifier);
        (
            reference.clone(),
            reference.clone(),
            reference.clone(),
            reference.clone(),
            reference.clone(),
            reference.clone(),
            reference.clone(),
            reference,
        )
    };
    assert!(eight(&a).alpha_equivalent(&eight(&a)));
    assert!(!eight(&a).alpha_equivalent(&eight(&c)));
    let mut last_differs = eight(&a);
    last_differs.7 = reference(&c);
    assert!(!eight(&a).alpha_equivalent(&last_differs));
}

#[test]
fn a_nested_container_compares_through_every_level() {
    let [a, b, c] = build_identifiers(["a", "b", "c"]);
    let nested = |first: &Identifier, last: &Identifier| {
        vec![
            (Some(Box::new(reference(first))), references([first, last])),
            (None, Vec::new()),
        ]
    };

    assert!(nested(&a, &b).alpha_equivalent(&nested(&a, &b)));
    assert!(!nested(&a, &b).alpha_equivalent(&nested(&a, &c)));
    assert!(!nested(&a, &b).alpha_equivalent(&nested(&a, &b)[..1].to_vec()));
}

#[test]
fn a_downstream_type_compares_its_fields_through_the_impls() {
    let [a, b, c, input, output] = build_identifiers(["a", "b", "c", "input", "output"]);
    let renaming = renaming_of(&[(&a, &b), (&input, &output)]);
    let access = |base: &Identifier, index: Option<&Identifier>, strides: Vec<Reference>| Access {
        base: reference(base),
        index: index.map(reference),
        strides,
    };
    let left = access(&a, Some(&input), references([&input, &a]));

    assert!(left.alpha_equivalent_under(
        &access(&b, Some(&output), references([&output, &b])),
        &renaming
    ));
    assert!(!left.alpha_equivalent_under(&access(&b, None, references([&output, &b])), &renaming));
    assert!(
        !left.alpha_equivalent_under(&access(&b, Some(&output), references([&output])), &renaming)
    );
    assert!(!left.alpha_equivalent_under(
        &access(&b, Some(&output), references([&output, &c])),
        &renaming
    ));
    assert!(left.alpha_equivalent(&access(&a, Some(&input), references([&input, &a]))));
}

#[test]
fn a_container_stops_at_the_first_difference_or_failure() {
    let log = RefCell::new(Vec::new());
    let probes = |payloads: &[i64]| -> Vec<Probe<'_>> {
        payloads
            .iter()
            .map(|&payload| Probe { payload, log: &log })
            .collect()
    };

    assert_eq!(
        probes(&[1, 2, 3]).is_alpha_equivalent(&probes(&[1, 9, 3])),
        Ok(false)
    );
    assert_eq!(*log.borrow(), [1, 2]);

    log.borrow_mut().clear();
    assert_eq!(
        probes(&[1, -2, 3]).is_alpha_equivalent(&probes(&[1, -2, 3])),
        Err(Failure("negative payload"))
    );
    assert_eq!(*log.borrow(), [1, -2]);

    // A length mismatch answers before any element is compared.
    log.borrow_mut().clear();
    assert_eq!(
        probes(&[-1]).is_alpha_equivalent(&probes(&[-1, -1])),
        Ok(false)
    );
    assert!(log.borrow().is_empty());

    // A tuple reports the first element's error type for all of its elements.
    log.borrow_mut().clear();
    let pair = |first, second| {
        (
            Probe {
                payload: first,
                log: &log,
            },
            Probe {
                payload: second,
                log: &log,
            },
        )
    };
    assert_eq!(
        pair(1, -2).is_alpha_equivalent(&pair(1, -2)),
        Err(Failure("negative payload"))
    );
    assert_eq!(pair(1, 2).is_alpha_equivalent(&pair(1, 2)), Ok(true));
    assert_eq!(
        Some(Probe {
            payload: -1,
            log: &log
        })
        .is_alpha_equivalent(&None),
        Ok(false)
    );
}

/// The identifiers the generated elements reference.
static POOL: LazyLock<[Identifier; 3]> = LazyLock::new(|| {
    [
        Identifier::new("q0"),
        Identifier::new("q1"),
        Identifier::new("q2"),
    ]
});

fn build_reference_strategy() -> impl Strategy<Value = Reference> {
    (0..POOL.len()).prop_map(|index| Reference(POOL[index].clone()))
}

/// Return `reference` with `q0` and `q1` exchanged: its counterpart under
/// the renaming that swaps them.
fn swap(reference: &Reference) -> Reference {
    if reference.0 == POOL[0] {
        Reference(POOL[1].clone())
    } else if reference.0 == POOL[1] {
        Reference(POOL[0].clone())
    } else {
        reference.clone()
    }
}

proptest! {
    /// Test a container of references is alpha-equivalent under no
    /// renaming exactly when the containers are equal.
    #[test]
    fn containers_agree_with_equality_under_the_empty_renaming(
        left in proptest::collection::vec(build_reference_strategy(), 0..4),
        right in proptest::collection::vec(build_reference_strategy(), 0..4),
        left_option in proptest::option::of(build_reference_strategy()),
        right_option in proptest::option::of(build_reference_strategy()),
    ) {
        prop_assert_eq!(left.alpha_equivalent(&right), left == right);
        prop_assert_eq!(left[..].alpha_equivalent(&right[..]), left == right);
        prop_assert_eq!(left_option.alpha_equivalent(&right_option), left_option == right_option);
        prop_assert_eq!(
            (left_option.clone(), left.clone()).alpha_equivalent(&(right_option.clone(), right.clone())),
            left_option == right_option && left == right
        );
        prop_assert_eq!(
            Box::new(left.clone()).alpha_equivalent(&Box::new(right.clone())),
            left == right
        );
    }

    /// Test a vector is alpha-equivalent under the renaming that swaps
    /// two identifiers exactly when the other is the first with them
    /// swapped, element by element.
    #[test]
    fn a_vec_agrees_with_the_swapped_elements_under_a_swap(
        left in proptest::collection::vec(build_reference_strategy(), 0..4),
        right in proptest::collection::vec(build_reference_strategy(), 0..4),
    ) {
        let renaming = renaming_of(&[(&POOL[0], &POOL[1]), (&POOL[1], &POOL[0])]);
        let swapped: Vec<Reference> = left.iter().map(swap).collect();

        prop_assert_eq!(left.alpha_equivalent_under(&right, &renaming), swapped == right);
        prop_assert!(left.alpha_equivalent_under(&swapped, &renaming));
    }

    /// Test alpha equivalence of containers is symmetric under a swap.
    #[test]
    fn container_comparison_is_symmetric_under_a_swap(
        left in proptest::collection::vec(build_reference_strategy(), 0..4),
        right in proptest::collection::vec(build_reference_strategy(), 0..4),
    ) {
        let renaming = renaming_of(&[(&POOL[0], &POOL[1]), (&POOL[1], &POOL[0])]);

        prop_assert_eq!(
            left.alpha_equivalent_under(&right, &renaming),
            right.alpha_equivalent_under(&left, &renaming)
        );
    }
}
