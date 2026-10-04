//! [`AlphaEquivalence`] for the standard containers of terms.
//!
//! Each impl compares its elements one by one under the one
//! [`AlphaRenaming`] it is given, with no frame of its own: a container
//! binds nothing, so it passes `renaming` on to its elements unchanged. A
//! container whose elements are terms is a term, so a downstream type
//! implements [`AlphaEquivalence`] for its own fields by calling these, not
//! by chaining its fields' comparisons by hand.
//!
//! - [`Option<T>`]: `None` matches only `None`, and `Some` matches `Some`
//!   when the contents do.
//! - [`[T]`](slice), [`Vec<T>`] and [`[T; N]`](array): the lengths match, and
//!   the elements match pairwise, in order.
//! - [`Box<T>`], [`Rc<T>`] and [`Arc<T>`]: the pointee, whether or not the
//!   two pointers are the same allocation. Pointer identity is no shortcut:
//!   a term that binds one identifier twice is alpha-equivalent to none,
//!   itself included.
//! - Tuples of one to eight elements: the elements match pairwise, in order.
//!   A tuple's elements share one error type, that of its first element.
//!
//! There is no impl for `&T`: with it, `value.is_alpha_equivalent_under(..)`
//! on a `&&T` would resolve to the reference's impl and fail to type-check
//! in code written before it. Compare through the references.
//!
//! Every comparison stops at the first element that differs or fails, so
//! an error after that element is never reported.
//!
//! Unordered collections ([`HashSet`](std::collections::HashSet),
//! [`HashMap`](std::collections::HashMap)) and ordered maps have no impl: a
//! set has no order to pair elements by, so the answer would depend on
//! iteration order, and a map's keys may be references or binders, which a
//! blanket impl cannot tell from plain data.
//! [`is_mapping_alpha_equivalent_under`](super::is_mapping_alpha_equivalent_under)
//! compares maps keyed by identifiers.

use std::rc::Rc;
use std::sync::Arc;

use super::binder::AlphaEquivalence;
use super::renaming::AlphaRenaming;

impl<T: AlphaEquivalence> AlphaEquivalence for Option<T> {
    type Error = T::Error;

    /// Compare `None` with `None` as equivalent, `Some` with `Some` by the
    /// contents under `renaming`, and `None` with `Some` as not.
    ///
    /// # Errors
    ///
    /// Returns the contents' error.
    fn is_alpha_equivalent_under(
        &self,
        other: &Self,
        renaming: &AlphaRenaming,
    ) -> Result<bool, T::Error> {
        match (self, other) {
            (None, None) => Ok(true),
            (Some(left), Some(right)) => left.is_alpha_equivalent_under(right, renaming),
            _ => Ok(false),
        }
    }
}

impl<T: AlphaEquivalence> AlphaEquivalence for [T] {
    type Error = T::Error;

    /// Compare the elements pairwise, in order, under `renaming`; slices of
    /// different lengths are not equivalent.
    ///
    /// # Errors
    ///
    /// Returns the error of the first pair that fails to compare, before
    /// any pair that differs after it.
    fn is_alpha_equivalent_under(
        &self,
        other: &Self,
        renaming: &AlphaRenaming,
    ) -> Result<bool, T::Error> {
        if self.len() != other.len() {
            return Ok(false);
        }
        for (left, right) in self.iter().zip(other) {
            if !left.is_alpha_equivalent_under(right, renaming)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

impl<T: AlphaEquivalence> AlphaEquivalence for Vec<T> {
    type Error = T::Error;

    /// Compare as the slices do.
    ///
    /// # Errors
    ///
    /// Returns what the slice comparison returns.
    fn is_alpha_equivalent_under(
        &self,
        other: &Self,
        renaming: &AlphaRenaming,
    ) -> Result<bool, T::Error> {
        self.as_slice()
            .is_alpha_equivalent_under(other.as_slice(), renaming)
    }
}

impl<T: AlphaEquivalence, const N: usize> AlphaEquivalence for [T; N] {
    type Error = T::Error;

    /// Compare as the slices do; the arrays have one length.
    ///
    /// # Errors
    ///
    /// Returns what the slice comparison returns.
    fn is_alpha_equivalent_under(
        &self,
        other: &Self,
        renaming: &AlphaRenaming,
    ) -> Result<bool, T::Error> {
        self.as_slice()
            .is_alpha_equivalent_under(other.as_slice(), renaming)
    }
}

/// Implement [`AlphaEquivalence`] for a smart pointer to a term: compare the
/// pointees.
macro_rules! impl_for_pointer {
    ($($pointer:ident),+) => {$(
        impl<T: AlphaEquivalence + ?Sized> AlphaEquivalence for $pointer<T> {
            type Error = T::Error;

            /// Compare the pointees under `renaming`, never the pointers.
            ///
            /// # Errors
            ///
            /// Returns the pointees' error.
            fn is_alpha_equivalent_under(
                &self,
                other: &Self,
                renaming: &AlphaRenaming,
            ) -> Result<bool, T::Error> {
                (**self).is_alpha_equivalent_under(&**other, renaming)
            }
        }
    )+};
}

impl_for_pointer!(Box, Rc, Arc);

/// Implement [`AlphaEquivalence`] for a tuple: the elements, in order, under
/// the one renaming, with the error of the first.
macro_rules! impl_for_tuple {
    ($first:ident $(, $name:ident : $index:tt)*) => {
        impl<$first: AlphaEquivalence $(, $name)*> AlphaEquivalence for ($first, $($name,)*)
        where
            $($name: AlphaEquivalence<Error = $first::Error>,)*
        {
            type Error = $first::Error;

            /// Compare the elements pairwise, in order, under `renaming`.
            ///
            /// # Errors
            ///
            /// Returns the error of the first pair that fails to compare,
            /// before any pair that differs after it.
            fn is_alpha_equivalent_under(
                &self,
                other: &Self,
                renaming: &AlphaRenaming,
            ) -> Result<bool, $first::Error> {
                if !self.0.is_alpha_equivalent_under(&other.0, renaming)? {
                    return Ok(false);
                }
                $(
                    if !self.$index.is_alpha_equivalent_under(&other.$index, renaming)? {
                        return Ok(false);
                    }
                )*
                Ok(true)
            }
        }
    };
}

impl_for_tuple!(A);
impl_for_tuple!(A, B: 1);
impl_for_tuple!(A, B: 1, C: 2);
impl_for_tuple!(A, B: 1, C: 2, D: 3);
impl_for_tuple!(A, B: 1, C: 2, D: 3, E: 4);
impl_for_tuple!(A, B: 1, C: 2, D: 3, E: 4, F: 5);
impl_for_tuple!(A, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6);
impl_for_tuple!(A, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7);
