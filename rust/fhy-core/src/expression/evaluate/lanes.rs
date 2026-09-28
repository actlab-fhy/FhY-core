//! What a backend of the evaluator provides: containers of lanes, and the
//! maps that apply a per-lane kernel over them, broadcasting their shapes.
//!
//! The walk ([`walk`](super::walk)) is written once over [`Lanes`], so the
//! scalar backend here and the array backend of the `ndarray` feature apply
//! the same kernels in the same order.

use crate::expression::builtins::BuiltinFunction;

use super::error::{EvaluationError, LaneFailure};

/// The type of one lane.
pub(super) trait Lane: Copy + Default + Send + Sync + 'static {}

impl Lane for bool {}
impl Lane for i64 {}
impl Lane for f64 {}
impl Lane for u32 {}

/// The containers of lanes of a backend, and the maps over them.
///
/// A failing kernel leaves its lane holding the lane type's default; the
/// `try_` maps report whether any lane failed, so the walk can compute
/// which lanes did in a second pass, only when one did.
pub(super) trait Lanes {
    /// A container of lanes of type `T`.
    type Of<T: Lane>;

    /// Return a container of the one lane `value`, which broadcasts
    /// against any shape.
    fn splat<T: Lane>(&self, value: T) -> Self::Of<T>;

    /// Apply `f` to each lane of `a`.
    fn map1<A: Lane, B: Lane>(&self, a: &Self::Of<A>, f: impl Fn(A) -> B) -> Self::Of<B>;

    /// Apply `f` to each lane of `a`, and return whether any lane failed.
    fn try_map1<A: Lane, B: Lane>(
        &self,
        a: &Self::Of<A>,
        f: impl Fn(A) -> Result<B, LaneFailure>,
    ) -> (Self::Of<B>, bool);

    /// Apply `f` to each pair of lanes of `a` and `b`, broadcast together.
    ///
    /// # Errors
    ///
    /// Returns [`EvaluationError::Shape`] if the shapes do not broadcast.
    fn map2<A: Lane, B: Lane, C: Lane>(
        &self,
        a: &Self::Of<A>,
        b: &Self::Of<B>,
        f: impl Fn(A, B) -> C,
    ) -> Result<Self::Of<C>, EvaluationError>;

    /// Apply `f` to each pair of lanes of `a` and `b`, broadcast together,
    /// and return whether any lane failed.
    ///
    /// # Errors
    ///
    /// Returns [`EvaluationError::Shape`] if the shapes do not broadcast.
    fn try_map2<A: Lane, B: Lane, C: Lane>(
        &self,
        a: &Self::Of<A>,
        b: &Self::Of<B>,
        f: impl Fn(A, B) -> Result<C, LaneFailure>,
    ) -> Result<(Self::Of<C>, bool), EvaluationError>;

    /// Apply `f` to each triple of lanes of `a`, `b` and `c`, broadcast
    /// together.
    ///
    /// # Errors
    ///
    /// Returns [`EvaluationError::Shape`] if the shapes do not broadcast.
    fn map3<A: Lane, B: Lane, C: Lane, D: Lane>(
        &self,
        a: &Self::Of<A>,
        b: &Self::Of<B>,
        c: &Self::Of<C>,
        f: impl Fn(A, B, C) -> D,
    ) -> Result<Self::Of<D>, EvaluationError>;

    /// Apply `f` to each lane of `a`, reusing `a`'s storage when the
    /// backend can.
    fn map1_reusing<A: Lane>(&self, a: Self::Of<A>, f: impl Fn(A) -> A) -> Self::Of<A> {
        self.map1(&a, f)
    }

    /// Apply `f` to each pair of lanes of `a` and `b`, broadcast together,
    /// reusing the storage of `a`, or else of `b`, when it is owned and has
    /// the result's shape.
    ///
    /// # Errors
    ///
    /// Returns [`EvaluationError::Shape`] if the shapes do not broadcast.
    fn map2_reusing<A: Lane>(
        &self,
        a: Operand<'_, Self::Of<A>>,
        b: Operand<'_, Self::Of<A>>,
        f: impl Fn(A, A) -> A,
    ) -> Result<Self::Of<A>, EvaluationError> {
        self.map2(a.get(), b.get(), f)
    }

    /// Return the first lane of `a` other than zero, in C order, and its
    /// flat index in the result an array backend computes, or `None` for
    /// the scalar backend's one lane.
    fn first_nonzero(&self, a: &Self::Of<u32>) -> Option<(Option<usize>, u32)>;

    /// Return the real value of the native built-in `function` at each lane
    /// of `argument`.
    ///
    /// # Errors
    ///
    /// Returns [`EvaluationError::Kernel`] if a plugged-in kernel fails.
    fn native(
        &self,
        function: BuiltinFunction,
        argument: Self::Of<f64>,
    ) -> Result<Self::Of<f64>, EvaluationError>;
}

/// An operand of a map: owned, so a backend may reuse its storage, or
/// borrowed.
pub(super) enum Operand<'v, T> {
    /// An operand no one else holds.
    Owned(T),
    /// An operand others hold too.
    Borrowed(&'v T),
}

impl<T> Operand<'_, T> {
    /// Return the operand.
    pub(super) fn get(&self) -> &T {
        match self {
            Self::Owned(value) => value,
            Self::Borrowed(value) => value,
        }
    }
}

/// The scalar backend: one lane per container.
#[derive(Debug, Clone, Copy)]
pub(super) struct ScalarLanes;

impl Lanes for ScalarLanes {
    type Of<T: Lane> = T;

    fn splat<T: Lane>(&self, value: T) -> T {
        value
    }

    fn map1<A: Lane, B: Lane>(&self, a: &A, f: impl Fn(A) -> B) -> B {
        f(*a)
    }

    fn try_map1<A: Lane, B: Lane>(
        &self,
        a: &A,
        f: impl Fn(A) -> Result<B, LaneFailure>,
    ) -> (B, bool) {
        f(*a).map_or_else(|_| (B::default(), true), |value| (value, false))
    }

    fn map2<A: Lane, B: Lane, C: Lane>(
        &self,
        a: &A,
        b: &B,
        f: impl Fn(A, B) -> C,
    ) -> Result<C, EvaluationError> {
        Ok(f(*a, *b))
    }

    fn try_map2<A: Lane, B: Lane, C: Lane>(
        &self,
        a: &A,
        b: &B,
        f: impl Fn(A, B) -> Result<C, LaneFailure>,
    ) -> Result<(C, bool), EvaluationError> {
        Ok(f(*a, *b).map_or_else(|_| (C::default(), true), |value| (value, false)))
    }

    fn map3<A: Lane, B: Lane, C: Lane, D: Lane>(
        &self,
        a: &A,
        b: &B,
        c: &C,
        f: impl Fn(A, B, C) -> D,
    ) -> Result<D, EvaluationError> {
        Ok(f(*a, *b, *c))
    }

    fn first_nonzero(&self, a: &u32) -> Option<(Option<usize>, u32)> {
        (*a != 0).then_some((None, *a))
    }

    fn native(&self, function: BuiltinFunction, argument: f64) -> Result<f64, EvaluationError> {
        Ok(function
            .native_value(argument)
            .unwrap_or_else(|| unreachable!("the walk computes native built-ins only")))
    }
}
