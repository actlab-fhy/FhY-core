//! The array backend of the evaluator, behind the `ndarray` feature.

use std::cell::Cell;
use std::collections::HashMap;
use std::hash::BuildHasher;
use std::marker::PhantomData;

use ndarray::{ArrayD, ArrayViewD, CowArray, IxDyn, Zip};

use crate::expression::builtins::BuiltinFunction;
use crate::expression::pattern::CallbackError;
use crate::expression::symbol_type::SymbolType;
use crate::identifier::Identifier;

use super::error::LaneFailure;
use super::lanes::{Lane, Lanes};
use super::walk::{Data, Walk};
use super::{EvaluationError, Prepared};

/// An array bound to an identifier for an array evaluation: a view of
/// Booleans, 64-bit integers or reals, of any shape and strides.
#[expect(
    clippy::exhaustive_enums,
    reason = "the three domains of the value kinds, which callers match"
)]
#[derive(Debug, Clone)]
pub enum ArrayBinding<'a> {
    /// Booleans.
    Bool(ArrayViewD<'a, bool>),
    /// 64-bit signed integers.
    Int(ArrayViewD<'a, i64>),
    /// Binary64 reals.
    Real(ArrayViewD<'a, f64>),
}

impl ArrayBinding<'_> {
    /// Return the value kind of the binding's domain.
    #[must_use]
    pub fn symbol_type(&self) -> SymbolType {
        match self {
            Self::Bool(_) => SymbolType::Bool,
            Self::Int(_) => SymbolType::Int,
            Self::Real(_) => SymbolType::Real,
        }
    }

    /// Return the binding's shape.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::Bool(array) => array.shape(),
            Self::Int(array) => array.shape(),
            Self::Real(array) => array.shape(),
        }
    }
}

/// The result of an array evaluation: an owned array of Booleans, 64-bit
/// integers or reals, in the standard layout.
#[expect(
    clippy::exhaustive_enums,
    reason = "the three domains of the value kinds, which callers match"
)]
#[derive(Debug, Clone, PartialEq)]
pub enum ArrayValue {
    /// Booleans.
    Bool(ArrayD<bool>),
    /// 64-bit signed integers.
    Int(ArrayD<i64>),
    /// Binary64 reals.
    Real(ArrayD<f64>),
}

impl ArrayValue {
    /// Return the value kind of the result's domain.
    #[must_use]
    pub fn symbol_type(&self) -> SymbolType {
        match self {
            Self::Bool(_) => SymbolType::Bool,
            Self::Int(_) => SymbolType::Int,
            Self::Real(_) => SymbolType::Real,
        }
    }

    /// Return the result's shape.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::Bool(array) => array.shape(),
            Self::Int(array) => array.shape(),
            Self::Real(array) => array.shape(),
        }
    }
}

/// Array kernels an array evaluation computes native built-ins with,
/// instead of its own per-lane kernels.
///
/// An implementation must compute each lane of its output from the same
/// lane of its input, with the output's shape the input's.
pub trait ArrayKernels {
    /// Return whether [`native`](Self::native) computes `function`; the
    /// evaluator's own kernel computes it otherwise.
    fn handles(&self, function: BuiltinFunction) -> bool;

    /// Return the real value of the native built-in `function` at every
    /// lane of `argument`, before its result sort's cast.
    ///
    /// # Errors
    ///
    /// Returns the kernel's error, which the evaluation reports as
    /// [`EvaluationError::Kernel`].
    fn native(
        &self,
        function: BuiltinFunction,
        argument: CowArray<'_, f64, IxDyn>,
    ) -> Result<ArrayD<f64>, CallbackError>;
}

/// The [`ArrayKernels`] that compute nothing, so an array evaluation uses
/// the evaluator's own kernels for every native built-in.
#[expect(
    clippy::exhaustive_structs,
    reason = "a stateless unit type that callers name as a value"
)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct CoreKernels;

impl ArrayKernels for CoreKernels {
    fn handles(&self, _function: BuiltinFunction) -> bool {
        false
    }

    fn native(
        &self,
        function: BuiltinFunction,
        _argument: CowArray<'_, f64, IxDyn>,
    ) -> Result<ArrayD<f64>, CallbackError> {
        Err(format!("no array kernel computes {}", function.name()).into())
    }
}

impl Prepared<'_> {
    /// Return the value of the expression over the arrays of
    /// `environment`, broadcast together as `NumPy` broadcasts, computing
    /// the native built-ins `kernels` handles with it.
    ///
    /// Every lane is computed as [`evaluate`](Self::evaluate) computes the
    /// scalar evaluation of that lane's bindings. The result is a new
    /// array, never a binding's.
    ///
    /// # Errors
    ///
    /// Returns what [`evaluate`](Self::evaluate) returns, for the first
    /// failed lane of the result in C order, and also
    /// [`EvaluationError::Shape`] for operands that do not broadcast and
    /// [`EvaluationError::Kernel`] for a failing kernel.
    pub fn evaluate_array<S: BuildHasher>(
        &self,
        environment: &HashMap<Identifier, ArrayBinding<'_>, S>,
        kernels: &dyn ArrayKernels,
    ) -> Result<ArrayValue, EvaluationError> {
        self.check(
            |identifier| environment.get(identifier).map(ArrayBinding::symbol_type),
            |identifier| environment.contains_key(identifier),
        )?;
        let lanes = ArrayLanes {
            kernels,
            bindings: PhantomData,
        };
        let data = Walk::new(&lanes, self.registry, |identifier: &Identifier| {
            environment.get(identifier).map(|binding| match binding {
                ArrayBinding::Bool(view) => Data::Bool(CowArray::from(view.view())),
                ArrayBinding::Int(view) => Data::Int(CowArray::from(view.view())),
                ArrayBinding::Real(view) => Data::Real(CowArray::from(view.view())),
            })
        })
        .run(&self.expression)?;
        Ok(match data {
            Data::Bool(lanes) => ArrayValue::Bool(into_standard_layout(lanes)),
            Data::Int(lanes) => ArrayValue::Int(into_standard_layout(lanes)),
            Data::Real(lanes) => ArrayValue::Real(into_standard_layout(lanes)),
        })
    }
}

/// Return `lanes` as an owned array in the standard layout, copying only
/// a view or an array in another layout.
fn into_standard_layout<T: Lane>(lanes: CowArray<'_, T, IxDyn>) -> ArrayD<T> {
    let owned = lanes.into_owned();
    if owned.is_standard_layout() {
        owned
    } else {
        owned.as_standard_layout().into_owned()
    }
}

/// The array backend: containers are arrays, borrowed for the bindings,
/// which live for `'a`, and owned for the values computed from them.
struct ArrayLanes<'a, 'k> {
    kernels: &'k dyn ArrayKernels,
    bindings: PhantomData<&'a ()>,
}

/// Return the shape `left` and `right` broadcast to, as `NumPy` broadcasts:
/// aligned at their last axes, each pair of lengths equal or one of them 1.
fn broadcast_shape(left: &[usize], right: &[usize]) -> Result<Vec<usize>, EvaluationError> {
    let rank = left.len().max(right.len());
    let mut shape = vec![0; rank];
    for (axis, length) in shape.iter_mut().enumerate() {
        let from_end = rank - axis;
        let a = left
            .len()
            .checked_sub(from_end)
            .map_or(1, |index| left[index]);
        let b = right
            .len()
            .checked_sub(from_end)
            .map_or(1, |index| right[index]);
        *length = if a == b || b == 1 {
            a
        } else if a == 1 {
            b
        } else {
            return Err(EvaluationError::Shape {
                left: left.to_vec(),
                right: right.to_vec(),
            });
        };
    }
    Ok(shape)
}

/// Return a view of `array` broadcast to `shape`.
fn broadcast_view<'v, T>(array: &'v CowArray<'_, T, IxDyn>, shape: &[usize]) -> ArrayViewD<'v, T> {
    if array.shape() == shape {
        return array.view();
    }
    array
        .broadcast(IxDyn(shape))
        .unwrap_or_else(|| unreachable!("the shape is a broadcast of the array's"))
}

impl<'a> Lanes for ArrayLanes<'a, '_> {
    type Of<T: Lane> = CowArray<'a, T, IxDyn>;

    fn splat<T: Lane>(&self, value: T) -> CowArray<'a, T, IxDyn> {
        CowArray::from(ndarray::arr0(value).into_dyn())
    }

    fn map1<A: Lane, B: Lane>(
        &self,
        a: &CowArray<'a, A, IxDyn>,
        f: impl Fn(A) -> B,
    ) -> CowArray<'a, B, IxDyn> {
        CowArray::from(a.map(|&x| f(x)))
    }

    fn try_map1<A: Lane, B: Lane>(
        &self,
        a: &CowArray<'a, A, IxDyn>,
        f: impl Fn(A) -> Result<B, LaneFailure>,
    ) -> (CowArray<'a, B, IxDyn>, bool) {
        let failed = Cell::new(false);
        let lanes = a.map(|&x| {
            f(x).unwrap_or_else(|_| {
                failed.set(true);
                B::default()
            })
        });
        (CowArray::from(lanes), failed.get())
    }

    fn map2<A: Lane, B: Lane, C: Lane>(
        &self,
        a: &CowArray<'a, A, IxDyn>,
        b: &CowArray<'a, B, IxDyn>,
        f: impl Fn(A, B) -> C,
    ) -> Result<CowArray<'a, C, IxDyn>, EvaluationError> {
        let shape = broadcast_shape(a.shape(), b.shape())?;
        let lanes = Zip::from(broadcast_view(a, &shape))
            .and(broadcast_view(b, &shape))
            .map_collect(|&x, &y| f(x, y));
        Ok(CowArray::from(lanes))
    }

    fn try_map2<A: Lane, B: Lane, C: Lane>(
        &self,
        a: &CowArray<'a, A, IxDyn>,
        b: &CowArray<'a, B, IxDyn>,
        f: impl Fn(A, B) -> Result<C, LaneFailure>,
    ) -> Result<(CowArray<'a, C, IxDyn>, bool), EvaluationError> {
        let failed = Cell::new(false);
        let lanes = self.map2(a, b, |x, y| {
            f(x, y).unwrap_or_else(|_| {
                failed.set(true);
                C::default()
            })
        })?;
        Ok((lanes, failed.get()))
    }

    fn map3<A: Lane, B: Lane, C: Lane, D: Lane>(
        &self,
        a: &CowArray<'a, A, IxDyn>,
        b: &CowArray<'a, B, IxDyn>,
        c: &CowArray<'a, C, IxDyn>,
        f: impl Fn(A, B, C) -> D,
    ) -> Result<CowArray<'a, D, IxDyn>, EvaluationError> {
        let shape = broadcast_shape(&broadcast_shape(a.shape(), b.shape())?, c.shape())?;
        let lanes = Zip::from(broadcast_view(a, &shape))
            .and(broadcast_view(b, &shape))
            .and(broadcast_view(c, &shape))
            .map_collect(|&x, &y, &z| f(x, y, z));
        Ok(CowArray::from(lanes))
    }

    fn first_nonzero(&self, a: &CowArray<'a, u32, IxDyn>) -> Option<u32> {
        a.iter().copied().find(|&id| id != 0)
    }

    fn native(
        &self,
        function: BuiltinFunction,
        argument: CowArray<'a, f64, IxDyn>,
    ) -> Result<CowArray<'a, f64, IxDyn>, EvaluationError> {
        if !self.kernels.handles(function) {
            let lanes = argument.map(|&x| {
                function
                    .native_value(x)
                    .unwrap_or_else(|| unreachable!("the walk computes native built-ins only"))
            });
            return Ok(CowArray::from(lanes));
        }
        let shape = argument.shape().to_vec();
        let lanes = self
            .kernels
            .native(function, argument)
            .map_err(|source| EvaluationError::Kernel { function, source })?;
        if lanes.shape() != shape {
            return Err(EvaluationError::Kernel {
                function,
                source: format!(
                    "the kernel returned shape {:?} for an argument of shape {shape:?}",
                    lanes.shape()
                )
                .into(),
            });
        }
        Ok(CowArray::from(lanes))
    }
}
