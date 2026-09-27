//! The array backend of the evaluator, behind the `ndarray` feature.

use std::cell::Cell;
use std::collections::HashMap;
use std::hash::BuildHasher;
use std::marker::PhantomData;

use ndarray::{ArrayD, ArrayViewD, CowArray, IxDyn, Zip};

use crate::expression::builtins::BuiltinFunction;
use crate::expression::symbol_type::SymbolType;
use crate::foreign::BoxError;
use crate::identifier::Identifier;

use super::error::LaneFailure;
use super::lanes::{Lane, Lanes, Operand};
use super::walk::{Data, Walk};
use super::{EvaluationError, Prepared};

/// An array bound to an identifier for an array evaluation: a view of
/// Booleans, 64-bit integers or reals, of any shape and strides.
#[expect(
    clippy::exhaustive_enums,
    reason = "the three domains of the value kinds, which callers match"
)]
#[derive(Debug, Clone)]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
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
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
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
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
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
    ) -> Result<ArrayD<f64>, BoxError>;
}

/// The [`ArrayKernels`] that compute nothing, so an array evaluation uses
/// the evaluator's own kernels for every native built-in.
#[expect(
    clippy::exhaustive_structs,
    reason = "a stateless unit type that callers name as a value"
)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
pub struct CoreKernels;

impl ArrayKernels for CoreKernels {
    fn handles(&self, _function: BuiltinFunction) -> bool {
        false
    }

    fn native(
        &self,
        function: BuiltinFunction,
        _argument: CowArray<'_, f64, IxDyn>,
    ) -> Result<ArrayD<f64>, BoxError> {
        Err(format!("no array kernel computes {}", function.name()).into())
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
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
    pub fn evaluate_array<S: BuildHasher + Sync>(
        &self,
        environment: &HashMap<Identifier, ArrayBinding<'_>, S>,
        kernels: &dyn ArrayKernels,
    ) -> Result<ArrayValue, EvaluationError> {
        self.check(
            |identifier| environment.get(identifier).map(ArrayBinding::symbol_type),
            |identifier| environment.contains_key(identifier),
        )?;
        let mut referenced: Vec<(&Identifier, &ArrayBinding<'_>)> = environment
            .iter()
            .filter(|(identifier, _)| self.free_identifiers.contains(*identifier))
            .collect();
        referenced.sort_by_key(|(identifier, _)| identifier.id());
        let mut shape = Vec::new();
        for (_, binding) in &referenced {
            shape = broadcast_shape(&shape, binding.shape())?;
        }
        let lane_count: usize = shape.iter().product();
        let lanes = ArrayLanes {
            kernels,
            bindings: PhantomData,
        };
        if lane_count <= CHUNK_LANES {
            let data = Walk::new(&lanes, self.registry, |identifier: &Identifier| {
                environment.get(identifier).map(ArrayBinding::to_data)
            })
            .run(&self.expression)?;
            return Ok(into_value(data, &shape));
        }
        let sources: Vec<(Identifier, ChunkSource<'_>)> = referenced
            .iter()
            .map(|(identifier, binding)| ((*identifier).clone(), ChunkSource::new(binding, &shape)))
            .collect();
        let mut output: Option<Output> = None;
        for start in (0..lane_count).step_by(CHUNK_LANES) {
            let length = CHUNK_LANES.min(lane_count - start);
            let chunk: HashMap<&Identifier, ArrayBinding<'_>> = sources
                .iter()
                .map(|(identifier, source)| (identifier, source.chunk(start, length)))
                .collect();
            let data = Walk::new(&lanes, self.registry, |identifier: &Identifier| {
                chunk.get(identifier).map(ArrayBinding::to_data)
            })
            .run(&self.expression)?;
            output
                .get_or_insert_with(|| Output::for_data(&data, lane_count))
                .push(&data, length);
        }
        Ok(output
            .unwrap_or_else(|| unreachable!("an evaluation of lanes has a chunk"))
            .into_value(&shape))
    }
}

/// How many lanes an array evaluation computes at once. The evaluation of
/// more lanes goes chunk by chunk, so its intermediate values stay small.
const CHUNK_LANES: usize = 65_536;

impl<'a> ArrayBinding<'a> {
    /// Return the binding as the walk's lanes.
    fn to_data<'k>(&self) -> Data<ArrayLanes<'a, 'k>> {
        match self {
            Self::Bool(view) => Data::Bool(CowArray::from(view.clone())),
            Self::Int(view) => Data::Int(CowArray::from(view.clone())),
            Self::Real(view) => Data::Real(CowArray::from(view.clone())),
        }
    }
}

/// Where the lanes of one binding come from, chunk by chunk.
enum ChunkSource<'a> {
    /// A binding of one lane, broadcast to every lane.
    Scalar(ScalarLane),
    /// A binding in the standard layout, of the result's shape: each chunk
    /// is a slice of it.
    Contiguous(ArrayBinding<'a>),
    /// Any other binding, broadcast to the result's shape and copied once
    /// into the standard layout.
    Copied(OwnedLanes),
}

/// The one lane of a single-lane binding.
enum ScalarLane {
    Bool(ArrayD<bool>),
    Int(ArrayD<i64>),
    Real(ArrayD<f64>),
}

/// Lanes owned in the standard layout.
enum OwnedLanes {
    Bool(ArrayD<bool>),
    Int(ArrayD<i64>),
    Real(ArrayD<f64>),
}

/// Return `view` in the standard layout of `shape`, broadcast, as a flat
/// owned array.
fn standard_lanes<T: Lane>(view: &ArrayViewD<'_, T>, shape: &[usize]) -> ArrayD<T> {
    let broadcast = view
        .broadcast(IxDyn(shape))
        .unwrap_or_else(|| unreachable!("the shape is a broadcast of the binding's"));
    let flat: Vec<T> = broadcast.iter().copied().collect();
    ArrayD::from_shape_vec(IxDyn(&[flat.len()]), flat)
        .unwrap_or_else(|_| unreachable!("the lanes fill a flat shape"))
}

/// Return whether `view` holds the lanes of `shape` in the standard layout.
fn is_standard<T>(view: &ArrayViewD<'_, T>, shape: &[usize]) -> bool {
    view.shape() == shape && view.is_standard_layout()
}

/// Return the flat slice `start..start + length` of the standard-layout
/// `view`.
fn slice_lanes<'v, T>(view: &ArrayViewD<'v, T>, start: usize, length: usize) -> ArrayViewD<'v, T> {
    let lanes = view
        .to_slice()
        .unwrap_or_else(|| unreachable!("the binding is in the standard layout"));
    ArrayViewD::from_shape(IxDyn(&[length]), &lanes[start..start + length])
        .unwrap_or_else(|_| unreachable!("the slice fills a flat shape"))
}

/// Return the only lane of `view`, as a 0-d array.
fn only_lane<T: Lane>(view: &ArrayViewD<'_, T>) -> ArrayD<T> {
    let lane = *view
        .iter()
        .next()
        .unwrap_or_else(|| unreachable!("the binding has one lane"));
    ndarray::arr0(lane).into_dyn()
}

impl<'a> ChunkSource<'a> {
    /// Return the source of `binding`'s lanes, broadcast to `shape`.
    fn new(binding: &ArrayBinding<'a>, shape: &[usize]) -> Self {
        let lane_count: usize = binding.shape().iter().product();
        if lane_count == 1 {
            let lane = match binding {
                ArrayBinding::Bool(view) => ScalarLane::Bool(only_lane(view)),
                ArrayBinding::Int(view) => ScalarLane::Int(only_lane(view)),
                ArrayBinding::Real(view) => ScalarLane::Real(only_lane(view)),
            };
            return Self::Scalar(lane);
        }
        let is_contiguous = match binding {
            ArrayBinding::Bool(view) => is_standard(view, shape),
            ArrayBinding::Int(view) => is_standard(view, shape),
            ArrayBinding::Real(view) => is_standard(view, shape),
        };
        if is_contiguous {
            return Self::Contiguous(binding.clone());
        }
        Self::Copied(match binding {
            ArrayBinding::Bool(view) => OwnedLanes::Bool(standard_lanes(view, shape)),
            ArrayBinding::Int(view) => OwnedLanes::Int(standard_lanes(view, shape)),
            ArrayBinding::Real(view) => OwnedLanes::Real(standard_lanes(view, shape)),
        })
    }

    /// Return the lanes `start..start + length` of the flattened broadcast
    /// binding, or its one lane.
    fn chunk(&self, start: usize, length: usize) -> ArrayBinding<'_> {
        match self {
            Self::Scalar(ScalarLane::Bool(lane)) => ArrayBinding::Bool(lane.view()),
            Self::Scalar(ScalarLane::Int(lane)) => ArrayBinding::Int(lane.view()),
            Self::Scalar(ScalarLane::Real(lane)) => ArrayBinding::Real(lane.view()),
            Self::Contiguous(ArrayBinding::Bool(view)) => {
                ArrayBinding::Bool(slice_lanes(view, start, length))
            }
            Self::Contiguous(ArrayBinding::Int(view)) => {
                ArrayBinding::Int(slice_lanes(view, start, length))
            }
            Self::Contiguous(ArrayBinding::Real(view)) => {
                ArrayBinding::Real(slice_lanes(view, start, length))
            }
            Self::Copied(OwnedLanes::Bool(lanes)) => {
                ArrayBinding::Bool(slice_lanes(&lanes.view(), start, length))
            }
            Self::Copied(OwnedLanes::Int(lanes)) => {
                ArrayBinding::Int(slice_lanes(&lanes.view(), start, length))
            }
            Self::Copied(OwnedLanes::Real(lanes)) => {
                ArrayBinding::Real(slice_lanes(&lanes.view(), start, length))
            }
        }
    }
}

/// The lanes of a chunked evaluation's result, gathered chunk by chunk.
enum Output {
    Bool(Vec<bool>),
    Int(Vec<i64>),
    Real(Vec<f64>),
}

/// Append the lanes of `chunk`, of `length` lanes or one broadcast lane, to
/// `lanes`.
fn append_lanes<T: Lane>(lanes: &mut Vec<T>, chunk: &CowArray<'_, T, IxDyn>, length: usize) {
    if chunk.len() == length {
        match chunk.as_slice() {
            Some(slice) => lanes.extend_from_slice(slice),
            None => lanes.extend(chunk.iter().copied()),
        }
    } else {
        let lane = *chunk
            .iter()
            .next()
            .unwrap_or_else(|| unreachable!("a chunk's value has its lanes or one"));
        lanes.extend(std::iter::repeat_n(lane, length));
    }
}

impl Output {
    /// Return the empty output of `lane_count` lanes in the domain of
    /// `data`.
    fn for_data<L: Lanes>(data: &Data<L>, lane_count: usize) -> Self {
        match data {
            Data::Bool(_) => Self::Bool(Vec::with_capacity(lane_count)),
            Data::Int(_) => Self::Int(Vec::with_capacity(lane_count)),
            Data::Real(_) => Self::Real(Vec::with_capacity(lane_count)),
        }
    }

    /// Append the chunk `data` of `length` lanes.
    fn push(&mut self, data: &Data<ArrayLanes<'_, '_>>, length: usize) {
        match (self, data) {
            (Self::Bool(lanes), Data::Bool(chunk)) => append_lanes(lanes, chunk, length),
            (Self::Int(lanes), Data::Int(chunk)) => append_lanes(lanes, chunk, length),
            (Self::Real(lanes), Data::Real(chunk)) => append_lanes(lanes, chunk, length),
            _ => unreachable!("every chunk has the result's domain"),
        }
    }

    /// Return the output shaped as `shape`.
    fn into_value(self, shape: &[usize]) -> ArrayValue {
        let shaped = |error| -> ! { unreachable!("the output fills the shape: {error}") };
        match self {
            Self::Bool(lanes) => ArrayValue::Bool(
                ArrayD::from_shape_vec(IxDyn(shape), lanes).unwrap_or_else(|error| shaped(error)),
            ),
            Self::Int(lanes) => ArrayValue::Int(
                ArrayD::from_shape_vec(IxDyn(shape), lanes).unwrap_or_else(|error| shaped(error)),
            ),
            Self::Real(lanes) => ArrayValue::Real(
                ArrayD::from_shape_vec(IxDyn(shape), lanes).unwrap_or_else(|error| shaped(error)),
            ),
        }
    }
}

/// Return the value of a whole evaluation, broadcast to `shape` and in the
/// standard layout.
fn into_value(data: Data<ArrayLanes<'_, '_>>, shape: &[usize]) -> ArrayValue {
    match data {
        Data::Bool(lanes) => ArrayValue::Bool(into_standard_layout(lanes, shape)),
        Data::Int(lanes) => ArrayValue::Int(into_standard_layout(lanes, shape)),
        Data::Real(lanes) => ArrayValue::Real(into_standard_layout(lanes, shape)),
    }
}

/// Return `lanes`, broadcast to `shape`, as an owned array in the standard
/// layout, copying only a view, a broadcast, or an array in another layout.
fn into_standard_layout<T: Lane>(lanes: CowArray<'_, T, IxDyn>, shape: &[usize]) -> ArrayD<T> {
    if lanes.shape() != shape {
        let broadcast = lanes
            .broadcast(IxDyn(shape))
            .unwrap_or_else(|| unreachable!("the result broadcasts to its bindings' shape"));
        return broadcast.as_standard_layout().into_owned();
    }
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

/// Return the one lane of the 0-d `array`, when the other operand of a map
/// has the result's `shape` and `array` merely broadcasts against it.
fn single_lane<T: Lane>(array: &CowArray<'_, T, IxDyn>, shape: &[usize]) -> Option<T> {
    (array.ndim() == 0 && !shape.is_empty())
        .then(|| array.first().copied())
        .flatten()
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
        if let Some(y) = single_lane(b, &shape) {
            return Ok(CowArray::from(a.map(|&x| f(x, y))));
        }
        if let Some(x) = single_lane(a, &shape) {
            return Ok(CowArray::from(b.map(|&y| f(x, y))));
        }
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

    fn map1_reusing<A: Lane>(
        &self,
        a: CowArray<'a, A, IxDyn>,
        f: impl Fn(A) -> A,
    ) -> CowArray<'a, A, IxDyn> {
        if a.is_view() {
            return self.map1(&a, f);
        }
        let mut owned = a.into_owned();
        owned.mapv_inplace(f);
        CowArray::from(owned)
    }

    fn map2_reusing<A: Lane>(
        &self,
        a: Operand<'_, CowArray<'a, A, IxDyn>>,
        b: Operand<'_, CowArray<'a, A, IxDyn>>,
        f: impl Fn(A, A) -> A,
    ) -> Result<CowArray<'a, A, IxDyn>, EvaluationError> {
        let shape = broadcast_shape(a.get().shape(), b.get().shape())?;
        let is_reusable = |operand: &Operand<'_, CowArray<'a, A, IxDyn>>| matches!(operand, Operand::Owned(array) if !array.is_view() && array.shape() == shape.as_slice());
        if is_reusable(&a) {
            let Operand::Owned(target) = a else {
                unreachable!("a reusable operand is owned")
            };
            let mut owned = target.into_owned();
            if let Some(y) = single_lane(b.get(), &shape) {
                owned.mapv_inplace(|x| f(x, y));
            } else {
                Zip::from(&mut owned)
                    .and(broadcast_view(b.get(), &shape))
                    .for_each(|x, &y| *x = f(*x, y));
            }
            return Ok(CowArray::from(owned));
        }
        if is_reusable(&b) {
            let Operand::Owned(target) = b else {
                unreachable!("a reusable operand is owned")
            };
            let mut owned = target.into_owned();
            if let Some(x) = single_lane(a.get(), &shape) {
                owned.mapv_inplace(|y| f(x, y));
            } else {
                Zip::from(&mut owned)
                    .and(broadcast_view(a.get(), &shape))
                    .for_each(|y, &x| *y = f(x, *y));
            }
            return Ok(CowArray::from(owned));
        }
        self.map2(a.get(), b.get(), f)
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
            return Ok(self.map1_reusing(argument, |x| {
                function
                    .native_value(x)
                    .unwrap_or_else(|| unreachable!("the walk computes native built-ins only"))
            }));
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
