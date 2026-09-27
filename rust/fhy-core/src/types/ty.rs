//! Types: numerical arrays, index ranges, and types defined outside this
//! crate.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::expression::{Expression, FormatOptions, IdentifierStyle};

use super::data_type::DataType;
use super::extension::TypeExtension;

/// A compiler type.
///
/// Cloning a numerical or index type shares it. `==` and `Hash` are
/// structural for the built-in variants, and go through
/// [`TypeExtension::eq_extension`] and [`TypeExtension::hash_extension`] for
/// an extension. `Display` writes a numerical type as `int32[4, (N::7 +
/// 1)]`, an index type as `index(0:N::7:1)`, with identifier ids, and an
/// extension through its own `Display`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Type {
    /// A numerical array; a scalar has the empty shape.
    Numerical(NumericalType),
    /// An index range, as a Python `range(start, stop, step)`.
    Index(IndexType),
    /// A type defined outside this crate.
    Extension(Arc<dyn TypeExtension>),
}

/// A numerical array type: a data type over a shape of dimensions.
///
/// A shape of exactly one [`Dimension::Wildcard`] is the full-shape
/// wildcard of template binding; a wildcard among other dimensions matches
/// one dimension.
#[derive(Debug, Clone)]
pub struct NumericalType(Arc<NumericalParts>);

#[derive(Debug)]
struct NumericalParts {
    data_type: DataType,
    shape: Box<[Dimension]>,
}

/// One dimension of a numerical type's shape.
#[expect(
    clippy::exhaustive_enums,
    reason = "a dimension is an expression or the wildcard, as in Python's shapes"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Dimension {
    /// A dimension of the given extent.
    Expression(Expression),
    /// A dimension that matches any extent, Python's `...`.
    Wildcard,
}

/// An index range type: a lower bound, an upper bound and a stride.
#[derive(Debug, Clone)]
pub struct IndexType(Arc<IndexParts>);

#[derive(Debug)]
struct IndexParts {
    lower_bound: Expression,
    upper_bound: Expression,
    stride: Expression,
}

impl NumericalType {
    /// Return the array of `data_type` over `shape`.
    #[must_use]
    pub fn new(data_type: impl Into<DataType>, shape: impl IntoIterator<Item = Dimension>) -> Self {
        Self(Arc::new(NumericalParts {
            data_type: data_type.into(),
            shape: shape.into_iter().collect(),
        }))
    }

    /// Return the scalar of `data_type`: the array of the empty shape.
    #[must_use]
    pub fn scalar(data_type: impl Into<DataType>) -> Self {
        Self::new(data_type, [])
    }

    /// Return the data type.
    #[must_use]
    pub fn data_type(&self) -> &DataType {
        &self.0.data_type
    }

    /// Return the shape.
    #[must_use]
    pub fn shape(&self) -> &[Dimension] {
        &self.0.shape
    }

    /// Return whether the type is a scalar: its shape is empty.
    #[must_use]
    pub fn is_scalar(&self) -> bool {
        self.0.shape.is_empty()
    }

    /// Return whether the shape is the full-shape wildcard: exactly one
    /// [`Dimension::Wildcard`].
    #[must_use]
    pub fn is_wildcard_shape(&self) -> bool {
        matches!(&*self.0.shape, [Dimension::Wildcard])
    }

    /// Return whether `this` and `other` are handles to the same type.
    #[must_use]
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        Arc::ptr_eq(&this.0, &other.0)
    }
}

impl IndexType {
    /// Return the range from `lower_bound` to `upper_bound` in steps of
    /// `stride`.
    #[must_use]
    pub fn new(lower_bound: Expression, upper_bound: Expression, stride: Expression) -> Self {
        Self(Arc::new(IndexParts {
            lower_bound,
            upper_bound,
            stride,
        }))
    }

    /// Return the lower bound.
    #[must_use]
    pub fn lower_bound(&self) -> &Expression {
        &self.0.lower_bound
    }

    /// Return the upper bound.
    #[must_use]
    pub fn upper_bound(&self) -> &Expression {
        &self.0.upper_bound
    }

    /// Return the stride.
    #[must_use]
    pub fn stride(&self) -> &Expression {
        &self.0.stride
    }

    /// Return whether `this` and `other` are handles to the same type.
    #[must_use]
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        Arc::ptr_eq(&this.0, &other.0)
    }
}

impl Type {
    /// Return the name of the type's kind, as a refusal names it:
    /// `NumericalType`, `IndexType`, or the extension's name.
    #[must_use]
    pub fn kind_name(&self) -> String {
        match self {
            Self::Numerical(_) => "NumericalType".to_owned(),
            Self::Index(_) => "IndexType".to_owned(),
            Self::Extension(extension) => extension.type_name().into_owned(),
        }
    }

    /// Return whether `this` and `other` are handles to the same type.
    #[must_use]
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        match (this, other) {
            (Self::Numerical(left), Self::Numerical(right)) => NumericalType::ptr_eq(left, right),
            (Self::Index(left), Self::Index(right)) => IndexType::ptr_eq(left, right),
            (Self::Extension(left), Self::Extension(right)) => Arc::ptr_eq(left, right),
            _ => false,
        }
    }
}

impl From<NumericalType> for Type {
    fn from(numerical: NumericalType) -> Self {
        Self::Numerical(numerical)
    }
}

impl From<IndexType> for Type {
    fn from(index: IndexType) -> Self {
        Self::Index(index)
    }
}

impl From<Expression> for Dimension {
    fn from(expression: Expression) -> Self {
        Self::Expression(expression)
    }
}

impl PartialEq for NumericalType {
    fn eq(&self, other: &Self) -> bool {
        Self::ptr_eq(self, other)
            || (self.0.data_type == other.0.data_type && self.0.shape == other.0.shape)
    }
}

impl Eq for NumericalType {}

impl Hash for NumericalType {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.data_type.hash(state);
        self.0.shape.hash(state);
    }
}

impl PartialEq for IndexType {
    fn eq(&self, other: &Self) -> bool {
        Self::ptr_eq(self, other)
            || (self.0.lower_bound == other.0.lower_bound
                && self.0.upper_bound == other.0.upper_bound
                && self.0.stride == other.0.stride)
    }
}

impl Eq for IndexType {}

impl Hash for IndexType {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.lower_bound.hash(state);
        self.0.upper_bound.hash(state);
        self.0.stride.hash(state);
    }
}

impl PartialEq for Type {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Numerical(left), Self::Numerical(right)) => left == right,
            (Self::Index(left), Self::Index(right)) => left == right,
            (Self::Extension(left), Self::Extension(right)) => {
                Arc::ptr_eq(left, right) || left.eq_extension(right.as_ref())
            }
            _ => false,
        }
    }
}

impl Eq for Type {}

impl Hash for Type {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Numerical(numerical) => numerical.hash(state),
            Self::Index(index) => index.hash(state),
            Self::Extension(extension) => extension.hash_extension(state),
        }
    }
}

/// The options expressions print with in a type: identifiers with ids.
fn expression_options() -> FormatOptions {
    FormatOptions::default().with_identifier_style(IdentifierStyle::NameHintWithId)
}

impl fmt::Display for Dimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Expression(expression) => {
                write!(f, "{}", expression.display(expression_options()))
            }
            Self::Wildcard => f.write_str("..."),
        }
    }
}

impl fmt::Display for NumericalType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}[", self.0.data_type)?;
        for (index, dimension) in self.0.shape.iter().enumerate() {
            if index > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{dimension}")?;
        }
        f.write_str("]")
    }
}

impl fmt::Display for IndexType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let options = expression_options();
        write!(
            f,
            "index({}:{}:{})",
            self.0.lower_bound.display(options),
            self.0.upper_bound.display(options),
            self.0.stride.display(options)
        )
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Numerical(numerical) => write!(f, "{numerical}"),
            Self::Index(index) => write!(f, "{index}"),
            Self::Extension(extension) => write!(f, "{extension}"),
        }
    }
}
