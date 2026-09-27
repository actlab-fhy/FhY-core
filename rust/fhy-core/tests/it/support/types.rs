//! Builders of types and dimensions for the type tests.

use fhy_core::expression::{Expression, LiteralValue};
use fhy_core::identifier::Identifier;
use fhy_core::types::{
    CoreDataType, DataType, Dimension, IndexType, NumericalType, TemplateDataType, Type,
};

/// Return the dimension of the literal `value`.
pub(crate) fn literal_dimension(value: impl Into<LiteralValue>) -> Dimension {
    Dimension::Expression(Expression::from(value.into()))
}

/// Return the dimension referencing `identifier`.
pub(crate) fn identifier_dimension(identifier: &Identifier) -> Dimension {
    Dimension::Expression(Expression::from(identifier.clone()))
}

/// Return the numerical type of `data_type` over `shape`.
pub(crate) fn array(
    data_type: impl Into<DataType>,
    shape: impl IntoIterator<Item = Dimension>,
) -> Type {
    Type::Numerical(NumericalType::new(data_type, shape))
}

/// Return the scalar of `core_data_type`.
pub(crate) fn scalar(core_data_type: CoreDataType) -> Type {
    Type::Numerical(NumericalType::scalar(core_data_type))
}

/// Return the template data type of `identifier`, without widths.
pub(crate) fn template(identifier: &Identifier) -> DataType {
    DataType::Template(TemplateDataType::new(identifier.clone()))
}

/// Return the template data type of `identifier` constrained to `widths`.
pub(crate) fn constrained_template(identifier: &Identifier, widths: &[u32]) -> DataType {
    DataType::Template(
        TemplateDataType::with_widths(identifier.clone(), widths.iter().copied())
            .expect("every width is positive"),
    )
}

/// Return the index type from `lower` to `upper` in steps of `stride`.
pub(crate) fn index(
    lower: impl Into<Expression>,
    upper: impl Into<Expression>,
    stride: impl Into<Expression>,
) -> Type {
    Type::Index(IndexType::new(lower.into(), upper.into(), stride.into()))
}

/// Structural equivalence of a value whose comparison cannot fail, as no
/// test type extension's does.
pub(crate) trait Equivalent {
    /// Return whether `self` and `other` are structurally equivalent.
    fn is_equivalent(&self, other: &Self) -> bool;
}

macro_rules! impl_equivalent {
    ($($Type:ty),+ $(,)?) => {
        $(
            impl Equivalent for $Type {
                fn is_equivalent(&self, other: &Self) -> bool {
                    self.is_structurally_equivalent(other)
                        .expect("the comparison runs no failing extension")
                }
            }
        )+
    };
}

impl_equivalent!(
    Type,
    DataType,
    fhy_core::types::TypeUnificationEnvironment,
    fhy_core::symbol_table::SymbolFrame,
    fhy_core::symbol_table::VariableFrame,
    fhy_core::symbol_table::FunctionFrame,
    fhy_core::symbol_table::SymbolTable<fhy_core::symbol_table::SymbolFrame>,
);
