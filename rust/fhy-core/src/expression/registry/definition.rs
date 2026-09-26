//! The entries a [`FunctionRegistry`](super::FunctionRegistry) holds: user
//! functions defined by an expression, functions computed natively, and
//! named constants.

use std::collections::HashSet;
use std::sync::Arc;

use crate::expression::callee::FunctionName;
use crate::expression::literal::LiteralValue;
use crate::expression::node::Expression;
use crate::expression::sort::FunctionSort;
use crate::identifier::Identifier;

use super::error::{ConstantValueError, FunctionDefinitionError};

/// A user function defined by an expression over its parameters.
///
/// A definition holds a [`FunctionName`], its parameters in positional
/// order, one [`FunctionSort`] per parameter, the result's sort, and the
/// body. Building one checks only its own shape: one sort per parameter,
/// and no parameter named twice. Which identifiers the body may refer to
/// depends on the registry it joins, which checks them when it registers
/// the definition.
///
/// Cloning is cheap: the definition is shared behind an `Arc`.
///
/// # Examples
///
/// ```
/// use fhy_core::expression::registry::FunctionDefinition;
/// use fhy_core::expression::{Expression, FunctionName, FunctionSort};
/// use fhy_core::identifier::Identifier;
///
/// let x = Identifier::new("x");
/// let increment = FunctionDefinition::try_new(
///     FunctionName::try_new("increment")?,
///     [x.clone()],
///     [FunctionSort::Int],
///     FunctionSort::Int,
///     Expression::from(x) + 1,
/// )?;
/// assert_eq!(increment.name().as_str(), "increment");
/// assert_eq!(increment.parameter_sorts(), [FunctionSort::Int]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct FunctionDefinition(Arc<DefinitionData>);

/// The fields of a [`FunctionDefinition`].
#[derive(Debug)]
struct DefinitionData {
    name: FunctionName,
    parameters: Box<[Identifier]>,
    parameter_sorts: Box<[FunctionSort]>,
    result_sort: FunctionSort,
    body: Expression,
}

impl FunctionDefinition {
    /// Define the function `name` of `parameters`, whose sorts are
    /// `parameter_sorts` in the same order, returning `result_sort`, as
    /// `body`.
    ///
    /// # Errors
    ///
    /// Returns [`FunctionDefinitionError::SortCountMismatch`] if the
    /// parameters and the sorts differ in number, and
    /// [`FunctionDefinitionError::RepeatedParameter`] naming the first
    /// parameter that occurs twice.
    pub fn try_new(
        name: FunctionName,
        parameters: impl IntoIterator<Item = Identifier>,
        parameter_sorts: impl IntoIterator<Item = FunctionSort>,
        result_sort: FunctionSort,
        body: Expression,
    ) -> Result<Self, FunctionDefinitionError> {
        let parameters: Box<[Identifier]> = parameters.into_iter().collect();
        let parameter_sorts: Box<[FunctionSort]> = parameter_sorts.into_iter().collect();
        if parameters.len() != parameter_sorts.len() {
            return Err(FunctionDefinitionError::SortCountMismatch {
                function: name,
                parameters: parameters.len(),
                sorts: parameter_sorts.len(),
            });
        }
        let mut seen = HashSet::with_capacity(parameters.len());
        if let Some(repeated) = parameters.iter().find(|parameter| !seen.insert(*parameter)) {
            return Err(FunctionDefinitionError::RepeatedParameter {
                function: name,
                parameter: repeated.clone(),
            });
        }
        Ok(Self(Arc::new(DefinitionData {
            name,
            parameters,
            parameter_sorts,
            result_sort,
            body,
        })))
    }

    /// Return the function's name.
    #[must_use]
    pub fn name(&self) -> &FunctionName {
        &self.0.name
    }

    /// Return the parameters in positional order.
    #[must_use]
    pub fn parameters(&self) -> &[Identifier] {
        &self.0.parameters
    }

    /// Return the sort of each parameter, in positional order.
    #[must_use]
    pub fn parameter_sorts(&self) -> &[FunctionSort] {
        &self.0.parameter_sorts
    }

    /// Return the sort of the result.
    #[must_use]
    pub fn result_sort(&self) -> FunctionSort {
        self.0.result_sort
    }

    /// Return the body, an expression over the parameters.
    #[must_use]
    pub fn body(&self) -> &Expression {
        &self.0.body
    }
}

/// A user function computed natively, outside the expression language.
///
/// It has a name and a signature, and no body: a registry records that a
/// call of the name computes a value of the result sort from arguments of
/// the parameter sorts, and inlining keeps such a call as it is.
///
/// # Examples
///
/// ```
/// use fhy_core::expression::registry::NativeFunction;
/// use fhy_core::expression::{FunctionName, FunctionSort};
///
/// let softplus = NativeFunction::new(
///     FunctionName::try_new("softplus")?,
///     [FunctionSort::Real],
///     FunctionSort::Real,
/// );
/// assert_eq!(softplus.parameter_sorts(), [FunctionSort::Real]);
/// # Ok::<(), fhy_core::expression::FunctionNameError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NativeFunction {
    name: FunctionName,
    parameter_sorts: Arc<[FunctionSort]>,
    result_sort: FunctionSort,
}

impl NativeFunction {
    /// Declare the native function `name` taking arguments of
    /// `parameter_sorts`, in order, and returning `result_sort`.
    #[must_use]
    pub fn new(
        name: FunctionName,
        parameter_sorts: impl IntoIterator<Item = FunctionSort>,
        result_sort: FunctionSort,
    ) -> Self {
        Self {
            name,
            parameter_sorts: parameter_sorts.into_iter().collect(),
            result_sort,
        }
    }

    /// Return the function's name.
    #[must_use]
    pub fn name(&self) -> &FunctionName {
        &self.name
    }

    /// Return the sort of each parameter, in positional order.
    #[must_use]
    pub fn parameter_sorts(&self) -> &[FunctionSort] {
        &self.parameter_sorts
    }

    /// Return the sort of the result.
    #[must_use]
    pub fn result_sort(&self) -> FunctionSort {
        self.result_sort
    }
}

/// A named constant holding a literal value of its sort.
///
/// A registry mints one identifier for each constant it registers, and an
/// expression refers to the constant by a reference to that identifier.
///
/// # Examples
///
/// ```
/// use fhy_core::expression::registry::NativeConstant;
/// use fhy_core::expression::{FunctionName, FunctionSort, LiteralValue};
///
/// let answer = NativeConstant::try_new(FunctionName::try_new("answer")?, FunctionSort::Nat, 42)?;
/// assert_eq!(answer.value(), &LiteralValue::from(42));
/// assert!(NativeConstant::try_new(FunctionName::try_new("minus")?, FunctionSort::Nat, -1).is_err());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct NativeConstant {
    name: FunctionName,
    sort: FunctionSort,
    value: LiteralValue,
}

impl NativeConstant {
    /// Declare the constant `name` of `sort` holding `value`.
    ///
    /// # Errors
    ///
    /// Returns [`ConstantValueError`] unless `sort`
    /// [accepts](FunctionSort::accepts_literal) the value: a Boolean for
    /// [`FunctionSort::Bool`], a non-negative integer for
    /// [`FunctionSort::Nat`], an integer for [`FunctionSort::Int`], and an
    /// integer, a float or a decimal for [`FunctionSort::Real`].
    pub fn try_new(
        name: FunctionName,
        sort: FunctionSort,
        value: impl Into<LiteralValue>,
    ) -> Result<Self, ConstantValueError> {
        let value = value.into();
        if !sort.accepts_literal(&value) {
            return Err(ConstantValueError::new(name, sort, value));
        }
        Ok(Self { name, sort, value })
    }

    /// Return the constant's name.
    #[must_use]
    pub fn name(&self) -> &FunctionName {
        &self.name
    }

    /// Return the constant's sort.
    #[must_use]
    pub fn sort(&self) -> FunctionSort {
        self.sort
    }

    /// Return the constant's value.
    #[must_use]
    pub fn value(&self) -> &LiteralValue {
        &self.value
    }
}
