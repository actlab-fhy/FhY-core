//! Evaluation of expressions to values, and folding of native calls.
//!
//! An [`Evaluator`] reads the built-in catalogue and a [`FunctionRegistry`]
//! it borrows, and does two things:
//!
//! - [`fold`](Evaluator::fold) replaces each native call whose arguments
//!   are literals by the literal it computes, and each reference to a
//!   constant by its value, keeping every other node;
//! - [`evaluate`](Evaluator::evaluate) computes an expression's value over
//!   an environment binding its free identifiers: to [`Scalar`]s, or, with
//!   the `ndarray` feature, to arrays ([`Prepared::evaluate_array`]).
//!
//! # Semantics
//!
//! An evaluation computes in three domains, one per [`SymbolType`]: `bool`,
//! `i64` and `f64`. A literal integer outside the 64-bit range, and a
//! decimal no binary float equals, are refused.
//!
//! - **Arithmetic** on two integers is checked: an overflow, and a floor
//!   division or remainder by zero, fail. Mixed integer and real operands
//!   convert the integer to the nearest real, in comparisons too. `/` is
//!   the real quotient even of two integers. `//` rounds toward negative
//!   infinity, and `%` has the sign of the divisor, over reals as Python's
//!   `divmod` computes them. An integer raised to a negative integer power
//!   fails; a real power is `powf`. Real arithmetic is IEEE's.
//! - **Booleans** compare with `==` and `!=`; any other arithmetic or
//!   ordering use of a Boolean is refused.
//! - **Connectives** reduce their operands in order, and a **piecewise**
//!   takes the value of its first case whose condition holds. Every branch
//!   is computed; its failures are discarded where it is not selected.
//! - **Native built-ins** compute [`BuiltinFunction::native_value`], and an
//!   integer-sorted result (`round`, `floor`, `ceil`) must be finite and in
//!   the 64-bit range.
//!
//! A failure of one lane of a computation — an overflow, a division by
//! zero, a negative integer exponent, or a cast of a value with no integer
//! — is a [`LaneFailure`]. It stays with its lane and is raised only if the
//! lane reaches the result: a piecewise that selects another branch in that
//! lane, or a connective whose other operand is `false` (for `&&`) or
//! `true` (for `||`) there, discards it. A scalar evaluation is one lane,
//! and an array evaluation computes every lane exactly as the scalar
//! evaluation of that lane's bindings would.
//!
//! [`SymbolType`]: crate::expression::SymbolType
//! [`BuiltinFunction::native_value`]: crate::expression::builtins::BuiltinFunction::native_value
//!
//! # Examples
//!
//! ```
//! use std::collections::HashMap;
//!
//! use fhy_core::expression::evaluate::{Evaluator, Scalar};
//! use fhy_core::expression::registry::FunctionRegistry;
//! use fhy_core::expression::Expression;
//! use fhy_core::identifier::Identifier;
//!
//! let registry = FunctionRegistry::new();
//! let x = Identifier::new("x");
//! let tree = (Expression::from(x.clone()) * 3).floor_divide(2);
//!
//! let value = Evaluator::new(&registry).evaluate(&tree, &HashMap::from([(x, Scalar::Int(7))]))?;
//!
//! assert_eq!(value, Scalar::Int(10));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod error;
mod fold;
mod kernel;
mod lanes;
mod value;
mod walk;

#[cfg(feature = "ndarray")]
mod array;

use std::collections::{HashMap, HashSet};
use std::hash::BuildHasher;

use crate::identifier::Identifier;

use super::builtins::BuiltinConstant;
use super::callee::Callee;
use super::literal::LiteralValue;
use super::node::Expression;
use super::pattern::CallbackError;
use super::registry::{FunctionRegistry, NativeFunction};
use super::screen::BooleanScreen;
use super::symbol_type::SymbolType;
use lanes::ScalarLanes;
use walk::{Data, Walk};

pub use error::{EvaluationError, FoldError, LaneFailure, NearMiss};
pub use value::Scalar;

#[cfg(feature = "ndarray")]
pub use array::{ArrayBinding, ArrayKernels, ArrayValue, CoreKernels};

/// The implementations of native user functions, which a
/// [`fold`](Evaluator::fold) calls.
///
/// The built-in natives are computed by the evaluator itself.
pub trait NativeCalls {
    /// Return the value of the native user function `function` at
    /// `arguments`, which have its parameters' sorts and hold no decimal.
    ///
    /// # Errors
    ///
    /// Returns the implementation's error, which the fold reports as
    /// [`FoldError::Native`].
    fn call(
        &self,
        function: &NativeFunction,
        arguments: &[LiteralValue],
    ) -> Result<LiteralValue, CallbackError>;
}

/// The [`NativeCalls`] with no implementation: every call of a native user
/// function fails.
#[expect(
    clippy::exhaustive_structs,
    reason = "a stateless unit type that callers name as a value"
)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct NoNativeCalls;

impl NativeCalls for NoNativeCalls {
    fn call(
        &self,
        function: &NativeFunction,
        _arguments: &[LiteralValue],
    ) -> Result<LiteralValue, CallbackError> {
        Err(format!(
            "native function {:?} has no implementation",
            function.name().as_str()
        )
        .into())
    }
}

/// The result of a [`fold`](Evaluator::fold).
#[derive(Debug, Clone)]
pub struct Folding {
    output: Expression,
    not_inlined: Vec<Callee>,
}

impl Folding {
    /// Return the folded expression.
    #[must_use]
    pub fn output(&self) -> &Expression {
        &self.output
    }

    /// Return the folded expression, consuming the folding.
    #[must_use]
    pub fn into_output(self) -> Expression {
        self.output
    }

    /// Return the functions with a body, composed built-ins and user
    /// functions, whose calls the fold kept, once each, in the order first
    /// reached.
    #[must_use]
    pub fn not_inlined(&self) -> &[Callee] {
        &self.not_inlined
    }
}

/// Folding and evaluation over the built-in catalogue and a registry.
#[derive(Debug, Clone, Copy)]
pub struct Evaluator<'r> {
    registry: &'r FunctionRegistry,
}

impl<'r> Evaluator<'r> {
    /// Create the evaluator reading user functions and constants from
    /// `registry`.
    #[must_use]
    pub fn new(registry: &'r FunctionRegistry) -> Self {
        Self { registry }
    }

    /// Replace each native call whose arguments are literals by the
    /// literal it computes, and each reference to a constant's identifier
    /// by the constant's value.
    ///
    /// The walk is bottom-up, so nested native calls fold inside out. A
    /// call of a native built-in is computed by
    /// [`BuiltinFunction::native_value`](crate::expression::builtins::BuiltinFunction::native_value),
    /// an integer-sorted result becoming the exact integer, and a call of a
    /// native user function by `natives`, its result checked against the
    /// function's result sort. A decimal argument becomes the binary float
    /// equal to it. A call of a composed built-in or of a user function
    /// with a body is kept, and the function is listed in
    /// [`Folding::not_inlined`]. Every other node is kept, with its
    /// children folded; arithmetic is not folded.
    ///
    /// Each distinct node is folded once, and the walk keeps its pending
    /// nodes on the heap, so a tree of any depth folds. The output is
    /// `expression` itself ([`Expression::ptr_eq`]) when nothing folds.
    ///
    /// # Errors
    ///
    /// Returns [`FoldError::UnknownFunction`] for a call of an unknown name
    /// and [`FoldError::NotCallable`] for a call of a constant. A native
    /// call with literal arguments is checked, in this order, for its
    /// argument count ([`FoldError::Arity`]), its arguments' sorts
    /// ([`FoldError::ArgumentSort`]) and exact decimals
    /// ([`FoldError::InexactDecimal`]); then a built-in's integer result
    /// must be finite ([`FoldError::NonFiniteCast`]), and a user function's
    /// implementation must succeed ([`FoldError::Native`]) and return a
    /// value of its result sort ([`FoldError::ResultSort`]). A call's
    /// arguments are folded before the call.
    pub fn fold(
        &self,
        expression: &Expression,
        natives: &dyn NativeCalls,
    ) -> Result<Folding, FoldError> {
        fold::fold(self.registry, natives, expression)
    }

    /// Inline `expression`, ready for evaluation over environments.
    ///
    /// # Errors
    ///
    /// Returns [`EvaluationError::Inline`] if inlining fails.
    pub fn prepare(&self, expression: &Expression) -> Result<Prepared<'r>, EvaluationError> {
        let expression = self
            .registry
            .inline(expression)
            .map_err(EvaluationError::Inline)?;
        let free_identifiers = expression.free_identifiers();
        Ok(Prepared {
            registry: self.registry,
            expression,
            free_identifiers,
        })
    }

    /// Return the value of `expression` over `environment`, as
    /// [`prepare`](Self::prepare) and [`Prepared::evaluate`] compute it.
    ///
    /// # Errors
    ///
    /// Returns what those two return.
    pub fn evaluate<S: BuildHasher>(
        &self,
        expression: &Expression,
        environment: &HashMap<Identifier, Scalar, S>,
    ) -> Result<Scalar, EvaluationError> {
        self.prepare(expression)?.evaluate(environment)
    }
}

/// An inlined expression, ready to be evaluated over environments.
#[derive(Debug, Clone)]
pub struct Prepared<'r> {
    registry: &'r FunctionRegistry,
    expression: Expression,
    free_identifiers: HashSet<Identifier>,
}

impl Prepared<'_> {
    /// Return the inlined expression.
    #[must_use]
    pub fn expression(&self) -> &Expression {
        &self.expression
    }

    /// Return the free identifiers of the inlined expression: the ones an
    /// environment must bind, besides the constants'.
    #[must_use]
    pub fn free_identifiers(&self) -> &HashSet<Identifier> {
        &self.free_identifiers
    }

    /// Return the value of the expression over `environment`.
    ///
    /// Bindings of identifiers the expression does not refer to are
    /// ignored. Before the walk, the expression is screened with the value
    /// kinds of the bindings ([`BooleanScreen::check_logical_operands`]),
    /// and a binding of a constant's identifier it refers to is refused.
    ///
    /// [`BooleanScreen::check_logical_operands`]: crate::expression::BooleanScreen::check_logical_operands
    ///
    /// # Errors
    ///
    /// Returns, in this order: [`EvaluationError::IllTyped`] from the
    /// screen; [`EvaluationError::BoundNativeConstant`]; then the refusals
    /// the walk meets, such as [`EvaluationError::Unbound`],
    /// [`EvaluationError::BooleanArithmetic`] or
    /// [`EvaluationError::Unsupported`] for a native user function; and
    /// [`EvaluationError::Lane`] if the result's lane failed.
    pub fn evaluate<S: BuildHasher>(
        &self,
        environment: &HashMap<Identifier, Scalar, S>,
    ) -> Result<Scalar, EvaluationError> {
        self.check(
            |identifier| environment.get(identifier).map(|value| value.symbol_type()),
            |identifier| environment.contains_key(identifier),
        )?;
        let data = Walk::new(&ScalarLanes, self.registry, |identifier: &Identifier| {
            environment.get(identifier).map(|value| match *value {
                Scalar::Bool(value) => Data::Bool(value),
                Scalar::Int(value) => Data::Int(value),
                Scalar::Real(value) => Data::Real(value),
            })
        })
        .run(&self.expression)?;
        Ok(match data {
            Data::Bool(value) => Scalar::Bool(value),
            Data::Int(value) => Scalar::Int(value),
            Data::Real(value) => Scalar::Real(value),
        })
    }

    /// Screen the expression with the value kinds `symbol_type` gives its
    /// identifiers, then refuse the constants it refers to that `is_bound`.
    fn check(
        &self,
        symbol_type: impl Fn(&Identifier) -> Option<SymbolType>,
        is_bound: impl Fn(&Identifier) -> bool,
    ) -> Result<(), EvaluationError> {
        BooleanScreen::new()
            .with_sorts(self.registry)
            .with_symbol_types(&symbol_type)
            .check_logical_operands(&self.expression)
            .map_err(EvaluationError::IllTyped)?;
        let mut bound_constants: Vec<Identifier> = self
            .free_identifiers
            .iter()
            .filter(|identifier| {
                is_bound(identifier)
                    && (BuiltinConstant::of_identifier(identifier).is_some()
                        || self.registry.constant(identifier).is_some())
            })
            .cloned()
            .collect();
        if bound_constants.is_empty() {
            return Ok(());
        }
        bound_constants.sort_by_key(Identifier::id);
        Err(EvaluationError::BoundNativeConstant(bound_constants))
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Evaluator<'static>>();
    assert_send_sync::<Prepared<'static>>();
    assert_send_sync::<Folding>();
    assert_send_sync::<Scalar>();
};
