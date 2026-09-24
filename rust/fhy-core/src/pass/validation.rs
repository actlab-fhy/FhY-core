//! Validators, the adapter that runs a pass as one, and collect-all
//! validation pipelines.

use std::borrow::Cow;
use std::fmt;

use super::analysis::AnalysisCache;
use super::compiler_pass::{CompilerPass, PassFailure, run_check, short_type_name};
use super::context::PassContext;
use super::error::render_chain;
use crate::diagnostic::{Diagnostic, DiagnosticLevel, Note, ValidationReport};
use crate::identifier::{HasIdentifier, Identifier};

/// A collect-all check over IR of type `I`.
///
/// A validator reports every problem it finds as a diagnostic into its
/// [`PassContext`], so one check can report many problems; it returns an
/// error only when it cannot finish the check. A [`ValidationManager`] runs
/// validators into one report. [`PassValidator`] runs a
/// [`CompilerPass`] that outputs `()` as a validator.
///
/// # Examples
///
/// ```
/// use fhy_core::diagnostic::DiagnosticLevel;
/// use fhy_core::identifier::Identifier;
/// use fhy_core::pass::{PassContext, PassFailure, ValidationManager, Validator};
///
/// struct NonNegative;
///
/// impl Validator<i64> for NonNegative {
///     fn validate(&mut self, ir: &i64, cx: &mut PassContext<'_>) -> Result<(), PassFailure> {
///         if *ir < 0 {
///             cx.report_text(DiagnosticLevel::Error, format!("{ir} is negative"), None);
///         }
///         Ok(())
///     }
/// }
///
/// let mut manager = ValidationManager::new(Identifier::new("checks"));
/// manager.add(NonNegative);
///
/// let report = manager.validate(&-1);
///
/// assert!(report.has_errors());
/// assert_eq!(report.records()[0].validator_name(), "NonNegative");
/// ```
pub trait Validator<I> {
    /// Return the validator's name, the source of its diagnostics.
    ///
    /// By default: [`short_type_name::<Self>()`](short_type_name).
    fn name(&self) -> Cow<'static, str> {
        short_type_name::<Self>()
    }

    /// Check `ir`, reporting the problems found as diagnostics into `cx`.
    ///
    /// # Errors
    ///
    /// Returns an error if the validator cannot finish the check. The
    /// [`ValidationManager`] then records the validator as failed, and adds
    /// an error diagnostic unless the validator reported one.
    fn validate(&mut self, ir: &I, cx: &mut PassContext<'_>) -> Result<(), PassFailure>;
}

/// Forward to the borrowed validator, so a pipeline can run a validator its
/// caller keeps and reads afterwards.
impl<I, V: Validator<I> + ?Sized> Validator<I> for &mut V {
    fn name(&self) -> Cow<'static, str> {
        (**self).name()
    }

    fn validate(&mut self, ir: &I, cx: &mut PassContext<'_>) -> Result<(), PassFailure> {
        (**self).validate(ir, cx)
    }
}

/// Forward to the boxed validator.
impl<I, V: Validator<I> + ?Sized> Validator<I> for Box<V> {
    fn name(&self) -> Cow<'static, str> {
        (**self).name()
    }

    fn validate(&mut self, ir: &I, cx: &mut PassContext<'_>) -> Result<(), PassFailure> {
        (**self).validate(ir, cx)
    }
}

/// The adapter that runs a [`CompilerPass`] outputting `()` as a
/// [`Validator`].
///
/// It is named after the pass. A check runs the pass's
/// [`validate_input`](CompilerPass::validate_input), then
/// [`skip`](CompilerPass::skip), which ends the check when it returns an
/// output, then [`run`](CompilerPass::run) and
/// [`validate_output`](CompilerPass::validate_output). A hook that fails
/// fails the check with a [`PassError`](super::PassError), after the error
/// diagnostic recording it. [`did_change`](CompilerPass::did_change) and
/// [`preserved_analyses`](CompilerPass::preserved_analyses) are not called,
/// since a check changes nothing.
///
/// # Examples
///
/// ```
/// use fhy_core::expr::Expression;
/// use fhy_core::identifier::Identifier;
/// use fhy_core::pass::{PassValidator, ValidationManager, WalkPass};
/// use fhy_core::tree::{TraversalOrder, TreeVisitor};
///
/// #[derive(Default)]
/// struct NodeCounter(usize);
///
/// impl<C: ?Sized> TreeVisitor<Expression, C> for NodeCounter {
///     type Error = std::convert::Infallible;
///
///     fn visit(&mut self, _node: &Expression, _cx: &mut C) -> Result<(), Self::Error> {
///         self.0 += 1;
///         Ok(())
///     }
/// }
///
/// let mut walk = WalkPass::new(NodeCounter::default(), TraversalOrder::Pre);
/// let mut manager = ValidationManager::new(Identifier::new("checks"));
/// manager.add(PassValidator::new(&mut walk));
///
/// let report = manager.validate(&(Expression::from(Identifier::new("x")) + 1));
/// drop(manager);
///
/// assert!(!report.has_errors());
/// assert_eq!(walk.visitor().0, 3);
/// ```
#[derive(Debug)]
pub struct PassValidator<P> {
    pass: P,
}

impl<P> PassValidator<P> {
    /// Create the validator that runs `pass`.
    #[must_use]
    pub fn new(pass: P) -> Self {
        Self { pass }
    }

    /// Return the pass.
    #[must_use]
    pub fn pass(&self) -> &P {
        &self.pass
    }

    /// Return the pass for mutation.
    #[must_use]
    pub fn pass_mut(&mut self) -> &mut P {
        &mut self.pass
    }

    /// Return the pass, consuming the validator.
    #[must_use]
    pub fn into_pass(self) -> P {
        self.pass
    }
}

impl<I, P: CompilerPass<I, ()>> Validator<I> for PassValidator<P> {
    fn name(&self) -> Cow<'static, str> {
        self.pass.name()
    }

    fn validate(&mut self, ir: &I, cx: &mut PassContext<'_>) -> Result<(), PassFailure> {
        run_check(&mut self.pass, ir, cx).map_err(|error| Box::new(error) as PassFailure)
    }
}

/// The record of one validator in a [`ValidationReport`].
///
/// The record holds no diagnostics of its own: they are a slice of the
/// report's diagnostics, read with [`diagnostics_in`](Self::diagnostics_in),
/// so a report stores each diagnostic once.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ValidatorRecord {
    validator_name: Cow<'static, str>,
    first_diagnostic: usize,
    end_diagnostic: usize,
    failed: bool,
}

impl ValidatorRecord {
    /// Return the name of the validator.
    #[must_use]
    pub fn validator_name(&self) -> &str {
        &self.validator_name
    }

    /// Return whether the validator's [`Validator::validate`] returned an
    /// error.
    #[must_use]
    pub fn is_failed(&self) -> bool {
        self.failed
    }

    /// Return the validator's diagnostics, in emission order: a slice of
    /// `report`'s diagnostics.
    ///
    /// # Panics
    ///
    /// Panics if `report` is not the report this record came from and holds
    /// fewer diagnostics than the record refers to.
    #[must_use]
    pub fn diagnostics_in<'r>(
        &self,
        report: &'r ValidationReport<ValidatorRecord>,
    ) -> &'r [Diagnostic] {
        &report.diagnostics()[self.first_diagnostic..self.end_diagnostic]
    }
}

/// Return the error diagnostic for the validator `validator_name` that failed
/// with `error` without reporting an error itself.
fn synthesize_silent_failure(validator_name: Cow<'static, str>, error: &PassFailure) -> Diagnostic {
    let message = format!(
        "validator {validator_name:?} failed without reporting an error: {}",
        render_chain(error.as_ref())
    );
    Diagnostic::error(Note::with_other_kind(message), validator_name)
}

/// A sequence of validators whose diagnostics aggregate into one report.
///
/// Unlike a [`PassManager`](super::PassManager), validation never stops
/// early: every validator runs, even after earlier ones reported errors or
/// failed. Run on its own, each validator computes the analyses it requests
/// afresh; as a pipeline's verifier, validators share the pipeline's
/// analysis cache.
pub struct ValidationManager<'p, I> {
    name: Identifier,
    validators: Vec<Box<dyn Validator<I> + Send + 'p>>,
}

impl<'p, I> ValidationManager<'p, I> {
    /// Create the empty validation pipeline `name`.
    #[must_use]
    pub fn new(name: Identifier) -> Self {
        Self {
            name,
            validators: Vec::new(),
        }
    }

    /// Return the pipeline's name.
    #[must_use]
    pub fn name(&self) -> &Identifier {
        &self.name
    }

    /// Append `validator` to the pipeline.
    ///
    /// Pass `&mut validator` to keep the validator and read its state after
    /// a validation. The validator is `Send`, so the pipeline can move to
    /// another thread.
    pub fn add(&mut self, validator: impl Validator<I> + Send + 'p) {
        self.validators.push(Box::new(validator));
    }

    /// Return the validators' names, in pipeline order.
    pub fn validator_names(&self) -> impl Iterator<Item = Cow<'static, str>> + '_ {
        self.validators.iter().map(Validator::name)
    }

    /// Run every validator over `ir` and return the aggregated report.
    ///
    /// Each validator runs with a fresh [`PassContext`] named after it. The
    /// report's diagnostics are every validator's diagnostics in pipeline
    /// order, and its records hold one [`ValidatorRecord`] per validator,
    /// whose diagnostics are its slice of the report's. A validator that
    /// fails keeps the diagnostics it emitted; when none of them is an
    /// error, the report gains the error `validator "<name>" failed without
    /// reporting an error: <chain>`, where `<chain>` is the failure's message
    /// followed by each of its sources, joined by `: `.
    #[must_use]
    pub fn validate(&mut self, ir: &I) -> ValidationReport<ValidatorRecord> {
        self.run_validators(ir, None)
    }

    /// Run every validator over `ir` as [`validate`](Self::validate) does,
    /// with their analyses cached in the pipeline cache `cache`.
    pub(super) fn validate_in(
        &mut self,
        ir: &I,
        cache: &mut AnalysisCache,
    ) -> ValidationReport<ValidatorRecord> {
        self.run_validators(ir, Some(cache))
    }

    /// Run every validator over `ir`, caching their analyses in `cache` when
    /// one is given.
    fn run_validators(
        &mut self,
        ir: &I,
        mut cache: Option<&mut AnalysisCache>,
    ) -> ValidationReport<ValidatorRecord> {
        let mut diagnostics = Vec::new();
        let mut records = Vec::with_capacity(self.validators.len());
        for validator in &mut self.validators {
            let mut cx = PassContext::new(validator.name(), cache.as_deref_mut());
            let result = validator.validate(ir, &mut cx);
            let (validator_name, mut captured) = cx.into_parts();
            let failed = match result {
                Ok(()) => false,
                Err(error) => {
                    let reported_error = captured
                        .iter()
                        .any(|diagnostic| diagnostic.level() == DiagnosticLevel::Error);
                    if !reported_error {
                        captured.push(synthesize_silent_failure(validator_name.clone(), &error));
                    }
                    true
                }
            };
            let first_diagnostic = diagnostics.len();
            diagnostics.append(&mut captured);
            records.push(ValidatorRecord {
                validator_name,
                first_diagnostic,
                end_diagnostic: diagnostics.len(),
                failed,
            });
        }
        ValidationReport::new(diagnostics, records)
    }
}

impl<I> Default for ValidationManager<'_, I> {
    /// Create the empty validation pipeline `validation-pipeline`.
    fn default() -> Self {
        Self::new(Identifier::new("validation-pipeline"))
    }
}

impl<I> HasIdentifier for ValidationManager<'_, I> {
    fn identifier(&self) -> &Identifier {
        &self.name
    }
}

/// Render the pipeline's name and its validators' names.
impl<I> fmt::Debug for ValidationManager<'_, I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let validator_names: Vec<_> = self.validator_names().collect();
        f.debug_struct("ValidationManager")
            .field("name", &self.name)
            .field("validators", &validator_names)
            .finish()
    }
}
