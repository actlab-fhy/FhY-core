//! The compiler passes over expressions.
//!
//! [`RewriteRuleApplier`] applies a list of rewrite rules bottom-up over an
//! expression, and [`ExpressionPrettyFormatter`] formats an expression as
//! text. [`register_expression_passes`] registers the passes a
//! [`PassRegistry`] can build by name.

use std::borrow::Cow;

use crate::diagnostic::DiagnosticLevel;
use crate::pass::{CompilerPass, PassContext, PassFailure, PassRegistrationError, PassRegistry};

use super::display::FormatOptions;
use super::node::Expression;
use super::pattern::{FiredRule, RewriteRule, Rule, RuleRun, run_rewrite_rules};

/// A compiler pass applying a list of rewrite rules bottom-up over an
/// expression, once per run.
///
/// The rules are of any one [`Rule`] type, [`RewriteRule`] by default; a
/// list of `Box<dyn Rule + Send>` mixes rule types. A run is
/// [`apply_rewrite_rules`](super::pattern::apply_rewrite_rules) with the
/// pass's rules, and it changed the IR exactly when its output is a
/// different node from its input ([`Expression::ptr_eq`]). Each firing of a
/// named rule reports the informational diagnostic
/// `applied rewrite rule "<name>"`, the name escaped as `Debug` writes a
/// string, and the last run's firings are kept for [`fired`](Self::fired).
/// A failing callback or a refused rebuild fails the run with the
/// [`RewriteError`](super::pattern::RewriteError), the
/// [`source`](std::error::Error::source) of the resulting
/// [`PassError`](crate::pass::PassError).
///
/// The pass is named [`NAME`](Self::NAME). An applier of no rules needs its
/// rule type named, as `RewriteRuleApplier::<RewriteRule>::new([])`.
///
/// # Examples
///
/// ```
/// use fhy_core::identifier::Identifier;
/// use fhy_core::pass::ExecutePass;
/// use fhy_core::expr::passes::RewriteRuleApplier;
/// use fhy_core::expr::pattern::{Capture, Pattern, RewriteRule};
/// use fhy_core::expr::{BinaryOperation, Expression};
///
/// // `x * 1 -> x`
/// let x = Capture::new("x");
/// let pattern = Pattern::binary(BinaryOperation::Multiply, Pattern::capture(&x), Pattern::literal(1));
/// let rule = RewriteRule::new(pattern, move |bindings| Ok(bindings[&x].clone()))
///     .with_name("x * 1 -> x");
/// let a = Expression::from(Identifier::new("a"));
/// let mut applier = RewriteRuleApplier::new([rule]);
///
/// let outcome = applier.execute(&(&a * 1))?;
///
/// assert!(Expression::ptr_eq(outcome.output(), &a));
/// assert!(outcome.is_changed());
/// assert_eq!(applier.fired().len(), 1);
/// assert_eq!(
///     outcome.diagnostics()[0].message_text(),
///     "applied rewrite rule \"x * 1 -> x\""
/// );
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct RewriteRuleApplier<R = RewriteRule> {
    rules: Vec<R>,
    fired: Vec<FiredRule>,
}

impl RewriteRuleApplier {
    /// The name of the pass, which it is registered under: a stable
    /// registry key, not a Rust path.
    pub const NAME: &'static str = "fhy_core.symbolic.expression.apply_rewrite_rules";

    /// The description of the pass.
    pub const DESCRIPTION: &'static str =
        "Apply a sequence of rewrite rules bottom-up over an expression tree.";
}

impl<R: Rule> RewriteRuleApplier<R> {
    /// Create the pass applying `rules`, tried in the order given.
    #[must_use]
    pub fn new(rules: impl IntoIterator<Item = R>) -> Self {
        Self {
            rules: rules.into_iter().collect(),
            fired: Vec::new(),
        }
    }

    /// Return the rules, in the order they are tried.
    #[must_use]
    pub fn rules(&self) -> &[R] {
        &self.rules
    }

    /// Return the firings of the last run, in the order of
    /// [`RewriteOutcome::fired`](super::pattern::RewriteOutcome::fired):
    /// empty before the first run, and those before the failure after a
    /// failed run.
    #[must_use]
    pub fn fired(&self) -> &[FiredRule] {
        &self.fired
    }
}

impl<R: Rule> CompilerPass<Expression> for RewriteRuleApplier<R> {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed(RewriteRuleApplier::NAME)
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed(RewriteRuleApplier::DESCRIPTION)
    }

    fn run(
        &mut self,
        ir: &Expression,
        cx: &mut PassContext<'_>,
    ) -> Result<Expression, PassFailure> {
        let RuleRun { output, fired } = run_rewrite_rules(ir, &self.rules);
        for name in fired.iter().filter_map(FiredRule::name) {
            let message = format!("applied rewrite rule {name:?}");
            cx.report_text(DiagnosticLevel::Info, message, None);
        }
        self.fired = fired;
        output.map_err(|error| Box::new(error) as PassFailure)
    }

    fn did_change(&mut self, input: &Expression, output: &Expression) -> Result<bool, PassFailure> {
        Ok(!Expression::ptr_eq(input, output))
    }
}

/// A compiler pass formatting an expression as text under
/// [`FormatOptions`], as [`Expression::display`] does.
///
/// Every run counts as a change, since its text output is never its input
/// expression. The default formatter uses the default options.
///
/// # Examples
///
/// ```
/// use fhy_core::identifier::Identifier;
/// use fhy_core::pass::ExecutePass;
/// use fhy_core::expr::passes::ExpressionPrettyFormatter;
/// use fhy_core::expr::{Expression, FormatOptions, Notation};
///
/// let x = Expression::from(Identifier::new("x"));
/// let options = FormatOptions::default().with_notation(Notation::Functional);
/// let mut formatter = ExpressionPrettyFormatter::new(options);
///
/// let outcome = formatter.execute(&(&x + 1))?;
///
/// assert_eq!(outcome.output(), "(add x 1)");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ExpressionPrettyFormatter {
    options: FormatOptions,
}

impl ExpressionPrettyFormatter {
    /// Create the pass formatting under `options`.
    #[must_use]
    pub fn new(options: FormatOptions) -> Self {
        Self { options }
    }

    /// Return the options the pass formats under.
    #[must_use]
    pub fn options(&self) -> FormatOptions {
        self.options
    }
}

impl CompilerPass<Expression, String> for ExpressionPrettyFormatter {
    fn run(&mut self, ir: &Expression, _cx: &mut PassContext<'_>) -> Result<String, PassFailure> {
        Ok(ir.display(self.options).to_string())
    }

    fn did_change(&mut self, _input: &Expression, _output: &String) -> Result<bool, PassFailure> {
        Ok(true)
    }
}

/// Register the expression passes in `registry`.
///
/// Registers [`RewriteRuleApplier`] under its [`NAME`](RewriteRuleApplier::NAME)
/// and [`DESCRIPTION`](RewriteRuleApplier::DESCRIPTION);
/// [`PassRegistry::create`] then builds an applier with no rules.
/// Registering again in the same registry changes nothing.
///
/// # Errors
///
/// Returns an error if another pass is registered under one of the names.
///
/// # Examples
///
/// ```
/// use fhy_core::expr::Expression;
/// use fhy_core::expr::passes::register_expression_passes;
/// use fhy_core::pass::PassRegistry;
///
/// let mut registry = PassRegistry::new();
/// register_expression_passes(&mut registry)?;
/// let applier = registry
///     .create::<Expression, Expression>("fhy_core.symbolic.expression.apply_rewrite_rules")?;
///
/// assert_eq!(
///     applier.description(),
///     "Apply a sequence of rewrite rules bottom-up over an expression tree."
/// );
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn register_expression_passes(
    registry: &mut PassRegistry,
) -> Result<(), PassRegistrationError> {
    registry.register::<RewriteRuleApplier, Expression, Expression>(|| {
        RewriteRuleApplier::<RewriteRule>::new([])
    })
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<RewriteRuleApplier>();
    assert_send_sync::<ExpressionPrettyFormatter>();
};
