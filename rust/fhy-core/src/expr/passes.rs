//! The compiler passes over expressions.
//!
//! [`RewriteRuleApplier`] applies a list of rewrite rules bottom-up over an
//! expression, and [`ExpressionPrettyFormatter`] formats an expression as
//! text. [`register_expression_passes`] registers the passes a
//! [`PassRegistry`] can build by name. This is the only module of
//! [`expr`](super) that depends on [`crate::pass`]; the pass framework
//! itself does not depend on expressions.

use std::borrow::Cow;

use crate::diagnostic::DiagnosticLevel;
use crate::pass::{CompilerPass, PassContext, PassFailure, PassRegistrationError, PassRegistry};

use super::display::FormatOptions;
use super::node::Expression;
use super::pattern::{FiredRule, RewriteRule, RuleRun, run_rewrite_rules};

/// The name of [`RewriteRuleApplier`], which it is registered under: a
/// stable registry key, not a Rust path.
const RULE_APPLIER_PASS_NAME: &str = "fhy_core.symbolic.expression.apply_rewrite_rules";

/// The description of [`RewriteRuleApplier`].
const RULE_APPLIER_PASS_DESCRIPTION: &str =
    "Apply a sequence of rewrite rules bottom-up over an expression tree.";

/// A compiler pass applying a list of rewrite rules bottom-up over an
/// expression, once per run.
///
/// A run is [`apply_rewrite_rules`](super::pattern::apply_rewrite_rules)
/// with the pass's rules: its output is the rewritten tree, and it changed
/// the IR exactly when the output is a different node from the input
/// ([`Expression::ptr_eq`]); see
/// [`RewriteOutcome::is_changed`](super::pattern::RewriteOutcome::is_changed).
/// Each firing of a named rule reports an informational diagnostic, `Applied rewrite rule "<name>".`, the name
/// escaped as `Debug` writes a string, and the firings of the last run are
/// kept for [`fired`](Self::fired). A failing callback or a refused rebuild
/// fails the run with the
/// [`RewriteError`](super::pattern::RewriteError), which the resulting
/// [`PassError`](crate::pass::PassError) holds as its
/// [`source`](std::error::Error::source).
///
/// The pass is named `fhy_core.symbolic.expression.apply_rewrite_rules`, a
/// stable registry key rather than a Rust path, and
/// [`register_expression_passes`] registers it under that name.
///
/// # Examples
///
/// ```
/// use fhy_core::identifier::Identifier;
/// use fhy_core::pass::ExecutePass;
/// use fhy_core::expr::passes::RewriteRuleApplier;
/// use fhy_core::expr::pattern::{CallbackError, Pattern, RewriteRule};
/// use fhy_core::expr::{BinaryOperation, Expression, LiteralValue};
///
/// // `x * 1 -> x`
/// let rule = RewriteRule::new(
///     Pattern::binary(
///         Some(BinaryOperation::Multiply),
///         Pattern::capture("x", Pattern::wildcard())?,
///         Pattern::literal(Some(LiteralValue::from(1))),
///     ),
///     |bindings| {
///         bindings
///             .get("x")
///             .cloned()
///             .ok_or_else(|| CallbackError::new("`x` is unbound"))
///     },
/// )
/// .with_name("x * 1 -> x");
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
///     "Applied rewrite rule \"x * 1 -> x\"."
/// );
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct RewriteRuleApplier {
    rules: Vec<RewriteRule>,
    fired: Vec<FiredRule>,
}

impl RewriteRuleApplier {
    /// Create the pass applying `rules`, tried in the order given.
    #[must_use]
    pub fn new(rules: impl IntoIterator<Item = RewriteRule>) -> Self {
        Self {
            rules: rules.into_iter().collect(),
            fired: Vec::new(),
        }
    }

    /// Return the rules, in the order they are tried.
    #[must_use]
    pub fn rules(&self) -> &[RewriteRule] {
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

impl CompilerPass<Expression> for RewriteRuleApplier {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed(RULE_APPLIER_PASS_NAME)
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed(RULE_APPLIER_PASS_DESCRIPTION)
    }

    fn run(
        &mut self,
        ir: &Expression,
        cx: &mut PassContext<'_>,
    ) -> Result<Expression, PassFailure> {
        let RuleRun { output, fired } = run_rewrite_rules(ir, &self.rules);
        for name in fired.iter().filter_map(FiredRule::name) {
            let message = format!("Applied rewrite rule {name:?}.");
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

    fn did_change(&mut self, input: &Expression, output: &String) -> Result<bool, PassFailure> {
        let _ = (input, output);
        Ok(true)
    }
}

/// Register the expression passes in `registry`.
///
/// Registers [`RewriteRuleApplier`] under its name,
/// `fhy_core.symbolic.expression.apply_rewrite_rules`, and its
/// description; [`PassRegistry::create`] then builds an applier with no
/// rules. The name is a stable registry key, not a Rust path. Registering
/// again in the same registry changes nothing.
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
    registry.register::<RewriteRuleApplier, Expression, Expression>(|| RewriteRuleApplier::new([]))
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<RewriteRuleApplier>();
    assert_send_sync::<ExpressionPrettyFormatter>();
};
