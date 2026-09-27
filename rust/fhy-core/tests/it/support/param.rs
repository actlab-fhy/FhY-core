//! Helpers of the param tests: values and constraints, a simplifier that
//! evaluates ground expressions, an observer that records its events, a
//! test-local custom domain, and opaque values with an order.

use std::any::Any;
use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use fhy_core::constraint::{
    Constraint, ConstraintError, EquationConstraint, Event, Member, MemberKind, Opaque,
    OpaqueValue, Outcome, Polarity, SetConstraint, Value,
};
use fhy_core::expression::evaluate::{Evaluator, Scalar};
use fhy_core::expression::registry::FunctionRegistry;
use fhy_core::expression::{BigInt, Expression, LiteralValue, SymbolType};
use fhy_core::foreign::BoxError;
use fhy_core::identifier::Identifier;
use fhy_core::param::{
    CustomDomain, IntervalProfile, ParamContext, ParamDomain, ParamEvent, ParamObserver,
    ScreenReason, Side,
};
use fhy_core::solver::{SatResult, Simplifier, SimplifyContext, SmtSolver, Solver};

use super::constraint::{TestValueError, member_set};
use super::solver::{FakeBackendError, RecordingSmtSolver};

/// Return the Boolean value `value`.
pub(crate) fn boolean(value: bool) -> Value {
    Value::Bool(value)
}

/// Return the float value `value`.
pub(crate) fn float(value: f64) -> Value {
    Value::Float(value)
}

/// Return the reference to `identifier`.
pub(crate) fn reference(identifier: &Identifier) -> Expression {
    Expression::from(identifier)
}

/// Return the integer literal `value`.
pub(crate) fn literal(value: i64) -> Expression {
    Expression::literal(LiteralValue::Int(BigInt::from(value)))
}

/// Return the equation `x >= k`.
pub(crate) fn at_least(x: &Identifier, k: i64) -> Constraint {
    Constraint::from(EquationConstraint::new(
        reference(x).greater_equal(literal(k)),
    ))
}

/// Return the equation `x > k`.
pub(crate) fn above(x: &Identifier, k: i64) -> Constraint {
    Constraint::from(EquationConstraint::new(reference(x).greater(literal(k))))
}

/// Return the equation `x <= k`.
pub(crate) fn at_most(x: &Identifier, k: i64) -> Constraint {
    Constraint::from(EquationConstraint::new(reference(x).less_equal(literal(k))))
}

/// Return the equation `x < y`.
pub(crate) fn less_than(x: &Identifier, y: &Identifier) -> Constraint {
    Constraint::from(EquationConstraint::new(reference(x).less(reference(y))))
}

/// Return the set constraint on `x` of `values` with `polarity`.
pub(crate) fn set_constraint(
    x: &Identifier,
    values: impl IntoIterator<Item = Value>,
    polarity: Polarity,
) -> Constraint {
    Constraint::from(SetConstraint::new(x.clone(), member_set(values), polarity))
}

/// Return the in-set constraint on `x` of `values`.
pub(crate) fn in_set(x: &Identifier, values: impl IntoIterator<Item = Value>) -> Constraint {
    set_constraint(x, values, Polarity::In)
}

/// Return the not-in-set constraint on `x` of `values`.
pub(crate) fn not_in_set(x: &Identifier, values: impl IntoIterator<Item = Value>) -> Constraint {
    set_constraint(x, values, Polarity::NotIn)
}

/// Return the values of the integers `values`.
pub(crate) fn ints(values: impl IntoIterator<Item = i64>) -> Vec<Value> {
    values.into_iter().map(super::constraint::int).collect()
}

/// Return a readable text of `member`, for comparing members in tests.
pub(crate) fn describe(member: &Member) -> String {
    match member.kind() {
        MemberKind::Bool(value) => format!("bool:{value}"),
        MemberKind::Int(value) => format!("int:{value}"),
        MemberKind::Float(value) => format!("float:{value}"),
        MemberKind::Str(value) => format!("str:{value}"),
        MemberKind::Tuple(members) => format!(
            "tuple:({})",
            members.iter().map(describe).collect::<Vec<_>>().join(",")
        ),
        MemberKind::FrozenSet(members) => format!(
            "frozenset:{{{}}}",
            members.iter().map(describe).collect::<Vec<_>>().join(",")
        ),
        MemberKind::Opaque(value) => format!("opaque:{}", value.get().ordering_key()),
    }
}

/// Return the readable texts of `members`, in order.
pub(crate) fn describe_all(members: &[Member]) -> Vec<String> {
    members.iter().map(describe).collect()
}

/// A [`Simplifier`] that evaluates ground expressions to a literal and
/// returns any other expression itself, as a simplifier that finds
/// nothing simpler does.
#[derive(Debug)]
pub(crate) struct EvaluatingSimplifier {
    registry: FunctionRegistry,
    inputs: Mutex<Vec<Expression>>,
}

impl EvaluatingSimplifier {
    /// Return the simplifier.
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            registry: FunctionRegistry::new(),
            inputs: Mutex::new(Vec::new()),
        })
    }

    /// Return every input so far, in order.
    pub(crate) fn inputs(&self) -> Vec<Expression> {
        self.inputs
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

impl Simplifier for EvaluatingSimplifier {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed("evaluating")
    }

    fn simplify(
        &self,
        expression: &Expression,
        _context: &SimplifyContext<'_>,
    ) -> Result<Expression, BoxError> {
        self.inputs
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(expression.clone());
        if !expression.free_identifiers().is_empty() {
            return Ok(expression.clone());
        }
        let value = Evaluator::new(&self.registry)
            .evaluate(expression, &HashMap::<Identifier, Scalar>::new())
            .map_err(|error| Box::new(FakeBackendError(error.to_string())) as BoxError)?;
        Ok(Expression::literal(match value {
            Scalar::Bool(value) => LiteralValue::Bool(value),
            Scalar::Int(value) => LiteralValue::Int(BigInt::from(value)),
            Scalar::Real(value) => LiteralValue::Float(value),
        }))
    }
}

/// Return a solver evaluating ground equations with `simplifier` and
/// answering questions with `smt`.
pub(crate) fn build_solver(
    simplifier: &Arc<EvaluatingSimplifier>,
    smt: Option<&Arc<RecordingSmtSolver>>,
) -> Solver {
    let solver =
        Solver::new().with_shared_simplifier(Arc::clone(simplifier) as Arc<dyn Simplifier>);
    match smt {
        Some(smt) => solver.with_shared_smt_solver(Arc::clone(smt) as Arc<dyn SmtSolver>),
        None => solver,
    }
}

/// Return a solver evaluating ground equations and answering every
/// question `answer`, and its SMT backend.
pub(crate) fn scripted_solver(answer: SatResult) -> (Solver, Arc<RecordingSmtSolver>) {
    let smt = RecordingSmtSolver::answering(answer);
    (build_solver(&EvaluatingSimplifier::new(), Some(&smt)), smt)
}

/// Return the SMT backend a test that needs real answers runs on: the z3
/// backend under the `z3` feature, and otherwise the executable
/// `FHY_SMT_SOLVER` names, if any.
#[cfg(feature = "z3")]
#[expect(
    clippy::unnecessary_wraps,
    reason = "without the feature, there may be no backend"
)]
pub(crate) fn real_smt_backend() -> Option<Arc<dyn SmtSolver>> {
    Some(Arc::new(fhy_core::solver::Z3Solver::new()))
}

/// Return the SMT backend a test that needs real answers runs on: the z3
/// backend under the `z3` feature, and otherwise the executable
/// `FHY_SMT_SOLVER` names, if any.
#[cfg(not(feature = "z3"))]
pub(crate) fn real_smt_backend() -> Option<Arc<dyn SmtSolver>> {
    let configured = std::env::var("FHY_SMT_SOLVER").ok()?;
    let mut words = configured.split_whitespace();
    let program = words.next()?;
    Some(Arc::new(
        fhy_core::solver::SmtLib2Process::new(program).with_args(words),
    ))
}

/// Return a solver evaluating ground equations and answering questions
/// with [`real_smt_backend`], if there is one.
pub(crate) fn real_solver() -> Option<Solver> {
    let backend = real_smt_backend()?;
    Some(
        Solver::new()
            .with_shared_simplifier(EvaluatingSimplifier::new() as Arc<dyn Simplifier>)
            .with_shared_smt_solver(backend),
    )
}

/// An owned copy of a [`ParamEvent`].
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum RecordedParamEvent {
    /// A constraint's own event, by its ordering key.
    Member(String, String),
    UndecidedMember(String),
    BridgeFailed(String),
    /// A question's event, by the system's size.
    Question(usize, String),
    Screened(String, Identifier, String),
    EnumerationUndecided(Identifier, Vec<String>),
    SubsetEnumerationUndecided(Identifier, Identifier, Vec<String>),
    SatisfiedOnInexactSystem(Identifier),
    ViolatedUnderKindConflation(Identifier),
    SatisfiabilityUndecided(Identifier),
    ImplicationUndecided(Identifier, Identifier),
    ImplicationDowngraded(Outcome, Identifier, Identifier),
    WitnessOutside(Identifier, usize),
}

/// Return the name of a constraint event's kind.
fn event_name(event: &Event<'_>) -> String {
    match event {
        Event::Unbound { .. } => "unbound",
        Event::SymbolicBinding { .. } => "symbolic_binding",
        Event::BoundNativeConstants { .. } => "bound_native_constants",
        Event::Residual { .. } => "residual",
        Event::Refused { .. } => "refused",
        Event::GaveUp { .. } => "gave_up",
        _ => "other",
    }
    .to_owned()
}

impl RecordedParamEvent {
    /// Return the owned copy of `event`.
    fn of(event: &ParamEvent<'_>) -> Self {
        match *event {
            ParamEvent::Member {
                constraint, event, ..
            } => Self::Member(constraint.ordering_key(), event_name(event)),
            ParamEvent::UndecidedMember { constraint } => {
                Self::UndecidedMember(constraint.ordering_key())
            }
            ParamEvent::BridgeFailed { constraint, .. } => {
                Self::BridgeFailed(constraint.ordering_key())
            }
            ParamEvent::Question { system, event, .. } => {
                Self::Question(system.constraints().len(), event_name(event))
            }
            ParamEvent::Screened {
                constraint,
                variable,
                reason,
            } => Self::Screened(
                constraint.ordering_key(),
                variable.clone(),
                match reason {
                    ScreenReason::DependentScope => "dependent_scope".to_owned(),
                    ScreenReason::ForeignVariable => "foreign_variable".to_owned(),
                    ScreenReason::UnliftableMember(_) => "unliftable_member".to_owned(),
                    ScreenReason::NoLiftableMember => "no_liftable_member".to_owned(),
                    ScreenReason::Narrowed { liftable, excluded } => format!(
                        "narrowed:{:?}:{:?}",
                        describe_all(liftable),
                        describe_all(excluded)
                    ),
                    _ => "other".to_owned(),
                },
            ),
            ParamEvent::EnumerationUndecided {
                variable,
                candidates,
            } => Self::EnumerationUndecided(variable.clone(), describe_all(candidates)),
            ParamEvent::SubsetEnumerationUndecided {
                own,
                other,
                candidates,
            } => Self::SubsetEnumerationUndecided(
                own.clone(),
                other.clone(),
                describe_all(candidates),
            ),
            ParamEvent::SatisfiedOnInexactSystem { variable } => {
                Self::SatisfiedOnInexactSystem(variable.clone())
            }
            ParamEvent::ViolatedUnderKindConflation { variable } => {
                Self::ViolatedUnderKindConflation(variable.clone())
            }
            ParamEvent::SatisfiabilityUndecided { variable } => {
                Self::SatisfiabilityUndecided(variable.clone())
            }
            ParamEvent::ImplicationUndecided { own, other } => {
                Self::ImplicationUndecided(own.clone(), other.clone())
            }
            ParamEvent::ImplicationDowngraded {
                outcome,
                own,
                other,
            } => Self::ImplicationDowngraded(outcome, own.clone(), other.clone()),
            ParamEvent::WitnessOutside {
                variable,
                permitted,
            } => Self::WitnessOutside(variable.clone(), permitted),
            _ => unreachable!("the param tests know every event"),
        }
    }
}

/// A [`ParamObserver`] that records every event, and judges failures
/// undecidable by the default rule unless told to judge none so.
#[derive(Debug, Default)]
pub(crate) struct RecordingParamObserver {
    events: Mutex<Vec<RecordedParamEvent>>,
    is_strict: bool,
}

impl RecordingParamObserver {
    /// Return the observer judging no failure undecidable.
    pub(crate) fn strict() -> Self {
        Self {
            events: Mutex::new(Vec::new()),
            is_strict: true,
        }
    }

    /// Return the events so far, in order.
    pub(crate) fn events(&self) -> Vec<RecordedParamEvent> {
        self.events
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Return the events so far that are not a constraint's own or a
    /// question's, in order.
    pub(crate) fn param_events(&self) -> Vec<RecordedParamEvent> {
        self.events()
            .into_iter()
            .filter(|event| {
                !matches!(
                    event,
                    RecordedParamEvent::Member(..)
                        | RecordedParamEvent::Question(..)
                        | RecordedParamEvent::UndecidedMember(_)
                )
            })
            .collect()
    }
}

impl ParamObserver for RecordingParamObserver {
    fn notify(&self, event: &ParamEvent<'_>) {
        self.events
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(RecordedParamEvent::of(event));
    }

    fn is_undecidable(&self, error: &ConstraintError) -> bool {
        !self.is_strict
            && matches!(
                error,
                ConstraintError::Solve(fhy_core::solver::SolveError::Backend { .. })
            )
    }
}

/// Return the context of `solver` reporting to `observer`.
pub(crate) fn context<'a>(
    solver: &'a Solver,
    observer: &'a RecordingParamObserver,
) -> ParamContext<'a> {
    ParamContext::new(solver).with_observer(observer)
}

/// An opaque value of a test type ordered by its payload: equal to, and
/// ordered against, another of the same type name only.
#[derive(Debug, Clone)]
pub(crate) struct Level {
    pub(crate) type_name: &'static str,
    pub(crate) payload: i64,
}

impl Level {
    /// Return the level `payload` of type `Level`.
    pub(crate) fn value(payload: i64) -> Value {
        Value::Opaque(Opaque::new(Self {
            type_name: "Level",
            payload,
        }))
    }

    /// Return the level `payload` of another type, `Grade`.
    pub(crate) fn grade(payload: i64) -> Value {
        Value::Opaque(Opaque::new(Self {
            type_name: "Grade",
            payload,
        }))
    }
}

impl OpaqueValue for Level {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed(self.type_name)
    }

    fn is_member_shaped(&self) -> bool {
        true
    }

    fn is_equal(&self, other: &dyn OpaqueValue) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| other.type_name == self.type_name && other.payload == self.payload)
    }

    fn check_hashable(&self) -> Result<(), BoxError> {
        Ok(())
    }

    fn ordering_key(&self) -> Cow<'_, str> {
        Cow::Owned(format!("{}:{}", self.type_name, self.payload))
    }

    fn order_against(&self, other: &dyn OpaqueValue) -> Option<Ordering> {
        let other = other.as_any().downcast_ref::<Self>()?;
        (other.type_name == self.type_name).then(|| self.payload.cmp(&other.payload))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// An opaque value whose order is rock-paper-scissors: no total order.
#[derive(Debug, Clone)]
pub(crate) struct Hand(pub(crate) u8);

impl Hand {
    /// Return the hand `index` modulo three.
    pub(crate) fn value(index: u8) -> Value {
        Value::Opaque(Opaque::new(Self(index % 3)))
    }
}

impl OpaqueValue for Hand {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed("Hand")
    }

    fn is_member_shaped(&self) -> bool {
        true
    }

    fn is_equal(&self, other: &dyn OpaqueValue) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| other.0 == self.0)
    }

    fn check_hashable(&self) -> Result<(), BoxError> {
        Ok(())
    }

    fn ordering_key(&self) -> Cow<'_, str> {
        Cow::Owned(format!("Hand:{}", self.0))
    }

    fn order_against(&self, other: &dyn OpaqueValue) -> Option<Ordering> {
        let other = other.as_any().downcast_ref::<Self>()?;
        Some(if self.0 == other.0 {
            Ordering::Equal
        } else if (self.0 + 1) % 3 == other.0 {
            Ordering::Less
        } else {
            Ordering::Greater
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A custom domain of the even integers, recording each hook it is asked,
/// and failing every hook when told to.
#[derive(Debug, Default)]
pub(crate) struct EvenDomain {
    calls: Mutex<Vec<String>>,
    is_failing: bool,
}

impl EvenDomain {
    /// Return the domain, as a [`ParamDomain`], and a handle on its calls.
    pub(crate) fn build(is_failing: bool) -> (ParamDomain, Arc<Self>) {
        let domain = Arc::new(Self {
            calls: Mutex::new(Vec::new()),
            is_failing,
        });
        (
            ParamDomain::Custom(Arc::clone(&domain) as Arc<dyn CustomDomain>),
            domain,
        )
    }

    /// Return the hooks asked so far, in order.
    pub(crate) fn calls(&self) -> Vec<String> {
        self.calls
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Record `hook`, and fail if told to.
    fn record(&self, hook: &str) -> Result<(), BoxError> {
        self.calls
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(hook.to_owned());
        if self.is_failing {
            Err(Box::new(TestValueError(format!("{hook} failed"))))
        } else {
            Ok(())
        }
    }
}

impl CustomDomain for EvenDomain {
    fn symbol_type(&self) -> Result<Option<SymbolType>, BoxError> {
        self.record("symbol_type")?;
        Ok(Some(SymbolType::Int))
    }

    fn is_value_admissible(&self, value: &Value) -> Result<bool, BoxError> {
        self.record("is_value_admissible")?;
        Ok(matches!(value, Value::Int(number) if number % 2 == BigInt::from(0)))
    }

    fn validate_constraint(
        &self,
        _constraint: &Constraint,
        _variable: &Identifier,
    ) -> Result<(), BoxError> {
        self.record("validate_constraint")
    }

    fn implied_constraints(&self, variable: &Identifier) -> Result<Vec<Constraint>, BoxError> {
        self.record("implied_constraints")?;
        Ok(vec![at_least(variable, 0)])
    }

    fn interval_profile(&self) -> Result<Option<IntervalProfile>, BoxError> {
        self.record("interval_profile")?;
        Ok(None)
    }

    fn is_value_set_subset(&self, _other: &ParamDomain) -> Result<bool, BoxError> {
        self.record("is_value_set_subset")?;
        Ok(false)
    }

    fn feasibility_subset(
        &self,
        _own: Side<'_>,
        _other_domain: &ParamDomain,
        _other: Side<'_>,
    ) -> Result<Outcome, BoxError> {
        self.record("feasibility_subset")?;
        Ok(Outcome::Undecided)
    }

    fn has_feasible_value(&self, _side: Side<'_>) -> Result<Outcome, BoxError> {
        self.record("has_feasible_value")?;
        Ok(Outcome::Satisfied)
    }

    fn union(
        &self,
        _own: Side<'_>,
        _other_domain: &ParamDomain,
        _other: Side<'_>,
        _variable: &Identifier,
    ) -> Result<Option<(ParamDomain, Vec<Constraint>)>, BoxError> {
        self.record("union")?;
        Ok(None)
    }

    fn intersection(
        &self,
        _own: Side<'_>,
        _other_domain: &ParamDomain,
        _other: Side<'_>,
        _variable: &Identifier,
    ) -> Result<(ParamDomain, Vec<Constraint>), BoxError> {
        self.record("intersection")?;
        Ok((
            ParamDomain::Custom(Arc::new(Self::default()) as Arc<dyn CustomDomain>),
            Vec::new(),
        ))
    }

    fn is_structurally_equivalent(&self, other: &ParamDomain) -> bool {
        self.record("is_structurally_equivalent")
            .expect("equivalence of a failing domain is not asked");
        matches!(other, ParamDomain::Custom(other) if other.as_any().is::<Self>())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Return the undecided answer of an SMT backend.
pub(crate) fn unknown() -> SatResult {
    SatResult::Unknown {
        reason: "gave up".to_owned(),
    }
}

/// Return a solver whose simplifier fails, as a backend that cannot lower
/// an expression does.
pub(crate) fn failing_simplifier_solver() -> Solver {
    super::solver::RecordingSimplifier::failing("cannot lower").solver()
}
