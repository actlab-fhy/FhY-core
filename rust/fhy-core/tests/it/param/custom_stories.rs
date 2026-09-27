//! Stories of a custom domain: the procedures reach it through its hooks,
//! and its failures propagate.

use fhy_core::constraint::{Constraint, Outcome, Value};
use fhy_core::expression::SymbolType;
use fhy_core::foreign::{BoxError, ForeignPart, Part};
use fhy_core::identifier::Identifier;
use fhy_core::param::{
    CustomDomain, IntegerDomain, IntervalProfile, OrdinalDomain, Param, ParamBuildError,
    ParamContext, ParamDomain, ParamError, Side,
};
use fhy_core::param::{Sign, ZeroInclusion};
use fhy_core::solver::SatResult;

use crate::support::constraint::{TestValueError, int};
use crate::support::param::{
    EvenDomain, RecordingParamObserver, at_least, context, in_set, ints, scripted_solver,
};

#[test]
fn custom_domain_answers_through_its_hooks() {
    let x = Identifier::new("x");
    let (domain, handle) = EvenDomain::build(false);
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);

    assert_eq!(
        domain.symbol_type().expect("answers"),
        Some(SymbolType::Int)
    );
    assert!(domain.is_value_admissible(&int(4)).expect("answers"));
    assert!(!domain.is_value_admissible(&int(3)).expect("answers"));
    domain
        .validate_constraint(&at_least(&x, 0), &x)
        .expect("allows");
    assert_eq!(domain.implied_constraints(&x).expect("answers").len(), 1);
    assert_eq!(domain.interval_profile().expect("answers"), None);
    assert_eq!(
        domain
            .has_feasible_value(Side::new(&[], &x), &context)
            .expect("answers"),
        Outcome::Satisfied
    );
    assert!(
        domain
            .union(
                Side::new(&[], &x),
                &domain,
                Side::new(&[], &x),
                &x,
                &context
            )
            .expect("answers")
            .is_none()
    );
    let (twin, _twin_handle) = EvenDomain::build(false);
    assert!(domain.is_structurally_equivalent(&domain));
    assert!(domain.is_structurally_equivalent(&twin));

    assert_eq!(
        handle.calls(),
        [
            "symbol_type",
            "is_value_admissible",
            "is_value_admissible",
            "validate_constraint",
            "implied_constraints",
            "interval_profile",
            "has_feasible_value",
            "union",
            "eq_part",
        ],
        "the same part is equal without asking its hook"
    );
}

#[test]
fn numeric_subset_asks_a_custom_other_side_its_sort_and_its_values() {
    let x = Identifier::new("x");
    let y = Identifier::new("y");
    let (other, handle) = EvenDomain::build(false);
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let own_constraints = [in_set(&x, ints([2, 3]))];

    let outcome = ParamDomain::from(IntegerDomain::new(Sign::Any, ZeroInclusion::Included))
        .feasibility_subset(
            Side::new(&own_constraints, &x),
            &other,
            Side::new(&[], &y),
            &context(&solver, &observer),
        )
        .expect("decides");

    assert_eq!(outcome, Outcome::Violated);
    assert_eq!(
        handle.calls(),
        [
            "symbol_type",
            "implied_constraints",
            "is_value_admissible",
            "is_value_admissible"
        ]
    );
}

#[test]
fn finite_domain_never_asks_a_custom_other_side() {
    let x = Identifier::new("x");
    let (other, handle) = EvenDomain::build(false);
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let own = ParamDomain::from(OrdinalDomain::new(ints([1])).expect("ordinal"));

    let outcome = own
        .feasibility_subset(
            Side::new(&[], &x),
            &other,
            Side::new(&[], &x),
            &context(&solver, &observer),
        )
        .expect("decides");

    assert_eq!(outcome, Outcome::Violated);
    assert!(!own.is_structurally_equivalent(&other));
    assert!(handle.calls().is_empty());
}

#[test]
fn custom_domain_failures_propagate() {
    let x = Identifier::new("x");
    let (domain, _handle) = EvenDomain::build(true);
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);

    assert!(matches!(domain.symbol_type(), Err(ParamError::Custom(_))));
    assert!(matches!(
        domain.is_value_admissible(&int(2)),
        Err(ParamError::Custom(_))
    ));
    assert!(matches!(
        domain.has_feasible_value(Side::new(&[], &x), &context),
        Err(ParamError::Custom(_))
    ));
    assert!(matches!(
        ParamDomain::from(IntegerDomain::new(Sign::Any, ZeroInclusion::Included))
            .is_value_set_subset(&domain, &context),
        Err(ParamError::Custom(_))
    ));
}

// =============================================================================
// A recording domain, driven on either side of the set procedures
// =============================================================================

/// A custom domain that records each hook it is asked, with what it
/// received, and fails every hook when told to.
#[derive(Debug, Default)]
struct RecordingDomain {
    calls: std::sync::Mutex<Vec<String>>,
    is_failing: bool,
}

/// Return a short text of `side`: its variable's name hint and its number
/// of constraints.
fn describe_side(side: Side<'_>) -> String {
    format!(
        "{}/{}",
        side.variable().name_hint(),
        side.constraints().len()
    )
}

impl RecordingDomain {
    /// Return the domain, as a `ParamDomain`, and a handle on its calls.
    fn build(is_failing: bool) -> (ParamDomain, std::sync::Arc<Self>) {
        let domain = std::sync::Arc::new(Self {
            calls: std::sync::Mutex::default(),
            is_failing,
        });
        let handle = std::sync::Arc::clone(&domain);
        (ParamDomain::Custom(Part::from_arc(domain)), handle)
    }

    /// Return the calls so far, and forget them.
    fn take_calls(&self) -> Vec<String> {
        std::mem::take(&mut *self.calls.lock().expect("unpoisoned"))
    }

    /// Record `call`, and fail if told to.
    fn record(&self, call: String) -> Result<(), BoxError> {
        let failure = format!("{call} failed");
        self.calls.lock().expect("unpoisoned").push(call);
        if self.is_failing {
            Err(Box::new(TestValueError(failure)))
        } else {
            Ok(())
        }
    }
}

impl ForeignPart for RecordingDomain {
    fn type_name(&self) -> std::borrow::Cow<'_, str> {
        std::borrow::Cow::Borrowed("RecordingDomain")
    }
}

impl CustomDomain for RecordingDomain {
    fn symbol_type(&self) -> Result<Option<SymbolType>, BoxError> {
        self.record("symbol_type".to_owned())?;
        Ok(Some(SymbolType::Int))
    }

    fn is_value_admissible(&self, _value: &Value) -> Result<bool, BoxError> {
        self.record("is_value_admissible".to_owned())?;
        Ok(true)
    }

    fn validate_constraint(
        &self,
        _constraint: &Constraint,
        variable: &Identifier,
    ) -> Result<(), BoxError> {
        self.record(format!("validate_constraint({})", variable.name_hint()))
    }

    fn implied_constraints(&self, variable: &Identifier) -> Result<Vec<Constraint>, BoxError> {
        self.record(format!("implied_constraints({})", variable.name_hint()))?;
        Ok(Vec::new())
    }

    fn interval_profile(&self) -> Result<Option<IntervalProfile>, BoxError> {
        self.record("interval_profile".to_owned())?;
        Ok(None)
    }

    fn is_value_set_subset(
        &self,
        other: &ParamDomain,
        _context: &ParamContext<'_>,
    ) -> Result<bool, BoxError> {
        self.record(format!("is_value_set_subset({})", other.kind().name()))?;
        Ok(true)
    }

    fn feasibility_subset(
        &self,
        own: Side<'_>,
        other_domain: &ParamDomain,
        other: Side<'_>,
        _context: &ParamContext<'_>,
    ) -> Result<Outcome, BoxError> {
        self.record(format!(
            "feasibility_subset({}, {}, {})",
            describe_side(own),
            other_domain.kind().name(),
            describe_side(other)
        ))?;
        Ok(Outcome::Satisfied)
    }

    fn has_feasible_value(
        &self,
        side: Side<'_>,
        _context: &ParamContext<'_>,
    ) -> Result<Outcome, BoxError> {
        self.record(format!("has_feasible_value({})", describe_side(side)))?;
        Ok(Outcome::Satisfied)
    }

    fn union(
        &self,
        own: Side<'_>,
        other_domain: &ParamDomain,
        other: Side<'_>,
        variable: &Identifier,
        _context: &ParamContext<'_>,
    ) -> Result<Option<(ParamDomain, Vec<Constraint>)>, BoxError> {
        self.record(format!(
            "union({}, {}, {}, {})",
            describe_side(own),
            other_domain.kind().name(),
            describe_side(other),
            variable.name_hint()
        ))?;
        Ok(Some((other_domain.clone(), Vec::new())))
    }

    fn intersection(
        &self,
        own: Side<'_>,
        other_domain: &ParamDomain,
        other: Side<'_>,
        variable: &Identifier,
        _context: &ParamContext<'_>,
    ) -> Result<(ParamDomain, Vec<Constraint>), BoxError> {
        self.record(format!(
            "intersection({}, {}, {}, {})",
            describe_side(own),
            other_domain.kind().name(),
            describe_side(other),
            variable.name_hint()
        ))?;
        Ok((other_domain.clone(), Vec::new()))
    }

    fn eq_part(&self, other: &dyn CustomDomain) -> bool {
        self.calls
            .lock()
            .expect("unpoisoned")
            .push("eq_part".to_owned());
        other.as_any().is::<Self>()
    }
}

/// Return the param over `domain` whose variable is named `name`, with one
/// bound on it.
fn param_over(domain: ParamDomain, name: &str, context: &ParamContext<'_>) -> Param {
    let variable = Identifier::new(name);
    let bound = at_least(&variable, 0);
    Param::new(domain, variable, [bound], context).expect("the domain allows the bound")
}

#[test]
fn a_custom_domain_on_the_left_answers_each_set_procedure_through_its_hook() {
    let (domain, handle) = RecordingDomain::build(false);
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);
    let own = param_over(domain.clone(), "x", &context);
    let other = param_over(
        ParamDomain::from(IntegerDomain::new(Sign::Any, ZeroInclusion::Included)),
        "y",
        &context,
    );
    assert_eq!(
        handle.take_calls(),
        ["validate_constraint(x)", "implied_constraints(x)"]
    );

    assert!(own.is_value_set_subset(&other, &context).expect("answers"));
    assert_eq!(handle.take_calls(), ["is_value_set_subset(integer)"]);

    assert_eq!(
        own.check_subset(&other, &context).expect("decides"),
        Outcome::Satisfied
    );
    assert_eq!(
        handle.take_calls(),
        ["feasibility_subset(x/1, integer, y/1)"]
    );

    let union = own
        .union(&other, Identifier::new("u"), &context)
        .expect("the hook builds the union");
    assert!(union.domain().is_structurally_equivalent(other.domain()));
    assert_eq!(handle.take_calls(), ["union(x/1, integer, y/1, u)"]);

    let intersection = own
        .intersection(&other, Identifier::new("i"), &context)
        .expect("the hook builds the intersection");
    assert!(
        intersection
            .domain()
            .is_structurally_equivalent(other.domain())
    );
    assert_eq!(
        handle.take_calls(),
        ["interval_profile", "intersection(x/1, integer, y/1, i)"],
        "the intersection first asks whether the operands coerce to intervals"
    );

    let (twin, _twin_handle) = RecordingDomain::build(false);
    assert!(own.domain().is_structurally_equivalent(&domain));
    assert!(!own.is_structurally_equivalent(&other));
    assert_eq!(
        handle.take_calls(),
        Vec::<String>::new(),
        "the same part, or a built-in domain, needs no hook"
    );
    assert!(own.domain().is_structurally_equivalent(&twin));
    assert_eq!(handle.take_calls(), ["eq_part"]);
}

#[test]
fn a_custom_domain_on_the_right_is_asked_only_what_the_left_side_needs() {
    let (domain, handle) = RecordingDomain::build(false);
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);
    let integer = param_over(
        ParamDomain::from(IntegerDomain::new(Sign::Any, ZeroInclusion::Included)),
        "x",
        &context,
    );
    let ordinal = Param::new(
        ParamDomain::from(OrdinalDomain::new(ints([1, 2])).expect("ordinal")),
        Identifier::new("x"),
        [],
        &context,
    )
    .expect("an ordinal param");
    let custom = param_over(domain, "y", &context);
    handle.take_calls();

    assert!(
        integer
            .is_value_set_subset(&custom, &context)
            .expect("answers")
    );
    assert_eq!(
        handle.take_calls(),
        ["symbol_type", "implied_constraints(value)"]
    );

    assert!(
        !ordinal
            .is_value_set_subset(&custom, &context)
            .expect("answers")
    );
    assert!(!integer.is_structurally_equivalent(&custom));
    let union = integer.union(&custom, Identifier::new("u"), &context);
    assert!(
        matches!(union, Err(ParamError::UnsupportedUnion(_))),
        "{union:?}"
    );
    let intersection = integer.intersection(&custom, Identifier::new("i"), &context);
    assert!(
        matches!(intersection, Err(ParamError::KindMismatch { .. })),
        "{intersection:?}"
    );
    assert_eq!(
        handle.take_calls(),
        ["interval_profile"],
        "only the intersection asks, whether the right side coerces to an interval"
    );
}

#[test]
fn a_failing_custom_domain_s_error_surfaces_from_each_set_procedure() {
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);
    let other = param_over(
        ParamDomain::from(IntegerDomain::new(Sign::Any, ZeroInclusion::Included)),
        "y",
        &context,
    );
    let (domain, handle) = RecordingDomain::build(true);
    let x = Identifier::new("x");
    let own = Param::new(domain.clone(), x.clone(), [], &context);
    assert!(matches!(own, Err(ParamBuildError::Custom(_))));
    assert_eq!(handle.take_calls(), ["implied_constraints(x)"]);

    let failures = [
        domain
            .is_value_set_subset(other.domain(), &context)
            .map(|_| ()),
        domain
            .feasibility_subset(
                Side::new(&[], &x),
                other.domain(),
                Side::new(other.constraints(), other.variable()),
                &context,
            )
            .map(|_| ()),
        domain
            .union(
                Side::new(&[], &x),
                other.domain(),
                Side::new(&[], &x),
                &x,
                &context,
            )
            .map(|_| ()),
        domain
            .intersection(
                Side::new(&[], &x),
                other.domain(),
                Side::new(&[], &x),
                &x,
                &context,
            )
            .map(|_| ()),
        ParamDomain::from(IntegerDomain::new(Sign::Any, ZeroInclusion::Included))
            .is_value_set_subset(&domain, &context)
            .map(|_| ()),
    ];

    for failure in failures {
        let Err(ParamError::Custom(source)) = failure else {
            panic!("a custom error, got {failure:?}");
        };
        assert!(source.downcast_ref::<TestValueError>().is_some());
        assert!(source.to_string().ends_with(" failed"), "{source}");
    }
    assert_eq!(
        handle.take_calls(),
        [
            "is_value_set_subset(integer)",
            "feasibility_subset(x/0, integer, y/1)",
            "union(x/0, integer, x/0, x)",
            "intersection(x/0, integer, x/0, x)",
            "symbol_type",
        ]
    );
}
