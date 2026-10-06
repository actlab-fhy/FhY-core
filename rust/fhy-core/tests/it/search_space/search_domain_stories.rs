//! Tests for the domain a static step over a decision of a space offers:
//! derived from a choice's alternatives and from a variable's param, or
//! offered by an implementation's `search_domain`; the signature of a
//! static step; and clause 7 of the implementor contract.

use std::borrow::Cow;

use fhy_core::diagnostic::Note;
use fhy_core::foreign::{BoxError, ForeignPart, Part};
use fhy_core::identifier::Identifier;
use fhy_core::param::{OrdinalDomain, Param, ParamContext, ParamDomain};
use fhy_core::search_space::{
    Coordinate, PendingStep, Recorder, SearchOracle, Space, StepDomain, StridedDomain, Variable,
};
use fhy_core::solver::Solver;

use crate::support::constraint::{TestValueError, int};
use crate::support::param::EvenDomain;
use crate::support::search::{
    bounded_param, build_tiling_space, index, permutation_param, run, with_context,
};
use crate::support::search_space::{chosen, int_param, plain_variable, space_of};

/// An oracle answering the first coordinate of every step, keeping each
/// step's domain.
#[derive(Debug, Default)]
struct DomainCollector {
    domains: Vec<StepDomain>,
}

impl SearchOracle for DomainCollector {
    fn decide(&mut self, step: &PendingStep<'_>) -> Result<Coordinate, BoxError> {
        self.domains.push(step.domain().clone());
        Ok(match step.domain() {
            StepDomain::Order(domain) => Coordinate::Order(
                (0..u32::try_from(domain.elements().len()).expect("small")).collect(),
            ),
            _ => Coordinate::Index(0),
        })
    }
}

/// Return the domain the one top-level variable of `param`, named `v`,
/// offers a static step.
fn collect_variable_domain(param: Param) -> StepDomain {
    let v = Identifier::new("v");
    let space = space_of(
        &Identifier::new("one"),
        vec![plain_variable(&v, param)],
        Vec::new(),
    );
    let mut collector = DomainCollector::default();
    let mut recorder = Recorder::over(&space);
    with_context(|context| recorder.decide(&v, &mut collector, context))
        .expect("the first value is admissible");
    collector.domains.pop().expect("the oracle was asked once")
}

/// A variable over a custom domain of even integers that offers its own
/// search domain, or fails to.
#[derive(Debug)]
struct EvenVariable {
    name: Identifier,
    param: Param,
    is_failing: bool,
}

impl EvenVariable {
    /// Return the variable `name` over the even integers.
    fn part(name: &Identifier, is_failing: bool) -> Part<dyn Variable> {
        let solver = Solver::new();
        let (domain, _calls) = EvenDomain::build(false);
        let param = Param::new(
            domain,
            Identifier::new("e"),
            Vec::new(),
            &ParamContext::new(&solver),
        )
        .expect("a custom-domain param");
        Part::new(Self {
            name: name.clone(),
            param,
            is_failing,
        })
    }
}

impl ForeignPart for EvenVariable {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed("EvenVariable")
    }
}

impl Variable for EvenVariable {
    fn kind(&self) -> Cow<'_, str> {
        Cow::Borrowed("test.even_variable")
    }

    fn name(&self) -> &Identifier {
        &self.name
    }

    fn param(&self) -> &Param {
        &self.param
    }

    fn notes(&self) -> &[Note] {
        &[]
    }

    fn search_domain(&self) -> Result<Option<StepDomain>, BoxError> {
        if self.is_failing {
            return Err(Box::new(TestValueError("no domain today".to_owned())));
        }
        Ok(Some(StepDomain::from(
            StridedDomain::new(vec![run(0, 10, 2)]).expect("a valid run"),
        )))
    }
}

/// Test a choice offers its alternatives' names, as identifier values, in
/// declared order.
#[test]
fn static_step_over_a_choice_offers_its_alternatives_names() {
    let tiling = build_tiling_space();
    let mut recorder = Recorder::over(&tiling.space);
    let mut collector = DomainCollector::default();

    with_context(|context| {
        recorder.decide(&tiling.t, &mut collector, context)?;
        recorder.decide(&tiling.c, &mut collector, context)
    })
    .expect("admissible answers");

    let StepDomain::Choice(alternatives) = &collector.domains[1] else {
        panic!("expected a choice domain, got {:?}", collector.domains[1]);
    };
    assert_eq!(
        alternatives.values(),
        [chosen(&tiling.a), chosen(&tiling.b)]
    );
}

/// Test a categorical variable offers its categories, in their canonical
/// order, whatever its constraints refuse.
#[test]
fn static_step_over_a_categorical_variable_offers_its_categories() {
    let domain = collect_variable_domain(int_param(&[3, 1, 2]));

    let StepDomain::Choice(categories) = &domain else {
        panic!("expected a choice domain, got {domain:?}");
    };
    assert_eq!(categories.values(), [int(1), int(2), int(3)]);
}

/// Test an ordinal variable offers its values, ascending.
#[test]
fn static_step_over_an_ordinal_variable_offers_its_values_ascending() {
    let solver = Solver::new();
    let param = Param::new(
        ParamDomain::from(OrdinalDomain::new(vec![int(30), int(10), int(20)]).expect("ordered")),
        Identifier::new("o"),
        Vec::new(),
        &ParamContext::new(&solver),
    )
    .expect("an ordinal param");

    let domain = collect_variable_domain(param);

    let StepDomain::Choice(values) = &domain else {
        panic!("expected a choice domain, got {domain:?}");
    };
    assert_eq!(values.values(), [int(10), int(20), int(30)]);
}

/// Test a permutation variable offers the orderings of its members.
#[test]
fn static_step_over_a_permutation_variable_offers_its_orderings() {
    let [i, j, k] = ["i", "j", "k"].map(Identifier::new);

    let domain = collect_variable_domain(permutation_param(&[&i, &j, &k]));

    let StepDomain::Order(orderings) = &domain else {
        panic!("expected an order domain, got {domain:?}");
    };
    assert_eq!(orderings.elements(), [chosen(&i), chosen(&j), chosen(&k)]);
}

/// Test an integer variable bounded at both ends offers one run over its
/// interval.
#[test]
fn static_step_over_a_bounded_integer_offers_its_interval() {
    let domain = collect_variable_domain(bounded_param(-2, 3));

    let StepDomain::Strided(interval) = &domain else {
        panic!("expected a strided domain, got {domain:?}");
    };
    assert_eq!(interval.runs(), [run(-2, 4, 1)]);
}

/// Test an implementation's search domain is the one a static step offers.
#[test]
fn static_step_over_an_implementation_offers_its_search_domain() {
    let even = Identifier::new("even");
    let space = space_of(
        &Identifier::new("evens"),
        vec![EvenVariable::part(&even, false)],
        Vec::new(),
    );
    let mut recorder = Recorder::over(&space);
    let mut oracle = crate::support::search::ScriptedOracle::new([index(2)]);

    let value = with_context(|context| recorder.decide(&even, &mut oracle, context))
        .expect("an admissible answer");

    assert_eq!(value, int(4));
    assert_eq!(oracle.seen[0].cardinality, num_bigint::BigUint::from(5_u8));
}

/// Test an implementation's failing search domain stops the run as a hook
/// failure.
#[test]
fn static_step_reports_a_failing_search_domain() {
    let even = Identifier::new("even");
    let space = space_of(
        &Identifier::new("evens"),
        vec![EvenVariable::part(&even, true)],
        Vec::new(),
    );
    let mut recorder = Recorder::over(&space);
    let mut oracle = crate::support::search::ScriptedOracle::new([index(0)]);

    let result = with_context(|context| recorder.decide(&even, &mut oracle, context));

    let Err(fhy_core::search_space::TraceError::Hook { decision, source }) = result else {
        panic!("expected a hook failure, got {result:?}");
    };
    assert_eq!(decision, even);
    assert_eq!(
        source.downcast_ref::<TestValueError>(),
        Some(&TestValueError("no domain today".to_owned()))
    );
    assert_eq!(oracle.seen, []);
}

/// Test a static step's signature writes the names its space binds by
/// position, so relabeled copies of a space give equal signatures, unlike a
/// dynamic step over the same names.
#[test]
fn static_step_signature_writes_bound_names_by_position() {
    let record_choice = |tiling: &crate::support::search::TilingSpace| {
        let mut recorder = Recorder::over(&tiling.space);
        let mut oracle = crate::support::search::ScriptedOracle::new([index(0), index(0)]);
        with_context(|context| {
            recorder.decide(&tiling.t, &mut oracle, context)?;
            recorder.decide(&tiling.c, &mut oracle, context)
        })
        .expect("admissible answers");
        recorder.trace().steps()[1].signature().clone()
    };
    let original = build_tiling_space();
    let copy = build_tiling_space();
    let dynamic = StepDomain::from(
        fhy_core::search_space::ChoiceDomain::new(vec![chosen(&original.a), chosen(&original.b)])
            .expect("distinct names"),
    )
    .signature();

    let (left, right) = (record_choice(&original), record_choice(&copy));

    assert_eq!(left, right);
    assert_ne!(left, dynamic);
}

/// Test a space holding a variable over a custom domain with no search
/// domain is valid, and only asking that variable is refused.
#[test]
fn static_step_over_a_custom_domain_without_a_search_domain_is_refused() {
    let even = Identifier::new("even");
    let solver = Solver::new();
    let (domain, _calls) = EvenDomain::build(false);
    let param = Param::new(
        domain,
        Identifier::new("e"),
        Vec::new(),
        &ParamContext::new(&solver),
    )
    .expect("a custom-domain param");
    let space: Space = space_of(
        &Identifier::new("custom"),
        vec![plain_variable(&even, param)],
        Vec::new(),
    );
    let mut recorder = Recorder::over(&space);

    let result = with_context(|context| {
        recorder.decide(
            &even,
            &mut crate::support::search::ScriptedOracle::new([index(0)]),
            context,
        )
    });

    assert!(
        matches!(&result, Err(fhy_core::search_space::TraceError::NotEnumerable { decision }) if *decision == even),
        "{result:?}"
    );
}

/// Test a coordinate past a search domain's last value is outside the
/// step's domain: the step's coordinates index the search domain.
#[test]
fn static_step_values_come_from_the_search_domain() {
    let even = Identifier::new("even");
    let space = space_of(
        &Identifier::new("evens"),
        vec![EvenVariable::part(&even, false)],
        Vec::new(),
    );
    let mut recorder = Recorder::over(&space);
    let mut oracle = crate::support::search::ScriptedOracle::new([index(5)]);

    let result = with_context(|context| recorder.decide(&even, &mut oracle, context));

    assert!(
        matches!(
            result,
            Err(fhy_core::search_space::TraceError::CoordinateOutOfDomain { position: 0 })
        ),
        "{result:?}"
    );
}

// ---------------------------------------------------------------------------
// Clause 7 (behind the `testing` feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "testing")]
mod conformance {
    use std::borrow::Cow;
    use std::sync::atomic::{AtomicBool, Ordering};

    use fhy_core::foreign::{
        BoxError, Foreign, ForeignError, ForeignPart, NoForeign, Part, Resolve,
    };
    use fhy_core::identifier::Identifier;
    use fhy_core::param::Param;
    use fhy_core::search_space::testing::{ContractClause, check_variable_conformance};
    use fhy_core::search_space::{ChoiceDomain, StepDomain, Variable};
    use rstest::rstest;
    use serde::{Deserialize, Serialize};

    use crate::support::constraint::int;
    use crate::support::search_space::int_param;

    const DOMAINED: &str = "test.domained_variable";

    /// How a [`Domained`] variable's search domain relates to its param.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    enum Offer {
        /// Exactly the param's values.
        Faithful,
        /// Fewer values than the param admits.
        Narrower,
        /// A different domain on every call.
        Unstable,
        /// No domain: the param's is derived.
        Derived,
    }

    /// A variable over the integers `{1, 2, 3}` offering a search domain as
    /// its [`Offer`] says.
    #[derive(Debug)]
    struct Domained {
        name: Identifier,
        param: Param,
        offer: Offer,
        flipped: AtomicBool,
    }

    #[derive(Serialize, Deserialize)]
    struct DomainedPayload {
        name: Identifier,
        param: Param,
        offer: Offer,
    }

    impl Domained {
        fn part(offer: Offer) -> Part<dyn Variable> {
            Self::build(Identifier::new("domained"), int_param(&[1, 2, 3]), offer)
        }

        fn build(name: Identifier, param: Param, offer: Offer) -> Part<dyn Variable> {
            Part::new(Self {
                name,
                param,
                offer,
                flipped: AtomicBool::new(false),
            })
        }
    }

    impl ForeignPart for Domained {
        fn type_name(&self) -> Cow<'_, str> {
            Cow::Borrowed("Domained")
        }

        fn to_foreign(&self) -> Result<Foreign, ForeignError> {
            let payload = DomainedPayload {
                name: self.name.clone(),
                param: self.param.clone(),
                offer: self.offer,
            };
            let data = serde_json::to_string(&payload).map_err(|error| ForeignError::Failed {
                type_id: DOMAINED.to_owned(),
                source: Box::new(error),
            })?;
            Ok(Foreign::new(DOMAINED, data))
        }
    }

    impl Variable for Domained {
        fn kind(&self) -> Cow<'_, str> {
            Cow::Borrowed(DOMAINED)
        }

        fn name(&self) -> &Identifier {
            &self.name
        }

        fn param(&self) -> &Param {
            &self.param
        }

        fn search_domain(&self) -> Result<Option<StepDomain>, BoxError> {
            let values = match self.offer {
                Offer::Derived => return Ok(None),
                Offer::Faithful => vec![int(1), int(2), int(3)],
                Offer::Narrower => vec![int(1), int(2)],
                Offer::Unstable => {
                    if self.flipped.fetch_xor(true, Ordering::SeqCst) {
                        vec![int(3), int(2), int(1)]
                    } else {
                        vec![int(1), int(2), int(3)]
                    }
                }
            };
            Ok(Some(StepDomain::from(ChoiceDomain::new(values)?)))
        }
    }

    /// Reads [`Domained`] variables back.
    #[derive(Debug, Clone, Copy)]
    struct DomainedResolver;

    impl Resolve<Part<dyn Variable>> for DomainedResolver {
        fn resolve(&self, foreign: &Foreign) -> Result<Part<dyn Variable>, ForeignError> {
            if foreign.type_id() != DOMAINED {
                return NoForeign.resolve(foreign);
            }
            let payload: DomainedPayload =
                serde_json::from_str(foreign.data()).map_err(|error| ForeignError::Failed {
                    type_id: DOMAINED.to_owned(),
                    source: Box::new(error),
                })?;
            Ok(Domained::build(payload.name, payload.param, payload.offer))
        }
    }

    /// Test a variable offering exactly its param's values, or none, keeps
    /// clause 7.
    #[rstest]
    #[case::faithful(Offer::Faithful)]
    #[case::derived(Offer::Derived)]
    fn a_faithful_or_derived_search_domain_conforms(#[case] offer: Offer) {
        let samples = [Domained::part(offer), Domained::part(offer)];

        let result = check_variable_conformance(&samples, &DomainedResolver);

        assert_eq!(result.map_err(|violation| violation.to_string()), Ok(()));
    }

    /// Test a search domain narrower than the param's, or different on a
    /// second call, breaks clause 7.
    #[rstest]
    #[case::narrower(Offer::Narrower)]
    #[case::unstable(Offer::Unstable)]
    fn an_unfaithful_search_domain_breaks_clause_seven(#[case] offer: Offer) {
        let samples = [Domained::part(offer), Domained::part(offer)];

        let result = check_variable_conformance(&samples, &DomainedResolver);

        let violation = result.expect_err("clause 7 is broken");
        assert_eq!(violation.clause(), ContractClause::SearchDomain);
        assert!(
            violation.to_string().contains("breaks clause 7"),
            "{violation}"
        );
    }
}
