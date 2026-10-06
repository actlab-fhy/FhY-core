//! Tests for `Recorder`: recording dynamic steps, asking the decisions of
//! a space in an order that respects activity, refusing answers outside
//! the domain or not admissible, the oracle's own failures, realizing a
//! configuration, and what a finished run gives; and the `PendingStep` an
//! oracle sees.

use fhy_core::constraint::Value;
use fhy_core::foreign::BoxError;
use fhy_core::identifier::Identifier;
use fhy_core::search_space::{
    Activity, Coordinate, DecisionKind, PendingStep, Recorder, SearchOracle, TraceError,
};
use num_bigint::BigUint;

use crate::support::constraint::int;
use crate::support::param::in_set;
use crate::support::search::{
    OracleRefusal, RefusingOracle, ScriptedOracle, SeenStep, TilingSpace, build_tiling_space,
    index, int_choices, kind, order, order_of, unbounded_space, with_context,
};
use crate::support::search_space::{
    categorical_where, chosen, configure, plain_variable, space_of,
};

/// An oracle answering the first coordinate, recording which coordinates
/// of each step's domain it was told are admissible.
#[derive(Debug, Default)]
struct AdmissionProbe {
    admitted: Vec<Vec<bool>>,
}

impl SearchOracle for AdmissionProbe {
    fn decide(&mut self, step: &PendingStep<'_>) -> Result<Coordinate, BoxError> {
        let count = u64::try_from(step.domain().cardinality()).expect("a small domain");
        let admitted = (0..count)
            .map(|position| step.admits(&Coordinate::Index(position)))
            .collect::<Result<Vec<bool>, _>>()?;
        self.admitted.push(admitted);
        Ok(Coordinate::Index(0))
    }
}

/// Return the recorder over the tiling space, and the space's names.
fn record_tiling() -> (Recorder, TilingSpace) {
    let tiling = build_tiling_space();
    (Recorder::over(&tiling.space), tiling)
}

// ---------------------------------------------------------------------------
// Dynamic steps
// ---------------------------------------------------------------------------

/// Test a dynamic step is put to the oracle and recorded with what was
/// asked and the answer.
#[test]
fn recorder_records_the_step_and_its_answer() {
    let mut recorder = Recorder::new();
    let mut oracle = ScriptedOracle::new([index(1)]);
    let subject = Identifier::new("operand");
    let domain = int_choices(&[10, 20, 30]);

    let answer = with_context(|context| {
        recorder.decide_dynamic(
            &kind("moga.cir.option"),
            &subject,
            &domain,
            &mut oracle,
            context,
        )
    })
    .expect("an admissible answer");

    let trace = recorder.trace();
    assert_eq!(answer, index(1));
    assert_eq!(trace.len(), 1);
    let step = &trace.steps()[0];
    assert_eq!(step.kind().as_str(), "moga.cir.option");
    assert_eq!(step.subject(), &subject);
    assert_eq!(step.decision(), None);
    assert_eq!(step.signature(), &domain.signature());
    assert_eq!(step.value(), Some(&int(20)));
}

/// Test the oracle sees a dynamic step's kind, subject, position and
/// domain, and no decision or configuration.
#[test]
fn recorder_shows_the_oracle_each_dynamic_step() {
    let mut recorder = Recorder::new();
    let mut oracle = ScriptedOracle::new([index(0), index(2)]);
    let subject = Identifier::new("value");

    with_context(|context| {
        recorder.decide_dynamic(
            &kind("a"),
            &subject,
            &int_choices(&[1, 2]),
            &mut oracle,
            context,
        )?;
        recorder.decide_dynamic(
            &kind("b"),
            &subject,
            &int_choices(&[1, 2, 3]),
            &mut oracle,
            context,
        )
    })
    .expect("admissible answers");

    assert_eq!(
        oracle.seen,
        [
            SeenStep {
                kind: "a".to_owned(),
                subject: subject.clone(),
                position: 0,
                cardinality: BigUint::from(2_u8),
                decision: None,
                has_configuration: false,
            },
            SeenStep {
                kind: "b".to_owned(),
                subject,
                position: 1,
                cardinality: BigUint::from(3_u8),
                decision: None,
                has_configuration: false,
            },
        ]
    );
}

/// Test an answer outside the domain stops the run, naming the step, and
/// is not recorded.
#[test]
fn an_answer_outside_the_domain_is_refused_and_not_recorded() {
    let mut recorder = Recorder::new();
    let mut oracle = ScriptedOracle::new([index(0), index(3)]);
    let subject = Identifier::new("s");
    let domain = int_choices(&[10, 20, 30]);

    let result = with_context(|context| {
        recorder.decide_dynamic(&kind("k"), &subject, &domain, &mut oracle, context)?;
        recorder.decide_dynamic(&kind("k"), &subject, &domain, &mut oracle, context)
    });

    assert!(
        matches!(
            result,
            Err(TraceError::CoordinateOutOfDomain { position: 1 })
        ),
        "{result:?}"
    );
    assert_eq!(recorder.trace().len(), 1);
}

/// Test an order answer to an index domain is outside it.
#[test]
fn an_answer_of_the_wrong_shape_is_outside_the_domain() {
    let mut recorder = Recorder::new();
    let mut oracle = ScriptedOracle::new([order(&[0, 1])]);

    let result = with_context(|context| {
        recorder.decide_dynamic(
            &kind("k"),
            &Identifier::new("s"),
            &int_choices(&[1, 2]),
            &mut oracle,
            context,
        )
    });

    assert!(
        matches!(
            result,
            Err(TraceError::CoordinateOutOfDomain { position: 0 })
        ),
        "{result:?}"
    );
    assert!(recorder.trace().is_empty());
}

/// Test an oracle's failure stops the run as `Oracle`, naming the step and
/// carrying the oracle's error.
#[test]
fn an_oracle_failure_stops_the_run_with_its_error() {
    let mut recorder = Recorder::new();
    let mut oracle = RefusingOracle;

    let result = with_context(|context| {
        recorder.decide_dynamic(
            &kind("k"),
            &Identifier::new("s"),
            &int_choices(&[1]),
            &mut oracle,
            context,
        )
    });

    let Err(TraceError::Oracle { position, source }) = result else {
        panic!("expected an oracle failure, got {result:?}");
    };
    assert_eq!(position, 0);
    assert_eq!(source.downcast_ref::<OracleRefusal>(), Some(&OracleRefusal));
}

/// Test an order step takes an order answer.
#[test]
fn recorder_records_an_order_answer() {
    let [i, j, k] = ["i", "j", "k"].map(Identifier::new);
    let domain = order_of(&[&i, &j, &k]);
    let mut recorder = Recorder::new();
    let mut oracle = ScriptedOracle::new([order(&[2, 0, 1])]);

    let answer = with_context(|context| {
        recorder.decide_dynamic(&kind("walk"), &i, &domain, &mut oracle, context)
    })
    .expect("a permutation");

    assert_eq!(answer, order(&[2, 0, 1]));
    assert_eq!(
        recorder.trace().steps()[0].value(),
        Some(&Value::Tuple(vec![
            Value::Identifier(k),
            Value::Identifier(i),
            Value::Identifier(j),
        ]))
    );
}

/// Test a recorder over no space answers no static step.
#[test]
fn recorder_without_a_space_refuses_a_static_step() {
    let mut recorder = Recorder::new();
    let mut oracle = ScriptedOracle::new([index(0)]);

    let result =
        with_context(|context| recorder.decide(&Identifier::new("t"), &mut oracle, context));

    assert!(matches!(result, Err(TraceError::NoSpace)), "{result:?}");
    assert!(oracle.seen.is_empty());
}

// ---------------------------------------------------------------------------
// Static steps
// ---------------------------------------------------------------------------

/// Test a static step answers a variable with its value and a choice with
/// its alternative's name, and grows the configuration.
#[test]
fn recorder_answers_static_steps_and_grows_the_configuration() {
    let (mut recorder, tiling) = record_tiling();
    let mut oracle = ScriptedOracle::new([index(0), index(0), index(2)]);

    let values = with_context(|context| {
        Ok::<_, TraceError>([
            recorder.decide(&tiling.t, &mut oracle, context)?,
            recorder.decide(&tiling.c, &mut oracle, context)?,
            recorder.decide(&tiling.x, &mut oracle, context)?,
        ])
    })
    .expect("admissible answers");

    assert_eq!(values, [int(1), chosen(&tiling.a), int(3)]);
    let configuration = recorder.configuration().expect("a run over a space");
    assert!(configuration.is_complete());
    assert_eq!(configuration.value(&tiling.x), Some(&int(3)));
}

/// Test a static step records its decision's canonical position, its name
/// and its kind: the choice kind for a choice, the variable's own kind.
#[test]
fn recorder_records_static_steps_by_canonical_position() {
    let (mut recorder, tiling) = record_tiling();
    let mut oracle = ScriptedOracle::new([index(0), index(0)]);

    with_context(|context| {
        recorder.decide(&tiling.t, &mut oracle, context)?;
        recorder.decide(&tiling.c, &mut oracle, context)
    })
    .expect("admissible answers");

    let trace = recorder.trace();
    let described: Vec<(Option<usize>, Identifier, String)> = trace
        .steps()
        .iter()
        .map(|step| {
            (
                step.decision(),
                step.subject().clone(),
                step.kind().as_str().to_owned(),
            )
        })
        .collect();
    assert_eq!(
        described,
        [
            (
                Some(0),
                tiling.t.clone(),
                "search_space.variable".to_owned()
            ),
            (Some(1), tiling.c.clone(), DecisionKind::CHOICE.to_owned()),
        ]
    );
}

/// Test the oracle sees a static step's decision and the configuration so
/// far, and positions count the run's steps.
#[test]
fn recorder_shows_the_oracle_each_static_step() {
    let (mut recorder, tiling) = record_tiling();
    let mut oracle = ScriptedOracle::new([index(1), index(0)]);

    with_context(|context| {
        recorder.decide(&tiling.t, &mut oracle, context)?;
        recorder.decide(&tiling.c, &mut oracle, context)
    })
    .expect("admissible answers");

    let seen: Vec<(usize, Option<Identifier>, bool, BigUint)> = oracle
        .seen
        .iter()
        .map(|step| {
            (
                step.position,
                step.decision.clone(),
                step.has_configuration,
                step.cardinality.clone(),
            )
        })
        .collect();
    assert_eq!(
        seen,
        [
            (0, Some(tiling.t.clone()), true, BigUint::from(2_u8)),
            (1, Some(tiling.c.clone()), true, BigUint::from(2_u8)),
        ]
    );
}

/// Test a decision under a choice not yet decided is pending, and is
/// refused without asking the oracle.
#[test]
fn recorder_refuses_a_decision_under_an_undecided_choice_as_pending() {
    let (mut recorder, tiling) = record_tiling();
    let mut oracle = ScriptedOracle::new([index(0)]);

    let result = with_context(|context| recorder.decide(&tiling.x, &mut oracle, context));

    let Err(TraceError::NotActive { name, activity }) = &result else {
        panic!("expected NotActive, got {result:?}");
    };
    assert_eq!(name, &tiling.x);
    assert_eq!(*activity, Activity::Pending);
    assert!(oracle.seen.is_empty());
}

/// Test a decision under an alternative the choice did not choose, or whose
/// condition fails, is inactive and refused.
#[test]
fn recorder_refuses_an_inactive_decision() {
    let (mut by_choice, tiling) = record_tiling();
    let mut by_condition = Recorder::over(&tiling.space);
    let mut oracle = ScriptedOracle::new([index(0), index(1), index(1), index(0)]);

    let results = with_context(|context| {
        by_choice.decide(&tiling.t, &mut oracle, context)?;
        by_choice.decide(&tiling.c, &mut oracle, context)?;
        by_condition.decide(&tiling.t, &mut oracle, context)?;
        by_condition.decide(&tiling.c, &mut oracle, context)?;
        Ok::<_, TraceError>([
            by_choice.decide(&tiling.x, &mut oracle, context),
            by_condition.decide(&tiling.x, &mut oracle, context),
        ])
    })
    .expect("the setup steps are admissible");

    for result in results {
        assert!(
            matches!(
                &result,
                Err(TraceError::NotActive { activity: Activity::Inactive, name }) if *name == tiling.x
            ),
            "{result:?}"
        );
    }
}

/// Test a decision named twice in a run is refused the second time.
#[test]
fn recorder_refuses_a_decision_decided_already() {
    let (mut recorder, tiling) = record_tiling();
    let mut oracle = ScriptedOracle::new([index(0), index(1)]);

    let result = with_context(|context| {
        recorder.decide(&tiling.t, &mut oracle, context)?;
        recorder.decide(&tiling.t, &mut oracle, context)
    });

    assert!(
        matches!(&result, Err(TraceError::AlreadyDecided { name }) if *name == tiling.t),
        "{result:?}"
    );
    assert_eq!(recorder.trace().len(), 1);
}

/// Test a name that is no decision of the space is refused.
#[test]
fn recorder_refuses_an_unknown_decision() {
    let (mut recorder, _tiling) = record_tiling();
    let mut oracle = ScriptedOracle::new([index(0)]);
    let stranger = Identifier::new("stranger");

    let result = with_context(|context| recorder.decide(&stranger, &mut oracle, context));

    assert!(
        matches!(&result, Err(TraceError::UnknownDecision { name }) if *name == stranger),
        "{result:?}"
    );
}

/// Test a value that completes a holding forbidden clause is not
/// admissible: the answer is refused and not recorded.
#[test]
fn recorder_refuses_an_answer_a_forbidden_clause_rules_out() {
    let (mut recorder, tiling) = record_tiling();
    let mut oracle = ScriptedOracle::new([index(1), index(1)]);

    let result = with_context(|context| {
        recorder.decide(&tiling.t, &mut oracle, context)?;
        recorder.decide(&tiling.c, &mut oracle, context)
    });

    assert!(
        matches!(
            &result,
            Err(TraceError::Inadmissible { position: 1, coordinate }) if *coordinate == index(1)
        ),
        "{result:?}"
    );
    assert_eq!(recorder.trace().len(), 1);
    assert_eq!(
        recorder
            .configuration()
            .expect("over a space")
            .value(&tiling.c),
        None
    );
}

/// Test the oracle is told which coordinates are admissible: after `t = 2`
/// the choice may take `a` but not `b`.
#[test]
fn pending_step_admits_only_what_the_configuration_accepts() {
    let (mut recorder, tiling) = record_tiling();
    let mut setup = ScriptedOracle::new([index(1)]);
    let mut probe = AdmissionProbe::default();

    with_context(|context| {
        recorder.decide(&tiling.t, &mut setup, context)?;
        recorder.decide(&tiling.c, &mut probe, context)
    })
    .expect("the probe answers an admissible coordinate");

    assert_eq!(probe.admitted, [vec![true, false]]);
}

/// Test a value the variable's param refuses is not admissible.
#[test]
fn recorder_refuses_a_value_the_param_constraints_refuse() {
    let [name, k] = ["narrow", "k"].map(Identifier::new);
    let param = categorical_where(vec![int(1), int(2)], |p| vec![in_set(p, [int(1)])]);
    let space = space_of(&name, vec![plain_variable(&k, param)], Vec::new());
    let mut recorder = Recorder::over(&space);
    let mut probe = AdmissionProbe::default();
    let mut oracle = ScriptedOracle::new([index(1)]);

    let probed = with_context(|context| recorder.decide(&k, &mut probe, context));
    let mut second = Recorder::over(&space);
    let refused = with_context(|context| second.decide(&k, &mut oracle, context));

    assert_eq!(probed.expect("the first value is admissible"), int(1));
    assert_eq!(probe.admitted, [vec![true, false]]);
    assert!(
        matches!(refused, Err(TraceError::Inadmissible { position: 0, .. })),
        "{refused:?}"
    );
}

/// Test a variable whose domain is not finite cannot be asked.
#[test]
fn recorder_refuses_a_variable_without_a_finite_domain() {
    let [name, n] = ["counting", "n"].map(Identifier::new);
    let space = unbounded_space(&name, &n);
    let mut recorder = Recorder::over(&space);
    let mut oracle = ScriptedOracle::new([index(0)]);

    let result = with_context(|context| recorder.decide(&n, &mut oracle, context));

    assert!(
        matches!(&result, Err(TraceError::NotEnumerable { decision }) if *decision == n),
        "{result:?}"
    );
    assert!(oracle.seen.is_empty());
}

/// Test a failed step leaves the earlier steps in the trace.
#[test]
fn recorder_keeps_the_prefix_after_a_failed_step() {
    let (mut recorder, tiling) = record_tiling();
    let mut oracle = ScriptedOracle::new([index(0), index(0)]);
    let mut refusing = RefusingOracle;

    let result = with_context(|context| {
        recorder.decide(&tiling.t, &mut oracle, context)?;
        recorder.decide(&tiling.c, &mut oracle, context)?;
        recorder.decide(&tiling.x, &mut refusing, context)
    });

    assert!(
        matches!(result, Err(TraceError::Oracle { position: 2, .. })),
        "{result:?}"
    );
    assert_eq!(
        recorder.trace().coordinates().cloned().collect::<Vec<_>>(),
        [index(0), index(0)]
    );
}

/// Test a finished run gives its trace and its configuration, complete or
/// not.
#[test]
fn recorder_finish_gives_the_trace_and_the_configuration() {
    let (mut recorder, tiling) = record_tiling();
    let mut oracle = ScriptedOracle::new([index(0)]);
    with_context(|context| recorder.decide(&tiling.t, &mut oracle, context))
        .expect("an admissible answer");
    let trace = recorder.trace();

    let recorded = recorder.finish().expect("a run over a space finishes");

    assert_eq!(recorded.trace(), &trace);
    let configuration = recorded.configuration().expect("a run over a space");
    assert!(!configuration.is_complete());
    assert_eq!(configuration.value(&tiling.t), Some(&int(1)));
    let (into_trace, into_configuration) = recorded.into_parts();
    assert_eq!(into_trace, trace);
    assert!(into_configuration.is_some());
}

/// Test a run of dynamic steps finishes with no configuration.
#[test]
fn recorder_of_dynamic_steps_finishes_without_a_configuration() {
    let mut recorder = Recorder::new();
    let mut oracle = ScriptedOracle::new([index(0)]);
    with_context(|context| {
        recorder.decide_dynamic(
            &kind("k"),
            &Identifier::new("s"),
            &int_choices(&[4]),
            &mut oracle,
            context,
        )
    })
    .expect("an admissible answer");

    let recorded = recorder.finish().expect("a dynamic run finishes");

    assert_eq!(recorded.trace().len(), 1);
    assert!(recorded.configuration().is_none());
}

// ---------------------------------------------------------------------------
// Realizing a configuration
// ---------------------------------------------------------------------------

/// Test a realizing recorder answers the decisions its configuration
/// assigns from it, without asking the oracle, and records their
/// coordinates.
#[test]
fn realizing_answers_assigned_decisions_from_the_configuration() {
    let tiling = build_tiling_space();
    let configuration = configure(
        &tiling.space,
        [
            (tiling.t.clone(), int(1)),
            (tiling.c.clone(), chosen(&tiling.a)),
            (tiling.x.clone(), int(2)),
        ],
    );
    let mut recorder = Recorder::realizing(&configuration);
    let mut refusing = RefusingOracle;

    let values = with_context(|context| {
        Ok::<_, TraceError>([
            recorder.decide(&tiling.t, &mut refusing, context)?,
            recorder.decide(&tiling.c, &mut refusing, context)?,
            recorder.decide(&tiling.x, &mut refusing, context)?,
        ])
    })
    .expect("every decision is answered from the configuration");

    assert_eq!(values, [int(1), chosen(&tiling.a), int(2)]);
    let recorded = recorder
        .finish()
        .expect("every assigned decision was asked");
    assert_eq!(
        recorded.trace().coordinates().cloned().collect::<Vec<_>>(),
        [index(0), index(0), index(1)]
    );
    assert_eq!(
        recorded
            .configuration()
            .map(fhy_core::search_space::Configuration::key),
        Some(configuration.key())
    );
}

/// Test a realizing recorder asks the oracle for a decision its
/// configuration leaves unassigned, and for dynamic steps.
#[test]
fn realizing_asks_the_oracle_for_unassigned_decisions() {
    let tiling = build_tiling_space();
    let configuration = configure(&tiling.space, [(tiling.t.clone(), int(1))]);
    let mut recorder = Recorder::realizing(&configuration);
    let mut oracle = ScriptedOracle::new([index(1), index(0)]);

    let (choice, dynamic) = with_context(|context| {
        recorder.decide(&tiling.t, &mut oracle, context)?;
        let choice = recorder.decide(&tiling.c, &mut oracle, context)?;
        let dynamic = recorder.decide_dynamic(
            &kind("moga.cir.address"),
            &Identifier::new("value"),
            &int_choices(&[0, 8]),
            &mut oracle,
            context,
        )?;
        Ok::<_, TraceError>((choice, dynamic))
    })
    .expect("admissible answers");

    assert_eq!(choice, chosen(&tiling.b));
    assert_eq!(dynamic, index(0));
    assert_eq!(oracle.seen.len(), 2);
}

/// Test a realizing recorder refuses to finish while a decision its
/// configuration assigns was never asked, naming those decisions in
/// canonical order.
#[test]
fn realizing_refuses_to_finish_with_unasked_decisions() {
    let tiling = build_tiling_space();
    let configuration = configure(
        &tiling.space,
        [
            (tiling.t.clone(), int(1)),
            (tiling.c.clone(), chosen(&tiling.a)),
            (tiling.x.clone(), int(3)),
        ],
    );
    let mut recorder = Recorder::realizing(&configuration);
    let mut refusing = RefusingOracle;
    with_context(|context| recorder.decide(&tiling.t, &mut refusing, context))
        .expect("answered from the configuration");

    let result = recorder.finish();

    let Err(TraceError::Unasked { decisions }) = result else {
        panic!("expected Unasked, got {result:?}");
    };
    assert_eq!(decisions, [tiling.c.clone(), tiling.x.clone()]);
}

/// Test `PendingStep::dynamic` builds a step an oracle can be asked outside
/// a recorder, every coordinate of its domain admissible.
#[test]
fn pending_step_dynamic_admits_every_coordinate_of_its_domain() {
    let domain = int_choices(&[5, 6, 7]);
    let step_kind = kind("k");
    let subject = Identifier::new("s");

    let admitted = with_context(|context| {
        let step = PendingStep::dynamic(&step_kind, &subject, &domain, 4, context);
        assert_eq!(step.position(), 4);
        assert!(step.decision().is_none());
        assert!(step.configuration().is_none());
        [index(0), index(2), index(3)].map(|coordinate| step.admits(&coordinate))
    });

    assert!(
        matches!(admitted, [Ok(true), Ok(true), Ok(false)]),
        "{admitted:?}"
    );
}

/// Test `PendingStep::of_decision` builds a static step over a
/// configuration's decision, admitting what the configuration accepts, and
/// none for a name its space lacks.
#[test]
fn pending_step_of_decision_admits_what_the_configuration_accepts() {
    let tiling = build_tiling_space();
    let configuration = configure(&tiling.space, [(tiling.t.clone(), int(2))]);
    let alternatives = fhy_core::search_space::StepDomain::from(
        fhy_core::search_space::ChoiceDomain::new(vec![chosen(&tiling.a), chosen(&tiling.b)])
            .expect("distinct names"),
    );
    let choice_kind = DecisionKind::choice();

    let (admitted, missing) = with_context(|context| {
        let step = PendingStep::of_decision(
            &choice_kind,
            &configuration,
            &tiling.c,
            &alternatives,
            1,
            context,
        )
        .expect("the space has the choice");
        let missing = PendingStep::of_decision(
            &choice_kind,
            &configuration,
            &Identifier::new("stranger"),
            &alternatives,
            1,
            context,
        )
        .is_none();
        (
            [index(0), index(1)].map(|coordinate| step.admits(&coordinate)),
            missing,
        )
    });

    assert!(matches!(admitted, [Ok(true), Ok(false)]), "{admitted:?}");
    assert!(missing);
}

/// Test a pending step displays its kind, subject and size.
#[test]
fn pending_step_displays_its_kind_subject_and_size() {
    let domain = int_choices(&[5, 6, 7]);
    let step_kind = kind("moga.cir.tile");
    let subject = Identifier::new("tile");

    let text = with_context(|context| {
        PendingStep::dynamic(&step_kind, &subject, &domain, 0, context).to_string()
    });

    assert!(text.starts_with("moga.cir.tile step for "), "{text}");
    assert!(text.ends_with(" over 3 value(s)"), "{text}");
}
