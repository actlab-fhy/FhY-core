//! Helpers of the search-stream tests: domain builders, oracles that answer
//! from a script, refuse, or record what they were asked, and the spaces
//! the stories search.

use std::collections::VecDeque;
use std::error::Error;
use std::fmt;

use fhy_core::constraint::Value;
use fhy_core::expression::{BigInt, LiteralValue};
use fhy_core::foreign::BoxError;
use fhy_core::identifier::Identifier;
use fhy_core::param::{
    BoundSide, IntegerDomain, Param, ParamContext, ParamDomain, PermutationDomain, Sign,
    ZeroInclusion,
};
use fhy_core::search_space::{
    ChoiceDomain, Configuration, Coordinate, DecisionKind, OrderDomain, PendingStep, Recorder,
    SearchOracle, Space, StepDomain, StridedDomain, StridedRun, Trace,
};
use fhy_core::solver::Solver;
use num_bigint::BigUint;

use super::constraint::int;
use super::search_space::{
    bare_alternative, choice_of, chooses, chosen, condition, forbidden, int_variable,
    plain_alternative, plain_variable,
};

/// Return the kind named `kind`.
///
/// # Panics
///
/// Panics if `kind` is empty.
pub(crate) fn kind(kind: &str) -> DecisionKind {
    DecisionKind::new(kind).expect("the kind has a name")
}

/// Return the choice domain over the integers `values`.
///
/// # Panics
///
/// Panics if the domain is refused.
pub(crate) fn int_choices(values: &[i64]) -> StepDomain {
    StepDomain::from(
        ChoiceDomain::new(values.iter().copied().map(int).collect())
            .expect("the choices are valid"),
    )
}

/// Return the order domain over the identifiers `elements`.
///
/// # Panics
///
/// Panics if the domain is refused.
pub(crate) fn order_of(elements: &[&Identifier]) -> StepDomain {
    StepDomain::from(
        OrderDomain::new(
            elements
                .iter()
                .map(|&element| Value::Identifier(element.clone()))
                .collect(),
        )
        .expect("the elements are valid"),
    )
}

/// Return the strided run from `start` below `stop` by `stride`.
///
/// # Panics
///
/// Panics if the run is refused.
pub(crate) fn run(start: i64, stop: i64, stride: u64) -> StridedRun {
    StridedRun::new(
        BigInt::from(start),
        BigInt::from(stop),
        BigUint::from(stride),
    )
    .expect("the run is valid")
}

/// Return the strided domain of the unit-stride runs `[start, stop)`.
///
/// # Panics
///
/// Panics if the domain is refused.
pub(crate) fn strided(runs: &[(i64, i64)]) -> StepDomain {
    StepDomain::from(
        StridedDomain::new(
            runs.iter()
                .map(|&(start, stop)| run(start, stop, 1))
                .collect(),
        )
        .expect("the runs are valid"),
    )
}

/// Return the coordinate `index`.
pub(crate) const fn index(index: u64) -> Coordinate {
    Coordinate::Index(index)
}

/// Return the order coordinate `positions`.
pub(crate) fn order(positions: &[u32]) -> Coordinate {
    Coordinate::Order(positions.into())
}

/// Return the indices of `coordinates`, which must all be indices.
///
/// # Panics
///
/// Panics on an order coordinate.
pub(crate) fn indices<'a>(coordinates: impl IntoIterator<Item = &'a Coordinate>) -> Vec<u64> {
    coordinates
        .into_iter()
        .map(|coordinate| match coordinate {
            Coordinate::Index(index) => *index,
            Coordinate::Order(_) => panic!("expected an index, got {coordinate:?}"),
        })
        .collect()
}

/// Run `ask` with a context whose solver decides ground equations.
pub(crate) fn with_context<T>(ask: impl FnOnce(&ParamContext<'_>) -> T) -> T {
    let solver = super::search_space::ground_solver();
    ask(&ParamContext::new(&solver))
}

/// What a [`ScriptedOracle`] saw of one step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SeenStep {
    pub(crate) kind: String,
    pub(crate) subject: Identifier,
    pub(crate) position: usize,
    pub(crate) cardinality: BigUint,
    /// The name of a static step's decision.
    pub(crate) decision: Option<Identifier>,
    /// Whether a static step carried the run's configuration.
    pub(crate) has_configuration: bool,
}

/// An oracle answering from a script, in order, recording each step it was
/// asked.
#[derive(Debug, Default)]
pub(crate) struct ScriptedOracle {
    answers: VecDeque<Coordinate>,
    pub(crate) seen: Vec<SeenStep>,
}

impl ScriptedOracle {
    /// Return the oracle answering `answers`, in order.
    pub(crate) fn new(answers: impl IntoIterator<Item = Coordinate>) -> Self {
        Self {
            answers: answers.into_iter().collect(),
            seen: Vec::new(),
        }
    }
}

impl SearchOracle for ScriptedOracle {
    fn decide(&mut self, step: &PendingStep<'_>) -> Result<Coordinate, BoxError> {
        self.seen.push(SeenStep {
            kind: step.kind().as_str().to_owned(),
            subject: step.subject().clone(),
            position: step.position(),
            cardinality: step.domain().cardinality(),
            decision: step.decision().map(|decision| decision.name().clone()),
            has_configuration: step.configuration().is_some(),
        });
        self.answers
            .pop_front()
            .ok_or_else(|| Box::new(ScriptExhausted) as BoxError)
    }
}

/// The error of a [`ScriptedOracle`] asked past its script.
#[derive(Debug)]
pub(crate) struct ScriptExhausted;

impl fmt::Display for ScriptExhausted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("the script has no answer left")
    }
}

impl Error for ScriptExhausted {}

/// The error a [`RefusingOracle`] fails with.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct OracleRefusal;

impl fmt::Display for OracleRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("the oracle refuses")
    }
}

impl Error for OracleRefusal {}

/// An oracle that fails every step.
#[derive(Debug, Default)]
pub(crate) struct RefusingOracle;

impl SearchOracle for RefusingOracle {
    fn decide(&mut self, _step: &PendingStep<'_>) -> Result<Coordinate, BoxError> {
        Err(Box::new(OracleRefusal))
    }
}

/// The names of the tiling space: the top-level variable `t` over `{1, 2}`;
/// the choice `c` among `a`, holding `x` over `{1, 2, 3}`, and `b`, holding
/// nothing; `x` active only while `t` is 1; and no configuration with `t`
/// 2 that chooses `b`.
///
/// Its complete configurations are five: `t = 1, c = a` with each `x`;
/// `t = 1, c = b`; and `t = 2, c = a` with `x` inactive. Its relaxation,
/// dropping the condition and the clause, has eight.
pub(crate) struct TilingSpace {
    pub(crate) space: Space,
    pub(crate) t: Identifier,
    pub(crate) c: Identifier,
    pub(crate) a: Identifier,
    pub(crate) x: Identifier,
    pub(crate) b: Identifier,
}

/// Return the tiling space, its names fresh.
///
/// # Panics
///
/// Panics if the space is refused.
pub(crate) fn build_tiling_space() -> TilingSpace {
    let [name, top, layout, tiled, tile, flat] =
        ["tiling", "t", "c", "a", "x", "b"].map(Identifier::new);
    let choice = choice_of(
        &layout,
        vec![
            plain_alternative(&tiled, vec![int_variable(&tile, &[1, 2, 3])], Vec::new()),
            bare_alternative(&flat),
        ],
    );
    let space = Space::new(
        name,
        vec![int_variable(&top, &[1, 2])],
        vec![choice],
        vec![condition(&tile, [super::param::in_set(&top, [int(1)])])],
        vec![forbidden([
            super::param::in_set(&top, [int(2)]),
            chooses(&layout, &[&flat]),
        ])],
    )
    .expect("the tiling space is valid");
    TilingSpace {
        space,
        t: top,
        c: layout,
        a: tiled,
        x: tile,
        b: flat,
    }
}

/// Return the values the tiling space's five complete configurations give
/// `(t, c, x)`, `x` `None` where it is inactive, in the lexicographic order
/// of their coordinates in decision order.
pub(crate) fn tiling_configurations(tiling: &TilingSpace) -> Vec<(i64, Identifier, Option<i64>)> {
    vec![
        (1, tiling.a.clone(), Some(1)),
        (1, tiling.a.clone(), Some(2)),
        (1, tiling.a.clone(), Some(3)),
        (1, tiling.b.clone(), None),
        (2, tiling.a.clone(), None),
    ]
}

/// Return the entries of the tiling configuration `(t, c, x)`.
pub(crate) fn tiling_entries(
    tiling: &TilingSpace,
    (t, c, x): &(i64, Identifier, Option<i64>),
) -> Vec<(Identifier, Value)> {
    let mut entries = vec![(tiling.t.clone(), int(*t)), (tiling.c.clone(), chosen(c))];
    if let Some(x) = x {
        entries.push((tiling.x.clone(), int(*x)));
    }
    entries
}

/// Record a run of the tiling space: `t = 1`, `c = a`, a dynamic step of
/// the kind `moga.cir.address` about `subject`, over `[0, 64)` and answered
/// `address`, and `x = 2`. Return the run's trace and its complete
/// configuration.
///
/// # Panics
///
/// Panics if an answer is refused.
pub(crate) fn record_tiling_run(
    tiling: &TilingSpace,
    address: u64,
    subject: &Identifier,
) -> (Trace, Configuration) {
    let mut recorder = Recorder::over(&tiling.space);
    let mut oracle = ScriptedOracle::new([index(0), index(0), index(address), index(1)]);
    with_context(|context| {
        recorder.decide(&tiling.t, &mut oracle, context)?;
        recorder.decide(&tiling.c, &mut oracle, context)?;
        recorder.decide_dynamic(
            &kind("moga.cir.address"),
            subject,
            &strided(&[(0, 64)]),
            &mut oracle,
            context,
        )?;
        recorder.decide(&tiling.x, &mut oracle, context)
    })
    .expect("admissible answers");
    let configuration = recorder
        .configuration()
        .expect("a run over a space")
        .clone();
    (recorder.trace(), configuration)
}

/// Return the space `name` of one top-level variable over the natural
/// numbers, unbounded above.
///
/// # Panics
///
/// Panics if the space is refused.
pub(crate) fn unbounded_space(name: &Identifier, variable: &Identifier) -> Space {
    super::search_space::space_of(
        name,
        vec![plain_variable(
            variable,
            super::search_space::natural_param(),
        )],
        Vec::new(),
    )
}

/// Return the param over the integers in `[lower, upper]`, both inclusive.
///
/// # Panics
///
/// Panics if the param is refused.
pub(crate) fn bounded_param(lower: i64, upper: i64) -> Param {
    let solver = Solver::new();
    let context = ParamContext::new(&solver);
    Param::new(
        ParamDomain::from(IntegerDomain::new(Sign::Any, ZeroInclusion::Included)),
        Identifier::new("n"),
        Vec::new(),
        &context,
    )
    .and_then(|param| {
        param.with_bound(
            &LiteralValue::Int(BigInt::from(lower)),
            BoundSide::Lower,
            true,
            &context,
        )
    })
    .and_then(|param| {
        param.with_bound(
            &LiteralValue::Int(BigInt::from(upper)),
            BoundSide::Upper,
            true,
            &context,
        )
    })
    .expect("the bounded param is valid")
}

/// Return the param over the permutations of the identifiers `members`.
///
/// # Panics
///
/// Panics if the param is refused.
pub(crate) fn permutation_param(members: &[&Identifier]) -> Param {
    let solver = Solver::new();
    Param::new(
        ParamDomain::from(
            PermutationDomain::new(members.iter().map(|&member| chosen(member)).collect())
                .expect("the members are distinct"),
        ),
        Identifier::new("order"),
        Vec::new(),
        &ParamContext::new(&solver),
    )
    .expect("the param is valid")
}
