# What a Rust MOGA-VM needs from fhy-core

## Summary

A gap analysis of fhy-core 0.2.0-dev against MOGA-VM (about 68k lines of
Python) found six gaps and six smaller helpers that a Rust MOGA-VM needs.
Each was reproduced in the sketch crate `moga_kinds`, which implements
MOGA-VM's CIR kinds (knobs, realization options, tile specs) against
`fhy_core::search_space` in pure Rust. This plan closes all of them in one
pass:

| Item | What | Kind of change |
|---|---|---|
| G1 | `Space::mutate` is cubic in the number of decisions | behaviour (performance) |
| G2 | `cardinality` counts values that sampling refuses under a solver without a simplifier | behaviour (param and search space) |
| G3 | no pure-Rust analysis that cancels free identifiers, such as `(3*S + T) - T` | new API: `AffineForm`, `Rational` |
| G4 | `Provenance` is closed, and Python cannot subclass it on 0.2.0 | new API, breaking |
| G5 | measurements are keyed only by configuration | new API, breaking |
| G6 | a categorical domain refuses tuple categories | behaviour (param) |
| H1 | a repairing replay oracle, and crossover built on it | new API |
| H2 | the Pareto front of measurements | new API |
| H3 | composing several crates' resolvers without hand dispatch | new API (Rust only) |
| H4 | editing a space, and completing a configuration | new API |
| H5 | per-choice completeness; cheap one-entry-at-a-time building | new API and behaviour |
| H6 | the implementor contract's "one kind per type" | docs |

The interface stub is committed beside this document. Bodies are
`todo!()`. G1, G2, G6 and the cheaper check in H5 change behaviour only, so
their stub is the updated rustdoc.

## Motivation

The sketch crate and MOGA-VM's sources show what callers cannot do today:

- **G1.** On the sketch's candidate table (one choice per entry, 4
  realization options of 4 knobs each), a release build mutates a
  configuration of 10 entries in 28 ms, 50 entries in 1.27 s and 100 in
  9.3 s. A MOGA-VM table holds hundreds of entries.
- **G2.** A byte-address knob is a natural param bounded `[16, 4095]`.
  Under `Solver::new()` its space's `cardinality` is `Exact(4080)`, while
  `sample` stops with `DeadEnd` and `Configuration::new` refuses every value
  ("cannot be assigned to its param").
- **G3.** `walk_grammar._affine_coefficient` (MOGA-VM
  `cir/checks/walk_grammar.py:652-686`) infers a stride by substituting
  0, 1 and 2 for an index symbol and letting SymPy cancel the difference,
  as in `(3*1 + T) - (3*0 + T)` → `3`, with `T` free. `tiling.py`'s
  `_axis_coefficient` and `walk_geometry.py` do the same through
  `simplify_expression`. The coefficients it needs are integers; offsets
  use `+`, `-`, `*` by a literal, and negation.
- **G4.** `EdgePropagationProvenance` (MOGA-VM `cir/ir/provenance.py:29-61`)
  is a frozen dataclass subclass of `Provenance` with one field, `edge_id`,
  registered as `moga.cir.provenance.edge_propagation`. On 0.2.0-dev,
  constructing it raises `TypeError: object.__new__(EP) is not safe, use
  fhy_core._rs.Provenance.__new__()`, because `_rs.Provenance` has no
  `#[new]` and PyO3 sets `Py_TPFLAGS_DISALLOW_INSTANTIATION`. In Rust there
  is no way to define such a provenance at all.
- **G5.** MOGA-VM's point identity is the coordinates of every step in ask
  order, the dynamic address and boundary-namespace steps included
  (`decisions.py:552-561`); two runs differing only in where an allocator
  put a buffer are different points with different measurements. A
  `ConfigurationKey` cannot tell them apart.
- **G6.** MOGA-VM's tile knob is a categorical param over tile shapes, a
  flat tuple of integers per index symbol (`builder/candidates/knobs.py:
  205-238`). `CategoricalDomain::new` refuses `Value::Tuple` with
  `NotALeafValue`, so the sketch wraps each shape in an opaque value whose
  order is a string.
- **H1–H6.** The sketch hand-writes a resolver that dispatches on type ids
  and carries its own `ParamContext`, rebuilds a space with `Space::new` per
  added choice (quadratic), completes a transplanted configuration with a
  hand loop over `Recorder::realizing`, and has no way to repair a crossed
  point. MOGA-VM's harness expects "a crossover in a conditional space"
  to repair "inside its own oracle" (`harness.py:35-37`), and its
  `ExtractedPointOracle` already repairs by answering what it can and
  falling back to a tail oracle.

## Crate and module placement

Everything lands in the existing crates and modules; no new crate and no
new dependency.

| Item | `fhy-core` | `fhy-core-py` | Python |
|---|---|---|---|
| G1 | `search_space::exploration`, `counting`, `configuration` | none | none |
| G2 | `param::parameter`, `param::decide`; `search_space::step` | none | none |
| G3 | `expression::affine` (new, private), `expression::literal::exact` | `expression/affine.rs` (new) | `fhy_core.symbolic.expression.passes.affine` (new) |
| G4 | `provenance`, `provenance::wire` (new) | `provenance.rs` | `fhy_core.provenance` (no new names) |
| G5 | `search_space::trace`, `measurement`, `wire` | `search_space/trace.rs`, `measurement.rs` | `fhy_core.search_space` |
| G6 | `param::domain` | `param/value.rs` | none |
| H1 | `search_space::oracle`, `exploration` | `search_space/oracle.rs`, `space.rs` | `fhy_core.search_space` |
| H2 | `search_space::measurement` | `search_space/measurement.rs` | `fhy_core.search_space` |
| H3 | `search_space::wire`, `error` | none | none |
| H4 | `search_space::space`, `exploration`, `error` | `search_space/space.rs` | none (methods) |
| H5 | `search_space::configuration` | `search_space/configuration.rs` | none (method) |
| H6 | `search_space` module docs | none | none |

Layering (CONTRIBUTING, "Module paths follow Rust layering") holds:
`provenance` (layer 2) uses only `foreign` (layer 1); `expression::affine`
uses only `expression` and `identifier`; everything else stays inside
`search_space` (layer 11), which may read `param`, `constraint` and
`expression`.

**`num-rational` is not added.** The crate already has an exact rational,
`expression::literal::exact::Rational` (crate-private, `BigInt` numerator
and positive denominator in lowest terms, `Ord` by cross-multiplication),
which the solver lowering, the param bounds and the ordinal order share.
G3 makes it public instead of adding a second rational type and a
dependency that `deny.toml` would have to admit.

**Python alignment.** One row joins the table in CONTRIBUTING:

| Python | Rust |
|---|---|
| `fhy_core.symbolic.expression.passes.affine` | `fhy_core::expression` (`AffineForm`, `Expression::affine_form`, `Rational`) |

`TraceKey`, `GuidedOracle` and `non_dominated` fall under the existing
`fhy_core.search_space` ↔ `fhy_core::search_space` row.

## Public API

Every item below is in the stub with its rustdoc; the signatures here match
it. "Errors" lists the `Err` cases; nothing panics on caller input.

### G1 `Space::mutate` (no API change)

Root cause, measured against the source
(`search_space/exploration.rs`):

1. `find_candidates` visits every assigned decision `d` (n of them) and
   every other coordinate `v` of its domain (k; 1 to 3 on the table), and
   asks `completes(previous ∪ {d: v})`.
2. `completes` runs up to `COMPLETION_BUDGET = 1024` walks of the **whole
   space** with an `ExhaustiveOracle`; on the table the first walk
   succeeds, so it is one walk of n steps.
3. Each step calls `try_extend`, which calls `Configuration::extended`,
   which runs `Checker::run` over **every** decision in decision order and
   every forbidden clause, and clones the value vector: O(n) per step.

So one mutation costs O(k · n · n · n) = O(k n³). The timings fit: from 50
to 100 entries (n doubles), 1.27 s → 9.3 s is ×7.3 against ×8.

The fix keeps the documented semantics (uniform over the decisions that may
change, then uniform over their other values that complete, then repair),
and changes how "may change" is found:

1. **Closed-form components need no search.** `counting::find_components`
   already splits a space into components linked by conditions and
   forbidden clauses, and `is_closed_form` says which have no condition, no
   forbidden clause and no param constraint beyond an integer variable's
   bounds. In such a component every value of a variable's step domain is
   admissible (with G2's exact bound check), and no condition or clause
   reaches outside it. So, for a decision there:
   - a variable may take **every** other coordinate of its domain;
   - a choice may take each other alternative whose subtree admits a
     completion: the product of its decisions' relaxed counts
     (`count_relaxed` with `EmptyVariable::Empty`) is not zero.
   The other components keep the values of the current, complete
   configuration, which already complete them.
2. **Linked components search only themselves.** `completes` walks only
   the component's members (`walk(space, |p| component.members[p], ..)`,
   as `count_by_enumeration` does) with only the component's earlier
   decisions pinned. The other components' answers do not change, since no
   condition or clause links them.
3. **A step costs what it changes.** `extended` (and `with_entries`, H5)
   re-derive activity only for the decisions that depend on the changed one
   (its subtree, and transitively the targets of conditions naming it, in
   decision order) and re-check only the forbidden clauses naming a
   decision whose value or activity changed. `Space` precomputes, per
   decision, its dependents (crate-private). The value vector is still
   copied per step (O(n) memcpy, no checks).
4. **Options are not materialized.** A decision's other values are kept as
   "every coordinate but the current one" (an index count, or the
   `m(m-1)/2` swaps of an ordering) and the `i`-th is computed in the order
   `list_neighbours` lists them today, so a 65 536-value domain costs no
   list.

Cost after: O(n) to find candidates on a closed-form space, plus one repair
walk of O(n · (affected + n memcpy)). Target: a 200-entry table (about
1 000 decisions) mutates in under 100 ms in a release build.

**Seeded outputs.** The candidate list keeps its order (decision order) and
each decision's options keep their order, and the random draws are the
same (`rng.below(candidates)`, `rng.below(options)`, then the repair's
uniform draws). So a seeded mutation returns the same configuration as
today whenever today's search answered exactly. It differs only where the
old search guessed: a value whose 1 024-run search ran out was taken as
completable, and the closed form now answers exactly (for example an
alternative holding eleven Boolean variables and then a variable with an
empty domain: 2 048 paths, so the old search guessed yes; the closed form
says no). I checked every place that runs `mutate` with a seed: the Rust
stories (`exploration_stories.rs`, `trace_properties.rs`,
`stream_error_text_stories.rs`), the Python tests (`test_trace.py`,
`test_trace_properties.py`) and the benchmark
(`test_option_space_mutate`) assert properties (complete, one decision
changed, replays), never a seeded result, and no golden corpus records a
mutation. So no pinned output changes; this is not on the
Needs-the-user list beyond a note.

**Docs changed in the stub:** `Space::mutate` now says how the decisions
that may change are found per component.

### G2 `cardinality` against sampling (no new API)

Root cause, traced through the source:

1. `Param::check_value` evaluates each constraint with
   `param::decide::evaluate_constraints`, which calls
   `Equation::evaluate`, which always asks `context.solver().simplify(..)`,
   even for a ground comparison such as `2955 >= 16` once the variable is
   bound.
2. `Solver::new()` holds no simplifier, so `simplify` fails with
   `SolveError::NoCapableBackend(Simplification)`.
3. The default `ParamObserver::is_undecidable` counts only
   `SolveError::Backend` as undecided, so the error propagates as
   `AssignmentError::Constraint(..)`. `Configuration::new` reports it as
   `ConfigurationError::Assignment` ("cannot be assigned to its param").
4. `search_space::step::try_extend` treats **every**
   `ConfigurationError::Assignment` as "not admissible", whatever the
   `AssignmentError` is. So the run sees each value refused and ends in
   `DeadEnd`, instead of reporting that nothing could decide the value.
5. `cardinality` never asks the solver: `is_closed_form` and
   `step::interval_domain` decode the bounds syntactically
   (`effective_interval`) and count the strided run.

Two procedures decide membership: counting decodes bounds; construction
asks a simplifier. The solver behaves as designed: a `Solver` is a holder
of pluggable backends, and the ground simplifier is opt-in
(CONTRIBUTING, "The ground simplifier"); it is not a solver bug.

Options weighed:

- (a) Count through the same admissibility check: consistent, but the count
  becomes `Exact(0)` under `Solver::new()` (useless) and every closed form
  turns into an enumeration (slow).
- (b) Decide bound members exactly in the param: the param already owns
  bounds as a concept (`with_bound`, `effective_interval`,
  `IntervalIntegerDomain`), and comparing an integer value with an integer
  literal is exact, so the answer is the one every correct simplifier
  gives.
- (c) Require a ground-deciding context and document it: the disagreement
  stays.

**Chosen: (b), plus the error fix in `try_extend`.**

- `Param::evaluate_constraints`, and so `Param::check_value`,
  `ParamAssignment::new`/`restore` and `Configuration::new`, decide a
  member that `decode_bound` reads as a bound of the param's own variable
  (`x >= c`, `x > c`, `x <= c`, `x < c`, `c` an integer literal, either
  side), while the environment binds the variable to `Value::Int`, by
  comparing the integers: no solver call and no `ParamEvent::Member` for
  that member. Every other member goes to the solver as today. The free
  function `param::evaluate_constraints` is unchanged (it does not know
  the variable).
- `try_extend` counts as inadmissible only `AssignmentError::Inadmissible`,
  `ViolatedConstraint` and `UnverifiedConstraint` (plus `Forbidden`, as
  today). `AssignmentError::Constraint`, `Custom` and
  `BindingsBindVariable` become `TraceError::Configuration`, so a context
  that cannot decide a non-bound constraint stops the run with that
  error instead of `DeadEnd`. The same holds for `count_by_enumeration`,
  which then reports the error instead of a zero count.

After the fix, the reproduction gives `Exact(4080)` and `sample` draws a
value in `[16, 4095]` under `Solver::new()`. A param with a non-bound
constraint (`x % 4 == 0`) under `Solver::new()` fails `cardinality` and
`sample` alike, with the same `TraceError::Configuration`.

This changes param semantics beyond the search space (an assignment that
used to fail without a simplifier now succeeds, and its bound members no
longer report member events), so it is on the Needs-the-user list (N2).

**Docs changed in the stub:** `Param::evaluate_constraints`.

### G3 `AffineForm` and `Rational` (`fhy_core::expression`)

```rust
pub struct Rational { /* numerator: BigInt, denominator: BigInt */ }
impl Rational {
    pub fn new(numerator: BigInt, denominator: BigInt) -> Option<Self>; // None for a zero denominator
    pub fn numerator(&self) -> &BigInt;      // of the rational's sign
    pub fn denominator(&self) -> &BigInt;    // positive
    pub fn is_integer(&self) -> bool;
    pub fn to_integer(&self) -> Option<&BigInt>;
    pub fn is_zero(&self) -> bool;
}
impl From<BigInt> for Rational;
impl fmt::Display for Rational;   // "3", "-1/2"
// Debug, Clone, PartialEq, Eq, Hash (derived); PartialOrd, Ord (numeric); Neg

pub struct AffineForm { /* terms: BTreeMap<id-ordered Identifier, Rational>, constant: Rational */ }
impl AffineForm {
    pub fn coefficient(&self, identifier: &Identifier) -> Rational; // zero when absent
    pub fn constant(&self) -> &Rational;
    pub fn terms(&self) -> impl ExactSizeIterator<Item = (&Identifier, &Rational)> + '_; // by id
    pub fn is_constant(&self) -> bool;
    pub fn to_expression(&self) -> Expression;
}
impl fmt::Display for AffineForm; // the canonical expression's text
// Debug, Clone, PartialEq, Eq, Hash

impl Expression {
    pub fn affine_form(&self) -> Option<AffineForm>;
}
```

Contract of `affine_form`: it reads bottom-up an integer or decimal
literal (its exact rational), an identifier (coefficient 1, any sort),
negation and unary plus, `+` and `-`, `*` when one side is constant, `/` by
a non-zero constant, and `//`, `%` and `**` when both operands are constant
(folded exactly as the ground simplifier folds them; `**` only with an
integer exponent, never `0 ** negative`). It answers `None` for a float or
Boolean literal, a comparison, a logical operation, a piecewise, a call, a
product of two non-constant forms, a division by a non-constant form or by
zero, `//`, `%` or `**` with a non-constant operand, a tree nested more
than 256 levels (the ground simplifier's bound), and a coefficient or
constant whose numerator or denominator would exceed 4 096 bits (the
ground simplifier's bound on fractions). Cancelled terms are dropped.

**Integer or rational coefficients.** MOGA-VM needs integers (its
`_literal_int` refuses anything else). The coefficients are rational
anyway: `x / 2 + x / 2` is affine with coefficient 1, and an
integer-only analysis would have to decline it or round; a rational keeps
the analysis closed under every operation it reads, and the caller asks
`to_integer()`. Since the crate already has the type, it costs no
dependency.

`to_expression` builds `c * x` terms in order of the identifiers' ids
(`x` for 1, `-x` for -1), then the constant when it is non-zero or there
is no term; a non-integer coefficient is the quotient of two integer
literals.

**Python** (`fhy_core.symbolic.expression.passes.affine`, re-exported from
`fhy_core.symbolic.expression`):

```python
class AffineForm:                     # _rs.AffineForm, frozen, no constructor
    def coefficient(self, identifier: Identifier) -> Fraction: ...
    @property
    def constant(self) -> Fraction: ...
    @property
    def terms(self) -> dict[Identifier, Fraction]: ...   # ordered by id
    def is_constant(self) -> bool: ...
    def to_expression(self) -> Expression: ...
    # __eq__/__hash__ structural, __str__ the canonical text

def affine_form(expression: Expression) -> AffineForm | None: ...
```

It is a function in `passes/`, not a method of `Expression`, because the
methods of `Expression` are the `Term` protocol and every derived analysis
is a module function (`fold_expression`, `inline_functions`,
`evaluate_expression`). Coefficients are `fractions.Fraction`. Raises
`TypeError` for an argument that is no `Expression`.

### G4 custom provenances (`fhy_core::provenance`)

```rust
pub enum Provenance {
    Unknown, File(FileProvenance), Named(NamedProvenance),
    CallSite(CallSiteProvenance), Fused(FusedProvenance),
    Custom(Part<dyn CustomProvenance>),           // new
}

pub trait CustomProvenance: ForeignPart + fmt::Display {
    fn eq_part(&self, other: &dyn CustomProvenance) -> bool { /* identity */ }
    fn hash_part(&self, state: &mut dyn Hasher) { /* nothing */ }
}
// impl_part!(CustomProvenance): Part<dyn CustomProvenance> is Eq + Hash through the hooks

pub mod wire {
    pub struct ProvenanceData(/* private repr */);   // Serialize, Deserialize, Debug, Clone
    impl ProvenanceData {
        pub fn of(provenance: &Provenance) -> Result<Self, ForeignError>;
        pub fn build<R: Resolve<Part<dyn CustomProvenance>> + ?Sized>(self, resolver: &R)
            -> Result<Provenance, BuildError>;
    }
}
```

This follows the crate's open-type pattern (CONTRIBUTING, "Serialization is
plain serde"): the part lives in a `Part`, serializes as a `Foreign` under
the tag `custom` (`{"custom": {"type_id", "data"}}`, the tag
`Constraint::Custom` uses), the module's `wire` submodule mirrors the
whole tree with the parts left as `Foreign`s, and `Provenance`'s own
`Deserialize` builds with `NoForeign`, which refuses a custom part with
"no implementation for the foreign part `<type id>`". `Display` writes the
part's own `Display`. `fuse` keeps a custom provenance whole, as it keeps
every variant but an unlabelled fusion. Errors: `ProvenanceData::of`
returns the part's `ForeignError` (`NoWireForm` by default);
`Serialize for Provenance` fails with its text; `build` returns
`BuildError::Foreign` for a part the resolver refuses and
`BuildError::Invalid` with `NamedProvenanceError::EmptyName`.

Contract (rustdoc of `CustomProvenance`): getters stable for the value's
life; `Display` is one line; `eq_part` is an equivalence relation that
compares type-strictly and agrees with `hash_part`; `to_foreign` writes the
type id the resolver turns back into an equal value.

A Rust MOGA-VM writes `struct EdgePropagation { edge_id: Identifier }`,
`Display` as `edge<e3>`, and resolves it with its own
`Resolve<Part<dyn CustomProvenance>>`. Provenance resolvers are not part of
H3's registry: no search-space part holds a provenance.

**Python subclassing** (`rust/fhy-core-py/src/provenance.rs`):

- `_rs.Provenance` gains `#[new] (*_args, **_kwargs)` (the `Variable`/`Type`
  pattern), so a Python subclass defined outside `fhy_core`, such as a
  frozen dataclass, constructs; its own `__init__` takes the arguments.
- The base keeps `Option<Provenance>`: `Some` for a variant, `None` for a
  Python-defined subclass. Wherever the core needs the value (a variant's
  constructor given it as a child, `fuse`, `==` or `hash` of a variant
  holding it, serialization), the binding builds
  `Provenance::Custom(Part::new(PythonProvenance::new(object)))` on demand,
  as `try_read_variable` builds a `PythonVariable`, so no reference cycle
  forms. The adapter asks the object, through the pending-exception slot,
  for `==` (`eq_part`), `hash` (`hash_part`), `str` (`Display`), and
  `read_foreign(object, family = true)` (`to_foreign`: the registered type
  id and `serialize_data_to_dict()`).
- `provenance_to_python` returns the adapter's own object for a custom
  part built from Python, and resolves any other custom part through
  `fhy_core.serialization._resolve_foreign` (the class registered under
  its type id, `deserialize_data_from_dict`), so
  `Provenance.deserialize_from_dict({"custom": ..})` and a variant decoding
  a custom child return the subclass instance, as MOGA-VM's
  `test_provenance.py:41-51` expects.
- Pickling: `_rs.Provenance.__reduce__` pickles a Python-defined subclass
  as `(copyreg.__newobj__, (cls,), state)`, its `__dict__` the state; the
  variants keep their own `__reduce__`.
- Equality in Python stays the subclass's own (`@dataclass(frozen=True)`
  generates it), and the variants keep exact-class equality.
- `fhy_core.provenance` gains no name: subclassing `Provenance` works
  again. Its docstring says how.

**Breaking** (Rust): `Provenance` gains a variant and stays exhaustive
(`#[expect(clippy::exhaustive_enums)]`, reason updated), so a downstream
exhaustive `match` must handle `Custom`. On the Needs-the-user list (N4).

### G5 trace keys and measurement keys (`fhy_core::search_space`)

```rust
impl Trace {
    pub fn key(&self) -> TraceKey;
}

pub struct TraceKey(/* Arc<[step]> */);   // Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize
impl TraceKey {
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn coordinates(&self) -> impl ExactSizeIterator<Item = &Coordinate> + '_;
}

pub enum MeasurementKey {               // exhaustive (expect), Debug, Clone, Eq, Hash, Serialize, Deserialize
    Configuration(ConfigurationKey),
    Trace(TraceKey),
}
impl From<ConfigurationKey> for MeasurementKey;
impl From<TraceKey> for MeasurementKey;
impl PartialEq<ConfigurationKey> for MeasurementKey;
impl PartialEq<TraceKey> for MeasurementKey;

impl Measurement {
    pub fn ok(key: impl Into<MeasurementKey>, values: Vec<(Objective, f64)>) -> Result<Self, MeasurementError>;
    pub fn infeasible(key: impl Into<MeasurementKey>, reason: impl Into<String>) -> Self;
    pub fn failed(key: impl Into<MeasurementKey>, reason: impl Into<String>) -> Self;
    pub fn timeout(key: impl Into<MeasurementKey>) -> Self;
    pub fn key(&self) -> &MeasurementKey;            // was &ConfigurationKey
}

pub trait Measurer<S: ?Sized> {
    fn objectives(&self) -> &[Objective];
    fn measure(&mut self, key: &MeasurementKey, subject: &S) -> Result<Measurement, BoxError>; // was &ConfigurationKey
}
```

A `TraceKey` step is `(kind, decision, signature, coordinate)`: the step's
`DecisionKind`, its static decision's canonical position (`None` for a
dynamic step), the `DomainSignature` (which writes a space-bound
identifier by position and any other as `identifier`, so it is the same in
every process and every module build) and the `Coordinate`. It drops the
subject (minted per run for dynamic steps, per build for static ones) and
the value. So two runs that answered alike over the same domains have equal
keys, also across alpha-renamed spaces, while two runs that placed an
address differently have different keys. `Trace`'s own `==` keeps comparing
subjects. `serde`: `{"steps": [{"kind", "decision", "domain",
"coordinate"}, ..]}`; reading refuses a coordinate its signature does not
contain (as `TraceStep` does). No wire `Data` type is needed: a signature
holds no opaque value.

**Breaking against additive.** An additive design would keep
`Measurement::key() -> &ConfigurationKey` and add a separate
`TraceMeasurement` (or an `Option<&TraceKey>` accessor beside a key that
would then have to be optional). Both split one concept in two, double
`dominates`/`non_dominated`/serialization, or make the configuration key
optional and its accessor fallible. The enum keeps one `Measurement`. Its
cost: `key()`'s return type, `Measurer::measure`'s parameter and the wire
shape change; the constructors keep compiling for every caller through
`impl Into`, and `PartialEq<ConfigurationKey>` keeps
`assert_eq!(measurement.key(), &configuration.key())` compiling. 0.2.0 is
unreleased (the last tag is v0.1.8, which has no search space), so the
break reaches only MOGA's pin to `dev-rust`. Recommended: breaking (N3).

Wire: `Measurement` writes `"key": {"configuration": <key>}` or
`"key": {"trace": <key>}` (`MeasurementData` holds the configuration key
as `ConfigurationKeyData` so opaque values resolve). No golden corpus
holds a measurement (checked: `search_space_cases.json`'s `status` fields
are MOGA-VM selection statuses); `measurement_serde_stories.rs` pins the
old shape and is updated in the test phase.

**Python:** `TraceKey` (`_rs.TraceKey`, frozen, no constructor,
structural `==`/`hash`, `len`, `coordinates`, pickles through `_from_wire`
as `ConfigurationKey` does); `Trace.key() -> TraceKey`;
`Measurement.ok/infeasible/failed/timeout` accept a `ConfigurationKey` or
a `TraceKey` (no signature change); `Measurement.key` returns either; the
`Measurer` protocol's `measure(key: ConfigurationKey | TraceKey, subject)`.
`TypeError` for a key of another type, as today.

### G6 tuple categories (no new API)

`CategoricalDomain::new` accepts a category that is a leaf value (Boolean,
integer, string, identifier, member-shaped opaque value) or a
`Value::Tuple` or `Value::FrozenSet` of categories, at any depth up to
`constraint::wire::MAX_VALUE_DEPTH`. A float or a decimal stays refused,
inside a tuple too (`NotALeafValue`, with the index of the top-level
value). `Member` already supports tuples and frozen sets, and its
canonical order (`MemberSet`) orders them deterministically, which
MOGA-VM's replay needs ("category order must be deterministic").

`OrdinalDomain` and `PermutationDomain` keep refusing composite values.
An ordinal domain would need an order on tuples (lexicographic is the
obvious one, but mixing kinds inside tuples has no total order), and no
caller asks for it: MOGA-VM orders tile shapes in Python with `min` and
`max` outside the domain. A permutation of tuples has no caller either.
Both can be relaxed later without breaking anything.

Search: `step::member_value` already maps a tuple member to a
`Value::Tuple`, `ChoiceDomain` accepts tuples, `DomainSignature` writes a
plain tuple as itself, `ConfigurationKey` handles tuples, and a set
constraint over a tuple member works (the sketch's forbidden clause `tile
in {(8, 8)}`). Wire: the categorical domain's values already serialize as
values, so the change only accepts payloads that were refused. Python:
`read_finite_value` (binding `param/value.rs`) converts a hashable Python
`tuple`/`frozenset` for a categorical domain the way `read_bound_value`
does, and `categories` returns tuples.

**Docs changed in the stub:** `CategoricalDomain` and
`CategoricalDomain::new`.

### H1 `GuidedOracle` and `Space::crossover`

```rust
pub struct GuidedOracle<O> { /* static guide by position, dynamic guide per kind, fallback */ } // Debug, Clone
impl<O: SearchOracle> GuidedOracle<O> {
    pub fn new(guide: &Trace, fallback: O) -> Self;
    pub fn fallback(&self) -> &O;
    pub fn into_fallback(self) -> O;
}
impl<O: SearchOracle> SearchOracle for GuidedOracle<O>;

impl Space {
    pub fn crossover(&self, first: &Configuration, second: &Configuration, rng: &mut Rng,
                     context: &ParamContext<'_>, attempts: NonZeroU32) -> Result<Recorded, TraceError>;
}
```

`GuidedOracle` answers a static step with the guide's static step at the
same canonical position when the guide has one over an equal signature and
its coordinate is admissible; a dynamic step of kind `k` with the guide's
next untaken dynamic step of kind `k` (FIFO per kind, as MOGA-VM's
`ExtractedPointOracle` consumes repeated keys) when the signature is equal
and the coordinate in the domain, the guide step being taken either way;
everything else goes to the fallback. It never errors itself; the
fallback's error stops the run as it would unguided. Guiding by canonical
position (not by subject name) is what makes a trace of one module build
guide a run over the next, whose identifiers are fresh.

`crossover` draws, per decision in canonical order, `rng.below(2)` to pick
the parent it inherits from, and walks decision order: a decision takes
its picked parent's value when that parent assigns it and it is
admissible, else the other parent's under the same condition, else a value
drawn uniformly from the admissible ones with `rng`. A dead end restarts
with new picks, up to `attempts`. Parents need not be complete. Errors, in
order: `TraceError::OtherSpace` when either parent is of another space,
`TraceError::AttemptsExhausted`, and what a step returns. The operator is
uniform crossover; another operator would be another method (N9).

**Python:** `GuidedOracle(guide: Trace, fallback)` with `guide`,
`fallback` and `decide(step)`; `Space.crossover(first, second, rng, *,
attempts=16) -> (Configuration, Trace)`. A Python fallback oracle is held
in a GC-visible slot, as `Recorder` holds its oracle.

### H2 `non_dominated`

```rust
pub fn non_dominated(measurements: &[Measurement]) -> Result<Vec<&Measurement>, MeasurementError>;
```

Returns the successful measurements no other successful one dominates, in
the given order. A measurement that did not succeed is left out (not an
error, unlike `dominates`). Two with equal values are both kept unless a
third dominates them. Error: `MeasurementError::DifferentObjectives` for
two successful measurements over different objectives. O(m²) comparisons;
fine for a population. Python: `fhy_core.search_space.non_dominated(
measurements) -> list[Measurement]`, `TypeError` for a non-`Measurement`.
MOGA-VM has no Pareto filter today (its objective is one float); the
helper is the obvious consumer of `dominates` that a multi-objective
search needs, and it is small.

### H3 `ResolverRegistry` (`search_space::wire`)

```rust
pub type VariableResolverFn =
    fn(&Foreign, &dyn SearchSpaceResolver, &ParamContext<'_>) -> Result<Part<dyn Variable>, ForeignError>;
pub type AlternativeResolverFn =
    fn(&Foreign, &dyn SearchSpaceResolver, &ParamContext<'_>) -> Result<Part<dyn Alternative>, ForeignError>;
pub type OpaqueValueResolverFn = fn(&Foreign) -> Result<Part<dyn OpaqueValue>, ForeignError>;
pub type CustomConstraintResolverFn = fn(&Foreign) -> Result<Part<dyn CustomConstraint>, ForeignError>;
pub type CustomDomainResolverFn = fn(&Foreign) -> Result<Part<dyn CustomDomain>, ForeignError>;

#[derive(Debug, Clone, Default)]
pub struct ResolverRegistry { /* one HashMap<Arc<str>, fn> per family */ }
impl ResolverRegistry {
    pub fn new() -> Self;
    pub fn with_variable_kind(self, type_id: &str, resolve: VariableResolverFn) -> Result<Self, RegistryError>;
    pub fn with_alternative_kind(self, type_id: &str, resolve: AlternativeResolverFn) -> Result<Self, RegistryError>;
    pub fn with_opaque_value(self, type_id: &str, resolve: OpaqueValueResolverFn) -> Result<Self, RegistryError>;
    pub fn with_custom_constraint(self, type_id: &str, resolve: CustomConstraintResolverFn) -> Result<Self, RegistryError>;
    pub fn with_custom_domain(self, type_id: &str, resolve: CustomDomainResolverFn) -> Result<Self, RegistryError>;
    pub fn merge(self, other: Self) -> Result<Self, RegistryError>;
    pub fn resolver<'a, 'c>(&'a self, context: &'a ParamContext<'c>) -> RegistryResolver<'a, 'c>;
}

#[derive(Debug, Clone, Copy)]
pub struct RegistryResolver<'a, 'c> { /* registry, context */ }
// Resolve<Part<dyn Variable | Alternative | OpaqueValue | CustomConstraint | CustomDomain>>,
// hence SearchSpaceResolver
```

A part whose type id the registry holds is built by its function, a
variable or alternative function being handed the `RegistryResolver`
itself (as `&dyn SearchSpaceResolver`) for the parts inside it and the
context; any other part is refused with `ForeignError::Unresolved`, as
`NoForeign` refuses it. Errors: `RegistryError::ReservedTypeId` for
`PlainVariable::KIND`/`PlainAlternative::KIND` (tagged `plain` on the
wire, never resolved), `RegistryError::RepeatedTypeId` for an id already
in that family; `merge` reports the first id both hold, families in the
order above, ids ascending. The variable and alternative signatures are
the binding's `KindEntry` resolve signature, so one function serves both a
pure-Rust program and the Python registration
(`convert::search_space::register_variable_kind`).

It removes both sketch workarounds: the resolver no longer carries its own
`ParamContext` (the registry lends one), and two crates' kinds compose by
`merge` instead of a hand-written `match` on type ids. Function pointers,
not closures: they are `Copy + Send + Sync`, need no state (everything a
function needs is its arguments), and match the binding. A registry holds
no global state; a program builds one and passes it.

**No Python exposure:** the binding already keeps its own kind registry
in module state (`_rs._search_space_kinds`), which the same functions feed.

### H4 space editing and completion

```rust
impl Space {
    pub fn with_decisions(&self, variables: Vec<Part<dyn Variable>>, choices: Vec<Choice>) -> Result<Self, SpaceError>;
    pub fn without_decisions(&self, names: impl IntoIterator<Item = Identifier>) -> Result<Self, SpaceError>;
    pub fn complete(&self, configuration: &Configuration, oracle: &mut dyn SearchOracle,
                    context: &ParamContext<'_>) -> Result<Recorded, TraceError>;
}
// SpaceError::NotTopLevelDecision { name: Identifier }  — new variant (SpaceError is non_exhaustive)
```

- `with_decisions`: each given variable (choice) replaces the top-level
  variable (choice) of the same name **in its slot**, or is appended after
  the last top-level variable (choice), in the order given. Name,
  conditions, forbidden clauses and notes are kept; the result is checked
  as `Space::new` checks one, in one rebuild for any number of decisions.
  This is MOGA-VM's one editing pattern: rebuilding a candidate table by
  `merged = dict(table.entries); merged.update(new)` (`mapping/
  operations.py:310`, `memory/tiles.py:189`, `memory/bind.py:305`).
  Errors: what `Space::new` returns, such as `DuplicateName` for a
  variable named as a top-level choice and `UnknownReference` for a
  condition naming a decision a replaced choice no longer holds.
- `without_decisions`: removes top-level decisions and the conditions
  **targeting** them or anything under them; a condition or clause that
  **names** a removed decision is refused (`UnknownReference`), no
  auto-pruning (as decided for `without_entries`). A name given twice is
  removed once. Errors: `SpaceError::NotTopLevelDecision` ("{name:?} is
  not a top-level decision of the space") for the first name that is not
  one, then what `Space::new` returns. Mirrors
  `CandidateTable.without_entry`.
- **Which edits keep trace positions.** Canonical order is top-level
  variables, then top-level choices with their subtrees. Every decision
  before the first one an edit changes keeps its canonical position, and
  so do trace steps over it. Appending choices keeps every existing
  position; appending a variable moves every choice's subtree by one;
  replacing a choice by one with another number of decisions in its
  subtree moves everything after it; removing moves everything after the
  first removed. A replay or `GuidedOracle` over an edited space is
  therefore exact only for the prefix; to carry values across an edit,
  build `Configuration::new(&edited, old.entries())` and `complete` it.
- `complete`: answers every decision `configuration` assigns from it
  (without asking the oracle) and every other active one from `oracle`, in
  decision order, returning the completed configuration and the full trace
  (which `replay` turns back into it). A complete configuration comes back
  as it is, with its trace. Errors: `TraceError::OtherSpace`, then what
  `Recorder::decide` returns. It replaces the sketch's hand loop over
  `Recorder::realizing`.

`concat` (two spaces into one) is **dropped**: MOGA-VM never concatenates
spaces, and merging two spaces' conditions and names has no single right
answer; `with_decisions(other.variables().to_vec(),
other.choices().to_vec())` covers appending another space's decisions
when its conditions are not needed.

Python: `Space.with_decisions(variables=None, choices=None)`,
`Space.without_decisions(names)`, `Space.complete(configuration, oracle)
-> (Configuration, Trace)`.

### H5 per-choice completeness and cheap one-entry building

```rust
impl Configuration {
    pub fn is_complete_under(&self, name: &Identifier) -> Option<bool>;
}
```

`Some(true)` when the decision `name` and every decision under it at every
depth are assigned or inactive (a decision under an alternative the choice
did not choose is inactive); `None` for no such decision. MOGA-VM's
`SelectionStatus` maps onto it: a choice unassigned is UNSELECTED, assigned
and complete under itself SELECTED, assigned and not complete PARTIAL. (In
MOGA-VM the status is set by hand today and PARTIAL appears only in tests
and in the recorded corpus, so this is a convenience, not a blocker.)
Python: `Configuration.is_complete_under(name) -> bool | None`.

**Cheaper `with_entry`/`with_entries`.** One entry at a time over 3 400
decisions costs 0.49 s today because each call re-checks every held value
against its param (`ValueCheck::New`, a solver call per value) and
re-derives every decision's activity. After: the held values are not
re-checked (the configuration's own check accepted them; the type's
invariant is that every configuration is valid), only the new entries'
values are checked, and activity and forbidden clauses are re-derived only
for the decisions that depend on the entries (the dependents from G1 step
3). One-by-one building is then linear in the entries plus one copy of the
value vector per call. Target: 3 400 decisions one entry at a time in under
50 ms in release. No builder type is added: `Configuration::new` already
takes all entries at once, and the incremental check serves both the
search (`extended`) and one-by-one callers. This changes what
`with_entries` checks (N5). **Docs changed in the stub:** `with_entry`,
`with_entries`.

### H6 docs

The implementor contract's clause 2 (one kind per implementing type) now
names the marker-generic pattern, `Knob<K: KnobKind>` with `K::KIND`,
which the sketch uses for MOGA-VM's five knob kinds, and why one type
answering several kinds by a field breaks the `is_extension_*` downcasts.
The opaque-value workaround for tuple categories is no longer needed
(G6). An ordinal domain over tuples would still need an opaque value with
an ordering key; no caller needs one, so no doc is added for it.

## Visibility

Every new `pub` item and its caller outside the crate (MOGA-VM's Rust port
unless named otherwise):

| Item | Caller |
|---|---|
| `expression::Rational` (+ `new`, `numerator`, `denominator`, `is_integer`, `to_integer`, `is_zero`, `From<BigInt>`, `Display`) | `AffineForm`'s coefficients; walk grammar strides |
| `expression::AffineForm`, `Expression::affine_form` | walk grammar `_affine_coefficient`, tiling `_axis_coefficient`; the binding |
| `provenance::Provenance::Custom`, `CustomProvenance` | `EdgePropagationProvenance`; the binding's Python adapter |
| `provenance::wire::ProvenanceData` | decoding IR holding custom provenances; the binding |
| `search_space::TraceKey`, `Trace::key` | measurement cache keyed by runs with dynamic steps; the binding |
| `search_space::MeasurementKey` (+ `From`, `PartialEq`) | `Measurement`, `Measurer` implementors |
| `search_space::non_dominated` | multi-objective selection; the binding |
| `search_space::GuidedOracle` | genetic operators, transplants; the binding |
| `Space::crossover`, `Space::complete`, `Space::with_decisions`, `Space::without_decisions` | search strategies, table rebuilds; the binding |
| `Configuration::is_complete_under` | selection status; the binding |
| `wire::ResolverRegistry`, `wire::RegistryResolver`, the five `*ResolverFn` aliases | every crate that defines search-space parts |
| `search_space::RegistryError` | `ResolverRegistry` callers |
| `SpaceError::NotTopLevelDecision` | `without_decisions` callers |

Crate-internal items the implementation adds (not in the stub): the
per-decision dependents of a `Space` (`pub(super)`), the closed-form and
component helpers shared by `counting` and `exploration` (`pub(super)`),
the exact bound decision in `param::decide` (`pub(super)`), and the
binding's `PythonProvenance` adapter (private to `provenance.rs`).

Matchable enums: `MeasurementKey` (exhaustive: a measurement is of a
configuration or a run) and `Provenance` (exhaustive, gaining `Custom`).
`RegistryError` and `SpaceError` are `#[non_exhaustive]`. Privacy
boundaries for invariant-carrying types: `expression/affine.rs` (leaf,
`AffineForm`'s no-zero-coefficient invariant), `expression/literal/exact.rs`
(leaf, `Rational`'s lowest terms), `search_space/trace.rs` (leaf,
`TraceKey`), `search_space/wire.rs` (leaf, the registry's no-repeat
invariant). `TraceKeyStep` and the registry's maps stay private; the wire
reprs are private.

## Ownership and data model

`AffineForm` owns its `BTreeMap` (ordered by identifier id, so iteration,
`Display` and `to_expression` are deterministic) and its `Rational`s;
`affine_form` borrows the expression. `TraceKey` is an `Arc<[step]>`, so
cloning shares it like `Trace` and `ConfigurationKey`. `MeasurementKey`
owns its key. `GuidedOracle<O>` owns its guide tables (copied out of the
trace once: coordinates and signatures, both cheap to clone) and its
fallback by value, so `GuidedOracle<&mut RandomOracle>` borrows a fallback
through the existing blanket impl. `ResolverRegistry` owns `fn` pointers in
`HashMap<Arc<str>, _>`; `RegistryResolver` borrows the registry and the
context and is `Copy`. A custom provenance is shared through `Part`'s
`Arc`. All new types are `Send + Sync` (`GuidedOracle<O>` when `O` is), as
the existing search-space types are.

## Extensibility decisions

| Point | Mechanism | Why |
|---|---|---|
| Provenance kinds | closed enum + `Custom(Part<dyn CustomProvenance>)` | the crate's open-variant pattern: core variants stay matchable, other crates add theirs |
| `CustomProvenance` | open trait, not sealed, defaults for `eq_part`/`hash_part` | downstream implements it; new methods get defaults |
| Measurement subject | closed exhaustive enum `MeasurementKey` | two kinds of point exist (configuration, run); callers match both |
| Repair oracle | generic `GuidedOracle<O: SearchOracle>` | the fallback is one concrete type per use; no `dyn` needed (and `Box<dyn SearchOracle>` is itself an oracle) |
| Resolver composition | owned table of `fn` pointers per family | runtime composition of several crates, mirroring the binding; no global state |
| Affine analysis | inherent method, `Option` result | one analysis with no configuration; a declined input is not an error |

`RegistryError` is `#[non_exhaustive]`; `MeasurementKey` and `Provenance`
are exhaustive with `#[expect(clippy::exhaustive_enums, reason = ..)]`, as
CONTRIBUTING asks for enums callers match exhaustively.

## Error model

New variants and their `Display` texts (one lowercase line, no period):

| Type | Variant | `Display` |
|---|---|---|
| `SpaceError` | `NotTopLevelDecision { name }` | `{name:?} is not a top-level decision of the space` |
| `RegistryError` (new, `#[non_exhaustive]`) | `RepeatedTypeId { type_id }` | `the type id {type_id:?} is registered twice` |
| `RegistryError` | `ReservedTypeId { type_id }` | `the type id {type_id:?} is the search space's own` |

Existing errors reused: `TraceError::OtherSpace`, `AttemptsExhausted`,
`Configuration` (now also for an assignment the context cannot decide,
G2); `MeasurementError::DifferentObjectives`; `ForeignError::Unresolved`,
`NoWireForm`; `BuildError::{Foreign, Invalid}`. `affine_form` and
`is_complete_under` answer `Option`; nothing new panics on caller input.

Python: `SpaceError::NotTopLevelDecision` raises `SearchSpaceError`;
`MeasurementError` raises `MeasurementError`; the rest as today.

## Behavior

- G1: a 200-entry table (4 options of 4 knobs) mutates in under 100 ms in
  release; `mutate(c, Rng::new(s))` returns today's result for every space
  whose old search was exact.
- G2: param `x: nat`, `16 <= x <= 4095`, `Solver::new()`:
  `cardinality` → `Exact(4080)`; `sample` → a value in `[16, 4095]`;
  `Configuration::new(.., [(x, 2955)])` → `Ok`; `[(x, 15)]` →
  `Assignment(ViolatedConstraint)`. Param `x % 4 == 0` under
  `Solver::new()`: `sample` → `TraceError::Configuration`, not `DeadEnd`.
- G3: `(3*s + t) - t` → `{s: 3}`, constant 0; `x/2 + x/2` → `{x: 1}`;
  `2*(i + 1) - 1` → `{i: 2}`, constant 1; `i*j` → `None`; `i // 2` →
  `None`; `7 // 2 + i` → `{i: 1}`, constant 3; `x + 0.5` (float) → `None`;
  `x / 0` → `None`; `5` → constant 5, `is_constant()`; `x - x` → constant
  0, no term, `to_expression()` is `0`.
- G4: `Provenance::Custom(edge("e3"))` displays `edge<e3>`, serializes as
  `{"custom": {"type_id": "pkg.edge_propagation", "data": "e3"}}`;
  `serde_json::from_str::<Provenance>` of it fails with "no implementation
  for the foreign part `pkg.edge_propagation`"; `ProvenanceData` with a
  resolver gives it back equal. `fuse([a, Custom, b])` keeps it whole.
  Python: `EdgePropagationProvenance(edge_id=i)` constructs, `str` is
  `edge<..>`, `NamedProvenance("n", ep).child` is `ep` itself,
  `Provenance.deserialize_from_dict(ep.serialize_to_dict())` is an
  `EdgePropagationProvenance` equal to `ep`, and pickling round-trips.
- G5: two runs with equal static steps and different address coordinates:
  configuration keys equal, trace keys differ. The same run recorded twice
  with fresh subjects: traces differ (`!=`), trace keys equal.
  `Measurement::ok(trace.key(), ..)` round-trips through JSON and postcard.
- G6: `CategoricalDomain::new([(4,4), (8,8)])` → `Ok`, categories in
  canonical order; `[(4, 4.5)]` → `NotALeafValue { index: 0 }`;
  `OrdinalDomain::new([(4,4)])` → `NotALeafValue` as today.
- H1: guide = trace of `c`, fallback a `RandomOracle`, run over `c`'s
  space → `c` itself, the fallback never asked. Over a space where `c`'s
  choice value is now forbidden → that step from the fallback, the rest
  from the guide. `crossover(a, a, ..)` → `a`.
- H2: `[ok(1,2), ok(2,1), ok(2,2), failed]` (minimize both) →
  `[ok(1,2), ok(2,1)]`; `[]` → `[]`; two equal → both.
- H4: `with_decisions(vec![], vec![new_entry])` → the new choice last,
  every old position kept; `with_decisions` with a choice named like an
  existing one → replaced in place; `without_decisions([v])` where a
  condition names `v` → `UnknownReference`; `without_decisions([unknown])`
  → `NotTopLevelDecision`. `complete(partial, RandomOracle)` → complete,
  its entries a superset of `partial`'s.
- H5: `is_complete_under(choice)` on a configuration choosing an
  alternative with an unassigned variable → `Some(false)`; after assigning
  it → `Some(true)`; for a variable → whether it is assigned or inactive;
  unknown name → `None`.

## Non-goals

- Ordinal or permutation domains over tuples (G6).
- Provenance resolvers in `ResolverRegistry` (no search-space part holds a
  provenance); a provenance-holding IR resolves its own.
- `Space::concat` (H4, dropped above).
- A `ConfigurationBuilder` (H5): the incremental check covers one-by-one
  building.
- Operators other than uniform crossover (H1).
- Folding non-affine expressions with free identifiers (a general pure-Rust
  symbolic simplifier); G3 is exact linear analysis only.
- `#[derive(AlphaEquivalence)]` (on the Needs-the-user list).

## Test plan

Rust tests go in `rust/fhy-core/tests/it/` (one module per area); Python in
`tests/`. Every new serialized type gets a JSON and a postcard round trip.

**G1**
- Unit: candidate options of a closed-form component equal the old
  search's on small spaces (choice with an empty alternative, ordering
  domain, 2^16+1 domain left out).
- Property: on random small spaces (conditions, forbidden clauses, empty
  domains), the set of mutable decisions and their options equals a
  brute-force oracle (enumerate all complete configurations; a decision
  may change to `v` iff some configuration keeps the earlier decisions and
  takes `v`); every mutation is complete, changes exactly one non-repaired
  decision, and replays from its trace.
- Seeded regression: record today's `mutate` outputs for a table of seeds
  over the existing story spaces before the change, and pin them
  (documents the "unchanged" claim).
- Benchmark (below).

**G2**
- Param unit: bound members decided without a simplifier (`>=`, `>`, `<=`,
  `<`, literal on either side, negative bounds, big integers); a non-bound
  member still asks the solver; no member event for a bound member.
- Search: the reproduction (`Exact(4080)`, sample in range,
  `Configuration::new` accepts and refuses at the edges) under
  `Solver::new()`; the `x % 4 == 0` param errors with
  `TraceError::Configuration` in `sample`, `cardinality` and `mutate`.
- Property: for random bound-only integer params, `cardinality` equals the
  number of values `Configuration::new` accepts, under `Solver::new()` and
  under the ground simplifier.
- Python: an existing param test that observes member logging for bound
  members, if any, is updated (risk noted).

**G3**
- Unit: the behaviour table above; every declined shape; depth 257
  declined; 4 097-bit coefficient declined; `Rational` display, sign
  normalization, ordering, `to_integer`.
- Property: for random affine-shaped trees (sums, constant multiples,
  constant divisions, negations), `affine_form(e).to_expression()`
  evaluates equal to `e` under random integer environments (oracle: the
  evaluator); `affine_form(a - b)` equals the difference of the forms.
- Differential (binding, with SymPy, beside `ground_differential`): the
  coefficient agrees with SymPy's `expand` + `coeff` on random affine
  trees.
- Python: `affine_form` returns `Fraction`s, `None` for non-affine,
  `TypeError` for a non-expression; stub test covers names.

**G4**
- Rust: `Custom` display, `==`/`Hash` through the hooks, serialize text,
  `Deserialize` refusal text, `ProvenanceData` round trip with a resolver
  (JSON and postcard), nested inside named/call-site/fused, `fuse` keeps it,
  `NoWireForm` fails serialization with its text.
- Golden: the existing provenance cases replay unchanged.
- Python: MOGA-VM's `EdgePropagationProvenance` shape as a test class:
  construct, `str`, `==`/`hash`, as a child of each variant, through
  `fuse`, `serialize_to_dict`/`deserialize_from_dict` (V2 and V1),
  `to_json`/`from_json`, pickle and `copy.deepcopy`, an exception from its
  `__eq__` surfaces, GC (`gc.collect` with a cycle through it).

**G5**
- Rust: `Trace::key` drops subjects and values (fresh subjects → equal
  keys), differs on a dynamic coordinate, equal across alpha-renamed spaces;
  serde round trips and the refusal of an out-of-domain coordinate;
  `MeasurementKey` equality with both key types; `Measurement` with a trace
  key through `MeasurementData` and the plain `Deserialize`.
- Update `measurement_serde_stories.rs` to the tagged `key` shape.
- Python: `Trace.key()`, `TraceKey` hash/eq/pickle/len/coordinates,
  `Measurement.ok(trace_key, ..)`, `.key` type, `Measurer` protocol check.

**G6**
- Rust: tuple and frozen-set categories accepted, nested; float inside a
  tuple refused with the top-level index; ordinal/permutation still refuse;
  canonical order deterministic; a space over a tuple categorical samples,
  enumerates, keys and replays; set constraint with a tuple member.
- Python: `CategoricalDomain({(4, 4), (8, 8)})`, `categories` gives tuples,
  a `Configuration` entry `(8, 8)` is accepted.

**H1**
- Rust: `GuidedOracle` stories (full guide, inadmissible guide step,
  signature mismatch, dynamic FIFO per kind with an extra and a missing
  step, empty guide = fallback); `crossover` stories (identical parents,
  disjoint alternatives, a forbidden combination repaired, incomplete
  parents, other space, attempts exhausted).
- Property: crossover results are complete and every value comes from a
  parent or is admissible-drawn; seeded crossover is deterministic.
- Python: `GuidedOracle` with a Python fallback oracle (and its exception
  propagating), `Space.crossover`.

**H2**
- Rust: the behaviour table; different objectives error; report objectives
  ignored. Property: no returned measurement is dominated by any input;
  every omitted successful one is dominated by a returned one (oracle:
  pairwise `dominates`). Python: `non_dominated`.

**H3**
- Rust: register each family, repeated and reserved ids, `merge` conflicts
  and order, unknown id refused with `Unresolved`, a variable resolver
  receiving the registry for its param's custom domain, two crates' kinds
  merged and decoding one space (the sketch's `MogaResolver` rewritten
  over it).

**H4**
- Rust: replace-in-place keeps positions; append keeps positions; replace a
  choice with a different subtree size moves later positions (assert with
  `Trace` positions); removal drops conditions on the removed subtree and
  refuses references; `NotTopLevelDecision` for a nested or unknown name;
  `complete` on empty, partial and complete configurations, other space.
- Python: the three methods.

**H5**
- Rust: `is_complete_under` table above, nested choices; with_entries no
  longer re-checks held values (a held value a later context could not
  verify stays); activity changes propagated through conditions at depth.
- Property: building any valid configuration one entry at a time gives the
  same configuration (and the same errors at the first bad entry) as
  `Configuration::new` with all entries (oracle: `new`).
- Python: `is_complete_under`.

**Benchmarks to add** (`benchmarks/test_search_space.py`, pytest-benchmark,
public API only): `test_table_space_mutate` (200 entries × 4 options of 4
knobs), `test_configuration_built_one_entry_at_a_time` (about 3 400
decisions), `test_table_space_crossover`, `test_affine_form_of_an_offset`.
In Rust, an `#[ignore]`d `search_space::timing` test (as
`ground_differential::timing`) prints mutate at 10/50/100/200 entries and
one-entry building at 3 400 decisions in a release build; the acceptance
numbers are G1 < 100 ms at 200 entries and H5 < 50 ms at 3 400 decisions.

## Breaking changes

1. `Provenance` gains the variant `Custom` (exhaustive enum): downstream
   exhaustive matches must add an arm. (Rust)
2. `Measurement::key` returns `&MeasurementKey`; `Measurer::measure` takes
   `&MeasurementKey`; the constructors take `impl Into<MeasurementKey>`
   (source-compatible for `ConfigurationKey` callers). (Rust)
3. `Measurement`'s wire shape: `"key"` becomes `{"configuration": ..}` or
   `{"trace": ..}`. (Rust and Python V2 payloads)
4. Python `Measurement.key` may be a `TraceKey`; the `Measurer` protocol's
   `key` parameter widens.
5. `Configuration::with_entry`/`with_entries` no longer re-check held
   values (behaviour).
6. `Param::evaluate_constraints` and everything over it decide integer
   bounds without the solver and without member events (behaviour).
7. `try_extend`'s classification: a run under a context that cannot decide
   a non-bound constraint fails with `TraceError::Configuration` instead
   of `DeadEnd`, and `cardinality` with that error instead of a count
   (behaviour).
8. `expression::Rational` becomes public API (additive, but a commitment).

None reaches a released version: v0.1.8 has no Rust search space and no
Rust provenance binding. In Python, `_rs.Provenance` gaining a constructor
restores 0.1.8 behaviour.

## Needs the user (decided 2026-10-07)

N1. **`#[derive(AlphaEquivalence)]` in a new `fhy-core-derive` crate.**
MOGA-VM has about 15 IR classes using `DerivedEquivalenceMixin`. A
proc-macro crate must be a separate crate, so it would be a third
published crate (#104 publishes `fhy-core` and `fhy-core-py`), with its
own release step, version lock-step with `fhy-core`, MSRV and `deny`
coverage, and `syn`/`quote`/`proc-macro2` as new dependencies.
*Recommendation:* not now. The container impls (`Option`, `Vec`, slices,
tuples up to 8) make a hand impl a few `&&`-ed lines per type, about 100
lines for 15 types. Revisit once the Rust MOGA-VM IR exists and shows more
types or drift; if a shortcut is wanted before that, a `macro_rules!`
`impl_alpha_equivalence!(Type { field, field })` in `fhy-core` gives most of
the benefit with no new crate.

N2. **G2: decide integer bounds in the param without the solver.** This
changes param semantics beyond the search space: `ParamAssignment::new`
succeeds without a simplifier for bound-only params, and bound members
stop emitting member events (the Python param observer logs fewer events;
SymPy is asked less, so Python gets faster). *Recommendation:* yes. The
answer is exact and the one every correct simplifier gives; the param
already treats bounds as structure. The fallback, if declined: keep only
the `try_extend` error fix and document that the search needs a
ground-deciding context (the count/sample disagreement stays).

N3. **G5: break `Measurement::key`, `Measurer::measure` and the wire
shape for `MeasurementKey`.** *Recommendation:* break now, while 0.2.0 is
unreleased; the additive alternatives split `Measurement` in two or make
its key optional.

N4. **G4: `Provenance` gains `Custom` and stays exhaustive.**
*Recommendation:* yes, exhaustive: consumers that render or walk
provenances must decide what to do with a custom one. Making it
`#[non_exhaustive]` instead would hide future variants from the compiler
for no current need.

N5. **H5: `with_entry`/`with_entries` stop re-checking held values.**
Differs only when a configuration built under one context gets an entry
under a weaker one that could not verify an old value. *Recommendation:*
yes; validity is the type's invariant, and it is what makes one-by-one
building linear.

N6. **G3: make `expression::Rational` public** rather than add
`num-rational`. *Recommendation:* yes; no dependency, one rational type in
the crate.

N7. **G1 seeded outputs.** Unchanged except where the old 1 024-run search
guessed; nothing pins them. *Recommendation:* accept; informational.

N8. **G6: categorical domains accept tuple and frozen-set categories;
ordinal and permutation domains do not.** *Recommendation:* yes; relax the
others when a caller needs an order on tuples.

N9. **H1: the crossover operator is uniform crossover only.**
*Recommendation:* yes; another operator (one-point over decision order) is
another method when a search asks for it.

## Open questions

None beyond the Needs-the-user list. The implementation phase measures G1
and H5 against their targets before refactoring; if the value-vector copy
dominates at 200 entries, sharing values in chunks is an internal change.
