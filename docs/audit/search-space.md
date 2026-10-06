# Audit: the MOGA-VM search-space design, as compiler infrastructure

## Scope

- **Audited:** MOGA-VM `origin/dev` at `3d93ba3`, read-only. Paths below are
  relative to MOGA-VM's `src/moga_vm/` unless they start with `tests/`.
  - **Primary:** the portable core `cir/space/core/` (`decision.py`,
    `equivalence.py`, `knob.py`, `metric.py`, `option.py`), which the
    import-linter contract in MOGA-VM's `pyproject.toml` (lines 192-215)
    keeps on `fhy_core` alone "so it can be promoted into fhy_core
    unchanged".
  - **For scope and generality only:** `cir/space/{knobs,realization_option,table}.py`
    and `cir/lowering/search/{decisions,oracle,extraction,errors}.py`.
- **Question asked:** is the design sound as compiler infrastructure, and
  what must change when it moves into fhy-core. This audit is descriptive;
  `docs/design/search-space.md` turns it into decisions.
- **User decision (2026-10-05):** only the generic parts move into
  fhy-core; the CIR-specific layer (`knobs.py`, `realization_option.py`,
  `table.py`, anything tied to `HardwareDFG` or `moga`) stays in MOGA-VM as
  implementations of the generic vocabulary, and the Rust core is built on
  traits. The section "The CIR-specific layer" lists what those traits'
  contracts must prevent.
- **Severity scale:** the hardening skill's. Equivalence relations that
  answer wrongly are rated by consequence: a false "equivalent" (unsound)
  outranks a false "not equivalent" (too strict), since fhy-core's own
  derived-equivalence engine accepts "too strict, never unsound"
  (`src/fhy_core/term/derived_equivalence.py`, module docstring).

## Inferred purpose

A static description of one compile-time choice: a `DecisionPoint` holds a
`DecisionSpace` of alternative `Option`s, each carrying tunable `Knob`s (a
named `fhy_core` `Param`) and `Metric`s, plus an optional `Selection` that
commits one option and assigns some of its knobs (`KnobAssignment`). The
values are frozen, compared by structural and alpha equivalence (so that two
CIR modules built independently can be compared up to renaming of their
labels), and serialized through `fhy_core.serialization`. MOGA-VM keeps one
`DecisionPoint` per placeholder vertex (`cir/space/table.py`), subclasses
`Option` and `Knob` with hardware-specific data, and searches the space
through a separate, dynamic decision stream (`cir/lowering/search/`).

## Inferred public interface

| Item | Kind | Contract (as implemented) |
|---|---|---|
| `Knob[_T]` | frozen dataclass, family root, unregistered | `name`, `param: Param`, `notes`; `assign(v)` builds a `KnobAssignment` |
| `ArrayKnob` | registered marker subclass (`moga.cir.array_knob`) | no fields |
| `KnobAssignment[_T]` | frozen, registered (`moga.cir.knob_assignment`) | `knob_identifier`, `assignment: ParamAssignment` |
| `Metric`, `MetricKind` | frozen, registered (`moga.cir.metric`); `StrEnum` | `name`, `kind` (COST/BENEFIT/DIAGNOSTIC), `value: Expression`, `notes` |
| `Option` | frozen dataclass ABC, family root, unregistered | `name`, `knobs`, `metrics`, `notes`; `collect_alpha_label_bindings` hook |
| `DecisionSpace` | frozen, registered (`moga.cir.decision_space`) | `name`, `options`, `notes` |
| `Selection` | frozen dataclass ABC, family root | `selected_option_identifier`, `name`, `status`, `knob_assignments`, `metrics`, `notes` |
| `ArraySelection` | registered (`moga.cir.array_selection`) | the only concrete `Selection` |
| `SelectionStatus` | `StrEnum` | UNSELECTED, PARTIAL, SELECTED |
| `DecisionPoint` | frozen, registered (`moga.cir.decision_point`) | `space`, `name`, `selection`, `notes`; validates the selection in `__post_init__` |
| `SelectionConsistencyError` | `ValueError` | unknown option, unknown knob, assignments while UNSELECTED |
| 8 `are_*`/`is_param_*`/`collect_*` helpers, 5 `is_valid_*_data`, 4 `*Data` TypedDicts | functions, types | exported in `cir/space/core/__init__.py:12-43`; only `are_identifier_sequences_alpha_equivalent_under`, `KnobData`, `is_valid_knob_data`, `OptionData`, `is_valid_option_data` have users outside the core (`cir/space/knobs.py`, `cir/space/realization_option.py`) |

## Current state

### The oracle

The core's tests were run as the oracle in two environments. Scripts are in
the session scratchpad (`oracle/`), not in either repository.

| Environment | fhy_core | Result |
|---|---|---|
| Reference: the MOGA-VM pin (`fhy-core>=0.1.8`) | v0.1.8 (`ad9a311`), pure Python, from `git archive v0.1.8` | **50 passed** of 50 collected |
| Current: this repository | 0.2.0 (`6548984`), Rust-backed | **does not import** without shims (F-SS-008); with module-path aliases, **44 passed, 6 failed** |

- **Collected:** `tests/cir/space/test_alpha_standalone.py` (12),
  `test_selection_validation.py` (7), `test_structural_equivalence.py` (24),
  `test_core_import_boundary.py` (1) and `tests/cir/test_space.py` (6).
- **Not run:** `tests/cir/space/test_table.py` (17) and
  `test_table_partition.py` (25). Their fixtures build `moga` capabilities,
  and `moga` is not installable here; they test MOGA-bound code
  (`TableEntry`, `CandidateTable`) that stays in MOGA-VM.
- **Stand-ins** (`oracle/shim`, `oracle/pkg/moga_vm/cir/ir`): `moga.alias`
  (`SymbolName`/`NamespaceName` as `Identifier`), `WordIndexedAddressingScheme`,
  `PortRole`, `ArrayLayoutKind`, `NestedWalkVertex`, and an empty
  `ImmutableHardwareDFG` that is equivalent only to itself, as the real empty
  DFG is (`tests/cir/space/test_alpha_standalone.py:30-36`). The `space`
  package itself is the unmodified export.
- **The six failures on 0.2.0** are all in `cir/space/knobs.py:282`
  (`_build_bound_nat_param`): `EquationConstraint(identifier, expression)`
  is now `EquationConstraint(expression)`. They are MOGA-bound code, not
  the core.
- **Probes.** `oracle/probes.py` holds one counterexample per finding below;
  every verdict cited is identical on both environments.
  `oracle/relation_sweep.py` draws 3,000 random decision points with label
  reuse (seed 0): of 2,879 constructible ones, relabeling each occurrence
  freshly gives a non-equivalent verdict for 904 and an asymmetric verdict
  for 1,162; a sharing-preserving relabeling gives none. Structural without
  alpha equivalence never occurred.

### Summary

| ID | Title | Sev | Category | Triage |
|---|---|---|---|---|
| F-SS-001 | Alpha equivalence of categorical and permutation knobs ignores their constraints | Critical | Correctness | a |
| F-SS-002 | Decision-point alpha equivalence is not symmetric when one side reuses a label | High | Correctness | a |
| F-SS-003 | Identifier domain members are matched by `resolve`, which lets a free name capture a bound one | High | Correctness | a |
| F-SS-004 | Python `==` makes `1`, `True` and `1.0` alpha-equivalent members and values | High | Correctness | a |
| F-SS-005 | Selections are validated incompletely: duplicates, conflicting and out-of-domain assignments pass | High | Correctness | a |
| F-SS-006 | The core cannot round-trip on its own: no concrete `Option`, `Knob` writes an unreadable type id | High | Serialization | a |
| F-SS-007 | Knob names have two inconsistent binding scopes; equivalent points compare unequal | Medium | Correctness | a |
| F-SS-008 | The core no longer imports against fhy_core 0.2; "promoted unchanged" is false | Medium | Foundations | a |
| F-SS-009 | Identifier-valued domain members are opaque in fhy_core 0.2 and are never renamed | Medium | Foundations | b |
| F-SS-010 | `SelectionStatus` has no checkable meaning beyond UNSELECTED-with-assignments | Medium | API | b |
| F-SS-011 | Knob name and param variable are two names for one variable; assignments carry a third | Medium | API | b |
| F-SS-012 | Metric names are unbound references with an undocumented role; metrics have no producer | Medium | API | b |
| F-SS-013 | Two vocabularies: the static space cannot be enumerated, indexed or sampled | Medium | Generality | b |
| F-SS-014 | Empty decision spaces are accepted, unlike every neighboring abstraction | Low | Correctness | b |
| F-SS-015 | Default names mint a fresh identifier per construction | Low | API | b |
| F-SS-016 | Kind checks mix `type(self) is type(other)` and `isinstance`; helpers dispatch on the left | Low | Correctness | a |
| F-SS-017 | Labels are re-bound at several levels; standalone and nested verdicts disagree | Low | Correctness | a |
| F-SS-018 | Option order is significant, but the domain is documented as a set | Low | Documentation | b |
| F-SS-019 | Implementation helpers and payload shapes are exported as API; MOGA type ids on generic classes | Low | API | a |

Totals: 1 Critical, 5 High, 7 Medium, 6 Low. Triage: **a** = fix in the
port as an approved divergence; **b** = needs a user decision (see the
design's "Needs the user").

## Is this the right component set?

This section asks whether `Knob`, `KnobAssignment`, `Metric`, `Option`,
`DecisionSpace`, `Selection` and `DecisionPoint` are the right vocabulary
for a general compiler search space, before any of it is fixed in traits.

### What each component is for

| Component | Role | Assessment |
|---|---|---|
| `Knob` | a decision variable: a name over a `Param` domain with constraints | **Right.** It is the unit every design has (OpenTuner's parameters, ConfigSpace's hyperparameters, MetaSchedule's sampled values). The name duplicates the param's variable (F-SS-011). In MOGA-VM it is also used for facts: a single-category `NamespaceKnob` is "a hardware fact read off the port's declaration, not a choice" (`cir/lowering/search/decisions.py:25-31`). |
| `KnobAssignment` | one variable's value in a point | **Right as a value**, but it is a pair `(knob, value)` wrapped around a whole `ParamAssignment` whose param must equal the knob's (F-SS-005, F-SS-011). |
| `Option` | an alternative: a named choice that brings its own knobs | **Right.** Alternatives that open private sub-decisions are the conditional-space core of Ansor sketches and ConfigSpace conditions. It is limited to one level: an option's knobs cannot themselves gate sub-decisions. |
| `DecisionSpace` | the alternatives of one decision | **Partly redundant.** It is `(name, options, notes)`; its name is never read, and MOGA-VM sets it to the vertex id it also gives the decision point (`cir/space/table.py:140-144`). It is the only place the option list lives, so it is the natural "space of one decision". |
| `Selection` | the committed alternative plus some knob values | **Conflated.** It is a *point* (a configuration), but it is stored inside the space's `DecisionPoint`, carries its own `name` and `metrics`, and has a three-valued `status` nothing checks (F-SS-010). A configuration in other designs is a plain map from variables to values with an identity of its own (Optuna's `trial.params`, ConfigSpace's `Configuration`, MetaSchedule's trace decisions). |
| `DecisionPoint` | a decision placed in a program: a space plus at most one selection | **Conflates a space with one point in it.** Its real job is the validation and binding boundary between a space and a selection (`cir/space/core/decision.py:343-383`); its name duplicates the space's. |
| `Metric` | a COST/BENEFIT/DIAGNOSTIC expression on an option or a selection | **Misplaced and unused.** Measured results belong in an evaluation layer keyed by a point (MetaSchedule's tuning records, Optuna's trial values), not inside the space. A cost *estimate* over knob variables (an expression) can live in the space, but then it must reference knobs, and the metric's own name needs a vocabulary (F-SS-012). No MOGA-VM source constructs one. |

### How MOGA-VM uses the components

From `git grep` on `origin/dev`, outside `cir/space/`:

| Component or feature | Built at | Read at | Load |
|---|---|---|---|
| `Knob.param.domain` as the admissible set | `cir/builder/candidates/knobs.py:47-257` | `cir/lowering/default/memory/{bind.py:90, namespaces.py:92, plan.py:155, tile_policy.py:104-203}`, `cir/lowering/search/{extraction.py:554, policies.py:202}` | **carries load** |
| knob kind (subclass) as the knob's role | same | `isinstance` dispatch in `memory/{bind.py:325, namespaces.py:189,235, plan.py:88,123,174, tile_policy.py:291, tiles.py:158}`, `search/extraction.py:552` | **carries load** |
| `ArrayTileKnob`, `NamespaceKnob`, `PortBoundKnob` | `builder/candidates/knobs.py:66-77, 234` | the sites above | carries load |
| `WalkOrderKnob` | `builder/candidates/knobs.py:243-251` | `render/text.py:269` only; walk order is chosen from the `WalkPlan` (`lowering/default/walks/schedule.py:59-73, 171`) | **vestigial** |
| `ByteAddressKnob`, `WordAddressKnob` | `builder/candidates/knobs.py:159-166` | none: "built ... and consumed nowhere" (`lowering/search/__init__.py:46-56`) | **vestigial** |
| `Option` (`RealizationOption`) and the option list | builder; `TableEntry` | `mapping/operations.py`, every `memory/*` site, `search/extraction.py` | carries load |
| `Selection.selected_option_identifier` | `builder/authoring/dsl.py:150`, `mlir_to_cir/lower.py:206`, `mapping/operations.py:185` | `memory/{bind.py:102, namespaces.py:76, plan.py:72, tile_policy.py:118, tiles.py:52}` | **carries load**: the commit |
| `KnobAssignment` values | appended at `memory/bind.py:322-333`, `memory/tiles.py:67` | `render/explorer/graphs.py:468, 539`, `render/text.py:238` only | **write-only**: decisions flow through policies, not assignments |
| `SelectionStatus` | `SELECTED` only (3 sites) | renderers | PARTIAL, UNSELECTED vestigial |
| `Metric` | none | none | **vestigial** |
| `DecisionSpace`, `DecisionPoint` | only inside `TableEntry.__init__` (`table.py:140-144`) | through `TableEntry`; renderers | thin; the validation is what carries load |
| structural and alpha equivalence | | `Function` and `Module` comparisons (`cir/program.py:114-140`) and the test suites | carries load |
| serialization | | the search harness serializes the module once and rebuilds a fresh module per draw (`lowering/search/harness.py:697-698`) | **carries load** |

So the load-bearing core is small: a decision variable with a domain, its
role (kind), alternatives with their variables, the committed alternative,
equivalence, and a lossless round trip. The search itself does not read the
static space's points at all: it re-derives domains and records coordinates
in its own stream (F-SS-013).

### What is missing

| Capability | In MOGA-VM | Established designs |
|---|---|---|
| Conditional, hierarchical spaces (a choice gates sub-decisions) | one level (option → knobs); deeper conditionality only in the dynamic stream; the explorer already expects "a nested `DecisionPoint` embedded inside an option's knobs" (`render/explorer/graphs.py:444`) | Ansor sketches then annotations; ConfigSpace conditions; Optuna define-by-run; MetaSchedule traces |
| Constraints between knobs, within an alternative or across decision points | none; a `Param` constrains its own variable only | Ansor split-factor products; ConfigSpace forbidden clauses; MetaSchedule postprocessors that reject invalid schedules |
| Product and composition of spaces | `CandidateTable`, a MOGA-specific map of decision points, with no relation between entries | ConfigSpace's single space over all hyperparameters; MetaSchedule's whole-module trace |
| Dependencies and order between decision points | implicit in pass order | MetaSchedule's instruction order; Optuna's call order |
| Enumeration and cardinality | none on the static space; per-path products in the stream (`decisions.py:534-544`), one-level structural count in `extraction.py:293-313` | OpenTuner's search-space size; ConfigSpace's grid enumeration |
| Sampling and mutation hooks | in the stream only (`UniformRandomOracle`); none on the space | OpenTuner manipulators (`random`, mutation operators per parameter); Optuna samplers; MetaSchedule mutators |
| Canonical point identity, a hash for caching and dedup | none: identity `==`, identity-based identifiers; the stream invented coordinates to compensate (`decisions.py:39-51`) | MetaSchedule's structural hash of the scheduled IR plus the trace; Optuna's parameter dict |
| Trace and replay of decisions | `SearchPoint`, `RecordingOracle`, `ReplayOracle` in `lowering/search` | MetaSchedule traces replay their decisions; Optuna `enqueue_trial` |
| Separate evaluation results | `Metric` inside the space; `CandidateRecord` in `lowering/search/records.py` | tuning records, trial values with directions |

### Established designs, briefly

- **TVM MetaSchedule:** the space is a program, a *trace* of schedule
  primitives in which sampling instructions (`SamplePerfectTile`,
  `SampleCategorical`, `SampleComputeLocation`) record decisions;
  postprocessors validate and repair the result; mutators edit decisions
  and replay the trace. Conditionality is free; there is no static space
  object.
- **Ansor (TVM auto-scheduler):** sketches (structural alternatives
  generated by rules) then random annotations (tile sizes, unrolling),
  evolutionary search over complete programs with a learned cost model.
  Alternatives-with-parameters is close to option → knobs.
- **Halide autoscheduler (Adams et al. 2019):** a beam search over
  per-stage scheduling decisions made in order, with a learned cost model;
  the space is implicit in the decision procedure.
- **OpenTuner:** a flat set of typed parameters (integer, float, enum,
  boolean, permutation, ...) in a configuration manipulator, with
  per-parameter random and mutation operators; conditionality by
  convention.
- **Optuna:** define-by-run; a trial asks `suggest_*` as the objective
  runs, so the space is conditional by control flow; multi-objective
  studies with directions.
- **ConfigSpace:** a declarative space of hyperparameters with conditions
  (a parameter is active only under a parent value) and forbidden clauses
  (combinations excluded), and configurations as plain dictionaries.

MOGA-VM sits between them: its static vocabulary is ConfigSpace-like but
one level deep and without forbidden clauses, and its dynamic stream is
MetaSchedule-like. The two are not connected.

### Verdict and options

- **(A) Keep the set as is,** with only the audit's fixes. Cheapest for
  MOGA-VM. It ports the conflations (`Selection` inside the space,
  `Metric` inside the space, two names per decision) and adds no
  hierarchy, constraints, identity or enumeration.
- **(B) Keep it, with specific changes** (recommended). The seven names
  survive, so MOGA-VM's migration is mechanical. The changes:
  1. `Option` gains *sub-decisions*: an alternative may own nested
     decision points that exist only when it is selected (hierarchy).
  2. `DecisionPoint` gains a canonical, renaming-invariant point key
     (alternative index and knob values by position, recursively) with a
     hash, and a cardinality for finite spaces.
  3. `Metric` is documented as a declared *estimate* (an expression over
     knob names, a direction by kind); measured results go to a separate
     results layer with the decision stream.
  4. The selection's status is narrowed (N-9), and an assignment refers to
     its knob only (its param is the knob's).
  5. Later, additive: an alternative-level constraint system over knob
     names, a product `SearchSpace` of keyed decision points with
     cross-point constraints and an order (the generic successor of
     `CandidateTable`), sampling and mutation, and the decision stream.
- **(C) A revised set:** `Variable` (knob), `Alternative`, `Choice` (one
  decision's alternatives), `Space` (a product of choices with conditions
  and forbidden clauses), `Configuration` (a point, a map with a canonical
  key), `Trace`, `Measurement`. It is the cleanest model and the closest
  to ConfigSpace plus MetaSchedule, but MOGA-VM would rewrite `TableEntry`,
  its seven construction sites, five selection readers and the renderers,
  and its payloads change shape.

**Decision (the user, 2026-10-05): (C).** The audit had recommended (B),
because it keeps MOGA-VM's migration mechanical. The user chose the revised
set, which removes the conflations outright: the selection leaves the space
as a `Configuration`, and `Metric` leaves it for a results layer. It also
adds conditions, forbidden clauses and a trace. `docs/design/search-space.md`
is built on (C) and gives MOGA-VM's migration map.

---

## Findings

### Critical

#### F-SS-001: Alpha equivalence of categorical and permutation knobs ignores their constraints

- **Severity:** Critical · **Confidence:** High (reproduced on both oracles)
- **Location:** `cir/space/core/equivalence.py:49-58` (`is_param_alpha_equivalent_under`
  short-circuits to the member helpers), `:201-231`
  (`_do_categorical_param_categories_match_under`), `:181-198`
  (permutation); called from `cir/space/core/knob.py:102`.

**Issue:** For a categorical or permutation domain the helper compares only
the members and returns, never reaching `Param.is_alpha_equivalent_under`,
which compares the constraint system under the param's variable. fhy_core
allows in-set and not-in-set constraints on categorical params.

**Evidence (probe A5):** `Knob(k, {1, 2})` against `Knob(k, {1, 2} where x
in {1})`: `is_alpha_equivalent` is **True**, `is_structurally_equivalent` is
False, and the second param refuses `2`. So alpha equivalence holds where
structural equivalence fails and the admitted value sets differ; the
documented contract "structurally-equivalent pairs are always
alpha-equivalent" survives only because it is one-directional.

**Why it matters:** Alpha equivalence is how two CIR modules are compared
(`tests/cir/test_alpha_equivalence.py` compares whole modules, and
`CandidateTable` compares through `TableEntry` and `DecisionPoint`). Any
memoization, deduplication or regression check keyed on it reuses a
decision for a space that admits different values.

**Suggested fix:** Compare members, then the constraint systems under the
variable binder, as `fhy_core::param::Param`'s `AlphaEquivalence` impl does
(`rust/fhy-core/src/param/parameter.rs:599-625`).

### High

#### F-SS-002: Decision-point alpha equivalence is not symmetric when one side reuses a label

- **Severity:** High · **Confidence:** High
- **Location:** `cir/space/core/equivalence.py:310-345` (`collect_decision_point_label_bindings`
  builds a `dict`, so a repeated left label keeps only its last pairing),
  `cir/space/core/option.py:97-117` (the same for knob names),
  `cir/space/core/decision.py:411-418` (a non-injective right side makes
  `extend` raise, caught as `False`).

**Issue:** A label repeated on the left collapses silently; the same label
repeated on the right is a non-injective map and fails. The outcome depends
on which side holds the repetition.

**Evidence (probes A1, A1b; sweep):**
- `dp(options=(A, B))` vs `dp(options=(S, S))`: **False**; reversed: **True**.
- An option listing one knob twice, `o[k, k]`, vs `o[k1, k2]`: **True**;
  reversed: **False**.
- The random sweep: 1,162 asymmetric verdicts among 2,879 decision points.

**Why it matters:** `AlphaEquivalence` requires reflexivity, symmetry and
transitivity (`src/fhy_core/term/alpha_equivalence.py`, protocol
docstring). An asymmetric relation makes the answer depend on argument
order, so a set or cache of "equivalent" spaces is not well defined.

**Suggested fix:** Refuse repeated labels where they are binders (option
names within a space, knob names within an option, assigned knobs within a
selection) at construction, and pair binder lists with
`AlphaRenaming::enter_binders` (`rust/fhy-core/src/term/renaming.rs:155-190`),
which refuses repeats on either side. See F-SS-007 for the scoping that
makes a knob shared by two options legitimate.

#### F-SS-003: Identifier domain members are matched by `resolve`, which lets a free name capture a bound one

- **Severity:** High · **Confidence:** High
- **Location:** `cir/space/core/equivalence.py:249-277` (identifier and
  identifier-tuple categories resolved on the left, then compared to the
  right set by `frozenset` equality).

**Issue:** `renaming.resolve` looks only at the left side. A left member
that is free resolves to itself and matches a right member with the same
identity even when that right identifier is bound by a frame, which
`are_identifiers_alpha_equivalent` would refuse.

**Evidence (probe A7):** Left: option `A` with a knob over `{Z}`, `Z` free.
Right: an option *named* `Z` with a knob over `{Z}`. Under the decision
point's bindings `{A: Z}`, the left `Z` resolves to itself and matches:
**True**. The right `Z` refers to the option binder; the left one to a free
name.

**Why it matters:** A capture is the textbook unsoundness of alpha
equivalence. It is rare with fresh identifiers, but MOGA-VM reuses
identifiers on purpose (namespace names, walk axes, vertex ids as decision
point and space names).

**Suggested fix:** Decide each member pair with `is_corresponding`: for a
categorical set, every left member must correspond to a right member and
the sizes agree (frames are injective, so this is a bijection).

#### F-SS-004: Python `==` makes `1`, `True` and `1.0` alpha-equivalent members and values

- **Severity:** High · **Confidence:** High
- **Location:** `cir/space/core/equivalence.py:229` (`frozenset` equality of
  categories), `:307` (`left == right` for assignment values).

**Issue:** Python equality unifies `1 == True == 1.0`. fhy_core 0.2 matches
values type-strictly everywhere (S16's P-9), and the structural relation
here uses the param's own type-strict comparison.

**Evidence (probe A6):** knob over `{1}` vs knob over `{True}`: alpha
**True**. `KnobAssignment(q, {1}.assign(1))` vs `KnobAssignment(q,
{True}.assign(True))`: alpha **True**, structural False.

**Why it matters:** Boolean knobs (`unroll: {True, False}`) and integer
knobs (`{0, 1}`) are common; they compare equal in alpha mode only. Like
F-SS-001, alpha accepts what structural refuses.

**Suggested fix:** Type-strict member and value comparison, which the Rust
`Value` gives for free.

#### F-SS-005: Selections are validated incompletely: duplicates, conflicting and out-of-domain assignments pass

- **Severity:** High · **Confidence:** High
- **Location:** `cir/space/core/decision.py:347-383` (`_validate_selection`).

**Issue and evidence:**
- **Duplicate option identifiers** in one space are accepted, and
  validation looks the selected option up in a dict that keeps the last
  (`:360`). Probe V1: options `(dup[kx], dup[ky])`, a selection of `dup`
  assigning `kx` is refused as "unknown knob" although the first `dup`
  declares it.
- **Duplicate knob identifiers** in one option are accepted (probe V2).
- **Conflicting assignments** pass: assigned names are collected into a set
  (`:369-371`), so `kx = 1` and `kx = 2` in one selection are accepted
  (probe V3). A real path produces this: `cir/lowering/default/memory/bind.py:318-334`
  appends namespace assignments to whatever the selection already holds.
- **Out-of-domain values** pass: nothing checks that an assignment's param
  is the knob's param, so `KnobAssignment(kx, {99}.assign(99))` is accepted
  for a knob over `{1, 2}` (probe V4).

**Why it matters:** A `DecisionPoint` is the committed point a lowering
realizes. Every one of these produces a point that is not a point of its
own space, and downstream code reads the first or last assignment it finds.

**Suggested fix:** Unique option names per space, unique knob names per
option, unique assigned knobs per selection, and assignment param equal to
the knob's param, all checked at construction with distinct errors.

#### F-SS-006: The core cannot round-trip on its own: no concrete `Option`, `Knob` writes an unreadable type id

- **Severity:** High · **Confidence:** High
- **Location:** `cir/space/core/option.py:52-58` (`Option` is an unregistered
  ABC), `cir/space/core/knob.py:51-58` (`Knob` unregistered), `:203-205`
  (`ArrayKnob`), `cir/space/core/decision.py:184-191, 268-270` (`Selection`
  root, `ArraySelection` the only concrete).

**Evidence (probe S1):**
- `Option(name=...)` raises `SerializationDerivationError` on both
  versions: the core has no constructible option at all, so every core test
  uses MOGA's `RealizationOption`.
- `Knob(...)` constructs and serializes under its default type id
  `moga_vm.cir.space.core.knob.Knob` (the module path), and reading it back
  raises `UnknownTypeIdError` because the class is not registered.
- Every registered generic class carries a `moga.cir.*` type id, and the
  generic concrete classes carry MOGA names (`ArrayKnob`, `ArraySelection`).

**Why it matters:** "Promotable unchanged" requires the core to describe and
persist a space without the downstream package. Today a space of plain
knobs writes a payload nothing can read.

**Suggested fix:** Concrete, registered `Knob`, `Option` and `Selection` in
the core under neutral type ids; MOGA's markers stay in MOGA-VM.

### Medium

#### F-SS-007: Knob names have two inconsistent binding scopes; equivalent points compare unequal

- **Severity:** Medium (too strict, not unsound) · **Confidence:** High
- **Location:** `cir/space/core/knob.py:100` (each knob binds its own name
  for its own param), `cir/space/core/equivalence.py:336-342` (every knob
  name of every option is also bound in one decision-point-wide frame),
  `cir/space/core/decision.py:228-240` (a selection's knob references resolve
  through that wide frame).

**Issue:** Knob names are scoped per option for the knob's own comparison,
but per decision point for the selection's references. A knob identifier
shared by two options (one `Knob` object reused, which nothing forbids) is
bound once in the wide frame, to whichever right-hand knob came last.

**Evidence (probe A2):** left options `la[ks]`, `lb[ks]` with a selection
of `la` assigning `ks = 1`; right options `ra[k1]`, `rb[k2]` selecting `ra`
with `k1 = 1`. The two are equal up to renaming; the verdict is **False**
both ways. The sweep's 904 non-equivalent fresh relabelings are this
finding and F-SS-002 together.

**Suggested fix:** A selection's knob references resolve in the scope of the
*selected* option: compare assignments under that option's knob frame
only.

#### F-SS-008: The core no longer imports against fhy_core 0.2; "promoted unchanged" is false

- **Severity:** Medium · **Confidence:** High
- **Location:** every core module (`fhy_core.param`, `fhy_core.expression`,
  `fhy_core.constraint`, `fhy_core.traits.AlphaRenaming`,
  `fhy_core.traits.AlphaEquivalenceMixin`); `cir/space/knobs.py:282`.

**Issue:** fhy_core 0.2 moved `param`, `expression` and `constraint` under
`fhy_core.symbolic`, and the renaming and alpha mixin to `fhy_core.term`.
`EquationConstraint` takes one expression. MOGA-VM's `requirements.txt`
pins `fhy-core>=0.1.8`, which now resolves to a version the code cannot
import.

**Evidence:** `AttributeError: module 'fhy_core' has no attribute 'param'`;
with four module aliases (`oracle/compat/fhy_compat.py`) all 44 core tests
pass on 0.2, so the drift is in names, not in the core's semantics.

**Suggested fix:** None in the port itself; MOGA-VM's migration to the
ported classes removes these imports. Pin `fhy-core<0.2` in MOGA-VM until
then.

#### F-SS-009: Identifier-valued domain members are opaque in fhy_core 0.2 and are never renamed

- **Severity:** Medium · **Confidence:** High
- **Location (fhy-core):** `rust/fhy-core/src/constraint/value.rs:102-118`
  (`Value` has no identifier kind), `rust/fhy-core-py/src/constraint/value.rs:149`
  (`PyOpaqueValue`), `rust/fhy-core/src/param/parameter.rs:599-625`
  (domains compared structurally in alpha mode).
- **Location (MOGA-VM):** `cir/space/core/equivalence.py:249-277` relies on
  renaming identifier members; `cir/space/realization_option.py:42-50, 100-127`
  binds walk axes that `WalkOrderKnob`'s permutation members reference.

**Issue:** A Python `Identifier` inside a categorical or permutation domain
reaches the Rust core as an opaque value (`{"opaque": {"type_id": "id",
...}}`). `Param.is_alpha_equivalent_under` therefore never renames it:
`{a}` against `{b}` under the free renaming `{a: b}` is False on 0.2.0.
MOGA-VM needs member renaming for walk orders. A Rust-native consumer cannot
build such a domain at all.

**Why it matters:** The port cannot express MOGA's walk-order knobs
faithfully, nor fix F-SS-003, without a way for the core to see identifier
members.

**Suggested fix:** A user decision (design N-5): a first-class identifier
kind in `fhy_core::constraint`'s values and members, or a narrower hook.

#### F-SS-010: `SelectionStatus` has no checkable meaning beyond UNSELECTED-with-assignments

- **Severity:** Medium · **Confidence:** High
- **Location:** `cir/space/core/decision.py:139-145, 196, 352-353, 379-383`.

**Issue and evidence (probe V5):** SELECTED with no assignment, PARTIAL with
full coverage, and PARTIAL with no assignment are all accepted. UNSELECTED
still requires `selected_option_identifier`, which must name an existing
option, so "unselected" names a choice; and `selection=None` is a second
encoding of "no choice". In MOGA-VM's sources only SELECTED is ever
constructed (`cir/builder/authoring/dsl.py:151`,
`cir/lowering/default/memory/bind.py:332`, `mlir_to_cir/lower.py:207`);
UNSELECTED is only the default and PARTIAL is unused.

**Why it matters:** A status nothing enforces cannot be relied on by a
search or a verifier.

**Suggested fix:** A user decision (design N-9): document the parity
meaning, or remove the redundant encodings.

#### F-SS-011: Knob name and param variable are two names for one variable; assignments carry a third

- **Severity:** Medium · **Confidence:** High
- **Location:** `cir/space/core/knob.py:61-62, 72-82` (a knob's `name` and its
  `param.variable` are unrelated identifiers unless the caller makes them
  equal, as `cir/space/knobs.py:255-283` does), `:147-181` (`KnobAssignment`
  holds `knob_identifier` and a whole `ParamAssignment`).

**Issue:** Alpha equivalence binds the knob name (`knob.py:100`) and,
inside `Param`, the variable; constraints and any metric expression can only
reference one of them coherently, and nothing says which. `KnobAssignment`
alpha ignores its own param (`:170-181`) while structural compares it
(`:162-168`), and nothing checks it against the knob's (F-SS-005).

**Suggested fix:** State that expressions outside the param reference the
knob name; check an assignment's param against its knob; or build a knob
from its param's variable (design N-10).

#### F-SS-012: Metric names are unbound references with an undocumented role; metrics have no producer

- **Severity:** Medium · **Confidence:** High
- **Location:** `cir/space/core/metric.py:14-24, 61-67, 84-93`;
  `cir/lowering/search/decisions.py:33-37`.

**Issue and evidence (probe A4):** The metric's name is compared through
the renaming but bound by nobody, so it is a free reference: two options
built independently, each with its own `Identifier("latency")`, are not
alpha-equivalent. That is right if metric names form a shared vocabulary
and wrong if each metric is a local label; the code documents neither.
No source module constructs a `Metric` (only docstrings mention
`MetricKind.COST`), and `decisions.py:33-37` says so. A metric's value is an
arbitrary `Expression` with no unit, no aggregation rule, no relation to the
knobs it might depend on, and DIAGNOSTIC with literal `0` as the default.
`Selection.metrics` are unrelated to the selected option's metrics.

**Why it matters:** Multi-objective search needs a fixed objective
vocabulary with directions; this one is a placeholder.

**Suggested fix:** Port the shape, document metric names as references to a
metric vocabulary, and design objectives with the search slice (design
N-3).

#### F-SS-013: Two vocabularies: the static space cannot be enumerated, indexed or sampled

- **Severity:** Medium (design) · **Confidence:** High
- **Location:** `cir/lowering/search/decisions.py:10-23, 39-51` (the stream
  exists because the static space cannot express conditional decisions, and
  because identity-equal objects cannot be recorded), `extraction.py:24-33,
  109-121, 293-313` (axes re-derived from the table, one level of
  conditionality), `decisions.py:93-120` (`DecisionKind` is a closed,
  MOGA-specific enum).

**Issue:** The search does not consume `DecisionSpace` or `Knob` domains.
It rebuilds `ChoiceDomain`, `OrderDomain` and `AddressDomain` from the table
and the pipeline, numbers them with coordinates, and replays by position.
The static space has no cardinality, no coordinates, no cross-knob
constraints and no nesting deeper than option → knobs. See "Is this the
right component set?" above.

**Suggested fix:** Port the static core first, with a scoping model a
coordinate system can later be laid over; port the decision stream as its
own slice with an open decision kind.

### Low

#### F-SS-014: Empty decision spaces are accepted, unlike every neighboring abstraction

- **Location:** `cir/space/core/decision.py:76-86`; compare `cir/space/table.py:136-139`
  (`TableEntry` refuses zero options) and `cir/lowering/search/decisions.py:146-148`
  (`ChoiceDomain` refuses zero choices).
- **Evidence (probe V6):** `DecisionPoint(space=DecisionSpace(options=()))`
  constructs. No selection can be valid for it.
- **Suggested fix:** Refuse, or document an empty space as "infeasible"
  (design N-3).

#### F-SS-015: Default names mint a fresh identifier per construction

- **Location:** `cir/space/core/decision.py:84, 195, 339`, `option.py:61`,
  `knob.py:62`.
- **Evidence (probe V7):** two `Knob(param=p)` get different names, so
  default-built values are never structurally equivalent, and a missing
  name goes unnoticed.
- **Suggested fix:** Keep the defaults for parity, or require names (design
  N-3).

#### F-SS-016: Kind checks mix `type(self) is type(other)` and `isinstance`; helpers dispatch on the left

- **Location:** `metric.py:77, 88` and `knob.py:164, 174` use `isinstance`;
  `option.py:75, 85`, `knob.py:86, 95` and `decision.py:97, 106, 212, 226,
  394, 406` require equal classes; `equivalence.py:65-89, 114-135` call the
  left element's method only.
- **Issue:** A subclass of `Metric` or `KnobAssignment` that overrides
  equivalence makes the relation order-dependent. No such subclass exists
  today.
- **Suggested fix:** One kind rule (equal kinds) for every node.

#### F-SS-017: Labels are re-bound at several levels; standalone and nested verdicts disagree

- **Location:** `decision.py:109` (space name, already bound by the decision
  point), `:231` (selection name, likewise), `knob.py:100`, `option.py:88`.
- **Issue:** Harmless under frame shadowing, but it hides the scoping model.
  A standalone `DecisionSpace` comparison binds no option labels, so it says
  **True** for `(A, B)` against `(S, S)` while the enclosing decision point
  says False (probe A1).
- **Suggested fix:** One binding site per label, the same standalone and
  nested.

#### F-SS-018: Option order is significant, but the domain is documented as a set

- **Location:** `equivalence.py:65-89, 330-342` (positional pairing);
  `decisions.py:125-137` (`ChoiceDomain`: "Order carries no meaning to the
  domain itself").
- **Issue:** Two spaces offering the same options in another order are not
  equivalent. Positional order is what coordinates index, so this is
  defensible, but undocumented.
- **Suggested fix:** Document options and knobs as ordered sequences.

#### F-SS-019: Implementation helpers and payload shapes are exported as API; MOGA type ids on generic classes

- **Location:** `cir/space/core/__init__.py:12-43`; type ids at
  `decision.py:74, 268, 328`, `knob.py:147, 203`, `metric.py:51`.
- **Issue:** Seventeen comparison helpers, shape predicates and TypedDicts are
  public; fhy-core's port policy deletes such helpers (python-switch D-S16-1).
  Payload keys differ from attribute names (`identifier` for `name`,
  `selected_identifier` for `selected_option_identifier`).
- **Suggested fix:** Drop the helpers from the public API; keep the payload
  keys for compatibility; neutral type ids.

---

## The CIR-specific layer, and what the trait contracts must prevent

By the user's decision, `cir/space/knobs.py`, `realization_option.py` and
`table.py` stay in MOGA-VM as implementations of the generic vocabulary.
They are outside the port, but they show what an implementor gets wrong
when the base classes leave the comparison to each subclass. Each item is a
contract the design's traits state.

| # | Observation (MOGA-VM) | Location | Contract that prevents it |
|---|---|---|---|
| C-1 | `RealizationOption` compares its realization under the *incoming* renaming, then extends it with the walk axes; inside a decision point the axes were already bound by `collect_alpha_label_bindings`. Standalone and nested comparisons use different frames. | `cir/space/realization_option.py:74-98, 100-127` | An alternative *declares* the labels it binds (`bound_identifiers`); the core enters one frame and compares the extension under it, the same standalone and nested. |
| C-2 | `_extend_with_walk_axes` returns `None` on a non-injective pairing and the unchanged renaming when there are no axes: the F-SS-002 asymmetry again, per subclass. | `realization_option.py:100-113` | Declared binder lists are pairwise distinct; the core pairs them with `enter_binders`. |
| C-3 | `TableEntry` checks its vertex id as a *reference* (`table.py:183-186`), then the wrapped decision point re-binds the same id as its name and its space's name, a *binder*. | `cir/space/table.py:140-144, 175-195` | Each label has one role. A wrapper that needs a reference id keeps it outside the decision point's own labels. |
| C-4 | `TableEntry` compares partitions by left `resolve` and set equality: F-SS-003's capture. | `table.py:68-88` | Hooks consult the renaming only through `is_corresponding`. |
| C-5 | Knob kinds carry the knob's role, read by `isinstance` at ten lowering sites; a class's identity is load-bearing. | see "How MOGA-VM uses the components" | A kind is stable and unique per implementing type; two values of one kind are the same type. |
| C-6 | Passes *append* assignments to a selection (`memory/bind.py:322-333`, `memory/tiles.py:67`), which produces duplicate knob assignments that today's validation accepts (F-SS-005). | as cited | A `Configuration` holds one value per decision, so a duplicate cannot be expressed. |
| C-7 | Knobs that are facts, not choices (single-category `NamespaceKnob`), and knobs nobody reads (`WalkOrderKnob`, both address knobs). | `lowering/search/decisions.py:25-31`; `lowering/search/__init__.py:46-56` | None needed; a cardinality of 1 identifies a fact, and the vestigial kinds are MOGA-VM's to remove. |
| C-8 | `PortBoundKnob` and `ArrayTileKnob` compare their extra fields correctly (equality and `are_identifier_sequences_alpha_equivalent_under`), and each calls `super()` for the base fields. Correctness depends on every subclass remembering to. | `cir/space/knobs.py:100-150, 176-230` | The core compares the base fields; a hook compares only the implementor's own data. |

## Open questions

1. Are option, knob, space and decision-point names *labels* (alpha-renamable
   binders) or *names* (identity matters)? The code treats them as binders;
   `TableEntry` reuses the vertex id as decision-point and space name and
   checks it as a reference first (`table.py:175-195`). The design assumes
   binders.
2. Are metric names a shared vocabulary (references) or per-option labels
   (binders)? (F-SS-012)
3. What does PARTIAL mean, and may SELECTED leave knobs unassigned? The test
   `test_decision_point_accepts_selected_status_with_incomplete_coverage`
   says yes for the allocator's sake. (F-SS-010)
4. Should a knob's domain members that are identifiers be references (MOGA
   walk orders) or values (fhy_core 0.2's `Param`)? (F-SS-009)

## Existing tests review

- **Strong:** `test_structural_equivalence.py` pairs every reflexivity case
  with a discriminating perturbation and checks both directions;
  `test_selection_validation.py` asserts the exact exception class and the
  offending identifiers in the message.
- **Gaps that hid the findings:**
  - no test uses a label twice, so F-SS-002, -005 and -007 are untested;
  - no test puts a constraint on a categorical knob (F-SS-001) or a `bool`
    member (F-SS-004);
  - no test compares alpha equivalence in both directions for decision
    points, only for knobs, options and spaces;
  - every "core" test needs MOGA's `RealizationOption` and `HardwareDFG`,
    because the core has no concrete option (F-SS-006);
  - `test_core_import_boundary.py` parses imports, so it passes while the
    imports no longer resolve (F-SS-008).
- **Locks in a suspected bug:** `test_decision_point_accepts_selected_status_with_incomplete_coverage`
  pins F-SS-010's open semantics; keep it only if N-9 keeps parity.

## Triage

### (a) Fix in the port as approved divergences (adopted, N-3; restated for the (C) design)

| ID | Fix in the port | Behavior change visible to MOGA-VM |
|---|---|---|
| F-SS-001 | knob params compared member-wise, then constraints under the variable binder | some spaces stop being alpha-equivalent |
| F-SS-002 | names unique space-wide; one frame paired by `enter_binders` | duplicate names refused; verdicts symmetric |
| F-SS-003 | member correspondence by `is_corresponding` | captures refused |
| F-SS-004 | type-strict values | `1`, `True`, `1.0` distinct |
| F-SS-005 | `Space::new` and `Configuration::new` checks | invalid spaces and configurations refused |
| F-SS-006 | concrete registered `PlainVariable`, `PlainAlternative`; `search_space.*` type ids | MOGA kinds registered by MOGA |
| F-SS-007 | one space-wide frame; a configuration names decisions directly | one knob identifier can no longer serve two alternatives (names are unique) |
| F-SS-016 | equal kinds everywhere | none today |
| F-SS-017 | one binding site per label | standalone verdicts agree with nested ones |
| F-SS-019 | helpers private; payload keys kept | helper imports removed |

### (b) Need a user decision

Decided by the user on 2026-10-05 (see the design's "Earlier decisions,
translated to (C)"): F-SS-009 (SS0, identifier members), F-SS-013 and
F-SS-018 (component set (C)), F-SS-010 (the status gives way to
configuration validity and completeness), F-SS-011 (a `Variable`'s name vs
its param's variable), F-SS-012 (`Metric` leaves the space), F-SS-014 (empty
choices refused), F-SS-015 (default names kept in Python).
