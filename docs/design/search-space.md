# Search space: a generic search-space vocabulary for fhy-core, from MOGA-VM's

- **Status:** designed 2026-10-05 on `feat/search-space` (`6548984`), revised
  the same day for the user's decisions. Every decision is recorded under
  "Decisions". Implementation starts with SS0. SS0 and SS1 are landed;
  SS2 and SS3 were planned on 2026-10-06 ("SS2: traces, oracles,
  enumeration and sampling (plan)", "SS3: objectives and measurements
  (plan)"); their choices (N-S1 to N-S5) were decided by the user the same
  day, under "Decisions".
- **Decided by the user (2026-10-05):**
  - Only generic parts move into fhy-core. Everything CIR-specific stays in
    MOGA-VM as subclasses or implementors of the generic vocabulary.
  - The core is built on traits that downstream structs implement.
  - **N-1 (C):** the revised component set `Variable`, `Alternative`,
    `Choice`, `Space` (with conditions and forbidden clauses),
    `Configuration` (replacing `Selection` and `DecisionPoint`), `Trace`
    and `Measurement`.
  - **N-4 (a):** both extension paths: subclassable pyclasses with hooks,
    and a kind registry in `convert::search_space` as append-only module
    state (approved).
  - **N-5 (a):** the SS0 slice, a first-class identifier kind in
    `fhy_core::constraint`'s values and members.
  - The recommendations for N-2, N-3, N-6 to N-10 were accepted; their
    translation to (C) is under "Earlier decisions, translated".
- **Source:** MOGA-VM `origin/dev` at `3d93ba3`, read-only:
  `src/moga_vm/cir/space/core/` and, for the trace, `cir/lowering/search/`.
  MOGA-VM pins `fhy-core>=0.1.8`; its tests pass on v0.1.8 (`ad9a311`).
- **Audit:** `docs/audit/search-space.md`: the component-set evaluation,
  F-SS-001 to F-SS-019, and the CIR-layer contracts C-1 to C-8.
- **Paths:** MOGA-VM paths are relative to its `src/moga_vm/`. References
  such as D-S11-9 and P-5 are to the S-slices of
  `docs/design/python-switch.md` (local, untracked).

## The component set

| Component | Rust | Role |
|---|---|---|
| `Variable` | **trait**, plain `PlainVariable` | a decision variable: a name over a `Param` (domain plus constraints over the param's own variable) |
| `Alternative` | **trait**, plain `PlainAlternative` | one option of a choice: a name, the variables and sub-choices that exist only when it is chosen, and identifiers it binds |
| `Choice` | struct | a named decision among one or more alternatives |
| `Space` | struct | the whole space: top-level variables and choices, conditions, forbidden clauses; the scoping and validation boundary |
| `Condition` | struct | when a variable or choice is active, as fhy_core constraints |
| `Forbidden` | struct | a combination of values no configuration may take, as fhy_core constraints |
| `Configuration` | struct | a point: one value per assigned, active decision, validated against a `Space`; partial or complete |
| `ConfigurationKey` | struct | the canonical, renaming-invariant identity of a configuration within its space (`Eq + Hash`) |
| `Trace`, `TraceStep` | structs | the replayable, ordered record of the decisions that produced a configuration |
| `Objective`, `Direction` | struct, enum | what is measured and which way is better |
| `Measurement` | struct | the result of evaluating one configuration: a value per objective and a status |

### Traits, and why not more of them

- **Traits:** `Variable` and `Alternative`. These are what MOGA-VM extends
  with its own data: eight knob classes and `RealizationOption`
  (`cir/space/knobs.py`, `cir/space/realization_option.py`).
- **`Choice`: a struct.** It holds only a name and its alternatives. MOGA-VM
  adds nothing to it: `TableEntry`'s vertex id becomes the choice's name,
  and its realized partition is post-commit bookkeeping that belongs beside
  the choice, not in it. A choice is also the binder of its alternatives'
  names, and so part of the scoping the core must own (C-1 to C-4).
- **`Space`: a struct.** It is where uniqueness, acyclicity, condition
  scopes and binder frames are checked once for everything implementors
  supply. A trait would make every implementor re-establish those. MOGA-VM's
  `CandidateTable` wraps a `Space` instead.
- **`Measurement`: a struct.** Results are data. The open behavior is
  producing them (a profiler, a simulator, a cost model), which is a
  `Measurer` trait in a later slice. A downstream record (MOGA-VM's
  `CandidateRecord`) wraps a `Measurement`.
- **Trace steps: structs.** A step is data: a decision, its domain's
  signature, and a coordinate. The open parts are the decision *kind*, an
  open vocabulary (a namespaced string, stable across processes; SS2's
  plan says why not an interned tag), the `SearchOracle` trait that
  answers steps, and `Variable::search_domain` for a variable whose param
  has a custom domain. The domain shapes (choice, order, strided runs) are
  closed.
- **`Measurer`: a trait** (SS3), generic over what it measures.
- **Conditions and forbidden clauses: structs** over `constraint::Constraint`,
  whose `Custom` variant is already fhy_core's extension point for
  Python-defined and Rust-defined constraints.

## Scope and slicing

| Slice | Contents |
|---|---|
| **SS0** | identifier values and members in `fhy_core::constraint` (N-5 (a)) |
| **SS1** (recommended first (C) slice) | `Variable`, `Alternative` (traits and plain implementations), `Choice`, `Space` with hierarchy, `Condition`, `Forbidden`, `Configuration` with validation, completeness and `ConfigurationKey`; equivalence; serialization; the Python interface with both extension paths |
| **SS2** | `Trace` and `TraceStep` over the static space *and* over dynamic decisions (choice, order and strided-run domains, an open decision kind); `SearchOracle`, the random, replay and exhaustive oracles, the `Recorder`; enumeration and cardinality; sampling (per step and uniform) and mutation; the `Rng` |
| **SS3** | `Objective`, `Direction`, `Measurement`, a `Measurer` trait; declared estimates deferred (N-S3) |
| stays in MOGA-VM | its knob classes, `RealizationOption`, `TableEntry`, `CandidateTable` (as wrappers of `Choice` and `Space`), the lowering policies, placement, harness, records and observers |

**Why `Trace` and `Measurement` are not in SS1** (N-C4). MOGA-VM's real use
of a trace is the dynamic stream, whose decisions (addresses, boundary
namespaces) are not in any static space (`cir/lowering/search/decisions.py:10-37`).
A static-only `Trace` in SS1 would be redesigned in SS2. `Measurement` has
no producer in fhy-core or MOGA-VM today (audit F-SS-012), and its design
depends on how SS2's harness records results. SS1 still fixes the identity
both will key on: `ConfigurationKey`.

## Survey: the Python API being replaced

Public by `__all__` (`cir/space/core/__init__.py:12-43`); "External users"
are MOGA-VM modules outside the core (`git grep` on `origin/dev`).

| Python symbol | External users | Becomes |
|---|---|---|
| `Knob`, `ArrayKnob` | 8 subclasses in `cir/space/knobs.py` | `Variable`; MOGA kinds implement it |
| `KnobAssignment` | 9 files | one entry of a `Configuration` |
| `Metric`, `MetricKind` | docstrings only | gone from the space; `Objective` and `Direction` in SS3 (N-C3) |
| `Option` | `RealizationOption` | `Alternative` |
| `DecisionSpace` | `TableEntry` | `Choice` |
| `Selection`, `ArraySelection`, `SelectionStatus` | 8 files, 7 files | `Configuration` (validity and completeness replace the status) |
| `DecisionPoint` | `TableEntry`, renderers | gone: a `Choice` lives in a `Space`; its value lives in a `Configuration` |
| `SelectionConsistencyError` | tests | `ConfigurationError` |
| `are_*`, `is_param_alpha_equivalent_under`, `collect_decision_point_label_bindings`, `is_valid_*_data`, `*Data` | `knobs.py`, `realization_option.py` | deleted (D-SS-9) |
| `Option.collect_alpha_label_bindings` | `RealizationOption` | `Alternative::bound_identifiers` |

## Foundations reused, exactly

| Need | fhy_core item | Use |
|---|---|---|
| names | `identifier::Identifier` | every name; binders |
| frames | `term::AlphaRenaming::{enter_binders, extended, is_corresponding}`, `term::AlphaEquivalence` | alpha equivalence |
| domains | `param::Param` (domain, constraints over its variable, alpha binding its variable), finite domains' `values()`, `ParamAssignment::new` / `restore` | a variable's domain; checking a configuration's values |
| conditions, forbidden clauses | `constraint::Constraint` (`Equation`: a Boolean `expression::Expression` over identifiers; `Set`: membership of one identifier's value; `Custom(Part<dyn CustomConstraint>)`), `ConstraintSystem::{new, evaluate, is_structurally_equivalent}` and its `AlphaEquivalence`, `Constraint::free_identifiers`, `constraint::Bindings`, `ConstraintContext`, `Outcome` | the condition and forbidden semantics below; evaluated with the configuration's values as bindings |
| choice values in constraints | SS0's identifier `Value` and member | a choice's value is the identifier value of its chosen alternative's name, so `Set(choice in {a, b})` works unchanged (equality is `in {a}`); an `Equation` cannot name a choice, since SS0 refuses an identifier binding in an equation (`UnusableBinding { reason: NotALiteral }`), and `Space::new` refuses one that does (N-C5) |
| notes | `diagnostic::Note` | on every component |
| implementor parts | `foreign::{ForeignPart, Part, Foreign, Resolve, NoForeign, BoxError}`, `impl_part!` | `Part<dyn Variable>`, `Part<dyn Alternative>` |
| big counts | `num-bigint` (already a dependency) | SS2's cardinality |
| Python classes | binding `util::{public_class, frozen, foreign, gc, hook, serialization, exceptions}`; D-S11-11's subclassable pyclass base | P2 classes, P3 adapters |
| serialization | `WrappedFamilySerializable`, `register_serializable(type_id=, alias=)`; V2 from the core's serde shape, foreign parts as `{"type_id", "data"}` | as below |

## SS0: identifier values and members (plan)

The concrete plan for SS0, in fhy-development-rs's planning template. The
stub follows it.

### Summary

`fhy_core::constraint`'s values and members gain an identifier kind. A
Python `Identifier` used as a param category, a permutation member, a
set-constraint member or a bound value reaches the core as that kind,
instead of as an opaque Python object.

### Motivation

- In 0.2.0 an identifier member is a `PyOpaqueValue`
  (`{"opaque": {"type_id": "id", ...}}`). The core cannot see it as an
  identifier: Rust code cannot build such a domain, and SS1 cannot compare
  members by correspondence (audit F-SS-009).
- Its canonical order follows the `repr` text of its payload, so id `100`
  sorts before id `99`.

### Placement

| Crate | Files |
|---|---|
| `fhy-core` | `constraint/value.rs` (`Value`, `Member`, `MemberKind`, `OpaqueValue`, order, equality, hash, display); `constraint/wire.rs` (the shape, legacy normalization); `constraint/equation.rs` (an identifier binding is no literal); `constraint/key.rs` (member key); `param/value.rs`, `param/domain.rs`, `param/wire.rs` (member conversion, leaf check, ordinal order) |
| `fhy-core-py` | `constraint/value.rs` (readers and writers; `PyOpaqueValue::identifier`); `param/value.rs` (the finite-domain reader) |
| Python | docstrings only: `serialization.serialize_value`'s V2 shapes and `symbolic/constraint/members.py`'s canonical order |

No new modules and no new dependencies.

### Public API

| Item | Change | Semver |
|---|---|---|
| `constraint::Value::Identifier(Identifier)` | new variant | additive: `Value` is `#[non_exhaustive]` |
| `constraint::MemberKind::Identifier(&'a Identifier)` | new variant | **breaking**: `MemberKind` is exhaustive (its `#[expect(clippy::exhaustive_enums)]` says the binding converts every kind) |
| `constraint::OpaqueValue::identifier(&self) -> Option<Identifier>` | new provided method, default `None` | additive for implementors |
| wire shape `{"identifier": {"id": .., "name_hint": ..}}` | new tag of `ValueData` and of members in `ConstraintData` and `param::wire`, declared last so the other tags keep their indices in non-self-describing formats | additive for readers; writers now write it |

`OpaqueValue::identifier` is the identifier an opaque value stands for.

- It exists for payloads written before this kind: a resolver turns
  `{"opaque": {"type_id": "id"}}` into an opaque value, and the wire
  forms' `build` reads every opaque value that reports an identifier as the
  identifier kind.
- The core's own decoding cannot parse the foreign text, since it does not
  depend on `serde_json`.

### Visibility

- No new `pub` item other than those above.
- The rank table and comparisons stay private.

### Behavior

| Question | Answer |
|---|---|
| equality | type-strict: an identifier equals an identifier with the same id (`Identifier`'s `==`), and nothing else, not a string with its name hint |
| hash | the identifier's hash, after the kind |
| canonical member order | kinds by name: `bool`, `float`, `frozenset`, `identifier`, `int`, `str`, `tuple`, then opaque values; identifiers by id ascending |
| `Display` | the name hint, as `Identifier`'s `Display` |
| `Member::kind_name` | `"identifier"` |
| member-shaped | yes; no NaN, decimal or ordering-key concerns |
| lifts to an expression | no. An identifier expression names a variable, not a constant, so `x in {a}` does not lower to `x == a`, and `SetConstraint::to_expression` refuses it with `UnliftableMember` |
| a constraint's free identifiers | unchanged: a member identifier is a constant, not a variable occurrence. (This revises the design's first sketch, which counted it; counting it would make a param treat `x in {a}` as depending on `a`, and leave it undecided.) |
| alpha equivalence of set constraints | unchanged: members compare by value. SS1 decides correspondence for its conditions. |
| an equation's binding | an identifier value is not a literal: `UnusableBinding { reason: NotALiteral }`, as a tuple's |
| ordinal domains | identifiers do not order: `compare_ordinal` answers `None`, as for unrelated kinds. Python's ordinal reader still refuses an `Identifier`, as today. |
| categorical and permutation domains | an identifier is a leaf value |
| member key (`constraint/key.rs`) | `identifier:<id>` |
| wire, writing | `{"identifier": {"id": 60000, "name_hint": "a"}}`, in both value and member positions |
| wire, reading | the new tag, and `{"opaque": <part>}` whose resolved part reports an identifier |
| Python reading | an `Identifier` (exact class or subclass, by `read_identifier_id`) becomes the identifier kind in member and bound-value readers |
| Python writing | an identifier value or member becomes a Python `Identifier` through `identifier_to_python`. It is equal by id to the one given, not the same object. |

### Ownership and data model

`Identifier` is a cheap clone (an id and a shared name hint). The variant
holds it by value. Nothing else changes.

### Error model

- No new error types.
- `MemberError` is unchanged: an identifier is always a valid member.

### Non-goals

- Identifier members as references in alpha equivalence (SS1).
- Free-identifier reporting of members.
- Any change to `Param`'s or `ConstraintSystem`'s alpha equivalence.

### Test plan

- **Rust integration** (`tests/it/constraint/identifier_value_stories.rs`,
  through the public API, as the other constraint tests are):
  - the order of identifiers among kinds; `Display`; `kind_name`;
    `lifts_to_expression`;
  - type-strict equality (identifier vs string, vs another id);
  - hash agreeing with `==`;
  - `MemberSet` order by id and kind;
  - `contains_value`;
  - an identifier binding refused by an equation;
  - an in-set constraint over identifiers decided under bindings;
  - `to_expression` refusing an identifier member;
  - JSON and postcard round trips of values, members, set constraints and
    categorical and permutation params;
  - the legacy opaque form read through a test resolver whose part reports
    an identifier;
  - categorical and permutation domains over identifiers;
  - an ordinal domain's comparison answering `None`.
- **Properties** (proptest, `constraint_properties.rs`, whose value
  strategy draws identifiers, and `param/serde_stories.rs`'s domain round
  trip): serde round trips of identifier-bearing values and members; the
  legacy opaque rewrite of any value; `Member` order is total and agrees
  with `==` and `hash`; `MemberSet::new` is independent of input order;
  `Value` equality is reflexive and symmetric on identifier-bearing values;
  an identifier never equals a string or an integer.
- **Python** (`tests/symbolic/constraint/test_identifier_members.py`):
  - the V2 shapes of `serialize_value`, a set constraint and a categorical
    param pinned;
  - the legacy opaque form decoding to an equal `Identifier`;
  - V1 round trip;
  - `serialization_upgrade` writing the new form;
  - type-strict admission (an `Identifier` vs its name);
  - category order by id;
  - an ordinal param still refusing identifiers;
  - pickling;
  - a user `Identifier` subclass read as an identifier.
- **Golden row** (`generate_serialization_cases.py`): a set constraint and
  a categorical param over identifiers, and an identifier value, replayed
  by `serialization_golden.rs`.

### Open questions

None. The breaking change to `MemberKind` is the price of the decided
first-class kind; the commit is marked `!`.

## SS1: the Rust core (plan)

The concrete plan for SS1.1 to SS1.4, in fhy-development-rs's planning
template. The stub (`rust/fhy-core/src/search_space.rs` and
`search_space/`) follows it; where it differs from the "Rust sketch"
below, this plan holds. SS1.5 onward (the binding and Python) is planned
with its own slice.

### Summary

`fhy_core::search_space`: the `Variable` and `Alternative` traits with
their plain implementations, `Choice`, `Space` with its conditions and
forbidden clauses, `Configuration` with its validation, activity,
completeness and `ConfigurationKey`, the equivalence walk, and the wire
forms. Every name in a space is a binder, and every identifier member is a
reference resolved through the renaming.

### Module tree and visibility

| Path | Visibility | Contents |
|---|---|---|
| `search_space` | `pub mod` | module docs (the activity rules, equivalence, the implementor contract), explicit `pub use` |
| `search_space::variable` | private | `Variable`, `PlainVariable`; the inherent and `AlphaEquivalence` relations of `Part<dyn Variable>` |
| `search_space::alternative` | private | `Alternative`, `PlainAlternative`; the relations of `Part<dyn Alternative>` |
| `search_space::choice` | private, leaf | `Choice`; caches each choice's names in canonical order |
| `search_space::space` | private, leaf | `Space`, `Condition`, `Forbidden`, `Decision`; the checks of `Space::new`, the canonical and decision orders; `pub(super)` read-only views of the decision graph for the walk and the configuration |
| `search_space::configuration` | private, leaf | `Configuration`, `ConfigurationKey`, `Activity`; the three validation passes; `pub(super) fn restore` for `wire` |
| `search_space::equivalence` | private | the frame, member, domain and system correspondence, the structural and alpha walks (`pub(super)` functions only) |
| `search_space::error` | private | `SpaceError`, `ConfigurationError`, `ConfigurationErrors` (its `new` is `pub(super)`), `EquivalenceError` |
| `search_space::wire` | `pub mod` | `SearchSpaceResolver`, `VariableData`, `AlternativeData`, `ChoiceData`, `SpaceData`, `ConfigurationData`; the serde impls |
| `search_space::testing` | `pub mod`, `testing` feature | `check_variable_conformance`, `check_alternative_conformance`, `ConformanceViolation`, `ContractClause` |

Each public item has one path: `search_space::{Variable, PlainVariable,
Alternative, PlainAlternative, Choice, Space, Condition, Forbidden,
Decision, Configuration, ConfigurationKey, Activity, SpaceError,
ConfigurationError, ConfigurationErrors, EquivalenceError}`, and the
`wire` and `testing` items in their modules. No field is public. The
`testing` feature is new and adds nothing to the default build; it is
test-only API, as `fhy-core-py`'s `testing` feature is.

Layer 11 in CONTRIBUTING and in the crate docs: `search_space` depends on
`param`, `constraint`, `solver` (for the ground simplifier of a decoded
configuration), `expression`, `term`, `diagnostic`, `foreign` and
`identifier`, and never on `pass`, `types`, `symbol_table`, `stack` or
`scope`.

### Public API

```rust
pub trait Variable: ForeignPart {
    fn kind(&self) -> Cow<'_, str>;
    fn name(&self) -> &Identifier;
    fn param(&self) -> &Param;
    fn notes(&self) -> &[Note] { &[] }
    fn is_extension_structurally_equivalent(&self, other: &dyn Variable) -> Result<bool, BoxError> { Ok(true) }
    fn is_extension_alpha_equivalent_under(&self, other: &dyn Variable, renaming: &AlphaRenaming) -> Result<bool, BoxError> { Ok(true) }
    fn eq_part(&self, other: &dyn Variable) -> bool { is_same_part(self, other) }
    fn hash_part(&self, state: &mut dyn Hasher) {}
}
impl Part<dyn Variable> { pub fn is_structurally_equivalent(&self, other: &Self) -> Result<bool, EquivalenceError>; }
impl AlphaEquivalence for Part<dyn Variable> { type Error = EquivalenceError; }   // binds the names standalone
pub struct PlainVariable { /* name, param, notes */ }                          // Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize
impl PlainVariable {
    pub const KIND: &'static str = "search_space.variable";
    pub fn new(name: Identifier, param: Param) -> Self;
    pub fn with_notes(self, notes: Vec<Note>) -> Self;
}

pub trait Alternative: ForeignPart {
    fn kind(&self) -> Cow<'_, str>;
    fn name(&self) -> &Identifier;
    fn variables(&self) -> &[Part<dyn Variable>];
    fn choices(&self) -> &[Choice] { &[] }
    fn notes(&self) -> &[Note] { &[] }
    fn bound_identifiers(&self) -> Result<Vec<Identifier>, BoxError> { Ok(Vec::new()) }
    fn is_extension_structurally_equivalent(&self, other: &dyn Alternative) -> Result<bool, BoxError> { Ok(true) }
    fn is_extension_alpha_equivalent_under(&self, other: &dyn Alternative, renaming: &AlphaRenaming) -> Result<bool, BoxError> { Ok(true) }
    fn eq_part(&self, other: &dyn Alternative) -> bool { is_same_part(self, other) }
    fn hash_part(&self, state: &mut dyn Hasher) {}
}
impl Part<dyn Alternative> { pub fn is_structurally_equivalent(&self, other: &Self) -> Result<bool, EquivalenceError>; }
impl AlphaEquivalence for Part<dyn Alternative> { type Error = EquivalenceError; }
pub struct PlainAlternative { /* name, variables, choices, notes */ }        // Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize
impl PlainAlternative {
    pub const KIND: &'static str = "search_space.alternative";
    pub fn new(name: Identifier, variables: Vec<Part<dyn Variable>>, choices: Vec<Choice>) -> Result<Self, SpaceError>;
    pub fn with_notes(self, notes: Vec<Note>) -> Self;
}

pub struct Choice(Arc<..>);                                                  // Debug, Clone, PartialEq, Eq, Hash, AlphaEquivalence, serde
impl Choice {
    pub fn new(name: Identifier, alternatives: Vec<Part<dyn Alternative>>) -> Result<Self, SpaceError>;
    pub fn with_notes(self, notes: Vec<Note>) -> Self;
    pub fn name(&self) -> &Identifier;
    pub fn alternatives(&self) -> &[Part<dyn Alternative>];
    pub fn notes(&self) -> &[Note];
    pub fn is_structurally_equivalent(&self, other: &Self) -> Result<bool, EquivalenceError>;
}

pub struct Condition { /* target, when */ }                                   // Debug, Clone, PartialEq, Eq, Hash
impl Condition { pub fn new(target: Identifier, when: ConstraintSystem) -> Self; pub fn target(&self) -> &Identifier; pub fn when(&self) -> &ConstraintSystem; }
pub struct Forbidden { /* when */ }                                           // Debug, Clone, PartialEq, Eq, Hash
impl Forbidden { pub fn new(when: ConstraintSystem) -> Self; pub fn when(&self) -> &ConstraintSystem; }
pub enum Decision<'a> { Variable(&'a Part<dyn Variable>), Choice(&'a Choice) }  // Debug, Clone, Copy; exhaustive
impl<'a> Decision<'a> { pub fn name(&self) -> &'a Identifier; }

pub struct Space(Arc<..>);                                                   // Debug, Clone, PartialEq, Eq, Hash, AlphaEquivalence, serde
impl Space {
    pub fn new(name: Identifier, variables: Vec<Part<dyn Variable>>, choices: Vec<Choice>,
               conditions: Vec<Condition>, forbidden: Vec<Forbidden>) -> Result<Self, SpaceError>;
    pub fn with_notes(self, notes: Vec<Note>) -> Self;
    pub fn name(&self) -> &Identifier;
    pub fn variables(&self) -> &[Part<dyn Variable>];
    pub fn choices(&self) -> &[Choice];
    pub fn conditions(&self) -> &[Condition];            // one per target, canonical order of targets
    pub fn forbidden(&self) -> &[Forbidden];             // as given
    pub fn notes(&self) -> &[Note];
    pub fn decisions(&self) -> impl ExactSizeIterator<Item = Decision<'_>> + '_;   // canonical order
    pub fn decision(&self, name: &Identifier) -> Option<Decision<'_>>;
    pub fn decision_order(&self) -> &[Identifier];      // topological, ties in canonical order
    pub fn is_structurally_equivalent(&self, other: &Self) -> Result<bool, EquivalenceError>;
}

pub enum Activity { Active, Inactive, Pending }                              // Debug, Clone, Copy, PartialEq, Eq, Hash; exhaustive
pub struct Configuration(Arc<..>);                                           // Debug, Clone, PartialEq, Eq, Hash, AlphaEquivalence, serde
impl Configuration {
    pub fn new(space: &Space, entries: impl IntoIterator<Item = (Identifier, Value)>, context: &ParamContext<'_>) -> Result<Self, ConfigurationErrors>;
    pub(super) fn restore(..) -> Result<Self, ConfigurationErrors>;           // ParamAssignment::restore for values
    pub fn with_entry(&self, name: Identifier, value: Value, context: &ParamContext<'_>) -> Result<Self, ConfigurationErrors>;
    pub fn with_entries(&self, entries: impl IntoIterator<Item = (Identifier, Value)>, context: &ParamContext<'_>) -> Result<Self, ConfigurationErrors>;
    pub fn space(&self) -> &Space;
    pub fn value(&self, name: &Identifier) -> Option<&Value>;
    pub fn alternative(&self, choice: &Identifier) -> Option<&Part<dyn Alternative>>;
    pub fn entries(&self) -> impl ExactSizeIterator<Item = (&Identifier, &Value)> + '_;   // canonical order
    pub fn activity(&self, name: &Identifier) -> Option<Activity>;
    pub fn is_complete(&self) -> bool;
    pub fn key(&self) -> ConfigurationKey;
    pub fn is_structurally_equivalent(&self, other: &Self) -> Result<bool, EquivalenceError>;
}
pub struct ConfigurationKey(Arc<[KeyEntry]>);                                // Debug, Clone, PartialEq, Eq, Hash

#[non_exhaustive] pub enum SpaceError {
    DuplicateName { name }, EmptyChoice { choice }, UnknownConditionTarget { target },
    UnknownReference { name }, EquationOverChoice { choice }, ConditionReferencesSubtree { target, name },
    EmptyForbidden { index }, CyclicDependency { cycle: Vec<Identifier> },
    Hook { alternative: Identifier, source: BoxError }, Constraint(ConstraintError),
}
#[non_exhaustive] pub enum ConfigurationError {
    UnknownDecision { name }, DuplicateEntry { name }, InactiveDecision { name },
    UnknownAlternative { choice, value: Value }, Assignment { variable, error: AssignmentError },
    Forbidden { index }, UndecidedCondition { target }, UndecidedForbidden { index },
    FailedCondition { target, error: ConstraintError }, FailedForbidden { index, error: ConstraintError },
}
pub struct ConfigurationErrors(Vec<ConfigurationError>);                     // pub fn errors(&self) -> &[ConfigurationError]
#[non_exhaustive] pub enum EquivalenceError { Constraint(ConstraintError), Extension(BoxError) }

pub mod wire {
    pub trait SearchSpaceResolver: ParamResolver + Resolve<Part<dyn Variable>> + Resolve<Part<dyn Alternative>> {}  // blanket impl
    pub struct VariableData;    // of(&Part<dyn Variable>), foreign(), build(resolver, context) -> Part<dyn Variable>
    pub struct AlternativeData; // of, foreign, build -> Part<dyn Alternative>
    pub struct ChoiceData;      // of, build -> Choice (Choice::new)
    pub struct SpaceData;       // of, build -> Space (Space::new)
    pub struct ConfigurationData; // of, build -> Configuration (restore semantics)
}

#[cfg(feature = "testing")] pub mod testing {
    pub fn check_variable_conformance<R: Resolve<Part<dyn Variable>> + ?Sized>(samples: &[Part<dyn Variable>], resolver: &R) -> Result<(), ConformanceViolation>;
    pub fn check_alternative_conformance<R: Resolve<Part<dyn Alternative>> + ?Sized>(samples: &[Part<dyn Alternative>], resolver: &R) -> Result<(), ConformanceViolation>;
    pub struct ConformanceViolation; // clause() -> ContractClause, kind() -> &str; Display, Error
    #[non_exhaustive] pub enum ContractClause { StableGetters, UniqueKind, DistinctBoundIdentifiers, EquivalenceHooks, WireForm }
}
```

The `Display` texts are decided in the stub (`search_space/error.rs`), one
lowercase line each, naming identifiers as `name::id`.

### Ownership and data model

- `Choice`, `Space` and `Configuration` are an `Arc` of their data:
  clones share, and a configuration holds its `Space` by value (a
  reference count). `PlainVariable` and `PlainAlternative` are plain
  structs; a container holds them as `Part`s.
- `Choice::new` reads each alternative's `bound_identifiers` once and keeps
  the choice's names in canonical order, so `Space::new`, the walk and the
  key never call the hook again. The space keeps its names (its own name
  first), its decisions in canonical order with each one's parent (choice
  and alternative position), a name index, one merged condition per
  target with the decisions it names, each forbidden clause's decisions,
  and the decision order.
- A configuration keeps one optional value and one `Activity` per
  decision, in canonical order; `activity`, `value` and `is_complete` are
  lookups.
- Every type is `Send + Sync`: parts are, through `ForeignPart`.

### Extensibility decisions

- `Variable` and `Alternative` are open, object-safe traits held as
  `Part<dyn _>` (decided). Not sealed: downstream crates implement them.
  New methods get defaults.
- The relations are inherent on `Part<dyn Variable>` and
  `Part<dyn Alternative>` (`is_structurally_equivalent`) and
  `AlphaEquivalence` impls, not trait methods, so no implementation
  overrides the base comparison (C-8). The design's sketch put them on
  `dyn Variable`; on the `Part` they read like `Choice`'s and `Space`'s.
- `Decision` and `Activity` are exhaustive (callers match every case);
  the error enums and `ContractClause` are `#[non_exhaustive]`.

### Error model

- Construction of a `Choice`, `PlainAlternative` or `Space` stops at the
  first problem, in the documented order (`SpaceError`).
- `Configuration::new` collects every problem (`ConfigurationErrors`),
  as fhy_core's validators do, in three passes: the entries in the order
  given (`UnknownDecision`, `DuplicateEntry`); the decisions in decision
  order (`UndecidedCondition`, `FailedCondition`, `InactiveDecision`,
  `UnknownAlternative`, `Assignment`); the forbidden clauses in order
  (`Forbidden`, `UndecidedForbidden`, `FailedForbidden`). A refused entry
  counts as unassigned for every later check, so a condition or clause
  naming a variable whose value is refused is not evaluated (its target is
  pending, the clause does not apply yet), and only the value's problem is
  reported: a failing simplifier that fails both a value's check and a
  condition over that value reports the `Assignment` alone.
- Comparisons return `EquivalenceError` for a failing hook or custom
  constraint. Two spaces of different shapes, or a standalone alternative
  whose bound identifiers repeat, answer `false`, not an error.
- No panics on input. `expect` only for the documented invariant that a
  space's names are distinct, which makes pairing two spaces of one shape
  infallible.

### Behavior

| Question | Answer |
|---|---|
| a space's names | its own name, then each decision's, alternative's and bound identifier's in canonical order (an alternative's: its name, its bound identifiers, its variables, its sub-choices); all distinct, else `DuplicateName` naming the first repeat |
| canonical order of decisions | top-level variables, then top-level choices, each choice followed by its alternatives' variables then sub-choices, depth first |
| a condition's references | its free identifiers, checked in id order: each a decision (`UnknownReference`), not a choice inside an equation (`EquationOverChoice`), not the target or under it (`ConditionReferencesSubtree`) |
| several conditions on one target | conjoined into one condition (one `ConstraintSystem` holding all their members) |
| a forbidden clause with no reference | refused (`EmptyForbidden`): it would forbid every configuration |
| dependency | a decision depends on its choice and on every decision its condition names; a cycle is `CyclicDependency`, listed from its decision first in canonical order |
| decision order | Kahn's order taking, at each step, the ready decision first in canonical order |
| activity | as `Configuration`'s docs state: inactive wins over pending, pending over active |
| a condition naming a choice | `ch in {a}` reads the choice's value, the identifier value of the chosen alternative's name |
| an undecided or failing condition | reported (N-C2), and its target counts as pending, so an entry for it is also `InactiveDecision` |
| completeness | every decision is assigned or inactive |
| `with_entry`, `with_entries` | replace or add entries, then check the whole configuration as `new` does; no entry is dropped implicitly |
| key | per decision: inactive, unassigned, the chosen alternative's position, or the value with each identifier the space binds written as its position among the space's names; an identifier the space does not bind stays itself |
| structural equivalence | the walk with names by `==`, params by `Param::is_structurally_equivalent`, systems by `ConstraintSystem::is_structurally_equivalent`, hooks for the implementation's data |
| alpha equivalence | equal shapes; one frame pairing the two spaces' names in canonical order; under it each part, the conditions by target position and the forbidden clauses in order; notes by `==` |
| identifier members (D-SS-3) | resolved by the walk itself through `is_corresponding`: a categorical domain's and a set constraint's members as a bijection, a permutation's and an ordinal's in order, inside tuples and frozen sets too; every other member type-strictly by `==`. A free identifier never matches a bound one |
| constraint systems in the walk | members paired up in any order (greedy matching, sound because correspondence under one renaming is an equivalence), since canonical order follows ids; equations and custom constraints through `Constraint`'s alpha equivalence, set constraints by polarity, corresponding variables and members |
| params in the walk | domains as above, then the constraints under one more frame pairing the params' variables |
| standalone `Choice`, alternative, variable | binds its own names (same order) on top of the renaming given, then compares as nested |
| configuration equivalence | spaces as above; values under the spaces' frame: same alternative position for a choice, corresponding values for a variable |
| `==`, `Hash` | structural through `eq_part`/`hash_part`; a configuration's include its space |
| wire | as `wire`'s docs; decoding a configuration checks values with `restore` semantics; `Configuration`'s own `Deserialize` uses a solver holding the ground simplifier, so ground equations in conditions evaluate |

### Non-goals

- The binding, Python, `Trace`, `Measurement` (SS1.5 on, SS2, SS3).
- Enumeration, cardinality, sampling (SS2).
- Checking that a set constraint's members on a choice name its
  alternatives: such a member never matches, and is allowed.
- Bounding the nesting depth of a decoded space: serde recurses once per
  choice level, as the space that wrote it did. (Values keep their
  `MAX_VALUE_DEPTH`.) The binding (SS1.6) must refuse a payload nested
  deeper than Python's recursion limit before it reaches the core, as
  `fhy-core-py`'s `provenance.rs` does for provenance.

### Test plan (SS1.3)

In `rust/fhy-core/tests/it/search_space/`, through the public API, with
builders, strategies and the test implementors in
`tests/it/support/search_space.rs`:

- `variable_stories.rs`, `alternative_stories.rs`, `choice_stories.rs`:
  construction, kinds, `==`/`Hash`, standalone equivalence, identifier
  members as references, constraints in alpha mode, type-strict members;
- `space_stories.rs`: every `SpaceError` in its documented order, names
  across levels, merged conditions, canonical and decision orders,
  `decision`, `decisions`, `with_notes`;
- `activity_stories.rs`: each activity rule (hierarchy, conditions,
  pending, undecided and failing conditions);
- `forbidden_stories.rs`: inactive, pending, holding, violated, undecided
  and failing clauses;
- `configuration_stories.rs`: every `ConfigurationError`, collect-all and
  its order, completeness, `with_entry`/`with_entries`, accessors, the key;
- `implementor_stories.rs`: test implementors shaped like `ArrayTileKnob`
  and `RealizationOption`, in spaces, standalone and nested, and one
  breaking each checkable contract clause; the conformance checks run
  under the `testing` feature;
- `equivalence_stories.rs`: the audit's A1 to A7, C-1, C-2, C-4, both
  directions, configurations;
- `error_text_stories.rs`: every `Display` text;
- `serde_stories.rs`: JSON and postcard round trips, the pinned shapes,
  foreign parts through a resolver, refusals;
- `properties.rs` (proptest): reflexive, symmetric (the F-SS-002 sweep with
  label reuse), structural implies alpha, invariant under bijective
  relabeling, a perturbation breaks it, constraint-aware (F-SS-001), no
  capture (F-SS-003), no `1`/`true`/`1.0` conflation (F-SS-004), equal
  keys for corresponding configurations and `Eq`/`Hash` agreement, activity
  against a brute-force reference evaluator, serde round trips;
- `tests/it/search_space_golden.rs`: the oracle corpus, recorded by
  `tests/golden/record_search_space_cases.py` (how to regenerate it is in
  "Equivalence plan"); CI only replays it;
- non-vacuity guards in `properties.rs`: each property's strategy drawn by
  a fixed-seed runner over 256 cases, with a floor on how often its
  interesting branch is reached (models holding a variable, perturbations
  that apply, accepted repaired configurations and pairs of them, and
  structurally equivalent pairs).

### Changes to the design's sketch

- `Configuration::is_active(name) -> Activity` is
  `activity(name) -> Option<Activity>`: three answers are not a predicate,
  and a name the space lacks answers `None`.
- `ConfigurationError::Undecided { at }` is `UndecidedCondition { target }`
  and `UndecidedForbidden { index }`, since a clause has no name;
  `Constraint(ConstraintError)` is `FailedCondition` and `FailedForbidden`,
  which say where. `DuplicateEntry` is new: the constructor takes a list of
  entries, which can repeat a decision (F-SS-005, C-6).
- `SpaceError` gains `EquationOverChoice` (N-C5) and `EmptyForbidden`, and
  `Hook` names its alternative. `EquivalenceError::Param` is dropped: the
  walk compares params itself, so nothing returns a `ParamError`.
- The relations are on `Part<dyn _>`, not on `dyn _`.
- The key writes a bound identifier as its position among the space's
  names, not in the variable's domain (see "`ConfigurationKey`").
- `Configuration::with_entries` is added, which MOGA-VM's migration map
  uses; conditions on one target are merged into one.
- The golden corpus has a recorder, not a generator (see "Equivalence
  plan").

### Open questions

None. The lead's decision on conditions over choices and D-SS-3's
resolution are recorded under "Decisions".

## SS1.5 to SS1.7: the binding and the Python package (plan)

The concrete plan for the binding (SS1.5, SS1.6) and `fhy_core.search_space`
(SS1.7), in fhy-development-rs's planning template. The stub follows it.
It applies "The Python interface (N-4 (a): both paths)" and "Serialization
and type ids"; where it is more specific, this plan holds.

### Placement

| Where | Files |
|---|---|
| `fhy-core-py` | `search_space.rs` (module docs, `pub(crate) use`); `search_space/variable.rs` (`PyVariableBase`), `alternative.rs` (`PyAlternativeBase`), `choice.rs` (`PyChoice`), `space.rs` (`PySpace`, `PyCondition`, `PyForbidden`), `configuration.rs` (`PyConfiguration`, `PyConfigurationKey`), `adapter.rs` (`PythonVariable`, `PythonAlternative`), `kinds.rs` (the kind registry), `errors.rs` (the exception classes and the error conversions), `wire.rs` (the `Resolve` impls of `PyResolver`, the payload depth check); `convert/search_space.rs` (public); `lib.rs` (`register_part_4`) |
| Python | `src/fhy_core/search_space/__init__.py` (re-exports, `__all__`), `core.py` (the public classes, `Activity`), `errors.py` (the exceptions); `fhy_core/__init__.py` imports it; `_rs.pyi`; `fhy_core.traits.frozen` receives `_FrozenAfterInit` from `fhy_core.types.core`, which both packages then use |
| `rust/example-aggregate` | two downstream Rust kinds and their registration (see "Path 2") |
| tests | `tests/search_space/{__init__,conftest}.py`, `test_search_space.py`, `test_search_space_rust_binding.py`, `test_extension.py`, `test_search_space_properties.py`, `test_composed_search_space.py`; `_ENTRY_POINTS` of `tests/test_import_graph.py` |

The design's `classes.rs` is split by class, since the five classes
together would exceed 3000 lines, and its `PyVariable`/`PyAlternative`
adapter names become `PythonVariable`/`PythonAlternative`, since the
binding's `Py*` names are pyclasses.

### Visibility

- No pyclass is `pub`. Every pyclass and helper is `pub(crate)` or
  narrower; `search_space`'s submodules are private and the parent
  re-exports what `lib.rs`, `wire.rs` and `convert` need.
- The public surface is `convert::search_space` (below), the only new
  `pub` items of the crate.
- `term::renaming` gains a `pub(crate)` constructor of a Python
  `AlphaRenaming` from a core one (the alpha hook's argument), and
  `wire::PyResolver` gains two `Resolve` impls from `search_space::wire`.

### The Python classes

Every class is `module = "fhy_core._rs"`, frozen, and `==`/`hash` by
identity except `ConfigurationKey`. Each public class is a thin subclass of
its pyclass, registered with `_register_public_class()` at import, so an
object the binding builds is an instance of it. Each container keeps the
objects it was given and returns them (`choice.alternatives[0] is
alternative`); an object built from a core value (a decoded payload, a
merged condition) is built once and kept. A `name` left `None` is a fresh
`Identifier` named `variable`, `alternative`, `choice` or `space`.

Arguments are checked strictly, with `util::dataclass`'s message
`<Class> <field> must be <expected>, got <type>.` (`TypeError`): a name or
target an `Identifier`; a param a `Param`; notes `Note`s; variables
`Variable`s; alternatives `Alternative`s; choices `Choice`s; conditions
`Condition`s; forbidden clauses `Forbidden`s; `when` a `ConstraintSystem`
or an iterable of `Constraint`s; a space a `Space`; entries a `Mapping` or
an iterable of pairs whose names are `Identifier`s.

| Class (`_rs` name) | Construction | Attributes and methods | Errors |
|---|---|---|---|
| `Variable` (`PyVariableBase`, `subclass`) | `_rs.Variable.__new__(cls, *args, **kwargs)` takes anything; the public `Variable.__init__(self, param, name=None, notes=())` calls `_rs.Variable._initialize(self, param, name, notes)`, which sets the base fields once | `name`, `param`, `notes` (tuple), `kind` (`"search_space.variable"`, a subclass's registered type id); the equivalences; the serialization methods; the hooks (below) | `TypeError` for an argument; `RuntimeError` for `_initialize` called twice and for using a subclass instance whose `__init__` never called `Variable.__init__` |
| `Alternative` (`PyAlternativeBase`, `subclass`) | as `Variable`: `Alternative.__init__(self, variables=(), choices=(), name=None, notes=())` calls `_rs.Alternative._initialize` | `name`, `variables`, `choices`, `notes`, `kind`; the equivalences; the serialization methods; the hooks | as `Variable`; `DuplicateNameError` as `PlainAlternative::new` refuses its names, for a subclass too; `RecursionError` (depth) |
| `Choice` (`PyChoice`) | `Choice(alternatives, name=None, notes=())` | `name`, `alternatives`, `notes`; the equivalences; the serialization methods | `SearchSpaceError` (empty), `DuplicateNameError`, a hook's exception, `RecursionError` |
| `Condition` (`PyCondition`) | `Condition(target, when)` | `target`, `when` (a `ConstraintSystem`: the one given, or one built from the constraints given) | `TypeError` |
| `Forbidden` (`PyForbidden`) | `Forbidden(when)` | `when` | `TypeError` |
| `Space` (`PySpace`) | `Space(variables=(), choices=(), conditions=(), forbidden=(), name=None, notes=())` | `name`, `variables`, `choices`, `conditions` (one per target in canonical order of the targets, as the core keeps them: a target's one condition is the object given, a merged one is built over the constraint objects given), `forbidden`, `notes`, `decisions` (the decision objects in canonical order), `decision(name)` (or `None`), `decision_order` (the decisions' name objects); the equivalences; the serialization methods | `SearchSpaceError` for each `SpaceError` but `DuplicateName` (`DuplicateNameError`), `Hook` (the hook's exception itself) and `Constraint` (the constraint error, as the constraint module raises it); `RecursionError` |
| `Configuration` (`PyConfiguration`) | `Configuration(space, entries=())`: a choice's value is the chosen alternative's name, an `Identifier`; checked under the default solver's context (`with_param_context`) | `space`, `entries` (`(name, value)` pairs in canonical order, the names the decisions' own objects and the values the objects given), `value(name)`, `alternative(choice)` (the alternative object), `activity(name)` (an `Activity` or `None`), `is_complete()`, `key()`, `with_entry(name, value)`, `with_entries(entries)` (both keep the other entries' objects); the equivalences; the serialization methods | `ConfigurationError` carrying every problem; a Python-defined constraint's exception itself when one raised; `TypeError` |
| `ConfigurationKey` (`PyConfigurationKey`, no public subclass: `fhy_core.search_space.ConfigurationKey` is `_rs.ConfigurationKey`) | no constructor; `Configuration.key()` builds it | structural `==` and `hash` (another type is `NotImplemented`); `repr` `ConfigurationKey(...)`; pickles, under every protocol, as `(ConfigurationKey._from_wire, (<V2 text>,))` to an equal key of one hash | `_from_wire` raises `DeserializationValueError` for a text of another shape |
| `Activity` (Python `StrEnum`) | `ACTIVE = "active"`, `INACTIVE = "inactive"`, `PENDING = "pending"` | | |

**The equivalences** of `Variable`, `Alternative`, `Choice`, `Space` and
`Configuration`: `is_structurally_equivalent(other)`,
`is_alpha_equivalent(other)` and `is_alpha_equivalent_under(other,
renaming)`, over the core's relations. An `other` that is not of the same
class (`Variable` and its subclasses and kinds count as one) answers
`False`. `renaming` must be an `AlphaRenaming` (`TypeError`). A hook's
exception propagates as itself; a failing custom constraint raises as the
constraint module raises it.

**The serialization methods:** `serialize_to_dict()`, `to_json(*,
indent=None, sort_keys=None)`, the class methods `deserialize_from_dict(data)`
and `from_json(payload)`; `Variable` and `Alternative` add
`serialize_data_to_dict()` and the class method
`deserialize_data_from_dict(data)` (see "Wire").

**Representation:** `repr` lists the fields as a dataclass's does,
`<Class>(name=..., ...)`, with the class's own name for a subclass; a
configuration shows its space by name: `Configuration(space=s::1,
entries=(...))`.

**Pickling:** a call of the class with its fields
(`(cls, (fields...))`). A `Variable` or `Alternative` subclass pickles
through `copyreg.__newobj__` with the state `(base fields, __dict__)`,
which the public base's `__setstate__` restores, frozen.

**Freezing:** the pyclasses refuse attribute assignment and deletion
(`FrozenMutationError`). `Variable` and `Alternative` freeze a subclass
after its outermost `__init__`, through `_FrozenAfterInit`, as `Type` does,
and are virtual subclasses of `FrozenMixin`.

**Errors** (`fhy_core.search_space.errors`, re-exported):
`SearchSpaceError(ValueError)`; `DuplicateNameError(SearchSpaceError)`;
`ConfigurationError(SearchSpaceError)` with `problems: tuple[str, ...]`,
each problem's `Display` text in the core's order, and `str` the
`ConfigurationErrors` text. Every message is the core error's `Display`,
which names identifiers as their `repr`, `name::id`.

### Path 1: Python subclasses and their adapters

- **Hooks**, defined on the public `Variable` and `Alternative` with their
  defaults:
  - `extension_is_structurally_equivalent(self, other) -> bool` (`True`);
  - `extension_is_alpha_equivalent_under(self, other, renaming:
    AlphaRenaming) -> bool` (`True`);
  - `Alternative.extension_bound_identifiers(self) -> Iterable[Identifier]`
    (`()`).
- **The adapters** `PythonVariable` and `PythonAlternative` implement the
  core traits for a subclass instance. A container builds one when it
  reads such an object, inside its `collect_slots`, and owns the slot
  holding the object.
  - An adapter copies the base fields from the pyclass, with no Python
    call, and reads the kind once: the class's
    `get_serialization_class_type_id()`.
  - Each hook calls the method of the same name once per call of the core
    (once per pair of extended nodes); `other` is the other side's object
    and `renaming` a new `AlphaRenaming` of the core's.
  - A result of the wrong type raises `TypeError`: `<Class>.<hook> must
    return a bool, got <type>.`, and `<Class>.extension_bound_identifiers
    must return Identifiers, got <type>.`.
  - An exception a hook raises is boxed as the hook's error and raised by
    the entry point as the same object; `KeyboardInterrupt` passes
    through the same way.
  - `eq_part` and `hash_part` are the object's identity.
  - `to_foreign` is `util::foreign::read_foreign(object, family=True)`:
    the class's type id and its `serialize_data_to_dict()` text.

### Path 2: downstream Rust kinds and the kind registry

`convert::search_space` (public):

```rust
pub type VariableFromPython = fn(&Bound<'_, PyAny>) -> PyResult<Part<dyn Variable>>;
pub type VariableToPython = for<'py> fn(Python<'py>, &Part<dyn Variable>) -> PyResult<Bound<'py, PyAny>>;
pub type VariableResolver = fn(&Foreign) -> Result<Part<dyn Variable>, ForeignError>;
pub type AlternativeFromPython = fn(&Bound<'_, PyAny>) -> PyResult<Part<dyn Alternative>>;
pub type AlternativeToPython = for<'py> fn(Python<'py>, &Part<dyn Alternative>) -> PyResult<Bound<'py, PyAny>>;
pub type AlternativeResolver = fn(&Foreign) -> Result<Part<dyn Alternative>, ForeignError>;

pub fn variable_from_python(object: &Bound<'_, PyAny>) -> PyResult<Part<dyn Variable>>;
pub fn variable_to_python<'py>(py: Python<'py>, variable: &Part<dyn Variable>) -> PyResult<Bound<'py, PyAny>>;
pub fn alternative_from_python(object: &Bound<'_, PyAny>) -> PyResult<Part<dyn Alternative>>;
pub fn alternative_to_python<'py>(py: Python<'py>, alternative: &Part<dyn Alternative>) -> PyResult<Bound<'py, PyAny>>;
pub fn choice_from_python(object: &Bound<'_, PyAny>) -> PyResult<Choice>;
pub fn choice_to_python<'py>(py: Python<'py>, choice: &Choice) -> PyResult<Bound<'py, PyAny>>;
pub fn register_variable_kind(module: &Bound<'_, PyModule>, kind: &str, class: &Bound<'_, PyType>,
    from_python: VariableFromPython, to_python: VariableToPython, resolve: VariableResolver) -> PyResult<()>;
pub fn register_alternative_kind(module: &Bound<'_, PyModule>, kind: &str, class: &Bound<'_, PyType>,
    from_python: AlternativeFromPython, to_python: AlternativeToPython, resolve: AlternativeResolver) -> PyResult<()>;
```

- **Reading** an object (`*_from_python`, and every container reading
  its arguments) tries, in order: the public plain class (exactly
  `Variable`, read to a `PlainVariable`); a registered kind, whose
  `class` the object is an instance of (its `from_python`); a Python
  subclass of `_rs.Variable` (the adapter). Anything else is `TypeError`.
  `choice_from_python` reads a `Choice` (its core value, shared).
- **Writing** a part (`*_to_python`, and every getter of a part built from
  a core value) is: a `PlainVariable` as a new public `Variable`; an
  adapter's own object; a registered kind's `to_python`; an unregistered
  kind is `TypeError` naming the kind.
- **Decoding** a foreign part, `PyResolver` tries a registered kind's
  `resolve` by the part's type id, then the Python registry
  (`_resolve_foreign`, which must give an instance of the family).
- **The registry** is append-only module state: the attribute
  `fhy_core._rs._search_space_kinds`, a private pyclass
  (`_SearchSpaceKindRegistry`) holding a `Mutex<Arc<KindRegistryState>>`
  of one map per family from kind to its class and three functions.
  `register` creates it beside the verification registry. A registration
  swaps in a new state; a lookup takes the current `Arc` and releases the
  lock before any Python call; the binding reaches it through a
  write-once import cache, as it does the verification registry. No Rust
  `static` holds kinds.
- **Registration refuses** (`ValueError`): a built-in kind
  (`search_space.variable`, `search_space.alternative`), a kind registered
  already, a class registered already; and (`RuntimeError`) a module
  without `fhy_core`'s binding. It never replaces an entry.
- **Virtual subclass:** the class is registered as a virtual subclass of
  the public `Variable` or `Alternative` once both exist: at registration,
  if the public class is registered, and otherwise by
  `_register_public_class()` when `fhy_core.search_space` is imported, since
  an aggregate registers its kinds while `fhy_core` itself is being
  imported.
- **Why a resolver:** the design listed `(module, kind, class,
  from_python, to_python)`. Decoding a Rust kind's foreign part without
  a call into Python needs the kind's own resolver, which its contract
  (clause 6) already requires it to have.
- **The example aggregate** gains `TiledVariable` (a `Variable` kind
  `example.tiled_variable` with index symbols, compared by its hooks as
  `ArrayTileKnob` is) and `AxisAlternative` (an `Alternative` kind
  `example.axis_alternative` binding its axes, as `RealizationOption`
  does), their `#[pyclass]`es in `module = "fhy_example_aggregate"`, and
  their registration from its `#[pymodule]`. It depends on `serde` and
  `serde_json` (workspace crates, no new crate in the lock file) for their
  foreign payloads.

### Wire

| Class | Type id | V2 dict (`serialize_to_dict`) |
|---|---|---|
| `Variable` | `search_space.variable` | the tagged part, `{"plain": {"identifier", "param", "notes"}}` or `{"foreign": {"type_id", "data"}}` (`VariableData`) |
| `Alternative` | `search_space.alternative` | `{"plain": {"identifier", "variables", "choices", "notes"}}` or `{"foreign": ..}` (`AlternativeData`) |
| `Choice` | `search_space.choice` | `{"identifier", "alternatives", "notes"}` |
| `Space` | `search_space.space` | `{"identifier", "variables", "choices", "conditions", "forbidden", "notes"}` |
| `Configuration` | `search_space.configuration` | `{"space", "entries"}` |
| `ConfigurationKey` | (none: pickled, not a `Serializable`) | the core's `{"entries": [..]}`, one per decision in canonical order (see `fhy_core::search_space::wire`) |

- The text is the core's serde text; `to_json()` is byte-identical to the
  core's `serde_json` text, and the dict is what `json.loads` makes of it.
- `serialize_data_to_dict()` of a `Variable` is the plain fields
  `{"identifier", "param", "notes"}` (an `Alternative`'s likewise), which
  a subclass extends with its own keys: it is the `data` of the subclass's
  foreign part. `deserialize_data_from_dict(data)` reads exactly those
  keys (`DeserializationValueError` for another) and calls
  `cls(param=, name=, notes=)`, so a subclass with data of its own
  overrides it.
- `deserialize_from_dict` and `from_json` build through the constructors
  with `PyResolver` under the default solver's context; they return an
  instance of the class they are called on, else `SerializationError`; a
  payload of another shape is `DeserializationValueError` (`Invalid V2
  payload for "<Class>": ..`), and an exception a part's hook raises
  propagates as itself.
- **Depth:** a container records its depth, the levels of choices it
  nests (a variable 0, an alternative its deepest sub-choice, a choice
  one more). Building one deeper than `sys.getrecursionlimit()` raises
  `RecursionError` (`maximum recursion depth exceeded: the <class> is N
  levels deep`), and so does decoding a payload whose choices nest deeper,
  measured on the payload before any of it reaches the core, as
  `provenance.rs` refuses a deep provenance.
- **The key's wire form.** A key is self-contained: every identifier its
  space binds is already written as its position among the space's names,
  and any other value as itself. So the core gains `Serialize` and
  `Deserialize` for `ConfigurationKey` and `wire::ConfigurationKeyData`
  (`of`, `build(resolver)` for opaque values), the binding's only change to
  the core's public API, and the key pickles through its V2 text.
- **No V1 form.** These classes are new, so no V1 payload of them exists:
  `serialize_to_dict` and `to_json` raise `SerializationError` inside
  `wire_version(WireVersion.V1)`, and the readers read V2 only. (The
  design's "V1 is read until 0.3.0" concerns classes that had one.)

### GC

Every class but `ConfigurationKey` has `__traverse__` visiting the objects
it keeps (its field objects, its built and kept objects, and the `Slots`
of the adapters its construction made). The registry visits its classes
under `try_lock`. A cycle through a Python subclass instance, such as an
alternative whose attribute holds the choice holding it, is collected.

### `_rs.pyi`

`Variable`, `Alternative` (each with `__init__(self, *_args, **_kwargs)`,
`_initialize`, the getters, the equivalences, the serialization methods,
`_register_public_class`), `Choice`, `Condition`, `Forbidden`, `Space`,
`Configuration` (with their constructors, getters and methods) and
`ConfigurationKey`.

### `fhy_core.search_space`

```python
__all__ = ["Activity", "Alternative", "Choice", "Condition", "Configuration",
           "ConfigurationError", "ConfigurationKey", "DuplicateNameError",
           "Forbidden", "SearchSpaceError", "Space", "Variable"]
```

`Variable(_rs.Variable, _FrozenAfterInit, WrappedFamilySerializable,
Generic[_T])`, `Alternative(_rs.Alternative, _FrozenAfterInit,
WrappedFamilySerializable)`, and `Choice`, `Space`, `Configuration` as
`@final` subclasses of their pyclasses and `Serializable`, each
`@register_serializable(type_id=...)` with its id; `Condition` and
`Forbidden` are `@final` thin subclasses. The package imports
`fhy_core.symbolic` (params, constraints, the default solver) and
`fhy_core.diagnostic`, and never `fhy_core.types`, `symbol_table` or
`pass_infrastructure` (layer 11).

### Test plan (SS1.5)

In `tests/search_space/`, through the public API:

- `test_search_space.py`, the interface suite: construction and every
  argument check; every error class and message; activity, completeness,
  values and alternatives; `with_entry`/`with_entries`; the key; the
  equivalences; and the ported MOGA-VM tests (the Python half of
  "Traceability" below), each docstring citing its MOGA-VM test.
- `test_search_space_rust_binding.py`: the class structure, frozen-ness,
  identity `==`/`hash`, `ConfigurationKey`'s structural `==`/`hash`, kept
  objects, `repr`, pickling, the V2 shapes and type ids and round trips,
  the refusals of the readers, depth refusal, GC.
- `test_extension.py`: subclasses modelled on `ArrayTileKnob`,
  `PortBoundKnob`, a marker knob and `RealizationOption`, through their
  hooks, nested and standalone, round-tripping under their own type ids;
  each hook's exception, `KeyboardInterrupt` and wrong result type; the
  call counts.
- `test_composed_search_space.py` (slow, subprocess): the example
  aggregate's two Rust kinds read, compared, nested, written and decoded,
  their virtual subclassing, and the registry's refusals.
- `test_search_space_properties.py` (Hypothesis): round trips of random
  spaces and configurations; reflexive and symmetric relations;
  structural implies alpha; a relabeled copy is alpha-equivalent and its
  configurations' keys are equal.

## Semantics

### Structure and names

- A `Space` holds, in declared order, top-level variables, top-level
  choices, conditions and forbidden clauses.
- A `Choice` holds, in declared order, one or more alternatives.
- An `Alternative` holds, in declared order, its variables and its
  sub-choices, and may bind further identifiers (`bound_identifiers`, such
  as walk axes).
- **Decisions** are the variables and choices, at every depth. The
  **canonical order** is the pre-order walk: a space's top-level decisions in
  declared order, and, under each choice, each alternative's variables then
  its sub-choices.
- **Names are unique space-wide:** every decision name, every alternative
  name and every bound identifier is distinct (N-C1). A flat namespace lets
  conditions, forbidden clauses and configurations name any decision.

### Activity (conditions and hierarchy)

A decision `d` is **active** in a configuration `c` when:

1. its structural parent is active: `d` is top-level, or `d` sits under
   alternative `a` of choice `ch`, `ch` is active, and `c` chooses `a`; and
2. every `Condition` whose target is `d` holds.

A `Condition` is `{ target: Identifier, when: ConstraintSystem }` (one
system per target; conditions on one target conjoin).

- Its free identifiers must name decisions outside `d`'s subtree.
- The dependency graph (parent edges plus condition edges) must be acyclic;
  `Space::new` checks this and keeps a topological **decision order**, which
  SS2's traces and samplers follow.
- **Evaluation:**
  - If a referenced decision is inactive, the condition is false: a child of
    an inactive parent is inactive, as in ConfigSpace.
  - If a referenced decision is active but unassigned, the target's activity
    is *pending*, and a pending target must be unassigned.
  - Otherwise `ConstraintSystem::evaluate` runs with bindings mapping each
    referenced decision's name to its value; a choice's value is the
    identifier of its chosen alternative. `Satisfied` means true, `Violated`
    means false, and `Undecided` is an error (N-C2).

### Forbidden clauses

A `Forbidden` is `{ when: ConstraintSystem }`: a configuration violates it
when every decision it references is active and assigned, and the system
evaluates to `Satisfied`.

- A clause that references an inactive decision does not apply.
- A clause with an unassigned reference is pending; a partial configuration
  may still complete either way.
- `Undecided` is an error (N-C2).
- A positive cross-decision constraint `p` is written `Forbidden(not p)`.

### Configuration validity and completeness

`Configuration` holds `(decision name → value)` entries; a choice's value is
an alternative name.

`space.check(&configuration, &ParamContext)` returns the configuration's
status, or every problem found (collect-all, as validators in fhy_core do).

| Problem | Refused when |
|---|---|
| `UnknownDecision` | an entry names no decision of the space |
| `DuplicateEntry` | an entry names a decision an earlier entry named (F-SS-005, C-6) |
| `InactiveDecision` | an entry assigns a decision that is inactive or pending (replaces "UNSELECTED with assignments") |
| `UnknownAlternative` | a choice's value names no alternative of that choice |
| `Assignment` (`Inadmissible`, `ViolatedConstraint`, `UnverifiedConstraint`) | a variable's value fails `ParamAssignment::new` (`restore` semantics when built from a payload) |
| `Forbidden { index }` | a forbidden clause applies and holds |
| `UndecidedCondition`, `UndecidedForbidden` | a condition or clause evaluates `Undecided` |
| `FailedCondition`, `FailedForbidden` | a condition or clause fails to evaluate |

- A **valid** configuration has none of these problems.
- A **complete** configuration is valid and assigns every active decision.
- `Configuration` values are validated on construction against a space
  (`Configuration::new(&space, entries, ctx) -> Result<_, ConfigurationErrors>`)
  and keep an `Arc` of their space. So "valid" holds by construction, and
  `is_complete()` is a query.
- This replaces `SelectionStatus` (N-9 translated): MOGA-VM's "SELECTED with
  knobs left unassigned" is a valid, incomplete configuration.

### `ConfigurationKey`

The key is one entry per decision in canonical order:

- `Inactive`;
- `Unassigned` (active or pending, no value);
- `Alternative(index)`, the chosen alternative's position in its choice;
- `Value(key value)`: the value, with every identifier the space binds
  replaced by its position among the space's names. (A position in the
  variable's finite domain would not do: a categorical domain keeps its
  members in id order, which a renaming changes.) An identifier the space
  does not bind stays itself.

Properties:

- It mentions no identifier the space binds, so it is invariant under
  renaming them. Two configurations of alpha-equivalent spaces whose
  values correspond under the spaces' pairing have equal keys (a property
  test).
- It is `Eq + Hash`, and the Python object is hashable with structural `==`
  (N-7 translated).
- A key is meaningful within its space. Caches and measurements are keyed by
  `(space, key)`, the space compared by alpha equivalence or held by the
  caller.

### Equivalence and binder roles

| Label | Role | Bound by | Scope |
|---|---|---|---|
| decision names (variables, choices), at every depth | binders, paired in canonical order | the `Space` | the whole space: structure, conditions, forbidden clauses, and configurations and traces over it |
| alternative names | binders, paired by position within their choice | the `Space` (in canonical order with the decisions) | as above; they are a choice's values |
| `bound_identifiers` | binders, paired by position | the `Space`, at the alternative's place in canonical order | the alternative's data, and everything beneath it |
| a `Choice`'s, `Alternative`'s or `Variable`'s own labels | the same binders, re-entered | the node | standalone comparisons only |
| a param's variable | binder | the `Param` | its constraints |
| identifier members of a domain, identifier values | references | | the space's frame and outward |
| identifiers in conditions and forbidden clauses | references | | the space's frame |
| the space's own name | binder | the `Space` | the space |
| notes, kinds, directions, statuses, non-identifier values | values, type-strict | | |

Alpha equivalence of two spaces `l` and `r` under `R`:

1. Equal shapes: the same numbers of top-level decisions and the same kinds
   in canonical order.
2. `R1 = R` plus one frame pairing, in canonical order, every decision name,
   alternative name and bound identifier (`enter_binders`; unique by
   construction, so it cannot fail).
3. Under `R1`:
   - each variable pair: equal kinds; domains with identifier members
     compared by `is_corresponding` (a categorical set as a bijection);
     constraints under the param's variable frame; the implementor's hook;
     notes;
   - each alternative pair: equal kinds; the implementor's hook; notes;
   - the conditions (paired by target) and forbidden clauses (positionally),
     through `ConstraintSystem`'s alpha equivalence.

Configurations and traces compare through their keys and their spaces'
equivalence. **Structural equivalence** is the same walk with labels
compared by `==`, so structural implies alpha (a property test).

The audit's fixes carry over:

| Fix | How it holds in (C) |
|---|---|
| D-SS-1 | constraints compared under the variable frame |
| D-SS-2 | one frame from unique lists; standalone nodes re-enter the same pairs |
| D-SS-3 | the walk resolves identifier members, in domains and in set constraints, through `is_corresponding` (SS1 plan, "Behavior") |
| D-SS-4 | type-strict `Value`s |
| D-SS-5 | the checks of `Space::new` and `Configuration::new` |
| D-SS-6 | a choice needs one or more alternatives |
| D-SS-7 | plain implementations, concrete and registered |
| D-SS-9 | helpers deleted |
| D-SS-12 | implementor hooks compare only implementor data |

## Crate and module placement

**Core: `rust/fhy-core/src/search_space.rs`**, layer 11 ("`search_space`,
which depends on `param`, `constraint` and `expression`, and never on
`pass`, `types` or `symbol_table`"):

| File | Contents |
|---|---|
| `search_space.rs` | module docs, explicit `pub use` |
| `search_space/variable.rs` | trait `Variable`, `PlainVariable` |
| `search_space/alternative.rs` | trait `Alternative`, `PlainAlternative` |
| `search_space/choice.rs` | `Choice` |
| `search_space/space.rs` | `Space`, `Condition`, `Forbidden`, activity, canonical and decision orders (leaf: owns the invariants) |
| `search_space/configuration.rs` | `Configuration`, `ConfigurationKey`, validation (leaf) |
| `search_space/equivalence.rs` | private: frames, member matching, the walk |
| `search_space/error.rs` | `SpaceError`, `ConfigurationError(s)`, `EquivalenceError` |
| `search_space/wire.rs` | wire forms with `Foreign` parts; `build(resolver)` |
| `search_space/testing.rs` | behind the `testing` feature: the conformance checks of the implementor contract |
| SS2: `search_space/rng.rs`, `domain.rs`, `trace.rs`, `oracle.rs`, `recorder.rs`, `exploration.rs` | the generator, the stream, and sampling, replay, enumeration, counting and mutation over a `Space` (SS2's plan) |
| SS3: `search_space/measurement.rs` | `Objective`, `Direction`, `MeasurementStatus`, `Measurement`, `Measurer` |

**Binding:** `rust/fhy-core-py/src/search_space.rs` with
`search_space/{classes,adapter,kinds,errors,wire}.rs`, and
`convert::search_space`.

**Python:** `src/fhy_core/search_space/__init__.py` (`fhy_core.search_space`),
imported by `fhy_core/__init__.py`. CONTRIBUTING gains the layer 11 row and
the mapping row.

## The traits

```rust
pub trait Variable: ForeignPart {
    /// The implementor's registered type id: unique per Rust type, stable.
    fn kind(&self) -> Cow<'_, str>;
    fn name(&self) -> &Identifier;
    fn param(&self) -> &Param;
    fn notes(&self) -> &[Note] { &[] }
    /// Compare the implementor's own data; `other` has the same kind.
    fn is_extension_structurally_equivalent(&self, other: &dyn Variable) -> Result<bool, BoxError> { Ok(true) }
    /// The same under `renaming`, which holds the space's frame.
    fn is_extension_alpha_equivalent_under(&self, other: &dyn Variable, renaming: &AlphaRenaming) -> Result<bool, BoxError> { Ok(true) }
    fn eq_part(&self, other: &dyn Variable) -> bool { is_same_part(self, other) }
    fn hash_part(&self, state: &mut dyn Hasher) { let _ = state; }
}

pub trait Alternative: ForeignPart {
    fn kind(&self) -> Cow<'_, str>;
    fn name(&self) -> &Identifier;
    fn variables(&self) -> &[Part<dyn Variable>];
    fn choices(&self) -> &[Choice] { &[] }          // sub-choices: hierarchy
    fn notes(&self) -> &[Note] { &[] }
    /// Identifiers this alternative introduces beyond its decisions, in order.
    fn bound_identifiers(&self) -> Result<Vec<Identifier>, BoxError> { Ok(Vec::new()) }
    fn is_extension_structurally_equivalent(&self, other: &dyn Alternative) -> Result<bool, BoxError> { Ok(true) }
    fn is_extension_alpha_equivalent_under(&self, other: &dyn Alternative, renaming: &AlphaRenaming) -> Result<bool, BoxError> { Ok(true) }
    fn eq_part(&self, other: &dyn Alternative) -> bool { is_same_part(self, other) }
    fn hash_part(&self, state: &mut dyn Hasher) { let _ = state; }
}
```

- **Object safety:** a choice mixes alternative types, and an alternative
  mixes variable types, so the containers hold `Part<dyn Variable>` and
  `Part<dyn Alternative>`. A generic `Choice<A: Alternative>` cannot mix
  kinds and cannot cross into Python; an associated type breaks object
  safety. Both are rejected.
- **Template methods:** the full relations are inherent methods on
  `dyn Variable` and `dyn Alternative`, and functions over `Space`. They
  cannot be overridden, so the core always compares names, params,
  sub-structure and kinds, and calls a hook only for implementor data
  (C-8).
- **Kind equality** replaces `type(self) is type(other)`: `l.kind() ==
  r.kind()` before any hook. A Rust hook downcasts
  `other.as_any().downcast_ref::<Self>()` and answers `false` if that
  fails. A Python subclass's kind is its registered type id.
- **The implementor contract** (rustdoc; checked by
  `search_space::testing::check_*_conformance` behind the `testing`
  feature):
  1. Getters are immutable for the value's life.
  2. `kind()` is unique per type, stable, and the type id the part
     serializes under.
  3. `bound_identifiers` are distinct, deterministic, and of equal length
     for values that should correspond (C-1, C-2).
  4. Hooks are equivalences; the structural hook implies the alpha hook
     under the empty renaming; they read the renaming only through
     `is_corresponding` (C-4); they compare type-strictly, never compare
     base fields (C-8), and fail with `Err`, never a panic.
  5. Labels an implementor introduces are declared in `bound_identifiers`;
     any other identifier in its data is a reference.
  6. `to_foreign` round-trips through the downstream resolver.
- **Plain implementations:** `PlainVariable` and `PlainAlternative`
  validate on construction. Their kinds are `search_space.variable` and
  `search_space.alternative`, their hooks keep the defaults, and their
  `eq_part`/`hash_part` are structural.

## Rust sketch

The direction the design started from. The SS1 plan above supersedes it
where they differ.

```rust
pub struct Choice(/* Arc: name, alternatives: Vec<Part<dyn Alternative>> (non-empty, unique names), notes */);
impl Choice { pub fn new(name: Identifier, alternatives: Vec<Part<dyn Alternative>>) -> Result<Self, SpaceError>; }

pub struct Condition { /* target: Identifier, when: ConstraintSystem */ }
pub struct Forbidden { /* when: ConstraintSystem */ }

pub struct Space(/* Arc: name, variables, choices, conditions, forbidden, notes; canonical and decision orders, name index */);
impl Space {
    pub fn new(name: Identifier, variables: Vec<Part<dyn Variable>>, choices: Vec<Choice>,
               conditions: Vec<Condition>, forbidden: Vec<Forbidden>) -> Result<Self, SpaceError>;
    pub fn decisions(&self) -> impl ExactSizeIterator<Item = Decision<'_>>;      // canonical order
    pub fn decision(&self, name: &Identifier) -> Option<Decision<'_>>;
    pub fn decision_order(&self) -> &[Identifier];                              // topological
    pub fn is_structurally_equivalent(&self, other: &Self) -> Result<bool, EquivalenceError>;
}
// impl AlphaEquivalence for Space, Choice, Part<dyn Variable>, Part<dyn Alternative>
pub enum Decision<'a> { Variable(&'a Part<dyn Variable>), Choice(&'a Choice) }

pub struct Configuration(/* Arc<Space>, entries in canonical order */);
impl Configuration {
    pub fn new(space: &Space, entries: impl IntoIterator<Item = (Identifier, Value)>, context: &ParamContext<'_>)
        -> Result<Self, ConfigurationErrors>;
    pub fn with_entry(&self, name: Identifier, value: Value, context: &ParamContext<'_>) -> Result<Self, ConfigurationErrors>;
    pub fn value(&self, name: &Identifier) -> Option<&Value>;
    pub fn alternative(&self, choice: &Identifier) -> Option<&Part<dyn Alternative>>;
    pub fn is_active(&self, name: &Identifier) -> Activity;   // Active | Inactive | Pending
    pub fn is_complete(&self) -> bool;
    pub fn key(&self) -> ConfigurationKey;
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)] pub struct ConfigurationKey(/* Vec<KeyEntry> */);

#[non_exhaustive] pub enum SpaceError {
    DuplicateName { name: Identifier }, EmptyChoice { choice: Identifier },
    UnknownConditionTarget { target: Identifier }, ConditionReferencesSubtree { target: Identifier, name: Identifier },
    CyclicDependency { cycle: Vec<Identifier> }, UnknownReference { name: Identifier }, Hook(BoxError), Constraint(ConstraintError),
}
pub struct ConfigurationErrors(/* Vec<ConfigurationError>, collect-all */);
#[non_exhaustive] pub enum ConfigurationError {
    UnknownDecision { name: Identifier }, InactiveDecision { name: Identifier }, UnknownAlternative { choice: Identifier, value: Value },
    Assignment { variable: Identifier, error: AssignmentError }, Forbidden { index: usize }, Undecided { at: Identifier }, Constraint(ConstraintError),
}
#[non_exhaustive] pub enum EquivalenceError { Param(ParamError), Constraint(ConstraintError), Extension(BoxError) }
pub mod wire { /* SpaceWire, ChoiceWire, parts as Plain | Foreign; build(&impl Resolve<..>) */ }
```

- Every type is `Send + Sync`, and clones are reference counts.
- `PartialEq`/`Hash` are structural for the containers and through
  `eq_part`/`hash_part` for parts.
- No `Default`, setters or `&mut` accessors.
- Error enums are `#[non_exhaustive]` with a one-line lowercase `Display`.

**SS2 and SS3 sketches** (for direction, settled in their slices):

```rust
pub struct TraceStep { /* decision: Identifier, kind: Canonical<DecisionKind>, shape: DomainShape, coordinate: Coordinate, value: Value */ }
pub struct Trace(/* Vec<TraceStep> */);
impl Space { pub fn replay(&self, trace: &Trace, context: &ParamContext<'_>) -> Result<Configuration, TraceError>; }
pub trait SearchOracle { fn decide(&mut self, step: &PendingStep<'_>) -> Result<Coordinate, BoxError>; }

pub enum Direction { Minimize, Maximize, Report }
pub struct Objective { /* name: Identifier, direction: Direction */ }
pub struct Measurement { /* key: ConfigurationKey, values: Vec<(Objective, Value)>, status: MeasurementStatus, notes */ }
```

## SS2: traces, oracles, enumeration and sampling (plan)

The concrete plan for SS2, in fhy-development-rs's planning template, as
SS1's plan is. It replaces this design's first sketch of the slice; where
it differs from the "SS2 and SS3 sketches" above, this plan holds. Its
choices were decided by the user on 2026-10-06 (N-S1 to N-S6 under
"Decisions"; the options weighed under "Needs the user (SS2/SS3)").

### Summary

SS2 gives a `Space` the second half of MOGA-VM's search vocabulary: the
decision stream.

- A **step** is one decision offered to an oracle with the set it may be
  answered from (its **domain**) and answered by a **coordinate**, a
  position in that domain. A **static** step asks a decision of a `Space`
  and takes its domain from the space. A **dynamic** step asks a decision
  the space does not declare (MOGA-VM's addresses and boundary
  namespaces) and carries its own domain.
- A `Trace` is the steps of one run, recorded in ask order; it is
  MOGA-VM's `SearchPoint` (`cir/lowering/search/decisions.py:503-580`).
- A `SearchOracle` answers steps. fhy-core ships a random, a replay and an
  exhaustive oracle; MOGA-VM's searches and Python oracles implement the
  same trait.
- A `Recorder` drives one run: it builds each step, asks the oracle,
  checks the answer and records it, and, over a space, grows the run's
  `Configuration`.
- Over a `Space` alone, fhy-core samples (per step, or uniformly over the
  complete configurations), replays a trace into a configuration,
  enumerates, counts and mutates.

### Motivation

- MOGA-VM's search does not read its static space at all. It re-derives
  the domains in `extraction.py:450-569`, numbers them with coordinates and
  replays by position (audit F-SS-013). SS1 gave the static space; SS2
  lays the coordinates over it, so a search can name its axes, count them
  and replay a point against a relabeled copy.
- The stream itself is generic: nothing in `decisions.py` or `oracle.py`
  imports MOGA, and the audit of it (F-SS-020 to F-SS-029) finds defects
  a generic core fixes once: a replay check that compares sizes only,
  Python equality in choice domains, and no persistent form of a point.

### What is generic and what stays in MOGA-VM

| Generic, in fhy-core | Stays in MOGA-VM |
|---|---|
| the domain shapes: choice, order, strided runs (`ChoiceDomain`, `OrderDomain`, `AddressDomain` generalized) | what each shape is built from: options, walk levels, fitting tiles, free address runs, boundary pools (`policies.py`, `placement.py`) |
| the open decision kind, a string; fhy-core's own kinds for static steps | MOGA's kinds (`moga.cir.address`, `moga.cir.boundary_namespace`) as constants |
| `TraceStep`, `Trace`, their wire form | `CandidateRecord`, `SearchHistory`, `SearchStatistics`, `JsonlTraceObserver` (which embeds a trace's wire form) |
| `SearchOracle`; the random, replay and exhaustive oracles | `SearchBasedLoweringStrategy`, `SearchBudget`, `RandomSearch`, `SampledPointLoweringStrategy`, the policies and the allocator |
| `Recorder`: driving, checking, recording, realizing a configuration | `build_oracle_driven_strategy` and the pipeline wiring |
| over a `Space`: sampling, replay, enumeration, cardinality, mutation | `StructuralSearchSpaceExtractor`, which now returns a `Space` |
| the random-number generator | seeds and budgets of MOGA's runs |

### Module tree and visibility

| Path | Visibility | Contents |
|---|---|---|
| `search_space::rng` | private, leaf | `Rng`: SplitMix64 (N-S1) |
| `search_space::domain` | private, leaf | `DecisionKind`, `Coordinate`, `ChoiceDomain`, `OrderDomain`, `StridedRun`, `StridedDomain`, `StepDomain`, `DomainSignature`; a static step's domain from a decision (`pub(super)`) |
| `search_space::trace` | private, leaf | `TraceStep`, `Trace` |
| `search_space::oracle` | private | `SearchOracle`, `PendingStep`, `RandomOracle`, `ReplayOracle`, `ExhaustiveOracle` |
| `search_space::recorder` | private, leaf | `Recorder`, `Recorded` |
| `search_space::exploration` | private | `impl Space { sample, sample_uniform, replay, enumerate, cardinality, mutate }`, `impl Configuration { trace }`, `Cardinality`, `Enumeration`; the relaxed counts (private) |
| `search_space::error` | private | adds `StepDomainError`, `EmptyKind`, `TraceError`, `ReplayError` |
| `search_space::wire` | `pub mod` | adds `TraceData` (the trace's shape; no resolver: a trace holds no foreign part) |
| `search_space::testing` | `pub mod`, `testing` feature | `ContractClause::SearchDomain`, checked by `check_variable_conformance` |

- New public paths: `search_space::{Rng, DecisionKind, Coordinate,
  ChoiceDomain, OrderDomain, StridedRun, StridedDomain, StepDomain,
  DomainSignature, TraceStep, Trace, SearchOracle, PendingStep,
  RandomOracle, ReplayOracle, ExhaustiveOracle, Recorder, Recorded,
  Cardinality, Enumeration, StepDomainError, EmptyKind, TraceError,
  ReplayError}` and `wire::TraceData`. No field is public.
- One existing trait gains a provided method: `Variable::search_domain`
  (below). Additive: it has a default.
- One crate-internal widening: `param::interval::effective_interval`
  (`rust/fhy-core/src/param/interval.rs:247`) goes from `pub(super)` to
  `pub(crate)`, so a bounded integer variable's domain is its interval.
- Layer 11 is unchanged: the new modules depend on `param`, `constraint`,
  `identifier`, `foreign` and `num-bigint`, and the `Space` and
  `Configuration` of SS1.

### Public API

```rust
// search_space::rng (SplitMix64, N-S1)
pub struct Rng { /* state */ }                         // Debug, Clone, PartialEq, Eq, Serialize, Deserialize
impl Rng {
    pub const ALGORITHM: &'static str = "splitmix64";
    pub fn new(seed: u64) -> Self;
    pub fn next_u64(&mut self) -> u64;
    pub fn below(&mut self, bound: NonZeroU64) -> u64;  // uniform in [0, bound), no modulo bias
    pub fn below_big(&mut self, bound: &BigUint) -> BigUint;   // panics on 0: documented bug
    pub fn shuffle<T>(&mut self, items: &mut [T]);      // Fisher-Yates, last index first
    pub fn split(&mut self) -> Self;                    // an independent stream
}

// search_space::domain
pub struct DecisionKind(Arc<str>);                     // Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Display, serde
impl DecisionKind {
    pub const CHOICE: &'static str = "search_space.choice";
    pub fn new(kind: &str) -> Result<Self, EmptyKind>;
    pub fn choice() -> Self;
    pub fn as_str(&self) -> &str;
}
pub enum Coordinate { Index(u64), Order(Box<[u32]>) }   // Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, serde; exhaustive
pub struct ChoiceDomain(Arc<[Value]>);                  // Debug, Clone, PartialEq, Eq, Hash
impl ChoiceDomain {
    pub fn new(values: Vec<Value>) -> Result<Self, StepDomainError>;
    pub fn values(&self) -> &[Value];
    pub fn cardinality(&self) -> u64;
    pub fn value_at(&self, index: u64) -> Option<&Value>;
    pub fn coordinate_of(&self, value: &Value) -> Option<u64>;
}
pub struct OrderDomain(Arc<[Value]>);                   // the same derives
impl OrderDomain {
    pub fn new(elements: Vec<Value>) -> Result<Self, StepDomainError>;
    pub fn elements(&self) -> &[Value];
    pub fn cardinality(&self) -> BigUint;               // n!
    pub fn value_at(&self, positions: &[u32]) -> Option<Value>;   // a Value::Tuple
    pub fn coordinate_of(&self, value: &Value) -> Option<Box<[u32]>>;
}
pub struct StridedRun { /* start: BigInt, stop: BigInt, stride: BigUint */ }   // Debug, Clone, PartialEq, Eq, Hash
impl StridedRun {
    pub fn new(start: BigInt, stop: BigInt, stride: BigUint) -> Result<Self, StepDomainError>;
    pub fn start(&self) -> &BigInt; pub fn stop(&self) -> &BigInt; pub fn stride(&self) -> &BigUint;
    pub fn width(&self) -> BigUint;
}
pub struct StridedDomain(Arc<[StridedRun]>);            // the same derives
impl StridedDomain {
    pub fn new(runs: Vec<StridedRun>) -> Result<Self, StepDomainError>;
    pub fn runs(&self) -> &[StridedRun];
    pub fn cardinality(&self) -> u64;
    pub fn value_at(&self, index: u64) -> Option<BigInt>;
    pub fn coordinate_of(&self, value: &BigInt) -> Option<u64>;
}
#[non_exhaustive]
pub enum StepDomain { Choice(ChoiceDomain), Order(OrderDomain), Strided(StridedDomain) }   // Debug, Clone, PartialEq, Eq, Hash
impl StepDomain {
    pub fn cardinality(&self) -> BigUint;
    pub fn value_at(&self, coordinate: &Coordinate) -> Option<Value>;
    pub fn coordinate_of(&self, value: &Value) -> Option<Coordinate>;
    pub fn signature(&self) -> DomainSignature;         // identifiers as `identifier`
}
pub struct DomainSignature(/* private repr */);          // Debug, Clone, PartialEq, Eq, Hash, serde
impl DomainSignature { pub fn cardinality(&self) -> BigUint; pub fn shape(&self) -> &'static str; }

// search_space::variable (SS1's trait gains one provided method)
pub trait Variable: ForeignPart {
    // ... SS1's methods ...
    /// The domain a static step over this variable offers, or `None` to derive it from the param.
    fn search_domain(&self) -> Result<Option<StepDomain>, BoxError> { Ok(None) }
}

// search_space::trace
pub struct TraceStep { /* kind, subject, decision: Option<u32>, signature, coordinate, value: Option<Value> */ }  // Debug, Clone, PartialEq, Eq, Hash
impl TraceStep {
    pub fn dynamic(kind: DecisionKind, subject: Identifier, domain: &StepDomain, coordinate: Coordinate)
        -> Result<Self, TraceError>;                   // refuses a coordinate outside the domain
    pub fn kind(&self) -> &DecisionKind;
    pub fn subject(&self) -> &Identifier;               // a static step's decision name
    pub fn decision(&self) -> Option<usize>;            // a static step's canonical position in its space
    pub fn signature(&self) -> &DomainSignature;
    pub fn cardinality(&self) -> BigUint;
    pub fn coordinate(&self) -> &Coordinate;
    pub fn value(&self) -> Option<&Value>;
}
pub struct Trace(Arc<[TraceStep]>);                     // Debug, Clone, PartialEq, Eq, Hash, Default, Display, Serialize, Deserialize
impl Trace {
    pub fn new(steps: Vec<TraceStep>) -> Self;
    pub fn steps(&self) -> &[TraceStep];
    pub fn len(&self) -> usize; pub fn is_empty(&self) -> bool;
    pub fn coordinates(&self) -> impl ExactSizeIterator<Item = &Coordinate> + '_;
    pub fn of_kind<'a>(&'a self, kind: &'a DecisionKind) -> impl Iterator<Item = &'a TraceStep> + 'a;
    pub fn traversed_cardinality(&self) -> BigUint;
}

// search_space::oracle
pub trait SearchOracle {
    fn decide(&mut self, step: &PendingStep<'_>) -> Result<Coordinate, BoxError>;
}
impl<O: SearchOracle + ?Sized> SearchOracle for &mut O;
impl<O: SearchOracle + ?Sized> SearchOracle for Box<O>;
pub struct PendingStep<'a> { /* ... */ }
impl<'a> PendingStep<'a> {
    pub fn kind(&self) -> &'a DecisionKind;
    pub fn subject(&self) -> &'a Identifier;
    pub fn domain(&self) -> &'a StepDomain;
    pub fn position(&self) -> usize;                    // the step's index in the run's trace
    pub fn decision(&self) -> Option<Decision<'a>>;     // a static step's decision
    pub fn configuration(&self) -> Option<&'a Configuration>;   // the run's configuration so far
    pub fn admits(&self, coordinate: &Coordinate) -> Result<bool, TraceError>;
    pub fn draw_uniform(&self, rng: &mut Rng) -> Result<Coordinate, TraceError>;
}
pub struct RandomOracle { /* rng */ }                   // Debug, Clone
impl RandomOracle { pub fn new(seed: u64) -> Self; pub fn from_rng(rng: Rng) -> Self; pub fn rng(&self) -> &Rng; }
pub struct ReplayOracle { /* trace, position */ }       // Debug, Clone
impl ReplayOracle {
    pub fn new(trace: Trace) -> Self;
    pub fn trace(&self) -> &Trace;
    pub fn is_exhausted(&self) -> bool;
    pub fn finish(self) -> Result<(), ReplayError>;     // refuses unasked steps
}
pub struct ExhaustiveOracle { /* path */ }              // Debug, Clone, Default
impl ExhaustiveOracle {
    pub fn new() -> Self;
    pub fn advance(&mut self) -> bool;                  // the next run's path; false when every path was taken
    pub fn is_backtrack(error: &BoxError) -> bool;      // a run abandoned on an exhausted branch
}

// search_space::recorder
pub struct Recorder<'r, 'c> { /* oracle, context, space, configuration, preset, steps */ }
impl<'r, 'c> Recorder<'r, 'c> {
    pub fn new(oracle: &'r mut dyn SearchOracle, context: &'r ParamContext<'c>) -> Self;          // dynamic steps only
    pub fn over(space: &Space, oracle: &'r mut dyn SearchOracle, context: &'r ParamContext<'c>) -> Self;
    pub fn realizing(configuration: &Configuration, oracle: &'r mut dyn SearchOracle,
                     context: &'r ParamContext<'c>) -> Self;
    pub fn decide(&mut self, decision: &Identifier) -> Result<Value, TraceError>;
    pub fn decide_dynamic(&mut self, kind: &DecisionKind, subject: &Identifier, domain: &StepDomain)
        -> Result<Coordinate, TraceError>;
    pub fn trace(&self) -> Trace;                       // the steps so far, also after a failure
    pub fn configuration(&self) -> Option<&Configuration>;
    pub fn finish(self) -> Result<Recorded, TraceError>;
}
pub struct Recorded { /* trace, configuration */ }      // Debug, Clone; trace(), configuration(), into_parts()

// search_space::exploration
impl Space {
    pub fn sample(&self, oracle: &mut dyn SearchOracle, context: &ParamContext<'_>) -> Result<Recorded, TraceError>;
    pub fn sample_uniform(&self, rng: &mut Rng, context: &ParamContext<'_>, attempts: NonZeroU32) -> Result<Recorded, TraceError>;
    pub fn replay(&self, trace: &Trace, context: &ParamContext<'_>) -> Result<Configuration, ReplayError>;
    pub fn enumerate<'s>(&'s self, context: &'s ParamContext<'s>) -> Enumeration<'s>;
    pub fn cardinality(&self, context: &ParamContext<'_>, budget: u64) -> Result<Cardinality, TraceError>;
    pub fn mutate(&self, configuration: &Configuration, rng: &mut Rng, context: &ParamContext<'_>, attempts: NonZeroU32)
        -> Result<Recorded, TraceError>;
}
impl Configuration { pub fn trace(&self, context: &ParamContext<'_>) -> Result<Trace, TraceError>; }
pub enum Cardinality { Exact(BigUint), AtLeast(BigUint), Unbounded { decision: Identifier }, Unknown { decision: Identifier } }  // Debug, Clone, PartialEq, Eq; exhaustive
pub struct Enumeration<'s> { /* ... */ }                // Iterator<Item = Result<Configuration, TraceError>>

// search_space::error
pub struct EmptyKind;                                   // Debug, Clone, Copy, PartialEq, Eq, Display, Error
#[non_exhaustive] pub enum StepDomainError {
    EmptyChoice, EmptyOrder, EmptyRuns, NanValue { index: usize }, RepeatedValue { first: usize, second: usize },
    EmptyRun { index: usize }, ZeroStride { index: usize }, UnorderedRuns { index: usize }, TooLarge,
}
#[non_exhaustive] pub enum TraceError {
    Domain(StepDomainError), UnknownDecision { name: Identifier }, AlreadyDecided { name: Identifier },
    NotActive { name: Identifier, activity: Activity }, NotEnumerable { decision: Identifier },
    NoSpace, CoordinateOutOfDomain { position: usize }, Inadmissible { position: usize, coordinate: Coordinate },
    DeadEnd { decision: Identifier }, Oracle { position: usize, source: BoxError },
    Hook { decision: Identifier, source: BoxError }, Configuration(ConfigurationErrors),
    Unasked { decisions: Vec<Identifier> }, Incomplete, OtherSpace, NothingToMutate, AttemptsExhausted { attempts: u32 },
}
#[non_exhaustive] pub enum ReplayError {
    Exhausted { position: usize }, KindMismatch { position: usize }, DecisionMismatch { position: usize },
    DomainMismatch { position: usize }, CoordinateOutOfDomain { position: usize },
    Inadmissible { position: usize }, Unconsumed { position: usize },
    MissingStep { decision: Identifier }, RepeatedStep { decision: Identifier },
    Configuration(ConfigurationErrors), Trace(Box<TraceError>),
}
```

The `Display` texts are decided in the stub, one lowercase line each,
naming identifiers as `name::id` and positions as `step <n>`.

### Ownership and data model

- Domains, `Trace` and `DomainSignature` are `Arc`s of immutable data;
  clones share. A `TraceStep` holds a signature, not its domain: a
  recorded step never holds a dynamic domain's values beyond the one
  answered, so a trace of an address stream stays as small as its
  coordinates.
- A `Recorder` borrows its oracle mutably and its context; it owns a clone
  of its space (a reference count), the configuration it grows, and its
  steps. `trace()` copies the steps out, so a failed run still reports its
  prefix (MOGA's `last_point` convention, `sampling.py:139-146`).
- `PendingStep` borrows from the recorder; an oracle cannot keep it. A
  Python oracle receives an owned snapshot (see "Python interface").
- `SearchOracle` has no `Send` bound: an oracle answers one run on one
  thread. `RandomOracle`, `ReplayOracle`, `ExhaustiveOracle` and `Rng`
  are `Send + Sync`.
- `SearchOracle` is not a `ForeignPart`: oracles are never stored in a
  space, compared or serialized.

### Extensibility decisions

- **The decision kind is open and is a string**, not an interned
  identifier tag. A `DescribedTag` is keyed by an `Identifier`
  (`rust/fhy-core/src/described_tag.rs:1-17`), and only fhy-core's
  reserved identifiers have the same id in every process
  (`rust/fhy-core/src/identifier.rs:7-10`): a MOGA-VM kind minted in one
  process would not equal the same kind decoded from another process's
  trace. A namespaced string is stable everywhere, as a part's `kind()` is
  (`search_space/variable.rs:33`). Its only check is non-emptiness.
- **Kinds of static steps** are fhy-core's: `search_space.choice` for a
  choice and the variable's own `kind()` for a variable
  (`search_space.variable`, or an implementor's such as
  `moga.cir.array_tile_knob`). A dynamic step's kind is the caller's.
- **The domain shapes are closed** (`StepDomain` is `#[non_exhaustive]`, no
  custom shape). Every shipped oracle, the replay check and the wire form
  must understand every shape; MOGA-VM's three shapes cover every free
  choice it makes (`decisions.py:425-432`). A new shape is additive.
- **`SearchOracle` is open and object safe**; not sealed. Its one method
  answers a coordinate, not a value: coordinates are what a trace records
  and what makes an oracle independent of the objects a domain holds.
  `PendingStep::domain().coordinate_of(value)` turns a value into one.
- **`Variable::search_domain`** lets an implementor whose param has a
  custom domain offer a step domain (or narrow nothing: the default
  derives one from the param). Contract clause 7: the domain holds exactly
  the values the param's domain admits, before its constraints, in an
  order fixed for the value's life; checked by `ContractClause::SearchDomain`
  where the param's domain is finite.
- `Coordinate` and `Cardinality` are exhaustive (two shapes of answer;
  four answers to "how many"). The error enums are `#[non_exhaustive]`.

### Domains

| Shape | Values | Coordinate | Cardinality | Refused |
|---|---|---|---|---|
| choice | one or more `Value`s, distinct by `Value`'s type-strict `==` | `Index(i)`, `i < n` | `n` | none (`EmptyChoice`), a NaN float (`NanValue`), a repeat (`RepeatedValue`) |
| order | the permutations of one or more distinct `Value`s; a value is a `Value::Tuple` holding each element once | `Order(p)`: `p[k]` is the position among the elements of the one placed `k`-th, as `decisions.py:225-248` | `n!` (a `BigUint`) | none (`EmptyOrder`), NaN, a repeat |
| strided | the integers `start + k * stride` below `stop`, for each run; runs disjoint and ascending | `Index(i)` over the runs flattened in order (`decisions.py:369-396`), so a uniform index is uniform over the union | the sum of the widths | no run (`EmptyRuns`), `stop <= start` (`EmptyRun`), a zero stride (`ZeroStride`), a run starting below the previous run's stop (`UnorderedRuns`), more than `u64::MAX` values (`TooLarge`) |

- `admits` is `coordinate_of(value).is_some()`. Values compare
  type-strictly: `true` is not the integer `1` (F-SS-022), and a strided
  domain admits only `Value::Int`.
- A static step's domain:
  - a choice: a choice domain over its alternatives' names, as identifier
    values, in declared order;
  - a variable: `search_domain()` if it gives one; else from the param's
    domain (`rust/fhy-core/src/param/domain.rs:562-576`): categorical and
    ordinal as a choice domain over `values()` (`domain.rs:467, 503`),
    permutation as an order domain over `values()` (`domain.rs:531`),
    integer and interval integer as one strided run over the effective
    interval of the param's bound constraints when both ends are finite;
    an unbounded integer, a real or a custom domain has none
    (`TraceError::NotEnumerable`).
- **Admissible coordinates.** A coordinate of a static step is admissible
  when the run's configuration with that value (`Configuration::with_entry`,
  `search_space/configuration.rs:144`) is accepted: the param's
  constraints hold for it, and no forbidden clause it completes holds. Any
  other refusal (an undecided condition or clause, a failing custom
  constraint) is an error of the run, not an inadmissible value. Every
  coordinate of a dynamic step is admissible: its domain is the
  admissible set, as MOGA-VM's are built (`placement.py:423-500`).
  Coordinates index the declared domain, not the admissible subset, so a
  coordinate means the same value whatever a forbidden clause filters.

### Signatures and replay

A `DomainSignature` is what replay compares: the parts of a domain that
mean the same thing in every module and every process.

- Its shape and cardinality.
- A choice's values and an order's elements, each written as:
  - itself, if it is a plain value (Boolean, integer, float, decimal,
    string, or a tuple or frozen set of plain values);
  - its position among the space's names, if it is an identifier the
    step's space binds (static steps; as `ConfigurationKey` writes one,
    `search_space/configuration.rs:414-420`);
  - `identifier`, for any other identifier;
  - `opaque`, for an opaque value (a Python object such as MOGA's
    `RealizationOption`, whose `==` is identity).
- A strided domain's runs, exactly.

So a replay against a module whose free address runs moved, or whose
plain choices were reordered, is refused (F-SS-020), while a replay of
options or walk levels against a freshly built module, whose objects are
new, still works, which is the reason MOGA-VM compares sizes only
(`oracle.py:287-295`). Two alpha-equivalent spaces give equal signatures
for corresponding static steps.

**`ReplayOracle`** answers the `n`-th step asked with the `n`-th step
recorded, checking, in order:

1. a recorded step exists (`Exhausted`);
2. equal kinds (`KindMismatch`);
3. both static over the same canonical position, or both dynamic
   (`DecisionMismatch`); a dynamic step's subject is not compared, as
   MOGA-VM's is not (an address's subject repeats, and subjects are
   module objects), and the documentation now says so (F-SS-027);
4. equal signatures (`DomainMismatch`);
5. the recorded coordinate lies in the offered domain
   (`CoordinateOutOfDomain`) and is admissible (`Inadmissible`).

`finish()` refuses a replay that left recorded steps unasked
(`Unconsumed`): a stream shorter than its trace is a different path
(F-SS-021). `Space::replay` and `Recorder::finish` over a `ReplayOracle`
call it.

**`Space::replay(trace)`** builds the configuration a trace's static steps
describe, whatever order they were asked in (a pipeline asks every option
before any tile, while decision order interleaves them):

- each static step names a canonical position; two steps for one decision
  are `RepeatedStep`;
- it walks the space in decision order; each active decision takes its
  step (`MissingStep` if none), checked by kind, signature and coordinate
  as above, and an inactive decision must have none (`DecisionMismatch`);
- dynamic steps are skipped: they are answered back to whatever asked
  them, by a `ReplayOracle` in a `Recorder`.

The result is the configuration of this space, so a trace recorded over
one space replays into a relabeled copy, and the two configurations have
equal keys (a property test).

### The recorder

- `Recorder::new(oracle, context)` records dynamic steps only;
  `decide` is `NoSpace`.
- `Recorder::over(space, ...)` starts from the empty configuration of
  `space`. `decide(name)`:
  - the name is a decision of the space (`UnknownDecision`), not decided
    yet in this run (`AlreadyDecided`), and active in the configuration so
    far (`NotActive` with its activity: a decision whose choice is still
    undecided is pending);
  - it builds the step and its `PendingStep`, asks the oracle (`Oracle`
    wraps the oracle's error, which a Python oracle's exception is), and
    refuses a coordinate outside the domain (`CoordinateOutOfDomain`) or
    not admissible (`Inadmissible`); a refused answer is not recorded
    (`oracle.py:188-208`);
  - it records the step, extends the configuration, and returns the value
    (a choice's value is its alternative's name).
- `decide_dynamic(kind, subject, domain)` does the same for a domain the
  caller built, and returns the coordinate; the caller maps it to its own
  object.
- `Recorder::realizing(configuration, ...)` is MOGA-VM's
  `ExtractedPointOracle` (`extraction.py:358-428`) made generic: a static
  decision the configuration assigns is answered from it, without asking
  the oracle, and recorded with its coordinate; any other decision, static
  or dynamic, goes to the oracle. Because names are unique in a space, the
  first-in-first-out matching of repeated axis keys disappears, and with
  it F-SS-026.
- `finish()` returns the trace and, over a space, the configuration
  (complete or not: `is_complete()` says). A realizing recorder refuses to
  finish while a decision its configuration assigns was never asked
  (`Unasked`, naming them in canonical order): MOGA-VM's strict mode
  (`sampling.py:194-214`), always on. A caller that tolerates it reads
  `trace()` and `configuration()` instead.

### Oracles fhy-core ships

| Oracle | Answers | Notes |
|---|---|---|
| `RandomOracle` | `step.draw_uniform(rng)`: uniform over the step's admissible coordinates | uniform per step, not per configuration, as MOGA-VM's (`oracle.py:85-96`) |
| `ReplayOracle` | the recorded step at the same position | checks above; `finish` |
| `ExhaustiveOracle` | the paths of a conditional stream in lexicographic order of coordinates | across runs: `advance()` between them |

- **`draw_uniform`**: for a finite domain of `n` coordinates, draw
  `rng.below(n)` (an order domain: shuffle the positions); if the
  coordinate is not admissible, draw again, up to 64 draws; then list the
  admissible coordinates (when `n <= 2^16`) and draw among them; none is
  `DeadEnd`. Rejection keeps the draw exactly uniform over the admissible
  set; the cap bounds the work on a domain that is mostly forbidden.
- **`ExhaustiveOracle`** keeps the path of the last run: per position, the
  coordinate answered and the domain's cardinality. A run answers the
  path's coordinates in order and then, at each new position, the first
  admissible coordinate. At a position whose remaining coordinates are all
  inadmissible it answers with a backtrack error (`is_backtrack`), which
  abandons the run. `advance()` moves to the next coordinate of the
  deepest position with one left (an order domain's next permutation in
  lexicographic order), drops the positions after it, and returns `false`
  when none is left. A run whose shape differs from the path at a
  replayed position is a `ReplayError` (the stream is not deterministic).
  Over a space it enumerates every complete configuration once; over
  MOGA-VM's pipeline it is the sweep its design names
  (`docs/design/search_based_lowering_strategy.md`, "Behavior").

### Sampling, enumeration, cardinality and mutation over a `Space`

**`Space::sample(oracle)`**: a `Recorder::over` asking every active
decision in decision order (`search_space/space.rs:290`). Decision order
puts every decision after those it depends on, so each decision is active
or inactive when reached, never pending; an undecided or failing
condition is `TraceError::Configuration`. The configuration returned is
complete.

**`Space::sample_uniform(rng, attempts)`** draws a complete configuration
uniformly from all of them, by constructive proposal and rejection:

1. The **relaxation** of the space drops its conditions and forbidden
   clauses and its params' constraints. Its count `R(d)` is closed form:
   a variable's declared cardinality; a choice's sum over its
   alternatives of the product of their decisions' counts; the space's the
   product of its top-level decisions'. `R` must be finite
   (`NotEnumerable` otherwise).
2. Draw an index uniformly below `R(space)` (`rng.below_big`) and decode
   it, mixed radix, into a relaxed point: an alternative per reached
   choice, a coordinate per reached variable.
3. Apply the space: walk decision order, keep the relaxed values of
   active decisions, drop those of decisions a condition makes inactive.
   Reject if a value is refused (a param constraint) or a forbidden clause
   holds.
4. A configuration `c` is the image of `m(c)` relaxed points: the product
   of `R(d)` over the decisions `d` that are inactive in `c` by a
   condition while their parent is active. Accept with probability
   `1/m(c)` (draw below `m(c)`, accept on zero).

Each complete configuration is then proposed with probability `m(c)/R`
and accepted with `1/m(c)`: exactly uniform. With no conditions every
`m(c)` is 1. After `attempts` rejections, `AttemptsExhausted`. The trace
returned holds the accepted configuration's steps, as
`Configuration::trace` gives them. MOGA-VM's `draw_uniform`
(`extraction.py:315-344`) is the per-step sampler, `sample` with a
`RandomOracle`; the probe X3 shows the difference (an option exposing 3
tiles against one exposing 2 is drawn 0.5/0.5 per step, 0.6/0.4 per
configuration).

**`Space::enumerate()`** drives `sample` with an `ExhaustiveOracle` and
yields each complete configuration in lexicographic order of its
coordinates in decision order. A backtrack yields nothing; a decision
with no finite domain yields one `NotEnumerable` and ends.

**`Space::cardinality(budget)`** counts the complete configurations:

- The space splits into **components**: top-level decisions are in one
  component when a condition or a forbidden clause names decisions of
  both. Components multiply.
- A component with no condition, forbidden clause or param constraint
  beyond the bounds that define a strided domain has a closed-form count
  (as the relaxation's); otherwise it is counted by enumerating it,
  checking at most `budget` configurations.
- Each component answers:
  - `Exact(n)`;
  - `AtLeast(n)`: the budget ran out after `n`;
  - `Unbounded { decision }`: a valid configuration activates a variable
    whose domain is an unbounded integer or a real, so the count is
    infinite unless a constraint the count does not solve excludes every
    value, which it does not try;
  - `Unknown { decision }`: a custom domain with no `search_domain`.
- Combined: any `Exact(0)` is `Exact(0)`; else any `Unknown` is
  `Unknown`; else any `Unbounded` is `Unbounded` if every other component
  has at least one configuration, and `Unknown` otherwise; else any
  `AtLeast` is `AtLeast` of the product; else `Exact` of the product.
- MOGA-VM's one-level count (`extraction.py:293-313`) is the closed form
  of a space whose choices hold variables only; the equivalence corpus
  checks the two agree.

**`Space::mutate(configuration, rng, attempts)`** changes one decision and
repairs the rest, the MetaSchedule mutator shape:

1. `configuration` must be complete (`Incomplete`) and belong to this
   space by `Space`'s `==` (`OtherSpace`).
2. Pick uniformly an active decision with two or more admissible values
   (`NothingToMutate` if none).
3. Its new value: an order domain swaps two positions drawn uniformly; any
   other shape draws uniformly among the other admissible coordinates.
4. Re-walk decision order with a private oracle that answers the mutated
   decision with its new coordinate, every other decision with its old
   coordinate while that stays admissible, and anything else (a decision
   the change activated, or an old value now inadmissible) by
   `draw_uniform`.
5. A dead end retries from 2, up to `attempts` (`AttemptsExhausted`).

**`Configuration::trace()`** is the trace of a configuration's assigned
decisions, in decision order: the steps `Space::replay` turns back into
it.

### Determinism and the random-number generator

- Every random draw in fhy-core comes from one `Rng` passed in by the
  caller: nothing reads a clock, the environment or a global.
- **The stream contract:** for a seed, `Rng`'s outputs, and for each
  operation (`below`, `below_big`, `shuffle`, `draw_uniform`, `sample`,
  `sample_uniform`, `mutate`), which draws it takes, in which order. Both
  are documented and pinned by golden vectors in the tests; changing
  either is a breaking change noted in the changelog. Arithmetic is
  integer only (wrapping `u64`, `u128` products), so the streams are the
  same on every platform.
- `below(n)` is Lemire's multiply-and-reject method; `below_big` draws
  whole 64-bit limbs, masks the top limb to the bound's bit length and
  rejects at or above the bound; `shuffle` is Fisher-Yates from the last
  index down; `split` seeds a new generator from one output.
- **Python.** The binding exposes the same `Rng`, so a seed reproduces the
  same run from Python and from Rust. Python's `random.Random` stream
  (MT19937, and CPython's `randrange` and `shuffle`, which the language
  does not promise to keep) cannot be reproduced by either option below:
  MOGA-VM's existing seeds do not carry over, and its seed-pinned tests
  are re-baselined when it migrates (a divergence, D-SS2-6).
- **The generator (N-S1, decided: (b), SplitMix64):**

  | | (a) `rand` + `rand_chacha` | (b) in house (recommended) |
  |---|---|---|
  | generator | ChaCha12 (`rand_chacha` documents its streams as portable and reproducible) | SplitMix64 (64-bit state; passes BigCrush; Java's `SplittableRandom`) or PCG64 |
  | new dependencies | two normal dependencies of the published crate. Both are in `Cargo.lock` already at 0.9.5 and 0.9.0, as `proptest`'s dev dependencies (`Cargo.lock:515-533`), so the lock does not change, but fhy-core's users then build them, and `rand_core`'s `os_rng` feature pulls `getrandom` unless default features are off | none |
  | stream stability | the generator's, yes; `rand`'s range sampling and shuffling can change value output in a minor (0.x) release, as 0.9 did for integer ranges, so fhy-core would write `below` and `shuffle` itself over `RngCore::next_u64` to keep its contract across `rand` bumps | fhy-core owns the algorithm and the draws; golden vectors pin them |
  | code | the newtype and the draws (about 80 lines) | the generator, the newtype and the draws (about 120 lines) |
  | API | `Rng` wraps the third-party type; nothing of `rand` in fhy-core's signatures | `Rng` |
  | Python interop | the same binding either way | the same |

  The recommendation is (b): the draws must be written in house under
  (a) as well to keep the contract, which leaves `rand` providing only a
  generator that is a few lines of code, at the cost of two published
  dependencies. `Rng::ALGORITHM` names the generator, so a stronger one
  can be added later without changing old streams.

### Error model

- Domain construction stops at the first problem (`StepDomainError`, in the
  table's order).
- A run stops at the first problem (`TraceError`): a run is a path, and a
  later step depends on the earlier ones. The oracle's own error is
  boxed in `Oracle { position, source }`; through Python it is the
  exception itself.
- Replay stops at the first mismatch (`ReplayError`), naming the step's
  position. Through a `Recorder`, the `ReplayOracle`'s error arrives as
  `TraceError::Oracle` with the `ReplayError` as its source; the binding
  raises it as `ReplayMismatchError`.
- A configuration refused while being grown is
  `TraceError::Configuration`, holding SS1's collected errors.
- No panics on input. `Rng::below_big` with a zero bound panics: every
  caller passes a cardinality, which is at least one by construction.

### Behavior

| Question | Answer |
|---|---|
| a step whose domain has one value | asked and recorded, as MOGA-VM's are (a fact is a decision with cardinality 1, C-7) |
| `traversed_cardinality` of an empty trace | 1 |
| a choice domain over `1` and `true` | two values; `coordinate_of(true)` is 1 (MOGA-VM: 0, F-SS-022) |
| a strided domain and `true` | not admitted (MOGA-VM's `coordinate_of` accepts it, probe C3) |
| an order domain over no element | refused (MOGA-VM accepts it with cardinality 1; its policies never ask one) |
| a dynamic step's subject in replay | not compared |
| a replay against a same-size domain with other plain values | refused (`DomainMismatch`) |
| a replay that asks fewer steps than recorded | refused by `finish` (`Unconsumed`) |
| `Space::replay` and step order | static steps by canonical position, any order |
| a trace's equality | structural over its steps; traces from two modules differ in their subjects, so compare `coordinates()` |
| `Trace`'s `Display` | `0 steps`, or `5 steps (2 search_space.choice, 3 moga.cir.address)`, kinds in first-seen order (MOGA's `summarize`, `decisions.py:567-575`) |

### Serialization

- Type id `search_space.trace`. The core's serde shape (V2, no V1 form):

  ```json
  {"steps": [
    {"kind": "search_space.choice", "subject": {"id": 60001, "name_hint": "layout"},
     "decision": 3, "domain": {"choice": [{"bound": 4}, {"bound": 5}]},
     "coordinate": {"index": 1}, "value": {"identifier": {"id": 60005, "name_hint": "flat"}}},
    {"kind": "moga.cir.address", "subject": {"id": 70210, "name_hint": "x"}, "decision": null,
     "domain": {"strided": [{"start": "0", "stop": "64", "stride": "1"}]},
     "coordinate": {"index": 17}, "value": {"int": "17"}}
  ]}
  ```

  A coordinate is `{"index": n}` or `{"order": [..]}`, externally tagged so
  non-self-describing formats (postcard) read it; a signature member is
  `{"value": <value>}`, `{"bound": n}`, `"identifier"` or `"opaque"`.
- A step's value is written unless it is or holds an opaque value, which
  is written `null`: an opaque value is a module's object, which another
  process cannot read back. Replay never reads values.
- A trace holds no foreign part, so `Trace` implements `Serialize` and
  `Deserialize` itself; decoding checks each coordinate against its
  signature. `wire::TraceData` is the shape, for the binding.
- `Rng` serializes its algorithm and state, so a `RandomOracle` resumes.
- So a MOGA-VM trace row can carry `trace.to_json()`, and a row replays
  without the process that wrote it (F-SS-024).

### Python interface

New in `fhy_core.search_space` (the `_rs` pyclasses behind thin public
classes, as SS1's):

| Class | Construction | Attributes and methods |
|---|---|---|
| `Rng` | `Rng(seed)`: an `int` in `[0, 2**64)` (`ValueError` otherwise) | `seed`, `next_u64()`, `below(n)`, `shuffle(list)` (in place), `split()`; pickles with its state |
| `ChoiceDomain` | `ChoiceDomain(choices)`: any objects; each is read as a `Value`, else as an opaque value compared by its own `==` (identity for an `eq=False` dataclass, as MOGA-VM's options) | `choices` (the objects given), `cardinality`, `admits(value)`, `value_at(i)` (the object given: `domain.value_at(0) is choices[0]`), `coordinate_of(value)` |
| `OrderDomain` | `OrderDomain(elements)` | `elements`, `cardinality`, `admits`, `value_at(positions)` (a tuple of the objects given), `coordinate_of` |
| `StridedRun` | `StridedRun(start, stop, stride=1)` | `start`, `stop`, `stride`, `width`, `admits` |
| `StridedDomain` | `StridedDomain(runs)` | `runs`, `cardinality`, `admits`, `value_at`, `coordinate_of` |
| `PendingStep` | none (the binding builds it) | `kind` (a `str`), `subject`, `domain` (the domain object a dynamic step was given; a built one for a static step), `position`, `decision` (the `Variable` or `Choice` object, or `None`), `configuration` (or `None`), `admits(coordinate)`, `coordinate_of(value)`, `draw_uniform(rng)` |
| `SearchOracle` | a `typing.Protocol`, runtime checkable: `decide(self, step: PendingStep) -> int \| tuple[int, ...]` | |
| `RandomOracle` | `RandomOracle(seed=None, *, rng=None)`; both given is `ValueError`; `seed=None` draws a seed from `os.urandom` | `seed` (the seed used, so an unseeded run can be reproduced, unlike MOGA-VM's, F-SS-025), `rng`, `decide` |
| `ReplayOracle` | `ReplayOracle(trace)` | `trace`, `is_exhausted`, `finish()`, `decide` |
| `ExhaustiveOracle` | `ExhaustiveOracle()` | `advance()`, `decide` |
| `Recorder` | `Recorder(oracle, *, space=None, configuration=None)`: `configuration` makes it realizing; `space` and `configuration` together is `ValueError` | `decide(name)` (the value: for a choice the alternative object), `decide_dynamic(kind, subject, domain)` (the domain's object at the answered coordinate), `trace`, `configuration`, `finish()` (a `(trace, configuration)` pair) |
| `TraceStep` | none | `kind`, `subject`, `decision`, `signature` (read only, `repr` only), `cardinality`, `coordinate`, `value` (the object answered, kept, for a trace recorded in this process; the decoded value otherwise, `None` for an opaque one) |
| `Trace` | `Trace(steps=())` | `steps`, `len`, iteration, `coordinates`, `of_kind(kind)`, `traversed_cardinality`, `str` (the `Display` text); structural `==` and `hash` (N-S2); the serialization methods; pickles through its V2 text |
| `Cardinality` | none | `kind` (`CardinalityKind`, a `StrEnum`: `EXACT`, `AT_LEAST`, `UNBOUNDED`, `UNKNOWN`), `count` (`int` or `None`), `decision` (or `None`) |

- `Space` gains `sample(oracle)`, `sample_uniform(rng, *, attempts=1000)`,
  `replay(trace)`, `enumerate()` (an iterator), `cardinality(*,
  budget=100_000)` and `mutate(configuration, rng, *, attempts=16)`;
  `sample`, `sample_uniform` and `mutate` return a `(configuration,
  trace)` pair. `Configuration` gains `trace()`. Each runs under the
  default solver's context, as `Configuration(space, entries)` does.
- `Variable` gains the hook `extension_search_domain(self) -> ChoiceDomain
  | OrderDomain | StridedDomain | None` (default `None`), adapted as SS1's
  hooks are: called once per step built, a wrong result type is
  `TypeError`, an exception propagates as itself.
- **Errors:** `StepDomainError(SearchSpaceError)`;
  `TraceError(SearchSpaceError)` with `InadmissibleAnswerError`,
  `ReplayMismatchError`, `NotEnumerableError` and `DeadEndError` beneath
  it. As MOGA-VM's (`errors.py:1-19`), none of them derives from anything
  a search catches as an infeasible program.

**Path 1: a Python oracle.** Any object with a callable `decide` (the
`SearchOracle` protocol; no base class to subclass). The binding wraps it
in an adapter, `PythonOracle`, which implements the trait:

- each `decide` call receives an owned `PendingStep` snapshot (it holds
  the space and the configuration so far, so `admits` and `draw_uniform`
  still work after the call, under the default solver's context);
- the result must be an `int` or a tuple of `int`s (`bool` refused):
  `TypeError: <Class>.decide must return an int or a tuple of ints, got
  <type>.`;
- an exception it raises is boxed and raised by the entry point as the same
  object; `KeyboardInterrupt` passes through (D-S8-11's rules);
- a Python oracle drawing randomly can use the step's
  `draw_uniform(rng)` with an `Rng` to stay on fhy-core's stream.

**Path 2: a downstream Rust oracle.** `convert::search_space` gains:

```rust
pub type OracleLease = for<'a, 'py> fn(&'a Bound<'py, PyAny>) -> PyResult<Box<dyn SearchOracle + 'a>>;
pub fn register_oracle_kind(module: &Bound<'_, PyModule>, kind: &str, class: &Bound<'_, PyType>,
    lease: OracleLease) -> PyResult<()>;
pub fn variable_search_domain_to_python<'py>(py: Python<'py>, domain: &StepDomain) -> PyResult<Bound<'py, PyAny>>;
pub fn step_domain_from_python(object: &Bound<'_, PyAny>) -> PyResult<StepDomain>;
```

- A lease borrows the oracle out of its pyclass for one run, typically a
  `PyRefMut` wrapped in a forwarding type, so the run calls the Rust
  oracle with no Python call per step. A second lease while the first is
  held fails as `PyRefMut` does (`RuntimeError`).
- Reading an oracle argument tries, in order: fhy-core's own oracle
  classes (native); a registered kind, of whose class the object is an
  instance (its lease); any object with a callable `decide` (the adapter);
  anything else is `TypeError`.
- The registry is SS1's (`rust/fhy-core-py/src/search_space/kinds.rs:96`),
  gaining a third family; its refusals are SS1's (a kind or a class
  registered twice, a built-in kind). Oracles are never decoded, so the
  family has no resolver.
- `rust/example-aggregate` gains `CountingOracle`, a Rust oracle kind
  answering every step with coordinate 0 and counting the steps, and its
  registration.

### Non-goals

- A custom domain shape; a step over an unbounded integer or a real domain
  (its param must bound it, or `search_domain` must offer one).
- Crossover and other operators over two configurations: a caller builds
  one from coordinates by canonical position and `Space::replay`.
- Learned or model-based oracles, budgets, harnesses, records, observers:
  MOGA-VM's (or a later slice's).
- Parallel runs: an oracle answers one run on one thread; `Rng::split`
  gives independent streams for runs a caller parallelizes.
- An incremental configuration check for `Recorder::decide`: each step
  re-checks the configuration through `with_entry` (4.2 µs at SS1.8's
  8×4 space). A `pub(super)` incremental path is an implementation choice
  if the benchmarks need it, not API.

### Test plan (SS2.3)

In `rust/fhy-core/tests/it/search_space/`, through the public API:

- `rng_stories.rs`: golden vectors (the first 16 outputs for seeds 0, 1
  and `u64::MAX`; `below` for bounds 1, 3, `2^63 + 1`; `below_big`;
  `shuffle` of 0..10; `split`); `below` never reaches its bound; a
  chi-square bound on 60 000 draws of `below(6)`; serde resumes a stream.
- `domain_stories.rs`: every `StepDomainError` in order; cardinalities;
  `value_at`/`coordinate_of` round trips and refusals; type-strict
  admission; strided flattening, gaps, strides; `TooLarge`; signatures
  (plain, bound, free identifier, opaque).
- `trace_stories.rs`: `TraceStep::dynamic` refusals; `traversed_cardinality`,
  `of_kind`, `coordinates`, `Display`; equality.
- `recorder_stories.rs`: every `TraceError` a recorder raises; nothing
  recorded on a refused answer; the prefix kept after an oracle's error;
  activity order (a variable under an undecided choice is pending);
  forbidden clauses filtering the completing step; `realizing` answering
  from the configuration and `Unasked`.
- `oracle_stories.rs`: `RandomOracle` reproducible, seed-sensitive,
  covering every choice, both runs of a split strided domain, every
  permutation of a small order domain, never an inadmissible coordinate;
  `ReplayOracle`'s five checks and `finish`; `ExhaustiveOracle` over a
  conditional stream visiting each path once, backtracking over an
  exhausted branch, refusing a nondeterministic stream.
- `exploration_stories.rs`: `sample` complete and valid; `replay` of a
  pipeline-ordered trace, `MissingStep`, `RepeatedStep`, a step for an
  inactive decision; `Configuration::trace` round trip; `enumerate` order;
  `cardinality` in each answer and each combination; `mutate`'s
  refusals and its locality (one decision changed when nothing depends on
  it).
- `properties.rs` (proptest, on the brute-force spaces SS1's properties
  draw): `replay(c.trace()) == c`; a relabeled space replays a trace into a
  configuration with an equal key; `enumerate` yields `Exact` many
  distinct, valid, complete configurations, each once; `sample` and
  `mutate` give valid complete configurations; `sample_uniform`'s
  frequencies over a small conditional space with a forbidden clause pass
  a chi-square test at a fixed seed (the non-vacuity guard draws spaces
  where conditions deactivate something); serde round trips of traces and
  of `Rng`.
- `serde_stories.rs`: the pinned trace shape above; `null` for opaque
  values; refusal of a coordinate outside its signature.
- `implementor_stories.rs`: a variable whose param has a custom domain
  offering `search_domain`; `ContractClause::SearchDomain`.
- `tests/it/search_space_trace_golden.rs`: the oracle corpus (below).

In `tests/search_space/`: `test_trace.py` (the interface suite and the
ported MOGA-VM tests, each docstring citing its MOGA-VM test),
`test_trace_rust_binding.py` (class structure, kept objects, pickling,
the V2 shape, GC), `test_oracle_extension.py` (Python oracles, their
result types, exceptions and `KeyboardInterrupt`; the
`extension_search_domain` hook), `test_composed_search_space.py` (the
example aggregate's `CountingOracle`), `test_trace_properties.py`
(Hypothesis: replay round trips, enumeration counts).

### Traceability (SS2 and SS3)

Every test in MOGA-VM's `tests/cir/lowering/search/` (152 tests over 10
files). "Rust" is `rust/fhy-core/tests/it/search_space/`; "Python" is
`tests/search_space/`.

| MOGA-VM test | Behavior | Port test(s) | Status |
|---|---|---|---|
| `test_decisions.py::test_choice_domain_rejects_an_empty_candidate_set` | no empty choice | Rust `domain_stories::choice_domain_refuses_no_values`; Python `test_trace.py::test_choice_domain_refuses_no_choices` | ported |
| `::test_choice_domain_admits_by_identity_for_eq_less_values` | `eq=False` objects match themselves | Python `test_choice_domain_admits_eq_less_objects_by_identity`; Rust `domain_stories::choice_domain_admits_opaque_values_by_their_own_equality` | ported |
| `::test_order_domain_counts_permutations_and_admits_only_permutations` | `n!`; prefixes, repeats and lists refused | `domain_stories::order_domain_counts_and_admits_only_permutations`; Python the same | ported |
| `::test_order_domain_rejects_repeated_elements` | distinct elements | `domain_stories::order_domain_refuses_a_repeat`; Python | ported |
| `::test_address_interval_rejects_an_empty_run` | no empty run | `domain_stories::strided_run_refuses_an_empty_run`; Python | ported |
| `::test_address_domain_flattens_disjoint_runs_into_one_index_space` | flattening | `domain_stories::strided_domain_flattens_its_runs`; Python | ported |
| `::test_address_domain_admits_nothing_between_its_runs` | gaps, `True` refused | `domain_stories::strided_domain_admits_nothing_between_runs_nor_a_boolean`; Python | ported |
| `::test_address_domain_index_out_of_range_raises` | out of range | `domain_stories::strided_value_at_past_the_end_is_none`; Python `IndexError` | ported |
| `::test_address_domain_rejects_overlapping_or_unsorted_runs` | disjoint, ascending | `domain_stories::strided_domain_refuses_unordered_runs`; Python | ported |
| `::test_address_domain_rejects_an_empty_union` | no runs | `domain_stories::strided_domain_refuses_no_runs`; Python | ported |
| `::test_search_point_traversed_cardinality_multiplies_along_the_path` | product of sizes | `trace_stories::traversed_cardinality_multiplies_the_steps`; Python | ported |
| `::test_search_point_groups_by_kind_in_ask_order` | `of_kind` | `trace_stories::of_kind_keeps_ask_order`; Python | ported |
| `::test_empty_search_point_has_a_traversed_cardinality_of_one` | empty product; "0 decisions" | `trace_stories::empty_trace_counts_one_and_displays_zero_steps`; Python | ported ("0 steps") |
| `::test_choice_domain_round_trips_a_value_through_its_coordinate` | inverse maps | `domain_stories::choice_coordinates_round_trip`; the property; Python | ported |
| `::test_order_domain_round_trips_a_permutation_through_its_coordinate` | positions replay onto fresh elements | `domain_stories::order_coordinates_round_trip_onto_fresh_elements`; Python | ported |
| `::test_order_domain_rejects_a_coordinate_that_is_not_a_permutation` | `(0, 0)` refused | `domain_stories::order_value_at_refuses_a_non_permutation`; Python `ValueError` | ported |
| `::test_strided_address_interval_admits_only_aligned_addresses` | strides | `domain_stories::strided_run_admits_only_its_stride`; Python | ported |
| `::test_strided_address_domain_counts_and_indexes_only_aligned_addresses` | strided union | `domain_stories::strided_domain_indexes_only_its_strides`; Python | ported |
| `::test_address_domain_round_trips_an_address_through_its_coordinate` | inverse maps | `domain_stories::strided_coordinates_round_trip`; Python | ported |
| `::test_address_interval_rejects_a_non_positive_step` | stride at least 1 | `domain_stories::strided_run_refuses_a_zero_stride`; Python (negative and zero) | ported (a negative stride is unrepresentable in Rust) |
| `::test_search_point_coordinates_are_a_bare_vector` | coordinates only | `trace_stories::coordinates_are_the_bare_vector`; Python | ported |
| `test_oracle.py::test_uniform_oracle_is_a_search_oracle` | protocol conformance | Python `test_shipped_oracles_are_search_oracles` | ported (Rust: the trait impls, compile time) |
| `::test_uniform_oracle_reproduces_a_whole_stream_from_its_seed` | seed reproduces | `oracle_stories::random_oracle_reproduces_a_stream_from_its_seed`; Python | ported |
| `::test_uniform_oracle_differs_across_seeds` | seeds differ | `oracle_stories::random_oracle_differs_across_seeds`; Python | ported |
| `::test_uniform_oracle_covers_every_admissible_choice` | coverage | `oracle_stories::random_oracle_covers_every_choice`; Python | ported |
| `::test_uniform_oracle_reaches_both_runs_of_a_split_address_domain` | flattened draw | `oracle_stories::random_oracle_reaches_every_run`; Python | ported |
| `::test_uniform_oracle_draws_every_permutation_of_a_small_order_domain` | permutations | `oracle_stories::random_oracle_draws_every_permutation`; Python | ported |
| `::test_uniform_oracle_refuses_a_seed_and_a_generator_together` | one source | Python `test_random_oracle_refuses_a_seed_and_an_rng` | ported (Rust: two constructors) |
| `::test_recording_oracle_records_what_was_asked_and_what_was_answered` | the step as offered | `recorder_stories::recorder_records_the_step_and_its_answer`; Python | ported (`Recorder`) |
| `::test_recording_oracle_rejects_an_answer_outside_the_offered_domain` | refused, not recorded | `recorder_stories::an_answer_outside_the_domain_is_refused_and_not_recorded`; Python `InadmissibleAnswerError` | ported |
| `::test_replay_oracle_reproduces_a_recorded_stream_exactly` | replay, exhausted | `oracle_stories::replay_reproduces_a_stream`; Python | ported |
| `::test_replay_oracle_refuses_a_stream_that_asks_a_different_decision` | shape mismatch | `oracle_stories::replay_refuses_another_shape`; Python | ported |
| `::test_replay_oracle_refuses_the_same_decision_over_a_different_domain` | size mismatch | `oracle_stories::replay_refuses_another_size`, `::replay_refuses_a_moved_run_of_the_same_size`, `::replay_refuses_reordered_plain_choices` | ported, strengthened (F-SS-020, D-SS2-1) |
| `::test_replay_oracle_refuses_a_stream_longer_than_the_recorded_point` | longer stream | `oracle_stories::replay_refuses_a_longer_stream`, `::replay_finish_refuses_a_shorter_stream` | ported, strengthened (F-SS-021, D-SS2-2) |
| `test_extraction.py::test_realization_refuses_a_drifted_domain` | drift refused; unmatched decisions go to the tail | `recorder_stories::realizing_asks_the_oracle_for_unassigned_decisions`; `oracle_stories::replay_refuses_another_size` | ported (a space's own domains cannot drift; the tail is the oracle) |
| `::test_strict_lowering_refuses_an_unconsumed_assignment` | unasked assignment refused | `recorder_stories::realizing_refuses_to_finish_with_unasked_decisions` | ported (strict always; tolerant callers read `trace()`) |
| `::test_same_key_axes_are_consumed_first_in_first_out` | repeated keys in order | `oracle_stories::replay_answers_repeated_subjects_in_ask_order` | divergence D-SS2-4: static names are unique; repeated dynamic subjects replay positionally |
| `::test_extraction_names_the_single_shot_option_axis`, `::test_extraction_conditions_walk_and_tile_axes_on_their_option` | extraction against the pipeline | analogs `exploration_stories::cardinality_of_a_single_choice`, `::cardinality_of_a_choice_sums_its_alternatives_products` | stay in MOGA-VM |
| `::test_a_sampled_point_lowers_and_the_stream_realizes_its_assignments`, `::test_the_same_seed_reproduces_the_same_lowering` | pipeline | - | stay in MOGA-VM |
| `test_observers.py::test_coordinates_serialize_choice_and_order_decisions_faithfully` | `[1, [1, 2, 0]]` | `trace_serde_stories::trace_writes_an_order_coordinate_as_an_array`, `::coordinate_serializes_tagged_by_its_shape`; Python `test_trace_coordinates_serialize_tagged_by_their_shape` | ported, tagged (`{"index": 1}`, `{"order": [1, 2, 0]}`), so postcard reads it |
| `test_observers.py` (9 others) | logging and JSONL files | - | stay in MOGA-VM |
| `test_records.py::test_a_score_on_an_infeasible_record_is_invalid` | no score without success | SS3 `measurement_stories::a_failed_measurement_holds_no_values`; Python | ported (SS3) |
| `::test_a_score_on_a_lowered_record_is_valid` | score kept | SS3 `measurement_stories::an_ok_measurement_keeps_its_values`; Python | ported (SS3) |
| `::test_best_returns_the_lowest_scored_feasible_record_with_earliest_tie_break` | lower is better | SS3 `objective_stories::minimize_prefers_the_lower_value` (the comparison only) | stays in MOGA-VM (selection); the comparison ported |
| `test_records.py` (31 others) | `CandidateRecord`, `SearchHistory`, `SearchStatistics` | - | stay in MOGA-VM |
| `test_search_budget.py` (8) | `SearchBudget` | - | stay in MOGA-VM |
| `test_search_harness.py` (34) | the harness, unwrapping, gates, selection | - | stay in MOGA-VM |
| `test_random_search.py::test_a_recorded_point_replays_onto_a_freshly_built_module` | replay onto a fresh copy | analog `properties.rs::a_relabeled_space_replays_to_an_equal_key`; Python `test_trace_replays_onto_a_relabeled_space` | stays in MOGA-VM (pipeline); the generic property ported |
| `test_random_search.py` (14 others), `test_random_search_seed_sweep.py` (3) | the pipeline's axes, addresses, seeds | - | stay in MOGA-VM (seed-pinned ones re-baselined, D-SS2-6) |
| `test_search_strategy_integration.py::test_a_trace_row_replays_to_the_same_committed_addresses` | a trace row replays | analog `test_trace.py::test_a_trace_replays_from_its_json` | stays in MOGA-VM; the generic round trip ported, and the row now carries a trace (F-SS-024) |
| `test_search_strategy_integration.py` (6 others) | the harness end to end | - | stay in MOGA-VM |

Counts: 40 ported (21 domain and trace, 13 oracle, 3 extraction, 1
observer, 2 records; one of them a divergence), 112 stay in MOGA-VM, 5 of
which have a generic analog.

### Equivalence plan

The oracle is MOGA-VM `3d93ba3`'s `cir/lowering/search/{decisions,oracle,
errors,extraction}.py`, which import only `fhy_core` and, for
`extraction.py`, MOGA types that stand-ins replace (the audit's probes do
exactly this: `target/scratch/search-space-ss2/probes.py`).

- **Golden corpus:** `rust/fhy-core/tests/golden/record_trace_cases.py`
  writes `trace_cases.json`, replayed by
  `tests/it/search_space_trace_golden.rs` and
  `tests/search_space/test_trace_golden.py`; a recorder, not a generator,
  for SS1's reason (it needs a MOGA-VM checkout; CI only replays). It
  assembles the oracle as SS1's recorder does, with `git archive` of the
  four modules at `3d93ba3`.
- **Cases:**
  - domains: random choice domains over distinct plain values, order
    domains of 1 to 6 elements, strided domains of 1 to 5 runs with
    strides 1 to 64; recorded: cardinality, every `value_at`, every
    `coordinate_of`, `admits` on members, gaps, out-of-range values and
    Booleans;
  - traces: random step lists; recorded: `traversed_cardinality`,
    `of_kind`, `coordinates`;
  - replay: a recorded domain against an offered one (equal; another
    shape; another size; a moved run of one size; reordered plain
    choices; identifier choices rebuilt fresh) and streams longer and
    shorter than the point; recorded: refuse or the values replayed;
  - extraction: random one-level `ExtractedSearchSpace`s (option axes over
    1 to 4 options, each exposing 0 to 3 choice or order axes), recorded
    with `cardinality()`, and translated to a `Space` (a choice per option
    axis, a categorical or permutation variable per conditioned axis);
    `Space::cardinality` must be `Exact` of the same number.
- **Tagged divergences**, expected values written from the rule: F-SS-020
  (moved runs, reordered plain choices), F-SS-021 (shorter streams),
  F-SS-022 (mixed `1`/`True`/`1.0` and repeated choices), the empty order
  domain.
- **Not stream-equivalent:** draws (D-SS2-6). The per-step sampler is
  checked statistically instead: over the corpus's extraction cases,
  MOGA-VM's `draw_uniform` marginal frequency of each option and the
  port's `sample` with a `RandomOracle` agree within a chi-square bound at
  10 000 draws each (both are per-step uniform).
- **Not oracle-backed:** `sample_uniform`, `enumerate`, `cardinality` with
  conditions and forbidden clauses, `mutate`, deeper hierarchy: checked
  against the brute-force evaluator of SS1's properties.
- Harness honesty as SS1.8: flip one recorded verdict and see the replay
  name the case.

### Benchmark plan

`benchmarks/test_search_space.py` gains the rows below in SS2.1, skipping
until the API exists. "Before" is MOGA-VM `3d93ba3`'s modules, measured as
SS1.8's were (`target/scratch/search-space-ss2/bench_compare.py before`).

| Row | Before (MOGA-VM) | After |
|---|---|---|
| one random draw: choice of 8, strided of 4 runs × 2^16, order of 4 | `UniformRandomOracle.decide` | `RandomOracle` through a `Recorder` |
| record a 100-step dynamic stream | `RecordingOracle(UniformRandomOracle)` | `Recorder` with `RandomOracle` |
| replay a 100-step stream | `ReplayOracle` | `ReplayOracle` |
| `value_at` and `coordinate_of` over 64 strided runs | `AddressDomain` | `StridedDomain` |
| count 8 option axes × 4 options × 2 axes | `ExtractedSearchSpace.cardinality` | `Space::cardinality` |
| draw one point of that space | `draw_uniform` | `Space::sample` (`RandomOracle`) |
| (new) `sample_uniform`, `mutate`, `enumerate` 10 000 configurations, `cardinality` with a condition and a clause, a trace's JSON round trip, one step through a Python oracle | - | |

Expected cost: a static step re-checks the configuration (`with_entry`,
4.2 µs at SS1.8), and a Python oracle adds a Python call per step, where
MOGA-VM's dynamic steps were plain Python. The 10% rule of CONTRIBUTING
binds MOGA-VM's adoption, as for SS1.

### Symbol and visibility mapping (SS2)

| MOGA-VM (`cir.lowering.search`) | Rust (`fhy_core::search_space`) | Visibility | Public Python | Notes |
|---|---|---|---|---|
| `Coordinate` (`int \| tuple[int, ...]`) | `Coordinate` (enum) | pub | `int` or `tuple[int, ...]` | |
| `DecisionKind` (closed `StrEnum`) | `DecisionKind` (open string) | pub | `str` | MOGA keeps its five as constants |
| `ChoiceDomain` | `ChoiceDomain` | pub | `ChoiceDomain` | distinct, type-strict |
| `OrderDomain` | `OrderDomain` | pub | `OrderDomain` | non-empty |
| `AddressInterval` | `StridedRun` | pub | `StridedRun` | |
| `AddressDomain` | `StridedDomain` | pub | `StridedDomain` | |
| `DecisionDomain` (union) | `StepDomain` | pub | the three classes | |
| `Decision` | `PendingStep` | pub | `PendingStep` | built by the recorder |
| `Decision.describe` | `Display` of `PendingStep` | | `str(step)` | |
| `RecordedDecision`, `RecordedDecision.of` | `TraceStep`, `TraceStep::dynamic` | pub | `TraceStep` | holds a signature, not the domain |
| `SearchPoint` | `Trace` | pub | `Trace` | serializable |
| `SearchPoint.traversed_cardinality`, `of_kind`, `coordinates`, `from_records` | `traversed_cardinality`, `of_kind`, `coordinates`, `Trace::new` | pub | same | |
| `SearchPoint.extended_with` | (none) | - | - | the recorder appends |
| `SearchPoint.summarize` | `Display` of `Trace` | | `str(trace)` | |
| `SearchOracle` | `SearchOracle` (trait) | pub | `SearchOracle` (Protocol) | answers coordinates |
| `UniformRandomOracle` | `RandomOracle` | pub | `RandomOracle` | other stream (D-SS2-6) |
| `RecordingOracle` | `Recorder` | pub | `Recorder` | also drives a space |
| `ReplayOracle` | `ReplayOracle` | pub | `ReplayOracle` | `finish` |
| (none) | `ExhaustiveOracle`, `Rng`, `Cardinality`, `Enumeration` | pub | same | |
| `InadmissibleDecisionError` | `TraceError::Inadmissible`, `CoordinateOutOfDomain` | pub | `InadmissibleAnswerError` | |
| `SearchPointMismatchError` | `ReplayError` | pub | `ReplayMismatchError` | |
| `ExtractedSpaceMismatchError` | `TraceError::Unasked`; `ReplayError` | pub | `TraceError`, `ReplayMismatchError` | |
| `SearchSpaceError` (MOGA) | (the binding's `SearchSpaceError`) | | `SearchSpaceError` | SS1's base |
| `ExtractedSearchSpace` | `Space` | pub | `Space` | built by MOGA's extractor |
| `ExtractedSearchSpace.cardinality`, `draw_uniform`, `option_axes`, `conditioned_under`, `axis`, `axes_for` | `Space::cardinality`, `Space::sample`, `Space::choices`, `Choice::alternatives`, `Space::decision` | pub | same | |
| `SearchAxis`, `AxisKey`, `AxisCondition` | `Decision`, its name, hierarchy | pub (SS1) | | |
| `ExtractedPoint` | `Configuration` (and its `trace()`) | pub (SS1) | | |
| `ExtractedPointOracle` | `Recorder::realizing` | pub | `Recorder(oracle, configuration=...)` | strict |
| a domain's derivation for a static decision | `step::decision_domain` | `pub(super)` | - | |
| `_require_agreement` | `ReplayOracle::answer` | private | - | |
| `SearchSpaceExtractor`, `StructuralSearchSpaceExtractor`, the policies, the allocator, the harness, records, observers, sampling strategies | - | - | - | stay in MOGA-VM |

### Intended divergences (SS2)

| # | Behavior | MOGA-VM | After |
|---|---|---|---|
| D-SS2-1 | replay's domain check | shape and size (F-SS-020) | equal signatures |
| D-SS2-2 | a shorter stream | accepted (F-SS-021) | `finish` refuses it |
| D-SS2-3 | choice values | Python `==`, repeats allowed (F-SS-022) | type-strict, distinct |
| D-SS2-4 | repeated axis keys | first in, first out over all axes of a key (F-SS-026) | names unique; dynamic steps positional |
| D-SS2-5 | an oracle's answer | a value | a coordinate |
| D-SS2-6 | the random stream | `random.Random` | fhy-core's `Rng`; seeds re-baselined |
| D-SS2-7 | an empty order domain | cardinality 1 | refused |
| D-SS2-8 | uniform sampling | per axis only | per step (`sample`) and per configuration (`sample_uniform`) |
| D-SS2-9 | a point's persistence | none (F-SS-024) | `search_space.trace` |

### Changes to the design's sketch

- `TraceStep` holds a `DomainSignature`, not the domain's shape and
  cardinality alone, and an optional value; its kind is a string, not a
  `Canonical` tag (see "Extensibility decisions").
- `SearchOracle::decide` takes `&PendingStep` and answers a `Coordinate`,
  as sketched; `RecordingOracle` becomes the `Recorder`, which also
  drives a space.
- `Space::replay` reads static steps by canonical position, not
  positionally.
- `ExtractedSearchSpace` becomes a `Space`, but not the candidate table's
  own space: a tile variable's domain is the fitting set for the target
  (`policies.py:202-214`), so MOGA-VM's extractor builds the space per
  module and target.

### SS2.2: the stub, as built (2026-10-06)

The stub (`rust/fhy-core/src/search_space/{rng,domain,trace,oracle,recorder,exploration}.rs`,
the binding's `search_space/{rng,domain,trace,oracle,recorder}.rs`, the
Python classes and `_rs.pyi`) follows this plan, with these changes:

1. **`Recorder` holds only the run's state.** It has no lifetimes: each
   `decide` and `decide_dynamic` takes the oracle and the context, so a
   run can span calls that each borrow them anew, which the Python
   `Recorder` needs (its oracle is a Python object, leased per call).
   `Recorder::new()`, `over(&Space)` and `realizing(&Configuration)` build
   it; it is `Clone` and `Default`.
2. **`PendingStep` has two public constructors,** `dynamic(kind, subject,
   domain, position, context)` and `of_decision(kind, configuration,
   decision, domain, position, context)`, so an oracle can be asked
   outside a recorder: a downstream oracle's own tests, and the Python
   `decide` of the core's oracles on a step snapshot.
3. **No `wire::TraceData`.** A trace holds no foreign part, so `Trace`
   implements `Serialize` and `Deserialize` itself and the binding uses
   them; `DomainSignature` likewise.
4. **Domain API details:** `StepDomain::contains(&Coordinate)` and
   `DomainSignature::contains`, `admits` on every domain and run; a run's
   own refusals (`EmptyRun`, `ZeroStride`) carry no index;
   `ExhaustiveOracle::is_backtrack` takes a `&TraceError`;
   `TraceError::OtherSpace` refuses a configuration of another space to
   `mutate`.
5. **Python:** the new classes are `_rs` classes exported as they are,
   as `ConfigurationKey` is, except `Trace`, a public subclass registered
   under `search_space.trace`; `Cardinality` (a frozen dataclass) and
   `CardinalityKind` are Python classes that the public `Space.cardinality`
   builds from the private `_rs.Space._cardinality`; `TraceStep.signature`
   is the signature's V2 text; `Rng.below` takes any positive `int`.
6. **The example aggregate** gains `OracleRegistrar` beside
   `CountingOracle`, for the registry's refusals, as `KindRegistrar` does
   for the kinds.
7. `param::interval::effective_interval` and its `Interval` are
   `pub(crate)` (approved); the re-export from `param` lands with the code
   that calls it.

**Encapsulation checklist** (fhy-development-rs), on the stub:

| Check | Result |
|---|---|
| every `pub` item has a caller outside the crate | the binding and MOGA-VM call every one; `PendingStep`'s constructors serve downstream oracle tests and the binding |
| no `pub` fields | none: `Recorder`, `TraceStep`, `Trace`, the domains and oracles hold private fields |
| public enums are intended API | `Coordinate` and `Cardinality` (exhaustive: callers match them), `StepDomain` (`#[non_exhaustive]`), the error enums (`#[non_exhaustive]`); `DomainSignature` wraps a private representation |
| public traits | `SearchOracle`, one method, open by design; `Variable` gains a provided method |
| invariant-carrying types in leaf modules | `domain`, `trace`, `rng` and `recorder` are leaves |
| no `&mut` to internals, no `&Vec` | accessors return slices and references |
| `Default`, `From` | `Default` builds a valid empty `Recorder`, `Trace` and `ExhaustiveOracle`; `From` wraps a validated domain into `StepDomain`; `DecisionKind` reads a string through `TryFrom`, which refuses an empty one |
| one public path per item | the `pub use` list of `search_space` |
| no visibility widened for tests | none; the one widening (`effective_interval`) is for the implementation |

### SS2.3: the red tests, as written (2026-10-06)

| Where | Files | Tests |
|---|---|---|
| Rust, `tests/it/search_space/` | `rng_stories`, `domain_stories`, `trace_stories`, `recorder_stories`, `oracle_stories`, `exploration_stories`, `search_domain_stories`, `trace_serde_stories`, `stream_error_text_stories`, `trace_properties`; helpers in `support/search.rs` | 269: 231 failing, 38 passing |
| Rust, the oracle corpus | `tests/golden/record_trace_cases.py` writes `trace_cases.json` (95 domain, 51 replay, 5 stream and 40 extraction cases, 24 of them tagged D-SS2-1, -2, -3 or -7); `tests/it/search_space_trace_golden.rs` replays it | 6: 4 failing, 2 passing |
| Python, `tests/search_space/` | `test_trace.py`, `test_trace_rust_binding.py`, `test_oracle_extension.py`, `test_trace_properties.py`, and three tests added to `test_composed_search_space.py` | 230 failing, 22 passing |

- **Red for the right reason:** every failing test fails at a `todo!()`
  (in Python, `PanicException: not yet implemented`, in a subprocess for
  the composed tests), except the two clause-7 conformance tests, which
  fail because the implemented `check_variable_conformance` does not yet
  check clause 7.
- **Green against the stub, by design:** the error texts (decided in the
  stub), the property and strategy guards, the two conformance controls
  (a faithful and a derived search domain conform), the corpus's own
  checks (every untagged case agrees with the oracle; every family and
  tag is present), and in Python the class-structure tests, the
  Python-only `Cardinality` and `CardinalityKind`, the hook's default,
  and the oracle registry's refusals (real plumbing).
- **Statistical tests** use fixed seeds and a 0.001 chi-square critical
  value: `below(6)` over 60 000 draws (5 degrees of freedom, 20.52),
  `sample_uniform` over the five configurations of a space with a
  condition and a forbidden clause (4, 18.47), and per-step sampling
  against its per-step probabilities (4, 18.47).
- **Golden vectors** of the generator come from an independent reference
  of the documented algorithms; seed 0's first number is SplitMix64's
  published `0xe220a8397b1dcdaf`.
- **Written later in phase 2 (coordinator's addition):** the corpus's
  Python replay, `tests/search_space/test_trace_golden.py`, checking every
  divergence tag both ways, with a control that changes one expected
  answer per family and sees the replay name the case; and `Coordinate`'s
  serde, now externally tagged so postcard (not self-describing) reads it,
  with postcard and JSON round trips of `Coordinate`, `TraceStep` and
  `Trace`.
- **Small API changes the tests settled:** `Trace::of_kind` borrows its
  kind only for the call; `TraceError::Inadmissible`'s text no longer
  writes the coordinate; `Rng`'s refusal of another algorithm names it;
  `below_big`'s panic message is pinned; an oracle error that is itself a
  `TraceError` stops a run as that error.

### SS2.4 to SS2.7: as built (2026-10-06)

- **Core** (`680b268`, then `6a0ca1a` and `b1e4478`): as planned, in the
  plan's order. Two private modules beyond the plan's tree: `step` (a
  static step's kind and domain, growing a configuration by one value,
  the coordinate walk the oracles share) and `counting` (components,
  closed forms, bounded enumeration, the relaxed counts of
  `sample_uniform`). `ExhaustiveOracle` keeps only each position's
  coordinate and domain signature, and finds the next coordinate from the
  signature, so it holds no value of a domain.
- **Binding** (`140f1ad`): every class of the stub; an oracle argument is
  read at each step (a core oracle class, a registered lease, then any
  object with a callable `decide`), so `Recorder(None)` builds and its
  first step raises `TypeError`. `ReplayOracle.finish` raises
  `ReplayMismatchError` and leaves the replay open when a recorded step
  was never asked; after a successful `finish`, `decide` and `finish`
  raise `RuntimeError`. A core oracle class locked by a run refuses to be
  asked again from inside it (`RuntimeError`), as does a busy `Recorder`.
  `Space.enumerate` returns `_rs.SpaceEnumeration`, an iterator that runs
  one exhaustive pass per configuration.
- **GC:** `ChoiceDomain` and `OrderDomain` own the references their
  opaque values keep; a `Recorder` owns those of the steps it recorded;
  a `TraceStep` recorded in this process re-reads its domain's objects
  when its value holds an opaque one, so it owns its value's references.
  `Rng`, `StridedRun` and `ExhaustiveOracle` hold no Python object and
  are exempt in `tests/test_gc_cycles.py`, with reasons;
  `test_an_exhaustive_oracle_keeps_no_value_of_a_domain_alive` pins the
  last.
- **Test corrections** (`5f42dde`, accepted by the coordinator): the
  mutation property asserts `mutate`'s documented refusal exactly when
  the space has one complete configuration (the empty space included),
  in Python and in Rust; the GC exemptions above.

**Encapsulation checklist**, on the implementation:

| Check | Result |
|---|---|
| public items | exactly the stub's: no item was added or widened; the one crate widening stays `effective_interval` |
| fields | none public; `Recorder`, `TraceStep`, `Trace`, the domains, the oracles and `Enumeration` keep theirs private |
| new modules | `step` and `counting`, private, their helpers `pub(super)` |
| leaves | `rng`, `domain`, `trace` and `recorder` hold the invariants and import no sibling but `error` and `domain` |
| panics | `Rng::below_big` asserts a positive bound (documented); no `unwrap` or `expect` in library code; unreachable conversions fall back with `unwrap_or` |
| casts | none: the 128-bit product and the limbs go through bytes |
| binding | new helpers are `pub(super)` within `search_space`; crate-wide additions are `PyRng`'s `seeded`, `from_seed_object`, `seed`, `snapshot`, `restore` and `with_rng`, and `convert::param::run_attached_with_context` (a context question that holds the interpreter, for a Python oracle) |

## SS2.8: equivalence runs, benchmarks and the divergence log

### Equivalence runs (2026-10-06)

| Corpus | Cases | Rust core (`search_space_trace_golden.rs`) | Python binding (`test_trace_golden.py`) |
|---|---|---|---|
| committed (`--seed 0 --random-count 40`) | 191 (95 domain, 51 replay, 5 stream, 40 extraction) | all replay | all replay |
| expanded (`--seed 7 --random-count 2000`, in `target/` only) | 8031 (4015, 2011, 5, 2000) | all replay (the corpus swapped in for one run) | all replay (`_find_mismatches` and `_find_mistagged` on it) |

- **Recorder check:** re-recording the committed corpus after the
  recorder's typing changes gives the committed file but for the
  provenance commit.
- **Harness honesty:** changing one replay case's recorded outcome fails
  both Rust replays naming `same_integers`; the Python suite's control
  test does the same for one case per family.
- **Draws** (not stream-equivalent, D-SS2-6): over the 40 committed
  extraction cases, 10 000 draws each, MOGA-VM's `draw_uniform` and the
  port's `Space.sample` with a `RandomOracle` choose each option with
  frequencies a two-sample chi-square test does not tell apart: 58
  entries with two or more options, none above the 0.001 critical value,
  the largest statistic 0.59 of it
  (`target/scratch/search-space-ss2/eq/draws_{moga,port}.py`).

### Divergence log

Counts over the committed (expanded) corpus; every other case agrees
with the oracle, in both languages.

| Divergence | Cases | What the port does instead |
|---|---|---|
| D-SS2-1 (replay compares signatures) | 18 (992) | refuses a replay over a moved run of one width or reordered plain choices, which MOGA-VM accepted by shape and size |
| D-SS2-2 (`finish` refuses a shorter stream) | 2 (2) | names the first recorded step never asked |
| D-SS2-3 (type-strict, distinct choices) | 3 (3) | tells `1`, `True` and `1.0` apart and refuses a repeated choice |
| D-SS2-7 (empty order domain) | 1 (1) | refuses it, where MOGA-VM counted one ordering |
| none | 167 (7033) | answers as the oracle does |

D-SS2-4, -5, -6, -8 and -9 are API changes no corpus case can exercise:
positional dynamic steps, coordinates as answers, the stream (checked
statistically above), `sample_uniform`, and the trace's type id.

### Benchmarks (CPython 3.11, this machine, back to back)

"Before" is MOGA-VM `3d93ba3`'s modules on fhy_core v0.1.8, assembled as
the corpus recorder does (`target/scratch/search-space-ss2/bench_compare.py
before`); "after" is the same rows through `fhy_core.search_space`, after
the two `perf(search_space)` commits (`4889e98`, `a369668`). The machine
drifts by up to 15% between runs, so the table is one back-to-back pair;
`benchmarks/test_search_space.py` under pytest-benchmark gives the new
rows. Median per call, µs.

| Row | Before (MOGA-VM) | After (port) | Ratio |
|---|---|---|---|
| **one draw, choice of 8** | 0.59 | 0.88 | **1.49, slower** |
| one draw, strided, 4 runs × 2^16 | 3.11 | 0.86 | 0.28 |
| one draw, order of 4 | 2.71 | 1.39 | 0.51 |
| record a 100-step stream | 547 | 152 | 0.28 |
| replay a 100-step stream | 226 | 117 | 0.52 |
| `value_at` and `coordinate_of`, 64 strided runs | 21.9 | 0.42 | 0.019 |
| count 8 option axes × 4 options × 2 axes | 230 | 34.3 | 0.15 |
| **draw one point of that space** | 30.7 | 64.0 | **2.09, slower** |
| (new) `sample_uniform`, that space | - | 131 | |
| (new) `mutate`, that space | - | 4 140 | |
| (new) enumerate 10 000 configurations | - | 84 400 | |
| (new) count with a condition and a clause | - | 20 400 | |
| (new) trace JSON round trip, 100 steps | - | 789 | |
| (new) one step through a Python oracle | - | 1.13 | |

**The performance pass** (before it, the two rows were 1.93 and 6.6
times slower):

- `4889e98`: a static step's configuration was checked twice per step
  (when the oracle asked whether its answer is admissible, and when the
  recorder grew the configuration), and each check ran every assigned
  value's param check again. The step now hands the recorder the
  configuration its admitted answer grew to, and a run grows its
  configuration checking only the new value against its param. A
  dynamic step's `admits` no longer builds the value. Draw one point:
  205 to 64 µs. `space_sample_with_a_random_oracle_follows_the_pinned_stream`
  (new, `a775487`, written first; green before and after, as a
  behavior-preserving change must be) pins the coordinates and the
  generator's next number for seeds 0 to 11; the corpus replays, the
  Rng golden vectors and the `RandomOracle` stream stories
  (`oracle_stories.rs`, `test_random_oracle_draws_*`) pin the dynamic
  stream.
- `a369668`: a signature copied each member of its domain, so each
  recorded step allocated its domain's values again; a plain domain's
  signature now shares them. With the dynamic `admits`, the core's
  dynamic step went from 0.60 to 0.21 µs, and one draw from 1.07 to
  0.88 µs.

**The two rows that stay slower**, for the user's decision (as B-SS1
was):

- **One choice draw (1.49x):** the core's part is 0.21 µs; the rest is
  the binding: each step builds the param context (the default solver,
  the registry snapshot, the observer's backend name; about 0.2 µs, as
  `PendingStep.admits` against `ChoiceDomain.admits` shows), reads the
  subject `Identifier` from Python and keeps the step's objects for its
  trace. MOGA-VM's `UniformRandomOracle.decide` records nothing; recording
  a stream, the row that compares like with like, is 3.6 times faster in
  the port.
- **One point of the option space (2.09x):** each of its 24 static steps
  still re-derives the activity of every decision (72 here), the
  conditions and the forbidden clauses when the configuration grows, so a
  point costs quadratically in its decisions; `draw_uniform` checks
  nothing. An incremental check (re-deriving only the dependents of the
  new decision) would remove most of it, but rewrites the configuration
  checker, so it is not done here. `sample_uniform` (131 µs) draws the
  same space uniformly per configuration.
- `mutate` (4.1 ms) searches the space's completions per candidate value
  (1024 runs at most each); counting with a condition and a clause
  enumerates its component (2048 configurations). Neither has a MOGA-VM
  counterpart.

## SS2.9: the MOGA-VM migration note

The map under "MOGA-VM migration map (SS2 and SS3)" holds as written for
SS2, with these changes from the implementation:

- `Recorder(oracle, ...)` reads the oracle at each step: a MOGA-VM
  oracle class needs only a callable `decide` answering a coordinate (an
  `int`, or a tuple of `int`s for an order); `step.draw_uniform(rng)`
  stays on an `Rng`'s stream.
- `ReplayOracle.finish()` leaves the replay open when it refuses, so a
  harness may report the unconsumed step and keep the oracle.
- `Space.enumerate()` is an iterator (`_rs.SpaceEnumeration`); a
  `sampling.py` strategy that enumerates small spaces iterates it.
- `sampling.py`'s draws through `space.sample` pay the static steps'
  checks (the 2.09x row above): CONTRIBUTING's 10% rule binds that
  adoption, as it did SS1's.
- Seeds are re-baselined (D-SS2-6); `Rng` pickles mid-stream, so a
  harness that checkpoints its generator keeps its stream.

## SS3: objectives and measurements (plan)

The concrete plan for SS3, in the same template.

### Summary

`Objective`, `Direction`, `Measurement` and `Measurer`: the vocabulary of
what a search measures and which way is better, and the record of one
measured configuration. `Metric` left the space in SS1 (N-C3); this is
where its kind and its name go.

### What MOGA-VM needs

- **MOGA is the target-machine model, not a multi-objective search.** The
  MOGA of MOGA-VM is "the description of the target machine"
  (`docs/design/moga_vm_design.md:8-9, 22-24`), and the search harness is
  single-objective: an `Objective` is a callable returning one `float`,
  lower better (`harness.py:119-124`), a record holds one `score`
  (`records.py:224`), and `best` is the minimum (`records.py:337-353`). No
  source mentions Pareto dominance.
- What it needs from fhy-core: a direction, so a benefit is no longer
  negated by the caller (`harness.py:122-123`); a record of a measured
  configuration that is keyed by `ConfigurationKey`, refuses a NaN
  (F-SS-023), and distinguishes an infeasible configuration (data about
  the space) from a failure of the measurement; and a trait for what
  measures.

### Module tree and visibility

| Path | Visibility | Contents |
|---|---|---|
| `search_space::measurement` | private, leaf | `Direction`, `Objective`, `MeasurementStatus`, `Measurement`, `Measurer` |
| `search_space::error` | private | adds `MeasurementError` |
| `search_space::wire` | `pub mod` | adds `MeasurementData`, `ConfigurationKeyData` reused |

New public paths: `search_space::{Direction, Objective,
MeasurementStatus, Measurement, Measurer, MeasurementError}` and
`wire::MeasurementData`.

### Public API

```rust
pub enum Direction { Minimize, Maximize, Report }        // Debug, Clone, Copy, PartialEq, Eq, Hash, Display, serde; exhaustive
pub struct Objective { /* name: Arc<str>, direction */ }  // Debug, Clone, PartialEq, Eq, Hash, Display, serde
impl Objective {
    pub fn new(name: &str, direction: Direction) -> Result<Self, MeasurementError>;
    pub fn name(&self) -> &str;
    pub fn direction(&self) -> Direction;
    pub fn compare(&self, left: f64, right: f64) -> Option<Ordering>;   // Greater: left is better; None for Report
}
#[non_exhaustive]
pub enum MeasurementStatus { Ok, Infeasible { reason: String }, Failed { reason: String }, Timeout }   // Debug, Clone, PartialEq, Eq, Hash, serde
pub struct Measurement(Arc<..>);                         // Debug, Clone, PartialEq, Eq, Hash, serde
impl Measurement {
    pub fn ok(key: ConfigurationKey, values: Vec<(Objective, f64)>) -> Result<Self, MeasurementError>;
    pub fn infeasible(key: ConfigurationKey, reason: String) -> Self;
    pub fn failed(key: ConfigurationKey, reason: String) -> Self;
    pub fn timeout(key: ConfigurationKey) -> Self;
    pub fn with_notes(self, notes: Vec<Note>) -> Self;
    pub fn key(&self) -> &ConfigurationKey;
    pub fn status(&self) -> &MeasurementStatus;
    pub fn is_ok(&self) -> bool;
    pub fn values(&self) -> &[(Objective, f64)];          // in the order given
    pub fn value(&self, objective: &str) -> Option<f64>;
    pub fn notes(&self) -> &[Note];
    pub fn dominates(&self, other: &Self) -> Result<bool, MeasurementError>;   // N-S4
}
pub trait Measurer<S: ?Sized> {
    fn objectives(&self) -> &[Objective];
    fn measure(&mut self, key: &ConfigurationKey, subject: &S) -> Result<Measurement, BoxError>;
}
#[non_exhaustive] pub enum MeasurementError {
    EmptyName, NoValues, RepeatedObjective { name: String }, NonFiniteValue { objective: String },
    NotOk, DifferentObjectives,
}
```

### Decisions in this plan

- **A measurement records** its configuration's key, its status, one
  finite value per objective when the status is `Ok` and none otherwise
  (`records.py:111-168`'s rule), and notes.
- **Statuses:** `Ok`; `Infeasible`, the configuration cannot be realized
  (MOGA-VM's pipeline, validation and acceptance rejections, with the
  stage in the reason); `Failed`, the measurement was attempted and broke
  (a crashed simulator); `Timeout`. The brief named ok, failed and timeout;
  `Infeasible` is added because MOGA-VM's harness keeps exactly that
  distinction: a rejection is data about the space, a failure is not
  (`harness.py:26-37`).
- **Values are `f64`**, finite: NaN and the infinities are
  `NonFiniteValue`, so `min` and dominance are total and the JSON is
  standard (F-SS-023). Integers such as cycles and bytes are exact up to
  2^53.
- **An objective's name is a string**, for the reason a decision kind is:
  measurements are keyed by `ConfigurationKey`, which pickles across
  processes, and an `Identifier` name would not match across them. It is
  a shared vocabulary, the role the audit gave metric names (F-SS-012).
  Two objectives are equal when their names and directions are.
- **`Direction`:** `MetricKind.COST` is `Minimize`, `BENEFIT` is
  `Maximize`, `DIAGNOSTIC` is `Report` (recorded, never compared).
- **`Measurer<S>`** is the extension trait, generic over what it measures
  (MOGA-VM measures a `LoweredProgram` and its trace, `harness.py:119`;
  another caller a configuration). An `Err` is a defect of the measurer
  and propagates; a subject that cannot be measured is an
  `Ok(Measurement)` with a failing status (MOGA-VM's doctrine,
  `harness.py:26-37`). Not object-safe across subjects by design; `dyn
  Measurer<S>` is object-safe for one `S`.
- **`Estimate`** (the old declared-estimate role of `Metric`) is not in
  SS3 (N-S3): nothing in MOGA-VM produces or reads an estimate
  (`decisions.py:33-37`, audit F-SS-012), and when one appears it is a
  `Measurer<Configuration>` over an expression, needing no new type.
- **Multi-objective comparison** is `dominates` alone (N-S4): both
  measurements `Ok` (`NotOk`), over the same objectives by name and
  direction (`DifferentObjectives`); `Report` objectives ignored; at least
  as good on every compared objective and better on one. No Pareto
  archive, crowding or ranking: MOGA-VM needs none, and a search that
  does builds them on `dominates`.

### Error model

`MeasurementError` from the constructors (stop at the first problem: an
empty name, no value for an `Ok`, a repeated objective, a non-finite
value) and from `dominates`. Nothing panics.

### Serialization

Type ids `search_space.objective` and `search_space.measurement`:

```json
{"key": {"entries": [..]}, "status": {"ok": null},
 "values": [{"objective": {"name": "latency_cycles", "direction": "minimize"}, "value": 1532.0}],
 "notes": []}
```

A failing status is `{"infeasible": {"reason": ".."}}`, `{"failed":
{"reason": ".."}}` or `{"timeout": null}`, with no values. The key is
`wire::ConfigurationKeyData`'s shape (SS1.5's deviation 7). A measurement
means nothing without its space, as its key does not; a cache keeps the
space beside it.

### Python interface

| Class | Construction | Attributes and methods |
|---|---|---|
| `Direction` | `StrEnum`: `MINIMIZE = "minimize"`, `MAXIMIZE = "maximize"`, `REPORT = "report"` | |
| `Objective` | `Objective(name, direction)` | `name`, `direction`, `compare(left, right)` (`1`, `0`, `-1` or `None`); structural `==` and `hash` (N-S2); serialization methods; pickles |
| `MeasurementStatus` | `StrEnum`: `OK`, `INFEASIBLE`, `FAILED`, `TIMEOUT` | |
| `Measurement` | class methods `ok(key, values)` (a mapping of `Objective` to `float`, or pairs), `infeasible(key, reason)`, `failed(key, reason)`, `timeout(key)` | `key`, `status`, `reason` (or `None`), `is_ok`, `values` (a `dict` in the order given), `value(objective)` (an `Objective` or a name), `notes`, `with_notes(notes)`, `dominates(other)`; identity `==`; serialization methods; pickles |
| `Measurer` | a `typing.Protocol`, generic in the subject: `objectives: Sequence[Objective]`, `measure(key, subject) -> Measurement` | |

- Errors: `MeasurementError(SearchSpaceError)`; a `bool` value is refused
  (`TypeError`), as is a non-number.
- **Extension paths.** A Python measurer is any object of the protocol
  (path 1). No fhy-core entry point calls a measurer in SS3, so path 2
  (a registered Rust measurer kind) has no reader yet; it is added, as
  the oracle family is, with the first fhy-core function that takes a
  measurer.

### Test plan (SS3)

- Rust `measurement_stories.rs`: constructors and every
  `MeasurementError`; values kept in order; `value`; `with_notes`;
  `==`/`Hash`; serde shapes pinned and round trips; `dominates` over
  minimize, maximize and report objectives, ties, mixed objective sets,
  non-ok measurements. `objective_stories.rs`: `compare` per direction.
  A property: `dominates` is irreflexive, asymmetric and transitive.
- Python `tests/search_space/test_measurement.py`: the interface,
  type-strict values, pickling, the V2 shapes, and the two ported
  MOGA-VM record tests (traceability above).
- Equivalence: no MOGA-VM behavior is ported beyond the record rule and
  "lower is better", which the stories pin; no corpus.
- Benchmarks: building and serializing a measurement of 4 objectives (new
  rows; no before).

### Symbol and visibility mapping (SS3)

| MOGA-VM | Rust | Visibility | Public Python |
|---|---|---|---|
| `MetricKind` (`cir.space.core`, gone in SS1) | `Direction` | pub | `Direction` |
| `Metric.name` | `Objective` | pub | `Objective` |
| `Metric.value` (a declared expression) | (none; N-S3) | - | - |
| `Objective` (`harness.py:119`, a callable) | `Measurer<S>` | pub | `Measurer` |
| `CandidateRecord.score` | `Measurement::values` | pub | `Measurement.values` |
| `CandidateOutcome.LOWERED`, `REJECTED_BY_*` | `MeasurementStatus::Ok`, `Infeasible` (stage in the reason) | pub | `MeasurementStatus` |
| `SearchHistory.best`'s ordering | `Objective::compare` | pub | `Objective.compare` |
| `CandidateRecord`, `CandidateTimings`, `SearchHistory`, `SearchStatistics` | - | - | stay in MOGA-VM, holding a `Measurement` |

### SS3.1: the stub, as built (2026-10-06)

The stub (`rust/fhy-core/src/search_space/measurement.rs`, the
`MeasurementError` in `error.rs`, `wire::MeasurementData`, the binding's
`search_space/measurement.rs`, the Python classes and `_rs.pyi`) follows
the plan, with these refinements:

1. **NaN (F-SS-023).** A measurement refuses a NaN or an infinity
   (`NonFiniteValue`), and `Objective::compare`, which takes any two
   `f64`s, ranks a NaN worse than every number in either direction and
   equal to another NaN: it never wins a comparison, and a number after it
   is better, so a running best never sticks on it.
2. **`-0.0` is kept as `0.0`**, so `==` and `Hash` (over the values'
   bits) agree.
3. **`dominates` compares objective sets, not orders:** the two
   measurements must hold the same objectives by name and direction, in
   any order; `Report` objectives are matched but not compared, so two
   measurements of only `Report` objectives never dominate.
4. **The status's serde** is serde's external tagging: `"ok"`,
   `"timeout"`, `{"infeasible": {"reason"}}`, `{"failed": {"reason"}}`
   (the plan's sketch wrote `{"ok": null}`).
5. **`wire::MeasurementData`** has `of` and `build`, as the key's own
   data does, so a key holding another crate's opaque value is read with
   a resolver; `Measurement`'s own `serde` builds with `NoForeign`.
6. **Constructors:** `infeasible` and `failed` take `impl Into<String>`.
   A payload of a measurement that did not succeed holding values is
   refused with a seventh error, `UnexpectedValues` (added with the
   tests).
7. **Python:** `Objective(name, direction)` takes a `Direction` or its
   value; `Measurement` has no constructor (`TypeError`): `ok`,
   `infeasible`, `failed`, `timeout` build it; `is_ok()` is a method, as
   `Configuration.is_complete()` is; `reason` is `None` for `OK` and
   `TIMEOUT`; `value(objective)` matches an `Objective` by name and
   direction and a `str` by name; `values` is a new `dict` each time.
   `Objective` and `Measurement` are public subclasses
   registered as `search_space.objective` and `search_space.measurement`,
   as `Trace` is.

**Encapsulation checklist**, on the stub:

| Check | Result |
|---|---|
| every `pub` item has a caller outside the crate | the binding and MOGA-VM's harness |
| fields | none public: `Objective` (name, direction), `Measurement` (an `Arc` of private fields) |
| public enums | `Direction` exhaustive (three directions by nature, matched by callers); `MeasurementStatus` and `MeasurementError` `#[non_exhaustive]` |
| public trait | `Measurer<S: ?Sized>`, open by design; object-safe for one `S` |
| leaf module | `measurement` imports only `configuration`'s key, `error` and `diagnostic` |
| one public path per item | the `pub use` list of `search_space` and `wire::MeasurementData` |
| panics | none planned |
| binding | `PyObjective` and `PyMeasurement` `pub(crate)` for the module registration; their helpers `pub(super)` |

### SS3.2: the red tests, as written (2026-10-06)

| Where | Files | Tests |
|---|---|---|
| Rust, `tests/it/search_space/` | `objective_stories`, `measurement_stories`, `measurement_serde_stories`, `measurement_properties`; builders in `support/measurement.rs` | 112: 96 failing, 16 passing |
| Python, `tests/search_space/` | `test_measurement.py`, `test_measurement_rust_binding.py`, `test_measurement_properties.py`; `Objective` and `Measurement` added to `tests/test_gc_cycles.py`'s exemptions | 150: 140 failing, 10 passing |

- **Red for the right reason:** every failing test fails at a `todo!()`
  (in Python, `PanicException: not yet implemented`; the module-level
  objectives are built lazily so each test fails on its own).
- **Green against the stub, by design:** the `Direction` names and texts
  (derived), the error texts (decided in the stub), payloads of another
  shape (refused by the derived wire form), the Python enums, the
  exception's base and the class structure.
- **F-SS-023:** a NaN is refused by every constructor and every reader
  (JSON cannot carry one; a forged postcard payload, with a control that
  decodes the same payload with a finite value, can); `compare` ranks a
  NaN last in both directions, and a running best that meets a NaN first
  still reaches the best number, in Rust and in Python.
- **Properties** (Rust proptest, 256 cases; Python hypothesis, 200):
  `dominates` is irreflexive, asymmetric and transitive; reversing every
  direction reverses it; over one compared objective it is `<` or `>`;
  it ignores reported values and the order of the values; measurements
  round-trip through JSON and postcard. Values come from `{0, 1, 2, 3}`,
  so ties are common, and a guard checks a fixed sample holds both
  dominating and non-dominating comparable pairs.
- **Ported:** `test_records.py`'s two record tests and the comparison of
  its `best` test, named in their docstrings.
- **Fake-test pass:** one test passed whatever the value's sign (a
  `-0.0 or 0.0` that is falsy); fixed before the commit.
- **Persona pass:** added that `value` matches an `Objective` by name and
  direction (a `str` by name), that `values` is a copy, the extreme finite
  values kept bit for bit, a repeated objective reported before a NaN, a
  failed measurement's empty reason round-tripping, a non-ASCII name, and
  a key with an opaque value that cannot be written.

### MOGA-VM migration map (SS2 and SS3)

None of this is done by the port; it is what MOGA-VM changes to use it.

| Site | Today | After |
|---|---|---|
| `decisions.py` | domains, `Decision`, `RecordedDecision`, `SearchPoint`, `DecisionKind` | deleted; `fhy_core.search_space`'s domains, `Trace`; MOGA's kinds become string constants (`moga.cir.address`, `moga.cir.boundary_namespace`), and the static ones become the space's (`search_space.choice` for options, the knob kinds for tiles and walk orders) |
| `oracle.py` | three oracles | deleted; `RandomOracle`, `ReplayOracle`, `Recorder` |
| `errors.py` | `SearchSpaceError` family | `fhy_core.search_space`'s `TraceError` family; MOGA's harness lets it propagate as today |
| `policies.py:64-114` (`RandomMapOperationsPolicy`) | builds a `ChoiceDomain` of options, asks the oracle | `recorder.decide(entry.vertex_id)` over the function's space; returns the `RealizationOption` |
| `policies.py:117-153` (walk order) | `OrderDomain` of level indices | `recorder.decide(walk_order_variable.name)` if the space declares it, else `recorder.decide_dynamic("moga.cir.walk_order", levels[0], OrderDomain(levels))` |
| `policies.py:156-214` (tile) | `ChoiceDomain(fitting)` | `recorder.decide(tile_variable.name)` (the space's tile variable holds the fitting set) |
| `placement.py:250-323` (address) | `AddressDomain(intervals)` | `recorder.decide_dynamic("moga.cir.address", value.symbol_name, StridedDomain(runs))`; `_append_run` builds `StridedRun`s |
| `placement.py:603-623` (boundary namespace) | `ChoiceDomain(pool)` | `decide_dynamic("moga.cir.boundary_namespace", vertex, ChoiceDomain(pool))` |
| `extraction.py` | `ExtractedSearchSpace` and its oracle | `StructuralSearchSpaceExtractor.extract` returns a `Space` (a choice per uncommitted entry named by its vertex id; per option a permutation variable per walk and a categorical variable of fitting tiles); `ExtractedPointOracle` becomes `Recorder(oracle, configuration=point)` |
| `sampling.py` | `draw_uniform(rng)` with `random.Random` | `space.sample(RandomOracle(rng=rng))` or `space.sample_uniform(rng)`; the strategies keep their shape; `strict` is always on (`finish`) |
| `harness.py:568-571` | `RecordingOracle(oracle)` around each draw | a `Recorder` over the draw's space; `CandidateRecord.point` becomes a `Trace` |
| `harness.py:119-124`, `643-655` | `Objective` returns a float, lower better | a `Measurer[LoweredProgram]` with an `Objective`; `score` becomes a `Measurement`; `was_new_best` through `Objective.compare`; NaN refused |
| `records.py` | `CandidateRecord`, statistics | keep; `point: Trace`, `score` from the measurement; `rejection_depth` is `len(trace)`, `last_decided_kind` the last step's kind |
| `observers.py:125-160` | coordinates, kinds, cardinalities | add `"trace": trace.serialize_to_dict()`, so a row replays through `ReplayOracle(Trace.deserialize_from_dict(row["trace"]))` |
| seeds | `random.Random(seed)` | `Rng(seed)`; seed-pinned expectations re-recorded |
| tests | `tests/cir/lowering/search/` | the 40 ported tests are deleted with their modules; the 112 stay and move to the new API |

### Implementation checklist (SS2, SS3)

Every step ends with the S16 gate, as SS1's did.

1. **SS2.0:** the user's choices under "Needs the user (SS2/SS3)" (decided
   2026-10-06); N-S1
   blocks SS2.2.
2. **SS2.1:** the benchmark rows; the before numbers.
3. **SS2.2:** the Rust stub: `rng`, `domain`, `trace`, `oracle`,
   `recorder`, `exploration`, the errors, `wire::TraceData`,
   `Variable::search_domain`, `ContractClause::SearchDomain`; the
   `effective_interval` widening; rustdoc with the stream contract.
4. **SS2.3:** the Rust tests, the golden recorder and corpus; red.
5. **SS2.4:** the core, in order:
   1. `Rng` and its golden vectors;
   2. domains, signatures, coordinates;
   3. `TraceStep`, `Trace`, serde;
   4. `PendingStep`, admissibility, `draw_uniform`, `Recorder`;
   5. `RandomOracle`, `ReplayOracle`, `ExhaustiveOracle`;
   6. `Space::sample`, `Space::replay`, `Configuration::trace`;
   7. `enumerate` and `cardinality`;
   8. `sample_uniform`;
   9. `mutate`.
6. **SS2.5:** the binding stub, `convert::search_space`'s oracle and
   domain functions, `_rs.pyi`; the Python suites; red.
7. **SS2.6:** the binding: the pyclasses, `PythonOracle`, the
   `extension_search_domain` adapter, the oracle family of the registry,
   `CountingOracle` in the example aggregate, errors, wire, GC.
8. **SS2.7:** `fhy_core.search_space`'s new classes; the ported Python
   tests.
9. **SS2.8:** equivalence runs, the benchmarks after, the divergence log
   (done, "SS2.8" below the SS2 plan).
10. **SS2.9:** the MOGA-VM migration note (the map above; done, "SS2.9").
11. **SS3.1:** (stub done, "SS3.1" above) the stub (`measurement`, `MeasurementError`,
    `wire::MeasurementData`) and its tests; red.
12. **SS3.2:** the core.
13. **SS3.3:** the binding and `fhy_core.search_space`'s classes; the
    Python tests.
14. **SS3.4:** benchmarks; the migration note's SS3 rows.

## Serialization and type ids

- **Type ids (N-6 translated):** `search_space.variable`,
  `search_space.alternative`, `search_space.choice`, `search_space.space`,
  `search_space.configuration`; later `search_space.trace`,
  `search_space.objective`, `search_space.measurement`. Conditions and
  forbidden clauses are nested shapes without ids.
- **Families:** `Variable` and `Alternative` are `WrappedFamilySerializable`
  roots. An implementor registers its own id with `register_serializable`
  (Python) or answers it from `kind()`/`to_foreign` (Rust). Decoding
  dispatches by type id through the registry or the resolver; the binding's
  resolver tries registered Rust kinds, then the Python registry.
- **Shapes (V2, the core's serde shape):**

  | Type | Shape |
  |---|---|
  | `Variable` (plain) | `{identifier, param, notes}`, the old `Knob` keys |
  | `Alternative` (plain) | `{identifier, variables, choices, notes}` |
  | `Choice` | `{identifier, alternatives, notes}` |
  | `Space` | `{identifier, variables, choices, conditions: [{target, when}], forbidden: [{when}], notes}` |
  | `Configuration` | `{space, entries: [{name, value}]}` in canonical order |

  A part inside a container is externally tagged: `{"plain": {...}}` or
  `{"foreign": {"type_id": ..., "data": ...}}`, as `param::wire` writes a
  custom constraint.
- **V1 and pickles:** V1 is read until 0.3.0; pickles are a call of the
  class with its fields.
- **Old `moga.cir.*` payloads:**
  - Knob payloads keep their keys, so MOGA-VM's knob kinds, re-registered
    under their own ids, still decode as `Variable` implementors.
  - Decision-point, decision-space, selection and table payloads change
    shape (the selection leaves the space), so `alias=True` cannot read
    them.
  - MOGA-VM's repository stores none (`git grep moga.cir.` finds only
    source files), and the search harness's snapshot is written and read in
    one process (`cir/lowering/search/harness.py:697-698`). **No migration
    tool is needed.** If a stored payload appears, a MOGA-VM function that
    reads the old table with the 0.1.x classes and writes a `Space` plus a
    `Configuration` converts it, before 0.3.0 removes V1.

## The Python interface (N-4 (a): both paths)

**Classes:**
- Subclassable: `Variable` and `Alternative`, `#[pyclass(subclass,
  frozen)]` bases whose `#[new]` accepts `*args, **kwargs` (D-S11-11), with
  thin public classes `Variable(_rs.Variable, WrappedFamilySerializable,
  Generic[_T])` and `Alternative(_rs.Alternative, WrappedFamilySerializable)`.
- Frozen, not subclassable: `Choice`, `Space`, `Condition`, `Forbidden`,
  `Configuration`, `ConfigurationKey`.
- Errors: `SearchSpaceError(ValueError)` with `DuplicateNameError` and
  `ConfigurationError` (carrying the collected problems).
- `Direction` is a P1 `StrEnum` (SS3).

**Path 1, Python subclasses.**
- A subclass has its own `__init__` and attributes, calls `super().__init__`,
  and freezes after `__init__`.
- When a container is built, the binding wraps the subclass instance in an
  adapter (`PyVariable`, `PyAlternative`) that implements the trait.
  - It reads the base fields from the pyclass, without a Python call.
  - Its kind is the class's registered type id.
  - Its hooks call `extension_is_structurally_equivalent(self, other)`,
    `extension_is_alpha_equivalent_under(self, other, renaming)` and
    `extension_bound_identifiers(self)`; the defaults are `True`, `True`
    and `()`.
- The adapter rules of D-S8-11 hold: exceptions propagate as the same
  object, `KeyboardInterrupt` passes through, a wrong result type raises
  `TypeError`, and a hook is called once per pair of extended nodes.
- The container, not the object, holds the adapter; containers traverse
  their Python objects for GC (`util::gc`).

**Path 2, downstream Rust kinds.**
- `convert::search_space` exposes `variable_from_python`,
  `variable_to_python`, `alternative_from_python`, `alternative_to_python`,
  `register_variable_kind(module, kind, class, from_python, to_python)` and
  `register_alternative_kind(...)`.
- The registry is append-only, write-once per kind, in the extension's
  module state (approved). The downstream pyclass is registered as a
  virtual subclass of the public `Variable` or `Alternative`.
- Reading an object, the binding tries, in order: the plain class, a
  registered Rust kind, then the Python adapter.

**Equality (N-7 translated):** identity `==` and `hash` for every class
except `ConfigurationKey`, whose `==` and `hash` are structural: it exists
to be a dictionary key. Equivalence of everything else is spelled
`is_structurally_equivalent` and `is_alpha_equivalent`.

## Implementors: `ArrayTileKnob` and `RealizationOption`

**As downstream Rust (a future MOGA-VM crate in the aggregate):**

```rust
#[derive(Debug)]
pub struct ArrayTileKnob { name: Identifier, param: Param, notes: Vec<Note>, index_symbols: Vec<Identifier> }

impl Variable for ArrayTileKnob {
    fn kind(&self) -> Cow<'_, str> { Cow::Borrowed("moga.cir.array_tile_knob") }
    fn name(&self) -> &Identifier { &self.name }
    fn param(&self) -> &Param { &self.param }
    fn notes(&self) -> &[Note] { &self.notes }
    fn is_extension_structurally_equivalent(&self, other: &dyn Variable) -> Result<bool, BoxError> {
        let Some(other) = other.as_any().downcast_ref::<Self>() else { return Ok(false) };
        Ok(self.index_symbols == other.index_symbols)
    }
    fn is_extension_alpha_equivalent_under(&self, other: &dyn Variable, renaming: &AlphaRenaming) -> Result<bool, BoxError> {
        let Some(other) = other.as_any().downcast_ref::<Self>() else { return Ok(false) };
        Ok(self.index_symbols.len() == other.index_symbols.len()
            && self.index_symbols.iter().zip(&other.index_symbols).all(|(l, r)| renaming.is_corresponding(l, r)))
    }
}
// ForeignPart: type_name "ArrayTileKnob"; to_foreign writes the payload under "moga.cir.array_tile_knob".

#[derive(Debug)]
pub struct RealizationOption { name: Identifier, variables: Vec<Part<dyn Variable>>, notes: Vec<Note>, realization: ImmutableHardwareDfg }

impl Alternative for RealizationOption {
    fn kind(&self) -> Cow<'_, str> { Cow::Borrowed("moga.cir.realization_option") }
    fn name(&self) -> &Identifier { &self.name }
    fn variables(&self) -> &[Part<dyn Variable>] { &self.variables }
    fn bound_identifiers(&self) -> Result<Vec<Identifier>, BoxError> { Ok(self.realization.walk_axes_in_topological_order()) }
    fn is_extension_alpha_equivalent_under(&self, other: &dyn Alternative, renaming: &AlphaRenaming) -> Result<bool, BoxError> {
        let Some(other) = other.as_any().downcast_ref::<Self>() else { return Ok(false) };
        // `renaming` already pairs the walk axes, standalone or nested (C-1).
        Ok(self.realization.is_alpha_equivalent_under(&other.realization, renaming)?)
    }
}
```

The walk-order and tile variables of a realization's nested walks could
become sub-choices of the alternative (hierarchy) instead of flat variables
with walk axes as references; MOGA-VM decides that (see the recommendation under "Decisions").

**As Python subclasses (MOGA-VM after migration):**

```python
@register_serializable(type_id="moga.cir.array_tile_knob")
class ArrayTileKnob(fhy_core.search_space.Variable):
    def __init__(self, *, index_symbols: tuple[SymbolName, ...], **fields: Any) -> None:
        super().__init__(**fields)
        self.index_symbols = tuple(index_symbols)

    def extension_is_structurally_equivalent(self, other: "ArrayTileKnob") -> bool:
        return self.index_symbols == other.index_symbols

    def extension_is_alpha_equivalent_under(self, other: "ArrayTileKnob", renaming: AlphaRenaming) -> bool:
        return len(self.index_symbols) == len(other.index_symbols) and all(
            renaming.are_identifiers_alpha_equivalent(l, r)
            for l, r in zip(self.index_symbols, other.index_symbols))

    def serialize_data_to_dict(self) -> SerializedDict:
        return {**super().serialize_data_to_dict(),
                "index_symbols": [s.serialize_to_dict() for s in self.index_symbols]}


@register_serializable(type_id="moga.cir.realization_option")
class RealizationOption(fhy_core.search_space.Alternative):
    def __init__(self, *, realization: ImmutableHardwareDFG, **fields: Any) -> None:
        super().__init__(**fields)  # name=, variables=, choices=, notes=
        self.realization = realization

    def extension_bound_identifiers(self) -> tuple[Identifier, ...]:
        return tuple(_walk_axes_in_topological_order(self.realization))

    def extension_is_structurally_equivalent(self, other: "RealizationOption") -> bool:
        return self.realization.is_structurally_equivalent(other.realization)

    def extension_is_alpha_equivalent_under(self, other: "RealizationOption", renaming: AlphaRenaming) -> bool:
        return self.realization.is_alpha_equivalent_under(other.realization, renaming)
```

`PortBoundKnob` adds `port_role` and `port_index` as attributes compared by
`==` in both hooks. The marker knobs (`ArrayKnob`, `WalkOrderKnob`,
`NamespaceKnob`, `AddressKnob`, `ByteAddressKnob`, `WordAddressKnob`) are
subclasses with no hooks, distinguished by kind.

## Symbol mapping

| MOGA-VM (`cir.space.core`) | Rust (`fhy_core::search_space`) | Public Python (`fhy_core.search_space`) | Type id |
|---|---|---|---|
| `Knob` | trait `Variable`, `PlainVariable` | `Variable` | `search_space.variable` |
| `Knob.name`, `.param`, `.notes` | `name()`, `param()`, `notes()` | same attributes | |
| `Knob.assign(v)` | `Configuration::with_entry(name, v, ctx)` | `configuration.with_entry(name, v)` | |
| `KnobAssignment(knob_identifier, assignment)` | an entry `(name, Value)` of a `Configuration` | `configuration.value(name)` | (inside `search_space.configuration`) |
| `Option` | trait `Alternative`, `PlainAlternative` | `Alternative` | `search_space.alternative` |
| `Option.knobs` | `Alternative::variables` | `variables` | |
| `Option.collect_alpha_label_bindings` | `Alternative::bound_identifiers` | `extension_bound_identifiers` | |
| (new) | `Alternative::choices` | `choices` | |
| `Option.metrics`, `Selection.metrics`, `Metric`, `MetricKind` | gone; SS3 `Objective`, `Direction`, `Measurement`, `Estimate` | SS3 | SS3 |
| `DecisionSpace(name, options)` | `Choice::new(name, alternatives)` | `Choice` | `search_space.choice` |
| `DecisionPoint(space, name, selection)` | a `Choice` inside a `Space`, plus an entry of a `Configuration` | `Space`, `Configuration` | `search_space.space`, `search_space.configuration` |
| `Selection`, `ArraySelection` | `Configuration` | `Configuration` | `search_space.configuration` |
| `Selection.selected_option_identifier` | `Configuration::alternative(choice)` | `configuration.alternative(choice)` | |
| `Selection.knob_assignments` | `Configuration` entries | `configuration.entries` | |
| `SelectionStatus` | `Configuration::is_complete`, `is_active` | `is_complete()`, `activity(name)` | |
| `SelectionConsistencyError` | `ConfigurationErrors` | `ConfigurationError` | |
| (none) | `Condition`, `Forbidden` | same | nested |
| (none) | `ConfigurationKey` | `ConfigurationKey` | |
| `is_structurally_equivalent`, `is_alpha_equivalent(_under)` | inherent on `dyn` parts; `AlphaEquivalence` for containers | methods | |
| subclass overrides of the relations | `is_extension_*` | `extension_*` hooks | |
| `equivalence.py` helpers, `is_valid_*`, `*Data` | private `equivalence.rs` | deleted | |
| `cir/lowering/search`: `RecordedDecision`, `SearchPoint`, `DecisionKind` | SS2 `TraceStep`, `Trace`, `DecisionKind` (a string) | SS2 | `search_space.trace` |
| `cir/lowering/search`: `ChoiceDomain`, `OrderDomain`, `AddressDomain`, `SearchOracle`, `ReplayOracle` | SS2 domains, `SearchOracle`, oracles; the full table is "Symbol and visibility mapping (SS2)" | SS2 | |

The `Option` vs `Alternative` naming question (old N-2) disappears: (C)
names it `Alternative` in both languages.

## MOGA-VM migration map

None of this is done by the port; it is what MOGA-VM changes to use it.

| Site | Today | After |
|---|---|---|
| `TableEntry` (`cir/space/table.py:92-195`) | wraps a `DecisionPoint` named by the vertex id, with the selection inside | MOGA's wrapper of a `Choice` named by the vertex id, plus `realized_vertex_partition`; no selection |
| `CandidateTable` (`table.py:257-455`) | map vertex → entry | wraps a `Space` whose top-level choices are the entries' choices, plus the partitions; `partition_for`, `decision_point_for_vertex` and the iterators keep their meaning |
| the committed selections | inside each entry | one `Configuration` of the function's space, a new field of `Function` (recommended to MOGA-VM; see "Decisions") |
| `builder/authoring/dsl.py:311, 317` | `TableEntry(vertex_id, options, selection)` | build the `Choice`; a given selection becomes `configuration.with_entry(vertex_id, option.name)` |
| `builder/authoring/dsl.py:364` | `TableEntry(...)` | the same |
| `mlir_to_cir/lower.py:283` (selection at `:206`) | entry with a SELECTED selection | `Choice`, and a configuration entry |
| `lowering/default/mapping/operations.py:163` (`replace(selection, ...)` at `:185`) | rebuilds the entry with a re-pointed selection | `configuration.with_entry(vertex_id, option.name)`; the entry keeps its choice and gains the partition |
| `lowering/default/memory/bind.py:305` (`:318-334`) | rebuilds the entry; appends namespace assignments; marks SELECTED | `configuration.with_entries(...)`: one value per namespace variable, so duplicates are impossible (C-6) |
| `lowering/default/memory/tiles.py:165` (`:67`) | rebuilds the entry; appends a tile assignment | `configuration.with_entry(tile_variable.name, tile)` |
| the five selection readers: `memory/bind.py:102`, `memory/namespaces.py:76`, `memory/plan.py:72`, `memory/tile_policy.py:118`, `memory/tiles.py:52` | `option.name == selection.selected_option_identifier` over `entry.options` | `configuration.alternative(entry.vertex_id)` |
| `render/text.py:205-270` | formatters for `DecisionPoint`, `DecisionSpace`, `Selection`, `WalkOrderKnob` | formatters for `Choice`, `Space`, `Configuration`, variables |
| `render/explorer/graphs.py:436-470, 539` | decision-point and selection attributes | the choice's alternatives and variables; the configuration's entries for that choice |
| `cir/program.py:114-140` | compares candidate tables | compares the spaces and the configurations (keys, under the spaces' equivalence) |
| knob classes (`cir/space/knobs.py`) | `Knob` subclasses | `Variable` subclasses or Rust implementors, as above |
| `RealizationOption` | `Option` subclass | `Alternative` subclass or Rust implementor, as above |
| `lowering/search` | its own stream | SS2's `Trace`, oracles and `Recorder`; `ExtractedSearchSpace` becomes a `Space` built by MOGA's extractor; site by site in "MOGA-VM migration map (SS2 and SS3)" |
| payloads | `moga.cir.decision_point`, `decision_space`, `array_selection`, `table_entry`, `candidate_table` shapes | MOGA's table and entry payloads hold a `search_space.choice` and a partition; the function holds a `search_space.space` and a `search_space.configuration`; knob payloads unchanged |

## Earlier decisions, translated to (C)

| Item | Accepted recommendation | Under (C) |
|---|---|---|
| N-2 names | `fhy_core.search_space`, `fhy_core::search_space`, `Plain*` | unchanged: `PlainVariable`, `PlainAlternative`; the Option/Alternative question is gone |
| N-3 fixes | D-SS-1 to D-SS-5, 7, 9, 12; refuse empty spaces; keep default names | adopted; "empty space" is now an empty `Choice`; Python keeps default names (minted per construction) for `Variable`, `Alternative`, `Choice` and `Space`; Rust constructors take names |
| N-6 type ids | generic `search_space.*`; MOGA-VM owns its aliases | as listed; aliases now help only knob payloads; no migration tool needed |
| N-7 `==` | identity | identity, except `ConfigurationKey` (structural) |
| N-8 dependencies | none | none in SS0 and SS1; SS2's random draws use a small in-house PRNG or the stdlib, never Python's `random.Random` stream (a divergence) |
| N-9 status | parity, documented | moot: validity (by construction) and `is_complete()` replace the status |
| N-10 name vs variable | keep both, check against the knob | a `Variable`'s name and its param's variable stay distinct. Conditions, forbidden clauses, configuration entries and traces name the `Variable`; the param's constraints name the param's variable. A configuration value is checked with `ParamAssignment::new` against the variable's param. No separate assignment param exists to mismatch. |

### Deviations of SS1.5 to SS1.7 from this design (accepted 2026-10-06)

1. `register_variable_kind` and `register_alternative_kind` take a sixth
   argument, the kind's resolver, so decoding a Rust kind needs no call
   into Python.
2. The search-space classes have no V1 form: writing one inside
   `wire_version(WireVersion.V1)` raises `SerializationError`, since no V1
   payload of them can exist.
3. The binding's `classes.rs` is split into one file per class, and the
   adapters are named `PythonVariable` and `PythonAlternative`.
4. `_FrozenAfterInit` moves from `fhy_core.types.core` to
   `fhy_core.traits.frozen`, shared by the types and the search space.
5. The example aggregate depends on `serde` and `serde_json`, workspace
   crates already in the lock file, for its kinds' foreign payloads.
6. Two existing tests change with the new subsystem: the pinned top-level
   `fhy_core.__all__` and the GC-flag exemption of `ConfigurationKey`.
7. `ConfigurationKey` pickles through its wire form (not a deviation from
   the design, but from the first plan, which refused to pickle it), which
   adds `Serialize`/`Deserialize` and `wire::ConfigurationKeyData` to the
   core's public API.

## Intended divergences from MOGA-VM

| # | Behavior | MOGA-VM | After |
|---|---|---|---|
| D-SS-1 to D-SS-4 | equivalence | the audit's defects | fixed as above |
| D-SS-5 | validation | three checks in `DecisionPoint` | `Space::new` and `Configuration::new` checks; collect-all for configurations |
| D-SS-6 | empty decision | accepted | an empty `Choice` is refused |
| D-SS-7 | concrete classes | `Option` abstract, `Knob` unregistered | `PlainVariable`, `PlainAlternative`, registered |
| D-SS-9 | helpers | 17 public | deleted |
| D-SS-12 | subclass customization | override the relations | `extension_*` hooks; the core compares base fields |
| D-SS-13 | wire | V1, MOGA ids | V2 core shape, `search_space.*` ids, tagged parts |
| D-SS-17 | the vocabulary | seven components | (C)'s components (decided) |
| D-SS-18 | selection placement | inside the decision point | in a `Configuration`, outside the space |
| D-SS-19 | names | unique per list | unique space-wide (N-C1) |
| D-SS-20 | metrics | on options and selections | gone from the space; SS3 |

## Test plan and traceability

**Rust, written first,** in `rust/fhy-core/tests/it/search_space/`:

- `variable_stories.rs`, `alternative_stories.rs`, `choice_stories.rs`:
  construction, kinds, standalone equivalence, members as references,
  constraints in alpha mode, type-strict values.
- `space_stories.rs`: uniqueness, empty choices, condition scopes, cycles,
  canonical and decision orders, hierarchy.
- `activity_stories.rs`: each activity rule, including pending and
  undecided.
- `forbidden_stories.rs`: inactive, pending, holding and undecided clauses.
- `configuration_stories.rs`: every `ConfigurationError`, collect-all,
  completeness, the key.
- `implementor_stories.rs`: implementors shaped like `ArrayTileKnob` and
  `RealizationOption`, and one breaking each contract clause.
- `equivalence_stories.rs`: the audit's A1 to A7 and C-1, C-2, C-4, both
  directions.
- `properties.rs` (proptest):
  - reflexive, symmetric, structural implies alpha;
  - invariant under bijective relabeling;
  - a perturbation breaks equivalence;
  - equal keys for corresponding configurations;
  - activity agrees with a brute-force reference evaluator on small
    spaces.
- `serde_stories.rs`, and `search_space_golden.rs`.

**Python:**
- `tests/search_space/test_search_space.py`;
- `test_search_space_rust_binding.py` (class structure, frozen, identity
  `==`, `ConfigurationKey` hashing, kept objects, pickling, errors);
- `test_extension.py`: Python subclasses through hooks; a registered Rust
  kind through `rust/example-aggregate`'s pattern;
- `test_search_space_properties.py`.

### Traceability

Every test in MOGA-VM's `tests/cir/space/` and `tests/cir/test_space.py`.
"plain alternative" replaces `RealizationOption`; "test implementor" is a
test-local implementor standing in for MOGA-VM's.

| MOGA-VM test | Behavior | Port test(s) | Status |
|---|---|---|---|
| `test_alpha_standalone.py::test_knob_alpha_equivalent_standalone_with_distinct_names` | a variable binds its name standalone | `variable_stories::variables_with_distinct_names_are_alpha_equivalent_standalone` | ported |
| `::test_knob_not_alpha_equivalent_when_param_domain_differs_standalone` | domains discriminate | `variable_stories::variables_over_different_domains_are_not_alpha_equivalent` | ported |
| `::test_array_option_alpha_equivalent_standalone_with_distinct_names` | an alternative binds its labels | `alternative_stories::alternatives_with_distinct_labels_are_alpha_equivalent_standalone` | ported (plain alternative) |
| `::test_array_option_not_alpha_equivalent_when_knob_param_differs_standalone` | variable params discriminate | `alternative_stories::alternatives_with_different_variable_domains_are_not_alpha_equivalent` | ported (plain alternative) |
| `::test_array_decision_space_alpha_equivalent_standalone_with_distinct_names` | a choice binds its labels | `choice_stories::choices_with_distinct_labels_are_alpha_equivalent_standalone` | ported (`DecisionSpace` → `Choice`) |
| `::test_array_decision_space_not_alpha_equivalent_when_option_param_differs` | nested params discriminate | `choice_stories::choices_with_different_variable_domains_are_not_alpha_equivalent` | ported |
| `::test_options_not_alpha_equivalent_when_param_domains_differ` | with a positive control | `alternative_stories::alternatives_differing_only_in_a_domain_are_not_alpha_equivalent` | ported |
| `::test_selection_alpha_equivalent_when_space_labels_seeded` | selection references resolve through a seeded renaming | `configuration_stories::configurations_of_relabeled_spaces_have_equal_keys`, `equivalence_stories::configuration_entries_correspond_under_the_space_frame` | divergence D-SS-18: a configuration is compared with its space, not standalone |
| `::test_selection_not_alpha_equivalent_without_seeding` | unseeded references are free | `equivalence_stories::configurations_over_unrelated_spaces_are_not_equivalent` | divergence D-SS-18 |
| `::test_selection_not_alpha_equivalent_when_status_differs_under_seeding` | status discriminates | `configuration_stories::complete_and_incomplete_configurations_have_different_keys` | divergence (status replaced by completeness) |
| `::test_selection_not_alpha_equivalent_when_knob_value_differs_under_seeding` | values discriminate | `configuration_stories::different_values_give_different_keys` | ported |
| `::test_address_knobs_alpha_equivalent_when_bounds_match_under_renaming` | bounded params over renamed variables | `variable_stories::bounded_variables_whose_param_variables_are_renamed_are_alpha_equivalent` | ported (`WordAddressKnob` stays in MOGA-VM) |
| `test_selection_validation.py::test_decision_point_accepts_fully_covered_selection` | full coverage valid | `configuration_stories::complete_configuration_is_valid_and_complete` | ported |
| `::test_decision_point_without_selection_is_unconstrained` | no choice made | `configuration_stories::empty_configuration_is_valid_and_incomplete` | ported |
| `::test_decision_point_accepts_partial_status_with_some_coverage` | partial assignment valid | `configuration_stories::partial_configuration_is_valid_and_incomplete` | ported |
| `::test_decision_point_rejects_unknown_selected_option` | unknown alternative refused, message names it | `configuration_stories::unknown_alternative_is_refused`; Python message | ported |
| `::test_decision_point_rejects_assignment_to_unknown_knob` | a value for a variable of another alternative refused | `configuration_stories::value_for_an_inactive_variable_is_refused` | ported (now "inactive decision") |
| `::test_decision_point_accepts_selected_status_with_incomplete_coverage` | committed alternative with unassigned variables | `configuration_stories::chosen_alternative_with_unassigned_variables_is_valid_and_incomplete` | ported |
| `::test_decision_point_rejects_unselected_status_with_assignments` | values without a choice refused | `configuration_stories::values_under_an_unchosen_choice_are_refused` | ported (status gone) |
| `test_structural_equivalence.py::test_structural_equivalence_is_reflexive[metric, knob, empty-option, knobbed-option, space, unselected-point, selected-point]` | reflexive | `equivalence_stories::structural_equivalence_is_reflexive` (variable, plain and implementor alternatives, choice, space, configuration); the property | ported, strengthened; `metric` case moves to SS3 |
| `::test_structural_equivalence_discriminates_a_perturbed_field[...7]` | one perturbed field breaks it, both ways | `equivalence_stories::structural_equivalence_discriminates_a_perturbed_field` | ported (test implementor for `knobbed-option`); `metric` to SS3 |
| `::test_options_structurally_equivalent_when_sharing_all_identifiers` | same parts | `alternative_stories::alternatives_sharing_all_parts_are_structurally_equivalent` | ported |
| `::test_options_not_structurally_equivalent_when_name_differs` | names nominal | `alternative_stories::alternatives_with_different_names_are_not_structurally_equivalent` | ported |
| `::test_options_not_structurally_equivalent_when_realization_domain_differs` | implementor data counts | `implementor_stories::alternatives_with_different_implementor_data_are_not_structurally_equivalent`; Python `test_extension.py` | ported (test implementor) |
| `::test_decision_point_structural_equivalence_symmetric_for_selection_presence` | presence of a choice's value | `configuration_stories::assigning_a_choice_changes_the_key` | ported (configuration) |
| `::test_decision_space_not_equivalent_to_shorter_option_list` | prefix | `choice_stories::choice_is_not_equivalent_to_a_prefix` (rstest, both orders) | ported |
| `::test_decision_space_not_equivalent_to_longer_option_list` | converse | same rstest | ported |
| `::test_option_not_equivalent_to_shorter_knob_list` | prefix of variables | `alternative_stories::alternative_is_not_equivalent_to_a_variable_prefix` (both orders) | ported |
| `::test_option_not_equivalent_to_longer_knob_list` | converse | same rstest | ported |
| `::test_distinct_knob_kinds_not_structurally_equivalent` | kinds discriminate | `implementor_stories::variables_of_different_kinds_are_not_structurally_equivalent`; Python marker subclass | ported |
| `::test_same_knob_kind_structurally_equivalent_for_shared_name_and_param` | same kind, same parts | `variable_stories::variables_of_one_kind_sharing_parts_are_structurally_equivalent` | ported |
| `test_core_import_boundary.py::test_core_modules_import_only_fhy_core_stdlib_or_core_siblings` | the core depends on fhy_core only | `tests/test_import_graph.py` row; layer 11 | replaced: the crate's layering |
| `tests/cir/test_space.py` (6 address and namespace knob tests) | MOGA knob constructors | - | stays in MOGA-VM |
| `test_table.py` (17) | `TableEntry`, `CandidateTable` | core analogs: unknown alternative, round trip, clone equivalence | stays in MOGA-VM; rewritten there over `Choice` and `Space` |
| `test_table_partition.py` (25) | partitions over `HardwareDFG` | - | stays in MOGA-VM |

Counts over the 82 tests (50 in scope, 48 staying in MOGA-VM, the
parametrized cases counted singly): 38 ported, 3 divergences (D-SS-18 and
the status), 2 `metric` cases moved to SS3, 1 replaced, 48 stay in MOGA-VM.

### Traceability, Python half

The same MOGA-VM tests, ported to the Python interface in
`tests/search_space/` (`test_search_space.py` unless noted); each test's
docstring cites the MOGA-VM test.

| MOGA-VM test | Python test(s) | Status |
|---|---|---|
| `test_alpha_standalone.py::test_knob_alpha_equivalent_standalone_with_distinct_names` | `test_variables_with_distinct_names_are_alpha_equivalent_standalone` | ported |
| `::test_knob_not_alpha_equivalent_when_param_domain_differs_standalone` | `test_variables_over_different_domains_are_not_alpha_equivalent` | ported |
| `::test_array_option_alpha_equivalent_standalone_with_distinct_names` | `test_alternatives_with_distinct_labels_are_alpha_equivalent_standalone`; `test_extension.py::test_realizations_are_alpha_equivalent_standalone` | ported (plain alternative; a `RealizationOption`-shaped subclass) |
| `::test_array_option_not_alpha_equivalent_when_knob_param_differs_standalone` | `test_alternatives_with_different_variable_domains_are_not_alpha_equivalent` | ported |
| `::test_array_decision_space_alpha_equivalent_standalone_with_distinct_names` | `test_choices_with_distinct_labels_are_alpha_equivalent_standalone` | ported |
| `::test_array_decision_space_not_alpha_equivalent_when_option_param_differs` | `test_choices_with_different_variable_domains_are_not_alpha_equivalent` | ported |
| `::test_options_not_alpha_equivalent_when_param_domains_differ` | `test_alternatives_differing_only_in_a_domain_are_not_alpha_equivalent` | ported |
| `::test_selection_alpha_equivalent_when_space_labels_seeded` | `test_configurations_of_relabeled_spaces_have_equal_keys`, `test_configurations_of_relabeled_spaces_are_alpha_equivalent` | divergence D-SS-18 |
| `::test_selection_not_alpha_equivalent_without_seeding` | `test_configurations_of_unrelated_spaces_are_not_alpha_equivalent` | divergence D-SS-18 |
| `::test_selection_not_alpha_equivalent_when_status_differs_under_seeding` | `test_complete_and_incomplete_configurations_have_different_keys` | divergence (status replaced by completeness) |
| `::test_selection_not_alpha_equivalent_when_knob_value_differs_under_seeding` | `test_different_values_give_different_keys`, `test_configurations_with_different_values_are_not_alpha_equivalent` | ported |
| `::test_address_knobs_alpha_equivalent_when_bounds_match_under_renaming` | `test_bounded_variables_whose_names_are_renamed_are_alpha_equivalent_under_it` | ported (an integer-range param stands for `WordAddressKnob`'s) |
| `test_selection_validation.py::test_decision_point_accepts_fully_covered_selection` | `test_complete_configuration_is_valid_and_complete` | ported |
| `::test_decision_point_without_selection_is_unconstrained` | `test_empty_configuration_is_valid_and_incomplete` | ported |
| `::test_decision_point_accepts_partial_status_with_some_coverage` | `test_partial_configuration_is_valid_and_incomplete` | ported |
| `::test_decision_point_rejects_unknown_selected_option` | `test_unknown_alternative_is_refused_naming_it` | ported; the message names the choice and the value, not the alternatives offered |
| `::test_decision_point_rejects_assignment_to_unknown_knob` | `test_value_for_an_inactive_variable_is_refused` | ported (now "inactive decision") |
| `::test_decision_point_accepts_selected_status_with_incomplete_coverage` | `test_chosen_alternative_with_unassigned_variables_is_valid_and_incomplete` | ported |
| `::test_decision_point_rejects_unselected_status_with_assignments` | `test_values_under_an_unchosen_choice_are_refused` | ported (status gone) |
| `test_structural_equivalence.py::test_structural_equivalence_is_reflexive[7]` | `test_structural_equivalence_is_reflexive[6]` (variable, empty alternative, choice, space, two configurations); `knobbed-option` in `test_extension.py::test_tile_knobs_with_equal_data_are_structurally_equivalent` | ported; `metric` to SS3 |
| `::test_structural_equivalence_discriminates_a_perturbed_field[7]` | `test_structural_equivalence_discriminates_a_perturbed_field[6]`; `test_extension.py::test_alternatives_with_different_own_data_are_not_structurally_equivalent` | ported; `metric` to SS3 |
| `::test_options_structurally_equivalent_when_sharing_all_identifiers` | `test_alternatives_sharing_all_parts_are_structurally_equivalent` | ported |
| `::test_options_not_structurally_equivalent_when_name_differs` | `test_alternatives_with_different_names_are_not_structurally_equivalent` | ported |
| `::test_options_not_structurally_equivalent_when_realization_domain_differs` | `test_extension.py::test_alternatives_with_different_own_data_are_not_structurally_equivalent` | ported (subclass) |
| `::test_decision_point_structural_equivalence_symmetric_for_selection_presence` | `test_assigning_a_choice_changes_the_configuration` | ported (configuration) |
| `::test_decision_space_not_equivalent_to_shorter_option_list` | `test_choice_is_not_structurally_equivalent_to_a_prefix[longer]` | ported |
| `::test_decision_space_not_equivalent_to_longer_option_list` | `test_choice_is_not_structurally_equivalent_to_a_prefix[shorter]` | ported |
| `::test_option_not_equivalent_to_shorter_knob_list` | `test_alternative_is_not_structurally_equivalent_to_a_variable_prefix[longer]` | ported |
| `::test_option_not_equivalent_to_longer_knob_list` | `test_alternative_is_not_structurally_equivalent_to_a_variable_prefix[shorter]` | ported |
| `::test_distinct_knob_kinds_not_structurally_equivalent` | `test_extension.py::test_variables_of_different_kinds_are_not_structurally_equivalent` | ported (marker subclass) |
| `::test_same_knob_kind_structurally_equivalent_for_shared_name_and_param` | `test_variables_of_one_kind_sharing_parts_are_structurally_equivalent`; `test_extension.py::test_variables_of_one_subclass_kind_sharing_parts_are_structurally_equivalent` | ported |
| `test_core_import_boundary.py::test_core_modules_import_only_fhy_core_stdlib_or_core_siblings` | `tests/test_import_graph.py` (`fhy_core.search_space` entry point) | replaced: the package's import graph |
| `tests/cir/test_space.py`, `test_table.py`, `test_table_partition.py` | - | stay in MOGA-VM |

## Equivalence plan

The oracle (MOGA-VM `3d93ba3` on fhy_core v0.1.8, with stand-ins, as the
audit ran it) still defines the behavior (C) keeps.

- **Golden corpus:** `rust/fhy-core/tests/golden/record_search_space_cases.py`
  writes `search_space_cases.json`, which `tests/it/search_space_golden.rs`
  replays. It assembles the oracle in a temporary directory: `git archive`
  of MOGA-VM's `src/moga_vm/cir/space` at `3d93ba3` from the checkout
  `--moga-vm-repo` names, `git archive` of this repository's `src/fhy_core`
  at the tag `v0.1.8`, and the audit's stand-ins for the rest of
  `moga_vm.cir` and for `moga`, which the recorder holds as text. To
  regenerate, from the repository root: `uv run --no-sync --with networkx
  python rust/fhy-core/tests/golden/record_search_space_cases.py
  --moga-vm-repo <MOGA-VM checkout>` (fhy_core v0.1.8 imports `networkx`).
  CI only replays the corpus. It is a recorder, not a `generate_*.py` generator: the drift
  check (`tests/test_golden_corpora.py`) and the `golden_expanded` session
  rerun every generator in CI, which cannot install MOGA-VM, and this
  oracle is frozen (MOGA-VM `3d93ba3` on fhy_core v0.1.8), so its corpus is
  a fixed regression corpus, as a deleted Python implementation's is.
  - Each case is described once (labels with sharing indices, finite
    domains with constraints, a commitment and values) and built twice: as
    a MOGA-VM `DecisionPoint` (the oracle) and as the corresponding `Choice`
    in a `Space` plus a `Configuration` (the port).
  - It records the oracle's verdicts (structural and alpha, both
    directions) and validation outcomes, with provenance.
  - Each verdict a divergence changes (A1 to A7, duplicates, status) is
    tagged, its expected value written from the rule, and reviewed by hand.
  - Each case the port builds also carries its V2 texts, the space's and
    the configuration's, written by the recorder from the documented
    shapes with identifiers at fixed ids: the golden rows the serde replay
    reads and writes back byte-identically.
- **Expanded corpus:** the audit sweep's distribution (`--seed 7
  --random-count 2000`), recorded by hand and replayed by the ignored
  `search_space_golden` test from the file `FHY_SEARCH_SPACE_CORPUS`
  names. It is not in `noxfile.py`'s `EXPANDED_GOLDEN_CORPORA`, for the
  reason above.
- **Not oracle-backed:** conditions, forbidden clauses and hierarchy beyond
  one level, which MOGA-VM lacks. These are checked against a brute-force
  reference evaluator written in the test (proptest), not the oracle.
- **Harness honesty, coverage and mutation** as in the port skill: flip a
  scoping rule and see the replay fail; `cargo llvm-cov`; `cargo mutants`
  on `search_space`.

## Benchmark plan

`benchmarks/test_search_space.py`, written in SS1.1 against the planned
Python API; it skips until `fhy_core.search_space` exists (SS1.7), which
adapts it to the final signatures. The before numbers are MOGA-VM's Python
core on fhy_core 0.2.0 (audit oracle, 2026-10-05, CPython 3.11): building
8×4 options with params 660 µs; validating a decision point 9.8 µs;
structural self-equivalence 33 µs; alpha against a relabeled copy 184 µs;
serializing a selection 10.7 µs; serializing a knob 2.8 µs; `assign`
5.0 µs.

Added for the port:
- `Space::new` with conditions and forbidden clauses;
- `Configuration::new` and `with_entry`;
- `key()`;
- a comparison with one Python-hook alternative, and with a registered
  Rust kind.

Rule 5 of python-switch's cross-cutting rules applies.

## SS1.8: equivalence runs, benchmarks and the divergence log

### Equivalence runs (2026-10-06)

| Corpus | Cases | Rust core (`search_space_golden.rs`) | Python binding (`test_search_space_golden.py`) |
|---|---|---|---|
| committed (`--seed 0 --random-count 60`) | 138 (18 probes, 120 random) | all replay | all replay |
| expanded (`--seed 7 --random-count 2000`, recorded by hand into `target/`, not committed) | 4018 | all replay (`FHY_SEARCH_SPACE_CORPUS`) | all replay (`FHY_SEARCH_SPACE_CORPUS`) |

Each replay checks the outcome (built, space refused, configuration
refused), the structural and alpha verdicts in both directions, and the V2
texts of the space and the configuration, written and read back byte for
byte. A case without a tagged divergence must agree with the oracle.

- **Harness honesty:** flipping one case's recorded alpha verdict, and
  adding one space to one recorded text, each fails the Python replay
  naming the case.
- **Recorder fix:** the recorder picked a selected option by its name
  alone, and failed on a seed-7 draw whose options share a name. It now
  picks the first option of that name holding every assigned knob; the
  committed corpus records identically.

### Divergence log

Every case where the port answers otherwise than the oracle is tagged
with an approved divergence, and its expected answer is written from the
rule. Counts over the committed (expanded) corpus:

| Divergence | Cases | What the port does instead |
|---|---|---|
| D-SS-19 (names unique space-wide) | 94 (2896) | refuses a space repeating a name, where MOGA-VM built it and answered asymmetrically |
| D-SS-5 (validation) | 2 (7) | refuses a configuration MOGA-VM's three checks accepted (a knob assigned twice; a value outside the domain, assigned from another param) |
| D-SS-1 (constraints under the variable frame) | 1 (1) | a categorical knob's constraint compares under its param's variable |
| D-SS-3 (capture-free identifier members) | 1 (1) | a free category no longer matches a bound name |
| D-SS-4 (type-strict values) | 1 (1) | `1` and `True` are different categories |
| D-SS-6 (empty choice) | 1 (1) | an empty choice is refused |
| D-SS-18 (selection outside the space) | 1 (1) | the status is gone: an UNSELECTED selection assigning its chosen alternative's knob is a valid configuration |
| none | 37 (1110) | answers as the oracle does |

No untagged case disagrees with the oracle, in either corpus or either
language.

### Benchmarks (CPython 3.11, this machine, back to back)

"Before" is MOGA-VM `3d93ba3` on fhy_core v0.1.8, the recorder's oracle
(`target/scratch/search-space-py/bench_compare.py before`); the design's
earlier numbers were on fhy_core 0.2.0. "After" is the same rows through
`fhy_core.search_space`, and `benchmarks/test_search_space.py` under
`nox -s benchmark-3.11` agrees within 8%. Median per call, µs.

| Row | Before (MOGA-VM) | After (port) | Ratio |
|---|---|---|---|
| build 8×4 options with params | 640 | 599 | 0.94 |
| validate a decision point / build a configuration | 8.8 | 4.0 | 0.46 |
| structural self-equivalence | 853 | 4.8 | 0.006 |
| alpha against a relabeled copy | 187 | 18.0 | 0.10 |
| **serialize a selection / a configuration** | 17.4 | 75.2 | **4.3, slower (accepted)** |
| serialize a knob / a variable | 3.8 | 2.1 | 0.54 |
| **`knob.assign` / `with_entry`** | 3.4 | 4.2 | **1.21, slower (accepted)** |
| (new) space with a condition and a clause | - | 34.4 | |
| (new) `key()` | - | 0.34 | |
| (new) alpha through a Python subclass's hooks | - | 0.86 | |

The two slower rows are inherent to the design, not the binding:

- A configuration's payload holds its whole space (`{"space",
  "entries"}`), 8×4 params here, where MOGA-VM's selection wrote only its
  assignments. Writing only the entries would need the space supplied
  separately when reading; a cache keyed by `ConfigurationKey` with the
  space beside it avoids writing the space at all.
- `with_entry` checks the whole configuration again (activity, conditions,
  forbidden clauses), where `knob.assign` checked one value. The absolute
  cost is under 5 µs.

The user accepted both on 2026-10-06 as recorded costs of the design
(B-SS1 under "Decisions"). The rule they were measured against is
CONTRIBUTING's "Replacing a Python class" (more than 10% slower on a
row); neither path replaces a fhy_core Python class, so it binds
MOGA-VM's adoption rather than this package.

## SS1.9: the MOGA-VM migration note

What MOGA-VM changes to use `fhy_core.search_space` (fhy_core 0.2.x with
SS1), in order; the map above lists every site.

1. **Depend** on the fhy_core release with SS1, and import from
   `fhy_core.search_space`, never from `moga_vm.cir.space.core`, whose
   modules are deleted with their tests (the port's tests replace them; see
   "Traceability").
2. **Knobs** become `Variable` subclasses, each `@register_serializable`
   under its existing `moga.cir.*` id, so knob payloads still decode:
   - `ArrayTileKnob` keeps `index_symbols` and overrides the two
     `extension_*` hooks, comparing the symbols through
     `renaming.are_identifiers_alpha_equivalent` (see "Implementors");
   - `PortBoundKnob` compares `port_role` and `port_index` by `==`;
   - the marker knobs subclass with no hooks: their kind tells them apart;
   - each subclass with data of its own extends `serialize_data_to_dict`
     and overrides `deserialize_data_from_dict`, reading the base keys
     through `Variable.deserialize_data_from_dict`.
3. **`RealizationOption`** becomes an `Alternative` subclass:
   `extension_bound_identifiers` returns the realization's walk axes in
   topological order (C-1), and both hooks compare the realization.
4. **`TableEntry` and `CandidateTable`** wrap a `Choice` named by the
   vertex id and a `Space` of those choices; the committed selections leave
   the entries for one `Configuration` of the function's space (recommended
   as a field of `Function`).
5. **Writers of selections** become `configuration.with_entry(choice,
   option.name)` or `with_entries(...)` (`dsl.py`, `mlir_to_cir/lower.py`,
   `mapping/operations.py`, `memory/bind.py`, `memory/tiles.py`).
6. **Readers of selections** become `configuration.alternative(vertex_id)`
   (the five memory passes), and `SelectionStatus` checks become
   `is_complete()` and `activity(name)`.
7. **Comparisons** of candidate tables (`cir/program.py`) compare the
   spaces by `is_alpha_equivalent` and the configurations by `key()` or
   `is_alpha_equivalent`. Caches of measured configurations key on
   `(space, configuration.key())`; keys pickle, so they cross processes.
8. **Renderers** (`render/text.py`, `render/explorer/graphs.py`) format
   `Choice`, `Space`, `Configuration` and the variables.
9. **Errors:** `SelectionConsistencyError` becomes `ConfigurationError`,
   whose `problems` lists every problem; a repeated name anywhere in a space
   is `DuplicateNameError` (D-SS-19), so a table that reused a knob name
   across options must rename before it migrates.
10. **Payloads:** decision-point, decision-space, selection, table-entry
    and candidate-table payloads change shape; MOGA-VM stores none, so no
    conversion is needed. Knob payloads are unchanged.
11. **A Rust MOGA-VM crate**, when it exists, implements `Variable` and
    `Alternative` for its kinds and registers each from its aggregate's
    `#[pymodule]` with `convert::search_space::register_variable_kind` or
    `register_alternative_kind`, passing the kind's resolver;
    `rust/example-aggregate`'s `TiledVariable` and `AxisAlternative` are
    the template.

## Implementation checklist

Every step ends with the S16 gate:
- `pytest`, the `property` session and `tests_minimal`;
- lint, type check and the stub test;
- fmt, clippy `-D warnings` with and without `--all-features`, tests with
  default and all features, doc, deny, `cargo +1.85 check`.

1. **SS0:** identifier values and members; binding reader and writer;
   stories; golden row.
2. **SS1.1:** benchmarks file.
3. **SS1.2:** Rust stub (traits, plain implementations, `Choice`, `Space`,
   `Condition`, `Forbidden`, `Configuration`, `ConfigurationKey`, errors,
   wire), with rustdoc carrying the implementor contract; the layer 11 and
   mapping rows.
4. **SS1.3:** Rust tests, conformance helpers, the golden recorder and
   corpus; red.
5. **SS1.4:** the core, in this order:
   1. variables and alternatives;
   2. choices and `Space::new` (names, scopes, orders);
   3. activity and forbidden evaluation;
   4. `Configuration` and its key;
   5. the equivalence walk;
   6. serde and wire.
6. **SS1.5:** binding stub, `convert::search_space`, `_rs.pyi`; the
   interface and extension suites; red.
7. **SS1.6:** the binding: P2 classes, adapters, kind registry, GC
   traversal, errors, wire.
8. **SS1.7:** `fhy_core.search_space`; the ported Python tests.
9. **SS1.8:** equivalence runs, benchmarks after, the divergence log.
10. **SS1.9:** the MOGA-VM migration note (the map above).
11. **SS2:** SS2.0 to SS2.9, under "Implementation checklist (SS2,
    SS3)".
12. **SS3:** SS3.1 to SS3.4, in the same list; `Estimate` deferred
    (N-S3).

## Decisions

All decided by the user on 2026-10-05, except N-C5 and D-SS-3's resolution,
decided on 2026-10-06 for SS1. Nothing is open for SS0 and SS1; SS2's
and SS3's choices, decided 2026-10-06, are N-S1 to N-S5 below.

| # | Question | Decision |
|---|---|---|
| — | scope | only generic parts move into fhy-core; CIR-specific types stay in MOGA-VM as subclasses or implementors; the core is built on traits |
| N-1 | component set | (C): `Variable`, `Alternative`, `Choice`, `Space` (conditions, forbidden clauses), `Configuration`, `Trace`, `Measurement` |
| N-2 | names | `fhy_core.search_space` / `fhy_core::search_space`; plain implementations `PlainVariable`, `PlainAlternative`; `Alternative` in both languages |
| N-3 | audit fixes | adopt D-SS-1 to D-SS-5, 7, 9, 12, and refuse empty choices; Python keeps default names |
| N-4 | Python extension paths | (a) both: subclassable pyclasses with `extension_*` hooks, and the kind registry in `convert::search_space`, append-only module state (the user approves the new binding state) |
| N-5 | identifier members | (a) the SS0 slice: a first-class identifier kind in `fhy_core::constraint`'s values and members |
| N-6 | type ids | generic `search_space.*`; MOGA-VM owns any aliases; no migration tool, since MOGA-VM stores no payloads |
| N-7 | Python `==` | identity, except `ConfigurationKey` (structural, hashable) |
| N-8 | dependencies | none new; SS2's random-number generator is in house (N-S1) |
| N-S1 | SS2's random-number generator (decided 2026-10-06) | in-house SplitMix64, with fhy-core's own range sampling (`below`, `below_big`) and shuffle; the stream is pinned by golden tests; no new dependency |
| N-S2 | Python `==` of `Trace` and `Objective` (2026-10-06) | structural `==` and `hash` for both, exceptions to N-7 beside `ConfigurationKey` |
| N-S3 | `Estimate` (2026-10-06) | deferred; a later `Measurer<Configuration>` over an expression if a producer appears |
| N-S4 | multi-objective comparison (2026-10-06) | `Measurement::dominates` only |
| N-S5 | measurement statuses (2026-10-06) | `Ok`, `Infeasible`, `Failed`, `Timeout` |
| N-S6 | decision kinds and objective names (2026-10-06) | strings, stable across processes, not `Identifier`-keyed tags |
| B-SS1 | SS1's two slower benchmark rows (2026-10-06) | accepted as recorded costs: serializing a configuration (4.3x, its payload carries the space) and `with_entry` (1.21x, it re-validates the whole configuration) |
| N-9 | selection status | replaced by configuration validity and `is_complete()` |
| N-10 | name vs param variable | a `Variable`'s name and its param's variable stay distinct; constraints outside the param name the `Variable` |
| N-C1 | name scope | names unique across the whole `Space`; a duplicate is a `SpaceError` |
| N-C2 | undecided conditions and forbidden clauses | a `ConfigurationError` |
| N-C3 | `Metric` | dropped from the space; `Direction` and `Objective` come with `Measurement` in SS3 |
| N-C4 | first slice | SS1 without `Trace` and `Measurement` |
| N-C5 | choices in conditions and forbidden clauses (decided 2026-10-06, for SS1) | a condition or forbidden clause names a choice only in a set constraint (`choice in {a, b}`; equality is `in {a}`); an `Equation` naming a choice is a `SpaceError` (`EquationOverChoice`) from `Space::new`, never a failure at evaluation; equations over variables with numeric or Boolean params are fine |
| D-SS-3 | capture-free identifier matching (settled in SS1's plan) | the search-space walk resolves identifier members itself, through `is_corresponding`, in domains and in the set constraints of params, conditions and forbidden clauses; `constraint`'s own set-constraint alpha equivalence still compares members by value |

**A recommendation for MOGA-VM, not decided here** (it is MOGA-VM's call):
the committed configuration becomes a field of `Function`, and the
walk-order and tile knobs stay variables until a pass needs them as
sub-choices of the realization alternative.

## Needs the user (SS2/SS3)

The choices SS2's and SS3's plans left open, with the options weighed.
**All five were decided by the user on 2026-10-06 as recommended**
(recorded under "Decisions" as N-S1 to N-S6).

| # | Question | Options | Recommendation |
|---|---|---|---|
| N-S1 | the random-number generator (N-8 deferred it to SS2; it blocks SS2.2) | (a) `rand` 0.9 + `rand_chacha` 0.9 (ChaCha12): two new normal dependencies of the published crate, already in `Cargo.lock` through `proptest`; fhy-core still writes its own `below` and `shuffle` to keep its stream across `rand` releases. (b) In house: SplitMix64 (or PCG64), about 120 lines, no dependency, the stream fhy-core's alone. Either way the binding exposes the same `Rng` to Python, and Python's `random.Random` streams cannot be reproduced | **(b)**, SplitMix64, with `Rng::ALGORITHM` naming it so a later generator is additive (see "Determinism and the random-number generator") |
| N-S2 | Python `==` of `Trace` and `Objective` (N-7 made everything identity but `ConfigurationKey`) | (a) structural `==` and `hash` for both: a trace is a point's data (MOGA-VM compares points' coordinates, and caches dedupe them), an objective a vocabulary value used as a dictionary key; (b) identity, as N-7 | **(a)**; `Measurement`, `Recorder` and the domains stay identity |
| N-S3 | `Estimate`, the declared-estimate role of `Metric` | (a) defer: nothing produces or reads one, and one is a `Measurer<Configuration>` over an expression when it appears; (b) `Estimate { objective, expression }` on a `Space` in SS3, as this design first sketched | **(a)** |
| N-S4 | multi-objective comparison | (a) `Measurement::dominates` alone; (b) nothing: MOGA-VM's harness is single-objective (MOGA is its machine model); (c) a Pareto front and ranking utilities | **(a)**: generic, about 30 lines, and what any multi-objective search builds on |
| N-S5 | measurement statuses | (a) `Ok`, `Infeasible`, `Failed`, `Timeout`; (b) `Ok`, `Failed`, `Timeout`, with infeasibility folded into `Failed` | **(a)**: MOGA-VM's harness separates a rejected configuration (data about the space) from a defect (`harness.py:26-37`) |
