# Search space: a generic search-space vocabulary for fhy-core, from MOGA-VM's

- **Status:** designed 2026-10-05 on `feat/search-space` (`6548984`), revised
  the same day for the user's decisions. Every decision is recorded under
  "Decisions". Implementation starts with SS0.
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
- **Trace steps: structs.** A step is data: a decision, its domain's shape,
  and a coordinate. The open parts are the decision *kind*, an open
  vocabulary (an interned tag, as `OpAttribute` is), and, in SS2, dynamic
  domains (`Domain::Custom(Part<dyn CustomDomain>)`) and the `SearchOracle`
  trait that answers steps.
- **Conditions and forbidden clauses: structs** over `constraint::Constraint`,
  whose `Custom` variant is already fhy_core's extension point for
  Python-defined and Rust-defined constraints.

## Scope and slicing

| Slice | Contents |
|---|---|
| **SS0** | identifier values and members in `fhy_core::constraint` (N-5 (a)) |
| **SS1** (recommended first (C) slice) | `Variable`, `Alternative` (traits and plain implementations), `Choice`, `Space` with hierarchy, `Condition`, `Forbidden`, `Configuration` with validation, completeness and `ConfigurationKey`; equivalence; serialization; the Python interface with both extension paths |
| **SS2** | `Trace` and `TraceStep` over the static space *and* over dynamic decisions (choice, order and strided-run domains, an open decision kind); `SearchOracle`, uniform, recording and replay oracles; enumeration and cardinality; sampling and mutation hooks |
| **SS3** | `Objective`, `Direction`, `Measurement`, a `Measurer` trait, and declared estimates (N-C3) |
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
  counts as unassigned for every later check.
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
  `MAX_VALUE_DEPTH`.)

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
- `tests/it/search_space_golden.rs`: the oracle corpus, below.

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
| SS2: `search_space/trace.rs`, `domain.rs`, `oracle.rs` | the trace and the stream |
| SS3: `search_space/measurement.rs` | `Objective`, `Direction`, `Measurement`, `Measurer` |

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

## `Trace` and MOGA-VM's decision stream (SS2)

- **A trace is the stream, recorded.** `TraceStep` generalizes MOGA-VM's
  `RecordedDecision` (`cir/lowering/search/decisions.py:474-500`):
  - the decision's name (MOGA's `subject`);
  - an open kind, an interned tag replacing MOGA's closed `DecisionKind`
    (`decisions.py:93-120`);
  - the domain's shape and cardinality (what `ReplayOracle` checks,
    `oracle.py:287-317`);
  - the coordinate (the replayable half);
  - the value (for reading back).
- **Two kinds of step.**
  - A step about a decision of the `Space` takes its domain from the space:
    a choice's alternatives, or a finite variable's values.
  - A step about a decision the space does not declare (MOGA's addresses
    and boundary namespaces) carries its own domain: a choice, order or
    strided-run domain, ported from `ChoiceDomain`, `OrderDomain` and
    `AddressDomain`.
- **Replay** is positional and by coordinate, as `ReplayOracle` is.
  `Space::replay` turns the static steps of a trace into a `Configuration`;
  dynamic steps are answered back to the pipeline.
- **`SearchPoint`** becomes `Trace`. `ExtractedSearchSpace` becomes
  unnecessary: the static prefix MOGA-VM extracts (option per entry, walk
  order and tile per option) *is* a `Space`, with hierarchy where MOGA-VM
  had `AxisCondition`.

## `Measurement` and where `Metric` goes (SS3)

- `Metric` disappears from the space (N-C3). It was declared on options and
  selections, and nothing produced or read it.
- Its kind becomes `Direction`: COST is `Minimize`, BENEFIT is `Maximize`,
  DIAGNOSTIC is `Report`.
- Its name becomes an `Objective`, a shared vocabulary, as the audit
  recommended (F-SS-012).
- Measured values are `Measurement`s keyed by `ConfigurationKey`, with a
  status (feasible, infeasible, failed).
- A declared estimate, an expression over variable names, becomes an
  `Estimate { objective, expression }` attached to a `Space` and evaluated
  against a complete configuration into a `Measurement` marked estimated.
  It lands in SS3, not SS1.

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
| `cir/lowering/search`: `RecordedDecision`, `SearchPoint`, `DecisionKind` | SS2 `TraceStep`, `Trace`, an interned kind | SS2 | `search_space.trace` |
| `cir/lowering/search`: `ChoiceDomain`, `OrderDomain`, `AddressDomain`, `SearchOracle`, `ReplayOracle` | SS2 domains, `SearchOracle`, oracles | SS2 | |

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
| `lowering/search` | its own stream | SS2's `Trace` and oracles; `ExtractedSearchSpace` becomes the `Space` |
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

## Equivalence plan

The oracle (MOGA-VM `3d93ba3` on fhy_core v0.1.8, with stand-ins, as the
audit ran it) still defines the behavior (C) keeps.

- **Golden corpus:** `rust/fhy-core/tests/golden/record_search_space_cases.py`,
  run by hand with `--moga-vm-src` naming the oracle's import paths, writes
  `search_space_cases.json`, which `tests/it/search_space_golden.rs`
  replays. It is a recorder, not a `generate_*.py` generator: the drift
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
11. **SS2** (designed separately): `Trace`, domains, oracles, enumeration
    and cardinality, sampling and mutation.
12. **SS3** (designed separately): `Objective`, `Direction`,
    `Measurement`, `Measurer`, `Estimate`.

## Decisions

All decided by the user on 2026-10-05, except N-C5 and D-SS-3's resolution,
decided on 2026-10-06 for SS1. Nothing is open.

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
| N-8 | dependencies | none new |
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
