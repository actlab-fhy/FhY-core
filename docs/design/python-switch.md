# Switching the Python package to the Rust implementation

- **Status:** plan, with decisions signed off 2026-09-24 (see "Decisions").
- **Scope:** how each concept already ported to `fhy-core` becomes the
  implementation behind the Python API when the Rust backend is selected, and
  the patterns every later port follows.
- **Related:** `docs/design/rust-workspace.md` (the crate's design) and
  CONTRIBUTING "Porting to Rust".

## Goal

`import fhy_core` exposes one Python API. With the Rust backend selected
(`fhy_core.RUST_BACKEND_SELECTED`), each switched concept runs on the Rust
implementation. With `FHY_CORE_NO_EXTENSIONS=1`, it runs on the pure-Python
implementation. Existing Python code, including user subclasses of the
framework classes, keeps working unchanged on both backends. The existing
Python test suite, run on both backends (the nox `tests` sessions), is the
equivalence harness.

A switched concept is *defined in both languages* in the sense of decision 3.
Its Python-visible behavior (names, signatures, exceptions, messages, reprs,
pickles) must be identical on both backends. The binding crate
(`fhy-core-py`) adapts the idiomatic Rust core to that Python API. The core
crate does not bend to Python.

## The three binding patterns

Every concept uses exactly one of these. Which one is recorded in the slice
table below.

### P1: Python value class, converted at the boundary

The class stays a pure-Python class on both backends. When Rust code needs
one, the binding converts it by value, in both directions (a `FromPyObject`
and an `IntoPyObject` implementation in `fhy-core-py`).

- **When:** a small immutable value whose conversion is lossless and cheap,
  and which carries no registry or identity state that would then live twice.
- **Why:** attribute reads, `==` and `hash` stay plain Python speed. The
  measured reason `Identifier` stayed in Python still holds.
- **Example: `Identifier`.** Both backends share one id counter and identity
  is by id, so converting through `(id, name_hint)` is lossless. The Python
  class is rebuilt through its deserialization path, which never issues a new
  id.

### P2: Rust-backed class

The Python class is a `#[pyclass]` holding the Rust value (usually an
`Arc`-backed handle, so it's cheap to clone).

- **Python protocols.** A pyclass cannot inherit from a Python class. So
  whatever the framework needs from Python bases (`Serializable`,
  `FrozenMixin`, `StructuralEquivalence`, `Visitable`, ...) is provided one of
  two ways:
  - the public class is a thin Python subclass of the pyclass that mixes the
    Python protocols in; or
  - the pyclass implements the protocol methods itself and is registered as a
    virtual subclass (`Serializable.register(...)`).

  Pick per class. Prefer the pure pyclass when no Python base carries state.
- **Class hierarchies.** A Python class hierarchy, such as `Expression` and
  its node classes, becomes a `#[pyclass(subclass)]` base with
  `#[pyclass(extends = Base)]` node classes. `isinstance` checks against the
  node classes keep working.
- **When:** the value is behavior-rich or performance-relevant, a Rust
  container must hold it, or it carries registry or identity state. For
  identity state, the Rust registry becomes the only registry under the Rust
  backend, as CONTRIBUTING's "switch in one step" rule requires.
- **Canonical identity.** For interned types, the binding keeps a cache from
  the canonical Rust handle to its Python object, so `a is b` holds for equal
  canonical values (CONTRIBUTING "Canonical values keep their identity in
  Python").

### P3: Rust trait exposed as a Python abstract class

For framework traits users implement in Python: `CompilerPass`, `Analysis`,
`Validator`, rewrite `Rule`s, tree visitors.

- **Rust base.** A `#[pyclass(subclass)]` base implements the shared
  behavior in Rust. Its `#[new]` accepts `*args, **kwargs`, so Python
  subclasses with their own `__init__` construct.
- **Python ABC.** The public Python class is
  `class X(_rs.XBase, abc.ABC, Generic[...])` and declares the abstract hooks.
  Abstract methods are enforced; this was verified with PyO3 0.29 on
  CPython 3.12.
- **Python implementations.** A Python subclass is driven from Rust through
  an adapter: a Rust struct holding `Py<PyAny>` that implements the Rust
  trait by calling the Python methods under the GIL. A raised exception
  becomes the hook's error; `PyErr` boxes as `PassFailure`.
- **Rust implementations.** A Rust implementation is an
  `#[pyclass(extends = XBase)]` class, registered with the ABC
  (`X.register(...)`). The base holds its native Rust implementation, so a
  pipeline runs it without calling back into Python.
- **Type erasure.** Python has no generics at runtime. The binding therefore
  uses one type-erased IR, a `PyIr(Py<PyAny>)` newtype that implements
  `NodeHandle` with identity from the object pointer. Rust passes over a
  concrete IR (`Expression`) get an adapter that extracts and rewraps it.
- **Borrowed Rust state never reaches Python.** A hook receives an *owned*
  context object that collects diagnostics and forwards analysis requests.
  Its contents are moved into the Rust context when the hook returns, and it
  is invalidated so a retained reference raises instead of mutating stale
  state. The crate forbids `unsafe`, so the "lend `&mut` to Python" trick is
  unavailable.
- **Granularity.** Python callbacks happen per pass hook, never per tree
  node. Per-node visitors are Rust-native; Python gets whole-pass hooks and
  pattern rules.

## Cross-cutting rules for every switch

1. **Multiplexing lives in the Python module** that defines the concept:
   `if IS_RUST_BACKEND_SELECTED: <Rust-backed> else: <pure Python>`, as
   `identifier.py` does for its counter. Exactly one implementation, and one
   registry per concept, is live in a process.
2. **Parity is checked by the existing Python tests on both backends.** A
   test that exercises the pure-Python internals (private attributes,
   implementation details) is rewritten against the public API, or marked
   backend-specific with a reason. It is never deleted silently.
3. **Messages and exception classes.** The binding raises the same Python
   exception class, with the same message, as the pure-Python
   implementation, through one `IntoPyErr` implementation per core error
   type. The core crate's `Display` text stays Rust-style.
4. **Serialization.** A switched class keeps the Python wire format it has
   today, because Python payloads cross both backends. The binding produces
   the `__type__`/`__data__` envelope and the Python field shapes (decision 4:
   the envelope lives in the binding). Pickles load under either backend.
5. **Benchmark before switching.** Before a class switches, a benchmark
   compares both backends on its hot paths: construction, attribute access,
   `==`, `hash`, and the module's main operations. It lives in the
   `benchmarks/` directory with the harness from S0. A switch that makes a
   hot path slower either changes pattern (for example to P1) or is recorded
   as an accepted cost, with numbers.
6. **Stubs.** Every pyclass and function added to `_rs` goes into
   `src/fhy_core/_rs.pyi`, which `tests/test_rs_stub.py` checks.
7. **Bottom-up.** A concept switches only after everything it holds has
   switched, or is P1-converted.

## Slices, in order

| Slice | Concepts | Pattern | Depends on | Payoff |
|---|---|---|---|---|
| S0 | Binding infrastructure: error-mapping trait (exists), envelope helper, identity cache, `PyIr`, benchmark harness, stub conventions | - | - | Every later slice |
| S1 | `Identifier` | P1 (conversion traits only; the class stays Python) | S0 | Rust objects can hold identifiers |
| S2 | `OpAttribute`, `NoteKind`, `ValueDomain` (interned tags) | P2 with the identity cache | S1 | One registry per concept; prerequisite for diagnostics |
| S3 | `Note`, `Diagnostic`, `DiagnosticLevel`, `ValidationReport`, `Provenance` | P2 (`DiagnosticLevel`: P1 enum conversion) | S2 | Rust passes and analyses can report to Python |
| S4 | `Expression` and its node classes, operations, literals, builtins, printing | P2 hierarchy | S1, S3 | Fast expression construction, equality, hashing and substitution; Rust IR for Rust passes |
| S5 | Patterns and rewrite rules | P2 for patterns, P3 for callbacks and `Rule` | S4 | Rust rewrite walk from Python |
| S6 | `CompilerPass`, `Analysis`, `Validator`, `PassManager`, `ValidationManager`, verification | P3 (traits), P2 (managers) | S3, S4 | Rust-native passes appear in Python as `CompilerPass` subclasses; mixed pipelines run in Rust |

Each slice follows the port workflow:

1. Benchmark the Python implementation.
2. Write the binding and the multiplexing, with the Python suite green on
   both backends.
3. Compare the benchmark.
4. Commit per concept.

`Identifier`'s serialization, and the id-counter parity that already exists,
are the model to copy.

## Decisions (2026-09-24)

1. **The pure-Python backend is temporary.** It stays until every concept is
   switched and trusted, then the pure-Python implementations are deleted.
   Two-backend parity is a transition requirement, not a permanent one.
2. **Pattern choice is evidence-based.** For each class, pick P1 or P2 from
   its benchmark and from how much logic it carries:
   - Logic-rich machinery with many moving parts goes to Rust (P2/P3), for
     example the pass infrastructure.
   - A simple value class stays Python (P1) when the benchmark shows Python
     attribute access, `==` and `hash` beat crossing into the extension.
     `Provenance` is decided this way.
   - `Identifier` stays P1, per the existing measured decision.
   - The registry rule overrides speed: a type with registry or identity
     state is P2, so only one registry is live.

   Each slice records the numbers behind its choice.
3. **Benchmarks use `pytest-benchmark`** (a new dev dependency), in a
   `benchmarks/` directory. A nox session runs them on both backends.
4. **Tests.** When a concept switches, its behavioral tests are rewritten as
   Rust tests in `rust/fhy-core/tests/`, which become the specification.
   Python keeps a small interface suite for the concept, proving that the
   Python API over the Rust implementation works: construction, the main
   operations, exceptions, pickling and serialization. The full Python
   behavioral tests stay in place until the pure-Python backend is deleted,
   since they are the parity check during the transition.

## S0 baseline (2026-09-24, 2ad1e75)

Median time per call of each benchmark in `benchmarks/`, from
`uv run nox -s "benchmark-3.11(backend='python')"
"benchmark-3.11(backend='rust')"`, run back to back on Python 3.11.13 with
pytest-benchmark 5.3.0 at its default rounds. Every benchmarked class is
still pure Python on both backends; only `Identifier`'s id counter runs in
Rust under the Rust backend, and only identifier construction and
deserialization touch it. The machine (an Intel Core i9-7920X, 24 threads)
is shared and had a load average of about 4 during the runs, so the numbers
are indicative. Differences of a few percent between the columns are noise,
as is the `OpAttribute` hash row, whose code is identical on both backends.
A slice compares against these numbers only after rerunning both backends
on the same machine.

| Benchmark | python | rust |
|---|--:|--:|
| `test_identifier_construction` | 4.22 µs | 4.06 µs |
| `test_identifier_id_access` | 74 ns | 74 ns |
| `test_identifier_name_hint_access` | 74 ns | 75 ns |
| `test_identifier_eq` | 99 ns | 102 ns |
| `test_identifier_hash` | 101 ns | 101 ns |
| `test_identifier_dict_lookup` | 97 ns | 97 ns |
| `test_identifier_deserialize_from_dict` | 3.64 µs | 3.51 µs |
| `test_identifier_pickle_round_trip` | 9.00 µs | 9.04 µs |
| `test_interned_tag_construction_of_existing_key[OpAttribute]` | 16.8 µs | 16.9 µs |
| `test_interned_tag_construction_of_existing_key[NoteKind]` | 16.7 µs | 17.0 µs |
| `test_interned_tag_construction_of_existing_key[ValueDomain]` | 16.7 µs | 16.9 µs |
| `test_interned_tag_lookup[OpAttribute]` | 668 ns | 659 ns |
| `test_interned_tag_lookup[NoteKind]` | 665 ns | 658 ns |
| `test_interned_tag_lookup[ValueDomain]` | 658 ns | 665 ns |
| `test_interned_tag_eq[OpAttribute]` | 157 ns | 160 ns |
| `test_interned_tag_eq[NoteKind]` | 162 ns | 160 ns |
| `test_interned_tag_eq[ValueDomain]` | 152 ns | 153 ns |
| `test_interned_tag_hash[OpAttribute]` | 286 ns | 202 ns |
| `test_interned_tag_hash[NoteKind]` | 198 ns | 195 ns |
| `test_interned_tag_hash[ValueDomain]` | 200 ns | 200 ns |
| `test_value_domain_is_subdomain_of_root` | 87.5 µs | 87.1 µs |
| `test_value_domain_is_subdomain_of_unrelated` | 87.0 µs | 86.9 µs |
| `test_note_construction` | 2.02 µs | 2.07 µs |
| `test_note_eq` | 155 ns | 154 ns |
| `test_note_str` | 345 ns | 352 ns |
| `test_diagnostic_construction` | 1.15 µs | 1.16 µs |
| `test_diagnostic_eq` | 296 ns | 293 ns |
| `test_validation_report_construction` | 854 ns | 904 ns |
| `test_validation_report_build_of_100_diagnostics` | 326.7 µs | 328.5 µs |
| `test_validation_report_eq` | 27.8 µs | 27.9 µs |
| `test_validation_report_format` | 39.9 µs | 39.2 µs |
| `test_span_construction` | 7.37 µs | 7.50 µs |
| `test_unknown_provenance_construction` | 1.76 µs | 1.75 µs |
| `test_file_provenance_construction` | 2.07 µs | 2.17 µs |
| `test_named_provenance_construction` | 2.10 µs | 2.07 µs |
| `test_call_site_provenance_construction` | 2.08 µs | 2.07 µs |
| `test_fused_provenance_construction` | 2.07 µs | 2.05 µs |
| `test_provenance_eq[unknown]` | 91 ns | 91 ns |
| `test_provenance_eq[file]` | 154 ns | 155 ns |
| `test_provenance_eq[named]` | 267 ns | 266 ns |
| `test_provenance_eq[call_site]` | 594 ns | 597 ns |
| `test_provenance_eq[fused]` | 511 ns | 508 ns |
| `test_provenance_hash[unknown]` | 104 ns | 105 ns |
| `test_provenance_hash[file]` | 561 ns | 566 ns |
| `test_provenance_hash[named]` | 659 ns | 658 ns |
| `test_provenance_hash[call_site]` | 914 ns | 927 ns |
| `test_provenance_hash[fused]` | 915 ns | 930 ns |
| `test_provenance_fuse_of_two` | 3.60 µs | 3.58 µs |
| `test_provenance_fuse_with_reductions` | 4.57 µs | 4.49 µs |
| `test_compiler_pass_execute` | 13.6 µs | 13.6 µs |
| `test_pass_manager_run_of_5_passes` | 381.3 µs | 373.5 µs |
| `test_analysis_manager_cache_hit` | 9.23 µs | 9.22 µs |

Two rows stand out before any switch. `ValueDomain.is_subdomain_of` takes
about 87 µs on a four-domain chain: each step up the chain runs a
structural-equivalence check of about 22 µs, most of it spent in
`isinstance` checks against runtime-checkable protocols. Constructing a tag
whose key is already registered takes about 17 µs.

## S2: interned tags (`OpAttribute`, `NoteKind`, `ValueDomain`)

**Pattern: P2.** The registry rule requires it: after S2, the Rust registry is
the only registry for these three concepts on the Rust backend.

### Shape

- **`fhy-core-py` pyclasses.** One `#[pyclass(subclass, frozen)]` per concept
  wraps the canonical Rust handle (`Canonical<OpAttribute>`,
  `Canonical<NoteKind>`, `Canonical<ValueDomain>`). It implements in Rust:
  - `name`, `description` and `parent` (`ValueDomain` only);
  - `get_identifier`, `get_intern_key`, `is_subdomain_of`;
  - `__eq__` and `__hash__` by key, `__repr__`;
  - the class methods `get_interned`, `require_interned` and
    `construct_from_fields`;
  - `serialize_to_dict` and `deserialize_from_dict`, in exactly today's
    Python payload shapes. `ValueDomain` nests its parent as today; the Rust
    core's flat wire form stays internal to Rust serde.
- **The public Python classes.** On the Rust backend, `OpAttribute`,
  `NoteKind` and `ValueDomain` are thin Python subclasses of those pyclasses:
  - they set `__slots__ = ()`, so instances stay immutable;
  - they mix in the stateless Python protocols the classes have today
    (`HasIdentifier`, the structural/alpha/derived equivalence mixins,
    `Serializable` through `register_serializable` with the same type ids);
  - they never mix in `InternedMixin` itself. Instead they are registered as
    virtual subclasses of `InternedMixin` (and `FrozenMixin`), so
    `isinstance` checks still hold.

  Mutating an attribute raises `FrozenMutationError` with today's message.
- **Construction.** `OpAttribute(name, description)`,
  `NoteKind(name, description)` and
  `ValueDomain(name, description, parent=None)` register through the Rust
  registry.
  - **Canonical identity (D-S2-4).** When the key exists, construction
    returns the canonical Python object itself, so `a is b` holds for equal
    keys. A per-concept identity cache in the binding maps each canonical
    key to its single Python object, so `get_interned(key) is DATA_DOMAIN`.
  - **Descriptions.** A new description for an existing key is ignored. The
    first one wins (decision D-4 of the crate spec).
- **`Identifier` (P1).** It converts through `(id, name_hint)`. The binding
  restores the Rust identifier through a now-public
  `Identifier::try_restore(id, name_hint) -> Result<Identifier, IdOutOfRange>`
  in the core crate. It caches the Python `Identifier` object on each tag,
  so reading `.name` doesn't rebuild it.

### Backend multiplexing

- **Where it happens.** `op_attribute.py`, `value_domain.py` and the
  `NoteKind` part of `diagnostic.py` each choose their implementation with
  `if IS_RUST_BACKEND_SELECTED:`. The pure-Python classes stay unchanged for
  the Python backend.
- **Shipped constants.** The module constants (`COMMUTATIVE`, `ASSOCIATIVE`,
  `PURE`, `ELEMENTWISE`, `DATA_DOMAIN`, `ADDRESS_DOMAIN` and the four note
  kinds) are the Rust shipped tags on the Rust backend.

### Decisions (signed off 2026-09-24)

- **D-S2-1: `clear_interned_registry` and `register_default_instances` are
  unsupported on the Rust backend.** They raise `NotImplementedError`, with
  a message saying the Rust registries are append-only. The Python tests
  that use them become Python-backend-only, with a skip reason. Their
  behavior is covered by the Rust tests over local registries.
- **D-S2-2: shipped tags use the reserved ids on both backends.** The
  pure-Python backend builds its shipped tags' identifiers with the fixed ids
  from `fhy_core::identifier::reserved`, through a private helper in
  `identifier.py` that mirrors the deserialization path and never touches
  the counter. A shipped tag, and a pickle of one, is then identical across
  backends. This revises R-3 of the crate spec.
- **D-S2-3: `ValueDomain` follows Rust semantics on the Rust backend.**
  Constructing a domain whose name is registered with a different parent
  raises the conflict error; Python's own deserialization already raises
  `DeserializationValueError` for this. Domains compare by name. The Python
  tests that expect the silent shadow or `(name, parent)` equality become
  Python-backend-only, with a reason.
- **D-S2-4: construction of an existing key returns the canonical object**
  on the Rust backend. The Python backend keeps returning a fresh, equal,
  non-canonical object until it is deleted.

### Tests

- **During the transition, the whole Python suite passes on both backends.**
  The exceptions are tests that pin D-S2-1 or D-S2-3 behavior; those are
  skipped on the Rust backend, with a reason naming the decision.
- **Python interface tests.** A small new file,
  `tests/test_interned_tags_rust_binding.py`, covers the Python interface
  over the Rust implementation:
  - construction and canonical identity;
  - `get_interned` and `require_interned`;
  - payload round trips and pickles;
  - frozen errors;
  - the `NotImplementedError` of D-S2-1;
  - the conflict error of D-S2-3.
- **Rust behavior tests** already exist in `rust/fhy-core/tests/it/`. New
  Rust tests cover only behavior the binding adds, such as the Python
  payload shapes, which are exercised through Python.
- **Benchmarks.** Run `benchmarks/test_interned_tags.py` on both backends
  before and after. Record the results here.

### Status

Implemented on 2026-09-24 in four commits: the public
`Identifier::try_restore` (fad791b), the binding (a114adc), the backend
multiplexing (d39df06) and the interface tests (a8669dc).

### Benchmarks (before and after)

Median time per call from `uv run --python 3.11 nox -s
"benchmark-3.11(backend='python')" "benchmark-3.11(backend='rust')"`, run
back to back on the S0 machine with Python 3.11.13: "before" at 10565ab,
"after" at a8669dc. The machine is shared, so differences of a few percent,
and all differences between the two "before" columns, are noise. The table
lists the interned-tag benchmarks and the `Note` ones, the other benchmarks
that touch a tag; every other benchmark is unchanged within noise on both
backends.

| Benchmark | python before | rust before | python after | rust after | rust after / rust before |
|---|--:|--:|--:|--:|--:|
| `test_interned_tag_construction_of_existing_key[OpAttribute]` | 15.9 µs | 16.1 µs | 19.8 µs | 317 ns | 0.02 |
| `test_interned_tag_construction_of_existing_key[NoteKind]` | 15.8 µs | 16.2 µs | 16.6 µs | 316 ns | 0.02 |
| `test_interned_tag_construction_of_existing_key[ValueDomain]` | 16.3 µs | 16.3 µs | 16.0 µs | 310 ns | 0.02 |
| `test_interned_tag_lookup[OpAttribute]` | 630 ns | 621 ns | 633 ns | 136 ns | 0.22 |
| `test_interned_tag_lookup[NoteKind]` | 627 ns | 625 ns | 637 ns | 144 ns | 0.23 |
| `test_interned_tag_lookup[ValueDomain]` | 636 ns | 511 ns | 647 ns | 145 ns | 0.28 |
| `test_interned_tag_eq[OpAttribute]` | 151 ns | 152 ns | 154 ns | 74 ns | 0.49 |
| `test_interned_tag_eq[NoteKind]` | 179 ns | 240 ns | 158 ns | 74 ns | 0.31 |
| `test_interned_tag_eq[ValueDomain]` | 191 ns | 146 ns | 152 ns | 67 ns | 0.46 |
| `test_interned_tag_hash[OpAttribute]` | 340 ns | 190 ns | 191 ns | 57 ns | 0.30 |
| `test_interned_tag_hash[NoteKind]` | 334 ns | 189 ns | 211 ns | 57 ns | 0.30 |
| `test_interned_tag_hash[ValueDomain]` | 331 ns | 199 ns | 213 ns | 57 ns | 0.29 |
| `test_value_domain_is_subdomain_of_root` | 93.1 µs | 84.7 µs | 89.3 µs | 67 ns | 0.001 |
| `test_value_domain_is_subdomain_of_unrelated` | 83.1 µs | 84.2 µs | 83.1 µs | 68 ns | 0.001 |
| `test_note_construction` | 2.0 µs | 1.9 µs | 1.9 µs | 2.0 µs | 1.02 |
| `test_note_eq` | 146 ns | 146 ns | 150 ns | 146 ns | 1.00 |
| `test_note_str` | 346 ns | 336 ns | 347 ns | 325 ns | 0.97 |

Every tag hot path is faster on the Rust backend, so no path needs a
pattern change or an accepted cost:

- Construction of a registered key drops from about 16 µs to about 0.3 µs:
  it reads the key's id and returns the cached canonical object without
  touching the Rust registry.
- `is_subdomain_of` drops from about 85 µs to about 70 ns: the walk runs in
  Rust and compares names, instead of running a Python structural
  equivalence check per level.
- Lookup, `==` and `hash` each cost one call into the extension. The `==`
  benchmark compares a canonical tag with "a distinct, equal tag", which on
  the Rust backend is the canonical tag itself (D-S2-4).
- `Note` construction, `==` and `str` hold a tag but call none of its
  methods on their hot path, except `str`, which now calls the Rust
  `NoteKind.__str__`; all three are unchanged.

A first "after" run, started while the machine's load average was about 8,
measured `test_diagnostic_construction` at 8.4 µs on the Rust backend. The
rerun above measured 1.1 µs, and that path never touches a tag, so the
outlier was load.

### Implementation notes

Choices the plan above left open, made while implementing S2:

- **Core additions.** Only `Identifier::try_restore` became public, with a
  rustdoc example as its test. The binding needed nothing else: the tags'
  `register` functions, `Interned::intern_registry`, `Canonical` and the
  `ValueDomainConflict` accessors were public already.
- **Two-step construction.** A `PyO3` `#[new]` either builds a new instance
  of the requested subtype or returns an existing object, never both. So
  each pyclass's own `__new__` only builds an instance from a private seed
  class that the binding creates and does not export, and the public class
  sets `__new__ = staticmethod(_rs.X._new_canonical)`. `_new_canonical`
  returns the cached canonical object, or registers the tag in Rust and
  builds its object through `_rs.X.__new__(cls, seed)`. No Python code can
  build a second object for a canonical tag.
- **Identity cache.** One per concept, a `Mutex<HashMap<u64, Py<PyAny>>>`
  keyed by the name's identifier id; the lock is never held across a call
  into Python. Its objects live for the rest of the process, as registry
  entries do. An object is created as an instance of the class the call
  came through (`cls`). S3 will return tags that Rust code reached, such as
  `Note.kind`, with no class at hand, so it will need to register the
  public classes with the binding.
- **Fast path.** A cached key returns the cached object without touching
  the Rust registry, since the first registration wins either way. A value
  domain takes the fast path only when the cached domain's parent has the
  requested parent's name; anything else goes through `register_root` or
  `register_child`, so the core reports every conflict.
- **Python protocols.** `DerivedEquivalenceMixin` needs a dataclass, so the
  Rust-backed classes mix in `StructuralEquivalence` and
  `AlphaEquivalenceMixin` and implement `is_structurally_equivalent` and
  `is_alpha_equivalent_under` in Rust as "same class and same name", which
  is what the derived plan computes for these fields. They are virtual
  subclasses of `InternedMixin` and `FrozenMixin` (so `Note`'s frozen field
  check accepts `NoteKind`), and implement those mixins' remaining members
  themselves: `is_frozen` is always true, `freeze`, `assert_frozen` and
  `register_interned_instance` do nothing, and `__setattr__` and
  `__delattr__` raise `FrozenMutationError` with the mixin's message. The
  subclasses do get an instance `__dict__` from their Python mixins, so
  `__slots__ = ()` alone would not block new attributes.
- **Errors.** Constructing a value domain under a conflicting parent
  (D-S2-3) raises `ValueError` with the Rust `ValueDomainConflict` text,
  through `IntoPyErr`: `value_domain` is a Rust-defined module, and
  construction had no Python error to match. Decoding the same conflict,
  through `construct_from_fields` or `deserialize_from_dict`, raises the
  `DeserializationValueError` message of `InternedMixin`, which is
  dual-defined. Malformed payloads raise the serialization framework's own
  errors because the binding raises them through the framework: it builds
  `DeserializationDictStructureError(cls, expected, data)` itself, and
  decodes names through `Identifier.deserialize_from_dict`. A description
  that differs from the canonical one is logged through the
  `fhy_core.traits.interned` logger, in the Python format.
- **Stricter arguments.** On the Rust backend a name must be an
  `Identifier` and a description a `str`, and each raises `TypeError`
  otherwise; the dataclasses accepted any value. `get_interned` of a key
  that is not an identifier returns `None`, or raises `TypeError` for an
  unhashable key, as a dict lookup does.
- **Hashes.** A tag hashes to its name's id on the Rust backend. The
  dataclasses hash their field tuple. Hash values are not part of the
  parity contract; equal tags still hash equally.
- **Pickles.** The pure-Python classes gained the Rust-backed classes'
  `__reduce__`, `(Class.deserialize_from_dict, (payload,))`, so a shipped
  tag pickles to the same bytes on both backends and any tag's pickle loads
  under either. Unpickling and copying now return the canonical instance on
  the Python backend too, where they used to return a fresh equal copy.
- **Reserved table.** `identifier.py` holds the table as one
  `_RESERVED_<entry>` constant per entry of `reserved.rs`, named after the
  Rust entry, and a test parses `reserved.rs` and requires the same entries.
  The built-in expression constants `pi`, `e`, `inf` and `nan` were the
  first identifiers the counter issued after the shipped tags, so their
  pinned ids move from 65,544 to 65,547 down to 65,536 to 65,539 on both
  backends. A payload that names them, or a shipped tag, by an old id no
  longer resolves to them; that follows from D-S2-2.
- **Typing.** The modules branch on
  `TYPE_CHECKING or not IS_RUST_BACKEND_SELECTED`, so type checkers see the
  pure-Python classes, which describe the public API, and mypy does not
  check the Rust branch. `_rs.pyi` declares the pyclasses. It types
  `_new_canonical` loosely, because `tests/test_rs_stub.py` would count a
  module-level type variable as a stub name the extension lacks.
- **Tests skipped on the Rust backend.** Six behavioral tests, each marked
  with the decision it pins: the three registry-reset tests, one in
  `test_op_attribute.py` and two in `test_value_domain.py` (D-S2-1),
  `test_value_domain_unequal_when_parents_differ` (D-S2-3), and the two
  `..._first_constructed_with_key_is_canonical` tests, which assert that a
  second construction returns a new object (D-S2-4). D-S2-4 was not listed
  among the skip reasons above, but those two tests contradict it directly;
  the new interface suite asserts the Rust behavior instead.

## S3: diagnostics and provenance

### Pattern choices (from the S0 baseline)

| Concept | Pattern | Why |
|---|---|---|
| `DiagnosticLevel` | P1 | A three-value `StrEnum`. It converts by value in both directions (`"error"`, `"warning"`, `"info"`) |
| `Note` | P2 | Holds a `NoteKind`, which is Rust-backed since S2. Construction is 2.0 µs in Python |
| `Diagnostic` | P2 | S6's Rust pass infrastructure produces diagnostics. Construction is 1.15 µs and `==` 296 ns in Python |
| `ValidationReport` | P2, generic over Python records | Building a report of 100 diagnostics takes 327 µs in Python, `==` 27.8 µs, `format` 39.9 µs. The binding stores `ValidationReport<Py<PyAny>>`, because records are arbitrary Python objects (the same exception as `PyIr`: framework payloads supplied by Python) |
| `ValidationFailedError` | stays a Python exception class | The binding raises it with the report attached and the message Python uses today |
| `Position`, `Span`, the `Provenance` family (S3b) | P2 hierarchy, unless its before/after benchmark shows a regression | Nothing in `src` depends on it. Its operations take 0.1–7 µs in Python |

### Shape

This follows S2's shape:

- **Classes.** `#[pyclass(subclass, frozen)]` classes wrap the Rust values.
  The public classes are thin Python subclasses that mix in the stateless
  Python protocols: `Serializable` with the same type ids, `FrozenMixin` or
  `EqualMixin` membership, and `WrappedFamilySerializable` for the
  provenance family.
- **Python API unchanged.** Constructor signatures, keyword names, fields,
  properties and methods are exactly today's, including `Diagnostic.message`
  (a `Note`), `message_text`, `ValidationReport.format()`, `errors()`,
  `raise_if_failed()`, `Provenance.fuse(sources, metadata=None)`,
  `FusedProvenance.metadata` and `FileProvenance.file_path` (a
  `pathlib.Path`). The binding maps them onto the Rust API:
  - `metadata` maps to `label`;
  - Span's four keyword arguments map to Rust's typed constructors.
- **Text.** `ValidationReport.format()`, `str`/`repr`, and the exception
  messages produce Python's current text: `[ERROR] source: message` and the
  `No validation diagnostics.` placeholder. The Rust core's `Display` stays
  Rust-style (decision 3 at the binding boundary).
- **Rust objects handed back to Python.** When Rust returns a value such as
  `Note.kind`, a diagnostic inside a report, or a provenance child, the
  binding builds the *public* Python class. Each public class registers
  itself with the binding once, at import. This follows up S2's
  implementation note. Canonical tags come from S2's identity cache.
- **Payloads and pickles.** Today's Python payload shapes are kept, and
  pickles load across backends.

### Tests and benchmarks

These are as in S2:

- the whole Python suite passes on both backends, and every skip names a
  decision;
- a new interface test file covers each concept;
- benchmarks run before and after, and the results are recorded here.

### S3a status

S3 lands in two steps. S3a switches `DiagnosticLevel` (P1), `Note`,
`Diagnostic` and `ValidationReport` (P2); `ValidationFailedError` stays a
Python exception class. S3b switches the provenance family; see "S3b
status" below.

S3a was implemented on 2026-09-24 in seven commits: the new benchmarks
(6d0a28d), the public-class registration with S2's tags moved onto it
(2bd613a), the binding (27d449a), the backend multiplexing (50712a1), the
interface tests (5ad125d), and two fixes of field-read regressions the
benchmarks found (02e90cc, 40eaa23). The whole Python suite passes on both
backends with no test skipped or changed; the only new skips are the new
interface suite's, which runs on the Rust backend only.

### S3a benchmarks (before and after)

Median time per call, from `uv run --python 3.11 nox -s
"benchmark-3.11(backend='python')" "benchmark-3.11(backend='rust')"`
restricted with `-k` to the diagnostic, pass-infrastructure, identifier
and some provenance benchmarks, on the S0 machine with Python 3.11.13.
"Before" is 38092cd plus the benchmarks of 6d0a28d, "after" is 40eaa23.
The machine was shared, with a load average of 12 to 18, so each
configuration ran three times, interleaved (before and after alternating
on each backend), and the table lists the best of the three medians; a
single configuration's medians spread by up to a factor of two between
runs. The identifier and provenance benchmarks, whose code S3a does not
change, came out within 5% of 1.00 after/before on both backends, except
`test_provenance_eq[named]` at 1.30 on the Rust backend only, whose code
is identical in both trees. The 3.8 µs "rust before" note construction is
also noise: the same code measured 2.3 µs on the Python backend and in the
S2 run.

| Benchmark | python before | rust before | python after | rust after | rust after / rust before |
|---|--:|--:|--:|--:|--:|
| `test_note_construction` | 2.3 µs | 3.8 µs | 2.3 µs | 300 ns | 0.08 |
| `test_note_eq` | 176 ns | 176 ns | 176 ns | 80 ns | 0.46 |
| `test_note_str` | 388 ns | 371 ns | 388 ns | 321 ns | 0.87 |
| `test_note_hash` | 420 ns | 297 ns | 422 ns | 102 ns | 0.34 |
| `test_note_attribute_access` | 89 ns | 92 ns | 90 ns | 93 ns | 1.01 |
| `test_diagnostic_construction` | 1.3 µs | 1.3 µs | 1.3 µs | 496 ns | 0.39 |
| `test_diagnostic_eq` | 330 ns | 329 ns | 332 ns | 90 ns | 0.27 |
| `test_diagnostic_hash` | 544 ns | 409 ns | 553 ns | 144 ns | 0.35 |
| `test_diagnostic_attribute_access` | 126 ns | 139 ns | 127 ns | 122 ns | 0.88 |
| `test_validation_report_construction` | 988 ns | 1 µs | 999 ns | 14.5 µs | 14.21 |
| `test_validation_report_build_of_100_diagnostics` | 371 µs | 387 µs | 367 µs | 107 µs | 0.28 |
| `test_validation_report_eq` | 31.5 µs | 31.3 µs | 31.7 µs | 1.1 µs | 0.04 |
| `test_validation_report_format` | 43.9 µs | 44.6 µs | 44.2 µs | 3.6 µs | 0.08 |
| `test_validation_report_errors` | 10.2 µs | 10.1 µs | 10.1 µs | 721 ns | 0.07 |
| `test_validation_report_has_errors` | 459 ns | 464 ns | 459 ns | 63 ns | 0.13 |
| `test_compiler_pass_execute` | 15.3 µs | 16.1 µs | 15.6 µs | 14.2 µs | 0.89 |
| `test_pass_manager_run_of_5_passes` | 431 µs | 424 µs | 428 µs | 424 µs | 1.00 |
| `test_analysis_manager_cache_hit` | 10.5 µs | 10.3 µs | 10.6 µs | 10.5 µs | 1.02 |

Every hot path but one is faster on the Rust backend, or unchanged within
noise:

- **Construction.** A note drops from about 2.3 µs to 0.3 µs and a
  diagnostic from 1.3 µs to 0.5 µs: construction is a type check and a
  copy of the strings, with no dataclass `__init__`, `FrozenMixin.__new__`
  or field-type bookkeeping. Building 100 notes, 100 diagnostics and a
  report of them drops from about 370 µs to 107 µs.
- **Equality, hashing and the report operations** run in Rust: `==` of
  two reports of 100 diagnostics drops from 31 µs to 1.1 µs, `format()`
  from 44 µs to 3.6 µs, `errors()` from 10 µs to 0.7 µs.
- **Field reads** cost the same as the dataclass's, after the fix
  described in the implementation notes; the first version measured them
  at 1.5 to 2 times slower.
- **The pass infrastructure** is unchanged within noise: a pass run
  creates no diagnostics on the benchmarked paths.

**Regression: constructing a report from 100 existing diagnostics** takes
about 14.5 µs instead of 1 µs. The dataclass stores the tuple it is given;
the Rust class also builds the `ValidationReport<Py<PyAny>>` the plan
calls for, which clones each Rust `Diagnostic` (three or four `String`
allocations each) out of its pyclass, and collects the records into a
`Vec`. It is about 145 ns per diagnostic, paid once per report, and it is
what makes the report's later operations 10 to 30 times faster; a report
built together with its diagnostics, the realistic path, is still 3.5
times faster overall. Options, for the maintainer to decide:

1. accept it as a recorded cost (CONTRIBUTING "Replacing a Python class");
2. build the Rust report lazily, on the first operation that needs it,
   which moves the cost rather than removing it;
3. for reports built in Python, keep only the tuples and run
   `errors()`, `format()` and `==` over the Rust diagnostics borrowed
   from each `Diagnostic` pyclass, with no clone, keeping
   `ValidationReport<Py<PyAny>>` for the reports S6 produces in Rust.

Option 3 would make every path faster than the dataclass, at the price of
two representations in one class. S6 is the natural point to decide,
since it adds the Rust-produced reports.

### S3a implementation notes

Choices the plan above left open, made while implementing S3a:

- **Core additions.** None. The binding uses the public API of
  `fhy_core::diagnostic` as it is: `Note::new`/`with_other_kind`,
  `Diagnostic::new`/`with_detail` and its getters, and
  `ValidationReport::new`, `diagnostics` and `has_errors`. The core's
  `Display` impls never reach Python; the binding renders `str`, `repr`
  and `format()` itself.
- **Public-class registration.** Each pyclass has a private class method
  `_register_public_class`, which the public class calls once, at import,
  right after its definition. It fills a write-once `PublicClass` slot
  (`rust/fhy-core-py/src/public_class.rs`, a `PyOnceLock<Py<PyType>>`);
  registering the same class again does nothing, and registering another
  class raises `RuntimeError`, so the class a Rust-built value becomes
  never changes. CONTRIBUTING's "Process-global state" section records the
  slot next to the identity caches. `OpAttribute` and `NoteKind` register
  too: the described-tag macro's `to_python` now takes an optional class
  and falls back to the registered one, which is how `Note.kind` builds
  the object of a canonical kind that has none yet. `ValueDomain` does not
  register yet, because nothing hands a domain back from Rust before S4.
- **What the pyclasses hold.** Each holds the Rust value next to the
  Python objects its fields return, as the S2 tags do. `Note` holds the
  Rust `Note`, the message `str` and its kind's single Python object;
  every Python `NoteKind` is canonical, so the object passed in is the
  one kept, and an omitted kind is `OTHER_NOTE_KIND` from the identity
  cache. `Diagnostic` holds the Rust `Diagnostic`, the `DiagnosticLevel`
  member, the `Note` object and the source and detail objects it was
  built from, so `diagnostic.message is note` holds as it does for the
  dataclass. `ValidationReport` holds the planned
  `ValidationReport<Py<PyAny>>` next to the diagnostic and record tuples
  it was given: `report.diagnostics is diagnostics` and
  `report.errors()[0] is diagnostics[0]` hold, and `errors()`,
  `warnings()` and `infos()` pick from the tuple by index. A first
  version built each field's object on every read; the benchmarks showed
  field reads at 1.5 to 2 times the dataclass, so the objects are kept
  (commits 02e90cc and 40eaa23). No path in S3a creates a diagnostic
  or a report in Rust, so the builders that turn a Rust `Diagnostic` or
  `ValidationReport` into the registered public classes arrive with S6,
  the first slice that produces them; the classes register now, so S6
  changes only Rust.
- **`DiagnosticLevel` (P1).** The orphan rule forbids `FromPyObject` for
  a core type, so, as for `Identifier`, the conversion is two functions.
  They share a table built once from `DiagnosticLevel(level.as_str())`
  for each Rust level. A member converts by identity; anything else goes
  through `DiagnosticLevel(value)`, so `"error"` is accepted and stored as
  the member, and any other value raises the enum's own `ValueError`
  (`'fatal' is not a valid DiagnosticLevel`).
- **Stricter arguments.** On the Rust backend a note's `message` must be a
  `str` and its `kind` a `NoteKind`, a diagnostic's `message` a `Note`,
  its `source` a `str` and its `detail` a `str` or `None`, and a report's
  diagnostics `Diagnostic`s. Each raises `TypeError` with a message in
  S2's style (`Note kind must be a NoteKind, got NoneType.`); the
  dataclasses accepted anything. An explicit `kind=None` raises instead of
  meaning the default: the binding tells an omitted argument from `None`.
  A report stores any iterable as a tuple, where the dataclass kept a list
  as a list. A missing required argument names `Note.__new__()` instead of
  `Note.__init__()` in PyO3's `TypeError`.
- **Python protocols.** `EqualMixin` and `PartialEqualMixin` carry no
  state, so the public classes inherit them, and
  `supports_equality`/`supports_partial_equality` answer as before
  (`Diagnostic` and `ValidationReport` still lack `supports_equality`).
  `FrozenMixin` has slots, so, as in S2, the classes are virtual
  subclasses and implement `is_frozen`, `freeze`, `assert_frozen` and the
  frozen `__setattr__`/`__delattr__` in Rust. `ValidationReport` keeps
  `FrozenMixin`'s one carve-out: `typing` may store `__orig_class__` for
  `ValidationReport[int](...)`. The classes set `__match_args__` as the
  dataclasses do.
- **Equality and hashing.** `==` returns `NotImplemented` unless both
  objects have exactly the same class, as a dataclass's `__eq__` does, and
  then compares the Rust values; a report also compares its record tuples
  with Python `==`. Hashes come from Rust's `DefaultHasher`, and a report's
  hash includes its record tuple's Python hash, so a report with an
  unhashable record still raises `TypeError`. Hash values differ from the
  dataclasses'; equal objects still hash equally.
- **Payloads.** Only `Note` is serializable, as before. Its payload nests
  the kind's payload, which the tag macro now builds from the canonical
  tag in an associated function. Decoding checks the structure with the
  same helper S2 uses, decodes the kind through the registered
  `NoteKind.deserialize_from_dict`, and calls `cls(message, kind)`, so
  malformed payloads raise the framework's `DeserializationDictStructureError`
  for `Note` or `NoteKind` with the pure-Python text. `construct_from_fields`
  is `Serializable`'s default, which calls the class.
- **Pickles.** All three classes pickle as a call of their class with
  their fields, `(type(self), fields)`, on both backends: the pure-Python
  dataclasses gained that `__reduce__`, because their default pickles
  restore slot or dict state that the Rust classes cannot take. A note's
  kind pickles as S2's canonical-kind payload. A pickle written by an
  older version on the Python backend still loads there, but not on the
  Rust backend.
- **`ValidationFailedError`.** `raise_if_failed` imports the Python class
  once and raises `ValidationFailedError(report)`, so the message is the
  class's own `report.format()` and `.report` is the report itself.
- **Tests.** No existing test was skipped or changed. The new
  `tests/test_diagnostic_rust_binding.py` (55 tests, Rust backend only)
  covers the class structure and registration, the argument checks, the
  reprs and `format()` text, equality and hashing, the note payload and
  its structure errors, `raise_if_failed`, the frozen errors and the
  `__orig_class__` carve-out, and pickles, including a round trip through
  a Python-backend subprocess.

### S3b status

S3b switches `Position`, `Span`, `Provenance` and its five variant
classes to pattern P2, as a class hierarchy; `HasProvenance` stays a
Python protocol. It was implemented on 2026-09-24 in seven commits: the
new benchmarks (af5b1cd), the helpers shared with the diagnostics binding
(653375b), the binding (f509ffd), the backend multiplexing (44b5596), the
interface tests (9219e89), and a fix of a `str` regression the first
benchmark run found (1babb93). The whole Python suite passes on both
backends with no test skipped or changed; the only new skips are the new
interface suite's, which runs on the Rust backend only.

### S3b benchmarks (before and after)

Median time per call, from `uv run --python 3.11 nox -s
"benchmark-3.11(backend='python')" "benchmark-3.11(backend='rust')"`
restricted with `-k "provenance or span or position"`, on the S0 machine
with Python 3.11.13. "Before" is af5b1cd, which adds the field-read,
`str`, `repr`, ordering and round-trip benchmarks to the S0 set; "after"
is 1babb93. The machine was shared, with a load average of about 18, so
each configuration ran three times, interleaved as in S3a, and the table
lists the best of the three medians. Within a configuration the medians
agreed to a few percent, but whole configurations were offset from each
other: the "python before" and "python after" columns measure identical
code throughout, and the "rust before" column, which also measures the
dataclasses, is up to twice as fast as both. The field-read rows show it
most, at 95 ns against 178 ns for the same dataclass read. So the ratio
column overstates the cost of every cheap path; compare the "rust after"
column with "python after" as well.

| Benchmark | python before | rust before | python after | rust after | rust after / rust before |
|---|--:|--:|--:|--:|--:|
| `test_span_construction` | 15.4 µs | 13.2 µs | 15.7 µs | 1.51 µs | 0.11 |
| `test_unknown_provenance_construction` | 3.73 µs | 3.10 µs | 3.69 µs | 319 ns | 0.10 |
| `test_file_provenance_construction` | 4.39 µs | 4.20 µs | 4.32 µs | 1.01 µs | 0.24 |
| `test_named_provenance_construction` | 4.50 µs | 3.86 µs | 4.55 µs | 724 ns | 0.19 |
| `test_call_site_provenance_construction` | 4.40 µs | 3.74 µs | 4.43 µs | 785 ns | 0.21 |
| `test_fused_provenance_construction` | 4.46 µs | 3.80 µs | 4.24 µs | 810 ns | 0.21 |
| `test_provenance_eq[unknown]` | 237 ns | 167 ns | 221 ns | 191 ns | 1.15 |
| `test_provenance_eq[file]` | 368 ns | 344 ns | 367 ns | 204 ns | 0.59 |
| `test_provenance_eq[named]` | 571 ns | 582 ns | 562 ns | 228 ns | 0.39 |
| `test_provenance_eq[call_site]` | 1.25 µs | 975 ns | 1.24 µs | 256 ns | 0.26 |
| `test_provenance_eq[fused]` | 1.25 µs | 1.01 µs | 1.25 µs | 273 ns | 0.27 |
| `test_provenance_hash[unknown]` | 254 ns | 193 ns | 249 ns | 219 ns | 1.13 |
| `test_provenance_hash[file]` | 1.18 µs | 940 ns | 1.18 µs | 394 ns | 0.42 |
| `test_provenance_hash[named]` | 1.38 µs | 1.08 µs | 1.37 µs | 431 ns | 0.40 |
| `test_provenance_hash[call_site]` | 1.93 µs | 1.55 µs | 1.94 µs | 591 ns | 0.38 |
| `test_provenance_hash[fused]` | 1.97 µs | 1.53 µs | 1.94 µs | 613 ns | 0.40 |
| `test_provenance_fuse_of_two` | 7.06 µs | 6.26 µs | 7.27 µs | 1.18 µs | 0.19 |
| `test_provenance_fuse_with_reductions` | 9.26 µs | 7.89 µs | 9.11 µs | 1.82 µs | 0.23 |
| `test_position_construction` | 4.66 µs | 4.13 µs | 4.65 µs | 377 ns | 0.09 |
| `test_position_attribute_access` | 177 ns | 96 ns | 178 ns | 198 ns | 2.06 |
| `test_position_lt` | 370 ns | 201 ns | 367 ns | 181 ns | 0.90 |
| `test_span_attribute_access` | 247 ns | 126 ns | 245 ns | 266 ns | 2.11 |
| `test_span_str` | 1.36 µs | 752 ns | 1.41 µs | 705 ns | 0.94 |
| `test_provenance_attribute_access[file]` | 178 ns | 95 ns | 175 ns | 190 ns | 2.00 |
| `test_provenance_attribute_access[named]` | 188 ns | 95 ns | 188 ns | 201 ns | 2.12 |
| `test_provenance_attribute_access[call_site]` | 175 ns | 95 ns | 177 ns | 200 ns | 2.12 |
| `test_provenance_attribute_access[fused]` | 190 ns | 95 ns | 187 ns | 192 ns | 2.02 |
| `test_provenance_str[unknown]` | 214 ns | 97 ns | 203 ns | 346 ns | 3.58 |
| `test_provenance_str[file]` | 1.60 µs | 1.02 µs | 1.86 µs | 897 ns | 0.88 |
| `test_provenance_str[named]` | 2.20 µs | 1.35 µs | 2.48 µs | 1.03 µs | 0.76 |
| `test_provenance_str[call_site]` | 2.87 µs | 1.76 µs | 3.27 µs | 1.22 µs | 0.69 |
| `test_provenance_str[fused]` | 3.36 µs | 2.14 µs | 3.86 µs | 1.32 µs | 0.62 |
| `test_provenance_repr[unknown]` | 832 ns | 499 ns | 937 ns | 605 ns | 1.21 |
| `test_provenance_repr[file]` | 4.55 µs | 2.85 µs | 5.13 µs | 4.14 µs | 1.45 |
| `test_provenance_repr[named]` | 5.41 µs | 3.50 µs | 6.09 µs | 5.01 µs | 1.43 |
| `test_provenance_repr[call_site]` | 7.89 µs | 5.15 µs | 9.02 µs | 7.47 µs | 1.45 |
| `test_provenance_repr[fused]` | 8.51 µs | 5.50 µs | 9.80 µs | 8.23 µs | 1.50 |
| `test_provenance_dict_round_trip[unknown]` | 8.40 µs | 5.48 µs | 9.65 µs | 3.70 µs | 0.68 |
| `test_provenance_dict_round_trip[file]` | 65.8 µs | 46.5 µs | 76.5 µs | 34.1 µs | 0.73 |
| `test_provenance_dict_round_trip[named]` | 94.9 µs | 66.4 µs | 107.6 µs | 59.9 µs | 0.90 |
| `test_provenance_dict_round_trip[call_site]` | 137.8 µs | 137.9 µs | 157.4 µs | 86.8 µs | 0.63 |
| `test_provenance_dict_round_trip[fused]` | 146.5 µs | 144.9 µs | 176.5 µs | 72.4 µs | 0.50 |

Because of that offset, the same operations were also timed in one
process: a Rust-backend interpreter loaded a second copy of the
pure-Python classes (the module's Python branch, executed without type
registration) and alternated the two implementations of each operation
under `timeit`, seven rounds of 20,000 calls, keeping the best. Those
ratios are the basis for the verdict:

| Operation | dataclass | Rust-backed | ratio |
|---|--:|--:|--:|
| `Position(2, 5)` | 4.5 µs | 389 ns | 0.09 |
| `Span(...)`, four bounds | 5.7 µs | 619 ns | 0.11 |
| `NamedProvenance(...)` | 4.3 µs | 607 ns | 0.14 |
| position fields | 176 ns | 176 ns | 1.00 |
| span fields | 224 ns | 225 ns | 1.01 |
| provenance fields (file, named, call site, fused) | 212–214 ns | 217–220 ns | 1.02–1.03 |
| position `<` | 310 ns | 160 ns | 0.52 |
| span `str` | 876 ns | 413 ns | 0.47 |
| unknown `==`, `hash` | 162, 171 ns | 140, 163 ns | 0.86, 0.95 |
| unknown `str` | 79 ns | 98 ns | 1.24 |
| unknown `repr` | 799 ns | 403 ns | 0.50 |
| file, named, call site, fused `==` | 1.5–2.8 µs | 202–250 ns | 0.09–0.13 |
| file, named, call site, fused `hash` | 1.1–1.9 µs | 415–649 ns | 0.31–0.38 |
| file, named, call site, fused `str` | 1.8–3.9 µs | 651 ns–1.0 µs | 0.27–0.36 |
| file, named, call site, fused `repr` | 5.2–10.0 µs | 3.8–7.7 µs | 0.72–0.77 |
| `Provenance.fuse` of two | 7.5 µs | 1.2 µs | 0.16 |

(The provenance `==` rows compare two separately built trees; the nox
benchmark compares trees that share their path and span objects, which
the dataclass's tuple comparison short-cuts by identity.)

Every hot path but one is faster on the Rust backend, or unchanged within
noise, so no path needs a pattern change and provenance stays P2:

- **Construction** drops by 4 to 10 times: a type check per argument and
  a Rust value, with no dataclass `__init__`, `__post_init__`,
  `FrozenMixin.__new__` or field-type bookkeeping. A file provenance's
  benchmark includes reading the path's text, and a span's includes its
  two positions.
- **`==`, `hash`, `str` and `fuse`** run in Rust: 3 to 10 times faster on
  the composite provenances.
- **Field reads** cost what a dataclass's cost, since the variant classes
  keep their field objects as struct members.
- **`repr`** is somewhat faster: it still calls each field's `repr`, and
  a `pathlib.Path`'s is Python code.
- **The dict round trip** is 1.1 to 2 times faster: the envelope and the
  derived decoding's `construct_from_fields` stay Python.

**The one slower path: `str` of an unknown provenance**, 98 ns against
79 ns. The dataclass's `__str__` is a Python function returning a
constant; the Rust one returns one interned `str` (1babb93 cut it from
about 3 times the dataclass's cost), and the rest is the cost of
entering the extension. It is about 20 ns per call, on a path that
renders no information; the maintainer may accept it as a recorded cost
(CONTRIBUTING "Replacing a Python class") or give the public
`UnknownProvenance` a Python `__str__` returning the constant.

### S3b implementation notes

Choices the plan above left open, made while implementing S3b:

- **Core additions.** None. The binding uses the public API of
  `fhy_core::provenance`: `Position::try_new`, the `Span` builders,
  `FileProvenance::new`, `NamedProvenance::try_new`,
  `CallSiteProvenance::new`, `FusedProvenance::new` and `labelled`, and
  `Provenance`'s `Eq`, `Hash` and `Display`. It builds a span bound by
  bound (`with_start_offset`, `with_end_offset`, then the positions), so
  the core reports the pair out of order that the dataclass's
  `__post_init__` reports first.
- **`str` from the core's `Display`.** Unlike the diagnostics, the
  provenance types' `Display` impls render exactly the Python text, and
  their rustdoc specifies it, so `__str__` returns the core's rendering
  instead of a second renderer in the binding. The suites of both backends
  pin the text. The unknown provenance returns one interned `str`: the
  first benchmark run measured the formatted `String` at several times
  the dataclass's constant (1babb93).
- **The hierarchy.** `_rs.Provenance` is a `#[pyclass(subclass, frozen)]`
  base holding the Rust `Provenance` and the tree's depth, and implements
  equality, hashing, the frozen members, `unknown` and `fuse`. Each variant
  is an `#[pyclass(extends = PyProvenance, subclass, frozen)]` class
  holding its field objects as `#[pyo3(get)]` `Py` members, which Python
  reads as struct members, as S3a's classes do. The public classes are
  `Provenance(_rs.Provenance, WrappedFamilySerializable, EqualMixin, ABC)`,
  which keeps `__str__` abstract, and `X(_rs.X, Provenance)` for each
  variant. That MRO puts the public `Provenance` between a variant's `_rs`
  class and `_rs.Provenance`, so each variant's `_rs` class defines its
  own `__str__`: one defined on `_rs.Provenance` would be shadowed by the
  abstract stub. `_rs.Provenance` has no constructor, so `Provenance()`
  raises `PyO3`'s `TypeError` ("No constructor defined") where the Python
  backend raises the ABC's `TypeError`.
- **The envelope comes from the Python mixin.** The public classes
  inherit `serialize_to_dict` and `deserialize_from_dict` from
  `WrappedFamilySerializable`, which writes and reads
  `{"__type__": .., "__data__": ..}` and resolves the type id through the
  Python registry, and the `_rs` classes implement only
  `serialize_data_to_dict` and `deserialize_data_from_dict`. `Position`
  and `Span` are plain `Serializable`s and implement both dict methods.
  The data payloads are built from the field objects; a nested
  serializable field is encoded by its own `serialize_to_dict`, as the
  derived codec does. Decoding mirrors the derived path: the structure
  check (`FieldShape` gained `Int`, `OptionalInt`, `OptionalStr` and
  `PayloadList`), nested values decoded through the registered public
  classes (`Position`, `Span`, and `Provenance` for children, as the
  derived codecs decode through the annotated classes), then
  `cls.construct_from_fields`, with a `ValueError` or `TypeError` outside
  the serialization hierarchy wrapped as `DeserializationValueError` with
  its message, in a new shared helper `construct_from_decoded_fields`.
- **Parity check.** Besides the suites, a differential probe printed, on
  each backend, the reprs, `str`, dict, JSON and binary payloads,
  round-trips and pickles of every shape, `fuse` results with their debug
  log lines, and every validation and malformed-payload error with its
  class and message; the two outputs were identical. The new suite
  compares the JSON payloads with a pure-Python-backend subprocess.
- **No seed construction.** Provenance has no canonical identity, so each
  class's own `__new__` is its constructor, and S2's seed and
  `_new_canonical` pattern, which exists because a `PyO3` `#[new]` cannot
  return an existing object, is not needed. The values the binding
  creates in Rust, `unknown()` and `fuse`'s results, are built by calling
  the registered public class.
- **Registration and Rust-created children.** All eight classes register
  their public class at import. In S3b, every child a provenance returns
  (`child`, `callee`, `caller`, `sources`) is the Python object it was
  built from, and nothing builds a provenance tree in Rust: the only
  provenances built in Rust are `fuse`'s and `unknown()`'s results, whose
  children are the inputs. As in S3a, the builder from a Rust
  `Provenance` tree to the public classes arrives with the first slice
  that produces trees in Rust, S4, and changes only Rust.
- **`fuse` walks the Python objects.** It does not call the core's
  `Provenance::fuse`. It walks the argument objects with the same explicit
  stack and the same flattening rule, reading each object's Rust variant,
  so a single survivor is returned itself and a fused result's sources
  are the input objects, as in Python; no Python object is rebuilt (a
  `pathlib.Path` alone costs about 1 µs); and the debug log reports
  Python's counts, which the core does not expose. `metadata` becomes the
  Rust label. The property tests of `fuse` run on both backends.
- **File paths.** `file_path` may be a `str` or an `os.PathLike` whose
  `__fspath__` returns a `str`; a `PurePath` is read through `str`. The
  core normalizes the text; its rules match `pathlib.PurePosixPath`,
  checked on Python 3.11 to 3.14 for every path of up to four tokens from
  `a`, `b`, `/`, `.`, `..`, `~`, a space, a backslash and `C:`, and for
  random longer ones. The field returns the given `PurePath` when its text
  is already normal, so `provenance.file_path is path` holds for a
  `Path`, as for the dataclass, and otherwise a new `pathlib.Path` of the
  normal text. The dataclass stored whatever it was given, so a `str`
  path now comes back as a `Path`, and `FileProvenance("a")` equals
  `FileProvenance(Path("a"))`. A path whose text is not valid UTF-8 (a
  surrogate-escaped name) raises `UnicodeEncodeError`, since the core
  stores a `String`. On Windows, the core treats a backslash as an
  ordinary character and compares case-sensitively, so a `WindowsPath`
  does not normalize as `PureWindowsPath` does; CI runs POSIX only.
- **Stricter arguments.** `__post_init__`'s errors keep their classes and
  messages, in Python's check order. Beyond them, on the Rust backend: a
  line, column or offset above `2**64 - 1` raises `OverflowError` (the
  dataclasses accepted any `int`); a span's positions must be `Position`s
  or `None`, a file provenance's span a `Span` or `None`, a name a `str`,
  a child, callee, caller, source or `fuse` argument a `Provenance`, and
  `metadata` a `str` or `None`. Each raises `TypeError` in S2's style
  (`NamedProvenance child must be a Provenance, got NoneType.`). The
  dataclasses accepted anything, and a span failed only when it compared
  two positions of another class. `sources` may be any iterable and is
  stored as a tuple.
- **Equality and hashing.** `==` returns `NotImplemented` unless both
  objects have exactly the same class, as a dataclass's does, and then
  compares the Rust values. Below the top level, the Rust values compare
  by variant, so a child of a user subclass of a variant equals the same
  child of the variant itself, where the dataclasses compared the classes
  at every level. Hashes come from `DefaultHasher`; equal values still
  hash equally.
- **Recursion guard.** Equality, hashing and `Display` recurse on the Rust
  stack. Each provenance records its tree's depth, and those three raise
  `RecursionError` when it exceeds `sys.getrecursionlimit()`, which the
  binding reads only for trees deeper than 64 levels. The dataclasses
  raise `RecursionError` near the same depth. Without the guard, `str` of
  a 200,000-level chain crashed the interpreter with a segmentation
  fault (measured with the recursion limit raised to let it through).
  `repr`, payloads and pickles recurse through Python calls, which
  Python's own limit guards, and deallocating such a chain is safe. With
  a raised recursion limit, a tree deep enough can still overflow the
  stack, as a C-level recursion in the dataclasses can.
- **Pickles.** All eight classes pickle as a call of their class with
  their fields, `(type(self), fields)`, on both backends; the pure-Python
  dataclasses gained that `__reduce__`, as S3a's did. A pickle written by
  an older version on the Python backend still loads there, but not on the
  Rust backend.
- **Dataclass machinery.** On the Rust backend the classes are not
  dataclasses, so `dataclasses.fields`, `replace` and `asdict` do not
  apply to them. Nothing in `src` uses them on provenance.
- **Shared helpers.** The argument checks, dataclass equality, hashing and
  iterable-to-tuple helpers of S3a's `diagnostic.rs` moved to a new
  `rust/fhy-core-py/src/dataclass.rs`, which both bindings use (653375b).
- **Tests.** No existing test was skipped or changed. The new
  `tests/test_provenance_rust_binding.py` (102 tests, Rust backend only)
  covers the hierarchy and registration, the argument checks and their
  messages, path normalization, the reprs, equality, ordering and pattern
  matching, the payload shapes and errors, `fuse`'s identity and log, the
  recursion guard, the frozen errors, and pickles, including payloads and
  pickles checked against a Python-backend subprocess.

## S4: expressions, then retiring the pure-Python backend

Survey: the S4 API diff (kept in the session notes). It found that Python
expressions differ from the Rust core in ways S2/S3 did not:

- identity `==`/`hash` against Rust's structural equality;
- child object identity;
- binary `LOGICAL_AND`/`LOGICAL_OR` against Rust's n-ary `Logical` node;
- literal spellings against Rust's normalized literals;
- a runtime function registry;
- binder-frame alpha renaming;
- Python text in error messages.

Making the Rust backend reproduce all of that would turn Rust nodes into
shells around Python objects.

### Decisions (signed off 2026-09-25)

- **D-S4-1: Rust semantics.** On the Rust backend the Python expression API
  takes the Rust core's semantics, and the consumers and tests are updated to
  match. Parity with the pure-Python implementation is not kept for
  expressions. In particular:
  - **Equality.** `==` and `hash` are structural.
  - **Literals are normalized.** `.value` is a `bool`, `int`, `float`, or
    `decimal.Decimal`, and spellings such as `"05"` or `"1.50"` are not
    kept.
  - **Logical connectives.** A new n-ary `LogicalExpression` class, with a
    `LogicalOperation` of `AND` or `OR`, replaces
    `BinaryExpression(LOGICAL_AND/LOGICAL_OR, ...)`.
  - **Calls.** Built-in function names are reserved, per the crate's D-9.
  - **Text.** Printed text (`str`, `repr`, `pformat_expression`) follows
    the Rust display conventions (`true`, `NaN`, `(a && b && c)`).
  - **Errors.** Expression errors map onto Python exception classes through
    `IntoPyErr`.
- **D-S4-2: some Python names stay.** Where the meaning is the same, the
  Python name stays: `BinaryOperation.MODULO`, `CallExpression.function_name`
  as the callee's name, the node class names, and the visitor suffixes.
- **D-S4-3: alpha renaming with binder frames moves to Rust.** The frame
  stack and its capture rules from `term/alpha_equivalence.py` are ported to
  Rust `AlphaRenaming`, with Rust tests. `RegisteredFunction` and `Param`
  keep comparing under frames.
- **D-S4-4: the function registry stays Python for now.** It is
  `symbolic/expression/registry`, with runtime registration, until the
  modules that own it are ported. The Rust Boolean-position screen reads it
  through a `SortLookup` adapter in the binding.
- **D-S4-5: payloads keep the `__type__`/`__data__` envelope,** because
  Python containers embed expressions. The data inside follows the new
  semantics, for example a `logical_expression` type id and normalized
  literals. No reader for the old format is needed.
- **D-S4-6: the pure-Python backend is retired right after S4.**
  - The extension becomes mandatory.
  - `FHY_CORE_NO_EXTENSIONS` and the backend parametrization of the nox
    sessions go.
  - The pure-Python implementations kept only for that backend are deleted:
    the S2/S3 parity classes, the Python id counter, and the Python
    expression core.

  Modules not yet ported stay ordinary Python on top of the Rust types.

### Steps

1. **S4.1: expression benchmarks.** Add `benchmarks/test_expression.py`
   and record a baseline on the current implementation.
2. **S4.2: Rust core additions,** with Rust tests. These are the frame-based
   `AlphaRenaming` and anything else the binding needs.
3. **S4.3: switch expressions onto Rust.** Add the binding, the Python
   module on the Rust backend, and the migration of the consumers and tests
   to Rust semantics:
   - the solver, type checker, `types/dispatch.py`,
     `constraint/ordering.py`, and the sympy/z3/numpy/evaluator/inline
     passes;
   - the `mock_identifier` fixture, which becomes real `Identifier`s.

   The Rust-backend suite must be green. The pure-Python backend is expected
   to break for expression code from here until S4.4.
4. **S4.4: retire the pure-Python backend** per D-S4-6, and update
   CONTRIBUTING, the README, CI and nox.
5. **Benchmarks after,** recorded here.

### S4.1 baseline (2026-09-25, 6faac85 plus the new benchmarks)

`benchmarks/test_expression.py` covers the hot paths of the survey's
benchmark gaps: construction per node kind and through the builders and
operators, field reads and `isinstance` dispatch, `==`, `hash`,
structural and alpha equivalence on a deep tree and on a shared DAG,
`substitute`, free identifiers, `pformat_expression`, a `VisitablePass`
walk, the Boolean-position screen, the dict, JSON and pickle round trips.
The deep trees are 100 operations over four identifiers, mixing unary,
binary, identifier and literal nodes; the shared DAG stacks 10 additions of
a node to itself (11 distinct nodes, 2,047 occurrences).

The benchmarks call only API that D-S4-1 keeps, and assert no result that
D-S4-1 changes: `==` of two distinct, equal trees is timed but not checked
(identity and false today, structural and true after S4.3). The one call
whose node shape changes, building a conjunction, sits in one helper,
`_build_conjunction`, marked `D-S4-1` for S4.3.

Median time per call, from `uv run --python 3.11 nox -s
"benchmark-3.11(backend='python')" "benchmark-3.11(backend='rust')" --
-k test_expression` on the S0 machine with Python 3.11.13 and
pytest-benchmark 5.3.0. The load average was 5 to 10, so each backend ran
three times, interleaved, and the table lists the best of the three
medians; every row's three medians agreed within 30%. The expression code
is pure Python on both backends today, so the two columns measure the same
code and their differences are noise.

| Benchmark | python | rust |
|---|--:|--:|
| `test_identifier_expression_construction` | 894 ns | 861 ns |
| `test_literal_expression_construction[int]` | 1.04 µs | 976 ns |
| `test_literal_expression_construction[big_int]` | 1.02 µs | 980 ns |
| `test_literal_expression_construction[float]` | 1.04 µs | 979 ns |
| `test_literal_expression_construction[integer_text]` | 1.36 µs | 1.26 µs |
| `test_literal_expression_construction[decimal_text]` | 1.60 µs | 1.49 µs |
| `test_literal_expression_construction[bool]` | 1.03 µs | 965 ns |
| `test_unary_expression_construction` | 1.06 µs | 1.01 µs |
| `test_binary_expression_construction` | 1.17 µs | 1.12 µs |
| `test_make_binary_expression` | 3.45 µs | 3.36 µs |
| `test_logical_not_construction` | 1.69 µs | 1.67 µs |
| `test_conjunction_construction` | 4.10 µs | 4.03 µs |
| `test_piecewise_construction` | 7.66 µs | 7.47 µs |
| `test_call_construction_of_a_builtin` | 4.62 µs | 4.53 µs |
| `test_call_construction_of_a_user_function` | 4.64 µs | 4.58 µs |
| `test_deep_tree_construction` | 323.8 µs | 309.9 µs |
| `test_binary_operator_of_two_expressions[add]` | 2.18 µs | 2.15 µs |
| `test_binary_operator_of_two_expressions[multiply]` | 2.18 µs | 2.14 µs |
| `test_binary_operator_of_two_expressions[true_divide]` | 2.19 µs | 2.13 µs |
| `test_binary_operator_of_two_expressions[floor_divide]` | 2.19 µs | 2.16 µs |
| `test_binary_operator_of_two_expressions[modulo]` | 2.18 µs | 2.14 µs |
| `test_binary_operator_of_two_expressions[power]` | 2.19 µs | 2.15 µs |
| `test_binary_operator_of_two_expressions[less]` | 2.18 µs | 2.13 µs |
| `test_binary_operator_of_two_expressions[greater_equal]` | 2.20 µs | 2.14 µs |
| `test_binary_operator_of_two_expressions[equals]` | 2.13 µs | 2.11 µs |
| `test_binary_operator_of_two_expressions[not_equals]` | 2.13 µs | 2.10 µs |
| `test_add_operator_with_int` | 3.65 µs | 3.52 µs |
| `test_reflected_subtract_operator_with_int` | 3.71 µs | 3.59 µs |
| `test_unary_operator[neg]` | 1.75 µs | 1.72 µs |
| `test_unary_operator[pos]` | 1.75 µs | 1.70 µs |
| `test_node_attribute_access[unary]` | 85 ns | 85 ns |
| `test_node_attribute_access[binary]` | 99 ns | 99 ns |
| `test_node_attribute_access[identifier]` | 51 ns | 51 ns |
| `test_node_attribute_access[literal]` | 51 ns | 51 ns |
| `test_node_attribute_access[piecewise]` | 99 ns | 99 ns |
| `test_node_attribute_access[call]` | 85 ns | 85 ns |
| `test_isinstance_of_node_kind` | 42 ns | 42 ns |
| `test_isinstance_dispatch_over_node_kinds` | 2.74 µs | 2.85 µs |
| `test_eq_of_one_node` | 88 ns | 88 ns |
| `test_eq_of_distinct_equal_trees` | 134 ns | 134 ns |
| `test_eq_of_distinct_equal_deep_trees` | 137 ns | 137 ns |
| `test_hash_of_small_tree` | 51 ns | 51 ns |
| `test_hash_of_deep_tree` | 51 ns | 51 ns |
| `test_dict_lookup_by_deep_tree` | 46 ns | 46 ns |
| `test_structural_equivalence_of_deep_trees` | 2.22 ms | 2.23 ms |
| `test_structural_equivalence_of_shared_dags` | 22.0 ms | 22.0 ms |
| `test_alpha_equivalence_under_free_renaming_of_deep_trees` | 2.47 ms | 2.46 ms |
| `test_substitute_in_deep_tree` | 246.9 µs | 244.3 µs |
| `test_substitute_in_shared_dag` | 2.18 ms | 2.12 ms |
| `test_free_identifiers_of_deep_tree` | 65.0 µs | 65.4 µs |
| `test_pformat_expression_of_deep_tree[symbolic]` | 917.7 µs | 915.7 µs |
| `test_pformat_expression_of_deep_tree[functional]` | 928.0 µs | 924.5 µs |
| `test_pformat_expression_of_deep_tree[show_id]` | 903.1 µs | 915.9 µs |
| `test_visitable_pass_walk_of_deep_tree` | 906.4 µs | 923.4 µs |
| `test_validate_logical_operands_of_deep_conjunction` | 1.33 ms | 1.33 ms |
| `test_validate_predicate_of_nested_piecewise` | 499.4 µs | 494.6 µs |
| `test_validate_predicate_of_comparison` | 6.75 µs | 6.68 µs |
| `test_serialize_to_dict_of_deep_tree` | 238.3 µs | 245.2 µs |
| `test_deserialize_from_dict_of_deep_tree` | 44.8 ms | 45.8 ms |
| `test_json_round_trip_of_deep_tree` | 45.8 ms | 46.8 ms |
| `test_pickle_round_trip_of_deep_tree` | 532.4 µs | 524.3 µs |

What the baseline shows for S4.3:

- **Construction** costs 0.9 to 1.6 µs per node, and 3.5 to 7.7 µs through
  a coercing builder or operator. A 100-deep tree takes about 0.3 ms.
- **Field reads, `==` and `hash`** are at the interpreter's floor (50 to
  140 ns): `==` and `hash` are identity today, so they do not depend on
  the tree's size. On the Rust semantics both become structural; the deep
  rows show what an uncached structural hash will be compared with.
- **`isinstance` misses are slow.** A hit takes 42 ns, but the node
  classes' metaclass is `typing._ProtocolMeta` (they inherit the
  `HasOperands` protocol), whose `__instancecheck__` runs Python code for
  a miss: about 560 ns each, measured with `timeit`. The dispatch cascade
  over the six classes, as the passes dispatch, therefore takes 2.7 µs.
  Pyclasses with an ordinary metaclass would answer a miss in tens of
  nanoseconds.
- **The whole-tree operations** are where Rust should win most: structural
  and alpha equivalence of two deep trees take about 2.2 and 2.5 ms, of two
  shared DAGs 22 ms (Python walks all 2,047 occurrences), `substitute`
  0.25 ms (2.1 ms on the DAG), `pformat_expression` and the counting
  visitor 0.9 ms, the screen of a 100-operand conjunction 1.3 ms, and
  decoding a deep tree from a dict or JSON 45 ms.
