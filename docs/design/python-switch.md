# Switching the Python package to the Rust implementation

- **Status:** plan, with decisions signed off 2026-09-24 (see "Decisions").
  Since S4.4 (2026-09-25) the package has one backend: it requires the
  extension, and the pure-Python implementations are deleted. The sections
  that describe two backends (the goal, the multiplexing and parity rules,
  and the slices up to S4.3) record how the switch was done.
- **Scope:** how each concept already ported to `fhy-core` becomes the
  implementation behind the Python API when the Rust backend is selected, and
  the patterns every later port follows.
- **Related:** `docs/design/rust-workspace.md` (the crate's design) and
  CONTRIBUTING "Porting to Rust".

## Progress checklist

This is kept up to date after every step, so work can resume from here if a
session ends. Environment for resuming: rustup's cargo is in
`~/.cargo/bin`, and a uv install lives in `target/tooling/pyenv` (gitignored;
recreate it with `python3.11 -m venv target/tooling/pyenv && target/tooling/pyenv/bin/pip install uv`).

- [x] S0: benchmark harness and baseline
- [x] S1: `Identifier` stays P1; conversion is in the binding
- [x] S2: interned tags
- [x] S3a: diagnostics
- [x] S3b: provenance
- [x] S4.1: expression benchmarks
- [x] S4.2: frame-based `AlphaRenaming` in Rust, plus the Python capture-rule fix
- [x] S4.3a: expressions on the Rust core with Rust semantics
- [x] S4.3b: consumers migrated. On the Rust backend: the suite is green (6,960 passed), slow tests pass (7,008), properties pass (280), lint and mypy are clean, and the Rust gate passes (2,630)
- [x] S4.4: retire the pure-Python backend. The suite is green (6,949 passed), slow tests pass (6,982), properties pass (280), lint and mypy are clean, and the Rust gate passes (2,630)
- [ ] S5: patterns and rewrite rules (designed; N-S5-1 needs the user before S5.4)
  - [ ] S5.1: pattern benchmarks and baseline
  - [ ] S5.2: core additions, if any, with Rust tests
  - [ ] S5.3: the pattern binding
  - [ ] S5.4: the Python switch
  - [ ] S5.5: pattern tests migrated, and the interface suite
  - [ ] S5.6: benchmarks after, and docs
- [ ] S6: pass infrastructure (`CompilerPass`, `Analysis`, `Validator`, managers)
- Leftovers:
  - [ ] the `ValidationReport` construction cost (S6)
  - [ ] the unknown-provenance `str` cost
  - [ ] Windows paths
  - [x] mypy over the Rust branches (S4.4)
  - [ ] slow callee-name parsing in the core
  - [ ] platform wheels in the release workflow, now that the extension is required (S4.4)

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

### S4.2 status

S4.2 ports the binder frames and capture rules of the Python
`AlphaRenaming` (`src/fhy_core/term/alpha_equivalence.py`) to the Rust
`fhy_core::expression::AlphaRenaming`, per D-S4-3. Nothing Python-visible
changes; the binding does not use the renaming yet. The Rust tests were
written first and failed against `todo!()` stubs (43 of their 46 cases;
the other three pin behavior the flat renaming already had), then passed.

### S4.2 implementation notes

- **The Rust API.** `AlphaRenaming` keeps its private fields, now a stack of
  binder frames (outermost first) over the free renaming, each an injective
  map with its image set:
  - `try_new(free_renaming: HashMap<Identifier, Identifier, S>)`, as before,
    is the renaming with no frame; `Default` maps nothing;
  - `enter_binder(&mut self, bindings: HashMap<..>) -> Result<(),
    NonInjectiveRenamingError>` pushes an innermost frame, and leaves the
    renaming unchanged when it refuses one;
  - `leave_binder(&mut self) -> bool` pops it and says whether there was
    one; `binder_depth()` counts the frames;
  - `resolve(&self, identifier) -> &Identifier` is Python's `resolve`: the
    innermost frame binding it, then the free renaming, then itself;
  - `is_corresponding(left, right)` is Python's
    `are_identifiers_alpha_equivalent` (renamed by the crate's S-7), with
    the capture rules, and `is_empty()` now means "maps no identifier in
    any frame or in the free renaming".

  Python's `extend` returns a new renaming; the Rust frame is pushed in
  place, and a caller that needs the old value clones first, as the
  binding's `extend` will. `NonInjectiveRenamingError` gained
  `#[non_exhaustive]` and `part() -> RenamingPart` (a new
  `#[non_exhaustive]` enum, `FreeRenaming` or `BinderFrame`), and displays
  `a binder frame must be injective, ...` for a frame. The part is what
  the binding needs to raise Python's two messages, which name the
  parameter (`free_renaming` or `bindings`).
- **The flat-map API is kept, not migrated.** `try_new`, `Default`,
  `is_corresponding` and `is_empty` keep their signatures and, without
  frames, their meaning, so the existing tests in
  `tests/it/expression/node_stories.rs` and `properties.rs` pass unchanged.
  `Expression::is_alpha_equivalent_under` is unchanged in code and
  compares identifiers through `is_corresponding`, so it works under
  frames; its shared-subtree shortcut still applies only when `is_empty()`
  holds, and its pair memo stays sound because an expression has no
  binders, so the renaming is the same everywhere in one comparison.
- **Divergence from Python: shadowing on the other side (needs sign-off).**
  Python's `are_identifiers_alpha_equivalent` returns at the innermost frame
  binding `left`, even when a frame inside it binds `right` on the other
  side. So it finds `\x. \y. x` alpha-equivalent to `\a. \a. a`, whose body
  refers to the inner `a`, and `\a. \a. a` not equivalent to `\x. \y. x`
  (checked on the Python backend): the relation is neither symmetric nor
  transitive there, against its own contract. The Rust rule lets the
  innermost frame that binds `left` *or* has `right` as an image decide,
  which is the de Bruijn reading, and agrees with Python everywhere else:
  - a differential run of 20,000 random cases (up to three frames and a
    free renaming over four identifiers), generated from the Python
    oracle, found `resolve` identical in all of them and `is_corresponding`
    identical in 19,819; the other 181 are exactly this case;
  - implementing Python's rule in Rust fails exactly the five tests that
    pin it (`alpha_renaming_is_corresponding_lets_an_inner_image_shadow_an_outer_binding`,
    the symmetry and transitivity tests over the example binder terms, and
    the symmetry and de Bruijn properties in `alpha_properties.rs`), and
    passes every other test.

  - the Python suites of `term`, `BinderMixin`, the derived equivalence
    and `symbolic` (4,549 tests) all pass with Python's method replaced by
    the Rust rule (through a pytest plugin kept outside the repo).

  Python binder terms (`RegisteredFunction`, `Param`, `BinderMixin`
  users) reach this case only when one side's nested binders reuse a
  name, and no Python test does. S4.3 inherits the Rust rule when the
  binding backs Python's `AlphaRenaming`; the maintainer may instead want
  the Python rule fixed first, which is a two-line change.
- **Duplicate bound identifiers.** A frame is a map, as Python's `dict` is,
  so it cannot bind one identifier twice. Python builds it with
  `dict(zip(self_bound, other_bound))`, which keeps the last pairing of a
  repeated self-side name; the Rust docs of `enter_binder` tell a caller
  pairing parameter lists to refuse a repeated name instead. This is left
  to the caller on both sides.
- **Hashing.** Python's `AlphaRenaming` is hashable; the Rust one is not,
  since its maps are `HashMap`s. No Rust caller needs it. See the proposals
  below.
- **Tests.** `tests/it/expression/alpha_stories.rs` (43 cases, counting
  `rstest` cases) and `alpha_properties.rs` (3 properties: symmetry under
  the inverse renaming, agreement with a de Bruijn model of the frames,
  and `leave_binder` undoing `enter_binder`). A Rust expression has no
  binder, so where a Python test compares two binder terms, the Rust test
  compares their bodies under the frames the binders would enter, through
  a test-local `Binders` term of nested one-parameter binders. The
  property file's regression seeds are the two shrunk shadowing cases.

Traceability of the Python tests (`test_alpha_equivalence.py` is `A`,
`test_binder.py` is `B`, `test_derived_equivalence.py` is `D`):

| Python test | Rust test | Note |
|---|---|---|
| A `test_alpha_renaming_empty_returns_renaming_instance` | none | a Rust type needs no instance check |
| A `test_alpha_renaming_empty_resolves_to_identity` | `alpha_renaming_default_resolves_every_identifier_to_itself` | |
| A `test_alpha_renaming_with_free_renaming_stores_mapping` | `alpha_renaming_try_new_resolves_a_mapped_identifier_to_its_image` | |
| A `test_alpha_renaming_with_free_renaming_empty_matches_empty` | `alpha_renaming_try_new_of_an_empty_map_is_the_default` | |
| A `test_alpha_renaming_with_free_renaming_rejects_non_injective_mapping` | `alpha_renaming_try_new_names_the_free_renaming_in_its_refusal`; existing `alpha_renaming_try_new_refuses_a_non_injective_map` | |
| A `test_alpha_renaming_extend_returns_new_instance`, `..._does_not_mutate_receiver` | `alpha_renaming_leave_binder_restores_the_renaming_before_the_frame`, `alpha_renaming_enter_binder_refuses_a_non_injective_frame` (unchanged on refusal) | adapted: Rust pushes in place, so the laws are "leaving restores" and "a refusal changes nothing"; also `alpha_renaming_leave_binder_undoes_enter_binder` (property) |
| A `test_alpha_renaming_extend_empty_frame_resolves_to_identity` | `alpha_renaming_enter_binder_of_an_empty_frame_keeps_resolution_at_identity` | |
| A `test_alpha_renaming_extend_resolves_bound_identifier` | `alpha_renaming_enter_binder_resolves_a_bound_identifier_to_its_image` | |
| A `test_alpha_renaming_extend_inner_frame_shadows_outer_frame` | `alpha_renaming_inner_frame_shadows_an_outer_frame` | |
| A `test_alpha_renaming_extend_permits_other_side_value_reuse_across_frames` | `alpha_renaming_frames_may_share_an_image` | |
| A `test_alpha_renaming_extend_rejects_non_injective_frame` | `alpha_renaming_enter_binder_refuses_a_non_injective_frame` | also pins the message and the part |
| A `test_alpha_renaming_extend_preserves_free_renaming_for_unbound_keys` | `alpha_renaming_enter_binder_keeps_the_free_renaming_of_unbound_identifiers` | |
| A `test_alpha_renaming_resolve_falls_back_to_free_renaming_for_unframed_key` | `alpha_renaming_resolve_falls_back_to_the_free_renaming_outside_every_frame` | |
| A `test_alpha_renaming_resolve_falls_back_to_identity_for_unmapped_key` | `alpha_renaming_resolve_falls_back_to_identity_for_an_unmapped_identifier` | |
| A `test_alpha_renaming_frame_overrides_free_renaming_for_same_key` | `alpha_renaming_frame_takes_precedence_over_the_free_renaming` | |
| A `test_are_identifiers_alpha_equivalent_returns_true_for_resolved_match`, `..._false_for_resolution_mismatch`, `..._true_for_identity_match`, `..._false_for_distinct_unmapped` | `alpha_renaming_is_corresponding_follows_one_frame` (4 cases) | |
| A `test_are_identifiers_alpha_equivalent_agrees_with_resolve` | `alpha_renaming_is_corresponding_agrees_with_resolve` | |
| A `test_alpha_renaming_equal_by_structure` | `alpha_renaming_equal_frames_make_equal_renamings` | |
| A `test_alpha_renaming_hashable` | none | no `Hash` in Rust; see the proposals |
| A `test_alpha_renaming_distinguishes_frame_order` | `alpha_renaming_frame_order_distinguishes_renamings` | |
| A `test_alpha_renaming_distinguishes_free_renaming` | `alpha_renaming_free_renaming_distinguishes_renamings` | |
| A `test_alpha_renaming_empty_equals_empty` | `alpha_renaming_empty_frame_distinguishes_renamings` | also pins that an empty frame counts, as Python's tuple of frames does |
| A `test_mapping_helper_*` (11 tests) | none | `is_identifier_mapping_alpha_equivalent_under` compares Python IR mappings and stays Python; it needs only `resolve` and `is_corresponding`, which are ported |
| A `test_alpha_equivalence_runtime_protocol_*`, `test_alpha_equivalence_mixin_*`, `test_alpha_equivalence_returns_false_for_unrelated_type`, `test_map_bag_*` | none | Python protocols and mixins, and cross-type checks the Rust types rule out |
| A `test_binder_alpha_equivalence_handles_parameter_rename` | `binders_renaming_their_parameter_are_alpha_equivalent` | |
| A `test_binder_alpha_equivalence_distinguishes_diverging_bodies` | `binders_with_a_free_body_on_one_side_are_not_alpha_equivalent` | |
| A `test_binder_alpha_equivalence_handles_nested_shadowing_match` | `binders_nested_over_one_name_match_the_inner_binder` | |
| A `test_binder_alpha_equivalence_handles_nested_shadowing_mismatch` | `binders_nested_over_one_name_do_not_match_the_outer_binder` | |
| A `test_binder_alpha_equivalence_rejects_capture` | `binders_refuse_to_capture_a_free_identifier`, `alpha_renaming_is_corresponding_refuses_an_identifier_bound_only_on_the_other_side` | |
| A `test_binder_alpha_equivalence_swapped_arguments_under_swapped_binders` | `binders_swapped_with_their_operands_are_alpha_equivalent` | |
| A `test_binder_alpha_equivalence_handles_free_renaming_in_body` | `binders_compare_free_identifiers_of_their_body_under_the_free_renaming` | |
| A `test_binder_alpha_equivalence_self_bound_to_self_bound` | `binders_over_the_same_parameter_are_alpha_equivalent` | |
| A `test_alpha_equivalence_is_reflexive`, `..._is_symmetric`, `..._is_transitive` | `binders_alpha_equivalence_is_reflexive`, `..._is_symmetric`, `..._is_transitive` | the example terms add three terms whose nested binders reuse a name, where Python's rule breaks symmetry |
| B `test_identity_lambdas_are_alpha_equivalent` | `binders_renaming_their_parameter_are_alpha_equivalent` | |
| B `test_lambdas_with_distinct_free_bodies_are_not_alpha_equivalent`, D `test_binder_distinguishes_free_identifiers_in_the_body` | `binders_over_distinct_free_bodies_are_not_alpha_equivalent`, `binders_sharing_a_free_body_are_alpha_equivalent` | |
| B `test_lambdas_sharing_a_free_identifier_are_alpha_equivalent` | `binders_sharing_a_free_body_are_alpha_equivalent` | |
| B `test_lambdas_with_different_arity_are_not_alpha_equivalent`, D `test_binder_distinguishes_bound_arity` | `binders_of_different_depths_are_not_alpha_equivalent` | arity is the binder's check, before any frame; the test-local term checks it |
| B `test_non_injective_binding_is_not_alpha_equivalent`, D `test_binder_with_non_injective_binding_returns_false` | `binder_frame_pairing_two_parameters_with_one_is_refused` | |
| D `test_reference_field_consults_the_renaming_in_alpha_mode` | `alpha_renaming_is_corresponding_follows_one_frame`, `expression_is_not_alpha_equivalent_to_itself_under_a_frame_renaming_its_identifier` | the second also pins that a shared handle is not skipped under a frame |
| D `test_reference_field_requires_identifier_equality_without_renaming` | existing `expression_is_alpha_equivalent_under_empty_renaming_is_structural_equality` | |
| D `test_nested_binders_shadow_outer_bindings` | `binders_nested_over_one_name_match_the_inner_binder`, `binders_nested_over_one_name_do_not_match_the_outer_binder` | |
| D `test_binder_renames_bound_identifiers_in_the_scoped_body`, `test_binder_is_alpha_equivalent_to_itself` | `binders_renaming_their_parameter_are_alpha_equivalent`, `binders_alpha_equivalence_is_reflexive` | |
| B substitution, free-identifier and `_Block` tests; D field-schema derivation tests | none | `BinderMixin` and the derived plan stay Python; they call the renaming only through `extend`, `resolve` and `are_identifiers_alpha_equivalent` |

Proposals for S4.3 (core additions the binding may need; none is
implemented):

1. **Nothing is needed to build calls by name.** `Callee`'s `FromStr`
   gives the built-in for a reserved name and `Named` otherwise, and
   `Expression::call(callee, arguments)` builds the node, so
   `CallExpression.function_name` round-trips through `Callee::name()`.
2. **A cheap clone of the frame stack**, if the binding's `extend`
   (clone, then `enter_binder`) shows up in the derived-equivalence
   benchmarks: frames as `Arc`s, or a persistent list, would make the clone
   O(1) instead of copying every frame's maps. Python copies only the tuple
   of frame references.
3. **`Hash` for `AlphaRenaming`**, order-independent, if the binding wants
   Python's `hash(renaming)` without walking the maps itself; otherwise the
   binding hashes sorted `(id, id)` pairs.
4. **A cached structural hash.** On the Rust semantics `hash(expression)`
   walks the whole tree (B3 §7 keeps caching out of the core). The
   baseline's dict lookup by a deep tree takes 46 ns today; the binding can
   cache the hash in each pyclass, which is immutable, without a core
   change.
5. **`Decimal` from parts.** `LiteralExpression(decimal.Decimal(...))`
   reaches the core only through positional text today (`format(d, "f")`
   parsed by `Decimal::from_str`, not `LiteralValue::parse_text`, which
   reads `"100"` as an integer). A `Decimal` constructor from a coefficient
   and an exponent would avoid the text, and the binding must decide what a
   negative `decimal.Decimal` becomes, since the core's `Decimal` is
   non-negative.

### S4.3a status

S4.3 lands in two steps. S4.3a switches the expression core onto the Rust
implementation with the Rust semantics of D-S4-1 to D-S4-5 and migrates
the expression package's own core tests; S4.3b migrates the consumers
(the solver, types, constraints, params, the expression passes and
patterns, the symbol table) and their tests, using the input list below.

S4.3a was implemented on 2026-09-25 in eight commits: the binding
(6c918b4), the Python switch with the import-time shims below (f755e0d),
narrower stub types (a3fa4d7), the migrated strategies (340e468) and core
tests (a6026e1), a one-pass payload decoder the first benchmark run
called for (fc3c391), a fix of a registry-clearing test (5fa1bfd), and
the benchmark helpers (bae95d0). As planned, the consumers' tests fail
on the Rust backend where they rely on the old semantics, and the
pure-Python backend fails the migrated tests; both are fixed in S4.3b and
S4.4.

State at bae95d0 on the Rust backend (`-m 'not slow'`): 5585 passed, 290
failed, 7 skipped, 1 xfailed, and 192 errors, which are eight consumer
test modules failing collection once per xdist worker. Of the migrated
expression-package tests, 1,098 pass and 10 fail, each in a consumer
bridge (listed below). On the pure-Python backend: 6332 passed, 242
failed and 25 errors, all in the migrated tests and in property tests
drawing from the migrated strategies. The Rust gate is green (fmt,
clippy `-D warnings`, 2,630 tests, doc `-D warnings`, deny, `cargo +1.85
check`); ruff check and format pass; mypy reports 53 errors, all in
consumer modules and their tests.

### S4.3a benchmarks (before and after)

Median time per call, from `uv run --python 3.11 nox -s
"benchmark-3.11(backend='rust')" -- -k test_expression`, on the S0
machine with Python 3.11.13, at bae95d0. The load average was 3 to 5;
the benchmarks ran three times and the table lists the best of the three
medians. "Before" is the S4.1 baseline's Rust column, which measured the
pure-Python classes. Every row's medians agreed within 30% except six
sub-microsecond construction and operator rows (up to 1.6 times) and the
small-tree hash (2.2 times).

| Benchmark | before (S4.1) | after | after / before |
|---|--:|--:|--:|
| `test_identifier_expression_construction` | 861 ns | 449 ns | 0.52 |
| `test_literal_expression_construction[int]` | 976 ns | 210 ns | 0.21 |
| `test_literal_expression_construction[big_int]` | 980 ns | 705 ns | 0.72 |
| `test_literal_expression_construction[float]` | 979 ns | 201 ns | 0.21 |
| `test_literal_expression_construction[integer_text]` | 1.26 µs | 414 ns | 0.33 |
| `test_literal_expression_construction[decimal_text]` | 1.49 µs | 770 ns | 0.52 |
| `test_literal_expression_construction[bool]` | 965 ns | 202 ns | 0.21 |
| `test_unary_expression_construction` | 1.01 µs | 283 ns | 0.28 |
| `test_binary_expression_construction` | 1.12 µs | 315 ns | 0.28 |
| `test_make_binary_expression` | 3.36 µs | 1.02 µs | 0.30 |
| `test_logical_not_construction` | 1.67 µs | 653 ns | 0.39 |
| `test_conjunction_construction` | 4.03 µs | 1.32 µs | 0.33 |
| `test_piecewise_construction` | 7.47 µs | 2.05 µs | 0.27 |
| `test_call_construction_of_a_builtin` | 4.53 µs | 1.42 µs | 0.31 |
| `test_call_construction_of_a_user_function` | 4.58 µs | 4.06 µs | 0.89 |
| `test_deep_tree_construction` | 309.9 µs | 79.9 µs | 0.26 |
| `test_binary_operator_of_two_expressions[add]` | 2.15 µs | 378 ns | 0.18 |
| `test_binary_operator_of_two_expressions[multiply]` | 2.14 µs | 382 ns | 0.18 |
| `test_binary_operator_of_two_expressions[true_divide]` | 2.13 µs | 381 ns | 0.18 |
| `test_binary_operator_of_two_expressions[floor_divide]` | 2.16 µs | 382 ns | 0.18 |
| `test_binary_operator_of_two_expressions[modulo]` | 2.14 µs | 382 ns | 0.18 |
| `test_binary_operator_of_two_expressions[power]` | 2.15 µs | 388 ns | 0.18 |
| `test_binary_operator_of_two_expressions[less]` | 2.13 µs | 383 ns | 0.18 |
| `test_binary_operator_of_two_expressions[greater_equal]` | 2.14 µs | 389 ns | 0.18 |
| `test_binary_operator_of_two_expressions[equals]` | 2.11 µs | 394 ns | 0.19 |
| `test_binary_operator_of_two_expressions[not_equals]` | 2.10 µs | 387 ns | 0.18 |
| `test_add_operator_with_int` | 3.52 µs | 875 ns | 0.25 |
| `test_reflected_subtract_operator_with_int` | 3.59 µs | 926 ns | 0.26 |
| `test_unary_operator[neg]` | 1.72 µs | 321 ns | 0.19 |
| `test_unary_operator[pos]` | 1.70 µs | 316 ns | 0.19 |
| `test_node_attribute_access[unary]` | 85 ns | 82 ns | 0.97 |
| `test_node_attribute_access[binary]` | 99 ns | 95 ns | 0.96 |
| `test_node_attribute_access[identifier]` | 51 ns | 49 ns | 0.96 |
| `test_node_attribute_access[literal]` | 51 ns | 49 ns | 0.97 |
| `test_node_attribute_access[piecewise]` | 99 ns | 95 ns | 0.96 |
| `test_node_attribute_access[call]` | 85 ns | 82 ns | 0.97 |
| `test_isinstance_of_node_kind` | 42 ns | 41 ns | 0.99 |
| `test_isinstance_dispatch_over_node_kinds` | 2.85 µs | 919 ns | 0.32 |
| `test_eq_of_one_node` | 88 ns | 68 ns | 0.77 |
| `test_eq_of_distinct_equal_trees` | 134 ns | 210 ns | 1.56 |
| `test_eq_of_distinct_equal_deep_trees` | 137 ns | 4.76 µs | 34.74 |
| `test_hash_of_small_tree` | 51 ns | 69 ns | 1.34 |
| `test_hash_of_deep_tree` | 51 ns | 151 ns | 2.96 |
| `test_dict_lookup_by_deep_tree` | 46 ns | 63 ns | 1.38 |
| `test_structural_equivalence_of_deep_trees` | 2.23 ms | 4.81 µs | 0.00 |
| `test_structural_equivalence_of_shared_dags` | 22.0 ms | 648 ns | 0.00 |
| `test_alpha_equivalence_under_free_renaming_of_deep_trees` | 2.46 ms | 8.95 µs | 0.00 |
| `test_substitute_in_deep_tree` | 244.3 µs | 73.4 µs | 0.30 |
| `test_substitute_in_shared_dag` | 2.12 ms | 8.60 µs | 0.00 |
| `test_free_identifiers_of_deep_tree` | 65.4 µs | 10.1 µs | 0.15 |
| `test_pformat_expression_of_deep_tree[symbolic]` | 915.7 µs | 8.72 µs | 0.01 |
| `test_pformat_expression_of_deep_tree[functional]` | 924.5 µs | 7.98 µs | 0.01 |
| `test_pformat_expression_of_deep_tree[show_id]` | 915.9 µs | 10.5 µs | 0.01 |
| `test_visitable_pass_walk_of_deep_tree` | 923.4 µs | 291.0 µs | 0.32 |
| `test_validate_logical_operands_of_deep_conjunction` | 1.33 ms | 77.2 µs | 0.06 |
| `test_validate_predicate_of_nested_piecewise` | 494.6 µs | 17.3 µs | 0.04 |
| `test_validate_predicate_of_comparison` | 6.68 µs | 1.52 µs | 0.23 |
| `test_serialize_to_dict_of_deep_tree` | 245.2 µs | 113.8 µs | 0.46 |
| `test_deserialize_from_dict_of_deep_tree` | 45.8 ms | 262.2 µs | 0.01 |
| `test_json_round_trip_of_deep_tree` | 46.8 ms | 1.21 ms | 0.03 |
| `test_pickle_round_trip_of_deep_tree` | 524.3 µs | 189.5 µs | 0.36 |

What the numbers show:

- **Construction** is 2 to 5 times faster: a node is a type check per
  field, a Rust node and a children tuple. The operators, which coerce
  through the registered public classes, are 4 to 5.5 times faster.
  Constructing a call of a user function gains little (4.1 µs against
  4.6 µs): the core parses a name that is not a built-in's through
  `BuiltinFunction::from_str` twice (`Callee::from_str`, then
  `FunctionName::try_new`), and each miss formats serde's "unknown
  variant" message listing all 35 built-ins; see the proposals below.
- **Field reads and a hit of `isinstance`** stay at the floor, since the
  nodes keep their field objects as struct members. The dispatch cascade
  over the node kinds is 3 times faster, because the classes' metaclass
  is `ABCMeta` instead of `typing`'s protocol metaclass (see the notes).
- **The whole-tree operations** are where Rust wins: structural and alpha
  equivalence of two deep trees take 5 to 9 µs instead of 2.2 to 2.5 ms,
  of two shared DAGs 0.65 µs instead of 22 ms; substitution is 3.3 times
  faster on the deep tree and 250 times on the DAG; free identifiers 6.5
  times; `pformat_expression` 90 to 115 times; the screen 4 to 29 times;
  decoding a deep tree from a dict 175 times (after fc3c391; the first run
  measured it unchanged at 44 ms, see the notes), from JSON 39 times; a
  pickle round trip 2.8 times.
- **The visitor walk** is 3.2 times faster, although it is Python code:
  the children tuple is prebuilt and the dispatch suffix is cached.

**Accepted costs of D-S4-1's structural `==` and `hash`**, for the
maintainer to confirm (CONTRIBUTING "Replacing a Python class"):

- `==` of two separately built equal trees compares them: 210 ns for a
  small tree and 4.8 µs for the 100-level one, against 134 to 137 ns for
  the identity comparison it replaces. The binding answers at once for
  one shared node, and for two trees whose cached hashes differ.
- `hash` is cached per node, but computed on the first call (the cost of
  the structural digest, linear in the distinct nodes) and read through
  the extension after that: 69 ns for a small tree, 151 ns for the deep
  one, 63 ns for a dict lookup, against 46 to 51 ns for `object.__hash__`.

### S4.3a implementation notes

Choices the decisions left open, made while implementing S4.3a:

- **Core additions: none.** The binding uses the public API of
  `fhy_core::expression` as S4.2 left it, including the frame-based
  `AlphaRenaming`. S4.2's proposals 1 (building calls by name) and 5
  (decimals from text) hold as written, and proposals 2 to 4 are not
  needed (the per-node hash cache lives in the binding). No Rust test was
  added.
- **The hierarchy.** `_rs.Expression` is a `#[pyclass(subclass, frozen)]`
  base holding the Rust `Expression`, the tuple of its children's Python
  objects in visiting order, and a lazily computed structural hash
  (`OnceLock<u64>`). It implements equality, hashing, the operators,
  `str`, `repr`, `accept`, `get_visit_method_suffix` (cached per class),
  `get_visit_children`, free identifiers, substitution, structural and
  alpha equivalence, whole-payload decoding and the frozen members. Each
  of the seven node classes, `LogicalExpression` included, extends it
  with its field objects as `#[pyo3(get)]` members and implements
  `get_operands`, `rebuild_with_visit_children`, `__reduce__` and its data
  payload. The public classes are `Expression(_rs.Expression,
  WrappedFamilySerializable, AlphaEquivalenceMixin,
  RewritableMixin["Expression"])` and `Node(_rs.Node, Expression)`; each
  registers itself at import (S3's mechanism), and the binding builds
  every node it creates through the registered class.
- **No protocol metaclass.** `VisitableMixin` derives from the
  `Visitable` protocol, so inheriting it made `typing._ProtocolMeta` the
  classes' metaclass, whose `isinstance` misses run Python code (575 ns
  each, measured). The classes register as virtual subclasses of
  `VisitableMixin` and `FrozenMixin` instead, as S2/S3's did for
  `FrozenMixin`, and no longer inherit `HasOperands`; the runtime
  protocols (`HasOperands`, `Visitable`, `StructuralEquivalence`, `Term`)
  still hold structurally. The metaclass is `ABCMeta`, from
  `Serializable`.
- **Child objects.** A node built from Python keeps the objects it was
  given, so `node.left is left` holds and a field read is a struct-member
  read. A tree the core builds (a substitution's result) is materialized
  at once, top-down beside the input's objects: a result node that is the
  handle of the input object at the same place, or of a replacement, is
  that object, and any other node is built from its children's objects
  through its public class; a node the core shares is built once. So
  every Python node's children are always Python node objects whose Rust
  handles are the Rust node's children, and a field read never builds an
  object. The walks keep their pending nodes on the heap: a 20,000-level
  tree compares, hashes, prints, substitutes and screens without
  recursion, and a 200,000-level one deallocates (the Python subclasses'
  trashcan bounds the recursion).
- **Equality and hashing.** `==` and `!=` are structural against any
  expression and `NotImplemented` against anything else, so an
  expression never equals a Python value; a user subclass of a node class
  compares by kind, as the core does. A node's hash is the core's
  structural digest, computed on the first `hash` and cached.
  `is_structurally_equivalent` is `==`, `is_alpha_equivalent` is `==`
  (expressions bind nothing), and `is_alpha_equivalent_under` converts
  the renaming (below). `bool(expression)` still raises `TypeError`, for
  the chained-comparison trap.
- **Literals.** `LiteralExpression` accepts a `bool`, an `int` (or a
  subclass other than `bool`), a `float` (or a subclass), a finite
  non-negative `decimal.Decimal`, or a `str` of the core's grammar (ASCII
  digits with at most one decimal point, `LiteralValue::parse_text`).
  `value` is the normalized `bool`, `int`, `float` or `decimal.Decimal`:
  an exact `bool`, `int` or `float` is kept as given, anything else is
  the conversion of the Rust value (`"05"` gives `5`, `"1.50"`
  `Decimal("1.5")`, `"100.0"` `Decimal("1E+2")`). Unicode digits, which
  Python's `\d` accepted, are refused, as the core refuses them.
- **Negative decimals are refused.** The core's `Decimal` is
  non-negative, because its literal grammar has no sign and a negative
  number is the negation of a literal. `LiteralExpression` cannot return
  a negation, so it raises `ValueError` for a negative `Decimal` (and for
  a NaN or infinite one), and the message says to write the negation of
  the literal of its magnitude; the coercing builders go through the
  constructor and refuse it too. A negative zero is the decimal zero. A
  `Decimal` reaches the core as its positional text (`format(d, "f")`)
  through `Decimal::from_str`, and comes back as
  `Decimal(f"{coefficient}E{exponent}")`.
- **Operations.** `UnaryOperation`, `BinaryOperation` and the new
  `LogicalOperation` stay Python `StrEnum`s (P1) whose values are the
  Rust operations' names, so a member converts by value, a payload holds
  the core's name, and a functional `pformat` prints `operation.value`.
  `BinaryOperation.MODULO` keeps its name (D-S4-2) and takes the value
  `"floor_mod"`; `LOGICAL_AND`/`LOGICAL_OR` are gone; `LogicalOperation`
  is `AND = "and"`, `OR = "or"`. The binding keeps each enum's members in
  a table, so converting is an index one way and an identity scan the
  other; a constructor also accepts a value the enum maps to a member.
- **Connectives.** `logical_and(*expressions)` and `logical_or(...)`
  build one `LogicalExpression` over all their operands, coerced, never
  flattening a nested one (the core's D-5), and keep refusing fewer than
  two operands with `ValueError`, rather than taking the core's
  `all`/`any` folding to a literal or to the lone operand, so they always
  return a `LogicalExpression`; the instance methods do the same. No
  `&`/`|` operator was added: Python never had one, and the core rejects
  `BitAnd`/`BitOr` (B3 §7). `LogicalExpression(operation, operands)`
  takes any iterable of at least two expressions.
- **Calls.** `CallExpression(function_name, arguments)` parses the name
  with `Callee::from_str`, so a built-in's name is the built-in (D-9) and
  any other non-empty name a user function; `function_name` is the given
  `str`. A new `is_builtin` property says which. Rebuilding a call takes
  exactly its own argument count, as the core does, where Python took
  any count.
- **Text.** `str` is the core's `Display`; `repr` is the node's class
  name around the core's bounded `Debug` body (the functional notation
  with ids, eliding after 1,000 nodes), for example
  `BinaryExpression((add x::7 1))`; `pformat_expression` calls the core's
  `display` with the matching `FormatOptions`. `ExpressionPrettyFormatter`
  stays a Python `VisitablePass` for subclasses and renders the same
  text: it prints a literal as `str(literal)` and gained
  `visit_logical_expression`; a property test checks it against the core
  under every option.
- **Errors**, through `IntoPyErr` with the core's messages:
  `PiecewiseError`, `RebuildError` (a rebuild with the wrong child count,
  which is also what a leaf given children raises now, instead of
  `NotImplementedError`), `FunctionNameError`, `LiteralTextError` and
  `NonInjectiveRenamingError` raise `ValueError`;
  `NonBooleanLogicalOperandError` raises the Python class of that name
  (`errors.py`, a `TypeError`), with the core's text followed by the
  reprs of the operand and of the node taking it: `operand 0 of a logical
  or provably denotes a number but sits in a boolean position:
  LiteralExpression(2) in LogicalExpression((or 2 4))`, and `the
  predicate provably denotes a number: ...` for a root. The binding's
  own checks raise `TypeError` in S2's style for a field of the wrong
  type (`UnaryExpression operand must be an Expression, got int.`), and
  `ValueError` for unequal piecewise lengths and fewer than two logical
  operands. The builders keep Python's texts for a bare `bool` (reworded,
  since `expr == k` is now a structural comparison) and for an operand
  that cannot be cast.
- **Substitution** is the core's `substitute`, then the materializer. A
  mapping that replaces no reference returns the expression itself;
  every replaced reference is the replacement object itself. Keys that
  are not `Identifier`s are ignored, and a value that is not an
  expression raises `TypeError` only when its key occurs in the
  expression, as before.
- **Alpha renaming (D-S4-3) is converted at the boundary.** The Python
  `fhy_core.term.AlphaRenaming` stays a Python value class: the term
  package's binder machinery (`BinderMixin`, the derived equivalence
  plan, `RegisteredFunction`, `Param`) builds, extends and consults it
  per identifier, and is not ported. `is_alpha_equivalent_under` converts
  it once per comparison, the free renaming through `try_new` and each
  frame, outermost first, through `enter_binder`, and compares the two
  trees in Rust; an empty renaming is plain equality. Python's
  `are_identifiers_alpha_equivalent` follows the Rust rule since 5a7802c,
  so a comparison answers alike either way. The cost is linear in the
  renaming's size per comparison; backing the Python class by the Rust
  one would instead cross the boundary on every `resolve` the binder
  machinery makes.
- **The screen (D-S4-4).** `validate_logical_operands` and
  `validate_predicate` run the core's `BooleanScreen` with a
  `RegistrySorts` adapter implementing `SortLookup`: a native constant's
  sort comes from `try_get_native_constant_for_identifier`, called with
  the Python identifier the trees hold (collected once from the
  expression and the environment), and a named call's from
  `try_get_registered_result_sort`, each cached per identifier or name
  for the call; an error a lookup raises is raised after the screen. A
  built-in call is judged by the core's catalogue (names are reserved);
  a test checks that the catalogue agrees with the registry on every
  built-in's result sort. An environment value must be an expression
  (`TypeError` otherwise), and a declared sort a `SymbolType`.
- **Payloads (D-S4-5).** The public classes inherit the envelope from
  `WrappedFamilySerializable`, and each node class writes and reads its
  data: `{"operation": "negate", "operand": ..}`, `{"operation":
  "floor_mod", "left": .., "right": ..}`, `{"operation": "and",
  "operands": [..]}` under the new `logical_expression` type id,
  `{"identifier": {"id": .., "name_hint": ..}}`, `{"value": ..}`,
  `{"conditions": [..], "values": [..], "otherwise": ..}` and
  `{"function_name": .., "arguments": [..]}`. A literal's value is its
  normalized `bool`, `int` or `float`, and a decimal is its positional
  text with a decimal point (`"1.5"`, `"100.0"`), which the grammar reads
  back as the same decimal rather than as an integer.
  `_rs.Expression.deserialize_from_dict` decodes a payload of exactly
  these shapes in one pass (fc3c391): the framework's per-node path
  checks every node's whole nested payload, which is quadratic in the
  depth, and measured 44 ms for 100 levels. Any other payload, and any
  one a constructor refuses, goes through the framework's path, so which
  payloads decode and how a malformed one fails are unchanged.
- **Pickles** are a call of the node's class with its fields,
  `(type(self), fields)`, as S3's are; a pickle written by the
  pure-Python backend does not load on the Rust backend, as D-S4-5
  allows.
- **Python multiplexing.** The pure-Python implementation moved
  unchanged to `symbolic/expression/_python_core.py`, which `core.py`
  re-exports on the pure-Python backend and which S4.4 deletes; it gained
  `LogicalOperation` and a `LogicalExpression` that refuses construction,
  so the package imports on both backends. The Rust branch of `core.py`
  defines the enums, the public classes, the builders, the screen
  wrappers, and `build_literal_equivalence_key`/`is_integer_valued_literal`
  over the normalized values (a `Decimal` is in the decimal bucket, so
  the key still agrees with `==`). `LiteralType` gains `decimal.Decimal`.
  `registry/` and `builtins.py` are unchanged: they build their bodies
  with the Rust-backed nodes, so `xor`'s body is now
  `((a || b) && (!(a && b)))` of `LogicalExpression`s.
- **Type checkers see the Rust-backed API.** `core.py` branches on
  `TYPE_CHECKING or IS_RUST_BACKEND_SELECTED`, the reverse of S2/S3's
  convention: for expressions the Rust branch now describes the public
  API, so mypy checks the consumers against it (53 errors, all in
  consumers; S4.3b input). `_rs.pyi` types fields and results with the
  public classes, imported from `core` (a stub may import cyclically),
  declares `__setattr__`/`__delattr__` as `FrozenMixin` did, and narrows
  each node's `rebuild_with_visit_children`.
- **Import-time shims in consumers.** Five consumer modules built tables
  of `BinaryOperation.LOGICAL_AND`/`LOGICAL_OR` at import time (the NumPy,
  SymPy and Z3 bridges' operator tables, the solver's and the type
  checker's connective sets), which made the whole package fail to import
  on the Rust backend. They now build those entries only where the
  members exist, with a comment; nothing else in the consumers changed.
  S4.3b removes the shims when it gives the consumers `LogicalExpression`.
- **Test identifiers.** The migrated tests use real `Identifier`s. The
  shared pools of `tests/strategies/identifiers.py` hold real identifiers
  with the same fixed ids, restored through
  `Identifier.deserialize_from_dict`, so they stay deterministic; the
  serialization pins' shared variable uses the fixed id 60000, in the
  reserved range but clear of every shipped tag (its old id 0 is the
  `rationale` note kind's). `mock_identifier` stays in `tests/conftest.py`
  for the consumer tests.

Tests migrated in S4.3a, all to the new semantics, none skipped:
`tests/symbolic/expression/`: `test_core.py`, `test_piecewise_and_call.py`,
`test_builtins.py`, `test_registry.py`, `test_pprint.py`,
`test_pprint_properties.py`, `test_sort.py` (unchanged, passes),
`test_term.py`, `test_cross_cutting.py`, `test_functions_stories.py`,
`test_native_stories.py`, `test_core_properties.py` and
`test_piecewise_properties.py`; `tests/symbolic/test_serialization_pins.py`
(19 pinned type ids with `logical_expression`; the goldens were
regenerated from the code and reviewed: the only changes are the shared
variable's id and the new blob, and the `Param` goldens keep id 1, since
they are compared by alpha equivalence);
`tests/symbolic/test_pickle_round_trips_properties.py` (plain pickles
now that no identifier is a mock, plus an expression round trip); the
strategies `tests/strategies/expressions.py`, `structural_expressions.py`,
`literals.py` and `identifiers.py`, and the expression parts of
`tests/test_strategies_properties.py`. A new Rust-backend-only suite,
`test_expression_rust_binding.py` (32 tests), covers the class structure
and registration, argument checks, child objects, deep trees, payload
decoding, and pickles.

No test was deleted without a rewrite. These were renamed because they
now pin the opposite behavior: `test_literal_expression_keeps_integer_shaped_string_as_str`
and `..._keeps_float_shaped_string_as_str` (now `..._normalizes_...`),
`test_binary_dunder_lifts_str_operand_on_right` (now also a `Decimal`),
`test_distinct_expression_instances_are_unequal_under_eq` and
`test_set_of_distinct_field_equal_expressions_keeps_both_members` (now
equal, and one member), `test_piecewise_expression_hash_is_defined_and_follows_identity`
(now structural), `test_module_level_logical_builder_folds_three_args_right_associatively`
and `test_logical_and_right_folds_three_operands`/`_four_operands` and
`test_logical_or_right_folds_three_operands` (now one n-ary node),
`test_new_expression_subclass_derives_equivalence_without_registration`
(now: the node kinds are closed, a Python subclass of `Expression` that
is no node class cannot be built, and a subclass of a node class is that
kind), and `test_exactly_eighteen_pinned_type_ids_are_covered` (now
nineteen).

Proposals (none is implemented):

1. **Parse a user function's name without serde's error.**
   `BuiltinFunction::from_str` goes through serde's `StrDeserializer`, so
   every name that is not a built-in's formats "unknown variant, expected
   one of" with all 35 names, twice per `Callee::from_str`. A `match` over
   the names in `impl_name_text` would make a user call as cheap to build
   as a built-in's (about 1.4 µs instead of 4.1 µs).
2. **Structural `==` of distinct deep trees** could compare the cached
   hashes first when only one is known, computing the other; it cannot
   avoid the walk for equal trees.

### S4.3b input: the consumers failing on the Rust backend

From a Rust-backend run at bae95d0 (`-m 'not slow'`), grouped by the
consumer whose code causes the failure; the counts are failing tests,
and a module failing collection counts once. The pass-infrastructure
`ERROR` log lines in the output are these failures' passes logging.

| Cause (consumer module) | Tests failing | What fails |
|---|--:|---|
| `passes/z3.py`: no `visit_logical_expression` | about 190: `test_solver.py` 62, `param/test_param_intersection.py` 36, `constraint/test_constraint_system.py` 35, `param/test_sound_feasibility.py` 24, `param/test_tri_state_feasibility.py` 24, `param/test_subset_relations.py` 7, `test_strategies_properties.py` 6, `param/test_real_param.py` 4, `param/test_param_intersection_properties.py` 4, `param/test_number_subclass_values.py` 3, `test_solver_properties.py` 3, and 1 or 2 each in `constraint/test_bindings_evaluation.py`, `test_user_stories.py`, `test_constraint_system_properties.py`, `param/test_bound_int_param.py`, `test_dependent_param_story.py`, `test_feasibility.py`, `test_feasibility_properties.py`, `test_int_param.py`, `test_nat_param.py`, `test_param_multiplication.py` and `test_subset_relations_properties.py` | every conjunction the constraint systems and params build is a `LogicalExpression`, which the Z3 bridge's `VisitablePass` does not dispatch |
| `passes/z3.py`, `passes/sympy.py`: literal lowering | `expression/test_cross_cutting.py` 9, `test_solver.py` 1, `test_solver_properties.py` 1, `constraint/test_bindings_evaluation.py` 1 | "Unsupported literal type: Decimal": a decimal literal's value is a `decimal.Decimal`, where it was a `str` |
| `passes/numpy.py`: `LOGICAL_AND` read at call time | `test_solver_properties.py` 3, `test_strategies_properties.py` 2, and 1 each in `expression/test_piecewise_properties.py`, `passes/test_evaluator_properties.py`, `passes/test_inline_pass_properties.py`, `pattern/test_rewrite_properties.py` | `_evaluate_binary` compares with the removed members, and there is no `visit_logical_expression` |
| `passes/sympy.py`: no logical node | `test_solver_properties.py` 1 | the SymPy bridge has no `visit_logical_expression`, and its lift builds binary connectives |
| `types/checking/type_checker.py`, `body_type_checker.py` | `test_type_checker.py` 4, `test_type_checker_properties.py` 2, `test_strategies_properties.py` 3, `test_registry_body_sweep.py` 7, `test_builtin_bodies.py` 1 | no `LogicalExpression` case ("Unsupported expression type"), so the `xor`/`nand`/`nor`/`implies` bodies fail the body check and every sweep reports them; a `Decimal` literal is unsupported, and a string literal no longer reaches its refusal (it is a number now); `object.__setattr__` on a node raises `TypeError` ("can't apply this __setattr__"), where it could patch a dataclass |
| `constraint/`: conjunction shape | `constraint/test_constraint_system.py` 11 | `convert_to_expression` of several members is a `LogicalExpression`, and the tests build `BinaryOperation.LOGICAL_AND` |
| `param/`: decimal values | `param/test_real_param.py` 9, `param/test_sound_feasibility.py` 1 | a real param's string bounds are `Decimal` values now, so exact decimals beyond the float range and string-valued members no longer validate, and the old literal-grammar message (`Invalid string-form literal expression`) is now the core's (`invalid literal text ...`) |
| `passes/evaluate.py` | `passes/test_evaluator.py` 1 | a string-form float argument is a `Decimal` literal now, which the evaluator folds instead of refusing |
| Test modules reading `LOGICAL_AND`/`LOGICAL_OR` at import | 8 modules fail collection: `constraint/test_convert_to_expression.py`, `constraint/test_equation_constraint.py`, `passes/test_numpy_evaluator.py`, `passes/test_sympy_pass.py`, `passes/test_sympy_pass_properties.py`, `passes/test_z3_pass.py`, `types/checking/test_type_checker_booleans.py`, and `param/test_param_serialization_properties.py` (its module-level strategies run the Z3 bridge) | the tests themselves build the removed members |

mypy's 53 errors are in `passes/sympy.py` (7), `passes/numpy.py` (2),
`passes/native_lowering.py` (1), `types/checking/type_checker.py` (1),
and eight consumer test modules (42): the removed members, and
`LiteralExpression.value` now including `Decimal`. Not in the table,
because nothing fails yet, but for S4.3b to check:
`constraint/ordering.py` keys members by the class name and
`operation.value` (`MODULO`'s value is now `"floor_mod"`, and a
conjunction is a `LogicalExpression`), which orders `ConstraintSystem`
members; `types/dispatch.py` recurses only into binary, unary and
identifier nodes; `pattern/core.py` matches literals by stored type and
connectives as `BinaryExpressionPattern`s; and consumers that detect a
no-op rewrite by `is` get the same objects back only where the core
kept the subtree.


### S4.3b notes

The agent that did S4.3b stopped when its session ended, after these
commits. The remaining checks (slow tests, the property session, the Rust
gate, benchmarks) were then run and recorded directly.

- `2d7dbdb`: the z3, sympy, numpy, evaluate and inline passes lower
  `LogicalExpression` and `Decimal` literals.
- `c416242`: Python patterns match `LogicalExpression` (a new
  `LogicalExpressionPattern`) and normalized literals.
- `78e2a05`: the solver's hazard screens classify a `LogicalExpression` as
  Boolean.
- `ab2bbc6`: the type checker checks `LogicalExpression` and unifies
  through it.
- `8de9b85`: constraints are keyed and decided over the Rust expression
  semantics.
- `ab3e6d3`: the param tests pin the core's literal-text message for a
  malformed real bound.
- `7d50809`: the README lists `LogicalExpressionPattern`.
- The five import-time shims from S4.3a are gone (`grep` finds none).

Benchmarks (Rust backend, 3.11, after S4.3b):

| Benchmark | S0/S4.1 baseline | After S4.3b |
|---|--:|--:|
| `visitable_pass_walk_of_deep_tree` | 906 us | 297 us |
| `compiler_pass_execute` | 13.6 us | 11.4 us |
| `pass_manager_run_of_5_passes` | 381 us | 355 us |
| `analysis_manager_cache_hit` | 9.2 us | 8.7 us |

The pass-infrastructure rows are within noise of the baseline; they change
in S6.

### S4.4 status

S4.4 retires the pure-Python backend per D-S4-6. It was implemented on
2026-09-25 in six commits: the mandatory extension, with the backend switch
and the tests that only made sense on or against the pure-Python backend
(80618a8); the deletion of the pure-Python tags, diagnostics and
provenance, with mypy over the Rust-backed classes (1aafced); the deletion
of the Python id counter (232a41d) and of the Python expression core
(67178b7); the binding's docs and its append-only message (70a2a88); and
these docs. No test is skipped any more. At the end: `pytest` 6,949 passed,
`-m "not very_slow"` 6,982 passed, the `property` session 280 passed, the
`tests-3.11` session (coverage, slow tests included) 6,702 passed with the
43 property modules skipped because its `test` group has no Hypothesis,
`lint` and `type_check` clean, and the Rust gate green (fmt, clippy
`-D warnings`, 2,630 tests, doc `-D warnings`, deny, `cargo +1.85 check`).

### S4.4 implementation notes

- **A missing or broken extension is an `ImportError`.** The new
  `fhy_core._extension` module imports `fhy_core._rs` and checks it;
  `fhy_core/__init__.py` imports it before any other module, so the check
  runs before any module imports the extension. It raises `ImportError`
  (with `name="fhy_core._rs"` and the original error, if any, as its cause)
  whose message names the requirement, the cause and the fix:
  - not installed: `fhy_core requires its Rust extension fhy_core._rs,
    which is not installed. Install a fhy_core wheel for this platform, or
    build the extension from the source checkout with `uv sync`.`;
  - built only for other interpreters: the builds found and the suffix this
    interpreter loads, then the rebuild advice (`uv sync`, or reinstall the
    wheel);
  - failing to import: `... which is installed but failed to import
    (ImportError: ...)`, then the rebuild advice;
  - stale, that is, a `__version__` that is missing, not PEP 440, or
    another version than the installed package's: both versions, then the
    rebuild advice.

  The old module's checks and its PEP 440 normalization are kept; only the
  warnings and the fallback are gone. A stale extension is an error rather
  than a warning because the Python sources now depend on the extension's
  classes, so a mismatched pair cannot be trusted to work.
- **`fhy_core.RUST_BACKEND_SELECTED` is removed.** It was documented and in
  `fhy_core.__all__`, but it was never released: it exists only on the dev
  branches, not on `main` or in any tag. With one backend it would be a
  constant `True`, so no caller could use it for anything; keeping it would
  keep a vestigial API. The commit is marked breaking anyway, since it
  removes a documented name and `FHY_CORE_NO_EXTENSIONS`.
- **`FHY_CORE_NO_EXTENSIONS` is removed** from the code, the test helpers,
  nox (the `backend` parametrization of the `tests`, `property` and
  `benchmark` sessions, and `_select_backend`; `golden_expanded` no longer
  forces the pure-Python backend), CONTRIBUTING and the README. CI already
  called `tests-<python>`, which now runs one session instead of two; only
  its header comment changed. The benchmark session writes
  `.benchmarks/<python>.json` and saves under `.benchmarks/storage/`, and
  the benchmarks no longer record a backend in the machine info.
- **Deleted code.** `symbolic/expression/_python_core.py` (1,650 lines);
  the dataclass branches of `op_attribute.py`, `value_domain.py`,
  `diagnostic.py` and `provenance.py`, with the default-instance tuples
  only their `register_default_instances` read; `identifier.py`'s
  `_PythonIdCounter` and the extension messages it copied; the two backend
  branches of `pprint.py`; `fhy_core._backend`, which kept an always-true
  `IS_RUST_BACKEND_SELECTED` while the branches were removed; the test
  helpers `skip_on_rust_backend`, `build_backend_environment` and
  `NO_EXTENSIONS_VARIABLE`.
- **Kept, although the deleted classes used it.** `InternedMixin` is public
  in `fhy_core.traits`, has its own tests (`tests/test_basic_traits.py`),
  and is the oracle of the interned golden corpus, whose generator interns
  its own dataclass through it; the Rust-backed tags stay virtual
  subclasses of it. `_build_reserved_identifier` and the reserved table
  build the shipped tags' names for `require_interned`.
  `DerivedEquivalenceMixin` is public in `fhy_core.term`. `provenance`
  keeps `_LOGGER`, which the binding's `fuse` logs through. The golden
  corpus stays live, since its oracle still runs.
- **mypy checks the Rust-backed classes** (the leftover "mypy over the Rust
  branches"). With the `TYPE_CHECKING or not IS_RUST_BACKEND_SELECTED`
  branches gone, mypy saw 224 errors, which needed:
  - `_rs.pyi` types fields, arguments and results with the public classes
    (`fhy_core.diagnostic.Note`, `fhy_core.provenance.Provenance`, ...), as
    it already did for expressions, and declares each provenance variant's
    `__str__`, so the variants are not abstract to mypy;
  - the three tags declare their constructor under `TYPE_CHECKING` in the
    class body, since the stub cannot type `_new_canonical` as the public
    subclass's `__new__` (the runtime assignment stays in the `else`), and
    `ValidationReport` declares its generic `__new__` and `records` the
    same way, since `_rs.ValidationReport` cannot be subscripted at runtime
    and so cannot be generic in the stub;
  - `FrozenMixin.register(Provenance)` ignores `type-abstract`: `register`
    takes an abstract class;
  - test ignores that went stale (the stub accepts a level's `str`, any
    iterable, and any object in `is_subdomain_of`), and one that changed
    code.

  `tests/test_rs_stub.py` needed no change: no module-level type variable
  was needed.
- **The binding's text.** The `NotImplementedError` of
  `clear_interned_registry` and `register_default_instances` now reads
  `X.method is not supported: the Rust intern registries are append-only,
  ...`, without "on the Rust backend"; the tests match `append-only` and the
  `X.method is not` prefix. The binding's docs no longer place classes "on
  the Rust backend" or defer to a Python counter.
- **The interface suites keep their names.** `test_*_rust_binding.py` still
  describe what they test, the binding's additions over the core (the
  pyclass structure, the public-class registration, argument checks,
  payload shapes, pickles); the README now says so. Renaming them would
  also break every reference to them above.
- **Release packaging (leftover).** `python-release.yml` runs `uv build`
  on one Linux runner and publishes the result. Before S4.4 a user without
  a matching wheel still got a working pure-Python package; now installing
  from the source distribution needs a Rust toolchain, and the one wheel
  covers only that runner's platform and Python. Publishing needs wheels
  for every supported platform and Python (for example with
  `maturin-action`), which is outside this step; the README says a source
  install needs Rust.
- **Behavioral Python tests are kept.** Decision 4 kept them until the
  pure-Python backend was deleted; they now test the Python API over Rust,
  and many consumers' tests rely on them. Pruning the ones the Rust tests
  already specify is left for a later decision.

Tests deleted or rewritten, each with its reason:

| Test | Change | Reason |
|---|---|---|
| `test_backend.py` | becomes `test_extension.py` | The selection is gone. Deleted: `test_backend_flag_reflects_this_process_environment`, `test_disabling_variable_value_selects_the_python_backend` (6 cases), `test_unset_or_enabling_variable_value_selects_the_rust_backend` (7), `test_disabling_the_extension_skips_the_broken_extension_warning`, `test_disabling_the_extension_skips_the_version_check`. Rewritten to pin the `ImportError`: `test_a_missing_extension_raises_import_error`, `test_an_extension_built_for_other_interpreters_raises_import_error`, `test_a_broken_extension_raises_import_error_naming_the_cause` (2), `test_a_stale_extension_raises_import_error`, `test_a_versionless_extension_raises_import_error`. Kept: the foreign-build finder and the version normalization. New: `test_the_package_imported_its_extension`, `test_the_package_imports_in_a_fresh_interpreter` |
| `test_op_attribute.py::test_op_attribute_first_constructed_with_key_is_canonical`, `test_value_domain.py::test_value_domain_first_constructed_with_key_is_canonical` | deleted | D-S2-4: pinned a fresh object per construction; `test_construction_of_a_registered_key_returns_the_canonical_instance` pins the Rust behavior |
| `test_op_attribute.py::test_register_default_instances_restores_module_level_op_attributes`, `test_value_domain.py::test_clearing_registry_desyncs_module_level_constants_without_default_restore`, `test_value_domain.py::test_register_default_instances_restores_module_level_canonicals` | deleted | D-S2-1: the registries are append-only; `test_registry_reset_raises_not_implemented` pins it |
| `test_value_domain.py::test_value_domain_unequal_when_parents_differ` | deleted | D-S2-3: a name has one parent; `test_value_domain_construction_under_another_parent_raises` and `test_tags_compare_and_hash_by_name` pin it |
| `test_identifier.py::test_python_counter_allocates_only_while_holding_its_lock`, `..._advances_only_while_holding_its_lock`, `test_python_counter_starts_at_the_reserved_block` | deleted | tested `_PythonIdCounter`; `test_fresh_process_issues_ids_upward_from_the_reserved_block` pins the start of the counter that runs |
| `test_identifier.py::test_deserializing_the_largest_payload_id_leaves_construction_working` | its `python` case deleted | one counter |
| `test_identifier.py::test_pickle_written_under_this_backend_loads_under_the_other`, `..._under_the_other_backend_loads_under_this_one` | now `test_pickle_written_in_this_process_loads_in_a_fresh_one`, `test_pickle_written_in_a_fresh_process_loads_in_this_one` | the other process runs the same backend; the tests still pin that unpickling advances that process's counter |
| `test_identifier_rust_binding.py` | the counter tests' `python` cases deleted (8); `test_counters_issue_the_same_relative_ids_for_the_same_operations` now `test_counter_issues_the_expected_relative_ids_for_a_script`; `test_python_counter_issues_the_largest_id_then_fails` and `test_public_identifier_leaves_the_rust_counter_alone_when_unselected` deleted; `..._draws_ids_from_the_rust_counter_when_selected` unconditional, without the suffix; new `test_public_identifier_deserialization_advances_the_rust_counter` | the Python reference counter is gone; the Rust counter's exhaustion is a Rust unit test |
| `test_identifier_rust_binding_properties.py` | the property compares the Rust counter with a model of its contract | the Python reference counter is gone |
| `test_interned_tags_rust_binding.py::test_pickles_of_shipped_tags_are_identical_on_both_backends`, `test_pickle_of_a_new_domain_loads_under_the_python_backend`, `test_pickle_from_the_python_backend_loads_as_the_canonical_tag` | now `..._identical_in_every_process`, `..._loads_in_a_fresh_process`, `test_pickle_from_a_fresh_process_loads_as_the_canonical_tag` | D-S2-2 still holds across processes |
| `test_diagnostic_rust_binding.py::test_pickles_load_across_backends`, `test_provenance_rust_binding.py::test_pickles_load_across_backends` | now `test_pickles_load_across_processes` | as above |
| `test_provenance_rust_binding.py::test_payloads_match_the_python_backend` | now `test_payloads_keep_the_pinned_json_text` | the pure-Python payloads, which the Rust classes matched, are pinned as data |
| the four interface suites | their module-level Rust-backend skip removed | one backend |
| `tests/symbolic/test_namespace.py` | `RUST_BACKEND_SELECTED` no longer in `fhy_core.__all__` | removed |
| `tests/test_golden_corpora.py` | docstring only | the generators run on the one backend |

### S4.4 benchmarks

`uv run --python 3.11 nox -s benchmark-3.11`, now one session with no
backend parameter, ran all 143 benchmarks on the S0 machine (load average
about 1.3) in 2 minutes and wrote `.benchmarks/3.11.json`. S4.4 changes no
hot path: the Rust-backed classes were already the ones that ran, and the
deleted code was never on a Rust-backend path. Spot checks against the
S4.3a and S4.3b tables agree within noise, for example
`test_deep_tree_construction` 75.9 µs (79.9 µs in S4.3a),
`test_eq_of_distinct_equal_deep_trees` 5.1 µs (4.8 µs),
`test_visitable_pass_walk_of_deep_tree` 288 µs (297 µs in S4.3b),
`test_note_construction` 254 ns (300 ns in S3a),
`test_validation_report_construction` 14.6 µs (14.5 µs, the recorded S3a
cost), and `test_identifier_construction` 3.85 µs (4.06 µs in S0). No new
table is needed.

## S5: patterns and rewrite rules

- **Status:** design, 2026-09-25, at fbbb5af. Nothing is implemented.
  D-S5-1 to D-S5-16 apply the policy the user already set. N-S5-1 is not
  covered by it and needs the user before S5.4.
- **Pattern:** P2 for captures, patterns, bindings and rules; P3 for the
  Python callbacks and the `Rule` trait, as the slice table says.

### Survey: the Python API

The package is `src/fhy_core/symbolic/expression/pattern/`: `__init__.py`
(63 lines, re-exports), `core.py` (752) and `rewrite.py` (178). It is pure
Python over the Rust-backed expressions since S4.3b (c416242), and
`symbolic/expression/__init__.py` re-exports its 19 names.

**Patterns (`core.py`).**

- `Pattern(FrozenMixin, ABC)` is open to user subclasses. `match(expression)
  -> MatchBindings | None` calls the abstract `match_under(expression,
  bindings)`, the public seam that threads bindings through compound
  patterns. A structural mismatch returns `None`; only callbacks raise.
- Twelve `@final` frozen dataclasses implement it:

| Class | Constructor | Meaning |
|---|---|---|
| `WildcardPattern` | `()` | any expression, captures nothing |
| `CapturePattern` | `(name, sub_pattern)` | binds `name: str` after `sub_pattern` matched; the same name in two places is one capture, whose expressions must be structurally equal; the first-bound object is kept |
| `LiteralPattern` | `(value=None)` | `None`: any literal; otherwise equality with `LiteralExpression(value)` (normalized: `"05"` matches `5`, NaN matches NaN, `-0.0` matches `0.0`, `1` matches neither `1.0` nor `True`); `.value` keeps the given value |
| `IdentifierPattern` | `(identifier=None)` | `None`: any reference; otherwise the same identifier id |
| `UnaryExpressionPattern` | `(operation, operand)` | `operation=None`: any operation |
| `BinaryExpressionPattern` | `(operation, left, right)` | as above; left, then right |
| `LogicalExpressionPattern` | `(operation, operands)` | `operands=None`: any; otherwise at least two patterns and the exact count |
| `PiecewiseExpressionPattern` | `(cases, otherwise)` | `cases=None`: any; otherwise a non-empty tuple of `(condition, value)` pairs and the exact count, then `otherwise` |
| `CallExpressionPattern` | `(function_name, arguments)` | `function_name: str` compared with the call's `function_name`; `arguments=None`: any; `()`: no arguments |
| `PredicatePattern` | `(predicate)` | matches when `predicate(expression)` is truthy; captures nothing; its exceptions propagate |
| `AlternativesPattern` | `(alternatives)` | non-empty; the first that matches wins, a failed one leaves no captures, and the choice is final |

- **Construction errors** are `ValueError`s: a sub-pattern that is not a
  `Pattern` (`BinaryExpressionPattern left must be a Pattern instance, but
  got value 1 of type int.`), a capture name that is not a `str` or is
  empty (`CapturePattern name must not be empty.`), fewer than two logical
  operand patterns, empty piecewise cases, a case that is not a pair, and
  no alternatives. Sequences are copied into tuples.
- **Dataclass protocols.** `==` and `hash` by fields (callables by
  identity); `repr` such as `BinaryExpressionPattern(operation=None,
  left=CapturePattern(name='a', sub_pattern=WildcardPattern()),
  right=LiteralPattern(value=1))`; default pickling, which fails only for a
  callable that does not pickle.
- **`MatchBindings(bindings: immutabledict[str, Expression] = ...)`**: a
  public constructor with `TypeError` checks on keys and values, `empty()`,
  `is_empty()`, `names() -> frozenset[str]`, `get(name)` (`KeyError` when
  unbound), `has(name)`, and `try_bind(name, expression) -> MatchBindings |
  None`, which returns the receiver itself on a structurally equal rebind.
  `==` compares the key sets and the values structurally; `hash` hashes
  the key set only. It is a `FrozenMixin`.
- `match_pattern(pattern, expression)` and `does_pattern_match(pattern,
  expression)` are the free-function forms of `match`.

**Rewriting (`rewrite.py`).**

- `RewriteRule(pattern, rewrite, guard=None, name=None)` is a frozen
  dataclass, equal and hashable by content.
- `apply_rewrite_rule(rule, expression) -> Expression | None` matches at the
  root, calls `guard(bindings)` (truthiness) and then `rewrite(bindings)`.
  An exception from a predicate, the guard or the rewrite propagates
  unchanged. A result that is not an `Expression`, `None` included, raises
  `TypeError("RewriteRule 'n' rewrite callable must return an Expression;
  got NoneType.")`, with `'<unnamed>'` for an unnamed rule.
- `apply_rewrite_rules(expression, rules) -> Expression` is
  `RewriteRuleApplier(rules)(expression)`, so its errors are the pass
  framework's. A callback's exception is logged as an ERROR diagnostic with
  its traceback, then raised as `PassExecutionError('Pass
  "fhy_core.symbolic.expression.apply_rewrite_rules" failed with
  ValueError: boom')`, with the exception as `__cause__`.
- `RewriteRuleApplier(rules)` is a `RewritablePass[Expression]` registered
  with `@register_pass("fhy_core.symbolic.expression.apply_rewrite_rules",
  "Apply a sequence of rewrite rules bottom-up over an expression tree.")`.
  - `rules` is the tuple it copied.
  - Its `visit_unknown(node)` hook tries the rules in order through
    `apply_rewrite_rule`, and the first result that is not `None` wins. A
    named rule's firing reports the INFO diagnostic `Applied rewrite rule
    'name'.`
  - The walk is `RewritablePass`'s recursive, bottom-up `transform`, and
    `did_change` is `output is not input`.
  - `CompilerPass.create(name)` raises `TypeError`, because `rules` is
    required.

Probed at fbbb5af:

- **A rule that returns the matched node itself fires.** It reports its
  diagnostic, and later rules are not tried. At the root, the output is the
  input. At a child, the parent is rebuilt around the identical child, so
  the output is a new, equal tree and `changed` is `True`.
- **A shared subtree is rewritten once per occurrence.** In `s + s`, the
  rewrite of `s` ran twice.
- **Depth.** A 3,000-level unary chain raises `PassExecutionError` from a
  `RecursionError`. Patterns recurse in Python too.

### Survey: the Rust API

`fhy_core::expression::pattern` (`pattern.rs` 21 lines, `matching.rs` 789,
`rewrite.rs` 613) and `fhy_core::expression::passes` (227). The Rust design
fixed two Python-driven findings here: string-keyed captures (F-019) and a
rewrite returning its own input counting as a change (F-009).

- **`Capture::new(&str)`** is an identity handle. Two captures are equal
  only when one is a clone of the other, and hashing agrees. The name
  serves only `Debug`, `Display` and panic messages, and may be empty.
- **`Pattern`** is closed, an `Arc` of a private `PatternKind`, and its
  constructors never fail:
  - `wildcard`, `nothing`, `capture(&c)` and `captured_as(self, &c)`;
  - `any_literal`/`literal(v)` and `any_identifier`/`identifier(id)`;
  - `unary`/`unary_any_operation` and `binary`/`binary_any_operation`;
  - `logical`, `logical_any_operation`, `logical_any_operands` and
    `any_logical`;
  - `piecewise`/`piecewise_any_cases`;
  - `call`, `call_any_arguments`, `call_any_callee` and `any_call`, over a
    `Callee`;
  - `predicate`/`try_predicate` and `alternatives`.

  Fewer than two logical operand patterns, no piecewise case, or no
  alternative builds a pattern that matches nothing. `matches(&e) ->
  Result<Option<MatchBindings>, CallbackError>` and `is_match` test the
  root only. There is no `PartialEq`, no `Hash`, no field getter and no
  serde; `Debug` is derived. Matching recurses once per pattern level.
- **`MatchBindings`** is a trail in binding order: `new`/`Default` (no
  public way to bind), `is_empty`, `len`, `contains`, `get -> Option`,
  `iter`, and `Index`, which panics with ``capture `x` is not bound``.
  Equality and hashing ignore order and compare the expressions
  structurally.
- **`CallbackError`** is `Box<dyn Error + Send + Sync>`. `matches` and
  `RewriteRule::apply` return it unchanged.
- **`Rule`** is a trait: `apply(&e) -> Result<Option<Expression>,
  CallbackError>`, and `name() -> Option<&str>`, `None` by default. It has
  blanket impls for `&R`, `Box<R>` and `Arc<R>`, so a `Box<dyn Rule>` list
  mixes rule types. Returning the input itself means declining.
- **`RewriteRule`**:
  - `new(pattern, rewrite)` takes a rewrite that always returns an
    expression; `new_partial` takes one that may return `Ok(None)` to
    decline;
  - `with_guard` appends a guard: every guard must return `Ok(true)`, in
    order, and the first that does not stops the rest;
  - `with_name`, `name` and `apply`. `apply` declines a replacement that is
    the input itself.

  Cloning shares the callbacks, which are `Send + Sync`. A rule has no
  equality.
- **`apply_rewrite_rules(&e, &[R]) -> Result<RewriteOutcome,
  RewriteError>`** walks bottom-up once, on its own work stack, so a tree of
  any depth works. It rewrites a node that occurs in several places once
  and reuses the result, and it never revisits a replacement.
  `RewriteOutcome` has `output`, `is_changed` and `fired`: a list of
  `FiredRule { rule_index, name }` in walk order.
- **`RewriteError`** (`#[non_exhaustive]`) has two variants:
  - `Callback { rule_index, rule_name, source }` displays as `rewrite rule
    0 (n) failed`;
  - `Rebuild { rule_index, rule_name, source: RebuildError }` displays as
    `rebuilding a node after rewrite rule 0 (n) failed`, and blames the
    rule that produced the refused child.
- **`passes::RewriteRuleApplier<R = RewriteRule>`**:
  - `new(rules)`, `rules()` and `fired()`, which holds the last run's
    firings, and after a failed run the firings before the failure;
  - `NAME` and `DESCRIPTION` equal Python's;
  - an INFO diagnostic `applied rewrite rule "n"` per named firing;
  - a failure is the `RewriteError` boxed as `PassFailure`, and
    `did_change` is `!ptr_eq`.

  `register_expression_passes(&mut PassRegistry)` registers a factory that
  builds an applier of no rules.

No pattern binding exists yet. The screen's `SortLookup` adapter
(`rust/fhy-core-py/src/expression/screen.rs`) is the binding's one
precedent for calling Python from Rust, and S4.3a's materializer
(`expression/materialize.rs`) turns a tree the core built into Python
objects, reusing the input's objects.

### Divergences visible from Python

| # | Python today | Rust core |
|---|---|---|
| V-1 | Captures are `str` names. Equal names are one capture everywhere; the name must be a non-empty `str` | `Capture` identity handles. Two `Capture::new("x")` are independent; the name may be empty |
| V-2 | `MatchBindings` is keyed by name. It has a public constructor, `try_bind`, the `bindings` mapping and `names()`; `get` raises `KeyError`; `hash` covers keys only | Keyed by `Capture`. There is no public bind; `get` returns `None`, and indexing panics; it has `len` and `iter` in binding order; `hash` covers the entries |
| V-3 | `Pattern` is an open ABC with a public `match_under` | A closed type with no threading API |
| V-4 | Patterns have dataclass `==`, `hash`, `repr` and pickling | Only a derived `Debug` |
| V-5 | Degenerate or ill-typed construction raises `ValueError`: an empty name, alternatives or cases, fewer than two operand patterns, a non-`Pattern` | Construction never fails; the degenerate patterns match nothing; types are checked by the compiler |
| V-6 | `CallExpressionPattern` compares `function_name` strings; `""` matches nothing | Compares `Callee`s; an empty name is not a `Callee` (`FunctionNameError`) |
| V-7 | Predicate and guard results are read by truthiness | `bool` |
| V-8 | One optional `guard` attribute | A list of guards (`with_guard`) |
| V-9 | A rewrite must return an `Expression`; `None` raises `TypeError` | `new` always returns one; `new_partial`'s `None` declines |
| V-10 | Returning the matched node itself fires: it reports its diagnostic, stops later rules, and rebuilds the ancestors | Declines: the next rule is tried, nothing is rebuilt, nothing is recorded |
| V-11 | A shared subtree is rewritten, and its callbacks called, once per occurrence | Once |
| V-12 | Trees and patterns deeper than about 1,000 levels raise `RecursionError` | Trees of any depth; pattern matching recurses on the Rust stack |
| V-13 | `apply_rewrite_rules` returns an `Expression` and raises `PassExecutionError` | Returns a `RewriteOutcome` and raises a `RewriteError`, which names the rule's index and name and has a rebuild variant |
| V-14 | Firings are not recorded, apart from the diagnostics | `RewriteOutcome.fired`, `RewriteRuleApplier::fired` |
| V-15 | Diagnostic `Applied rewrite rule 'n'.` | `applied rewrite rule "n"` |
| V-16 | The applier is a `RewritablePass` with a per-node `visit_unknown` hook; `create(name)` fails for want of rules | A `CompilerPass` with no per-node hook; the registry builds an applier of no rules |
| V-17 | Rules are `RewriteRule`s only | The `Rule` trait, and mixed lists |
| V-18 | `RewriteRule` is equal and hashable by content | No equality |

Unchanged in meaning:

- the matching order and the threading of bindings;
- repeated captures: they need structural equality and keep the first
  object;
- literal and identifier equality, and exact operand counts;
- root-only matching;
- one bottom-up pass that does not revisit a replacement, and the input
  itself as the output when nothing fires;
- unchanged propagation of callback errors from a single match or rule;
- the pass's name and description.

### Consumers and tests

- **`src`.** No module outside the package uses it. The only users are the
  re-exports in `symbolic/expression/__init__.py` (192 lines) and the
  README's feature row. No pass, solver, type checker or constraint code
  matches patterns.
- **Python tests.** 188 tests, all in `tests/symbolic/expression/pattern/`:

  | File | Lines | Tests |
  |---|--:|--:|
  | `test_core.py` | 1,624 | 132 |
  | `test_rewrite.py` | 702 | 42 |
  | `test_user_stories.py` | 267 | 7 |
  | `test_core_properties.py` | 206 | 4 |
  | `test_rewrite_properties.py` | 315 | 3 |

  They draw trees from `tests/strategies/structural_expressions.py`, and
  some use `mock_identifier`. No other test module imports the package.
- **Rust tests**, which already specify the core:

  | File | Lines | Tests |
  |---|--:|--:|
  | `tests/it/expression/pattern/stories.rs` | 1,980 | 105 |
  | `tests/it/expression/pattern/rewrite_stories.rs` | 1,342 | 66 |
  | `tests/it/expression/pattern/properties.rs` | 714 | 12 properties |
  | `tests/it/expression/pattern/user_stories.rs` | 295 | 9 |
  | `tests/it/support/pattern.rs` | 133 | helpers |
  | `tests/it/expression/pass_stories.rs` | 630 | 25 (the applier and the formatter) |
- **Benchmarks.** None cover patterns.

### Pattern choice

- **P2: `Capture`, the pattern hierarchy, `MatchBindings`, `RewriteRule`
  and `FiredRule`.** Patterns are logic-rich machinery, and a Rust walk has
  to hold the patterns and rules, so they go to Rust (decision 2). The
  alternative, P1, would keep the Python dataclasses and convert a rule
  list to Rust patterns on every `apply_rewrite_rules` call. `match` would
  stay Python, and the conversion would be paid per call. The benchmarks
  below check the choice.
- **P3: the callbacks and `Rule`.**
  - A `PredicatePattern`'s predicate, a rule's guards and its rewrite are
    Rust closures holding the Python callable (`Py<PyAny>`). Each call
    attaches to the interpreter, which is already held because the call
    came from Python, then calls it and converts the result. A raised
    exception becomes the `CallbackError`, a boxed `PyErr`, and the
    binding downcasts it back at the boundary.
  - A Python subclass of the new `Rule` ABC is driven through an adapter
    struct holding `Py<PyAny>` that implements the Rust `Rule` trait.
    `RewriteRule`, the Rust implementation, is registered with the ABC.
  - This is the pattern-rule exception that P3's granularity rule allows.
    A callback runs per candidate node only after the Rust pattern around
    it matched; a predicate at a pattern's root runs at every node.
- **`RewriteRuleApplier` stays a Python `CompilerPass` until S6.** Its
  `run_pass` calls the Rust walk.
- **The objects Python sees.** A callback gets the Python object of the
  input tree's node at that place, so `bindings[x] is node` holds. It gets
  a new object, built once, only for a node the walk rebuilt around
  rewritten children. Each replacement a rewrite returns is kept by
  identity, so the output, materialized beside the input as `substitute`'s
  is, shares every object it can.

**Benchmark plan: `benchmarks/test_pattern.py` (S5.1).** It reuses
`test_expression.py`'s deep tree (100 operations over four identifiers)
and its doubling DAG. "A few rules" means four: `x + 0 -> x`, `x * 1 -> x`,
`-(-x) -> x` and `x - x -> 0`. As in S4.1, every call whose spelling S5
changes sits in a helper marked with its decision: `_capture(name)`
(D-S5-2) and `_rewritten(result)` (N-S5-1). The baseline measures today's
Python classes.

| Benchmark | Measures |
|---|---|
| `test_capture_pattern_construction`, `test_binary_expression_pattern_construction`, `test_rewrite_rule_construction` | construction (dataclass `__post_init__` against a pyclass) |
| `test_pattern_attribute_access` | field reads |
| `test_match_of_a_small_pattern[hit]`, `[miss]` | `x + 0` against a match and a root mismatch |
| `test_match_of_a_mirroring_pattern_of_a_deep_tree` | a pattern mirroring the deep tree, capturing every leaf |
| `test_match_of_a_repeated_capture_over_deep_operands` | `x - x` over two distinct, equal deep trees (unification) |
| `test_match_of_alternatives` | four alternatives, the last of which matches |
| `test_match_bindings_get` | reading one capture |
| `test_apply_rewrite_rule_at_the_root` | one rule once: the per-call overhead |
| `test_apply_rewrite_rules_to_a_deep_tree[no_firing]`, `[firing]` | the four rules over the deep tree, firing nowhere, and over a variant in which about a quarter of the nodes fire |
| `test_apply_rewrite_rules_to_a_shared_dag` | the four rules over the DAG (V-11) |
| `test_rewrite_rule_applier_execute_of_a_deep_tree` | the same through the pass, with diagnostics |
| `test_apply_rewrite_rules_with_a_predicate_at_every_node` | callback-heavy: one `PredicatePattern` rule that never fires, so one Python call per node |
| `test_apply_rewrite_rules_with_a_guard_and_rewrite_at_every_leaf` | callback-heavy: a captured leaf pattern with a guard and a rewrite that fire at every leaf, so the cost of building bindings and objects for Python |

After S5.4 the file gains `test_apply_rewrite_rules_with_a_python_rule`, a
`Rule` subclass at every node; it has no baseline. The verdict follows
cross-cutting rule 5: a slower hot path changes pattern or is recorded as
an accepted cost, with numbers. The paths at risk:

- field reads, which stay struct-member reads;
- the per-leaf callback case, where each call builds a `MatchBindings`
  object and looks up node objects. If it loses to today's immutabledict,
  the fix is to build the objects lazily, on first access.

### Decisions (proposed 2026-09-25)

Each names the policy it follows: D-S4-1 (Rust semantics where the two
differ), D-S4-2 (Python names where the meaning is the same), the S6 rule
(`CompilerPass` hook names stay Python), "no fallback", or "tests rewritten,
not skipped". Where a decision follows an S2 to S4 implementation note, it
says so.

- **D-S5-1: one implementation, no fallback** (no pure-Python fallback).
  `core.py` and `rewrite.py` become thin public classes over `_rs`, as the
  expression core did. The dataclasses are deleted, not kept behind a
  switch.
- **D-S5-2: captures are Rust `Capture` handles** (D-S4-1; V-1).
  - A new `Capture(name)` class with identity `==`/`hash`, `name`, `str` as
    the name, and `repr` `Capture('x')`. Any `str` is a valid name,
    including the empty one.
  - `CapturePattern(capture, sub_pattern=WildcardPattern())` has the fields
    `capture` and `sub_pattern`; the default is Rust's `Pattern::capture`.
  - The same `Capture` object in two places means "must be equal". Two
    captures with the same name are independent.

  This is the most visible change. It touches only the pattern tests and
  the README, since `src` has no consumer.
- **D-S5-3: `MatchBindings` has Rust semantics** (D-S4-1; V-2) **and keeps
  Python names where the meaning is the same** (D-S4-2).
  - Only a match produces non-empty bindings. `MatchBindings()` and
    `empty()` are empty.
  - Kept: `is_empty()`, `has(capture)` (and `in`), and `get(capture)`,
    which now returns the expression or `None`.
  - `bindings[capture]` raises `KeyError` with the core's text,
    ``capture `x` is not bound``.
  - `len()`, iteration over the captures in binding order, and `items()`.
  - `==` and `hash` ignore order and compare the expressions structurally.
  - Removed: `try_bind`, `names()`, the `bindings` mapping and the
    constructor from entries.
  - The values are the node objects the match saw.
- **D-S5-4: patterns are a closed P2 hierarchy** (D-S4-1; V-3; S4.3a's
  closed node kinds). The twelve class names, their fields and `None`
  meaning "any" stay (D-S4-2), mapped onto the Rust constructors. `match`,
  `match_pattern` and `does_pattern_match` keep their names (D-S4-2).
  - `match_under` is removed.
  - A Python subclass of `Pattern` that is none of the twelve cannot be
    built (`TypeError`), and a subclass of one of them is that kind.
  - Each class keeps its field objects as struct members (S3/S4 practice),
    so `.value` is the given value and `.operands` a tuple.
  - Rust's `nothing` is not exposed; `AlternativesPattern(())` is the same
    pattern.
- **D-S5-5: construction follows the core** (D-S4-1; V-5, V-6).
  - `LogicalExpressionPattern` with fewer than two operand patterns,
    `PiecewiseExpressionPattern` with `cases=()`, and `AlternativesPattern`
    with no alternative build and match nothing.
  - The binding's own type checks raise `TypeError` in S2's style
    (`BinaryExpressionPattern left must be a Pattern, got int.`), per the
    "stricter arguments" notes of S2 to S4.
  - `LiteralPattern` keeps raising `LiteralExpression`'s errors.
  - `CallExpressionPattern` parses `function_name` with `Callee::from_str`,
    so `""` raises the core's `FunctionNameError` as `ValueError`, as
    `CallExpression("")` does, and matching compares callees.
- **D-S5-6: callback results are read by truthiness** (D-S4-2; V-7). The
  meaning ("the guard allows it") is the same, and Rust's `bool` has no
  counterpart for other Python values, so the binding takes
  `bool(result)`. An exception from `__bool__` is the callback's error.
- **D-S5-7: callback errors.** Two rules:
  - The Rust core returns a `CallbackError` unchanged from `match`,
    `match_pattern`, `does_pattern_match`, `apply_rewrite_rule` and
    `RewriteRule.apply`, so the same exception object propagates (D-S4-1).
  - An exception that is not an `Exception` (`KeyboardInterrupt`,
    `SystemExit`, `GeneratorExit`) always propagates unchanged and is
    never wrapped in `RewriteError`, so `except Exception` cannot swallow
    it (D-S4-2: the core does not distinguish them).
- **D-S5-8: `RewriteRule` has the core's semantics** (D-S4-1; V-8, V-9,
  V-10, V-18) **under Python names** (D-S4-2).
  - `RewriteRule(pattern, rewrite, guard=None, name=None)` is `new`, plus
    `with_guard` and `with_name`.
  - `RewriteRule.new_partial(pattern, rewrite, guard=None, name=None)`
    lets a rewrite return `None` to decline.
  - `with_guard(guard)` and `with_name(name)` return new rules.
  - The fields are `pattern`, `rewrite`, `guards` (a tuple, which replaces
    `guard`) and `name`.
  - `apply(expression)` is the method; `apply_rewrite_rule(rule,
    expression)` stays as its free-function form.
  - A rewrite result that is neither an `Expression` nor, for
    `new_partial`, `None` raises `TypeError` naming the rule
    (`RewriteRule 'n' rewrite must return an Expression, got NoneType.`).
  - A result that is the matched node itself declines (F-009).
  - Rules compare and hash by identity, since callbacks cannot be
    compared. They are virtual `FrozenMixin`s.
- **D-S5-9: a `Rule` ABC** (P3 in the slice table; V-17). `class
  Rule(_rs.RuleBase, ABC)` declares the abstract `apply(expression) ->
  Expression | None` and a `name` property that is `None` by default. These
  are the Rust hook names, since Python has no counterpart. `RewriteRule`
  is registered as a virtual subclass. A rule list may mix both kinds. A
  Python rule's `name` is read once per walk.
- **D-S5-10: the walk is the core's** (D-S4-1; V-10, V-11, V-12).
  - A shared node is rewritten once.
  - A rule returning its input declines, and the next rule is tried.
  - A tree of any depth is walked.
  - When nothing fires, the output is the input object itself; otherwise
    it shares the input's objects wherever the core kept the node.
- **D-S5-11: the walk's errors are the core's** (D-S4-1: errors map
  through `IntoPyErr`; V-13).
  - `pattern/rewrite.py` defines `RewriteError(RuntimeError)`, with
    `rule_index` and `rule_name`, and a subclass for each variant:
    `RewriteCallbackError` and `RewriteRebuildError`.
  - The message is the core's `Display`, and `__cause__` is the callback's
    exception or the rebuild's `ValueError`.
  - The base is `RuntimeError` because `PassExecutionError`, which it
    replaces at the free function, is one, so `except RuntimeError` keeps
    catching it.
  - `apply_rewrite_rules` raises it directly.
- **D-S5-12: `RewriteRuleApplier` stays a Python `CompilerPass` until S6**
  (the S6 rule; V-14, V-15, V-16). It is `CompilerPass[Expression,
  Expression]`, registered under the same name and description, with the
  Python hooks `run_pass`, `get_noop_output` and `did_change` (identity).
  - It is no longer a `RewritablePass` (D-S4-1: the Rust applier is no
    per-node visitor), so `visit_unknown`, `visit` and `transform` go.
  - `RewriteRuleApplier(rules=())` defaults to no rules, so
    `CompilerPass.create(NAME)` builds one, as the Rust registry does.
  - `rules` is a tuple, and the new `fired` holds the `FiredRule(rule_index,
    name)` values of the last run, or the firings before the failure after
    a failed run.
  - Each named firing reports INFO `applied rewrite rule "name"` (the core's
    text, D-S4-1).
  - A `RewriteError` fails the run through `execute`'s guard as
    `PassExecutionError`, with the `RewriteError` as `__cause__`.
- **D-S5-13: pickles** (S3/S4 practice).
  - Patterns, captures and rules pickle as a call of their class with their
    fields. A capture shared within one pickle stays shared, through the
    pickle memo.
  - A `PredicatePattern` or a rule pickles only if its callables do, as
    today.
  - `MatchBindings` does not pickle (`TypeError`): only a match produces
    bindings (D-S4-1).
- **D-S5-14: reprs** (D-S4-2).
  - The core has no `Display` for patterns, only a derived `Debug`, so no
    Rust text convention applies. Patterns keep Python's dataclass-style
    `repr`, rendered from the field objects.
  - A capture is `Capture('x')`.
  - Bindings are `MatchBindings({Capture('x'): LiteralExpression(1)})`.
  - A rule is `RewriteRule(pattern=..., name='n')`, callables omitted, as
    the core's `Debug` omits them.
- **D-S5-15: a depth guard for patterns** (S3b's recursion guard). The
  Rust matcher recurses once per pattern level, so each pattern records its
  depth. `match` and the walk raise `RecursionError` for a pattern deeper
  than `sys.getrecursionlimit()`, instead of overflowing the stack. Trees
  have no limit (D-S5-10).
- **D-S5-16: tests are rewritten, not skipped** (the tests rule).
  - The 188 behavioral Python tests stay, as S4.4 kept them, and are
    rewritten to the new semantics. Each rename or rewrite is recorded with
    its reason, as S4.4's table does.
  - The Rust tests already specify the core. New Rust tests come only with
    core additions (S5.2).

### Needs the user

- **N-S5-1: what `apply_rewrite_rules` returns.** The function name and
  its meaning stay: one bottom-up pass. The Rust function returns a
  `RewriteOutcome` (`output`, `is_changed`, `fired`), where Python returns
  the tree. D-S4-1 and D-S4-2 point opposite ways here, since the result's
  meaning is the same and its shape is not.
  - (a) **Return the `Expression`, as today.** Firings are available
    through `RewriteRuleApplier.fired` and, as today, through
    `PassResult.changed` and its diagnostics. Callers change only for
    D-S5-2.
  - (b) **Return a `RewriteOutcome` pyclass**, with `output`, `is_changed`
    and `fired`. Every caller adds `.output`: 18 of the 42 rewrite tests,
    6 of the 7 user stories and the 3 rewrite properties.
  - (c) **Both:** (a) plus a new `apply_rewrite_rules_with_outcome`.

  Recommendation: (a). The free function is the convenience form, as
  `CompilerPass.__call__` is beside `execute`. The pass already gives
  `changed`, and it gains `fired`. Either way it raises `RewriteError`
  (D-S5-11).

### Steps

1. **S5.1: benchmarks.** Add `benchmarks/test_pattern.py` as planned above
   and record the baseline here, on today's Python classes.
2. **S5.2: core additions, with Rust tests.** None are expected. The
   binding needs only public API: the `Pattern` constructors, `Capture`,
   `MatchBindings::iter`, `RewriteRule` and its builders, `Rule`,
   `apply_rewrite_rules`, `RewriteError`'s accessors, and
   `NodeHandle::identity` for expressions. There is one candidate: the
   Python applier must report the firings before a failure (D-S5-12),
   which the public
   `apply_rewrite_rules` drops. The binding records them in its own rule
   adapters. If that duplicates the walk's firing rule too closely, the
   fallback is a public `RewriteError::fired()`, which would need a Rust
   test and an update of the crate's README.
3. **S5.3: the binding.** Add `rust/fhy-core-py/src/expression/pattern.rs`,
   with submodules for captures, patterns, bindings, rules and the walk.
   - **Pyclasses.** `_rs.Capture`; a `_rs.Pattern` base,
     `#[pyclass(subclass, frozen)]`, holding the Rust `Pattern`, its depth,
     and the `Capture` objects it contains, with one `extends` class per
     pattern kind holding its field objects; `_rs.MatchBindings`, holding
     the Rust bindings and the `(Capture, node)` objects in binding order;
     `_rs.RewriteRule`, holding the Rust rule and its Python callables;
     `_rs.RuleBase`; `_rs.FiredRule`.
   - **The object table.** A per-call table maps a node's identity to its
     Python object. It is kept on a thread-local stack while a match or a
     walk runs, so nested calls from callbacks work; the interpreter is held
     throughout.
     - It is filled from the input tree's objects on first need, and from
       each replacement a rewrite returns.
     - A node without an object is built once from its children's
       objects, as S4.3a's materializer does.
   - **Adapters and results.** The callback closures and the Python `Rule`
     adapter hold `Py<PyAny>`. `RewriteError` maps onto the Python classes
     of D-S5-11. The public classes register with the binding, as S3's do.
   - **Stubs.** Everything new goes into `_rs.pyi`.
4. **S5.4: the Python switch.**
   - `pattern/core.py` and `rewrite.py` define the public classes over
     `_rs` (with `FrozenMixin` virtual membership), the `Rule` ABC, the
     error classes, and `RewriteRuleApplier` as a Python `CompilerPass`
     calling the Rust walk.
   - The package's `__all__` and `symbolic/expression/__init__.py` gain
     `Capture`, `Rule`, `FiredRule`, `RewriteError`, `RewriteCallbackError`
     and `RewriteRebuildError`.
   - The README's feature row changes.
5. **S5.5: tests.** Migrate the pattern tests and add the interface suite
   (the test plan below).
6. **S5.6: benchmarks after,** recorded here with the verdict, then the
   status, the implementation notes and this checklist.

Commit per step. Every step ends with `pytest` and `-m "not very_slow"`
green, the `property` session, `lint` and `type_check` clean, and the Rust
gate green (fmt, clippy `-D warnings`, tests, doc `-D warnings`, deny,
`cargo +1.85 check`).

### Test plan

**The interface suite:** `tests/symbolic/expression/pattern/test_pattern_rust_binding.py`,
next to S4.3a's `test_expression_rust_binding.py`. It covers what the
binding adds over the core:

- **Class structure and registration.** Each pattern class extends its
  `_rs` class, and a new `Pattern` kind cannot be built. `FrozenMixin`
  membership holds, and so does `isinstance(rule, Rule)` for a
  `RewriteRule`. The frozen errors are raised.
- **Argument checks and their messages.** The `TypeError`s; the
  degenerate patterns that build and match nothing; `""` as a call name;
  the literal errors.
- **`Capture`.** Identity equality and hash, the name (the empty one
  included), and `repr`.
- **`MatchBindings`.** `get`, `[]`, `has`, `in`, `len`, iteration order,
  equality and hash. No public constructor binds. The bound objects are
  the input's (`is`), and a rebuilt node's object is built once.
- **Callbacks.**
  - The arguments are the right objects, and results are read by
    truthiness.
  - The same exception object propagates from `match` and
    `apply_rewrite_rule`, and a `KeyboardInterrupt` passes through a walk
    unwrapped.
  - Nested matches and walks inside a callback work.
  - The rewrite-result `TypeError`, `new_partial`'s decline, and a result
    that is the matched node itself, which declines so the next rule fires.
- **The walk.** A shared subtree is rewritten once (by a callback count),
  and the input comes back itself when nothing fires. A 20,000-level tree
  rewrites. A pattern deeper than the recursion limit raises
  `RecursionError` and does not crash.
- **Walk errors.** The `RewriteError` subclasses, `rule_index`,
  `rule_name` and `__cause__`, and the rule a refused rebuild blames.
- **`Rule`.** Abstract methods are enforced. A Python subclass is driven
  from Rust, mixed with `RewriteRule`, and its name appears in the errors
  and diagnostics.
- **The applier.**
  - It is registered, and `CompilerPass.create(NAME)` builds one with no
    rules.
  - The diagnostic text, `fired` after a success and after a failure, and
    `PassExecutionError` wrapping the `RewriteError`.
  - `changed` is by identity.
- **Pickles.** Patterns, a capture shared within one pickle, and rules
  with module-level callables round-trip. A lambda predicate fails as
  pickle fails. `MatchBindings` refuses.
- **Stubs.** `tests/test_rs_stub.py` covers the new stubs.

**Migrating the existing 188 tests.** No test is skipped or deleted
without a rewrite, and each change is recorded with its reason:

- **`test_core.py` (132).**
  - Every capture is spelled with a `Capture` (D-S5-2).
  - The 18 `MatchBindings` tests are rewritten through matching (D-S5-3).
    `try_bind`'s first-bound and repeated-bind laws become repeated-capture
    matches; the constructor type checks become "no public constructor
    binds"; `get` of an unbound capture pins `None`, and `[]` pins
    `KeyError`.
  - `test_capture_pattern_siblings_with_shared_name_share_the_same_capture`
    is rewritten to pin that one `Capture` is shared and that two
    same-named captures are independent.
  - The construction-error tests for empty cases, empty alternatives,
    fewer than two operands and an empty name are rewritten to pin
    "builds and matches nothing" (D-S5-5).
  - The non-`Pattern` tests pin `TypeError`.
  - The two `match_under` tests
    (`test_wildcard_pattern_match_under_returns_input_bindings`,
    `test_predicate_pattern_captures_nothing`) are rewritten through a
    compound pattern whose other capture shows the threading (D-S5-4).
  - The coercion and list-mutation tests keep their meaning.
- **`test_rewrite.py` (42).**
  - `test_rewrite_rule_is_hashable_and_equal_by_content` is rewritten to
    identity (D-S5-8), and `test_rewrite_rule_defaults_for_optional_fields`
    to `guards == ()`.
  - The two tests that pin `PassExecutionError` from `apply_rewrite_rules`
    are rewritten to `RewriteCallbackError` with its cause (D-S5-11). The
    two applier tests keep `PassExecutionError` and add the cause.
  - The diagnostic test pins the core's text (D-S5-12).
  - `test_apply_rewrite_rules_with_identity_rewrite_reports_unchanged` is
    extended with the next rule firing (D-S5-10).
  - `test_rewrite_rule_accepts_arbitrary_pattern_subclasses` is rewritten
    to "accepts every pattern kind" (D-S5-4).
  - The rest change only by D-S5-2 and N-S5-1.
- **`test_user_stories.py` (7)** and the two property modules (7) change
  only by D-S5-2 and N-S5-1. The "identity holds iff no rule fired"
  property also checks `RewriteRuleApplier.fired`.
