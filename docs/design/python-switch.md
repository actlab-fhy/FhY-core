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
- [x] S5: patterns and rewrite rules (N-S5-1 resolved as (a)). The suite is green (7,054 passed), slow tests pass (7,087), properties pass (280), lint and mypy are clean, and the Rust gate passes (2,630)
  - [x] S5.1: pattern benchmarks and baseline
  - [x] S5.2: core additions, if any, with Rust tests (none were needed)
  - [x] S5.3: the pattern binding
  - [x] S5.4: the Python switch
  - [x] S5.5: pattern tests migrated, and the interface suite
  - [x] S5.6: benchmarks after, and docs
- [x] S6: pass infrastructure (`CompilerPass`, `Analysis`, `Validator`, managers). The suite is green (7,163 passed), slow tests pass (7,196), properties pass (280), lint and mypy are clean, and the Rust gate passes (2,660)
  - [x] N-S6-1 to N-S6-3 decided (2026-09-25; see "S6 resolutions")
  - [x] S6.1: pass-infrastructure benchmarks and baseline
  - [x] S6.2: core additions, with Rust tests (`NodeIdentity::of_ptr`, analysis ids from an `Identifier`, the detached analysis cache)
  - [x] S6.3: the `ValidationReport` representation (D-S6-17)
  - [x] S6.4: the pass binding
  - [x] S6.5: the Python switch
  - [x] S6.6: tests migrated, and the interface suite
  - [x] S6.7: benchmarks after, and docs
- [x] Final verification (2026-09-25). The Rust CI jobs replay cleanly, and so do nox `lint`, `type_check`, `tests` on 3.10 to 3.14 with `FORCE_COLOR=1` as in CI, `coverage`, `property` and `golden_expanded`. Two 3.13+-only test issues this found are fixed: colored tracebacks, and `pathlib._local`
- Leftovers:
  - [x] the `ValidationReport` construction cost (S6.3)
  - [x] the unknown-provenance `str` cost: accepted as a recorded cost of about 20 ns, the fixed price of calling into the extension
  - [x] Windows paths: by design (crate decision D-16), provenance paths use platform-independent POSIX normalization, while the old pure-Python class kept a `pathlib.Path` as given (`WindowsPath` on Windows). Watch the Windows CI leg of release pull requests
  - [x] mypy over the Rust branches (S4.4)
  - [x] slow callee-name parsing in the core: a user-function call is built in 1.6 us, down from 4.1 us, because the variant-name parser no longer formats serde's list of variants
  - [x] platform wheels in the release workflow, now that the extension is required (S4.4). `python-release.yml` builds maturin wheels for Linux (x86_64 and aarch64, manylinux), macOS (x86_64 and arm64) and Windows x64, one per CPython 3.10 to 3.14, plus an sdist, and publishes them all with trusted publishing. The builds and a wheel install were checked locally; the workflow itself first runs on the next release
- [x] S7: the function registry (N-S7-1 to N-S7-3 resolved as (a)). The suite is green (7,327 passed), slow tests pass (7,360), properties pass (281), lint and mypy are clean, `tests-3.13` passes with `FORCE_COLOR=1`, and the Rust gate passes (2,808)
  - [x] N-S7-1 to N-S7-3 decided (2026-09-25; see "S7 resolutions")
  - [x] S7.1: registry benchmarks and baseline
  - [x] S7.2: core additions, test-first, with Rust tests (`FunctionRegistry`, `FunctionSort::admits`, built-in constant identifiers, the screen's constant rule, `FunctionRegistry::inline`)
  - [x] S7.3: the registry binding, the screen on the Rust registry, and the built-in bodies' differential check
  - [x] S7.4: the Python switch
  - [x] S7.5: tests migrated, and the interface suite
  - [x] S7.6: benchmarks after, and docs

## Goal

`import fhy_core` exposes one Python API. Each switched concept runs on
the Rust implementation, and the extension is required (D-S4-6 retired the
pure-Python backend). Existing Python code, including user subclasses of
the framework classes, keeps working as far as the recorded decisions
allow.

Until S4.4, a switched concept was *defined in both languages*: its
Python-visible behavior had to be identical on both backends, and the
Python test suite run on both backends was the equivalence harness. For
expressions and everything after them, the Python API takes the Rust
semantics where the two differ, and keeps the Python names where the
meaning is the same (D-S4-1, D-S4-2). The binding crate (`fhy-core-py`)
adapts the idiomatic Rust core to the Python API. The core crate does not
bend to Python.

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

- **Status:** designed 2026-09-25 at fbbb5af, and implemented the same
  day; see "S5 status" below. D-S5-1 to D-S5-16 apply the policy the user
  already set, and N-S5-1 was resolved as option (a).
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

- **N-S5-1 (resolved 2026-09-25, as option (a)): what `apply_rewrite_rules` returns.** It returns the `Expression`, as today. The pass exposes the firings. The function name and
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

### S5.1 baseline (2026-09-25, 1207faf plus the new benchmarks)

`benchmarks/test_pattern.py` implements the benchmark plan above. It
imports the deep tree, the doubling DAG and their sizes from
`test_expression.py`. The "firing" variant of the deep tree replaces every
fourth multiplication with an addition of zero, so `x + 0 -> x` fires 25
times in its 100 levels. The leaf benchmark negates every identifier
reference, so each call reads a binding and builds a node. `_capture(name)`
is the one helper marked with a decision (D-S5-2). N-S5-1's resolution, (a),
keeps `apply_rewrite_rules` returning the expression, so the planned
`_rewritten(result)` helper is not needed.

Median time per call, from `uv run --python 3.11 nox -s benchmark-3.11 --
-k test_pattern` on the S0 machine with Python 3.11.13 and pytest-benchmark
5.3.0, measuring today's pure-Python pattern classes. The load average was
about 1, and the table lists the best of three runs' medians.

| Benchmark | before |
|---|--:|
| `test_capture_pattern_construction` | 880 ns |
| `test_binary_expression_pattern_construction` | 1.15 µs |
| `test_rewrite_rule_construction` | 1.06 µs |
| `test_pattern_attribute_access` | 90 ns |
| `test_match_of_a_small_pattern[hit]` | 3.10 µs |
| `test_match_of_a_small_pattern[miss]` | 1.28 µs |
| `test_match_of_a_mirroring_pattern_of_a_deep_tree` | 967.6 µs |
| `test_match_of_a_repeated_capture_over_deep_operands` | 8.15 µs |
| `test_match_of_alternatives` | 2.39 µs |
| `test_match_bindings_get` | 118 ns |
| `test_apply_rewrite_rule_at_the_root` | 3.52 µs |
| `test_apply_rewrite_rules_to_a_deep_tree[no_firing]` | 1.88 ms |
| `test_apply_rewrite_rules_to_a_deep_tree[firing]` | 1.98 ms |
| `test_apply_rewrite_rules_to_a_shared_dag` | 15.2 ms |
| `test_rewrite_rule_applier_execute_of_a_deep_tree` | 1.99 ms |
| `test_apply_rewrite_rules_with_a_predicate_at_every_node` | 568.0 µs |
| `test_apply_rewrite_rules_with_a_guard_and_rewrite_at_every_leaf` | 925.7 µs |

The walk dominates: about 9 µs per node of the deep tree whether or not a
rule fires, since each node runs the Python visitor dispatch and tries four
Python patterns, and 7 µs per occurrence of the DAG, whose 2,047
occurrences are each walked. Matching a 100-level mirroring pattern takes
about 1 ms, a small pattern 1.3 to 3.1 µs.

### S5 status

S5 was implemented on 2026-09-25 in seven commits: the benchmarks and
their baseline (4ff400a); the binding, with no core addition (a9ef7b5);
the Python switch, marked breaking (31bdf7e); the migrated pattern tests
and the new interface suite (7a4d01b); a cleanup of that suite's
overrides (b034fe2); the Python-rule benchmark (10d81b1); and these docs.
No pattern test was skipped or deleted. At the end: `pytest` 7,054
passed (188 pattern tests became 199 after the rewrites, plus the
interface suite's 94), `-m "not very_slow"` 7,087 passed, the `property`
session 280 passed, `lint` and `type_check` clean, and the Rust gate green
(fmt, clippy `-D warnings`, 2,630 tests, doc `-D warnings`, deny,
`cargo +1.85 check`).

### S5 benchmarks (before and after)

Median time per call, from `uv run --python 3.11 nox -s benchmark-3.11 --
-k test_pattern` on the S0 machine with Python 3.11.13 and
pytest-benchmark 5.3.0. "Before" is 4ff400a, the S5.1 baseline's tree, run
from a detached worktree under `target/`; "after" is 10d81b1. The two ran
three times each, interleaved (before, then after, in each round), with a
load average of 4 to 6, and the table lists the best of the three
medians. The "before" column agrees with the S5.1 table within 10%.

| Benchmark | before | after | after / before |
|---|--:|--:|--:|
| `test_capture_pattern_construction` | 929 ns | 239 ns | 0.26 |
| `test_binary_expression_pattern_construction` | 1.21 µs | 256 ns | 0.21 |
| `test_rewrite_rule_construction` | 1.08 µs | 617 ns | 0.57 |
| `test_pattern_attribute_access` | 96 ns | 88 ns | 0.92 |
| `test_match_of_a_small_pattern[hit]` | 3.37 µs | 983 ns | 0.29 |
| `test_match_of_a_small_pattern[miss]` | 1.33 µs | 232 ns | 0.17 |
| `test_match_of_a_mirroring_pattern_of_a_deep_tree` | 1.04 ms | 37.6 µs | 0.04 |
| `test_match_of_a_repeated_capture_over_deep_operands` | 8.65 µs | 5.72 µs | 0.66 |
| `test_match_of_alternatives` | 2.55 µs | 703 ns | 0.28 |
| `test_match_bindings_get` | 127 ns | 93 ns | 0.73 |
| `test_apply_rewrite_rule_at_the_root` | 3.78 µs | 1.31 µs | 0.35 |
| `test_apply_rewrite_rules_to_a_deep_tree[no_firing]` | 2.01 ms | 28.8 µs | 0.01 |
| `test_apply_rewrite_rules_to_a_deep_tree[firing]` | 2.08 ms | 130.7 µs | 0.06 |
| `test_apply_rewrite_rules_to_a_shared_dag` | 15.45 ms | 2.82 µs | 0.0002 |
| `test_rewrite_rule_applier_execute_of_a_deep_tree` | 2.07 ms | 225.8 µs | 0.11 |
| `test_apply_rewrite_rules_with_a_predicate_at_every_node` | 596.5 µs | 46.5 µs | 0.08 |
| `test_apply_rewrite_rules_with_a_guard_and_rewrite_at_every_leaf` | 955.9 µs | 179.4 µs | 0.19 |
| `test_apply_rewrite_rules_with_a_python_rule` | - | 44.7 µs | - |

Every hot path is faster, so P2 stands, no path changes pattern, and no
cost needs accepting (cross-cutting rule 5):

- **Construction** of a pattern is 4 to 5 times faster: a type check per
  field and a Rust pattern, with no dataclass `__post_init__`. A rule
  builds its Rust closures and collects its pattern's captures, so it
  gains less, 1.75 times.
- **Field reads** stay struct-member reads. `bindings[x]` is a scan of a
  short tuple by identity (`_read` in the "after" run; `get(name)`, a
  dict lookup, before).
- **Matching** a small pattern is 3 to 6 times faster, a 100-level
  mirroring pattern 28 times: the match runs in Rust, and building the
  bindings of 100 captures finds each node object in one walk of the
  input's objects. The repeated capture over deep operands gains least,
  1.5 times, since the structural comparison of the two deep trees
  dominates both.
- **The walk** with native rules only never calls Python: over the deep
  tree it drops from 2 ms to 29 µs, and over the DAG, which it walks once
  per distinct node (V-11), from 15 ms to 2.8 µs. Where rules fire, each
  firing calls a Python rewrite with a bindings object and the rebuilt
  spine is materialized once: 16 times faster. The pass adds its
  diagnostics and their logging.
- **The callback-heavy walks** are 5 to 13 times faster: a Python call per
  node remains, but the walk, the matching and the dispatch around it run
  in Rust. The per-leaf callback case, the path the plan put at risk, is
  5 times faster with eager bindings objects, so they are not made lazy.
  A Python `Rule` costs about 230 ns per node of the deep tree's 191, a
  predicate about the same.

### S5 implementation notes

Choices the decisions left open, made while implementing S5:

- **Core additions: none (S5.2).** The binding uses the public API of
  `fhy_core::expression::pattern`. The Python applier's firings before a
  failure (D-S5-12) are recorded by the binding's walk-rule wrapper, which
  records a firing when its rule returns a replacement that is not the
  node it was tried on, the same comparison the core's walk makes; so
  `RewriteError::fired()` was not needed. A pattern nested 200,000 levels
  deep was checked to deallocate without overflowing the stack, so
  `PatternKind` needed no iterative `Drop`. No Rust test was added.
- **Module layout.** `rust/fhy-core-py/src/expression/pattern.rs`, with
  `capture.rs`, `kinds.rs` (the `Pattern` base and the eleven kinds),
  `bindings.rs`, `rules.rs` (`RewriteRule`, `RuleBase`, the Python-rule
  adapter, `FiredRule`), `walk.rs` and `objects.rs` (the object table).
  `OptionalArgument` and `format_dataclass_repr` moved from the
  diagnostics and provenance bindings to `dataclass.rs`.
- **The object table** (`objects.rs`) is the plan's per-call table, on a
  thread-local stack while a match, a rule's `apply` or a walk runs:
  - It maps a Rust node's identity to its Python object and holds the node,
    so the identity stays unique. It is filled on first need by walking
    the input's objects beside their handles, recording every object it
    passes, and from each replacement a callback returns.
  - A node the walk rebuilt gets an object built once through the public
    class of its kind. That object holds its own, equal, Rust handle, so
    the table also maps the object back to the node it stands for: a
    callback returning it returns that node, and a rule returning the node
    it matched still declines.
  - The last bindings object is reused for the same bindings, so a rule's
    guards and rewrite receive one object. No borrow of a table is held
    across a call into Python code. CONTRIBUTING's "Process-global state"
    section records the stack, which is empty whenever no match or walk
    runs.
- **Pattern equality and hashing follow the fields, as the dataclasses'
  did.** D-S5-4 and D-S5-14 left them open: the core has no `PartialEq`
  for patterns, and D-S5-8 gives rules identity equality because
  callables cannot be compared. Patterns keep the dataclasses' `==` and
  `hash` over their field objects, as they keep their `repr`: a capture
  compares by identity and a predicate by its callable's `==`, so
  `CapturePattern(x) == CapturePattern(x)`, but a pattern loaded from a
  pickle, which holds new captures, equals the original only if it has
  none. `==` requires exactly the same class, as a dataclass's does.
- **`MatchBindings` truthiness.** With `len()` (D-S5-3), bindings of no
  capture are falsy, as an empty container is; the dataclass was always
  true. A match is tested with `is not None`, as the package and its tests
  always did, but a caller writing `if pattern.match(e):` for a pattern
  without captures would now read a match as a miss. The docstring says
  so; the maintainer may prefer a `__bool__` that is always true.
- **`MatchBindings` details.** `get(key)` returns `None`, and `has(key)`
  and `in` are false, for a key that is not a bound `Capture` (anything
  else included); `bindings[key]` raises `KeyError` with the core's text,
  using `str(key)` for the name. `items()` returns a tuple of pairs.
  Hashing is the core's, over the entries; the bindings hold the node
  objects the match saw. The public class is built through a private seed
  class, as S2's tags are, so `MatchBindings(anything)` raises
  `TypeError` (`only a match produces bindings that bind a capture`).
- **Rules.** `RewriteRule.new_partial`, `with_guard` and `with_name` build
  new rules through a private seed; the constructor's `rewrite` defaults
  to `None` at runtime only, for the seed, and the stub keeps it required.
  A rule pickles as `RewriteRule._from_fields(pattern, rewrite, guards,
  name, is_partial)`, a class method, since `guards` has no constructor
  argument (D-S5-13 said "a call of their class"). The rewrite-result
  `TypeError` keeps `'<unnamed>'` for an unnamed rule, so the old test
  keeps its meaning: `RewriteRule '<unnamed>' rewrite must return an
  Expression, got int.`, and `... an Expression or None, ...` for a
  partial rule. `apply_rewrite_rule(rule, e)` is `rule.apply(e)`, so it
  takes a Python `Rule` too. mypy does not see `Rule.register`, so the
  functions and the applier are typed with `Rule | RewriteRule`.
- **Python rules.** `apply_rewrite_rules` and the walk take any
  `_rs.RewriteRule` natively, any `_rs.RuleBase` instance through the
  adapter, and refuse anything else (`apply_rewrite_rules rules must be
  Rules, got int.`). A Python rule's `name` must be a `str` or `None`,
  and its `apply` must return an `Expression` or `None`; anything else
  raises `TypeError` (`_Wrong.apply must return an Expression or None, got
  int.`), which the walk reports as the rule's `RewriteCallbackError`.
- **The walk's entry point.** `_rs.apply_rewrite_rules(expression, rules,
  fired=None)` appends each firing to the list `fired` if given, also on
  failure; the public function passes two arguments, and the applier its
  list. A private top-level `_rs` function would not do: the stub test
  counts every stub name but only the extension's public ones.
- **`FiredRule`** has a public constructor `FiredRule(rule_index, name)`,
  compares and hashes by its fields, prints as a dataclass, and pickles as
  a call of its class. Its private `_diagnostic_message()` renders the
  core's text, `applied rewrite rule "name"`, with the name escaped as
  Rust's `Debug` writes a string, which Python's `repr` would not match.
- **Errors.** The three error classes live in `pattern/rewrite.py` and
  pickle with their fields. The binding imports them when it raises one,
  and sets `__cause__` to the callback's exception or to the rebuild's
  `ValueError`, whose text is the core's `RebuildError`.
- **Stricter arguments**, beyond D-S5-5's list: a piecewise case that is
  not a pair of patterns raises `TypeError` (`PiecewiseExpressionPattern
  cases must be (condition, value) pairs of Patterns, got int.`), where it
  raised `ValueError`; a predicate must be callable, an identifier pattern
  an `Identifier` or `None`, a call pattern's name a `str` or `None`, a
  rule's rewrite and guards callables and its name a `str`, each raising
  `TypeError`; the dataclasses accepted anything there. `match`, `apply`
  and the walk refuse a non-expression with `TypeError`. An operation
  given by value (`"negate"`) is stored as its member.
- **The depth guard** (D-S5-15) reads the recursion limit only for a
  pattern deeper than 64 levels, as S3b's does, and raises
  `maximum recursion depth exceeded: the pattern is N levels deep`. With
  the recursion limit raised, a deep enough pattern still overflows the
  stack: 20,000 levels matched and 50,000 crashed with a limit of
  100,000, as S3b recorded for provenance.
- **Public classes.** Each is a thin Python subclass with `__slots__ =
  ()`, a virtual `FrozenMixin` (and `RewriteRule` a virtual `Rule`), and
  registers its public class; the pattern kinds keep `@final` for type
  checkers and set `__match_args__`. `_rs.Pattern` has no constructor, so a
  new kind raises PyO3's `TypeError` ("No constructor defined").
- **Benchmarks.** `_read(bindings, capture)` joined `_capture` as a
  helper whose spelling S5 changes (`bindings.get(name)` before,
  `bindings[capture]` after), and `test_apply_rewrite_rules_with_a_python_rule`
  was added after the switch, as planned.

Tests migrated in S5.5 (`test_core.py` is `C`, `test_rewrite.py` `R`); every
other test changed only by D-S5-2 (captures) or not at all, and none was
skipped or deleted. `test_user_stories.py`, `test_core_properties.py` and
`test_rewrite_properties.py` changed only by D-S5-2, except that the
mirror property builds its altered pattern with the constructor instead
of `dataclasses.replace`, and the identity property also checks
`RewriteRuleApplier.fired`.

| Test | Now | Reason |
|---|---|---|
| C `test_match_bindings_try_bind_records_first_binding` | `test_match_bindings_of_a_capture_match_record_the_binding` | D-S5-3: only a match binds |
| C `test_match_bindings_try_bind_leaves_receiver_untouched` | `test_matching_leaves_earlier_bindings_untouched` | as above |
| C `test_match_bindings_try_bind_repeated_with_equivalent_returns_self` | `test_repeated_capture_of_equal_expressions_binds_the_capture_once` | `try_bind`'s law as a repeated-capture match |
| C `test_match_bindings_try_bind_retains_original_expression_on_reconfirm` | `test_repeated_capture_keeps_the_first_bound_expression` | as above |
| C `test_match_bindings_try_bind_repeated_with_distinct_returns_none` | `test_repeated_capture_of_distinct_expressions_does_not_match` | as above |
| C `test_match_bindings_try_bind_repeated_with_equivalent_compound` | `test_repeated_capture_of_equal_compound_expressions_matches` | as above |
| C `test_match_bindings_get_raises_key_error_for_unbound_name` | `test_match_bindings_get_returns_none_for_an_unbound_capture`, and new `test_match_bindings_index_raises_key_error_for_an_unbound_capture` | D-S5-3: `get` returns `None`, `[]` raises `KeyError` |
| C `test_match_bindings_has_reports_bound_name` | `test_match_bindings_has_reports_bound_capture` | captures, and `in` |
| C `test_match_bindings_names_after_multiple_binds` | `test_match_bindings_iterate_over_bound_captures_in_binding_order` | `names()` is gone; iteration and `len` |
| C `test_match_bindings_hash_collides_on_matching_key_sets` | `test_match_bindings_equality_and_hash_ignore_binding_order` | the hash covers the entries now; order is what it ignores |
| C `test_match_bindings_with_different_key_sets_are_unequal` | `test_match_bindings_with_different_captures_are_unequal` | captures instead of names |
| C `test_match_bindings_post_init_rejects_non_string_keys` | `test_match_bindings_constructor_refuses_an_argument` | no public constructor binds |
| C `test_match_bindings_post_init_rejects_non_expression_values` | `test_match_bindings_constructed_without_arguments_bind_nothing` | as above |
| C the other five `MatchBindings` tests | same names | rewritten through matching |
| C `test_wildcard_pattern_match_under_returns_input_bindings` | `test_wildcard_pattern_keeps_the_bindings_threaded_to_it` | D-S5-4: through a sibling capture |
| C `test_predicate_pattern_captures_nothing` | same name | as above |
| C `test_capture_pattern_stores_name_as_plain_string` | `test_capture_pattern_stores_its_capture_and_a_wildcard_by_default` | D-S5-2: the field is `capture` |
| C `test_capture_pattern_siblings_with_shared_name_share_the_same_capture` | `test_capture_pattern_siblings_share_one_capture_object` | D-S5-2: one object is shared, same-named ones are independent |
| C `test_capture_pattern_rejects_a_non_string_name` | `test_capture_pattern_rejects_a_capture_that_is_not_a_capture` | D-S5-2 |
| C `test_capture_pattern_rejects_an_empty_name` | `test_capture_with_an_empty_name_binds_like_any_other` | D-S5-2: any `str` is a name |
| C `test_piecewise_expression_pattern_rejects_empty_cases_tuple`, `test_logical_expression_pattern_rejects_fewer_than_two_operands` (2), `test_alternatives_pattern_with_empty_alternatives_raises_value_error` | `..._with_empty_cases_matches_nothing`, `..._with_fewer_than_two_operands_matches_nothing` (2), `..._with_empty_alternatives_matches_nothing` | D-S5-5 |
| C the non-`Pattern` tests (4 parametrized cases, and the piecewise, call, logical and alternatives ones) | same names | D-S5-5: `TypeError` with the binding's message; the piecewise pair-length test too |
| R `test_rewrite_rule_defaults_for_optional_fields` | same name | D-S5-8: `guards == ()` |
| R `test_rewrite_rule_is_hashable_and_equal_by_content` | `test_rewrite_rule_compares_and_hashes_by_identity` | D-S5-8 |
| R `test_apply_rewrite_rules_wraps_guard_exception_as_pass_execution_error`, `..._rewrite_exception_...` | `..._as_rewrite_callback_error` | D-S5-11, with the cause |
| R the two applier `PassExecutionError` tests | same names | D-S5-12: they also pin the cause chain |
| R `test_rewrite_rule_applier_emits_diagnostic_when_named_rule_fires` | same name | D-S5-12: the core's text |
| R `test_apply_rewrite_rules_with_identity_rewrite_reports_unchanged` | same name | D-S5-10: extended with the next rule firing |
| R `test_rewrite_rule_accepts_arbitrary_pattern_subclasses` | `test_rewrite_rule_accepts_every_pattern_kind` (11 cases) | D-S5-4 |

The new `tests/symbolic/expression/pattern/test_pattern_rust_binding.py`
(94 tests) covers the test plan above: the class structure and
registration, the frozen errors, the argument checks and their messages,
the degenerate patterns, captures, bindings (including their falsiness
and repr, and their refusal to pickle), the node objects callbacks receive
and a rebuilt node's single object, truthiness and callback exceptions
(the same object, `KeyboardInterrupt` unwrapped, nested matches and
walks), `new_partial`, declining by returning the matched node, guards,
the walk (a shared subtree rewritten once, the input returned itself, a
20,000-level tree, a pattern deeper than the recursion limit), the error
classes and the rule a refused rebuild blames, Python rules, the applier,
pickles and reprs.

#### S5 follow-up: `MatchBindings` truthiness

S5.4 left an empty `MatchBindings` falsy, a side effect of D-S5-3 giving
it `len()`. The old dataclass had no `__len__`, so it was always truthy,
and `if pattern.match(expression):` read every hit as a hit. With the
falsy version, a capture-free pattern's hit read as a miss. Bindings are
now always truthy, as `re.Match` is. `len()` still counts captures, and
the interface test pins both.

## S6: pass infrastructure

- **Status:** designed 2026-09-25 at 64598a2. D-S6-1 to D-S6-20 apply
  the policy the user already set, and N-S6-1 to N-S6-3 are decided (see
  "S6 resolutions"). S6.1 to S6.7 are implemented (see "S6.1 to S6.3
  status" and "S6.4 to S6.7 status").
- **Pattern:** P3 for `CompilerPass`, `Analysis` and `Validator`; P2 for
  the managers, `PassResult`, `PreservedAnalyses` and the records, as the
  slice table says.

### Survey: the Python API

The package is `src/fhy_core/pass_infrastructure/`:

| File | Lines | Contents |
|---|--:|---|
| `__init__.py` | 61 | re-exports 25 names |
| `core.py` | 1,016 | `CompilerPass`, the registry, the errors, `PassResult`, `PreservedAnalyses`, `PassInfo`, `TraversalOrder`, `VisitablePass`, `AnalysisVisitablePass`, `RewritablePass`, `register_pass` |
| `manager.py` | 626 | `Analysis`, `AnalysisManager`, `PassManager`, `FixpointPassGroup`, the records |
| `validation.py` | 209 | `ValidationManager` |
| `verification.py` | 241 | `VerificationRegistry`, `VerificationAnalysis`, `register_verification`, `run_verification` |
| `traits/verifiable.py` | 159 | `Verifiable`, `VerifiableMixin`, `VerificationError` |

All of it is pure Python today. It runs over Rust-backed values only
where a pass's IR, or its diagnostics, happen to be Rust-backed.

**`CompilerPass(ABC, Generic[I, O])` (`core.py`).**

- **The lifecycle.** `execute(ir) -> PassResult[O]` resets the pass's
  diagnostic list and then runs:
  1. `validate_input(ir)`, then auto-verification of the input;
  2. `should_run(ir)`; when it is false, `get_noop_output(ir)` and
     `get_preserved_analyses(ir, output, changed=False)` end the run,
     unchanged;
  3. the run counters, then `run_pass(ir)`;
  4. `validate_output(ir, output)`, then auto-verification of the output;
  5. `did_change(ir, output)` and
     `get_preserved_analyses(ir, output, changed=...)`.

  `__call__(ir)` is `execute(ir).output`.
- **Hook defaults.** `run_pass` and `get_noop_output` are abstract.
  - `validate_input` rejects `None`: it reports an ERROR and raises
    `PassValidationError('Pass "X" does not accept None input.')`.
  - `should_run` is `True`, and `validate_output` does nothing.
  - `did_change` is `input != output`, falling back to `is not` when `!=`
    raises.
  - `get_preserved_analyses` is none when changed and all otherwise.
- **The guards.** Every hook runs inside a guard.
  - A validation hook's unexpected exception becomes `PassValidationError`
    after an ERROR diagnostic: `Pass "X" failed validate_input with
    ValueError: boom`.
  - Any other hook's becomes `PassExecutionError`: `Pass "X" failed
    should_run with ...`, and for `run_pass`, `Pass "X" failed with
    ValueError: boom`.
  - The original exception is the `__cause__` and is logged as `exc_info`.
  - **Pass-through.** A `PassValidationError` raised in a validation hook,
    a `PassExecutionError` raised in another hook, and either one raised in
    `run_pass` propagate as the same object, with no diagnostic added.
- **Names.** `get_pass_name()` and `get_pass_description()` are class
  methods: the registered name, or the class's `__name__`; the registered
  description, or the docstring, or the `__name__`.
- **The global registry.** It is a class-level dict behind a lock.
  - `@register_pass(name, description)` sets the class's name and
    description. It is idempotent for the same class and description, and
    raises `PassRegistrationError` for a blank name or description, a class
    that is not a `CompilerPass`, a name taken by another class, and a new
    description for the same class, each with its own message.
  - `CompilerPass.create(name, *args, **kwargs)` calls the registered class
    with the arguments, or raises `PassRegistrationError('Unknown pass
    "x".')`.
  - `get_registered_passes() -> Mapping[str, PassInfo]`, where `PassInfo`
    is the frozen dataclass `(name, description, pass_type)`.
- **Run counters.** `get_run_count()` per class name and
  `get_total_run_count()` are process-global. They count every run that
  reached `run_pass`, failed ones included, and never a skipped one.
- **Diagnostics.** `report(level, message: str | Note, detail=None, *,
  exc_info=None)` appends a `Diagnostic` whose source is the pass name, and
  logs it on `fhy_core.pass_infrastructure.core.<pass-name>` at the
  matching level, appending ` | detail: <detail>` when given.
  `diagnostics` returns the list of the current or most recent run, and
  `report` also works outside a run. `execute` logs DEBUG lines on entry
  (`entering (prospective run #N, input type=T)`), on a skip and on exit.
- **Analysis binding.** `bind_analysis_manager(manager)` (which refuses
  `None` with `TypeError`), `unbind_analysis_manager()` and
  `get_analysis_manager()` attach an `AnalysisManager`. With none bound,
  `get_analysis(analysis_type, ir)` computes `analysis_type().run(ir)`;
  otherwise it asks the manager.
- **Auto-verification.** The class variable `_auto_verify = True` makes
  both validation guards call `get_analysis(VerificationAnalysis, ir)`,
  standalone too. A report with errors reports an ERROR with the report's
  `format()` as its detail, and raises `PassValidationError('Pass "X"
  rejected input IR: verification reported N error(s).', report=report)`,
  or `produced invalid output IR` for the output.
- **The error classes.** `PassRegistrationError`, `PassValidationError`
  and `PassExecutionError` are `RuntimeError`s registered with
  `register_error`. `PassValidationError(message="", *, report=None)` has
  a `report` property.
- **`PassResult(output, changed, diagnostics=(), preserved_analyses=none)`**
  is a frozen, generic dataclass with `PartialEqualMixin`.
- **`PreservedAnalyses(preserve_all=False, analysis_names=frozenset())`**
  is a frozen dataclass keyed by `Identifier`.
  - Setting both fields raises `ValueError`.
  - `all()`, `none()`, `is_preserved(name)`.
  - `preserve(name)` returns the receiver itself when the name is already
    covered.
- **`VisitablePass`** is a `CompilerPass` whose `run_pass` is `visit`. It
  dispatches to `visit_<suffix>` by `get_visit_method_suffix()`, and
  `visit_unknown` raises `NotImplementedError`.
- **`AnalysisVisitablePass(traversal_order=PRE)`** is a
  `VisitablePass[N, None]`.
  - Its `walk` recurses over `get_visit_children()`, with
    `before_visit_<suffix>` and `after_visit_<suffix>` hooks around each
    node; the after-hook runs in a `finally`.
  - `visit_unknown` does nothing, `get_noop_output` is `None` and
    `did_change` is `False`. `TraversalOrder` is a `StrEnum`.
- **`RewritablePass(CompilerPass[N, N])`** has a recursive bottom-up
  `transform`.
  - It rebuilds a node through `rebuild_with_visit_children` when a child
    changed, then calls `visit_<suffix>` on the result. `None` keeps the
    node. `visit_unknown` returns `None`.
  - `did_change` is `is not`, and the input comes back itself when nothing
    changed.
  - A visitor that returns its input still counts as a change, and a
    shared subtree is visited once per occurrence.

**`manager.py`.**

- **`Analysis(ABC, Generic[IR, R])`** has the abstract `run(ir)`.
  - `get_analysis_name()` lazily creates one `Identifier` per class, named
    `<module>.<qualname>`.
  - `__init_subclass__` rejects an `__init__` with a required parameter
    (`TypeError`), because the manager builds analyses with no arguments.
- **`AnalysisManager()`** is a public, thread-safe cache (an `RLock`) that
  lives as long as its owner.
  - `get(analysis_type, ir)` caches only `Frozen` IR that `is_frozen`,
    keyed by `id(ir)`. It evicts a bucket through a `weakref.finalize`
    when the IR is collected, so it never pins the IR, and computes
    uncached when the finalizer cannot be registered.
  - `clear(ir)` drops the IR's bucket.
  - `invalidate(ir, preserved)` drops the analyses `preserved` does not
    keep.
  - `transfer(from_ir, to_ir, preserved)` moves the kept results from one
    bucket to the other, dropping `from_ir`'s bucket and `to_ir`'s own
    results; for the same IR it is `invalidate`.
  - It logs cache hits, misses and evictions at DEBUG.
- **`PassManager(name=None)`** (default name `pipeline`) is a
  `HasIdentifier`.
  - `add_pass`, `add_fixpoint_group` and `run(ir) -> PassManagerResult`.
  - `analysis_manager` is a property whose cache persists across runs.
  - Each pass runs through `execute` with the manager bound, then
    `analysis_manager.transfer(input, output, preserved)`.
  - A pass's exception propagates unchanged, with no records.
  - It logs INFO on start and finish, with the elapsed time, and DEBUG per
    item.
- **`FixpointPassGroup(name, *, max_iterations=10,
  fail_on_non_convergence=True)`**.
  - `max_iterations < 1` raises `ValueError('"max_iterations" must be >=
    1.')`.
  - `passes` is a tuple, and `add_pass` works after the group was added to
    a manager, since the manager holds the group object.
  - Non-convergence raises `PassExecutionError('Fixpoint group "g" did not
    converge in N iterations.')` after an ERROR log, with no record.
- **The records** are frozen dataclasses with `PartialEqualMixin`:
  - `PassRunRecord(pass_name, changed, diagnostics, preserved_analyses)`;
  - `FixpointIterationRecord(iteration, changed, pass_runs)`;
  - `FixpointGroupRecord(group_name, iteration_records, converged)`, with
    an `iterations` property;
  - `PassManagerResult(output, records)`.

**`validation.py`.** `ValidationManager(name=None)` (default name
`validation-pipeline`) has `add`, `validators` and `validate(ir) ->
ValidationReport[PassRunRecord]`.

- Each validator is a `CompilerPass` run through its whole `execute`,
  auto-verification included, with no analysis manager.
- A `PassValidationError` or `PassExecutionError` keeps the validator's
  diagnostics. When none of them is an ERROR, it adds `Validator "X"
  raised "T" without reporting a diagnostic: msg`.
- Any other exception adds `Validator "X" crashed with T: msg`.
- A record is `PassRunRecord(name, changed=False, diagnostics,
  PreservedAnalyses.all())`. It logs INFO counts.

**`verification.py`.**

- **`VerificationRegistry`** is a class-level registry keyed by IR type.
  `register(ir_type, pass_class)` is idempotent, and
  `get_passes_for(ir_type)` walks the reversed MRO, dropping duplicates.
- **`VerificationAnalysis`** is an `Analysis` whose result is the
  `ValidationReport` of a fresh `ValidationManager` over the registered
  passes. It is empty when none is registered.
- **`register_verification(ir_type, name, description)`** registers the
  class with both registries and sets `_auto_verify = False`, which stops
  the recursion.
- **`run_verification(ir)`** runs the analysis uncached.

**`traits/verifiable.py`.** `VerifiableMixin.__new__` refuses a class that
neither overrides `verify` nor has a registered verification pass,
walking the MRO; the default `verify` is `run_verification(self)`.
`Verifiable` is a runtime protocol, and `VerificationError` a plain
`Exception`.

### Survey: the Rust API

`fhy_core::pass` has `compiler_pass.rs` (671 lines with tests),
`context.rs` 108, `error.rs` 451, `analysis.rs` 555, `manager.rs` 622,
`validation.rs` 349, `registry.rs` 404, `preserved.rs` 214 and
`adapters.rs` 204. `fhy_core::tree` has `node.rs` 108, `walk.rs` 128,
`rewrite.rs` 281 and `hash.rs` 34, and `fhy_core::expression::passes` has
227. The design is B5 of `rust-workspace.md`: it removed the process-global
state (the registry and the counters, F-006), made the analysis cache
merge-only and dropped verification caching (F-016, R-6), and replaced
pass-through with `Nested` (F-014).

- **`CompilerPass<I, O = I>`** is an object-safe trait with `&mut self`
  hooks. Each hook receives the run's `PassContext` except `did_change`
  and `preserved_analyses`, which receive none.
  - `name() -> Cow<'static, str>`, by default
    `short_type_name::<Self>()`, and `description()`, by default the name.
  - `validate_input`, which accepts every input by default.
  - `skip(ir, cx) -> Result<Option<O>, _>`: `Some(output)` skips the run,
    and the default is `None`. It replaces `should_run` and the no-op
    output.
  - `run`, which is required.
  - `validate_output`, which does nothing by default.
  - `did_change`, which is required and has no default.
  - `preserved_analyses(input, output, changed)`.

  Blanket impls cover `&mut P` and `Box<P>`.
- **`ExecutePass::execute(&ir) -> Result<PassOutcome<O>, PassError>`**.
  `PassOutcome` has `output`, `into_output`, `is_changed`, `is_skipped`,
  `diagnostics` and `preserved_analyses`. A standalone run computes every
  analysis afresh and never verifies.
- **`PassContext`** is borrowed by the lifecycle for each hook call.
  - `report(Diagnostic)`.
  - `report_text(level, message, detail)`, attributed to the pass.
  - `analysis::<A>(ir) -> Arc<A::Output>` for `A: Analysis + Default`.
  - `diagnostics()` and `pass_name()`.
- **`PassError`** is one boxed pointer. `kind()` returns a
  `#[non_exhaustive]` view:
  - `Hook { pass_name, hook, source }`: a hook returned an error;
  - `Nested { pass_name, hook, inner }`: a hook returned a `PassError`,
    for example from running another pass;
  - `Verification { pass_name, point, report }`: the pipeline's verifier
    rejected the input or a changed output;
  - `NonConvergence { group_name, max_iterations }`.

  The other accessors:
  - `class()` is `Validation` or `Execution`. A hook has its hook's class,
    and `Nested` under `run` has the inner error's class.
  - `pass_name()`, `diagnostics()` (ending with the error diagnostic of the
    failure) and `records()` (the pipeline work before the failure).
  - `Display` is one line that never repeats the source: `pass "X" failed
    in run`, `verification rejected the output of pass "X" (errors: 2)`,
    `fixpoint group "g" did not converge (max iterations: 10)`.
  - The lifecycle's diagnostic appends the cause chain: `pass "X" failed in
    run: <chain>`.
  - `PassHook::as_str()` is `validate_input`, `skip`, `run`,
    `validate_output`, `did_change` or `preserved_analyses`.
- **`Analysis: 'static`** has an associated `Ir` and `Output: Send + Sync`,
  and `run(&self, &Ir) -> Output`; it cannot fail. The cache builds an
  analysis through `Default`.
- **`AnalysisId`** has the one constructor `of::<A>()`, a `TypeId` with
  its type name. `PreservedAnalyses` has `all`, `none`, `preserve::<A>`,
  `preserve_id`, `is_preserved`, `is_id_preserved`, `preserves_all` and
  `preserved_ids`.
- **The analysis cache** is `pub(super)` and lives for one `PassManager::run`.
  - It keys results by `(NodeIdentity, handle TypeId)` and by
    `AnalysisId`, and pins every cached node by holding a handle clone
    until the run ends.
  - `transfer` is merge-only: the output gains each preserved result it
    has none of its own for, the input keeps its results, and the same
    node changes nothing.
  - Nothing is removed during a run, and there is no public cache API.
- **`PassManager<'p, I: NodeHandle>`** owns boxed `Send` passes and
  groups.
  - `new(name)`, `add_pass`, `add_fixpoint_group`, `set_verifier` and
    `run(&ir)`.
  - With a verifier, the run verifies the input once, blaming the first
    pass, and every output a pass reports as changed, blaming that pass.
    This is uncached (R-6); the verifier's validators share the run's
    analysis cache.
  - `PassManagerResult` has `output`, `records` (`PipelineRecord::Pass`
    or `FixpointGroup`), `pass_runs()` and `run_count()`, which counts the
    runs that were not skipped.
  - `PassRunRecord` adds `is_skipped`. A failure inside a group ends the
    error's records with the group's partial record.
- **`FixpointPassGroup`** has `new(name)`, the builders
  `with_max_iterations(NonZeroUsize)` and
  `with_fail_on_non_convergence`, and `add_pass`. It moves into the
  pipeline.
- **`Validator<I>`** has `name()` and `validate(ir, cx) -> Result<(), _>`.
  Problems are diagnostics; an error means the check could not finish.
  - `PassValidator<P>` runs a `CompilerPass<I, ()>` as a check:
    `validate_input`, `skip`, `run`, then `validate_output`, with no
    `did_change` and no `preserved_analyses`.
  - `ValidationManager` runs every validator into one
    `ValidationReport<ValidatorRecord>`. A failed validator that reported
    no error gains `validator "X" failed without reporting an error:
    <chain>`.
  - `ValidatorRecord` has `validator_name`, `is_failed` and
    `diagnostics_in(&report)`. It holds a range into the report, so each
    diagnostic is stored once.
- **`PassRegistry`** is owned, not global.
  - `register::<P, I, O>(factory)` reads the name and description from one
    instance the factory builds.
  - A registration's identity is `(TypeId of P, I, O)`, with the variants
    `EmptyName`, `EmptyDescription`, `NameTaken` and
    `DescriptionConflict`.
  - `create::<I, O>(name)` takes no arguments and fails with
    `UnknownPass` or `IrTypeMismatch`. There are also `info`, `iter`, `len`
    and `is_empty`.
- **No run counters.** Run statistics come from each run:
  `PassOutcome::is_skipped`, `PassRunRecord::is_skipped` and
  `PassManagerResult::{pass_runs, run_count}`.
- **`WalkPass<V>` and `RewritePass<R>`** adapt `tree::TreeVisitor` and
  `tree::Rewriter`, which take the run's `PassContext` as their context.
  - `walk_tree` and `rewrite_tree` keep their own work stacks, so trees of
    any depth work.
  - A walk calls `before_visit`, `visit` and `after_visit`, and asks
    `walks_children`.
  - A rewrite handles each distinct node once, and a replacement that is
    the original node counts as no change (F-009).
  - `WalkPass` outputs `()`, and each pass is named after its visitor.
- **`NodeIdentity`** has the one constructor `of_arc`.
- **Rust-native passes.** `expression::passes::RewriteRuleApplier` (S5's
  applier; `NAME` and `DESCRIPTION` equal Python's) and
  `ExpressionPrettyFormatter::new(FormatOptions)` (`CompilerPass<Expression,
  String>`, every run a change). `register_expression_passes` registers the
  applier.

No pass binding exists yet. The S5 binding already drives Rust from Python
callbacks (the `Rule` adapter, `objects.rs`) and materializes Rust trees
into Python objects (`expression/materialize.rs`).

### Divergences visible from Python

| # | Python today | Rust core |
|---|---|---|
| W-1 | `should_run` plus an abstract `get_noop_output` | one `skip` hook; a pass that never skips needs no no-op output |
| W-2 | `validate_input` rejects `None` by default | accepts every input |
| W-3 | `did_change` defaults to `!=`, falling back to `is not` | required, with no default |
| W-4 | `report` and `get_analysis` work in every hook, and outside a run | only hooks given a `PassContext` report or read analyses; `did_change` and `preserved_analyses` get none |
| W-5 | A process-global registry of classes; `create(name, *args, **kwargs)`; `PassInfo(name, description, pass_type)` | an owned `PassRegistry` of factories; `create` takes no arguments; identity is `(pass type, I, O)`; Rust-style messages |
| W-6 | Process-global run counters | per-run `is_skipped`, `pass_runs()` and `run_count()` |
| W-7 | A `Pass*Error` raised in a hook of its class passes through as the same object, with the outer diagnostics dropped | a `PassError` from a hook is wrapped as `Nested`; the outer run's diagnostics are kept; any other error is a `Hook` failure |
| W-8 | Messages such as `Pass "X" failed with ValueError: boom`, which name the cause's class and text | `pass "X" failed in run`; the cause chain goes only into the diagnostic; the error carries its diagnostics and records |
| W-9 | Every pass run, standalone too, auto-verifies its input and its output, cached as an analysis; `_auto_verify` opts a class out | only a pipeline with a verifier verifies: its input once, and each changed output, blaming the producer; uncached (R-6) |
| W-10 | Verification passes are found per IR type in a global registry, walking the MRO | the verifier is an explicit `ValidationManager` |
| W-11 | A public, standalone `AnalysisManager` whose cache persists across runs; `clear`, `invalidate`, `transfer`; caches only frozen IR; weakref eviction, so it never pins | a private cache for one run; merge-only transfer; nothing removed; pins cached nodes until the run ends |
| W-12 | `transfer` moves results and drops the output's own ones; `invalidate` drops results of the same IR | merge-only; the output's own results win; the same node keeps everything |
| W-13 | Analyses are named by an `Identifier` from `<module>.<qualname>`; one class may analyse any IR | `AnalysisId` from a Rust type; one analysis type per `Ir` |
| W-14 | `bind_analysis_manager`, `unbind_analysis_manager` and `get_analysis_manager` on the pass | the context carries the cache for the length of the run |
| W-15 | `PassResult` and the records have no skipped flag | `is_skipped` on the outcome and on each record |
| W-16 | A pipeline's pass error propagates without records; non-convergence has no record | errors carry the records of the completed work, a group's partial record included |
| W-17 | `ValidationManager` runs whole `execute`s, auto-verification included; it synthesizes `crashed with` and `raised ... without reporting` diagnostics; its records are `PassRunRecord`s | the `Validator` trait; `PassValidator` runs the check part of the lifecycle; one synthesized text, only when no error was reported; `ValidatorRecord` records |
| W-18 | `FixpointPassGroup` is a mutable Python object the manager refers to | groups and passes move into the pipeline |
| W-19 | Logging: the lifecycle, every diagnostic, the pipelines and the cache | no logging |
| W-20 | `VisitablePass`, `AnalysisVisitablePass` and `RewritablePass` dispatch per node in Python, recursively; a rewrite returning its input counts as a change, and shared subtrees are rewritten per occurrence | `WalkPass` and `RewritePass` over `TreeVisitor`/`Rewriter`: iterative, once per distinct node, and returning the input is no change |
| W-21 | `ExpressionPrettyFormatter` is a `VisitablePass` whose `visit_*` methods a subclass may override | a Rust pass with no per-node hook |
| W-22 | Arguments are duck-typed | typed |

Unchanged in meaning: the order of the lifecycle; the error classes
(validation and execution); the preservation rule (none when changed, all
otherwise); a pipeline feeding each pass the previous output; fixpoint
convergence at the first iteration that changes nothing, within a budget;
collect-all validation; and the pass and group names.

### Consumers and tests

**`src`.** Eleven pass classes use the package; ten are registered.

| Class | Base | Module | Hooks it defines |
|---|---|---|---|
| `ExpressionPrettyFormatter` | `VisitablePass` | `symbolic/expression/pprint.py` (156 lines) | `get_noop_output` (raises), `__call__` |
| `NumpyExpressionEvaluator` | `VisitablePass` | `passes/numpy.py` (728) | `get_noop_output` (raises), `did_change` |
| `ExpressionToZ3Converter` | `VisitablePass` | `passes/z3.py` (727) | `get_noop_output` (raises) |
| `ExpressionToSympyConverter` | `VisitablePass` | `passes/sympy.py` (1,738) | `get_noop_output` (raises) |
| `SympyVariableSubstitutionPass`, `SymPyToExpressionConverter` | `CompilerPass` | `passes/sympy.py` | `get_noop_output` (raises), `run_pass` |
| `ExpressionTypeChecker` | `VisitablePass` | `types/checking/type_checker.py` (1,513) | `get_noop_output` (raises); `synthesize` and `check` call `visit` directly, outside `execute` |
| `RegisteredFunctionBodyTypeChecker` | `CompilerPass[Expression, None]` | `types/checking/body_type_checker.py` (372) | `run_pass`, `get_noop_output`, `did_change` |
| `FunctionInliner` | `RewritablePass` | `passes/inline.py` (169) | per-node visitors |
| `ExpressionEvaluator` | `RewritablePass` | `passes/evaluate.py` (186) | per-node visitors; calls `report` |
| `RewriteRuleApplier` | `CompilerPass` | `symbolic/expression/pattern/rewrite.py` (296) | `run_pass`, `get_noop_output`, `did_change`; calls `report` |

- **How they are used.** Every entry point calls a pass through
  `__call__`, except the type checker's `visit`. None overrides
  `validate_input`, `validate_output`, `should_run` or
  `get_preserved_analyses`. None calls `get_analysis`, the counters,
  `create` or the analysis binding.
- **Other users.** `Lattice` (`lattice.py`, 206 lines) and `SymbolTable`
  (`symbol_table.py`, 758) are `VerifiableMixin`s that override `verify`
  and build their `ValidationReport` in Python. No module outside the
  package uses `PassManager`, `FixpointPassGroup`, `AnalysisManager` or
  `ValidationManager`.

**Python tests.** There are 222 tests in `tests/pass_infrastructure/`, 5,239
lines; one of them is deselected by default, a `very_slow` one:

| File | Lines | Tests |
|---|--:|--:|
| `test_core.py` | 817 | 34 |
| `test_manager.py` | 1,169 | 43 |
| `test_manager_properties.py` | 159 | 2 |
| `test_validation.py` | 547 | 24 |
| `test_verification.py` | 1,439 | 58 |
| `test_visitable.py` | 396 | 21 |
| `test_rewritable_pass.py` | 712 | 39 |

Consumer tests pin the errors. The consumers' test modules hold 68
`pytest.raises(PassExecutionError | PassValidationError)` checks. 27 of
them pass `match=`, and 22 of those match the cause's class or text in
the message; they are in `passes/test_evaluator.py` (7),
`test_inline_pass.py` (7), `test_sympy_pass.py` (4),
`test_sympy_natives.py` (2), `test_z3_pass.py` (1) and
`test_numpy_evaluator.py` (1). The other 5 call `get_noop_output`
directly. Many of the checks also read `__cause__`, which keeps its
meaning. `test_error.py`
checks the error registration, and `test_core_traits.py` and
`test_basic_traits.py` check `VerifiableMixin` (7 references).

**Rust tests**, which already specify the core. Counts are test functions;
a property file counts its `proptest!` blocks:

| File | Lines | Tests |
|---|--:|--:|
| `tests/it/pass/core_stories.rs` | 1,606 | 56 |
| `tests/it/pass/manager_stories.rs` | 1,471 | 48 |
| `tests/it/pass/validation_stories.rs` | 860 | 25 |
| `tests/it/pass/manager_properties.rs` | 279 | 4 |
| `tests/it/tree/stories.rs` | 1,539 | 61 |
| `tests/it/tree/properties.rs` | 213 | 6 |
| `tests/it/expression/pass_stories.rs` | 630 | 25 |

**Benchmarks.** `benchmarks/test_pass_infrastructure.py` has three
benchmarks, at 11.4 µs, 355 µs and 8.7 µs after S4.3b:
`test_compiler_pass_execute`, `test_pass_manager_run_of_5_passes` and
`test_analysis_manager_cache_hit`. The fixtures are in
`benchmarks/conftest.py`: a frozen `Box` IR, `BoxValueAnalysis`, and the
identity, increment and read-analysis passes.

### Pattern choice

- **P3: `CompilerPass`, `Analysis` and `Validator`.**
  - The public classes are `class CompilerPass(_rs.CompilerPassBase, ABC,
    Generic[I, O])`, `class Analysis(_rs.AnalysisBase, ABC, Generic[IR,
    R])` and `class Validator(_rs.ValidatorBase, ABC, Generic[IR])`. Each
    base's `#[new]` accepts `*args, **kwargs`, so subclasses with their own
    `__init__` construct; the probe verified abstract methods on this
    layering.
  - A Python subclass is driven from Rust through an adapter holding
    `Py<PyAny>`, which implements `CompilerPass<PyIr>`, `Analysis` or
    `Validator<PyIr>` by calling the Python hooks under the interpreter.
    A raised exception becomes the hook's `PassFailure`, a boxed `PyErr`.
  - The Rust-native passes, `RewriteRuleApplier` and
    `ExpressionPrettyFormatter`, are `#[pyclass(extends =
    CompilerPassBase)]` classes whose base holds the native pass, so a
    pipeline runs them without calling Python. They sit over an adapter
    that extracts the `Expression` from the `PyIr` and rewraps the output
    through S4.3a's materializer, which returns the input object when
    nothing changed.
- **P2: the rest of the machinery.** `PassManager`, `FixpointPassGroup`,
  `ValidationManager`, `AnalysisManager`, `PassResult`,
  `PreservedAnalyses`, `PassRunRecord`, `FixpointIterationRecord`,
  `FixpointGroupRecord`, `PassManagerResult` and the new `ValidatorRecord`
  are pyclasses. They are logic-rich and hold passes that a Rust pipeline
  runs (decision 2).
- **P1 or plain Python.** `TraversalOrder` stays a `StrEnum`. `PassInfo`,
  the verification registry and `VerifiableMixin` stay Python; the pass
  registry depends on N-S6-1. `VisitablePass`, `AnalysisVisitablePass` and
  `RewritablePass` stay Python subclasses of the new `CompilerPass`,
  pending N-S6-3.
- **One type-erased IR.** The binding's pipelines are `PassManager<'_,
  PyIr>`, where `PyIr(Py<PyAny>)` implements `NodeHandle`. It clones
  through `clone_ref` under `Python::attach`, so the crate needs no
  `py-clone` feature, and its identity is the Python object's pointer. The
  cache pins a node by holding a `PyIr` clone, which is a strong
  reference, so no other object can take its address during a run.
- **Borrowed Rust state never reaches Python.** Python hooks keep their
  signatures and take no context argument. While a Python hook runs, the
  adapter binds an owned context object to the pass instance, which
  `report`, `get_analysis` and `get_analysis_manager` use.
  - It collects the reported diagnostics, and forwards analysis requests
    to the run's cache through the detached handle of S6.2.
  - When the hook returns, its diagnostics move into the Rust
    `PassContext` and the handle goes back to the cache. The object is then
    invalidated, so a retained reference raises instead of reaching stale
    state.
- **Granularity.** Python is called once per pass hook. A Rust-native pass
  in a mixed pipeline calls no Python at all, apart from the Python
  callbacks of its rewrite rules (S5).

**Benchmark plan (S6.1).** The existing three benchmarks stay, and
`benchmarks/test_pass_infrastructure.py` gains the rows below, with
fixtures in `benchmarks/conftest.py`. As in S4.1 and S5.1, every call
whose spelling S6 changes sits in a helper marked with its decision:

- `_warm_cache(...)`: D-S6-8 removes the standalone `AnalysisManager`, so
  the cache-hit row times a second `get_analysis` inside one pass run;
- `_run_count(result)`: N-S6-1.

The baseline measures today's Python classes.

| Benchmark | Measures |
|---|---|
| `test_compiler_pass_execute` (exists) | an identity pass with only the abstract hooks: the lifecycle's floor |
| `test_compiler_pass_call` | `__call__` of the same pass |
| `test_compiler_pass_execute_with_every_hook_overridden` | the worst case: all seven hooks are Python |
| `test_compiler_pass_execute_skipped` | `should_run` false, then `get_noop_output` |
| `test_compiler_pass_execute_failing` | `run_pass` raises: the wrapping, the diagnostic, logging with `exc_info` |
| `test_compiler_pass_report_of_100_diagnostics` | a pass reporting 100 diagnostics: per-diagnostic cost and the object table |
| `test_compiler_pass_create` | `CompilerPass.create(name)` |
| `test_pass_manager_run_of_5_passes` (exists) | the small mixed pipeline |
| `test_pass_manager_run_of_50_passes` | the per-pass cost of a pipeline |
| `test_pass_manager_fixpoint_group_of_10_iterations` | a group converging on its tenth iteration |
| `test_pass_manager_run_with_verification` | 5 passes over an IR type with one registered verification pass |
| `test_analysis_manager_cache_hit` (exists) | a cached analysis read in a run, through `_warm_cache` |
| `test_analysis_preserved_across_5_passes` | one analysis computed once and carried through 5 preserving passes |
| `test_validation_manager_validate_of_10_validators` | collect-all over ten validators, some reporting |
| `test_run_verification` | `run_verification` of an IR with two registered passes |
| `test_mixed_pipeline_over_a_deep_expression` | `RewriteRuleApplier` (S5's four rules), the formatter and a Python pass over S4.1's deep tree |

These are rerun, not added: `test_rewrite_rule_applier_execute_of_a_deep_tree`
(`test_pattern.py`), `test_visitable_pass_walk_of_deep_tree` and the
formatter rows (`test_expression.py`), and `test_validation_report_*`
(`test_diagnostic.py`, for D-S6-17). A slower hot path changes pattern, or
is recorded as an accepted cost with numbers (cross-cutting rule 5). The
paths at risk:

- the floor, if the adapter called Python for hooks the class does not
  override (D-S6-3 avoids it);
- `report`, which clones each Rust diagnostic once into the context;
- the pipeline, which builds its Rust items on every run (D-S6-11).

### Decisions (proposed 2026-09-25)

Each names the policy it follows:

- D-S4-1: Rust semantics where the two differ;
- D-S4-2: Python names where the meaning is the same;
- the S6 rule: the `CompilerPass` hook names stay Python;
- the subclass rule: existing Python pass subclasses in `src` and `tests`
  keep working unchanged as far as possible;
- "tests rewritten, not skipped";
- "no fallback".

- **D-S6-1: one implementation, no fallback** ("no fallback"). `core.py`,
  `manager.py`, `validation.py` and `verification.py` become public
  classes over `_rs`, and the Python lifecycle, guards, cache and
  pipelines are deleted, not kept behind a switch.
- **D-S6-2: the Python hook names drive the Rust lifecycle** (the S6 rule;
  W-1). The adapter maps the hooks one to one:

  | Python hook | Rust |
  |---|---|
  | `validate_input(ir)` | `validate_input` |
  | `should_run(ir)`, then `get_noop_output(ir)` when it is false | `skip`: `Some(get_noop_output(ir))` or `None` |
  | `run_pass(ir)` | `run` |
  | `validate_output(input_ir, output)` | `validate_output` |
  | `did_change(input_ir, output)` | `did_change` |
  | `get_preserved_analyses(input_ir, output, *, changed)` | `preserved_analyses` |
  | `get_pass_name()`, `get_pass_description()` (class methods) | `name`, `description`, read once per run |

  `execute`, `__call__`, `report`, `diagnostics`, `get_analysis` and
  `get_analysis_manager` keep their names (D-S4-2). `run_pass` stays
  abstract. `get_noop_output` is no longer abstract (D-S4-1: only a
  skipping pass needs one). Its default raises `NotImplementedError`
  naming the pass, so a class whose `should_run` returns false without one
  fails in that hook. Every subclass that defines it works unchanged.
- **D-S6-3: the defaults are Rust's, except where Rust has none** (D-S4-1;
  W-2, W-3).
  - `validate_input` accepts every input, `None` included.
  - `did_change` keeps Python's `!=` with the `is not` fallback, since the
    Rust trait has no default to follow.
  - `should_run` and `get_preserved_analyses` already agree.
  - The adapter resolves once per class which hooks the class overrides,
    and runs the defaults of the others in Rust without calling Python.
    Only the floor benchmark can tell.
- **D-S6-4: diagnostics** (D-S4-2 for the API; D-S4-1 for which hooks
  report; W-4, W-14).
  - `report(level, message, detail=None, *, exc_info=None)` keeps its
    signature. During a hook it records into that hook's owned context.
  - In `did_change` and `get_preserved_analyses`, which have no context
    in Rust, `report` and `get_analysis` raise `RuntimeError` naming the
    hook, which fails the hook. No consumer reports there.
  - Outside a run, as in a direct `visit` or `transform` call, `report`
    records on the pass as before, and the next `execute` clears it.
  - `diagnostics` is the current run's list during a hook, and afterwards
    the last run's, failed runs included.
  - A diagnostic Python reported comes back as the same object in the
    result, the records and the error; every other one is built once
    (D-S6-18).
- **D-S6-5: logging is kept** (D-S4-2; W-19). The core has no logging, so
  the binding logs on the same loggers and levels:
  - each diagnostic, when it is reported or, for one the core produced,
    when it reaches Python, with `exc_info` for a failed hook's exception;
  - the lifecycle's DEBUG lines, whose entering line drops the run number,
    `entering (input type=T)`;
  - the pipelines' INFO and DEBUG lines, and the validation counts.

  The analysis cache's hit and miss lines go, since the cache is the
  core's.
- **D-S6-6: the errors are the core's, under the Python classes**
  (D-S4-1, D-S4-2; W-7, W-8, W-16).
  - The class follows `PassError::class()`: `PassValidationError` or
    `PassExecutionError`, which stay `RuntimeError`s registered as now.
    Python can still raise both, and `PassValidationError(message, *,
    report=None)` keeps its constructor.
  - The message is the core's one line, verbatim: `pass "X" failed in
    run`, `verification rejected the output of pass "X" (errors: 2)`,
    `fixpoint group "g" did not converge (max iterations: 10)`. The
    failure's diagnostic is the core's too, `pass "X" failed in run:
    ValueError: boom`, with `PyErr`'s own rendering of the cause.
  - So the texts name the core's hooks (`skip` for `should_run` and
    `get_noop_output`, `run` for `run_pass`, `preserved_analyses` for
    `get_preserved_analyses`). The core writes these texts in the
    lifecycle, in `PassValidator` reports and in the errors, and
    re-rendering them in the binding would duplicate its text rules. The
    method names, which the S6 rule covers, stay Python, and the new
    `hook` attribute names the Python hook that failed.
  - `__cause__` is the hook's exception itself, or the inner error's
    Python exception for `Nested` (N-S6-2).
  - New attributes: `pass_name`, `hook` (the Python hook's name, such as
    `"should_run"`, or `None`), `diagnostics` and `records`. `report` is the verifier's report for a
    verification failure.
  - A non-convergence and a pipeline failure carry the records of the
    completed work.
  - The 22 consumer tests that match the cause in the message are
    rewritten to assert on `__cause__`.
- **D-S6-7: `Analysis` is a P3 ABC** (D-S4-2; W-13). `run(ir)`,
  `get_analysis_name() -> Identifier` and the no-argument constructor
  check stay. The check is the core's `Default` requirement under its
  Python spelling. A Python analysis's id is built from that `Identifier`
  (S6.2), so `PreservedAnalyses.preserve(A.get_analysis_name())` keeps
  working. Results are cached as Python objects, and one class may still
  analyse any Python IR, since all Python IR is one `PyIr` type.
- **D-S6-8: the analysis cache is the core's, for one run** (D-S4-1;
  W-11, W-12, W-14; S5's removal of Python-only API such as `try_bind`).
  - A run caches per node and pins cached nodes until it ends.
  - Transfer is merge-only, a node never loses its own results, and
    nothing is removed.
  - As today, the binding caches only IR that is `Frozen` and
    `is_frozen`: that is what `NodeHandle` requires (a node cannot change
    while a handle is alive), and other IR is computed afresh.
  - A standalone `execute` computes afresh, as an unbound pass does today.
  - `AnalysisManager` stays as the name of the run's cache view that
    `get_analysis_manager()` returns during a hook. It has only `get`, and
    raises after its hook.
  - Removed: constructing an `AnalysisManager`, `clear`, `invalidate`,
    `transfer`, `bind_analysis_manager`, `unbind_analysis_manager`, and
    `PassManager.analysis_manager`.
- **D-S6-9: `PreservedAnalyses` is P2 with its Python API** (D-S4-2).
  - `preserve_all`, `analysis_names`, `all()`, `none()`, `preserve(name)`
    (the receiver itself when already covered) and `is_preserved(name)`
    keep their meaning, keyed by `Identifier`.
  - It holds the core's set of ids, and the `ValueError` for both fields
    stays.
  - A set that came from a Rust-native pass names no Python analysis, so
    its `analysis_names` is empty unless it preserves all.
- **D-S6-10: `PassResult` and the records are P2, with the core's
  additions** (D-S4-1; W-6, W-15).
  - The fields and constructors stay (D-S4-2).
  - `PassResult`, the three records and `PassManagerResult` compare, hash
    and print as the dataclasses do (S3 practice), keep
    `PartialEqualMixin`, and pickle as a call of their class with their
    fields.
  - New: `PassResult.skipped` and `PassRunRecord.skipped`, keyword
    arguments that default to `False`, and `PassManagerResult.pass_runs()`
    and `run_count()`.
  - `output` is the Python object itself.
- **D-S6-11: `PassManager` and `FixpointPassGroup` are P2 pyclasses over
  Python item lists** (the subclass rule; W-18).
  - They keep today's API minus `analysis_manager` (D-S6-8). The name
    defaults stay, and so does the `max_iterations` `ValueError`
    (D-S4-2).
  - `run` builds the Rust `PassManager<'_, PyIr>` from the current items
    and runs it, so a group or pass changed after it was added behaves as
    today. Each run gets its own pipeline and cache, so concurrent runs of
    one manager are independent.
  - The same pass object added twice runs twice.
- **D-S6-12: verification is the pipeline's** (D-S4-1, R-6; W-9, W-10;
  the subclass rule for the default).
  - A standalone `execute` never verifies.
  - A `PassManager` verifies its input once and every changed output,
    blaming the producer, uncached.
  - Its default verifier is the registry verifier. That is a `Validator`
    adapter which runs, as a `ValidationManager` would, the verification
    passes `VerificationRegistry.get_passes_for(type(ir))` returns. So a
    pipeline over a type with registered passes still verifies without
    setup.
  - `set_verifier(verifier)` is new (the Rust name, since Python has no
    counterpart, as in D-S5-9). It takes a `ValidationManager`, or `None`
    to verify nothing.
  - `_auto_verify` goes: verification passes run as validators, which
    never verify, so nothing can recurse.
  - `register_verification`, `VerificationRegistry`,
    `VerificationAnalysis`, `run_verification` and `VerifiableMixin` stay
    Python, with their semantics. The registry is a Python registry of
    Python classes, as the pass registry is.
- **D-S6-13: `Validator` is a new P3 ABC, and `ValidationManager` is P2
  with the core's semantics** (D-S4-1, W-17; D-S5-9 for the new names).
  - `Validator` declares the abstract `validate(ir)` and a `name` property
    that defaults to the class's `__name__`, and has `report` and
    `get_analysis` as a pass does.
  - `ValidationManager` keeps `name`, `add`, `validators` and `validate`.
    `add` takes a `Validator`, or a `CompilerPass`, which runs as the
    core's `PassValidator` does: `validate_input`, `should_run` and
    `get_noop_output`, `run_pass`, then `validate_output`.
  - `validate` returns a `ValidationReport` of `ValidatorRecord`s. A
    record has `validator_name`, `failed`, and `diagnostics`: the record's
    slice of the report's diagnostic objects, shared, not copied (D-S4-2).
  - The two synthesized Python texts are replaced by the core's
    `validator "X" failed without reporting an error: <chain>`. A failing
    pass reports its hook's failure diagnostic, so it never needs the
    synthesized one.
- **D-S6-14: Rust-native passes are `extends` classes** (P3; W-21).
  - `RewriteRuleApplier` becomes the native applier over S5's rules,
    Python `Rule`s included, with S5's API (`rules`, `fired`, the
    diagnostic text) and its registration.
  - `ExpressionPrettyFormatter(is_id_shown=False,
    is_printed_functional=False)` becomes the native formatter, pending
    N-S6-3's formatter question.
  - A Python subclass that overrides a hook is driven through the Python
    adapter, so the override is honored; otherwise the native pass runs.
    Which applies is resolved once per class.
- **D-S6-15: arguments are typed** (S2 to S5's "stricter arguments";
  W-22). Passes, groups, validators, analysis classes, `PreservedAnalyses`
  and hook results are checked, raising `TypeError` in S2's style
  (`PassManager.add_pass pass must be a CompilerPass, got int.`). A hook
  result of the wrong type (`should_run` not a `bool`,
  `get_preserved_analyses` not a `PreservedAnalyses`) fails the hook.
  `did_change` and `should_run` results are read by truthiness, as S5's
  callbacks are (D-S5-6).
- **D-S6-16: pickles** (S3 to S5 practice). The P2 values pickle as a call
  of their class with their fields. A pass instance pickles through its
  `__dict__`, as before, since the base keeps no state outside a run.
  Managers and the cache view do not pickle.
- **D-S6-17: the `ValidationReport` cost (S3a leftover): one
  representation, the tuples** (recommendation). The S3a pyclass keeps the
  diagnostic and record tuples and drops the `ValidationReport<Py<PyAny>>`
  beside them.
  - `errors()`, `warnings()`, `infos()`, `has_errors()`, `format()`, `==`
    and `hash` run in Rust over the `&Diagnostic`s borrowed from each
    frozen `Diagnostic` pyclass, with no clone.
  - Construction becomes a type check per item and the tuple, which should
    beat the dataclass's 1 µs where it now takes 14.5 µs, and the other
    operations keep their S3a speed.
  - This is S3a's option 3 without its second representation: nothing
    needs a Rust `ValidationReport` of Python records, because the reports
    the Rust side produces are converted once when they reach Python
    (D-S6-18). `raise_if_failed` and `ValidationFailedError` are
    unchanged.
  - The S3a benchmarks confirm it, or the cost is recorded.
- **D-S6-18: Rust diagnostics and reports reaching Python** (S3a's
  registration; recommendation). This is the S3a leftover, and S3a's
  classes already register their public classes for it.
  - The binding gains builders from a Rust `Diagnostic` to the public
    `Diagnostic`: the `Note`, with its kind from S2's identity cache, the
    `DiagnosticLevel` member from S3a's table, and the `str` fields.
  - A Rust `ValidationReport<ValidatorRecord>` becomes the public
    `ValidationReport` of `ValidatorRecord`s.
  - Every conversion happens once, where a value leaves Rust: the
    outcome's diagnostics, each record's, a `PassError`'s diagnostics,
    records and report, and a validation report. Each Rust diagnostic is
    cloned once, since the core exposes them only by reference.
  - A per-run diagnostic table maps each position a Python `report` filled
    to its object, so those come back as themselves, and one diagnostic
    shared by a record and the result is one object.
- **D-S6-19: `VisitablePass`, `AnalysisVisitablePass` and
  `RewritablePass` stay Python classes over the new `CompilerPass`**,
  pending N-S6-3. Their walks, dispatch and semantics are unchanged.
- **D-S6-20: tests are rewritten, not skipped** (the tests rule). The 222
  pass-infrastructure tests and the consumer tests are rewritten where a
  decision changes what they pin, and each rename or rewrite is recorded
  with its reason, as S4.4 and S5 did. New Rust tests come only with the
  core additions (S6.2).

### Needs the user

- **N-S6-1: the global registry and the run counters** (W-5, W-6). The
  core replaced both with owned or per-run state. The Python API has
  `register_pass`, `CompilerPass.create(name, *args, **kwargs)`,
  `get_registered_passes()`, `PassInfo`, `get_run_count()` and
  `get_total_run_count()`. The core's `PassRegistry` cannot hold Python
  classes faithfully: every Python pass is one adapter type, so its
  identity check could not tell two classes apart, and `create` takes no
  arguments.
  - (a) **Keep the registry as a Python registry, and remove the
    counters.** The registry keeps its API and messages, and the native
    passes register in it. The counters give way to
    `PassResult.skipped`, `PassRunRecord.skipped` and
    `PassManagerResult.run_count()`. Two tests of the counters, and
    the counter assertions of two more, are rewritten, all in
    `test_core.py`.
  - (b) Keep both. The binding keeps process-global counters, restoring
    the state B5 removed.
  - (c) Bind the core's `PassRegistry` as an owned `PassRegistry` class of
    factories, with a default instance that `register_pass` and `create`
    use, and remove the counters.

  Recommendation: (a). The registry is a registry of Python classes, as
  `VerificationRegistry` is. Only one registry per concept is live, since
  the binding never builds a core `PassRegistry`. The counters are global
  state the core removed on purpose (F-006), and nothing in `src` reads
  them.
- **N-S6-2: pass-through against `Nested`** (W-7). In Python, a
  `PassValidationError` raised in a validation hook, a `PassExecutionError`
  raised in another hook, and either one raised in `run_pass` propagate
  as the same object, without the outer diagnostics. In the core, a
  `PassError` a hook returns is wrapped as `Nested`, which keeps the outer
  run's diagnostics and takes the inner class only under `run`; every
  other error is a `Hook` failure.
  - (a) **Rust semantics.** An error of a nested `execute` or pipeline run
    is `Nested`: the Python exception keeps its Rust `PassError`, which
    the adapter hands back when it propagates through a hook. The outer
    exception names the outer pass and hook, and its `__cause__` is the
    inner one. A `Pass*Error` the hook's own code raises is a `Hook`
    failure like any exception, with the same wrapping. This rewrites the
    2 pass-through tests; the 7 consumer `get_noop_output`s that raise
    `PassExecutionError` keep working, wrapped.
  - (b) **Python semantics.** Every `Pass*Error` of the hook's class
    propagates as itself. The binding would unwrap `Nested` and drop the
    outer diagnostics, against the core.
  - (c) A mix: nested runs are `Nested`, and a `Pass*Error` raised by the
    hook's own Python code passes through.

  Recommendation: (a). It follows D-S4-1, keeps every diagnostic, and
  makes one rule for Python and Rust passes. `except PassExecutionError`
  still catches everything that caught it before.
- **N-S6-3: per-node Python visitors** (W-20, W-21). P3's granularity rule
  keeps Python out of per-node callbacks from Rust. The core's `WalkPass`
  and `RewritePass` are per-node traits.
  - (a) **The three classes stay Python** (D-S6-19). `VisitablePass`,
    `AnalysisVisitablePass` and `RewritablePass` subclass the new
    `CompilerPass`, and their walks run in Python inside `run_pass`. They
    keep today's semantics: recursion depth, per-occurrence rewriting, and
    a visitor returning its input counting as a change.
  - (b) Drive `RewritablePass` and `AnalysisVisitablePass` from the core's
    `rewrite_tree` and `walk_tree`, with per-node Python callbacks, so
    they take the Rust semantics: any depth, once per distinct node, and
    F-009. This needs a `Tree` implementation for Python nodes and calls
    Python several times per node.
  - (c) (a) now, and (b) when the IR they walk is ported.

  Also: whether `ExpressionPrettyFormatter` becomes the Rust-native pass,
  as S6 plans. That drops the per-node override its docstring advertises;
  one test (`test_pprint.py::_BracketedLiterals`) relies on it, and one
  calls its `get_noop_output`. The class is not in `pprint`'s `__all__`,
  and `pformat_expression` already renders through the core.

  Recommendation: (a) for the three classes, since six `src` passes
  besides the formatter, and the type checker's direct `visit`, depend on
  their dispatch, and (b) would cost Python calls per node. For the
  formatter: make it native, refuse at class creation any subclass that
  defines a `visit_*` method (a `TypeError` naming this decision), so no
  override is ignored silently, and rewrite the two tests.

### Steps

1. **S6.1: benchmarks.** Extend `benchmarks/test_pass_infrastructure.py`
   as planned above, and record the baseline here, on today's Python
   classes.
2. **S6.2: core additions, with Rust tests.** The binding cannot build
   these from the public API:
   - `NodeIdentity::of_ptr<T: ?Sized>(pointer: *const T)`: an identity from
     an address, for handles to foreign objects such as `PyIr`. It is
     documented under `NodeHandle`'s contract (the holder keeps the
     pointee alive), and needs no `unsafe`.
   - Analysis ids for analyses that no Rust type names:
     `AnalysisId::of_identifier(&Identifier)`. It stays `Copy`, is equal
     exactly for one identifier, never equals an `of::<A>()` id, and
     orders and displays by the name hint. It comes with
     `PassContext::analysis_by_id(ir, id, compute) -> Arc<V>`, which
     caches under the id as `analysis` does and computes afresh outside a
     pipeline.
   - An owned handle to the run's cache for the length of a callback:
     `PassContext::with_detached_analyses(|handle| ...)`. It moves the
     cache out of the context into a `'static`, `Send` handle that can
     serve `analysis_by_id`, and moves it back when the closure returns,
     even on a panic. A clone kept past that point finds nothing and
     reports it.

   These need Rust tests in `tests/it/pass/` (the cache semantics by id,
   merge-only transfer of preserved dynamic ids, a detached handle after
   its callback, an identity from a pointer) and an update of the crate's
   README. If S6.4 shows the handle is not needed, it is dropped, and the
   Python context forwards nothing across a hook.
3. **S6.3: the `ValidationReport` representation** (D-S6-17), with its
   S3a benchmarks before and after. It is independent of the passes, so
   it lands first.
4. **S6.4: the binding.** Add `rust/fhy-core-py/src/pass.rs` with these
   submodules:
   - `ir.rs`: `PyIr`;
   - `context.rs`: the owned hook context and the `AnalysisManager` view;
   - `compiler_pass.rs`: `CompilerPassBase`, the Python adapter with its
     per-class hook table, and the native holder;
   - `analysis.rs`: `AnalysisBase`, the ids, `PreservedAnalyses`;
   - `validation.rs`: `ValidatorBase`, `ValidationManager`,
     `ValidatorRecord`, and the registry verifier;
   - `manager.rs`: `PassManager`, `FixpointPassGroup`, the records and the
     results;
   - `error.rs`: `PassError` to the Python classes, and the Rust error an
     exception carries for N-S6-2;
   - `convert.rs`: D-S6-18's builders and the diagnostic table;
   - `native.rs`: the Expression adapter, `RewriteRuleApplier` and
     `ExpressionPrettyFormatter`.

   Everything new goes into `_rs.pyi`, and CONTRIBUTING's "Process-global
   state" section records any new global state.
5. **S6.5: the Python switch.** `core.py`, `manager.py`, `validation.py`
   and `verification.py` define the public classes over `_rs` per D-S6-1
   to D-S6-19, and export `Validator` and `ValidatorRecord`. `pprint.py`
   and `pattern/rewrite.py` define the native passes. The consumers in
   `src` change only if a decision forces it: none is expected, since they
   use only `__call__`, `visit`, `report` and the kept hooks. The README's
   feature row changes.
6. **S6.6: tests.** Migrate the tests and add the interface suite (the
   test plan below).
7. **S6.7: benchmarks after,** recorded here with the verdict, then the
   status, the implementation notes and this checklist.

Commit per step. Every step ends with `pytest` and `-m "not very_slow"`
green, the `property` session, `lint` and `type_check` clean, and the Rust
gate green (fmt, clippy `-D warnings`, tests, doc `-D warnings`, deny,
`cargo +1.85 check`).

### Test plan

**The interface suite,
`tests/pass_infrastructure/test_pass_infrastructure_rust_binding.py`**,
covers what the binding adds over the core:

- **Class structure.**
  - Each public class extends its `_rs` class, and abstract methods are
    enforced (`run_pass`, `Analysis.run`, `Validator.validate`).
  - A subclass with its own `__init__`, with or without
    `super().__init__()`, constructs.
  - The native passes extend `CompilerPassBase`, `isinstance(...,
    CompilerPass)` holds, and they register.
  - The stubs are covered by `tests/test_rs_stub.py`.
- **The hook mapping.** A recording pass sees each hook in the lifecycle's
  order, and `should_run` false calls `get_noop_output` and no `run_pass`.
  Hooks a class does not override run in Rust, and the defaults of
  D-S6-3 hold. A class with a false `should_run` and no `get_noop_output`
  fails in `get_noop_output`.
- **The context.**
  - `report` inside each context hook, and outside a run.
  - `report` and `get_analysis` raise in `did_change` and
    `get_preserved_analyses`.
  - A retained `get_analysis_manager()` view raises after its hook.
  - Diagnostics come back as the same objects, and core-made ones once.
- **Errors.** For each hook: the class, the core's message, `hook` as
  the Python hook's name, `pass_name`, `diagnostics`, `records`, and
  `__cause__` as the same exception object. Also `KeyboardInterrupt` unwrapped, `Nested`
  per N-S6-2, the verification report, and non-convergence with its
  records.
- **Analyses.**
  - Caching by identity within a run, only for frozen IR.
  - Merge-only transfer: the output's own result wins.
  - Pinning during a run, and release after it, with a `weakref`.
  - Uncached standalone runs.
  - `PreservedAnalyses` by `Identifier`, and preserving a Python analysis
    through a native pass's all-or-nothing set.
- **Pipelines.**
  - A group changed after it was added, and one pass added twice.
  - Two concurrent runs of one manager from two threads.
  - A re-entrant run inside a hook.
  - `skipped`, `pass_runs()` and `run_count()`.
- **Verification.** The default registry verifier blames the producer and
  verifies only changed outputs. `set_verifier(None)` turns it off. A
  standalone `execute` does not verify.
- **Validation.**
  - A `Validator` subclass and a `CompilerPass` validator, collect-all.
  - The silent-failure text, and the `ValidatorRecord` slices as shared
    objects.
- **Native passes.** A mixed pipeline over an `Expression` calls no Python
  for the native passes (checked by a counting rule and a counting pass),
  and the output is the input object when nothing changed. A Python
  subclass overriding `run_pass` is honored. A formatter subclass that
  defines a `visit_*` method is refused, per N-S6-3.
- **Logging.** The loggers and levels of D-S6-5.
- **Pickles.** The pickles of D-S6-16.

**Migrating the existing tests.** No test is skipped or deleted without a
rewrite:

- **`test_core.py` (34).**
  - The counter tests (`test_run_counter_*`, and the counter parts of
    `test_compiler_pass_executes_and_tracks_stats` and
    `test_compiler_pass_skip_path_uses_noop_output`) change per N-S6-1.
  - The two pass-through tests change per N-S6-2.
  - `test_compiler_pass_rejects_none_input` pins that `None` is accepted
    and that an override can refuse it (D-S6-3).
  - The six wrap tests pin the core's messages, the `hook` attribute and
    the cause (D-S6-6).
  - `test_execute_emits_lifecycle_debug` pins the new entering line
    (D-S6-5).
  - The registry tests keep their meaning under N-S6-1's (a).
- **`test_manager.py` (43).**
  - The 14 tests that use an `AnalysisManager` directly (`clear`,
    `invalidate`, `transfer`, frozen-only caching, the weakref fallback,
    eviction, the concurrency, the logging, and the cache after a run) and
    the 5 on the binding are rewritten to observe the cache through
    analysis run counts inside pipelines (D-S6-8), as the Rust manager
    stories do. `test_pass_manager_configuration_is_read_only` drops its
    `analysis_manager` half.
  - `test_analysis_manager_does_not_block_ir_from_garbage_collection`
    pins release after the run.
  - The record tests gain `skipped`, and non-convergence pins the core's
    text and the records.
- **`test_manager_properties.py` (2)** is unchanged.
- **`test_validation.py` (24).** The report tests are unchanged. The
  manager tests pin `ValidatorRecord`s and the core's silent-failure text;
  the two tests of the synthesized texts are rewritten (D-S6-13).
- **`test_verification.py` (58).**
  - The registry, `register_verification`, `run_verification` and
    `VerifiableMixin` tests keep their meaning.
  - The auto-verification tests move into pipelines (D-S6-12): standalone
    pre and post checks become pipeline input and output checks.
  - The `_auto_verify` tests become `set_verifier` tests, and
    `test_auto_verify_uses_analysis_manager_cache_across_passes` pins R-6
    (a changed output is verified again).
  - The blame user story keeps its meaning, with the core's text.
- **`test_visitable.py` (21) and `test_rewritable_pass.py` (39)** are
  unchanged under N-S6-3's (a), except that assertions on hook defaults
  follow D-S6-2 and D-S6-3.
- **Consumers.** The 22 `match=` checks move to `__cause__` (D-S6-6).
  `test_pprint.py`'s two formatter tests change per N-S6-3. The
  benchmarks' `warm_analysis_manager` fixture becomes `_warm_cache`.


### S6 resolutions (2026-09-25)

These were decided under the user's standing policy: Rust semantics where
the two differ, Python names where the meaning is the same, the
CompilerPass hook names stay Python, and work continues unless a decision
is critical.

- **N-S6-1: option (a).** `register_pass`, `CompilerPass.create`,
  `get_registered_passes` and `PassInfo` keep their Python API as a
  registry of Python classes; the native passes register in it. The binding
  never builds a core `PassRegistry`, so one registry is live.
  `get_run_count` and `get_total_run_count` are removed, in favor of
  `PassResult.skipped`, `PassRunRecord.skipped` and
  `PassManagerResult.run_count()`.
- **N-S6-2: option (a), Rust semantics.** A nested run's error is
  `Nested`: the outer exception names the outer pass and hook, keeps the
  outer diagnostics, and has the inner exception as `__cause__`. A
  `Pass*Error` raised by a hook's own code is a `Hook` failure like any
  other exception.
- **N-S6-3: option (a) for the three visitor classes, and the formatter
  stays Python.** `VisitablePass`, `AnalysisVisitablePass` and
  `RewritablePass` stay Python subclasses of the Rust-backed
  `CompilerPass`, with their walks inside `run_pass` and their semantics
  unchanged. `ExpressionPrettyFormatter` also stays a Python `VisitablePass`,
  not a native pass: `pformat_expression` already renders through the core,
  so a native formatter would gain little and would break the per-node
  override its docstring advertises. This amends D-S6-14, which keeps only
  `RewriteRuleApplier` as a native `extends` class if the binding makes
  that worthwhile.
- **Error text (amends D-S6-6).** Messages keep the core's wording but name
  the *Python* hooks, for example `pass "X" failed in run_pass`, not
  `failed in run`, since the hook names are Python. The binding maps the
  hook when it renders the message. `hook` stays as an attribute.

### S6.1 baseline (2026-09-25, a5f5eb2 plus the new benchmarks)

`benchmarks/test_pass_infrastructure.py` implements the benchmark plan
above, with its passes, IR types and verification passes in
`benchmarks/conftest.py`.

- **The IR.** `VerifiedBox` has one registered verification pass, and
  `TwiceVerifiedBox`, a subclass, adds a second, so `run_verification`
  finds two through the MRO. `IncrementPass` now builds a box of its
  input's type, so the verified pipeline stays over `VerifiedBox`.
- **The helpers.** `_warm_cache` (D-S6-8) times a second `get_analysis`
  inside one pass run of a pipeline. This replaces the S0 row's
  `AnalysisManager.get` on a standalone manager and the
  `warm_analysis_manager` fixture. Today the two paths cost the same,
  8.7 µs, since a bound pass asks its manager. `_run_count` (N-S6-1)
  counts the pass records, which equals the run count when no pass
  skips; after the switch it is `result.run_count()`.
- **The other rows.** The preserving pipeline chains five passes that
  each read a counting analysis and return a new, equal box. The run is
  unchanged, so the analysis is computed once and carried from box to box.
  The mixed pipeline runs `RewriteRuleApplier` with S5's four rules, a
  Python identity pass, and `ExpressionPrettyFormatter` over the deep
  tree. It is a `PassManager[Any]`, since the formatter's output is a
  `str`.

Median time per call, from `uv run --python 3.11 nox -s benchmark-3.11 --
-k "test_pass_infrastructure or rewrite_rule_applier_execute or
visitable_pass_walk or pformat_expression or validation_report"`, measuring
today's pure-Python pass infrastructure. The machine is the S0 one, with
Python 3.11.13 and pytest-benchmark 5.3.0. The load average was about 1,
and the table lists the best of three runs' medians.

| Benchmark | before |
|---|--:|
| `test_compiler_pass_execute` | 11.3 µs |
| `test_compiler_pass_call` | 11.4 µs |
| `test_compiler_pass_execute_with_every_hook_overridden` | 10.7 µs |
| `test_compiler_pass_execute_skipped` | 6.60 µs |
| `test_compiler_pass_execute_failing` | 145.2 µs |
| `test_compiler_pass_report_of_100_diagnostics` | 1.50 ms |
| `test_compiler_pass_create` | 493 ns |
| `test_pass_manager_run_of_5_passes` | 343.7 µs |
| `test_pass_manager_run_of_50_passes` | 3.42 ms |
| `test_pass_manager_fixpoint_group_of_10_iterations` | 680.6 µs |
| `test_pass_manager_run_with_verification` | 436.3 µs |
| `test_analysis_manager_cache_hit` | 8.71 µs |
| `test_analysis_preserved_across_5_passes` | 372.0 µs |
| `test_validation_manager_validate_of_10_validators` | 300.5 µs |
| `test_run_verification` | 28.2 µs |
| `test_mixed_pipeline_over_a_deep_expression` | 500.3 µs |
| `test_rewrite_rule_applier_execute_of_a_deep_tree` (rerun) | 222.9 µs |
| `test_visitable_pass_walk_of_deep_tree` (rerun) | 287.3 µs |
| `test_pformat_expression_of_deep_tree[symbolic]` (rerun) | 8.81 µs |
| `test_pformat_expression_of_deep_tree[functional]` (rerun) | 7.71 µs |
| `test_pformat_expression_of_deep_tree[show_id]` (rerun) | 10.3 µs |
| `test_validation_report_construction` (rerun) | 14.9 µs |
| `test_validation_report_build_of_100_diagnostics` (rerun) | 93.8 µs |
| `test_validation_report_eq` (rerun) | 939 ns |
| `test_validation_report_format` (rerun) | 3.04 µs |
| `test_validation_report_errors` (rerun) | 541 ns |
| `test_validation_report_has_errors` (rerun) | 54 ns |

- **One pass.** A run costs about 11 µs, whichever hooks are Python: the
  every-hook pass matches the floor, since its hooks are cheaper than the
  defaults it replaces (`is not` for `!=`, no `None` check). A skipped run
  takes 6.6 µs.
- **Failures and diagnostics.** A failing run takes 145 µs and each
  reported diagnostic about 15 µs. Both paths log: the failure with its
  traceback, and each diagnostic at its level. The benchmarks run under
  pytest's log capture, which receives every record, since the package's
  loggers are at DEBUG.
- **Pipelines.** A pipeline costs about 68 µs per pass, about six times a
  standalone run: 5 passes take 344 µs, 50 take 3.4 ms, and 10 fixpoint
  iterations 681 µs. Verification adds 93 µs over the 5 passes.
  Validation costs 30 µs per validator.
- **The rest.** A cache hit costs 8.7 µs. The expression pipeline takes
  500 µs, of which the applier is about 220 µs.

### S6.3 benchmarks: the `ValidationReport` representation (before and after)

D-S6-17 is implemented as planned: the `_rs.ValidationReport` pyclass keeps
the diagnostic and record tuples and drops the `ValidationReport<Py<PyAny>>`
beside them. Construction checks that each item is a `Diagnostic`.
`errors()`, `warnings()`, `infos()`, `has_errors()`, `format()`,
`raise_if_failed()`, `==` and `hash` walk the diagnostic tuple through
borrowed references. Each borrows the Rust `Diagnostic` from its frozen
pyclass, which costs one type check per item and no clone. The Python API, the reprs, the text and the pickles are unchanged,
and no test changed.

Median time per call, from `uv run --python 3.11 nox -s benchmark-3.11 --
-k "validation_report or run_verification or validation_manager or
diagnostic"` on the S0 machine with Python 3.11.13 and pytest-benchmark
5.3.0. "Before" is 7f2a554, exported with `git archive` under `target/`;
"after" is the S6.3 commit. The two ran three times each, interleaved
(before, then after, in each round), with a load average of about 2, and
the table lists the best of the three medians. The dataclass column is
the "python after" column of the S3a table: the retired pure-Python
dataclass, measured on 40eaa23 under a heavier load. The `Note` and `Diagnostic` rows, whose code S6.3 does not change,
came out within 4% of 1.00.

| Benchmark | dataclass (S3a) | before | after | after / before |
|---|--:|--:|--:|--:|
| `test_validation_report_construction` | 999 ns | 14.1 µs | 469 ns | 0.03 |
| `test_validation_report_build_of_100_diagnostics` | 367 µs | 93.0 µs | 80.9 µs | 0.87 |
| `test_validation_report_eq` | 31.7 µs | 982 ns | 1.75 µs | 1.78 |
| `test_validation_report_format` | 44.2 µs | 3.11 µs | 3.46 µs | 1.11 |
| `test_validation_report_errors` | 10.1 µs | 509 ns | 857 ns | 1.68 |
| `test_validation_report_has_errors` | 459 ns | 53 ns | 58 ns | 1.10 |
| `test_compiler_pass_report_of_100_diagnostics` | - | 1.53 ms | 1.53 ms | 1.00 |
| `test_validation_manager_validate_of_10_validators` | - | 306.0 µs | 305.2 µs | 1.00 |
| `test_run_verification` | - | 27.7 µs | 27.7 µs | 1.00 |

The S3a leftover is resolved. Building a report from 100 existing
diagnostics drops from 14.1 µs to 469 ns, half the dataclass's 1 µs, and
building 100 diagnostics together with their report gets 13% faster.

The planned "other operations keep their S3a speed" held only in part:

- `has_errors()` stops at the first error, and `format()` is dominated by
  the text, so they stay within 11%.
- `==` of two reports of 100 distinct diagnostics goes from 0.98 to
  1.75 µs, and `errors()` from 0.51 to 0.86 µs. The type check costs about
  4 ns per item: `==` makes two per pair, and `errors()` one per item on
  top of building its tuple. Both stay 12 to 18 times faster than the
  dataclass.

This is recorded as an accepted cost (cross-cutting rule 5), since the
alternative measured worse overall. That alternative kept, beside the
tuple, a boxed slice of `Py<Diagnostic>` handles, typed once at
construction:

- `==`, `format()` and `errors()` came out at 1.01, 3.10 and 0.42 µs,
  with `has_errors()` unchanged.
- Construction took 1.39 µs, more than the dataclass, for the handles'
  allocation and reference counts.

A report is usually built once and asked once or twice, as
`raise_if_failed()` and the verification path do. Construction plus one
`errors()` then costs 1.33 µs with the tuples alone and 1.81 µs with the
handles. The handles would also be a second list of the same diagnostics,
which D-S6-17 set out to avoid.

### S6.1 to S6.3 status

S6.1 to S6.3 were implemented on 2026-09-25 in four commits: the
benchmarks and their baseline (9452b46), the core additions (7f2a554),
the `ValidationReport` representation (30dbf7e), and these notes. At the
end: `pytest` 7,054 passed, `-m "not very_slow"` 7,087 passed, the
`property` session 280 passed, `lint` and `type_check` clean, and the Rust
gate green (fmt, clippy `-D warnings`, 2,660 tests, doc `-D warnings`,
deny, `cargo +1.85 check`). No Python test changed.

### S6.2 implementation notes

The additions are in `rust/fhy-core/src/tree/node.rs`,
`src/pass/preserved.rs`, `src/pass/context.rs` and the new
`src/pass/detached.rs`, and the crate README lists them. The API:

```rust
impl NodeIdentity {
    pub fn of_ptr<T: ?Sized>(pointer: *const T) -> Self;
}

impl AnalysisId {                       // now Clone, no longer Copy
    pub fn of_identifier(name: &Identifier) -> Self;
    pub fn identifier(&self) -> Option<&Identifier>;
}

impl PreservedAnalyses {
    pub fn is_id_preserved(&self, id: &AnalysisId) -> bool;         // was by value
    pub fn preserved_ids(&self) -> impl Iterator<Item = &AnalysisId> + '_; // was owned
}

impl PassContext<'_> {
    pub fn analysis_by_id<T: NodeHandle, V: Send + Sync + 'static>(
        &mut self, ir: &T, id: &AnalysisId, compute: impl FnOnce(&T) -> V,
    ) -> Arc<V>;
    pub fn with_detached_analyses<R>(
        &mut self, callback: impl FnOnce(&DetachedAnalyses) -> R,
    ) -> R;
}

#[derive(Debug, Clone)]
pub struct DetachedAnalyses { /* Arc<Mutex<..>> */ }   // Send + Sync + 'static

impl DetachedAnalyses {
    pub fn analysis_by_id<T: NodeHandle, V: Send + Sync + 'static>(
        &self, ir: &T, id: &AnalysisId, compute: impl FnOnce(&T) -> V,
    ) -> Result<Arc<V>, DetachedAnalysesExpired>;
    pub fn is_expired(&self) -> bool;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct DetachedAnalysesExpired;     // "the detached analyses expired when their callback returned"
```

Choices the plan left open, and where the shape differs from it:

- **`NodeIdentity::of_ptr`** is as planned. It drops a wide pointer's
  metadata, so a slice, a trait object and a thin pointer to the same
  address agree, and `of_arc` now calls it. Its rustdoc states the
  `NodeHandle` contract: the handle keeps the pointee alive.
- **`AnalysisId` is `Clone`, not `Copy`** (a different shape from the
  plan's "it stays `Copy`"). An id of an identifier holds the
  `Identifier`, whose name hint is an `Arc<str>`. To stay `Copy` and still
  display the name hint, the id would need a `&'static str`, which means
  leaking each name hint or keeping a process-global table of them. B5
  removed global state from the pass framework (F-006), so the id holds
  the identifier, and a clone costs one reference-count increment.
  - `is_id_preserved` therefore takes the id by reference, and
    `preserved_ids` yields references. `preserve_id` still takes it by
    value.
  - Four lines of `tests/it/pass/core_stories.rs` changed for these
    signatures only: three `is_id_preserved` calls and one
    `preserved_ids().cloned()`.
  - The commit is not marked breaking; the crate is unpublished
    (decision 9).
- **The order of ids.** Ids of types come first, by type name as before.
  Ids of identifiers follow, ordered by the identifier's id, not by its
  name hint as the plan said. Identifiers are equal exactly when their
  ids are, and `Identifier::try_restore` can give an equal identifier
  another name hint. An order by name hint would then disagree with `==`,
  and `BTreeSet` lookups in a preservation set could miss. Display is the
  name hint, as planned, and `Hash` includes which kind of id it is.
- **`AnalysisId::identifier`** is new. S6.4's
  `PreservedAnalyses.analysis_names` needs it to rebuild the Python
  identifiers from a set of ids.
- **`analysis_by_id`'s contract.** `compute` receives the node. The
  caller keeps one computation, with one result type, per id. The id of
  an analysis type reaches that type's cached result when `V` is its
  output. A result cached under the id with another type is recomputed
  and replaced, the cache's existing rule. The private cache now has
  separate lookup and insert steps, so the detached handle can compute
  between them.
- **The detached handle.**
  - **Shape.** The handle is an `Arc<Mutex<_>>` over one of three states:
    the run's cache, no cache (a standalone run, where every request
    computes afresh as `PassContext::analysis` does), or expired.
  - **Moving the cache.** The cache moves in and out with `mem::take`, so
    nothing is copied. A drop guard moves it back, which also covers an
    unwinding callback.
  - **Lending it.** The callback receives the handle by reference, as
    `thread::scope` lends its scope, and clones it to keep it.
  - **No lock while computing.** `compute` runs with no lock held. A
    request from inside `compute`, or from another thread, therefore
    cannot deadlock, and the S6.4 binding can call Python there without
    holding a Rust lock across a call that may wait for the interpreter.
    When a nested request caches the same id and node while `compute`
    runs, its result is the one returned, so one result per id and node
    stays true.
  - **Expiry.** An expired handle holds nothing: the state becomes
    "expired" when the cache moves back, so a clone kept past the callback
    pins no node. The tests check this with the node's handle count.
  - **Only `analysis_by_id`.** The handle has no typed `analysis::<A>()`,
    since the binding needs only ids. A typed request reaches the same
    result through `AnalysisId::of::<A>()`.
- **Tests first.** The new `tests/it/pass/dynamic_analysis_stories.rs`
  has 23 tests: ids of identifiers and their equality, order, display and
  preservation; the cache by id, within a pass, standalone, per id and
  node, through a type's id, and across a type change; merge-only
  transfer of preserved ids; and the detached handle, covering serving
  the cache, standalone runs, expiry, releasing nodes, a panicking
  callback, another thread, and nested requests. `core_stories.rs` gained
  three `of_ptr` tests, and the rustdoc four examples.
  - The tests were written before the code, and failed to compile
    against the old API.
  - Two mutations checked that they bite. Not moving the cache back
    failed three tests. Comparing identifier ids by name hint, with ids of
    different kinds equal, failed two.
- **Still to confirm.** If S6.4 finds the handle unnecessary, it is
  dropped, as planned.

### S6.3 implementation notes

- **Borrowing.** The operations walk the tuple with `iter_borrowed` and
  borrow each Rust `Diagnostic` through `Borrowed::cast` and
  `Borrowed::get`, with no `unsafe`. A type check per item is the price
  of reading a Python tuple as typed data. Construction's `TypeError`
  text is unchanged.
- **Hashing.** The hash covers the diagnostic count, each Rust
  diagnostic and the records' Python hash. The values differ from
  before, but equal reports still hash equally, and an unhashable record
  still raises `TypeError`.
- **Reports built in Rust.** They will reach Python through D-S6-18's
  builders, which build the `Diagnostic` objects and then the tuple. So
  one representation serves both origins, and no Rust
  `ValidationReport` of Python records is needed.

### S6.4 to S6.7 status

S6.4 to S6.7 were implemented on 2026-09-25 in five commits: the binding
(b4bb6ba); the Python switch, marked breaking (5f1a9aa); the migrated
tests and the new interface suite (d9cf3b6); the constructor-argument
check moved into the Rust bases, which the benchmarks called for
(441f77d); and these notes (af12d0c). No test was skipped or deleted. At the end: `pytest` 7,163 passed (the
pass-infrastructure tests gained one in the rewrites, and the interface
suite adds 108), `-m "not very_slow"` 7,196 passed, the `property` session
280 passed, `lint` and `type_check` clean, `tests/test_rs_stub.py` green,
and the Rust gate green (fmt, clippy `-D warnings`, 2,660 tests, doc
`-D warnings`, deny, `cargo +1.85 check`).

### S6 benchmarks (before and after)

Median time per call, from `uv run --python 3.11 nox -s benchmark-3.11 --
-k "test_pass_infrastructure or rewrite_rule_applier_execute or
visitable_pass_walk or pformat_expression or validation_report"` on the S0
machine with Python 3.11.13 and pytest-benchmark 5.3.0. "Before" is
a308100, the last commit with the pure-Python pass infrastructure,
exported with `git archive` under `target/`; "after" is the S6.7 tree. The
two ran three times each, interleaved (before, then after, in each round),
with a load average of 9 to 12, and the table lists the best of the three
medians. The "before" column is 5 to 15% above the S6.1 table, which was
measured at a load of about 1; the ratios compare runs under the same
load.

| Benchmark | before | after | after / before |
|---|--:|--:|--:|
| `test_validation_report_construction` | 641 ns | 486 ns | 0.76 |
| `test_validation_report_build_of_100_diagnostics` | 88.5 µs | 88.3 µs | 1.00 |
| `test_validation_report_eq` | 1.9 µs | 2.1 µs | 1.12 |
| `test_validation_report_format` | 3.9 µs | 3.8 µs | 0.98 |
| `test_validation_report_errors` | 945 ns | 959 ns | 1.01 |
| `test_validation_report_has_errors` | 65 ns | 64 ns | 0.99 |
| `test_pformat_expression_of_deep_tree[symbolic]` | 9.6 µs | 9.7 µs | 1.01 |
| `test_pformat_expression_of_deep_tree[functional]` | 8.5 µs | 8.7 µs | 1.02 |
| `test_pformat_expression_of_deep_tree[show_id]` | 11.2 µs | 11.5 µs | 1.02 |
| `test_visitable_pass_walk_of_deep_tree` | 303.5 µs | 285.2 µs | 0.94 |
| `test_compiler_pass_execute` | 12.2 µs | 2.9 µs | 0.24 |
| `test_compiler_pass_call` | 12.5 µs | 2.2 µs | 0.18 |
| `test_compiler_pass_execute_with_every_hook_overridden` | 11.9 µs | 4.1 µs | 0.35 |
| `test_compiler_pass_execute_skipped` | 7.4 µs | 2.8 µs | 0.37 |
| `test_compiler_pass_execute_failing` | 158.4 µs | 51.1 µs | 0.32 |
| `test_compiler_pass_report_of_100_diagnostics` | 1.66 ms | 1.70 ms | 1.03 |
| `test_compiler_pass_create` | 542 ns | 546 ns | 1.01 |
| `test_pass_manager_run_of_5_passes` | 382.6 µs | 26.6 µs | 0.07 |
| `test_pass_manager_run_of_50_passes` | 3.73 ms | 212.8 µs | 0.06 |
| `test_pass_manager_fixpoint_group_of_10_iterations` | 760.1 µs | 48.8 µs | 0.06 |
| `test_pass_manager_run_with_verification` | 484.1 µs | 34.1 µs | 0.07 |
| `test_analysis_manager_cache_hit` | 9.8 µs | 778 ns | 0.08 |
| `test_analysis_preserved_across_5_passes` | 420.9 µs | 34.1 µs | 0.08 |
| `test_validation_manager_validate_of_10_validators` | 329.9 µs | 147.0 µs | 0.45 |
| `test_run_verification` | 30.5 µs | 12.5 µs | 0.41 |
| `test_mixed_pipeline_over_a_deep_expression` | 561.2 µs | 309.1 µs | 0.55 |
| `test_rewrite_rule_applier_execute_of_a_deep_tree` | 245.6 µs | 232.2 µs | 0.95 |

Every pass-infrastructure hot path is faster or unchanged, so P3 and P2
stand and no path changes pattern (cross-cutting rule 5):

- **One pass.** The lifecycle's floor drops from 12.2 to 2.9 µs (`__call__`
  2.2 µs): the hooks a class does not override run in Rust, so an
  identity pass makes one Python call, to `run_pass`. With all seven hooks
  in Python a run takes 4.1 µs, about 0.2 µs per hook. A skipped run takes
  2.8 µs, and a failing one 51 µs, a third of before; the rest is the
  failure's logging with its traceback.
- **Pipelines** are 13 to 17 times faster: about 4.3 µs per pass instead
  of 75 µs. The per-run cost is the Rust pipeline and its adapters built
  from the current items (D-S6-11), the records built once through their
  public classes, and the registry verifier's lookup of
  `VerificationRegistry.get_passes_for` for the input and each changed
  output. Verification adds 7.5 µs over the five passes, where it added
  100 µs.
- **Analyses.** A cache hit costs 0.78 µs instead of 9.8 µs, and the
  preserving pipeline 34 µs instead of 421 µs.
- **Validation** is 2.2 to 2.4 times faster; each validator's cost is now
  mostly its Python `visit` and the log lines of the report.
- **Expressions.** The mixed pipeline is 1.8 times faster; the applier
  alone is unchanged within noise, since its walk was already Rust (S5),
  and so are the visitor walk and the formatter rows, which are Python.
- **Unchanged rows.** The `ValidationReport` rows run code S6 does not
  change; their ratios of 0.76 to 1.12 are the noise of this load, as
  the reruns show (the first set of rounds had 0.96 to 1.00).

Two rows needed a second look:

- **`test_compiler_pass_create`** came out at 1.44 in a first set of
  rounds: 761 against 530 ns. The cost was structural and in the Python
  layer: `CompilerPass.__init__`, kept so a pass refuses constructor
  arguments it does not take, added a Python frame and a `super()` call,
  about 160 ns, to every construction. The check moved into the Rust
  bases' `__new__`, which runs it only when arguments are given (and
  then raises `X() takes no arguments`, as `object` does), and
  `CompilerPass`, `Analysis` and `Validator` define no `__init__`.
  Construction dropped from 280 to 114 ns, and the row to 1.01; the table
  is the second set of rounds.
- **`test_compiler_pass_report_of_100_diagnostics`** is 1.01 and 1.03 in
  the two sets, so about 2% slower, within the noise of this load but
  consistently on the slow side. A reported diagnostic now also crosses
  into Rust twice: once to record it in the hook's frame, and once, when
  the run ends, to be found again in the run's scope by value (D-S6-18).
  Outside pytest's log capture, which costs about 15 µs per diagnostic in
  both columns, that is about 0.3 µs per diagnostic of 3 µs. This is
  recorded as an accepted cost (cross-cutting rule 5): the per-diagnostic
  cost is the log capture's, and the only way to remove the lookup would
  be position-keyed tables, which the notes below explain the adapter
  cannot build reliably.

### S6.4 to S6.7 implementation notes

Choices the decisions left open, made while implementing S6.4 to S6.7,
and where the shape differs from the plan:

- **Module layout.** `rust/fhy-core-py/src/pass.rs`, with `ir.rs`
  (`PyIr`), `scope.rs` (the state of one run), `context.rs` (the hook
  frames and the `AnalysisManager` view), `compiler_pass.rs`
  (`CompilerPassBase` and the Python-pass adapter), `analysis.rs`
  (`AnalysisBase`, `PreservedAnalyses`), `validation.rs` (`ValidatorBase`,
  `ValidationManager`, the validators the binding runs), `manager.rs`
  (`PassManager`, `FixpointPassGroup`), `records.rs` (`PassResult` and the
  records, through one macro), `error.rs` and `convert.rs`. There is no
  `native.rs`: no pass became native (see below). The records share
  `diagnostic.rs`'s new builders, `diagnostic_to_python` and
  `report_to_python`.
- **The hook table** is `CompilerPass._python_hooks`, a bit mask that
  `CompilerPass.__init_subclass__` computes once per class from the class
  that defines each hook with a default (`validate_input`, `should_run`,
  `validate_output`, `did_change`, `get_preserved_analyses`,
  `get_pass_name`). The adapter reads it once per run and runs the
  defaults of the unset bits in Rust. A hook assigned to a class after it
  was created, or to an instance, is not seen; no code in `src` or
  `tests` does either. `get_noop_output` has no bit: it is called only on
  a skip, where calling Python is required anyway.
- **The owned context** is a frame on a thread-local stack, found by the
  pass object's address, rather than an object bound to the instance.
  So a pass that runs itself inside its own hook, and one pass object run
  from two threads, each see their own frame. A frame holds the detached
  analysis handle of S6.2, which is therefore kept, and the diagnostics
  its hook reported; it expires when the hook returns, and an
  `AnalysisManager` view kept past that raises. `get_analysis_manager`
  raises in `did_change` and `get_preserved_analyses` too, as `report`
  and `get_analysis` do.
- **Diagnostics that come back as themselves** (D-S6-18) are found by
  value, not by position. The run's scope keeps each object a hook
  reported under the hash of its Rust diagnostic, in report order, and a
  conversion takes the first equal one. The plan's positions would need
  the adapter to know which core record a context becomes, which it does
  not, and a verifier's contexts are discarded when verification passes.
  The only difference: a diagnostic the core made that is equal to one a
  hook reported in the same run could come back as that equal object.
- **Failure texts name the Python hook everywhere.** The resolution
  amended the error message; the binding also rewrites the core's
  failure diagnostic, the last one of a hook failure, to the same
  wording, `pass "X" failed in run_pass: ValueError: boom`, and renders a
  nested chain with Python names at every level. A pass run as a check
  (in a validation or as a verification pass) runs through the binding's
  own four-hook sequence, the one the core's `PassValidator` runs, so it
  writes its failure diagnostic with the Python hook's name. A failing
  `Validator` gets the core's synthesized text, logged by the binding
  with the exception.
- **Nested errors** (N-S6-2). An exception the binding raises keeps its
  Rust `PassError` in a private `_rust_error` cell, which its pickles
  leave out. When it propagates out of a lifecycle hook, the adapter
  takes the error out and returns it, so the core builds `Nested`, and
  the run's scope keeps the exception to become the outer `__cause__`.
  In a validation a nested error does not nest; its chain goes into the
  failure text.
- **Exceptions that are not `Exception`s** (`KeyboardInterrupt`) are
  recorded in the run's scope; later hooks of that run call no Python, and
  the run's boundary raises the exception itself, also from a validation,
  whose collect-all loop would otherwise go on.
- **Analysis results** are cached as a cell per analysis id and node, which
  the binding fills after the Python analysis returns, so an analysis
  that raises caches nothing and runs again on the next request. The
  cacheability check reads `Frozen`'s members (`freeze`, `assert_frozen`,
  a true `is_frozen`) instead of the runtime protocol's slow `isinstance`.
- **The default verifier** is one validator, named `verification`, that
  runs a new instance of each registered verification pass as a check,
  so a verification error's report has one `ValidatorRecord`.
  `VerificationAnalysis` and `run_verification` still build a
  `ValidationManager` with one record per pass.
- **Logging.** The pipeline's per-item DEBUG lines and a group's INFO
  line are written from the records after the run, since the core has no
  hooks for them; the per-pass line gains `skipped=`, the group's loses its
  elapsed time, and their `funcName` is the logging helper's. A
  non-convergence is logged at ERROR where the pipeline raises it. The
  binding logs through `core._log_diagnostic` and `core._lifecycle_logger`,
  and logs a pass's lifecycle lines only when its logger is enabled for
  DEBUG.
- **Stricter arguments**, beyond D-S6-15's list: the records check their
  fields (a `bool` for `changed`, `skipped`, `converged` and `failed`,
  `Diagnostic`s, a `PreservedAnalyses`, an `Identifier` group name, an
  `int` iteration); the pipelines' names must be `Identifier`s; a
  `Validator`'s `name` must be a `str`; `PassValidationError` and
  `PassExecutionError` gain the keyword arguments `pass_name`, `hook`,
  `diagnostics` and `records`.
- **Smaller shapes.** `skipped` is the last positional-or-keyword
  parameter of `PassResult` and `PassRunRecord`, so a pickle is a plain
  call. A `FixpointGroupRecord`'s `group_name` is the group's own
  `Identifier` object, and a preserved set that crossed the core rebuilds
  its Python identifiers from its ids (equal, not identical).
  `AnalysisManager` is `_rs.AnalysisManager` itself, without a public
  subclass. A pass, an analysis or a validator whose class defines no
  `__init__` still refuses constructor arguments, through its Rust base's
  `__new__` (see the benchmarks). The name is read once
  per run; the description is never read, since the core uses it only in
  its registry. `ValidatorBase` keeps no diagnostics, so a `Validator`'s
  `report` outside a check is only logged.
- **No native pass** (D-S6-14 as N-S6-3 amends it). `RewriteRuleApplier`
  stays a Python `CompilerPass` whose `run_pass` calls the Rust walk: the
  adapter's floor is about 3 µs of the applier's 220 µs run over the deep
  tree, so a native holder, with its Expression adapter and output
  materializer, would not be worth its code. `ExpressionPrettyFormatter`
  stays a Python `VisitablePass`, as N-S6-3 decided.

Tests migrated in S6.6 (`test_core.py` is `C`, `test_manager.py` `M`,
`test_validation.py` `V`, `test_verification.py` `F`). Every other test
of the seven files kept its name; the six wrap tests of `C`, the two
`test_visitable.py` error tests, and the 22 consumer checks that matched
the cause in the message (in `test_evaluator.py`, `test_inline_pass.py`,
`test_sympy_pass.py`, `test_sympy_natives.py`, `test_z3_pass.py` and
`test_numpy_evaluator.py`) now assert on the core's message, `hook` and
`__cause__` (D-S6-6). None was skipped or deleted; `F` gained one test.

| Test | Now | Reason |
|---|---|---|
| C `test_compiler_pass_executes_and_tracks_stats` | `test_compiler_pass_executes_and_reports_run_statistics` | N-S6-1: `PassResult.skipped` |
| C `test_compiler_pass_rejects_none_input` | `test_compiler_pass_accepts_none_input_unless_an_override_refuses_it` | D-S6-3 |
| C `test_compiler_pass_skip_path_uses_noop_output` | same name | N-S6-1: `skipped` instead of the counters |
| C `test_explicit_pass_{validation,execution}_error_passes_through_unchanged` (2) | `..._is_wrapped_as_a_hook_failure` | N-S6-2 |
| C `test_run_counter_counts_only_real_executions` | `test_run_statistics_count_only_real_executions` | N-S6-1: `skipped`, `run_count()` |
| C `test_run_counter_counts_attempts_even_when_run_pass_raises` | `test_failed_run_has_no_record_but_the_completed_work_does` | N-S6-1: a failed run has no record; the error carries the earlier ones |
| C `test_execute_emits_lifecycle_debug` | same name | D-S6-5: the new entering line |
| M `test_pass_manager_applies_analysis_preservation_and_invalidation`, `test_analysis_manager_does_not_cache_non_frozen_ir`, `test_analysis_manager_does_not_block_ir_from_garbage_collection` | same names | D-S6-8: observed inside a run; release after the run |
| M `test_bind_and_get_analysis_manager_are_public_accessors` | `test_get_analysis_manager_is_the_hook_view_of_the_run_cache` | D-S6-8 |
| M `test_analysis_manager_falls_back_when_weakref_finalize_raises` | `test_analysis_of_ir_that_is_not_frozen_yet_is_computed_uncached` | D-S6-8: no weakrefs; uncacheable IR |
| M `test_analysis_manager_clear_drops_cached_entries` | `test_each_run_starts_with_an_empty_cache` | D-S6-8 |
| M `test_analysis_manager_invalidate_with_none_drops_everything` | `test_pass_returning_its_input_keeps_every_result_even_when_preserving_none` | W-12: merge-only |
| M `test_analysis_manager_invalidate_with_preserve_all_keeps_everything` | `test_pass_returning_its_input_and_preserving_all_keeps_every_result` | D-S6-8 |
| M `test_analysis_manager_invalidate_keeps_only_preserved_entries` | `test_changing_pass_carries_only_its_preserved_results` | D-S6-8 |
| M `test_analysis_manager_transfer_moves_preserved_entries` | `test_changing_pass_preserving_all_carries_every_result` | D-S6-8 |
| M `test_analysis_manager_transfer_drops_when_not_preserved` | `test_changing_pass_preserving_none_carries_no_result` | D-S6-8 |
| M `test_bind_analysis_manager_rejects_none` | `test_get_analysis_manager_raises_in_a_hook_without_a_context` | D-S6-4, D-S6-8 |
| M `test_unbind_analysis_manager_clears_binding` | `test_retained_analysis_manager_raises_after_its_hook` | D-S6-8 |
| M `test_unbind_analysis_manager_is_idempotent` | `test_get_analysis_manager_is_none_outside_a_run` | D-S6-8 |
| M `test_analysis_manager_get_is_safe_under_concurrent_callers` | `test_concurrent_runs_of_one_pipeline_are_independent` | D-S6-11 |
| M `test_analysis_manager_survives_concurrent_get_and_gc_eviction` (slow) | `test_concurrent_runs_survive_garbage_collection_of_their_ir` | D-S6-8, D-S6-11 |
| M `test_analysis_manager_logs_cache_hit_and_miss` | `test_analysis_cache_logs_no_hit_or_miss_lines` | D-S6-5 |
| M `test_pass_manager_configuration_is_read_only` | same name | D-S6-8: no `analysis_manager` half |
| M `test_fixpoint_convergence_logs_info` | same name | the line no longer comes from `_run_fixpoint_group` |
| M `test_pass_manager_fixpoint_group_raises_on_non_convergence`, `test_pass_run_record_stores_preserved_analyses_directly` | same names | the core's text and records; `skipped` |
| V `test_validation_manager_synthesizes_error_when_pass_raises_without_reporting` | `test_validation_manager_reports_the_failure_of_a_validation_hook` | D-S6-3, D-S6-13 |
| V `test_validation_manager_record_reports_no_changes_and_preserves_all` | `test_validation_manager_record_holds_the_validators_slice_of_the_report` | D-S6-13: `ValidatorRecord` |
| V `test_validation_manager_synthesizes_balanced_quotes_around_exception_type` | `test_validation_manager_reports_silent_failures_with_the_cores_text` | D-S6-13 |
| V three tests reading `pass_name` of a record | same names | `validator_name` |
| F `test_register_verification_disables_auto_verify_on_decorated_class` | `test_register_verification_leaves_the_class_a_pass_that_never_verifies` | D-S6-12: `_auto_verify` gone |
| F `test_auto_verify_default_is_true_on_compiler_pass` | `test_standalone_execute_never_verifies` | D-S6-12 |
| F `test_auto_verify_pre_raises_...`, `test_auto_verify_post_raises_...` | `test_pipeline_raises_pass_validation_error_when_its_input_is_malformed`, `test_pipeline_raises_when_a_pass_produces_malformed_output` | D-S6-12: pipeline input and output checks |
| F `test_auto_verify_false_on_pass_class_disables_pre_and_post` | `test_set_verifier_none_disables_input_and_output_verification` | D-S6-12 |
| F `test_verification_pass_does_not_recurse_into_auto_verify` | `test_verification_pass_does_not_recurse_into_verification` | D-S6-12 |
| F `test_auto_verify_uses_analysis_manager_cache_across_passes` | `test_pipeline_verifies_its_input_once_and_no_unchanged_output`, and new `test_pipeline_verifies_a_changed_output_again` | R-6 |
| F `test_auto_verify_recomputes_when_pass_changes_ir`, `test_auto_verify_is_silent_when_no_passes_registered_for_ir_type`, `test_pass_manager_surfaces_auto_verify_failure_to_caller` | `test_pipeline_verifies_the_output_of_a_changing_pass`, `test_pipeline_verification_is_silent_when_no_passes_registered_for_ir_type`, `test_pass_manager_surfaces_verification_failure_to_caller` | D-S6-12: renamed, the last two now run a pipeline |
| F `test_pass_manager_continues_when_pass_disables_auto_verify` | `test_pass_manager_continues_with_a_verifier_that_finds_nothing` | D-S6-12: `set_verifier` |
| F `test_registry_mutation_between_passes_does_not_invalidate_cached_results`, `test_verification_analysis_cache_does_not_pin_ir` | same names | D-S6-8: inside runs |
| F `test_user_story_pipeline_blames_pass_that_produced_invalid_ir` | same name | the core's text, the blamed pass and the records |

The new `tests/pass_infrastructure/test_pass_infrastructure_rust_binding.py`
(108 tests) covers the test plan above, except its native-pass items,
since no pass became native: it pins that `RewriteRuleApplier` runs in a
pipeline and keeps its input object when nothing fires.

## S7: the function registry

- **Status:** designed 2026-09-25 at 0bc8952, and implemented the same
  day; see "S7 status" below. D-S7-1 to D-S7-17 apply the policy the user
  already set, and N-S7-1 to N-S7-3 were resolved as option (a).
- **Pattern:** P2 for the three entry classes and for the registry, whose
  core type is a new owned `FunctionRegistry`. The binding keeps one in
  module state, so Python keeps its global `register_function` API.
- **Retires D-S4-4.** S4 kept the registry in Python and gave the Rust
  screen a `SortLookup` adapter over it (`RegistrySorts`). S7 makes the
  Rust registry the only one, and the adapter goes.

### Survey: the Python API

The package is `src/fhy_core/symbolic/expression/registry/`, 794 lines.
`builtins.py` (437) fills it at import, and `errors.py` (227) defines its
two error classes. It is pure Python over the Rust-backed expressions since
S4.3a.

| File | Lines | Contents |
|---|--:|---|
| `__init__.py` | 69 | re-exports 17 names |
| `entries.py` | 322 | `RegisteredFunction`, `NativeFunction`, `NativeConstant`, the `RegisteredEntry` and `CallTargetResolver` aliases, the construction checks |
| `storage.py` | 204 | the three dicts and the lock, the lookups, `set_registry_state_for_tests` |
| `api.py` | 199 | `register_function`, `register_native_function`, `register_native_constant`, `try_get_registered_result_sort` |

**The entries (`entries.py`).** All three are frozen dataclasses, open to
direct construction without registering.

- **`RegisteredFunction(name, parameters, parameter_sorts, result_sort,
  body)`**, a `DerivedEquivalenceMixin`.
  - `__post_init__` raises `ValueError` for an empty name, for sorts whose
    count differs from the parameters' (`RegisteredFunction 'f':
    parameter_sorts length (2) does not match parameters length (1).`),
    and for a body whose free identifiers are not all parameters or
    canonical identifiers of registered constants (`RegisteredFunction
    'f': body references identifiers not in its parameters: x, y.`, sorted
    by name hint).
  - The capture check reads the *global* registry at construction, so it
    is order-dependent: a body naming a constant is refused before the
    constant is registered and accepted after.
  - A repeated parameter is accepted. Substitution then keeps the last
    argument, and a binder frame the last pairing.
  - A call is a reference by name, so a recursive body is accepted.
  - `==` and `hash` are the dataclass's, over every field, `name`
    included; the body compares structurally (D-S4-1).
- **`NativeFunction(name, parameter_sorts, result_sort, implementation)`.**
  `implementation` is a Python callable taking and returning `bool`, `int`
  or `float`. Construction raises `ValueError` for an empty name, or when
  `inspect.signature` shows the callable cannot take
  `len(parameter_sorts)` positional arguments. A callable with no
  inspectable signature (`max`, some `math` functions) is not checked.
  `==` compares every field, the callable by its own `==`.
- **`NativeConstant(name, sort, value)`.** `value` is a `bool`, `int` or
  `float` that `is_python_value_compatible_with_sort` accepts: a `bool`
  only for `BOOL`, a non-negative `int` for `NAT`, an `int` for `INT`, an
  `int` or `float` for `REAL`. The entry holds no identifier.
- Mutating a field raises `dataclasses.FrozenInstanceError`. There is no
  serialization; default pickling works for picklable callables.

**Storage (`storage.py`).**

- Three module dicts behind one `threading.Lock`: name to entry, constant
  name to canonical identifier, and canonical identifier to constant.
- Functions and constants share one namespace.
- `register_native_constant` mints `Identifier(name)` outside the lock,
  then publishes the entry and its identifier under it, so no reader sees
  a constant without its identifier.
- Every read takes the lock:
  - `get_registered_entry(name)` returns the stored object, or raises
    `EntryLookupError("No entry is registered under the name 'x'.")`.
  - `get_registered_entries()` returns an `immutabledict` snapshot, in
    registration order.
  - `is_entry_registered(name)`.
  - `get_native_constant_identifier(name)` returns the identifier, or
    raises `EntryLookupError("No native constant is registered under the
    name 'x'.")`, also for a function's name.
  - `try_get_native_constant_for_identifier(identifier)` returns the
    constant, or `None`. It is by identity: an identifier merely named
    `pi` is an ordinary variable.
- **`set_registry_state_for_tests(state)`** replaces the whole registry
  with `state`. A constant keeps its identifier only if `state` holds the
  very object registered under its name. So a constant registered inside a
  test stops resolving once the snapshot is restored. The
  `function_registry_snapshot` fixture in `tests/conftest.py` snapshots
  with `dict(get_registered_entries())` and restores through it.
  Built-ins can be dropped this way; one test does
  (`test_core.py::test_the_screen_judges_a_builtin_call_by_the_builtin_catalogue`).

**Registration (`api.py`).**

- `register_function(name, parameters, parameter_sorts, result_sort,
  body)` builds the entry, wraps its `ValueError` as
  `EntryRegistrationError(str(exc))` (a `RuntimeError`), then inserts it.
  A taken name raises `EntryRegistrationError("A name is already
  registered: 'f'.")`.
- `register_native_function` and `register_native_constant` do the same.
- Registration never type-checks a body, and a body may call a name not
  yet registered.
- `try_get_registered_result_sort(name)` returns a function's result sort,
  or `None` for a constant or an unknown name.
- The errors: `EntryRegistrationError(RuntimeError)` and
  `EntryLookupError(KeyError)`, both `register_error`ed. The 14 Python
  tests that match their messages match only the entry's name.

**The built-ins (`builtins.py`).** Importing the module registers, in this
order:

1. the four constants `pi`, `e`, `inf`, `nan` (`math` values), so the
   composed bodies may name them. Their identifiers are the first ids the
   counter issues, 65,536 to 65,539, and two tests pin those ids, one in a
   fresh interpreter;
2. the 19 native functions, `math` callables plus `_exp2` and the builtin
   `round`, whose banker's rounding a test pins;
3. the 16 composed functions, each body built from Python-minted
   parameters.

`BUILTIN_FUNCTIONS` and `BUILTIN_CONSTANTS` are read-only `TypedDict`s
over `immutabledict`s of the registered entry objects.

**Consumers in `src`.** None mutates the registry; `builtins.py` is its
only writer.

| Module | Lines | Reads |
|---|--:|---|
| `passes/inline.py` | 169 | `FunctionInliner(RewritablePass)`: per call, `get_registered_entry`, dispatch by entry class. It inlines a `RegisteredFunction` by `body.substitute(dict(zip(parameters, arguments)))`, then transforms the result, with an in-progress set that raises `RecursionError`. It checks a native's arity and keeps the call, and refuses a constant with `FunctionArityError` (a `ValueError`, defined here) |
| `passes/evaluate.py` | 186 | per call, the entry: a `NativeFunction` with literal arguments is folded through `implementation`, its result checked against `result_sort` (`NativeResultSortError`); a `RegisteredFunction` gets a WARNING; constants are folded through `native_lowering.try_get_native_constant_value` |
| `passes/native_lowering.py` | 118 | `try_get_native_constant_for_identifier` |
| `passes/z3.py` | 727 | the entry kind, only to word the refusal of a call; the constant screen |
| `passes/sympy.py` | 1,738 | the entry kind for refusals; constants lowered by entry name (`_NATIVE_CONSTANT_LOWER`), lifted back through `get_native_constant_identifier` |
| `passes/numpy.py` | 728 | a native's `result_sort` for the cast; entry kinds for the unbound-identifier message; constants |
| `types/checking/type_checker.py` | 1,513 | `resolve_call_target` (injected, `get_registered_entry` by default): arity and parameter sorts, the result sort; constant types by identifier |
| `types/checking/body_type_checker.py` | 372 | `check_all_registered_function_bodies` iterates `get_registered_entries()` over the `RegisteredFunction`s, built-ins included, and resolves calls through `get_registered_entry` |
| `symbolic/solver.py` | 1,687 | `try_get_registered_result_sort` (numeric kind of a call), the constant screen |
| `constraint/core.py`, `constraint/system.py` | 1,117, 848 | `try_get_native_constant_for_identifier` |
| `param/core.py` | 2,175 | a parameter may not be a constant's identifier |

`types/checking/__init__.py` (53) and `symbolic/expression/__init__.py`
(204) re-export. The binding's `expression/screen.rs` (345 lines) is the
S4 adapter: it collects the Python identifiers of the trees screened, then
calls `try_get_native_constant_for_identifier` and
`try_get_registered_result_sort` once per identifier and per name.

**`RegisteredFunction` under binder frames.** `parameters` is declared
`compared_as_binder(scopes_over=("body",))`, and `name` is
`excluded_from_equivalence()`. The derived plan compares the sorts by
value and the parameter counts. Then it extends the Python `AlphaRenaming`
(a P1 value class, S4.3a) with the frame `dict(zip(self.parameters,
other.parameters))`, and compares the bodies through
`Expression.is_alpha_equivalent_under`. That converts the renaming, frames
included, into the Rust `AlphaRenaming` once per comparison, with the S4.2
capture rules. A non-injective frame is not equivalent. Structural
equivalence requires the same parameters. Two tests pin this, and nothing
in `src` compares entries.

**Probed at 0bc8952.**

- A lookup costs 230 to 300 ns: `get_registered_entry("max")` 229 ns,
  `try_get_native_constant_for_identifier` 268 to 275 ns, and
  `try_get_registered_result_sort("max")` 303 ns.
- **The inliner is exponential in nested calls whose body repeats a
  parameter.** `max`'s body names `a` twice, so `relu(relu(...(x)))`
  inlines in 0.2 ms at depth 4, 2.9 ms at 8, 11 ms at 10 and 44 ms at 12;
  depth 100 did not finish in five minutes. `RewritablePass` walks the
  substituted body per occurrence, so each level doubles the work,
  although the result shares the argument.

### Survey: the Rust API

- **`fhy_core::expression::builtins`** (`builtins.rs`, 695 lines) is the
  catalogue, data only: "it registers nothing, and it does not compute
  native functions".
  - `BuiltinFunction` has the 35 functions in catalogue order (16
    composed, then 19 native), with `iter`, `name`, `parameter_sorts`,
    `result_sort`, `composed`, `FromStr` and serde by name.
  - `ComposedFunction` has `function`, `parameters` and `body`. Its
    parameters are minted lazily, by a `LazyLock`, on the first call that
    asks for a composed function (R-2), and its body calls only
    `Callee::Builtin`s.
  - `BuiltinConstant` (`Pi`, `E`, `Inf`, `Nan`) has `iter`, `name`,
    `sort` (`Real` for all four) and `value() -> f64`, and **no
    identifier**.
- **`Callee`** is `Builtin(BuiltinFunction) | Named(FunctionName)`.
  `FromStr` routes a built-in's name to `Builtin`. `FunctionName::try_new`
  refuses `""` (`FunctionNameError::Empty`) and a built-in's name
  (`Builtin(f)`) (D-9). Constants are not callees, and their names are
  not reserved.
- **`SortLookup`** has `native_constant_sort(&Identifier)` and
  `call_result_sort(&FunctionName)`, both `None` by default;
  `NoRegisteredSorts` knows nothing. `BooleanScreen` judges
  `Callee::Builtin(f)` by `f.result_sort()` without asking.
- **`FunctionSort`** is the closed `Bool | Nat | Int | Real`, with no
  check of values.
- **There is no registry, no user-function type, no inliner and no
  evaluator.** `Expression::substitute` exists. The crate's process-global
  state is limited to identity (`lib.rs`, CONTRIBUTING "Process-global
  state is limited to identity"), and every registry but the intern
  registries is owned, as `PassRegistry` is (F-006).
- **Rust tests:** `tests/it/expression/builtins_stories.rs` (816 lines,
  26 tests), `screen_stories.rs` (1,486 lines, 62 tests; 15 mention
  `SortLookup`), and the callee tests in `builders_stories.rs` (1,005
  lines).

### Divergences visible from Python

| # | Python today | Rust core |
|---|---|---|
| X-1 | Built-ins are ordinary entries registered at import, in the namespace user entries share | a fixed catalogue that nothing registers; built-in function names are reserved (D-9) |
| X-2 | Each composed built-in has a Python body over Python-minted parameters | a second definition, `ComposedFunction`, over lazily minted parameters. The printed bodies agree; only the result sorts are checked to agree (S4.3a) |
| X-3 | Native built-ins carry `math` callables; `round` rounds half to even | no implementation |
| X-4 | A constant owns an identifier, minted at registration; the built-ins' are 65,536 to 65,539, by import order | `BuiltinConstant` has no identifier; `native_constant_sort` leaves identifiers to the caller |
| X-5 | Names are any non-empty `str`, one namespace for functions and constants | `Callee`/`FunctionName`: built-in names are not user names; constants have no name type |
| X-6 | `RegisteredFunction`, `NativeFunction`, `NativeConstant` | no user-entry type; `Callee::Named` is only a name |
| X-7 | Construction checks the name, the sort count and captured identifiers, the last against the global registry; a repeated parameter is accepted | no validation; S4.2's `enter_binder` docs tell a caller pairing parameter lists to refuse a repeated name |
| X-8 | A lookup returns the entry or raises `EntryLookupError`; entries in registration order | `SortLookup` returns only sorts, as `Option`s |
| X-9 | The screen asks the Python registry per identifier and per name (D-S4-4) | the screen asks a `SortLookup`; built-in calls come from the catalogue |
| X-10 | `FunctionInliner` rewrites per occurrence, recursively (exponential above), and raises `EntryLookupError`, `FunctionArityError` or `RecursionError` | no inliner |
| X-11 | The evaluator folds natives through their callables | no evaluator |
| X-12 | One global dict behind a lock, replaceable by `set_registry_state_for_tests` | owned, `Send + Sync` values; no test seam needed |
| X-13 | Dataclass `==`/`hash` over every field; derived binder equivalence without `name` | `ComposedFunction` has no equality |
| X-14 | Messages are sentences with periods and quotes (`A name is already registered: 'f'.`) | lowercase one-line messages (I.3 rule 3) |
| X-15 | `dataclasses.FrozenInstanceError` | the Rust-backed classes raise `FrozenMutationError` (S2 to S6) |
| X-16 | Registration order: constants, natives, composed | catalogue order: composed, then natives |
| X-17 | Constant values are `bool`, `int` or `float`, checked by `is_python_value_compatible_with_sort` | `LiteralValue` (with `Decimal`); `FunctionSort` checks no value |

Unchanged in meaning: the sort vocabulary, and a call's type from its
callee's declared sorts; the built-ins' names, signatures, bodies and
values; constants recognized by identifier identity; one namespace for
calls; registration that never type-checks a body and accepts forward
references.

### Consumers and tests

- **`src`.** The 13 modules in the consumer table above, plus
  `builtins.py`. All of them read through the public lookups. The type
  checker also takes an injected `resolve_call_target`.
- **Python tests.** 30 files mention the registry API. About 140 test
  requests use `function_registry_snapshot`: `test_registry.py` 49,
  `test_evaluator.py` 25, `test_type_checker_sorts.py` 17,
  `test_inline_pass.py` 14, `test_body_type_checker.py` 13,
  `test_registry_body_sweep.py` 10, and 1 to 3 in six more.

  | File | Lines | Tests | Registry references |
  |---|--:|--:|--:|
  | `symbolic/expression/test_registry.py` | 1,240 | 74 | 293 |
  | `symbolic/expression/test_builtins.py` | 627 | 169 | 44 |
  | `expression/passes/test_evaluator.py` | 651 | 33 | 49 |
  | `types/checking/test_type_checker_sorts.py` | 398 | 17 | 42 |
  | `types/checking/test_body_type_checker.py` | 324 | 13 | 39 |
  | `expression/passes/test_inline_pass.py` | 437 | 19 | 37 |
  | `types/checking/test_registry_body_sweep.py` | 338 | 10 | 29 |
  | `symbolic/expression/test_core.py` | 2,916 | 524 | 22 (3 registry tests) |
  | `expression/passes/test_numpy_evaluator.py` | 1,739 | 153 | 22 |
  | `symbolic/expression/test_sympy_natives.py` | 421 | 59 | 20 |
  | `constraint/test_constraint_system.py` | 3,382 | 233 | 13 |
  | `expression/passes/test_inline_pass_properties.py` | 363 | 4 | 9 |
  | `symbolic/expression/test_native_stories.py` | 293 | 16 | 6 |
  | `test_conftest_fixtures.py` | 105 | 12 | 6 |
  | `symbolic/expression/test_functions_stories.py` | 231 | 6 | 3 |
  | `types/checking/test_builtin_bodies.py` | 72 | 2 | 5 |
  | 14 more, 7 references or fewer each (the solver, the z3 and sympy passes, constraints, params, the type checker, `conftest.py`, the strategies) | | | 60 |

- **Rust tests:** as surveyed above; none covers a registry.
- **Benchmarks.** None covers the registry.
  `test_call_construction_of_a_user_function` and the three screen rows of
  `test_expression.py` touch it.

### Pattern choice

- **P2: the registry, `RegisteredFunction`, `NativeFunction` and
  `NativeConstant`.** The registry holds identity state (canonical
  constant identifiers), so decision 2's registry rule applies: one
  registry, in Rust. A Rust screen and a Rust inliner must hold the
  entries. The binding keeps one core `FunctionRegistry` in module state,
  next to a table from names to each entry's Python object (N-S7-2).
- **P1:** `FunctionSort` stays a `StrEnum`, and `Identifier` stays P1.
- **Plain Python:** the `RegisteredEntry` and `CallTargetResolver`
  aliases, `is_python_value_compatible_with_sort`, `builtins.py`'s table
  of native implementations, and every consumer except the inliner's walk.

**Benchmark plan: `benchmarks/test_registry.py` (S7.1).** Registration
mutates the process registry, so its rows restore a snapshot in
`pedantic` setup, through a helper `_restore(snapshot)`. The inliner rows
use `inline_functions`. The nested rows use depth 10, where today's
inliner takes about 11 ms; depth 100 does not finish today, and the
"after" run adds it with no baseline. The baseline measures today's
Python registry.

| Benchmark | Measures |
|---|---|
| `test_register_function[small]`, `[deep_body]` | registration of `x + 1`, and of a 100-operation body (the capture check walks it) |
| `test_register_native_function`, `test_register_native_constant` | the `inspect` arity check; minting the identifier |
| `test_registered_function_construction`, `_eq`, `_hash`, `_alpha_equivalence` | an entry value: construction, `==`, `hash`, and binder equivalence of two parameter-renamed functions |
| `test_get_registered_entry[user]`, `[builtin]`, `[miss]` | the lookup a pass makes per call; `[miss]` includes the `EntryLookupError` |
| `test_is_entry_registered`, `test_try_get_registered_result_sort[user]`, `[miss]` | the solver's and the screen's lookups |
| `test_try_get_native_constant_for_identifier[hit]`, `[miss]`, `test_get_native_constant_identifier` | the constant lookups the solver and the constraints make per identifier |
| `test_get_registered_entries` | the snapshot, with 50 user entries |
| `test_validate_predicate_of_user_calls` | the screen over a conjunction of 100 calls of five user functions and a constant reference: the adapter's cost (D-S7-6) |
| `test_inline_functions[no_calls]` | S4.1's deep tree, which calls nothing: the pass's floor |
| `test_inline_functions[nested_builtins]` | `relu` nested 10 deep: X-10 |
| `test_inline_functions[user_chain]` | ten user functions, each calling the next |
| `test_inline_functions[shared_dag]` | S4.1's doubling DAG with a `sigmoid` call at its leaf |
| `test_evaluate_after_inline` | `evaluate_expression(inline_functions(...))` of the user chain with literal arguments: the evaluator's lookup per call |
| `test_check_all_registered_function_bodies` | the sweep over the built-ins and 20 user functions |

Rerun, not added: `test_call_construction_of_a_user_function` and the
three `test_validate_*` rows (`test_expression.py`). A slower hot path
changes pattern, or is recorded as an accepted cost with numbers
(cross-cutting rule 5). The paths at risk:

- the lookups, which cross into the extension and take a lock where
  today they take a Python lock and index a dict. They must return the
  cached entry object, never build one;
- `try_get_native_constant_for_identifier`, which reads an `Identifier`'s
  id (P1) before the Rust lookup;
- registration of a small function, which now builds a Rust definition
  and converts its fields.

### Decisions (proposed 2026-09-25)

Each names the policy it follows:

- D-S4-1: Rust semantics where the two differ;
- D-S4-2: Python names where the meaning is the same;
- "no fallback";
- "tests rewritten, not skipped";
- the crate's conventions in `rust-workspace.md` Part I: owned registries
  and no global state beyond identity (F-006, CONTRIBUTING), D-9, R-2,
  and the layering of §I.2.

Where a decision follows an earlier slice's decision or note, it says so.

- **D-S7-1: one implementation, no fallback** ("no fallback"). The
  `registry/` modules become thin public classes and functions over
  `_rs`. The dataclasses, the three dicts and the lock are deleted, not
  kept behind a switch. `builtins.py` registers nothing any more.
- **D-S7-2: the core gains an owned `FunctionRegistry`** (crate
  conventions: an owned value, as `PassRegistry` is; I.3 rules 3 to 5
  and 7). It lives in a new `fhy_core::expression::registry` module,
  which imports no `pass` (§I.2):

  ```rust
  #[derive(Debug, Clone, Default)]            // Send + Sync; entries Arc-backed
  pub struct FunctionRegistry { /* registration order + indexes */ }
  impl FunctionRegistry {
      pub fn new() -> Self;
      pub fn register_function(&mut self, function: FunctionDefinition) -> Result<(), RegistrationError>;
      pub fn register_native_function(&mut self, function: NativeFunction) -> Result<(), RegistrationError>;
      /// Mints the constant's identifier and returns it.
      pub fn register_constant(&mut self, constant: NativeConstant) -> Result<Identifier, RegistrationError>;
      pub fn entry(&self, name: &str) -> Option<RegistryEntry<'_>>;
      pub fn contains(&self, name: &str) -> bool;
      pub fn iter(&self) -> impl ExactSizeIterator<Item = RegistryEntry<'_>> + '_;  // registration order
      pub fn constant_identifier(&self, name: &str) -> Option<&Identifier>;
      pub fn constant(&self, identifier: &Identifier) -> Option<&NativeConstant>;
      pub fn result_sort(&self, name: &FunctionName) -> Option<FunctionSort>;
      pub fn len(&self) -> usize;
      pub fn is_empty(&self) -> bool;
      pub fn inline(&self, expression: &Expression) -> Result<Expression, InlineError>;   // D-S7-7
  }
  impl SortLookup for FunctionRegistry;

  #[derive(Debug, Clone, Copy)]
  #[non_exhaustive]
  pub enum RegistryEntry<'r> { Function(&'r FunctionDefinition), Native(&'r NativeFunction), Constant(&'r NativeConstant, &'r Identifier) }

  #[derive(Debug, Clone)]                     // Arc inside; no PartialEq (as ComposedFunction)
  pub struct FunctionDefinition { /* name, parameters, parameter_sorts, result_sort, body */ }
  impl FunctionDefinition {
      pub fn try_new(name: FunctionName, parameters: impl IntoIterator<Item = Identifier>,
          parameter_sorts: impl IntoIterator<Item = FunctionSort>, result_sort: FunctionSort,
          body: Expression) -> Result<Self, FunctionDefinitionError>;
      // name, parameters, parameter_sorts, result_sort, body
  }
  pub struct NativeFunction { /* name, parameter_sorts, result_sort */ }   // new(...), infallible
  pub struct NativeConstant { /* name, sort, value: LiteralValue */ }       // try_new(...) -> Result<_, ConstantValueError>

  #[non_exhaustive] pub enum FunctionDefinitionError { SortCountMismatch { parameters: usize, sorts: usize }, RepeatedParameter(Identifier) }
  #[non_exhaustive] pub enum RegistrationError { NameTaken(FunctionName), BuiltinConstantName(BuiltinConstant), CapturedIdentifiers { function: FunctionName, identifiers: Vec<Identifier> } }
  #[non_exhaustive] pub struct ConstantValueError { /* sort, value */ }
  #[non_exhaustive] pub enum InlineError { UnknownFunction(FunctionName), ArityMismatch { callee: Callee, expected: usize, actual: usize }, NotCallable(FunctionName), Recursive(FunctionName) }
  ```

  - Every entry is keyed by a `FunctionName`, constants included, since
    they share the call namespace. `RegistryEntry` is a borrowed view, so
    a lookup copies nothing.
  - Cloning a registry is cheap: the entries are `Arc`s, and a clone is
    an independent registry.
  - Every `Display` is one lowercase line, for example `"f" is already
    registered` or `function "f" captures identifiers that are not its
    parameters: x, y`.
  - The core's names follow its own vocabulary (`FunctionDefinition`
    beside `ComposedFunction`), and the Python names stay (D-S7-9). The
    exact shape is settled test-first in S7.2.
- **D-S7-3: built-ins stay in the catalogue, and the registry holds only
  user entries** (D-9; X-1). The registry refuses a built-in function's
  name, through `FunctionName`. It also refuses a built-in constant's
  name (`BuiltinConstantName`), so the one namespace keeps its meaning
  (D-S4-2: Python refuses `"pi"` today, as a taken name). A built-in is
  never an entry of a `FunctionRegistry`. How Python sees the built-ins is
  N-S7-3.
- **D-S7-4: built-in constants get identifiers in the core, and the
  screen judges them by the catalogue** (D-9 extended to constants; X-4).
  - `BuiltinConstant::identifier(self) -> &'static Identifier` and
    `BuiltinConstant::of_identifier(&Identifier) -> Option<Self>` are
    added. Where the ids come from is N-S7-1.
  - `BooleanScreen` judges a built-in constant's identifier by
    `BuiltinConstant::sort()` without asking the `SortLookup`, as it
    judges built-in calls.
  - Registration exempts the four identifiers from the capture check.
- **D-S7-5: validation follows the owned registry** (D-S4-1; the crate's
  "no global state"; the S4.2 note on repeated names; X-7).
  - `FunctionDefinition::try_new` refuses mismatched sort counts and a
    repeated parameter (new).
  - Registration refuses a taken name, and a body capturing identifiers
    that are neither parameters, nor constants of *this* registry, nor
    built-in constants.
  - The capture check stays order-dependent, as today, but reads the
    registry it registers into. A value cannot see a registry, so direct
    construction of a `RegisteredFunction` no longer runs it.
  - `NativeConstant::try_new` checks the value with a new
    `FunctionSort::admits(&LiteralValue)`: `Bool` admits only a Boolean,
    `Nat` a non-negative integer, `Int` an integer, and `Real` an
    integer, a float or a decimal. For Python's `bool`, `int` and `float`
    these are `is_python_value_compatible_with_sort`'s rules, which a
    test pins.
- **D-S7-6: the screen reads the Rust registry, and the S4 adapter goes**
  (D-S4-4 retired; X-9). `validate_logical_operands` and
  `validate_predicate` screen with the registry snapshot of D-S7-13 as
  their `SortLookup`, with no Python call. `RegistrySorts`, its
  identifier-collection walk and its deferred lookup error are deleted.
  The environment and symbol-type conversions stay.
- **D-S7-7: inlining moves into the core; evaluation stays Python**
  (D-S4-1; X-10, X-11; D-S5-12 for the pass).
  - **`FunctionRegistry::inline`** is data-only: the registry, the
    catalogue's `ComposedFunction`s and `substitute`.
    - A call of a composed built-in, or of a user function, is replaced
      by its body over the inlined arguments, and the result is inlined.
    - A call of a native built-in or a native user function keeps its
      node, with its arity checked against the parameter sorts.
    - A call of a constant's name is `NotCallable`, an unknown name
      `UnknownFunction`, and a function reached again while its own body
      is being inlined `Recursive`.
    - It walks on its own stack and handles each distinct node once, so a
      tree of any depth works, and X-10's nested calls take linear time
      in the shared DAG. It returns the input itself when it inlines
      nothing.
  - **`FunctionInliner`** stays the registered Python pass
    `fhy_core.symbolic.expression.inline_functions`. It becomes a
    `CompilerPass[Expression, Expression]` whose `run_pass` calls the
    Rust inliner, as D-S5-12 did for `RewriteRuleApplier`.
    - It is no longer a `RewritablePass`, so `visit_call_expression` and
      `transform` go.
    - `did_change` is by identity.
    - The output is materialized beside the input, as S4.3a's
      `substitute` result is.
  - **The errors** are the core's text, under today's classes (D-S4-2):
    `UnknownFunction` raises `EntryLookupError`, `ArityMismatch` and
    `NotCallable` raise `FunctionArityError`, and `Recursive` raises
    `RecursionError`. A pass run wraps each in `PassExecutionError`, with
    the error as `__cause__`, as S6 does.
  - **Evaluation stays Python.** The core computes no native function
    (B3 §3.10), and Python's semantics are pinned: `round` rounds half to
    even, and results follow the platform's C `math`. `ExpressionEvaluator`
    stays a Python `RewritablePass` (N-S6-3 (a)) and folds through each
    entry's `implementation`. The 19 built-ins' `math` callables stay a
    table in `builtins.py`, which the binding reads once, at import.
- **D-S7-8: consumers keep reading through the lookups** (D-S4-2; I.8:
  types, constraints and params are not ported). No consumer in `src`
  changes, except `inline.py` (D-S7-7), `builtins.py` (D-S7-1, D-S7-7)
  and the registry package. The type checker's injected
  `resolve_call_target` and the body sweep keep their code.
- **D-S7-9: the three entry classes are P2, under their Python names and
  fields** (decision 2; D-S4-2; S3 to S5 practice; X-6, X-13, X-15).
  - `RegisteredFunction(name, parameters, parameter_sorts, result_sort,
    body)`, `NativeFunction(name, parameter_sorts, result_sort,
    implementation)` and `NativeConstant(name, sort, value)` keep their
    constructors.
  - Each keeps its field objects as struct members, so
    `entry.body is body` and `native.implementation is max` hold.
  - Each is a `#[pyclass(frozen)]` with a thin public subclass, and a
    virtual `FrozenMixin`. A mutation raises `FrozenMutationError`, an
    `AttributeError` like `FrozenInstanceError` (X-15).
  - `NativeFunction`'s `inspect` arity check stays Python: it is about
    Python callables, and Rust has none.
  - `==`, `hash` and `repr` follow the fields, as the dataclasses' did
    (S5's note on pattern equality). Pickles are a call of the class with
    its fields.
  - The binding checks argument types strictly: `TypeError` in S2's style
    (`RegisteredFunction body must be an Expression, got int.`), for a
    constant value that is not a `bool`, `int` or `float` too.
- **D-S7-10: binder equivalence is computed in the binding over the core's
  `AlphaRenaming`** (D-S4-3; the S4.3a conversion). `RegisteredFunction`
  implements `is_structurally_equivalent`, `is_alpha_equivalent` and
  `is_alpha_equivalent_under` itself, with the derived plan's meaning:
  - `name` is excluded;
  - the sorts must be equal, and the parameter counts too;
  - the bodies are compared under a frame pairing the parameters, which
    `enter_binder` refuses, and the comparison fails, when it is not
    injective;
  - structural equivalence requires the same parameters.

  A given Python renaming is converted once, as expressions convert it.
  The core gains no equality for definitions: `enter_binder` and
  `Expression::is_alpha_equivalent_under` are enough, and built-in
  entries, which are `ComposedFunction`s, compare the same way.
- **D-S7-11: the Python lookup API keeps its names, meaning and object
  identity** (D-S4-2; X-8). `register_function`,
  `register_native_function`, `register_native_constant`,
  `get_registered_entry`, `get_registered_entries`, `is_entry_registered`,
  `get_native_constant_identifier`,
  `try_get_native_constant_for_identifier` and
  `try_get_registered_result_sort` keep their signatures.
  - A registration returns the object that later lookups return, as
    today: the binding keeps each entry's Python object beside the Rust
    registry.
  - `get_registered_entries()` is an `immutabledict` in registration
    order.
  - Their view of the built-ins is N-S7-3.
- **D-S7-12: errors** (D-S4-1: the core's text under the Python classes;
  X-14).
  - A `RegistrationError`, a `FunctionDefinitionError` or a
    `ConstantValueError` met while registering raises
    `EntryRegistrationError` with the core's message.
  - At direct construction, the same errors raise `ValueError`, as the
    dataclasses did. An empty name raises the core's `FunctionNameError`
    as `ValueError`, as `CallExpression("")` does.
  - A lookup miss has no core error (the core returns `Option`), so
    `EntryLookupError` keeps Python's text. The 14 message-matching tests
    match names, which the core's texts keep.
- **D-S7-13: thread safety and snapshots** (S2's rule: no lock held
  across a call into Python; X-12).
  - The binding's state is an `Arc` of the registry and the object table
    behind a `Mutex`. Registration builds the new entry, with no lock
    held, then locks, clones the `Arc`'d state, inserts, and swaps it in
    (copy-on-write; registrations are rare).
  - A lookup locks only to clone the `Arc`. So the screen and the
    inliner run on one consistent snapshot, with no lock held while they
    run, even if another thread registers meanwhile.
- **D-S7-14: `set_registry_state_for_tests` keeps its meaning for user
  entries** (D-S4-2; X-12).
  - It rebuilds the user registry from the entries in `state`, whose
    values are the entry objects themselves.
  - A constant keeps its identifier only if `state` holds the object
    registered under its name, as today. Otherwise it is registered
    anew, with a new identifier.
  - The built-ins are no state (D-S7-3). A built-in name in `state` is
    ignored, and a missing one is not removed.
  - The `function_registry_snapshot` fixture is unchanged. Whether the
    seam stays at all is part of N-S7-2.
- **D-S7-15: the built-ins' bodies are the catalogue's** (D-9; X-2). The
  16 Python bodies in `builtins.py` are deleted, after S7.3's differential
  check shows each alpha-equivalent to its `ComposedFunction` under the
  frame pairing the parameters. The built-in entries' parameters are the
  catalogue's identifiers, so they now draw from the counter lazily, on
  the first use of any composed built-in (R-2), instead of at import.
  They are never serialized, so no id is pinned.
- **D-S7-16: order is registration order for user entries** (D-S4-1 for
  the built-ins, X-16). Built-ins, where a view lists them (N-S7-3), come
  in catalogue order: the constants, then the composed functions, then
  the natives. The sweep's diagnostics stay in the order it lists
  entries.
- **D-S7-17: tests are rewritten, not skipped** (the tests rule). The
  Python behavioral tests stay and are rewritten where a decision changes
  what they pin, each change recorded with its reason, as S4.4 to S6 did.
  The core's behavior is specified first by new Rust tests (S7.2), with a
  traceability table from the Python tests.

### Needs the user

- **N-S7-1: where the built-in constants' identifiers come from** (X-4,
  D-S7-4). They are serialized: an expression naming `pi` holds its
  identifier, and a payload read in another process resolves to the
  constant only if the id is the same there. Python pins 65,536 to
  65,539, which holds only while the constants take the counter's first
  ids at import. R-2 made the built-in *parameters* lazy because they are
  never serialized, which does not hold for constants. The policy does
  not cover this, since the core has no constant identifiers to follow.
  - (a) **Reserved ids.** Four entries in `identifier::reserved` and in
    `identifier.py`'s mirror of it, in a new block for expression
    constants, `48..64` (`pi` 48, `e` 49, `inf` 50, `nan` 51). They hold
    in every process and in Rust-only programs, whatever is used first,
    as D-S2-2 made the shipped tags hold. The two pinned-id tests pin the
    new ids. A payload that names a constant by 65,536 to 65,539 no
    longer resolves to it, as D-S2-2 moved the old pins.
  - (b) **Lazy, as R-2.** A `LazyLock` mints them on first use. The ids
    depend on what the process did first, so a serialized reference
    resolves only by chance. The pins are dropped.
  - (c) **Lazy, forced first at import.** Option (b), with the binding
    touching the four at initialization, so a Python process keeps 65,536
    to 65,539 while nothing draws an id before. A Rust-only program gets
    other ids.

  Recommendation: (a). It is the only option where a serialized constant
  reference means the same in every process, and it follows D-S2-2. It
  extends B1's reserved table, which R-2 had cut down to the shipped tags,
  so it is recorded as a revision of R-2, as D-S2-2 revised R-3.
- **N-S7-2: the binding's process-global registry** (X-12). CONTRIBUTING
  ("Process-global state is limited to identity") requires the
  maintainer's agreement for a new process-global static with interior
  mutability. This one also differs from every existing kind: it is not
  append-only, because `set_registry_state_for_tests` replaces it.
  - (a) **A replaceable static**: D-S7-13's `Mutex<Arc<_>>` in the
    binding, with `set_registry_state_for_tests` kept as the test seam
    (D-S7-14), recorded in CONTRIBUTING's section as the one registry the
    binding holds for the Python API.
  - (b) **An append-only static.** As D-S2-1 did for the intern
    registries, `set_registry_state_for_tests` raises
    `NotImplementedError`. The fixture goes, and each test registers
    under a unique name. About 140 fixture uses and the pruning test are
    rewritten.
  - (c) **A Python registry of P2 entries**, as N-S6-1 (a) kept the pass
    registry. Then no Rust registry is live, and the screen and the
    inliner build a core `FunctionRegistry` from the Python dict on every
    call. That costs linear time in the entries per call, and keeps two
    representations.

  Recommendation: (a). It is how CONTRIBUTING says a shared instance for
  the Python API should be held, "in the extension's module state". The
  core stays free of global state, and the seam keeps the tests
  isolated. (b) would make the tests depend on name hygiene, and (c)
  undoes D-S7-6's gain.
- **N-S7-3: whether the Python lookups see the built-ins** (X-1, D-S7-3).
  D-S4-1 and D-S4-2 point opposite ways, as they did in N-S5-1: in the
  core, a built-in is no registry entry, while in Python "registered"
  means "a call of this name resolves". Every consumer resolves both
  kinds through one `get_registered_entry` and dispatches by entry class.
  - (a) **A resolution view.** The lookups consult the catalogue, then
    the registry.
    - `BUILTIN_FUNCTIONS` and `BUILTIN_CONSTANTS` stay, holding entry
      objects the binding builds once, at import. A composed built-in is
      a `RegisteredFunction` over the catalogue's parameters and body
      (materialized once), a native built-in a `NativeFunction` with its
      `math` callable, and a constant a `NativeConstant`.
    - `get_registered_entries()` lists them first (D-S7-16).
    - Consumers and the sweep work unchanged. The built-ins cannot be
      removed, so the `test_core.py` test that drops `max` and `xor` is
      rewritten.
  - (b) **Registry only.** The lookups see user entries only, and
    `get_registered_entry("max")` raises `EntryLookupError`. A new
    `get_builtin_entry(name)` and `CallExpression.is_builtin` let
    consumers dispatch. The inliner, evaluator, type checker, the three
    bridges, the sweep and the constant helpers all change, and so do the
    `test_builtins.py` registration tests.
  - (c) (a) for the lookups, but `get_registered_entries()` lists user
    entries only, and the sweep adds the built-ins itself.

  Recommendation: (a). The meaning a consumer asks for, "what does this
  call resolve to", is unchanged, so D-S4-2 governs the Python surface,
  while the core keeps D-9 exactly. (b) moves the distinction into a
  dozen consumers that the design otherwise leaves untouched (D-S7-8).

### Steps

1. **S7.1: benchmarks.** Add `benchmarks/test_registry.py` as planned
   above, and record the baseline here, on today's Python registry.
2. **S7.2: core additions, test-first, with Rust tests.**
   - The `expression::registry` module of D-S7-2, with its errors.
   - `FunctionSort::admits`.
   - `BuiltinConstant::identifier` and `of_identifier`, with the ids
     N-S7-1 chooses (and, for (a), the reserved entries in Rust and in
     `identifier.py`, whose mirror test checks them).
   - The screen's catalogue rule for constants (D-S7-4).
   - `FunctionRegistry::inline`.

   The tests are written first and fail against `todo!()` stubs, as in
   S4.2. The crate README and `lib.rs`'s module table list the new
   module.
3. **S7.3: the binding.** Add `rust/fhy-core-py/src/expression/registry.rs`
   with these submodules:
   - `entries.rs`: the three pyclasses, each holding the Rust value, or
     for a built-in its catalogue item, and its field objects;
   - `state.rs`: the module state of D-S7-13, and the built-in entry
     objects under N-S7-3 (a);
   - `lookups.rs`: the lookup and registration functions,
     `set_registry_state_for_tests`, and `inline`.

   `screen.rs` switches to the snapshot (D-S7-6). Everything new goes
   into `_rs.pyi`. The step ends with the differential check of D-S7-15:
   a temporary test compares each `builtins.py` body with the catalogue's
   under the parameter frame, and its result is recorded here.
4. **S7.4: the Python switch.**
   - `registry/` becomes the thin layer, and `builtins.py` keeps only the
     `TypedDict`s and the table of native implementations.
   - `passes/inline.py` defines `FunctionInliner` over the Rust inliner.
   - `CONTRIBUTING`'s "Process-global state" section records the
     registry (N-S7-2), and the README's expression row changes
     (`FunctionInliner` is no longer a `RewritablePass`).
5. **S7.5: tests.** Migrate the tests and add the interface suite (the
   test plan below).
6. **S7.6: benchmarks after,** recorded here with the verdict, then the
   status, the implementation notes and this checklist.

Commit per step. Every step ends with `pytest` and `-m "not very_slow"`
green, the `property` session, `lint` and `type_check` clean,
`tests/test_rs_stub.py` green, and the Rust gate green (fmt, clippy `-D
warnings`, tests, doc `-D warnings`, deny, `cargo +1.85 check`).

### Test plan

**Rust tests, written first (S7.2).**

- **`tests/it/expression/registry_stories.rs`:**
  - registration of each kind, and lookups by name and by identifier;
  - registration order;
  - every `FunctionDefinitionError`, `RegistrationError` and
    `ConstantValueError` variant, with its `Display`;
  - built-in names refused;
  - the capture check's order dependence and its exemption of built-in
    constants;
  - a constant's minted identifier, and one named like it that is not
    it;
  - `result_sort` for each kind;
  - the `SortLookup` answers;
  - a clone's independence;
  - `Send + Sync`, by a compile-time assertion.
- **`tests/it/expression/inline_stories.rs`:**
  - user functions, composed built-ins, and nested and chained calls;
  - a native call kept, with its arity checked;
  - a constant's name as a callee;
  - an unknown name;
  - arity both ways;
  - self- and mutual recursion;
  - the input handle itself back when nothing is inlined (`ptr_eq`);
  - a shared DAG inlined once per distinct node, by a counting check on
    the output's distinct nodes;
  - X-10's nested `relu` at depth 1,000 in linear time;
  - a 100,000-level tree on a small stack.
- **`tests/it/expression/registry_properties.rs`:**
  - inlining leaves no call of a composed built-in or a user function;
  - inlining is idempotent;
  - an inlined call and the original agree under a reference evaluation
    of the Boolean built-ins, as the Python property does.
- **`builtins_stories.rs` and `screen_stories.rs` gain:**
  - the constants' identifiers: stable, distinct, `of_identifier` as the
    inverse, and with (a) the reserved ids;
  - `FunctionSort::admits`'s table;
  - the screen judging a built-in constant without a lookup.

  A traceability table maps the Python registry, inline and story tests to
  them, as S4.2's did.

**The interface suite:
`tests/symbolic/expression/test_registry_rust_binding.py`**, next to
S4.3a's and S5's, covers what the binding adds over the core:

- **Class structure.**
  - Each entry class extends its `_rs` class, is a virtual `FrozenMixin`
    and raises `FrozenMutationError`.
  - `__match_args__`, reprs and pickles are covered.
  - `tests/test_rs_stub.py` covers the stubs.
- **Arguments.** The `TypeError`s, and the `ValueError`s of direct
  construction with the core's texts.
- **Identity.** A registration returns the object later lookups return.
  The fields are the objects given. Under N-S7-3 (a), a built-in's entry
  is one object across lookups and `BUILTIN_FUNCTIONS`, and its body is
  built once.
- **Constants.** Identity lookups; pruning on restore; the pinned
  identifiers of N-S7-1; and the agreement of the binding's value check
  (the core's `FunctionSort::admits`) with
  `is_python_value_compatible_with_sort` over `bool`, `int`, `float`,
  negatives and `bool`-as-`int`.
- **Binder equivalence.** The name is excluded, a renamed parameter list
  is equivalent, a non-injective pairing is not, and a given free
  renaming is honored.
- **The screen.** It calls no Python: the test replaces the Python lookup
  functions with ones that raise, and the screen still judges user calls
  and constants.
- **The inliner.**
  - The error classes and `__cause__` through `PassExecutionError`.
  - `did_change` is by identity, and the input object comes back when
    nothing is inlined.
  - X-10's depth-100 nesting finishes.
  - The pass is registered, and `CompilerPass.create` builds it.
- **Threads.** Concurrent registration of distinct names, and of one
  name, where exactly one wins. A lookup and a screen during
  registrations see a consistent snapshot.
- **The built-ins' bodies.** Each printed body is pinned as data, since
  D-S7-15 deletes the Python bodies that S7.3 compared.

**Migrating the existing tests.** No test is skipped, or deleted without
a rewrite:

- **`test_registry.py` (74).**
  - The three frozen tests pin `FrozenMutationError` (D-S7-9).
  - `test_registered_function_direct_construction_rejects_captured_identifier`
    becomes a registration test (D-S7-5), and a new test pins that direct
    construction refuses a repeated parameter.
  - The pinned-id tests follow N-S7-1.
  - The snapshot and pruning tests keep their meaning (D-S7-14, pending
    N-S7-2).
  - The message tests keep matching names under the core's texts
    (D-S7-12).
- **`test_builtins.py` (169)** keeps its meaning under N-S7-3 (a). The
  parameter and body tests read the catalogue's, and the `round` and
  `exp2` tests keep pinning Python's `math` semantics (D-S7-7).
- **`test_core.py`.**
  - The test that drops `max` and `xor` pins that built-ins cannot be
    removed (D-S7-3, D-S7-14).
  - The catalogue-agreement test pins that each built-in entry's sorts
    are the catalogue's.
- **`test_inline_pass.py` (19) and its properties (4)** keep their
  meaning. The error tests pin the core's texts and their classes as
  `__cause__` (D-S7-7), and the properties also run at depths the old
  inliner could not reach.
- **The evaluator, type checker, body checker, sweep, solver, constraint,
  param and bridge tests** change only where they pin a message or order
  that D-S7-12 or D-S7-16 changes.
- **Each rename or rewrite** is recorded here with its reason, as in S4.4
  to S6.


### S7 resolutions (decided by the user, 2026-09-25)

- **N-S7-1: (a) reserved ids.** `pi`, `e`, `inf` and `nan` get the fixed
  ids 48, 49, 50 and 51 in a new block for expression constants in
  `identifier::reserved`, mirrored in `identifier.py`. This revises R-2,
  as D-S2-2 revised R-3.
- **N-S7-2: (a) a replaceable static.** The binding holds one
  `Mutex<Arc<FunctionRegistry>>` in the extension's module state for
  Python's global API, with `set_registry_state_for_tests` kept as the test
  seam. This is the maintainer's agreement CONTRIBUTING requires, and it is
  recorded there.
- **N-S7-3: (a) a resolution view.** The Python lookups consult the
  catalogue first, then the registry. `BUILTIN_FUNCTIONS` and
  `BUILTIN_CONSTANTS` hold entry objects built once, at import, and the
  consumers are unchanged.

### S7.1 baseline (2026-09-25, f051be9 plus the new benchmarks)

`benchmarks/test_registry.py` implements the benchmark plan above.

- **Registration rows** restore the snapshot before each of their 2,000
  rounds (`pedantic` with `_restore` as setup), so each round registers
  into the same registry. The `Benchmark` protocol in
  `benchmarks/conftest.py` gained `pedantic`.
- **The other rows** that register user entries restore the snapshot
  after the benchmark, through a `snapshot` fixture.
- **The screened conjunction** joins 100 calls of five Boolean user
  functions and a reference to a Boolean user constant.
- **The user chain** is ten functions, each adding one to a call of the
  next. The evaluator row uses a chain of ten functions each flooring a
  call of the next, called with a literal, so the evaluator folds one
  native call per level after inlining; the evaluator folds no
  arithmetic, so the first chain would leave nothing to fold.

Median time per call, from `.nox/benchmark-3-11/bin/python -m pytest
benchmarks -k "test_registry or call_construction_of_a_user or
test_validate" -n 0 --benchmark-only`, the benchmark session's
environment, measuring today's pure-Python registry and inliner. The
machine is the S0 one, with Python 3.11.13 and pytest-benchmark 5.3.0.
The load average was about 0.6, and the table lists the best of three
runs' medians.

| Benchmark | before |
|---|--:|
| `test_register_function[small]` | 3.38 µs |
| `test_register_function[deep_body]` | 12.9 µs |
| `test_register_native_function` | 70.8 µs |
| `test_register_native_constant` | 5.65 µs |
| `test_registered_function_construction` | 2.56 µs |
| `test_registered_function_eq` | 514 ns |
| `test_registered_function_hash` | 319 ns |
| `test_registered_function_alpha_equivalence` | 50.9 µs |
| `test_get_registered_entry[user]` | 247 ns |
| `test_get_registered_entry[builtin]` | 249 ns |
| `test_get_registered_entry[miss]` | 670 ns |
| `test_is_entry_registered` | 245 ns |
| `test_try_get_registered_result_sort[user]` | 324 ns |
| `test_try_get_registered_result_sort[miss]` | 570 ns |
| `test_try_get_native_constant_for_identifier[hit]` | 295 ns |
| `test_try_get_native_constant_for_identifier[miss]` | 295 ns |
| `test_get_native_constant_identifier` | 247 ns |
| `test_get_registered_entries` | 1.11 µs |
| `test_validate_predicate_of_user_calls` | 31.7 µs |
| `test_inline_functions[no_calls]` | 188.5 µs |
| `test_inline_functions[nested_builtins]` | 11.2 ms |
| `test_inline_functions[user_chain]` | 110.0 µs |
| `test_inline_functions[shared_dag]` | 17.4 ms |
| `test_evaluate_after_inline` | 166.2 µs |
| `test_check_all_registered_function_bodies` | 7.18 ms |
| `test_call_construction_of_a_user_function` (rerun) | 1.42 µs |
| `test_validate_logical_operands_of_deep_conjunction` (rerun) | 72.8 µs |
| `test_validate_predicate_of_nested_piecewise` (rerun) | 16.7 µs |
| `test_validate_predicate_of_comparison` (rerun) | 1.48 µs |

- **Lookups** cost 245 to 325 ns, a miss of `get_registered_entry` 670 ns
  with its `EntryLookupError`.
- **Registration** of a small function takes 3.4 µs, of a 100-operation
  body 12.9 µs (the capture check walks it), and of a native function
  71 µs, almost all of it `inspect.signature`.
- **The inliner** is where the registry is slow. Even the tree with no
  call takes 189 µs, the Python walk over 191 nodes. `relu` nested ten deep
  takes 11.2 ms, doubling per level (X-10), and the DAG with one `sigmoid`
  call at its leaf 17.4 ms, since the walk rewrites each of its 2,047
  occurrences.
- **The binder equivalence** of two functions takes 51 µs, most of it the
  derived plan in Python around one Rust comparison of the bodies.
- **The sweep** over the 16 composed built-ins and 20 user functions takes
  7.2 ms.

### S7.2 implementation notes

The tests were written first, against `todo!()` stubs of every checking
constructor, lookup, `inline`, and the constant identifiers: 133 of the
156 new or filtered tests failed (the rest pin pure data, such as a
`Display` of a hand-built error, or are older tests the filter matched),
and all pass now. The additions are the new module
`rust/fhy-core/src/expression/registry.rs` with `registry/definition.rs`,
`registry/error.rs` and `registry/inline.rs`; `BuiltinConstant::identifier`
and `of_identifier` in `builtins.rs`; the screen's constant rule in
`screen.rs`; and four entries of `identifier/reserved.rs`, mirrored in
`identifier.py`. `lib.rs` and the crate README list the module.

Where the shape differs from D-S7-2's sketch, or fills it in:

- **`FunctionSort::admits` is the existing `FunctionSort::accepts_literal`.**
  The core already had it, in `literal.rs`, with exactly D-S7-5's table
  (a Boolean only `bool`, a non-negative integer `nat`, an integer `int`
  and `real`, a float or a decimal only `real`), and tests in
  `literal_stories.rs`. A second method of the same meaning would give
  one rule two names, so `NativeConstant::try_new` calls it, and
  `native_constant_accepts_exactly_the_values_of_its_sort` pins the table
  again through the constant, `bool`-as-`int`, negatives, big integers,
  NaN, infinities and decimals included.
- **The errors name the entry.** D-S7-12 relies on the core's texts
  keeping the names the Python message tests match, so
  `FunctionDefinitionError`'s variants are struct variants with the
  function's name (`SortCountMismatch { function, parameters, sorts }`,
  `RepeatedParameter { function, parameter }`), and `ConstantValueError`
  holds the constant's name beside its sort and value, with accessors.
  The texts: `function "f" has 1 parameter but 2 parameter sorts`,
  `function "f" repeats the parameter "x"`, `constant "c" of sort nat
  cannot hold -1`, `"f" is already registered`, `"pi" is the name of a
  built-in constant`, `function "f" captures identifiers that are not its
  parameters: x, y` (by name hint, then by id), `no function is registered
  under "f"`, `"f" takes 2 arguments but the call passes 1`, `"c" is a
  constant, not a function`, `function "f" is recursive and cannot be
  inlined`.
- **`InlineError::Piecewise(PiecewiseError)` is new.** A body using a
  parameter as a piecewise condition, called with a numeric literal, puts
  that literal in the condition, which `substitute` refuses; Python raised
  the same refusal as a `ValueError` from `substitute`. It displays
  `inlining built an invalid piecewise` with the piecewise's error as its
  source.
- **`FunctionRegistry::retain(keep)` is new**, for S7.3's
  `set_registry_state_for_tests` (D-S7-14): it keeps the chosen entries in
  their order, each constant with its identifier, and checks nothing
  again. Registering a constant always mints a new identifier, so without
  it the seam could not restore a snapshot's constants with their
  identifiers. It is the owned-value counterpart of `Vec::retain`; the
  core still has no global state.
- **`RegistryEntry::name()`** returns an entry's name whatever its kind.
- **The entry types.** `FunctionDefinition` is one `Arc` of its fields,
  `NativeFunction` holds its sorts in an `Arc<[FunctionSort]>`, and both
  are cheap to clone; `NativeFunction` derives `PartialEq`, `Eq` and
  `Hash`, `NativeConstant` `PartialEq` (its value may be a float).
  `FunctionDefinition` has no equality, as D-S7-2 planned.
- **Lookups by `&str`.** `entry`, `contains` and `constant_identifier`
  take a `&str`, so a caller can ask about any text, a built-in's name
  included, which finds nothing; `result_sort` takes a `FunctionName`,
  as `SortLookup::call_result_sort` does. `SortLookup` answers for the
  registry's own constants and functions only: the screen judges the
  built-in constants itself.
- **The screen's constant rule** (D-S7-4) is one private helper,
  `find_constant_sort`: a built-in constant's identifier has its
  catalogue sort, and any other identifier is asked of the lookup, both
  for judging an operand and for ignoring a constant's binding.
  `BooleanScreen`'s and `SortLookup`'s rustdoc say so.
- **The constant identifiers** are built once, by a `LazyLock` over the
  reserved entries `PI_CONSTANT` (48), `E_CONSTANT` (49), `INF_CONSTANT`
  (50) and `NAN_CONSTANT` (51), which draws nothing from the counter.
  `of_identifier` compares ids, so an identifier restored with id 48 and
  any name hint is `pi`, and one merely named `pi` is not.
- **The inliner** (`registry/inline.rs`) is a post-order walk on an
  explicit stack. It remembers each node's result by the node's identity,
  holding the node so that no other node can take its address during the
  walk, and each new result as its own result, since a result has nothing
  left to inline. A substituted body holds the arguments' results, so the
  walk meets them as remembered nodes: `relu` nested 1,000 deep inlines
  into at most five new nodes per level, and a 64-level doubling DAG with
  a `sigmoid` call at its leaf into 64 additions over one inlined
  `sigmoid`. The user functions whose bodies are being walked are a set,
  entered when a body is pushed and left when its result is taken, which
  the stack order makes exactly the functions of the current path; a
  remembered result never hides a recursion, since a node whose inlining
  reaches a function in progress never finished its first inlining. The
  order of checks follows Python's: arguments first, then recursion, then
  the lookup, then the arity.
- **Tests.** `registry_stories.rs` (72 tests, counting `rstest` cases),
  `inline_stories.rs` (36) and `registry_properties.rs` (3 properties)
  are new; `builtins_stories.rs` gained 8 and `screen_stories.rs` 19. A
  proptest regression file written while the stubs failed was deleted:
  its seeds were stub failures, not findings.

Traceability of the Python tests (`test_registry.py` is `G`,
`test_inline_pass.py` `I`, `test_inline_pass_properties.py` `P`):

| Python tests | Rust tests | Note |
|---|---|---|
| G `test_register_function_stores_name_parameters_and_body`, `..._records_parameter_sorts_and_result_sort`, `..._returns_a_registered_function_instance`, `..._with_multiple_parameters_records_order` | `function_definition_keeps_its_fields`, `registered_function_is_found_by_name` | |
| G `test_register_function_accepts_self_recursive_body`, `..._accepts_body_calling_an_unregistered_name`, `..._accepts_a_body_whose_forward_reference_is_incompatible`, `test_registered_function_direct_construction_accepts_self_recursive_call` | `registration_accepts_a_body_calling_an_unregistered_or_recursive_name` | a call is a reference by name |
| G `test_register_function_rejects_duplicate_name`, `test_register_native_function_rejects_duplicate_name`, `..._rejects_collision_with_registered_function`, `test_register_native_constant_rejects_duplicate_name`, `..._rejects_collision_with_function` | `registration_refuses_a_taken_name_whatever_the_kinds` (6 cases), `refused_registration_leaves_the_registry_unchanged` | |
| G `test_register_function_rejects_captured_free_identifier`, `..._lists_multiple_captured_identifiers_in_sorted_order`, `test_registered_function_direct_construction_rejects_captured_identifier` | `registration_refuses_a_body_capturing_free_identifiers`, `captured_identifiers_sharing_a_name_hint_are_ordered_by_id` | D-S7-5: a registration check |
| G `test_register_function_accepts_subset_of_parameters_used_in_body`, `..._accepts_literal_only_body` | `capture_check_exempts_the_builtin_constants`, `registry_answers_the_screens_sort_lookup` (a literal body) | |
| G `test_register_function_rejects_sort_arity_mismatch` | `function_definition_refuses_a_sort_count_other_than_its_parameter_count` (3 cases) | |
| none | `function_definition_refuses_a_repeated_parameter`, `function_definition_accepts_distinct_parameters_sharing_a_name_hint` | D-S7-5, new |
| G `test_register_function_accepts_body_referencing_registered_constant`, `..._rejects_body_identifier_merely_named_like_a_constant` | `capture_check_reads_the_constants_registered_so_far`, `capture_check_refuses_a_look_alike_of_a_builtin_constant` | the order dependence, per registry |
| G `test_get_registered_entry_*`, `test_is_entry_registered_*`, `test_get_registered_entries_includes_registered_entry`, `..._snapshot_includes_all_entry_kinds` | `registered_function_is_found_by_name`, `registered_native_function_is_found_by_name`, `registered_constant_is_found_by_name_and_by_its_minted_identifier`, `new_registry_is_empty`, `entries_iterate_in_registration_order` | |
| G `test_get_registered_entries_returns_immutable_snapshot`, `test_function_registry_snapshot_restores_state_after_test_a`/`_b` | `registry_clone_is_independent`, `registry_clone_keeps_the_constants_identifiers` | an owned value needs no test seam (X-12); the seam is the binding's |
| G `test_restoring_a_registry_snapshot_drops_identifiers_it_does_not_carry` | `retain_keeps_the_chosen_entries_in_order_with_their_identifiers`, `constant_registered_again_after_retain_gets_a_new_identifier` | |
| G `test_registered_functions_with_separately_built_equal_bodies_are_equal`, `..._swapping_their_parameters_are_alpha_equivalent`, the `NativeFunction`/`NativeConstant` equality tests | none | entry equality is the binding's (D-S7-9, D-S7-10); the core compares bodies with `is_alpha_equivalent_under` |
| G the three `..._dataclass_is_frozen` tests | none | Rust values are immutable |
| G `test_native_function_constructs_with_declared_sorts_and_implementation`, `test_register_native_function_stores_supplied_fields`, `..._returns_native_function_instance` | `native_function_keeps_its_signature`, `registered_native_function_is_found_by_name` | the core holds no implementation |
| G `test_native_function_direct_construction_rejects_arity_mismatch`, `..._accepts_signature_less_callable`, `test_register_native_function_rejects_arity_mismatch` | none | the `inspect` check stays Python (D-S7-9) |
| G `test_native_constant_constructs_with_name_sort_and_value`, `test_register_native_constant_stores_supplied_fields`, `..._returns_native_constant_instance`, `..._rejects_sort_value_incompatibility`, `..._rejects_bool_for_int_sort` | `native_constant_accepts_exactly_the_values_of_its_sort` (19 cases), `constant_value_error_displays_the_constant_its_sort_and_the_value` (3) | |
| G `test_register_native_constant_mints_an_identifier_for_the_new_constant`, `test_try_get_native_constant_for_identifier_rejects_a_same_named_identifier`, `..._rejects_a_function_named_identifier`, `test_get_native_constant_identifier_raises_for_an_unregistered_name`, `..._for_a_function_name` | `registered_constant_is_found_by_name_and_by_its_minted_identifier`, `identifier_merely_named_like_a_constant_is_no_reference_to_it`, `each_registered_constant_gets_a_new_identifier`, `constant_identifier_is_none_for_a_function_or_an_unknown_name` | |
| G `test_try_get_registered_result_sort_*` (4) | `result_sort_answers_for_functions_only`, `registry_answers_the_screens_sort_lookup` | |
| G `test_pi_is_registered_at_import_time`, `test_pi_lookup_returns_a_native_constant`, `test_builtin_constants_mapping_covers_seeded_constants`, `test_get_native_constant_identifier_is_stable_across_calls`, `test_each_seeded_constant_owns_a_distinct_identifier`, `test_builtin_constants_keep_their_pinned_canonical_ids` (and its fresh-interpreter twin) | `builtin_constant_identifier_holds_its_reserved_id_and_name` (4 cases), `builtin_constant_identifier_is_one_value_across_calls_and_threads`, `builtin_constant_identifiers_are_distinct`, `builtin_constant_of_identifier_inverts_identifier`, `builtin_constant_of_identifier_goes_by_the_id_not_the_name`, `builtin_names_find_no_entry` | N-S7-1 (a): the reserved ids; built-ins are no entries (D-S7-3) |
| I `test_inline_functions_returns_literal_unchanged`, `..._returns_identifier_unchanged`, `..._traverses_binary_expression_without_calls`, `..._preserves_piecewise_expression_structurally`, `..._does_not_modify_input_expression` | `inline_returns_the_input_itself_when_nothing_is_inlined`, `inline_shares_every_subtree_without_a_call_to_inline`, `inline_returns_a_deep_tree_without_calls_itself_on_a_small_stack` | the input itself, by `ptr_eq` |
| I `test_inline_functions_substitutes_call_with_registered_body`, `..._substitutes_parameter_in_piecewise_body`, `..._inlines_call_nested_inside_arithmetic` | `inline_replaces_a_user_call_by_its_body_over_the_arguments`, `inline_substitutes_the_argument_object_at_every_use`, `inline_replaces_a_composed_builtin_call_by_the_catalogue_body` | |
| I `test_inline_functions_recursively_inlines_nested_calls`, `..._inlines_call_that_references_another_registered_function`, `..._result_contains_no_call_expression` | `inline_expands_nested_calls_inside_out`, `inline_follows_a_chain_of_user_functions`, `inline_expands_a_user_body_calling_a_composed_builtin`, `inline_uses_a_function_registered_after_its_caller`, `inline_leaves_no_composed_call_of_a_composed_builtin` (5 cases) | |
| I `test_inline_functions_raises_for_unknown_function_name` | `inline_refuses_a_call_of_an_unknown_name`, `inline_checks_the_arguments_before_the_call_taking_them` | |
| I `test_inline_functions_raises_when_argument_count_exceeds_parameters`, `..._is_too_few`, `..._rejects_wrong_arity_to_native_function` | `inline_refuses_a_user_call_of_the_wrong_arity` (3 cases), `inline_refuses_a_builtin_call_of_the_wrong_arity` (4), `arity_mismatch_displays_both_counts` (2) | |
| I `test_inline_functions_raises_for_recursive_function`, `..._for_mutually_recursive_functions` | `inline_refuses_a_self_recursive_function`, `inline_refuses_mutually_recursive_functions_naming_the_first_reached_again`, `inline_accepts_a_function_called_twice_but_not_inside_itself` | |
| I `test_inline_functions_rejects_call_to_native_constant` | `inline_refuses_a_call_of_a_constant` | |
| I `test_inline_functions_passes_through_native_function_call_unchanged` | `inline_keeps_a_native_user_call`, `inline_keeps_a_native_builtin_call_and_inlines_its_arguments` | |
| none | `inline_refuses_a_substitution_that_breaks_a_piecewise_condition`, `inline_expands_a_shared_call_once_per_distinct_node`, `inline_of_nested_composed_calls_takes_linear_time`, `inline_walks_a_deep_tree_on_a_small_stack` | new: X-10, sharing and depth |
| P `test_inline_functions_leaves_no_registered_function_call`, `..._is_idempotent` | `inline_leaves_no_call_of_a_composed_builtin_or_a_user_function`, `inline_is_idempotent` (by `ptr_eq`) | |
| P `test_inline_functions_evaluates_like_the_reference_table_for_real_builtins`, `..._for_bool_builtins` | `inline_keeps_the_reference_meaning_of_the_calls` | one generator of Boolean trees over numeric comparisons, both kinds of built-ins and three user functions |
| none | `boolean_screen_judges_a_builtin_constant_by_the_catalogue` (16 cases), `boolean_screen_refuses_a_builtin_constant_predicate_root_without_a_lookup`, `boolean_screen_reads_a_builtin_constant_by_its_sort_not_a_binding`, `boolean_screen_accepts_a_builtin_constant_in_a_numeric_position` | D-S7-4 |

### S7.3 status and the differential check

The binding is `rust/fhy-core-py/src/expression/registry.rs` with
`registry/entries.rs` (the three pyclasses), `registry/state.rs` (the
module state and the built-in entries) and `registry/lookups.rs` (the
functions), exported from `_rs` and declared in `_rs.pyi`. Nothing in
Python uses it yet, so the suite is unchanged (7,163 passed).

**`screen.rs` switches in S7.4, not here.** Until the Python modules
register through the binding, the Rust registry holds no user entry, so a
screen reading it would pass numeric user calls that the Python registry
knows, and the suite would fail between the two commits. The switch moves
with the Python switch, so each step stays green.

**The differential check of D-S7-15** ran as a temporary script, kept
outside the repo: it registered temporary public classes, installed the
built-ins from `builtins.py`'s `_NATIVE_FUNCTION_SPECS`, and compared each
of the 35 Python built-in entries with the binding's:

- the 16 composed bodies are alpha-equivalent to the catalogue's under the
  frame pairing the parameters, whose name hints agree in order, and print
  the same text: `max` `{a if (a > b); b otherwise}`, `min`, `abs`
  `{x if (x >= 0); (-x) otherwise}`, `sign`, `clamp` `min(max(x, lo),
  hi)`, `clamp_symmetric`, `relu` `max(x, 0)`, `leaky_relu`, `xor`
  `((a || b) && (!(a && b)))`, `nand`, `nor`, `implies`, `iff`, `sigmoid`
  `(1 / (1 + exp((-x))))`, `silu` and `gelu` `((0.5 * x) * (1 + erf((x /
  sqrt(2)))))`;
- all 35 have the same parameter and result sorts, and the 19 native
  entries hold the very callables of `builtins.py`;
- the four constants have the same sorts and values, and their
  identifiers the reserved ids 48 to 51 with their names.

So the Python bodies can be deleted (S7.4), and the interface suite pins
the printed bodies as data.

### S7.4 and S7.5 status

The Python switch (b4ca537, marked breaking) left 7 tests failing and
`test_builtins.py` failing collection; the test commit after it migrates
them and adds the interface suite. One failure was a core bug the
migration found: a call of a built-in constant's name, such as `pi()`,
was an unknown function to the inliner, where Python called it not
callable. The one call namespace of D-S7-3 means a built-in constant
is refused as `NotCallable` as a registered one is (52fad7e, with its
Rust test). `test_numpy_evaluator.py::test_raises_when_calling_native_constant`
pins it unchanged.

At the end of S7.5: `pytest` 7,327 passed, `-m "not very_slow"` 7,360
passed, the `property` session 281 passed, `lint` and `type_check` clean,
`tests/test_rs_stub.py` green, and the Rust gate green.

Tests migrated in S7.5. None was skipped or deleted; every other test,
the message tests of `test_registry.py` included, passes unchanged.

| Test | Now | Reason |
|---|---|---|
| `test_registry.py::test_registered_function_dataclass_is_frozen`, `test_native_function_dataclass_is_frozen`, `test_native_constant_dataclass_is_frozen` | same names | D-S7-9: `FrozenMutationError` |
| `test_registry.py::test_registered_function_direct_construction_rejects_captured_identifier` | `test_registered_function_direct_construction_leaves_captures_to_registration` | D-S7-5: an entry built directly holds the body; registering it refuses the capture, naming it |
| none | `test_registry.py::test_registered_function_direct_construction_rejects_a_repeated_parameter` | D-S7-5, new |
| `test_registry.py::test_builtin_constants_keep_their_pinned_canonical_ids`, and its fresh-interpreter twin | same names | N-S7-1 (a): the ids 48 to 51, and the name hints |
| `test_builtins.py::test_builtin_parameter_sort_table_is_a_tuple` (4 cases) | `test_builtin_entry_parameter_sorts_are_a_tuple` (4 cases) | D-S7-15: the private sort tables are gone, and the entries' sorts are the catalogue's |
| `test_builtins.py::test_builtin_native_functions_table_item_assignment_raises_type_error` | `test_builtin_native_implementations_table_item_assignment_raises_type_error` | D-S7-7: the native callables are the table `_NATIVE_IMPLEMENTATIONS` |
| `test_core.py::test_the_screen_judges_a_builtin_call_by_the_builtin_catalogue` | same name | D-S7-3, D-S7-14: restoring a state without `max` and `xor` keeps both |
| `test_inline_pass.py`: the unknown-name, the two arity, the two recursion, the constant and the native-arity tests | same names | D-S7-7, D-S7-12: they also pin the core's text of the cause |
| `test_inline_pass.py::test_inline_functions_passes_through_native_function_call_unchanged` | same name | D-S7-7: the input comes back itself |
| `test_inline_pass_properties.py::test_inline_functions_is_idempotent` | same name | D-S7-7: the second inlining returns its input itself, not a structurally equal tree |
| none | `test_inline_pass_properties.py::test_inline_functions_leaves_no_registered_function_call_in_deep_nesting` | X-10: nests of 20 to 300 calls, beyond the Python inliner's reach |
| `tests/conftest.py::function_registry_snapshot` | docstring | the built-ins are no state |

The new `tests/symbolic/expression/test_registry_rust_binding.py` (162
tests, counting parametrized cases) covers the test plan above: the class
structure, frozen errors, `__match_args__`, reprs and pickles (a built-in
unpickles as itself); the argument checks and the core's `ValueError`
texts, the Python arity check, the registration errors and their causes,
and the Python text of lookup misses; identity across registration, the
lookups, `BUILTIN_FUNCTIONS` and the snapshot, whose order it pins; the
built-in constants' identifiers, resolution by id, pruning and keeping on
restore, and the value check against `is_python_value_compatible_with_sort`
over 11 values and the four sorts; binder equivalence; the screen with the
Python lookups replaced by raising ones; the inliner's pass, its identity
result, its errors as causes, and `relu` nested 100 deep; threads (distinct
names, one contested name, and whole snapshots while another thread
registers); and the 16 printed bodies.

### S7 status

S7 was implemented on 2026-09-25 in eight commits: the benchmarks and
their baseline (b131516); the core additions, test-first (3bda92b); the
binding (9c34e3b); the Python switch, marked breaking (b4ca537); the
inliner's refusal of a call of a built-in constant, which the migration
found (52fad7e); the migrated tests and the interface suite (bca3227);
the entries' equality in Rust, which the benchmarks called for
(1c6805c); and these docs, with the depth-100 benchmark. No test was
skipped or deleted. At the end: `pytest` 7,327 passed, `-m "not
very_slow"` 7,360 passed, the `property` session 281 passed, `lint` and
`type_check` clean, `tests/test_rs_stub.py` green, `FORCE_COLOR=1 nox -s
tests-3.13` green, and the Rust gate green (fmt, clippy `-D warnings`,
2,808 tests, doc `-D warnings`, deny, `cargo +1.85 check`).

### S7 benchmarks (before and after)

Median time per call, from the benchmark session's environment, `pytest
benchmarks -k "test_registry or call_construction_of_a_user or
test_validate" -n 0 --benchmark-only`, on the S0 machine with Python
3.11.13 and pytest-benchmark 5.3.0. "Before" is b131516, the S7.1
baseline's tree, exported with `git archive` under `target/` and built
there; "after" is the S7.6 tree. The two ran three times each,
interleaved (before, then after, in each round), with a load average of
3 to 5, and the table lists the best of the three medians. The "before"
column agrees with the S7.1 table within 7%.

| Benchmark | before | after | after / before |
|---|--:|--:|--:|
| `test_register_function[small]` | 3.56 µs | 3.09 µs | 0.87 |
| `test_register_function[deep_body]` | 13.3 µs | 9.19 µs | 0.69 |
| `test_register_native_function` | 72.6 µs | 77.4 µs | 1.07 |
| `test_register_native_constant` | 5.94 µs | 1.25 µs | 0.21 |
| `test_registered_function_construction` | 2.54 µs | 1.09 µs | 0.43 |
| `test_registered_function_eq` | 531 ns | 284 ns | 0.53 |
| `test_registered_function_hash` | 330 ns | 273 ns | 0.83 |
| `test_registered_function_alpha_equivalence` | 52.1 µs | 856 ns | 0.02 |
| `test_get_registered_entry[user]` | 261 ns | 120 ns | 0.46 |
| `test_get_registered_entry[builtin]` | 261 ns | 86 ns | 0.33 |
| `test_get_registered_entry[miss]` | 658 ns | 552 ns | 0.84 |
| `test_is_entry_registered` | 239 ns | 119 ns | 0.50 |
| `test_try_get_registered_result_sort[user]` | 331 ns | 179 ns | 0.54 |
| `test_try_get_registered_result_sort[miss]` | 569 ns | 167 ns | 0.29 |
| `test_try_get_native_constant_for_identifier[hit]` | 297 ns | 119 ns | 0.40 |
| `test_try_get_native_constant_for_identifier[miss]` | 296 ns | 138 ns | 0.47 |
| `test_get_native_constant_identifier` | 247 ns | 73 ns | 0.30 |
| `test_get_registered_entries` | 1.04 µs | 150 ns | 0.14 |
| `test_validate_predicate_of_user_calls` | 32.0 µs | 18.2 µs | 0.57 |
| `test_inline_functions[no_calls]` | 189.6 µs | 18.5 µs | 0.10 |
| `test_inline_functions[nested_builtins]` (relu 10 deep) | 11.4 ms | 46.9 µs | 0.004 |
| `test_inline_functions[user_chain]` | 111.5 µs | 30.3 µs | 0.27 |
| `test_inline_functions[shared_dag]` | 17.9 ms | 22.5 µs | 0.001 |
| `test_inline_functions_of_builtins_nested_a_hundred_deep` (relu 100 deep) | did not finish in 5 min (probe) | 367.0 µs | - |
| `test_evaluate_after_inline` | 174.0 µs | 62.7 µs | 0.36 |
| `test_check_all_registered_function_bodies` | 7.33 ms | 7.22 ms | 0.98 |
| `test_call_construction_of_a_user_function` (rerun) | 1.45 µs | 1.57 µs | 1.08 |
| `test_validate_logical_operands_of_deep_conjunction` (rerun) | 72.8 µs | 16.8 µs | 0.23 |
| `test_validate_predicate_of_nested_piecewise` (rerun) | 16.8 µs | 11.1 µs | 0.66 |
| `test_validate_predicate_of_comparison` (rerun) | 1.50 µs | 497 ns | 0.33 |

Every hot path is faster or within the 10% CONTRIBUTING allows, so P2
stands and no cost needs the maintainer (cross-cutting rule 5):

- **The lookups** the passes make per call, which the plan put at risk,
  are 1.2 to 3.5 times faster: one call into the extension, a lock held
  only to read the current state, and the cached entry object returned.
  The snapshot is 7 times faster, since it is built once per state. A
  miss of `get_registered_entry` still pays for its `EntryLookupError`.
- **The inliner** is where S7 pays off: X-10's nested `relu` goes from
  11.4 ms at depth 10, doubling per level, to 47 µs, and depth 100, which
  the Python inliner did not finish, takes 0.37 ms, linear in the depth;
  the DAG with a `sigmoid` call at its leaf drops from 17.9 ms to 23 µs,
  since each distinct node is inlined once. A tree with no call costs the
  pass's floor, 18 µs, most of it the lifecycle and the walk's memo; the
  Python walk took 190 µs. The evaluator after inlining is 2.8 times
  faster.
- **The screen** no longer calls Python: the user-call conjunction is 1.8
  times faster, and the S4.1 screen rows 1.5 to 4.3 times, since they
  lost the identifier-collection walk of the old adapter.
- **Entry values.** Construction is 2.3 times faster; the binder
  equivalence of two functions 61 times, one Rust comparison instead of
  the derived plan; `==` 1.9 times, after 1c6805c (the first run
  measured 1.12 with the field tuples, see the notes), and `hash` 1.2.
- **Registration** of a function or a constant is faster. A native
  function takes 1.07 times as long: `inspect.signature` still dominates
  it, and the binding adds a call through the public class, the Python
  arity check called from Rust, and the state's clone. It is within the
  10%, and registrations are rare.
- **Unchanged rows.** The body sweep runs the Python type checker, which
  S7 does not change. `test_call_construction_of_a_user_function`'s code
  does not change either; it measured 0.98 in the first set of rounds.

### S7 implementation notes

Choices the decisions left open, made while implementing S7.3 to S7.6,
and where the shape differs from the plan (S7.2's are in its own notes
above):

- **Module layout.** `rust/fhy-core-py/src/expression/registry.rs`, with
  `entries.rs` (the three pyclasses and the `FunctionSort` conversion),
  `state.rs` (the module state and the built-in entries) and `lookups.rs`
  (the `_rs` functions). The materializer gained `materialize_beside`,
  for the inliner's output, and `materialize_expression`, for the
  built-ins' bodies. `screen.rs` lost `RegistrySorts`, its
  identifier-collection walk and its deferred lookup error, and reads
  `snapshot().registry()` as its `SortLookup`.
- **The Python functions are the extension's.** `storage.py` and
  `api.py` re-export `_rs.get_registered_entry` and the other eight
  functions directly, so a lookup is one call into the extension; their
  documentation is in the Rust doc comments, the stub and the modules'
  docstrings.
- **The module state** (N-S7-2 (a), D-S7-13) is a `Mutex<Arc<_>>` of the
  core registry and each entry's Python object, by name and, for a
  constant, by its identifier's id. A registration builds the entry
  object by calling the public class, with no lock held, then under the
  lock clones the state, registers the object's Rust value into the
  clone and swaps it in; the old state is dropped after the lock is
  released, since dropping it may run Python finalizers. A lookup locks
  only to read the current `Arc`.
- **A constant's Python identifier is built on first request**, in a cell
  shared by every later state that keeps the constant, so it is one
  object. Building it calls Python (`Identifier.deserialize_from_dict`),
  which the registration's critical section must not; the core mints the
  Rust identifier there.
- **`get_registered_entries` returns one `immutabledict` per state**,
  built on the first request and returned again until the next
  registration: the built-ins in catalogue order (the constants, the
  composed functions, the native functions, D-S7-16), then the user
  entries in registration order. The Python order was constants, natives,
  composed functions.
- **The built-in entries** are built once, when `builtins.py` calls
  `_rs.NativeFunction._install_builtins(_NATIVE_IMPLEMENTATIONS)` at
  import, a class method since the stub test admits no private
  module-level function. So the catalogue's composed parameters are
  minted at import, when the entries' bodies are materialized, not on
  their first use as D-S7-15 put it: N-S7-3 (a) builds the entries at
  import. A fresh process now issues its first identifier after the 27
  catalogue parameters (65,563), where it used to follow the four
  constants and 27 Python parameters. A lookup before the installation
  raises `RuntimeError`; the package installs them on import.
  `builtins.py` imports `core` first, whose classes the bodies are built
  from.
- **A built-in's entry** holds its catalogue item instead of a Rust
  value: `ComposedFunction`, or just its name for a native function or
  a constant. It is built through a private seed, as S2's tags are, and
  pickles as `Class._builtin(name)`, so it unpickles as itself. A
  built-in constant's `value` is the `float` of the core's
  `BuiltinConstant::value()`, equal to `math.pi` and the others (and a
  NaN for `nan`), not the `math` objects themselves.
- **Equality, hashing and `repr`** follow the fields, as the frozen
  dataclasses did, with `NotImplemented` against another class. A native
  function or a constant compares and hashes its field tuple through
  Python, since its implementation or value is a Python object. A
  function entry compares its Rust values instead (the name, the
  parameters' ids, the sorts and the bodies' structural equality, which
  is what the field objects' `==` computes), and hashes them with the
  body object's cached hash: the first benchmark run measured the
  tuple comparison at 1.12 times the dataclass's `==` (493 against
  440 ns), and the Rust one takes 0.54 times. The classes are not
  `@final`, as the dataclasses were not, and set `__match_args__`.
- **Binder equivalence** (D-S7-10) reads the Rust parameters and bodies,
  the catalogue's for a built-in, and compares with `type(self) is
  type(other)`, as the derived plan did. With repeated parameters refused
  (D-S7-5), a frame pairing two parameter lists is always injective, so
  the plan's "non-injective pairing" case cannot arise any more; the
  binding still reads a refused frame as "not equivalent".
- **Registration errors.** A `ValueError` from building the entry is
  raised as `EntryRegistrationError` with its text and the `ValueError`
  as its `__cause__`, as Python did; a `TypeError` passes through; a
  refusal of the core registry raises `EntryRegistrationError` with the
  core's text. A lookup miss keeps Python's text (D-S7-12); the
  inliner's unknown name raises `EntryLookupError` with the core's.
- **Stricter arguments**, beyond D-S7-9: a sort must be a `FunctionSort`
  member (a bare `"real"` is refused), a parameter an `Identifier`, a
  missing constructor argument raises `TypeError` naming it, and a
  built-in function's name is refused at direct construction, not only
  at registration. A built-in constant's name is accepted at direct
  construction and refused at registration.
- **`set_registry_state_for_tests`** keeps each entry whose very object
  the state holds under its name, in its place and, for a constant, with
  its identifier; every other entry of the state is registered anew after
  them, in the state's order. Python replaced the dict with the state's
  order exactly; the fixture's states only drop later entries, where the
  two agree. Built-in names are ignored, a value that is no user entry
  raises `TypeError`, and a failure leaves the registry unchanged.
- **`FunctionInliner`** is a `CompilerPass[Expression, Expression]` with
  the hooks `run_pass`, `get_noop_output` (its input) and `did_change`
  (identity), as D-S5-12's applier; `inline_functions` still runs it, so
  its errors are still `PassExecutionError`s with the cause. A
  `RecursionError` carries the core's text; an invalid piecewise raises
  `ValueError` with the core's text and the piecewise refusal after it.
- **The mock identifiers of the tests** cannot take the ids 48 to 51
  any more, since `mock_identifier` refuses a constant's id; no test
  used them.

## Plan after S7 (the user, 2026-09-25)

1. **S8: the solver, ported to Rust with pluggable backends.** An SMT
   backend lowers to SMT-LIB2 in pure Rust, shipped in every build. z3 is
   optional: `z3-solver` in Python, or a `z3` cargo feature in Rust.
2. **A CAS backend for simplification.** Research the most widely used
   computer algebra system that can be driven from Rust, with a license
   that allows it as an optional dependency, and build a Rust backend for
   it.
3. **The sympy adapter stays,** but only for Python.
4. **S9: the numpy evaluator in Rust.** It uses the `numpy` crate with
   `ndarray`; numpy is imported lazily, so it stays an optional extra.
5. **Symbolica is not used.** It is source-available, and its license
   restricts distribution.
