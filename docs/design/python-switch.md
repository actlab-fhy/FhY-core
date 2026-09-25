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
