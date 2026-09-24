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
