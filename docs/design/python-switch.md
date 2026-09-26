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
- [x] S8: the solver and its backends (N-S8-1 resolved as (a), N-S8-2 as (b)). The suite is green (7,389 passed), slow tests pass (7,422), properties pass (282), `tests_minimal` passes (5,764 passed, 608 skipped), lint and mypy are clean, `tests-3.13` passes with `FORCE_COLOR=1`, and the Rust gate passes (3,040; 3,072 with the `z3` feature)
  - [x] N-S8-1 decided as (a), N-S8-2 as (b)
  - [x] S8.1: solver benchmarks and baseline (14 rows; see "S8.1 baseline")
  - [x] S8.2: core additions, test-first, with Rust tests (`fhy_core::solver`: the screens, the SMT-LIB2 lowering, the backend traits, the facade, the process backend). 224 new tests; the Rust gate passes (3,040); see "S8.2 implementation notes"
  - [x] S8.3: the `z3` cargo feature and its backend, with the CI changes. Built against the z3-solver wheel's libz3 4.16 (D-S8-18's fallback; the local libz3 4.8.7 is below z3-sys's 4.13.3); 3,072 Rust tests with the feature; see "S8.3 status"
  - [x] S8.4: the solver binding (the P3 bases and adapters, `Solver`, `SatResult`, the stubs). The suite is unchanged (7,327 passed); see "S8.4 status"
  - [x] S8.5: the Python switch (the z3-solver and sympy adapters, lazy imports). 6,228 passed; exactly the four migrated modules (`test_solver.py`, `test_z3_pass.py`, `test_sympy_pass.py`, `test_cross_cutting.py`) fail collection until S8.6
  - [x] S8.6: tests migrated, and the interface suite (57). The suite is green (7,387 passed), `-m "not very_slow"` 7,420, properties 281; see "S8.5 and S8.6 status"
  - [x] S8.7: optional extras, backend markers and the minimal-install session. `tests_minimal` passes (5,764 passed, 608 skipped); the suite 7,389 passed; see "S8.7 status"
  - [x] S8.8: benchmarks after, and docs (every solver row faster, or within 10%, after c0af152; see "S8 benchmarks")
- [x] S10: terms (N-S10-1 resolved as (a), N-S10-2 as (b)). Rebased onto S8 (7078ca2): the suite is green (7,438 passed), slow tests pass (7,471), properties pass (282), `tests_minimal` passes (5,813 passed, 608 skipped), lint and mypy are clean, and the Rust gate passes (3,104; 3,136 with the `z3` feature)
  - [x] N-S10-1 decided as (a), N-S10-2 as (b)
  - [x] S10.1: term benchmarks and baseline (32 rows; see "S10.1 baseline")
  - [x] S10.2: core additions, test-first, with Rust tests (`fhy_core::term`: `AlphaRenaming` moved there with shared frames, `Hash`, `extended` and `enter_binders`; the `AlphaEquivalence`, `FreeIdentifiers`, `Term` and `Binder` traits; the mapping comparison)
  - [x] S10.3: the term binding (`AlphaRenaming`, the `Binder` adapter, the derived-equivalence engine and its roles, the mapping helper, the stubs)
  - [x] S10.4: the Python switch
  - [x] S10.5: tests migrated, and the interface suite
  - [x] S10.6: benchmarks after, and docs
- [x] S9: the expression evaluators (N-S9-1 resolved as (a), N-S9-2 as (b)). Rebased onto S8 and S10 (c93f76f): the suite is green (7,556 passed), slow tests pass (7,589), properties pass (282), `tests_minimal` passes (5,646 passed, 626 skipped), lint and mypy are clean, and the Rust gate passes (3,271; 3,303 with all features)
  - [x] N-S9-1 decided as (a), N-S9-2 as (b) (2026-09-26; see "S9 resolutions")
  - [x] S9.1: evaluator benchmarks and baseline (26 rows; see "S9.1 baseline")
  - [x] S9.2: core additions, test-first, with Rust tests (`fhy_core::expression::evaluate`: the values, the kernels, the walk, the fold; `Decimal::to_f64_exact`)
  - [x] S9.3: the `ndarray` cargo feature and the array backend
  - [x] S9.4: the evaluator binding (the rust-numpy conversions, the fold's adapter, the built-in implementations, the stubs)
  - [x] S9.5: the Python switch
  - [x] S9.6: tests migrated, and the interface suite
  - [x] S9.7: after the rebase onto S8: the `numpy` marker, `tests_minimal` without NumPy, and the README. `tests_minimal` passes (5,646 passed, 626 skipped)
  - [x] S9.8: benchmarks after, and docs (every row faster or within 10% except `float32` arrays, 1.62, an accepted cost, accepted by the maintainer; see "S9 benchmarks")
- [ ] S12: the Rust SymPy simplifier backend (designed 2026-09-26; see "S12: the Rust SymPy simplifier backend")
  - [x] N-S12-1 decided as (a) (2026-09-26; see "S12 resolutions")
  - [x] S12.1: SymPy benchmarks and baseline, on today's Python adapter (12 rows; see "S12.1 baseline")
  - [x] S12.2: the simplify context carries the function registry (core, test-first; `SimplifyContext::from_registry`, `Solver::simplify` taking the context)
  - [x] S12.3: the `sympy` cargo feature and `SympySimplifier`, with its stories and the CI changes (144 new tests; see "S12.2 and S12.3 status")
  - [x] S12.4: the binding enables the feature (`_rs.SympySimplifier`, the error mapping, the stubs; see "S12.4 status")
  - [ ] S12.5: the Python switch (the thin `passes/sympy.py`, the default solver)
  - [ ] S12.6: tests migrated, and the interface suite
  - [ ] S12.7: benchmarks after, and docs

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
   it. The research (2026-09-25) ranked SymPy first: it is the most used
   open-source CAS by far (BSD-3) and the only candidate that covers
   floor, mod, min, max, piecewise, Booleans and assumptions. The user
   chose **SymPy through pyo3**: a Rust `Simplifier` behind an
   off-by-default `sympy` cargo feature of `fhy-core` (pyo3 optional,
   without `auto-initialize`), which needs Python with SymPy at run time.
   The Python binding can then use the same Rust backend, leaving one
   copy of the mapping. Runners-up were SymEngine (MIT C++, fast, but
   dead Rust bindings, no symbolic Mod and a weak simplify) and egg or
   egglog (rewriting, not a CAS).
3. **The sympy adapter stays,** but only for Python.
4. **S9: the numpy evaluator in Rust.** It uses the `numpy` crate with
   `ndarray`; numpy is imported lazily, so it stays an optional extra.
5. **Symbolica is not used.** It is source-available, and its license
   restricts distribution.

## S8: the solver and its backends

- **Status:** designed 2026-09-25 at 17a5502, with D-S8-1 to D-S8-20
  applying the policy the user already set and the direction in "Plan
  after S7"; N-S8-1 and N-S8-2 were decided by the user as (a) and (b).
  Implemented 2026-09-26 in ten commits; see "S8 status".
- **Pattern:** the query logic moves into a new core module,
  `fhy_core::solver`. Its two backend traits, one for SMT solving and one
  for simplification, are P3: Python implementations are driven through
  `Py<PyAny>` adapters, and Rust implementations are registered with the
  Python ABCs. The facade and the result values are P2. The sympy
  integration stays Python, as a P3 simplification backend.
- **Scope.** This covers `symbolic/solver.py`, the z3 bridge
  (`symbolic/expression/passes/z3.py`), the sympy bridge's role as the
  simplifier (`passes/sympy.py`), and making `sympy` and `z3-solver`
  optional. The Rust CAS backend is the next slice (Plan after S7, item 2),
  and the numpy evaluator is S9.

### Survey: the Python API

**`src/fhy_core/symbolic/solver.py` (1,687 lines)** is pure Python over
the Rust-backed expressions. Its module docstring (83 lines) is its
contract. It exports eleven names:

| Name | Kind | Meaning |
|---|---|---|
| `SolverBackend` | `StrEnum` | `SYMPY = "sympy"`, `Z3 = "z3"` |
| `SolverQueryKind` | `StrEnum` | `SIMPLIFICATION`, `SATISFIABILITY`, `IMPLICATION`, `UNIVERSAL_VALIDITY` |
| `SolverCapabilityError` | `ValueError`, `register_error`ed | the backend cannot answer the query kind: `Backend ... cannot answer ... queries; it supports [...].`, with both enums' `repr`s |
| `get_backend_capabilities(backend)` | function | reads the private `_BACKEND_CAPABILITIES` table: SYMPY answers SIMPLIFICATION, Z3 the other three; an unknown backend has none |
| `validate_timeout_milliseconds(t)` | function | `None` or a positive strict `int` (`bool` and `float` refused), else `ValueError("timeout_milliseconds must be None or a positive integer, but got ...")`. Public so `ConstraintSystem` can hold up the precondition on paths that never reach the solver |
| `simplify_expression(expression, environment=None, *, backend=SYMPY)` | query | the sympy bridge's `simplify_expression`; no timeout |
| `check_expression_satisfiability(expression, symbol_types, *, backend=Z3, timeout_milliseconds=None)` | query, `bool \| None` | encoded as `not does_expression_imply(expression, False)` on the bridge |
| `does_expression_imply(antecedent, consequent, symbol_types, *, ...)` | query, `bool \| None` | the bridge's implication |
| `holds_for_all_free_assignments(considered_identifiers, expression, symbol_types, *, ...)` | query, `bool \| None` | `forall free. exists considered. expression` |
| `assert_expression_implies`, `assert_holds_for_all_free_assignments` | query, `bool` | the strict companions: `UndecidableError` instead of `None` |

- **Query kinds.** Four kinds, each answered by exactly one backend. Every
  query function starts with the capability check and then dispatches
  without looking at `backend`, under an `INVARIANT` comment saying that a
  second capable backend needs real dispatch. The Z3 questions are
  tri-state: `True`, `False`, or `None` for Z3's `unknown` or a screened
  expression. The strict companions raise `UndecidableError(message,
  reason=...)` instead, whose `reason` is Z3's `reason_unknown()` (such
  as `"timeout"`) or the fixed marker `"hazard_screen"`.
- **The order of checks** in each Z3 question, which the tests pin:
  1. the capability check, then `validate_timeout_milliseconds`;
  2. `symbol_types` covers every free identifier of every expression, and
     every considered identifier, except a registered native constant's
     canonical identifier (`KeyError("symbol_types is missing entries for
     identifiers: [...]")`, sorted by id);
  3. `validate_predicate` of each expression, in order (the Rust
     `BooleanScreen` since S4.3a): an ill-typed expression raises
     `NonBooleanLogicalOperandError`, ahead of any hazard;
  4. the hazard screen of each expression, in order, stopping at the
     first hazard. Each expression is screened on its own, never the
     combined formula.
- **The hazard screen** (about 1,000 lines) refuses five shapes the z3
  bridge mis-lowers, in this order. The first found is logged at WARNING
  on `fhy_core.symbolic.solver`, naming the entry point and the refused
  constants or node (and, for the last three, the identifier sorts at
  that node), and ending "bounding timeout_milliseconds cannot change
  this outcome":
  1. **Native constants:** a reference to a registered native constant's
     canonical identifier (`pi`, `e`, `inf`, `nan`, or a user constant),
     since Z3 has no term for one and a variable would let the solver
     choose it;
  2. **Non-finite literals:** a `float` infinity or NaN anywhere, since
     `float.as_integer_ratio` has no rational for it;
  3. **Boolean coercion:** arithmetic with a Boolean operand, a comparison
     mixing a Boolean and a number, or a piecewise whose branches mix
     them, since the z3 Python API rewrites a Boolean in a numeric context
     to `If(b, 1, 0)`;
  4. **Partial operations:** `DIVIDE` unless the divisor is a finite
     nonzero literal and one operand is provably real (Z3 truncates two
     ints); `FLOOR_DIVIDE` and `MODULO` unless the divisor is a finite
     positive literal (Z3 is Euclidean); `POWER` unless the exponent is an
     integer literal of at least one;
  5. **Mixed int/real equality:** an `EQUAL` or `NOT_EQUAL` with a numeric
     literal on one side and, on the other, an operand whose evaluated
     int/real kind differs or cannot be determined, since Z3's `ToReal`
     collapses the package's type-strict `1` against `1.0`.

  The screen reads two classifications off the tree: the Z3 sort a node
  lowers to (`_LoweredSort`: Boolean, numeric or undetermined), mirroring
  the converter, and the int/real kind the IR evaluates to
  (`_classify_operand_numeric_kind`), which reads a call's result sort
  through `try_get_registered_result_sort`. Every walk is recursive
  Python.
- **Simplification.** `simplify_expression` checks the capability and
  calls the sympy bridge. It has no timeout; a signature test pins that.
- **Caching:** none. Every query builds a fresh `z3.Solver`, and nothing
  is memoized across calls.
- **Timeouts** are passed to `z3.Solver.set(timeout=...)` unchanged.
- **Errors.** `SolverCapabilityError`, `KeyError`,
  `NonBooleanLogicalOperandError`, `ValueError` for a bad timeout, and
  `RuntimeError` for an unexpected Z3 result. A lowering failure inside a
  bridge pass (a call, or a `Z3Exception`) arrives as
  `PassExecutionError` with the cause, since both bridges are
  `VisitablePass`es.

**The z3 bridge (`passes/z3.py`, 727 lines).** It imports `z3` at module
level.

- `ExpressionToZ3Converter(symbol_types)`, a `VisitablePass` registered
  as `fhy_core.symbolic.expression.to_z3`, lowers through the z3 Python
  API's operators:
  - identifiers become `z3.Int`, `z3.Real` or `z3.Bool` named
    `<name_hint>_<id>`, recorded in `identifier_to_z3_expression`;
  - a `bool` literal becomes `BoolVal`, an `int` `IntVal`, and a `float`
    or `Decimal` the exact `RatVal` of its value;
  - `DIVIDE` is `/`, which is integer division on two ints;
    `FLOOR_DIVIDE` is `_z3_floor_divide` (`ToInt` of a real quotient,
    Euclidean `div` on ints); `MODULO` is Euclidean `%`; `POWER` is `**`;
  - `LogicalExpression` is n-ary `z3.And` or `z3.Or`, a piecewise a
    right-folded `z3.If`;
  - every call is refused with a `TypeError`, a user function asking for
    `inline_functions` first.
- `convert_expression_to_z3_expression(expression, symbol_types=None) ->
  (z3 expression, identifier map)` checks the symbol types, screens with
  `validate_logical_operands`, refuses native constants
  (`NativeConstantLoweringError`), then runs the converter. The package
  re-exports it from `fhy_core.symbolic.expression`.
- `holds_for_all_free_assignments`, `does_expression_imply` and their
  strict `assert_*` companions are the bridge's own questions, without the
  hazard screen. The universal check asserts `ForAll(considered, Not(e))`
  (or `Not(e)` when nothing is considered) and reads `unsat` as `True`.
  The implication runs it over `a && !c` with every identifier
  considered, so it asserts a closed formula quantified over every
  identifier it mentions. `unknown`
  is logged at WARNING with `reason_unknown()`.

**The sympy bridge (`passes/sympy.py`, 1,738 lines).** It imports `sympy`
at module level, and defines a `sympy.Piecewise` subclass there.

- Three registered passes: `ExpressionToSympyConverter` (`to_sympy`),
  `SympyVariableSubstitutionPass` and `SymPyToExpressionConverter`
  (`from_sympy`), plus tables that lower and lift the native functions
  and constants.
- Public functions: `convert_expression_to_sympy_expression`,
  `substitute_sympy_expression_variables` (simultaneous, refusing a bound
  native constant), `convert_sympy_expression_to_expression` (exact
  rationals, `oo` and `nan` to the constants, `zoo` refused), and
  `simplify_expression(expression, environment=None)`. That last one
  screens with `validate_logical_operands(expression, environment)`,
  refuses an environment binding a referenced constant
  (`NativeConstantBindingError`), lowers, substitutes, simplifies, and
  lifts. Simplification is best-effort: `PrecisionExhausted`, a dropped
  otherwise branch and an unfoldable relational each keep the substituted
  but unsimplified form.
- The package re-exports the first three functions.

**Who uses the bridges directly, not through the solver.**

- **In `src`:** only `solver.py`, which imports four z3 functions and the
  sympy `simplify_expression`, and the re-exports in
  `symbolic/expression/__init__.py`. Every other module goes through the
  solver.
- **In tests:** `test_z3_pass.py` (the converter, `_z3_floor_divide`,
  17 calls of the unscreened questions), `test_sympy_pass.py` (the three
  passes, the private tables, one call of the bridge's
  `simplify_expression`), `test_cross_cutting.py` (the pass registrations
  and both lowerings of one literal), and `test_solver.py` (the bridge's
  `simplify_expression`, as the oracle of one test).

**Import cost.** `import fhy_core` imports both packages today:
`symbolic/expression/__init__.py` re-exports the bridges, and `solver.py`
imports them. One `-X importtime` run of `import fhy_core` measured
393 ms in all, 208 ms of it `sympy` and 14 ms `z3`.

### Survey: the Rust API

- **Nothing solves.** The core has no solver, no SMT lowering and no
  simplifier.
- **What the port builds on:**
  - `BooleanScreen`, with its `SymbolTypes`, `Environment` and
    `SortLookup` traits (`expression/screen.rs`): `validate_predicate`
    and `validate_logical_operands`;
  - `Expression::free_identifiers` and `substitute`;
  - `SymbolType` (`Real`, `Int`, `Bool`);
  - `LiteralValue`: `Bool`, `Int(BigInt)`, `Float(f64)`, `Decimal`;
  - `BuiltinConstant::of_identifier`, and the `FunctionRegistry` (S7),
    which is a `SortLookup` for user constants and call result sorts.
  - The operations' semantics: `Divide` is the exact real quotient even
    of two integers, `FloorDivide` rounds toward negative infinity, and
    `FloorMod`'s sign follows the divisor (F-010).
- **The `z3` crate** (prove-rs/z3.rs): version 0.21.1, MIT, `rust-version`
  1.85, the crate's MSRV.
  - It depends on `z3-sys` 0.13.1 (MIT) and `log`. `z3-sys` links a
    system `libz3` found through `pkg-config` by default, and has the
    mutually exclusive link features `vendored` (build from source with
    `z3-src`), `gh-release` (download a release; pulls `reqwest` with
    `rustls`, `zip` and `serde_json`), `vcpkg`, and `bundled` (an alias of
    `vendored`). With more than one enabled it warns and falls back to
    `pkg-config`.
  - The API uses a thread-local default `Context`. `with_z3_config(&cfg,
    || ...)` runs a closure in a fresh context, and `Solver` has
    `from_string`, `check`, `get_reason_unknown`, `set_params`,
    `push`/`pop` and `get_model`.
- **z3-solver** (the Python package), probed: `Solver.from_string` reads a
  whole SMT-LIB2 script, with `set-logic`, `check-sat` and `get-info`
  lines ignored, and raises `Z3Exception` on an ill-sorted term;
  `set(timeout=1)` then `check()` returns `unknown` with
  `reason_unknown()` `"timeout"`; `parse_smt2_string` returns the
  assertions.

### Consumers and tests

**`src`.** Two modules call the solver, and two more reach it through
them:

| Module | Lines | Solver use |
|---|--:|---|
| `constraint/core.py` | 1,117 | `simplify_expression`, once: `EquationConstraint.evaluate_with_bindings` substitutes the bindings and simplifies, reading `True`/`False` literals as SATISFIED/VIOLATED |
| `constraint/system.py` | 848 | `check_expression_satisfiability` (in `_decide_satisfiability`, behind `check_satisfiability` and `check_satisfiability_with_bindings`), `does_expression_imply` (in `check_implication`), `validate_timeout_milliseconds` (3 calls). `_classify_solver_answer` maps `True`/`False`/`None` to SATISFIED/VIOLATED/UNDECIDED, the one tri-state rule |
| `param/core.py` | 2,175 | through `ConstraintSystem` and `Constraint`: `is_value_valid`, `validate_value`, `check_feasibility`, `check_subset` |
| `param/domains.py` | 2,612 | through `ConstraintSystem`: `_numeric_has_feasible_value` (satisfiability), `_does_own_admit_a_value_outside` (a witness system), `compute_constraint_implication_subset` (implication), and value checks through `is_satisfied_with_bindings` |

- `symbolic/__init__.py` re-exports the module as a namespace. `types`,
  `symbol_table` and the passes do not use the solver; `numpy.py` and
  `evaluate.py` only mention it in docstrings.
- `holds_for_all_free_assignments` and the two `assert_*` functions have
  no caller in `src`.
- **Every param with an equation constraint needs the simplifier to
  validate a value.** A nat param's bound is `EquationConstraint(x >= 0)`,
  so `is_value_valid(3)` reaches `sympy.simplify`. Finite domains, set
  constraints and enumeration never reach a backend.

**Python tests.** Counts are collected tests, parametrized cases included:

| File | Lines | Functions | Collected |
|---|--:|--:|--:|
| `symbolic/test_solver.py` | 3,046 | 127 | 346 |
| `symbolic/test_solver_properties.py` | 434 | 8 | 8 |
| `expression/passes/test_z3_pass.py` | 1,588 | 59 | 108 |
| `expression/passes/test_sympy_pass.py` | 4,646 | 141 | 627 |
| `expression/passes/test_sympy_pass_properties.py` | 303 | 6 | 6 |
| `expression/test_sympy_natives.py` | 421 | 21 | 59 |
| `expression/test_cross_cutting.py` | 193 | 6 | 20 |
| `constraint/test_constraint_system.py` | 3,382 | | 233 |
| `constraint/test_bindings_evaluation.py` | 809 | | 102 |
| `constraint/test_equation_constraint.py` | 1,027 | | 81 |
| `param/test_tri_state_feasibility.py` | 1,064 | | 100 |
| `param/test_param_intersection.py` | 1,204 | | 76 |
| `param/test_sound_feasibility.py` | 944 | | 49 |
| `param/test_subset_relations.py` | 603 | | 50 |

- The solver properties compare simplification with evaluation on integer
  and Boolean trees, and each Z3 question with a brute-force enumeration.
- **Fakes.** 27 `monkeypatch.setattr` calls, 11 in `test_solver.py` and
  16 in `test_z3_pass.py`, patch `z3.Solver.check`, `reason_unknown` or
  `set` to force `unknown` or observe the timeout. Six places in
  `test_constraint_system.py` patch `system.check_expression_satisfiability`
  itself.
- **Markers.** The `z3` marker exists, and `tests/conftest.py` skips
  marked tests when `find_spec("z3")` finds nothing. It marks 474
  collected tests in 18 files, but it is not accurate, and never mattered
  while `z3-solver` was required. There is no `sympy` marker. Five test
  modules import `z3` or `sympy` at module level.
- **Which tests reach a backend.** A probe kept outside the repo wrapped
  `z3.Solver.check` and both converters' constructors and ran the default
  suite (`-m "not slow"`): 1,239 tests reach a backend, 335 of them z3,
  967 sympy, and 63 both. Of the 335, 76 are not marked `z3`, in 12 files
  (for example 23 in `test_tri_state_feasibility.py`, 19 in
  `test_sound_feasibility.py`, 4 in `test_real_param.py`), while 213 of
  the 472 marked tests in that run never reach z3. The sympy users are
  mostly the sympy pass tests (531) and the param and constraint tests
  that evaluate constraints with bindings (351 in 29 files).

**Rust tests:** none for a solver. The screen's stories
(`screen_stories.rs`) cover `BooleanScreen`.

**Benchmarks:** none cover the solver, the bridges, constraints or
params.

### Backend-neutral and backend-specific

| Part | Today | Neutral? |
|---|---|---|
| the capability table and check, the timeout check | `solver.py` | neutral |
| the `symbol_types` coverage check and its native-constant exemption | `solver.py`, repeated in `z3.py` | neutral |
| the ill-typedness screen (`validate_predicate`, `validate_logical_operands`) | Rust core | neutral |
| the order of the checks, and per-expression screening | `solver.py` | neutral |
| the hazard screens | `solver.py` | neutral across SMT solvers: each refuses what SMT-LIB2 arithmetic cannot say in the package's semantics (no term for a constant or a non-finite float, a total function at a zero divisor, type-strict int/real equality), except the Boolean-coercion rule, which exists because the z3 Python API coerces. Strict SMT-LIB2 refuses those terms instead, so the rule stays as the refusal |
| the encodings of the three questions, and reading `sat`/`unsat`/`unknown` | `z3.py` | neutral: they are SMT-LIB2 commands |
| the tri-state result, the strict companions, the `hazard_screen` reason | both | neutral |
| lowering to z3 terms; `z3.Solver`, `set(timeout=)`, `reason_unknown()` | `z3.py` | z3-specific |
| the simplification's screen, the constant-binding refusal, the substitution | `sympy.py` | neutral (the substitution matches `Expression.substitute`, by its own docstring) |
| lowering to sympy, `sympy.simplify` and its best-effort fallbacks, lifting | `sympy.py` | sympy-specific |

### Divergences visible from Python

The port lowers to SMT-LIB2 with the core's semantics. Where the z3
bridge lowered differently, Python sees:

| # | Python today | After S8 |
|---|---|---|
| Y-1 | `DIVIDE` of two ints truncates in Z3 | exact real division, as `Divide` means in the core. The screen still refuses it (D-S8-5), so no answer changes |
| Y-2 | `FLOOR_DIVIDE` and `MODULO` are Euclidean on ints, and `MODULO` of reals raises `Z3Exception` | floor semantics on ints and reals |
| Y-3 | `POWER` is Z3's `**` for any exponent, and `Int ** Int` has sort Real | an integer literal exponent of at least one is a product with the base's sort; any other exponent is refused |
| Y-4 | a Boolean in a numeric context is coerced to `If(b, 1, 0)` | refused by the lowering, as strict SMT-LIB2 refuses it |
| Y-5 | mixed int/real arithmetic converts through the z3 API | an explicit `to_real`, with the same meaning |
| Y-6 | the implication always asserts a closed `ForAll` | quantifier-free where the question allows it (D-S8-7) |
| Y-7 | a call, or a term Z3 refuses, raises `PassExecutionError` with the cause | the lowering's `TypeError`, with the core's text |
| Y-8 | `convert_expression_to_z3_expression` names constants `x_7` and builds terms through the Python API | the parsed lowering: quoted `\|x_7\|`, with Y-1 to Y-5 |
| Y-9 | the simplifier substitutes after lowering, in sympy | the core's `substitute`, before lowering |
| Y-10 | `import fhy_core` imports `sympy` and `z3` | neither, until a query needs one |

Unchanged in meaning: the four query kinds and their answers, the
tri-state results and the strict companions' reasons, the order of the
checks, what each hazard refuses, the capability table, and what a
timeout bounds.

### Pattern choice

- **The query logic goes to Rust** (decision 2: logic-rich machinery).
  The screens, encodings and checks are about 1,000 lines of recursive
  Python walks, which a Rust facade runs over the Rust trees in one call,
  and a Rust backend must be able to use them without Python.
- **P3: the two backend traits.** Following P3:
  - `class SmtSolver(_rs.SmtSolverBase, abc.ABC)` and `class
    Simplifier(_rs.SimplifierBase, abc.ABC)` declare the abstract hooks;
    each base's `#[new]` accepts `*args, **kwargs`;
  - a Python subclass is driven from Rust through an adapter holding
    `Py<PyAny>`;
  - a Rust implementation (the process backend) is an `#[pyclass(extends
    = SmtSolverBase)]` registered with the ABC.

  The granularity rule holds: Python is called once per query, never per
  node.
- **P2:** the facade `Solver`, `SmtScript` and `SatResult`. **P1:**
  `SolverBackend`, `SolverQueryKind` and `SymbolType` stay `StrEnum`s.
- **Plain Python:** the z3-solver and sympy adapters, which talk to
  Python packages; `validate_timeout_milliseconds`, whose text is
  Python's; and the resolution of a `SolverBackend` member to an
  adapter.

**Benchmark plan: `benchmarks/test_solver.py` (S8.1).** The baseline
runs it against today's Python solver; every call whose spelling S8
changes sits in a helper marked with its decision. The trees reuse
`test_expression.py`'s deep tree (100 operations over four identifiers)
where a row names it.

| Benchmark | Measures |
|---|---|
| `test_screen_of_a_deep_predicate` | the hazard screens over a 100-operation comparison, through a question whose backend is a fake that answers at once: the screen's cost |
| `test_lower_to_z3_of_a_deep_tree` | `convert_expression_to_z3_expression` of the deep tree: lowering throughput, Python visitor before, Rust lowering and a z3 parse after |
| `test_lower_to_smtlib2_of_a_deep_tree` | after only: `convert_expression_to_smtlib2` |
| `test_check_satisfiability_of_bounds` | `0 < x && x < 10` over an int: the per-query floor |
| `test_check_satisfiability_of_a_conjunction_of_50_bounds` | 50 interval bounds over five identifiers |
| `test_does_expression_imply_of_bounds` | `x >= 1` implies `x >= 0` |
| `test_holds_for_all_free_assignments_with_a_witness` | for every `x` there is a `y > x`: the quantified encoding |
| `test_check_satisfiability_refused_by_the_screen` | a screened hazard: the screen and its warning |
| `test_simplify_expression_of_a_ground_comparison` | a fully bound comparison: the param validation path |
| `test_simplify_expression_symbolic` | `x + x - x` |
| `test_equation_constraint_evaluate_with_bindings` | the constraint layer over the simplifier |
| `test_constraint_system_check_implication` | the constraint layer over the implication |
| `test_nat_param_is_value_valid` | a nat param's value check |
| `test_int_param_intersection_feasibility` | a param intersection that asks the solver |
| `test_import_fhy_core` | a fresh interpreter importing `fhy_core`, five rounds (D-S8-16) |

The verdict follows cross-cutting rule 5. The paths at risk:

- the small queries, which now render a script and have z3 parse it, where
  the bridge built terms through the z3 API. The Python visitor costs a
  call per node, so the text path is expected to win on anything but a
  one-node tree;
- simplification, which gains a crossing into Rust and back to the Python
  adapter. It is a few microseconds against sympy's milliseconds;
- backend resolution per call (D-S8-13), which must stay a cached lookup.

### Decisions (proposed 2026-09-25)

Each names the policy it follows:

- D-S4-1: Rust semantics where the two differ;
- D-S4-2: Python names where the meaning is the same;
- "no fallback";
- "tests rewritten, not skipped";
- the crate's conventions in `rust-workspace.md` Part I: owned values and
  no global mutable state beyond identity (F-006, CONTRIBUTING),
  `#[non_exhaustive]` errors with one-line lowercase `Display` (I.3
  rule 3), the naming rules (I.3 rule 5), the layering (§I.2), and MSRV
  1.85;
- the user's direction in "Plan after S7" and for this slice: an
  SMT-LIB2 backend in pure Rust in every build, z3 optional in both
  languages, a pluggable simplifier, and sympy as a Python-only adapter
  (cited as "the direction").

Where a decision follows an earlier slice's decision or note, it says so.

- **D-S8-1: one implementation, no fallback** ("no fallback"; D-S7-1).
  - The screens, the question encodings, the capability and precondition
    checks, and the simplification's screening and substitution move to
    Rust. `solver.py` becomes thin functions over `_rs`.
  - The Python z3 lowering (`ExpressionToZ3Converter` and its
    `to_z3` registration, `_z3_floor_divide`) and the bridge's unscreened
    questions (`holds_for_all_free_assignments`, `does_expression_imply`
    and their `assert_*` companions in `passes/z3.py`) are deleted, not
    kept beside the Rust path. The solver's functions are the questions.
  - `passes.sympy.simplify_expression` is deleted; the solver's
    `simplify_expression` is the one pipeline.
- **D-S8-2: a new core module, `fhy_core::solver`** (crate conventions;
  §I.2). It depends on `expression` (with `builtins` and `registry`) and
  `identifier`, never on `pass`, so it sits beside `expression::passes` in
  the layering. Its name mirrors `fhy_core.symbolic.solver`, and it joins
  CONTRIBUTING's module table. The sketch below is settled test-first in
  S8.2, as D-S7-2's was:

  ```rust
  // fhy_core::solver
  #[derive(Debug, Clone, Default)]            // owned; backends behind Arc
  pub struct Solver { /* Option<Arc<dyn SmtSolver>>, Option<Arc<dyn Simplifier>> */ }
  impl Solver {
      pub fn new() -> Self;                                       // no backend
      pub fn with_smt_solver(self, backend: impl SmtSolver + 'static) -> Self;
      pub fn with_simplifier(self, backend: impl Simplifier + 'static) -> Self;
      pub fn can_answer(&self, kind: QueryKind) -> bool;
      pub fn ask(&self, question: &Question<'_>, context: &QueryContext<'_>) -> Result<Answer, SolveError>;
      pub fn simplify<S: BuildHasher>(&self, expression: &Expression,
          environment: &HashMap<Identifier, Expression, S>, sorts: &dyn SortLookup) -> Result<Expression, SolveError>;
  }
  #[non_exhaustive] pub enum QueryKind { Simplification, Satisfiability, Implication, UniversalValidity }
  #[non_exhaustive] pub enum Question<'a> {
      Satisfiability(&'a Expression),
      Implication { antecedent: &'a Expression, consequent: &'a Expression },
      UniversalValidity { considered: &'a HashSet<Identifier>, expression: &'a Expression },
  }
  pub struct QueryContext<'a> { /* symbol types, sorts, limits */ }   // builder: new(&dyn SymbolTypes).with_sorts(..).with_limits(..)
  #[non_exhaustive] pub struct CheckLimits { /* timeout: Option<Duration> */ }  // new(), with_timeout(Duration), timeout()

  #[derive(Debug, Clone, PartialEq)]
  pub enum Answer { Yes, No, Unknown(UnknownReason) }   // exhaustive: a question has these three answers
  #[non_exhaustive] pub enum UnknownReason { Refused(Hazard), GaveUp { reason: String } }
  #[non_exhaustive] pub enum Hazard {
      NativeConstant(Vec<Identifier>), NonFiniteLiteral(Expression), BooleanCoercion(Expression),
      PartialOperation(Expression), MixedIntRealEquality(Expression),
  }
  #[non_exhaustive] pub enum SolveError {
      NoCapableBackend(QueryKind), MissingSymbolTypes(Vec<Identifier>),
      IllTyped(NonBooleanLogicalOperandError), BoundNativeConstant(Vec<Identifier>),
      Lowering(LoweringError), Backend { backend: String, source: BackendError },
  }
  ```

  - `Answer::decided() -> Option<bool>` gives the Python tri-state.
  - A `Hazard` holds the node it refused, so the binding can name it.
  - Every `Display` is one lowercase line, for example `symbol_types is
    missing entries for identifiers: x, y`, `the backends of this solver
    cannot answer satisfiability queries`, or `the expression applies a
    partial operation off the domain its lowering is sound on`. The
    texts keep the phrases the Python message tests match, as D-S7-12's
    did.
  - Nothing is global. A `Solver` is a value its user builds, and cloning
    one shares its backends.
- **D-S8-3: two backend traits, so every backend is pluggable** (crate
  conventions; the direction for a pluggable CAS; D-10's
  `CallbackError`):

  ```rust
  pub type BackendError = Box<dyn std::error::Error + Send + Sync + 'static>;
  pub trait SmtSolver: Send + Sync + fmt::Debug {
      fn name(&self) -> Cow<'_, str>;
      fn check(&self, script: &SmtScript, limits: &CheckLimits) -> Result<SatResult, BackendError>;
  }
  pub trait Simplifier: Send + Sync + fmt::Debug {
      fn name(&self) -> Cow<'_, str>;
      fn simplify(&self, expression: &Expression) -> Result<Expression, BackendError>;
  }
  #[derive(Debug, Clone, PartialEq, Eq)]
  pub enum SatResult { Sat, Unsat, Unknown { reason: String } }   // exhaustive: check-sat's three answers
  ```

  - Both are object-safe, so the facade holds `Arc<dyn _>`.
  - The later Rust CAS backend implements `Simplifier`. The facade and
    the Python API do not change for it; `SolverBackend` gains a member
    naming it.
  - `simplify` receives the expression with the environment already
    substituted (D-S8-12). It may return its input, which means "nothing
    simpler", as the sympy bridge's best-effort cases do.
- **D-S8-4: capability follows the backends a solver holds** (D-S4-2 for
  the meaning; crate conventions: no global table). An SMT backend
  answers satisfiability, implication and universal validity; a
  simplifier answers simplification. Asking a `Solver` without a capable
  backend is `SolveError::NoCapableBackend`, checked first, as the
  capability check is today. The core ships no default backend; defaults
  are the binding's business (D-S8-13, N-S8-2).
- **D-S8-5: the screens move to Rust unchanged** (D-S4-2: the same
  meaning; "tests rewritten, not skipped": the 346 solver tests pin it).
  `fhy_core::solver::screen` keeps the five hazards, their order, the
  per-expression rule, the precedence (symbol types, then ill-typedness,
  then hazards) and both classifications, reading call result sorts and
  user constants through the `SortLookup` it is given, and the built-in
  constants through `BuiltinConstant::of_identifier`. The walks keep their
  pending nodes on the heap, as the core's other walks do.
  - Y-1 and Y-2 make `DIVIDE` of two ints and floor operations with a
    negative divisor lower soundly, so the partial-operation rule could
    later narrow to what SMT-LIB2 cannot say (a zero or non-literal
    divisor, an unsafe exponent). That changes which questions are
    decided, so it is a separate change with its own tests, recorded
    here as a follow-up, not part of S8.
- **D-S8-6: the SMT-LIB2 lowering is pure Rust with the core's
  semantics** (D-S4-1; the direction). `fhy_core::solver::smt` lowers an
  expression and its symbol types to an `SmtScript`: a logic, declarations
  and assertions over crate-private typed terms. `Display` writes the
  standard text, and `declarations()` pairs each identifier with its
  symbol and sort.

  | Expression | SMT-LIB2 |
  |---|---|
  | identifier | `(declare-const \|<name_hint>_<id>\| Int)` (`Real`, `Bool`); `\|` and `\` in a name hint become `_`, and the id keeps symbols distinct |
  | `bool` literal | `true`, `false` |
  | integer literal | a numeral; a negative one is `(- n)` |
  | `float`, `Decimal` literal | the exact rational of its value, `(/ p.0 q.0)` (or `p.0` for an integer value), with Real numerals written as decimals, since strict solvers do not read an integer numeral as a real. A non-finite float is refused |
  | `+`, `-`, `*`, negation | `+`, `-`, `*`, `(- x)`; `+x` is `x` |
  | mixed Int and Real operands | the Int side under `to_real`, in arithmetic, comparisons and piecewise branches |
  | `DIVIDE` | `(/ a b)` over reals, `to_real` on an Int side: the exact quotient |
  | `FLOOR_DIVIDE` | ints: `(div a b)` for a literal positive divisor, `(div (- a) (- b))` for a negative one, and `(ite (> b 0) (div a b) (div (- a) (- b)))` otherwise; reals: `(to_real (to_int (/ a b)))`, which keeps the Real sort the IR gives it |
  | `MODULO` (`FloorMod`) | ints: `(mod a b)`, `(- (mod (- a) (- b)))` or the `ite` of both, by the divisor's sign; reals: `(- a (* b (to_real (to_int (/ a b)))))` |
  | `POWER` | an integer literal exponent of at least one is a product built by squaring through `let`, so its size is logarithmic in the exponent; any other exponent is refused |
  | comparisons | `=`, `distinct`, `<`, `<=`, `>`, `>=` |
  | `LogicalExpression`, `!` | n-ary `and`, `or`; `not` |
  | piecewise | a right-folded `ite`, first match wins |
  | call | refused, naming the call; a user function names `inline_functions` |
  | native constant's identifier | refused |
  | a Boolean where a number is required, or the reverse | refused |

  - Division by zero is left to SMT-LIB2, where `/`, `div` and `mod` are
    total with an unspecified value at zero. The screen refuses those
    shapes before lowering, as today.
  - **The logic** is the narrowest of `QF_LIA`, `QF_LRA`, `QF_NIA`,
    `QF_NRA`, `LIA`, `LRA`, `NIA` and `NRA` that fits (nonlinear when a
    product has two non-constant factors, or a power appears), and `ALL`
    otherwise: mixed Int and Real terms, or only Booleans.
  - A refusal is a `LoweringError` (`#[non_exhaustive]`), which the
    facade only meets for calls, since the screens refuse the other
    shapes first.
- **D-S8-7: the questions are encoded by the facade** (D-S4-2: the same
  questions; D-S4-1: the core's encoding; Y-6). Each is one script:

  | Question | Script | `Yes` when |
  |---|---|---|
  | satisfiability of `e` | `(assert e)` | `sat` |
  | `a` implies `c` | `(assert (and a (not c)))` | `unsat` |
  | universal validity, free `F`, considered `C` (those of `e`'s identifiers) | `F` and `C` both non-empty: `(assert (forall (C) (not e)))`, with `F` declared; `C` empty: `(assert (not e))`; `F` empty: `(assert e)` | `unsat`, `unsat`, `sat` |

  A quantifier appears only where the question alternates them, so a
  quantifier-free solver answers every other question. `sat` and `unsat`
  map to `Yes` or `No` by the table, and `unknown` to
  `Unknown(GaveUp { reason })`. Satisfiability no longer goes through the
  implication. A considered identifier the expression does not mention
  needs a sort but quantifies nothing, as today.
- **D-S8-8: one-shot scripts; no sessions or models; timeouts as a
  `Duration`** (crate conventions; D-S4-2 for the Python name).
  - Each query builds one script and one check. Today's bridge builds a
    fresh `z3.Solver` per query, so nothing incremental is lost.
    `push`/`pop` sessions, which a param's many related questions could
    use, and models, which no consumer reads, are non-goals. A later
    `SmtSession` trait can add them without changing `SmtSolver`.
  - `CheckLimits::with_timeout(Duration)` bounds a check, and each backend
    enforces it its own way (D-S8-9, D-S8-10). A backend that runs out of
    time answers `Unknown { reason: "timeout" }`.
  - Python keeps `timeout_milliseconds` and `validate_timeout_milliseconds`
    with its text. A value above `u64::MAX` milliseconds raises the same
    `ValueError`.
- **D-S8-9: the backends in Rust** (the direction; crate conventions).
  - **`SmtLib2Process`, in every build.** It drives any SMT-LIB2
    executable over standard input and output, with `std::process` only:
    `SmtLib2Process::new(program).with_args(args)`, for example `z3 -in`
    or `cvc5 --lang=smt2`. It writes the script, `(check-sat)`,
    `(get-info :reason-unknown)` and `(exit)`, and reads `sat`, `unsat` or
    `unknown` with the reason. An `(error ...)` line, a missing
    executable or an unexpected answer is a `BackendError`. A timeout
    kills the child and answers `Unknown { reason: "timeout" }`. The
    output is read on a helper thread, so the wait can time out. It is
    configured explicitly; nothing searches `PATH`.
  - **`Z3Solver`, behind the off-by-default `z3` cargo feature.**
    `fhy-core` gains `z3 = { version = "0.21", optional = true }` and
    `[features] z3 = ["dep:z3"]`.
    - It links the system `libz3` through `pkg-config`, `z3-sys`'s
      default. A downstream crate that wants a vendored or downloaded
      z3 enables `vendored` or `gh-release` on its own `z3` dependency,
      which Cargo's feature unification applies. `fhy-core` re-exports
      neither, so its all-features graph gains only `z3`, `z3-sys`,
      `log` and `pkg-config`, all MIT or Apache-2.0, and `deny.toml`
      needs no new license.
    - Each check runs in a fresh context through `with_z3_config`, with
      the timeout in its `Config`, so no z3 state outlives a query: the
      crate's rule that nothing but identity is global. `Z3Solver`
      holds only its configuration and is `Send + Sync`.
    - It builds z3 terms from the script's typed terms, with no text in
      between, so a malformed term cannot pass silently through the
      crate's `from_string`, which returns no error.
- **D-S8-10: the published wheels reach z3 through z3-solver** (the
  direction; "no fallback"). The Python extension cannot enable a cargo
  feature at install time, and linking `libz3` into every wheel would
  make z3 a required dependency and ship it twice beside z3-solver. So
  `fhy-core-py` never enables `z3`. Python's `SolverBackend.Z3` is a
  Python `SmtSolver` over z3-solver, in `passes/z3.py`:
  - `Z3Solver.check(script, timeout_milliseconds)` creates a
    `z3.Solver()`, applies `set(timeout=...)`, calls
    `from_string(script.text)`, then `check()`, and reads
    `reason_unknown()` for `unknown`;
  - z3-solver releases the interpreter during its ctypes calls, as it does
    today, so threads keep running.
- **D-S8-11: the Python backend classes are P3** (P3; D-S5-9 for new
  names; D-S5-7 for errors).
  - `SmtSolver` declares the abstract `check(script: SmtScript, *,
    timeout_milliseconds: int | None) -> SatResult` and a `name` property
    that defaults to the class's `__name__`. `SmtScript` is a frozen
    pyclass with `text` (the whole script), `logic` and `declarations`.
  - `Simplifier` declares the abstract `simplify(expression) ->
    Expression` and the same `name`.
  - `SatResult` is a frozen pyclass: `SatResult.SAT`, `SatResult.UNSAT`,
    `SatResult.unknown(reason)`, with `status` (a `SatStatus` `StrEnum`
    of `"sat"`, `"unsat"`, `"unknown"`) and `reason`. It compares by value
    and pickles as a call.
  - `SmtLib2ProcessSolver(program, args=())` is the Rust
    `SmtLib2Process` as an `extends = SmtSolverBase` class, registered with
    `SmtSolver`.
  - The adapters hold `Py<PyAny>` and call the hook once per query. A
    result of the wrong type raises `TypeError` in S2's style. An
    exception the hook raises propagates as the same object, and a
    `KeyboardInterrupt` passes through unwrapped.
  - While a Rust-native backend runs, the binding detaches from the
    interpreter, so other Python threads run, as they do during a
    z3-solver call.
- **D-S8-12: the sympy simplifier is a Python-only P3 backend, and the
  facade substitutes** (the direction; D-S4-1 for the substitution; Y-9).
  - `SympySimplifier(Simplifier)` in `passes/sympy.py` lowers, runs
    `_try_simplify_sympy_expression` with its three best-effort cases
    unchanged, and lifts.
  - The facade's `simplify` first screens with `validate_logical_operands`
    and the environment, then refuses an environment binding a referenced
    native constant (`NativeConstantBindingError`), then substitutes with
    the core's `substitute`, and hands the result to the simplifier. The
    sympy bridge's own substitution already promised the IR's
    simultaneous semantics, and a Rust CAS backend should not have to
    re-implement it.
  - `ExpressionToSympyConverter`, `SympyVariableSubstitutionPass`,
    `SymPyToExpressionConverter` and the three bridge functions stay
    public Python, with their registrations (D-S4-2).
- **D-S8-13: the Python API keeps its names and meaning** (D-S4-2).
  - The eleven names keep their signatures. A `SolverBackend` member
    resolves to its adapter: `Z3` to `passes.z3.Z3Solver`, `SYMPY` to
    `passes.sympy.SympySimplifier`, each created on first use and
    reused, since they hold no state. `get_backend_capabilities` keeps
    its static table, a property of the kind of backend.
  - New names (the Rust names, D-S5-9): `Solver`, a pyclass over the Rust
    facade with the functions' names as methods, minus `backend`;
    `SmtSolver`, `Simplifier`, `SmtScript`, `SatResult`, `SatStatus`,
    `SmtLib2ProcessSolver`; `is_backend_available(backend)`;
    `convert_expression_to_smtlib2(expression, symbol_types)`, which
    returns the script's text; and the errors of D-S8-14 and D-S8-15.
  - `convert_expression_to_z3_expression` keeps its name and result
    shape: it parses the Rust lowering with `z3.parse_smt2_string`, and
    maps each identifier to its declared constant. Its terms follow
    D-S8-6 (Y-8).
  - How module functions pick a backend when `backend` is omitted, and
    whether constraints and params can use a plugged backend, is N-S8-2.
- **D-S8-14: errors are the core's text under the Python classes, and the
  warnings are kept** (D-S4-1, D-S7-12; D-S6-5 for logging).

  | Core | Python |
  |---|---|
  | `NoCapableBackend` | `SolverCapabilityError`, whose text keeps "cannot answer" |
  | `MissingSymbolTypes` | `KeyError`, whose text keeps "symbol_types is missing" |
  | `IllTyped` | `NonBooleanLogicalOperandError`, as S4.3a maps it |
  | `BoundNativeConstant` | `NativeConstantBindingError` |
  | `Lowering`, for a call or another refused term | `TypeError`, no longer wrapped in `PassExecutionError` (Y-7); for a native constant in `convert_expression_to_*`, `NativeConstantLoweringError` |
  | `Backend` from a Python backend | the backend's exception itself (D-S8-11) |
  | `Backend` from a Rust backend | the new `SolverBackendError(RuntimeError)`, with the backend's name and the source's text |

  A screened hazard is logged at WARNING on `fhy_core.symbolic.solver`,
  as today: the binding writes the entry point's name, the core's hazard
  text, the node's Python `repr` and the identifier sorts at that node.
  A backend's `unknown` is logged at WARNING with its reason, as the z3
  bridge logs it. The core does not log.
- **D-S8-15: a missing backend is an error at query time, never a
  degraded answer** ("no fallback"; the direction).
  - Resolving `SolverBackend.Z3` without z3-solver, or `SYMPY` without
    sympy, raises the new `SolverBackendUnavailableError(ImportError)`,
    `register_error`ed, whose message names the package and the extra to
    install, as the numpy evaluator's `ImportError` does. It is raised
    when a query needs the backend, never at import.
  - Constraints and params let it propagate. They do not turn it into
    UNDECIDED: a missing package is a configuration error, not an
    undecided question, and reporting UNDECIDED would silently weaken
    every validation. Paths that never reach a backend (finite domains,
    set constraints, enumeration, the screens) keep working without
    either package.
  - Capabilities can be asked without running a query:
    `get_backend_capabilities(backend)` (what a kind of backend answers),
    `is_backend_available(backend)` (whether its package imports), and
    `Solver.can_answer(kind)`.
- **D-S8-16: `import fhy_core` imports neither package** (the direction;
  Y-10). `fhy_core.symbolic.expression` re-exports the four bridge
  functions through a module `__getattr__` (PEP 562), declared under
  `TYPE_CHECKING` for type checkers, and `solver.py` imports the adapters
  on first use. `passes/sympy.py` and `passes/z3.py` keep their
  module-level imports, since they subclass `sympy.Piecewise` and use `z3`
  throughout; importing either without its package raises
  `SolverBackendUnavailableError`. A fresh-interpreter test pins that
  neither is in `sys.modules` after `import fhy_core`, and
  `test_import_graph.py` keeps passing, since function-level imports are
  no load-time edges.
- **D-S8-17: tests that need a backend are marked, and a minimal
  session proves the marks** ("tests rewritten, not skipped").
  - Two markers, `z3` and a new `sympy`, and `tests/conftest.py` skips a
    marked test when its package is not installed, as it does for `z3`
    today. That is the only skip, for a documented configuration.
  - Markers follow what a test reaches, from the probe: the 76 unmarked
    z3 users and every sympy user are marked, and marks on tests that
    reach no backend are removed. The five modules that import `z3` or
    `sympy` at module level mark the whole module and import the package
    inside the tests or through `pytest.importorskip`.
  - A new nox session, `tests_minimal`, installs the package with neither
    package and runs the suite. An unmarked test that reaches a missing
    backend fails there with `SolverBackendUnavailableError`, so a wrong
    mark cannot hide. CI runs it on one Python version.
  - The `test` dependency group gains each package that becomes an
    extra, so the ordinary sessions run everything.
  - Under N-S8-1 (b), `tests_minimal` leaves out z3-solver only, and the
    `sympy` marks take effect when sympy becomes an extra.
- **D-S8-18: the `z3` feature in CI and docs** (crate conventions; MSRV
  1.85).
  - CI's `rust` job builds with `--all-features`, so it installs a
    `libz3` for the feature, and the `z3` executable for the process
    backend's tests; S8.3 checks the distribution's `libz3`
    against `z3-sys` 0.13 first. If it is too old, the job links the
    `libz3` that the z3-solver wheel ships, through
    `Z3_LIBRARY_PATH_OVERRIDE`, or enables `z3/gh-release` on that job's
    command line only, which `deny.toml`'s manifest graph never sees.
  - The `rust-msrv` job keeps checking the default features. The `z3`
    crate's own `rust-version` is 1.85, so the feature does not raise the
    MSRV.
  - `deny` keeps `all-features = true` and passes with the crates of
    D-S8-9.
  - The crate README documents the feature and how to choose the link
    method. docs.rs builds the default features.
- **D-S8-19: the Rust tests specify the solver first** (the tests rule;
  S7.2's test-first practice). The screens, the lowering, the encodings,
  the facade's errors and order, and the process protocol are specified by
  Rust tests written against `todo!()` stubs, with a traceability table
  from `test_solver.py` and `test_z3_pass.py`. The tests that need a real
  solver run under the `z3` feature.
- **D-S8-20: the Python tests are rewritten, not skipped** (the tests
  rule). The behavioral tests stay and change only where a decision
  changes what they pin; each change is recorded with its reason, as S4.4
  to S7 did.

### Needs the user

- **N-S8-1 (resolved 2026-09-25 by the user, as option (a)): when
  `sympy` and `z3-solver` become optional extras.** Both
  are required in `pyproject.toml` today. D-S8-16 makes both lazy either
  way, so the question is only what `pip install fhy_core` brings. The
  catch is sympy: it is today's only simplifier, and every param with an
  equation constraint, a nat param included, needs it to validate a value
  (the survey above); 351 param and constraint tests reach it that way.
  z3-solver is needed only by the solver's questions: feasibility,
  intersection, subset and implication checks.
  - (a) **Both become extras now:** `fhy_core[z3]`, `fhy_core[sympy]`,
    and `fhy_core[solvers]` for both. Without sympy, `is_value_valid` of a
    nat param raises `SolverBackendUnavailableError`.
  - (b) **z3-solver becomes an extra now; sympy stays a required
    dependency** (lazily imported) until the Rust CAS backend of the next
    slice can decide a ground constraint, and becomes an extra then.
  - (c) **Both stay required** (lazily imported) until Rust backends cover
    both.

  Recommendation: (b). It makes z3 optional where only the questions
  that name a solver need it, keeps a plain install able to validate
  params, and saves the 208 ms sympy import either way. (a) is the full
  "optional like Python" state, at the price of breaking value validation
  for a plain install until the CAS slice.
- **N-S8-2 (resolved 2026-09-25 by the user, as option (b)): the default
  backends, and whether constraints and params can use a plugged one.** The module functions default to `backend=Z3` or
  `SYMPY`, and `constraint` and `param` call them with the defaults. So a
  backend a user builds, such as an `SmtLib2ProcessSolver` for cvc5, a
  Python `SmtSolver`, or the later Rust CAS, reaches only direct calls of
  a `Solver`. The policy does not cover this, since CONTRIBUTING requires
  the maintainer's agreement for a new process-global static with
  interior mutability.
  - (a) **No global state.** `backend` keeps its defaults and resolves
    per call (D-S8-13). Plugged backends work through `Solver` objects
    only, and constraints and params always use z3-solver and sympy.
  - (b) **A replaceable default `Solver` in the extension's module
    state,** as N-S7-2 (a) held the function registry: a `Mutex<Arc<_>>`,
    never locked across a call into Python, with
    `set_default_solver(solver)` and `get_default_solver()`. The module
    functions' `backend` defaults to `None`, meaning the default solver,
    whose initial value holds the z3-solver and sympy adapters. A named
    member still selects its adapter. Constraints and params then use a
    plugged backend unchanged, and CONTRIBUTING's section records the
    static.
  - (c) **Thread a `solver` argument** through `ConstraintSystem`'s
    entry points, `Constraint.evaluate_with_bindings` and the param
    checks that reach them: more than a dozen public signatures.

  Recommendation: (b). It is the only option in which a plugged backend
  reaches the layers that ask the most questions, without changing their
  signatures, and it keeps the core free of global state, as the S7
  registry did.

### Steps

1. **S8.1: benchmarks.** Add `benchmarks/test_solver.py` as planned
   above, and record the baseline here, on today's Python solver.
2. **S8.2: core additions, test-first, with Rust tests.**
   - `fhy_core::solver` (`solver.rs`) with `solver/screen.rs`,
     `solver/smt.rs` (with `smt/term.rs`, `smt/lower.rs`, `smt/print.rs`),
     `solver/backend.rs` (the traits, `SatResult`, `CheckLimits`),
     `solver/error.rs` and `solver/process.rs`.
   - The tests are written first and fail against `todo!()` stubs, as in
     S4.2 and S7.2. `lib.rs`, the crate README and CONTRIBUTING's module
     table list the module.
   - Nothing in Python changes, so the suite stays green.
3. **S8.3: the `z3` feature.** `solver/z3.rs` under `#[cfg(feature =
   "z3")]`, its tests under the same `cfg`, the manifest and README, and
   the CI changes of D-S8-18. The step ends with `deny` and the `rust`
   job green with the feature built.
4. **S8.4: the binding.** Add `rust/fhy-core-py/src/solver.rs` with:
   - `backends.rs`: `SmtSolverBase`, `SimplifierBase`, the two Python
     adapters, and `SmtLib2ProcessSolver`;
   - `values.rs`: `SmtScript`, `SatResult`, and the result conversions;
   - `facade.rs`: `Solver`, and the functions `solver.py` calls, which
     take the registry snapshot of S7 as their `SortLookup`;
   - `error.rs`: D-S8-14's mapping, and the warnings;
   - `state.rs`, only under N-S8-2 (b).

   Everything new goes into `_rs.pyi`. Nothing in Python uses it yet, so
   the suite stays green.
5. **S8.5: the Python switch** (marked breaking). `solver.py` becomes the
   thin layer. `passes/z3.py` becomes the z3-solver adapter (`Z3Solver`,
   `convert_expression_to_z3_expression`) and loses the converter and the
   unscreened questions. `passes/sympy.py` gains `SympySimplifier` and
   loses `simplify_expression`. `symbolic/expression/__init__.py`
   re-exports lazily (D-S8-16). The README's expression row changes. It
   lands together with S8.6 when the migration is small enough to review
   in one commit; otherwise it leaves exactly the tests of the migration
   table failing, as S7.4 did.
6. **S8.6: tests.** Migrate the tests and add the interface suite (the
   test plan below).
7. **S8.7: optional extras.** `pyproject.toml` per N-S8-1, the markers and
   `conftest.py` of D-S8-17, the `test` group, the `tests_minimal` nox
   session and its CI job, and the README's install line. The step ends
   with `tests_minimal` green.
8. **S8.8: benchmarks after,** recorded here with the verdict, then the
   status, the implementation notes and this checklist.

Commit per step. Every step ends with `pytest` and `-m "not very_slow"`
green, the `property` session, `lint` and `type_check` clean,
`tests/test_rs_stub.py` green, and the Rust gate green (fmt, clippy `-D
warnings`, tests, doc `-D warnings`, deny, `cargo +1.85 check`), with the
`z3` feature built from S8.3 on.

### Test plan

**Rust tests, written first (S8.2 and S8.3),** in a new
`tests/it/solver/` area:

- **`screen_stories.rs`:** each hazard, the shapes it refuses and the
  neighbours it admits, one case per screen test of `test_solver.py`;
  the order of the five; the first hazard wins; each
  expression screened on its own; the classifications, call result sorts
  through a `SortLookup` included; built-in and user constants, and an
  identifier merely named like one; a 100,000-level tree on a small
  stack.
- **`smt_lowering_stories.rs`:** the script text for each row of
  D-S8-6's table, pinned as strings: the exact rationals of `0.1`, `1e16`
  and a decimal; `to_real` in each position; the three floor encodings
  per divisor sign, and the real ones; a power by squaring, with its size
  logarithmic in the exponent; the logic chosen for each theory mix;
  symbol quoting; every refusal and its `Display`.
- **`solver_stories.rs`**, with a recording fake `SmtSolver` and a fake
  `Simplifier`:
  - capabilities, and `NoCapableBackend` checked first;
  - the order: missing symbol types, then ill-typedness, then the hazards,
    each pinned where the Python tests pin it;
  - the script of each question, and the answer from each `SatResult`
    (D-S8-7), quantifiers only where both sets are non-empty;
  - `Unknown` reasons, screened and given up;
  - the timeout reaching the backend;
  - a backend's error as `SolveError::Backend` with its source;
  - simplification: the screen with the environment, the constant-binding
    refusal, the substitution before the simplifier, and the simplifier's
    input itself when the environment binds nothing.
- **`process_stories.rs`**, `cfg(unix)`, with `sh` scripts as fake
  solvers: the protocol for each answer and reason, an `(error ...)`
  line, a missing program, a solver that never answers (the timeout kills
  it), and a process that dies. Tests against a real `z3` executable run
  when the `FHY_SMT_SOLVER` variable names one; CI's `rust` job installs
  the distribution's `z3` and sets it.
- **`z3_stories.rs`**, `cfg(feature = "z3")`: every decided case of
  `test_solver.py` decided the same way, `unknown` on a timeout,
  concurrent checks on several threads, and nothing kept between queries.
- **`solver_properties.rs`:** the lowering of a random screened-safe
  ground tree prints a script whose satisfiability, under the `z3`
  feature, agrees with an exact rational reference evaluation; and the
  three questions agree with brute force over small integer domains, as
  the Python properties do (feature-gated).
- A traceability table maps `test_solver.py`, `test_solver_properties.py`
  and `test_z3_pass.py` to them, as S4.2 and S7.2 did.

**The interface suite, `tests/symbolic/test_solver_rust_binding.py`,**
covers what the binding adds over the core:

- **Class structure.** `SmtSolver` and `Simplifier` enforce their
  abstract methods; a subclass with its own `__init__` constructs;
  `SmtLib2ProcessSolver` extends `SmtSolverBase` and is an `SmtSolver`;
  `Solver`, `SmtScript` and `SatResult` are frozen; the stubs are covered
  by `tests/test_rs_stub.py`.
- **P3 driving.** A Python `SmtSolver` receives the script text and the
  timeout, once per query; each `SatResult` becomes its answer; a wrong
  result type raises `TypeError`; an exception propagates as the same
  object; a `KeyboardInterrupt` passes through; a nested query inside a
  hook works. The same for a Python `Simplifier`, whose input is the
  substituted expression object.
- **`SatResult`**: construction, equality, `repr`, pickles.
- **Backends.** `SolverBackend.Z3` and `SYMPY` resolve to their adapters,
  one object each; `is_backend_available`; `SolverBackendUnavailableError`
  and its message, in a subprocess with `sys.modules["z3"] = None` (and
  the same for `sympy`); `get_backend_capabilities` unchanged.
- **Imports.** A fresh `import fhy_core` imports neither `sympy` nor `z3`;
  the lazy re-exports resolve; importing a bridge without its package
  raises `SolverBackendUnavailableError`.
- **Errors and warnings.** Each row of D-S8-14, and the warning's logger,
  level, entry point, node `repr` and sorts.
- **The lowering.** `convert_expression_to_smtlib2` pins a few scripts,
  and `convert_expression_to_z3_expression`'s identifier map holds the
  declared constants.
- **Threads.** Concurrent queries from several threads, each with a
  Python backend and with the process backend.
- **Under N-S8-2 (b):** the default solver, replacing it, and
  constraints and params reaching a plugged backend.

**Migrating the existing tests.** No test is skipped or deleted without a
rewrite, and each change is recorded with its reason:

- **`test_solver.py` (346).**
  - The 11 `monkeypatch.setattr` calls on `z3.Solver`, which force
    `unknown` or observe the timeout, use a fake `SmtSolver` through a
    `Solver`, or patch `Z3Solver.check` (D-S8-11).
  - The white-box `_BACKEND_CAPABILITIES` drift test reads
    `get_backend_capabilities` over every `SolverBackend` member.
  - `test_simplify_expression_matches_direct_bridge_pipeline` compares
    with `SympySimplifier` over the substituted expression (D-S8-1,
    D-S8-12).
  - The warning tests keep matching the phrases the core's texts keep;
    any that do not are rewritten to the core's text (D-S8-14).
  - The tests that describe a `Z3Exception` wrapped as
    `PassExecutionError` pin the refusal that replaces it (Y-7).
  - The rest keep their meaning; markers follow D-S8-17.
- **`test_solver_properties.py` (8)** is unchanged, with markers.
- **`test_z3_pass.py` (108).**
  - The converter's shape tests become Rust lowering stories, and the
    Python file keeps tests of `convert_expression_to_z3_expression` over
    the new lowering, rewritten where Y-1 to Y-5 and Y-8 change a term.
  - `_z3_floor_divide`'s tests become the floor-encoding stories.
  - The 17 calls of the unscreened questions call the solver's functions.
    Where a test relied on the missing screen, it pins the screen's
    refusal or asks through `convert_expression_to_smtlib2` instead.
- **`test_cross_cutting.py` (20):** the registration list loses `to_z3`
  (D-S8-1), and the one-literal agreement compares the SMT-LIB2 numeral
  with sympy's.
- **`test_sympy_pass.py` (627)** changes only for its one call of the
  bridge's `simplify_expression` and its markers.
  `test_sympy_pass_properties.py` and `test_sympy_natives.py` change only
  their markers.
- **Constraint and param tests** change only where they pin a message or
  a `PassExecutionError` that Y-7 or D-S8-14 changes, and in their
  markers. The six places in `test_constraint_system.py` that patch the
  solver's functions keep working, since those names stay.
- **`tests/test_import_graph.py`** gains no edge; a new test pins D-S8-16.

### S8.1 baseline (2026-09-25, 87dd88f plus the new benchmarks)

`benchmarks/test_solver.py` implements the benchmark plan above, except
`test_lower_to_smtlib2_of_a_deep_tree`, which has nothing to measure
before S8 and joins the file with `convert_expression_to_smtlib2`.

- **The screen row** asks a satisfiability question whose backend answers
  at once: before S8 the fixture `_instant_backend` replaces the z3
  bridge's implication by a function answering `False`, so the row runs
  the capability, timeout, symbol-type, ill-typedness and hazard checks
  and no solver. The helper `_ask_an_instant_backend` carries the D-S8-11
  mark; after S8 it asks a `Solver` holding a Python `SmtSolver` that
  answers `sat`, so the row adds the lowering and one Python call.
- **The 50 bounds** alternate `b > -i` and `b < i + 100` over five
  integer identifiers.
- **The refused question** divides two real variables, a partial
  operation the screen refuses, logging its warning each round.
- **The import row** starts a fresh interpreter importing `fhy_core`, five
  rounds through `pedantic`.

Median time per call, from `.nox/benchmark-3-11/bin/python -m pytest
benchmarks/test_solver.py -n 0 --benchmark-only`, the benchmark
session's environment, measuring today's Python solver and bridges. The
machine is the S0 one, with Python 3.11.13 and pytest-benchmark 5.3.0.
The load average was below 1, and the table lists the best of three
runs' medians.

| Benchmark | before |
|---|--:|
| `test_screen_of_a_deep_predicate` | 467.8 µs |
| `test_lower_to_z3_of_a_deep_tree` | 2.12 ms |
| `test_check_satisfiability_of_bounds` | 1.56 ms |
| `test_check_satisfiability_of_a_conjunction_of_50_bounds` | 5.78 ms |
| `test_does_expression_imply_of_bounds` | 1.36 ms |
| `test_holds_for_all_free_assignments_with_a_witness` | 1.39 ms |
| `test_check_satisfiability_refused_by_the_screen` | 48.0 µs |
| `test_simplify_expression_of_a_ground_comparison` | 128.8 µs |
| `test_simplify_expression_symbolic` | 72.7 µs |
| `test_equation_constraint_evaluate_with_bindings` | 141.3 µs |
| `test_constraint_system_check_implication` | 1.36 ms |
| `test_nat_param_is_value_valid` | 145.3 µs |
| `test_int_param_intersection_feasibility` | 1.95 ms |
| `test_import_fhy_core` | 487 ms |

- **The screens** cost 468 µs over the 100-operation comparison: five
  recursive Python walks and the classifications they repeat per node.
- **The z3 lowering** of the deep tree takes 2.1 ms, a Python visitor
  call and a z3 API call per node.
- **The smallest question** takes 1.4 to 1.6 ms, most of it z3's own
  `check` and building the terms; the 50 bounds 5.8 ms.
- **Simplification** of a bound comparison takes 129 µs, and a nat
  param's value check 145 µs, with sympy's cache warm across rounds.
- **The import** of `fhy_core` takes 487 ms in a fresh interpreter, which
  includes sympy and z3.

### S8.2 implementation notes

The tests were written first, against `todo!()` stubs of `Hazard::find`,
`SmtScript::lower` and its `Display`, `Solver::ask` and `simplify`, and
`SmtLib2Process::check`: 213 of the 224 new integration tests failed (the
other 11 pin plain data, such as a `Display` of a hand-built error), and
all pass now. A proptest regression file written while the stubs failed
was deleted, as in S7.2. The new module is `rust/fhy-core/src/solver.rs`
(the facade, the questions, answers and context) with `solver/backend.rs`
(the traits, `SatResult`, `CheckLimits`, `SimplifyContext`),
`solver/error.rs`, `solver/screen.rs`, `solver/process.rs` and
`solver/smt.rs` with `smt/term.rs`, `smt/lower.rs` and `smt/print.rs`.
Every public item has one path, `fhy_core::solver::X`. `lib.rs`, the
crate README and manifest description, and CONTRIBUTING's layering list
and module table list the module.

Where the shape differs from D-S8-2's and D-S8-3's sketches, or fills
them in:

- **`Simplifier::simplify` takes a `SimplifyContext`** besides the
  expression. The next slice's Rust CAS backend, SymPy through pyo3, will
  need more than the expression, at least the sorts of native constants
  and named functions, and perhaps their values or limits. A context
  struct with private fields and accessors can gain those without
  changing the trait, which the user asked for. It carries the solver's
  `SortLookup` today.
- **`Hazard::find(expression, symbol_types, sorts)` is public.** The
  screen is a function of its own, as `BooleanScreen` is, so the stories
  pin each hazard directly and another caller can screen without a
  backend. `Hazard::node()` returns the refused node.
- **The facade holds shared backends.** `Solver` also has
  `with_shared_smt_solver(Arc<dyn SmtSolver>)`,
  `with_shared_simplifier`, and the accessors `smt_solver()` and
  `simplifier()`, for the binding, which keeps its adapters behind an
  `Arc`, and for the tests' recording fakes.
- **`SmtScript::lower` checks in the order of the Python
  `convert_expression_to_z3_expression`:** missing symbol types, then the
  logical-operand screen, then native constants, then the first node in
  post-order without a term (`LoweringError::MissingSymbolTypes`,
  `IllTyped`, `NativeConstants`, `NonFiniteLiteral`, `Call`,
  `SortMismatch`, `UnsupportedPower`). The facade runs its own checks
  first and lowers without repeating them.
- **A non-Boolean expression is named.** `SmtScript::lower` of a numeric
  expression declares the constant `value` of its sort and asserts
  `(= value e)`, and `value_sort()` reports the sort. The binding's
  `convert_expression_to_z3_expression` must keep converting numeric
  expressions (D-S4-2), and z3's parser returns only assertions, so it
  reads the term as the assertion's second argument. No identifier's
  symbol can be `value`, since those end in `_` and digits.
- **Shared terms are written once, under `let`.** Terms live in one arena
  and refer to their arguments by id, so a node an expression shares, and
  the operands a floor encoding or a power by squaring repeats, are one
  term. The printer binds each application an assertion reaches from
  several places to `t!1`, `t!2`, ... in id order, nested lets at the top
  of the assertion (inside a quantifier), so a 64-level doubling DAG
  prints in linear size. Printing and lowering run on explicit work
  lists, so a 100,000-level tree lowers and prints on a small stack.
- **Numerals.** An integer literal in a real position is a real numeral
  (`1.0`), not `(to_real 1)`, and the negation of a constant is folded,
  so `x // -3` is `(div (- |x|) 3)`. Rationals are in lowest terms:
  `0.1` is `(/ 3602879701896397.0 36028797018963968.0)`, the decimal `2.50`
  is `(/ 5.0 2.0)`.
- **Linearity is syntactic.** The property that compares satisfiability
  with brute force found z3 refusing `(* (+ 1 2) x)` in `QF_LIA`: solvers
  read a coefficient or a divisor as linear only when it is a numeral. So
  a product is nonlinear when two of its factors are not numerals, and a
  quotient, `div` or `mod` when its divisor is not one, which refines
  D-S8-6's "two non-constant factors, or a power". A power of a variable
  is a product of two variables, so the rule covers it. The three shrunk
  cases, such as `(* x (mod 0 1))`, stay as seeds in
  `tests/proptest-regressions/solver/solver_properties.txt`. A script with
  both numeric sorts, or with only Booleans, is `ALL`, as D-S8-6 says;
  `to_int` in a real floor operation makes a script mixed.
- **Declarations** are ordered by identifier id, and `value` comes last.
  A universally quantified identifier is bound by the `forall`, never
  declared.
- **Texts.** The identifiers in `SolveError`, `LoweringError` and
  `Hazard` texts are written as `name::id`, their `Debug` form and also
  their Python `repr`, since Python's message tests match `repr(x)` in the
  `KeyError`. The Boolean-coercion text is lowercase, `lowers a boolean
  operand into a numeric context`; the one Python test matching the old
  capital `Boolean` is rewritten in S8.6. `SolveError::Substitution` is new
  and cannot arise after the screen, which refuses a number bound into a
  case condition first; it keeps `substitute`'s error typed.
- **The screen is Python's, including one quirk.** A division is safe only
  with a provably real operand, and "provably real" follows negation but
  not unary plus, as `_does_operand_lower_to_real_sort` did;
  `partial_operation_hazard_reads_a_real_dividend_through_arithmetic_but_not_unary_plus`
  pins it, and D-S8-5's follow-up can widen it. The classifications are
  computed bottom-up on a work list and remembered per node, and the
  pre-order walks skip a shared node met again, so a 64-level doubling DAG
  screens at once and a 100,000-level chain or piecewise nest screens on a
  small stack.
- **The process protocol** writes the script and `(check-sat)`, reads the
  first non-blank line, asks `(get-info :reason-unknown)` only after
  `unknown`, and then writes `(exit)` and closes the input. A reason is
  read from `(:reason-unknown "r")` or `(:reason-unknown r)`; any other
  reply, such as an `(error ...)` to `get-info`, gives the empty reason.
  A write the program refuses is dropped, since its output or its exit
  then tells why. Standard error is discarded, and the output thread ends
  when the program's output closes.
- **Solver-backed properties** run when `FHY_SMT_SOLVER` names an
  SMT-LIB2 executable (locally the z3 4.16 of the z3-solver wheel, `z3
  -in`), and under the `z3` feature from S8.3. Each check is bounded to
  2 s and an `unknown` is accepted, so the properties pin that every
  decided answer agrees: without the bound, z3 used 69 GB of memory on
  one generated quantified nonlinear case before it was killed.
- **Tests.** `tests/it/solver/`: `screen_stories.rs` (92 tests, counting
  `rstest` cases), `smt_lowering_stories.rs` (76), `solver_stories.rs`
  (38), `process_stories.rs` (12, `cfg(unix)`, `sh` scripts as fake
  solvers) and `solver_properties.rs` (6); the fakes are in
  `tests/it/support/solver.rs`. The Rust gate: fmt, clippy `-D warnings`,
  3,040 tests, doc `-D warnings`, deny, `cargo +1.85 check`, and the
  public-paths checks.

Traceability of the Python tests (`test_solver.py` is `S`,
`test_solver_properties.py` `P`, `test_z3_pass.py` `Z`); the Python tests
stay, and S8.6 migrates them:

| Python tests | Rust tests | Note |
|---|---|---|
| S the capability tests (5) | `capabilities_follow_the_backends_a_solver_holds`, `missing_backend_is_reported_before_every_other_check` | the Python table stays in Python (D-S8-13) |
| S `test_check_expression_satisfiability_true_for_*`, `false_for_*`, `does_expression_imply_reports_true_*`, `holds_for_all_free_assignments_reports_true_*`, the `assert_*` decided tests | `satisfiability_asserts_the_expression_and_is_yes_when_sat`, `implication_asserts_a_counterexample_and_is_yes_when_unsat`, the three `universal_validity_*` stories, `real_solver_decides_the_three_questions` | the scripts, then the answer of each `check-sat` result |
| S `..._returns_none_on_unknown`, the `raises_undecidable_error_on_unknown` tests | `unknown_answers_unknown_with_the_reason_the_backend_gave` (3 cases) | the strict companions' error is the binding's |
| S `..._raises_key_error_for_missing_symbol_type`, `rejects_unmapped_considered_id`, `z3_question_raises_a_missing_symbol_type_ahead_of_ill_typedness`, `still_requires_a_sort_for_a_variable_beside_a_constant`, `needs_no_sort_for_a_considered_constant` | `missing_symbol_types_are_reported_before_ill_typedness_naming_every_identifier`, `missing_symbol_types_cover_both_sides_of_an_implication_but_no_constant` | |
| S `is_threaded_through_symbol_type`, `accepts_a_mapped_considered_id` | `universal_validity_without_considered_identifiers_asserts_the_negation`, `real_solver_decides_the_three_questions` | a considered identifier the expression does not mention quantifies nothing |
| S the `timeout` tests (threading and rejection) | `limits_reach_the_backend`, `process_is_killed_at_the_timeout_and_answers_unknown`, `real_solver_answers_unknown_at_the_timeout` | the value check stays Python (D-S8-8) |
| S the Boolean-coercion screen tests (5) | `boolean_coercion_hazard_*` (7) | |
| S the division, floor, modulo and power screen tests (15) | `partial_operation_hazard_*` (13 functions, 36 cases) | |
| S the mixed int/real equality screen tests (20) | `mixed_equality_hazard_*` (11 functions, 30 cases) | |
| S the native-constant screen tests (7) | `native_constant_hazard_*` (5), `native_constant_is_refused_by_the_screen_needing_no_symbol_type`, `user_constant_is_read_from_the_sorts_of_the_context` | |
| S the non-finite literal tests (5) | `non_finite_literal_hazard_*` (3 functions, 10 cases) | |
| S the ill-typedness order tests (15) | `ill_typedness_is_reported_before_a_hazard_on_either_side`, `numeric_root_is_ill_typed`, `symbol_typed_operand_in_a_boolean_position_is_ill_typed_despite_a_hazard` | |
| S `does_expression_imply_hazardous_premise_returns_none`, `screens_a_hazard_in_the_consequent`, `screens_a_nested_int_float_equality` | `hazard_answers_unknown_without_asking_the_backend`, `each_expression_is_screened_on_its_own_antecedent_first`, `mixed_equality_hazard_is_found_below_the_root` | |
| S the warning tests | `hazard_displays_one_lowercase_line_per_kind`, `hazard_node_is_the_refused_node_except_for_constants` | the warning is the binding's (D-S8-14) |
| S the simplification tests | `simplification_*` (4), `simplifier_*` (2) | the screen, the constant refusal and the substitution |
| none | `hazard_kinds_are_checked_in_order_whatever_the_node_order`, `hazard_of_a_kind_is_the_first_node_in_pre_order`, `hazard_screen_walks_a_deep_*_on_a_small_stack` (2), `hazard_screen_classifies_a_shared_dag_in_linear_time` | new: order, depth and sharing |
| Z `test_convert_expression_to_z3_expression`, `symbol_type_maps_to_correct_z3_sort` | `script_sets_the_logic_declares_each_constant_and_asserts_the_predicate`, `declarations_pair_each_identifier_with_its_symbol_and_sort_ordered_by_id` | |
| Z the literal tests | `boolean_literal_*`, `integer_literal_*`, `float_literal_is_its_exact_binary_rational` (8 cases), `decimal_literal_*` (4), `non_finite_float_is_refused` | |
| Z `test_z3_floor_divide_rejects_non_int_non_real_expression` | `integer_floor_division_divides_by_the_sign_of_the_divisor`, `integer_floor_modulo_takes_the_sign_of_the_divisor`, `real_floor_operations_go_through_to_int`, `integer_floor_division_by_a_real_literal_is_a_real_floor` | Y-2 |
| Z the piecewise tests (4) | `piecewise_is_a_right_folded_ite_whose_first_match_wins`, `integer_side_meeting_a_real_is_converted_with_to_real_in_every_position` | |
| Z `test_convert_call_expression_to_z3_rejects_unresolved_call` | `call_is_refused_naming_the_callee` (3 cases), `call_has_no_lowering_after_the_screens_pass` | Y-7 |
| Z `test_z3_rewrites_a_bool_operand_compared_against_an_integer`, `bool_coercion_yields_a_model_this_package_rejects` | `boolean_meeting_a_number_is_refused` (4 cases), `piecewise_mixing_a_boolean_and_a_number_is_refused` | Y-4 |
| Z the missing-sort, ill-typedness and native-constant order tests of the conversion | `missing_symbol_types_are_refused_first_naming_every_identifier_by_id`, `ill_typed_expression_is_refused_before_a_native_constant`, `native_constants_are_refused_before_the_nodes` | |
| Z the questions' own tests (17) | `solver_stories.rs`'s scripts and answers | the unscreened questions are deleted (D-S8-1) |
| none | `arithmetic_is_the_smt_lib2_operator` (5), `comparison_is_the_smt_lib2_predicate` (6), `connectives_are_n_ary_and_or_and_not`, `division_is_exact_over_the_reals`, the `power_*` stories (4), the `logic_*` stories (4), the symbol stories (2), `numeric_expression_is_named_by_the_value_constant`, `shared_*` (2), `deep_tree_lowers_and_prints_on_a_small_stack` | new: D-S8-6's table |
| P the Z3 properties (3) | `satisfiability_agrees_with_brute_force_over_a_small_domain`, `implication_agrees_with_brute_force_over_a_small_domain`, `universal_validity_agrees_with_brute_force_over_a_small_domain` | with a real solver |
| none | `screened_safe_tree_lowers_to_a_balanced_script_declaring_its_identifiers`, `ground_ordering_lowers_to_a_script_as_satisfiable_as_it_is_true` | new: the exact rational reference |
| none | `process_*` (10) | new: the process protocol |

### S8.3 status: the `z3` feature

`fhy-core` gains `z3 = { workspace = true, optional = true }` (`z3 =
"0.21"` in the workspace table) and `[features] z3 = ["dep:z3"]`. The
backend is `rust/fhy-core/src/solver/z3.rs`, `Z3Solver`, exported as
`fhy_core::solver::Z3Solver` under the feature, with `Z3TermError` for a
term z3 cannot build (only a malformed script holds one). It builds the
z3 terms from the script's arena in id order, so every argument is built
before the term applying it and a deep script builds without recursion,
and runs each check in a fresh context from `with_z3_config`, with the
timeout in the context's configuration. It uses z3's default solver, as
the z3-solver path does (`Solver.from_string` ignores `set-logic`), not
`Solver::new_for_logic`. Integers and rationals of any size are built
from their decimal text (`Int::from_str`, `Real::from_rational_str`), so
the `z3` crate's `num` feature, which would pull a second `num-bigint`,
is not needed. The all-features graph gains `z3`, `z3-sys`, `log` and
`pkg-config`, and `cargo deny check` passes unchanged.

**The build against the local libz3.** The machine's libz3 is 4.8.7 (with
headers, and no `pkg-config` file or `z3` executable). `z3-sys` 0.13
supports 4.13.3 and newer; with neither detection it assumes that minimum
and links `-lz3`, and the binaries do link against 4.8.7, since the few
symbols the backend uses exist there, but the enum tables `z3-sys` checks
against are those of 4.13.3. So, as D-S8-18's fallback says, the feature
is built against the libz3 4.16 of the z3-solver wheel:
`Z3_LIBRARY_PATH_OVERRIDE=<site-packages>/z3/lib`,
`Z3_SYS_Z3_VERSION=4.16.0`, and `LD_LIBRARY_PATH` for running (the
library's soname is `libz3.so.4.16`); `ldd` of the test binary shows the
wheel's library. The CI `rust` job does the same: it installs z3-solver
into a venv, exports those three variables and `FHY_SMT_SOLVER` (the
wheel's `z3 -in`), runs `cargo test --workspace --all-features`, and then
the solver's stories with the default features, where the process
backend runs against the z3 executable. `rust-msrv` keeps checking the
default features, and `cargo +1.85 check -p fhy-core --features z3`
passes too. The crate README documents the feature and the link methods.

**Tests.** `tests/it/solver/z3_stories.rs` (31, `cfg(feature = "z3")`):
25 decided cases of `test_solver.py` decided the same way, float
arithmetic in exact rationals (`(1e16 + 1.0) == 1e16` and `0.1 + 0.2 ==
0.3` are false, the decimal tenths true), implication and universal
validity, `unknown` at a 50 ms timeout, eight concurrent checks, and a
symbol redeclared with another sort in the next check. Under the feature
the solver-backed properties run on `Z3Solver`. The Rust gate: fmt,
clippy `-D warnings` with and without `--all-features`, 3,072 tests with
the feature and 3,040 without, doc `-D warnings`, deny, `cargo +1.85
check`.

### S8.4 status: the binding

The binding is `rust/fhy-core-py/src/solver.rs` with `solver/backends.rs`
(`SmtSolverBase`, `SimplifierBase`, the two adapters, and
`SmtLib2ProcessSolver`), `solver/values.rs` (`SmtScript`, `SatResult`,
and the conversions of symbol types and query kinds), `solver/facade.rs`
(`Solver`), `solver/error.rs` (D-S8-14's mapping and the warnings) and
`solver/state.rs` (the default solver of N-S8-2 (b)), exported from `_rs`
and declared in `_rs.pyi`. The expression binding lends it the registry
snapshot, `PyExpression::expression`, and a new
`materialize_substituted`; `refuse_unused_arguments` of the pass binding
is shared. Nothing in Python uses it yet, so the suite is unchanged
(7,327 passed), and the Rust gate passes (3,072 tests with the feature).

Choices made here:

- **`SmtScript.lower(expression, symbol_types=None)`** is a static method,
  the core's `SmtScript::lower` with its errors mapped (`KeyError`,
  `NonBooleanLogicalOperandError`, `NativeConstantLoweringError`,
  `TypeError`). `convert_expression_to_smtlib2` and
  `convert_expression_to_z3_expression` (S8.5) are built on it. A script
  is not picklable.
- **The Python `SmtSolver` adapter** hands the hook a new `SmtScript`
  holding a copy of the core script, whose `text` and `declarations` are
  built on first access, and `timeout_milliseconds` as an `int` or
  `None`. A hook's exception propagates as the same object, a
  `KeyboardInterrupt` included, and a result other than a `SatResult`
  raises `TypeError` naming the class (`Fake.check must return a
  SatResult, got int.`). A backend's `name` is read only when an error or
  a warning needs it.
- **Simplification keeps the objects.** Each `simplify_expression` pushes
  a frame on a thread-local stack with the input's object and the
  environment's value objects; the adapter materializes the substituted
  expression beside them, so the hook receives the input object itself
  when nothing is bound and the bound value objects in place, and the
  object the hook returns is the object the caller gets. A nested
  question inside a hook gets its own frame.
- **Every question runs detached** from the interpreter, as D-S8-11 asks
  for a native backend; a Python backend attaches again for its one call.
  The detached closure builds its `QueryContext` from owned values, since
  the context's lookups are trait objects that are not `Send`.
- **The order in a `Solver` method**: the capability (the core's
  `NoCapableBackend` as `SolverCapabilityError`), then
  `timeout_milliseconds` through the Python `validate_timeout_milliseconds`
  (a value at or above `2**64` raises the same `ValueError`), then the
  core's checks.
- **Warnings** are logged on `fhy_core.symbolic.solver` through
  `get_logger`: `<entry point>: <hazard text>: node <repr>; identifier
  sorts at that node: x::7: INT. The expression is not handed to the
  solver; bounding timeout_milliseconds cannot change this outcome.`, and
  `<entry point>: the backend z3 answered unknown (timeout)` for a
  backend's `unknown`, for the lenient and the strict entry points alike.
- **The strict companions' `UndecidableError`** keeps the phrase the
  Python tests match (`refused by the solver seam's hazard screen`) and
  the `hazard_screen` reason, and names the backend for `unknown`, whose
  reason is the backend's.
- **The default solver** is a `Mutex<Option<Py<Solver>>>`, unset until
  `fhy_core.symbolic.solver` sets it at import; `get_default_solver`
  raises `RuntimeError` before then.
- **`SatResult.status`** returns a member of the Python `SatStatus` of
  `fhy_core.symbolic.solver`, which S8.5 adds; the stub declares it as
  `str` until then.

### S8.5 and S8.6 status

The Python switch (7b2d574, marked breaking) left exactly the four
modules of the migration plan failing collection (6,228 passed); the test
commit after it migrates them and adds the interface suite. At the end of
S8.6: `pytest` 7,387 passed, `-m "not very_slow"` 7,420 passed, the
`property` session 281 passed, `lint` and `type_check` clean, and
`tests/test_rs_stub.py` green. The constraint and param tests pass
unchanged on the new solver, the six places in `test_constraint_system.py`
that patch the solver's functions included, and so do the solver tests
that force z3's `unknown` or observe its timeout by patching `z3.Solver`,
since the adapter drives a `z3.Solver` too.

Tests migrated in S8.6. None was skipped or deleted without a rewrite:

| Test | Now | Reason |
|---|---|---|
| `test_solver.py::test_every_solver_backend_has_a_capability_table_entry` | same name | reads `get_backend_capabilities` over every member, not the private table |
| `test_solver.py::test_simplify_expression_matches_direct_bridge_pipeline` | same name | compares with `SympySimplifier` (D-S8-1, D-S8-12) |
| `test_solver.py::test_check_expression_satisfiability_screens_a_conjunction_compared_to_an_int` | same name | the core's lowercase `boolean operand into a numeric context` (D-S8-14) |
| `test_sympy_pass.py::test_sympy_simplify_expression_accepts_an_immutabledict_environment` | same name | through `simplify_expression(..., backend=SolverBackend.SYMPY)`; the bridge's is gone |
| `test_sympy_pass.py::test_simplify_expression_refuses_to_compare_a_numeric_piecewise_with_a_boolean` (2 cases) | `test_simplify_expression_compares_a_bound_numeric_piecewise_with_a_boolean_strictly` | Y-9: the environment is substituted before lowering, so the piecewise has picked its number, which compares unequal to a Boolean under the IR's type-strict equality, as `1 == True` did before S8; the old raise came from lowering the piecewise with its identifier free |
| `test_cross_cutting.py`: the registration list | the list without `to_z3`, and `test_z3_lowering_is_no_registered_pass` | D-S8-1 |
| `test_cross_cutting.py`: the three rational-agreement tests | same names | compare `z3.simplify` of the parsed term: z3 parses `(/ p.0 q.0)` as a division term, not a numeral (Y-8) |
| `test_z3_pass.py::test_convert_expression_to_z3_expression` (27 cases) | same name | pinned by sort and meaning (an equivalence check), not shape; a real floor division stays real and a power is a product (Y-2, Y-3, Y-8) |
| `test_z3_pass.py::test_holds_for_all_free_assignments_maps_solver_result_to_satisfiability` (3 cases) | same name (6) | D-S8-7: with every identifier considered, the script asserts the expression and `sat` means it holds |
| `test_z3_pass.py::test_z3_floor_divide_rejects_non_int_non_real_expression` | the Rust floor-encoding stories | no Python floor helper is left |
| `test_z3_pass.py::test_convert_expression_to_z3_returned_mapping_is_immutable` | same name | through `convert_expression_to_z3_expression` |
| `test_z3_pass.py::test_z3_visit_identifier_rejects_invalid_symbol_type`, `test_z3_visit_literal_unsupported_value_raises` | `test_convert_expression_to_z3_rejects_an_invalid_symbol_type`, `..._rejects_a_value_that_is_no_expression` | no Python visitor; the arguments' `TypeError`s |
| `test_z3_pass.py::test_z3_converter_get_noop_output_raises` | `test_z3_solver_decides_a_script_and_is_named_z3` | the converter pass is gone (D-S8-1); the adapter is what the module defines |
| `test_z3_pass.py::test_expression_to_z3_converter_accepts_an_immutabledict_symbol_types`, `..._snapshots_symbol_types_at_construction` | `test_smt_script_lowering_accepts_an_immutabledict_symbol_types`, `test_smt_script_snapshots_symbol_types_when_it_is_lowered` | the lowering takes the symbol types |
| `test_z3_pass.py`: the bridge questions' `immutabledict` and numeric-root tests (8) | same names | through the solver's functions (D-S8-1) |
| `test_z3_pass.py::test_holds_for_all_free_assignments_logs_z3s_reason_for_unknown` | same name | on the solver's logger (D-S8-14) |
| `test_z3_pass.py::test_convert_single_case_piecewise_expression_to_z3_if` | same name | an `If`, compared by meaning: z3's parser writes `(- x)` as `-1*x` |
| `test_z3_pass.py::test_convert_call_expression_to_z3_rejects_unresolved_call`, `test_non_finite_float_literal_is_refused_rather_than_lowered` (3) | same names | the lowering's `TypeError`, no `PassExecutionError` (Y-7) |
| `test_z3_pass.py::test_z3_rewrites_a_bool_operand_compared_against_an_integer`, `test_z3_bool_coercion_yields_a_model_this_package_rejects` | `test_lowering_refuses_a_bool_operand_compared_against_an_integer`, `test_z3_parser_rewrites_a_bool_operand_compared_against_an_integer`, `test_z3_bool_coercion_yields_a_model_this_package_rejects` | Y-4: the lowering refuses the term; z3's SMT-LIB2 parser still coerces it, which is why the lowering must never emit one and the Boolean-coercion screen stays |
| `test_z3_pass.py::test_bridge_question_refuses_a_native_constant_it_would_decide_wrongly` | `test_question_refuses_a_native_constant_it_would_decide_wrongly` | the bridge's questions are gone; the solver's screen refuses the constant, with the reason `hazard_screen` |
| none | `tests/symbolic/test_solver_rust_binding.py` (57) | the interface suite |

The interface suite covers the test plan: the ABCs' abstract hooks,
their own and refused constructor arguments, `SmtLib2ProcessSolver` as a
native, registered `SmtSolver`, frozen values, `Solver`'s backends,
`can_answer` and `repr`; a Python `SmtSolver` receiving the script's text
and the timeout once per query, each `SatResult` as the answer, a wrong
result type, an exception and a `KeyboardInterrupt` propagating as the
same object, a nested question, the strict companions' reason, and the
timeout checks; a Python `Simplifier` receiving the input object itself
when nothing is bound and the environment's objects in place, its result
object returned, its errors, and a nested simplification; `SatResult`'s
values, status, reason, `repr` and pickles, and `SmtScript`'s parts; three
pinned scripts and the z3 conversion's identifier map; the adapters, one
object each, availability, capabilities, and, in subprocesses with
`sys.modules[package] = None`, `SolverBackendUnavailableError` and its
message and a fresh `import fhy_core` importing neither package; the lazy
re-exports; each row of D-S8-14 and both warnings; the process backend
driving a fake SMT-LIB2 program run with the interpreter, through a
`Solver` and directly; the default solver, replacing it for the module
functions, a constraint system and a param, and a named backend ignoring
it; and eight threads asking one solver with a Python and with the
process backend.

### S8.7 status: optional extras

Per N-S8-1 (a), `sympy` and `z3-solver` left the required dependencies:
`pyproject.toml` has the extras `fhy_core[z3]`, `fhy_core[sympy]` and
`fhy_core[solvers]`, the `test` dependency group installs both, and a new
`test-minimal` group, which `test` includes, holds the suite's tools
without them. `tests/conftest.py` skips a test marked `z3` or `sympy` when
its package is missing, and the new nox session `tests_minimal` installs
`test-minimal`, checks that neither package is importable, and runs the
suite (`-m "slow or not slow"`); CI runs it as the `tests-minimal` job on
Python 3.12, which `ci-ok` requires. The README documents the extras and
the session, and CONTRIBUTING the marking rule.

**How the marks were set.** The marks follow what a test reaches (D-S8-17),
found by a probe kept outside the repo: a pytest plugin that drops the
optional-backend skips and records, per test, whether it fails with
`SolverBackendUnavailableError` (or an `ImportError`) naming z3 or sympy.
A run with both packages missing reports only the first package a test
reaches, so the suite ran twice in the `tests_minimal` environment, with
hypothesis added, once with sympy and without z3-solver and once the other
way round; the union gave each test function's needs. Marks were added
where a function needs its package and removed where it ran without it,
and the probe was repeated until nothing changed. A test failing without a
package for another reason (an assertion about availability, a subprocess)
keeps its mark by hand. At the end, 395 collected tests carry `z3` and
1,104 `sympy` (1,411 either). The probe's own count before any change was
564 failures without both packages.

**Modules that need a package throughout** skip without it through
`pytest.importorskip` before importing it, and carry a module mark:
`test_z3_pass.py` (`z3`), `test_sympy_pass.py`, `test_sympy_natives.py`,
`test_sympy_pass_properties.py` (`sympy`) and `test_cross_cutting.py`
(both). `test_solver.py` imports `z3` inside the eight tests that patch
it and `SympySimplifier` inside its one test, so its screen tests run
without either package; `test_native_stories.py` and
`test_piecewise_properties.py` import the sympy bridge inside the tests
that use it.

Tests changed in S8.7 beyond their marks:

| Test | Now | Reason |
|---|---|---|
| `test_param_serialization_properties.py::test_derived_param_result_round_trips_through_every_format`'s `@example` of an interval intersection | `test_interval_param_intersection_round_trips_through_every_format` (`z3`) | the example was built at import, and an intersection of numeric operands asks the solver whether it is empty, so the module needed z3-solver to import |
| `test_bindings_evaluation.py`: the `z3` mark of one `pytest.param` of `_EQUATION_BACKED_BINDINGS_METHODS` | removed | that case reaches no z3 |
| `test_param_intersection_properties.py`'s module-level `z3` mark | on the four properties that reach z3 | the other two reach none |
| `test_solver_rust_binding.py::test_importing_fhy_core_imports_neither_sympy_nor_z3` | the same, and `test_lazy_bridge_export_imports_its_package_on_first_access` (`z3`) | the first half runs in the minimal session |

At the end of S8.7: `tests_minimal` 5,764 passed and 608 skipped, `pytest`
7,389 passed, `lint` and `type_check` clean.

### S8 status

S8 was implemented on 2026-09-26 in ten commits: the benchmarks and their
baseline (ab05802); the core's solver, test-first (aebf889); the `z3`
feature (8da2be5); the binding (e33df12) and a stub fix after it
(3e49df9); the Python switch, marked breaking (7b2d574); the migrated
tests and the interface suite (869e41a); the optional extras, the markers
and the minimal session, marked breaking (3559b79); the z3 adapter's
simple solver, which the benchmarks called for (c0af152); and these docs,
with the benchmark rows of the new API. No test was skipped or deleted
without a rewrite. At the end: `pytest` 7,389 passed, `-m "not
very_slow"` 7,422 passed, the `property` session 282 passed,
`tests_minimal` 5,764 passed and 608 skipped, `lint` and `type_check`
clean, `tests/test_rs_stub.py` green, `FORCE_COLOR=1 nox -s tests-3.13`
green (7,140 passed), and the Rust gate green (fmt, clippy `-D warnings`
with and without `--all-features`, 3,040 tests and 3,072 with the `z3`
feature, doc `-D warnings`, deny, `cargo +1.85 check` with and without
the feature, and the packaging checks). The `z3` feature is built
against the libz3 4.16 of the z3-solver wheel (S8.3); the CI workflow's
new steps, the `rust` job's z3 installation and the `tests-minimal` job,
first run on the next pull request.

### S8 benchmarks (before and after)

Median time per call of `benchmarks/test_solver.py`, `pytest
benchmarks/test_solver.py -n 0 --benchmark-only`, on the S0 machine with
Python 3.11.13 and pytest-benchmark 5.3.0. "Before" is ab05802, the S8.1
baseline's tree, exported with `git archive` under `target/` and built
there with the same z3-solver (4.16) and sympy (1.14); "after" is the
S8.8 tree in the benchmark session's environment. The two ran three times
each, interleaved, with a load average of 8 to 15 from other work on the
machine, and the table lists the best of the three medians; the "before"
column agrees with the S8.1 table within 8%.

| Benchmark | before | after | after / before |
|---|--:|--:|--:|
| `test_screen_of_a_deep_predicate` | 503.3 µs | 80.6 µs | 0.16 |
| `test_lower_to_z3_of_a_deep_tree` | 2.14 ms | 384.8 µs | 0.18 |
| `test_lower_to_smtlib2_of_a_deep_tree` | - | 68.2 µs | - |
| `test_check_satisfiability_of_bounds` | 1.63 ms | 443.6 µs | 0.27 |
| `test_check_satisfiability_of_a_conjunction_of_50_bounds` | 6.01 ms | 917.1 µs | 0.15 |
| `test_does_expression_imply_of_bounds` | 1.38 ms | 428.4 µs | 0.31 |
| `test_holds_for_all_free_assignments_with_a_witness` | 1.40 ms | 952.0 µs | 0.68 |
| `test_check_satisfiability_refused_by_the_screen` | 50.1 µs | 27.1 µs | 0.54 |
| `test_simplify_expression_of_a_ground_comparison` | 139.4 µs | 71.6 µs | 0.51 |
| `test_simplify_expression_symbolic` | 72.6 µs | 77.7 µs | 1.07 |
| `test_equation_constraint_evaluate_with_bindings` | 147.6 µs | 60.1 µs | 0.41 |
| `test_constraint_system_check_implication` | 1.39 ms | 445.0 µs | 0.32 |
| `test_nat_param_is_value_valid` | 147.5 µs | 61.5 µs | 0.42 |
| `test_int_param_intersection_feasibility` | 2.10 ms | 675.0 µs | 0.32 |
| `test_import_fhy_core` | 509 ms | 221 ms | 0.43 |

Every row is faster or within the 10% CONTRIBUTING allows, so the pattern
choice stands and no cost needs the maintainer (cross-cutting rule 5):

- **The first run after the switch** had four rows slower: the smallest
  satisfiability question 1.61 times (2.65 against 1.65 ms), the param
  intersection 1.48, and the two implications 1.14 and 1.18. The Rust
  part of a question costs a few microseconds (the lowering of `0 < x &&
  x < 10` takes 3 µs, its text 5 µs); the rest was z3's. A default
  `z3.Solver()` sets up its incremental, tactic-driven front end, about
  3 ms a check, and asserting `e` and answering `sat` took longer there
  than the bridge's closed `ForAll` encoding answering `unsat`. The adapter
  now checks with `z3.SimpleSolver()` (c0af152), the SMT kernel alone,
  which decides these scripts in 0.4 to 0.5 ms.
- **The screens** are 6 times faster: one Rust call over the tree, where
  five recursive Python walks repeated their classifications per node. The
  row now also lowers the question and calls a Python backend once.
- **The z3 lowering** of the deep tree is 5.6 times faster, the Rust
  lowering and one parse against a Python visitor call per node; the
  SMT-LIB2 text alone takes 68 µs.
- **The questions** are 1.5 to 6.6 times faster, the 50 bounds most, since
  the old encoding quantified every identifier; the witness question
  gains least, since its `forall` is z3's work either way.
- **Simplification** of a bound comparison, and the constraint and param
  value checks over it, are about twice as fast: the facade substitutes in
  Rust (Y-9) and sympy lowers a smaller tree. The symbolic `x + x - x`,
  which substitutes nothing, takes 1.07 times as long: sympy's own work
  dominates, and the binding adds the crossing into Rust and back into
  the Python adapter, the few microseconds the plan expected.
- **Importing `fhy_core`** takes 221 ms in a fresh interpreter, down from
  509 ms, since neither sympy nor z3 is imported (D-S8-16).

### S8 implementation notes

Choices the decisions left open, made while implementing S8.5 to S8.8
(S8.2's, S8.3's and S8.4's are in their own notes above):

- **The default solver's backends are deferred.** Its initial value holds
  `_DeferredSmtSolver(SolverBackend.Z3)` and
  `_DeferredSimplifier(SolverBackend.SYMPY)`, Python backends that resolve
  the named adapter on their first question and delegate to it, named
  `z3` and `sympy`. So importing `fhy_core.symbolic.solver` imports
  neither package, and a missing one raises
  `SolverBackendUnavailableError` from the question that needs it. The
  adapters are created once, by `functools.cache` (`_resolve_adapter`),
  and so is the `Solver` of each named member; `is_backend_available`
  tries to resolve the adapter.
- **A named backend keeps the Python capability text.** `backend=Z3` for a
  simplification still raises `Backend <SolverBackend.Z3: 'z3'> cannot
  answer ...`, checked against the static table before any solver is
  asked; a solver without a capable backend, the default one included,
  raises the core's `the backends of this solver cannot answer ...`.
- **The new names live in `fhy_core.symbolic.solver`:** `SatStatus`,
  `SolverBackendError`, `SolverBackendUnavailableError` (both
  `register_error`ed), `convert_expression_to_smtlib2`,
  `is_backend_available`, and the binding's classes and functions. The two
  bridges import the ABCs and the error from it; since the solver module
  imports them only when a question needs them, there is no import cycle.
- **`convert_expression_to_z3_expression`** reads a Boolean expression's
  term as the script's one assertion, and a named value's as the second
  argument of `(= value e)`; the identifier map holds `z3.Int`,
  `z3.Real` or `z3.Bool` constants of the declared symbols, which z3
  identifies with the parsed ones by name and sort.
- **z3's SMT-LIB2 parser is lenient**: `from_string` and
  `parse_smt2_string` coerce `(= true 1)` to `If(True, 1, 0) == 1` rather
  than refusing it, so the lowering must never write a Boolean where a
  number is required. The Boolean-coercion screen stays for that reason,
  and two characterization tests in `test_z3_pass.py` pin it.
- **The simple solver** reports running out of time as `canceled`; the
  adapter maps that reason to `timeout` when a timeout was set (D-S8-8).
  The Rust `Z3Solver` keeps the `z3` crate's default solver; switching it
  to the SMT kernel (`Tactic::new("smt").solver()`) would match the Python
  adapter, and is a possible follow-up, since the published wheels never
  enable the feature.
- **The sympy bridge** lost its `simplify_expression` and the helper only
  that function used; `SympySimplifier` lowers, simplifies with the
  unchanged best-effort cases, and lifts.
- **The stub** types `Solver.smt_solver` and `Solver.simplifier` as the
  Python ABCs, which every backend is an instance or a registered virtual
  subclass of, and `SatResult.status` as `SatStatus`.
- **The markers** were set by probing, as S8.7 records; the design's
  counts (474 `z3` marks, 76 unmarked users) came from the old solver, so
  the new counts (395 `z3`, 1,104 `sympy`) are not comparable one to one.

Left for later, as the design says:

- **D-S8-5's follow-up**: with Y-1 and Y-2, the partial-operation screen
  could narrow to what SMT-LIB2 cannot say (a zero or non-literal divisor,
  an unsafe exponent). That changes which questions are decided, so it is
  a separate change with its own tests.
- **The Rust CAS backend** of the next slice: SymPy through pyo3, behind
  an off-by-default `sympy` feature of `fhy-core`. The `Simplifier` trait
  takes a `SimplifyContext` (S8.2), so it can plug in, and gain what it
  needs from the context, without changing the trait or the facade.
- **Sessions and models** (`push`/`pop`, a `get_model`): non-goals of
  D-S8-8; a later `SmtSession` trait can add them.

## S10: terms

- **Status:** designed 2026-09-26 at ab05802, and implemented the same
  day; see "S10 status" below. D-S10-1 to D-S10-16 apply the policy the
  user already set and the user's direction for this slice ("port the
  `fhy_core.term` package to Rust"). The user resolved N-S10-1 as (a) and
  N-S10-2 as (b); see "S10 resolutions".
- **Pattern:** the logic moves into a new core module, `fhy_core::term`,
  which takes over `AlphaRenaming` from `fhy_core::expression` and adds
  traits for alpha equivalence, free identifiers, terms and binders.
  `AlphaRenaming` becomes P2. `BinderMixin` is P3 over the core's `Binder`
  trait, with a Python base instead of a Rust one (D-S10-7). The
  derived-equivalence engine moves into the binding (D-S10-8). The
  protocols, `AlphaEquivalenceMixin` and the field-role functions stay
  Python.
- **Scope.** `src/fhy_core/term/` (`alpha_equivalence.py`, `binder.py`,
  `derived_equivalence.py`, `__init__.py`). This revises the non-goal of
  `rust-workspace.md` §I.8 for `term` only; `constraint`, `param`, `types`
  and `symbol_table` stay unported Python (D-S10-13).
- **Coordination.** S8 and S9 run in parallel, and this branch is rebased
  onto `dev-rust` after them. "Coordination with S8 and S9" below lists the
  shared files S10 touches.

### Survey: the Python API

The package is 1,150 lines of pure Python. `__init__.py` (57 lines)
re-exports 16 names.

| File | Lines | Public names |
|---|--:|---|
| `alpha_equivalence.py` | 387 | `AlphaEquivalence`, `AlphaEquivalenceMixin`, `AlphaRenaming`, `is_identifier_mapping_alpha_equivalent_under` |
| `binder.py` | 161 | `HasFreeIdentifiers`, `Term`, `BinderMixin` |
| `derived_equivalence.py` | 545 | `EQUIVALENCE_METADATA_KEY`, `DerivedEquivalenceMixin`, `EquivalenceDerivationError`, `FieldComparator`, `compared_as_value`, `compared_as_reference`, `compared_as_binder`, `compared_with`, `excluded_from_equivalence` |

**`alpha_equivalence.py`.**

- **`AlphaEquivalence`**, a `runtime_checkable` protocol with
  `is_alpha_equivalent(other)` and `is_alpha_equivalent_under(other,
  renaming)`. Both return `False` rather than raise for an unrelated
  `other`, and the contract asks for an equivalence relation on
  well-formed terms.
- **`AlphaEquivalenceMixin(ABC)`**: `is_alpha_equivalent_under` is
  abstract, and `is_alpha_equivalent` calls it with
  `AlphaRenaming.empty()`.
- **`AlphaRenaming`**, a `@final` frozen dataclass of
  `_frames: tuple[immutabledict, ...]` (outermost first) and
  `_free_renaming: immutabledict`. It is equal by structure and hashable,
  and an empty frame counts.
  - `empty()` builds a new instance on every call.
  - `with_free_renaming(mapping)` and `extend(bindings)` refuse a map with
    a repeated value (``ValueError("`bindings` must be injective; got
    duplicate other-side values in mapping.")``). `extend` returns a new
    renaming with one more innermost frame.
  - `resolve(identifier)` returns the image in the innermost frame binding
    it, else the free image, else the identifier itself. It returns the
    very object stored in the map.
  - `are_identifiers_alpha_equivalent(left, right)` has the capture rule
    and follows the Rust rule since 5a7802c (S4.2's shadowing note).
  - Nothing checks that keys and values are `Identifier`s.
- **`is_identifier_mapping_alpha_equivalent_under(left, right,
  renaming)`** compares two identifier-keyed maps: equal sizes, keys
  resolved through the renaming without collision, equal key sets, then
  per pair the capture check and the values' `is_alpha_equivalent_under`,
  in the left map's order, stopping at the first mismatch.

**`binder.py`.**

- **`HasFreeIdentifiers`** (`get_free_identifiers() -> frozenset`) and
  **`Term`** (`AlphaEquivalence`, `HasFreeIdentifiers` and
  `substitute(replacements) -> Term`) are `runtime_checkable` protocols.
- **`BinderMixin(AlphaEquivalenceMixin)`** has four abstract hooks,
  `get_bound_identifiers`, `get_scoped_children`,
  `rename_bound_identifier(old, new)` and
  `rebuild_with_scoped_children(children)`. It derives three methods:
  - `is_alpha_equivalent_under`: the same concrete type, the same numbers
    of bound identifiers and of scoped children, then the children
    compared pairwise under `renaming.extend(dict(zip(self_bound,
    other_bound)))`. A refused frame answers `False`.
  - `get_free_identifiers`: the union over the children, minus the bound
    set.
  - `substitute`: replacements keyed by a bound identifier are dropped.
    With none left it returns `self`. Otherwise every bound identifier
    that is free in a replacement is renamed to a fresh
    `Identifier(name_hint)` through `rename_bound_identifier`, then each
    child is substituted and the node rebuilt.

**`derived_equivalence.py`** derives `is_structurally_equivalent` and
`is_alpha_equivalent_under` for a dataclass from its fields.

- **The plan.** It is built from `dataclasses.fields` on the first
  comparison of a class and cached forever in the module dict
  `_PLAN_CACHE`, keyed by the class. A field with `compare=False` or
  `excluded_from_equivalence()` is skipped.
- **The roles.** Each is stored under `EQUIVALENCE_METADATA_KEY` in the
  field's metadata:
  - `compared_as_value(key=None)` compares by `==`, through `key`;
  - `compared_as_reference()` compares by `==` structurally and by
    `are_identifiers_alpha_equivalent` in alpha mode;
  - `compared_as_binder(scopes_over=(...))` marks one `Identifier` or a
    sequence of them. Structurally the names compare as tuples. In alpha
    mode the two sides must have equal lengths, and each field in
    `scopes_over` is compared under the renaming extended by
    `dict(zip(...))`. Several binders scoping one field nest in field
    order, and a refused frame answers `False`;
  - `compared_with(comparator)` supplies a `FieldComparator`;
  - any other field uses the default dispatch.
- **The default dispatch**, in this order: `None` against anything by
  `is`; an `AlphaEquivalence` value by its method (in alpha mode); a
  `StructuralEquivalence` value by its method; a `tuple` or `list`
  against a `tuple` or `list` element-wise; a `bool`, `int`, `float`,
  `str`, `Enum` or `PartialEqual` value by `==`. Anything else raises
  `EquivalenceDerivationError`, naming the class and field, and whether
  the value sat in a sequence. The two protocol checks are `isinstance`
  against runtime-checkable protocols, which runs Python code for each
  check.
- **Other refusals.** A class that is not a dataclass, or a `scopes_over`
  name that is no field, raises `EquivalenceDerivationError` on the first
  comparison, with Python text that the tests match (`dataclass`,
  `"bdy" is not a field`, `payload`, `sequence element`).
- **Opting out.** Overriding either method by hand opts that one out, and
  `type(self) is type(other)` is required.
- **Recursion.** The walks recurse in Python, one or two frames per level
  of nesting.

**Probed at ab05802** (Python 3.11, `timeit`, best of five; indicative
only):

| Operation | Time |
|---|--:|
| `AlphaRenaming.empty()` | 789 ns |
| `extend` of one pair | 1.27 µs |
| `resolve` | 300 ns |
| `are_identifiers_alpha_equivalent` | 376 ns |
| `==` of two one-frame renamings | 1.20 µs |
| `hash` | 222 ns |
| `with_free_renaming` of 50 pairs | 8.72 µs |
| a `BinderMixin` lambda `\x. x z` against `\y. y z` | 4.94 µs |
| ten nested `BinderMixin` lambdas | 30.6 µs |
| `BinderMixin.substitute` that renames to avoid capture | 7.62 µs |
| a derived `\x. x` against `\y. y` (`compared_as_binder`, alpha) | 8.24 µs |
| the same term structurally | 4.57 µs |
| the mapping helper over 50 entries | 52.4 µs |
| `Param` alpha equivalence: integer, natural, integer between bounds | 33.2, 45.0, 54.8 µs |
| `Param` structural equivalence: integer, integer between bounds | 21.4, 35.8 µs |
| `create_integer_param_between(0, 10)` (dedupes its constraints structurally) | 77.1 µs |
| `EquationConstraint` structural, alpha | 4.59, 6.61 µs |
| `Expression.is_alpha_equivalent_under` of `x + z`, under one frame, under ten frames | 1.34, 9.91 µs |

The derived walks cost microseconds per node, most of it the protocol
`isinstance` checks. S7 measured the same plan comparing two functions in
51 µs, and the Rust comparison in 856 ns. An expression compared under a
renaming converts it on every call, in time linear in its frames.

### Survey: the Rust API

- **`fhy_core::expression::AlphaRenaming`** (`expression/alpha.rs`, 229
  lines) is S4.2's port of the Python class. It has a stack of injective
  frames over an injective free renaming, and these methods: `try_new`,
  `Default`, `enter_binder` (in place), `leave_binder`, `binder_depth`,
  `resolve`, `is_corresponding` (Python's
  `are_identifiers_alpha_equivalent`) and `is_empty`.
  - It derives `Clone`, `PartialEq` and `Eq`, but not `Hash`. A clone
    copies every frame's maps.
  - It has no read access to its frames or its free renaming.
  - Its errors, `NonInjectiveRenamingError` and `RenamingPart`, live in
    `expression/error.rs`.
  - Its rustdoc tells a caller that pairs parameter lists to refuse a
    repeated identifier.
- **`Expression`** has inherent `is_alpha_equivalent_under(&self, other,
  &AlphaRenaming)`, `free_identifiers() -> HashSet<Identifier>` and
  `substitute(&HashMap<Identifier, Expression>) -> Result<Expression,
  PiecewiseError>`. It has no traits for them; an expression binds
  nothing.
- **`fhy_core::tree`** has `Tree` (`children`, `rebuild_with_children`,
  `RebuildError`) over children of the node's own type, with iterative
  walks. A binder's scoped children are generally another type than the
  binder (a lambda's body is a term; a function's body is an expression),
  so `Binder` cannot be a `Tree`. It borrows `Tree`'s shape: a rebuild
  hook with an associated error.
- **`FunctionDefinition`** (S7) has no equality (D-S7-2). The binding
  compares two entries under a parameter frame itself (D-S7-10,
  `registry/entries.rs`).
- **The binding.** `expression/alpha.rs` (74 lines) converts a Python
  `AlphaRenaming` by reading its private `_frames` and `_free_renaming`.
  `node.rs` (`is_alpha_equivalent_under`) and `registry/entries.rs` call
  it on every comparison.
- **Rust tests.** `tests/it/expression/alpha_stories.rs` (652 lines, 43
  cases, including a test-local `Binders` term of nested one-parameter
  binders), `alpha_properties.rs` (167 lines, 3 properties), and the
  alpha cases of `node_stories.rs` and `properties.rs`.
- **Non-goal.** `rust-workspace.md` §I.8 listed porting `term` as a
  non-goal. The user's direction for S10 replaces that for `term`.

### Consumers and tests

**`src`.** No module outside the package uses `BinderMixin`, `Term`,
`HasFreeIdentifiers` or the mapping helper, except in docs and stubs.

| Module | Lines | Uses |
|---|--:|---|
| `symbolic/expression/core.py` | 807 | `Expression(_rs.Expression, ..., AlphaEquivalenceMixin, ...)`; its docs name `Term` and `AlphaRenaming` |
| `diagnostic.py`, `op_attribute.py`, `value_domain.py` | 284, 107, 107 | `AlphaEquivalenceMixin`, mixed into the Rust-backed tags beside their pyclass bases |
| `symbol_table.py` | 758 | the `SymbolTableFrame` family (frozen dataclasses): `DerivedEquivalenceMixin`. `SymbolTable.is_structurally_equivalent` compares frames through it |
| `symbolic/constraint/core.py` | 1,117 | the `Constraint` family: `DerivedEquivalenceMixin`; the set constraints' `variable` is `compared_as_reference()`, their `values` `compared_as_value(key=_wrap_member_collection)` |
| `symbolic/constraint/system.py` | 848 | `ConstraintSystem`: `DerivedEquivalenceMixin` |
| `symbolic/param/core.py` | 2,175 | `Param`: `variable` is `compared_as_binder(scopes_over=("constraint_system",))`; `ParamAssignment`: `value` is `compared_as_value()`. Construction dedupes constraints with `is_structurally_equivalent`, quadratically |
| binding: `expression/alpha.rs`, `node.rs`, `registry/entries.rs` | | the conversion above |
| `_rs.pyi` | | types `AlphaRenaming` and `Term` |
| `benchmarks/test_expression.py`, `test_registry.py` | | `AlphaRenaming.with_free_renaming`, `AlphaRenaming.empty()` |

`types` imports nothing from the package. Its types reach the derived
engine only as field values of symbol-table frames, which compare
through their own `is_structurally_equivalent`. The constraint and param
slices come later. Nothing in S10 changes their code (D-S10-13).

**Python tests.** Counts are collected tests.

| File | Lines | Collected | What it pins |
|---|--:|--:|---|
| `test_alpha_equivalence.py` | 863 | 59 | the renaming, the mapping helper, the protocol and mixin, toy binders |
| `test_binder.py` | 288 | 17 | `BinderMixin` over a toy lambda calculus |
| `test_derived_equivalence.py` | 858 | 58 | roles, dispatch, errors, laws, depth 500 |
| `test_derived_equivalence_properties.py` | 203 | 4 | the laws over random value, reference and sequence holders |
| `symbolic/expression/test_term.py` | 325 | 28 | expressions as `Term`s, under a renaming |
| `symbolic/param/test_alpha_equivalence.py` | 401 | 25 | `Param` and `ParamAssignment` equivalence |
| `symbolic/constraint/test_structural_equivalence.py` | 388 | 41 | constraint equivalence, `EquivalenceDerivationError` |
| `test_symbol_table.py` | 717 | 48 | frame and table equivalence (7 calls) |
| also `test_constraint_system.py`, `test_ordering_key.py`, `test_param_intersection.py`, `expression/test_core.py`, `test_core_properties.py`, `test_registry.py`, `test_registry_rust_binding.py`, `test_pickle_round_trips_properties.py`, `test_serialization_pins.py` | | | one to 14 equivalence calls each |

The term tests use only public names. No test reads `_frames`,
`_free_renaming` or `_PLAN_CACHE`, or pins `FrozenInstanceError` for a
renaming.

**Benchmarks.** Only the S4.1 row
`test_alpha_equivalence_under_free_renaming_of_deep_trees` and the S7 row
`test_registered_function_alpha_equivalence` touch the package.

### Divergences visible from Python

| # | Python today | After S10 |
|---|---|---|
| Z-1 | `AlphaRenaming` is a frozen dataclass. A mutation raises `FrozenInstanceError`, the private fields are readable, the `repr` is the dataclass's, and pickling is the default | a final, frozen Rust-backed class: `FrozenMutationError` (as X-15), no private fields, `repr` `AlphaRenaming(frames=[{x::7: y::8}], free_renaming={})`, and it pickles as a call |
| Z-2 | Keys and values may be any hashable object | `Identifier`s only; anything else raises `TypeError` in S2's style. The same holds for bound identifiers a hook or a binder field returns |
| Z-3 | `empty()` builds a new instance | returns one shared instance, which is immutable |
| Z-4 | A binder list that repeats an identifier pairs by `dict(zip(...))`. The last pairing wins on the left, and a repeat on the right is not injective. So `\x x. x` matches `\a b. b` but not the reverse (probed) | a list that repeats an identifier, on either side, pairs with nothing (N-S10-2 (b)), so `\x x. x` matches no binder, itself included |
| Z-5 | Messages of a refused map: `` `free_renaming` must be injective; got duplicate other-side values in mapping. `` | the core's text, `a free-identifier renaming must be injective, but more than one identifier maps to y::8`, or `a binder frame ...`, as expressions already raise it. The tests match `injective` |
| Z-6 | The derived walks and `BinderMixin` recurse in Python | nested derived values are walked on the heap, at any depth. A hand-written method or a hook still recurses in Python |
| Z-7 | `hash` values | different values, still equal for equal renamings |

Unchanged in meaning: resolution order and the capture rule; the
protocols and the mixins' contracts; each role, the dispatch order, the
opt-outs, the plan cache's lifetime and every `EquivalenceDerivationError`
text; the capture-avoiding substitution (a fresh identifier with the old
name hint) and `substitute` returning `self` when nothing applies;
`extend` returning a new renaming; `resolve` returning the objects given.

### Pattern choice

- **Core: `fhy_core::term`.** The renaming, the binder algorithms and the
  mapping comparison are logic. A Rust IR with binders will need them,
  so they live in the core with Rust tests (decision 2).
- **P2: `AlphaRenaming`.** It is a value without registry state, but
  Rust code holds and extends it. The binder engine, the derived engine
  and expressions all read it, and S4.3a's per-comparison conversion goes
  away once it is Rust-backed. It pays a crossing per `resolve` from
  Python, which S4.3a named as the cost of this choice. After S10 only
  hand-written leaf hooks call it from Python; the machinery that called
  it per identifier runs in Rust (D-S10-5).
- **P3 with a Python base: `BinderMixin`** (D-S10-7). Python binders are
  driven through an adapter implementing the core's `Binder` trait.
- **Binding engine: `DerivedEquivalenceMixin`** (D-S10-8). The plan is
  reflection over Python dataclasses, so it lives in `fhy-core-py`, not in
  the core. The walk runs in Rust and compares Rust-backed values without
  calling Python.
- **Plain Python:** the protocols (`AlphaEquivalence`,
  `HasFreeIdentifiers`, `Term`, `FieldComparator`),
  `AlphaEquivalenceMixin`, the role functions and
  `EQUIVALENCE_METADATA_KEY`, and `EquivalenceDerivationError`.

**Benchmark plan: `benchmarks/test_term.py` (S10.1).** It is written
against the public API only, with its own toy classes: the lambda
calculus of `test_binder.py` for `BinderMixin`, and derived `Var`, `Lam`,
`Const` and `Add` dataclasses. The baseline measures today's Python
package.

| Benchmark | Measures |
|---|---|
| `test_alpha_renaming_empty`, `_with_free_renaming[1]`, `[50]` | construction |
| `test_alpha_renaming_extend[depth_1]`, `[depth_10]` | `extend`, and its cost over a deep stack (D-S10-3's shared frames) |
| `test_alpha_renaming_resolve[frame]`, `[free]`, `[identity]` | the lookup a hand-written hook makes |
| `test_are_identifiers_alpha_equivalent[frame]`, `[capture]` | the correspondence a hand-written hook makes |
| `test_alpha_renaming_eq`, `_hash` | value semantics |
| `test_binder_alpha_equivalence[flat]`, `[nested_10]` | `BinderMixin` driving Python hooks |
| `test_binder_free_identifiers`, `test_binder_substitute[no_capture]`, `[capture]` | the other two derived methods |
| `test_derived_structural_equivalence_of_a_deep_tree`, `test_derived_alpha_equivalence_of_a_deep_tree` | 100 nested `Add`s: the walk |
| `test_derived_alpha_equivalence_of_binders` | `\x. x` against `\y. y` through `compared_as_binder` |
| `test_derived_equivalence_over_expressions` | a derived holder of a 100-operation expression, alpha under a binder: the native path |
| `test_mapping_helper[50]` | the helper over 50 entries |
| `test_param_alpha_equivalence[integer]`, `[natural_between]`, `test_param_structural_equivalence` | the consumers that ask most |
| `test_param_construction_between_bounds` | construction, which dedupes constraints structurally |
| `test_constraint_structural_equivalence`, `test_symbol_table_structural_equivalence` | the other consumers |
| `test_expression_alpha_equivalence_under_frames[1]`, `[10]` | expressions under a framed renaming: the conversion S10 removes |

Rerun, not added: `test_alpha_equivalence_under_free_renaming_of_deep_trees`
(`test_expression.py`) and `test_registered_function_alpha_equivalence`
(`test_registry.py`). The verdict follows cross-cutting rule 5. The paths
at risk:

- `resolve` and `are_identifiers_alpha_equivalent` from Python. Each now
  crosses into the extension and reads an `Identifier`'s id (P1), where
  today it runs 300 to 380 ns of Python;
- `BinderMixin` comparisons, which now call their Python hooks from Rust
  and build a Python renaming object per binder level;
- `extend` over a deep stack, which must not copy the frames' maps.

### Decisions (proposed 2026-09-26)

Each names the policy it follows:

- D-S4-1: Rust semantics where the two differ;
- D-S4-2: Python names where the meaning is the same;
- "no fallback";
- "tests rewritten, not skipped";
- the crate's conventions in `rust-workspace.md` Part I: one public path
  per item, the layering (§I.2, CONTRIBUTING "Module paths follow Rust
  layering"), `#[non_exhaustive]` errors with one-line lowercase
  `Display` (I.3 rule 3), the naming rules (I.3 rule 5), a single crate
  (decision 13), and no global state beyond identity;
- the binding patterns P1 to P3 and cross-cutting rules 5 to 7;
- the direction: port `fhy_core.term` to Rust.

Where a decision follows an earlier slice's decision or note, it says so.

- **D-S10-1: one implementation, no fallback** ("no fallback"; D-S7-1).
  These are deleted, not kept beside the Rust path:
  - the `AlphaRenaming` dataclass and `_check_injective`;
  - the Python body of the mapping helper;
  - `BinderMixin`'s three derived bodies;
  - `derived_equivalence.py`'s role dataclasses, the value and reference
    comparators, `_Plan`, `_build_plan`, `_auto_structural`,
    `_auto_alpha` and `_inference_error`.

  The Python modules keep the protocols, the mixin classes with their
  abstract hooks and one-line delegating methods, the role functions, the
  metadata key, the error class and the plan cache dict (D-S10-8).
- **D-S10-2: a new core module, `fhy_core::term`, which takes over
  `AlphaRenaming`** (one public path; the layering; I.3 rule 6, since the
  crate is unpublished; the direction). `AlphaRenaming`,
  `NonInjectiveRenamingError` and `RenamingPart` move out of `expression`
  ("Errors belong to their module"). The path
  `fhy_core::expression::AlphaRenaming` goes, and `expression` imports
  `term`. `term` depends only on `identifier`, so it joins layer 4 beside
  `tree` in CONTRIBUTING's list, and its row maps `fhy_core.term` to
  `fhy_core::term`. `alpha_stories.rs` and `alpha_properties.rs` move to
  `tests/it/term/`. The sketch below is settled test-first in S10.2, as
  D-S7-2's was:

  ```rust
  // fhy_core::term
  #[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]   // frames shared behind Arc (D-S10-3)
  pub struct AlphaRenaming { /* private */ }
  impl AlphaRenaming {
      pub fn try_new<S: BuildHasher>(free_renaming: HashMap<Identifier, Identifier, S>) -> Result<Self, NonInjectiveRenamingError>; // moved
      pub fn enter_binder<S: BuildHasher>(&mut self, bindings: HashMap<Identifier, Identifier, S>) -> Result<(), NonInjectiveRenamingError>; // moved
      pub fn enter_binders(&mut self, left: &[Identifier], right: &[Identifier]) -> Result<(), BinderPairingError>; // new: pairs two binder lists by position (N-S10-2)
      pub fn extended<S: BuildHasher>(&self, bindings: HashMap<Identifier, Identifier, S>) -> Result<Self, NonInjectiveRenamingError>; // new: Python's `extend`
      pub fn leave_binder(&mut self) -> bool;                                        // moved, as are
      pub fn binder_depth(&self) -> usize;                                           // binder_depth,
      pub fn resolve<'a>(&'a self, identifier: &'a Identifier) -> &'a Identifier;     // resolve,
      pub fn is_corresponding(&self, left: &Identifier, right: &Identifier) -> bool;  // is_corresponding
      pub fn is_empty(&self) -> bool;                                                 // and is_empty
      pub fn free_renaming(&self) -> impl ExactSizeIterator<Item = (&Identifier, &Identifier)> + '_; // new: read views,
      pub fn frames(&self) -> impl ExactSizeIterator<Item = BinderFrame<'_>> + '_;                    // outermost first
  }

  pub trait AlphaEquivalence {
      fn is_alpha_equivalent_under(&self, other: &Self, renaming: &AlphaRenaming) -> bool;
      fn is_alpha_equivalent(&self, other: &Self) -> bool { self.is_alpha_equivalent_under(other, &AlphaRenaming::default()) }
  }
  pub trait FreeIdentifiers { fn free_identifiers(&self) -> HashSet<Identifier>; }
  pub trait Term: AlphaEquivalence + FreeIdentifiers + Clone {
      type SubstituteError;
      fn substitute<S: BuildHasher>(&self, replacements: &HashMap<Identifier, Self, S>) -> Result<Self, Self::SubstituteError>;
  }
  /// A node binding identifiers over its scoped children.
  pub trait Binder: Clone {
      type Child: Term;
      type RebuildError: From<<Self::Child as Term>::SubstituteError>;
      fn bound_identifiers(&self) -> &[Identifier];
      fn scoped_children(&self) -> &[Self::Child];
      fn rename_bound_identifier(&self, old: &Identifier, new: Identifier) -> Result<Self, Self::RebuildError>;
      fn rebuild_with_scoped_children(&self, children: Vec<Self::Child>) -> Result<Self, Self::RebuildError>;
      // provided: what BinderMixin derives today
      fn is_binder_alpha_equivalent_under(&self, other: &Self, renaming: &AlphaRenaming) -> bool;
      fn binder_free_identifiers(&self) -> HashSet<Identifier>;
      fn substitute_avoiding_capture<S: BuildHasher>(&self, replacements: &HashMap<Identifier, Self::Child, S>) -> Result<Self, Self::RebuildError>;
  }
  pub fn is_mapping_alpha_equivalent_under<V: AlphaEquivalence, S: BuildHasher>(
      left: &HashMap<Identifier, V, S>, right: &HashMap<Identifier, V, S>, renaming: &AlphaRenaming) -> bool;

  #[non_exhaustive] pub struct NonInjectiveRenamingError { /* image, part */ }       // moved
  #[non_exhaustive] pub enum RenamingPart { FreeRenaming, BinderFrame }                // moved
  #[non_exhaustive] pub enum BinderPairingError { ArityMismatch { left: usize, right: usize }, RepeatedIdentifier(Identifier) }  // N-S10-2 (b)
  ```

  - The traits use Rust names (I.3 rule 5: no `get_`), and Python keeps
    its names (D-S4-2).
  - The provided `Binder` methods have their own names, so that a term
    enum with a binder variant can call them from its `AlphaEquivalence`
    and `Term` impls.
  - `Expression` implements `AlphaEquivalence`, `FreeIdentifiers` and
    `Term` (with `SubstituteError = PiecewiseError`), forwarding to its
    inherent methods.
- **D-S10-3: `AlphaRenaming` gains shared frames, `Hash` and read views**
  (S4.2's proposals 2 and 3; D-S4-2, since Python's renaming is hashable
  and `extend` copies only a tuple of references).
  - The frames and the free renaming sit behind `Arc`s, so a clone or
    `extended` costs one reference count per frame, and never copies a
    map.
  - `Hash` is order-independent within each map and ordered across the
    frames, consistent with the derived `Eq`: an empty frame counts, and
    frame order counts.
  - The read views serve the binding's `repr`, pickling and `resolve`
    (D-S10-5).
  - The existing methods keep their signatures and meaning, so the S4.2
    tests pass unchanged after the move.
- **D-S10-4: the binder algorithms are the core's** (the direction;
  decision 2; "tests rewritten, not skipped"). `Binder`'s provided methods
  implement exactly what `BinderMixin` derives today:
  - the arity and child-count checks, then the pairing of the bound
    lists, with `enter_binders`, and the children compared under it;
  - free identifiers as the children's union minus the bound set;
  - substitution that drops bound keys, returns a clone of `self` (the
    same handle) when nothing applies, renames each captured binder to
    `Identifier::new(name_hint)`, then substitutes the children and
    rebuilds.

  S4.2's test-local `Binders` term becomes an implementation of `Binder`
  in the tests, over a test-local lambda calculus. `is_mapping_alpha_equivalent_under`
  ports the mapping helper's algorithm.
- **D-S10-5: `AlphaRenaming` is P2, under its Python name and API**
  (decision 2; D-S4-2; S2 to S7 practice; Z-1 to Z-3, Z-5, Z-7). This
  retires S4.3a's P1 conversion.
  - `_rs.AlphaRenaming` is a `#[pyclass(frozen)]` without `subclass`, so
    it is final. `fhy_core.term.AlphaRenaming` is that class itself: it
    has no Python base to mix in, so it needs no thin subclass. It is a
    virtual `FrozenMixin`, and a mutation raises `FrozenMutationError`.
  - It keeps `empty()`, `with_free_renaming(mapping)`, `extend(bindings)`,
    `resolve(identifier)`, `are_identifiers_alpha_equivalent(left,
    right)`, `==` and `hash`. `empty()` returns one shared instance.
  - It keeps each image's Python `Identifier` object beside the Rust
    frames, keyed by id, so `resolve` returns the object that was given,
    as today. An identifier that nothing maps is returned itself.
  - Mapping arguments must hold `Identifier`s (`TypeError`). A refused
    map raises `ValueError` with the core's text.
  - `repr` lists the frames and the free renaming. Pickles are a call of a
    private class method with the free renaming and the frames. (A
    class method, since the stub test admits no private module-level
    function, as S7 found.)
  - `_rs.Expression.is_alpha_equivalent_under` and
    `RegisteredFunction.is_alpha_equivalent_under` borrow the Rust
    renaming from the pyclass, so nothing is converted per comparison.
    `expression/alpha.rs` is deleted, and a non-renaming argument still
    raises S4.3a's `TypeError`.
- **D-S10-6: the protocols and `AlphaEquivalenceMixin` stay Python**
  (P2's note on Python protocols; S4.3a's "no protocol metaclass").
  `AlphaEquivalence`, `HasFreeIdentifiers`, `Term` and `FieldComparator`
  are typing vocabulary with no logic, and `isinstance` against them must
  keep working. `AlphaEquivalenceMixin` is stateless, and the Rust-backed
  tags and expressions mix it in beside their pyclass bases. Its default
  `is_alpha_equivalent` passes the shared empty renaming.
- **D-S10-7: `BinderMixin` is P3 with a Python base** (P3; D-S8-11 for
  the adapter rules; N-S10-1).
  - `BinderMixin(AlphaEquivalenceMixin, ABC)` keeps its four abstract
    hooks, under their Python names. Its three derived methods call
    public `_rs` functions, which drive the node through an adapter
    implementing the core's `Binder`.
  - The adapter reads the node's bound identifiers and scoped children
    once, through the hooks, as a view. It calls `rename_bound_identifier`
    and `rebuild_with_scoped_children` only when a substitution needs them.
  - A child is compared, asked its free identifiers and substituted
    through its own Python methods. An `_rs.Expression` child, whose
    class does not override those methods, is handled in Rust instead,
    with the same answer.
  - The base is Python, not a `#[pyclass(subclass)]` as P3 describes. The
    users are frozen dataclasses. A Python class can have only one native
    base layout, so a pyclass base would stop a binder class from also
    being a Rust-backed class, or from mixing in another one. The mixin
    holds no state. (The same reason keeps `AlphaEquivalenceMixin`
    Python.)
  - The P3 adapter rules hold:
    - an exception a hook raises propagates as the same object, and a
      `KeyboardInterrupt` passes through;
    - a hook result of the wrong type raises `TypeError` in S2's style
      (Z-2);
    - a child's answer is read by truthiness, as `all(...)` reads it;
    - a comparison hook that raises stops the comparison where Python's
      `all(...)` stopped it: the adapter keeps the first error, answers
      `false` so the core stops, and the binding raises the error. This
      is S4's deferred-error pattern, and it keeps the core traits
      infallible.
- **D-S10-8: the derived-equivalence engine moves into the binding**
  (decision 2: logic-rich and slow in Python, see the probes; D-S4-2 for
  its meaning; CONTRIBUTING's process-global rule; N-S10-1).
  - `DerivedEquivalenceMixin` keeps its bases and its two methods, which
    call public `_rs` functions.
  - The binding builds a class's plan in Rust from `dataclasses.fields`
    on its first comparison: the roles, the binders and their scopes, and
    whether the class overrides either method. The error texts are
    Python's.
  - The plan is stored in the module's `_PLAN_CACHE` dict, as today. That
    dict is Python module state, which the binding reads through a
    write-once import, so the binding adds no static with interior
    mutability.
  - The walk keeps the dispatch order and the role semantics. It compares
    these values natively, without the protocol checks and without calling
    their Python comparison methods, with the answers those methods give:
    `None`; an `Identifier`, by its id; an `_rs.Expression`, by the core's
    `==` or `is_alpha_equivalent_under`; a value of exactly `bool`, `int`,
    `float` or `str`, by the interpreter's `==`; and a nested derived
    value whose class does not override the method. It walks nested
    derived values on its own stack.
  - Every other value is compared through Python, as today: user
    comparators, `key` normalizers, hand-written methods, the
    `StructuralEquivalence` and `AlphaEquivalence` capability checks, and
    `==`.
  - It carries Rust renamings, and builds an `AlphaRenaming` object only
    when a Python method or comparator must receive one.
  - Field reads, `==` and user hooks run with the interpreter attached.
    An exception propagates as the same object. `EquivalenceDerivationError`
    keeps its Python class and texts, as D-S7-12 kept the text of errors
    that have no core counterpart.
  - The engine is binding code, not core. The plan is reflection over
    Python dataclasses. The Rust counterpart of "derive equivalence from
    the fields" would be a derive macro, which needs a second crate and
    `syn`/`quote` (decision 13 keeps one crate), and no Rust type needs it
    yet. Rust IR implements the traits by hand, or with `Binder`.
- **D-S10-9: the roles are `_rs` values** (D-S4-2). `compared_as_value`,
  `compared_as_reference`, `compared_as_binder`, `compared_with` and
  `excluded_from_equivalence` keep their signatures. Each returns
  `{EQUIVALENCE_METADATA_KEY: role}`, where `role` is a frozen
  `_rs.EquivalenceRole` the plan builder reads without calling Python.
  `field(compare=False)` is still honored.
- **D-S10-10: a binder list that repeats an identifier pairs with
  nothing** (N-S10-2 (b); D-S4-1; S4.2's `enter_binder` note; D-S7-5). One
  rule, in `AlphaRenaming::enter_binders`, covers `Binder`, `BinderMixin`
  and `compared_as_binder`: two lists of different lengths are refused
  (`ArityMismatch`), and so is a list that repeats an identifier on either
  side (`RepeatedIdentifier`). A refused pairing answers "not equivalent",
  never an error, as a refused frame does today. So a term with a
  repeated binder is alpha-equivalent to nothing, itself included, and
  the laws (reflexivity, and "structurally equivalent implies
  alpha-equivalent") hold for terms whose binder lists repeat no
  identifier. Every law test and property states that as an explicit
  precondition on the terms it generates, not as a skip. This is the rule
  `enter_binder`'s rustdoc already asks of a caller pairing lists, and the
  one D-S7-5 applies to function parameters; the rustdoc now points to
  `enter_binders`.
- **D-S10-11: the mapping helper is the core's function under the Python
  name** (D-S4-2). `is_identifier_mapping_alpha_equivalent_under` is an
  `_rs` function over the core's rule. It compares the values in the left
  mapping's iteration order, as today, so which value hook raises first
  stays deterministic. It uses D-S10-7's native paths and deferred
  errors.
- **D-S10-12: errors** (D-S4-1; D-S7-12; CONTRIBUTING "Errors belong to
  their module").
  - `NonInjectiveRenamingError` raises `ValueError` with the core's text,
    as S4.3a maps it.
  - `BinderPairingError` never reaches Python: it means "not equivalent".
  - Argument types raise `TypeError` in S2's style.
  - `EquivalenceDerivationError` keeps Python's texts.
  - Exceptions from user code propagate unchanged.
- **D-S10-13: consumers keep their code** (D-S4-2; §I.8 for `constraint`,
  `param`, `types` and `symbol_table`). The expression core, the tags,
  `symbol_table`, `constraint` and `param` change nothing: they reach the
  Rust engine through the mixins and roles they already use. When the
  constraint and param slices make those classes Rust-backed, they can
  implement equivalence on the core traits and drop the mixin, as
  `RegisteredFunction` did (D-S7-10).
- **D-S10-14: `RegisteredFunction`'s binder equivalence stays in the
  binding** (D-S7-10). It reads the P2 renaming (D-S10-5) and pairs the
  parameters with `enter_binders`. The core still gains no equality for
  definitions: `FunctionDefinition` refuses repeated parameters (D-S7-5),
  so its pairing never meets N-S10-2's case. Implementing `Binder` for
  `FunctionDefinition` and `ComposedFunction` would unify the two, and is
  left as a follow-up.
- **D-S10-15: the Rust tests specify the core first** (the tests rule;
  S7.2's practice). The new traits, `enter_binders`, `extended`, `Hash`
  and the views are specified by Rust tests written against `todo!()`
  stubs, with a traceability table from the Python term tests. It extends
  S4.2's table.
- **D-S10-16: the Python tests are rewritten, not skipped** (the tests
  rule). The behavioral tests stay and change only where Z-1 to Z-7 or
  N-S10-2 change what they pin, each change recorded with its reason, as
  S4.4 to S8 did.

### Needs the user

- **N-S10-1 (resolved 2026-09-26 by the user, as option (a)): whether
  Rust may drive Python per node for the term mixins.** The direction says to port the package. P3's granularity rule
  says "Python callbacks happen per pass hook, never per tree node". But
  `BinderMixin`'s hooks and the derived fields are per node by nature.
  D-S10-7 has Rust call a binder's four hooks and its children's methods.
  D-S10-8 has Rust read each dataclass field, and call user comparators
  and hand-written methods, per node. The two conflict for this slice.
  - (a) **Port both engines** (D-S10-7, D-S10-8). This is recorded as the
    term package's exception to the granularity rule. Rust compares
    Rust-backed values natively and walks nested derived values itself. It
    calls into Python only for what only Python can answer: hooks, field
    reads, comparators and hand-written methods. No Python walk stays
    beside the Rust one.
  - (b) **Port only the renaming, the pairing rule, the mapping helper
    and the core traits.** The `BinderMixin` and derived walks stay
    Python over the P2 renaming. The core `Binder` then repeats
    `BinderMixin`'s algorithm for Rust IR, which leaves two copies of one
    algorithm. The derived plan keeps its cost: 33 to 55 µs per `Param`
    comparison, and 77 µs to build a bounded integer param.
  - (c) **(a) for the derived engine only.** `BinderMixin`, which no
    module in `src` uses, stays Python over the core-backed renaming and
    pairing.

  Recommendation: (a). It is the only option with one implementation of
  each algorithm. The derived engine is where the time goes (S7 measured
  51 µs against 0.86 µs for the same comparison). Each call into Rust
  answers one comparison that Python asked for, as a pass hook answers one
  run.
- **N-S10-2 (resolved 2026-09-26 by the user, as option (b)): a binder
  list that repeats an identifier** (Z-4). Python's
  rule is asymmetric. The Rust core has only a doc line: `enter_binder`'s
  rustdoc tells a caller that pairs lists to refuse a repeated name.
  D-S7-5 refuses one at construction for functions. Neither fixes what a
  comparison answers.
  - (a) **Positional shadowing.** The last occurrence of an identifier in
    a list binds it, as nested binders would. `enter_binders` pairs the
    positions that are last on both sides. An identifier whose last
    position the other side shadows is bound with no partner, so it
    corresponds to nothing.
    - `\x x. x` matches `\a b. b` and itself, and `\x y. x` matches
      neither.
    - The relation stays symmetric and reflexive, and "structurally
      equivalent implies alpha-equivalent" holds for every term.
    - A frame gains one-sided bindings, a small private change to the
      core's frame type. `resolve` of an identifier bound with no partner
      returns the identifier itself, and `is_corresponding` refuses it.
      This is S4.2's de Bruijn reading, extended to one list.
  - (b) **Refusal.** A list that repeats an identifier, on either side,
    pairs with nothing. The relation is symmetric, and it matches the
    rustdoc note and D-S7-5. But `\x x. x` is not alpha-equivalent to
    itself, which breaks reflexivity and "structural implies alpha" for
    such terms.
  - (c) **Keep Python's rule.** The relation is asymmetric, against the
    protocol's contract.

  Recommendation: (a). It keeps every law of the contract, it follows
  S4.2's choice of the de Bruijn reading when Python's rule broke
  symmetry, and no existing test changes: every Python test of a repeated
  binder expects `False`, and gets it under (a).

### S10 resolutions (decided by the user, 2026-09-26)

- **N-S10-1: (a) port both engines.** `BinderMixin` (D-S10-7) and the
  derived-equivalence engine (D-S10-8) run in Rust. This is the term
  package's recorded exception to P3's granularity rule: Rust calls a
  binder's hooks, its children's methods, dataclass field reads,
  comparators and hand-written methods per node, because those are per
  node by nature, and every call into Rust answers one comparison, query
  or substitution Python asked for. CONTRIBUTING's "Porting to Rust"
  records the exception.
- **N-S10-2: (b) refusal.** A binder list that repeats an identifier, on
  either side, pairs with nothing (D-S10-10). The user chose this knowing
  that `\x x. x` is then not alpha-equivalent to itself: reflexivity and
  "structurally equivalent implies alpha-equivalent" hold for terms whose
  binder lists repeat no identifier, and the law tests say so as explicit
  preconditions.

### Steps

1. **S10.1: benchmarks.** Add `benchmarks/test_term.py` as planned above,
   and record the baseline here, on today's Python package.
2. **S10.2: core additions, test-first, with Rust tests.**
   - The new module `rust/fhy-core/src/term.rs`, with `term/renaming.rs`
     (moved from `expression/alpha.rs`, with D-S10-3 and `enter_binders`
     under N-S10-2), `term/binder.rs` (the traits),
     `term/mapping.rs` and `term/error.rs` (the moved errors and
     `BinderPairingError`).
   - `expression` imports `term`, and `Expression` implements the traits.
   - The tests are written first and fail against `todo!()` stubs, as in
     S4.2 and S7.2. `lib.rs`, the crate README and CONTRIBUTING's tables
     list the module.
   - The binding changes only its import paths, so the Python suite stays
     green.
3. **S10.3: the binding.** Add `rust/fhy-core-py/src/term.rs` with:
   - `renaming.rs`: `_rs.AlphaRenaming`;
   - `binder.rs`: the `Binder` adapter and the `BinderMixin` functions;
   - `derived.rs`: `_rs.EquivalenceRole`, the plan and the walks;
   - `mapping.rs`: the helper.

   Everything new goes into `_rs.pyi`. Nothing in Python uses it yet, so
   the suite stays green.
4. **S10.4: the Python switch** (marked breaking). The three modules
   become the thin layer of D-S10-1. `node.rs` and `registry/entries.rs`
   read the pyclass, and `expression/alpha.rs` is deleted in the same
   commit, since the Python class and the conversion cannot change apart.
   The README's three term rows change. The step lands with S10.5 when the
   migration is small enough to review in one commit (the survey expects
   it to be). Otherwise it leaves exactly the tests of the migration table
   failing, as S7.4 did.
5. **S10.5: tests.** Migrate the tests and add the interface suite (the
   test plan below).
6. **S10.6: benchmarks after,** recorded here with the verdict, then the
   status, the implementation notes and this checklist.

Commit per step. Every step ends with these green:

- `pytest`, and `pytest -m "not very_slow"`;
- the `property` session;
- `lint` and `type_check`, clean;
- `tests/test_rs_stub.py`;
- the Rust gate: fmt, clippy `-D warnings`, tests, doc `-D warnings`,
  deny, and `cargo +1.85 check`.

### Test plan

**Rust tests, written first (S10.2),** in a new `tests/it/term/` area
(`tests/it/term.rs`):

- **`renaming_stories.rs`.** It holds the renaming cases of
  `expression/alpha_stories.rs`, moved unchanged, and these new cases:
  - `extended` leaves the receiver unchanged, and equals a clone followed
    by `enter_binder`;
  - clones share frames without sharing changes: entering or leaving a
    frame on one clone leaves the other as it was;
  - `Hash` agrees with `==`: an empty frame and frame order count, and map
    order does not;
  - the read views, outermost first;
  - `enter_binders`: an arity mismatch, and a repeat on the left, on the
    right and on both sides refused (N-S10-2 (b)), each leaving the
    renaming unchanged, with its `Display`.
- **`renaming_properties.rs`.** It holds `alpha_properties.rs`, moved,
  and these new properties:
  - `hash` is consistent with `==` over random frame stacks;
  - `enter_binders` accepts exactly the equal-length lists without
    repeats, and then agrees with `enter_binder` of their zip.
- **`binder_stories.rs`,** over a test-local lambda calculus (`Var`,
  `App`, and `Lam` over a parameter list) that implements `Binder`:
  - S4.2's `Binders` cases, moved onto it;
  - `test_binder.py`'s cases: arity and child counts, free identifiers,
    a shadowed replacement key, substitution without capture, the
    capture-avoiding rename (a fresh id with the same name hint, the
    original identifier left free), no applicable replacement returning
    the same handle, and several children;
  - `Expression` through the traits, agreeing with its inherent methods.
- **`binder_properties.rs`:**
  - alpha equivalence is symmetric and transitive over random terms,
    repeated binders included, and agrees with a de Bruijn model that
    refuses repeated binders;
  - alpha equivalence is reflexive, and implied by structural
    equivalence, over terms whose binder lists repeat no identifier: the
    generator of those properties builds only such terms, and the
    property states the precondition;
  - a term with a repeated binder is alpha-equivalent to no term;
  - substitution respects alpha equivalence;
  - substitution never captures: the free identifiers of a result are the
    input's, minus the substituted ones that occur, plus the
    replacements' free identifiers.
- **`mapping_stories.rs`:** the mapping helper's 11 cases.
- **A traceability table** maps `test_alpha_equivalence.py`,
  `test_binder.py` and `test_derived_equivalence.py` to the Rust tests,
  extending S4.2's. The derived plan's tests map to the interface suite,
  since the engine is binding code.

**The interface suite, `tests/test_term_rust_binding.py`,** covers what
the binding adds over the core:

- **`AlphaRenaming`.**
  - It is `_rs.AlphaRenaming`, final (subclassing raises `TypeError`)
    and frozen (`FrozenMutationError`, a virtual `FrozenMixin`).
  - `repr`, pickle and `copy` round trips.
  - `==` and `hash` across construction paths; `empty()` is one object.
  - The `TypeError`s for a non-mapping and for non-`Identifier` keys and
    values, and the `ValueError` texts naming each part.
  - `resolve` returns the given objects (`is`).
  - Expressions and registry entries accept it and refuse anything else.
- **`BinderMixin` driving.**
  - The hooks a comparison, a free-identifier query and a substitution
    call, and how often.
  - An exception from each hook propagates as the same object, stopping
    where Python stopped, and a `KeyboardInterrupt` passes through.
  - Wrong result types raise `TypeError`, and a truthy non-`bool` child
    answer counts as `True`.
  - An expression child answers as its own method does.
  - A binder nested in a binder receives the extended renaming.
- **The derived engine.**
  - A plan is built once per class, and a class that overrides either
    method has that method called for its nested values.
  - A comparator receives an `AlphaRenaming` under the binders in scope,
    and a `key` normalizer is called on both sides.
  - Nested derived values 10,000 deep compare without `RecursionError`.
  - Every `EquivalenceDerivationError` text, and exceptions from
    comparators and `==` propagate.
  - Expressions and identifiers compare natively, with the same answers as
    their methods.
- **The mapping helper:** values compared in the left mapping's order, and
  a raising value hook.
- **Stubs:** `tests/test_rs_stub.py` covers everything new.

**Migrating the existing tests.** No test is skipped, or deleted without a
rewrite. The survey expects few changes, and each is recorded here with
its reason:

- **`test_alpha_equivalence.py` (59), `test_binder.py` (17),
  `test_derived_equivalence.py` (58) and its properties (4)** use only
  public names and match messages on `injective`, `dataclass`,
  `"bdy" is not a field`, `payload` and `sequence element`, which the core
  and the binding keep. Their repeated-binder tests expect `False`, which
  N-S10-2 (b) answers. Any test that expects `True` for a repeated binder
  is rewritten to the refusal and recorded here.
- **New tests for N-S10-2 (b),** in `test_binder.py` and
  `test_derived_equivalence.py`: a repeat on either side refused in both
  directions, and a term with a repeated binder not alpha-equivalent to
  itself. The law tests of those files state their precondition, binder
  lists without repeats, in their docstrings and in the terms they
  build.
- **`test_term.py` (28)**, the param, constraint and symbol-table
  equivalence tests, and the registry binding tests should pass unchanged.
  Any test that pins a message or a `repr` Z-1 or Z-5 changes is
  rewritten to the new text.

### Coordination with S8 and S9

This branch is rebased onto `dev-rust` after S8 and S9, so its edits to
shared files stay small and additive:

- **Untouched:** `pyproject.toml`, `noxfile.py`, `Cargo.toml`,
  `Cargo.lock` (S10 adds no dependency), `tests/conftest.py` and the
  benchmark `conftest.py`.
- **`rust/fhy-core/src/lib.rs`:** one `pub mod term;` line, one row of the
  module table, and one clause of the layering sentence. S8 adds
  `pub mod solver;` just before it, so the conflict, if any, is two
  adjacent added lines.
- **`rust/fhy-core/src/expression.rs`:** it drops `mod alpha;`,
  `pub use alpha::AlphaRenaming;` and two names from the `error`
  re-export.
- **`rust/fhy-core/tests/it/main.rs`:** one `mod term;` line. S8 adds
  `mod solver;`. `tests/it/expression.rs` drops its two `alpha_*` lines.
- **`rust/fhy-core-py/src/lib.rs`:** one `mod term;` line and one
  `#[pymodule_export]` block.
- **`src/fhy_core/_rs.pyi`:** one new block for the term names. Its
  existing `from fhy_core.term import AlphaRenaming, Term` stays.
- **CONTRIBUTING:** one row in the Python-to-Rust table and `term` in
  layer 4 of the layering list, beside S8's `solver` row.
- **READMEs:** the Python README's three "Term Traits" rows and its
  Rust-backed list, and one line of the crate README.
- **This document:** this section, appended, and one entry at the end of
  the Progress checklist. S8 ticks its own entries, and S9 appends its
  own, next to it.

S10 does not use the solver or the evaluators, and they do not use
`AlphaRenaming`, so no code depends across the slices.

### S10.1 baseline (2026-09-26, d5241db plus the new benchmarks)

`benchmarks/test_term.py` implements the benchmark plan above, with its
own toy classes: a lambda calculus over `BinderMixin` (`_Var`, `_App`,
`_Lam`), derived terms (`_DerivedConst`, `_DerivedAdd`, `_DerivedVar`,
`_DerivedLam`) and `_DerivedExpressionHolder`, a binder over S4.1's
100-operation tree, as a param binds its constraints. The param rows use
`create_integer_param()` and `create_natural_param_between(1, 10)`, and the
symbol-table row two tables of 20 variables.

Median time per call, from `.venv/bin/python -m pytest
benchmarks/test_term.py <the two reruns> -n 0 --benchmark-only` in the
worktree's environment, measuring today's Python package. The machine is
the S0 one, with Python 3.11.13 and pytest-benchmark 5.3.0. The load
average was about 2, and the table lists the best of three runs' medians.

| Benchmark | before |
|---|--:|
| `test_alpha_renaming_empty` | 915 ns |
| `test_alpha_renaming_with_free_renaming[1]` | 1.17 µs |
| `test_alpha_renaming_with_free_renaming[50]` | 4.43 µs |
| `test_alpha_renaming_extend[depth_1]` | 1.27 µs |
| `test_alpha_renaming_extend[depth_10]` | 1.29 µs |
| `test_alpha_renaming_resolve[frame]` | 304 ns |
| `test_alpha_renaming_resolve[free]` | 409 ns |
| `test_alpha_renaming_resolve[identity]` | 336 ns |
| `test_are_identifiers_alpha_equivalent[frame]` | 462 ns |
| `test_are_identifiers_alpha_equivalent[capture]` | 310 ns |
| `test_alpha_renaming_eq` | 1.38 µs |
| `test_alpha_renaming_hash` | 303 ns |
| `test_binder_alpha_equivalence[flat]` | 4.86 µs |
| `test_binder_alpha_equivalence[nested_10]` | 27.3 µs |
| `test_binder_free_identifiers` | 1.21 µs |
| `test_binder_substitute[no_capture]` | 2.25 µs |
| `test_binder_substitute[capture]` | 8 µs |
| `test_derived_structural_equivalence_of_a_deep_tree` | 2.11 ms |
| `test_derived_alpha_equivalence_of_a_deep_tree` | 2.31 ms |
| `test_derived_alpha_equivalence_of_binders` | 7.92 µs |
| `test_derived_equivalence_over_expressions` | 16.7 µs |
| `test_mapping_helper` | 50.2 µs |
| `test_param_alpha_equivalence[integer]` | 32.4 µs |
| `test_param_alpha_equivalence[natural_between]` | 61.6 µs |
| `test_param_structural_equivalence` | 41.1 µs |
| `test_param_construction_between_bounds` | 71.3 µs |
| `test_constraint_structural_equivalence` | 3.85 µs |
| `test_symbol_table_structural_equivalence` | 829.0 µs |
| `test_expression_alpha_equivalence_under_frames[1]` | 1.42 µs |
| `test_expression_alpha_equivalence_under_frames[10]` | 10.1 µs |
| `test_alpha_equivalence_under_free_renaming_of_deep_trees` (rerun) | 8.67 µs |
| `test_registered_function_alpha_equivalence` (rerun) | 848 ns |

- **The derived walk** is the slow path: 21 µs per node over the
  100-addition tree, structurally or alpha, and 41 µs per frame over the
  symbol tables. Each field pays the runtime-protocol `isinstance`
  checks.
- **The consumers** pay it per comparison: a param comparison takes 32 to
  62 µs, and building a bounded integer param 71 µs.
- **`BinderMixin`** costs about 2.7 µs per binder level, and a
  substitution that renames to avoid capture 8 µs.
- **The renaming** answers a lookup from Python in 300 to 460 ns, and
  `extend` takes 1.3 µs at any depth, since it copies only the tuple of
  frames. An expression compared under ten frames takes 10 µs, most of it
  S4.3a's conversion.

### S10.2 implementation notes

The tests were written first, against stubs of every new method (a
`todo!()`, or an empty view): 58 of the 107 tests of the new
`tests/it/term/` area failed, and all pass now. The 49 that passed are the
43 cases moved from `expression/alpha_stories.rs`, the three moved
properties, and three that pin what the stubs could not break. A proptest
regression file written while the stubs failed was deleted, as in S7.2;
its seeds were stub failures, not findings.

- **Layout.** `rust/fhy-core/src/term.rs` with `term/renaming.rs` (moved
  from `expression/alpha.rs`), `term/binder.rs` (the four traits),
  `term/mapping.rs` and `term/error.rs` (the moved
  `NonInjectiveRenamingError` and `RenamingPart`, and the new
  `BinderPairingError`). `expression` imports `term`; the two error names
  and `AlphaRenaming` left its re-exports. The binding changed only its
  import paths. `lib.rs`, the crate README and CONTRIBUTING's two tables
  list the module; `term` sits beside `tree` in layer 4.
- **Where the shape differs from D-S10-2's sketch, or fills it in:**
  - `BinderPairingError` is `ArityMismatch { left, right }` or
    `RepeatedIdentifier(Identifier)`, displaying `binder lists of 1 and 2
    identifiers cannot be paired` and `a binder list repeats the
    identifier x::7`. Lengths are checked first, then the left list, then
    the right, so a list repeated on both sides names its left repeat.
    `enter_binders` builds the frame directly: two distinct lists pair
    injectively, so it cannot meet `NonInjectiveRenamingError`.
  - The views are one type, `RenamingMap<'_>` (`get`, `iter`, `len`,
    `is_empty`), returned by `free_renaming()` and, outermost first, by
    `frames()`, which is double-ended and exact-size. A map's pairs come in
    no particular order.
  - `Hash` feeds the free renaming, the number of frames, then each frame,
    each map as its length and its pairs sorted by the key's id, so it
    agrees with the derived `Eq`, which compares maps as sets and counts
    empty frames and frame order.
  - The frames and the free renaming are `Arc<Bijection>`s, so `Clone` and
    `extended` copy one reference per frame. `enter_binder`'s rustdoc now
    names `enter_binders` for pairing lists (N-S10-2 (b)).
  - `Binder`'s provided methods are `is_binder_alpha_equivalent_under`,
    `binder_free_identifiers` and `substitute_avoiding_capture`, with
    `BinderMixin`'s order of checks: bound counts, child counts, the
    pairing, then the children in order. A substitution with no applying
    key returns `self.clone()`, the same handle for an `Arc`-backed node.
  - `is_mapping_alpha_equivalent_under` takes the left map as an
    exact-size iterator of pairs, so its values are compared in an order
    the caller chooses (the Python helper's dict order, D-S10-11), and the
    right map as a `HashMap`.
  - `Expression` implements `AlphaEquivalence`, `FreeIdentifiers` and
    `Term` by forwarding to its inherent methods, which keep their names:
    an inherent method is found first, so no caller changed.
- **The laws under N-S10-2 (b).** `binder_properties.rs` checks alpha
  equivalence against a de Bruijn model that has no form for a term with a
  repeated binder list. Symmetry, the model and "a repeated binder matches
  no term" run over all terms. Reflexivity, transitivity and "substitution
  respects alpha equivalence" run over `build_distinct_term_strategy`,
  whose lambdas drop a repeated parameter, and each property asserts that
  precondition before the law. "Substitution never captures" holds for all
  terms.
- **S4.2's `Binders` cases stay renaming stories.** They compare expression
  bodies under frames entered per binder level, which exercises the
  renaming itself, so they moved unchanged with the file. The `Binder`
  versions of the same stories are new, over the lambda calculus in
  `tests/it/support/lambda.rs`, which `binder_stories.rs` and
  `binder_properties.rs` share.
- **Tests.** `renaming_stories.rs` (43 moved cases and 14 new, counting
  `rstest` cases), `renaming_properties.rs` (3 moved properties and 2 new),
  `binder_stories.rs` (27), `binder_properties.rs` (7) and
  `mapping_stories.rs` (11).

Traceability of the Python tests to the new Rust tests (`test_binder.py`
is `B`, `test_alpha_equivalence.py` `A`), extending S4.2's table, whose
rows keep their Rust tests in `term/renaming_stories.rs`:

| Python tests | Rust tests | Note |
|---|---|---|
| B `test_identity_lambdas_are_alpha_equivalent`, `test_lambdas_with_distinct_free_bodies_are_not_alpha_equivalent`, `test_lambdas_sharing_a_free_identifier_are_alpha_equivalent`, `test_lambdas_with_different_arity_are_not_alpha_equivalent` | `identity_lambdas_over_different_parameters_are_alpha_equivalent`, `lambdas_over_distinct_free_bodies_are_not_alpha_equivalent`, `lambdas_sharing_a_free_identifier_are_alpha_equivalent`, `lambdas_binding_different_numbers_of_identifiers_are_not_alpha_equivalent` | |
| B `test_blocks_with_different_statement_counts_are_not_alpha_equivalent`, `test_block_alpha_equivalence_recurses_over_all_statements` | `blocks_with_different_numbers_of_children_are_not_alpha_equivalent`, `block_alpha_equivalence_compares_every_child_under_the_frame` | |
| B `test_free_identifiers_*`, `test_block_free_identifiers_union_over_all_statements` | `free_identifiers_leave_out_a_bound_parameter`, `free_identifiers_hold_an_unbound_body_reference`, `free_identifiers_are_the_union_over_the_children_minus_the_bound_set` | |
| B `test_substitute_skips_shadowed_bound_identifier`, `test_substitute_with_no_replacements_returns_self` | `substitute_leaves_a_key_the_lambda_binds_shadowed`, `substitute_with_no_applying_key_returns_the_same_handle` | the same handle, by `Arc::ptr_eq` |
| B `test_substitute_into_body_without_capture`, `test_substitute_avoids_capture_by_renaming_binder` | `substitute_rewrites_a_free_identifier_of_the_body_in_place`, `substitute_renames_a_binder_that_would_capture_a_replacement`, `substitute_renames_only_the_parameters_a_replacement_would_capture` | |
| B `test_binder_is_not_alpha_equivalent_to_non_binder` | `a_lambda_is_not_alpha_equivalent_to_a_variable` | |
| B `test_non_injective_binding_is_not_alpha_equivalent` | `a_lambda_repeating_a_parameter_matches_no_lambda_on_either_side`, `a_lambda_repeating_a_parameter_is_not_alpha_equivalent_to_itself`, `a_repeated_parameter_nested_inside_a_term_makes_the_whole_term_match_nothing`, `alpha_renaming_enter_binders_refuses_a_list_that_repeats_an_identifier` (3 cases) | N-S10-2 (b) |
| B `test_binder_is_a_term` | none | a Python protocol check |
| A `test_binder_alpha_equivalence_*` | `lambdas_nested_over_one_name_match_the_inner_binder`, `a_lambda_refuses_to_capture_a_free_identifier`, `lambdas_swapping_their_parameters_and_arguments_are_alpha_equivalent`, `a_lambda_compares_its_free_identifiers_under_the_free_renaming` | beside S4.2's `binders_*` stories |
| A `test_alpha_equivalence_is_reflexive`, `..._is_symmetric`, `..._is_transitive` | `alpha_equivalence_is_reflexive_on_terms_without_repeated_binders`, `alpha_equivalence_is_symmetric`, `alpha_equivalence_is_transitive_on_terms_without_repeated_binders`, `alpha_equivalence_agrees_with_the_de_bruijn_model` | reflexivity and transitivity under their precondition |
| A `test_alpha_renaming_extend_returns_new_instance`, `..._does_not_mutate_receiver` | `alpha_renaming_extended_leaves_the_receiver_unchanged`, `alpha_renaming_extended_equals_a_clone_that_enters_the_frame`, `alpha_renaming_clones_share_frames_without_sharing_changes` | `extended` is Python's `extend` |
| A `test_alpha_renaming_hashable` | `alpha_renaming_equal_renamings_built_in_different_orders_hash_alike`, `alpha_renaming_hash_agrees_with_equality` | S4.2 had no Rust test |
| A `test_mapping_helper_*` (11) | `mapping_stories.rs` (11) | the order of value comparisons is new |
| none | `alpha_renaming_views_*`, `alpha_renaming_enter_binders_*`, `a_binder_over_expressions_*`, `expression_through_the_traits_agrees_with_its_methods`, `substitution_*` | new: the views, the pairing, a binder over expressions, the traits on `Expression`, substitution laws |

### S10.3 status

The binding is `rust/fhy-core-py/src/term.rs` with `term/renaming.rs`
(`_rs.AlphaRenaming` and the identifier-object tables), `term/adapter.rs`
(the context, `PyTerm` and `PyBinder` over the core traits),
`term/binder.rs` and `term/mapping.rs` (the functions `BinderMixin` and the
mapping helper call), and `term/derived.rs` (`_rs.EquivalenceRole`, the
plans and the walks). Everything new is exported from `_rs` and declared in
`_rs.pyi`, whose import of the Python `AlphaRenaming` is aliased until the
switch. `NonInjectiveRenamingError`'s `IntoPyErr` moved from
`expression/node.rs` to `term/renaming.rs`, and `PyExpression` exposes its
handle and structural equality to the crate. Nothing in Python uses the
binding yet, so the suite is unchanged (7,327 passed).

### S10.4 status

`fhy_core.term` runs on the binding. `alpha_equivalence.py` re-exports
`_rs.AlphaRenaming` (registered as a virtual `FrozenMixin`) and
`_rs.is_identifier_mapping_alpha_equivalent_under`, and keeps the protocol
and the mixin; `binder.py` keeps the protocols and `BinderMixin`'s abstract
hooks, whose three derived methods call `_rs`; `derived_equivalence.py`
keeps the role functions, now returning `_rs.EquivalenceRole` values, the
metadata key, the error class, `FieldComparator`, the `_PLAN_CACHE` dict and
the mixin, whose two methods call `_rs`. `expression/node.rs` and
`registry/entries.rs` read the Rust renaming from the pyclass, and
`expression/alpha.rs`, the per-comparison conversion, is deleted. The
function entries pair their parameters with `enter_binders` (D-S10-14).
The README's three term rows changed. The whole suite passed unchanged
(7,327 passed): no existing test pins a behavior Z-1 to Z-7 or N-S10-2
changes, so S10.5 migrates nothing and only adds tests.

### S10.5 status

No existing test needed a rewrite (S10.4), and none was skipped or
deleted. At the end of S10.5: `pytest` 7,376 passed, `-m "not very_slow"`
7,409 passed, the `property` session 281 passed, `lint` and `type_check`
clean, `tests/test_rs_stub.py` green.

| Test | Now | Reason |
|---|---|---|
| `test_alpha_equivalence.py`: the `example_terms` fixture and the reflexivity and transitivity tests | same names | N-S10-2 (b): their docstrings state the precondition, binder lists without a repeated identifier, which the example terms meet |
| `test_derived_equivalence.py::test_structural_equivalence_implies_alpha_equivalence` | same name | the same precondition, stated in its docstring |
| none | `test_binder.py::test_lambda_repeating_a_parameter_matches_no_lambda_in_either_direction`, `..._is_not_alpha_equivalent_to_itself` | N-S10-2 (b), new |
| none | `test_derived_equivalence.py::test_binder_repeating_an_identifier_matches_no_binder_in_either_direction` | N-S10-2 (b), new |

No Python test expected `True` for a repeated binder: the two that
compare one, `test_binder.py::test_non_injective_binding_is_not_alpha_equivalent`
and `test_derived_equivalence.py::test_binder_with_non_injective_binding_returns_false`,
expect `False` and pass unchanged.

The new `tests/test_term_rust_binding.py` (46 tests, counting parametrized
cases) covers the test plan's interface suite: the renaming's class
structure (the extension class itself, final, frozen), its argument checks
and the core's `ValueError` texts, `empty()` as one object, equality and
hash across construction orders, the `repr`, pickling and copying, and
`resolve` returning the objects given; expressions and registry entries
reading it and refusing anything else; the hooks a binder comparison, a
free-identifier query and a substitution call, a hook's exception as the
same object stopping where `all` stopped, a `KeyboardInterrupt`, a bound
identifier of the wrong type, truthy child answers, an expression child,
the extended renaming a child receives, and repeated parameters; the
derived engine's plan cache, a 10,000-deep chain in both modes, a nested
value with its own method, comparators and `key` normalizers receiving
their arguments, exceptions from user code, native expressions and
identifiers, repeated binder fields (scoped or not), argument types, the
roles as `_rs.EquivalenceRole` values, and the unknown-scope text; and the
mapping helper's order, a raising value, and a key of the wrong type.

### S10 status

S10 was implemented on 2026-09-26 in eight commits after the design: the
user's decisions (d5241db); the benchmarks and their baseline (388a69d);
the core module, test-first (63d5fa7); the binding (c41a29b); the Python
switch, marked breaking (1efe852); the new tests and the interface suite
(35fe0b5); faster identifier and map reading, which the benchmarks called
for (bac34da); and these docs. No test was skipped or deleted. At the end:
`pytest` 7,376 passed, `-m "not very_slow"` 7,409 passed, the `property`
session 281 passed, nox `lint` and `type_check` clean,
`tests/test_rs_stub.py` green, and the Rust gate green (fmt, clippy
`--all-targets -D warnings`, 2,872 tests, doc `-D warnings`, deny,
`cargo +1.85 check`).

### S10 benchmarks (before and after)

Median time per call, from the worktree's environment, `pytest
benchmarks/test_term.py <the three reruns> -n 0 --benchmark-only`, on the
S0 machine with Python 3.11.13 and pytest-benchmark 5.3.0. "Before" is
388a69d, the S10.1 baseline's tree, exported with `git archive` under
`target/` and built there; "after" is the S10.6 tree. The two ran three
times each, interleaved (before, then after, in each round), with a load
average of 3 to 8 (other builds shared the machine), and the table lists
the best of the three medians. The "before" column agrees with the S10.1
table within 8%.

| Benchmark | before | after | after / before |
|---|--:|--:|--:|
| `test_alpha_renaming_empty` | 958 ns | 57 ns | 0.06 |
| `test_alpha_renaming_with_free_renaming[1]` | 1.29 µs | 710 ns | 0.55 |
| `test_alpha_renaming_with_free_renaming[50]` | 4.65 µs | 16.6 µs | 3.58 |
| `test_alpha_renaming_extend[depth_1]` | 1.36 µs | 664 ns | 0.49 |
| `test_alpha_renaming_extend[depth_10]` | 1.41 µs | 872 ns | 0.62 |
| `test_alpha_renaming_resolve[frame]` | 323 ns | 171 ns | 0.53 |
| `test_alpha_renaming_resolve[free]` | 428 ns | 204 ns | 0.48 |
| `test_alpha_renaming_resolve[identity]` | 353 ns | 169 ns | 0.48 |
| `test_are_identifiers_alpha_equivalent[frame]` | 462 ns | 321 ns | 0.69 |
| `test_are_identifiers_alpha_equivalent[capture]` | 323 ns | 245 ns | 0.76 |
| `test_alpha_renaming_eq` | 1.43 µs | 101 ns | 0.07 |
| `test_alpha_renaming_hash` | 315 ns | 147 ns | 0.47 |
| `test_binder_alpha_equivalence[flat]` | 5.24 µs | 2.43 µs | 0.46 |
| `test_binder_alpha_equivalence[nested_10]` | 29.4 µs | 19.1 µs | 0.65 |
| `test_binder_free_identifiers` | 1.28 µs | 2.58 µs | 2.01 |
| `test_binder_substitute[no_capture]` | 2.39 µs | 3.2 µs | 1.34 |
| `test_binder_substitute[capture]` | 8.47 µs | 9.05 µs | 1.07 |
| `test_derived_structural_equivalence_of_a_deep_tree` | 2.18 ms | 37 µs | 0.02 |
| `test_derived_alpha_equivalence_of_a_deep_tree` | 2.37 ms | 37.6 µs | 0.02 |
| `test_derived_alpha_equivalence_of_binders` | 8.11 µs | 2.59 µs | 0.32 |
| `test_derived_equivalence_over_expressions` | 17 µs | 11.1 µs | 0.65 |
| `test_mapping_helper` | 51.3 µs | 30 µs | 0.59 |
| `test_param_alpha_equivalence[integer]` | 32.6 µs | 5.52 µs | 0.17 |
| `test_param_alpha_equivalence[natural_between]` | 63.6 µs | 8.53 µs | 0.13 |
| `test_param_structural_equivalence` | 41.2 µs | 6.57 µs | 0.16 |
| `test_param_construction_between_bounds` | 75.1 µs | 64.6 µs | 0.86 |
| `test_constraint_structural_equivalence` | 3.84 µs | 1.3 µs | 0.34 |
| `test_symbol_table_structural_equivalence` | 860.8 µs | 187.2 µs | 0.22 |
| `test_expression_alpha_equivalence_under_frames[1]` | 1.45 µs | 269 ns | 0.18 |
| `test_expression_alpha_equivalence_under_frames[10]` | 10.3 µs | 520 ns | 0.05 |
| `test_alpha_equivalence_under_free_renaming_of_deep_trees` (rerun) | 8.95 µs | 6.63 µs | 0.74 |
| `test_registered_function_alpha_equivalence` (rerun) | 846 ns | 658 ns | 0.78 |
| `test_identifier_expression_construction` (rerun) | 437 ns | 372 ns | 0.85 |

The hot paths the plan put at risk are faster, and the consumers are
where S10 pays off:

- **The derived engine** walks the 100-addition trees 59 to 63 times
  faster (2.2 ms to 37 µs), a binder over a 100-operation expression 1.5
  times, and `\x. x` against `\y. y` 3.1 times. Its consumers follow: a
  param comparison is 5.9 to 7.5 times faster (33 to 64 µs down to 5.5 to
  8.5 µs), a constraint comparison 3 times, and two 20-variable symbol
  tables 4.6 times. Building a bounded param, which dedupes its constraints
  structurally but spends most of its time elsewhere, is 14% faster.
- **The renaming.** `resolve` and the correspondence, which hand-written
  hooks call from Python, take 0.5 to 0.8 times as long; `extend` 0.5 to
  0.6; `==` 14 times less and `hash` half, since both are Rust's; and
  `empty()` returns one object. An expression compared under a framed
  renaming no longer converts it: 5.4 times faster under one frame, 20
  times under ten.
- **`BinderMixin` comparisons** are 1.5 to 2.2 times faster, although the
  core calls the Python hooks, and the mapping helper 1.7 times.
- **The reruns.** Expressions under a free renaming and function entries
  compare 1.3 times faster (no conversion), and identifier expressions are
  built 15% faster, from the faster identifier reading of bac34da.

**Three rows are slower than 10%, and are recorded as accepted costs**
(cross-cutting rule 5). The maintainer accepted them on 2026-09-26, as
part of moving the term package to Rust (CONTRIBUTING "Replacing a
Python class"):

| Benchmark | after / before | Why | Who pays it |
|---|--:|---|---|
| `test_alpha_renaming_with_free_renaming[50]` | 3.6 | each key and value becomes a Rust identifier (about 160 ns per pair) and is kept with its Python object, where Python copied the dict into an `immutabledict` in C. One pair is 1.8 times faster | building a free renaming of many pairs: no module in `src` does; tests and benchmarks do, once per comparison |
| `test_binder_free_identifiers` | 2.0 | the identifiers cross into Rust and back: the children's frozensets become Rust sets, and the result becomes Python objects again | `BinderMixin.get_free_identifiers`, which no module in `src` uses |
| `test_binder_substitute[no_capture]` | 1.3 | the same crossings for the capturable set, plus a Python dict built per child call | `BinderMixin.substitute`, likewise; with a capture it is within 7% |

`BinderMixin` has no consumer in `src`, and N-S10-1 (a) chose one
implementation of its algorithms, in the core, over keeping Python's. The
first bac34da pass removed part of each cost: an exact `Identifier` is now
read through its instance attributes, not its properties (98 against
10 ns per read, measured), and a dict is iterated directly.

### S10 implementation notes

Choices the decisions left open, made while implementing S10.3 to S10.6,
and where the shape differs from the plan (S10.2's are in its own notes):

- **The identifier objects.** `_rs.AlphaRenaming` keeps, beside the Rust
  renaming, a chain of tables from ids to the Python identifier objects
  given as keys or images, one table per extension, shared between
  renamings, and dropped on the heap so a long chain cannot overflow the
  stack. `resolve` returns the newest object held for the image's id, or
  its argument when nothing maps it; two equal identifiers in different
  frames are one id, so it may return the equal object of another frame.
  Pickling rebuilds the maps from these objects, so an unpickled renaming
  resolves to equal identifiers.
- **Stricter arguments (Z-2).** A key, value, resolved identifier, bound
  identifier, reference field or replacement key that is not an
  `Identifier` raises `TypeError` in S2's style (`AlphaRenaming key must be
  an Identifier, got str.`, `BinderMixin bound identifier must be an
  Identifier, got str.`, `compared_as_binder identifier must be ...`,
  `compared_as_reference identifier must be ...`); Python accepted any
  hashable. A mapping argument that is not a mapping raises `TypeError`.
  A `Mock(spec=Identifier)` passes, through the properties, since only an
  exact `Identifier` takes the attribute fast path.
- **The shared empty renaming** is a write-once slot, recorded in
  CONTRIBUTING's "Process-global state" section: it is immutable, like the
  public-class slots. The maintainer approved it on 2026-09-26. The derived plans stay in the Python module's
  `_PLAN_CACHE` dict, so the engine adds no Rust static for them; a plan
  is an unexported `_rs.EquivalencePlan` object there.
- **The adapters' error context** (D-S10-7). Each entry function creates
  one context for its call. A hook's exception is kept, the adapter
  answers `false`, an empty set or an empty list, every later hook call is
  skipped, and the entry raises the exception itself; a
  `KeyboardInterrupt` is kept and raised the same way. The bound
  identifiers and children are read on first use, so a comparison reads
  them in `BinderMixin`'s order: both bound lists, the arity check, then
  both child lists.
- **What the adapters hand back to Python.** A child is compared under a
  new `AlphaRenaming` object built from the core's extended renaming and
  the objects seen so far, or under the given object when the core passes
  the given renaming on. The results of `rename_bound_identifier` and
  `rebuild_with_scoped_children` are not type-checked, as Python did not.
  A fresh identifier is minted by the core (`Identifier::new`) and built
  in Python through `Identifier.deserialize_from_dict`.
- **Native paths.** An `_rs.Expression` whose class keeps
  `_rs.Expression`'s method is compared by the core without calling
  Python, in the adapters and in the derived engine; so is an exact
  `Identifier` by id in the engine's `==`, and a nested value whose class
  keeps `DerivedEquivalenceMixin`'s method, walked on the engine's own
  stack. Every other value is asked in Python.
- **The engine's capability checks** ask the value, as a
  runtime-checkable protocol does (a `Mock` carries its attributes on the
  instance), and the answers are cached per class for one call.
- **A binder field that scopes over nothing** still has its pairing
  checked, so a repeated identifier there makes the node alpha-equivalent
  to nothing; Python checked a pairing only while extending a scoped
  field. This is D-S10-10's "pairs with nothing" for every binder field,
  pinned by `test_derived_binder_repeating_an_identifier_matches_nothing`.
- **Roles.** `_rs.EquivalenceRole` has the static constructors `value`,
  `reference`, `binder`, `excluded` and `explicit`, and the getters `kind`,
  `key`, `scopes_over` and `comparator`, and a `repr` such as
  `EquivalenceRole.reference()`. A binder's `scopes_over` must hold `str`s
  (`TypeError`). An object under the metadata key that is not a role
  compares by the default dispatch, as before.
- **The stub.** `_rs.pyi` declares `AlphaRenaming`, `EquivalenceRole` and
  the six functions, and types the renaming parameters of the expression,
  tag and entry classes with the stub's `AlphaRenaming`, which is now the
  public class.
- **Follow-ups, not done here.** `Binder` for `FunctionDefinition` and
  `ComposedFunction` (D-S10-14); a Rust counterpart of the derived plan
  for Rust IR, which would be a derive macro in a second crate (D-S10-8).

### S10 rebase onto S8 (2026-09-26)

The branch was rebased onto `dev-rust` at 7078ca2, S8's last commit. The
conflicts were all additive: S8's and S10's checklist entries and sections
(S8's kept intact, S10's after them), the module tables and layering
sentences of `lib.rs`, the crate README and CONTRIBUTING (`solver` and
`term` both listed; `term` sits beside `tree`, `solver` after `expression`),
the `mod`/`#[pymodule_export]` lines of the binding's `lib.rs`, the stub
(S8's version, with S10's alias and block reapplied), and CONTRIBUTING's
process-global section (S8's default solver, then S10's slot). S8 uses no
term item: its binding converts identifiers through `restore_identifier`
and `read_identifier_id`, whose meaning 529cabb's fast path (now bac34da)
keeps, so no code changed. Every rebased commit that touches Rust was
checked to build (`cargo check --workspace --all-targets`). After the
rebase: `pytest` 7,438 passed, `-m "not very_slow"` 7,471, the `property`
session 282, `tests_minimal` 5,813 passed and 608 skipped, nox `lint` and
`type_check` clean, and the Rust gate green: 3,104 tests with the default
features and 3,136 with `--all-features` (the `z3` feature against the
z3-solver wheel's libz3 4.16, as S8.3 builds it), clippy `--all-targets
-D warnings` with and without the feature, fmt, doc, deny and `cargo +1.85
check`. A single run of the term benchmarks after the rebase found no row
slower than the table above by more than 15%, so the table stands. The
commit hashes in this section are the rebased ones.

## S9: the expression evaluators

- **Status:** designed 2026-09-26 at ab05802, in parallel with S8's
  implementation, and implemented the same day; see "S9 status" below.
  D-S9-1 to D-S9-20 apply the policy the user already set and the
  direction in "Plan after S7" (item 4). N-S9-1 was resolved as (a) and
  N-S9-2 as (b); see "S9 resolutions".
- **Pattern:** the evaluation logic moves into a new core module,
  `fhy_core::expression::evaluate`. One evaluator walk is generic over
  its values: a scalar backend in every build, and an `ndarray` backend
  behind an off-by-default `ndarray` cargo feature, which the binding
  enables. The binding converts NumPy arrays to and from `ndarray` with
  rust-numpy. The two Python passes become `CompilerPass`es over the core,
  as S7 did for `FunctionInliner`.
- **Scope:** `passes/numpy.py`, `passes/evaluate.py`,
  `passes/native_lowering.py` and `pprint.py`, the native kernels of the
  19 native built-ins, and the helpers they need. The solver and the z3
  and sympy passes are S8's.
- **Revises D-S7-7's "evaluation stays Python"**, which kept the fold in
  Python because the core computed no native function. S9 gives the core
  those kernels (D-S9-7).

### Survey: the Python API

The four modules are 1,188 lines, pure Python over the Rust-backed
expressions and registry.

| File | Lines | Public names |
|---|--:|---|
| `passes/numpy.py` | 728 | `evaluate_expression_with_numpy` (`__all__`); `NumpyExpressionEvaluator`, a `VisitablePass` registered as `fhy_core.symbolic.expression.evaluate_with_numpy`; six private tables |
| `passes/evaluate.py` | 186 | `evaluate_expression`, `ExpressionEvaluator`, a `RewritablePass` registered as `fhy_core.symbolic.expression.evaluate` |
| `passes/native_lowering.py` | 118 | `coerce_literal_value`, `is_decimal_text_exactly_binary`, `try_get_native_constant_value` |
| `pprint.py` | 156 | `pformat_expression` (`__all__`), `ExpressionPrettyFormatter`, a `VisitablePass` |

`symbolic/expression/__init__.py` re-exports `evaluate_expression`,
`evaluate_expression_with_numpy`, `pformat_expression` and the evaluator
errors. `builtins.py` holds `_NATIVE_IMPLEMENTATIONS`, the 19 `math`
callables (plus `_exp2` and the builtin `round`) that the fold calls.

**The NumPy evaluator (`numpy.py`).**

- `evaluate_expression_with_numpy(expression, environment)`:
  1. imports NumPy, or raises `ImportError("NumPy is required for
     ... pip install fhy_core[numpy].")`;
  2. runs `inline_functions`, so composed built-ins and user functions
     disappear;
  3. reads a `SymbolType` from the dtype kind of each binding the inlined
     tree references (`b` Boolean, `i`/`u` integer, `f` real; any other
     kind undeclared), and screens with `validate_logical_operands`;
  4. refuses an environment binding a referenced native constant
     (`NativeConstantBindingError`);
  5. runs `NumpyExpressionEvaluator(environment, numpy)` as a pass.

  Steps 3 and 4 raise directly. Every error in 2 and 5 arrives as
  `PassExecutionError` with the cause.
- **The walk** calls one NumPy function per node, over whole arrays:
  - the 13 binary operations through ufuncs (`true_divide`,
    `floor_divide`, `mod`, `power`, ...);
  - `LogicalExpression` by reducing its operands with `logical_and` or
    `logical_or`, and the three unary operations;
  - a piecewise as a right-folded chain of `numpy.where`, which evaluates
    every branch for every element;
  - 18 native built-ins through ufuncs (`round` is `numpy.round`).
    `erf`, and so `gelu`, and every user native raise
    `UnsupportedNumpyLoweringError`.
- **Values.**
  - An identifier is `numpy.asarray(binding)`, once per occurrence, or a
    native constant's value. Anything else raises `UnboundVariableError`,
    with three wordings: not bound, merely named like a constant, or the
    name of a registered function.
  - A literal goes through `coerce_literal_value`. A decimal with no
    exact binary float raises `StringLiteralPrecisionError`.
  - Dtypes follow NumPy's promotion. A `float32` input stays `float32`,
    int64 arithmetic wraps silently, `x // 0` on integers is `0` with a
    warning, and a Boolean takes part in arithmetic as `0` or `1`.
- **Casts.** An `INT`-, `NAT`- or `BOOL`-sorted native result (`round`,
  `floor`, `ceil`) is cast to `int64` or `bool_`. A non-finite value there
  raises `NonFiniteCastError`. Inside a piecewise branch the offending
  lanes are recorded in a mask instead, folded through the same `where`
  chain, and raised only if the selected branch returns one. So
  `{floor(sqrt(x)) if x >= 0; 0 otherwise}` works. A finite value out of
  `int64`'s range is cast silently to a platform value.
- **Guards.** A connective operand and a piecewise condition must be
  Boolean-dtyped. The static screen of step 3 catches the provably
  numeric ones. A runtime check catches the rest, for example an object
  dtype: `NonBooleanLogicalOperandError` for an operand, `TypeError` for
  a condition, both wrapped.
- **Floating-point domain errors** follow NumPy: `sqrt(-1)`, `log(0)` and
  division by zero give `nan` or `inf`, with NumPy's `RuntimeWarning`.
  NumPy still raises `ValueError` for an integer to a negative integer
  power.
- **Results.** A 0-d result is a NumPy or Python scalar, except that a
  piecewise- or identifier-rooted tree gives a rank-0 `ndarray`. An
  identifier root returns `asarray` of the binding, which may be the
  caller's own array. `did_change` is always `True`. Bindings the tree
  does not reference are ignored, and are never passed to `asarray`.

**The fold (`evaluate.py`).** `ExpressionEvaluator` is a `RewritablePass`
with two rewrites, bottom-up and once per occurrence:

- A call is looked up with `get_registered_entry`:
  - an unknown name raises `EntryLookupError`;
  - a constant's name raises `FunctionArityError`;
  - a `RegisteredFunction`, composed built-ins included, is kept, and
    reports a WARNING recommending `inline_functions`;
  - a `NativeFunction` whose arguments are all literals, after
    `coerce_literal_value`, is replaced by
    `LiteralExpression(implementation(*values))`. The result is checked
    with `is_python_value_compatible_with_sort` (`NativeResultSortError`).
    Arity is never checked, so `sin(1.0, 2.0)` raises the callable's
    `TypeError`.
- An identifier that is a native constant's canonical identifier becomes
  its value.

Literal arithmetic is not folded. A native's own exception
(`math.sqrt(-1)`'s `ValueError`, a `ZeroDivisionError`, an
`OverflowError`) propagates. Every error arrives as `PassExecutionError`
with the cause.

**`native_lowering.py`.**

- `is_decimal_text_exactly_binary(text)` is `Decimal(text) ==
  Decimal(float(text))`.
- `coerce_literal_value(value)` passes `bool`, `int` and `float` through,
  converts integer-grammar text with `int`, and converts a decimal to
  `float` only when exact. Otherwise it raises
  `StringLiteralPrecisionError("cannot coerce decimal literal '0.1' ...")`.
- `try_get_native_constant_value(identifier)` is the constant's `value`,
  or `None`.
- `passes/sympy.py` imports `is_decimal_text_exactly_binary` (it is S8's
  file).

**`pprint.py`.**

- `pformat_expression(expression, show_id=False, functional=False)`
  already renders through the core (`Expression._format`, S4.3a).
- `ExpressionPrettyFormatter(is_id_shown, is_printed_functional)` is the
  Python `VisitablePass` that N-S6-3 kept for subclasses. It is a second
  renderer of the core's text, and `test_pretty_formatter_agrees_with_the_core_text`
  checks the two agree. Its `__call__` refuses a non-`str` result.
- Its consumers: no `src` module; one benchmark pipeline
  (`test_mixed_pipeline_over_a_deep_expression`); `test_pprint.py`.

**Consumers in `src`.**

- `pformat_expression`: `types/core.py` (4 calls), `constraint/core.py`
  (1) and `types/checking/type_checker.py` (1). These are unchanged.
- The two evaluators have no caller in `src` besides the re-exports. The
  registry's docstrings mention `evaluate_expression`.
- `is_decimal_text_exactly_binary`: `passes/sympy.py`.

**Probed at ab05802** (this machine: an i9-7920X with AVX-512, Python
3.11.13, NumPy 2.4.6; best of five `timeit` repeats):

| Case | Time |
|---|--:|
| `evaluate_expression_with_numpy` of `x*x + 2x + 1`, `x` a float | 29.6 µs |
| the same, over 1,000 floats | 35.4 µs |
| the same, over 10^6 floats | 6.86 ms (NumPy by hand: 3.56 ms) |
| `sigmoid(x)`, `x` a float | 56.1 µs |
| `sigmoid(x)` over 10^6 floats | 3.51 ms |
| `exp`, `log`, `sqrt`, `tanh`, `sin` over 10^6 floats | 1.07, 1.38, 0.85, 1.96 and 10.0 ms, within 5% of the bare ufunc |
| `evaluate_expression(exp(1.0))` | 7.25 µs |
| `evaluate_expression` of S4.1's deep tree (nothing to fold) | 194 µs |
| `ExpressionPrettyFormatter()` of the deep tree | 223 µs (`pformat_expression`: 8.4 µs) |

- **Scalar environments** pay for the pass lifecycle twice, the screen,
  and a Python visitor call and a NumPy call per node: about 30 µs for a
  four-operation tree, against 39 ns for NumPy by hand.
- **Large arrays** pay NumPy's own cost plus a fixed overhead.
- **NumPy's float64 `exp`, `log` and `tanh` are SIMD kernels** (AVX-512
  here). A plain Rust loop over 10^6 floats, built with `rustc -O` for the
  baseline x86-64 target the wheels use, measured:

  | Kernel | Rust std | NumPy ufunc | Rust / NumPy |
  |---|--:|--:|--:|
  | `exp` | 4.84 ms | 1.03 ms | 4.7 |
  | `ln` | 4.65 ms | 1.35 ms | 3.4 |
  | `tanh` | 19.0 ms | 1.91 ms | 9.9 |
  | `sin` | 11.0 ms | 9.88 ms | 1.1 |
  | `sqrt` | 0.77 ms | 0.80 ms | 1.0 |
  | fused `x*x + 2x + 1` | 0.68 ms | 3.56 ms (4 ufuncs) | 0.19 |
  | fused sigmoid | 4.75 ms | 3.51 ms (evaluator) | 1.35 |

  This is N-S9-2.

### Survey: the Rust API

- **No evaluator.** The builtins module says the catalogue "does not
  compute native functions". Nothing converts a `Decimal` to an `f64`,
  and nothing depends on `ndarray`.
- **What the port builds on:**
  - `FunctionRegistry::inline` (S7), which takes linear time in the
    distinct nodes and keeps native calls with their arity checked;
    `FunctionRegistry::constant` and `entry`; `NativeFunction`'s sorts;
  - `BuiltinConstant::of_identifier` and `value()`, and
    `BuiltinFunction`'s sorts;
  - `BooleanScreen` over `SymbolTypes`, and `Expression::free_identifiers`;
  - `LiteralValue`, `Decimal` (`coefficient`, `exponent`) and
    `FunctionSort::accepts_literal`, the table the fold's result check
    needs;
  - `ExpressionDisplay`/`FormatOptions`, and
    `expression::passes::ExpressionPrettyFormatter`, a core pass that
    formats under `FormatOptions`;
  - `pattern::CallbackError` (D-10);
  - the operations' documented semantics: `Divide` is the exact real
    quotient, `FloorDivide` rounds toward negative infinity, and
    `FloorMod`'s sign follows the divisor (F-010).
- **Crates** (crates.io, 2026-09-26):
  - **`numpy` 0.29.0** (rust-numpy) is the release for pyo3 0.29. It is
    BSD-2-Clause, with `rust-version` 1.83, below our 1.85. It depends on
    `pyo3 ^0.29` with `macros`, on `ndarray >=0.15, <=0.17`, and on
    `libc`, `num-complex`, `num-integer`, `num-traits` and `rustc-hash`.
    It links nothing from NumPy at build time: it loads NumPy's C API
    from the `_ARRAY_API` capsule on first use, so an extension using it
    imports without NumPy installed.
  - **`ndarray` 0.17.2** is MIT OR Apache-2.0, with `rust-version` 1.64.
    It depends on `matrixmultiply`, `num-complex`, `num-integer`,
    `num-traits`, `rawpointer` and `portable-atomic`, all MIT or Apache.
  - **`libm` 0.2.16** is MIT, with `rust-version` 1.63, and has no
    dependencies. It is the pure-Rust port of musl's libm. `f64::erf` is
    still unstable (`float_erf`), checked with rustc 1.98.1.
  - `deny.toml` allows no BSD-2-Clause crate today.

### Consumers and tests

**Python tests.** Counts are collected tests, parametrized cases included:

| File | Lines | Functions | Collected |
|---|--:|--:|--:|
| `expression/passes/test_numpy_evaluator.py` | 1,739 | 93 | 153 |
| `expression/passes/test_evaluator.py` | 651 | 32 | 33 |
| `expression/passes/test_evaluator_properties.py` | 135 | 3 | 3 |
| `expression/test_pprint.py` | 499 | 28 | 57 |
| `expression/test_pprint_properties.py` | 55 | 3 | 3 |

- **`test_numpy_evaluator.py`:**
  - 19 tests match `PassExecutionError` and its cause;
  - 12 handle NumPy's warnings: 11 `np.errstate` blocks and one
    `pytest.warns`;
  - 5 mention `float32`;
  - 8 use the private tables, `_cast_to_result_sort` or the class
    directly;
  - it imports NumPy through `pytest.importorskip` at module level.
- **`test_evaluator.py`:** 7 tests match `PassExecutionError`, and most
  register a user native backed by a `math` callable.
- **The NumPy evaluator is the oracle of other properties.**
  `test_solver_properties.py` (10 references),
  `test_sympy_pass_properties.py` (10), `test_rewrite_properties.py` (4),
  `test_inline_pass_properties.py` (4), `test_piecewise_properties.py` (2)
  and `test_strategies_properties.py` (3) compare with it. The integer
  trees of `tests/strategies/expressions.py` rely on NumPy's int64
  arithmetic. None of these imports NumPy through `importorskip`.
- **Also:**
  - `test_builtins.py` (5 calls of `evaluate_expression`, and
    `test_seeded_native_implementation_matches_math_callable`, which pins
    each built-in's `implementation is math.<f>`);
  - `test_native_stories.py` (8 calls);
  - `test_pprint.py`'s `_BracketedLiterals` and `_NonStringFormatter`
    subclasses, and its `get_noop_output` test;
  - `benchmarks/test_registry.py::test_evaluate_after_inline` and
    `test_pass_infrastructure.py::test_mixed_pipeline_over_a_deep_expression`.

**Rust tests:** none for evaluation. `pprint_stories.rs` and
`pprint_properties.rs` cover the display. `inline_stories.rs` and
`registry_properties.rs` cover the inliner, whose properties evaluate
Boolean trees with a reference evaluator of their own.

**Benchmarks:** none for either evaluator besides
`test_evaluate_after_inline`.

### Divergences visible from Python

Where the Rust semantics differ from today's NumPy and `math` ones
(D-S4-1):

| # | Python today | After S9 |
|---|---|---|
| Z-1 | NumPy's dtype promotion: `float32` kept, every integer and float width | three value domains: `bool`, `int64`, `float64`. Narrower inputs are widened, and results are one of the three (D-S9-4) |
| Z-2 | int64 arithmetic wraps; `x // 0` and `x % 0` on integers are `0` with a warning | an overflow and an integer division by zero are lane failures (D-S9-5, D-S9-6) |
| Z-3 | a Boolean is `0`/`1` in arithmetic and ordering | refused, as S8's lowering refuses it (Y-4); `==` and `!=` compare Booleans |
| Z-4 | a finite out-of-range value cast to `int64` becomes a platform value | a lane failure |
| Z-5 | an error inside an unselected branch: only a non-finite cast is discarded | every lane failure is discarded where a piecewise or a connective does not need that lane (D-S9-6) |
| Z-6 | `erf` and `gelu` are refused by the NumPy evaluator | computed, with `libm`'s `erf` (D-S9-7) |
| Z-7 | the fold computes built-ins with `math`: `sqrt(-1)` raises `ValueError`, `exp(1000)` `OverflowError`, `floor(inf)` `OverflowError` | IEEE results (`NaN`, `inf`), and `NonFiniteCastError` for a non-finite value that an integer-sorted result needs (D-S9-7) |
| Z-8 | the fold never checks arity or argument sorts | it checks both (D-S9-8) |
| Z-9 | NumPy's `RuntimeWarning`s | no warnings |
| Z-10 | `evaluate_expression_with_numpy` wraps walk errors in `PassExecutionError` | it raises them directly (D-S9-10); a run of the pass still wraps them |
| Z-11 | object, complex and string dtypes reach NumPy, and a runtime guard catches some | refused when the environment is converted, with `TypeError` (D-S9-4) |
| Z-12 | 0-d results: a scalar, or a rank-0 `ndarray` for a piecewise or identifier root | always a NumPy scalar; other results are new arrays, never the caller's (D-S9-13) |
| Z-13 | the fold's walk is per occurrence and recursive; so is NumPy's | once per distinct node, on the heap, at any depth |
| Z-14 | messages are sentences (`identifier 'x' is not bound in the environment ...`) | the core's lowercase lines (I.3 rule 3) |

Unchanged in meaning: what is folded and what is kept; the value domains
of the sorts; constants by identifier identity; the refusal of a bound
constant; the Boolean screen and when it runs; the non-finite cast rule;
first-match-wins piecewise; broadcasting; bindings ignored when not
referenced; NumPy as an optional extra.

### Pattern choice

- **The evaluation logic goes to Rust** (decision 2: logic-rich machinery;
  the direction). The walk, the kernels, the casts, the lane failures and
  the checks are specified by Rust tests (decision 4), and Rust users get
  an evaluator.
- **No new P2 or P3 class.**
  - The two passes stay Python `CompilerPass`es whose `run_pass` calls
    `_rs`, as `FunctionInliner` does (D-S7-7, D-S5-12).
  - A user native's Python `implementation` is called from Rust through
    an adapter, once per folded call. This is the fold's per-call
    callback: it existed before, and it calls a user function, not a
    visitor, so P3's granularity rule is not at stake.
- **Plain Python:** `try_get_native_constant_value` and the module
  functions' docstrings.

**Benchmark plan: `benchmarks/test_evaluate.py` (S9.1).** The baseline
runs it against today's Python evaluators. Array rows use seeded
`float64` data, except where a row names another dtype. The deep tree is
S4.1's, 100 operations over four identifiers. It uses float bindings,
since its integer products overflow `int64`.

| Benchmark | Measures |
|---|---|
| `test_evaluate_expression_of_a_builtin_native_call` | `exp(1.0)`: the fold's floor, one pass run |
| `test_evaluate_expression_of_a_user_native_call` | a user native backed by `math.atan2`: the Python callback |
| `test_evaluate_expression_of_the_deep_tree` | nothing to fold: the walk |
| `test_evaluate_expression_of_nested_native_calls` | ten nested built-in natives over a literal |
| `test_evaluate_expression_of_constant_references` | a sum of 100 references to `pi` and `e` |
| `test_evaluate_with_numpy_of_scalars[poly]`, `[sigmoid]`, `[deep_tree]` | scalar environments: the per-call floor |
| `test_evaluate_with_numpy_of_arrays[poly-1e3]`, `[poly-1e6]` | arithmetic throughput |
| `test_evaluate_with_numpy_of_arrays[exp-1e6]`, `[tanh-1e6]`, `[sigmoid-1e6]` | native kernel throughput (N-S9-2) |
| `test_evaluate_with_numpy_of_arrays[piecewise-1e6]` | `{x if x > 0; -x otherwise}`, and a guarded `floor(sqrt(x))` |
| `test_evaluate_with_numpy_of_arrays[integer-1e6]` | `int64` `x // 7 + x % 5` |
| `test_evaluate_with_numpy_of_arrays[logical-1e6]` | `(x > 0) && (y < 1) \|\| !(x == y)` over two arrays |
| `test_evaluate_with_numpy_of_arrays[deep_tree-1e4]` | per-node cost over mid-sized arrays |
| `test_evaluate_with_numpy_of_float32_arrays` | `poly` over 10^6 `float32`s: the widening copy of Z-1 |
| `test_evaluate_with_numpy_with_unused_bindings` | 100 bindings, two referenced |
| `test_pretty_formatter_of_the_deep_tree` | `ExpressionPrettyFormatter()` (N-S9-1) |

Rerun, not added:

- `test_evaluate_after_inline` (`test_registry.py`);
- `test_mixed_pipeline_over_a_deep_expression`
  (`test_pass_infrastructure.py`);
- the three `test_pformat_expression_of_deep_tree` rows
  (`test_expression.py`).

The verdict follows cross-cutting rule 5. The paths at risk:

- **native-heavy large arrays** (`exp`, `tanh`, `sigmoid`), where the
  probe shows plain Rust kernels 3 to 10 times slower than NumPy's SIMD
  ones; that is N-S9-2;
- **`float32` inputs**, which pay a widening copy (Z-1);
- **`evaluate_expression` of tiny trees**, whose floor stays one pass run;
- **environment conversion**, one `numpy.asarray` per referenced binding
  that is not a Python scalar.

### Decisions (proposed 2026-09-26)

Each names the policy it follows:

- D-S4-1: Rust semantics where the two differ;
- D-S4-2: Python names where the meaning is the same;
- "no fallback";
- "tests rewritten, not skipped";
- the crate's conventions in `rust-workspace.md` Part I:
  - owned values, and no global state beyond identity (F-006,
    CONTRIBUTING);
  - `#[non_exhaustive]` errors with one-line lowercase `Display` (I.3
    rule 3);
  - the naming rules (I.3 rule 5) and the layering (§I.2);
  - `unsafe_code = "forbid"` and MSRV 1.85;
- the user's direction for this slice: the NumPy evaluator in Rust, with
  the `numpy` crate and `ndarray`, and NumPy imported lazily as an
  optional extra (cited as "the direction").

Where a decision follows an earlier slice's decision or note, it says so.

- **D-S9-1: one implementation, no fallback** ("no fallback"; D-S7-1,
  D-S8-1).
  - These are deleted, not kept beside the Rust path:
    - the NumPy visitor and its tables;
    - the dtype-kind screen reader;
    - the deferred non-finite masks;
    - the `RewritablePass` fold and its visitor methods;
    - `coerce_literal_value`'s and `is_decimal_text_exactly_binary`'s
      Python bodies;
    - `builtins.py`'s table of `math` callables (D-S9-9).
  - `numpy.py`, `evaluate.py` and `native_lowering.py` become thin
    layers over `_rs`. `pprint.py` follows N-S9-1.
- **D-S9-2: a new core module, `fhy_core::expression::evaluate`**
  (crate conventions; §I.2).
  - It depends on `expression` with `builtins` and `registry`, never on
    `pass`, so it sits below `expression::passes`.
  - The Python paths `passes.evaluate`, `passes.numpy` and
    `passes.native_lowering` map to it in CONTRIBUTING's table.
  - The sketch is settled test-first in S9.2, as D-S7-2's and D-S8-2's
    were:

  ```rust
  // fhy_core::expression::evaluate
  #[derive(Debug, Clone, Copy, PartialEq)]
  pub enum Scalar { Bool(bool), Int(i64), Real(f64) }       // exhaustive: the three domains
  #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
  pub enum Domain { Bool, Int, Real }                        // exhaustive; SymbolType's meaning

  #[derive(Debug, Clone, Copy)]
  pub struct Evaluator<'r> { /* &FunctionRegistry */ }
  impl<'r> Evaluator<'r> {
      pub fn new(registry: &'r FunctionRegistry) -> Self;
      /// evaluate.py: fold native calls with literal arguments, resolve constants.
      pub fn fold(&self, expression: &Expression, natives: &dyn NativeCalls) -> Result<Folding, FoldError>;
      /// Inline, screen, refuse bound constants, then walk (D-S9-10).
      pub fn evaluate<S: BuildHasher>(&self, expression: &Expression,
          environment: &HashMap<Identifier, Scalar, S>) -> Result<Scalar, EvaluationError>;
      #[cfg(feature = "ndarray")]
      pub fn evaluate_array<S: BuildHasher>(&self, expression: &Expression,
          environment: &HashMap<Identifier, ArrayValue<'_>, S>) -> Result<ArrayValue<'static>, EvaluationError>;
  }
  pub trait NativeCalls {                                    // user natives only; built-ins are computed here
      fn call(&self, function: &NativeFunction, arguments: &[LiteralValue]) -> Result<LiteralValue, CallbackError>;
  }
  #[derive(Debug, Clone)]
  pub struct Folding { /* output, not_inlined: Vec<FunctionName> */ }   // output(), into_output(), not_inlined()

  #[cfg(feature = "ndarray")]
  #[derive(Debug, Clone)]
  pub enum ArrayValue<'a> {                                  // exhaustive: the three domains
      Bool(CowArray<'a, bool, IxDyn>), Int(CowArray<'a, i64, IxDyn>), Real(CowArray<'a, f64, IxDyn>),
  }

  #[non_exhaustive] pub enum EvaluationError {
      Inline(InlineError), IllTyped(NonBooleanLogicalOperandError), BoundNativeConstant(Vec<Identifier>),
      Unbound { identifier: Identifier, near_miss: Option<NearMiss> }, InexactDecimal(Decimal),
      IntegerLiteralOutOfRange(BigInt), Unsupported(Callee), BooleanOperand(Expression),
      MixedBranches(Expression), Shape { left: Vec<usize>, right: Vec<usize> },
      Lane { failure: LaneFailure, node: Expression },
  }
  #[non_exhaustive] pub enum LaneFailure {
      IntegerOverflow, DivisionByZero, NegativeIntegerExponent, NonFiniteCast(FunctionSort), OutOfRangeCast(FunctionSort),
  }
  #[non_exhaustive] pub enum FoldError {
      UnknownFunction(FunctionName), NotCallable(FunctionName), Arity { callee: Callee, expected: usize, actual: usize },
      ArgumentSort { callee: Callee, position: usize, sort: FunctionSort, argument: LiteralValue },
      ResultSort { function: FunctionName, sort: FunctionSort, value: LiteralValue }, InexactDecimal(Decimal),
      NonFiniteCast { function: BuiltinFunction, value: f64 }, Native { function: FunctionName, source: CallbackError },
  }
  ```

  - `Decimal` gains `to_f64_exact(&self) -> Option<f64>`: the nearest
    `f64`, and `Some` only when it equals the decimal exactly. This is
    `is_decimal_text_exactly_binary`'s rule, computed without text.
  - `BuiltinFunction` gains the native kernels (D-S9-7). The builtins
    module's "does not compute native functions" becomes "computes its
    19 native functions".
  - Every `Display` is one lowercase line naming the node or name:
    `identifier "x" is not bound`, `integer overflow in (x * y)`,
    `cannot cast NaN to int`, and so on. The texts keep the phrases the
    Python message tests match, as D-S7-12's did.
  - Nothing is global. An `Evaluator` borrows the registry it reads, as
    the inliner does.
- **D-S9-3: one walk, generic over its values; a scalar backend always,
  and an `ndarray` backend behind a feature** (the direction; decision 4;
  crate conventions; the design question of this slice).
  - The walk is written once, over a crate-private trait of value
    operations. It has two implementations:
    - `Scalar`, in every build;
    - `ArrayValue`, under `[features] ndarray = ["dep:ndarray"]` with
      `ndarray = { version = "0.17", optional = true, default-features =
      false, features = ["std"] }`.
  - Both call the same per-element kernels, so lane *i* of an array
    evaluation is, by construction, the scalar evaluation with lane *i*'s
    bindings. A Rust property pins it.
  - The walk keeps its pending nodes on the heap and evaluates each
    distinct node once, as the inliner does.
  - The binding enables `ndarray`. rust-numpy lives in the binding only,
    so `fhy-core` never depends on PyO3 (CONTRIBUTING).
  - `ndarray` 0.17's types appear in the feature's API. The crate is
    unpublished (I.3 rule 6), and the README says so. docs.rs builds the
    default features, as for S8's `z3`.
  - **Rejected alternatives:**
    - *The evaluator only in the binding:* no Rust test could specify it,
      Rust users would get nothing, and the fold would still need its
      kernels in the core.
    - *A Rust walk calling NumPy's ufuncs per node:* NumPy's semantics
      instead of Rust's (Z-1 to Z-4 unchanged), against D-S4-1 and the
      direction.
    - *An `ndarray`-only evaluator with 0-d arrays for scalars:* every
      evaluation would need the feature, and every node of a scalar
      evaluation would allocate.
- **D-S9-4: three value domains** (D-S4-1: the core's sorts and
  literals; Z-1, Z-11).
  - A value is `Bool`, `Int` (`i64`) or `Real` (`f64`), the domains of
    `SymbolType`.
  - **Literals:**
    - a `bool` is `Bool`;
    - an integer is `Int`, or `IntegerLiteralOutOfRange` outside `i64`;
    - a float is `Real`;
    - a decimal is `Real` through `to_f64_exact`, or `InexactDecimal`.
  - **Constants:** a built-in constant is `Real`, and a user constant
    converts its `LiteralValue` the same way.
  - **Array bindings** convert by dtype (the binding, D-S9-12):
    - `bool_` is `Bool`;
    - every signed integer width, and unsigned up to 32 bits, is `Int`;
    - `uint64` is `Int`, or `OverflowError` for a value above
      `i64::MAX`;
    - `float16`, `float32` and `float64` are `Real`.

    Object, complex, string, bytes, datetime and void dtypes are
    refused with `TypeError`, naming the identifier and the dtype.
  - Narrower inputs are widened, since the core's real domain is `f64`
    (Z-1). A fourth domain for `float32` could be added later without
    changing these rules; a benchmark row measures the widening.
- **D-S9-5: operation semantics** (D-S4-1: the operations' documented
  meaning, F-010; S8's Y-4 for Booleans; Z-2, Z-3). Mixed `Int` and `Real`
  operands convert the `Int` to the nearest `f64` in every operation,
  comparisons included, as S8's `to_real` does.

  | Operation | `Int`, `Int` | with a `Real` |
  |---|---|---|
  | `+`, `-`, `*`, negation | checked; overflow is a lane failure | IEEE |
  | `/` (`Divide`) | the quotient of the two `f64`s, `Real` | IEEE; `x / 0` is `inf` or `NaN` |
  | `//` (`FloorDivide`) | rounds toward negative infinity; `x // 0` is a lane failure; `i64::MIN // -1` overflows | the floor of the exact quotient, by the `fmod`-based `divmod` NumPy and Python use (`1 // 0.1` is `9`); `x // 0` is IEEE's `a / b` |
  | `%` (`FloorMod`) | the sign follows the divisor; `x % 0` is a lane failure | `fmod` with the divisor's sign; `x % 0` is `NaN` |
  | `**` (`Power`) | a non-negative exponent: checked `pow`; a negative one: a lane failure (`ValueError`, as NumPy) | `powf` |
  | `==`, `!=` | exact | IEEE after conversion |
  | `<`, `<=`, `>`, `>=` | exact | IEEE after conversion |

  - **Booleans:**
    - `==` and `!=` compare two `Bool`s;
    - a `Bool` in arithmetic, in an ordering, beside a number in `==`,
      under negation or `+x`, or a piecewise whose branches mix `Bool`
      with a number, is refused (`BooleanOperand`, `MixedBranches`),
      naming the node;
    - `Int` and `Real` branches mix as `Real`.
  - **Connectives and conditions:** a connective reduces its operands in
    order with `&&` or `||`, and `!` negates. A piecewise selects the
    first case whose condition holds, per lane.
  - **Broadcasting** follows NumPy's rules. A shape mismatch is `Shape`,
    raised as `ValueError`.
- **D-S9-6: failures are per lane, and a guard discards them** (D-S4-2:
  it extends today's non-finite deferral to the failures Rust semantics
  add; Z-2, Z-4, Z-5).
  - An integer overflow, an integer division by zero, a negative integer
    exponent, or a cast of a non-finite or out-of-range value to an
    integer or Boolean sort marks its lane as failed, and the lane
    carries a zero.
  - A lane failure is discarded:
    - by a piecewise, when that lane selects another branch;
    - by `&&`, when another operand is `false` in that lane;
    - by `||`, when another operand is `true` in that lane.

    So `{x // y if y != 0; 0 otherwise}` and `(y != 0) && (x // y > 1)`
    work over integer arrays, as they do with NumPy today (without its
    garbage lanes). Every other node passes a failure on.
  - A failure that reaches the result raises `Lane`, for the first
    failed lane in C order, naming the failing node.
  - A scalar is one lane, so the scalar backend meets the same rule.
  - A static refusal (an out-of-range literal, an inexact decimal, a
    Boolean operand, a shape mismatch) raises at once.
- **D-S9-7: the native kernels are the core's** (D-S4-1; Z-6, Z-7).
  - `exp`, `exp2`, `log`, `log2`, `log10`, `sqrt`, the trigonometric and
    hyperbolic functions, `floor` and `ceil` are the `std` `f64` methods.
    These call the platform's libm, as Python's `math` does.
  - `round` is `f64::round_ties_even`, Python's rounding (stable since
    Rust 1.77).
  - `erf` is `libm::erf`, a new dependency of `fhy-core` (MIT, no
    dependencies). `erf` and `gelu` become evaluable over arrays too.
  - Results are IEEE: `sqrt(-1)` is `NaN`, `log(0)` is `-inf`, and
    `exp(1000)` is `inf`.
  - An integer-sorted result (`round`, `floor`, `ceil`) needs a finite
    value:
    - the fold makes it an exact `BigInt`, or `NonFiniteCast`;
    - the evaluator makes it an `i64`, or a lane failure
      (`NonFiniteCast`, `OutOfRangeCast`).
  - A built-in's integer argument converts to the nearest `f64`, and
    becomes `±inf` beyond its range.
  - How the array evaluator computes the transcendental kernels is
    N-S9-2.
- **D-S9-8: the fold** (D-S4-2 for what it folds; D-S4-1 for how;
  D-S7-7's inliner for the walk; Z-7, Z-8, Z-13). `Evaluator::fold` is
  data-only, apart from the `NativeCalls` it is given.
  - It walks bottom-up on its own stack, handles each distinct node
    once, and returns the input itself when it folds nothing.
  - **A call:**
    - an unknown name is `UnknownFunction`, and a constant's name is
      `NotCallable`;
    - a composed built-in or a user function is kept, and its name is
      recorded in `not_inlined`, once, in first-reached order;
    - a native call whose arguments are all literals is checked for
      arity and argument sorts (`FunctionSort::accepts_literal`), then
      folded. A built-in is computed by its kernel, and a user native by
      `NativeCalls::call`, whose result is checked against the result
      sort (`ResultSort`).
  - **A built-in or user constant's identifier** becomes its value.
  - **Decimals:** a decimal argument converts through `to_f64_exact` or
    is `InexactDecimal`, as `coerce_literal_value` does.
  - **Order of checks:** arguments first, then the lookup, then arity,
    then sorts, then the call, as Python's order (and the inliner's) is.
- **D-S9-9: the built-in entries' `implementation` is the core's kernel**
  ("no fallback"; D-S7-9's entry objects; Z-7).
  - The 19 native built-in entries hold a small Rust-backed callable,
    `BuiltinNativeImplementation`, instead of a `math` callable:
    - calling it computes the kernel of D-S9-7 on a `bool`, `int` or
      `float`, as the fold does;
    - it has `__name__` and a `repr` naming the function;
    - it pickles by name;
    - it compares by identity, one object per built-in.
  - So `BUILTIN_FUNCTIONS["sqrt"].implementation(-1.0)` agrees with
    `evaluate_expression`, and no second implementation of a built-in
    remains. `_NATIVE_IMPLEMENTATIONS` and `_exp2` are deleted, and
    `NativeFunction._install_builtins()` takes no table.
  - A user native keeps its Python callable.
- **D-S9-10: `evaluate_expression_with_numpy` is one call into the
  extension, and raises its errors directly** (D-S4-2 for the name and
  signature; S8's Y-7 for unwrapped errors; the benchmark rationale; Z-10).
  - The call runs these in order:
    1. the NumPy check (D-S9-11);
    2. inlining;
    3. the conversion of the bindings the inlined tree references
       (D-S9-12);
    4. the screen, over the `SymbolType`s of the converted domains;
    5. the bound-constant refusal;
    6. the walk.

    That is today's order: the conversion takes the place of reading
    each binding's dtype for the screen.
  - Every error is raised as its own class (D-S9-14), not wrapped. The
    function thereby skips two pass lifecycles, a large part of today's
    30 µs scalar floor.
  - **`NumpyExpressionEvaluator(environment)`** stays the registered pass
    `fhy_core.symbolic.expression.evaluate_with_numpy`:
    - it is a `CompilerPass[Expression, Any]`, no longer a
      `VisitablePass`;
    - it snapshots the mapping at construction, and its `run_pass` makes
      the same call;
    - the pass framework wraps its errors as it wraps every pass's;
    - `did_change` stays `True`, and `get_noop_output` still raises;
    - the `numpy_module` argument goes, since rust-numpy finds NumPy
      itself;
    - it now inlines, so it evaluates composed calls instead of refusing
      them.
- **D-S9-11: NumPy stays optional and is imported lazily** (the
  direction; S8's D-S8-15 and D-S8-16 for the pattern).
  - The extension imports and works without NumPy: rust-numpy touches
    NumPy's C API only on first use.
  - `evaluate_expression_with_numpy` and the pass run `import numpy`
    first, on every call. That honors `sys.modules["numpy"] = None` and
    costs a `sys.modules` lookup. On failure they raise today's
    `ImportError`, with its `pip install fhy_core[numpy]` guidance and
    the original as `__cause__`.
  - `import fhy_core`, `evaluate_expression` and every other path never
    import NumPy. A fresh-interpreter test pins that.
  - The `numpy` extra stays. S9.4 checks the oldest NumPy rust-numpy 0.29
    supports and gives the extra that lower bound if one is needed.
  - The marker and the minimal session follow D-S8-17 (S9.7).
- **D-S9-12: the binding's conversions** (the direction; crate
  conventions: no `unsafe`; D-S8-11 for detaching).
  - **Only referenced bindings** of the inlined tree are converted, as
    today.
  - **Python values:**
    - a `bool`, `int` or `float` converts directly, with no NumPy call;
    - anything else goes through `numpy.asarray`. A `bool_`, `int64` or
      `float64` array in native byte order is then borrowed as a
      read-only `ndarray` view, whatever its strides, with no copy;
    - any other admitted dtype is cast once, by NumPy, to one of the
      three.
  - **The result** is an owned `ndarray`, moved into a new NumPy array
    without a copy (`PyArray::from_owned_array`).
  - **Threads.** The binding detaches from the interpreter while it
    walks, so other Python threads run. The input views stay borrowed
    through rust-numpy's borrow checking; as with NumPy's own ufuncs,
    another thread writing to an input array meanwhile is the caller's
    race.
  - Every rust-numpy call the binding needs is in its safe API. The
    binding gains `numpy = "0.29"` and uses `numpy::ndarray`, which Cargo
    unifies with the core's 0.17.
- **D-S9-13: result types** (D-S4-1; NumPy's scalar convention; Z-12).
  - `Bool` is `numpy.bool_`, `Int` is `numpy.int64`, and `Real` is
    `numpy.float64`.
  - A 0-d result is a NumPy scalar of that type. Any other result is a
    new C-contiguous, writeable array that owns its data, never an input
    array.
- **D-S9-14: errors are the core's text under the Python classes**
  (D-S4-1, D-S7-12, D-S8-14; Z-14).

  | Core | Python |
  |---|---|
  | `Inline` | as S7 maps `InlineError`: `EntryLookupError`, `FunctionArityError`, `RecursionError` |
  | `IllTyped` | `NonBooleanLogicalOperandError` |
  | `BoundNativeConstant` | `NativeConstantBindingError` |
  | `Unbound` | `UnboundVariableError`, keeping the three wordings as `near_miss` |
  | `InexactDecimal` | `StringLiteralPrecisionError` |
  | `IntegerLiteralOutOfRange`, `Lane(IntegerOverflow)`, `Lane(OutOfRangeCast)` | `OverflowError` |
  | `Lane(DivisionByZero)` | `ZeroDivisionError` |
  | `Lane(NegativeIntegerExponent)`, `Shape` | `ValueError` |
  | `Lane(NonFiniteCast)`, `FoldError::NonFiniteCast` | `NonFiniteCastError` |
  | `Unsupported` (a user native in the evaluator) | `UnsupportedNumpyLoweringError` |
  | `BooleanOperand`, `MixedBranches`, a refused dtype, `ArgumentSort` | `TypeError` |
  | `FoldError::UnknownFunction` | `EntryLookupError` |
  | `FoldError::NotCallable`, `FoldError::Arity` | `FunctionArityError` |
  | `FoldError::ResultSort` | `NativeResultSortError` |
  | `FoldError::Native` | the implementation's exception itself; `KeyboardInterrupt` passes through |

  `UnsupportedNumpyLoweringError`'s and `NonFiniteCastError`'s docstrings
  follow Z-6 and D-S9-7.
- **D-S9-15: `ExpressionEvaluator` becomes a `CompilerPass` over the core
  fold** (D-S7-7's `FunctionInliner`; D-S5-12; D-S4-2 for the names).
  - It is `CompilerPass[Expression, Expression]`, registered as today.
  - Its `run_pass` calls `_rs`, then `report`s one WARNING per name in
    `not_inlined`, with today's advice to run `inline_functions` first.
  - `did_change` is by identity, and the output is materialized beside
    the input, as the inliner's is.
  - `evaluate_expression` still runs the pass, so its errors stay
    `PassExecutionError`s with the cause, as `inline_functions`' do.
  - It is no longer a `RewritablePass`: `visit_identifier_expression`
    and `visit_call_expression` go.
- **D-S9-16: `native_lowering.py` keeps its three names** (D-S4-2).
  - `is_decimal_text_exactly_binary(text)` and
    `coerce_literal_value(value)` call `_rs`, over `Decimal::to_f64_exact`
    and the literal conversion, with the core's text in
    `StringLiteralPrecisionError`. They accept today's inputs:
    integer- and float-grammar text, `Decimal`, `bool`, `int` and
    `float`.
  - `try_get_native_constant_value` stays two lines of Python over the
    registry lookups.
  - `passes/sympy.py`'s import is unchanged, so S8's file needs no edit.
- **D-S9-17: the formatter** follows N-S9-1. `pformat_expression` is
  unchanged: it already renders through the core.
- **D-S9-18: Rust tests specify the evaluators first** (the tests rule;
  S7.2's and S8.2's test-first practice). The fold, the scalar walk, the
  kernels, the lane failures and the array backend are specified by Rust
  tests written against `todo!()` stubs, with a traceability table from
  `test_numpy_evaluator.py`, `test_evaluator.py` and
  `test_evaluator_properties.py`.
- **D-S9-19: the Python tests are rewritten, not skipped** (the tests
  rule). The behavioral tests stay, and change only where a decision
  changes what they pin; each change is recorded with its reason, as S4.4
  to S8 did.
- **D-S9-20: edits to files S8 also changes stay small and additive**
  (the coordination the user asked for). The branch is rebased onto
  `dev-rust` after S8 lands.
  - **Files S9 never touches:** `symbolic/solver.py`, `passes/z3.py`,
    `passes/sympy.py`, `symbolic/expression/__init__.py`, and S8's
    Rust modules.
  - **Shared files, and what S9 adds to each:**
    - `Cargo.toml`: three `[workspace.dependencies]` lines (`numpy`,
      `ndarray`, `libm`);
    - `rust/fhy-core/Cargo.toml`: the `libm` and optional `ndarray`
      dependencies, and the `ndarray` line of `[features]`, a table S8
      also creates, for `z3`;
    - `rust/fhy-core-py/Cargo.toml`: `numpy`, and `features =
      ["ndarray"]` on `fhy-core`;
    - `Cargo.lock`: regenerated after the rebase, never hand-merged;
    - `deny.toml`: `"BSD-2-Clause"` in `allow`, for rust-numpy;
    - `rust/fhy-core-py/src/lib.rs`: one new `#[pymodule_export]` block;
    - `src/fhy_core/_rs.pyi`: one new section;
    - `rust/fhy-core/src/lib.rs`: the `expression` row of the module
      table;
    - CONTRIBUTING: the Python-to-Rust table's rows;
    - the crate README: a feature paragraph beside S8's;
    - `rust/fhy-core/tests/it/expression.rs`: new test modules, not
      `main.rs`, which S8 edits.
  - **The Python-side extras step** (the marker, `conftest.py`,
    `tests_minimal`, and the README's install line and expression row,
    which S8.5 and S8.7 edit) is S9.7, done after the rebase on S8's
    versions.
  - This checklist entry and this section sit after S8's, so the rebase
    conflicts only where both append.

### Needs the user

- **N-S9-1: whether `ExpressionPrettyFormatter` stays a Python
  `VisitablePass`.** N-S6-3 kept it Python for its per-node overrides.
  This slice's scope names `pprint.py`, and "no fallback" argues against
  a second renderer of the core's text. The policy does not settle an
  earlier resolution against a later scope.
  - (a) **Native.** The class becomes a `CompilerPass[Expression, str]`
    whose `run_pass` calls the core, with today's constructor. A subclass
    that defines a `visit_*` method is refused when the class is created,
    with a `TypeError` naming this decision, so no override is ignored
    silently. `_BracketedLiterals`, `_NonStringFormatter`, the
    `get_noop_output` test and the agreement property are rewritten, and
    the formatter's run takes the pass floor instead of 223 µs.
  - (b) **Keep N-S6-3.** The formatter stays a Python `VisitablePass`, and
    S9 changes nothing in `pprint.py`.

  Recommendation: (a). No `src` module subclasses or runs the formatter,
  the core already owns its text, and the property test exists only
  because there are two renderers. It records a revision of N-S6-3.
- **N-S9-2: how the array evaluator computes the transcendental natives**
  (D-S9-7; cross-cutting rule 5). NumPy dispatches SIMD kernels for
  float64 `exp`, `log` and `tanh` at run time. The wheels target baseline
  x86-64, where the probe measured `std`'s kernels 4.7, 3.4 and 9.9 times
  slower than NumPy's over 10^6 lanes (`sin` 1.1, `sqrt` 1.0). Arithmetic,
  comparisons and piecewise do not depend on this; scalar environments
  gain either way.
  - (a) **The core's kernels everywhere.** One kernel set for the fold,
    the scalar walk and arrays, specified in Rust. Native-heavy large
    arrays get slower: about 2 times for `sigmoid`, and 4 to 10 times for
    a lone `exp` or `tanh`. That is recorded as an accepted cost.
  - (b) **NumPy's ufuncs as the array kernels of the 14 transcendental
    natives, plugged in by the binding.**
    - `sqrt`, `round`, `floor` and `ceil` keep the core's kernels, which
      match NumPy's speed, and NumPy has no `erf`.
    - The core's array backend takes an `ArrayKernels` trait whose
      default is the core's kernels, so Rust users and the Rust tests
      keep (a).
    - The binding's implementation reattaches to the interpreter at each
      native node, hands NumPy the lane array (moved without a copy when
      it is a temporary; copied when it is a borrowed input), and copies
      the result back.
    - Native-heavy arrays stay near today: a lone `exp` over 10^6 lanes
      is estimated at 1.5 to 1.9 times today's 1.07 ms, and `sigmoid`
      near today's 3.5 ms.
    - Array natives keep NumPy's accuracy, which already differs from
      `math`'s by a few ULPs today, while scalars use the core's.
  - (c) **(a) now, and SIMD kernels in the core in a later slice**, once
    a runtime-dispatched kernel set with documented error bounds is
    chosen. No stable-Rust crate found in this survey provides one.

  Recommendation: (b), since the user named array throughput as a goal,
  and it is the only option that keeps native-heavy arrays near today's
  speed while the rest of the walk runs in Rust. (a) is the simpler
  design if one kernel set matters more than that throughput.

### S9 resolutions (decided by the user, 2026-09-26)

- **N-S9-1: (a) native.** `ExpressionPrettyFormatter` becomes a
  `CompilerPass[Expression, str]` whose `run_pass` renders through the
  core, and a subclass defining a `visit_*` method is refused at class
  creation. This revises N-S6-3, which kept the formatter a Python
  `VisitablePass`; N-S6-3's decision for `VisitablePass`,
  `AnalysisVisitablePass` and `RewritablePass` stands.
- **N-S9-2: (b) NumPy's ufuncs for the 14 transcendental natives.** The
  core's array backend takes an `ArrayKernels` trait, whose default is the
  core's own kernels, so Rust users and the Rust tests keep them. The
  binding plugs in NumPy's ufuncs for `exp`, `exp2`, `log`, `log2`,
  `log10`, `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`, `sinh`,
  `cosh` and `tanh`; `sqrt`, `round`, `floor`, `ceil` and `erf` keep the
  core's kernels.
- **`deny.toml` allows BSD-2-Clause**, the license of rust-numpy (the
  `numpy` crate), with a comment naming this slice. It is the only
  license S9's crates add: `ndarray`, `libm` and rust-numpy's other
  dependencies are MIT or Apache-2.0.

### S9.1 baseline (2026-09-26, da8a9c1 plus the new benchmarks)

`benchmarks/test_evaluate.py` implements the benchmark plan above. The
piecewise row has a second case, `guarded-1e6`, the guarded
`floor(sqrt(x))` the plan names beside it. The integer row draws `int64`s
in [-10^6, 10^6), and the deep-tree rows bind floats in [0, 1).

Median time per call, from `.venv/bin/python -m pytest
benchmarks/test_evaluate.py <the rerun rows> -n 0 --benchmark-only` in the
worktree's own environment (the release build uv installs), measuring
today's Python evaluators and formatter. The machine is the S0 one, with
Python 3.11.13, NumPy 2.4.6 and pytest-benchmark 5.3.0. The load average
was below 1.6, and the table lists the best of three runs' medians.

| Benchmark | before |
|---|--:|
| `test_evaluate_expression_of_a_builtin_native_call` | 10.3 µs |
| `test_evaluate_expression_of_a_user_native_call` | 7.4 µs |
| `test_evaluate_expression_of_the_deep_tree` | 196 µs |
| `test_evaluate_expression_of_nested_native_calls` | 37.7 µs |
| `test_evaluate_expression_of_constant_references` | 397 µs |
| `test_evaluate_with_numpy_of_scalars[poly]` | 30.7 µs |
| `test_evaluate_with_numpy_of_scalars[sigmoid]` | 55.6 µs |
| `test_evaluate_with_numpy_of_scalars[deep_tree]` | 484 µs |
| `test_evaluate_with_numpy_of_arrays[poly-1e3]` | 35.5 µs |
| `test_evaluate_with_numpy_of_arrays[poly-1e6]` | 6.71 ms |
| `test_evaluate_with_numpy_of_arrays[exp-1e6]` | 890 µs |
| `test_evaluate_with_numpy_of_arrays[tanh-1e6]` | 1.76 ms |
| `test_evaluate_with_numpy_of_arrays[sigmoid-1e6]` | 5.83 ms |
| `test_evaluate_with_numpy_of_arrays[piecewise-1e6]` | 12.85 ms |
| `test_evaluate_with_numpy_of_arrays[guarded-1e6]` | 20.42 ms |
| `test_evaluate_with_numpy_of_arrays[integer-1e6]` | 19.99 ms |
| `test_evaluate_with_numpy_of_arrays[logical-1e6]` | 2.02 ms |
| `test_evaluate_with_numpy_of_arrays[deep_tree-1e4]` | 758 µs |
| `test_evaluate_with_numpy_of_float32_arrays` | 1.67 ms |
| `test_evaluate_with_numpy_with_unused_bindings` | 42.1 µs |
| `test_pretty_formatter_of_the_deep_tree` | 246 µs |
| `test_evaluate_after_inline` | 62.1 µs |
| `test_mixed_pipeline_over_a_deep_expression` | 286 µs |
| `test_pformat_expression_of_deep_tree[symbolic]` | 8.3 µs |
| `test_pformat_expression_of_deep_tree[functional]` | 7.8 µs |
| `test_pformat_expression_of_deep_tree[show_id]` | 10.1 µs |

- **The fold** costs 10 µs for one built-in call and 7 µs for a user
  native; the walk over the 191-node deep tree 196 µs, and the sum of 100
  constant references 397 µs, a Python visitor call per occurrence.
- **Scalar environments** cost 31 µs for the polynomial and 484 µs for
  the deep tree: a Python visitor call and a NumPy call per node.
- **Arrays.** The million-element polynomial takes 6.7 ms, NumPy's four
  temporaries; `exp` and `tanh` are NumPy's SIMD kernels, 0.9 and 1.8 ms.
  The piecewise rows take 13 and 20 ms: two `numpy.where`s per case, the
  non-finite masks and the casts. The integer row takes 20 ms, NumPy's
  integer floor division and modulo.
- **The formatter** takes 246 µs over the deep tree, where
  `pformat_expression` takes 8 µs.

### Steps

1. **S9.1: benchmarks.** Add `benchmarks/test_evaluate.py` as planned
   above, and record the baseline here, on today's Python evaluators.
2. **S9.2: core additions, test-first, with Rust tests.**
   - `fhy_core::expression::evaluate` (`evaluate.rs`) with:
     - `evaluate/value.rs` (`Scalar`, `Domain`, the literal
       conversions);
     - `evaluate/kernel.rs` (the per-element kernels, crate-private);
     - `evaluate/walk.rs` (the generic walk and its value trait);
     - `evaluate/fold.rs`;
     - `evaluate/error.rs`.
   - `Decimal::to_f64_exact`, and the native kernels on
     `BuiltinFunction`.
   - The `libm` dependency.
   - The tests are written first and fail against `todo!()` stubs, as in
     S7.2. `lib.rs`, the crate README and CONTRIBUTING's table list the
     module. Nothing in Python changes, so the suite stays green.
3. **S9.3: the `ndarray` feature.**
   - `evaluate/array.rs` under `#[cfg(feature = "ndarray")]`, with its
     tests under the same `cfg`, and the manifest and README.
   - Under N-S9-2 (b), the `ArrayKernels` trait.
   - CI's `rust` job already builds `--all-features`, and `deny` checks
     the graph; `rust-msrv` checks the workspace, whose binding enables
     the feature from S9.4 on.
4. **S9.4: the binding.** Add `rust/fhy-core-py/src/expression/evaluate.rs`
   with:
   - `fold.rs`: the `NativeCalls` adapter over the registry state's
     Python entries, and the `_rs` fold;
   - `numpy.rs`: the NumPy check, the conversions of D-S9-12, the
     results of D-S9-13, detaching, and, under N-S9-2 (b), the NumPy
     kernels;
   - `builtins.rs`: `BuiltinNativeImplementation` (D-S9-9);
   - `literal.rs`: the two `native_lowering` functions;
   - `deny.toml` gains BSD-2-Clause.

   Everything new goes into `_rs.pyi`. Nothing in Python uses it yet, so
   the suite stays green.
5. **S9.5: the Python switch** (marked breaking).
   - `numpy.py`, `evaluate.py` and `native_lowering.py` become the thin
     layers.
   - `builtins.py` drops its table.
   - `pprint.py` follows N-S9-1.
   - `errors.py`'s two docstrings change.

   The switch lands together with S9.6 when the migration is small
   enough to review in one commit; otherwise it leaves exactly the tests
   of the migration table failing, as S7.4 did.
6. **S9.6: tests.** Migrate the tests and add the interface suite (the
   test plan below).
7. **S9.7: after the rebase onto S8.**
   - A `numpy` marker, and `conftest.py`'s skip for it, in S8's scheme.
   - `tests_minimal` also leaves NumPy out, and its CI job stays green.
   - The README's install line and expression row, which S8 rewrites.
8. **S9.8: benchmarks after,** recorded here with the verdict, then the
   status, the implementation notes and this checklist.

Commit per step. Every step ends with these green or clean:

- `pytest`, and `-m "not very_slow"`;
- the `property` session, `lint` and `type_check`;
- `tests/test_rs_stub.py`;
- the Rust gate: fmt, clippy `-D warnings`, tests, doc `-D warnings`,
  deny, and `cargo +1.85 check`, with `--all-features` covering
  `ndarray` from S9.3 on.

### Test plan

**Rust tests, written first (S9.2 and S9.3),** in
`tests/it/expression/`:

- **`fold_stories.rs`:**
  - each of the 19 built-ins folded, against pinned `f64` values;
  - `round` half to even;
  - integer-sorted results as exact `BigInt`s (`floor(1e300)`), and
    `NonFiniteCast` for `NaN` and the infinities;
  - an integer argument's conversion, `±inf` beyond `f64`;
  - user natives through a recording fake `NativeCalls`: the arguments,
    once per distinct node, a `ResultSort` refusal, and a callback error
    kept as the source;
  - arity and argument sorts;
  - unknown and not-callable names;
  - `not_inlined`, once per name in order;
  - constants, built-in and user, and a look-alike left alone;
  - inexact decimals;
  - the input itself back when nothing folds (`ptr_eq`);
  - a shared DAG folded once per distinct node;
  - a 100,000-level tree on a small stack;
  - each `Display`.
- **`literal_stories.rs` additions:** `to_f64_exact` over `0.5`, `0.1`,
  long texts, large and small exponents, the subnormal edge, and values
  beyond `f64::MAX`.
- **`evaluate_stories.rs` (scalar):**
  - every row of D-S9-5's table: Python's `divmod` reference values for
    both signs and both domains, and `1 // 0.1`;
  - overflow at `i64::MAX` and `i64::MIN`;
  - the power rules;
  - mixed promotion in each position;
  - Boolean refusals and `==` of Booleans;
  - n-ary connectives, and first-match piecewise;
  - each lane failure, raised, and discarded by a piecewise and by each
    connective (D-S9-6);
  - the static refusals;
  - the order of the checks: inline errors, then the screen, then the
    bound-constant refusal, then the walk;
  - `Unbound` with each near miss;
  - `erf` and `gelu`;
  - a deep tree on a small stack.
- **`evaluate_array_stories.rs`,** under `cfg(feature = "ndarray")`:
  - broadcasting: 0-d, empty, higher rank, and a mismatch;
  - result domains;
  - borrowed views not copied (the output shares nothing with an input,
    and an input view's pointer is the one given);
  - lane failures in selected and unselected lanes, and nested
    piecewise, one story per guard test of `test_numpy_evaluator.py`;
  - the first failed lane in C order;
  - a million-lane run;
  - under N-S9-2 (b), a fake `ArrayKernels` receiving the lanes.
- **`evaluate_properties.rs`:**
  - lane *i* of a random array evaluation equals the scalar evaluation
    of lane *i*'s bindings. The trees use every operation, and the lanes
    include zeros, `NaN`s, infinities and overflow-prone integers
    (feature-gated);
  - folding, then evaluating, agrees with evaluating;
  - evaluation after inlining agrees with a reference evaluation of the
    composed built-ins, as `registry_properties.rs` does for Booleans.
- A traceability table maps the three Python evaluator test files to
  them, as S4.2, S7.2 and S8.2 did.

**The interface suite,
`tests/symbolic/expression/passes/test_evaluate_rust_binding.py`,** covers
what the binding adds over the core:

- **NumPy optional.** In a subprocess with `sys.modules["numpy"] = None`:
  - `import fhy_core` and `evaluate_expression` work;
  - `evaluate_expression_with_numpy` raises the guiding `ImportError`.

  A fresh `import fhy_core` leaves `numpy` out of `sys.modules`, and so
  does a fold.
- **Conversions:**
  - every dtype of D-S9-4, admitted or refused, and `uint64` at the edge;
  - native and swapped byte order, and Fortran order;
  - negative and zero strides, 0-d, empty;
  - Python scalars and nested lists;
  - an unreferenced ragged binding ignored;
  - a Python `int` beyond `i64`.
- **Results:** 0-d NumPy scalars of the three types; new, writeable,
  C-contiguous arrays that own their data; never the input array.
- **Errors:** each row of D-S9-14 under its class, raised directly by the
  function and wrapped by a pass run; the order of the checks.
- **The passes:**
  - both are registered;
  - `NumpyExpressionEvaluator` snapshots its environment, and its run
    inlines;
  - `ExpressionEvaluator` reports its WARNINGs through `report`, and
    `did_change` is by identity.
- **User natives:** the implementation receives Python values; its
  exception propagates as the same object; `KeyboardInterrupt` passes
  through; a wrong result type is refused.
- **Built-in implementations (D-S9-9):**
  - calling them agrees with the fold;
  - one object per built-in;
  - `repr`, pickling, and a `NaN` for `sqrt(-1.0)`.
- **Threads:** another Python thread makes progress during a large
  evaluation; concurrent evaluations agree.
- **The formatter,** per N-S9-1.

**Migrating the existing tests.** No test is skipped, or deleted without
a rewrite, and each change is recorded with its reason:

- **`test_numpy_evaluator.py` (153):**
  - The 19 `PassExecutionError` tests match the error itself (D-S9-10).
  - The 12 `errstate` blocks and warning checks go, or pin that no
    warning is raised (Z-9).
  - `test_real_sort_native_preserves_float_width` pins `float64` (Z-1).
  - The three `_cast_to_result_sort` tests become Rust cast stories;
    no built-in is `NAT`- or `BOOL`-sorted, so Python cannot reach them.
  - The four lowering-table tests become behavior tests over every
    operation and every native built-in.
  - The two object-dtype backstop tests pin the conversion's `TypeError`
    (Z-11).
  - The `erf` and `gelu` refusals pin their values against `math.erf`
    (Z-6).
  - The integer-power test keeps `ValueError` under the core's text.
  - The snapshot test drops the `numpy` argument.
  - The piecewise- and identifier-root result tests pin scalars (Z-12).
- **`test_evaluator.py` (33)** keeps its `PassExecutionError`s (D-S9-15).
  Its message tests follow the core's texts, and a test pins the arity
  check that replaces the callable's `TypeError` (Z-8).
- **`test_evaluator_properties.py` (3)** keeps its meaning.
- **`test_builtins.py`:** the `math`-identity test becomes the agreement
  of each built-in's `implementation` with `math` on finite in-domain
  inputs, and with IEEE outside them (D-S9-9). The table-mutation test
  goes with the table, rewritten to pin that the entries are frozen.
- **The property oracles** (solver, sympy pass, rewrite, inline,
  piecewise, strategies) change only where Z-2 or Z-3 changes what they
  compare. An integer tree whose evaluation overflows `int64` is no
  sample for a value comparison, and is assumed away or drawn from
  bounded values. The numpy-reaching tests get the `numpy` marker in
  S9.7.
- **`test_pprint.py`** follows N-S9-1.

### S9.2 implementation notes

The tests were written first, against `todo!()` stubs of
`BuiltinFunction::native_value`, `Decimal::to_f64_exact`,
`Evaluator::fold`, `Evaluator::prepare`, `Prepared::evaluate` and (in
S9.3) `Prepared::evaluate_array`: all 159 new tests of S9.2 and S9.3
failed, and all pass now. The new module is
`rust/fhy-core/src/expression/evaluate.rs`, with `evaluate/value.rs`
(`Scalar` and the literal conversion), `evaluate/kernel.rs` (the per-lane
integer and real kernels), `evaluate/lanes.rs` (the backend trait and the
scalar backend), `evaluate/walk.rs` (the generic walk), `evaluate/fold.rs`
and `evaluate/error.rs`. `lib.rs`, the crate README and CONTRIBUTING's
table list it.

Where the shape differs from D-S9-2's sketch, or fills it in:

- **No `Domain` type.** The three domains are `SymbolType`'s, whose meaning
  is the same, so `Scalar::symbol_type` returns a `SymbolType`, and the
  screen reads the bindings' `SymbolType`s directly (D-S4-2).
- **`Evaluator::prepare` and `Prepared`.** The binding converts only the
  bindings the *inlined* tree refers to (D-S9-12), so inlining is its own
  step: `prepare` inlines and returns a `Prepared` holding the inlined tree
  and its free identifiers, whose `evaluate` (and, in S9.3,
  `evaluate_array`) screens, refuses bound constants and walks.
  `Evaluator::evaluate` is `prepare` then `evaluate`.
- **`Folding::not_inlined` lists `Callee`s,** composed built-ins included,
  since Python's evaluator warns for both kinds today.
- **The walk** (`walk.rs`) is written once over a crate-private `Lanes`
  trait with a generic associated type `Of<T>`, the container of lanes:
  `T` itself for the scalar backend, and a `CowArray` for the array one.
  Its maps apply the same per-lane kernels (`kernel.rs`). A value carries
  an optional container of failure ids: zero for a lane that did not fail,
  and otherwise an index into the walk's table of `(LaneFailure, node)`.
  A fallible map first computes the lanes, noting whether any failed, and
  only then, in a second pass, the ids, so a walk with no failure pays
  nothing for them. The walk borrows the tree, keeps its pending nodes on
  the heap, and remembers the value of every node the tree shares
  (`Tree::is_shared`) by identity, behind an `Rc`.
- **The failure rules** (D-S9-6): a node passes its operands' failures on,
  the first operand's first; a connective computes, per lane, whether an
  operand that did not fail there holds its absorbing value, and drops
  the failure where one does; a piecewise folds the failure ids through
  the same selection as the values, and a failed condition fails its lane
  unless an earlier case's condition holds there.
- **Error variants beyond the sketch.** `EvaluationError::NumberAsBoolean`
  is the walk's own check of a connective operand, a negation or a
  piecewise condition, which the screen normally refuses first;
  `IntegerLiteralOutOfRange` is named `IntegerOutOfRange`, since a user
  constant's value can be out of range too; `Kernel` is the failure of a
  plugged-in array kernel (S9.3). `FoldError::Piecewise` is new: a native
  call with literal arguments used as a piecewise condition folds to a
  literal, and a non-Boolean one breaks the piecewise, which
  `rebuild_with_children` refuses, as `InlineError::Piecewise` does.
  `LaneFailure` is payload-free; the failing node gives the sort.
- **`NoNativeCalls`** is a `NativeCalls` with no implementation, for Rust
  users and tests that fold only built-ins.
- **The fold refuses eagerly.** A literal call the fold cannot compute,
  such as `round(nan)`, is refused wherever it sits, as Python's fold did,
  while an evaluation discards its lane when a guard does not need it. The
  property `folding_does_not_change_the_value` therefore compares only
  trees the fold accepts, and compares outcomes by kind, since folding
  changes the node a lane failure names.
- **`BuiltinFunction::native_value`** is the kernel of D-S9-7, a method of
  the catalogue; its rustdoc says the last bits follow the platform's math
  library, except `erf` (`libm`) and `round` (`round_ties_even`).
- **`Decimal::to_f64_exact`** parses the decimal's text to the nearest
  `f64` (Rust's parsing rounds correctly) and compares it with the decimal
  as integers: the float's mantissa and binary exponent against the
  coefficient and the decimal exponent, scaling whichever side has a
  negative exponent.
- **Tests.** `fold_stories.rs` (35 tests, counting `rstest` cases),
  `evaluate_stories.rs` (69), `evaluate_properties.rs` (2 properties; the
  lane property joins in S9.3); `builtins_stories.rs` gained 26 and
  `literal_stories.rs` 11. A proptest regression file written while the
  stubs failed was deleted, as in S7.2.

Traceability of the Python tests (`test_evaluator.py` is `E`,
`test_evaluator_properties.py` `EP`, `test_numpy_evaluator.py` `N`; the
array stories are S9.3's):

| Python tests | Rust tests | Note |
|---|---|---|
| E `test_evaluate_folds_native_call_with_single_literal_argument`, `..._multiple_literal_arguments`, `..._returning_int_to_int_literal` | `fold_calls_a_user_native_with_its_literal_arguments`, `fold_makes_an_integer_sorted_builtin_an_integer`, `fold_computes_a_real_builtin` | built-ins now computed by the core (D-S9-7) |
| E `test_evaluate_extracts_int_value_from_int_literal_argument` | `fold_converts_an_integer_argument_to_the_nearest_real` | |
| E `test_evaluate_leaves_native_call_with_identifier_argument_alone`, `..._mixed_arguments_alone` | `fold_keeps_a_native_call_with_a_non_literal_argument_unchecked` | |
| E `test_evaluate_leaves_expression_bodied_call_alone_even_with_literal_args`, `test_evaluate_preserves_call_to_registered_function_unchanged` | `fold_keeps_calls_of_functions_with_a_body_and_lists_them_once` | the WARNING is the binding's (D-S9-15) |
| E `test_evaluate_substitutes_canonical_constant_identifier_with_its_value`, `..._leaves_identifier_alone_when_name_not_a_constant`, `..._leaves_an_identifier_merely_named_like_a_constant_alone`, `..._folds_native_call_with_constant_argument` | `fold_resolves_builtin_and_user_constants`, `fold_leaves_an_identifier_merely_named_like_a_constant` | |
| E `test_evaluate_folds_nested_native_calls_bottom_up` | `fold_folds_nested_native_calls_inside_out` | |
| E `test_evaluate_does_not_fold_binary_addition_of_literals`, `..._unary_negation_of_literal`, `..._literal_only_piecewise`, `test_evaluate_returns_literal_unchanged`, `..._identifier_unchanged_when_not_a_constant` | `fold_does_not_fold_arithmetic_or_a_literal_piecewise`, `fold_keeps_a_native_call_with_a_non_literal_argument_unchecked` | the input itself, by `ptr_eq` |
| E `test_evaluate_recurses_into_binary_expression_children`, `..._piecewise_branches`, `test_evaluate_does_not_mutate_input_expression` | `fold_recurses_into_children_and_shares_what_it_keeps` | |
| E `test_evaluate_wraps_native_value_error_in_pass_execution_error`, `..._zero_division_error_...` | `fold_keeps_a_user_native_failure_as_its_source` | the wrapping is the binding's |
| E `test_evaluate_rejects_string_form_float_literal_argument`, `..._coerces_decimal_literal_argument_like_its_text`, `..._coerces_string_form_float_literal_with_exact_binary_value`, `..._folds_native_call_with_exact_binary_float_string_argument`, `..._coerces_string_form_integer_literal_to_int` | `fold_converts_an_exact_decimal_argument_and_refuses_an_inexact_one`, `fold_hands_a_user_native_its_decimal_arguments_as_floats`, `decimal_to_f64_exact_*` (4) | |
| E `test_evaluate_raises_for_unregistered_call_name`, `..._for_call_to_native_constant` | `fold_refuses_an_unknown_name_and_a_constant_called` | |
| E `test_evaluate_raises_native_result_sort_error_for_wrong_return_type` | `fold_refuses_a_user_native_result_outside_its_result_sort` | |
| none | `fold_checks_the_arity_of_a_folded_call`, `fold_checks_the_argument_sorts_of_a_folded_call`, `fold_checks_the_arguments_before_the_call_taking_them`, `fold_refuses_a_non_finite_integer_sorted_result`, `fold_follows_ieee_where_math_would_raise`, `fold_makes_a_huge_integer_sorted_result_its_exact_integer`, `fold_calls_a_shared_user_native_once`, `fold_folds_a_shared_dag_once_per_distinct_node`, `fold_walks_a_deep_tree_on_a_small_stack` | new: Z-7, Z-8, Z-13 |
| EP `test_evaluate_expression_preserves_evaluation`, `..._is_idempotent`, `..._folds_every_literal_argument_call` | `folding_does_not_change_the_value` | idempotence stays a Python property |
| N the arithmetic, comparison, logical, unary and literal cases | `evaluate_computes_integer_arithmetic_exactly` (14 cases), `evaluate_divides_integers_to_a_real`, `evaluate_floor_divides_reals_as_python_does` (5), `evaluate_promotes_an_integer_beside_a_real`, `evaluate_compares_integers` (6), `evaluate_reduces_connectives_in_order`, `evaluate_reads_literals_in_their_domains` | D-S9-5 |
| N `test_integer_base_to_negative_integer_power_raises_value_error` | `evaluate_fails_the_lane_of_an_integer_error` (8 cases) | overflow and division by zero are new (Z-2) |
| N the dtype-screen tests (`..._raises_directly`) | `evaluate_screens_with_the_value_kinds_of_the_bindings`, `evaluate_screens_before_refusing_a_bound_constant` | |
| N the constant and binding tests | `evaluate_resolves_builtin_and_user_constants`, `evaluate_refuses_a_bound_constant_it_refers_to_and_ignores_one_it_does_not`, `evaluate_reads_bindings_and_ignores_unreferenced_ones` | |
| N `test_raises_for_unbound_variable`, `..._unbound_identifier_merely_named_like_a_native_constant`, `..._unbound_identifier_matching_a_native_function_name` | `evaluate_names_the_near_miss_of_an_unbound_identifier` | |
| N the float-grammar string literal tests | `evaluate_refuses_an_inexact_decimal_and_an_out_of_range_integer` | |
| N the non-finite cast, guarded and nested piecewise tests | `a_piecewise_guards_a_non_finite_cast`, `a_piecewise_raises_the_failure_of_the_branch_it_selects`, `a_nested_piecewise_is_guarded_by_its_outer_condition`, `a_piecewise_discards_the_failure_of_a_branch_it_does_not_select` | extended to every lane failure (Z-5) |
| N the piecewise selection tests | `evaluate_takes_the_first_piecewise_case_that_holds`, `evaluate_mixes_integer_and_real_branches_as_reals` | |
| N `test_sqrt_of_negative_returns_nan_without_raising`, `test_native_function_matches_hand_computed_value`, `test_integer_sort_native_returns_integer_dtype` | `evaluate_computes_native_builtins_and_casts_integer_results`, `native_builtin_computes_its_function` (19), `native_builtins_follow_ieee_outside_their_domains` | |
| N `test_raises_for_unsupported_erf`, `test_raises_for_gelu_due_to_unsupported_erf` | `evaluate_inlines_composed_builtins_and_user_functions` (gelu) | Z-6: computed now |
| N `test_raises_for_native_function_without_numpy_mapping`, `test_raises_for_unregistered_function_name`, `test_raises_for_recursive_function` | `evaluate_refuses_a_native_user_function`, `evaluate_reports_an_inlining_error` | |
| N the auto-inline tests | `evaluate_inlines_composed_builtins_and_user_functions`, `composed_builtins_evaluate_as_their_definitions` | |
| none | `evaluate_refuses_a_boolean_used_as_a_number` (3), `evaluate_refuses_a_boolean_under_negation_and_as_a_native_argument`, `evaluate_refuses_a_piecewise_mixing_booleans_and_numbers`, `a_conjunction_discards_*`, `a_disjunction_discards_*`, `a_failed_piecewise_condition_*`, `a_failure_passes_through_every_other_node`, `evaluate_negation_of_the_smallest_integer_overflows`, `evaluate_compares_nan_as_ieee_does`, `prepare_exposes_*`, `evaluate_walks_a_deep_tree_on_a_small_stack` | new: Z-3, D-S9-6, depth |

### S9.3 implementation notes

The `ndarray` feature is `rust/fhy-core/src/expression/evaluate/array.rs`,
with `ndarray = { version = "0.17", default-features = false, features =
["std"] }` (MIT or Apache-2.0, `rust-version` 1.64) as an optional
dependency and `[features] ndarray = ["dep:ndarray"]`. CI's `rust` job
already builds and tests `--all-features`, `deny` checks the all-features
graph, and `cargo +1.85 check -p fhy-core --all-features` passes, so the
workflow needs no change. The crate README documents the feature.

- **The API.** `ArrayBinding<'a>` holds a view (`ArrayViewD`) of
  Booleans, `i64`s or `f64`s, of any strides; `ArrayValue` an owned
  result. `Prepared::evaluate_array(environment, kernels)` takes the
  `ArrayKernels` N-S9-2 (b) plugs NumPy in through: `handles(function)`
  and `native(function, argument: CowArray<f64, IxDyn>)`, which receives
  a borrowed view for a bound identifier and an owned temporary
  otherwise, so an implementation can move a temporary out without a
  copy. `CoreKernels` handles nothing. A kernel's error, or a result of
  another shape, is `EvaluationError::Kernel`.
- **The backend** implements `Lanes` with `Of<T> = CowArray<'a, T,
  IxDyn>`: the bindings stay borrowed, and every computed value is owned.
  The maps broadcast by NumPy's rules and apply the kernel with
  `ndarray::Zip::map_collect`, which keeps the inputs' memory order; a
  constant is a 0-d array. The result is an owned array in the standard
  layout, copied only when it is a binding or in another order.
- **Tests.** `evaluate_array_stories.rs` (15 tests) under
  `cfg(feature = "ndarray")`, and the property
  `each_lane_of_an_array_evaluation_is_the_scalar_evaluation_of_that_lane`
  in `evaluate_properties.rs`: random trees over an integer, a real and a
  Boolean identifier, with every arithmetic operation, comparisons,
  connectives, piecewise and six natives, over up to eight lanes of
  values including zeros, NaNs, infinities and 64-bit extremes. Each lane
  of the array evaluation equals that lane's scalar evaluation, and an
  array evaluation's lane failure is the first failing lane's.

### S9.4 implementation notes

The binding is `rust/fhy-core-py/src/expression/evaluate.rs`, with
`evaluate/fold.rs` (`fold_expression` and the `NativeCalls` adapter),
`evaluate/numpy.rs` (`evaluate_expression_with_numpy` and the NumPy
kernels), `evaluate/builtins.rs` (`BuiltinNativeImplementation`),
`evaluate/literal.rs` (`coerce_literal_value` and
`is_decimal_text_exactly_binary`) and `evaluate/error.rs` (D-S9-14's
mapping), exported from `_rs` and declared in `_rs.pyi`. The binding
depends on `numpy` 0.29 (rust-numpy, whose `ndarray` Cargo unifies with
the core's 0.17), `num-traits`, and `fhy-core` with the `ndarray`
feature; `deny.toml`'s BSD-2-Clause allowance is now used. Nothing in
Python uses the new functions yet, so the suite is unchanged (7,327
passed).

- **`fold_expression(expression)`** returns the folded expression,
  materialized beside the input as the inliner's is, and the names of the
  functions with a body whose calls it kept. The adapter reads each native
  user function's Python `implementation` from the registry snapshot the
  fold reads (a new `RegistryState::native_implementation`), passes it
  `bool`, `int` and `float` arguments, and returns its exception unchanged
  as the callback error; a result that is no `bool`, `int` or `float` is
  refused with `NativeResultSortError` by the adapter itself.
- **`BuiltinNativeImplementation`** is a frozen pyclass with one object per
  native built-in, `__call__(value, /)`, `__name__`, a `repr`, and
  `__reduce__` through the class method `_of(name)`, so it unpickles as
  itself. `NativeFunction._install_builtins` takes its table as an
  optional argument for the one step until S9.5; without it the built-in
  entries hold these objects.
- **The NumPy entry point** imports NumPy first, on every call, and raises
  the guiding `ImportError` with the failure as `__cause__`: rust-numpy
  panics if it touches the C API without NumPy, so nothing else may run
  before. It inlines through `Evaluator::prepare`, converts only the
  bindings of the inlined tree's free identifiers (by id, so no key is
  restored in vain), and takes the scalar backend when every binding is a
  scalar, a 0-d array included. A `uint64` array above `i64::MAX`, and a
  Python `int` outside `i64`, raise `OverflowError`; `float128` and other
  dtypes wider than 64 bits are refused with the other unsupported dtypes.
- **The NumPy kernels** (N-S9-2 (b)) call the ufunc named like the
  built-in (the 14 names are NumPy's) inside `numpy.errstate(all="ignore")`,
  with the interpreter reattached for the call. A real binding NumPy
  already holds is passed as its own array; other lanes are copied into a
  NumPy array the ufunc overwrites (`out=`), and an owned temporary then
  takes the result back in place.

**Performance work the benchmarks called for, in the core** (the design
left the array backend's memory use open):

- **Operand storage is reused.** An operand value no one else holds is
  consumed by its parent, and real arithmetic, real negation and the
  native kernels compute into its storage (`Lanes::map2_reusing`,
  `map1_reusing`), as NumPy elides temporaries. The walk counts, before it
  starts, how often the tree reaches each node, and keeps a shared node's
  value only until its last use, since `Tree::is_shared` also counts the
  handles Python objects hold, which made every node look shared.
- **Arrays of more than 65,536 lanes are evaluated in chunks** of that
  many lanes: the bindings the expression refers to are broadcast
  together once, a binding in the standard layout of the result's shape is
  sliced per chunk without a copy, any other one is copied once into that
  layout, and a one-lane binding stays a 0-d array. Each chunk is one
  walk, and the chunks' lanes are appended to one output buffer. The lanes
  are computed as before, and a lane failure is still the first in C
  order, since the chunks go in order. The reason is the allocator: on this
  machine (glibc 2.31, Linux 5.4) every fresh 8 MB buffer costs about
  1,940 page faults, 3 to 4 ms, while NumPy's buffers use transparent huge
  pages (`madvise`, which the crate's `unsafe_code = "forbid"` rules out).
  With whole-array temporaries the polynomial over 10^6 floats took 15 ms
  with 3,874 faults per call; chunked, its temporaries stay small and are
  reused, and it takes 2.4 ms with no fault, where NumPy by hand takes
  3.0 ms.
- Four Rust tests cover chunks: a piecewise over 200,003 lanes, bindings
  broadcast and transposed across chunks, the first failed lane in a
  later chunk, and a kernel handed each chunk. `evaluate_array_stories.rs`
  has 19 tests now.

### S9.5 status

The Python switch (marked breaking) makes `passes/numpy.py`,
`passes/evaluate.py` and `passes/native_lowering.py` thin layers over
`_rs`, drops `builtins.py`'s table of `math` callables (the built-in
entries hold `BuiltinNativeImplementation`s, and
`NativeFunction._install_builtins()` takes no argument any more), makes
`ExpressionPrettyFormatter` a `CompilerPass[Expression, str]` over the
core (N-S9-1), and rewrites two docstrings of `errors.py`. It leaves the
tests of S9.6's migration failing: three modules fail collection, since
they import the private tables (`test_numpy_evaluator.py`,
`test_builtins.py`, `test_registry_rust_binding.py`), and the two
`test_pprint.py` subclass tests fail (6,841 passed). Every other test,
the property oracles included, passes unchanged. `pformat_expression`
raises `TypeError` for an argument that is not an `Expression`, where it
used to run the Python formatter over it.

### S9.6 status

The tests are migrated, and the interface suite is new. At the end:
`pytest` 7,435 passed, `-m "not very_slow"` 7,468 passed, the `property`
session (`HYPOTHESIS_PROFILE=thorough -m property`) 281 passed, `ruff` and
`mypy` over `src tests benchmarks rust/fhy-core/tests/golden` clean,
`tests/test_rs_stub.py` green, and the Rust gate green (fmt, clippy `-D
warnings` with `--all-features`, 2,481 tests of `tests/it`, doc `-D
warnings`, deny, `cargo +1.85 check`). No test was skipped, or deleted
without a rewrite; every other test, the property oracles that compare
with the NumPy evaluator included, passes unchanged. None of their integer
trees overflows `int64`, so Z-2 changed nothing there.

Tests migrated in S9.6:

| Test | Now | Reason |
|---|---|---|
| `test_numpy_evaluator.py`: the 15 tests matching `PassExecutionError`'s `__cause__` (non-finite casts, guarded piecewise, inexact decimals, unbound identifiers, unknown and recursive functions, a constant called, a native user function, the integer power) | same names, `pytest.raises(<the cause's class>)` | D-S9-10: raised directly |
| `test_numpy_evaluator.py`: the 11 `np.errstate(all="ignore")` blocks | `_refuse_warnings()`, which turns every warning into an error | Z-9: the evaluator warns nothing |
| `test_numpy_evaluator.py::test_unselected_case_domain_error_still_warns_and_is_discarded` | `test_unselected_case_domain_error_is_discarded_without_a_warning` | Z-9 |
| `test_numpy_evaluator.py::test_real_sort_native_preserves_float_width` | `test_real_sort_native_widens_a_float32_binding_to_float64` | Z-1 |
| `test_numpy_evaluator.py::test_cast_to_result_sort_casts_to_declared_dtype` (3), `..._passes_real_through_unchanged`, `..._raises_for_non_finite_value` (3) | `test_native_results_take_the_dtype_of_their_result_sort` | the private cast helper is gone; the casts are Rust stories (`evaluate_computes_native_builtins_and_casts_integer_results`), and Python reaches only the `INT` and `REAL` ones |
| `test_numpy_evaluator.py::test_every_binary_operation_has_a_lowering`, `test_every_unary_operation_has_a_lowering`, `test_every_lowered_native_name_resolves_to_a_numpy_callable`, `test_every_builtin_native_function_has_a_lowering` | `test_every_binary_operation_is_evaluated` (13 cases), `test_every_unary_operation_is_evaluated`, `test_every_builtin_native_function_is_evaluated` | the ufunc tables are gone; each operation and native is compared with NumPy or `math` |
| `test_numpy_evaluator.py::test_logical_not_of_an_object_dtype_array_raises_as_a_runtime_backstop`, `test_piecewise_with_object_dtype_condition_array_raises_during_evaluation` | `..._is_refused_when_converted` (both) | Z-11: `TypeError` when converted |
| `test_numpy_evaluator.py::test_raises_for_unsupported_erf`, `test_raises_for_gelu_due_to_unsupported_erf` | `test_evaluates_erf_as_math_does`, `test_evaluates_gelu_through_erf` | Z-6 |
| `test_numpy_evaluator.py::test_raises_for_unbound_identifier_merely_named_like_a_native_constant`, `..._matching_a_native_function_name` | same names | D-S9-14: the core's texts ("shares its name with the constant", "names a function") |
| `test_numpy_evaluator.py::test_numpy_expression_evaluator_snapshots_environment_at_construction` | same name | D-S9-10: no `numpy_module` argument |
| `test_builtins.py::test_seeded_native_implementation_matches_math_callable` (17) | `test_seeded_native_implementation_agrees_with_math_inside_its_domain` (17), `test_seeded_native_implementation_follows_ieee_outside_its_domain` | D-S9-9: the implementation is the core's kernel, one `BuiltinNativeImplementation` per built-in |
| `test_builtins.py::test_builtin_native_implementations_table_item_assignment_raises_type_error` | `test_builtin_native_entry_implementation_cannot_be_replaced` | D-S9-9: the table is gone; the entry is frozen |
| `test_registry_rust_binding.py::test_builtin_function_entry_is_one_object_with_its_body_built_once` | same name | D-S9-9: a native built-in's implementation is its `BuiltinNativeImplementation` |
| `test_pprint.py::test_pretty_formatter_call_rejects_non_string_formatted_result` | `test_pretty_formatter_refuses_a_subclass_defining_a_visitor` | N-S9-1 |
| `test_pprint.py::test_pretty_formatter_subclass_overrides_one_node_kind` | `test_pretty_formatter_subclass_without_visitors_formats_as_the_core`, `test_pformat_expression_refuses_a_value_that_is_no_expression` | N-S9-1 |

The new `tests/symbolic/expression/passes/test_evaluate_rust_binding.py`
(101 tests, counting parametrized cases) covers the test plan: NumPy
optional (three subprocess tests: `import fhy_core` and a fold without
NumPy, NumPy left unimported, the guiding `ImportError` and its cause);
every admitted and refused dtype, `uint64` at the edge, a Python `int`
beyond `int64`, swapped byte order, Fortran order, strides, negative and
zero strides, Python scalars, nested lists, an unreadable unreferenced
binding, empty and 0-d bindings; 0-d results as NumPy scalars whatever
the root, and array results that are new, writeable and C-contiguous; the
12 error rows raised directly and wrapped by a pass run; the check order;
both passes registered, the NumPy pass's snapshot and inlining, the fold's
warnings and identity; the arity check of Z-8; native user functions
receiving Python values, their exception as the cause itself, a
`KeyboardInterrupt` passing through, a non-numeric result refused; the
built-ins' implementations agreeing with the fold, one object each,
pickling as themselves; the literal helpers; another thread running
during a large evaluation, and concurrent evaluations agreeing.

**Without NumPy** (a scratch venv in `target/` with the test group but
no NumPy): `-m "not very_slow"` has 6,676 passed, 38 skipped (the modules
that import NumPy through `importorskip`) and 371 failed. Every failure is
the guiding `ImportError`: 368 in `test_sympy_pass.py` and 3 in
`test_solver_properties.py`, which evaluate with NumPy as an oracle
without importing it. They failed the same way before S9, since the
Python evaluator raised the same `ImportError`; S9.7 marks them.

### S9.8 performance work

The first "after" run (before this commit) measured the lone transcendentals
and the Boolean row slower than NumPy's own ufuncs: `exp-1e6` 2.40, `tanh-1e6`
1.86, `logical-1e6` 1.30 times the baseline. Three changes, with tests:

- **A lone transcendental over a real array binding** in C order is
  NumPy's ufunc over the binding itself, in the binding
  (`evaluate_lone_kernel_call`): N-S9-2 (b) makes NumPy's ufunc that node's
  kernel anyway, and the core's chunked path copies the ufunc's output
  twice (into the chunk, then into the result), which safe code cannot
  avoid, since NumPy cannot write into Rust memory without `unsafe`. The
  bound-constant refusal still applies: a binding of a constant's
  identifier takes the ordinary path, which refuses it.
- **A chunk of a binding is passed to a ufunc as a NumPy view** of the
  binding (a `reshape(-1)` slice), not as a copy.
- **Maps against a one-lane operand** (a literal or a scalar binding) take
  `ndarray`'s contiguous `map` and `mapv_inplace`, not a broadcast `Zip`;
  connectives no longer copy their first operand, and `!` reuses its
  operand's storage.

Tests: `test_a_transcendental_native_is_numpys_ufunc` (9 cases: a lone
call, a Fortran-ordered binding, 200,003 lanes in chunks, and the same
call inside a compound tree, each equal to the ufunc, C-contiguous, new,
and without a warning under `np.errstate(all="raise")`) and
`test_a_lone_native_over_a_bound_constant_is_still_refused`.

### S9.7 status

The branch was rebased onto `dev-rust` at c93f76f, which holds S8 and S10.
The design doc's conflicts were resolved by keeping `dev-rust`'s text and
appending S9's checklist entry and section after S10's; `lib.rs`'s module
table, the crate README's feature sections, `fhy-core`'s `[features]`
table (`z3` and `ndarray` in one table) and the binding's
`expression.rs` (S10 moved `alpha` out, S8 added the materializer
re-exports) were merged by hand, keeping both sides; `Cargo.lock` was
regenerated by Cargo, not merged. Every rebased commit builds.

Then, following S8's scheme (D-S8-17):

- **The `numpy` marker** (`pyproject.toml`), with `tests/conftest.py`'s
  skip for it beside `z3` and `sympy`. It marks the 27 test functions that
  evaluate with NumPy as an oracle without importing it (24 in
  `test_sympy_pass.py`, 3 in `test_solver_properties.py`, found by running
  the suite without NumPy; every case of each function reaches NumPy), and
  the nine modules that import NumPy at module level through
  `pytest.importorskip`.
- **`tests_minimal` leaves NumPy out too**: the `test-minimal` dependency
  group no longer holds `numpy`, the `test` group adds it, and the session
  checks that neither `sympy`, `z3` nor `numpy` is installed. It passes
  (5,646 passed, 626 skipped). CI's job for it needs no change.
- **The README** documents the `numpy` extra's `ImportError` and that
  nothing else imports NumPy, and its expression row describes the Rust
  fold, the NumPy evaluator and the formatter pass.

### S9 status

S9 was implemented on 2026-09-26 in nine commits after the design and
the resolutions: the benchmarks and their baseline; the core evaluator,
test-first; the `ndarray` backend behind its feature; the binding; the
Python switch, marked breaking; the migrated tests and the interface
suite; the performance work the benchmarks called for; the `numpy`
marker and `tests_minimal` without NumPy, after the rebase onto S8 and
S10; and these docs, with this correction. No test was skipped or
deleted without a rewrite. At the end: `pytest tests` 7,556 passed, `-m
"not very_slow"` 7,589 passed, the `property` session 282 passed, nox
`tests_minimal` (5,646 passed, 626 skipped), `lint` and `type_check`
clean, `tests/test_rs_stub.py` green, and the Rust gate green: fmt, clippy
`-D warnings` with and without `--all-features`, 3,271 tests with the
default features and 3,303 with all of them (`z3` built against the
z3-solver wheel's libz3 4.16, as S8.3 records), doc `-D warnings` both
ways, `cargo deny check` (BSD-2-Clause now used by rust-numpy), and `cargo
+1.85 check`, with and without `ndarray`.

### S9 benchmarks (before and after)

Median time per call, from each tree's own environment, `pytest
benchmarks/test_evaluate.py <the rerun rows> -n 0 --benchmark-only`, on
the S0 machine with Python 3.11.13, NumPy 2.4.6 and pytest-benchmark
5.3.0. "Before" is 6a84ae3, the rebased S9.1 tree (S8 and S10 with the
benchmark file), exported with `git archive` under `target/` and built
there; "after" is the S9.7 tree. The two ran three times each,
interleaved, with a load average of 3 to 10 from other work on the
machine, and the table lists the best of the three medians. The
"before" column agrees with the S9.1 table within 10%, except the two
single-call fold rows (10.3 and 7.4 µs in S9.1, 6.7 and 6.5 µs here), a
difference of the load and of the rebased pass framework.

| Benchmark | before | after | after / before |
|---|--:|--:|--:|
| `test_evaluate_expression_of_a_builtin_native_call` | 6.7 µs | 3.4 µs | 0.51 |
| `test_evaluate_expression_of_a_user_native_call` | 6.5 µs | 3.8 µs | 0.59 |
| `test_evaluate_expression_of_the_deep_tree` | 203 µs | 21.1 µs | 0.10 |
| `test_evaluate_expression_of_nested_native_calls` | 37.0 µs | 5.3 µs | 0.14 |
| `test_evaluate_expression_of_constant_references` | 384 µs | 70.7 µs | 0.18 |
| `test_evaluate_with_numpy_of_scalars[poly]` | 29.2 µs | 3.5 µs | 0.12 |
| `test_evaluate_with_numpy_of_scalars[sigmoid]` | 55.2 µs | 4.3 µs | 0.08 |
| `test_evaluate_with_numpy_of_scalars[deep_tree]` | 471 µs | 44.9 µs | 0.10 |
| `test_evaluate_with_numpy_of_arrays[poly-1e3]` | 35.1 µs | 7.0 µs | 0.20 |
| `test_evaluate_with_numpy_of_arrays[poly-1e6]` | 6.89 ms | 1.96 ms | 0.28 |
| `test_evaluate_with_numpy_of_arrays[exp-1e6]` | 961 µs | 888 µs | 0.92 |
| `test_evaluate_with_numpy_of_arrays[tanh-1e6]` | 1.84 ms | 1.82 ms | 0.99 |
| `test_evaluate_with_numpy_of_arrays[sigmoid-1e6]` | 5.92 ms | 3.27 ms | 0.55 |
| `test_evaluate_with_numpy_of_arrays[piecewise-1e6]` | 12.72 ms | 2.68 ms | 0.21 |
| `test_evaluate_with_numpy_of_arrays[guarded-1e6]` | 20.71 ms | 20.01 ms | 0.97 |
| `test_evaluate_with_numpy_of_arrays[integer-1e6]` | 20.14 ms | 17.50 ms | 0.87 |
| `test_evaluate_with_numpy_of_arrays[logical-1e6]` | 2.00 ms | 1.90 ms | 0.95 |
| `test_evaluate_with_numpy_of_arrays[deep_tree-1e4]` | 741 µs | 309 µs | 0.42 |
| `test_evaluate_with_numpy_of_float32_arrays` | 1.62 ms | 2.63 ms | 1.62 |
| `test_evaluate_with_numpy_with_unused_bindings` | 39.6 µs | 15.5 µs | 0.39 |
| `test_pretty_formatter_of_the_deep_tree` | 234 µs | 10.2 µs | 0.04 |
| `test_evaluate_after_inline` | 60.7 µs | 25.8 µs | 0.43 |
| `test_mixed_pipeline_over_a_deep_expression` | 276 µs | 45.0 µs | 0.16 |
| `test_pformat_expression_of_deep_tree[symbolic]` | 8.1 µs | 8.3 µs | 1.03 |
| `test_pformat_expression_of_deep_tree[functional]` | 7.4 µs | 7.7 µs | 1.03 |
| `test_pformat_expression_of_deep_tree[show_id]` | 9.8 µs | 9.8 µs | 1.00 |

Every row is faster or within the 10% CONTRIBUTING allows, except one:

- **`float32` arrays, 1.62 times as long: an accepted cost, which the
  maintainer accepted on 2026-09-26** (cross-cutting rule 5). The evaluator computes reals in
  `float64` (D-S9-4, Z-1), so a `float32` binding is widened by one NumPy
  cast and the arithmetic moves twice the bytes; NumPy computes in
  `float32`. Converting chunk by chunk inside the core would save the cast
  (an estimated 1.3 times), but only a `float32` domain would reach
  parity, which D-S9-4 left for later.
- **Scalars and the fold** are 2 to 12 times faster: one call into the
  extension instead of a pass lifecycle and a Python visitor call per
  node.
- **Arithmetic over arrays** is 3.5 times faster for the million-element
  polynomial and 4.7 times for a piecewise: the chunked, storage-reusing
  walk (see "S9.4 implementation notes") keeps temporaries in cache,
  where NumPy writes each one out.
- **The transcendentals** run NumPy's ufuncs (N-S9-2 (b)): a lone `exp`
  or `tanh` over a binding is NumPy's call itself (0.92 and 0.99), and
  `sigmoid` is 1.8 times faster.
- **Integer arithmetic** (0.87) and the guarded non-finite casts (0.97),
  whose element failures take the walk's second pass, stay near NumPy's
  cost; the connectives are at 0.95.
- **The formatter** takes 10 µs, the core's rendering, where the Python
  visitor took 234 µs (N-S9-1).

### S9 implementation notes

Choices the decisions left open, made while implementing S9.5 to S9.8
(S9.2 to S9.4 have their own notes above):

- **The chunked array evaluation and operand reuse** (S9.4 notes) are the
  core's answer to the allocator costs this machine showed; the lanes
  and their failures are exactly those of the whole-array walk, which the
  lane property pins.
- **The lone-transcendental fast path** is in the binding, not the core:
  it is where NumPy's ufunc is the node's kernel, and it only skips
  copies, so the core's semantics are unchanged.
- **`pformat_expression` refuses a non-expression** with `TypeError`;
  it used to run the Python formatter over one.
- **`NumpyExpressionEvaluator(environment)`** lost its `numpy_module`
  argument, and it inlines, so a composed call evaluates in a pass run as
  it does through the function.
- **`BuiltinNativeImplementation` is only in `_rs`**: Python reaches it as
  a built-in entry's `implementation`; no public module re-exports it.
- **`coerce_literal_value`** refuses a value of another type with
  `TypeError`, and text outside the core's literal grammar (such as
  `"1e5"`) with `ValueError`, where Python passed other types through and
  read any `int()` or `Decimal()` text.
- **Without NumPy**, the 27 test functions S9.7 marks are the only
  unmarked ones that reached NumPy; they failed the same way before S9,
  since the Python evaluator raised the same `ImportError`.

## S12: the Rust SymPy simplifier backend

- **Status:** designed 2026-09-26 at c93f76f. D-S12-1 to D-S12-18 apply
  the policy the user already set, the precedent of S8 to S10, and the
  user's direction in "Plan after S7", item 2: a Rust `Simplifier` over
  SymPy through pyo3, behind an off-by-default `sympy` cargo feature of
  `fhy-core`, with pyo3 optional and without `auto-initialize`, which the
  Python binding then uses too, so one copy of the mapping is left. The
  user resolved N-S12-1 as (a); see "S12 resolutions".
- **Pattern:** the lowering to SymPy, the simplify call with its
  workarounds, and the lifting back move into the core, as
  `fhy_core::solver::SympySimplifier` under the feature. In Python it
  becomes a Rust implementation of the P3 `Simplifier` ABC, as
  `SmtLib2ProcessSolver` is of `SmtSolver`. `passes/sympy.py` keeps its
  public names as thin Python over it.
- **Scope.** `src/fhy_core/symbolic/expression/passes/sympy.py` and the
  sympy wiring of `symbolic/solver.py`. This supersedes "Plan after S7",
  item 3 ("the sympy adapter stays, but only for Python"), which described
  S8. The rest of the solver, and the z3 adapter, are unchanged.
- **Coordination.** S9 (the evaluators) and S11 (types) run in parallel,
  and this branch is rebased onto `dev-rust` after S9 lands. "Coordination
  with S9 and S11" below lists the shared files and the one S9 item S12
  uses.

### Survey: the Python API

**`passes/sympy.py` (1,715 lines)** imports `sympy` at module level
and raises `SolverBackendUnavailableError` without it (D-S8-16). It
exports four names and registers three passes:

| Name | Kind | Meaning |
|---|---|---|
| `ExpressionToSympyConverter` | `VisitablePass[Expression, Any]`, `fhy_core.symbolic.expression.to_sympy` | the lowering: one `visit_*` method per node kind, class tables of operators, and the static `format_identifier` (`<name_hint>_<id>`) |
| `SympyVariableSubstitutionPass(replacements)` | `CompilerPass`, `...substitute_sympy_variables` | a simultaneous, `xreplace`-like substitution of sympy symbols that keeps every Boolean position Boolean |
| `SymPyToExpressionConverter` | `CompilerPass`, `...from_sympy` | the lifting: `convert`, `convert_expr`, `convert_bool` and `convert_relational` over three first-match dispatch tables |
| `convert_expression_to_sympy_expression(expression)` | function | `validate_logical_operands`, then the converter pass |
| `substitute_sympy_expression_variables(sympy_expression, environment)` | function | refuses a bound native constant free in the expression (`NativeConstantBindingError`), lowers each value, runs the substitution pass |
| `convert_sympy_expression_to_expression(sympy_expression)` | function | the lifting pass |
| `SympySimplifier` | Python `Simplifier` (S8) | lowers, runs `_try_simplify_sympy_expression`, lifts; `name` is `"sympy"` |

`fhy_core.symbolic.expression` re-exports the three functions lazily
(PEP 562), and `solver.py` resolves `SolverBackend.SYMPY` to a cached
`SympySimplifier()` (`_resolve_adapter`). The default solver holds
`_DeferredSimplifier(SYMPY)`, a Python backend that resolves the adapter
on its first question and delegates to it, so `import fhy_core` imports no
sympy.

**The mapping.**

- **Lowering.** `bool` to `sympy.true`/`false`; `int` to `Integer`;
  `float` to `Float` (a non-finite one to `oo`, `-oo` or `nan`); a
  `Decimal` to the exact `Rational` (through `Fraction`).
  - An identifier becomes `Symbol("<name_hint>_<id>")`, and a native
    constant's canonical identifier its value: `pi`, `E`, `oo` and `nan`
    for the built-ins, and a user constant's `bool`, `int` or `float` value
    as a sympy number.
  - The operators are `operator.*`, except `FLOOR_DIVIDE` as `floor(x/y)`,
    `LOGICAL_NOT` as `sympy.Not`, and `And`/`Or` as the sympy constructors,
    never the bitwise `&`/`|`.
  - A piecewise becomes a `_ParityOpaquePiecewise` with a final `True`
    branch, built with `evaluate=False`.
  - The 19 native built-ins lower through `_NATIVE_FUNCTION_LOWER`
    (`round` through the `sympy.Function` `_SYMPY_ROUND`, whose `eval` hook
    folds only an integer argument).
  - Any other call is refused with a `TypeError`, whose text depends on
    what the registry says the name is: an expression-bodied function
    ("call `inline_functions` first"), a constant, a native function
    without a lowering, or unknown.
- **Lifting**, in order: the constants (`oo`, `nan`, `pi`, `E`; `-oo` as
  the negation of `inf`); the 15 native function classes; `Pow(x, 1/2)` as
  `sqrt`; then the dispatch.
  - `Add` and `Mul` fold to the right; `Mod` and `Pow` are binary.
  - `Integer` becomes an `int` literal and `Float` a `float` literal. A
    `Rational` becomes the decimal-string literal of its value when its
    decimal expansion ends and a binary `float` equals it (through
    `native_lowering.is_decimal_text_exactly_binary`), and `DIVIDE` of its
    numerator and denominator otherwise; a negative one is negated.
  - A `Symbol` becomes the identifier restored from its name. A name
    without `_` raises `RuntimeError`.
  - `Xor`, `Nor`, `Nand` and `ITE` lift through their definitions;
    `Implies` logs a WARNING and raises `NotImplementedError`.
  - `zoo` raises `ComplexInfinityLiftError`, and a piecewise without a
    final `True` raises `PartialPiecewiseError`.
- **The workarounds**, each pinned by tests:
  1. `_ParityOpaquePiecewise`, a `sympy.Piecewise` subclass. It hides
     parity, since SymPy 1.14's `Mul._eval_is_integer` misjudges a
     quotient of a known-even value. Its `eval` refuses to leave a partial
     piecewise, and its `_eval_simplify` rebuilds plain piecewise nodes
     as itself.
  2. A Boolean piecewise is expanded into connectives
     (`_convert_piecewise_to_sympy_boolean`) wherever SymPy needs a
     `Boolean`. This avoids both `ITE` routes, which drop branches.
  3. An `Eq`/`Ne` of a symbol and a relational has its symbol side
     negated and its relation inverted.
  4. A piecewise inside a piecewise condition is folded.
  5. A piecewise is folded out of a relational whose piecewise has a bare
     symbol in a Boolean position. The fold's `TypeError` becomes
     `_UnfoldableRelationalError`.
  6. A piecewise is folded out of `Mod`, `floor` and `ceiling`.
  7. While `sympy.simplify` runs, each comparison between Booleans is
     masked as a `Dummy`.
  8. During substitution, the Boolean positions of each rebuilt node are
     converted.
- **Best effort.** `PrecisionExhausted`, `_UnfoldableRelationalError`, and
  a simplified result holding a partial piecewise each keep the
  unsimplified form, with a DEBUG log. Every other failure propagates.
- **Errors, by phase.**
  - The screen's `NonBooleanLogicalOperandError` is raised unwrapped.
  - A failure while lowering, substituting or lifting runs inside a pass,
    so it arrives as `PassExecutionError` with the cause: a `TypeError`
    from SymPy, `ComplexInfinityLiftError` or `PartialPiecewiseError`.
  - A failure of `sympy.simplify` itself propagates raw.
  - `simplify_expression(..., backend=SYMPY)` sees the same, through the
    adapter.
- **Every walk is recursive Python.** The lowering is a visitor call per
  node, and the workarounds are `sympy.replace` calls with Python lambdas.

**Where the time goes.** This was one `timeit` probe of today's adapter,
under load, not a baseline:
- `simplify_expression` of `x < 10` with `x` bound took 99 µs.
- The adapter's own `simplify` took 42 µs of that:
  - lowering the ground comparison, 20 µs;
  - the workaround pipeline, 19 µs, wrapped around a cached
    `sympy.simplify` of 2.6 µs;
  - the rest was lifting.
- The remainder of the 99 µs was the facade, the substitution and the
  deferred adapter's extra hop through Python.
- Lowering the 100-operation deep tree of `test_expression.py` took
  916 µs, and lifting it back 137 µs.

### Survey: the Rust API

- **`Simplifier`** (`solver/backend.rs`, D-S8-3 with S8.2's note):
  `name()` and `simplify(&Expression, &SimplifyContext<'_>) ->
  Result<Expression, BackendError>`, `Send + Sync + Debug`.
  `SimplifyContext` is a `Copy` struct with private fields, so it can grow.
  Today it carries only `&dyn SortLookup`, which gives the sorts of native
  constants and call results, but not a constant's value or what kind of
  entry a name is.
- **`Solver::simplify(expression, environment, sorts: &dyn SortLookup)`**
  screens, refuses a bound native constant, substitutes, and builds the
  context from `sorts`. The binding calls it detached, with the registry
  snapshot of S7 as the `SortLookup`, and materializes the result beside
  the input (`materialize_substituted`) unless a Python simplifier
  returned the object.
- **What the lowering and lifting need exists:**
  - `LiteralValue` (`Bool`, `Int(BigInt)`, `Float(f64)`, `Decimal`, with
    `coefficient()` and `exponent()`);
  - `BuiltinConstant::of_identifier`, `identifier()` and `value()`, and
    `BuiltinFunction` with its 19 natives;
  - `FunctionRegistry::constant(&Identifier)`, which holds the value, and
    `entry(name)`, which gives the entry's kind;
  - `Identifier::try_restore(id, name_hint)`, and
    `BooleanScreen::check_logical_operands`.
- **From S9:**
  - `Decimal::to_f64_exact() -> Option<f64>`, which is
    `is_decimal_text_exactly_binary`'s rule computed without text;
  - `fhy_core::expression::evaluate`, a Rust oracle for the properties.
- **The binding's backends** (`fhy-core-py/src/solver/backends.rs`):
  - `SimplifierBase`, and the Python adapter `PythonSimplifier`, which
    attaches once per question;
  - `build_simplifier`, which wraps any `SimplifierBase` instance in that
    adapter;
  - `SmtLib2ProcessSolver`, the model of a native backend: an `extends =
    SmtSolverBase` pyclass holding the core value, which a `Solver` calls
    without Python.
- **The binding's error mapping** (`solver/error.rs`) downcasts a
  backend's `BackendError`. A `PyErr` is raised as itself, and anything
  else becomes `SolverBackendError`.
- **Nothing in the core touches Python.** CONTRIBUTING says
  `rust/fhy-core` "never depends on PyO3" and "the core crate never holds
  Python objects". The user's direction revises both for the feature
  (D-S12-2).

### Survey: pyo3 0.29 and embedding

Checked on crates.io and in the 0.29.2 sources on 2026-09-26:

- **Versions.** pyo3 0.29.2 (2026-08-05) is the latest release, and the
  lock file's. Its `rust-version` is 1.83, below the crate's MSRV of 1.85.
  Python 3.8 is the minimum, and 3.13t is dropped in favor of 3.14t.
  `pyo3-ffi` declares `links = "python"`, so a build graph holds exactly
  one pyo3-ffi, and so one pyo3 minor version.
- **Features.** `default = ["macros"]`, and `auto-initialize` is a
  separate opt-in. `extension-module` is deprecated. `pyo3-build-config`
  leaves libpython unlinked when `PYO3_BUILD_EXTENSION_MODULE` is set,
  which maturin 1.9.4 and later do for a wheel build. So a plain `cargo
  build` or `cargo test` links libpython, and a wheel does not. Its
  `num-bigint` feature targets num-bigint 0.4, while the crate uses 0.5.
- **The interpreter.**
  - `Python::attach` panics when the interpreter is not initialized and
    `auto-initialize` is off.
  - `Python::try_attach` returns `None` in that case, and also while the
    interpreter finalizes or during a GC traversal.
  - `Python::initialize()` is a safe function. It initializes with signal
    handling disabled, and does nothing if Python is already running.
  - `py.detach` releases the GIL.
  - `Py<T>` is `Send + Sync`. It is `Clone` only with the `py-clone`
    feature, which `clone_ref(py)` replaces.
  - Dropping a `Py<T>` while not attached queues the decref.
  - `PyOnceLock<T>` is a GIL-aware once-cell that can be a struct field.
  - `PyCFunction::new_closure` turns a `Fn + Send + 'static` Rust closure
    into a Python callable, and needs no `macros`.
  - `PyModule::from_code` runs source through `PyImport_ExecCodeModuleEx`,
    which puts the module in `sys.modules`.
- **Finding the interpreter at build time:** `PYO3_PYTHON`, then an active
  virtualenv, then `python`, then `python3` on `PATH`.
  - Dynamic embedding needs that Python's shared libpython at link time,
    and at run time on the loader's path.
  - SymPy must be importable by the embedded interpreter. An embedded
    interpreter does not read a virtualenv's `pyvenv.cfg`, so a venv's
    site-packages reach it only through `PYTHONPATH`.
- **Probes on this machine**, in `target/scratch`, with one binary that
  embeds Python, imports SymPy and simplifies `x + x - x`:
  - **The known issue, reproduced.** With the tooling pyenv's
    `python3.11` (deadsnakes 3.11.13, `Py_ENABLE_SHARED=1` but no
    `libpython3.11.so` installed) the link fails: `rust-lld: error: unable
    to find library -lpython3.11`.
  - **The system `python3.10`** ships `libpython3.10.so`. A venv of it with
    sympy 1.14, as `PYO3_PYTHON`, links. The binary runs with that venv's
    site-packages on `PYTHONPATH`, and fails with `ModuleNotFoundError`
    without it.
  - **A uv-managed CPython 3.11.16** ships `lib/libpython3.11.so`. It
    links, and runs with `LD_LIBRARY_PATH` set to that `lib`.
  - **`Python::try_attach` before `Python::initialize()`** returns `None`,
    without a panic.
  - **Feature unification**, in a two-member scratch workspace where `b`
    enables `a/extra`: `cargo test -p a` builds `a` without `extra`, and
    `cargo test --workspace` builds it with `extra`.

### Consumers and tests

| File | Lines | Functions | Collected | What changes |
|---|--:|--:|--:|---|
| `expression/passes/test_sympy_pass.py` | 4,659 | 141 | 627 | imports three private tables and `_ParityOpaquePiecewise`; 50 calls of `convert_expr`, `convert_bool` or `convert_relational`; one `visit_literal_expression`; four `monkeypatch.setattr(sympy, "simplify", ...)`, which the backend must keep reaching; one patch of `convert_expression_to_sympy_expression`, which it cannot; texts matched by `match=` |
| `expression/passes/test_sympy_pass_properties.py` | 304 | 6 | 6 | none expected |
| `expression/test_sympy_natives.py` | 426 | 21 | 59 | `test_lowered_round_node_lifts_after_a_pickle_round_trip` (Y-S12-1) |
| `expression/test_cross_cutting.py` | 193 | 6 | 20 | the registrations stay |
| `symbolic/test_solver.py` | | | | `test_simplify_expression_matches_direct_bridge_pipeline` compares with `SympySimplifier` |
| `symbolic/test_solver_rust_binding.py` | | | | the adapter identity, `name`, availability, the default solver's simplifier name and the lazy imports keep holding |
| constraint and param tests | | | 351 (S8 probe) | reach sympy through the default solver; unchanged |

`test_native_stories.py` and `test_piecewise_properties.py` import the
bridge inside their tests. Markers stay as S8.7 set them: a test marked
`sympy` still reaches SymPy.

**Benchmarks:** `benchmarks/test_solver.py` has four simplification
rows (the ground comparison, `x + x - x`, the equation constraint, the
nat param) and the import row. Nothing measures the bridge functions on
their own.

### Divergences visible from Python

| # | Python today | After S12 |
|---|---|---|
| Y-S12-1 | the `round` function and the parity-opaque piecewise class live in `passes/sympy.py`, so a pickle of a lowered expression holding either loads wherever `fhy_core` is installed | they live in the core's prelude module `_fhy_core_sympy` (D-S12-7). A pickle holding either loads in a process where a SymPy backend has loaded, which importing `passes.sympy` does. A pickle written before S12 names the old module path and no longer loads |
| Y-S12-2 | `ExpressionToSympyConverter` is a `VisitablePass` with `visit_*` methods and operator tables | a thin `CompilerPass` over the core; `format_identifier`, `get_noop_output` and the registration stay (D-S12-10) |
| Y-S12-3 | error texts of the Python bridge (`Unsupported node type: ...`, `Cannot lower an expression-bodied function call ...`) | the core's one-line lowercase texts, keeping the phrases the tests match; the classes and the `PassExecutionError` wrapping stay (D-S12-11) |
| Y-S12-4 | a real user constant holding a decimal would lower to `sympy.Float` of it | its exact `Rational`, as a decimal literal lowers (D-S4-1) |
| Y-S12-5 | a tree deeper than Python's recursion limit raises `RecursionError` in the bridge's own walks | the lowering, lifting and substitution walks keep their work on the heap; SymPy's own recursion is unchanged |
| Y-S12-6 | patching `passes.sympy.convert_expression_to_sympy_expression` changes what `SympySimplifier` lowers | the backend calls no function of the Python module; patching `sympy.simplify` still reaches it (D-S12-8) |
| Y-S12-7 | `SympySimplifier` is a Python class, which a user can subclass | a native class, frozen and not subclassable, registered with `Simplifier`, as `SmtLib2ProcessSolver` is |
| Y-S12-8 | the default solver's simplifier is `_DeferredSimplifier`, a Python backend | a native `SympySimplifier`, which imports SymPy on its first question |
| Y-S12-9 | the best-effort cases log at DEBUG on the bridge's logger | not logged: the core does not log (D-S8-14). The `Implies` WARNING stays, written by the binding |

Unchanged in meaning:
- every lowered form, simplification result and lifted form, the
  best-effort cases included;
- the symbol naming;
- the exception classes, and the phases that wrap them;
- the lazy imports;
- the markers.

### Pattern choice

- **The mapping goes to Rust.** The user's direction places the lowering,
  the simplify call with its workarounds, and the lifting in a Rust
  `Simplifier`, and wants one copy of the mapping. So the Python bridge
  keeps no walk of its own.
- **P3, Rust implementation.**
  - `_rs.SympySimplifier` is an `#[pyclass(extends = SimplifierBase,
    frozen)]` holding an `Arc` of the core backend. It is registered with
    the `Simplifier` ABC.
  - `build_simplifier` recognizes it and hands the `Solver` the core
    value, so a question runs from the facade into SymPy with no Python
    hop.
  - The granularity rule holds: no user Python code is called per node.
    The backend's own per-node calls into SymPy are its lowering, as the
    z3 crate's are libz3's.
- **Plain Python over `_rs`:** the four public names, the three
  registered passes, the import guard and the texts that name pip extras
  stay in `passes/sympy.py`.
- **One Python file in the core.** SymPy is extended by subclassing and by
  hook functions, which only Python code can define. That file is a small
  prelude the backend loads (D-S12-7).

### Decisions (proposed 2026-09-26)

Each names the policy it follows:

- D-S4-1: Rust semantics where the two differ.
- D-S4-2: Python names where the meaning is the same.
- "No fallback".
- "Tests rewritten, not skipped".
- The crate's conventions in `rust-workspace.md` Part I:
  - one public path per item, and the layering (§I.2);
  - `#[non_exhaustive]` errors with one-line lowercase `Display` (I.3
    rule 3), and the naming rules (I.3 rule 5);
  - every change allowed but classified (I.3 rule 6, since the crate is
    unpublished);
  - no global state beyond identity, and MSRV 1.85.
- The binding patterns, and cross-cutting rules 5 to 7.
- "The direction": the user's choice in "Plan after S7", item 2, and this
  slice's brief.

Where a decision follows an earlier slice's decision or note, it says so.

- **D-S12-1: one mapping, in the core; no fallback** ("no fallback";
  the direction; D-S8-1).
  - These are deleted from `passes/sympy.py`:
    - the tables `_NATIVE_FUNCTION_LOWER`,
      `_NATIVE_FUNCTION_LIFT_DISPATCH`, `_NATIVE_CONSTANT_LOWER` and
      `_NATIVE_CONSTANT_LIFT`;
    - `_SYMPY_ROUND` and `_ParityOpaquePiecewise`;
    - every helper of the lowering, lifting, substitution and workaround
      walks;
    - the visitor and dispatch bodies;
    - the Python `SympySimplifier` class.
  - `solver.py` loses `_DeferredSimplifier`.
  - `passes/sympy.py` stops importing `native_lowering`.
  - The Python API that stays is D-S12-10's.
- **D-S12-2: pyo3 in the core, behind the `sympy` feature** (the
  direction; crate conventions).
  - The workspace table becomes `pyo3 = { version = "0.29",
    default-features = false }`. The binding asks for `features =
    ["macros"]`. The core gains `pyo3 = { workspace = true, optional =
    true }` and `[features] sympy = ["dep:pyo3"]`.
  - One entry in the workspace table keeps both crates on one version.
    `links = "python"` would refuse two versions anyway.
  - The core needs no pyo3 feature. `intern!`, `PyOnceLock` and
    `PyCFunction::new_closure` need no `macros`. pyo3's `num-bigint`
    feature is left off, since it targets num-bigint 0.4; the backend
    converts a `BigInt` through `i64` or decimal text, as the binding's
    `big_int_to_python` does.
  - pyo3 is a public dependency under the feature, since `lower` and
    `lift` take pyo3 types (D-S12-4). The crate README says so.
  - The feature leaves the MSRV at 1.85, since pyo3 0.29 needs 1.83.
  - CONTRIBUTING changes in two places:
    - "Porting to Rust" now says that `rust/fhy-core` depends on PyO3
      only under the off-by-default `sympy` feature;
    - "Canonical values keep their identity in Python" now says that the
      core holds no Python objects outside that feature's
      `SympySimplifier`, which holds its SymPy handles.
- **D-S12-3: the binding enables `fhy-core/sympy`** (the direction;
  D-S9's `ndarray` precedent).
  - `fhy-core-py` depends on `fhy-core` with `features = ["sympy"]`,
    beside S9's `"ndarray"`.
  - Cargo unifies features over the packages a command selects (the
    probe above):
    - every workspace build (`cargo build`, `clippy`, `doc` and `test
      --workspace`) compiles the backend and its stories;
    - `cargo test -p fhy-core`, `golden_expanded`, the packaged-crate test
      and docs.rs build the default features.
  - In a wheel, maturin sets `PYO3_BUILD_EXTENSION_MODULE`, so the one
    pyo3 in the extension links no libpython. The core's backend then
    attaches to the host interpreter, as the binding does.
  - The rejected alternative was a binding feature that maturin enables
    through `[tool.maturin] features`. It would keep plain workspace tests
    free of SymPy, but it puts `cfg` branches in the binding, and a build
    without the feature leaves `SolverBackend.SYMPY` without a backend.
    N-S12-1 treats the test-time consequence.
- **D-S12-4: the core's public API** (crate conventions; D-S8-9's
  shape for `Z3Solver`). Everything below is under `#[cfg(feature =
  "sympy")]` in a new private `solver/sympy.rs` (with `sympy/load.rs`,
  `lower.rs`, `lift.rs`, `simplify.rs`, `error.rs` and `prelude.py`),
  and each item has the one path `fhy_core::solver::X`. S12.3 settles the
  sketch test-first, as D-S8-2's was:

  ```rust
  #[derive(Debug, Default)]
  #[non_exhaustive]
  pub struct SympySimplifier { /* PyOnceLock<Handles>: the sympy module, cached classes, the prelude */ }
  impl SympySimplifier {
      pub fn new() -> Self;                                         // loads nothing
      pub fn with_embedded_python() -> Self;                        // Python::initialize(), then new()
      pub fn load(&self) -> Result<(), SympyUnavailableError>;      // import SymPy and the prelude now
      pub fn lower<'py>(&self, py: Python<'py>, expression: &Expression,
          context: &SimplifyContext<'_>) -> Result<Bound<'py, PyAny>, SympyError>;
      pub fn lift(&self, object: &Bound<'_, PyAny>) -> Result<Expression, SympyError>;
      pub fn simplify_object<'py>(&self, object: &Bound<'py, PyAny>) -> Result<Bound<'py, PyAny>, SympyError>;
      pub fn substitute<'py>(&self, object: &Bound<'py, PyAny>,
          environment: &HashMap<Identifier, Expression>, context: &SimplifyContext<'_>)
          -> Result<Bound<'py, PyAny>, SympyError>;
      pub fn substitute_symbols<'py>(&self, object: &Bound<'py, PyAny>,
          replacements: &Bound<'py, PyDict>) -> Result<Bound<'py, PyAny>, SympyError>;
  }
  impl Simplifier for SympySimplifier { /* name "sympy"; attach, lower, simplify_object, lift */ }

  #[non_exhaustive] pub enum SympyUnavailableError { NoInterpreter, MissingSympy(PyErr) }
  #[non_exhaustive] pub enum SympyPhase { Lowering, Simplification, Substitution, Lifting }
  #[non_exhaustive] pub enum SympyError {
      Unavailable(SympyUnavailableError),
      IllTyped(NonBooleanLogicalOperandError),         // the screen of `lower`
      CallNeedsInlining(Callee), ConstantCalled(FunctionName),
      NoSympyLowering(FunctionName), UnknownFunction(FunctionName),
      ConstantValueUnknown(Identifier),                // a context without a registry
      BoundNativeConstant(Vec<Identifier>),            // `substitute`
      ComplexInfinity, PartialPiecewise(String), UnsupportedNode(String),
      Implies, UnnamedSymbol(String),
      Python { phase: SympyPhase, source: PyErr },     // an exception SymPy raised
  }
  impl SympyError { pub fn phase(&self) -> SympyPhase; }
  ```

  - `lower` runs `BooleanScreen::check_logical_operands` with the
    context's sorts first, as `SmtScript::lower` runs its checks. The
    facade has screened already, and `Simplifier::simplify` lowers
    without repeating the screen.
  - `simplify_object` is the best-effort simplification of D-S12-8, from
    SymPy to SymPy. `substitute` is `substitute_sympy_expression_variables`,
    and `substitute_symbols` is the substitution pass. The Python API
    reaches the mapping through these, so it has one copy.
  - The `Simplifier` implementation attaches once, then lowers,
    simplifies and lifts. It returns a `SympyError` boxed as the
    `BackendError`, which the binding downcasts.
  - `SympySimplifier` is `Send + Sync`, since its handles are `Py`s, and
    it is not `Clone`, since cloning a `Py` needs the GIL. The binding
    shares it behind an `Arc`.
  - Every `Display` is one lowercase line naming the node or name. The
    texts keep the phrases the Python message tests match, as D-S7-12's
    did.
- **D-S12-5: no interpreter, no SymPy, and the embedding helper** (the
  direction: pyo3 without `auto-initialize`; "no fallback"; crate
  conventions: no global state, D-S8-15).
  - **Nothing initializes Python implicitly.**
    - Every entry point attaches through `Python::try_attach`. With no
      interpreter it answers `SympyUnavailableError::NoInterpreter`, and
      never panics.
    - A failing `import sympy` answers `MissingSympy`, holding the
      `ImportError`.
    - Either one reaches a `Solver` as `SolveError::Backend`, at query
      time, never as a degraded answer.
  - **Loading is lazy and per value.**
    - `new()` imports nothing, so the Python default solver can hold one
      without importing SymPy at `import fhy_core` (D-S8-16).
    - The first operation loads SymPy's module, the classes the lifting
      dispatches on, and the prelude into the value's `PyOnceLock`.
      Only success is kept, so a later call retries a failed import, as
      `functools.cache` retries today's `_resolve_adapter`.
  - **The capability query** is `load()`: `Ok` means that simplification
    can run. The binding's `is_backend_available(SYMPY)` keeps its
    meaning over it.
  - **The embedding helper** is `SympySimplifier::with_embedded_python()`,
    an explicit opt-in.
    - It calls the safe `Python::initialize()`, which does nothing inside
      a running interpreter, such as an extension module, and then
      returns `new()`.
    - The embedded interpreter is the one linked at build time (through
      `PYO3_PYTHON` or `PATH`), and it runs without signal handlers.
    - It finds SymPy through `PYTHONPATH` or `PYTHONHOME`, and a
      non-system libpython through the loader's path (`LD_LIBRARY_PATH`
      on Linux). The crate README documents this with the probe's recipe.
    - A pure-Rust user can instead call `pyo3::Python::initialize()`
      themselves.
    - The core never finalizes the interpreter.
- **D-S12-6: attaching, threads and signals** (D-S8-11's detaching;
  crate conventions: `Send + Sync` backends).
  - Each operation attaches once for its whole run, since building and
    walking SymPy objects needs the GIL throughout.
  - The binding's facade still detaches around `Solver::simplify` (S8.4),
    and the backend attaches again inside. The backend never holds a Rust
    lock across a call into Python.
  - Several threads may share one backend. Their SymPy work runs one at a
    time on the GIL, as it does in Python today.
  - An exception raised inside SymPy, `KeyboardInterrupt` included,
    returns as `SympyError::Python` with the `PyErr` itself. The binding
    re-raises it (D-S12-11).
- **D-S12-7: the prelude, the one Python file in the core** (the
  direction; crate conventions: no global state in Rust).
  - **The file.** `solver/sympy/prelude.py` holds only what SymPy's
    extension points require Python code for:
    - `ParityOpaquePiecewise`, with `eval`, `_eval_is_even`,
      `_eval_is_odd` and `_eval_simplify`;
    - `hide_piecewise_parity` and `holds_partial_piecewise`, which those
      methods call;
    - the `round` function and its plain-function `eval` hook, which keeps
      a lowered node picklable.

    That is about 70 lines. The Rust side calls the same functions, so
    nothing is duplicated.
  - **Loading.**
    - It is compiled in with `include_str!`.
    - On the first load in an interpreter, the backend runs it into a new
      module object and publishes that with
      `sys.modules.setdefault("_fhy_core_sympy", module)`, which is atomic
      under the GIL.
    - Every backend then uses the module that won, so every lowered
      piecewise has one class.
    - The state is the interpreter's module table, not a Rust `static`,
      so CONTRIBUTING's global-state section gains nothing.
  - **Tooling.** The crate's `include` list and CI's package-contents
    pattern gain this one `.py` file. `noxfile.py`'s `SOURCES` gains its
    directory, so `ruff` and `mypy` check it.
  - **Pickles.** Y-S12-1 records the consequence for pickles.
- **D-S12-8: the mapping keeps today's semantics** (D-S4-2: the same
  meaning, pinned by 627 tests; D-S4-1 where the core's values differ).
  - **The tables.** Each table of the survey becomes a Rust `match` over
    `BuiltinFunction`, `BuiltinConstant`, `LiteralValue` and the operation
    enums.
  - **Literals and constants.**
    - A user constant lowers to its value exactly as a literal of that
      value lowers, through the context's registry. A decimal becomes a
      `Rational` (Y-S12-4).
    - A rational lifts through S9's `Decimal::to_f64_exact` in place of
      `is_decimal_text_exactly_binary`.
    - A symbol lifts through `Identifier::try_restore`, which advances the
      counter as today's `deserialize_from_dict` does.
  - **The workarounds**, one for one:
    - The prelude holds workaround 1 and the `round` hook.
    - Workarounds 2 to 8 and the best-effort cases are Rust.
    - Each `sympy.replace` stays a `replace` call, with a
      `PyCFunction::new_closure` predicate and replacement, so SymPy's
      traversal rules are unchanged.
    - The masking dummies are SymPy `Dummy`s.
  - **What is read when.**
    - The backend caches the classes it dispatches on.
    - It reads the functions it calls on a whole expression
      (`sympy.simplify` and `piecewise_fold`) from the module on each
      call, as the Python bridge did, so the four tests that patch
      `sympy.simplify` keep working.
  - **Deep trees.** The lowering, the lifting, and the Boolean-position
    walks of the substitution keep their pending nodes on the heap, as
    the core's other walks do (Y-S12-5).
  - **Optimizations.** A walk may skip SymPy work it can prove changes
    nothing, such as the three piecewise folds of a tree without a
    piecewise. Each such shortcut is pinned by a story comparing its
    output with the unshortened path.
- **D-S12-9: the simplify context carries the function registry, and
  `Solver::simplify` takes the context** (crate conventions: I.3 rule 6;
  S8.2's note, which kept the context open for "the sorts of native
  constants and named functions, and perhaps their values"; D-S8-2's
  `ask(question, &QueryContext)` shape).
  - `SimplifyContext::from_registry(&FunctionRegistry)` reads both sorts
    and entries from one registry, and `registry() ->
    Option<&FunctionRegistry>` returns it.
  - `SimplifyContext::new(&dyn SortLookup)` stays, with no registry. The
    SymPy backend then refuses a user constant as
    `ConstantValueUnknown`, and names every call it cannot lower
    `UnknownFunction`.
  - `Solver::simplify(expression, environment, context:
    &SimplifyContext<'_>)` replaces the `sorts` parameter. It screens
    with `context.sorts()`, and passes the context through.
  - The binding passes `SimplifyContext::from_registry` of its snapshot,
    built inside the detached closure, as S8.4's `QueryContext` is.
  - This lands in S12.2 without the feature, since it is independent of
    SymPy. The breaking change has two call sites: the binding's facade,
    and the solver stories. `Simplifier` itself does not change.
- **D-S12-10: the Python API keeps its names** (D-S4-2; D-S5-9 for new
  names; D-S9's pass precedent).
  - `SympySimplifier` is `_rs.SympySimplifier`, re-exported from
    `passes/sympy.py`.
    - It is a native `Simplifier` with `name` `"sympy"` and
      `simplify(expression)`, the whole pipeline over the registry
      snapshot.
    - It has the Rust names `lower(expression)`, `lift(sympy_expression)`,
      `substitute(sympy_expression, environment)` and
      `substitute_symbols(sympy_expression, replacements)`.
    - It pickles as a call. Construction loads nothing.
  - **The four functions keep their signatures and phases.**
    `convert_expression_to_sympy_expression` screens with
    `validate_logical_operands`, then runs the converter pass.
    `substitute_sympy_expression_variables` returns a Python `bool` as it
    stands, then runs the substitution. `convert_sympy_expression_to_expression`
    runs the lifting pass.
  - **The three registered passes stay**, as thin `CompilerPass`es over a
    module-level `SympySimplifier()`:
    - `ExpressionToSympyConverter` keeps the static `format_identifier`;
    - `SymPyToExpressionConverter` keeps `convert`, and `convert_expr`,
      `convert_bool` and `convert_relational`, which check the node's
      SymPy kind in Python and then lift;
    - both keep `get_noop_output`;
    - the `visit_*` methods and the class tables go (Y-S12-2), as S9
      dropped `ExpressionEvaluator`'s visitor methods;
    - the module-level backend is loaded when `passes/sympy.py` is
      imported, since that import has imported SymPy already, so the
      prelude module exists wherever the bridge has been imported.
  - **Resolution.**
    - `SolverBackend.SYMPY` resolves, through `_resolve_adapter`, to one
      `SympySimplifier`, after importing `passes/sympy.py` for its import
      guard.
    - The default solver holds its own `SympySimplifier()` directly
      (Y-S12-8).
    - `get_backend_capabilities` and `is_backend_available` are
      unchanged in meaning.
    - `SolverBackend` gains no member. D-S8-3 expected a new member for
      "the later Rust CAS backend", but this backend is the same CAS
      under the same name.
  - **Stubs.** Everything new goes into `_rs.pyi`.
- **D-S12-11: errors keep their Python classes and phases** (D-S4-2;
  D-S7-12 for texts; this refines D-S8-14's row "`Backend` from a Rust
  backend", which the SymPy backend no longer follows, since its failures
  have a Python meaning today).

  | Core | Python |
  |---|---|
  | `Unavailable(MissingSympy)` | `SolverBackendUnavailableError`, with today's message naming `fhy_core[sympy]` and `fhy_core[solvers]` (the binding's text) |
  | `Unavailable(NoInterpreter)` | cannot arise inside the extension; mapped to `SolverBackendError` |
  | `IllTyped` | `NonBooleanLogicalOperandError`, unwrapped |
  | `BoundNativeConstant` | `NativeConstantBindingError`, unwrapped, as today's check before the pass |
  | `CallNeedsInlining`, `ConstantCalled`, `NoSympyLowering`, `UnknownFunction`, `ConstantValueUnknown` | `TypeError`, inside `PassExecutionError` from the lowering pass |
  | `ComplexInfinity` | `ComplexInfinityLiftError`, inside `PassExecutionError` from the lifting pass |
  | `PartialPiecewise` | `PartialPiecewiseError`, likewise |
  | `UnsupportedNode` | `TypeError`, likewise |
  | `Implies` | `NotImplementedError`, likewise, after the binding logs today's WARNING on the bridge's logger |
  | `UnnamedSymbol` | `RuntimeError`, likewise |
  | `Python { phase: Lowering, Substitution or Lifting }` | the exception itself, inside `PassExecutionError` from that phase's pass |
  | `Python { phase: Simplification }` | the exception itself, raw, as `sympy.simplify`'s failures are today |

  - "Inside `PassExecutionError`" means what the pass infrastructure
    raises today, naming the registered pass, with the error as
    `__cause__`. The thin passes get it by raising the mapped error from
    `run_pass`. `simplify_expression` gets the same wrapping from the
    binding, keyed on the phase.
  - A `BaseException` that is not an `Exception`, such as
    `KeyboardInterrupt`, is never wrapped.
- **D-S12-12: imports stay lazy** (D-S8-16).
  - `import fhy_core` imports no SymPy. `passes/sympy.py` keeps its
    module-level guard, since importing the bridge without SymPy raising
    `SolverBackendUnavailableError` is pinned.
  - Constructing `_rs.SympySimplifier` imports nothing.
  - The fresh-interpreter test and `test_import_graph.py` keep passing.
- **D-S12-13: testing the feature under `cargo test`** (the tests rule;
  D-S8-19's `cfg` stories; D-S8-18's CI pattern; CONTRIBUTING's
  fresh-process rule).
  - **The stories.** They sit under `cfg(feature = "sympy")` in
    `tests/it/solver/`.
    - A support helper builds one `SympySimplifier::with_embedded_python()`
      for the binary and calls `load()`.
    - Without SymPy, that call fails with a message naming `PYO3_PYTHON`,
      `PYTHONPATH` and `LD_LIBRARY_PATH` and the recipe (N-S12-1, resolved
      as (a)); nothing skips.
    - The tests share one interpreter, and the GIL serializes them.
  - **A fresh-process target.** `tests/sympy_unavailable.rs`, with
    `required-features = ["sympy"]`, is one `#[test]` in a fresh process:
    - `NoInterpreter` before any initialization;
    - then, after `Python::initialize()` with `sys.modules["sympy"] =
      None`, `MissingSympy`;
    - then a `load()` that succeeds once the entry is removed.

    CI's integration-target list becomes `id_cap_decode it
    sympy_unavailable`.
  - **The CI `rust` job.**
    - The step that installs z3-solver into a venv also installs sympy,
      and exports `PYO3_PYTHON` (that venv's interpreter) and `PYTHONPATH`
      (its site-packages). The runner's Python has a shared libpython,
      since the job already links it for the binding.
    - `cargo test --workspace --locked --all-features` then runs the
      stories.
    - The `rust-msrv` job needs no change (`cargo +1.85 check`, which also
      builds the feature through the binding), and neither does the
      packaged-crate test (default features).
    - The step writes `PYO3_PYTHON` and `PYTHONPATH` to `$GITHUB_ENV`, so
      every later step has them: each `cargo test` that builds the
      workspace (`--workspace --all-features`) links the venv's Python and
      finds its SymPy. The default-feature steps (`-p fhy-core`, the
      packaged crate) do not enable the feature and ignore them.
  - **Locally.** The known `-lpython3.11` failure comes from the tooling's
    deadsnakes 3.11, which has no `libpython3.11.so`. The Rust gate
    therefore runs with a Python that has one. On this machine it is a
    uv-managed CPython 3.11.16, installed with `--no-bin` inside the
    worktree, and a venv of it holding sympy 1.14:

    ```bash
    G=$PWD/target/gate-python
    uv python install --no-bin --install-dir "$G/pythons" 3.11.16
    uv venv --python "$G"/pythons/cpython-3.11.16-*/bin/python3.11 "$G/venv"
    VIRTUAL_ENV="$G/venv" uv pip install sympy==1.14.0
    export PYO3_PYTHON="$G/venv/bin/python"
    export PYTHONPATH="$G/venv/lib/python3.11/site-packages"
    export LD_LIBRARY_PATH="$(echo "$G"/pythons/cpython-3.11.16-*/lib)"
    ```

    With the `z3` feature too, `LD_LIBRARY_PATH` also names the
    z3-solver wheel's `lib` (S8.3). CONTRIBUTING's porting section records
    the recipe, which nothing outside the worktree needs.
- **D-S12-14: `deny.toml` does not change** (D-S8-9's reasoning).
  - `cargo deny` checks the workspace graph with `all-features = true`.
    pyo3 and its tree (`pyo3-ffi`, `pyo3-build-config`, `pyo3-macros`,
    `target-lexicon` and the rest) are already in that graph through the
    binding.
  - The feature adds no crate, since no pyo3 feature is enabled
    (D-S12-2). So the licenses, bans and sources pass unchanged, which
    S12.3 confirms with `cargo deny check`.
  - A downstream crate enabling `sympy` gets pyo3 without `macros`, under
    MIT OR Apache-2.0.
- **D-S12-15: the Rust tests specify the backend first** (the tests
  rule; S7.2's and S8.2's test-first practice).
  - The lowering, lifting, substitution, workarounds, best-effort cases,
    errors and loading are specified by Rust tests written against
    `todo!()` stubs.
  - A traceability table maps `test_sympy_pass.py`,
    `test_sympy_natives.py` and `test_sympy_pass_properties.py` to them.
- **D-S12-16: the Python tests are rewritten, not skipped** (the tests
  rule; D-S8-20).
  - The behavioral tests stay, and now run through the Rust backend.
  - A test changes only where a decision changes what it pins, and each
    change is recorded with its reason.
- **D-S12-17: benchmarks before and after** (cross-cutting rule 5;
  CONTRIBUTING's 10%). See the plan below. A row more than 10% slower is
  optimized, or recorded as an accepted cost for the maintainer.
- **D-S12-18: docs.**
  - The crate README gains a "The `sympy` feature" section beside "The
    `z3` feature": what it adds, the public pyo3 dependency, embedding,
    finding SymPy, and the link recipe. docs.rs keeps the default
    features.
  - CONTRIBUTING changes as D-S12-2 and D-S12-13 say, and its
    Python-to-Rust table maps `symbolic.expression.passes.sympy` to
    `fhy_core::solver` (`SympySimplifier`).
  - The Python README's expression and solver rows name the Rust-backed
    SymPy backend.

### Benchmark plan

`benchmarks/test_sympy.py` is new in S12.1. The baseline runs it against
today's Python bridge. Every call whose spelling changes sits in a helper
marked with its decision. The trees reuse `test_expression.py`'s deep tree
(100 operations over four identifiers).

| Benchmark | Measures |
|---|---|
| `test_lower_to_sympy_of_a_deep_tree` | `convert_expression_to_sympy_expression` of the deep tree: the lowering's throughput, a Python visitor before and the Rust walk after |
| `test_lift_from_sympy_of_a_deep_tree` | `convert_sympy_expression_to_expression` of its lowered form |
| `test_substitute_sympy_variables_of_a_deep_tree` | `substitute_sympy_expression_variables` binding the four identifiers |
| `test_sympy_simplifier_of_a_ground_comparison` | `SympySimplifier().simplify(3 < 10)`: the backend alone, without the facade |
| `test_simplify_expression_of_a_bound_piecewise` | a piecewise with Boolean case conditions, bound: the piecewise workarounds |
| `test_simplify_expression_of_a_boolean_comparison` | `(x < 1) == b`, bound: the masking of Boolean comparisons |
| `test_first_simplification_in_a_fresh_interpreter` | a fresh interpreter importing `fhy_core` and simplifying once, five rounds through `pedantic`: the import, the lazy load and the prelude |

The rows of `benchmarks/test_solver.py` are compared too:

- `test_simplify_expression_of_a_ground_comparison`, the param validation
  path;
- `test_simplify_expression_symbolic`;
- `test_equation_constraint_evaluate_with_bindings`;
- `test_nat_param_is_value_valid`;
- `test_import_fhy_core`, which must not import SymPy.

The paths at risk:

- **The lowering.** It trades a Python visitor call per node for a Rust
  call into SymPy per node. SymPy's constructors dominate either way, so
  the lowering should gain by the visitor's overhead, about half of the
  deep tree's 916 µs.
- **The workaround walks** (19 µs of the ground comparison) become Rust
  closures called by SymPy's `replace`, or are skipped where D-S12-8
  allows.
- **The lifting** becomes Rust dispatch on cached classes.
- **Simplification** loses the deferred adapter's hop into Python and
  back. Every simplification row is expected to be faster, while
  `sympy.simplify` itself does not change.
- **The fresh-interpreter row** adds the prelude's load, a few
  milliseconds once per process against SymPy's 208 ms import.

### Needs the user

- **N-S12-1 (resolved 2026-09-26 by the user, as option (a)): what the
  SymPy stories do when no SymPy is importable.** Every workspace build enables the feature (D-S12-3), so `cargo test
  --workspace` runs the stories. Today, the only extra that command needs
  is a linkable libpython for the binding's test harness; with this
  change it would also need SymPy on the embedded interpreter's path. The
  precedents point two ways:
  - the `z3` feature's stories simply require libz3;
  - the process backend's real-solver tests run only when
    `FHY_SMT_SOLVER` names a solver, and CI sets it.
  - (a) **Required.** Under the feature, the stories fail when SymPy
    cannot be imported, with a message giving the recipe. CI cannot pass
    them by accident, but every local `cargo test --workspace` needs
    `PYTHONPATH` (and, on this machine, `PYO3_PYTHON`).
  - (b) **Opt-in by environment**, as `FHY_SMT_SOLVER` is. The stories
    run when `FHY_SYMPY_TESTS=1`, which CI's `rust` job sets, and pass
    without running otherwise. Local runs need nothing, but a local run
    can go green without testing the backend.
  - (c) **A maturin-only binding feature** (D-S12-3's rejected
    alternative). Plain workspace builds have no backend, and
    `--all-features` has it, which CI already uses. The binding gains
    `cfg` branches, and a build of the extension without the feature has
    no SymPy backend.

  Recommendation: (a). It is the only option in which a green Rust gate
  always means the backend was tested, as the `z3` feature's gate does.
  The local cost is one pair of environment variables, which the gate
  already needs on this machine for the binding's harness, since the
  tooling Python has no shared libpython.

### S12 resolutions (decided by the user, 2026-09-26)

- **N-S12-1: (a) required.** Wherever the `sympy` feature is enabled,
  which is every workspace build, the SymPy stories fail when SymPy cannot
  be imported, and the failure message gives the setup recipe:
  `PYO3_PYTHON`, `PYTHONPATH`, and `LD_LIBRARY_PATH` where the linked
  libpython is not on the loader's path. There is no opt-in variable.
  D-S12-13 records the recipe for CI and for this machine.

### S12 rebase onto S9 (2026-09-26)

The design commit was rebased onto `dev-rust` at 3a53195, S9's last
commit. The one conflict was additive: S9's and S12's checklist entries
and sections, S12's after S9's. Checked against S9's code, the design
holds with these adjustments:

- `rust/fhy-core/Cargo.toml`'s `[features]` now holds `z3` and `ndarray`,
  and `sympy` joins them. The binding's `fhy-core` dependency already
  names `features = ["ndarray"]`, and gains `"sympy"`.
- rust-numpy (`numpy` 0.29) depends on pyo3 with `macros`, so in the
  workspace graph pyo3 has `macros` whatever the core asks for. The core
  still asks for no pyo3 feature (D-S12-2), and `cargo deny` sees no new
  crate (D-S12-14).
- `Decimal::to_f64_exact` and `fhy_core::expression::evaluate` have
  landed, so S12.3 uses them directly (D-S12-8; the properties' oracle).
- `native_lowering.py` is now a thin layer over `_rs`, and
  `passes/sympy.py` still imports `is_decimal_text_exactly_binary` from
  it; S12.5 drops that import.
- S9 marked 13 tests of `test_sympy_pass.py` `numpy`, since they tabulate
  with the NumPy evaluator as the oracle. The marks stay.

### Steps

1. **S12.1: benchmarks.** Add `benchmarks/test_sympy.py` as planned
   above, and record the baseline here on today's Python bridge, with the
   `test_solver.py` rows beside it.
2. **S12.2: the simplify context** (D-S12-9), in the core, test-first,
   without the feature:
   - `SimplifyContext::from_registry` and `registry()`;
   - `Solver::simplify` taking the context;
   - the solver stories and the binding's facade call site.

   The Python suite stays green.
3. **S12.3: the `sympy` feature**, test-first, against `todo!()` stubs:
   - the manifests (D-S12-2);
   - `solver/sympy.rs` with its submodules and `prelude.py`;
   - the stories and the fresh-process target;
   - the crate README, the `include` list and the package-contents
     pattern;
   - `noxfile.py`'s `SOURCES`;
   - the CI `rust` job (D-S12-13).

   The step ends with the Rust gate green with the feature, and with
   `cargo deny check`.
4. **S12.4: the binding.**
   - `fhy-core-py` enables `sympy` (D-S12-3).
   - `solver/sympy.rs` holds `_rs.SympySimplifier`, and
     `build_simplifier` learns it.
   - `solver/error.rs` gains D-S12-11's mapping and the `Implies`
     warning.
   - Everything new goes into `_rs.pyi`.

   Nothing in Python uses it yet, so the suite stays green.
5. **S12.5: the Python switch** (marked breaking). `passes/sympy.py`
   becomes the thin layer of D-S12-10, and `solver.py` loses
   `_DeferredSimplifier`. It lands with S12.6 when the migration is small
   enough to review in one commit. Otherwise it leaves exactly the tests
   of the migration list failing, as S7.4 did.
6. **S12.6: tests.** Migrate the tests and add the interface suite (the
   test plan below).
7. **S12.7: benchmarks after,** recorded here with the verdict, then the
   status, the implementation notes, the docs of D-S12-18, and this
   checklist.

Commit per step. Every step ends with these green:

- `pytest`, and `pytest -m "not very_slow"`;
- the `property` session and `tests_minimal`;
- `lint` and `type_check`, clean;
- `tests/test_rs_stub.py`;
- the Rust gate: fmt, clippy `-D warnings` with and without
  `--all-features`, tests with the default features and with
  `--all-features`, doc `-D warnings`, deny, and `cargo +1.85 check`.

The feature is built from S12.3 on.

### Test plan

**Rust tests, written first (S12.2 and S12.3),** in `tests/it/solver/`:

- **`solver_stories.rs` (S12.2):**
  - `from_registry` gives the facade's screen the registry's sorts;
  - a simplifier receives the context the solver was given, its registry
    included;
  - `new(sorts)` has no registry.
- **`sympy_lowering_stories.rs`**, pinned by SymPy's `srepr` text:
  - each literal form: a big integer, `0.1` as its binary `Float`, the
    non-finite floats, and a decimal as its exact `Rational`;
  - the symbol naming;
  - the built-in constants, and a user constant of each sort through a
    registry, a decimal included;
  - `ConstantValueUnknown` without a registry;
  - each operation, with `FLOOR_DIVIDE` as `floor` of a quotient and
    `LOGICAL_NOT` as `Not`;
  - n-ary `And`/`Or`, with a Boolean piecewise operand expanded;
  - `Eq`/`Ne` of Booleans, with a piecewise operand expanded;
  - a piecewise as `ParityOpaquePiecewise` with its final `True` branch;
  - conditions with a negated symbol side, and folded piecewise
    conditions;
  - the 19 natives, `round`'s integer fold, and the four call refusals
    with their texts;
  - the screen's refusal before any SymPy call;
  - a 100,000-level tree lowering on a small stack, where SymPy allows
    it.
- **`sympy_lifting_stories.rs`:**
  - `Add` and `Mul` folded to the right;
  - `Mod`, `Pow`, and `sqrt` from `Pow(x, 1/2)`;
  - the 15 native classes;
  - the constants, and `-oo` as the negation of `inf`;
  - the rational rule: decimal text or `DIVIDE`, and the sign;
  - `Float`, and a big `Integer`;
  - a symbol restored, with the counter advanced past it, and a name
    without `_`;
  - `Xor`, `Nor`, `Nand` and `ITE`;
  - the refusals, each with its text: `zoo`, a partial piecewise,
    `Implies`, and an unsupported node;
  - a deep SymPy tree lifting on a small stack.
- **`sympy_simplify_stories.rs`:**
  - ground comparisons deciding to `true` or `false`;
  - each workaround's pinned example from `test_sympy_pass.py`: the
    parity cases of SymPy 1.14, the partial-evaluation skip, the masked
    Boolean comparisons, the folds out of relationals and integer parts;
  - each best-effort case. `PrecisionExhausted` is forced by patching
    `sympy.simplify` through `py.run` in the embedded interpreter, as the
    Python tests patch it;
  - `sympy.simplify` read at call time;
  - the substitution: simultaneous, keeping Boolean positions Boolean,
    and refusing a bound constant;
  - the phases of `SympyError`;
  - `name()`;
  - a `Solver` holding the backend, end to end;
  - eight threads sharing one backend;
  - `KeyboardInterrupt` raised from a patched `sympy.simplify` returning
    as `SympyError::Python`;
  - the prelude loaded once and shared by two backends, and a lowered
    `round` node pickling within the process.
- **`sympy_properties.rs`:**
  - a random ground integer or Boolean tree simplifies to the literal
    S9's evaluator computes;
  - lifting the lowering of a random screened tree gives an expression
    whose evaluation at random bindings agrees with the original's;
  - substituting in SymPy agrees with substituting in the core.
- **`tests/sympy_unavailable.rs`:** the fresh-process target of D-S12-13.
- **A traceability table**, as S8.2's, from `test_sympy_pass.py`,
  `test_sympy_natives.py` and `test_sympy_pass_properties.py`.

**The interface suite,
`tests/symbolic/expression/passes/test_sympy_rust_binding.py`,** covers
what the binding adds over the core:

- **The class.**
  - `SympySimplifier` extends `SimplifierBase`, is a registered
    `Simplifier`, is frozen and refuses subclassing, is named `sympy`,
    and pickles.
  - Construction imports no SymPy. This is checked in a subprocess with
    `sys.modules["sympy"] = None`, where construction succeeds and
    `simplify` raises `SolverBackendUnavailableError` with its message.
- **The native path.**
  - A `Solver` holding it calls no Python backend: a counting
    `sys.setprofile` hook sees no Python frame of `passes/sympy.py` during
    `simplify_expression`.
  - The result is materialized beside the input.
- **The methods.** `lower`, `lift`, `substitute` and
  `substitute_symbols` each agree with the public function over them.
- **Errors.** Each row of D-S12-11, including the pass named by
  `PassExecutionError`, the unwrapped screen and constant errors, a raw
  simplification failure, `KeyboardInterrupt` passing through, and the
  `Implies` warning.
- **Pickles.** A lowered `round` node and a lowered piecewise pickle
  within the process, and load in a fresh one after `passes.sympy` is
  imported (Y-S12-1).
- **Resolution.** `SolverBackend.SYMPY` resolves to one object, the
  default solver's simplifier is a `SympySimplifier`, and
  `is_backend_available` holds.
- **Threads.** Concurrent simplifications from several threads through
  one backend.

**Migrating the existing tests.** No test is skipped or deleted without a
rewrite, and each change is recorded with its reason:

- **`test_sympy_pass.py` (627).**
  - The imports of `_NATIVE_CONSTANT_LIFT`, `_NATIVE_CONSTANT_LOWER`,
    `_NATIVE_FUNCTION_LOWER` and `_ParityOpaquePiecewise` go.
    - The table tests become checks over the public lowering and lifting
      of every built-in name and constant.
    - The parity class is reached as the type of a lowered piecewise.
  - `test_sympy_converter_visit_literal_unsupported_value_raises`
    becomes a check that the lowering refuses a value that is no
    `Expression`, since there is no Python visitor (Y-S12-2).
  - The test that patches `convert_expression_to_sympy_expression` to
    reach the partial-piecewise fallback patches `sympy.simplify` to
    return a partial piecewise instead (Y-S12-6).
  - `match=` texts follow the core's texts where they differ (Y-S12-3).
  - The rest keep their meaning.
- **`test_sympy_natives.py` (59).** The pickle test keeps its meaning
  within the process.
- **`test_sympy_pass_properties.py` (6), `test_cross_cutting.py` (20) and
  the constraint and param tests:** unchanged.
- **`test_solver.py`.**
  `test_simplify_expression_matches_direct_bridge_pipeline` compares with
  the native `SympySimplifier`.
- **`test_solver_rust_binding.py`.** The adapter and default-solver tests
  keep their assertions. The S8 interface tests of a Python
  `Simplifier` keep using their fakes.

### Coordination with S9 and S11

This branch is rebased onto `dev-rust` after S9, so its edits to shared
files stay small and additive:

- **S9 items S12 uses:**
  - `Decimal::to_f64_exact` (D-S12-8);
  - the evaluator, as the properties' oracle.

  S12.3 starts after the rebase. If it has to start before, a private
  equivalent stands in, and the rebase replaces it.
- **`native_lowering.py`:** S12 stops importing it from
  `passes/sympy.py` and never edits it. After the rebase, one sentence of
  its module docstring may still describe the Python lifter. If so, it is
  corrected in S12.7, on S9's version.
- **`Cargo.toml`:** the one `pyo3` line of the workspace table. S9 adds
  the `numpy`, `ndarray` and `libm` lines.
- **`rust/fhy-core/Cargo.toml`:**
  - the optional `pyo3` line;
  - `sympy` in `[features]`, beside `z3` and S9's `ndarray`;
  - the prelude in `include`.
- **`rust/fhy-core-py/Cargo.toml`:** `features = ["macros"]` on `pyo3`,
  and `"sympy"` beside S9's `"ndarray"` on `fhy-core`.
- **`Cargo.lock`:** regenerated after the rebase, never hand-merged.
- **Untouched:** `deny.toml` (D-S12-14), `pyproject.toml` (the `sympy`
  extra and markers stay as S8.7 set them), and `tests/conftest.py`.
- **Additive edits:**
  - the binding's `lib.rs` (one `#[pymodule_export]` line);
  - `_rs.pyi` (one block);
  - CONTRIBUTING (D-S12-2, D-S12-13 and one table row);
  - the crate README (one feature section);
  - `noxfile.py` (one `SOURCES` entry);
  - the CI workflow (the install step's two exports, the target list, and
    the package-contents pattern).
- **This document:** this section, appended, and one entry at the end of
  the Progress checklist.
- **S11 (types)** uses neither the simplifier nor the SymPy bridge, and
  S12 does not use types, so no code depends across the two.

### S12.1 baseline (2026-09-26, a641699 plus the new benchmarks)

`benchmarks/test_sympy.py` implements the benchmark plan's seven rows.
Every call keeps its spelling across S12, so no helper carries a decision
mark.

- **The bound piecewise** is `(x + 1 if b; x * 2 if x > 0; x - 1
  otherwise) > 3` with `x = 2` and `b = false`.
- **The Boolean comparison** is `(x < 1) == b && x > -5` with `b = true`,
  which leaves `x` free. So `sympy.simplify` works on a relational, and
  the row is dominated by SymPy.
- **The fresh-interpreter row** runs `simplify_expression` of a bound
  comparison once in a new process, five rounds through `pedantic`.

The measurements:

- Command: `.venv/bin/python -m pytest benchmarks/test_sympy.py
  benchmarks/test_solver.py -k '...' -n 0 --benchmark-only`, with the five
  simplification rows of `test_solver.py` selected.
- Measured: the median time per call of today's Python bridge.
- Environment: the S0 machine, with Python 3.11.13, sympy 1.14.0 and
  pytest-benchmark 5.3.0.
- The load average was below 2, and the table lists the best of three
  runs' medians.

| Benchmark | before |
|---|--:|
| `test_lower_to_sympy_of_a_deep_tree` | 433.4 µs |
| `test_lift_from_sympy_of_a_deep_tree` | 104.6 µs |
| `test_substitute_sympy_variables_of_a_deep_tree` | 76.6 µs |
| `test_sympy_simplifier_of_a_ground_comparison` | 41.6 µs |
| `test_simplify_expression_of_a_bound_piecewise` | 105.1 µs |
| `test_simplify_expression_of_a_boolean_comparison` | 8.55 ms |
| `test_first_simplification_in_a_fresh_interpreter` | 444 ms |
| `test_solver.py::test_simplify_expression_of_a_ground_comparison` | 50.8 µs |
| `test_solver.py::test_simplify_expression_symbolic` | 72.6 µs |
| `test_solver.py::test_equation_constraint_evaluate_with_bindings` | 60.2 µs |
| `test_solver.py::test_nat_param_is_value_valid` | 63.3 µs |
| `test_solver.py::test_import_fhy_core` | 203 ms |

What the numbers show:

- **The lowering** of the deep tree takes 433 µs, a visitor call and a
  SymPy constructor per node. Lifting takes 105 µs, and substituting four
  literals 77 µs.
- **The simplifier alone** takes 42 µs on a ground comparison. Through
  `simplify_expression` with a binding it takes 51 µs, and the value
  checks of the constraint and param layers take about 60 µs.
- **The Boolean comparison** takes 8.6 ms, almost all of it
  `sympy.simplify` on a relational.
- **A fresh interpreter** that imports `fhy_core` and simplifies once
  takes 444 ms, of which SymPy's import is most of the difference from
  the 203 ms plain import.

### S12.2 and S12.3 status: the context and the `sympy` feature

**S12.2.** `SimplifyContext::from_registry(&FunctionRegistry)` and
`registry()` join `SimplifyContext::new`. `Solver::simplify` takes the
context instead of a `SortLookup`, and the binding's facade passes the
context of its registry snapshot. The three new solver stories were written
before the change, and failed to compile against the old API.

**S12.3: the manifests.**

- The workspace declares `pyo3 = { version = "0.29", default-features =
  false }`, and the binding asks for `features = ["macros"]`.
- `fhy-core` gains `pyo3 = { workspace = true, optional = true }` and
  `[features] sympy = ["dep:pyo3"]`. Its `include` list gains
  `/src/**/*.py`, for the prelude.
- The binding does not enable the feature yet; S12.4 does.

**S12.3: the backend.** `rust/fhy-core/src/solver/sympy.rs` holds
`SympySimplifier`, with these submodules:

- `sympy/load.rs`: the handles, and publishing the prelude;
- `sympy/lower.rs`;
- `sympy/lift.rs`;
- `sympy/boolean.rs`: the piecewise expansion, the Boolean positions, and
  `replace` with Rust hooks;
- `sympy/substitute.rs`: a bottom-up rebuild on a work list, and the
  substitution;
- `sympy/simplify.rs`: the masking, the folds, and the best-effort cases;
- `sympy/error.rs`;
- `sympy/prelude.py`.

Every public item has the one path `fhy_core::solver::X`: `SympySimplifier`,
`SympyError`, `SympyErrorKind`, `SympyPhase` and `SympyUnavailableError`.

**Tests.** `tests/it/solver/` gains:

- `sympy_lowering_stories.rs`;
- `sympy_lifting_stories.rs`;
- `sympy_simplify_stories.rs`;
- `sympy_properties.rs`, with the core's evaluator as the oracle.

The shared embedded interpreter and the thread-local patching of a SymPy
function are in `tests/it/support/sympy.rs`. Together these are 143 tests,
counting `rstest` cases and the two properties. `tests/sympy_unavailable.rs`,
with `required-features = ["sympy"]`, is the fresh-process target of
D-S12-13.

**The Rust gate** passed on this machine with `target/gate-python/env.sh`
(D-S12-13's recipe, plus the z3 variables of S8.3):

- fmt;
- clippy `-D warnings` with and without `--all-features`;
- 3,275 tests with the default features, and 3,452 with `--all-features`;
- doc `-D warnings`;
- `cargo deny check`, with nothing added, as D-S12-14 said;
- `cargo +1.85 check`, and `cargo +1.85 check -p fhy-core --features
  sympy,z3`.

The CI `rust` job installs sympy into its z3 venv, and exports
`PYO3_PYTHON`, `PYTHONPATH` and the libpython directory to every later
step. Its target list becomes `id_cap_decode it sympy_unavailable`, and its
package-contents pattern admits the prelude. `noxfile.py`'s `SOURCES` and
ty's `include` gain the prelude's directory, and a new `clippy.toml` lets
rustdoc write SymPy without backticks.

### S12.4 status: the binding

`fhy-core-py` enables `fhy-core/sympy`. The new class is
`rust/fhy-core-py/src/solver/sympy.rs`'s `_rs.SympySimplifier`, an
`extends = SimplifierBase` pyclass that is frozen and not subclassable.

- **Its methods:** `name` (`"sympy"`), `load()`, `simplify(expression)`,
  `lower`, `lift`, `simplify_object`, `substitute(sympy_expression,
  environment)`, `substitute_symbols` and `__reduce__`. Every method reads
  the registry snapshot, as the facade does.
- **The native path.** `build_simplifier` hands a `Solver` the class's core
  backend, so a solver holding one simplifies with no Python backend in
  between.
- **D-S12-11's mapping** is `sympy_error_to_py(error, wrap)`, and the facade
  uses it for a `SympyError` behind `SolveError::Backend`.
  - With `wrap`, as `simplify` and `substitute` use it, a lowering,
    substitution or lifting failure raises `PassExecutionError('pass "<the
    phase's pass>" failed in run_pass', pass_name=..., hook="run_pass")`
    with the mapped exception as its `__cause__`.
  - The screen's refusal, a bound constant and a missing SymPy are never
    wrapped, and neither is a `BaseException` that is not an `Exception`.
  - `lower`, `lift` and `substitute_symbols` raise the mapped exception
    itself, and the thin passes of S12.5 wrap it.
- **Messages.** A missing SymPy raises `SolverBackendUnavailableError` with
  the bridge's message and the `ImportError` as its cause. An `Implies` is
  logged at WARNING on `fhy_core.symbolic.expression.passes.sympy`, as the
  bridge logged it.
- **The stub** declares the class, and `tests/test_rs_stub.py` passes.
- **Nothing in Python uses the class yet.** `pytest` has 7,556 passed, and
  the Rust gate passes. `cargo test --workspace` now runs the SymPy
  stories: 3,420 tests.
