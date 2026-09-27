# Audit: the Rust port, slices S7–S18 (`rust/fhy-core`, `rust/fhy-core-py`)

## Scope

- **Slices covered:** S7 to S18 of `docs/design/python-switch.md`, the
  work ported since the first audit (`docs/audit/rust-workspace.md`, whose
  triage was implemented at `6fdbe68`).
- **Commit audited:** `a5d3e1a` on `dev-rust`, with a clean working tree.
- **Seven audit areas.** Each area has its own report and ID prefix:
  1. **Architecture and public API** (`ARCH-*`): layering, the public
     surface, semver hazards, features, duplication across slices, and docs.
  2. **Expression, tree, term and foreign** (`EXP-*`): nodes, literals, the
     wire format, display, patterns, the inliner and the evaluator.
  3. **Solver, passes, lattice and foundation modules** (`SOL-*`): the
     screens, the SMT lowering, the process, z3 and SymPy backends, passes,
     the lattice, identifiers and interning.
  4. **Types, param, constraint and symbol table** (`TYP-*`).
  5. **The PyO3 binding** (`PY-*`): `rust/fhy-core-py`, its stub, and the
     CONTRIBUTING rules that shape it.
  6. **Test coverage** (`TST-*`): line and region coverage from both suites,
     a reading of every uncovered region, and test quality.
  7. **Adversarial bug hunt** (`BUG-*`): 18 probe strategies against the
     installed extension and a scratch workspace.
- **Sources.** The seven reports are in `target/audit/*.md`. `target/` is
  git-ignored, so the reports and every probe path cited below
  (`target/audit/scratch-*`, `target/audit/coverage/`) exist only in this
  checkout.
- **Path conventions.** Source paths in findings are relative to
  `rust/fhy-core/src/`, except:
  - paths starting with `fhy-core-py/src/`, which are under `rust/`;
  - `tests/it/…`, `tests/id_cap_decode.rs` and `tests/golden/…`, which are
    under `rust/fhy-core/`;
  - `tests/test_*.py` and `src/fhy_core/…`, which belong to the Python
    package at the repository root.
- **Not re-reported:** the first audit's F-001 to F-040 and N-1 to N-4. N-2
  (registry growth on untrusted input) and N-3 (deep `Provenance`
  recursion) stay deferred.
- **Brief,** as in the first audit: check that the port is idiomatic Rust,
  and look for bugs and missing coverage. The user accepts behavior changes
  that make the Rust idiomatic, so each finding records whether it is
  **Python-driven** (inherited from the pre-port Python) and what behavior
  change its fix implies.
- **Verification for this report.**
  - Each High finding was spot-read at its cited location, and its probe
    was re-run at `a5d3e1a`: `scratch-architecture/probe/examples/dag_key.rs`
    (F2-001), `scratch-bugs/py/p25_idcap.py` (F2-002) and
    `scratch-binding/probe_gc_cycles2.py` (F2-003). All three reproduce.
  - These Medium and Low locations were also spot-read: F2-009
    (`expression/evaluate.rs:11`), F2-010 (`types/checking/error.rs:157-170`),
    F2-013 (`fhy-core-py/src/wire.rs:265`), F2-014 (`solver/process.rs:265-270`),
    F2-015 (`solver/smt/lower.rs:25-30`), F2-018 (`types/unify.rs:571-573`),
    F2-019 (`types/checking/body.rs:150-160`), F2-023
    (`fhy-core-py/src/constraint/value.rs:42-50`), F2-034
    (`expression/builtins.rs:578-590`) and F2-043 (`fhy-core-py/src/lib.rs:33`).

## Current state

### Gates

- **fmt:** clean (`cargo fmt --all --check`, run for this report).
- **clippy:** clean. That holds for `--workspace --all-targets --all-features
  -D warnings`, and for `-p fhy-core --all-targets` with no features and
  with each feature alone (ARCH, PY).
- **Stricter clippy** (`-W clippy::nursery -W clippy::pedantic`): 408
  warnings in the core and about 810 in the binding, nearly all `const fn`,
  `use_self`, `redundant_pub_crate` and `option_if_let_else`. The few
  meaningful ones are listed in `target/audit/architecture.md`. They are
  style work, not findings.
- **MSRV (1.85):** `cargo +1.85 check -p fhy-core --all-targets` passes with
  no features and with each feature alone. The test targets show 6
  `dead_code` warnings on 1.85 that CI's `--workspace --lib` job never sees
  (F2-009).
- **Docs:** `RUSTDOCFLAGS=-D warnings cargo doc -p fhy-core --no-deps`
  **fails** with the default features, and also with `z3` or `sympy`
  alone, on an unresolved link. It passes with `ndarray`, with
  `--all-features` and with `--workspace`, which is what CI runs (F2-009).
- **Unsafe:** none in either crate (`unsafe_code = "forbid"`).
- **Package:** `cargo package --list -p fhy-core` gives 266 files, exactly
  D-19's list.
- **Layering:** every non-test `crate::` edge respects CONTRIBUTING's ten
  layers.

### Tests

- **Rust, all features:** 4,208 pass, 0 fail, 2 ignored under
  `cargo llvm-cov`, which runs no doctests: 397 unit, 3,809 in `it`, 1 in
  `id_cap_decode` and 1 in `sympy_unavailable`. The EXP area's plain
  `cargo test --all-features` in `rust/fhy-core` passed 4,337 with 2
  ignored; the difference is presumably the doctests.
- **Rust, default features:** 4,013 pass.
- **Python suite,** under the instrumented extension: 8,222 passed, 2
  xfailed and 64 warnings, in 20 minutes with 16 xdist workers.
  - **The run stalled at 99%.** After 8,224 results, all 16 workers sat
    idle on futexes for more than 10 minutes, and `pytest-timeout` never
    fired. A SIGINT shut them down cleanly. The slice checklist records
    8,281 passing tests, so about 57 tests may not have run. This was seen
    only with an instrumented build under xdist. **It is unverified on
    normal builds and needs a check:** one normal `pytest` run to
    completion, and a second with `-n 16` (TST process notes).
  - **64 V1 `DeprecationWarning`s.** They all come from tests that read the
    V1 wire format on purpose: `tests/test_provenance.py`,
    `test_value_domain.py`, `types/test_serialization.py` and several
    `*_rust_binding.py` suites (`target/audit/coverage/pytest.log`).
    `pyproject.toml` has no `filterwarnings`, so they bury real warnings.
    These tests should assert the warning with `pytest.warns`, or filter it
    with a `filterwarnings` mark.

### Coverage

Measured with cargo-llvm-cov 0.9.1 on stable 1.98.1 (TST). Outputs are in
`target/audit/coverage/`.

| Run | Lines | Regions | Functions |
|---|---|---|---|
| `fhy-core`, all features | **93.68%** (18,513/19,763) | **91.50%** (28,759/31,431) | 92.55% (2,546/2,751) |
| `fhy-core`, default features | **94.36%** (16,620/17,614) | **93.07%** (25,456/27,352) | 93.49% (2,340/2,503) |
| `fhy-core`, all features, in-file `#[cfg(test)]` excluded (lcov basis) | 93.84% | – | – |
| `fhy-core`, Rust suite + Python suite (lcov basis) | 95.72% (18,337/19,157) | – | – |
| `fhy-core`, Python suite alone (lcov basis) | 82.65% | – | – |
| `fhy-core-py` (binding), Python suite | **90.61%** (19,019/20,989) | – | – |

| Module | Lines (all) | Line % all / default | Rust + Python line % |
|---|---|---|---|
| `param` | 2,308 | 90.4 / 90.4 | 95.2 |
| `solver` | 3,294 | 90.8 / 94.0 | 93.8 |
| `types` | 2,465 | 91.2 / 91.2 | 93.3 |
| `constraint` | 1,235 | 91.6 / 91.6 | 93.8 |
| `tree` | 187 | 94.1 / 94.1 | 93.9 |
| `expression` | 4,815 | 94.7 / 95.3 | 96.5 |
| `pass` | 1,720 | 95.0 / 95.0 | 96.4 |
| `interned`, `identifier`, `symbol_table`, `diagnostic`, `value_domain` | 2,442 | 96.1–99.8 | 96.8–99.8 |
| `described_tag`, `lattice`, `op_attribute`, `provenance`, `scope`, `stack`, `term` | 1,246 | 100.0 | 100.0 |

- **Missed lines, classified.** Of the lines the Rust suite misses, about
  45% are real behavior (F2-025 to F2-030 and F2-046), 25% error `Display`
  and `source()` bodies (F2-029), 15% defensive or unreachable arms
  (F2-047), and 15% trivial code.
- **The Python suite fills gaps.** It covers 286 of the 1,106 lines the
  Rust suite misses, mostly in `param`, `solver::sympy` and
  `types::checking`. 820 lines are reached by neither suite.
- **Least-covered files.** In the core: `param/error.rs` (48.6%) and
  `solver/sympy/error.rs` (59.3%). In the binding: `param/custom.rs`
  (33.9%) and `constraint/custom.rs` (46.3%).
- **Branch coverage was not measured.** `--branch` and `--doctests` need a
  nightly toolchain, and only stable and 1.85 are installed.

### Tools used

- **Rust:** stable 1.98.1 and 1.85 (MSRV) through rustup, with clippy,
  rustfmt, rustdoc and cargo-llvm-cov 0.9.1. The environment was
  `target/tooling/gate-env.sh`, with a separate `CARGO_TARGET_DIR` per area
  under `target/audit/`.
- **Solvers:** the z3-solver wheel's `z3 -in` (4.16) as `FHY_SMT_SOLVER`,
  and its libz3 for the `z3` feature. No strict SMT-LIB2 solver such as
  cvc5 was installed, so claims about strict solvers are plausible, not
  verified.
- **Python:** CPython 3.11 from the repo's `.venv`, with NumPy 2.4.6 and
  SymPy. The coverage run used an instrumented `maturin build` in a scratch
  venv. There was no free-threaded interpreter.
- **Probes:** six probe crates or workspace copies, 40 Python bug-hunt
  scripts, 15 binding scripts, and proptest oracles (brute-force intervals,
  set algebra, feasibility against z3, lattice laws, evaluator lane
  agreement). The real crates were never modified.
- **Not run:** cargo-deny, miri, mutation testing and semver checks.

### Housekeeping left by the audit

The coverage run installed **`maturin==1.15.0` into the repo's `.venv`**
by mistake. It is still there. To revert it, run
`target/tooling/pyenv/bin/uv pip uninstall --python .venv/bin/python maturin`.

## Overall assessment

The later slices kept the first audit's discipline. Almost every expression
walk is iterative and memoized by `NodeIdentity`. Errors are structured,
`#[non_exhaustive]` and one line. The core has no global state beyond
identity. The solver and param decisions were sound against brute force and
real z3, and the binding's lock and interrupt discipline held up under
probing. Most of what remains comes from slices written in parallel: each
solved a shared problem locally, and the few places that skipped the
crate's DAG or depth discipline are now the worst failures. Three findings
are High: an exponential ordering key that aborts on ordinary DAG
equations, one payload that makes every later identifier unreadable, and a
binding that no garbage collector can see into.

**Recurring root causes:**

- **Walks that ignore sharing.** The constraint key and the type checker
  (F2-001), error and `Display` text (F2-010), SymPy lifting and
  shape-variable substitution (F2-038), and the wire encoder's
  identity-only deduplication (F2-011).
- **Unbounded recursion outside `Expression`.** `Pattern`, constraint
  `Value` and the binding's dict and member readers (F2-013). F-003's fix
  covered only `Expression`.
- **Extension traits with inconsistent shapes and no error channel.** Five
  trait shapes, infallible hooks and no context (F2-004). The binding's
  thread-local side channels follow from them (F2-023, F2-031), and so do
  the non-reflexive defaults (F2-022).
- **Error messages that embed whole subtrees** (F2-010).
- **Code duplicated by the parallel slices.** Boxed-error aliases,
  contexts, constructor conventions (F2-007), exact arithmetic (F2-008),
  and binding boilerplate and guards (F2-031, F2-033).
- **Decoders that disagree with the constructors.** Payload ids (F2-002),
  symbol tables and assignments (F2-020), and non-canonical float text
  (F2-036).
- **Missing resource bounds at the process boundary.** Broadcast sizes
  (F2-012), solver timeouts (F2-014), and `Decimal` and big-int
  conversions (F2-045).
- **Test gaps where only the Python side exercises a path,** and tests
  that pass vacuously: custom hooks (F2-025), evaluator and checker shapes
  (F2-026), decision rules (F2-028), the body sweep (F2-019), and the
  solver properties (F2-027).

---

## Findings

### Summary

Triage groups: **a** = obvious fix, **b** = needs a decision, **c** =
test-only, **d** = deferrable cleanup. "Py" = Python-driven.

| ID | Title | Sev | Category | Sources | Py | Triage |
|---|---|---|---|---|---|---|
| F2-001 | Constraint ordering keys and the type checker are exponential on shared DAGs | High | perf | ARCH-001, TYP-004 | partly | a |
| F2-002 | One payload id just below the cap makes every later fresh identifier unreadable | High | bug | SOL-003, BUG-001 | yes | a |
| F2-003 | No binding class takes part in cyclic GC; cycles through Rust-held `Py<>` leak | High | bug | PY-001 | no | a |
| F2-004 | Extension-point traits: five shapes, no error channel, no context | Medium | api | ARCH-004, ARCH-005, TYP-013, TYP-007 | partly | b |
| F2-005 | pyo3 and ndarray in the public API; exhaustive public data; `Simplifier` has no limits | Medium | api | ARCH-008, SOL-011 | no | b |
| F2-006 | `ParamError` is one 30-variant enum for every `param` operation | Medium | api | ARCH-006 | no | b |
| F2-007 | API conventions diverge across the parallel slices | Medium | idiom | ARCH-003, TYP-013, ARCH-010, ARCH-011, ARCH-012, ARCH-014 | mixed | d |
| F2-008 | Exact-number arithmetic is re-implemented in four modules | Medium | idiom | ARCH-007, BUG (plausible) | no | d |
| F2-009 | The default-feature doc build fails; docs.rs will hide gated APIs; CI checks less than it says | Medium | doc | ARCH-002, ARCH-016 | no | a |
| F2-010 | Error messages and `Display` write every occurrence of a shared subtree | Medium | bug | EXP-001, SOL-008, TYP-004, EXP-003 | no | a |
| F2-011 | Serialization depends on sharing: equal expressions encode to different bytes | Medium | bug | EXP-002 | no | b |
| F2-012 | The NumPy evaluator aborts the interpreter or panics on large broadcasts | Medium | bug | BUG-002 | no | a |
| F2-013 | Unbounded recursion outside `Expression` aborts the process | Medium | bug | EXP-006, TYP-011, BUG-003 | no | a |
| F2-014 | The process backend's timeout does not bound the call | Medium | bug | SOL-002, BUG-005, SOL-004, SOL-010, BUG (plausible) | no | a |
| F2-015 | Control characters in a name hint reach the solver: a NUL panics or breaks the script | Medium | bug | SOL-001, BUG-006 | no | a |
| F2-016 | SymPy simplification turns `y / x` into `y * x ** -1`, which the evaluator refuses | Medium | bug | BUG-004 | partly | a |
| F2-017 | The checker types `-(5)` as `uint8` and refuses `-(128)` as `int8` | Medium | bug | TYP-001 | yes | a |
| F2-018 | Unification silently uses the unsubstituted expression when a substitution is refused | Medium | bug | TYP-002 | no | b |
| F2-019 | `check_all_function_bodies` never checks a composed built-in; its test passes vacuously | Medium | bug | TYP-003, TST-001 | no | a |
| F2-020 | Decoders disagree with the checked constructors: valid tables fail, invalid assignments pass | Medium | bug | TYP-005, TYP-009 | partly | a |
| F2-021 | Domain-level procedures ignore the domain's own restriction | Medium | api | TYP-006 | yes | b |
| F2-022 | Extension type defaults are not reflexive; structural equivalence is asymmetric | Medium | bug | TYP-007, ARCH-013 | no | a |
| F2-023 | The constraint error slot keeps the first exception; a later `KeyboardInterrupt` is lost | Medium | bug | PY-002 | no | a |
| F2-024 | `PartiallyOrderedSet` and `Lattice` no longer pickle, copy or deep-copy | Medium | bug | PY-003, BUG-007 | no | a |
| F2-025 | Custom domain and custom constraint hooks are mostly untested in either language | Medium | test-gap | TST-002 | no | c |
| F2-026 | Evaluator and checker: shapes specified only by Python; IEEE and array edges untested | Medium | test-gap | TST-003, TST-004, TST-005, EXP-008 | no | c |
| F2-027 | Solver properties pass vacuously on `Unknown` and draw only integer predicates | Medium | test-gap | SOL-005, TST-006 | no | c |
| F2-028 | Param and constraint decision rules are tested only from Python, or not at all | Medium | test-gap | TST-007, TST-008, TYP-014 | no | c |
| F2-029 | Error text and source chains untested; loose and weak assertions; small untested behaviors | Medium | test-gap | TST-009, TST-011, TST-014 | no | c |
| F2-030 | Binding and interface-suite gaps; stub drift the stub test cannot see | Medium | test-gap | TST-012, PY-008 | no | c |
| F2-031 | Three of the binding's six thread-local stacks have no unwind guard | Low | idiom | ARCH-009, PY-007 | no | a |
| F2-032 | `constraint`/`param` values lack `PartialEq`/`Eq`/`Hash`/`Display`; template widths compare as a list | Low | api | ARCH-013, TYP-013 | mixed | d |
| F2-033 | Binding boilerplate duplicated: imports, exceptions, frozen protocol, seeds, error conversion | Low | idiom | ARCH-015, PY-010 | partly | d |
| F2-034 | Composed `max`/`min`/`relu`/`clamp`/`abs` give order-dependent NaN results | Low | bug | EXP-004 | yes | b |
| F2-035 | `substitute_avoiding_capture` renames needlessly, and renames a repeated binder twice | Low | bug | EXP-005 | no | a |
| F2-036 | Float literal text: lax decoding, and 300-character encodings of extreme magnitudes | Low | idiom | EXP-007 | no | b |
| F2-037 | `Children` has no size hint; walks re-count children; unary `+` copies lanes | Low | perf | EXP-009 | no | d |
| F2-038 | Three more super-linear paths: SymPy lifting, shape substitution, permutation enumeration | Low | perf | SOL-006, TYP-008, TYP-012, BUG (plausible) | partly | a |
| F2-039 | Any module at `sys.modules["_fhy_core_sympy"]` is trusted as the prelude | Low | bug | SOL-007 | no | a |
| F2-040 | The mixed int/real equality hazard refuses questions the crate can decide | Low | doc | SOL-009 | yes | b |
| F2-041 | `ValidatorRecord::diagnostics_in` panics on, or silently misreads, another report | Low | api | SOL-012 | no | a |
| F2-042 | Colliding opaque keys: equal ordering keys without equivalence | Low | bug | TYP-010 | yes | a |
| F2-043 | Binding threading: implicit free-threading, and NumPy inputs read in place while detached | Low | soundness | PY-006, PY-004, BUG (plausible) | no | b |
| F2-044 | Python code runs under a `PyRef`/`PyRefMut` borrow; re-entrant callbacks fail | Low | bug | PY-005 | no | a |
| F2-045 | Big Python numbers enter the core through decimal text | Low | bug | PY-009, BUG-009, BUG-008 | no | a |
| F2-046 | Serde round trips are example-only for params, types and symbol tables | Low | test-gap | TST-010, TYP-014 | no | c |
| F2-047 | Dead, unreachable and trivial code: make the guarantees types | Low | idiom | TST-013, TST-003 | no | d |

**Totals:** 3 High, 27 Medium, 17 Low (47 findings from 84 source
findings). "BUG (plausible)" marks an item from `bugs.md`'s "Plausible, not
reproduced" list.

**Severity adjustments.** Each finding takes the highest severity a source
gave, with two exceptions:
- **F2-019** (TST-001 said High) is **Medium**. No current output is wrong,
  since all 16 composed bodies check when called directly. The harm is a
  dead guard and a false contract.
- **F2-025** (TST-002 said High) is **Medium**. It is a test gap with no
  defect found on those paths, and the first audit rated comparable test
  gaps Medium.

---

### High

#### F2-001: Constraint ordering keys and the type checker are exponential on shared DAGs

- **Severity:** High · **Category:** perf (the key also aborts on memory) ·
  **Confidence:** High (verified; re-run for this report) ·
  **Python-driven:** partly. Python rendered keys the same way. F-002 made
  `==`, `Hash` and `free_identifiers` DAG-aware, but not these two walks.
- **Sources:** ARCH-001, TYP-004 (also TYP-014 item 6, the missing DAG-cost
  story)
- **Location:**
  - `constraint/key.rs:19-21` (`equation_key`) and `:61-97`
    (`write_expression_key`, pre-order with no visited set);
  - `constraint/system.rs:26-37` (`ConstraintSystem::new` keys and sorts
    every member);
  - `types/checking/checker.rs:480-700` (the work list pushes each child on
    every visit, with no `NodeIdentity` memo);
  - `fhy-core-py/src/constraint/system.rs:261`.

**Issue:**
- **The key is a tree print.** It renders every occurrence of a shared
  subtree. Every `ConstraintSystem::new` computes it eagerly for every
  member, and so does every `Param::new` and `with_constraint(s)` (through
  the system), and the Python `ConstraintSystem` constructor.
- **The checker does the same.** It re-infers a shared subtree once per
  path.
- **The exception to the rule.** Every other walk the later slices wrote is
  memoized: `solver/screen.rs:185`, `solver/smt/lower.rs:147`,
  `evaluate/walk.rs:844`, `registry/inline.rs` and `wire.rs`.
- **An unstable format.** The key embeds `call[{:?}]`, the `Debug` text of a
  callee name, which is not a stable contract.

**Evidence:**
- `target/audit/scratch-architecture/probe/examples/dag_key.rs` builds
  `e_{k+1} = e_k + e_k` and `ConstraintSystem::new([e_k == 0])`. Re-run for
  this report, release build:

  | depth | hash | key | `ConstraintSystem::new` |
  |---|---|---|---|
  | 16 | 14 µs | 2.2 MB | 10 ms |
  | 20 | 3 µs | 34.6 MB | 139 ms |
  | 22 | 5 µs | 138 MB | 558 ms |

  The key grows 4× per two levels. At depth 30 it would be about 35 GB, and
  the process aborts on allocation.
- The TYP probe (`scratch-types/fhy-core/tests/it/audit_probes.rs`) shows
  `synthesize` at 8, 37 and 153 ms for depths 14, 18 and 20, while
  `free_identifiers` stays in microseconds.

**Why it matters:** DAGs are the normal output of `substitute` and
`rewrite_tree`. One such equation in a param or a system hangs or aborts
the whole compile.

**Fix:**
- **The key.** Render it over a post-order node table like the one the wire
  format builds: each distinct node once, children as indices. That keeps
  "equal key exactly when structurally equal". Better still, replace the
  string key with a structural `Ord` on `Constraint` that uses the cached
  hash and a memo of visited pairs, which also serves F2-032.
- **The checker.** Memoize `Infer` results per `NodeIdentity` when there is
  no expected type (most nodes), and per `(node, expected)` otherwise.
- **Regression tests.** Add depth-40 to depth-64 doubling-DAG stories for a
  system and for the checker.

**Behavior change:**
- The key text changes. Python's `build_ordering_key` returns it, so the
  change is Python-visible.
- The canonical member order of systems may change, and with it their V2
  serialization order and the golden corpus. D-S13-6 fixes only the key's
  prefix.

#### F2-002: One payload id just below the cap makes every later fresh identifier unreadable

- **Severity:** High · **Category:** bug (robustness on untrusted input) ·
  **Confidence:** High (verified; re-run for this report) ·
  **Python-driven:** yes. The concept is defined in both languages, and
  Python has the same cap (D-3), so both backends break the same way.
- **Sources:** SOL-003, BUG-001
- **Location:**
  - `identifier.rs:17-22` and `:46-50` (the cap, documented with "Fresh ids
    may exceed it");
  - `:99-111` (`PayloadId::new` and `successor`);
  - `:139-155` (`new`/`try_new` then issue ids at or above `ID_CAP`);
  - `:258-262` (`advance_past`);
  - `src/fhy_core/identifier.py` (the same field check);
  - `tests/id_cap_decode.rs:3-7`, which records the effect only in a
    comment.

**Issue:** Decoding `{"id": 2^63 - 1, ...}` is accepted, and it moves the
counter to `ID_CAP`. Every fresh id from then on is at least `2^63`, which
every reader refuses: Rust's `PayloadId::new` and Python's field check.
The F-001 fix moved the failure from "every later `Identifier::new`
panics" to "every later identifier is unreadable". `to_json` still
succeeds, so the failure shows only when the text is read back: after a
pickle, a `copy`, a multiprocessing hop or a cache load.

**Evidence:** `target/audit/scratch-bugs/py/p25_idcap.py`, re-run: after the
payload, a fresh `k` has id `9223372036854775811`. `Identifier.from_json`
and `pickle.loads` raise `DeserializationValueError` ("Expected a
non-negative integer below 2**63"), and `Expression.from_json` raises
`OverflowError`. The Rust side: `scratch-solver/ws/rust/fhy-core/tests/audit_id_cap_probe.rs`.

**Fix:**
- Separate the two bounds. A payload id that *advances* the counter must be
  below a lower cap, such as `2^62`. Decoding accepts any id below
  `ID_CAP`, so `2^62` fresh allocations stay readable after a worst-case
  payload.
- Make `try_allocate_id` fail once the counter reaches the readable limit,
  rather than issue ids that cannot be read.
- Mirror the change in `identifier.py`.
- Add a story: restore the highest accepted payload id, create one
  identifier, and round-trip it.
- **Related (SOL-003).** `try_restore` accepts reserved ids that no shipped
  tag uses, such as 100. A tag shipped later at that id would equal old
  payload identifiers. Consider rejecting unassigned reserved ids.
- **An alternative that was not chosen.** Accept, without advancing the
  counter, any id this process has already issued (`id < NEXT_ID`). That
  fixes only round trips within one process.

**Behavior change:** payload ids in `[2^62, 2^63)` are rejected, in Rust
and Python alike.

#### F2-003: No binding class takes part in cyclic GC; cycles through Rust-held `Py<>` leak

- **Severity:** High · **Category:** bug (memory leak) · **Confidence:**
  High (verified; re-run for this report) · **Python-driven:** no. The
  pure-Python classes these replaced were all collectable.
- **Sources:** PY-001
- **Location:**
  - No `__traverse__` or `__clear__` exists anywhere in
    `rust/fhy-core-py/src` (grep: 0 hits).
  - Classes that hold user objects, all under `fhy-core-py/src/`:
    - `pass/manager.rs:57-61` (`PyFixpointPassGroup.passes`)
      and `:217-220` (`PyPassManager.items`, `verifier`);
    - `pass/validation.rs:265-267`;
    - `expression/pattern/rules.rs:98-125`, `:191-203`;
    - `lattice.rs:22-27`, `:182-186`, `:312-315`;
    - `symbol_table/table.rs:178-183`;
    - `solver/facade.rs:139-142`;
    - `param/parameter.rs:227-230`;
    - `expression/registry/entries.rs`.

**Issue:** None of the 91 exported classes has `Py_TPFLAGS_HAVE_GC`. Every
Rust field that holds a user object is a strong reference the collector
cannot see, so any cycle through one leaks for the life of the process.
The cycles are ordinary Python:
- a pass that keeps `self.manager`;
- a rewrite callback that closes over its own rule list;
- a simplifier that falls back to the solver that holds it;
- a lattice element that points back at its container.

**Evidence:** `target/audit/scratch-binding/probe_gc_cycles.py` and
`probe_gc_cycles2.py`, re-run. A plain Python cycle, the control, is
collected. The `RewriteRule`, `PartiallyOrderedSet`, `Solver` and
`PassManager` cycles all leak, and 91 of 91 classes lack the GC flag.

**Why it matters:** In a compile server, notebook or test session, each
leaked pipeline keeps its passes, diagnostics, IR and analysis results
alive, and nothing reports it.

**Fix:**
- Give each class that holds user objects `__traverse__`, and `__clear__`
  where it owns the references. PyO3 0.29 supports both on frozen classes.
- Where the references sit behind a `Mutex`, traverse through `try_lock`.
  A traversal must never block.
- **Hidden references.** `RewriteRule`, the `Solver` and the Python
  adapters also keep `Py` clones inside Rust closures and core trait
  objects, where no traversal can reach them. Keep each Python object in
  one visible slot, such as `Arc<Mutex<Option<Py<PyAny>>>>`, which the
  pyclass traverses and clears and the closure borrows from.
- Add a regression test per class family, modeled on the probe.

**Behavior change:** none, except that unreachable cycles are freed.

---

### Medium

#### F2-004: Extension-point traits: five shapes, no error channel, no context

- **Severity:** Medium · **Category:** api · **Confidence:** High for the
  design; the fix's cost is plausible · **Python-driven:** partly. The
  `Option<Result>` hooks emulate `singledispatch`; the rest comes from the
  parallel slices.
- **Sources:** ARCH-004, ARCH-005, TYP-013 (context-less custom hooks),
  TYP-007 (the `Option<Result>` protocol)
- **Location:**
  - the traits: `constraint/value.rs:34-84` (`OpaqueValue`, with `is_equal`
    at `:47` and `ordering_key` at `:56`) and `:87-114` (`Opaque`);
    `types/extension.rs:26-164`, with the `Option<Result>` hooks at
    `:238-267`; `constraint/custom.rs:27-75`, with "an implementation that
    fails reports no identifier" at `:28-30`; `param/custom.rs:18-130`;
    `term/binder.rs`;
  - the extra channel: `constraint/binding.rs:47-52`, `:98-110`
    (`Bindings::source`);
  - the binding's side channels, under `fhy-core-py/src/`:
    `constraint/value.rs:36-80` (`PENDING_ERROR`), `types/adapter.rs:60-112`,
    `term/adapter.rs:1-20` and `wire.rs`.

**Issue:**
- **Five shapes.** S10, S11, S13 and S16 each designed their extension
  point separately:
  - the handle is an `Opaque` newtype in one trait and a bare `Arc<dyn _>`
    in the others;
  - the equality hook has four names;
  - one trait has a hash hook and the rest have none;
  - `type_name` is absent from `CustomConstraint` and `CustomDomain`, so
    their `NoWireForm` error says only "custom constraint";
  - every implementor hand-writes `as_any`.

  `Bindings::source` is a sixth, type-erased channel.
- **No error channel.** `is_equal`, `ordering_key`, `free_identifiers`,
  `is_structurally_equivalent` and the term hooks are infallible.
  - **In Rust,** a failing hook answers `false`, an empty set or an empty
    key, with no error. A failing `CustomConstraint` "reports no
    identifier", so the constraint can look closed when it is not.
  - **In the binding,** each of three subsystems stashes the `PyErr` in
    thread-local state and re-raises it after the core returns. F2-023's
    bug lives in one of these side channels.
- **No context.** `CustomDomain::has_feasible_value`, `feasibility_subset`,
  `union` and `intersection`, and `CustomConstraint::evaluate`, receive no
  `ParamContext` or `ConstraintContext`. A Rust implementor cannot ask the
  solver, report events or reuse the core procedures.
- **Emulated dispatch.** The `types` hooks return `Option<Result<T, E>>`,
  where `None` means "run the core default".

**Fix:** one breaking batch, before the crates.io release. See triage (b).
- Add a `foreign::ForeignPart` supertrait with `type_name` and a provided
  `to_foreign`. Add a blanket `AsAny` helper, or use trait upcasting once
  the MSRV reaches 1.86. Make every extension trait `: ForeignPart`, pick
  one handle convention, and pick one equality-hook name.
- Hooks that can fail return `Result<_, BoxError>`, propagated through each
  module's error type (`ConstraintError::Custom`, `UnificationError`, …).
  `eq_extension` and `hash_extension`, which back `PartialEq` and `Hash`,
  stay infallible, and the binding keeps a side channel only for those two.
- Pass the context to the custom hooks. The binding's adapters take this
  additively.
- Replace `Option<Result>` with provided methods whose bodies call public
  default functions, such as `types::default_bind_template(..)`.
- Document `Bindings::source` as binding-only, or replace it with a typed
  parameter.

**Behavior change:**
- Implementors add `type_name` to the two `Custom*` traits and drop
  `as_any`.
- A failing hook becomes an `Err` from the core operation, not a silent
  `false` or empty set.
- `NoWireForm` names the real type, and the binding's side channels shrink
  to the equality and hash case.

#### F2-005: pyo3 and ndarray in the public API; exhaustive public data; `Simplifier` has no limits

- **Severity:** Medium · **Category:** api (semver) · **Confidence:** High
  for the exposure; Medium for its cost · **Python-driven:** no
- **Sources:** ARCH-008, SOL-011
- **Location:**
  - `solver/sympy.rs:160-300`: `lower`, `lift`, `simplify_object`,
    `substitute` and `substitute_symbols` take and return `Python<'py>` and
    `Bound<'py, PyAny>`;
  - `solver/sympy/error.rs:20-23`, `:135` (`PyErr` inside public variants);
  - `rust/fhy-core/Cargo.toml` (`sympy = ["dep:pyo3"]`);
  - `expression/evaluate/array.rs:27, 65, 158` (`ndarray` 0.17 types);
  - `expression.rs:58` (`pub use num_bigint::BigInt`);
  - `param/domain.rs:83-93` (`IntervalProfile`: exhaustive, with four
    public `bool` fields); `constraint/value.rs:127` (`Value`, exhaustive);
  - `solver/backend.rs:78-98` (`Simplifier::simplify` takes no limits).

**Issue:**
- **pyo3.** Under the `sympy` feature, pyo3 0.29 is a public dependency of
  a crate headed for crates.io, which D-S12-2 acknowledges. pyo3 makes a
  breaking release about every quarter, and `links = "python"` allows one
  pyo3 per build. So:
  - every pyo3 upgrade in the binding forces a breaking `fhy-core` release;
  - a downstream crate on any other pyo3 cannot enable `sympy` at all.

  `SympySimplifier` is a leaf: nothing in the core calls it, and the
  binding is its only user.
- **ndarray and num-bigint.** Their exposure is milder: they rarely break,
  and the README documents both.
- **Exhaustive public data.** `IntervalProfile` and `Value` freeze their
  shapes.
- **No limits.** `sympy.simplify` can run without bound while it holds the
  interpreter. Unlike `SmtSolver`, a `Simplifier` receives no bound.

**Fix:** where the SymPy backend lives is triage (b). Independently of that:
- make `IntervalProfile` `#[non_exhaustive]`, with a constructor or a
  builder, and make `Value` `#[non_exhaustive]`;
- add a limits field to `SimplifyContext`, which exists so it "can carry
  more without changing the trait".

**Behavior change:** the `fhy_core::solver::SympySimplifier`/`Sympy*` paths
move or become hidden, and matching on `Value` needs a wildcard arm.

#### F2-006: `ParamError` is one 30-variant enum for every `param` operation

- **Severity:** Medium · **Category:** api (errors) · **Confidence:** High ·
  **Python-driven:** no
- **Sources:** ARCH-006
- **Location:** `param/error.rs:38-150`. Returned by `domain.rs:297, 326,
  362`, `parameter.rs:76-540`, `interval.rs:211`, `decide.rs:74-428` and
  `assignment.rs:31-65`.

**Issue:** One enum covers domain construction, param construction, set
algebra, assignment, interval arithmetic and two wrappers. The largest
error elsewhere has 18 variants, and most have 2 to 8. `OrdinalDomain::new`
can return 5 of the 30. This breaks CONTRIBUTING's "one type per family of
related operations": callers cannot match exhaustively on what an
operation actually returns, and every new param feature grows the enum.
The file is also the least-covered in the core (48.6%; F2-029).

**Fix:** split along the families: `DomainError`, `ParamBuildError`,
`AssignmentError`, `IntervalError`, and a `ParamError` for the questions
that wraps `ConstraintError` and `BoxError`. See triage (b).

**Behavior change:** error types change per operation. The Python exception
classes need not change.

#### F2-007: API conventions diverge across the parallel slices

- **Severity:** Medium · **Category:** idiom / api · **Confidence:** High ·
  **Python-driven:** mixed. `Ok(None)` is Python's `NotImplemented`; the
  rest comes from the parallel slices.
- **Sources:** ARCH-003, TYP-013 (the aliases, and `Ok(None)` against
  `Err`), ARCH-010, ARCH-011, ARCH-012, ARCH-014

**Issues and locations:**
- **Five boxed-error aliases (ARCH-003, TYP-013).** They are
  `CallbackError` (`expression/pattern/matching.rs:49`), `OpaqueError`
  (`constraint/value.rs:26`), `CustomError` (`constraint/custom.rs:20`),
  `BackendError` (`solver/backend.rs:19`) and `PassFailure`
  (`pass/compiler_pass.rs:17`). The same type is spelled out inline at
  `foreign.rs:127, 162`.
  - `types` (`types/error.rs:7`, `checking/checker.rs:7`) and `evaluate`
    (`evaluate.rs:86`, `evaluate/error.rs:12`, `evaluate/array.rs:11`)
    import the *pattern* module's alias, which is false coupling.
  - Fix: one `BoxError` in a foundation module.
- **Lookup traits lack `Sync` (ARCH-010).** `Environment`, `SymbolTypes`
  and `SortLookup` (`expression/screen.rs:26, 53, 538`) have no `Sync`
  bound. So `SimplifyContext` (`solver/backend.rs:134`), `QueryContext`
  (`solver.rs:207`), `BooleanScreen` (`screen.rs:239`) and `TypeChecker`
  (`checker.rs:153`) are `!Send + !Sync`, and one `QueryContext` cannot be
  shared across a `rayon` fan-out of `Solver::ask`.
  - Fix: add a `Sync` supertrait to the three core-implemented traits;
    document `IdentifierTypes` and `CallTargets`; add `assert_send_sync`
    checks.
- **Duplicate context, observer and event types (ARCH-011).**
  `constraint/context.rs:17-190` has `Event` and `Observer`;
  `param/context.rs:43-270` has `ParamEvent` and `ParamObserver`, with the
  same fields and methods. `ParamContext::constraint_context` rebuilds a
  `ConstraintContext` on every call.
  - Fix: name them symmetrically, and have `ParamContext` hold a
    `ConstraintContext`.
- **Constructors and conversions (ARCH-012).**
  - Six types have `try_new` with no `new` (`expression/callee.rs:126`,
    `provenance.rs:56, 684`, `term/renaming.rs:101`,
    `expression/registry/definition.rs:70, 228`), while `param` uses
    `new -> Result` (`param/domain.rs:297, 326, 362`,
    `assignment.rs:31`).
  - `Member::try_from_value` and `Value::from_literal` should be the std
    `TryFrom` and `From`.
  - Anonymous `bool` arguments: `IntegerDomain::new(bool, bool)`,
    `IntervalIntegerDomain::new(bool, bool, bool)` and
    `check_bounds_are_ordered(.., bool, bool)`.
  - `ParamAssignment::new_unchecked` is a safe `fn`, where the `_unchecked`
    suffix conventionally marks an `unsafe` one.
- **One condition, two signals (TYP-013).**
  `Param::checked_add`/`sub`/`mul`/`reverse_sub` return `Ok(None)` when
  neither side is an interval operand, and `checked_neg` returns
  `Err(NotAnIntervalOperand)` (`param/parameter.rs:437`, `:524`). `Err` is
  the idiomatic choice, and the binding maps it to `NotImplemented`.
- **A cross-module `FromStr` error (ARCH-014).**
  `types/core_data_type.rs:441`, `types/qualifier.rs:53` and
  `symbol_table/frame.rs:236` return `expression::UnknownNameError`
  (`expression/operation.rs:74-130`). Six name enums have no `FromStr`:
  `DiagnosticLevel`, `PassHook`, `QueryKind`, `Logic`, `SympyPhase` and
  `DomainKind`.
  - Fix: move the macro and the error to layer 1, and apply the macro to
    every enum with a stable name text.

**Fix:** batch everything into the pre-release API cleanup.

**Behavior change:** renames and signatures only. Closures used as
`SymbolTypes` must be `Sync`, and `checked_*` returns `Err` where it now
returns `Ok(None)`.

#### F2-008: Exact-number arithmetic is re-implemented in four modules

- **Severity:** Medium · **Category:** idiom (duplication) ·
  **Confidence:** High for the duplication; Medium that it diverges ·
  **Python-driven:** no
- **Sources:** ARCH-007, and `bugs.md`'s plausible "Truncated decimal
  exponents in the bound order"
- **Location:**
  - **IEEE-754 decomposition, three copies:**
    `expression/literal/decimal.rs:19-28`, `param/interval.rs:115-137` and
    `solver/smt/lower.rs:48-75`.
  - **Decimal to rational, three copies:** `param/interval.rs:171-181`,
    `solver/sympy/lower.rs:56-66` and `expression/literal/decimal.rs:155-165`.
  - **Exact int/float ordering, two copies:** `param/value.rs:40-58` and
    `param/interval.rs:139-141`.
  - **Rational to decimal:** `solver/sympy/lift.rs:423-445`.

**Issue:** The copies already disagree on their fallback for out-of-range
conversions: `unwrap_or(0)`, `unwrap_or(usize::MAX)` and `expect`. At
`interval.rs:171`, `u32::try_from(exponent.unsigned_abs()).unwrap_or(0)`
makes a decimal whose exponent exceeds `2^32` in magnitude compare as its
bare coefficient, so `check_bounds_are_ordered` would answer wrongly.
- **Not reachable today.** The `Decimal` grammar refuses such exponents at
  parse (probed with `"1e4294967296"`), and from Python F2-045 hangs first.
- **Why it matters.** The solver lowering, the param bounds, literal
  equality and the SymPy lowering must agree on what a literal denotes.

**Fix:** add one crate-private exact-arithmetic module beside `literal`,
holding `Rational { numerator: BigInt, denominator: BigInt }`,
`Rational::of_f64`, `Decimal::to_rational` and `LiteralValue::exact_cmp`.
Have `param`, `solver::smt` and `solver::sympy` call it.

**Behavior change:** none intended.

#### F2-009: The default-feature doc build fails; docs.rs will hide gated APIs; CI checks less than it says

- **Severity:** Medium · **Category:** doc / CI · **Confidence:** High ·
  **Python-driven:** no
- **Sources:** ARCH-002, ARCH-016
- **Location:**
  - `expression/evaluate.rs:11`, the link to `Prepared::evaluate_array`,
    which exists only under `ndarray`;
  - `rust/fhy-core/Cargo.toml`: no `[package.metadata.docs.rs]`; the
    `description` at `:3`; `repository` at `:13` ends in `.git`;
  - `.github/workflows/python-package.yml:97` (clippy, `--all-features`
    only), `:126-132` (docs), `:227` (MSRV, `--workspace --lib`);
  - `lib.rs:1-4` and `:82-83`; the README.

**Issue:**
- **The broken link.** The default-feature `cargo doc -p fhy-core` fails
  under `-D warnings`, and so do `z3` and `sympy` alone.
- **CI does not check what it claims.** The docs step's comment reads "The
  default features, as docs.rs builds them", but the step runs `--workspace`,
  where resolver 2 unifies the binding's features into the core. The
  clippy step never builds features alone, and the MSRV job never builds
  `fhy-core` alone. On 1.85 the test targets show 6 `dead_code` warnings
  that the job never sees (`tests/it/support/expression.rs:204, 232, 257`,
  `tests/it/expression/vocabulary_stories.rs:31, 50`,
  `src/test_support.rs:11`).
- **docs.rs will show none of the gated API.** Without docs.rs metadata,
  these items won't appear, though the README points to them:
  - `Z3Solver` and `Z3TermError`;
  - `SympySimplifier` and its four errors;
  - the `ndarray` evaluation API.

  No item carries `doc(cfg)`.
- **Drift.**
  - The `lib.rs` summary omits the solver, constraints, params, types,
    symbol table, lattice and stack/scope.
  - `lib.rs:82` claims that every `Serialize` type is also `Deserialize`.
    `DescribedTag<K>` and `ValueDomain` are `Serialize`-only by design
    (F-020).
  - The Cargo `description` omits the symbol table, stack and scope.

**Fix:**
- Add docs.rs metadata with `features = ["ndarray", "sympy"]` and `--cfg
  docsrs`. Add `z3` too if docs.rs can build `z3-sys`, which is unverified.
- Mark each gated item with `cfg_attr(docsrs, doc(cfg(..)))`.
- Write the link as code text, or gate that doc line.
- Give CI per-crate steps, each about 15 s warm: a default-feature
  `cargo doc -p fhy-core`, and per feature `cargo clippy -p fhy-core
  --all-targets` and `cargo +1.85 check -p fhy-core`.
- Fix the summary, the serialization claim ("…or its `Canonical<T>`
  does"), the description and the repository URL.

**Behavior change:** none.

#### F2-010: Error messages and `Display` write every occurrence of a shared subtree

- **Severity:** Medium · **Category:** bug · **Confidence:** High
  (verified) · **Python-driven:** no
- **Sources:** EXP-001, SOL-008, TYP-004 (`TypeCheckError::Rule`), EXP-003
- **Location:**
  - `expression/evaluate/error.rs:222-230`: the `Display` arms of
    `BooleanArithmetic`, `NumberAsBoolean`, `MixedBranches` and `Lane`
    write `{node}` (variants at `:141`, `:146`, `:150`, `:164`);
  - `solver/error.rs:174, 196, 199, 203`: the `LoweringError` variants
    `NonFiniteLiteral`, `Call`, `SortMismatch` and `UnsupportedPower`;
  - `types/checking/error.rs:157-170`: `TypeCheckError::Rule` writes both
    `root` and `at`;
  - `expression/wire.rs:215-331` (the decoder shares every
    multiply-referenced node, with no limit), `display.rs:413-416` and
    `:444-446` (display writes per occurrence, as documented), and
    `tree/walk.rs:93`.

**Issue:**
- **The messages.** These errors embed the failing node's full `Display`,
  which is a tree print. On a DAG the message never ends, although the
  operation that failed ran in linear time. On a tree it holds the whole
  subtree. The binding raises the same text as the Python message.
  `NonBooleanLogicalOperandError` already avoids this: it writes no
  expression (`expression/error.rs:162`).
- **Decoded DAGs need no builder.** The flat wire table lets a payload of n
  nodes describe `2^n` occurrences. Any `from_json` or
  `deserialize_from_dict` of a few kilobytes can yield a DAG that
  `Display`, `display()`, `walk_tree` and every visitor pass never finish.
  `print(expr)`, a logging call, or one of these error texts then hangs.
- **Lane errors are anonymous.** A `Lane` failure does not say which lane
  failed.

**Evidence:**
- `target/audit/scratch-expression/probes/tests/probes.rs`:
  - `p1`: a `Lane` error's `Display` on a 63-level doubling DAG is still
    writing after 10 MB. On a 100,000-term tree it is a 600,045-byte line.
  - `p3`: a 3,155-byte payload of 61 nodes decodes at once. Its
    `to_string()` exceeds 10 MB, and `walk_tree` is still running after
    10^7 visits. `Debug` stays at 7,595 bytes.
- SOL probe J (`scratch-solver/ws/rust/fhy-core/tests/audit_solver_probes.rs`):
  lowering fails in 0.1 ms, but `to_string()` takes 8 ms (24.6 KB) at
  depth 12, 55 ms (393 KB) at depth 16, and 700 ms (6.3 MB) at depth 20.

**Fix:**
- Write the node through a bounded rendering, such as the existing
  1,000-node `Debug` budget (`display.rs:338-347`), or name only its
  operation ("a boolean and a number meet in an addition"). Keep the node
  in the error's field.
- Give `Lane` the flat C-order index of the first failed lane
  (`lane: Option<usize>`).
- Add `Expression::occurrence_count() -> u64`, saturating and linear,
  documented as the guard for per-occurrence work. Have the binding's
  `__str__` and `__repr__` fall back to the bounded form above a threshold.
- Optionally, add a node budget to `FormatOptions`, a display style that
  writes a shared subtree once, and a decoding entry point with an
  occurrence limit for untrusted input.

**Behavior change:** long error texts are truncated or reworded, and the
binding's messages change with them. Tests that pin full messages, in
`evaluate_stories.rs` and the Python suite, need updating. `repr` changes
only above the threshold.

#### F2-011: Serialization depends on sharing: equal expressions encode to different bytes

- **Severity:** Medium · **Category:** bug · **Confidence:** High for the
  encoding; Medium for the downstream effect · **Python-driven:** no
- **Sources:** EXP-002
- **Location:** `expression/wire.rs:136-168` (`encode_nodes` dedups by
  `NodeIdentity` only); `:170-195` (the `Serialize` doc).

**Issue:** The encoder writes a node once only when it meets the same `Arc`
again. So two `==` expressions that share differently encode differently.

**Evidence:** `p2` in `scratch-expression/probes/tests/probes.rs`.
- `&l + &l` over one shared leaf encodes 2 nodes.
- `x + x` built from two leaves encodes 3 nodes.
- `s * s` and `s * s'`, where `s'` is `x + 1` built a second time, take 184
  and 302 bytes.

All three pairs are equal values.

**Why it matters:**
- **Ordering keys.** S17 made a Python `Serializable` member's canonical
  ordering key the `repr` of its V2 payload (W-12,
  `src/fhy_core/serialization.py:851`). A member that embeds an expression
  therefore orders by how its tree was built. This is plausible, not
  verified.
- **Comparing text.** Content-addressed caches, deduplication by JSON text,
  and golden comparisons all see equal values as different.

**Fix:** a decision, triage (b).
- **(a) Canonical encoding.** Hash-cons while encoding. Key the `written`
  map by the structural digest, which `compute_structural_digest` already
  computes bottom-up, and confirm equality on a hit.
- **(b) Document** that the text depends on sharing, and stop using payload
  text as an ordering key.

Either way, pin `x + x` built both ways in a test.

**Behavior change:** under (a), decoded values share every repeated
subtree, payloads shrink, and the golden corpus is regenerated. Under (b),
W-12's key changes.

#### F2-012: The NumPy evaluator aborts the interpreter or panics on large broadcasts

- **Severity:** Medium · **Category:** bug · **Confidence:** High ·
  **Python-driven:** no
- **Sources:** BUG-002
- **Location:**
  - `expression/evaluate/array.rs:176` (`shape.iter().product()` wraps in
    release builds);
  - `:256-263` (`standard_lanes` collects the whole broadcast);
  - `:374-376` (`Vec::with_capacity(lane_count)`);
  - `:485` (`broadcast_view`'s `unreachable!`).

**Evidence:** `target/audit/scratch-bugs/py/p01_numpy_basic.py` and
`p02_numpy_huge.py` bind two zero-stride views, of shapes `(2^n, 1)` and
`(1, 2^n)`.
- **n = 20:** "memory allocation of 8796093022208 bytes failed", and the
  interpreter dies with SIGABRT.
- **n = 33:** the product wraps to 0, and the call raises a
  `PanicException` from the `unreachable!`.

NumPy's own `np.add` raises `MemoryError` and `ValueError: iterator is too
large` for the same inputs.

**Fix:**
- Compute the lane count with `checked_mul`, and return a new
  `EvaluationError` (a `ValueError` in Python) on overflow.
- Use `try_reserve` for the output, mapped to `MemoryError`.
- Slice a broadcast binding per chunk, instead of materializing it whole
  (`ChunkSource::Copied`).

**Behavior change:** catchable errors instead of aborts, and memory for
broadcast inputs bounded by the chunk size.

#### F2-013: Unbounded recursion outside `Expression` aborts the process

- **Severity:** Medium · **Category:** bug (robustness) · **Confidence:**
  High · **Python-driven:** no
- **Sources:** EXP-006, TYP-011, BUG-003
- **Location:**
  - **`Pattern`:** `expression/pattern/matching.rs:129-236`, with a derived
    `Debug` on `PatternKind`, no `Drop` on `Pattern(Arc<..>)`, and the
    `match_into`/`match_node` recursion.
  - **Constraint `Value`:** `constraint/wire.rs:19-20` (documented: "JSON
    limits the depth, and postcard does not"), `constraint/value.rs:456`
    (`build_member`), `compare_canonically`, and the derived `Drop`.
  - **The binding:** `fhy-core-py/src/wire.rs:265-325` (`read_json_value`,
    followed by a recursive drop of `serde_json::Value`),
    `fhy-core-py/src/constraint/value.rs:432-470` (`read_member_value`) and
    `:320-348` (`value_to_python`).

**Issue:** F-003 gave `Expression` an iterative `Drop` and a bounded
`Debug`. Three other recursive structures still recurse once per level:
- `Pattern`, in drop, `Debug` and matching;
- constraint `Value`, in postcard decode, build, compare and drop;
- the binding's S17 dict fast path and member reader, which have no depth
  guard and no `Py_EnterRecursiveCall`.

A stack overflow kills the whole Python process, with no exception.

**Evidence:**
- **Patterns** (`scratch-expression/probes/tests/deep_pattern.rs`, log
  `p4-deep-pattern.log`), on a 1 MiB stack: dropping a 200,000-level
  pattern aborts, and so does matching a 20,000-level one. A 200,000-level
  expression, the control, drops fine.
- **Values** (TYP probe): 200,000 levels of `Tuple` fit in a 400 KB
  postcard payload, and decoding it ends in "stack overflow, aborting".
- **The binding** (`scratch-bugs/py/p04_deepdict.py`, `p40_deepdict_cls.py`):
  `Expression.deserialize_from_dict` of a 30,000-deep list segfaults (exit
  139). `p26_deeptuple.py` does the same with an `InSetConstraint` over a
  20,000-deep tuple. The same payload as JSON text is refused cleanly, by
  serde_json's 128-level limit.

**Fix:**
- **`Pattern`:** an iterative `Drop`, like `move_children_of_last_handle`,
  and a budgeted `Debug`. Optionally, an explicit-stack matcher.
- **`Value`:** a hand-written `Deserialize` for `ValueRepr` with a depth
  cap of 128.
- **The binding:** a depth limit of 128 in `read_json_value` (V2 shapes are
  shallow). In `read_member_value`, a depth check against
  `sys.getrecursionlimit()` that raises `RecursionError`, as the provenance
  binding does.

**Behavior change:** absurdly deep inputs are refused with an error instead
of aborting, and the `Debug` text of deep patterns changes.

#### F2-014: The process backend's timeout does not bound the call

- **Severity:** Medium · **Category:** bug · **Confidence:** High (items
  1–4); Medium for item 5, since no strict solver was installed ·
  **Python-driven:** no
- **Sources:** SOL-002, BUG-005, SOL-004, SOL-010, and `bugs.md`'s
  plausible "Leaked reader threads"
- **Location:** `solver/process.rs`:
  - `:186`, `:216-220` (`send`: a blocking `write_all` before any deadline
    check);
  - `:124-127` (`Failure::Closed`: an unbounded `child.wait()`);
  - `:265-270` (`reap`: after an answer, drops stdin and waits with no
    deadline);
  - `:101-110` (spawn) and `:276-283` (`kill`);
  - `:185-209` (`run`: the first non-blank line must be an answer).

**Issue:** Only `read_line` observes the deadline, so the call can outlast
`CheckLimits::timeout` in five ways:
1. **The write.** A solver that stops reading blocks `write_all` once the
   64 KiB pipe is full.
2. **Closed without exiting.** A program that closes stdout without
   exiting is waited on without bound, and the result is an error instead
   of a timeout.
3. **After the answer.** `reap` waits for the child without bound. A real
   solver that is slow to tear down, such as z3 with a large heap or a
   wrapper script that runs cleanup, holds the caller that long.
4. **Grandchildren.** The kill signals only the direct child. A wrapper
   that does not `exec` its solver (`sh -c 'z3 -in; …'`,
   `timeout 60 z3 -in`) leaves the solver running at full CPU, holding the
   pipe, and the reader thread leaks until it exits. Repeated timeouts
   accumulate orphans.
5. **`:print-success`.** SMT-LIB 2.6 defaults the option to `true`. The
   backend never turns it off, so a conforming solver's first `success`
   becomes `UnexpectedAnswer`. z3 and cvc5 default it to off.

**Evidence:**
- SOL probes in `scratch-solver/ws/rust/fhy-core/tests/audit_solver_probes.rs`,
  each with a 200 ms timeout:
  - C: a 389 KB script to `sh -c 'sleep 3'` returns after 3.006 s.
  - D: `sh -c 'exec 1>&-; sleep 3'` returns after 3.006 s, with an error.
  - E: `sh -c 'sleep 7.4321; true'` returns on time, but the grandchild
    survives.
- `scratch-bugs/py/p30_proc1.py`: `echo sat; exec sleep 30` with a 1 s
  timeout blocks for 30 s.

**Fix:**
- Write the script from a writer thread.
- In the `Closed` arm and in `reap`, poll `try_wait` until the deadline,
  then `kill`. Without a deadline, allow a short grace period after
  `(exit)`, then kill.
- On Unix, spawn with `CommandExt::process_group(0)` and signal the group.
  The crate forbids `unsafe`, so `killpg` needs `libc` or a `kill -KILL
  -<pgid>` subprocess. At the least, document that the program must `exec`
  its solver.
- Write `(set-option :print-success false)` before the script.
- Add an `sh`-fake story for each case.

**Behavior change:** these cases answer on time, with
`Unknown { reason: "timeout" }` or their answer. Nothing changes for z3 or
cvc5 run directly.

#### F2-015: Control characters in a name hint reach the solver: a NUL panics or breaks the script

- **Severity:** Medium · **Category:** bug · **Confidence:** High for the
  panic; Medium for strict solvers · **Python-driven:** no
- **Sources:** SOL-001, BUG-006
- **Location:**
  - `solver/smt/lower.rs:25-36` (`sanitize` replaces only `|` and `\`) and
    `:237`;
  - `solver/z3.rs:112-119` (`new_const(symbol.name.as_str())`, which in
    z3 0.21.1 calls `CString::new(s).unwrap()`).

**Issue:** A name hint is any string: `Identifier::new`, `try_restore` and
serde all accept it unchecked.
- **The `z3` feature.** A NUL panics inside `Z3Solver::check`, where the
  `Solver` API promises a `Result`.
- **The text path.** Control characters are written into a quoted symbol,
  and SMT-LIB 2.6 allows only printable characters and whitespace there.
  The z3-solver Python backend reads the script as a C string, so a NUL
  truncates it. `z3 -in` happens to accept it.

**Evidence:** SOL probes A and B (the panic, at `z3-0.21.1/src/symbol.rs:14`).
`scratch-bugs/py/p14_smtname.py`: a hint `"nul\x00x"` raises `Z3Exception`
("unexpected end of quoted symbol") on the z3 backend, while the process
backend answers `True`.

**Fix:**
- In `sanitize`, map every `char::is_control` character to `_`. The `_<id>`
  suffix keeps symbols distinct.
- Optionally, name z3 constants by `Symbol::Int(index)` and avoid the
  `CString` entirely.
- Add lowering and z3 stories for `"a\0b"`.

**Behavior change:** symbol text differs only for hints with control
characters.

#### F2-016: SymPy simplification turns `y / x` into `y * x ** -1`, which the evaluator refuses

- **Severity:** Medium · **Category:** bug · **Confidence:** High ·
  **Python-driven:** partly. S12 kept Python's lifted forms, and S9's
  refusal of negative integer powers made them invalid.
- **Sources:** BUG-004
- **Location:** `solver/sympy/lift.rs:171-180` (every SymPy `Pow` becomes
  `BinaryOperation::Power`).

**Evidence:** `target/audit/scratch-bugs/py/p17_pow.py`.
- `simplify(y / x)` is `(y * (x ** -1))`.
- At `x = 2, y = 3`, the original evaluates to 1.5, and the simplified
  form raises `ValueError: an integer raised to a negative integer power`.
- `p16_sympy.py` found 49 such mismatches in 684 random integer trees:
  every simplified division by an identifier fails.

**Fix:**
- Lift `Pow(b, -1)` as `1 / b`, and `Pow(b, -k)` as `1 / b ** k`. Fold
  `Mul(..., Pow(b, -1))` into a division node.
- Add a property: on integer grids, simplifying then evaluating equals
  evaluating.

**Behavior change:** simplified quotients come back as divisions.

#### F2-017: The checker types `-(5)` as `uint8` and refuses `-(128)` as `int8`

- **Severity:** Medium · **Category:** bug (checker soundness) ·
  **Confidence:** High · **Python-driven:** yes (`type_checker.py:660-700`
  at `ab2bbc6` had the same rule)
- **Sources:** TYP-001
- **Location:** `types/checking/checker.rs:604-607` (a unary `Infer` hands
  `expected` to its operand) and `:872-910` (the negate rule; the
  weak-literal rule is at `:888-907`).

**Issue:** The operand literal `5` resolves in the expected context
(`uint8`), so it is no longer weak, and the negate rule keeps the operand's
type. `-Expression::from(5)` is the ordinary way to build a negative number
(`build.rs:440-455`).

**Evidence (TYP probe):**

| Check | Result |
|---|---|
| `-(5)` against `uint8` | `Ok(uint8)` |
| literal `-5` against `uint8` | `Err` |
| `-(128)` against `int8` | `Err` (`int16` is wider) |
| literal `-128` against `int8` | `Ok` |

**Fix:** infer a `Negate(Literal)` node as the literal `-v` against
`expected`, or give the operand no expected type and check the negated
result.

**Behavior change:** `-(5): uint8` becomes a literal-range error, and
`-(128): int8` checks.

#### F2-018: Unification silently uses the unsubstituted expression when a substitution is refused

- **Severity:** Medium · **Category:** bug · **Confidence:** High ·
  **Python-driven:** no
- **Sources:** TYP-002 (also TYP-014 item 5, the missing story)
- **Location:** `types/unify.rs:571-573`
  (`.unwrap_or_else(|_refused| expression.clone())`), used by
  `bind_placeholder` (`:662-668`) and `Type::substitute_template` (`:265`,
  `:278-280`).

**Issue:** `Expression::substitute` refuses to put a non-Boolean literal
into a piecewise condition. `substitute_avoiding` then returns the
unsubstituted expression, and two things go wrong:
- `bind_placeholder` runs its occurs check on that form, so it misses an
  occurrence hidden behind a bound variable;
- `substitute_template` returns `Ok` with bound shape variables still in
  the type, although its doc promises they are replaced.

**Evidence (TYP probe),** in the environment `C := 5, Y := X + 1`:
- `unify(X, {Y if C; 0 otherwise})` returns `Ok`, accepting the cycle
  `X → Y → X`;
- the control, `unify(X, Y * 2)`, fails the occurs check as it should.

**Fix:** a decision, triage (b).

**Behavior change:** these calls return an error where they now succeed
with a wrong answer.

#### F2-019: `check_all_function_bodies` never checks a composed built-in; its test passes vacuously

- **Severity:** Medium (TST-001 said High; lowered, see the adjustments
  above) · **Category:** bug / test-gap · **Confidence:** High ·
  **Python-driven:** no. It is a port regression from D-9's reserved
  built-in names.
- **Sources:** TYP-003, TST-001 (also TYP-014 item 7)
- **Location:**
  - `types/checking/body.rs:151-168`, the loop, with
    `let Ok(name) = FunctionName::try_new(function.name()) else { continue; };`
    at `:155-157`;
  - `expression/callee.rs:126-132` (`try_new` refuses built-in names);
  - `tests/it/types/checking/body_stories.rs:215-218`;
  - `types/checking/body.rs:78-91` (`FunctionSignature::new` zips
    parameters with sorts).

**Issue:**
- **The sweep skips every built-in.** The Rust doc and the Python docstring
  (`body_type_checker.py:195-223`) promise that the sweep checks the
  composed built-ins in catalogue order. But `try_new` refuses every
  built-in name, so the loop skips all of them. The Python
  `check_all_registered_function_bodies()` skips them too.
- **The test is vacuous.** It asserts that the result is empty, which
  always holds.
- **A silent truncation.** `FunctionSignature::new` truncates a length
  mismatch between parameters and sorts.

**Evidence:**
- **lcov:** `body.rs:151` runs 70 times, `:155-156` (the `continue`) 32
  times, and `:158-168` 0 times, in both suites.
- **Probe** (`target/audit/coverage/probe`): `try_new` accepts 0 of the 16
  composed names. Called directly under stand-in names, all 16 bodies
  return `Checked`.

**Fix:**
- Key failures by something that can name a built-in, such as a
  `FunctionLabel::{Builtin, User}` or a `&str` label.
- Have the sweep return the checked labels or their count, and assert it
  against `BuiltinFunction::iter().filter(composed)`. Or inject a broken
  body through a crate-private seam.
- Delete or rename the vacuous test.
- Assert matching lengths in `FunctionSignature::new`, or return an error.

**Behavior change:** the sweep checks the built-ins, and its return type
changes.

#### F2-020: Decoders disagree with the checked constructors: valid tables fail, invalid assignments pass

- **Severity:** Medium · **Category:** bug · **Confidence:** High ·
  **Python-driven:** partly. D-S15-11 replays the checks as Python did, and
  assignments decode "as unpickling does".
- **Sources:** TYP-005, TYP-009
- **Location:**
  - `symbol_table/wire.rs:252-274` (`SymbolTableData::build` replays
    `add_namespace`/`add_symbol` in namespace order);
  - `symbol_table/table.rs:17-21` (the doc: "an inner namespace never
    shadows an outer one"), `:225-247` (`add_symbol` checks ancestors
    only) and `:452-483` (`violations`);
  - `param/wire.rs:369` (`ParamAssignment::new_unchecked`).

**Issue:**
- **Valid symbol tables fail to decode.** `add_namespace` accepts a forward
  parent, and `add_symbol` checks only the namespace's ancestors. So two
  tables built entirely through the checked API, each with an empty
  `violations()`, fail to decode (TYP probe):
  - a child added before its parent fails with "references missing parent
    namespace";
  - `y` added to a child, then to its parent, fails with "already defined
    in … an ancestor".

  The shadowing invariant breaks silently, and S17 requires a round trip.
- **Invalid assignments decode.** Decoding a `ParamAssignment` checks
  nothing. An integer param assigned `"not an integer"` round-trips.
  D-S17-9 says validated types decode through their constructors, and
  `ParamAssignment::restore` exists for this: it checks admissibility and
  the decided violations without a solver.

**Fix:**
- Make `add_symbol` also refuse a symbol that a descendant defines, and
  add `Violation::ShadowedSymbol`.
- Have `build` add every namespace first and then every symbol, or use
  `insert_namespace` followed by `violations()`.
- Decode assignments through `restore`, using the context that `Param`'s
  `Deserialize` already builds.

**Behavior change:**
- Some `add_symbol` calls that succeed today are refused, which enforces
  the documented invariant.
- Table payloads that fail to decode today will decode.
- Inadmissible assignment payloads fail to decode.

#### F2-021: Domain-level procedures ignore the domain's own restriction

- **Severity:** Medium · **Category:** api · **Confidence:** High ·
  **Python-driven:** yes (`domains.py` at `a44860c`:
  `is_value_admissible = is_strict_int`)
- **Sources:** TYP-006
- **Location:** `param/domain.rs:522-524` (`is_value_admissible`) and
  `:638-643` (`is_value_set_subset` compares only symbol types);
  `param/decide.rs:597-606` and `:624-647` (`has_feasible_value` and
  `feasibility_subset` use only the side's constraints).

**Issue:** A natural-number domain's restriction lives only in
`implied_constraints`, which `Param::new` folds in. The public `ParamDomain`
methods and `compute_constraint_implication_subset` never add it, and
their docs don't say the caller must. A Rust `CustomDomain` that reuses
them inherits the wrong answers.

**Evidence (TYP probe, real z3):**

| Question | Answer |
|---|---|
| `nat.is_value_admissible(-5)` | `true` |
| `nat.has_feasible_value(x <= -1)` | `Satisfied` |
| `integer.feasibility_subset([], nat, [])` (ℤ ⊆ ℕ) | `Satisfied` |
| `integer.is_value_set_subset(nat)` | `true` |
| `Param(nat, x <= -1).check_feasibility()` | `Violated` (correct only through `Param`) |

**Fix:** a decision, triage (b).

**Behavior change:** the domain-level answers become correct, or become
unavailable.

#### F2-022: Extension type defaults are not reflexive; structural equivalence is asymmetric

- **Severity:** Medium · **Category:** bug · **Confidence:** High ·
  **Python-driven:** no
- **Sources:** TYP-007, ARCH-013 (the asymmetry)
- **Location:** `types/extension.rs:216-221`, `:293-298`
  (`is_structurally_equivalent` defaults to `false`); `types/unify.rs:31`,
  `:179` (only the left-hand extension is asked); `types/ty.rs:234-258`.

**Issue:**
- **Not reflexive.** The default `eq_extension` is identity, but the
  default `is_structurally_equivalent` is `false`. So an extension that
  overrides only `type_name` is `==` to itself, yet cannot unify or bind
  with itself. D-S11-10 says the two are one relation on built-in parts.
- **Asymmetric.** `(Extension(e), _)` asks `e`, while `(Numerical,
  Extension)` answers `false`. Nothing documents that `eq_extension` must
  be symmetric.

**Evidence (TYP probe):**
- `t == t` is true, but `t.is_structurally_equivalent(&t)` is false.
- `t.unify(&t)` fails with "structural mismatch".
- `a == b` holds over one extension data type, but `a.bind_template(&b)`
  fails.

**Fix:**
- Default `is_structurally_equivalent` to the same identity as
  `eq_extension`.
- Make `(X, Extension)` consult the right-hand side, or document the
  symmetry requirement.
- The reshaping of the `Option<Result>` protocol belongs to F2-004.

**Behavior change:** an extension with no overrides unifies with itself,
and equivalence becomes symmetric.

#### F2-023: The constraint error slot keeps the first exception; a later `KeyboardInterrupt` is lost

- **Severity:** Medium · **Category:** bug (error mapping) ·
  **Confidence:** High · **Python-driven:** no
- **Sources:** PY-002
- **Location:**
  - `fhy-core-py/src/constraint/value.rs:42-50` (`record_pending_error`
    keeps only the first); `:224-238` (`is_equal` calls `==` whatever is
    pending); `:241-248` (`ordering_key` caches `""` after an error);
  - the other writers, also in the binding: `constraint/custom.rs:162, 218, 242`,
    `constraint/observer.rs:182`, `constraint/system.rs:184`,
    `param/custom.rs:292`, `param/observer.rs:340` and `wire.rs:362`.

**Issue:** The slot keeps the first exception raised during one core call,
answers `false`, and lets the core go on calling Python. A later exception
is dropped, `KeyboardInterrupt` and `SystemExit` included. The other
adapters get this right:
- the term and types adapters skip every hook after the first failure
  (`term/adapter.rs:231-248`, `types/adapter.rs:69-82`);
- `pass/scope.rs:134-147` lets a non-`Exception` outrank everything.

A smaller issue sits next to it (plausible, not reproduced): a failing
lazy `ordering_key` caches `String::new()` for good, so a transient failure
permanently changes the value's canonical order.

**Evidence:** `target/audit/scratch-binding/probe_kbi_pending.py`. An
`InSetConstraint` over two members whose `__eq__` raise `ValueError` and
then `KeyboardInterrupt` raises the `ValueError`, and both `__eq__` calls
run.

**Fix:**
- In `record_pending_error`, replace a kept `Exception` with a new error
  that is not an `Exception`.
- Check for a pending error at the top of `PyOpaqueValue::is_equal`,
  `order_against` and the custom-constraint hooks, and return the fallback
  without calling Python.
- Don't cache the fallback key; F2-004's fallible hooks remove the need
  for it.

**Behavior change:** `KeyboardInterrupt` and `SystemExit` always win, and
after the first exception, no further member `==` runs within that call.

#### F2-024: `PartiallyOrderedSet` and `Lattice` no longer pickle, copy or deep-copy

- **Severity:** Medium · **Category:** bug (API regression) ·
  **Confidence:** High · **Python-driven:** no. The Python classes pickled
  through `__dict__`.
- **Sources:** PY-003, BUG-007
- **Location:** `fhy-core-py/src/lattice.rs:182` (`PyPartiallyOrderedSet`)
  and `:312` (`PyLattice`) define no `__reduce__`, `__getstate__`,
  `__copy__` or `__deepcopy__`. The Python subclasses are
  `src/fhy_core/utils/poset.py:19` and `src/fhy_core/lattice.py:17`.

**Issue:** Before `05730d3`, both were plain Python classes over an
`nx.DiGraph`. The switch lost pickling, and neither S11's nor S11a's
divergence table (`python-switch.md:12339`) records it. Every other
switched container pickles, subclasses included, and no test covers these
two.

**Evidence:** `target/audit/scratch-binding/probe_pickle.py` and
`scratch-bugs/py/p20_pickle.py`: `TypeError: cannot pickle
'PartiallyOrderedSet' object`, and the same for `Lattice`, through
`pickle`, `copy.copy` and `copy.deepcopy`.
`scratch-binding/probe_subclass_pickle.py` is the control: the other
switched containers pickle.

**Why it matters:** any object that holds a lattice, such as a user IR with
a type-promotion lattice, can no longer be deep-copied or sent to a worker
process.

**Fix:**
- Add a `__reduce__` returning `(type(self), (), state)`. The state is the
  elements in insertion order plus the order pairs.
- Add a `__setstate__` that replays `add_element` and `add_order` and
  restores `__dict__` for subclasses, as `SymbolTable` does.
- Add pickle, `copy` and `deepcopy` tests.

**Behavior change:** restores the pre-switch behavior.

#### F2-025: Custom domain and custom constraint hooks are mostly untested in either language

- **Severity:** Medium (TST-002 said High; lowered, see the adjustments
  above) · **Category:** test-gap · **Confidence:** High ·
  **Python-driven:** no
- **Sources:** TST-002
- **Location:**
  - core ranges neither suite reaches: `param/domain.rs:663-665`
    (`CustomDomain::is_value_set_subset`), `param/decide.rs:648-650`
    (`feasibility_subset`) and `param/algebra.rs:144-147`
    (`intersection`);
  - Python only: `constraint.rs:124, 159, 197-198` (the `Constraint::Custom`
    arms);
  - the binding: `fhy-core-py/src/param/custom.rs:145-296`, where 7 adapter
    hooks are never called (33.9% covered), and
    `fhy-core-py/src/constraint/custom.rs:205`, `:224` (46.3%).

**Issue:** A user-defined domain is `param`'s extension point, but only 4
Rust stories exist (`tests/it/param/custom_stories.rs`). They never reach
subset, feasibility-subset, union, intersection or equivalence. The S16
test plan promised a test-local `CustomDomain` driven from each procedure,
and a Python-defined one too. Coverage of these paths is 0 in both suites
(`target/audit/coverage/scripts/check.py`).

**Fix:**
- **Rust:** a recording `CustomDomain` in `custom_stories.rs`. Drive it,
  on either side, through `is_value_set_subset`, `check_subset`,
  `Param::union`, `Param::intersection` and `is_structurally_equivalent`.
  Assert which hook ran, what it received, and that its error surfaces as
  `ParamError::Custom`.
- **Python:** the same in `test_domain_rust_binding.py`, with call
  counting. In `test_constraint_rust_binding.py`, a custom constraint
  compared with `is_structurally_equivalent` and
  `is_alpha_equivalent_under`.

#### F2-026: Evaluator and checker: shapes specified only by Python; IEEE and array edges untested

- **Severity:** Medium · **Category:** test-gap · **Confidence:** High ·
  **Python-driven:** no
- **Sources:** TST-003, TST-004, TST-005, EXP-008
- **Location:**
  - **evaluator walk:** `expression/evaluate/walk.rs:500-504` (`+x`, Python
    only), `:749-756` (a Boolean piecewise, Python only);
  - **array evaluator** (`evaluate/array.rs`), reached by neither suite:
    `:295-339` (Boolean and copied real bindings), `:358-365`, `:374-395`,
    `:421-424` (broadcast output) and `:642-649` (the kernel-shape check);
  - **kernels and chunks:** `evaluate/kernel.rs:57-80` (`real_divmod`),
    `evaluate/value.rs:28` (`Scalar` derives IEEE `PartialEq`),
    `evaluate/array.rs:215` (`CHUNK_LANES` is private);
  - **checker:** `types/checking/checker.rs:869-870` (`+x`), `:918-919`
    (`!p`), `:105-109` and `:1152` (Python only); `:1281-1288`,
    `:466-474` and `:647-648` (neither suite);
  - **tests:** `tests/it/expression/evaluate_stories.rs:248-273` and
    `tests/it/types/checking/checker_properties.rs`.

**Issue:**
- **Evaluator.** No Rust test evaluates `+x` on a number, or a
  Boolean-valued piecewise, and `tree_strategies`
  (`evaluate_properties.rs:67-116`) generates neither.
- **Array evaluator.** Untested:
  - Boolean array bindings;
  - strided real inputs that need a copy;
  - a result smaller than the broadcast shape;
  - a plugged-in kernel that returns the wrong shape.
- **Kernel edges.** The real floor division and modulo have 5 finite cases.
  Signed zeros, infinities, zero divisors and NaN are untested, and
  `assert_eq!` on `Scalar` cannot pin `-0.0` or NaN anyway: the existing
  `is_same_scalar` helper is not used in the stories.
- **Chunks.** Chunking has no property: the lane property runs on at most 8
  lanes, and the chunk size cannot be lowered for a test.
- **Checker.** No Rust test checks a well-typed `+x` or `!p`, or promotion
  in the `float ⊕ int` order. The property draws only `+` and `*` over
  `Int8`/`Int16`/`Int32`.

**Evidence:**
- Coverage: lcov and `check.py`.
- The code is right today. `scratch-expression/probes/tests/kernel_probe.rs`
  gives 26 results that equal NumPy 2.4.6 (`numpy_divmod.py`), and
  `chunk_probe.rs` checks 300,003 lanes against scalar evaluation. Both
  probes can be lifted into tests as they are.

**Fix:**
- Add `Positive` and Boolean piecewise branches to `tree_strategies`, plus
  rstest cases for `+3`, `+2.5` and a Boolean piecewise, in both the scalar
  and the array stories.
- Add the 13-row IEEE edge table, comparing `to_bits()` or using
  `is_same_scalar`, and an integer divmod law against an `i128` oracle.
- Make the chunk size a crate-private parameter, and run the lane property
  with chunks of 1 to 3 lanes.
- Add Boolean and transposed array bindings above and below the chunk
  size, and a test `ArrayKernels` that returns the wrong shape.
- For the checker: rstests for `+x` and `!p`, and `x_f32 + y_i16` in both
  orders. Broaden the property to unsigned and float types, comparisons,
  connectives, unary operators and piecewise. Add two laws: synthesis is
  symmetric for commutative operations, and the checker's result kind
  agrees with the evaluator's.

#### F2-027: Solver properties pass vacuously on `Unknown` and draw only integer predicates

- **Severity:** Medium · **Category:** test-gap · **Confidence:** High ·
  **Python-driven:** no
- **Sources:** SOL-005, TST-006
- **Location:**
  - `tests/it/solver/solver_properties.rs`: `:299`, `:329`, `:354`
    (`answer.decided().is_none_or(..)`), `:506`, the generators at
    `:100-200`, and `:278-280` (returns `Ok(())` when no backend is
    configured);
  - **z3, never run** (the binding does not enable `z3`):
    `solver/z3.rs:103, 105, 116, 237, 275-279, 290, 300, 310`;
  - **the lowering:** `solver/smt/lower.rs:59`, `:213-218`;
  - **SymPy, Python only:** `solver/sympy/simplify.rs:235-293` (the
    `piecewise_fold` workaround CONTRIBUTING cites) and
    `sympy/boolean.rs:170-188`;
  - **SymPy, neither suite:** `sympy/lift.rs:255-260`, `:472-477` (`Nand`,
    `Nor`), `:217`, `:239-241`, `:266-268`, `:286`, `:302`, `:314-316`
    (the `Arity` and `Unsupported*` errors), and `boolean.rs:66, 118` and
    `simplify.rs:90` (the `MAX_NESTING` guards).

**Issue:**
- **Vacuous passes.** Every solver-backed property accepts any undecided
  answer, including the hazard screen's `Refused`.
- **Narrow generators.** They draw only integer predicates over two `Int`
  variables: never a Boolean symbol, a real or a piecewise. The z3 term
  builder's Boolean and ITE arms go unexercised.
- **No comparison between backends.** Nothing compares `Z3Solver` with
  `SmtLib2Process`.

**Evidence:** a scratch mutant that made `Hazard::find` refuse every
expression left all six properties green (SOL).

**Fix:**
- Assert that no answer is `Unknown(Refused(_))` for these safe
  generators, and allow `GaveUp` only under a small budget.
- Add Boolean identifiers and piecewise leaves to `predicate()`,
  brute-forced over `p ∈ {false, true}`.
- Under `cfg(feature = "z3")`, add a differential property against the
  process backend when `FHY_SMT_SOLVER` is set.
- Make the no-backend skip visible: `#[ignore]` when the variable is
  missing, and a failure under `CI`.
- Add SymPy stories: lifting `Nand` and `Nor`, `Eq(a, b, c)` failing with
  `Arity`, the Boolean-condition piecewise inside a relational, and a hook
  that fails mid-walk.

#### F2-028: Param and constraint decision rules are tested only from Python, or not at all

- **Severity:** Medium · **Category:** test-gap · **Confidence:** High ·
  **Python-driven:** no
- **Sources:** TST-007, TST-008, TYP-014 items 1–3
- **Location:**
  - **Python only:**
    - `param/interval.rs:228-236`, `:271-274` (a bound written `5 <= x`),
      `:121-127`;
    - `param/context.rs:205-210`, `:226-228`, `:238` (`with_registry`);
    - `param/domain.rs:650-654` (categorical subset);
    - `param/value.rs:169-188` (type-strict equality; Rust tests only `5`
      against `5.0`);
    - `constraint.rs:107-116` (`Constraint::ptr_eq`).
  - **Neither suite:**
    - `param/interval.rs:373-389` (the infinity product);
    - `param/decide.rs:276-284`, `param/parameter.rs:381-383` and
      `param/screen.rs:93-96`;
    - `constraint/value.rs:493-516`, `:598-604` (opaque membership);
    - `constraint/binding.rs:74-75` (rebinding) and
      `constraint/system.rs:216` (an undecided leaf).
  - **The interval property:** `tests/it/param/param_properties.rs:223-258`
    uses only bounded, inclusive, non-natural operands.

**Issue:** CONTRIBUTING says the Rust tests "specify the concept's
behavior", and S16b.1's 44 tests were "written after the code". Finite
union and intersection have no brute-force property, permutation domains
appear in no property, and only one story asks numeric questions of the
real solver.

**Evidence:** these TYP probes pass today and can be adopted
(`scratch-types/fhy-core/tests/it/audit_probes.rs`):
- the interval hull with unbounded, natural and exclusive operands: 2,000
  cases;
- finite and integer intersection (`batch4`);
- numeric feasibility and subset against z3 (`batch6`): 300 cases, 0
  unsound answers, about 85% decided.

**Fix:**
- Adopt the three probes, gating the z3 one as
  `system_satisfiability_agrees_with_brute_force` is gated.
- Add rstests for both bound spellings, a `with_registry` story, and
  categorical subsets.
- Test assignment equivalence per `Value` kind, including `-0.0` against
  `0.0`, and `1` against `True` in tuples.
- Test opaque membership, alone and inside a tuple or frozenset, and
  extend the set-membership property to draw opaque members.
- Test that rebinding an identifier keeps its position, and that a system
  with one undecided leaf reports `Undecided`.

#### F2-029: Error text and source chains untested; loose and weak assertions; small untested behaviors

- **Severity:** Medium · **Category:** test-gap · **Confidence:** High ·
  **Python-driven:** no
- **Sources:** TST-009, TST-011, TST-014
- **Location:**
  - **Low-coverage error files:** `param/error.rs` (48.6%;
    `natural_bound_text` at `:259-281` picks one of 8 messages),
    `solver/sympy/error.rs` (59.3%), `types/checking/error.rs` (71.1%),
    `expression/evaluate/error.rs` (71.7%), `constraint/error.rs` (72.7%),
    `solver/error.rs` (78.9%), `solver/process.rs:308-332` and
    `foreign.rs:187-193`.
  - **Loose `{ .. }` matches:** e.g. `param_stories.rs:116, 286`,
    `algebra_stories.rs:403`, `body_stories.rs:80, 178`,
    `fold_stories.rs:330, 400` and `extension_stories.rs:265-266, 353`.
  - **Weak assertions:** `sympy_simplify_stories.rs:115-124` (named "is
    simplified", asserts `is_ok()`), `registry_stories.rs:750, 782`,
    `unification_stories.rs:424`, `pass/core_stories.rs:1437`,
    `inline_stories.rs:548` and `constraint/equation_stories.rs:166`.
  - **Small untested behaviors:** `expression/pattern/rewrite.rs:118-126`,
    `:172-174` (blame through a grandchild), `symbol_table/table.rs:420,
    436`, `types/unify.rs:71, 283-285, 366`, `types/environment.rs:46-56`,
    `solver/process.rs:193-199`, `:318-321` and `param/context.rs:162-164`.

**Issue:** Error text is Rust-defined everywhere except in the
dual-defined concepts, yet the Rust suite checks little of it. Python pins
some of it only through the binding (`test_nat_param.py:577-673`), and
`source()` chains are rarely asserted. Most error tests are precise (264
`expect_err`, 122 `to_string()` comparisons); the gap is concentrated in
the files above.

**Fix:**
- One rstest table per error enum, checking each variant's `to_string()`
  and whether `source()` is `Some`, and of which type.
- For `NaturalBound`, assert the fields for each of the 8 combinations.
- Use full variants where the fields are the point.
- Assert the simplified form; compare values instead of `is_some()`;
  downcast `source()` to the expected type.
- Add the small stories: rewrite blame through a grandchild, table
  inequivalence in both directions, and an `unknown` whose reason line
  cannot be read.

#### F2-030: Binding and interface-suite gaps; stub drift the stub test cannot see

- **Severity:** Medium · **Category:** test-gap · **Confidence:** Medium
  for the member cross-check (name matching is coarse); High for the stub
  drift · **Python-driven:** no
- **Sources:** TST-012, PY-008
- **Location:**
  - **Binding coverage,** under `fhy-core-py/src/`: `types/adapter.rs`
    (70.1%; `:302`, `:468`, `:472`, `:542`), `types/dispatch.rs:55-95`
    (72.3%), `param/objects.rs:142`, `:296` (74.1%), and `wire.rs:155`,
    `:425`;
  - **The stub:** `tests/test_rs_stub.py:125-160` (top-level names and
    function parameters only), and `src/fhy_core/_rs.pyi:250` and `:468`.

**Issue:**
- **Members.** 34 of 741 public members are named in no interface suite
  (`target/audit/coverage/members.json`). `TypeUnificationEnvironment.type_bindings`
  appears in no test at all, and its core counterpart
  (`types/environment.rs:111`) has no Rust test.
- **Error paths.** Those of Python-defined type dispatch are untested, and
  so is refusing a foreign part of the wrong kind on decode.
- **Stub drift** (`scratch-binding/stub_compare.py` and `.out`):
  - `_rs.ValueDomain.from_json` is declared, but exists only on the public
    subclass;
  - `rebuild_with_visit_children` is declared on the base class, but only
    the node classes define it;
  - nothing checks the stub's `__new__` parameters against what the Rust
    parser accepts.

**Fix:**
- Test `type_bindings` from both languages.
- Test a dispatcher handler that returns the wrong kind, a `Foreign`
  payload of the wrong kind, and a Python extension type through
  `substitute_template`.
- Name each listed `Param` and `Solver` method in the interface suites,
  with its object identity and `KeyboardInterrupt` behavior, as the S16
  plan lists.
- Extend `test_rs_stub.py` to compare each class's non-dunder members in
  both directions and their descriptor kinds, with an allowlist for
  PyO3's dunders. Then fix the two stub entries.

---

### Low

#### F2-031: Three of the binding's six thread-local stacks have no unwind guard

- **Severity:** Low · **Category:** idiom (panic safety) · **Confidence:**
  High for the code; Medium for the impact · **Python-driven:** no
- **Sources:** ARCH-009, PY-007
- **Location,** all under `fhy-core-py/src/`:
  - the unguarded three: `types/adapter.rs:93-112`
    (`run_in_context`), `solver/backends.rs:187-201`
    (`run_simplification`) and `constraint/value.rs:54-81`
    (`with_pending_errors`, `capture_pending_errors`);
  - the guarded three: `pass/scope.rs:150-186` (`ScopeGuard`),
    `pass/context.rs:92-151` (`FrameGuard`) and
    `expression/pattern/objects.rs:168-212` (`ActiveTable`).

**Issue:** The three push, run the body, then pop, with no `Drop` guard.
After a panic, which PyO3 turns into `PanicException`, the thread keeps the
stale frame:
- `current_context()` returns a dead context;
- `current_input_object` reuses a dead simplification's input;
- `PENDING_ERROR`'s outer value is never restored.

The bodies contain `unreachable!` calls and slice indexing, so a panic is
possible.

**Fix:** one `ScopedStack<T>` helper whose guard pops in `Drop`, used for
all six stacks. This joins F2-033's consolidation.

**Behavior change:** none, except after a panic.

#### F2-032: `constraint`/`param` values lack `PartialEq`/`Eq`/`Hash`/`Display`; template widths compare as a list

- **Severity:** Low · **Category:** api · **Confidence:** High ·
  **Python-driven:** mixed
- **Sources:** ARCH-013, TYP-013 (missing equality traits)
- **Location:**
  - `constraint.rs:94`, `constraint/{equation,set,system}.rs`,
    `param/domain.rs:403`, `param/parameter.rs:49` and
    `constraint/value.rs:127`;
  - `types/ty.rs:234-258`, `types/unify.rs:169` and
    `symbol_table/frame.rs:35, 68`;
  - `types/data_type.rs:343-377`.

**Issue:**
- **Only `Debug` and `Clone`.** `Constraint`, `EquationConstraint`,
  `SetConstraint`, `ConstraintSystem`, `ParamDomain`, `Param`,
  `ParamAssignment`, `Value` and `Binding` derive nothing else. Equality is
  a method, and ordering is a `String` key. So these types cannot go in a
  `HashSet` or `BTreeSet`, cannot be used with `assert_eq!`, and have no
  `Display`.
- **Two equalities, undocumented.** `Type`, `DataType` and `SymbolFrame`
  have `PartialEq` and a separate `is_structurally_equivalent`, and nothing
  says which is which.
- **Widths as a list.** `TemplateDataType` stores its widths as given, so
  `[8, 16] != [16, 8]`, and an empty list never binds.

**Fix:**
- Implement `PartialEq`, `Eq` and `Hash` as structural equivalence, with
  `Custom` parts going through `is_structurally_equivalent`.
- Derive `Ord` from the key's structure (F2-001), and add `Display`.
- Document the two equalities in `types`.
- Sort and deduplicate template widths, and refuse an empty list.

**Behavior change:** additive, except for the width normalization and the
refusal of an empty list.

#### F2-033: Binding boilerplate duplicated: imports, exceptions, frozen protocol, seeds, error conversion

- **Severity:** Low · **Category:** idiom (duplication) · **Confidence:**
  High · **Python-driven:** partly (the frozen protocol mirrors
  `FrozenMixin`)
- **Sources:** ARCH-015, PY-010
- **Location** (all under `fhy-core-py/src/`):
  - **import helpers:** `term/derived.rs:36-45`, `wire.rs:52-60, 116-124,
    519`, `expression/registry/lookups.rs:25-27`,
    `param/parameter.rs:54-56`, `param/value.rs:18-20`,
    `param/objects.rs:33`, `solver/error.rs:28-41`, `solver/sympy.rs:50`,
    `symbol_table/frames.rs:906`, `constraint/value.rs:84-86`,
    `types/checking.rs:57`, `expression/evaluate/error.rs:22` and
    `expression/pattern/walk.rs:32`;
  - **duplicate exception caches:** `serialization.rs:23-42` against
    `wire.rs:125-150`;
  - **object tables:** `expression/materialize.rs:49`,
    `expression/pattern/objects.rs:50`, `types/adapter.rs:54-56`,
    `solver/backends.rs:175`, `term/adapter.rs:40`,
    `types/checking.rs:145` and `constraint/kinds.rs:516`;
  - **the frozen protocol:** 31 copies, e.g. `provenance.rs:351-423` and
    `value_domain.rs:295-335`;
  - **duplicate converters:** `expression/pattern/kinds.rs:51` against
    `expression/evaluate/error.rs:60`;
  - **seeds:** `expression/pattern/rules.rs:284`,
    `expression/registry/entries.rs:506`, `param/parameter.rs:642-647`,
    `:1582-1600` and `types/environment.rs:463-478`.

**Issue:**
- 128 `PyOnceLock` caches sit behind 15 local copies of one import helper.
- Two exception classes each have two caches.
- Seven node-to-object tables each preserve Python identity their own way.
- About 450 lines of frozen protocol are hand-written.
- The exception-construction sequence is written out about 24 times.
- There are 20 `IntoPyErr` impls and 18 free `*_to_py`/`*_to_python`
  functions, against CONTRIBUTING's rule.
- Seeds follow three conventions. The `_state` one silently builds an
  empty environment on reuse, and swallows lookup errors.

**Fix:**
- Add a `python.rs` helper (`cached_attr!` or `ImportedAttr`) and an
  `exceptions.rs` with one constructor per class, plus `unbox_py_err`.
- Add one `ObjectTable` with the guarded scope of F2-031.
- Move the frozen protocol onto the Python classes, or generate it with a
  macro.
- Add one `Seed<T>` that is taken once and raises on reuse.
- Amend CONTRIBUTING to allow converters that take context, or convert
  them.

**Behavior change:** a reused environment seed raises instead of building
an empty environment.

#### F2-034: Composed `max`/`min`/`relu`/`clamp`/`abs` give order-dependent NaN results

- **Severity:** Low · **Category:** bug (numeric semantics) ·
  **Confidence:** High · **Python-driven:** yes, a faithful port of
  `builtins.py:179-188` at `67a8ab6`
- **Sources:** EXP-004
- **Location:** `expression/builtins.rs:578-606` (`build_max_body`,
  `build_min_body`, `build_abs_body`, `build_sign_body` and
  `build_relu_body`; `clamp` and `clamp_symmetric` build on `max` and
  `min`).

**Issue:** `max(a, b)` is `a if a > b else b`, and a comparison with NaN is
false, so the result depends on operand order. That is neither IEEE
`maximum` (NaN-propagating, as in NumPy) nor `maxNum` (NaN-dropping, as in
`f64::max`).
- `max` is not commutative on NaN, so a rewrite that swaps its arguments
  changes values.
- `relu(nan) = 0` hides NaNs, the opposite of the NumPy semantics the
  evaluator otherwise follows.
- No test mentions NaN for the composed built-ins.

**Evidence** (`p6` in `scratch-expression/probes/tests/probes.rs`):

| Call | Result |
|---|---|
| `max(nan, 1)` | `1.0` |
| `max(1, nan)` | `NaN` |
| `min(nan, 1)` | `1.0` |
| `min(1, nan)` | `NaN` |
| `relu(nan)` | `0.0` |
| `abs(-0.0)` | `-0.0` |
| `sign(nan)` | `0` |

**Fix:** a decision, triage (b).

**Behavior change** under NaN propagation: NaN inputs propagate through
`max`, `min`, `clamp`, `clamp_symmetric`, `relu` and `leaky_relu`, and
`abs(-0.0)` is `0.0`. The Z3 and SymPy lowerings inline the same bodies,
so only their tests' expected texts change.

#### F2-035: `substitute_avoiding_capture` renames needlessly, and renames a repeated binder twice

- **Severity:** Low · **Category:** bug · **Confidence:** High ·
  **Python-driven:** no
- **Sources:** EXP-005
- **Location:** `term/binder.rs:258-289` (`capturable` at `:271`, the
  rename loop at `:276`).

**Issue:**
- **Needless renaming.** `capturable` gathers the free identifiers of
  every replacement, including keys that never occur in the body. A binder
  in that set is renamed and rebuilt although nothing changes. This spends
  a fresh id, loses sharing and `ptr_eq`, and in the binding costs one
  Python hook call per node.
- **Renaming what is no longer bound.** The loop walks the original bound
  identifiers while renaming. Under `\x x. z` with `z ↦ x`, the second
  iteration hands the user's `rename_bound_identifier` an `old` identifier
  the binder no longer binds.

**Evidence:** `scratch-expression/probes/tests/term_probe.rs`, `p8` and
`p8b`.

**Fix:** restrict the active keys to the binder's free identifiers, and
iterate over the *distinct* capturable identifiers. Add both probes as
stories.

**Behavior change:** substitutions that replace nothing inside a binder
return the same handle and allocate no ids. Results stay
alpha-equivalent.

#### F2-036: Float literal text: lax decoding, and 300-character encodings of extreme magnitudes

- **Severity:** Low · **Category:** idiom (wire and display) ·
  **Confidence:** High · **Python-driven:** no; D-7 is a port decision
- **Sources:** EXP-007
- **Location:** `expression/literal.rs:317-342` (`float_text` decodes with
  `f64::from_str`), against `:276-313` (`integer_text` accepts only
  canonical text); `:190-199` (`Display` writes `{value}`).

**Issue:**
- **Only integers are canonical on decode.** The float decoder accepts
  anything Rust parses, and the decimal decoder any grammar text, so a
  decoded payload can re-encode to different bytes.
- **The `{}` form is positional.** It never uses an exponent, so extreme
  floats inflate the wire text, `Display`, constraint keys and error
  messages about sixty-fold.

**Evidence:** `p7` and `float_text_probe.rs`.
- `"Infinity"`, `"+inf"`, `"1e5"`, `"+1.5"`, `".5"`, `"5."`, `"1E-2"`,
  `"nan"`, `"-NaN"` and `"00.10"` all decode as floats, and re-encode
  differently.
- `"01"`, `"+1"` and `"1e5"` are refused as integers.
- `1e300` encodes as 301 characters, `5e-324` as 326, and `f64::MAX` as
  309.

**Fix:** a decision, triage (b).

**Behavior change:** non-canonical payloads are refused; no writer in the
repository produces them. Revising D-7 changes the V2 text of extreme
floats, and the committed corpus is regenerated.

#### F2-037: `Children` has no size hint; walks re-count children; unary `+` copies lanes

- **Severity:** Low · **Category:** perf · **Confidence:** Medium (read,
  not benchmarked) · **Python-driven:** no
- **Sources:** EXP-009
- **Location:**
  - `expression/node.rs:31-92` (`Children` has the default `size_hint`),
    `:516` and `:697` (the private `count_children`, unused);
  - `evaluate/walk.rs:171`, `:314-322` and `:500-505`;
  - `registry/inline.rs` and `evaluate/fold.rs` (`Step::Exit`), and
    `wire.rs:152`.

**Issue:** Every post-order walk re-counts children by iterating, and
collections start from zero capacity. Unary `+` copies its operand's
lanes, a full array allocation for an identity. `reserve_failures` clones
the failing node once per failure kind.

**Fix:**
- Implement `size_hint`, `ExactSizeIterator` and `FusedIterator` for
  `Children`, and use `len()` in the walks.
- Pass unary `+`'s operand through.
- Store the failing node once.

**Behavior change:** none. `children()` gains `ExactSizeIterator`, which is
additive.

#### F2-038: Three more super-linear paths: SymPy lifting, shape substitution, permutation enumeration

- **Severity:** Low · **Category:** perf · **Confidence:** High ·
  **Python-driven:** partly. TYP-008 and TYP-012 inherit Python's
  algorithms; SOL-006 is new in the port.
- **Sources:** SOL-006, TYP-008, TYP-012, and `bugs.md`'s plausible
  "Exponential or recursive expression substitution in unification"
- **Location:**
  - `solver/sympy/lift.rs:88-110` (`Lifter::lift`, no memo) and
    `solver/sympy/substitute.rs:38-95` (`rebuild_bottom_up`, no memo);
  - `types/unify.rs:549-574` (`substitute_avoiding` re-substitutes each
    binding every time it is reached) and `types/environment.rs:61-83`
    (each `with_*` copies the whole map);
  - `param/decide.rs:537-593` (`Permutations`), `:609-617` and `:662-672`.

**Issue:**
- **SymPy lifting.** The lowering keeps sharing, but lifting and
  substitution walk SymPy's `args` as a tree. The cost is exponential, and
  the lifted result loses the sharing.
- **Shape substitution.** Bindings shaped like `N_i := N_{i+1} + N_{i+2}`
  cost Fibonacci time. By reading, a chain of about 10^5 bindings would
  also overflow the stack (plausible, not built).
- **Permutations.** Feasibility and subset walk all `n!` permutations,
  even when an in-set constraint already lists every candidate.

**Evidence:**

| Path | Growth | Probe |
|---|---|---|
| lifting `e_{k+1} = sin(e_k) + cos(e_k)` | 3, 46, 186 and 835 ms at depths 8, 12, 14, 16; lowering stays linear | SOL `audit_sympy_probes.rs`, S1 |
| the Fibonacci bindings | 19, 75 and 524 ms at n = 16, 20, 24 (debug) | TYP |
| a permutation param with the in-set `{99}` | 6.5, 29 and 102 ms at n = 6, 7, 8; about 9 s at n = 10 | TYP |

**Fix:**
- Memoize the lift and the rebuild by `id(object)`, keeping the objects
  alive in the memo.
- Memoize each identifier's substituted form within one call, and detect
  cycles by white/grey/black marking. This is valid once F2-018 keeps the
  environment acyclic.
- When an in-set is present, enumerate `in_set_candidates` filtered by
  `is_permutation`.

**Behavior change:** none, except that SymPy results are shared.

#### F2-039: Any module at `sys.modules["_fhy_core_sympy"]` is trusted as the prelude

- **Severity:** Low · **Category:** bug · **Confidence:** High ·
  **Python-driven:** no
- **Sources:** SOL-007
- **Location:** `solver/sympy/load.rs:441-453` (`prelude`) and `:354-355`.

**Issue:** The backend reuses whatever module is published under the fixed
name, as long as it has the five attributes. So two copies of fhy-core in
one interpreter share the first copy's prelude, even when their sources
differ, and so does a stale or user module under that name.

**Evidence:** SOL probe S2 pre-publishes a module whose `ROUND` is
`sympy.floor`. `load()` succeeds, and `round(3.5)` then simplifies to 3,
where the prelude's round-half-even gives 4.

**Fix:** publish under a versioned name (`_fhy_core_sympy_<version>_<hash
of PRELUDE_SOURCE>`), or check a `__fhy_core_prelude__` hash attribute and
report a mismatch as `SympyUnavailableError::Incompatible`.

**Behavior change:** mismatched preludes fail to load, instead of loading
silently.

#### F2-040: The mixed int/real equality hazard refuses questions the crate can decide

- **Severity:** Low · **Category:** doc / idiom · **Confidence:** High ·
  **Python-driven:** yes, a leftover from the deleted z3-Python bridge
  (D-S8-5 kept the screens unchanged)
- **Sources:** SOL-009
- **Location:** `solver/screen.rs:62-66` (the doc: "a solver's numeric
  comparison would equate where this crate keeps `1` and `1.0` apart") and
  `:490-514`.

**Issue:** The hazard refuses `x_int == 1.0`, but the crate's evaluator
equates the two, and the lowering `(= (to_real x) 1.0)` means exactly
that. The same comparison without a literal, `x_int == y_real`, is not
screened and is answered.

**Evidence:** SOL probe K, at `x = 1` and `y = 1.0`:
- `evaluate(x_int == y_real)` and `evaluate(x_int == 1.0)` are both true;
- `x_int == y_real` is answered `Yes`;
- the literal form is refused as `Unknown(Refused(MixedIntRealEquality))`.

**Fix:** a decision, triage (b).

**Behavior change:** if the hazard is dropped, more questions are answered
instead of refused.

#### F2-041: `ValidatorRecord::diagnostics_in` panics on, or silently misreads, another report

- **Severity:** Low · **Category:** api · **Confidence:** High ·
  **Python-driven:** no
- **Sources:** SOL-012
- **Location:** `pass/validation.rs:201-212`. `ValidationReport::new` is
  public (`diagnostic.rs:344`), so the ranges are unchecked.

**Issue:** A record stores index ranges into its report and slices
whatever report it is handed. A shorter report panics, as documented under
`# Panics`, and a longer one silently returns another validator's
diagnostics.

**Evidence:** SOL probe P (`audit_pass_probe.rs`): a record from a report
of 3 diagnostics, sliced against a report of 0, panics at
`validation.rs:212:30`.

**Fix:** keep D-S6-17's representation, but read through the report, with
`ValidationReport::diagnostics_of(index)` or an iterator of `(record,
diagnostics)` pairs. Make `diagnostics_in` use `get(range)` and return an
`Option`.

**Behavior change:** the signature changes, and nothing panics.

#### F2-042: Colliding opaque keys: equal ordering keys without equivalence

- **Severity:** Low · **Category:** bug / doc · **Confidence:** High ·
  **Python-driven:** yes
- **Sources:** TYP-010
- **Location:** `constraint.rs:19-20` and `constraint/key.rs:1-9` (the
  claim that keys are equal exactly when constraints are equivalent);
  `key.rs:132` (an opaque member renders only its own key);
  `constraint/value.rs:55`; `constraint/system.rs:26-31`.

**Issue:** `OpaqueValue::ordering_key` need only be "equal for equal
values". So two in-set constraints over unequal opaque values can share a
key, and the stable sort then keeps their input order. This breaks the
module contract and the property that a system's order is independent of
construction order; that property's generator never produces collisions.
`Param` equivalence depends on system order.

**Evidence (TYP probe):** the keys are equal and the constraints are not
equivalent. The same members in two orders give systems that are not
equivalent.

**Fix:** break key ties by `is_structurally_equivalent` groups, sorting by
key and then by a per-group order independent of input. Otherwise,
document the order as canonical only up to key collisions. The tie-break
is recommended, and it folds into F2-001's structural `Ord` if that lands.

**Behavior change:** systems order canonically.

#### F2-043: Binding threading: implicit free-threading, and NumPy inputs read in place while detached

- **Severity:** Low · **Category:** soundness / doc · **Confidence:** High
  (verified; the race was observed) · **Python-driven:** no
- **Sources:** PY-006, PY-004, and `bugs.md`'s plausible "Data race on
  NumPy bindings while detached"
- **Location:**
  - `fhy-core-py/src/lib.rs:33` (`#[pyo3::pymodule(name = "_rs")]`, with no
    `gil_used`); pyo3-macros-backend 0.29.2 `module.rs:394`;
  - `.github/workflows/python-package.yml:239` and
    `python-release.yml:53-58` (3.10 to 3.14, no free-threaded builds);
  - `fhy-core-py/src/expression/evaluate/numpy.rs:53-62`, `:125`,
    `:461-475` and `:502-504`.

**Issue:**
- **Free-threading is implicit.** PyO3 0.29 made free-threading support
  opt-out (#5564), so a build on 3.14t declares `Py_MOD_GIL_NOT_USED`, and
  nothing has checked that this is safe. The known consequences:
  - the non-frozen classes raise borrow errors under contention (F2-044);
  - `PyOpaqueValue.key`'s `OnceLock` runs Python in its initializer;
  - several "never held across Python" invariants were argued only for
    the GIL build.

  No decision records the choice.
- **A data race.** A `float64` NumPy input is evaluated through `&[f64]`
  views of the caller's own buffer inside `py.detach`. Another thread
  writing that array (NumPy releases the GIL in ufunc loops) races memory
  that Rust holds as shared, which is undefined behavior. The old
  pure-NumPy evaluator also gave torn results, so this is not a
  user-visible regression, and rust-numpy's documented position blames the
  mutating code.

**Evidence:** `target/audit/scratch-binding/probe_numpy_race.py` evaluates
`(x*2.0) - (x+x)` while another thread runs `np.add(arr, 1, out=arr)`. The
result is nonzero in 20 of 20 runs, and in none without the writer.

**Fix:** a decision, triage (b).

**Behavior change:** with `gil_used = true`, importing on 3.14t re-enables
the GIL, with CPython's `RuntimeWarning`. Copying costs one extra copy per
array input.

#### F2-044: Python code runs under a `PyRef`/`PyRefMut` borrow; re-entrant callbacks fail

- **Severity:** Low · **Category:** bug (re-entrancy) · **Confidence:**
  High · **Python-driven:** no; the Python classes allowed it
- **Sources:** PY-005
- **Location:** all under `fhy-core-py/src/`:
  - `symbol_table/table.rs:267-268` and `:398-399`, where `add_namespace`
    and `add_symbol` take `borrow_mut()`; `:203-215` and `:123-137`, where
    `Entry::new` reads a Python frame's `name` under that borrow;
  - `lattice.rs:71-84`, where `Elements::insert` hashes and compares the
    element under `add_element`'s borrow (`:261`, `:370`);
  - `lattice.rs:109-114` with `:239-246`, where `ranks` calls the user's
    `key` under the `iter_stable` borrow.

**Issue:** A frame's `name` property, a key callback, or an element's
`__hash__` that reads the container fails with PyO3's generic "Already
borrowed" error. Nothing is corrupted.

**Evidence:** `target/audit/scratch-binding/probe_reentrancy.py`: three
`RuntimeError`s ("Already borrowed" or "Already mutably borrowed").

**Fix:** do every Python read before taking the borrow:
- in `add_symbol`, build the `Entry` and read the identifiers first;
- in `Elements::insert`, run `contains` and `set_item` on a cloned
  `Py<PyDict>`;
- in `iter_stable`, compute the ranks over a cloned element list.

**Behavior change:** re-entrant reads work, and re-entrant mutations see
consistent state.

#### F2-045: Big Python numbers enter the core through decimal text

- **Severity:** Low · **Category:** bug (resource use) · **Confidence:**
  High · **Python-driven:** no
- **Sources:** PY-009, BUG-009, BUG-008
- **Location,** under `fhy-core-py/src/`: `expression/literal.rs:75-98`
  (`read_decimal` formats with `"f"`), `:45-56` (`read_big_int` goes
  through `int.__repr__`), `:60-68` (`big_int_to_python` goes through
  `int(str)`), and `expression/evaluate/literal.rs:31-40`.

**Issue:**
- **`Decimal` exponents.** The binding expands a `Decimal` in exponent form
  into all its fixed-point digits, so time and memory grow with the
  exponent. Payloads cannot trigger this, since the core grammar refuses
  exponent text; only Python callers can.
- **Big ints.** They go through decimal text and so hit CPython's
  4,300-digit guard.

**Evidence:**
- **The Decimal cost.** `scratch-binding/probe_decimal_exponent.py`:
  `Decimal('1e100000000')` takes 0.585 s and 235 MiB. In
  `scratch-bugs/py/p32_dec1.py`, `LiteralExpression(Decimal("1e+5000000000"))`
  and `check_param_bounds_are_ordered(Decimal("1e-4000000000"), ...)`
  give no answer within 15 s, while memory keeps growing.
- **The digit guard.** `p24_bigint.py`: `LiteralExpression(10**5000)`
  raises CPython's digit-limit `ValueError`. In `p23_wire_adv.py`,
  `Expression.from_json` of a 5,001-digit integer is accepted by Rust, but
  materializing it raises a bare `ValueError`, not a
  `DeserializationValueError`.

**Fix:**
- Read `Decimal.as_tuple()` (the digits and the exponent) and build the
  core `Decimal` directly. Refuse exponents beyond a documented bound,
  shared with the core parser (see F2-008).
- Convert ints with `int.to_bytes`/`int.from_bytes`, or through pyo3's
  `num-bigint` feature. That has no digit limit and is linear.

**Behavior change:** absurdly scaled `Decimal`s are handled or refused at
once, and ints of any size are valid literals that decode.

#### F2-046: Serde round trips are example-only for params, types and symbol tables

- **Severity:** Low · **Category:** test-gap · **Confidence:** High ·
  **Python-driven:** no
- **Sources:** TST-010, TYP-014 item 4
- **Location:**
  - no `proptest!`: `tests/it/param/serde_stories.rs` (7 tests),
    `tests/it/types/serde_stories.rs` (12) and
    `tests/it/symbol_table/serde_stories.rs` (8);
  - `param/wire.rs:127, 131-133` (domain members of kind bool, tuple and
    frozenset are never serialized in Rust);
  - `tests/it/serialization_golden.rs:97` (a `"Value"` replay arm with no
    corpus case).

**Issue:** CONTRIBUTING's JSON-plus-postcard round trip holds by example
only, so the recursive types get no generated shapes. A symbol-table
round-trip property over `table_properties.rs`'s operation model would
have caught F2-020. The 89 golden cases include no `Value`.

**Fix:**
- Add proptest strategies for `ParamDomain` (all six kinds, nested
  members), `Type` and `DataType` (nested templates and shapes), and
  `SymbolTable` (from the operation model). Each round-trips through JSON
  and postcard, and re-serializes to byte-identical JSON.
- Add a `Value` case, a frozenset of tuples, to the corpus generator.

#### F2-047: Dead, unreachable and trivial code: make the guarantees types

- **Severity:** Low · **Category:** idiom · **Confidence:** High ·
  **Python-driven:** no
- **Sources:** TST-013, TST-003 (the dead parts)
- **Location:**
  - **dead by construction:** `expression/evaluate/walk.rs:613-628` (the
    Real arm of `arithmetic`), `:493-494`, `:512-513`, and `:497, 644, 700`
    with `evaluate/error.rs:146, 223` (`EvaluationError::NumberAsBoolean`);
  - **probably unreachable, written as errors:**
    `types/checking/checker.rs:415-420`, `:894-901`, `:1129-1137` and
    `:1170-1178`.

**Issue:**
- **Dead variant.** `Prepared::evaluate` runs the Boolean screen first
  (`evaluate.rs:320-329`), so `NumberAsBoolean` is never produced: the
  coverage probe's `!x`, `all(x, p)` and `piecewise(x -> 1, 0)` all return
  `IllTyped`.
- **Dead arms.** `combine` (`walk.rs:268-289`) routes every Real and
  Boolean case away from the three dead arms.
- **Errors that cannot happen.** Every integral width has a float, so
  "no real float for bit width" cannot happen.

**Fix:**
- Delete `NumberAsBoolean`, or make its sites `unreachable!("screened")`.
- Have `arithmetic` take integer slices.
- Back the checker arms with a `const` assertion over `CoreDataType`, then
  `expect`.
- Leave trivial `Debug`, `From`, getter and `const` code out of any
  coverage gate: gate on per-module region coverage with those excluded,
  or accept about 93% as the ceiling.

**Behavior change:** a public error variant disappears. This is
source-breaking only.

---

## Triage

### (a) Obvious fixes to approve in bulk (24)

These are bugs with a clear fix and no user-visible semantic choice.
Behavior changes are listed where they exist.

| ID | Fix | Behavior change |
|---|---|---|
| F2-001 | DAG-linear key and memoized checker | Key text and system order change (Python-visible); regenerate the corpus |
| F2-002 | Separate advance and read caps; `try_allocate_id` fails at the readable limit (Rust and Python) | Payload ids in `[2^62, 2^63)` rejected |
| F2-003 | `__traverse__`/`__clear__`, with Python objects in visible slots | Cycles are freed |
| F2-009 | Docs.rs metadata, `doc(cfg)`, fixed link, per-crate CI, doc drift | none |
| F2-010 | Bounded node rendering in errors, lane index, `occurrence_count` | Long messages truncated |
| F2-012 | Checked lane count, `try_reserve`, per-chunk broadcast slicing | Errors instead of aborts |
| F2-013 | Iterative `Pattern` drop and `Debug`; depth caps on `Value` and the binding readers | Deep inputs refused |
| F2-014 | Writer thread, deadline-bounded waits, process group, `:print-success false` | Timeouts honored |
| F2-015 | `sanitize` maps control characters | Symbol text for such hints |
| F2-016 | Lift negative powers as divisions | Simplified quotients print as `/` |
| F2-017 | Negated literal resolved as one literal | `-(5): uint8` refused, `-(128): int8` accepted |
| F2-019 | Sweep keyed by a label that can name built-ins; non-vacuous test | Return type changes |
| F2-020 | `add_symbol` checks descendants; `build` order; decode assignments through `restore` | Some `add_symbol` calls refused; invalid assignment payloads refused |
| F2-022 | Reflexive default; symmetric equivalence | Default extensions unify with themselves |
| F2-023 | Interrupts outrank; stop calling Python after the first error; no cached fallback key | `KeyboardInterrupt` always wins |
| F2-024 | `__reduce__`/`__setstate__` for poset and lattice | Restores pickling |
| F2-031 | One `ScopedStack` guard for all six stacks | none |
| F2-035 | Restrict and deduplicate capture renaming | Same handle when nothing is replaced |
| F2-038 | Memoize SymPy lift and rebuild, and shape substitution; in-set candidates for permutations | none |
| F2-039 | Versioned prelude name or hash check | Mismatched preludes fail to load |
| F2-041 | Read diagnostics through the report | Signature change |
| F2-042 | Tie-break colliding keys by equivalence group | Canonical system order |
| F2-044 | Python reads before borrows | Re-entrant reads work |
| F2-045 | `as_tuple` for `Decimal`; `to_bytes` for ints | Huge `Decimal`s handled or refused at once |

**Sequencing notes:**
- F2-001 and F2-011(a) both want a canonical DAG rendering. If F2-011
  takes option (a), build the key on the same encoder.
- F2-038's substitution memo assumes F2-018's fix.
- F2-031 and F2-033 share the guard helper.
- Land the F2-001, F2-011 and F2-036 corpus regenerations together.

### (b) Findings that need a user decision (10)

1. **F2-004 — Unify the extension traits?**
   - **Options:**
     - (i) The full unification: one `ForeignPart` supertrait, one handle
       and equality convention, fallible hooks (`Result<_, BoxError>`)
       except `eq`/`hash`, a context passed to the custom hooks, and
       provided methods in place of `Option<Result>`.
     - (ii) Only the fallible hooks, which removes the silent wrong answers
       and most binding side channels.
     - (iii) Keep the traits, and document the failure-swallowing
       semantics.
   - **Recommendation:** (i), as one breaking batch before the crates.io
     release. The traits cannot change cheaply after publishing, and (ii)
     alone would still leave four shapes.
2. **F2-005 — Move `SympySimplifier` out of the core, to take pyo3 out of
   its public API?**
   - **Options:**
     - (i) Move the SymPy backend into `fhy-core-py`, its only user.
     - (ii) Move it into a sibling crate, `fhy-core-sympy`, that
       implements `fhy_core::solver::Simplifier`.
     - (iii) Keep it, and hide the pyo3-typed steps behind
       `#[doc(hidden)]` or a binding-only module with no stability
       promise.
   - **Recommendation:** (i). Nothing in the core calls it, and nothing
     links `fhy-core` yet (first audit, decision 5). The core then drops
     pyo3, the `sympy` feature, a test target and the CONTRIBUTING
     caveats. Choose (ii) instead if a Rust user of SymPy simplification
     is expected. Either way, make `IntervalProfile` and `Value`
     `#[non_exhaustive]` and give `SimplifyContext` a limits field.
3. **F2-006 — Split `ParamError`?**
   - **Options:**
     - (i) Split it into `DomainError`, `ParamBuildError`,
       `AssignmentError`, `IntervalError` and a question-level
       `ParamError`.
     - (ii) Keep one enum, and document per method which variants it can
       return.
   - **Recommendation:** (i), before the release. It follows
     CONTRIBUTING's "one type per family" rule, and the Python classes are
     unchanged.
4. **F2-011 — Should equal expressions serialize to equal bytes?**
   - **Options:**
     - (a) Canonical encoding: hash-cons by structural digest while
       encoding.
     - (b) Document that the encoding depends on sharing, and stop using
       V2 payload text as W-12's ordering key.
   - **Recommendation:** (a). It makes W-12's key and text comparison
     sound, and shrinks payloads. The cost is regenerating the golden
     corpus, and matching any Python-side V2 expression writer.
5. **F2-018 — What should unify do when a substitution is refused?** Today
   it continues with the unsubstituted expression.
   - **Options:**
     - (i) Add `UnificationError::Substitution(PiecewiseError)`, and
       propagate it, which makes `bind_placeholder` and
       `substitute_template` fail in this case.
     - (ii) Keep the fallback, but have the occurs check walk the binding
       graph, which prevents the cycle. `substitute_template` still
       returns a partly substituted type.
   - **Recommendation:** (i). An error is better than a silently wrong
     type. Adding (ii) as well is cheap defense in depth.
6. **F2-021 — Should domain-level methods enforce their domain's
   restriction (TYP-006)?**
   - **Options:**
     - (a) Fold `implied_constraints` into every `ParamDomain` procedure,
       and make `is_value_admissible` and `is_value_set_subset` respect
       it.
     - (b) Make the side-taking procedures `pub(crate)`, and expose the
       questions only on `Param`.
   - **Recommendation:** (a). These are public crates.io entry points, and
     `CustomDomain` implementors reuse them. An answer that is wrong
     unless the caller knows a hidden rule is worse than a slightly larger
     surface.
7. **F2-034 — What are the NaN semantics of `min`/`max` (EXP-004)?**
   - **Options:**
     - (a) NaN-propagating, as NumPy's `maximum`: `max(a, b) = a if (a > b
       or a != a) else b`, and `abs(x) = x if x > 0 else -x`, which gives
       `abs(-0.0) = 0.0`.
     - (b) Keep Python's faithful, order-dependent bodies, and document and
       pin them.
   - **Recommendation:** (a). It makes `max` and `min` commutative, and
     matches the NumPy semantics the evaluator follows everywhere else.
     The Z3 and SymPy lowerings follow automatically.
8. **F2-036 — Canonical float text on the wire, and should D-7 change?**
   - **Options:**
     - (i) Make the float and decimal wire decoders canonical: parse,
       re-render, and refuse a mismatch.
     - (ii) Revise the signed decision D-7 to the shortest round-trip text,
       with an exponent outside `[1e-5, 1e16)`.
   - **Recommendation:** (i) now, since no writer produces non-canonical
     text. (ii) needs the maintainer's sign-off; if it is wanted, land it
     in the same corpus regeneration as F2-001 and F2-011.
9. **F2-040 — Keep the mixed int/real equality hazard?**
   - **Options:**
     - (a) Drop hazard 5, and let the lowering decide.
     - (b) Keep it, and rewrite the doc to give the Python-parity reason.
   - **Recommendation:** (a). The evaluator and the lowering agree, and
     the first audit's decision 3 lets Rust define behavior where it
     replaces the Python implementation. Record it with D-S8-5's
     follow-up.
10. **F2-043 — Free-threading `gil_used`, and NumPy input isolation.**
    - **Options for free-threading:**
      - (i) Declare `gil_used = true` now, and record the decision.
      - (ii) Support 3.14t: add a CI job, add threaded stress tests of the
        registries, the default solver and the non-frozen classes, and fix
        F2-044 first.
    - **Options for NumPy inputs:**
      - (i) Document that inputs must not be mutated during the call,
        which is rust-numpy's contract.
      - (ii) Copy the inputs before detaching.
    - **Recommendation:** `gil_used = true` until a 3.14t job exists, and
      document the NumPy contract. A copy costs bandwidth on every call,
      to protect a case that the old evaluator also got wrong.

### (c) Test-only work (7)

- **F2-025** custom hooks, **F2-026** evaluator and checker shapes,
  **F2-027** solver properties, **F2-028** decision rules, **F2-029**
  error text and weak assertions, **F2-030** binding and stub, **F2-046**
  serde properties.
- **Also, from Current state (not numbered findings):**
  - Filter or assert the 64 V1 `DeprecationWarning`s in the tests that
    read V1 on purpose.
  - Check whether the Python suite's 99% stall reproduces on a normal
    build, with and without xdist, and account for the roughly 57 tests
    that may not have run.
- Several (a) fixes carry their own regression tests: F2-001, F2-013,
  F2-014, F2-019, F2-020 and F2-024.

### (d) Deferrable idiom and cleanup (6)

- **F2-007** API conventions, **F2-008** exact arithmetic, **F2-032** std
  traits, **F2-033** binding boilerplate, **F2-037** walk nits, **F2-047**
  dead code.
- F2-007 and F2-032 are breaking, so batch them into the pre-release API
  cleanup with F2-004 to F2-006. F2-008 should land before the next
  change to `Decimal` or the literal model.

---

## Noted, not findings

- **Committed choice in `Alternatives`** (`bugs.md`, plausible;
  `expression/pattern/matching.rs:619-626`). The first alternative that
  matches wins, with no backtracking when a later non-linear capture
  fails. So `Binary(Add, Alt[Capture(c, _), _], Capture(c))` does not
  match `1 + 2`. This was not compared with the pre-S5 Python matcher, so
  it may be intended. Check it before calling it a bug.
- **Intended but surprising behaviors** (`bugs.md` L-1 to L-6):
  - a 100,000-level expression cannot be pickled or deep-copied
    (`RecursionError`), unlike its other operations;
  - `repr` hides the literal kind (`1`, `1.0` and `Decimal("1.0")` all
    print `LiteralExpression(1)`);
  - set constraints are type-strict while their expressions are numeric;
  - `REAL`-sorted natives can fold to integers;
  - scalar and array transcendentals differ in the last ULP (N-S9-2);
  - SymPy's own simplification defects show through.
- **EXP minor notes:**
  - `BuiltinFunction` has seven hand-written parallel tables; one
    declarative table would finish F-025's intent;
  - `floor(9223372036854775807)` folds to `…808`, but evaluation refuses
    it; a note on `Evaluator::fold` would help;
  - identity-based change detection re-reports equal rebuilds, as
    documented.
- **About 150 `PyOnceLock` import caches** pin module attributes, so
  monkeypatching or reloading those modules is silently ignored.
  CONTRIBUTING could say so (PY).

## Strengths

- **Clean layering and one path per item.** Every non-test `crate::` edge
  respects CONTRIBUTING's ten layers, and only `expression::passes`
  imports `pass`. There are no glob re-exports, and CI checks the rendered
  docs for duplicates (ARCH).
- **Hygienic features.** `cfg(feature)` appears in five places, all at
  module granularity. Every feature compiles and passes clippy alone, on
  stable and on 1.85. The `z3` backend exposes no z3 types. The package
  contents are exactly D-19's list (ARCH).
- **No global state beyond identity.** The solver, the registries, and the
  pass and verification registries are owned values (ARCH, SOL).
- **DAG-aware and iterative almost everywhere.**
  - Drop, equality, hashing, substitution, free identifiers, display,
    screen, wire, inlining, folding and evaluation keep explicit stacks
    and memoize by `NodeIdentity`.
  - A 200,000-level expression drops on a 1 MiB stack, and `Debug` is
    bounded.
  - F2-001, F2-010 and F2-038 are the exceptions (EXP, TYP, SOL).
- **Numerics that match references.**
  - Integer kernels report overflow, division by zero and negative
    exponents per lane.
  - The real divmod kernel matched NumPy 2.4.6 on every probed IEEE edge.
  - Scalar and array evaluation agreed on 30,400 generated cases, across
    the chunk boundary (EXP, BUG).
- **Sound decisions against brute force and real z3.**
  - Satisfiability, implication and validity held on about 4,400 queries.
  - Floor division and modulo held on 88 cases, with 0 mismatches.
  - Interval arithmetic is exact with `BigInt`, and its hull held on 2,000
    cases.
  - Param feasibility and subset gave no unsound answer in 300 cases.
  - Lattice meet and join held on 87,437 pairs, and finite set algebra
    matched set semantics (SOL, TYP, BUG).
- **A careful SMT lowering.** It uses floor semantics for every divisor
  sign, exact rationals, `to_real` promotion, and the narrowest logic,
  which z3 enforces. Its scripts are deterministic, and quoting prevents
  injection (SOL).
- **Lawful literals and consistent hashing.** Literal `==` and `hash`
  handle NaN, `-0.0`, decimals and big ints, in Rust and through Python.
  `Hash` agrees with `Eq` across the type and symbol-table values (EXP,
  TYP, PY, BUG).
- **A defensive wire decoder.** It refuses forward and self references,
  unused nodes, duplicate keys, bad number grammar and out-of-range ids,
  and its nesting depth is constant. V2 round trips agree over 269 corpus,
  foreign and random cases (EXP, BUG).
- **Conventional errors.** Every public error is `#[non_exhaustive]` or has
  private fields. `Display` is one lowercase line and never repeats
  `source()`. No reachable `unwrap` was found in `types`, `param`,
  `constraint` or `symbol_table` (ARCH, EXP, TYP).
- **An idiomatic pass framework.**
  - A guarded lifecycle, owned `Send` pipelines checked at compile time,
    and an identity-keyed analysis cache that pins its nodes.
  - A panic-safe `Detachment` (SOL).
- **Binding discipline.**
  - No `unsafe`, and clippy is clean.
  - Registries and the default solver swap immutable snapshots and drop
    the old one after unlocking.
  - Three stacks are guarded, and hook objects expire.
  - `KeyboardInterrupt` passes through every P3 hook, and subclasses
    pickle with their `__dict__`.
  - Every static documents its lock rule (PY).
- **Strong tests.**
  - 93.7% of lines and 91.5% of regions with every feature, and 90.6% of
    the binding. Seven modules are at 100% of lines.
  - The brute-force oracles are real, not re-implementations.
  - 264 precise `expect_err` calls, and no test without an assertion.
  - Golden replay of 89 Python-written V2 texts, byte for byte.
  - Deep-tree, small-stack and fresh-process targets.
  - 21 Python interface suites naming every `_rs` class (TST, SOL, TYP).
- **Honest rustdoc on costs.** Per-occurrence display, the limits of
  identity-based change detection, and pattern recursion depth are all
  documented (EXP).

---

## Resolutions (the maintainer, 2026-09-27)

**Group (b) decisions:**

| Finding | Resolution |
|---|---|
| F2-004 | (i) Full unification: one `ForeignPart` supertrait, one handle and equality convention, fallible hooks (`Result<_, BoxError>`) except `eq`/`hash`, a context passed to the custom hooks, and provided methods in place of `Option<Result>`. Done as one breaking batch before the crates.io release |
| F2-005 | (i) Move the SymPy backend into `fhy-core-py`. The core drops pyo3 and the `sympy` feature. Also make `IntervalProfile` and `Value` `#[non_exhaustive]`, and add a limits field to `SimplifyContext` |
| F2-006 | (i) Split `ParamError` by family: `DomainError`, `ParamBuildError`, `AssignmentError`, `IntervalError`, and a question-level `ParamError` |
| F2-011 | (a) Canonical encoding: hash-cons by structural digest while encoding |
| F2-018 | Both: add `UnificationError::Substitution` and propagate it, **and** make the occurs check walk the binding graph |
| F2-021 | (a) Every domain-level procedure enforces the domain's own restriction |
| F2-034 | (a) NaN-propagating `max`/`min`/`clamp`/`relu`/`leaky_relu`, as NumPy's `maximum`; `abs(-0.0) = 0.0` |
| F2-036 | (ii) Canonical float and decimal decoders, **and** a D-7 revision: shortest round-trip text, with exponent form outside `[1e-5, 1e16)`. The corpus regeneration lands together with F2-001's and F2-011's |
| F2-040 | (a) Drop the mixed int/real equality hazard. **Revised by the maintainer on 2026-09-27:** drop the hazard except for set-constraint residuals, whose membership is type-strict (option 1 of the Track D notes, N-D1, in `docs/design/rust-port-fixes.md`) |
| F2-043 | `gil_used = true` until a free-threaded CI job exists, and document the NumPy contract that inputs must not be mutated during a call |

**The other groups:**
- **(a)** All 24 approved as written, with their listed behavior changes.
- **(c)** All approved, including filtering the 64 V1 `DeprecationWarning`s and checking whether the 99% xdist stall reproduces.
- **(d)** All six done in the pre-release breaking batch, with F2-004 to F2-006.

**Also settled:**
- **S17's slower benchmark rows.** V2 decoding of a literal-heavy tree (2.10) is optimized in the fix work, on top of F2-011's encoder rewrite. The other four rows (1.11 to 2.05; the 2.05 row is the cost of N-S17-2 (a)) are accepted.
- **V1 removal.** The V1 wire format is removed in **0.3.0**, not 0.4.0. The deprecation texts and docs change to say so.
- **Committed choice in `Alternatives`.** Compare it with the pre-S5 Python matcher first. Fix it only if Python backtracked.
