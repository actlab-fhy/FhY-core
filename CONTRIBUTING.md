# Contributing to *FhY* Core

This guide covers the development setup, the test and CI lanes, and the
conventions the Rust core and its Python binding follow. Read the sections
relevant to your change; the Rust architecture sections matter for any
change under `rust/`.

- [Setup](#setup)
- [Repository layout](#repository-layout)
- [Development commands](#development-commands)
- [Testing](#testing)
- [Continuous integration](#continuous-integration)
- [Rust architecture](#rust-architecture)
- [Python style](#python-style)
- [Pull requests](#pull-requests)
- [Releases](#releases)

## Setup

Requirements: [uv](https://docs.astral.sh/uv/) and
[rustup](https://rustup.rs/). `rust-toolchain.toml` pins the toolchain CI
lints with; the crates themselves build on 1.85 or newer.

```bash
git clone https://github.com/actlab-fhy/FhY-core.git -b dev
cd FhY-core
uv sync                          # editable install, dev group, builds fhy_core._rs
uv run pre-commit install
uv run pre-commit run --all-files
```

## Repository layout

| Path | Contents |
| :--- | :--- |
| `src/fhy_core/` | Python package. `_rs.pyi` is the hand-written stub of the extension. |
| `rust/fhy-core/` | Pure-Rust core library. No PyO3, no Python at build or test time. |
| `rust/fhy-core-py/` | PyO3 bindings, as an `rlib`. Also holds the SymPy simplifier backend. |
| `rust/fhy-core-ext/` | The `cdylib` maturin builds into `fhy_core._rs`. Calls `fhy_core_py::register`. |
| `rust/example-aggregate/` | Test-only aggregate extension (see [One extension module per process](#one-extension-module-per-process)). Not published. |
| `rust/fhy-core/tests/golden/` | Golden corpora and the Python scripts that record them. |
| `tests/` | Python tests. `tests/strategies/` holds the shared Hypothesis strategies. |
| `benchmarks/` | pytest-benchmark suites for the public API's hot paths. |
| `noxfile.py` | All developer sessions. |

## Development commands

### Nox sessions

Every session runs through uv. Sessions parametrized over Python cover
3.11 to 3.14; uv installs missing interpreters.

| Command | What it does |
| :--- | :--- |
| `uv run nox` | Default sessions: `lint`, `type_check`, `tests`, `coverage` |
| `uv run nox -s lint` | `ruff check` and `ruff format` |
| `uv run nox -s type_check` | `mypy --strict`, plus `ty` (advisory while it is in preview) |
| `uv run nox -s tests-3.12` | Test suite under coverage for one Python version |
| `uv run nox -s tests_minimal` | Test suite without sympy, z3-solver and NumPy |
| `uv run nox -s coverage` | Combines the `.coverage.*` files left by `tests`. Skips on a clean tree. |
| `uv run nox -s property` | Property suite under the `thorough` profile. Opt-in. |
| `uv run nox -s golden_expanded` | Large random golden corpora replayed by the Rust tests. Needs `cargo`. Opt-in. |
| `uv run nox -s benchmark-3.12` | Benchmarks. Opt-in. |
| `uv run nox -s mutation -- <module>` | Mutation testing of one module. Opt-in. |

### Rust

```bash
cargo check -p fhy-core            # core only, no Python
cargo test --workspace --locked --all-features
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked
```

`fhy-core-py`'s tests embed Python and need extra environment; see
[Rust tests](#rust-tests).

### Rebuilding the extension

`uv sync`, and any `uv run`, rebuilds `fhy_core._rs` when a cache key
changes. The keys cover `rust/**/*.rs`, every `Cargo.toml` and
`Cargo.lock`.

```bash
uv sync                                  # rebuild after a Rust edit
uv sync --reinstall-package fhy-core     # force a rebuild
uv run python -c "import fhy_core._rs as rs; print(rs.__version__)"
```

For fast incremental debug builds, install maturin 1.9.4 or newer
(`uv tool install maturin`) and run `uv run --no-sync maturin develop --uv`.
Use `uv run --no-sync` for the commands that follow; a syncing `uv run`
replaces the `maturin develop` build with uv's own.

`import fhy_core` imports the extension first. It raises `ImportError`,
naming the cause, when the extension is missing, was built for another
Python, fails to import, or is stale. Stale means its `__version__` differs
from the installed package's. The version is set once, in the workspace
`Cargo.toml`. maturin derives the Python version from it by PEP 440
normalization, so a Cargo `0.3.0-rc.1` extension matches the `0.3.0rc1`
package.

### Do not rebuild while tests run

The install is editable, so `uv sync` and every nox session write
`src/fhy_core/_rs.*.so` in the source tree. maturin deletes the old file
and writes the new one in place. A process that imports the package during
the write, such as a starting xdist worker or a test's subprocess, loads a
partial file and dies with `SIGBUS`. The symptom is a test that fails once
and passes on rerun, often reported as `worker 'gwN' crashed`. Run the
Python gates one after another, or give each concurrent run its own
checkout.

## Testing

### Python tests

```bash
uv run pytest                        # full suite, parallel via xdist
uv run pytest tests/test_identifier.py
```

A test that reaches a solver backend is marked `z3` or `sympy`, by what it
reaches rather than what it imports. It is skipped when that package is
missing. `tests_minimal` installs neither package, and an unmarked test
that reaches a backend fails there.

The `*_rust_binding.py` suites test what each binding adds over the Rust
core: class structure, argument checks, payload shapes, pickling. The other
suites test the behavior of the Python API.

### Property-based tests

Property tests use [Hypothesis](https://hypothesis.readthedocs.io/). A
property states a rule that holds for every input in a space, and
Hypothesis searches for a counterexample and shrinks it to a minimal one.
Under the default profile a failure is saved in `.hypothesis/` and replayed
first on the next run.

Much of this codebase is symbolic: expression trees, the SymPy and z3
bridges, interval arithmetic, constraint systems, type lattices. The
interesting inputs are combinations (which operation nests under which,
which sorts the operands have, whether a bound is inclusive), and
hand-picked examples miss them. Properties found, among others:

- a piecewise with a bare Boolean identifier as a condition lost its later
  branches when lowered to SymPy, so a constraint reported `VIOLATED` for a
  satisfying assignment;
- `(b ? 2 : 0) % -6` simplified to `0`, because SymPy treats an even
  expression divided by 6 as an integer;
- two real bounds that differ below float precision were compared after
  rounding, so a non-empty interval was rejected.

#### When to write one

Write a property only when the input space is combinatorial and there is an
independent oracle: a reference evaluator, brute-force enumeration over a
small domain, an inverse to round-trip through, or an algebraic law. A
property that computes the expected value the same way the code does proves
nothing. If ten hand-picked inputs would convince a reviewer, write ten
examples. Exhaustive parametrized tables over a small finite domain also
stay as examples.

Examples remain the right tool for pinning an exact error message or
exception type, a golden value (serialized shape, `repr` text), and for
readable usage stories.

#### Writing one

Put properties in `test_<unit>_properties.py` next to `test_<unit>.py`.
Mark the module with `pytestmark = pytest.mark.property`. Call
`pytest.importorskip("hypothesis")` immediately after `import pytest` and
before importing `hypothesis`, `tests.strategies` or `fhy_core`, because
the `tests` lane does not install the `property` group. Mark properties that
reach z3 with `pytest.mark.z3`.

```python
"""Hypothesis property tests for `format_comma_separated_list`."""

import pytest

pytest.importorskip("hypothesis")

from hypothesis import example, given
from hypothesis import strategies as st

from fhy_core.utils.str_utils import format_comma_separated_list

pytestmark = pytest.mark.property


@example(items=[])
@example(items=[7])
@given(items=st.lists(st.integers()))
def test_format_comma_separated_list_separates_each_pair_of_items(
    items: list[int],
) -> None:
    """Test the output holds one separator between each pair of items.

    Oracle: counting separators, independent of how the function joins.
    """
    result = format_comma_separated_list(items, add_space=True)
    assert result.count(", ") == max(len(items) - 1, 0)
```

Rules:

- Name the oracle in the docstring. Use `@example` for inputs that must run
  every time; they run before any generated input.
- Reuse the strategies in `tests/strategies/` (identifiers, literals,
  expressions, params, constraints, types, orders, serializables) and
  extend them when needed. Import them only from property modules, never
  from a `conftest.py`, which must import without `hypothesis`.
- When you add a shape to a strategy, add a reachability test to
  `tests/test_strategies_properties.py` that uses `hypothesis.find` to show
  the strategy produces it. `--hypothesis-show-statistics` shows how many
  inputs a property actually tried.
- Generate valid input by construction. Do not use `assume()` or `.filter()`
  to discard invalid draws, and do not suppress health checks; a firing
  health check is a strategy bug. To exclude a shape, add an explicit
  strategy parameter and say why in the docstring.
- When an example test compares the code to an oracle on hand-picked
  inputs, convert it: write the property, pin each old input with
  `@example`, and delete the example test in the same commit.

#### Profiles

| Profile | Inputs | Notes |
| :--- | :--- | :--- |
| `dev` (default) | 25 | Random, with the example database |
| `thorough` | 400 | Derandomized, no database. The release gate. Set by `nox -s property` or `HYPOTHESIS_PROFILE=thorough`. A failure prints a `@reproduce_failure` blob. |
| `mutation` | 25 | Derandomized, no database, so every mutant sees the same inputs. Set by `nox -s mutation`. |

`thorough` draws the same inputs every run. Before a release, also run the
suite under a few seeds, e.g. `uv run pytest -m property
--hypothesis-seed=1234`.

No profile has a deadline; under xdist, scheduler contention trips
deadlines more often than test cost does. To limit an expensive test (a z3
call, a large tree), use `@cap_max_examples(N)` from
`tests/strategies/settings.py`, or assign `cap_max_examples(N)` to a state
machine's `TestCase.settings`. It runs `min(profile, N)` inputs. Do not use
a bare `@settings(max_examples=N)`: it replaces the profile's count, so an
`N` above 25 raises the count under `dev` and `mutation`.

#### When a property fails

1. Reproduce the shrunk `Falsifying example` as an example test in
   `test_<unit>.py`. PRs into `dev` do not run properties, so this example
   is what guards the fix there.
2. Pin the same input on the property with `@example(...)`.
3. Fix the code. If the bug is upstream (SymPy, z3) and cannot be worked
   around, add the input as a separate example test marked
   `@pytest.mark.xfail(strict=True, raises=...)` with a reason that names the
   upstream issue. Never mark the property itself `xfail`: Hypothesis stops
   at the first failing `@example`, so the property would never generate
   anything.

#### Hazards

- Function-scoped fixtures are not reset between Hypothesis inputs. Never
  use `function_registry_snapshot` in a `@given` test.
- `Expression.__bool__` raises. Keep expressions out of truth contexts in
  strategies and reference evaluators.
- Logical connectives and piecewise conditions must be Boolean-sorted.
  Generate sort-aware trees and bind Boolean identifiers to `bool` values.
- Strategies use `mock_identifier`, not `Identifier`. Mocks compare and hash
  by id, so pools must not overlap: integer pools start at id 10000 and
  Boolean pools at 15000 (`tests/strategies/identifiers.py`).
- `@example` cannot pin values drawn through `st.data()`. Use a
  `@st.composite` strategy for dependent draws.
- SymPy has global random state (`sympy.core.random`). A test that reseeds
  it must restore it.

### Golden corpora

The Rust equivalence tests replay JSON corpora in
`rust/fhy-core/tests/golden/`.

- `generate_*.py` scripts record a corpus from the Python implementation.
  Today these cover `interned` and serialization (V2 texts Python writes
  and Rust must read and write back byte-identically).
  `tests/test_golden_corpora.py` reruns each generator in a fresh
  interpreter and fails, printing a diff and the regeneration command, if
  the committed corpus differs outside its `provenance` block or is missing.
  The `golden-corpora` pre-commit hook runs the same check on commits that
  touch `src/fhy_core/`, the golden directory, or that test.
- `record_*.py` scripts record from an external oracle (MOGA-VM on
  `fhy_core` 0.1.8) and are not rerun by the drift check.

Generators are linted and type-checked with the package. After changing
behavior a corpus records, regenerate and commit it.

Each generator can also write a large random corpus, which an ignored test
replays from a path in an environment variable. `nox -s golden_expanded`
does both. A new generator needs an entry in `EXPANDED_GOLDEN_CORPORA` in
`noxfile.py`.

### Rust tests

`fhy-core`'s integration tests are one binary, `tests/it/`, with a module
per crate module. Shared helpers live in `tests/it/support` at
`pub(crate)`, so an unused helper triggers a dead-code warning. A helper
used by one module stays in that module.

A test that needs a fresh process, because it moves process-global state
further than other tests tolerate, is its own target `tests/<name>.rs` with
exactly one `#[test]` and a comment explaining why. Add it to the target
list the CI `rust` job checks. The only one today is `id_cap_decode`, which
moves the id counter to `ADVANCE_CAP`. Nothing re-executes a test binary;
a test that needs a process without SymPy is a Python subprocess test
(`test_missing_sympy_reports_unavailable`).

`cargo test -p fhy-core` needs no Python. `fhy-core-py`'s tests link
libpython, embed an interpreter, and import SymPy (the SymPy backend's
stories) and NumPy (`convert::numpy`), and fail with instructions when
they cannot. They need:

- `PYO3_PYTHON` pointing at a Python with a shared libpython and `sympy` and
  `numpy` installed;
- `PYTHONPATH` set to that Python's `site-packages`, since an embedded
  interpreter ignores `pyvenv.cfg`;
- `LD_LIBRARY_PATH` set to libpython's directory if the loader cannot find
  it.

Distribution Pythons often lack a shared libpython and fail with
`unable to find library -lpython3.11`. uv-managed CPython has one:

```bash
G=$PWD/target/gate-python
uv python install --no-bin --install-dir "$G/pythons" 3.11
uv venv --python "$G"/pythons/cpython-3.11*/bin/python3.11 "$G/venv"
VIRTUAL_ENV="$G/venv" uv pip install sympy numpy
export PYO3_PYTHON="$G/venv/bin/python"
export PYTHONPATH="$G/venv/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="$(echo "$G"/pythons/cpython-3.11*/lib)"
```

### Mutation testing

Mutation testing checks that the suite notices small source changes. The
`mutation` session runs [cosmic-ray](https://cosmic-ray.readthedocs.io/)
on one Python module, named relative to `fhy_core`:

```bash
uv run nox -s mutation -- symbolic.param.core
```

The session writes the cosmic-ray config itself (`_MUTATION_CONFIG` in
`noxfile.py`), runs `cosmic-ray baseline` to stop early if the unmutated
suite fails or overruns the timeout, then mutates, and writes
`mutation-report.html`. It selects the `mutation` Hypothesis profile.
cosmic-ray counts a mutant whose tests time out as killed, so the 120 s
timeout is kept at about four times the suite's serial run time; raise it
if the suite grows.

Only Python source is mutated. A module that only re-exports a `_rs` class
yields no mutants; test the Rust implementation with Rust tests instead.

cosmic-ray edits the source file in place and restores it after each
mutant. While a run is active, do not run other tests, commit, or edit
`src/`. A run killed mid-mutant can leave the mutated file behind, so check
`git status` afterwards.

### Benchmarks

`benchmarks/` holds [pytest-benchmark](https://pytest-benchmark.readthedocs.io/)
suites, grouped by concept, using the public API only, so one benchmark
measures a class before and after a port. The `benchmark` session is
opt-in and not part of CI. Results go to `.benchmarks/<python>.json` and
`.benchmarks/storage/` (gitignored).

```bash
git switch --detach <before>
uv run nox -s benchmark-3.12
cp .benchmarks/3.12.json .benchmarks/3.12-before.json
git switch -
uv run nox -s benchmark-3.12
uv run --group bench pytest-benchmark compare --group-by=name --columns=median \
    .benchmarks/3.12-before.json .benchmarks/3.12.json
```

Machine load moves the numbers; compare back-to-back runs on one machine.

## Continuous integration

`.github/workflows/python-package.yml` runs on pull requests into `dev` and
`main`, on pushes to `main`, and on manual dispatch. `ci-ok` is the single
required check.

| Job | Runs | Checks |
| :--- | :--- | :--- |
| `style` | always | `nox -s lint`, `nox -s type_check` |
| `rust` | always | See below |
| `rust-msrv` | always | `cargo check --workspace --lib` and `cargo check -p fhy-core --all-targets` (no features, each feature alone) on the MSRV, `-D warnings` |
| `deny` | always | cargo-deny against `deny.toml` |
| `tests` | always | Ubuntu with 3.11 and 3.14 on PRs into `dev`; 3 OSes × 4 Pythons otherwise |
| `tests-minimal` | always | `nox -s tests_minimal` |
| `property` | PRs into `main`, dispatch | `nox -s property` |
| `golden-expanded` | PRs into `main`, dispatch | `nox -s golden_expanded` |
| `coverage` | after `tests` | Combined report to Codecov |

The `rust` job:

- `cargo fmt --check` and workspace `clippy -D warnings`. A workspace build
  unifies the binding's features into `fhy-core`, so it also runs
  `clippy -p fhy-core --all-targets` with no features and with each feature
  alone.
- Checks that `fhy-core`'s test targets are exactly `it` and
  `id_cap_decode`.
- `cargo test --workspace --locked --all-features`, then the solver tests
  without the `z3` feature, which drive the `z3` executable instead.
- `cargo doc` for `fhy-core` with default features and for the workspace,
  both `-D warnings`.
- Public-path checks: no glob re-exports, no re-export of an item that
  already has a public path, no item under two paths.
- `cargo package -p fhy-core`, a check of the package contents, and
  `cargo test` inside the unpacked crate, so tests pass on the crate as
  published.

docs.rs builds `fhy-core` with both features and `--cfg docsrs`. Mark each
new feature-gated item with
`#[cfg_attr(docsrs, doc(cfg(feature = "...")))]`.

The MSRV in `Cargo.toml` is a promise to consumers.
`.cargo/config.toml` makes the resolver prefer dependency versions that
build on it, and `cargo update` keeps `Cargo.lock` within it.

`deny` requires every dependency to have an allowed permissive license,
come from crates.io, and have no wildcard version requirement. RustSec
advisories show as warnings. Run it locally with `cargo deny check`
(`cargo install cargo-deny --locked`). A new license or an ignored
advisory goes in `deny.toml` with a reason.

Dependabot opens a weekly PR against `dev` for Cargo and GitHub Actions,
grouping minor and patch updates per ecosystem.

## Rust architecture

### Crates

- `fhy-core` is the library. It has no PyO3 dependency.
- `fhy-core-py` holds the PyO3 bindings as a library: `register(py, module)`
  adds every class, function and piece of module state to a given module.
  It also holds the SymPy simplifier (`solver::sympy`) and the Python class
  of the ground simplifier (`solver::ground`).
- `fhy-core-ext` is the `cdylib` maturin builds as `fhy_core._rs`
  (`manifest-path` in `[tool.maturin]`). It calls `register` and sets
  `__version__`.
- `rust/example-aggregate` is a test-only aggregate extension
  (`publish = false`).

The workspace declares `pyo3` once, because `pyo3-ffi` links `python` and a
build can hold only one copy. A port adds its types to `fhy-core` and their
bindings to `fhy-core-py`.

### Module layering

Each public item has exactly one public path, and every `pub use` is
explicit; CI rejects globs and duplicate paths. A module depends only on
the layers above it:

1. `identifier`, `interned`, `foreign` (the serialized form of parts other
   implementations define, and `BoxError`), `error` (errors shared across
   layers, such as `UnknownNameError`)
2. `described_tag`, `value_domain`, `provenance`
3. `diagnostic`, `op_attribute`
4. `tree`, `term`, `lattice` (independent of each other)
5. `expression` (with `expression::pattern`, `expression::builtins`) and
   `pass` (independent of each other)
6. `expression::passes` (depends on both); `solver` and `types` (depend on
   `expression`, not on `pass`)
7. `constraint` (depends on `solver`)
8. `symbol_table` (depends on `types`)
9. `param` (depends on `constraint`)
10. `stack`, `scope` (depend on nothing, including each other)
11. `search_space` (depends on `param`, `constraint`, `expression`; not on
    `pass`, `types` or `symbol_table`)

Layers 6 to 11 never depend on `pass`.

A module with submodules is `foo.rs` beside a `foo/` directory; there are
no `mod.rs` files. Never name a private module `core`: it shadows the
`core` crate.

### Module map

This table is the one place that maps Python paths to Rust paths. A port
adds its row.

| Python | Rust |
|---|---|
| `fhy_core.identifier` | `fhy_core::identifier` |
| `fhy_core.traits.interned` | `fhy_core::interned` |
| `fhy_core.diagnostic` | `fhy_core::diagnostic` |
| `fhy_core.provenance` | `fhy_core::provenance`; a Python `Provenance` subclass is a `Provenance::Custom` |
| `fhy_core.op_attribute` | `fhy_core::op_attribute` |
| `fhy_core.value_domain` | `fhy_core::value_domain` |
| `fhy_core.symbolic.symbol_type` | `fhy_core::expression` (`SymbolType`) |
| `fhy_core.symbolic.expression` (`core`, `errors`, `pprint`, `sort`) | `fhy_core::expression` |
| `fhy_core.symbolic.expression.builtins` | `fhy_core::expression::builtins` |
| `fhy_core.symbolic.expression.registry`, `passes.inline` | `fhy_core::expression::registry` |
| `fhy_core.symbolic.expression.passes.evaluate`, `passes.numpy`, `passes.native_lowering` | `fhy_core::expression::evaluate` |
| `fhy_core.symbolic.expression.pattern` (`core`, `rewrite`) | `fhy_core::expression::pattern`; the rule-applier pass is in `fhy_core::expression::passes` |
| `fhy_core.symbolic.expression.passes` | `fhy_core::expression::passes` |
| `fhy_core.symbolic.expression.passes.affine` | `fhy_core::expression` (`AffineForm`, `Expression::affine_form`, `Rational`) |
| `fhy_core.pass_infrastructure` | `fhy_core::pass`; tree traversal is in `fhy_core::tree` |
| `fhy_core.symbolic.solver`, `symbolic.expression.passes.z3` (lowering) | `fhy_core::solver` |
| `fhy_core.symbolic.expression.passes.sympy` (lowering, simplification, lifting) | `fhy-core-py`'s `solver::sympy`, a `fhy_core::solver::Simplifier` |
| `fhy_core.symbolic.solver.GroundSimplifier` (`SolverBackend.GROUND`, `GROUND_THEN_SYMPY`) | `fhy_core::solver::{GroundSimplifier, GroundWithFallback}`; the Python class is `fhy-core-py`'s `solver::ground` |
| `fhy_core.term` | `fhy_core::term`; the derived-equivalence engine, which reads Python dataclasses, is in the binding |
| `fhy_core.lattice`, `fhy_core.utils.poset` | `fhy_core::lattice` |
| `fhy_core.types` (`core`, `dispatch`) | `fhy_core::types`; `singledispatch` registration of Python-defined types stays in Python |
| `fhy_core.types.checking` | `fhy_core::types::checking`; the body-check pass stays a Python `CompilerPass` over it |
| `fhy_core.symbolic.constraint` | `fhy_core::constraint`; Python-defined constraints and members only Python can compare go through the binding's adapters |
| `fhy_core.symbolic.param` (`values`, `domains`) | `fhy_core::param`; Python-defined domains and values only Python can compare or order go through the binding's adapters |
| `fhy_core.symbol_table` | `fhy_core::symbol_table`; the abstract `SymbolTableFrame` stays in Python |
| `fhy_core.utils.stack` | `fhy_core::stack`, for Rust users; the Python `Stack` is a separate implementation |
| `fhy_core.utils.scope` | `fhy_core::scope`, for Rust users; the Python `Scope` is a separate implementation |
| `fhy_core.search_space` | `fhy_core::search_space`; ported from MOGA-VM's `moga_vm.cir.space.core`, whose CIR-specific kinds implement its traits in MOGA-VM |

### One extension module per process

All Rust code that uses *FhY* Core's types compiles into one Python
extension module per process. The identifier counter and each `Interned`
type's `InternRegistry` are Rust `static`s, one per compiled copy of the
crate, and PyO3 creates a separate Python type per extension module. A
second extension linking the crate would issue colliding ids, keep
registries whose canonical values never match, and fail `isinstance`
against the first one's classes. A downstream package with Rust code
therefore depends on the crate as a Rust library and compiles into one
combined module. It never ships its own extension linking `fhy-core`.

#### The binding is a library

`fhy-core-py` is an `rlib`. `register(py, module)` refuses a module it has
already registered into. It is not a `cdylib` because a `#[pymodule]`
exports `PyInit_<name>`: linked into an aggregate, the crate would export
`PyInit__rs` from every aggregate, and one whose module is also named
`_rs` would not link. That is why the module itself lives in `fhy-core-ext`.

Its public surface for downstream crates:

| Module | Contents |
| :--- | :--- |
| `convert` | `…_from_python` / `…_to_python` for `Identifier`, `Expression`, `Type`, `Param`, `ParamAssignment`, `ValueDomain`, `OpAttribute`, `Diagnostic`, `ValidationReport` |
| `convert::numpy` | `require_numpy`; `NumpyValue` (`from_python`, `as_binding`, `to_array_value`, `as_scalar`), which borrows native-order `bool_`/`int64`/`float64` arrays and casts other admitted dtypes once; `NumpyKernels` (the 14 transcendental natives via NumPy ufuncs); `array_value_to_numpy`, `scalar_to_numpy`, `evaluation_error_to_python`. No `rust-numpy` type appears in signatures. |
| `convert::param` | `with_param_context(py, detach, question)`: runs a param question with the same `ParamContext` `fhy_core`'s own methods use (default solver, function-registry snapshot, logging observer), optionally detached from the interpreter, and re-raises any Python exception a hook raised |
| `convert::search_space` | Conversions for `fhy_core.search_space` objects; `register_variable_kind` and `register_alternative_kind` for downstream `Variable` and `Alternative` kinds |
| `util::python` | `Seed`, `ImportedAttr`, `cached_attr!` (exported at the crate root), `read_type_name` |
| `util::exceptions` | `ExceptionClass` (declared as a `static`), `unbox_py_err`, `boxed_error_to_py`, and the framework's exception classes (`SERIALIZATION_ERROR`, `FROZEN_MUTATION_ERROR`, ...) |
| `util::interned` | `IdentityCache<K>` and the `InternedMixin` helpers |
| `util::public_class` | `PublicClass::new`, `PublicClass::in_module` |
| `util::frozen` | `refuse_attribute_assignment`, `refuse_attribute_deletion` |
| `util::dataclass` | `compare_as_dataclass`, `is_same_or_equal`, `hash_value`, `collect_tuple`, `format_dataclass_repr`, `build_argument_type_error`, `read_str`, `OptionalArgument` |
| `util::serialization` | Payload readers: `read_payload_fields`, `PayloadFields::allowing_extra`, `read_constructor_fields`, `read_nested_value`, `read_nested_list`, `keep_fields`, `serialize_nested`, `is_serialized_dict`, `construct_from_decoded_fields` (and `..._reporting_overflow`) |
| `util::scoped` | `ScopedStack`, `ScopedGuard` |
| `util::pending` | Pending exceptions of infallible hooks: `record_pending_error`, `has_pending_error`, `with_pending_errors`, `capture_pending_errors` |
| `util::hook` | `ask`: calls a Python hook behind an infallible trait method, returning the fallback while an exception is pending |
| `util::frames` | `Frames<T>`: per-call context. A read with no frame records a `RuntimeError`, so a missed push is an error, not a silent default. |
| `util::integers` | `read_unsigned`, `read_unsigned_lenient`, `classify_unsigned`, `Reading`, `build_too_large_error` |
| `util::gc` | `Slot`, `Slots`, `collect_slots`, `traverse_locked`, `clear_locked`, `traverse_all` |
| `util::foreign` | `read_foreign`, `record_foreign_failure`, `RaisedError` |
| `util::testing` | Behind the `testing` feature; enable it only in `[dev-dependencies]`. Stand-in `fhy_core` modules for embedded-interpreter tests: `with_stand_ins`, `install_module`, `evaluate`, `define`, `entry`. |

Each item's rustdoc states its errors and panics. Search-space kind
resolvers have the shape of the core's
`fhy_core::search_space::wire::VariableResolverFn` /
`AlternativeResolverFn`, so the same function serves a pure-Rust program
through a `ResolverRegistry`. A class that keeps a Python-defined part
reads it inside `util::gc::collect_slots`, keeps the `Slots`, and visits
them from `__traverse__`, as `rust/example-aggregate` does.

If a downstream crate needs a conversion `convert` lacks, add it there as a
documented `pub fn` over the `pub(crate)` one. No `#[pyclass]` becomes
`pub`.

#### Aggregates

An aggregate is one `cdylib` per product. Its `#[pymodule]` calls
`fhy_core_py::register`, then each of its own crates' registration
functions. `rust/example-aggregate` is the template and the test; it
registers one class, `Tagger`, that takes and returns an `Identifier` and
an `OpAttribute`. A downstream aggregate must:

- name its classes' package explicitly (`module = "..."`);
- be a top-level module that imports no `fhy_core` Python code at import,
  because `fhy_core` imports it while `fhy_core` itself is importing;
- have its package import `fhy_core` before calling into the module, since
  the binding imports `fhy_core` modules on first use and doing so from
  inside an earlier call can deadlock on a once-initialized cache;
- depend on the same `fhy-core` source and version as `fhy-core-py`, since
  two sources are two copies of the statics.

#### Class identity

Every `#[pyclass]` sets `module = "fhy_core._rs"`, so `__module__`, `repr`,
`pickle` and qualified names are the same whichever native module holds
the class. The binding locates its own state, and the Python code locates
its classes, by that name. The loader therefore installs an aggregate as
`sys.modules["fhy_core._rs"]` and as `fhy_core._rs`.
`tests/test_composed_extension.py` checks `isinstance`, `pickle`, `repr`
and `__module__` against an aggregate named `_fhy_example_aggregate`.

#### The loader

`fhy_core._extension` picks the process's native module at import:

1. `FHY_CORE_NATIVE_MODULE`, if set, overrides everything.
2. Otherwise, the module named by entry points in the group
   `fhy_core.native` (`[project.entry-points."fhy_core.native"]`,
   `product = "module_name"`). Two entry points naming different modules
   raise `ImportError` listing both.
3. Otherwise, `fhy_core._rs`.

An aggregate reports the `fhy_core` version it contains as
`__fhy_core_version__` (set by `register`), and a mismatch is refused like
a stale `_rs`. A named module that fails to import raises `ImportError`; it
never falls back to `fhy_core._rs`. If `fhy_core._rs` already names a
different module, loading an aggregate raises `ImportError`.

`tests/test_extension.py` covers the loader without a build.
`tests/test_composed_extension.py` builds `rust/example-aggregate` with
`cargo build` into `target/composition-<python>` and checks, in a fresh
interpreter, that one counter, one registry per interned type, and one set
of classes serve both the aggregate and `fhy_core`.

#### Shipping an aggregate

There can be one aggregate per set of packages that may share a process,
so per-product wheels do not work once two products have Rust code. The
intended design is an umbrella distribution (working name `fhy-native`)
whose extension aggregates every `-py` crate of the stack, pinned to
matching releases, declaring the `fhy_core.native` entry point; products
depend on it through an optional extra. A `-py` crate pins
`fhy-core-py = "=X.Y.Z"` to the `fhy_core` release it installs with.
`fhy_core` keeps shipping its own `_rs`. Until the umbrella exists, a
single downstream product with Rust code may ship its own aggregate under
the entry point and hand over to the umbrella later. No aggregate release
packaging lives in this repository.

#### Free-threaded Python

`fhy-core-ext` declares `#[pymodule(gil_used = true)]`, so importing it on
3.13t or 3.14t re-enables the GIL with a `RuntimeWarning`. PyO3 0.29
assumes free-threading support otherwise, and the binding has not been
shown safe without the GIL: non-frozen classes raise borrow errors under
contention, the opaque value's ordering key runs Python inside a
`OnceLock` initializer, and several "never held across Python" invariants
were argued for the GIL build only. The declaration stays until a
free-threaded CI job exists and those points are checked.

Separately, the NumPy evaluator reads `float64` inputs in place with the
GIL released, so callers must not write an input array from another
thread during the call.

### Process-global state

Only two kinds of state are process-global: the identifier counter and each
`Interned` type's `InternRegistry`. Both are append-only: ids are never
reissued and canonical values are never replaced or removed. Everything
else (pass registries, run statistics, caches) is an owned value the
caller creates and passes. Where the Python API needs a shared instance,
the binding keeps it in the extension's module state. A new process-global
`static` with interior mutability needs the maintainer's agreement and an
entry in this section.

Ids `0..RESERVED_ID_COUNT` (65,536) are reserved for identifiers the crate
ships, each with a fixed id in a crate-private table. A new shipped
identifier takes an unused reserved id. The counter issues ids from 65,536
up to, but never at or above, `ID_CAP` (2^63). A decoded id below
`ADVANCE_CAP` (2^62) advances the counter past it. An id in
`ADVANCE_CAP..ID_CAP` decodes only if this process issued it. Any other id
is refused. No payload can push the counter past 2^62, so every fresh id
reads back.

#### Binding state

Write-once and append-only:

- one identity cache per Rust-backed interned class, mapping canonical keys
  to Python objects (append-only, like the registry);
- one write-once slot per Rust-backed class for the public Python class
  registered at import (`util/public_class.rs`);
- the shared empty `AlphaRenaming` returned by `AlphaRenaming.empty()`
  (`term/renaming.rs`). Derived-equivalence plans stay in the Python
  module's `_PLAN_CACHE`.

Shared registries for the Python API. Each is swapped whole on update, and
no lock is ever held across a call into Python:

| Registry | Storage | Append-only |
| :--- | :--- | :--- |
| Function registry (`register_function`) | `Mutex<Arc<_>>` of the core `FunctionRegistry` plus Python objects (`expression/registry/state.rs`). Built-ins are built once at import. | No; `set_registry_state_for_tests` replaces it |
| Default solver | `Mutex<Option<Py<Solver>>>` (`solver/state.rs`), set at import, replaced by `set_default_solver` | No |
| Verification registry | Module attribute `_rs._verification_registry`, a `Mutex<Arc<_>>` of the core `VerificationRegistry` (`pass/verification.rs`) | Yes |
| Search-space kinds | Module attribute `_rs._search_space_kinds`, one map per family from kind to class and functions (`search_space/kinds.rs`) | Yes |

A search-space kind and a class registered with `register_serializable`
never share a type id, because decoding consults kinds first. Whichever
registration comes second is refused: registering a kind checks
`fhy_core.serialization`'s registry (from `sys.modules`, before taking the
lock) and raises `ValueError`; `register_serializable` checks
`get_search_space_kind_class` and raises `SerializationError`.

Per-call, thread-local state. Every frame is pushed through a
`ScopedStack` guard (`util/scoped.rs`) that pops it on drop, including on
unwind, so a panic (raised as `PanicException`) never leaves a stale frame:

- per-call object tables (`expression/pattern/objects.rs`), mapping the
  Rust nodes a match or rewrite reaches to their Python objects;
- scopes of pass runs, pipeline runs and validations (`pass/scope.rs`), and
  frames of Python hook calls (`pass/context.rs`) that `report` and
  `get_analysis` look up;
- simplifications in progress (`solver/backends.rs`), giving a Python
  simplifier the objects of its input and environment;
- type-system calls in progress (`types/adapter.rs`): the Python arguments,
  the environment class, and the first exception a Python-defined type's
  `==` or `hash` raised inside the core's infallible equality or hashing;
- the pending-exception slot (`util/pending.rs`): the first exception raised
  by a Python member's `==` or a Python-defined constraint's or domain's
  structural equivalence during one core call, re-raised when the core
  returns. A non-`Exception` (e.g. `KeyboardInterrupt`) replaces a kept
  `Exception`, and once one is kept no further comparisons call Python in
  that call. It also holds exceptions from a Python-defined part's
  serialization hook (`util/foreign.rs`). An exception raised outside any
  call goes to a base frame;
- slot collections (`util/gc.rs`): a Python object held inside a Rust
  closure or core trait object is invisible to the cycle collector, so it
  is held in a `Slot` created inside `collect_slots`, and the owning
  object's `__traverse__` visits it exactly once.

The wire version is a Python context variable, not Rust state.

Tests never clear a process-global registry. A test needing a controlled
registry builds a local one. The exceptions are the Python tests' function
registry, restored by the `function_registry_snapshot` fixture, and the
default solver, which tests restore after replacing.

### Serialization

`fhy-core` uses `#[derive(Serialize, Deserialize)]` wherever possible, and
those serde shapes are the Python package's V2 wire format. A Rust-backed
Python class reads and writes through the core's serde, so Python and Rust
produce byte-identical text; the golden serialization corpus enforces
this. The `__type__`/`__data__` envelope exists only in the binding, as
deprecated V1.

Rules:

- Every value a Python class serializes encodes as a map. A unit variant
  carries empty fields (`{"unknown": {}}`).
- Impls must work with non-self-describing formats. Every serialized type
  has a round-trip test through JSON and through postcard.
- `src/` never uses `#[serde(tag)]`, `untagged`, `flatten` or
  `skip_serializing_if`, never calls `deserialize_any`, and never names a
  `serde_json` type. `serde_json` is a dev-dependency only.
- `BigInt` serializes as a decimal string in every format.

Decoding has two monotonic side effects: an `Identifier` advances the
counter past its id, and a `Canonical<T>` interns its value. A decode that
fails partway may leave both behind. The affected types document this.

A type with an open variant, meaning a part another implementation defines
(a `Type` or `DataType` extension, a custom constraint or domain, an
opaque value, a custom provenance), holds it in a `foreign::Part` and
serializes it as a `foreign::Foreign`: the type id it registered under plus
its payload as text, from `ForeignPart::to_foreign` (whose default
refuses). The module's `wire` submodule defines the shape once, as a plain
data type with the parts left as `Foreign`s. `Serialize` converts into it.
Its `build` method takes a `Resolve`r for the parts and goes through the
public constructors, so decoding validates what construction validates.
The type's own `Deserialize` uses `NoForeign`, which refuses every part.
The core never reads a part's payload and holds no resolver.

### Errors

Each module defines the errors for its own operations, one type per family
of related operations. There is no crate-wide error enum.

- A public error is a `#[non_exhaustive]` enum or a struct with structured
  fields, so callers match on variants and fields, not text. No `is_*`
  classifiers where `kind()` or a match would do.
- `Display` is one lowercase line, no trailing period, and does not repeat
  the text of `source()`.
- `Display` and `std::error::Error` are implemented by hand.

The binding converts core errors through its local `IntoPyErr` trait. For
`identifier` and `interned`, which exist in both languages, it raises the
Python implementation's exception with the same message. Elsewhere it
raises the exception class the Python API documents, with the Rust
`Display` text.

### Public enums and structs

A public enum that may gain variants is `#[non_exhaustive]`. An enum that
callers match exhaustively (`ExpressionKind`, the operation enums,
`LiteralValue`, `Callee`, `Provenance`) stays exhaustive and justifies it
with `#[expect(clippy::exhaustive_enums, reason = "...")]`. The workspace
lints `clippy::exhaustive_enums` and `clippy::exhaustive_structs` reject
any other exhaustive public enum or all-public-field struct.

### Comparing term fields

`AlphaEquivalence` is implemented for `Option<T>`, `[T]`, `Vec<T>`,
`[T; N]` and tuples of up to eight terms (`term/containers.rs`). A type
with term fields implements it by calling these on its fields in order,
combined with `&&` and `?`, under the renaming it was given. Elements
compare in order, lengths must match, `None` matches only `None`.

There are no impls for `HashSet` or `HashMap`, since the result would
depend on iteration order; identifier-keyed maps go through
`is_mapping_alpha_equivalent_under`. There are no impls for `&T`, `Box<T>`,
`Rc<T>` or `Arc<T>`, since they would change which method
`value.is_alpha_equivalent_under(..)` resolves to on a `&&T` or on a
pointer to a type with an inherent method such as `Expression`.

### Python parity

Rust matches Python's behavior and text only for concepts defined in both
languages at once: `identifier` and `interned`. Code that exists only for
that parity starts its doc comment with "Matches the Python
implementation:". Elsewhere Rust conventions apply: `true`/`false`, Rust's
shortest round-trip float formatting (positional in `[1e-5, 1e16)`,
exponent outside it), lowercase error messages, `Display` instead of
`repr` emulation, no `get_`/`list_` prefixes, and shipped defaults as
associated functions (`OpAttribute::commutative()`). Rustdoc describes Rust
behavior and does not refer to the Python implementation.

`stack` and `scope` are implemented natively in each language with no
binding. One list of test cases pins their shared behavior in both suites;
each keeps its own language's errors and names.

### Binding crate layout

`fhy-core-py`'s `lib.rs` lists every registration by hand, grouped by core
module; add a new class or function there. Each core module's bindings live
in a file of the same name. The `_rs` namespace stays flat, because PyO3
submodules cannot be imported as packages.

- A conversion that needs only the value implements `IntoPyErr`
  (`error.rs`) for a core error.
- A conversion that needs context (the interpreter token, the other
  operand, the objects a call has seen) is a free function taking it:
  `…_to_py(…, context)` for an error, `…_to_python(…, context)` for a
  value.
- Shared helpers have one home each and no module writes its own:
  `util/python.rs`, `util/exceptions.rs`, `object_table.rs`
  (`ObjectTable`: Python objects of the nodes and identifiers a call has
  seen, so returned nodes keep their objects), `util/scoped.rs`,
  `util/gc.rs`.

An imported attribute (`ImportedAttr`, `cached_attr!`) is kept for the
life of the process, so monkeypatching or reloading its module later does
not reach the binding.

`src/fhy_core/_rs.pyi` is hand-written; `tests/test_rs_stub.py` checks its
names and parameters against the built extension.

When a canonical Rust value such as an interned `OpAttribute` reaches
Python, the binding returns the same Python object every time, so `is`
holds as it does for values interned in Python. The cache is an
`IdentityCache` per interned class (`util/interned.rs`); the core holds no
Python objects.

### The ground simplifier

`fhy_core::solver::GroundSimplifier` is a `Simplifier` that needs neither
Python nor SymPy. It drives an ordered list of `SimplificationStrategy`
values (`solver::strategy`). Each strategy is a local rewrite of one node
whose children are already simplified. The default list covers exact
integer and rational arithmetic, comparisons, logical operators, decided
piecewise expressions, the exact built-ins SymPy folds itself, registered
constants, and decimal literal forms. Composed built-ins (`max`, `abs`,
`clamp`, `xor`, ...), which SymPy refuses until inlined, are the opt-in
`ComposedBuiltins` strategy.

The driver rewrites bottom-up, applying strategies in order to each node
until none fires. Limits:

- a rewrite bound (`with_max_rewrites`, default 100,000); at the bound it
  stops and keeps what it has;
- the `timeout` in `SimplifyLimits`, checked once per node and per rewrite
  when set; on timeout it declines entirely and never returns a partial
  result;
- trees nested deeper than 256 are declined;
- a single rewrite is not interruptible, so strategies that can be slow on
  large numbers bound their inputs (default: integers up to 2^20 bits,
  fraction parts up to 4096 bits, since reduction is quadratic).

Callers configure strategies with `with_strategy`, `with_strategy_first`,
`without` and `empty`.

Contract for every strategy:

- A rewrite produces exactly what the SymPy backend would return for that
  node, in the SymPy lifting's form (an `Int` literal, a `Bool`, a decimal
  literal for a rational some binary float equals, negated when negative,
  or a quotient of two integers), or the strategy declines.
- It never approximates. It declines floats, free identifiers, user
  functions, irrational or undefined values, and anything else it cannot
  be sure SymPy answers the same way.
- It is local and deterministic, assumes nothing about other strategies,
  and returns `None` when it has nothing to rewrite.
- If unsure of SymPy's form for a partly folded expression, it leaves it.
  The driver returns a rewritten expression only when it is fully decided
  (`with_partial_rewrites` is for strategies that do match SymPy's form of
  a larger expression). Every piecewise branch is folded, including
  untaken ones, because SymPy lowers all of them and raises on a modulo by
  zero in any.

`GroundWithFallback` sends what the ground simplifier declines to another
simplifier. In Python, `GroundSimplifier()` (`SolverBackend.GROUND`) is the
ground simplifier alone and `GroundSimplifier(fallback)`
(`SolverBackend.GROUND_THEN_SYMPY` with SymPy) is the chain. The chain
shares one timeout budget; SymPy itself cannot be cancelled. SymPy remains
the default solver's simplifier.

To add a strategy:

1. Implement `SimplificationStrategy` in a new file under
   `rust/fhy-core/src/solver/strategy/`, with rustdoc stating what it
   rewrites and declines. If it should run by default, add it to
   `default_strategies` and the table in `strategy.rs`.
2. Test it alone in `tests/it/solver/ground_strategy_stories.rs` (rewrites
   and declines through `rewrite`, and in a simplifier holding only it),
   and add pipeline behavior to `ground_stories.rs`.
3. Add cases to the differential tests in
   `rust/fhy-core-py/src/solver/sympy/ground_differential.rs`, which check
   each strategy against SymPy on tables and random nodes, and the default
   pipeline on random trees. Run with
   `cargo test -p fhy-core-py ground_differential` (environment as in
   [Rust tests](#rust-tests)).
   `... ground_differential::timing -- --ignored --nocapture` in a release
   build prints the pipeline's cost against SymPy's.

### Porting a Python class to Rust

- **Bottom-up.** A class moves to Rust only once everything it holds is in
  Rust, so Rust never stores Python objects.
- **In one step.** The binding replaces the Python class outright. A Python
  and a Rust registry for the same concept are never live at once.
- **Benchmark first.** Measure construction, equality, hashing, attribute
  access and the module's main operations with the `benchmark` session
  before and after. If the Rust-backed version is at most 10% slower on
  every path, it replaces the Python one. Otherwise the maintainer decides.
  A class that stays in Python moves only the parts that benefit and says
  why in its module docstring. Example: `Identifier` stays in Python and
  only its counter is in Rust. The PR description gives the numbers, and a
  slowdown on a hot path is either fixed or explicitly accepted there.
- **Retire the generator.** Once a module's Python implementation is
  deleted, its golden generator has no oracle. Delete the generator and its
  `EXPANDED_GOLDEN_CORPORA` entry, and keep the committed JSON as a fixed
  regression corpus.
- **Call Python per hook, not per node.** A Rust walk calls a Python pass,
  analysis or rule once per run or per match and walks the nodes itself.
  Exceptions, where per-node calls are inherent:
  - `fhy_core.term`: `BinderMixin` and `DerivedEquivalenceMixin` call a
    node's hooks, its children's methods, its dataclass fields and user
    comparators per node; each call into Rust still answers one question
    Python asked.
  - `fhy_core.types.dispatch`: the core calls a Python-defined `Type` or
    `DataType` subclass's registered handler once per such node; a class
    without a handler gets the core's default rule.
  - `fhy_core.symbol_table`: the core calls a Python `SymbolTableFrame`
    subclass's `is_structurally_equivalent` and `serialize_to_dict` once
    per frame, and reads its `name` once when added.
  - `fhy_core.provenance`: a Python `Provenance` subclass instance is a
    `Provenance::Custom`, and the core calls its `==`, `hash`, `str` and
    serialization once per comparison, hash, rendering or write.
- **No fallback.** There is no pure-Python copy and no switch to select
  one. Rust tests in `rust/fhy-core/tests/` specify the behavior; a Python
  interface suite covers the API over it. Unported modules stay plain
  Python on top of the Rust-backed types.

## Python style

- [ruff](https://docs.astral.sh/ruff/) for lint and format, mypy
  `--strict` for types, and [ty](https://github.com/astral-sh/ty) as an
  advisory check. We follow the
  [Google Python style guide](https://google.github.io/styleguide/pyguide.html)
  where ruff does not decide.
- Methods that override a base class or implement a `Protocol` method are
  decorated with `@override` from `fhy_core.utils.override`
  (`typing.override` on 3.12+, `typing_extensions.override` on 3.11).
  mypy's `explicit-override` check enforces this.
- Docstrings are Google style, in active voice. The first line is a
  one-sentence summary ending in a period, followed by a blank line and
  further detail if needed. Public functions document `Args`, `Returns` and
  `Raises`. Add other sections such as `Notes` or `Usage` where they help.

## Pull requests

- Open an issue first for anything beyond a small fix, to agree on the
  approach.
- Branch from `dev` and target `dev`. Only release PRs target `main`.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/):
  `type(scope): summary`, with `!` and a `BREAKING CHANGE:` footer for
  breaking changes.
- Include tests and documentation, and make sure `uv run nox` passes. Work
  in progress is fine to open as a draft.
- Contributions are licensed under the project's
  [BSD-3-Clause license](LICENSE). If you include code you did not write,
  make sure its license is compatible and keep its notice, or get the
  author's permission to relicense it.

## Releases

1. Set the version in the workspace `Cargo.toml` (`[workspace.package]`
   and the exact `fhy-core` requirement under `[workspace.dependencies]`).
   The Python version follows from it.
2. Run `uv run nox -s property` and `uv run nox -s golden_expanded`, and the
   property suite under a few random seeds.
3. Open a PR from `dev` into `main`. This runs the full CI matrix, the
   property job and the expanded golden corpora.
4. Publish a GitHub release from `main`. `python-release.yml` builds wheels
   for Linux, macOS and Windows plus an sdist and publishes them to PyPI.
   `rust-release.yml` publishes `fhy-core` and `fhy-core-py` to crates.io.
   Both use trusted publishing; a crate version already on crates.io is
   skipped.
