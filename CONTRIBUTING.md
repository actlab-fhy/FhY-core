# Contributing to *FhY* Core

Pull requests are always welcome, and the *FhY* community appreciates any help you give.

## Working with *FhY* - For Developers

1. Download the development branch of the *FhY* Core source code.

```bash
git clone https://github.com/actlab-fhy/FhY-core.git -b dev
cd FhY-core
```

2. Install [uv](https://docs.astral.sh/uv/) (used for environment and dependency management) and a Rust toolchain (stable, 1.85 or newer; [rustup](https://rustup.rs/) picks the channel from `rust-toolchain.toml`), then create the development environment. This installs *FhY* Core in editable mode along with the default `dev` dependency group, and compiles the Rust extension `fhy_core._rs` with maturin.

```bash
uv sync
```

3. Initialize pre-commit
```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

4. Run the developer tasks with [nox](https://nox.thea.codes/) (driven by uv). The test sessions span Python 3.10–3.14; uv installs any interpreters you are missing automatically.
```bash
uv run nox              # lint, type_check, tests, coverage
uv run nox -s lint      # ruff check + format
uv run nox -s type_check  # ty (advisory) + mypy --strict
uv run nox -s tests-3.12  # a single Python version
uv run nox -s tests_minimal  # without the optional solver packages
```

A test that reaches a solver backend is marked `z3` or `sympy` (by what it
reaches, not what it imports), so it is skipped where that package is
missing; `tests_minimal`, which installs neither, catches a missing mark.

The `coverage` session combines the `.coverage.*` data left by the `tests`
sessions, so run `tests` first. A bare `uv run nox` runs `tests` then
`coverage` in order; running `coverage` alone on a clean tree simply skips.

Do not rebuild the extension while tests run in the same checkout. The
install is editable, so `uv sync`, and every nox session, which syncs its
own environment editable too, writes `src/fhy_core/_rs.*.so` in the source
tree: maturin removes the old file and writes the new one in place. A
running process keeps the old file, but one that imports the package
meanwhile, such as an xdist worker starting or a test's subprocess, loads a
partly written extension and dies with `SIGBUS`. The run then reports a
failed test that passes when rerun: a subprocess test whose child died, or
`worker 'gwN' crashed`. Run the Python gates one after another, or give
each concurrent run its own checkout.

## Property-based testing

Most of the test suite is example-based: a test picks an input, runs the
code, and checks the output. *FhY* Core also uses property-based testing
with [Hypothesis](https://hypothesis.readthedocs.io/). A property test
states a rule that must hold for every input in a space ("simplifying an
expression never changes its value", "serializing and deserializing
returns an equal object"), and Hypothesis generates inputs that try to
break it. When one does, Hypothesis shrinks it to a minimal failing input
and prints it as a `Falsifying example`. Under the default local profile it
also saves the failure in `.hypothesis/` and replays it first on the next
run, so a fix is checked against the exact input that broke.

### Why we use it

Much of *FhY* Core is symbolic: expression trees, the SymPy and Z3
bridges, interval arithmetic over parameters, constraint systems, and type
lattices. Their input spaces are combinatorial. What matters is which
operation nests under which, which sort each operand has, whether a bound
is inclusive, whether a set is empty, and whether a number is too large for
a float. Hand-picked examples cover the combinations their authors think
of, and bugs live in the ones nobody thought of. Property tests have found
bugs of exactly that kind here, each of which passed every hand-written
test, for example:

- a piecewise whose condition was a bare Boolean identifier lost its later
  branches when lowered to SymPy, so a constraint reported `VIOLATED` for
  an assignment that satisfies it;
- `(b ? 2 : 0) % -6` simplified to `0`, because SymPy decides that an even
  expression divided by 6 is an integer;
- two real bounds that differ beyond float precision were compared after
  rounding, so a non-empty interval was rejected.

The same code usually comes with an independent way to compute the right
answer, and that is what makes a property worth writing: a reference
evaluator to compare the SymPy bridge against, a brute-force enumeration
over a small domain to compare the solver against, an inverse to
round-trip through, or an algebraic law such as associativity or
absorption to check.

Properties complement examples; they do not replace them. Examples keep
three jobs a property cannot do: pinning an exact error message or
exception type, pinning a golden value (serialization shape, `repr`/`str`
text), and serving as a readable worked example or user story.

### When to write a property

If ten well-chosen hand-picked inputs would convince a reviewer, write ten
examples instead. Write a property only when the input space is
combinatorial and you can name an independent oracle for it. A property
that re-derives the answer the same way the code does is a tautology, not
a test. Exhaustive parametrize tables over a small finite domain also stay
as examples, since they check every case in every run.

### Writing a property

A property test file sits next to the unit it tests, named
`test_<unit>_properties.py` alongside `test_<unit>.py`. Mark the module
with `pytestmark = pytest.mark.property`, and call
`pytest.importorskip("hypothesis")` right after `import pytest`, before
importing `hypothesis`, `tests.strategies`, or `fhy_core`: `hypothesis`
lives in the `property` dependency group, which the `tests` lane never
installs. Mark any property that exercises the Z3 bridge with
`pytest.mark.z3`. A minimal property looks like this:

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

Name the oracle in the docstring. `@example` pins an input that runs on
every run, before any generated input.

**Shared strategies** live in `tests/strategies/`, one module per domain
(identifiers, literals, expressions, params, constraints, types, orders,
serializables). Reuse them rather than writing a local generator, and
extend them when a property needs a new shape. Import them only from
property files, never from a `conftest.py`; a `conftest.py` must import
cleanly without `hypothesis` installed.

**Prove the strategy reaches the shape.** A property is only as good as
the inputs it sees: a strategy that never draws a Boolean piecewise under
`==` cannot find a bug there, and the property passes anyway. When you add
a shape to a strategy, add a reachability self-test to
`tests/test_strategies_properties.py` that uses `hypothesis.find` to show
the strategy generates it. Run a property with
`--hypothesis-show-statistics` to see how many inputs it actually tried and
how many were discarded.

**Never filter your way to valid input.** Do not use `assume()` or
`.filter()` to rescue a strategy that draws invalid input, and never
suppress a health check; a health check firing is a strategy bug to fix.
Generate valid input by construction instead, for example sort-aware
expression trees. When a shape must stay out of a property, turn it off
with an explicit strategy parameter and say why in the property's
docstring.

**Lifting an example into a property.** When an example test compares the
implementation to an oracle on hand-picked inputs, lift it: write the
property against a strategy from `tests/strategies`, pin every hand-picked
input with `@example(...)`, and delete the example test, all in the same
commit.

### Running property tests

`uv sync` installs `hypothesis` with the `dev` group, so a local
`uv run pytest` runs property tests alongside the examples. Profiles set
how hard Hypothesis looks:

- `dev` (the default): 25 random inputs per property, with the example
  database in `.hypothesis/`.
- `thorough`: 400 inputs, derandomized, with no database. This is the
  release gate. Set `HYPOTHESIS_PROFILE=thorough`, or run
  `uv run nox -s property`, which sets it for you. A failure prints a
  `@reproduce_failure` blob that replays it exactly.
- `mutation`: 25 inputs, derandomized, with no database, so every mutant
  sees the same inputs. The `mutation` nox session selects it.

Because `thorough` is derandomized, it draws the same inputs on every run.
Before a release, also run the property suite under a handful of random
seeds, for example `uv run pytest -m property --hypothesis-seed=1234`,
since a seed reaches inputs the fixed run never tries. A failure found this
way reproduces with the same seed.

Every profile runs without a deadline; under `pytest-xdist`, scheduler
contention rather than test cost is what trips one. When a test is
genuinely expensive (a Z3 call, a large tree), cap its input count with
`@cap_max_examples(N)` from `tests/strategies/settings.py`, or assign
`cap_max_examples(N)` to a state machine's `TestCase.settings`. It runs the
profile's count or `N`, whichever is lower. Never write a bare
`@settings(max_examples=N)`: it replaces the profile's count instead of
capping it, so under `dev` and `mutation` an `N` above 25 runs more inputs
than the profile.

### When a property fails

1. Read the shrunk `Falsifying example` and reproduce it as a plain
   example test in `test_<unit>.py`. The `tests` lane that runs on pull
   requests into `dev` never runs properties, so this example test is what
   guards the fix there.
2. Pin the same input on the property with `@example(...)`.
3. Fix the code. If the bug is upstream (in SymPy or Z3, say) and cannot be
   fixed or worked around yet, pin the input as a separate example test
   marked `@pytest.mark.xfail(strict=True, raises=<ExceptionType>)`, with a
   reason that names the upstream bug. Never mark the property itself as
   an expected failure: Hypothesis runs the `@example` inputs first and
   stops at the first failure, so the property would never generate
   another input.

### Hazards specific to this codebase

- Function-scoped fixtures do not reset between inputs: Hypothesis runs
  the test body many times inside one pytest test. Never request the
  `function_registry_snapshot` fixture in a `@given` test; write it as an
  example test instead if it must touch the registry.
- `Expression.__bool__` raises. Never put an `Expression` in a truth
  context, in a strategy or in a reference evaluator.
- Logical connectives and piecewise conditions must be Boolean-sorted.
  Generate sort-aware trees so no draw is ever refused by
  `validate_logical_operands` or a bridge, and bind Boolean identifiers to
  `bool` values in an environment.
- Strategies use `mock_identifier`, never a real `Identifier`. A mock
  compares and hashes by id alone, so identifier pools must not share ids:
  integer pools start at id 10000 and Boolean pools at 15000
  (`tests/strategies/identifiers.py`).
- An `@example` cannot pin values drawn through `st.data()`. Put dependent
  draws in a `@st.composite` strategy instead.
- SymPy keeps global random state (`sympy.core.random`) that some of its
  simplification paths read. A test that reseeds it must restore it.

### CI policy and mutation testing

The `property` and `golden-expanded` jobs only run on pull requests into
`main` and on manual dispatch; they do not run on pull requests into `dev`.
Run `uv run nox -s property` and `uv run nox -s golden_expanded` locally
before opening a release pull request.

The `rust` and `rust-msrv` jobs run on every pull request, and `ci-ok`
requires both. `rust` runs `cargo fmt --all --check`, `cargo clippy
--workspace --all-targets --all-features --locked -- -D warnings`, and,
since a workspace build unifies the binding's features into `fhy-core`,
`cargo clippy -p fhy-core --all-targets` with no features and with each
feature alone; `cargo test --workspace --locked --all-features`; `cargo
doc -p fhy-core --no-deps` with the default features and `cargo doc
--workspace --no-deps`, both with `-D warnings`; and `cargo package
--locked -p fhy-core`, then unpacks the packaged crate and runs `cargo
test --locked` in it, so the tests pass on the crate as a consumer
receives it. docs.rs builds `fhy-core` with both features and `--cfg
docsrs` (`[package.metadata.docs.rs]`), so each feature's items carry a
`doc(cfg)` marking: a new item behind a feature gets
`#[cfg_attr(docsrs, doc(cfg(feature = "...")))]`. `rust-msrv` runs `cargo
check --workspace --lib --locked` on the `rust-version` in `Cargo.toml`,
so the MSRV covers both `fhy-core` and the PyO3 binding crate
`fhy-core-py`, and `cargo check -p fhy-core --all-targets` with no features
and with each feature alone, with `-D warnings`. That version is a promise to the crate's
consumers, so `.cargo/config.toml` has the resolver fall back to dependency
releases that build on it and `cargo update` keeps `Cargo.lock` within it.

The `deny` job runs [cargo-deny](https://embarkstudios.github.io/cargo-deny/)
against `deny.toml`, and `ci-ok` requires it. Every dependency must be under
one of the allowed permissive licenses, come from crates.io, and have no
wildcard version requirement. A known RustSec advisory against a dependency
shows as a warning on the job rather than failing it. Run the same checks
locally with `cargo deny check` (`cargo install cargo-deny --locked`); a new
license or an ignored advisory goes into `deny.toml` with its reason.
Dependabot (`.github/dependabot.yml`) opens a weekly pull request against
`dev` for Cargo and GitHub Actions updates, with minor and patch updates
grouped into one pull request per ecosystem.

The Rust equivalence tests replay golden corpora under `rust/fhy-core/tests/golden/`,
each recorded from the Python implementation by the `generate_*.py` script
beside it. `tests/test_golden_corpora.py` reruns every generator in a fresh
interpreter, so the `tests` sessions check the corpora on every supported
Python. It fails,
printing the regeneration command and a diff, if a committed corpus differs
outside its `provenance` block or a generator has no committed corpus. The
`golden-corpora` pre-commit hook runs the same module whenever a commit
touches `src/fhy_core/`, `rust/fhy-core/tests/golden/`, or
`tests/test_golden_corpora.py`. The generators are type-checked and linted
with the package, since they are the oracle the Rust port is held to. After
changing the Python behavior a corpus records, regenerate it and commit the
result.

Each generator can also write a much larger random corpus, which an ignored
test in its equivalence test file replays from the path in an environment
variable. `uv run nox -s golden_expanded` writes every expanded corpus from
its Python oracle and runs those ignored tests on them (it needs
`cargo`); the `golden-expanded` job runs it on the same triggers as the
`property` job. A new generator needs an entry in `EXPANDED_GOLDEN_CORPORA`
in `noxfile.py`, or the session fails.

Mutation testing measures whether a property earned its place: it makes
small changes to the source and checks that some test fails. Each targeted
module has its own config under `cosmic-ray/`; run one with
`uv run nox -s mutation -- <module>` (the module defaults to `lattice`).

## Benchmarks

`benchmarks/` holds [pytest-benchmark](https://pytest-benchmark.readthedocs.io/)
benchmarks of the public API's hot paths, grouped by concept, with shared
fixtures in `benchmarks/conftest.py`. They use the public API only, so the
same benchmark measures a class before and after it switches to Rust. The
opt-in `benchmark` session runs them and is neither a default session nor
a CI job. It writes each run's results to `.benchmarks/<python>.json` and
saves the run under `.benchmarks/storage/` (both gitignored), so
`-- --benchmark-compare` compares a run with the previous one:

```bash
git switch --detach <before>
uv run nox -s benchmark-3.12
cp .benchmarks/3.12.json .benchmarks/3.12-before.json
git switch -
uv run nox -s benchmark-3.12
uv run --group bench pytest-benchmark compare --group-by=name --columns=median \
    .benchmarks/3.12-before.json .benchmarks/3.12.json
```

The machine's load moves the numbers, so compare runs made back to back on
the same machine.

A class's benchmark must be run before it switches to Rust and again
after. A class without a benchmark gets one in `benchmarks/` first,
covering construction, attribute access, `==`, `hash` and the module's main
operations. The port records the numbers behind its pattern choice, and a
switch that makes a hot path slower either changes pattern or is recorded
as an accepted cost.

## Porting to Rust

*FhY* Core is moving to Rust one module at a time. The Rust code is a
Cargo workspace. `rust/fhy-core` is the pure-Rust library, with no PyO3 and
no Python at build or test time. `rust/fhy-core-py` holds the PyO3
bindings, as a library: it builds no extension module, and its
`register(py, module)` adds the bindings to a module it is given. It also
holds the SymPy backend, `solver::sympy`, a `fhy_core::solver::Simplifier`
that drives SymPy in the interpreter the extension runs in, and the Python
class of the pure-Rust ground simplifier, `solver::ground`.
`rust/fhy-core-ext` is the thin `cdylib` that maturin builds into the
extension module `fhy_core._rs`, and `rust/example-aggregate` is a
test-only aggregate extension (see "One extension module per process").
The workspace table holds the one `pyo3`, since `pyo3-ffi` links `python`
and a build holds one. A port adds its types to `fhy-core` and their
bindings to `fhy-core-py`. Every port follows these rules.

### One extension module per process

All Rust code that uses *FhY* Core's Rust types compiles into a single
Python extension module. The identifier id counter and each `Interned`
type's `InternRegistry` are Rust `static`s, which exist once per compiled
copy of the crate, and PyO3 creates a separate Python type for each
extension module. A second extension linking the crate would issue ids that
collide with the first one's, keep registries whose canonical instances
never match, and fail `isinstance` checks against the first one's classes.
A downstream *FhY* package that gains Rust code depends on the crate as a
Rust library and is compiled into one combined extension module; it never
ships an extension of its own that links the crate. The mechanism has four
parts.

**The binding is a library.** `fhy-core-py` is an `rlib`. Its
`register(py, module)` adds every class, function and piece of module state
to a module it is given, and refuses a module it has registered into
already. Its `convert` module is the public conversion surface: for an
`Identifier`, an `Expression`, a `Type`, a `Param`, a `ParamAssignment`, a
`ValueDomain`, an `OpAttribute`, a `Diagnostic` and a `ValidationReport`,
`…_from_python` reads a Python object as the Rust value and `…_to_python`
builds the object of a Rust value through the public class. A downstream
binding crate calls them at the boundary of its own `#[pyfunction]`s and
`#[pymethods]`. Its `convert::numpy` module is the public conversion surface
for `NumPy` arrays, the one `evaluate_expression_with_numpy` is written over:
`require_numpy` imports `NumPy` or raises the `ImportError` naming the
caller's entry point; `NumpyValue::from_python` converts a Python number or
anything `numpy.asarray` accepts to a scalar or an array of the core's three
domains (`bool`, `i64`, `f64`), borrowing `bool_`, `int64` and `float64`
arrays in native byte order and casting every other admitted dtype once;
`NumpyValue::as_binding`, `to_array_value` and `as_scalar` hand the value to
the core's evaluators; `NumpyKernels` computes the 14 transcendental natives
with `NumPy`'s ufuncs; `array_value_to_numpy` and `scalar_to_numpy` convert
results back; and `evaluation_error_to_python` raises the same exceptions
`evaluate_expression_with_numpy` does. No `rust-numpy` type appears in its
signatures, so a downstream crate needs no `numpy` dependency of its own.
Its `convert::param` module is the public entry to the param questions:
`with_param_context(py, detach, question)` runs `question` with the
`ParamContext` that `fhy_core`'s own param methods use (the default solver
`set_default_solver` chose, a snapshot of the function registry, and the
param observer that logs to `fhy_core`'s loggers), detached from the
interpreter when `detach`, and re-raises after the question the Python
exception a hook raised during it. `fhy_core`'s methods run over the same
code, so a downstream crate decides a question as `fhy_core` does.
Its `kit` module is the public surface for writing a Rust-backed class the way
`fhy_core`'s are written, which `fhy_core`'s own classes use and a downstream
`-py` crate copies no longer. `kit::python` has `Seed` (the contents a private
seed class hands a class's `__new__`, taken once), `ImportedAttr` and the
`cached_attr!` macro (an attribute of a Python module, imported on first use
and kept; the macro is exported at the crate root and re-exported there) and
`type_name`. `kit::exceptions` has `ExceptionClass` (`new`, `class`, `build`,
`err`, `is_instance_of`, declared as a `static`), `unbox_py_err`,
`boxed_error_to_py` and the framework's classes `SERIALIZATION_ERROR`,
`DESERIALIZATION_VALUE_ERROR`, `DESERIALIZATION_DICT_STRUCTURE_ERROR`,
`MALFORMED_PAYLOAD_ERROR`, `FROZEN_MUTATION_ERROR` and
`EQUIVALENCE_DERIVATION_ERROR`. `kit::interned` has `IdentityCache<K>`, the
`is` identity of canonical values, generic over its key (the identifier id,
`u64`, by default, and a `String` or tuple for a downstream class, read by the
borrowed form) and the `InternedMixin` helpers. `kit::public_class` has
`PublicClass::new` and `PublicClass::in_module`, which names a downstream
module in its messages; `kit::frozen` the `FrozenMixin` refusals;
`kit::dataclass` `compare_as_dataclass` (the equality answers a `bool` or a
`PyResult<bool>`, through `Outcome`), `is_same_or_equal`, `hash_value`,
`collect_tuple`, `format_dataclass_repr`, the argument checks and
`OptionalArgument`; `kit::serialization` the payload readers,
`read_payload_fields` over an array of `(name, FieldShape)` pairs or
`PayloadFields::allowing_extra` for a reader that ignores other keys,
`read_constructor_fields`, `read_nested_value`, `read_nested_list`,
`keep_fields`, `serialize_nested`, `is_serialized_dict` and
`construct_from_decoded_fields`, with
`construct_from_decoded_fields_reporting_overflow` for a class that takes
machine integers; `kit::scoped` `ScopedStack` and `ScopedGuard`;
`kit::pending` the pending exception of an infallible hook
(`record_pending_error`, `has_pending_error`, `with_pending_errors`,
`capture_pending_errors`); `kit::gc` `Slot`, `Slots`, `collect_slots`,
`traverse_locked`, `clear_locked` and `traverse_all`; and `kit::foreign`
`foreign_of`, `foreign_failure` and `RaisedError`, which turn a Python-defined
part into a core `Foreign`. Each item is documented with its errors and
panics, and the stories in `kit/*/tests.rs` run them in the embedded
interpreter against small stand-ins for the `fhy_core` modules the kit
imports (`kit::testing`, since `fhy_core` itself is not importable there).
A conversion that another crate needs and `convert` lacks is
added there, as a documented `pub fn` over the `pub(crate)` one, and no
`#[pyclass]` becomes `pub`. The crate is a library and not a `cdylib` with
an `rlib` beside it because a `#[pymodule]` exports a `PyInit_<name>`
symbol: linked into an aggregate as an `rlib`, the crate would export its
own `PyInit__rs` from every aggregate, and one whose module is also called
`_rs` would not link. So the module lives in the separate `fhy-core-ext`
crate, which only calls `register` and sets `__version__`, and maturin
builds that (`manifest-path` in `[tool.maturin]`); the wheel of `fhy_core`
alone is unchanged.

**An aggregate is one `cdylib` per product.** Its `#[pymodule]` calls
`fhy_core_py::register`, then the registration function of each of its own
crates, into one module. `rust/example-aggregate` is the template and the
test: it registers `fhy-core-py` and one class, `Tagger`, that takes and
returns an `Identifier` and an interned `OpAttribute`. It is a workspace
member (`publish = false`) and no wheel ships it. A downstream aggregate
follows four rules: its classes name the package they belong to
(`module = "..."`); its module is a top-level one that imports no `fhy_core`
Python code when it is imported, since `fhy_core` imports it while `fhy_core`
is itself being imported; its package imports `fhy_core` before anything
calls into the module, since the binding imports `fhy_core`'s Python
modules on first use, and doing that from inside a call made before
`fhy_core` was imported can deadlock on a once-initialized cache; and it depends on the same `fhy-core` source and
version as `fhy-core-py`, since two sources are two copies of the statics.

**Class identity does not depend on the module's name.** Every `#[pyclass]`
names its module explicitly, `module = "fhy_core._rs"`, so `__module__`,
`repr`, `pickle` (which imports the class by that name) and the qualified
names users see are the same whichever native module holds the class. The
binding also finds its own module state, and `fhy_core`'s Python code its
classes, by the name `fhy_core._rs`. The loader therefore installs an
aggregate under that name too, `sys.modules["fhy_core._rs"]`, and as the
`_rs` attribute of the package, so `from fhy_core import _rs` and every
`py.import("fhy_core._rs")` reach the aggregate. The composition tests in
`tests/test_composed_extension.py` check `isinstance`, `pickle`, `repr` and
`__module__` against an aggregate named `_fhy_example_aggregate`.

**The loader chooses the module.** `fhy_core._extension` runs when the
package is imported and picks the one native module of the process:

1. the module named by the environment variable `FHY_CORE_NATIVE_MODULE`,
   if set, which overrides everything else (`fhy_core._rs` names the module
   `fhy_core` ships);
2. otherwise the module named by the entry points of the group
   `fhy_core.native`. A product's wheel declares its aggregate with
   `[project.entry-points."fhy_core.native"]`, `product = "module_name"`.
   Two entry points that name different modules raise `ImportError` naming
   both, with the distributions that advertise them;
3. otherwise `fhy_core._rs`, when `fhy_core` is used alone.

An aggregate must report the `fhy_core` version it holds as
`__fhy_core_version__`, which `register` sets from the crate version, and
`fhy_core` refuses a stale one as it refuses a stale `_rs`. A module that
is named and fails to import raises `ImportError`; it is never replaced by
`fhy_core._rs`, which would run a second copy of the Rust code beside the one
the product expects. If `fhy_core._rs` already names a different module,
such as one that was imported first, loading an aggregate raises
`ImportError` naming both. The loader tests in `tests/test_extension.py`
cover the entry points, the environment variable, the refusals and the
version check without a build; `tests/test_composed_extension.py` builds
`rust/example-aggregate` with `cargo build` (cargo must be on the path) into
`target/composition-<python version>`, and checks in a fresh interpreter that
one identifier counter, one registry per interned type and one set of
classes serve the aggregate's class and `fhy_core`.

**Which wheel ships an aggregate.** One aggregate per Python process, so one
per set of packages that can meet in a process. A wheel per product cannot
do that: the aggregates of two products that are used together would be two
native modules, and the loader refuses that. The recommended shape is an
umbrella distribution (working name `fhy-native`) whose extension is the
aggregate of every `-py` crate of the stack, whose version is pinned to the
matching releases of `fhy_core` and of each product, and that declares the
`fhy_core.native` entry point; the products depend on it through an optional
extra. `fhy_core` alone keeps shipping its own `fhy_core._rs`, so nothing
changes for users without the umbrella. Until the umbrella exists, a single
downstream product, which is the only one with Rust code, may ship its own
aggregate under the same entry point and hand the role to the umbrella when a
second product gains Rust code. No release packaging for an aggregate is
built here.

The module declares that it uses the GIL, `#[pymodule(gil_used = true)]`
in `rust/fhy-core-ext/src/lib.rs`, so importing it on a free-threaded
interpreter (3.13t, 3.14t) re-enables the GIL, with CPython's
`RuntimeWarning`. PyO3 0.29 declares free-threading support unless told
otherwise, and nothing has shown the binding safe without the GIL: its
non-frozen classes raise borrow errors under contention, the opaque
value's ordering key runs Python in a `OnceLock` initializer, and several
"never held across Python" invariants were argued for the GIL build only.
The declaration stays until a free-threaded CI job exists and those are
checked; then the job, not this paragraph, decides. Independently of the
GIL, the NumPy evaluator reads a `float64` input array in place while it
releases the GIL, so a caller must not write an input array from another
thread during the call; `evaluate_expression_with_numpy`'s docstring, the
stub and the README say so.

### Process-global state is limited to identity

Exactly two kinds of state are process-global: the identifier id counter
and each `Interned` type's `InternRegistry`. Both are append-only: an id is
never reissued, and a canonical value is never replaced or removed.
Everything else a port keeps between calls, such as the pass registry, run
statistics or caches, is an owned value that its user creates and passes
explicitly. Where the Python API needs one shared instance, the binding
holds it in the extension's module state. A new process-global `static`
with interior mutability needs the maintainer's agreement and a line in
this section.

The binding crate keeps three kinds of write-once or append-only state.
Each Rust-backed interned class has an identity cache from canonical keys
to their Python objects, which is append-only like the registry it
mirrors. Each Rust-backed class has a write-once slot for the public
Python class that registers itself at import
(`rust/fhy-core-py/src/kit/public_class.rs`), so a value the binding builds
from Rust is an instance of that class. The shared empty `AlphaRenaming`
that `AlphaRenaming.empty()` returns is a write-once slot
(`rust/fhy-core-py/src/term/renaming.rs`), an immutable value built on
first use, as a public class slot is; the derived-equivalence plans stay in
the Python module's `_PLAN_CACHE` dict.

The binding holds three shared registries for the Python API. The function
registry behind `register_function` and the lookups of
`fhy_core.symbolic.expression.registry` is a `Mutex<Arc<_>>` of the core's
owned `FunctionRegistry` and each entry's Python object
(`rust/fhy-core-py/src/expression/registry/state.rs`). A registration swaps
in a new state whole, and the lock is never held across a call into Python.
It is not append-only: `set_registry_state_for_tests`, the tests' snapshot
seam, replaces it. The built-in entries beside it are built once, at
import, and never change. The default solver the functions of
`fhy_core.symbolic.solver`, and so the constraints and params, ask when no
backend is named is a `Mutex<Option<Py<Solver>>>`
(`rust/fhy-core-py/src/solver/state.rs`), set when that module is imported
and replaced whole by `set_default_solver`; the lock is never held across a
call into Python, and like the function registry it is not append-only.
The verification registry of `fhy_core.pass_infrastructure.verification`
lives in the extension's module state: `pymodule_init` sets the private
attribute `fhy_core._rs._verification_registry` to a `Mutex<Arc<_>>` of the
core's owned `VerificationRegistry` and the Python objects its keys stand
for (`rust/fhy-core-py/src/pass/verification.rs`), which the binding
reaches through a write-once import cache. A registration swaps in a new
state whole, and the lock is never held across a call into Python. It is
append-only and adds no Rust `static` with interior mutability. The core
crate stays free of all three, as of all global state beyond identity.

The rest of the binding's state is thread-local and lives only for one
call:

- a stack of per-call object tables
  (`rust/fhy-core-py/src/expression/pattern/objects.rs`), which maps the
  Rust nodes a match or a rewrite walk reaches to their Python objects
  while it runs;
- a stack of the scopes of the pass runs, pipeline runs and validations in
  progress (`rust/fhy-core-py/src/pass/scope.rs`), which records the
  diagnostics Python hooks report so they return as themselves, and a stack
  of the frames of the Python hook calls in progress (`pass/context.rs`),
  which `report` and `get_analysis` find by the pass object;
- a stack of the simplifications in progress
  (`rust/fhy-core-py/src/solver/backends.rs`), which hands a Python
  simplifier the objects of its input and environment;
- a stack of the type-system calls in progress
  (`rust/fhy-core-py/src/types/adapter.rs`), each a context holding the
  Python objects the call was given, the class of its environment, and the
  first exception a Python-defined type's `==` or `hash` raised inside the
  core's infallible equality or hashing;
- a pending-exception slot (`rust/fhy-core-py/src/kit/pending.rs`)
  holding the first exception a Python member's `==`, or a Python-defined
  constraint's or domain's structural equivalence, raised during one call
  into the core, which the call raises when the core returns: those back
  the core's infallible `==`, while every other hook's exception is its own
  error; an exception that is not an `Exception`, such as
  `KeyboardInterrupt`, replaces a kept `Exception`, and once one is kept no
  comparison calls Python again during that call. The same slot holds the
  exception a Python-defined part's serialization hook raises while the
  core serializes or resolves it (`rust/fhy-core-py/src/kit/foreign.rs`); the wire
  version is a Python context variable, not Rust state;
- a stack of slot collections (`rust/fhy-core-py/src/kit/gc.rs`): a Python
  object the binding keeps inside a Rust closure or a core trait object,
  where the cycle collector cannot see it, is held in a `Slot`, and the
  construction that makes it runs inside `collect_slots`, so the object it
  builds owns the slot and its `__traverse__` visits it, at most once
  however the core shares the value.

Each frame lives only for its call, so every stack is empty whenever no
such call runs. Every one of these, the pending-exception slot included,
is a `ScopedStack` (`rust/fhy-core-py/src/kit/scoped.rs`): a frame is pushed
only through a guard that pops it when dropped, on unwind included, so a
panic, which PyO3 raises as `PanicException`, never leaves a stale frame
for the thread's next call; the slot keeps an exception raised outside
every call in a base frame.

Tests never clear a process-global registry; a test that needs an empty or
controlled registry builds a local one, except that the Python tests
restore the function registry through the `function_registry_snapshot`
fixture, and the default solver after replacing it.

Ids `0..RESERVED_ID_COUNT` (65,536 ids) are reserved for the identifiers
the crate ships, such as the built-in tags, and each shipped identifier has
a fixed id in the crate-private reserved table. The counter issues fresh
ids from 65,536 upward and never at or above `ID_CAP` (2^63). A payload id
below `ADVANCE_CAP` (2^62) is decoded or restored and advances the counter
past it; one from `ADVANCE_CAP` up to `ID_CAP` is decoded only if this
process issued it, so it needs no advance; any other is refused. So no
payload can raise the counter past 2^62, and every fresh id, even one
issued after a worst-case payload, reads back. A newly shipped identifier takes an unused
id from the reserved table rather than drawing one from the counter.

### Serialization is plain serde

`fhy-core` serializes with `#[derive(Serialize, Deserialize)]` wherever it
can, in shapes Rust defines, and those shapes are the Python package's wire
format, V2: a Rust-backed class writes and reads its value through the
core's serde, so Python's and Rust's texts are byte-identical, and the
golden serialization corpus holds them to it. There is no
`__type__`/`__data__` envelope in the core crate; the binding keeps it only
as the deprecated V1 format until V1 is removed. Every value a Python class
serializes encodes as a map, so a unit variant carries empty fields
(`{"unknown": {}}`). Serde impls must work with
non-self-describing formats as well as JSON: every serialized type has a
round-trip test through JSON and one through postcard, the binary test
format. `src/` never uses `#[serde(tag)]`, `untagged`, `flatten` or
`skip_serializing_if`, never calls `deserialize_any`, and never names a
`serde_json` type; `serde_json` is a dev-dependency only. A `BigInt`
serializes as a decimal string in every format. Decoding has two side
effects, both monotonic: an `Identifier` advances the id counter past its
id, and a `Canonical<T>` interns its value. A decode that fails partway may
leave the counter advanced and some canonical values registered. The
affected types document this; decoding is not ordered to prevent it.

A type with an open variant, one that holds a part another implementation
defines (a `Type` or `DataType` extension, a custom constraint or domain,
an opaque value), holds it in a `fhy_core::foreign::Part`, and serializes
that part as a `fhy_core::foreign::Foreign`: the type id its
implementation registered under and its own payload as text, from the
`to_foreign` of the `ForeignPart` supertrait, whose default refuses. Its module's
`wire` submodule defines the shape once, as a plain data type that derives
both traits with the parts left as `Foreign`s; `Serialize` converts the
value into it, and its `build` method takes a `Resolve`r of the parts and
builds through the public constructors, so decoding validates what
construction validates. The type's own `Deserialize` builds with
`NoForeign`, which refuses every part by its type id. The core never reads
a part's payload and holds no resolver of its own.

### Replacing a Python class

- Port bottom-up. A class switches to Rust only once everything it holds
  is already in Rust, so Rust code never stores Python objects.
- Switch in one step. The binding replaces the Python class outright; a
  Python registry and a Rust registry for the same concept are never live
  at the same time.
- Benchmark before replacing. Measure the module's hot paths
  (construction, equality, hashing, attribute access, and whatever the
  module does most) with the `benchmark` session (see "Benchmarks") on the
  Python implementation, then again on the Rust-backed one. When the
  Rust-backed version is at most 10% slower on every measured path, it
  replaces the Python implementation. When it is more than 10% slower on
  any path, usually because every call crosses into the extension, the
  maintainer decides whether to keep the Python class. A kept class moves
  only the parts that gain from Rust and says why in its module docstring.
  `fhy_core.identifier` is the example: `Identifier` stays in Python and
  only its id counter runs in Rust.
- Freeze the golden corpus. Golden corpora exist only for concepts defined
  in both languages, today `identifier`, `interned`, and serialization,
  whose V2 texts Python writes and Rust reads and writes back
  byte-identically (`generate_serialization_cases.py`). Once a module's
  Python implementation is deleted, its generator has no oracle left to
  run. Delete the generator (the drift check and `golden_expanded` find
  generators by the `generate_*.py` pattern) and its
  `EXPANDED_GOLDEN_CORPORA` entry, and keep the committed JSON as a fixed
  regression corpus.
- Call back into Python per hook, not per tree node. A Rust walk over an
  IR calls a Python pass, analysis or rule once per run or match, and walks
  the nodes itself. The first exception is `fhy_core.term`: `BinderMixin`
  and `DerivedEquivalenceMixin` run in Rust but call a node's own hooks,
  its children's methods, its dataclass fields and user comparators per
  node, since those are per node by nature; each call into Rust still
  answers one comparison, query or substitution that Python asked for. The
  second is `fhy_core.types.dispatch`: the core calls the handler a `Type`
  or `DataType` subclass Python defines registered on a dispatcher once per
  such node it meets, since only Python can answer for a class Python
  defines; a class without a handler takes the core's default rule with no
  call into Python. The third is `fhy_core.symbol_table`: the core table
  asks a frame Python defines, a `SymbolTableFrame` subclass, its own
  `is_structurally_equivalent` and `serialize_to_dict` once per such frame
  it holds, and reads its `name` once when it is added.
- Keep no fallback. The package requires the extension: importing
  `fhy_core` raises `ImportError` when `fhy_core._rs` is missing, fails to
  import, or does not match the package version (`fhy_core._extension`).
  A switched module defines only its Rust-backed classes; there is no
  pure-Python copy to keep in parity and no switch that selects one. Rust
  tests in `rust/fhy-core/tests/` specify the concept's behavior, and a
  Python interface suite covers the Python API over it. Modules not yet
  ported stay ordinary Python on top of the Rust-backed types.

### Module paths follow Rust layering

A Rust module's path follows the crate's layering, not the Python package.
Each public item has exactly one public path, every `pub use` is explicit
(no globs), and CI rejects an item re-exported under a second path. A
module depends only on the layers before it:

1. `identifier`, `interned`, `foreign`, the serialized form of parts other
   implementations define and the one boxed error, `BoxError`, their hooks
   report, and `error`, the errors every layer shares, such as the
   `UnknownNameError` a name enum's `FromStr` refuses with
2. `described_tag`, `value_domain`, `provenance`
3. `diagnostic` and `op_attribute`, whose tags are `described_tag`
   vocabularies
4. `tree`, `term` and `lattice`, which do not depend on one another
5. `expression` (with `expression::pattern` and `expression::builtins`) and `pass`, which do
   not depend on each other
6. `expression::passes`, the passes over expressions, which depends on both,
   and `solver`, the questions about expressions and their backends, which
   depends on `expression` and never on `pass`; and `types`, the IR type system, which
   depends on `expression` and never on `pass`
7. `constraint`, the constraints over identifiers, which depends on `solver`
   and never on `pass`
8. `symbol_table`, the symbol table and its frames, which depends on `types` and
   never on `pass`
9. `param`, the value domains of params and their questions, which depends
   on `constraint` and never on `pass`
10. `stack` and `scope`, the last-in, first-out stack and the lexical
    scope, which depend on no other module, each other included

A module with submodules is a `foo.rs` file next to a `foo/` directory;
there are no `mod.rs` files. A private module is never named `core`, which
shadows the `core` crate. A port records its Python module in this table,
the one place that maps Python paths to Rust ones:

| Python | Rust |
|---|---|
| `fhy_core.identifier` | `fhy_core::identifier` |
| `fhy_core.traits.interned` | `fhy_core::interned` |
| `fhy_core.diagnostic` | `fhy_core::diagnostic` |
| `fhy_core.provenance` | `fhy_core::provenance` |
| `fhy_core.op_attribute` | `fhy_core::op_attribute` |
| `fhy_core.value_domain` | `fhy_core::value_domain` |
| `fhy_core.symbolic.symbol_type` | `fhy_core::expression` (`SymbolType`) |
| `fhy_core.symbolic.expression` (`core`, `errors`, `pprint`, `sort`) | `fhy_core::expression` |
| `fhy_core.symbolic.expression.builtins` | `fhy_core::expression::builtins` |
| `fhy_core.symbolic.expression.registry`, `passes.inline` | `fhy_core::expression::registry` |
| `fhy_core.symbolic.expression.passes.evaluate`, `passes.numpy`, `passes.native_lowering` | `fhy_core::expression::evaluate` |
| `fhy_core.symbolic.expression.pattern` (`core`, `rewrite`) | `fhy_core::expression::pattern`; the rule-applier pass is in `fhy_core::expression::passes` |
| `fhy_core.symbolic.expression.passes` | `fhy_core::expression::passes` |
| `fhy_core.pass_infrastructure` | `fhy_core::pass`; tree traversal is in `fhy_core::tree` |
| `fhy_core.symbolic.solver`, `symbolic.expression.passes.z3` (the lowering) | `fhy_core::solver` |
| `fhy_core.symbolic.expression.passes.sympy` (the lowering, simplification and lifting) | the binding (`fhy-core-py`'s `solver::sympy`), a `fhy_core::solver::Simplifier` |
| `fhy_core.symbolic.solver.GroundSimplifier` (`SolverBackend.GROUND`, `GROUND_THEN_SYMPY`) | `fhy_core::solver::{GroundSimplifier, GroundWithFallback}`; the binding's `solver::ground` is the Python class |
| `fhy_core.term` | `fhy_core::term`; the derived-equivalence engine, which reads Python dataclasses, is in the binding |
| `fhy_core.lattice`, `fhy_core.utils.poset` | `fhy_core::lattice` |
| `fhy_core.types` (`core`, `dispatch`) | `fhy_core::types`; the `singledispatch` registration of Python-defined types stays in Python |
| `fhy_core.types.checking` | `fhy_core::types::checking`; the body-check pass stays a Python `CompilerPass` over it |
| `fhy_core.symbolic.constraint` | `fhy_core::constraint`; the Python-defined constraints and the member objects only Python compares reach it through the binding's adapters |
| `fhy_core.symbolic.param` (`values`, `domains`) | `fhy_core::param`; the Python-defined domains and the values only Python compares or orders reach it through the binding's adapters |
| `fhy_core.symbol_table` | `fhy_core::symbol_table`; the abstract `SymbolTableFrame` that Python-defined frames subclass stays in Python |
| `fhy_core.utils.stack` | `fhy_core::stack`, for Rust users; the Python `Stack` stays a separate Python implementation with the same behavior |
| `fhy_core.utils.scope` | `fhy_core::scope`, for Rust users; the Python `Scope` stays a separate Python implementation with the same behavior |

### Errors belong to their module

Each module defines the error types for its own operations, one type per
family of related operations; the crate has no crate-wide error enum. A
public error is a `#[non_exhaustive]` enum, or a struct with structured
fields, so callers match variants and fields rather than text, and it has
no `is_*` classifiers where a `kind()` or a direct match would do.
`Display` writes one lowercase line with no trailing period and does not
repeat the text of its `source()`, which returns the underlying cause.
`Display` and `std::error::Error` are implemented by hand. The binding
converts each core error it raises through its local `IntoPyErr` trait. For
`identifier` and `interned`, which are defined in both languages, it raises
the Python implementation's exception class with the same message. For
every other module, Rust defines the behavior: the binding raises the
exception class the Python API documents, with the Rust error's
`Display` text.

### Public enums and structs

A public enum that may gain variants is `#[non_exhaustive]`. An enum that
passes and callers match exhaustively, such as `ExpressionKind`, the
operation enums, `LiteralValue`, `Callee` or `Provenance`, stays exhaustive
and says why in an `#[expect(clippy::exhaustive_enums, reason = "...")]`;
the workspace lints `clippy::exhaustive_enums` and
`clippy::exhaustive_structs` reject any other exhaustive public enum, or
struct with only public fields.

### Python parity is limited to dual-defined concepts

Rust matches the Python implementation's behavior and text only for
concepts defined in both languages at once, today `identifier` and
`interned`. Code that exists only to match Python starts its doc comment
with "Matches the Python implementation:". Everywhere else, Rust
conventions decide: `true`/`false`, Rust's shortest round-trip float
formatting (positional for a magnitude in `[1e-5, 1e16)` and with an
exponent outside it, such as `1e300`, the one text the core writes of a
float), lowercase error messages, `Display` impls instead of Python
`repr` emulation, and names without `get_` or `list_` prefixes, with
shipped defaults as associated functions such as
`OpAttribute::commutative()`. Rustdoc describes Rust behavior and does not
narrate the Python implementation.
`stack` and `scope` are implemented twice, natively in each language and
with no binding between them: the two share their behavior, which one
list of test cases pins in both test suites, and each keeps its own
language's errors and names.

### Binding crate layout

`fhy-core-py` is a library, and `fhy-core-ext`'s one `#[pymodule]` declares
`fhy_core._rs` by calling its `register(py, module)`, which `lib.rs` lists
by hand: a new class or function is added to it, in the part for its core
module. Each core module's bindings live in a file of the same name. The
Python namespace of `_rs` stays flat, since PyO3 submodules cannot be
imported as packages.

A conversion that needs nothing but the value implements the local
`IntoPyErr` trait of `error.rs` for a core error. A conversion that needs
context, such as the interpreter token, the other operand, or the objects
a call has seen, is a free function that takes it, named for what it
converts: `fn …_to_py(…, context)` for an error and
`fn …_to_python(…, context)` for a value.

Shared helpers live in their own files, and no module writes its own:
- `python.rs`: `cached_attr!` and `ImportedAttr`, for an attribute of a
  Python module imported on first use, and `Seed`, the contents a private
  seed class hands a class's `__new__`, taken once;
- `exceptions.rs`: one `ExceptionClass` per Python exception class the
  binding raises, and `unbox_py_err` for a Python exception a core error
  boxed;
- `object_table.rs`: `ObjectTable`, the Python objects of the nodes and
  identifiers a call has seen, so that a node the call returns keeps its
  object;
- `scoped.rs`: `ScopedStack`, a thread-local stack whose guard pops its
  frame, on unwind included;
- `gc.rs`: the slots through which a class takes part in cyclic garbage
  collection.

An imported attribute is kept for the life of the process, so
monkeypatching or reloading its module afterwards does not reach the
binding.
`src/fhy_core/_rs.pyi` is written by hand, and `tests/test_rs_stub.py`
checks its names and parameters against the built extension.

### The ground simplifier

`fhy_core::solver::GroundSimplifier` is a `Simplifier` that needs neither
Python nor SymPy. It is a driver over an ordered list of strategies
(`fhy_core::solver::strategy`): each strategy is a local rewrite of one
node whose children are already simplified, one concern each, and the
default list is exact integer and rational arithmetic, comparisons, logical
operators, a decided `piecewise`, exact built-ins, registered constants and
the form of decimal literals. The driver rewrites bottom-up, tries the
strategies in order on each node until none rewrites it, and stops at a
documented bound of rewrites (`with_max_rewrites`, 100 000 by default). A
caller adds, removes and reorders strategies with `with_strategy`,
`with_strategy_first`, `without` and `empty`, without touching the driver.

The contract every strategy keeps, and every change to one:

- **A rewrite is exactly what the SymPy backend returns for the node it
  rewrites, or the strategy declines.** Exactly means the same expression,
  in the form the SymPy lifting gives: an `Int` literal, a `Bool`, a decimal
  literal for a rational some binary float equals (negated when negative),
  and the quotient of two integers otherwise.
- It never approximates. It declines a float, a free identifier, a user
  function, an irrational or undefined value, and anything else it is not
  sure SymPy answers alike.
- It is local, deterministic and assumes nothing about the other
  strategies; it returns `None` when it has nothing to rewrite.
- A strategy that is not sure of SymPy's form of a partly folded
  expression leaves it: the driver returns a rewritten expression only when
  it is decided, a literal in SymPy's form (`with_partial_rewrites` is for
  strategies that match SymPy's form of a larger one). It folds every
  branch of a piecewise, the ones it does not take too, because SymPy lowers
  them all and raises on a modulo by zero in any.

`GroundWithFallback` is the composition that asks another simplifier for
what the ground one declines; the Python class `GroundSimplifier`
(`SolverBackend.GROUND`) is the ground simplifier with its default
strategies, and `GroundSimplifier(fallback)` (`SolverBackend.GROUND_THEN_SYMPY`
with SymPy) is the chain. SymPy stays the default solver's simplifier.

To add a strategy:

1. Implement `SimplificationStrategy` in a file of
   `rust/fhy-core/src/solver/strategy/`, with rustdoc that says what it
   rewrites and declines, and add it to `default_strategies` and the table
   in `strategy.rs` if it should run by default.
2. Test it alone in `rust/fhy-core/tests/it/solver/ground_strategy_stories.rs`
   (what it rewrites and what it declines through `rewrite`, and in a
   simplifier holding only it), and add to
   `ground_stories.rs` what the whole pipeline does with it.
3. Add its cases to the differential tests in
   `rust/fhy-core-py/src/solver/sympy/ground_differential.rs`, which check
   each strategy's rewrites against SymPy as the oracle, on tables of nodes
   and on random nodes, and the default pipeline on random trees. They need
   Python with SymPy, as the SymPy backend's stories do:
   `cargo test -p fhy-core-py ground_differential` (see "Rust test layout").
   `... ground_differential::timing -- --ignored --nocapture`, in a release
   build, prints the cost of the strategies' pipeline against SymPy's.

### Rust test layout

`fhy-core`'s integration tests form one binary, `rust/fhy-core/tests/it/`,
with one module per area mirroring the crate's modules. Shared helpers
live in the `support` module (`tests/it/support.rs` and
`tests/it/support/`) at `pub(crate)`, so a helper no test uses is a
dead-code warning. A helper that only one test module uses lives in that
module. A test that needs a fresh process, because it moves process-global
state further than an ordinary test tolerates, is its own test target, a
file `tests/<name>.rs` beside `tests/it/` with exactly one `#[test]` and a
comment saying why, and it is added to the target list the CI `rust` job
checks. Today there is one: `id_cap_decode`, which moves the id counter
to `ADVANCE_CAP`. Nothing re-executes a test binary to get a fresh process; a
test that needs a process without SymPy is a Python subprocess test
(`test_missing_sympy_reports_unavailable`).

`fhy-core` needs no Python: `cargo test -p fhy-core`, with any features,
runs in a shell with no Python environment at all. The binding's tests do:
the SymPy backend's stories are `#[cfg(test)]` modules of `fhy-core-py`'s
`solver::sympy`, and `cargo test -p fhy-core-py`, and so `cargo test
--workspace`, builds a test binary that links libpython, embeds an
interpreter and imports SymPy, and fails them, with the recipe, when SymPy
cannot be imported. The `convert::numpy` stories likewise need `numpy`; the `convert::param` stories need only the standard library. Build and run them with `PYO3_PYTHON` naming a Python
that has a shared libpython and the `sympy` and `numpy` packages, `PYTHONPATH` naming
that Python's `site-packages` (an embedded interpreter does not read a
virtualenv's `pyvenv.cfg`), and `LD_LIBRARY_PATH` naming its libpython's
directory when the loader does not find it. A Python built without a
shared libpython, such as a distribution's `python3.11` without
`libpython3.11.so`, fails to link with `unable to find library
-lpython3.11`; a uv-managed CPython has one:

```bash
G=$PWD/target/gate-python
uv python install --no-bin --install-dir "$G/pythons" 3.11
uv venv --python "$G"/pythons/cpython-3.11*/bin/python3.11 "$G/venv"
VIRTUAL_ENV="$G/venv" uv pip install sympy numpy
export PYO3_PYTHON="$G/venv/bin/python"
export PYTHONPATH="$G/venv/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="$(echo "$G"/pythons/cpython-3.11*/lib)"
```

### Canonical values keep their identity in Python

When a canonical Rust value, such as an interned `OpAttribute`, reaches
Python, the binding returns the same Python object for the same canonical
instance every time, so `is` holds exactly as it does for values interned
in Python. The binding crate keeps that cache, an `IdentityCache` per
interned class in `rust/fhy-core-py/src/kit/interned.rs`; the core crate holds
no Python objects.

## Creating a new Pull Request
When submitting a pull request, we ask you to check the following:

1. First create an issue on *FhY* Core to reference before starting a pull request and discuss
   possible implementation details, or nuances.

2. Unit tests, documentation, and code style are in order.
   1. It's also OK to submit work in progress if you're unsure of what this exactly means, in which case you'll likely be asked to make some further changes.

3. The contributed code will be licensed under *FhY*'s [license](https://github.com/actlab-fhy/FhY/blob/main/LICENSE). If you did not write the code yourself, you ensure the existing license is compatible and include the license information in the contributed files, or obtain permission from the original author to relicense the contributed code.


## Coding style

Most of our code is automatically linted and formatted using [ruff](https://docs.astral.sh/ruff/), and type-checked with [ty](https://github.com/astral-sh/ty) (advisory, while it is in preview) and [mypy](https://mypy.readthedocs.io/) in strict mode.
For reference, we also take inspiration from [Google's style guide](https://google.github.io/styleguide/pyguide.html).

Methods that override a base class or implement a `Protocol` method must be decorated with `@override` (imported from `fhy_core.utils.override`, which resolves to `typing.override` on Python 3.12+ and `typing_extensions.override` below it).
mypy's `explicit-override` check enforces this.

### Doctstrings

We are slightly picky about docstrings.
We use google style docstrings in active voice.
The first line should succintly summarize the function or class, ending in a period.
Further explanation may be provided on other lines after a break.
`Arguments`, `Returns` , and `Raises` should be documented in public functions.
Other sections are optional, and should be provided as seen fit, for example a `Usage` or `Notes` section may be helpful.
