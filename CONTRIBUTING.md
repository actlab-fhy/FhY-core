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
```

The `coverage` session combines the `.coverage.*` data left by the `tests`
sessions, so run `tests` first. A bare `uv run nox` runs `tests` then
`coverage` in order; running `coverage` alone on a clean tree simply skips.

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
  `uv run nox -s property`, which sets it for you and runs the suite once
  per backend, as `property(backend='rust')` and
  `property(backend='python')`. Like `tests`, each fails before testing if
  the package does not report the backend it was asked for. A failure
  prints a `@reproduce_failure` blob that replays it exactly.
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
--workspace --all-targets --all-features --locked -- -D warnings`, `cargo
test --workspace --locked --all-features`, and `cargo package --locked -p
fhy-core`, then unpacks the packaged crate and runs `cargo test --locked`
in it, so the tests pass on the crate as a consumer receives it.
`rust-msrv` runs `cargo check --workspace --lib --locked` on the
`rust-version` in `Cargo.toml`, so the MSRV covers both `fhy-core` and the
PyO3 binding crate `fhy-core-py`. That version is a promise to the crate's
consumers, so `.cargo/config.toml` has the resolver fall back to dependency
releases that build on it and `cargo update` keeps `Cargo.lock` within it.

The Rust equivalence tests replay golden corpora under `rust/fhy-core/tests/golden/`,
each recorded from the Python implementation by the `generate_*.py` script
beside it. `tests/test_golden_corpora.py` reruns every generator in a fresh
interpreter on the backend the test run selected, so the `tests` sessions
check the corpora on both backends and every supported Python. It fails,
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
the pure-Python backend and runs those ignored tests on them (it needs
`cargo`); the `golden-expanded` job runs it on the same triggers as the
`property` job. A new generator needs an entry in `EXPANDED_GOLDEN_CORPORA`
in `noxfile.py`, or the session fails.

Mutation testing measures whether a property earned its place: it makes
small changes to the source and checks that some test fails. Each targeted
module has its own config under `cosmic-ray/`; run one with
`uv run nox -s mutation -- <module>` (the module defaults to `lattice`).

## Porting to Rust

*FhY* Core is moving to Rust one module at a time. The Rust code is a
Cargo workspace with two crates. `rust/fhy-core` is the pure-Rust library
and never depends on PyO3. `rust/fhy-core-py` holds the PyO3 bindings, and
maturin builds it into the extension module `fhy_core._rs`. A port adds its
types to `fhy-core` and their bindings to `fhy-core-py`. Every port follows
these rules.

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
ships an extension of its own that links the crate. No downstream crate
links `fhy-core` yet, so `fhy-core-py` does not yet offer the library form
that such a combined module needs; it gains one before the first
downstream crate does.

### Process-global state is limited to identity

Exactly two kinds of state are process-global: the identifier id counter
and each `Interned` type's `InternRegistry`. Both are append-only: an id is
never reissued, and a canonical value is never replaced or removed.
Everything else a port keeps between calls, such as the pass registry, run
statistics or caches, is an owned value that its user creates and passes
explicitly. Where the Python API needs one shared instance, the binding
holds it in the extension's module state. A new process-global `static`
with interior mutability needs the maintainer's agreement and a line in
this section. Tests never clear a process-global registry; a test that
needs an empty or controlled registry builds a local one.

Ids `0..RESERVED_ID_COUNT` (65,536 ids) are reserved for the identifiers
the crate ships, such as the built-in tags, and each shipped identifier has
a fixed id in the crate-private reserved table. The counter, in Rust and in
the Python fallback alike, issues fresh ids from 65,536 upward, and no
payload id at or above `ID_CAP` (2^63) is decoded or restored, so no
payload can exhaust the counter. A newly shipped identifier takes an unused
id from the reserved table rather than drawing one from the counter.

### Serialization is plain serde

`fhy-core` serializes with `#[derive(Serialize, Deserialize)]` wherever it
can, in shapes Rust defines. There is no `__type__`/`__data__` envelope in
the core crate: the binding adds it where Python's serialization framework
embeds a Rust value in a Python container. Serde impls must work with
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

### Replacing a Python class

- Port bottom-up. A class switches to Rust only once everything it holds
  is already in Rust, so Rust code never stores Python objects.
- Switch in one step. The binding replaces the Python class outright; a
  Python registry and a Rust registry for the same concept are never live
  at the same time.
- Benchmark before deleting. Measure the module's hot paths (construction,
  equality, hashing, attribute access, and whatever the module does most)
  on both backends with a throwaway script, and delete the script once the
  decision is made. When the Rust-backed version is at most 10% slower
  than the Python one on every measured path, delete the pure-Python
  implementation. When it is more than 10% slower on any path, usually
  because every call crosses into the extension, the maintainer decides
  whether to keep the Python class. A kept class moves only the parts that
  gain from Rust and says why in its module docstring.
  `fhy_core.identifier` is the example: `Identifier` stays in Python and
  only its id counter runs in Rust.
- Freeze the golden corpus. Golden corpora exist only for concepts defined
  in both languages, today `identifier` and `interned`. Once a module's
  Python implementation is deleted, its generator has no oracle left to
  run. Delete the generator (the drift check and `golden_expanded` find
  generators by the `generate_*.py` pattern) and its
  `EXPANDED_GOLDEN_CORPORA` entry, and keep the committed JSON as a fixed
  regression corpus.
- From the first deletion on, the package requires the extension, and
  `FHY_CORE_NO_EXTENSIONS` selects the pure-Python implementation only for
  modules that still have one.

### Module paths follow Rust layering

A Rust module's path follows the crate's layering, not the Python package.
Each public item has exactly one public path, every `pub use` is explicit
(no globs), and CI rejects an item re-exported under a second path. A
module depends only on the layers before it:

1. `identifier`, `interned`
2. `described_tag`, `value_domain`, `provenance`
3. `diagnostic` and `op_attribute`, whose tags are `described_tag`
   vocabularies
4. `tree`
5. `expr` (with `expr::pattern` and `expr::builtins`) and `pass`, which do
   not depend on each other
6. `expr::passes`, the passes over expressions, which depends on both

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
| `fhy_core.symbolic.symbol_type` | `fhy_core::expr` (`SymbolType`) |
| `fhy_core.symbolic.expression` (`core`, `errors`, `pprint`, `sort`) | `fhy_core::expr` |
| `fhy_core.symbolic.expression.builtins` | `fhy_core::expr::builtins` |
| `fhy_core.symbolic.expression.pattern` (`core`, `rewrite`) | `fhy_core::expr::pattern`; the rule-applier pass is in `fhy_core::expr::passes` |
| `fhy_core.symbolic.expression.passes` | `fhy_core::expr::passes` |
| `fhy_core.pass_infrastructure` | `fhy_core::pass`; tree traversal is in `fhy_core::tree` |

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
exception class the replaced Python API documents, with the Rust error's
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
formatting, lowercase error messages, `Display` impls instead of Python
`repr` emulation, and names without `get_` or `list_` prefixes, with
shipped defaults as associated functions such as
`OpAttribute::commutative()`. Rustdoc describes Rust behavior and does not
narrate the Python implementation.

### Binding crate layout

`fhy-core-py` declares `fhy_core._rs` with one declarative `#[pymodule]`
in `lib.rs`. Each core module's bindings live in a file of the same name
and are exported with `#[pymodule_export]`, and implements the local
`IntoPyErr` trait of `error.rs` for the core errors they raise. The Python
namespace of `_rs` stays flat, since PyO3 submodules cannot be imported as
packages.
`src/fhy_core/_rs.pyi` is written by hand, and `tests/test_rs_stub.py`
checks its names and parameters against the built extension.

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
checks. Today the one such target is `id_cap_decode`, which moves the id
counter to `ID_CAP`. Nothing re-executes a test binary to get a fresh
process.

### Canonical values keep their identity in Python

When a canonical Rust value, such as an interned `OpAttribute`, reaches
Python, the binding returns the same Python object for the same canonical
instance every time, so `is` holds exactly as it does for values interned
in Python. The binding crate keeps that cache; the core crate never holds
Python objects. The cache is added with the first binding that returns a
canonical value.

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
