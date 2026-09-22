# Contributing to *FhY* Core

Pull requests are always welcome, and the *FhY* community appreciates any help you give.

## Working with *FhY* - For Developers

1. Download the development branch of the *FhY* Core source code.

```bash
git clone https://github.com/actlab-fhy/FhY-core.git -b dev
cd FhY-core
```

2. Install [uv](https://docs.astral.sh/uv/) (used for environment and dependency management), then create the development environment. This installs *FhY* Core in editable mode along with the default `dev` dependency group.

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

The `property` job only runs on pull requests into `main` and on manual
dispatch; it does not run on pull requests into `dev`. Run
`uv run nox -s property` locally before opening a release pull request.

The `rust` and `rust-msrv` jobs run on every pull request, and `ci-ok`
requires both. `rust` runs `cargo fmt --all --check`, `cargo
clippy --all-targets --all-features --locked -- -D warnings`, `cargo test
--locked --all-features`, and `cargo package --locked`, then unpacks the
packaged crate and runs `cargo test --locked --features testing` in it, so
the tests pass on the crate as a consumer receives it. `rust-msrv`
type-checks the library with `cargo check --lib --locked` on the
`rust-version` in `Cargo.toml`, once with default features and once with
`--all-features`. That version is a promise to the crate's consumers, so
`.cargo/config.toml` has the resolver fall back to dependency releases that
build on it and `cargo update` keeps `Cargo.lock` within it.

The Rust equivalence tests replay golden corpora under `rust/tests/golden/`,
each recorded from the Python implementation by the `generate_*.py` script
beside it. `tests/test_golden_corpora.py` reruns every generator in a fresh
interpreter on the backend the test run selected, so the `tests` sessions
check the corpora on both backends and every supported Python. It fails,
printing the regeneration command and a diff, if a committed corpus differs
outside its `provenance` block or a generator has no committed corpus. After
changing the Python behavior a corpus records, regenerate it and commit the
result.

Mutation testing measures whether a property earned its place: it makes
small changes to the source and checks that some test fails. Each targeted
module has its own config under `cosmic-ray/`; run one with
`uv run nox -s mutation -- <module>` (the module defaults to `lattice`).

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
