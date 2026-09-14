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

## Property tests

Some modules carry Hypothesis-based property tests alongside the example
suite. Use the following rules to decide which kind of test to write, and
how to place it.

**When to write a property.** If ten well-chosen hand-picked inputs would
convince a reviewer, write ten examples instead. Write a property only when
the input space is combinatorial and you can name an independent oracle for
it (a reference evaluator, an inverse function, a brute-force enumeration,
an algebraic law). A property that only re-derives the answer the same way
the code does is a tautology, not a test. Examples keep three jobs a
property cannot do: pinning an exact error message or exception type,
pinning a golden value (serialization shape, `repr`/`str` text), and serving
as a readable worked example or user story. Exhaustive parametrize tables
over a finite domain also stay as examples.

**Where they live.** A property test file sits next to the unit it tests,
named `test_<unit>_properties.py` alongside `test_<unit>.py`. Set
`pytestmark = pytest.mark.property` at module level, and call
`pytest.importorskip("hypothesis")` as the first statement, since
`hypothesis` lives in the `property` dependency group and the `tests` lane
never installs it. Mark any property that exercises the Z3 bridge with
`pytest.mark.z3`.

**Shared strategies** live in `tests/strategies/`, one module per domain.
Import them only from property files, never from a `conftest.py`; a
`conftest.py` must import cleanly without `hypothesis` installed.

**Profiles.** A bare local `pytest` runs under the `dev` profile (25
examples). The release gate runs under `thorough` (400 examples,
derandomized): set `HYPOTHESIS_PROFILE=thorough`, or just run
`uv run nox -s property`, which sets it for you. Mutation runs use a third
profile, `mutation` (25 examples, derandomized, no example database), which
`scripts/run-mutation.sh` selects so every mutant sees the same draws.
Every property uses `deadline=None`; under `pytest-xdist`, scheduler
contention rather than test cost is what trips a deadline. Add
`@settings(max_examples=N)` on top of the profile only when a test is
genuinely expensive (a Z3 call, a large tree).

**Hazards specific to this codebase.**

- Function-scoped fixtures do not reset between examples: Hypothesis runs
  the test body many times inside one pytest test. Never request the
  `function_registry_snapshot` fixture in a `@given` test; write it as an
  example test instead if it must touch the registry.
- `Expression.__bool__` raises. Never put an `Expression` in a truth
  context, in a strategy or in a reference evaluator.
- Logical connectives and piecewise conditions must be boolean-sorted.
  Generate sort-aware trees so no draw is ever refused by
  `validate_logical_operands` or a bridge.
- `st.data()` blocks `@example`. Put dependent draws in a `@st.composite`
  strategy instead.

**Lifting an example into a property.** When an example test compares the
implementation to an oracle on hand-picked inputs, lift it: write the
property against a strategy from `tests/strategies`, pin every hand-picked
input with `@example(...)`, and delete the example test, all in the same
commit. Never use `assume()` to rescue a bad strategy, and never suppress a
health check; a health check firing is a strategy bug to fix.

**CI policy.** The `property` job only runs on pull requests into `main`
and on manual dispatch; it does not run on pull requests into `dev`. Run
`uv run nox -s property` locally before opening a release pull request.

Mutation testing measures whether a property earned its place. Each
targeted module has its own config under `cosmic-ray/`; run one with
`uv run nox -s mutation -- <module>` or directly with
`scripts/run-mutation.sh <module>` (both default to `lattice`).

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
