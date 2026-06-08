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
