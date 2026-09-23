# *FhY* Core

[![PyPI version](https://img.shields.io/pypi/v/fhy_core.svg)](https://pypi.org/project/fhy_core/)
[![Python versions](https://img.shields.io/pypi/pyversions/fhy_core.svg)](https://pypi.org/project/fhy_core/)
[![CI](https://github.com/actlab-fhy/FhY-core/actions/workflows/python-package.yml/badge.svg)](https://github.com/actlab-fhy/FhY-core/actions/workflows/python-package.yml)
[![codecov](https://codecov.io/gh/actlab-fhy/FhY-core/branch/main/graph/badge.svg)](https://codecov.io/gh/actlab-fhy/FhY-core)

*FhY* Core provides the shared foundation used by the *FhY* compiler and its companion tooling: identifier and symbol management, an extensible expression and type system, parameter and constraint modeling, pass/analysis/validation
infrastructure, serialization, reusable compiler traits, and supporting data structures and utilities.
Each utility is independently usable and designed to be extended downstream.

The expression, constraint, and parameter families, together with their shared symbol-type vocabulary and solver seam, live under the `fhy_core.symbolic` namespace: `fhy_core.symbolic.expression`, `.constraint`, `.param`, `.solver`, and `.symbol_type`.

| Utility                                  | Description                                                            |
| :--------------------------------------: | :--------------------------------------------------------------------- |
| Identifier                               | Globally unique identity (`Identifier`) pairing a human-readable name hint with a process-unique ID for stable referencing of compiler entities. |
| Error                                    | Base `FhYError` hierarchy and a registration mechanism for downstream packages to declare their own typed compiler errors. |
| Expression                               | Pure-expression IR (literals, identifiers, unary/binary/piecewise/call nodes) with a pretty printer, sort-aware type checker, sympy and z3 lowering, and a process-wide function registry (`RegisteredFunction` for expression bodies, `NativeFunction` for Python-backed math, `NativeConstant` for named values like `pi`/`e`) driven by a `FunctionSort` signature system (`BOOL`/`NAT`/`INT`/`REAL`). A native constant is referenced through its canonical identifier, `get_native_constant_identifier(name)`; `IdentifierExpression(Identifier("pi"))` is an ordinary variable, not the constant. `simplify_expression` and the NumPy evaluator raise `NativeConstantBindingError` when an environment binds a constant the expression references, and constraint evaluation reports such a binding as UNDECIDED. `BUILTIN_FUNCTIONS` and `BUILTIN_CONSTANTS` seed the standard transcendental, trigonometric, rounding, and activation helpers. `FunctionInliner` and `ExpressionEvaluator` (both built on `RewritablePass[Expression]`) compose into the rewriter pipeline. `PiecewiseExpression` is an ordered, total first-match-wins conditional (built via the `piecewise()` helper or `Expression.piecewise`), lowered by every backend (sympy, z3, numpy) and rendered by the pretty printer. |
| Expression Pattern Matching              | `Pattern` algebra over the expression IR (`WildcardPattern`, `CapturePattern`, `LiteralPattern`, `IdentifierPattern`, `UnaryExpressionPattern`/`BinaryExpressionPattern`/`PiecewiseExpressionPattern`/`CallExpressionPattern`, `PredicatePattern`, `AlternativesPattern`) with `match_pattern`/`does_pattern_match` returning `MatchBindings` keyed by capture name. `RewriteRule` pairs a pattern with a rewrite function, and `apply_rewrite_rule`/`apply_rewrite_rules`/`RewriteRuleApplier` drive rule application across an expression tree. |
| Constraint                               | Logical constraint objects (`EquationConstraint`, `InSetConstraint`, `NotInSetConstraint`) built over expressions, suitable for parameter bounds, guards, and solver-facing predicates. Each constraint's `evaluate_with_bindings`/`is_satisfied_with_bindings` decide it against a `Mapping[Identifier, value]` rather than a single value, which is what makes multi-variable (dependent) constraints usable. `ConstraintSystem` (built via `create_constraint_system`) is the companion set-level value object: a canonically ordered conjunction of constraints, possibly spanning several variables, with `check_satisfiability`/`check_satisfiability_with_bindings` deciding joint satisfiability through the solver seam. |
| Solver                                   | `fhy_core.symbolic.solver`, the backend-agnostic entry point for symbolic queries: `simplify_expression` (SymPy-backed), plus `check_expression_satisfiability`, `does_expression_imply`, and `holds_for_all_free_assignments` (Z3-backed), each returning `True`/`False`/`None`, where `None` means the query is undecided: either Z3 returned unknown, or the solver refused a query it cannot lower soundly (a native constant, a non-finite literal, a Boolean coerced to a number, a partial operation, or an int/float equality). An ill-typed query raises `NonBooleanLogicalOperandError`. `assert_expression_implies` and `assert_holds_for_all_free_assignments` raise `UndecidableError` instead of returning `None`. Backend selection is explicit via `SolverBackend`; an unsupported `SolverBackend`/`SolverQueryKind` pairing raises `SolverCapabilityError`, and `get_backend_capabilities` reports what each backend can answer. Free identifiers are typed for the query via a `SymbolType` (`REAL`/`INT`/`BOOL`, `fhy_core.symbolic.symbol_type`). |
| Parameter                                | Real, integer, ordinal, categorical, and permutation parameter types with constraint attachment and sampling/validation hooks for tuning and search. Interval-integer parameters support arithmetic, including `__mul__`, that widens bounds soundly; `create_union_param`/`create_intersection_param` (equivalently `__or__`/`__and__`) combine two parameters into the union or intersection of their feasible value sets. |
| Types                                    | Extensible type system with open dispatchers for binding, substitution, unification, and structural equivalence; expression type checking layered on top. |
| Symbol Table                             | Lexically nested symbol table with scope push/pop, shadowing rules, and lookup utilities for compiler frontends. |
| Pass Infrastructure                      | `CompilerPass`, `VisitablePass`, `RewritablePass`, `AnalysisVisitablePass`, and `register_pass` for authoring IR passes, with `PassInfo`/`PassResult`/`PreservedAnalyses` metadata and `PassRegistrationError`/`PassValidationError`/`PassExecutionError` for typed failures. |
| Pass Manager                             | `PassManager` sequences transformations and returns `PassManagerResult`/`PassRunRecord`; `FixpointPassGroup` drives until-fixpoint iteration with `FixpointGroupRecord`/`FixpointIterationRecord` traces. |
| Analysis Manager                         | `Analysis`/`AnalysisVisitablePass` with `AnalysisManager` for caching analysis results and invalidating them across pass runs. |
| Validation Manager                       | `ValidationManager` runs every validator against the IR (collect-all, never fail-fast) and aggregates their diagnostics into a single report. |
| Verification Registry                    | `VerificationRegistry` + `@register_verification` declare per-class verification passes (walking MRO); `run_verification` and `VerificationAnalysis` execute them, backing the default `VerifiableMixin.verify`. |
| Diagnostic                               | `Diagnostic`/`DiagnosticLevel` and `Note` (tagged by an open `NoteKind` registry of message roles) for structured compiler messages, plus `ValidationReport` for aggregating them and `ValidationFailedError` (raised by `ValidationReport.raise_if_failed`) for surfacing ERROR diagnostics as exceptions. |
| Serializable Trait                       | `Serializable`/`WrappedFamilySerializable` with dict, JSON, and binary formats plus registered type IDs for round-tripping IR and metadata. |
| Value Domain                             | `ValueDomain` open registry classifying the kind of value an IR operation handles (e.g., `DATA_DOMAIN`, `ADDRESS_DOMAIN`), with optional parent hierarchies. |
| Op Attribute                             | `OpAttribute` open registry of semantic tags attachable to compiler operations (`COMMUTATIVE`, `ASSOCIATIVE`, `PURE`, `ELEMENTWISE`). |
| Identifier Trait                         | `HasIdentifier` mixin giving an object a stable `Identifier` for referencing across passes. |
| Provenance Trait                         | `HasProvenance` mixin attaching source location/origin metadata for diagnostics and traceability. |
| Compiler Traits - Type Carrier           | `HasType` mixin for nodes that carry an explicit, queryable type. |
| Compiler Traits - Operands               | `HasOperands` mixin exposing a uniform operand interface for operation/expression nodes. |
| Compiler Traits - Results                | `HasResults` mixin for operation-like nodes producing one or more named results. |
| Compiler Traits - Rewritable             | `Rewritable` protocol and `RewritableMixin` for IR nodes that can be reconstructed from a new child sequence; the inverse of `Visitable.get_visit_children`, used by generic bottom-up rewriters such as `RewritablePass`. |
| Compiler Traits - Freezing               | `Frozen` protocol and `FrozenMixin` for runtime and dataclass immutability, with optional auto-freeze-on-init and `FrozenMutationError`/`FrozenValidationError` for mutation and verification failures. |
| Compiler Traits - Equality               | `PartialEqual`/`Equal` for dataclass-aware structural equality. |
| Compiler Traits - Ordering               | `PartialOrderable`/`Orderable` for dataclass-aware ordering and comparison. |
| Compiler Traits - Verification           | `Verifiable` protocol and `VerifiableMixin` for self-verifying IR nodes; `verify()` returns the aggregated diagnostic report from passes registered via `register_verification`. `VerificationError` remains for fail-fast structural-correctness helpers. |
| Compiler Traits - Canonicalization       | `Canonicalizable` hook for local rewrites into a canonical form. |
| Compiler Traits - Structural Equivalence | `StructuralEquivalence` for shape- and value-level comparisons between IR fragments. |
| Term Traits - Alpha Equivalence          | `AlphaEquivalence` protocol and `AlphaEquivalenceMixin` for comparing IR fragments up to consistent identifier renaming, driven by an injective `AlphaRenaming` map; `is_identifier_mapping_alpha_equivalent_under` is the per-`Identifier` building block used by expression nodes. |
| Term Traits - Derived Equivalence        | `DerivedEquivalenceMixin` derives both structural and alpha equivalence from a dataclass's fields, with the `compared_as_value`/`compared_as_reference`/`compared_as_binder`/`compared_with`/`excluded_from_equivalence` field markers to control how each field participates. |
| Term Traits - Binding                    | `BinderMixin` with the `Term`/`HasFreeIdentifiers` protocols for IR nodes that introduce a binding scope; derives alpha-equivalence, free-identifier computation, and capture-avoiding substitution from a few per-node hooks. |
| Compiler Traits - Interned               | `Interned` mixin for components with hash-consed, deduplicated instances. |
| Data Structure - Lattice                 | Order-theoretic lattice built on a POSET, with join/meet operations for dataflow-style analyses. |
| _General Utility_ - Logging              | Centralized logging configuration and helpers shared by all compiler components. |
| _General Utility_ - Python 3.11 Enums    | Backports of `StrEnum` and `IntEnum` semantics introduced in Python 3.11. |
| _General Utility_ - Stack                | Lightweight stack wrapping `collections.deque` with a clearer interface. |
| _General Utility_ - Scope                | `Scope` stack of LIFO name-binding frames with innermost-first shadowing lookup and a context manager for scoped push/pop; generalizes the scoping used by `SymbolTable`. |
| _General Utility_ - POSET                | Partially ordered set represented as a directed graph, with reachability and transitive-closure queries. |
| _General Utility_ - Dictionary Utilities | Helper functions for common dictionary manipulations not covered by the standard library. |
| _General Utility_ - Numeric Predicates   | `is_strict_int` rejects `bool` so contexts requiring a strict integer do not silently accept `True`/`False`. |


## Quick Examples

Piecewise expressions -- an ordered, total, first-match-wins conditional:

```python
from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression import IdentifierExpression, piecewise

x = IdentifierExpression(Identifier("x"))
bucket = piecewise((x >= 90, 4), (x >= 80, 3), otherwise=0)
```

Multi-variable constraint bindings and joint satisfiability via `ConstraintSystem`:

```python
from fhy_core.identifier import Identifier
from fhy_core.symbolic.constraint import EquationConstraint, create_constraint_system
from fhy_core.symbolic.expression import IdentifierExpression
from fhy_core.symbolic.symbol_type import SymbolType

x, y = Identifier("x"), Identifier("y")
x_lt_y = EquationConstraint(IdentifierExpression(x) < IdentifierExpression(y))
system = create_constraint_system(x_lt_y)

system.is_satisfied_with_bindings({x: 3, y: 5})                      # True
system.check_satisfiability({x: SymbolType.INT, y: SymbolType.INT})  # ConstraintOutcome.SATISFIED
```

Parameter multiplication and set algebra (`create_union_param`/`create_intersection_param`):

```python
from fhy_core.symbolic.param import (
    create_categorical_param,
    create_interval_integer_param_between,
    create_intersection_param,
    create_union_param,
)

scaled = create_interval_integer_param_between(0, 10) * 2  # widens to [0, 20]
precisions = create_union_param(
    create_categorical_param(["fp16", "fp32"]), create_categorical_param(["bf16"])
)
tile_sizes = create_intersection_param(
    create_interval_integer_param_between(0, 10),
    create_interval_integer_param_between(5, 15),
)
```

The backend-agnostic solver seam (`fhy_core.symbolic.solver`):

```python
from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression import IdentifierExpression
from fhy_core.symbolic.solver import check_expression_satisfiability
from fhy_core.symbolic.symbol_type import SymbolType

z = Identifier("z")
check_expression_satisfiability(IdentifierExpression(z) > 0, {z: SymbolType.INT})  # True
```

## Installation

### Install from PyPI

```bash
pip install fhy_core
```

### Build from Source

This project uses [uv](https://docs.astral.sh/uv/) for environment and dependency management and [maturin](https://www.maturin.rs/) for building the Rust extension.
[Install uv](https://docs.astral.sh/uv/getting-started/installation/), then:

1. Clone the repository.

    ```bash
    git clone https://github.com/actlab-fhy/FhY-core.git
    cd FhY-core
    ```

2. Create the environment and install the package.

    ```bash
    # Runtime dependencies only
    uv sync --no-default-groups

    # For contributors (default dev group: test, lint, type, property, nox, pre-commit)
    uv sync
    ```

   `uv sync` creates `.venv` and installs `fhy_core` in editable mode.
   It also compiles the Rust extension `fhy_core._rs` with maturin, so building from source needs a Rust toolchain (stable, 1.85 or newer; `rust-toolchain.toml` selects the channel for rustup).
   Editable mode covers only the Python sources: edits under `src/` take effect immediately, while the compiled extension changes only when it is rebuilt (see [Rebuilding the Python Extension](#rebuilding-the-python-extension)).
   Prefix commands with `uv run` (e.g. `uv run python`) or activate the environment with `source .venv/bin/activate`.
   Contributors also have three opt-in nox sessions not run by default: `uv run nox -s property` (the Hypothesis property suite under the thorough profile, on both backends), `uv run nox -s golden_expanded` (large random golden corpora from the Python oracle, replayed by the Rust equivalence tests; needs `cargo`), and `uv run nox -s mutation -- <module>` (cosmic-ray mutation testing for one module); see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Rust Crate

Parts of FhY Core are implemented in Rust, in the crate `fhy-core` under `rust/fhy-core/`: identifiers, interning, and the `OpAttribute` and `ValueDomain` tag types. The crate serves two purposes:

- **Standalone Rust library**: usable by any Rust project. The crate is not published to crates.io, so depend on it through git: `fhy-core = { git = "https://github.com/actlab-fhy/FhY-core.git" }`. Cargo finds the crate in the repository's workspace by package name.
- **Python extension module**: the separate `fhy-core-py` crate under `rust/fhy-core-py/` depends on `fhy-core` and wraps it with [PyO3](https://pyo3.rs/) bindings. [maturin](https://www.maturin.rs/) compiles it into the Python package as `fhy_core._rs`, which currently backs identifier id allocation. The Python API is the same with or without the extension.

`fhy-core` itself has no PyO3 dependency, so pure-Rust consumers never pull in a Python dependency; only `fhy-core-py` does.

The `testing` feature exposes `fhy_core::testing`, the Rust counterpart of `fhy_core.testing_patches`. Inside a `DeterministicIdentifierScope` scope, identifiers created with the same name hint compare equal, so a test can compare an object graph whose identifiers were created inside the code under test with one it built itself. A scope belongs to the thread that entered it; other threads join it through a handle from `share()`. Enable the feature only for tests:

```toml
[dev-dependencies]
fhy-core = { git = "https://github.com/actlab-fhy/FhY-core.git", features = ["testing"] }
```

### Building and Testing the Rust Crate

Requires a stable Rust toolchain (1.85+). The `rust-toolchain.toml` at the repo root pins the channel.

```bash
# Check the pure-Rust library (no Python dependency)
cargo check -p fhy-core

# Check the PyO3 extension crate too
cargo check -p fhy-core-py

# Run all Rust tests, as CI does
cargo test --workspace --locked --all-features

# Format and lint, as CI does
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
```

### Rebuilding the Python Extension

`uv sync` builds the extension: maturin compiles `fhy-core-py` and installs the native module as `fhy_core._rs`. The project's uv cache keys cover `rust/**/*.rs`, every workspace member's `Cargo.toml`, the root `Cargo.toml`, and `Cargo.lock`, so after a Rust edit the next `uv sync`, or any `uv run` (which syncs first), rebuilds the extension.

```bash
# Rebuild the extension after editing Rust sources
uv sync

# Force a rebuild even when no cache key changed
uv sync --reinstall-package fhy-core

# Verify the Rust backend is selected
uv run python -c "import fhy_core; print(fhy_core.RUST_BACKEND_SELECTED)"
# => True
```

To rebuild in place with an unoptimized, incremental build while iterating on Rust, install [maturin](https://www.maturin.rs/) 1.9.4 or newer (`uv tool install maturin`) and run `uv run --no-sync maturin develop --uv`. Run later commands with `uv run --no-sync` as well: a syncing `uv run` reinstalls uv's own build over the one `maturin develop` installed.

The backend is selected once, when `fhy_core` is imported. The package runs on the Rust extension iff the extension imports, its `__version__` matches the installed `fhy_core` package, and the `FHY_CORE_NO_EXTENSIONS` environment variable does not disable it; otherwise it runs on its pure-Python implementation. The variable disables the extension when it holds anything other than an empty string or one of `0`, `false`, `no`, and `off` (case-insensitive, ignoring surrounding whitespace), so `FHY_CORE_NO_EXTENSIONS=1` forces the pure-Python backend even when the extension is installed. An extension whose `__version__` does not match the installed package, or that has no `__version__` at all, is stale and falls back to the pure-Python backend with a `RuntimeWarning`, the same as an extension that fails to import or that is built only for other Python versions (for example after the virtual environment's interpreter changes); `uv sync` rebuilds such an extension. The package version is set once, in the workspace `Cargo.toml`: maturin derives the Python package version from it by PEP 440 normalization, and the extension reports it as written, so a Cargo `0.3.0-rc.1` extension matches the `0.3.0rc1` package. `fhy_core.RUST_BACKEND_SELECTED` reports the selection: it is `True` only when the extension is installed, importable, at the installed package's version, and not disabled by `FHY_CORE_NO_EXTENSIONS`. The public API is the same on both backends: `fhy_core.Identifier`, for example, draws its ids from the selected backend's counter, and exactly one counter issues ids in a process.

### Testing the Python Package

```bash
# Run the full Python test suite on the Rust backend
# (uses pytest with xdist for parallelism; uv run rebuilds a stale extension first)
uv run pytest

# Run the full Python test suite on the pure-Python backend
FHY_CORE_NO_EXTENSIONS=1 uv run pytest

# Run specific test modules
uv run pytest tests/test_identifier.py

# Run the suite on both backends for every supported Python version
uv run nox -s tests

# Run it on both backends for a single Python version
uv run nox -s tests-3.12

# Run it on one backend for a single Python version
uv run nox -s "tests-3.12(backend='python')"

# Run the property-based test suite (Hypothesis, thorough profile) on both backends
uv run nox -s property
```

The `tests` nox session is parametrized over the backend: each Python version runs the full suite once with `FHY_CORE_NO_EXTENSIONS=0` (Rust) and once with `FHY_CORE_NO_EXTENSIONS=1` (pure Python), and a session fails before testing if the package does not report the backend it was asked for. A default `uv run nox` therefore tests both backends. Tests that compare the two implementations directly, such as `tests/test_identifier_rust_binding.py`, run in both sessions whenever the extension is installed. Tests can read `fhy_core.RUST_BACKEND_SELECTED` to tell which backend they run on.

## Contributing

Interested in contributing to *FhY* Core? See the [contribution guide](CONTRIBUTING.md).
