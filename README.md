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
| Expression                               | Pure-expression IR (literals, identifiers, unary/binary/piecewise/call nodes) with a pretty printer, sort-aware type checker, lowering to sympy, SMT-LIB2 and z3, and a process-wide function registry (`RegisteredFunction` for expression bodies, `NativeFunction` for Python-backed math, `NativeConstant` for named values like `pi`/`e`) driven by a `FunctionSort` signature system (`BOOL`/`NAT`/`INT`/`REAL`), backed by the Rust core's registry. A native constant is referenced through its canonical identifier, `get_native_constant_identifier(name)`, which for a built-in constant has a fixed id in every process (`pi` 48, `e` 49, `inf` 50, `nan` 51); `IdentifierExpression(Identifier("pi"))` is an ordinary variable, not the constant. `simplify_expression` and the NumPy evaluator raise `NativeConstantBindingError` when an environment binds a constant the expression references, and constraint evaluation reports such a binding as UNDECIDED. `BUILTIN_FUNCTIONS` and `BUILTIN_CONSTANTS` seed the standard transcendental, trigonometric, rounding, and activation helpers. `FunctionInliner` (a `CompilerPass` over the Rust inliner, linear in the distinct nodes, so nested calls such as `relu(relu(...))` inline at any depth) and `ExpressionEvaluator` (a `CompilerPass` over the Rust fold, which computes native built-ins with the core's IEEE kernels and registered natives with their Python implementations) compose into the rewriter pipeline. `evaluate_expression_with_numpy` evaluates an expression over NumPy arrays in the Rust core, in the `bool`, `int64` and `float64` domains, with checked integers and element failures that a piecewise or a connective discards where it does not need the element; it computes the transcendental natives with NumPy's ufuncs. `PiecewiseExpression` is an ordered, total first-match-wins conditional (built via the `piecewise()` helper or `Expression.piecewise`), lowered by every backend (sympy, SMT-LIB2, z3) and the evaluators, and rendered by the pretty printer (`pformat_expression`, or the `ExpressionPrettyFormatter` pass). |
| Expression Pattern Matching              | `Pattern` algebra over the expression IR (`WildcardPattern`, `CapturePattern`, `LiteralPattern`, `IdentifierPattern`, `UnaryExpressionPattern`/`BinaryExpressionPattern`/`LogicalExpressionPattern`/`PiecewiseExpressionPattern`/`CallExpressionPattern`, `PredicatePattern`, `AlternativesPattern`), backed by the Rust core, with `match_pattern`/`does_pattern_match` returning `MatchBindings` keyed by `Capture` handles, which compare by identity. `RewriteRule` pairs a pattern with a rewrite function and optional guards, the `Rule` ABC lets a rule be written in Python, and `apply_rewrite_rule`/`apply_rewrite_rules`/`RewriteRuleApplier` drive rule application across an expression tree in one bottom-up pass of any depth, which rewrites a shared node once, records each firing as a `FiredRule`, and reports a failing rule as a `RewriteError`. |
| Constraint                               | Logical constraint objects (`EquationConstraint`, `InSetConstraint`, `NotInSetConstraint`) built over expressions, suitable for parameter bounds, guards, and solver-facing predicates. Each constraint's `evaluate_with_bindings`/`is_satisfied_with_bindings` decide it against a `Mapping[Identifier, value]` rather than a single value, which is what makes multi-variable (dependent) constraints usable. The three kinds run on the Rust core: members compare type-strictly (`True`, `1` and `1.0` are three members) and are kept in one canonical order, by kind and then by value, which `values`, `repr`, the payload and `convert_to_expression` share; a `Serializable` member is compared by Python's `==`. `Constraint` stays an abstract base third parties subclass. `ConstraintSystem` (built via `create_constraint_system`) is the companion set-level value object: a canonically ordered conjunction of constraints, possibly spanning several variables, with `check_satisfiability`/`check_satisfiability_with_bindings` deciding joint satisfiability through the solver seam; the system runs on the Rust core too, asks the default solver, and drives a `Constraint` subclass defined in Python through its own methods. |
| Solver                                   | `fhy_core.symbolic.solver`, questions about expressions answered by pluggable backends, with the query logic in the Rust core: `simplify_expression`, plus `check_expression_satisfiability`, `does_expression_imply`, and `holds_for_all_free_assignments`, each returning `True`/`False`/`None`, where `None` means the question is undecided: either the backend answered unknown, or the solver refused a question it cannot lower soundly (a native constant, a non-finite literal, a Boolean coerced to a number, a partial operation, or an int/float equality). An ill-typed query raises `NonBooleanLogicalOperandError`. `assert_expression_implies` and `assert_holds_for_all_free_assignments` raise `UndecidableError` instead of returning `None`. A `Solver` holds an `SmtSolver` for the logical questions (the z3-solver adapter that `SolverBackend.Z3` names, `SmtLib2ProcessSolver` for any SMT-LIB2 executable such as `z3 -in` or `cvc5 --lang=smt2`, or a Python subclass) and a `Simplifier` (`SympySimplifier`, the Rust core's SymPy backend that `SolverBackend.SYMPY` names, which lowers to SymPy, simplifies and lifts back in Rust, or a Python subclass). The functions take `backend`: `None` asks the default solver, which `set_default_solver` replaces for the constraints and params too, and a `SolverBackend` member asks its adapter; naming a backend that cannot answer raises `SolverCapabilityError`, and `get_backend_capabilities` reports what each backend can answer. A question whose backend's package is not installed raises `SolverBackendUnavailableError`. `convert_expression_to_smtlib2` returns an expression's SMT-LIB2 script. Free identifiers are typed for the query via a `SymbolType` (`REAL`/`INT`/`BOOL`, `fhy_core.symbolic.symbol_type`). |
| Parameter                                | Real, integer, ordinal, categorical, and permutation parameter types with constraint attachment and sampling/validation hooks for tuning and search. Interval-integer parameters support arithmetic, including `__mul__`, that widens bounds soundly; `create_union_param`/`create_intersection_param` (equivalently `__or__`/`__and__`) combine two parameters into the union or intersection of their feasible value sets. `Param`, `ParamAssignment`, the value domains and their procedures (value checks, feasibility, subsets, bounds, interval arithmetic, union and intersection) run on the Rust core: values match type-strictly (`True`, `1` and `1.0` are three values), ordinal values ascend numerically across `bool`, `int` and `float`, categories keep the constraint members' canonical order, and a `Serializable` value is compared and ordered by its own `==` and `<`. `ParamDomain` stays an abstract base a new kind of parameter subclasses. |
| Types                                    | Extensible type system backed by the Rust core: `PrimitiveDataType`/`TemplateDataType` data types and `NumericalType`/`IndexType` types, which compare and hash structurally, `CoreDataType` promotion over the integer and float/complex promotion orders, and binding, substitution and unification into a `TypeUnificationEnvironment` through open `functools.singledispatch` dispatchers; a `Type` or `DataType` subclass Python defines takes part through the handlers it registers. Expression type checking (`fhy_core.types.checking`) is layered on top: the bidirectional checker, the sort tables and the function-body checks run in the Rust core, calling back only the caller's identifier lookup and a custom call-target resolver. |
| Symbol Table                             | `SymbolTable` of namespaces, each naming an optional parent namespace and mapping its symbols to frames (`ImportSymbolTableFrame`, `VariableSymbolTableFrame`, `FunctionSymbolTableFrame`, or a `SymbolTableFrame` subclass Python defines), backed by the Rust core: a lookup in a namespace walks up its parents, a namespace never redefines a symbol an ancestor defines, and `verify()` reports missing or cyclic parents and frames that name another symbol. |
| Pass Infrastructure                      | `CompilerPass`, `VisitablePass`, `RewritablePass`, `AnalysisVisitablePass`, and `register_pass` for authoring IR passes, with `PassInfo`/`PassResult`/`PreservedAnalyses` metadata and `PassRegistrationError`/`PassValidationError`/`PassExecutionError` for typed failures. Backed by the Rust core: a pass's Python hooks run through the core's lifecycle, the defaults of the hooks a class does not override run in Rust, and a failure names the pass and the Python hook, keeps the run's diagnostics, and nests the error of a pass run inside a hook. Run statistics come from each run (`PassResult.skipped`, `PassManagerResult.run_count()`). |
| Pass Manager                             | `PassManager` sequences transformations and returns `PassManagerResult`/`PassRunRecord`; `FixpointPassGroup` drives until-fixpoint iteration with `FixpointGroupRecord`/`FixpointIterationRecord` traces. A run verifies its input and every changed output with the registered verification passes, blaming the producing pass, unless `set_verifier` replaces the verifier; `ValidationManager` runs `Validator`s and passes collect-all into a `ValidationReport` of `ValidatorRecord`s. |
| Analysis Manager                         | `Analysis`/`AnalysisVisitablePass`, with results cached per IR node for one pipeline run and carried to a pass's output when the pass preserves them; a hook reads them with `get_analysis` or through the `AnalysisManager` view `get_analysis_manager()` returns. |
| Validation Manager                       | `ValidationManager` runs every validator against the IR (collect-all, never fail-fast) and aggregates their diagnostics into a single report. |
| Verification Registry                    | `VerificationRegistry` + `@register_verification` declare per-class verification passes (walking MRO), in a registry backed by the Rust core; `run_verification` and `VerificationAnalysis` execute them collect-all, backing the default `VerifiableMixin.verify`. |
| Diagnostic                               | `Diagnostic`/`DiagnosticLevel` and `Note` (tagged by an open `NoteKind` registry of message roles) for structured compiler messages, plus `ValidationReport` for aggregating them and `ValidationFailedError` (raised by `ValidationReport.raise_if_failed`) for surfacing ERROR diagnostics as exceptions. |
| Serializable Trait                       | `Serializable`/`WrappedFamilySerializable` with dict, JSON, and binary formats plus registered type IDs for round-tripping IR and metadata. Payloads are written in the V2 wire format, the Rust core's own serde shapes, so a Rust-backed value's `to_json()` is byte-identical to what the Rust crate writes, and Rust reads it back; a Python-defined part of a Rust value, such as a third-party `Constraint` in a `ConstraintSystem`, travels as a foreign part, `{"type_id": .., "data": ..}`, that its registered class decodes. The older `{"__type__": .., "__data__": ..}` envelope format (V1) is deprecated: it is still read, and written inside `with wire_version(WireVersion.V1):`, both with a `DeprecationWarning`, until a later release (0.4.0 is proposed) removes it. Convert stored V1 payloads with `fhy_core.serialization.upgrade_v1_payload`, or `python -m fhy_core.serialization_upgrade old.json > new.json`. |
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
| Term Traits - Alpha Equivalence          | `AlphaEquivalence` protocol and `AlphaEquivalenceMixin` for comparing IR fragments up to consistent identifier renaming, driven by `AlphaRenaming`, an immutable stack of injective binder frames over a free renaming, backed by the Rust core; `is_identifier_mapping_alpha_equivalent_under` compares identifier-keyed mappings. |
| Term Traits - Derived Equivalence        | `DerivedEquivalenceMixin` derives both structural and alpha equivalence from a dataclass's fields, with the `compared_as_value`/`compared_as_reference`/`compared_as_binder`/`compared_with`/`excluded_from_equivalence` field markers to control how each field participates. The comparison runs in Rust, at any depth, and compares identifiers, expressions and nested derived values without calling Python. |
| Term Traits - Binding                    | `BinderMixin` with the `Term`/`HasFreeIdentifiers` protocols for IR nodes that introduce a binding scope; derives alpha-equivalence, free-identifier computation, and capture-avoiding substitution from a few per-node hooks, through the Rust core's binder algorithms. A binder list that repeats an identifier pairs with none, so its node is alpha-equivalent to no node, itself included. |
| Compiler Traits - Interned               | `Interned` mixin for components with hash-consed, deduplicated instances. |
| Data Structure - Lattice                 | Order-theoretic lattice over hashable elements, backed by the Rust core, with join/meet operations and a `verify()` report of the pairs that lack one, for dataflow-style analyses. |
| _General Utility_ - Logging              | Centralized logging configuration and helpers shared by all compiler components. |
| _General Utility_ - Python 3.11 Enums    | Backports of `StrEnum` and `IntEnum` semantics introduced in Python 3.11. |
| _General Utility_ - Stack                | Lightweight stack wrapping `collections.deque` with a clearer interface. |
| _General Utility_ - Scope                | `Scope` stack of LIFO name-binding frames with innermost-first shadowing lookup and a context manager for scoped push/pop; generalizes the scoping used by `SymbolTable`. |
| _General Utility_ - POSET                | Partially ordered set over hashable elements, backed by the Rust core: reflexive order queries answered from each element's up-set, and a topological iteration that insertion order (or `iter_stable`'s key) makes deterministic. |
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

The package runs on its compiled Rust extension, `fhy_core._rs`, and cannot run without it. A wheel includes the extension; installing from a source distribution compiles it, which needs a Rust toolchain (stable, 1.85 or newer).

The solver's shipped backends are optional extras, each imported only when a question needs it:

```bash
pip install "fhy_core[z3]"       # the z3 SMT backend: satisfiability, implication, universal validity
pip install "fhy_core[sympy]"    # the sympy simplifier: simplification, and so validating values of params with equation constraints
pip install "fhy_core[solvers]"  # both
pip install "fhy_core[numpy]"    # the NumPy evaluator, evaluate_expression_with_numpy
```

Without a backend's package, a question that needs it raises `SolverBackendUnavailableError`, which names the extra to install; everything that asks no solver question, such as finite domains, set constraints and enumeration, works without either. Without NumPy, `evaluate_expression_with_numpy` raises `ImportError` naming the `numpy` extra; `import fhy_core`, `evaluate_expression` and everything else never import NumPy. `SmtLib2ProcessSolver` needs no Python package: it drives any SMT-LIB2 executable, such as `z3 -in`, and a `Solver` holding it can become the default with `set_default_solver`.

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

    # Runtime dependencies with the solver backends
    uv sync --no-default-groups --extra solvers

    # For contributors (default dev group: test, lint, type, property, nox, pre-commit)
    uv sync
    ```

   `uv sync` creates `.venv` and installs `fhy_core` in editable mode.
   It also compiles the Rust extension `fhy_core._rs` with maturin, so building from source needs a Rust toolchain (stable, 1.85 or newer; `rust-toolchain.toml` selects the channel for rustup).
   Editable mode covers only the Python sources: edits under `src/` take effect immediately, while the compiled extension changes only when it is rebuilt (see [Rebuilding the Python Extension](#rebuilding-the-python-extension)).
   Prefix commands with `uv run` (e.g. `uv run python`) or activate the environment with `source .venv/bin/activate`.
   Contributors also have three opt-in nox sessions not run by default: `uv run nox -s property` (the Hypothesis property suite under the thorough profile), `uv run nox -s golden_expanded` (large random golden corpora from the Python oracle, replayed by the Rust equivalence tests; needs `cargo`), and `uv run nox -s mutation -- <module>` (cosmic-ray mutation testing for one module); see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Rust Crate

Parts of FhY Core are implemented in Rust, in the crate `fhy-core` under `rust/fhy-core/`. Its modules are `identifier`, `interned`, `described_tag`, `diagnostic`, `op_attribute`, `value_domain`, `provenance`, `tree`, `expression` (with `expression::builtins`, `expression::pattern` and `expression::passes`) and `pass`; the crate's [README](rust/fhy-core/README.md) describes each. Where a concept is defined in both languages (`identifier`, `interned`), the Rust behavior matches Python's; elsewhere Rust defines it. The crate serves two purposes:

- **Standalone Rust library**: usable by any Rust project, from crates.io: `fhy-core = "0.2"`.
- **Python extension module**: the separate `fhy-core-py` crate under `rust/fhy-core-py/` depends on `fhy-core` and wraps it with [PyO3](https://pyo3.rs/) bindings. [maturin](https://www.maturin.rs/) compiles it into the Python package as `fhy_core._rs`, which the package requires: the interned tags, diagnostics, provenance, expressions, patterns, the pass infrastructure, the term traits' renaming and engines, the constraints, the params, and the SymPy simplifier's lowering, simplification and lifting are Rust-backed, and `Identifier` draws its ids from the Rust counter.

`fhy-core` itself has no PyO3 dependency, so pure-Rust consumers never pull in a Python dependency; only `fhy-core-py` does.

### Building and Testing the Rust Crate

Requires a stable Rust toolchain (1.85+). The `rust-toolchain.toml` at the repo root pins the channel.

```bash
# Check the pure-Rust library (no Python dependency)
cargo check -p fhy-core

# Check the PyO3 extension crate too
cargo check -p fhy-core-py

# Run all Rust tests, as CI does
cargo test --workspace --locked --all-features

# Format, lint and document, as CI does
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked
```

### Rebuilding the Python Extension

`uv sync` builds the extension: maturin compiles `fhy-core-py` and installs the native module as `fhy_core._rs`. The project's uv cache keys cover `rust/**/*.rs`, every workspace member's `Cargo.toml`, the root `Cargo.toml`, and `Cargo.lock`, so after a Rust edit the next `uv sync`, or any `uv run` (which syncs first), rebuilds the extension.

```bash
# Rebuild the extension after editing Rust sources
uv sync

# Force a rebuild even when no cache key changed
uv sync --reinstall-package fhy-core

# Verify the extension imports
uv run python -c "import fhy_core._rs as rs; print(rs.__version__)"
```

To rebuild in place with an unoptimized, incremental build while iterating on Rust, install [maturin](https://www.maturin.rs/) 1.9.4 or newer (`uv tool install maturin`) and run `uv run --no-sync maturin develop --uv`. Run later commands with `uv run --no-sync` as well: a syncing `uv run` reinstalls uv's own build over the one `maturin develop` installed.

The package requires the extension. Importing `fhy_core` imports it first and raises `ImportError`, naming the cause and the fix, when it is not installed, is built only for other Python versions (for example after the virtual environment's interpreter changes), fails to import, or is stale: its `__version__` is missing or does not match the installed `fhy_core` package. `uv sync` rebuilds such an extension. The package version is set once, in the workspace `Cargo.toml`: maturin derives the Python package version from it by PEP 440 normalization, and the extension reports it as written, so a Cargo `0.3.0-rc.1` extension matches the `0.3.0rc1` package.

### Testing the Python Package

```bash
# Run the full Python test suite
# (uses pytest with xdist for parallelism; uv run rebuilds a stale extension first)
uv run pytest

# Run specific test modules
uv run pytest tests/test_identifier.py

# Run the suite under coverage for every supported Python version
uv run nox -s tests

# Run it for a single Python version
uv run nox -s tests-3.12

# Run the property-based test suite (Hypothesis, thorough profile)
uv run nox -s property

# Run the suite without the optional solver packages (sympy, z3-solver)
uv run nox -s tests_minimal
```

Tests that reach a solver backend carry the `sympy` or `z3` marker and are skipped when its package is not installed, which `tests_minimal` checks: there, an unmarked test that reaches a missing backend fails.

The `*_rust_binding.py` suites cover what each binding adds over the Rust core, such as the class structure, argument checks, payload shapes and pickles; the other suites test the Python API's behavior.

## Contributing

Interested in contributing to *FhY* Core? See the [contribution guide](CONTRIBUTING.md).
