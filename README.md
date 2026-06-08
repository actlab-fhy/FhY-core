# *FhY* Core

[![PyPI version](https://img.shields.io/pypi/v/fhy_core.svg)](https://pypi.org/project/fhy_core/)
[![Python versions](https://img.shields.io/pypi/pyversions/fhy_core.svg)](https://pypi.org/project/fhy_core/)
[![CI](https://github.com/actlab-fhy/FhY-core/actions/workflows/python-package.yml/badge.svg)](https://github.com/actlab-fhy/FhY-core/actions/workflows/python-package.yml)
[![codecov](https://codecov.io/gh/actlab-fhy/FhY-core/branch/main/graph/badge.svg)](https://codecov.io/gh/actlab-fhy/FhY-core)

*FhY* Core provides the shared foundation used by the *FhY* compiler and its companion tooling: identifier and symbol management, an extensible expression and type system, parameter and constraint modeling, pass/analysis/validation
infrastructure, serialization, reusable compiler traits, and supporting data structures and utilities.
Each utility is independently usable and designed to be extended downstream.

| Utility                                  | Description                                                            |
| :--------------------------------------: | :--------------------------------------------------------------------- |
| Identifier                               | Globally unique identity (`Identifier`) pairing a human-readable name hint with a process-unique ID for stable referencing of compiler entities. |
| Error                                    | Base `FhYError` hierarchy and a registration mechanism for downstream packages to declare their own typed compiler errors. |
| Expression                               | Pure-expression IR (literals, identifiers, unary/binary/ternary/call nodes) with a pretty printer, sort-aware type checker, sympy and z3 lowering, and a process-wide function registry (`RegisteredFunction` for expression bodies, `NativeFunction` for Python-backed math, `NativeConstant` for named values like `pi`/`e`) driven by a `FunctionSort` signature system (`BOOL`/`NAT`/`INT`/`REAL`). `BUILTIN_FUNCTIONS` and `BUILTIN_CONSTANTS` seed the standard transcendental, trigonometric, rounding, and activation helpers. `FunctionInliner` and `ExpressionEvaluator` (both built on `RewritablePass[Expression]`) compose into the rewriter pipeline. |
| Expression Pattern Matching              | `Pattern` algebra over the expression IR (`WildcardPattern`, `CapturePattern`, `LiteralPattern`, `IdentifierPattern`, `UnaryExpressionPattern`/`BinaryExpressionPattern`/`TernaryExpressionPattern`/`CallExpressionPattern`, `PredicatePattern`, `AlternativesPattern`) with `match_pattern`/`does_pattern_match` returning `MatchBindings` keyed by capture name. `RewriteRule` pairs a pattern with a rewrite function, and `apply_rewrite_rule`/`apply_rewrite_rules`/`RewriteRuleApplier` drive rule application across an expression tree. |
| Constraint                               | Logical constraint objects built over expressions, suitable for parameter bounds, guards, and solver-facing predicates. |
| Parameter                                | Real, integer, ordinal, categorical, and permutation parameter types with constraint attachment and sampling/validation hooks for tuning and search. |
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
| Compiler Traits - Identity               | `HasIdentifier` mixin giving an object a stable `Identifier` for referencing across passes. |
| Compiler Traits - Provenance             | `HasProvenance` mixin attaching source location/origin metadata for diagnostics and traceability. |
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
| Compiler Traits - Alpha Equivalence      | `AlphaEquivalence` protocol and `AlphaEquivalenceMixin` for comparing IR fragments up to consistent identifier renaming, driven by an injective `AlphaRenaming` map; `is_identifier_mapping_alpha_equivalent_under` is the per-`Identifier` building block used by expression nodes. |
| Compiler Traits - Derived Equivalence    | `DerivedEquivalenceMixin` derives both structural and alpha equivalence from a dataclass's fields, with the `compared_as_value`/`compared_as_reference`/`compared_as_binder`/`compared_with`/`excluded_from_equivalence` field markers to control how each field participates. |
| Compiler Traits - Binding                | `BinderMixin` with the `Term`/`HasFreeIdentifiers` protocols for IR nodes that introduce a binding scope; derives alpha-equivalence, free-identifier computation, and capture-avoiding substitution from a few per-node hooks. |
| Compiler Traits - Interned               | `Interned` mixin for components with hash-consed, deduplicated instances. |
| Data Structure - Lattice                 | Order-theoretic lattice built on a POSET, with join/meet operations for dataflow-style analyses. |
| _General Utility_ - Logging              | Centralized logging configuration and helpers shared by all compiler components. |
| _General Utility_ - Python 3.11 Enums    | Backports of `StrEnum` and `IntEnum` semantics introduced in Python 3.11. |
| _General Utility_ - Stack                | Lightweight stack wrapping `collections.deque` with a clearer interface. |
| _General Utility_ - Scope                | `Scope` stack of LIFO name-binding frames with innermost-first shadowing lookup and a context manager for scoped push/pop; generalizes the scoping used by `SymbolTable`. |
| _General Utility_ - POSET                | Partially ordered set represented as a directed graph, with reachability and transitive-closure queries. |
| _General Utility_ - Dictionary Utilities | Helper functions for common dictionary manipulations not covered by the standard library. |
| _General Utility_ - Numeric Predicates   | `is_strict_int` rejects `bool` so contexts requiring a strict integer do not silently accept `True`/`False`. |


## Installation

### Install from PyPI

```bash
pip install fhy_core
```

### Build from Source

This project uses [uv](https://docs.astral.sh/uv/) for environment and dependency management.
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
   Prefix commands with `uv run` (e.g. `uv run python`) or activate the environment with `source .venv/bin/activate`.

## Contributing

Interested in contributing to *FhY* Core? See the [contribution guide](CONTRIBUTING.md).
