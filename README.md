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
| Expression                               | Pure-expression AST (literals, identifiers, unary/binary ops) with a textual parser, pretty printer, visitor/transformer base classes, and structural utilities. |
| Constraint                               | Logical constraint objects built over expressions, suitable for parameter bounds, guards, and solver-facing predicates. |
| Parameter                                | Real, integer, ordinal, categorical, and permutation parameter types with constraint attachment and sampling/validation hooks for tuning and search. |
| Types                                    | Extensible type system with open dispatchers for binding, substitution, unification, and structural equivalence; expression type checking layered on top. |
| Symbol Table                             | Lexically nested symbol table with scope push/pop, shadowing rules, and lookup utilities for compiler frontends. |
| Pass Infrastructure                      | `CompilerPass`, `VisitablePass`, and `register_pass` for authoring IR passes, with `PassDiagnostic`/`DiagnosticLevel` reporting and preserved-analysis tracking. |
| Pass Manager                             | `PassManager` sequences transformations and returns `PassManagerResult`/`PassRunRecord`; `FixpointPassGroup` drives until-fixpoint iteration. |
| Analysis Manager                         | `Analysis`/`AnalysisVisitablePass` with `AnalysisManager` for caching analysis results and invalidating them across pass runs. |
| Validation Manager                       | `ValidationManager` runs every validator against the IR (collect-all, never fail-fast) and returns a `ValidationReport`; `ValidationFailedError` surfaces ERROR diagnostics. |
| Serializable Trait                       | `Serializable`/`WrappedFamilySerializable` with dict, JSON, and binary formats plus registered type IDs for round-tripping IR and metadata. |
| Value Domain                             | `ValueDomain` open registry classifying the kind of value an IR operation handles (e.g., `DATA_DOMAIN`, `ADDRESS_DOMAIN`), with optional parent hierarchies. |
| Op Attribute                             | `OpAttribute` open registry of semantic tags attachable to compiler operations (`COMMUTATIVE`, `ASSOCIATIVE`, `PURE`, `ELEMENTWISE`). |
| Compiler Traits - Identity               | `HasIdentifier` mixin giving an object a stable `Identifier` for referencing across passes. |
| Compiler Traits - Provenance             | `HasProvenance` mixin attaching source location/origin metadata for diagnostics and traceability. |
| Compiler Traits - Type Carrier           | `HasType` mixin for nodes that carry an explicit, queryable type. |
| Compiler Traits - Operands               | `HasOperands` mixin exposing a uniform operand interface for operation/expression nodes. |
| Compiler Traits - Results                | `HasResults` mixin for operation-like nodes producing one or more named results. |
| Compiler Traits - Freezing               | `Frozen`/`FrozenMixin` for runtime and dataclass immutability with explicit freeze semantics. |
| Compiler Traits - Equality               | `PartialEqual`/`Equal` for dataclass-aware structural equality. |
| Compiler Traits - Ordering               | `PartialOrderable`/`Orderable` for dataclass-aware ordering and comparison. |
| Compiler Traits - Verification           | `Verifiable` + `VerificationError` for structural invariant checks on IR nodes. |
| Compiler Traits - Folding                | `Foldable` hook for constant-fold-style evaluation of nodes. |
| Compiler Traits - Canonicalization       | `Canonicalizable` hook for local rewrites into a canonical form. |
| Compiler Traits - Structural Equivalence | `StructuralEquivalence` for shape- and value-level comparisons between IR fragments. |
| Compiler Traits - Interned               | `Interned` mixin for components with hash-consed, deduplicated instances. |
| Data Structure - Lattice                 | Order-theoretic lattice built on a POSET, with join/meet operations for dataflow-style analyses. |
| _General Utility_ - Logging              | Centralized logging configuration and helpers shared by all compiler components. |
| _General Utility_ - Python 3.11 Enums    | Backports of `StrEnum` and `IntEnum` semantics introduced in Python 3.11. |
| _General Utility_ - Stack                | Lightweight stack wrapping `collections.deque` with a clearer interface. |
| _General Utility_ - POSET                | Partially ordered set represented as a directed graph, with reachability and transitive-closure queries. |
| _General Utility_ - Dictionary Utilities | Helper functions for common dictionary manipulations not covered by the standard library. |
| _General Utility_ - Numeric Predicates   | `is_strict_int` rejects `bool` so contexts requiring a strict integer do not silently accept `True`/`False`. |


## Installation

### Install from PyPI

```bash
pip install fhy_core
```

### Build from Source

1. Clone the repository.

    ```bash
    git clone https://github.com/actlab-fhy/FhY-core.git
    ```

2. Create and prepare a Python virtual environment.

    ```bash
    cd FhY-core
    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -U pip
    pip install setuptools wheel
    ```

3. Install the package.

    ```bash
    # Standard installation
    pip install .

    # For contributors
    pip install ".[dev]"
    ```

## Contributing

Interested in contributing to *FhY* Core? See the
[contribution guide](CONTRIBUTING.md).
