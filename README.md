# *FhY* Core

[![PyPI version](https://img.shields.io/pypi/v/fhy_core.svg)](https://pypi.org/project/fhy_core/)
[![Python versions](https://img.shields.io/pypi/pyversions/fhy_core.svg)](https://pypi.org/project/fhy_core/)
[![CI](https://github.com/actlab-fhy/FhY-core/actions/workflows/python-package.yml/badge.svg)](https://github.com/actlab-fhy/FhY-core/actions/workflows/python-package.yml)
[![codecov](https://codecov.io/gh/actlab-fhy/FhY-core/branch/main/graph/badge.svg)](https://codecov.io/gh/actlab-fhy/FhY-core)
[![License](https://img.shields.io/github/license/actlab-fhy/FhY-core.svg)](LICENSE)

*FhY* Core is a collection of utilities for *FhY* and other parts of the compiler.

| Utility                                  | Description                                                            |
| :--------------------------------------: | :--------------------------------------------------------------------- |
| Identifier                               | Unique naming class with a non-unique name hint and a unique ID.       |
| Error                                    | Custom error registration and core errors for the compiler.            |
| Expression                               | General expression represented as an AST with a parser and printer.    |
| Constraint                               | General logical constraint.                                            |
| Parameter                                | Real, integer, ordinal, categorical, and permutation parameters.       |
| Types                                    | Core type system for the compiler w/ type checking for expressions.    |
| Symbol Table                             | Nested symbol table.                                                   |
| Pass Infrastructure                      | `CompilerPass`, `VisitablePass`, and `register_pass` for building IR passes with diagnostics (`PassDiagnostic`, `DiagnosticLevel`) and preserved-analysis tracking. |
| Pass Manager                             | `PassManager` sequences transformations and returns `PassManagerResult`/`PassRunRecord`; `FixpointPassGroup` drives until-fixpoint iteration. |
| Analysis Manager                         | `Analysis`/`AnalysisVisitablePass` with `AnalysisManager` for caching and invalidating analysis results across pass runs. |
| Validation Manager                       | `ValidationManager` runs every validator against the IR (collect-all, never fail-fast) and returns a `ValidationReport`; `ValidationFailedError` surfaces ERROR diagnostics. |
| Serializable Trait                       | `Serializable`/`WrappedFamilySerializable` with dict, JSON, and binary formats plus registered type IDs. |
| Compiler Traits - Identity               | `HasIdentifier` for stable object identity.                            |
| Compiler Traits - Provenance             | `HasProvenance` for source/origin tracking.                            |
| Compiler Traits - Type Carrier           | `HasType` for objects carrying an explicit type.                       |
| Compiler Traits - Operands               | `HasOperands` for operand-bearing operation/expression nodes.          |
| Compiler Traits - Results                | `HasResults` for multi-result operation-like nodes.                    |
| Compiler Traits - Freezing               | `Frozen`/`FrozenMixin` for runtime and dataclass immutability. |
| Compiler Traits - Equality               | `PartialEqual`/`Equal` for dataclass-aware equality semantics.         |
| Compiler Traits - Ordering               | `PartialOrderable`/`Orderable` for dataclass-aware ordering semantics. |
| Compiler Traits - Verification           | `Verifiable` + `VerificationError` for structural invariant checks.    |
| Compiler Traits - Folding                | `Foldable` for constant-fold-like evaluation hooks.                    |
| Compiler Traits - Canonicalization       | `Canonicalizable` for local canonical form rewrites.                   |
| Compiler Traits - Structural Equivalence | `StructuralEquivalence` for shape/value-level IR comparisons.          |
| Compiler Traits - Interned               | `Interned` for interned components.                                    |
| Data Structure - Lattice                 | Lattice (order theory) data structure represented with a POSET.        |
| _General Utility_ - Logging              | Core logging utilities for all compiler components.                    |
| _General Utility_ - Python 3.11 Enums    | String and integer enum types only introduced in Python 3.11           |
| _General Utility_ - Stack                | General stack utility that wraps `deque`.                              |
| _General Utility_ - POSET                | General partially ordered set utility represented as a directed graph. |
| _General Utility_ - Dictionary Utilities | Additional dictionary helper functions.                                |


## Table of Contents
- [Installing *FhY* Core](#installing-fhy-core)
  - [Install *FhY* Core from PyPi](#install-fhy-core-from-pypi)
  - [Build *FhY* Core from Source Code](#build-fhy-core-from-source-code)
- [Contributing - For Developers](#contributing---for-developers)

### Install *FhY* Core from PyPi
**Coming Soon**

### Build *FhY* Core from Source Code

1. Clone the repository from GitHub.

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

3. Install *FhY*.

    ```bash
    # Standard Installation
    pip install .

    # For contributors
    pip install ".[dev]"
    ```

## Contributing - For Developers
Want to start contributing the *FhY* Core? Please take a look at our
[contribution guide](CONTRIBUTING.md)
