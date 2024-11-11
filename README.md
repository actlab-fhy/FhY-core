# *FhY* Core

*FhY* Core is a collection of utilities for *FhY* and other parts of the compiler.

| Utility                                  | Description                                                            |
| :--------------------------------------: | :--------------------------------------------------------------------- |
| Identifier                               | Unique naming class with a non-unique name hint and a unique ID.       |
| Error                                    | Custom error registration and core errors for the compiler.            |
| Expression                               | General expression represented as an AST with a parser and printer.    |
| Constraint                               | General logical constraint.                                            |
| Parameter                                | Real, integer, orginal, categorical, and permutation parameters.       |
| Types                                    | Core type system for the compiler.                                     |
| Memory Instance                          | Extensible record for an instance of data in memory.                   |
| Symbol Table                             | Nested symbol table.                                                   |
| _General Utility_ - Python 3.11 Enums    | String and integer enum types only introduced in Python 3.11           |
| _General Utility_ - Stack                | General stack utility that wraps `deque`.                              |
| _General Utility_ - POSET                | General partially ordered set utility represented as a directed graph. |
| _General Utility_ - Lattice              | General lattice (order theory) utility represented with a POSET.       |
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
