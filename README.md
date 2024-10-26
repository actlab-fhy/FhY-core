# *FhY* Core

*FhY* Core is a collection of utilities for FhY and other parts of the compiler.
The current list of utilities are as follows.


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

3. Install FhY.

    ```bash
    # Standard Installation
    pip install .

    # For contributors
    pip install ".[dev]"
    ```

## Contributing - For Developers
Want to start contributing the *FhY* Core? Please take a look at our
[contribution guide](CONTRIBUTING.md)
