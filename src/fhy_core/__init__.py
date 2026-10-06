"""FhY compiler core utilities.

The public API is organized by subsystem. Import what you need from the relevant
submodule (``fhy_core.symbolic.param``, ``fhy_core.symbolic.expression``,
``fhy_core.types``, ``fhy_core.types.checking``, ``fhy_core.symbolic.constraint``,
``fhy_core.traits``, ``fhy_core.term``, ``fhy_core.pass_infrastructure``,
``fhy_core.serialization``, ``fhy_core.search_space``, and so on), or reach
it as a namespace after ``import fhy_core`` (e.g.
``fhy_core.symbolic.param.create_integer_param``). Only the ownerless
primitives used across every subsystem are re-exported at the top level.

The package requires its compiled Rust extension, ``fhy_core._rs``: importing
``fhy_core`` raises ``ImportError`` naming the cause when the extension is
missing, fails to import, or does not match the installed package's version.
"""

from importlib.metadata import version

# Checks the required extension before any module imports it.
from . import _extension  # isort: skip
from . import (
    diagnostic,
    error,
    identifier,
    lattice,
    logger,
    op_attribute,
    pass_infrastructure,
    provenance,
    search_space,
    serialization,
    symbol_table,
    symbolic,
    term,
    testing_patches,
    traits,
    types,
    utils,
    value_domain,
)
from .identifier import Identifier

__version__ = version("fhy_core")

__all__ = [
    "Identifier",
    "diagnostic",
    "error",
    "identifier",
    "lattice",
    "logger",
    "op_attribute",
    "pass_infrastructure",
    "provenance",
    "search_space",
    "serialization",
    "symbol_table",
    "symbolic",
    "term",
    "testing_patches",
    "traits",
    "types",
    "utils",
    "value_domain",
]
