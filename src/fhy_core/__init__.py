"""FhY compiler core utilities.

The public API is organized by subsystem. Import what you need from the relevant
submodule (``fhy_core.param``, ``fhy_core.expression``, ``fhy_core.types``,
``fhy_core.constraint``, ``fhy_core.traits``, ``fhy_core.pass_infrastructure``,
``fhy_core.serialization``, and so on), or reach it as a namespace after
``import fhy_core`` (e.g. ``fhy_core.param.create_integer_param``). Only the
ownerless primitives used across every subsystem are re-exported at the top
level.
"""

from importlib.metadata import version

from . import (
    constraint,
    diagnostic,
    error,
    expression,
    identifier,
    lattice,
    logger,
    op_attribute,
    param,
    pass_infrastructure,
    provenance,
    serialization,
    symbol_table,
    symbol_type,
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
    "constraint",
    "diagnostic",
    "error",
    "expression",
    "identifier",
    "lattice",
    "logger",
    "op_attribute",
    "param",
    "pass_infrastructure",
    "provenance",
    "serialization",
    "symbol_table",
    "symbol_type",
    "testing_patches",
    "traits",
    "types",
    "utils",
    "value_domain",
]
