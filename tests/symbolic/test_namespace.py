"""Tests pinning the `fhy_core` package namespace shape.

Covers two contracts: `fhy_core.__all__` excludes the family-owned
`expression`, `constraint`, `param`, and `symbol_type` names and
includes `symbolic`; and `fhy_core.symbolic` aggregates its five
submodule namespaces plus the full solver surface and `SymbolType`,
while `symbolic.expression` does not export the five query functions
that live in `solver`.
"""

import types

import fhy_core
import fhy_core.symbolic
import fhy_core.symbolic.expression
from fhy_core.symbolic import solver


def test_fhy_core_top_level_excludes_the_family_owned_names() -> None:
    """Test `fhy_core.__all__` excludes the family-owned subsystem names."""
    for name in ("expression", "constraint", "param", "symbol_type"):
        assert name not in fhy_core.__all__


def test_fhy_core_top_level_exposes_symbolic_as_a_module() -> None:
    """Test `fhy_core.__all__` names `symbolic`, itself a module."""
    assert "symbolic" in fhy_core.__all__
    assert isinstance(fhy_core.symbolic, types.ModuleType)


def test_fhy_core_top_level_still_has_only_identifier_as_a_re_exported_symbol() -> None:
    """Test `Identifier` remains the only concrete re-exported symbol.

    Asserts exact set equality against the curated top-level export list,
    not mere membership: a regression that re-adds a family symbol (e.g.
    `LiteralExpression` or `Param`) to `fhy_core.__all__` alongside
    `Identifier` must fail here.
    """
    assert set(fhy_core.__all__) == {
        "Identifier",
        "diagnostic",
        "error",
        "identifier",
        "lattice",
        "logger",
        "op_attribute",
        "pass_infrastructure",
        "provenance",
        "serialization",
        "symbol_table",
        "symbolic",
        "testing_patches",
        "traits",
        "types",
        "utils",
        "value_domain",
    }


def test_symbolic_namespace_exposes_its_five_submodules() -> None:
    """Test `fhy_core.symbolic` aggregates all five family submodules."""
    for name in ("constraint", "expression", "param", "solver", "symbol_type"):
        assert name in fhy_core.symbolic.__all__
        assert hasattr(fhy_core.symbolic, name)


def test_symbolic_namespace_reexports_the_full_solver_surface() -> None:
    """Test `fhy_core.symbolic` re-exports every name in `solver.__all__`."""
    for name in solver.__all__:
        assert name in fhy_core.symbolic.__all__
        assert getattr(fhy_core.symbolic, name) is getattr(solver, name)


def test_symbolic_namespace_reexports_symbol_type() -> None:
    """Test `fhy_core.symbolic.SymbolType` is re-exported at the family level."""
    assert "SymbolType" in fhy_core.symbolic.__all__
    assert fhy_core.symbolic.SymbolType is fhy_core.symbolic.symbol_type.SymbolType


def test_solver_query_functions_are_absent_from_expression_exports() -> None:
    """Test `symbolic.expression` does not export the five solver query functions."""
    solver_query_names = (
        "simplify_expression",
        "does_expression_imply",
        "holds_for_all_free_assignments",
        "assert_expression_implies",
        "assert_holds_for_all_free_assignments",
    )
    for name in solver_query_names:
        assert name not in fhy_core.symbolic.expression.__all__
        assert not hasattr(fhy_core.symbolic.expression, name)
