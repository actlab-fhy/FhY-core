"""Tests pinning the `fhy_core` package namespace shape.

Covers two contracts. `fhy_core.__all__` re-exports only the ownerless
primitives, so the family-owned `expression`, `constraint`, `param`, and
`symbol_type` names are absent and `symbolic` stands in for them. And
`fhy_core.symbolic` is a namespace of its five submodules plus the shared
`SymbolType` vocabulary: it does not flatten any submodule's contents,
and neither it nor `symbolic.expression` re-exports the solver queries
that live in `symbolic.solver`.
"""

import types

import fhy_core
import fhy_core.symbolic
import fhy_core.symbolic.expression
from fhy_core.symbolic import solver

_SOLVER_QUERY_NAMES = (
    "simplify_expression",
    "check_expression_satisfiability",
    "does_expression_imply",
    "holds_for_all_free_assignments",
    "assert_expression_implies",
    "assert_holds_for_all_free_assignments",
)


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
        "RUST_BACKEND_SELECTED",
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
        "term",
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


def test_symbolic_namespace_exports_only_its_submodules_and_symbol_type() -> None:
    """Test `fhy_core.symbolic` flattens nothing out of its submodules.

    Asserts exact set equality, not mere membership: re-exporting a
    submodule's contents here (`simplify_expression`, `Param`, and so on)
    gives the same name two documented homes, which is what the curated
    namespace exists to prevent.
    """
    assert set(fhy_core.symbolic.__all__) == {
        "SymbolType",
        "constraint",
        "expression",
        "param",
        "solver",
        "symbol_type",
    }


def test_symbolic_namespace_reexports_symbol_type() -> None:
    """Test `fhy_core.symbolic.SymbolType` is re-exported at the family level."""
    assert "SymbolType" in fhy_core.symbolic.__all__
    assert fhy_core.symbolic.SymbolType is fhy_core.symbolic.symbol_type.SymbolType


def test_solver_owns_its_full_query_surface() -> None:
    """Test every query this module names is exported by `symbolic.solver`."""
    for name in _SOLVER_QUERY_NAMES:
        assert name in solver.__all__
        assert hasattr(solver, name)


def test_solver_query_functions_are_absent_from_the_family_namespaces() -> None:
    """Test only `symbolic.solver` exports the solver query functions.

    Reaching a query through `fhy_core.symbolic` or through
    `symbolic.expression` would make the solver seam optional, letting a
    caller bypass the documented entry point.
    """
    for name in _SOLVER_QUERY_NAMES:
        assert name not in fhy_core.symbolic.__all__
        assert not hasattr(fhy_core.symbolic, name)
        assert name not in fhy_core.symbolic.expression.__all__
        assert not hasattr(fhy_core.symbolic.expression, name)
