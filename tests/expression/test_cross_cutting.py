"""Cross-cutting tests that span multiple modules in `fhy_core.expression`."""

import pytest

from fhy_core.expression.passes.sympy import (
    ExpressionToSympyConverter,
    SymPyToExpressionConverter,
)
from fhy_core.expression.passes.type_checker import ExpressionTypeChecker
from fhy_core.expression.passes.z3 import ExpressionToZ3Converter
from fhy_core.pass_infrastructure import CompilerPass, PassInfo

# =============================================================================
# Pass registry - every expression pass must self-register
# =============================================================================

_EXPECTED_REGISTRATIONS: list[tuple[str, type]] = [
    ("fhy_core.expression.type_checker", ExpressionTypeChecker),
    ("fhy_core.expression.from_sympy", SymPyToExpressionConverter),
    ("fhy_core.expression.to_sympy", ExpressionToSympyConverter),
    ("fhy_core.expression.to_z3", ExpressionToZ3Converter),
]


@pytest.mark.parametrize("pass_name, pass_class", _EXPECTED_REGISTRATIONS)
def test_expression_pass_is_registered_under_expected_name(
    pass_name: str, pass_class: type
) -> None:
    """Test each expression pass registers under its expected name and class."""
    registered = CompilerPass.get_registered_passes()
    assert pass_name in registered
    info = registered[pass_name]
    assert isinstance(info, PassInfo)
    assert info.pass_type is pass_class
    assert info.description.strip() != ""
