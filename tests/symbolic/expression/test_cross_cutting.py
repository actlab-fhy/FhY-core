"""Cross-cutting tests that span multiple modules in `fhy_core.symbolic.expression`."""

import pytest

from fhy_core.pass_infrastructure import CompilerPass, PassInfo
from fhy_core.symbolic.expression.passes.sympy import (
    ExpressionToSympyConverter,
    SymPyToExpressionConverter,
)
from fhy_core.symbolic.expression.passes.z3 import ExpressionToZ3Converter

# =============================================================================
# Pass registry - every expression pass must self-register
# =============================================================================

_EXPECTED_REGISTRATIONS: list[tuple[str, type]] = [
    ("fhy_core.symbolic.expression.from_sympy", SymPyToExpressionConverter),
    ("fhy_core.symbolic.expression.to_sympy", ExpressionToSympyConverter),
    ("fhy_core.symbolic.expression.to_z3", ExpressionToZ3Converter),
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
