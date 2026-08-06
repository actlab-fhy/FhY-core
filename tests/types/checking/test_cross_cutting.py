"""Cross-cutting tests that span multiple modules in `fhy_core.types.checking`."""

import pytest

from fhy_core.pass_infrastructure import CompilerPass, PassInfo
from fhy_core.types.checking.type_checker import ExpressionTypeChecker

# =============================================================================
# Pass registry - every type-checking pass must self-register
# =============================================================================

_EXPECTED_REGISTRATIONS: list[tuple[str, type]] = [
    ("fhy_core.types.checking.type_checker", ExpressionTypeChecker),
]


@pytest.mark.parametrize("pass_name, pass_class", _EXPECTED_REGISTRATIONS)
def test_checking_pass_is_registered_under_expected_name(
    pass_name: str, pass_class: type
) -> None:
    """Test each type-checking pass registers under its expected name and class."""
    registered = CompilerPass.get_registered_passes()
    assert pass_name in registered
    info = registered[pass_name]
    assert isinstance(info, PassInfo)
    assert info.pass_type is pass_class
    assert info.description.strip() != ""
