"""Tests for structural rejection of malformed param serialization payloads.

Under the schema-derived format a parameter serializes as
``{"domain": {...}, "variable": {...}, "constraints": [...]}`` where ``domain``
is a wrapped family envelope. The derived ``deserialize_from_dict`` validates the
exact top-level key set and each field's coarse structure before decoding. Each
test below feeds a payload with exactly one structural fault and asserts the
shape-level rejection surfaces as `DeserializationDictStructureError`.
"""

from collections.abc import Callable
from typing import Any

import pytest

from fhy_core.identifier import Identifier
from fhy_core.param import Param, create_integer_param, create_ordinal_param
from fhy_core.serialization import (
    DeserializationDictStructureError,
    DeserializationValueError,
)

_PARAM_STRUCTURE_ERROR_MATCH = 'deserializing to "Param"'


def _valid_param_payload() -> dict[str, Any]:
    """Return a well-formed derived-format integer-param payload."""
    return create_integer_param(name=Identifier("x")).serialize_to_dict()


def _drop_domain_field(payload: dict[str, Any]) -> None:
    """Remove the ``domain`` field from a param payload."""
    del payload["domain"]


def _drop_variable_field(payload: dict[str, Any]) -> None:
    """Remove the ``variable`` field from a param payload."""
    del payload["variable"]


def _drop_constraints_field(payload: dict[str, Any]) -> None:
    """Remove the ``constraints`` field from a param payload."""
    del payload["constraints"]


def _add_unexpected_field(payload: dict[str, Any]) -> None:
    """Add an unexpected extra key to a param payload."""
    payload["unexpected"] = 1


def _set_variable_to_non_dict(payload: dict[str, Any]) -> None:
    """Replace the ``variable`` field with a non-dict value."""
    payload["variable"] = "not-a-dict"


def _set_constraints_to_non_list(payload: dict[str, Any]) -> None:
    """Replace the ``constraints`` field with a non-list value."""
    payload["constraints"] = "not-a-list"


def _set_domain_to_non_dict(payload: dict[str, Any]) -> None:
    """Replace the ``domain`` field with a non-dict value."""
    payload["domain"] = "not-a-dict"


# =============================================================================
# Top-level param payload - derived structure rejection
# =============================================================================


@pytest.mark.parametrize(
    "corrupt_payload",
    [
        pytest.param(_drop_domain_field, id="missing-domain-field"),
        pytest.param(_drop_variable_field, id="missing-variable-field"),
        pytest.param(_drop_constraints_field, id="missing-constraints-field"),
        pytest.param(_add_unexpected_field, id="unexpected-extra-field"),
        pytest.param(_set_variable_to_non_dict, id="variable-not-a-dict"),
        pytest.param(_set_constraints_to_non_list, id="constraints-not-a-list"),
        pytest.param(_set_domain_to_non_dict, id="domain-not-a-dict"),
    ],
)
def test_rejects_payload_with_structural_fault(
    corrupt_payload: Callable[[dict[str, Any]], None],
) -> None:
    """Test the deserializer rejects a payload with a single structural fault."""
    payload = _valid_param_payload()
    corrupt_payload(payload)

    with pytest.raises(
        DeserializationDictStructureError, match=_PARAM_STRUCTURE_ERROR_MATCH
    ):
        Param.deserialize_from_dict(payload)


# =============================================================================
# Finite-domain wrapped-leaf value rejection
# =============================================================================


def test_rejects_finite_domain_with_unwrapped_value() -> None:
    """Test an ordinal payload with an unwrapped (raw) value is rejected.

    The value list under ``domain.__data__.sorted_values`` must be wrapped
    registry dicts; a bare integer is rejected by the wrapped-leaf codec.
    """
    payload: dict[str, Any] = create_ordinal_param([1, 2, 3]).serialize_to_dict()
    payload["domain"]["__data__"]["sorted_values"] = [1, 2, 3]

    with pytest.raises(DeserializationValueError, match="sorted_values"):
        Param.deserialize_from_dict(payload)
