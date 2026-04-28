"""Tests for structural rejection of malformed param serialization payloads.

Each public ``deserialize_from_dict`` entry point validates the structural
shape of its input through a private predicate (`is_valid_param_data`,
`_is_valid_ordinal_categorical_perm_param_data`). Replacing any one of the
short-circuited ``and`` operators in those predicates with ``or`` weakens the
guard. Each test below feeds a payload with exactly one structural fault and
asserts the shape-level rejection surfaces as
`DeserializationDictStructureError`.
"""

from typing import Any

import pytest

from fhy_core.identifier import Identifier
from fhy_core.param import (
    CategoricalParam,
    IntParam,
    OrdinalParam,
    Param,
    PermParam,
    RealParam,
)
from fhy_core.serialization import (
    DeserializationDictStructureError,
    serialize_registry_wrapped_value,
)


def _wrap(type_id: str, inner: dict[str, Any]) -> dict[str, Any]:
    """Wrap an inner ``__data__`` payload with the given ``__type__`` envelope."""
    return {"__type__": type_id, "__data__": inner}


@pytest.fixture
def valid_variable_data() -> dict[str, Any]:
    return Identifier("x").serialize_to_dict()


# =============================================================================
# Param data — `is_valid_param_data` rejection paths
# =============================================================================


@pytest.mark.parametrize(
    "param_class, type_id",
    [
        pytest.param(RealParam, "real_param", id="real-param"),
        pytest.param(IntParam, "int_param", id="int-param"),
    ],
)
class TestParamDeserializeRejectsMalformedParamData:
    """Tests rejecting structurally malformed `Param` payloads."""

    def test_rejects_payload_missing_variable_field(
        self, param_class: type[Param[Any]], type_id: str
    ) -> None:
        """Test the deserializer rejects payload missing the ``variable`` field."""
        payload = _wrap(type_id, {"constraints": []})
        with pytest.raises(DeserializationDictStructureError):
            param_class.deserialize_from_dict(payload)

    def test_rejects_payload_with_variable_not_serialized_dict(
        self, param_class: type[Param[Any]], type_id: str
    ) -> None:
        """Test the deserializer rejects payload whose ``variable`` is not a dict."""
        payload = _wrap(type_id, {"variable": "not-a-dict", "constraints": []})
        with pytest.raises(DeserializationDictStructureError):
            param_class.deserialize_from_dict(payload)

    def test_rejects_payload_missing_constraints_field(
        self,
        param_class: type[Param[Any]],
        type_id: str,
        valid_variable_data: dict[str, Any],
    ) -> None:
        """Test the deserializer rejects payload missing the ``constraints`` field."""
        payload = _wrap(type_id, {"variable": valid_variable_data})
        with pytest.raises(DeserializationDictStructureError):
            param_class.deserialize_from_dict(payload)

    def test_rejects_payload_with_constraints_not_a_list(
        self,
        param_class: type[Param[Any]],
        type_id: str,
        valid_variable_data: dict[str, Any],
    ) -> None:
        """Test the deserializer rejects payload whose ``constraints`` is not a list."""
        payload = _wrap(
            type_id,
            {"variable": valid_variable_data, "constraints": "not-a-list"},
        )
        with pytest.raises(DeserializationDictStructureError):
            param_class.deserialize_from_dict(payload)

    def test_rejects_payload_with_constraint_entry_not_serialized_dict(
        self,
        param_class: type[Param[Any]],
        type_id: str,
        valid_variable_data: dict[str, Any],
    ) -> None:
        """Test the deserializer rejects a constraint entry that is not a dict."""
        payload = _wrap(
            type_id,
            {"variable": valid_variable_data, "constraints": [42]},
        )
        with pytest.raises(DeserializationDictStructureError):
            param_class.deserialize_from_dict(payload)


# =============================================================================
# Wrapped-leaf params — `_is_valid_ordinal_categorical_perm_param_data` rejection
# =============================================================================


def _build_valid_wrapped_int(value: int) -> dict[str, Any]:
    return serialize_registry_wrapped_value(value)


@pytest.mark.parametrize(
    "param_class, type_id, possible_value",
    [
        pytest.param(
            OrdinalParam, "ordinal_param", _build_valid_wrapped_int(1), id="ordinal"
        ),
        pytest.param(
            CategoricalParam,
            "categorical_param",
            _build_valid_wrapped_int(1),
            id="categorical",
        ),
        pytest.param(PermParam, "perm_param", _build_valid_wrapped_int(1), id="perm"),
    ],
)
class TestWrappedLeafParamDeserializeRejectsMalformedPayload:
    """Tests rejecting structurally malformed wrapped-leaf param payloads."""

    def test_rejects_payload_missing_possible_values_field(
        self,
        param_class: type[Param[Any]],
        type_id: str,
        possible_value: dict[str, Any],
        valid_variable_data: dict[str, Any],
    ) -> None:
        """Test the deserializer rejects payload missing ``possible_values``."""
        del possible_value  # unused for this case
        payload = _wrap(type_id, {"variable": valid_variable_data, "constraints": []})
        with pytest.raises(DeserializationDictStructureError):
            param_class.deserialize_from_dict(payload)

    def test_rejects_payload_with_possible_values_not_a_list(
        self,
        param_class: type[Param[Any]],
        type_id: str,
        possible_value: dict[str, Any],
        valid_variable_data: dict[str, Any],
    ) -> None:
        """Test the deserializer rejects ``possible_values`` that is not a list."""
        del possible_value  # unused for this case
        payload = _wrap(
            type_id,
            {
                "variable": valid_variable_data,
                "constraints": [],
                "possible_values": "not-a-list",
            },
        )
        with pytest.raises(DeserializationDictStructureError):
            param_class.deserialize_from_dict(payload)

    def test_rejects_payload_failing_inner_param_data_validation(
        self,
        param_class: type[Param[Any]],
        type_id: str,
        possible_value: dict[str, Any],
        valid_variable_data: dict[str, Any],
    ) -> None:
        """Test the deserializer rejects payload that fails the inner param-data check.

        The payload has a valid ``possible_values`` list but is missing the
        ``constraints`` field that the inner param-data predicate requires.
        """
        payload = _wrap(
            type_id,
            {
                "variable": valid_variable_data,
                "possible_values": [possible_value],
            },
        )
        with pytest.raises(DeserializationDictStructureError):
            param_class.deserialize_from_dict(payload)
