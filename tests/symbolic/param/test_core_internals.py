"""Tests for the leaf values the finite domains serialize and refuse.

The Python helper `serialize_wrapped_leaf_value` these tests once called
directly was deleted when the domains moved to the Rust core (S16). A
finite domain's payload now serializes each of its values through the
serialization framework's wrapped registry, and the domain refuses an
unsupported value when it is built, so the rules are pinned through the
public domains here.
"""

from typing import Any

import pytest

from fhy_core.symbolic.param import CategoricalDomain, OrdinalDomain

from .conftest import mock_identifier

# =============================================================================
# Leaf values in a payload
# =============================================================================


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(True, id="bool"),
        pytest.param(1, id="int"),
        pytest.param(1.5, id="float"),
        pytest.param("text", id="str"),
        pytest.param(mock_identifier("x", 0), id="serializable"),
    ],
)
def test_domain_payload_serializes_each_supported_leaf_type(value: Any) -> None:
    """Test a domain serializes each supported leaf-value type without raising.

    A categorical domain holds no float, so the float goes in an ordinal one.
    """
    domain = (
        OrdinalDomain((value,))
        if isinstance(value, float)
        else CategoricalDomain((value,))
    )

    payload = domain.serialize_to_dict()["__data__"]

    assert isinstance(payload, dict)
    (values,) = payload.values()
    assert isinstance(values, list)
    assert len(values) == 1


@pytest.mark.parametrize(
    "value",
    [
        pytest.param([1, 2, 3], id="list"),
        pytest.param({1: 2}, id="dict"),
        pytest.param(object(), id="opaque-object"),
    ],
)
def test_domain_refuses_an_unsupported_leaf_value(value: Any) -> None:
    """Test a domain refuses a value of an unsupported type when it is built."""
    with pytest.raises(TypeError, match="Categorical values"):
        CategoricalDomain((value,))


# =============================================================================
# Constraint ordering
#
# `Param` constraint ordering goes through `Constraint.build_ordering_key`,
# covered directly under `tests/symbolic/constraint/**`.
# `test_scope_attachment.py::test_param_constraint_tuple_matches_build_ordering_key_order`  # noqa: E501
# pins the same ordering property at the `Param` level.
# =============================================================================
