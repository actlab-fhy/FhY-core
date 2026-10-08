"""Identifiers as constraint members and param values.

An ``Identifier`` used as a set-constraint member, a category, a
permutation member or a bound value reaches the Rust core as an identifier:
it compares by id, type-strictly, orders by id, and writes the V2 shape
``{"identifier": {"id": .., "name_hint": ..}}``. The opaque form 0.2.0
wrote, ``{"opaque": {"type_id": "id", ..}}``, still reads.
"""

import json
import pickle
from typing import Any

import pytest

from fhy_core.identifier import Identifier
from fhy_core.serialization import (
    DeserializationValueError,
    SerializedDict,
    deserialize_value,
    serialize_value,
    upgrade_v1_payload,
)
from fhy_core.symbolic.constraint import InSetConstraint, NotInSetConstraint
from fhy_core.symbolic.param import (
    CategoricalDomain,
    Param,
    ParamAssignment,
    create_categorical_param,
    create_ordinal_param,
    create_permutation_param,
)

from ...v1 import reading_v1, writing_v1
from .conftest import SerializableEqualHashable

# Ids whose decimal texts and name hints order opposite to the ids.
_EARLY_ID = 99
_LATE_ID = 100


def _restored(identifier_id: int, name_hint: str) -> Identifier:
    """Return the identifier with ``identifier_id``, as a payload restores it."""
    return Identifier.deserialize_from_dict(
        {"id": identifier_id, "name_hint": name_hint}
    )


def _shape(identifier: Identifier) -> SerializedDict:
    """Return the V2 shape of ``identifier`` as a member or value."""
    return {"identifier": {"id": identifier.id, "name_hint": identifier.name_hint}}


def _legacy_shape(identifier: Identifier) -> SerializedDict:
    """Return the opaque shape 0.2.0 wrote for ``identifier`` as a member."""
    payload = json.dumps(
        {"id": identifier.id, "name_hint": identifier.name_hint},
        separators=(",", ":"),
    )
    return {"opaque": {"type_id": "id", "data": payload}}


def _bind(variable: Identifier, value: Any) -> dict[Identifier, Any]:
    """Return the bindings of ``variable`` to ``value``, of any kind."""
    return {variable: value}


def _assert_is_identifier(value: Any, expected: Identifier) -> None:
    """Assert ``value`` is an `Identifier` equal to ``expected``, same name hint."""
    assert type(value) is Identifier
    assert value == expected
    assert value.name_hint == expected.name_hint


@pytest.fixture
def early() -> Identifier:
    """Return the identifier with the lower id, named ``"z"``."""
    return _restored(_EARLY_ID, "z")


@pytest.fixture
def late() -> Identifier:
    """Return the identifier with the higher id, named ``"a"``."""
    return _restored(_LATE_ID, "a")


# =============================================================================
# The V2 shape
# =============================================================================


def test_serialize_value_writes_an_identifier_as_its_identifier_shape(
    late: Identifier,
) -> None:
    """Test `serialize_value` writes an identifier, and reads it back."""
    data = serialize_value(late)

    assert data == _shape(late)
    _assert_is_identifier(deserialize_value(data), late)


def test_serialize_value_writes_an_identifier_inside_a_tuple(
    late: Identifier,
) -> None:
    """Test an identifier nested in a tuple keeps its shape and reads back."""
    data = serialize_value((late, 1))

    assert data == {"tuple": [_shape(late), {"int": "1"}]}
    decoded = deserialize_value(data)
    assert isinstance(decoded, tuple)
    _assert_is_identifier(decoded[0], late)
    assert decoded[1] == 1


def test_set_constraint_writes_identifier_members_by_id_before_integers(
    early: Identifier, late: Identifier
) -> None:
    """Test identifier members order by id, between frozensets and integers."""
    x = _restored(61_950, "x")

    constraint = InSetConstraint(x, [late, 3, early])

    assert constraint.serialize_to_dict() == {
        "in_set": {
            "variable": {"id": 61_950, "name_hint": "x"},
            "values": [_shape(early), _shape(late), {"int": "3"}],
        }
    }
    _assert_is_identifier(constraint.values[0], early)
    _assert_is_identifier(constraint.values[1], late)
    assert constraint.values[2] == 3


def test_categorical_param_writes_its_identifier_categories_by_id(
    early: Identifier, late: Identifier
) -> None:
    """Test a categorical param orders identifier categories by id."""
    param: Param[Identifier] = create_categorical_param([late, early])

    payload = param.serialize_to_dict()

    assert payload["domain"] == {
        "categorical": {"categories": [_shape(early), _shape(late)]}
    }
    assert isinstance(param.domain, CategoricalDomain)
    assert [category.id for category in param.domain.categories] == [
        _EARLY_ID,
        _LATE_ID,
    ]


def test_permutation_param_keeps_its_identifier_members_in_order(
    early: Identifier, late: Identifier
) -> None:
    """Test a permutation param keeps the order given and admits a permutation."""
    param: Param[tuple[Identifier, ...]] = create_permutation_param([late, early])

    assert param.serialize_to_dict()["domain"] == {
        "permutation": {"ordered_members": [_shape(late), _shape(early)]}
    }
    assert param.is_value_admissible((early, late))
    assert not param.is_value_admissible(("z", late))


# =============================================================================
# Reading what earlier versions wrote
# =============================================================================


def test_legacy_opaque_identifier_value_reads_as_an_identifier(
    late: Identifier,
) -> None:
    """Test the opaque form 0.2.0 wrote for an identifier reads as one."""
    decoded = deserialize_value(_legacy_shape(late))

    _assert_is_identifier(decoded, late)
    assert serialize_value(decoded) == _shape(late)


def test_legacy_opaque_set_constraint_reads_and_writes_the_new_shape(
    early: Identifier, late: Identifier
) -> None:
    """Test a set constraint 0.2.0 wrote reads its identifier members as such."""
    x = _restored(61_951, "x")
    legacy: SerializedDict = {
        "not_in_set": {
            "variable": {"id": 61_951, "name_hint": "x"},
            "values": [_legacy_shape(late), _legacy_shape(early)],
        }
    }

    constraint = NotInSetConstraint.deserialize_from_dict(legacy)

    assert constraint.is_structurally_equivalent(NotInSetConstraint(x, [early, late]))
    written: Any = constraint.serialize_to_dict()
    assert written["not_in_set"]["values"] == [
        _shape(early),
        _shape(late),
    ]


def test_legacy_opaque_categorical_param_reads_and_writes_the_new_shape(
    early: Identifier, late: Identifier
) -> None:
    """Test a categorical param 0.2.0 wrote reads its identifier categories."""
    param: Param[Identifier] = create_categorical_param([early, late])
    legacy = param.serialize_to_dict()
    legacy["domain"] = {
        "categorical": {"categories": [_legacy_shape(late), _legacy_shape(early)]}
    }

    restored: Param[Identifier] = Param.deserialize_from_dict(legacy)

    assert restored.is_structurally_equivalent(param)
    assert restored.serialize_to_dict()["domain"] == {
        "categorical": {"categories": [_shape(early), _shape(late)]}
    }
    assert restored.is_value_admissible(early)


def test_v1_set_constraint_round_trips_its_identifier_members(
    early: Identifier, late: Identifier
) -> None:
    """Test a V1 payload of identifier members reads back as identifiers."""
    constraint = InSetConstraint(_restored(61_952, "x"), [late, early])
    with writing_v1():
        payload = constraint.serialize_to_dict()

    with reading_v1():
        restored = InSetConstraint.deserialize_from_dict(payload)

    assert restored.is_structurally_equivalent(constraint)
    _assert_is_identifier(restored.values[0], early)
    assert restored.serialize_to_dict() == constraint.serialize_to_dict()


def test_upgrade_writes_identifier_members_in_the_new_shape(
    early: Identifier, late: Identifier
) -> None:
    """Test `upgrade_v1_payload` writes identifier members as identifiers."""
    param: Param[Identifier] = create_categorical_param([late, early])
    with writing_v1():
        payload = param.serialize_to_dict()

    upgraded = upgrade_v1_payload(payload, Param)

    assert upgraded == param.serialize_to_dict()
    assert isinstance(upgraded, dict)
    assert upgraded["domain"] == {
        "categorical": {"categories": [_shape(early), _shape(late)]}
    }


@pytest.mark.parametrize(
    "data",
    [
        pytest.param({"identifier": "a"}, id="not-a-payload"),
        pytest.param({"identifier": {"id": 61_953}}, id="no-name-hint"),
        pytest.param({"identifier": {"id": -1, "name_hint": "a"}}, id="negative-id"),
    ],
)
def test_malformed_identifier_value_is_refused(data: SerializedDict) -> None:
    """Test a malformed identifier shape raises `DeserializationValueError`."""
    with pytest.raises(DeserializationValueError):
        deserialize_value(data)


# =============================================================================
# Type-strict comparison
# =============================================================================


def test_identifier_member_matches_only_an_identifier_with_its_id(
    late: Identifier,
) -> None:
    """Test membership compares identifiers by id, never with their names."""
    x = _restored(61_954, "x")
    constraint = InSetConstraint(x, [late])

    assert constraint.is_satisfied_with_bindings(_bind(x, late))
    assert constraint.is_satisfied_with_bindings(_bind(x, _restored(_LATE_ID, "other")))
    assert not constraint.is_satisfied_with_bindings(_bind(x, "a"))
    assert not constraint.is_satisfied_with_bindings(_bind(x, _LATE_ID))
    assert not constraint.is_satisfied_with_bindings(_bind(x, _restored(61_955, "a")))


def test_frozenset_member_matches_a_frozenset_of_identifiers_by_id(
    early: Identifier, late: Identifier
) -> None:
    """Test identifiers nested in a frozenset member compare by id."""
    x = _restored(61_961, "x")
    constraint = InSetConstraint(x, [frozenset({early, late})])

    renamed = frozenset({_restored(_EARLY_ID, "p"), _restored(_LATE_ID, "q")})
    assert constraint.is_satisfied_with_bindings(_bind(x, renamed))
    assert not constraint.is_satisfied_with_bindings(_bind(x, frozenset({"z", "a"})))


def test_string_member_does_not_match_an_identifier_with_its_name(
    late: Identifier,
) -> None:
    """Test a string member is no identifier, even one named alike."""
    x = _restored(61_956, "x")

    assert not InSetConstraint(x, ["a"]).is_satisfied_with_bindings(_bind(x, late))


def test_serializable_member_does_not_match_an_identifier(
    late: Identifier,
) -> None:
    """Test an identifier binding is decided against a `Serializable` member."""
    x = _restored(61_957, "x")
    constraint = InSetConstraint(x, [SerializableEqualHashable(_LATE_ID)])

    assert not constraint.is_satisfied_with_bindings(_bind(x, late))


def test_categorical_param_admits_identifiers_by_id_only(
    early: Identifier, late: Identifier
) -> None:
    """Test a categorical param admits an identifier, and not its name."""
    param: Param[Identifier] = create_categorical_param([early, late])

    assert param.is_value_admissible(late)
    assert param.is_value_admissible(_restored(_LATE_ID, "renamed"))
    assert not param.is_value_admissible("a")
    assert not param.is_value_admissible(_restored(61_958, "a"))


def test_ordinal_param_still_refuses_identifiers(
    early: Identifier, late: Identifier
) -> None:
    """Test identifiers do not order, so an ordinal param refuses them."""
    with pytest.raises(TypeError, match="Ordinal values must satisfy orderable"):
        create_ordinal_param([early, late])  # type: ignore[type-var]  # test: no order


def test_user_identifier_subclass_is_read_as_an_identifier() -> None:
    """Test an `Identifier` subclass member is an identifier, and reads back.

    The member is stored by id and name hint, so it reads back as an
    `Identifier` equal to it, not as an instance of the subclass.
    """

    class Labelled(Identifier):  # type: ignore[misc]  # test: `Identifier` is final
        """A user subclass of `Identifier`."""

    x = _restored(61_959, "x")
    labelled = Labelled("s")
    constraint = InSetConstraint(x, [labelled])

    payload: Any = constraint.serialize_to_dict()
    restored = InSetConstraint.deserialize_from_dict(payload)

    assert payload["in_set"]["values"] == [_shape(labelled)]
    _assert_is_identifier(restored.values[0], labelled)
    assert constraint.is_satisfied_with_bindings(_bind(x, _restored(labelled.id, "s")))


# =============================================================================
# Pickling
# =============================================================================


def test_pickling_keeps_identifier_members(early: Identifier, late: Identifier) -> None:
    """Test a set constraint and a param over identifiers survive pickling."""
    constraint = InSetConstraint(_restored(61_960, "x"), [late, early])
    param: Param[Identifier] = create_categorical_param([late, early])

    constraint_copy = pickle.loads(pickle.dumps(constraint))
    param_copy = pickle.loads(pickle.dumps(param))

    assert constraint_copy.is_structurally_equivalent(constraint)
    _assert_is_identifier(constraint_copy.values[1], late)
    assert param_copy.is_structurally_equivalent(param)
    assert param_copy.serialize_to_dict() == param.serialize_to_dict()


def test_assignment_of_an_identifier_round_trips(
    early: Identifier, late: Identifier
) -> None:
    """Test a param assignment's identifier value writes its shape and reads back."""
    param: Param[Identifier] = create_categorical_param([late, early])
    assignment = ParamAssignment(param, late)

    payload = assignment.serialize_to_dict()
    restored: ParamAssignment[Identifier] = ParamAssignment.deserialize_from_dict(
        payload
    )

    assert payload["value"] == _shape(late)
    _assert_is_identifier(restored.value, late)
    assert restored.is_structurally_equivalent(assignment)
