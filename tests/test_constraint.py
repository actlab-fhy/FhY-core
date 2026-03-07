"""Tests the constraint utility."""

import pytest
from fhy_core.constraint import EquationConstraint, InSetConstraint, NotInSetConstraint
from fhy_core.expression import (
    BinaryExpression,
    BinaryOperation,
    IdentifierExpression,
    LiteralExpression,
    LiteralType,
    UnaryExpression,
    UnaryOperation,
)
from fhy_core.identifier import Identifier
from fhy_core.serialization import Serializable, SerializedDict, register_serializable
from fhy_core.trait import Frozen, FrozenMutationError, StructuralEquivalence

from .conftest import mock_identifier


@register_serializable(type_id="test_constraint_member")
class _SerializableConstraintMember(Serializable):
    """Serializable + hashable test helper value."""

    _value: int

    def __init__(self, value: int) -> None:
        self._value = value

    @property
    def value(self) -> int:
        return self._value

    def serialize_to_dict(self) -> SerializedDict:
        return {"value": self._value}

    @classmethod
    def deserialize_from_dict(
        cls, data: SerializedDict
    ) -> "_SerializableConstraintMember":
        return cls(int(data["value"]))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _SerializableConstraintMember)
            and self._value == other._value
        )

    def __hash__(self) -> int:
        return hash(self._value)


@pytest.mark.parametrize(
    "constraint, value, expected_outcome",
    [
        (
            EquationConstraint(mock_identifier("x", 0), LiteralExpression(True)),
            LiteralExpression(0),
            True,
        ),
        (
            EquationConstraint(mock_identifier("x", 0), LiteralExpression(False)),
            LiteralExpression(0),
            False,
        ),
        (
            EquationConstraint(
                mock_identifier("x", 0), IdentifierExpression(mock_identifier("x", 0))
            ),
            LiteralExpression(True),
            True,
        ),
        (
            EquationConstraint(
                mock_identifier("x", 0), IdentifierExpression(mock_identifier("x", 0))
            ),
            LiteralExpression(False),
            False,
        ),
        (
            EquationConstraint(
                mock_identifier("x", 0),
                UnaryExpression(UnaryOperation.LOGICAL_NOT, LiteralExpression(True)),
            ),
            LiteralExpression(True),
            False,
        ),
        (
            EquationConstraint(
                mock_identifier("x", 0),
                UnaryExpression(UnaryOperation.LOGICAL_NOT, LiteralExpression(False)),
            ),
            LiteralExpression(True),
            True,
        ),
        (
            EquationConstraint(
                mock_identifier("x", 0),
                BinaryExpression(
                    BinaryOperation.LOGICAL_AND,
                    LiteralExpression(True),
                    LiteralExpression(True),
                ),
            ),
            LiteralExpression(True),
            True,
        ),
        (
            EquationConstraint(
                mock_identifier("x", 0),
                BinaryExpression(
                    BinaryOperation.LOGICAL_AND,
                    LiteralExpression(True),
                    LiteralExpression(False),
                ),
            ),
            LiteralExpression(True),
            False,
        ),
        (
            EquationConstraint(
                mock_identifier("x", 0),
                BinaryExpression(
                    BinaryOperation.LOGICAL_OR,
                    LiteralExpression(True),
                    LiteralExpression(False),
                ),
            ),
            LiteralExpression(0),
            True,
        ),
        (
            EquationConstraint(
                mock_identifier("x", 0),
                BinaryExpression(
                    BinaryOperation.LOGICAL_OR,
                    LiteralExpression(False),
                    LiteralExpression(False),
                ),
            ),
            LiteralExpression(0),
            False,
        ),
        (
            EquationConstraint(
                mock_identifier("x", 0),
                BinaryExpression(
                    BinaryOperation.EQUAL,
                    LiteralExpression(True),
                    LiteralExpression(True),
                ),
            ),
            LiteralExpression(0),
            True,
        ),
        (
            EquationConstraint(
                mock_identifier("x", 0),
                BinaryExpression(
                    BinaryOperation.EQUAL,
                    LiteralExpression(True),
                    LiteralExpression(False),
                ),
            ),
            LiteralExpression(0),
            False,
        ),
        (
            EquationConstraint(
                mock_identifier("x", 0),
                BinaryExpression(
                    BinaryOperation.NOT_EQUAL,
                    LiteralExpression(True),
                    LiteralExpression(True),
                ),
            ),
            LiteralExpression(0),
            False,
        ),
        (
            EquationConstraint(
                mock_identifier("x", 0),
                BinaryExpression(
                    BinaryOperation.NOT_EQUAL,
                    LiteralExpression(True),
                    LiteralExpression(False),
                ),
            ),
            LiteralExpression(0),
            True,
        ),
        (
            EquationConstraint(
                mock_identifier("x", 0),
                BinaryExpression(
                    BinaryOperation.LESS, LiteralExpression(5), LiteralExpression(10)
                ),
            ),
            LiteralExpression(0),
            True,
        ),
        (
            EquationConstraint(
                mock_identifier("x", 0),
                BinaryExpression(
                    BinaryOperation.LESS, LiteralExpression(10), LiteralExpression(5)
                ),
            ),
            LiteralExpression(0),
            False,
        ),
        (
            EquationConstraint(
                mock_identifier("x", 0),
                BinaryExpression(
                    BinaryOperation.LESS_EQUAL,
                    LiteralExpression(10),
                    LiteralExpression(10),
                ),
            ),
            LiteralExpression(0),
            True,
        ),
        (
            EquationConstraint(
                mock_identifier("x", 0),
                BinaryExpression(
                    BinaryOperation.LESS_EQUAL,
                    LiteralExpression(10),
                    LiteralExpression(5),
                ),
            ),
            LiteralExpression(0),
            False,
        ),
        (
            EquationConstraint(
                mock_identifier("x", 0),
                BinaryExpression(
                    BinaryOperation.GREATER, LiteralExpression(10), LiteralExpression(5)
                ),
            ),
            LiteralExpression(0),
            True,
        ),
        (
            EquationConstraint(
                mock_identifier("x", 0),
                BinaryExpression(
                    BinaryOperation.GREATER, LiteralExpression(5), LiteralExpression(10)
                ),
            ),
            LiteralExpression(0),
            False,
        ),
        (
            EquationConstraint(
                mock_identifier("x", 0),
                BinaryExpression(
                    BinaryOperation.GREATER_EQUAL,
                    LiteralExpression(10),
                    LiteralExpression(10),
                ),
            ),
            LiteralExpression(0),
            True,
        ),
        (
            EquationConstraint(
                mock_identifier("x", 0),
                BinaryExpression(
                    BinaryOperation.GREATER_EQUAL,
                    LiteralExpression(5),
                    LiteralExpression(10),
                ),
            ),
            LiteralExpression(0),
            False,
        ),
    ],
)
def test_equation_constraint_checks_correctly(
    constraint: EquationConstraint,
    value: LiteralExpression,
    expected_outcome: bool,
):
    """Test the equation constraint evaluates correctly when checked."""
    assert constraint.is_satisfied(value) == expected_outcome


@pytest.mark.parametrize(
    "constraint, value, expected_outcome",
    [
        (
            InSetConstraint(mock_identifier("x", 0), {1, 2, 3}),
            1,
            True,
        ),
        (
            InSetConstraint(mock_identifier("x", 0), {1, 2, 3}),
            4,
            False,
        ),
        (
            InSetConstraint(mock_identifier("x", 0), {"a", "b", "c"}),
            "a",
            True,
        ),
        (
            InSetConstraint(mock_identifier("y", 1), {"a", "b", "c"}),
            "d",
            False,
        ),
    ],
)
def test_in_set_constraint_checks_correctly(
    constraint: InSetConstraint,
    value: LiteralExpression,
    expected_outcome: bool,
):
    """Test the in-set constraint evaluates correctly when checked."""
    assert constraint.is_satisfied(value) == expected_outcome


@pytest.mark.parametrize(
    "constraint, values, expected_outcome",
    [
        (
            NotInSetConstraint(mock_identifier("x", 0), {1, 2, 3}),
            1,
            False,
        ),
        (
            NotInSetConstraint(mock_identifier("x", 0), {1, 2, 3}),
            4,
            True,
        ),
        (
            NotInSetConstraint(mock_identifier("x", 0), {"a", "b", "c"}),
            "a",
            False,
        ),
        (
            NotInSetConstraint(mock_identifier("y", 1), {"a", "b", "c"}),
            "d",
            True,
        ),
    ],
)
def test_not_in_set_constraint_checks_correctly(
    constraint: NotInSetConstraint,
    values: dict[Identifier, LiteralType],
    expected_outcome: bool,
):
    """Test the not-in-set constraint evaluates correctly when checked."""
    assert constraint.is_satisfied(values) == expected_outcome


# TODO: If there is ever an equality for constraints, use this instead of
#       accessing private attributes for the three following tests.
def test_copy_equation_constraint():
    """Test the equation constraint is copied correctly."""
    constraint = EquationConstraint(mock_identifier("x", 0), LiteralExpression(True))
    copy = constraint.copy()
    assert constraint.variable == copy.variable
    assert constraint._expression.is_structurally_equivalent(copy._expression)
    assert constraint is not copy


def test_copy_in_set_constraint():
    """Test the in-set constraint is copied correctly."""
    constraint = InSetConstraint(mock_identifier("x", 0), {1, 2, 3})
    copy = constraint.copy()
    assert constraint.variable == copy.variable
    assert constraint._valid_values == copy._valid_values
    assert constraint is not copy


def test_copy_not_in_set_constraint():
    """Test the not-in-set constraint is copied correctly."""
    constraint = NotInSetConstraint(mock_identifier("x", 0), {1, 2, 3})
    copy = constraint.copy()
    assert constraint.variable == copy.variable
    assert constraint._invalid_values == copy._invalid_values
    assert constraint is not copy


def test_convert_equation_constraint_to_expression():
    """Test the equation constraint is converted to an expression correctly."""
    constraint_expression = BinaryExpression(
        BinaryOperation.EQUAL,
        IdentifierExpression(mock_identifier("x", 0)),
        LiteralExpression(True),
    )
    constraint = EquationConstraint(mock_identifier("x", 0), constraint_expression)
    expression = constraint.convert_to_expression()
    assert constraint_expression.is_structurally_equivalent(expression)


def test_convert_in_set_constraint_to_expression():
    """Test the in-set constraint is converted to an expression correctly."""
    constraint = InSetConstraint(mock_identifier("x", 0), {1, 2})
    expression = constraint.convert_to_expression()
    expected_expression = BinaryExpression(
        BinaryOperation.LOGICAL_OR,
        BinaryExpression(
            BinaryOperation.EQUAL,
            IdentifierExpression(mock_identifier("x", 0)),
            LiteralExpression(1),
        ),
        BinaryExpression(
            BinaryOperation.EQUAL,
            IdentifierExpression(mock_identifier("x", 0)),
            LiteralExpression(2),
        ),
    )
    assert expected_expression.is_structurally_equivalent(expression)


def test_convert_not_in_set_constraint_to_expression():
    """Test the not-in-set constraint is converted to an expression correctly."""
    constraint = NotInSetConstraint(mock_identifier("x", 0), {1, 2})
    expression = constraint.convert_to_expression()
    expected_expression = BinaryExpression(
        BinaryOperation.LOGICAL_AND,
        BinaryExpression(
            BinaryOperation.NOT_EQUAL,
            IdentifierExpression(mock_identifier("x", 0)),
            LiteralExpression(1),
        ),
        BinaryExpression(
            BinaryOperation.NOT_EQUAL,
            IdentifierExpression(mock_identifier("x", 0)),
            LiteralExpression(2),
        ),
    )
    assert expected_expression.is_structurally_equivalent(expression)


def test_equation_constraint_dict_serialization():
    """Test the equation constraint can be serialized/deserialized via a dictionary."""
    x = mock_identifier("x", 0)
    x_data = x.serialize_to_dict()
    constraint_expression = BinaryExpression(
        BinaryOperation.EQUAL,
        IdentifierExpression(x),
        LiteralExpression(True),
    )
    constraint = EquationConstraint(x, constraint_expression)
    expected_dict = {
        "__type__": "equation_constraint",
        "__data__": {
            "variable": x_data,
            "expression": constraint_expression.serialize_to_dict(),
        },
    }
    dictionary = constraint.serialize_to_dict()
    assert dictionary == expected_dict
    constraint_deserialized = EquationConstraint.deserialize_from_dict(dictionary)
    assert isinstance(constraint_deserialized, EquationConstraint)
    assert constraint_deserialized.variable == x
    assert constraint_deserialized._expression.is_structurally_equivalent(
        constraint_expression
    )


def test_in_set_constraint_dict_serialization():
    """Test the in-set constraint can be serialized/deserialized via a dictionary."""
    x = mock_identifier("x", 0)
    x_data = x.serialize_to_dict()
    constraint = InSetConstraint(x, {1, 2})
    expected_dict = {
        "__type__": "in_set_constraint",
        "__data__": {
            "variable": x_data,
            "valid_values": [
                {"__type__": "builtins.int", "__data__": 1},
                {"__type__": "builtins.int", "__data__": 2},
            ],
        },
    }
    dictionary = constraint.serialize_to_dict()
    assert dictionary == expected_dict
    constraint_deserialized = InSetConstraint.deserialize_from_dict(dictionary)
    assert isinstance(constraint_deserialized, InSetConstraint)
    assert constraint_deserialized.variable == x
    assert set(constraint_deserialized._valid_values) == {1, 2}


def test_not_in_set_constraint_dict_serialization():
    """Test the NiS constraint can be serialized/deserialized via a dictionary."""
    x = mock_identifier("x", 0)
    x_data = x.serialize_to_dict()
    constraint = NotInSetConstraint(x, {1, 2})
    expected_dict = {
        "__type__": "not_in_set_constraint",
        "__data__": {
            "variable": x_data,
            "invalid_values": [
                {"__type__": "builtins.int", "__data__": 1},
                {"__type__": "builtins.int", "__data__": 2},
            ],
        },
    }
    dictionary = constraint.serialize_to_dict()
    assert dictionary == expected_dict
    constraint_deserialized = NotInSetConstraint.deserialize_from_dict(dictionary)
    assert isinstance(constraint_deserialized, NotInSetConstraint)
    assert constraint_deserialized.variable == x
    assert set(constraint_deserialized._invalid_values) == {1, 2}


def test_in_set_constraint_dict_serialization_with_serializable_member():
    """Test in-set serialization supports serializable+hashable members."""
    x = mock_identifier("x", 0)
    member = _SerializableConstraintMember(7)
    constraint = InSetConstraint(x, {member})

    constraint_deserialized = InSetConstraint.deserialize_from_dict(
        constraint.serialize_to_dict()
    )

    assert constraint_deserialized._valid_values == frozenset({member})


def test_not_in_set_constraint_dict_serialization_with_serializable_member():
    """Test not-in-set serialization supports serializable+hashable members."""
    x = mock_identifier("x", 0)
    member = _SerializableConstraintMember(8)
    constraint = NotInSetConstraint(x, {member})

    constraint_deserialized = NotInSetConstraint.deserialize_from_dict(
        constraint.serialize_to_dict()
    )

    assert constraint_deserialized._invalid_values == frozenset({member})


def test_in_set_constraint_rejects_none_member():
    """Test in-set constraints reject `None` as a member."""
    with pytest.raises(ValueError):
        InSetConstraint(mock_identifier("x", 0), {None})


def test_not_in_set_constraint_rejects_unserializable_member():
    """Test not-in-set constraints reject unsupported members."""
    with pytest.raises(ValueError):
        NotInSetConstraint(mock_identifier("x", 0), [{"a": 1}])


def test_in_set_constraint_supports_tuple_member():
    """Test in-set constraints support tuple members."""
    constraint = InSetConstraint(mock_identifier("x", 0), [(1, "a", True)])
    assert constraint.is_satisfied((1, "a", True))


def test_in_set_constraint_supports_frozenset_member():
    """Test in-set constraints support frozenset members."""
    constraint = InSetConstraint(mock_identifier("x", 0), [frozenset({1, 2, 3})])
    assert constraint.is_satisfied(frozenset({1, 2, 3}))


def test_in_set_constraint_serializes_nested_collection_member():
    """Test in-set constraints serialize nested tuple/frozenset members."""
    x = mock_identifier("x", 0)
    nested_member = (1, (2, 3), frozenset({4, 5}))
    constraint = InSetConstraint(x, [nested_member])

    constraint_deserialized = InSetConstraint.deserialize_from_dict(
        constraint.serialize_to_dict()
    )

    assert constraint_deserialized.is_satisfied(nested_member)


def test_constraint_structural_equivalence_runtime_protocol():
    """Test `Constraint` satisfies `StructuralEquivalence` runtime protocol."""
    constraint = InSetConstraint(mock_identifier("x", 0), {1, 2})
    assert isinstance(constraint, StructuralEquivalence)


def test_constraint_family_is_frozen_on_construction():
    """Test all core constraint classes are frozen after construction."""
    x = mock_identifier("x", 0)
    equation = EquationConstraint(x, LiteralExpression(True))
    in_set = InSetConstraint(x, {1, 2})
    not_in_set = NotInSetConstraint(x, {3, 4})

    for constraint in (equation, in_set, not_in_set):
        assert isinstance(constraint, Frozen)
        assert constraint.is_frozen
        with pytest.raises(FrozenMutationError):
            constraint._freeze_probe = "mutation"


def test_equation_constraint_structural_equivalence_true():
    """Test equation constraints are structurally equivalent when equal."""
    x = mock_identifier("x", 0)
    left = EquationConstraint(x, LiteralExpression(True))
    right = EquationConstraint(x, LiteralExpression(True))
    assert left.is_structurally_equivalent(right)


def test_in_set_constraint_structural_equivalence_false_for_values():
    """Test in-set constraints differ structurally for distinct value sets."""
    x = mock_identifier("x", 0)
    left = InSetConstraint(x, {1, 2})
    right = InSetConstraint(x, {1, 3})
    assert not left.is_structurally_equivalent(right)


def test_not_in_set_constraint_structural_equivalence_false_for_type():
    """Test set constraints differ structurally across constraint kinds."""
    x = mock_identifier("x", 0)
    left = InSetConstraint(x, {1, 2})
    right = NotInSetConstraint(x, {1, 2})
    assert not left.is_structurally_equivalent(right)


# TODO: Check serialization structure errors and value errors for all types.
