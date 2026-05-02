"""Tests ABC contract for `Constraint`.

Each abstract method on `Constraint` must remain abstract, so a subclass
that omits an override fails to instantiate. Removing any single
``@abstractmethod`` decorator would relax the contract for that one
method without making the class as a whole concrete; this module pins
all four declarations.
"""

from typing import Any

import pytest

from fhy_core.constraint import Constraint
from fhy_core.expression import Expression, LiteralExpression

from .conftest import mock_identifier

_CONSTRAINT_OWN_ABSTRACT_METHODS = (
    "is_satisfied",
    "convert_to_expression",
    "__repr__",
    "__str__",
)


def _make_constraint_subclass_omitting(method_name: str) -> type[Constraint]:
    """Build a `Constraint` subclass that overrides every abstract method
    EXCEPT ``method_name``.

    `Constraint` inherits two further abstract methods from the
    `Serializable` family (``serialize_data_to_dict``/
    ``deserialize_data_from_dict``); those are always provided so that
    the only missing method is the one we're probing.
    """
    namespace: dict[str, Any] = {
        "serialize_data_to_dict": lambda self: {},
        "deserialize_data_from_dict": classmethod(lambda cls, data: None),
    }
    if method_name != "is_satisfied":
        namespace["is_satisfied"] = lambda self, value: True
    if method_name != "convert_to_expression":
        namespace["convert_to_expression"] = lambda self: LiteralExpression(True)
    if method_name != "__repr__":
        namespace["__repr__"] = lambda self: "stub"
    if method_name != "__str__":
        namespace["__str__"] = lambda self: "stub"
    return type(f"_StubMissing_{method_name.strip('_')}", (Constraint,), namespace)


@pytest.mark.parametrize("missing", _CONSTRAINT_OWN_ABSTRACT_METHODS)
def test_constraint_subclass_missing_abstract_method_cannot_instantiate(
    missing: str,
) -> None:
    """Test each abstract method on `Constraint` is required for instantiation.

    Removing the ``@abstractmethod`` decorator on any one of them would
    let a partial-override subclass be instantiated successfully.
    """
    cls = _make_constraint_subclass_omitting(missing)
    x = mock_identifier("x", 0)
    with pytest.raises(TypeError):
        cls(x)


def test_constraint_subclass_with_full_overrides_instantiates() -> None:
    """Test a subclass overriding every abstract method instantiates cleanly."""

    class _ConcreteConstraint(Constraint):
        def is_satisfied(self, value: object) -> bool:
            return True

        def convert_to_expression(self) -> Expression:
            return LiteralExpression(True)

        def __repr__(self) -> str:
            return "ConcreteConstraint"

        def __str__(self) -> str:
            return "ConcreteConstraint"

        def serialize_data_to_dict(self) -> dict[str, Any]:
            return {}

        @classmethod
        def deserialize_data_from_dict(
            cls, data: dict[str, Any]
        ) -> "_ConcreteConstraint":
            return cls(mock_identifier("stub", 0))

    x = mock_identifier("x", 0)
    instance = _ConcreteConstraint(x)
    assert instance.variable is x
    assert instance.is_satisfied(0) is True
    assert instance({x: 0}) is True
    assert isinstance(instance.convert_to_expression(), LiteralExpression)


def test_constraint_class_advertises_abstract_methods() -> None:
    """Test ``Constraint.__abstractmethods__`` includes all four contract methods."""
    abstract = set(Constraint.__abstractmethods__)
    for method_name in _CONSTRAINT_OWN_ABSTRACT_METHODS:
        assert method_name in abstract, (
            f"Constraint must declare {method_name!r} abstract; got {abstract}."
        )
