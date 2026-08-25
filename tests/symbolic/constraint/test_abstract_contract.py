"""Tests for the ABC contract on `Constraint`."""

from dataclasses import dataclass
from typing import Any

import pytest

from fhy_core.identifier import Identifier
from fhy_core.symbolic.constraint import (
    Constraint,
    ConstraintBindings,
    ConstraintOutcome,
)
from fhy_core.symbolic.expression import (
    Expression,
    IdentifierExpression,
    LiteralExpression,
)
from fhy_core.utils.override import override

from .conftest import mock_identifier

_CONSTRAINT_OWN_ABSTRACT_METHODS = (
    "get_free_identifiers",
    "evaluate_with_bindings",
    "convert_to_expression",
    "__repr__",
    "__str__",
)
"""Every method the new scope-based `Constraint` declares abstract.

Notably absent: `evaluate`, `is_satisfied`, `__call__` (removed), and
`is_satisfied_with_bindings` (concrete, derived from
`evaluate_with_bindings`).
"""


def _make_constraint_subclass_omitting(method_name: str) -> type[Constraint]:
    """Return a `Constraint` subclass that omits ``method_name``.

    Every other abstract method (plus the serialization hooks a still-
    ``WrappedFamilySerializable``-derived base might require) is stubbed
    so the only outstanding abstractness comes from ``method_name``.
    """
    namespace: dict[str, Any] = {
        "serialize_data_to_dict": lambda self: {},
        "deserialize_data_from_dict": classmethod(lambda cls, data: None),
    }
    if method_name != "get_free_identifiers":
        namespace["get_free_identifiers"] = lambda self: frozenset()
    if method_name != "evaluate_with_bindings":
        namespace["evaluate_with_bindings"] = lambda self, bindings: (
            ConstraintOutcome.SATISFIED
        )
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
    """Test each abstract method on `Constraint` is required for instantiation."""
    cls = _make_constraint_subclass_omitting(missing)

    with pytest.raises(TypeError, match="abstract"):
        cls()  # pylint: disable=abstract-class-instantiated


def test_constraint_subclass_with_full_overrides_instantiates() -> None:
    """Test a dataclass subclass overriding every abstract method instantiates."""

    @dataclass(frozen=True, eq=False)
    class _ConcreteConstraint(Constraint):
        identifier: Identifier

        @override
        def get_free_identifiers(self) -> frozenset[Identifier]:
            return frozenset({self.identifier})

        @override
        def evaluate_with_bindings(
            self, bindings: ConstraintBindings
        ) -> ConstraintOutcome:
            return ConstraintOutcome.SATISFIED

        @override
        def convert_to_expression(self) -> Expression:
            return IdentifierExpression(self.identifier)

        @override
        def __repr__(self) -> str:
            return "ConcreteConstraint"

        @override
        def __str__(self) -> str:
            return "ConcreteConstraint"

    x = mock_identifier("x", 0)
    instance = _ConcreteConstraint(x)

    assert instance.get_free_identifiers() == frozenset({x})
    assert instance.evaluate_with_bindings({}) is ConstraintOutcome.SATISFIED
    assert instance.is_satisfied_with_bindings({}) is True
    assert isinstance(instance.convert_to_expression(), IdentifierExpression)


def test_constraint_class_advertises_abstract_methods() -> None:
    """Test ``Constraint.__abstractmethods__`` includes all five contract methods."""
    abstract = set(Constraint.__abstractmethods__)

    for method_name in _CONSTRAINT_OWN_ABSTRACT_METHODS:
        assert method_name in abstract, (
            f"Constraint must declare {method_name!r} abstract; got {abstract}."
        )


def test_constraint_class_does_not_advertise_removed_methods_as_abstract() -> None:
    """Test the removed unary contract is not part of the abstract method set."""
    abstract = set(Constraint.__abstractmethods__)

    assert "evaluate" not in abstract
    assert "is_satisfied" not in abstract
    assert "__call__" not in abstract


def test_is_satisfied_with_bindings_is_concrete_not_abstract() -> None:
    """Test `is_satisfied_with_bindings` is not part of the abstract method set.

    It is derived from `evaluate_with_bindings`, so a leaf implementing
    only the five abstract methods gets it for free.
    """
    assert "is_satisfied_with_bindings" not in Constraint.__abstractmethods__
