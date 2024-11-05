"""Core parameter structures."""

from abc import ABC
from collections.abc import Sequence
from typing import Any, Generic, TypeVar

from fhy_core.constraint import (
    Constraint,
    InSetConstraint,
    NotInSetConstraint,
)
from fhy_core.expression import IdentifierExpression, LiteralExpression
from fhy_core.identifier import Identifier


def _is_values_unique_in_sequence_without_set(values: Sequence) -> bool:
    for i, value_1 in enumerate(values):
        for value_2 in values[i + 1 :]:
            if value_1 == value_2:
                return False
    return True


def _is_values_unique_in_sorted_sequence(values: Sequence) -> bool:
    return all(values[i] != values[i + 1] for i in range(len(values) - 1))


def _is_values_unique_in_sequence_with_set(values: Sequence) -> bool:
    return len(values) == len(set(values))


T = TypeVar("T")


class Param(ABC, Generic[T]):
    """Abstract base class for constrained parameters."""

    _value: T | None
    _variable: Identifier
    _constraints: list[Constraint]

    def __init__(self, name: Identifier | None = None) -> None:
        self._value = None
        self._variable = name or Identifier("param")
        self._constraints = []

    @property
    def variable(self) -> Identifier:
        return self._variable

    @property
    def variable_expression(self) -> IdentifierExpression:
        return IdentifierExpression(self._variable)

    def is_set(self) -> bool:
        """Return True if the parameter is set; False otherwise."""
        return self._value is not None

    def is_constraints_satisfied(self, value: Any) -> bool:
        """Check if the value satisfies all constraints.

        Args:
            value: Value to check.

        Returns:
            True if the value satisfies all constraints; False otherwise.

        """
        for constraint in self._constraints:
            if not constraint.is_satisfied({self._variable: value}):
                return False
        return True

    def get_value(self) -> T:
        """Return the parameter value.

        Raises:
            ValueError: If the parameter is not set.

        """
        if self._value is None:
            raise ValueError("Parameter is not set.")
        return self._value

    def set_value(self, value: T) -> None:
        """Set the parameter value.

        Args:
            value: New parameter value.

        Raises:
            ValueError: If the value is not valid.

        """
        if not self.is_constraints_satisfied(value):
            raise ValueError("Parameter value does not satisfy constraints.")
        self._value = value

    def add_constraint(self, constraint: Constraint) -> None:
        """Add a constraint to the parameter.

        Args:
            constraint: Constraint to add.

        """
        self._constraints.append(constraint)


class RealParam(Param[str | float]):
    """Real-valued parameter."""

    def is_constraints_satisfied(self, value: str | float) -> bool:
        return super().is_constraints_satisfied(LiteralExpression(value))

    def set_value(self, value: str | float) -> None:
        if not isinstance(value, (str, float)):
            raise ValueError("Value must be a string or a float.")
        return super().set_value(value)


class IntParam(Param[int]):
    """Integer-valued parameter."""

    def is_constraints_satisfied(self, value: int) -> bool:
        return super().is_constraints_satisfied(LiteralExpression(value))

    def set_value(self, value: int) -> None:
        if not isinstance(value, int):
            raise ValueError("Value must be an integer.")
        return super().set_value(value)


class OrdinalParam(Param[Any]):
    """Ordinal-valued parameter.

    Note:
        All values must be of the same type and comparable.

    """

    _all_values: tuple[Any]

    def __init__(self, values: Sequence[Any], name: Identifier | None = None):
        super().__init__(name)
        values = tuple(sorted(values))
        if not _is_values_unique_in_sorted_sequence(values):
            raise ValueError("Values must be unique.")
        self._all_values = values

    def set_value(self, value: Any) -> None:
        if value not in self._all_values:
            raise ValueError("Value is not in the set of allowed values.")
        return super().set_value(value)

    def add_constraint(self, constraint: Constraint) -> None:
        if not isinstance(constraint, (InSetConstraint, NotInSetConstraint)):
            raise ValueError(
                "Only in-set and not-in-set constraints are allowed for "
                "ordinal parameters."
            )
        return super().add_constraint(constraint)


class CategoricalParam(Param[Any]):
    """Categorical parameter.

    Note:
        All values must be hashable.

    """

    _categories: set[Any]

    def __init__(self, categories: Sequence[Any], name: Identifier | None = None):
        super().__init__(name)
        if not _is_values_unique_in_sequence_with_set(categories):
            raise ValueError("Values must be unique.")
        self._categories = set(categories)

    def set_value(self, value: Any) -> None:
        if value not in self._categories:
            raise ValueError("Value is not in the set of allowed categories.")
        return super().set_value(value)

    def add_constraint(self, constraint: Constraint) -> None:
        if not isinstance(constraint, (InSetConstraint, NotInSetConstraint)):
            raise ValueError(
                "Only in-set and not-in-set constraints are allowed for "
                "categorical parameters."
            )
        return super().add_constraint(constraint)


class PermParam(Param[tuple[Any, ...]]):
    """Permutation parameter.

    Note:
        All values must be unique.

    """

    _all_values: tuple[Any, ...]

    def __init__(self, all_values: Sequence[Any], name: Identifier | None = None):
        super().__init__(name)
        if not _is_values_unique_in_sequence_without_set(all_values):
            raise ValueError("Values must be unique.")
        self._all_values = tuple(all_values)

    def _is_value_valid_permutation(self, value: Sequence[Any]) -> bool:
        return (
            all(value_element in self._all_values for value_element in value)
            and len(value) == len(self._all_values)
            and _is_values_unique_in_sequence_without_set(value)
        )

    def is_constraints_satisfied(self, value: Sequence[Any]) -> bool:
        value = tuple(value)
        return super().is_constraints_satisfied(value)

    def set_value(self, value: Sequence[Any]) -> None:
        if not self._is_value_valid_permutation(value):
            raise ValueError("Value is not a valid permutation.")
        return super().set_value(tuple(value))

    def add_constraint(self, constraint: Constraint) -> None:
        if not isinstance(constraint, (InSetConstraint, NotInSetConstraint)):
            raise ValueError(
                "Only in-set and not-in-set constraints are allowed for "
                "permutation parameters."
            )
        return super().add_constraint(constraint)