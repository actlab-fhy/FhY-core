"""Core parameter structures."""

__all__ = [
    "Param",
    "RealParam",
    "IntParam",
    "OrdinalParam",
    "CategoricalParam",
    "PermParam",
    "ParamData",
    "is_valid_param_data",
    "finalize_param_construction_from_data",
]

from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Hashable, Sequence
from typing import Any, Generic, TypedDict, TypeGuard, TypeVar

from fhy_core.constraint import (
    Constraint,
    EquationConstraint,
    InSetConstraint,
    NotInSetConstraint,
)
from fhy_core.expression import (
    Expression,
    IdentifierExpression,
    SymbolType,
    is_satisfiable,
    replace_identifiers,
)
from fhy_core.identifier import Identifier
from fhy_core.serialization import (
    InvalidSerializationDataValueError,
    InvalidSerializationDictStructureError,
    SerializedDict,
    SerializedDictBase,
    WrappedFamilySerializable,
)
from fhy_core.utils import Self, format_comma_separated_list

_H = TypeVar("_H", bound=Hashable)
_T = TypeVar("_T")


def _is_values_unique_in_sequence_without_set(values: Sequence[Any]) -> bool:
    for i, value_1 in enumerate(values):
        for value_2 in values[i + 1 :]:
            if value_1 == value_2:
                return False
    return True


def _is_values_unique_in_sorted_sequence(values: Sequence[Any]) -> bool:
    return all(values[i] != values[i + 1] for i in range(len(values) - 1))


def _is_values_unique_in_sequence_with_set(values: Collection[_H]) -> bool:
    return len(values) == len(set(values))


class Param(WrappedFamilySerializable, ABC, Generic[_T]):
    """Abstract base class for constrained parameters."""

    _value: _T | None
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

    @abstractmethod
    def get_symbol_type(self) -> SymbolType:
        """Return the symbol type of the parameter."""

    def is_value_set(self) -> bool:
        """Return True if the parameter value is set; False otherwise."""
        return self._value is not None

    @classmethod
    def with_value(cls: type[Self], value: _T, name: Identifier | None = None) -> Self:
        """Create a parameter with a fixed value.

        Args:
            value: Parameter value.
            name: Optional parameter name. If not provided, a default name
                will be used.

        Returns:
            Parameter with the specified fixed value.

        """
        param = cls(name)
        param.set_value(value)
        return param

    def is_constraints_satisfied(self, value: Any) -> bool:
        """Check if the value satisfies all constraints.

        Args:
            value: Value to check.

        Returns:
            True if the value satisfies all constraints; False otherwise.

        """
        return self._is_constraints_satisfied_with_failing_constraint(value)[0]

    def _is_constraints_satisfied_with_failing_constraint(
        self, value: Any
    ) -> tuple[bool, Constraint | None]:
        for constraint in self._constraints:
            if not constraint.is_satisfied(value):
                return False, constraint
        return True, None

    def is_subset(self, other: "Param[_T]") -> bool:
        """Return if the current parameter is a subset of another parameter."""
        if not isinstance(other, self.__class__):
            return False

        if self.is_value_set() and other.is_value_set():
            return self.get_value() == other.get_value()

        constrained_variable = Identifier("var")

        def convert_param_constraints(
            param_: Param[_T],
        ) -> Expression | None:
            if param_.is_value_set():
                return IdentifierExpression(constrained_variable).equals(
                    param_.get_value()
                )

            constraint_expressions_: list[Expression] = []
            for constraint_ in param_._constraints:
                constraint_expression_ = constraint_.convert_to_expression()
                constraint_expression_ = replace_identifiers(
                    constraint_expression_, {constraint_.variable: constrained_variable}
                )
                constraint_expressions_.append(constraint_expression_)
            if len(constraint_expressions_) == 0:
                return None
            elif len(constraint_expressions_) == 1:
                return constraint_expressions_[0]
            else:
                return Expression.logical_and(*constraint_expressions_)

        self_constraint_expression = convert_param_constraints(self)
        other_constraint_expression = convert_param_constraints(other)

        # TODO: Match code not passing MyPy... using if-else chain instead
        #       but fix this in the future when can get to pass
        # match (self_constraint_expression, other_constraint_expression):
        #     case (None, None):
        #         return True
        #     case (_, None):
        #         return True
        #     case (None, _):
        #         return False
        #     case (e1, e2):
        #         all_constraints_expression = Expression.logical_and(
        #             e1, Expression.logical_not(e2)
        #         )
        #         return not is_satisfiable(
        #             {constrained_variable},
        #             all_constraints_expression,
        #             {constrained_variable: self.get_symbol_type()},
        #         )

        if (
            self_constraint_expression is not None
            and other_constraint_expression is not None
        ):
            e1 = self_constraint_expression
            e2 = other_constraint_expression
            all_constraints_expression = Expression.logical_and(
                e1, Expression.logical_not(e2)
            )
            return not is_satisfiable(
                {constrained_variable},
                all_constraints_expression,
                {constrained_variable: self.get_symbol_type()},
            )
        elif (
            self_constraint_expression is not None
            and other_constraint_expression is None
        ):
            return True
        elif (
            self_constraint_expression is None
            and other_constraint_expression is not None
        ):
            return False
        else:
            return True

    def get_value(self) -> _T:
        """Return the parameter value.

        Raises:
            ValueError: If the parameter is not set.

        """
        if self._value is None:
            raise ValueError("Parameter is not set.")
        return self._value

    def set_value(self, value: _T) -> None:
        """Set the parameter value.

        Args:
            value: New parameter value.

        Raises:
            ValueError: If the value is not valid.

        """
        is_constraint_satisfied, failing_constraint = (
            self._is_constraints_satisfied_with_failing_constraint(value)
        )
        if not is_constraint_satisfied:
            raise ValueError(
                f"Value ({value}) does not satisfy the constraint: {failing_constraint}"
            )
        self._value = value

    def add_constraint(self, constraint: Constraint) -> None:
        """Add a constraint to the parameter.

        Args:
            constraint: Constraint to add.

        """
        self._constraints.append(constraint)

    def serialize_data_to_dict(self) -> SerializedDict:
        return {
            "variable": self._variable.serialize_to_dict(),
            "value": self._value,  # type: ignore[dict-item]
            "constraints": [
                constraint.serialize_to_dict() for constraint in self._constraints
            ],
        }

    def copy(self) -> Self:
        """Return a shallow copy of the parameter."""
        new_param = self.__class__(self._variable)
        new_param._value = self._value
        self.copy_constraints_to_new_param(self, new_param)
        return new_param

    @staticmethod
    def copy_constraints_to_new_param(
        param: "Param[_T]", new_param: "Param[_T]"
    ) -> None:
        """Copy constraints from one parameter to another.

        Args:
            param: Parameter to copy constraints from.
            new_param: Parameter to copy constraints to.

        """
        new_param._constraints = [
            constraint.copy() for constraint in param._constraints
        ]

    def __copy__(self) -> Self:
        return self.copy()

    def __repr__(self) -> str:
        param_set_repr = self._get_param_set_repr()
        if param_set_repr:
            param_set_repr = f"{param_set_repr}, "
        return (
            f"{self.__class__.__name__}({repr(self._variable)}, {param_set_repr}"
            f"value={repr(self._value)}, constraints={repr(self._constraints)})"
        )

    def _get_param_set_repr(self) -> str:
        """Return a string representation of the parameter set.

        Note:
            If the representation of the parameter set is implicit to the class,
            this method should return an empty string.

        """
        return ""

    def __str__(self) -> str:
        if self.is_value_set():
            return f"{{{self.get_value()}}}"
        else:
            land = " /\\ "
            return (
                "{" + f"{self._variable} in {self._get_param_set_str()} | "
                f"{land.join(str(c) for c in self._constraints)}"
                "}"
            )

    @abstractmethod
    def _get_param_set_str(self) -> str:
        """Return a string representation of the parameter set."""


def _create_lower_bound_constraint(
    param_variable: Identifier,
    lower_bound: int | float | str,
    is_inclusive: bool = True,
) -> EquationConstraint:
    param_variable_expression = IdentifierExpression(param_variable)
    if is_inclusive:
        constraint_equation = param_variable_expression >= lower_bound
    else:
        constraint_equation = param_variable_expression > lower_bound
    return EquationConstraint(param_variable, constraint_equation)


def _create_upper_bound_constraint(
    param_variable: Identifier,
    upper_bound: int | float | str,
    is_inclusive: bool = True,
) -> EquationConstraint:
    param_variable_expression = IdentifierExpression(param_variable)
    if is_inclusive:
        constraint_equation = param_variable_expression <= upper_bound
    else:
        constraint_equation = param_variable_expression < upper_bound
    return EquationConstraint(param_variable, constraint_equation)


class ParamData(TypedDict):
    """Structure of parameter data for serialization."""

    variable: SerializedDict
    value: Any | None
    constraints: list[SerializedDict]


def is_valid_param_data(data: SerializedDict) -> TypeGuard[ParamData]:
    """Return True if the given data is a valid parameter data; False otherwise."""
    return (
        "variable" in data
        and isinstance(data["variable"], SerializedDictBase)
        and "value" in data
        and (
            data["value"] is None
            or isinstance(data["value"], (int, float, str, list, tuple, dict))
        )
        and "constraints" in data
        and isinstance(data["constraints"], list)
    )


def finalize_param_construction_from_data(
    param: Param[Any],
    data: ParamData,
    value_check_function: Callable[[Any], bool],
    value_description_phrase: str,
) -> None:
    """Finalize the construction of a parameter from serialized data.

    Args:
        param: Parameter to finalize construction for.
        data: Serialized parameter data.
        value_check_function: Function to check if the value in the data is
            valid for the parameter type.
        value_description_phrase: Phrase describing the valid value type.

    """
    if data["value"] is not None:
        if not value_check_function(data["value"]):
            raise InvalidSerializationDataValueError(
                type(param), "value", value_description_phrase, data["value"]
            )
        param.set_value(data["value"])
    for constraint_data in data["constraints"]:
        constraint = Constraint.deserialize_from_dict(constraint_data)
        param.add_constraint(constraint)


class RealParam(Param[str | float]):
    """Real-valued parameter."""

    def get_symbol_type(self) -> SymbolType:
        return SymbolType.REAL

    def is_constraints_satisfied(self, value: str | float) -> bool:
        return super().is_constraints_satisfied(value)

    def set_value(self, value: str | float) -> None:
        if not isinstance(value, (str, float)):
            raise ValueError("Value must be a string or a float.")
        if isinstance(value, str):
            try:
                float(value)
            except ValueError as e:
                raise ValueError("String value must be a number.") from e
        return super().set_value(value)

    def _get_param_set_str(self) -> str:
        return "R"

    @classmethod
    def between(
        cls: type[Self],
        lower_bound: float | str,
        upper_bound: float | str,
        name: Identifier | None = None,
        is_lower_inclusive: bool = True,
        is_upper_inclusive: bool = True,
    ) -> Self:
        """Create a bounded real-valued parameter.

        Args:
            lower_bound: Lower bound of the parameter.
            upper_bound: Upper bound of the parameter.
            name: Optional parameter name. If not provided, a default name
                will be used.
            is_lower_inclusive: Whether the lower bound is inclusive.
            is_upper_inclusive: Whether the upper bound is inclusive.

        Returns:
            Bounded real-valued parameter.

        """
        if float(lower_bound) > float(upper_bound) or (
            float(lower_bound) == float(upper_bound)
            and not (is_lower_inclusive and is_upper_inclusive)
        ):
            raise ValueError("Lower bound must be less than or equal to upper bound.")
        param = cls(name)
        param.add_lower_bound_constraint(lower_bound, is_lower_inclusive)
        param.add_upper_bound_constraint(upper_bound, is_upper_inclusive)
        return param

    @classmethod
    def with_lower_bound(
        cls: type[Self],
        lower_bound: float | str,
        name: Identifier | None = None,
        is_inclusive: bool = True,
    ) -> Self:
        """Create a real-valued parameter with a lower bound.

        Args:
            lower_bound: Lower bound of the parameter.
            name: Optional parameter name. If not provided, a default name
                will be used.
            is_inclusive: Whether the lower bound is inclusive.

        Returns:
            Real-valued parameter with a lower bound.

        """
        param = cls(name)
        param.add_lower_bound_constraint(lower_bound, is_inclusive)
        return param

    @classmethod
    def with_upper_bound(
        cls: type[Self],
        upper_bound: float | str,
        name: Identifier | None = None,
        is_inclusive: bool = True,
    ) -> Self:
        """Create a real-valued parameter with an upper bound.

        Args:
            upper_bound: Upper bound of the parameter.
            name: Optional parameter name. If not provided, a default name
                will be used.
            is_inclusive: Whether the upper bound is inclusive.

        Returns:
            Real-valued parameter with an upper bound.

        """
        param = cls(name)
        param.add_upper_bound_constraint(upper_bound, is_inclusive)
        return param

    def add_upper_bound_constraint(
        self,
        upper_bound: float | str,
        is_inclusive: bool = True,
    ) -> None:
        """Add an upper bound constraint to the parameter.

        Args:
            upper_bound: Upper bound of the parameter.
            is_inclusive: Whether the upper bound is inclusive.

        """
        upper_bound_constraint = _create_upper_bound_constraint(
            self.variable, upper_bound, is_inclusive
        )
        self.add_constraint(upper_bound_constraint)

    def add_lower_bound_constraint(
        self,
        lower_bound: float | str,
        is_inclusive: bool = True,
    ) -> None:
        """Add a lower bound constraint to the parameter.

        Args:
            lower_bound: Lower bound of the parameter.
            is_inclusive: Whether the lower bound is inclusive.

        """
        lower_bound_constraint = _create_lower_bound_constraint(
            self.variable, lower_bound, is_inclusive
        )
        self.add_constraint(lower_bound_constraint)

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "RealParam":
        if not is_valid_param_data(data):
            raise InvalidSerializationDictStructureError(cls, ParamData, data)
        param = RealParam(Identifier.deserialize_from_dict(data["variable"]))
        finalize_param_construction_from_data(
            param,
            data,
            lambda v: isinstance(v, (str, float)),
            "a float or a string representing a number",
        )
        return param


class IntParam(Param[int]):
    """Integer-valued parameter."""

    def get_symbol_type(self) -> SymbolType:
        return SymbolType.INT

    def is_constraints_satisfied(self, value: int) -> bool:
        return super().is_constraints_satisfied(value)

    def set_value(self, value: int) -> None:
        if not isinstance(value, int):
            raise ValueError("Value must be an integer.")
        return super().set_value(value)

    def _get_param_set_str(self) -> str:
        return "Z"

    @classmethod
    def between(
        cls: type[Self],
        lower_bound: int,
        upper_bound: int,
        name: Identifier | None = None,
        is_lower_inclusive: bool = True,
        is_upper_inclusive: bool = True,
    ) -> Self:
        """Create a bounded integer-valued parameter.

        Args:
            lower_bound: Lower bound of the parameter.
            upper_bound: Upper bound of the parameter.
            name: Optional parameter name. If not provided, a default name
                will be used.
            is_lower_inclusive: Whether the lower bound is inclusive.
            is_upper_inclusive: Whether the upper bound is inclusive.

        Returns:
            Bounded integer-valued parameter.

        """
        if lower_bound > upper_bound or (
            lower_bound == upper_bound
            and not (is_lower_inclusive and is_upper_inclusive)
        ):
            raise ValueError("Lower bound must be less than or equal to upper bound.")
        param = cls(name)
        param.add_lower_bound_constraint(lower_bound, is_lower_inclusive)
        param.add_upper_bound_constraint(upper_bound, is_upper_inclusive)
        return param

    @classmethod
    def with_lower_bound(
        cls: type[Self],
        lower_bound: int,
        name: Identifier | None = None,
        is_inclusive: bool = True,
    ) -> Self:
        """Create an integer-valued parameter with a lower bound.

        Args:
            lower_bound: Lower bound of the parameter.
            name: Optional parameter name. If not provided, a default name
                will be used.
            is_inclusive: Whether the lower bound is inclusive.

        Returns:
            Integer-valued parameter with a lower bound.

        """
        param = cls(name)
        param.add_lower_bound_constraint(lower_bound, is_inclusive)
        return param

    @classmethod
    def with_upper_bound(
        cls: type[Self],
        upper_bound: int,
        name: Identifier | None = None,
        is_inclusive: bool = True,
    ) -> Self:
        """Create an integer-valued parameter with an upper bound.

        Args:
            upper_bound: Upper bound of the parameter.
            name: Optional parameter name. If not provided, a default name
                will be used.
            is_inclusive: Whether the upper bound is inclusive.

        Returns:
            Integer-valued parameter with an upper bound.

        """
        param = cls(name)
        param.add_upper_bound_constraint(upper_bound, is_inclusive)
        return param

    def add_upper_bound_constraint(
        self,
        upper_bound: int,
        is_inclusive: bool = True,
    ) -> None:
        """Add an upper bound constraint to the parameter.

        Args:
            upper_bound: Upper bound of the parameter.
            is_inclusive: Whether the upper bound is inclusive.

        """
        upper_bound_constraint = _create_upper_bound_constraint(
            self.variable, upper_bound, is_inclusive
        )
        self.add_constraint(upper_bound_constraint)

    def add_lower_bound_constraint(
        self,
        lower_bound: int,
        is_inclusive: bool = True,
    ) -> None:
        """Add a lower bound constraint to the parameter.

        Args:
            lower_bound: Lower bound of the parameter.
            is_inclusive: Whether the lower bound is inclusive.

        """
        lower_bound_constraint = _create_lower_bound_constraint(
            self.variable, lower_bound, is_inclusive
        )
        self.add_constraint(lower_bound_constraint)

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "IntParam":
        if not is_valid_param_data(data):
            raise InvalidSerializationDictStructureError(cls, ParamData, data)
        param = IntParam(Identifier.deserialize_from_dict(data["variable"]))
        finalize_param_construction_from_data(
            param,
            data,
            lambda v: isinstance(v, int),
            "an integer",
        )
        return param


class _OrdinalCategorialPermParamData(ParamData):
    possible_values: list[Any]


def _is_valid_ordinal_categorical_perm_param_data(
    data: SerializedDict,
) -> TypeGuard[_OrdinalCategorialPermParamData]:
    return (
        "possible_values" in data
        and isinstance(data["possible_values"], list)
        and is_valid_param_data(data)
    )


class OrdinalParam(Param[Any]):
    """Ordinal-valued parameter.

    Note:
        All values must be of the same type and comparable.

    """

    _all_values: tuple[Any, ...]

    def __init__(self, values: Sequence[Any], name: Identifier | None = None):
        super().__init__(name)
        values = tuple(sorted(values))
        if not _is_values_unique_in_sorted_sequence(values):
            raise ValueError("Values must be unique.")
        self._all_values = values

    def get_symbol_type(self) -> SymbolType:
        return SymbolType.REAL

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

    def serialize_data_to_dict(self) -> SerializedDict:
        super_dict = super().serialize_data_to_dict()
        super_dict["possible_values"] = list(self._all_values)
        return super_dict

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "OrdinalParam":
        if not _is_valid_ordinal_categorical_perm_param_data(data):
            raise InvalidSerializationDictStructureError(
                cls, _OrdinalCategorialPermParamData, data
            )
        param = OrdinalParam(
            data["possible_values"], Identifier.deserialize_from_dict(data["variable"])
        )
        finalize_param_construction_from_data(
            param,
            data,
            lambda v: v in param._all_values,
            f"a value in {param._all_values}",
        )
        return param

    def copy(self) -> "OrdinalParam":
        new_param = OrdinalParam(self._all_values, self._variable)
        self.copy_constraints_to_new_param(self, new_param)
        return new_param

    def _get_param_set_repr(self) -> str:
        return f"{{{format_comma_separated_list(self._all_values)}}}"

    def _get_param_set_str(self) -> str:
        return f"{{{format_comma_separated_list(self._all_values, str_func=str)}}}"


class CategoricalParam(Param[_H]):
    """Categorical parameter.

    Note:
        All values must be hashable.

    """

    _categories: set[_H]

    def __init__(self, categories: Collection[_H], name: Identifier | None = None):
        super().__init__(name)
        if not _is_values_unique_in_sequence_with_set(categories):
            raise ValueError("Values must be unique.")
        self._categories = set(categories)

    def get_symbol_type(self) -> SymbolType:
        return SymbolType.REAL

    def set_value(self, value: _H) -> None:
        if value not in self._categories:
            raise ValueError("Value is not in the set of allowed categories.")
        return super().set_value(value)

    def get_possible_values(self) -> set[_H]:
        """Return the set of possible values for the parameter."""
        return self._categories.copy()

    def add_constraint(self, constraint: Constraint) -> None:
        if not isinstance(constraint, (InSetConstraint, NotInSetConstraint)):
            raise ValueError(
                "Only in-set and not-in-set constraints are allowed for "
                "categorical parameters."
            )
        return super().add_constraint(constraint)

    def serialize_data_to_dict(self) -> SerializedDict:
        super_dict = super().serialize_data_to_dict()
        super_dict["possible_values"] = list(self._categories)  # type: ignore[arg-type]
        return super_dict

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "CategoricalParam[_H]":
        if not _is_valid_ordinal_categorical_perm_param_data(data):
            raise InvalidSerializationDictStructureError(
                cls, _OrdinalCategorialPermParamData, data
            )
        param = CategoricalParam(
            data["possible_values"], Identifier.deserialize_from_dict(data["variable"])
        )
        finalize_param_construction_from_data(
            param,
            data,
            lambda v: v in param._categories,
            f"a value in {param._categories}",
        )
        return param

    def copy(self) -> "CategoricalParam[_H]":
        new_param = CategoricalParam[_H](self._categories.copy(), self._variable)
        self.copy_constraints_to_new_param(self, new_param)
        return new_param

    def _get_param_set_repr(self) -> str:
        return f"{{{format_comma_separated_list(self._categories)}}}"

    def _get_param_set_str(self) -> str:
        return f"{{{format_comma_separated_list(self._categories, str_func=str)}}}"


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

    def get_symbol_type(self) -> SymbolType:
        return SymbolType.REAL

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

    def serialize_data_to_dict(self) -> SerializedDict:
        super_dict = super().serialize_data_to_dict()
        super_dict["possible_values"] = list(self._all_values)
        return super_dict

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "PermParam":
        if not _is_valid_ordinal_categorical_perm_param_data(data):
            raise InvalidSerializationDictStructureError(
                cls, _OrdinalCategorialPermParamData, data
            )
        param = PermParam(
            data["possible_values"], Identifier.deserialize_from_dict(data["variable"])
        )
        finalize_param_construction_from_data(
            param,
            data,
            lambda v: param._is_value_valid_permutation(v),
            f"a permutation of {param._all_values}",
        )
        return param

    def copy(self) -> "PermParam":
        new_param = PermParam(self._all_values, self._variable)
        self.copy_constraints_to_new_param(self, new_param)
        return new_param

    def _get_param_set_repr(self) -> str:
        return f"{{{format_comma_separated_list(self._all_values)}}}"

    def _get_param_set_str(self) -> str:
        return f"{{{format_comma_separated_list(self._all_values, str_func=str)}}}"
