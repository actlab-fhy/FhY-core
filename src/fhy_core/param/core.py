"""Core parameter structures."""

__all__ = [
    "Param",
    "ParamAssignment",
    "ParamError",
    "NumericParam",
    "RealParam",
    "IntParam",
    "SerializableEqualValue",
    "SerializableOrderableValue",
    "CategoricalValue",
    "OrdinalValue",
    "PermutationMemberValue",
    "OrdinalParam",
    "CategoricalParam",
    "PermParam",
    "ParamData",
    "is_valid_param_data",
    "finalize_param_construction_from_data",
    "create_single_valid_value_param",
]

import itertools
import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Hashable, Sequence
from typing import (
    Any,
    Generic,
    Protocol,
    TypeAlias,
    TypedDict,
    TypeGuard,
    TypeVar,
    cast,
)

from fhy_core.constraint import (
    Constraint,
    EquationConstraint,
    InSetConstraint,
    NotInSetConstraint,
)
from fhy_core.error import register_error
from fhy_core.expression import (
    Expression,
    IdentifierExpression,
    does_expression_imply,
    replace_identifiers,
)
from fhy_core.identifier import Identifier
from fhy_core.logger import get_logger
from fhy_core.serialization import (
    DeserializationDictStructureError,
    DeserializationValueError,
    Serializable,
    SerializedDict,
    WrappedFamilySerializable,
    deserialize_registry_wrapped_value,
    is_serialized_dict,
    register_serializable,
    serialize_registry_wrapped_value,
)
from fhy_core.symbol_type import SymbolType
from fhy_core.traits import Equal, FrozenMixin, Orderable, StructuralEquivalence
from fhy_core.utils import Self, format_comma_separated_list, is_strict_int

_LOGGER = get_logger(__name__)

_T = TypeVar("_T")


@register_error
class ParamError(ValueError):
    """Domain error for parameter construction, validation, and assignment.

    Subclasses ``ValueError`` so call sites that previously caught
    ``ValueError`` continue to work; new code should prefer ``ParamError``
    when distinguishing parameter-domain failures from generic value errors.
    """


def _constraint_structural_ordering_key(constraint: Constraint) -> str:
    """Return a deterministic key for canonical constraint ordering.

    Uses ``json.dumps`` with ``sort_keys=True`` so the key does not depend on
    the insertion order of keys produced by ``Constraint.serialize_to_dict``.
    """
    return json.dumps(constraint.serialize_to_dict(), sort_keys=True, default=str)


class _ParamAssignmentData(TypedDict):
    param: SerializedDict
    value: SerializedDict


def _is_valid_param_assignment_data(
    data: SerializedDict,
) -> TypeGuard[_ParamAssignmentData]:
    return (
        "param" in data
        and is_serialized_dict(data["param"])
        and "value" in data
        and is_serialized_dict(data["value"])
    )


@register_serializable(type_id="param_assignment")
class ParamAssignment(
    Serializable,
    FrozenMixin,
    Generic[_T],
):
    """Immutable binding of a parameter definition to a concrete value."""

    _param: "Param[_T]"
    _value: _T

    def __init__(self, param: "Param[_T]", value: _T) -> None:
        """Create an assignment after validating value against parameter constraints."""
        if not param.is_value_admissible(value):
            raise ParamError(
                f"Value {value!r} is not admissible for parameter {param!r}."
            )

        is_constraint_satisfied, failing_constraint = (
            param._is_constraints_satisfied_with_failing_constraint(value)
        )
        if not is_constraint_satisfied:
            raise ParamError(
                f"Value {value!r} violates constraint {failing_constraint!r} "
                f"for parameter {param!r}."
            )

        self._param = param
        self._value = value

    @property
    def param(self) -> "Param[_T]":
        return self._param

    @property
    def value(self) -> _T:
        return self._value

    def is_value_set(self) -> bool:
        """Return whether this assignment has a value."""
        return True

    def serialize_to_dict(self) -> SerializedDict:
        return {
            "param": self._param.serialize_to_dict(),
            "value": serialize_registry_wrapped_value(cast(Any, self._value)),
        }

    @classmethod
    def deserialize_from_dict(cls, data: SerializedDict) -> "ParamAssignment[Any]":
        if not _is_valid_param_assignment_data(data):
            raise DeserializationDictStructureError(
                cls, _ParamAssignmentData.__annotations__, data
            )

        param: Param[Any] = Param.deserialize_from_dict(data["param"])
        try:
            value = deserialize_registry_wrapped_value(data["value"])
        except (DeserializationDictStructureError, DeserializationValueError) as exc:
            raise DeserializationValueError(
                cls,
                "value",
                f"a wrapped serializable value (underlying error: {exc})",
                data["value"],
            ) from exc

        try:
            return cls(param, cast(Any, value))
        except ValueError as exc:
            raise DeserializationValueError(
                f"Invalid parameter assignment values: {exc}"
            ) from exc


class Param(
    WrappedFamilySerializable,
    FrozenMixin,
    StructuralEquivalence,
    ABC,
    Generic[_T],
):
    """Abstract base class for constrained parameters."""

    _variable: Identifier
    _constraints: tuple[Constraint, ...]

    def __init__(
        self,
        *,
        name: Identifier | None = None,
        constraints: Sequence[Constraint] = (),
    ) -> None:
        self._variable = name or Identifier("param")
        self._constraints = self._validate_and_deduplicate_constraints(constraints)

    def _validate_and_deduplicate_constraints(
        self, constraints: Sequence[Constraint]
    ) -> tuple[Constraint, ...]:
        """Validate each constraint and drop structural duplicates, preserving order."""
        accumulated: list[Constraint] = []
        for constraint in constraints:
            self.validate_constraint(constraint)
            if any(
                existing.is_structurally_equivalent(constraint)
                for existing in accumulated
            ):
                continue
            accumulated.append(constraint)
        return tuple(accumulated)

    @property
    def variable(self) -> Identifier:
        return self._variable

    @property
    def variable_expression(self) -> IdentifierExpression:
        return IdentifierExpression(self._variable)

    @property
    def constraints(self) -> tuple[Constraint, ...]:
        """Return the constraints attached to this parameter."""
        return self._constraints

    def with_new_constraints(self, constraints: Sequence[Constraint]) -> Self:
        """Return a frozen copy of this parameter with its constraints replaced.

        The parameter's definition state (variable and value domain) is
        preserved unchanged; only the constraint set is replaced. Each
        constraint is validated against this parameter's variable and
        structural duplicates are dropped, matching constructor semantics.

        Args:
            constraints: Constraints for the returned parameter.

        Returns:
            A new frozen parameter of the same concrete type.

        """
        clone = type(self).__new__(type(self))
        # The frozen flag lives in ``FrozenMixin.__slots__``, not ``__dict__``,
        # so copying ``__dict__`` yields a still-mutable clone that ``freeze``
        # seals once its constraints are set.
        clone.__dict__.update(self.__dict__)
        clone._constraints = self._validate_and_deduplicate_constraints(constraints)
        clone.freeze()
        return clone

    @classmethod
    def with_value(
        cls: type[Self], value: _T, *, name: Identifier | None = None
    ) -> ParamAssignment[_T]:
        """Create a parameter assignment from a parameter definition.

        Args:
            value: Parameter value.
            name: Optional parameter name. If not provided, a default name
                will be used.

        Returns:
            Assignment with the specified fixed value.

        """
        param = cls(name=name)
        return param.assign(value)

    def is_value_valid(self, value: Any) -> bool:
        """Return whether a value is valid for this parameter."""
        return self.is_value_admissible(value) and self.is_constraints_satisfied(value)

    @abstractmethod
    def is_value_admissible(self, value: Any) -> bool:
        """Return whether a value is admissible for this parameter kind."""

    def is_constraints_satisfied(self, value: Any) -> bool:
        """Check if the value satisfies all constraints.

        Args:
            value: Value to check.

        Returns:
            True if the value satisfies all constraints; False otherwise.

        """
        return self._is_constraints_satisfied_with_failing_constraint(value)[0]

    def is_structurally_equivalent(self, other: object) -> bool:
        if not isinstance(other, Param):
            return False
        if self.variable != other.variable:
            return False

        self_constraints = self._get_constraints_in_structural_order()
        other_constraints = other._get_constraints_in_structural_order()
        if len(self_constraints) != len(other_constraints):
            return False

        return all(
            left_constraint.is_structurally_equivalent(right_constraint)
            for left_constraint, right_constraint in zip(
                self_constraints, other_constraints, strict=True
            )
        )

    def _get_constraints_in_structural_order(self) -> tuple[Constraint, ...]:
        return tuple(sorted(self._constraints, key=_constraint_structural_ordering_key))

    def _is_constraints_satisfied_with_failing_constraint(
        self, value: Any
    ) -> tuple[bool, Constraint | None]:
        for constraint in self._constraints:
            if not constraint.is_satisfied(value):
                return False, constraint
        return True, None

    @abstractmethod
    def is_value_set_subset(self, other: "Param[_T]") -> bool:
        """Return whether this parameter's value set is a subset of `other`'s.

        The "value set" is the underlying domain the parameter is defined
        over before constraints narrow it (e.g., the reals for ``RealParam``;
        the enumerated possible values for ``OrdinalParam``).
        """

    @abstractmethod
    def is_subset(self, other: "Param[_T]") -> bool:
        """Return whether this parameter's feasibility set is a subset of `other`'s.

        Concrete subclasses define the semantics: :class:`NumericParam`
        reasons about continuous and integer domains via Z3 implication,
        while the discrete-set subclasses (:class:`OrdinalParam`,
        :class:`CategoricalParam`, :class:`PermParam`) iterate over their
        explicit feasibility sets and test membership directly.
        """

    def assign(self, value: _T) -> ParamAssignment[_T]:
        """Assign a value to the parameter, returning a parameter assignment.

        Args:
            value: Value to assign to the parameter.

        Returns:
            A parameter assignment with the provided value.

        Raises:
            ValueError: If the value is not valid.

        """
        return ParamAssignment(self, value)

    def add_constraint(self, constraint: Constraint) -> Self:
        """Return a new parameter with an additional constraint.

        Constraints are deduplicated by structural equivalence: if an
        equivalent constraint is already present, ``self`` is returned
        unchanged.

        Args:
            constraint: Constraint to add.

        Returns:
            A new parameter instance with the added constraint, or ``self``
            when an equivalent constraint is already present.

        """
        self.validate_constraint(constraint)
        if any(
            existing.is_structurally_equivalent(constraint)
            for existing in self._constraints
        ):
            _LOGGER.debug(
                "deduplicated structurally-equivalent constraint on %r",
                self._variable,
            )
            return self
        new_param = self.with_new_constraints(self._constraints + (constraint,))
        _LOGGER.debug(
            "added constraint on %r (total=%d)",
            self._variable,
            len(new_param._constraints),
        )
        return new_param

    def add_constraints(self, constraints: Collection[Constraint]) -> Self:
        """Return a new parameter with multiple additional constraints.

        Each constraint is added through :meth:`add_constraint`, so
        structural duplicates (whether against the existing set or among
        the incoming constraints) are skipped.
        """
        result = self
        for constraint in constraints:
            result = result.add_constraint(constraint)
        return result

    def validate_constraint(self, constraint: Constraint) -> None:
        """Validate whether a constraint can be added to this parameter.

        Args:
            constraint: Constraint to validate.

        Raises:
            ValueError: If the constraint is not valid for this parameter.
            TypeError: If the constraint type is not supported by this parameter
                (may be raised by subclasses).

        """
        if constraint.variable != self.variable:
            raise ParamError("Constraint variable must match parameter variable.")

    def serialize_data_to_dict(self) -> SerializedDict:
        return {
            "variable": self._variable.serialize_to_dict(),
            "constraints": [
                constraint.serialize_to_dict() for constraint in self._constraints
            ],
        }

    def __repr__(self) -> str:
        param_set_repr = self._get_param_set_repr()
        if param_set_repr:
            param_set_repr = f"{param_set_repr}, "
        return (
            f"{self.__class__.__name__}({repr(self._variable)}, {param_set_repr}"
            f"constraints={repr(self._constraints)})"
        )

    def _get_param_set_repr(self) -> str:
        """Return a string representation of the parameter set.

        Note:
            If the representation of the parameter set is implicit to the class,
            this method should return an empty string.

        """
        return ""

    def __str__(self) -> str:
        land = " /\\ "
        return (
            "{" + f"{self._variable} in {self._get_param_set_str()} | "
            f"{land.join(str(c) for c in self._constraints)}"
            "}"
        )

    @abstractmethod
    def _get_param_set_str(self) -> str:
        """Return a string representation of the parameter set."""


def _create_bound_constraint(
    param_variable: Identifier,
    bound: int | float | str,
    *,
    is_lower: bool,
    is_inclusive: bool,
) -> EquationConstraint:
    variable_expression = IdentifierExpression(param_variable)
    if is_lower:
        constraint_equation = (
            variable_expression >= bound
            if is_inclusive
            else variable_expression > bound
        )
    else:
        constraint_equation = (
            variable_expression <= bound
            if is_inclusive
            else variable_expression < bound
        )
    return EquationConstraint(param_variable, constraint_equation)


def _create_lower_bound_constraint(
    param_variable: Identifier,
    lower_bound: int | float | str,
    is_inclusive: bool = True,
) -> EquationConstraint:
    return _create_bound_constraint(
        param_variable, lower_bound, is_lower=True, is_inclusive=is_inclusive
    )


def _create_upper_bound_constraint(
    param_variable: Identifier,
    upper_bound: int | float | str,
    is_inclusive: bool = True,
) -> EquationConstraint:
    return _create_bound_constraint(
        param_variable, upper_bound, is_lower=False, is_inclusive=is_inclusive
    )


class ParamData(TypedDict):
    """Structure of parameter data for serialization."""

    variable: SerializedDict
    constraints: list[SerializedDict]


def is_valid_param_data(data: SerializedDict) -> TypeGuard[ParamData]:
    """Return True if the given data is a valid parameter data; False otherwise."""
    return (
        "variable" in data
        and is_serialized_dict(data["variable"])
        and "constraints" in data
        and isinstance(data["constraints"], list)
        and all(
            is_serialized_dict(constraint_data)
            for constraint_data in data["constraints"]
        )
    )


def finalize_param_construction_from_data(
    param: Param[Any],
    data: ParamData,
    constraint_filter_function: Callable[[Constraint], bool] | None = None,
) -> Param[Any]:
    """Finalize the construction of a parameter from serialized data.

    Args:
        param: Parameter to finalize construction for.
        data: Serialized parameter data.
        constraint_filter_function: Optional function to filter constraints to add
            to the parameter. If not provided, all constraints in the data will be
            added.

    """
    constraints_to_add: list[Constraint] = []
    for constraint_data in data["constraints"]:
        constraint = Constraint.deserialize_from_dict(constraint_data)
        if constraint_filter_function is None or constraint_filter_function(constraint):
            constraints_to_add.append(constraint)
    return param.add_constraints(constraints_to_add)


class NumericParam(Param[_T], ABC, Generic[_T]):
    """Abstract intermediate base for parameters whose domain is numeric.

    A :class:`NumericParam` reasons about its feasibility set via Z3
    implication: every value admitted by ``self``'s constraints must also
    satisfy ``other``'s constraints. The Z3 sort used for that reasoning
    is determined by :meth:`get_symbol_type`.

    Subclasses (currently :class:`RealParam` and :class:`IntParam`, with
    transitive descendants :class:`NatParam`, :class:`BoundIntParam`,
    :class:`BoundNatParam`) commit to a continuous or integer domain that
    Z3 can model. Discrete-set parameter classes that do not have a
    natural Z3 sort (e.g. :class:`CategoricalParam` over arbitrary
    serializable values) inherit from :class:`Param` directly and define
    their own set-semantics :meth:`is_subset`.
    """

    @abstractmethod
    def get_symbol_type(self) -> SymbolType:
        """Return the Z3 sort used to reason about this parameter's domain."""

    def is_subset(self, other: "Param[_T]") -> bool:
        """Return whether this parameter's feasibility set is a subset of `other`'s.

        The check has two parts:

        1. ``self.is_value_set_subset(other)`` -- the underlying value domain
           of this parameter must be a subset of ``other``'s.
        2. Constraint satisfiability -- every value admitted by ``self``'s
           constraints must also satisfy ``other``'s constraints, checked
           by Z3 over the sort returned by :meth:`get_symbol_type`.

        Returns ``False`` immediately if ``other`` is not an instance of
        ``self.__class__``.
        """
        if not isinstance(other, self.__class__):
            return False
        if not self.is_value_set_subset(other):
            return False

        constrained_variable = Identifier("var")

        def convert_param_constraints(
            param_: Param[_T],
        ) -> Expression | None:
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

        if (
            self_constraint_expression is not None
            and other_constraint_expression is not None
        ):
            implies = does_expression_imply(
                self_constraint_expression,
                other_constraint_expression,
                {constrained_variable: self.get_symbol_type()},
            )
            if implies is None:
                _LOGGER.debug(
                    "Z3 returned unknown; treating as subset=True (self=%r, other=%r)",
                    self._variable,
                    other._variable,
                )
            # Z3 `unknown` is treated as "not a counterexample": when the
            # solver cannot decide implication, the subset relation is
            # reported as True rather than as a proof of failure.
            return implies is None or implies
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


@register_serializable(type_id="real_param")
class RealParam(NumericParam[str | float]):
    """Real-valued parameter."""

    def get_symbol_type(self) -> SymbolType:
        return SymbolType.REAL

    def is_value_admissible(self, value: Any) -> bool:
        if isinstance(value, bool):
            return False
        if isinstance(value, float):
            return True
        if isinstance(value, str):
            try:
                float(value)
            except ValueError:
                return False
            return True
        return False

    def is_constraints_satisfied(self, value: str | float) -> bool:
        return super().is_constraints_satisfied(value)

    def assign(self, value: str | float) -> ParamAssignment[str | float]:
        return super().assign(value)

    def _get_param_set_str(self) -> str:
        return "R"

    @classmethod
    def between(
        cls: type[Self],
        lower_bound: float | str,
        upper_bound: float | str,
        *,
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
            raise ParamError("Lower bound must be less than or equal to upper bound.")
        param = cls(name=name)
        param = param.add_lower_bound_constraint(
            lower_bound, is_inclusive=is_lower_inclusive
        )
        param = param.add_upper_bound_constraint(
            upper_bound, is_inclusive=is_upper_inclusive
        )
        return param

    @classmethod
    def with_lower_bound(
        cls: type[Self],
        lower_bound: float | str,
        *,
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
        param = cls(name=name)
        param = param.add_lower_bound_constraint(lower_bound, is_inclusive=is_inclusive)
        return param

    @classmethod
    def with_upper_bound(
        cls: type[Self],
        upper_bound: float | str,
        *,
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
        param = cls(name=name)
        param = param.add_upper_bound_constraint(upper_bound, is_inclusive=is_inclusive)
        return param

    def add_upper_bound_constraint(
        self,
        upper_bound: float | str,
        *,
        is_inclusive: bool = True,
    ) -> Self:
        """Add an upper bound constraint to the parameter.

        Args:
            upper_bound: Upper bound of the parameter.
            is_inclusive: Whether the upper bound is inclusive.

        """
        upper_bound_constraint = _create_upper_bound_constraint(
            self.variable, upper_bound, is_inclusive
        )
        return self.add_constraint(upper_bound_constraint)

    def add_lower_bound_constraint(
        self,
        lower_bound: float | str,
        *,
        is_inclusive: bool = True,
    ) -> Self:
        """Add a lower bound constraint to the parameter.

        Args:
            lower_bound: Lower bound of the parameter.
            is_inclusive: Whether the lower bound is inclusive.

        """
        lower_bound_constraint = _create_lower_bound_constraint(
            self.variable, lower_bound, is_inclusive
        )
        return self.add_constraint(lower_bound_constraint)

    def is_structurally_equivalent(self, other: object) -> bool:
        return isinstance(other, RealParam) and super().is_structurally_equivalent(
            other
        )

    def is_value_set_subset(self, other: "Param[str | float]") -> bool:
        return isinstance(other, RealParam)

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "RealParam":
        if not is_valid_param_data(data):
            raise DeserializationDictStructureError(
                cls, ParamData.__annotations__, data
            )
        param = RealParam(name=Identifier.deserialize_from_dict(data["variable"]))
        param = cast(
            RealParam,
            finalize_param_construction_from_data(
                param,
                data,
            ),
        )
        return param


@register_serializable(type_id="int_param")
class IntParam(NumericParam[int]):
    """Integer-valued parameter."""

    def get_symbol_type(self) -> SymbolType:
        return SymbolType.INT

    def is_value_admissible(self, value: Any) -> bool:
        return is_strict_int(value)

    def is_constraints_satisfied(self, value: int) -> bool:
        return super().is_constraints_satisfied(value)

    def assign(self, value: int) -> ParamAssignment[int]:
        return super().assign(value)

    def _get_param_set_str(self) -> str:
        return "Z"

    @classmethod
    def between(
        cls: type[Self],
        lower_bound: int,
        upper_bound: int,
        *,
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
            raise ParamError("Lower bound must be less than or equal to upper bound.")
        param = cls(name=name)
        param = param.add_lower_bound_constraint(
            lower_bound, is_inclusive=is_lower_inclusive
        )
        param = param.add_upper_bound_constraint(
            upper_bound, is_inclusive=is_upper_inclusive
        )
        return param

    @classmethod
    def with_lower_bound(
        cls: type[Self],
        lower_bound: int,
        *,
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
        param = cls(name=name)
        param = param.add_lower_bound_constraint(lower_bound, is_inclusive=is_inclusive)
        return param

    @classmethod
    def with_upper_bound(
        cls: type[Self],
        upper_bound: int,
        *,
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
        param = cls(name=name)
        param = param.add_upper_bound_constraint(upper_bound, is_inclusive=is_inclusive)
        return param

    def add_upper_bound_constraint(
        self,
        upper_bound: int,
        *,
        is_inclusive: bool = True,
    ) -> Self:
        """Add an upper bound constraint to the parameter.

        Args:
            upper_bound: Upper bound of the parameter.
            is_inclusive: Whether the upper bound is inclusive.

        """
        upper_bound_constraint = _create_upper_bound_constraint(
            self.variable, upper_bound, is_inclusive
        )
        return self.add_constraint(upper_bound_constraint)

    def add_lower_bound_constraint(
        self,
        lower_bound: int,
        *,
        is_inclusive: bool = True,
    ) -> Self:
        """Add a lower bound constraint to the parameter.

        Args:
            lower_bound: Lower bound of the parameter.
            is_inclusive: Whether the lower bound is inclusive.

        """
        lower_bound_constraint = _create_lower_bound_constraint(
            self.variable, lower_bound, is_inclusive
        )
        return self.add_constraint(lower_bound_constraint)

    def is_structurally_equivalent(self, other: object) -> bool:
        return isinstance(other, IntParam) and super().is_structurally_equivalent(other)

    def is_value_set_subset(self, other: "Param[int]") -> bool:
        return isinstance(other, IntParam)

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "IntParam":
        if not is_valid_param_data(data):
            raise DeserializationDictStructureError(
                cls, ParamData.__annotations__, data
            )
        param = IntParam(name=Identifier.deserialize_from_dict(data["variable"]))
        param = cast(
            IntParam,
            finalize_param_construction_from_data(
                param,
                data,
            ),
        )
        return param


class _SerializableValueLike(Protocol):
    """Structural instance-side serialization contract for param values."""

    @classmethod
    def get_serialization_class_type_id(cls) -> str: ...

    def serialize_to_dict(self) -> SerializedDict: ...


class SerializableEqualValue(Equal, _SerializableValueLike, Protocol):
    """Value with equality semantics and serializable instance behavior."""


class SerializableOrderableValue(Orderable, _SerializableValueLike, Protocol):
    """Value with ordering semantics and serializable instance behavior."""


# CategoricalValue omits `float`: floating-point equality is unreliable for
# category membership.
CategoricalValue: TypeAlias = bool | int | str | SerializableEqualValue
OrdinalValue: TypeAlias = bool | int | float | str | SerializableOrderableValue
PermutationMemberValue: TypeAlias = bool | int | float | str | SerializableEqualValue

_CategoricalValueT = TypeVar("_CategoricalValueT", bound=CategoricalValue)
_OrdinalValueT = TypeVar("_OrdinalValueT", bound=OrdinalValue)
_PermutationMemberValueT = TypeVar(
    "_PermutationMemberValueT", bound=PermutationMemberValue
)


def _is_values_unique_in_sequence_without_set(values: Sequence[Any]) -> bool:
    for i, value_1 in enumerate(values):
        for value_2 in values[i + 1 :]:
            if value_1 == value_2:
                return False
    return True


def _is_values_unique_in_sorted_sequence(values: Sequence[Any]) -> bool:
    return all(values[i] != values[i + 1] for i in range(len(values) - 1))


def _is_values_unique_in_sequence_with_set(values: Collection[Hashable]) -> bool:
    return len(values) == len(set(values))


def _supports_equal_value_semantics(value: Any) -> bool:
    if isinstance(value, Equal):
        return value.supports_equality
    value_type = type(value)
    return (
        getattr(value_type, "__eq__", object.__eq__) is not object.__eq__
        and getattr(value_type, "__hash__", None) is not None
    )


def _supports_orderable_value_semantics(value: Any) -> bool:
    if isinstance(value, Orderable):
        return value.supports_ordering
    return any("__lt__" in cls.__dict__ for cls in type(value).__mro__[:-1])


def _is_categorical_value(value: Any) -> TypeGuard[CategoricalValue]:
    return isinstance(value, (bool, int, str)) or (
        isinstance(value, Serializable) and _supports_equal_value_semantics(value)
    )


def _is_ordinal_value(value: Any) -> TypeGuard[OrdinalValue]:
    return isinstance(value, (bool, int, float, str)) or (
        isinstance(value, Serializable) and _supports_orderable_value_semantics(value)
    )


def _is_permutation_member_value(value: Any) -> TypeGuard[PermutationMemberValue]:
    return isinstance(value, (bool, int, float, str)) or (
        isinstance(value, Serializable) and _supports_equal_value_semantics(value)
    )


def _has_bool_numeric_mismatch(value_1: Any, value_2: Any) -> bool:
    return (
        isinstance(value_1, bool)
        and isinstance(value_2, (int, float))
        and not isinstance(value_2, bool)
    ) or (
        isinstance(value_2, bool)
        and isinstance(value_1, (int, float))
        and not isinstance(value_1, bool)
    )


def _param_values_match(candidate: Any, allowed_value: Any) -> bool:
    if _has_bool_numeric_mismatch(candidate, allowed_value):
        return False
    return cast(bool, candidate == allowed_value)


def _contains_param_value(
    allowed_values: Collection[Any] | Sequence[Any], candidate: Any
) -> bool:
    return any(
        _param_values_match(candidate, allowed_value)
        for allowed_value in allowed_values
    )


_ParamValueT = TypeVar("_ParamValueT")


def _deserialize_typed_wrapped_leaf_values(
    owner_cls: type[Any],
    serialized_values: Sequence[Any],
    value_type_guard: Callable[[Any], TypeGuard[_ParamValueT]],
    expected_description: str,
) -> list[_ParamValueT]:
    if not all(is_serialized_dict(value) for value in serialized_values):
        raise DeserializationValueError(
            owner_cls,
            "possible_values",
            expected_description,
            serialized_values,
        )
    wrapped_values = [
        deserialize_registry_wrapped_value(value) for value in serialized_values
    ]
    typed_values: list[_ParamValueT] = []
    for value in wrapped_values:
        if not value_type_guard(value):
            raise DeserializationValueError(
                owner_cls,
                "possible_values",
                expected_description,
                value,
            )
        typed_values.append(value)
    return typed_values


def _serialize_typed_wrapped_leaf_value(value: object) -> SerializedDict:
    """Serialize a validated leaf value through the wrapped registry.

    Param generics are stricter than the wrapped-registry alias, so we validate
    the runtime shape here and then hand the value to the shared serializer.
    """
    if not isinstance(value, (bool, int, float, str, Serializable)):
        raise ParamError(
            "Parameter values must be serializable leaf values "
            "(bool/int/float/str/Serializable)."
        )
    return serialize_registry_wrapped_value(cast(Any, value))


class _OrdinalCategoricalPermParamData(ParamData):
    possible_values: list[Any]


def _is_valid_ordinal_categorical_perm_param_data(
    data: SerializedDict,
) -> TypeGuard[_OrdinalCategoricalPermParamData]:
    return (
        "possible_values" in data
        and isinstance(data["possible_values"], list)
        and is_valid_param_data(data)
    )


@register_serializable(type_id="ordinal_param")
class OrdinalParam(Param[_OrdinalValueT], Generic[_OrdinalValueT]):
    """Ordinal-valued parameter.

    Note:
        All values must be wrapped-leaf serializable and orderable.

    """

    _sorted_values: tuple[_OrdinalValueT, ...]

    def __init__(
        self, values: Sequence[_OrdinalValueT], *, name: Identifier | None = None
    ) -> None:
        super().__init__(name=name)
        all_values = tuple(values)
        if not all_values:
            raise ParamError("Values must be non-empty.")
        for value in all_values:
            if not _is_ordinal_value(value):
                raise TypeError(
                    "Ordinal values must satisfy orderable semantics and be "
                    "serializable, or be primitive bool/int/float/str values."
                )
        try:
            sorted_values = tuple(sorted(all_values))
        except TypeError as exc:
            raise TypeError(
                "Ordinal values must be mutually comparable for sorting."
            ) from exc
        if not _is_values_unique_in_sorted_sequence(sorted_values):
            raise ParamError("Values must be unique.")
        self._sorted_values = sorted_values

    @property
    def possible_values(self) -> tuple[_OrdinalValueT, ...]:
        return self._sorted_values

    def is_value_admissible(self, value: Any) -> bool:
        return _is_ordinal_value(value) and _contains_param_value(
            self._sorted_values, value
        )

    def assign(self, value: _OrdinalValueT) -> ParamAssignment[_OrdinalValueT]:
        return super().assign(value)

    def validate_constraint(self, constraint: Constraint) -> None:
        super().validate_constraint(constraint)
        if not isinstance(constraint, (InSetConstraint, NotInSetConstraint)):
            raise ParamError(
                "Only in-set and not-in-set constraints are allowed for "
                "ordinal parameters."
            )

    def is_structurally_equivalent(self, other: object) -> bool:
        return (
            isinstance(other, OrdinalParam)
            and super().is_structurally_equivalent(other)
            and self._sorted_values == other._sorted_values
        )

    def is_value_set_subset(self, other: "Param[_OrdinalValueT]") -> bool:
        if not isinstance(other, OrdinalParam):
            return False
        return all(
            _contains_param_value(other._sorted_values, value)
            for value in self._sorted_values
        )

    def is_subset(self, other: "Param[_OrdinalValueT]") -> bool:
        """Return whether this parameter's feasibility set is a subset of `other`'s.

        Uses set semantics: every value in ``self._sorted_values`` that
        satisfies ``self``'s constraints must also be admissible by
        ``other`` and satisfy ``other``'s constraints.
        """
        if not isinstance(other, OrdinalParam):
            return False
        for value in self._sorted_values:
            if not self.is_value_valid(value):
                continue
            if not other.is_value_valid(value):
                return False
        return True

    def serialize_data_to_dict(self) -> SerializedDict:
        super_dict = super().serialize_data_to_dict()
        super_dict["possible_values"] = [
            _serialize_typed_wrapped_leaf_value(value) for value in self._sorted_values
        ]
        return super_dict

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> Self:
        if not _is_valid_ordinal_categorical_perm_param_data(data):
            raise DeserializationDictStructureError(
                cls, _OrdinalCategoricalPermParamData.__annotations__, data
            )
        values = _deserialize_typed_wrapped_leaf_values(
            cls,
            data["possible_values"],
            _is_ordinal_value,
            (
                "a list of orderable serializable values or primitive "
                "bool/int/float/str values"
            ),
        )
        param = OrdinalParam(
            values,
            name=Identifier.deserialize_from_dict(data["variable"]),
        )
        final_param = cast(Self, finalize_param_construction_from_data(param, data))
        return final_param

    def _get_param_set_repr(self) -> str:
        return f"{{{format_comma_separated_list(self._sorted_values)}}}"

    def _get_param_set_str(self) -> str:
        return f"{{{format_comma_separated_list(self._sorted_values, str_func=str)}}}"


@register_serializable(type_id="categorical_param")
class CategoricalParam(Param[_CategoricalValueT], Generic[_CategoricalValueT]):
    """Categorical parameter.

    Note:
        All values must be wrapped-leaf serializable, hashable, and support
        equality semantics.

    """

    _categories: frozenset[_CategoricalValueT]

    def __init__(
        self,
        categories: Collection[_CategoricalValueT],
        *,
        name: Identifier | None = None,
    ) -> None:
        super().__init__(name=name)
        category_values = tuple(categories)
        if not category_values:
            raise ParamError("Categories must be non-empty.")
        for category in category_values:
            if not _is_categorical_value(category):
                raise TypeError(
                    "Categorical values must satisfy equal semantics and be "
                    "serializable, or be primitive bool/int/str values."
                )
        if not _is_values_unique_in_sequence_with_set(category_values):
            raise ParamError("Values must be unique.")
        self._categories = frozenset(category_values)

    @property
    def categories(self) -> frozenset[_CategoricalValueT]:
        return self._categories

    def is_value_admissible(self, value: Any) -> bool:
        return _is_categorical_value(value) and _contains_param_value(
            self._categories, value
        )

    def assign(self, value: _CategoricalValueT) -> ParamAssignment[_CategoricalValueT]:
        return super().assign(value)

    def validate_constraint(self, constraint: Constraint) -> None:
        super().validate_constraint(constraint)
        if not isinstance(constraint, (InSetConstraint, NotInSetConstraint)):
            raise ParamError(
                "Only in-set and not-in-set constraints are allowed for "
                "categorical parameters."
            )

    def is_structurally_equivalent(self, other: object) -> bool:
        return (
            isinstance(other, CategoricalParam)
            and super().is_structurally_equivalent(other)
            and self._categories == other._categories
        )

    def is_value_set_subset(self, other: "Param[_CategoricalValueT]") -> bool:
        if not isinstance(other, CategoricalParam):
            return False
        return all(
            _contains_param_value(other._categories, category)
            for category in self._categories
        )

    def is_subset(self, other: "Param[_CategoricalValueT]") -> bool:
        """Return whether this parameter's feasibility set is a subset of `other`'s.

        Uses set semantics: every category in ``self._categories`` that
        satisfies ``self``'s constraints must also be admissible by
        ``other`` and satisfy ``other``'s constraints.
        """
        if not isinstance(other, CategoricalParam):
            return False
        for value in self._categories:
            if not self.is_value_valid(value):
                continue
            if not other.is_value_valid(value):
                return False
        return True

    def serialize_data_to_dict(self) -> SerializedDict:
        super_dict = super().serialize_data_to_dict()
        super_dict["possible_values"] = [
            _serialize_typed_wrapped_leaf_value(category)
            for category in self._categories
        ]
        return super_dict

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> Self:
        if not _is_valid_ordinal_categorical_perm_param_data(data):
            raise DeserializationDictStructureError(
                cls, _OrdinalCategoricalPermParamData.__annotations__, data
            )
        values = _deserialize_typed_wrapped_leaf_values(
            cls,
            data["possible_values"],
            _is_categorical_value,
            ("a list of equal serializable values or primitive bool/int/str values"),
        )
        param = CategoricalParam(
            values,
            name=Identifier.deserialize_from_dict(data["variable"]),
        )
        final_param = cast(Self, finalize_param_construction_from_data(param, data))
        return final_param

    def _get_param_set_repr(self) -> str:
        return f"{{{format_comma_separated_list(self._categories)}}}"

    def _get_param_set_str(self) -> str:
        return f"{{{format_comma_separated_list(self._categories, str_func=str)}}}"


def create_single_valid_value_param(
    value: _CategoricalValueT, *, name: Identifier | None = None
) -> CategoricalParam[_CategoricalValueT]:
    """Return a parameter that can only take a single valid value."""
    return CategoricalParam([value], name=name)


@register_serializable(type_id="perm_param")
class PermParam(
    Param[tuple[_PermutationMemberValueT, ...]], Generic[_PermutationMemberValueT]
):
    """Permutation parameter.

    Note:
        All values must be unique.

    """

    _ordered_members: tuple[_PermutationMemberValueT, ...]

    def __init__(
        self,
        all_values: Sequence[_PermutationMemberValueT],
        *,
        name: Identifier | None = None,
    ) -> None:
        super().__init__(name=name)
        all_member_values = tuple(all_values)
        if not all_member_values:
            raise ParamError("Members must be non-empty.")
        for value in all_member_values:
            if not _is_permutation_member_value(value):
                raise TypeError(
                    "Permutation members must satisfy equal semantics and be "
                    "serializable, or be primitive bool/int/float/str values."
                )
        if not _is_values_unique_in_sequence_without_set(all_values):
            raise ParamError("Values must be unique.")
        self._ordered_members = all_member_values

    @property
    def members(self) -> tuple[_PermutationMemberValueT, ...]:
        return self._ordered_members

    def is_value_admissible(self, value: Any) -> bool:
        return (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes, bytearray))
            and self._is_value_valid_permutation(value)
        )

    def _is_value_valid_permutation(self, value: Sequence[Any]) -> bool:
        return (
            all(
                _is_permutation_member_value(value_element)
                and _contains_param_value(self._ordered_members, value_element)
                for value_element in value
            )
            and len(value) == len(self._ordered_members)
            and _is_values_unique_in_sequence_without_set(value)
        )

    def is_constraints_satisfied(
        self, value: Sequence[_PermutationMemberValueT]
    ) -> bool:
        value = tuple(value)
        return super().is_constraints_satisfied(value)

    def assign(
        self, value: Sequence[_PermutationMemberValueT]
    ) -> ParamAssignment[tuple[_PermutationMemberValueT, ...]]:
        return super().assign(tuple(value))

    def validate_constraint(self, constraint: Constraint) -> None:
        super().validate_constraint(constraint)
        if not isinstance(constraint, (InSetConstraint, NotInSetConstraint)):
            raise ParamError(
                "Only in-set and not-in-set constraints are allowed for "
                "permutation parameters."
            )

    def is_structurally_equivalent(self, other: object) -> bool:
        return (
            isinstance(other, PermParam)
            and super().is_structurally_equivalent(other)
            and self._ordered_members == other._ordered_members
        )

    def is_value_set_subset(
        self, other: "Param[tuple[_PermutationMemberValueT, ...]]"
    ) -> bool:
        if not isinstance(other, PermParam):
            return False
        if len(self._ordered_members) != len(other._ordered_members):
            return False
        return all(
            _contains_param_value(other._ordered_members, member)
            for member in self._ordered_members
        )

    def is_subset(self, other: "Param[tuple[_PermutationMemberValueT, ...]]") -> bool:
        """Return whether this parameter's feasibility set is a subset of `other`'s.

        Uses set semantics: every permutation of ``self._ordered_members``
        that satisfies ``self``'s constraints must also be admissible by
        ``other`` and satisfy ``other``'s constraints.

        Note:
            The cost is ``O(n!)`` in the number of members; restrict the
            universe with ``InSetConstraint`` for large permutation sets.
        """
        if not isinstance(other, PermParam):
            return False
        if len(self._ordered_members) != len(other._ordered_members):
            return False
        for permutation in itertools.permutations(self._ordered_members):
            if not self.is_value_valid(permutation):
                continue
            if not other.is_value_valid(permutation):
                return False
        return True

    def serialize_data_to_dict(self) -> SerializedDict:
        super_dict = super().serialize_data_to_dict()
        super_dict["possible_values"] = [
            _serialize_typed_wrapped_leaf_value(value)
            for value in self._ordered_members
        ]
        return super_dict

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> Self:
        if not _is_valid_ordinal_categorical_perm_param_data(data):
            raise DeserializationDictStructureError(
                cls, _OrdinalCategoricalPermParamData.__annotations__, data
            )
        values = _deserialize_typed_wrapped_leaf_values(
            cls,
            data["possible_values"],
            _is_permutation_member_value,
            (
                "a list of equal serializable values or primitive "
                "bool/int/float/str values"
            ),
        )
        param = PermParam(
            values,
            name=Identifier.deserialize_from_dict(data["variable"]),
        )
        final_param = cast(Self, finalize_param_construction_from_data(param, data))
        return final_param

    def _get_param_set_repr(self) -> str:
        return f"{{{format_comma_separated_list(self._ordered_members)}}}"

    def _get_param_set_str(self) -> str:
        return f"{{{format_comma_separated_list(self._ordered_members, str_func=str)}}}"
