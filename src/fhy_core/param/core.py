"""Constrained parameters built by composing a value domain.

A :class:`Param` pairs a variable identifier and a set of constraints with a
:class:`~fhy_core.param.domains.ParamDomain` that supplies all kind-specific
behavior. There is a single concrete ``Param`` class; the common kinds are built
through the ``create_*`` factory functions.

A parameter serializes through the schema-derived ``Serializable`` engine; the
``domain`` field carries a wrapped family envelope identifying the concrete
domain.
"""

import json
import operator
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, cast

from fhy_core.constraint import Constraint, ConstraintOutcome, EquationConstraint
from fhy_core.expression import (
    BinaryExpression,
    BinaryOperation,
    IdentifierExpression,
    LiteralExpression,
)
from fhy_core.identifier import Identifier
from fhy_core.logger import get_logger
from fhy_core.serialization import (
    FieldCodec,
    Serializable,
    _SerializableFieldCodec,
    deserialize_registry_wrapped_value,
    make_field_codec,
    register_serializable,
    serialize_registry_wrapped_value,
)
from fhy_core.symbol_type import SymbolType
from fhy_core.traits import (
    DerivedEquivalenceMixin,
    FrozenMixin,
    compared_as_binder,
    compared_as_value,
)
from fhy_core.utils.override import override

from .domains import (
    IntegerDomain,
    IntervalIntegerDomain,
    ParamDomain,
    RealDomain,
    build_categorical_domain,
    build_ordinal_domain,
    build_permutation_domain,
    is_bound_expression,
)
from .values import (
    CategoricalValue,
    OrdinalValue,
    ParamError,
    PermutationMemberValue,
    SerializableEqualValue,
    SerializableOrderableValue,
    _CategoricalValueT,
    _OrdinalValueT,
    _PermutationMemberValueT,
)

__all__ = [
    "CategoricalValue",
    "OrdinalValue",
    "Param",
    "ParamAssignment",
    "ParamError",
    "PermutationMemberValue",
    "SerializableEqualValue",
    "SerializableOrderableValue",
    "create_categorical_param",
    "create_integer_param",
    "create_integer_param_between",
    "create_integer_param_with_lower_bound",
    "create_integer_param_with_upper_bound",
    "create_interval_integer_param",
    "create_interval_integer_param_between",
    "create_interval_integer_param_exactly",
    "create_interval_integer_param_with_lower_bound",
    "create_interval_integer_param_with_upper_bound",
    "create_interval_natural_param",
    "create_natural_param",
    "create_ordinal_param",
    "create_permutation_param",
    "create_real_param",
    "create_real_param_between",
    "create_real_param_with_lower_bound",
    "create_real_param_with_upper_bound",
    "create_single_valid_value_param",
]

_LOGGER = get_logger(__name__)

_T = TypeVar("_T")


def _constraint_structural_ordering_key(constraint: Constraint) -> str:
    """Return a deterministic key for canonical constraint ordering."""
    return json.dumps(constraint.serialize_to_dict(), sort_keys=True, default=str)


_WRAPPED_VALUE_CODEC: FieldCodec = make_field_codec(
    serialize_registry_wrapped_value, deserialize_registry_wrapped_value
)


# ---------------------------------------------------------------------------
# Param container
# ---------------------------------------------------------------------------


@register_serializable(type_id="param")
@dataclass(frozen=True, eq=False)
class Param(Serializable, FrozenMixin, DerivedEquivalenceMixin, Generic[_T]):
    """A constrained parameter that composes a value domain.

    A parameter is defined by its variable, a set of constraints, and a
    :class:`~fhy_core.param.domains.ParamDomain` that supplies admissibility,
    subset, equivalence, and serialization behavior. Construct one directly with
    a domain, or use a ``create_*`` factory for the common kinds.

    The ``Param[_T]`` type parameter is an advisory hint for call-site inference
    only; the admissible value type is enforced by the domain at runtime, not by
    ``_T``.

    Two parameters are structurally equivalent when they have the same variable,
    domain, and constraints; under alpha comparison they are equivalent up to
    renaming of the bound variable. Construction validates and de-duplicates the
    constraints, appends the domain's implied constraints, and sorts them into a
    canonical order so that constraint-set order does not affect equivalence.
    """

    domain: ParamDomain
    variable: Identifier = field(
        default_factory=lambda: Identifier("param"),
        metadata=compared_as_binder(scopes_over=("constraints",)),
    )
    constraints: tuple[Constraint, ...] = ()

    def __post_init__(self) -> None:
        canonical = self._build_canonical_constraints(self.constraints)
        object.__setattr__(self, "constraints", canonical)

    def _build_canonical_constraints(
        self, constraints: Sequence[Constraint]
    ) -> tuple[Constraint, ...]:
        accumulated = self._validate_and_deduplicate_constraints(constraints)
        for implied in self.domain.get_implied_constraints(self.variable):
            if not any(
                existing.is_structurally_equivalent(implied) for existing in accumulated
            ):
                accumulated = (*accumulated, implied)
        return tuple(sorted(accumulated, key=_constraint_structural_ordering_key))

    def _validate_and_deduplicate_constraints(
        self, constraints: Sequence[Constraint]
    ) -> tuple[Constraint, ...]:
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
    def variable_expression(self) -> IdentifierExpression:
        """Return the parameter's variable as an identifier expression."""
        return IdentifierExpression(self.variable)

    @property
    def symbol_type(self) -> SymbolType | None:
        """Return the domain's numeric symbol type, or ``None`` if non-numeric."""
        return self.domain.symbol_type

    def replace_constraints(self, constraints: Sequence[Constraint]) -> "Param[_T]":
        """Return a copy of this parameter with its constraints replaced.

        The domain and variable are preserved; only the constraint set is
        replaced, validated, de-duplicated by structural equivalence, and
        re-canonicalized.

        Args:
            constraints: Constraints for the returned parameter.

        Returns:
            A new parameter with the same domain and variable.

        """
        return Param(
            self.domain, variable=self.variable, constraints=tuple(constraints)
        )

    def is_value_valid(self, value: Any) -> bool:
        """Return whether a value is admissible and satisfies all constraints."""
        return self.is_value_admissible(value) and self.is_constraints_satisfied(value)

    def is_value_admissible(self, value: Any) -> bool:
        """Return whether a value lies in this parameter's value domain."""
        return self.domain.is_value_admissible(value)

    def is_constraints_satisfied(self, value: Any) -> bool:
        """Return whether the value satisfies all constraints."""
        normalized = self.domain.normalize_value(value)
        return self._find_failing_constraint(normalized)[0]

    def _find_failing_constraint(
        self, value: Any
    ) -> tuple[bool, Constraint | None, ConstraintOutcome | None]:
        """Return the first non-satisfied constraint and its outcome.

        Constraints are checked in canonical order; the first constraint
        whose ``evaluate`` does not return ``SATISFIED`` short-circuits.

        Returns:
            A ``(is_satisfied, constraint, outcome)`` triple. When every
            constraint is satisfied, this is ``(True, None, None)``.
            Otherwise ``is_satisfied`` is ``False``, ``constraint`` is the
            first non-satisfied constraint, and ``outcome`` is its
            ``ConstraintOutcome`` (``VIOLATED`` or ``UNDECIDED``).

        """
        for constraint in self.constraints:
            outcome = constraint.evaluate(value)
            if outcome is not ConstraintOutcome.SATISFIED:
                return False, constraint, outcome
        return True, None, None

    def validate_value(self, value: Any) -> None:
        """Raise if ``value`` is not a valid assignment for this parameter.

        Raises:
            ParamError: If the value is not admissible, violates a constraint,
                or cannot be verified against a constraint.

        """
        if not self.is_value_admissible(value):
            raise ParamError(
                f"Value {value!r} is not admissible for parameter {self!r}."
            )
        _, failing_constraint, outcome = self._find_failing_constraint(value)
        if failing_constraint is None:
            return
        if outcome is ConstraintOutcome.UNDECIDED:
            raise ParamError(
                f"Value {value!r} could not be verified against constraint "
                f"{failing_constraint!r} for parameter {self!r}."
            )
        raise ParamError(
            f"Value {value!r} violates constraint {failing_constraint!r} "
            f"for parameter {self!r}."
        )

    def is_value_set_subset(self, other: "Param[_T]") -> bool:
        """Return whether this parameter's value set is a subset of ``other``'s."""
        return self.domain.is_value_set_subset(other.domain)

    def is_subset(self, other: "Param[_T]") -> bool:
        """Return whether this parameter's feasible set is a subset of ``other``'s.

        Comparison is gated on value space: numeric parameters compare only with
        numeric parameters sharing the same numeric symbol type (integers with
        integers, reals with reals), and finite-set parameters compare only
        within their own family. Cross-space and cross-family queries return
        ``False``.

        When the solver cannot decide a numeric implication, the relation is
        assumed to hold, so a ``True`` result means "not disproven", not
        "proven".
        """
        return self.domain.compute_feasibility_subset(
            self.constraints, other.domain, other.constraints
        )

    def is_feasible(self) -> bool:
        """Return whether some value satisfies the domain and all constraints.

        The constraints already include the domain's implied constraints, so the
        domain reasons only about the constraints it is given.
        """
        return self.domain.has_feasible_value(self.constraints)

    def is_empty(self) -> bool:
        """Return whether no value satisfies the domain and all constraints."""
        return not self.is_feasible()

    def assign(self, value: _T) -> "ParamAssignment[_T]":
        """Assign a value to the parameter, returning a parameter assignment.

        Args:
            value: Value to assign; normalized by the domain before binding.

        Returns:
            A parameter assignment with the normalized value.

        Raises:
            ParamError: If the value is not admissible or violates a constraint.

        """
        return ParamAssignment(self, cast(_T, self.domain.normalize_value(value)))

    def add_constraint(self, constraint: Constraint) -> "Param[_T]":
        """Return a new parameter with an additional constraint.

        Structurally-equivalent duplicates are dropped, returning ``self``
        unchanged when the constraint is already present.
        """
        self.validate_constraint(constraint)
        if any(
            existing.is_structurally_equivalent(constraint)
            for existing in self.constraints
        ):
            return self
        return self.replace_constraints((*self.constraints, constraint))

    def add_constraints(self, constraints: Collection[Constraint]) -> "Param[_T]":
        """Return a new parameter with multiple additional constraints."""
        result = self
        for constraint in constraints:
            result = result.add_constraint(constraint)
        return result

    def validate_constraint(self, constraint: Constraint) -> None:
        """Validate whether a constraint can be added to this parameter.

        Raises:
            ParamError: If the constraint variable does not match, or the domain
                rejects the constraint.
            TypeError: If the domain forbids the constraint's type.

        """
        if constraint.variable != self.variable:
            raise ParamError("Constraint variable must match parameter variable.")
        self.domain.validate_constraint(constraint, self.variable)

    def add_lower_bound_constraint(
        self, lower_bound: int | float | str, *, is_inclusive: bool = True
    ) -> "Param[_T]":
        """Return a new parameter with an added lower-bound constraint."""
        _validate_natural_bound(
            self.domain, lower_bound, is_lower=True, is_inclusive=is_inclusive
        )
        return self.add_constraint(
            _create_bound_constraint(
                self.variable, lower_bound, is_lower=True, is_inclusive=is_inclusive
            )
        )

    def add_upper_bound_constraint(
        self, upper_bound: int | float | str, *, is_inclusive: bool = True
    ) -> "Param[_T]":
        """Return a new parameter with an added upper-bound constraint."""
        _validate_natural_bound(
            self.domain, upper_bound, is_lower=False, is_inclusive=is_inclusive
        )
        return self.add_constraint(
            _create_bound_constraint(
                self.variable, upper_bound, is_lower=False, is_inclusive=is_inclusive
            )
        )

    # -- interval arithmetic (interval-integer domains only) ----------------

    def _require_interval_domain(self) -> IntervalIntegerDomain:
        if not isinstance(self.domain, IntervalIntegerDomain):
            raise TypeError(
                "Arithmetic is only supported on interval-integer parameters."
            )
        return self.domain

    def _coerce_interval_operand(self, other: Any) -> "Param[int] | None":
        # Both operands are the single ``Param`` type, and Python skips the
        # reflected dunder when operands share a type. A non-interval ``self``
        # must therefore coerce and handle ``non_interval OP interval`` itself
        # rather than relying on the interval operand's reflected method.
        if isinstance(other, Param) and isinstance(other.domain, IntervalIntegerDomain):
            return _coerce_to_interval_param(other, self)
        return None

    def __add__(self, other: Any) -> "Param[int]":
        if not isinstance(self.domain, IntervalIntegerDomain):
            coerced_self = self._coerce_interval_operand(other)
            if coerced_self is None:
                return NotImplemented
            return coerced_self.__add__(other)
        domain = self.domain
        coerced = _coerce_to_interval_param(self, other)
        self_min, self_max = _get_effective_min_max(self.constraints, self.variable)
        other_min, other_max = _get_effective_min_max(
            coerced.constraints, coerced.variable
        )
        new_min = _combine_optional_bounds(self_min, other_min, operator.add)
        new_max = _combine_optional_bounds(self_max, other_max, operator.add)
        return _create_class_preserved_interval_param(
            self, coerced, new_min, new_max, domain
        )

    def __radd__(self, other: Any) -> "Param[int]":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "Param[int]":
        if not isinstance(self.domain, IntervalIntegerDomain):
            coerced_self = self._coerce_interval_operand(other)
            if coerced_self is None:
                return NotImplemented
            return coerced_self.__sub__(other)
        domain = self.domain
        coerced = _coerce_to_interval_param(self, other)
        self_min, self_max = _get_effective_min_max(self.constraints, self.variable)
        other_min, other_max = _get_effective_min_max(
            coerced.constraints, coerced.variable
        )
        new_min = _combine_optional_bounds(self_min, other_max, operator.sub)
        new_max = _combine_optional_bounds(self_max, other_min, operator.sub)
        return _create_widened_interval_param(self.variable, new_min, new_max, domain)

    def __rsub__(self, other: Any) -> "Param[int]":
        if not isinstance(self.domain, IntervalIntegerDomain):
            return NotImplemented
        return _coerce_to_interval_param(self, other).__sub__(self)

    def __neg__(self) -> "Param[int]":
        domain = self._require_interval_domain()
        self_min, self_max = _get_effective_min_max(self.constraints, self.variable)
        new_min = None if self_max is None else -self_max
        new_max = None if self_min is None else -self_min
        return _create_widened_interval_param(self.variable, new_min, new_max, domain)

    @override
    def __repr__(self) -> str:
        set_repr = self.domain.render_set_repr()
        if set_repr:
            set_repr = f"{set_repr}, "
        return (
            f"{type(self).__name__}({self.variable!r}, {set_repr}"
            f"constraints={self.constraints!r})"
        )

    @override
    def __str__(self) -> str:
        land = " /\\ "
        return (
            "{" + f"{self.variable} in {self.domain.render_set_string()} | "
            f"{land.join(str(c) for c in self.constraints)}"
            "}"
        )


# ---------------------------------------------------------------------------
# Parameter assignment
# ---------------------------------------------------------------------------


# The ``param`` field is annotated ``Param[_T]``, a parameterized generic the
# engine cannot resolve to the ``Param`` class for codec inference, so supply the
# serializable-class codec explicitly.
_PARAM_CODEC: FieldCodec = _SerializableFieldCodec(Param)


@register_serializable(type_id="param_assignment")
@dataclass(frozen=True, eq=False)
class ParamAssignment(Serializable, FrozenMixin, DerivedEquivalenceMixin, Generic[_T]):
    """Immutable binding of a parameter definition to a concrete value.

    Two assignments are structurally equivalent when their parameters are
    equivalent and their bound values compare equal.
    """

    param: "Param[_T]" = field(metadata={"serialize_codec": _PARAM_CODEC})
    value: _T = field(
        metadata={
            "serialize_codec": _WRAPPED_VALUE_CODEC,
            **compared_as_value(),
        }
    )

    def __post_init__(self) -> None:
        self.param.validate_value(self.value)

    def is_value_set(self) -> bool:
        """Return whether this assignment has a value."""
        return True


# ---------------------------------------------------------------------------
# Bound constraint helpers and natural-number gates
# ---------------------------------------------------------------------------


def _create_bound_constraint(
    variable: Identifier,
    bound: int | float | str,
    *,
    is_lower: bool,
    is_inclusive: bool,
) -> EquationConstraint:
    variable_expression = IdentifierExpression(variable)
    if is_lower:
        equation = (
            variable_expression >= bound
            if is_inclusive
            else variable_expression > bound
        )
    else:
        equation = (
            variable_expression <= bound
            if is_inclusive
            else variable_expression < bound
        )
    return EquationConstraint(variable, equation)


def _validate_natural_lower_bound(
    bound: int, *, zero_included: bool, is_inclusive: bool
) -> None:
    if zero_included:
        if bound < 0:
            raise ParamError("Lower bound must be non-negative.")
        if not is_inclusive and bound < 1:
            raise ParamError(
                "Lower bound must be at least 1 if zero is included and "
                "bound is exclusive."
            )
    elif is_inclusive:
        if bound < 1:
            raise ParamError(
                "Lower bound must be at least 1 when zero is not included."
            )
    elif bound < 0:
        raise ParamError(
            "Lower bound must be non-negative when zero is not included "
            "and bound is exclusive."
        )


def _validate_natural_upper_bound(
    bound: int, *, zero_included: bool, is_inclusive: bool
) -> None:
    if zero_included:
        if is_inclusive:
            if bound < 0:
                raise ParamError(
                    "Upper bound must be non-negative when zero is included."
                )
        elif bound < 1:
            raise ParamError(
                "Upper bound must be at least 1 if zero is included and "
                "bound is exclusive."
            )
    elif is_inclusive:
        if bound < 1:
            raise ParamError(
                "Upper bound must be at least 1 when zero is not included."
            )
    elif bound < 2:  # noqa: PLR2004
        raise ParamError(
            "Upper bound must be at least 2 when zero is not included "
            "and bound is exclusive."
        )


def _validate_natural_bound(
    domain: ParamDomain,
    bound: int | float | str,
    *,
    is_lower: bool,
    is_inclusive: bool,
) -> None:
    """Apply the natural-number bound gates for non-negative integer domains."""
    if not isinstance(domain, (IntegerDomain, IntervalIntegerDomain)):
        return
    if not domain.non_negative:
        return
    if not isinstance(bound, int):
        return
    zero_included = domain.zero_included
    if is_lower:
        _validate_natural_lower_bound(
            bound, zero_included=zero_included, is_inclusive=is_inclusive
        )
    else:
        _validate_natural_upper_bound(
            bound, zero_included=zero_included, is_inclusive=is_inclusive
        )


# ---------------------------------------------------------------------------
# Interval arithmetic helpers
# ---------------------------------------------------------------------------


def _invert_comparison(operation: BinaryOperation) -> BinaryOperation:
    inverses = {
        BinaryOperation.GREATER: BinaryOperation.LESS,
        BinaryOperation.GREATER_EQUAL: BinaryOperation.LESS_EQUAL,
        BinaryOperation.LESS: BinaryOperation.GREATER,
        BinaryOperation.LESS_EQUAL: BinaryOperation.GREATER_EQUAL,
    }
    if operation not in inverses:
        raise ValueError(f"Cannot invert non-comparison operation: {operation}")
    return inverses[operation]


def _bound_from_literal(
    literal: LiteralExpression, operation: BinaryOperation
) -> tuple[bool, int, bool]:
    value = literal.value
    if not isinstance(value, int):  # pragma: no cover
        raise RuntimeError("Bound expression literal is not an integer.")
    is_lower = operation in (BinaryOperation.GREATER, BinaryOperation.GREATER_EQUAL)
    is_inclusive = operation in (
        BinaryOperation.GREATER_EQUAL,
        BinaryOperation.LESS_EQUAL,
    )
    return is_lower, value, is_inclusive


def _bound_from_constraint(constraint: Constraint) -> tuple[bool, int, bool]:
    if not isinstance(constraint, EquationConstraint):
        raise RuntimeError(
            "Interval parameter has a non-EquationConstraint constraint: "
            f"{type(constraint)}"
        )
    if not is_bound_expression(constraint.convert_to_expression()):
        raise RuntimeError(
            f"Interval parameter has a non-bound constraint: {constraint!r}"
        )
    expression = constraint.convert_to_expression()
    if not isinstance(expression, BinaryExpression):  # pragma: no cover
        raise RuntimeError("Interval parameter has a non-bound constraint.")
    if isinstance(expression.left, IdentifierExpression) and isinstance(
        expression.right, LiteralExpression
    ):
        return _bound_from_literal(expression.right, expression.operation)
    if isinstance(expression.right, IdentifierExpression) and isinstance(
        expression.left, LiteralExpression
    ):
        return _bound_from_literal(
            expression.left, _invert_comparison(expression.operation)
        )
    raise RuntimeError("Interval bound expression is malformed.")  # pragma: no cover


def _iter_interval_bounds(
    constraints: Sequence[Constraint],
) -> list[tuple[bool, int, bool]]:
    return [_bound_from_constraint(constraint) for constraint in constraints]


def _get_effective_min_max(
    constraints: Sequence[Constraint], variable: Identifier
) -> tuple[int | None, int | None]:
    min_int: int | None = None
    max_int: int | None = None
    for is_lower, bound, inclusive in _iter_interval_bounds(constraints):
        if is_lower:
            effective = bound if inclusive else bound + 1
            min_int = effective if min_int is None else max(min_int, effective)
        else:
            effective = bound if inclusive else bound - 1
            max_int = effective if max_int is None else min(max_int, effective)
    if min_int is not None and max_int is not None and min_int > max_int:
        raise ParamError(
            f"Empty integer interval represented by constraints for {variable}."
        )
    return min_int, max_int


def _combine_optional_bounds(
    left: int | None,
    right: int | None,
    combine: Callable[[int, int], int],
) -> int | None:
    """Combine two optional bounds, yielding ``None`` if either is unbounded."""
    if left is None or right is None:
        return None
    return combine(left, right)


def _apply_interval_bounds(
    param: "Param[int]", min_int: int | None, max_int: int | None
) -> "Param[int]":
    domain = cast(IntervalIntegerDomain, param.domain)
    if min_int is not None:
        if domain.prefer_inclusive:
            param = param.add_lower_bound_constraint(min_int, is_inclusive=True)
        else:
            param = param.add_lower_bound_constraint(min_int - 1, is_inclusive=False)
    if max_int is not None:
        if domain.prefer_inclusive:
            param = param.add_upper_bound_constraint(max_int, is_inclusive=True)
        else:
            param = param.add_upper_bound_constraint(max_int + 1, is_inclusive=False)
    return param


def _create_widened_interval_param(
    variable: Identifier,
    min_int: int | None,
    max_int: int | None,
    template_domain: IntervalIntegerDomain,
) -> "Param[int]":
    param: Param[int] = Param(
        IntervalIntegerDomain(prefer_inclusive=template_domain.prefer_inclusive),
        variable=variable,
    )
    return _apply_interval_bounds(param, min_int, max_int)


def _create_class_preserved_interval_param(
    template: "Param[Any]",
    other: "Param[Any]",
    min_int: int | None,
    max_int: int | None,
    template_domain: IntervalIntegerDomain,
) -> "Param[int]":
    other_domain = other.domain
    if (
        isinstance(other_domain, IntervalIntegerDomain)
        and template_domain.non_negative
        and other_domain.non_negative
    ):
        param: Param[int] = Param(
            IntervalIntegerDomain(
                prefer_inclusive=template_domain.prefer_inclusive,
                non_negative=True,
                zero_included=template_domain.zero_included,
            ),
            variable=template.variable,
        )
        return _apply_interval_bounds(param, min_int, max_int)
    return _create_widened_interval_param(
        template.variable, min_int, max_int, template_domain
    )


def _coerce_to_interval_param(template: "Param[Any]", other: Any) -> "Param[int]":
    template_domain = cast(IntervalIntegerDomain, template.domain)
    if isinstance(other, bool):
        raise TypeError(f"Unsupported operand type: {type(other)}")
    if isinstance(other, int):
        return create_interval_integer_param_exactly(
            other, prefer_inclusive=template_domain.prefer_inclusive
        )
    if isinstance(other, Param) and isinstance(other.domain, IntervalIntegerDomain):
        return other
    if isinstance(other, Param) and isinstance(other.domain, IntegerDomain):
        for constraint in other.constraints:
            if not is_bound_expression(constraint.convert_to_expression()):
                raise TypeError(
                    "Cannot coerce an integer parameter with non-bound "
                    "constraints to an interval parameter."
                )
        return Param(
            IntervalIntegerDomain(prefer_inclusive=template_domain.prefer_inclusive),
            variable=other.variable,
            constraints=other.constraints,
        )
    raise TypeError(f"Unsupported operand type: {type(other)}")


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def create_integer_param(
    *, name: Identifier | None = None, constraints: Sequence[Constraint] = ()
) -> Param[int]:
    """Create an integer-valued parameter."""
    return Param(
        IntegerDomain(),
        variable=name or Identifier("param"),
        constraints=tuple(constraints),
    )


def create_natural_param(
    *,
    name: Identifier | None = None,
    zero_included: bool = True,
    constraints: Sequence[Constraint] = (),
) -> Param[int]:
    """Create a natural-number (non-negative integer) parameter."""
    return Param(
        IntegerDomain(non_negative=True, zero_included=zero_included),
        variable=name or Identifier("param"),
        constraints=tuple(constraints),
    )


def create_real_param(
    *, name: Identifier | None = None, constraints: Sequence[Constraint] = ()
) -> Param[str | float]:
    """Create a real-valued parameter."""
    return Param(
        RealDomain(),
        variable=name or Identifier("param"),
        constraints=tuple(constraints),
    )


def create_integer_param_between(
    lower_bound: int,
    upper_bound: int,
    *,
    name: Identifier | None = None,
    is_lower_inclusive: bool = True,
    is_upper_inclusive: bool = True,
) -> Param[int]:
    """Create an integer parameter bounded to ``[lower_bound, upper_bound]``."""
    if lower_bound > upper_bound or (
        lower_bound == upper_bound and not (is_lower_inclusive and is_upper_inclusive)
    ):
        raise ParamError("Lower bound must be less than or equal to upper bound.")
    param = create_integer_param(name=name)
    param = param.add_lower_bound_constraint(
        lower_bound, is_inclusive=is_lower_inclusive
    )
    return param.add_upper_bound_constraint(
        upper_bound, is_inclusive=is_upper_inclusive
    )


def create_integer_param_with_lower_bound(
    lower_bound: int, *, name: Identifier | None = None, is_inclusive: bool = True
) -> Param[int]:
    """Create an integer parameter with a lower bound."""
    return create_integer_param(name=name).add_lower_bound_constraint(
        lower_bound, is_inclusive=is_inclusive
    )


def create_integer_param_with_upper_bound(
    upper_bound: int, *, name: Identifier | None = None, is_inclusive: bool = True
) -> Param[int]:
    """Create an integer parameter with an upper bound."""
    return create_integer_param(name=name).add_upper_bound_constraint(
        upper_bound, is_inclusive=is_inclusive
    )


def create_real_param_between(
    lower_bound: float | str,
    upper_bound: float | str,
    *,
    name: Identifier | None = None,
    is_lower_inclusive: bool = True,
    is_upper_inclusive: bool = True,
) -> Param[str | float]:
    """Create a real parameter bounded to ``[lower_bound, upper_bound]``."""
    if float(lower_bound) > float(upper_bound) or (
        float(lower_bound) == float(upper_bound)
        and not (is_lower_inclusive and is_upper_inclusive)
    ):
        raise ParamError("Lower bound must be less than or equal to upper bound.")
    param = create_real_param(name=name)
    param = param.add_lower_bound_constraint(
        lower_bound, is_inclusive=is_lower_inclusive
    )
    return param.add_upper_bound_constraint(
        upper_bound, is_inclusive=is_upper_inclusive
    )


def create_real_param_with_lower_bound(
    lower_bound: float | str,
    *,
    name: Identifier | None = None,
    is_inclusive: bool = True,
) -> Param[str | float]:
    """Create a real parameter with a lower bound."""
    return create_real_param(name=name).add_lower_bound_constraint(
        lower_bound, is_inclusive=is_inclusive
    )


def create_real_param_with_upper_bound(
    upper_bound: float | str,
    *,
    name: Identifier | None = None,
    is_inclusive: bool = True,
) -> Param[str | float]:
    """Create a real parameter with an upper bound."""
    return create_real_param(name=name).add_upper_bound_constraint(
        upper_bound, is_inclusive=is_inclusive
    )


def create_interval_integer_param(
    *,
    name: Identifier | None = None,
    prefer_inclusive: bool = True,
    non_negative: bool = False,
    zero_included: bool = True,
) -> Param[int]:
    """Create an interval-integer parameter (supports interval arithmetic)."""
    return Param(
        IntervalIntegerDomain(
            prefer_inclusive=prefer_inclusive,
            non_negative=non_negative,
            zero_included=zero_included,
        ),
        variable=name or Identifier("param"),
    )


def create_interval_integer_param_between(
    lower_bound: int,
    upper_bound: int,
    *,
    name: Identifier | None = None,
    is_lower_inclusive: bool = True,
    is_upper_inclusive: bool = True,
    prefer_inclusive: bool = True,
) -> Param[int]:
    """Create an interval-integer parameter bounded to ``[lower, upper]``."""
    param = create_interval_integer_param(name=name, prefer_inclusive=prefer_inclusive)
    param = param.add_lower_bound_constraint(
        lower_bound, is_inclusive=is_lower_inclusive
    )
    param = param.add_upper_bound_constraint(
        upper_bound, is_inclusive=is_upper_inclusive
    )
    _get_effective_min_max(param.constraints, param.variable)
    return param


def create_interval_integer_param_with_lower_bound(
    lower_bound: int,
    *,
    name: Identifier | None = None,
    is_inclusive: bool = True,
    prefer_inclusive: bool = True,
) -> Param[int]:
    """Create an interval-integer parameter with a lower bound."""
    return create_interval_integer_param(
        name=name, prefer_inclusive=prefer_inclusive
    ).add_lower_bound_constraint(lower_bound, is_inclusive=is_inclusive)


def create_interval_integer_param_with_upper_bound(
    upper_bound: int,
    *,
    name: Identifier | None = None,
    is_inclusive: bool = True,
    prefer_inclusive: bool = True,
) -> Param[int]:
    """Create an interval-integer parameter with an upper bound."""
    return create_interval_integer_param(
        name=name, prefer_inclusive=prefer_inclusive
    ).add_upper_bound_constraint(upper_bound, is_inclusive=is_inclusive)


def create_interval_integer_param_exactly(
    value: int, *, name: Identifier | None = None, prefer_inclusive: bool = True
) -> Param[int]:
    """Create an interval-integer parameter bounded to exactly ``value``."""
    param = create_interval_integer_param(name=name, prefer_inclusive=prefer_inclusive)
    param = param.add_lower_bound_constraint(value, is_inclusive=True)
    return param.add_upper_bound_constraint(value, is_inclusive=True)


def create_interval_natural_param(
    *,
    name: Identifier | None = None,
    zero_included: bool = True,
    prefer_inclusive: bool = True,
) -> Param[int]:
    """Create a non-negative interval-integer parameter."""
    return create_interval_integer_param(
        name=name,
        prefer_inclusive=prefer_inclusive,
        non_negative=True,
        zero_included=zero_included,
    )


def create_ordinal_param(
    values: Sequence[_OrdinalValueT], *, name: Identifier | None = None
) -> Param[_OrdinalValueT]:
    """Create an ordinal parameter over a finite, ordered value set."""
    return Param(build_ordinal_domain(values), variable=name or Identifier("param"))


def create_categorical_param(
    categories: Collection[_CategoricalValueT], *, name: Identifier | None = None
) -> Param[_CategoricalValueT]:
    """Create a categorical parameter over a finite, unordered value set."""
    return Param(
        build_categorical_domain(tuple(categories)),
        variable=name or Identifier("param"),
    )


def create_permutation_param(
    members: Sequence[_PermutationMemberValueT], *, name: Identifier | None = None
) -> Param[tuple[_PermutationMemberValueT, ...]]:
    """Create a permutation parameter over a fixed, ordered set of members."""
    return Param(
        build_permutation_domain(members), variable=name or Identifier("param")
    )


def create_single_valid_value_param(
    value: _CategoricalValueT, *, name: Identifier | None = None
) -> Param[_CategoricalValueT]:
    """Create a parameter that admits only a single value."""
    return create_categorical_param([value], name=name)
