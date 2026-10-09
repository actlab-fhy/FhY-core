"""Constrained parameters built by composing a value domain.

A :class:`Param` pairs a variable identifier and a
:class:`~fhy_core.symbolic.constraint.ConstraintSystem` with a
:class:`~fhy_core.symbolic.param.domains.ParamDomain` that supplies all kind-specific
behavior. There is a single concrete ``Param`` class; the common kinds are built
through the ``create_*`` factory functions.

``Param`` and ``ParamAssignment`` run on the Rust core (``fhy_core::param``):
each is a thin subclass of its ``fhy_core._rs`` class, which implements
construction and the canonical constraints, value checks, the questions,
bounds and their natural-number gates, interval arithmetic, union and
intersection, equivalence, freezing, pickling and the payload, and keeps the
Python objects it was given. Their ``==`` and ``hash`` are identity. A
parameter serializes as ``{"domain": ..., "variable": ...,
"constraint_system": ...}``, the ``domain`` field a wrapped family envelope
identifying the concrete domain.
"""

from collections.abc import Collection, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from fhy_core import _rs
from fhy_core.identifier import Identifier
from fhy_core.serialization import Serializable, register_serializable
from fhy_core.symbolic._native_slots import copy_native_attributes
from fhy_core.symbolic.constraint import (
    Constraint,
    ConstraintBindings,
    ConstraintOutcome,
    ConstraintSystem,
    create_constraint_system,
)
from fhy_core.symbolic.expression import IdentifierExpression
from fhy_core.symbolic.symbol_type import SymbolType
from fhy_core.traits import FrozenMixin
from fhy_core.utils.override import override

from .domains import (
    IntegerDomain,
    IntervalIntegerDomain,
    ParamDomain,
    RealDomain,
    build_categorical_domain,
    build_ordinal_domain,
    build_permutation_domain,
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
    "create_intersection_param",
    "create_interval_integer_param",
    "create_interval_integer_param_between",
    "create_interval_integer_param_exactly",
    "create_interval_integer_param_with_lower_bound",
    "create_interval_integer_param_with_upper_bound",
    "create_interval_natural_param",
    "create_natural_param",
    "create_natural_param_between",
    "create_natural_param_with_lower_bound",
    "create_natural_param_with_upper_bound",
    "create_ordinal_param",
    "create_permutation_param",
    "create_real_param",
    "create_real_param_between",
    "create_real_param_with_lower_bound",
    "create_real_param_with_upper_bound",
    "create_single_valid_value_param",
    "create_union_param",
]

_T = TypeVar("_T")


@register_serializable(type_id="param")
class Param(_rs.Param, Serializable, Generic[_T]):
    """A constrained parameter that composes a value domain.

    A parameter is defined by its variable, a set of constraints, and a
    :class:`~fhy_core.symbolic.param.domains.ParamDomain` that supplies admissibility,
    subset, set algebra, equivalence, and serialization behavior. Construct one
    directly with a domain, or use a ``create_*`` factory for the common kinds.

    The ``Param[_T]`` type parameter is an advisory hint for call-site inference
    only; the admissible value type is enforced by the domain at runtime, not by
    ``_T``.

    Two parameters are structurally equivalent when they have the same variable,
    domain, and constraints; under alpha comparison they are equivalent up to
    renaming of the bound variable. Construction validates and de-duplicates the
    constraints, appends the domain's implied constraints, and stores the result
    as a :class:`~fhy_core.symbolic.constraint.ConstraintSystem`, which orders its
    members canonically by construction, so that constraint-set order does not
    affect equivalence.
    """

    # The attributes are copied into slots on construction, so reading one
    # costs a slot read rather than a call into the extension.
    __slots__ = ("constraint_system", "constraints", "domain", "variable")

    domain: ParamDomain
    variable: Identifier
    constraint_system: ConstraintSystem
    constraints: tuple[Constraint, ...]

    if TYPE_CHECKING:
        # The Rust class's methods, typed over the value type `_T`.

        @property
        @override
        def variable_expression(self) -> IdentifierExpression: ...
        @property
        @override
        def symbol_type(self) -> SymbolType | None: ...
        @override
        def replace_constraints(
            self, constraints: Sequence[Constraint]
        ) -> "Param[_T]": ...
        @override
        def is_value_set_subset(self, other: "Param[_T]") -> bool: ...
        @override
        def check_subset(self, other: "Param[_T]") -> ConstraintOutcome: ...
        @override
        def is_subset(self, other: "Param[_T]") -> bool: ...
        @override
        def assign(
            self, value: _T, *, bindings: ConstraintBindings | None = None
        ) -> "ParamAssignment[_T]": ...
        @override
        def add_constraint(self, constraint: Constraint) -> "Param[_T]": ...
        @override
        def add_constraints(
            self, constraints: Collection[Constraint]
        ) -> "Param[_T]": ...
        @override
        def add_lower_bound_constraint(
            self, lower_bound: int | float | str, *, is_inclusive: bool = True
        ) -> "Param[_T]": ...
        @override
        def add_upper_bound_constraint(
            self, upper_bound: int | float | str, *, is_inclusive: bool = True
        ) -> "Param[_T]": ...
        @override
        def __add__(self, other: Any) -> "Param[int]": ...
        @override
        def __radd__(self, other: Any) -> "Param[int]": ...
        @override
        def __sub__(self, other: Any) -> "Param[int]": ...
        @override
        def __rsub__(self, other: Any) -> "Param[int]": ...
        @override
        def __mul__(self, other: Any) -> "Param[int]": ...
        @override
        def __rmul__(self, other: Any) -> "Param[int]": ...
        @override
        def __neg__(self) -> "Param[int]": ...
        @override
        def __or__(self, other: "Param[_T]") -> "Param[_T]": ...
        @override
        def __and__(self, other: "Param[_T]") -> "Param[_T]": ...
        @override
        def union(
            self, other: "Param[_T]", name: Identifier | None = None
        ) -> "Param[_T]": ...
        @override
        def intersection(
            self, other: "Param[_T]", name: Identifier | None = None
        ) -> "Param[_T]": ...

    def __init__(
        self,
        domain: ParamDomain,
        variable: Identifier | None = None,
        constraint_system: ConstraintSystem | None = None,
    ) -> None:
        copy_native_attributes(
            self,
            Param,
            _rs.Param,
            "constraint_system",
            "constraints",
            "domain",
            "variable",
        )


@register_serializable(type_id="param_assignment")
class ParamAssignment(_rs.ParamAssignment, Serializable, Generic[_T]):
    """Immutable binding of a parameter definition to a concrete value.

    Two assignments are structurally equivalent when their parameters are
    equivalent and their bound values compare equal.

    Raises:
        ParamError: If ``value`` is not a valid assignment for ``param``,
            as :meth:`Param.validate_value` decides it without bindings.
        ConstraintError: As :meth:`Param.validate_value` raises it;
            without bindings, only for ``value`` itself.
        NonBooleanLogicalOperandError: As :meth:`Param.validate_value`
            raises it.

    """

    # The attributes are copied into slots on construction, so reading one
    # costs a slot read rather than a call into the extension.
    __slots__ = ("param", "value")

    param: Param[_T]
    value: _T

    if TYPE_CHECKING:
        # The Rust class's methods, typed over the value type `_T`.

        @classmethod
        @override
        def construct_from_fields(
            cls, fields: Mapping[str, Any]
        ) -> "ParamAssignment[Any]": ...

    def __init__(self, param: Param[_T], value: _T) -> None:
        copy_native_attributes(
            self, ParamAssignment, _rs.ParamAssignment, "param", "value"
        )


# The classes are registered, not derived: `FrozenMixin` carries an instance
# layout a Rust-backed class cannot share.
FrozenMixin.register(Param)
FrozenMixin.register(ParamAssignment)


def create_integer_param(
    *, name: Identifier | None = None, constraints: Sequence[Constraint] = ()
) -> Param[int]:
    """Create an integer-valued parameter."""
    return Param(
        IntegerDomain(),
        variable=name or Identifier("param"),
        constraint_system=create_constraint_system(*constraints),
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
        constraint_system=create_constraint_system(*constraints),
    )


def create_real_param(
    *, name: Identifier | None = None, constraints: Sequence[Constraint] = ()
) -> Param[str | float]:
    """Create a real-valued parameter."""
    return Param(
        RealDomain(),
        variable=name or Identifier("param"),
        constraint_system=create_constraint_system(*constraints),
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
    _rs.check_param_bounds_are_ordered(
        lower_bound, upper_bound, is_lower_inclusive, is_upper_inclusive
    )
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


def create_natural_param_between(
    lower_bound: int,
    upper_bound: int,
    *,
    name: Identifier | None = None,
    zero_included: bool = True,
    is_lower_inclusive: bool = True,
    is_upper_inclusive: bool = True,
) -> Param[int]:
    """Create a natural-number parameter between ``lower_bound`` and ``upper_bound``.

    Each bound admits its own value only when its inclusivity flag is set.
    Bound factories otherwise reject only bounds that enclose no value in
    any number system; the natural factories are the one exception, since
    they also reject a bound literal the natural domain does not admit (see
    Args), even when the set it bounds is not empty. With zero excluded,
    for example, ``[0, 3]`` is refused although it holds 1, 2, and 3.

    Args:
        lower_bound: Lower bound. With zero included, it must be at least 0
            when inclusive and at least 1 when exclusive; with zero
            excluded, at least 1 when inclusive and at least 0 when
            exclusive.
        upper_bound: Upper bound. It must admit the domain's least member,
            0 with zero included and 1 without, so be at least that member
            when inclusive and exceed it when exclusive. It must also not
            lie below ``lower_bound``, and may equal it only when both
            bounds are inclusive.
        name: Variable for the parameter; defaults to a fresh
            ``Identifier("param")``.
        zero_included: Whether zero belongs to the domain.
        is_lower_inclusive: Whether ``lower_bound`` itself is admitted.
        is_upper_inclusive: Whether ``upper_bound`` itself is admitted.

    Returns:
        The bounded natural-number parameter. Bounds that pass every check
        yet enclose no integer, such as ``(1, 2)`` with both ends exclusive,
        give an empty parameter.

    Raises:
        ParamError: If the bounds are reversed or equal with an exclusive
            side, or if either bound falls below the minimum Args gives for
            it.

    """
    _rs.check_param_bounds_are_ordered(
        lower_bound, upper_bound, is_lower_inclusive, is_upper_inclusive
    )
    param = create_natural_param(name=name, zero_included=zero_included)
    param = param.add_lower_bound_constraint(
        lower_bound, is_inclusive=is_lower_inclusive
    )
    return param.add_upper_bound_constraint(
        upper_bound, is_inclusive=is_upper_inclusive
    )


def create_natural_param_with_lower_bound(
    lower_bound: int,
    *,
    name: Identifier | None = None,
    zero_included: bool = True,
    is_inclusive: bool = True,
) -> Param[int]:
    """Create a natural-number parameter with a lower bound.

    Args:
        lower_bound: Lower bound. With zero included, it must be at least 0
            when inclusive and at least 1 when exclusive; with zero
            excluded, at least 1 when inclusive and at least 0 when
            exclusive.
        name: Variable for the parameter; defaults to a fresh
            ``Identifier("param")``.
        zero_included: Whether zero belongs to the domain.
        is_inclusive: Whether ``lower_bound`` itself is admitted.

    Returns:
        The natural-number parameter bounded below by ``lower_bound``, which
        is never empty.

    Raises:
        ParamError: If ``lower_bound`` falls below the minimum Args gives for
            it.

    """
    return create_natural_param(
        name=name, zero_included=zero_included
    ).add_lower_bound_constraint(lower_bound, is_inclusive=is_inclusive)


def create_natural_param_with_upper_bound(
    upper_bound: int,
    *,
    name: Identifier | None = None,
    zero_included: bool = True,
    is_inclusive: bool = True,
) -> Param[int]:
    """Create a natural-number parameter with an upper bound.

    Args:
        upper_bound: Upper bound. It must admit the domain's least member,
            0 with zero included and 1 without, so be at least that member
            when inclusive and exceed it when exclusive.
        name: Variable for the parameter; defaults to a fresh
            ``Identifier("param")``.
        zero_included: Whether zero belongs to the domain.
        is_inclusive: Whether ``upper_bound`` itself is admitted.

    Returns:
        The natural-number parameter bounded above by ``upper_bound``, which
        is never empty.

    Raises:
        ParamError: If ``upper_bound`` does not admit the domain's least
            member.

    """
    return create_natural_param(
        name=name, zero_included=zero_included
    ).add_upper_bound_constraint(upper_bound, is_inclusive=is_inclusive)


def create_real_param_between(
    lower_bound: float | str,
    upper_bound: float | str,
    *,
    name: Identifier | None = None,
    is_lower_inclusive: bool = True,
    is_upper_inclusive: bool = True,
) -> Param[str | float]:
    """Create a real parameter bounded to ``[lower_bound, upper_bound]``."""
    _rs.check_param_bounds_are_ordered(
        lower_bound, upper_bound, is_lower_inclusive, is_upper_inclusive
    )
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
    _rs.check_param_bounds_are_ordered(
        lower_bound, upper_bound, is_lower_inclusive, is_upper_inclusive
    )
    param = create_interval_integer_param(name=name, prefer_inclusive=prefer_inclusive)
    param = param.add_lower_bound_constraint(
        lower_bound, is_inclusive=is_lower_inclusive
    )
    return param.add_upper_bound_constraint(
        upper_bound, is_inclusive=is_upper_inclusive
    )


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
    """Create an ordinal parameter over a finite, ordered value set.

    Raises:
        ParamError: If ``values`` is empty, contains duplicates, or
            contains NaN.
        TypeError: If a value is not ordinal or values are not mutually
            comparable.

    """
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
    """Create a permutation parameter over a fixed, ordered set of members.

    Raises:
        ParamError: If ``members`` is empty, contains duplicates, or
            contains NaN.
        TypeError: If a member is not a permutation member value.

    """
    return Param(
        build_permutation_domain(members), variable=name or Identifier("param")
    )


def create_single_valid_value_param(
    value: _CategoricalValueT, *, name: Identifier | None = None
) -> Param[_CategoricalValueT]:
    """Create a parameter that admits only a single value."""
    return create_categorical_param((value,), name=name)


# ---------------------------------------------------------------------------
# Set algebra
# ---------------------------------------------------------------------------


def create_union_param(
    left: Param[_T],
    right: Param[_T],
    *,
    name: Identifier | None = None,
) -> Param[_T]:
    """Create a parameter admitting exactly the values valid for either operand.

    Both operands' constraints are folded into the result: each operand's
    member set is filtered by its own constraints before the sets are merged,
    so the result carries no constraints of its own.

    Args:
        left: Left operand; must have an ordinal or categorical domain.
        right: Right operand; must have the same domain kind as ``left``.
        name: Variable for the result; defaults to a fresh
            ``Identifier("param")``.

    Returns:
        A new parameter over the union of the operands' effective value sets.

    Raises:
        TypeError: If either operand's domain kind does not support union, the
            kinds differ, or merged ordinal values are not mutually comparable.
        ParamError: If both operands' effective value sets are empty, so the
            union would be empty.

    """
    return left.union(right, name)


def create_intersection_param(
    left: Param[_T],
    right: Param[_T],
    *,
    name: Identifier | None = None,
) -> Param[_T]:
    """Create a parameter admitting exactly the values valid for both operands.

    Finite-set operands are intersected by baking both effective value sets;
    permutation operands keep their member set, and numeric operands merge
    domain attributes conservatively -- both of these kinds carry the
    conjunction of both operands' constraints with both operands' variables
    renamed to the result variable, so a constraint relating the two
    operands becomes a constraint on the result alone. A mixed pair of one
    interval-integer parameter and one plain integer parameter whose
    constraints are all bound expressions is supported by coercing the
    plain parameter to interval form first; the coerced operand contributes
    any sign bound as a carried constraint rather than as a domain
    attribute.

    Args:
        left: Left operand.
        right: Right operand; must have the same domain kind as ``left``
            (modulo the interval/integer coercion above).
        name: Variable for the result; defaults to a fresh
            ``Identifier("param")``.

    Returns:
        A new parameter over the intersection of the operands' feasible
        sets. A conjunction the solver leaves undecided is returned live;
        its :meth:`Param.check_feasibility` reports ``UNDECIDED``, which
        tells it apart from one decided feasible.

    Raises:
        TypeError: If the domain kinds are incompatible, or a mixed
            interval-integer/plain-integer pair's plain operand carries a
            non-bound constraint, so it has no interval form.
        ParamError: If the intersection is provably empty: an empty
            finite-set intersection, permutation operands over different
            member sets, a numeric conjunction the enumeration or the
            solver proves infeasible, or an operand that is itself proven
            infeasible.
        NonBooleanLogicalOperandError: If a carried constraint holds a
            provably numeric operand in a Boolean position, as the
            emptiness check through :meth:`Param.check_feasibility`
            raises it.

    """
    return left.intersection(right, name)
