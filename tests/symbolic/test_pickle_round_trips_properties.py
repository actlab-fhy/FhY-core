"""Pickle and deepcopy round trips for the frozen types that hold identifiers.

An ``EquationConstraint``/``InSetConstraint``/``NotInSetConstraint``, a
``Param``, and a ``ParamAssignment`` are all frozen value objects that reach
an ``Identifier`` -- a constraint through the free identifiers of its
expression, a parameter through the variable it binds. Duplicating one must
return an independent object that is still frozen and still equivalent to
its source, so neither the freeze flag nor any derived state is lost or
shared on the way through.

A test identifier is a ``Mock(spec=Identifier)``, which pickle refuses to
serialize. Handing every identifier to the pickler as a persistent reference
keeps the object under test itself on the real ``dumps``/``loads`` path (P28).
"""

import copy
import io
import pickle
from collections.abc import Callable
from typing import Any, Final, TypeVar

import pytest

pytest.importorskip("hypothesis")
from hypothesis import example, given
from hypothesis import strategies as st

from fhy_core.identifier import Identifier
from fhy_core.symbolic.constraint import Constraint, EquationConstraint
from fhy_core.symbolic.expression import IdentifierExpression, LiteralExpression
from fhy_core.symbolic.param import (
    Param,
    ParamAssignment,
    create_categorical_param,
    create_integer_param,
    create_integer_param_between,
    create_natural_param,
    create_ordinal_param,
    create_permutation_param,
    create_real_param_between,
    create_single_valid_value_param,
)
from fhy_core.utils.override import override

from ..strategies.constraints import (
    draw_bound_equation_constraint,
    draw_in_set_constraint,
    draw_not_in_set_constraint,
)
from ..strategies.identifiers import build_identifier_pool, build_identifier_strategy
from ..strategies.params import (
    build_categorical_value_set_strategy,
    build_ordinal_value_set_strategy,
    build_permutation_member_set_strategy,
    draw_ordered_optional_bounds,
)
from .conftest import mock_identifier
from .param.conftest import build_interval_integer_param

pytestmark = pytest.mark.property

_T = TypeVar("_T")


class _IdentifierByReferencePickler(pickle.Pickler):
    """Pickler that emits identifiers as external references."""

    referenced: dict[str, Identifier]

    def __init__(self, file: Any, referenced: dict[str, Identifier]) -> None:
        super().__init__(file)
        self.referenced = referenced

    @override
    def persistent_id(self, obj: Any) -> str | None:
        if isinstance(obj, Identifier):
            key = str(id(obj))
            self.referenced[key] = obj
            return key
        return None


class _IdentifierByReferenceUnpickler(pickle.Unpickler):
    """Unpickler resolving the external identifier references by key."""

    referenced: dict[str, Identifier]

    def __init__(self, file: Any, referenced: dict[str, Identifier]) -> None:
        super().__init__(file)
        self.referenced = referenced

    @override
    def persistent_load(self, pid: Any) -> Identifier:
        return self.referenced[pid]


def round_trip_through_pickle(value: _T) -> _T:
    """Return the value after a ``pickle.dumps``/``loads`` round trip.

    Routes every ``Identifier`` through a persistent reference (see the
    module docstring): a ``Mock(spec=Identifier)`` is otherwise unpicklable.
    """
    referenced: dict[str, Identifier] = {}
    buffer = io.BytesIO()
    _IdentifierByReferencePickler(buffer, referenced).dump(value)
    buffer.seek(0)
    restored = _IdentifierByReferenceUnpickler(buffer, referenced).load()
    assert isinstance(restored, type(value))
    return restored


Duplicator = Callable[[Any], Any]

_DUPLICATORS = [
    pytest.param(round_trip_through_pickle, id="pickle"),
    pytest.param(copy.deepcopy, id="deepcopy"),
]


# =============================================================================
# Constraints
# =============================================================================

_CONSTRAINT_POOL: Final = build_identifier_pool(4, name_prefix="k")


@st.composite
def draw_constraint_over_one_identifier(draw: st.DrawFn) -> Constraint:
    """Draw an equation, in-set, or not-in-set constraint over a pooled identifier."""
    variable = draw(build_identifier_strategy(_CONSTRAINT_POOL))
    result: Constraint = draw(
        st.one_of(
            draw_bound_equation_constraint(variable),
            draw_in_set_constraint(variable),
            draw_not_in_set_constraint(variable),
        )
    )
    return result


_WELD_CONSTRAINT_VARIABLE = mock_identifier("x", 0)
_WELD_CONSTRAINT = EquationConstraint(
    IdentifierExpression(_WELD_CONSTRAINT_VARIABLE) < LiteralExpression(5)
)
"""The hand-built constraint the pre-generalization version of this test used."""


@pytest.mark.parametrize("duplicate", _DUPLICATORS)
@example(constraint=_WELD_CONSTRAINT)
@given(constraint=draw_constraint_over_one_identifier())
def test_constraint_survives_duplication(
    constraint: Constraint, duplicate: Duplicator
) -> None:
    """Test a constraint reaching an identifier duplicates frozen and equivalent.

    Oracle: the deleted ``tests/symbolic/test_pickle_round_trips.py``
    assertions, generalized from one hand-built ``EquationConstraint`` to
    every constraint kind the shared strategies draw.
    """
    duplicated = duplicate(constraint)

    assert duplicated is not constraint
    assert duplicated.is_frozen
    assert duplicated.get_free_identifiers() == constraint.get_free_identifiers()
    assert duplicated.is_structurally_equivalent(constraint)
    assert duplicated.is_alpha_equivalent(constraint)


# =============================================================================
# Params
# =============================================================================


_CATEGORICAL_ALPHABET: Final = tuple("abcdefgh")


@st.composite
def draw_interval_integer_param(draw: st.DrawFn, limit: int = 25) -> Param[int]:
    """Draw an interval-integer param over optional, ordered bounds."""
    lower, upper = draw(draw_ordered_optional_bounds(limit))
    return build_interval_integer_param(lower, upper)


@st.composite
def draw_bounded_integer_param(draw: st.DrawFn, limit: int = 25) -> Param[int]:
    """Draw an integer param bounded to ``[lower, upper]``, ``lower <= upper``."""
    lower = draw(st.integers(min_value=-limit, max_value=limit))
    extra = draw(st.integers(min_value=0, max_value=2 * limit))
    upper = min(lower + extra, limit)
    return create_integer_param_between(lower, upper)


@st.composite
def draw_natural_param(draw: st.DrawFn) -> Param[int]:
    """Draw a natural-number param, with or without zero included."""
    zero_included = draw(st.booleans())
    return create_natural_param(zero_included=zero_included)


@st.composite
def draw_ordinal_param(draw: st.DrawFn) -> Param[int]:
    """Draw an ordinal param over a finite, sorted set of ints."""
    values = draw(build_ordinal_value_set_strategy())
    return create_ordinal_param(values)


@st.composite
def draw_categorical_param(draw: st.DrawFn) -> Param[str]:
    """Draw a categorical param over a finite set of letters."""
    categories = draw(build_categorical_value_set_strategy())
    return create_categorical_param(categories)


@st.composite
def draw_permutation_param(draw: st.DrawFn) -> Param[tuple[str, ...]]:
    """Draw a permutation param over a fixed, ordered set of letters."""
    members = draw(build_permutation_member_set_strategy())
    return create_permutation_param(members)


@st.composite
def draw_bounded_real_param(draw: st.DrawFn, limit: int = 25) -> Param[str | float]:
    """Draw a real param bounded to ``[lower, upper]`` with integer-valued bounds."""
    lower = draw(st.integers(min_value=-limit, max_value=limit))
    extra = draw(st.integers(min_value=0, max_value=2 * limit))
    upper = lower + extra
    return create_real_param_between(float(lower), float(upper))


@st.composite
def draw_single_valid_value_param(draw: st.DrawFn) -> Param[str]:
    """Draw a param admitting exactly one letter from a fixed alphabet."""
    value = draw(st.sampled_from(_CATEGORICAL_ALPHABET))
    return create_single_valid_value_param(value)


@st.composite
def draw_param_over_any_domain(draw: st.DrawFn) -> Param[Any]:
    """Draw a param over any of this module's domain kinds."""
    result: Param[Any] = draw(
        st.one_of(
            draw_interval_integer_param(),
            draw_bounded_integer_param(),
            draw_natural_param(),
            draw_ordinal_param(),
            draw_categorical_param(),
            draw_permutation_param(),
            draw_bounded_real_param(),
            draw_single_valid_value_param(),
        )
    )
    return result


_WELD_PARAM = create_integer_param(name=mock_identifier("p", 1))
"""The hand-built parameter the pre-generalization version of this test used."""


@pytest.mark.parametrize("duplicate", _DUPLICATORS)
@example(param=_WELD_PARAM)
@given(param=draw_param_over_any_domain())
def test_param_survives_duplication(param: Param[Any], duplicate: Duplicator) -> None:
    """Test a parameter duplicates frozen, with its binder and domain intact.

    Oracle: the deleted ``tests/symbolic/test_pickle_round_trips.py``
    assertions, generalized from one integer param to every domain kind the
    shared strategies draw.
    """
    duplicated = duplicate(param)

    assert duplicated is not param
    assert duplicated.is_frozen
    assert duplicated.variable == param.variable
    assert duplicated.domain.is_structurally_equivalent(param.domain)
    assert duplicated.is_structurally_equivalent(param)
    assert duplicated.is_alpha_equivalent(param)


# =============================================================================
# Param assignments
# =============================================================================
#
# ``ParamAssignment.__post_init__`` re-validates its value against the
# param's domain and constraints, so each domain kind below is paired with a
# value proven valid by construction (an interval/bounded endpoint, one of a
# finite set's own members, the identity permutation, or the single
# admissible value) rather than a value drawn from a wider, possibly-invalid
# superset.


@st.composite
def draw_interval_integer_param_with_valid_value(
    draw: st.DrawFn, limit: int = 25
) -> tuple[Param[int], int]:
    """Draw an interval-integer param with one integer inside its interval."""
    lower, upper = draw(draw_ordered_optional_bounds(limit))
    param = build_interval_integer_param(lower, upper)
    if lower is not None:
        value = lower
    elif upper is not None:
        value = upper
    else:
        value = 0
    return param, value


@st.composite
def draw_bounded_integer_param_with_valid_value(
    draw: st.DrawFn, limit: int = 25
) -> tuple[Param[int], int]:
    """Draw a bounded integer param with its (always-valid) lower bound."""
    lower = draw(st.integers(min_value=-limit, max_value=limit))
    extra = draw(st.integers(min_value=0, max_value=2 * limit))
    upper = min(lower + extra, limit)
    return create_integer_param_between(lower, upper), lower


@st.composite
def draw_natural_param_with_valid_value(draw: st.DrawFn) -> tuple[Param[int], int]:
    """Draw a natural-number param with 1, valid whether or not zero is included."""
    zero_included = draw(st.booleans())
    return create_natural_param(zero_included=zero_included), 1


@st.composite
def draw_ordinal_param_with_valid_value(draw: st.DrawFn) -> tuple[Param[int], int]:
    """Draw an ordinal param with one of its own admissible values."""
    values = draw(build_ordinal_value_set_strategy())
    chosen: int = draw(st.sampled_from(values))
    return create_ordinal_param(values), chosen


@st.composite
def draw_categorical_param_with_valid_value(draw: st.DrawFn) -> tuple[Param[str], str]:
    """Draw a categorical param with one of its own categories."""
    categories = draw(build_categorical_value_set_strategy())
    chosen: str = draw(st.sampled_from(categories))
    return create_categorical_param(categories), chosen


@st.composite
def draw_permutation_param_with_valid_value(
    draw: st.DrawFn,
) -> tuple[Param[tuple[str, ...]], tuple[str, ...]]:
    """Draw a permutation param with the identity ordering of its own members."""
    members = draw(build_permutation_member_set_strategy())
    return create_permutation_param(members), tuple(members)


@st.composite
def draw_bounded_real_param_with_valid_value(
    draw: st.DrawFn, limit: int = 25
) -> tuple[Param[str | float], float]:
    """Draw a bounded real param with its (always-valid) lower bound."""
    lower = draw(st.integers(min_value=-limit, max_value=limit))
    extra = draw(st.integers(min_value=0, max_value=2 * limit))
    upper = lower + extra
    return create_real_param_between(float(lower), float(upper)), float(lower)


@st.composite
def draw_single_valid_value_param_with_valid_value(
    draw: st.DrawFn,
) -> tuple[Param[str], str]:
    """Draw a single-valid-value param with its one admissible value."""
    value = draw(st.sampled_from(_CATEGORICAL_ALPHABET))
    return create_single_valid_value_param(value), value


@st.composite
def draw_param_with_valid_value(draw: st.DrawFn) -> tuple[Param[Any], Any]:
    """Draw a param, over any domain kind, paired with a value valid for it."""
    result: tuple[Param[Any], Any] = draw(
        st.one_of(
            draw_interval_integer_param_with_valid_value(),
            draw_bounded_integer_param_with_valid_value(),
            draw_natural_param_with_valid_value(),
            draw_ordinal_param_with_valid_value(),
            draw_categorical_param_with_valid_value(),
            draw_permutation_param_with_valid_value(),
            draw_bounded_real_param_with_valid_value(),
            draw_single_valid_value_param_with_valid_value(),
        )
    )
    return result


@st.composite
def draw_param_assignment(draw: st.DrawFn) -> ParamAssignment[Any]:
    """Draw a ``ParamAssignment`` by assigning a guaranteed-valid value to a param."""
    param, value = draw(draw_param_with_valid_value())
    return param.assign(value)


_WELD_ASSIGNMENT_PARAM = create_integer_param(name=mock_identifier("p", 2))
_WELD_ASSIGNMENT = ParamAssignment(_WELD_ASSIGNMENT_PARAM, 5)
"""The hand-built assignment the pre-generalization version of this test used."""


@pytest.mark.parametrize("duplicate", _DUPLICATORS)
@example(assignment=_WELD_ASSIGNMENT)
@given(assignment=draw_param_assignment())
def test_param_assignment_survives_duplication(
    assignment: ParamAssignment[Any], duplicate: Duplicator
) -> None:
    """Test an assignment duplicates frozen, keeping its parameter and value.

    Oracle: the deleted ``tests/symbolic/test_pickle_round_trips.py``
    assertions, generalized from one integer assignment to every domain kind
    the shared strategies draw.
    """
    duplicated = duplicate(assignment)

    assert duplicated is not assignment
    assert duplicated.is_frozen
    assert duplicated.value == assignment.value
    assert duplicated.param.is_structurally_equivalent(assignment.param)
    assert duplicated.is_structurally_equivalent(assignment)
    assert duplicated.is_alpha_equivalent(assignment)
