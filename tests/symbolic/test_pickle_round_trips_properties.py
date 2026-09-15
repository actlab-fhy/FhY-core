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
keeps the object under test itself on the real ``dumps``/``loads`` path.
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
    draw_param_over_any_domain,
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
# Constraints: duplication stays frozen and equivalent
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


_PINNED_CONSTRAINT_VARIABLE = mock_identifier("x", 0)
_PINNED_CONSTRAINT = EquationConstraint(
    IdentifierExpression(_PINNED_CONSTRAINT_VARIABLE) < LiteralExpression(5)
)
"""A hand-picked equation constraint pinned as an example."""


@pytest.mark.parametrize("duplicate", _DUPLICATORS)
@example(constraint=_PINNED_CONSTRAINT)
@given(constraint=draw_constraint_over_one_identifier())
def test_constraint_survives_duplication(
    constraint: Constraint, duplicate: Duplicator
) -> None:
    """Test a constraint reaching an identifier duplicates frozen and equivalent.

    Oracle: the constraint itself, compared field-by-field with its
    duplicate across free identifiers, structural equivalence, and alpha
    equivalence.
    """
    duplicated = duplicate(constraint)

    assert duplicated is not constraint
    assert duplicated.is_frozen
    assert duplicated.get_free_identifiers() == constraint.get_free_identifiers()
    assert duplicated.is_structurally_equivalent(constraint)
    assert duplicated.is_alpha_equivalent(constraint)


# =============================================================================
# Params: duplication stays frozen and equivalent
# =============================================================================


_CATEGORICAL_ALPHABET: Final = tuple("abcdefgh")


_PINNED_PARAM = create_integer_param(name=mock_identifier("p", 1))
"""A hand-picked integer parameter pinned as an example."""


@pytest.mark.parametrize("duplicate", _DUPLICATORS)
@example(param=_PINNED_PARAM)
@given(param=draw_param_over_any_domain())
def test_param_survives_duplication(param: Param[Any], duplicate: Duplicator) -> None:
    """Test a parameter duplicates frozen, with its binder and domain intact.

    Oracle: the parameter itself, compared field-by-field with its
    duplicate across its binder, domain, structural equivalence, and alpha
    equivalence.
    """
    duplicated = duplicate(param)

    assert duplicated is not param
    assert duplicated.is_frozen
    assert duplicated.variable == param.variable
    assert duplicated.domain.is_structurally_equivalent(param.domain)
    assert duplicated.is_structurally_equivalent(param)
    assert duplicated.is_alpha_equivalent(param)


# =============================================================================
# Param assignments: duplication stays frozen and equivalent
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


_PINNED_ASSIGNMENT_PARAM = create_integer_param(name=mock_identifier("p", 2))
_PINNED_ASSIGNMENT = ParamAssignment(_PINNED_ASSIGNMENT_PARAM, 5)
"""A hand-picked integer assignment pinned as an example."""


@pytest.mark.parametrize("duplicate", _DUPLICATORS)
@example(assignment=_PINNED_ASSIGNMENT)
@given(assignment=draw_param_assignment())
def test_param_assignment_survives_duplication(
    assignment: ParamAssignment[Any], duplicate: Duplicator
) -> None:
    """Test an assignment duplicates frozen, keeping its parameter and value.

    Oracle: the assignment itself, compared field-by-field with its
    duplicate across its value, parameter, structural equivalence, and
    alpha equivalence.
    """
    duplicated = duplicate(assignment)

    assert duplicated is not assignment
    assert duplicated.is_frozen
    assert duplicated.value == assignment.value
    assert duplicated.param.is_structurally_equivalent(assignment.param)
    assert duplicated.is_structurally_equivalent(assignment)
    assert duplicated.is_alpha_equivalent(assignment)
