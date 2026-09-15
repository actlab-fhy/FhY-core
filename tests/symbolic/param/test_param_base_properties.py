"""Hypothesis property tests for `Param.assign` and factory construction.

Covers two invariants stated in ``Param``'s own module and method
docstrings: ``assign`` succeeds exactly on an admissible, constraint-
satisfying value, and each ``create_*`` factory is equivalent to building
its ``Param`` directly from the same domain and variable.
"""

from typing import Any

import pytest

pytest.importorskip("hypothesis")

from hypothesis import example, given
from hypothesis import strategies as st

from fhy_core.identifier import Identifier
from fhy_core.symbolic.param import (
    CategoricalDomain,
    IntegerDomain,
    OrdinalDomain,
    Param,
    ParamError,
    create_categorical_param,
    create_integer_param,
    create_integer_param_between,
    create_natural_param,
    create_ordinal_param,
)
from fhy_core.symbolic.param.domains import (
    build_categorical_domain,
    build_ordinal_domain,
)

from ...strategies.identifiers import build_identifier_pool, build_identifier_strategy
from ...strategies.params import (
    build_categorical_value_set_strategy,
    build_ordinal_value_set_strategy,
    draw_param_with_candidate,
)
from .conftest import mock_identifier

pytestmark = pytest.mark.property

# Every property runs without a hypothesis deadline. Every example takes
# about a millisecond and no draw is filtered, so a deadline here would time
# only the scheduler: on a contended machine one example can be descheduled
# for hundreds of milliseconds, failing a law for a reason unrelated to it.
# The `dev`/`thorough` profiles already set `deadline=None`.

_IDENTIFIER_POOL = build_identifier_pool(5)


# =============================================================================
# Property: assign succeeds exactly when the value is valid
# =============================================================================


@given(case=draw_param_with_candidate())
def test_assign_succeeds_exactly_when_the_value_is_valid(
    case: tuple[Param[Any], Any],
) -> None:
    """Test ``assign(v)`` returns ``v`` iff ``is_value_valid(v)``, else raises.

    ``Param.assign`` documents ``ParamError`` for an inadmissible value,
    a constraint-violating one, or one a constraint cannot verify --
    every way ``is_value_valid`` can be ``False`` for a value drawn by
    these strategies, none of which uses dependent bindings or a
    Boolean-position numeric operand that would raise a different type.
    """
    param, candidate = case

    if param.is_value_valid(candidate):
        assignment = param.assign(candidate)
        assert assignment.value == candidate
    else:
        with pytest.raises(ParamError):
            param.assign(candidate)


# =============================================================================
# Property: factories match direct construction from the same domain
# =============================================================================


@example(identifier=mock_identifier("x", 1))
@given(identifier=build_identifier_strategy(_IDENTIFIER_POOL))
def test_plain_integer_param_factory_matches_direct_construction(
    identifier: Identifier,
) -> None:
    """Test ``create_integer_param`` matches direct ``IntegerDomain`` construction."""
    via_factory = create_integer_param(name=identifier)
    direct: Param[int] = Param(IntegerDomain(), variable=identifier)

    assert direct.is_structurally_equivalent(via_factory)
    assert direct.is_value_admissible(5)
    assert not direct.is_value_admissible("not an int")


@st.composite
def draw_bounded_integer_factory_args(
    draw: st.DrawFn,
) -> tuple[Identifier, int, int, bool, bool]:
    """Draw arguments for ``create_integer_param_between`` plus its identifier.

    ``width`` is forced to ``0`` only alongside inclusive bounds: the
    factory itself raises ``ParamError`` for an empty exclusive-at-a-
    point range, so that combination is never drawn rather than filtered.
    """
    identifier = draw(build_identifier_strategy(_IDENTIFIER_POOL))
    lower = draw(st.integers(min_value=-25, max_value=25))
    width = draw(st.integers(min_value=0, max_value=25))
    upper = lower + width
    if width == 0:
        is_lower_inclusive = True
        is_upper_inclusive = True
    else:
        is_lower_inclusive = draw(st.booleans())
        is_upper_inclusive = draw(st.booleans())
    return identifier, lower, upper, is_lower_inclusive, is_upper_inclusive


@given(args=draw_bounded_integer_factory_args())
def test_bounded_integer_param_factory_matches_direct_construction(
    args: tuple[Identifier, int, int, bool, bool],
) -> None:
    """Test ``create_integer_param_between`` matches direct construction.

    Direct construction chains ``Param(IntegerDomain(), ...)`` with the
    same ``add_lower_bound_constraint``/``add_upper_bound_constraint``
    calls the factory itself makes.
    """
    identifier, lower, upper, is_lower_inclusive, is_upper_inclusive = args

    via_factory = create_integer_param_between(
        lower,
        upper,
        name=identifier,
        is_lower_inclusive=is_lower_inclusive,
        is_upper_inclusive=is_upper_inclusive,
    )
    direct: Param[int] = Param(IntegerDomain(), variable=identifier)
    direct = direct.add_lower_bound_constraint(lower, is_inclusive=is_lower_inclusive)
    direct = direct.add_upper_bound_constraint(upper, is_inclusive=is_upper_inclusive)

    assert direct.is_structurally_equivalent(via_factory)


@given(
    identifier=build_identifier_strategy(_IDENTIFIER_POOL), zero_included=st.booleans()
)
def test_natural_param_factory_matches_direct_construction(
    identifier: Identifier, zero_included: bool
) -> None:
    """Test ``create_natural_param`` matches direct construction.

    Direct construction needs no explicit constraint: the non-negative
    bound is ``IntegerDomain``'s own implied constraint, added by
    ``Param.__post_init__`` the same way for both paths.
    """
    via_factory = create_natural_param(name=identifier, zero_included=zero_included)
    direct: Param[int] = Param(
        IntegerDomain(non_negative=True, zero_included=zero_included),
        variable=identifier,
    )

    assert direct.is_structurally_equivalent(via_factory)


@given(
    identifier=build_identifier_strategy(_IDENTIFIER_POOL),
    values=build_ordinal_value_set_strategy(),
)
def test_ordinal_param_factory_matches_direct_construction(
    identifier: Identifier, values: list[int]
) -> None:
    """Test ``create_ordinal_param`` matches direct ``build_ordinal_domain`` use."""
    via_factory = create_ordinal_param(values, name=identifier)
    direct: Param[int] = Param(build_ordinal_domain(values), variable=identifier)

    assert direct.is_structurally_equivalent(via_factory)
    assert isinstance(direct.domain, OrdinalDomain)


@given(
    identifier=build_identifier_strategy(_IDENTIFIER_POOL),
    categories=build_categorical_value_set_strategy(),
)
def test_categorical_param_factory_matches_direct_construction(
    identifier: Identifier, categories: list[str]
) -> None:
    """Test ``create_categorical_param`` matches direct construction.

    Direct construction builds ``build_categorical_domain`` from the
    same categories, tupled the way the factory itself tuples them.
    """
    via_factory = create_categorical_param(categories, name=identifier)
    direct: Param[str] = Param(
        build_categorical_domain(tuple(categories)), variable=identifier
    )

    assert direct.is_structurally_equivalent(via_factory)
    assert isinstance(direct.domain, CategoricalDomain)
