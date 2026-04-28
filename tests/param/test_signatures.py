"""Tests pinning down keyword-only signatures across the public param API.

Each public constructor takes its ``name`` (or other post-``*`` parameter) as
keyword-only. Mutating the keyword-only marker to a positional-only marker is a
common mutation that survives generic value-based tests; this file rejects it
by asserting that positional misuse raises ``TypeError`` while keyword use
succeeds.
"""

from typing import Any, Callable

import pytest
from _pytest.mark.structures import ParameterSet

from fhy_core.identifier import Identifier
from fhy_core.param import (
    CategoricalParam,
    IntParam,
    OrdinalParam,
    PermParam,
    RealParam,
    create_single_valid_value_param,
)

# =============================================================================
# Constructors taking ``name`` as a keyword-only argument
# =============================================================================


_KEYWORD_ONLY_NAME_CALLABLES: list[ParameterSet,] = [
    pytest.param(IntParam, (), id="int-param-init"),
    pytest.param(RealParam, (), id="real-param-init"),
    pytest.param(RealParam.with_value, (1.0,), id="real-with-value"),
    pytest.param(IntParam.with_value, (1,), id="int-with-value"),
    pytest.param(RealParam.between, (1.0, 2.0), id="real-between"),
    pytest.param(IntParam.between, (1, 2), id="int-between"),
    pytest.param(RealParam.with_lower_bound, (1.0,), id="real-with-lower-bound"),
    pytest.param(RealParam.with_upper_bound, (2.0,), id="real-with-upper-bound"),
    pytest.param(IntParam.with_lower_bound, (1,), id="int-with-lower-bound"),
    pytest.param(IntParam.with_upper_bound, (2,), id="int-with-upper-bound"),
    pytest.param(
        lambda *args, **kwargs: OrdinalParam([1, 2, 3], *args, **kwargs),
        (),
        id="ordinal-init",
    ),
    pytest.param(
        lambda *args, **kwargs: CategoricalParam({"a", "b"}, *args, **kwargs),
        (),
        id="categorical-init",
    ),
    pytest.param(
        lambda *args, **kwargs: PermParam(["n", "c"], *args, **kwargs),
        (),
        id="perm-init",
    ),
    pytest.param(create_single_valid_value_param, ("only",), id="single-valid-value"),
]


@pytest.mark.parametrize("callable_, positional_args", _KEYWORD_ONLY_NAME_CALLABLES)
def test_callable_accepts_name_as_keyword(
    callable_: Callable[..., Any], positional_args: tuple[Any, ...]
) -> None:
    """Test the public callable accepts ``name`` as a keyword argument."""
    callable_(*positional_args, name=Identifier("x"))


@pytest.mark.parametrize("callable_, positional_args", _KEYWORD_ONLY_NAME_CALLABLES)
def test_callable_rejects_name_passed_positionally(
    callable_: Callable[..., Any], positional_args: tuple[Any, ...]
) -> None:
    """Test the public callable rejects ``name`` passed positionally."""
    with pytest.raises(TypeError):
        callable_(*positional_args, Identifier("x"))


# =============================================================================
# Constructors taking ``is_inclusive`` (or its lower/upper variants) as kw-only
# =============================================================================


_KEYWORD_ONLY_IS_INCLUSIVE_CALLABLES: list[ParameterSet] = [
    pytest.param(RealParam.with_lower_bound, (1.0,), id="real-with-lower-bound"),
    pytest.param(RealParam.with_upper_bound, (2.0,), id="real-with-upper-bound"),
    pytest.param(IntParam.with_lower_bound, (1,), id="int-with-lower-bound"),
    pytest.param(IntParam.with_upper_bound, (2,), id="int-with-upper-bound"),
    pytest.param(
        RealParam().add_lower_bound_constraint,
        (1.0,),
        id="real-add-lower-bound-constraint",
    ),
    pytest.param(
        RealParam().add_upper_bound_constraint,
        (2.0,),
        id="real-add-upper-bound-constraint",
    ),
    pytest.param(
        IntParam().add_lower_bound_constraint,
        (1,),
        id="int-add-lower-bound-constraint",
    ),
    pytest.param(
        IntParam().add_upper_bound_constraint,
        (2,),
        id="int-add-upper-bound-constraint",
    ),
]


@pytest.mark.parametrize(
    "callable_, positional_args", _KEYWORD_ONLY_IS_INCLUSIVE_CALLABLES
)
def test_callable_rejects_is_inclusive_passed_positionally(
    callable_: Callable[..., Any], positional_args: tuple[Any, ...]
) -> None:
    """Test the public callable rejects ``is_inclusive`` passed positionally."""
    with pytest.raises(TypeError):
        callable_(*positional_args, True)


@pytest.mark.parametrize(
    "callable_, positional_args",
    [
        pytest.param(RealParam.between, (1.0, 2.0), id="real-between"),
        pytest.param(IntParam.between, (1, 2), id="int-between"),
    ],
)
def test_between_rejects_is_lower_inclusive_passed_positionally(
    callable_: Callable[..., Any], positional_args: tuple[Any, ...]
) -> None:
    """Test ``between`` rejects ``is_lower_inclusive`` passed positionally."""
    with pytest.raises(TypeError):
        callable_(*positional_args, True)
