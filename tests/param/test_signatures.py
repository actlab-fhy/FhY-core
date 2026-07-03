"""Tests pinning down keyword-only signatures across the public param API.

Each public factory takes its ``name`` (or other post-``*`` parameter) as
keyword-only. Mutating the keyword-only marker to a positional-only marker is a
common mutation that survives generic value-based tests; this file rejects it
by asserting that positional misuse raises ``TypeError`` while keyword use
succeeds.
"""

from collections.abc import Callable
from typing import Any

import pytest
from _pytest.mark.structures import ParameterSet

from fhy_core.identifier import Identifier
from fhy_core.param import (
    create_categorical_param,
    create_integer_param,
    create_integer_param_between,
    create_integer_param_with_lower_bound,
    create_integer_param_with_upper_bound,
    create_ordinal_param,
    create_permutation_param,
    create_real_param,
    create_real_param_between,
    create_real_param_with_lower_bound,
    create_real_param_with_upper_bound,
    create_single_valid_value_param,
)

# =============================================================================
# Constructors taking ``name`` as a keyword-only argument
# =============================================================================


_KEYWORD_ONLY_NAME_CALLABLES: list[ParameterSet,] = [
    pytest.param(create_integer_param, (), id="int-param-init"),
    pytest.param(create_real_param, (), id="real-param-init"),
    pytest.param(create_real_param_between, (1.0, 2.0), id="real-between"),
    pytest.param(create_integer_param_between, (1, 2), id="int-between"),
    pytest.param(
        create_real_param_with_lower_bound, (1.0,), id="real-with-lower-bound"
    ),
    pytest.param(
        create_real_param_with_upper_bound, (2.0,), id="real-with-upper-bound"
    ),
    pytest.param(
        create_integer_param_with_lower_bound, (1,), id="int-with-lower-bound"
    ),
    pytest.param(
        create_integer_param_with_upper_bound, (2,), id="int-with-upper-bound"
    ),
    pytest.param(
        lambda *args, **kwargs: create_ordinal_param([1, 2, 3], *args, **kwargs),
        (),
        id="ordinal-init",
    ),
    pytest.param(
        lambda *args, **kwargs: create_categorical_param({"a", "b"}, *args, **kwargs),
        (),
        id="categorical-init",
    ),
    pytest.param(
        lambda *args, **kwargs: create_permutation_param(["n", "c"], *args, **kwargs),
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
    with pytest.raises(TypeError, match="positional argument"):
        callable_(*positional_args, Identifier("x"))


# =============================================================================
# Constructors taking ``is_inclusive`` (or its lower/upper variants) as kw-only
# =============================================================================


_KEYWORD_ONLY_IS_INCLUSIVE_CALLABLES: list[ParameterSet] = [
    pytest.param(
        create_real_param_with_lower_bound, (1.0,), id="real-with-lower-bound"
    ),
    pytest.param(
        create_real_param_with_upper_bound, (2.0,), id="real-with-upper-bound"
    ),
    pytest.param(
        create_integer_param_with_lower_bound, (1,), id="int-with-lower-bound"
    ),
    pytest.param(
        create_integer_param_with_upper_bound, (2,), id="int-with-upper-bound"
    ),
    pytest.param(
        create_real_param().add_lower_bound_constraint,
        (1.0,),
        id="real-add-lower-bound-constraint",
    ),
    pytest.param(
        create_real_param().add_upper_bound_constraint,
        (2.0,),
        id="real-add-upper-bound-constraint",
    ),
    pytest.param(
        create_integer_param().add_lower_bound_constraint,
        (1,),
        id="int-add-lower-bound-constraint",
    ),
    pytest.param(
        create_integer_param().add_upper_bound_constraint,
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
    with pytest.raises(TypeError, match="positional argument"):
        callable_(*positional_args, True)


@pytest.mark.parametrize(
    "callable_, positional_args",
    [
        pytest.param(create_real_param_between, (1.0, 2.0), id="real-between"),
        pytest.param(create_integer_param_between, (1, 2), id="int-between"),
    ],
)
def test_between_rejects_is_lower_inclusive_passed_positionally(
    callable_: Callable[..., Any], positional_args: tuple[Any, ...]
) -> None:
    """Test ``between`` rejects ``is_lower_inclusive`` passed positionally."""
    with pytest.raises(TypeError, match="positional argument"):
        callable_(*positional_args, True)
