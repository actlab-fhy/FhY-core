"""Hypothesis property tests for `is_strict_int`.

`is_strict_int`'s documented rule is "a Python `int` whose runtime type is
not `bool`". The property below checks that rule against a fixed per-kind
expectation table built independently of `is_strict_int`'s own
`isinstance` logic, spanning plain ints, bools, floats, and NumPy scalar
types.

NumPy ints are not ambiguous under this NumPy version: `numpy>=2.0` fully
decouples its scalar types from the Python `int`/`bool` hierarchy (verified
empirically: `isinstance(np.int64(0), int)` and
`isinstance(np.bool_(True), (int, bool))` are both `False`), so
`is_strict_int` correctly rejects every NumPy scalar tested here.
"""

import pytest

pytest.importorskip("hypothesis")

from hypothesis import example, given
from hypothesis import strategies as st

from fhy_core.utils.numeric_utils import is_strict_int

pytestmark = pytest.mark.property

np = pytest.importorskip("numpy")

_KIND_PYTHON_INT = "python_int"
_KIND_PYTHON_BOOL = "python_bool"
_KIND_FLOAT = "float"
_KIND_NUMPY_INT64 = "numpy_int64"
_KIND_NUMPY_BOOL = "numpy_bool"

# Built from is_strict_int's docstring plus the empirical NumPy fact above,
# independent of is_strict_int's own isinstance(value, int) expression.
_EXPECTED_BY_KIND: dict[str, bool] = {
    _KIND_PYTHON_INT: True,
    _KIND_PYTHON_BOOL: False,
    _KIND_FLOAT: False,
    _KIND_NUMPY_INT64: False,
    _KIND_NUMPY_BOOL: False,
}

_INT64_BOUND = 2**63 - 1


@st.composite
def draw_kinded_value(draw: st.DrawFn) -> tuple[str, object]:
    """Draw a (kind, value) pair spanning int, bool, float, and NumPy scalars.

    The kind is drawn first and determines both the value's runtime type and
    the independent expectation looked up in `_EXPECTED_BY_KIND`.
    """
    kind = draw(st.sampled_from(list(_EXPECTED_BY_KIND)))
    value: object
    if kind == _KIND_PYTHON_INT:
        value = draw(st.integers(min_value=-_INT64_BOUND, max_value=_INT64_BOUND))
    elif kind == _KIND_PYTHON_BOOL:
        value = draw(st.booleans())
    elif kind == _KIND_FLOAT:
        value = draw(st.floats(allow_nan=False, allow_infinity=False, width=64))
    elif kind == _KIND_NUMPY_INT64:
        value = np.int64(
            draw(st.integers(min_value=-_INT64_BOUND, max_value=_INT64_BOUND))
        )
    else:
        value = np.bool_(draw(st.booleans()))
    return kind, value


@example(kinded_value=(_KIND_PYTHON_INT, 0))
@example(kinded_value=(_KIND_PYTHON_INT, 2**62))
@example(kinded_value=(_KIND_PYTHON_INT, -(2**62)))
@example(kinded_value=(_KIND_PYTHON_BOOL, True))
@example(kinded_value=(_KIND_PYTHON_BOOL, False))
@given(kinded_value=draw_kinded_value())
def test_is_strict_int_matches_type_based_expectation(
    kinded_value: tuple[str, object],
) -> None:
    """Test is_strict_int agrees with an independent per-kind expectation table.

    Oracle: `_EXPECTED_BY_KIND`, built from the docstring's rule plus the
    empirical NumPy scalar-type fact documented at module level.
    """
    kind, value = kinded_value
    assert is_strict_int(value) is _EXPECTED_BY_KIND[kind]


@given(
    value=st.one_of(
        st.none(),
        st.text(),
        st.binary(),
        st.lists(st.integers()),
        st.tuples(st.integers()),
    )
)
def test_is_strict_int_rejects_non_numeric_types(value: object) -> None:
    """Test is_strict_int is False for values that are never int instances.

    Oracle: none of None, str, bytes, list, or tuple are instances of int,
    independent of is_strict_int's own implementation.
    """
    assert is_strict_int(value) is False
