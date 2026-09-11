"""Hypothesis property tests for the annotation-introspection helpers.

Draws from a small, fixed set of annotations (`int`, `str`, `list[int]`,
`dict[str, int]`, `int | str`, `Union[int, str]`, `Optional[int]`, `None`,
plus a couple of `Annotated`/`Final` wrappers for the idempotence check) and
checks four helpers against the standard library's own `typing`
introspection, which is independent of `fhy_core`'s implementation:

- `split_optional` recovers the wrapped type and `True` for optional
  annotations, and returns `(annotation, False)` unchanged otherwise.
- `get_union_members` agrees with `typing.get_args` on unions, `None`
  otherwise.
- `unwrap_annotation` is idempotent.
- `get_origin_and_arguments` agrees with `typing.get_origin`/`get_args` for
  container generics, `None` for unions, `Annotated`, `Final`, and plain
  types.
"""

from types import NoneType, UnionType
from typing import Annotated, Any, Final, Optional, Union, get_args, get_origin

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given
from hypothesis import strategies as st

from fhy_core.utils.type_hint_utils import (
    get_origin_and_arguments,
    get_union_members,
    split_optional,
    unwrap_annotation,
)

pytestmark = pytest.mark.property

_BASE_ANNOTATIONS: tuple[Any, ...] = (int, str, list[int], dict[str, int])
_UNION_ANNOTATIONS: tuple[Any, ...] = (int | str, Union[int, str])  # noqa: UP007
# `Optional[T]` is deliberately spelled the `typing` way (not `T | None`) in
# a couple of entries so the property exercises both optional spellings.
_OPTIONAL_ANNOTATIONS: tuple[tuple[Any, Any], ...] = (
    (Optional[int], int),  # noqa: UP045
    (int | None, int),
    (Optional[str], str),  # noqa: UP045
)
_NON_OPTIONAL_ANNOTATIONS: tuple[Any, ...] = (
    *_BASE_ANNOTATIONS,
    *_UNION_ANNOTATIONS,
    NoneType,
)
_ALL_FIXED_ANNOTATIONS: tuple[Any, ...] = (
    *_BASE_ANNOTATIONS,
    *_UNION_ANNOTATIONS,
    Optional[int],  # noqa: UP045
    NoneType,
)
_UNWRAP_ANNOTATIONS: tuple[Any, ...] = (
    *_ALL_FIXED_ANNOTATIONS,
    Annotated[int, "meta"],
    Final[int],
    Annotated[Final[int], "meta"],
)


@given(pair=st.sampled_from(_OPTIONAL_ANNOTATIONS))
def test_split_optional_recovers_wrapped_type_for_optional_annotations(
    pair: tuple[Any, Any],
) -> None:
    """Test split_optional(Optional[T]) == (T, True) for a fixed set of optionals.

    Oracle: a hand-computed (inner, True) pair per optional annotation,
    independent of split_optional's own union-peeling logic.
    """
    annotation, expected_inner = pair
    assert split_optional(annotation) == (expected_inner, True)


@given(annotation=st.sampled_from(_NON_OPTIONAL_ANNOTATIONS))
def test_split_optional_returns_unchanged_for_non_optional_annotations(
    annotation: Any,
) -> None:
    """Test split_optional(T) == (T, False) for a fixed set of non-optional annotations.

    Oracle: the annotation is passed through unchanged when it is not one of
    the drawn optional annotations.
    """
    assert split_optional(annotation) == (annotation, False)


@given(annotation=st.sampled_from(_ALL_FIXED_ANNOTATIONS))
def test_get_union_members_agrees_with_typing_get_args_on_unions(
    annotation: Any,
) -> None:
    """Test get_union_members matches typing.get_args on unions, None otherwise.

    Oracle: typing.get_origin/typing.get_args, the standard library's own
    union recognition, independent of get_union_members's implementation.
    """
    origin = get_origin(annotation)
    if origin is Union or origin is UnionType:
        assert get_union_members(annotation) == get_args(annotation)
    else:
        assert get_union_members(annotation) is None


@given(annotation=st.sampled_from(_UNWRAP_ANNOTATIONS))
def test_unwrap_annotation_is_idempotent(annotation: Any) -> None:
    """Test unwrap_annotation(unwrap_annotation(x)) == unwrap_annotation(x).

    Oracle: idempotence is checked directly by comparing a second
    application's result to the first, independent of how many layers the
    first application peeled.
    """
    once = unwrap_annotation(annotation)
    twice = unwrap_annotation(once)
    assert twice == once


@given(annotation=st.sampled_from(_ALL_FIXED_ANNOTATIONS))
def test_get_origin_and_arguments_agrees_with_typing_for_fixed_annotations(
    annotation: Any,
) -> None:
    """Test get_origin_and_arguments matches typing.get_origin/get_args, or None.

    Oracle: typing.get_origin/typing.get_args directly, applying the same
    union/Annotated/Final exclusion get_origin_and_arguments documents.
    """
    origin = get_origin(annotation)
    if origin is None or origin in (Union, UnionType, Annotated, Final):
        assert get_origin_and_arguments(annotation) is None
    else:
        assert get_origin_and_arguments(annotation) == (origin, get_args(annotation))
