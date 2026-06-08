"""Tests for the shared type-hint introspection utilities.

These helpers were extracted from ``FrozenMixin``'s internals into
``fhy_core.utils.type_hint_utils`` so the frozen trait and the serialization
derivation engine can share one annotation-introspection implementation.
"""

from typing import Annotated, Final, Optional, Union

import pytest

from fhy_core.utils.type_hint_utils import (
    get_field_names,
    get_origin_and_arguments,
    get_union_members,
    resolve_annotation,
    resolve_field_annotations,
    split_optional,
    unwrap_annotation,
)

_NONE_TYPE = type(None)


# Module-level classes so string/forward-reference annotations resolve through
# this module's globals, the way they do for a real class under test.


class _Leaf:
    pass


class _Base:
    a: int
    b: str


class _Derived(_Base):
    b: int  # type: ignore[assignment]  # redeclared; most-derived wins
    c: "_Leaf"  # forward reference, resolvable through module globals


class _HasUnresolvable:
    ok: int
    # Intentionally references an undefined name to exercise the per-field
    # resolution fallback.
    bad: "_NameThatDoesNotExistAnywhere"  # type: ignore[name-defined]  # noqa: F821


class _NoAnnotations:
    pass


# ============================================================================
# resolve_annotation
# ============================================================================


def test_resolve_annotation_returns_concrete_type_unchanged() -> None:
    """Test an already-resolved type is returned as itself."""
    assert resolve_annotation(int) is int


def test_resolve_annotation_resolves_builtin_name_string() -> None:
    """Test a string naming a builtin resolves to that builtin type."""
    assert resolve_annotation("int") is int


def test_resolve_annotation_resolves_forward_reference_via_localns() -> None:
    """Test a forward-reference string resolves through the local namespace."""
    assert resolve_annotation("_Leaf", localns={"_Leaf": _Leaf}) is _Leaf


def test_resolve_annotation_resolves_optional_forward_reference() -> None:
    """Test an optional forward reference resolves to a union with ``None``."""
    resolved = resolve_annotation("_Leaf | None", localns={"_Leaf": _Leaf})

    assert get_union_members(resolved) == (_Leaf, _NONE_TYPE)


def test_resolve_annotation_preserves_annotated_extras() -> None:
    """Test ``Annotated`` metadata survives resolution."""
    resolved = resolve_annotation(
        'Annotated[int, "meta"]', globalns={"Annotated": Annotated}
    )

    assert resolved == Annotated[int, "meta"]


def test_resolve_annotation_raises_on_unresolvable_name() -> None:
    """Test resolving an unknown name raises ``NameError``."""
    with pytest.raises(NameError):
        resolve_annotation("_NameThatDoesNotExistAnywhere")


# ============================================================================
# resolve_field_annotations
# ============================================================================


def test_resolve_field_annotations_resolves_all_fields_including_inherited() -> None:
    """Test inherited and forward-referenced fields all resolve."""
    resolved = resolve_field_annotations(_Derived)

    assert resolved == {"a": int, "b": int, "c": _Leaf}


def test_resolve_field_annotations_skips_unresolvable_field() -> None:
    """Test one unresolvable field is omitted while the rest resolve."""
    resolved = resolve_field_annotations(_HasUnresolvable)

    assert resolved == {"ok": int}
    assert "bad" not in resolved


def test_resolve_field_annotations_returns_empty_for_no_annotations() -> None:
    """Test a class with no annotated fields resolves to an empty mapping."""
    assert resolve_field_annotations(_NoAnnotations) == {}


# ============================================================================
# get_field_names
# ============================================================================


def test_get_field_names_orders_base_before_derived_and_dedupes() -> None:
    """Test names come base-first in declaration order, de-duplicated."""
    assert get_field_names(_Derived) == ["a", "b", "c"]


def test_get_field_names_excludes_object() -> None:
    """Test ``object`` contributes no names."""
    assert get_field_names(_Base) == ["a", "b"]


def test_get_field_names_applies_predicate_filter() -> None:
    """Test only MRO classes passing the predicate contribute their fields."""
    assert get_field_names(_Derived, predicate=lambda klass: klass is _Base) == [
        "a",
        "b",
    ]


def test_get_field_names_empty_for_no_annotations() -> None:
    """Test a class with no annotations yields no names."""
    assert get_field_names(_NoAnnotations) == []


# ============================================================================
# unwrap_annotation
# ============================================================================


def test_unwrap_annotation_passes_plain_type_through() -> None:
    """Test a plain type is returned unchanged."""
    assert unwrap_annotation(int) is int


def test_unwrap_annotation_strips_annotated() -> None:
    """Test ``Annotated[T, ...]`` unwraps to ``T``."""
    assert unwrap_annotation(Annotated[int, "meta"]) is int


def test_unwrap_annotation_strips_final() -> None:
    """Test ``Final[T]`` unwraps to ``T``."""
    assert unwrap_annotation(Final[int]) is int


def test_unwrap_annotation_strips_nested_wrappers() -> None:
    """Test nested ``Annotated``/``Final`` layers all unwrap to the core type."""
    assert unwrap_annotation(Final[Annotated[int, "meta"]]) is int


def test_unwrap_annotation_leaves_union_untouched() -> None:
    """Test a union is not unwrapped (it is not an Annotated/Final layer)."""
    assert unwrap_annotation(int | str) == int | str


# ============================================================================
# get_union_members
# ============================================================================


def test_get_union_members_returns_pep604_members() -> None:
    """Test a PEP 604 union yields its member types."""
    assert get_union_members(int | str) == (int, str)


def test_get_union_members_returns_typing_union_members() -> None:
    """Test a ``typing.Union`` yields its member types."""
    assert get_union_members(Union[int, str]) == (int, str)


def test_get_union_members_includes_none_type_for_optional() -> None:
    """Test ``Optional[T]`` yields ``T`` and ``NoneType``."""
    assert get_union_members(Optional[int]) == (int, _NONE_TYPE)


def test_get_union_members_returns_none_for_non_union() -> None:
    """Test a non-union annotation yields ``None``."""
    assert get_union_members(int) is None


def test_get_union_members_returns_none_for_container() -> None:
    """Test a parameterized container is not treated as a union."""
    assert get_union_members(tuple[int, ...]) is None


# ============================================================================
# split_optional
# ============================================================================


def test_split_optional_unwraps_optional_single_member() -> None:
    """Test ``Optional[T]`` splits into ``(T, True)``."""
    assert split_optional(Optional[int]) == (int, True)


def test_split_optional_unwraps_pep604_optional() -> None:
    """Test ``T | None`` splits into ``(T, True)``."""
    assert split_optional(int | None) == (int, True)


def test_split_optional_keeps_remaining_union_members() -> None:
    """Test ``A | B | None`` splits into a union of the non-``None`` members."""
    inner, is_optional = split_optional(int | str | None)

    assert is_optional is True
    assert get_union_members(inner) == (int, str)


def test_split_optional_non_optional_returns_false() -> None:
    """Test a non-optional type is returned unchanged with ``False``."""
    assert split_optional(int) == (int, False)


def test_split_optional_non_optional_union_returns_false() -> None:
    """Test a union without ``None`` is not optional."""
    inner, is_optional = split_optional(int | str)

    assert is_optional is False
    assert get_union_members(inner) == (int, str)


# ============================================================================
# get_origin_and_arguments
# ============================================================================


def test_get_origin_and_arguments_for_homogeneous_tuple() -> None:
    """Test a variadic tuple yields its origin and element args."""
    assert get_origin_and_arguments(tuple[int, ...]) == (tuple, (int, ...))


def test_get_origin_and_arguments_for_frozenset() -> None:
    """Test a parameterized frozenset yields its origin and element arg."""
    assert get_origin_and_arguments(frozenset[str]) == (frozenset, (str,))


def test_get_origin_and_arguments_for_list() -> None:
    """Test a parameterized list yields its origin and element arg."""
    assert get_origin_and_arguments(list[int]) == (list, (int,))


def test_get_origin_and_arguments_returns_none_for_plain_type() -> None:
    """Test a non-parameterized type yields ``None``."""
    assert get_origin_and_arguments(int) is None


def test_get_origin_and_arguments_excludes_union() -> None:
    """Test a union is not reported as a container generic."""
    assert get_origin_and_arguments(int | str) is None


def test_get_origin_and_arguments_excludes_optional() -> None:
    """Test an optional is not reported as a container generic."""
    assert get_origin_and_arguments(Optional[int]) is None


def test_get_origin_and_arguments_excludes_annotated() -> None:
    """Test an ``Annotated`` form is not reported as a container generic."""
    assert get_origin_and_arguments(Annotated[int, "meta"]) is None


# ============================================================================
# Integration: classify a class's fields the way a consumer would
# ============================================================================


class _Representative:
    """Stands in for a real serializable dataclass under inspection."""

    name: int
    tag: str | None
    items: tuple[int, ...]
    child: "_Leaf"


def test_consumer_can_resolve_and_classify_every_field() -> None:
    """Test the helpers compose to classify each field of a real-looking class.

    Mirrors how the serialization engine walks a class: resolve the field
    annotations, then peel optionality and recognize containers per field.
    """
    resolved = resolve_field_annotations(_Representative)

    # Plain scalar.
    assert split_optional(resolved["name"]) == (int, False)
    assert get_origin_and_arguments(resolved["name"]) is None

    # Optional scalar.
    inner, is_optional = split_optional(resolved["tag"])
    assert (inner, is_optional) == (str, True)

    # Homogeneous container.
    container_inner, container_optional = split_optional(resolved["items"])
    assert container_optional is False
    assert get_origin_and_arguments(container_inner) == (tuple, (int, ...))

    # Nested object reference.
    assert resolved["child"] is _Leaf
    assert get_origin_and_arguments(resolved["child"]) is None
