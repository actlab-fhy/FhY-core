"""Tests for the version-compatible ``ReadOnly``/``TypedDict`` shim.

typeshed does not model PEP 705's ``__readonly_keys__`` and
``__mutable_keys__`` on ``TypedDict`` class objects, so the tests read
them through ``Any``.
"""

from typing import Any, cast

from fhy_core.utils.typed_dict import ReadOnly, TypedDict


class _ExampleTypedDict(TypedDict):
    """A ``TypedDict`` with one read-only field and one mutable field."""

    read_only_field: ReadOnly[int]
    mutable_field: int


def test_typed_dict_records_read_only_field_in_readonly_keys() -> None:
    """Test a ``ReadOnly``-annotated field is recorded as a read-only key."""
    assert "read_only_field" in cast(Any, _ExampleTypedDict).__readonly_keys__


def test_typed_dict_records_plain_field_in_mutable_keys() -> None:
    """Test a plain field is recorded as a mutable key."""
    assert "mutable_field" in cast(Any, _ExampleTypedDict).__mutable_keys__
