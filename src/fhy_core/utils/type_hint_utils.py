"""Type-hint and annotation introspection utilities.

Generic helpers for resolving and inspecting type annotations, extracted from
the internals of :class:`fhy_core.traits.frozen.FrozenMixin` so they can be
shared by any consumer that reasons about a class's field types. Two consumers
use them today: the frozen-field immutability check and the schema-derived
serialization engine.

Three concerns are covered:

- **Resolution** -- turning annotations (including string / forward-reference
  forms under ``from __future__ import annotations`` and PEP 749) into runtime
  type objects, tolerant of references that cannot yet be resolved.
- **Field discovery** -- listing the annotated field names contributed by a
  class and its bases, in declaration order.
- **Structure inspection** -- peeling ``Annotated`` / ``Final`` wrappers,
  splitting ``Optional`` / ``Union`` annotations, and recognizing
  parameterized container generics.
"""

__all__ = [
    "resolve_annotation",
    "resolve_field_annotations",
    "get_field_names",
    "unwrap_annotation",
    "get_union_members",
    "split_optional",
    "get_origin_and_arguments",
]

import inspect
import sys
from collections.abc import Callable
from types import NoneType, UnionType
from typing import (
    Annotated,
    Any,
    Final,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from fhy_core.logger import get_logger

_LOGGER = get_logger(__name__)


def resolve_annotation(
    annotation: Any,
    *,
    globalns: dict[str, Any] | None = None,
    localns: dict[str, Any] | None = None,
) -> Any:
    """Resolve a single annotation to a runtime type object.

    Resolves string and forward-reference annotations using the public
    :func:`typing.get_type_hints` API by wrapping the annotation on a throwaway
    probe class, so one annotation can be resolved in isolation without a
    sibling's unresolvable annotation failing the whole class.

    Args:
        annotation: The annotation to resolve. May be a type, a string, or a
            ``typing`` special form. Already-resolved types are returned
            unchanged.
        globalns: Global namespace used to resolve names. Defaults to ``None``
            (the caller's globals are not assumed).
        localns: Local namespace used to resolve names.

    Returns:
        The resolved annotation, with ``Annotated`` extras preserved.

    Raises:
        NameError: If the annotation references a name that cannot be resolved
            in the provided namespaces.
    """
    probe = type("_AnnotationProbe", (), {"__annotations__": {"_field": annotation}})
    return get_type_hints(
        probe, globalns=globalns, localns=localns, include_extras=True
    )["_field"]


def resolve_field_annotations(cls: type) -> dict[str, Any]:
    """Resolve a class's annotated field types, degrading per field.

    Attempts a single whole-class resolution via :func:`typing.get_type_hints`
    first. On failure (typically an unresolved forward reference), falls back to
    resolving each annotated field individually across the class's MRO so one
    unresolvable annotation does not suppress resolution of the rest.

    Args:
        cls: The class whose annotated fields to resolve.

    Returns:
        A mapping of field name to resolved type, including inherited fields and
        ``Annotated`` extras. Fields that cannot be resolved are omitted (and
        logged), so the caller can detect partial resolution by comparing the
        returned keys against :func:`get_field_names`.
    """
    try:
        return get_type_hints(cls, include_extras=True)
    except Exception as exc:  # noqa: BLE001 - resolution failure modes vary
        _LOGGER.warning(
            "annotation resolution for %s: whole-class resolution failed "
            "(%s: %s); falling back to per-field resolution",
            cls.__name__,
            type(exc).__name__,
            exc,
        )

    resolved: dict[str, Any] = {}
    for klass in reversed(cls.__mro__):
        if klass is object:
            continue
        module = sys.modules.get(klass.__module__)
        globalns = getattr(module, "__dict__", {})
        localns = dict(vars(cls))
        for name, raw in inspect.get_annotations(klass).items():
            try:
                resolved[name] = resolve_annotation(
                    raw, globalns=globalns, localns=localns
                )
            except Exception as exc:  # noqa: BLE001 - per-field failure modes vary
                _LOGGER.warning(
                    "annotation resolution for %s: could not resolve field %r "
                    "(%s: %s); skipping that field",
                    cls.__name__,
                    name,
                    type(exc).__name__,
                    exc,
                )
    return resolved


def get_field_names(
    cls: type,
    *,
    predicate: Callable[[type], bool] | None = None,
) -> list[str]:
    """Return the annotated field names contributed by a class and its bases.

    Walks the MRO in reverse (base-to-derived) so names appear in declaration
    order, de-duplicating names redeclared by a subclass. ``object`` is always
    excluded. Uses :func:`inspect.get_annotations` so the lookup works under
    PEP 749 (Python 3.14+) deferred annotations.

    Args:
        cls: The class whose annotated field names to collect.
        predicate: Optional filter applied to each MRO class; only classes for
            which it returns ``True`` contribute their annotations. ``None``
            (the default) includes every MRO class except ``object``.

    Returns:
        The ordered, de-duplicated list of annotated field names.
    """
    seen: dict[str, None] = {}
    for klass in reversed(cls.__mro__):
        if klass is object:
            continue
        if predicate is not None and not predicate(klass):
            continue
        for name in inspect.get_annotations(klass):
            seen.setdefault(name, None)
    return list(seen.keys())


def unwrap_annotation(annotation: Any) -> Any:
    """Strip ``Annotated`` and ``Final`` wrappers from an annotation.

    Repeatedly removes outer ``Annotated[T, ...]`` and ``Final[T]`` layers,
    returning the innermost wrapped annotation ``T``. Annotations that are not
    so wrapped are returned unchanged.

    Args:
        annotation: The annotation to unwrap.

    Returns:
        The annotation with any ``Annotated`` / ``Final`` layers removed.
    """
    origin = get_origin(annotation)
    while origin is Annotated or origin is Final:
        annotation = get_args(annotation)[0]
        origin = get_origin(annotation)
    return annotation


def get_union_members(annotation: Any) -> tuple[Any, ...] | None:
    """Return the member types of a union annotation, or ``None``.

    Recognizes both ``typing.Union[...]`` and PEP 604 ``X | Y`` unions.

    Args:
        annotation: The annotation to inspect.

    Returns:
        The tuple of union member annotations if ``annotation`` is a union,
        otherwise ``None``.
    """
    origin = get_origin(annotation)
    if origin is Union or origin is UnionType:
        return get_args(annotation)
    return None


def split_optional(annotation: Any) -> tuple[Any, bool]:
    """Split an optional annotation into its non-``None`` part and a flag.

    Treats an annotation as optional when it is a union that includes
    ``NoneType`` (e.g. ``Optional[X]``, ``X | None``, ``X | Y | None``).

    Args:
        annotation: The annotation to inspect.

    Returns:
        A ``(inner, is_optional)`` pair. When optional, ``inner`` is the
        annotation with ``NoneType`` removed -- a single type if one member
        remains, else a union of the remaining members -- and ``is_optional``
        is ``True``. When not optional, returns ``(annotation, False)``.
    """
    members = get_union_members(annotation)
    if members is None or NoneType not in members:
        return annotation, False
    remaining = tuple(member for member in members if member is not NoneType)
    if len(remaining) == 1:
        return remaining[0], True
    return Union[remaining], True


def get_origin_and_arguments(annotation: Any) -> tuple[Any, tuple[Any, ...]] | None:
    """Return the origin and arguments of a parameterized container generic.

    Recognizes parameterized generics such as ``tuple[int, ...]``,
    ``frozenset[str]``, and ``list[X]``. Unions, ``Annotated``, and ``Final``
    are deliberately excluded -- use :func:`get_union_members` /
    :func:`split_optional` / :func:`unwrap_annotation` for those -- so a
    non-``None`` result unambiguously denotes a container generic.

    Args:
        annotation: The annotation to inspect.

    Returns:
        An ``(origin, args)`` pair (e.g. ``(tuple, (int, Ellipsis))``) when
        ``annotation`` is a parameterized container generic, otherwise ``None``.
    """
    origin = get_origin(annotation)
    if origin is None or origin in (Union, UnionType, Annotated, Final):
        return None
    return origin, get_args(annotation)
