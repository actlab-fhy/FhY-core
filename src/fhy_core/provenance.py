"""Provenance tracking for compiler objects.

This module exposes a ``Provenance`` abstract base class and the five
canonical variants used by MLIR's ``Location`` attribute family. Each
variant captures one well-defined concept; richer concepts (builtins,
library symbols, lowered objects) are expressed by composition rather than
by adding new variants.

Variants:

- ``UnknownProvenance``: no source information is available.
- ``FileProvenance``: a region within a source code file.
- ``NamedProvenance``: wraps a child provenance with a human-readable name.
- ``CallSiteProvenance``: a value created at a call site (inlining,
    macro expansion).
- ``FusedProvenance``: N provenances combined by a transformation.

Combining provenances during transformations is done through
``Provenance.fuse``, which applies a small set of reduction rules to keep
fusion trees compact.

``Position``, ``Span`` and the provenance classes are backed by the Rust
implementation. Their arguments are type-checked at construction, and
``FileProvenance`` stores its path as a ``pathlib.Path`` in the normal form
``pathlib.PurePosixPath`` gives it. ``HasProvenance`` is a Python protocol.
"""

__all__ = [
    "CallSiteProvenance",
    "FileProvenance",
    "FusedProvenance",
    "HasProvenance",
    "NamedProvenance",
    "Position",
    "Provenance",
    "Span",
    "UnknownProvenance",
]

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from fhy_core import _rs
from fhy_core.logger import get_logger
from fhy_core.serialization import (
    Serializable,
    WrappedFamilySerializable,
    register_serializable,
)
from fhy_core.traits.equality import EqualMixin
from fhy_core.traits.frozen import FrozenMixin
from fhy_core.utils.override import override

# The binding's `Provenance.fuse` logs its reductions through this logger.
_LOGGER = get_logger(__name__)


@register_serializable(type_id="position")
class Position(_rs.Position, Serializable, EqualMixin):
    """A 1-indexed line/column position in a source text.

    Backed by the Rust implementation: ``fhy_core._rs.Position`` holds
    the Rust position and implements the fields, ordering, equality,
    hashing, ``str``, ``repr`` and payloads. This class mixes in the
    stateless Python protocols, and is registered as a virtual subclass
    of ``FrozenMixin``. Positions are immutable, and pickle as a call of
    their class with their fields.

    Attributes:
        line: The 1-indexed line, a strict ``int``.
        column: The 1-indexed column, a strict ``int``.

    """

    __slots__ = ()
    __match_args__ = ("line", "column")


FrozenMixin.register(Position)
Position._register_public_class()


@register_serializable(type_id="span")
class Span(_rs.Span, Serializable, EqualMixin):
    """A file-agnostic byte/position range.

    Owned by ``FileProvenance``; does not carry a file path itself.
    Backed by the Rust implementation: ``fhy_core._rs.Span`` holds the
    Rust span and implements the fields, ``is_unknown``, equality,
    hashing, ``str``, ``repr`` and payloads. This class mixes in the
    stateless Python protocols, and is registered as a virtual subclass
    of ``FrozenMixin``. A position must be a :class:`Position` or
    ``None``, and raises ``TypeError`` otherwise. Spans are immutable,
    and pickle as a call of their class with their fields.

    Attributes:
        start_offset: The byte offset the span starts at, or ``None``.
        end_offset: The byte offset the span ends at, or ``None``.
        start_position: The position the span starts at, or ``None``.
        end_position: The position the span ends at, or ``None``.

    """

    __slots__ = ()
    __match_args__ = (
        "start_offset",
        "end_offset",
        "start_position",
        "end_position",
    )


FrozenMixin.register(Span)
Span._register_public_class()


class Provenance(_rs.Provenance, WrappedFamilySerializable, EqualMixin, ABC):
    """Origin information for a compiler object. Abstract base.

    Backed by the Rust implementation: ``fhy_core._rs.Provenance`` holds
    the Rust provenance and implements equality, hashing, ``unknown`` and
    ``fuse``; each variant's ``_rs`` class implements its fields,
    ``str``, ``repr`` and data payload. The classes mix in the stateless
    Python protocols, including the ``WrappedFamilySerializable``
    envelope, and are registered as virtual subclasses of
    ``FrozenMixin``. Provenances are immutable, and pickle as a call of
    their class with their fields.
    """

    __slots__ = ()

    @abstractmethod
    @override
    def __str__(self) -> str: ...


# `register` takes any class, abstract or not.
FrozenMixin.register(Provenance)  # type: ignore[type-abstract]
Provenance._register_public_class()


@register_serializable(type_id="provenance.unknown")
class UnknownProvenance(_rs.UnknownProvenance, Provenance):
    """Provenance with no source information."""

    __slots__ = ()
    __match_args__ = ()


UnknownProvenance._register_public_class()


@register_serializable(type_id="provenance.file")
class FileProvenance(_rs.FileProvenance, Provenance):
    """Provenance pointing to a region within a source code file.

    ``file_path`` must be a ``str`` or an ``os.PathLike`` of a ``str``,
    and is stored as a ``pathlib.Path`` in the normal form
    ``pathlib.PurePosixPath`` gives it; a ``PurePath`` already in normal
    form is kept as given. ``span`` must be a :class:`Span` or ``None``.
    Either raises ``TypeError`` otherwise.

    Attributes:
        file_path: The path of the file.
        span: The region of the file, or ``None``.

    """

    __slots__ = ()
    __match_args__ = ("file_path", "span")


FileProvenance._register_public_class()


@register_serializable(type_id="provenance.named")
class NamedProvenance(_rs.NamedProvenance, Provenance):
    """Wraps a child provenance with a human-readable label.

    ``name`` must be a non-empty ``str`` and ``child`` a
    :class:`Provenance`; a wrong type raises ``TypeError``.

    Attributes:
        name: The label.
        child: The labelled provenance.

    """

    __slots__ = ()
    __match_args__ = ("name", "child")


NamedProvenance._register_public_class()


@register_serializable(type_id="provenance.call_site")
class CallSiteProvenance(_rs.CallSiteProvenance, Provenance):
    """Provenance for a value created at a call site.

    ``callee`` and ``caller`` must be :class:`Provenance` instances, and
    raise ``TypeError`` otherwise.

    Attributes:
        callee: The provenance of the called code.
        caller: The provenance of the call site.

    """

    __slots__ = ()
    __match_args__ = ("callee", "caller")


CallSiteProvenance._register_public_class()


@register_serializable(type_id="provenance.fused")
class FusedProvenance(_rs.FusedProvenance, Provenance):
    """N provenances combined by a transformation.

    ``sources`` may be any iterable of :class:`Provenance` instances and
    is stored as a tuple; ``metadata`` must be a ``str`` or ``None``.
    Either raises ``TypeError`` otherwise.

    Attributes:
        sources: The fused provenances, in order.
        metadata: The label of the transformation, or ``None``.

    """

    __slots__ = ()
    __match_args__ = ("sources", "metadata")


FusedProvenance._register_public_class()


@runtime_checkable
class HasProvenance(Protocol):
    """Protocol for objects that carry provenance information.

    Provenance records an object's origin: source span, lowering steps,
    original node, etc. The protocol lives beside :class:`Provenance`
    rather than in :mod:`fhy_core.traits` because its signature names a
    provenance value, which makes it vocabulary of this module rather
    than a generic structural contract.
    """

    def get_provenance(self) -> Provenance:
        """Return the object's provenance information."""
