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

On the Rust backend (``fhy_core.RUST_BACKEND_SELECTED``), ``Position``,
``Span`` and the provenance classes are backed by the Rust implementation,
with the same API, text, payloads and pickles. Their arguments are
type-checked at construction, and ``FileProvenance`` stores its path as a
``pathlib.Path`` in the normal form ``pathlib.PurePosixPath`` gives it.
``HasProvenance`` stays a Python protocol on both backends.
"""

from fhy_core.utils.override import override

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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from fhy_core._backend import IS_RUST_BACKEND_SELECTED
from fhy_core.logger import get_logger
from fhy_core.serialization import (
    Serializable,
    WrappedFamilySerializable,
    register_serializable,
)
from fhy_core.traits.equality import EqualMixin
from fhy_core.traits.frozen import FrozenMixin
from fhy_core.utils.numeric_utils import is_strict_int

_LOGGER = get_logger(__name__)


if TYPE_CHECKING or not IS_RUST_BACKEND_SELECTED:

    @register_serializable(type_id="position")
    @dataclass(frozen=True, slots=True, order=True)
    class Position(Serializable, FrozenMixin, EqualMixin):
        """A 1-indexed line/column position in a source text.

        Pickling a position stores a call of its class with its fields, so a
        pickle loads under either backend.
        """

        line: int
        column: int

        @property
        def supports_ordering(self) -> bool:
            """Whether this position defines a total order."""
            return True

        @property
        def supports_partial_ordering(self) -> bool:
            """Whether this position defines a partial order."""
            return True

        def __post_init__(self) -> None:
            if not is_strict_int(self.line):
                raise TypeError(
                    f'"line" must be a strict int, got {type(self.line).__name__}'
                )
            if not is_strict_int(self.column):
                raise TypeError(
                    f'"column" must be a strict int, got {type(self.column).__name__}'
                )
            if self.line < 1:
                raise ValueError(f'"line" must be >= 1, got {self.line}')
            if self.column < 1:
                raise ValueError(f'"column" must be >= 1, got {self.column}')

        @override
        def __str__(self) -> str:
            return f"{self.line}:{self.column}"

        @override
        def __reduce__(self) -> tuple[type["Position"], tuple[int, int]]:
            return (type(self), (self.line, self.column))

    @register_serializable(type_id="span")
    @dataclass(frozen=True, slots=True)
    class Span(Serializable, FrozenMixin, EqualMixin):
        """A file-agnostic byte/position range.

        Owned by ``FileProvenance``; does not carry a file path itself.
        Pickling a span stores a call of its class with its fields, so a
        pickle loads under either backend.
        """

        start_offset: int | None = None
        end_offset: int | None = None
        start_position: Position | None = None
        end_position: Position | None = None

        def __post_init__(self) -> None:
            for offset_name, offset_value in (
                ("start_offset", self.start_offset),
                ("end_offset", self.end_offset),
            ):
                if offset_value is not None and not is_strict_int(offset_value):
                    raise TypeError(
                        f'"{offset_name}" must be a strict int or None, got '
                        f"{type(offset_value).__name__}"
                    )
            if self.start_offset is not None and self.start_offset < 0:
                raise ValueError(
                    f'"start_offset" must be >= 0, got {self.start_offset}'
                )
            if self.end_offset is not None and self.end_offset < 0:
                raise ValueError(f'"end_offset" must be >= 0, got {self.end_offset}')
            if (
                self.start_offset is not None
                and self.end_offset is not None
                and self.end_offset < self.start_offset
            ):
                raise ValueError(
                    '"end_offset" must be >= "start_offset", got '
                    f"{self.end_offset} < {self.start_offset}"
                )
            if (
                self.start_position is not None
                and self.end_position is not None
                and self.end_position < self.start_position
            ):
                raise ValueError(
                    '"end_position" must be >= "start_position", got '
                    f"{self.end_position} < {self.start_position}"
                )

        def is_unknown(self) -> bool:
            """Whether the span carries no offset or position information."""
            return (
                self.start_offset is None
                and self.end_offset is None
                and self.start_position is None
                and self.end_position is None
            )

        @override
        def __str__(self) -> str:
            if self.is_unknown():
                return "<unknown>"
            if self.start_position is not None or self.end_position is not None:
                start = (
                    str(self.start_position) if self.start_position is not None else "?"
                )
                end = str(self.end_position) if self.end_position is not None else "?"
                return f"{start}-{end}"
            start_offset = (
                str(self.start_offset) if self.start_offset is not None else "?"
            )
            end_offset = str(self.end_offset) if self.end_offset is not None else "?"
            return f"@{start_offset}-{end_offset}"

        @override
        def __reduce__(
            self,
        ) -> tuple[
            type["Span"],
            tuple[int | None, int | None, Position | None, Position | None],
        ]:
            return (
                type(self),
                (
                    self.start_offset,
                    self.end_offset,
                    self.start_position,
                    self.end_position,
                ),
            )

    class Provenance(WrappedFamilySerializable, FrozenMixin, EqualMixin, ABC):
        """Origin information for a compiler object. Abstract base.

        Pickling a provenance stores a call of its class with its fields, so
        a pickle loads under either backend.
        """

        @abstractmethod
        @override
        def __str__(self) -> str: ...

        @staticmethod
        def unknown() -> "Provenance":
            """Return the unknown provenance sentinel."""
            return UnknownProvenance()

        @staticmethod
        def fuse(
            *provenances: "Provenance", metadata: str | None = None
        ) -> "Provenance":
            """Return the provenance formed by fusing the given provenances."""
            flat: list[Provenance] = []
            pending: list[Provenance] = list(reversed(provenances))
            unknowns_dropped = 0
            fused_collapsed = 0
            while pending:
                provenance = pending.pop()
                if isinstance(provenance, UnknownProvenance):
                    unknowns_dropped += 1
                    continue
                if (
                    isinstance(provenance, FusedProvenance)
                    and provenance.metadata is None
                ):
                    fused_collapsed += 1
                    pending.extend(reversed(provenance.sources))
                    continue
                flat.append(provenance)

            if unknowns_dropped or fused_collapsed:
                _LOGGER.debug(
                    "reduced input (inputs=%d, unknowns_dropped=%d, "
                    "fused_collapsed=%d, final_sources=%d)",
                    len(provenances),
                    unknowns_dropped,
                    fused_collapsed,
                    len(flat),
                )

            if not flat:
                return UnknownProvenance()
            if len(flat) == 1 and metadata is None:
                return flat[0]
            return FusedProvenance(sources=tuple(flat), metadata=metadata)

    @register_serializable(type_id="provenance.unknown")
    @dataclass(frozen=True, slots=True)
    class UnknownProvenance(Provenance):
        """Provenance with no source information."""

        @override
        def __str__(self) -> str:
            return "<unknown>"

        @override
        def __reduce__(self) -> tuple[type["UnknownProvenance"], tuple[()]]:
            return (type(self), ())

    @register_serializable(type_id="provenance.file")
    @dataclass(frozen=True, slots=True)
    class FileProvenance(Provenance):
        """Provenance pointing to a region within a source code file."""

        file_path: Path
        span: Span | None = None

        @override
        def __str__(self) -> str:
            if self.span is None or self.span.is_unknown():
                return str(self.file_path)
            else:
                return f"{self.file_path}:{self.span}"

        @override
        def __reduce__(
            self,
        ) -> tuple[type["FileProvenance"], tuple[Path, Span | None]]:
            return (type(self), (self.file_path, self.span))

    @register_serializable(type_id="provenance.named")
    @dataclass(frozen=True, slots=True)
    class NamedProvenance(Provenance):
        """Wraps a child provenance with a human-readable label."""

        name: str
        child: Provenance

        def __post_init__(self) -> None:
            if not self.name:
                raise ValueError('"name" must be non-empty')

        @override
        def __str__(self) -> str:
            if isinstance(self.child, UnknownProvenance):
                return self.name
            else:
                return f"{self.name} ({self.child})"

        @override
        def __reduce__(self) -> tuple[type["NamedProvenance"], tuple[str, Provenance]]:
            return (type(self), (self.name, self.child))

    @register_serializable(type_id="provenance.call_site")
    @dataclass(frozen=True, slots=True)
    class CallSiteProvenance(Provenance):
        """Provenance for a value created at a call site."""

        callee: Provenance
        caller: Provenance

        @override
        def __str__(self) -> str:
            return f"{self.callee} at {self.caller}"

        @override
        def __reduce__(
            self,
        ) -> tuple[type["CallSiteProvenance"], tuple[Provenance, Provenance]]:
            return (type(self), (self.callee, self.caller))

    @register_serializable(type_id="provenance.fused")
    @dataclass(frozen=True, slots=True)
    class FusedProvenance(Provenance):
        """N provenances combined by a transformation."""

        sources: tuple[Provenance, ...]
        metadata: str | None = None

        @override
        def __str__(self) -> str:
            label = self.metadata if self.metadata is not None else "fused"
            rendered_sources = ", ".join(str(source) for source in self.sources)
            return f"{label}[{rendered_sources}]"

        @override
        def __reduce__(
            self,
        ) -> tuple[type["FusedProvenance"], tuple[tuple[Provenance, ...], str | None]]:
            return (type(self), (self.sources, self.metadata))

else:
    from fhy_core import _rs

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

    FrozenMixin.register(Provenance)
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
