"""Open, registry-backed semantic tags for compiler operations.

Every layer of a compiler stack carries semantic attributes on its
operations -- algebraic properties such as commutativity and
associativity, purity, elementwise application, and family-specific tags
contributed by particular IRs. ``OpAttribute`` exposes that classification
as an open class with canonical interning, so layers can share the four
generic algebraic/semantic attributes shipped here while contributing
their own without modifying ``fhy_core``.

``OpAttribute`` is a free-standing tag primitive: it has no dependency on
any particular operation type and is not specialized for any layer.
Callers should import the canonical names rather than constructing fresh
``OpAttribute`` instances with the same ``name_hint``, because
``Identifier`` uses id-equality and a freshly-constructed
``Identifier("commutative")`` would not match the canonical one.

The shipped attributes hold fixed identifier ids, the same in every process
and on either backend, so their payloads and pickles are portable. On the
Rust backend (``fhy_core.RUST_BACKEND_SELECTED``), ``OpAttribute`` is backed
by the Rust implementation, whose registry is the only one in the process:
constructing an attribute whose name is registered returns the canonical
instance itself, and the registry cannot be cleared.
"""

from fhy_core.utils.override import override

__all__ = [
    "ASSOCIATIVE",
    "COMMUTATIVE",
    "ELEMENTWISE",
    "PURE",
    "OpAttribute",
]

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ._backend import IS_RUST_BACKEND_SELECTED
from .identifier import (
    _RESERVED_ASSOCIATIVE,
    _RESERVED_COMMUTATIVE,
    _RESERVED_ELEMENTWISE,
    _RESERVED_PURE,
    HasIdentifier,
    Identifier,
    _build_reserved_identifier,
)
from .serialization import (
    Serializable,
    SerializedDict,
    register_serializable,
)
from .term import AlphaEquivalenceMixin, DerivedEquivalenceMixin
from .traits import FrozenMixin, InternedMixin, StructuralEquivalence

if TYPE_CHECKING or not IS_RUST_BACKEND_SELECTED:

    @register_serializable(type_id="op_attribute")
    @dataclass(frozen=True)
    class OpAttribute(
        HasIdentifier,
        FrozenMixin,
        DerivedEquivalenceMixin,
        InternedMixin[Identifier],
        Serializable,
    ):
        """Open semantic tag attached to a compiler operation.

        Each ``OpAttribute`` is uniquely identified by an ``Identifier`` and
        is canonicalized through the ``InternedMixin`` registry: the first
        instance constructed for a given ``Identifier`` becomes the canonical
        entry and subsequent constructions with the same key shadow into that
        entry without replacing it.

        ``description`` is human-readable metadata only -- it does not
        participate in equality, structural equivalence, hashing, or
        interning. The first instance registered for a given ``Identifier``
        becomes canonical; subsequent constructions and deserializations
        with a different description are not rejected but do not update the
        canonical description. Deserializing a payload whose description
        differs from the canonical's emits a warning. Pickling an attribute
        stores its payload, so unpickling returns the canonical instance.

        Attributes:
            name: Stable, process-global identifier for this attribute.
            description: Short human-readable description (surfaced in error
                messages, documentation, and pass-author guidance; excluded
                from structural equivalence).

        """

        name: Identifier
        description: str = field(compare=False)

        def __post_init__(self) -> None:
            self.register_interned_instance()

        @override
        def get_identifier(self) -> Identifier:
            return self.name

        @override
        def get_intern_key(self) -> Identifier:
            return self.name

        @override
        def __reduce__(
            self,
        ) -> tuple[Callable[[SerializedDict], "OpAttribute"], tuple[SerializedDict]]:
            return (OpAttribute.deserialize_from_dict, (self.serialize_to_dict(),))

        @classmethod
        @override
        def register_default_instances(cls) -> None:
            """Re-register the canonical default ``OpAttribute``s shipped here.

            After :meth:`clear_interned_registry` wipes the registry, call this
            method to restore ``COMMUTATIVE``, ``ASSOCIATIVE``, ``PURE``, and
            ``ELEMENTWISE`` so the module-level constants remain canonical.
            """
            for instance in _DEFAULT_INSTANCES:
                instance.register_interned_instance()

    COMMUTATIVE: OpAttribute = OpAttribute(
        _build_reserved_identifier(_RESERVED_COMMUTATIVE),
        "Op output is invariant under operand swap.",
    )
    ASSOCIATIVE: OpAttribute = OpAttribute(
        _build_reserved_identifier(_RESERVED_ASSOCIATIVE),
        "Op composes associatively across applications.",
    )
    PURE: OpAttribute = OpAttribute(
        _build_reserved_identifier(_RESERVED_PURE),
        "Op has no side effects and produces deterministic outputs.",
    )
    ELEMENTWISE: OpAttribute = OpAttribute(
        _build_reserved_identifier(_RESERVED_ELEMENTWISE),
        "Op acts independently on each element of its operands.",
    )

else:
    from . import _rs

    @register_serializable(type_id="op_attribute")
    class OpAttribute(
        _rs.OpAttribute,
        HasIdentifier,
        StructuralEquivalence,
        AlphaEquivalenceMixin,
        Serializable,
    ):
        """Open semantic tag attached to a compiler operation.

        Backed by the Rust implementation: the Rust registry holds the
        canonical attributes, and ``fhy_core._rs.OpAttribute`` implements
        the attributes, equality, hashing, interning and payloads. This class
        mixes in the stateless Python protocols, and is registered as a
        virtual subclass of ``InternedMixin`` and ``FrozenMixin``.

        Constructing an attribute whose ``name`` is registered returns the
        canonical instance itself, keeping its description. Attributes are
        immutable, compare and hash by ``name``, and pickle as their payload,
        so unpickling returns the canonical instance. The registry is
        append-only: ``clear_interned_registry`` and
        ``register_default_instances`` raise ``NotImplementedError``.

        Attributes:
            name: Stable, process-global identifier for this attribute.
            description: Short human-readable description (excluded from
                equality and structural equivalence).

        """

        __slots__ = ()
        __new__ = staticmethod(_rs.OpAttribute._new_canonical)

    InternedMixin.register(OpAttribute)
    FrozenMixin.register(OpAttribute)

    COMMUTATIVE = OpAttribute.require_interned(
        _build_reserved_identifier(_RESERVED_COMMUTATIVE)
    )
    ASSOCIATIVE = OpAttribute.require_interned(
        _build_reserved_identifier(_RESERVED_ASSOCIATIVE)
    )
    PURE = OpAttribute.require_interned(_build_reserved_identifier(_RESERVED_PURE))
    ELEMENTWISE = OpAttribute.require_interned(
        _build_reserved_identifier(_RESERVED_ELEMENTWISE)
    )

_DEFAULT_INSTANCES: tuple[OpAttribute, ...] = (
    COMMUTATIVE,
    ASSOCIATIVE,
    PURE,
    ELEMENTWISE,
)
