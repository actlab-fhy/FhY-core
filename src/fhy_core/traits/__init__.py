"""Compiler object traits package.

Each capability is expressed as a structural ``Protocol`` (the contract) paired,
where there is reusable default behavior, with a ``*Mixin`` (the implementation).
Implementers inherit the Protocol explicitly and decorate the implementing
methods with ``@override``.

Every trait here is a generic structural contract: it constrains an object's
shape or its Python data-model behavior without naming any value of the term
language. Traits whose signatures mention an ``Identifier`` belong to
:mod:`fhy_core.term` instead, which keeps this package a leaf of the
dependency graph.

Most mixins are stateless. Two are stateful and cooperate during construction,
so their relative order in a class's MRO matters:

- ``FrozenMixin`` seals an instance after ``__init__`` completes.
- ``InternedMixin`` finalizes (and, if the instance is ``Frozen``/``Verifiable``,
  freezes and verifies) the instance before registering it in the interning
  registry.

When a class mixes both, list ``InternedMixin`` before ``FrozenMixin`` so the
freeze wrap is the inner wrap and the instance is already frozen by the time the
interner's finalize hook runs. See ``FrozenMixin.__init_subclass__`` and
``InternedMixin`` for the per-site notes.
"""

__all__ = [
    "Canonicalizable",
    "Equal",
    "EqualMixin",
    "Frozen",
    "FrozenFieldTypeError",
    "FrozenMixin",
    "FrozenMutationError",
    "FrozenValidationError",
    "HasOperands",
    "HasResults",
    "HasType",
    "Interned",
    "InternedMixin",
    "Orderable",
    "OrderableMixin",
    "PartialEqual",
    "PartialEqualMixin",
    "PartialOrderable",
    "PartialOrderableMixin",
    "Rewritable",
    "RewritableMixin",
    "StructuralEquivalence",
    "Verifiable",
    "VerifiableMixin",
    "VerificationError",
    "Visitable",
    "VisitableMixin",
]

from .canonicalizable import Canonicalizable
from .equality import Equal, EqualMixin, PartialEqual, PartialEqualMixin
from .frozen import (
    Frozen,
    FrozenFieldTypeError,
    FrozenMixin,
    FrozenMutationError,
    FrozenValidationError,
)
from .has_operands import HasOperands
from .has_results import HasResults
from .has_type import HasType
from .interned import Interned, InternedMixin
from .orderable import (
    Orderable,
    OrderableMixin,
    PartialOrderable,
    PartialOrderableMixin,
)
from .rewritable import Rewritable, RewritableMixin
from .structural_equivalence import StructuralEquivalence
from .verifiable import Verifiable, VerifiableMixin, VerificationError
from .visitable import Visitable, VisitableMixin
