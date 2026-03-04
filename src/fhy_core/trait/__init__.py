"""Compiler object traits package."""

__all__ = [
    "Canonicalizable",
    "CanonicalizableMixin",
    "Foldable",
    "FoldableMixin",
    "HasIdentifier",
    "HasIdentifierMixin",
    "HasOperands",
    "HasOperandsMixin",
    "HasProvenance",
    "HasProvenanceMixin",
    "HasResults",
    "HasResultsMixin",
    "HasType",
    "HasTypeMixin",
    "StructuralEquivalence",
    "StructuralEquivalenceMixin",
    "Verifiable",
    "VerifiableMixin",
    "Visitable",
    "VisitableMixin",
    "VerificationError",
]

from .canonicalizable import Canonicalizable, CanonicalizableMixin
from .foldable import Foldable, FoldableMixin
from .has_identifier import HasIdentifier, HasIdentifierMixin
from .has_operands import HasOperands, HasOperandsMixin
from .has_provenance import HasProvenance, HasProvenanceMixin
from .has_results import HasResults, HasResultsMixin
from .has_type import HasType, HasTypeMixin
from .structural_equivalence import (
    StructuralEquivalence,
    StructuralEquivalenceMixin,
)
from .verifiable import Verifiable, VerifiableMixin, VerificationError
from .visitable import Visitable, VisitableMixin
