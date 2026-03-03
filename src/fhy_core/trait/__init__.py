"""Compiler object traits package."""

__all__ = [
    "HasIdentifier",
    "HasIdentifierMixin",
    "HasProvenance",
    "HasProvenanceMixin",
]

from .has_identifier import HasIdentifier, HasIdentifierMixin
from .has_provenance import HasProvenance, HasProvenanceMixin
