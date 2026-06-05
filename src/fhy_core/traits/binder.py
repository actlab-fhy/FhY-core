"""`BinderMixin` and the companion `HasFreeIdentifiers` / `Term` traits.

A binder node (let, lambda, loop, function) introduces identifiers that scope
over some of its children. This module turns the existing alpha-equivalence
layer into a reusable base: an implementation declares the identifiers it binds
and the children those bindings scope over, plus two construction hooks, and
the mixin derives alpha-equivalence, free-identifier computation, and
capture-avoiding substitution.

The children a binder scopes over (and the values substituted into them) are
``Term``s: they compare by alpha-equivalence, report their free identifiers,
and substitute. ``BinderMixin`` is itself a ``Term``, so binders nest.
"""

__all__ = ["BinderMixin", "HasFreeIdentifiers", "Term"]

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from fhy_core.identifier import Identifier
from fhy_core.utils import Self

from .alpha_equivalence import AlphaEquivalence, AlphaEquivalenceMixin, AlphaRenaming


@runtime_checkable
class HasFreeIdentifiers(Protocol):
    """Protocol for terms that can report their free identifiers."""

    def get_free_identifiers(self) -> frozenset[Identifier]:
        """Return the identifiers that occur free in this term."""


@runtime_checkable
class Term(AlphaEquivalence, HasFreeIdentifiers, Protocol):
    """Protocol for terms a binder scopes over.

    A term compares by alpha-equivalence (:class:`AlphaEquivalence`), reports
    its free identifiers (:class:`HasFreeIdentifiers`), and supports
    capture-avoiding substitution of free identifiers.
    """

    def substitute(self, replacements: "Mapping[Identifier, Term]") -> "Term":
        """Return this term with free identifiers replaced per ``replacements``."""


class BinderMixin(AlphaEquivalenceMixin):
    """Base for IR nodes that introduce a binding scope.

    An implementation provides four hooks:

    - :meth:`get_bound_identifiers` -- the identifiers this node binds;
    - :meth:`get_scoped_children` -- the terms those bindings scope over;
    - :meth:`rename_bound_identifier` -- rebuild with one bound identifier
      consistently renamed (used for capture avoidance);
    - :meth:`rebuild_with_scoped_children` -- rebuild with replacement
      scoped children.

    From those, the mixin derives :meth:`is_alpha_equivalent_under`,
    :meth:`get_free_identifiers`, and :meth:`substitute`.
    """

    @abstractmethod
    def get_bound_identifiers(self) -> Sequence[Identifier]:
        """Return the identifiers this node binds over its scoped children."""

    @abstractmethod
    def get_scoped_children(self) -> Sequence[Term]:
        """Return the terms the bound identifiers are in scope for."""

    @abstractmethod
    def rename_bound_identifier(self, old: Identifier, new: Identifier) -> Self:
        """Return a copy with bound identifier ``old`` renamed to ``new``.

        The rename must be consistent: every occurrence of ``old`` bound by
        this node -- in both the bound-identifier list and the scoped
        children -- becomes ``new``. ``new`` is assumed fresh (not free in
        the scoped children).
        """

    @abstractmethod
    def rebuild_with_scoped_children(self, new_children: Sequence[Term]) -> Self:
        """Return a copy of this node with its scoped children replaced.

        The bound identifiers are unchanged; ``new_children`` is positional
        with respect to :meth:`get_scoped_children`.
        """

    def is_alpha_equivalent_under(self, other: object, renaming: AlphaRenaming) -> bool:
        """Return whether self and other are alpha-equivalent under the renaming.

        Derived: ``other`` must be the same concrete type with the same
        number of bound identifiers and scoped children; the renaming is
        extended with the pairwise binding of the two nodes' bound
        identifiers and the scoped children are compared under it.
        """
        if not isinstance(other, BinderMixin) or type(self) is not type(other):
            return False
        self_bound = self.get_bound_identifiers()
        other_bound = other.get_bound_identifiers()
        if len(self_bound) != len(other_bound):
            return False
        self_children = self.get_scoped_children()
        other_children = other.get_scoped_children()
        if len(self_children) != len(other_children):
            return False
        try:
            extended = renaming.extend(dict(zip(self_bound, other_bound)))
        except ValueError:
            return False
        return all(
            self_child.is_alpha_equivalent_under(other_child, extended)
            for self_child, other_child in zip(self_children, other_children)
        )

    def get_free_identifiers(self) -> frozenset[Identifier]:
        """Return the free identifiers of the scoped children minus the bound set."""
        bound = frozenset(self.get_bound_identifiers())
        free: frozenset[Identifier] = frozenset()
        for child in self.get_scoped_children():
            free |= child.get_free_identifiers()
        return free - bound

    def substitute(self, replacements: Mapping[Identifier, Term]) -> Self:
        """Return this node with free identifiers replaced, avoiding capture.

        Bound identifiers shadow ``replacements`` (entries keyed by a bound
        identifier do not apply within this node). When a replacement term
        would be captured by one of this node's binders, that binder is
        first renamed to a fresh identifier via
        :meth:`rename_bound_identifier`.
        """
        bound = frozenset(self.get_bound_identifiers())
        active = {
            identifier: term
            for identifier, term in replacements.items()
            if identifier not in bound
        }
        if not active:
            return self

        capturable: frozenset[Identifier] = frozenset()
        for term in active.values():
            capturable |= term.get_free_identifiers()

        safe = self
        for bound_identifier in self.get_bound_identifiers():
            if bound_identifier in capturable:
                fresh = Identifier(bound_identifier.name_hint)
                safe = safe.rename_bound_identifier(bound_identifier, fresh)

        substituted_children = [
            child.substitute(active) for child in safe.get_scoped_children()
        ]
        return safe.rebuild_with_scoped_children(substituted_children)
