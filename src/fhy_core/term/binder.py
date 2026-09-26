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

The derived methods run the Rust core's ``fhy_core::term::Binder``
algorithms, which call the node's hooks and its children's methods (S10 of
``docs/design/python-switch.md``). A binder list that repeats an identifier,
on either side, pairs with none, so such a binder is alpha-equivalent to no
binder, itself included.
"""

from fhy_core.utils.override import override

__all__ = ["BinderMixin", "HasFreeIdentifiers", "Term"]

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import Protocol, cast, runtime_checkable

from fhy_core import _rs
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

    @override
    def is_alpha_equivalent_under(self, other: object, renaming: AlphaRenaming) -> bool:
        """Return whether self and other are alpha-equivalent under the renaming.

        Derived: ``other`` must be the same concrete type with the same
        number of bound identifiers and scoped children; the renaming is
        extended with a frame pairing the two nodes' bound identifiers by
        position and the scoped children are compared under it, in order,
        stopping at the first that differs. A pairing of lists that repeat
        an identifier is refused, and the nodes are then not equivalent.
        """
        return _rs.binder_is_alpha_equivalent_under(self, other, renaming)

    def get_free_identifiers(self) -> frozenset[Identifier]:
        """Return the free identifiers of the scoped children minus the bound set."""
        return _rs.binder_get_free_identifiers(self)

    def substitute(self, replacements: Mapping[Identifier, Term]) -> Self:
        """Return this node with free identifiers replaced, avoiding capture.

        Bound identifiers shadow ``replacements`` (entries keyed by a bound
        identifier do not apply within this node), and with no entry left
        the node itself is returned. When a replacement term would be
        captured by one of this node's binders, that binder is first
        renamed to a fresh identifier with the same name hint via
        :meth:`rename_bound_identifier`.
        """
        return cast(Self, _rs.binder_substitute(self, replacements))
