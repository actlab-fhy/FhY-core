"""Tests `BinderMixin` via a small lambda-calculus toy IR.

The toy IR is three term kinds: ``_Var`` (a reference), ``_App`` (application,
a non-binder composite), and ``_Lam`` (a binder over a tuple of parameters).
``_Lam`` is the system under test; ``_Var``/``_App`` are the surrounding terms
the binder scopes over.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from fhy_core.identifier import Identifier
from fhy_core.term import (
    AlphaEquivalenceMixin,
    AlphaRenaming,
    BinderMixin,
    HasFreeIdentifiers,
    Term,
)
from fhy_core.utils.override import override

from .conftest import mock_identifier


@dataclass(frozen=True)
class _Var(AlphaEquivalenceMixin):
    """A reference to an identifier."""

    identifier: Identifier

    def get_free_identifiers(self) -> frozenset[Identifier]:
        return frozenset({self.identifier})

    @override
    def is_alpha_equivalent_under(self, other: object, renaming: AlphaRenaming) -> bool:
        return isinstance(other, _Var) and renaming.are_identifiers_alpha_equivalent(
            self.identifier, other.identifier
        )

    def substitute(self, replacements: Mapping[Identifier, Term]) -> Term:
        return replacements.get(self.identifier, self)


@dataclass(frozen=True)
class _App(AlphaEquivalenceMixin):
    """Application of one term to another (a non-binder composite)."""

    function: Term
    argument: Term

    def get_free_identifiers(self) -> frozenset[Identifier]:
        return (
            self.function.get_free_identifiers() | self.argument.get_free_identifiers()
        )

    @override
    def is_alpha_equivalent_under(self, other: object, renaming: AlphaRenaming) -> bool:
        return (
            isinstance(other, _App)
            and self.function.is_alpha_equivalent_under(other.function, renaming)
            and self.argument.is_alpha_equivalent_under(other.argument, renaming)
        )

    def substitute(self, replacements: Mapping[Identifier, Term]) -> Term:
        return _App(
            self.function.substitute(replacements),
            self.argument.substitute(replacements),
        )


@dataclass(frozen=True)
class _Lam(BinderMixin):
    """A lambda binding a tuple of parameters over a body term."""

    parameters: tuple[Identifier, ...]
    body: Term

    @override
    def get_bound_identifiers(self) -> Sequence[Identifier]:
        return self.parameters

    @override
    def get_scoped_children(self) -> Sequence[Term]:
        return (self.body,)

    @override
    def rename_bound_identifier(self, old: Identifier, new: Identifier) -> "_Lam":
        renamed_parameters = tuple(
            new if parameter == old else parameter for parameter in self.parameters
        )
        return _Lam(renamed_parameters, self.body.substitute({old: _Var(new)}))

    @override
    def rebuild_with_scoped_children(self, new_children: Sequence[Term]) -> "_Lam":
        (new_body,) = new_children
        return _Lam(self.parameters, new_body)


@dataclass(frozen=True)
class _Block(BinderMixin):
    """A binder scoping its parameters over a variable-length statement list."""

    parameters: tuple[Identifier, ...]
    statements: tuple[Term, ...]

    @override
    def get_bound_identifiers(self) -> Sequence[Identifier]:
        return self.parameters

    @override
    def get_scoped_children(self) -> Sequence[Term]:
        return self.statements

    @override
    def rename_bound_identifier(self, old: Identifier, new: Identifier) -> "_Block":
        renamed_parameters = tuple(
            new if parameter == old else parameter for parameter in self.parameters
        )
        renamed_statements = tuple(
            statement.substitute({old: _Var(new)}) for statement in self.statements
        )
        return _Block(renamed_parameters, renamed_statements)

    @override
    def rebuild_with_scoped_children(self, new_children: Sequence[Term]) -> "_Block":
        return _Block(self.parameters, tuple(new_children))


def test_binder_is_a_term() -> None:
    """Test a binder satisfies the `Term` and `HasFreeIdentifiers` protocols."""
    lam = _Lam((mock_identifier("x", 1),), _Var(mock_identifier("x", 1)))
    assert isinstance(lam, Term)
    assert isinstance(lam, HasFreeIdentifiers)


def test_identity_lambdas_are_alpha_equivalent() -> None:
    """Test two identity lambdas with different parameter names are equivalent."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    assert _Lam((x,), _Var(x)).is_alpha_equivalent(_Lam((y,), _Var(y)))


def test_lambdas_with_distinct_free_bodies_are_not_alpha_equivalent() -> None:
    """Test lambdas whose bodies reference distinct free identifiers differ."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    free_a = mock_identifier("a", 4)
    free_b = mock_identifier("b", 5)
    assert not _Lam((x,), _Var(free_a)).is_alpha_equivalent(_Lam((y,), _Var(free_b)))


def test_lambdas_sharing_a_free_identifier_are_alpha_equivalent() -> None:
    """Test lambdas that bind differently but share a free identifier are equivalent."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    shared = mock_identifier("shared", 7)
    assert _Lam((x,), _Var(shared)).is_alpha_equivalent(_Lam((y,), _Var(shared)))


def test_lambdas_with_different_arity_are_not_alpha_equivalent() -> None:
    """Test binders with different numbers of bound identifiers are not equivalent."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    z = mock_identifier("z", 3)
    one_parameter = _Lam((x,), _Var(x))
    two_parameters = _Lam((y, z), _Var(y))
    assert not one_parameter.is_alpha_equivalent(two_parameters)


def test_free_identifiers_excludes_bound_parameter() -> None:
    """Test a parameter bound by the lambda is not free in it."""
    x = mock_identifier("x", 1)
    assert _Lam((x,), _Var(x)).get_free_identifiers() == frozenset()


def test_free_identifiers_includes_unbound_body_reference() -> None:
    """Test an identifier referenced but not bound is free."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    assert _Lam((x,), _Var(y)).get_free_identifiers() == frozenset({y})


def test_free_identifiers_of_application_unions_children() -> None:
    """Test free identifiers of a body application union over its sub-terms."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    z = mock_identifier("z", 3)
    lam = _Lam((x,), _App(_Var(x), _App(_Var(y), _Var(z))))
    assert lam.get_free_identifiers() == frozenset({y, z})


def test_substitute_skips_shadowed_bound_identifier() -> None:
    """Test a replacement keyed by a bound identifier does not apply inside."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    identity = _Lam((x,), _Var(x))

    result = identity.substitute({x: _Var(y)})

    assert result.is_alpha_equivalent(identity)


def test_substitute_into_body_without_capture() -> None:
    """Test substitution of a free identifier rewrites the body in place."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    z = mock_identifier("z", 3)
    lam = _Lam((x,), _App(_Var(x), _Var(y)))

    result = lam.substitute({y: _Var(z)})

    assert result == _Lam((x,), _App(_Var(x), _Var(z)))


def test_substitute_avoids_capture_by_renaming_binder() -> None:
    """Test substitution renames a binder when a replacement would be captured.

    Substituting ``y := x`` into ``\\x. y`` must not yield ``\\x. x`` (which
    would capture the free ``x``). The binder is renamed to a fresh
    parameter, leaving ``x`` free in the result and the body equal to the
    substituted reference.
    """
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    lam = _Lam((x,), _Var(y))

    result = lam.substitute({y: _Var(x)})

    assert result.get_free_identifiers() == frozenset({x})
    assert result.parameters[0] != x
    assert result.body == _Var(x)
    assert not result.is_alpha_equivalent(_Lam((x,), _Var(x)))


def test_substitute_with_no_replacements_returns_self() -> None:
    """Test substituting an empty replacement map returns the same object."""
    x = mock_identifier("x", 1)
    lam = _Lam((x,), _Var(x))
    assert lam.substitute({}) is lam


def test_binder_is_not_alpha_equivalent_to_non_binder() -> None:
    """Test a binder is not alpha-equivalent to a term of a different kind."""
    x = mock_identifier("x", 1)
    assert not _Lam((x,), _Var(x)).is_alpha_equivalent(_Var(x))


def test_non_injective_binding_is_not_alpha_equivalent() -> None:
    """Test a binder that binds one identifier twice is not equivalent.

    Pairing the bound identifiers yields a non-injective renaming, which
    cannot be a valid bijection, so the comparison is ``False``.
    """
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    a = mock_identifier("a", 4)
    well_formed = _Lam((x, y), _Var(x))
    binds_one_identifier_twice = _Lam((a, a), _Var(a))
    assert not well_formed.is_alpha_equivalent(binds_one_identifier_twice)


def test_blocks_with_different_statement_counts_are_not_alpha_equivalent() -> None:
    """Test binders with different numbers of scoped children are not equivalent."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    one_statement = _Block((x,), (_Var(x),))
    two_statements = _Block((y,), (_Var(y), _Var(y)))
    assert not one_statement.is_alpha_equivalent(two_statements)


def test_block_free_identifiers_union_over_all_statements() -> None:
    """Test a multi-child binder unions free identifiers across its children."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    z = mock_identifier("z", 3)
    block = _Block((x,), (_Var(x), _Var(y), _Var(z)))
    assert block.get_free_identifiers() == frozenset({y, z})


def test_block_alpha_equivalence_recurses_over_all_statements() -> None:
    """Test a multi-child binder compares every scoped child under the renaming."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    left = _Block((x,), (_Var(x), _Var(x)))
    right = _Block((y,), (_Var(y), _Var(y)))
    mismatch = _Block((y,), (_Var(y), _Var(mock_identifier("free", 6))))
    assert left.is_alpha_equivalent(right)
    assert not left.is_alpha_equivalent(mismatch)
