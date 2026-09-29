"""Benchmarks of the term package: renamings, binders and derived equivalence.

They measure ``fhy_core.term``, which the Rust core backs, through the
public API only. The binder rows use a small lambda calculus over
``BinderMixin``, as ``tests/test_binder.py`` does; the derived rows use
frozen dataclasses over ``DerivedEquivalenceMixin``. The consumer rows
compare params, constraints and symbol tables, whose equivalence the derived
engine computes.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pytest

from fhy_core.identifier import Identifier
from fhy_core.symbol_table import SymbolTable, VariableSymbolTableFrame
from fhy_core.symbolic.constraint import EquationConstraint
from fhy_core.symbolic.expression import Expression, IdentifierExpression
from fhy_core.symbolic.param import (
    Param,
    create_integer_param,
    create_integer_param_between,
    create_natural_param_between,
)
from fhy_core.term import (
    AlphaEquivalenceMixin,
    AlphaRenaming,
    BinderMixin,
    DerivedEquivalenceMixin,
    Term,
    compared_as_binder,
    compared_as_reference,
    compared_as_value,
    is_identifier_mapping_alpha_equivalent_under,
)
from fhy_core.types import CoreDataType, NumericalType, PrimitiveDataType, TypeQualifier
from fhy_core.utils.override import override

from .conftest import Benchmark
from .test_expression import _DEEP_TREE_DEPTH, _build_deep_tree, _build_identifiers

pytestmark = pytest.mark.benchmark(group="term")

# How many pairs the large free renaming and the mapping rows hold.
_MAPPING_SIZE = 50
# How many frames the deep renaming stacks, and how deep the nested binders go.
_FRAME_DEPTH = 10
# How many nested additions the deep derived trees hold.
_DERIVED_TREE_DEPTH = 100
# How many variables the benchmarked symbol table holds.
_SYMBOL_TABLE_SIZE = 20


# ---------------------------------------------------------------------------
# A lambda calculus over `BinderMixin`
# ---------------------------------------------------------------------------


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
    """Application of one term to another."""

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
    """A lambda binding a tuple of parameters over a body."""

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
        renamed = tuple(
            new if parameter == old else parameter for parameter in self.parameters
        )
        return _Lam(renamed, self.body.substitute({old: _Var(new)}))

    @override
    def rebuild_with_scoped_children(self, new_children: Sequence[Term]) -> "_Lam":
        (body,) = new_children
        return _Lam(self.parameters, body)


def _build_nested_lambdas(parameters: Sequence[Identifier]) -> Term:
    """Return one-parameter lambdas over `parameters`, outermost first."""
    term: Term = _Var(parameters[-1])
    for parameter in reversed(parameters):
        term = _Lam((parameter,), term)
    return term


# ---------------------------------------------------------------------------
# Terms over `DerivedEquivalenceMixin`
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class _DerivedTerm(DerivedEquivalenceMixin):
    """Base of the derived terms."""


@dataclass(frozen=True, eq=False)
class _DerivedConst(_DerivedTerm):
    """A constant compared by value."""

    value: int


@dataclass(frozen=True, eq=False)
class _DerivedAdd(_DerivedTerm):
    """A sum of two terms."""

    left: _DerivedTerm
    right: _DerivedTerm


@dataclass(frozen=True, eq=False)
class _DerivedVar(_DerivedTerm):
    """A reference compared through the renaming."""

    identifier: Identifier = field(metadata=compared_as_reference())


@dataclass(frozen=True, eq=False)
class _DerivedLam(_DerivedTerm):
    """A binder over a body."""

    parameter: Identifier = field(metadata=compared_as_binder(scopes_over=("body",)))
    body: _DerivedTerm


@dataclass(frozen=True, eq=False)
class _DerivedExpressionHolder(DerivedEquivalenceMixin):
    """A binder over an expression, as a param binds its constraints."""

    parameter: Identifier = field(
        metadata=compared_as_binder(scopes_over=("expression",))
    )
    expression: Expression
    label: str = field(default="holder", metadata=compared_as_value())


@dataclass(frozen=True)
class _Leaf(AlphaEquivalenceMixin):
    """A value compared by payload, for the mapping row."""

    payload: int

    @override
    def is_alpha_equivalent_under(self, other: object, renaming: AlphaRenaming) -> bool:
        del renaming
        return isinstance(other, _Leaf) and self.payload == other.payload


def _build_derived_tree(depth: int) -> _DerivedTerm:
    """Return `depth` nested additions of one to zero."""
    term: _DerivedTerm = _DerivedConst(0)
    for _ in range(depth):
        term = _DerivedAdd(term, _DerivedConst(1))
    return term


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def pair() -> tuple[Identifier, Identifier]:
    """Return two fresh identifiers."""
    return Identifier("x"), Identifier("y")


@pytest.fixture()
def framed_renaming(pair: tuple[Identifier, Identifier]) -> AlphaRenaming:
    """Return a renaming with one frame pairing `pair`, over a free pair."""
    x, y = pair
    return AlphaRenaming.with_free_renaming({Identifier("a"): Identifier("b")}).extend(
        {x: y}
    )


@pytest.fixture()
def deep_renaming() -> AlphaRenaming:
    """Return a renaming of `_FRAME_DEPTH` one-pair frames."""
    renaming = AlphaRenaming.empty()
    for left, right in zip(
        _build_identifiers(_FRAME_DEPTH, "l"),
        _build_identifiers(_FRAME_DEPTH, "r"),
        strict=True,
    ):
        renaming = renaming.extend({left: right})
    return renaming


def _build_symbol_table(names: Sequence[Identifier]) -> SymbolTable:
    """Return a table of one namespace holding a variable per name."""
    namespace = names[0]
    table = SymbolTable()
    table.add_namespace(namespace)
    int32 = NumericalType(PrimitiveDataType(CoreDataType.INT32))
    for name in names:
        table.add_symbol(
            namespace, name, VariableSymbolTableFrame(name, int32, TypeQualifier.STATE)
        )
    return table


# ---------------------------------------------------------------------------
# AlphaRenaming
# ---------------------------------------------------------------------------


def test_alpha_renaming_empty(benchmark: Benchmark) -> None:
    """Benchmark the empty renaming."""
    benchmark(AlphaRenaming.empty)


@pytest.mark.parametrize("size", [1, _MAPPING_SIZE])
def test_alpha_renaming_with_free_renaming(benchmark: Benchmark, size: int) -> None:
    """Benchmark building a free renaming of `size` pairs."""
    mapping = dict(
        zip(_build_identifiers(size, "a"), _build_identifiers(size, "b"), strict=True)
    )
    benchmark(AlphaRenaming.with_free_renaming, mapping)


@pytest.mark.parametrize("depth", ["depth_1", "depth_10"])
def test_alpha_renaming_extend(
    benchmark: Benchmark,
    framed_renaming: AlphaRenaming,
    deep_renaming: AlphaRenaming,
    depth: str,
) -> None:
    """Benchmark pushing a one-pair frame on a shallow and on a deep stack."""
    renaming = framed_renaming if depth == "depth_1" else deep_renaming
    frame = {Identifier("p"): Identifier("q")}
    benchmark(renaming.extend, frame)


@pytest.mark.parametrize("where", ["frame", "free", "identity"])
def test_alpha_renaming_resolve(
    benchmark: Benchmark,
    pair: tuple[Identifier, Identifier],
    where: str,
) -> None:
    """Benchmark resolving an identifier a frame, the free renaming or nothing maps."""
    x, y = pair
    free = Identifier("a")
    renaming = AlphaRenaming.with_free_renaming({free: Identifier("b")}).extend({x: y})
    identifier = {"frame": x, "free": free, "identity": Identifier("z")}[where]
    benchmark(renaming.resolve, identifier)


@pytest.mark.parametrize("case", ["frame", "capture"])
def test_are_identifiers_alpha_equivalent(
    benchmark: Benchmark,
    framed_renaming: AlphaRenaming,
    pair: tuple[Identifier, Identifier],
    case: str,
) -> None:
    """Benchmark the correspondence of a bound pair, and a refused capture."""
    x, y = pair
    left = x if case == "frame" else y
    expected = case == "frame"
    assert (
        benchmark(framed_renaming.are_identifiers_alpha_equivalent, left, y) is expected
    )


def test_alpha_renaming_eq(
    benchmark: Benchmark, pair: tuple[Identifier, Identifier]
) -> None:
    """Benchmark `==` of two equal renamings built apart."""
    x, y = pair
    left = AlphaRenaming.empty().extend({x: y})
    right = AlphaRenaming.empty().extend({x: y})
    assert benchmark(left.__eq__, right)


def test_alpha_renaming_hash(
    benchmark: Benchmark, framed_renaming: AlphaRenaming
) -> None:
    """Benchmark `hash` of a renaming."""
    benchmark(hash, framed_renaming)


# ---------------------------------------------------------------------------
# BinderMixin
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["flat", "nested_10"])
def test_binder_alpha_equivalence(benchmark: Benchmark, shape: str) -> None:
    """Benchmark comparing two lambdas that rename their parameters."""
    if shape == "flat":
        x, y, z = Identifier("x"), Identifier("y"), Identifier("z")
        left: Term = _Lam((x,), _App(_Var(x), _Var(z)))
        right: Term = _Lam((y,), _App(_Var(y), _Var(z)))
    else:
        left = _build_nested_lambdas(_build_identifiers(_FRAME_DEPTH, "x"))
        right = _build_nested_lambdas(_build_identifiers(_FRAME_DEPTH, "y"))
    assert benchmark(left.is_alpha_equivalent, right)


def test_binder_free_identifiers(benchmark: Benchmark) -> None:
    """Benchmark the free identifiers of a lambda."""
    x, y, z = Identifier("x"), Identifier("y"), Identifier("z")
    lam = _Lam((x,), _App(_Var(x), _App(_Var(y), _Var(z))))
    assert benchmark(lam.get_free_identifiers) == frozenset({y, z})


@pytest.mark.parametrize("case", ["no_capture", "capture"])
def test_binder_substitute(benchmark: Benchmark, case: str) -> None:
    """Benchmark substituting into a lambda, renaming its parameter to avoid capture."""
    x, y, z = Identifier("x"), Identifier("y"), Identifier("z")
    lam = _Lam((x,), _App(_Var(x), _Var(y)))
    replacement = _Var(z) if case == "no_capture" else _Var(x)
    benchmark(lam.substitute, {y: replacement})


# ---------------------------------------------------------------------------
# DerivedEquivalenceMixin
# ---------------------------------------------------------------------------


def test_derived_structural_equivalence_of_a_deep_tree(benchmark: Benchmark) -> None:
    """Benchmark structural equivalence of two equal 100-addition trees."""
    left = _build_derived_tree(_DERIVED_TREE_DEPTH)
    right = _build_derived_tree(_DERIVED_TREE_DEPTH)
    assert benchmark(left.is_structurally_equivalent, right)


def test_derived_alpha_equivalence_of_a_deep_tree(benchmark: Benchmark) -> None:
    """Benchmark alpha equivalence of two equal 100-addition trees."""
    left = _build_derived_tree(_DERIVED_TREE_DEPTH)
    right = _build_derived_tree(_DERIVED_TREE_DEPTH)
    assert benchmark(left.is_alpha_equivalent, right)


def test_derived_alpha_equivalence_of_binders(benchmark: Benchmark) -> None:
    r"""Benchmark `\x. x` against `\y. y` through `compared_as_binder`."""
    x, y = Identifier("x"), Identifier("y")
    left = _DerivedLam(x, _DerivedVar(x))
    right = _DerivedLam(y, _DerivedVar(y))
    assert benchmark(left.is_alpha_equivalent, right)


def test_derived_equivalence_over_expressions(benchmark: Benchmark) -> None:
    """Benchmark a binder over a 100-operation expression, renaming the binder."""
    left_identifiers = _build_identifiers(4, "u")
    right_identifiers = (Identifier("w"), *left_identifiers[1:])
    left = _DerivedExpressionHolder(
        left_identifiers[0], _build_deep_tree(left_identifiers, _DEEP_TREE_DEPTH)
    )
    right = _DerivedExpressionHolder(
        right_identifiers[0], _build_deep_tree(right_identifiers, _DEEP_TREE_DEPTH)
    )
    assert benchmark(left.is_alpha_equivalent, right)


def test_mapping_helper(benchmark: Benchmark) -> None:
    """Benchmark the identifier-keyed mapping helper over 50 renamed entries."""
    left_keys = _build_identifiers(_MAPPING_SIZE, "a")
    right_keys = _build_identifiers(_MAPPING_SIZE, "b")
    left = {key: _Leaf(index) for index, key in enumerate(left_keys)}
    right = {key: _Leaf(index) for index, key in enumerate(right_keys)}
    renaming = AlphaRenaming.with_free_renaming(
        dict(zip(left_keys, right_keys, strict=True))
    )
    assert benchmark(
        is_identifier_mapping_alpha_equivalent_under, left, right, renaming
    )


# ---------------------------------------------------------------------------
# Consumers
# ---------------------------------------------------------------------------


def _build_param(kind: str) -> Param[int]:
    """Return a fresh param of `kind`."""
    if kind == "integer":
        return create_integer_param()
    return create_natural_param_between(1, 10)


@pytest.mark.parametrize("kind", ["integer", "natural_between"])
def test_param_alpha_equivalence(benchmark: Benchmark, kind: str) -> None:
    """Benchmark two params of one kind over distinct variables."""
    left = _build_param(kind)
    right = _build_param(kind)
    assert benchmark(left.is_alpha_equivalent, right)


def test_param_structural_equivalence(benchmark: Benchmark) -> None:
    """Benchmark a bounded natural param against itself, structurally."""
    param = _build_param("natural_between")
    assert benchmark(param.is_structurally_equivalent, param)


def test_param_construction_between_bounds(benchmark: Benchmark) -> None:
    """Benchmark building a bounded integer param, which dedupes its constraints."""
    benchmark(create_integer_param_between, 0, 10)


def test_constraint_structural_equivalence(benchmark: Benchmark) -> None:
    """Benchmark two equal equation constraints built apart."""
    x = Identifier("x")
    left = EquationConstraint(IdentifierExpression(x) > 0)
    right = EquationConstraint(IdentifierExpression(x) > 0)
    assert benchmark(left.is_structurally_equivalent, right)


def test_symbol_table_structural_equivalence(benchmark: Benchmark) -> None:
    """Benchmark two equal symbol tables of 20 variables."""
    names = _build_identifiers(_SYMBOL_TABLE_SIZE, "s")
    left = _build_symbol_table(names)
    right = _build_symbol_table(names)
    assert benchmark(left.is_structurally_equivalent, right)


@pytest.mark.parametrize("depth", [1, _FRAME_DEPTH])
def test_expression_alpha_equivalence_under_frames(
    benchmark: Benchmark, depth: int
) -> None:
    """Benchmark `x + z` against `y + z` under a renaming of `depth` frames."""
    x, y, z = Identifier("x"), Identifier("y"), Identifier("z")
    renaming = AlphaRenaming.empty()
    for left, right in zip(
        _build_identifiers(depth - 1, "l"),
        _build_identifiers(depth - 1, "r"),
        strict=True,
    ):
        renaming = renaming.extend({left: right})
    renaming = renaming.extend({x: y})
    left_expression = IdentifierExpression(x) + IdentifierExpression(z)
    right_expression = IdentifierExpression(y) + IdentifierExpression(z)
    assert benchmark(
        left_expression.is_alpha_equivalent_under, right_expression, renaming
    )
