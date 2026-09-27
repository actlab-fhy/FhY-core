"""Tests for the Python interface over the Rust-backed term package.

``AlphaRenaming`` is ``fhy_core._rs.AlphaRenaming``, over the Rust core's
``fhy_core::term::AlphaRenaming``; ``BinderMixin``'s derived methods run the
core's ``Binder`` algorithms over a node's Python hooks; and
``DerivedEquivalenceMixin``'s engine runs in the binding (S10 of
``docs/design/python-switch.md``). These tests cover what the binding adds
around the core: the renaming's class structure, argument checks, value
semantics and object identity; which hooks a binder's derived methods call,
how their exceptions and results are handled, and the renamings they pass
on; the derived engine's plans, native paths, depth, callbacks and errors;
and the mapping helper's order.
"""

import copy
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from fhy_core import _rs
from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression import (
    Expression,
    IdentifierExpression,
    LiteralExpression,
    RegisteredFunction,
)
from fhy_core.symbolic.expression.sort import FunctionSort
from fhy_core.term import (
    AlphaEquivalenceMixin,
    AlphaRenaming,
    BinderMixin,
    DerivedEquivalenceMixin,
    EquivalenceDerivationError,
    Term,
    compared_as_binder,
    compared_as_reference,
    compared_as_value,
    compared_with,
    excluded_from_equivalence,
    is_identifier_mapping_alpha_equivalent_under,
)
from fhy_core.term.derived_equivalence import (
    _PLAN_CACHE,
    EQUIVALENCE_METADATA_KEY,
)
from fhy_core.traits import FrozenMixin, FrozenMutationError
from fhy_core.utils.override import override

# ===========================================================================
# Helpers
# ===========================================================================


@dataclass(frozen=True)
class _Var(AlphaEquivalenceMixin):
    """A reference that records the renamings it is compared under."""

    identifier: Identifier
    renamings: list[AlphaRenaming] = field(
        default_factory=list, compare=False, hash=False
    )

    def get_free_identifiers(self) -> frozenset[Identifier]:
        return frozenset({self.identifier})

    @override
    def is_alpha_equivalent_under(self, other: object, renaming: AlphaRenaming) -> bool:
        self.renamings.append(renaming)
        return isinstance(other, _Var) and renaming.are_identifiers_alpha_equivalent(
            self.identifier, other.identifier
        )

    def substitute(self, replacements: Mapping[Identifier, Term]) -> Term:
        return replacements.get(self.identifier, self)


class _Answer(AlphaEquivalenceMixin):
    """A child answering every comparison with a fixed value, or raising."""

    def __init__(self, answer: object, error: BaseException | None = None) -> None:
        self.answer = answer
        self.error = error
        self.calls = 0

    def get_free_identifiers(self) -> frozenset[Identifier]:
        return frozenset()

    @override
    def is_alpha_equivalent_under(self, other: object, renaming: AlphaRenaming) -> Any:
        del other, renaming
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.answer

    def substitute(self, replacements: Mapping[Identifier, Term]) -> Term:
        del replacements
        return self


class _Lam(BinderMixin):
    """A binder that records each hook call, and can be told to raise."""

    def __init__(
        self,
        parameters: Sequence[object],
        body: Sequence[object],
        raising: dict[str, BaseException] | None = None,
    ) -> None:
        self.parameters = tuple(parameters)
        self.body = tuple(body)
        self.raising = raising or {}
        self.calls: list[str] = []

    def _record(self, hook: str) -> None:
        self.calls.append(hook)
        if hook in self.raising:
            raise self.raising[hook]

    @override
    def get_bound_identifiers(self) -> Sequence[Identifier]:
        self._record("get_bound_identifiers")
        return self.parameters  # type: ignore[return-value]

    @override
    def get_scoped_children(self) -> Sequence[Term]:
        self._record("get_scoped_children")
        return self.body  # type: ignore[return-value]

    @override
    def rename_bound_identifier(self, old: Identifier, new: Identifier) -> "_Lam":
        self._record("rename_bound_identifier")
        parameters = tuple(new if p == old else p for p in self.parameters)
        body = tuple(child.substitute({old: _Var(new)}) for child in self.body)  # type: ignore[attr-defined]
        return _Lam(parameters, body)

    @override
    def rebuild_with_scoped_children(self, new_children: Sequence[Term]) -> "_Lam":
        self._record("rebuild_with_scoped_children")
        return _Lam(self.parameters, new_children)


# ===========================================================================
# AlphaRenaming: class structure and arguments
# ===========================================================================


def test_alpha_renaming_is_the_extension_class() -> None:
    """Test the public class is the Rust-backed class itself."""
    assert AlphaRenaming is _rs.AlphaRenaming
    assert isinstance(AlphaRenaming.empty(), FrozenMixin)


def test_alpha_renaming_is_final() -> None:
    """Test the renaming cannot be subclassed."""
    with pytest.raises(TypeError):
        type("_Sub", (AlphaRenaming,), {})


@pytest.mark.parametrize("action", ["modify", "delete"])
def test_alpha_renaming_mutation_raises_frozen_mutation_error(action: str) -> None:
    """Test mutating a renaming raises ``FrozenMutationError``."""
    renaming = AlphaRenaming.empty()
    with pytest.raises(FrozenMutationError, match=f'Cannot {action} "x"'):
        if action == "modify":
            renaming.x = 1
        else:
            del renaming.x  # type: ignore[attr-defined]
    assert renaming.is_frozen


def test_alpha_renaming_refuses_a_non_mapping() -> None:
    """Test a free renaming or frame that is not a mapping raises ``TypeError``."""
    with pytest.raises(TypeError, match="free_renaming must be a mapping, got list"):
        AlphaRenaming.with_free_renaming([])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bindings must be a mapping, got int"):
        AlphaRenaming.empty().extend(3)  # type: ignore[arg-type]


@pytest.mark.parametrize("position", ["key", "value"])
def test_alpha_renaming_refuses_a_non_identifier(position: str) -> None:
    """Test a key or value that is not an ``Identifier`` raises ``TypeError``."""
    x = Identifier("x")
    mapping: dict[Any, Any] = {"x": x} if position == "key" else {x: "x"}
    with pytest.raises(
        TypeError, match=f"AlphaRenaming {position} must be an Identifier, got str"
    ):
        AlphaRenaming.with_free_renaming(mapping)
    with pytest.raises(
        TypeError, match="AlphaRenaming identifier must be an Identifier"
    ):
        AlphaRenaming.empty().resolve("x")  # type: ignore[arg-type]


def test_alpha_renaming_refusals_name_the_part_that_is_not_injective() -> None:
    """Test the ``ValueError`` of a non-injective map names its part."""
    a, b, target = Identifier("a"), Identifier("b"), Identifier("target")
    with pytest.raises(
        ValueError, match=r"^a free-identifier renaming must be injective, "
    ):
        AlphaRenaming.with_free_renaming({a: target, b: target})
    with pytest.raises(
        ValueError,
        match=(
            r"^a binder frame must be injective, but more than one identifier "
            rf"maps to target::{target.id}$"
        ),
    ):
        AlphaRenaming.empty().extend({a: target, b: target})


# ===========================================================================
# AlphaRenaming: values
# ===========================================================================


def test_alpha_renaming_empty_is_one_object() -> None:
    """Test ``empty()`` returns one shared renaming."""
    assert AlphaRenaming.empty() is AlphaRenaming.empty()
    assert AlphaRenaming.with_free_renaming({}) == AlphaRenaming.empty()
    assert AlphaRenaming.empty().extend({}) != AlphaRenaming.empty()


def test_alpha_renaming_equality_and_hash_ignore_the_construction_order() -> None:
    """Test equal renamings built in different orders are equal and hash alike."""
    a, b, c, d = (Identifier(name) for name in "abcd")
    left = AlphaRenaming.with_free_renaming({a: b, c: d}).extend({b: a})
    right = AlphaRenaming.with_free_renaming({c: d, a: b}).extend({b: a})

    assert left == right
    assert hash(left) == hash(right)
    assert left != object()


def test_alpha_renaming_repr_lists_the_frames_and_the_free_renaming() -> None:
    """Test the ``repr`` shows the frames, outermost first, and the free renaming."""
    a, b, x, y = (Identifier(name) for name in "abxy")
    renaming = AlphaRenaming.with_free_renaming({a: b}).extend({x: y}).extend({})

    assert repr(renaming) == (
        f"AlphaRenaming(frames=[{{x::{x.id}: y::{y.id}}}, {{}}], "
        f"free_renaming={{a::{a.id}: b::{b.id}}})"
    )


@pytest.mark.parametrize(
    "duplicate",
    [pickle.loads, copy.copy, copy.deepcopy],
    ids=["pickle", "copy", "deepcopy"],
)
def test_alpha_renaming_round_trips(duplicate: Any) -> None:
    """Test a renaming survives pickling and copying with its identifiers."""
    a, b, x, y = (Identifier(name) for name in "abxy")
    renaming = AlphaRenaming.with_free_renaming({a: b}).extend({x: y})
    argument = pickle.dumps(renaming) if duplicate is pickle.loads else renaming

    restored = duplicate(argument)

    assert restored == renaming
    assert restored.resolve(x) == y
    assert restored.resolve(a) == b
    assert restored.are_identifiers_alpha_equivalent(x, y)


def test_alpha_renaming_resolve_returns_the_objects_given() -> None:
    """Test ``resolve`` returns the image object given, or its argument."""
    a, b, x, y, z = (Identifier(name) for name in "abxyz")
    renaming = AlphaRenaming.with_free_renaming({a: b}).extend({x: y})

    assert renaming.resolve(x) is y
    assert renaming.resolve(a) is b
    assert renaming.resolve(z) is z


def test_expressions_and_registry_entries_read_the_renaming_directly() -> None:
    """Test expressions and function entries read the renaming, refusing others."""
    x, y = Identifier("x"), Identifier("y")
    renaming = AlphaRenaming.empty().extend({x: y})
    left = IdentifierExpression(x) + 1
    right = IdentifierExpression(y) + 1
    function = RegisteredFunction(
        "f", [x], [FunctionSort.REAL], FunctionSort.REAL, left
    )
    renamed = RegisteredFunction(
        "g", [y], [FunctionSort.REAL], FunctionSort.REAL, right
    )

    assert left.is_alpha_equivalent_under(right, renaming)
    assert function.is_alpha_equivalent_under(renamed, AlphaRenaming.empty())
    with pytest.raises(TypeError, match="renaming must be an AlphaRenaming, got dict"):
        left.is_alpha_equivalent_under(right, {})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="renaming must be an AlphaRenaming, got dict"):
        function.is_alpha_equivalent_under(renamed, {})  # type: ignore[arg-type]


# ===========================================================================
# BinderMixin: hooks
# ===========================================================================


def test_binder_comparison_reads_each_hook_once_per_node() -> None:
    """Test a comparison reads each binder's bound identifiers, then children."""
    x, y = Identifier("x"), Identifier("y")
    left = _Lam([x], [_Var(x)])
    right = _Lam([y], [_Var(y)])

    assert left.is_alpha_equivalent(right)
    assert left.calls == ["get_bound_identifiers", "get_scoped_children"]
    assert right.calls == ["get_bound_identifiers", "get_scoped_children"]


def test_binder_comparison_reads_no_children_when_the_arities_differ() -> None:
    """Test binders of different arities are compared without their children."""
    x, y, z = Identifier("x"), Identifier("y"), Identifier("z")
    left = _Lam([x], [_Var(x)])
    right = _Lam([y, z], [_Var(y)])

    assert not left.is_alpha_equivalent(right)
    assert left.calls == ["get_bound_identifiers"]
    assert right.calls == ["get_bound_identifiers"]


def test_binder_substitution_calls_the_rebuild_hooks_it_needs() -> None:
    """Test a substitution renames only a capturing binder, then rebuilds."""
    x, y, z = Identifier("x"), Identifier("y"), Identifier("z")
    plain = _Lam([x], [_Var(y)])
    capturing = _Lam([x], [_Var(y)])

    plain_result = plain.substitute({y: _Var(z)})
    captured_result = capturing.substitute({y: _Var(x)})

    assert plain.calls == [
        "get_bound_identifiers",
        "get_scoped_children",
        "rebuild_with_scoped_children",
    ]
    assert capturing.calls[-1] == "rename_bound_identifier"
    assert isinstance(plain_result, _Lam)
    assert plain_result.parameters == (x,)
    assert isinstance(captured_result, _Lam)
    fresh = captured_result.parameters[0]
    assert isinstance(fresh, Identifier)
    assert fresh != x
    assert fresh.name_hint == "x"
    assert captured_result.get_free_identifiers() == frozenset({x})


def test_binder_substitution_of_a_key_it_does_not_mention_renames_nothing() -> None:
    """Test a key absent from the binder leaves it as itself, whatever it maps to."""
    x, y, z = Identifier("x"), Identifier("y"), Identifier("z")
    binder = _Lam([y], [_Var(x)])

    result = binder.substitute({z: _Var(y)})

    assert result is binder
    assert binder.calls == ["get_bound_identifiers", "get_scoped_children"]


def test_binder_substitution_that_applies_nothing_returns_the_binder_itself() -> None:
    """Test a substitution of only bound keys returns the node itself."""
    x, y = Identifier("x"), Identifier("y")
    binder = _Lam([x], [_Var(x)])

    assert binder.substitute({x: _Var(y)}) is binder
    assert binder.substitute({}) is binder


def test_binder_free_identifiers_are_the_objects_the_children_returned() -> None:
    """Test the free identifiers are the children's own objects."""
    x, y = Identifier("x"), Identifier("y")
    free = _Lam([x], [_Var(x), _Var(y)]).get_free_identifiers()

    assert free == frozenset({y})
    assert next(iter(free)) is y


@pytest.mark.parametrize("hook", ["get_bound_identifiers", "get_scoped_children"])
def test_binder_hook_exception_propagates_as_the_same_object(hook: str) -> None:
    """Test an exception a hook raises reaches the caller itself."""
    x, y = Identifier("x"), Identifier("y")
    error = LookupError("from the hook")
    left = _Lam([x], [_Var(x)], raising={hook: error})
    right = _Lam([y], [_Var(y)])

    with pytest.raises(LookupError) as raised:
        left.is_alpha_equivalent(right)
    assert raised.value is error
    with pytest.raises(LookupError) as raised_again:
        left.get_free_identifiers()
    assert raised_again.value is error
    assert right.calls == (
        [] if hook == "get_bound_identifiers" else ["get_bound_identifiers"]
    )


def test_binder_rebuild_exception_propagates_from_a_substitution() -> None:
    """Test an exception from ``rebuild_with_scoped_children`` propagates."""
    x, y, z = Identifier("x"), Identifier("y"), Identifier("z")
    error = RuntimeError("cannot rebuild")
    binder = _Lam([x], [_Var(y)], raising={"rebuild_with_scoped_children": error})

    with pytest.raises(RuntimeError) as raised:
        binder.substitute({y: _Var(z)})
    assert raised.value is error


def test_binder_child_exception_stops_the_comparison_at_that_child() -> None:
    """Test a child that raises stops the comparison, as ``all`` did."""
    x, y = Identifier("x"), Identifier("y")
    error = ValueError("child")
    first, second = _Answer(True, error), _Answer(True)
    left = _Lam([x], [first, second])
    right = _Lam([y], [_Answer(True), _Answer(True)])

    with pytest.raises(ValueError) as raised:
        left.is_alpha_equivalent(right)
    assert raised.value is error
    assert (first.calls, second.calls) == (1, 0)


def test_binder_keyboard_interrupt_passes_through() -> None:
    """Test a ``KeyboardInterrupt`` from a hook is not wrapped."""
    x, y = Identifier("x"), Identifier("y")
    left = _Lam([x], [_Var(x)], raising={"get_scoped_children": KeyboardInterrupt()})

    with pytest.raises(KeyboardInterrupt):
        left.is_alpha_equivalent(_Lam([y], [_Var(y)]))


def test_binder_bound_identifier_of_the_wrong_type_raises_type_error() -> None:
    """Test a bound identifier that is not an ``Identifier`` raises ``TypeError``."""
    left = _Lam(["x"], [])
    with pytest.raises(
        TypeError, match="BinderMixin bound identifier must be an Identifier, got str"
    ):
        left.is_alpha_equivalent(_Lam([Identifier("y")], []))


def test_binder_reads_a_child_answer_by_truthiness() -> None:
    """Test a truthy child answer counts as equivalent, a falsy one as not."""
    x, y = Identifier("x"), Identifier("y")

    assert _Lam([x], [_Answer(1)]).is_alpha_equivalent(_Lam([y], [_Answer(1)]))
    assert not _Lam([x], [_Answer([])]).is_alpha_equivalent(_Lam([y], [_Answer(1)]))


def test_binder_over_an_expression_compares_it_as_the_expression_does() -> None:
    """Test an expression child answers as its own method under the frame."""
    x, y, z = Identifier("x"), Identifier("y"), Identifier("z")
    left = _Lam([x], [IdentifierExpression(x) + IdentifierExpression(z)])
    right = _Lam([y], [IdentifierExpression(y) + IdentifierExpression(z)])
    capturing = _Lam([y], [IdentifierExpression(x) + IdentifierExpression(z)])

    assert left.is_alpha_equivalent(right)
    assert not left.is_alpha_equivalent(capturing)


def test_binder_children_receive_the_extended_renaming() -> None:
    """Test a child is compared under the renaming with the binder's frame."""
    x, y, a, b = (Identifier(name) for name in "xyab")
    child = _Var(x)
    outer = AlphaRenaming.with_free_renaming({a: b})

    assert _Lam([x], [child]).is_alpha_equivalent_under(_Lam([y], [_Var(y)]), outer)
    (received,) = child.renamings
    assert received == outer.extend({x: y})
    assert received.resolve(x) is y
    assert received.resolve(a) is b


def test_binder_repeating_a_parameter_matches_no_binder() -> None:
    """Test a repeated bound identifier pairs with nothing, in both directions."""
    x, a, b = Identifier("x"), Identifier("a"), Identifier("b")
    repeating = _Lam([x, x], [_Var(x)])
    distinct = _Lam([a, b], [_Var(b)])

    assert not repeating.is_alpha_equivalent(distinct)
    assert not distinct.is_alpha_equivalent(repeating)
    assert not repeating.is_alpha_equivalent(repeating)


# ===========================================================================
# DerivedEquivalenceMixin
# ===========================================================================


@dataclass(frozen=True, eq=False)
class _Node(DerivedEquivalenceMixin):
    """A derived chain node."""

    value: int
    next: "_Node | None" = None


@dataclass(frozen=True, eq=False)
class _Ref(DerivedEquivalenceMixin):
    """A derived reference."""

    identifier: Identifier = field(metadata=compared_as_reference())


@dataclass(frozen=True, eq=False)
class _Binds(DerivedEquivalenceMixin):
    """A derived binder over a body."""

    parameters: tuple[Identifier, ...] = field(
        metadata=compared_as_binder(scopes_over=("body",))
    )
    body: object


def _build_chain(depth: int) -> _Node:
    node = _Node(0)
    for value in range(1, depth):
        node = _Node(value, node)
    return node


def test_derived_plan_is_built_once_and_kept_in_the_plan_cache() -> None:
    """Test the first comparison caches a plan the next one reuses."""

    @dataclass(frozen=True, eq=False)
    class _Fresh(DerivedEquivalenceMixin):
        value: int

    assert _Fresh not in _PLAN_CACHE
    assert _Fresh(1).is_structurally_equivalent(_Fresh(1))
    plan = _PLAN_CACHE[_Fresh]
    assert type(plan).__name__ == "EquivalencePlan"
    assert _Fresh(1).is_alpha_equivalent(_Fresh(1))
    assert _PLAN_CACHE[_Fresh] is plan


@pytest.mark.parametrize("mode", ["structural", "alpha"])
def test_derived_comparison_of_a_deep_chain_uses_no_python_recursion(mode: str) -> None:
    """Test nested derived values 10,000 deep compare without ``RecursionError``."""
    left, right, different = (
        _build_chain(10_000),
        _build_chain(10_000),
        _build_chain(9_999),
    )

    if mode == "structural":
        assert left.is_structurally_equivalent(right)
        assert not left.is_structurally_equivalent(different)
    else:
        assert left.is_alpha_equivalent(right)
        assert not left.is_alpha_equivalent(different)


def test_derived_nested_value_with_its_own_method_is_asked() -> None:
    """Test a nested value whose class overrides the method has it called."""
    calls: list[AlphaRenaming] = []

    @dataclass(frozen=True, eq=False)
    class _Custom(DerivedEquivalenceMixin):
        value: int

        @override
        def is_alpha_equivalent_under(
            self, other: object, renaming: AlphaRenaming
        ) -> bool:
            calls.append(renaming)
            return True

    x, y = Identifier("x"), Identifier("y")

    assert _Binds((x,), _Custom(1)).is_alpha_equivalent(_Binds((y,), _Custom(2)))
    (received,) = calls
    assert received.resolve(x) is y


def test_derived_comparator_and_key_receive_their_arguments() -> None:
    """Test an explicit comparator gets the renaming in scope and a key both sides."""
    keyed: list[object] = []
    renamings: list[AlphaRenaming] = []

    class _Comparator:
        def is_structurally_equivalent(self, left: Any, right: Any) -> bool:
            return bool(left == right)

        def is_alpha_equivalent_under(
            self, left: Any, right: Any, renaming: AlphaRenaming
        ) -> bool:
            renamings.append(renaming)
            return bool(left == right)

    def _key(value: object) -> object:
        keyed.append(value)
        return value

    @dataclass(frozen=True, eq=False)
    class _Holder(DerivedEquivalenceMixin):
        parameter: Identifier = field(
            metadata=compared_as_binder(scopes_over=("custom",))
        )
        custom: int = field(metadata=compared_with(_Comparator()))
        keyed: int = field(metadata=compared_as_value(key=_key))

    x, y = Identifier("x"), Identifier("y")

    assert _Holder(x, 1, 2).is_alpha_equivalent(_Holder(y, 1, 2))
    assert keyed == [2, 2]
    (received,) = renamings
    assert received.resolve(x) is y


def test_derived_exceptions_from_user_code_propagate() -> None:
    """Test exceptions from a comparator and from ``==`` reach the caller."""

    class _Raising:
        def is_structurally_equivalent(self, left: Any, right: Any) -> bool:
            raise ArithmeticError("comparator")

        def is_alpha_equivalent_under(
            self, left: Any, right: Any, renaming: AlphaRenaming
        ) -> bool:
            raise ArithmeticError("comparator")

    class _BadEquality:
        supports_partial_equality = True

        @override
        def __eq__(self, other: object) -> bool:
            raise LookupError("eq")

        __hash__ = object.__hash__

    @dataclass(frozen=True, eq=False)
    class _Holder(DerivedEquivalenceMixin):
        value: object = field(metadata=compared_with(_Raising()))

    @dataclass(frozen=True, eq=False)
    class _Equality(DerivedEquivalenceMixin):
        value: object

    with pytest.raises(ArithmeticError, match="comparator"):
        _Holder(1).is_structurally_equivalent(_Holder(1))
    with pytest.raises(ArithmeticError, match="comparator"):
        _Holder(1).is_alpha_equivalent(_Holder(1))
    with pytest.raises(LookupError, match="eq"):
        _Equality(_BadEquality()).is_structurally_equivalent(_Equality(_BadEquality()))


def test_derived_expressions_and_identifiers_compare_as_their_methods_do() -> None:
    """Test expression and identifier fields answer as their own comparisons."""

    @dataclass(frozen=True, eq=False)
    class _Holder(DerivedEquivalenceMixin):
        parameter: Identifier = field(
            metadata=compared_as_binder(scopes_over=("body",))
        )
        body: Expression
        label: Identifier

    x, y, label = Identifier("x"), Identifier("y"), Identifier("label")
    left = _Holder(x, IdentifierExpression(x) + 1, label)
    right = _Holder(y, IdentifierExpression(y) + 1, label)
    copied = _Holder(
        x,
        IdentifierExpression(x) + 1,
        Identifier.deserialize_from_dict(label.serialize_to_dict()),
    )

    assert left.is_alpha_equivalent(right)
    assert not left.is_structurally_equivalent(right)
    assert left.is_structurally_equivalent(copied)
    assert not left.is_alpha_equivalent(
        _Holder(y, IdentifierExpression(y) + LiteralExpression(2), label)
    )


def test_derived_binder_repeating_an_identifier_matches_nothing() -> None:
    """Test a binder field repeating an identifier pairs with nothing, even unscoped."""

    @dataclass(frozen=True, eq=False)
    class _Unscoped(DerivedEquivalenceMixin):
        parameters: tuple[Identifier, ...] = field(metadata=compared_as_binder())

    x, a, b = Identifier("x"), Identifier("a"), Identifier("b")
    repeating = _Binds((x, x), _Ref(x))

    assert not repeating.is_alpha_equivalent(repeating)
    assert not repeating.is_alpha_equivalent(_Binds((a, b), _Ref(b)))
    assert not _Binds((a, b), _Ref(b)).is_alpha_equivalent(repeating)
    assert not _Unscoped((x, x)).is_alpha_equivalent(_Unscoped((x, x)))
    assert _Unscoped((a, b)).is_alpha_equivalent(_Unscoped((x, b)))
    assert repeating.is_structurally_equivalent(_Binds((x, x), _Ref(x)))


def test_derived_binder_of_the_wrong_type_raises_type_error() -> None:
    """Test a bound or referenced value that is not an ``Identifier`` raises."""
    with pytest.raises(
        TypeError, match="compared_as_binder identifier must be an Identifier, got str"
    ):
        _Binds(("x",), _Ref(Identifier("x"))).is_alpha_equivalent(  # type: ignore[arg-type]
            _Binds(("y",), _Ref(Identifier("y")))  # type: ignore[arg-type]
        )
    with pytest.raises(
        TypeError,
        match="compared_as_reference identifier must be an Identifier, got int",
    ):
        _Ref(1).is_alpha_equivalent(_Ref(2))  # type: ignore[arg-type]


def test_derived_alpha_equivalence_refuses_a_renaming_of_another_type() -> None:
    """Test the renaming must be an ``AlphaRenaming``."""
    with pytest.raises(TypeError, match="renaming must be an AlphaRenaming, got dict"):
        _Node(1).is_alpha_equivalent_under(_Node(1), {})  # type: ignore[arg-type]


def test_equivalence_roles_are_extension_values() -> None:
    """Test the role functions store ``_rs.EquivalenceRole`` values."""

    def _key(value: object) -> object:
        return value

    comparator = object()
    roles = {
        "value": compared_as_value(),
        "reference": compared_as_reference(),
        "binder": compared_as_binder(scopes_over=("body", "tail")),
        "excluded": excluded_from_equivalence(),
        "explicit": compared_with(comparator),  # type: ignore[arg-type]
    }

    for kind, metadata in roles.items():
        role = metadata[EQUIVALENCE_METADATA_KEY]
        assert isinstance(role, _rs.EquivalenceRole)
        assert role.kind == kind
    keyed = compared_as_value(key=_key)[EQUIVALENCE_METADATA_KEY]
    assert keyed.key is _key
    assert roles["binder"][EQUIVALENCE_METADATA_KEY].scopes_over == ("body", "tail")
    assert roles["explicit"][EQUIVALENCE_METADATA_KEY].comparator is comparator
    assert (
        repr(roles["reference"][EQUIVALENCE_METADATA_KEY])
        == "EquivalenceRole.reference()"
    )
    with pytest.raises(TypeError, match="scopes_over must hold str names, got int"):
        compared_as_binder(scopes_over=(1,))  # type: ignore[arg-type]


def test_derived_unknown_scope_name_raises_the_derivation_error() -> None:
    """Test a binder scoping over no field raises ``EquivalenceDerivationError``."""

    @dataclass(frozen=True, eq=False)
    class _Typo(DerivedEquivalenceMixin):
        parameter: Identifier = field(metadata=compared_as_binder(scopes_over=("bdy",)))
        body: int

    x = Identifier("x")
    with pytest.raises(
        EquivalenceDerivationError,
        match=(
            r'^Cannot derive equivalence for "_Typo": binder field "parameter" '
            r'declares scopes_over=\("bdy", \.\.\.\), but "bdy" is not a field of '
            r'"_Typo"\.$'
        ),
    ):
        _Typo(x, 1).is_structurally_equivalent(_Typo(x, 1))


# ===========================================================================
# The mapping helper
# ===========================================================================


def test_mapping_helper_compares_values_in_the_left_order() -> None:
    """Test values are compared in the left order, stopping at a difference."""
    a, b, c = (Identifier(name) for name in "abc")
    first, second, third = _Answer(True), _Answer(False), _Answer(True)
    left = {c: first, a: second, b: third}
    right = {a: _Answer(True), b: _Answer(True), c: _Answer(True)}

    assert not is_identifier_mapping_alpha_equivalent_under(
        left, right, AlphaRenaming.empty()
    )
    assert (first.calls, second.calls, third.calls) == (1, 1, 0)


def test_mapping_helper_value_exception_propagates() -> None:
    """Test an exception from a value's comparison reaches the caller itself."""
    a = Identifier("a")
    error = ZeroDivisionError("value")

    with pytest.raises(ZeroDivisionError) as raised:
        is_identifier_mapping_alpha_equivalent_under(
            {a: _Answer(True, error)}, {a: _Answer(True)}, AlphaRenaming.empty()
        )
    assert raised.value is error


def test_mapping_helper_refuses_a_key_that_is_not_an_identifier() -> None:
    """Test a key that is not an ``Identifier`` raises ``TypeError``."""
    with pytest.raises(TypeError, match="key must be an Identifier, got str"):
        is_identifier_mapping_alpha_equivalent_under(
            {"a": 1},  # type: ignore[dict-item]
            {"a": 1},  # type: ignore[dict-item]
            AlphaRenaming.empty(),
        )
