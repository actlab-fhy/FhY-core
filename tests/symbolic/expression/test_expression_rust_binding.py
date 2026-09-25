"""Tests for the Python interface over the Rust-backed expressions.

On the Rust backend the expression classes are thin Python subclasses of
``fhy_core._rs`` classes (pattern P2 of ``docs/design/python-switch.md``).
These tests cover what the binding adds around the core: the class
structure and the public-class registration, argument checks, the child
objects a node keeps, deep trees, and the protocols the classes stand in
for.
"""

import pickle
import weakref

import pytest

import fhy_core
from fhy_core import _rs
from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression import (
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    LogicalExpression,
    LogicalOperation,
    PiecewiseExpression,
    UnaryExpression,
    UnaryOperation,
    logical_and,
    piecewise,
    validate_logical_operands,
)
from fhy_core.traits import FrozenMixin, HasOperands, VisitableMixin

pytestmark = pytest.mark.skipif(
    not fhy_core.RUST_BACKEND_SELECTED, reason="covers the Rust backend's binding"
)

_NODE_CLASSES = [
    (UnaryExpression, _rs.UnaryExpression),
    (BinaryExpression, _rs.BinaryExpression),
    (LogicalExpression, _rs.LogicalExpression),
    (IdentifierExpression, _rs.IdentifierExpression),
    (LiteralExpression, _rs.LiteralExpression),
    (PiecewiseExpression, _rs.PiecewiseExpression),
    (CallExpression, _rs.CallExpression),
]


# =============================================================================
# Class structure and registration
# =============================================================================


@pytest.mark.parametrize(("public_class", "rust_class"), _NODE_CLASSES)
def test_public_node_class_subclasses_its_rust_class_and_expression(
    public_class: type[Expression], rust_class: type
) -> None:
    """Test each public node class extends its ``_rs`` class and ``Expression``."""
    assert issubclass(public_class, rust_class)
    assert issubclass(public_class, Expression)
    assert issubclass(public_class, _rs.Expression)


@pytest.mark.parametrize(("public_class", "rust_class"), _NODE_CLASSES)
def test_registering_another_public_class_is_refused(
    public_class: type[Expression], rust_class: type
) -> None:
    """Test the registered public class never changes once registered."""

    class _Impostor(public_class):  # type: ignore[misc,valid-type]
        pass

    public_class._register_public_class()
    with pytest.raises(RuntimeError, match="registered already"):
        _Impostor._register_public_class()


def test_the_classes_stand_in_for_the_python_mixins() -> None:
    """Test the classes are virtual ``FrozenMixin`` and ``VisitableMixin`` subclasses.

    They implement those mixins' members themselves, so ``isinstance``
    against the mixins holds without the protocol metaclass the mixins
    would bring.
    """
    expression = IdentifierExpression(Identifier("x")) + 1

    assert isinstance(expression, FrozenMixin)
    assert isinstance(expression, VisitableMixin)
    assert isinstance(expression, HasOperands)
    assert type(type(expression)).__name__ == "ABCMeta"


def test_an_expression_can_be_weakly_referenced() -> None:
    """Test an expression supports weak references, as analysis caches need."""
    expression = LiteralExpression(1)

    reference = weakref.ref(expression)

    assert reference() is expression


def test_accept_hands_the_node_to_the_visitor() -> None:
    """Test ``accept`` returns the visitor's ``visit`` of the node."""

    class _Visitor:
        def visit(self, node: Expression) -> str:
            return type(node).__name__

    assert LiteralExpression(1).accept(_Visitor()) == "LiteralExpression"


# =============================================================================
# Argument checks
# =============================================================================


@pytest.mark.parametrize(
    ("build", "message"),
    [
        pytest.param(
            lambda: UnaryExpression(UnaryOperation.NEGATE, 1),  # type: ignore[arg-type]
            "UnaryExpression operand must be an Expression, got int.",
            id="unary_operand",
        ),
        pytest.param(
            lambda: BinaryExpression(BinaryOperation.ADD, LiteralExpression(1), None),  # type: ignore[arg-type]
            "BinaryExpression right must be an Expression, got NoneType.",
            id="binary_right",
        ),
        pytest.param(
            lambda: IdentifierExpression("x"),  # type: ignore[arg-type]
            "IdentifierExpression identifier must be an Identifier, got str.",
            id="identifier",
        ),
        pytest.param(
            lambda: CallExpression(b"f", ()),  # type: ignore[arg-type]
            "CallExpression function_name must be a str, got bytes.",
            id="call_name",
        ),
    ],
)
def test_a_field_of_the_wrong_type_is_refused(build: object, message: str) -> None:
    """Test a field of the wrong type raises ``TypeError`` naming it."""
    with pytest.raises(TypeError, match=message):
        build()  # type: ignore[operator]


def test_an_operation_is_read_from_its_member_or_its_value() -> None:
    """Test an operation may be given as its member or its value's text."""
    by_value = BinaryExpression("add", LiteralExpression(1), LiteralExpression(2))  # type: ignore[arg-type]

    assert by_value.operation is BinaryOperation.ADD
    with pytest.raises(ValueError, match="BinaryOperation"):
        BinaryExpression("logical_and", LiteralExpression(1), LiteralExpression(2))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="UnaryOperation"):
        UnaryExpression(BinaryOperation.ADD, LiteralExpression(1))  # type: ignore[arg-type]


def test_keyword_construction_names_the_fields() -> None:
    """Test every node constructs from its field names, as payload decoding does."""
    x = IdentifierExpression(identifier=Identifier("x"))

    expression = PiecewiseExpression(
        conditions=(x > 0,), values=(LiteralExpression(value=1),), otherwise=x
    )

    assert expression.otherwise is x


# =============================================================================
# Child objects
# =============================================================================


def test_a_node_keeps_the_objects_it_was_built_from() -> None:
    """Test field reads return the objects given, the same object every read."""
    left = IdentifierExpression(Identifier("x"))
    right = LiteralExpression(1)
    expression = BinaryExpression(BinaryOperation.ADD, left, right)

    assert expression.left is left
    assert expression.left is expression.left
    assert expression.get_visit_children()[1] is right


def test_a_substitution_result_reuses_the_objects_of_unchanged_subtrees() -> None:
    """Test the materialized result shares every subtree the core kept."""
    x = Identifier("x")
    y = Identifier("y")
    kept = IdentifierExpression(y) * 2
    replacement = LiteralExpression(3)
    tree = logical_and(IdentifierExpression(x) > kept, kept < 5)

    result = tree.substitute({x: replacement})

    assert isinstance(result, LogicalExpression)
    first = result.operands[0]
    assert isinstance(first, BinaryExpression)
    assert first.left is replacement
    assert first.right is kept
    assert result.operands[1] is tree.operands[1]


def test_a_rebuilt_node_holds_the_children_it_was_given() -> None:
    """Test ``rebuild_with_visit_children`` keeps the given child objects."""
    expression = UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(1))
    child = LiteralExpression(2)

    rebuilt = expression.rebuild_with_visit_children((child,))

    assert rebuilt.operand is child
    assert type(rebuilt) is UnaryExpression


# =============================================================================
# Deep trees
# =============================================================================


def test_a_deep_tree_compares_hashes_prints_and_screens_without_recursion() -> None:
    """Test the core's walks handle a tree deeper than Python's recursion limit.

    Equality, hashing, printing, free identifiers, substitution and the
    screen keep their pending nodes on the heap.
    """
    x = Identifier("x")
    depth = 20_000
    left: Expression = IdentifierExpression(x)
    right: Expression = IdentifierExpression(x)
    for _ in range(depth):
        left = left + 1
        right = right + 1

    assert left == right
    assert hash(left) == hash(right)
    assert str(left).startswith("(" * depth + "x + 1)")
    assert left.get_free_identifiers() == {x}
    assert left.substitute({x: LiteralExpression(0)}) != left
    validate_logical_operands(left)


def test_a_shared_dag_is_compared_and_hashed_in_time_linear_in_its_nodes() -> None:
    """Test a DAG of 2**60 occurrences compares and hashes by its 61 nodes."""
    x = Identifier("x")
    node: Expression = IdentifierExpression(x)
    twin: Expression = IdentifierExpression(x)
    other: Expression = IdentifierExpression(Identifier("y"))
    for _ in range(60):
        node = node + node
        twin = twin + twin
        other = other + other

    assert node == twin
    assert hash(node) == hash(twin)
    assert node != other


# =============================================================================
# Pickles
# =============================================================================


def test_an_expression_pickles_as_a_call_of_its_class() -> None:
    """Test a pickle round trip rebuilds an equal expression of the same class."""
    condition = LogicalExpression(
        LogicalOperation.AND,
        (IdentifierExpression(Identifier("x")) > 0, LiteralExpression(True)),
    )
    expression = piecewise((condition, 1), otherwise=LiteralExpression(0))

    restored = pickle.loads(pickle.dumps(expression))

    assert restored == expression
    assert type(restored) is PiecewiseExpression
    assert expression.__reduce__()[0] is PiecewiseExpression
