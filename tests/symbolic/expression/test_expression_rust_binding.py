"""Tests for the Python interface over the Rust-backed expressions.

The expression classes are thin Python subclasses of ``fhy_core._rs``
classes (pattern P2 of ``docs/design/python-switch.md``).
These tests cover what the binding adds around the core: the class
structure and the public-class registration, argument checks, the child
objects a node keeps, deep trees, and the protocols the classes stand in
for.
"""

import math
import pickle
import re
import time
import weakref
from decimal import Decimal

import pytest

from fhy_core import _rs
from fhy_core.identifier import Identifier
from fhy_core.serialization import SerializedDict
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


@pytest.mark.parametrize(
    "value",
    [
        math.nan,
        -0.0,
        1e300,
        2**80 + 7,
        -(2**70),
        Decimal("3.125"),
        Decimal("100"),
        True,
        0,
    ],
    ids=[
        "nan",
        "negative_zero",
        "huge_float",
        "big_int",
        "negative_big_int",
        "decimal",
        "whole_decimal",
        "bool",
        "zero",
    ],
)
def test_a_decoded_literal_is_the_constructed_one(value: object) -> None:
    """Test a V2-decoded literal equals, hashes, prints and reads as the built one.

    The decoder builds a literal from the core's decoded value, without the
    public constructor's parsing, and computes ``value`` on its first read.
    """
    constructed = LiteralExpression(value)  # type: ignore[arg-type]

    decoded = Expression.deserialize_from_dict(constructed.serialize_to_dict())

    assert type(decoded) is LiteralExpression
    assert decoded == constructed
    assert hash(decoded) == hash(constructed)
    assert str(decoded) == str(constructed)
    assert repr(decoded) == repr(constructed)
    assert type(decoded.value) is type(constructed.value)
    assert decoded.value is decoded.value
    if isinstance(value, float) and math.isnan(value):
        assert math.isnan(decoded.value)
    else:
        assert decoded.value == constructed.value
        assert str(decoded.value) == str(constructed.value)
    assert pickle.loads(pickle.dumps(decoded)) == constructed
    assert decoded.serialize_to_dict() == constructed.serialize_to_dict()


def test_a_decoded_table_shares_its_repeated_nodes_and_identifiers() -> None:
    """Test decoding builds a repeated node once, and one ``Identifier`` per id."""
    x = Identifier("x")
    twin = Identifier.deserialize_from_dict(x.serialize_to_dict())
    tree = (IdentifierExpression(x) + 1) * (IdentifierExpression(twin) + 1)

    decoded = Expression.deserialize_from_dict(tree.serialize_to_dict())

    assert isinstance(decoded, BinaryExpression)
    assert decoded == tree
    assert decoded.left is decoded.right
    assert isinstance(decoded.left, BinaryExpression)
    assert isinstance(decoded.left.left, IdentifierExpression)
    assert decoded.left.left.identifier == x


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"nodes": []}, "expression payload has no nodes"),
        (
            {"nodes": [{"literal": {"int": "1"}}, {"literal": {"int": "2"}}]},
            "node 0 is not referenced",
        ),
        (
            {"nodes": [{"unary": {"operation": "negate", "operand": 0}}]},
            "node 0 refers to node 0, which does not precede it",
        ),
        ({"nodes": [{"literal": {"float": "1e5"}}]}, "not canonical"),
        ({"nodes": [{"literal": {"int": "01"}}]}, 'invalid integer literal "01"'),
        (
            {
                "nodes": [
                    {"literal": {"int": "1"}},
                    {"logical": {"operation": "and", "operands": [0]}},
                ]
            },
            "logical node 1 has 1 operands",
        ),
    ],
    ids=["empty", "unreferenced", "forward", "float_text", "int_text", "one_operand"],
)
def test_a_table_the_fast_path_declines_raises_the_cores_error(
    payload: SerializedDict, message: str
) -> None:
    """Test a malformed V2 table raises the core decoder's error."""
    from fhy_core.serialization import DeserializationValueError  # noqa: PLC0415

    with pytest.raises(DeserializationValueError, match=re.escape(message)):
        Expression.deserialize_from_dict(payload)


def test_str_of_a_decoded_doubling_dag_is_bounded() -> None:
    """Test ``str`` of a decoded 61-node payload of 2**61 - 1 occurrences.

    Above a million occurrences, ``str`` writes the first thousand and
    ``…`` instead of the exponential full text; ``repr`` is bounded too.
    """
    x = Identifier("x")
    node: Expression = IdentifierExpression(x)
    for _ in range(60):
        node = node + node
    decoded = Expression.deserialize_from_dict(node.serialize_to_dict())

    started = time.perf_counter()
    text = str(decoded)
    representation = repr(decoded)
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0
    assert text.startswith("((((")
    assert text.endswith("…")
    assert len(text) < 10_000
    assert len(representation) < 20_000
    assert str(IdentifierExpression(x) + 1) == "(x + 1)"


@pytest.mark.usefixtures("v1_wire")
def test_a_deep_payload_decodes_in_one_pass() -> None:
    """Test decoding a payload deeper than the recursion limit.

    The binding decodes an expression payload of its own shapes in one
    pass with its pending nodes on the heap; the framework's per-node path
    would recurse once per level.
    """
    x = Identifier("x")
    tree: Expression = IdentifierExpression(x)
    payload: SerializedDict = tree.serialize_to_dict()
    for _ in range(5_000):
        payload = {
            "__type__": "unary_expression",
            "__data__": {"operation": "negate", "operand": payload},
        }
        tree = -tree

    assert Expression.deserialize_from_dict(payload) == tree


def test_a_payload_the_fast_path_declines_raises_the_framework_error() -> None:
    """Test a malformed payload still raises the serialization framework's error."""
    from fhy_core.serialization import DeserializationValueError  # noqa: PLC0415

    payload = {
        "__type__": "binary_expression",
        "__data__": {
            "operation": "logical_and",
            "left": LiteralExpression(True).serialize_to_dict(),
            "right": LiteralExpression(False).serialize_to_dict(),
        },
    }

    with pytest.raises(DeserializationValueError, match="a valid BinaryOperation"):
        Expression.deserialize_from_dict(payload)  # type: ignore[arg-type]


def test_decoding_through_a_node_class_refuses_another_kind() -> None:
    """Test ``Node.deserialize_from_dict`` refuses a payload of another node kind."""
    from fhy_core.serialization import SerializationError  # noqa: PLC0415

    with pytest.raises(SerializationError, match="not a subclass"):
        BinaryExpression.deserialize_from_dict(LiteralExpression(1).serialize_to_dict())


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
