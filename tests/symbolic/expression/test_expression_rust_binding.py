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
from collections.abc import Callable
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
from fhy_core.utils.override import override

from ...v1 import reads_v1

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


@reads_v1
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


class _KeyRaisingOnce(str):
    """A ``str`` key whose first comparison raises ``exception``.

    Its hash is the ``str`` hash, so a lookup of the same text compares
    with it, and every comparison after the first is the ``str`` one.
    """

    __slots__ = ("exception", "has_raised")

    exception: BaseException
    has_raised: bool

    def __new__(cls, text: str, exception: BaseException) -> "_KeyRaisingOnce":
        key = super().__new__(cls, text)
        key.exception = exception
        key.has_raised = False
        return key

    @override
    def __eq__(self, other: object) -> bool:
        if not self.has_raised:
            self.has_raised = True
            raise self.exception
        return str.__eq__(self, other)

    __hash__ = str.__hash__


def _build_literal_payload_with_key(key: str) -> SerializedDict:
    """Return the V1 payload of ``LiteralExpression(1)`` with ``key`` for ``value``."""
    return {"__type__": "literal_expression", "__data__": {key: 1}}


@reads_v1
def test_a_payload_the_fast_path_fails_on_falls_back_to_the_framework() -> None:
    """Test an ``Exception`` in the fast path falls back to the framework's path."""
    key = _KeyRaisingOnce("value", ValueError("refused"))

    decoded = Expression.deserialize_from_dict(_build_literal_payload_with_key(key))

    assert key.has_raised
    assert decoded == LiteralExpression(1)


@reads_v1
@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_a_base_exception_in_the_fast_path_propagates(
    exception: type[BaseException],
) -> None:
    """Test a ``BaseException`` that is no ``Exception`` is not retried."""
    raised = exception()
    key = _KeyRaisingOnce("value", raised)

    with pytest.raises(exception) as exception_info:
        Expression.deserialize_from_dict(_build_literal_payload_with_key(key))

    assert exception_info.value is raised


def _build_negation_table_with_keys(
    nodes_key: str, operation_key: str
) -> SerializedDict:
    """Return the V2 table of ``-1`` with the given keys."""
    return {
        nodes_key: [
            {"literal": {"int": "1"}},
            {"unary": {operation_key: "negate", "operand": 0}},
        ]
    }


_TABLE_KEY_PLACES = ["table", "node"]


def _build_key_raising_at(place: str, exception: BaseException) -> _KeyRaisingOnce:
    """Return the key raising ``exception`` for the table key or a node's field."""
    return _KeyRaisingOnce("nodes" if place == "table" else "operation", exception)


def _build_table_payload_raising_at(place: str, key: _KeyRaisingOnce) -> SerializedDict:
    """Return the V2 table of ``-1`` with ``key`` at ``place``."""
    if place == "table":
        return _build_negation_table_with_keys(key, "operation")
    return _build_negation_table_with_keys("nodes", key)


@pytest.mark.parametrize("place", _TABLE_KEY_PLACES)
def test_a_table_the_fast_path_fails_on_falls_back_to_the_core(place: str) -> None:
    """Test an ``Exception`` in the V2 fast path falls back to the core's path."""
    key = _build_key_raising_at(place, ValueError())

    decoded = Expression.deserialize_from_dict(
        _build_table_payload_raising_at(place, key)
    )

    assert key.has_raised
    assert decoded == UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(1))


@pytest.mark.parametrize("place", _TABLE_KEY_PLACES)
@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_a_base_exception_in_the_table_fast_path_propagates(
    exception: type[BaseException], place: str
) -> None:
    """Test a ``BaseException`` in the V2 fast path is not retried."""
    raised = exception()
    key = _build_key_raising_at(place, raised)

    with pytest.raises(exception) as exception_info:
        Expression.deserialize_from_dict(_build_table_payload_raising_at(place, key))

    assert exception_info.value is raised


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


# =============================================================================
# Big integers (R2-045)
#
# Ints cross into the core as bytes, not decimal text, so CPython's
# 4,300-digit guard on int-to-text conversion does not apply.
# =============================================================================


def test_a_literal_of_an_int_past_the_digit_limit_is_built() -> None:
    """Test `LiteralExpression(10**5000)` builds and keeps its value."""
    value = 10**5000

    literal = LiteralExpression(value)

    assert literal.value == value
    assert LiteralExpression.from_json(literal.to_json()).value == value


def test_a_payload_of_a_5001_digit_int_materializes() -> None:
    """Test decoding a literal of 5,001 digits builds the Python int."""
    value = -(10**5000) - 7
    payload = LiteralExpression(value).serialize_to_dict()

    rebuilt = Expression.deserialize_from_dict(payload)

    assert isinstance(rebuilt, LiteralExpression)
    assert rebuilt.value == value
    assert type(rebuilt.value) is int


# Decimals cross as `as_tuple()` parts into `Decimal::from_parts`, which
# refuses an exponent beyond 10,000 in magnitude at once, where the text
# route expanded every digit first (R2-045).


@pytest.mark.parametrize(
    ("build", "exponent"),
    [
        pytest.param(
            lambda: LiteralExpression(Decimal("1e100000000")), 100000000, id="literal"
        ),
        pytest.param(
            lambda: LiteralExpression(Decimal("1e+5000000000")),
            5000000000,
            id="literal_p32",
        ),
        pytest.param(
            lambda: _rs.check_param_bounds_are_ordered(
                Decimal("1e-4000000000"), 1, True, True
            ),
            -4000000000,
            id="param_bound_p32",
        ),
        pytest.param(
            lambda: _rs.is_decimal_text_exactly_binary(Decimal("1e100000000")),
            100000000,
            id="exactly_binary",
        ),
        pytest.param(
            lambda: _rs.coerce_literal_value(Decimal("-1e100000000")),
            100000000,
            id="coerce",
        ),
    ],
)
def test_an_absurdly_scaled_decimal_is_refused_at_once(
    build: Callable[[], object], exponent: int
) -> None:
    """Test a `Decimal` beyond the exponent bound raises `ValueError` at once.

    The message names the exponent and the bound. The time bound is loose,
    for a loaded machine: before, the first case took 0.6 s and 235 MiB, and
    the two `p32` cases gave no answer within 15 s.
    """
    started = time.perf_counter()
    with pytest.raises(
        ValueError, match=rf"decimal exponent {exponent} exceeds the bound of 10000"
    ):
        build()

    assert time.perf_counter() - started < 1.0


@pytest.mark.parametrize(
    "text", ["1E+10000", "1E-10000", "1.50", "0", "-0", "12345678901234567890.5"]
)
def test_a_decimal_within_the_bound_keeps_its_value(text: str) -> None:
    """Test a `Decimal` up to the bound reads back as its normalized value."""
    value = Decimal(text)

    literal = LiteralExpression(value.copy_abs())
    read = literal.value

    assert isinstance(read, Decimal)
    assert read == value.copy_abs()
    assert value == 0 or read.as_tuple() == value.copy_abs().normalize().as_tuple()


def test_a_decimal_of_many_digits_is_read_exactly() -> None:
    """Test a `Decimal` of 20,000 digits within the bound keeps every one.

    Its exponent is -9,999, so its digits pass CPython's text limit while
    the exponent stays within the bound; one more fractional digit is
    refused, and the message does not repeat the value.
    """
    value = Decimal("1" * 10_001 + "." + "7" * 9_999)
    beyond = Decimal("0." + "7" * 10_001)

    assert LiteralExpression(value).value == value
    assert _rs.is_decimal_text_exactly_binary(value) is False
    with pytest.raises(ValueError, match="exponent -10001 exceeds") as refused:
        LiteralExpression(beyond)
    assert len(str(refused.value)) < 120
