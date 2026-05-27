"""Unique identifier for named compiler objects."""

__all__ = ["Identifier"]

from threading import Lock
from typing import TYPE_CHECKING, Any, ClassVar, TypedDict, TypeGuard, final

from fhy_core.utils import is_strict_int

from .logger import get_logger
from .serialization import (
    DeserializationDictStructureError,
    DeserializationValueError,
    Serializable,
    SerializedDict,
    register_serializable,
)
from .trait.equality import EqualMixin
from .trait.frozen import FrozenMixin

if TYPE_CHECKING:
    from .expression import (
        BinaryExpression,
        BinaryOperation,
        UnaryExpression,
        UnaryOperation,
    )

_LOGGER = get_logger(__name__)


def _get_binary_operation_type() -> "type[BinaryOperation]":
    from .expression import BinaryOperation as _BinaryOperation  # noqa: PLC0415

    return _BinaryOperation


def _get_unary_operation_type() -> "type[UnaryOperation]":
    from .expression import UnaryOperation as _UnaryOperation  # noqa: PLC0415

    return _UnaryOperation


def _make_binary_expression(
    op: "BinaryOperation", left: Any, right: Any
) -> "BinaryExpression":
    from .expression import make_binary_expression  # noqa: PLC0415

    return make_binary_expression(op, left, right)


def _make_unary_expression(op: "UnaryOperation", operand: Any) -> "UnaryExpression":
    from .expression import make_unary_expression  # noqa: PLC0415

    return make_unary_expression(op, operand)


class _IdentifierData(TypedDict):
    id: int
    name_hint: str


_IDENTIFIER_DATA_KEYS: frozenset[str] = frozenset({"id", "name_hint"})


def _is_valid_identifier_data(data: SerializedDict) -> TypeGuard[_IdentifierData]:
    if data.keys() != _IDENTIFIER_DATA_KEYS:
        return False
    id_value = data["id"]
    if not is_strict_int(id_value):
        return False
    return isinstance(data["name_hint"], str)


@final
@register_serializable(type_id="id")
class Identifier(Serializable, FrozenMixin, EqualMixin, freeze_on_init=True):
    """Process-globally unique, named compiler symbol.

    Two ``Identifier`` instances are equal iff they share the same ``id``;
    ``name_hint`` is a debugging aid and is not consulted by ``__eq__`` or
    ``__hash__``. Ids are drawn from a single process-global,
    monotonically-increasing counter and are never reused.

    Construction and deserialization are thread-safe and share the same
    counter: a deserialized id cannot collide with a subsequently
    constructed id, regardless of interleaving. Deserializing an id
    greater than the next-to-be-issued value advances the counter past it.

    ``repr`` of an ``Identifier`` returns ``"<name_hint>::<id>"``. The form
    is for debugging only --- it is not a serialization protocol and is not
    round-trippable through the constructor. Use the structured ``id`` and
    ``name_hint`` properties (or ``serialize_to_dict``) when a parseable
    representation is needed.

    The arithmetic and comparison dunders (``+``, ``-``, ``*``, ``<``,
    ``>=``, etc.) build ``BinaryExpression`` / ``UnaryExpression`` nodes
    so callers can write ``a + b`` instead of wrapping each operand in
    ``IdentifierExpression``. Because the comparison dunders return
    expression nodes rather than ``bool``, sorting ``Identifier``
    instances is not supported via ``<`` / ``>``; callers must supply an
    explicit ``key=`` to ``sorted`` (for example, ``key=lambda i: i.id``
    for a stable ordering on the unique ID).

    The class is ``@final`` and is not intended to be subclassed; callers
    should treat it as a closed implementation that provides a single
    process-global id space.
    """

    _next_id: ClassVar[int] = 0
    _id_lock: ClassVar[Lock] = Lock()
    _id: int
    _name_hint: str

    def __init__(self, name_hint: str) -> None:
        with Identifier._id_lock:
            self._id = Identifier._next_id
            Identifier._next_id += 1
        self._name_hint = name_hint

    @property
    def name_hint(self) -> str:
        return self._name_hint

    @property
    def id(self) -> int:
        return self._id

    def serialize_to_dict(self) -> SerializedDict:
        return {"id": self._id, "name_hint": self._name_hint}

    @classmethod
    def deserialize_from_dict(cls, data: SerializedDict) -> "Identifier":
        if not _is_valid_identifier_data(data):
            raise DeserializationDictStructureError(
                cls, _IdentifierData.__annotations__, data
            )
        if data["id"] < 0:
            raise DeserializationValueError(
                cls, "id", "a non-negative integer", data["id"]
            )
        identifier = cls.__new__(cls)
        identifier._id = data["id"]
        identifier._name_hint = data["name_hint"]
        advanced = False
        with Identifier._id_lock:
            if identifier._id >= Identifier._next_id:
                Identifier._next_id = identifier._id + 1
                advanced = True
        if advanced:
            _LOGGER.debug(
                "advanced _next_id past %d (name_hint=%r)",
                identifier._id,
                identifier._name_hint,
            )
        identifier.freeze()
        return identifier

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Identifier) and self._id == other._id

    def __hash__(self) -> int:
        return hash(self._id)

    def __str__(self) -> str:
        return self._name_hint

    def __repr__(self) -> str:
        return f"{self._name_hint}::{self._id}"

    def __neg__(self) -> "UnaryExpression":
        return _make_unary_expression(_get_unary_operation_type().NEGATE, self)

    def __pos__(self) -> "UnaryExpression":
        return _make_unary_expression(_get_unary_operation_type().POSITIVE, self)

    def __add__(self, other: Any) -> "BinaryExpression":
        return _make_binary_expression(_get_binary_operation_type().ADD, self, other)

    def __radd__(self, other: Any) -> "BinaryExpression":
        return _make_binary_expression(_get_binary_operation_type().ADD, other, self)

    def __sub__(self, other: Any) -> "BinaryExpression":
        return _make_binary_expression(
            _get_binary_operation_type().SUBTRACT, self, other
        )

    def __rsub__(self, other: Any) -> "BinaryExpression":
        return _make_binary_expression(
            _get_binary_operation_type().SUBTRACT, other, self
        )

    def __mul__(self, other: Any) -> "BinaryExpression":
        return _make_binary_expression(
            _get_binary_operation_type().MULTIPLY, self, other
        )

    def __rmul__(self, other: Any) -> "BinaryExpression":
        return _make_binary_expression(
            _get_binary_operation_type().MULTIPLY, other, self
        )

    def __truediv__(self, other: Any) -> "BinaryExpression":
        return _make_binary_expression(_get_binary_operation_type().DIVIDE, self, other)

    def __rtruediv__(self, other: Any) -> "BinaryExpression":
        return _make_binary_expression(_get_binary_operation_type().DIVIDE, other, self)

    def __floordiv__(self, other: Any) -> "BinaryExpression":
        return _make_binary_expression(
            _get_binary_operation_type().FLOOR_DIVIDE, self, other
        )

    def __rfloordiv__(self, other: Any) -> "BinaryExpression":
        return _make_binary_expression(
            _get_binary_operation_type().FLOOR_DIVIDE, other, self
        )

    def __mod__(self, other: Any) -> "BinaryExpression":
        return _make_binary_expression(_get_binary_operation_type().MODULO, self, other)

    def __rmod__(self, other: Any) -> "BinaryExpression":
        return _make_binary_expression(_get_binary_operation_type().MODULO, other, self)

    def __pow__(self, other: Any) -> "BinaryExpression":
        return _make_binary_expression(_get_binary_operation_type().POWER, self, other)

    def __rpow__(self, other: Any) -> "BinaryExpression":
        return _make_binary_expression(_get_binary_operation_type().POWER, other, self)

    def __lt__(self, other: Any) -> "BinaryExpression":
        return _make_binary_expression(_get_binary_operation_type().LESS, self, other)

    def __le__(self, other: Any) -> "BinaryExpression":
        return _make_binary_expression(
            _get_binary_operation_type().LESS_EQUAL, self, other
        )

    def __gt__(self, other: Any) -> "BinaryExpression":
        return _make_binary_expression(
            _get_binary_operation_type().GREATER, self, other
        )

    def __ge__(self, other: Any) -> "BinaryExpression":
        return _make_binary_expression(
            _get_binary_operation_type().GREATER_EQUAL, self, other
        )

    def logical_not(self) -> "UnaryExpression":
        """Build a ``LOGICAL_NOT`` expression with this identifier as operand."""
        return _make_unary_expression(_get_unary_operation_type().LOGICAL_NOT, self)

    def logical_and(self, *others: Any) -> "BinaryExpression":
        """Build a right-folded ``LOGICAL_AND`` tree with this and ``others``."""
        from .expression import logical_and as _logical_and  # noqa: PLC0415

        return _logical_and(self, *others)

    def logical_or(self, *others: Any) -> "BinaryExpression":
        """Build a right-folded ``LOGICAL_OR`` tree with this and ``others``."""
        from .expression import logical_or as _logical_or  # noqa: PLC0415

        return _logical_or(self, *others)
