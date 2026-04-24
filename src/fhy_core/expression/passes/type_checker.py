"""Bidirectional type checking and inference for expressions.

Errors from this module are always :class:`FhYCoreTypeError` instances with
messages of one of two shapes:

- "Type error while inferring type of `<root>`: <reason>"
- "Type error while inferring type of `<root>` at sub-expression `<sub>`:
  <reason>"

where ``<root>`` is the top-level expression passed to
:meth:`ExpressionTypeChecker.synthesize` / :meth:`check`, ``<sub>`` is the
sub-expression where the failure surfaced, and ``<reason>`` describes the
specific rule that was violated along with the types and values involved.
This is implemented via :class:`_TypeCheckContext`, which maintains a stack
of enclosing expressions as the checker recurses so every
:class:`FhYCoreTypeError` is enriched with that trace.
"""

__all__ = [
    "get_core_data_type_from_literal_type",
    "synthesize_expression_type",
    "check_expression_type",
]

from contextlib import contextmanager
from typing import Callable, Iterator, TypeAlias

from fhy_core.expression.core import (
    BinaryExpression,
    BinaryOperation,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    LiteralType,
    UnaryExpression,
    UnaryOperation,
)
from fhy_core.expression.pprint import pformat_expression
from fhy_core.identifier import Identifier
from fhy_core.pass_infrastructure import (
    PassExecutionError,
    VisitablePass,
    register_pass,
)
from fhy_core.types import (
    CoreDataType,
    FhYCoreTypeError,
    IndexType,
    NumericalType,
    PrimitiveDataType,
    TupleType,
    Type,
    TypeQualifier,
    get_core_data_type_bit_width,
    is_weak_core_data_type,
    promote_core_data_types,
    promote_primitive_data_types,
    promote_type_qualifiers,
    resolve_literal_core_data_type,
)
from fhy_core.utils import Stack

ExpressionValueType: TypeAlias = NumericalType | IndexType

_ARITHMETIC_OPERATIONS = frozenset(
    {
        BinaryOperation.ADD,
        BinaryOperation.SUBTRACT,
        BinaryOperation.MULTIPLY,
        BinaryOperation.DIVIDE,
        BinaryOperation.FLOOR_DIVIDE,
        BinaryOperation.MODULO,
        BinaryOperation.POWER,
    }
)

_UNSIGNED_CORE_DATA_TYPES = frozenset(
    {
        CoreDataType.UINT,
        CoreDataType.UINT8,
        CoreDataType.UINT16,
        CoreDataType.UINT32,
    }
)

_SIGNED_CORE_DATA_TYPES = frozenset(
    {
        CoreDataType.INT,
        CoreDataType.INT8,
        CoreDataType.INT16,
        CoreDataType.INT32,
        CoreDataType.INT64,
    }
)

_INTEGRAL_CORE_DATA_TYPES = _UNSIGNED_CORE_DATA_TYPES | _SIGNED_CORE_DATA_TYPES

_REAL_FLOAT_CORE_DATA_TYPES = frozenset(
    {
        CoreDataType.FLOAT,
        CoreDataType.FLOAT16,
        CoreDataType.FLOAT32,
        CoreDataType.FLOAT64,
    }
)

_COMPLEX_CORE_DATA_TYPES = frozenset(
    {
        CoreDataType.COMPLEX32,
        CoreDataType.COMPLEX64,
        CoreDataType.COMPLEX128,
    }
)

_FLOAT_LIKE_CORE_DATA_TYPES = _REAL_FLOAT_CORE_DATA_TYPES | _COMPLEX_CORE_DATA_TYPES


def get_core_data_type_from_literal_type(literal: LiteralType) -> CoreDataType:
    """Return the weak core data type assigned to a literal.

    Only numeric literals (``int`` and ``float``) participate in the type
    system; ``bool`` and ``str`` values are accepted by
    :class:`~fhy_core.expression.core.LiteralExpression` for other purposes
    (e.g. serialization round-trips) but have no numeric core data type and
    are rejected here with :class:`NotImplementedError`. Callers that may
    receive non-numeric literals should either filter them earlier or catch
    ``NotImplementedError`` explicitly.

    Raises:
        NotImplementedError: If ``literal`` is a ``bool`` or ``str``.
        ValueError: If ``literal`` is none of the supported literal types.

    """
    match literal:
        case bool():
            raise NotImplementedError("Boolean literals are not yet supported.")
        case int() if literal >= 0:
            return CoreDataType.UINT
        case int() if literal < 0:
            return CoreDataType.INT
        case float():
            return CoreDataType.FLOAT
        case str():
            raise NotImplementedError("String literals are not yet supported.")
        case _:
            raise ValueError(f"Unsupported literal type: {type(literal)}.")


def _format_expression(expression: Expression) -> str:
    return pformat_expression(expression, show_id=True)


def _is_weak_numerical_type(numerical_type: NumericalType) -> bool:
    return is_weak_core_data_type(
        _get_primitive_data_type(numerical_type).core_data_type
    )


def _get_primitive_data_type(numerical_type: NumericalType) -> PrimitiveDataType:
    data_type = numerical_type.data_type
    if not isinstance(data_type, PrimitiveDataType):
        raise FhYCoreTypeError(f"expected a primitive data type, got {data_type}")
    return data_type


def _get_numeric_literal_value(literal_expression: LiteralExpression) -> int | float:
    literal_value = literal_expression.value
    if isinstance(literal_value, bool | str):
        raise FhYCoreTypeError(
            f"expected a numeric literal value, got {literal_value!r}"
        )
    return literal_value


def _is_integral_numerical_type(numerical_type: NumericalType) -> bool:
    return (
        _get_primitive_data_type(numerical_type).core_data_type
        in _INTEGRAL_CORE_DATA_TYPES
    )


def _get_real_float_core_data_type_for_bit_width(bit_width: int | None) -> CoreDataType:
    core_data_type_bit_widths: list[tuple[int, CoreDataType]] = []
    for core_data_type in _REAL_FLOAT_CORE_DATA_TYPES:
        core_data_type_bit_width = get_core_data_type_bit_width(core_data_type)
        if core_data_type_bit_width is not None:
            core_data_type_bit_widths.append((core_data_type_bit_width, core_data_type))
    core_data_type_bit_widths = sorted(core_data_type_bit_widths)

    if bit_width is None:
        return CoreDataType.FLOAT
    else:
        for core_data_type_bit_width, core_data_type in core_data_type_bit_widths:
            if core_data_type_bit_width >= bit_width:
                return core_data_type
        raise FhYCoreTypeError(
            f"no real float core data type found for bit width {bit_width}"
        )


class _TypeCheckContext:
    """Tracks the enclosing expression trace for context-rich error messages.

    The checker's public entry points (``synthesize``, ``check``) and each
    recursive ``_infer*`` call enter a new scope via :meth:`entering`, which
    pushes the expression being processed onto a stack. When an error is
    raised via :meth:`type_error`, the message includes the root expression
    (the first one pushed) and, when different, the current sub-expression
    (the most recently pushed one still live).

    The reason string passed to :meth:`type_error` should describe *what*
    went wrong without re-stating the expression itself — the context layer
    is responsible for that framing.
    """

    _stack: Stack[Expression]

    def __init__(self) -> None:
        self._stack = Stack[Expression]()

    @contextmanager
    def entering(self, expression: Expression) -> Iterator[None]:
        """Push ``expression`` onto the trace for the duration of the ``with`` block."""
        self._stack.push(expression)
        try:
            yield
        finally:
            self._stack.pop()

    def type_error(self, reason: str) -> FhYCoreTypeError:
        """Build a :class:`FhYCoreTypeError` framed by the current context trace."""
        if not self._stack:
            return FhYCoreTypeError(f"Type error: {reason}")
        stack_elements = tuple(self._stack)
        root = stack_elements[0]
        current = stack_elements[-1]
        root_text = _format_expression(root)
        if current is root:
            return FhYCoreTypeError(
                f"Type error while inferring type of `{root_text}`: {reason}"
            )
        current_text = _format_expression(current)
        return FhYCoreTypeError(
            f"Type error while inferring type of `{root_text}` "
            f"at sub-expression `{current_text}`: {reason}"
        )


@register_pass(
    "fhy_core.expression.type_check",
    "Bidirectionally synthesize and check expression types.",
)
class ExpressionTypeChecker(VisitablePass[Expression, tuple[Type, TypeQualifier]]):
    """Bidirectional type checker for expressions."""

    _get_identifier_type: Callable[[Identifier], tuple[Type, TypeQualifier]]
    _context: _TypeCheckContext

    def __init__(
        self, get_identifier_type: Callable[[Identifier], tuple[Type, TypeQualifier]]
    ) -> None:
        super().__init__()
        self._get_identifier_type = get_identifier_type
        self._context = _TypeCheckContext()

    def synthesize(self, expression: Expression) -> tuple[Type, TypeQualifier]:
        """Synthesize a type for an expression."""
        return self._infer(expression)

    def check(
        self, expression: Expression, expected_type: Type
    ) -> tuple[Type, TypeQualifier]:
        """Check an expression against an expected type."""
        actual_type, actual_qualifier = self._infer(expression, expected_type)
        with self._context.entering(expression):
            self._check_expected_type(expression, actual_type, expected_type)
        return actual_type, actual_qualifier

    # --- Visitor dispatch ----------------------------------------------------

    def visit_unary_expression(
        self, unary_expression: UnaryExpression
    ) -> tuple[Type, TypeQualifier]:
        return self._infer_unary_expression(unary_expression)

    def visit_binary_expression(
        self, binary_expression: BinaryExpression
    ) -> tuple[Type, TypeQualifier]:
        return self._infer_binary_expression(binary_expression)

    def visit_identifier_expression(
        self, identifier_expression: IdentifierExpression
    ) -> tuple[Type, TypeQualifier]:
        with self._context.entering(identifier_expression):
            identifier_type, identifier_qualifier = self._get_identifier_type(
                identifier_expression.identifier
            )
            if identifier_qualifier == TypeQualifier.OUTPUT:
                raise self._context.type_error(
                    f"identifier `{_format_expression(identifier_expression)}` has "
                    'type qualifier "output" and cannot be read from'
                )
            return (
                self._as_expression_value_type(identifier_expression, identifier_type),
                identifier_qualifier,
            )

    def visit_literal_expression(
        self, literal_expression: LiteralExpression
    ) -> tuple[Type, TypeQualifier]:
        return (
            NumericalType(
                PrimitiveDataType(
                    get_core_data_type_from_literal_type(literal_expression.value)
                )
            ),
            TypeQualifier.PARAM,
        )

    def get_noop_output(self, ir: Expression) -> tuple[Type, TypeQualifier]:
        raise PassExecutionError(
            f'Pass "{self.get_pass_name()}" does not define noop output for {ir!r}.'
        )

    # --- Core inference ------------------------------------------------------

    def _infer(
        self, expression: Expression, expected_type: Type | None = None
    ) -> tuple[Type, TypeQualifier]:
        match expression:
            case UnaryExpression():
                return self._infer_unary_expression(expression, expected_type)
            case BinaryExpression():
                return self._infer_binary_expression(expression, expected_type)
            case IdentifierExpression():
                return self.visit_identifier_expression(expression)
            case LiteralExpression():
                return self._infer_literal_expression(expression, expected_type)
            case _:
                raise NotImplementedError(
                    f"Unsupported expression type: {type(expression)}"
                )

    def _infer_literal_expression(
        self,
        literal_expression: LiteralExpression,
        expected_type: Type | None = None,
    ) -> tuple[Type, TypeQualifier]:
        with self._context.entering(literal_expression):
            if expected_type is None:
                return self.visit_literal_expression(literal_expression)

            expected_value_type = self._as_expression_value_type(
                literal_expression, expected_type
            )
            if isinstance(expected_value_type, IndexType):
                raise self._context.type_error(
                    f"a literal value cannot be checked against index type "
                    f"{expected_type}"
                )

            resolved_core_data_type = resolve_literal_core_data_type(
                _get_numeric_literal_value(literal_expression),
                _get_primitive_data_type(expected_value_type).core_data_type,
            )
            return (
                NumericalType(PrimitiveDataType(resolved_core_data_type)),
                TypeQualifier.PARAM,
            )

    def _infer_unary_expression(
        self,
        unary_expression: UnaryExpression,
        expected_type: Type | None = None,
    ) -> tuple[Type, TypeQualifier]:
        with self._context.entering(unary_expression):
            operand_type, operand_qualifier = self._infer(
                unary_expression.operand, expected_type
            )
            if operand_qualifier == TypeQualifier.OUTPUT:
                raise RuntimeError(
                    '"output" type qualifier should not be possible here.'
                )

            operand_value_type = self._as_expression_value_type(
                unary_expression, operand_type
            )

            match unary_expression.operation:
                case UnaryOperation.POSITIVE:
                    return operand_value_type, operand_qualifier
                case UnaryOperation.NEGATE:
                    if isinstance(operand_value_type, IndexType):
                        raise self._context.type_error(
                            "unary negation is not defined for index types; the "
                            "resulting bounds and stride cannot be inferred safely"
                        )
                    if isinstance(
                        unary_expression.operand, LiteralExpression
                    ) and _is_weak_numerical_type(operand_value_type):
                        return (
                            NumericalType(
                                PrimitiveDataType(
                                    get_core_data_type_from_literal_type(
                                        -_get_numeric_literal_value(
                                            unary_expression.operand
                                        )
                                    )
                                )
                            ),
                            operand_qualifier,
                        )
                    return operand_value_type, operand_qualifier
                case UnaryOperation.LOGICAL_NOT:
                    raise NotImplementedError(
                        "Boolean result types are not yet supported."
                    )

    def _infer_binary_expression(  # noqa: PLR0912, PLR0915
        self,
        binary_expression: BinaryExpression,
        expected_type: Type | None = None,
    ) -> tuple[Type, TypeQualifier]:
        with self._context.entering(binary_expression):
            left_expected_type = None
            right_expected_type = None
            if (
                binary_expression.operation in _ARITHMETIC_OPERATIONS
                and isinstance(expected_type, NumericalType)
                and expected_type.is_scalar()
            ):
                if isinstance(binary_expression.left, LiteralExpression):
                    left_expected_type = expected_type
                if isinstance(binary_expression.right, LiteralExpression):
                    right_expected_type = expected_type

            left_type, left_qualifier = self._infer(
                binary_expression.left, left_expected_type
            )
            right_type, right_qualifier = self._infer(
                binary_expression.right, right_expected_type
            )
            if TypeQualifier.OUTPUT in {left_qualifier, right_qualifier}:
                raise RuntimeError(
                    '"output" type qualifier should not be possible here.'
                )

            left_value_type = self._as_expression_value_type(
                binary_expression.left, left_type
            )
            right_value_type = self._as_expression_value_type(
                binary_expression.right, right_type
            )

            if (
                binary_expression.operation in _ARITHMETIC_OPERATIONS
                and isinstance(binary_expression.left, LiteralExpression)
                and isinstance(left_value_type, NumericalType)
                and _is_weak_numerical_type(left_value_type)
                and isinstance(right_value_type, NumericalType)
                and not _is_weak_numerical_type(right_value_type)
            ):
                checked_left_type, left_qualifier = self.check(
                    binary_expression.left, right_value_type
                )
                left_value_type = self._as_expression_value_type(
                    binary_expression.left, checked_left_type
                )
            if (
                binary_expression.operation in _ARITHMETIC_OPERATIONS
                and isinstance(binary_expression.right, LiteralExpression)
                and isinstance(right_value_type, NumericalType)
                and _is_weak_numerical_type(right_value_type)
                and isinstance(left_value_type, NumericalType)
                and not _is_weak_numerical_type(left_value_type)
            ):
                checked_right_type, right_qualifier = self.check(
                    binary_expression.right, left_value_type
                )
                right_value_type = self._as_expression_value_type(
                    binary_expression.right, checked_right_type
                )

            left_value_type = self._as_expression_value_type(
                binary_expression.left, left_value_type
            )
            right_value_type = self._as_expression_value_type(
                binary_expression.right, right_value_type
            )

            operation = binary_expression.operation
            if operation not in _ARITHMETIC_OPERATIONS:
                raise NotImplementedError("Boolean result types are not yet supported.")

            if isinstance(left_value_type, NumericalType) and isinstance(
                right_value_type, NumericalType
            ):
                if operation == BinaryOperation.DIVIDE:
                    return (
                        NumericalType(
                            self._get_division_primitive_data_type(
                                left_value_type, right_value_type
                            )
                        ),
                        promote_type_qualifiers(left_qualifier, right_qualifier),
                    )
                if operation == BinaryOperation.FLOOR_DIVIDE:
                    return (
                        NumericalType(
                            self._get_floor_division_primitive_data_type(
                                left_value_type, right_value_type
                            )
                        ),
                        promote_type_qualifiers(left_qualifier, right_qualifier),
                    )
                return (
                    NumericalType(
                        promote_primitive_data_types(
                            _get_primitive_data_type(left_value_type),
                            _get_primitive_data_type(right_value_type),
                        )
                    ),
                    promote_type_qualifiers(left_qualifier, right_qualifier),
                )
            elif operation in {
                BinaryOperation.DIVIDE,
                BinaryOperation.FLOOR_DIVIDE,
            } and (
                isinstance(left_value_type, IndexType)
                or isinstance(right_value_type, IndexType)
            ):
                raise self._context.type_error(
                    "division is not defined for operands of index type"
                )
            elif (
                operation == BinaryOperation.ADD
                and isinstance(left_value_type, IndexType)
                and isinstance(right_value_type, NumericalType)
            ):
                if not _is_integral_numerical_type(right_value_type):
                    raise self._context.type_error(
                        f"index shift requires an integral scalar offset, but the "
                        f"right operand has type {right_value_type}"
                    )
                return (
                    self._shift_index_type(left_value_type, binary_expression.right),
                    promote_type_qualifiers(left_qualifier, right_qualifier),
                )
            elif (
                operation == BinaryOperation.ADD
                and isinstance(left_value_type, NumericalType)
                and isinstance(right_value_type, IndexType)
            ):
                if not _is_integral_numerical_type(left_value_type):
                    raise self._context.type_error(
                        f"index shift requires an integral scalar offset, but the "
                        f"left operand has type {left_value_type}"
                    )
                return (
                    self._shift_index_type(right_value_type, binary_expression.left),
                    promote_type_qualifiers(left_qualifier, right_qualifier),
                )
            elif (
                operation == BinaryOperation.SUBTRACT
                and isinstance(left_value_type, IndexType)
                and isinstance(right_value_type, NumericalType)
            ):
                if not _is_integral_numerical_type(right_value_type):
                    raise self._context.type_error(
                        f"index shift requires an integral scalar offset, but the "
                        f"right operand has type {right_value_type}"
                    )
                return (
                    self._shift_index_type(
                        left_value_type,
                        binary_expression.right,
                        subtract=True,
                    ),
                    promote_type_qualifiers(left_qualifier, right_qualifier),
                )
            elif (
                operation == BinaryOperation.MULTIPLY
                and isinstance(left_value_type, IndexType)
                and isinstance(right_value_type, NumericalType)
            ):
                if not isinstance(binary_expression.right, LiteralExpression):
                    raise self._context.type_error(
                        "index scaling requires a positive integer literal scalar, "
                        f"but the right operand "
                        f"`{_format_expression(binary_expression.right)}` is not a "
                        "literal expression"
                    )
                if not _is_integral_numerical_type(right_value_type):
                    raise self._context.type_error(
                        "index scaling requires a positive integer literal scalar, "
                        f"but the right operand has non-integral type "
                        f"{right_value_type}"
                    )
                return (
                    self._scale_index_type(left_value_type, binary_expression.right),
                    promote_type_qualifiers(left_qualifier, right_qualifier),
                )
            elif (
                operation == BinaryOperation.MULTIPLY
                and isinstance(left_value_type, NumericalType)
                and isinstance(right_value_type, IndexType)
            ):
                if not isinstance(binary_expression.left, LiteralExpression):
                    raise self._context.type_error(
                        "index scaling requires a positive integer literal scalar, "
                        f"but the left operand "
                        f"`{_format_expression(binary_expression.left)}` is not a "
                        "literal expression"
                    )
                if not _is_integral_numerical_type(left_value_type):
                    raise self._context.type_error(
                        "index scaling requires a positive integer literal scalar, "
                        f"but the left operand has non-integral type "
                        f"{left_value_type}"
                    )
                return (
                    self._scale_index_type(right_value_type, binary_expression.left),
                    promote_type_qualifiers(left_qualifier, right_qualifier),
                )
            elif isinstance(left_value_type, IndexType) and isinstance(
                right_value_type, IndexType
            ):
                if operation == BinaryOperation.ADD:
                    return (
                        IndexType(
                            left_value_type.lower_bound + right_value_type.lower_bound,
                            left_value_type.upper_bound + right_value_type.upper_bound,
                            self._combine_index_strides_for_add(
                                left_value_type.stride, right_value_type.stride
                            ),
                        ),
                        promote_type_qualifiers(left_qualifier, right_qualifier),
                    )
                elif operation == BinaryOperation.SUBTRACT:
                    raise self._context.type_error(
                        "subtraction is not defined between two index types; the "
                        "resulting stride semantics have not been defined"
                    )
                else:
                    raise self._context.type_error(
                        f"the {operation.name.lower()} operation is not defined "
                        "between two index types"
                    )
            else:
                raise self._context.type_error(
                    f"the {operation.name.lower()} operation is not defined for "
                    f"operands of types {left_value_type} and {right_value_type}; "
                    "the resulting index bounds and stride cannot be inferred safely"
                )

    # --- Context-aware helpers ----------------------------------------------

    def _as_expression_value_type(
        self, expression: Expression, type_: Type
    ) -> ExpressionValueType:
        if isinstance(type_, TupleType):
            raise NotImplementedError(
                f"sub-expression `{_format_expression(expression)}` resolves to "
                f"tuple type {type_}, which is not valid inside an expression"
            )
        elif isinstance(type_, IndexType):
            return type_
        elif not isinstance(type_, NumericalType):
            raise self._context.type_error(
                f"sub-expression `{_format_expression(expression)}` must resolve "
                f"to a scalar numerical type or index type, but got {type_}"
            )
        elif not isinstance(type_.data_type, PrimitiveDataType):
            raise self._context.type_error(
                f"sub-expression `{_format_expression(expression)}` must resolve "
                f"to a primitive numerical type, but got {type_}"
            )
        elif not type_.is_scalar():
            raise NotImplementedError(
                f"sub-expression `{_format_expression(expression)}` resolves to "
                f"tensor type {type_}, but only scalar numerical and index types "
                "are allowed in expressions"
            )
        else:
            return type_

    def _check_expected_type(
        self, expression: Expression, actual_type: Type, expected_type: Type
    ) -> None:
        if isinstance(actual_type, IndexType) and isinstance(expected_type, IndexType):
            if not actual_type.is_structurally_equivalent(expected_type):
                raise self._context.type_error(
                    f"synthesized index type {actual_type} is not structurally "
                    f"equivalent to the expected index type {expected_type}"
                )
            return

        actual_value_type = self._as_expression_value_type(expression, actual_type)
        expected_value_type = self._as_expression_value_type(expression, expected_type)
        if isinstance(actual_value_type, IndexType) or isinstance(
            expected_value_type, IndexType
        ):
            raise self._context.type_error(
                f"synthesized type {actual_type} is incompatible with the "
                f"expected type {expected_type}: one is an index type and the "
                "other is a numerical type"
            )

        promoted_type = promote_primitive_data_types(
            _get_primitive_data_type(actual_value_type),
            _get_primitive_data_type(expected_value_type),
        )
        if (
            promoted_type.core_data_type
            != _get_primitive_data_type(expected_value_type).core_data_type
        ):
            raise self._context.type_error(
                f"synthesized type {actual_type} is wider than the expected type "
                f"{expected_type}; promoting them yields {promoted_type}, which "
                "does not match the expected type"
            )

    def _get_literal_stride_value(self, stride: Expression | None) -> int:
        if stride is None:
            return 1
        if not isinstance(stride, LiteralExpression):
            raise self._context.type_error(
                "combining two index types requires both strides to be integer "
                f"literals, but got non-literal stride `{_format_expression(stride)}`"
            )
        value = stride.value
        if isinstance(value, bool) or not isinstance(value, int):
            raise self._context.type_error(
                f"an index stride literal must be an integer, but got {value!r}"
            )
        return value

    @staticmethod
    def _get_literal_stride_expression(value: int) -> Expression | None:
        return None if value == 1 else LiteralExpression(value)

    def _combine_index_strides_for_add(
        self, left_stride: Expression | None, right_stride: Expression | None
    ) -> Expression | None:
        return self._get_literal_stride_expression(
            min(
                self._get_literal_stride_value(left_stride),
                self._get_literal_stride_value(right_stride),
            )
        )

    def _scale_index_type(
        self, index_type: IndexType, scalar: LiteralExpression
    ) -> IndexType:
        value = scalar.value
        if isinstance(value, bool) or not isinstance(value, int):
            raise self._context.type_error(
                f"index scaling requires an integer literal scalar, but got scalar "
                f"value {value!r}"
            )
        if value <= 0:
            raise self._context.type_error(
                f"index scaling requires a positive integer literal scalar, but "
                f"got scalar value {value}"
            )
        new_stride: Expression | None
        if index_type.stride is None:
            new_stride = None if value == 1 else LiteralExpression(value)
        else:
            new_stride = scalar * index_type.stride
        return IndexType(
            scalar * index_type.lower_bound,
            scalar * index_type.upper_bound,
            new_stride,
        )

    @staticmethod
    def _shift_index_type(
        index_type: IndexType,
        offset_expression: Expression,
        *,
        subtract: bool = False,
    ) -> IndexType:
        if subtract:
            return IndexType(
                index_type.lower_bound - offset_expression,
                index_type.upper_bound - offset_expression,
                index_type.stride,
            )
        return IndexType(
            index_type.lower_bound + offset_expression,
            index_type.upper_bound + offset_expression,
            index_type.stride,
        )

    def _get_division_primitive_data_type(
        self, left_type: NumericalType, right_type: NumericalType
    ) -> PrimitiveDataType:
        left_core_data_type = _get_primitive_data_type(left_type).core_data_type
        right_core_data_type = _get_primitive_data_type(right_type).core_data_type

        if (
            left_core_data_type in _INTEGRAL_CORE_DATA_TYPES
            and right_core_data_type in _INTEGRAL_CORE_DATA_TYPES
        ):
            concrete_bit_widths = [
                bit_width
                for bit_width in (
                    get_core_data_type_bit_width(left_core_data_type),
                    get_core_data_type_bit_width(right_core_data_type),
                )
                if bit_width is not None
            ]
            return PrimitiveDataType(
                _get_real_float_core_data_type_for_bit_width(
                    max(concrete_bit_widths) if concrete_bit_widths else None
                )
            )

        if (
            left_core_data_type in _INTEGRAL_CORE_DATA_TYPES
            and right_core_data_type in _FLOAT_LIKE_CORE_DATA_TYPES
        ):
            left_core_data_type = _get_real_float_core_data_type_for_bit_width(
                get_core_data_type_bit_width(left_core_data_type)
            )
        elif (
            left_core_data_type in _FLOAT_LIKE_CORE_DATA_TYPES
            and right_core_data_type in _INTEGRAL_CORE_DATA_TYPES
        ):
            right_core_data_type = _get_real_float_core_data_type_for_bit_width(
                get_core_data_type_bit_width(right_core_data_type)
            )

        return PrimitiveDataType(
            promote_core_data_types(left_core_data_type, right_core_data_type)
        )

    def _get_floor_division_primitive_data_type(
        self, left_type: NumericalType, right_type: NumericalType
    ) -> PrimitiveDataType:
        left_core_data_type = _get_primitive_data_type(left_type).core_data_type
        right_core_data_type = _get_primitive_data_type(right_type).core_data_type

        if (
            left_core_data_type in _INTEGRAL_CORE_DATA_TYPES
            and right_core_data_type in _INTEGRAL_CORE_DATA_TYPES
        ):
            return promote_primitive_data_types(
                _get_primitive_data_type(left_type),
                _get_primitive_data_type(right_type),
            )

        if (
            left_core_data_type in _COMPLEX_CORE_DATA_TYPES
            or right_core_data_type in _COMPLEX_CORE_DATA_TYPES
        ):
            raise self._context.type_error(
                f"floor division is not defined for complex numerical types "
                f"(operand types were {left_type} and {right_type})"
            )

        if (
            left_core_data_type in _INTEGRAL_CORE_DATA_TYPES
            and right_core_data_type in _REAL_FLOAT_CORE_DATA_TYPES
        ):
            left_core_data_type = _get_real_float_core_data_type_for_bit_width(
                get_core_data_type_bit_width(left_core_data_type)
            )
        elif (
            left_core_data_type in _REAL_FLOAT_CORE_DATA_TYPES
            and right_core_data_type in _INTEGRAL_CORE_DATA_TYPES
        ):
            right_core_data_type = _get_real_float_core_data_type_for_bit_width(
                get_core_data_type_bit_width(right_core_data_type)
            )

        return PrimitiveDataType(
            promote_core_data_types(left_core_data_type, right_core_data_type)
        )


def synthesize_expression_type(
    expression: Expression,
    get_identifier_type: Callable[[Identifier], tuple[Type, TypeQualifier]],
) -> tuple[Type, TypeQualifier]:
    """Synthesize a type for an expression.

    Args:
        expression: The expression to synthesize a type for.
        get_identifier_type: A function that returns the type of an identifier.

    Returns:
        A tuple containing the synthesized type and the type qualifier.

    """
    return ExpressionTypeChecker(get_identifier_type).synthesize(expression)


def check_expression_type(
    expression: Expression,
    expected_type: Type,
    get_identifier_type: Callable[[Identifier], tuple[Type, TypeQualifier]],
) -> tuple[Type, TypeQualifier]:
    """Check an expression against an expected type.

    Args:
        expression: The expression to check.
        expected_type: The expected type to check against.
        get_identifier_type: A function that returns the type of an identifier.

    Returns:
        A tuple containing the checked type and the type qualifier.

    """
    return ExpressionTypeChecker(get_identifier_type).check(expression, expected_type)
