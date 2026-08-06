"""Bidirectional type checking and inference for expressions.

:func:`synthesize_expression_type` infers a type purely from the leaves of
an expression; :func:`check_expression_type` propagates an expected type
into the expression so weak literals can adopt the surrounding context,
then verifies the synthesized result is assignment-compatible with the
expected type. Both entry points return ``(Type, TypeQualifier)``.

This module raises :class:`FhYCoreTypeError` for type-rule violations and
:class:`NotImplementedError` for expression shapes / operations that are
not yet supported (string literals, tensor operands, unknown
``UnaryOperation`` / ``Expression`` subclasses).

:class:`FhYCoreTypeError` instances raised through
:meth:`_TypeCheckContext.type_error` carry a context-framed message of one
of two shapes:

- "Type error while inferring type of `<root>`: <reason>"
- "Type error while inferring type of `<root>` at sub-expression `<sub>`:
  <reason>"

where ``<root>`` is the outermost expression on the context stack,
``<sub>`` is the sub-expression where the failure surfaced, and
``<reason>`` describes the specific rule that was violated along with the
types and values involved. The framing is provided by
:class:`_TypeCheckContext`, which the checker pushes / pops as it recurses.

A small number of module-level helpers (``_get_primitive_data_type``,
``_get_numeric_literal_value``, ``_get_real_float_core_data_type_for_bit_width``)
raise bare :class:`FhYCoreTypeError` instances without that framing - they
have no access to a context, and the checker does not catch and reframe
them, so callers can see either form on a failed type check.
"""

from fhy_core.utils.override import override

__all__ = [
    "ExpressionTypeChecker",
    "check_expression_type",
    "get_core_data_type_from_literal_type",
    "synthesize_expression_type",
]

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TypeAlias

from fhy_core.identifier import Identifier
from fhy_core.pass_infrastructure import (
    PassExecutionError,
    VisitablePass,
    register_pass,
)
from fhy_core.symbolic.expression.core import (
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    LiteralType,
    TernaryExpression,
    UnaryExpression,
    UnaryOperation,
)
from fhy_core.symbolic.expression.errors import EntryLookupError
from fhy_core.symbolic.expression.pprint import pformat_expression
from fhy_core.symbolic.expression.registry.entries import (
    CallTargetResolver,
    NativeConstant,
    NativeFunction,
    RegisteredFunction,
)
from fhy_core.symbolic.expression.registry.storage import get_registered_entry
from fhy_core.symbolic.expression.sort import FunctionSort
from fhy_core.utils import Stack, is_strict_int

from ..core import (
    CoreDataType,
    FhYCoreTypeError,
    IndexType,
    NumericalType,
    PrimitiveDataType,
    Type,
    TypeQualifier,
    get_core_data_type_bit_width,
    is_weak_core_data_type,
    promote_core_data_types,
    promote_primitive_data_types,
    promote_type_qualifiers,
    resolve_literal_core_data_type,
)
from .sort_compatibility import (
    get_result_core_data_type_for_sort,
    is_core_data_type_compatible_with_sort,
)

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

_LOGICAL_BOOLEAN_OPERATIONS = frozenset(
    {BinaryOperation.LOGICAL_AND, BinaryOperation.LOGICAL_OR}
)

_EQUALITY_OPERATIONS = frozenset({BinaryOperation.EQUAL, BinaryOperation.NOT_EQUAL})

_ORDERING_OPERATIONS = frozenset(
    {
        BinaryOperation.LESS,
        BinaryOperation.LESS_EQUAL,
        BinaryOperation.GREATER,
        BinaryOperation.GREATER_EQUAL,
    }
)

_COMPARISON_OPERATIONS = _EQUALITY_OPERATIONS | _ORDERING_OPERATIONS

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


def _build_real_float_core_data_types_by_bit_width() -> tuple[
    tuple[int, CoreDataType], ...
]:
    entries: list[tuple[int, CoreDataType]] = []
    for core_data_type in _REAL_FLOAT_CORE_DATA_TYPES:
        bit_width = get_core_data_type_bit_width(core_data_type)
        if bit_width is not None:
            entries.append((bit_width, core_data_type))
    return tuple(sorted(entries))


_REAL_FLOAT_CORE_DATA_TYPES_BY_BIT_WIDTH = (
    _build_real_float_core_data_types_by_bit_width()
)


def get_core_data_type_from_literal_type(literal: LiteralType) -> CoreDataType:
    """Return the core data type assigned to a literal.

    Numeric literals (``int`` and ``float``) participate in the type
    system via the weak ``UINT``/``INT``/``FLOAT`` types; ``bool``
    literals are concrete ``BOOL``. ``str`` values are accepted by
    :class:`~fhy_core.symbolic.expression.core.LiteralExpression` for other
    purposes (e.g. serialization round-trips) but have no core data type
    and are rejected here with :class:`NotImplementedError`. Callers that
    may receive string literals should either filter them earlier or
    catch ``NotImplementedError`` explicitly.

    Raises:
        NotImplementedError: If ``literal`` is a ``str``.
        ValueError: If ``literal`` is none of the supported literal types.

    """
    match literal:
        case bool():
            return CoreDataType.BOOL
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
    """Format an expression for inclusion in error messages."""
    return pformat_expression(expression, show_id=True)


def _is_weak_numerical_type(numerical_type: NumericalType) -> bool:
    return is_weak_core_data_type(
        _get_primitive_data_type(numerical_type).core_data_type
    )


_BOOLEAN_NUMERICAL_TYPE = NumericalType(PrimitiveDataType(CoreDataType.BOOL))


def _is_boolean_numerical_type(value_type: ExpressionValueType) -> bool:
    return (
        isinstance(value_type, NumericalType)
        and isinstance(value_type.data_type, PrimitiveDataType)
        and value_type.data_type.core_data_type == CoreDataType.BOOL
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
    if bit_width is None:
        return CoreDataType.FLOAT
    for candidate_bit_width, candidate in _REAL_FLOAT_CORE_DATA_TYPES_BY_BIT_WIDTH:
        if candidate_bit_width >= bit_width:
            return candidate
    raise FhYCoreTypeError(
        f"no real float core data type found for bit width {bit_width}"
    )


def _lift_integral_against_float_like(
    left_core_data_type: CoreDataType, right_core_data_type: CoreDataType
) -> tuple[CoreDataType, CoreDataType]:
    """Promote whichever side is integral to a real-float of matching bit width.

    Used by both true and floor division to align an integer operand with a
    float-like operand before falling through to ``promote_core_data_types``.
    Floor division rejects complex operands upstream, so the float-like check
    reduces to a real-float check there; true division relies on the lift to
    bridge the integer / complex lattice gap.
    """
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
    return left_core_data_type, right_core_data_type


class _TypeCheckContext:
    """Tracks the enclosing expression trace for context-rich error messages.

    The checker's public entry points (``synthesize``, ``check``) and each
    recursive ``_infer*`` call enter a new scope via :meth:`entering`, which
    pushes the expression being processed onto a stack. When an error is
    raised via :meth:`type_error`, the message includes the root expression
    (the first one pushed) and, when different, the current sub-expression
    (the most recently pushed one still live).

    The reason string passed to :meth:`type_error` should describe *what*
    went wrong without re-stating the expression itself; the context layer
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


def _try_resolve_native_constant_type(
    identifier_name: str,
    resolve_call_target: CallTargetResolver,
) -> tuple[Type, TypeQualifier] | None:
    """Return the IR type for a constant reference, or ``None`` if absent."""
    try:
        entry = resolve_call_target(identifier_name)
    except EntryLookupError:
        return None
    if not isinstance(entry, NativeConstant):
        return None
    return (
        NumericalType(
            PrimitiveDataType(get_result_core_data_type_for_sort(entry.sort))
        ),
        TypeQualifier.PARAM,
    )


def _promote_argument_qualifiers(
    argument_types: tuple[tuple[Type, TypeQualifier], ...],
) -> TypeQualifier:
    """Promote a sequence of argument qualifiers to a single result qualifier."""
    result: TypeQualifier = TypeQualifier.PARAM
    for _, qualifier in argument_types:
        result = promote_type_qualifiers(result, qualifier)
    return result


@register_pass(
    "fhy_core.types.checking.type_checker",
    "Bidirectionally synthesize and check expression types.",
)
class ExpressionTypeChecker(VisitablePass[Expression, tuple[Type, TypeQualifier]]):
    """Bidirectional type checker for expressions.

    Args:
        get_identifier_type: Callable mapping an :class:`Identifier` to
            its IR type and qualifier. Raises :class:`KeyError` to
            signal an unbound identifier; the type checker catches this
            and falls back to a registered ``NativeConstant`` lookup
            before raising a typed-error. Any other exception
            propagates unchanged.
        resolve_call_target: Callable that maps a function or constant
            name to its registered entry. Injected rather than hard-
            wired to the global registry so the type checker stays
            decoupled from registry-load ordering and is independently
            testable.
        defer_on_unknown_call: When ``True``, an unresolved call name
            in :meth:`_resolve_call_function` propagates its raw
            :class:`EntryLookupError` instead of being framed as a
            type error. Used by
            :class:`RegisteredFunctionBodyTypeChecker` to tolerate
            forward references inside a function body. Defaults to
            ``False`` (raise framed type errors).
    """

    _get_identifier_type: Callable[[Identifier], tuple[Type, TypeQualifier]]
    _resolve_call_target: CallTargetResolver
    _context: _TypeCheckContext
    _defer_on_unknown_call: bool

    def __init__(
        self,
        get_identifier_type: Callable[[Identifier], tuple[Type, TypeQualifier]],
        *,
        resolve_call_target: CallTargetResolver,
        defer_on_unknown_call: bool = False,
    ) -> None:
        super().__init__()
        self._get_identifier_type = get_identifier_type
        self._resolve_call_target = resolve_call_target
        self._context = _TypeCheckContext()
        self._defer_on_unknown_call = defer_on_unknown_call

    def synthesize(self, expression: Expression) -> tuple[Type, TypeQualifier]:
        """Synthesize a type for an expression."""
        return self._infer(expression)

    def check(
        self, expression: Expression, expected_type: Type
    ) -> tuple[Type, TypeQualifier]:
        """Check an expression against an expected type."""
        actual_type, actual_qualifier = self._infer(expression, expected_type)
        # `_infer` unwinds any `entering` scopes it opens before returning.
        # Re-enter `expression` here so `_check_expected_type` is framed
        # correctly whether `check` is called at top level or from within an
        # existing outer context (e.g. the late weak-literal rescue).
        with self._context.entering(expression):
            self._check_expected_type(expression, actual_type, expected_type)
        return actual_type, actual_qualifier

    # --- Visitor dispatch ----------------------------------------------------

    def visit_unary_expression(
        self, unary_expression: UnaryExpression
    ) -> tuple[Type, TypeQualifier]:
        """Infer and check the type of the unary expression."""
        return self._infer_unary_expression(unary_expression)

    def visit_binary_expression(
        self, binary_expression: BinaryExpression
    ) -> tuple[Type, TypeQualifier]:
        """Infer and check the type of the binary expression."""
        return self._infer_binary_expression(binary_expression)

    def visit_identifier_expression(
        self, identifier_expression: IdentifierExpression
    ) -> tuple[Type, TypeQualifier]:
        """Infer the type of the identifier expression."""
        with self._context.entering(identifier_expression):
            # The lookup contract: callers raise ``KeyError`` to signal
            # an unbound identifier. On ``KeyError`` we fall back to a
            # registered native constant before framing the failure as
            # a type error. Any other exception (e.g. ``AssertionError``
            # from a test fixture, or an unexpected runtime error)
            # propagates unchanged.
            try:
                identifier_type, identifier_qualifier = self._get_identifier_type(
                    identifier_expression.identifier
                )
            except KeyError as exc:
                constant_type = _try_resolve_native_constant_type(
                    identifier_expression.identifier.name_hint,
                    self._resolve_call_target,
                )
                if constant_type is not None:
                    return constant_type
                raise self._context.type_error(
                    f"identifier "
                    f"`{_format_expression(identifier_expression)}` is not bound"
                ) from exc
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
        """Infer the type of the literal expression."""
        return (
            NumericalType(
                PrimitiveDataType(
                    get_core_data_type_from_literal_type(literal_expression.value)
                )
            ),
            TypeQualifier.PARAM,
        )

    def visit_ternary_expression(
        self, ternary_expression: TernaryExpression
    ) -> tuple[Type, TypeQualifier]:
        """Infer and check the type of the ternary expression."""
        return self._infer_ternary_expression(ternary_expression)

    def visit_call_expression(
        self, call_expression: CallExpression
    ) -> tuple[Type, TypeQualifier]:
        """Infer and check the type of the call expression."""
        return self._infer_call_expression(call_expression)

    @override
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
            case TernaryExpression():
                return self._infer_ternary_expression(expression, expected_type)
            case CallExpression():
                return self._infer_call_expression(expression, expected_type)
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

            expected_core_data_type = _get_primitive_data_type(
                expected_value_type
            ).core_data_type
            literal_value = literal_expression.value
            if isinstance(literal_value, bool):
                resolved_core_data_type = resolve_literal_core_data_type(
                    literal_value, expected_core_data_type
                )
            else:
                resolved_core_data_type = resolve_literal_core_data_type(
                    _get_numeric_literal_value(literal_expression),
                    expected_core_data_type,
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

            operand_value_type = self._as_expression_value_type(
                unary_expression, operand_type
            )

            match unary_expression.operation:
                case UnaryOperation.POSITIVE:
                    if _is_boolean_numerical_type(operand_value_type):
                        raise self._context.type_error(
                            "unary positive is not defined for boolean operands"
                        )
                    return operand_value_type, operand_qualifier
                case UnaryOperation.NEGATE:
                    if isinstance(operand_value_type, IndexType):
                        raise self._context.type_error(
                            "unary negation is not defined for index types; the "
                            "resulting bounds and stride cannot be inferred safely"
                        )
                    if _is_boolean_numerical_type(operand_value_type):
                        raise self._context.type_error(
                            "unary negation is not defined for boolean operands"
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
                    if not _is_boolean_numerical_type(operand_value_type):
                        raise self._context.type_error(
                            f"logical NOT requires a boolean operand, but got "
                            f"{operand_value_type}"
                        )
                    return _BOOLEAN_NUMERICAL_TYPE, operand_qualifier
                case _:
                    raise NotImplementedError(
                        f"unary operation {unary_expression.operation} is not supported"
                    )

    def _infer_binary_expression(
        self,
        binary_expression: BinaryExpression,
        expected_type: Type | None = None,
    ) -> tuple[Type, TypeQualifier]:
        with self._context.entering(binary_expression):
            operation = binary_expression.operation
            left_expected, right_expected = (
                self._propagate_expected_type_to_literal_operands(
                    binary_expression, expected_type
                )
            )

            left_type, left_qualifier = self._infer(
                binary_expression.left, left_expected
            )
            right_type, right_qualifier = self._infer(
                binary_expression.right, right_expected
            )

            left_value_type = self._as_expression_value_type(
                binary_expression.left, left_type
            )
            right_value_type = self._as_expression_value_type(
                binary_expression.right, right_type
            )

            left_value_type, left_qualifier = self._apply_late_weak_literal_rescue(
                binary_expression.left,
                left_value_type,
                left_qualifier,
                right_value_type,
                operation,
            )
            right_value_type, right_qualifier = self._apply_late_weak_literal_rescue(
                binary_expression.right,
                right_value_type,
                right_qualifier,
                left_value_type,
                operation,
            )

            if operation in _LOGICAL_BOOLEAN_OPERATIONS:
                return self._infer_logical_binary(
                    operation,
                    left_value_type,
                    right_value_type,
                    left_qualifier,
                    right_qualifier,
                )
            if operation in _COMPARISON_OPERATIONS:
                return self._infer_comparison_binary(
                    operation,
                    left_value_type,
                    right_value_type,
                    left_qualifier,
                    right_qualifier,
                )

            if _is_boolean_numerical_type(
                left_value_type
            ) or _is_boolean_numerical_type(right_value_type):
                raise self._context.type_error(
                    f"the {operation.name.lower()} operation is not defined "
                    f"for boolean operands"
                )

            return self._dispatch_arithmetic_binary(
                binary_expression,
                operation,
                left_value_type,
                right_value_type,
                left_qualifier,
                right_qualifier,
            )

    def _infer_logical_binary(
        self,
        operation: BinaryOperation,
        left_value_type: ExpressionValueType,
        right_value_type: ExpressionValueType,
        left_qualifier: TypeQualifier,
        right_qualifier: TypeQualifier,
    ) -> tuple[Type, TypeQualifier]:
        if not (
            _is_boolean_numerical_type(left_value_type)
            and _is_boolean_numerical_type(right_value_type)
        ):
            raise self._context.type_error(
                f"logical {operation.name.lower()} requires boolean operands, "
                f"but got {left_value_type} and {right_value_type}"
            )
        return (
            _BOOLEAN_NUMERICAL_TYPE,
            promote_type_qualifiers(left_qualifier, right_qualifier),
        )

    def _infer_comparison_binary(
        self,
        operation: BinaryOperation,
        left_value_type: ExpressionValueType,
        right_value_type: ExpressionValueType,
        left_qualifier: TypeQualifier,
        right_qualifier: TypeQualifier,
    ) -> tuple[Type, TypeQualifier]:
        result_qualifier = promote_type_qualifiers(left_qualifier, right_qualifier)
        left_is_bool = _is_boolean_numerical_type(left_value_type)
        right_is_bool = _is_boolean_numerical_type(right_value_type)

        if operation in _ORDERING_OPERATIONS and (left_is_bool or right_is_bool):
            raise self._context.type_error(
                f"ordering operation {operation.name.lower()} is not defined for "
                f"boolean operands"
            )

        if operation in _EQUALITY_OPERATIONS and (left_is_bool or right_is_bool):
            if not (left_is_bool and right_is_bool):
                raise self._context.type_error(
                    f"equality operation {operation.name.lower()} is not defined "
                    f"between boolean and non-boolean operands "
                    f"({left_value_type}, {right_value_type})"
                )
            return _BOOLEAN_NUMERICAL_TYPE, result_qualifier

        if isinstance(left_value_type, IndexType) and isinstance(
            right_value_type, IndexType
        ):
            if operation in _EQUALITY_OPERATIONS:
                if not left_value_type.is_structurally_equivalent(right_value_type):
                    raise self._context.type_error(
                        f"equality on index types requires structural equivalence, "
                        f"but {left_value_type} and {right_value_type} differ"
                    )
                return _BOOLEAN_NUMERICAL_TYPE, result_qualifier
            raise self._context.type_error(
                f"ordering operation {operation.name.lower()} is not defined "
                f"between two index types"
            )

        if isinstance(left_value_type, IndexType) or isinstance(
            right_value_type, IndexType
        ):
            non_index_value_type = (
                right_value_type
                if isinstance(left_value_type, IndexType)
                else left_value_type
            )
            raise self._context.type_error(
                f"comparison {operation.name.lower()} is not defined between "
                f"index type and {non_index_value_type}"
            )

        # Both are non-boolean ``NumericalType`` here. The comparison
        # result is always ``BOOL``; this call only validates that the
        # two operand types share a common promoted type, raising
        # ``FhYCoreTypeError`` otherwise.
        promote_primitive_data_types(
            _get_primitive_data_type(left_value_type),
            _get_primitive_data_type(right_value_type),
        )
        return _BOOLEAN_NUMERICAL_TYPE, result_qualifier

    @staticmethod
    def _propagate_expected_type_to_literal_operands(
        binary_expression: BinaryExpression, expected_type: Type | None
    ) -> tuple[Type | None, Type | None]:
        if not (
            binary_expression.operation in _ARITHMETIC_OPERATIONS
            and isinstance(expected_type, NumericalType)
            and expected_type.is_scalar()
        ):
            return None, None
        left_expected = (
            expected_type
            if isinstance(binary_expression.left, LiteralExpression)
            else None
        )
        right_expected = (
            expected_type
            if isinstance(binary_expression.right, LiteralExpression)
            else None
        )
        return left_expected, right_expected

    def _apply_late_weak_literal_rescue(
        self,
        operand_expression: Expression,
        operand_value_type: ExpressionValueType,
        operand_qualifier: TypeQualifier,
        other_value_type: ExpressionValueType,
        operation: BinaryOperation,
    ) -> tuple[ExpressionValueType, TypeQualifier]:
        if not (
            (
                operation in _ARITHMETIC_OPERATIONS
                or operation in _ORDERING_OPERATIONS
                or operation in _EQUALITY_OPERATIONS
            )
            and isinstance(operand_expression, LiteralExpression)
            and isinstance(operand_value_type, NumericalType)
            and _is_weak_numerical_type(operand_value_type)
            and isinstance(other_value_type, NumericalType)
            and not _is_weak_numerical_type(other_value_type)
            and not _is_boolean_numerical_type(other_value_type)
        ):
            return operand_value_type, operand_qualifier
        checked_type, operand_qualifier = self.check(
            operand_expression, other_value_type
        )
        operand_value_type = self._as_expression_value_type(
            operand_expression, checked_type
        )
        return operand_value_type, operand_qualifier

    def _infer_ternary_expression(
        self,
        ternary_expression: TernaryExpression,
        expected_type: Type | None = None,
    ) -> tuple[Type, TypeQualifier]:
        with self._context.entering(ternary_expression):
            condition_type, condition_qualifier = self._infer(
                ternary_expression.condition, _BOOLEAN_NUMERICAL_TYPE
            )
            condition_value_type = self._as_expression_value_type(
                ternary_expression.condition, condition_type
            )
            if not _is_boolean_numerical_type(condition_value_type):
                raise self._context.type_error(
                    f"ternary condition must be boolean, but got {condition_value_type}"
                )

            branch_expected = self._ternary_branch_expected_type(expected_type)

            true_type, true_qualifier = self._infer(
                ternary_expression.true_value, branch_expected
            )
            false_type, false_qualifier = self._infer(
                ternary_expression.false_value, branch_expected
            )

            true_value_type = self._as_expression_value_type(
                ternary_expression.true_value, true_type
            )
            false_value_type = self._as_expression_value_type(
                ternary_expression.false_value, false_type
            )

            if not (
                isinstance(true_value_type, NumericalType)
                and isinstance(false_value_type, NumericalType)
            ):
                raise self._context.type_error(
                    "ternary branches must both be scalar numerical types, but "
                    f"got {true_value_type} and {false_value_type}"
                )

            result_primitive = promote_primitive_data_types(
                _get_primitive_data_type(true_value_type),
                _get_primitive_data_type(false_value_type),
            )
            result_qualifier = promote_type_qualifiers(
                condition_qualifier,
                promote_type_qualifiers(true_qualifier, false_qualifier),
            )
            return NumericalType(result_primitive), result_qualifier

    @staticmethod
    def _ternary_branch_expected_type(
        expected_type: Type | None,
    ) -> Type | None:
        if isinstance(expected_type, NumericalType) and expected_type.is_scalar():
            return expected_type
        return None

    def _infer_call_expression(
        self,
        call_expression: CallExpression,
        expected_type: Type | None = None,
    ) -> tuple[Type, TypeQualifier]:
        with self._context.entering(call_expression):
            entry = self._resolve_call_function(call_expression)
            if isinstance(entry, NativeConstant):
                raise self._context.type_error(
                    f"{call_expression.function_name!r} is a registered "
                    f"constant, not a function; reference it as an identifier "
                    f"instead of calling it"
                )
            self._check_call_arity(
                entry.name,
                len(entry.parameter_sorts),
                len(call_expression.arguments),
            )
            argument_types = tuple(
                self._infer(argument) for argument in call_expression.arguments
            )
            self._check_arguments_against_sorts(
                entry.name, entry.parameter_sorts, argument_types
            )
            result_type = NumericalType(
                PrimitiveDataType(get_result_core_data_type_for_sort(entry.result_sort))
            )
            return result_type, _promote_argument_qualifiers(argument_types)

    def _resolve_call_function(
        self, call_expression: CallExpression
    ) -> RegisteredFunction | NativeFunction | NativeConstant:
        try:
            return self._resolve_call_target(call_expression.function_name)
        except EntryLookupError as exc:
            if self._defer_on_unknown_call:
                raise
            raise self._context.type_error(
                f"call to unknown function {call_expression.function_name!r}: {exc}"
            ) from exc

    def _check_call_arity(
        self, function_name: str, expected_count: int, actual_count: int
    ) -> None:
        if actual_count == expected_count:
            return
        raise self._context.type_error(
            f"function {function_name!r} expects {expected_count} argument(s), "
            f"but got {actual_count}"
        )

    def _check_arguments_against_sorts(
        self,
        function_name: str,
        parameter_sorts: tuple[FunctionSort, ...],
        argument_types: tuple[tuple[Type, TypeQualifier], ...],
    ) -> None:
        for index, (parameter_sort, (argument_type, _)) in enumerate(
            zip(parameter_sorts, argument_types, strict=True)
        ):
            self._check_argument_sort(
                function_name, index, parameter_sort, argument_type
            )

    def _check_argument_sort(
        self,
        function_name: str,
        argument_index: int,
        parameter_sort: FunctionSort,
        argument_type: Type,
    ) -> None:
        if not isinstance(argument_type, NumericalType) or not isinstance(
            argument_type.data_type, PrimitiveDataType
        ):
            raise self._context.type_error(
                f"argument {argument_index} of function {function_name!r} "
                f"expects sort {parameter_sort}, but got {argument_type}"
            )
        core_data_type = argument_type.data_type.core_data_type
        if not is_core_data_type_compatible_with_sort(core_data_type, parameter_sort):
            raise self._context.type_error(
                f"argument {argument_index} of function {function_name!r} "
                f"expects sort {parameter_sort}, but got {argument_type}"
            )

    def _dispatch_arithmetic_binary(
        self,
        binary_expression: BinaryExpression,
        operation: BinaryOperation,
        left_value_type: ExpressionValueType,
        right_value_type: ExpressionValueType,
        left_qualifier: TypeQualifier,
        right_qualifier: TypeQualifier,
    ) -> tuple[Type, TypeQualifier]:
        result_qualifier = promote_type_qualifiers(left_qualifier, right_qualifier)

        if isinstance(left_value_type, NumericalType) and isinstance(
            right_value_type, NumericalType
        ):
            return (
                self._infer_arithmetic_numerical_pair(
                    operation, left_value_type, right_value_type
                ),
                result_qualifier,
            )

        elif operation in {
            BinaryOperation.DIVIDE,
            BinaryOperation.FLOOR_DIVIDE,
        }:
            raise self._context.type_error(
                "division is not defined for operands of index type"
            )

        elif isinstance(left_value_type, IndexType) and isinstance(
            right_value_type, IndexType
        ):
            raise self._context.type_error(
                f"the {operation.name.lower()} operation between two index types "
                "is not supported; arithmetic on two strided ranges does not "
                "generally produce a result whose index bounds and stride can be "
                "inferred safely. Index types are valid as operands of shift "
                "(``index +/- int``) or scale (``index * positive int literal``)."
            )

        elif operation == BinaryOperation.ADD:
            return (
                self._infer_index_shift_add(
                    binary_expression, left_value_type, right_value_type
                ),
                result_qualifier,
            )
        elif (
            operation == BinaryOperation.SUBTRACT
            and isinstance(left_value_type, IndexType)
            and isinstance(right_value_type, NumericalType)
        ):
            return (
                self._infer_index_shift_subtract(
                    binary_expression, left_value_type, right_value_type
                ),
                result_qualifier,
            )
        elif operation == BinaryOperation.MULTIPLY:
            return (
                self._infer_index_scale(
                    binary_expression, left_value_type, right_value_type
                ),
                result_qualifier,
            )
        else:
            raise self._context.type_error(
                f"the {operation.name.lower()} operation is not defined for "
                f"operands of types {left_value_type} and {right_value_type}; "
                "the resulting index bounds and stride cannot be inferred safely"
            )

    def _infer_arithmetic_numerical_pair(
        self,
        operation: BinaryOperation,
        left_value_type: NumericalType,
        right_value_type: NumericalType,
    ) -> NumericalType:
        if operation == BinaryOperation.DIVIDE:
            return NumericalType(
                self._get_division_primitive_data_type(
                    left_value_type, right_value_type
                )
            )
        if operation == BinaryOperation.FLOOR_DIVIDE:
            return NumericalType(
                self._get_floor_division_primitive_data_type(
                    left_value_type, right_value_type
                )
            )
        return NumericalType(
            promote_primitive_data_types(
                _get_primitive_data_type(left_value_type),
                _get_primitive_data_type(right_value_type),
            )
        )

    def _infer_index_shift_add(
        self,
        binary_expression: BinaryExpression,
        left_value_type: ExpressionValueType,
        right_value_type: ExpressionValueType,
    ) -> IndexType:
        if isinstance(left_value_type, IndexType) and isinstance(
            right_value_type, NumericalType
        ):
            if not _is_integral_numerical_type(right_value_type):
                raise self._context.type_error(
                    f"index shift requires an integral scalar offset, but the "
                    f"right operand has type {right_value_type}"
                )
            return self._shift_index_type(left_value_type, binary_expression.right)
        if isinstance(left_value_type, NumericalType) and isinstance(
            right_value_type, IndexType
        ):
            if not _is_integral_numerical_type(left_value_type):
                raise self._context.type_error(
                    f"index shift requires an integral scalar offset, but the "
                    f"left operand has type {left_value_type}"
                )
            return self._shift_index_type(right_value_type, binary_expression.left)
        # Unreachable: the dispatcher routes both-numerical and both-index cases
        # away before this helper, so mixed shapes are the only remaining ones.
        raise self._context.type_error(  # pragma: no cover
            f"the add operation is not defined for operands of types "
            f"{left_value_type} and {right_value_type}; the resulting "
            "index bounds and stride cannot be inferred safely"
        )

    def _infer_index_shift_subtract(
        self,
        binary_expression: BinaryExpression,
        left_value_type: IndexType,
        right_value_type: NumericalType,
    ) -> IndexType:
        if not _is_integral_numerical_type(right_value_type):
            raise self._context.type_error(
                f"index shift requires an integral scalar offset, but the "
                f"right operand has type {right_value_type}"
            )
        return self._shift_index_type(
            left_value_type, binary_expression.right, subtract=True
        )

    def _infer_index_scale(
        self,
        binary_expression: BinaryExpression,
        left_value_type: ExpressionValueType,
        right_value_type: ExpressionValueType,
    ) -> IndexType:
        if isinstance(left_value_type, IndexType) and isinstance(
            right_value_type, NumericalType
        ):
            scalar_literal = self._validate_index_scale_scalar(
                binary_expression.right, right_value_type, side="right"
            )
            return self._scale_index_type(left_value_type, scalar_literal)
        if isinstance(left_value_type, NumericalType) and isinstance(
            right_value_type, IndexType
        ):
            scalar_literal = self._validate_index_scale_scalar(
                binary_expression.left, left_value_type, side="left"
            )
            return self._scale_index_type(right_value_type, scalar_literal)
        # Unreachable: the dispatcher routes both-numerical and both-index cases
        # away before this helper, so mixed shapes are the only remaining ones.
        raise self._context.type_error(  # pragma: no cover
            f"the multiply operation is not defined for operands of types "
            f"{left_value_type} and {right_value_type}; the resulting "
            "index bounds and stride cannot be inferred safely"
        )

    def _validate_index_scale_scalar(
        self,
        scalar_expression: Expression,
        scalar_value_type: NumericalType,
        *,
        side: str,
    ) -> LiteralExpression:
        if not isinstance(scalar_expression, LiteralExpression):
            raise self._context.type_error(
                "index scaling requires a positive integer literal scalar, "
                f"but the {side} operand "
                f"`{_format_expression(scalar_expression)}` is not a "
                "literal expression"
            )
        if not _is_integral_numerical_type(scalar_value_type):
            raise self._context.type_error(
                "index scaling requires a positive integer literal scalar, "
                f"but the {side} operand has non-integral type "
                f"{scalar_value_type}"
            )
        return scalar_expression

    # --- Context-aware helpers ----------------------------------------------

    def _as_expression_value_type(
        self, expression: Expression, type_: Type
    ) -> ExpressionValueType:
        if isinstance(type_, IndexType):
            stride = type_.stride
            if (
                isinstance(stride, LiteralExpression)
                and is_strict_int(stride.value)
                and stride.value == 0
            ):
                raise self._context.type_error(
                    "index type with stride `0` is not allowed; stride must be "
                    "a non-zero integer or a non-literal expression"
                )
            return type_
        elif not isinstance(type_, NumericalType):  # pragma: no cover
            # Defensive: callers always supply IndexType, NumericalType, or a
            # type that becomes one through Type.bind_template; a future Type
            # subclass that bypasses both paths should fail loud.
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

    def _scale_index_type(
        self, index_type: IndexType, scalar: LiteralExpression
    ) -> IndexType:
        value = scalar.value
        if not is_strict_int(value):  # pragma: no cover
            # Defensive: `_validate_index_scale_scalar` rejects non-integer
            # scalars before this method is reached from the dispatcher.
            raise self._context.type_error(
                f"index scaling requires an integer literal scalar, but got scalar "
                f"value {value!r}"
            )
        if value <= 0:
            raise self._context.type_error(
                f"index scaling requires a positive integer literal scalar, but "
                f"got scalar value {value}"
            )
        stride = index_type.stride
        if isinstance(stride, LiteralExpression) and is_strict_int(stride.value):
            new_stride: Expression = LiteralExpression(value * stride.value)
        else:
            new_stride = scalar * stride
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

        left_core_data_type, right_core_data_type = _lift_integral_against_float_like(
            left_core_data_type, right_core_data_type
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

        left_core_data_type, right_core_data_type = _lift_integral_against_float_like(
            left_core_data_type, right_core_data_type
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
    return ExpressionTypeChecker(
        get_identifier_type, resolve_call_target=get_registered_entry
    ).synthesize(expression)


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
    return ExpressionTypeChecker(
        get_identifier_type, resolve_call_target=get_registered_entry
    ).check(expression, expected_type)
