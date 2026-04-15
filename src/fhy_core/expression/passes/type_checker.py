"""Bidirectional type checking and inference for expressions."""

__all__ = [
    "get_core_data_type_from_literal_type",
    "synthesize_expression_type",
    "check_expression_type",
]

from typing import Callable, TypeAlias

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
    is_weak_core_data_type,
    promote_primitive_data_types,
    promote_type_qualifiers,
    resolve_literal_core_data_type,
)

ExpressionValueType: TypeAlias = NumericalType | IndexType

_ARITHMETIC_OPERATIONS = {
    BinaryOperation.ADD,
    BinaryOperation.SUBTRACT,
    BinaryOperation.MULTIPLY,
    BinaryOperation.DIVIDE,
    BinaryOperation.FLOOR_DIVIDE,
    BinaryOperation.MODULO,
    BinaryOperation.POWER,
}


def get_core_data_type_from_literal_type(literal: LiteralType) -> CoreDataType:
    """Return the weak core data type assigned to a literal."""
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


def _as_expression_value_type(
    expression: Expression, type_: Type
) -> ExpressionValueType:
    if isinstance(type_, TupleType):
        raise FhYCoreTypeError(
            f"Expression {_format_expression(expression)} resolves to tuple type "
            f"{type_}, which is not valid here."
        )
    elif isinstance(type_, IndexType):
        return type_
    elif not isinstance(type_, NumericalType):
        raise FhYCoreTypeError(
            f"Expression {_format_expression(expression)} must resolve to a scalar "
            f"numerical type or index type, not {type_}."
        )

    elif not isinstance(type_.data_type, PrimitiveDataType):
        raise FhYCoreTypeError(
            f"Expression {_format_expression(expression)} must resolve to a primitive "
            f"numerical type, not {type_}."
        )
    elif not type_.is_scalar():
        raise FhYCoreTypeError(
            f"Expression {_format_expression(expression)} resolves to tensor type "
            f"{type_}, but only scalar numerical and index types are allowed in "
            "expressions."
        )
    else:
        return type_


def _is_weak_numerical_type(numerical_type: NumericalType) -> bool:
    return is_weak_core_data_type(numerical_type.data_type.core_data_type)


def _check_expected_type(
    expression: Expression, actual_type: Type, expected_type: Type
) -> None:
    if isinstance(actual_type, IndexType) and isinstance(expected_type, IndexType):
        if not actual_type.is_structurally_equivalent(expected_type):
            raise FhYCoreTypeError(
                f"Expression {_format_expression(expression)} has type {actual_type}, "
                f"which is incompatible with expected type {expected_type}."
            )
        return

    actual_value_type = _as_expression_value_type(expression, actual_type)
    expected_value_type = _as_expression_value_type(expression, expected_type)
    if isinstance(actual_value_type, IndexType) or isinstance(
        expected_value_type, IndexType
    ):
        raise FhYCoreTypeError(
            f"Expression {_format_expression(expression)} has type {actual_type}, "
            f"which is incompatible with expected type {expected_type}."
        )

    promoted_type = promote_primitive_data_types(
        actual_value_type.data_type,
        expected_value_type.data_type,
    )
    if promoted_type.core_data_type != expected_value_type.data_type.core_data_type:
        raise FhYCoreTypeError(
            f"Expression {_format_expression(expression)} has type {actual_type}, "
            f"which is incompatible with expected type {expected_type}."
        )


def _shift_index_type(
    index_type: IndexType, offset_expression: Expression, *, subtract: bool = False
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


@register_pass(
    "fhy_core.expression.type_check",
    "Bidirectionally synthesize and check expression types.",
)
class ExpressionTypeChecker(VisitablePass[Expression, tuple[Type, TypeQualifier]]):
    """Bidirectional type checker for expressions."""

    _get_identifier_type: Callable[[Identifier], tuple[Type, TypeQualifier]]

    def __init__(
        self, get_identifier_type: Callable[[Identifier], tuple[Type, TypeQualifier]]
    ) -> None:
        super().__init__()
        self._get_identifier_type = get_identifier_type

    def synthesize(self, expression: Expression) -> tuple[Type, TypeQualifier]:
        """Synthesize a type for an expression."""
        return self._infer(expression)

    def check(
        self, expression: Expression, expected_type: Type
    ) -> tuple[Type, TypeQualifier]:
        """Check an expression against an expected type."""
        actual_type, actual_qualifier = self._infer(expression, expected_type)
        _check_expected_type(expression, actual_type, expected_type)
        return actual_type, actual_qualifier

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
        identifier_type, identifier_qualifier = self._get_identifier_type(
            identifier_expression.identifier
        )
        if identifier_qualifier == TypeQualifier.OUTPUT:
            raise FhYCoreTypeError(
                f"Cannot read from variable "
                f'"{_format_expression(identifier_expression)}" '
                'with "output" type qualifier.'
            )
        return identifier_type, identifier_qualifier

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
        if expected_type is None:
            return self.visit_literal_expression(literal_expression)

        expected_value_type = _as_expression_value_type(
            literal_expression, expected_type
        )
        if isinstance(expected_value_type, IndexType):
            raise FhYCoreTypeError(
                f"Literal {_format_expression(literal_expression)} cannot be checked "
                f"against index type {expected_type}."
            )

        resolved_core_data_type = resolve_literal_core_data_type(
            literal_expression.value, expected_value_type.data_type.core_data_type
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
        operand_type, operand_qualifier = self._infer(
            unary_expression.operand, expected_type
        )
        if operand_qualifier == TypeQualifier.OUTPUT:
            raise RuntimeError('"output" type qualifier should not be possible here.')

        operand_value_type = _as_expression_value_type(unary_expression, operand_type)

        match unary_expression.operation:
            case UnaryOperation.POSITIVE:
                return operand_value_type, operand_qualifier
            case UnaryOperation.NEGATE:
                if isinstance(operand_value_type, IndexType):
                    raise FhYCoreTypeError(
                        "Index negation is not supported because the resulting "
                        "bounds and stride are not inferred safely."
                    )
                if isinstance(
                    unary_expression.operand, LiteralExpression
                ) and _is_weak_numerical_type(operand_value_type):
                    return (
                        NumericalType(
                            PrimitiveDataType(
                                get_core_data_type_from_literal_type(
                                    -unary_expression.operand.value
                                )
                            )
                        ),
                        operand_qualifier,
                    )
                return operand_value_type, operand_qualifier
            case UnaryOperation.LOGICAL_NOT:
                raise NotImplementedError("Boolean result types are not yet supported.")

    def _infer_binary_expression(
        self,
        binary_expression: BinaryExpression,
        expected_type: Type | None = None,
    ) -> tuple[Type, TypeQualifier]:
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
            raise RuntimeError('"output" type qualifier should not be possible here.')

        left_value_type = _as_expression_value_type(binary_expression.left, left_type)
        right_value_type = _as_expression_value_type(
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
            left_value_type, left_qualifier = self.check(
                binary_expression.left, right_value_type
            )
        if (
            binary_expression.operation in _ARITHMETIC_OPERATIONS
            and isinstance(binary_expression.right, LiteralExpression)
            and isinstance(right_value_type, NumericalType)
            and _is_weak_numerical_type(right_value_type)
            and isinstance(left_value_type, NumericalType)
            and not _is_weak_numerical_type(left_value_type)
        ):
            right_value_type, right_qualifier = self.check(
                binary_expression.right, left_value_type
            )

        left_value_type = _as_expression_value_type(
            binary_expression.left, left_value_type
        )
        right_value_type = _as_expression_value_type(
            binary_expression.right, right_value_type
        )

        if binary_expression.operation in _ARITHMETIC_OPERATIONS:
            if isinstance(left_value_type, NumericalType) and isinstance(
                right_value_type, NumericalType
            ):
                return (
                    NumericalType(
                        promote_primitive_data_types(
                            left_value_type.data_type, right_value_type.data_type
                        )
                    ),
                    promote_type_qualifiers(left_qualifier, right_qualifier),
                )
            if (
                binary_expression.operation == BinaryOperation.ADD
                and isinstance(left_value_type, IndexType)
                and isinstance(right_value_type, NumericalType)
            ):
                return (
                    _shift_index_type(left_value_type, binary_expression.right),
                    promote_type_qualifiers(left_qualifier, right_qualifier),
                )
            if (
                binary_expression.operation == BinaryOperation.ADD
                and isinstance(left_value_type, NumericalType)
                and isinstance(right_value_type, IndexType)
            ):
                return (
                    _shift_index_type(right_value_type, binary_expression.left),
                    promote_type_qualifiers(left_qualifier, right_qualifier),
                )
            if (
                binary_expression.operation == BinaryOperation.SUBTRACT
                and isinstance(left_value_type, IndexType)
                and isinstance(right_value_type, NumericalType)
            ):
                return (
                    _shift_index_type(
                        left_value_type,
                        binary_expression.right,
                        subtract=True,
                    ),
                    promote_type_qualifiers(left_qualifier, right_qualifier),
                )
            raise FhYCoreTypeError(
                "This operation is not supported for index types because the "
                "resulting index bounds/stride are not inferred safely."
            )

        raise NotImplementedError("Boolean result types are not yet supported.")


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
