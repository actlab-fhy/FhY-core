"""Tests for expression type checking and literal promotion."""

import pytest
from fhy_core.expression.core import (
    BinaryExpression,
    BinaryOperation,
    IdentifierExpression,
    LiteralExpression,
    UnaryExpression,
    UnaryOperation,
)
from fhy_core.expression.passes.type_checker import (
    ExpressionTypeChecker,
    get_core_data_type_from_literal_type,
)
from fhy_core.identifier import Identifier
from fhy_core.types import (
    CoreDataType,
    FhYCoreTypeError,
    IndexType,
    NumericalType,
    PrimitiveDataType,
    TupleType,
    TypeQualifier,
)


def _assert_core_data_type(type_, expected_core_data_type: CoreDataType) -> None:
    assert isinstance(type_, NumericalType)
    assert isinstance(type_.data_type, PrimitiveDataType)
    assert type_.data_type.core_data_type == expected_core_data_type


def _assert_index_type(type_: object, expected_type: IndexType) -> None:
    assert isinstance(type_, IndexType)
    assert type_.is_structurally_equivalent(expected_type)


def test_get_core_data_type_from_literal_type_returns_weak_types():
    """Test get_core_data_type_from_literal_type function with various literals."""
    assert get_core_data_type_from_literal_type(1) == CoreDataType.UINT
    assert get_core_data_type_from_literal_type(-1) == CoreDataType.INT
    assert get_core_data_type_from_literal_type(1.5) == CoreDataType.FLOAT


def test_unary_negation_of_positive_integer_literal_becomes_weak_int():
    """Test unary negation of positive integer literal."""
    checker = ExpressionTypeChecker(
        lambda _: (
            NumericalType(PrimitiveDataType(CoreDataType.INT32)),
            TypeQualifier.PARAM,
        )
    )

    result_type, result_qualifier = checker.visit(
        UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(5))
    )

    _assert_core_data_type(result_type, CoreDataType.INT)
    assert result_qualifier == TypeQualifier.PARAM


def test_positive_integer_literal_upgrades_to_signed_context():
    """Test positive integer literal upgrades to signed context."""
    identifier = Identifier("x")
    checker = ExpressionTypeChecker(
        lambda seen_identifier: (
            NumericalType(PrimitiveDataType(CoreDataType.INT32)),
            TypeQualifier.PARAM,
        )
        if seen_identifier == identifier
        else (_ for _ in ()).throw(AssertionError("Unexpected identifier lookup"))
    )

    result_type, result_qualifier = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            IdentifierExpression(identifier),
            LiteralExpression(1),
        )
    )

    _assert_core_data_type(result_type, CoreDataType.INT32)
    assert result_qualifier == TypeQualifier.PARAM


def test_integer_literals_remain_weak_without_context():
    """Test that integer literals remain weak without context."""
    checker = ExpressionTypeChecker(
        lambda _: (
            NumericalType(PrimitiveDataType(CoreDataType.INT32)),
            TypeQualifier.PARAM,
        )
    )

    result_type, result_qualifier = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            LiteralExpression(255),
            LiteralExpression(256),
        )
    )

    _assert_core_data_type(result_type, CoreDataType.UINT)
    assert result_qualifier == TypeQualifier.PARAM


def test_integer_literal_can_upgrade_to_float_context():
    """Test that integer literals can upgrade to float context."""
    identifier = Identifier("y")
    checker = ExpressionTypeChecker(
        lambda seen_identifier: (
            NumericalType(PrimitiveDataType(CoreDataType.FLOAT32)),
            TypeQualifier.PARAM,
        )
        if seen_identifier == identifier
        else (_ for _ in ()).throw(AssertionError("Unexpected identifier lookup"))
    )

    result_type, result_qualifier = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            IdentifierExpression(identifier),
            LiteralExpression(1),
        )
    )

    _assert_core_data_type(result_type, CoreDataType.FLOAT32)
    assert result_qualifier == TypeQualifier.PARAM


def test_check_literal_against_expected_type_resolves_weak_literal():
    """Test that checking a literal against an expected type resolves weak literal."""
    checker = ExpressionTypeChecker(
        lambda _: (
            NumericalType(PrimitiveDataType(CoreDataType.INT32)),
            TypeQualifier.PARAM,
        )
    )

    result_type, result_qualifier = checker.check(
        LiteralExpression(256),
        NumericalType(PrimitiveDataType(CoreDataType.UINT16)),
    )

    _assert_core_data_type(result_type, CoreDataType.UINT16)
    assert result_qualifier == TypeQualifier.PARAM


def test_check_integer_literal_against_float_context_uses_context_type():
    """Test integer literal checking preserves concrete float context."""
    checker = ExpressionTypeChecker(
        lambda _: (
            NumericalType(PrimitiveDataType(CoreDataType.INT32)),
            TypeQualifier.PARAM,
        )
    )

    result_type, result_qualifier = checker.check(
        LiteralExpression(1),
        NumericalType(PrimitiveDataType(CoreDataType.FLOAT32)),
    )

    _assert_core_data_type(result_type, CoreDataType.FLOAT32)
    assert result_qualifier == TypeQualifier.PARAM


def test_check_binary_expression_uses_expected_type_bidirectionally():
    """Test checking an arithmetic expression uses expected type bidirectionally."""
    checker = ExpressionTypeChecker(
        lambda _: (
            NumericalType(PrimitiveDataType(CoreDataType.INT32)),
            TypeQualifier.PARAM,
        )
    )

    result_type, result_qualifier = checker.check(
        BinaryExpression(
            BinaryOperation.ADD,
            LiteralExpression(255),
            LiteralExpression(256),
        ),
        NumericalType(PrimitiveDataType(CoreDataType.UINT16)),
    )

    _assert_core_data_type(result_type, CoreDataType.UINT16)
    assert result_qualifier == TypeQualifier.PARAM


def test_integer_division_produces_float_type():
    """Test that integer division produces a float result type."""
    left_identifier = Identifier("left")
    right_identifier = Identifier("right")
    checker = ExpressionTypeChecker(
        lambda seen_identifier: (
            NumericalType(PrimitiveDataType(CoreDataType.INT32)),
            TypeQualifier.PARAM,
        )
        if seen_identifier in {left_identifier, right_identifier}
        else (_ for _ in ()).throw(AssertionError("Unexpected identifier lookup"))
    )

    result_type, result_qualifier = checker.visit(
        BinaryExpression(
            BinaryOperation.DIVIDE,
            IdentifierExpression(left_identifier),
            IdentifierExpression(right_identifier),
        )
    )

    _assert_core_data_type(result_type, CoreDataType.FLOAT32)
    assert result_qualifier == TypeQualifier.PARAM


def test_weak_integer_division_produces_weak_float_type():
    """Test that weak integer division stays weakly typed as float."""
    checker = ExpressionTypeChecker(
        lambda _: (
            NumericalType(PrimitiveDataType(CoreDataType.INT32)),
            TypeQualifier.PARAM,
        )
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.DIVIDE,
            LiteralExpression(4),
            LiteralExpression(2),
        )
    )

    _assert_core_data_type(result_type, CoreDataType.FLOAT)


def test_floor_division_of_integers_produces_integer_type():
    """Test that integer floor division produces an integer result type."""
    left_identifier = Identifier("left")
    right_identifier = Identifier("right")
    checker = ExpressionTypeChecker(
        lambda seen_identifier: (
            NumericalType(PrimitiveDataType(CoreDataType.INT32)),
            TypeQualifier.PARAM,
        )
        if seen_identifier in {left_identifier, right_identifier}
        else (_ for _ in ()).throw(AssertionError("Unexpected identifier lookup"))
    )

    result_type, result_qualifier = checker.visit(
        BinaryExpression(
            BinaryOperation.FLOOR_DIVIDE,
            IdentifierExpression(left_identifier),
            IdentifierExpression(right_identifier),
        )
    )

    _assert_core_data_type(result_type, CoreDataType.INT32)
    assert result_qualifier == TypeQualifier.PARAM


def test_floor_division_of_float_and_int_produces_float_type():
    """Test that floor division preserves float typing when a float is present."""
    left_identifier = Identifier("left")
    right_identifier = Identifier("right")
    checker = ExpressionTypeChecker(
        lambda seen_identifier: (
            NumericalType(PrimitiveDataType(CoreDataType.FLOAT32)),
            TypeQualifier.PARAM,
        )
        if seen_identifier == left_identifier
        else (
            NumericalType(PrimitiveDataType(CoreDataType.INT32)),
            TypeQualifier.PARAM,
        )
        if seen_identifier == right_identifier
        else (_ for _ in ()).throw(AssertionError("Unexpected identifier lookup"))
    )

    result_type, result_qualifier = checker.visit(
        BinaryExpression(
            BinaryOperation.FLOOR_DIVIDE,
            IdentifierExpression(left_identifier),
            IdentifierExpression(right_identifier),
        )
    )

    _assert_core_data_type(result_type, CoreDataType.FLOAT32)
    assert result_qualifier == TypeQualifier.PARAM


def test_tensor_type_is_rejected_in_expression_typing():
    """Test that tensor-valued expressions are rejected in expression typing."""
    identifier = Identifier("tensor")
    checker = ExpressionTypeChecker(
        lambda seen_identifier: (
            NumericalType(
                PrimitiveDataType(CoreDataType.FLOAT32),
                [LiteralExpression(4)],
            ),
            TypeQualifier.PARAM,
        )
        if seen_identifier == identifier
        else (_ for _ in ()).throw(AssertionError("Unexpected identifier lookup"))
    )

    with pytest.raises(NotImplementedError):
        checker.visit(
            BinaryExpression(
                BinaryOperation.ADD,
                IdentifierExpression(identifier),
                LiteralExpression(1),
            )
        )


def test_tensor_identifier_is_rejected_immediately():
    """Test that tensor identifiers are rejected outside compound expressions."""
    identifier = Identifier("tensor")
    checker = ExpressionTypeChecker(
        lambda seen_identifier: (
            NumericalType(
                PrimitiveDataType(CoreDataType.FLOAT32),
                [LiteralExpression(4)],
            ),
            TypeQualifier.PARAM,
        )
        if seen_identifier == identifier
        else (_ for _ in ()).throw(AssertionError("Unexpected identifier lookup"))
    )

    with pytest.raises(NotImplementedError):
        checker.visit(IdentifierExpression(identifier))


def test_tuple_identifier_is_rejected_immediately():
    """Test that tuple identifiers are rejected in expressions."""
    identifier = Identifier("pair")
    checker = ExpressionTypeChecker(
        lambda seen_identifier: (
            TupleType(
                [
                    NumericalType(PrimitiveDataType(CoreDataType.INT32)),
                    NumericalType(PrimitiveDataType(CoreDataType.INT32)),
                ]
            ),
            TypeQualifier.PARAM,
        )
        if seen_identifier == identifier
        else (_ for _ in ()).throw(AssertionError("Unexpected identifier lookup"))
    )

    with pytest.raises(NotImplementedError):
        checker.visit(IdentifierExpression(identifier))


def test_index_plus_scalar_produces_shifted_index_type():
    """Test that adding a scalar to an index produces shifted index type."""
    identifier = Identifier("idx")
    upper_bound_identifier = Identifier("N")
    index_type = IndexType(
        LiteralExpression(1),
        IdentifierExpression(upper_bound_identifier),
        LiteralExpression(2),
    )
    checker = ExpressionTypeChecker(
        lambda seen_identifier: (index_type, TypeQualifier.PARAM)
        if seen_identifier == identifier
        else (_ for _ in ()).throw(AssertionError("Unexpected identifier lookup"))
    )

    result_type, result_qualifier = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            IdentifierExpression(identifier),
            LiteralExpression(3),
        )
    )

    expected_type = IndexType(
        LiteralExpression(1) + LiteralExpression(3),
        IdentifierExpression(upper_bound_identifier) + LiteralExpression(3),
        LiteralExpression(2),
    )
    _assert_index_type(result_type, expected_type)
    assert result_qualifier == TypeQualifier.PARAM


def test_scalar_plus_index_produces_shifted_index_type():
    """Test that adding an index to a scalar produces shifted index type."""
    identifier = Identifier("idx")
    index_type = IndexType(
        LiteralExpression(1),
        LiteralExpression(10),
        None,
    )
    checker = ExpressionTypeChecker(
        lambda seen_identifier: (index_type, TypeQualifier.PARAM)
        if seen_identifier == identifier
        else (_ for _ in ()).throw(AssertionError("Unexpected identifier lookup"))
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            LiteralExpression(2),
            IdentifierExpression(identifier),
        )
    )

    expected_type = IndexType(
        LiteralExpression(1) + LiteralExpression(2),
        LiteralExpression(10) + LiteralExpression(2),
        None,
    )
    _assert_index_type(result_type, expected_type)


def test_index_minus_scalar_produces_shifted_index_type():
    """Test that subtracting a scalar from an index produces shifted index type."""
    identifier = Identifier("idx")
    index_type = IndexType(
        LiteralExpression(2),
        LiteralExpression(12),
        LiteralExpression(3),
    )
    checker = ExpressionTypeChecker(
        lambda seen_identifier: (index_type, TypeQualifier.PARAM)
        if seen_identifier == identifier
        else (_ for _ in ()).throw(AssertionError("Unexpected identifier lookup"))
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.SUBTRACT,
            IdentifierExpression(identifier),
            LiteralExpression(1),
        )
    )

    expected_type = IndexType(
        LiteralExpression(2) - LiteralExpression(1),
        LiteralExpression(12) - LiteralExpression(1),
        LiteralExpression(3),
    )
    _assert_index_type(result_type, expected_type)


def test_unary_positive_index_preserves_index_type():
    """Test that unary positive preserves index types."""
    identifier = Identifier("idx")
    index_type = IndexType(
        LiteralExpression(1),
        LiteralExpression(8),
        None,
    )
    checker = ExpressionTypeChecker(
        lambda seen_identifier: (index_type, TypeQualifier.PARAM)
        if seen_identifier == identifier
        else (_ for _ in ()).throw(AssertionError("Unexpected identifier lookup"))
    )

    result_type, _ = checker.visit(
        UnaryExpression(UnaryOperation.POSITIVE, IdentifierExpression(identifier))
    )

    _assert_index_type(result_type, index_type)


def test_unary_negation_of_index_is_rejected():
    """Test that unary negation of index types is rejected."""
    identifier = Identifier("idx")
    index_type = IndexType(
        LiteralExpression(1),
        LiteralExpression(8),
        None,
    )
    checker = ExpressionTypeChecker(
        lambda seen_identifier: (index_type, TypeQualifier.PARAM)
        if seen_identifier == identifier
        else (_ for _ in ()).throw(AssertionError("Unexpected identifier lookup"))
    )

    with pytest.raises(FhYCoreTypeError):
        checker.visit(
            UnaryExpression(UnaryOperation.NEGATE, IdentifierExpression(identifier))
        )


def test_index_times_positive_integer_literal_scales_bounds_and_stride():
    """idx * k scales bounds and stride by a positive integer literal k."""
    identifier = Identifier("idx")
    index_type = IndexType(
        LiteralExpression(1), LiteralExpression(8), LiteralExpression(2)
    )
    checker = ExpressionTypeChecker(
        lambda seen_identifier: (index_type, TypeQualifier.PARAM)
        if seen_identifier == identifier
        else (_ for _ in ()).throw(AssertionError("Unexpected identifier lookup"))
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.MULTIPLY,
            IdentifierExpression(identifier),
            LiteralExpression(3),
        )
    )

    expected = IndexType(
        LiteralExpression(3) * LiteralExpression(1),
        LiteralExpression(3) * LiteralExpression(8),
        LiteralExpression(3) * LiteralExpression(2),
    )
    _assert_index_type(result_type, expected)


def test_positive_integer_literal_times_index_scales_symmetrically():
    """k * idx produces the same shifted type as idx * k."""
    identifier = Identifier("idx")
    index_type = IndexType(LiteralExpression(1), LiteralExpression(8), None)
    checker = ExpressionTypeChecker(
        lambda seen_identifier: (index_type, TypeQualifier.PARAM)
        if seen_identifier == identifier
        else (_ for _ in ()).throw(AssertionError("Unexpected identifier lookup"))
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.MULTIPLY,
            LiteralExpression(3),
            IdentifierExpression(identifier),
        )
    )

    expected = IndexType(
        LiteralExpression(3) * LiteralExpression(1),
        LiteralExpression(3) * LiteralExpression(8),
        LiteralExpression(3),
    )
    _assert_index_type(result_type, expected)


def test_index_scaling_by_one_preserves_unit_stride():
    """k=1 keeps stride=None (unit) since there is no real scaling."""
    identifier = Identifier("idx")
    index_type = IndexType(LiteralExpression(1), LiteralExpression(8), None)
    checker = ExpressionTypeChecker(
        lambda seen_identifier: (index_type, TypeQualifier.PARAM)
        if seen_identifier == identifier
        else (_ for _ in ()).throw(AssertionError("Unexpected identifier lookup"))
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.MULTIPLY,
            LiteralExpression(1),
            IdentifierExpression(identifier),
        )
    )

    expected = IndexType(
        LiteralExpression(1) * LiteralExpression(1),
        LiteralExpression(1) * LiteralExpression(8),
        None,
    )
    _assert_index_type(result_type, expected)


def test_index_times_non_positive_integer_literal_is_rejected():
    """Index scaling requires a positive integer literal (zero rejected)."""
    identifier = Identifier("idx")
    index_type = IndexType(LiteralExpression(1), LiteralExpression(8), None)
    checker = ExpressionTypeChecker(
        lambda seen_identifier: (index_type, TypeQualifier.PARAM)
        if seen_identifier == identifier
        else (_ for _ in ()).throw(AssertionError("Unexpected identifier lookup"))
    )

    with pytest.raises(FhYCoreTypeError):
        checker.visit(
            BinaryExpression(
                BinaryOperation.MULTIPLY,
                IdentifierExpression(identifier),
                LiteralExpression(0),
            )
        )


def test_index_times_non_literal_scalar_is_rejected():
    """Index scaling fails if the scalar is not a literal integer."""
    identifier = Identifier("idx")
    scalar = Identifier("k")
    index_type = IndexType(LiteralExpression(1), LiteralExpression(8), None)
    scalar_type = NumericalType(PrimitiveDataType(CoreDataType.INT32))
    checker = ExpressionTypeChecker(
        lambda seen_identifier: (index_type, TypeQualifier.PARAM)
        if seen_identifier == identifier
        else (scalar_type, TypeQualifier.PARAM)
        if seen_identifier == scalar
        else (_ for _ in ()).throw(AssertionError("Unexpected identifier lookup"))
    )

    with pytest.raises(FhYCoreTypeError):
        checker.visit(
            BinaryExpression(
                BinaryOperation.MULTIPLY,
                IdentifierExpression(scalar),
                IdentifierExpression(identifier),
            )
        )


def test_index_times_float_literal_is_rejected():
    """Index scaling requires an integer literal, not a float."""
    identifier = Identifier("idx")
    index_type = IndexType(LiteralExpression(1), LiteralExpression(8), None)
    checker = ExpressionTypeChecker(
        lambda seen_identifier: (index_type, TypeQualifier.PARAM)
        if seen_identifier == identifier
        else (_ for _ in ()).throw(AssertionError("Unexpected identifier lookup"))
    )

    with pytest.raises(FhYCoreTypeError):
        checker.visit(
            BinaryExpression(
                BinaryOperation.MULTIPLY,
                IdentifierExpression(identifier),
                LiteralExpression(2.0),
            )
        )


def test_index_division_is_rejected():
    """Test that index division is rejected explicitly."""
    identifier = Identifier("idx")
    index_type = IndexType(
        LiteralExpression(1),
        LiteralExpression(8),
        None,
    )
    checker = ExpressionTypeChecker(
        lambda seen_identifier: (index_type, TypeQualifier.PARAM)
        if seen_identifier == identifier
        else (_ for _ in ()).throw(AssertionError("Unexpected identifier lookup"))
    )

    with pytest.raises(FhYCoreTypeError):
        checker.visit(
            BinaryExpression(
                BinaryOperation.DIVIDE,
                IdentifierExpression(identifier),
                LiteralExpression(2),
            )
        )


def test_index_plus_float_is_rejected():
    """Test that index offsets must be integral scalars."""
    identifier = Identifier("idx")
    index_type = IndexType(
        LiteralExpression(1),
        LiteralExpression(8),
        None,
    )
    checker = ExpressionTypeChecker(
        lambda seen_identifier: (index_type, TypeQualifier.PARAM)
        if seen_identifier == identifier
        else (_ for _ in ()).throw(AssertionError("Unexpected identifier lookup"))
    )

    with pytest.raises(FhYCoreTypeError):
        checker.visit(
            BinaryExpression(
                BinaryOperation.ADD,
                IdentifierExpression(identifier),
                LiteralExpression(1.5),
            )
        )


def _make_index_checker(
    bindings: dict[Identifier, tuple[IndexType, TypeQualifier]],
) -> ExpressionTypeChecker:
    def lookup(seen: Identifier) -> tuple[IndexType, TypeQualifier]:
        if seen in bindings:
            return bindings[seen]
        raise AssertionError(f"Unexpected identifier lookup: {seen}")

    return ExpressionTypeChecker(lookup)


def test_index_plus_index_combines_bounds_when_strides_are_none():
    """i ∈ [0, 10] + j ∈ [1, 5] ⟹ [0+1, 10+5] = [1, 15] with unit stride."""
    left = Identifier("i")
    right = Identifier("j")
    left_index = IndexType(LiteralExpression(1), LiteralExpression(10), None)
    right_index = IndexType(LiteralExpression(1), LiteralExpression(5), None)
    checker = _make_index_checker(
        {
            left: (left_index, TypeQualifier.PARAM),
            right: (right_index, TypeQualifier.PARAM),
        }
    )

    result_type, result_qualifier = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            IdentifierExpression(left),
            IdentifierExpression(right),
        )
    )

    expected = IndexType(
        LiteralExpression(1) + LiteralExpression(1),
        LiteralExpression(10) + LiteralExpression(5),
        None,
    )
    _assert_index_type(result_type, expected)
    assert result_qualifier == TypeQualifier.PARAM


def test_index_plus_index_preserves_matching_stride():
    """Matching literal strides combine via min; min(s, s) = s preserves them."""
    left = Identifier("i")
    right = Identifier("j")
    left_index = IndexType(
        LiteralExpression(1), LiteralExpression(10), LiteralExpression(2)
    )
    right_index = IndexType(
        LiteralExpression(1), LiteralExpression(6), LiteralExpression(2)
    )
    checker = _make_index_checker(
        {
            left: (left_index, TypeQualifier.PARAM),
            right: (right_index, TypeQualifier.PARAM),
        }
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            IdentifierExpression(left),
            IdentifierExpression(right),
        )
    )

    expected = IndexType(
        LiteralExpression(1) + LiteralExpression(1),
        LiteralExpression(10) + LiteralExpression(6),
        LiteralExpression(2),
    )
    _assert_index_type(result_type, expected)


def test_index_plus_index_treats_literal_one_stride_as_none():
    """An explicit stride of LiteralExpression(1) is equivalent to None (unit)."""
    left = Identifier("i")
    right = Identifier("j")
    left_index = IndexType(
        LiteralExpression(1), LiteralExpression(10), LiteralExpression(1)
    )
    right_index = IndexType(LiteralExpression(1), LiteralExpression(6), None)
    checker = _make_index_checker(
        {
            left: (left_index, TypeQualifier.PARAM),
            right: (right_index, TypeQualifier.PARAM),
        }
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            IdentifierExpression(left),
            IdentifierExpression(right),
        )
    )

    expected = IndexType(
        LiteralExpression(1) + LiteralExpression(1),
        LiteralExpression(10) + LiteralExpression(6),
        None,
    )
    _assert_index_type(result_type, expected)


def test_index_plus_index_canonicalizes_matching_unit_literal_stride():
    """Two LiteralExpression(1) strides canonicalize to None in the result."""
    left = Identifier("i")
    right = Identifier("j")
    unit = LiteralExpression(1)
    left_index = IndexType(LiteralExpression(1), LiteralExpression(10), unit)
    right_index = IndexType(LiteralExpression(1), LiteralExpression(6), unit)
    checker = _make_index_checker(
        {
            left: (left_index, TypeQualifier.PARAM),
            right: (right_index, TypeQualifier.PARAM),
        }
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            IdentifierExpression(left),
            IdentifierExpression(right),
        )
    )

    expected = IndexType(
        LiteralExpression(1) + LiteralExpression(1),
        LiteralExpression(10) + LiteralExpression(6),
        None,
    )
    _assert_index_type(result_type, expected)


def test_index_plus_index_unit_stride_dominates_non_unit_stride():
    """gcd(1, s) = 1 — combining unit stride with stride 2 yields unit stride."""
    left = Identifier("i")
    right = Identifier("j")
    left_index = IndexType(LiteralExpression(1), LiteralExpression(10), None)
    right_index = IndexType(
        LiteralExpression(1), LiteralExpression(6), LiteralExpression(2)
    )
    checker = _make_index_checker(
        {
            left: (left_index, TypeQualifier.PARAM),
            right: (right_index, TypeQualifier.PARAM),
        }
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            IdentifierExpression(left),
            IdentifierExpression(right),
        )
    )

    expected = IndexType(
        LiteralExpression(1) + LiteralExpression(1),
        LiteralExpression(10) + LiteralExpression(6),
        None,
    )
    _assert_index_type(result_type, expected)


def test_index_plus_index_uses_min_of_literal_strides():
    """Mismatched literal strides combine via min: min(2, 3) = 2."""
    left = Identifier("i")
    right = Identifier("j")
    left_index = IndexType(
        LiteralExpression(1), LiteralExpression(10), LiteralExpression(2)
    )
    right_index = IndexType(
        LiteralExpression(1), LiteralExpression(9), LiteralExpression(3)
    )
    checker = _make_index_checker(
        {
            left: (left_index, TypeQualifier.PARAM),
            right: (right_index, TypeQualifier.PARAM),
        }
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            IdentifierExpression(left),
            IdentifierExpression(right),
        )
    )

    expected = IndexType(
        LiteralExpression(1) + LiteralExpression(1),
        LiteralExpression(10) + LiteralExpression(9),
        LiteralExpression(2),
    )
    _assert_index_type(result_type, expected)


def test_index_plus_index_rejects_non_literal_stride():
    """Non-literal strides cannot be combined and must fail synthesis."""
    left = Identifier("i")
    right = Identifier("j")
    stride_identifier = Identifier("s")
    left_index = IndexType(
        LiteralExpression(1),
        LiteralExpression(10),
        IdentifierExpression(stride_identifier),
    )
    right_index = IndexType(LiteralExpression(1), LiteralExpression(6), None)
    checker = _make_index_checker(
        {
            left: (left_index, TypeQualifier.PARAM),
            right: (right_index, TypeQualifier.PARAM),
        }
    )

    with pytest.raises(FhYCoreTypeError):
        checker.visit(
            BinaryExpression(
                BinaryOperation.ADD,
                IdentifierExpression(left),
                IdentifierExpression(right),
            )
        )


def test_index_minus_index_is_rejected():
    """Subtraction of two index types is not yet supported (stride semantics TBD)."""
    left = Identifier("i")
    right = Identifier("j")
    index_type = IndexType(LiteralExpression(1), LiteralExpression(10), None)
    checker = _make_index_checker(
        {
            left: (index_type, TypeQualifier.PARAM),
            right: (index_type, TypeQualifier.PARAM),
        }
    )

    with pytest.raises(FhYCoreTypeError):
        checker.visit(
            BinaryExpression(
                BinaryOperation.SUBTRACT,
                IdentifierExpression(left),
                IdentifierExpression(right),
            )
        )


def test_index_plus_index_with_symbolic_bounds():
    """i ∈ [0, N] + j ∈ [0, M] ⟹ [0, N + M] with symbolic upper bounds."""
    left = Identifier("i")
    right = Identifier("j")
    upper_n = Identifier("N")
    upper_m = Identifier("M")
    left_index = IndexType(LiteralExpression(1), IdentifierExpression(upper_n), None)
    right_index = IndexType(LiteralExpression(1), IdentifierExpression(upper_m), None)
    checker = _make_index_checker(
        {
            left: (left_index, TypeQualifier.PARAM),
            right: (right_index, TypeQualifier.PARAM),
        }
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            IdentifierExpression(left),
            IdentifierExpression(right),
        )
    )

    expected = IndexType(
        LiteralExpression(1) + LiteralExpression(1),
        IdentifierExpression(upper_n) + IdentifierExpression(upper_m),
        None,
    )
    _assert_index_type(result_type, expected)


def test_nested_index_plus_index_synthesizes_bottom_up_on_left():
    """(idx_a + idx_b) + idx_c synthesizes leaves first, combining strides bottom-up.

    Left subexpression yields IndexType with stride min(2, 3) = 2; the outer
    combine then takes min(2, 6) = 2 and chains the bounds.
    """
    a = Identifier("a")
    b = Identifier("b")
    c = Identifier("c")
    idx_a = IndexType(LiteralExpression(1), LiteralExpression(10), LiteralExpression(2))
    idx_b = IndexType(LiteralExpression(1), LiteralExpression(10), LiteralExpression(3))
    idx_c = IndexType(LiteralExpression(1), LiteralExpression(10), LiteralExpression(6))
    checker = _make_index_checker(
        {
            a: (idx_a, TypeQualifier.PARAM),
            b: (idx_b, TypeQualifier.PARAM),
            c: (idx_c, TypeQualifier.PARAM),
        }
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            BinaryExpression(
                BinaryOperation.ADD,
                IdentifierExpression(a),
                IdentifierExpression(b),
            ),
            IdentifierExpression(c),
        )
    )

    expected = IndexType(
        (LiteralExpression(1) + LiteralExpression(1)) + LiteralExpression(1),
        (LiteralExpression(10) + LiteralExpression(10)) + LiteralExpression(10),
        LiteralExpression(2),
    )
    _assert_index_type(result_type, expected)


def test_nested_index_plus_index_synthesizes_bottom_up_on_right():
    """idx_a + (idx_b + idx_c): right subtree synthesized first, then combined."""
    a = Identifier("a")
    b = Identifier("b")
    c = Identifier("c")
    idx_a = IndexType(LiteralExpression(1), LiteralExpression(10), LiteralExpression(6))
    idx_b = IndexType(LiteralExpression(1), LiteralExpression(10), LiteralExpression(2))
    idx_c = IndexType(LiteralExpression(1), LiteralExpression(10), LiteralExpression(3))
    checker = _make_index_checker(
        {
            a: (idx_a, TypeQualifier.PARAM),
            b: (idx_b, TypeQualifier.PARAM),
            c: (idx_c, TypeQualifier.PARAM),
        }
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            IdentifierExpression(a),
            BinaryExpression(
                BinaryOperation.ADD,
                IdentifierExpression(b),
                IdentifierExpression(c),
            ),
        )
    )

    expected = IndexType(
        LiteralExpression(1) + (LiteralExpression(1) + LiteralExpression(1)),
        LiteralExpression(10) + (LiteralExpression(10) + LiteralExpression(10)),
        LiteralExpression(2),
    )
    _assert_index_type(result_type, expected)


def test_scaled_shifted_index_plus_index_synthesizes_bottom_up():
    """2 * (i1 - 1) + i2: scale of a shifted index, added to another index.

    Synthesis chain:
      (i1 - 1) → IndexType(0-1, 10-1, None) (stride 1)
      2 * (i1 - 1) → IndexType(2*(0-1), 2*(10-1), LiteralExpression(2))
      ... + i2 → IndexType(..., min(2, 1) = 1 = None)
    """
    i1 = Identifier("i1")
    i2 = Identifier("i2")
    index_type = IndexType(LiteralExpression(1), LiteralExpression(10), None)
    checker = _make_index_checker(
        {
            i1: (index_type, TypeQualifier.PARAM),
            i2: (index_type, TypeQualifier.PARAM),
        }
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            BinaryExpression(
                BinaryOperation.MULTIPLY,
                LiteralExpression(2),
                BinaryExpression(
                    BinaryOperation.SUBTRACT,
                    IdentifierExpression(i1),
                    LiteralExpression(1),
                ),
            ),
            IdentifierExpression(i2),
        )
    )

    expected = IndexType(
        (LiteralExpression(2) * (LiteralExpression(1) - LiteralExpression(1)))
        + LiteralExpression(1),
        (LiteralExpression(2) * (LiteralExpression(10) - LiteralExpression(1)))
        + LiteralExpression(10),
        None,
    )
    _assert_index_type(result_type, expected)


def test_shifted_index_plus_index_propagates_stride_bottom_up():
    """(idx_a + 1) + idx_b: the scalar shift preserves idx_a's stride, which then
    combines with idx_b's stride in the outer node."""
    a = Identifier("a")
    b = Identifier("b")
    idx_a = IndexType(LiteralExpression(1), LiteralExpression(10), LiteralExpression(2))
    idx_b = IndexType(LiteralExpression(1), LiteralExpression(10), LiteralExpression(3))
    checker = _make_index_checker(
        {
            a: (idx_a, TypeQualifier.PARAM),
            b: (idx_b, TypeQualifier.PARAM),
        }
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            BinaryExpression(
                BinaryOperation.ADD,
                IdentifierExpression(a),
                LiteralExpression(1),
            ),
            IdentifierExpression(b),
        )
    )

    expected = IndexType(
        (LiteralExpression(1) + LiteralExpression(1)) + LiteralExpression(1),
        (LiteralExpression(10) + LiteralExpression(1)) + LiteralExpression(10),
        LiteralExpression(2),
    )
    _assert_index_type(result_type, expected)


@pytest.mark.parametrize(
    "operation",
    [
        BinaryOperation.MULTIPLY,
        BinaryOperation.DIVIDE,
        BinaryOperation.FLOOR_DIVIDE,
        BinaryOperation.MODULO,
        BinaryOperation.POWER,
    ],
)
def test_non_additive_index_index_operations_are_rejected(operation):
    """Only ADD and SUBTRACT are supported between two index operands."""
    left = Identifier("i")
    right = Identifier("j")
    index_type = IndexType(LiteralExpression(1), LiteralExpression(8), None)
    checker = _make_index_checker(
        {
            left: (index_type, TypeQualifier.PARAM),
            right: (index_type, TypeQualifier.PARAM),
        }
    )

    with pytest.raises(FhYCoreTypeError):
        checker.visit(
            BinaryExpression(
                operation,
                IdentifierExpression(left),
                IdentifierExpression(right),
            )
        )
