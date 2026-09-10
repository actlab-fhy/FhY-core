"""Cross-cutting tests that span multiple modules in `fhy_core.symbolic.expression`."""

import pytest
import sympy  # type: ignore[import-untyped]
import z3  # type: ignore[import-untyped]

from fhy_core.pass_infrastructure import CompilerPass, PassInfo
from fhy_core.symbolic.expression import (
    LiteralExpression,
    convert_expression_to_sympy_expression,
    convert_expression_to_z3_expression,
)
from fhy_core.symbolic.expression.passes.sympy import (
    ExpressionToSympyConverter,
    SymPyToExpressionConverter,
)
from fhy_core.symbolic.expression.passes.z3 import ExpressionToZ3Converter

# =============================================================================
# Pass registry - every expression pass must self-register
# =============================================================================

_EXPECTED_REGISTRATIONS: list[tuple[str, type]] = [
    ("fhy_core.symbolic.expression.from_sympy", SymPyToExpressionConverter),
    ("fhy_core.symbolic.expression.to_sympy", ExpressionToSympyConverter),
    ("fhy_core.symbolic.expression.to_z3", ExpressionToZ3Converter),
]


@pytest.mark.parametrize("pass_name, pass_class", _EXPECTED_REGISTRATIONS)
def test_expression_pass_is_registered_under_expected_name(
    pass_name: str, pass_class: type
) -> None:
    """Test each expression pass registers under its expected name and class."""
    registered = CompilerPass.get_registered_passes()
    assert pass_name in registered
    info = registered[pass_name]
    assert isinstance(info, PassInfo)
    assert info.pass_type is pass_class
    assert info.description.strip() != ""


# =============================================================================
# Literal lowering is a congruence on structural-equivalence classes
# =============================================================================


@pytest.mark.z3
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(1, id="int"),
        pytest.param("1", id="integer_grammar_string"),
        pytest.param("01", id="integer_grammar_string_leading_zero"),
    ],
)
def test_integer_valued_literal_lowers_to_an_integer_in_both_bridges(
    value: int | str,
) -> None:
    """Test every spelling of an integer literal reaches both backends as an integer.

    ``LiteralExpression`` compares literals by bucket, so ``1``, ``"1"``,
    and ``"01"`` form one structural-equivalence class. Lowering has to be
    a congruence on that class: a bridge sending the string forms to a
    different sort than the ``int`` form would let a solver decide for one
    member of the class what it refuses, or answers differently, for
    another.
    """
    literal = LiteralExpression(value)
    assert literal.is_structurally_equivalent(LiteralExpression(1))

    lowered_z3, _ = convert_expression_to_z3_expression(literal, {})
    lowered_sympy = convert_expression_to_sympy_expression(literal)

    assert lowered_z3.sort() == z3.IntSort()
    assert lowered_sympy == sympy.Integer(1)
    assert lowered_sympy.is_Integer


@pytest.mark.z3
@pytest.mark.parametrize(
    "value, expected_sympy_value",
    [
        pytest.param(1.5, sympy.Float(1.5), id="float_binary"),
        pytest.param("1.5", sympy.Rational(3, 2), id="float_grammar_string"),
        pytest.param("1.0", sympy.Integer(1), id="float_grammar_string_integral_value"),
    ],
)
def test_float_valued_literal_lowers_to_a_real_in_both_bridges(
    value: float | str, expected_sympy_value: sympy.Expr
) -> None:
    """Test a float-valued literal stays real and exact in both bridges.

    The float buckets are unaffected by the integer lowering: a
    float-grammar string is not integer-valued even when its value has no
    fractional part, so ``"1.0"`` reaches Z3 as a ``Real`` rather than
    joining the class of the integer ``1``. SymPy has no sorts to hold
    apart, so what it has to hold instead is the value, which pins the
    exact number each form lowers to -- ``sympy.Rational`` for the
    exact-decimal string form, whose unit-denominator case normalizes to
    ``sympy.Integer``.
    """
    literal = LiteralExpression(value)
    assert not literal.is_structurally_equivalent(LiteralExpression(1))

    lowered_z3, _ = convert_expression_to_z3_expression(literal, {})
    lowered_sympy = convert_expression_to_sympy_expression(literal)

    assert lowered_z3.sort() == z3.RealSort()
    assert lowered_sympy.is_real
    assert lowered_sympy == expected_sympy_value


# =============================================================================
# Both bridges honour each literal form's precision contract
# =============================================================================


@pytest.mark.z3
def test_binary_float_literal_lowers_to_its_ieee_value_in_both_bridges() -> None:
    """Test a Python ``float`` reaches both backends as the value its bits denote.

    ``LiteralExpression(0.1)`` stores an IEEE-754 binary value that is
    strictly greater than one tenth. Reinterpreting its shortest repr as
    exact decimal -- which is what handing the ``float`` to
    ``z3.RealVal`` does -- would send the solver a different number than
    the one simplification works on.
    """
    literal = LiteralExpression(0.1)
    ieee_numerator, ieee_denominator = (0.1).as_integer_ratio()

    lowered_z3, _ = convert_expression_to_z3_expression(literal, {})
    lowered_sympy = convert_expression_to_sympy_expression(literal)

    assert lowered_z3.eq(z3.RatVal(ieee_numerator, ieee_denominator))
    assert not lowered_z3.eq(z3.RealVal("1/10"))
    assert lowered_sympy == sympy.Float(0.1)
    assert sympy.Rational(lowered_sympy) == sympy.Rational(
        ieee_numerator, ieee_denominator
    )


@pytest.mark.z3
def test_decimal_string_literal_lowers_to_its_exact_decimal_in_both_bridges() -> None:
    """Test a float-grammar string reaches both backends as its exact decimal.

    ``LiteralExpression("0.1")`` stores exact decimal text, so both
    bridges have to produce exactly one tenth. Rounding the text to
    binary -- which ``sympy.Float`` does -- would leave simplification
    working on a number the solver never sees.
    """
    literal = LiteralExpression("0.1")

    lowered_z3, _ = convert_expression_to_z3_expression(literal, {})
    lowered_sympy = convert_expression_to_sympy_expression(literal)

    assert lowered_z3.eq(z3.RatVal(1, 10))
    assert lowered_sympy == sympy.Rational(1, 10)
    assert lowered_sympy != sympy.Float(0.1)


@pytest.mark.z3
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(0.1, id="float_binary"),
        pytest.param("0.1", id="float_grammar_string"),
        pytest.param(0.3, id="float_binary_three_tenths"),
        pytest.param("0.3", id="float_grammar_string_three_tenths"),
        pytest.param(1.5, id="float_binary_exactly_representable"),
        pytest.param("1.5", id="float_grammar_string_exactly_representable"),
    ],
)
def test_float_literal_lowers_to_the_same_rational_in_both_bridges(
    value: float | str,
) -> None:
    """Test both bridges lower one literal form to one rational.

    This is the property that keeps ``simplify_expression`` and the solver
    seam from deciding the same ground comparison differently: whichever
    precision contract a literal form carries, both backends read it the
    same way, so neither can be working on a number the other never saw.
    """
    literal = LiteralExpression(value)

    lowered_z3, _ = convert_expression_to_z3_expression(literal, {})
    lowered_sympy = convert_expression_to_sympy_expression(literal)

    sympy_rational = sympy.Rational(lowered_sympy)
    assert lowered_z3.eq(z3.RatVal(int(sympy_rational.p), int(sympy_rational.q)))
