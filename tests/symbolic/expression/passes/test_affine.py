"""Tests for the exact affine form of an expression.

``affine_form`` reads an expression as exact rational multiples of its free
identifiers plus an exact rational constant, or declines with ``None``. The
tests pin the behavior table of the design (cancellation, rational
coefficients, constant folding, every declined shape), the accessors of
``AffineForm`` and its canonical expression, and the argument check.
"""

from decimal import Decimal
from fractions import Fraction
from typing import Any

import pytest

import fhy_core.symbolic.expression as expression_package
from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression import (
    AffineForm,
    BinaryOperation,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    affine_form,
    call,
    make_binary_expression,
)
from fhy_core.symbolic.expression.passes import affine as affine_module

_X = Identifier("x")
_Y = Identifier("y")
_S = Identifier("s")
_T = Identifier("t")
_I = Identifier("i")
_J = Identifier("j")


def _id(identifier: Identifier) -> IdentifierExpression:
    """Return the identifier expression of `identifier`."""
    return IdentifierExpression(identifier)


def _lit(value: Any) -> LiteralExpression:
    """Return the literal expression of `value`."""
    return LiteralExpression(value)


def _form(expression: Expression) -> AffineForm:
    """Return the affine form of `expression`, which must be affine."""
    form = affine_form(expression)
    assert form is not None
    return form


# ===========================================================================
# The behavior table
# ===========================================================================


def test_a_cancelled_term_leaves_the_other_term() -> None:
    """Test `(3 * s + t) - t` is `3 * s`, whatever `t` is."""
    form = _form((3 * _id(_S) + _id(_T)) - _id(_T))

    assert form.coefficient(_S) == Fraction(3)
    assert form.coefficient(_T) == 0
    assert form.constant == 0
    assert form.terms == {_S: Fraction(3)}
    assert not form.is_constant()


def test_two_halves_make_a_whole() -> None:
    """Test `x / 2 + x / 2` has the integer coefficient 1."""
    x = _id(_X)

    form = _form(x / 2 + x / 2)

    assert form.coefficient(_X) == Fraction(1)
    assert form.terms == {_X: Fraction(1)}
    assert form.constant == 0


def test_a_fractional_coefficient_is_an_exact_fraction() -> None:
    """Test `x / 3` has the coefficient 1/3, not a float."""
    form = _form(_id(_X) / 3)

    coefficient = form.coefficient(_X)
    assert isinstance(coefficient, Fraction)
    assert coefficient == Fraction(1, 3)


def test_an_offset_scales_through_the_sum() -> None:
    """Test `2 * (i + 1) - 1` is `2 * i + 1`."""
    form = _form(2 * (_id(_I) + 1) - 1)

    assert form.coefficient(_I) == Fraction(2)
    assert form.constant == Fraction(1)
    assert form.terms == {_I: Fraction(2)}


def test_a_floor_division_of_constants_is_folded() -> None:
    """Test `7 // 2 + i` is `i + 3`."""
    form = _form(_lit(7) // _lit(2) + _id(_I))

    assert form.coefficient(_I) == Fraction(1)
    assert form.constant == Fraction(3)


@pytest.mark.parametrize(
    ("build", "constant"),
    [
        (lambda: _lit(7) % _lit(3), 1),
        (lambda: _lit(2) ** _lit(10), 1024),
        (lambda: _lit(-7) // _lit(2), -4),
        (lambda: _lit(1) / _lit(4), Fraction(1, 4)),
    ],
    ids=["modulo", "power", "floor_division_of_a_negative", "true_division"],
)
def test_constant_operations_are_folded_exactly(
    build: Any, constant: int | Fraction
) -> None:
    """Test `%`, `**`, `//` and `/` of constants fold to an exact constant."""
    form = _form(build())

    assert form.is_constant()
    assert form.constant == Fraction(constant)
    assert form.terms == {}


def test_a_decimal_literal_is_an_exact_fraction() -> None:
    """Test `x + 0.5` written as a decimal literal has the constant 1/2."""
    form = _form(_id(_X) + _lit(Decimal("0.5")))

    assert form.coefficient(_X) == Fraction(1)
    assert form.constant == Fraction(1, 2)


def test_a_literal_is_a_constant_form() -> None:
    """Test `5` is the constant 5 with no term."""
    form = _form(_lit(5))

    assert form.is_constant()
    assert form.constant == Fraction(5)
    assert form.terms == {}


def test_an_identifier_is_its_own_term() -> None:
    """Test `x` has the coefficient 1 and the constant 0."""
    form = _form(_id(_X))

    assert form.coefficient(_X) == Fraction(1)
    assert form.constant == 0
    assert not form.is_constant()


def test_a_difference_of_equal_terms_is_the_zero_constant() -> None:
    """Test `x - x` is the constant 0 with no term."""
    form = _form(_id(_X) - _id(_X))

    assert form.is_constant()
    assert form.constant == 0
    assert form.terms == {}
    assert form.coefficient(_X) == 0


def test_negation_and_a_negative_scale_flip_the_signs() -> None:
    """Test `-(x - 2 * y)` is `-x + 2 * y`."""
    form = _form(-(_id(_X) - 2 * _id(_Y)))

    assert form.coefficient(_X) == Fraction(-1)
    assert form.coefficient(_Y) == Fraction(2)


def test_a_product_with_a_constant_form_on_the_right_scales() -> None:
    """Test `(x + 1) * (2 + 1)` scales by the constant form 3."""
    form = _form((_id(_X) + 1) * (_lit(2) + _lit(1)))

    assert form.coefficient(_X) == Fraction(3)
    assert form.constant == Fraction(3)


def test_the_coefficient_of_an_absent_identifier_is_zero() -> None:
    """Test an identifier the expression does not hold has the coefficient 0."""
    form = _form(_id(_X) + 1)

    coefficient = form.coefficient(_Y)
    assert isinstance(coefficient, Fraction)
    assert coefficient == 0


# ===========================================================================
# Declined shapes
# ===========================================================================


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: _id(_I) * _id(_J), id="product_of_two_forms"),
        pytest.param(lambda: _id(_I) // 2, id="floor_division_of_a_form"),
        pytest.param(lambda: _id(_I) % 2, id="modulo_of_a_form"),
        pytest.param(lambda: _id(_I) ** 2, id="power_of_a_form"),
        pytest.param(lambda: _lit(2) ** _id(_I), id="power_with_a_form_exponent"),
        pytest.param(lambda: _id(_X) + _lit(0.5), id="float_literal"),
        pytest.param(lambda: _lit(True), id="boolean_literal"),
        pytest.param(lambda: _id(_X) / 0, id="division_by_zero"),
        pytest.param(lambda: _id(_X) / _id(_Y), id="division_by_a_form"),
        pytest.param(lambda: _lit(1) / _id(_Y), id="constant_over_a_form"),
        pytest.param(lambda: _lit(1) // _lit(0), id="floor_division_by_zero"),
        pytest.param(lambda: _lit(1) % _lit(0), id="modulo_by_zero"),
        pytest.param(
            lambda: make_binary_expression(BinaryOperation.LESS, _id(_X), _lit(1)),
            id="comparison",
        ),
        pytest.param(lambda: call("floor", _id(_X)), id="call"),
        pytest.param(lambda: _id(_X) + call("floor", _id(_Y)), id="call_inside_a_sum"),
    ],
)
def test_a_non_affine_expression_declines(build: Any) -> None:
    """Test an expression outside the affine grammar answers `None`."""
    assert affine_form(build()) is None


def test_a_tree_nested_beyond_the_bound_declines() -> None:
    """Test a tree nested more than 256 levels declines, one within it reads."""
    shallow: Expression = _id(_X)
    for _ in range(100):
        shallow = shallow + 1
    deep: Expression = _id(_X)
    for _ in range(300):
        deep = deep + 1

    within = affine_form(shallow)
    assert within is not None
    assert within.constant == Fraction(100)
    assert affine_form(deep) is None


def test_a_constant_beyond_4096_bits_declines() -> None:
    """Test a constant whose numerator would exceed 4096 bits declines."""
    fits = _id(_X) + _lit(2**4000)
    too_big = _id(_X) + _lit(2**4100)

    form = affine_form(fits)
    assert form is not None
    assert form.constant == Fraction(2**4000)
    assert affine_form(too_big) is None


# ===========================================================================
# AffineForm
# ===========================================================================


def test_terms_are_ordered_by_identifier_id() -> None:
    """Test `terms` lists identifiers by id, whatever order the sum names them."""
    first, second, third = (
        Identifier("first"),
        Identifier("second"),
        Identifier("third"),
    )

    form = _form(_id(third) + 2 * _id(first) + 3 * _id(second))

    assert list(form.terms) == [first, second, third]
    assert list(form.terms.values()) == [Fraction(2), Fraction(3), Fraction(1)]
    assert all(isinstance(value, Fraction) for value in form.terms.values())


def test_terms_holds_no_zero_coefficient() -> None:
    """Test a cancelled identifier is no key of `terms`."""
    form = _form(_id(_X) + _id(_Y) - _id(_X))

    assert list(form.terms) == [_Y]


def test_terms_is_a_copy() -> None:
    """Test changing the `terms` dict leaves the form as it was."""
    form = _form(_id(_X) + 1)

    form.terms[_X] = Fraction(9)
    form.terms.clear()

    assert form.terms == {_X: Fraction(1)}
    assert form.coefficient(_X) == Fraction(1)


def test_the_constant_is_a_fraction() -> None:
    """Test `constant` is a `Fraction` even for an integer."""
    constant = _form(_id(_X) + 4).constant

    assert isinstance(constant, Fraction)
    assert constant == Fraction(4)


def test_forms_of_commuted_sums_are_equal_and_hash_alike() -> None:
    """Test `x + y` and `y + x` give equal forms with one hash."""
    left = _form(_id(_X) + _id(_Y))
    right = _form(_id(_Y) + _id(_X))

    assert left is not right
    assert left == right
    assert hash(left) == hash(right)
    assert len({left, right}) == 1


def test_forms_differ_in_a_coefficient_a_constant_or_an_identifier() -> None:
    """Test forms are unequal when any part differs."""
    base = _form(_id(_X) + _id(_Y))

    assert base != _form(_id(_X) + 2 * _id(_Y))
    assert base != _form(_id(_X) + _id(_Y) + 1)
    assert base != _form(_id(_X) + _id(_J))
    assert base != "x + y"


def test_a_form_equals_the_form_of_an_equal_affine_rewrite() -> None:
    """Test `(3 * s + t) - t` and `s + 2 * s` are the same form."""
    assert _form((3 * _id(_S) + _id(_T)) - _id(_T)) == _form(_id(_S) + 2 * _id(_S))


def test_the_form_is_frozen_and_has_no_constructor() -> None:
    """Test `AffineForm()` is refused and an attribute cannot be set."""
    form = _form(_id(_X))

    with pytest.raises(TypeError):
        AffineForm()  # type: ignore[call-arg]  # test: no constructor
    with pytest.raises((AttributeError, TypeError)):
        form.extra = 1  # type: ignore[attr-defined]  # test: frozen


# ===========================================================================
# to_expression
# ===========================================================================


def test_to_expression_builds_the_canonical_sum() -> None:
    """Test `2 * x + 1` comes back as the expression `2 * x + 1`."""
    form = _form(_id(_X) + 1 + _id(_X))

    expression = form.to_expression()

    assert isinstance(expression, Expression)
    assert expression == 2 * _id(_X) + 1


def test_to_expression_writes_unit_coefficients_bare() -> None:
    """Test a coefficient of 1 is `x` and of -1 is `-x`, with no constant when 0."""
    assert _form(_id(_X)).to_expression() == _id(_X)
    assert _form(-_id(_X)).to_expression() == -_id(_X)


def test_to_expression_orders_the_terms_by_identifier_id() -> None:
    """Test a sum naming the later identifier first comes back earliest id first."""
    first, second = Identifier("first"), Identifier("second")

    expression = _form(_id(second) + _id(first)).to_expression()

    assert expression == _id(first) + _id(second)


def test_to_expression_of_a_zero_form_is_the_literal_zero() -> None:
    """Test `x - x` rebuilds as `0`."""
    assert _form(_id(_X) - _id(_X)).to_expression() == _lit(0)


def test_to_expression_of_a_constant_is_the_literal() -> None:
    """Test `5` rebuilds as `5`."""
    assert _form(_lit(2) + _lit(3)).to_expression() == _lit(5)


def test_to_expression_writes_a_fraction_as_a_quotient_of_integers() -> None:
    """Test the coefficient 1/2 is the quotient `1 / 2` times the identifier."""
    expression = _form(_id(_X) / 2).to_expression()

    assert affine_form(expression) == _form(_id(_X) / 2)
    assert expression.is_structurally_equivalent(
        make_binary_expression(
            BinaryOperation.MULTIPLY,
            make_binary_expression(BinaryOperation.DIVIDE, _lit(1), _lit(2)),
            _id(_X),
        )
    )


def test_to_expression_reads_back_as_the_same_form() -> None:
    """Test the canonical expression of a form has that form."""
    form = _form(3 * (_id(_X) - 2 * _id(_Y)) + _id(_Y) / 4 - 7)

    assert affine_form(form.to_expression()) == form


def test_str_is_the_canonical_expression_text() -> None:
    """Test `str` of a form is the text of its canonical expression."""
    form = _form(_id(_X) + 1 + _id(_X))

    assert str(form) == str(form.to_expression())


# ===========================================================================
# Arguments and exports
# ===========================================================================


@pytest.mark.parametrize(
    "argument",
    [3, "x", None, _X, Fraction(1, 2), (_id(_X),)],
    ids=["int", "str", "none", "identifier", "fraction", "tuple"],
)
def test_a_non_expression_argument_is_a_type_error(argument: Any) -> None:
    """Test an argument that is no `Expression` raises `TypeError`."""
    with pytest.raises(TypeError):
        affine_form(argument)


def test_the_names_are_exported_from_the_expression_package() -> None:
    """Test `AffineForm` and `affine_form` are the pass module's objects."""
    assert expression_package.affine_form is affine_module.affine_form
    assert expression_package.AffineForm is affine_module.AffineForm
    assert {"AffineForm", "affine_form"} <= set(expression_package.__all__)


def test_affine_form_is_a_function_not_a_method_of_expression() -> None:
    """Test `Expression` gains no `affine_form` method."""
    assert not hasattr(Expression, "affine_form")
