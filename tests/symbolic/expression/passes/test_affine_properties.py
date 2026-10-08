"""Hypothesis property tests for `affine_form`.

Trees come from `build_affine_expression_strategy`, affine in a pool of
three identifiers by construction (sums, negations, literal multiples,
divisions by a non-zero literal), so `affine_form` must read every one of
them. The oracles: SymPy's `expand` and `coeff` on the lowered tree (the
independent computation of each coefficient and of the constant), the
difference of two forms against the form of the difference, and the inverse
`to_expression` then `affine_form`.
"""

from fractions import Fraction
from typing import Final

import pytest

pytest.importorskip("hypothesis")
pytest.importorskip("sympy")

import sympy  # type: ignore[import-untyped]
from hypothesis import example, given

from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression import (
    BinaryOperation,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    affine_form,
    convert_expression_to_sympy_expression,
    make_binary_expression,
)

from ....strategies.expressions import build_affine_expression_strategy
from ....strategies.identifiers import build_identifier_pool
from ....strategies.settings import cap_max_examples

pytestmark = [pytest.mark.property, pytest.mark.sympy]

_POOL: Final[tuple[Identifier, ...]] = build_identifier_pool(3)
_TREES: Final = build_affine_expression_strategy(_POOL)


def _to_fraction(number: sympy.Expr) -> Fraction:
    """Return the exact rational SymPy `number` as a `Fraction`."""
    assert number.is_Rational
    return Fraction(int(number.p), int(number.q))


@cap_max_examples(200)
@example(
    expression=IdentifierExpression(_POOL[0]) / LiteralExpression(2)
    + IdentifierExpression(_POOL[0]) / LiteralExpression(2)
)
@given(_TREES)
def test_coefficients_and_constant_match_sympys_expansion(
    expression: Expression,
) -> None:
    """Test every coefficient and the constant equal SymPy's `expand` and `coeff`.

    Oracle: `sympy.expand` of the lowered tree, read with `coeff` per
    symbol and the value at all symbols zero for the constant.
    """
    form = affine_form(expression)
    assert form is not None
    lowered = sympy.expand(convert_expression_to_sympy_expression(expression))
    symbols = {
        f"{identifier.name_hint}_{identifier.id}": identifier for identifier in _POOL
    }

    for symbol_name, identifier in symbols.items():
        expected = _to_fraction(lowered.coeff(sympy.Symbol(symbol_name)))
        assert form.coefficient(identifier) == expected
    constant_term = lowered.subs({sympy.Symbol(name): 0 for name in symbols})
    assert form.constant == _to_fraction(sympy.nsimplify(constant_term))
    assert set(form.terms) == {
        identifier
        for name, identifier in symbols.items()
        if lowered.coeff(sympy.Symbol(name)) != 0
    }


@cap_max_examples(200)
@given(_TREES, _TREES)
def test_the_form_of_a_difference_is_the_difference_of_the_forms(
    left: Expression, right: Expression
) -> None:
    """Test `affine_form(a - b)` has the coefficients and constant `a - b`.

    Oracle: exact `Fraction` subtraction of the two operands' own forms.
    """
    left_form, right_form = affine_form(left), affine_form(right)
    difference = affine_form(
        make_binary_expression(BinaryOperation.SUBTRACT, left, right)
    )
    assert left_form is not None
    assert right_form is not None
    assert difference is not None

    for identifier in _POOL:
        assert difference.coefficient(identifier) == left_form.coefficient(
            identifier
        ) - right_form.coefficient(identifier)
    assert difference.constant == left_form.constant - right_form.constant


@cap_max_examples(200)
@given(_TREES)
def test_the_canonical_expression_reads_back_as_the_same_form(
    expression: Expression,
) -> None:
    """Test `affine_form(form.to_expression())` is `form`.

    Oracle: the inverse operation, `to_expression`.
    """
    form = affine_form(expression)
    assert form is not None

    assert affine_form(form.to_expression()) == form


@cap_max_examples(200)
@given(_TREES)
def test_a_tree_minus_itself_is_the_zero_form(expression: Expression) -> None:
    """Test `e - e` has no term and the constant 0, whatever `e` holds.

    Oracle: the algebraic law `e - e = 0`.
    """
    zero = affine_form(
        make_binary_expression(BinaryOperation.SUBTRACT, expression, expression)
    )

    assert zero is not None
    assert zero.is_constant()
    assert zero.terms == {}
    assert zero.constant == 0


@cap_max_examples(200)
@given(_TREES)
def test_the_terms_hold_no_zero_and_run_by_identifier_id(
    expression: Expression,
) -> None:
    """Test `terms` lists non-zero coefficients in increasing identifier id.

    Oracle: sorting the identifiers by id and comparing with each
    `coefficient` query.
    """
    form = affine_form(expression)
    assert form is not None

    identifiers = list(form.terms)
    assert [identifier.id for identifier in identifiers] == sorted(
        identifier.id for identifier in identifiers
    )
    assert all(coefficient != 0 for coefficient in form.terms.values())
    assert all(form.terms[i] == form.coefficient(i) for i in identifiers)
    assert form.is_constant() == (not identifiers)
