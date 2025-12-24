"""Bound numerical parameters."""

__all__ = ["BoundIntParam", "BoundNatParam"]

from typing import Any, Iterable, Optional, Tuple

from fhy_core.constraint import Constraint, EquationConstraint
from fhy_core.expression import (
    BinaryExpression,
    BinaryOperation,
    Expression,
    IdentifierExpression,
    LiteralExpression,
)
from fhy_core.identifier import Identifier
from fhy_core.utils import Self

from .core import IntParam, Param
from .fundamental import NatParam


def _is_valid_bound_expression(expression: Expression) -> bool:
    if not isinstance(expression, BinaryExpression):
        return False
    if expression.operation not in (
        BinaryOperation.GREATER_EQUAL,
        BinaryOperation.GREATER,
        BinaryOperation.LESS_EQUAL,
        BinaryOperation.LESS,
    ):
        return False
    if not (
        (
            isinstance(expression.left, IdentifierExpression)
            or isinstance(expression.right, IdentifierExpression)
        )
        and (
            isinstance(expression.left, LiteralExpression)
            or isinstance(expression.right, LiteralExpression)
        )
    ):
        return False
    literal_expression: LiteralExpression
    if isinstance(expression.left, LiteralExpression):
        literal_expression = expression.left
    elif isinstance(expression.right, LiteralExpression):
        literal_expression = expression.right
    else:
        raise RuntimeError(
            "Somehow failed to find LiteralExpression in bound expression."
        )
    return isinstance(literal_expression.value, int)


def _invert_binary_comparison_operation(op: BinaryOperation) -> BinaryOperation:
    if op == BinaryOperation.GREATER:
        return BinaryOperation.LESS
    elif op == BinaryOperation.GREATER_EQUAL:
        return BinaryOperation.LESS_EQUAL
    elif op == BinaryOperation.LESS:
        return BinaryOperation.GREATER
    elif op == BinaryOperation.LESS_EQUAL:
        return BinaryOperation.GREATER_EQUAL
    else:
        raise ValueError("Cannot invert non-comparison binary operation: " f"{op}")


def _get_bound_from_expression(
    literal_expression: LiteralExpression, op: BinaryOperation
) -> Tuple[bool, int, bool]:
    value = literal_expression.value
    if not isinstance(value, int):
        raise RuntimeError(
            "Somehow failed to find integer LiteralExpression in bound expression."
        )
    return (
        True
        if op in (BinaryOperation.GREATER, BinaryOperation.GREATER_EQUAL)
        else False,
        value,
        True
        if op in (BinaryOperation.GREATER_EQUAL, BinaryOperation.LESS_EQUAL)
        else False,
    )


class BoundIntParam(IntParam):
    """An interval integer parameter that supports basic arithmetic.

    Results can be represented as inclusive or exclusive via prefer_inclusive.
    """

    _prefer_inclusive: bool

    def __init__(
        self, name: Identifier | None = None, prefer_inclusive: bool = True
    ) -> None:
        super().__init__(name)
        self._prefer_inclusive = prefer_inclusive

    def add_constraint(self, constraint: Constraint) -> None:
        if not isinstance(constraint, EquationConstraint):
            raise TypeError(
                "BoundIntParam only supports EquationConstraint constraints."
            )
        if not _is_valid_bound_expression(constraint.convert_to_expression()):
            raise ValueError(
                "BoundIntParam only supports bound expressions of the form "
                '"x >= k", "x > k", "x <= k", or "x < k" where k is an integer.'
            )
        return super().add_constraint(constraint)

    def _iter_bounds(self) -> Iterable[Tuple[bool, int, bool]]:
        """Yield (is_lower, bound_value, inclusive) for constraints.

        Fail if any constraint is not a valid bound expression on the same
        variable.
        """
        for constraint in self._constraints:
            if not isinstance(constraint, EquationConstraint):
                raise RuntimeError(
                    "BoundIntParam somehow has non-EquationConstraint constraint: "
                    f"{type(constraint)}"
                )
            if not _is_valid_bound_expression(constraint.convert_to_expression()):
                raise RuntimeError(
                    "BoundIntParam somehow has non-bound expression constraint: "
                    f"{repr(constraint)}"
                )

            expression = constraint.convert_to_expression()
            identifier_expression: IdentifierExpression | None = None

            def set_or_check_identifier_expr(e: IdentifierExpression) -> None:
                nonlocal identifier_expression
                if identifier_expression is None:
                    identifier_expression = e
                elif identifier_expression != e:
                    raise RuntimeError(
                        "BoundIntParam somehow has bound constraints on "
                        "different variables."
                    )

            if not isinstance(expression, BinaryExpression):
                raise RuntimeError(
                    "Somehow bound constraint expression is not a BinaryExpression."
                )
            if isinstance(expression.left, IdentifierExpression):
                if not isinstance(expression.right, LiteralExpression):
                    raise RuntimeError(
                        "Somehow bound expression is not in the expected form."
                    )
                set_or_check_identifier_expr(expression.left)
                yield _get_bound_from_expression(expression.right, expression.operation)
            elif isinstance(expression.right, IdentifierExpression):
                if not isinstance(expression.left, LiteralExpression):
                    raise RuntimeError(
                        "Somehow bound expression is not in the expected form."
                    )
                set_or_check_identifier_expr(expression.right)
                yield _get_bound_from_expression(
                    expression.left,
                    _invert_binary_comparison_operation(expression.operation),
                )
            else:
                raise RuntimeError(
                    "Somehow bound expression is not in the expected form."
                )

    def _get_effective_min_max(self) -> Tuple[int | None, int | None]:
        """Return (min_int, max_int) represented by constraints.

        Semantics:
          x > k  => min_int >= k+1
          x >= k => min_int >= k
          x < k  => max_int <= k-1
          x <= k => max_int <= k
        None means unbounded.
        """
        min_int: int | None = None
        max_int: int | None = None

        for is_lower, bound, inclusive in self._iter_bounds():
            if is_lower:
                eff = bound if inclusive else bound + 1
                min_int = eff if min_int is None else max(min_int, eff)
            else:
                eff = bound if inclusive else bound - 1
                max_int = eff if max_int is None else min(max_int, eff)

        if min_int is not None and max_int is not None and min_int > max_int:
            raise ValueError(
                "Empty integer interval represented by constraints for "
                f"{self.variable}."
            )

        return min_int, max_int

    @staticmethod
    def _create_param_from_min_max(
        param: "BoundIntParam",
        min_int: Optional[int],
        max_int: Optional[int],
    ) -> "BoundIntParam":
        """Create a new BoundIntParam from min/max integers.

        Emits either inclusive or exclusive style depending on param.prefer_inclusive.
        """
        out = BoundIntParam(
            name=param.variable, prefer_inclusive=param._prefer_inclusive
        )
        if min_int is not None:
            if out._prefer_inclusive:
                out.add_lower_bound_constraint(min_int, True)
            else:
                out.add_lower_bound_constraint(min_int - 1, False)

        if max_int is not None:
            if out._prefer_inclusive:
                out.add_upper_bound_constraint(max_int, True)
            else:
                out.add_upper_bound_constraint(max_int + 1, False)
        return out

    @classmethod
    def between(
        cls: type[Self],
        lower_bound: int,
        upper_bound: int,
        name: Identifier | None = None,
        is_lower_inclusive: bool = True,
        is_upper_inclusive: bool = True,
        prefer_inclusive: bool = True,
    ) -> "BoundIntParam":
        p = cls(name, prefer_inclusive=prefer_inclusive)
        p.add_lower_bound_constraint(lower_bound, is_lower_inclusive)
        p.add_upper_bound_constraint(upper_bound, is_upper_inclusive)
        p._get_effective_min_max()
        return p

    @classmethod
    def with_lower_bound(
        cls: type[Self],
        lower_bound: int,
        name: Identifier | None = None,
        is_inclusive: bool = True,
        prefer_inclusive: bool = True,
    ) -> "BoundIntParam":
        p = cls(name, prefer_inclusive=prefer_inclusive)
        p.add_lower_bound_constraint(lower_bound, is_inclusive)
        p._get_effective_min_max()
        return p

    @classmethod
    def with_upper_bound(
        cls: type[Self],
        upper_bound: int,
        name: Identifier | None = None,
        is_inclusive: bool = True,
        prefer_inclusive: bool = True,
    ) -> "BoundIntParam":
        p = cls(name, prefer_inclusive=prefer_inclusive)
        p.add_upper_bound_constraint(upper_bound, is_inclusive)
        p._get_effective_min_max()
        return p

    @classmethod
    def exactly(
        cls: type[Self],
        value: int,
        name: Identifier | None = None,
        prefer_inclusive: bool = True,
    ) -> "BoundIntParam":
        p = cls(name, prefer_inclusive=prefer_inclusive)
        p.add_lower_bound_constraint(value, is_inclusive=True)
        p.add_upper_bound_constraint(value, is_inclusive=True)
        return p

    def _coerce_other(self, other: Any) -> "BoundIntParam":
        if isinstance(other, int):
            return BoundIntParam.exactly(other, prefer_inclusive=self._prefer_inclusive)
        elif isinstance(other, BoundIntParam):
            return other
        elif isinstance(other, IntParam):
            for other_constraint in other._constraints:
                if not _is_valid_bound_expression(
                    other_constraint.convert_to_expression()
                ):
                    raise TypeError(
                        "Cannot coerce IntParam with non-bound constraints "
                        "to BoundIntParam."
                    )
            wrapper_param = BoundIntParam(
                other.variable, prefer_inclusive=self._prefer_inclusive
            )
            wrapper_param._value = other._value
            Param.copy_constraints_to_new_param(other, wrapper_param)
            return wrapper_param
        raise TypeError(f"Unsupported operand type: {type(other)}")

    def __add__(self, other: Any) -> "BoundIntParam":
        coerced_other = self._coerce_other(other)
        if self.is_value_set() and coerced_other.is_value_set():
            return BoundIntParam.with_value(
                self.get_value() + coerced_other.get_value()
            )

        self_min, self_max = self._get_effective_min_max()
        other_min, other_max = coerced_other._get_effective_min_max()
        new_min = (
            None if (self_min is None or other_min is None) else self_min + other_min
        )
        new_max = (
            None if (self_max is None or other_max is None) else self_max + other_max
        )
        return self._create_param_from_min_max(self, new_min, new_max)

    def __radd__(self, other: Any) -> "BoundIntParam":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "BoundIntParam":
        coerced_other = self._coerce_other(other)
        if self.is_value_set() and coerced_other.is_value_set():
            return BoundIntParam.with_value(
                self.get_value() - coerced_other.get_value()
            )

        self_min, self_max = self._get_effective_min_max()
        other_min, other_max = coerced_other._get_effective_min_max()
        new_min = (
            None if (self_min is None or other_max is None) else self_min - other_max
        )
        new_max = (
            None if (self_max is None or other_min is None) else self_max - other_min
        )
        return self._create_param_from_min_max(self, new_min, new_max)

    def __rsub__(self, other: Any) -> "BoundIntParam":
        return self._coerce_other(other).__sub__(self)

    def __neg__(self) -> "BoundIntParam":
        if self.is_value_set():
            return BoundIntParam.with_value(-self.get_value())

        self_min, self_max = self._get_effective_min_max()
        new_min = None if self_max is None else -self_max
        new_max = None if self_min is None else -self_min
        return self._create_param_from_min_max(self, new_min, new_max)


class BoundNatParam(BoundIntParam, NatParam):
    """A bounded natural number parameter (i.e., non-negative integers)."""

    def __init__(
        self,
        name: Identifier | None = None,
        is_zero_included: bool = True,
        prefer_inclusive: bool = True,
    ) -> None:
        NatParam.__init__(self, name, is_zero_included)
        BoundIntParam.__init__(self, name, prefer_inclusive=prefer_inclusive)
