"""Hypothesis strategies for `fhy_core.types` instances.

Shapes and index bounds use literal ints or pool identifiers, mirroring
the leaf choices in :mod:`tests.strategies.expressions`.
"""

from collections.abc import Sequence
from typing import Final

from hypothesis import strategies as st

from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression import (
    Expression,
    IdentifierExpression,
    LiteralExpression,
)
from fhy_core.types import (
    CoreDataType,
    IndexType,
    NumericalType,
    PrimitiveDataType,
    TemplateDataType,
    Type,
    TypeQualifier,
)

from .identifiers import build_identifier_strategy

__all__ = [
    "build_core_data_type_strategy",
    "build_primitive_data_type_strategy",
    "build_type_qualifier_strategy",
    "draw_index_type",
    "draw_numerical_type",
    "draw_shape",
    "draw_template_data_type",
    "draw_template_free_type",
]

_MIN_SHAPE_LITERAL: Final = 1
_MAX_SHAPE_LITERAL: Final = 8
_MAX_STRIDE_LITERAL: Final = 3
_MAX_INDEX_BOUND: Final = 20
_MAX_TEMPLATE_WIDTH: Final = 8
_MAX_TEMPLATE_RANK: Final = 3


def build_core_data_type_strategy() -> st.SearchStrategy[CoreDataType]:
    """Return a strategy sampling every ``CoreDataType`` member."""
    return st.sampled_from(CoreDataType)


def build_type_qualifier_strategy() -> st.SearchStrategy[TypeQualifier]:
    """Return a strategy sampling every ``TypeQualifier`` member."""
    return st.sampled_from(TypeQualifier)


def build_primitive_data_type_strategy() -> st.SearchStrategy[PrimitiveDataType]:
    """Return a strategy for a ``PrimitiveDataType`` over every core data type."""
    return build_core_data_type_strategy().map(PrimitiveDataType)


@st.composite
def draw_shape(
    draw: st.DrawFn, identifiers: Sequence[Identifier], max_rank: int = 2
) -> tuple[Expression, ...]:
    """Draw a shape of at most ``max_rank`` literal-int or identifier dimensions."""
    rank = draw(st.integers(min_value=0, max_value=max_rank))
    dimensions: list[Expression] = []
    for _ in range(rank):
        use_identifier = bool(identifiers) and draw(st.booleans())
        if use_identifier:
            identifier = draw(build_identifier_strategy(identifiers))
            dimensions.append(IdentifierExpression(identifier))
        else:
            literal_value = draw(
                st.integers(min_value=_MIN_SHAPE_LITERAL, max_value=_MAX_SHAPE_LITERAL)
            )
            dimensions.append(LiteralExpression(literal_value))
    return tuple(dimensions)


@st.composite
def draw_numerical_type(
    draw: st.DrawFn, identifiers: Sequence[Identifier], max_rank: int = 2
) -> NumericalType:
    """Draw a template-free ``NumericalType`` over a primitive data type and a shape."""
    data_type = draw(build_primitive_data_type_strategy())
    shape = draw(draw_shape(identifiers, max_rank))
    return NumericalType(data_type, shape)


@st.composite
def draw_index_type(draw: st.DrawFn, identifiers: Sequence[Identifier]) -> IndexType:
    """Draw an ``IndexType`` with literal bounds (``lower <= upper``) and stride.

    Args:
        draw: The active Hypothesis draw function.
        identifiers: Accepted for a signature consistent with this
            module's other ``draw_*`` functions; bounds and stride are
            always literal.

    Returns:
        An ``IndexType`` with literal lower and upper bounds and a
        literal stride in ``[1, 3]``.

    """
    del identifiers
    lower = draw(st.integers(min_value=0, max_value=_MAX_INDEX_BOUND))
    extra = draw(st.integers(min_value=0, max_value=_MAX_INDEX_BOUND))
    upper = lower + extra
    stride = draw(st.integers(min_value=1, max_value=_MAX_STRIDE_LITERAL))
    return IndexType(
        LiteralExpression(lower), LiteralExpression(upper), LiteralExpression(stride)
    )


@st.composite
def draw_template_free_type(draw: st.DrawFn, identifiers: Sequence[Identifier]) -> Type:
    """Draw a template-free ``Type``: either a ``NumericalType`` or an ``IndexType``."""
    if draw(st.booleans()):
        return draw(draw_numerical_type(identifiers))
    return draw(draw_index_type(identifiers))


@st.composite
def draw_template_data_type(
    draw: st.DrawFn, identifiers: Sequence[Identifier]
) -> TemplateDataType:
    """Draw a ``TemplateDataType`` naming a pool identifier, with optional widths."""
    identifier = draw(build_identifier_strategy(identifiers))
    widths = draw(
        st.one_of(
            st.none(),
            st.lists(
                st.integers(min_value=1, max_value=_MAX_TEMPLATE_WIDTH),
                min_size=1,
                max_size=_MAX_TEMPLATE_RANK,
            ),
        )
    )
    return TemplateDataType(identifier, widths)
