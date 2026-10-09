"""Hypothesis strategy for a tagged union of `Serializable` instances.

Covers every family the generic DICT round-trip property in
``tests/test_strategies_properties.py`` exercises: expressions,
constraints, constraint systems, params, param domains, types, and
provenance (including its ``Position`` and ``Span`` leaves). A mock
identifier is deliberately not one of these families: its
``serialize_to_dict``/``deserialize_from_dict`` hooks
(``tests/conftest.py::mock_identifier``) are instance attributes on the
``Mock``, not classmethods, so the generic
``type(instance).deserialize_from_dict(...)`` round trip this module
supports raises ``AttributeError`` for one. See the module docstring's
sibling note in ``tests/test_strategies_properties.py`` for detail.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from hypothesis import strategies as st

from fhy_core.provenance import (
    CallSiteProvenance,
    FileProvenance,
    FusedProvenance,
    NamedProvenance,
    Position,
    Provenance,
    Span,
    UnknownProvenance,
)
from fhy_core.serialization import Serializable
from fhy_core.types import (
    is_structurally_equivalent as types_are_structurally_equivalent,
)

from .constraints import (
    draw_bound_equation_constraint,
    draw_constraint_system,
    draw_in_set_constraint,
    draw_not_in_set_constraint,
)
from .identifiers import build_identifier_pool, build_identifier_strategy
from .params import draw_param_over_any_domain
from .structural_expressions import build_structural_expression_strategy
from .types import draw_template_data_type, draw_template_free_type

__all__ = [
    "SerializableCase",
    "draw_serializable_case",
]

_POOL: Final = build_identifier_pool(4, name_prefix="s")
_MAX_PROVENANCE_DEPTH: Final = 2
_MAX_POSITION_LINE: Final = 500
_MAX_POSITION_EXTRA: Final = 5
_MAX_SPAN_OFFSET: Final = 1000


@dataclass(frozen=True)
class SerializableCase:
    """A serializable instance with the equivalence check its round trip must satisfy.

    Attributes:
        label: Short name for the family this case belongs to, for
            failure messages.
        instance: The serializable instance under test.
        are_equivalent: Predicate deciding whether a restored instance
            is an acceptable round trip of ``instance``.

    """

    label: str
    instance: Serializable
    are_equivalent: Callable[[Any, Any], bool]


def _is_equal(left: Any, right: Any) -> bool:
    """Return whether ``left == right``."""
    return bool(left == right)


def _are_structurally_equivalent(left: Any, right: Any) -> bool:
    """Return whether ``left.is_structurally_equivalent(right)``."""
    result: bool = left.is_structurally_equivalent(right)
    return result


def _are_types_structurally_equivalent(left: Any, right: Any) -> bool:
    """Return whether ``left`` and ``right`` are structurally equivalent types."""
    return types_are_structurally_equivalent(left, right)


@st.composite
def _draw_expression_case(draw: st.DrawFn) -> SerializableCase:
    """Draw a ``SerializableCase`` wrapping a structural expression tree."""
    instance = draw(build_structural_expression_strategy(_POOL))
    return SerializableCase("expression", instance, _are_structurally_equivalent)


@st.composite
def _draw_constraint_case(draw: st.DrawFn) -> SerializableCase:
    """Draw a ``SerializableCase`` wrapping a single constraint."""
    variable = draw(build_identifier_strategy(_POOL))
    instance = draw(
        st.one_of(
            draw_in_set_constraint(variable),
            draw_not_in_set_constraint(variable),
            draw_bound_equation_constraint(variable),
        )
    )
    return SerializableCase("constraint", instance, _are_structurally_equivalent)


@st.composite
def _draw_constraint_system_case(draw: st.DrawFn) -> SerializableCase:
    """Draw a ``SerializableCase`` wrapping a constraint system."""
    system, _members = draw(draw_constraint_system(_POOL))
    return SerializableCase("constraint_system", system, _are_structurally_equivalent)


@st.composite
def _draw_param_case(draw: st.DrawFn) -> SerializableCase:
    """Draw a ``SerializableCase`` wrapping a param over some domain."""
    instance = draw(draw_param_over_any_domain())
    return SerializableCase("param", instance, _are_structurally_equivalent)


@st.composite
def _draw_param_domain_case(draw: st.DrawFn) -> SerializableCase:
    """Draw a ``SerializableCase`` wrapping a param's domain."""
    param = draw(draw_param_over_any_domain())
    return SerializableCase("param_domain", param.domain, _are_structurally_equivalent)


@st.composite
def _draw_type_case(draw: st.DrawFn) -> SerializableCase:
    """Draw a ``SerializableCase`` wrapping a template-free or template data type."""
    instance = draw(
        st.one_of(draw_template_free_type(_POOL), draw_template_data_type(_POOL))
    )
    return SerializableCase("type", instance, _are_types_structurally_equivalent)


def _build_position_strategy() -> st.SearchStrategy[Position]:
    """Return a strategy for a ``Position`` with a positive line and column."""
    return st.builds(
        Position,
        line=st.integers(min_value=1, max_value=_MAX_POSITION_LINE),
        column=st.integers(min_value=1, max_value=_MAX_POSITION_LINE),
    )


@st.composite
def _draw_ordered_position_pair(draw: st.DrawFn) -> tuple[Position, Position]:
    """Draw ``(start, end)`` positions with ``start <= end`` by construction."""
    start_line = draw(st.integers(min_value=1, max_value=_MAX_POSITION_LINE))
    start_column = draw(st.integers(min_value=1, max_value=_MAX_POSITION_LINE))
    extra_lines = draw(st.integers(min_value=0, max_value=_MAX_POSITION_EXTRA))
    if extra_lines == 0:
        extra_columns = draw(st.integers(min_value=0, max_value=_MAX_POSITION_LINE))
        end_column = start_column + extra_columns
    else:
        end_column = draw(st.integers(min_value=1, max_value=_MAX_POSITION_LINE))
    end_line = start_line + extra_lines
    return Position(start_line, start_column), Position(end_line, end_column)


@st.composite
def _draw_span(draw: st.DrawFn) -> Span:
    """Draw a ``Span``: unknown, offset-bounded, or position-bounded."""
    kind = draw(st.integers(min_value=0, max_value=2))
    if kind == 0:
        return Span()
    if kind == 1:
        start_offset = draw(st.integers(min_value=0, max_value=_MAX_SPAN_OFFSET))
        extra = draw(st.integers(min_value=0, max_value=_MAX_SPAN_OFFSET))
        return Span(start_offset=start_offset, end_offset=start_offset + extra)
    start_position, end_position = draw(_draw_ordered_position_pair())
    return Span(start_position=start_position, end_position=end_position)


def _build_file_path_strategy() -> st.SearchStrategy[Path]:
    """Return a strategy for a short relative file path."""
    return st.sampled_from(("a.fhy", "b/c.fhy", "package/module.fhy")).map(Path)


@st.composite
def _draw_provenance(
    draw: st.DrawFn, max_depth: int = _MAX_PROVENANCE_DEPTH
) -> Provenance:
    """Draw a provenance tree bounded to at most ``max_depth`` levels of nesting."""
    if max_depth <= 0:
        return draw(_draw_provenance_leaf())
    if draw(st.booleans()):
        return draw(_draw_provenance_leaf())
    return draw(_draw_provenance_branch(max_depth))


def _draw_provenance_leaf() -> st.SearchStrategy[Provenance]:
    """Return a strategy for a leaf ``Provenance``: unknown or file-anchored."""
    file_provenance: st.SearchStrategy[Provenance] = st.builds(
        FileProvenance, file_path=_build_file_path_strategy(), span=_draw_span()
    )
    unknown_provenance: st.SearchStrategy[Provenance] = st.builds(UnknownProvenance)
    return st.one_of(unknown_provenance, file_provenance)


@st.composite
def _draw_provenance_branch(draw: st.DrawFn, max_depth: int) -> Provenance:
    """Draw a composite ``Provenance`` node wrapping one or more child provenances."""
    kind = draw(st.integers(min_value=0, max_value=2))
    if kind == 0:
        name = draw(st.text(min_size=1, max_size=6))
        child = draw(_draw_provenance(max_depth - 1))
        return NamedProvenance(name, child)
    if kind == 1:
        callee = draw(_draw_provenance(max_depth - 1))
        caller = draw(_draw_provenance(max_depth - 1))
        return CallSiteProvenance(callee, caller)
    sources = draw(st.lists(_draw_provenance(max_depth - 1), min_size=1, max_size=3))
    return FusedProvenance(tuple(sources))


@st.composite
def _draw_provenance_case(draw: st.DrawFn) -> SerializableCase:
    """Draw a ``SerializableCase`` for a ``Position``, ``Span``, or ``Provenance``."""
    instance = draw(
        st.one_of(_build_position_strategy(), _draw_span(), _draw_provenance())
    )
    return SerializableCase("provenance", instance, _is_equal)


@st.composite
def draw_serializable_case(draw: st.DrawFn) -> SerializableCase:
    """Draw a ``Serializable`` from one covered family with its equivalence check."""
    result: SerializableCase = draw(
        st.one_of(
            _draw_expression_case(),
            _draw_constraint_case(),
            _draw_constraint_system_case(),
            _draw_param_case(),
            _draw_param_domain_case(),
            _draw_type_case(),
            _draw_provenance_case(),
        )
    )
    return result
