"""Hypothesis properties of `fhy_core.search_space`, mirroring the Rust laws.

A space is drawn as a shape (top-level variables' domains, and choices
whose alternatives hold variables and at most one level of sub-choices) and
built with fresh names, so building one shape twice gives two spaces that
are the same up to renaming. A configuration is drawn as a list of picks,
read in canonical order: an alternative's position for each choice reached,
a category's position for each variable reached.
"""

from collections.abc import Iterator, Sequence
from typing import Any

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given
from hypothesis import strategies as st

from fhy_core.identifier import Identifier
from fhy_core.search_space import (
    Alternative,
    Choice,
    Configuration,
    Space,
    Variable,
)
from fhy_core.symbolic.param import create_categorical_param

from ..strategies.settings import cap_max_examples

pytestmark = pytest.mark.property

_Domain = frozenset[int]
_AlternativeShape = tuple[list[_Domain], list[Any]]
_ChoiceShape = list[_AlternativeShape]
_SpaceShape = tuple[list[_Domain], list[_ChoiceShape]]

_DOMAINS = st.frozensets(st.sampled_from((1, 2, 3)), min_size=1)


def _alternative_shapes(depth: int) -> st.SearchStrategy[_AlternativeShape]:
    """Return alternatives' shapes nesting at most `depth` more choices."""
    sub_choices = (
        st.lists(_choice_shapes(depth - 1), max_size=1) if depth > 0 else st.just([])
    )
    return st.tuples(st.lists(_DOMAINS, max_size=2), sub_choices)


def _choice_shapes(depth: int) -> st.SearchStrategy[_ChoiceShape]:
    """Return choices' shapes of one to three alternatives."""
    return st.lists(_alternative_shapes(depth), min_size=1, max_size=3)


_SPACE_SHAPES: st.SearchStrategy[_SpaceShape] = st.tuples(
    st.lists(_DOMAINS, max_size=2), st.lists(_choice_shapes(1), max_size=2)
)
_SPACE_SHAPES_WITH_A_VARIABLE: st.SearchStrategy[_SpaceShape] = st.tuples(
    st.lists(_DOMAINS, min_size=1, max_size=2), st.lists(_choice_shapes(1), max_size=2)
)
_PICKS = st.lists(st.integers(min_value=0, max_value=5), min_size=1, max_size=12)


def _build_variable(domain: _Domain) -> Variable[Any]:
    return Variable(param=create_categorical_param(domain), name=Identifier("v"))


def _build_choice(shape: _ChoiceShape) -> Choice:
    return Choice(
        tuple(
            Alternative(
                variables=tuple(_build_variable(domain) for domain in domains),
                choices=tuple(_build_choice(sub) for sub in sub_choices),
                name=Identifier("a"),
            )
            for domains, sub_choices in shape
        ),
        name=Identifier("c"),
    )


def _build_space(shape: _SpaceShape) -> Space:
    """Return a space of `shape`, every name fresh."""
    domains, choices = shape
    return Space(
        variables=tuple(_build_variable(domain) for domain in domains),
        choices=tuple(_build_choice(choice) for choice in choices),
        name=Identifier("s"),
    )


def _cycle(picks: Sequence[int]) -> Iterator[int]:
    while True:
        yield from picks


def _entries(space: Space, picks: Sequence[int]) -> list[tuple[Identifier, Any]]:
    """Return a complete configuration's entries, choosing by `picks`."""
    pick = _cycle(picks)
    entries: list[tuple[Identifier, Any]] = []

    def assign(variables: Sequence[Variable[Any]], choices: Sequence[Choice]) -> None:
        for variable in variables:
            domain: Any = variable.param.domain
            categories = sorted(domain.categories)
            entries.append((variable.name, categories[next(pick) % len(categories)]))
        for choice in choices:
            chosen = choice.alternatives[next(pick) % len(choice.alternatives)]
            entries.append((choice.name, chosen.name))
            assign(chosen.variables, chosen.choices)

    assign(space.variables, space.choices)
    return entries


def _perturbed(shape: _SpaceShape) -> _SpaceShape:
    """Return `shape` with its first top-level domain changed."""
    domains, choices = shape
    first = domains[0]
    changed = first - {min(first)} if len(first) > 1 else first | {max(first) % 3 + 1}
    return ([changed, *domains[1:]], choices)


@cap_max_examples(60)
@given(_SPACE_SHAPES)
def test_space_text_round_trips(shape: _SpaceShape) -> None:
    """Test decoding a space's text writes the same text and an equivalent space."""
    space = _build_space(shape)
    text = space.to_json()

    decoded = Space.from_json(text)

    assert decoded.to_json() == text
    assert decoded.is_structurally_equivalent(space)


@cap_max_examples(60)
@given(_SPACE_SHAPES)
def test_relabeled_spaces_are_alpha_equivalent_both_ways(shape: _SpaceShape) -> None:
    """Test two builds of one shape are alpha- and not structurally equivalent."""
    left, right = _build_space(shape), _build_space(shape)

    assert left.is_structurally_equivalent(left)
    assert left.is_alpha_equivalent(right)
    assert right.is_alpha_equivalent(left)
    assert not left.is_structurally_equivalent(right)


@cap_max_examples(60)
@given(_SPACE_SHAPES)
def test_structural_equivalence_implies_alpha_equivalence(shape: _SpaceShape) -> None:
    """Test a space rebuilt from the same parts is equivalent both ways."""
    space = _build_space(shape)
    rebuilt = Space(variables=space.variables, choices=space.choices, name=space.name)

    assert rebuilt.is_structurally_equivalent(space)
    assert space.is_structurally_equivalent(rebuilt)
    assert rebuilt.is_alpha_equivalent(space)


@cap_max_examples(60)
@given(_SPACE_SHAPES_WITH_A_VARIABLE)
def test_a_changed_domain_breaks_alpha_equivalence(shape: _SpaceShape) -> None:
    """Test changing one domain makes a relabeled build inequivalent."""
    left, right = _build_space(shape), _build_space(_perturbed(shape))

    assert not left.is_alpha_equivalent(right)
    assert not right.is_alpha_equivalent(left)


@cap_max_examples(60)
@given(_SPACE_SHAPES, _PICKS)
def test_corresponding_configurations_share_a_key(
    shape: _SpaceShape, picks: list[int]
) -> None:
    """Test configurations picked alike in relabeled spaces have one key."""
    left_space, right_space = _build_space(shape), _build_space(shape)

    left = Configuration(left_space, _entries(left_space, picks))
    right = Configuration(right_space, _entries(right_space, picks))

    assert left.is_complete()
    assert left.key() == right.key()
    assert hash(left.key()) == hash(right.key())
    assert left.is_alpha_equivalent(right)


@cap_max_examples(60)
@given(_SPACE_SHAPES, _PICKS)
def test_configuration_text_round_trips(shape: _SpaceShape, picks: list[int]) -> None:
    """Test a configuration decodes to one with its text and its key."""
    space = _build_space(shape)
    configuration = Configuration(space, _entries(space, picks))
    text = configuration.to_json()

    decoded = Configuration.from_json(text)

    assert decoded.to_json() == text
    assert decoded.key() == configuration.key()
