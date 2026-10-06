"""Benchmarks of the search-space package.

They measure ``fhy_core.search_space``, which the Rust core backs, through
the public API only, on the shape MOGA-VM's audit baseline measured: one
choice of eight alternatives, each holding four variables over the
categories ``{1, 2, 3, 4}``. The before numbers are MOGA-VM's Python core
on fhy_core 0.2.0 (``docs/design/search-space.md``, "Benchmark plan").

The Python API lands in SS1.7, which adapts these rows to its final
signatures; until then the module is absent and every row skips.
"""

from typing import Any

import pytest

from fhy_core.identifier import Identifier
from fhy_core.symbolic.constraint import InSetConstraint
from fhy_core.symbolic.param import create_categorical_param

from .conftest import Benchmark

pytestmark = pytest.mark.benchmark(group="search_space")

search_space = pytest.importorskip("fhy_core.search_space")

# The shape of the benchmarked space: alternatives of the one choice, and
# variables per alternative.
_ALTERNATIVE_COUNT = 8
_VARIABLE_COUNT = 4
_CATEGORIES = frozenset({1, 2, 3, 4})


def _build_space(
    tag: str, *, conditions: tuple[Any, ...] = (), forbidden: tuple[Any, ...] = ()
) -> Any:
    """Return the benchmarked space, every name minted fresh under ``tag``."""
    alternatives = tuple(
        search_space.Alternative(
            name=Identifier(f"{tag}_alternative_{index}"),
            variables=tuple(
                search_space.Variable(
                    name=Identifier(f"{tag}_variable_{index}_{position}"),
                    param=create_categorical_param(_CATEGORIES),
                )
                for position in range(_VARIABLE_COUNT)
            ),
        )
        for index in range(_ALTERNATIVE_COUNT)
    )
    choice = search_space.Choice(
        name=Identifier(f"{tag}_choice"), alternatives=alternatives
    )
    return search_space.Space(
        name=Identifier(f"{tag}_space"),
        choices=(choice,),
        conditions=conditions,
        forbidden=forbidden,
    )


def _first_entries(space: Any) -> dict[Identifier, Any]:
    """Return entries choosing the first alternative and assigning its variables."""
    (choice,) = space.choices
    first = choice.alternatives[0]
    entries: dict[Identifier, Any] = {choice.name: first.name}
    entries.update({variable.name: 1 for variable in first.variables})
    return entries


def _build_tagged_alternative(name: Identifier, tag: int) -> Any:
    """Return an alternative with data of its own, compared through hooks."""
    base: Any = search_space.Alternative

    class TaggedAlternative(base):  # type: ignore[misc]
        def __init__(self, *, tag: int, **fields: Any) -> None:
            super().__init__(**fields)
            self.tag = tag

        def extension_is_structurally_equivalent(self, other: Any) -> bool:
            return bool(self.tag == other.tag)

        def extension_is_alpha_equivalent_under(
            self, other: Any, renaming: Any
        ) -> bool:
            del renaming
            return bool(self.tag == other.tag)

    return TaggedAlternative(name=name, tag=tag)


def _build_hooked_space(tag: str) -> Any:
    """Return a space of one choice whose one alternative has hooks."""
    alternative = _build_tagged_alternative(Identifier(f"{tag}_alternative"), 1)
    choice = search_space.Choice(
        name=Identifier(f"{tag}_choice"), alternatives=(alternative,)
    )
    return search_space.Space(name=Identifier(f"{tag}_space"), choices=(choice,))


def test_space_construction(benchmark: Benchmark) -> None:
    """Benchmark building the space, its params and parts included."""
    benchmark(_build_space, "built")


def test_space_construction_with_conditions_and_forbidden(benchmark: Benchmark) -> None:
    """Benchmark building a space with a condition and a forbidden clause."""
    template = _build_space("template")
    (choice,) = template.choices
    first, second = choice.alternatives[0], choice.alternatives[1]

    def build() -> Any:
        return search_space.Space(
            name=Identifier("guarded_space"),
            choices=(choice,),
            conditions=(
                search_space.Condition(
                    target=second.variables[0].name,
                    when=(InSetConstraint(choice.name, frozenset({second.name})),),
                ),
            ),
            forbidden=(
                search_space.Forbidden(
                    when=(InSetConstraint(first.variables[0].name, frozenset({4})),)
                ),
            ),
        )

    benchmark(build)


def test_configuration_construction(benchmark: Benchmark) -> None:
    """Benchmark building and checking a complete configuration."""
    space = _build_space("validated")
    entries = _first_entries(space)
    benchmark(search_space.Configuration, space, entries)


def test_configuration_with_entry(benchmark: Benchmark) -> None:
    """Benchmark replacing one value of a configuration."""
    space = _build_space("assigned")
    configuration = search_space.Configuration(space, _first_entries(space))
    variable = space.choices[0].alternatives[0].variables[0]
    benchmark(configuration.with_entry, variable.name, 2)


def test_configuration_key(benchmark: Benchmark) -> None:
    """Benchmark a configuration's key."""
    space = _build_space("keyed")
    configuration = search_space.Configuration(space, _first_entries(space))
    benchmark(configuration.key)


def test_space_structural_self_equivalence(benchmark: Benchmark) -> None:
    """Benchmark structural equivalence of a space with itself."""
    space = _build_space("structural")
    benchmark(space.is_structurally_equivalent, space)


def test_space_alpha_equivalence_against_a_relabeled_copy(benchmark: Benchmark) -> None:
    """Benchmark alpha equivalence against a copy with fresh names."""
    left, right = _build_space("left"), _build_space("right")
    benchmark(left.is_alpha_equivalent, right)


def test_space_alpha_equivalence_through_python_hooks(benchmark: Benchmark) -> None:
    """Benchmark alpha equivalence calling a Python subclass's hooks."""
    left, right = (
        _build_hooked_space("hooked_left"),
        _build_hooked_space("hooked_right"),
    )
    benchmark(left.is_alpha_equivalent, right)


def test_configuration_serialize_to_dict(benchmark: Benchmark) -> None:
    """Benchmark serializing a configuration."""
    space = _build_space("serialized")
    configuration = search_space.Configuration(space, _first_entries(space))
    benchmark(configuration.serialize_to_dict)


def test_variable_serialize_to_dict(benchmark: Benchmark) -> None:
    """Benchmark serializing a variable."""
    variable = search_space.Variable(
        name=Identifier("serialized_variable"),
        param=create_categorical_param(_CATEGORIES),
    )
    benchmark(variable.serialize_to_dict)
