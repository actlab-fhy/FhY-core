"""Builders shared by the search-space suites.

The builders name every part with a fresh `Identifier`, so two calls build
spaces that are the same up to renaming: alpha-equivalent, and not
structurally equivalent.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from fhy_core.identifier import Identifier
from fhy_core.search_space import (
    Alternative,
    Choice,
    Condition,
    Configuration,
    Forbidden,
    Space,
    Variable,
)
from fhy_core.symbolic.constraint import (
    Constraint,
    ConstraintBindings,
    ConstraintOutcome,
    InSetConstraint,
)
from fhy_core.symbolic.expression import Expression, LiteralExpression
from fhy_core.symbolic.param import Param, create_categorical_param
from fhy_core.utils.override import override

__all__ = [
    "EXPLOSIONS",
    "ExplodingConstraint",
    "Explosion",
    "TilingSpace",
    "build_chain",
    "build_complete_configuration",
    "build_tiling_space",
    "categorical",
    "make_alternative",
    "make_choice",
    "make_variable",
]


def categorical(*values: Any) -> Param[Any]:
    """Return a categorical param over `values`, `{1, 2}` when none are given."""
    return create_categorical_param(frozenset(values or (1, 2)))


def make_variable(label: str, *values: Any) -> Variable[Any]:
    """Return a plain variable named `label` over the categories `values`."""
    return Variable(param=categorical(*values), name=Identifier(label))


def make_alternative(
    label: str,
    variables: Iterable[Variable[Any]] = (),
    choices: Iterable[Choice] = (),
) -> Alternative:
    """Return a plain alternative named `label`."""
    return Alternative(
        variables=tuple(variables), choices=tuple(choices), name=Identifier(label)
    )


def make_choice(label: str, *alternatives: Alternative) -> Choice:
    """Return a choice named `label` among `alternatives`."""
    return Choice(alternatives, name=Identifier(label))


@dataclass(frozen=True)
class TilingSpace:
    """A space choosing between a tiled and a flat layout.

    `unroll` is a top-level variable over `{1, 2, 4}`; `layout` chooses
    `tiled`, which holds `tile` over `{4, 8}`, or `flat`, which holds
    nothing. Canonical order: `unroll`, `layout`, `tile`.
    """

    space: Space
    unroll: Variable[Any]
    layout: Choice
    tiled: Alternative
    flat: Alternative
    tile: Variable[Any]
    conditions: tuple[Condition, ...] = field(default=())
    forbidden: tuple[Forbidden, ...] = field(default=())


def build_tiling_space(
    *,
    unroll_on_tiled_only: bool = False,
    forbid_unroll_four: bool = False,
) -> TilingSpace:
    """Return a fresh tiling space.

    Args:
        unroll_on_tiled_only: Add the condition that `unroll` is active
            only while `layout` is `tiled`.
        forbid_unroll_four: Add the forbidden clause `unroll in {4}`.

    """
    unroll = make_variable("unroll", 1, 2, 4)
    tile = make_variable("tile", 4, 8)
    tiled = make_alternative("tiled", (tile,))
    flat = make_alternative("flat")
    layout = make_choice("layout", tiled, flat)
    conditions = (
        (Condition(unroll.name, (InSetConstraint(layout.name, {tiled.name}),)),)
        if unroll_on_tiled_only
        else ()
    )
    forbidden = (
        (Forbidden((InSetConstraint(unroll.name, {4}),)),) if forbid_unroll_four else ()
    )
    space = Space(
        variables=(unroll,),
        choices=(layout,),
        conditions=conditions,
        forbidden=forbidden,
        name=Identifier("tiling"),
    )
    return TilingSpace(
        space=space,
        unroll=unroll,
        layout=layout,
        tiled=tiled,
        flat=flat,
        tile=tile,
        conditions=conditions,
        forbidden=forbidden,
    )


def build_complete_configuration(tiling: TilingSpace, tile: int = 4) -> Configuration:
    """Return the complete configuration of `tiling` choosing `tiled`."""
    return Configuration(
        tiling.space,
        {
            tiling.unroll.name: 2,
            tiling.layout.name: tiling.tiled.name,
            tiling.tile.name: tile,
        },
    )


def build_chain(depth: int) -> Choice:
    """Return a choice nesting `depth` levels of choices, one alternative each."""
    choice = make_choice("level_1", make_alternative("leaf"))
    for level in range(2, depth + 1):
        choice = make_choice(
            f"level_{level}", make_alternative(f"holder_{level}", choices=(choice,))
        )
    return choice


class Explosion(Exception):
    """The exception the test hooks and constraints raise."""


EXPLOSIONS: dict[str, Explosion] = {}
"""The exceptions `ExplodingConstraint`s raise, by their labels."""


@dataclass(frozen=True, eq=False)
class ExplodingConstraint(Constraint):
    """A Python-defined constraint whose evaluation raises `EXPLOSIONS[label]`.

    The exception lives in a module dict, not in a field: a constraint's
    fields must be immutable and serializable.
    """

    variable: Identifier
    label: str

    @override
    def get_free_identifiers(self) -> frozenset[Identifier]:
        return frozenset({self.variable})

    @override
    def evaluate_with_bindings(self, bindings: ConstraintBindings) -> ConstraintOutcome:
        raise EXPLOSIONS[self.label]

    @override
    def convert_to_expression(self) -> Expression:
        return LiteralExpression(True)

    @override
    def build_ordering_key(self) -> str:
        return f"ExplodingConstraint|{self.variable.id}"

    @override
    def __repr__(self) -> str:
        return f"ExplodingConstraint({self.variable!r})"

    @override
    def __str__(self) -> str:
        return f"explodes({self.variable!r})"
