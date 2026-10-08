"""The interface suite of `fhy_core.search_space`.

Construction and its refusals, activity, completeness, configurations and
their keys, and the equivalences, through the public API. The tests ported
from MOGA-VM's `tests/cir/space/` (on its `origin/dev`) cite the MOGA-VM
test in their docstrings.
"""

from collections.abc import Callable
from typing import Any

import pytest

from fhy_core.diagnostic import Note
from fhy_core.identifier import Identifier
from fhy_core.search_space import (
    Activity,
    Alternative,
    Cardinality,
    CardinalityKind,
    Choice,
    Condition,
    Configuration,
    ConfigurationError,
    ConfigurationKey,
    DuplicateNameError,
    Forbidden,
    RandomOracle,
    SearchSpaceError,
    Space,
    Variable,
)
from fhy_core.symbolic.constraint import (
    ConstraintSystem,
    EquationConstraint,
    InSetConstraint,
    NotInSetConstraint,
)
from fhy_core.symbolic.expression import IdentifierExpression, LiteralExpression
from fhy_core.symbolic.param import create_integer_param_between
from fhy_core.term import AlphaRenaming

from ..serialization.foreign_parts import GoldenEven
from .conftest import (
    EXPLOSIONS,
    ExplodingConstraint,
    Explosion,
    build_complete_configuration,
    build_tiling_space,
    categorical,
    make_alternative,
    make_choice,
    make_variable,
)

# ===========================================================================
# Construction
# ===========================================================================


def test_variable_keeps_its_fields() -> None:
    """Test a variable returns the name, param and notes it was built from."""
    name = Identifier("k")
    param = categorical(1, 2)

    variable = Variable(param=param, name=name)

    assert variable.name is name
    assert variable.param is param
    assert variable.notes == ()
    assert variable.kind == "search_space.variable"


def test_variable_without_a_name_gets_a_fresh_one() -> None:
    """Test each unnamed variable is named by a new `variable` identifier."""
    first = Variable(param=categorical())
    second = Variable(param=categorical())

    assert first.name.name_hint == "variable"
    assert second.name.name_hint == "variable"
    assert first.name != second.name


@pytest.mark.parametrize(
    ("build", "hint"),
    [
        (Alternative, "alternative"),
        (lambda: Choice((Alternative(),)), "choice"),
        (Space, "space"),
    ],
    ids=["alternative", "choice", "space"],
)
def test_unnamed_parts_get_fresh_names(build: Callable[[], Any], hint: str) -> None:
    """Test an unnamed alternative, choice or space is named by its kind."""
    first, second = build(), build()

    assert first.name.name_hint == hint
    assert first.name != second.name


def test_alternative_keeps_its_fields() -> None:
    """Test an alternative returns its variables and sub-choices as given."""
    tile = make_variable("tile")
    inner = make_choice("inner", make_alternative("only"))

    alternative = Alternative(
        variables=[tile], choices=(inner,), name=Identifier("tiled")
    )

    assert alternative.variables == (tile,)
    assert alternative.variables[0] is tile
    assert alternative.choices[0] is inner
    assert alternative.kind == "search_space.alternative"


def test_notes_are_kept_as_given() -> None:
    """Test every class with notes returns the note objects given."""
    note = Note("hot")
    alternative = Alternative(notes=(note,))

    for value in (
        Variable(param=categorical(), notes=[note]),
        alternative,
        Choice((alternative,), notes=(note,)),
        Space(notes=(note,)),
    ):
        assert value.notes == (note,)
        assert value.notes[0] is note


def test_choice_keeps_its_alternatives_in_order() -> None:
    """Test a choice returns its alternatives as given, in order."""
    first, second = make_alternative("a"), make_alternative("b")

    choice = Choice([first, second], name=Identifier("c"))

    assert choice.alternatives == (first, second)
    assert choice.alternatives[1] is second


def test_space_keeps_its_parts() -> None:
    """Test a space returns its variables, choices and clauses as given."""
    tiling = build_tiling_space(forbid_unroll_four=True)

    assert tiling.space.variables == (tiling.unroll,)
    assert tiling.space.choices[0] is tiling.layout
    assert tiling.space.forbidden == tiling.forbidden
    assert tiling.space.name.name_hint == "tiling"


def test_condition_keeps_a_system_given() -> None:
    """Test a condition given a `ConstraintSystem` returns that system."""
    target, choice = Identifier("t"), Identifier("c")
    system = ConstraintSystem((InSetConstraint(choice, {Identifier("a")}),))

    condition = Condition(target, system)

    assert condition.target is target
    assert condition.when is system


def test_condition_builds_a_system_of_constraints() -> None:
    """Test a condition given constraints holds a system of exactly those."""
    member = InSetConstraint(Identifier("c"), {Identifier("a")})

    condition = Condition(Identifier("t"), [member])

    assert isinstance(condition.when, ConstraintSystem)
    assert condition.when.constraints == (member,)


def test_forbidden_builds_a_system_of_constraints() -> None:
    """Test a forbidden clause given constraints holds a system of them."""
    member = InSetConstraint(Identifier("x"), {4})

    clause = Forbidden((member,))

    assert clause.when.constraints == (member,)


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (
            lambda: Variable(param=3),  # type: ignore[arg-type]
            "Variable param must be a Param, got int.",
        ),
        (
            lambda: Variable(param=categorical(), name="k"),  # type: ignore[arg-type]
            "Variable name must be an Identifier, got str.",
        ),
        (
            lambda: Variable(param=categorical(), notes=(1,)),  # type: ignore[arg-type]
            "Variable notes must be Notes, got int.",
        ),
        (
            lambda: Alternative(variables=(make_alternative("a"),)),  # type: ignore[arg-type]
            "Alternative variables must be Variables, got Alternative.",
        ),
        (
            lambda: Alternative(choices=(make_alternative("a"),)),  # type: ignore[arg-type]
            "Alternative choices must be Choices, got Alternative.",
        ),
        (
            lambda: Choice((make_variable("v"),)),  # type: ignore[arg-type]
            "Choice alternatives must be Alternatives, got Variable.",
        ),
        (
            lambda: Choice(make_alternative("a")),  # type: ignore[arg-type]
            "Choice alternatives must be Alternatives, got",
        ),
        (
            lambda: Space(conditions=(Forbidden(()),)),  # type: ignore[arg-type]
            "Space conditions must be Conditions, got Forbidden.",
        ),
        (
            lambda: Space(forbidden=(Condition(Identifier("t"), ()),)),  # type: ignore[arg-type]
            "Space forbidden must be Forbidden clauses, got Condition.",
        ),
        (
            lambda: Condition("t", ()),  # type: ignore[arg-type]
            "Condition target must be an Identifier, got str.",
        ),
        (
            lambda: Condition(Identifier("t"), (3,)),  # type: ignore[arg-type]
            "Condition when must be a ConstraintSystem or Constraints, got int.",
        ),
        (
            lambda: Forbidden(3),  # type: ignore[arg-type]
            "Forbidden when must be a ConstraintSystem or Constraints, got int.",
        ),
        (
            lambda: Configuration(3),  # type: ignore[arg-type]
            "Configuration space must be a Space, got int.",
        ),
    ],
    ids=[
        "variable-param",
        "variable-name",
        "variable-notes",
        "alternative-variables",
        "alternative-choices",
        "choice-alternatives",
        "choice-alternatives-not-iterable",
        "space-conditions",
        "space-forbidden",
        "condition-target",
        "condition-when",
        "forbidden-when",
        "configuration-space",
    ],
)
def test_constructors_refuse_arguments_of_the_wrong_type(
    build: Callable[[], Any], message: str
) -> None:
    """Test each constructor names the argument of the wrong type."""
    with pytest.raises(TypeError) as excinfo:
        build()

    assert str(excinfo.value).startswith(message)


def test_configuration_refuses_an_entry_name_that_is_no_identifier() -> None:
    """Test an entry named by a `str` is refused as a type error."""
    tiling = build_tiling_space()

    with pytest.raises(
        TypeError, match="Configuration entry name must be an Identifier"
    ):
        Configuration(tiling.space, {"unroll": 1})  # type: ignore[arg-type]


def test_configuration_refuses_entries_that_are_no_pairs() -> None:
    """Test an entry that is not a `(name, value)` pair is refused."""
    tiling = build_tiling_space()

    with pytest.raises(TypeError, match="Configuration entries must be"):
        Configuration(tiling.space, [tiling.unroll.name])  # type: ignore[list-item]


# ===========================================================================
# Refusals of the core
# ===========================================================================


def test_choice_without_alternatives_is_refused() -> None:
    """Test an empty choice raises `SearchSpaceError` naming the choice."""
    name = Identifier("empty")

    with pytest.raises(SearchSpaceError) as excinfo:
        Choice((), name=name)

    assert type(excinfo.value) is SearchSpaceError
    assert str(excinfo.value) == f"the choice {name!r} has no alternative"


def test_alternative_repeating_a_name_is_refused() -> None:
    """Test an alternative whose variable carries its own name is refused."""
    name = Identifier("twice")

    with pytest.raises(DuplicateNameError) as excinfo:
        Alternative(variables=(Variable(param=categorical(), name=name),), name=name)

    assert str(excinfo.value) == f"the name {name!r} is used more than once"


def test_choice_repeating_an_alternative_name_is_refused() -> None:
    """Test two alternatives of one name in a choice are refused."""
    name = Identifier("same")

    with pytest.raises(DuplicateNameError, match=f"the name {name!r} is used"):
        Choice((Alternative(name=name), Alternative(name=name)))


def test_space_repeating_a_name_across_levels_is_refused() -> None:
    """Test a top-level variable sharing a nested variable's name is refused."""
    name = Identifier("shared")
    nested = Variable(param=categorical(), name=name)
    choice = make_choice("c", make_alternative("a", (nested,)))

    with pytest.raises(DuplicateNameError) as excinfo:
        Space(variables=(Variable(param=categorical(), name=name),), choices=(choice,))

    assert str(excinfo.value) == f"the name {name!r} is used more than once"


def test_space_naming_itself_like_a_decision_is_refused() -> None:
    """Test a space whose own name is a decision's is refused."""
    variable = make_variable("v")

    with pytest.raises(DuplicateNameError):
        Space(variables=(variable,), name=variable.name)


def test_condition_on_an_unknown_target_is_refused() -> None:
    """Test a condition whose target is no decision is refused."""
    tiling = build_tiling_space()
    ghost = Identifier("ghost")
    condition = Condition(
        ghost, (InSetConstraint(tiling.layout.name, {tiling.tiled.name}),)
    )

    with pytest.raises(SearchSpaceError) as excinfo:
        Space(
            variables=(tiling.unroll,),
            choices=(tiling.layout,),
            conditions=(condition,),
        )

    assert str(excinfo.value) == (
        f"the condition's target {ghost!r} is not a decision of the space"
    )


def test_condition_naming_an_unknown_decision_is_refused() -> None:
    """Test a condition that names something that is no decision is refused."""
    tiling = build_tiling_space()
    ghost = Identifier("ghost")
    condition = Condition(tiling.unroll.name, (InSetConstraint(ghost, {1}),))

    with pytest.raises(SearchSpaceError) as excinfo:
        Space(
            variables=(tiling.unroll,),
            choices=(tiling.layout,),
            conditions=(condition,),
        )

    assert str(excinfo.value) == f"{ghost!r} is not a decision of the space"


def test_condition_naming_no_decision_is_refused() -> None:
    """Test a condition whose constraints name nothing is refused, not run."""
    tiling = build_tiling_space()
    closed = EquationConstraint(LiteralExpression(5).equals(LiteralExpression(5)))
    condition = Condition(tiling.unroll.name, (closed,))

    with pytest.raises(SearchSpaceError) as excinfo:
        Space(
            variables=(tiling.unroll,),
            choices=(tiling.layout,),
            conditions=(condition,),
        )

    assert str(excinfo.value) == (
        f"the condition on {tiling.unroll.name!r} names no decision"
    )


def test_set_constraint_naming_no_alternative_of_its_choice_is_refused() -> None:
    """Test a misspelled alternative in a condition on a choice is refused."""
    tiling = build_tiling_space()
    condition = Condition(
        tiling.unroll.name,
        (InSetConstraint(tiling.layout.name, {Identifier("bogus")}),),
    )

    with pytest.raises(SearchSpaceError) as excinfo:
        Space(
            variables=(tiling.unroll,),
            choices=(tiling.layout,),
            conditions=(condition,),
        )

    assert str(excinfo.value) == (
        f"the choice {tiling.layout.name!r} has no alternative bogus"
    )


def test_equation_naming_a_choice_is_refused() -> None:
    """Test an equation over a choice is refused when the space is built."""
    tiling = build_tiling_space()
    equation = EquationConstraint(
        IdentifierExpression(tiling.layout.name).equals(LiteralExpression(1))
    )
    condition = Condition(tiling.unroll.name, (equation,))

    with pytest.raises(SearchSpaceError) as excinfo:
        Space(
            variables=(tiling.unroll,),
            choices=(tiling.layout,),
            conditions=(condition,),
        )

    assert str(excinfo.value) == (
        f"an equation names the choice {tiling.layout.name!r}, which conditions "
        "and forbidden clauses name only in set constraints"
    )


def test_condition_naming_its_subtree_is_refused() -> None:
    """Test a condition on a choice naming a variable under it is refused."""
    tiling = build_tiling_space()
    condition = Condition(tiling.layout.name, (InSetConstraint(tiling.tile.name, {4}),))

    with pytest.raises(SearchSpaceError) as excinfo:
        Space(
            variables=(tiling.unroll,),
            choices=(tiling.layout,),
            conditions=(condition,),
        )

    assert str(excinfo.value) == (
        f"the condition on {tiling.layout.name!r} names {tiling.tile.name!r}, "
        "which is the target or under it"
    )


def test_forbidden_clause_naming_nothing_is_refused() -> None:
    """Test a forbidden clause with no reference is refused by its index."""
    tiling = build_tiling_space()
    ground = EquationConstraint(LiteralExpression(True))

    with pytest.raises(SearchSpaceError) as excinfo:
        Space(
            variables=(tiling.unroll,),
            choices=(tiling.layout,),
            forbidden=(
                Forbidden((InSetConstraint(tiling.unroll.name, {4}),)),
                Forbidden((ground,)),
            ),
        )

    assert str(excinfo.value) == "the forbidden clause 1 names no decision"


def test_conditions_depending_on_each_other_are_refused() -> None:
    """Test two decisions conditioned on each other are a cycle, listed in order."""
    first, second = make_variable("first"), make_variable("second")
    conditions = (
        Condition(second.name, (InSetConstraint(first.name, {1}),)),
        Condition(first.name, (InSetConstraint(second.name, {1}),)),
    )

    with pytest.raises(SearchSpaceError) as excinfo:
        Space(variables=(first, second), conditions=conditions)

    assert str(excinfo.value) == (
        f"the decisions {first.name!r}, {second.name!r} depend on each other in a cycle"
    )


# ===========================================================================
# Decisions, orders and conditions
# ===========================================================================


def test_space_lists_its_decisions_in_canonical_order() -> None:
    """Test the decisions are top-level variables, then choices, depth first."""
    tiling = build_tiling_space()

    decisions = tiling.space.decisions

    assert decisions == (tiling.unroll, tiling.layout, tiling.tile)
    assert decisions[2] is tiling.tile


def test_space_finds_a_decision_by_name() -> None:
    """Test `decision` returns the decision object, and `None` for another name."""
    tiling = build_tiling_space()

    assert tiling.space.decision(tiling.tile.name) is tiling.tile
    assert tiling.space.decision(tiling.layout.name) is tiling.layout
    assert tiling.space.decision(tiling.tiled.name) is None
    assert tiling.space.decision(Identifier("ghost")) is None


def test_space_decision_refuses_a_name_that_is_no_identifier() -> None:
    """Test `decision` refuses a `str`."""
    with pytest.raises(TypeError):
        build_tiling_space().space.decision("tile")  # type: ignore[arg-type]


def test_decision_order_puts_a_decision_after_what_it_depends_on() -> None:
    """Test a condition moves its target after the decision it names."""
    tiling = build_tiling_space(unroll_on_tiled_only=True)

    order = tiling.space.decision_order

    assert order == (tiling.layout.name, tiling.unroll.name, tiling.tile.name)
    assert order[0] is tiling.layout.name


def test_decision_order_without_conditions_is_canonical() -> None:
    """Test with no condition the decision order is the canonical one."""
    tiling = build_tiling_space()

    assert tiling.space.decision_order == (
        tiling.unroll.name,
        tiling.layout.name,
        tiling.tile.name,
    )


def test_space_keeps_a_lone_condition_object() -> None:
    """Test a target's one condition is returned as the object given."""
    tiling = build_tiling_space(unroll_on_tiled_only=True)

    assert tiling.space.conditions == tiling.conditions
    assert tiling.space.conditions[0] is tiling.conditions[0]


def test_space_merges_conditions_on_one_target() -> None:
    """Test two conditions on one target become one holding both member objects."""
    tiling = build_tiling_space()
    on_tiled = InSetConstraint(tiling.layout.name, {tiling.tiled.name})
    not_flat = NotInSetConstraint(tiling.layout.name, {tiling.flat.name})

    space = Space(
        variables=(tiling.unroll,),
        choices=(tiling.layout,),
        conditions=(
            Condition(tiling.unroll.name, (on_tiled,)),
            Condition(tiling.unroll.name, (not_flat,)),
        ),
    )

    (merged,) = space.conditions
    assert merged.target == tiling.unroll.name
    assert {id(member) for member in merged.when.constraints} == {
        id(on_tiled),
        id(not_flat),
    }


# ===========================================================================
# Configurations: activity and completeness
# ===========================================================================


def test_empty_configuration_is_valid_and_incomplete() -> None:
    """Test a configuration with no entries constructs and is incomplete.

    Ported from MOGA-VM
    `test_selection_validation.py::test_decision_point_without_selection_is_unconstrained`.
    """
    tiling = build_tiling_space()

    configuration = Configuration(tiling.space)

    assert configuration.entries == ()
    assert not configuration.is_complete()
    assert configuration.alternative(tiling.layout.name) is None
    assert configuration.activity(tiling.unroll.name) is Activity.ACTIVE
    assert configuration.activity(tiling.layout.name) is Activity.ACTIVE
    assert configuration.activity(tiling.tile.name) is Activity.PENDING


def test_complete_configuration_is_valid_and_complete() -> None:
    """Test a configuration assigning every active decision is complete.

    Ported from MOGA-VM
    `test_selection_validation.py::test_decision_point_accepts_fully_covered_selection`.
    """
    tiling = build_tiling_space()

    configuration = Configuration(
        tiling.space,
        {
            tiling.unroll.name: 2,
            tiling.layout.name: tiling.tiled.name,
            tiling.tile.name: 4,
        },
    )

    assert configuration.is_complete()
    assert configuration.alternative(tiling.layout.name) is tiling.tiled
    assert configuration.value(tiling.tile.name) == 4
    assert configuration.activity(tiling.tile.name) is Activity.ACTIVE


def test_partial_configuration_is_valid_and_incomplete() -> None:
    """Test assigning some of an alternative's variables is valid and partial.

    Ported from MOGA-VM
    `test_selection_validation.py::test_decision_point_accepts_partial_status_with_some_coverage`.
    """
    first, second = make_variable("k0"), make_variable("k1")
    option = make_alternative("opt", (first, second))
    choice = make_choice("s", option)
    space = Space(choices=(choice,))

    configuration = Configuration(space, {choice.name: option.name, first.name: 1})

    assert not configuration.is_complete()
    assert configuration.value(first.name) == 1
    assert configuration.value(second.name) is None
    assert configuration.activity(second.name) is Activity.ACTIVE


def test_chosen_alternative_with_unassigned_variables_is_valid_and_incomplete() -> None:
    """Test choosing an alternative and leaving one variable unassigned is valid.

    Ported from MOGA-VM
    `test_selection_validation.py::test_decision_point_accepts_selected_status_with_incomplete_coverage`.
    """
    tiling = build_tiling_space()

    configuration = Configuration(
        tiling.space, {tiling.unroll.name: 1, tiling.layout.name: tiling.tiled.name}
    )

    assert not configuration.is_complete()
    assert configuration.value(tiling.tile.name) is None
    assert configuration.alternative(tiling.layout.name) is tiling.tiled


def test_choosing_the_empty_alternative_completes_the_configuration() -> None:
    """Test the decisions under an unchosen alternative are inactive."""
    tiling = build_tiling_space()

    configuration = Configuration(
        tiling.space, {tiling.unroll.name: 1, tiling.layout.name: tiling.flat.name}
    )

    assert configuration.activity(tiling.tile.name) is Activity.INACTIVE
    assert configuration.is_complete()


def test_condition_makes_its_target_inactive_when_violated() -> None:
    """Test `unroll` is inactive once `flat` is chosen, and complete without it."""
    tiling = build_tiling_space(unroll_on_tiled_only=True)

    configuration = Configuration(tiling.space, {tiling.layout.name: tiling.flat.name})

    assert configuration.activity(tiling.unroll.name) is Activity.INACTIVE
    assert configuration.is_complete()


def test_condition_on_an_unassigned_choice_leaves_its_target_pending() -> None:
    """Test a condition reading an unassigned choice leaves its target pending."""
    tiling = build_tiling_space(unroll_on_tiled_only=True)

    configuration = Configuration(tiling.space)

    assert configuration.activity(tiling.unroll.name) is Activity.PENDING


def test_condition_holding_makes_its_target_active() -> None:
    """Test `unroll` is active once `tiled` is chosen."""
    tiling = build_tiling_space(unroll_on_tiled_only=True)

    configuration = Configuration(tiling.space, {tiling.layout.name: tiling.tiled.name})

    assert configuration.activity(tiling.unroll.name) is Activity.ACTIVE
    assert not configuration.is_complete()


def test_nested_choice_follows_its_alternative() -> None:
    """Test a sub-choice is pending under an unassigned choice, then decides."""
    inner_variable = make_variable("depth")
    deep = make_alternative("deep", (inner_variable,))
    inner = make_choice("inner", deep, make_alternative("shallow"))
    nested = make_alternative("nested", choices=(inner,))
    outer = make_choice("outer", nested, make_alternative("none"))
    space = Space(choices=(outer,))

    unassigned = Configuration(space)
    chosen = Configuration(space, {outer.name: nested.name, inner.name: deep.name})
    other = Configuration(space, {outer.name: outer.alternatives[1].name})

    assert unassigned.activity(inner.name) is Activity.PENDING
    assert unassigned.activity(inner_variable.name) is Activity.PENDING
    assert chosen.activity(inner_variable.name) is Activity.ACTIVE
    assert not chosen.is_complete()
    assert other.activity(inner.name) is Activity.INACTIVE
    assert other.activity(inner_variable.name) is Activity.INACTIVE
    assert other.is_complete()


def test_activity_of_a_name_the_space_lacks_is_none() -> None:
    """Test `activity` answers `None` for an alternative's name or a stranger."""
    tiling = build_tiling_space()
    configuration = Configuration(tiling.space)

    assert configuration.activity(tiling.tiled.name) is None
    assert configuration.activity(Identifier("ghost")) is None


def test_activity_members_are_its_values() -> None:
    """Test `Activity` is a string enum of the three activities."""
    assert [member.value for member in Activity] == ["active", "inactive", "pending"]


# ===========================================================================
# Configurations: refusals
# ===========================================================================


def _problems(build: Callable[[], Any]) -> ConfigurationError:
    """Return the `ConfigurationError` `build` raises."""
    with pytest.raises(ConfigurationError) as excinfo:
        build()
    return excinfo.value


def test_unknown_alternative_is_refused_naming_it() -> None:
    """Test a choice's value naming no alternative is refused, naming both.

    Ported from MOGA-VM
    `test_selection_validation.py::test_decision_point_rejects_unknown_selected_option`;
    the message names the choice by its `repr` and the value by its name.
    """
    tiling = build_tiling_space()
    ghost = Identifier("ghost")

    error = _problems(lambda: Configuration(tiling.space, {tiling.layout.name: ghost}))

    assert error.problems == (
        f"the choice {tiling.layout.name!r} has no alternative ghost",
    )
    assert str(error) == (
        "the configuration is invalid: "
        f"the choice {tiling.layout.name!r} has no alternative ghost"
    )


def test_choice_value_that_is_a_string_is_an_unknown_alternative() -> None:
    """Test a choice's value is an `Identifier`: its name hint does not do."""
    tiling = build_tiling_space()

    error = _problems(
        lambda: Configuration(tiling.space, {tiling.layout.name: "tiled"})
    )

    (problem,) = error.problems
    assert problem.startswith(f"the choice {tiling.layout.name!r} has no alternative")
    assert "tiled" in problem


def test_value_for_an_inactive_variable_is_refused() -> None:
    """Test a value for a variable of an unchosen alternative is refused.

    Ported from MOGA-VM
    `test_selection_validation.py::test_decision_point_rejects_assignment_to_unknown_knob`,
    whose stray knob is now a variable of another alternative.
    """
    tiling = build_tiling_space()

    error = _problems(
        lambda: Configuration(
            tiling.space,
            {tiling.layout.name: tiling.flat.name, tiling.tile.name: 4},
        )
    )

    assert error.problems == (
        f"the decision {tiling.tile.name!r} is given a value but is not active",
    )
    assert repr(tiling.unroll.name) not in str(error)


def test_values_under_an_unchosen_choice_are_refused() -> None:
    """Test values for a choice's variables without a choice are refused.

    Ported from MOGA-VM
    `test_selection_validation.py::test_decision_point_rejects_unselected_status_with_assignments`:
    an unassigned choice leaves its variables pending, and a pending
    decision takes no value.
    """
    tiling = build_tiling_space()

    error = _problems(lambda: Configuration(tiling.space, {tiling.tile.name: 4}))

    assert error.problems == (
        f"the decision {tiling.tile.name!r} is given a value but is not active",
    )


def test_unknown_decision_is_refused() -> None:
    """Test an entry naming no decision is refused."""
    tiling = build_tiling_space()
    ghost = Identifier("ghost")

    error = _problems(lambda: Configuration(tiling.space, {ghost: 1}))

    assert error.problems == (f"{ghost!r} is not a decision of the space",)


def test_duplicate_entry_is_refused() -> None:
    """Test two pairs naming one decision are refused."""
    tiling = build_tiling_space()

    error = _problems(
        lambda: Configuration(
            tiling.space, [(tiling.unroll.name, 1), (tiling.unroll.name, 2)]
        )
    )

    assert error.problems == (
        f"the decision {tiling.unroll.name!r} is given more than one value",
    )


def test_inadmissible_value_is_refused() -> None:
    """Test a value outside the variable's domain is refused."""
    tiling = build_tiling_space()

    error = _problems(lambda: Configuration(tiling.space, {tiling.unroll.name: 3}))

    assert error.problems == (
        f"the value of the variable {tiling.unroll.name!r} cannot be assigned "
        "to its param",
    )


def test_value_of_another_type_is_refused() -> None:
    """Test values are type-strict: `True` is not the category `1`."""
    tiling = build_tiling_space()

    error = _problems(lambda: Configuration(tiling.space, {tiling.unroll.name: True}))

    assert error.problems == (
        f"the value of the variable {tiling.unroll.name!r} cannot be assigned "
        "to its param",
    )


def test_forbidden_combination_is_refused() -> None:
    """Test a configuration taking a forbidden combination is refused by index."""
    tiling = build_tiling_space(forbid_unroll_four=True)

    error = _problems(lambda: Configuration(tiling.space, {tiling.unroll.name: 4}))

    assert error.problems == ("the configuration takes the forbidden combination 0",)


def test_forbidden_clause_with_an_unassigned_reference_does_not_apply() -> None:
    """Test a clause naming an unassigned decision is pending, not violated.

    The clause `unroll in {4} and tile in {4}` forbids nothing until `tile`
    has a value, and forbids its combination once it has.
    """
    tiling = build_tiling_space()
    space = Space(
        variables=(tiling.unroll,),
        choices=(tiling.layout,),
        forbidden=(
            Forbidden(
                (
                    InSetConstraint(tiling.unroll.name, {4}),
                    InSetConstraint(tiling.tile.name, {4}),
                )
            ),
        ),
    )
    pending = {tiling.unroll.name: 4, tiling.layout.name: tiling.tiled.name}

    configuration = Configuration(space, pending)

    assert configuration.value(tiling.unroll.name) == 4
    error = _problems(lambda: configuration.with_entry(tiling.tile.name, 4))
    assert error.problems == ("the configuration takes the forbidden combination 0",)


def test_forbidden_clause_over_an_inactive_decision_does_not_apply() -> None:
    """Test a clause naming an inactive decision never forbids.

    `unroll in {1} and tile not in {8}` would hold if an inactive `tile`
    counted as having no forbidden value; the clause does not apply.
    """
    tiling = build_tiling_space()
    space = Space(
        variables=(tiling.unroll,),
        choices=(tiling.layout,),
        forbidden=(
            Forbidden(
                (
                    InSetConstraint(tiling.unroll.name, {1}),
                    NotInSetConstraint(tiling.tile.name, {8}),
                )
            ),
        ),
    )

    configuration = Configuration(
        space, {tiling.unroll.name: 1, tiling.layout.name: tiling.flat.name}
    )

    assert configuration.activity(tiling.tile.name) is Activity.INACTIVE
    assert configuration.is_complete()


def test_undecided_condition_is_refused() -> None:
    """Test a condition that evaluates to undecided is a problem of its target."""
    word = make_variable("word", "a", "b")
    gated = make_variable("gated")
    space = Space(
        variables=(word, gated),
        conditions=(Condition(gated.name, (GoldenEven(word.name),)),),
    )

    error = _problems(lambda: Configuration(space, {word.name: "a"}))

    assert error.problems == (f"the condition on {gated.name!r} could not be decided",)


def test_undecided_forbidden_clause_is_refused() -> None:
    """Test a forbidden clause that evaluates to undecided is refused by index."""
    word = make_variable("word", "a", "b")
    space = Space(variables=(word,), forbidden=(Forbidden((GoldenEven(word.name),)),))

    error = _problems(lambda: Configuration(space, {word.name: "a"}))

    assert error.problems == ("the forbidden clause 0 could not be decided",)


def test_failing_condition_raises_the_constraint_exception() -> None:
    """Test a Python-defined constraint's exception propagates as itself."""
    source, gated = make_variable("source"), make_variable("gated")
    explosion = Explosion("condition")
    EXPLOSIONS["condition"] = explosion
    space = Space(
        variables=(source, gated),
        conditions=(
            Condition(gated.name, (ExplodingConstraint(source.name, "condition"),)),
        ),
    )

    with pytest.raises(Explosion) as excinfo:
        Configuration(space, {source.name: 1})

    assert excinfo.value is explosion


def test_failing_forbidden_clause_raises_the_constraint_exception() -> None:
    """Test a raising forbidden clause's exception propagates as itself."""
    source = make_variable("source")
    explosion = Explosion("forbidden")
    EXPLOSIONS["forbidden"] = explosion
    space = Space(
        variables=(source,),
        forbidden=(Forbidden((ExplodingConstraint(source.name, "forbidden"),)),),
    )

    with pytest.raises(Explosion) as excinfo:
        Configuration(space, {source.name: 1})

    assert excinfo.value is explosion


def test_every_problem_is_reported_in_order() -> None:
    """Test the entries' problems come first, then the decisions', in order."""
    tiling = build_tiling_space(forbid_unroll_four=True)
    ghost = Identifier("ghost")

    error = _problems(
        lambda: Configuration(
            tiling.space,
            [
                (ghost, 1),
                (tiling.unroll.name, 3),
                (tiling.layout.name, tiling.flat.name),
                (tiling.tile.name, 4),
            ],
        )
    )

    assert error.problems == (
        f"{ghost!r} is not a decision of the space",
        f"the value of the variable {tiling.unroll.name!r} cannot be assigned "
        "to its param",
        f"the decision {tiling.tile.name!r} is given a value but is not active",
    )


# ===========================================================================
# Configurations: values and updates
# ===========================================================================


def test_configuration_keeps_its_space_and_value_objects() -> None:
    """Test a configuration returns its space and the value objects given."""
    big = 10**30
    wide = make_variable("wide", big)
    space = Space(variables=(wide,))
    value = int(str(big))
    assert value is not big

    configuration = Configuration(space, {wide.name: value})

    assert configuration.space is space
    assert configuration.value(wide.name) is value
    assert configuration.entries[0][1] is value


def test_configuration_returns_the_alternative_name_given() -> None:
    """Test a choice's value is the name object given."""
    tiling = build_tiling_space()

    configuration = Configuration(tiling.space, {tiling.layout.name: tiling.tiled.name})

    assert configuration.value(tiling.layout.name) is tiling.tiled.name


def test_configuration_reads_entries_from_a_generator() -> None:
    """Test entries may be any iterable of pairs, read once."""
    tiling = build_tiling_space()
    pairs = ((name, value) for name, value in [(tiling.unroll.name, 2)])

    configuration = Configuration(tiling.space, pairs)

    assert configuration.value(tiling.unroll.name) == 2


def test_configuration_entries_follow_canonical_order() -> None:
    """Test entries are in canonical order, named by the decisions' own names."""
    tiling = build_tiling_space()

    configuration = Configuration(
        tiling.space,
        [
            (tiling.tile.name, 4),
            (tiling.layout.name, tiling.tiled.name),
            (tiling.unroll.name, 1),
        ],
    )

    assert configuration.entries == (
        (tiling.unroll.name, 1),
        (tiling.layout.name, tiling.tiled.name),
        (tiling.tile.name, 4),
    )
    assert configuration.entries[2][0] is tiling.tile.name


def test_value_of_an_unassigned_or_unknown_name_is_none() -> None:
    """Test `value` answers `None` for an unassigned decision or a stranger."""
    tiling = build_tiling_space()
    configuration = Configuration(tiling.space, {tiling.unroll.name: 1})

    assert configuration.value(tiling.tile.name) is None
    assert configuration.value(Identifier("ghost")) is None


def test_lookups_refuse_names_that_are_no_identifiers() -> None:
    """Test `value`, `alternative` and `activity` refuse a `str`."""
    configuration = Configuration(build_tiling_space().space)

    for lookup in (
        configuration.value,
        configuration.alternative,
        configuration.activity,
    ):
        with pytest.raises(TypeError):
            lookup("unroll")  # type: ignore[arg-type]


def test_with_entry_adds_a_value_and_keeps_the_others() -> None:
    """Test `with_entry` returns a new configuration with one more value."""
    tiling = build_tiling_space()
    original = Configuration(tiling.space, {tiling.layout.name: tiling.tiled.name})

    updated = original.with_entry(tiling.tile.name, 8)

    assert updated is not original
    assert updated.value(tiling.tile.name) == 8
    assert updated.value(tiling.layout.name) is tiling.tiled.name
    assert original.value(tiling.tile.name) is None
    assert updated.space is tiling.space


def test_with_entry_replaces_a_value() -> None:
    """Test `with_entry` on an assigned decision replaces its value."""
    tiling = build_tiling_space()
    original = Configuration(tiling.space, {tiling.unroll.name: 1})

    updated = original.with_entry(tiling.unroll.name, 4)

    assert updated.value(tiling.unroll.name) == 4
    assert original.value(tiling.unroll.name) == 1


def test_with_entry_checks_the_whole_configuration() -> None:
    """Test switching a choice away strands its variable's value, refused."""
    tiling = build_tiling_space()
    original = Configuration(
        tiling.space, {tiling.layout.name: tiling.tiled.name, tiling.tile.name: 4}
    )

    error = _problems(lambda: original.with_entry(tiling.layout.name, tiling.flat.name))

    assert error.problems == (
        f"the decision {tiling.tile.name!r} is given a value but is not active",
    )


def test_with_entries_adds_several_values() -> None:
    """Test `with_entries` adds a choice and its variable together."""
    tiling = build_tiling_space()
    original = Configuration(tiling.space, {tiling.unroll.name: 1})

    updated = original.with_entries(
        {tiling.layout.name: tiling.tiled.name, tiling.tile.name: 8}
    )

    assert updated.is_complete()
    assert updated.value(tiling.unroll.name) == 1


def test_with_entries_refuses_a_duplicate_pair() -> None:
    """Test `with_entries` given one decision twice is refused."""
    tiling = build_tiling_space()
    original = Configuration(tiling.space)

    error = _problems(
        lambda: original.with_entries(
            [(tiling.unroll.name, 1), (tiling.unroll.name, 2)]
        )
    )

    assert error.problems == (
        f"the decision {tiling.unroll.name!r} is given more than one value",
    )


def test_without_entry_removes_a_value_and_keeps_the_others() -> None:
    """Test `without_entry` returns a new configuration with one value fewer."""
    tiling = build_tiling_space()
    original = build_complete_configuration(tiling)

    updated = original.without_entry(tiling.unroll.name)

    assert updated is not original
    assert updated.value(tiling.unroll.name) is None
    assert updated.value(tiling.tile.name) == 4
    assert original.value(tiling.unroll.name) == 2
    assert updated.space is tiling.space
    assert (
        updated.key()
        == Configuration(
            tiling.space, {tiling.layout.name: tiling.tiled.name, tiling.tile.name: 4}
        ).key()
    )


def test_without_entries_then_with_entry_switches_an_alternative() -> None:
    """Test dropping an alternative's variables lets its choice switch away."""
    tiling = build_tiling_space()
    original = build_complete_configuration(tiling)

    switched = original.without_entries([tiling.tile.name]).with_entry(
        tiling.layout.name, tiling.flat.name
    )

    assert switched.value(tiling.layout.name) is tiling.flat.name
    assert switched.value(tiling.tile.name) is None
    assert switched.is_complete()
    assert (
        switched.key()
        == Configuration(
            tiling.space, {tiling.unroll.name: 2, tiling.layout.name: tiling.flat.name}
        ).key()
    )


def test_without_entry_of_an_unassigned_decision_changes_nothing() -> None:
    """Test removing the value of a decision that holds none is a no-op."""
    tiling = build_tiling_space()
    original = Configuration(tiling.space, {tiling.unroll.name: 2})

    same = original.without_entries([tiling.layout.name, tiling.tile.name])

    assert same.entries == original.entries
    assert same.key() == original.key()


def test_without_entry_refuses_a_name_that_is_no_decision() -> None:
    """Test removing a name the space lacks is an unknown decision."""
    tiling = build_tiling_space()
    original = Configuration(tiling.space, {tiling.unroll.name: 2})
    ghost = Identifier("ghost")

    error = _problems(lambda: original.without_entry(ghost))

    assert error.problems == (f"{ghost!r} is not a decision of the space",)


def test_without_entry_refuses_a_choice_whose_variables_hold_values() -> None:
    """Test unassigning a choice strands its alternative's variable, refused."""
    tiling = build_tiling_space()
    original = build_complete_configuration(tiling)

    error = _problems(lambda: original.without_entry(tiling.layout.name))

    assert error.problems == (
        f"the decision {tiling.tile.name!r} is given a value but is not active",
    )


def test_without_entries_refuses_a_name_that_is_no_identifier() -> None:
    """Test a name to remove must be an `Identifier`."""
    tiling = build_tiling_space()
    original = Configuration(tiling.space, {tiling.unroll.name: 2})

    with pytest.raises(TypeError):
        original.without_entries(["unroll"])  # type: ignore[list-item]
    with pytest.raises(TypeError):
        original.without_entry("unroll")  # type: ignore[arg-type]


# ===========================================================================
# Keys
# ===========================================================================


def test_configurations_of_relabeled_spaces_have_equal_keys() -> None:
    """Test corresponding configurations of two relabeled spaces share a key.

    Ported from MOGA-VM
    `test_alpha_standalone.py::test_selection_alpha_equivalent_when_space_labels_seeded`
    (divergence configuration-compared-with-its-space: a configuration is
    compared with its space, and its key is renaming-invariant without a
    seeded renaming).
    """
    left, right = (
        build_complete_configuration(build_tiling_space()),
        build_complete_configuration(build_tiling_space()),
    )

    assert isinstance(left.key(), ConfigurationKey)
    assert left.key() == right.key()
    assert hash(left.key()) == hash(right.key())
    assert {left.key(): "measured"}[right.key()] == "measured"


def test_different_values_give_different_keys() -> None:
    """Test a different value of one variable changes the key.

    Ported from MOGA-VM
    `test_alpha_standalone.py::test_selection_not_alpha_equivalent_when_knob_value_differs_under_seeding`.
    """
    tiling = build_tiling_space()

    assert (
        build_complete_configuration(tiling, 4).key()
        != build_complete_configuration(tiling, 8).key()
    )


def test_complete_and_incomplete_configurations_have_different_keys() -> None:
    """Test leaving a variable unassigned changes the key.

    Ported from MOGA-VM
    `test_alpha_standalone.py::test_selection_not_alpha_equivalent_when_status_differs_under_seeding`
    (divergence: the status is replaced by completeness).
    """
    tiling = build_tiling_space()
    complete = build_complete_configuration(tiling)
    incomplete = Configuration(
        tiling.space, {tiling.unroll.name: 2, tiling.layout.name: tiling.tiled.name}
    )

    assert complete.key() != incomplete.key()


def test_assigning_a_choice_changes_the_configuration() -> None:
    """Test a configuration with a choice made differs from one without, both ways.

    Ported from MOGA-VM
    `test_structural_equivalence.py::test_decision_point_structural_equivalence_symmetric_for_selection_presence`.
    """
    tiling = build_tiling_space()
    chosen = Configuration(tiling.space, {tiling.layout.name: tiling.flat.name})
    unchosen = Configuration(tiling.space)

    assert chosen.key() != unchosen.key()
    assert not chosen.is_structurally_equivalent(unchosen)
    assert not unchosen.is_structurally_equivalent(chosen)


def test_keys_of_choices_differ_by_the_alternative() -> None:
    """Test choosing another alternative changes the key."""
    tiling = build_tiling_space()

    tiled = Configuration(tiling.space, {tiling.layout.name: tiling.tiled.name})
    flat = Configuration(tiling.space, {tiling.layout.name: tiling.flat.name})

    assert tiled.key() != flat.key()


# ===========================================================================
# Equivalence
# ===========================================================================


def test_variables_with_distinct_names_are_alpha_equivalent_standalone() -> None:
    """Test two matching variables with distinct names are alpha-equivalent.

    Ported from MOGA-VM
    `test_alpha_standalone.py::test_knob_alpha_equivalent_standalone_with_distinct_names`.
    """
    left, right = make_variable("k", 1), make_variable("k", 1)

    assert left.name != right.name
    assert not left.is_structurally_equivalent(right)
    assert left.is_alpha_equivalent(right)
    assert right.is_alpha_equivalent(left)


def test_variables_over_different_domains_are_not_alpha_equivalent() -> None:
    """Test standalone variables over different domains are not alpha-equivalent.

    Ported from MOGA-VM
    `test_alpha_standalone.py::test_knob_not_alpha_equivalent_when_param_domain_differs_standalone`.
    """
    left, right = make_variable("k", 1), make_variable("k", 2)

    assert not left.is_alpha_equivalent(right)
    assert not right.is_alpha_equivalent(left)


def test_alternatives_with_distinct_labels_are_alpha_equivalent_standalone() -> None:
    """Test two matching alternatives with distinct labels are alpha-equivalent.

    Ported from MOGA-VM
    `test_alpha_standalone.py::test_array_option_alpha_equivalent_standalone_with_distinct_names`
    (a plain alternative in place of `RealizationOption`).
    """
    left = make_alternative("left_opt", (make_variable("left_knob", 1),))
    right = make_alternative("right_opt", (make_variable("right_knob", 1),))

    assert not left.is_structurally_equivalent(right)
    assert left.is_alpha_equivalent(right)
    assert right.is_alpha_equivalent(left)


def test_alternatives_with_different_variable_domains_are_not_alpha_equivalent() -> (
    None
):
    """Test standalone alternatives differing in a variable's domain differ.

    Ported from MOGA-VM
    `test_alpha_standalone.py::test_array_option_not_alpha_equivalent_when_knob_param_differs_standalone`.
    """
    left = make_alternative("left_opt", (make_variable("k", 1),))
    right = make_alternative("right_opt", (make_variable("k", 2),))

    assert not left.is_alpha_equivalent(right)
    assert not right.is_alpha_equivalent(left)


def test_choices_with_distinct_labels_are_alpha_equivalent_standalone() -> None:
    """Test two matching choices with distinct labels are alpha-equivalent.

    Ported from MOGA-VM
    `test_alpha_standalone.py::test_array_decision_space_alpha_equivalent_standalone_with_distinct_names`.
    """

    def build(prefix: str) -> Choice:
        knob = make_variable(f"{prefix}_knob", 1)
        return make_choice(
            f"{prefix}_space", make_alternative(f"{prefix}_opt", (knob,))
        )

    left, right = build("left"), build("right")

    assert not left.is_structurally_equivalent(right)
    assert left.is_alpha_equivalent(right)
    assert right.is_alpha_equivalent(left)


def test_choices_with_different_variable_domains_are_not_alpha_equivalent() -> None:
    """Test standalone choices differing in a nested domain are not equivalent.

    Ported from MOGA-VM
    `test_alpha_standalone.py::test_array_decision_space_not_alpha_equivalent_when_option_param_differs`.
    """
    left = make_choice("left", make_alternative("opt", (make_variable("k", 1),)))
    right = make_choice("right", make_alternative("opt", (make_variable("k", 2),)))

    assert not left.is_alpha_equivalent(right)
    assert not right.is_alpha_equivalent(left)


def test_alternatives_differing_only_in_a_domain_are_not_alpha_equivalent() -> None:
    """Test a domain alone discriminates, with a positive control.

    Ported from MOGA-VM
    `test_alpha_standalone.py::test_options_not_alpha_equivalent_when_param_domains_differ`.
    """

    def build(*values: int) -> Alternative:
        return make_alternative("o", (make_variable("k", *values),))

    left, right, matching = build(1), build(2), build(1)

    assert not left.is_alpha_equivalent(right)
    assert not right.is_alpha_equivalent(left)
    assert left.is_alpha_equivalent(matching)


def test_configurations_of_relabeled_spaces_are_alpha_equivalent() -> None:
    """Test corresponding configurations of relabeled spaces are alpha-equivalent.

    Ported from MOGA-VM
    `test_alpha_standalone.py::test_selection_alpha_equivalent_when_space_labels_seeded`,
    both directions.
    """
    left, right = (
        build_complete_configuration(build_tiling_space()),
        build_complete_configuration(build_tiling_space()),
    )

    assert not left.is_structurally_equivalent(right)
    assert left.is_alpha_equivalent(right)
    assert right.is_alpha_equivalent(left)


def test_configurations_of_unrelated_spaces_are_not_alpha_equivalent() -> None:
    """Test configurations of spaces of different shapes are not equivalent.

    Ported from MOGA-VM
    `test_alpha_standalone.py::test_selection_not_alpha_equivalent_without_seeding`
    (divergence configuration-compared-with-its-space: a configuration's
    names resolve through its own space, so what fails to correspond is the
    spaces).
    """
    tiling = build_tiling_space()
    other_choice = make_choice("other", make_alternative("only"))
    other = Configuration(
        Space(choices=(other_choice,)),
        {other_choice.name: other_choice.alternatives[0].name},
    )
    mine = Configuration(tiling.space, {tiling.layout.name: tiling.flat.name})

    assert not mine.is_alpha_equivalent(other)
    assert not other.is_alpha_equivalent(mine)


def test_configurations_with_different_values_are_not_alpha_equivalent() -> None:
    """Test a different value breaks alpha equivalence of relabeled configurations.

    Ported from MOGA-VM
    `test_alpha_standalone.py::test_selection_not_alpha_equivalent_when_knob_value_differs_under_seeding`.
    """
    left, right = (
        build_complete_configuration(build_tiling_space(), 4),
        build_complete_configuration(build_tiling_space(), 8),
    )

    assert not left.is_alpha_equivalent(right)
    assert not right.is_alpha_equivalent(left)


def test_bounded_variables_whose_names_are_renamed_are_alpha_equivalent_under_it() -> (
    None
):
    """Test variables over equal bounds correspond under a renaming of their names.

    Ported from MOGA-VM
    `test_alpha_standalone.py::test_address_knobs_alpha_equivalent_when_bounds_match_under_renaming`
    (`WordAddressKnob` stays in MOGA-VM; its param is an integer range).
    """
    left_name, right_name = Identifier("addr_port0"), Identifier("addr_port1")
    left = Variable(param=create_integer_param_between(0, 15), name=left_name)
    right = Variable(param=create_integer_param_between(0, 15), name=right_name)

    assert not left.is_structurally_equivalent(right)
    renaming = AlphaRenaming.empty().extend({left_name: right_name})
    assert left.is_alpha_equivalent_under(right, renaming)


def test_alpha_equivalence_under_a_renaming_pairing_other_names_still_binds() -> None:
    """Test an unrelated outer renaming does not stop a variable binding its name."""
    left, right = make_variable("k", 1), make_variable("k", 1)
    renaming = AlphaRenaming.empty().extend({Identifier("x"): Identifier("y")})

    assert left.is_alpha_equivalent_under(right, renaming)


def test_alpha_equivalence_refuses_a_renaming_of_another_type() -> None:
    """Test `is_alpha_equivalent_under` needs an `AlphaRenaming`."""
    variable = make_variable("k")

    with pytest.raises(TypeError, match="renaming must be an AlphaRenaming"):
        variable.is_alpha_equivalent_under(variable, {})  # type: ignore[arg-type]


def test_relabeled_spaces_are_alpha_but_not_structurally_equivalent() -> None:
    """Test two builds of one space differ by name only."""
    left = build_tiling_space(unroll_on_tiled_only=True, forbid_unroll_four=True).space
    right = build_tiling_space(unroll_on_tiled_only=True, forbid_unroll_four=True).space

    assert not left.is_structurally_equivalent(right)
    assert left.is_alpha_equivalent(right)
    assert right.is_alpha_equivalent(left)


def test_spaces_differing_in_a_condition_are_not_alpha_equivalent() -> None:
    """Test a condition counts in alpha equivalence."""
    left = build_tiling_space(unroll_on_tiled_only=True).space
    right = build_tiling_space().space

    assert not left.is_alpha_equivalent(right)
    assert not right.is_alpha_equivalent(left)


def test_spaces_differing_in_a_forbidden_clause_are_not_alpha_equivalent() -> None:
    """Test a forbidden clause counts in alpha equivalence."""
    left = build_tiling_space(forbid_unroll_four=True).space
    right = build_tiling_space().space

    assert not left.is_alpha_equivalent(right)


def test_identifier_categories_are_references_resolved_through_the_space() -> None:
    """Test a domain naming a bound identifier corresponds to its counterpart.

    A variable whose categories are the space's alternatives' names compares
    as references: equal under the space's pairing, unequal to a free name.
    """

    def build(free: Identifier | None = None) -> Space:
        tiled, flat = make_alternative("tiled"), make_alternative("flat")
        layout = make_choice("layout", tiled, flat)
        mirror = Variable(
            param=categorical(tiled.name, free or flat.name), name=Identifier("mirror")
        )
        return Space(variables=(mirror,), choices=(layout,))

    assert build().is_alpha_equivalent(build())
    assert not build().is_alpha_equivalent(build(Identifier("stranger")))


def test_equivalences_answer_false_for_another_class() -> None:
    """Test comparing with an object of another class is `False`, not an error."""
    tiling = build_tiling_space()

    assert not tiling.unroll.is_structurally_equivalent(tiling.layout)
    assert not tiling.layout.is_alpha_equivalent(tiling.tiled)
    assert not tiling.space.is_structurally_equivalent(3)
    assert not Configuration(tiling.space).is_alpha_equivalent(tiling.space)


def test_alternatives_sharing_all_parts_are_structurally_equivalent() -> None:
    """Test two distinct alternatives sharing name and variables are equivalent.

    Ported from MOGA-VM
    `test_structural_equivalence.py::test_options_structurally_equivalent_when_sharing_all_identifiers`.
    """
    name = Identifier("o")
    knob = make_variable("k")
    left = Alternative(variables=(knob,), name=name)
    right = Alternative(variables=(knob,), name=name)

    assert left is not right
    assert left.is_structurally_equivalent(right)
    assert right.is_structurally_equivalent(left)


def test_alternatives_with_different_names_are_not_structurally_equivalent() -> None:
    """Test names are compared by `==`: two `o`s of different ids differ.

    Ported from MOGA-VM
    `test_structural_equivalence.py::test_options_not_structurally_equivalent_when_name_differs`.
    """
    knob = make_variable("k")
    left = Alternative(variables=(knob,), name=Identifier("o"))
    renamed = Alternative(variables=(knob,), name=Identifier("o"))

    assert not left.is_structurally_equivalent(renamed)


@pytest.mark.parametrize("longer_first", [True, False], ids=["longer", "shorter"])
def test_choice_is_not_structurally_equivalent_to_a_prefix(longer_first: bool) -> None:
    """Test a choice and one holding a prefix of its alternatives differ.

    Ported from MOGA-VM
    `test_structural_equivalence.py::test_decision_space_not_equivalent_to_shorter_option_list`
    and `::test_decision_space_not_equivalent_to_longer_option_list`.
    """
    name = Identifier("s")
    first, second = make_alternative("a"), make_alternative("b")
    longer, shorter = Choice((first, second), name=name), Choice((first,), name=name)
    left, right = (longer, shorter) if longer_first else (shorter, longer)

    assert not left.is_structurally_equivalent(right)


@pytest.mark.parametrize("longer_first", [True, False], ids=["longer", "shorter"])
def test_alternative_is_not_structurally_equivalent_to_a_variable_prefix(
    longer_first: bool,
) -> None:
    """Test an alternative and one holding a prefix of its variables differ.

    Ported from MOGA-VM
    `test_structural_equivalence.py::test_option_not_equivalent_to_shorter_knob_list`
    and `::test_option_not_equivalent_to_longer_knob_list`.
    """
    name = Identifier("o")
    first, second = make_variable("k0"), make_variable("k1")
    longer = Alternative(variables=(first, second), name=name)
    shorter = Alternative(variables=(first,), name=name)
    left, right = (longer, shorter) if longer_first else (shorter, longer)

    assert not left.is_structurally_equivalent(right)


def test_variables_of_one_kind_sharing_parts_are_structurally_equivalent() -> None:
    """Test two plain variables sharing name and param are equivalent.

    Ported from MOGA-VM
    `test_structural_equivalence.py::test_same_knob_kind_structurally_equivalent_for_shared_name_and_param`
    (the subclass half is in `test_extension.py`).
    """
    name = Identifier("port0")
    param = categorical(Identifier("affine"))

    assert Variable(param=param, name=name).is_structurally_equivalent(
        Variable(param=param, name=name)
    )


# Each entry builds a fresh object and a sibling sharing its identifiers that
# differs in exactly one structurally meaningful field.
_PerturbedPair = tuple[Callable[[], Any], Callable[[Any], Any]]


def _widen_param(variable: Any) -> Any:
    return Variable(param=categorical(1, 2, 3), name=variable.name)


def _add_variable(alternative: Any) -> Any:
    return Alternative(
        variables=(*alternative.variables, make_variable("extra")),
        name=alternative.name,
    )


def _drop_alternative(choice: Any) -> Any:
    return Choice(choice.alternatives[:-1], name=choice.name)


def _add_condition(space: Any) -> Any:
    (layout,) = space.choices
    (unroll,) = space.variables
    return Space(
        variables=space.variables,
        choices=space.choices,
        conditions=(
            Condition(
                unroll.name,
                (InSetConstraint(layout.name, {layout.alternatives[0].name}),),
            ),
        ),
        name=space.name,
    )


def _choose(configuration: Any) -> Any:
    (layout,) = configuration.space.choices
    return configuration.with_entry(layout.name, layout.alternatives[1].name)


def _unchoose(configuration: Any) -> Any:
    return Configuration(configuration.space)


def _build_chosen() -> Configuration:
    tiling = build_tiling_space()
    return Configuration(tiling.space, {tiling.layout.name: tiling.flat.name})


_PERTURBED_PAIRS: list[_PerturbedPair] = [
    (lambda: make_variable("k"), _widen_param),
    (lambda: make_alternative("o"), _add_variable),
    (
        lambda: make_choice("s", make_alternative("a"), make_alternative("b")),
        _drop_alternative,
    ),
    (lambda: build_tiling_space().space, _add_condition),
    (lambda: Configuration(build_tiling_space().space), _choose),
    (_build_chosen, _unchoose),
]
_PERTURBED_IDS = [
    "variable",
    "empty-alternative",
    "choice",
    "space",
    "unchosen-configuration",
    "chosen-configuration",
]


@pytest.mark.parametrize(("build", "perturb"), _PERTURBED_PAIRS, ids=_PERTURBED_IDS)
def test_structural_equivalence_is_reflexive(
    build: Callable[[], Any], perturb: Callable[[Any], Any]
) -> None:
    """Test every search-space object is structurally equivalent to itself.

    Ported from MOGA-VM
    `test_structural_equivalence.py::test_structural_equivalence_is_reflexive`
    (the `metric` case is covered with the measurements; `knobbed-option` is in
    `test_extension.py`).
    """
    obj = build()

    assert obj.is_structurally_equivalent(obj)
    assert obj.is_alpha_equivalent(obj)


@pytest.mark.parametrize(("build", "perturb"), _PERTURBED_PAIRS, ids=_PERTURBED_IDS)
def test_structural_equivalence_discriminates_a_perturbed_field(
    build: Callable[[], Any], perturb: Callable[[Any], Any]
) -> None:
    """Test changing one structural field breaks equivalence, both ways.

    Ported from MOGA-VM
    `test_structural_equivalence.py::test_structural_equivalence_discriminates_a_perturbed_field`.
    """
    obj = build()
    perturbed = perturb(obj)

    assert not obj.is_structurally_equivalent(perturbed)
    assert not perturbed.is_structurally_equivalent(obj)


# ===========================================================================
# Tuple categories
# ===========================================================================


def _tile_space() -> tuple[Space, Variable[Any]]:
    """Return a space of one variable `tile` over the shapes `(4, 4)` and `(8, 8)`."""
    shapes: Any = ((4, 4), (8, 8))
    tile = Variable(param=categorical(*shapes), name=Identifier("tile"))
    return Space(variables=(tile,)), tile


def test_a_configuration_entry_may_be_a_tuple_category() -> None:
    """Test `(8, 8)` is accepted for a variable over tile shapes, and read back."""
    space, tile = _tile_space()

    configuration = Configuration(space, {tile.name: (8, 8)})

    assert configuration.value(tile.name) == (8, 8)
    assert configuration.is_complete()


def test_a_configuration_refuses_a_tuple_that_is_no_category() -> None:
    """Test `(5, 5)` is no shape of the variable: `ConfigurationError`."""
    space, tile = _tile_space()

    with pytest.raises(ConfigurationError):
        Configuration(space, {tile.name: (5, 5)})


def test_configurations_over_tuple_categories_have_keys_that_tell_them_apart() -> None:
    """Test keys of the two shapes differ and a key of one shape repeats."""
    space, tile = _tile_space()

    small = Configuration(space, {tile.name: (4, 4)}).key()
    large = Configuration(space, {tile.name: (8, 8)}).key()

    assert small != large
    assert large == Configuration(space, {tile.name: (8, 8)}).key()
    assert hash(large) == hash(Configuration(space, {tile.name: (8, 8)}).key())


def test_a_space_over_tuple_categories_counts_enumerates_and_samples() -> None:
    """Test the two shapes are counted, enumerated, and sampled as tuples."""
    space, tile = _tile_space()

    sampled, trace = space.sample(RandomOracle(seed=3))

    assert space.cardinality() == Cardinality(CardinalityKind.EXACT, 2, None)
    assert {c.value(tile.name) for c in space.enumerate()} == {(4, 4), (8, 8)}
    assert sampled.value(tile.name) in {(4, 4), (8, 8)}
    assert space.replay(trace).key() == sampled.key()


def test_a_forbidden_clause_over_a_tuple_category_removes_it() -> None:
    """Test `tile in {(8, 8)}` forbids that shape, as the sketch's clause does."""
    tile = Variable(param=categorical(*((4, 4), (8, 8))), name=Identifier("tile"))
    space = Space(
        variables=(tile,),
        forbidden=(Forbidden((InSetConstraint(tile.name, {(8, 8)}),)),),
    )

    assert {c.value(tile.name) for c in space.enumerate()} == {(4, 4)}
    with pytest.raises(ConfigurationError):
        Configuration(space, {tile.name: (8, 8)})
