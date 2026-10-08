"""The interface suite of space editing, completion and per-decision completeness.

`Space.with_decisions` replaces top-level variables and choices in their
slots or appends them, `Space.without_decisions` removes top-level
decisions with what depends on them, `Space.complete` fills a partial
configuration from an oracle, and `Configuration.is_complete_under` asks
whether a decision and everything under it is assigned or inactive.
"""

from typing import Any

import pytest

from fhy_core.identifier import Identifier
from fhy_core.search_space import (
    Choice,
    Condition,
    Configuration,
    DuplicateNameError,
    RandomOracle,
    SearchSpaceError,
    Space,
    Trace,
    TraceError,
    Variable,
)
from fhy_core.symbolic.constraint import InSetConstraint

from .conftest import (
    build_complete_configuration,
    build_tiling_space,
    categorical,
    make_alternative,
    make_choice,
    make_variable,
)


class _Constant:
    """An oracle answering coordinate 0 to every step, recording who asked."""

    def __init__(self) -> None:
        self.subjects: list[Identifier] = []

    def decide(self, step: Any) -> int:
        self.subjects.append(step.subject)
        return 0


class _Forbidden:
    """An oracle that fails when asked."""

    def decide(self, step: Any) -> Any:
        raise AssertionError("the oracle was asked")


class _Boom(Exception):
    """The exception the failing oracle raises."""


class _Failing:
    """An oracle that raises a given exception when asked."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    def decide(self, step: Any) -> Any:
        raise self._error


# ===========================================================================
# Space.with_decisions
# ===========================================================================


def test_with_decisions_appends_a_choice_after_the_last_one() -> None:
    """Test a new choice goes last and every old position is kept."""
    tiling = build_tiling_space()
    inner = make_variable("inner", 1, 2)
    extra = make_choice(
        "extra", make_alternative("on", (inner,)), make_alternative("off")
    )

    edited = tiling.space.with_decisions(choices=[extra])

    assert edited is not tiling.space
    assert edited.choices == (tiling.layout, extra)
    assert edited.variables == (tiling.unroll,)
    assert edited.decision_order == (
        *tiling.space.decision_order,
        extra.name,
        inner.name,
    )
    assert edited.choices[-1] is extra


def test_with_decisions_keeps_the_trace_positions_of_appended_choices() -> None:
    """Test appending a choice moves no earlier step: the old steps are a prefix."""
    tiling = build_tiling_space()
    extra = make_choice("extra", make_alternative("on"), make_alternative("off"))
    edited = tiling.space.with_decisions(choices=[extra])

    _, old_trace = tiling.space.sample(_Constant())
    _, new_trace = edited.sample(_Constant())

    old_positions = [step.decision for step in old_trace]
    new_positions = [step.decision for step in new_trace]
    assert new_positions[: len(old_positions)] == old_positions
    assert new_positions[len(old_positions) :] == [len(old_positions)]


def test_with_decisions_appends_variables_in_the_order_given() -> None:
    """Test new variables follow the last top-level variable, as given."""
    tiling = build_tiling_space()
    first, second = make_variable("first", 1, 2), make_variable("second", 3, 4)

    edited = tiling.space.with_decisions(variables=[first, second])

    assert edited.variables == (tiling.unroll, first, second)
    assert edited.choices == (tiling.layout,)
    assert edited.decision_order == (
        tiling.unroll.name,
        first.name,
        second.name,
        tiling.layout.name,
        tiling.tile.name,
    )


def test_with_decisions_replaces_a_variable_in_its_slot() -> None:
    """Test a variable named like a top-level one takes its place."""
    first, middle, last = (make_variable(name, 1, 2) for name in ("a", "b", "c"))
    space = Space(variables=(first, middle, last))
    replacement = Variable(param=categorical(1, 2, 3), name=middle.name)

    edited = space.with_decisions(variables=[replacement])

    assert edited.variables == (first, replacement, last)
    assert edited.variables[1] is replacement
    assert edited.decision_order == space.decision_order


def test_with_decisions_replaces_a_choice_in_its_slot() -> None:
    """Test a choice named like a top-level one takes its place."""
    first = make_choice("first", make_alternative("p"), make_alternative("q"))
    second = make_choice("second", make_alternative("r"), make_alternative("s"))
    space = Space(choices=(first, second))
    replacement = Choice((make_alternative("t"),), name=first.name)

    edited = space.with_decisions(choices=[replacement])

    assert edited.choices == (replacement, second)
    assert edited.choices[0] is replacement
    assert edited.decision_order == space.decision_order


def test_with_decisions_replaces_and_appends_in_one_call() -> None:
    """Test a replacement and a new variable are both applied by one call."""
    first, second = make_variable("a", 1, 2), make_variable("b", 1, 2)
    space = Space(variables=(first, second))
    replacement = Variable(param=categorical(7, 8), name=first.name)
    appended = make_variable("c", 5, 6)

    edited = space.with_decisions(variables=[appended, replacement])

    assert edited.variables == (replacement, second, appended)


def test_with_decisions_keeps_the_name_conditions_forbidden_and_notes() -> None:
    """Test the edit changes the decisions only."""
    tiling = build_tiling_space(unroll_on_tiled_only=True, forbid_unroll_four=True)
    extra = make_variable("extra", 1, 2)

    edited = tiling.space.with_decisions(variables=[extra])

    assert edited.name == tiling.space.name
    assert edited.conditions == tiling.space.conditions
    assert edited.forbidden == tiling.space.forbidden
    assert edited.notes == tiling.space.notes


def test_with_decisions_leaves_the_original_space_as_it_was() -> None:
    """Test the space edited is unchanged."""
    tiling = build_tiling_space()
    before = tiling.space.decision_order

    tiling.space.with_decisions(variables=[make_variable("extra", 1, 2)])

    assert tiling.space.decision_order == before
    assert tiling.space.variables == (tiling.unroll,)


def test_with_decisions_of_nothing_is_the_same_space_structurally() -> None:
    """Test no decision given returns a structurally equivalent space."""
    tiling = build_tiling_space()

    edited = tiling.space.with_decisions()

    assert edited.is_structurally_equivalent(tiling.space)
    assert edited.decision_order == tiling.space.decision_order


def test_with_decisions_refuses_a_variable_named_like_a_choice() -> None:
    """Test a new variable named like a top-level choice is a `DuplicateNameError`."""
    tiling = build_tiling_space()
    clash = Variable(param=categorical(1, 2), name=tiling.layout.name)

    with pytest.raises(DuplicateNameError):
        tiling.space.with_decisions(variables=[clash])


def test_with_decisions_refuses_a_name_used_below() -> None:
    """Test a new top-level variable named like a nested one is refused."""
    tiling = build_tiling_space()
    clash = Variable(param=categorical(1, 2), name=tiling.tile.name)

    with pytest.raises(DuplicateNameError):
        tiling.space.with_decisions(variables=[clash])


def test_with_decisions_refuses_a_condition_naming_a_decision_no_longer_there() -> None:
    """Test replacing a choice by one without a conditioned decision is refused."""
    tiling = build_tiling_space()
    guarded = Space(
        variables=(tiling.unroll,),
        choices=(tiling.layout,),
        conditions=(
            Condition(tiling.tile.name, (InSetConstraint(tiling.unroll.name, {1}),)),
        ),
    )
    without_tile = Choice((make_alternative("only"),), name=tiling.layout.name)

    with pytest.raises(SearchSpaceError):
        guarded.with_decisions(choices=[without_tile])


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param({"variables": [3]}, id="variable"),
        pytest.param({"choices": ["choice"]}, id="choice"),
    ],
)
def test_with_decisions_refuses_what_is_no_decision(arguments: dict[str, Any]) -> None:
    """Test an element that is no `Variable` or `Choice` raises `TypeError`."""
    with pytest.raises(TypeError):
        build_tiling_space().space.with_decisions(**arguments)


# ===========================================================================
# Space.without_decisions
# ===========================================================================


def test_without_decisions_removes_a_top_level_variable() -> None:
    """Test a removed variable leaves the others in order."""
    tiling = build_tiling_space()

    edited = tiling.space.without_decisions([tiling.unroll.name])

    assert edited.variables == ()
    assert edited.choices == (tiling.layout,)
    assert edited.decision_order == (tiling.layout.name, tiling.tile.name)
    assert edited.name == tiling.space.name


def test_without_decisions_removes_a_choice_with_its_subtree() -> None:
    """Test removing a choice removes the decisions under it."""
    tiling = build_tiling_space()

    edited = tiling.space.without_decisions([tiling.layout.name])

    assert edited.choices == ()
    assert edited.variables == (tiling.unroll,)
    assert edited.decision_order == (tiling.unroll.name,)
    assert edited.decision(tiling.tile.name) is None


def test_without_decisions_removes_the_conditions_targeting_what_goes() -> None:
    """Test a condition on a removed decision, or on one under it, is dropped."""
    tiling = build_tiling_space()
    on_tile = Condition(tiling.tile.name, (InSetConstraint(tiling.unroll.name, {1}),))
    guarded = Space(
        variables=(tiling.unroll,), choices=(tiling.layout,), conditions=(on_tile,)
    )

    edited = guarded.without_decisions([tiling.layout.name])

    assert edited.conditions == ()
    assert guarded.conditions == (on_tile,)


def test_without_decisions_refuses_a_condition_naming_a_removed_decision() -> None:
    """Test a condition that names a removed decision is refused, not pruned."""
    tiling = build_tiling_space(unroll_on_tiled_only=True)

    with pytest.raises(
        SearchSpaceError, match="is not a decision of the space"
    ) as excinfo:
        tiling.space.without_decisions([tiling.layout.name])

    assert type(excinfo.value) is SearchSpaceError
    assert repr(tiling.layout.name) in str(excinfo.value)


def test_without_decisions_refuses_a_forbidden_clause_naming_a_removed_decision() -> (
    None
):
    """Test a forbidden clause naming a removed decision is refused."""
    tiling = build_tiling_space(forbid_unroll_four=True)

    with pytest.raises(
        SearchSpaceError, match="is not a decision of the space"
    ) as excinfo:
        tiling.space.without_decisions([tiling.unroll.name])

    assert type(excinfo.value) is SearchSpaceError
    assert repr(tiling.unroll.name) in str(excinfo.value)


def test_without_decisions_removes_a_name_given_twice_once() -> None:
    """Test a repeated name is no error and removes the decision once."""
    tiling = build_tiling_space()

    edited = tiling.space.without_decisions([tiling.unroll.name, tiling.unroll.name])

    assert edited.decision_order == (tiling.layout.name, tiling.tile.name)


def test_without_decisions_removes_several_decisions() -> None:
    """Test a variable and a choice go in one call."""
    tiling = build_tiling_space()

    edited = tiling.space.without_decisions([tiling.layout.name, tiling.unroll.name])

    assert edited.decision_order == ()
    assert edited.variables == ()
    assert edited.choices == ()


def test_without_decisions_of_nothing_is_the_same_space_structurally() -> None:
    """Test no name given returns a structurally equivalent space."""
    tiling = build_tiling_space()

    edited = tiling.space.without_decisions([])

    assert edited.is_structurally_equivalent(tiling.space)


def test_without_decisions_leaves_the_original_space_as_it_was() -> None:
    """Test the space edited is unchanged."""
    tiling = build_tiling_space()
    before = tiling.space.decision_order

    tiling.space.without_decisions([tiling.unroll.name])

    assert tiling.space.decision_order == before


@pytest.mark.parametrize("which", ["nested", "unknown", "alternative"])
def test_without_decisions_refuses_a_name_that_is_not_a_top_level_decision(
    which: str,
) -> None:
    """Test a nested decision, an unknown name or an alternative is refused."""
    tiling = build_tiling_space()
    name = {
        "nested": tiling.tile.name,
        "unknown": Identifier("ghost"),
        "alternative": tiling.tiled.name,
    }[which]

    with pytest.raises(
        SearchSpaceError, match="is not a top-level decision"
    ) as excinfo:
        tiling.space.without_decisions([name])

    assert type(excinfo.value) is SearchSpaceError
    assert repr(name) in str(excinfo.value)


def test_without_decisions_names_the_first_name_that_is_not_top_level() -> None:
    """Test the first offending name, in the order given, is the one reported."""
    tiling = build_tiling_space()
    ghost = Identifier("ghost")

    with pytest.raises(SearchSpaceError) as excinfo:
        tiling.space.without_decisions([tiling.unroll.name, ghost, tiling.tile.name])

    assert repr(ghost) in str(excinfo.value)


def test_without_decisions_refuses_a_name_that_is_no_identifier() -> None:
    """Test a name that is no `Identifier` raises `TypeError`."""
    with pytest.raises(TypeError):
        build_tiling_space().space.without_decisions(["unroll"])  # type: ignore[list-item]  # test: invalid input


# ===========================================================================
# Space.complete
# ===========================================================================


def test_complete_fills_what_the_partial_leaves_open() -> None:
    """Test the result is complete and keeps every entry the partial held."""
    tiling = build_tiling_space()
    partial = Configuration(tiling.space, {tiling.unroll.name: 2})
    oracle = _Constant()

    completed, trace = tiling.space.complete(partial, oracle)

    assert isinstance(completed, Configuration)
    assert isinstance(trace, Trace)
    assert completed.space is tiling.space
    assert completed.is_complete()
    assert set(partial.entries) <= set(completed.entries)
    assert completed.value(tiling.unroll.name) == 2
    assert tiling.unroll.name not in oracle.subjects
    assert set(oracle.subjects) == {tiling.layout.name, tiling.tile.name}


def test_complete_returns_the_full_trace_of_the_result() -> None:
    """Test the trace has a step per decision and replays to the result."""
    tiling = build_tiling_space()
    partial = Configuration(tiling.space, {tiling.layout.name: tiling.tiled.name})

    completed, trace = tiling.space.complete(partial, _Constant())

    assert len(trace) == len(completed.entries)
    assert tiling.space.replay(trace).key() == completed.key()


def test_complete_of_a_complete_configuration_never_asks() -> None:
    """Test a complete configuration comes back as it is, with its trace."""
    tiling = build_tiling_space()
    configuration = build_complete_configuration(tiling)

    completed, trace = tiling.space.complete(configuration, _Forbidden())

    assert completed.key() == configuration.key()
    assert completed.entries == configuration.entries
    assert tiling.space.replay(trace).key() == configuration.key()
    assert len(trace) == len(configuration.entries)


def test_complete_of_the_empty_configuration_is_a_sample() -> None:
    """Test completing nothing asks every decision, as `sample` does."""
    tiling = build_tiling_space()

    completed, trace = tiling.space.complete(
        Configuration(tiling.space), RandomOracle(seed=9)
    )
    sampled, sampled_trace = tiling.space.sample(RandomOracle(seed=9))

    assert completed.key() == sampled.key()
    assert trace == sampled_trace


def test_complete_follows_a_condition_made_inactive_by_the_partial() -> None:
    """Test a decision the partial makes inactive is not asked."""
    tiling = build_tiling_space(unroll_on_tiled_only=True)
    partial = Configuration(tiling.space, {tiling.layout.name: tiling.flat.name})
    oracle = _Constant()

    completed, _ = tiling.space.complete(partial, oracle)

    assert completed.is_complete()
    assert oracle.subjects == []
    assert completed.value(tiling.unroll.name) is None


def test_complete_is_deterministic_from_a_seeded_oracle() -> None:
    """Test one seed over one partial gives one result."""
    tiling = build_tiling_space()
    partial = Configuration(
        tiling.space, {tiling.layout.name: tiling.tiled.name, tiling.tile.name: 8}
    )

    left = tiling.space.complete(partial, RandomOracle(seed=5))
    right = tiling.space.complete(partial, RandomOracle(seed=5))

    assert left[0].key() == right[0].key()
    assert left[1] == right[1]


def test_complete_refuses_a_configuration_of_another_space() -> None:
    """Test a configuration of another space raises `TraceError`."""
    tiling, other = build_tiling_space(), build_tiling_space()

    with pytest.raises(TraceError):
        tiling.space.complete(Configuration(other.space), RandomOracle(seed=0))


def test_complete_propagates_an_exception_of_the_oracle() -> None:
    """Test the error a Python oracle raises is the one `complete` raises."""
    tiling = build_tiling_space()
    error = _Boom("the oracle failed")

    with pytest.raises(_Boom, match="the oracle failed") as info:
        tiling.space.complete(Configuration(tiling.space), _Failing(error))

    assert info.value is error


def test_complete_after_an_edit_carries_the_values_across() -> None:
    """Test the values of a configuration survive an edit through `complete`."""
    tiling = build_tiling_space()
    old = build_complete_configuration(tiling, tile=8)
    extra = make_variable("extra", 1, 2)
    edited = tiling.space.with_decisions(variables=[extra])

    carried = Configuration(edited, old.entries)
    completed, _ = edited.complete(carried, _Constant())

    assert completed.is_complete()
    assert completed.value(tiling.tile.name) == 8
    assert completed.value(tiling.unroll.name) == 2
    assert completed.value(extra.name) in {1, 2}


# ===========================================================================
# Configuration.is_complete_under
# ===========================================================================


def test_an_unassigned_variable_is_not_complete_under_itself() -> None:
    """Test an active variable with no value is `False`, and `True` once assigned."""
    tiling = build_tiling_space()

    empty = Configuration(tiling.space)
    assigned = empty.with_entry(tiling.unroll.name, 2)

    assert empty.is_complete_under(tiling.unroll.name) is False
    assert assigned.is_complete_under(tiling.unroll.name) is True


def test_an_unassigned_choice_is_not_complete_under_itself() -> None:
    """Test a choice with no alternative chosen is `False`."""
    tiling = build_tiling_space()

    assert Configuration(tiling.space).is_complete_under(tiling.layout.name) is False


def test_a_choice_over_an_unassigned_variable_is_not_complete() -> None:
    """Test a chosen alternative with an unassigned variable is `False`, then `True`."""
    tiling = build_tiling_space()
    chosen = Configuration(tiling.space, {tiling.layout.name: tiling.tiled.name})

    assigned = chosen.with_entry(tiling.tile.name, 4)

    assert chosen.is_complete_under(tiling.layout.name) is False
    assert assigned.is_complete_under(tiling.layout.name) is True


def test_a_choice_of_an_empty_alternative_is_complete_once_chosen() -> None:
    """Test choosing the alternative that holds nothing completes the choice."""
    tiling = build_tiling_space()
    flat = Configuration(tiling.space, {tiling.layout.name: tiling.flat.name})

    assert flat.is_complete_under(tiling.layout.name) is True
    assert flat.is_complete() is False


def test_a_decision_under_an_unchosen_alternative_is_complete_as_inactive() -> None:
    """Test a variable that is inactive counts as complete."""
    tiling = build_tiling_space()
    flat = Configuration(tiling.space, {tiling.layout.name: tiling.flat.name})

    assert flat.is_complete_under(tiling.tile.name) is True


def test_an_inactive_conditional_variable_is_complete_as_inactive() -> None:
    """Test a variable whose condition fails counts as complete, unassigned."""
    tiling = build_tiling_space(unroll_on_tiled_only=True)
    flat = Configuration(tiling.space, {tiling.layout.name: tiling.flat.name})
    tiled = Configuration(tiling.space, {tiling.layout.name: tiling.tiled.name})

    assert flat.is_complete_under(tiling.unroll.name) is True
    assert tiled.is_complete_under(tiling.unroll.name) is False


def test_completeness_under_a_choice_descends_through_every_depth() -> None:
    """Test a nested choice and its variable count towards the outer choice."""
    leaf = make_variable("leaf", 1, 2)
    inner = make_choice(
        "inner", make_alternative("deep", (leaf,)), make_alternative("shallow")
    )
    outer = make_choice("outer", make_alternative("holder", choices=(inner,)))
    space = Space(choices=(outer,))
    holder, deep = outer.alternatives[0], inner.alternatives[0]

    outer_only = Configuration(space, {outer.name: holder.name})
    inner_chosen = outer_only.with_entry(inner.name, deep.name)
    leaf_assigned = inner_chosen.with_entry(leaf.name, 1)

    assert outer_only.is_complete_under(outer.name) is False
    assert outer_only.is_complete_under(inner.name) is False
    assert inner_chosen.is_complete_under(outer.name) is False
    assert inner_chosen.is_complete_under(inner.name) is False
    assert leaf_assigned.is_complete_under(outer.name) is True
    assert leaf_assigned.is_complete_under(inner.name) is True
    assert leaf_assigned.is_complete_under(leaf.name) is True


@pytest.mark.parametrize("which", ["unknown", "alternative", "space"])
def test_a_name_that_is_no_decision_is_none(which: str) -> None:
    """Test an unknown name, an alternative's name and the space's name give `None`."""
    tiling = build_tiling_space()
    configuration = build_complete_configuration(tiling)
    name = {
        "unknown": Identifier("ghost"),
        "alternative": tiling.tiled.name,
        "space": tiling.space.name,
    }[which]

    assert configuration.is_complete_under(name) is None


def test_a_complete_configuration_is_complete_under_every_decision() -> None:
    """Test `is_complete()` and `is_complete_under` agree when complete."""
    tiling = build_tiling_space()
    configuration = build_complete_configuration(tiling)

    assert configuration.is_complete()
    for name in tiling.space.decision_order:
        assert configuration.is_complete_under(name) is True


def test_completeness_of_a_partial_configuration_is_per_decision() -> None:
    """Test each decision of a partial configuration answers for itself."""
    tiling = build_tiling_space()
    partial = Configuration(
        tiling.space, {tiling.unroll.name: 1, tiling.layout.name: tiling.tiled.name}
    )

    assert partial.is_complete() is False
    assert [
        partial.is_complete_under(name) for name in tiling.space.decision_order
    ] == [True, False, False]
