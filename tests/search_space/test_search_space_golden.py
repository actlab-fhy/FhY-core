"""Replay of the search-space corpus through the Python interface.

The corpus at `rust/fhy-core/tests/golden/search_space_cases.json` records
MOGA-VM's decision points (the oracle) with the port's answers beside the
oracle's; the Rust core replays it in `tests/it/search_space_golden.rs`.
This suite builds each point through `fhy_core.search_space` and checks the
binding answers as the corpus says: what it builds or refuses, the
structural and alpha verdicts both ways, and the V2 texts of the space and
the configuration, written and read back byte for byte.

An expanded corpus, recorded by hand from the oracle (see
`docs/design/search-space.md`, "Equivalence plan"), replays the same way
when the environment variable `FHY_SEARCH_SPACE_CORPUS` names it.
"""

import json
import os
import pathlib
from typing import Any

import pytest

from fhy_core.identifier import Identifier
from fhy_core.search_space import (
    Alternative,
    Choice,
    Configuration,
    ConfigurationError,
    SearchSpaceError,
    Space,
    Variable,
)
from fhy_core.symbolic.constraint import ConstraintSystem, InSetConstraint
from fhy_core.symbolic.param import CategoricalDomain, Param

_CORPUS = (
    pathlib.Path(__file__).parents[2]
    / "rust"
    / "fhy-core"
    / "tests"
    / "golden"
    / "search_space_cases.json"
)
_EXPANDED = os.environ.get("FHY_SEARCH_SPACE_CORPUS")


def _member(member: dict[str, Any], pool: list[Identifier]) -> Any:
    """Return the value of the corpus member `member`."""
    if "label" in member:
        return pool[member["label"]]
    if "bool" in member:
        return member["bool"]
    return member["int"]


def _param(knob: dict[str, Any], pool: list[Identifier]) -> Param[Any]:
    """Return the param of the knob `knob`, its variable its param's label."""
    variable = pool[knob["param"]]
    categories = tuple(_member(member, pool) for member in knob["categories"])
    kept = knob["kept"]
    constraints = (
        (InSetConstraint(variable, {_member(member, pool) for member in kept}),)
        if kept is not None
        else ()
    )
    return Param(CategoricalDomain(categories), variable, ConstraintSystem(constraints))


def _build(point: dict[str, Any], pool: list[Identifier]) -> tuple[str, Any]:
    """Return the outcome of the point `point` and its configuration, if built."""
    try:
        alternatives = tuple(
            Alternative(
                variables=tuple(
                    Variable(param=_param(knob, pool), name=pool[knob["name"]])
                    for knob in option["knobs"]
                ),
                name=pool[option["name"]],
            )
            for option in point["options"]
        )
        choice = Choice(alternatives, name=pool[point["choice"]])
        space = Space(choices=(choice,), name=pool[point["space"]])
    except SearchSpaceError:
        return "space_refused", None
    entries: list[tuple[Identifier, Any]] = []
    selection = point["selection"]
    if selection is not None:
        entries.append((choice.name, pool[selection["option"]]))
        entries.extend(
            (pool[label], _member(value, pool)) for label, value in selection["values"]
        )
    try:
        return "built", Configuration(space, entries)
    except ConfigurationError:
        return "configuration_refused", None


def _replay(case: dict[str, Any]) -> list[str]:
    """Replay `case`, returning what it found wrong."""
    name, port, wire = case["name"], case["port"], case.get("wire", {})
    pool = [
        Identifier.deserialize_from_dict(
            {"id": case["id_base"] + index, "name_hint": label}
        )
        for index, label in enumerate(case["labels"])
    ]
    problems: list[str] = []
    built: dict[str, Any] = {}
    for side in ("left", "right"):
        if case[side] is None:
            continue
        outcome, configuration = _build(case[side], pool)
        if outcome != port[side]:
            problems.append(f"{name}: the {side} point is {outcome}, not {port[side]}")
        if configuration is None:
            continue
        for key, text in (
            (f"{side}_space", configuration.space.to_json()),
            (f"{side}_configuration", configuration.to_json()),
        ):
            if wire[key] != text:
                problems.append(f"{name}: {key} writes {text}, not {wire[key]}")
        decoded = Configuration.from_json(wire[f"{side}_configuration"])
        if decoded.to_json() != wire[f"{side}_configuration"]:
            problems.append(f"{name}: the {side} configuration does not read back")
        if decoded.key() != configuration.key():
            problems.append(f"{name}: the {side} configuration decodes another key")
        built[side] = configuration
    if "left" in built and "right" in built:
        left, right = built["left"], built["right"]
        structural = [
            left.is_structurally_equivalent(right),
            right.is_structurally_equivalent(left),
        ]
        alpha = [left.is_alpha_equivalent(right), right.is_alpha_equivalent(left)]
        if structural != port["structural"] or alpha != port["alpha"]:
            problems.append(
                f"{name}: structural {structural} and alpha {alpha}, not "
                f"{port['structural']} and {port['alpha']}"
            )
    return problems


def _replay_document(path: pathlib.Path) -> int:
    """Replay every case of the corpus at `path`, returning how many it holds."""
    cases = json.loads(path.read_text())["cases"]
    problems = [problem for case in cases for problem in _replay(case)]
    assert not problems, "\n".join(problems[:20])
    return len(cases)


def test_every_committed_case_replays_as_recorded() -> None:
    """Test the binding answers every committed case as the corpus records."""
    assert _replay_document(_CORPUS) >= 100


@pytest.mark.slow
@pytest.mark.skipif(_EXPANDED is None, reason="FHY_SEARCH_SPACE_CORPUS is not set")
def test_every_expanded_case_replays_as_recorded() -> None:
    """Test the binding answers every case of the expanded corpus as recorded."""
    assert _EXPANDED is not None
    assert _replay_document(pathlib.Path(_EXPANDED)) > 0
