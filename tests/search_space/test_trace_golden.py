"""Replay of the search-stream corpus through the Python interface.

The corpus at `rust/fhy-core/tests/golden/trace_cases.json` records MOGA-VM's
decision stream (the oracle) with the port's answers beside the oracle's;
the Rust core replays it in `tests/it/search_space_trace_golden.rs`. This
suite runs each case through `fhy_core.search_space` and checks the binding
answers as the corpus's port answers say: the domains it builds or refuses
and their coordinates and admission, the replays it accepts or refuses, where
a replayed stream is refused and the text that says so, and the cardinality
of extracted spaces. Every case whose port answers differ from the oracle's
names the divergence that explains them, and no other case does.

Each replay collects the mismatches it finds instead of stopping at the
first, so a control can change one expected value and see the replay report
it.
"""

import copy
import json
import pathlib
from collections.abc import Callable
from typing import Any

import pytest

from fhy_core.identifier import Identifier
from fhy_core.search_space import (
    Alternative,
    Choice,
    ChoiceDomain,
    OrderDomain,
    Recorder,
    ReplayMismatchError,
    ReplayOracle,
    Space,
    StepDomainError,
    StridedDomain,
    StridedRun,
    Variable,
)
from fhy_core.symbolic.param import create_categorical_param, create_permutation_param

_CORPUS = (
    pathlib.Path(__file__).parents[2]
    / "rust"
    / "fhy-core"
    / "tests"
    / "golden"
    / "trace_cases.json"
)
_FAMILIES = ("domains", "replays", "streams", "extractions")
_DIVERGENCES = ("D-SS2-1", "D-SS2-2", "D-SS2-3", "D-SS2-7")

Case = dict[str, Any]


def _read_corpus() -> dict[str, Any]:
    """Return the parsed corpus."""
    corpus: dict[str, Any] = json.loads(_CORPUS.read_text(encoding="utf-8"))
    return corpus


class _Scripted:
    """An oracle answering the given coordinates, one per step, in order."""

    def __init__(self, answers: list[Any]) -> None:
        self._answers = iter(answers)

    def decide(self, step: Any) -> Any:
        del step
        return next(self._answers)


def _value(member: dict[str, Any], labels: list[Identifier]) -> Any:
    """Return the Python value of the corpus member `member`."""
    ((kind, payload),) = member.items()
    return labels[payload] if kind == "label" else payload


def _probe(probe: Any) -> Any:
    """Return a strided domain's probe: an integer or a Boolean, as given."""
    return probe


def _domain(spec: dict[str, Any]) -> ChoiceDomain | StridedDomain:
    """Return the domain `spec` describes, its labels fresh identifiers."""
    labels = [Identifier("label") for _ in range(8)]
    if spec["shape"] == "choice":
        return ChoiceDomain(tuple(_value(value, labels) for value in spec["values"]))
    return StridedDomain(tuple(StridedRun(*run) for run in spec["runs"]))


def _replay_choice(case: Case) -> dict[str, Any]:
    values = [_value(value, []) for value in case["spec"]["values"]]
    try:
        domain = ChoiceDomain(values)
    except StepDomainError as error:
        return {"outcome": "refused", "error": _name_domain_error(str(error))}
    return {
        "outcome": "built",
        "cardinality": domain.cardinality,
        "coordinates": [
            domain.coordinate_of(domain.value_at(index))
            for index in range(domain.cardinality)
        ],
        "admits": [
            domain.admits(_value(probe, [])) for probe in case["spec"]["probes"]
        ],
    }


def _name_domain_error(text: str) -> str:
    """Return the corpus's name of the domain refusal whose text is `text`."""
    if text == "a choice domain needs at least one value":
        return "empty_choice"
    if text == "an order domain needs at least one element":
        return "empty_order"
    if text.endswith("are equal"):
        return "repeated_value"
    return f"unknown: {text}"


def _replay_order(case: Case) -> dict[str, Any]:
    size = case["spec"]["size"]
    try:
        domain = OrderDomain(tuple(Identifier("element") for _ in range(size)))
    except StepDomainError as error:
        return {"outcome": "refused", "error": _name_domain_error(str(error))}
    permutations = [tuple(positions) for positions in case["spec"]["permutations"]]
    return {
        "outcome": "built",
        "cardinality": domain.cardinality,
        "orderings": [
            list(domain.coordinate_of(domain.value_at(positions)))
            for positions in permutations
        ],
        "round_trips": [
            domain.coordinate_of(domain.value_at(positions)) == positions
            for positions in permutations
        ],
    }


def _replay_strided(case: Case) -> dict[str, Any]:
    domain = StridedDomain(tuple(StridedRun(*run) for run in case["spec"]["runs"]))
    return {
        "outcome": "built",
        "cardinality": domain.cardinality,
        "values": [domain.value_at(index) for index in range(domain.cardinality)],
        "admits": [domain.admits(_probe(probe)) for probe in case["spec"]["probes"]],
    }


def _replay_domain(case: Case) -> dict[str, Any]:
    """Return the binding's answers to the domain case `case`."""
    replay: dict[str, Callable[[Case], dict[str, Any]]] = {
        "choice": _replay_choice,
        "order": _replay_order,
        "strided": _replay_strided,
    }
    return replay[case["shape"]](case)


def _replay_replay(case: Case) -> dict[str, Any]:
    """Return the binding's answers to the replay case `case`."""
    recorded = _domain(case["recorded"])
    offered = _domain(case["offered"])
    subject = Identifier("s")
    recorder = Recorder(_Scripted([case["coordinate"]]))
    recorder.decide_dynamic("moga.cir.option", subject, recorded)
    replaying = Recorder(ReplayOracle(recorder.trace))
    try:
        answer = replaying.decide_dynamic("moga.cir.option", subject, offered)
    except ReplayMismatchError as error:
        if "is offered another domain than the one recorded" not in str(error):
            return {"outcome": f"refused otherwise: {error}"}
        return {"outcome": "refused"}
    return {"outcome": "replayed", "coordinate": offered.coordinate_of(answer)}


def _replay_stream(case: Case) -> dict[str, Any]:
    """Return the binding's answers to the stream case `case`."""
    domain = ChoiceDomain((0, 1, 2))
    subject = Identifier("s")
    recorder = Recorder(_Scripted([index % 3 for index in range(case["recorded"])]))
    for _ in range(case["recorded"]):
        recorder.decide_dynamic("moga.cir.address", subject, domain)
    oracle = ReplayOracle(recorder.trace)
    replaying = Recorder(oracle)
    for position in range(case["asked"]):
        try:
            replaying.decide_dynamic("moga.cir.address", subject, domain)
        except ReplayMismatchError as error:
            expected = f"the trace has no step {position} for the run to replay"
            if expected not in str(error):
                return {"refused_at": f"otherwise: {error}"}
            return {"refused_at": position, "unconsumed_at_finish": None}
    try:
        oracle.finish()
    except ReplayMismatchError as error:
        text = str(error)
        prefix = "the run ended before asking the recorded step "
        if not text.startswith(prefix):
            return {"refused_at": None, "unconsumed_at_finish": f"otherwise: {text}"}
        return {"refused_at": None, "unconsumed_at_finish": int(text[len(prefix) :])}
    return {"refused_at": None, "unconsumed_at_finish": None}


def _alternative(axes: list[dict[str, int]]) -> Alternative:
    """Return the alternative holding a variable per extracted axis."""
    variables = []
    for axis in axes:
        param: Any
        if "choice" in axis:
            param = create_categorical_param(frozenset(range(axis["choice"])))
        else:
            param = create_permutation_param(
                tuple(Identifier("level") for _ in range(axis["order"]))
            )
        variables.append(Variable(param=param, name=Identifier("axis")))
    return Alternative(variables=variables, name=Identifier("option"))


def _replay_extraction(case: Case) -> dict[str, Any]:
    """Return the binding's answers to the extraction case `case`."""
    choices = [
        Choice(tuple(_alternative(axes) for axes in options), name=Identifier("entry"))
        for options in case["entries"]
    ]
    cardinality = Space(choices=choices, name=Identifier("extracted")).cardinality(
        budget=1_000_000
    )
    return {"cardinality": cardinality.count}


_REPLAYS: dict[str, Callable[[Case], dict[str, Any]]] = {
    "domains": _replay_domain,
    "replays": _replay_replay,
    "streams": _replay_stream,
    "extractions": _replay_extraction,
}


def _find_mismatches(corpus: dict[str, Any]) -> list[str]:
    """Return a line per case the binding answers otherwise than recorded."""
    mismatches = []
    for family in _FAMILIES:
        for case in corpus[family]:
            answers = _REPLAYS[family](case)
            if answers != case["port"]:
                mismatches.append(
                    f"{family}/{case['name']}: {answers} != {case['port']}"
                )
    return mismatches


def _find_mistagged(corpus: dict[str, Any]) -> list[str]:
    """Return the cases whose divergence tag disagrees with their answers."""
    return [
        f"{family}/{case['name']}"
        for family in _FAMILIES
        for case in corpus[family]
        if (case["oracle"] != case["port"]) != (case["divergence"] is not None)
    ]


def test_every_case_replays_through_the_binding_as_recorded() -> None:
    """Test the binding answers every corpus case as the port is recorded to."""
    corpus = _read_corpus()

    mismatches = _find_mismatches(corpus)

    assert mismatches == []


def test_a_case_differs_from_the_oracle_exactly_when_it_names_a_divergence() -> None:
    """Test each divergence tag explains a difference, and every difference has one."""
    corpus = _read_corpus()

    mistagged = _find_mistagged(corpus)

    assert mistagged == []


def test_the_corpus_tags_every_divergence_it_exercises() -> None:
    """Test every divergence the stream's port makes appears in the corpus."""
    corpus = _read_corpus()

    tags = {case["divergence"] for family in _FAMILIES for case in corpus[family]}

    assert set(_DIVERGENCES) <= tags


@pytest.mark.parametrize(
    ("family", "field", "change"),
    [
        ("domains", "cardinality", lambda value: value + 1),
        (
            "replays",
            "outcome",
            lambda value: "refused" if value != "refused" else "replayed",
        ),
        ("streams", "unconsumed_at_finish", lambda value: 99),
        ("extractions", "cardinality", lambda value: value + 1),
    ],
    ids=["domain", "replay", "stream", "extraction"],
)
def test_changing_one_expected_answer_makes_the_replay_report_it(
    family: str, field: str, change: Callable[[Any], Any]
) -> None:
    """Test the replay compares for real: one changed expectation is reported."""
    corpus = copy.deepcopy(_read_corpus())
    case = next(case for case in corpus[family] if field in case["port"])
    case["port"][field] = change(case["port"][field])
    only: dict[str, list[Case]] = {name: [] for name in _FAMILIES}
    only[family] = [case]

    mismatches = _find_mismatches(only)

    assert len(mismatches) == 1
    assert mismatches[0].startswith(f"{family}/{case['name']}:")
