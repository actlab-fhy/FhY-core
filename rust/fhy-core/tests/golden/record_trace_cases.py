r"""Record the search-stream golden corpus from the MOGA-VM oracle.

Four families of cases, each described once, answered by the oracle and,
beside it, by the port's rules (``docs/design/search-space.md``, "SS2:
traces, oracles, enumeration and sampling (plan)"):

- **domains**: a choice domain over integers, strings and Booleans, an
  order domain over labels, or a strided domain over runs; the oracle's
  ``ChoiceDomain``, ``OrderDomain`` and ``AddressDomain`` answer whether it
  constructs, its cardinality, the value at each coordinate, the
  coordinate of each value, and whether it admits probe values;
- **replays**: one step recorded over one domain and replayed over
  another; the oracle's ``ReplayOracle`` answers whether it replays and
  the coordinate it answers;
- **streams**: a stream of ``recorded`` steps replayed by a stream of
  ``asked`` steps; the oracle answers whether the replay is refused and
  where, and whether it is exhausted at the end;
- **extractions**: a one-level ``ExtractedSearchSpace`` of option axes whose
  options expose choice and order axes; the oracle answers its
  ``cardinality()``, which the port's ``Space::cardinality`` of the
  corresponding space must give exactly.

A case whose port answer differs from the oracle's names the divergence
that explains it (D-SS2-1, D-SS2-2, D-SS2-3, D-SS2-7); any other
difference stops the recorder for review. ``tests/it/search_space_trace_golden.rs``
replays the corpus through the Rust port.

The oracle is MOGA-VM's ``cir/lowering/search/{decisions,oracle,errors,extraction}.py``
at ``3d93ba3`` on fhy_core v0.1.8, assembled in a temporary directory:
``git archive`` of the four modules from the MOGA-VM checkout
``--moga-vm-repo`` names, ``git archive`` of this repository's
``src/fhy_core`` at the tag ``v0.1.8``, and stand-ins for the MOGA imports
``extraction.py`` makes and never calls here (``_STAND_INS``). The oracle is
frozen, so the committed corpus is a fixed regression corpus, as
``record_search_space_cases.py``'s is; CI only replays it. Run it by hand
from the repository root::

    uv run --no-sync --with networkx python \\
        rust/fhy-core/tests/golden/record_trace_cases.py \\
        --moga-vm-repo <MOGA-VM checkout>

This overwrites ``rust/fhy-core/tests/golden/trace_cases.json``.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import io
import random
import subprocess
import sys
import tarfile
import tempfile
import types
from math import factorial, prod
from pathlib import Path
from typing import Any

from _golden_support import build_provenance, write_document

GENERATOR_COMMAND = (
    "uv run --no-sync --with networkx python"
    " rust/fhy-core/tests/golden/record_trace_cases.py"
    " --moga-vm-repo <MOGA-VM checkout>"
)
ORACLE = {"moga_vm": "origin/dev 3d93ba3", "fhy_core": "v0.1.8 (ad9a311)"}
_MOGA_VM_COMMIT = "3d93ba3"
_FHY_CORE_TAG = "v0.1.8"
_SEARCH = "src/moga_vm/cir/lowering/search"
_MODULES = ("errors", "decisions", "oracle", "extraction")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_OUTPUT = Path(__file__).with_name("trace_cases.json")
_DEFAULT_SEED = 0
_DEFAULT_RANDOM_COUNT = 40
# How often a random extracted axis is a choice axis rather than an order.
_CHOICE_AXIS_PROBABILITY = 0.6

Json = dict[str, Any]


# ---------------------------------------------------------------------------
# The oracle
# ---------------------------------------------------------------------------


def _extract(repository: Path, revision: str, path: str, destination: Path) -> Path:
    """Extract `path` at `revision` of `repository` into `destination`."""
    archive = subprocess.run(
        ["git", "-C", str(repository), "archive", "--format=tar", revision, path],
        capture_output=True,
        check=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(destination, filter="data")
    return destination / path


def _identifier_class() -> type:
    """Return the oracle's ``Identifier``, from fhy_core v0.1.8 once assembled."""
    identifier: type = importlib.import_module("fhy_core").Identifier
    return identifier


class _Placeholder:
    """A stand-in for a MOGA type ``extraction.py`` imports and never uses here."""


def _stub_module(name: str, **attributes: object) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    for key, value in attributes.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def _assemble_oracle(
    moga_vm_repository: Path, root: Path
) -> dict[str, types.ModuleType]:
    """Assemble the oracle under `root` and return its four modules."""
    search = _extract(moga_vm_repository, _MOGA_VM_COMMIT, _SEARCH, root / "moga")
    fhy_core = _extract(
        _REPOSITORY_ROOT, _FHY_CORE_TAG, "src/fhy_core", root / "fhy_core"
    )
    sys.path.insert(0, str(fhy_core.parent))
    for name in (
        "moga_vm",
        "moga_vm.cir.lowering",
        "moga_vm.cir.lowering.default",
        "moga_vm.cir.lowering.default.memory",
    ):
        _stub_module(name)
    _stub_module("moga", MOGA=_Placeholder)
    _stub_module("moga_vm.cir", NestedWalkVertex=_Placeholder)
    _stub_module(
        "moga_vm.cir.lowering.default.memory.tile_policy",
        CapacityAwareTileArraysPolicy=_Placeholder,
    )
    _stub_module("moga_vm.cir.program", Module=_Placeholder)
    _stub_module(
        "moga_vm.cir.space",
        ArrayTileKnob=_Placeholder,
        RealizationOption=_Placeholder,
        TableEntry=_Placeholder,
    )
    package = _stub_module("moga_vm.cir.lowering.search")
    package.__path__ = [str(search)]
    modules = {}
    for short in _MODULES:
        full = f"moga_vm.cir.lowering.search.{short}"
        spec = importlib.util.spec_from_file_location(full, search / f"{short}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[full] = module
        spec.loader.exec_module(module)
        modules[short] = module
    return modules


# ---------------------------------------------------------------------------
# Values: the corpus writes a value as {"int": n}, {"str": s}, {"bool": b}
# or {"label": i}, the label an index into the case's identifiers.
# ---------------------------------------------------------------------------


def _python_value(member: Json, labels: list[Any]) -> Any:
    ((kind, payload),) = member.items()
    return labels[payload] if kind == "label" else payload


def _strict_key(member: Json) -> tuple[str, Any]:
    ((kind, payload),) = member.items()
    return kind, payload


# ---------------------------------------------------------------------------
# Domains
# ---------------------------------------------------------------------------


def _record_choice(
    decisions: types.ModuleType, values: list[Json], probes: list[Json]
) -> Json:
    keys = [_strict_key(value) for value in values]
    port: Json
    if not values:
        port = {"outcome": "refused", "error": "empty_choice"}
    elif len(set(keys)) != len(keys):
        port = {"outcome": "refused", "error": "repeated_value"}
    else:
        port = {
            "outcome": "built",
            "cardinality": len(values),
            "coordinates": list(range(len(values))),
            "admits": [_strict_key(probe) in keys for probe in probes],
        }
    try:
        domain = decisions.ChoiceDomain(
            tuple(_python_value(value, []) for value in values)
        )
    except ValueError:
        oracle: dict[str, Any] = {"outcome": "refused", "error": "empty_choice"}
    else:
        oracle = {
            "outcome": "built",
            "cardinality": domain.cardinality,
            "coordinates": [
                domain.coordinate_of(domain.value_at(index))
                for index in range(len(values))
            ],
            "admits": [domain.admits(_python_value(probe, [])) for probe in probes],
        }
    return {"oracle": oracle, "port": port}


def _record_order(decisions: types.ModuleType, size: int, rng: random.Random) -> Json:
    Identifier = _identifier_class()

    elements = tuple(Identifier(f"e{index}") for index in range(size))
    permutations = []
    for _ in range(min(6, factorial(size))):
        positions = list(range(size))
        rng.shuffle(positions)
        permutations.append(positions)
    domain = decisions.OrderDomain(elements)
    oracle = {
        "outcome": "built",
        "cardinality": domain.cardinality,
        "orderings": [
            [elements.index(element) for element in domain.value_at(tuple(positions))]
            for positions in permutations
        ],
        "round_trips": [
            list(domain.coordinate_of(domain.value_at(tuple(positions)))) == positions
            for positions in permutations
        ],
    }
    port: Json = (
        {"outcome": "refused", "error": "empty_order"}
        if size == 0
        else {
            "outcome": "built",
            "cardinality": factorial(size),
            "orderings": [list(positions) for positions in permutations],
            "round_trips": [True] * len(permutations),
        }
    )
    return {
        "spec": {"size": size, "permutations": permutations},
        "oracle": oracle,
        "port": port,
    }


def _strided_values(runs: list[list[int]]) -> list[int]:
    return [
        address for start, stop, step in runs for address in range(start, stop, step)
    ]


def _record_strided(
    decisions: types.ModuleType, runs: list[list[int]], probes: list[Any]
) -> Json:
    domain = decisions.AddressDomain(
        tuple(
            decisions.AddressInterval(start, stop, step) for start, stop, step in runs
        )
    )
    values = _strided_values(runs)
    oracle = {
        "outcome": "built",
        "cardinality": domain.cardinality,
        "values": [domain.value_at(index) for index in range(domain.cardinality)],
        "admits": [domain.admits(probe) for probe in probes],
    }
    port = {
        "outcome": "built",
        "cardinality": len(values),
        "values": values,
        "admits": [
            not isinstance(probe, bool) and isinstance(probe, int) and probe in values
            for probe in probes
        ],
    }
    return {"oracle": oracle, "port": port}


def _random_runs(rng: random.Random) -> list[list[int]]:
    runs = []
    cursor = rng.randrange(0, 16)
    for _ in range(rng.randint(1, 4)):
        step = rng.choice([1, 1, 2, 4, 8, 64])
        width = rng.randint(1, 6)
        start = cursor + rng.randrange(0, 8)
        stop = start + (width - 1) * step + 1
        runs.append([start, stop, step])
        cursor = stop + rng.randrange(0, 8)
    return runs


def _domain_cases(
    decisions: types.ModuleType, rng: random.Random, count: int
) -> list[Json]:
    cases: list[Json] = []

    def choice(
        name: str, values: list[Json], probes: list[Json], divergence: str | None = None
    ) -> None:
        answers = _record_choice(decisions, values, probes)
        cases.append(
            {
                "name": name,
                "shape": "choice",
                "spec": {"values": values, "probes": probes},
                "divergence": divergence,
                **answers,
            }
        )

    choice(
        "choice_of_three_integers",
        [{"int": 10}, {"int": 20}, {"int": 30}],
        [{"int": 20}, {"int": 99}],
    )
    choice(
        "choice_of_strings", [{"str": "a"}, {"str": "b"}], [{"str": "b"}, {"str": "z"}]
    )
    choice(
        "choice_of_one_true_and_one_point_zero",
        [{"int": 1}, {"bool": True}, {"float": 1.0}],
        [{"bool": True}, {"float": 1.0}],
        "D-SS2-3",
    )
    choice(
        "choice_of_one_admitting_true",
        [{"int": 1}, {"int": 2}],
        [{"bool": True}],
        "D-SS2-3",
    )
    choice("choice_with_a_repeat", [{"str": "x"}, {"str": "x"}], [], "D-SS2-3")
    choice("choice_of_nothing", [], [])
    for index in range(count):
        size = rng.randint(1, 5)
        values = [{"int": value} for value in rng.sample(range(-8, 9), size)]
        probes = [{"int": rng.randrange(-10, 11)} for _ in range(3)]
        choice(f"random_choice_{index}", values, probes)

    for size in [0, 1, 2, 3, 4, 5]:
        answers = _record_order(decisions, size, rng)
        cases.append(
            {
                "name": f"order_of_{size}",
                "shape": "order",
                "divergence": "D-SS2-7" if size == 0 else None,
                **answers,
            }
        )

    fixed_runs = [
        ("two_unequal_runs", [[0, 3, 1], [100, 105, 1]]),
        ("strided_banks", [[0, 100, 32], [256, 300, 32]]),
        ("adjacent_runs", [[0, 4, 1], [4, 8, 1]]),
    ]
    random_runs = [
        (f"random_runs_{index}", _random_runs(rng)) for index in range(count)
    ]
    for name, runs in fixed_runs + random_runs:
        integers = _strided_values(runs)
        run_probes: list[Any] = [integers[0], integers[-1], integers[-1] + 1, -1, True]
        run_probes += [rng.randrange(0, integers[-1] + 2) for _ in range(3)]
        cases.append(
            {
                "name": name,
                "shape": "strided",
                "spec": {"runs": runs, "probes": run_probes},
                "divergence": None,
                **_record_strided(decisions, runs, run_probes),
            }
        )
    return cases


# ---------------------------------------------------------------------------
# Replays
# ---------------------------------------------------------------------------


def _build_domain(decisions: types.ModuleType, spec: Json, labels: list[Any]) -> Any:
    if spec["shape"] == "choice":
        return decisions.ChoiceDomain(
            tuple(_python_value(value, labels) for value in spec["values"])
        )
    return decisions.AddressDomain(
        tuple(decisions.AddressInterval(*run) for run in spec["runs"])
    )


def _port_signature(spec: Json) -> tuple[Any, ...]:
    """Return what the port's signature keeps of a domain description."""
    if spec["shape"] == "choice":
        return (
            "choice",
            tuple(
                "identifier" if "label" in value else _strict_key(value)
                for value in spec["values"]
            ),
        )
    return ("strided", tuple(tuple(run) for run in spec["runs"]))


def _record_replay(
    modules: dict[str, types.ModuleType], recorded: Json, offered: Json, coordinate: int
) -> Json:
    Identifier = _identifier_class()

    decisions, oracle_module = modules["decisions"], modules["oracle"]
    kind = decisions.DecisionKind.OPTION
    recorded_domain = _build_domain(
        decisions, recorded, [Identifier(f"r{index}") for index in range(8)]
    )
    offered_domain = _build_domain(
        decisions, offered, [Identifier(f"o{index}") for index in range(8)]
    )
    record = decisions.RecordedDecision(
        decisions.Decision(kind=kind, subject=Identifier("s"), domain=recorded_domain),
        coordinate,
        recorded_domain.value_at(coordinate),
    )
    replay = oracle_module.ReplayOracle(decisions.SearchPoint.from_records([record]))
    try:
        replay.decide(
            decisions.Decision(
                kind=kind, subject=Identifier("s"), domain=offered_domain
            )
        )
    except modules["errors"].SearchPointMismatchError:
        oracle: Json = {"outcome": "refused"}
    else:
        oracle = {"outcome": "replayed", "coordinate": coordinate}
    port: Json = (
        {"outcome": "replayed", "coordinate": coordinate}
        if _port_signature(recorded) == _port_signature(offered)
        else {"outcome": "refused"}
    )
    return {"oracle": oracle, "port": port}


def _replay_cases(
    modules: dict[str, types.ModuleType], rng: random.Random, count: int
) -> list[Json]:
    def ints(values: list[int]) -> Json:
        return {"shape": "choice", "values": [{"int": value} for value in values]}

    def labels(size: int) -> Json:
        return {
            "shape": "choice",
            "values": [{"label": index} for index in range(size)],
        }

    def runs(*spans: tuple[int, int]) -> Json:
        return {"shape": "strided", "runs": [[start, stop, 1] for start, stop in spans]}

    described = [
        ("same_integers", ints([1, 2, 3]), ints([1, 2, 3]), None),
        ("reordered_integers", ints([1, 2, 3]), ints([3, 2, 1]), "D-SS2-1"),
        ("other_integers_of_one_size", ints([1, 2]), ints([1, 4]), "D-SS2-1"),
        ("fewer_integers", ints([1, 2, 3]), ints([1, 2]), None),
        ("fresh_labels_of_one_size", labels(4), labels(4), None),
        ("fresh_labels_of_another_size", labels(4), labels(3), None),
        ("same_run", runs((0, 64)), runs((0, 64)), None),
        ("moved_run_of_one_size", runs((0, 64)), runs((64, 128)), "D-SS2-1"),
        ("run_of_another_size", runs((0, 64)), runs((0, 32)), None),
        ("split_runs_of_one_size", runs((0, 32), (64, 96)), runs((0, 64)), "D-SS2-1"),
        ("another_shape", ints([0, 1, 2]), runs((0, 3)), None),
    ]
    for index in range(count):
        size = rng.randint(1, 6)
        start = rng.randrange(0, 32)
        moved = start + rng.choice([0, 0, size, 3])
        described.append(
            (
                f"random_run_{index}",
                runs((start, start + size)),
                runs((moved, moved + size)),
                None if moved == start else "D-SS2-1",
            )
        )
    cases = []
    for name, recorded, offered, divergence in described:
        cardinality = (
            len(recorded["values"])
            if recorded["shape"] == "choice"
            else len(_strided_values(recorded["runs"]))
        )
        coordinate = rng.randrange(cardinality)
        answers = _record_replay(modules, recorded, offered, coordinate)
        cases.append(
            {
                "name": name,
                "recorded": recorded,
                "offered": offered,
                "coordinate": coordinate,
                "divergence": divergence,
                **answers,
            }
        )
    return cases


def _stream_cases(modules: dict[str, types.ModuleType]) -> list[Json]:
    Identifier = _identifier_class()

    decisions, oracle_module, errors = (
        modules["decisions"],
        modules["oracle"],
        modules["errors"],
    )
    kind = decisions.DecisionKind.ADDRESS
    domain = decisions.ChoiceDomain((0, 1, 2))
    cases = []
    for recorded, asked in [(3, 3), (3, 4), (3, 1), (0, 1), (2, 0)]:
        records = [
            decisions.RecordedDecision.of(
                decisions.Decision(kind=kind, subject=Identifier("s"), domain=domain),
                index % 3,
            )
            for index in range(recorded)
        ]
        replay = oracle_module.ReplayOracle(decisions.SearchPoint.from_records(records))
        refused_at = None
        for position in range(asked):
            try:
                replay.decide(
                    decisions.Decision(
                        kind=kind, subject=Identifier("s"), domain=domain
                    )
                )
            except errors.SearchPointMismatchError:
                refused_at = position
                break
        oracle = {
            "refused_at": refused_at,
            "unconsumed_at_finish": None,
        }
        port: Json = {
            "refused_at": recorded if asked > recorded else None,
            "unconsumed_at_finish": asked if asked < recorded else None,
        }
        cases.append(
            {
                "name": f"recorded_{recorded}_asked_{asked}",
                "recorded": recorded,
                "asked": asked,
                "divergence": "D-SS2-2" if asked < recorded else None,
                "oracle": oracle,
                "port": port,
            }
        )
    return cases


# ---------------------------------------------------------------------------
# Extractions
# ---------------------------------------------------------------------------


def _extraction_cases(
    modules: dict[str, types.ModuleType], rng: random.Random, count: int
) -> list[Json]:
    Identifier = _identifier_class()

    decisions, extraction = modules["decisions"], modules["extraction"]
    cases = []
    for index in range(count):
        entries = []
        for _ in range(rng.randint(1, 3)):
            options = []
            for _ in range(rng.randint(1, 4)):
                axes = []
                for _ in range(rng.randint(0, 3)):
                    if rng.random() < _CHOICE_AXIS_PROBABILITY:
                        axes.append({"choice": rng.randint(1, 4)})
                    else:
                        axes.append({"order": rng.randint(1, 3)})
                options.append(axes)
            entries.append(options)
        axes = []
        for options in entries:
            option_subject = Identifier("entry")
            option_key = (decisions.DecisionKind.OPTION, option_subject)
            axes.append(
                extraction.SearchAxis(
                    kind=decisions.DecisionKind.OPTION,
                    subject=option_subject,
                    domain=decisions.ChoiceDomain(tuple(range(len(options)))),
                )
            )
            for position, option_axes in enumerate(options):
                for axis in option_axes:
                    if "choice" in axis:
                        kind = decisions.DecisionKind.TILE
                        domain = decisions.ChoiceDomain(tuple(range(axis["choice"])))
                    else:
                        kind = decisions.DecisionKind.WALK_ORDER
                        domain = decisions.OrderDomain(
                            tuple(Identifier("level") for _ in range(axis["order"]))
                        )
                    axes.append(
                        extraction.SearchAxis(
                            kind=kind,
                            subject=Identifier("axis"),
                            domain=domain,
                            condition=extraction.AxisCondition(option_key, position),
                        )
                    )
        oracle = extraction.ExtractedSearchSpace(tuple(axes)).cardinality()
        port = prod(
            sum(
                prod(
                    axis["choice"] if "choice" in axis else factorial(axis["order"])
                    for axis in option_axes
                )
                for option_axes in options
            )
            for options in entries
        )
        cases.append(
            {
                "name": f"extraction_{index}",
                "entries": entries,
                "divergence": None,
                "oracle": {"cardinality": oracle},
                "port": {"cardinality": port},
            }
        )
    return cases


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def _check(cases: list[Json]) -> None:
    """Stop when a case's answers differ without a divergence, or agree with one."""
    for case in cases:
        agree = case["oracle"] == case["port"]
        if not agree and case["divergence"] is None:
            raise SystemExit(f"untagged disagreement in {case['name']}: {case}")
        if agree and case["divergence"] is not None:
            raise SystemExit(f"the tagged case {case['name']} agrees: {case}")


def main() -> None:
    """Record the corpus."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--moga-vm-repo", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument("--random-count", type=int, default=_DEFAULT_RANDOM_COUNT)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        modules = _assemble_oracle(arguments.moga_vm_repo.resolve(), Path(directory))
        rng = random.Random(arguments.seed)
        document = {
            "domains": _domain_cases(modules["decisions"], rng, arguments.random_count),
            "replays": _replay_cases(modules, rng, arguments.random_count),
            "streams": _stream_cases(modules),
            "extractions": _extraction_cases(modules, rng, arguments.random_count),
        }
    for family in document.values():
        _check(family)
    provenance = build_provenance(_REPOSITORY_ROOT, GENERATOR_COMMAND)
    provenance["oracle"] = ORACLE
    provenance["seed"] = arguments.seed
    provenance["random_count"] = arguments.random_count
    write_document(arguments.output, {"provenance": provenance, **document})


if __name__ == "__main__":
    importlib.invalidate_caches()
    main()
