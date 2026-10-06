r"""Record the search-space golden corpus from the MOGA-VM oracle.

Each case describes one or two decision points once: labels by their
indices into the case's pool of names, knobs over categorical params
(integers, Booleans and labels as categories, with an optional in-set
constraint), and an optional selection. The recorder builds each
description as a MOGA-VM ``DecisionPoint`` on fhy_core v0.1.8 (the
oracle) and records whether it constructs and, for a pair, its
structural and alpha verdicts in both directions. Beside them it writes
what the port answers by its own rules: the point as a ``Space`` holding
one ``Choice`` of ``PlainAlternative``s and a ``Configuration`` of its
selection. A case whose answers differ carries the divergence that
explains it; any other difference stops the recorder for review. For each
point the port builds, the case also holds the V2 texts of the space and
the configuration, with every label at a fixed id.
``tests/it/search_space_golden.rs`` replays the corpus through the Rust port.

The divergences a case can carry:

- ``constraints-compared-under-variable-frame``: a categorical knob's constraint is
  compared under its param's variable
- ``identifier-members-never-captured``: a free identifier member never matches a
  bound name
- ``type-strict-values``: ``1`` and ``True`` are different categories
- ``space-and-configuration-checks``: refuses a knob assigned twice and a value
  outside its domain
- ``empty-choice-refused``: a choice needs one or more alternatives
- ``configuration-compared-with-its-space``: a configuration is compared with its
  space, not on its own
- ``names-unique-space-wide``: refuses a space repeating a name

The oracle is MOGA-VM's ``moga_vm.cir.space`` at ``3d93ba3`` (its
``origin/dev`` when the audit ran) on fhy_core v0.1.8, the version MOGA-VM
pins. The recorder assembles it in a temporary directory, so nothing of it
is installed or committed: ``git archive`` of ``src/moga_vm/cir/space`` at
``3d93ba3`` from the MOGA-VM checkout ``--moga-vm-repo`` names, ``git
archive`` of ``src/fhy_core`` at this repository's tag ``v0.1.8``, and the
audit's stand-ins for the rest of ``moga_vm.cir`` and for ``moga``, which
are written below (``_STAND_INS``): an empty hardware DFG equivalent only to
itself, and the enums and aliases the space package imports. fhy_core
v0.1.8 imports ``networkx``, which ``--with networkx`` provides.

The oracle is frozen, so the committed corpus is a fixed regression
corpus: this recorder is not one of the ``generate_*.py`` generators that
the drift check and the ``golden_expanded`` session rerun, and CI only
replays the corpus. Run it by hand from the repository root::

    uv run --no-sync --with networkx python \\
        rust/fhy-core/tests/golden/record_search_space_cases.py \\
        --moga-vm-repo <MOGA-VM checkout>

This overwrites ``rust/fhy-core/tests/golden/search_space_cases.json``. The
expanded corpus is ``--seed 7 --random-count 2000 --output <file>``, which
the ignored test ``search_space_golden::every_expanded_case_replays_as_recorded``
replays from the file ``FHY_SEARCH_SPACE_CORPUS`` names.
"""

from __future__ import annotations

import argparse
import importlib
import io
import json
import random
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from _golden_support import build_provenance, write_document

GENERATOR_COMMAND = (
    "uv run --no-sync --with networkx python"
    " rust/fhy-core/tests/golden/record_search_space_cases.py"
    " --moga-vm-repo <MOGA-VM checkout>"
)
ORACLE = {"moga_vm": "origin/dev 3d93ba3", "fhy_core": "v0.1.8 (ad9a311)"}
# The MOGA-VM commit and the fhy_core tag the oracle is built from.
_MOGA_VM_COMMIT = "3d93ba3"
_FHY_CORE_TAG = "v0.1.8"
# The audit's stand-ins, by path in the oracle's import root: the parts of
# `moga_vm` other than `moga_vm.cir.space`, and of `moga`, that the space
# package imports.
_STAND_INS = {
    "moga_vm/__init__.py": '"""Oracle root: moga_vm.cir.space is MOGA-VM code."""\n',
    "moga_vm/cir/__init__.py": (
        '"""Oracle moga_vm.cir: the space package over stand-in IR."""\n'
        "from .ir import HardwareDFG\n"
        "from .space import *  # noqa: F403\n"
        "from .space import __all__ as _space_all\n"
        '__all__ = ["HardwareDFG", *_space_all]\n'
    ),
    "moga_vm/cir/ir/__init__.py": (
        "from .immutable_dfg import HardwareDFG, ImmutableHardwareDFG\n"
        '__all__ = ["HardwareDFG", "ImmutableHardwareDFG"]\n'
    ),
    "moga_vm/cir/ir/immutable_dfg.py": (
        "from dataclasses import dataclass\n"
        "import fhy_core\n"
        "@dataclass(frozen=True, eq=False)\n"
        "class ImmutableHardwareDFG(fhy_core.traits.FrozenMixin):\n"
        "    def iter_in_topological_order(self):\n"
        "        return iter(())\n"
        "    def is_structurally_equivalent(self, other):\n"
        "        return other is self\n"
        "    def is_alpha_equivalent_under(self, other, renaming):\n"
        "        return other is self\n"
        "    def has_vertex(self, vertex_id):\n"
        "        return False\n"
        "class HardwareDFG:\n"
        "    def freeze(self):\n"
        "        return ImmutableHardwareDFG()\n"
        "    def has_vertex(self, vertex_id):\n"
        "        return False\n"
    ),
    "moga_vm/cir/ir/ports.py": (
        "from enum import Enum\n"
        "class PortRole(str, Enum):\n"
        '    INPUT = "input"\n'
        '    OUTPUT = "output"\n'
    ),
    "moga_vm/cir/ir/vertex.py": "class NestedWalkVertex:\n    pass\n",
    "moga_vm/cir/ir/memory.py": (
        "from enum import Enum\n"
        "class ArrayLayoutKind(str, Enum):\n"
        '    ROW_MAJOR = "row_major"\n'
    ),
    "moga_vm/cir/ir/alias.py": (
        "from fhy_core import Identifier\nVertexId = Identifier\n"
    ),
    "moga/__init__.py": "",
    "moga/alias.py": (
        "from fhy_core import Identifier\n"
        "NamespaceName = Identifier\n"
        "SymbolName = Identifier\n"
    ),
    "moga/abstraction/__init__.py": "",
    "moga/abstraction/namespace/__init__.py": "",
    "moga/abstraction/namespace/addressing.py": (
        "from dataclasses import dataclass\n"
        "@dataclass(frozen=True)\n"
        "class WordIndexedAddressingScheme:\n"
        "    word_count: int\n"
        "    word_width_in_bits: int\n"
        "    @property\n"
        "    def size_in_bytes(self):\n"
        "        return self.word_count * self.word_width_in_bits // 8\n"
    ),
}
_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_OUTPUT = Path(__file__).with_name("search_space_cases.json")
_DEFAULT_SEED = 0
_DEFAULT_RANDOM_COUNT = 60
# The fixed id of label 0 of case 0, and the ids each case's pool spans.
_ID_BASE = 70_000
_ID_STRIDE = 64
# How often a random draw selects an option, as the audit's sweep does.
_SELECTION_PROBABILITY = 0.6
# The rank of each member kind in the canonical member order.
_KIND_RANKS = {"bool": 0, "identifier": 3, "int": 4}

Member = dict[str, Any]
Point = dict[str, Any]


# ---------------------------------------------------------------------------
# Descriptions
# ---------------------------------------------------------------------------


@dataclass
class Case:
    """One case: its pool of labels and one or two points."""

    name: str
    origin: str
    labels: list[str]
    left: Point
    right: Point | None = None
    divergence: str | None = None
    note: str = ""
    record: dict[str, Any] = field(default_factory=dict)


def _knob(
    name: int, param: int, categories: list[Member], kept: list[Member] | None = None
) -> dict[str, Any]:
    return {"name": name, "param": param, "categories": categories, "kept": kept}


def _option(name: int, knobs: list[dict[str, Any]]) -> dict[str, Any]:
    return {"name": name, "knobs": knobs}


def _point(
    space: int,
    choice: int,
    options: list[dict[str, Any]],
    selection: dict[str, Any] | None = None,
) -> Point:
    return {
        "space": space,
        "choice": choice,
        "options": options,
        "selection": selection,
    }


def _selection(
    option: int, values: list[tuple[int, Member]], status: str = "selected"
) -> dict[str, Any]:
    return {
        "option": option,
        "values": [[label, value] for label, value in values],
        "status": status,
    }


def _int(value: int) -> Member:
    return {"int": value}


def _bool(value: bool) -> Member:
    return {"bool": value}


def _label(index: int) -> Member:
    return {"label": index}


# ---------------------------------------------------------------------------
# The probes
# ---------------------------------------------------------------------------


def _probes() -> list[Case]:
    """Return the audit's counterexamples and validation probes as cases."""
    one_two = [_int(1), _int(2)]
    cases: list[Case] = []
    # A1: the right side reuses one name for two options.
    cases.append(
        Case(
            "a1_one_name_for_two_alternatives",
            "probe",
            ["L_s", "L", "A", "B", "R_s", "R", "S"],
            _point(0, 1, [_option(2, []), _option(3, [])]),
            _point(4, 5, [_option(6, []), _option(6, [])]),
            "names-unique-space-wide",
            "the oracle answers asymmetrically; the port refuses the repeated name",
        )
    )
    # A1b: one knob listed twice in one option.
    cases.append(
        Case(
            "a1b_one_knob_twice_in_an_option",
            "probe",
            ["L_s", "L", "o", "k", "p", "R_s", "R", "o2", "k1", "k2", "q1", "q2"],
            _point(0, 1, [_option(2, [_knob(3, 4, one_two), _knob(3, 4, one_two)])]),
            _point(5, 6, [_option(7, [_knob(8, 10, one_two), _knob(9, 11, one_two)])]),
            "names-unique-space-wide",
            "the oracle answers asymmetrically; the port refuses the repeated name",
        )
    )
    # A2: a knob shared by two options, selected through the wide frame.
    cases.append(
        Case(
            "a2_one_knob_in_two_alternatives",
            "probe",
            [
                "L2_s",
                "L2",
                "la",
                "lb",
                "ks",
                "p",
                "R2_s",
                "R2",
                "ra",
                "rb",
                "k1",
                "k2",
                "q1",
                "q2",
            ],
            _point(
                0,
                1,
                [
                    _option(2, [_knob(4, 5, one_two)]),
                    _option(3, [_knob(4, 5, one_two)]),
                ],
                _selection(2, [(4, _int(1))]),
            ),
            _point(
                6,
                7,
                [
                    _option(8, [_knob(10, 12, one_two)]),
                    _option(9, [_knob(11, 13, one_two)]),
                ],
                _selection(8, [(10, _int(1))]),
            ),
            "names-unique-space-wide",
            "names are unique space-wide, so one knob cannot serve two alternatives",
        )
    )
    # The A2 shape with distinct names on both sides.
    cases.append(
        Case(
            "a2_distinct_knobs_relabeled",
            "probe",
            [
                "L_s",
                "L",
                "la",
                "lb",
                "k1",
                "k2",
                "p",
                "R_s",
                "R",
                "ra",
                "rb",
                "j1",
                "j2",
            ],
            _point(
                0,
                1,
                [
                    _option(2, [_knob(4, 6, one_two)]),
                    _option(3, [_knob(5, 6, one_two)]),
                ],
                _selection(2, [(4, _int(1))]),
            ),
            _point(
                7,
                8,
                [
                    _option(9, [_knob(11, 6, one_two)]),
                    _option(10, [_knob(12, 6, one_two)]),
                ],
                _selection(9, [(11, _int(1))]),
            ),
        )
    )
    # A5: a categorical knob's constraint.
    cases.append(
        Case(
            "a5_constraint_of_a_categorical_knob",
            "probe",
            ["L_s", "L", "o", "k", "p", "R_s", "R", "o2", "k2", "q"],
            _point(0, 1, [_option(2, [_knob(3, 4, one_two)])]),
            _point(5, 6, [_option(7, [_knob(8, 9, one_two, [_int(1)])])]),
            "constraints-compared-under-variable-frame",
            "the oracle ignores a categorical knob's constraints in alpha mode",
        )
    )
    # A6: 1 and True as categories, and as values.
    cases.append(
        Case(
            "a6_one_and_true_as_categories",
            "probe",
            ["L_s", "L", "o", "k", "p", "R_s", "R", "o2", "k2", "q"],
            _point(
                0,
                1,
                [_option(2, [_knob(3, 4, [_int(1)])])],
                _selection(2, [(3, _int(1))]),
            ),
            _point(
                5,
                6,
                [_option(7, [_knob(8, 9, [_bool(True)])])],
                _selection(7, [(8, _bool(True))]),
            ),
            "type-strict-values",
            "Python's == unifies 1 and True; the port compares type-strictly",
        )
    )
    # A7: a free category captured by an option's name.
    cases.append(
        Case(
            "a7_free_category_and_bound_name",
            "probe",
            ["Lz_s", "Lz", "A", "kz", "p", "Z", "Rz_s", "Rz", "kz2", "q"],
            _point(0, 1, [_option(2, [_knob(3, 4, [_label(5)])])]),
            _point(6, 7, [_option(5, [_knob(8, 9, [_label(5)])])]),
            "identifier-members-never-captured",
            "the oracle resolves the left category on the left only, which captures it",
        )
    )
    # V1: two options with one name.
    cases.append(
        Case(
            "v1_one_name_for_two_alternatives",
            "probe",
            ["s", "d", "dup", "kx", "ky", "p"],
            _point(
                0,
                1,
                [
                    _option(2, [_knob(3, 5, one_two)]),
                    _option(2, [_knob(4, 5, one_two)]),
                ],
                _selection(2, [(3, _int(1))]),
            ),
            note=(
                "both refuse: the port the repeated name, the oracle the knob, "
                "which it looks up in the last alternative of that name"
            ),
        )
    )
    # V2: two knobs with one name in one option.
    cases.append(
        Case(
            "v2_one_name_for_two_knobs",
            "probe",
            ["s", "d", "o", "kx", "p"],
            _point(0, 1, [_option(2, [_knob(3, 4, one_two), _knob(3, 4, one_two)])]),
            divergence="names-unique-space-wide",
            note="the oracle accepts a repeated knob name",
        )
    )
    # V3: one knob assigned twice.
    cases.append(
        Case(
            "v3_one_knob_assigned_twice",
            "probe",
            ["s", "d", "o3", "kx", "p"],
            _point(
                0,
                1,
                [_option(2, [_knob(3, 4, one_two)])],
                _selection(2, [(3, _int(1)), (3, _int(2))]),
            ),
            divergence="space-and-configuration-checks",
            note="the oracle accepts conflicting assignments",
        )
    )
    # V4: a value outside the knob's domain.
    cases.append(
        Case(
            "v4_value_outside_the_domain",
            "probe",
            ["s", "d", "o3", "kx", "p"],
            _point(
                0,
                1,
                [_option(2, [_knob(3, 4, one_two)])],
                _selection(2, [(3, _int(99))]),
            ),
            divergence="space-and-configuration-checks",
            note="the oracle accepts an assignment from a foreign param",
        )
    )
    # V5: statuses that say nothing checkable; both sides accept.
    for status, values in (
        ("selected", []),
        ("partial", [(3, _int(1))]),
        ("partial", []),
    ):
        cases.append(
            Case(
                f"v5_{status}_with_{len(values)}_of_1_assigned",
                "probe",
                ["s", "d", "o3", "kx", "p"],
                _point(
                    0,
                    1,
                    [_option(2, [_knob(3, 4, one_two)])],
                    _selection(2, values, status),
                ),
            )
        )
    # V6: an empty decision space.
    cases.append(
        Case(
            "v6_no_alternative",
            "probe",
            ["s", "d"],
            _point(0, 1, []),
            divergence="empty-choice-refused",
            note="the oracle accepts an empty decision space",
        )
    )
    # Validation the oracle and the port agree on.
    cases.append(
        Case(
            "unknown_selected_option",
            "probe",
            ["s", "d", "o", "kx", "p", "ghost"],
            _point(0, 1, [_option(2, [_knob(3, 4, one_two)])], _selection(5, [])),
        )
    )
    cases.append(
        Case(
            "assignment_to_a_knob_of_another_option",
            "probe",
            ["s", "d", "a", "b", "ka", "kb", "p"],
            _point(
                0,
                1,
                [
                    _option(2, [_knob(4, 6, one_two)]),
                    _option(3, [_knob(5, 6, one_two)]),
                ],
                _selection(2, [(5, _int(1))]),
            ),
        )
    )
    cases.append(
        Case(
            "unselected_with_an_assignment",
            "probe",
            ["s", "d", "o", "kx", "p"],
            _point(
                0,
                1,
                [_option(2, [_knob(3, 4, one_two)])],
                _selection(2, [(3, _int(1))], "unselected"),
            ),
            divergence="configuration-compared-with-its-space",
            note=(
                "the status is gone: a configuration assigning a chosen "
                "alternative's knob is valid"
            ),
        )
    )
    return cases


# ---------------------------------------------------------------------------
# Random cases (the audit's relation sweep)
# ---------------------------------------------------------------------------


def _random_cases(seed: int, count: int) -> list[Case]:
    """Return `count` random decision points, each with a relabeled copy.

    Draws as the audit's sweep does: a small pool of names, so names repeat;
    four knobs from it over two shared params; one to three options of up to
    two of those knobs; and, three times in five, a selection of one option
    assigning 1 to a prefix of its knobs. Each draw gives two cases: its copy
    with every name replaced by a fresh one, sharing kept, and its copy with
    every occurrence replaced by a fresh one.
    """
    rng = random.Random(seed)
    params = [(0, [_int(1), _int(2)]), (1, [_int(1), _int(2), _int(3)])]
    cases: list[Case] = []
    for draw in range(count):
        pool_size = rng.choice([2, 3, 6])
        labels = ["p0", "p1", "s", "d", "s2", "d2"] + [
            f"n{index}" for index in range(pool_size)
        ]
        first_name = 6
        knob_pool = []
        for _ in range(4):
            param, categories = rng.choice(params)
            knob_pool.append(
                _knob(first_name + rng.randrange(pool_size), param, categories)
            )
        options = []
        for _ in range(rng.randint(1, 3)):
            knobs = [rng.choice(knob_pool) for _ in range(rng.randint(0, 2))]
            options.append(_option(first_name + rng.randrange(pool_size), knobs))
        selection = None
        if rng.random() < _SELECTION_PROBABILITY:
            chosen = rng.choice(options)
            assigned = chosen["knobs"][: rng.randint(0, len(chosen["knobs"]))]
            selection = _selection(
                chosen["name"], [(knob["name"], _int(1)) for knob in assigned]
            )
        left = _point(2, 3, options, selection)
        for shared in (True, False):
            case_labels = list(labels)
            right = _relabel(left, case_labels, shared)
            cases.append(
                Case(
                    f"random_{draw}_{'shared' if shared else 'unshared'}",
                    "random",
                    case_labels,
                    left,
                    right,
                )
            )
    return cases


def _relabel(left: Point, labels: list[str], shared: bool) -> Point:
    """Return `left` with every name replaced by a fresh label of `labels`.

    With `shared`, one name gets one fresh label everywhere; otherwise each
    occurrence gets its own. The selection names the copies of what it
    named. The space and choice names become labels 4 and 5.
    """
    fresh: dict[int, int] = {}

    def rename(index: int) -> int:
        if shared and index in fresh:
            return fresh[index]
        labels.append(labels[index] + "'")
        fresh[index] = len(labels) - 1
        return fresh[index]

    options = []
    copied_knobs: list[list[int]] = []
    for option in left["options"]:
        knobs = [dict(knob, name=rename(knob["name"])) for knob in option["knobs"]]
        copied_knobs.append([knob["name"] for knob in knobs])
        options.append(_option(rename(option["name"]), knobs))
    selection = None
    if left["selection"] is not None:
        # Two options may share a name; the chosen one is the first of that
        # name that holds every knob the selection assigns.
        assigned = [label for label, _ in left["selection"]["values"]]
        position = next(
            index
            for index, option in enumerate(left["options"])
            if option["name"] == left["selection"]["option"]
            and all(
                label in [knob["name"] for knob in option["knobs"]]
                for label in assigned
            )
        )
        knob_names = [knob["name"] for knob in left["options"][position]["knobs"]]
        values = [
            (copied_knobs[position][knob_names.index(label)], value)
            for label, value in left["selection"]["values"]
        ]
        selection = _selection(options[position]["name"], values)
    return _point(4, 5, options, selection)


# ---------------------------------------------------------------------------
# The oracle
# ---------------------------------------------------------------------------


class Oracle:
    """MOGA-VM's decision points on fhy_core v0.1.8."""

    def __init__(self) -> None:
        self.fhy_core: Any = importlib.import_module("fhy_core")
        self.constraint: Any = importlib.import_module("fhy_core.constraint")
        self.param: Any = importlib.import_module("fhy_core.param")
        cir: Any = importlib.import_module("moga_vm.cir")
        self.space: Any = importlib.import_module("moga_vm.cir.space")
        self.realization = cir.HardwareDFG().freeze()

    def build(self, case: Case, point: Point, pool: list[Any]) -> Any:
        """Return the decision point of `point`, or None if it is refused."""
        space = self.space
        params: dict[int, Any] = {}

        def value(member: Member) -> Any:
            if "label" in member:
                return pool[member["label"]]
            return member.get("int", member.get("bool"))

        def param_of(knob: dict[str, Any]) -> Any:
            if knob["param"] not in params:
                param = self.param.create_categorical_param(
                    frozenset(value(member) for member in knob["categories"])
                )
                if knob["kept"] is not None:
                    kept = frozenset(value(member) for member in knob["kept"])
                    param = param.add_constraint(
                        self.constraint.InSetConstraint(param.variable, kept)
                    )
                params[knob["param"]] = param
            return params[knob["param"]]

        try:
            knobs_by_label: dict[int, Any] = {}
            knob_categories: dict[int, list[Member]] = {}
            options = []
            for option in point["options"]:
                knobs = []
                for knob in option["knobs"]:
                    built = space.Knob(name=pool[knob["name"]], param=param_of(knob))
                    knobs_by_label.setdefault(knob["name"], built)
                    knob_categories.setdefault(knob["name"], knob["categories"])
                    knobs.append(built)
                options.append(
                    space.RealizationOption(
                        name=pool[option["name"]],
                        knobs=tuple(knobs),
                        realization=self.realization,
                    )
                )
            selection = None
            if point["selection"] is not None:
                assignments = []
                for label, member in point["selection"]["values"]:
                    knob = knobs_by_label.get(label)
                    raw = value(member)
                    if knob is not None and member in knob_categories[label]:
                        assignments.append(knob.assign(raw))
                    else:
                        foreign = self.param.create_categorical_param(frozenset({raw}))
                        assignments.append(
                            space.KnobAssignment(pool[label], foreign.assign(raw))
                        )
                selection = space.ArraySelection(
                    selected_option_identifier=pool[point["selection"]["option"]],
                    status=space.SelectionStatus(point["selection"]["status"]),
                    knob_assignments=tuple(assignments),
                )
            return space.DecisionPoint(
                space=space.DecisionSpace(
                    name=pool[point["space"]], options=tuple(options)
                ),
                name=pool[point["choice"]],
                selection=selection,
            )
        except ValueError:
            return None

    def record(self, case: Case) -> dict[str, Any]:
        """Return the oracle's answers for `case`."""
        pool = [self.fhy_core.Identifier(label) for label in case.labels]
        left = self.build(case, case.left, pool)
        answers: dict[str, Any] = {"left": "built" if left is not None else "refused"}
        if case.right is None:
            return answers
        right = self.build(case, case.right, pool)
        answers["right"] = "built" if right is not None else "refused"
        if left is not None and right is not None:
            answers["structural"] = [
                bool(left.is_structurally_equivalent(right)),
                bool(right.is_structurally_equivalent(left)),
            ]
            answers["alpha"] = [
                bool(left.is_alpha_equivalent(right)),
                bool(right.is_alpha_equivalent(left)),
            ]
        return answers


# ---------------------------------------------------------------------------
# The port's rules
# ---------------------------------------------------------------------------


def _names(point: Point) -> list[int]:
    """Return a point's names as the port reads them, in canonical order."""
    names = [point["space"], point["choice"]]
    for option in point["options"]:
        names.append(option["name"])
        names.extend(knob["name"] for knob in option["knobs"])
    return names


def _port_outcome(point: Point) -> str:
    """Return what the port makes of `point`: built, or which part it refuses."""
    names = _names(point)
    if not point["options"] or len(set(names)) != len(names):
        return "space_refused"
    selection = point["selection"]
    if selection is None or _is_valid_selection(point, selection):
        return "built"
    return "configuration_refused"


def _is_valid_selection(point: Point, selection: dict[str, Any]) -> bool:
    """Return whether the port's configuration of `selection` is valid.

    The chosen alternative must exist, and each value must name, once, one
    of its knobs, and be a category its param keeps.
    """
    chosen = [
        option for option in point["options"] if option["name"] == selection["option"]
    ]
    if not chosen:
        return False
    knobs = {knob["name"]: knob for knob in chosen[0]["knobs"]}
    labels = [label for label, _ in selection["values"]]
    if len(set(labels)) != len(labels):
        return False
    return all(
        label in knobs
        and member in knobs[label]["categories"]
        and (knobs[label]["kept"] is None or member in knobs[label]["kept"])
        for label, member in selection["values"]
    )


def _entries(point: Point) -> dict[int, Member]:
    """Return a built point's configuration entries, by decision label."""
    selection = point["selection"]
    if selection is None:
        return {}
    entries: dict[int, Member] = {point["choice"]: _label(selection["option"])}
    entries.update(dict(selection["values"]))
    return entries


def _shape(point: Point) -> list[int]:
    return [len(option["knobs"]) for option in point["options"]]


def _is_structural(left: Point, right: Point) -> bool:
    """Return whether two built points are structurally equivalent in the port."""
    if (left["space"], left["choice"], _shape(left)) != (
        right["space"],
        right["choice"],
        _shape(right),
    ):
        return False
    for left_option, right_option in zip(
        left["options"], right["options"], strict=True
    ):
        if left_option["name"] != right_option["name"]:
            return False
        for left_knob, right_knob in zip(
            left_option["knobs"], right_option["knobs"], strict=True
        ):
            if (
                left_knob["name"] != right_knob["name"]
                or left_knob["param"] != right_knob["param"]
            ):
                return False
            if _as_set(left_knob["categories"]) != _as_set(right_knob["categories"]):
                return False
            if _as_optional_set(left_knob["kept"]) != _as_optional_set(
                right_knob["kept"]
            ):
                return False
    return _entries(left) == _entries(right)


def _as_set(members: list[Member]) -> set[str]:
    return {json.dumps(member, sort_keys=True) for member in members}


def _as_optional_set(members: list[Member] | None) -> set[str] | None:
    return None if members is None else _as_set(members)


def _is_alpha(left: Point, right: Point) -> bool:
    """Return whether two built points are alpha-equivalent in the port."""
    if _shape(left) != _shape(right):
        return False
    pairs = dict(zip(_names(left), _names(right), strict=True))
    images = set(pairs.values())

    def corresponds(left_member: Member, right_member: Member) -> bool:
        if "label" not in left_member or "label" not in right_member:
            return left_member == right_member
        left_label, right_label = left_member["label"], right_member["label"]
        if left_label in pairs:
            return bool(pairs[left_label] == right_label)
        return right_label not in images and left_label == right_label

    def bijective(
        left_members: list[Member] | None, right_members: list[Member] | None
    ) -> bool:
        if left_members is None or right_members is None:
            return left_members is None and right_members is None
        return len(left_members) == len(right_members) and all(
            any(corresponds(member, other) for other in right_members)
            for member in left_members
        )

    for left_option, right_option in zip(
        left["options"], right["options"], strict=True
    ):
        for left_knob, right_knob in zip(
            left_option["knobs"], right_option["knobs"], strict=True
        ):
            if not bijective(left_knob["categories"], right_knob["categories"]):
                return False
            if not bijective(left_knob["kept"], right_knob["kept"]):
                return False
    left_entries, right_entries = _entries(left), _entries(right)
    if {pairs[label] for label in left_entries} != set(right_entries):
        return False
    return all(
        corresponds(member, right_entries[pairs[label]])
        for label, member in left_entries.items()
    )


def _port_record(case: Case) -> dict[str, Any]:
    """Return the port's answers for `case`, by its rules."""
    answers: dict[str, Any] = {"left": _port_outcome(case.left)}
    if case.right is None:
        return answers
    answers["right"] = _port_outcome(case.right)
    if answers["left"] == "built" and answers["right"] == "built":
        answers["structural"] = [
            _is_structural(case.left, case.right),
            _is_structural(case.right, case.left),
        ]
        answers["alpha"] = [
            _is_alpha(case.left, case.right),
            _is_alpha(case.right, case.left),
        ]
    return answers


def _agree(oracle: dict[str, Any], port: dict[str, Any]) -> bool:
    """Return whether the oracle and the port answer `case` alike."""
    built = {
        key: "built" if port[key] == "built" else "refused"
        for key in ("left", "right")
        if key in port
    }
    return all(oracle[key] == built[key] for key in built) and all(
        oracle.get(key) == port.get(key) for key in ("structural", "alpha")
    )


def _random_divergence(
    case: Case, oracle: dict[str, Any], port: dict[str, Any]
) -> str | None:
    """Return the divergence that explains a random case's differing answers."""
    for side in ("left", "right"):
        point = case.left if side == "left" else case.right
        if (
            point is not None
            and oracle.get(side) == "built"
            and port.get(side) != "built"
        ):
            names = _names(point)
            return (
                "names-unique-space-wide"
                if len(set(names)) != len(names)
                else "space-and-configuration-checks"
            )
    return None


# ---------------------------------------------------------------------------
# The port's V2 texts
# ---------------------------------------------------------------------------


def _identifier(case_index: int, labels: list[str], index: int) -> dict[str, Any]:
    return {
        "id": _ID_BASE + _ID_STRIDE * case_index + index,
        "name_hint": labels[index],
    }


def _value(case_index: int, labels: list[str], member: Member) -> dict[str, Any]:
    if "label" in member:
        return {"identifier": _identifier(case_index, labels, member["label"])}
    if "bool" in member:
        return {"bool": member["bool"]}
    return {"int": str(member["int"])}


def _sorted_values(
    case_index: int, labels: list[str], members: list[Member]
) -> list[Any]:
    """Return the values of `members` in the canonical member order."""

    def key(member: Member) -> tuple[int, int]:
        if "label" in member:
            return (
                _KIND_RANKS["identifier"],
                _ID_BASE + _ID_STRIDE * case_index + member["label"],
            )
        if "bool" in member:
            return (_KIND_RANKS["bool"], int(member["bool"]))
        return (_KIND_RANKS["int"], member["int"])

    return [_value(case_index, labels, member) for member in sorted(members, key=key)]


def _space_text(
    case_index: int, labels: list[str], point: Point
) -> tuple[dict[str, Any], str]:
    """Return the port's space of a built point, as data and as V2 text."""

    def identifier(index: int) -> dict[str, Any]:
        return _identifier(case_index, labels, index)

    alternatives = []
    for option in point["options"]:
        variables = []
        for knob in option["knobs"]:
            variable = identifier(knob["param"])
            constraints = []
            if knob["kept"] is not None:
                constraints.append(
                    {
                        "in_set": {
                            "variable": variable,
                            "values": _sorted_values(case_index, labels, knob["kept"]),
                        }
                    }
                )
            param = {
                "domain": {
                    "categorical": {
                        "categories": _sorted_values(
                            case_index, labels, knob["categories"]
                        )
                    }
                },
                "variable": variable,
                "constraint_system": {"constraints": constraints},
            }
            variables.append(
                {
                    "plain": {
                        "identifier": identifier(knob["name"]),
                        "param": param,
                        "notes": [],
                    }
                }
            )
        alternatives.append(
            {
                "plain": {
                    "identifier": identifier(option["name"]),
                    "variables": variables,
                    "choices": [],
                    "notes": [],
                }
            }
        )
    space = {
        "identifier": identifier(point["space"]),
        "variables": [],
        "choices": [
            {
                "identifier": identifier(point["choice"]),
                "alternatives": alternatives,
                "notes": [],
            }
        ],
        "conditions": [],
        "forbidden": [],
        "notes": [],
    }
    return space, _compact(space)


def _configuration_text(
    case_index: int, labels: list[str], point: Point, space: dict[str, Any]
) -> str:
    """Return the V2 text of a built point's configuration."""
    entries = []
    selection = point["selection"]
    if selection is not None:
        entries.append(
            {
                "name": _identifier(case_index, labels, point["choice"]),
                "value": {
                    "identifier": _identifier(case_index, labels, selection["option"])
                },
            }
        )
        values = dict(selection["values"])
        chosen = next(
            option
            for option in point["options"]
            if option["name"] == selection["option"]
        )
        for knob in chosen["knobs"]:
            if knob["name"] in values:
                entries.append(
                    {
                        "name": _identifier(case_index, labels, knob["name"]),
                        "value": _value(case_index, labels, values[knob["name"]]),
                    }
                )
    return _compact({"space": space, "entries": entries})


def _compact(document: Any) -> str:
    return json.dumps(document, separators=(",", ":"), ensure_ascii=False)


# ---------------------------------------------------------------------------
# The corpus
# ---------------------------------------------------------------------------


def _record_case(case_index: int, case: Case, oracle: Oracle) -> dict[str, Any]:
    """Return `case`'s corpus entry, checking that its answers are explained."""
    answers = oracle.record(case)
    port = _port_record(case)
    divergence = case.divergence
    if case.origin == "random" and divergence is None:
        divergence = _random_divergence(case, answers, port)
    agrees = _agree(answers, port)
    if agrees == (divergence is not None):
        raise SystemExit(
            f"case {case.name}: oracle {answers} and port {port} "
            f"{'agree' if agrees else 'differ'}, but the divergence is {divergence!r}"
        )
    wire: dict[str, Any] = {}
    for side, point, outcome in (
        ("left", case.left, port["left"]),
        ("right", case.right, port.get("right")),
    ):
        if point is not None and outcome == "built":
            space, space_text = _space_text(case_index, case.labels, point)
            wire[f"{side}_space"] = space_text
            wire[f"{side}_configuration"] = _configuration_text(
                case_index, case.labels, point, space
            )
    return {
        "name": case.name,
        "origin": case.origin,
        "divergence": divergence,
        "note": case.note,
        "id_base": _ID_BASE + _ID_STRIDE * case_index,
        "labels": case.labels,
        "left": case.left,
        "right": case.right,
        "oracle": answers,
        "port": port,
        "wire": wire,
    }


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


def _assemble_oracle(moga_vm_repository: Path, root: Path) -> None:
    """Assemble the oracle's import root under `root` and put it on the path."""
    space = _extract(moga_vm_repository, _MOGA_VM_COMMIT, "src/moga_vm/cir/space", root)
    imports = root / "oracle"
    (imports / "moga_vm" / "cir").mkdir(parents=True)
    space.rename(imports / "moga_vm" / "cir" / "space")
    for relative, text in _STAND_INS.items():
        target = imports / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    fhy_core = _extract(
        _REPOSITORY_ROOT, _FHY_CORE_TAG, "src/fhy_core", root / "fhy_core"
    )
    sys.path[:0] = [str(imports), str(fhy_core.parent)]


def main() -> None:
    """Record the corpus."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--moga-vm-repo", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument("--random-count", type=int, default=_DEFAULT_RANDOM_COUNT)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        _assemble_oracle(arguments.moga_vm_repo.resolve(), Path(directory))
        _record(arguments)


def _record(arguments: argparse.Namespace) -> None:
    """Record the corpus `arguments` describe with the assembled oracle."""
    oracle = Oracle()
    cases = _probes() + _random_cases(arguments.seed, arguments.random_count)
    if len(cases) * _ID_STRIDE + _ID_BASE >= 2**40:
        raise SystemExit("too many cases for the fixed ids")
    for case in cases:
        if len(case.labels) > _ID_STRIDE:
            raise SystemExit(f"case {case.name} has more labels than the id stride")
    provenance = build_provenance(_REPOSITORY_ROOT, GENERATOR_COMMAND)
    provenance["oracle"] = ORACLE
    provenance["seed"] = arguments.seed
    provenance["random_count"] = arguments.random_count
    write_document(
        arguments.output,
        {
            "provenance": provenance,
            "cases": [
                _record_case(index, case, oracle) for index, case in enumerate(cases)
            ],
        },
    )


if __name__ == "__main__":
    main()
