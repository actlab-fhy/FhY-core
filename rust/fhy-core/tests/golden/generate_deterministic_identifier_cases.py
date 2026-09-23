"""Generate golden operation scripts for the Rust deterministic-identifiers port.

Drives the real
`fhy_core.testing_patches.deterministic_identifiers_by_name_hint` oracle
through hand-picked and randomly generated scripts of `enter`, `exit` and
`new(hint)` operations, recording the id each `new` receives relative to a
per-script anchor identifier. The Rust equivalence test
(`rust/fhy-core/tests/deterministic_identifiers_equivalence.rs`) replays these scripts
against `fhy_core::testing::DeterministicIdentifierScope` and compares every
observation.

A script's `enter` and `exit` operations are LIFO: `exit` always closes the
most recently entered scope that is still open, and every script ends with
every scope closed (a script definition need not close its own scopes; this
generator appends whatever `exit`s are missing before running it). Each
script starts outside any scope by creating an anchor identifier,
`Identifier("anchor")`. The one observation recorded for each `new(hint)` is
the constructed identifier's id minus the anchor's id, which alone pins
sharing within a scope, forgetting across scopes, nesting, and the rule that
a repeated hint allocates no id.

Run from the repository root:

    uv run --no-sync python \
        rust/fhy-core/tests/golden/generate_deterministic_identifier_cases.py

This overwrites `rust/fhy-core/tests/golden/deterministic_identifier_cases.json`.
Options select a larger random corpus written elsewhere, which the ignored
expanded-corpus equivalence test replays from the file named in
`FHY_DETERMINISTIC_IDENTIFIER_CORPUS`. `uv run nox -s golden_expanded`
generates and replays an expanded corpus for every generator.
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from _golden_support import add_corpus_arguments, build_provenance, write_document

from fhy_core.identifier import Identifier
from fhy_core.testing_patches import deterministic_identifiers_by_name_hint

GENERATOR_COMMAND = (
    "uv run --no-sync python "
    "rust/fhy-core/tests/golden/generate_deterministic_identifier_cases.py"
)

# Beyond plain ASCII letters, the alphabet holds the empty hint, a
# precomposed Latin letter, and an astral-plane emoji, so the random scripts
# also key the scope's table on hints of zero, two, and four UTF-8 bytes.
_RANDOM_HINTS = ["a", "b", "c", "", "é", "😀"]
_RANDOM_SEED = 20260922
_RANDOM_SCRIPT_COUNT = 130
_RANDOM_MAX_OPS = 24
_RANDOM_MAX_DEPTH = 3

# Relative weights for a random script's next operation, restricted to the
# operations feasible at the script's current nesting depth (see
# `_list_feasible_op_kinds`). Skewed toward `new` so most operations are
# actually observable creations rather than scope bookkeeping.
_OP_WEIGHTS: dict[str, int] = {"new": 60, "enter": 22, "exit": 18}


# =============================================================================
# Op builders (the input half of an operation, before it is run)
# =============================================================================


def _build_enter_op() -> dict[str, Any]:
    return {"op": "enter"}


def _build_exit_op() -> dict[str, Any]:
    return {"op": "exit"}


def _build_new_op(hint: str) -> dict[str, Any]:
    return {"op": "new", "hint": hint}


# =============================================================================
# Op execution (records the oracle's observation for one operation)
# =============================================================================


def _run_op(op: dict[str, Any], anchor_id: int) -> dict[str, Any]:
    """Run one operation against the real oracle and return its observation.

    `enter` and `exit` record nothing. `new` records the constructed
    identifier's id relative to the script's anchor id, which is the only
    thing that pins sharing, forgetting, nesting, and the no-allocation rule
    for a repeat.
    """
    kind = op["op"]
    if kind == "enter":
        deterministic_identifiers_by_name_hint.__enter__()
        return {}
    if kind == "exit":
        deterministic_identifiers_by_name_hint.__exit__(None, None, None)
        return {}
    if kind == "new":
        identifier = Identifier(op["hint"])
        return {"relative_id": identifier.id - anchor_id}
    raise ValueError(f"unknown op {kind!r}")


def _compute_net_depth(ops: Sequence[dict[str, Any]]) -> int:
    """Return the scope depth `ops` leaves open (0 if fully closed)."""
    depth = 0
    for op in ops:
        if op["op"] == "enter":
            depth += 1
        elif op["op"] == "exit":
            depth -= 1
    return depth


def _close_open_scopes(ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return `ops` followed by however many `exit`s close every open scope."""
    depth = _compute_net_depth(ops)
    return [*ops, *(_build_exit_op() for _ in range(depth))]


def _assert_no_scope_active() -> None:
    """Raise if a previous script left a deterministic-identifiers scope open.

    Every script this generator runs is closed by `_close_open_scopes`
    before execution, so this should never fire. Outside every scope two
    identifiers built with one name hint differ; a scope left open would
    make them equal and quietly contaminate every later script's anchor.
    The probe runs before the anchor is drawn, so it never shifts a
    recorded relative id.
    """
    if Identifier("scope-probe") == Identifier("scope-probe"):
        raise RuntimeError(
            "a previous script left a deterministic-identifiers scope open"
        )


def _run_script(name: str, ops: list[dict[str, Any]]) -> dict[str, Any]:
    _assert_no_scope_active()
    anchor_id = Identifier("anchor").id
    recorded_ops = []
    for op in _close_open_scopes(ops):
        expected = _run_op(op, anchor_id)
        recorded_ops.append({**op, "expected": expected})
    return {"name": name, "ops": recorded_ops}


# =============================================================================
# Hand-picked scripts
# =============================================================================


def _list_hand_picked_scripts() -> list[tuple[str, list[dict[str, Any]]]]:
    return [
        (
            "a_repeated_hint_in_one_scope_returns_the_earlier_number",
            [
                _build_enter_op(),
                _build_new_op("a"),
                _build_new_op("a"),
                _build_new_op("b"),
                _build_new_op("a"),
            ],
        ),
        (
            "exiting_a_scope_forgets_its_hints",
            [
                _build_enter_op(),
                _build_new_op("a"),
                _build_exit_op(),
                _build_enter_op(),
                _build_new_op("a"),
            ],
        ),
        (
            "nested_scopes_share_one_table",
            [
                _build_enter_op(),
                _build_new_op("a"),
                _build_enter_op(),
                _build_new_op("a"),
                _build_new_op("b"),
                _build_exit_op(),
                _build_new_op("b"),
                _build_new_op("c"),
            ],
        ),
        (
            "a_hint_first_created_inside_an_inner_scope_stays_shared_after_it_exits",
            [
                _build_enter_op(),
                _build_enter_op(),
                _build_new_op("shared"),
                _build_exit_op(),
                _build_new_op("shared"),
                _build_exit_op(),
                _build_new_op("shared"),
            ],
        ),
        (
            "out_of_scope_creations_interleave_with_scopes",
            [
                _build_new_op("a"),
                _build_enter_op(),
                _build_new_op("a"),
                _build_exit_op(),
                _build_new_op("a"),
                _build_enter_op(),
                _build_new_op("a"),
                _build_exit_op(),
                _build_new_op("a"),
            ],
        ),
        (
            "nesting_three_deep_shares_one_table_until_the_outermost_exit",
            [
                _build_enter_op(),
                _build_enter_op(),
                _build_enter_op(),
                _build_new_op("a"),
                _build_new_op("b"),
                _build_exit_op(),
                _build_new_op("a"),
                _build_exit_op(),
                _build_new_op("c"),
                _build_exit_op(),
                _build_new_op("a"),
            ],
        ),
        (
            "an_empty_scope_changes_nothing_outside_it",
            [
                _build_new_op("a"),
                _build_enter_op(),
                _build_exit_op(),
                _build_new_op("a"),
            ],
        ),
        (
            "a_script_with_no_scopes_at_all",
            [
                _build_new_op("a"),
                _build_new_op("a"),
                _build_new_op("b"),
                _build_new_op("a"),
            ],
        ),
        (
            "empty_and_non_ascii_hints_are_shared_like_any_other",
            [
                _build_enter_op(),
                _build_new_op(""),
                _build_new_op("é"),
                _build_new_op("😀"),
                _build_new_op("e"),
                _build_new_op(""),
                _build_new_op("😀"),
                _build_new_op("é"),
                _build_exit_op(),
                _build_new_op(""),
                _build_new_op("é"),
            ],
        ),
        (
            "the_anchor_hint_inside_a_scope_does_not_share_the_anchor",
            [
                _build_enter_op(),
                _build_new_op("anchor"),
                _build_enter_op(),
                _build_new_op("anchor"),
                _build_exit_op(),
                _build_new_op("anchor"),
                _build_exit_op(),
                _build_new_op("anchor"),
                _build_enter_op(),
                _build_new_op("anchor"),
            ],
        ),
    ]


# =============================================================================
# Random scripts
# =============================================================================


def _list_feasible_op_kinds(depth: int, max_depth: int) -> list[str]:
    kinds = ["new"]
    if depth < max_depth:
        kinds.append("enter")
    if depth > 0:
        kinds.append("exit")
    return kinds


def _choose_random_op(
    rng: random.Random, depth: int, max_depth: int, hints: Sequence[str]
) -> dict[str, Any]:
    feasible = _list_feasible_op_kinds(depth, max_depth)
    kind = rng.choices(feasible, weights=[_OP_WEIGHTS[k] for k in feasible], k=1)[0]
    if kind == "new":
        return _build_new_op(rng.choice(hints))
    if kind == "enter":
        return _build_enter_op()
    return _build_exit_op()


def _build_random_script_ops(
    rng: random.Random, hints: Sequence[str], max_ops: int, max_depth: int
) -> list[dict[str, Any]]:
    num_ops = rng.randint(5, max_ops)
    depth = 0
    ops: list[dict[str, Any]] = []
    for _ in range(num_ops):
        op = _choose_random_op(rng, depth, max_depth, hints)
        if op["op"] == "enter":
            depth += 1
        elif op["op"] == "exit":
            depth -= 1
        ops.append(op)
    return ops


def _run_random_script(
    rng: random.Random, index: int, hints: Sequence[str], max_ops: int, max_depth: int
) -> dict[str, Any]:
    ops = _build_random_script_ops(rng, hints, max_ops, max_depth)
    return _run_script(f"random_{index:03d}", ops)


# =============================================================================
# Document assembly
# =============================================================================


def _print_summary(cases: list[dict[str, Any]], output_path: Path) -> None:
    """Print case, op-kind, nesting-depth, and repeat/fresh `new` counts."""
    op_counts: dict[str, int] = {"enter": 0, "exit": 0, "new": 0}
    repeats = 0
    fresh = 0
    max_depth = 0
    for case in cases:
        depth = 0
        case_max_depth = 0
        seen_relative_ids: list[int] = []
        for op in case["ops"]:
            op_counts[op["op"]] += 1
            if op["op"] == "enter":
                depth += 1
                case_max_depth = max(case_max_depth, depth)
            elif op["op"] == "exit":
                depth -= 1
            else:
                relative_id = op["expected"]["relative_id"]
                if relative_id in seen_relative_ids:
                    repeats += 1
                else:
                    fresh += 1
                    seen_relative_ids.append(relative_id)
        max_depth = max(max_depth, case_max_depth)

    total_ops = sum(op_counts.values())
    print(
        f"wrote {len(cases)} cases, {total_ops} ops "
        f"(enter={op_counts['enter']} exit={op_counts['exit']} "
        f"new={op_counts['new']}), max nesting depth {max_depth}, "
        f"new repeats={repeats} fresh={fresh}, to {output_path}"
    )


def _parse_arguments(default_output: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    add_corpus_arguments(
        parser,
        seed=_RANDOM_SEED,
        random_count=_RANDOM_SCRIPT_COUNT,
        max_ops=_RANDOM_MAX_OPS,
        default_output=default_output,
    )
    parser.add_argument(
        "--hints",
        default=",".join(_RANDOM_HINTS),
        help="comma-separated alphabet of name hints for random scripts",
    )
    return parser.parse_args()


def main() -> None:
    """Run every script through the oracle and write the golden document."""
    script_path = Path(__file__).resolve()
    repository_root = script_path.parents[3]
    arguments = _parse_arguments(
        script_path.parent / "deterministic_identifier_cases.json"
    )
    output_path = arguments.output
    hints = arguments.hints.split(",")

    cases = [_run_script(name, ops) for name, ops in _list_hand_picked_scripts()]

    rng = random.Random(arguments.seed)
    for index in range(arguments.random_count):
        cases.append(
            _run_random_script(rng, index, hints, arguments.max_ops, _RANDOM_MAX_DEPTH)
        )

    document = {
        "provenance": build_provenance(repository_root, GENERATOR_COMMAND),
        "cases": cases,
    }

    write_document(output_path, document)

    _print_summary(cases, output_path)


if __name__ == "__main__":
    main()
