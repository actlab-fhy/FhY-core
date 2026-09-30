"""Generate golden operation scripts from the `InternedMixin` oracle.

Drives the Python `InternedMixin` oracle (`fhy_core.traits.interned`) through
hand-picked and randomly generated operation scripts, recording the observed
outcome of each operation. The Rust equivalence test
(`rust/fhy-core/tests/it/interned/equivalence.rs`) replays these scripts
against `fhy_core::interned::InternRegistry` and compares every observation.

Run from the repository root:

    uv run --no-sync python rust/fhy-core/tests/golden/generate_interned_cases.py

This overwrites `rust/fhy-core/tests/golden/interned_cases.json`. Options select a
larger random corpus written elsewhere, which the ignored expanded-corpus
equivalence test replays from the file named in `FHY_INTERNED_CORPUS`.
`uv run nox -s golden_expanded` generates and replays an expanded corpus for
every generator.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from _golden_support import add_corpus_arguments, build_provenance, write_document

from fhy_core.traits.interned import InternedMixin
from fhy_core.utils.override import override

GENERATOR_COMMAND = (
    "uv run --no-sync python rust/fhy-core/tests/golden/generate_interned_cases.py"
)

# Defaults catalogue: catalogue id -> ordered (key, note) pairs. Kept in sync
# by hand with the four `fn() -> Vec<GoldenTag>` catalogue functions in
# `rust/fhy-core/tests/it/interned/equivalence.rs`; the equivalence test
# asserts the two cannot drift apart.
DEFAULTS_CATALOGUE: dict[str, list[tuple[str, str]]] = {
    "none": [],
    "one": [("a", "default-a")],
    "two": [("a", "default-a"), ("b", "default-b")],
    "duplicate": [("a", "default-a-first"), ("a", "default-a-second")],
}

_RANDOM_KEYS = ["a", "b", "c", "é"]
_OP_KINDS = ["intern", "get", "require", "clear"]
_OP_WEIGHTS = [50, 20, 20, 10]
_RANDOM_SEED = 20260919
_RANDOM_CASE_COUNT = 120


@dataclass
class _GoldenTag(InternedMixin[str]):
    """Oracle type interned by `name`, carrying a unique `note` per instance."""

    name: str
    note: str = field(compare=False)

    _script_defaults: ClassVar[list[_GoldenTag]] = []

    def __post_init__(self) -> None:
        self.register_interned_instance()

    @override
    def get_intern_key(self) -> str:
        return self.name

    @classmethod
    @override
    def register_default_instances(cls) -> None:
        for instance in cls._script_defaults:
            instance.register_interned_instance()


_note_counter = 0


def _next_note() -> str:
    global _note_counter  # noqa: PLW0603
    note = f"n{_note_counter}"
    _note_counter += 1
    return note


def _intern_op(key: str) -> dict[str, Any]:
    return {"op": "intern", "key": key, "note": _next_note()}


def _get_op(key: str) -> dict[str, Any]:
    return {"op": "get", "key": key}


def _require_op(key: str) -> dict[str, Any]:
    return {"op": "require", "key": key}


def _clear_op() -> dict[str, Any]:
    return {"op": "clear"}


def _install_defaults(defaults_id: str) -> None:
    _GoldenTag.clear_interned_registry()
    _GoldenTag._script_defaults = []
    for key, note in DEFAULTS_CATALOGUE[defaults_id]:
        instance = _GoldenTag(key, note)
        _GoldenTag._script_defaults.append(instance)


def _run_op(op: dict[str, Any]) -> dict[str, Any]:
    kind = op["op"]
    if kind == "intern":
        instance = _GoldenTag(op["key"], op["note"])
        canonical = _GoldenTag.get_interned(op["key"])
        if canonical is None:
            raise RuntimeError("intern must register a canonical instance")
        return {"registered": canonical is instance, "canonical_note": canonical.note}
    if kind == "get":
        canonical = _GoldenTag.get_interned(op["key"])
        return {"canonical_note": canonical.note if canonical is not None else None}
    if kind == "require":
        try:
            instance = _GoldenTag.require_interned(op["key"])
        except KeyError:
            return {"error": "KeyError"}
        return {"canonical_note": instance.note}
    if kind == "clear":
        _GoldenTag.clear_interned_registry()
        _GoldenTag.register_default_instances()
        return {}
    raise ValueError(f"unknown op {kind!r}")


def _run_script(
    name: str, defaults_id: str, ops: list[dict[str, Any]]
) -> dict[str, Any]:
    _install_defaults(defaults_id)
    recorded_ops = []
    for op in ops:
        expected = _run_op(op)
        recorded_ops.append({**op, "expected": expected})
    return {"name": name, "defaults": defaults_id, "ops": recorded_ops}


def _hand_picked_scripts() -> list[tuple[str, str, list[dict[str, Any]]]]:
    return [
        (
            "repeated_intern_keeps_first_canonical",
            "none",
            [_intern_op("a"), _intern_op("a"), _get_op("a"), _require_op("a")],
        ),
        (
            "get_and_require_before_any_intern",
            "none",
            [_get_op("a"), _require_op("a")],
        ),
        (
            "intern_clear_get_reintern_get",
            "none",
            [
                _intern_op("a"),
                _clear_op(),
                _get_op("a"),
                _intern_op("a"),
                _get_op("a"),
            ],
        ),
        (
            "defaults_two_survive_clear_and_new_key_does_not",
            "two",
            [
                _get_op("a"),
                _intern_op("a"),
                _clear_op(),
                _get_op("a"),
                _intern_op("c"),
                _clear_op(),
                _get_op("c"),
            ],
        ),
        (
            "duplicate_defaults_keep_the_first",
            "duplicate",
            [_get_op("a"), _require_op("a"), _intern_op("a")],
        ),
        (
            "unicode_and_special_keys",
            "none",
            [
                _intern_op(""),
                _intern_op("é"),
                _intern_op("😀"),
                _intern_op("a b"),
                _get_op(""),
                _get_op("é"),
                _get_op("😀"),
                _get_op("a b"),
            ],
        ),
        (
            "clear_on_empty_registry_then_require",
            "none",
            [_clear_op(), _require_op("a")],
        ),
    ]


def _random_scripts(
    seed: int, count: int, keys: list[str], max_ops: int
) -> list[tuple[str, str, list[dict[str, Any]]]]:
    rng = random.Random(seed)
    defaults_ids = list(DEFAULTS_CATALOGUE.keys())
    scripts = []
    for index in range(count):
        defaults_id = rng.choice(defaults_ids)
        num_ops = rng.randint(5, max_ops)
        ops: list[dict[str, Any]] = []
        for _ in range(num_ops):
            kind = rng.choices(_OP_KINDS, weights=_OP_WEIGHTS, k=1)[0]
            if kind == "intern":
                ops.append(_intern_op(rng.choice(keys)))
            elif kind == "get":
                ops.append(_get_op(rng.choice(keys)))
            elif kind == "require":
                ops.append(_require_op(rng.choice(keys)))
            else:
                ops.append(_clear_op())
        scripts.append((f"random_{index:03d}", defaults_id, ops))
    return scripts


def _defaults_catalogue_as_json() -> dict[str, list[dict[str, str]]]:
    return {
        defaults_id: [{"key": key, "note": note} for key, note in pairs]
        for defaults_id, pairs in DEFAULTS_CATALOGUE.items()
    }


def _parse_arguments(default_output: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    add_corpus_arguments(
        parser,
        seed=_RANDOM_SEED,
        random_count=_RANDOM_CASE_COUNT,
        max_ops=30,
        default_output=default_output,
    )
    parser.add_argument(
        "--keys",
        default=",".join(_RANDOM_KEYS),
        help="comma-separated key alphabet; an empty item is the empty key",
    )
    return parser.parse_args()


def main() -> None:
    """Run every script through the oracle and write the golden document."""
    script_path = Path(__file__).resolve()
    repository_root = script_path.parents[3]
    arguments = _parse_arguments(script_path.parent / "interned_cases.json")
    output_path = arguments.output

    scripts = _hand_picked_scripts() + _random_scripts(
        arguments.seed,
        arguments.random_count,
        arguments.keys.split(","),
        arguments.max_ops,
    )
    cases = [_run_script(name, defaults_id, ops) for name, defaults_id, ops in scripts]

    document = {
        "provenance": build_provenance(repository_root, GENERATOR_COMMAND),
        "defaults_catalogue": _defaults_catalogue_as_json(),
        "cases": cases,
    }

    write_document(output_path, document)

    total_ops = sum(len(case["ops"]) for case in cases)
    print(f"wrote {len(cases)} cases, {total_ops} ops, to {output_path}")


if __name__ == "__main__":
    main()
