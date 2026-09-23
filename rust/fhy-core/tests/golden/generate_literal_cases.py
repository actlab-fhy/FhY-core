"""Generate golden literal-value cases from the Python literal oracle.

Each case is one literal: a Boolean, an integer of any size, a float (recorded
by its IEEE 754 bit pattern, so NaN payloads and the sign of zero survive the
JSON round trip), or a numeric text. For every literal the oracle records:

* whether ``LiteralExpression`` accepts it (only texts can be refused);
* ``build_literal_equivalence_key`` of the value;
* ``is_integer_valued_literal`` of the value;
* ``str`` of the value, the literal's own rendering;
* whether each ``FunctionSort`` accepts it, from
  ``is_python_value_compatible_with_sort``. That predicate is defined over
  ``bool``, ``int`` and ``float`` only, so a text is judged as the number its
  equivalence bucket denotes: an integer text as its ``int`` and a decimal
  text as a real (its ``float``).

A second section records pairwise structural equivalence
(``is_structurally_equivalent``) over a sample spanning every bucket, ``-0.0``,
NaNs of distinct sign, and decimals too long for ``Decimal``'s default
28-digit context.

Texts with non-ASCII digits are left out: the Rust literal grammar is ASCII
only, while the Python grammar accepts any Unicode decimal digit.

The Rust replay is ``rust/fhy-core/tests/expression_literal_equivalence.rs``.

Run from the repository root:

    uv run --no-sync python rust/fhy-core/tests/golden/generate_literal_cases.py

This overwrites ``rust/fhy-core/tests/golden/literal_cases.json``.
``--random-count`` sets how many random integers, float bit patterns and
texts are drawn, ``--max-ops`` sets the largest digit count of a random
integer or text, and ``--seed`` seeds the draws. The ignored expanded-corpus
test reads a corpus written with other options from the file named in
``FHY_LITERAL_CORPUS``.
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
import struct
from pathlib import Path
from typing import Any

from _golden_support import add_corpus_arguments, build_provenance, write_document

from fhy_core.symbolic.expression import (
    FunctionSort,
    LiteralExpression,
    build_literal_equivalence_key,
    is_integer_valued_literal,
    is_python_value_compatible_with_sort,
)

GENERATOR_COMMAND = (
    "uv run --no-sync python rust/fhy-core/tests/golden/generate_literal_cases.py"
)

_RANDOM_SEED = 20260922
_RANDOM_COUNT = 150
_RANDOM_MAX_DIGITS = 45

_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_OUTPUT = Path(__file__).resolve().with_name("literal_cases.json")

_SORTS = (FunctionSort.BOOL, FunctionSort.NAT, FunctionSort.INT, FunctionSort.REAL)

_HAND_PICKED_INTEGERS: list[tuple[str, int]] = [
    ("zero", 0),
    ("one", 1),
    ("negative_one", -1),
    ("five", 5),
    ("negative_three", -3),
    ("million", 1_000_000),
    ("i64_max", 2**63 - 1),
    ("i64_min", -(2**63)),
    ("just_above_i64_max", 2**63),
    ("just_below_i64_min", -(2**63) - 1),
    ("u64_max", 2**64 - 1),
    ("ten_to_the_thirty", 10**30),
    ("negative_ten_to_the_forty", -(10**40)),
    ("forty_ones", int("1" * 40)),
]

_HAND_PICKED_FLOATS: list[tuple[str, float]] = [
    ("zero", 0.0),
    ("negative_zero", -0.0),
    ("one", 1.0),
    ("negative_one", -1.0),
    ("five", 5.0),
    ("one_and_a_half", 1.5),
    ("negative_two_and_a_half", -2.5),
    ("three_point_one_four", 3.14),
    ("one_tenth", 0.1),
    ("one_tenth_plus_two_tenths", 0.1 + 0.2),
    ("one_ten_thousandth", 0.0001),
    ("one_hundred_thousandth", 1e-05),
    ("one_ten_billionth", 1e-10),
    ("integral_1e15", 1e15),
    ("integral_below_1e16", 9999999999999998.0),
    ("integral_1e16", 1e16),
    ("integral_1e22", 1e22),
    ("largest_finite", 1.7976931348623157e308),
    ("smallest_normal", 2.2250738585072014e-308),
    ("smallest_subnormal", 5e-324),
    ("negative_smallest_subnormal", -5e-324),
    ("positive_infinity", math.inf),
    ("negative_infinity", -math.inf),
]

_HAND_PICKED_FLOAT_BITS: list[tuple[str, int]] = [
    ("quiet_nan", 0x7FF8000000000000),
    ("negative_quiet_nan", 0xFFF8000000000000),
    ("signaling_nan_payload", 0x7FF0000000000001),
    ("negative_nan_full_payload", 0xFFFFFFFFFFFFFFFF),
]

_HAND_PICKED_TEXTS: list[tuple[str, str]] = [
    ("integer_zero", "0"),
    ("integer_zeros", "000"),
    ("integer_five", "5"),
    ("integer_forty_two", "42"),
    ("integer_leading_zero", "05"),
    ("integer_leading_zero_one", "01"),
    ("integer_double_zero", "00"),
    ("integer_forty_ones", "1" * 40),
    ("integer_just_above_i64_max", str(2**63)),
    ("decimal_zero_point_zero", "0.0"),
    ("decimal_bare_point_zero", ".0"),
    ("decimal_zero_bare_point", "0."),
    ("decimal_one_point_five", "1.5"),
    ("decimal_trailing_zero", "1.50"),
    ("decimal_leading_zero", "01.5"),
    ("decimal_bare_point_five", ".5"),
    ("decimal_zero_point_five", "0.5"),
    ("decimal_one_bare_point", "1."),
    ("decimal_one_point_zero", "1.0"),
    ("decimal_five_point_zero", "5.0"),
    ("decimal_three_point_one_four", "3.14"),
    ("decimal_one_tenth", "0.1"),
    ("decimal_one_tenth_trailing_zero", "0.10"),
    ("decimal_hundred_point_zero_zero_one", "100.001"),
    ("decimal_hundred_point_zero", "100.0"),
    ("decimal_ten_bare_point", "10."),
    ("decimal_million_bare_point", "1000000."),
    ("decimal_one_millionth", "0.000001"),
    ("decimal_one_ten_millionth", "0.0000001"),
    ("decimal_trailing_zeros_small", "0.00012300"),
    ("decimal_thirty_digits_last_one", "1." + "0" * 28 + "1"),
    ("decimal_thirty_digits_last_two", "1." + "0" * 28 + "2"),
    ("decimal_thirty_one_significant", "3.141592653589793238462643383279"),
    ("decimal_forty_digits_trailing_zeros", "1" * 40 + "." + "0" * 10),
    ("decimal_tiny", "." + "0" * 40 + "3"),
    ("rejected_empty", ""),
    ("rejected_bare_point", "."),
    ("rejected_word", "not_a_number"),
    ("rejected_infinity", "inf"),
    ("rejected_negative_infinity", "-inf"),
    ("rejected_infinity_word", "Infinity"),
    ("rejected_nan", "NaN"),
    ("rejected_exponent", "1e10"),
    ("rejected_fraction_exponent", "5.5e2"),
    ("rejected_hex", "0x1f"),
    ("rejected_negative_integer", "-5"),
    ("rejected_positive_sign", "+5"),
    ("rejected_blank", "  "),
    ("rejected_trailing_space", "5 "),
    ("rejected_leading_space", " 5"),
    ("rejected_two_points", "1.2.3"),
    ("rejected_underscore", "1_000"),
    ("rejected_comma", "1,5"),
    ("rejected_newline", "5\n"),
]

# The sample of `test_literal_equivalence_key_is_shared_exactly_by_equivalent_
# literals` (tests/symbolic/expression/test_core.py), as literal specs.
_EQUIVALENCE_SAMPLE: list[tuple[str, bool | int | float | str]] = [
    ("bool_true", True),
    ("bool_false", False),
    ("int_zero", 0),
    ("int_one", 1),
    ("int_five", 5),
    ("text_five", "5"),
    ("text_zero_five", "05"),
    ("text_one", "1"),
    ("float_zero", 0.0),
    ("float_negative_zero", -0.0),
    ("float_one", 1.0),
    ("float_five", 5.0),
    ("float_one_and_a_half", 1.5),
    ("float_infinity", math.inf),
    ("float_negative_infinity", -math.inf),
    ("float_nan", math.nan),
    ("float_negative_nan", -math.nan),
    ("text_zero_point_zero", "0.0"),
    ("text_bare_point_zero", ".0"),
    ("text_five_point_zero", "5.0"),
    ("text_one_point_five", "1.5"),
    ("text_one_point_fifty", "1.50"),
    ("text_one_point_zero", "1.0"),
    ("text_thirty_digits", "1.00000000000000000000000000001"),
    ("text_thirty_digits_other", "1.00000000000000000000000000002"),
]


def _float_to_bits(value: float) -> int:
    """Return the IEEE 754 binary64 bit pattern of ``value``."""
    return int.from_bytes(struct.pack(">d", value), "big")


def _bits_to_float(bits: int) -> float:
    """Return the binary64 float whose bit pattern is ``bits``."""
    (value,) = struct.unpack(">d", bits.to_bytes(8, "big"))
    return float(value)


def _encode_literal(value: bool | int | float | str) -> dict[str, Any]:
    """Return the JSON spec naming ``value``: its kind and a faithful value."""
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "int", "value": value}
    if isinstance(value, float):
        return {"kind": "float", "bits": f"{_float_to_bits(value):016x}"}
    return {"kind": "text", "value": value}


def _judge_sort_acceptance(value: bool | int | float | str) -> dict[str, bool]:
    """Return which sorts accept ``value``, judging a text by its bucket's number."""
    if isinstance(value, str):
        judged: bool | int | float = (
            int(value) if is_integer_valued_literal(value) else float(value)
        )
    else:
        judged = value
    return {
        str(sort): is_python_value_compatible_with_sort(judged, sort) for sort in _SORTS
    }


def _record_literal(name: str, value: bool | int | float | str) -> dict[str, Any]:
    """Return one golden case: the literal spec and everything the oracle says."""
    case: dict[str, Any] = {"name": name, "literal": _encode_literal(value)}
    try:
        literal = LiteralExpression(value)
    except ValueError:
        case["accepted"] = False
        return case
    case["accepted"] = True
    case["canonical_key"] = build_literal_equivalence_key(literal.value)
    case["is_integer_valued"] = is_integer_valued_literal(literal.value)
    case["text"] = str(literal.value)
    case["sorts"] = _judge_sort_acceptance(literal.value)
    return case


def _draw_random_integer(rng: random.Random, max_digits: int) -> int:
    """Return a random signed integer of up to ``max_digits`` digits."""
    digit_count = rng.randint(1, max_digits)
    magnitude = rng.randrange(10 ** (digit_count - 1), 10**digit_count)
    return rng.choice((-1, 1)) * magnitude


def _draw_random_text(rng: random.Random, max_digits: int) -> str:
    """Return a random grammar text: digits, optionally split by one point."""
    digits = "".join(
        rng.choice("0123456789") for _ in range(rng.randint(1, max_digits))
    )
    split = rng.randint(0, len(digits))
    shapes = (
        digits,
        "." + digits,
        digits + ".",
        digits[:split] + "." + digits[split:] + "0" * rng.randrange(3),
    )
    return rng.choice(shapes)


def build_literal_cases(
    seed: int, random_count: int, max_digits: int
) -> list[dict[str, Any]]:
    """Return every hand-picked and random literal case."""
    rng = random.Random(seed)
    cases = [_record_literal(f"bool_{value}".lower(), value) for value in (True, False)]
    cases += [
        _record_literal(f"int_{name}", value) for name, value in _HAND_PICKED_INTEGERS
    ]
    cases += [
        _record_literal(f"float_{name}", value) for name, value in _HAND_PICKED_FLOATS
    ]
    cases += [
        _record_literal(f"float_{name}", _bits_to_float(bits))
        for name, bits in _HAND_PICKED_FLOAT_BITS
    ]
    cases += [
        _record_literal(f"text_{name}", text) for name, text in _HAND_PICKED_TEXTS
    ]
    for index in range(random_count):
        cases.append(
            _record_literal(
                f"random_int_{index}", _draw_random_integer(rng, max_digits)
            )
        )
        cases.append(
            _record_literal(
                f"random_float_{index}", _bits_to_float(rng.getrandbits(64))
            )
        )
        cases.append(
            _record_literal(f"random_text_{index}", _draw_random_text(rng, max_digits))
        )
    return cases


def build_equivalence_pairs() -> list[dict[str, Any]]:
    """Return the oracle's structural equivalence over every pair of the sample."""
    pairs = []
    for (left_name, left), (right_name, right) in itertools.product(
        _EQUIVALENCE_SAMPLE, repeat=2
    ):
        pairs.append(
            {
                "left": {"name": left_name, "literal": _encode_literal(left)},
                "right": {"name": right_name, "literal": _encode_literal(right)},
                "equivalent": LiteralExpression(left).is_structurally_equivalent(
                    LiteralExpression(right)
                ),
            }
        )
    return pairs


def main() -> None:
    """Parse the options, run the oracle, and write the corpus."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_corpus_arguments(
        parser,
        seed=_RANDOM_SEED,
        random_count=_RANDOM_COUNT,
        max_ops=_RANDOM_MAX_DIGITS,
        default_output=_DEFAULT_OUTPUT,
    )
    arguments = parser.parse_args()
    document = {
        "provenance": build_provenance(_REPOSITORY_ROOT, GENERATOR_COMMAND),
        "cases": build_literal_cases(
            arguments.seed, arguments.random_count, arguments.max_ops
        ),
        "equivalence_pairs": build_equivalence_pairs(),
    }
    write_document(arguments.output, document)


if __name__ == "__main__":
    main()
