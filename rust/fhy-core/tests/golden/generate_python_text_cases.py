"""Generate golden text renderings of floats, decimal literal texts and Booleans.

Records three renderings the symbolic expression code needs byte for byte:

* ``repr(float)`` for hand-picked values and for values across every decimal
  magnitude from ``1e-330`` to ``1e308``, subnormals, signed zeros, infinities,
  NaNs with assorted signs and payloads, integral values, and random bit
  patterns. Each float is recorded by its IEEE 754 bit pattern, so NaN
  payloads and the sign of zero survive the JSON round trip.
* The exact normalization ``fhy_core.symbolic.expression.core`` applies to a
  float-grammar literal text: ``Decimal(text).normalize()`` under a context as
  wide as the coefficient with the widest exponent range, so nothing rounds.
  Each accepted text records the normalized coefficient digits, exponent and
  ``str`` text; each text outside the literal grammar records only that it
  was refused. Grammar acceptance is read from ``LiteralExpression``, and for
  float-grammar texts the recorded ``str`` text is checked against the public
  ``build_literal_equivalence_key`` before it is written. Texts with
  non-ASCII digits are left out, since the Rust grammar is ASCII only.
* ``str(bool)``.

The Rust replay is a unit test in ``rust/fhy-core/src/python_text.rs``,
which includes this corpus at compile time.

Run from the repository root:

    uv run --no-sync python rust/fhy-core/tests/golden/generate_python_text_cases.py

This overwrites ``rust/fhy-core/tests/golden/python_text_cases.json``.
``--random-count`` sets how many random float bit patterns and how many
random decimal texts are drawn, ``--max-ops`` sets the largest digit count of
a random decimal text, and ``--seed`` seeds both draws. The expanded-corpus
unit test reads a corpus written with other options from the file named in
``FHY_PYTHON_TEXT_CORPUS``.
"""

from __future__ import annotations

import argparse
import math
import random
import struct
from decimal import MAX_EMAX, MIN_EMIN, Context, Decimal
from pathlib import Path
from typing import Any

from _golden_support import add_corpus_arguments, build_provenance, write_document

from fhy_core.symbolic.expression import (
    LiteralExpression,
    build_literal_equivalence_key,
)

GENERATOR_COMMAND = (
    "uv run --no-sync python rust/fhy-core/tests/golden/generate_python_text_cases.py"
)

_RANDOM_SEED = 20260922
_RANDOM_COUNT = 300
_RANDOM_MAX_DIGITS = 60
_SMALLEST_DECIMAL_MAGNITUDE = -330
_LARGEST_DECIMAL_MAGNITUDE = 308
_MANTISSA_DIGITS = 17

_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_OUTPUT = Path(__file__).resolve().with_name("python_text_cases.json")

_HAND_PICKED_FLOATS: list[tuple[str, float]] = [
    ("zero", 0.0),
    ("negative_zero", -0.0),
    ("one", 1.0),
    ("negative_one", -1.0),
    ("one_and_a_half", 1.5),
    ("negative_two_and_a_half", -2.5),
    ("one_hundred", 100.0),
    ("one_tenth", 0.1),
    ("one_tenth_plus_two_tenths", 0.1 + 0.2),
    ("three_tenths_by_multiplication", 0.1 * 3),
    ("one_third", 1 / 3),
    ("two_thirds", 2 / 3),
    ("one_point_one", 1.1),
    ("fixed_decimal_123_456", 123.456),
    ("largest_fixed_exponent_one_ten_thousandth", 0.0001),
    ("just_below_one_ten_thousandth", 9.999999999999999e-05),
    ("one_hundred_thousandth", 1e-05),
    ("one_ten_millionth", 1e-07),
    ("integral_1e15", 1e15),
    ("integral_below_1e16", 9999999999999998.0),
    ("integral_1e16", 1e16),
    ("negative_integral_1e16", -1e16),
    ("integral_1e22", 1e22),
    ("integral_1e23", 1e23),
    ("two_to_the_52", 2.0**52),
    ("two_to_the_53", 2.0**53),
    ("two_to_the_53_plus_two", 2.0**53 + 2),
    ("two_to_the_63", 2.0**63),
    ("two_to_the_64", 2.0**64),
    ("seventeen_digit_integral", 123456789012345678.0),
    ("largest_finite", 1.7976931348623157e308),
    ("negative_largest_finite", -1.7976931348623157e308),
    ("smallest_normal", 2.2250738585072014e-308),
    ("largest_subnormal", 2.225073858507201e-308),
    ("smallest_subnormal", 5e-324),
    ("negative_smallest_subnormal", -5e-324),
    ("twice_smallest_subnormal", 1e-323),
    ("positive_infinity", math.inf),
    ("negative_infinity", -math.inf),
    ("pi", math.pi),
    ("e", math.e),
    ("machine_epsilon", 2.220446049250313e-16),
    ("one_plus_epsilon", 1.0000000000000002),
    ("three_smallest_subnormals", 5e-324 * 3),
]

_HAND_PICKED_FLOAT_BITS: list[tuple[str, int]] = [
    ("quiet_nan", 0x7FF8000000000000),
    ("negative_quiet_nan", 0xFFF8000000000000),
    ("signaling_nan_payload", 0x7FF0000000000001),
    ("negative_nan_full_payload", 0xFFFFFFFFFFFFFFFF),
    ("quiet_nan_payload", 0x7FF8000000000123),
    # Exactly halfway between two equally short digit strings: the text must
    # take the even last digit.
    ("halfway_tie_even_down", 0x4302FBD464D10462),
    ("halfway_tie_quarter", 0x431EFF490C10AE59),
    ("halfway_tie_sixteenth", 0x42D84F0570A50528),
    ("halfway_tie_many_fraction_digits", 0x42703C54C1CA8280),
    ("halfway_tie_long_fraction", 0x428A96FED6A292C0),
]

_HAND_PICKED_DECIMAL_TEXTS: list[tuple[str, str]] = [
    ("integer_zero", "0"),
    ("integer_zeros", "000"),
    ("integer_one", "1"),
    ("integer_five", "5"),
    ("integer_leading_zero", "05"),
    ("integer_trailing_zeros", "500"),
    ("integer_ten", "10"),
    ("integer_million", "1000000"),
    ("integer_forty_ones", "1" * 40),
    ("fraction_zero_point_zero", "0.0"),
    ("fraction_many_zeros", "0.000"),
    ("fraction_bare_point_zero", ".0"),
    ("fraction_zero_bare_point", "0."),
    ("fraction_one_point_five", "1.5"),
    ("fraction_trailing_zero", "1.50"),
    ("fraction_leading_zero", "01.5"),
    ("fraction_bare_point_five", ".5"),
    ("fraction_zero_point_five", "0.5"),
    ("fraction_five_bare_point", "5."),
    ("fraction_one_bare_point", "1."),
    ("fraction_one_point_zero", "1.0"),
    ("fraction_ten_bare_point", "10."),
    ("fraction_twelve_point_zero", "12.0"),
    ("fraction_hundred_twenty_point_zero", "120.0"),
    ("fraction_hundred_point_zero", "100.0"),
    ("fraction_million_bare_point", "1000000."),
    ("fraction_one_millionth", "0.000001"),
    ("fraction_one_ten_millionth", "0.0000001"),
    ("fraction_trailing_zeros_small", "0.00012300"),
    ("fraction_one_twenty_three_point_four_five", "123.45"),
    ("fraction_one_twenty_three_point_four_five_six", "123.456"),
    ("fraction_zero_point_one_zero", "0.10"),
    ("fraction_adjusted_exponent_minus_six", "0.0000012"),
    ("fraction_adjusted_exponent_minus_seven", "0.00000012"),
    ("fraction_one_e_minus_twenty_one", "0." + "0" * 20 + "1"),
    ("fraction_thirty_digits_last_one", "1." + "0" * 28 + "1"),
    ("fraction_thirty_digits_last_two", "1." + "0" * 28 + "2"),
    ("fraction_twenty_nine_significant", "3.1415926535897932384626433832"),
    ("fraction_thirty_one_significant", "3.141592653589793238462643383279"),
    ("fraction_forty_digits_trailing_zeros", "1" * 40 + "." + "0" * 10),
    ("fraction_sixty_nines", "9" * 30 + "." + "9" * 30),
    ("fraction_long_leading_zeros", "0" * 30 + "7.25"),
    ("fraction_long_trailing_zeros", "7.25" + "0" * 30),
    ("fraction_two_hundred_digits", "1234567890" * 10 + "." + "0987654321" * 10),
    ("fraction_large_integer_part", "1" + "0" * 80 + ".0"),
    ("fraction_tiny", "." + "0" * 120 + "3"),
    ("rejected_empty", ""),
    ("rejected_bare_point", "."),
    ("rejected_negative_integer", "-5"),
    ("rejected_positive_sign", "+5"),
    ("rejected_negative_fraction", "-1.5"),
    ("rejected_exponent", "1e10"),
    ("rejected_fraction_exponent", "1.5e3"),
    ("rejected_capital_exponent", "1E+2"),
    ("rejected_infinity", "inf"),
    ("rejected_nan", "NaN"),
    ("rejected_hex", "0x1f"),
    ("rejected_blank", "  "),
    ("rejected_trailing_space", "5 "),
    ("rejected_leading_space", " 5"),
    ("rejected_two_points", "1.2.3"),
    ("rejected_double_point", "1..5"),
    ("rejected_underscore", "1_000"),
    ("rejected_comma", "1,5"),
    ("rejected_letter", "12a"),
    ("rejected_newline", "5\n"),
]


def _float_to_bits(value: float) -> int:
    """Return the IEEE 754 binary64 bit pattern of ``value``."""
    return int(struct.unpack("<Q", struct.pack("<d", value))[0])


def _bits_to_float(bits: int) -> float:
    """Return the binary64 float whose bit pattern is ``bits``."""
    return float(struct.unpack("<d", struct.pack("<Q", bits))[0])


def _build_float_case(name: str, bits: int) -> dict[str, Any]:
    """Return the golden record of ``repr`` for the float with ``bits``."""
    value = _bits_to_float(bits)
    text = repr(value)
    if str(value) != text:
        raise AssertionError(f"str and repr disagree for {name}: {value!r}")
    return {"name": name, "bits": f"{bits:016x}", "repr": text}


def _build_float_cases(rng: random.Random, random_count: int) -> list[dict[str, Any]]:
    """Return every float case: hand-picked, per magnitude, and random."""
    cases = [
        _build_float_case(name, _float_to_bits(value))
        for name, value in _HAND_PICKED_FLOATS
    ]
    cases.extend(
        _build_float_case(name, bits) for name, bits in _HAND_PICKED_FLOAT_BITS
    )
    for exponent in range(_SMALLEST_DECIMAL_MAGNITUDE, _LARGEST_DECIMAL_MAGNITUDE + 1):
        power = float(f"1e{exponent}")
        cases.append(_build_float_case(f"power_1e{exponent}", _float_to_bits(power)))
        mantissa = rng.randrange(10 ** (_MANTISSA_DIGITS - 1), 10**_MANTISSA_DIGITS)
        scaled = float(f"{mantissa}e{exponent - _MANTISSA_DIGITS + 1}")
        cases.append(
            _build_float_case(f"random_mantissa_1e{exponent}", _float_to_bits(scaled))
        )
    for index in range(random_count):
        cases.append(_build_float_case(f"random_bits_{index}", rng.getrandbits(64)))
        subnormal_bits = rng.getrandbits(52) | (rng.getrandbits(1) << 63)
        cases.append(_build_float_case(f"random_subnormal_{index}", subnormal_bits))
    return cases


def _is_accepted_literal_text(text: str) -> bool:
    """Return whether ``LiteralExpression`` accepts ``text`` as a literal."""
    try:
        LiteralExpression(text)
    except ValueError:
        return False
    return True


def _build_decimal_case(name: str, text: str) -> dict[str, Any]:
    """Return the golden record of the exact normalization of ``text``."""
    if not text.isascii():
        raise AssertionError(f"decimal case {name} is not ASCII: {text!r}")
    if not _is_accepted_literal_text(text):
        return {"name": name, "text": text, "accepted": False}
    value = Decimal(text)
    precision = len(value.as_tuple().digits)
    normalized = value.normalize(Context(prec=precision, Emax=MAX_EMAX, Emin=MIN_EMIN))
    sign, digits, exponent = normalized.as_tuple()
    if sign != 0 or not isinstance(exponent, int):
        raise AssertionError(f"decimal case {name} normalized to {normalized!r}")
    rendered = str(normalized)
    if not text.isdigit():
        key = build_literal_equivalence_key(text)
        if key != f"float-decimal:{rendered}":
            raise AssertionError(f"decimal case {name}: key {key!r} vs {rendered!r}")
    return {
        "name": name,
        "text": text,
        "accepted": True,
        "digits": "".join(str(digit) for digit in digits),
        "exponent": exponent,
        "str": rendered,
    }


def _draw_digits(rng: random.Random, count: int) -> str:
    """Return ``count`` random ASCII digits."""
    return "".join(rng.choice("0123456789") for _ in range(count))


def _draw_decimal_text(rng: random.Random, max_digits: int) -> str:
    """Return a random text in the unsigned, exponent-free literal grammar."""
    leading_zeros = "0" * rng.choice([0, 0, 0, 1, 2, 5])
    trailing_zeros = "0" * rng.choice([0, 0, 0, 1, 3, 7])
    total_digits = rng.randint(1, max_digits)
    shape = rng.choice(["integer", "point_inside", "point_first", "point_last"])
    if shape == "integer":
        return leading_zeros + _draw_digits(rng, total_digits) + trailing_zeros
    if shape == "point_first":
        return "." + leading_zeros + _draw_digits(rng, total_digits) + trailing_zeros
    if shape == "point_last":
        return leading_zeros + _draw_digits(rng, total_digits) + "."
    split = rng.randint(1, total_digits)
    digits = _draw_digits(rng, total_digits)
    return leading_zeros + digits[:split] + "." + digits[split:] + trailing_zeros


def _build_decimal_cases(
    rng: random.Random, random_count: int, max_digits: int
) -> list[dict[str, Any]]:
    """Return every decimal case: hand-picked, then random grammar texts."""
    cases = [
        _build_decimal_case(name, text) for name, text in _HAND_PICKED_DECIMAL_TEXTS
    ]
    cases.extend(
        _build_decimal_case(f"random_text_{index}", _draw_decimal_text(rng, max_digits))
        for index in range(random_count)
    )
    return cases


def _build_bool_cases() -> list[dict[str, Any]]:
    """Return the ``str`` text of both Booleans."""
    return [{"value": value, "str": str(value)} for value in (False, True)]


def main() -> None:
    """Write the golden corpus of float, decimal and Boolean renderings."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    add_corpus_arguments(
        parser,
        seed=_RANDOM_SEED,
        random_count=_RANDOM_COUNT,
        max_ops=_RANDOM_MAX_DIGITS,
        default_output=_DEFAULT_OUTPUT,
    )
    arguments = parser.parse_args()
    rng = random.Random(arguments.seed)
    document = {
        "provenance": build_provenance(_REPOSITORY_ROOT, GENERATOR_COMMAND),
        "float_repr_cases": _build_float_cases(rng, arguments.random_count),
        "decimal_cases": _build_decimal_cases(
            rng, arguments.random_count, arguments.max_ops
        ),
        "bool_cases": _build_bool_cases(),
    }
    write_document(arguments.output, document)


if __name__ == "__main__":
    main()
