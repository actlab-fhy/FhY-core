//! The per-lane kernels every backend of the evaluator calls, so a lane of
//! an array evaluation is computed exactly as the scalar evaluation of that
//! lane is.

use super::error::LaneFailure;

/// Return `a // b` over integers, rounded toward negative infinity.
#[inline]
pub(super) fn int_floor_divide(a: i64, b: i64) -> Result<i64, LaneFailure> {
    if b == 0 {
        return Err(LaneFailure::DivisionByZero);
    }
    let quotient = a.checked_div(b).ok_or(LaneFailure::IntegerOverflow)?;
    if a % b != 0 && ((a < 0) != (b < 0)) {
        Ok(quotient - 1)
    } else {
        Ok(quotient)
    }
}

/// Return `a % b` over integers, with the sign of the divisor.
#[inline]
pub(super) fn int_floor_mod(a: i64, b: i64) -> Result<i64, LaneFailure> {
    if b == 0 {
        return Err(LaneFailure::DivisionByZero);
    }
    let remainder = a.wrapping_rem(b);
    if remainder != 0 && ((remainder < 0) != (b < 0)) {
        Ok(remainder + b)
    } else {
        Ok(remainder)
    }
}

/// Return `a ** b` over integers.
#[inline]
pub(super) fn int_power(a: i64, b: i64) -> Result<i64, LaneFailure> {
    if b < 0 {
        return Err(LaneFailure::NegativeIntegerExponent);
    }
    match a {
        0 => Ok(i64::from(b == 0)),
        1 => Ok(1),
        -1 => Ok(if b % 2 == 0 { 1 } else { -1 }),
        _ => u32::try_from(b)
            .ok()
            .and_then(|exponent| a.checked_pow(exponent))
            .ok_or(LaneFailure::IntegerOverflow),
    }
}

/// Return the floor division and the remainder of `a` by `b` over reals,
/// as Python's and `NumPy`'s `divmod` compute them: the remainder is `fmod`'s
/// with the divisor's sign, and the quotient is the exact quotient rounded
/// down. A zero divisor gives `a / b` and `NaN`.
#[inline]
fn real_divmod(a: f64, b: f64) -> (f64, f64) {
    if b == 0.0 {
        return (a / b, f64::NAN);
    }
    let mut remainder = a % b;
    let mut quotient = (a - remainder) / b;
    if remainder == 0.0 {
        remainder = 0.0_f64.copysign(b);
    } else if (b < 0.0) != (remainder < 0.0) {
        remainder += b;
        quotient -= 1.0;
    }
    let floored = if quotient == 0.0 {
        0.0_f64.copysign(a / b)
    } else {
        let floored = quotient.floor();
        if quotient - floored > 0.5 {
            floored + 1.0
        } else {
            floored
        }
    };
    (floored, remainder)
}

/// Return `a // b` over reals.
#[inline]
pub(super) fn real_floor_divide(a: f64, b: f64) -> f64 {
    real_divmod(a, b).0
}

/// Return `a % b` over reals, with the sign of the divisor.
#[inline]
pub(super) fn real_floor_mod(a: f64, b: f64) -> f64 {
    real_divmod(a, b).1
}

/// Return the integer an integral real denotes, when it is finite and in
/// the 64-bit range.
#[inline]
pub(super) fn real_to_int(value: f64) -> Result<i64, LaneFailure> {
    // 2^63, the first value above the range; every integral value below it
    // and at least -2^63 converts exactly.
    const LIMIT: f64 = 9_223_372_036_854_775_808.0;
    if !value.is_finite() {
        return Err(LaneFailure::NonFiniteCast);
    }
    if (-LIMIT..LIMIT).contains(&value) {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the value is integral and inside the range"
        )]
        Ok(value as i64)
    } else {
        Err(LaneFailure::OutOfRangeCast)
    }
}

/// Return the nearest real to `value`.
#[inline]
#[expect(
    clippy::cast_precision_loss,
    reason = "an integer meets a real as its nearest binary64 value"
)]
pub(super) fn int_to_real(value: i64) -> f64 {
    value as f64
}
