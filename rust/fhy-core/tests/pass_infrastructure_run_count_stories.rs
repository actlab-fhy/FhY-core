//! Tests for `fhy_core::pass::total_run_count`.
//!
//! The total counts every pass run in the process, so this binary holds a
//! single test: no other test can run a pass while it reads the total.

use std::error::Error;
use std::fmt;

use fhy_core::pass::{CompilerPass, ExecutePass, PassContext, PassFailure, total_run_count};

/// The error [`TotalCountedPass`] fails with.
#[derive(Debug)]
struct Refused;

impl fmt::Display for Refused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("refused")
    }
}

impl Error for Refused {}

/// Skips non-positive input, rejects input 100, and fails the run on input
/// 200.
struct TotalCountedPass;

impl CompilerPass<i64> for TotalCountedPass {
    fn validate_input(&mut self, ir: &i64, _cx: &mut PassContext<'_>) -> Result<(), PassFailure> {
        if *ir == 100 {
            return Err(Box::new(Refused));
        }
        Ok(())
    }

    fn should_run(&mut self, ir: &i64, _cx: &mut PassContext<'_>) -> Result<bool, PassFailure> {
        Ok(*ir > 0)
    }

    fn noop_output(&mut self, ir: &i64, _cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
        Ok(*ir)
    }

    fn run(&mut self, ir: &i64, _cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
        if *ir == 200 {
            return Err(Box::new(Refused));
        }
        Ok(ir + 1)
    }

    fn did_change(&mut self, input: &i64, output: &i64) -> Result<bool, PassFailure> {
        Ok(input != output)
    }
}

/// Test the total counts executed runs, including one that fails in `run`,
/// and ignores skipped runs and runs rejected before they start.
#[test]
fn total_run_count_counts_every_started_run() {
    let before = total_run_count();

    TotalCountedPass.execute(&1).expect("the run succeeds");
    let after_executed = total_run_count();
    TotalCountedPass
        .execute(&0)
        .expect("the skipped run succeeds");
    let after_skipped = total_run_count();
    TotalCountedPass
        .execute(&100)
        .expect_err("the input is rejected");
    let after_rejected = total_run_count();
    TotalCountedPass.execute(&200).expect_err("the run fails");
    let after_failed = total_run_count();

    assert_eq!(
        [after_executed, after_skipped, after_rejected, after_failed],
        [before + 1, before + 1, before + 1, before + 2]
    );
}
