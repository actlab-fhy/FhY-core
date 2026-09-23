//! Shared golden-replay scaffolding for the equivalence tests.
//!
//! `interned_equivalence`, `tag_type_equivalence`, and
//! `deterministic_identifiers_equivalence` each replay a golden JSON document
//! recorded by a Python oracle against the matching Rust API, and each reads
//! an expanded corpus from an environment variable for its `#[ignore]`d
//! variant. This module holds the replay loop the three share: parsing the
//! document, checking it is large enough to be a meaningful replay, running
//! an optional document-level check, replaying every case through a
//! per-file callback, and failing the test with whatever mismatches that
//! callback collects.

use serde_json::Value;

/// Assert `cases` is large enough that an empty or truncated golden file
/// cannot pass the calling test, checking both the case count and the total
/// op count across every case.
fn assert_case_and_op_counts(cases: &[Value], min_cases: usize, min_ops: usize) {
    assert!(
        cases.len() >= min_cases,
        "expected at least {min_cases} golden cases, found {}",
        cases.len()
    );
    let total_ops: usize = cases
        .iter()
        .map(|case| case["ops"].as_array().map_or(0, Vec::len))
        .sum();
    assert!(
        total_ops >= min_ops,
        "expected at least {min_ops} golden ops, found {total_ops}"
    );
}

/// Parse a golden document, check its size against `min_cases` and
/// `min_ops`, run `check_document` against the whole document, replay every
/// case in it with `replay_case`, and fail with the mismatches found.
///
/// `check_document` performs whatever document-level cross-check a test
/// needs beyond the case and op counts (for example, that a defaults
/// catalogue or a set of default slot names matches the Rust side); a test
/// with no such check passes a no-op closure. `corpus_path` names the
/// expanded-corpus file being replayed, included in the failure message; the
/// default (non-expanded) replay passes `None`.
pub(crate) fn replay_golden_document(
    json: &str,
    min_cases: usize,
    min_ops: usize,
    corpus_path: Option<&str>,
    check_document: impl FnOnce(&Value),
    mut replay_case: impl FnMut(&Value, &mut Vec<String>),
) {
    let document: Value = serde_json::from_str(json).expect("golden data is valid JSON");
    let cases = document["cases"]
        .as_array()
        .expect("golden data has a `cases` array");

    assert_case_and_op_counts(cases, min_cases, min_ops);
    check_document(&document);

    let mut mismatches = Vec::new();
    for case in cases {
        replay_case(case, &mut mismatches);
    }

    let location = corpus_path
        .map(|path| format!(" in {path}"))
        .unwrap_or_default();
    assert!(
        mismatches.is_empty(),
        "found {} mismatch(es){location}:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}

/// Read the expanded corpus file named by the environment variable
/// `var_name`, returning its path and contents.
///
/// # Panics
///
/// Panics if `var_name` is unset or the file it names cannot be read.
pub(crate) fn read_expanded_corpus(var_name: &str) -> (String, String) {
    let path = std::env::var(var_name)
        .unwrap_or_else(|_| panic!("{var_name} names an expanded corpus file"));
    let json = std::fs::read_to_string(&path).expect("expanded corpus file is readable");
    (path, json)
}
