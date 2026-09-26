//! Tests for `fhy_core::pass::VerificationRegistry` and `VerifierId`:
//! registration, lookups along a lineage of kinds, the verifier a lookup
//! builds, and verification into one report.
//!
//! Nothing here reads process-global state.

use crate::support::pass_ir;

use std::borrow::Cow;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use fhy_core::diagnostic::{DiagnosticLevel, ValidationReport};
use fhy_core::identifier::Identifier;
use fhy_core::pass::{
    CompilerPass, PassContext, PassError, PassErrorKind, PassFailure, PassManager, Validator,
    ValidatorRecord, VerificationPoint, VerificationRegistry, VerifierId,
};
use pass_ir::{BoxIr, ClosurePass};

// =============================================================================
// Helpers
// =============================================================================

/// The kinds of the registries under test: a general kind and two specific
/// ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Kind {
    Any,
    Loop,
    Call,
}

/// A validator type per `TAG`, named `check-<TAG>`, that reports one
/// warning saying its name, and an error when the box is negative.
struct Check<const TAG: u8>;

impl<const TAG: u8> Validator<BoxIr> for Check<TAG> {
    fn name(&self) -> Cow<'static, str> {
        Cow::Owned(format!("check-{TAG}"))
    }

    fn validate(&mut self, ir: &BoxIr, cx: &mut PassContext<'_>) -> Result<(), PassFailure> {
        cx.report_text(DiagnosticLevel::Warning, format!("check-{TAG} ran"), None);
        if ir.value() < 0 {
            cx.report_text(
                DiagnosticLevel::Error,
                format!("check-{TAG}: negative"),
                None,
            );
        }
        Ok(())
    }
}

/// A validator that fails without reporting an error.
struct Crash;

impl Validator<BoxIr> for Crash {
    fn validate(&mut self, _ir: &BoxIr, _cx: &mut PassContext<'_>) -> Result<(), PassFailure> {
        Err("the check could not finish".into())
    }
}

/// A validator named by the tag it was built with, so one type can stand for
/// many registrations.
struct Tagged(&'static str);

impl Validator<BoxIr> for Tagged {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed(self.0)
    }

    fn validate(&mut self, _ir: &BoxIr, _cx: &mut PassContext<'_>) -> Result<(), PassFailure> {
        Ok(())
    }
}

/// Return the names of `report`'s records, in order.
fn record_names(report: &ValidationReport<ValidatorRecord>) -> Vec<&str> {
    report
        .records()
        .iter()
        .map(ValidatorRecord::validator_name)
        .collect()
}

/// Return the messages of `report`'s diagnostics, in order.
fn messages(report: &ValidationReport<ValidatorRecord>) -> Vec<String> {
    report
        .diagnostics()
        .iter()
        .map(|diagnostic| diagnostic.message().message().to_owned())
        .collect()
}

/// Return the pipeline pass named `name` that adds `delta` to the box.
fn adding_pass(name: &'static str, delta: i64) -> ClosurePass<'static> {
    ClosurePass::new(name, move |ir, _cx| Ok(ir.derive(ir.value() + delta)))
}

/// Return the verification parts of `error`, failing the test for another
/// failure.
fn expect_verification(
    error: &PassError,
) -> (&str, VerificationPoint, &ValidationReport<ValidatorRecord>) {
    match error.kind() {
        PassErrorKind::Verification {
            pass_name,
            point,
            report,
            ..
        } => (pass_name, point, report),
        _ => panic!("expected a verification failure, got {error:?}"),
    }
}

// =============================================================================
// Registration
// =============================================================================

#[test]
fn an_empty_registry_has_no_registrations() {
    let registry = VerificationRegistry::<Kind, BoxIr>::new();

    assert!(registry.is_empty());
    assert_eq!(registry.len(), 0);
    assert!(registry.ids_for([&Kind::Any]).is_empty());
}

#[test]
fn an_empty_registry_verifies_to_an_empty_report() {
    let registry = VerificationRegistry::<Kind, BoxIr>::default();

    let report = registry.verify([&Kind::Any, &Kind::Loop], &BoxIr::new(-1));

    assert!(report.diagnostics().is_empty());
    assert!(report.records().is_empty());
}

#[test]
fn a_registered_validator_is_found_for_its_kind() {
    let mut registry = VerificationRegistry::new();

    assert!(registry.register(Kind::Loop, || Check::<1>));

    assert_eq!(registry.len(), 1);
    assert!(!registry.is_empty());
    assert_eq!(
        registry.ids_for([&Kind::Loop]),
        [VerifierId::of::<Check<1>>()]
    );
}

#[test]
fn registering_the_same_validator_again_changes_nothing() {
    let built_by_second_factory = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&built_by_second_factory);
    let mut registry = VerificationRegistry::new();
    registry.register(Kind::Loop, || Check::<1>);

    let added = registry.register(Kind::Loop, move || {
        counter.fetch_add(1, Ordering::Relaxed);
        Check::<1>
    });
    let validators = registry.validators_for([&Kind::Loop]);

    assert!(!added);
    assert_eq!(registry.len(), 1);
    assert_eq!(validators.len(), 1);
    assert_eq!(built_by_second_factory.load(Ordering::Relaxed), 0);
}

#[test]
fn validators_of_one_kind_keep_registration_order() {
    let mut registry = VerificationRegistry::new();
    registry.register(Kind::Loop, || Check::<2>);
    registry.register(Kind::Loop, || Check::<1>);
    registry.register(Kind::Loop, || Check::<3>);

    assert_eq!(
        registry.ids_for([&Kind::Loop]),
        [
            VerifierId::of::<Check<2>>(),
            VerifierId::of::<Check<1>>(),
            VerifierId::of::<Check<3>>(),
        ]
    );
}

#[test]
fn one_type_under_two_ids_is_two_registrations() {
    let first = Box::new(0_u8);
    let second = Box::new(0_u8);
    let mut registry = VerificationRegistry::new();

    let added_first =
        registry.register_with_id(Kind::Loop, VerifierId::of_ptr(&raw const *first), || {
            Tagged("first")
        });
    let added_second =
        registry.register_with_id(Kind::Loop, VerifierId::of_ptr(&raw const *second), || {
            Tagged("second")
        });
    let added_again =
        registry.register_with_id(Kind::Loop, VerifierId::of_ptr(&raw const *first), || {
            Tagged("ignored")
        });
    let report = registry.verify([&Kind::Loop], &BoxIr::new(0));

    assert!(added_first && added_second && !added_again);
    assert_eq!(registry.len(), 2);
    assert_eq!(record_names(&report), ["first", "second"]);
}

#[test]
fn a_validator_under_two_kinds_counts_twice_in_len() {
    let mut registry = VerificationRegistry::new();
    registry.register(Kind::Any, || Check::<1>);
    registry.register(Kind::Loop, || Check::<1>);

    assert_eq!(registry.len(), 2);
}

// =============================================================================
// Lookups along a lineage
// =============================================================================

#[test]
fn a_lineage_lists_base_kinds_first() {
    let mut registry = VerificationRegistry::new();
    registry.register(Kind::Loop, || Check::<2>);
    registry.register(Kind::Any, || Check::<1>);

    assert_eq!(
        registry.ids_for([&Kind::Any, &Kind::Loop]),
        [VerifierId::of::<Check<1>>(), VerifierId::of::<Check<2>>()]
    );
    assert_eq!(
        registry.ids_for([&Kind::Loop, &Kind::Any]),
        [VerifierId::of::<Check<2>>(), VerifierId::of::<Check<1>>()]
    );
}

#[test]
fn a_validator_under_two_kinds_of_a_lineage_appears_once() {
    let mut registry = VerificationRegistry::new();
    registry.register(Kind::Any, || Check::<1>);
    registry.register(Kind::Loop, || Check::<2>);
    registry.register(Kind::Loop, || Check::<1>);

    assert_eq!(
        registry.ids_for([&Kind::Any, &Kind::Loop]),
        [VerifierId::of::<Check<1>>(), VerifierId::of::<Check<2>>()]
    );
}

#[test]
fn a_validator_found_twice_is_built_by_its_first_registration() {
    let mut registry = VerificationRegistry::new();
    let base = Box::new(0_u8);
    let id = VerifierId::of_ptr(&raw const *base);
    registry.register_with_id(Kind::Any, id, || Tagged("from-any"));
    registry.register_with_id(Kind::Loop, id, || Tagged("from-loop"));

    let report = registry.verify([&Kind::Any, &Kind::Loop], &BoxIr::new(0));

    assert_eq!(record_names(&report), ["from-any"]);
}

#[test]
fn kinds_without_registrations_are_skipped() {
    let mut registry = VerificationRegistry::new();
    registry.register(Kind::Loop, || Check::<1>);

    assert!(registry.ids_for([&Kind::Call]).is_empty());
    assert!(registry.ids_for(std::iter::empty()).is_empty());
    assert_eq!(
        registry.ids_for([&Kind::Any, &Kind::Call, &Kind::Loop]),
        [VerifierId::of::<Check<1>>()]
    );
}

#[test]
fn every_lookup_builds_new_validators() {
    let builds = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&builds);
    let mut registry = VerificationRegistry::new();
    registry.register(Kind::Loop, move || {
        counter.fetch_add(1, Ordering::Relaxed);
        Check::<1>
    });
    let after_registration = builds.load(Ordering::Relaxed);

    let validators = registry.validators_for([&Kind::Loop]);
    let verifier = registry.verifier([&Kind::Loop]);
    let report = registry.verify([&Kind::Loop], &BoxIr::new(0));
    let empty_report = registry.verify([&Kind::Call], &BoxIr::new(0));

    assert_eq!(validators.len(), 1);
    assert_eq!(verifier.validator_names().count(), 1);
    assert_eq!(report.records().len(), 1);
    assert!(empty_report.records().is_empty());
    assert_eq!(after_registration, 0);
    assert_eq!(builds.load(Ordering::Relaxed), 3);
}

#[test]
fn validators_for_follows_the_order_of_ids_for() {
    let mut registry = VerificationRegistry::new();
    registry.register(Kind::Loop, || Check::<2>);
    registry.register(Kind::Any, || Check::<1>);

    let names: Vec<_> = registry
        .validators_for([&Kind::Any, &Kind::Loop])
        .iter()
        .map(Validator::name)
        .collect();

    assert_eq!(names, ["check-1", "check-2"]);
}

// =============================================================================
// Verification
// =============================================================================

#[test]
fn verify_has_one_record_per_registration_in_order() {
    let mut registry = VerificationRegistry::new();
    registry.register(Kind::Any, || Check::<1>);
    registry.register(Kind::Loop, || Check::<2>);

    let report = registry.verify([&Kind::Any, &Kind::Loop], &BoxIr::new(-5));

    assert_eq!(record_names(&report), ["check-1", "check-2"]);
    assert_eq!(
        messages(&report),
        [
            "check-1 ran",
            "check-1: negative",
            "check-2 ran",
            "check-2: negative"
        ]
    );
    assert_eq!(report.errors().count(), 2);
    let first = &report.records()[0];
    assert_eq!(first.diagnostics_in(&report).len(), 2);
    assert_eq!(first.diagnostics_in(&report)[0].source(), "check-1");
}

#[test]
fn a_failing_validator_does_not_stop_verification() {
    let mut registry = VerificationRegistry::new();
    registry.register(Kind::Any, || Crash);
    registry.register(Kind::Any, || Check::<1>);

    let report = registry.verify([&Kind::Any], &BoxIr::new(0));

    assert_eq!(record_names(&report), ["Crash", "check-1"]);
    assert!(report.records()[0].is_failed());
    assert!(!report.records()[1].is_failed());
    assert_eq!(
        messages(&report),
        [
            "validator \"Crash\" failed without reporting an error: the check could not finish",
            "check-1 ran"
        ]
    );
}

#[test]
fn the_verifier_is_named_verification_and_lists_the_validators() {
    let mut registry = VerificationRegistry::new();
    registry.register(Kind::Loop, || Check::<2>);
    registry.register(Kind::Any, || Check::<1>);

    let verifier = registry.verifier([&Kind::Any, &Kind::Loop]);

    assert_eq!(verifier.name().name_hint(), "verification");
    assert_eq!(
        verifier.validator_names().collect::<Vec<_>>(),
        ["check-1", "check-2"]
    );
}

#[test]
fn the_verifier_rejects_a_bad_input_blaming_the_first_pass() {
    let mut registry = VerificationRegistry::new();
    registry.register(Kind::Any, || Check::<1>);
    let mut manager = PassManager::new(Identifier::new("pipeline"));
    manager.add_pass(adding_pass("first", 1));
    manager.add_pass(adding_pass("second", 1));
    manager.set_verifier(registry.verifier([&Kind::Any]));

    let error = manager.run(&BoxIr::new(-5)).unwrap_err();
    let (pass_name, point, report) = expect_verification(&error);

    assert_eq!(pass_name, "first");
    assert_eq!(point, VerificationPoint::Input);
    assert_eq!(record_names(report), ["check-1"]);
    assert_eq!(report.errors().count(), 1);
}

#[test]
fn the_verifier_blames_the_producer_of_a_bad_output() {
    let mut registry = VerificationRegistry::new();
    registry.register(Kind::Any, || Check::<1>);
    let mut manager = PassManager::new(Identifier::new("pipeline"));
    manager.add_pass(adding_pass("clean", 1));
    manager.add_pass(adding_pass("corrupt", -100));
    manager.add_pass(adding_pass("never-runs", 1));
    manager.set_verifier(registry.verifier([&Kind::Any]));

    let error = manager.run(&BoxIr::new(0)).unwrap_err();
    let (pass_name, point, report) = expect_verification(&error);

    assert_eq!(pass_name, "corrupt");
    assert_eq!(point, VerificationPoint::Output);
    assert_eq!(report.errors().count(), 1);
    assert_eq!(error.records().len(), 1);
}

#[test]
fn a_passing_verifier_lets_the_pipeline_finish() {
    let mut registry = VerificationRegistry::new();
    registry.register(Kind::Any, || Check::<1>);
    let mut manager = PassManager::new(Identifier::new("pipeline"));
    manager.add_pass(adding_pass("clean", 1));
    manager.set_verifier(registry.verifier([&Kind::Any]));

    let result = manager.run(&BoxIr::new(0)).unwrap();

    assert_eq!(result.output().value(), 1);
}

// =============================================================================
// Values
// =============================================================================

#[test]
fn clones_share_their_registrations_and_then_diverge() {
    let mut registry = VerificationRegistry::new();
    registry.register(Kind::Any, || Check::<1>);

    let mut clone = registry.clone();
    clone.register(Kind::Any, || Check::<2>);

    assert_eq!(
        registry.ids_for([&Kind::Any]),
        [VerifierId::of::<Check<1>>()]
    );
    assert_eq!(
        clone.ids_for([&Kind::Any]),
        [VerifierId::of::<Check<1>>(), VerifierId::of::<Check<2>>()]
    );
}

#[test]
fn the_registry_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<VerificationRegistry<Kind, BoxIr>>();
    assert_send_sync::<VerifierId>();
}

#[test]
fn the_registry_debug_counts_kinds_and_registrations() {
    let mut registry = VerificationRegistry::new();
    registry.register(Kind::Any, || Check::<1>);
    registry.register(Kind::Loop, || Check::<1>);
    registry.register(Kind::Loop, || Check::<2>);

    assert_eq!(
        format!("{registry:?}"),
        "VerificationRegistry { kinds: 2, registrations: 3 }"
    );
}

#[test]
fn ids_of_types_are_equal_exactly_for_one_type() {
    assert_eq!(VerifierId::of::<Check<1>>(), VerifierId::of::<Check<1>>());
    assert_ne!(VerifierId::of::<Check<1>>(), VerifierId::of::<Check<2>>());
    assert_ne!(
        VerifierId::of::<Check<1>>(),
        VerifierId::of::<dyn CompilerPass<BoxIr>>()
    );
}

#[test]
fn ids_of_addresses_are_equal_exactly_for_one_address() {
    let values = [1_u32, 2];
    let slice: &[u32] = &values;

    assert_eq!(
        VerifierId::of_ptr(&raw const values[0]),
        VerifierId::of_ptr(slice.as_ptr())
    );
    assert_eq!(
        VerifierId::of_ptr(&raw const values[0]),
        VerifierId::of_ptr(&raw const *slice)
    );
    assert_ne!(
        VerifierId::of_ptr(&raw const values[0]),
        VerifierId::of_ptr(&raw const values[1])
    );
}

#[test]
fn an_id_of_a_type_never_equals_an_id_of_an_address() {
    let marker = Check::<1>;
    let by_address = VerifierId::of_ptr(&raw const marker);

    assert_ne!(VerifierId::of::<Check<1>>(), by_address);
    assert_ne!(by_address, VerifierId::of::<Check<1>>());
}

#[test]
fn an_id_debug_names_its_type_or_its_address() {
    let by_type = format!("{:?}", VerifierId::of::<Crash>());
    let by_address = format!(
        "{:?}",
        VerifierId::of_ptr(std::ptr::without_provenance::<u8>(0x10))
    );

    assert!(by_type.starts_with("VerifierId(") && by_type.ends_with("Crash)"));
    assert_eq!(by_address, "VerifierId(0x10)");
}
