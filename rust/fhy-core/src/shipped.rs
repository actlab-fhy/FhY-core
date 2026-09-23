//! The statics this crate ships that hold identifiers.
//!
//! The shipped note kinds, op attributes, value domains and composed
//! built-in functions mint their identifiers from the global id counter on
//! first use. Restoring an id near the end of the id space can leave the
//! counter too few ids to mint them later, after which every accessor of such
//! a static would panic. [`initialize_shipped_statics`] creates them all, and
//! the id counter calls it before any restored id advances it.

use crate::diagnostic::initialize_shipped_note_kinds;
use crate::op_attribute::initialize_shipped_attributes;
use crate::symbolic::expression::builtins::initialize_composed_functions;
use crate::value_domain::initialize_shipped_domains;

/// Create every shipped static that holds an identifier, if this is its first
/// use.
///
/// Every shipped static that holds an identifier must be created here. None
/// of them may restore an identifier while it is created: restoring calls
/// this function, and a static cannot wait on its own creation.
pub(crate) fn initialize_shipped_statics() {
    initialize_shipped_note_kinds();
    initialize_shipped_attributes();
    initialize_shipped_domains();
    initialize_composed_functions();
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use crate::diagnostic::{
        Note, get_other_note_kind, get_rationale_note_kind, get_remark_note_kind,
        get_suggestion_note_kind,
    };
    use crate::identifier::{
        IdSpaceExhausted, Identifier, try_advance_counter_past, try_allocate_id,
    };
    use crate::interned::Canonical;
    use crate::op_attribute::{
        OpAttribute, get_associative, get_commutative, get_elementwise, get_pure,
    };
    use crate::symbolic::expression::builtins::find_composed_function;
    use crate::symbolic::expression::{Expression, ExpressionKind};
    use crate::test_support::{assert_isolated_test_passes, is_isolated_run};
    use crate::value_domain::{ValueDomain, get_address_domain, get_data_domain};

    /// The paths by which an id reaches the id counter from outside.
    #[derive(Debug, Clone, Copy)]
    enum RestorePath {
        Identifier,
        Expression,
        Note,
        OpAttribute,
        ValueDomain,
        Restore,
        AdvanceCounter,
    }

    /// Return the JSON payload of an identifier named `restored` with `id`.
    fn encode_identifier_payload(id: u64) -> serde_json::Value {
        json!({"id": id, "name_hint": "restored"})
    }

    impl RestorePath {
        /// Bring `id` into the process along this path, returning the id the
        /// restored identifier holds.
        fn restore(self, id: u64) -> u64 {
            let identifier = encode_identifier_payload(id);
            let named = json!({"name": identifier, "description": "d"});
            match self {
                Self::Identifier => serde_json::from_value::<Identifier>(identifier)
                    .expect("the identifier decodes")
                    .id(),
                Self::Expression => {
                    let payload = json!({
                        "__type__": "identifier_expression",
                        "__data__": {"identifier": identifier},
                    });
                    let expression: Expression =
                        serde_json::from_value(payload).expect("the expression decodes");
                    let ExpressionKind::Identifier(restored) = expression.kind() else {
                        panic!("expected an identifier reference, got {expression:?}");
                    };
                    restored.id()
                }
                Self::Note => {
                    let payload = json!({"message": "m", "kind": named});
                    let note: Note = serde_json::from_value(payload).expect("the note decodes");
                    note.kind().name().id()
                }
                Self::OpAttribute => serde_json::from_value::<Canonical<OpAttribute>>(named)
                    .expect("the attribute decodes")
                    .name()
                    .id(),
                Self::ValueDomain => {
                    let payload = json!({"name": identifier, "description": "d", "parent": null});
                    serde_json::from_value::<Canonical<ValueDomain>>(payload)
                        .expect("the domain decodes")
                        .name()
                        .id()
                }
                Self::Restore => Identifier::restore(id, "restored".to_owned()).id(),
                Self::AdvanceCounter => {
                    try_advance_counter_past(id).expect("the id is below u64::MAX");
                    id
                }
            }
        }
    }

    /// Assert every shipped static is usable and holds only ids below
    /// `restored_id`.
    fn assert_every_shipped_static_is_usable(restored_id: u64) {
        let note_kinds = [
            (get_rationale_note_kind(), "rationale"),
            (get_suggestion_note_kind(), "suggestion"),
            (get_remark_note_kind(), "remark"),
            (get_other_note_kind(), "other"),
        ];
        for (kind, name_hint) in note_kinds {
            assert_eq!(kind.name().name_hint(), name_hint);
            assert!(kind.name().id() < restored_id, "{kind:?}");
        }
        let attributes = [
            (get_commutative(), "commutative"),
            (get_associative(), "associative"),
            (get_pure(), "pure"),
            (get_elementwise(), "elementwise"),
        ];
        for (attribute, name_hint) in attributes {
            assert_eq!(attribute.name().name_hint(), name_hint);
            assert!(attribute.name().id() < restored_id, "{attribute:?}");
        }
        let domains = [
            (get_data_domain(), "data"),
            (get_address_domain(), "address"),
        ];
        for (domain, name_hint) in domains {
            assert_eq!(domain.name().name_hint(), name_hint);
            assert!(domain.name().id() < restored_id, "{domain:?}");
        }
        for name in ["max", "gelu"] {
            let function = find_composed_function(name).expect("a composed built-in");
            assert_eq!(function.name(), name);
            assert!(
                function
                    .parameters()
                    .iter()
                    .all(|parameter| parameter.id() < restored_id),
                "{function:?}"
            );
        }
    }

    /// Bring an id near the end of the id space into a fresh process along
    /// `path` before any shipped static is used, then check every shipped
    /// static still works. Only meaningful in the child process that
    /// [`restoring_an_id_near_the_end_of_the_id_space_keeps_every_shipped_static`]
    /// starts.
    #[rstest]
    #[case::identifier(RestorePath::Identifier, u64::MAX - 1)]
    #[case::expression(RestorePath::Expression, u64::MAX - 1)]
    #[case::expression_leaving_four_ids(RestorePath::Expression, u64::MAX - 5)]
    #[case::note(RestorePath::Note, u64::MAX - 1)]
    #[case::op_attribute(RestorePath::OpAttribute, u64::MAX - 1)]
    #[case::value_domain(RestorePath::ValueDomain, u64::MAX - 1)]
    #[case::restore(RestorePath::Restore, u64::MAX - 1)]
    #[case::advance_counter(RestorePath::AdvanceCounter, u64::MAX - 1)]
    #[ignore = "exhausts the process-global counter; run through assert_isolated_test_passes"]
    fn restoring_an_id_near_the_end_of_the_id_space_keeps_every_shipped_static_in_isolation(
        #[case] path: RestorePath,
        #[case] id: u64,
    ) {
        if !is_isolated_run() {
            return;
        }

        let restored_id = path.restore(id);

        assert_eq!(restored_id, id);
        assert_every_shipped_static_is_usable(restored_id);
        if id == u64::MAX - 1 {
            assert_eq!(try_allocate_id(), Err(IdSpaceExhausted));
        }
    }

    #[rstest]
    #[case::identifier("case_1_identifier")]
    #[case::expression("case_2_expression")]
    #[case::expression_leaving_four_ids("case_3_expression_leaving_four_ids")]
    #[case::note("case_4_note")]
    #[case::op_attribute("case_5_op_attribute")]
    #[case::value_domain("case_6_value_domain")]
    #[case::restore("case_7_restore")]
    #[case::advance_counter("case_8_advance_counter")]
    fn restoring_an_id_near_the_end_of_the_id_space_keeps_every_shipped_static(
        #[case] case_name: &str,
    ) {
        assert_isolated_test_passes(&format!(
            "shipped::tests::\
             restoring_an_id_near_the_end_of_the_id_space_keeps_every_shipped_static_in_isolation::\
             {case_name}"
        ));
    }
}
