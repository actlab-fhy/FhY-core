//! Tests for the symbolic vocabulary enums: `SymbolType`, `FunctionSort`,
//! `UnaryOperation`, and `BinaryOperation`.
//!
//! Public API only. Each enum is checked for its text forms (`as_str`,
//! `symbol`, `Display`), its serialized form, and the strings its
//! deserialization refuses; the stories at the end use the enums the way an
//! expression payload or a function signature does.

use std::collections::{BTreeSet, HashMap};
use std::fmt::Debug;

use fhy_core::symbolic::expression::{BinaryOperation, FunctionSort, UnaryOperation};
use fhy_core::symbolic::symbol_type::SymbolType;
use rstest::rstest;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// Every `SymbolType`, in declaration order.
const ALL_SYMBOL_TYPES: [SymbolType; 3] = [SymbolType::Real, SymbolType::Int, SymbolType::Bool];

/// Every `FunctionSort`, in declaration order.
const ALL_FUNCTION_SORTS: [FunctionSort; 4] = [
    FunctionSort::Bool,
    FunctionSort::Nat,
    FunctionSort::Int,
    FunctionSort::Real,
];

/// Every `UnaryOperation`, in declaration order.
const ALL_UNARY_OPERATIONS: [UnaryOperation; 3] = [
    UnaryOperation::Negate,
    UnaryOperation::Positive,
    UnaryOperation::LogicalNot,
];

/// Every `BinaryOperation`, in declaration order.
const ALL_BINARY_OPERATIONS: [BinaryOperation; 15] = [
    BinaryOperation::Add,
    BinaryOperation::Subtract,
    BinaryOperation::Multiply,
    BinaryOperation::Divide,
    BinaryOperation::FloorDivide,
    BinaryOperation::Modulo,
    BinaryOperation::Power,
    BinaryOperation::LogicalAnd,
    BinaryOperation::LogicalOr,
    BinaryOperation::Equal,
    BinaryOperation::NotEqual,
    BinaryOperation::Less,
    BinaryOperation::LessEqual,
    BinaryOperation::Greater,
    BinaryOperation::GreaterEqual,
];

/// Every wire word of the four vocabularies, plus near misses of each: the
/// candidate strings the membership tests feed to deserialization.
const CANDIDATE_WIRE_WORDS: [&str; 30] = [
    "real",
    "int",
    "bool",
    "nat",
    "float",
    "integer",
    "boolean",
    "natural",
    "negate",
    "positive",
    "logical_not",
    "add",
    "subtract",
    "multiply",
    "divide",
    "floor_divide",
    "modulo",
    "power",
    "logical_and",
    "logical_or",
    "equal",
    "not_equal",
    "less",
    "less_equal",
    "greater",
    "greater_equal",
    "not",
    "and",
    "or",
    "mod",
];

/// Assert `value` serializes to the JSON string `expected` and that string
/// deserializes back to `value`.
fn assert_serializes_as_string<T>(value: &T, expected: &str)
where
    T: Serialize + DeserializeOwned + PartialEq + Debug,
{
    let serialized = serde_json::to_value(value).expect("vocabulary values serialize");
    assert_eq!(serialized, json!(expected), "serialized form of {value:?}");

    let restored: T = serde_json::from_value(json!(expected)).expect("wire text deserializes");
    assert_eq!(&restored, value, "deserialized form of {expected:?}");
}

/// Assert deserializing `input` as `T` fails with a data error.
fn assert_deserialization_rejects<T: DeserializeOwned + Debug>(input: &Value) {
    let result = serde_json::from_value::<T>(input.clone());

    let Err(error) = &result else {
        panic!("expected {input} to be rejected, got {result:?}");
    };
    assert!(
        error.is_data(),
        "expected a data error for {input}, got {error}"
    );
}

/// Return the candidate wire words that deserialize as `T`.
fn collect_accepted_wire_words<T: DeserializeOwned>() -> BTreeSet<&'static str> {
    CANDIDATE_WIRE_WORDS
        .into_iter()
        .filter(|word| serde_json::from_value::<T>(json!(word)).is_ok())
        .collect()
}

/// Test each symbol type's text is its lowercase value name.
#[rstest]
#[case::real(SymbolType::Real, "real")]
#[case::int(SymbolType::Int, "int")]
#[case::bool(SymbolType::Bool, "bool")]
fn symbol_type_as_str_is_the_lowercase_value(
    #[case] symbol_type: SymbolType,
    #[case] expected: &str,
) {
    let text = symbol_type.as_str();

    assert_eq!(text, expected);
}

/// Test a symbol type displays as its value text.
#[rstest]
#[case::real(SymbolType::Real, "real")]
#[case::int(SymbolType::Int, "int")]
#[case::bool(SymbolType::Bool, "bool")]
fn symbol_type_display_writes_the_value(#[case] symbol_type: SymbolType, #[case] expected: &str) {
    let text = symbol_type.to_string();

    assert_eq!(text, expected);
}

/// Test a symbol type serializes as its value string and deserializes from
/// it.
#[rstest]
#[case::real(SymbolType::Real, "real")]
#[case::int(SymbolType::Int, "int")]
#[case::bool(SymbolType::Bool, "bool")]
fn symbol_type_serializes_as_the_value_string(
    #[case] symbol_type: SymbolType,
    #[case] expected: &str,
) {
    assert_serializes_as_string(&symbol_type, expected);
}

/// Test symbol type deserialization refuses other casings, other words, and
/// non-string JSON.
#[rstest]
#[case::upper_case(json!("REAL"))]
#[case::title_case(json!("Int"))]
#[case::function_sort_word(json!("nat"))]
#[case::float_word(json!("float"))]
#[case::padded(json!(" bool"))]
#[case::empty(json!(""))]
#[case::integer(json!(1))]
#[case::boolean(json!(true))]
#[case::null(json!(null))]
#[case::array(json!(["real"]))]
fn symbol_type_deserialization_rejects_other_input(#[case] input: Value) {
    assert_deserialization_rejects::<SymbolType>(&input);
}

/// Test the symbol types are exactly `real`, `int`, and `bool`: those three
/// texts, each accepted on the wire, and no other candidate word.
#[test]
fn symbol_type_has_exactly_the_real_int_bool_values() {
    let listed: BTreeSet<&str> = ALL_SYMBOL_TYPES
        .into_iter()
        .map(SymbolType::as_str)
        .collect();
    let accepted = collect_accepted_wire_words::<SymbolType>();

    let expected = BTreeSet::from(["real", "int", "bool"]);
    assert_eq!(listed, expected);
    assert_eq!(accepted, expected);
}

/// Test each function sort's text is its lowercase member name.
#[rstest]
#[case::bool(FunctionSort::Bool, "bool")]
#[case::nat(FunctionSort::Nat, "nat")]
#[case::int(FunctionSort::Int, "int")]
#[case::real(FunctionSort::Real, "real")]
fn function_sort_as_str_is_the_lowercase_name(#[case] sort: FunctionSort, #[case] expected: &str) {
    let text = sort.as_str();

    assert_eq!(text, expected);
}

/// Test a function sort displays as its value text.
#[rstest]
#[case::bool(FunctionSort::Bool, "bool")]
#[case::nat(FunctionSort::Nat, "nat")]
#[case::int(FunctionSort::Int, "int")]
#[case::real(FunctionSort::Real, "real")]
fn function_sort_display_writes_the_value(#[case] sort: FunctionSort, #[case] expected: &str) {
    let text = sort.to_string();

    assert_eq!(text, expected);
}

/// Test a function sort serializes as its value string and deserializes from
/// it.
#[rstest]
#[case::bool(FunctionSort::Bool, "bool")]
#[case::nat(FunctionSort::Nat, "nat")]
#[case::int(FunctionSort::Int, "int")]
#[case::real(FunctionSort::Real, "real")]
fn function_sort_serializes_as_the_value_string(
    #[case] sort: FunctionSort,
    #[case] expected: &str,
) {
    assert_serializes_as_string(&sort, expected);
}

/// Test function sort deserialization refuses other casings, other words,
/// and non-string JSON.
#[rstest]
#[case::upper_case(json!("BOOL"))]
#[case::title_case(json!("Nat"))]
#[case::long_word(json!("natural"))]
#[case::float_word(json!("float"))]
#[case::padded(json!("real "))]
#[case::empty(json!(""))]
#[case::integer(json!(0))]
#[case::boolean(json!(false))]
#[case::null(json!(null))]
#[case::object(json!({"sort": "real"}))]
fn function_sort_deserialization_rejects_other_input(#[case] input: Value) {
    assert_deserialization_rejects::<FunctionSort>(&input);
}

/// Test the function sorts are exactly `bool`, `nat`, `int`, and `real`:
/// those four texts, each accepted on the wire, and no other candidate word.
#[test]
fn function_sort_has_exactly_the_bool_nat_int_real_members() {
    let listed: BTreeSet<&str> = ALL_FUNCTION_SORTS
        .into_iter()
        .map(FunctionSort::as_str)
        .collect();
    let accepted = collect_accepted_wire_words::<FunctionSort>();

    let expected = BTreeSet::from(["bool", "nat", "int", "real"]);
    assert_eq!(listed, expected);
    assert_eq!(accepted, expected);
}

/// Test each unary operation's wire name, symbol, and display text.
#[rstest]
#[case::negate(UnaryOperation::Negate, "negate", "-")]
#[case::positive(UnaryOperation::Positive, "positive", "+")]
#[case::logical_not(UnaryOperation::LogicalNot, "logical_not", "!")]
fn unary_operation_text_forms_are_the_wire_name_and_symbol(
    #[case] operation: UnaryOperation,
    #[case] expected_name: &str,
    #[case] expected_symbol: &str,
) {
    let name = operation.as_str();
    let symbol = operation.symbol();
    let displayed = operation.to_string();

    assert_eq!(name, expected_name);
    assert_eq!(symbol, expected_symbol);
    assert_eq!(displayed, expected_name);
}

/// Test a unary operation serializes as its wire name and deserializes from
/// it.
#[rstest]
#[case::negate(UnaryOperation::Negate, "negate")]
#[case::positive(UnaryOperation::Positive, "positive")]
#[case::logical_not(UnaryOperation::LogicalNot, "logical_not")]
fn unary_operation_serializes_as_the_wire_name(
    #[case] operation: UnaryOperation,
    #[case] expected: &str,
) {
    assert_serializes_as_string(&operation, expected);
}

/// Test unary operation deserialization refuses symbols, other casings,
/// binary operation names, and non-string JSON.
#[rstest]
#[case::symbol(json!("-"))]
#[case::not_symbol(json!("!"))]
#[case::upper_case(json!("NEGATE"))]
#[case::title_case(json!("LogicalNot"))]
#[case::hyphenated(json!("logical-not"))]
#[case::short_word(json!("not"))]
#[case::binary_name(json!("subtract"))]
#[case::empty(json!(""))]
#[case::integer(json!(0))]
#[case::null(json!(null))]
fn unary_operation_deserialization_rejects_other_input(#[case] input: Value) {
    assert_deserialization_rejects::<UnaryOperation>(&input);
}

/// Test the unary operations are exactly `negate`, `positive`, and
/// `logical_not` on the wire.
#[test]
fn unary_operation_accepts_exactly_its_three_wire_names() {
    let accepted = collect_accepted_wire_words::<UnaryOperation>();

    assert_eq!(
        accepted,
        BTreeSet::from(["negate", "positive", "logical_not"])
    );
}

/// Test each binary operation's wire name, symbol, and display text.
#[rstest]
#[case::add(BinaryOperation::Add, "add", "+")]
#[case::subtract(BinaryOperation::Subtract, "subtract", "-")]
#[case::multiply(BinaryOperation::Multiply, "multiply", "*")]
#[case::divide(BinaryOperation::Divide, "divide", "/")]
#[case::floor_divide(BinaryOperation::FloorDivide, "floor_divide", "//")]
#[case::modulo(BinaryOperation::Modulo, "modulo", "%")]
#[case::power(BinaryOperation::Power, "power", "**")]
#[case::logical_and(BinaryOperation::LogicalAnd, "logical_and", "&&")]
#[case::logical_or(BinaryOperation::LogicalOr, "logical_or", "||")]
#[case::equal(BinaryOperation::Equal, "equal", "==")]
#[case::not_equal(BinaryOperation::NotEqual, "not_equal", "!=")]
#[case::less(BinaryOperation::Less, "less", "<")]
#[case::less_equal(BinaryOperation::LessEqual, "less_equal", "<=")]
#[case::greater(BinaryOperation::Greater, "greater", ">")]
#[case::greater_equal(BinaryOperation::GreaterEqual, "greater_equal", ">=")]
fn binary_operation_text_forms_are_the_wire_name_and_symbol(
    #[case] operation: BinaryOperation,
    #[case] expected_name: &str,
    #[case] expected_symbol: &str,
) {
    let name = operation.as_str();
    let symbol = operation.symbol();
    let displayed = operation.to_string();

    assert_eq!(name, expected_name);
    assert_eq!(symbol, expected_symbol);
    assert_eq!(displayed, expected_name);
}

/// Test a binary operation serializes as its wire name and deserializes from
/// it.
#[rstest]
#[case::add(BinaryOperation::Add, "add")]
#[case::subtract(BinaryOperation::Subtract, "subtract")]
#[case::multiply(BinaryOperation::Multiply, "multiply")]
#[case::divide(BinaryOperation::Divide, "divide")]
#[case::floor_divide(BinaryOperation::FloorDivide, "floor_divide")]
#[case::modulo(BinaryOperation::Modulo, "modulo")]
#[case::power(BinaryOperation::Power, "power")]
#[case::logical_and(BinaryOperation::LogicalAnd, "logical_and")]
#[case::logical_or(BinaryOperation::LogicalOr, "logical_or")]
#[case::equal(BinaryOperation::Equal, "equal")]
#[case::not_equal(BinaryOperation::NotEqual, "not_equal")]
#[case::less(BinaryOperation::Less, "less")]
#[case::less_equal(BinaryOperation::LessEqual, "less_equal")]
#[case::greater(BinaryOperation::Greater, "greater")]
#[case::greater_equal(BinaryOperation::GreaterEqual, "greater_equal")]
fn binary_operation_serializes_as_the_wire_name(
    #[case] operation: BinaryOperation,
    #[case] expected: &str,
) {
    assert_serializes_as_string(&operation, expected);
}

/// Test binary operation deserialization refuses symbols, other casings,
/// other spellings, unary operation names, and non-string JSON.
#[rstest]
#[case::plus_symbol(json!("+"))]
#[case::floor_divide_symbol(json!("//"))]
#[case::upper_case(json!("ADD"))]
#[case::title_case(json!("FloorDivide"))]
#[case::hyphenated(json!("floor-divide"))]
#[case::joined(json!("floordivide"))]
#[case::short_word(json!("and"))]
#[case::abbreviation(json!("mod"))]
#[case::unary_name(json!("negate"))]
#[case::padded(json!(" add"))]
#[case::empty(json!(""))]
#[case::integer(json!(1))]
#[case::null(json!(null))]
#[case::array(json!(["add"]))]
fn binary_operation_deserialization_rejects_other_input(#[case] input: Value) {
    assert_deserialization_rejects::<BinaryOperation>(&input);
}

/// Test the binary operations are exactly their fifteen wire names on the
/// wire.
#[test]
fn binary_operation_accepts_exactly_its_fifteen_wire_names() {
    let accepted = collect_accepted_wire_words::<BinaryOperation>();

    let expected = BTreeSet::from([
        "add",
        "subtract",
        "multiply",
        "divide",
        "floor_divide",
        "modulo",
        "power",
        "logical_and",
        "logical_or",
        "equal",
        "not_equal",
        "less",
        "less_equal",
        "greater",
        "greater_equal",
    ]);
    assert_eq!(accepted, expected);
}

/// Return the message deserializing the JSON string `word` as `T` fails with.
fn describe_rejected_word<T: DeserializeOwned + Debug>(word: &str) -> String {
    let result = serde_json::from_value::<T>(json!(word));

    let Err(error) = &result else {
        panic!("expected {word:?} to be rejected, got {result:?}");
    };
    error.to_string()
}

/// Test each vocabulary's rejection of an unknown word names the word and
/// the names it expects.
#[rstest]
#[case::symbol_type(
    describe_rejected_word::<SymbolType>("float"),
    "invalid value: string \"float\", expected a symbol type name: real, int, or bool"
)]
#[case::function_sort(
    describe_rejected_word::<FunctionSort>("float"),
    "invalid value: string \"float\", expected a function sort name: bool, nat, int, or real"
)]
#[case::unary_operation(
    describe_rejected_word::<UnaryOperation>("not"),
    "invalid value: string \"not\", expected a unary operation name such as negate or \
     logical_not"
)]
#[case::binary_operation(
    describe_rejected_word::<BinaryOperation>("plus"),
    "invalid value: string \"plus\", expected a binary operation name such as add or \
     floor_divide"
)]
fn vocabulary_rejection_names_the_word_and_the_expected_names(
    #[case] message: String,
    #[case] expected: &str,
) {
    assert_eq!(message, expected);
}

/// Test a table from symbol back to operation recovers every unary and every
/// binary operation, so a parser can map printed operators back to
/// operations.
#[test]
fn operation_symbols_invert_to_their_operations() {
    let unary_by_symbol: HashMap<&str, UnaryOperation> = ALL_UNARY_OPERATIONS
        .into_iter()
        .map(|operation| (operation.symbol(), operation))
        .collect();
    let binary_by_symbol: HashMap<&str, BinaryOperation> = ALL_BINARY_OPERATIONS
        .into_iter()
        .map(|operation| (operation.symbol(), operation))
        .collect();

    assert_eq!(unary_by_symbol.len(), ALL_UNARY_OPERATIONS.len());
    assert_eq!(binary_by_symbol.len(), ALL_BINARY_OPERATIONS.len());
    for operation in ALL_UNARY_OPERATIONS {
        assert_eq!(unary_by_symbol[operation.symbol()], operation);
    }
    for operation in ALL_BINARY_OPERATIONS {
        assert_eq!(binary_by_symbol[operation.symbol()], operation);
    }
}

/// A payload carrying one value of each vocabulary, as an expression or
/// signature payload nests them.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct VocabularyPayload {
    unary: UnaryOperation,
    binary: BinaryOperation,
    sort: FunctionSort,
    symbol_type: SymbolType,
}

/// Test a payload holding each vocabulary decodes from, and encodes back to,
/// the exact JSON text a wire payload carries.
#[test]
fn vocabulary_payload_round_trips_through_wire_json_text() {
    let wire_text =
        r#"{"unary":"logical_not","binary":"floor_divide","sort":"nat","symbol_type":"bool"}"#;

    let payload: VocabularyPayload = serde_json::from_str(wire_text).expect("wire payload decodes");
    let encoded = serde_json::to_string(&payload).expect("payload encodes");

    assert_eq!(
        payload,
        VocabularyPayload {
            unary: UnaryOperation::LogicalNot,
            binary: BinaryOperation::FloorDivide,
            sort: FunctionSort::Nat,
            symbol_type: SymbolType::Bool,
        }
    );
    assert_eq!(encoded, wire_text);
}

/// Test a payload naming an operation by its symbol is refused as a whole.
#[test]
fn vocabulary_payload_rejects_an_operation_named_by_symbol() {
    let input = json!({"unary": "!", "binary": "//", "sort": "nat", "symbol_type": "bool"});

    assert_deserialization_rejects::<VocabularyPayload>(&input);
}

/// Test a function signature renders from the sorts' display text.
#[test]
fn function_sort_display_renders_a_signature() {
    let parameters = [FunctionSort::Real, FunctionSort::Int, FunctionSort::Nat];
    let result = FunctionSort::Bool;

    let signature = format!(
        "check({}) -> {result}",
        parameters.map(|sort| sort.to_string()).join(", ")
    );

    assert_eq!(signature, "check(real, int, nat) -> bool");
}
