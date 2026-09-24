//! Tests for the symbolic vocabulary enums: `SymbolType`, `FunctionSort`,
//! `UnaryOperation`, `BinaryOperation`, and `LogicalOperation`.
//!
//! Public API only. Each enum is checked for its text forms (`as_str`,
//! `symbol`, `Display`, `FromStr`), its serialized form, and the strings its
//! deserialization and parsing refuse; the stories at the end use the enums
//! the way an expression payload or a function signature does.

use crate::support::expression as expression_support;

use std::collections::{BTreeSet, HashMap};
use std::fmt::{self, Debug};
use std::str::FromStr;

use expression_support::{ALL_BINARY_OPERATIONS, ALL_LOGICAL_OPERATIONS, ALL_UNARY_OPERATIONS};
use fhy_core::expr::builtins::{BuiltinConstant, BuiltinFunction};
use fhy_core::expr::{
    BinaryOperation, FunctionSort, LogicalOperation, SymbolType, UnaryOperation, UnknownNameError,
};
use rstest::rstest;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// Every `SymbolType`, in declaration order.
const ALL_SYMBOL_TYPES: [SymbolType; 3] = [SymbolType::Real, SymbolType::Int, SymbolType::Bool];

/// Return the position of `symbol_type` in [`ALL_SYMBOL_TYPES`]; the
/// exhaustive `match` fails to compile when a variant is added, so the list
/// stays complete.
const fn index_symbol_type(symbol_type: SymbolType) -> usize {
    match symbol_type {
        SymbolType::Real => 0,
        SymbolType::Int => 1,
        SymbolType::Bool => 2,
    }
}

/// Every `FunctionSort`, in declaration order.
const ALL_FUNCTION_SORTS: [FunctionSort; 4] = [
    FunctionSort::Bool,
    FunctionSort::Nat,
    FunctionSort::Int,
    FunctionSort::Real,
];

/// Return the position of `sort` in [`ALL_FUNCTION_SORTS`]; the exhaustive
/// `match` fails to compile when a variant is added, so the list stays
/// complete.
const fn index_function_sort(sort: FunctionSort) -> usize {
    match sort {
        FunctionSort::Bool => 0,
        FunctionSort::Nat => 1,
        FunctionSort::Int => 2,
        FunctionSort::Real => 3,
    }
}

const _: () = {
    let mut index = 0;
    while index < ALL_SYMBOL_TYPES.len() {
        assert!(index_symbol_type(ALL_SYMBOL_TYPES[index]) == index);
        index += 1;
    }
    let mut index = 0;
    while index < ALL_FUNCTION_SORTS.len() {
        assert!(index_function_sort(ALL_FUNCTION_SORTS[index]) == index);
        index += 1;
    }
};

/// Every wire word of the four vocabularies, plus near misses of each: the
/// candidate strings the membership tests feed to deserialization.
const CANDIDATE_WIRE_WORDS: [&str; 31] = [
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
    "floor_mod",
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
#[case::floor_mod(BinaryOperation::FloorMod, "floor_mod", "%")]
#[case::power(BinaryOperation::Power, "power", "**")]
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
#[case::floor_mod(BinaryOperation::FloorMod, "floor_mod")]
#[case::power(BinaryOperation::Power, "power")]
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
#[case::old_modulo_name(json!("modulo"))]
#[case::unary_name(json!("negate"))]
#[case::padded(json!(" add"))]
#[case::empty(json!(""))]
#[case::integer(json!(1))]
#[case::null(json!(null))]
#[case::array(json!(["add"]))]
fn binary_operation_deserialization_rejects_other_input(#[case] input: Value) {
    assert_deserialization_rejects::<BinaryOperation>(&input);
}

/// Test the binary operations are exactly their thirteen wire names on the
/// wire.
#[test]
fn binary_operation_accepts_exactly_its_thirteen_wire_names() {
    let accepted = collect_accepted_wire_words::<BinaryOperation>();

    let expected = BTreeSet::from([
        "add",
        "subtract",
        "multiply",
        "divide",
        "floor_divide",
        "floor_mod",
        "power",
        "equal",
        "not_equal",
        "less",
        "less_equal",
        "greater",
        "greater_equal",
    ]);
    assert_eq!(accepted, expected);
}

/// Test each logical operation's wire name, symbol, and display text.
#[rstest]
#[case::and(LogicalOperation::And, "and", "&&")]
#[case::or(LogicalOperation::Or, "or", "||")]
fn logical_operation_text_forms_are_the_wire_name_and_symbol(
    #[case] operation: LogicalOperation,
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

/// Test a logical operation serializes as its wire name and deserializes
/// from it.
#[rstest]
#[case::and(LogicalOperation::And, "and")]
#[case::or(LogicalOperation::Or, "or")]
fn logical_operation_serializes_as_the_wire_name(
    #[case] operation: LogicalOperation,
    #[case] expected: &str,
) {
    assert_serializes_as_string(&operation, expected);
}

/// Test logical operation deserialization refuses symbols, other casings,
/// the old binary names, and non-string JSON.
#[rstest]
#[case::and_symbol(json!("&&"))]
#[case::title_case(json!("And"))]
#[case::upper_case(json!("OR"))]
#[case::old_binary_name(json!("logical_and"))]
#[case::other_connective(json!("xor"))]
#[case::empty(json!(""))]
#[case::integer(json!(1))]
#[case::null(json!(null))]
fn logical_operation_deserialization_rejects_other_input(#[case] input: Value) {
    assert_deserialization_rejects::<LogicalOperation>(&input);
}

/// Test the logical operations are exactly `and` and `or` on the wire.
#[test]
fn logical_operation_accepts_exactly_its_two_wire_names() {
    let accepted = collect_accepted_wire_words::<LogicalOperation>();

    assert_eq!(accepted, BTreeSet::from(["and", "or"]));
}

/// Assert parsing `word` as `T` fails with an [`UnknownNameError`] naming
/// `word` and displaying `expected`.
fn assert_from_str_refuses<T>(word: &str, expected: &str)
where
    T: FromStr<Err = UnknownNameError> + Debug,
{
    let error = word
        .parse::<T>()
        .expect_err("the word names no variant of the enum");

    assert_eq!(error.name(), word);
    assert_eq!(error.to_string(), expected);
}

/// Test each vocabulary's `FromStr` refuses an unknown word with its own
/// error, which names the word and the enum.
#[test]
fn vocabulary_from_str_refuses_an_unknown_word_naming_it() {
    assert_from_str_refuses::<SymbolType>("float", "unknown symbol type `float`");
    assert_from_str_refuses::<FunctionSort>("float", "unknown function sort `float`");
    assert_from_str_refuses::<UnaryOperation>("not", "unknown unary operation `not`");
    assert_from_str_refuses::<BinaryOperation>("plus", "unknown binary operation `plus`");
    assert_from_str_refuses::<LogicalOperation>("xor", "unknown logical operation `xor`");
    assert_from_str_refuses::<BinaryOperation>("Add", "unknown binary operation `Add`");
    assert_from_str_refuses::<BinaryOperation>("+", "unknown binary operation `+`");
    assert_from_str_refuses::<UnaryOperation>("", "unknown unary operation ``");
    assert_from_str_refuses::<BuiltinFunction>("softplus", "unknown built-in function `softplus`");
    assert_from_str_refuses::<BuiltinConstant>("tau", "unknown built-in constant `tau`");
}

/// Assert every variant in `variants` has one text in every form: `as_str`,
/// `Display`, the serialized JSON string, and the input `FromStr` and
/// `Deserialize` accept, and that no two variants share it.
fn assert_text_forms_agree<T>(variants: &[T], as_str: fn(T) -> &'static str)
where
    T: Copy + PartialEq + Debug + fmt::Display + FromStr + Serialize + DeserializeOwned,
    T::Err: Debug,
{
    let names: BTreeSet<&str> = variants.iter().map(|variant| as_str(*variant)).collect();
    assert_eq!(names.len(), variants.len(), "names are distinct");
    for &variant in variants {
        let name = as_str(variant);

        let displayed = variant.to_string();
        let parsed = name.parse::<T>().expect("the name parses");
        let serialized = serde_json::to_value(variant).expect("the variant serializes");
        let deserialized: T = serde_json::from_value(json!(name)).expect("the name deserializes");

        assert_eq!(displayed, name, "Display of {variant:?}");
        assert_eq!(parsed, variant, "FromStr of {name:?}");
        assert_eq!(serialized, json!(name), "serialized form of {variant:?}");
        assert_eq!(deserialized, variant, "deserialized form of {name:?}");
    }
}

/// Test `as_str`, `Display`, serde and `FromStr` agree for every variant of
/// every vocabulary enum, over lists whose completeness an exhaustive
/// `match` guards; the built-in enums are `#[non_exhaustive]`, so their
/// lists are their own catalogue iterators, which a unit test guards.
#[test]
fn vocabulary_as_str_display_serde_and_from_str_agree_for_every_variant() {
    let functions: Vec<BuiltinFunction> = BuiltinFunction::iter().collect();
    let constants: Vec<BuiltinConstant> = BuiltinConstant::iter().collect();

    assert_text_forms_agree(&ALL_SYMBOL_TYPES, SymbolType::as_str);
    assert_text_forms_agree(&ALL_FUNCTION_SORTS, FunctionSort::as_str);
    assert_text_forms_agree(&ALL_UNARY_OPERATIONS, UnaryOperation::as_str);
    assert_text_forms_agree(&ALL_BINARY_OPERATIONS, BinaryOperation::as_str);
    assert_text_forms_agree(&ALL_LOGICAL_OPERATIONS, LogicalOperation::as_str);
    assert_text_forms_agree(&functions, BuiltinFunction::name);
    assert_text_forms_agree(&constants, BuiltinConstant::name);
}

/// Test no two operations of one kind share a symbol, so a table from symbol
/// back to operation recovers every unary, binary and logical operation.
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

    let logical_by_symbol: HashMap<&str, LogicalOperation> = ALL_LOGICAL_OPERATIONS
        .into_iter()
        .map(|operation| (operation.symbol(), operation))
        .collect();

    assert_eq!(unary_by_symbol.len(), ALL_UNARY_OPERATIONS.len());
    assert_eq!(binary_by_symbol.len(), ALL_BINARY_OPERATIONS.len());
    assert_eq!(logical_by_symbol.len(), ALL_LOGICAL_OPERATIONS.len());
    for operation in ALL_UNARY_OPERATIONS {
        assert_eq!(unary_by_symbol[operation.symbol()], operation);
    }
    for operation in ALL_BINARY_OPERATIONS {
        assert_eq!(binary_by_symbol[operation.symbol()], operation);
    }
    for operation in ALL_LOGICAL_OPERATIONS {
        assert_eq!(logical_by_symbol[operation.symbol()], operation);
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
