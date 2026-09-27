//! The serde round trip every serialized type keeps (CONTRIBUTING,
//! "Serialization is plain serde"), as a property check.

use proptest::test_runner::TestCaseError;
use serde::Serialize;
use serde::de::DeserializeOwned;

/// Check that `value` round-trips through JSON and postcard to an equal
/// value, and that the JSON decoded re-encodes to the same text.
///
/// # Errors
///
/// Returns the failure of the first check that fails.
pub(crate) fn check_serde_round_trip<T>(value: &T) -> Result<(), TestCaseError>
where
    T: Serialize + DeserializeOwned + PartialEq + std::fmt::Debug,
{
    let json = serde_json::to_string(value)
        .map_err(|error| TestCaseError::fail(format!("encodes as JSON: {error}")))?;
    let decoded: T = serde_json::from_str(&json)
        .map_err(|error| TestCaseError::fail(format!("decodes {json}: {error}")))?;
    if &decoded != value {
        return Err(TestCaseError::fail(format!(
            "JSON round trip of {json} gives {decoded:?}"
        )));
    }
    let again = serde_json::to_string(&decoded)
        .map_err(|error| TestCaseError::fail(format!("re-encodes: {error}")))?;
    if again != json {
        return Err(TestCaseError::fail(format!(
            "re-encoding differs: {json} then {again}"
        )));
    }
    let bytes = postcard::to_allocvec(value)
        .map_err(|error| TestCaseError::fail(format!("encodes as postcard: {error}")))?;
    let from_bytes: T = postcard::from_bytes(&bytes)
        .map_err(|error| TestCaseError::fail(format!("decodes postcard of {json}: {error}")))?;
    if &from_bytes != value {
        return Err(TestCaseError::fail(format!(
            "postcard round trip of {json} gives {from_bytes:?}"
        )));
    }
    Ok(())
}
