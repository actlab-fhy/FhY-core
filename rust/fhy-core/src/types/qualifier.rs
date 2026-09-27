//! Type qualifiers: how a value may be used.

use serde::{Deserialize, Serialize};

use crate::error::impl_name_text;

/// How a value may be used: read as an input, written as an output, kept
/// as state, fixed as a parameter, or computed as a temporary.
///
/// Displays and parses as its lowercase name (`param`), which is also its
/// serde form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum TypeQualifier {
    /// Read by the program.
    Input,
    /// Written by the program, and not readable.
    Output,
    /// Kept across invocations.
    State,
    /// Fixed for a run: a parameter or a constant.
    Param,
    /// Computed and discarded.
    Temp,
}

impl TypeQualifier {
    /// Return the lowercase name, such as `param`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::Output => "output",
            Self::State => "state",
            Self::Param => "param",
            Self::Temp => "temp",
        }
    }

    /// Return the qualifier of a value computed from values qualified
    /// `self` and `other`: `Param` when both are, and `Temp` otherwise.
    #[must_use]
    pub const fn promote(self, other: Self) -> Self {
        if matches!((self, other), (Self::Param, Self::Param)) {
            Self::Param
        } else {
            Self::Temp
        }
    }
}

impl_name_text!(TypeQualifier, "type qualifier");
