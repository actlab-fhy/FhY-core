//! Where a piece of IR came from: source positions, spans, and provenance trees.
//!
//! A [`Position`] is a 1-indexed line and column in a source text, and a
//! [`Span`] is a range given by byte offsets, positions, or both. Neither
//! names a file: a [`FileProvenance`] pairs a path with an optional span.
//!
//! A [`Provenance`] records the origin of a compiler object as one of five
//! variants. [`Provenance::Unknown`] carries no information; a file region, a
//! named wrapper around a child provenance, a call site, and a fusion of
//! several provenances cover the rest. Richer origins, such as a builtin or a
//! library symbol, are compositions of these variants rather than new ones.
//!
//! A transformation that combines several objects combines their provenances
//! with [`Provenance::fuse`], which drops unknown inputs and splices in the
//! sources of unlabelled fusions. A fusion that `fuse` builds therefore never
//! lists an unknown provenance or an unlabelled fusion among its own
//! sources; labelled fusions and the other variants are kept whole, whatever
//! they contain.
//!
//! [`Position`], [`Span`] and [`Provenance`] serialize to JSON-compatible
//! dicts. A position is `{"line": .., "column": ..}`, a span names all four
//! of its fields with `null` for an absent one, and a provenance is wrapped
//! as `{"__type__": "provenance.<kind>", "__data__": {..}}`.

use std::fmt;
use std::hash::Hash;
use std::num::NonZeroU64;
use std::sync::Arc;

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer, de};

use crate::decode::deserialize_map_only;

/// A 1-indexed line and column in a source text.
///
/// Positions order lexicographically, by line and then by column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Position {
    line: NonZeroU64,
    column: NonZeroU64,
}

impl Position {
    /// Creates the position at `line` and `column`.
    ///
    /// # Errors
    ///
    /// Returns [`ProvenanceError::ZeroLine`] if `line` is zero, and otherwise
    /// [`ProvenanceError::ZeroColumn`] if `column` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use fhy_core::provenance::Position;
    ///
    /// let position = Position::try_new(2, 8)?;
    ///
    /// assert_eq!(position.to_string(), "2:8");
    /// # Ok::<(), fhy_core::provenance::ProvenanceError>(())
    /// ```
    pub fn try_new(line: u64, column: u64) -> Result<Self, ProvenanceError> {
        let line = NonZeroU64::new(line).ok_or(ProvenanceError::ZeroLine)?;
        let column = NonZeroU64::new(column).ok_or(ProvenanceError::ZeroColumn)?;
        Ok(Self { line, column })
    }

    /// Returns the 1-indexed line.
    #[must_use]
    pub fn line(&self) -> NonZeroU64 {
        self.line
    }

    /// Returns the 1-indexed column.
    #[must_use]
    pub fn column(&self) -> NonZeroU64 {
        self.column
    }
}

/// Renders the position as `line:column`, for example `2:8`.
impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.column)
    }
}

/// Encodes the position as `{"line": .., "column": ..}`.
impl Serialize for Position {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("Position", 2)?;
        state.serialize_field("line", &self.line.get())?;
        state.serialize_field("column", &self.column.get())?;
        state.end()
    }
}

/// A position payload, checked for its keys and types but not its values.
#[derive(Deserialize)]
#[serde(rename = "Position", deny_unknown_fields)]
struct PositionPayload {
    line: u64,
    column: u64,
}

/// Decodes `{"line": .., "column": ..}`, rejecting a missing or unknown key,
/// a value that is not an integer in `u64`, and a zero line or column.
impl<'de> Deserialize<'de> for Position {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let payload: PositionPayload = deserialize_map_only(deserializer)?;
        Position::try_new(payload.line, payload.column).map_err(de::Error::custom)
    }
}

/// A range in a source text given by byte offsets, positions, or both.
///
/// Each of the four bounds is optional. The offsets and the positions are
/// never checked against each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Span {
    start_offset: Option<u64>,
    end_offset: Option<u64>,
    start_position: Option<Position>,
    end_position: Option<Position>,
}

impl Span {
    /// Creates the span with the given bounds.
    ///
    /// # Errors
    ///
    /// Returns [`ProvenanceError::EndOffsetBeforeStartOffset`] if both offsets
    /// are set and `end_offset < start_offset`, and otherwise
    /// [`ProvenanceError::EndPositionBeforeStartPosition`] if both positions
    /// are set and `end_position < start_position`.
    ///
    /// # Examples
    ///
    /// ```
    /// use fhy_core::provenance::{Position, Span};
    ///
    /// let span = Span::try_new(
    ///     Some(0),
    ///     Some(3),
    ///     Some(Position::try_new(1, 1)?),
    ///     Some(Position::try_new(1, 4)?),
    /// )?;
    ///
    /// assert_eq!(span.to_string(), "1:1-1:4");
    /// # Ok::<(), fhy_core::provenance::ProvenanceError>(())
    /// ```
    pub fn try_new(
        start_offset: Option<u64>,
        end_offset: Option<u64>,
        start_position: Option<Position>,
        end_position: Option<Position>,
    ) -> Result<Self, ProvenanceError> {
        if let (Some(start_offset), Some(end_offset)) = (start_offset, end_offset) {
            if end_offset < start_offset {
                return Err(ProvenanceError::EndOffsetBeforeStartOffset {
                    start_offset,
                    end_offset,
                });
            }
        }
        if let (Some(start_position), Some(end_position)) = (start_position, end_position) {
            if end_position < start_position {
                return Err(ProvenanceError::EndPositionBeforeStartPosition {
                    start_position,
                    end_position,
                });
            }
        }
        Ok(Self {
            start_offset,
            end_offset,
            start_position,
            end_position,
        })
    }

    /// Returns the span with no bounds at all.
    #[must_use]
    pub fn unknown() -> Self {
        Self {
            start_offset: None,
            end_offset: None,
            start_position: None,
            end_position: None,
        }
    }

    /// Returns whether none of the four bounds is set.
    #[must_use]
    pub fn is_unknown(&self) -> bool {
        *self == Self::unknown()
    }

    /// Returns the byte offset the span starts at, if known.
    #[must_use]
    pub fn start_offset(&self) -> Option<u64> {
        self.start_offset
    }

    /// Returns the byte offset the span ends at, if known.
    #[must_use]
    pub fn end_offset(&self) -> Option<u64> {
        self.end_offset
    }

    /// Returns the position the span starts at, if known.
    #[must_use]
    pub fn start_position(&self) -> Option<Position> {
        self.start_position
    }

    /// Returns the position the span ends at, if known.
    #[must_use]
    pub fn end_position(&self) -> Option<Position> {
        self.end_position
    }
}

/// Render `bound`, or `?` when it is absent.
fn format_bound(f: &mut fmt::Formatter<'_>, bound: Option<impl fmt::Display>) -> fmt::Result {
    match bound {
        Some(bound) => write!(f, "{bound}"),
        None => f.write_str("?"),
    }
}

/// Renders `<unknown>` when no bound is set. Otherwise, when either position
/// is set, renders `start-end` from the positions alone, ignoring the
/// offsets; else renders `@start-end` from the offsets. A missing bound
/// renders as `?`, as in `1:1-?` or `@5-?`.
impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_unknown() {
            f.write_str("<unknown>")
        } else if self.start_position.is_some() || self.end_position.is_some() {
            format_bound(f, self.start_position)?;
            f.write_str("-")?;
            format_bound(f, self.end_position)
        } else {
            f.write_str("@")?;
            format_bound(f, self.start_offset)?;
            f.write_str("-")?;
            format_bound(f, self.end_offset)
        }
    }
}

/// Encodes the span as `{"start_offset", "end_offset", "start_position",
/// "end_position"}`, writing `null` for an absent bound.
impl Serialize for Span {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("Span", 4)?;
        state.serialize_field("start_offset", &self.start_offset)?;
        state.serialize_field("end_offset", &self.end_offset)?;
        state.serialize_field("start_position", &self.start_position)?;
        state.serialize_field("end_position", &self.end_position)?;
        state.end()
    }
}

/// A span payload, checked for its keys and types but not its bounds' order.
///
/// `serde` lets an `Option` field be missing and decode as `None`, but the
/// wire form always carries all four keys. Naming a `deserialize_with` turns
/// off that special case, so a payload without a key is rejected.
#[derive(Deserialize)]
#[serde(rename = "Span", deny_unknown_fields)]
struct SpanPayload {
    #[serde(deserialize_with = "Option::deserialize")]
    start_offset: Option<u64>,
    #[serde(deserialize_with = "Option::deserialize")]
    end_offset: Option<u64>,
    #[serde(deserialize_with = "Option::deserialize")]
    start_position: Option<Position>,
    #[serde(deserialize_with = "Option::deserialize")]
    end_position: Option<Position>,
}

/// Decodes the four-key span dict. Every key must be present (`null` for an
/// absent bound) and no other key may appear; the bounds are then checked as
/// in [`Span::try_new`].
impl<'de> Deserialize<'de> for Span {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let payload: SpanPayload = deserialize_map_only(deserializer)?;
        Span::try_new(
            payload.start_offset,
            payload.end_offset,
            payload.start_position,
            payload.end_position,
        )
        .map_err(de::Error::custom)
    }
}

/// Origin information for a compiler object.
///
/// Two provenances are equal when they are the same variant with equal
/// fields, compared recursively.
///
/// # Nesting depth
///
/// Equality, hashing, `Debug`, [`Display`](fmt::Display), serialization,
/// deserialization and dropping recurse through nested provenances, and
/// cloning recurses through nested fusions (a named or call-site child is
/// shared, not copied), so their stack use grows with the nesting depth of
/// the tree, and a tree nested deeply enough (on the order of tens of
/// thousands of levels on a default thread stack) overflows the stack.
/// [`Provenance::fuse`] walks an explicit stack instead, and a fusion it
/// builds never lists an unlabelled fusion among its own sources.
///
/// Decoding JSON text is also capped by `serde_json`'s nesting limit: its
/// text deserializer refuses input nested more than 127 JSON levels deep
/// with a `recursion limit exceeded` error. A named or call-site level takes
/// two JSON levels and a fused level three, since its sources sit in a list,
/// so `serde_json::from_str` decodes at most 62 nested named or call-site
/// levels over the unknown provenance. A `serde_json::Value` parsed from text
/// meets the same limit. A caller needing deeper trees can enable `serde_json`'s
/// `unbounded_depth` feature and decode through a `serde_json::Deserializer`
/// after calling its `disable_recursion_limit`, on a thread with a stack
/// large enough for the recursion above.
///
/// # Serialization
///
/// A provenance encodes as `{"__type__": <type id>, "__data__": <fields>}`
/// with the type ids `provenance.unknown`, `provenance.file`,
/// `provenance.named`, `provenance.call_site` and `provenance.fused`. The
/// unknown provenance's fields are the empty dict. Decoding requires exactly
/// those two keys and exactly the variant's field keys.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Provenance {
    /// No source information is available.
    Unknown,
    /// A region of a source file.
    File(FileProvenance),
    /// A child provenance under a human-readable name.
    Named(NamedProvenance),
    /// A value created at a call site, such as by inlining.
    CallSite(CallSiteProvenance),
    /// Several provenances combined by a transformation.
    Fused(FusedProvenance),
}

impl Provenance {
    /// Combines `provenances` into one provenance, keeping their order.
    ///
    /// The inputs are flattened first: every [`Provenance::Unknown`] is
    /// dropped, and an unlabelled [`FusedProvenance`] (one whose metadata is
    /// `None`) is replaced by its sources, at any depth. A labelled fusion,
    /// even one labelled with the empty string, and every other variant are
    /// kept whole. Equal inputs are not deduplicated.
    ///
    /// When nothing survives the result is [`Provenance::Unknown`], whatever
    /// `metadata` is. When exactly one input survives and `metadata` is
    /// `None`, the result is that input. Otherwise the result is a
    /// [`FusedProvenance`] of the survivors labelled with `metadata`.
    ///
    /// [`Provenance::Unknown`] is therefore an identity for `fuse`, and
    /// without metadata `fuse` is associative. The flattening walks an
    /// explicit stack, so deeply nested unlabelled fusions do not exhaust the
    /// call stack.
    ///
    /// # Examples
    ///
    /// ```
    /// use fhy_core::provenance::{FileProvenance, Provenance};
    ///
    /// let a = Provenance::File(FileProvenance::new("a.fhy", None));
    /// let b = Provenance::File(FileProvenance::new("b.fhy", None));
    ///
    /// let fused = Provenance::fuse([a, Provenance::Unknown, b], Some("loop-fusion"));
    ///
    /// assert_eq!(fused.to_string(), "loop-fusion[a.fhy, b.fhy]");
    /// ```
    #[must_use]
    pub fn fuse(
        provenances: impl IntoIterator<Item = Provenance>,
        metadata: Option<&str>,
    ) -> Provenance {
        let flat = flatten_fusion_inputs(provenances);
        match <[Provenance; 1]>::try_from(flat) {
            Ok([single]) if metadata.is_none() => single,
            Err(flat) if flat.is_empty() => Provenance::Unknown,
            Ok(single) => Provenance::Fused(FusedProvenance::new(
                Vec::from(single),
                metadata.map(str::to_owned),
            )),
            Err(flat) => Provenance::Fused(FusedProvenance::new(flat, metadata.map(str::to_owned))),
        }
    }

    /// Return the type id this provenance is wrapped under on the wire.
    fn type_id(&self) -> &'static str {
        match self {
            Provenance::Unknown => "provenance.unknown",
            Provenance::File(_) => "provenance.file",
            Provenance::Named(_) => "provenance.named",
            Provenance::CallSite(_) => "provenance.call_site",
            Provenance::Fused(_) => "provenance.fused",
        }
    }
}

/// Return `provenances` in order with every unknown provenance dropped and
/// every unlabelled fusion replaced by its sources, at any depth.
///
/// The walk keeps the not-yet-visited provenances on an explicit stack, in
/// reverse order, and moves each spliced fusion's sources onto it, so neither
/// the walk nor dropping the dismantled fusions recurses.
fn flatten_fusion_inputs(provenances: impl IntoIterator<Item = Provenance>) -> Vec<Provenance> {
    let mut pending: Vec<Provenance> = provenances.into_iter().collect();
    pending.reverse();
    let mut flat = Vec::with_capacity(pending.len());
    while let Some(provenance) = pending.pop() {
        match provenance {
            Provenance::Unknown => {}
            Provenance::Fused(fused) if fused.metadata.is_none() => {
                pending.extend(fused.sources.into_vec().into_iter().rev());
            }
            survivor => flat.push(survivor),
        }
    }
    flat
}

/// Renders a human-readable description: `<unknown>`, the file variant's
/// path and span, `name` or `name (child)`, `callee at caller`, or
/// `label[source, ...]`.
impl fmt::Display for Provenance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Provenance::Unknown => f.write_str("<unknown>"),
            Provenance::File(file) => {
                f.write_str(&file.file_path)?;
                match &file.span {
                    Some(span) if !span.is_unknown() => write!(f, ":{span}"),
                    _ => Ok(()),
                }
            }
            Provenance::Named(named) => match named.child() {
                Provenance::Unknown => f.write_str(&named.name),
                child => write!(f, "{} ({child})", named.name),
            },
            Provenance::CallSite(call_site) => {
                write!(f, "{} at {}", call_site.callee(), call_site.caller())
            }
            Provenance::Fused(fused) => {
                f.write_str(fused.metadata().unwrap_or("fused"))?;
                f.write_str("[")?;
                for (index, source) in fused.sources.iter().enumerate() {
                    if index > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{source}")?;
                }
                f.write_str("]")
            }
        }
    }
}

/// The fields of the unknown provenance: none, encoded as the empty dict.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct UnknownFields {}

/// The fields of a file provenance, borrowed for encoding.
#[derive(Serialize)]
struct FileFields<'a> {
    file_path: &'a str,
    span: Option<&'a Span>,
}

/// The fields of a named provenance, borrowed for encoding.
#[derive(Serialize)]
struct NamedFields<'a> {
    name: &'a str,
    child: &'a Provenance,
}

/// The fields of a call-site provenance, borrowed for encoding.
#[derive(Serialize)]
struct CallSiteFields<'a> {
    callee: &'a Provenance,
    caller: &'a Provenance,
}

/// The fields of a fused provenance, borrowed for encoding.
#[derive(Serialize)]
struct FusedFields<'a> {
    sources: &'a [Provenance],
    metadata: Option<&'a str>,
}

/// Encodes the provenance in the wrapped `__type__`/`__data__` form.
impl Serialize for Provenance {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut envelope = serializer.serialize_struct("Provenance", 2)?;
        envelope.serialize_field("__type__", self.type_id())?;
        match self {
            Provenance::Unknown => envelope.serialize_field("__data__", &UnknownFields {})?,
            Provenance::File(file) => envelope.serialize_field(
                "__data__",
                &FileFields {
                    file_path: &file.file_path,
                    span: file.span.as_ref(),
                },
            )?,
            Provenance::Named(named) => envelope.serialize_field(
                "__data__",
                &NamedFields {
                    name: &named.name,
                    child: named.child(),
                },
            )?,
            Provenance::CallSite(call_site) => envelope.serialize_field(
                "__data__",
                &CallSiteFields {
                    callee: call_site.callee(),
                    caller: call_site.caller(),
                },
            )?,
            Provenance::Fused(fused) => envelope.serialize_field(
                "__data__",
                &FusedFields {
                    sources: &fused.sources,
                    metadata: fused.metadata(),
                },
            )?,
        }
        envelope.end()
    }
}

/// A file provenance payload, checked for its keys and types.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FilePayload {
    file_path: String,
    // Required even when `null`; see `SpanPayload`.
    #[serde(deserialize_with = "Option::deserialize")]
    span: Option<Span>,
}

/// A named provenance payload, checked for its keys and types.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NamedPayload {
    name: String,
    child: Provenance,
}

/// A call-site provenance payload, checked for its keys and types.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CallSitePayload {
    callee: Provenance,
    caller: Provenance,
}

/// A fused provenance payload, checked for its keys and types.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FusedPayload {
    sources: Vec<Provenance>,
    // Required even when `null`; see `SpanPayload`.
    #[serde(deserialize_with = "Option::deserialize")]
    metadata: Option<String>,
}

/// A wrapped provenance payload, dispatched on its `__type__`.
#[derive(Deserialize)]
#[serde(tag = "__type__", content = "__data__", deny_unknown_fields)]
enum ProvenancePayload {
    #[serde(rename = "provenance.unknown")]
    #[serde(deserialize_with = "deserialize_map_only")]
    Unknown(UnknownFields),
    #[serde(rename = "provenance.file")]
    #[serde(deserialize_with = "deserialize_map_only")]
    File(FilePayload),
    #[serde(rename = "provenance.named")]
    #[serde(deserialize_with = "deserialize_map_only")]
    Named(NamedPayload),
    #[serde(rename = "provenance.call_site")]
    #[serde(deserialize_with = "deserialize_map_only")]
    CallSite(CallSitePayload),
    #[serde(rename = "provenance.fused")]
    #[serde(deserialize_with = "deserialize_map_only")]
    Fused(FusedPayload),
}

/// Decodes the wrapped `__type__`/`__data__` form, rejecting an unknown
/// type id, a missing or unknown key at either level, and field values the
/// variant's constructor rejects.
impl<'de> Deserialize<'de> for Provenance {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(match deserialize_map_only(deserializer)? {
            ProvenancePayload::Unknown(UnknownFields {}) => Provenance::Unknown,
            ProvenancePayload::File(file) => {
                Provenance::File(FileProvenance::new(file.file_path, file.span))
            }
            ProvenancePayload::Named(named) => Provenance::Named(
                NamedProvenance::try_new(named.name, named.child).map_err(de::Error::custom)?,
            ),
            ProvenancePayload::CallSite(call_site) => {
                Provenance::CallSite(CallSiteProvenance::new(call_site.callee, call_site.caller))
            }
            ProvenancePayload::Fused(fused) => {
                Provenance::Fused(FusedProvenance::new(fused.sources, fused.metadata))
            }
        })
    }
}

/// Return `path` normalized as Python's `PurePosixPath` normalizes it.
///
/// `/` is the only separator; every other character, a backslash or a
/// drive letter's colon included, is part of a component. Empty and `.`
/// components are removed, which drops repeated and trailing separators,
/// `..` components are kept, a leading `//` (exactly two separators) is kept
/// while three or more leading separators become one, and a path with no
/// root and no components becomes `.`. The result is the same on every
/// platform.
fn normalize_file_path(path: &str) -> String {
    let root = if path.starts_with("//") && !path.starts_with("///") {
        "//"
    } else if path.starts_with('/') {
        "/"
    } else {
        ""
    };
    let components: Vec<&str> = path
        .split('/')
        .filter(|component| !component.is_empty() && *component != ".")
        .collect();
    let normalized = format!("{root}{}", components.join("/"));
    if normalized.is_empty() {
        ".".to_owned()
    } else {
        normalized
    }
}

/// Provenance pointing at a region of a source file.
///
/// The path is text stored in normalized form, as Python's `PurePosixPath`
/// normalizes it on every platform: `/` is the only separator, repeated
/// separators and `.` components are removed and a trailing separator is
/// dropped, so `./a` and `a//b/` become `a` and `a/b`, and the empty path
/// becomes `.`. A `..` component, `~`, a leading `//`, a backslash and a
/// drive letter such as `C:` are kept as written, so `C:\src\a.fhy` is
/// one component. Equality, hashing and [`Display`](fmt::Display) use the
/// normalized text, so `//a` and `/a` differ.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FileProvenance {
    file_path: String,
    span: Option<Span>,
}

impl FileProvenance {
    /// Creates the provenance for `span` in the file at `file_path`, storing
    /// the path in normalized form.
    ///
    /// # Examples
    ///
    /// ```
    /// use fhy_core::provenance::FileProvenance;
    ///
    /// let provenance = FileProvenance::new("./src//main.fhy", None);
    ///
    /// assert_eq!(provenance.file_path(), "src/main.fhy");
    /// ```
    #[must_use]
    pub fn new(file_path: impl AsRef<str>, span: Option<Span>) -> Self {
        Self {
            file_path: normalize_file_path(file_path.as_ref()),
            span,
        }
    }

    /// Returns the normalized path of the file.
    #[must_use]
    pub fn file_path(&self) -> &str {
        &self.file_path
    }

    /// Returns the region of the file, if one was given.
    #[must_use]
    pub fn span(&self) -> Option<&Span> {
        self.span.as_ref()
    }
}

/// A child provenance under a human-readable name, such as a builtin over
/// [`Provenance::Unknown`] or a library symbol over the library's file.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NamedProvenance {
    name: String,
    child: Arc<Provenance>,
}

impl NamedProvenance {
    /// Creates the provenance naming `child` as `name`.
    ///
    /// # Errors
    ///
    /// Returns [`ProvenanceError::EmptyName`] if `name` is empty.
    pub fn try_new(name: impl Into<String>, child: Provenance) -> Result<Self, ProvenanceError> {
        let name = name.into();
        if name.is_empty() {
            return Err(ProvenanceError::EmptyName);
        }
        Ok(Self {
            name,
            child: Arc::new(child),
        })
    }

    /// Returns the name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the named provenance.
    #[must_use]
    pub fn child(&self) -> &Provenance {
        &self.child
    }
}

/// Provenance of a value created at a call site, such as by inlining or
/// macro expansion.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CallSiteProvenance {
    callee: Arc<Provenance>,
    caller: Arc<Provenance>,
}

impl CallSiteProvenance {
    /// Creates the provenance of a value from `callee` created at `caller`.
    #[must_use]
    #[expect(
        clippy::similar_names,
        reason = "callee and caller are the standard names for the two ends of a call"
    )]
    pub fn new(callee: Provenance, caller: Provenance) -> Self {
        Self {
            callee: Arc::new(callee),
            caller: Arc::new(caller),
        }
    }

    /// Returns the provenance of the called code.
    #[must_use]
    pub fn callee(&self) -> &Provenance {
        &self.callee
    }

    /// Returns the provenance of the call site.
    #[must_use]
    pub fn caller(&self) -> &Provenance {
        &self.caller
    }
}

/// Several provenances combined by a transformation, with an optional label.
///
/// The constructor keeps its sources as given: it accepts no sources, a
/// single unlabelled source, and nested unknown or unlabelled sources. Use
/// [`Provenance::fuse`] to build the flat form.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FusedProvenance {
    sources: Box<[Provenance]>,
    metadata: Option<String>,
}

impl FusedProvenance {
    /// Creates the fusion of `sources`, in order, labelled with `metadata`.
    #[must_use]
    pub fn new(sources: Vec<Provenance>, metadata: Option<String>) -> Self {
        Self {
            sources: sources.into_boxed_slice(),
            metadata,
        }
    }

    /// Returns the fused provenances, in order.
    #[must_use]
    pub fn sources(&self) -> &[Provenance] {
        &self.sources
    }

    /// Returns the label, or `None` for an unlabelled fusion.
    #[must_use]
    pub fn metadata(&self) -> Option<&str> {
        self.metadata.as_deref()
    }
}

/// Types that carry the provenance of the object they represent.
pub trait HasProvenance {
    /// Returns the object's provenance.
    #[must_use]
    fn provenance(&self) -> &Provenance;
}

/// A position, span, or named provenance was built from invalid parts.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ProvenanceError {
    /// A position's line was zero; lines start at 1.
    ZeroLine,
    /// A position's column was zero; columns start at 1.
    ZeroColumn,
    /// A span's end offset preceded its start offset.
    EndOffsetBeforeStartOffset {
        /// The offset the span was to start at.
        start_offset: u64,
        /// The offset the span was to end at.
        end_offset: u64,
    },
    /// A span's end position preceded its start position.
    EndPositionBeforeStartPosition {
        /// The position the span was to start at.
        start_position: Position,
        /// The position the span was to end at.
        end_position: Position,
    },
    /// A named provenance was given an empty name.
    EmptyName,
}

impl fmt::Display for ProvenanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProvenanceError::ZeroLine => f.write_str("a position's line must be at least 1"),
            ProvenanceError::ZeroColumn => f.write_str("a position's column must be at least 1"),
            ProvenanceError::EndOffsetBeforeStartOffset {
                start_offset,
                end_offset,
            } => write!(
                f,
                "a span's end offset {end_offset} precedes its start offset {start_offset}"
            ),
            ProvenanceError::EndPositionBeforeStartPosition {
                start_position,
                end_position,
            } => write!(
                f,
                "a span's end position {end_position} precedes its start position \
                 {start_position}"
            ),
            ProvenanceError::EmptyName => {
                f.write_str("a named provenance's name must be non-empty")
            }
        }
    }
}

impl std::error::Error for ProvenanceError {}
