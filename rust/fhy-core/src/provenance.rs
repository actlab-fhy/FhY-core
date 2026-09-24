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
//! with [`Provenance::fuse`], or [`Provenance::fuse_labelled`] to name the
//! transformation, which drop unknown inputs and splice in the sources of
//! unlabelled fusions. A fusion that either builds therefore never
//! lists an unknown provenance or an unlabelled fusion among its own
//! sources; labelled fusions and the other variants are kept whole, whatever
//! they contain.
//!
//! [`Position`], [`Span`] and [`Provenance`] serialize through plain serde
//! derives, in any serde format. A position is `{"line": .., "column": ..}`,
//! a span names all four of its fields with `null` for an absent one, and a
//! provenance is tagged by its variant name, as in `"unknown"` or `{"file":
//! {..}}`. Decoding checks the same invariants as the constructors.

use std::fmt;
use std::hash::Hash;
use std::num::NonZeroU64;
use std::ops::Range;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

/// A 1-indexed line and column in a source text.
///
/// Positions order lexicographically, by line and then by column.
///
/// A position encodes as `{"line": .., "column": ..}`, and decoding refuses
/// a zero line or column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields, expecting = "a position")]
pub struct Position {
    line: NonZeroU64,
    column: NonZeroU64,
}

impl Position {
    /// Create the position at `line` and `column`.
    ///
    /// # Errors
    ///
    /// Returns [`PositionError::ZeroLine`] if `line` is zero, and otherwise
    /// [`PositionError::ZeroColumn`] if `column` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use fhy_core::provenance::Position;
    ///
    /// let position = Position::try_new(2, 8)?;
    ///
    /// assert_eq!(position.to_string(), "2:8");
    /// # Ok::<(), fhy_core::provenance::PositionError>(())
    /// ```
    pub fn try_new(line: u64, column: u64) -> Result<Self, PositionError> {
        let line = NonZeroU64::new(line).ok_or(PositionError::ZeroLine)?;
        let column = NonZeroU64::new(column).ok_or(PositionError::ZeroColumn)?;
        Ok(Self { line, column })
    }

    /// Return the 1-indexed line.
    #[must_use]
    pub fn line(&self) -> NonZeroU64 {
        self.line
    }

    /// Return the 1-indexed column.
    #[must_use]
    pub fn column(&self) -> NonZeroU64 {
        self.column
    }
}

/// Render the position as `line:column`, for example `2:8`.
impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.column)
    }
}

/// A range in a source text given by byte offsets, positions, or both.
///
/// Each of the four bounds is optional. When both offsets are set the end
/// is at or after the start, and likewise for the positions; the offsets
/// and the positions are never checked against each other. Start from
/// [`Span::unknown`], [`Span::from_offsets`] or [`Span::from_positions`],
/// and set the other bounds with the `with_*` builders.
///
/// # Examples
///
/// ```
/// use fhy_core::provenance::{Position, Span};
///
/// let span = Span::from_offsets(0..3)?
///     .with_positions(Position::try_new(1, 1)?..Position::try_new(1, 4)?)?;
///
/// assert_eq!(span.to_string(), "1:1-1:4");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// A span encodes as `{"start_offset", "end_offset", "start_position",
/// "end_position"}`, with `null` for an absent bound. Decoding reads a
/// missing key as an absent bound and refuses a pair of bounds out of
/// order, as the builders do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "SpanData")]
pub struct Span {
    start_offset: Option<u64>,
    end_offset: Option<u64>,
    start_position: Option<Position>,
    end_position: Option<Position>,
}

impl Span {
    /// Return the span with no bounds at all.
    #[must_use]
    pub const fn unknown() -> Self {
        Self {
            start_offset: None,
            end_offset: None,
            start_position: None,
            end_position: None,
        }
    }

    /// Create the span from `offsets.start` to `offsets.end`, with no
    /// positions.
    ///
    /// # Errors
    ///
    /// Returns [`SpanError::EndOffsetBeforeStart`] if `offsets.end <
    /// offsets.start`. An empty range is accepted.
    pub fn from_offsets(offsets: Range<u64>) -> Result<Self, SpanError> {
        Self::unknown().with_offsets(offsets)
    }

    /// Create the span from `positions.start` to `positions.end`, with no
    /// offsets.
    ///
    /// # Errors
    ///
    /// Returns [`SpanError::EndPositionBeforeStart`] if `positions.end <
    /// positions.start`. An empty range is accepted.
    pub fn from_positions(positions: Range<Position>) -> Result<Self, SpanError> {
        Self::unknown().with_positions(positions)
    }

    /// Return the span with both offsets replaced by `offsets.start` and
    /// `offsets.end`, keeping the positions.
    ///
    /// # Errors
    ///
    /// Returns [`SpanError::EndOffsetBeforeStart`] if `offsets.end <
    /// offsets.start`.
    pub fn with_offsets(self, offsets: Range<u64>) -> Result<Self, SpanError> {
        Self {
            start_offset: Some(offsets.start),
            end_offset: Some(offsets.end),
            ..self
        }
        .check_order()
    }

    /// Return the span with both positions replaced by `positions.start` and
    /// `positions.end`, keeping the offsets.
    ///
    /// # Errors
    ///
    /// Returns [`SpanError::EndPositionBeforeStart`] if `positions.end <
    /// positions.start`.
    pub fn with_positions(self, positions: Range<Position>) -> Result<Self, SpanError> {
        Self {
            start_position: Some(positions.start),
            end_position: Some(positions.end),
            ..self
        }
        .check_order()
    }

    /// Return the span with its start offset replaced by `offset`.
    ///
    /// # Errors
    ///
    /// Returns [`SpanError::EndOffsetBeforeStart`] if the end offset is set
    /// and precedes `offset`.
    pub fn with_start_offset(self, offset: u64) -> Result<Self, SpanError> {
        Self {
            start_offset: Some(offset),
            ..self
        }
        .check_order()
    }

    /// Return the span with its end offset replaced by `offset`.
    ///
    /// # Errors
    ///
    /// Returns [`SpanError::EndOffsetBeforeStart`] if the start offset is
    /// set and `offset` precedes it.
    pub fn with_end_offset(self, offset: u64) -> Result<Self, SpanError> {
        Self {
            end_offset: Some(offset),
            ..self
        }
        .check_order()
    }

    /// Return the span with its start position replaced by `position`.
    ///
    /// # Errors
    ///
    /// Returns [`SpanError::EndPositionBeforeStart`] if the end position is
    /// set and precedes `position`.
    pub fn with_start_position(self, position: Position) -> Result<Self, SpanError> {
        Self {
            start_position: Some(position),
            ..self
        }
        .check_order()
    }

    /// Return the span with its end position replaced by `position`.
    ///
    /// # Errors
    ///
    /// Returns [`SpanError::EndPositionBeforeStart`] if the start position
    /// is set and `position` precedes it.
    pub fn with_end_position(self, position: Position) -> Result<Self, SpanError> {
        Self {
            end_position: Some(position),
            ..self
        }
        .check_order()
    }

    /// Return the span if each pair of bounds that is fully set is in
    /// order, checking the offsets first.
    fn check_order(self) -> Result<Self, SpanError> {
        match (self.start_offset, self.end_offset) {
            (Some(start), Some(end)) if end < start => {
                return Err(SpanError::EndOffsetBeforeStart { start, end });
            }
            _ => {}
        }
        match (self.start_position, self.end_position) {
            (Some(start), Some(end)) if end < start => {
                Err(SpanError::EndPositionBeforeStart { start, end })
            }
            _ => Ok(self),
        }
    }

    /// Return whether none of the four bounds is set.
    #[must_use]
    pub fn is_unknown(&self) -> bool {
        *self == Self::unknown()
    }

    /// Return the byte offset the span starts at, if known.
    #[must_use]
    pub fn start_offset(&self) -> Option<u64> {
        self.start_offset
    }

    /// Return the byte offset the span ends at, if known.
    #[must_use]
    pub fn end_offset(&self) -> Option<u64> {
        self.end_offset
    }

    /// Return the position the span starts at, if known.
    #[must_use]
    pub fn start_position(&self) -> Option<Position> {
        self.start_position
    }

    /// Return the position the span ends at, if known.
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

/// Render `<unknown>` when no bound is set. Otherwise, when either position
/// is set, render `start-end` from the positions alone, ignoring the
/// offsets; else render `@start-end` from the offsets. A missing bound
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

/// The fields of a [`Span`] as decoded, before their order is checked.
#[derive(Deserialize)]
#[serde(deny_unknown_fields, expecting = "a span")]
struct SpanData {
    start_offset: Option<u64>,
    end_offset: Option<u64>,
    start_position: Option<Position>,
    end_position: Option<Position>,
}

impl TryFrom<SpanData> for Span {
    type Error = SpanError;

    fn try_from(data: SpanData) -> Result<Self, SpanError> {
        Span {
            start_offset: data.start_offset,
            end_offset: data.end_offset,
            start_position: data.start_position,
            end_position: data.end_position,
        }
        .check_order()
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
/// Decoding JSON text with `serde_json` refuses input nested more than 127
/// JSON levels deep with an error, not a crash; a named or call-site level
/// takes two JSON levels and a fused level three. A format that is not
/// self-describing, such as postcard, has no such limit: decoding untrusted
/// bytes there recurses as deep as the input nests, at about two bytes per
/// named level, and can overflow the stack. A caller decoding untrusted
/// input bounds its size, or uses a format with a depth limit.
///
/// # Serialization
///
/// A provenance encodes externally tagged by its snake-case variant name:
/// the unknown provenance as `"unknown"`, and the other variants as a
/// one-key map such as `{"file": {"file_path": .., "span": ..}}`,
/// `{"named": {"name": .., "child": ..}}`, `{"call_site": {"callee": ..,
/// "caller": ..}}` and `{"fused": {"sources": [..], "label": ..}}`.
/// Decoding normalizes a file path and refuses an empty name.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
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
    /// Combine `provenances` into one unlabelled provenance, keeping their
    /// order.
    ///
    /// The inputs are flattened first: every [`Provenance::Unknown`] is
    /// dropped, and an unlabelled [`FusedProvenance`] is replaced by its
    /// sources, at any depth. A labelled fusion, even one labelled with the
    /// empty string, and every other variant are kept whole. Equal inputs
    /// are not deduplicated.
    ///
    /// When nothing survives the result is [`Provenance::Unknown`], when
    /// exactly one input survives it is that input, and otherwise it is an
    /// unlabelled [`FusedProvenance`] of the survivors.
    ///
    /// [`Provenance::Unknown`] is therefore an identity for `fuse`, and
    /// `fuse` is associative. The flattening walks an explicit stack, so
    /// deeply nested unlabelled fusions do not exhaust the call stack.
    ///
    /// # Examples
    ///
    /// ```
    /// use fhy_core::provenance::{FileProvenance, Provenance};
    ///
    /// let a = Provenance::File(FileProvenance::new("a.fhy", None));
    /// let b = Provenance::File(FileProvenance::new("b.fhy", None));
    ///
    /// let fused = Provenance::fuse([a.clone(), Provenance::Unknown, b]);
    ///
    /// assert_eq!(fused.to_string(), "fused[a.fhy, b.fhy]");
    /// assert_eq!(Provenance::fuse([a.clone()]), a);
    /// ```
    #[must_use]
    pub fn fuse(provenances: impl IntoIterator<Item = Provenance>) -> Provenance {
        let flat = flatten_fusion_inputs(provenances);
        match <[Provenance; 1]>::try_from(flat) {
            Ok([single]) => single,
            Err(flat) if flat.is_empty() => Provenance::Unknown,
            Err(flat) => Provenance::Fused(FusedProvenance::new(flat)),
        }
    }

    /// Combine `provenances` into one provenance labelled `label`, keeping
    /// their order.
    ///
    /// The inputs are flattened as in [`Provenance::fuse`]. When nothing
    /// survives the result is [`Provenance::Unknown`]; otherwise it is a
    /// [`FusedProvenance`] of the survivors labelled `label`, even for a
    /// single survivor. The empty string is a label.
    ///
    /// [`Provenance::Unknown`] is an identity for `fuse_labelled` too, but
    /// it is not associative: a labelled result nested in another fusion is
    /// kept whole.
    ///
    /// # Examples
    ///
    /// ```
    /// use fhy_core::provenance::{FileProvenance, Provenance};
    ///
    /// let a = Provenance::File(FileProvenance::new("a.fhy", None));
    /// let b = Provenance::File(FileProvenance::new("b.fhy", None));
    ///
    /// let fused = Provenance::fuse_labelled([a, Provenance::Unknown, b], "loop-fusion");
    ///
    /// assert_eq!(fused.to_string(), "loop-fusion[a.fhy, b.fhy]");
    /// ```
    #[must_use]
    pub fn fuse_labelled(
        provenances: impl IntoIterator<Item = Provenance>,
        label: impl Into<String>,
    ) -> Provenance {
        let flat = flatten_fusion_inputs(provenances);
        if flat.is_empty() {
            Provenance::Unknown
        } else {
            Provenance::Fused(FusedProvenance::labelled(flat, label))
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
            Provenance::Fused(fused) if fused.label.is_none() => {
                pending.extend(fused.sources.into_vec().into_iter().rev());
            }
            survivor => flat.push(survivor),
        }
    }
    flat
}

/// Render a human-readable description: `<unknown>`, the file variant's
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
                f.write_str(fused.label().unwrap_or("fused"))?;
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

/// Return `path` in its lexical normal form.
///
/// `/` is the only separator; every other character, a backslash or a
/// drive letter's colon included, is part of a component. Empty and `.`
/// components are removed, which drops repeated and trailing separators,
/// `..` components are kept, a root of exactly two separators (`//`) is
/// kept while one or three or more leading separators become the root `/`,
/// and a path with no root and no components becomes `.`. The result is the
/// same on every platform, and normalizing twice changes nothing.
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
/// The path is text stored in a lexical normal form, the same on every
/// platform: `/` is the only separator, repeated separators and `.`
/// components are removed and a trailing separator is dropped, so `./a` and
/// `a//b/` become `a` and `a/b`, and the empty path becomes `.`. A root of
/// exactly two separators stays `//`, while three or more leading
/// separators become `/`. A `..` component is never resolved, and `~`, a
/// backslash and a drive letter such as `C:` are ordinary characters, so
/// `C:\src\a.fhy` is one component. Equality, hashing and
/// [`Display`](fmt::Display) use the normalized text, so `//a` and `/a`
/// differ.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(from = "FileProvenanceData")]
pub struct FileProvenance {
    file_path: String,
    span: Option<Span>,
}

impl FileProvenance {
    /// Create the provenance for `span` in the file at `file_path`, storing
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

    /// Return the normalized path of the file.
    #[must_use]
    pub fn file_path(&self) -> &str {
        &self.file_path
    }

    /// Return the region of the file, if one was given.
    #[must_use]
    pub fn span(&self) -> Option<&Span> {
        self.span.as_ref()
    }
}

/// The fields of a [`FileProvenance`] as decoded, before the path is
/// normalized.
#[derive(Deserialize)]
#[serde(deny_unknown_fields, expecting = "a file provenance")]
struct FileProvenanceData {
    file_path: String,
    span: Option<Span>,
}

impl From<FileProvenanceData> for FileProvenance {
    fn from(data: FileProvenanceData) -> Self {
        FileProvenance::new(data.file_path, data.span)
    }
}

/// A child provenance under a human-readable name, such as a builtin over
/// [`Provenance::Unknown`] or a library symbol over the library's file.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "NamedProvenanceData")]
pub struct NamedProvenance {
    name: String,
    child: Arc<Provenance>,
}

impl NamedProvenance {
    /// Create the provenance naming `child` as `name`.
    ///
    /// # Errors
    ///
    /// Returns [`NamedProvenanceError::EmptyName`] if `name` is empty. A
    /// name of only whitespace is not empty.
    pub fn try_new(
        name: impl Into<String>,
        child: Provenance,
    ) -> Result<Self, NamedProvenanceError> {
        let name = name.into();
        if name.is_empty() {
            return Err(NamedProvenanceError::EmptyName);
        }
        Ok(Self {
            name,
            child: Arc::new(child),
        })
    }

    /// Return the name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return the named provenance.
    #[must_use]
    pub fn child(&self) -> &Provenance {
        &self.child
    }
}

/// The fields of a [`NamedProvenance`] as decoded, before the name is
/// checked.
#[derive(Deserialize)]
#[serde(deny_unknown_fields, expecting = "a named provenance")]
struct NamedProvenanceData {
    name: String,
    child: Provenance,
}

impl TryFrom<NamedProvenanceData> for NamedProvenance {
    type Error = NamedProvenanceError;

    fn try_from(data: NamedProvenanceData) -> Result<Self, NamedProvenanceError> {
        NamedProvenance::try_new(data.name, data.child)
    }
}

/// Provenance of a value created at a call site, such as by inlining or
/// macro expansion.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields, expecting = "a call-site provenance")]
pub struct CallSiteProvenance {
    callee: Arc<Provenance>,
    caller: Arc<Provenance>,
}

impl CallSiteProvenance {
    /// Create the provenance of a value from `callee` created at `caller`.
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

    /// Return the provenance of the called code.
    #[must_use]
    pub fn callee(&self) -> &Provenance {
        &self.callee
    }

    /// Return the provenance of the call site.
    #[must_use]
    pub fn caller(&self) -> &Provenance {
        &self.caller
    }
}

/// Several provenances combined by a transformation, with an optional label.
///
/// The constructors keep their sources as given: they accept no sources, a
/// single source, and nested unknown or unlabelled sources. Use
/// [`Provenance::fuse`] or [`Provenance::fuse_labelled`] to build the flat
/// form.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields, expecting = "a fused provenance")]
pub struct FusedProvenance {
    sources: Box<[Provenance]>,
    label: Option<String>,
}

impl FusedProvenance {
    /// Create the unlabelled fusion of `sources`, in order.
    #[must_use]
    pub fn new(sources: Vec<Provenance>) -> Self {
        Self {
            sources: sources.into_boxed_slice(),
            label: None,
        }
    }

    /// Create the fusion of `sources`, in order, labelled `label`.
    #[must_use]
    pub fn labelled(sources: Vec<Provenance>, label: impl Into<String>) -> Self {
        Self {
            sources: sources.into_boxed_slice(),
            label: Some(label.into()),
        }
    }

    /// Return the fused provenances, in order.
    #[must_use]
    pub fn sources(&self) -> &[Provenance] {
        &self.sources
    }

    /// Return the label, or `None` for an unlabelled fusion.
    #[must_use]
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }
}

/// Types that carry the provenance of the object they represent.
pub trait HasProvenance {
    /// Return the object's provenance.
    #[must_use]
    fn provenance(&self) -> &Provenance;
}

/// A [`Position`] was built from a zero line or column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PositionError {
    /// The line was zero; lines start at 1.
    ZeroLine,
    /// The column was zero; columns start at 1.
    ZeroColumn,
}

impl fmt::Display for PositionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PositionError::ZeroLine => f.write_str("a position's line must be at least 1"),
            PositionError::ZeroColumn => f.write_str("a position's column must be at least 1"),
        }
    }
}

impl std::error::Error for PositionError {}

/// A [`Span`] was given a pair of bounds out of order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SpanError {
    /// The end offset preceded the start offset.
    EndOffsetBeforeStart {
        /// The offset the span was to start at.
        start: u64,
        /// The offset the span was to end at.
        end: u64,
    },
    /// The end position preceded the start position.
    EndPositionBeforeStart {
        /// The position the span was to start at.
        start: Position,
        /// The position the span was to end at.
        end: Position,
    },
}

impl fmt::Display for SpanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpanError::EndOffsetBeforeStart { start, end } => {
                write!(
                    f,
                    "a span's end offset {end} precedes its start offset {start}"
                )
            }
            SpanError::EndPositionBeforeStart { start, end } => write!(
                f,
                "a span's end position {end} precedes its start position {start}"
            ),
        }
    }
}

impl std::error::Error for SpanError {}

/// A [`NamedProvenance`] was given an empty name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum NamedProvenanceError {
    /// The name was empty.
    EmptyName,
}

impl fmt::Display for NamedProvenanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NamedProvenanceError::EmptyName => {
                f.write_str("a named provenance's name must be non-empty")
            }
        }
    }
}

impl std::error::Error for NamedProvenanceError {}
