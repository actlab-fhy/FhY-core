# Rust workspace hardening spec

- **Status:** draft for sign-off.
- **Base:** `dev-rust` at `412e234`.
- **Audit:** `docs/audit/rust-workspace.md`, whose Decisions 1–13 and Triage
  this spec implements.

This document is the contract for the test-writing and implementation
phases. Part I is the cross-cutting design. It overrides the batch sections
in Part II wherever they disagree, and each override is listed in §I.4.
Part II holds the six batch sections: public interface, interface delta,
behavior, error model and test plan for each area.

## Contents

- Part I: cross-cutting design
  - I.1 Summary
  - I.2 Target layout and layering
  - I.3 Cross-cutting rules
  - I.4 Integration resolutions (these override Part II)
  - I.5 Decisions for sign-off
  - I.6 Implementation order
  - I.7 Findings resolution
  - I.8 Non-goals
- Part II: batch sections
  - B1: identity, interning and tag types
  - B2: serialization, diagnostics, provenance
  - B3: expression core (`fhy_core::expr`)
  - B4: patterns, rewrite rules, expression passes
  - B5: pass infrastructure (`fhy_core::pass`) and `fhy_core::tree`
  - B6: workspace, tests, docs, bindings, CONTRIBUTING

---

# Part I: cross-cutting design

## I.1 Summary

`fhy-core` becomes a Rust library whose shape follows Rust conventions
rather than the Python package it was ported from. Python-identical
behavior is kept only where a concept is defined in both languages at once:
`identifier` and `interned`. This is the result:

- **Identity.**
  - Identifier ids stay exact across serialization.
  - Built-in tags get fixed reserved ids (0..1024), and payload ids are
    capped below 2^63, so no input can exhaust the id space.
  - The id counter and the intern registries stay process-global and
    append-only, and cannot be cleared.
- **Layering.** There are no dependency cycles:
  `identifier → interned → {diagnostic, provenance, op_attribute,
  value_domain, described_tag, tree} → expr → pass`, with `expr::passes`
  above both.
- **Serialization.**
  - Plain serde derives, with no envelope, no decode framework and no
    `arbitrary_precision`.
  - Every serialized type round-trips through JSON and postcard.
- **Python-shaped API removed.** This covers stringly errors, `Option`-mode
  constructors, `get_`/`list_` names, string-keyed captures and built-ins,
  Python float text, and the global pass registry and run counters.
- **Correctness fixes:**
  - n-ary logical nodes and a bounded `Debug` (F-003);
  - identity rewrites count as no change (F-009);
  - `%` removed (F-010);
  - the test-isolation flakes fixed (F-032, F-033).

## I.2 Target layout and layering

Final public modules. Each item has exactly one public path, and there are
no glob re-exports.

| Module | Contents | Was |
|---|---|---|
| `fhy_core::identifier` | `Identifier`, `RESERVED_ID_COUNT`, `ID_CAP`, errors, binding-only counter functions | same |
| `fhy_core::interned` | `Interned`, `InternRegistry`, `Canonical`, errors | same |
| `fhy_core::described_tag` | `DescribedTag<K>`, `TagKind` (sealed) | private macro module |
| `fhy_core::op_attribute` | `OpAttribute` = `DescribedTag<OpAttributeVocabulary>` | same |
| `fhy_core::value_domain` | `ValueDomain` | same |
| `fhy_core::diagnostic` | `Diagnostic`, `Note`, `NoteKind`, `ValidationReport`, … | same |
| `fhy_core::provenance` | `Position`, `Span`, `Provenance`, … | same |
| `fhy_core::tree` | `Tree`, `NodeHandle`, `NodeIdentity`, `TreeVisitor<N, C = ()>`, `Rewriter<N, C = ()>`, `walk_tree`, `rewrite_tree`, `TraversalOrder`, `RewriteTreeError` | `pass_infrastructure::{tree, analysis}` |
| `fhy_core::expr` | `Expression`, node kinds, builders, literals, operations, `BooleanScreen`, `ExpressionDisplay`, `BigInt` | `symbolic::expression`, `symbolic::symbol_type` |
| `fhy_core::expr::builtins` | `BuiltinFunction`, `BuiltinConstant` | `symbolic::expression::builtins` |
| `fhy_core::expr::pattern` | `Pattern`, `Capture`, `MatchBindings`, `Rule`, `RewriteRule`, `apply_rewrite_rules`, … | `symbolic::expression::pattern` |
| `fhy_core::expr::passes` | `RewriteRuleApplier`, `ExpressionPrettyFormatter`, `register_expression_passes(&mut PassRegistry)` | `symbolic::expression::{registration, pattern::rewrite}` |
| `fhy_core::pass` | `CompilerPass`, `PassManager`, `PassRegistry`, `Validator`, `WalkPass`, `RewritePass`, errors, … | `pass_infrastructure` |

**Layering rules:**
- `tree` depends only on `std`.
- `pass` depends on `tree` and `diagnostic`, never on `expr`.
- In `expr`, only `expr::passes` may import `pass`.
- CI checks the public paths (B6 §5.4). The layering rules are checked by
  grep in review.

## I.3 Cross-cutting rules

These apply to every batch.

1. **Parity boundary (decision 3).**
   - Only `identifier` and `interned` keep Python-identical behavior and
     text. Code that exists only for that parity carries a doc line
     beginning "Matches the Python implementation:".
   - Everything else follows Rust conventions: lowercase one-line error
     messages without a trailing period, `true`/`false`, and Rust float
     formatting.
2. **Serde (decision 4, B2 §5.1).** Plain derives with Rust-defined shapes.
   - **Forbidden in `src`:** `tag`, `untagged`, `flatten`,
     `skip_serializing_if`, `deserialize_any`, and any `serde_json` type.
     `serde_json` becomes a dev-dependency.
   - **Formats:** every serialized type round-trips through JSON and
     postcard.
   - **Big integers:** `BigInt` serializes as a decimal string.
   - **Side effects:** deserializing an `Identifier` advances the counter,
     and deserializing a `Canonical<T>` interns. A decode that fails partway
     may leave both effects behind. This is documented, not prevented.
   - **Envelope:** the `__type__`/`__data__` envelope lives only in
     `fhy-core-py`.
3. **Errors (S-5).**
   - Public errors are `#[non_exhaustive]`, with structured fields.
   - `Display` is one line and never repeats `source()`.
   - There is one error type per operation family, with no `is_*`
     classifiers where a `kind()` or a direct match would do.
4. **Enums.** Enums that will grow are `#[non_exhaustive]`. Enums that
   passes match exhaustively stay exhaustive, and the batch sections list
   them. The lints `exhaustive_enums`/`exhaustive_structs` (B6 §2.6) are
   added at the end, with an `#[expect(..., reason)]` on each deliberate
   exception.
5. **Naming.**
   - No `get_`/`list_` prefixes.
   - Shipped defaults are associated functions, such as
     `OpAttribute::commutative()`.
   - Operations are methods on their receiver.
   - Predicates start with `is_`, `has_` or `contains`.
6. **Semver.** The crate is unpublished (decision 9), so every change is
   allowed. Each batch still classifies its changes and lists the in-repo
   call sites.
7. **Threading (decision 10).**
   - Stored passes and validators are `+ Send`.
   - IR handles stay `Send + Sync`.
   - `PassRegistry` is `Send + Sync`.

## I.4 Integration resolutions (these override Part II)

| # | Conflict or gap | Resolution | Affected text |
|---|---|---|---|
| R-1 | Where the expression passes live: B4 says `fhy_core::pass::expr`, B3 and B6 say `fhy_core::expr::passes` | **`fhy_core::expr::passes`.** The generic pass framework never depends on `expr`, and passes live with the IR they transform. B4 §2.9 and every `pass::expr` path in B4 read as `expr::passes`. B4's "nothing under `expr` imports `pass`" becomes "only `expr::passes` imports `pass`" | B4 |
| R-2 | Ids for built-in function parameters: B1 reserves ids 64–90 (`MAX_PARAMETERS` … `GELU_PARAMETERS`), while B3 keeps them lazy | **Lazy (B3).** `ID_CAP` removes the exhaustion that forced eager creation, and the parameters are never serialized. Delete the parameter rows and constants from B1's reserved table. The table then covers only the 10 shipped tags (note kinds 0–3, op attributes 16–19, value domains 32–33). B6 step 4's "built-in parameters follow B1's reserved ids" is void | B1 §2.2, B6 §5.1 |
| R-3 | How far the Python change reaches: B1 D-5 moves Python's shipped tags onto the reserved ids and adds a `reserved_identifiers.json` corpus | **Counter and cap only.** Python's `identifier.py` starts its fallback counter at 1024 and applies the 2^63 cap with the same errors, and nothing more. Python's own tags are created at import in a fixed order, so their ids are already stable, and they will be replaced by the Rust tags later. Drop the Python changes to `op_attribute.py`, `diagnostic.py`, `value_domain.py` and `builtins.py`, the new corpus and its noxfile change | B1 §2.9, §8, §9 |
| R-4 | B1 N-1: decoding a `Canonical<ValueDomain>` recurses once per parent level, and postcard has no nesting limit | **Fix in B1.** The wire form of a value domain is a flat, root-first list of `{name, description}` ancestors, decoded iteratively. Regression test: `value_domain_decodes_a_deep_chain_on_a_small_stack` | B1 §2.7, §8 |
| R-5 | B2: deep `Provenance` trees recurse in `Drop`/`Eq`/`Hash`/serde | **Document, don't restructure.** Real provenance chains are shallow. The `Provenance` rustdoc states the recursion, and a non-goal records it | B2 §7 |
| R-6 | B5: the verified-identity set will almost never get a hit either | **Remove verification caching entirely.** Every changed output is verified once when it is produced. This is simpler, and no measurable cost was shown | B5 §2.4, §5, §8 |
| R-7 | The clippy break at `tests/expression_pass_stories.rs:287`: B3 says use `assert_eq!` after the bounded `Debug`, B6 says use `#[expect]` first | **Step 0.1 uses `#[expect(clippy::manual_assert_eq, reason = "Debug of a 64-level doubling DAG never finishes")]`.** B3 removes the `#[expect]` and switches to `assert_eq!` when the bounded `Debug` lands | B3 §8, B6 §5.1 |
| R-8 | B6 D12: where `interned_equivalence` goes depends on B1's `clear` | B1 chose `clear(&mut self)`, which works on local registries, so **`interned_equivalence` merges into `tests/it`**. No separate test binaries remain | B6 §5.2 |
| R-9 | B2 left `manager.rs:342`'s inlined report detail and `PassContext::report`'s `Option<String>` detail parameter to B5 | **B5 owns them.** The verification diagnostic carries a short message, and the report is reachable through `PassErrorKind::Verification`. `PassContext::report` takes a `Diagnostic` built with B2's constructors | B5 |
| R-10 | B5 needs `Diagnostic::source` to take `Cow<'static, str>` | **B2 adopts it:** `Diagnostic` stores `source: Cow<'static, str>` | B2 §2.1 |
| R-11 | F-036: B6 found that `tests/test_golden_corpora.py` already regenerates every corpus in CI, and a `git diff` step would always fail on the provenance block | **F-036 resolved as already covered.** Its symbolic-corpus part is void under decision 3. No new CI step | B6 §5.3 |
| R-12 | F-030: B6 found that `tests/test_rs_stub.py` already checks the stub | **The stub-check part of F-030 is already covered.** The declarative module and error trait remain | B6 §2.4 |
| R-13 | Stack depth of `Drop` for deep expressions: B3 keeps `Drop` iterative | No change; stated for completeness | none |
| R-14 | D-1 changes the reserved block size | `RESERVED_ID_COUNT = 65_536`. B1's family sub-blocks keep their numbers (note kinds 0–3, op attributes 16–19, value domains 32–33). The Python fallback counter starts at 65,536 | B1 §2.1, §2.9 |
| R-15 | D-4 | Add `// TODO: warn here once the log dependency is added` where a differing description is dropped | B1 §5.5 |
| R-16 | D-7 | The float literal wire form is `{"float":"<shortest round-trip text>"}`; decode parses with `f64::from_str`. Serialization never fails for floats | B3 §5.6 |
| R-17 | D-8 | Keep `UnaryOperation::Positive` and the `Expression::positive()` builder (Rust has no unary `+` operator to overload). Strike every B3 row that removes it | B3 |
| R-18 | D-10 | `CallbackError` is a type alias for the boxed error. B4 §2.4's wrapper type, blanket `From` and conversions are dropped | B4 §2.4 |
| R-19 | D-11 | `RewriteRule` stores `guards: Vec<GuardFn>`, checked in order before the rewrite; add `new_partial` | B4 §2.6 |
| R-20 | D-13 | `PassRegistry::register::<P, I, O>(factory)`; `PassRegistrationError` loses `NameTaken`'s alias semantics (same name, different identity: still `NameTaken`) | B5 §2.5, B4 §5.5 |
| R-21 | D-14 | `Analysis` gets an associated `Ir`; `PassContext::analysis::<A>(ir: &A::Ir)`; the tests' `LabelAnalysis` becomes generic over its IR | B5 §2.4 |
| R-22 | D-15 | Proper pluralization in `ValidationFailedError` | B2 §2.1 |
| R-23 | D-16 | Keep the leading `//` | B2 §5 |
| R-24 | D-18 | `mod_module_files`, not `self_named_module_files`. Every `foo/mod.rs` becomes `foo.rs` in step 0.3/0.4 alongside the renames | B6 §2.6, §5.1 |
| R-25 | D-20 | CONTRIBUTING migration note | B6 §6 |

## I.5 Signed-off decisions (2026-09-24)

The user decided each of these individually. Where a decision differs from
the Part II text, this table governs; each such difference is also listed in
§I.4 as R-14 to R-24.

| # | Decision | Differs from Part II? |
|---|---|---|
| D-1 | `RESERVED_ID_COUNT = 65_536` (ids 0..65,535 reserved; both backends' counters start at 65,536); `ID_CAP = 2^63` | yes (R-14): B1 had 1024 |
| D-2 | `Canonical<T>` compares and hashes by key, matching Python's `==`; `Canonical::ptr_eq` for instance identity | no |
| D-3 | `_advance_counter_past` (both backends) raises `OverflowError` for any id outside [0, 2^63); public deserialization raises `DeserializationValueError` "a non-negative integer below 2**63" | no |
| D-4 | Re-registering a tag or value domain with the same name (and parent) and a different description returns the existing handle; the first description wins. A `// TODO:` at each place the new description is dropped (registration and `Canonical` decode) says to warn there once the planned `log` dependency is added | yes (R-15): adds the TODOs |
| D-5 | `Expression::all([])` is `true`, `any([])` is `false`, `all([x])` is `x`; the builders never fail; nested `and`/`or` is not flattened | no |
| D-6 | The expression wire format is the flat post-order node list with index references (B3 §5.6) | no |
| D-7 | Every float literal serializes as a string in Rust's shortest round-trip `{}` form (`"1.5"`, `"NaN"`, `"inf"`, `"-inf"`), so non-finite floats serialize in every format | yes (R-16): B3 refused non-finite floats |
| D-8 | `UnaryOperation::Positive` is **kept**. It asserts a numeric operand, which the Python type checker (`type_checker.py:662`) and the screen rely on | yes (R-17): B3 removed it |
| D-9 | Built-in names are reserved: `FunctionName` rejects them, and the screen uses the built-in catalogue's sorts | no |
| D-10 | `pub type CallbackError = Box<dyn std::error::Error + Send + Sync + 'static>;` with no custom wrapper; `RewriteError::source()` returns the boxed error | yes (R-18): B4 had a custom anyhow-style type |
| D-11 | Guards combine: all must pass, checked in the order added (stored as a list). Two constructors: `RewriteRule::new(pattern, f)` with `f -> Result<Expression, CallbackError>`, and `RewriteRule::new_partial(pattern, f)` with `f -> Result<Option<Expression>, CallbackError>`, where `None` declines | yes (R-19): B4 had last-added-first and one constructor |
| D-12 | The rewrite-rule pass keeps the registry name `fhy_core.symbolic.expression.apply_rewrite_rules`; the docs say it is a stable key, not a Rust path | no |
| D-13 | A pass's registry key is its own `name()`: `PassRegistry::register::<P, I, O>(factory)` reads `name()` and `description()` from one factory instance. There is no name argument and no aliases. Blank means `char::is_whitespace` | yes (R-20): B5 had decoupled keys and aliases |
| D-14 | `Analysis` uses an associated IR type: `trait Analysis: 'static { type Ir; type Output: Send + Sync + 'static; fn run(&self, ir: &Self::Ir) -> Self::Output; }`; `preserve::<A: Analysis>()`; no `AnalysisMarker` | yes (R-21): B5 had the marker trait |
| D-15 | Report lines are `error[source]: message` with the detail indented beneath; an empty report displays as the empty string; `ValidationFailedError` displays "validation failed with 1 error" or "with N errors" | yes (R-22): proper plural |
| D-16 | The path normalization rules stay as they are, **including a kept leading `//`**, and the docs describe the rules rather than citing Python. `fuse(sources)` and `fuse_labelled(sources, label)`; the field and wire key are `label` | yes (R-23): B2 collapsed `//` |
| D-17 | postcard is the non-JSON test format (dev-dependency). bincode 3.0.0 is a `compile_error!` placeholder (checked) | no |
| D-18 | Lints added in step 6: `clippy::exhaustive_enums`, `clippy::exhaustive_structs`, `clippy::cargo_common_metadata`, and **`clippy::mod_module_files`** (the `foo.rs` + `foo/` style). No nextest in CI | yes (R-24): B6 had `self_named_module_files` |
| D-19 | The published crate ships `src/`, `tests/`, the interned golden JSON, the README and the license, and never the Python generators | no |
| D-20 | The renames and the CONTRIBUTING text land in step 0. CONTRIBUTING carries a one-line note, "migration in progress, see `docs/design/rust-workspace.md`", which step 6 removes | yes (R-25): adds the note |

## I.6 Implementation order

This follows B6 §5.1, adjusted by R-2 and R-7. Each step leaves the
workspace green: fmt, clippy `-D warnings`, tests and doc.

| Step | Content | Needs |
|---|---|---|
| 0.1 | `#[expect]` on the clippy break (R-7) | none |
| 0.2 | CONTRIBUTING "Porting to Rust" replacement text (B6 §6) | none |
| 0.3 | Rename `symbolic` to `expr`, and `pattern/core.rs` to `matching.rs` | 0.1 |
| 0.4 | Rename `pass_infrastructure` to `pass` (tree items stay for now) | 0.3 |
| 0.5 | Merge the isolation-safe integration tests into `tests/it` | 0.4 |
| 1 | B1: identity, interning, tags, Python counter | 0 |
| 2 | B2: serde, diagnostics, provenance | 1 |
| 3 | B5: `tree` and `pass` | 0; may overlap step 2 once B2 freezes `Diagnostic`'s API |
| 4 | B3: expression core; create `expr::passes` | 1, 2, 3 |
| 5 | B4: patterns, rule applier into `expr::passes` | 3, 4 |
| 6 | B6 finish: last test merges, CI checks, lints, docs, manifest | 1–5 |

Each finding follows the test-first discipline:
1. Its tests from the batch's test plan are written first and fail against
   the current code.
2. The fix makes them pass.
3. The fix gets one commit whose body names the finding.

## I.7 Findings resolution

| F | Sev | Resolution | Batch | Notes |
|---|---|---|---|---|
| F-001 | High | Fix | B1 | cap at 2^63, `try_new`, `try_restore` |
| F-002 | High | Fixed already | none | 4db96b9; its tests stay |
| F-003 | High | Fix | B3 | n-ary `Logical`, bounded `Debug`, flat wire format |
| F-004 | High | Fix (narrowed) | B1 | append-only globals; `clear(&mut self)` |
| F-005 | High | Defer | none | decision 5; the CONTRIBUTING rule is kept and noted |
| F-006 | High | Fix | B5 | owned `PassRegistry`; pure names; per-run statistics |
| F-007 | High | Fix | B1 | reserved ids for 10 tags; `shipped.rs` deleted (R-2) |
| F-008 | High | Fix | B5 | `fhy_core::tree`, generic context |
| F-009 | Medium | Fix | B5, B4 | identity result counts as no change, in both the tree and the rules |
| F-010 | Medium | Fix | B3 | `%` removed; `floor_mod()` |
| F-011 | Medium | Fix | B2 | decode framework deleted; side effects documented |
| F-012 | Medium | Fix | B2 (+B1, B3) | `arbitrary_precision` removed; postcard round trips |
| F-013 | Medium | Fix | B2, B3 | `python_text.rs` deleted; normalization moves to `literal` |
| F-014 | Medium | Fix | B2–B5 | structured errors per module |
| F-015 | Medium | Fix | B2, B3 | `ExpressionDisplay`, report `Display` |
| F-016 | Medium | Fix (partial) | B5 | merge-only transfer; verification cache removed (R-6); pinning stays |
| F-017 | Medium | Fix | B5 | `Send` passes |
| F-018 | Medium | Fix | B1 | `testing` feature and scope deleted |
| F-019 | Medium | Fix | B4 | typed `Capture`, named constructors, trail bindings |
| F-020 | Medium | Fix | B1 | key equality; no bare `Deserialize` |
| F-021 | Medium | Fix | B1, B2 | `register*`, `Diagnostic::error`, span ranges, `fuse_labelled` |
| F-022 | Medium | Fix (partial) | B5 | `skip` hook, `Validator`, `AnalysisMarker`; dependent analyses deferred |
| F-023 | Medium | Fix | B3 | `BooleanScreen` builder; memo for nested piecewise (DAG part fixed in 4db96b9) |
| F-024 | Medium | Fix | B3 | `BuiltinFunction`/`Callee`, `From` conversions, `Not` |
| F-025 | Medium | Fix | B3 | serde derive + `FromStr`; `wire_name.rs` deleted |
| F-026 | Medium | Fix | B3 | normalized `LiteralValue` enum |
| F-027 | Low | Fix | B1 | `DescribedTag<K>` |
| F-028 | Low | Fix | B6 | layout of §I.2; single crate |
| F-029 | Low | Fix | B1, B3, B4 | naming |
| F-030 | Low | Fix (partial) | B6 | declarative module and `IntoPyErr`; stub check already covered (R-12); cache and split deferred |
| F-031 | Medium | Fix | B1, B5 | guards, unique pass types and absolute counts removed |
| F-032 | Medium | Fix | B1 | goes away with F-018 |
| F-033 | Medium | Fix | B1 | tests pin already-issued ids |
| F-034 | Medium | Fix | B6 | one `tests/it` binary (R-8) |
| F-035 | Medium | Fix | B2–B5 | tests assert variants, not message text; each weakening is flagged in its batch |
| F-036 | Medium | Resolved: already covered | none | R-11 |
| F-037 | Low | Fix | B3, B4, B6 | the hanging arms, HashMap order, hash inequality, re-implemented SUT |
| F-038 | Low | Fix | B1, B6 | harness deleted |
| F-039 | Low | Fix | B1, B4, B5 | read-lock fast path, identity hasher, `Cow` names |
| F-040 | Low | Fix | B6 | READMEs, crate docs, manifest metadata |
| N-1 | Low | Fix | B1 | flat value-domain chain (R-4) |
| N-2 | Low | Defer | none | registries grow without bound when decoding untrusted input |
| N-3 | Low | Defer | none | deep `Provenance` recursion (R-5) |
| N-4 | Low | Fix | B6 | CI never ran `cargo doc -D warnings` |

## I.8 Non-goals

- The binding-crate split into an rlib plus a cdylib, and a PyCapsule API
  (F-005).
- The canonical-object identity cache in the binding (F-030).
- Analyses that depend on other analyses (F-022).
- Unpinning analysed nodes during a pipeline run (F-016).
- Bounding intern registry growth on untrusted input (N-2).
- Making deep `Provenance` non-recursive (N-3).
- Splitting the crate (decision 13).
- Porting Python modules that are not yet in Rust (constraint, param,
  types, symbol_table, term).

---

# Part II: batch sections

The batch sections below are the detailed contracts. Where they disagree
with Part I, Part I governs (§I.4).


## B1: identity, interning and tag types

Scope: `rust/fhy-core/src/{identifier.rs, interned.rs, described_tag.rs,
op_attribute.rs, value_domain.rs, shipped.rs, testing.rs, test_support.rs}`,
the `NoteKind` part of `diagnostic.rs`, the reserved ids of the composed
built-ins in `symbolic/expression/builtins.rs`, `rust/fhy-core-py/src/*`,
and `src/fhy_core/identifier.py` (plus the four Python modules that own
shipped identifiers). All "before" signatures were checked against HEAD
`412e234`.

### 1. Summary

Identity stays process-global and append-only (decision 2), but it is now
predictable. Every shipped identifier gets a fixed id in a reserved block
`0..1024`. The same ids are used in Rust and Python. The counter starts at
1024, and a payload id at or above `2^63` is rejected, so no payload can
exhaust the id space (F-001, F-007).

The tag types become one generic `DescribedTag<K>` plus `ValueDomain`.
Their constructors register the value and return `Canonical<Self>` (or a
conflict error). Handles and values compare by key. Deserialize is plain
serde, implemented only for `Canonical<Tag>`.

The `testing` feature, the deterministic-identifier scope, `shipped.rs` and
all of the test machinery that isolated tests from global ids and registries
are deleted.

**Module placement (S-1):** `identifier` is again a leaf. It gains a
crate-private `identifier::reserved` table and imports nothing else from the
crate. `described_tag` becomes a public module that sits between `interned`
and `{diagnostic, op_attribute}`. `value_domain` depends on `identifier` and
`interned` only.

### 2. Desired public interface

#### 2.1 `fhy_core::identifier`

```rust
/// Number of ids reserved for shipped identifiers. Ids `0..RESERVED_ID_COUNT`
/// are the fixed ids of the identifiers this crate ships (see the reserved
/// table); the counter issues fresh ids from `RESERVED_ID_COUNT` upward.
/// Matches the Python implementation: `fhy_core.identifier._RESERVED_ID_COUNT`.
pub const RESERVED_ID_COUNT: u64 = 1024;                                  // NEW

/// Exclusive upper bound of a payload id. A deserialized or restored id must
/// be below it; fresh ids may exceed it.
/// Matches the Python implementation: `fhy_core.identifier._ID_CAP`.
pub const ID_CAP: u64 = 1 << 63;                                         // NEW

#[derive(Clone)]
pub struct Identifier { id: u64, name_hint: Arc<str> }                   // unchanged fields

impl Identifier {
    /// Draw a fresh id (>= RESERVED_ID_COUNT) from the global counter.
    /// # Panics
    /// If the counter is exhausted (reached u64::MAX). Payload ids are capped
    /// at ID_CAP, so this needs 2^63 fresh allocations; no input can cause it.
    #[must_use] pub fn new(name_hint: &str) -> Self;                      // CHANGED (doc/panic contract)

    /// Fallible form of `new`.
    /// # Errors
    /// `IdSpaceExhausted`, counter unchanged, when the counter is at u64::MAX.
    pub fn try_new(name_hint: &str) -> Result<Self, IdSpaceExhausted>;   // NEW

    /// Restore an identifier with a payload id, advancing the counter past it.
    /// # Errors
    /// `IdOutOfRange`, counter unchanged, if `id >= ID_CAP`.
    pub(crate) fn try_restore(id: u64, name_hint: &str)
        -> Result<Self, IdOutOfRange>;                                    // CHANGED: was pub fn restore(u64, String) -> Self (panicking)

    /// The identifier for one entry of the reserved table. Touches no counter.
    pub(crate) fn reserved(entry: ReservedIdentifier) -> Self;           // NEW (replaces new_unscoped)

    #[must_use] pub fn id(&self) -> u64;                                  // unchanged
    #[must_use] pub fn name_hint(&self) -> &str;                          // unchanged
}
// PartialEq/Eq/Hash by id; Display = name hint; Debug = "{name_hint}::{id}"  (unchanged)

/// Binding-only: draw the next fresh id. Used by fhy-core-py, which stores ids
/// in Python objects. Draws from the same counter as `Identifier::new`.
/// # Errors  `IdSpaceExhausted`, counter unchanged.
pub fn try_allocate_id() -> Result<u64, IdSpaceExhausted>;               // CHANGED (doc: binding-only; no scope)

/// Binding-only: advance the counter so `id` is never issued; no-op when the
/// counter is already past it (always the case for a reserved id).
/// # Errors  `IdOutOfRange`, counter unchanged, if `id >= ID_CAP`.
pub fn try_advance_counter_past(id: u64) -> Result<(), IdOutOfRange>;    // CHANGED: error type; no shipped-statics init

/// The counter is at u64::MAX and cannot issue another id.
/// Display (matches the Python implementation): "identifier id space exhausted"
#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[non_exhaustive]
pub struct IdSpaceExhausted;                                             // unchanged

/// A payload id at or above `ID_CAP`.
/// Display (matches the Python implementation):
///   "identifier id {id} is at or above the cap 9223372036854775808"
#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[non_exhaustive]
pub struct IdOutOfRange { id: u64 }                                      // NEW
impl IdOutOfRange { #[must_use] pub fn id(&self) -> u64; }               // NEW
impl std::error::Error for IdOutOfRange {}                                // source() = None

pub trait HasIdentifier { fn identifier(&self) -> &Identifier; }         // unchanged
```

Serde (S-2):

- **Serialize** is unchanged: the struct `"Identifier"` with fields `id: u64`
  and `name_hint: string`.
- **Deserialize** accepts the same shape, with `deny_unknown_fields`, through
  a private wire struct `IdentifierWire { id: PayloadId, name_hint: String }`
  followed by `try_restore`.
- **`PayloadId`** is private. It calls `deserialize_u64` with a visitor that
  accepts `visit_u64` and non-negative `visit_i64` below `ID_CAP`. Otherwise
  it returns `invalid_value(Unsigned|Signed)` or `invalid_type`, with
  expecting text `"an id from 0 to 9223372036854775807"`.
- **Formats.** It works with non-self-describing formats because it gives the
  u64 hint. It does not use `serde_json::Number`.

REMOVED:
- `pub fn Identifier::restore(u64, String) -> Identifier`. It becomes
  `pub(crate) try_restore`.
- `pub(crate) fn Identifier::new_unscoped`.
- `pub(crate) fn allocate_id`.
- `pub(crate) fn advance_counter_past`, the panicking variant.
- `pub(crate) struct IdentifierPayload`, together with its `Decode` impl and
  `IdentifierPayload::restore`.
- `struct IdRange`, which is replaced by the visitor's `expecting`.
- The `cfg(testing)` fork of `next_id`.

#### 2.2 `fhy_core::identifier::reserved` (crate-private, NEW file `src/identifier/reserved.rs`)

This is the single table of shipped ids. It is data only, so the leaf module
gains no crate imports.

```rust
#[derive(Debug, Clone, Copy)]
pub(crate) struct ReservedIdentifier { id: u64, name_hint: &'static str }
impl ReservedIdentifier {
    const fn new(id: u64, name_hint: &'static str) -> Self;   // private: only this table mints entries
    pub(crate) const fn id(self) -> u64;
}
```

The table is append-only. An assigned id never changes. A retired entry
leaves a hole. A new entry takes the next free id in its family block.

| Block | Const | id | name hint |
|---|---|---|---|
| note kinds `0..16` | `RATIONALE_NOTE_KIND` | 0 | rationale |
| | `SUGGESTION_NOTE_KIND` | 1 | suggestion |
| | `REMARK_NOTE_KIND` | 2 | remark |
| | `OTHER_NOTE_KIND` | 3 | other |
| op attributes `16..32` | `COMMUTATIVE` | 16 | commutative |
| | `ASSOCIATIVE` | 17 | associative |
| | `PURE` | 18 | pure |
| | `ELEMENTWISE` | 19 | elementwise |
| value domains `32..48` | `DATA_DOMAIN` | 32 | data |
| | `ADDRESS_DOMAIN` | 33 | address |
| composed-function parameters `64..128` (catalogue order) | `MAX_PARAMETERS: [_; 2]` | 64, 65 | a, b |
| | `MIN_PARAMETERS` | 66, 67 | a, b |
| | `ABS_PARAMETERS: [_; 1]` | 68 | x |
| | `SIGN_PARAMETERS` | 69 | x |
| | `CLAMP_PARAMETERS: [_; 3]` | 70, 71, 72 | x, lo, hi |
| | `CLAMP_SYMMETRIC_PARAMETERS` | 73, 74 | x, bound |
| | `RELU_PARAMETERS` | 75 | x |
| | `LEAKY_RELU_PARAMETERS` | 76, 77 | x, slope |
| | `XOR_PARAMETERS` | 78, 79 | a, b |
| | `NAND_PARAMETERS` | 80, 81 | a, b |
| | `NOR_PARAMETERS` | 82, 83 | a, b |
| | `IMPLIES_PARAMETERS` | 84, 85 | a, b |
| | `IFF_PARAMETERS` | 86, 87 | a, b |
| | `SIGMOID_PARAMETERS` | 88 | x |
| | `SILU_PARAMETERS` | 89 | x |
| | `GELU_PARAMETERS` | 90 | x |

- **Free space.** Ids `48..64` and `91..1024` are unassigned. That leaves 37
  used ids out of 1024.
- **Compile-time check.** `const TABLE: &[&[ReservedIdentifier]]` lists every
  entry, and `const _: () = assert_table_is_valid(TABLE);` is a `const fn`
  with nested `while` loops. It fails the build unless every id is below
  `RESERVED_ID_COUNT` and the ids are pairwise distinct.
- **Consumers.** These are named so the other batches can coordinate:
  - `diagnostic.rs`, for `NoteKind::{rationale, suggestion, remark,
    other}`;
  - `op_attribute.rs`;
  - `value_domain.rs`;
  - `symbolic/expression/builtins.rs`, whose owning batch uses it for
    `create_composed_function`. That function takes
    `parameters: [ReservedIdentifier; N]` in place of
    `parameter_names: [&str; N]`, and maps each entry through
    `Identifier::reserved`. `COMPOSED_FUNCTIONS` may stay a `LazyLock`,
    because its ids no longer depend on when it is first used.
    `initialize_composed_functions` is deleted.

#### 2.3 `fhy_core::interned`

```rust
pub trait Interned: Sized + Send + Sync + 'static {                      // unchanged
    type Key: Eq + Hash + Clone + fmt::Debug + Send + Sync + 'static;
    fn intern_key(&self) -> &Self::Key;
    fn intern_registry() -> &'static InternRegistry<Self>;
}

pub struct InternRegistry<T: Interned> { .. }                            // unchanged fields
impl<T: Interned> InternRegistry<T> {
    pub const fn new() -> Self;                                          // unchanged
    pub const fn with_defaults(create_defaults: fn() -> Vec<T>) -> Self; // unchanged
    /// Read-lock fast path: a key already registered returns
    /// `AlreadyCanonical` under the read lock only. Otherwise take the write
    /// lock and use the `entry` API, re-checking the key (F-039).
    pub fn intern(&self, value: T) -> InternOutcome<T>;                  // CHANGED (locking only)
    #[must_use] pub fn get<Q>(&self, key: &Q) -> Option<Canonical<T>> where ..;         // unchanged
    pub fn require<Q>(&self, key: &Q) -> Result<Canonical<T>, NotInternedError<T::Key>> where ..; // unchanged
    /// Unregister everything except the defaults. Takes `&mut self`, so a
    /// process-wide registry (`&'static`) can never be cleared; only a
    /// registry the caller owns can be (F-004, decision 2).
    pub fn clear(&mut self);                                             // CHANGED: was `&self`
}

#[must_use = "the outcome holds the canonical handle"] #[derive(Debug)]
pub enum InternOutcome<T> { Registered(Canonical<T>),
                            AlreadyCanonical { canonical: Canonical<T>, discarded: T } }   // unchanged
impl<T> InternOutcome<T> { canonical, into_canonical, is_registered }    // unchanged

pub struct Canonical<T>(Arc<T>);
impl<T> Deref, Clone, Debug, Display                                     // unchanged
/// Handles compare and hash by key (F-020). Same registry and key ⇒ same
/// instance, so this equals identity for process-wide registries.
impl<T: Interned> PartialEq for Canonical<T>  // Arc::ptr_eq fast path, then intern_key() ==   CHANGED (was ptr_eq, no bound)
impl<T: Interned> Eq for Canonical<T>                                    // CHANGED (bound)
impl<T: Interned> Hash for Canonical<T>       // hashes intern_key()     CHANGED (was pointer hash)
impl<T> Canonical<T> {
    /// Whether two handles point at the same registered instance.
    #[must_use] pub fn ptr_eq(this: &Self, other: &Self) -> bool;        // NEW
}
impl<T: Serialize> Serialize for Canonical<T>                            // unchanged (serializes as T)
/// Interns the decoded value; errors when it is unequal (T::Eq) to the
/// canonical instance already registered under its key.
impl<'de, T: Interned + Eq + Deserialize<'de>> Deserialize<'de> for Canonical<T>  // unchanged contract

#[derive(Debug, Clone, PartialEq, Eq)] #[non_exhaustive]
pub struct NotInternedError<K> { type_name: &'static str, key: K }       // CHANGED: + #[non_exhaustive]
```

Crate-private changes:

- `intern_decoded` becomes private to `interned`. Its only user is the
  generic `Canonical<T>` impl.
- `require_default` stays `pub(crate)`.
- `Interned`'s docs now say:
  - registries are append-only for the life of the process;
  - the `Interned` trait makes per-session registries impossible;
  - decoding untrusted payloads grows them without bound, which is known
    and deferred (decision 2).

#### 2.4 `fhy_core::described_tag` (NEW public module; was the private macro module)

```rust
/// Vocabulary marker of a described tag. Sealed: only this crate's vocabularies exist.
pub trait TagKind: sealed::Sealed + Send + Sync + 'static {}              // NEW
mod sealed {                        // #[expect(unnameable_types, reason = "..")] as in expression/build.rs
    pub trait Sealed: Sized {
        const TYPE_NAME: &'static str;                // "OpAttribute" / "NoteKind" for Debug and errors
        fn registry() -> &'static InternRegistry<DescribedTag<Self>> where Self: TagKind;
    }
}

/// An open vocabulary entry: an `Identifier` name (its identity) and a
/// description that takes no part in equality, hashing or interning.
pub struct DescribedTag<K: TagKind> {                                    // NEW (replaces define_described_tag!)
    name: Identifier,
    description: String,
    kind: PhantomData<fn() -> K>,   // #[serde(skip)]
}
impl<K: TagKind> DescribedTag<K> {
    /// Register the tag named `name` unless one is registered, and return the
    /// canonical handle. First registration wins; a later description is dropped.
    pub fn register(name: Identifier, description: impl Into<String>) -> Canonical<Self>;
    #[must_use] pub fn name(&self) -> &Identifier;
    #[must_use] pub fn description(&self) -> &str;
    fn create(name: Identifier, description: impl Into<String>) -> Self;  // private
}
impl<K: TagKind> Interned for DescribedTag<K> { type Key = Identifier; .. registry = K::registry() }
impl<K: TagKind> HasIdentifier, PartialEq/Eq/Hash (by name), Serialize ({"name","description"})
impl<K: TagKind> fmt::Debug   // debug_struct(K::TYPE_NAME).field("name").field("description")
impl<K: TagKind> fmt::Display // the name hint (was NoteKind-only; now also OpAttribute)
impl<'de, K: TagKind> Deserialize<'de> for Canonical<DescribedTag<K>>   // NEW concrete impl, see §5.6
// No `Deserialize for DescribedTag<K>`, and no `Clone`.
```

This relies on coherence's negative reasoning. `DescribedTag<K>` is local and
has no `Deserialize` impl, so the concrete `Canonical<DescribedTag<K>>` impl
does not overlap the blanket `Canonical<T>` impl. A probe crate verified that
this compiles. If someone later adds `Deserialize` to the bare type, the
build fails with an overlap error, which is the guard F-020 asks for.

#### 2.5 `fhy_core::op_attribute`

```rust
#[derive(Debug)] pub enum OpAttributeVocabulary {}      // NEW marker; TagKind + Sealed (TYPE_NAME "OpAttribute")
pub type OpAttribute = DescribedTag<OpAttributeVocabulary>;              // CHANGED: was a macro-generated struct
impl DescribedTag<OpAttributeVocabulary> {
    #[must_use] pub fn commutative() -> &'static Canonical<OpAttribute>; // id 16   CHANGED: was free fn get_commutative
    #[must_use] pub fn associative() -> &'static Canonical<OpAttribute>; // id 17
    #[must_use] pub fn pure()        -> &'static Canonical<OpAttribute>; // id 18
    #[must_use] pub fn elementwise() -> &'static Canonical<OpAttribute>; // id 19
}
```

- **Defaults.** The registry is
  `InternRegistry::with_defaults(create_default_attributes)`. That function
  builds the four defaults from `Identifier::reserved(reserved::COMMUTATIVE)`
  and the rest, in table order.
- **Statics.** Each accessor is a
  `LazyLock<Canonical<_>>` over
  `require_default(&Identifier::reserved(..))`. The id is fixed, so the order
  of first use does not matter and nothing can fail.
- **Removed.** `initialize_shipped_attributes` and the `*_NAME` statics are
  deleted.

#### 2.6 `fhy_core::diagnostic` (NoteKind part only; the rest belongs to its batch)

```rust
#[derive(Debug)] pub enum NoteKindVocabulary {}         // NEW marker (TYPE_NAME "NoteKind")
pub type NoteKind = DescribedTag<NoteKindVocabulary>;                    // CHANGED
impl DescribedTag<NoteKindVocabulary> {
    #[must_use] pub fn rationale()  -> &'static Canonical<NoteKind>;     // id 0   was get_rationale_note_kind
    #[must_use] pub fn suggestion() -> &'static Canonical<NoteKind>;     // id 1   was get_suggestion_note_kind
    #[must_use] pub fn remark()     -> &'static Canonical<NoteKind>;     // id 2   was get_remark_note_kind
    #[must_use] pub fn other()      -> &'static Canonical<NoteKind>;     // id 3   was get_other_note_kind
}
```

- **Display.** The hand-written `impl Display for NoteKind` is deleted,
  because `DescribedTag` provides it.
- **Note.** `Note`'s derived `Deserialize` reads `kind: Canonical<NoteKind>`
  through the impl in §5.6. The diagnostic batch owns everything else in
  `Note`.

#### 2.7 `fhy_core::value_domain`

```rust
#[derive(Debug, Serialize)]
pub struct ValueDomain { name: Identifier, description: String,
                         parent: Option<Canonical<ValueDomain>> }        // unchanged fields
impl ValueDomain {
    /// Register a root domain, or return the registered domain of that name.
    /// # Errors  `ValueDomainConflict` if `name` is registered with a parent.
    pub fn register_root(name: Identifier, description: impl Into<String>)
        -> Result<Canonical<ValueDomain>, ValueDomainConflict>;          // NEW (with register_child, replaces `new`)
    /// Register a child of `parent`, or return the registered domain of that name.
    /// # Errors  `ValueDomainConflict` if `name` is registered as a root or
    /// under a different parent.
    pub fn register_child(name: Identifier, description: impl Into<String>,
                          parent: &Canonical<ValueDomain>)
        -> Result<Canonical<ValueDomain>, ValueDomainConflict>;          // NEW
    #[must_use] pub fn name(&self) -> &Identifier;                       // unchanged
    #[must_use] pub fn description(&self) -> &str;                       // unchanged
    #[must_use] pub fn parent(&self) -> Option<&Canonical<ValueDomain>>; // unchanged
    /// O(depth): walks parents comparing names.
    #[must_use] pub fn is_subdomain_of(&self, other: &ValueDomain) -> bool;  // CHANGED complexity only
    #[must_use] pub fn data() -> &'static Canonical<ValueDomain>;        // id 32  was free fn get_data_domain
    #[must_use] pub fn address() -> &'static Canonical<ValueDomain>;     // id 33  was free fn get_address_domain
}
impl PartialEq/Eq/Hash for ValueDomain   // by name only                  CHANGED (was name + whole parent chain)
impl HasIdentifier, Interned (Key = Identifier)                          // unchanged
impl<'de> Deserialize<'de> for Canonical<ValueDomain>                    // NEW concrete impl, see §5.6
// No `Deserialize for ValueDomain`.                                     REMOVED

/// A domain name is already registered under a different parent.
#[derive(Debug, Clone, PartialEq, Eq)] #[non_exhaustive]
pub struct ValueDomainConflict {                                         // NEW
    name: Identifier, registered_parent: Option<Identifier>, requested_parent: Option<Identifier> }
impl ValueDomainConflict {
    #[must_use] pub fn name(&self) -> &Identifier;
    #[must_use] pub fn registered_parent(&self) -> Option<&Identifier>;
    #[must_use] pub fn requested_parent(&self) -> Option<&Identifier>;
}
// Display, one lowercase line, e.g.
//   "value domain `tile` is already registered with parent `data`, not `address`"
//   "value domain `tile` is already registered with no parent, not `data`"
//   "value domain `tile` is already registered with parent `data`, not as a root"
impl std::error::Error for ValueDomainConflict {}   // source() = None
```

REMOVED from `value_domain`:
- `ValueDomain::new(..) -> InternOutcome<Self>`;
- `get_data_domain` and `get_address_domain`;
- `impl Deserialize for ValueDomain`;
- `impl Decode for ValueDomain` and `ValueDomainPayload`;
- `initialize_shipped_domains`;
- `DATA_DOMAIN_NAME` and `ADDRESS_DOMAIN_NAME`.

#### 2.8 `fhy-core-py` (binding)

- **`allocate_identifier_id() -> PyResult<u64>`** is unchanged. It raises
  `RuntimeError("identifier id space exhausted")`.
- **`advance_identifier_counter_past(identifier_id: u64, /) -> PyResult<()>`**
  now maps `IdOutOfRange` to
  `PyOverflowError::new_err(error.to_string())` through a new private
  `convert_id_out_of_range`. PyO3's own conversion still raises
  `OverflowError` for negative values and values at or above `2^64`. The
  module doc and the function docs are updated.
- **`_rs.pyi`** is unchanged.

#### 2.9 Python `src/fhy_core/identifier.py`

```python
_RESERVED_ID_COUNT: Final = 1024   # Matches the Rust implementation: fhy_core::identifier::RESERVED_ID_COUNT
_ID_CAP: Final = 2**63             # Matches the Rust implementation: fhy_core::identifier::ID_CAP
_ID_OUT_OF_RANGE_MESSAGE = "identifier id {} is at or above the cap 9223372036854775808"

class _PythonIdCounter:
    def __init__(self, next_id: int = _RESERVED_ID_COUNT) -> None   # CHANGED: starts at 1024; the parameter is a test seam
    def allocate(self) -> int                                       # unchanged (RuntimeError at 2**64 - 1)
    def advance_past(self, identifier_id: int, /) -> None           # CHANGED:
        # < 0      -> OverflowError(_NEGATIVE_ID_MESSAGE)       (unchanged)
        # >= 2**64 -> OverflowError(_OVERSIZED_ID_MESSAGE)      (unchanged)
        # >= 2**63 -> OverflowError(_ID_OUT_OF_RANGE_MESSAGE.format(id))   NEW
        # the `== 2**64 - 1 -> RuntimeError` branch is deleted (subsumed)

def _create_reserved_identifier(identifier_id: int, name_hint: str) -> Identifier:   # NEW, private
    """Return the shipped identifier with fixed id `identifier_id`, drawing no id.
    Raises ValueError unless 0 <= identifier_id < _RESERVED_ID_COUNT.
    Matches the Rust implementation: Identifier::reserved."""

Identifier.deserialize_from_dict:                                   # CHANGED
    id >= _ID_CAP -> DeserializationValueError(cls, "id", "a non-negative integer below 2**63", id)
    (was: >= 2**64 - 1, "a non-negative integer below 2**64 - 1")
```

The class docstring is rewritten to say:

- ids `0..1024` are reserved for shipped identifiers;
- the counter starts at 1024;
- deserialization accepts `0 <= id < 2**63`, so no payload can exhaust the
  counter;
- construction still raises `RuntimeError` after `2**64 - 2`, which only a
  counter started near the end can reach.

The shipped Python constants move to reserved ids. This extends S-3 to four
more files and needs sign-off (D-5).

- `op_attribute.py`: `COMMUTATIVE`, `ASSOCIATIVE`, `PURE` and `ELEMENTWISE`
  use `_create_reserved_identifier(16..19, ...)`.
- `diagnostic.py`: `RATIONALE_NOTE_KIND`, `SUGGESTION_NOTE_KIND`,
  `REMARK_NOTE_KIND` and `OTHER_NOTE_KIND` use ids 0 to 3.
- `value_domain.py`: `DATA_DOMAIN` and `ADDRESS_DOMAIN` use ids 32 and 33.
- `symbolic/expression/builtins.py`: each `_register_<fn>` takes its
  parameter ids from the table (64 to 90).

### 3. Interface delta

All changes are breaking unless marked otherwise. The crate is unpublished
(S-8).

| Change | Item | Before (HEAD) | After | Semver | Call sites |
|---|---|---|---|---|---|
| NEW | `identifier::RESERVED_ID_COUNT`, `ID_CAP` | — | `pub const u64` | non-breaking | fhy-core-py docs only |
| NEW | `Identifier::try_new` | — | `fn(&str) -> Result<Self, IdSpaceExhausted>` | non-breaking | none yet |
| CHANGED | `Identifier::restore` | `pub fn restore(u64, String) -> Self` (panics on u64::MAX) | `pub(crate) fn try_restore(u64, &str) -> Result<Self, IdOutOfRange>` | breaking | identifier.rs tests (~12); value_domain.rs tests (`find_registered`, 3); op_attribute.rs (2); shipped.rs (deleted); tests/tag_type_equivalence.rs:105 and tests/deterministic_identifiers_equivalence.rs:81 (both deleted) |
| REMOVED | `Identifier::new_unscoped` | `pub(crate)` | `Identifier::reserved(ReservedIdentifier)` | none (crate) | described_tag.rs:155, value_domain.rs:238/242, builtins.rs:171, testing.rs:506 |
| CHANGED | `try_advance_counter_past` | `-> Result<(), IdSpaceExhausted>`, forces shipped statics | `-> Result<(), IdOutOfRange>`, rejects `>= ID_CAP` | breaking | fhy-core-py/src/identifier.rs:45; shipped.rs (deleted); identifier.rs tests |
| NEW | `IdOutOfRange` | — | struct plus `id()` | non-breaking | fhy-core-py |
| REMOVED | `allocate_id`, `advance_counter_past` (crate) | `pub(crate)` panicking | — | none | identifier.rs, testing.rs:40 |
| REMOVED | `IdentifierPayload` (crate) | `pub(crate)` | `Identifier: Deserialize` | none | described_tag.rs, value_domain.rs, symbolic/expression/wire.rs (3, owned by B2) |
| CHANGED | `InternRegistry::clear` | `pub fn clear(&self)` | `pub fn clear(&mut self)` | breaking | interned.rs tests (8); tests/interned_equivalence.rs:216; op_attribute.rs:301, value_domain.rs:493/520/535/565, diagnostic.rs:444 (tests deleted); tests/tag_type_equivalence.rs:747/931 (deleted) |
| CHANGED | `InternRegistry::intern` locking | always write lock | read-lock fast path | non-breaking | — |
| CHANGED | `Canonical<T>: PartialEq/Eq/Hash` | identity, no bound | by key, `T: Interned` | breaking (semantics and bound) | every `==` on handles. The semantics only differ across registries or across a clear, which only interned.rs tests do. |
| NEW | `Canonical::ptr_eq` | — | `fn(&Self, &Self) -> bool` | non-breaking | interned.rs tests, value_domain proptest, tag_type_stories |
| CHANGED | `NotInternedError` | struct | `#[non_exhaustive]` | non-breaking (private fields) | — |
| CHANGED | `OpAttribute` | macro struct | `type OpAttribute = DescribedTag<OpAttributeVocabulary>` | breaking (the path now names an alias) | — |
| CHANGED | `NoteKind` | macro struct | `type NoteKind = DescribedTag<NoteKindVocabulary>` | breaking | — |
| NEW | `described_tag::{DescribedTag, TagKind}`, `OpAttributeVocabulary`, `NoteKindVocabulary` | private module | public | non-breaking | — |
| REMOVED / NEW | `OpAttribute::new` becomes `OpAttribute::register` | `new(Identifier, impl Into<String>) -> InternOutcome<Self>` | `register(..) -> Canonical<Self>` | breaking | op_attribute.rs (21), tests/tag_type_stories.rs (3), tests/tag_type_equivalence.rs (deleted) |
| REMOVED / NEW | `NoteKind::new` becomes `NoteKind::register` | same | same | breaking | diagnostic.rs (2), tests/diagnostic_stories.rs (7; lines 129 and 147 match on `InternOutcome` and become `register` plus handle asserts) |
| CHANGED | tag getters become associated fns (F-029) | `get_commutative/associative/pure/elementwise()` | `OpAttribute::commutative()` and so on | breaking | op_attribute.rs (29), shipped.rs (deleted), tests/tag_type_stories.rs, tests/deterministic_identifiers_constants.rs (deleted), tests/tag_type_equivalence.rs (deleted) |
| CHANGED | note-kind getters | `get_rationale_note_kind()` and the other three | `NoteKind::rationale()` and the other three | breaking | diagnostic.rs (14), tests/diagnostic_stories.rs (38), tests/pass_infrastructure_core_stories.rs (2), tests/pass_infrastructure_validation_stories.rs (3), identifier.rs doc link:189 |
| CHANGED | domain getters | `get_data_domain()`, `get_address_domain()` | `ValueDomain::data()`, `::address()` | breaking | value_domain.rs (46 across getters and new), tests/tag_type_stories.rs |
| REMOVED / NEW | `ValueDomain::new` | `new(Identifier, impl Into<String>, Option<Canonical<ValueDomain>>) -> InternOutcome<Self>` | `register_root` / `register_child -> Result<_, ValueDomainConflict>` | breaking | value_domain.rs (41), tests/tag_type_stories.rs (4), tests/tag_type_equivalence.rs (deleted) |
| NEW | `ValueDomainConflict` | — | struct and accessors | non-breaking | — |
| CHANGED | `ValueDomain: PartialEq/Hash` | name plus the parent chain by value | name only | breaking (semantics) | value_domain.rs tests |
| REMOVED | bare `Deserialize` for `OpAttribute`, `NoteKind`, `ValueDomain` | public | only `Canonical<_>: Deserialize` | breaking | op_attribute.rs:496 test, value_domain.rs:1207 test (both deleted), tests/payload_form_stories.rs (B2) |
| NEW | `Display for OpAttribute` | — | name hint | non-breaking | — |
| REMOVED | `pub mod testing` (`DeterministicIdentifierScope`, `DeterministicIdentifierScopeHandle`) and feature `testing` | `cfg(any(test, feature = "testing"))` | — | breaking | 4 test binaries (deleted), Cargo.toml, CI, noxfile, READMEs, CONTRIBUTING |
| REMOVED | `shipped::initialize_shipped_statics` and the four `initialize_*` fns (crate) | `pub(crate)` | — | none | identifier.rs:198, shipped.rs |
| CHANGED | binding `advance_identifier_counter_past` for `2^63 <= id < 2^64` | allowed (and `2^64 - 1` raised `RuntimeError`) | `OverflowError("identifier id N is at or above the cap 9223372036854775808")` | breaking (Python-visible) | tests/test_identifier_rust_binding.py |
| CHANGED | Python `Identifier.deserialize_from_dict` cap | `< 2**64 - 1` | `< 2**63` | breaking (Python-visible) | tests/test_identifier.py |
| CHANGED | Python fresh ids | start at 0 | start at 1024 | breaking (Python-visible values) | tests/test_identifier.py:714 |

### 4. Encapsulation delta

- **Sealed vocabularies.** `TagKind` is sealed. Its methods (`TYPE_NAME`,
  `registry`) live on the private supertrait, so no one outside the crate can
  add a vocabulary or reach a registry except through
  `Interned::intern_registry`. The supertrait follows the
  `#[expect(unnameable_types, reason)]` convention of `expression/build.rs`.
- **Reserved ids.** `ReservedIdentifier::new` is private to the table, and
  `Identifier::reserved` is `pub(crate)`. Nothing outside the table can mint
  an id below `RESERVED_ID_COUNT` except by decoding a payload, which is
  intended: that is how shipped tags stay portable (§5.3).
- **Restore.** `Identifier::restore` is narrowed from `pub` to `pub(crate)`
  as `try_restore`. The only public way to pin an id is `Deserialize`.
- **Clearing.** `InternRegistry::clear` takes `&mut self`, so a process-wide
  registry cannot be cleared from anywhere. This replaces "test-only" in
  decision 2; see D-3.
- **Detached values.** Bare tag values can no longer be built outside their
  module. Every public constructor path registers the value.
  `DescribedTag::create` and `ValueDomain::create` stay private.
- **Deserialize.** `Deserialize` exists only on `Canonical<Tag>`, so a
  detached, unregistered tag value cannot be decoded (F-020).
- **Module tree.**
  - `described_tag` becomes `pub mod` and holds the invariant-carrying
    generic type.
  - `op_attribute` and `diagnostic` implement only their marker and the
    shipped accessors. They cannot touch `DescribedTag`'s fields, because
    they are sibling modules and not descendants.
  - `shipped` and `testing` are deleted.
  - `lib.rs` loses `mod shipped;` and
    `#[cfg(any(test, feature = "testing"))] pub mod testing;`.
- **Dependencies.** `identifier` no longer depends on `diagnostic`,
  `op_attribute`, `value_domain` or `builtins`. This removes the cycle in
  F-007.

### 5. Behavior

#### 5.1 Counter and ids

- **Start.** `NEXT_ID` starts at `RESERVED_ID_COUNT` (1024). Fresh ids are
  always `>= 1024`.
- **Restore.** `try_restore(id, _)` with `id < ID_CAP` runs
  `fetch_max(id + 1)` and cannot fail. With `id >= ID_CAP` it returns
  `Err(IdOutOfRange { id })` and leaves the counter unchanged.
- **Advance.** `try_advance_counter_past` follows the same rule. For a
  reserved id it is a no-op.
- **Payloads cannot exhaust the space.** A payload can raise the counter to
  at most `2^63`, which leaves `2^63 - 1` fresh ids. `Identifier::new` only
  panics after `2^63` fresh allocations. `try_new` is the non-panicking form,
  and `LazyLock` initializers never mint ids, so no initializer can be
  poisoned (F-001).
- **Largest payload id.** Decoding `{"id": 9223372036854775807, ...}`
  succeeds. The next `Identifier::new` returns id `2^63`, and the one after
  returns `2^63 + 1`.
- **Rejected payload ids.** Decoding `{"id": 9223372036854775808, ...}`, or
  any larger u64 up to `u64::MAX`, fails with
  `invalid value: integer `N`, expected an id from 0 to 9223372036854775807`.
  Negative ids fail with `invalid value: integer `-1`, ...`. Integers above
  `u64::MAX` reach serde_json as `f64` now that `arbitrary_precision` is
  gone (F-012), so they fail with `invalid type: floating point ...`. Every
  rejection leaves the counter unchanged.

#### 5.2 Reserved block

- **Entry points.** Every shipped identifier in either language is built by
  `Identifier::reserved`, or by `_create_reserved_identifier` in Python,
  from the table in §2.2. These identifiers never touch the counter.
- **Import time.** A fresh Python process draws no id while importing
  `fhy_core`. Its first `Identifier("x")` gets id 1024 on both backends.
- **First use.** The order in which Rust statics are first used no longer
  matters. `OpAttribute::commutative().name().id()` is 16 in every process,
  including one that first restored an id near `2^63` (F-007 and F-032).

#### 5.3 Portability of shipped tags

- **A payload names a shipped tag by its reserved id.** A payload with name
  `{"id": 16, "name_hint": <anything>}` and any description decodes to
  `OpAttribute::commutative()`. The canonical description is kept, and the
  payload's description and name hint are ignored.
- **Silent aliasing is fixed for shipped tags.** Before, a restored id could
  collide with a different shipped name in the reader. Now shipped names
  have the same ids everywhere, so that collision cannot happen.
- **Still documented, not fixed:**
  - a payload id in the reserved block that the table does not assign (for
    example 500) decodes as an ordinary identifier;
  - a user identifier restored from another process can still alias a
    different fresh identifier here. Decision 1 keeps exact ids, and the
    general fix would be F-001(a) remapping, which was not chosen.

#### 5.4 Interning (dual-defined; Python-identical behavior)

- **Unchanged.** `intern`, `get`, `require` and the defaults behave exactly
  as before. First registration wins. Defaults are registered on first use,
  in order, and the first duplicate key wins.
- **`intern` locking.** `intern` first takes the read lock. If the key is
  present, it returns `AlreadyCanonical { canonical, discarded: value }`
  without taking the write lock. Otherwise it drops the read lock, takes the
  write lock and calls `entries.entry(key)`. If another writer registered the
  key in between, it returns `AlreadyCanonical`. Exactly one `Registered`
  outcome per key still holds under concurrency.
- **`clear(&mut self)`** uses `RwLock::get_mut` and needs no lock. It
  restores the defaults with their identity, so `Canonical::ptr_eq` holds
  between a default's handle before and after the clear.
- **Handle equality is key equality.** Two handles from the same registry
  with no clear in between are equal iff they are `ptr_eq`.
- **Across registries or across a clear,** handles with equal keys are
  `==` but not `ptr_eq`. Use `Canonical::ptr_eq` to observe instance
  identity.

#### 5.5 Tag registration (F-021)

- **`DescribedTag::<K>::register(name, d)`.**
  - If `name` is unregistered, it registers and returns the new handle.
  - If `name` is registered, it returns the registered handle; the
    description `d` is dropped silently. The TODO about Python's warning log
    stays, and moves to `register`.
  - The same `Identifier` registered as an `OpAttribute` and as a
    `NoteKind` lands in two independent registries.
- **`ValueDomain::register_root(name, d)`.**
  - If `name` is unregistered, it registers a root.
  - If `name` is registered as a root, it returns that root and drops `d`.
  - If `name` is registered with a parent, it returns
    `Err(ValueDomainConflict { name, registered_parent: Some(p),
    requested_parent: None })`.
- **`ValueDomain::register_child(name, d, parent)`** is the same, with
  `requested_parent = Some(parent.name())`. A conflict means the registered
  parent's name differs, including when the registered domain is a root.
- **Equality.**
  - `ValueDomain == ValueDomain` compares names only. Registration
    guarantees that one name has one parent, so this is lawful for every
    reachable value.
  - `is_subdomain_of` walks `self` then its parents and compares
    `name == other.name`, which is O(depth).
  - The chain cannot cycle, because a parent exists before its child is
    registered and never changes.

#### 5.6 Serde shapes (S-2)

| Type | Serialize shape | Deserialize |
|---|---|---|
| `Identifier` | `{"id": u64, "name_hint": str}` | same shape, `deny_unknown_fields`, `id < 2^63`; advances the counter |
| `DescribedTag<K>` (`OpAttribute`, `NoteKind`) | `{"name": Identifier, "description": str}` | only as `Canonical<DescribedTag<K>>`: private `#[derive(Deserialize)] #[serde(deny_unknown_fields)] struct DescribedTagWire { name: Identifier, description: String }`, then `DescribedTag::<K>::register` |
| `ValueDomain` | `{"name": Identifier, "description": str, "parent": ValueDomain \| null}` | only as `Canonical<ValueDomain>`: private `ValueDomainWire { name: Identifier, description: String, #[serde(deserialize_with = "Option::deserialize")] parent: Option<Canonical<ValueDomain>> }`, with the `parent` key required. Then `register_root` / `register_child`, where a conflict becomes `D::Error::custom(conflict)`. |
| `Canonical<T>` | as `T` | generic impl for user `Interned` types; the concrete impls above for this crate's tags |

- **Derived forms.** serde's derived struct forms, including the sequence
  form, are accepted. There is no `MapOnly`.
- **No context prefix.** The "in `kind`: " prefixes from the decode framework
  go away with `decode.rs` (B2).
- **Non-self-describing formats.** Every shape round-trips through the
  workspace's non-self-describing dev-dependency (postcard or bincode; the
  choice belongs to the S-2 owner).
- **Documented partial side effects** (this replaces the "ordered partial
  side effects" contract):
  - a decode that fails can leave the counter advanced past any id it has
    already read;
  - it can leave already-decoded parents registered;
  - a `Canonical<ValueDomain>` decode registers parents before children, so
    a conflict at level *k* leaves levels above *k* unregistered and levels
    below *k* registered;
  - nothing else is guaranteed.

#### 5.7 Python behavior (dual-defined `identifier`)

- **Same counter semantics on both backends.** Given the same allocate and
  advance sequence, the Rust backend and the Python fallback return the same
  ids and raise the same exception classes and messages.
- **Allocation errors.** Allocation raises
  `RuntimeError("identifier id space exhausted")` when the counter is at
  `2**64 - 1`.
- **Advance errors.** `advance_past(id)` raises:
  - `OverflowError("can't convert negative int to unsigned")` for
    `id < 0`;
  - `OverflowError("int too big to convert")` for `id >= 2**64`;
  - `OverflowError("identifier id {id} is at or above the cap
    9223372036854775808")` for `2**63 <= id < 2**64`.
- **`deserialize_from_dict`** rejects `id >= 2**63` with
  `DeserializationValueError(..., "a non-negative integer below 2**63", id)`
  before it touches the counter. A pickle of a shipped identifier loads as
  that shipped identifier.

### 6. Error and panic model

| Condition | Result |
|---|---|
| `Identifier::new` with the counter at `u64::MAX` | panic `"identifier id space exhausted"` (needs 2^63 fresh ids, so no input can cause it) |
| `Identifier::try_new` / `try_allocate_id` with the counter at `u64::MAX` | `Err(IdSpaceExhausted)`, counter unchanged |
| `try_restore` / `try_advance_counter_past` with `id >= 2^63` | `Err(IdOutOfRange { id })`, counter unchanged |
| deserializing an id `>= 2^63`, negative or non-integer | serde `invalid_value` / `invalid_type` error, counter unchanged |
| `ValueDomain::register_*` with a parent conflict | `Err(ValueDomainConflict)` |
| `Canonical<ValueDomain>` decode with a parent conflict | serde custom error with the conflict's `Display` |
| generic `Canonical<T>` decode unequal to the registered value | serde custom error `"payload for {type} under key {key:?} conflicts with the canonical instance"` (unchanged) |
| `require` on a missing key | `Err(NotInternedError)` (unchanged) |
| shipped accessor | never panics: its default is registered with a fixed id |
| binding exhaustion | `RuntimeError`; binding out of range: `OverflowError` |

Removed panics:
- `Identifier::restore(u64::MAX)`;
- `advance_counter_past(u64::MAX)`;
- `LazyLock` poisoning caused by a starved shipped static.

### 7. Non-goals

- **F-004 in full.** There is no session value, and registries stay global
  and append-only (decision 2). Unbounded registry growth from untrusted
  payloads is documented and deferred.
- **F-001(a).** Decoded ids are not remapped (decision 1).
- **F-005.** The binding-crate split is deferred.
- **Python's `testing_patches.deterministic_identifiers_by_name_hint`.** It
  is Python-only and outside the audit's scope, so it stays.
- **Deep `ValueDomain` parent chains when decoding.** Decoding recurses
  through serde (see finding N-1 in §9). serde_json caps nesting at 128;
  other formats do not.
- **Logging a warning for a dropped description.** The TODO stays until the
  crate has logging.
- **`HasIdentifier`.** It is unchanged.
- **`NotInternedError::type_name`.** It still uses `std::any::type_name`.
  F-035 belongs to another batch.

### 8. Test plan

#### 8.1 Deleted (one line of reason each)

- `src/testing.rs` and all its unit tests: the feature is removed (F-018).
- `src/shipped.rs` and its tests
  (`restoring_an_id_near_the_end_of_the_id_space_keeps_every_shipped_static`
  and its `_in_isolation` variant): the shipped ids are fixed, so there is
  nothing to force before a restore (F-007).
- `tests/deterministic_identifiers_stories.rs`,
  `tests/deterministic_identifiers_constants.rs`,
  `tests/deterministic_identifiers_equivalence.rs`: they test the removed
  scope (decision 12 and F-032).
- `tests/builtins_scope_stories.rs`: it tests the removed scope. Its
  uniqueness property moves to `identifier_stories` (§8.4).
- `tests/golden/deterministic_identifier_cases.json` and
  `generate_deterministic_identifier_cases.py`: decision 12.
- `tests/tag_type_equivalence.rs`, `tests/golden/tag_type_cases.json` and
  `generate_tag_type_cases.py`: tag types are Rust replacements, so their
  Python-parity corpus goes (decision 3). The behaviors worth keeping move
  to `tag_type_stories.rs` (§8.4).
- `identifier.rs` tests:
  - `restoring_u64_max_panics`: there is no panicking restore any more;
  - `advance_counter_past_u64_max_panics`: the panicking variant is removed;
  - `allocate_id_shares_the_counter_with_construction`: it duplicates
    `try_allocate_id_shares_the_counter_with_construction` now that
    `allocate_id` is gone;
  - `serde_accepts_the_largest_issuable_id` and its `_in_isolation`
    variant: exhaustion by payload is impossible, and the replacement is
    the F-001 regression test.
- `op_attribute.rs` tests:
  - `clearing_the_registry_keeps_the_default_attributes_canonical`: the
    global registry can no longer be cleared, and
    `interned::tests::clear_keeps_the_identity_of_defaults` covers clearing;
  - `new_reports_a_matching_duplicate_as_already_canonical`: `InternOutcome`
    no longer comes from tag constructors;
  - `a_payload_rejected_after_its_name_restores_no_name`: the
    ordered-side-effect contract is removed (S-2);
  - `decoding_the_largest_id_as_the_first_use_keeps_the_defaults`,
    `decoding_a_bare_largest_id_as_the_first_use_keeps_the_defaults`, and
    both `_in_isolation` variants: exhaustion is impossible, there is no bare
    `Deserialize`, and the shipped ids are fixed.
- `value_domain.rs` tests:
  - `clearing_the_registry_keeps_the_default_domains_canonical`,
    `a_chain_rebuilt_after_a_clear_equals_the_chain_built_before_it`,
    `a_parent_taken_before_a_clear_is_kept_and_compares_by_value` and
    `domains_whose_parents_differ_further_up_the_chain_are_unequal`: global
    clear is impossible, and parents take no part in equality;
  - `new_reports_a_matching_duplicate_as_already_canonical`: `InternOutcome`
    is gone;
  - `a_payload_rejected_for_a_trailing_unknown_field_registers_its_fresh_parent_nowhere`,
    `a_payload_rejected_for_a_trailing_unknown_field_registers_no_fresh_ancestor`,
    `a_payload_rejected_inside_its_parent_restores_only_the_outer_name`,
    `a_payload_rejected_inside_its_grandparent_restores_the_names_above_it`,
    `a_payload_whose_parent_key_comes_first_restores_the_outer_name_first`,
    `a_conflicting_payload_restores_every_name_and_registers_its_fresh_parent`
    and `a_payload_whose_parent_conflicts_registers_only_the_fresh_grandparent`:
    they pin the removed ordered-side-effect contract (S-2);
  - `decoding_the_largest_id_as_the_first_use_keeps_the_defaults`,
    `decoding_a_bare_largest_id_over_a_parent_keeps_the_defaults`, and both
    `_in_isolation` variants: exhaustion is impossible, and there is no bare
    `Deserialize`.
- `diagnostic.rs` tests (coordinate with the diagnostic batch):
  - `clearing_the_registry_keeps_the_shipped_kinds_canonical`: global clear
    is impossible;
  - `decoding_the_largest_id_as_the_first_use_keeps_the_shipped_kinds` and
    its `_in_isolation` variant: exhaustion is impossible.
- `src/test_support.rs`:
  - `ID_HEADROOM`, `reserve_pinned_id`, `FAR_AHEAD_HEADROOM`,
    `FAR_AHEAD_SPACING`, `ID_COUNTER_GUARD`, `hold_id_counter`,
    `reserve_far_ahead_ids`, `has_counter_passed` and `RegistryGuard`: no
    test pins ids ahead of the counter or clears a global registry any more
    (F-031 and F-033);
  - `take_discarded`: tag constructors no longer return `InternOutcome`;
  - `ISOLATED_TEST_VARIABLE`, `is_isolated_run` and
    `assert_isolated_test_passes`: after the deletions above, no test uses
    them. This also resolves F-038 by deletion; confirm that no other batch
    adds an isolated test.
  - `assert_send_sync` and `compute_hash` stay.
- Every `static REGISTRY_GUARD` and every `REGISTRY_GUARD.hold()` call in
  `op_attribute.rs` (24), `value_domain.rs` (49) and `diagnostic.rs` (8):
  the guard is gone.

#### 8.2 Modified

These are all rewritten to the new API. Any change that weakens a test is
flagged **[weakens]** with its reason.

- **`identifier.rs`:**
  - `equality_and_hash_ignore_name_hint`,
    `display_returns_name_hint`, `debug_returns_name_hint_and_id`,
    `restore_advances_counter_past_a_future_id` and
    `restore_is_a_no_op_when_counter_already_ahead`: use `try_restore`.
  - `concurrent_construction_never_collides_with_ids_deserialized_ahead_of_it`:
    uses `try_restore`.
  - `advance_counter_past_keeps_later_ids_beyond_the_advanced_id` and
    `advance_counter_past_is_a_no_op_for_an_issued_id`: use
    `try_advance_counter_past`.
  - `advance_past_raises_the_counter_to_one_past_the_id`: now uses
    `ID_CAP - 1`, and the counter ends at `ID_CAP`.
  - `advance_past_refuses_to_wrap_at_u64_max`: becomes
    `advance_past_rejects_an_id_at_or_above_the_cap`, with the cases
    `ID_CAP` and `u64::MAX` and the counter unchanged.
  - `try_advance_counter_past_u64_max_returns_an_error`: becomes
    `try_advance_counter_past_rejects_an_id_at_or_above_the_cap`, which
    expects `IdOutOfRange { id }`.
  - `counter_issues_the_reference_counters_relative_ids_for_any_operation_sequence`:
    `start` is drawn from `0..ID_CAP - 1024`.
  - `serde_rejects_a_malformed_payload`, case `negative_id`: the expected
    text becomes `"expected an id from 0 to 9223372036854775807"`.
  - `serde_rejects_an_id_of_u64_max_or_more_as_out_of_range`: becomes
    `serde_rejects_an_id_at_or_above_the_cap`. The cases `2^63`, `2^63 + 1`
    and `u64::MAX` expect `invalid value: integer`. The cases `2^64` and
    `2^200` expect `invalid type: floating point` **[weakens]**: without
    `arbitrary_precision` the digits of an oversized integer cannot be
    echoed (F-012).
  - `serde_rejects_a_negative_or_fractional_id`: the `Expression` path is
    dropped, because the envelope is gone and B2 covers expression decode.
    Both of these nested-path tests assert `contains(expected)` instead of
    `starts_with(prefix + expected)` **[weakens]**: the "in `kind`:"
    context prefix is removed along with `decode.rs`.
  - The `Send`/`Sync` const block adds `IdOutOfRange`.
- **`interned.rs`:**
  - `with_defaults_runs_the_default_constructor_once`,
    `clear_as_the_first_operation_keeps_defaults`,
    `clear_unregisters_non_default_instances` and
    `clear_empties_a_registry_without_defaults`: bind the registry with
    `let mut`.
  - `clear_keeps_the_identity_of_defaults`: asserts `Canonical::ptr_eq`,
    which is stronger.
  - `intern_after_clear_registers_a_new_canonical_instance`: replaces
    `assert_ne!(after, before)`, which would flip under key equality, with
    `!Canonical::ptr_eq` plus `after == before`.
  - `canonical_handles_from_separate_registries_are_unequal`: becomes
    `canonical_handles_compare_by_key_across_registries`, which asserts
    `a == b` and `!ptr_eq`.
  - `canonical_handles_hash_by_identity`: becomes
    `canonical_handles_hash_by_key`, where the set length stays 1 after the
    handle from the other registry is inserted.
  - `registry_matches_a_first_wins_model_for_any_operation_sequence`: uses
    `let mut registry`, and every handle comparison uses
    `Canonical::ptr_eq`. Plain `==` would weaken the test into a key check,
    so this must switch to keep its strength.
- **`tests/interned_equivalence.rs`:** `replay_op` and `replay_case` take
  `&mut InternRegistry<GoldenTag>`. The corpus and generator stay
  (dual-defined). `common/mod.rs`'s doc now names only `interned`.
- **`op_attribute.rs`:**
  - `new_stores_the_name_and_description`,
    `has_identifier_returns_the_name`, `intern_key_is_the_name`,
    `identifiers_sharing_a_name_hint_intern_separately` and
    `attributes_with_different_names_are_unequal`: use `register`.
  - `new_keeps_the_first_attribute_canonical_for_a_repeated_name`: becomes
    `register_keeps_the_first_attribute_for_a_repeated_name`, which asserts
    that the handle is `ptr_eq` and the description is `"first"`. The
    `discarded` assertion goes because the contract changed; this is not a
    weakening.
  - `attributes_with_the_same_name_are_equal_whatever_the_description` and
    `equal_attributes_hash_equally`: move to `described_tag::tests`, which
    can use the private `create`.
  - The set tests and the three `a_default_attribute_*` /
    `the_default_attributes_are_pairwise_distinct` tests: use the
    associated fns.
  - `an_attribute_encodes_as_its_name_and_description` and
    `decoding_an_unregistered_name_registers_the_decoded_attribute`: pin an
    already-issued id (`Identifier::new(..).id()`) instead of
    `reserve_pinned_id`, which moves no counter (F-033).
  - `decoding_a_malformed_payload_is_rejected`: uses an issued id.
  - `debug_mentions_the_name_hint_and_description` is unchanged.
- **`value_domain.rs`:**
  - `intern_root` and `intern_child`: become `register_root(..).unwrap()`
    and `register_child(..).unwrap()`.
  - `new_stores_*`, `has_identifier_returns_the_name` and
    `intern_key_is_the_name`: use the new constructors.
  - `new_keeps_the_first_domain_canonical_for_a_repeated_name`: becomes
    `register_root_of_a_known_root_returns_the_first_domain`.
  - `domains_with_the_same_name_and_parent_are_equal` and
    `equal_domains_hash_equally`: use the private `create` and now assert
    name-only equality and hashing.
  - `domains_with_different_parents_are_unequal`: inverted to
    `equality_ignores_the_parent`.
  - The five `a_domain_is_*_subdomain_*` tests and the property
    `is_subdomain_of_matches_the_ancestor_relation_for_any_hierarchy`: use
    the new API.
  - `a_domain_in_any_hierarchy_round_trips_through_json`: adds
    `Canonical::ptr_eq`.
  - The two encode tests: use issued ids.
  - `decoding_a_conflicting_parent_is_rejected` and
    `decoding_a_dropped_parent_is_rejected`: assert that the message
    contains `"already registered with parent"`.
  - `a_nested_payload_missing_its_parent_is_rejected`: the counter
    assertion goes because its contract is removed (S-2).
  - `a_payload_whose_parent_is_not_a_map_restores_nothing`: becomes
    `a_parent_that_is_not_a_domain_is_rejected`, with the counter assertion
    dropped for the same reason.
  - `a_payload_whose_parent_holds_an_out_of_range_id_restores_only_the_outer_name`:
    becomes `a_parent_with_an_out_of_range_id_is_rejected`, with the
    counter assertion dropped for the same reason.
- **`diagnostic.rs`** (coordinate):
  - `a_valid_note_restores_and_registers_its_kind`: uses an issued id and
    no lock.
  - The five `a_note_*_restores_nothing` and
    `a_bare_note_kind_rejected_for_an_unknown_field_restores_nothing` tests
    keep only their rejection assertions. The counter assertions go because
    the ordered-side-effect contract is removed (S-2). The bare `NoteKind`
    case decodes `Canonical<NoteKind>` instead.
- **Integration tests:**
  - `tests/tag_type_stories.rs`: uses the new API, and
    `persisting_and_restoring_a_tagged_operation` also asserts `ptr_eq`.
    Its doc no longer says "`==` on `Canonical` is identity".
  - `tests/diagnostic_stories.rs`,
    `tests/pass_infrastructure_core_stories.rs` and
    `tests/pass_infrastructure_validation_stories.rs`: getters renamed, and
    lines 129 and 147 of `diagnostic_stories.rs` switch from `InternOutcome`
    to `register`.
  - `tests/payload_form_stories.rs`: the Identifier, NoteKind, OpAttribute
    and ValueDomain cases are removed because `MapOnly` is removed. B2 owns
    the file.
- **Python:**
  - `tests/test_identifier.py`:
    - `test_deserialize_id_of_2_pow_64_minus_1_or_more_raises_value_error`
      becomes
      `test_deserialize_id_at_or_above_2_pow_63_raises_value_error`, with
      the params `[2**63, 2**63 + 1, 2**64 - 1, 2**64, 2**200]` and the
      match `below 2\*\*63`.
    - `test_rejected_deserialization_of_2_pow_64_minus_1_leaves_the_counter`
      now uses `2**63`.
    - `test_constructing_past_the_largest_issuable_id_raises_runtime_error`
      is replaced by the F-001 regression below. On the Rust backend,
      exhaustion is no longer reachable from Python **[weakens]**: the
      Rust side is still covered by the `take_next_id` unit tests.
    - `test_fresh_process_issues_ids_upward_from_zero` becomes
      `test_fresh_process_issues_ids_upward_from_the_reserved_block`. It
      asserts that every import-time id is `< 1024`, that `first == 1024`
      and that `second == 1025`.
  - `tests/test_identifier_rust_binding.py`:
    - `test_counter_advancing_past_the_largest_64_bit_id_raises_runtime_error`
      becomes
      `test_counter_advancing_past_an_id_at_or_above_the_cap_raises_overflow_error`,
      with the params `[2**63, 2**64 - 1]` and the exact message. Both
      counters must agree.
    - `test_python_counter_issues_the_largest_id_then_fails` uses
      `_PythonIdCounter(next_id=2**64 - 2)` in place of
      `advance_past(2**64 - 3)`.
    - The docstring of the out-of-64-bits test is updated.
  - `tests/test_identifier_rust_binding_properties.py` and
    `symbolic/test_serialization_pins.py` are unchanged. Id 0 in those blobs
    is now a reserved id, and decoding it is still valid.
- **Tooling:**
  - `noxfile.py`: drop the `deterministic_identifier` and `tag_type`
    entries from `EXPANDED_GOLDEN_CORPORA`.
  - Drop `"--features", "testing"` and its comment from `golden_expanded`.
  - Allow a `None` value in `EXPANDED_GOLDEN_CORPORA`, meaning a fixed
    corpus with nothing to expand, for `generate_reserved_identifiers.py`,
    and skip it in `golden_expanded`.
  - `tests/test_golden_corpora.py`'s docstring no longer mentions the
    deterministic-identifier scope.

#### 8.3 Regression test per finding

| Finding | Test |
|---|---|
| F-001 | `identifier::tests::deserializing_the_largest_payload_id_leaves_construction_working`: decode `2^63 - 1`, then `Identifier::new` twice returns `2^63` and `2^63 + 1`, with no isolation. Also `identifier::tests::serde_rejects_an_id_at_or_above_the_cap`, and Python `test_deserializing_the_largest_payload_id_leaves_construction_working[rust\|python]`, which runs as a subprocess on both backends and has the same assertions plus `2**63` rejected. |
| F-004 (narrowed) | a `compile_fail` doctest on `InternRegistry::clear` (`OpAttribute::intern_registry().clear()` does not compile), paired with a compiling doctest that clears a local `let mut` registry |
| F-007 | `tests/identifier_stories.rs::shipped_identifiers_hold_their_reserved_ids`, which checks all 37 entries through the public accessors after first restoring `2^63 - 1`, and `tests/identifier_stories.rs::reserved_identifiers_match_the_python_oracle`, which replays the golden in §8.4 |
| F-018 | `tests/identifier_stories.rs::identifiers_created_with_one_name_hint_are_distinct`. It runs on the production `next_id` path now that the self dev-dependency is gone. |
| F-020 | `value_domain::tests::equality_and_hash_depend_only_on_the_name`, `interned::tests::canonical_handles_compare_by_key_across_registries`, and a `compile_fail` doctest on `OpAttribute` (`serde_json::from_str::<OpAttribute>("…")` does not compile) with a compiling `Canonical<OpAttribute>` counterpart |
| F-021 | `value_domain::tests::registering_a_known_name_under_another_parent_is_a_conflict`, `value_domain::tests::registering_a_known_child_as_a_root_is_a_conflict`, `op_attribute::tests::register_returns_the_first_attribute_for_a_known_name` |
| F-027 | `described_tag::tests::attribute_and_note_kind_registries_are_independent` (one `Identifier` registered in both, each registry sees only its own), and a `compile_fail` doctest on `TagKind` (implementing it outside the crate) |
| F-029 | `op_attribute::tests::commutative_holds_reserved_id_16`, one per shipped accessor, parametrized over all 10 tag accessors |
| F-031, F-033 | Removal of the machinery is the fix. The guard is that every rewritten decode test pins an already-issued id (`Identifier::new(..).id()`), so no test can move the counter into another test's window. There is no dedicated test. |
| F-032 | Goes away with the corpus. `shipped_identifiers_hold_their_reserved_ids` restores first, which is the order that used to break. |
| F-039 | `interned::tests::interns_of_a_registered_key_share_the_read_lock`. The key's `Hash` impl, when armed, rendezvouses two threads through a channel with a 5 s `recv_timeout` while the lock is held. Under read locks both threads enter; under the old write lock the rendezvous times out and the test fails instead of hanging (F-037). |

#### 8.4 New tests

- **`tests/identifier_stories.rs`** is a new binary. Five binaries are
  deleted, so the net count is −4 (F-034). It holds:
  - `identifiers_created_with_one_name_hint_are_distinct` (F-018);
  - `fresh_identifiers_are_never_reserved`: 1,000 `Identifier::new` ids are
    all `>= RESERVED_ID_COUNT`;
  - `shipped_identifiers_hold_their_reserved_ids` (F-007). It first
    deserializes `2^63 - 1`, then checks each of the 10 tags and the 27
    composed-function parameters by id and name hint against a
    hand-written expected table. The parameters are reached through the
    public `find_composed_function(name).parameters()`, which the builtins
    batch may rename.
  - `composed_function_parameters_are_pairwise_distinct_and_unaliased`.
    This is the property from the deleted `builtins_scope_stories`: the
    parameter ids are distinct, and none equals a fresh `Identifier::new`
    of `a`, `b`, `x`, `lo`, `hi`, `bound` or `slope`.
  - `a_shipped_tag_decoded_by_its_reserved_id_is_the_shipped_tag` (§5.3).
    A payload `{"name": {"id": 16, "name_hint": "whatever"}, "description":
    "other"}` decodes `ptr_eq` to `OpAttribute::commutative()`, with the
    canonical description.
  - `reserved_identifiers_match_the_python_oracle`. It replays the new
    golden `tests/golden/reserved_identifiers.json`, written by the new
    `generate_reserved_identifiers.py`. The generator imports the Python
    shipped constants and builtins and records `{"reserved_id_count",
    "id_cap", "entries": [{"owner", "constant", "id", "name_hint"}]}`. The
    test asserts entry-by-entry equality with the Rust accessors, and that
    both sides agree on the two constants. `identifier` is dual-defined, so
    a Python oracle corpus fits decision 3, and `test_golden_corpora.py`
    checks it for staleness.
- **`tests/tag_type_stories.rs`** gets these stories, carried over from the
  deleted tag-type corpus:
  - `registering_a_known_attribute_keeps_the_first_description`;
  - `a_decoded_domain_chain_registers_every_level`: a fresh three-level
    payload has every level registered, and each level is the parent of the
    next;
  - `a_decoded_domain_under_another_parent_is_rejected_and_the_canonical_domain_is_unchanged`;
  - `a_rejected_payload_leaves_the_canonical_domain_unchanged`. This is
    the only side-effect claim that stays pinned under S-2.
- **Unit tests:**
  - `identifier::tests::try_restore_rejects_an_id_at_or_above_the_cap` (the
    counter is unchanged);
  - `identifier::tests::reserved_identifiers_take_no_id_from_the_counter`:
    `Identifier::reserved(reserved::COMMUTATIVE)` twice, with the counter
    delta observed by two `Identifier::new` anchors being 1;
  - `identifier::tests::id_out_of_range_displays_the_id_and_the_cap`, an
    exact string, because the Python mirror depends on it;
  - `value_domain::tests::conflict_error_names_the_domain_and_both_parents`
    (the three `Display` forms);
  - `value_domain::tests::a_decoded_payload_with_a_conflicting_parent_reports_the_conflict`;
  - `described_tag::tests::display_renders_the_name_hint` (for both
    vocabularies).
- **Property tests:**
  - `identifier::tests::serde_round_trips_through_postcard_for_any_payload_id`:
    ids in `0..ID_CAP` and any name hint, using the non-self-describing
    dev-dependency (S-2);
  - `value_domain::tests::a_domain_in_any_hierarchy_round_trips_through_postcard`;
  - `interned::tests::registry_matches_a_first_wins_model_for_any_operation_sequence`,
    which is kept with `ptr_eq` as in §8.2.
- **Adversarial cases:**
  - an id of `2^63`, `u64::MAX`, `-1`, `1.5` or `1e3`, or a string, in every
    nested position: Identifier, `Canonical<OpAttribute>`,
    `Canonical<NoteKind>` via `Note`, and a `ValueDomain` parent;
  - a duplicate `id` key;
  - a `parent` of the wrong type;
  - a reserved id with a conflicting description (accepted, §5.3);
  - a reserved but unassigned id such as 500 (accepted as an ordinary
    identifier).
- **Python:**
  - `test_deserializing_the_largest_payload_id_leaves_construction_working[rust|python]`
    (F-001);
  - `test_python_counter_starts_at_the_reserved_block`:
    `_PythonIdCounter().allocate() == 1024`;
  - `test_create_reserved_identifier_rejects_an_id_outside_the_block`;
  - `test_pickled_shipped_identifier_loads_as_the_shipped_identifier`:
    `pickle.loads(pickle.dumps(COMMUTATIVE.name)) == COMMUTATIVE.name` and
    the id is 16.

#### 8.5 Config and doc deletions (F-018)

- **`rust/fhy-core/Cargo.toml`:** delete the `[features]` table
  (`default = []` and `testing = []`), the self dev-dependency
  `fhy-core = { path = ".", features = ["testing"] }`, and the four
  `[[test]] required-features` entries.
- **`.github/workflows/python-package.yml`:**
  - "Testing the Packaged Crate" is **kept**. It also proves that the
    packed crate ships everything its tests `include_str!`, such as the
    golden corpora. Only `--features testing` and the comment about the
    self dev-dependency are dropped.
  - In the `rust-msrv` job, the second
    `check --workspace --lib --all-features` line and its comment are
    dropped, because with no features left it is identical to the first.
    This is a revertible call.
- **Docs:**
  - `CONTRIBUTING.md` lines 243–251: drop `--features testing` and "once
    with default features and once with `--all-features`".
  - Root `README.md`, line 165: delete the `testing` paragraph.
  - `rust/fhy-core/README.md`, line 13: delete the `testing` line.
  - `lib.rs` crate doc: drop the "process-global statics" and
    `arbitrary_precision` narrative that this batch or B2 makes obsolete.
    F-040 owns the final wording.

### 9. Findings covered, and judgment calls

- **Covered:**
  - F-001, F-007, F-018, F-020, F-021 (tags, value domain and
    `Identifier::restore`), F-027, F-029 (tag getters) and F-039 (intern
    lock);
  - F-004, narrowed to decision 2 as the `clear(&mut self)` change plus
    docs;
  - F-031 and F-033, for the id and registry parts;
  - F-032, which goes away with F-018.
- **Also removed:** F-038's harness, which has no users left in this batch;
  confirm no other batch uses it.

#### Decisions needing sign-off

- **D-1: reserved block size and ids.**
  - `RESERVED_ID_COUNT = 1024`, with family blocks: note kinds `0..16`, op
    attributes `16..32`, value domains `32..48`, composed parameters
    `64..128`. That uses 37 of 1024 ids.
  - The table is append-only.
  - The public names are `RESERVED_ID_COUNT` and `ID_CAP`, where S-3 said
    `RESERVED`.
- **D-2: one central table** in `identifier/reserved.rs`, checked at compile
  time. The alternative is ids declared by each owner plus a test that
  checks them. I chose the central table because it gives one place to
  allocate and a build-time uniqueness check. It names builtins parameters
  inside the leaf module, but as data only.
- **D-3: `InternRegistry::clear(&mut self)` instead of `#[cfg(test)]`.**
  Decision 2 says "clear test-only".
  - A `cfg(test)` item is invisible to integration tests, so
    `interned_equivalence` could not replay the Python oracle's `clear` ops.
    `interned` is dual-defined, so that replay has to stay.
  - `&mut self` makes clearing a `&'static` registry impossible, which is
    the actual risk, and leaves local clearing legal.
- **D-4: `Canonical<T>` equality becomes key equality,** plus
  `Canonical::ptr_eq`. This is the literal reading of F-020's "one equality,
  by key". For global registries it is observably identical to identity.
  Tests that relied on identity across a clear switch to `ptr_eq`.
- **D-5: the Python side goes beyond `identifier.py`.**
  - `op_attribute.py`, `diagnostic.py`, `value_domain.py` and
    `symbolic/expression/builtins.py` switch to
    `_create_reserved_identifier`. Decision 1 says "identically in Python".
  - The builtins parameters are included so the table means the same thing
    in both languages, and so the fresh-process test can assert that import
    draws nothing.
  - Cheaper alternative: only the 10 tags are reserved in Python, the
    builtins keep drawing from the counter, and ids 64..91 go unused in
    Python.
  - D-5 also adds a new golden corpus, `reserved_identifiers.json`, and a
    small `noxfile.py` change to allow a generator with no expanded corpus.
- **D-6: Python-visible error change.** Advancing the counter past any id in
  `[2**63, 2**64)` now raises `OverflowError` on both backends with the
  message "identifier id N is at or above the cap 9223372036854775808". That
  includes `2**64 - 1`, which used to raise `RuntimeError`. The alternative
  is `ValueError`. I chose `OverflowError` so every out-of-range id raises
  the same class.
- **D-7: public path `fhy_core::described_tag`,** which is not in S-1's list,
  and the marker names `OpAttributeVocabulary` and `NoteKindVocabulary`.
- **D-8: `ValueDomain` registration semantics.**
  - A second registration with the same parent and a different description
    succeeds and returns the first domain, as the decoder always did.
  - Only a parent mismatch is a conflict.
  - Registering a known child as a root is a conflict.
- **D-9: CI.** The packaged-crate step is kept, since it does not exist only
  for the feature. The MSRV `--all-features` line is dropped.
- **D-10: tag types lose bare `Deserialize`.** This uses concrete
  `Canonical<Tag>` impls next to the generic one. It relies on coherence's
  negative reasoning for local types, which a probe crate verified compiles.

#### Possibly misjudged, or new

- **F-018's "CI packaged-crate step"** was listed as part of the feature's
  cost. The step does more than that (see D-9), so only its
  `--features testing` flag is feature cost.
- **F-001's `try_new`** is required by S-3. With the cap, exhaustion needs
  2^63 fresh allocations, so `try_new` exists for API completeness and not
  for reachable safety. Its value is low; keep it cheap.
- **F-033's severity (Medium)** was fair while the scheme existed. It
  disappears entirely here rather than being fixed.
- **N-1 (new, Low).** Decoding `Canonical<ValueDomain>` recurses once per
  parent level through serde. serde_json caps this at 128. A
  non-self-describing format such as postcard has no limit, so a crafted
  deep chain could overflow the stack. This is the same class as F-003,
  but for domains. It is left out of this batch. The fix, if wanted, is a
  flattened `ancestors` wire shape decoded iteratively. Please triage.
- **N-2 (acknowledged, deferred).** Registries still grow without bound on
  untrusted decode. F-004 names this, and decision 2 accepts it. §7 records
  it so it is not lost.

---

## B2: Serialization infrastructure, diagnostics, provenance, Python-text removal

Scope at HEAD `412e234`: `rust/fhy-core/src/{decode.rs, decode/buffered.rs,
python_text.rs, diagnostic.rs, provenance.rs, test_support.rs (harness only),
lib.rs (crate docs)}`, the workspace `Cargo.toml`, `rust/fhy-core/Cargo.toml`,
and the CONTRIBUTING rule "Decoding checks the payload before its side
effects".

Findings: F-011, F-012, F-013, F-014 (diagnostic and provenance errors),
F-015 (reports), F-021 (`Diagnostic::new`, `Span::try_new`, `fuse`), F-035
(the four listed test files), F-038.

Layering (S-1): `diagnostic` depends on `identifier`, `interned` and B1's
described-tag type. `provenance` depends on nothing in the crate. Neither
hosts passes, so the `expr::passes` versus `pass` choice does not apply.

---

### 1. Summary

This batch deletes the ordered-side-effect decode framework (`decode.rs`,
`decode/buffered.rs`) and `python_text.rs`. It also removes serde_json's
`arbitrary_precision` from the workspace and sets one serde rule that every
batch follows: plain derives, Rust-defined shapes, and every serialized type
must round-trip through JSON and through postcard. Diagnostics and provenance
get Rust-shaped constructors (`Diagnostic::error(..).with_detail(..)`, span
builders taking `Range`, `fuse`/`fuse_labelled`), per-constructor error types,
`Display` for reports, and a one-line `ValidationFailedError`. The
isolated-child-process test harness now checks the exit status instead of
libtest's output and can no longer pass vacuously.

---

### 2. Desired public interface

Visibility is `pub` at the path shown unless stated. Every item keeps
`#[must_use]` where it has it today.

#### 2.1 `fhy_core::diagnostic`

**NoteKind** (CHANGED; B1 owns its definition). `NoteKind` is B1's generic
described-tag type, instantiated with a diagnostic-owned kind marker. This
batch sets the following requirements on it and leaves the generic to B1:

- Wire shape `{"name": <Identifier>, "description": <String>}`, from a plain
  derive.
- `Canonical<NoteKind>: Deserialize` interns the value (B1's `interned`).
- The shipped kinds are associated functions (S-7). Each has a fixed reserved
  id (S-3), and each keeps today's name hint and description:

```rust
impl NoteKind {                     // an inherent impl on the instantiation; B1 decides where it lives
    /// The kind for notes that explain why a decision, transformation, or result occurred.
    pub fn rationale() -> &'static Canonical<NoteKind>;   // NEW (was get_rationale_note_kind)
    /// The kind for notes that suggest a fix or course of action.
    pub fn suggestion() -> &'static Canonical<NoteKind>;  // NEW (was get_suggestion_note_kind)
    /// The kind for neutral informational notes.
    pub fn remark() -> &'static Canonical<NoteKind>;      // NEW (was get_remark_note_kind)
    /// The kind for uncategorized notes.
    pub fn other() -> &'static Canonical<NoteKind>;       // NEW (was get_other_note_kind)
}
```

None of these panic. The ids are fixed constants, so they have no
first-use allocation. If B1 picks `Canonical<NoteKind>` by value as the
return type for `OpAttribute::commutative()`, these four use the same return
type.

`impl Display for NoteKind` (unchanged behavior: it renders the name hint)
stays in `diagnostic.rs`, unless B1's generic already provides `Display`.

**Note** (CHANGED: derive-based serde)

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Note { message: String, kind: Canonical<NoteKind> }

impl Note {
    pub fn new(message: impl Into<String>, kind: Canonical<NoteKind>) -> Self;   // unchanged
    pub fn with_other_kind(message: impl Into<String>) -> Self;                   // unchanged; uses NoteKind::other()
    pub fn message(&self) -> &str;                                                // unchanged
    pub fn kind(&self) -> &Canonical<NoteKind>;                                   // unchanged
}
impl Display for Note   // unchanged: "kind: message"
```

Wire shape: `{"message": <String>, "kind": <NoteKind>}` (unchanged in JSON).

**DiagnosticLevel** (CHANGED: `#[non_exhaustive]`)

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DiagnosticLevel { Error, Warning, Info }
impl DiagnosticLevel { pub fn as_str(self) -> &'static str; }   // unchanged: "error" | "warning" | "info"
impl Display for DiagnosticLevel                                // unchanged
```

`DiagnosticLevel` has no serde impls today and gets none here (non-goal).
It is non-exhaustive because a `hint` or `note` level is a likely addition.
It is not an enum that passes match to decide semantics, so the S-6
exception does not apply.

**Diagnostic** (CHANGED constructors, NEW `Display`)

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Diagnostic { level, message: Note, source: String, detail: Option<String> }  // fields private, unchanged

impl Diagnostic {
    /// CHANGED: three arguments, no detail.
    pub fn new(level: DiagnosticLevel, message: Note, source: impl Into<String>) -> Self;
    /// NEW: `new(DiagnosticLevel::Error, ..)`.
    pub fn error(message: Note, source: impl Into<String>) -> Self;
    /// NEW: `new(DiagnosticLevel::Warning, ..)`.
    pub fn warning(message: Note, source: impl Into<String>) -> Self;
    /// NEW: `new(DiagnosticLevel::Info, ..)`.
    pub fn info(message: Note, source: impl Into<String>) -> Self;
    /// NEW: replaces any detail with `detail`, stored as given (the empty string included).
    pub fn with_detail(self, detail: impl Into<String>) -> Self;

    pub fn level(&self) -> DiagnosticLevel;      // unchanged
    pub fn message(&self) -> &Note;              // unchanged
    pub fn message_text(&self) -> &str;          // unchanged
    pub fn source(&self) -> &str;                // unchanged
    pub fn detail(&self) -> Option<&str>;        // unchanged
}

/// NEW. Renders `{level}[{source}]: {message text}` and, when the detail is
/// present and non-empty, a second line `    detail: {detail}` (four spaces).
/// The note kind is not rendered. Newlines inside the message or detail are
/// written as they are. There is no trailing newline.
impl Display for Diagnostic
```

None of these panic or fail.

**ValidationReport** (CHANGED: `format` becomes `Display`)

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ValidationReport<R = ()> { diagnostics: Vec<Diagnostic>, records: Vec<R> }

impl<R> ValidationReport<R> {
    pub fn new(diagnostics: Vec<Diagnostic>, records: Vec<R>) -> Self;          // unchanged
    pub fn diagnostics(&self) -> &[Diagnostic];                                 // unchanged
    pub fn records(&self) -> &[R];                                              // unchanged
    pub fn errors(&self) -> impl Iterator<Item = &Diagnostic> + '_;             // unchanged
    pub fn warnings(&self) -> impl Iterator<Item = &Diagnostic> + '_;           // unchanged
    pub fn infos(&self) -> impl Iterator<Item = &Diagnostic> + '_;              // unchanged
    pub fn has_errors(&self) -> bool;                                           // unchanged
    pub fn into_result(self) -> Result<Self, ValidationFailedError<R>>;         // unchanged
    // REMOVED: pub fn format(&self) -> String
}

/// NEW. The `Display` of each diagnostic in emission order, joined by `\n`,
/// with no trailing newline. An empty report renders as the empty string.
/// Records are not rendered.
impl<R> Display for ValidationReport<R>
```

**ValidationFailedError** (CHANGED `Display`)

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationFailedError<R = ()> { report: ValidationReport<R> }   // private field, unchanged

impl<R> ValidationFailedError<R> {
    pub fn report(&self) -> &ValidationReport<R>;     // unchanged
    pub fn into_report(self) -> ValidationReport<R>;  // unchanged
}
/// CHANGED. One lowercase line: `validation failed with 1 error` or
/// `validation failed with {n} errors`, where n is `report.errors().count()`
/// and is at least 1 by construction.
impl<R> Display for ValidationFailedError<R>
/// unchanged: `source()` is `None`. The report is data, not a cause.
impl<R: Debug> std::error::Error for ValidationFailedError<R>
```

**REMOVED:** `get_rationale_note_kind`, `get_suggestion_note_kind`,
`get_remark_note_kind`, `get_other_note_kind`, and `ValidationReport::format`.

#### 2.2 `fhy_core::provenance`

**Position** (CHANGED: error type and derived serde)

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Position { line: NonZeroU64, column: NonZeroU64 }

impl Position {
    /// CHANGED error type.
    /// # Errors
    /// `PositionError::ZeroLine` if `line == 0`, else `PositionError::ZeroColumn` if `column == 0`.
    pub fn try_new(line: u64, column: u64) -> Result<Self, PositionError>;
    pub fn line(&self) -> NonZeroU64;     // unchanged
    pub fn column(&self) -> NonZeroU64;   // unchanged
}
impl Display for Position   // unchanged: "line:column"
```

Wire shape: `{"line": <u64>, "column": <u64>}` (unchanged in JSON). Serde's
`NonZeroU64` impl rejects a zero.

**Span** (CHANGED: builder constructors replace `try_new`)

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "SpanData")]   // SpanData: private, same field names and order, deny_unknown_fields
pub struct Span { start_offset: Option<u64>, end_offset: Option<u64>,
                  start_position: Option<Position>, end_position: Option<Position> }

impl Span {
    /// CHANGED to `const fn`. The span with no bounds.
    pub const fn unknown() -> Self;
    /// NEW. Offsets `offsets.start..offsets.end`, no positions.
    /// # Errors  `SpanError::EndOffsetBeforeStart` if `offsets.end < offsets.start`.
    pub fn from_offsets(offsets: Range<u64>) -> Result<Self, SpanError>;
    /// NEW. Positions `positions.start..positions.end`, no offsets.
    /// # Errors  `SpanError::EndPositionBeforeStart` if `positions.end < positions.start`.
    pub fn from_positions(positions: Range<Position>) -> Result<Self, SpanError>;
    /// NEW. `self` with both offsets replaced. Errors as `from_offsets`.
    pub fn with_offsets(self, offsets: Range<u64>) -> Result<Self, SpanError>;
    /// NEW. `self` with both positions replaced. Errors as `from_positions`.
    pub fn with_positions(self, positions: Range<Position>) -> Result<Self, SpanError>;
    /// NEW. One bound replaced, checked against the other bound of the same
    /// pair when that bound is set.
    /// # Errors  `EndOffsetBeforeStart` / `EndPositionBeforeStart`, as above.
    pub fn with_start_offset(self, offset: u64) -> Result<Self, SpanError>;
    pub fn with_end_offset(self, offset: u64) -> Result<Self, SpanError>;
    pub fn with_start_position(self, position: Position) -> Result<Self, SpanError>;
    pub fn with_end_position(self, position: Position) -> Result<Self, SpanError>;

    pub fn is_unknown(&self) -> bool;                 // unchanged
    pub fn start_offset(&self) -> Option<u64>;        // unchanged
    pub fn end_offset(&self) -> Option<u64>;          // unchanged
    pub fn start_position(&self) -> Option<Position>; // unchanged
    pub fn end_position(&self) -> Option<Position>;   // unchanged
    // REMOVED: pub fn try_new(Option<u64>, Option<u64>, Option<Position>, Option<Position>) -> Result<Self, ProvenanceError>
}
impl Display for Span   // unchanged
```

The offsets and the positions are still never checked against each other.
Wire shape: `{"start_offset", "end_offset", "start_position",
"end_position"}`, with `null` for an absent bound. On decode, a missing
`Option` key reads as `None` (serde's default; this is a behavior change),
and the order checks run as in the builders.

**Provenance** (CHANGED: `fuse` split, derived serde; stays exhaustive)

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]              // externally tagged (serde default)
pub enum Provenance {                            // NOT #[non_exhaustive]: see below
    Unknown,
    File(FileProvenance),
    Named(NamedProvenance),
    CallSite(CallSiteProvenance),
    Fused(FusedProvenance),
}

impl Provenance {
    /// CHANGED: no label argument. Flattens (drops `Unknown`, splices in the
    /// sources of unlabelled fusions at any depth, iteratively). The result
    /// is `Unknown` if nothing survives, the survivor itself if exactly one
    /// survives, and otherwise an unlabelled `Fused` of the survivors in
    /// order.
    pub fn fuse(provenances: impl IntoIterator<Item = Provenance>) -> Provenance;
    /// NEW. The same flattening. The result is `Unknown` if nothing survives,
    /// and otherwise a `Fused` labelled `label` (a single survivor is wrapped
    /// too). The empty string is a label.
    pub fn fuse_labelled(provenances: impl IntoIterator<Item = Provenance>, label: impl Into<String>) -> Provenance;
}
impl Display for Provenance   // unchanged text; a labelled fusion renders `label[..]`, an unlabelled one `fused[..]`
```

`Provenance` stays exhaustive. This is an explicit S-6 exception, which the
user signs off on. The module contract says richer origins are compositions
of these five variants rather than new variants, and consumers are meant to
match all five.

**FileProvenance** (CHANGED normalization and docs)

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(from = "FileProvenanceData")]            // private, same fields; normalizes on decode
pub struct FileProvenance { file_path: String, span: Option<Span> }
impl FileProvenance {
    pub fn new(file_path: impl AsRef<str>, span: Option<Span>) -> Self;   // unchanged signature
    pub fn file_path(&self) -> &str;                                      // unchanged
    pub fn span(&self) -> Option<&Span>;                                  // unchanged
}
```

**NamedProvenance** (CHANGED error type)

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "NamedProvenanceData")]       // private {name: String, child: Provenance}
pub struct NamedProvenance { name: String, child: Arc<Provenance> }
impl NamedProvenance {
    /// # Errors  `NamedProvenanceError::EmptyName` if `name` is empty.
    pub fn try_new(name: impl Into<String>, child: Provenance) -> Result<Self, NamedProvenanceError>;
    pub fn name(&self) -> &str;            // unchanged
    pub fn child(&self) -> &Provenance;    // unchanged
}
```

**CallSiteProvenance** (unchanged API; derived serde)

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CallSiteProvenance { callee: Arc<Provenance>, caller: Arc<Provenance> }
// new(callee, caller), callee(), caller(): unchanged
```

**FusedProvenance** (CHANGED: `label` replaces `metadata`)

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FusedProvenance { sources: Box<[Provenance]>, label: Option<String> }
impl FusedProvenance {
    /// CHANGED: unlabelled; sources kept exactly as given.
    pub fn new(sources: Vec<Provenance>) -> Self;
    /// NEW: labelled `label`; sources kept exactly as given.
    pub fn labelled(sources: Vec<Provenance>, label: impl Into<String>) -> Self;
    pub fn sources(&self) -> &[Provenance];   // unchanged
    /// RENAMED from `metadata`.
    pub fn label(&self) -> Option<&str>;
}
```

**HasProvenance**: unchanged.

**Errors** (NEW, replacing `ProvenanceError`; one type per constructor
family, per S-5)

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PositionError { ZeroLine, ZeroColumn }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SpanError {
    EndOffsetBeforeStart { start: u64, end: u64 },
    EndPositionBeforeStart { start: Position, end: Position },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum NamedProvenanceError { EmptyName }

impl Display + std::error::Error for each   // source() = None; Display below
```

| Variant | `Display` |
|---|---|
| `PositionError::ZeroLine` | `a position's line must be at least 1` |
| `PositionError::ZeroColumn` | `a position's column must be at least 1` |
| `SpanError::EndOffsetBeforeStart { start: 5, end: 3 }` | `a span's end offset 3 precedes its start offset 5` |
| `SpanError::EndPositionBeforeStart { start: 2:1, end: 1:1 }` | `a span's end position 1:1 precedes its start position 2:1` |
| `NamedProvenanceError::EmptyName` | `a named provenance's name must be non-empty` |

**REMOVED:** `ProvenanceError`, `Span::try_new`, `FusedProvenance::metadata`,
and the second parameter of `Provenance::fuse` and `FusedProvenance::new`.

#### 2.3 Crate-private items

- **REMOVED:** `crate::decode` (`Decode`, `deserialize_via_payload`,
  `deserialize_map_only`, `MapOnly`, `DeferredPayload`) and
  `crate::decode::buffered` (`BufferedValue`, `BufferedMap`).
- **REMOVED:** `crate::python_text`. §5.4 lists where each part goes.
- **CHANGED:** `crate::test_support`'s isolation harness (§5.6). Only
  `cfg(test)`.

#### 2.4 Manifests (NEW or CHANGED)

Workspace `Cargo.toml` `[workspace.dependencies]`:

```toml
serde = { version = "1", features = ["derive", "rc"] }   # CHANGED: + rc (Arc<Provenance> fields; B3 may rely on it too)
serde_json = "1"                                         # CHANGED: no features. Nothing in the workspace may enable arbitrary_precision on it.
postcard = { version = "1.1", default-features = false, features = ["use-std"] }   # NEW, dev-only
```

The comment block above `serde_json` (lines 17-22) is deleted.

`rust/fhy-core/Cargo.toml`:

```toml
[dependencies]
num-bigint = { workspace = true }
num-traits = { workspace = true }
serde = { workspace = true }
# serde_json REMOVED from [dependencies] (together with the line-26 comment) once B1 and B3 remove the last src uses (§5.2).

[dev-dependencies]
fhy-core = { path = ".", features = ["testing"] }   # unchanged (F-018 may delete the feature; not this batch)
postcard = { workspace = true }                      # NEW
proptest = { workspace = true }
rstest = { workspace = true }
serde_json = { workspace = true }                    # MOVED here from [dependencies]
```

`fhy-core-py` does not depend on serde_json today. If the binding later
needs big-integer JSON numbers for Python's envelope, it may enable
`arbitrary_precision` in its own manifest only. It is a `cdylib` with no
Rust dependents, so the feature cannot leak from there.

#### 2.5 Crate docs (`lib.rs`, lines 13-21 replaced)

The `arbitrary_precision` paragraph is replaced by a `# Serialization`
section with this content:

> Every public type that implements `Serialize` also implements
> `Deserialize`, with a plain serde shape that this crate defines and
> documents on the type. The impls are format-agnostic. They work with
> self-describing formats such as JSON and with non-self-describing formats
> such as postcard, and the tests round-trip every such type through both.
> Integers too large for `u64`/`i64` serialize as decimal strings.
> Deserializing an `Identifier` advances the process-global id counter past
> its id, and deserializing a `Canonical<T>` interns the value. A decode that
> fails partway may leave both effects behind for the parts it already
> decoded. Both effects only ever add ids and canonical values, so they never
> invalidate an existing identifier or canonical value. This crate enables
> no `serde_json` features and does not depend on `serde_json` at run time.
> The `__type__`/`__data__` envelope that Python's serialization framework
> uses belongs to the Python binding, not to these shapes.

---

### 3. Interface delta

Semver per S-8. Every change is allowed before first publish, but each one
is still classified.

| Change | Item | Before (HEAD) | After | Semver | Call sites to update |
|---|---|---|---|---|---|
| REMOVED | `diagnostic::get_rationale_note_kind` etc. (4) | `pub fn get_rationale_note_kind() -> &'static Canonical<NoteKind>` (macro-generated, `described_tag.rs:140`) | `NoteKind::rationale()` etc. | breaking | `src/pass_infrastructure/{context.rs:61, validation.rs:26, manager.rs:340}` via `Note::with_other_kind` (no change needed); `src/identifier.rs:189` (doc link, B1 file); `src/shipped.rs:34-35,117-120` (deleted by B1); `tests/diagnostic_stories.rs` (47 uses); `tests/pass_infrastructure_core_stories.rs:22,739`; `tests/pass_infrastructure_validation_stories.rs:9,346,367` |
| CHANGED | `Note: Deserialize` | hand-written through `Decode`/`DeferredPayload`; map-only; JSON-only | derived, `deny_unknown_fields` | breaking in behavior (the sequence form is accepted and effects happen in field order) | none in src; tests §8 |
| REMOVED | `diagnostic::NotePayload` | `pub(crate) struct` | none | non-breaking | `diagnostic.rs` only |
| CHANGED | `DiagnosticLevel` | exhaustive enum | `#[non_exhaustive]` | breaking | in-crate matches are unaffected. Out-of-crate exhaustive matches: `tests/provenance_diagnostic_properties.rs:380-384` (`render_report`, rewritten anyway) |
| CHANGED | `Diagnostic::new` | `pub fn new(level: DiagnosticLevel, message: Note, source: impl Into<String>, detail: Option<String>) -> Self` | `pub fn new(level, message, source) -> Self` | breaking | `src/pass_infrastructure/context.rs:49`, `manager.rs:338`, `validation.rs:24` (pass batch's files; the mechanical edit is listed here); `tests/diagnostic_stories.rs:33,43,402,418,570,671,677`; `tests/provenance_diagnostic_properties.rs:355` |
| NEW | `Diagnostic::{error, warning, info, with_detail}` | none | §2.1 | non-breaking | none |
| NEW | `impl Display for Diagnostic` | none | §2.1 | non-breaking | none |
| REMOVED | `ValidationReport::format` | `pub fn format(&self) -> String` | `impl Display for ValidationReport<R>` | breaking | `src/pass_infrastructure/manager.rs:342` (`Some(report.format())`, pass batch; see §5.5); `tests/diagnostic_stories.rs:468,516,537,550,562,577,591`; `tests/provenance_diagnostic_properties.rs:445,467`; `tests/pass_infrastructure_validation_stories.rs:572` |
| CHANGED | `ValidationFailedError: Display` | the whole multi-line `format()` text | `validation failed with N error(s)` | breaking in behavior | `tests/diagnostic_stories.rs:627-633,648,691`; `tests/provenance_diagnostic_properties.rs:467`; `tests/pass_infrastructure_manager_stories.rs:728,850` pin the *detail* text built from `format()` (pass batch) |
| CHANGED | `Position::try_new` | `-> Result<Self, ProvenanceError>` | `-> Result<Self, PositionError>` | breaking | `tests/provenance_stories.rs` (4), `tests/provenance_diagnostic_properties.rs` (4) |
| CHANGED | `Position: Deserialize` | `PositionPayload` via `deserialize_map_only` | derived on `NonZeroU64` fields | breaking in behavior | tests only |
| REMOVED | `Span::try_new` | `pub fn try_new(start_offset: Option<u64>, end_offset: Option<u64>, start_position: Option<Position>, end_position: Option<Position>) -> Result<Self, ProvenanceError>` | builders §2.2 | breaking | `tests/provenance_stories.rs` (13), `tests/provenance_diagnostic_properties.rs` (1) |
| NEW | `Span::{from_offsets, from_positions, with_offsets, with_positions, with_start_offset, with_end_offset, with_start_position, with_end_position}` | none | §2.2 | non-breaking | none |
| CHANGED | `Span::unknown` | `pub fn unknown() -> Self` | `pub const fn unknown() -> Self` | non-breaking | none |
| CHANGED | `Span: Deserialize` | `SpanPayload`, all four keys required, map-only | `try_from = "SpanData"`; a missing key reads as `None` | breaking in behavior | tests only |
| CHANGED | `Provenance::fuse` | `pub fn fuse(provenances: impl IntoIterator<Item = Provenance>, metadata: Option<&str>) -> Provenance` | `pub fn fuse(provenances) -> Provenance` | breaking | `tests/provenance_stories.rs` (35), `tests/provenance_diagnostic_properties.rs` (12) |
| NEW | `Provenance::fuse_labelled` | none | §2.2 | non-breaking | none |
| CHANGED | `Provenance` wire shape | `{"__type__": "provenance.<kind>", "__data__": {..}}` | externally tagged: `"unknown"`, `{"file": {..}}`, `{"named": {..}}`, `{"call_site": {..}}`, `{"fused": {..}}` | breaking (no persisted data, decision 4) | `tests/provenance_stories.rs:1064-1117, 1160-1170, 1323-1395`; `tests/payload_form_stories.rs` (deleted) |
| CHANGED | `FileProvenance::new` normalization | Python `PurePosixPath`: exactly two leading `/` are kept | any run of leading `/` becomes one `/` | breaking in behavior | `tests/provenance_stories.rs:393,406,417,418,440-450` |
| CHANGED | `NamedProvenance::try_new` | `-> Result<Self, ProvenanceError>` | `-> Result<Self, NamedProvenanceError>` | breaking | `tests/provenance_stories.rs` (4), `tests/provenance_diagnostic_properties.rs` (1) |
| CHANGED | `FusedProvenance::new` | `pub fn new(sources: Vec<Provenance>, metadata: Option<String>) -> Self` | `pub fn new(sources: Vec<Provenance>) -> Self` | breaking | `tests/provenance_stories.rs` (4), `tests/provenance_diagnostic_properties.rs` (1) |
| NEW | `FusedProvenance::labelled` | none | §2.2 | non-breaking | none |
| RENAMED | `FusedProvenance::metadata` | `pub fn metadata(&self) -> Option<&str>` | `pub fn label(&self) -> Option<&str>` | breaking | `tests/provenance_stories.rs` (4), `tests/provenance_diagnostic_properties.rs` (3) |
| REMOVED | `ProvenanceError` | `#[non_exhaustive] pub enum ProvenanceError { ZeroLine, ZeroColumn, EndOffsetBeforeStartOffset{..}, EndPositionBeforeStartPosition{..}, EmptyName }` | `PositionError`, `SpanError`, `NamedProvenanceError` | breaking | `tests/provenance_stories.rs` (16) |
| REMOVED | `decode` module (all items) | `pub(crate)` | none | non-breaking | §5.1 |
| REMOVED | `python_text` module (all items) | `pub(crate)` | none | non-breaking | §5.4 |
| CHANGED | workspace `serde_json` | `features = ["arbitrary_precision"]` | no features | breaking for dependents that relied on the leaked feature (none exist, decision 5) | §5.2 |
| CHANGED | fhy-core `serde_json` | `[dependencies]` | `[dev-dependencies]` | non-breaking (it was never re-exported) | §5.2 |
| NEW | workspace `postcard`, serde `rc` | none | §2.4 | non-breaking | none |

---

### 4. Encapsulation delta

- **Deleted crate-private surfaces:** `decode` and `decode::buffered`, which
  were reached from `identifier`, `described_tag`, `value_domain`,
  `diagnostic`, `provenance` and `symbolic::expression::wire`; and
  `python_text`, which was reached from `symbolic::expression::literal`. That
  removes two cross-layer dependencies from foundation modules into shared
  helpers.
- **Private shadow types:** provenance gets private `SpanData`,
  `FileProvenanceData` and `NamedProvenanceData`, which never appear in a
  public signature. They are named only in `#[serde(try_from/from = ..)]` and
  in `impl TryFrom<..>`/`impl From<..>` on the public type. These replace
  today's private `PositionPayload`, `SpanPayload`, `UnknownFields`,
  `FileFields`, `NamedFields`, `CallSiteFields`, `FusedFields`,
  `FilePayload`, `NamedPayload`, `CallSitePayload`, `FusedPayload` and
  `ProvenancePayload` (12 types become 3). `Position`, `CallSiteProvenance`,
  `FusedProvenance`, `Provenance` and `Note` need none.
- **Invariant-carrying types and bypass:** `Span` (bound order),
  `NamedProvenance` (non-empty name) and `FileProvenance` (normalized path)
  decode through `try_from`/`from`, so serde cannot build a value that skips
  the constructor. `Position` relies on `NonZeroU64`. `provenance.rs` stays a
  leaf module with no descendants.
- **Error enums** get structured fields, are `#[non_exhaustive]`, and each
  has only the variants its constructor can return.
- **`DiagnosticLevel`** becomes `#[non_exhaustive]`.
- Visibility widens nowhere. `unreachable_pub` and `unnameable_types` stay
  clean.

---

### 5. Behavior

#### 5.1 F-011: decode framework deleted; the replacement pattern for every type

`decode.rs` and `decode/buffered.rs` are deleted in the commit that removes
their last user. Each batch removes its own users:

| User at HEAD | What it uses | Owner |
|---|---|---|
| `diagnostic.rs:27,149-172` | `Decode`, `DeferredPayload`, `deserialize_via_payload`, `intern_decoded` | **B2** |
| `provenance.rs:33,110,289,580-601` | `deserialize_map_only` (7 uses) | **B2** |
| `identifier.rs:36,316-402` | `Decode for Identifier`, `IdentifierPayload`, `PayloadId` | B1 |
| `described_tag.rs:92-136` | macro-generated `Decode` and `$Payload` for `OpAttribute`/`NoteKind` | B1 (macro replaced by the generic) |
| `value_domain.rs:28-31,145-200` | `Decode`, `DeferredPayload` parent | B1 (assumed; confirm the owner) |
| `interned.rs:387-399` | `intern_decoded` (keep or inline; format-agnostic) | B1 |
| `symbolic/expression/wire.rs:18-19,465-491` | `Decode for Expression`, `ExpressionPayload` buffered as `serde_json::Value` | B3 |
| `tests/tag_type_equivalence.rs:749,933` | `"decode"` golden ops | whoever deletes the tag-type corpus (triage) |
| `CONTRIBUTING.md` "Decoding checks the payload before its side effects" (lines 311-324) | the rule itself | **B2** (text below) |

**Cross-cutting serde rule (binding on B1, B2, B3).** It replaces the
CONTRIBUTING section above, under the new heading "Serialization is plain
serde". In `rust/fhy-core/src`:

1. **Pattern A, plain derive.** `#[derive(Serialize, Deserialize)]` plus
   `#[serde(deny_unknown_fields)]` on structs, and `#[serde(rename_all =
   "snake_case")]` on enums, using serde's default externally tagged
   representation. Use this whenever every combination of field values is
   valid. B2 uses it for `Note`, `Position`, `CallSiteProvenance`,
   `FusedProvenance` and `Provenance`.
2. **Pattern B, validated.** Derive `Serialize` on the type, and use
   `#[serde(try_from = "XData")]` for `Deserialize`. `XData` is private, has
   **the same field names in the same order** as the type (postcard encodes
   by position), and carries `deny_unknown_fields`. `impl TryFrom<XData> for
   X` calls the public constructor, and its error's `Display` becomes the
   decode error through `de::Error::custom`. B2 uses it for `Span` and
   `NamedProvenance`.
3. **Pattern C, normalizing.** The same as B with `#[serde(from = "XData")]`
   when decoding normalizes and cannot fail. B2 uses it for
   `FileProvenance`.
4. **Pattern D, side-effecting fields.** A field of type `Identifier` or
   `Canonical<T>` needs no special handling. Its own `Deserialize` restores
   or interns when serde reaches it, in payload field order. No level is
   pre-checked, buffered or deferred. A decode that fails later leaves the
   earlier effects in place (S-2).
5. **Pattern E, hand-written impls.** Allowed only where a derive cannot
   express the shape or the algorithm, for example B3's iterative
   serializer for deep trees (F-003) or `Canonical<T>`'s intern step. A
   hand-written struct visitor implements both `visit_seq` and `visit_map`.
6. **Forbidden in `src/` outside `#[cfg(test)]`:**
   - `#[serde(tag)]`, `content`, `untagged`, `flatten` and
     `skip_serializing_if`;
   - calls to `deserialize_any`, `deserialize_ignored_any` or
     `deserialize_identifier` from a hand-written impl;
   - branching on `is_human_readable`;
   - any `serde_json` type (`Value`, `Number`, `Map`, `RawValue`). With
     serde_json only a dev-dependency (§2.4), using one no longer compiles.
7. **Numbers.** Fixed-width integers serialize as themselves. `BigInt`
   serializes as a decimal string in every format (S-2, implemented by B3).
8. **Every public `Serialize` type** has a JSON round-trip test and a
   postcard round-trip test (§8.4).

**Wire shapes this batch owns, before and after (JSON):**

| Type | HEAD | After |
|---|---|---|
| `Note` | `{"message": "m", "kind": {..}}` | same |
| `Position` | `{"line": 1, "column": 2}` | same |
| `Span` | four keys, `null` for absent; all required | four keys written; a missing key reads as `None` |
| `Provenance::Unknown` | `{"__type__": "provenance.unknown", "__data__": {}}` | `"unknown"` |
| `Provenance::File` | `{"__type__": "provenance.file", "__data__": {"file_path": "a", "span": null}}` | `{"file": {"file_path": "a", "span": null}}` |
| `Provenance::Named` | `…"provenance.named"… {"name", "child"}` | `{"named": {"name": "n", "child": <provenance>}}` |
| `Provenance::CallSite` | `…"provenance.call_site"… {"callee", "caller"}` | `{"call_site": {"callee": .., "caller": ..}}` |
| `Provenance::Fused` | `…"provenance.fused"… {"sources", "metadata"}` | `{"fused": {"sources": [..], "label": null}}` |

The JSON array form of a struct, meaning its fields in order without keys,
now decodes, because that is the derived behavior MapOnly used to refuse. It
is not a documented shape, and tests do not pin it either way.

**Impure `from_str` (F-011's last bullet).** Per S-2 this is documented, not
prevented. After a failed `serde_json::from_str::<Note>`, the kind's
identifier may have advanced the counter and the kind may be registered. A
retry sees the registered kind and returns the canonical handle, so a retry
gives the same result as the first successful attempt would have. The
misleading `invalid type: map` errors for floats or big integers inside a
deferred level disappear, because nothing is deferred and no number is
read as a map.

#### 5.2 F-012: `arbitrary_precision` removed; every dependent place

| Place at HEAD | What depends on the feature | Owner | After |
|---|---|---|---|
| `Cargo.toml:16-23` | `serde_json` features, comment | **B2** | §2.4 |
| `rust/fhy-core/Cargo.toml:26-27` | comment, `serde_json` in `[dependencies]` | **B2** | move to `[dev-dependencies]` once the next four rows land |
| `lib.rs:13-21` | crate docs | **B2** | §2.5 |
| `provenance.rs:316-325` | rustdoc on serde_json's nesting limit | **B2** | §5.3 |
| `identifier.rs:34,276-313` | `serde_json::Number` plus `deserialize_any` in `PayloadId`; token sniffing for `.`/`e`/`E` | B1 | a typed `u64` field and the `ID_CAP` check (S-3) |
| `symbolic/expression/wire.rs:16,142-162` | `Number::from_str` for integers beyond `u64` | B3 | decimal string (S-2) |
| `wire.rs:180-200,290-315,457-461` | docs, `parse_literal_value`'s re-stringified token, `Value` buffering | B3 | typed payload |
| `symbolic/expression/node.rs:363-377` | rustdoc on serde_json's nesting limit | B3 | B3 |
| `tests/provenance_stories.rs:1223-1233,1275-1278,1405-1420` | expects `"invalid number"` (text produced only under `arbitrary_precision`); `json!` with `u128` values above `u64::MAX` (these panic without the feature) | **B2** | §8.1 |
| `tests/diagnostic_stories.rs:349-352` | `PayloadId`'s message | **B2** | §8.1 |
| `tests/expression_wire_stories.rs:177-179,220-221,355-377` | big-integer JSON integers, serde_json messages | B3 | B3 |
| `src/identifier.rs:540-541,780-820` (unit tests) | `Number`-derived messages | B1 | B1 |

The cross-cutting rule is §5.1 item 6, plus this: no manifest in the
workspace except `fhy-core-py`'s may enable a serde_json feature. The
regression tests in §8.3 catch the feature coming back in any build that
includes fhy-core.

#### 5.3 Provenance

- **`fuse`/`fuse_labelled`.** The contract is the one `fuse(.., None)` and
  `fuse(.., Some(l))` have today (`provenance.rs:349-394`), with `metadata`
  renamed to `label`. `Unknown` is an identity for both. `fuse` is
  associative, and `fuse_labelled` is not associative across nested labels.
  Flattening walks an explicit stack.
- **Span builders.** `Span::unknown().with_end_offset(3)?.with_start_offset(5)`
  is `Err(SpanError::EndOffsetBeforeStart { start: 5, end: 3 })`. Equal
  bounds are accepted (`from_offsets(3..3)`). Every builder replaces the
  bound it names, so `with_offsets` on a span that already has offsets
  replaces both. `from_offsets(0..3)?.with_positions(p(1,1)..p(1,4))?`
  displays as `1:1-1:4`. Display rules are unchanged.
- **File path normalization (decision 3: kept as useful, Rust-defined
  behavior).** This is a platform-independent lexical normalization: `/` is
  the only separator, empty and `.` components are removed, `..` is kept, any
  run of leading separators becomes one root `/`, and a path with no root and
  no components becomes `.`. Backslashes, `C:` and `~` are ordinary
  characters. The rustdoc no longer mentions `PurePosixPath` or Python, and
  there is no "Matches the Python implementation" line, because this is not
  a parity item. **Change:** `//a` becomes `/a` (today it stays `//a`, a
  POSIX implementation-defined quirk that Python keeps), and `//` becomes
  `/`. Equality, hashing and `Display` use the normalized text, as today.
  The normalization is kept because it makes `./src//a.fhy` and `src/a.fhy`
  one provenance at no cost, and a frontend that already normalizes loses
  nothing.
- **Nesting depth doc.** The "Nesting depth" rustdoc is kept, since the
  recursion of `Eq`, `Hash`, `Debug`, serde and `Drop` is unchanged, but it
  is rewritten:
  - **JSON:** serde_json's text deserializer refuses input nested more than
    127 JSON levels deep, with an error, not a crash. A named or call-site
    level takes two JSON levels and a fused level three. No exact level count
    is stated.
  - **Non-self-describing formats:** postcard has **no** nesting limit, so
    decoding untrusted bytes can recurse as deep as the input allows, about
    two bytes per named level. A caller decoding untrusted input bounds its
    size, or uses a format with a depth limit.

#### 5.4 F-013: `python_text.rs` deleted

`python_text.rs` has exactly one consumer, `symbolic/expression/literal.rs:9-12`
(B3). Its parts go as follows:

| Item | Nature | Destination (owner) |
|---|---|---|
| `is_ascii_digit_run`, `split_decimal_text`, `NormalizedDecimal`, `normalize_decimal_text` | value-level exact-decimal normalization that literal equality and F-026's `Decimal(Normalized)` need; it is not Python text | move, private, into B3's literal module (for example `expr/literal/decimal.rs`), with their unit tests (§8.2) |
| `format_float_repr` and its helpers (`find_shortest_digits`, `is_round_trip`, `find_even_tie_digits`, `decode_float`, `raise_to_power`, `is_exactly_half_of`, `write_*`) | CPython `repr(float)` | delete; literal `Display` and `canonical_key` use Rust `{}` (S-4) (B3) |
| `format_bool` | `True`/`False` | delete; use `{}` (B3) |
| `format_normalized_decimal` | Python `Decimal` `E` notation | B3 decides the decimal `Display`. If it keeps a scientific form, the code lives in the literal module and is described in Rust terms |

Other Python-text emulation outside `python_text.rs`:

- **B2 (this spec):** `provenance.rs:619-657` (`PurePosixPath`, §5.3), and
  `diagnostic.rs:266-268,329-356` (`No validation diagnostics.`,
  `[ERROR] source: message`, §5.5).
- **Pass batch:** `pass_infrastructure/registry.rs:77-85`
  (`is_python_whitespace`), plus the Python-shaped messages in `manager.rs`
  and `validation.rs` (`Pass "…" …: verification reported N error(s).`,
  `Validator "…" raised …`).
- **Also:** `CONTRIBUTING.md:365-371`'s "same message" rule is narrowed by
  decision 3, and the owner of CONTRIBUTING edits applies that.

`lib.rs` drops `mod python_text;` and `mod decode;`.

#### 5.5 F-015 and F-014: reports and errors

```
report = [error(missing return, shape.check).with_detail("function foo() has no return statement"),
          warning(unused, scope.check), info(fyi, v3)]
report.to_string() ==
"error[shape.check]: missing return\n    detail: function foo() has no return statement\nwarning[scope.check]: unused\ninfo[v3]: fyi"
ValidationReport::<()>::new(vec![], vec![]).to_string() == ""
report.into_result().unwrap_err().to_string() == "validation failed with 1 error"
```

- An empty detail is omitted and a whitespace-only detail is rendered, as
  today. Braces are written literally.
- The error's `Display` is one line and never includes the report. A caller
  who wants the report's text prints `error.report()`.
- **`manager.rs:342` (pass batch).** It stores the multi-line report text as
  the failure diagnostic's detail. The recommendation is to pass no detail,
  because the `PassError` already carries the report
  (`verification_report()`). That removes F-015's "inlined into other
  diagnostics' detail". If the pass batch keeps a detail, it uses
  `report.to_string()`.
- **Provenance errors (F-014).** Each constructor returns only its own
  family. `Display` is one lowercase line with no trailing period and
  renders the fields. `source()` is `None`.
- **Decode errors.** A payload that violates an invariant produces
  `D::Error::custom(<crate error>)`, so the decode error's text contains the
  crate's `Display`. Structural errors (missing field, wrong type) are
  serde's.

#### 5.6 F-038: isolated-child-process harness (`test_support.rs:62-95`)

It is replaced by one function. There are no `#[ignore]` twins, and success
does not depend on libtest's output.

```rust
/// Environment variable naming the one test a child process of the test binary runs.
const ISOLATED_TEST_VARIABLE: &str = "FHY_CORE_ISOLATED_TEST";
/// Exit status a child reports after `body` returns. libtest exits 0 or 101, so this
/// status proves the body ran to completion.
const ISOLATED_BODY_RETURNED: i32 = 86;

/// Run `body` alone in a fresh child process of this test binary.
///
/// In the parent (variable unset), re-run the current test executable with
/// `[test_path, "--exact", "--test-threads=1"]` and the variable set to `test_path`,
/// then assert the child's exit code is `ISOLATED_BODY_RETURNED`. In the child
/// (variable equal to `test_path`), run `body` and `std::process::exit(ISOLATED_BODY_RETURNED)`.
///
/// # Panics
/// In the parent: if the child cannot start, or exits with any other status (the
/// body panicked, or no test matched `test_path`), with a message starting
/// `isolated test {test_path} failed:` followed by the status, stdout and stderr.
/// In a child whose variable names a different test: always.
#[track_caller]
pub(crate) fn run_in_isolated_process(test_path: &str, body: impl FnOnce());
```

- An isolated test is a plain `#[test] fn`. It is not `#[ignore]` and not an
  rstest case, because rstest's generated `case_N_*` names are not a
  contract. It calls `run_in_isolated_process("module::tests::name", ||
  { .. })`.
- Under `--include-ignored` nothing is ignored, so nothing passes vacuously.
- Under nextest it still works, since the child is a libtest-compatible
  process.
- `is_isolated_run` and `assert_isolated_test_passes` are REMOVED.
- **Users after this spec:**
  - `diagnostic.rs` has none. Its isolated test is deleted (§8.2): under S-3
    the shipped kinds have fixed reserved ids, and a payload id at or above
    `ID_CAP` is rejected before anything is restored, so "exhaust the counter
    before first use" can no longer happen.
  - B1 ports `identifier.rs:835-853` (restoring `ID_CAP - 1` exhausts the
    counter, which still needs a fresh process). The `value_domain.rs` and
    `op_attribute.rs` isolated tests are moot for the same reason as
    diagnostic's; B1 deletes or ports them.
  - `shipped.rs` is deleted (decision 1).

---

### 6. Error and panic model

| Condition | Result |
|---|---|
| `Position::try_new(0, _)` / `(_, 0)` | `Err(PositionError::ZeroLine)` / `Err(ZeroColumn)`; the line is checked first |
| A span builder given an end before its start (offsets or positions) | `Err(SpanError::EndOffsetBeforeStart { start, end })` / `EndPositionBeforeStart` |
| `NamedProvenance::try_new("", _)` | `Err(NamedProvenanceError::EmptyName)`; whitespace names are accepted |
| Decoding a zero line or column | serde error (from serde's `NonZeroU64` impl) |
| Decoding a span with reversed bounds, an empty named-provenance name | serde error whose text contains the `SpanError`/`NamedProvenanceError` `Display` |
| Decoding an unknown field, a missing non-`Option` field, a wrong type, an unknown variant | serde error. The text is serde's and is not a crate contract |
| Decoding a note whose kind id is at or above `ID_CAP` | serde error (B1's check); the counter does not move |
| Decoding a note whose kind conflicts with a registered one | serde error from `Canonical<T>` (B1); the id is already restored |
| JSON nested past serde_json's limit | serde error, no crash |
| Postcard input nested deeply enough to exhaust the stack | stack overflow (abort). Documented (§5.3), not prevented |
| `into_result` on a report with an error | `Err(ValidationFailedError)`; `source()` is `None` |
| Everything else in `diagnostic`/`provenance` | infallible, no panics |

Changes from HEAD: `ProvenanceError` is split. Nothing in this batch now
has an ordered no-side-effect guarantee on rejection. `ValidationFailedError`'s
`Display` is shorter. Decode error texts change: the `in `kind`:` prefix
goes, and the private type names `PositionPayload`, `UnknownFields` and
`ProvenancePayload` go.

---

### 7. Non-goals

- `DescribedTag<K>` itself, `NoteKind::new`'s registration result (F-021's
  "`new` registers globally"), and bare `NoteKind: Deserialize` (F-020): B1.
- Expression wire shape, `BigInt` decimal strings, literal `Display` and
  `canonical_key`, and where decimal normalization lands: B3.
- `PassContext::report(level, Note, Option<String>)` (another `Option`
  parameter) and pass-infra messages: pass batch.
- Serde for `Diagnostic`, `DiagnosticLevel` and `ValidationReport`. Python
  does not serialize them, and no caller asks for it.
- Iterative (non-recursive) `Drop`, `Eq`, `Hash` and serde for deep
  `Provenance` trees (§9, unraised).
- Collapsing `FileProvenance { span: None }` and `Some(Span::unknown())`
  (§9, unraised).
- `testing` feature and `RegistryGuard` removal (F-018, F-031); pinned-id
  headroom (F-033).
- Test-binary consolidation (F-034). New test files are named below as
  files. If F-034 lands first, they become `tests/it/<name>.rs` modules
  instead.

---

### 8. Test plan

#### 8.1 Existing integration tests

**`tests/diagnostic_stories.rs`**

- **Keep, with mechanical renames only** (`get_*_note_kind` becomes
  `NoteKind::*`, and `Diagnostic::new` plus `detail` becomes the new
  constructors in `build_diagnostic`/`build_detailed_diagnostic`):
  `shipped_note_kind_has_its_documented_name`,
  `shipped_note_kind_is_registered_under_its_name`,
  `shipped_note_kinds_are_distinct`,
  `note_kinds_sharing_a_name_hint_are_distinct`,
  `note_kind_decode_returns_the_shipped_handle`,
  `note_with_other_kind_uses_the_other_note_kind`,
  `note_new_carries_the_explicit_kind`,
  `note_display_renders_kind_and_message`,
  `note_equality_compares_message_and_kind`,
  `note_encodes_as_message_and_kind`, `note_round_trips_each_shipped_kind`,
  `note_with_custom_kind_round_trips`,
  `note_decode_keeps_the_canonical_kind_description`,
  `note_decode_registers_an_unknown_kind`,
  `diagnostic_level_as_str_is_the_lowercase_name`,
  `diagnostic_equality_is_by_value`, `report_filters_diagnostics_by_level`,
  `report_keeps_diagnostics_and_records_in_order`,
  `report_into_result_returns_a_report_without_errors`,
  `validation_failed_error_keeps_the_report_records`, and the `Send + Sync`
  const block.
- **Modify, following B1's API:** `note_kind_new_registers_a_new_kind`,
  `note_kind_equality_ignores_description` (`InternOutcome`), and
  `note_kind_encodes_as_name_and_description` (keep the JSON assertion,
  which is a crate shape).
- **Modify:**
  - `diagnostic_new_stores_every_field` and
    `diagnostic_message_text_omits_the_note_kind` use `Diagnostic::warning(..).with_detail(..)`
    and `Diagnostic::info(..)`.
  - `empty_report_has_no_errors_and_formats_placeholder_text` is renamed
    `empty_report_has_no_errors_and_displays_as_empty_text` and expects `""`.
  - The six `report_format_*` tests (`renders_level_source_message_and_detail`,
    `omits_an_empty_detail`, `keeps_a_whitespace_detail`,
    `keeps_embedded_newlines`, `omits_the_note_kind`,
    `renders_braces_literally`) are renamed to `report_display_*` and assert
    `report.to_string()` against the §5.5 format. The cases stay the same.
  - `report_into_result_escalates_errors_with_the_report` keeps the
    storage-pointer check and expects
    `error.to_string() == "validation failed with 1 error"`.
  - `validation_failed_error_is_a_standard_error` expects `source().is_none()`
    and the one-line message.
  - `a_failed_validation_run_is_reported_to_the_user` asserts the summary
    line, and asserts the full text through `failure.report().to_string()`.
- **Modify, weakening, called out:** `note_decode_rejects_malformed_payloads`
  (6 cases). The exact serde text (`"in `kind`: invalid value: integer `-1`,
  expected an id from 0 to 18446744073709551614"` and others) is replaced by
  `assert_eq!(error.classify(), Category::Data)`. The text was serde's, or
  came from B1's `PayloadId` and the deleted `DeferredPayload`, so it was
  never this crate's contract. One case is added: `kind_id_at_the_cap` (id
  `9223372036854775808`).
- **Delete:** `note_kind_bare_decode_does_not_register_the_kind`, **only
  if** B1 removes bare `NoteKind: Deserialize` (F-020). Otherwise keep it.

**`tests/provenance_stories.rs`**

- **Keep unchanged:** `position_try_new_stores_line_and_column`,
  `position_try_new_accepts_the_largest_u64_values`,
  `position_equality_is_by_value`,
  `position_orders_lexicographically_by_line_then_column`,
  `position_sorting_follows_line_then_column`,
  `position_display_renders_line_colon_column`, `span_unknown_has_no_bounds`,
  `file_provenance_new_without_span_stores_the_path`,
  `file_provenance_new_stores_the_span`,
  `file_provenance_equality_follows_the_normalized_path`,
  `file_provenance_without_span_differs_from_unknown_span`, the named,
  call-site, variant-inequality and hashing tests (lines 464-575), and
  `provenance_round_trips_through_json`,
  `position_and_span_round_trip_through_json` (after the builder rename),
  `decoded_unknown_provenances_compare_equal` (after the payload change) and
  `provenance_decode_normalizes_the_file_path` (after the payload change).
- **Modify, mechanical:**
  - `position_try_new_rejects_zero_components` uses `PositionError`.
  - Every `Span::try_new` call (lines 184-360, 13 sites) uses the §2.2
    builders. `span_try_new_*` becomes `span_builders_*` with the same
    cases:
    - `rejects_end_offset_before_start_offset` becomes
      `span_from_offsets_rejects_a_reversed_range`;
    - `reports_the_offset_error_first` becomes
      `span_with_end_offset_is_checked_against_the_start_offset`;
    - `accepts_bounds_it_cannot_order` keeps its four cases, built with the
      single-bound `with_*`.
  - `span_display_renders_positions_or_offsets` builds with the builders.
  - All 35 `Provenance::fuse` sites move to `fuse` or `fuse_labelled`.
  - `fuse_with_metadata_and_no_survivors_returns_unknown` is renamed
    `fuse_labelled_with_no_survivors_returns_unknown`.
  - `fuse_single_input_with_metadata_wraps_in_fused` is renamed
    `fuse_labelled_wraps_a_single_input`.
  - `fused_provenance_new_stores_the_metadata` is renamed
    `fused_provenance_labelled_stores_the_label`.
  - The metadata-named display cases are renamed to label.
- **Modify, behavior change:**
  - `file_provenance_new_normalizes_the_path` changes four cases:
    `two_leading_separators_kept` (`//a` becomes `/a`, renamed
    `two_leading_separators_collapse`), `double_root` (`//` becomes `/`),
    `two_leading_separators_normalized` (`//a/./b/` becomes `/a/b`), and
    `two_leading_separators_then_current_directory` (`//./a` becomes `/a`).
  - `file_provenance_keeps_paths_that_normalize_differently_apart` drops the
    `//a` versus `/a` assertion and keeps `a/../b` versus `b`. This is not a
    weakening: the pair is now asserted *equal* in the new regression test.
- **Modify, new shape:** `provenance_encodes_in_the_wrapped_form` is renamed
  `provenance_encodes_externally_tagged` and gets the §5.1 table as its
  expectations. `span_encodes_every_key` and
  `position_encodes_as_line_and_column` are unchanged apart from the
  builders.
- **Modify, weakening, called out (F-035):**
  - `position_decode_rejects_malformed_payloads` (10 cases),
    `span_decode_rejects_malformed_payloads` (11) and
    `provenance_decode_rejects_malformed_payloads` (17) are the 28+ exact
    serde and private-DTO strings at lines 1180-1395. The helper
    `assert_decode_rejected` becomes
    `assert_decode_rejected::<T>(payload_text: &str, crate_error: Option<&dyn Display>)`.
    It decodes with `from_str`, asserts `classify() == Category::Data`, and,
    when `crate_error` is given, that the message contains that `Display`.
    Cases that exercise a crate invariant pass their crate error: end offset
    before start, end position before start, invalid nested position, empty
    name, invalid nested source.
  - **Deleted cases, one line each:**
    - `position:not_a_map` and `span:list_position` now decode, because
      MapOnly is gone and the sequence form is serde's behavior.
    - `provenance:unknown_without_data`, `unknown_with_null_data`,
      `unknown_with_a_field`, `extra_envelope_key`, `missing_type`,
      `null_caller`'s "adjacently tagged" text, `not_a_map` and
      `unwrapped_child` test the removed `__type__`/`__data__` envelope.
      They are replaced by new-shape cases: an unknown variant name, a
      variant with a non-map body, the unit variant given a body, and two
      top-level keys.
    - `provenance:non_provenance_type_id` tests the envelope's `__type__`
      dispatch and is covered by the unknown-variant case.
    - `span:missing_*` (4 cases), `provenance:file_missing_span` and
      `provenance:missing_metadata` now decode to `None`. This is a
      behavior change, pinned positively by
      `span_decode_reads_a_missing_bound_as_absent` and
      `file_and_fused_decode_read_a_missing_option_as_absent`.
    - Kept, with a Data-category assertion: `unknown_type_id` (as an unknown
      variant name), `file_path_not_a_string`, `integer_metadata` (as
      `label`) and `null_sources`.
  - The number cases `float_line`, `negative_line` and `negative_offset` are
    kept with a Data-category assertion. Their `"invalid number"` text only
    exists under `arbitrary_precision`.
- **Modify:** `position_decode_rejects_values_beyond_u64` and
  `span_decode_rejects_an_offset_beyond_u64` build their payload as JSON
  *text*, because `json!` with a `u128` above `u64::MAX` panics without
  `arbitrary_precision`. They assert the Data category.
- **Modify, weakening, called out:**
  `provenance_json_text_decodes_up_to_the_serde_json_nesting_limit` becomes
  `provenance_nested_past_the_json_limit_is_refused_not_crashed`. It builds
  200 named levels and asserts `from_str` returns `Err`, and that 40 levels
  round-trip. It no longer pins serde_json's exact limit (62 levels) or the
  `recursion limit exceeded` text, neither of which this crate owns.
- **Modify:** `provenance_error_display_writes_the_full_message` is split
  across the three error types. It keeps the full-text table in §2.2,
  because this text is the crate's own contract.

**`tests/provenance_diagnostic_properties.rs`**

- **Keep:** `file_path_normalization_is_idempotent`,
  `provenance_round_trips_through_json`,
  `provenance_json_text_is_stable_across_a_round_trip`,
  `position_order_matches_line_column_pair_order`,
  `report_has_errors_matches_any_error_level_diagnostic` and
  `report_level_filters_partition_the_diagnostics_in_order`.
- **Modify:** `fuse_result_equals_the_flattened_input`,
  `fuse_is_idempotent_on_its_own_output`,
  `fuse_is_associative_without_metadata` and
  `fuse_treats_unknown_as_an_identity`. The `arbitrary_metadata` strategy
  becomes `Option<&str>`, dispatched by a helper to `fuse` or
  `fuse_labelled`. `arbitrary_diagnostic` uses the new constructors.
- **Modify:** `report_format_writes_the_documented_text` re-implements the
  SUT (F-037), so it becomes `report_display_joins_the_diagnostic_displays`:
  `report.to_string()` equals the diagnostics' `to_string()` joined by
  `"\n"`. Delete `render_report`.
- **Modify:** `report_into_result_fails_iff_it_has_errors` expects the
  summary line with the right count and plural.

**`tests/payload_form_stories.rs`: delete the whole file.** Its two tests,
`a_payload_given_as_a_sequence_is_refused` (14 cases) and
`a_refused_sequence_payload_registers_nothing`, pin MapOnly, which S-2
removes. The sequence form is now serde's derived behavior, and
"registers nothing on refusal" contradicts S-2's partial-effect contract.
Its useful residue, "each type decodes from a second, differently shaped
format", is covered by §8.4's postcard round trips.

**`tests/vocabulary_stories.rs`**

- **Keep:** everything except the next test. The `ALL_*` imports from
  `common/expression.rs` are B3's F-025 concern.
- **Modify, weakening, called out:**
  `vocabulary_rejection_names_the_word_and_the_expected_names` pins serde's
  `invalid value: string "…", expected` framing around `wire_name.rs`'s
  custom `expecting` text. Once B3 derives the enums (F-025) the whole
  message is serde's. It becomes
  `vocabulary_from_str_rejection_names_the_word`, which asserts that B3's
  `FromStr` error `Display` contains the rejected word. The serde path
  keeps only `assert_deserialization_rejects`'s `is_data()`.

**Other files touched by this batch's API changes** (mechanical; they are
owned elsewhere but listed so nothing is missed):

- `tests/pass_infrastructure_core_stories.rs:22,739`
- `tests/pass_infrastructure_validation_stories.rs:9,346,367,572-582`
  (the report text becomes the §5.5 format)
- `tests/pass_infrastructure_manager_stories.rs:728,850` (the detail text,
  depending on the pass batch's choice in §5.5)

#### 8.2 Existing unit tests

- **`decode.rs` tests: delete all five** (`a_payload_rejected_for_an_unknown_field_builds_nothing`,
  `a_malformed_child_is_not_checked_until_the_outer_build_reaches_it`,
  `an_error_two_levels_deep_names_the_path_to_its_level`,
  `a_non_map_child_is_rejected_before_anything_is_built`,
  `a_well_formed_payload_builds_outer_first_then_child`). They test the
  framework being deleted.
- **`diagnostic.rs` tests: delete all nine.**
  - `clearing_the_registry_keeps_the_shipped_kinds_canonical` clears the
    global registry, which decision 2 makes test-only on local registries;
    the shipped kinds' reserved ids are B1's test.
  - `a_valid_note_restores_and_registers_its_kind` is redundant with
    `note_decode_registers_an_unknown_kind` plus B1's exact-id identifier
    tests, and relies on far-ahead pinned ids (F-033).
  - `a_note_rejected_for_a_trailing_unknown_field_restores_nothing`,
    `a_note_whose_kind_precedes_a_malformed_message_restores_nothing`,
    `a_note_whose_kind_has_an_unknown_field_restores_nothing`,
    `a_note_whose_kind_lacks_a_description_restores_nothing` and
    `a_bare_note_kind_rejected_for_an_unknown_field_restores_nothing` lock
    in the ordered no-side-effect contract that S-2 replaces.
  - `decoding_the_largest_id_as_the_first_use_keeps_the_shipped_kinds_in_isolation`
    and its wrapper are moot under S-3: `u64::MAX - 1` is above `ID_CAP`
    and rejected, and the shipped kinds have fixed ids. They are replaced
    by `note_decode_rejects_a_kind_id_at_the_cap` (§8.3).
- **`python_text.rs` tests:**
  - **Move to B3's literal module unchanged:**
    `normalize_decimal_text_folds_every_zero_spelling`,
    `normalize_decimal_text_keeps_thirty_significant_digits_without_rounding`,
    `normalize_decimal_text_keeps_two_hundred_digits`,
    `normalize_decimal_text_handles_exponents_beyond_the_float_range`,
    `normalize_decimal_text_rejects_text_outside_the_grammar`, and the
    proptests `normalize_decimal_text_ignores_zero_padding`,
    `normalize_decimal_text_tracks_the_point_position_in_the_exponent` and
    `normalize_decimal_text_yields_a_canonical_coefficient`.
  - **Move, then B3 adjusts the expected text:**
    `normalize_decimal_text_strips_zeros_and_formats_the_result`, which
    uses `format_normalized_decimal`.
  - **Delete:** `format_float_repr_writes_the_expected_text`,
    `format_float_repr_breaks_an_exact_tie_toward_the_even_digit`,
    `format_float_repr_writes_every_nan_as_nan`, and the proptests
    `format_float_repr_writes_the_shortest_round_trip_digits`,
    `format_float_repr_breaks_every_exact_tie_toward_the_even_digit`,
    `format_float_repr_round_trips_every_non_nan_float` and
    `format_float_repr_never_writes_integer_shaped_text`. CPython
    `repr(float)` is removed and Rust's `{}` is std's responsibility.
    `format_bool_capitalizes` goes for the same reason (`True`/`False` is
    removed).
  - **B3 decides:** `format_normalized_decimal_chooses_notation_by_exponent`
    and `format_normalized_decimal_reads_back_as_the_same_decimal` follow
    the decimal `Display`: delete them if the Python `E` notation goes, move
    them if a scientific form stays.
- **`test_support.rs`:** there were no tests; the harness gets three
  (§8.3).

#### 8.3 Regression test per fixed finding

| Finding | Test (file :: name) | Fails at HEAD because |
|---|---|---|
| F-011 | `serde_format_stories::note_round_trips_through_postcard` (rstest: the four shipped kinds and one custom kind) | `DeferredPayload` calls `deserialize_any`, so postcard returns `WontImplement` (verified with a scratch crate at HEAD) |
| F-011 | `serde_format_stories::provenance_round_trips_through_postcard` (every variant, nested) and `position_round_trips_through_postcard`, `span_round_trips_through_postcard` | MapOnly forwards to `deserialize_map`, so postcard returns `WontImplement` (verified for Position and Provenance; Span gives `SerdeDeCustom`) |
| F-012 | `serde_format_stories::serde_json_numbers_compare_by_value_in_a_build_with_fhy_core`: `from_str::<Value>("1.0") == from_str::<Value>("1.00")` | the leaked feature makes them unequal (verified) |
| F-012 | `serde_format_stories::a_dependent_untagged_float_decodes_in_a_build_with_fhy_core`: a test-local `#[serde(untagged)] enum { Float(f64) }` decodes `1.5` | "data did not match any variant" under the leaked feature (verified in a downstream scratch crate) |
| F-013 (B2 part) | `provenance_stories::file_provenance_collapses_leading_separators_to_one_root` (`//a`, `///a`, `//` equal `/a`, `/a`, `/`) | the PurePosixPath rule keeps `//a` |
| F-013 (B2 part) | `diagnostic_stories::report_display_writes_lowercase_levels_without_a_placeholder` | `format()` writes `[ERROR]` and `No validation diagnostics.` |
| F-014 | `provenance_stories::provenance_constructors_return_their_own_error_family`: `let _: Result<Position, PositionError> = Position::try_new(0, 1);` and the same for `Span::from_offsets(5..3)` and `NamedProvenance::try_new("", ..)`, each matched to its single expected variant | the types do not exist |
| F-014 | `provenance_stories::provenance_error_display_is_one_lowercase_line` (rstest over all five variants: no `\n`, no trailing `.`, first character lowercase) | a compile error before the change (types); a characterization after |
| F-015 | `diagnostic_stories::validation_failed_error_displays_a_one_line_summary` (1 error: `validation failed with 1 error`; 3 errors plus a warning: `… with 3 errors`) | `Display` is the whole report |
| F-015 | `diagnostic_stories::report_display_renders_each_diagnostic_on_its_own_line` | `Display for ValidationReport` does not exist |
| F-021 | `diagnostic_stories::diagnostic_level_constructors_set_the_level` (rstest over `error`/`warning`/`info`) and `diagnostic_with_detail_replaces_the_detail` | the constructors do not exist |
| F-021 | `provenance_stories::span_single_bound_builders_check_their_counterpart` (`with_end_offset(3)` then `with_start_offset(5)` gives `EndOffsetBeforeStart { start: 5, end: 3 }`) | `try_new` only |
| F-021 | `provenance_stories::fuse_labelled_wraps_a_single_input` and `fuse_without_a_label_returns_a_single_input_unchanged` | `fuse` takes `Option<&str>` |
| F-035 | the modified `*_decode_rejects_malformed_payloads` (§8.1) assert `Category::Data` plus crate text, and nothing pins serde's text, a private DTO name, or serde_json's recursion limit | they pin `expected struct PositionPayload` and others |
| F-038 | `test_support::tests::run_in_isolated_process_runs_the_body_in_a_child` (the body asserts the variable equals its path) | `is_isolated_run` twin pattern |
| F-038 | `test_support::tests::run_in_isolated_process_fails_when_no_test_matches` (`#[should_panic(expected = "isolated test")]`, path `test_support::tests::no_such_test`) | today `"1 passed"` parsing, and under `--include-ignored` the body passes vacuously |
| F-038 | `test_support::tests::run_in_isolated_process_fails_when_the_body_panics` (`#[should_panic(expected = "isolated test")]`) | same |

Also, for S-3 at the Note level:
`diagnostic_stories::note_decode_rejects_a_kind_id_at_the_cap`. The id is
`9223372036854775808`. The test asserts a Data error, and that
`Identifier::try_new("after")` still returns `Ok` afterwards, which shows
the counter did not jump. This test depends on B1.

#### 8.4 New story, property and adversarial tests

**NEW `tests/serde_format_stories.rs`.** Module docs: public API only; every
public `Serialize` type round-trips through JSON text and through postcard.
It holds the helper `assert_round_trips<T: Serialize + DeserializeOwned +
PartialEq + Debug>(value: &T)`, which checks both formats, plus these cases:

- **B2 types (fail at HEAD):** `Note`, `Position`, `Span` (unknown,
  offsets only, positions only, each single bound, both pairs) and
  `Provenance` (each variant, labelled and unlabelled fused, a named
  call-site chain).
- **Types from other batches.** Each case goes green when its batch lands:
  - B1: `Identifier`, `Canonical<OpAttribute>`, `Canonical<ValueDomain>`
    (with a parent chain), `Canonical<NoteKind>`;
  - B3: `Expression` (every node kind; a `BigInt` literal beyond `u64` and
    below `i64::MIN`; a float; a decimal), `SymbolType`, `FunctionSort`,
    `UnaryOperation`, `BinaryOperation`.
- The two F-012 regression tests.
- `a_big_integer_literal_serializes_as_a_decimal_string_in_json` (B3's type,
  this batch's rule).

**Properties (`tests/provenance_diagnostic_properties.rs`):**

- `every_provenance_tree_round_trips_through_postcard` (the existing
  `arbitrary_tree`);
- `every_span_round_trips_through_postcard`;
- `every_diagnostic_display_has_a_detail_line_iff_the_detail_is_non_empty`;
- `validation_failed_error_display_never_contains_a_newline`.

**Stories (`tests/provenance_stories.rs`, `tests/diagnostic_stories.rs`):**

- `span_decode_reads_a_missing_bound_as_absent`;
- `span_decode_rejects_reversed_offsets_with_the_span_error_text`;
- `named_provenance_decode_rejects_an_empty_name_with_the_error_text`;
- `file_provenance_decode_normalizes_the_path_in_either_format`;
- `a_verifier_reports_through_diagnostic_builders`, a user story:
  `Diagnostic::warning(note, "tiling.check")` and
  `Diagnostic::error(note, "bounds.check").with_detail("bound -1 in loop i")`
  go into a report, and the failure prints its summary and then
  `error.report()`.

**Adversarial:**

- `truncated_postcard_bytes_are_an_error_not_a_panic`: every strict prefix
  of an encoded nested `Provenance` and `Note` decodes to `Err`.
- The property `arbitrary_bytes_never_panic_when_decoded_as_a_provenance`
  runs over bytes of length ≤ 256, which bounds nesting to about 128 levels
  (§5.3).
- `a_note_decode_that_fails_after_its_kind_leaves_the_kind_registered`
  pins S-2's documented partial effect: the payload has kind first and
  `"message": 5`. After the error, the kind's id is registered and a retry
  with a valid message returns the same canonical kind.

---

### 9. Findings covered, decisions to sign off, and misjudgments

**Covered:**

- F-011 (fixed; decode framework deleted; §5.1 pattern binds B1 and B3)
- F-012 (fixed; B1 and B3 rows in §5.2)
- F-013 (B2 part fixed; literal parts are B3's)
- F-014 (diagnostic and provenance errors)
- F-015 (reports and the failed-validation error)
- F-021 (`Diagnostic::new`, `Span::try_new`, `fuse`)
- F-035 (the four listed files)
- F-038 (fixed)

**Spec-author decisions in this section (the user signs off):**

1. **postcard 1.1 over bincode.** bincode's final release, 3.0.0, is a
   tombstone: its `lib.rs` is `compile_error!` and its README says
   development has ceased (checked in the registry). bincode 1.x/2.x are
   unmaintained. postcard is maintained, serde-native, has a stable
   documented wire format, and refuses `deserialize_any`,
   `deserialize_identifier` and `deserialize_ignored_any` outright
   (`WontImplement`). That makes it the strictest check of the "no
   self-describing assumptions" rule. It is dev-only.
2. **Provenance JSON shape** is serde's external tagging (`"unknown"`,
   `{"file": {..}}`, and so on). `Provenance` stays exhaustive, as an S-6
   exception.
3. **`metadata` becomes `label`** everywhere, including the wire field
   (`fuse_labelled`, `FusedProvenance::labelled`/`label()`).
4. **`FileProvenance` keeps a lexical, platform-independent
   normalization**, but collapses a leading `//` (the POSIX and Python
   quirk).
5. **`ProvenanceError` splits** into `PositionError`, `SpanError` and
   `NamedProvenanceError`, one per constructor family (S-5).
6. **Span builders** take `Range<u64>`/`Range<Position>`, plus four
   single-bound `with_*`, so partial spans stay expressible.
7. **Report text** is `error[source]: message`, with `    detail: …` on the
   next line. An empty report renders as `""`. The failed error renders
   `validation failed with N error(s)`.
8. **Serde defaults are accepted:** a missing `Option` key reads as `None`
   and the JSON array form of a struct decodes. `deny_unknown_fields` is
   kept everywhere.
9. **serde gains `rc`** workspace-wide, and **serde_json moves to
   fhy-core's dev-dependencies**.
10. **`DiagnosticLevel` becomes `#[non_exhaustive]`.**
11. **`Position` decodes through serde's `NonZeroU64`**, so a zero gives
    serde's message, not `PositionError`'s.

**Possibly misjudged or under-scoped in the audit:**

- **F-013 is not all "Python text".** Half of `python_text.rs`
  (`normalize_decimal_text`/`NormalizedDecimal`) is value-level decimal
  normalization that literal equality needs. It must move into B3's literal
  module, not into the binding and not be deleted. The audit's "move
  `python_text` into the binding crate" would break literal `Eq`/`Hash`.
- **F-011's "Impure `from_str`" and "ordered effects" are resolved by
  documentation, not by a fix.** That follows S-2's choice of exact-id
  semantics without a restore step. The audit's suggested `DeserializeSeed`
  or `restore(&mut Session)` is not done. This is recorded so the finding is
  not re-raised.
- **F-038's scope shrinks under S-3.** The diagnostic, value_domain and
  op_attribute isolated tests become unnecessary, not just fragile. Only
  identifier's cap-exhaustion test still needs a fresh process.
- **F-012 goes further than the audit suggested.** With B1 and B3's
  changes, `fhy-core` no longer needs serde_json at run time at all, so no
  "opt-in big-integer JSON feature" is needed in the core.

**Unraised issues found while specifying (not fixed here; for triage):**

- **Deep `Provenance` trees and the stack.** `Drop`, `Eq`, `Hash`, `Debug`,
  `Display` and serde all recurse, the same class of problem as F-003 for
  expressions. Postcard adds an unbounded-depth decode path, so crafted
  bytes of about 2 bytes per level can overflow the stack. Today this is
  documented only.
- **Two ways to write "no span".** `FileProvenance::new(p, None)` and
  `new(p, Some(Span::unknown()))` are unequal but display the same.
- **Note kind before message.** `Note`'s derived decode reads `kind`
  before `message` when the payload lists it first. This is fine under S-2,
  but it means a note payload with a bad message can still register a
  kind. That behavior is intended and pinned by the adversarial test in
  §8.4.

---

## B3: the symbolic expression core, becoming `fhy_core::expr`

Batch files at HEAD 412e234: `src/symbolic/{mod.rs, symbol_type.rs,
wire_name.rs}` and `src/symbolic/expression/{mod.rs, node.rs, literal.rs,
operation.rs, build.rs, builtins.rs, sort.rs, alpha.rs, error.rs, wire.rs,
pprint.rs, screen.rs}`. B3 also deletes `src/python_text.rs`, because
`literal.rs` is its only user (`grep python_text src` finds only
`literal.rs:9` and `lib.rs:34`).

B4 owns `pattern/*` and `registration.rs`. B5 owns the pass infrastructure
and `fhy_core::tree`. Where this section changes a type that B4 or B5
consume, it lists what they have to adapt to.

### 0. Status checked at HEAD

- **F-002: fixed by 4db96b9, verified, not respecified.**
  - `node.rs:174-199`: `is_tree_equal` memoizes pairs of shared nodes.
  - `node.rs:263-297`: `compute_structural_digest` computes a digest per
    distinct shared node.
  - `node.rs:299-327`: `FreeIdentifierCollector` prunes shared nodes it has
    already seen.
  - Alpha-equivalence uses `is_tree_equal`.
  - The regression tests pass: running `cargo test --test
    expression_node_stories --test expression_screen_stories --test
    expression_properties doubling` gave 18 passes, 0 failures.
  - All of these tests stay. Only their `assert_ne!(hash…)` lines change
    (§8).
  - Note: `Hash` still walks the distinct nodes on every call. The audit
    suggested caching a hash per node, but that is not needed for
    correctness (see Non-goals).
- **F-023, DAG re-walk: fixed by 4db96b9, verified.**
  - `screen.rs:57-63` and `:110-126` skip a node seen again with the same
    flags.
- **F-023, quadratic behavior on nested piecewise: NOT fixed.**
  - `is_provably_numeric` builds a fresh `checked` set on every call.
  - `find_numeric_operand` calls it once per Boolean-position operand at
    every level.
  - So a chain of `d` piecewise nodes, each in the otherwise branch of the
    one above and all in Boolean position, costs O(d²).
  - This part is specified in §5.7.
- **Clippy break.** `tests/expression_pass_stories.rs:287` is
  `assert!(outcome.output() == &build_doubling_dag(&a, 64))`. The test author
  avoided `assert_eq!` on purpose: its failure message would `Debug`-print a
  DAG with 2^64 occurrences, which never finishes. Once the bounded `Debug`
  in §5.4 exists, `assert_eq!` is safe, so the fix belongs in this batch's
  test plan (§8.3).

### 1. Summary

The expression core moves to `fhy_core::expr` and stops imitating Python's
runtime model:

- **Nodes:**
  - an n-ary `Logical` node replaces binary `LogicalAnd`/`LogicalOr` and the
    right-fold;
  - built-ins are a `BuiltinFunction` enum behind a `Callee`;
  - literals are four normalized forms.
- **Construction:** plain `From`/`Into` replaces the sealed `IntoOperand`.
- **Errors:** one error type per operation replaces `ExpressionBuildError`.
- **Text:** `Display` follows Rust conventions, and `Debug` is hand-written
  and bounded.
- **Serde:** plain derives over a flat, index-linked node table:
  - there is no recursion per tree level;
  - there is no envelope;
  - the implementation contains no JSON-specific code;
  - DAG sharing is preserved.

The Boolean-position screen becomes a builder over lookup traits and runs in
linear time.

### 2. Module layout (S-1)

```
src/expr/mod.rs          explicit re-exports only (no globs); `pub use num_bigint::BigInt;`
src/expr/node.rs         Expression, ExpressionKind, UnaryExpression, BinaryExpression,
                         LogicalExpression, PiecewiseExpression, CallExpression
src/expr/build.rs        constructors, From impls, operator impls
src/expr/callee.rs       Callee, FunctionName, FunctionNameError
src/expr/literal.rs      LiteralValue, Decimal, LiteralTextError      (absorbs the decimal part of python_text.rs)
src/expr/operation.rs    UnaryOperation, BinaryOperation, LogicalOperation, UnknownNameError
src/expr/sort.rs         FunctionSort
src/expr/symbol_type.rs  SymbolType                                   (was src/symbolic/symbol_type.rs)
src/expr/alpha.rs        AlphaRenaming
src/expr/error.rs        PiecewiseError, RebuildError, NonInjectiveRenamingError
src/expr/display.rs      FormatOptions, Notation, IdentifierStyle, ExpressionDisplay, `impl Debug/Display for Expression`  (was pprint.rs)
src/expr/screen.rs       BooleanScreen, Environment, SymbolTypes, SortLookup, NoRegisteredSorts,
                         BooleanPosition, NonBooleanLogicalOperandError
src/expr/wire.rs         private wire shapes; `impl Serialize/Deserialize for Expression`
src/expr/builtins.rs     `pub mod builtins`: BuiltinFunction, BuiltinConstant, ComposedFunction
src/expr/passes.rs       `pub mod passes`: ExpressionPrettyFormatter (+ B4's RewriteRuleApplier, register_expression_passes)
src/expr/pattern/        B4
```

- **Where the pass types go (S-1 choice for this batch): `fhy_core::expr::passes`.**
  - This is the only module under `expr` that imports `fhy_core::pass`.
  - `fhy_core::pass` does not import `expr`. I grepped
    `src/pass_infrastructure`: the only mentions of `symbolic` are doc
    examples in `tree.rs:424,519`, which move with the tree module in B5.
  - I recommend that B4 put `RewriteRuleApplier` and
    `register_expression_passes` in the same module.
- **What core `expr` depends on:**
  - `crate::identifier`;
  - `crate::tree`, for `Tree`, `NodeHandle`, `NodeIdentity`, `walk_tree`,
    `rewrite_tree` and the identity hasher. **Assumption about B5:** these
    move to `fhy_core::tree`, and the walkers are generic over a context
    `C`, so `expr` calls them with `()` and needs no `PassContext` (F-008).
- `symbolic/` and `wire_name.rs` are deleted.

### 3. Desired public interface

Visibility is `pub`, reachable at the path shown, unless stated otherwise.

#### 3.1 `fhy_core::expr::Expression` and node kinds (`node.rs`)

```rust
#[derive(Clone)]
pub struct Expression(Arc<ExpressionKind>);                      // CHANGED: Debug no longer derived (§5.4)

/// Exhaustive by design (S-6): passes match every node kind.
#[derive(Debug, Clone)]
pub enum ExpressionKind {                                         // CHANGED: + Logical
    Unary(UnaryExpression),
    Binary(BinaryExpression),
    Logical(LogicalExpression),                                   // NEW
    Identifier(Identifier),
    Literal(LiteralValue),
    Piecewise(PiecewiseExpression),
    Call(CallExpression),
}

#[derive(Debug, Clone)] pub struct UnaryExpression { /* private */ }
impl UnaryExpression {
    pub fn operation(&self) -> UnaryOperation;
    pub fn operand(&self) -> &Expression;
}

#[derive(Debug, Clone)] pub struct BinaryExpression { /* private */ }
impl BinaryExpression {
    pub fn operation(&self) -> BinaryOperation;
    pub fn left(&self) -> &Expression;
    pub fn right(&self) -> &Expression;
}

/// A conjunction or disjunction of two or more operands, in order.   NEW
#[derive(Debug, Clone)] pub struct LogicalExpression { /* private */ }
impl LogicalExpression {
    pub fn operation(&self) -> LogicalOperation;
    /// Always at least two operands.
    pub fn operands(&self) -> &[Expression];
}

#[derive(Debug, Clone)] pub struct PiecewiseExpression { /* private */ }
impl PiecewiseExpression {
    pub fn cases(&self) -> &[(Expression, Expression)];
    pub fn otherwise(&self) -> &Expression;
    // REMOVED from pub: try_new -> pub(crate); construct with Expression::piecewise
}

#[derive(Debug, Clone)] pub struct CallExpression { /* private */ }
impl CallExpression {
    pub fn callee(&self) -> &Callee;                              // NEW (replaces function_name)
    pub fn arguments(&self) -> &[Expression];
    // REMOVED: function_name() -> &str   (use callee().name())
    // REMOVED from pub: try_new -> pub(crate); construct with Expression::call
}

impl Expression {
    pub fn kind(&self) -> &ExpressionKind;
    pub fn ptr_eq(this: &Self, other: &Self) -> bool;
    /// Unary: operand. Binary: left, right. Logical: operands in order.
    /// Piecewise: c0, v0, c1, v1, ..., otherwise. Call: arguments. Leaves: none.
    pub fn children(&self) -> impl DoubleEndedIterator<Item = &Expression>;
    /// # Errors
    /// RebuildError::ChildCount if children.len() differs from the node's own
    /// count; RebuildError::Piecewise if a new case condition is a
    /// non-Boolean literal. Never flattens a Logical node (§5.1).
    pub fn rebuild_with_children(&self, children: Vec<Expression>)
        -> Result<Expression, RebuildError>;                      // CHANGED error type
    pub fn free_identifiers(&self) -> HashSet<Identifier>;
    /// # Errors
    /// PiecewiseError::NonBooleanConditionLiteral if a replacement puts a
    /// non-Boolean literal in a case condition.
    pub fn substitute<S: BuildHasher>(&self, replacements: &HashMap<Identifier, Expression, S>)
        -> Result<Expression, PiecewiseError>;                    // CHANGED error type
    pub fn is_alpha_equivalent_under(&self, other: &Expression, renaming: &AlphaRenaming) -> bool;
    /// Render under `options`; see §5.4.
    pub fn display(&self, options: FormatOptions) -> ExpressionDisplay<'_>;   // NEW (F-015)
}

impl PartialEq for Expression; impl Eq for Expression; impl Hash for Expression;
impl fmt::Debug for Expression;       // CHANGED: hand-written, iterative, bounded (§5.4)
impl fmt::Display for Expression;     // NEW: == self.display(FormatOptions::default())
impl NodeHandle for Expression;       // path fhy_core::tree (B5)
impl Tree for Expression { type RebuildError = RebuildError; }   // CHANGED assoc type
impl Drop for Expression;             // unchanged, iterative
impl Serialize for Expression; impl<'de> Deserialize<'de> for Expression;  // CHANGED shape (§5.6)
// REMOVED: From<UnaryExpression|BinaryExpression|PiecewiseExpression|CallExpression> for Expression
```

`PiecewiseExpression` keeps its invariants: at least one case, and no case
condition is a non-Boolean literal. `LogicalExpression` has at least two
operands. Both hold because the constructors are `pub(crate)` and the
fields are private.

#### 3.2 Construction (`build.rs`)

```rust
impl Expression {
    pub fn new_unary(operation: UnaryOperation, operand: impl Into<Expression>) -> Expression;   // CHANGED bound
    pub fn new_binary(operation: BinaryOperation,
                      left: impl Into<Expression>, right: impl Into<Expression>) -> Expression;  // CHANGED bound
    /// 0 operands: the literal `true` (And) / `false` (Or). 1 operand: that
    /// operand's handle, unchanged. 2+: one Logical node over exactly the
    /// given operands, in order. Never flattens (§5.1). Infallible.
    pub fn new_logical<I>(operation: LogicalOperation, operands: I) -> Expression
        where I: IntoIterator, I::Item: Into<Expression>;            // NEW
    pub fn all<I>(operands: I) -> Expression where I: IntoIterator, I::Item: Into<Expression>;  // NEW = new_logical(And, ..); was build_logical_and
    pub fn any<I>(operands: I) -> Expression where I: IntoIterator, I::Item: Into<Expression>;  // NEW = new_logical(Or, ..);  was build_logical_or
    /// # Errors
    /// PiecewiseError::NoCases if `cases` is empty;
    /// PiecewiseError::NonBooleanConditionLiteral { case_index } for the first
    /// case whose condition is a non-Boolean literal.
    pub fn piecewise<C, V, O>(cases: impl IntoIterator<Item = (C, V)>, otherwise: O)
        -> Result<Expression, PiecewiseError>
        where C: Into<Expression>, V: Into<Expression>, O: Into<Expression>;  // NEW; was build_piecewise
    /// Infallible: a FunctionName is non-empty by construction. The argument
    /// count is not checked against a built-in's arity (Non-goals).
    pub fn call<I>(callee: impl Into<Callee>, arguments: I) -> Expression
        where I: IntoIterator, I::Item: Into<Expression>;            // NEW; was build_call
    /// The explicit way to make a literal, including a Boolean one.
    pub fn literal(value: impl Into<LiteralValue>) -> Expression;    // NEW

    pub fn equals(&self, other: impl Into<Expression>) -> Expression;        // CHANGED bound (all below too)
    pub fn not_equals(&self, other: impl Into<Expression>) -> Expression;
    pub fn less(&self, other: impl Into<Expression>) -> Expression;
    pub fn less_equal(&self, other: impl Into<Expression>) -> Expression;
    pub fn greater(&self, other: impl Into<Expression>) -> Expression;
    pub fn greater_equal(&self, other: impl Into<Expression>) -> Expression;
    /// Division rounded toward negative infinity.
    pub fn floor_divide(&self, other: impl Into<Expression>) -> Expression;
    /// Remainder of floor division; its sign follows the divisor.        NEW (decision 11)
    pub fn floor_mod(&self, other: impl Into<Expression>) -> Expression;
    pub fn power(&self, other: impl Into<Expression>) -> Expression;
    pub fn and(&self, other: impl Into<Expression>) -> Expression;           // NEW = all([self, other])
    pub fn or(&self, other: impl Into<Expression>) -> Expression;            // NEW = any([self, other])
    // REMOVED: positive(), logical_not() (use `!expr`)
}

// Conversions (F-024). Each wraps the value in the leaf it denotes.
impl From<i32> for Expression;   impl From<i64> for Expression;   impl From<i128> for Expression;
impl From<u32> for Expression;   impl From<u64> for Expression;   impl From<usize> for Expression;
impl From<BigInt> for Expression; impl From<f64> for Expression;
impl From<Identifier> for Expression; impl From<&Identifier> for Expression;   // &Identifier NEW
impl From<LiteralValue> for Expression; impl From<&Expression> for Expression; // &Expression NEW
// No From<bool>, From<&str> or From<f32>: see the compile_fail doctests in §5.2.

// Operators (F-010: no Rem).
impl<R: Into<Expression>> Add<R> for Expression;  impl<R: Into<Expression>> Add<R> for &Expression;  // + Sub, Mul, Div
impl Add<Expression> for L; impl Add<&Expression> for L;   // + Sub, Mul, Div, for each L in:
    // i32, i64, i128, u32, u64, usize, BigInt, f64, Identifier, &Identifier, LiteralValue
impl Neg for Expression; impl Neg for &Expression;   // UnaryOperation::Negate
impl Not for Expression; impl Not for &Expression;   // NEW: UnaryOperation::LogicalNot
// REMOVED: Rem for everything; IntoOperand (trait and its sealed supertrait)
// REMOVED: free fns build_logical_and, build_logical_or, build_piecewise, build_call
```

#### 3.3 Operations (`operation.rs`)

All three are exhaustive, as S-6 requires for enums that passes match.
They share the same derives and trait implementations:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnaryOperation { Negate, LogicalNot }                    // CHANGED: Positive REMOVED

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BinaryOperation {                                        // CHANGED: 13 variants
    Add, Subtract, Multiply,
    /// True division: `x / 4` is the exact real quotient whatever the operand
    /// types, never truncating integer division (use floor_divide).
    Divide,
    FloorDivide,
    FloorMod,                     // RENAMED from Modulo; wire "floor_mod"
    Power,
    Equal, NotEqual, Less, LessEqual, Greater, GreaterEqual,
    // REMOVED: LogicalAnd, LogicalOr (now LogicalOperation)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogicalOperation { And, Or }                             // NEW

// For each of the three:
impl X {
    pub fn as_str(self) -> &'static str;   // == the serde name: "negate", "floor_mod", "and", ...
    pub fn symbol(self) -> &'static str;   // "-", "!", "+", "//", "%", "**", "&&", "||", ...
}
impl fmt::Display for X;                   // writes as_str()
impl FromStr for X { type Err = UnknownNameError; }  // parses through the serde derive (§5.5)

/// A name no variant of an enum has.                                     NEW
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownNameError { /* name: Box<str>, expected: &'static str */ }
impl UnknownNameError { pub fn name(&self) -> &str; }
impl fmt::Display for UnknownNameError;    // "unknown binary operation `plus`"
impl Error for UnknownNameError;
```

`FunctionSort` (`sort.rs`) and `SymbolType` (`symbol_type.rs`, which moves
from `fhy_core::symbolic::symbol_type::SymbolType` to
`fhy_core::expr::SymbolType`) get the same treatment:

- the derive and `rename_all` attributes above;
- `as_str`, `Display`, and `FromStr<Err = UnknownNameError>`;
- their variants stay unchanged;
- both stay exhaustive, because they are closed classifications that
  passes map one to one. This is a proposed addition to the S-6 exception
  list; see §10.
- `FunctionSort::accepts_literal(self, value: &LiteralValue) -> bool` stays.

#### 3.4 Callees (`callee.rs`)

```rust
/// Exhaustive: a call names a built-in or a user function.            NEW
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Callee { Builtin(BuiltinFunction), Named(FunctionName) }
impl Callee { pub fn name(&self) -> &str; }
impl From<BuiltinFunction> for Callee; impl From<FunctionName> for Callee;
impl FromStr for Callee { type Err = FunctionNameError; }  // built-in name -> Builtin, else Named
impl fmt::Display for Callee;                              // name()

/// A non-empty function name that no built-in function has.           NEW
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionName(Arc<str>);   // private field
impl FunctionName {
    /// # Errors
    /// FunctionNameError::Empty for ""; FunctionNameError::Builtin(f) when
    /// `name` is the name of built-in `f`.
    pub fn try_new(name: &str) -> Result<FunctionName, FunctionNameError>;
    pub fn as_str(&self) -> &str;
}
impl fmt::Display for FunctionName;
impl Serialize for FunctionName;           // as a string
impl<'de> Deserialize<'de> for FunctionName; // validates through try_new

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FunctionNameError { Empty, Builtin(BuiltinFunction) }       // NEW
// Display: "function name is empty" / "function name `max` is a built-in function"
```

#### 3.5 Literals (`literal.rs`)

```rust
/// A normalized constant. Exhaustive: passes match every literal form.
#[derive(Debug, Clone)]
pub enum LiteralValue {                                            // CHANGED: was an opaque struct
    Bool(bool),
    Int(BigInt),
    /// Any f64, NaN and the infinities included.
    Float(f64),
    Decimal(Decimal),
}
impl LiteralValue {
    /// "05" -> Int(5); "1.50", "1.", ".5" -> Decimal. The grammar is
    /// unchanged: ASCII digits with at most one '.', at least one digit.
    /// # Errors
    /// LiteralTextError if `text` is outside the grammar.
    pub fn parse_text(text: &str) -> Result<LiteralValue, LiteralTextError>;   // CHANGED: normalizes
    // REMOVED: from_bool, kind, canonical_key, is_integer_valued
}
impl PartialEq for LiteralValue; impl Eq for LiteralValue; impl Hash for LiteralValue;  // §5.3
impl fmt::Display for LiteralValue;                                              // CHANGED (§5.3)
impl From<bool>; From<i32>; From<i64>; From<i128>; From<u32>; From<u64>; From<usize>;
impl From<BigInt>; From<f64>; From<Decimal>;                   // for LiteralValue; i32/i128/u32/u64/usize/Decimal NEW
impl Serialize for LiteralValue; impl<'de> Deserialize<'de> for LiteralValue;  // derived, §5.6

/// A non-negative exact decimal, normalized.                         NEW (was python_text::NormalizedDecimal, pub(crate))
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Decimal { /* coefficient: BigInt (>= 0, no trailing zero unless 0), exponent: i64 (0 when 0) */ }
impl Decimal {
    pub fn coefficient(&self) -> &BigInt;   // value = coefficient * 10^exponent
    pub fn exponent(&self) -> i64;
}
impl FromStr for Decimal { type Err = LiteralTextError; }  // the literal grammar, any digits
impl fmt::Display for Decimal;                              // positional (§5.3)
impl Serialize for Decimal; impl<'de> Deserialize<'de> for Decimal;  // as its Display string

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiteralTextError { /* text */ }                 // unchanged
impl LiteralTextError { pub fn text(&self) -> &str; }
// REMOVED: LiteralKind<'a>
```

#### 3.6 Errors (`error.rs`, F-014, S-5)

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PiecewiseError {                                          // NEW
    NoCases,                                         // "piecewise has no cases"
    NonBooleanConditionLiteral { case_index: usize },// "condition of piecewise case {case_index} is a non-boolean literal"
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RebuildError {                                            // NEW
    ChildCount { expected: usize, actual: usize },   // "expected {expected} children, got {actual}"
    Piecewise(PiecewiseError),                       // "invalid piecewise"; source() = the PiecewiseError
}

pub struct NonInjectiveRenamingError { /* image */ }               // unchanged
// REMOVED: ExpressionBuildError (all five variants; TooFewLogicalOperands and
// EmptyFunctionName have no successor, since their operations are now infallible)
```

#### 3.7 Alpha renaming (`alpha.rs`)

```rust
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AlphaRenaming { /* private */ }                         // unchanged
impl AlphaRenaming {
    pub fn try_new<S: BuildHasher>(free_renaming: HashMap<Identifier, Identifier, S>)
        -> Result<Self, NonInjectiveRenamingError>;
    pub fn is_corresponding(&self, left: &Identifier, right: &Identifier) -> bool;  // RENAMED from are_identifiers_alpha_equivalent (S-7)
    pub fn is_empty(&self) -> bool;
}
```

#### 3.8 Display and Debug (`display.rs`, F-015)

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)] #[non_exhaustive]
pub enum Notation { #[default] Symbolic, Functional }             // CHANGED: + non_exhaustive
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)] #[non_exhaustive]
pub enum IdentifierStyle { #[default] NameHint, NameHintWithId }   // CHANGED: + non_exhaustive
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)] #[non_exhaustive]
pub struct FormatOptions { /* private */ }                         // unchanged API:
    // with_notation, with_identifier_style, notation(), identifier_style()

/// What Expression::display returns; implements Display.                NEW
#[derive(Debug, Clone, Copy)]
pub struct ExpressionDisplay<'a> { /* expression: &'a Expression, options: FormatOptions */ }
impl fmt::Display for ExpressionDisplay<'_>;
// REMOVED: format_expression (use expr.display(opts).to_string())
```

`fhy_core::expr::passes::ExpressionPrettyFormatter` moves here from
`pprint.rs`. Its API is unchanged: `new(FormatOptions)`, `options()`, and
`CompilerPass<Expression, String>`. Its `run` returns
`ir.display(self.options).to_string()`. Its default `name()` follows B5's
pass-naming decision (F-006).

#### 3.9 The Boolean-position screen (`screen.rs`, F-023, F-014)

```rust
/// Bindings the screen applies before judging an identifier.           NEW
pub trait Environment { fn binding(&self, identifier: &Identifier) -> Option<&Expression>; }
impl<S: BuildHasher> Environment for HashMap<Identifier, Expression, S>;

/// Declared value kinds of identifiers left free.                       NEW
pub trait SymbolTypes { fn symbol_type(&self, identifier: &Identifier) -> Option<SymbolType>; }
impl<S: BuildHasher> SymbolTypes for HashMap<Identifier, SymbolType, S>;
impl<F: Fn(&Identifier) -> Option<SymbolType>> SymbolTypes for F;   // coherence checked with rustc 1.98

pub trait SortLookup {                                              // CHANGED
    fn native_constant_sort(&self, identifier: &Identifier) -> Option<FunctionSort> { None }  // default NEW
    /// Only called for Callee::Named; built-ins' sorts come from the catalogue.
    fn call_result_sort(&self, name: &FunctionName) -> Option<FunctionSort> { None }         // CHANGED param type + default
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct NoRegisteredSorts;                                       // unchanged
impl SortLookup for NoRegisteredSorts {}

/// Refuses a number in a Boolean position.                              NEW (replaces the two free fns)
#[derive(Clone, Copy)]
pub struct BooleanScreen<'a> { /* sorts: &'a dyn SortLookup, environment: &'a dyn Environment,
                                  symbol_types: &'a dyn SymbolTypes; all default to "knows nothing" */ }
impl<'a> BooleanScreen<'a> {
    pub fn new() -> Self;
    pub fn with_sorts(self, sorts: &'a dyn SortLookup) -> Self;
    pub fn with_environment(self, environment: &'a dyn Environment) -> Self;
    pub fn with_symbol_types(self, symbol_types: &'a dyn SymbolTypes) -> Self;
    /// Was validate_logical_operands.
    /// # Errors
    /// NonBooleanLogicalOperandError for the first offending Boolean position in walk order.
    pub fn check_logical_operands(&self, expression: &Expression) -> Result<(), NonBooleanLogicalOperandError>;
    /// Was validate_predicate. Same, with the root itself in a Boolean position.
    pub fn check_predicate(&self, expression: &Expression) -> Result<(), NonBooleanLogicalOperandError>;
}
impl Default for BooleanScreen<'_>; impl fmt::Debug for BooleanScreen<'_>;  // Debug writes the type name only

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BooleanPosition {                                          // CHANGED
    NegatedOperand,
    LogicalOperand { operation: LogicalOperation, operand_index: usize },  // CHANGED fields
    CaseCondition { case_index: usize },
    CaseValue { case_index: usize },
    Otherwise,
    // REMOVED: PredicateRoot (now "no parent", see below)
}
impl fmt::Display for BooleanPosition;                               // NEW, §5.7

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NonBooleanLogicalOperandError { /* operand: Expression, parent: Option<(Expression, BooleanPosition)> */ }
impl NonBooleanLogicalOperandError {
    pub fn operand(&self) -> &Expression;
    /// The node that puts the operand in a Boolean position, and where; None
    /// exactly when the operand is the root of a predicate.
    pub fn parent(&self) -> Option<(&Expression, BooleanPosition)>;  // CHANGED: ties parent and position
    // REMOVED: position()
}
impl fmt::Display for NonBooleanLogicalOperandError;                 // CHANGED: short, bounded (§5.7)
impl Error for NonBooleanLogicalOperandError;
// REMOVED: validate_logical_operands, validate_predicate
```

#### 3.10 Built-ins (`fhy_core::expr::builtins`, F-024, F-029)

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum BuiltinFunction {                                          // NEW
    // composed, catalogue order
    Max, Min, Abs, Sign, Clamp, ClampSymmetric, Relu, LeakyRelu,
    Xor, Nand, Nor, Implies, Iff, Sigmoid, Silu, Gelu,
    // native, catalogue order
    Exp, Exp2, Log, Log2, Log10, Sqrt, Sin, Cos, Tan, Arcsin, Arccos, Arctan,
    Sinh, Cosh, Tanh, Erf, Round, Floor, Ceil,
}
impl BuiltinFunction {
    /// All 35, composed then native, in catalogue order.
    pub fn iter() -> impl ExactSizeIterator<Item = BuiltinFunction> + Clone;
    pub fn name(self) -> &'static str;                       // == serde name
    pub fn parameter_sorts(self) -> &'static [FunctionSort];
    pub fn result_sort(self) -> FunctionSort;
    /// The definition of a composed built-in; None for a native one.
    pub fn composed(self) -> Option<&'static ComposedFunction>;
}
impl fmt::Display for BuiltinFunction; impl FromStr for BuiltinFunction { type Err = UnknownNameError; }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum BuiltinConstant { Pi, E, Inf, Nan }                        // NEW (replaces NativeConstantSpec)
impl BuiltinConstant {
    pub fn iter() -> impl ExactSizeIterator<Item = BuiltinConstant> + Clone;
    pub fn name(self) -> &'static str;
    pub fn sort(self) -> FunctionSort;     // Real for all four
    pub fn value(self) -> f64;             // Nan is the bits 0x7ff8_0000_0000_0000
}
impl fmt::Display for BuiltinConstant; impl FromStr for BuiltinConstant { type Err = UnknownNameError; }

#[derive(Debug)]
pub struct ComposedFunction { /* private */ }
impl ComposedFunction {
    pub fn function(&self) -> BuiltinFunction;          // NEW (replaces name())
    pub fn parameters(&self) -> &[Identifier];
    pub fn body(&self) -> &Expression;
    // REMOVED: name, parameter_sorts, result_sort (use function().…)
}
// REMOVED: NativeFunctionSignature, NativeConstantSpec,
//          list_composed_functions, list_native_functions, list_native_constants,
//          find_composed_function, find_native_function, find_native_constant,
//          pub(crate) initialize_composed_functions
```

Lookup by name becomes `name.parse::<BuiltinFunction>()?.composed()`.
Listing the composed functions becomes
`BuiltinFunction::iter().filter_map(BuiltinFunction::composed)`.

### 4. Interface delta

All changes are breaking unless marked otherwise. S-8 says the crate is
unpublished, so breaking changes are allowed. Call sites were found by
grepping `src/` and `tests/` at HEAD. `fhy-core-py` has no call sites: it
exposes only the two identifier-counter functions.

| Change | Item | Before (HEAD) | After | Semver | Call sites |
|---|---|---|---|---|---|
| move | module | `fhy_core::symbolic::expression::*`, `fhy_core::symbolic::symbol_type::SymbolType` | `fhy_core::expr::*`, `fhy_core::expr::SymbolType` | breaking | every symbolic test file; `src/shipped.rs:12,44-45` (deleted by B1); `src/identifier.rs:714` (B1 test); `src/pass_infrastructure/tree.rs:424,519` (B5 doc examples) |
| change | `ExpressionKind` | 6 variants | + `Logical(LogicalExpression)` | breaking (exhaustive) | every `match expr.kind()`: `pattern/core.rs` (B4), `tests/common/expression.rs`, `pprint_properties.rs::count_inner_nodes`, the tree and pass tests |
| remove | `BinaryOperation::{LogicalAnd, LogicalOr}` | 15 variants | 13 variants; `LogicalOperation::{And, Or}` | breaking | builders, screen, pprint, vocabulary tests, `tests/common/expression.rs`, builtins_stories, pattern_stories (B4) |
| rename | `BinaryOperation::Modulo` | `Modulo`, wire `"modulo"` | `FloorMod`, wire `"floor_mod"` | breaking | builders, pprint, vocabulary tests, pattern_properties (B4), `tests/common/expression.rs` |
| remove | `UnaryOperation::Positive`, `Expression::positive` | present | removed | breaking | builders, node, pprint, vocabulary tests, pattern_stories, pattern_properties (B4), `tests/common/expression.rs` |
| remove | `Rem` impls | `impl<R: IntoOperand> Rem<R> for Expression` and more | none; `Expression::floor_mod` | breaking | `expression_node_stories.rs:292`, `expression_builders_stories.rs:66,252-254,335,339` |
| add | `Expression::floor_mod` | none | `fn floor_mod(&self, impl Into<Expression>) -> Expression` | non-breaking | none |
| remove | `IntoOperand` | `pub trait IntoOperand: sealed::Sealed {}` | `Into<Expression>` | breaking | `expression_builders_stories.rs`, `tests/common/expression.rs:14,66,80-83` |
| change | `new_unary`, `new_binary`, comparison and arithmetic methods | `impl IntoOperand` | `impl Into<Expression>` | breaking for generic callers only | tests only |
| add | `From<i32/i64/i128/u32/u64/usize/BigInt/f64/&Identifier/&Expression> for Expression` | only `From<Identifier>`, `From<LiteralValue>` | added | non-breaking | none |
| remove | `From<UnaryExpression/BinaryExpression/PiecewiseExpression/CallExpression> for Expression` | present | removed | breaking | `expression_node_stories.rs::expression_from_node_struct_rewraps_the_node` |
| replace | `build_logical_and`, `build_logical_or` | `fn build_logical_and<I>(I) -> Result<Expression, ExpressionBuildError>` (right-fold, ≥2) | `Expression::all(I) -> Expression`, `Expression::any`, `Expression::new_logical`, `Expression::and/or` | breaking | builders, pprint, screen tests; `tests/common/expression.rs::build_deep_conjunction`; builtins.rs xor/nand/nor/implies bodies |
| replace | `build_piecewise` | free fn, `Result<_, ExpressionBuildError>` | `Expression::piecewise`, `Result<_, PiecewiseError>` | breaking | builders, node, pass, pattern, screen, builtins, pprint, tree, wire and properties tests; `tests/common/{expression,pattern}.rs`; `pattern_rewrite_stories.rs` (B4) |
| replace | `build_call` | `fn build_call<I>(&str, I) -> Result<_, ExpressionBuildError>` | `Expression::call(impl Into<Callee>, I) -> Expression` | breaking | builders, tree, pattern_rewrite, pattern_properties, pass, screen, pprint, pattern, builtins, node and wire tests; `tests/common/expression.rs` |
| narrow | `PiecewiseExpression::try_new`, `CallExpression::try_new` | `pub` | `pub(crate)` | breaking | `expression_node_stories.rs`, `builtins_stories.rs`, `tests/common/expression.rs:97-113` |
| replace | `CallExpression::function_name` | `fn function_name(&self) -> &str` | `fn callee(&self) -> &Callee` | breaking | `pattern/core.rs:374-382` (B4); node, builders, tree, pattern_properties, builtins, pattern, screen, properties and wire tests |
| add | `Callee`, `FunctionName`, `FunctionNameError` | none | §3.4 | non-breaking | none |
| add | `BuiltinFunction`, `BuiltinConstant` | none | §3.10 | non-breaking | none |
| remove | `NativeFunctionSignature`, `NativeConstantSpec`, `list_*`, `find_*` | §3.10 before | enum methods | breaking | `builtins_stories.rs`, `builtins_scope_stories.rs`, `tests/common/expression.rs:210-222`, `src/shipped.rs:44` |
| change | `ComposedFunction::name` | `fn name(&self) -> &'static str` | `fn function(&self) -> BuiltinFunction` | breaking | builtins_stories, `tests/common/expression.rs` |
| remove | `initialize_composed_functions` | `pub(crate)` | removed | none | `src/shipped.rs:12,25` (file deleted by B1) |
| remove | `ExpressionBuildError` | 5-variant enum | `PiecewiseError`, `RebuildError` | breaking | `pattern/core.rs:24,626`, `pattern/rewrite.rs:16,43,114,401` (B4); builders, node, pattern_rewrite, tree, properties, pass, pattern and wire tests; `tests/common/tree_ir.rs` has its own `ToyRebuildError` and is not affected |
| change | `rebuild_with_children`, `Tree::RebuildError` | `ExpressionBuildError` | `RebuildError` | breaking | tree, properties and node tests; `tests/common/expression.rs::copy_deeply` |
| change | `substitute` error | `ExpressionBuildError` | `PiecewiseError` | breaking | node, properties and builtins tests |
| change | `LiteralValue` | opaque struct; `from_bool`, `kind`, `canonical_key`, `is_integer_valued` | public enum `Bool/Int/Float/Decimal`; those four methods removed | breaking | literal, properties, builders, wire, builtins and pattern_properties tests; `tests/common/expression.rs`; `pattern/core.rs:170-183` (B4, literal pattern semantics) |
| remove | `LiteralKind` | `enum LiteralKind<'a>` | match on `LiteralValue` | breaking | same files as above |
| change | `LiteralValue::parse_text` | keeps the spelling | normalizes | behavioral | builders, pattern_properties, literal, pprint, properties and pattern tests; `tests/common/expression.rs`; `pattern/core.rs` |
| add | `Decimal` | `pub(crate) NormalizedDecimal` in `python_text` | `pub struct Decimal` | non-breaking | none |
| change | `LiteralValue: Display` | Python text (`True`, `1e+16`, verbatim text) | Rust text (§5.3) | behavioral | pprint, literal, builtins, screen and pprint_properties tests |
| remove | `format_expression` | `fn(&Expression, FormatOptions) -> String` | `expr.display(opts)` | breaking | pprint_properties, pprint, pass and builtins tests |
| add | `Expression::display`, `ExpressionDisplay`, `impl Display for Expression` | none | §3.8 | non-breaking | none |
| change | `Expression: Debug` | derived, recursive | hand-written, bounded | behavioral | failure arms only |
| change | `Notation`, `IdentifierStyle` | exhaustive | `#[non_exhaustive]` | breaking | none outside the crate |
| move | `ExpressionPrettyFormatter` | `symbolic::expression` | `expr::passes` | breaking | `expression_pass_stories.rs` |
| replace | `validate_logical_operands`, `validate_predicate` | 4-parameter free fns taking `&HashMap`s | `BooleanScreen` builder | breaking | `expression_screen_stories.rs`, `expression_properties.rs` |
| change | `SortLookup::call_result_sort` | `(&self, &str)` | `(&self, &FunctionName)`, with defaults | breaking | screen stories `BuiltinSorts`; properties `:132-136` |
| change | `BooleanPosition` | `LogicalOperand { operation: BinaryOperation }`, `PredicateRoot` | `LogicalOperand { operation: LogicalOperation, operand_index }`; no `PredicateRoot` | breaking | screen stories |
| change | `NonBooleanLogicalOperandError::{parent, position}` | `parent() -> Option<&Expression>`, `position()` | `parent() -> Option<(&Expression, BooleanPosition)>` | breaking | screen stories |
| rename | `AlphaRenaming::are_identifiers_alpha_equivalent` | as named | `is_corresponding` | breaking | `node.rs` (internal); `expression_node_stories.rs` |
| change | enum serde and text | `impl_wire_name_traits!` + `ALL_*` arrays | derive + `FromStr` | behavioral (serde error text) | vocabulary_stories |
| add | `FromStr` for the operation, sort, symbol-type and built-in enums; `UnknownNameError` | none | §3.3 | non-breaking | none |
| change | `Expression` serde shape | recursive `{"__type__","__data__"}` envelope; big ints as JSON integers | node table (§5.6); big ints as decimal strings | breaking wire | wire, properties and pprint_properties tests; `src/identifier.rs:714` IdentifierPath::Expression (B1) |
| remove | `python_text.rs` | 884 lines, `pub(crate)` | deleted; the decimal normalizer moves into `literal.rs` | none | `literal.rs` only |
| remove | `wire_name.rs`, `ALL_*` const arrays in `src` | `pub(crate)` | deleted | none | internal |

### 5. Behavior

#### 5.1 Logical nodes (F-003)

- **Builders.** `Expression::all(ops)`, `Expression::any(ops)` and
  `Expression::new_logical(op, ops)` behave as follows:
  - **0 operands:** `all` returns the literal `true` and `any` returns
    `false`, each as a `Literal` node.
  - **1 operand:** returns that operand's handle, and
    `Expression::ptr_eq(&all([x.clone()]), &x)` holds. The Python consumer
    (sympy's `And(x)`, which is `x`) has the same semantics.
  - **2 or more operands:** returns one `Logical` node whose `operands()`
    are the inputs, in order.
- **No flattening.** `x.and(y.and(z))` is `Logical(And, [x, Logical(And,
  [y, z])])`, not a three-operand node. The reason: if a builder spliced a
  same-operation operand's children into the new node, it would unshare
  DAGs.
  - Take `d_{k+1} = d_k.and(&d_k)`, which has 65 distinct nodes at 64
    levels. With splicing it would become a single node with 2^64
    operands.
  - The screen's existing DAG tests build exactly this shape
    (`expression_screen_stories.rs::build_doubling_conjunction`).
  - Splicing only unshared operands would make the result's structure
    depend on reference counts, which is worse.
  - The depth problem F-003 reports came from right-folding one
    *iterator* into binary nodes. `all(iter)` makes one node, which solves
    it. The docs on `all` say: "build a large conjunction with one `all`
    call over an iterator; `acc = acc.and(c)` in a loop nests one level per
    step."
- **Rebuild.** `rebuild_with_children` on a `Logical` node needs exactly
  `operands().len()` children, so it can never produce a node with fewer
  than two operands. It keeps the child list exactly as given and does not
  flatten.
- **Equality and hashing are structural.** `all([a, b, c])` and
  `a.and(b.and(c))` are unequal.
- **Printing:**
  - symbolic: `(a && b && c)`, operands separated by ` && ` or ` || `;
  - functional: `(and a b c)` or `(or a b c)`.
- **Screen.** Every operand of a `Logical` node is a Boolean position,
  `BooleanPosition::LogicalOperand { operation, operand_index }`. Operands
  are checked left to right.

#### 5.2 Construction and operators (F-010, F-024)

- **Arithmetic operators.**
  - `+ - * /` build `Add`, `Subtract`, `Multiply` and `Divide`.
  - `/` is true division for every operand type (decision 11). The `Div`
    impls and `BinaryOperation::Divide` document that `x / 4` with integer
    operands is the exact real quotient, and point to `floor_divide`.
  - There is no `%`. A compile-fail doctest on `Expression` pins this:
    ```rust
    let r = Expression::from(Identifier::new("x")) % 3;
    ```
    fails with `E0369`.
- **Unary operators.** Unary `-` builds `Negate`, and unary `!` builds
  `LogicalNot`. Both work on owned and borrowed expressions: `!&p`.
- **`floor_mod(d)`** builds `FloorMod`. Its doc gives the semantics: the
  remainder of floor division, whose sign follows the divisor, so
  `-7 floor_mod 3 = 2` and `7 floor_mod -3 = -2`.
- **`Positive` is gone.** There is no Rust operator for it, and it is the
  identity. When the Python binding exists, it maps `__pos__` to the
  operand itself (§10).
- **Conversions from primitives.**
  - `Expression::from(n)` for `i32`, `i64`, `i128`, `u32`, `u64`, `usize`
    and `BigInt` gives `Literal(Int(n))`.
  - `f64` gives `Literal(Float(v))`, for any `v`.
  - An unsuffixed integer literal falls back to `i32` and an unsuffixed
    float to `f64`, so `&x + 1` and `x.less(2.5)` compile.
- **What does not convert.** `bool`, `&str` and `f32` have no `From` impl.
  Two existing compile-fail doctests keep this pinned, moved to
  `Expression`'s docs:
  - `Expression::new_unary(UnaryOperation::LogicalNot, true)` fails with
    `E0277`;
  - `Expression::new_unary(UnaryOperation::Negate, "1.5")` fails with
    `E0277`.
- **Explicit literals.** Write a Boolean literal as `Expression::literal(true)`.
- **Calls.** `Expression::call(BuiltinFunction::Max, [&a, &b])` builds a
  call. `Expression::call("f".parse::<Callee>()?, args)` builds a call of a
  named function.

#### 5.3 Literals (F-013, F-026, decision 6)

- **Normalization happens on parse.** No spelling is kept.
  - `parse_text("05")` gives `Int(5)`.
  - `parse_text("000")` gives `Int(0)`.
  - `parse_text("1.50")` gives the `Decimal` with coefficient 15 and
    exponent -1.
  - `parse_text("100.0")` gives the `Decimal` with coefficient 1 and
    exponent 2.
  - `parse_text("0.0")` gives the `Decimal` with coefficient 0 and exponent
    0.
  - `parse_text(".5")` gives the `Decimal` with coefficient 5 and exponent
    -1.
  - The grammar and `LiteralTextError` are unchanged.
- **Equality** is a lawful equivalence, and `Hash` agrees with it:
  - values of different variants are unequal: `Int(1) != Float(1.0) !=
    Decimal(1) != Bool(true)`;
  - `Int` and `Decimal` compare by value (for `Decimal`, the normalized
    fields);
  - `Float` compares by `==`, except that every NaN equals every NaN.
    `-0.0 == 0.0`;
  - the float hash writes nothing for NaN and `(v + 0.0).to_bits()`
    otherwise.
- **`Display` (S-4, Rust conventions):**
  - `Bool` writes `true` or `false`.
  - `Int` writes decimal digits, with a leading `-` when negative.
  - `Float` writes exactly what `format!("{}", v)` writes for an f64:
    `1.5`, `1` for `1.0`, `10000000000000000` for `1e16`, `NaN`, `inf`,
    `-inf`, and `-0` for `-0.0`.
  - `Decimal` writes the normalized value positionally, with no exponent:
    `1.5`, `100`, `0.001`, `0`, and `0.5` for `.5`. Its length is at most
    the parsed text's length plus one, so the text stays bounded by the
    input that produced it.
  - Consequence: `Int(1)`, `Float(1.0)` and `Decimal(1)` all print as `1`.
    The printer already says distinct trees may print alike.
- **`canonical_key` is removed.** It had no non-test caller at HEAD (grep:
  only `expression_literal_stories.rs` and `expression_properties.rs`).
  Equality now follows from the normalized representation, so there is
  nothing left for a key to do.
- **Pattern literals (B4).** "Exactly the stored form"
  (`pattern/core.rs:170-183`) collapses to "equal `LiteralValue`", except
  that B4 decides whether NaN matches.

#### 5.4 Display and Debug (F-003, F-015)

- **`display`.** `expr.display(opts)` returns an `ExpressionDisplay`.
  - Its `Display` writes straight into the `Formatter`. It allocates no
    intermediate `String` per node, literal or identifier.
  - An identifier is written as `write!(f, "{}::{}", name_hint, id)` or as
    the bare name hint, depending on the identifier style.
  - It uses an explicit work stack, as `format_expression` does today, so
    it never overflows the call stack.
  - The notation tables of today's `format_expression` doc carry over
    unchanged, plus the Logical rows from §5.1.
  - A negative literal operand stays bare. For example, `(-1 ** 2)` is the
    literal `-1` raised to 2, and `((-1) ** 2)` is a negation raised to 2.
    This is now Rust's own contract, not a Python spelling.
- **`impl Display for Expression`** is the same as
  `self.display(FormatOptions::default())`.
  - This is the full text. For a DAG the text is exponential in the depth,
    because the printer writes a shared subtree at every occurrence.
  - The doc says so, and points to `Debug` for diagnostics.
- **`impl Debug for Expression`** is hand-written:
  - it writes `Expression(` + the functional notation with
    `NameHintWithId` + `)`;
  - it is iterative;
  - it is bounded: after 1,000 printed nodes, every remaining unprinted
    subtree is written as `..` without being scheduled, with the open
    parentheses still closed;
  - so output and work are O(1,000 × max fan-out), whatever the depth or
    sharing;
  - `{:#?}` behaves the same way.
  - The output is not a stable contract. Tests do not pin it; they only
    check that it terminates and stays short.
- **Consequences:**
  - The derived `Debug` on `ExpressionKind`, the node structs,
    `NonBooleanLogicalOperandError` and `RebuildError` stays derived. It
    reaches expressions only through the bounded impl, so none of these can
    overflow the stack or hang.
  - `assert_eq!` on expressions becomes safe, including on 2^64-occurrence
    DAGs.

#### 5.5 Enum text and serde (F-025)

- **Serde:** `UnaryOperation`, `BinaryOperation`, `LogicalOperation`,
  `FunctionSort`, `SymbolType`, `BuiltinFunction` and `BuiltinConstant` use
  `#[derive(Serialize, Deserialize)]` with `rename_all = "snake_case"`.
  - `Exp2` becomes `"exp2"`, `Log10` becomes `"log10"`, and
    `ClampSymmetric` becomes `"clamp_symmetric"`.
  - The derive decodes a unit variant from a borrowed `&str` without
    allocating (this also covers the `wire_name.rs:24` part of F-039).
  - Casing is exact: `"Add"` and `"+"` are refused.
- **Display** writes `as_str()`, which is an exhaustive `match`.
- **`FromStr`** is `X::deserialize(s.into_deserializer())`, with serde's
  error mapped to `UnknownNameError { name: s, expected: "binary
  operation" }`. So parsing and deserializing share one source of truth,
  the derive.
- **The remaining risk** is that `as_str` diverges from the derive. The
  test in §8.2 (F-025) closes it by iterating a variant list whose
  completeness is enforced by an exhaustive `match` in the test.
- **`ALL_*` arrays in `src` are deleted.** Only `BuiltinFunction::iter`
  and `BuiltinConstant::iter` keep a private catalogue-order array. A unit
  test in `builtins.rs` maps every variant, by exhaustive `match`, to its
  index in that array and asserts a round trip, so adding a variant without
  listing it fails to compile.

#### 5.6 Wire format (S-2, F-003, F-012)

- **Shape.** Plain derives over private shapes, with no envelope:
  ```rust
  #[derive(Serialize, Deserialize)] #[serde(deny_unknown_fields)]
  struct ExpressionWire { nodes: Vec<WireNode> }            // post-order; root is the last node
  #[derive(Serialize, Deserialize)] #[serde(rename_all = "snake_case", deny_unknown_fields)]
  enum WireNode {
      Unary { operation: UnaryOperation, operand: u64 },
      Binary { operation: BinaryOperation, left: u64, right: u64 },
      Logical { operation: LogicalOperation, operands: Vec<u64> },
      Identifier(Identifier),                                // Identifier's own serde (B1)
      Literal(LiteralValue),
      Piecewise { cases: Vec<(u64, u64)>, otherwise: u64 },  // (condition, value)
      Call { callee: Callee, arguments: Vec<u64> },
  }
  // LiteralValue: derived, externally tagged, snake_case:
  //   {"bool":true} | {"int":"-12"} | {"float":1.5} | {"decimal":"1.5"}
  // Int goes through #[serde(with = "big_int_decimal")]; Float through a
  // with-module that refuses non-finite values; Decimal as its Display string.
  // Callee: {"builtin":"max"} | {"named":"f"}
  ```
  `Serialize`/`Deserialize` for `Expression` are thin manual impls that
  convert to and from `ExpressionWire`. Serialization uses a borrowed twin
  of the shape, so it clones nothing.
- **Example.** `x + 1`, with `x` having id 41, encodes as:
  ```json
  {"nodes":[{"identifier":{"id":41,"name_hint":"x"}},{"literal":{"int":"1"}},
            {"binary":{"operation":"add","left":0,"right":1}}]}
  ```
- **Encoding:**
  - Each distinct node, by identity, is written once, in post-order of
    first visit, from an explicit stack.
  - Only nodes that `is_shared()` go into the identity map. A node with a
    single handle can only be reached once.
  - So encode is linear in *distinct* nodes. Today a 64-level doubling DAG
    serializes 2^65 nodes; under this format it serializes 65.
  - Decoding rebuilds exactly the same sharing.
- **Nesting.** The serde nesting depth is constant, at most 5 JSON levels,
  whatever the tree depth. So `serde_json::from_str` decodes a tree of any
  depth, the 62-level limit disappears, and neither encode nor decode
  recurses per tree level.
- **Checks during decode.** Nodes are built in order. A node refers only to
  earlier indices. Decode fails with `de::Error::custom` carrying the crate's
  own one-line message when any of these holds:
  - the node list is empty: "expression payload has no nodes";
  - a child index is not below the node's own index: "node {i} refers to
    node {j}, which does not precede it";
  - a logical node has fewer than two operands: "logical node {i} has
    {n} operands, expected at least 2";
  - a piecewise node has no cases: "piecewise node {i} has no cases";
  - a case condition is a non-Boolean literal: "condition of case {c} of
    piecewise node {i} is a non-boolean literal";
  - a non-root node is not referenced: "node {i} is not referenced";
  - an integer string does not match `-?(0|[1-9][0-9]*)`, or is `"-0"`:
    "invalid integer literal {text:?}";
  - a decimal string is outside the grammar: `LiteralTextError`'s text;
  - a float is non-finite: "float literal {v} is not finite";
  - a function name is invalid: `FunctionNameError`'s text.

  Unknown variants and fields are serde's own errors. Tests classify those
  and do not pin their text.
- **Serialization errors.** A non-finite float literal fails to serialize
  with "float literal {v} is not finite", in every format. This keeps
  today's contract, and it keeps serde_json from writing `null`.
- **Side effects (S-2).** Deserializing each `Identifier` advances the id
  counter as it goes, so a payload refused later may already have advanced
  it. The `Deserialize` doc says so. The "refused payload restores no
  identifier" guarantee (`wire.rs:1-6, 194-201`) is dropped.
- **Formats.** No `serde_json` type appears in `expr`, and serde's
  `deserialize_any` is never called. A postcard or bincode round trip
  works: the wire shape is a struct holding a sequence of externally tagged
  enums with typed fields. That still depends on B1's `Identifier` serde
  being format-generic.
- **No reader for the old format** (decision 4).

#### 5.7 The Boolean-position screen (F-023, F-014)

- **Builder defaults.** A bare `BooleanScreen::new()` knows no sorts, no
  bindings and no symbol types. It behaves like today's call with empty maps
  and `&NoRegisteredSorts`.
- **What counts as "provably numeric".** The rules match today's, with two
  changes:
  - `Callee::Builtin(f)` is judged by `f.result_sort()` without consulting
    `SortLookup`. So `floor(1.5) && true` is refused even under
    `NoRegisteredSorts`. **This is a behavior change**: at HEAD an
    unregistered built-in passes.
  - `Callee::Named(n)` is judged by `sorts.call_result_sort(&n)`.
- **Linear time, including nested piecewise.** "Provably numeric" is
  memoized once per screen run, not once per call:
  - The key is `(NodeIdentity, is_bound_here)`, over every node whose
    answer needs a walk: piecewise nodes and bound identifiers.
  - The answer is computed bottom-up from an explicit stack.
  - Each distinct node is therefore judged at most twice (once per
    `is_bound_here` value), and a run costs O(distinct nodes).
  - This fixes the O(d²) behavior noted in §0.
- **Walk order is unchanged.** The walk is depth-first and pre-order:
  - at a node, all case conditions are checked first, then case values,
    then the otherwise branch;
  - connective operands are checked left to right.
- **Error shape.** `parent()` returns `None` exactly for a predicate root.
  Otherwise it returns the parent node and the position of the operand
  within it.
- **`Display`** is one bounded line that never contains an expression:
  - `"{position} provably denotes a number but sits in a boolean
    position"`;
  - or `"the predicate provably denotes a number"` for a predicate root.
- **`BooleanPosition`'s `Display`:**
  - "the operand of a logical not";
  - "operand {i} of a logical {and|or}";
  - "the condition of piecewise case {i}";
  - "the value of piecewise case {i}";
  - "the otherwise branch of a piecewise".
- **Operand text is the caller's choice.** A caller who wants the
  expressions in a message prints `error.operand().display(opts)` and the
  parent. The diagnostic layer and the binding do this.
- **Removed behavior.** Today's message embeds both expressions in full.
  For a DAG that message is unbounded, and it is hazardous inside
  `Display`.

#### 5.8 Built-ins (F-024, F-029)

- **The catalogue content is unchanged:**
  - the same 16 composed and 19 native functions;
  - the same 4 constants;
  - the same signatures, bodies and values.
- **Bodies now use the new node types:**
  - `xor` is `a.or(b).and(!a.and(b))`, and `nand`, `nor` and `implies` use
    the `Logical` node and `!`;
  - every call in a body is `Expression::call(BuiltinFunction::…, …)`.
- **Composed-function parameters** are still minted lazily by `LazyLock`
  on first use.
  - S-3 caps payload ids at 2^63, so the counter always has at least 2^63
    ids of headroom, and lazy minting cannot exhaust it. The F-007 hazard
    that `shipped.rs` guarded against is therefore gone.
  - `initialize_composed_functions` is deleted along with `shipped.rs`
    (B1).
  - The parameters are *not* in S-3's reserved id block, because they are
    not built-in tags (§10).

#### 5.9 Python

No Python change in this batch. Python code in `src/fhy_core` (the sympy,
z3 and numpy passes, the solver, constraints, and the type checker) has its
own expression implementation. It will consume Rust expressions only
through a future binding. Today `fhy_core._rs` exposes only
`allocate_identifier_id` and `advance_identifier_counter_past`. That future
binding will need to:

- map `UnaryOperation.POSITIVE` / `__pos__` to the operand
  (`src/fhy_core/symbolic/expression/core.py:437,652`; used by
  `passes/{numpy,z3,sympy}.py`, `solver.py:981` and
  `types/checking/type_checker.py:662`);
- map binary `LOGICAL_AND`/`LOGICAL_OR` chains to n-ary nodes;
- add the `__type__`/`__data__` envelope (S-2).

### 6. Error and panic model

| Operation | Result |
|---|---|
| `Expression::piecewise` | `PiecewiseError::{NoCases, NonBooleanConditionLiteral}` |
| `Expression::{all, any, new_logical, call, new_unary, new_binary, literal}`, operators, `From` | infallible, never panic |
| `FunctionName::try_new`, `Callee::from_str` | `FunctionNameError::{Empty, Builtin}` |
| `rebuild_with_children` / `Tree::rebuild_with_children` | `RebuildError::{ChildCount, Piecewise(_)}` |
| `substitute` | `PiecewiseError::NonBooleanConditionLiteral` only |
| `LiteralValue::parse_text`, `Decimal::from_str` | `LiteralTextError` |
| `FromStr` of the enums | `UnknownNameError` |
| `AlphaRenaming::try_new` | `NonInjectiveRenamingError` (unchanged) |
| `BooleanScreen::check_*` | `NonBooleanLogicalOperandError` |
| `Serialize for Expression` | serializer error for a non-finite float |
| `Deserialize for Expression` | the §5.6 checks, via `de::Error::custom` |

- **Display and `source()` (S-5).** Every `Display` is one lowercase line
  and never repeats `source()`. `RebuildError::Piecewise`'s `source()` is
  the `PiecewiseError`. Every other error here has no source.
- **Panics.**
  - The only reachable panic is the catalogue's own `expect` while building
    composed bodies. It is unreachable by construction, and a unit test
    covers every body.
  - `substitute` produces `PiecewiseError` through an internal rebuild
    whose child count is correct by construction. It maps no error to a
    panic.
- **Stack.** No operation recurses per tree level, including `Debug`,
  `Display`, serde and the screen. Drop is already iterative.

### 7. Non-goals

- **Hash caching** (F-002's suggested cached per-node hash). Hashing is
  linear in distinct nodes per call, which is correct. Caching is a perf
  follow-up.
- **Flattening nested logical nodes.** No builder or normalizing pass does
  it (§5.1). A `flatten_logical` rewrite could be a later pattern or pass.
- **Built-in arity in calls.** `Expression::call` does not check a
  built-in's arity.
- **`BitAnd`/`BitOr` operators for `and`/`or`.** These are not added,
  because they would read as bitwise.
- **Unifying `SymbolType` with `FunctionSort`.** The Python code keeps them
  distinct on purpose (`symbol_type.py` docstring).
- **Symbolic golden corpus from Python (F-036, symbolic part).** Under
  decision 3 the symbolic area is not dual-defined, so there is no Python
  oracle to record. See §10.
- **Pattern changes (B4):**
  - a logical pattern;
  - `Callee` matching;
  - literal-pattern semantics;
  - `find_refused_child_index` over `RebuildError::Piecewise`;
  - `RewriteRuleApplier` placement.
- **Pass naming** (F-006): B5.
- **F-034 test-binary consolidation.** Handled by whichever batch owns it.
  The lists below are keyed by today's files.

### 8. Test plan

#### 8.1 Existing tests: keep, modify, delete

Every file also gets the mechanical updates: the `fhy_core::expr` path,
`Expression::piecewise/call/all`, `impl Into<Expression>` helpers, and `!`
in place of `.logical_not()`. These are not repeated below.
`tests/common/expression.rs` changes as follows:

- `build_text_literal` becomes `build_decimal_literal`.
- `ALL_UNARY_OPERATIONS`/`ALL_BINARY_OPERATIONS` stay as test-only helpers
  (F-025 allows that). They get new contents (2 and 13 entries) and a new
  `ALL_LOGICAL_OPERATIONS`. Completeness is guarded by an exhaustive
  `match` in a `const fn` beside each array.
- `CALL_NAMES` is built from `BuiltinFunction::iter()` plus `f` and `g`.
- `copy_deeply` and the strategies gain `Logical` and `Decimal`.
- `build_deep_conjunction` uses `Expression::all([true, tree])`, which
  stays deep, as intended.
- `coerce_to_condition` matches `LiteralValue::Bool`.

**`expression_node_stories.rs` (84 tests)**

- **Keep:**
  - the accessor, `children_*`, `free_identifiers_*` and `substitute_*`
    tests (except the error-type ones below);
  - the equality and identity tests;
  - the `is_alpha_equivalent_under_*` tests;
  - every 4db96b9 DAG and small-stack test.
- **Modify:**
  - `piecewise_expression_try_new_*` (5 tests) now target
    `Expression::piecewise` and `PiecewiseError`.
  - `call_expression_exposes_name_and_arguments` and
    `call_expression_accepts_zero_arguments` use `callee()`.
  - `call_expression_try_new_rejects_empty_function_name` becomes
    `function_name_try_new_rejects_an_empty_name`.
  - `expression_rebuild_with_children_rejects_*` (4 tests) and
    `expression_substitute_refuses_a_number_in_a_piecewise_condition` use
    `RebuildError` and `PiecewiseError` variants.
  - The `Positive` and `decimal_text_literal` cases are updated.
  - `alpha_renaming_are_identifiers_alpha_equivalent_follows_the_renaming`
    is renamed to `…_is_corresponding_follows_the_renaming`.
  - `alpha_renaming_try_new_refuses_a_non_injective_map`: the
    `a_first`/`b_first` rstest cases are removed. Reversing a `Vec` before
    collecting it into a `HashMap` does not control the order (F-037), so
    it is now one plain test.
- **Modify (weakens the test):** remove the `assert_ne!(hash_of(..),
  hash_of(..))` lines (F-037) from these tests:
  - `expression_trees_differing_anywhere_are_unequal_and_hash_differently`
    (renamed `…_are_unequal`);
  - `expression_piecewise_nodes_with_different_case_counts_are_unequal`;
  - `expression_doubling_dags_over_different_leaves_are_unequal_and_hash_differently`;
  - `expression_equality_and_hash_of_deep_trees_reach_the_bottom_on_a_small_stack`;
  - `expression_equality_and_hash_of_deep_doubling_dags_reach_the_bottom_on_a_small_stack`.

  `Hash` promises only that equal values hash equally. What the removal
  gives up is a smoke test of digest quality. The equal-implies-equal-hash
  assertions stay.
- **Delete:**
  - `expression_from_node_struct_rewraps_the_node`: the `From<node struct>`
    impls it tests are removed, and it also uses `%`.

**`expression_builders_stories.rs` (32 tests)**

- **Keep:** `expression_negation_of_owned_and_borrowed_expressions_agree`
  and `expression_new_binary_shares_a_borrowed_operand`.
- **Modify:**
  - `expression_unary_builders_produce_the_matching_unary_node`: drop
    `Positive`.
  - `expression_binary_builders_produce_the_matching_binary_node`:
    `Modulo` becomes `FloorMod` via `floor_mod`, and `LogicalAnd`/`LogicalOr`
    are dropped.
  - `…_promote_a_plain_right_operand`, `…_promote_a_plain_left_operand`,
    `expression_reflected_operators_accept_every_left_operand_type` and
    `expression_arithmetic_operators_accept_every_operand_type_on_the_left`:
    add `i128/u64/usize` and `&Identifier`, and remove `%`.
  - `expression_new_binary_lifts_every_operand_type`: match on
    `LiteralValue`.
  - `expression_new_binary_constructs_with_literal_coercion` and
    `expression_new_unary_constructs_with_literal_coercion`: mechanical
    updates.
  - `expression_binary_builder_takes_a_parsed_text_operand`: assert the
    normalized `Decimal` and its Display, not the spelling.
  - `build_logical_folds_three_operands_to_the_right`,
    `build_logical_and_folds_four_operands_to_the_right` and
    `build_logical_includes_a_leading_expression` become
    `expression_all_and_any_build_one_node_over_every_operand`: the
    expected value is one n-ary node, not a right-fold.
  - `build_logical_accepts_two_operands`: n-ary node.
  - `build_logical_rejects_fewer_than_two_operands` becomes
    `expression_all_and_any_of_zero_or_one_operand` (true/false literal,
    and a `ptr_eq` handle).
  - `build_logical_and_keeps_both_bounds_of_a_range`: mechanical.
  - `expression_logical_not_wraps_the_operand`: uses `!`.
  - `build_piecewise_*` (7 tests) use `PiecewiseError`.
  - `build_call_*` (3 tests) use `Callee`.
  - `build_call_rejects_an_empty_function_name` becomes a `FunctionName`
    test.
  - `expression_build_error_display_describes_the_failure` becomes one
    Display table per new error type (`PiecewiseError`, `RebuildError`,
    `FunctionNameError`, `UnknownNameError`).
- **Delete:**
  - `expression_text_operand_keeps_its_spelling`: decision 6 removes
    spelling preservation, and the F-026 regression test replaces it.
  - The `too_few_operands` Display case: its variant is removed.

**`expression_literal_stories.rs` (43 tests)**

- **Keep:**
  - `literal_value_from_integer_keeps_the_integer`,
    `…_from_big_integer_keeps_every_digit`, `…_from_float_keeps_the_float`,
    `…_from_nan_keeps_a_nan` and `…_from_bool_keeps_the_boolean`: assert
    the variant directly.
  - `…_parse_text_rejects_text_outside_the_grammar`,
    `literal_text_error_display_names_the_text`, `…_differs_when_values_differ`,
    `…_float_differs_from_decimal_text`,
    `…_differs_across_bool_int_and_float`, `…_differs_across_buckets`,
    `…_decimals_differing_in_the_thirtieth_digit_are_unequal`,
    `…_nan_equals_every_nan`, `…_nan_equals_its_clone` and
    `…_negative_zero_equals_zero`.
  - `literal_value_equal_literals_hash_equally`.
  - `function_sort_*` (12 tests): integer-text specs become `Int`.
  - `big_int_re_export_is_the_num_bigint_type`: new path.
- **Modify:**
  - `…_parse_text_keeps_integer_text_verbatim` and
    `…_parse_text_keeps_decimal_text_verbatim` become
    `…_parse_text_normalizes_{integer,decimal}_text` (variant and fields).
  - `…_integer_equals_integer_text`,
    `…_integer_texts_with_distinct_spelling_are_equal` and
    `…_decimal_texts_with_distinct_spelling_are_equal`: also assert the same
    variant and the same Display.
  - `literal_value_display_writes_the_value_as_given` becomes
    `literal_value_display_follows_rust_conventions`. This is the F-013
    table; the Python spellings (`True`, `1e+16`, verbatim `05`) go.
  - `literal_value_edge_case_has_its_key_bucket_and_sorts`: drop the key
    column and keep the variant and sort columns.
- **Modify (weakens the test):**
  `literal_value_distinct_literals_are_unequal_and_hash_differently` drops
  its `assert_ne!` on hashes (F-037).
- **Delete:**
  - `literal_value_canonical_key_differs_in_the_thirtieth_digit`,
    `literal_value_canonical_key_is_shared_exactly_by_equal_literals` and
    `literal_value_canonical_key_renders_bucket_and_canonical_form`:
    `canonical_key` is removed, and the equality tests already cover these
    properties.
  - `literal_value_is_integer_valued_for_every_integer_bucket_form`,
    `…_is_not_integer_valued_for_other_buckets` and
    `…_is_integer_valued_agrees_across_equal_literals`: the method is
    removed. A variant match and `FunctionSort::Int.accepts_literal` replace
    it.

**`expression_wire_stories.rs` (23 tests)**

- **Keep:**
  - `expression_literal_round_trips_through_its_wire_form`;
  - `expression_literal_refuses_to_serialize_a_non_finite_float` (our own
    message);
  - `expression_piecewise_round_trips_through_a_json_value`,
    `…_through_json_text` and `expression_nested_piecewise_round_trips`;
  - `expression_round_trip_restores_the_same_identifier`.
- **Modify:**
  - `expression_serializes_every_node_kind_in_its_wire_shape`,
    `expression_serializes_fields_in_declaration_order`,
    `expression_literal_serializes_as_the_json_value_of_its_kind` and
    `expression_serializes_operations_by_wire_name`: move to the node-table
    shape of §5.6.
  - `expression_literal_writes_a_big_integer_as_a_json_integer` becomes
    `…_as_a_decimal_string` (S-2).
  - `expression_deep_tree_round_trips_through_a_json_value` becomes
    `…_through_json_text_on_a_small_stack`. The
    `SERIALIZATION_STACK_BYTES` stack is gone, so this strengthens the test.
  - `expression_deserialize_rejects_a_malformed_literal`,
    `…_rejects_a_malformed_node`, `…_rejects_a_node_its_constructor_refuses`
    and `…_rejects_a_nested_numeric_condition` use node-table payloads and
    assert our §5.6 message with `ends_with` or `contains`. The line and
    column suffix is serde_json's.
  - `expression_deserialize_rejects_an_unknown_type_id` becomes
    `…_rejects_an_unknown_node_kind` and asserts `error.is_data()` without
    pinning serde's text (F-035).
- **Delete:**
  - `expression_literal_keeps_the_json_number_kind`: int versus float is
    decided by the tag now, not by the JSON token.
  - `expression_literal_refuses_a_float_token_beyond_the_f64_range` and
    `expression_literal_reads_an_underflowing_float_token_as_a_signed_zero`:
    they pin serde_json's number parsing and its spelling of the number
    (`1e+400`), not ours (F-035).
  - `expression_literal_reads_a_text_with_its_spelling`: spelling is not
    kept (decision 6).
  - `expression_json_text_decodes_up_to_the_serde_json_nesting_limit`: it
    pins serde_json's recursion limit and message. The limit no longer
    applies; the F-003 regression tests replace it.
  - `expression_deserialize_restores_no_identifier_from_a_refused_payload`,
    plus its helpers `ID_COUNTER_LOCK`, `hold_id_counter`,
    `reserve_ahead_id`, `has_counter_passed` and
    `assert_refused_before_any_restore`: S-2 drops the no-side-effect
    contract.

**`expression_tree_stories.rs` (13 tests)**

- **Keep all.**
- **Modify:**
  - `expression_tree_rebuild_refuses_a_wrong_child_count` and
    `rewrite_tree_reports_a_refused_expression_rebuild` use `RebuildError`.
  - Pass-context plumbing follows B5.
  - `rewrite_pass_rewrites_a_doubling_expression_dag_once_per_distinct_node`,
    line 383: its failure arm `panic!("expected a sum, got {node:?}")`
    becomes `panic!("level {level} is not a sum")` (F-037). The bounded
    `Debug` would already make the old arm terminate. The change keeps the
    failure message small.

**`expression_screen_stories.rs` (56 tests)**

- **All:** the `Screen` helper enum's `run` calls
  `BooleanScreen::new().with_sorts(..).with_environment(..).with_symbol_types(..)`
  and `.check_logical_operands` or `.check_predicate`.
  - Position assertions use `parent()` and
    `LogicalOperand { operation: LogicalOperation::And, operand_index }`.
  - `PredicateRoot` assertions become `parent().is_none()`.
- **Keep (semantics unchanged):** every `validate_*` behavior test and every
  4db96b9 DAG and small-stack test.
- **Modify:**
  - `validate_logical_operands_accepts_a_call_the_lookup_does_not_know`
    uses `Callee::Named("f")`. `floor` is now a built-in with a known sort
    (§5.7), and `floor` moves into a new test (§8.3).
  - `BuiltinSorts` keeps only the constants.
  - `validate_logical_operands_takes_a_trait_object_lookup`:
    `with_sorts(&*boxed)`.
  - `non_boolean_logical_operand_error_display_describes_the_position`
    becomes one Display table for the new short text (F-035: the full
    sentence repeated across 5 tests goes).
  - The seven `assert!(result.is_ok(), …)` at lines 1175, 1233, 1263,
    1293, 1294, 1352 and 1354 become `assert_eq!(result, Ok(()))` (F-037,
    `assertions_on_result_states`). They were `is_ok()` only to avoid
    `Debug`-printing a DAG, which §5.4 makes safe. This strengthens them.
- **Delete:**
  - `non_boolean_logical_operand_error_display_writes_identifier_ids`: the
    Display no longer writes expressions.
  - `non_boolean_logical_operand_error_display_writes_a_deep_tree_on_a_small_stack`:
    the Display is bounded and contains no expression. The small-stack test
    for `display()` lives in pprint (§8.3).

**`pprint_stories.rs` (38 tests) and `pprint_properties.rs` (7 tests)**

- **Keep:** the structure, notation, parenthesization, identifier-style,
  piecewise, call and small-stack tests.
- **Modify:**
  - Every `format_expression(e, o)` becomes `e.display(o).to_string()`.
  - The literal cases follow §5.3 Display:
    - `"(!True)"` becomes `"(!true)"`;
    - `"True"` becomes `"true"`;
    - `"1e+16"` becomes `"10000000000000000"`;
    - `"1.2345678901234568e+17"` becomes `"123456789012345680"`;
    - `"{1 if True; 2 otherwise}"` becomes `"{1 if true; 2 otherwise}"`;
    - and so on.
  - `format_expression_writes_a_folded_conjunction_as_nested_pairs` becomes
    `…_writes_a_logical_node_with_every_operand`.
  - The `Positive` and `modulo` cases become `FloorMod` (`(x % 2)`,
    `(floor_mod x 2)`).
  - `format_expression_writes_integer_and_integer_text_alike` becomes
    `…_int_float_and_decimal_one_alike` (`1`).
  - `format_expression_writes_a_negative_literal_operand_bare` is kept,
    with the doc comment reworded. `(-1 ** 2)` is now Rust's contract, not
    a Python spelling.
  - The `boolean_word("True")` name-hint case is kept.
  - `DeepShape::RightConjunction` expected pieces use `(and true ` and
    `(true && `.
  - `pprint_properties.rs`: `count_inner_nodes` gains `logical`.
    `format_expression_writes_a_literal_as_its_display_text` is kept.
  - The rustfmt-skipped misindentation inside `proptest!` is re-indented
    by hand (F-037).

**`builtins_stories.rs` (29 tests)**

- **Keep:** order, uniqueness, signature, value, body, parameter and thread
  tests, now against `BuiltinFunction::iter`, `BuiltinConstant::iter` and
  `composed()`.
- **Modify:**
  - `composed_functions_lists_the_composed_builtins_in_catalogue_order`,
    `native_functions_lists…` and `native_constants_lists…` iterate the
    enums.
  - `builtin_catalogue_names_are_unique_across_the_tables` becomes
    `builtin_names_are_unique_and_round_trip_through_from_str`.
  - `composed_function_body_prints_as_its_documented_text`: the float
    spellings follow §5.3 (`1.0 / …` becomes `1 / …`).
  - `composed_function_body_calls_only_builtin_functions` checks that every
    callee is a `Callee::Builtin`.
  - `xor`, `nand`, `nor` and `implies` expected bodies use `Logical`.
  - The 16-name list copied four times (F-037) becomes
    `BuiltinFunction::iter()`.
  - Duplicated helpers merge into `common/expression.rs`.
- **Delete:** none.

**`builtins_scope_stories.rs` (1 test):**

- Delete `composed_functions_first_used_in_a_scope_keeps_its_parameters_unique`,
  under decision 12 (F-018). The deterministic-identifier scope it runs in
  is removed, and the file goes with the `testing` feature. The batch that
  deletes `testing` executes this.

**`vocabulary_stories.rs` (23 tests)**

- **Keep:**
  - the `as_str`, Display and serialize tests for all four enums;
  - `…_deserialization_rejects_other_input`;
  - `vocabulary_payload_round_trips_through_wire_json_text`, with
    `"floor_divide"` still valid;
  - `vocabulary_payload_rejects_an_operation_named_by_symbol`;
  - `function_sort_display_renders_a_signature`.
- **Modify:**
  - `unary_operation_accepts_exactly_its_three_wire_names` becomes `two`.
  - `binary_operation_accepts_exactly_its_fifteen_wire_names` becomes
    `thirteen`, with `floor_mod`.
  - Add `LogicalOperation` equivalents.
  - `operation_symbols_invert_to_their_operations` covers three enums.
  - `vocabulary_rejection_names_the_word_and_the_expected_names` pins serde
    wording (`invalid value: string …, expected …`), which F-035 flags. It
    becomes a check that `"plus".parse::<BinaryOperation>()` gives
    `UnknownNameError` with `name() == "plus"` and Display `unknown binary
    operation `plus``, which is our own text. Serde errors only get
    `is_data()`.

**`expression_properties.rs` (22 tests)**

- **Keep all.** `expression_equality_is_symmetric` already compares against
  a deep copy at HEAD (`:241-259`), so the "equal branch almost never hit"
  part of F-037 is already addressed.
- **Modify:**
  - `literal_value_canonical_key_agrees_for_integer_and_digit_text` becomes
    `literal_value_parse_text_of_digits_equals_the_integer`.
  - `literal_value_canonical_key_ignores_trailing_decimal_zeros` becomes
    `literal_value_decimal_ignores_trailing_zeros`.
  - `literal_value_equality_agrees_with_key_hash_and_bucket` and
    `…_over_every_literal` become "equality agrees with hash and variant".
  - `literal_value_display_writes_texts_verbatim_and_integers_as_digits`
    becomes `literal_value_display_parses_back_to_an_equal_int_or_decimal`
    (the property holds for `Int` and `Decimal`).
  - The JSON round-trip properties use the new shape and add postcard (§8.3).
  - The screen properties use `BooleanScreen`.

**`expression_pass_stories.rs` (24 tests; B4 owns the `RewriteRuleApplier` ones)**

- Line 287, `rewrite_rule_applier_rewrites_a_doubling_dag_once_per_distinct_node`:
  `assert!(outcome.output() == &build_doubling_dag(&a, 64))` becomes
  `assert_eq!(outcome.output(), &build_doubling_dag(&a, 64))`. This fixes
  the `clippy::manual_assert_eq` break.
  - It is safe only once §5.4's bounded `Debug` lands.
  - If this commit has to land first, use
    `#[expect(clippy::manual_assert_eq, reason = "Debug of a 2^64-occurrence DAG does not terminate until the bounded Debug lands")]`
    and remove it together with the `Debug` change.
- `expression_pretty_formatter_*` (6 tests) move to `expr::passes`.
  `…_execute_matches_format_expression` becomes `…_matches_display`.
- `expression_pretty_formatter_name_is_its_type_name` pins `type_name`
  output (F-035). It follows B5's naming decision.

**Unit tests in `src`**

- `operation.rs::tests` (2 rstest groups): keep, dropping `Positive` and
  the logical variants, and add a `LogicalOperation` group.
- `python_text.rs::tests`: delete with the file. The decimal-normalization
  cases move to `literal.rs::tests`. The `repr(float)` and
  `Decimal`-`E`-notation cases are deleted: that Python text is no longer
  produced.
- `shipped.rs::tests`: B1.

#### 8.2 One regression test per fixed finding

| Finding | Test (file) |
|---|---|
| F-003 (n-ary) | `expression_all_of_ten_thousand_comparisons_is_one_logical_node` (builders): `operands().len() == 10_000`, and its children are the comparisons |
| F-003 (decode depth) | `expression_conjunction_of_ten_thousand_comparisons_round_trips_through_json_text` (wire): `serde_json::from_str` succeeds; this failed beyond about 62 levels at HEAD |
| F-003 (no recursion) | `expression_deep_sum_round_trips_through_json_text_on_a_small_stack` (wire): 100,000-level sum, 256 KiB thread |
| F-003 (Debug) | `expression_debug_of_a_deep_tree_completes_on_a_small_stack` and `expression_debug_of_a_doubling_dag_is_bounded` (node): 64-level DAG, output shorter than 64 KiB |
| F-003 (DAG wire) | `expression_wire_encodes_a_doubling_dag_once_per_distinct_node` (wire): 64 levels, `nodes.len() == 65`, and the decoded result shares like the input (`ptr_eq` of left and right at each level) |
| F-010 | `expression_floor_mod_builds_a_floor_mod_node` (builders) and a compile-fail doctest `Expression % 3` (E0369) on `Expression`; `expression_div_operator_builds_true_division` (builders) |
| F-012 / S-2 | `expression_round_trips_through_postcard` (wire, over `build_expression_dag_strategy`, as a property) and `literal_big_int_serializes_as_a_decimal_string` (wire): `10^30` gives `{"int":"1000…"}` |
| F-013 | `literal_value_display_follows_rust_conventions` (literal): `true`, `NaN`, `inf`, `-0`, `1`, `10000000000000000`, `0.001` |
| F-014 | `expression_piecewise_errors_are_piecewise_errors` (builders, exhaustive `match` on `PiecewiseError`, no wildcard arm); `rebuild_error_display_does_not_repeat_its_source` (node): `Display` plus `source()` chain printed once; `boolean_screen_error_parent_is_none_only_at_a_predicate_root` (screen) |
| F-015 | `expression_display_equals_display_with_default_options` and `expression_display_of_a_deep_tree_completes_on_a_small_stack` (pprint) |
| F-023 (API) | `boolean_screen_new_knows_nothing` (screen): same answers as HEAD's empty maps plus `NoRegisteredSorts` over the existing fixtures; `boolean_screen_accepts_a_closure_for_symbol_types` |
| F-023 (quadratic) | `boolean_screen_judges_each_nested_piecewise_once` (screen): a 2,000-deep otherwise-chain of piecewise nodes whose values are `Named` calls, predicate screen; a counting `SortLookup` sees at most `2 × 2000` `call_result_sort` calls. At HEAD this is about d²/2 |
| F-024 | `expression_from_every_primitive_builds_its_literal` (builders, rstest over the 10 types); `callee_from_str_resolves_builtin_names` and `function_name_refuses_a_builtin_name` (builders); `expression_not_operator_builds_logical_not` (builders); `boolean_screen_knows_builtin_result_sorts_without_a_lookup` (screen: `floor(1.5) && true` refused under `new()`) |
| F-025 | `vocabulary_as_str_display_serde_and_from_str_agree_for_every_variant` (vocabulary), over the test-only exhaustive lists of all seven enums; the `builtins.rs` unit test `catalogue_order_array_lists_every_variant` |
| F-026 | `literal_value_parse_text_normalizes_spelling` (literal): `"05"` gives `Int(5)`; `"1.50"`, `"1.5"` and `"01.500"` are equal with Display `1.5`; `"100.0"` has Display `100` |
| F-029 | compile-level: no test uses `list_*`/`find_*`; `builtin_function_composed_is_none_exactly_for_natives` (builtins) |
| F-035/F-037 | the modifications in §8.1 (text tables, `is_ok()`, HashMap order, hash inequality, failure arms, the misindentation) |
| clippy | `expression_pass_stories.rs:287` fixed as above; CI clippy `-D warnings` is green |

#### 8.3 New story, property and adversarial tests

- **Stories:**
  - `expression_all_of_one_operand_is_that_operand` (`ptr_eq`);
  - `expression_all_and_any_of_nothing_are_true_and_false`;
  - `expression_and_does_not_flatten_a_nested_conjunction`;
  - `expression_rebuild_of_a_logical_node_keeps_its_operand_count`;
  - `expression_display_writes_a_logical_node_in_both_notations`;
  - `boolean_screen_reports_the_operand_index_of_a_logical_operand`;
  - `decimal_display_is_positional_and_normalized`;
  - `function_name_try_new_rejects_empty_and_builtin_names`.
- **Properties:**
  - `expression_wire_round_trip_preserves_structure_and_sharing` over DAG
    strategies, for both JSON and postcard. It checks equality, and it
    checks that the input's `ptr_eq` pairs among children equal the
    output's.
  - `literal_value_equality_is_an_equivalence_agreeing_with_hash`.
  - `decimal_from_str_of_display_is_identity`.
  - `expression_debug_is_bounded`: output length ≤ C for every generated
    DAG.
  - `boolean_screen_on_a_dag_answers_as_on_its_unshared_copy`: keeps the
    existing 4db96b9 property and extends it to `Logical`.
- **Adversarial decode.** Each case below must fail with the §5.6 message
  and must not panic:
  - an empty `nodes`;
  - a forward reference (`left: 5` in node 2);
  - a self-reference;
  - a `Logical` node with 1 operand;
  - a `Piecewise` node with `cases: []`;
  - a `literal: {"bool": 1}`;
  - an `int: "007"`, `"-0"`, `"+1"` or `"1e3"`;
  - a `decimal: "-1.5"`;
  - a `callee: {"named": "max"}`;
  - a `callee: {"named": ""}`;
  - an unreferenced node;
  - a `float` of `1e400` (a serde_json error, classified only);
  - 1,000,000 nodes of a linear chain, which must decode on a 256 KiB
    stack.
- **Postcard adversarial.** Truncated input, and an index out of range.

### 9. Encapsulation delta

- **Removed from the public surface:**
  - `IntoOperand` and `sealed::Sealed`;
  - `LiteralKind`;
  - `ExpressionBuildError`;
  - `format_expression`;
  - `validate_logical_operands` and `validate_predicate`;
  - `NativeFunctionSignature` and `NativeConstantSpec`;
  - the `list_*`/`find_*` functions;
  - the four `From<node struct> for Expression` impls;
  - `LiteralValue::{from_bool, kind, canonical_key, is_integer_valued}`;
  - `CallExpression::function_name`.
- **Narrowed to `pub(crate)`:** `PiecewiseExpression::try_new` and
  `CallExpression::try_new`. The node-invariant checks
  (`validate_case_count` and the others) stay `pub(super)`, shared with
  `wire.rs`.
- **Deleted crate-internal code:**
  - `wire_name.rs` and `impl_wire_name_traits!`;
  - every `ALL_*` array in `src`, except the private catalogue-order arrays
    behind `BuiltinFunction::iter`/`BuiltinConstant::iter`, whose
    completeness a unit test guards;
  - `python_text.rs`;
  - `ExpressionPayload`, `PayloadNode` and the `Decode` impl
    (`decode.rs` itself is deleted per S-2);
  - `initialize_composed_functions`;
  - `CanonicalForm`.
- **Deliberately widened.** `LiteralValue` becomes a public enum, so its
  representation is intended API. Normalization, the only invariant, lives
  in `Decimal`, which has private fields and is constructible only through
  `FromStr` and `Deserialize`. `Float(f64)` has no invariant: every f64,
  NaN included, is valid.
- **New public types with private fields:** `FunctionName` (non-empty and
  not a built-in name; there is no public tuple field), `Decimal`,
  `LogicalExpression`, `BooleanScreen`, `ExpressionDisplay` and
  `UnknownNameError`.
- **Private wire types.** `ExpressionWire` and `WireNode` are private, so
  the wire shape is documented, not exposed as types.
- **Trait-object lookups.** `BooleanScreen` stores `&dyn` lookups, so its
  signature exposes no generic map types.
- **Module privacy.** Node structs live in the leaf `expr::node`. Only
  `node.rs`, `build.rs` and `wire.rs` construct them, through `pub(crate)`
  constructors that check the invariants, and no descendant module touches
  their fields.
- **Single paths.** Each item has one public path, and `mod.rs` has no glob
  re-export. `BigInt`'s single path is `fhy_core::expr::BigInt`.

### 10. Findings covered and decisions for sign-off

**Fixed here:**

- F-003;
- F-010;
- F-012, the expression wire part;
- F-013, the literal and `canonical_key` part;
- F-014, the expression and screen errors;
- F-015, the expression part;
- F-023, the API and the quadratic part (the DAG part was already fixed);
- F-024;
- F-025;
- F-026;
- F-029, the builtins part;
- F-035 and F-037, the symbolic tests;
- F-039, the `wire_name` allocation.

**Verified already fixed:** F-002 and the DAG part of F-023.

Decisions I made that the user should confirm:

1. **No flattening in `all`/`any`/`and`/`or`.** Flattening at build time
   blows a shared DAG up exponentially (§5.1). Nested `and`s built step by
   step therefore stay nested.
2. **0 and 1 operands.** `all([])` is `true`, `any([])` is `false`, and one
   operand returns that operand, so the builders are infallible. The
   trade-off: `all([2])` is just `2`, so the screen no longer sees that
   Boolean position. The alternative is to keep a
   `TooFewLogicalOperands`-style error.
3. **Wire format is a flat node table**, not a derived recursive enum. This
   is what makes serde non-recursive with plain derives, and it makes DAG
   encoding linear. The cost is that the JSON is less readable by hand.
4. **Non-finite floats are still refused at serialization**, in every
   format, so JSON never gets `null`. NaN and infinity literals cannot go
   over the wire at all, even through postcard.
5. **`UnaryOperation::Positive` is removed.** Python passes still handle
   `POSITIVE`, and the future binding will have to map `+x` to `x`.
6. **`BinaryOperation::Modulo` is renamed `FloorMod`** (wire `floor_mod`),
   to match `floor_mod()`. `Divide` keeps its name. There is no
   `true_divide` builder; `/` is the one way to write true division. The
   named `floor_divide` stays.
7. **`logical_not()` is removed in favor of `!`.** There is no named twin,
   just as `-` has no `negate()`.
8. **`FunctionName` rejects built-in names**, which keeps a single
   representation of each call. `Callee::from_str` routes built-in names to
   `Builtin`.
9. **The screen knows built-in result sorts from the catalogue**, which is
   a behavior change (§5.7). An unregistered `floor(x) && p` is now
   refused.
10. **The screen error's `Display` no longer embeds expressions.** It is
    short and bounded, and callers print `operand()` and `parent()`
    themselves.
11. **Extensions to S-6's exhaustive list:** `LiteralValue`,
    `LogicalOperation`, `Callee`, `FunctionSort` and `SymbolType`. The
    first three have the same rationale as `ExpressionKind`. `FunctionSort`
    and `SymbolType` are closed classifications. `BuiltinFunction`,
    `BuiltinConstant`, `Notation`, `IdentifierStyle`, `BooleanPosition` and
    every error enum stay `#[non_exhaustive]`.
12. **Pass types go in `fhy_core::expr::passes`** (S-1 choice). B4 should
    match this for `RewriteRuleApplier` and registration.
13. **`display` returns a named `ExpressionDisplay<'_>`, not `impl
    Display`.** It follows `Path::display` and can be stored in a struct.
14. **`AlphaRenaming::are_identifiers_alpha_equivalent` is renamed
    `is_corresponding`** (S-7).
15. **Composed-function parameters stay lazily minted** and are not put in
    S-3's reserved block. Since S-3 caps payload ids at 2^63, lazy minting
    cannot exhaust the counter.

Possibly misjudged or incomplete in the audit:

- **F-023 was reported as fixed by 4db96b9, but only the DAG re-walk was.**
  The O(d²) behavior on nested piecewise (the audit's third bullet) is
  still present. It is specified above.
- **The audit located F-003's "tests can hang" at
  `tests/expression_tree_stories.rs:383`.** At HEAD the same hazard also
  explains the `is_ok()` workarounds in `expression_screen_stories.rs`
  (seven sites) and the `assert!(==)` that breaks clippy at
  `expression_pass_stories.rs:287`. All of these are one root cause, the
  recursive `Debug`, not independent test-quality nits.
- **F-036 (symbolic golden corpus) should be closed by decision 3**, not
  fixed. The symbolic area is not dual-defined, so there is no Python oracle
  to record. It is not in the triage's list of deferred findings; I suggest
  recording it as "resolved by decision 3".
- **F-010's suggested fix** (remove `Div` as well, and add `true_divide`)
  is superseded by decision 11. This spec follows the decision.

---

## B4: patterns, rewrite rules and the expression passes (`fhy_core::expr::pattern`, `fhy_core::pass::expr`)

Scope at HEAD 412e234:
- `rust/fhy-core/src/symbolic/expression/pattern/{mod.rs, core.rs, rewrite.rs}`
- `symbolic/expression/registration.rs`
- `ExpressionPrettyFormatter` in `symbolic/expression/pprint.rs:297-324`
- tests: `tests/{pattern_stories, pattern_properties, pattern_rewrite_stories, pattern_user_stories, expression_pass_stories}.rs` and `tests/common/pattern.rs`

Findings: F-009, F-014 (pattern and rewrite errors), F-019, F-029 (pattern part), F-039 (rewrite blame), the pattern-test parts of F-035/F-037, and PAT-06/PAT-07 from the pattern sub-audit.

### 1. Summary

Patterns get typed `Capture` handles, bindings that are indexed by handle, and constructors that cannot fail and take no `Option` mode arguments. Matching now uses one literal equality, and bindings are kept on a `&mut` trail. Rewrite rules become one composable `Rule` trait: a rule that declines or returns its own input does not fire, which fixes F-009 on the rule side. The two concrete expression passes and their registration move out of `expr` into `fhy_core::pass::expr`, and registration fills a caller-owned `PassRegistry` (decision 7). With that move, `expr` no longer imports anything from `pass`.

### 2. Desired public interface

Module layout (S-1). There are no glob re-exports, and every item has exactly one public path:

```
src/expr/pattern/mod.rs        module docs; explicit `pub use` of the items below
src/expr/pattern/matching.rs   (private; replaces pattern/core.rs) Capture, Pattern, MatchBindings, CallbackError
src/expr/pattern/rewrite.rs    (private) Rule, RewriteRule, RewriteOutcome, FiredRule, RewriteError, apply_rewrite_rules
src/pass/expr.rs               RewriteRuleApplier, ExpressionPrettyFormatter, register_expression_passes
```

Public paths: `fhy_core::expr::pattern::{Capture, Pattern, MatchBindings, CallbackError, Rule, RewriteRule, RewriteOutcome, FiredRule, RewriteError, apply_rewrite_rules}` and `fhy_core::pass::expr::{RewriteRuleApplier, ExpressionPrettyFormatter, register_expression_passes}`.

**Layering decision (S-1): the passes go to `fhy_core::pass::expr`, not `expr::passes`.** Reasons:
- Nothing under `src/expr/` imports `crate::pass`, so a single grep checks the rule "expr must not depend on pass".
- The dependency then points up the S-1 chain (`expr → pass`, so `pass` may use `expr`).
- In a later crate split (decision 13), the two passes land in the passes crate, which already sits above the symbolic crate. An `expr::passes` module would have to move out of its parent crate.

`expr::pattern` still uses `rewrite_tree`/`Rewriter`/`Tree` and the identity-keyed map. After F-008 (B5) these come from `fhy_core::tree`. This spec assumes `NodeIdentity` and the crate-private `BuildIdentityHasher` also live at or below `expr` (in `tree`). If B5 keeps them in `pass`, the blame map keys on a private address newtype in `expr::pattern` instead, with the same contract.

#### 2.1 `Capture` (NEW, `pub`)

```rust
/// A handle a pattern binds a matched expression to.
///
/// Identity, not name, decides equality: two handles are equal exactly when
/// one is a clone of the other. `Capture::new("x")` called twice gives two
/// independent captures. The name is only for `Debug`, `Display` and panic
/// messages, and it may be empty.
#[derive(Clone)]
pub struct Capture(Arc<CaptureName>);           // CaptureName: private struct holding Box<str>

impl Capture {
    #[must_use] pub fn new(name: &str) -> Self;
    #[must_use] pub fn name(&self) -> &str;
}
impl PartialEq for Capture {}   // Arc::ptr_eq
impl Eq for Capture {}
impl Hash for Capture {}        // hashes the Arc's address
impl fmt::Debug for Capture {}  // Capture("x")
impl fmt::Display for Capture {}// x
```

`Capture` is `Send + Sync`, and a `const _` block asserts it. `new` takes `&str` and allocates its own `Arc`, so the identity is always fresh; an `impl Into<Arc<str>>` parameter would let two captures share an `Arc` and compare equal by accident.

#### 2.2 `Pattern` (CHANGED, `pub`)

The representation becomes `Pattern(Arc<PatternKind>)`, which is private, so cloning a pattern or a rule is O(1) (PAT-13). Every constructor is infallible and `#[must_use]`. Sequence-taking constructors accept `impl IntoIterator`.

```rust
impl Pattern {
    pub fn wildcard() -> Self;                                         // KEPT
    pub fn nothing() -> Self;                                          // NEW: matches no expression
    pub fn capture(capture: &Capture) -> Self;                         // CHANGED: any expression, bound to `capture`
    pub fn captured_as(self, capture: &Capture) -> Self;               // NEW: replaces capture(name, sub_pattern)
    pub fn any_literal() -> Self;                                      // NEW: was literal(None)
    pub fn literal(value: impl Into<LiteralValue>) -> Self;            // CHANGED: was literal(Some(v))
    pub fn any_identifier() -> Self;                                   // NEW: was identifier(None)
    pub fn identifier(identifier: Identifier) -> Self;                 // CHANGED: was identifier(Some(i))
    pub fn unary(operation: UnaryOperation, operand: Pattern) -> Self; // CHANGED: was unary(Some(op), ..)
    pub fn unary_any_operation(operand: Pattern) -> Self;              // NEW: was unary(None, ..)
    pub fn binary(operation: BinaryOperation, left: Pattern, right: Pattern) -> Self; // CHANGED
    pub fn binary_any_operation(left: Pattern, right: Pattern) -> Self;               // NEW: was binary(None, ..)
    pub fn piecewise(
        cases: impl IntoIterator<Item = (Pattern, Pattern)>,
        otherwise: Pattern,
    ) -> Self;                                                         // CHANGED: was piecewise(Some(cases), o) -> Result
    pub fn piecewise_any_cases(otherwise: Pattern) -> Self;            // NEW: was piecewise(None, o)
    pub fn call(function_name: &str, arguments: impl IntoIterator<Item = Pattern>) -> Self; // CHANGED: was call(Some, Some)
    pub fn call_any_arguments(function_name: &str) -> Self;           // NEW: was call(Some(f), None)
    pub fn call_any_name(arguments: impl IntoIterator<Item = Pattern>) -> Self;             // NEW: was call(None, Some(a))
    pub fn any_call() -> Self;                                         // NEW: was call(None, None)
    pub fn predicate<F>(predicate: F) -> Self
    where F: Fn(&Expression) -> bool + Send + Sync + 'static;          // CHANGED: now infallible
    pub fn try_predicate<F>(predicate: F) -> Self
    where F: Fn(&Expression) -> Result<bool, CallbackError> + Send + Sync + 'static; // NEW: the old fallible form
    pub fn alternatives(alternatives: impl IntoIterator<Item = Pattern>) -> Self;    // CHANGED: was Result

    pub fn matches(&self, expression: &Expression)
        -> Result<Option<MatchBindings>, CallbackError>;               // NEW: replaces free match_pattern
    pub fn is_match(&self, expression: &Expression)
        -> Result<bool, CallbackError>;                                // NEW: replaces free does_pattern_match

    pub(super) fn match_into(&self, expression: &Expression, bindings: &mut MatchBindings)
        -> Result<bool, CallbackError>;                                // replaces pub match_under
}
```

Naming rule: a `_any_<part>` suffix leaves that part unconstrained, and an `any_` prefix leaves the whole node's parameters unconstrained. The audit's `any_binary(l, r)` is spelled `binary_any_operation(l, r)` under this scheme.

The name type of the call patterns follows whatever B3/F-024 picks for `CallExpression::function_name`. At HEAD it is `&str`, and the pattern compares it verbatim.

`Pattern` stays `Clone + Debug + Send + Sync` and keeps its hand-written opaque `Debug` for predicates. It has no `PartialEq`, because predicates cannot be compared.

#### 2.3 `MatchBindings` (CHANGED, `pub`)

```rust
#[derive(Debug, Clone, Default)]
pub struct MatchBindings { entries: Vec<(Capture, Expression)> }    // private: the trail

impl MatchBindings {
    #[must_use] pub fn new() -> Self;                                            // NEW: replaces empty()
    #[must_use] pub fn is_empty(&self) -> bool;                                  // KEPT
    #[must_use] pub fn len(&self) -> usize;                                      // NEW
    #[must_use] pub fn contains(&self, capture: &Capture) -> bool;               // NEW: replaces has(&str)
    #[must_use] pub fn get(&self, capture: &Capture) -> Option<&Expression>;     // CHANGED: was get(&str)
    pub fn iter(&self) -> impl Iterator<Item = (&Capture, &Expression)> + '_;    // NEW: replaces names()
    // pub(super): bind, truncate(len), clear  (the trail operations)
}
impl Index<&Capture> for MatchBindings { type Output = Expression; }             // NEW; panics if unbound
impl PartialEq for MatchBindings {}  impl Eq for MatchBindings {}  impl Hash for MatchBindings {} // KEPT: order-insensitive
```

`get` and `index` are a linear scan comparing pointers. A pattern holds few captures, so this beats a hash map in practice. The audit's suggestion of `Vec<Option<Expression>>` indexed by a capture id is rejected: dense ids need either a global counter (against decision 2) or a per-pattern "capture set", which callers could mix up across sets.

REMOVED from the public API: `MatchBindings::empty`, `has`, `names`, `get(&str)` and `try_bind`. Tests and callers now build bindings only through matching (see 5.2).

#### 2.4 `CallbackError` (CHANGED, `pub`). PAT-07 decision: erased, not generic

```rust
/// The failure of a caller-supplied callback: a predicate, a guard, a
/// rewrite, or a native `Rule`. It carries the callback's own error, type-erased.
///
/// It deliberately does not implement `std::error::Error` (as with
/// `anyhow::Error`). That is what makes the blanket `From` below coherent, so
/// callbacks can use `?` on any error type.
pub struct CallbackError(Box<dyn Error + Send + Sync>);

impl CallbackError {
    #[must_use] pub fn new(error: impl Into<Box<dyn Error + Send + Sync>>) -> Self;  // KEPT (&str, String, any error)
    #[must_use] pub fn inner(&self) -> &(dyn Error + Send + Sync + 'static);        // KEPT
    #[must_use] pub fn into_inner(self) -> Box<dyn Error + Send + Sync>;             // KEPT
    #[must_use] pub fn downcast_ref<E: Error + 'static>(&self) -> Option<&E>;        // NEW
}
impl fmt::Debug for CallbackError {}    // the inner error's Debug
impl fmt::Display for CallbackError {}  // the inner error's Display
impl<E: Error + Send + Sync + 'static> From<E> for CallbackError {}      // NEW: replaces the two From impls
impl From<CallbackError> for Box<dyn Error + Send + Sync + 'static> {}   // NEW: `?` into boxed errors
impl From<CallbackError> for Box<dyn Error + 'static> {}                 // NEW
```

Why erased rather than generic (`RewriteRule<E>`, `RewriteError<E>`):
1. A generic error would spread a type parameter over `Pattern` (predicates are fallible), `RewriteRule`, `Rule`, `RewriteError` and `RewriteRuleApplier`. It would also force every rule in one list to share one `E`, so rule libraries from different crates could not be mixed.
2. The pass layer erases the error to `PassFailure = Box<dyn Error + Send + Sync>` anyway, so a generic `E` buys nothing through a pipeline.
3. The cost of erasure is one box on the error path only.
4. Recovering the type is `downcast_ref`, and the blanket `From` removes the `.map_err(CallbackError::new)` boilerplate PAT-07 complained about.

I checked in a scratch crate that the blanket `From<E>` and the two `From<CallbackError> for Box<dyn Error..>` impls are coherent with `rustc` from `~/.cargo/bin`.

S-5 tension: `CallbackError` is a public error-like type that does not implement `Error`. The alternative would be implementing `Error` and giving up `?` on user errors. This needs sign-off (D-B4-3).

#### 2.5 `Rule` (NEW, `pub` trait, not sealed). PAT-06

```rust
/// A rewrite tried at the root of one expression.
pub trait Rule {
    /// Return the replacement for `expression`, or `Ok(None)` to decline.
    ///
    /// Returning a handle to `expression` itself (`Expression::ptr_eq`)
    /// means the same as declining: the walk does not record a firing and
    /// tries the next rule.
    fn apply(&self, expression: &Expression) -> Result<Option<Expression>, CallbackError>;

    /// Return the rule's name, used in firings, diagnostics and errors.
    fn name(&self) -> Option<&str> { None }
}
impl<R: Rule + ?Sized> Rule for &R {}
impl<R: Rule + ?Sized> Rule for Box<R> {}
impl<R: Rule + ?Sized> Rule for Arc<R> {}
impl Rule for RewriteRule {}
```

- There is no blanket impl for closures, because it would overlap the forwarding impls (`&F` and `Box<F>` are themselves `Fn`).
- A closure rule is written as `RewriteRule::new(Pattern::capture(&e), move |b| f(&b[&e]))`.
- A rule that borrows context, or that needs no `'static` bound, implements `Rule` on its own struct.
- `Rule` has no `Send`/`Sync` supertrait. The pass adds `Send` where decision 10 requires it.

#### 2.6 `RewriteRule` (CHANGED, `pub`)

```rust
#[derive(Clone)]
pub struct RewriteRule { pattern: Pattern, rewrite: RewriteFn, name: Option<Arc<str>> }  // private fields
// RewriteFn = Arc<dyn Fn(&MatchBindings) -> Result<Option<Expression>, CallbackError> + Send + Sync>

impl RewriteRule {
    #[must_use] pub fn new<F>(pattern: Pattern, rewrite: F) -> Self
    where F: Fn(&MatchBindings) -> Result<Option<Expression>, CallbackError> + Send + Sync + 'static;
                                                   // CHANGED: the rewrite returns Option; Ok(None) = declined
    #[must_use] pub fn with_guard<G>(self, guard: G) -> Self
    where G: Fn(&MatchBindings) -> Result<bool, CallbackError> + Send + Sync + 'static;
                                                   // CHANGED: composed into the rewrite; guards conjoin
    #[must_use] pub fn with_name(self, name: impl Into<Arc<str>>) -> Self;   // CHANGED: was &str
    #[must_use] pub fn name(&self) -> Option<&str>;                          // KEPT
    pub fn apply(&self, expression: &Expression)
        -> Result<Option<Expression>, CallbackError>;                        // NEW inherent (= Rule::apply); replaces apply_rewrite_rule
}
impl fmt::Debug for RewriteRule {}  // pattern and name; the callback is opaque. No `has_guard` field.
```

#### 2.7 `apply_rewrite_rules`, `RewriteOutcome`, `FiredRule`

```rust
pub fn apply_rewrite_rules<R: Rule>(expression: &Expression, rules: &[R])
    -> Result<RewriteOutcome, RewriteError>;          // CHANGED: generic over R (was &[RewriteRule])
```

`RewriteOutcome` keeps `output`, `into_output`, `is_changed` and `fired`, and gains `#[must_use]` on the type; only the `is_changed` docs change (5.3). `FiredRule` keeps `rule_index`, `name` and `Option<Arc<str>>`. Both types are unchanged otherwise.

`apply_rewrite_rule(rule, expr)` is REMOVED; use `rule.apply(&expr)`. That settles F-029's inconsistent argument order: the only remaining free function takes the expression first.

Crate-private seam used by `pass::expr`:

```rust
pub(crate) struct RuleRun { pub(crate) output: Result<Expression, RewriteError>, pub(crate) fired: Vec<FiredRule> }
pub(crate) fn run_rules<R: Rule>(expression: &Expression, rules: &[R]) -> RuleRun;
```

`apply_rewrite_rules` is `run_rules` plus the `ptr_eq` change check. The pass needs `run_rules` because it keeps the firings made before a failure.

#### 2.8 `RewriteError` (CHANGED, `pub`). F-014

```rust
#[derive(Debug)]
#[non_exhaustive]
pub enum RewriteError {
    #[non_exhaustive]
    Callback { rule_index: usize, rule_name: Option<Arc<str>>, source: CallbackError },
    #[non_exhaustive]
    Rebuild  { rule_index: usize, rule_name: Option<Arc<str>>, source: ExpressionBuildError },
}
impl RewriteError {
    #[must_use] pub fn rule_index(&self) -> usize;          // NEW
    #[must_use] pub fn rule_name(&self) -> Option<&str>;    // NEW
}
impl fmt::Display for RewriteError {}   // unchanged text, never repeats the source
impl Error for RewriteError {}          // source(): Callback -> the callback's own error (CallbackError::inner), Rebuild -> the build error
```

- `rule_name` changes from `Option<String>` to `Option<Arc<str>>`, the same type `FiredRule` uses, so building an error no longer allocates.
- The variant-level `#[non_exhaustive]` lets PAT-08's refused node be added later without breaking callers. It also means code outside the crate can no longer build these variants (see the test plan).
- `ExpressionBuildError` here is Expression's `Tree::RebuildError`. If B3 splits the builder errors (F-014, "per-operation error types"), this field takes the rebuild error B3 names.

#### 2.9 `fhy_core::pass::expr` (MOVED + CHANGED)

```rust
#[derive(Debug, Clone)]
pub struct RewriteRuleApplier<R = RewriteRule> { rules: Vec<R>, fired: Vec<FiredRule> }  // CHANGED: generic, moved

impl RewriteRuleApplier {                                 // R = RewriteRule, so `RewriteRuleApplier::NAME` infers
    pub const NAME: &'static str = "fhy_core.symbolic.expression.apply_rewrite_rules";           // NEW const (value unchanged)
    pub const DESCRIPTION: &'static str = "Apply a sequence of rewrite rules bottom-up over an expression tree."; // NEW const
}
impl<R: Rule> RewriteRuleApplier<R> {
    #[must_use] pub fn new(rules: impl IntoIterator<Item = R>) -> Self;   // CHANGED: generic
    #[must_use] pub fn rules(&self) -> &[R];                              // CHANGED: generic
    #[must_use] pub fn fired(&self) -> &[FiredRule];                      // KEPT
}
impl<R: Rule> CompilerPass<Expression> for RewriteRuleApplier<R> {}   // hook set and name type per B5

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ExpressionPrettyFormatter { options: FormatOptions }      // MOVED; API unchanged
impl ExpressionPrettyFormatter {
    #[must_use] pub fn new(options: FormatOptions) -> Self;
    #[must_use] pub fn options(&self) -> FormatOptions;
}
impl CompilerPass<Expression, String> for ExpressionPrettyFormatter {} // run = the B3 display API (`expr.display(options).to_string()`; `format_expression` at HEAD)

pub fn register_expression_passes(registry: &mut PassRegistry)
    -> Result<(), PassRegistrationError>;                // CHANGED: takes the owned registry (decision 7)
```

`RewriteRuleApplier::new([])` cannot infer `R`, because a type-parameter default does not apply in expression position (checked in a scratch crate). An empty applier is written `RewriteRuleApplier::<RewriteRule>::new([])`, or with a `let a: RewriteRuleApplier = …` annotation.

`R: Send` makes the applier `Send`, which decision 10's `Box<dyn CompilerPass + Send>` requires.

`PassRegistry`, its `register` method, and the shape of `PassRegistrationError` are B5's. `register_expression_passes` calls B5's register method once:
- name `RewriteRuleApplier::NAME`;
- description `RewriteRuleApplier::DESCRIPTION`;
- factory `|| RewriteRuleApplier::<RewriteRule>::new([])`.

It registers `ExpressionPrettyFormatter` under no name, as today. It follows B5's re-registration rule (at HEAD, the same type, name and description returns `Ok(())`).

### 3. Interface delta

All changes are breaking unless marked otherwise; S-8 allows every one. "p_stories" means `tests/pattern_stories.rs`, "p_props" `pattern_properties.rs`, "p_rewrite" `pattern_rewrite_stories.rs`, "p_user" `pattern_user_stories.rs`, "eps" `expression_pass_stories.rs`, "common" `tests/common/pattern.rs`. fhy-core-py has **no** call site for anything in this batch (grep: 0 hits).

| Change | Item | Before (HEAD) | After | Semver | Call sites |
|---|---|---|---|---|---|
| moved | module | `symbolic::expression::pattern` (`pattern/core.rs` + `rewrite.rs`) | `expr::pattern` (`matching.rs` + `rewrite.rs`, private) | breaking | every `use` in the 6 test files; doc tests in core.rs:120-137, 578-590, rewrite.rs:203-231, 468-504 |
| added | `Capture` | n/a (names were `&str`) | `Capture::new(&str)`, `name()`, identity `Eq`/`Hash` | non-breaking | new in all rule builders (common:155-222, p_props:269-283) |
| changed | `Pattern::capture` | `core.rs:160` `fn capture(name: &str, sub_pattern: Pattern) -> Result<Self, PatternError>` | `fn capture(&Capture) -> Self` + `fn captured_as(self, &Capture) -> Self` | breaking | p_props 3, p_rewrite 1, common 2, p_stories 1; wrappers `build_capture` (p_stories 34, p_rewrite 7, p_user 4, eps 3) and `build_capture_of` (p_stories 14, p_rewrite 2, p_user 2) |
| changed | `Pattern::literal` | `core.rs:182` `fn literal(value: Option<LiteralValue>) -> Self` | `literal(impl Into<LiteralValue>)`, `any_literal()` | breaking | p_stories 7, p_props 4, p_rewrite 2, common 1; wrapper `build_literal_pattern` (p_stories 24, p_rewrite 13, p_user 3, eps 2); rustdoc core.rs:129, rewrite.rs:215, 481 |
| changed | `Pattern::identifier` | `core.rs:190` `fn identifier(Option<Identifier>) -> Self` | `identifier(Identifier)`, `any_identifier()` | breaking | p_stories 8, p_rewrite 1 (434), eps 1 (95) |
| changed | `Pattern::unary` | `core.rs:197` `fn unary(Option<UnaryOperation>, Pattern) -> Self` | `unary(UnaryOperation, Pattern)`, `unary_any_operation(Pattern)` | breaking | p_stories 6, p_props 3, p_rewrite 1, common 2 |
| changed | `Pattern::binary` | `core.rs:208` `fn binary(Option<BinaryOperation>, Pattern, Pattern) -> Self` | `binary(op, l, r)`, `binary_any_operation(l, r)` | breaking | p_stories 23, p_props 5, p_rewrite 4, p_user 2, eps 1, common 4; rustdoc 3 |
| changed | `Pattern::piecewise` | `core.rs:228` `fn piecewise(Option<Vec<(Pattern, Pattern)>>, Pattern) -> Result<Self, PatternError>` | `piecewise(impl IntoIterator<Item=(Pattern,Pattern)>, Pattern) -> Self`, `piecewise_any_cases(Pattern)` | breaking | p_props 1 (360), common 1 (87); wrapper `build_piecewise_pattern` (p_stories 12) |
| changed | `Pattern::call` | `core.rs:249` `fn call(Option<&str>, Option<Vec<Pattern>>) -> Self` | `call(&str, args)`, `call_any_arguments(&str)`, `call_any_name(args)`, `any_call()` | breaking | p_stories 9 (1141-1269), p_props 1 (341) |
| changed | `Pattern::predicate` | `core.rs:262` `F: Fn(&Expression) -> Result<bool, CallbackError>` | `F: Fn(&Expression) -> bool`; fallible form is `try_predicate` | breaking | p_stories 8, p_rewrite 4 (724; the failing-predicate tests use `try_predicate`) |
| changed | `Pattern::alternatives` | `core.rs:281` `fn alternatives(Vec<Pattern>) -> Result<Self, PatternError>` | `alternatives(impl IntoIterator<Item=Pattern>) -> Self` | breaking | common 1 (74); wrapper `build_alternatives` (p_stories 11) |
| added | `Pattern::nothing`, `try_predicate` | n/a | see 2.2 | non-breaking | new tests |
| narrowed | `Pattern::match_under` | `core.rs:300` `pub fn match_under(&self, &Expression, &MatchBindings) -> Result<Option<MatchBindings>, CallbackError>` | removed; `pub(super) fn match_into(&self, &Expression, &mut MatchBindings) -> Result<bool, CallbackError>` | breaking | p_stories 5 (378-392, 1559-1570) |
| removed→method | `match_pattern` | `core.rs:648` `pub fn match_pattern(&Pattern, &Expression) -> Result<Option<MatchBindings>, CallbackError>` | `Pattern::matches(&self, &Expression)` | breaking | src: rewrite.rs:18, 586 and 12 doc lines; tests: p_stories/p_props/p_user/common (the `match_infallibly`/`expect_match` wrappers carry 91 p_stories uses) |
| removed→method | `does_pattern_match` | `core.rs:661` `pub fn does_pattern_match(&Pattern, &Expression) -> Result<bool, CallbackError>` | `Pattern::is_match(&self, &Expression)` | breaking | p_stories 10, p_props 5 |
| changed | `MatchBindings` | `core.rs:436-487`: `empty()`, `is_empty()`, `names()`, `get(&str)`, `has(&str)`, `try_bind(&str, &Expression) -> Option<MatchBindings>` | `new()`/`Default`, `is_empty()`, `len()`, `contains(&Capture)`, `get(&Capture)`, `iter()`, `Index<&Capture>`; `try_bind` becomes private `bind` | breaking | `empty` p_stories 24; `try_bind` p_stories 14; `has` p_stories 6; `names` p_stories 2, p_props 2; `get(&str)` via `expect_bound` p_stories 38, p_user 5, p_props 3, p_rewrite 1, eps 1, common 3 |
| removed | `PatternError` | `core.rs:543-569` enum `EmptyCaptureName`, `EmptyPiecewiseCases`, `EmptyAlternatives` | none (construction is infallible) | breaking | mod.rs:16; p_stories 8 (417-426, 950-958, 1418-1426, 1649-1670) |
| changed | `CallbackError` | `core.rs:592-639`: implements `Error` (`source` = inner's source); `From<ExpressionBuildError>`, `From<LiteralTextError>` | no `Error` impl; blanket `From<E: Error + Send + Sync + 'static>`; `From<CallbackError> for Box<dyn Error..>`; `downcast_ref` | breaking | p_stories 10 (1673-1760), p_rewrite 13, eps 4, p_props 3, common 5 (`expect_probe_error` uses `.inner().downcast_ref`) |
| added | `Rule` trait | n/a | see 2.5 | non-breaking | new tests |
| changed | `RewriteRule::new` | `rewrite.rs:244` `F: Fn(&MatchBindings) -> Result<Expression, CallbackError>` | `F: Fn(&MatchBindings) -> Result<Option<Expression>, CallbackError>` | breaking | p_rewrite 26, eps 6, common 5, p_props 4, p_user 2; rustdoc 2 |
| changed | `RewriteRule::with_guard` | `rewrite.rs:261`: replaces any earlier guard | composed into the rewrite; guards conjoin (behavior change) | breaking (behavior) | p_rewrite 10, eps 1 |
| changed | `RewriteRule::with_name` | `rewrite.rs:273` `fn with_name(self, name: &str)` | `impl Into<Arc<str>>` | non-breaking for `&str` callers | 26 test sites compile unchanged |
| removed→method | `apply_rewrite_rule` | `rewrite.rs:582` `pub fn apply_rewrite_rule(&RewriteRule, &Expression) -> Result<Option<Expression>, CallbackError>` | `RewriteRule::apply(&self, &Expression)` / `Rule::apply` | breaking | src rewrite.rs:137; p_rewrite 4 (`rewrite_root` helper :43, plus tests :198-360) |
| changed | `apply_rewrite_rules` | `rewrite.rs:623` `pub fn apply_rewrite_rules(&Expression, &[RewriteRule]) -> Result<RewriteOutcome, RewriteError>` | `apply_rewrite_rules<R: Rule>(&Expression, &[R])` | breaking only where `&[]` needs a type | p_props:498 (`&[]` needs `&[] as &[RewriteRule]`); 60 other test sites compile unchanged |
| changed | `RewriteOutcome::is_changed` semantics | `rewrite.rs:343-355`: an identity rewrite below the root counts as a change | an identity rewrite never fires (F-009) | behavior | p_rewrite:402-431, p_props:576-596, eps:173-186 |
| changed | `RewriteError` | `rewrite.rs:370-403`: variant fields matchable and constructible, `rule_name: Option<String>` | `#[non_exhaustive]` variants, `rule_name: Option<Arc<str>>`, `rule_index()`/`rule_name()` accessors, `source()` of `Callback` = user error | breaking | p_rewrite 12 (79-106, 959-1036), eps 5 (339-398) |
| moved + changed | `RewriteRuleApplier` | `symbolic::expression::pattern::RewriteRuleApplier` (`rewrite.rs:506`), `new(impl IntoIterator<Item = RewriteRule>)`, `rules() -> &[RewriteRule]` | `pass::expr::RewriteRuleApplier<R = RewriteRule>`, generic `new`/`rules`, `NAME`/`DESCRIPTION` consts | breaking | eps 21 (295 and 321 need `::<RewriteRule>`); src registration.rs:6, 38-42 |
| changed | pass diagnostic text | `rewrite.rs:560` `Applied rewrite rule {name:?}.` | `applied rewrite rule {name:?}` (S-4) | behavior | eps:216-240, rustdoc rewrite.rs:499-502 |
| moved | `ExpressionPrettyFormatter` | `symbolic::expression::ExpressionPrettyFormatter` (`pprint.rs:297-324`) | `pass::expr::ExpressionPrettyFormatter` | breaking | eps 11; src expression/mod.rs:34-36; rustdoc pprint.rs:280-295 |
| moved + changed | `register_expression_passes` | `symbolic::expression::register_expression_passes` (`registration.rs:37`) `pub fn register_expression_passes() -> Result<(), PassRegistrationError>` | `pass::expr::register_expression_passes(&mut PassRegistry) -> Result<(), PassRegistrationError>` | breaking | eps 6 (557-580); src expression/mod.rs:37; rustdoc registration.rs:23-34; future: fhy-core-py module init (decision 7) |
| removed | `registration.rs` | module | folded into `pass/expr.rs` | internal | none |

### 4. Encapsulation delta

- **`match_under`** (pub) becomes `Pattern::match_into`, `pub(super)`, visible only inside `expr::pattern`. **`try_bind`** (pub) becomes the private `MatchBindings::bind`. `truncate`/`clear` on the trail are `pub(super)`. Neither had a caller outside the crate.
- **`MatchBindings`** gains no public constructor that can bind a capture. Only a successful match produces non-empty bindings, so bindings always reflect a real match.
- **`Capture`** keeps its representation private (an `Arc` of a private name struct); identity is the whole contract.
- **`Pattern`** keeps its private kind; it changes from a `Box` tree to `Arc<PatternKind>`.
- **`RewriteError` variants** get `#[non_exhaustive]`, so code outside the crate can match them only with `..` and cannot build them. `RewriteError` itself was already `#[non_exhaustive]`.
- **Crate-private seam:** `run_rules`/`RuleRun` are `pub(crate)` and used only by `pass::expr`.
- **Layering:** `src/expr/**` no longer imports `crate::pass`. Today it imports it at rewrite.rs:20-23, pprint.rs:16 and registration.rs:3; the node.rs and screen.rs imports are B5's/B3's to move to `tree`. The removed `PassContext::new_standalone` use at rewrite.rs:627 was one of F-008's "crate-private back doors".
- **Module named `core`** (`pattern/mod.rs:11`, which shadows the `core` crate) is gone.
- **Blame helpers:** `find_refused_child_index` (rewrite.rs:43-48) hard-codes the piecewise child layout (`case_index * 2`). It moves next to `Expression::children` in node.rs as a `pub(crate)` method, so the child layout has one owner.

### 5. Behavior

#### 5.1 Captures and unification (F-019)

- `Pattern::capture(&x)` matches any expression and binds `x` to a handle to it. `p.captured_as(&x)` matches what `p` matches and binds `x` after `p`'s own captures, so inner captures come first, as today.
- Using the same `Capture` twice in one pattern means "must be equal". Two captures named `"x"` built by separate `Capture::new` calls are independent.
- **One literal equality.** Both literal patterns and capture unification use `LiteralValue`'s `==`: the crate's lawful, canonical equality, which `Expression ==` and `Hash` already use. So:
  - `Pattern::literal(5)` now matches `"05"` and `"5"`;
  - `Pattern::literal(f64::NAN)` matches every NaN literal;
  - `literal(0.0)` matches `-0.0`, as before;
  - `literal("1.5")` matches `"1.50"`;
  - `5`, `5.0`, `true` and `"5.0"` stay pairwise unequal.

  Why this relation rather than stored-form equality:
  - Stored-form equality is not reflexive (NaN), so `Pattern::literal(v)` would fail to match `Expression::from(v)`.
  - Stored-form equality would need a second structural-equality walk for unification.
  - F-026 (decision 6) normalizes literal spellings anyway, after which the two relations differ only on NaN.

  `x - x → 0` still rewrites `nan - nan` to `0`. That is the rule's unsoundness for IEEE floats, not the matcher's; see section 9.
- `Index<&Capture>` panics with the message `capture `{name}` is not bound` when the capture is unbound. That can happen only if the capture is not in the pattern, or sits under an alternative that did not match.

#### 5.2 Construction and matching

- Construction never fails:
  - `Pattern::nothing()` matches nothing.
  - `alternatives([])` behaves exactly like `nothing()`.
  - `piecewise([], o)` matches nothing, because a piecewise node always has at least one case.
  - An empty capture name is allowed.
- Matching is still anchored at the root, one-shot and committed-choice, with sub-patterns matched in the fixed order documented today.
- **Trail contract:** `match_into` returns `Ok(true)` with this pattern's captures appended to `bindings`, or `Ok(false)` with `bindings` truncated back to its length at entry. After `Err`, the contents of `bindings` are unspecified, and every public caller discards them.
  - An alternative saves the length, tries, and truncates on failure. No `MatchBindings` is cloned per alternative or per attempt (the audit's "Clones per attempt").
  - `matches` starts from `MatchBindings::new()`, which allocates only at the first bind.
- `is_match(e)` equals `matches(e)?.is_some()`.
- `predicate` runs its closure on every match attempt and cannot fail. `try_predicate`'s error ends the match and is returned unchanged.

#### 5.3 Rules, the walk and F-009

- `RewriteRule::apply(e)` does four things in order:
  1. matches the pattern at the root;
  2. runs the guards, which were composed in when the rule was built. The guard added last runs first, and the rule fires only if every guard returns `Ok(true)`;
  3. runs the rewrite;
  4. maps a result `ptr_eq` to `e` to `Ok(None)`.

  `Ok(None)` from the rewrite means "declined".
- `apply_rewrite_rules(e, rules)`, at each node in the existing bottom-up, share-once order:
  - It tries the rules in order on the node, rebuilt if a child changed.
  - A rule **fires** when it returns `Ok(Some(r))` with `!Expression::ptr_eq(&r, node)`. Only a firing is recorded in `fired()` and replaces the node.
  - A rule that declines, or that returns its input, is not recorded, and the next rule is tried.
  - Rule names are converted to `Arc<str>` once per rule per walk, not once per firing.
- **F-009 contract:**
  - A rule set in which every rule returns its input (or declines) leaves every node's handle in place. `is_changed()` is `false`, `fired()` is empty, and the output is `ptr_eq` to the input.
  - This holds at any depth. The rule-side fix alone prevents the parent rebuild, because the walk never receives `Some` for such a rule. B5's `rewrite_tree` fix, which treats a replacement `ptr_eq` to the visited node as `None`, is defense in depth, and it covers any `Rewriter`.
- `is_changed()` stays defined as `!ptr_eq(output, input)`. With the fix, `fired().is_empty()` implies `!is_changed()`.
  - The converse fails only in one case: a rule hands back a handle it held from outside the walk, and that handle is `ptr_eq` to the input root.
  - A rule that always builds a *fresh*, structurally equal node still counts as a change. The caller-driven fixpoint loop terminates for rules that return a subterm or their input; it does not terminate for rules that rebuild needlessly. This is documented on `apply_rewrite_rules`.
- `RewriteRuleApplier::did_change` (`!ptr_eq(input, output)`) was re-checked against F-009. It is consistent: an identity-only run outputs its input handle, so the pass reports no change, and a `FixpointPassGroup` converges.

#### 5.4 Rebuild blame (F-039 decision: keep precise blame, use the identity hasher)

- The walk keeps `HashMap<NodeIdentity, (Expression, usize), BuildIdentityHasher>`. It maps each replacement a rule returned to that rule's index, and holds the replacement handle so its address cannot be reused during the walk.
- It is read only when a rebuild fails. The blame rules are unchanged: first the rule that produced the refused child, then the rule behind the last rewritten child, then the last firing.
- **Why not drop the map:**
  - The only rebuild failure a rewrite walk can hit is `NonBooleanConditionLiteral`, and its refused child is exactly a replacement some rule returned. Precise blame is cheap to keep, and two existing tests pin it (p_rewrite:900, :931).
  - With the identity hasher, the per-firing cost is one multiply-hash insert and one `Arc` increment. The memory held is of the same order as `fired()` itself.
  - Dropping the map would coarsen blame to "the last rule that fired", which would name the wrong rule in exactly the shared-condition case those tests cover.

#### 5.5 The passes and registration

- `RewriteRuleApplier` run:
  - It calls `run_rules` with its rules.
  - For each firing of a named rule, in walk order, it reports an `Info` diagnostic `applied rewrite rule "<name>"`, with the name escaped as `Debug` writes a string.
  - It stores the firings; after a failed run it stores the firings made before the failure.
  - On failure it fails with the `RewriteError` as the `PassFailure`, which is the `PassError`'s source.
  - Change is `!ptr_eq`, and a skipped run outputs its input (implemented through B5's reworked skip hook, F-022).
  - `name()` is `RewriteRuleApplier::NAME` and `description()` is `DESCRIPTION`, whether or not the pass is registered.
- `ExpressionPrettyFormatter`: a run is the B3 display text under its options. Every run counts as a change, and a skipped run has no output. Its name follows B5's default-name rule (at HEAD, the type name `"ExpressionPrettyFormatter"`).
- `register_expression_passes(&mut registry)` registers the applier in the given registry only; there is no global state. Registering into two registries gives two independent registrations.

### 6. Error and panic model

| Condition | Result |
|---|---|
| `try_predicate`, guard, rewrite or native `Rule::apply` returns `Err` | `matches`/`is_match`/`RewriteRule::apply` return that `CallbackError` unchanged. The walk stops with `RewriteError::Callback { rule_index, rule_name, source }`, and `source()` returns the user's own error. |
| A node refuses rebuilding around rewritten children | `RewriteError::Rebuild { rule_index, rule_name, source }` with the blame described in 5.4 |
| Empty capture name, empty alternatives, empty piecewise cases | **no error** (was `PatternError`). The pattern builds; the last two match nothing. |
| `bindings[&c]` for an unbound `c` | **panic** `capture `{name}` is not bound` (documented under `# Panics`) |
| Rebuild failure with no firing recorded | unreachable, because a node is rebuilt only around a rewritten child. It stays an `expect` with the documented reason (rewrite.rs:123). |
| `RewriteRuleApplier` run failure | the `PassError` produced by B5's pass machinery, with `source()` = `RewriteError` |

`Display` texts stay lowercase one-liners without the source:
- `rewrite rule {i} failed`
- `rewrite rule {i} ({name}) failed`
- `rebuilding a node after rewrite rule {i} failed`
- `rebuilding a node after rewrite rule {i} ({name}) failed`

`CallbackError`'s `Display` and `Debug` are the wrapped error's. The printed chain for a callback failure is unchanged: the wrapper used to display as the inner error, and now the inner error itself is the source.

### 7. Non-goals

- A bounded `apply_rewrite_rules_to_fixpoint(expr, rules, max_passes)`. F-009 lists it as optional; `FixpointPassGroup` covers the pass case (see D-B4-10).
- `RewriteRule<'a>` with borrowing closures (PAT-06, optional). Borrowing rules implement `Rule` instead.
- The refused node or children as fields of `RewriteError::Rebuild` (PAT-08). The `#[non_exhaustive]` variants leave room to add them. They also need an opaque `Debug`, because a node field would make `Debug` of the error exponential on a DAG.
- Iterative `Drop`/`Clone`/`Debug` for very deep `Pattern`s (PAT-14, hand-written patterns only).
- A Python-side `Capture` table. The binding exposes no patterns today; that work belongs to the binding.
- Reusing one scratch `MatchBindings` across a whole walk. The `Rule::apply` signature has no scratch parameter, so this costs at most one allocation per match attempt that binds.
- Test binary consolidation (F-034), which belongs to the tests batch. The file names below are HEAD's; if that batch moves them, the test names carry over.

### 8. Test plan

Shared helper changes in `tests/common/pattern.rs`:
- **Delete** `build_capture`, `build_capture_of`, `build_alternatives` and `build_piecewise_pattern`. They existed only to hide the fallible constructors.
- **Delete** `build_literal_pattern`, which `Pattern::literal(v)` now covers directly.
- **Delete** `expect_bound`, which `Index` replaces.
- **Modify** `rewrite_to_capture(&Capture)` and `rewrite_to_literal` to return `Ok(Some(..))`.
- **Modify** the five rule builders to use `Capture::new`, and `expect_probe_error` to use `CallbackError::downcast_ref`.
- **Modify** `match_infallibly`/`expect_match` to call `pattern.matches`.

#### 8.1 `tests/pattern_stories.rs`

Keep, with API updates only:
- every wildcard, capture, identifier, unary, binary, piecewise, call, predicate and alternatives test not listed below;
- the four deep-stack tests (1769-1850);
- `callback_error_new_from_text_displays_the_text`.

Modify:
- `match_bindings_empty_binds_no_name` → `match_bindings_new_binds_no_capture`. Assert `is_empty`, `len() == 0`, `get(&x) == None`, and `MatchBindings::default() == MatchBindings::new()`.
- `match_bindings_get_returns_none_for_an_unbound_name` → `..._for_a_capture_the_pattern_lacks`.
- `match_bindings_has_reports_only_bound_names` → `match_bindings_contains_reports_only_bound_captures`.
- `match_bindings_names_lists_names_in_binding_order` → `match_bindings_iter_lists_captures_in_binding_order`.
- The equality and hash tests (282-350) build their bindings by matching, not by `try_bind`. For binding order, match `binary(Add, capture(x), capture(y))` on `1 + 2` against `binary(Subtract, capture(y), capture(x))` on `2 - 1`.
- `pattern_capture_binds_under_the_given_name` → `pattern_capture_binds_exactly_its_handle`.
- `pattern_literal_without_value_matches_every_literal` → `pattern_any_literal_matches_every_literal`.
- `pattern_literal_matches_an_exactly_stored_value` → `pattern_literal_matches_an_equal_literal`. It gains the cases `integer_and_integer_text`, `integer_text_and_integer`, `integer_texts_spelled_differently`, `decimal_texts_spelled_differently` and `nan_and_nan`.
- `pattern_literal_rejects_another_stored_form` → `pattern_literal_rejects_an_unequal_literal`. It drops those five cases, because they are now matches, and keeps the other-bucket and other-value cases. **The contract changes here; nothing is weakened:** each case moves to the opposite assertion.
- Predicate tests use `predicate(|_| bool)`. `pattern_predicate_error_is_returned_from_the_match` and `..._stops_the_match` use `try_predicate`.
- `callback_error_wraps_an_error_transparently` → `callback_error_downcasts_to_the_wrapped_error`.
- `callback_error_from_expression_build_error_wraps_it` and `callback_error_from_literal_text_error_wraps_it` → one `callback_error_converts_from_any_error_with_question_mark`, parametrized over `ProbeError`, `ExpressionBuildError` and `LiteralTextError`.
- `does_pattern_match_is_true_on_a_match` and `..._false_...` → `pattern_is_match_*`. The `match_pattern_*` tests → `pattern_matches_*`.
- **F-037, bare `is_some()` (14 sites: 711, 712, 739, 766, 825, 854, 970, 1145, 1171, 1256, 1291, 1321, 1407, 1408).** Each pattern there captures nothing, so assert `assert_eq!(result, Some(MatchBindings::new()))`. That is stronger: it also asserts that no capture leaked. The `is_none()` sites with a `got {result:?}` message stay; they already print the value.

Delete (one line each):
- `match_bindings_try_bind_records_an_unbound_name`: `try_bind` is no longer public; `pattern_capture_binds_the_matched_node` covers it.
- `match_bindings_try_bind_leaves_the_receiver_unchanged`: it tests the by-value semantics of a removed API.
- `match_bindings_try_bind_confirming_an_equal_expression_returns_the_same_bindings`: `pattern_capture_repeated_over_equal_operands_matches` covers it.
- `match_bindings_try_bind_confirming_keeps_the_first_bound_handle`: `pattern_capture_repeated_over_equal_literals_keeps_the_first` covers it.
- `match_bindings_try_bind_refuses_a_different_expression`: `pattern_capture_repeated_over_different_operands_fails` covers it.
- `match_bindings_try_bind_confirms_a_structurally_equal_compound`, `..._confirms_an_equal_literal_in_another_form` and `..._refuses_a_literal_of_another_bucket`: re-expressed as the pattern-level tests in 8.6.
- `pattern_wildcard_match_under_returns_the_given_bindings`: `match_under` is gone; the sequence tests cover threading.
- `pattern_capture_rejects_an_empty_name`: empty names are allowed now (replaced by `pattern_capture_accepts_an_empty_name`).
- `pattern_piecewise_rejects_an_empty_case_list`: construction is infallible (replaced by `pattern_piecewise_with_no_cases_matches_nothing`).
- `pattern_alternatives_rejects_an_empty_list`: same reason (replaced by `pattern_alternatives_of_none_matches_nothing`).
- `match_pattern_equals_match_under_from_empty_bindings`: `match_under` is gone.
- `pattern_error_display_describes_the_refusal`: `PatternError` is removed.
- `callback_error_source_is_the_wrapped_errors_source`: `CallbackError` no longer implements `Error`. The chain moves to `rewrite_error_callback_source_is_the_callers_error` (8.3).

#### 8.2 `tests/pattern_properties.rs`

- Keep, with API updates: `wildcard_pattern_matches_every_expression`, `mirror_pattern_with_another_root_operation_does_not_match`, `repeated_capture_matches_exactly_equal_operands` (its oracle `left == right` is now literally the matcher's relation), and the three `neutral_rules_*` properties.
- Modify:
  - `mirroring_pattern_matches_and_binds_every_leaf` records `(Capture, Expression)` pairs and asserts with `iter()` and `Index`.
  - `does_pattern_match_agrees_with_match_pattern` → `is_match_agrees_with_matches`.
  - `apply_rewrite_rules_with_no_rules_is_the_identity` needs `&[] as &[RewriteRule]`.
- Delete `identity_rewrite_changes_exactly_trees_it_fires_below_the_root`: it pins F-009's bug. Its replacement is below.

#### 8.3 `tests/pattern_rewrite_stories.rs`

Keep, with API updates:
- the walk-order, sharing, deep-stack, failure and blame tests (454-956, 1076-1128);
- `apply_rewrite_rules_supports_a_caller_driven_fixpoint`;
- `fired_rule_records_index_and_name_in_walk_order`;
- `rewrite_rule_clone_shares_the_callbacks`;
- `rewrite_rule_with_name_replaces_an_earlier_name`.

Modify:
- `rewrite_rule_new_is_unnamed_and_unguarded` → `rewrite_rule_new_is_unnamed`.
- The `apply_rewrite_rule_*` tests (198-360) → `rewrite_rule_apply_*`, calling `rule.apply(&e)`. `..._returns_the_predicate_error` uses `try_predicate`.
- `rewrite_rule_with_guard_replaces_an_earlier_guard` → `rewrite_rule_with_guard_requires_every_guard`. This is a behavior change: `with_guard(false).with_guard(true)` no longer fires. Add a case that asserts the guard added last runs first.
- `apply_rewrite_rules_with_an_identity_rewrite_at_the_root_is_unchanged`: now also `assert!(outcome.fired().is_empty())`, where HEAD asserts `[(0, Some("x -> x"))]`.
- `apply_rewrite_rules_rewrites_a_doubling_dag_once_per_distinct_node` (**F-037, line 713**): the failure arm `panic!("expected a product, got {node:?}")` Debug-prints a node of a 2^64-occurrence DAG and would never finish. Change it to `panic!("level {level} is not a product")`, printing no node.
- `rewrite_error_display_describes_the_failure`: the variants cannot be built outside the crate any more. It becomes one `Display` table over four errors produced by real walks (unnamed/named callback, unnamed/named rebuild), with the same four expected strings. Nothing is weakened.
- `rewrite_error_callback_source_is_the_callback_error` → `rewrite_error_callback_source_is_the_callers_error`: it downcasts `source()` straight to `ProbeError`, using an error produced by a walk.
- `rewrite_error_rebuild_source_is_the_build_error`: same change, produced by a walk.
- The `expect_rebuild_error`/`expect_callback_error` helpers match with `..` and read `rule_name` as `Option<&str>` through the accessors.

Delete:
- `rewrite_rule_debug_names_the_rule` (**F-035**): `Debug` text is not a contract, and `rewrite_rule_with_name_replaces_an_earlier_name` covers `name()`.
- `apply_rewrite_rules_with_an_identity_rewrite_below_the_root_is_changed`: it pins F-009's bug. Its replacement is below.

#### 8.4 `tests/pattern_user_stories.rs`

Keep all eight stories, with API updates only. The `x - x -> 0` rule uses one `Capture` twice.

#### 8.5 `tests/expression_pass_stories.rs`

Imports move to `fhy_core::pass::expr`. Keep:
- `rules_are_the_rules_it_was_built_with`
- `execute_matches_apply_rewrite_rules`
- `..._without_a_firing_...`
- `..._reports_a_change_...`
- `fired_lists_the_firings_of_the_last_run`
- `does_not_report_unnamed_firings`
- the two sharing tests
- `runs_in_a_pass_manager`
- `converges_in_a_fixpoint_pass_group`
- the formatter options and default tests

Modify:
- `rewrite_rule_applier_execute_with_an_identity_rewrite_at_the_root_is_unchanged`: it asserts `fired()` is empty and there are no diagnostics, where HEAD asserts `[(0, None)]`.
- `rewrite_rule_applier_reports_each_named_firing`: the expected text becomes `applied rewrite rule "x + 0 -> x"`.
- `rewrite_rule_applier_name_and_description_are_its_registered_ones`: it asserts against `RewriteRuleApplier::NAME`/`DESCRIPTION`, and pins the literal strings once. It needs `RewriteRuleApplier::<RewriteRule>::new([])`, and so does `did_change_compares_identity` (321).
- `rewrite_rule_applier_noop_output_is_the_input` and `expression_pretty_formatter_has_no_noop_output`: they follow B5's skip-hook API and B5's replacement for the fake-pass `run_with_pass_context` (F-008). `has_no_noop_output` asserts the error kind, not the text `"the pass has no no-op output"` (**F-035**).
- `rewrite_rule_applier_execute_fails_with_the_callback_error` and `..._with_the_rebuild_error`: they match `RewriteError` with `..` and use `rule_name()`. The guard signature is unchanged, but the failing-rewrite case returns `Result<Option<Expression>, _>`. The `PassError` assertions (`is_execution_failure`, `failed_hook`) follow B5's `PassError::kind()`.
- `expression_pretty_formatter_execute_matches_format_expression`: its oracle becomes B3's display API.
- `expression_pretty_formatter_name_is_its_type_name`: kept if B5 keeps type-name defaults, otherwise it follows B5.
- `register_expression_passes_registers_the_rule_applier`: it registers into a local `PassRegistry::new()` twice and reads the registration back from that registry. **It drops `info.type_name().ends_with("::RewriteRuleApplier")`** (F-035: `type_name` output is unstable). That is a weakening and is called out here. The remaining assertions (the created pass's name, and the identity run on `a + 0`) identify the registered type behaviorally. The file header's "only this test registers a pass" comment goes away, because nothing is global.

Optional strengthening: `rewrite_rule_applier_rewrites_a_doubling_dag_once_per_distinct_node` could use `is_doubling_dag_over(outcome.output(), &a, 64)` instead of structural `==`, which would also check that the output shares its nodes.

#### 8.6 Regression test per fixed finding

| Finding | Regression test (file) |
|---|---|
| F-009 (walk) | `apply_rewrite_rules_with_an_identity_rewrite_below_the_root_is_unchanged` (p_rewrite): `-x` with `capture(ident) -> itself`. Output `ptr_eq` input, `!is_changed()`, `fired()` empty, and the operand handle kept. |
| F-009 (property) | `identity_rewrite_never_fires_and_keeps_the_input` (p_props): for any tree, a rule mapping every literal to itself leaves output `ptr_eq` input, `fired()` empty and `!is_changed()` |
| F-009 (fixpoint) | `apply_rewrite_rules_caller_driven_fixpoint_terminates_with_an_identity_rule` (p_rewrite): rules `[x -> x, x + 0 -> x]` on `(a + 0) + 0`. The loop stops, with the walk count asserted. |
| F-009 (rule order) | `apply_rewrite_rules_tries_the_next_rule_after_an_identity_rewrite` (p_rewrite): `[x -> x, x + 0 -> x]` fires rule 1 only |
| F-009 (pass) | `rewrite_rule_applier_converges_in_a_fixpoint_group_with_an_identity_rule` (eps): iteration changes are `[true, false]` |
| F-009 (root `apply`) | `rewrite_rule_apply_returning_its_input_declines` (p_rewrite) |
| F-019 typed captures | `pattern_captures_with_the_same_name_are_independent` (p_stories): `binary(capture(Capture::new("x")), capture(Capture::new("x")))` matches `1 - 2` |
| F-019 infallible construction | `pattern_nothing_matches_no_expression`, `pattern_alternatives_of_none_matches_nothing`, `pattern_piecewise_with_no_cases_matches_nothing`, `pattern_capture_accepts_an_empty_name` (p_stories) |
| F-019 one equality | `pattern_literal_and_repeated_capture_agree_on_literal_equality` (p_stories, rstest over `5/"05"`, `nan/nan`, `0.0/-0.0`, `"1.5"/"1.50"`, `1/1.0`, `1/true`): for each pair, `literal(a).is_match(b)` equals `binary(capture(x), capture(x)).is_match(a - b)`. Plus property `literal_pattern_agrees_with_capture_unification` (p_props, arbitrary literal pairs, oracle `LiteralValue ==`). |
| F-019 trail | `pattern_alternatives_restore_bindings_after_a_later_sibling_fails` (p_stories): `binary(alternatives([captured_as(x) of literal, capture(y)]), literal(0))` on `5 + 1` does not match and leaves no bindings. The existing `pattern_alternatives_discards_captures_of_a_failed_alternative` is kept. |
| F-019 `Index` panic | `match_bindings_index_panics_for_an_unbound_capture` (p_stories, `#[should_panic(expected = "is not bound")]`) |
| F-014 (`RewriteError`) | `rewrite_error_callback_source_is_the_callers_error`, `rewrite_error_accessors_report_rule_index_and_name` (p_rewrite) |
| F-014 / PAT-07 (`CallbackError`) | `callback_error_converts_from_any_error_with_question_mark`, `callback_error_downcasts_to_the_wrapped_error` (p_stories); `callback_error_converts_into_a_boxed_error` (p_stories: `?` from a `Result<_, CallbackError>` in a `-> Result<(), Box<dyn Error>>` fn) |
| F-029 | `pattern_matches_*`, `pattern_is_match_*`, `rewrite_rule_apply_*` (renamed tests); `match_bindings_iter_lists_captures_in_binding_order` |
| F-039 | `apply_rewrite_rules_blames_the_right_rule_after_a_discarded_replacement` (p_rewrite). An inner rule's replacement is thrown away by a parent rule, and a later refused condition must still blame the rule that produced it. This guards the identity-keyed map against address reuse. The two existing blame tests are kept. |
| PAT-06 | `apply_rewrite_rules_accepts_a_native_rule_borrowing_its_context` (a `struct` rule holding `&HashMap<Identifier, i64>`); `apply_rewrite_rules_accepts_a_mixed_list_of_boxed_rules` (`Vec<Box<dyn Rule>>` with a `RewriteRule` and a native rule); `rewrite_rule_apply_returns_none_when_the_rewrite_declines` (p_rewrite); `rewrite_rule_applier_runs_native_rules` (eps) |
| Decision 7 | `register_expression_passes_into_two_registries_is_independent` (eps) |
| F-035/F-037 | the modifications above (line 713 arm, 14 `is_some` sites, deleted `Debug` assertion, dropped `type_name` assertion, `noop` text) |

#### 8.7 New story, property and adversarial cases

- **Story:** `algebraic_simplifier_with_a_normalizing_rule_reaches_a_fixpoint` (p_user). The simplifier set plus a normalizer that returns its input when the node is already normal. The caller loop terminates. This is the F-009 user story from the audit.
- **Property:** `failed_matches_leave_no_bindings`. For random patterns built from alternatives and captures over random trees, `matches` either returns bindings whose captures all occur in the pattern, or `None`. After a failed alternative inside a larger match, no binding from that alternative appears.
- **Adversarial:**
  - `pattern_alternatives_of_many_failing_branches_matches_in_linear_time`: 10 000 alternatives over a small tree, with no allocation growth. Assert on the result only, not on timing.
  - The existing 4000-level deep pattern on a 16 MiB stack now runs with `Arc` children.
  - `apply_rewrite_rules_with_an_identity_rule_on_a_doubling_dag_is_unchanged`: 64 levels. The output is `ptr_eq` to the input, and nothing prints the DAG.

### 9. Findings covered, decisions for sign-off, and possible misjudgments

Covered:
- **F-009**, rule side: 5.3. The `rewrite_tree` side is B5's.
- **F-014**, the pattern and rewrite part: `PatternError` removed; `CallbackError` redesigned; `RewriteError` variant fields `#[non_exhaustive]`; `rule_name: Option<Arc<str>>`.
- **F-019**: every bullet.
- **F-029**, pattern part: methods; `MatchBindings::new`/`contains`/`iter`; no module named `core`.
- **F-039**, rewrite blame: identity hasher.
- **PAT-06** (the `Rule` trait, `Ok(None)` = declined, guards composed into the rewrite) and **PAT-07** (erased error, with reasons).
- **PAT-12**, **PAT-13** (the trail, `Arc` patterns) and **PAT-14** (the `with_name` type, the `core` module name, the `rule_name` type).
- **F-035/F-037**, pattern tests: 8.1, 8.3, 8.5.

Decisions the user should sign off:
- **D-B4-1** The passes and registration go to `fhy_core::pass::expr`, not `fhy_core::expr::passes` (reasons in section 2).
- **D-B4-2** One literal equality, the canonical `LiteralValue ==`. `Pattern::literal(5)` starts matching `"05"`, and `literal(nan)` starts matching NaN.
- **D-B4-3** `CallbackError` stays type-erased and stops implementing `std::error::Error` (anyhow-style), so the blanket `From<E: Error>` is possible. This departs from S-5's shape rule for this one transport type.
- **D-B4-4** `with_guard` conjoins guards (the one added last runs first) instead of replacing the earlier guard. This follows from composing the guard into the rewrite.
- **D-B4-5** `RewriteRule::new`'s rewrite returns `Result<Option<Expression>, CallbackError>` (a single callback shape). Every always-firing rewrite gains `Some(..)`.
- **D-B4-6** Rebuild blame stays precise, using the identity hasher, rather than dropping the per-firing map (reasons in 5.4).
- **D-B4-7** `RewriteRuleApplier<R = RewriteRule>` is generic. The cost is a type annotation for an empty rule list.
- **D-B4-8** The registered pass name stays `"fhy_core.symbolic.expression.apply_rewrite_rules"`, even though Rust no longer has that path. It is the by-name key that Python drivers and Python's own registration use (decision 7), and renaming it gains nothing. The alternative is `"fhy_core.expr.apply_rewrite_rules"`.
- **D-B4-9** The pass diagnostic text becomes `applied rewrite rule "<name>"` under S-4's Rust conventions. S-4 names error messages only; this spec extends the rule to diagnostics.
- **D-B4-10** No bounded fixpoint helper for the free function.
- **D-B4-11** Capture identity is the handle, and empty names are allowed.

Findings I think were misjudged, or need a note:
- **F-019 "two equalities, so `nan - nan → 0` fires":** this is only half right. The two relations do disagree, and that is fixed. But once they are unified on the lawful relation, `nan - nan → 0` *still* fires. Structural equality of expressions is not IEEE equality: `x - x → 0` is unsound for floats (NaN, ±inf) whatever the matcher does. A rule that needs IEEE semantics must guard on it. Unifying on stored-form equality would stop that firing only by making the matcher non-reflexive.
- **F-009 "fixpoint loops never terminate":** the fix makes loops terminate for rules that return their input or a subterm. Rules that rebuild a structurally equal fresh node still loop, because the change test is `ptr_eq` by design. This is documented rather than fixed; a structural-equality change test would cost a full comparison per firing.
- **PAT-01's suggested storage `Vec<Option<Expression>>` indexed by capture id:** rejected (2.3). It needs globally unique dense ids (new global state, against decision 2) or per-pattern id sets (easy to misuse).
- **Path mismatch:** decision 13's example path `fhy_core::expr::Pattern` conflicts with S-1's `fhy_core::expr::pattern` module. This spec follows S-1, with `fhy_core::expr::pattern::Pattern` as the single path. The user may prefer `expr::Pattern`.
- **F-039 wording "drop the per-firing map":** for this crate, dropping it would lose correct blame in the shared-condition case (p_rewrite:931). The identity hasher alone addresses the perf nit.

Coordination:
- **With B5:**
  - `rewrite_tree` treating a replacement `ptr_eq` to its input as `None` (F-009);
  - `Rewriter<N, C = ()>` and `fhy_core::tree` (F-008);
  - the location of `NodeIdentity` and `BuildIdentityHasher`;
  - `PassRegistry` and `PassRegistrationError`;
  - the `CompilerPass` skip hook and name type (F-022, F-039);
  - the default pass names;
  - `PassError::kind()`.
- **With B3:**
  - the `Expression` display API behind `ExpressionPrettyFormatter`;
  - the name of the rebuild error type;
  - F-026 literal normalization, which makes D-B4-2's text/number cases moot;
  - the call-name type under F-024.

---

## B5: `fhy_core::pass` and `fhy_core::tree`

Batch scope: `rust/fhy-core/src/pass_infrastructure/*` (becomes
`fhy_core::pass`), the new foundation module `fhy_core::tree` (S-1), and the
tests `tests/pass_infrastructure_*.rs`, `tests/common/pass_ir.rs` and
`tests/common/tree_ir.rs`. Findings: F-006 (with decision 7), F-008, F-009
(tree side), F-014 (pass and tree errors), F-016 (partial), F-017 (with
decision 10), F-022 (partial), F-039 (pass nits), F-031 and F-035 (pass
tests). Every "before" below was checked against HEAD `412e234`.

### 1. Summary

The pass infrastructure stops depending on process-global state. Pass
names become pure `Cow<'static, str>` values, the registry becomes an owned
`PassRegistry` keyed by type identity, run statistics come back per run, and
stored passes are `Send`. The tree traversals move down to a new foundation
module, `fhy_core::tree`. They are generic over a context `C` that the caller
chooses (default `()`), so the IR layer no longer depends on the pass layer.
A rewrite that returns a node identical to its input now counts as no change.
Errors become structured, matchable and `#[non_exhaustive]`, and the analysis
cache becomes merge-only.

### 2. Desired public interface

Visibility is `pub` unless stated. Each item has exactly one public path.
There are no glob re-exports. The two `mod.rs` files list every re-export
explicitly.

#### 2.1 Module layout (S-1)

```
src/tree/mod.rs        docs + `pub use` of the items below (one path each)
src/tree/node.rs       NodeIdentity, NodeHandle, Tree
src/tree/hash.rs       IdentityHasher, BuildIdentityHasher        (pub(crate))
src/tree/walk.rs       TraversalOrder, TreeVisitor, walk_tree
src/tree/rewrite.rs    Rewriter, rewrite_tree, RewriteTreeError

src/pass/mod.rs        docs + `pub use`
src/pass/compiler_pass.rs  CompilerPass, ExecutePass, PassOutcome, PassFailure, short_type_name, lifecycle (private)
src/pass/context.rs    PassContext
src/pass/analysis.rs   Analysis, AnalysisMarker, AnalysisCache (pub(in crate::pass))
src/pass/preserved.rs  AnalysisId, PreservedAnalyses
src/pass/error.rs      PassError, PassErrorKind, FailureClass, PassHook, VerificationPoint
src/pass/registry.rs   PassRegistry, PassInfo, PassRegistrationError, CreatePassError
src/pass/manager.rs    PassManager, FixpointPassGroup, PassManagerResult, PipelineRecord, PassRunRecord, FixpointGroupRecord, FixpointIterationRecord
src/pass/validation.rs Validator, PassValidator, ValidationManager, ValidatorRecord
src/pass/adapters.rs   WalkPass, RewritePass
```

- The file is `compiler_pass.rs`, not `pass.rs`, because `pass::pass` trips
  `clippy::module_inception`.
- **Dependencies.** `tree` depends only on `std`. `pass` depends on `tree`,
  `diagnostic` and `identifier`. `pass` has **no** dependency on `expr`, so
  B4 may put `RewriteRuleApplier`, `ExpressionPrettyFormatter` and
  `register_expression_passes` in `fhy_core::pass` or in an `expr::passes`
  module without creating a cycle. B4 owns that choice.
- **Decision B5-1.** `NodeIdentity` and `NodeHandle` move to
  `fhy_core::tree`, together with the items S-1 lists. `Tree: NodeHandle`,
  and `expr` implements both for `Expression`. If they stayed in `pass`,
  `expr` would still depend on `pass`, which is the F-008 layering bug.

#### 2.2 `fhy_core::tree`

```rust
/// NEW path (was pass_infrastructure::NodeIdentity); unchanged otherwise.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NodeIdentity(usize);
impl NodeIdentity {
    #[must_use] pub fn of_arc<T: ?Sized>(node: &Arc<T>) -> Self;
}

/// NEW path; bounds unchanged (decision 10: IR stays Send + Sync).
pub trait NodeHandle: Clone + Send + Sync + 'static {
    #[must_use] fn identity(&self) -> NodeIdentity;
}

/// NEW path; items unchanged.
pub trait Tree: NodeHandle {
    type RebuildError: Error + Send + Sync + 'static;
    fn children(&self) -> impl Iterator<Item = &Self>;
    fn rebuild_with_children(&self, children: Vec<Self>) -> Result<Self, Self::RebuildError>;
    fn is_shared(&self) -> bool { true }
}

/// NEW path; unchanged. Stays exhaustive: two orders, and a walk matches on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TraversalOrder { #[default] Pre, Post }

/// CHANGED: gains the context parameter `C`, and every hook takes `&mut C`.
pub trait TreeVisitor<N: Tree, C: ?Sized = ()> {
    type Error;
    fn before_visit(&mut self, node: &N, cx: &mut C) -> Result<(), Self::Error> { Ok(()) }
    fn visit(&mut self, node: &N, cx: &mut C) -> Result<(), Self::Error> { Ok(()) }
    fn after_visit(&mut self, node: &N, cx: &mut C) -> Result<(), Self::Error> { Ok(()) }
    fn walks_children(&mut self, node: &N) -> bool { true }
}

/// CHANGED: generic over `C`.
/// # Errors: the first error a hook returns; no hook runs after it.
pub fn walk_tree<N, C, V>(visitor: &mut V, root: &N, order: TraversalOrder, cx: &mut C)
    -> Result<(), V::Error>
where N: Tree, C: ?Sized, V: TreeVisitor<N, C> + ?Sized;

/// CHANGED: gains `C`.
pub trait Rewriter<N: Tree, C: ?Sized = ()> {
    type Error;
    fn rewrite(&mut self, node: &N, cx: &mut C) -> Result<Option<N>, Self::Error>;
}

/// CHANGED: generic over `C`; F-009 identity rule (see 5.2).
/// # Errors: RewriteTreeError::Rewrite for the first rewriter error,
/// RewriteTreeError::Rebuild for the first refused rebuild.
pub fn rewrite_tree<N, C, R>(rewriter: &mut R, root: &N, cx: &mut C)
    -> Result<N, RewriteTreeError<N, R::Error>>
where N: Tree, C: ?Sized, R: Rewriter<N, C> + ?Sized;

/// CHANGED: #[non_exhaustive] on the enum and on `Rebuild`; Display changes (5.6).
#[non_exhaustive]
pub enum RewriteTreeError<N: Tree, E> {
    Rewrite(E),
    #[non_exhaustive]
    Rebuild { node: N, children: Vec<N>, source: N::RebuildError },
}
impl<N: Tree, E: fmt::Debug> fmt::Debug for RewriteTreeError<N, E>;     // unchanged shape
impl<N: Tree, E: fmt::Display> fmt::Display for RewriteTreeError<N, E>; // CHANGED bound: E: Display
impl<N: Tree, E: Error + 'static> Error for RewriteTreeError<N, E>;     // CHANGED source() for Rewrite

// pub(crate), MOVED from pass_infrastructure::tree:
pub(crate) struct IdentityHasher(u64);
pub(crate) type BuildIdentityHasher = BuildHasherDefault<IdentityHasher>;
```

Two ways to write a visitor or rewriter:

- **No context:** `impl TreeVisitor<Expression> for FreeIdentifierCollector`,
  called with `&mut ()`.
- **Any context:** `impl<C: ?Sized> TreeVisitor<ToyTree, C> for
  RecordingVisitor`. The same value then works for a direct walk and inside
  a `WalkPass`.

A visitor that needs the pass context writes `impl TreeVisitor<N,
PassContext<'_>> for V`. The elided impl-header lifetime makes this generic
over every `'a`, which is what `WalkPass` requires. A probe on rustc 1.98
confirms this compiles.

#### 2.3 `fhy_core::pass`: passes and names

```rust
/// unchanged
pub type PassFailure = Box<dyn Error + Send + Sync + 'static>;

/// NEW. The default pass name: the type name shortened by the rule in 5.1.
/// Borrowed whenever the result is a contiguous slice of `type_name::<T>()`.
/// Never panics; never allocates for a nominal type (generic or not).
#[must_use]
pub fn short_type_name<T: ?Sized>() -> Cow<'static, str>;

pub trait CompilerPass<I, O = I> {
    /// CHANGED: returns Cow; default is short_type_name::<Self>(), with no lookup.
    fn name(&self) -> Cow<'static, str> { short_type_name::<Self>() }
    /// CHANGED: returns Cow; default self.name().
    fn description(&self) -> Cow<'static, str> { self.name() }
    fn validate_input(&mut self, ir: &I, cx: &mut PassContext<'_>) -> Result<(), PassFailure> { Ok(()) }
    /// NEW (replaces should_run + noop_output). Some(output) skips the run
    /// with that output; None runs the pass.
    fn skip(&mut self, ir: &I, cx: &mut PassContext<'_>) -> Result<Option<O>, PassFailure> { Ok(None) }
    fn run(&mut self, ir: &I, cx: &mut PassContext<'_>) -> Result<O, PassFailure>;
    fn validate_output(&mut self, input: &I, output: &O, cx: &mut PassContext<'_>) -> Result<(), PassFailure> { Ok(()) }
    fn did_change(&mut self, input: &I, output: &O) -> Result<bool, PassFailure>;
    fn preserved_analyses(&mut self, input: &I, output: &O, changed: bool) -> Result<PreservedAnalyses, PassFailure>; // default unchanged
    // REMOVED: should_run, noop_output
}
impl<I, O, P: CompilerPass<I, O> + ?Sized> CompilerPass<I, O> for &mut P;  // forwards every hook, incl. skip
impl<I, O, P: CompilerPass<I, O> + ?Sized> CompilerPass<I, O> for Box<P>;  // forwards every hook, incl. skip

/// unchanged signature; lifecycle per 5.3.
pub trait ExecutePass<I, O>: CompilerPass<I, O> {
    fn execute(&mut self, ir: &I) -> Result<PassOutcome<O>, PassError>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassOutcome<O> { /* private */ }
impl<O> PassOutcome<O> {
    pub fn output(&self) -> &O;  pub fn into_output(self) -> O;
    pub fn is_changed(&self) -> bool;
    /// NEW: whether `skip` returned the output (the run did not happen).
    pub fn is_skipped(&self) -> bool;
    pub fn diagnostics(&self) -> &[Diagnostic];
    pub fn preserved_analyses(&self) -> &PreservedAnalyses;
}
```

#### 2.4 Context, analyses, preservation

```rust
/// CHANGED: pass_name is Cow<'static, str> internally.
/// REMOVED: new_standalone (pub(crate)). `new` stays pub(in crate::pass).
#[derive(Debug)]
pub struct PassContext<'a> { /* private */ }
impl<'a> PassContext<'a> {
    pub fn report(&mut self, level: DiagnosticLevel, message: Note, detail: Option<String>);   // unchanged (the diagnostic batch may reshape it)
    pub fn report_text(&mut self, level: DiagnosticLevel, message: impl Into<String>, detail: Option<String>); // unchanged
    pub fn analysis<A, T>(&mut self, ir: &T) -> Arc<A::Output> where A: Analysis<T> + Default, T: NodeHandle; // unchanged
    #[must_use] pub fn diagnostics(&self) -> &[Diagnostic];  // unchanged
    #[must_use] pub fn pass_name(&self) -> &str;             // unchanged
}

/// NEW. Marks a type as an analysis, so it can be named in a preservation set.
pub trait AnalysisMarker: 'static {}

/// CHANGED: gains the supertrait AnalysisMarker. `Default` is still required
/// by PassContext::analysis (analyses depending on analyses is deferred).
pub trait Analysis<T>: AnalysisMarker {
    type Output: Send + Sync + 'static;
    fn run(&self, ir: &T) -> Self::Output;
}

#[derive(Clone, Copy)] pub struct AnalysisId { /* private */ }
impl AnalysisId {
    /// CHANGED bound: A: AnalysisMarker (was A: ?Sized + 'static).
    #[must_use] pub fn of<A: AnalysisMarker>() -> Self;
}
// PartialEq/Eq/Hash by TypeId, Ord by type name then TypeId, Display = type name: unchanged.

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PreservedAnalyses { /* private */ }
impl PreservedAnalyses {
    pub fn all() -> Self;  pub fn none() -> Self;                        // unchanged
    /// CHANGED bound: A: AnalysisMarker. Carries a compile_fail doctest.
    pub fn preserve<A: AnalysisMarker>(self) -> Self;
    pub fn preserve_id(self, id: AnalysisId) -> Self;                     // unchanged
    /// CHANGED bound: A: AnalysisMarker.
    pub fn is_preserved<A: AnalysisMarker>(&self) -> bool;
    pub fn is_id_preserved(&self, id: AnalysisId) -> bool;               // unchanged
    pub fn preserves_all(&self) -> bool;                                 // unchanged
    pub fn preserved_ids(&self) -> impl Iterator<Item = AnalysisId> + '_; // unchanged
}
```

**Decision B5-2 (F-022 marker bound).** The bound is a separate marker trait,
`AnalysisMarker`, which `Analysis<T>` requires. It cannot be a blanket impl
over `Analysis<T>`, because `T` would be unconstrained. It cannot be
`A: Analysis<T>` on `preserve` either. A probe shows `T` is inferred only
while the analysis has exactly one `Analysis` impl: `LabelAnalysis`, which
has two, fails with E0283, and adding a second impl would break every
caller. The cost is one `impl AnalysisMarker for X {}` line per analysis
type.

#### 2.5 Registry (F-006, decision 7)

```rust
/// NEW. An owned registry of pass factories, looked up by name. Send + Sync.
#[derive(Default)]
pub struct PassRegistry { /* private: BTreeMap<Cow<'static, str>, Registration> */ }
impl fmt::Debug for PassRegistry;   // lists the PassInfo entries
impl PassRegistry {
    #[must_use] pub fn new() -> Self;
    /// Register pass type P from I to O under `name`, built by `factory`.
    /// A registration's identity is (TypeId::of::<P>(), TypeId::of::<I>(),
    /// TypeId::of::<O>()). Re-registering the same identity under the same
    /// name and description is Ok(()) and changes nothing (the new factory
    /// is dropped). One identity may be registered under several names
    /// (aliases). Registration never changes any pass's name().
    /// # Errors: see PassRegistrationError; the registry is unchanged on error.
    pub fn register<P, I, O>(
        &mut self,
        name: impl Into<Cow<'static, str>>,
        description: impl Into<Cow<'static, str>>,
        factory: impl Fn() -> P + Send + Sync + 'static,
    ) -> Result<(), PassRegistrationError>
    where P: CompilerPass<I, O> + Send + 'static, I: 'static, O: 'static;
    /// Build a new instance of the pass registered under `name`.
    /// # Errors: CreatePassError::UnknownPass, CreatePassError::IrTypeMismatch.
    pub fn create<I: 'static, O: 'static>(&self, name: &str)
        -> Result<Box<dyn CompilerPass<I, O> + Send>, CreatePassError>;
    #[must_use] pub fn info(&self, name: &str) -> Option<&PassInfo>;
    /// Every registration, ordered by name.
    pub fn iter(&self) -> impl Iterator<Item = &PassInfo> + '_;
    #[must_use] pub fn len(&self) -> usize;
    #[must_use] pub fn is_empty(&self) -> bool;
}

/// CHANGED: fields hold Cow; `type_name()` REMOVED; typed ids NEW.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassInfo { /* private */ }
impl PassInfo {
    pub fn name(&self) -> &str;
    pub fn description(&self) -> &str;
    /// NEW: the registered pass type, and its IR types.
    pub fn pass_type_id(&self) -> TypeId;
    pub fn input_type_id(&self) -> TypeId;
    pub fn output_type_id(&self) -> TypeId;
    /// RENAMED from type_name: std::any::type_name::<P>(), for messages only (not stable).
    pub fn pass_type_name(&self) -> &'static str;
}

/// CHANGED: struct { message } -> #[non_exhaustive] enum. Registration only.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PassRegistrationError {
    /// `name` is empty or only whitespace (char::is_whitespace).
    EmptyName,
    #[non_exhaustive] EmptyDescription { name: Cow<'static, str> },
    /// `name` is registered to a different (pass, input, output) identity.
    #[non_exhaustive] NameTaken { name: Cow<'static, str>, registered_pass_type_name: &'static str },
    /// `name` is registered to this identity with a different description.
    #[non_exhaustive] DescriptionConflict { name: Cow<'static, str>, registered: Cow<'static, str>, requested: Cow<'static, str> },
}
impl fmt::Display for PassRegistrationError; impl Error for PassRegistrationError {}  // source() = None

/// NEW: the lookup half of the old PassRegistrationError.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CreatePassError {
    #[non_exhaustive] UnknownPass { name: String },
    /// The type-name fields are std::any::type_name output, for messages only.
    #[non_exhaustive] IrTypeMismatch {
        name: String,
        registered_input: &'static str, registered_output: &'static str,
        requested_input: &'static str, requested_output: &'static str,
    },
}
impl fmt::Display for CreatePassError; impl Error for CreatePassError {}
```

- **REMOVED:** `register_pass`, `create_pass`, `registered_passes`,
  `run_count`, `run_count_of` and `total_run_count`. The `static REGISTRY`
  and `is_python_whitespace` go with them.
- **Decision B5-3.** `create` gets its own error type, `CreatePassError`,
  because its callers can never see `EmptyName` (S-5: one error type per
  operation). One pass may be registered under several names, as it can be
  today.

**For B4.** `register_expression_passes` takes the registry:

```rust
pub fn register_expression_passes(registry: &mut PassRegistry) -> Result<(), PassRegistrationError>
// body: registry.register::<RewriteRuleApplier, Expression, Expression>(
//           RULE_APPLIER_PASS_NAME, RULE_APPLIER_PASS_DESCRIPTION, || RewriteRuleApplier::new([]))
```

The call is idempotent on the same registry. Two registries are independent.

**Python binding (note only; the binding is later work).**

- **Where it lives.** `fhy-core-py` keeps one `PassRegistry` in its module
  state, for example a `#[pyclass(frozen)]` wrapping `RwLock<PassRegistry>`,
  stored as a module attribute and filled at module init by
  `register_expression_passes`.
- **What it holds.** It holds only Rust passes. Python's own
  `CompilerPass._registry` keeps registering Python classes, and a Python
  lookup falls back to the Rust registry by name.
- **Pipelines.** Pipelines are `Send` but not `Sync`, so the binding wraps
  them in a `Mutex` (decision 10).

#### 2.6 Errors (F-014)

```rust
/// CHANGED: #[non_exhaustive]; ShouldRun and NoopOutput REMOVED, Skip NEW.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PassHook { ValidateInput, Skip, Run, ValidateOutput, DidChange, PreservedAnalyses }
impl PassHook { #[must_use] pub fn as_str(self) -> &'static str; }  // "validate_input", "skip", "run", ...
impl fmt::Display for PassHook;                                     // = as_str

/// NEW (was the pub(super) PassErrorClass). What the binding maps to
/// PassValidationError / PassExecutionError.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FailureClass { Validation, Execution }

/// NEW public (was a private enum in manager.rs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum VerificationPoint { Input, Output }

/// CHANGED representation: Box<private inner>, so Result<_, PassError> is one pointer.
#[derive(Debug)]
pub struct PassError { /* private: Box<Inner> */ }
impl PassError {
    /// NEW: a borrowed, matchable view of what failed.
    #[must_use] pub fn kind(&self) -> PassErrorKind<'_>;
    /// NEW: Validation or Execution (rule in 5.5).
    #[must_use] pub fn class(&self) -> FailureClass;
    /// unchanged: the failing pass's name; None for NonConvergence.
    #[must_use] pub fn pass_name(&self) -> Option<&str>;
    /// unchanged meaning; now also for Nested (the outer pass's diagnostics).
    #[must_use] pub fn diagnostics(&self) -> &[Diagnostic];
    /// NEW: the records of the pipeline work completed before the failure (5.4).
    #[must_use] pub fn records(&self) -> &[PipelineRecord];
    // REMOVED: is_validation_failure, is_execution_failure, is_non_convergence,
    //          failed_hook, verification_report
}
impl fmt::Display for PassError;   // one lowercase line, never the source's text (5.6)
impl Error for PassError;          // source(): Hook -> the hook's error; Nested -> the inner PassError; else None

/// NEW.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum PassErrorKind<'a> {
    /// A hook returned an error that is not a PassError.
    #[non_exhaustive]
    Hook { pass_name: &'a str, hook: PassHook, source: &'a (dyn Error + Send + Sync + 'static) },
    /// A hook returned a PassError (for example from running a nested pass).
    #[non_exhaustive]
    Nested { pass_name: &'a str, hook: PassHook, inner: &'a PassError },
    /// The pipeline's verifier rejected IR at `point`, blaming `pass_name`.
    #[non_exhaustive]
    Verification { pass_name: &'a str, point: VerificationPoint, report: &'a ValidationReport<ValidatorRecord> },
    /// A fixpoint group that fails on non-convergence used its budget.
    #[non_exhaustive]
    NonConvergence { group_name: &'a Identifier, max_iterations: NonZeroUsize },
}
```

`PassError` constructors stay `pub(in crate::pass)`. `PassError` is
`Send + Sync + 'static`, which a static assertion in `error.rs` checks.

#### 2.7 Pipelines (F-017, decision 10; F-006 statistics)

```rust
pub struct PassManager<'p, I> { /* private: passes stored as Box<dyn CompilerPass<I> + Send + 'p> */ }
impl<'p, I: NodeHandle> PassManager<'p, I> {
    #[must_use] pub fn new(name: Identifier) -> Self;                           // unchanged
    #[must_use] pub fn name(&self) -> &Identifier;                              // unchanged
    /// CHANGED bound: + Send.
    pub fn add_pass(&mut self, pass: impl CompilerPass<I> + Send + 'p);
    pub fn add_fixpoint_group(&mut self, group: FixpointPassGroup<'p, I>);      // unchanged
    pub fn set_verifier(&mut self, verifier: ValidationManager<'p, I>);         // unchanged
    /// CHANGED error contents (records attached; 5.4).
    pub fn run(&mut self, ir: &I) -> Result<PassManagerResult<I>, PassError>;
}
// PassManager<'p, I>: Send for every I (static assertion). Default, HasIdentifier, Debug unchanged.

pub struct FixpointPassGroup<'p, I> { /* private */ }
impl<'p, I> FixpointPassGroup<'p, I> {
    // new, with_max_iterations, with_fail_on_non_convergence, name, max_iterations,
    // fails_on_non_convergence: unchanged
    /// CHANGED bound: + Send.
    pub fn add_pass(&mut self, pass: impl CompilerPass<I> + Send + 'p);
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassManagerResult<I> { /* private */ }
impl<I> PassManagerResult<I> {
    pub fn output(&self) -> &I;  pub fn into_output(self) -> I;  pub fn records(&self) -> &[PipelineRecord]; // unchanged
    /// NEW: every pass run in run order, fixpoint groups flattened.
    pub fn pass_runs(&self) -> impl Iterator<Item = &PassRunRecord> + '_;
    /// NEW: how many pass runs were not skipped (replaces the global counters).
    #[must_use] pub fn run_count(&self) -> usize;
}

/// CHANGED: #[non_exhaustive].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PipelineRecord { Pass(PassRunRecord), FixpointGroup(FixpointGroupRecord) }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassRunRecord { /* private; pass_name: Cow<'static, str> */ }
impl PassRunRecord {
    pub fn pass_name(&self) -> &str; pub fn is_changed(&self) -> bool;          // unchanged
    /// NEW.
    pub fn is_skipped(&self) -> bool;
    pub fn diagnostics(&self) -> &[Diagnostic]; pub fn preserved_analyses(&self) -> &PreservedAnalyses; // unchanged
}
// FixpointIterationRecord, FixpointGroupRecord: unchanged API.
```

#### 2.8 Validation (F-022)

```rust
/// NEW. A collect-all check over IR of type I.
pub trait Validator<I> {
    fn name(&self) -> Cow<'static, str> { short_type_name::<Self>() }
    /// Check `ir`, reporting problems as diagnostics into `cx`. An Err means
    /// the validator itself could not finish; see ValidationManager::validate.
    fn validate(&mut self, ir: &I, cx: &mut PassContext<'_>) -> Result<(), PassFailure>;
}
impl<I, V: Validator<I> + ?Sized> Validator<I> for &mut V;   // forwards
impl<I, V: Validator<I> + ?Sized> Validator<I> for Box<V>;   // forwards

/// NEW. Adapter: runs a CompilerPass<I, ()> as a validator.
#[derive(Debug)]
pub struct PassValidator<P> { /* private */ }
impl<P> PassValidator<P> {
    #[must_use] pub fn new(pass: P) -> Self;
    #[must_use] pub fn pass(&self) -> &P;
    #[must_use] pub fn pass_mut(&mut self) -> &mut P;
    #[must_use] pub fn into_pass(self) -> P;
}
impl<I, P: CompilerPass<I, ()>> Validator<I> for PassValidator<P>;  // name() = pass.name(); hooks per 5.7

pub struct ValidationManager<'p, I> { /* private: Box<dyn Validator<I> + Send + 'p> */ }
impl<'p, I> ValidationManager<'p, I> {
    #[must_use] pub fn new(name: Identifier) -> Self;       // unchanged
    #[must_use] pub fn name(&self) -> &Identifier;          // unchanged
    /// CHANGED: takes a Validator (was impl CompilerPass<I, ()> + 'p), + Send.
    pub fn add(&mut self, validator: impl Validator<I> + Send + 'p);
    /// CHANGED: iterator of Cow (was Vec<String>).
    pub fn validator_names(&self) -> impl Iterator<Item = Cow<'static, str>> + '_;
    /// CHANGED record type. Standalone: validators compute analyses afresh (unchanged).
    #[must_use] pub fn validate(&mut self, ir: &I) -> ValidationReport<ValidatorRecord>;
}
// pub(in crate::pass) fn validate_in(&mut self, ir: &I, cache: &mut AnalysisCache) -> ValidationReport<ValidatorRecord>;
// Default ("validation-pipeline"), HasIdentifier, Debug: unchanged. Send for every I.

/// NEW (replaces PassRunRecord inside validation reports).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ValidatorRecord { /* private: name, Range<usize>, failed */ }
impl ValidatorRecord {
    #[must_use] pub fn validator_name(&self) -> &str;
    /// Whether `validate` returned an error.
    #[must_use] pub fn is_failed(&self) -> bool;
    /// This validator's diagnostics: a slice of `report.diagnostics()`.
    /// # Panics: if `report` is not the report this record came from and is too short.
    #[must_use] pub fn diagnostics_in<'r>(&self, report: &'r ValidationReport<ValidatorRecord>) -> &'r [Diagnostic];
}
```

#### 2.9 Adapters (`pass/adapters.rs`)

```rust
#[derive(Debug)] pub struct WalkPass<V> { /* private */ }     // new, visitor, visitor_mut, into_visitor: unchanged
impl<N, V> CompilerPass<N, ()> for WalkPass<V>
where
    N: Tree,
    V: for<'a> TreeVisitor<N, PassContext<'a>>,
    for<'a> <V as TreeVisitor<N, PassContext<'a>>>::Error: Into<PassFailure>,
{ /* name() = short_type_name::<V>(); run = walk_tree(.., cx); did_change = Ok(false); no skip override */ }

#[derive(Debug)] pub struct RewritePass<R> { /* private */ }  // new, rewriter, rewriter_mut, into_rewriter: unchanged
impl<N, R> CompilerPass<N> for RewritePass<R>
where
    N: Tree,
    R: for<'a> Rewriter<N, PassContext<'a>>,
    for<'a> <R as Rewriter<N, PassContext<'a>>>::Error: Error + Send + Sync + 'static,
{ /* name() = short_type_name::<R>(); run = rewrite_tree(.., cx) boxed; did_change = identities differ */ }
```

Both bounds compile on rustc 1.98, as a scratch probe showed. That includes
boxing `WalkPass` and `RewritePass` as `dyn CompilerPass + Send`.

### 3. Interface delta

Semver: the crate is unpublished (S-8), so every change is allowed. Each
row is still classified. "C" marks call sites that change only mechanically
(paths, signatures).

| Change | Item | Before (HEAD) | After | Semver | Call sites |
|---|---|---|---|---|---|
| Moved | module | `fhy_core::pass_infrastructure` | `fhy_core::pass` | breaking | src: `symbolic/expression/{node,screen,pprint,registration}.rs`, `pattern/rewrite.rs`, all doctests in `pass_infrastructure/*`, `lib.rs` docs; tests: `pass_infrastructure_{core,manager,validation,tree}_stories.rs`, `pass_infrastructure_{manager,tree}_properties.rs`, `pass_infrastructure_run_count_stories.rs` (deleted), `expression_pass_stories.rs`, `expression_tree_stories.rs`, `common/{pass_ir,tree_ir}.rs`. fhy-core-py: none |
| Moved | `Tree`, `TreeVisitor`, `Rewriter`, `TraversalOrder`, `walk_tree`, `rewrite_tree`, `RewriteTreeError` | `pass_infrastructure::*` | `tree::*` | breaking | src: `node.rs:25-28`, `screen.rs:7`, `rewrite.rs:20-23`; tests: tree stories/properties, `expression_tree_stories.rs:15-18`, `common/tree_ir.rs:13-16` |
| Moved | `NodeHandle`, `NodeIdentity` | `pass_infrastructure::*` | `tree::*` (B5-1) | breaking | `node.rs`, `screen.rs`, `rewrite.rs`, `analysis.rs`, `manager.rs` doctest; tests: `common/{pass_ir,tree_ir}.rs`, core stories (2), manager properties (2) |
| Moved | `BuildIdentityHasher`, `IdentityHasher` | `pub(crate)` in `pass_infrastructure::tree` | `pub(crate)` in `crate::tree` | none | `node.rs:180,264,305`, `screen.rs:57,110` |
| Changed | `TreeVisitor` | `TreeVisitor<N: Tree>`, hooks take `&mut PassContext<'_>` | `TreeVisitor<N: Tree, C: ?Sized = ()>`, hooks take `&mut C` | breaking | `node.rs:310` (C = ()); tests: 7 impls in tree stories, 2 in `expression_tree_stories.rs`, `common/tree_ir.rs:434` |
| Changed | `Rewriter` | `Rewriter<N: Tree>`, `rewrite(&mut self, &N, &mut PassContext<'_>)` | `Rewriter<N: Tree, C: ?Sized = ()>`, `&mut C` | breaking | `node.rs:335`, `rewrite.rs:127` (B4); tests: tree stories (1), `expression_tree_stories.rs` (2), `common/tree_ir.rs:504` |
| Changed | `walk_tree` | `(visitor, root, order, cx: &mut PassContext<'_>)` | `(visitor, root, order, cx: &mut C)` | breaking | `node.rs:550`, adapters; tests: tree stories helper `walk`, tree properties (1) |
| Changed | `rewrite_tree` | `(rewriter, root, cx: &mut PassContext<'_>)` | `(rewriter, root, cx: &mut C)`, and a result identical to its input counts as no change | breaking (behavior) | `node.rs:578`, `rewrite.rs:171`; tests: tree stories helper `rewrite`, tree properties (3); behavior pins: tree stories `rewrite_tree_rebuilds_the_parent_of_a_child_returned_as_itself`, `pattern_properties.rs:576-596`, `pattern_rewrite_stories.rs:415-431` (B4) |
| Changed | `RewriteTreeError` | exhaustive; `Display` = fixed text for `Rewrite` | `#[non_exhaustive]` (+ `Rebuild` variant); `Rewrite` displays its error and `source()` is that error's source | breaking | matches at `node.rs:579`, `rewrite.rs:171` (in-crate, unaffected by `non_exhaustive`); tests: tree stories `rewrite_tree_error_describes_*`, `expression_tree_stories.rs:317` |
| Removed | `PassContext::new_standalone` | `pub(crate) fn(String) -> Self` | none | none | `node.rs:549,577`, `rewrite.rs:627` → use `&mut ()` |
| Changed | `CompilerPass::name` / `description` | `-> String`; registry lookup | `-> Cow<'static, str>`; pure | breaking | overrides: `rewrite.rs:540-546` (B4), `tree.rs:485,591`; tests: `common/pass_ir.rs:156`, `common/tree_ir.rs:558`, core stories (2), validation stories (1). Uses in `==` comparisons stay valid (`Cow<str>: PartialEq<&str>`) |
| Removed | `CompilerPass::should_run`, `noop_output` | two hooks | none | breaking | src: `tree.rs:490,596`, `rewrite.rs:548`; tests: core stories (8+7), run-count stories, tree stories (1), `expression_pass_stories.rs:308,543` |
| New | `CompilerPass::skip` | none | `-> Result<Option<O>, PassFailure>` | non-breaking (default) | overrides that replace `should_run`: core stories `RecordingPass`, `FailingHookPass`, `HandOverPass`, `GatedPass`, `SkippedPass` |
| New | `short_type_name` | none (private `strip_type_path`) | `pub fn` | non-breaking | adapters, `Validator::name`, default `name` |
| Changed | `PassHook` | 7 variants, exhaustive | 6 variants (`Skip`), `#[non_exhaustive]` | breaking | `error.rs`, `compiler_pass.rs`; tests: core stories (19 uses of `ShouldRun`/`NoopOutput`), `expression_pass_stories.rs:351` |
| Removed | `PassError::{is_validation_failure, is_execution_failure, is_non_convergence, failed_hook, verification_report}` | predicates / accessors | `kind()`, `class()` | breaking | `validation.rs:16`; tests: core stories (≈10), manager stories (≈8), tree stories (3), `expression_pass_stories.rs:350-351` |
| New | `PassError::{kind, class, records}`, `PassErrorKind`, `FailureClass`, `VerificationPoint` | none | see 2.6 | non-breaking | none |
| Changed | `PassError` Display | `Pass "x" failed run with <cause>`; verification and non-convergence end with "." | 5.6 (lowercase, no cause) | breaking (text) | tests: core stories (6), manager stories (6 literals: 228, 569, 625, 710, 828, 897), validation stories (3), tree stories |
| Changed | pass-through of a hook's `PassError` | returned unchanged, outer diagnostics dropped | wrapped as `Nested`, keeps the outer diagnostics | breaking (behavior) | tests: core stories `execute_hands_a_matching_pass_error_through_unchanged`, `execute_wraps_a_pass_error_of_the_other_class` |
| Changed | `PassRegistrationError` | struct `{ message: String }` | `#[non_exhaustive]` enum (2.5) | breaking | `registration.rs`; tests: core stories (≈12 text assertions) |
| New | `CreatePassError` | none | enum (2.5) | non-breaking | none |
| Removed | `register_pass`, `create_pass`, `registered_passes` | global fns | `PassRegistry::{register, create, iter, info}` | breaking | `registration.rs:3,37-44` and its doctest; tests: core stories (33/16/10 uses), tree stories (2), `expression_pass_stories.rs:26,557-580` |
| Removed | `run_count`, `run_count_of`, `total_run_count` | global counters | `PassManagerResult::{run_count, pass_runs}`, `PassOutcome::is_skipped`, `PassRunRecord::is_skipped` | breaking | `tree.rs:413` doc; tests: core stories (16), tree stories (5), run-count stories (whole file) |
| New | `PassRegistry` | none | 2.5 | non-breaking | B4 `register_expression_passes` |
| Changed | `PassInfo::type_name` | `-> &'static str` | renamed `pass_type_name`; plus `pass_type_id`, `input_type_id`, `output_type_id` | breaking | tests: core stories (2), `expression_pass_stories.rs:570-572` |
| Changed | `Analysis<T>` | `: 'static` | `: AnalysisMarker` | breaking | `analysis.rs` unit tests (4 impls); tests: `common/pass_ir.rs` (2), tree stories (1) |
| New | `AnalysisMarker` | none | marker trait | non-breaking | every analysis impl, and test marker types (`Alpha`, `Beta`, the `String` use in core stories) |
| Changed | `AnalysisId::of`, `PreservedAnalyses::{preserve, is_preserved}` | `A: ?Sized + 'static` | `A: AnalysisMarker` | breaking | core stories (27), manager stories (1), manager properties (1) |
| Changed | `PassManager::add_pass`, `FixpointPassGroup::add_pass` | `impl CompilerPass<I> + 'p` | `+ Send + 'p` | breaking | tests: closures capturing `RefCell`/`Cell` (manager stories 211, 380, 697, 809; manager properties 132, 198); `common/pass_ir.rs::RunHook` gains `+ Send` |
| Changed | `ValidationManager::add` | `impl CompilerPass<I, ()> + 'p` | `impl Validator<I> + Send + 'p` | breaking | tests: manager stories (`NegativeValueCheck`, `CountingCheck`), validation stories (all), tree stories `pass_manager_verifies_a_rewrite_with_a_walk_pass` (wrap in `PassValidator`) |
| Changed | `ValidationManager::validate` | `-> ValidationReport<PassRunRecord>` | `-> ValidationReport<ValidatorRecord>` | breaking | validation stories (records assertions) |
| Changed | `ValidationManager::validator_names` | `-> Vec<String>` | `-> impl Iterator<Item = Cow<'static, str>>` | breaking | validation stories (3) |
| New | `Validator`, `PassValidator`, `ValidatorRecord` | none | 2.8 | non-breaking | none |
| New | `PassOutcome::is_skipped`, `PassRunRecord::is_skipped`, `PassManagerResult::{pass_runs, run_count}` | none | 2.3, 2.7 | non-breaking | none |
| Changed | `PipelineRecord` | exhaustive | `#[non_exhaustive]` | breaking for outside matches | tests match both arms (manager stories, properties, tree stories). Integration tests are an outside crate, so they need a `_` arm |
| Changed | `WalkPass`/`RewritePass` impl bounds | `V: TreeVisitor<N>`; name via the registry | HRTB over `PassContext<'a>`; name = `short_type_name::<V or R>()` | breaking | tests: tree stories visitor impls, `expression_tree_stories.rs` |
| Changed | `lib.rs` crate docs | names the pass registry and run counters as process-global statics | drop that part | none | `lib.rs:3-11` |

### 4. Encapsulation delta

- **Removed state.** `static REGISTRY`, the `Registry` struct and
  `record_run` are deleted. The crate keeps no pass-related global state.
- **Narrowed.** `PassContext::new_standalone` (`pub(crate)`) is removed.
  `PassContext::new` stays `pub(in crate::pass)` and gains no public
  constructor. Outside a pipeline, callers use `C = ()` walks.
- **Moved without widening.** `IdentityHasher` and `BuildIdentityHasher`
  move from `pass_infrastructure::tree` to `crate::tree` and stay
  `pub(crate)`.
- **Widened on purpose.**
  - `PassErrorClass` (`pub(super)`) becomes the public `FailureClass`. It
    carries the same information as the removed public
    `is_validation_failure` and `is_execution_failure`, now as an enum
    (S-5), and the binding needs it to choose a Python exception class.
  - `VerificationPoint` (private) becomes public, because
    `PassErrorKind::Verification` exposes it.
- **Enums with public variants.** `PassErrorKind`, `PassRegistrationError`,
  `CreatePassError`, `RewriteTreeError`, `PipelineRecord`, `PassHook`,
  `FailureClass` and `VerificationPoint` are all `#[non_exhaustive]` (S-6).
  Variants with fields are `#[non_exhaustive]` too, so fields can be added.
  `TraversalOrder` stays exhaustive.
- **Private representation.**
  - `PassError` holds a `Box` of a private struct, and `kind()` is a
    borrowed view, so the representation can change freely.
  - `PassRegistry` keeps its private `Registration` and `PassKey`
    (`(TypeId, TypeId, TypeId)`).
  - `AnalysisCache` stays `pub(in crate::pass)`.
- **Trait surface.**
  - Every `CompilerPass` method is user-facing. The lifecycle driver stays
    a private free function.
  - `Validator` has two methods.
  - `AnalysisMarker` is a public marker. It is not sealed, because users
    define analyses.
- **Leaf modules.** `tree` holds no invariant-carrying state apart from the
  memo tables inside `rewrite_tree`. The cache invariant ("results hold
  handles that pin their nodes") stays inside `pass/analysis.rs`, and
  nothing else touches bucket fields.

### 5. Behavior

#### 5.1 Default names (F-006, F-039)

`short_type_name::<T>()` takes `s = std::any::type_name::<T>()` and returns
`s` with two changes:

1. Every generic argument list `<…>` is removed, with nesting counted. A `>`
   that is part of `->` does not close a list.
2. Every path `a::b::C` is replaced by its last segment. When that segment
   is a `{{…}}` segment (closure, coroutine, constant), the last two
   segments are kept (`outer::{{closure}}`).

Everything else is kept: parentheses, brackets, `&`, `mut`, `dyn`, `;`,
digits and `->`. The result is `Cow::Borrowed` whenever it is a contiguous
slice of `s`.

| `type_name` | `short_type_name` |
|---|---|
| `my_crate::passes::Fold<i64>` | `Fold` (borrowed) |
| `my_crate::Fold` | `Fold` (borrowed) |
| `(main::a::B, main::a::Fold<main::a::B>)` | `(B, Fold)` |
| `main::main::{{closure}}` | `main::{{closure}}` (borrowed) |
| `&mut main::a::B` | `&mut B` |
| `alloc::boxed::Box<dyn core::ops::function::Fn()>` | `Box` (borrowed) |
| `[main::a::B; 2]` | `[B; 2]` |
| `fn(i32) -> i32` | `fn(i32) -> i32` (borrowed) |

Every left-hand value except the two `my_crate` rows is rustc 1.98 output
from a scratch probe. For those two, the probe printed `main::a::Fold<i64>`.

- **Stability.** The rustdoc says `type_name` output is not guaranteed
  stable. A pass whose name is part of a contract (a registered name, or a
  diagnostic source that something matches) overrides `name()`.
- **Purity.** Default `name()` takes no lock and does no lookup.
  Registering a pass never changes any instance's `name()`.
- **Adapters.** `WalkPass<V>` is named `short_type_name::<V>()` and
  `RewritePass<R>` is named `short_type_name::<R>()`.

#### 5.2 `rewrite_tree` (F-008, F-009)

For each distinct node `n`, children first and in `Tree::children` order:

1. Let `results[i]` be the result of child `i`: `None` means the child is
   unchanged.
2. If any result is `Some`, `visited = n.rebuild_with_children(merged)`. An
   error becomes `RewriteTreeError::Rebuild { node: n, children: merged,
   source }`. Otherwise `visited = n`.
3. `r = rewriter.rewrite(&visited, cx)`. An error becomes
   `RewriteTreeError::Rewrite(e)`.
4. `result = r.or(rebuilt)`. **New (F-009):** if `result.identity() ==
   n.identity()`, the result is `None`.

The output is the root's result, or `root.clone()`.

- **Invariant.** A `Some` result is never identity-equal to its original. So
  a node is rebuilt only when some child really changed, and the output is
  identity-equal to `root` exactly when no node's result differs from it.
- **Cases:**
  - A rewriter that returns `Some(node.clone())` everywhere yields the root
    itself.
  - A rewriter that returns the *original* node after its children changed
    (a revert) yields that original: no change there, and no ancestor
    rebuild.
  - A hash-consing `Tree` whose `rebuild_with_children` returns an existing
    handle identical to `n` counts as unchanged.
- **Shared nodes.** Memoization of shared nodes is unchanged. The memo stores
  `None` for unchanged nodes.
- **Context.** `cx` is passed through untouched. The walkers never read it,
  and a `C = ()` caller passes `&mut ()`.
- **`walk_tree`.** Behavior is unchanged apart from the generic context.
- **For B3.** `Expression::free_identifiers` becomes `walk_tree(&mut
  collector, self, TraversalOrder::Pre, &mut ())`, with the collector
  implementing `TreeVisitor<Expression>`. `Expression::substitute` becomes
  `rewrite_tree(&mut substitution, self, &mut ())`. Both keep their error
  mapping. One visible consequence: substituting `x ↦ x` with the same
  handle now returns `self` itself.
- **For B4.**
  - `RuleApplier` implements `Rewriter<Expression>` with `C = ()`.
    `apply_rewrite_rules` and `RewriteRuleApplier::run` call
    `rewrite_tree(&mut applier, e, &mut ())`, and `run` reports the firing
    diagnostics through its own `cx` afterwards, as today.
  - The tree now drops a rule's identity rewrite as a change. Whether
    `fired()` still records that firing is B4's decision; the audit
    suggests "no".

#### 5.3 Pass lifecycle (F-022)

`execute` and the pipelines drive the same order:

1. `validate_input(ir)`.
2. `skip(ir)`. If it returns `Some(out)`, call `preserved_analyses(ir, &out,
   false)`. The outcome is `changed = false`, `skipped = true`, and
   `validate_output` and `did_change` are **not** called.
3. `run(ir)`, then `validate_output`, then `did_change`, then
   `preserved_analyses(.., changed)`.

A skipped run is excluded from `PassManagerResult::run_count()`. The
"missing no-op output" runtime error cannot occur any more: a pass that can
skip must supply the output in the same hook. `RewritePass`, `WalkPass` and
`RewriteRuleApplier` no longer override anything for skipping.

#### 5.4 Pipelines, records, statistics (F-006, F-014)

- **Records on success.** A successful run returns one `PipelineRecord` per
  item, as today.
  - `PassRunRecord::is_skipped` reports skipped runs.
  - `pass_runs()` flattens groups in run order.
  - `run_count()` counts the entries of `pass_runs()` that are not skipped.
  - Nothing is counted across runs or processes.
- **Records on failure.** A failed run returns a `PassError` whose
  `records()` holds every item record completed before the failure, in
  pipeline order.
  - If the failure happened inside a fixpoint group, the last record is that
    group's `FixpointGroupRecord`. It holds the iterations begun so far, and
    the last iteration lists only the pass runs that completed. The group is
    `is_converged() == false`.
  - For `NonConvergence`, the group record is complete, with every
    iteration of the budget.
  - The failing pass run itself has no record. Its diagnostics are
    `PassError::diagnostics()`.
  - `execute` always gives an empty `records()`.
- **Verification.** It still happens on the pipeline input (blaming the
  first pass; skipped for a pipeline without passes) and on every output
  reported changed.
  - A node already verified during this run is not verified again. The
    verified-identity set is a flag in the node's cache bucket, which pins
    the node, so its address cannot be reused mid-run.
  - The verifier runs with the pipeline's analysis cache (5.8).

#### 5.5 Errors (F-014)

- **Wrapping a hook's error.** A hook error `f` from hook `h` of pass `p`
  becomes:
  - `Nested { p, h, inner }` if `f` downcasts to `PassError`;
  - `Hook { p, h, source: f }` otherwise.
- **Diagnostics in both cases.** The outer context records one error
  diagnostic: message `pass {p:?} failed in {h}: {chain}`, where `{chain}`
  is `f`'s `Display` followed by each `source()` down the chain, joined with
  `": "`. It has no detail. `PassError::diagnostics()` is the outer
  context's diagnostics, *moved* out of the context (no `to_vec`), ending
  with that error.
- **`class()`:**
  - `Hook`: `Validation` for `ValidateInput` and `ValidateOutput`,
    `Execution` for every other hook.
  - `Nested`: the inner error's class when `h == Run`, and `h`'s class
    otherwise. This reproduces today's classification exactly: `run` handed
    both classes through, and every other hook wrapped the other class.
  - `Verification`: `Validation`.
  - `NonConvergence`: `Execution`.
- **`source()`:** `Hook` gives the hook's error, `Nested` gives the inner
  `PassError`, and the other kinds give `None`.
- **`pass_name()`:** `None` only for `NonConvergence`.
- **Verification failure.** The diagnostics are the blamed pass's run
  diagnostics (empty at `Input`), followed by one error whose message is the
  error's `Display` and whose detail is the rendered report. That rendering
  is `ValidationReport`'s `Display` once the diagnostic batch adds it
  (F-015), else `format()`. The report is moved into the error, not cloned.
- **`NonConvergence`** carries `group_name` and `max_iterations` as fields,
  with no diagnostics.

#### 5.6 Display tables (S-4, S-5)

No `Display` below repeats `source()`. Names are written with `{:?}`.

| Value | `Display` |
|---|---|
| `Hook`/`Nested` | `pass "fold" failed in run` |
| `Verification`, `Input` | `verification rejected the input of pass "first" (errors: 1)` |
| `Verification`, `Output` | `verification rejected the output of pass "corrupt" (errors: 1)` |
| `NonConvergence` | `fixpoint group "flip-group" did not converge (max iterations: 3)` |
| `PassRegistrationError::EmptyName` | `pass name is blank` |
| `…::EmptyDescription` | `pass "fold" has a blank description` |
| `…::NameTaken` | `pass name "fold" is already registered to another pass` |
| `…::DescriptionConflict` | `pass "fold" is already registered with a different description` |
| `CreatePassError::UnknownPass` | `no pass is registered as "fold"` |
| `CreatePassError::IrTypeMismatch` | `pass "fold" takes i64 to i64, not alloc::string::String to alloc::string::String` |
| `RewriteTreeError::Rewrite(e)` | `e`'s `Display`; `source()` = `e.source()` (transparent) |
| `RewriteTreeError::Rebuild` | `rebuilding a node around its rewritten children failed`; `source()` = the rebuild error |
| validator failed silently | diagnostic `validator "check" failed without reporting an error: {chain}` |
| `PassHook` | `validate_input`, `skip`, `run`, `validate_output`, `did_change`, `preserved_analyses` |

With these rules, a chain printer (anyhow style) prints each cause exactly
once. `RewritePass` diagnostics then read, for example, `pass "Rename"
failed in run: node "x" is frozen`, instead of today's bare "rewriting a
node failed".

#### 5.7 Validators

- **Running.** `ValidationManager::validate(ir)` runs every validator in
  order and never stops early.
  - Each validator gets a fresh `PassContext` named `validator.name()`.
  - Standalone, it has no analysis cache, which is unchanged: analyses are
    computed afresh.
  - When `validate` returns `Err(e)` and the validator reported no
    error-level diagnostic, the manager appends `validator {name:?} failed
    without reporting an error: {chain}`.
- **The report.** Its `diagnostics()` hold every validator's diagnostics in
  order, **stored once**. Each `ValidatorRecord` holds the validator's name,
  its `Range<usize>` into the report's diagnostics, and whether it failed.
  The ranges are contiguous, in order, and cover the report exactly.
- **`PassValidator<P>`.** `validate` runs `validate_input`, then `skip` (a
  `Some(())` ends the check with Ok), then `run`, then `validate_output`,
  with the usual guard. A failing hook becomes a `PassError`, which records
  its error diagnostic in `cx`, so no silent-failure diagnostic is added.
  `did_change` and `preserved_analyses` are not called: their results were
  always discarded.

#### 5.8 Analysis cache (F-016 partial, F-039)

- **The cache.** It maps (node identity, handle `TypeId`) to a bucket that
  holds a pinning handle clone, results by `AnalysisId`, and a `verified`
  flag. Both maps use `BuildIdentityHasher` instead of SipHash.
- **`transfer(from, to, preserved)`** after each pass is **merge-only**:
  - If `from` and `to` have the same key, it does nothing. A node's own
    results are never invalidated, even when its pass reported a change.
  - Otherwise, each result of `from` that `preserved` keeps is inserted into
    `to`'s bucket unless `to` already has a result for that id. Results
    computed on `to` itself win.
  - `from`'s bucket is left in place, and so are `to`'s existing results.
- **Nothing is removed during a run.** Every bucket, with its pinned handle,
  lives until the run ends. The triage kept pinning for the length of a run
  on purpose. `NodeIdentity` is an address, so releasing a node mid-run
  would let a new node reuse that address and inherit its cached results or
  its verified flag. The rustdoc says so.
- **Verifier.** The pipeline verifier runs through
  `ValidationManager::validate_in(ir, &mut cache)`. Its validators' `cx`
  therefore uses the pipeline cache, so an analysis a pass computed is not
  recomputed by a validator, and the reverse.
- **Standalone runs.** Standalone `execute` stays uncached (unchanged).

#### 5.9 Threading (F-017, decision 10)

- **`Send` everywhere.** Passes are stored as `Box<dyn CompilerPass<I, O> +
  Send + 'p>`, and validators as `Box<dyn Validator<I> + Send + 'p>`. So
  `PassManager`, `FixpointPassGroup` and `ValidationManager` are `Send` for
  every `I`. They are not required to be `Sync`.
- **Registry.** `PassRegistry` is `Send + Sync`, and `create` returns
  `+ Send` boxes.
- **Checks.** Static assertions in `manager.rs` and `registry.rs` check
  these bounds.

### 6. Error and panic model

- **Results.**
  - Every hook error becomes a `PassError`, and no hook error panics.
  - `PassRegistry::register` returns `PassRegistrationError`, leaving the
    registry unchanged.
  - `PassRegistry::create` returns `CreatePassError`.
  - `rewrite_tree` and `walk_tree` return their visitor's error or
    `RewriteTreeError`.
- **Diagnostics.** Problems a validator finds are diagnostics, not errors.
  A verifier rejection is an error, `PassErrorKind::Verification`.
- **Panics.**
  - A panic inside a user hook, rewriter or factory propagates. It is not
    caught, which is unchanged.
  - `ValidatorRecord::diagnostics_in` panics only when it is given a
    different, shorter report. That is documented.
  - `short_type_name` never panics. Unbalanced `<` in odd `type_name`
    output yields a best-effort string.
  - There is no `expect` on input-reachable paths in `pass` or `tree`.
- **Changes from today.**
  - Pass-through is replaced by `Nested`.
  - The "no no-op output" runtime error is gone.
  - Python-style messages become Rust-style.
  - Registration text becomes variants.
  - Records are attached to failures.

### 7. Non-goals (deferred or owned elsewhere)

- **Deferred.**
  - Analyses that depend on other analyses, and dropping the `Analysis:
    Default` requirement (F-022, deferred by the triage).
  - Evicting unreachable nodes from the cache during a run. Pinning stays
    (F-016, deferred by the triage; see 5.8 for why).
  - Standalone `execute` caching analyses within one run. Its behavior is
    unchanged.
- **Owned by other batches.**
  - The `Diagnostic` constructor and the type of `Diagnostic::source`
    belong to the diagnostic batch (F-021, F-015). Until `source` accepts a
    `Cow<'static, str>`, each reported diagnostic still allocates one copy
    of the pass name (the rest of F-039's name item).
  - `ValidationReport`'s `Display` belongs to the diagnostic batch (F-015).
  - `RuleApplier`'s `fired` semantics for identity rewrites, the SipHash
    replacement map in `pattern/rewrite.rs`, and where the expression passes
    live belong to B4.
  - Consolidating test binaries (F-034) belongs to its own batch; B5 only
    deletes `pass_infrastructure_run_count_stories.rs`.
- **Not in this batch.**
  - The binding-crate split (F-005, deferred), and the Python binding's
    registry and pipeline wrappers, which are a note in 2.5 only.
  - `FixpointPassGroup::with_fail_on_non_convergence(bool)` is a bool mode
    argument. No finding in this batch covers it, so it stays.

### 8. Test plan

Conventions: "sig" means only mechanical changes (paths,
`&mut PassContext<'_>` → `&mut C` or `&mut ()`, `-> String` → `->
Cow<'static, str>`, `impl AnalysisMarker`). Weakening modifications are
marked **WEAKENS**.

#### 8.1 `tests/common/pass_ir.rs`

- **Keep (sig):** `BoxIr`, `DoubleAnalysis` and `ParityAnalysis`. Add
  `impl AnalysisMarker`.
- **Modify:**
  - `RunHook` gains `+ Send`.
  - `ClosurePass::name` returns `Cow::Owned(self.name.clone())`, or stores a
    `Cow<'static, str>`.
- **Keep:** `build_add_pass`, `build_identity_pass`.

#### 8.2 `tests/common/tree_ir.rs`

- **Keep:** the toy tree, `ToyRebuildError`, `HookError` and `WalkHook`.
- **Modify:** `RecordingVisitor` and `ClosureRewriter` implement
  `TreeVisitor<ToyTree, C>` and `Rewriter<ToyTree, C>` for every `C:
  ?Sized`, so one value serves direct walks with `&mut ()` and
  `WalkPass`/`RewritePass`.
- **Delete:** `ContextLender`, `LentBody` and `run_with_pass_context`. Their
  only purpose was to borrow a `PassContext`, which F-008 removes; callers
  pass `&mut ()`.

#### 8.3 `tests/pass_infrastructure_core_stories.rs` (becomes `pass_core_stories.rs`)

Remove the file header's global-state rules (lines 6-10).

- **Keep (sig).**
  - The lifecycle tests: `execute_returns_the_output_of_a_changing_run`,
    `execute_of_an_unchanged_run_preserves_every_analysis`,
    `pass_outcome_into_output_returns_the_output`,
    `execute_starts_every_run_without_diagnostics`.
  - The context tests: `report_text_records_a_diagnostic_at_the_given_level`,
    `report_text_keeps_the_detail_separate_from_the_message`,
    `report_keeps_a_structured_note`,
    `pass_context_exposes_the_pass_name_and_diagnostics_so_far`,
    `pass_context_analysis_recomputes_on_every_call_outside_a_manager`.
  - Every preserved-analyses test except the two listed under Modify.
  - `analysis_id_distinguishes_analysis_types` and
    `analysis_id_orders_by_type_name` (add `AnalysisMarker` to `Alpha` and
    `Beta`).
  - The three node-identity tests (import from `fhy_core::tree`).
  - `borrowed_pass_forwards_every_hook_and_keeps_its_state`.
  - `boxed_pass_forwards_every_hook` (uses `skip`).
- **Modify.**
  - `execute_calls_the_hooks_in_lifecycle_order`: the expected list becomes
    `validate_input, skip, run, validate_output, did_change,
    preserved_analyses(changed=true)`.
  - `execute_skipped_run_calls_noop_output_instead_of_run` is renamed
    `execute_skipped_run_calls_skip_instead_of_run`. Expected:
    `validate_input, skip, preserved_analyses(changed=false)`.
  - `execute_skipped_run_outputs_the_noop_output` is renamed
    `execute_skipped_run_outputs_the_skip_output`. It also asserts
    `outcome.is_skipped()`.
  - `pass_hook_renders_the_method_name`: the cases become the 6-hook table.
    This is `PassHook`'s one `Display` table.
  - `execute_wraps_a_hook_error_naming_the_pass_and_hook`: it asserts
    `kind()` is `Hook { pass_name: "FailingHookPass", hook, source }` with
    `source.downcast_ref::<HookFailure>()`. It asserts `class()` per hook,
    `records().is_empty()`, and a last diagnostic whose message is
    `pass "FailingHookPass" failed in {hook}: {hook}-broken`. **WEAKENS
    (moved):** the exact `to_string()` check moves to
    `pass_error_display_table`. It also drops the `should_run` and
    `noop_output` cases, whose hooks no longer exist.
  - `execute_stops_at_the_first_failing_hook`: `should_run` becomes `skip`.
  - `execute_failure_keeps_the_diagnostics_emitted_before_it`: the expected
    error text becomes `pass "WarnThenCrash" failed in run: boom`.
  - `preserved_analyses_all_preserves_every_analysis`: the
    `AnalysisId::of::<String>()` call becomes a local marker type, because
    of the orphan rule.
  - `analysis_id_display_is_the_type_name`: it asserts against
    `std::any::type_name::<DoubleAnalysis>()` instead of the literal
    `"pass_infrastructure_core_stories::pass_ir::DoubleAnalysis"` (F-035).
  - `name_defaults_to_the_type_name_without_path_or_generics`: keep the
    assertions, and also assert `matches!(Increment.name(),
    Cow::Borrowed(_))`.
  - Registry stories, all using a local `PassRegistry::new()`:
    - `create_pass_builds_a_new_instance_of_the_registered_pass` →
      `registry_create_builds_a_new_instance_each_call`.
    - `create_pass_rejects_an_unknown_name` → matches
      `CreatePassError::UnknownPass { name, .. }`.
    - `create_pass_rejects_other_ir_types` → matches `IrTypeMismatch`, with
      its fields compared to `type_name::<T>()`.
    - `registered_passes_lists_a_registration` →
      `registry_iter_lists_registrations_by_name`, which checks
      `pass_type_id() == TypeId::of::<KeepInteger>()` in place of the
      type-name literal.
    - `register_pass_does_not_call_the_factory`.
    - `register_pass_rejects_an_empty_name`: matches `EmptyName`. It drops
      the `"\u{1c}"` case, which becomes a positive case in
      `register_accepts_a_name_rust_does_not_call_whitespace` (S-4). This
      changes behavior; it does not weaken the test.
    - `register_pass_rejects_an_empty_description`: matches
      `EmptyDescription`, plus `registry.is_empty()`.
    - `register_pass_rejects_a_name_taken_by_another_pass_type`: matches
      `NameTaken` and reuses `KeepInteger`/`Increment`.
    - `register_pass_rejects_a_name_taken_by_the_same_pass_over_other_ir_types`:
      matches `NameTaken` (a different identity, because the IR differs).
    - `register_pass_refuses_a_new_description_for_a_registered_pass`:
      matches `DescriptionConflict`.
    - `register_pass_is_idempotent_for_the_same_pass_and_description`.
  - `register_pass_under_a_second_name_makes_it_the_default_name` is
    rewritten as `registry_allows_aliases_without_renaming_the_pass`: both
    names create the pass, and `name()` stays the type's default.
- **Delete, with reasons.**
  - `execute_skipped_run_fails_without_a_noop_output`: the runtime error it
    pins cannot exist with `skip` (F-022).
  - `execute_hands_a_matching_pass_error_through_unchanged`: it pins the
    downcast pass-through that F-014 replaces with `Nested`. Its successor
    is `execute_nests_a_pass_error_keeping_the_outer_diagnostics`.
  - `execute_wraps_a_pass_error_of_the_other_class`: merged into
    `nested_pass_error_class_is_the_hooks_except_for_run`.
  - `name_of_a_registered_pass_is_its_registered_name`: it pins
    registry-derived names, which decision 7 removes. Its successor is
    `pass_registry_does_not_change_pass_names`.
  - `run_count_counts_each_executed_run`,
    `run_count_counts_a_run_that_fails_after_it_started`,
    `run_count_counts_a_registered_pass_under_its_registered_name`,
    `run_count_counts_an_overridden_name_under_that_name` and
    `run_count_of_an_unused_name_is_zero`: the global counters are removed
    (F-006).
  - `run_count_ignores_skipped_runs`: replaced by `is_skipped` assertions
    in `execute_skipped_run_outputs_the_skip_output` and
    `pass_manager_run_count_excludes_skipped_runs`.
  - The pass types that exist only to have a process-unique name
    (`RegisteredNamePass`, `CreatablePass`, `IntegerOnlyPass`, `ListedPass`,
    `LazilyBuiltPass`, `OwnerPass`, `IntruderPass`, `TwoIrPass`,
    `MismatchPass`, `IdempotentPass`, `TwoNamePass`, `RunCountedPass`,
    `GatedPass`, `CountedFailingPass`, `CountedRegisteredPass`,
    `RenamedPass`): F-031. With local registries, the tests reuse
    `Increment`, `KeepInteger` and `GenericNamedPass<T>`. `TwoIrPass` stays
    as the only pass over two IR types. No macro is needed.

#### 8.4 `tests/pass_infrastructure_run_count_stories.rs`

**Delete the whole binary.** It exists only to read the process-global
total alone in its process (F-006, F-031). The per-run successor is
`pass_manager_run_count_excludes_skipped_runs`, in manager stories.

#### 8.5 `tests/pass_infrastructure_manager_stories.rs`

- **Keep (sig).**
  - Order and records: `pass_manager_runs_passes_in_insertion_order`,
    `pass_manager_without_items_returns_the_input`,
    `pass_run_record_holds_the_outcome_of_the_run`,
    `pass_run_record_holds_a_specific_preservation_set`,
    `pass_manager_name_is_its_identifier`,
    `pass_manager_default_is_an_empty_pipeline_named_pipeline`.
  - Every analysis-under-manager test.
    `pass_context_analysis_caches_a_node_other_than_the_input` changes only
    its `RefCell` to a `Mutex`, because of `Send`. That is not weakening.
  - The fixpoint tests `converges_and_records_each_iteration`,
    `runs_its_passes_in_order_each_iteration`,
    `hands_on_the_last_ir_when_allowed_not_to_converge`,
    `without_passes_converges_immediately`,
    `pass_manager_runs_passes_and_groups_in_order` and
    `fixpoint_pass_group_new_uses_the_default_configuration`.
  - The verification tests, with the validators implementing `Validator`:
    `verifier_blames_the_first_pass_of_a_leading_group`,
    `skips_a_pipeline_without_passes`, `validates_unchanged_ir_once`,
    `validates_each_changed_output`, `skips_an_output_reported_unchanged`,
    `with_an_empty_verifier_accepts_any_ir`,
    `without_a_verifier_runs_on_invalid_ir`,
    `set_verifier_replaces_the_previous_verifier`.
  - The `Cell<bool>` captures at 211, 697 and 809 become `AtomicBool`.
- **Modify.**
  - `pass_manager_stops_at_the_first_failing_pass`: it asserts `kind()`
    `Hook` and `class() == Execution`, and that `records()` holds the one
    completed run. The literal `"Pass \"tests.pm.failing\" failed run with
    broken"` moves to the Display table.
  - `fixpoint_group_fails_the_run_when_it_does_not_converge` and
    `fixpoint_group_with_a_budget_of_one_needs_an_unchanged_first_iteration`:
    they match `NonConvergence { group_name, max_iterations }`, and
    `records()` ends with the complete, unconverged group record. The two
    `"... did not converge in N iterations."` literals (569, 625) are
    replaced by field checks (F-035).
  - `pass_manager_verifier_rejects_invalid_input_blaming_the_first_pass`,
    `..._blames_the_pass_that_produced_invalid_output` and
    `..._validates_changed_outputs_inside_a_group`: they match
    `Verification { pass_name, point, report }` and `class() ==
    Validation`. The diagnostic list keeps its levels and sources, and its
    message equals `error.to_string()`. The detail is checked by
    `is_some()` instead of equality with `report.format()`. **WEAKENS
    slightly:** the exact rendering belongs to the diagnostic batch's
    `ValidationReport` Display test. The literals at 710, 828 and 897 move
    to the table.

#### 8.6 `tests/pass_infrastructure_manager_properties.rs`

- **Keep:** `fixpoint_group_of_a_decrementing_pass_converges_after_the_largest_element`
  (sig).
- **Keep:** `pass_manager_runs_passes_in_the_order_added` (the `RefCell`
  becomes a `Mutex`). The `PipelineRecord` match gains a `_` arm.
- **Modify:** `pass_manager_cache_honors_the_preservation_contract`. Add the
  step `CacheStep::ComputeOnOutput(delta)`: a pass that derives its output,
  reads `DoubleAnalysis` of the output inside `run`, and preserves nothing.
  The model sets `cached = Some((value + delta) * 2)` and adds one run.
  Merge-only keeps that result (F-016). This strengthens the test.

#### 8.7 `tests/pass_infrastructure_validation_stories.rs`

- **Modify.** `ScriptedValidator`, `RejectInput` and `CrashInRun`
  implement `Validator`, or go through `PassValidator` where a test is about
  the pass adapter. Records are asserted through `validator_name()`,
  `is_failed()` and `diagnostics_in(&report)`.
  - Keep the substance of every test up to and including
    `validation_manager_keeps_a_structured_note`.
  - `validation_manager_records_each_validator_as_unchanged_and_preserving_all`
    is renamed `validation_manager_records_each_validator_with_its_diagnostics`.
    The `is_changed` and `preserved_analyses` assertions go away with the
    fields. **WEAKENS (fields removed):** those fields always held
    constants.
  - `validation_manager_records_a_failing_validator_and_runs_the_rest` and
    `validation_manager_keeps_the_diagnostics_a_validator_emitted_before_failing`:
    the expected texts become `pass "tests.vm.crasher" failed in run:
    internal boom`, via `PassValidator`.
  - `validation_manager_adds_nothing_when_a_failing_validator_reported_an_error`:
    keep.
  - `validation_manager_adds_an_error_for_a_validator_that_fails_silently`:
    a direct `Validator` returns `Err` without reporting. The expected
    message is `validator "tests.vm.silent" failed without reporting an
    error: {chain}`. The rstest's validation and execution cases collapse
    into one: the old `kind` word came from the Python exception classes.
  - `validation_manager_records_a_validator_that_rejects_its_input`: via
    `PassValidator`, with the new text.
  - `validation_manager_validator_names_lists_validators_in_order`:
    `.collect::<Vec<_>>()`.
  - `validation_manager_runs_validators_without_an_analysis_cache`: keep
    (standalone stays uncached).
- **Keep (sig):** `validation_manager_report_formats_and_escalates_failures`,
  `validation_manager_name_is_its_identifier`,
  `validation_manager_default_is_an_empty_pipeline_named_validation_pipeline`
  and `validation_manager_validates_afresh_on_every_call`.

#### 8.8 `tests/pass_infrastructure_tree_stories.rs` and `..._tree_properties.rs`

- **Keep (sig; `&mut ()` in place of `run_with_pass_context`).**
  - Every `walk_tree_*` test.
  - Every `rewrite_tree_*` test except the one under Modify.
  - The deep-chain tests.
  - The `walk_pass_*` tests except those under Delete.
  - `rewrite_pass_execute_reports_a_rewrite`,
    `rewrite_pass_execute_reports_no_change`,
    `rewrite_pass_reports_no_change_for_a_root_returned_as_itself`,
    `rewrite_pass_did_change_compares_identity`,
    `rewrite_pass_execute_fails_with_the_rewrite_error` (kind check),
    `rewrite_pass_execute_fails_with_a_refused_rebuild`,
    `rewrite_pass_exposes_its_rewriter`,
    `rewrite_pass_is_named_after_its_rewriter`.
  - `walk_pass_hooks_read_analyses`: `CountingVisitor` implements
    `TreeVisitor<ToyTree, PassContext<'_>>`, and `OccurrenceCountAnalysis`
    gains `AnalysisMarker`.
  - The three properties.
- **Modify.**
  - `rewrite_tree_rebuilds_the_parent_of_a_child_returned_as_itself` pins
    the F-009 bug. It is rewritten as
    `rewrite_tree_counts_a_node_returned_as_itself_as_unchanged`: the output
    `is_same_node(&tree)` (decision 8).
  - `rewrite_tree_error_describes_a_failing_rewrite`: `Display` equals
    `"inner"` (transparent), and `source()` is `HookError`'s source,
    `None`.
  - `rewrite_tree_error_describes_a_refused_rebuild`: keep the message and
    the source.
  - `walk_pass_is_named_after_its_visitor`: keep, and add a `Cow::Borrowed`
    check.
  - `walk_pass_execute_fails_with_the_hook_error`: kind check, not
    `is_execution_failure`.
  - `pass_manager_verifies_a_rewrite_with_a_walk_pass`: calls
    `verifier.add(PassValidator::new(&mut walk_pass))`, and the record match
    gains a `_` arm.
- **Delete, with reasons.**
  - `walk_pass_takes_its_registered_name`: registry names are removed
    (decision 7).
  - `walk_pass_counts_runs_under_its_visitor_name`: the global counters are
    removed.
  - `walk_pass_noop_output_is_unit` and
    `rewrite_pass_noop_output_is_the_input`: `noop_output` is removed. The
    adapters have no `skip` override, and `skip`'s default is covered in
    core stories.

`walk_pass_did_change_is_false` stays (sig).

#### 8.9 Call sites outside B5's files (owners B3 and B4; listed so they update in lockstep)

- **`expression_tree_stories.rs`.** Visitors and rewriters there use
  `TreeVisitor<Expression>` with `&mut ()`, or `PassContext<'_>` inside
  `WalkPass` and `RewritePass`. `rewrite_tree_reports_a_refused_expression_rebuild`
  keeps its shape.
- **`expression_pass_stories.rs`:**
  - `rewrite_rule_applier_noop_output_is_the_input` and
    `expression_pretty_formatter_has_no_noop_output` are deleted
    (`noop_output` is removed).
  - `rewrite_rule_applier_execute_fails_with_the_callback_error` uses
    `kind()` in place of `is_execution_failure` and `failed_hook`.
  - `register_expression_passes_registers_the_rule_applier` uses a local
    `PassRegistry`, and checks `pass_type_id()` in place of
    `type_name().ends_with(..)`.
  - The file header's global-registry note goes away.
- **F-009 pins owned by B4:** `pattern_properties.rs:576-596` and
  `pattern_rewrite_stories.rs:415-431` flip with the tree change.

#### 8.10 Regression test per fixed finding

| Finding | Test (file) |
|---|---|
| F-006 | `pass_registry_does_not_change_pass_names`: register `Increment` as `"a"`, then as `"b"`; `Increment.name()` and `registry.create("a")?.name()` are still `"Increment"` (core) |
| F-006 | `pass_registries_are_independent_values`: the same name maps to different passes in two registries (core) |
| F-006 | `same_named_pass_types_in_two_modules_have_separate_registrations`: `mod a { struct Fold }`, `mod b { struct Fold }` registered under different names are both creatable; registering `b::Fold` under `a::Fold`'s name gives `NameTaken` (core) |
| F-006 | `pass_manager_run_count_excludes_skipped_runs` (manager) |
| F-006 | `short_type_name_shortens_*` unit table over literal strings, for the private `shorten` fn (rows of 5.1), in `pass/compiler_pass.rs`, so it does not depend on rustc output |
| F-008 | `walk_tree_and_rewrite_tree_run_without_a_pass_context` (tree stories) |
| F-008 | `rewrite_tree_threads_a_caller_chosen_context`: `C = Vec<String>`, and the rewriter logs into it (tree stories) |
| F-009 | `rewrite_tree_counts_a_node_returned_as_itself_as_unchanged` (tree stories) |
| F-009 | `rewrite_tree_counts_a_reverted_node_as_unchanged`: children change and the rewriter returns the original parent (tree stories) |
| F-009 | `fixpoint_group_of_an_identity_rewrite_pass_converges`: `RewritePass` of `Some(node.clone())` converges in 1 iteration (tree stories) |
| F-014 | `pass_error_display_table`, `pass_registration_error_display_table`, `create_pass_error_display_table` (rstest; the one place text is pinned) |
| F-014 | `pass_error_chain_prints_each_cause_once`: a 3-deep `Nested` chain; walking `source()` shows each message exactly once (core) |
| F-014 | `execute_nests_a_pass_error_keeping_the_outer_diagnostics` (core) |
| F-014 | `nested_pass_error_class_is_the_hooks_except_for_run` (rstest over hooks × inner class; core) |
| F-014 | `failed_pipeline_error_keeps_the_completed_records` (manager) |
| F-014 | `failure_inside_a_fixpoint_group_records_the_partial_group` (manager) |
| F-014 | `non_convergence_error_is_structured` (manager) |
| F-014 | `rewrite_pass_diagnostic_names_the_rewriters_error`: the failure diagnostic contains the `HookError` text (tree stories) |
| F-016 | `pass_manager_keeps_results_computed_on_a_pass_output` (manager) |
| F-016 | `pass_manager_keeps_results_of_an_output_that_is_its_input`: a pass returns its input, reports changed, and preserves none; the next read does not recompute (manager) |
| F-016 | `pass_manager_verifies_each_node_once_per_run`: pass A `x→y`, pass B `y→x` (the same `x` handle), and a counting validator sees 2 nodes, not 3 (manager) |
| F-017 | `pipelines_are_send`: static `assert_send::<PassManager<'static, BoxIr>>()`, `FixpointPassGroup`, `ValidationManager`, `PassRegistry: Send + Sync`, and a pipeline moved into `std::thread::spawn` and run there (manager) |
| F-022 | `execute_skipped_run_outputs_the_skip_output` (core) |
| F-022 | `verifier_reads_the_pipeline_analysis_cache`: a pass reads `DoubleAnalysis`, the validator reads it again, and it runs once (manager) |
| F-022 | `pass_validator_runs_a_pass_as_a_validator` (validation) |
| F-022 | `compile_fail` doctest on `PreservedAnalyses::preserve` (`preserve::<String>()` must not compile) |
| F-039 | `default_pass_name_is_borrowed` (core) |
| F-039 | `validation_report_stores_each_diagnostic_once`: record ranges partition `report.diagnostics()` (validation) |
| F-031 | No test; enforced by deleting the run-count binary, the unique-name pass types and the header rules. `rg 'tests\.core\.' tests/` shows no global-name convention remains |
| F-035 | The display tables above, plus the type-name assertions rewritten against `type_name::<T>()` or `TypeId` |

#### 8.11 New property and adversarial tests

- **Properties (tree properties):**
  - `rewrite_tree_with_an_identity_rewriter_returns_the_root`: on random
    DAGs, a rewriter that returns `Some(node.clone())` everywhere gives
    `is_same_node(&dag)`, and the rewriter sees each distinct node once.
  - `rewrite_tree_output_is_the_root_iff_nothing_changed`: a random subset
    of leaves is doubled. The output is the root exactly when the subset is
    empty or every doubled value is 0.
- **Adversarial:**
  - A `Tree` whose rebuild returns `self` for equal children counts as
    unchanged (tree stories, a hash-consing toy variant).
  - `register` with the same name for `KeepInteger` over `i64` and over
    `String` gives `NameTaken`.
  - `create::<String, String>` on an `i64` pass gives `IrTypeMismatch`.
  - `short_type_name` handles a qualified path (`<a::B as c::Tr>::X` → `X`)
    and unbalanced input without panicking (unit).
  - A validator that panics propagates the panic (`#[should_panic]`,
    documenting that panics are not caught).
  - A `PassError` from a nested `PassManager::run` inside a hook becomes
    `Nested` and carries the inner records.

### 9. Findings covered

- **F-006: fixed.**
  - `PassRegistry` is an owned value, keyed by the (pass, input, output)
    `TypeId`s.
  - `name()` is pure, and registration never renames a pass.
  - The counters are replaced by per-run statistics.
  - The binding note is in 2.5.
- **F-008: fixed.** The traversals are generic over `C` in `fhy_core::tree`,
  `new_standalone` is gone, and the IR layer no longer imports `pass`.
- **F-009, tree side: fixed.** A result identical to its input counts as
  unchanged. B4 owns `fired()`.
- **F-014: fixed** for pass and tree.
  - `PassError` has `kind()`, `class()` and `records()`; the
    `NonConvergence` kind is structured and `Nested` replaces pass-through.
  - `Display` never repeats the source.
  - `PassRegistrationError` and `CreatePassError` are enums.
  - `RewriteTreeError` is transparent for `Rewrite` and
    `#[non_exhaustive]`, as are `PassHook` and `PipelineRecord`.
- **F-016: partly fixed,** as the triage asked. Preservation is merge-only,
  a node's own results are never invalidated, and a verified flag replaces
  the report cache. Pinning stays, and 5.8 documents why.
- **F-017: fixed.** Stored passes and validators are `+ Send`.
- **F-022: partly fixed,** as the triage asked. There is a single `skip`
  hook, a `Validator` trait with the `PassValidator` adapter, a verifier
  that shares the cache, and a marker bound on `preserve`. Analyses that
  depend on other analyses are deferred.
- **F-039, pass nits: fixed**, except the per-diagnostic copy of the pass
  name, which depends on the diagnostic batch (see 7). The fixes:
  - names are `Cow` values;
  - diagnostics are moved instead of `to_vec`'d;
  - reports are moved instead of cloned;
  - diagnostics are stored once in validation reports;
  - identity maps use the identity hasher.
- **F-031, pass tests: fixed** (8.3, 8.4).
- **F-035, pass tests: fixed** (8.3, 8.5, 8.10).

**Possibly misjudged: none rejected.** Two notes:

- **F-016's "verification-report cache can almost never hit".** This is
  true, but the set that replaces it almost never hits either. Outputs are
  verified only when they are reported changed, and a changed output is
  usually a new node. The set's real value is fewer clones and no report
  kept alive; the cache hit rate does not improve. The spec keeps it
  anyway, because the set is the simpler mechanism.
- **F-022 "validators forced into pass shape".** The adapter keeps
  `WalkPass`-based validators working. A direct `impl Validator<N> for
  WalkPass<V>` would save one wrapper at each call site, but the spec leaves
  it out as gold-plating.

---

## B6: Workspace structure, test architecture, docs, bindings and CONTRIBUTING

Verified against `412e234` (`dev-rust`). The public item list below comes from
`cargo doc --no-deps -p fhy-core` (`all.html`, 130 entries, no `Re-exports`
sections today). File lists come from `grep` over `rust/`.

### 1. Summary

This batch moves the crate to the S-1 module layout in mechanical, compiling
steps. It merges the integration tests into one binary with `pub(crate)`
helpers, removes the child-process isolation harness, cleans up the golden,
packaging and doc CI, prepares `fhy-core` for crates.io, and rewrites the
CONTRIBUTING "Porting to Rust" rules to match decisions 2, 3, 4 and 13. It
also sets the order in which B1 to B6 land.

Batch-level decisions (the user signs off on these; section 9 lists them
again):

- **B6-D1:** expression passes live in **`fhy_core::expr::passes`**, not in
  `fhy_core::pass`. The generic `pass` module then never depends on `expr`.
  This also matches Python's `fhy_core/symbolic/expression/passes/`.
- **B6-D2:** `NodeHandle`, `NodeIdentity` and the crate-private
  `BuildIdentityHasher` move to `fhy_core::tree`. `expr` uses them
  (`node.rs`, `screen.rs`), so leaving them in `pass` would make `expr`
  depend on `pass`. B5 confirms this.
- **B6-D3:** `pattern/core.rs` is **renamed** to `pattern/matching.rs`
  rather than folded into `pattern/mod.rs`. `Pattern` then stays in a leaf
  module that `pattern::rewrite` cannot reach into.
- **B6-D4:** the path moves and the CONTRIBUTING text land **first**, before
  B1. The test consolidation for the isolation-safe files lands with them.

### 2. Desired public interface

#### 2.1 `fhy-core` module tree (final, after B1 to B5)

```
fhy_core                      crate root: docs only, no items
├── identifier                unchanged path (B1 content)
├── interned                  unchanged path (B1 content)
├── diagnostic                unchanged path (B2 content)
├── provenance                unchanged path (B2 content)
├── op_attribute              unchanged path
├── value_domain              unchanged path
├── tree                NEW   B5: from pass_infrastructure::tree, plus NodeHandle/NodeIdentity (B6-D2)
├── expr                NEW   was symbolic::expression + symbolic::symbol_type
│   ├── pattern         NEW   was symbolic::expression::pattern
│   ├── builtins        NEW   was symbolic::expression::builtins
│   └── passes          NEW   B3/B4: ExpressionPrettyFormatter, RewriteRuleApplier, register_expression_passes
└── pass                NEW   was pass_infrastructure (WalkPass/RewritePass adapters stay here)
```

`fhy_core::symbolic`, `fhy_core::symbolic::symbol_type` and
`fhy_core::pass_infrastructure` are **REMOVED**. `fhy_core::testing` is
**REMOVED** by B1 (decision 12).

The layering has no cycles. An arrow means "may depend on":

```
identifier ← interned ← {diagnostic, provenance, op_attribute, value_domain} ← tree
tree ← expr (expr::pattern, expr::builtins)        expr never names crate::pass
tree, diagnostic, identifier ← pass                pass never names crate::expr
expr, pass ← expr::passes                          the only module that names both
```

At HEAD, `identifier.rs` and `shipped.rs` depend on `symbolic::expression`
(F-007). B1 removes that dependency. After B1, B6 checks the layering with
the grep in section 5.4.

#### 2.2 Path map: every current public path

"Step 0" is the path after this batch's first moves. "Final" is the path
after every batch. A name that another batch changes is named in the Final
column with its owner. B6 changes only the module part of a path.

| Current path (HEAD) | After step 0 | Final (owner) |
|---|---|---|
| `identifier::{Identifier, IdSpaceExhausted, HasIdentifier, try_allocate_id, try_advance_counter_past}` | unchanged | unchanged path. B1 adds `Identifier::try_new` and documents the two raw counter fns as binding-only (S-1, S-3) |
| `interned::{Interned, InternRegistry, Canonical, NotInternedError, InternOutcome}` | unchanged | unchanged path. `InternOutcome` fate: B1 (F-021) |
| `diagnostic::{Diagnostic, DiagnosticLevel, Note, NoteKind, ValidationReport, ValidationFailedError}` | unchanged | unchanged path (B2 content) |
| `diagnostic::{get_rationale_note_kind, get_suggestion_note_kind, get_remark_note_kind, get_other_note_kind}` | unchanged | `NoteKind::{rationale, suggestion, remark, other}()` (S-7) |
| `provenance::{Position, Span, Provenance, FileProvenance, NamedProvenance, CallSiteProvenance, FusedProvenance, HasProvenance, ProvenanceError}` | unchanged | unchanged path (B2 content) |
| `op_attribute::OpAttribute` | unchanged | unchanged |
| `op_attribute::{get_commutative, get_associative, get_pure, get_elementwise}` | unchanged | `OpAttribute::{commutative, associative, pure, elementwise}()` (S-7) |
| `value_domain::ValueDomain` | unchanged | unchanged |
| `value_domain::{get_data_domain, get_address_domain}` | unchanged | `ValueDomain::{data, address}()` (S-7) |
| `symbolic` (module) | **REMOVED** | removed |
| `symbolic::expression` (module) | `expr` | `expr` |
| `symbolic::expression::{AlphaRenaming, BigInt, BinaryExpression, CallExpression, Expression, ExpressionKind, PiecewiseExpression, UnaryExpression, BinaryOperation, UnaryOperation, LiteralKind, LiteralValue, LiteralTextError, ExpressionBuildError, BooleanPosition, NonBooleanLogicalOperandError, NonInjectiveRenamingError, FunctionSort, SortLookup, NoRegisteredSorts, IntoOperand, FormatOptions, IdentifierStyle, Notation}` | `expr::{same names}` | `expr::…`. Names and fate (e.g. `IntoOperand`, F-024) are B3's. `BigInt` keeps `expr::BigInt` as its single path |
| `symbolic::expression::{build_call, build_logical_and, build_logical_or, build_piecewise, validate_logical_operands, validate_predicate, format_expression}` | `expr::{same}` | `expr::…` (B3). `format_expression` becomes `expr.display(opts)` (S-7) |
| `symbolic::expression::ExpressionPrettyFormatter` | `expr::ExpressionPrettyFormatter` | `expr::passes::ExpressionPrettyFormatter` (B3, B6-D1) |
| `symbolic::expression::register_expression_passes` | `expr::register_expression_passes` | `expr::passes::register_expression_passes` (B3; signature takes B5's `PassRegistry`, decision 7) |
| `symbolic::symbol_type` (module) | **REMOVED** | removed |
| `symbolic::symbol_type::SymbolType` | `expr::SymbolType` | `expr::SymbolType` |
| `symbolic::expression::pattern` (module) | `expr::pattern` | `expr::pattern` |
| `symbolic::expression::pattern::{Pattern, MatchBindings, CallbackError, PatternError, RewriteRule, RewriteOutcome, RewriteError, FiredRule}` | `expr::pattern::{same}` | `expr::pattern::…` (B4 content) |
| `symbolic::expression::pattern::{match_pattern, does_pattern_match, apply_rewrite_rule, apply_rewrite_rules}` | `expr::pattern::{same}` | methods: `pattern.matches(&e)`, `rule.apply(&e)` … (S-7, B4) |
| `symbolic::expression::pattern::RewriteRuleApplier` | `expr::pattern::RewriteRuleApplier` | `expr::passes::RewriteRuleApplier` (B4, B6-D1) |
| `symbolic::expression::builtins` (module) | `expr::builtins` | `expr::builtins` |
| `symbolic::expression::builtins::{ComposedFunction, NativeFunctionSignature, NativeConstantSpec, find_composed_function, find_native_function, find_native_constant, list_composed_functions, list_native_functions, list_native_constants}` | `expr::builtins::{same}` | `expr::builtins::…`. No `list_` prefix (S-7). Enum-named built-ins (F-024) are B3's |
| `pass_infrastructure` (module) | `pass` | `pass` |
| `pass_infrastructure::{Analysis, AnalysisId, PreservedAnalyses, CompilerPass, ExecutePass, PassFailure, PassOutcome, PassContext, PassError, PassHook, PassRegistrationError, PassManager, PassManagerResult, PassRunRecord, PipelineRecord, FixpointPassGroup, FixpointGroupRecord, FixpointIterationRecord, ValidationManager, WalkPass, RewritePass}` | `pass::{same}` | `pass::…` (B5 content) |
| `pass_infrastructure::{register_pass, create_pass, registered_passes, PassInfo}` | `pass::{same}` | methods of an owned `pass::PassRegistry` (B5, decision 7) |
| `pass_infrastructure::{run_count, run_count_of, total_run_count}` | `pass::{same}` | **REMOVED** (B5, decision 2) |
| `pass_infrastructure::{Tree, TreeVisitor, Rewriter, TraversalOrder, walk_tree, rewrite_tree, RewriteTreeError}` | `pass::{same}` (temporary) | `tree::{same}`, generic over a context `C` (B5, F-008) |
| `pass_infrastructure::{NodeHandle, NodeIdentity}` | `pass::{same}` (temporary) | `tree::{same}` (B5, B6-D2) |
| `pass_infrastructure::BuildIdentityHasher` (`pub(crate)`) | `pass::BuildIdentityHasher` (`pub(crate)`) | `tree::BuildIdentityHasher` (`pub(crate)`) |
| `testing::{DeterministicIdentifierScope, DeterministicIdentifierScopeHandle}` (feature `testing`) | unchanged | **REMOVED** (B1, decision 12) |

Step 0 moves the private modules too:

| HEAD file | After step 0 |
|---|---|
| `src/symbolic/mod.rs` | deleted; its docs merge into `src/expr/mod.rs` |
| `src/symbolic/expression/mod.rs` | `src/expr/mod.rs` (gains `mod symbol_type; mod wire_name; pub use symbol_type::SymbolType;`) |
| `src/symbolic/expression/{alpha,build,builtins,error,literal,node,operation,pprint,registration,screen,sort,wire}.rs` | `src/expr/<same>.rs` |
| `src/symbolic/symbol_type.rs` | `src/expr/symbol_type.rs` (private module) |
| `src/symbolic/wire_name.rs` | `src/expr/wire_name.rs` (private; B2 may delete it under F-025) |
| `src/symbolic/expression/pattern/{mod,rewrite}.rs` | `src/expr/pattern/{mod,rewrite}.rs` |
| `src/symbolic/expression/pattern/core.rs` | `src/expr/pattern/matching.rs` (B6-D3; fixes F-028's `mod core` shadowing) |
| `src/pass_infrastructure/*.rs` | `src/pass/*.rs` |

Later batches create `src/tree.rs` (B5) and
`src/expr/passes/{mod,pretty,rewrite,registration}.rs` (B3, B4). The pass
names inside string constants, such as
`"fhy_core.symbolic.expression.apply_rewrite_rules"` in
`pattern/rewrite.rs:35`, are not module paths. B6 leaves them for B4 and B5,
who own pass naming.

#### 2.3 `lib.rs` (CHANGED)

`lib.rs` declares exactly `pub mod diagnostic; pub mod expr; pub mod
identifier; pub mod interned; pub mod op_attribute; pub mod pass; pub mod
provenance; pub mod tree; pub mod value_domain;` plus private modules. It has
no `pub use`. It replaces the current 21-line doc, which describes the pass
registry as global and describes `arbitrary_precision`. B6 lands the text
below in step 6, once B1, B2 and B5 have made it true:

```rust
//! Core data structures for the `FhY` compiler: identifiers, interned
//! vocabularies, diagnostics and provenance, symbolic expressions with
//! patterns and rewrite rules, and a compiler-pass framework.
//!
//! # Modules
//!
//! Each module depends only on the modules listed before it, except that
//! [`expr`] and [`pass`] are independent and [`expr::passes`] joins them.
//!
//! | Module | Contents |
//! |---|---|
//! | [`identifier`] | [`Identifier`](identifier::Identifier): a name hint and a process-unique id |
//! | [`interned`] | [`InternRegistry`](interned::InternRegistry) and [`Canonical`](interned::Canonical): one canonical value per key |
//! | [`diagnostic`] | diagnostics, notes and validation reports |
//! | [`provenance`] | source positions, spans and where a value came from |
//! | [`op_attribute`] | [`OpAttribute`](op_attribute::OpAttribute): open semantic tags on operations |
//! | [`value_domain`] | [`ValueDomain`](value_domain::ValueDomain): the hierarchy of value classifications |
//! | [`tree`] | the [`Tree`](tree::Tree) trait and iterative walks and rewrites over any tree-shaped IR |
//! | [`expr`] | symbolic expressions, their builders and analyses; [`expr::pattern`] and [`expr::builtins`] |
//! | [`pass`] | compiler passes, pipelines, fixpoint groups, analyses and the pass registry |
//!
//! # Example
//!
//! ```
//! use std::collections::HashMap;
//!
//! use fhy_core::expr::Expression;
//! use fhy_core::identifier::Identifier;
//!
//! let x = Identifier::new("x");
//! let y = Identifier::new("y");
//! let sum = &Expression::from(x.clone()) + 1;
//!
//! let replaced = sum.substitute(&HashMap::from([(x, Expression::from(y))]))?;
//!
//! assert_eq!(replaced.to_string(), "(y + 1)");
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Process-global state
//!
//! The identifier id counter and each [`Interned`](interned::Interned)
//! type's registry are process-global and append-only: an id is never
//! reissued and a canonical value is never replaced. A process must
//! therefore hold one compiled copy of this crate. A second copy would issue
//! ids that collide with the first copy's, and its canonical values would
//! never equal the first copy's. Everything else, including the pass
//! registry, is an owned value.
```

The example depends on B3's `Display` for `Expression` (F-015) and on the
operator and `substitute` signatures B3 ships. In step 6, B6 adjusts it to
whatever landed. The doctest is the check.

#### 2.4 `fhy-core-py` (CHANGED, F-030)

The Python surface of `fhy_core._rs` is unchanged: `__version__`,
`allocate_identifier_id()` and `advance_identifier_counter_past(identifier_id,
/)`. The `_rs.pyi` stub and `tests/test_rs_stub.py` therefore stay valid.

`rust/fhy-core-py/src/lib.rs`:

```rust
//! `PyO3` extension module exposing `fhy-core`'s Rust implementation to
//! Python as `fhy_core._rs`.

mod error;
mod identifier;

/// `fhy_core`'s Rust implementation.
#[pyo3::pymodule(name = "_rs")]
mod rs_module {
    use pyo3::prelude::*;

    #[pymodule_export]
    use super::identifier::{advance_identifier_counter_past, allocate_identifier_id};

    /// Set the extension's `__version__` to the crate version, which the
    /// package compares with its own before it selects the Rust backend.
    #[pymodule_init]
    fn init(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add("__version__", env!("CARGO_PKG_VERSION"))
    }
}
```

`rust/fhy-core-py/src/error.rs` (NEW, all `pub(crate)`):

```rust
/// Conversion of a core error into the Python exception the binding raises.
///
/// A local trait, so the orphan rule allows implementing it for the core
/// crate's error types (it forbids `From<CoreError> for PyErr`).
pub(crate) trait IntoPyErr {
    /// Return the Python exception for this error.
    fn into_py_err(self) -> PyErr;
}

/// `map_err` through [`IntoPyErr`] for any result whose error implements it.
pub(crate) trait IntoPyResult<T> {
    /// Return the value, or the error converted with [`IntoPyErr`].
    fn into_py_result(self) -> PyResult<T>;
}

impl<T, E: IntoPyErr> IntoPyResult<T> for Result<T, E> { … }
```

Each binding module implements `IntoPyErr` for the core errors it raises,
next to its functions. For example, `identifier.rs` has
`impl IntoPyErr for IdSpaceExhausted`, which raises
`RuntimeError("identifier id space exhausted")`. `convert_id_space_exhausted`
is **REMOVED**. The binding functions become
`rust_identifier::try_allocate_id().into_py_result()`.

- **Namespace:** the Python namespace stays **flat** (B6-D5). PyO3
  submodules are attributes, not importable packages (PyO3 #759, per the
  0.29 guide), so a submodule per core module would change
  `fhy_core/identifier.py:139-140` and complicate the stub for no gain.
  The Rust side is already one file per core module. The rule for later
  ports is in section 6 (CONTRIBUTING).
- **Borrowed message text:** today, out-of-range ids depend on PyO3's
  extraction message, which `src/fhy_core/identifier.py:48`
  (`_NEGATIVE_ID_MESSAGE`) copies. This lands **with B1's S-3 change**,
  because the id range changes in both languages in one commit.
  `advance_identifier_counter_past` takes `identifier_id: &Bound<'_, PyInt>`
  and checks the range itself. For any `int` outside the id range B1
  defines, it raises `OverflowError` with a message that is one `const` in
  `fhy-core-py/src/identifier.rs`, and `_PythonIdCounter.advance_past`
  raises the same text. The id is dual-defined, so the text is a contract
  (decision 3). The stub keeps `identifier_id: int, /`.
- **Deferred (decision 5):** the canonical-object identity cache (it arrives
  with the first binding that returns a canonical value) and the
  `rlib`/`cdylib` split for downstream crates (F-005(b)).

#### 2.5 `rust/fhy-core/Cargo.toml` (CHANGED, F-040)

This is the final manifest. B1 removes `testing`, the self dev-dependency
and the four `[[test]]` blocks. B2 adds its binary-format dev-dependency.

```toml
[package]
name = "fhy-core"
description = "Core IR building blocks for the FhY compiler: identifiers, interned tags, diagnostics, symbolic expressions, rewrite rules and a pass manager."
version.workspace = true
edition.workspace = true
rust-version.workspace = true
authors.workspace = true
license.workspace = true
repository.workspace = true
readme = "README.md"
documentation = "https://docs.rs/fhy-core"
keywords = ["compiler", "ir", "symbolic", "term-rewriting", "pass-manager"]
categories = ["compilers", "data-structures"]
# Ship the sources, the integration tests and their golden JSON; never the
# Python corpus generators. Cargo.toml and Cargo.lock are always included.
include = [
    "/src/**/*.rs",
    "/tests/**/*.rs",
    "/tests/golden/*.json",
    "/README.md",
    "/LICENSE",
]

[lib]
name = "fhy_core"

[dependencies]
num-bigint = { workspace = true }
num-traits = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }   # B2 decides whether this stays a normal dependency

[dev-dependencies]
proptest = { workspace = true }
rstest = { workspace = true }
# + B2's non-self-describing format (bincode or postcard)

[lints]
workspace = true
```

I checked the `include` list in a scratch copy with `cargo package --list`.
It ships `src/**`, `tests/**/*.rs`, `tests/golden/*.json`, `README.md`,
`LICENSE`, `Cargo.toml` and `Cargo.lock`, and no `.py` file. At HEAD the
package ships all four Python files from `tests/golden/`.

- **Tests in the package (B6-D8):** the packaged crate keeps its tests, so
  the existing "Testing the Packaged Crate" CI step still means something.
  The cost is about 280 KB of `interned_cases.json` once B1 deletes the
  other two corpora.
- **Explicit test targets:** none remain, unless B1 keeps a fresh-process
  test (section 5.2).

#### 2.6 Workspace `Cargo.toml` lint table (CHANGED)

The following are added under `[workspace.lints.clippy]`. Nothing is loosened.

| Lint | Why | Hits at HEAD (`--lib`) | Lands |
|---|---|---|---|
| `exhaustive_enums = "warn"` | Makes S-6 enforceable. The three enums passes match exhaustively (`ExpressionKind`, `UnaryOperation`, `BinaryOperation`) carry `#[expect(clippy::exhaustive_enums, reason = "passes match every kind")]` | 15 enums | step 6, after B1 to B5 have applied S-6 to their enums |
| `exhaustive_structs = "warn"` | The same for all-public-field structs. It skips structs with private fields | 1 (`NoRegisteredSorts`) | step 6 |
| `self_named_module_files = "warn"` | Pins the `mod.rs` convention that `expr/`, `expr/pattern/`, `expr/passes/`, `pass/` and `tests/it/` use. The only `x.rs` + `x/` pair is `decode.rs`, which B2 deletes | 1 (`decode.rs`) | step 6 |
| `cargo_common_metadata = "warn"` | Keeps the crates.io metadata complete. It skips `publish = false` crates (`fhy-core-py`) by default | 2 (keywords, categories) | step 6, with section 2.5 |

- **`manual_assert_eq`:** already in `pedantic`. The one hit at HEAD is
  B3's (see section 9, note on `expression_pass_stories.rs:287`).
- **Considered and not proposed:**
  - `unwrap_used`/`expect_used`: 11 library `expect`s, all on internal
    invariants.
  - `redundant_pub_crate`: nursery, and it conflicts with the
    `unreachable_pub` policy.
  - `unused_crate_dependencies`: false positives across test targets.
  - `multiple_crate_versions`: only transitive `getrandom`/`syn`
    duplicates, which cannot be acted on.

### 3. Interface delta

| Change | Item | Before | After | Semver | Call sites |
|---|---|---|---|---|---|
| move | 55 items and 3 modules under `symbolic::expression[::pattern\|::builtins]` | `fhy_core::symbolic::expression::…` | `fhy_core::expr::…` | breaking | src (16): `symbolic/expression/{alpha,build,builtins,error,literal,mod,node,operation,pprint,registration,screen,sort}.rs`, `pattern/{core,rewrite}.rs`, `symbolic/symbol_type.rs`, `pass_infrastructure/tree.rs` (doc links and doctests), plus `identifier.rs` and `shipped.rs` (`crate::symbolic`). tests (20): `builtins_scope_stories`, `builtins_stories`, `common/expression`, `common/pattern`, `expression_{builders,literal,node,pass,screen,tree,wire}_stories`, `expression_properties`, `pattern_{properties,rewrite_stories,stories,user_stories}`, `payload_form_stories`, `pprint_{properties,stories}`, `vocabulary_stories`. CONTRIBUTING.md:357 |
| move | `SymbolType` | `fhy_core::symbolic::symbol_type::SymbolType` | `fhy_core::expr::SymbolType` | breaking | 4 files (`screen.rs` via `crate::`, `vocabulary_stories.rs`, `expression_screen_stories.rs`, `expression_properties.rs`) |
| remove | module `symbolic`, `symbolic::symbol_type` | public modules | none | breaking | the rows above |
| move | 37 items of `pass_infrastructure` | `fhy_core::pass_infrastructure::…` | `fhy_core::pass::…` (tree items later `fhy_core::tree::…`, by B5) | breaking | src (9): `pass_infrastructure/{analysis,manager,pass,preserved,registry,tree}.rs` (doctests), `symbolic/expression/{node,screen,pprint,registration}.rs`, `pattern/rewrite.rs`. tests (11): `common/{pass_ir,tree_ir}`, `expression_{pass,tree}_stories`, `pass_infrastructure_{core,manager,run_count,tree,validation}_stories`, `pass_infrastructure_{manager,tree}_properties` |
| rename | private module | `pattern::core` | `pattern::matching` | none (private) | `pattern/mod.rs`, `pattern/rewrite.rs` (`super::core`) |
| change | `fhy_core._rs` init | function-style `#[pymodule] fn _rs` | declarative `#[pymodule(name = "_rs")] mod rs_module` | none (Python API identical) | `rust/fhy-core-py/src/lib.rs` |
| add | `IntoPyErr`, `IntoPyResult` | `convert_id_space_exhausted` fn | local traits, `pub(crate)` | none | `rust/fhy-core-py/src/{error,identifier}.rs` |
| change | `advance_identifier_counter_past` argument | `u64` (PyO3's `OverflowError` text) | `&Bound<PyInt>` with the binding's own range message | Python: message text changes (with B1) | `src/fhy_core/identifier.py:48-50,124-126`; `tests/test_identifier_rust_binding.py:120-133` |
| add | package metadata | no keywords, categories, documentation or include | section 2.5 | none | `rust/fhy-core/Cargo.toml` |
| add | lints | none | section 2.6 | none (lint only) | whole workspace |
| change | test layout | 35 `tests/*.rs` binaries plus `#[path] pub mod` helpers | `tests/it/` plus `mod support` (`pub(crate)`) | none (tests only) | section 8 |

Every public path move is breaking, and all are allowed (S-8). `fhy-core-py`
uses only `fhy_core::identifier`, so step 0 needs no binding change.

### 4. Encapsulation delta

- **Private module paths:** `symbolic/wire_name.rs` becomes the private
  `expr::wire_name`. Its items narrow from `pub(crate)` to `pub(super)`,
  because every user (`operation`, `sort`, `symbol_type`) is now a sibling in
  `expr`. `symbol_type` becomes a private module with one `pub use`.
- **Leaf module for `Pattern`:** `pattern::matching` is a leaf module.
  `Pattern`'s fields stay unreachable from `pattern::rewrite` and from B4's
  `expr::passes::rewrite`.
- **Identity hashing:** `BuildIdentityHasher` stays `pub(crate)` and moves
  with the tree module (B5).
- **No re-exports:** there are no `pub use` re-exports between public
  modules. The only `pub use` of a foreign item is `num_bigint::BigInt` at
  `expr::BigInt`. CI checks this (section 5.4).
- **Test helpers:** test helpers go from `pub` (which only silenced
  dead-code warnings) to `pub(crate)` inside `tests/it/support/`. The
  workspace's `unreachable_pub` then flags any `pub` left over, and
  `dead_code` flags any helper no test uses.
- **Binding crate:** every item in `fhy-core-py` is private or `pub(crate)`.
  `IntoPyErr` is `pub(crate)`.
- **`test_support.rs`:** `is_isolated_run`, `assert_isolated_test_passes`
  and `ISOLATED_TEST_VARIABLE` are deleted (section 5.2).

### 5. Behavior

Step 0 changes no behavior. Its contract is that `cargo test --workspace
--all-features` reports the same number of passed tests (2,422) and ignored
tests (17) before and after the moves. Only the binary count drops.

#### 5.1 Implementation order across batches

| Step | Batch | Content | Needs | Why here |
|---|---|---|---|---|
| 0.1 | B3 (content), landed first | Fix the clippy break at `tests/expression_pass_stories.rs:287` | none | CI is red at HEAD. Every later step must start from green |
| 0.2 | B6 | CONTRIBUTING "Porting to Rust" replacement text (section 6) | none | The rules steer everyone implementing B1 to B5. Left as they are, they require global registries, Python module paths and the decode ordering the batches remove |
| 0.3 | B6 | `symbolic` → `expr` (plus `pattern/core.rs` → `matching.rs`) | 0.1 | This is one atomic rename and must be one commit. Done before other work, it conflicts with nothing, and every later doctest and test is written against final paths. Done last, it would rewrite about 115 import sites after B3 and B4 had just touched them |
| 0.4 | B6 | `pass_infrastructure` → `pass`. Tree items stay in `pass` for now | 0.3 | Same reason. The tree items move once, in step 3, so their call sites change only once |
| 0.5 | B6 | Merge the isolation-safe integration tests into `tests/it` (section 5.2) | 0.4 | Later batches edit tests at their final location, and dead-helper detection works while they add helpers |
| 1 | B1 identity | Reserved ids, `ID_CAP`, `try_new`, delete `shipped.rs`, the `testing` feature, the deterministic tests and corpus, the tag-type corpus, `RegistryGuard` and pinned ids. Remove the last isolation-harness callers and the harness (section 5.2) | 0 | Breaks the `identifier → expr/diagnostic/op_attribute` cycle (F-007), which later layering depends on. Removes the `testing` feature that complicates CI and packaging. Deletes 5 of the 6 non-mergeable binaries |
| 2 | B2 serde, diagnostics, provenance | Plain serde, delete the decode framework, remove `arbitrary_precision`, S-5 errors for diagnostic and provenance | 1 | B2 rewrites `Identifier`/`Canonical` decoding on top of B1's cap and reserved-id semantics. With `shipped.rs` gone, the "initialize defaults before restore" ordering is gone too, so the decode framework can simply be deleted |
| 3 | B5 pass and tree | `src/tree.rs` generic over `C` (F-008), `NodeHandle`/`NodeIdentity` move, owned `PassRegistry`, run counters deleted, `Send` passes, F-009/F-016/F-022 | 0 (and 2, see next column) | `PassContext` stores B2's `Diagnostic`, so B5 follows B2's diagnostic API. B5 may start in parallel with B2 if B2 freezes `Diagnostic`'s constructor and accessor names first |
| 4 | B3 expression core | `Display`, `floor_mod`, n-ary logic, built-in enum, literals, S-5 errors. Create `expr::passes` and move `ExpressionPrettyFormatter` and `register_expression_passes` into it | 1, 2, 3 | `node.rs` and `substitute` need B5's generic `rewrite_tree` (no `PassContext`). The expression wire form follows B2's serde conventions. Built-in parameters follow B1's reserved ids |
| 5 | B4 patterns | Typed captures, `matches`/`apply` methods, `RewriteRuleApplier` → `expr::passes` | 3, 4 | Patterns match B3's expression kinds and built-in enum, and the applier rides on B5's `rewrite_tree` and hooks |
| 6 | B6 finish | Merge `interned_equivalence`, delete `tests/common/`, update the nox corpus settings, add the CI checks and lints, the `lib.rs` overview, both READMEs, the manifest metadata and the package steps | 1 to 5 | Needs the final names (the example and README) and the final enum decisions (lints). The last merges need B1's deletions |

#### 5.2 Test architecture (F-034, F-038)

**Target layout** (end of step 6):

```
rust/fhy-core/tests/
  golden/                     interned_cases.json, generate_interned_cases.py, _golden_support.py
  it/
    main.rs                   //! doc; `mod support;` plus one `mod` per area below
    support/
      mod.rs                  pub(crate) mod expression; hashing; pattern; stack; tree_ir; pass_ir;
      expression.rs  hashing.rs  pattern.rs  stack.rs  tree_ir.rs  pass_ir.rs
    diagnostic_stories.rs
    provenance_stories.rs
    provenance_diagnostic_properties.rs
    tag_type_stories.rs
    interned/{mod,stories,equivalence}.rs
    expr/{mod,builders_stories,literal_stories,node_stories,properties,screen_stories,
          tree_stories,wire_stories,vocabulary_stories,builtins_stories,pass_stories,
          pprint_stories,pprint_properties}.rs
    expr/pattern/{mod,stories,properties,rewrite_stories,user_stories}.rs
    pass/{mod,core_stories,manager_stories,manager_properties,validation_stories}.rs
    tree/{mod,stories,properties}.rs
```

`it/main.rs` doc (the contract for everything in the binary):

```rust
//! Integration tests for `fhy-core`, through its public API only.
//!
//! One binary, so the crate and the dev-dependencies link once and a helper
//! no test uses is reported as dead code. Every test here shares one
//! process with every other: none clears a process-global registry, and
//! none moves the identifier counter further than the ids it allocates.
```

**File map** (HEAD → module; "content owner" is the batch that edits the
file's assertions):

| HEAD file | Step | New location | Content owner |
|---|---|---|---|
| `builtins_stories.rs` | 0.5 | `it/expr/builtins_stories.rs` | B3 |
| `diagnostic_stories.rs` | 0.5 | `it/diagnostic_stories.rs` | B2 |
| `expression_builders_stories.rs` | 0.5 | `it/expr/builders_stories.rs` | B3 |
| `expression_literal_stories.rs` | 0.5 | `it/expr/literal_stories.rs` | B3 |
| `expression_node_stories.rs` | 0.5 | `it/expr/node_stories.rs` | B3 |
| `expression_pass_stories.rs` | 0.5 | `it/expr/pass_stories.rs` | B3 (formatter, registration), B4 (applier) |
| `expression_properties.rs` | 0.5 | `it/expr/properties.rs` | B3 |
| `expression_screen_stories.rs` | 0.5 | `it/expr/screen_stories.rs` | B3 |
| `expression_tree_stories.rs` | 0.5 | `it/expr/tree_stories.rs` | B3 |
| `expression_wire_stories.rs` | 0.5 | `it/expr/wire_stories.rs` | B2 (wire form), B3 (n-ary logic) |
| `vocabulary_stories.rs` | 0.5 | `it/expr/vocabulary_stories.rs` | B2 (F-025 serde derive), B3 |
| `pprint_stories.rs`, `pprint_properties.rs` | 0.5 | `it/expr/pprint_{stories,properties}.rs` | B3 |
| `pattern_{stories,properties,rewrite_stories,user_stories}.rs` | 0.5 | `it/expr/pattern/*.rs` | B4 |
| `pass_infrastructure_{core,manager,validation}_stories.rs`, `…_manager_properties.rs` | 0.5 | `it/pass/*.rs` | B5 |
| `pass_infrastructure_tree_{stories,properties}.rs` | 0.5 | `it/tree/{stories,properties}.rs` | B5 |
| `provenance_stories.rs`, `provenance_diagnostic_properties.rs` | 0.5 | `it/*.rs` | B2 |
| `tag_type_stories.rs` | 0.5 | `it/tag_type_stories.rs` | B1 |
| `interned_stories.rs` | 0.5 | `it/interned/stories.rs` | B1 |
| `payload_form_stories.rs` | 0.5 | `it/payload_form_stories.rs`, unless B2 deletes it first (it tests `MapOnly`, which S-2 deletes) | B2 |
| `common/{expression,hashing,pattern,stack,tree_ir,pass_ir}.rs` | 0.5 | `it/support/*.rs` | B6 (move), each batch (content) |
| `interned_equivalence.rs` | 6 | `it/interned/equivalence.rs` (or the lib unit tests, see below) | B1 |
| `common/mod.rs` (golden replay loop) | 6 | folded into `it/interned/equivalence.rs`, its only remaining user | B6 |
| `deterministic_identifiers_{stories,constants,equivalence}.rs` | 1 | deleted (decision 12) | B1 |
| `builtins_scope_stories.rs` | 1 | deleted with the `testing` feature | B1 |
| `tag_type_equivalence.rs` (+ `golden/tag_type_cases.json`, `generate_tag_type_cases.py`) | 1 | deleted (triage: decision 3) | B1 |
| `pass_infrastructure_run_count_stories.rs` | 3 | deleted, or `it/pass/run_count_stories.rs` if B5 keeps a local run statistic | B5 |

**Mechanics of step 0.5:**

- **Includes:** each `#[path = "common/X.rs"] pub mod Y;` becomes
  `use crate::support::X as Y;`, so the file's `use Y::{…}` lines do not
  change.
- **Dead helpers:** helpers that the `dead_code` lint now reports are
  deleted in the same commit. `grep` finds these candidates with no user
  outside `common/`: `common/expression.rs` `build_dag_node`,
  `build_dag_node_specification_strategy`, `build_finite_float_strategy`,
  `MAX_DAG_NODES`; `common/pass_ir.rs` `AnalysisRunCounters`, `BoxNode`,
  `RunHook`; `common/stack.rs` `SMALL_STACK_BYTES`; `common/tree_ir.rs`
  `build_toy_node`, `RewriteHook`, `ToyNode`. Some may be used inside their
  own helper file; the compiler decides.
- **One-user helpers:** a helper with exactly one user moves into that
  user's module. For example, `common/pattern.rs`'s `build_alternatives`,
  `expect_match` and `match_infallibly`, and `common/tree_ir.rs`'s
  `build_chain`.
- **Isolation binaries:** these stay separate until their batch deletes
  them: `deterministic_identifiers_*` and `builtins_scope_stories` (they
  need the `testing` feature), `tag_type_equivalence` (it clears global
  registries), `pass_infrastructure_run_count_stories` (it reads a
  process-wide total) and `interned_equivalence` (it shares `common/mod.rs`
  with two of them). `autotests` stays on, so no manifest change is needed
  for these.

**Binaries that stay separate in the end:** none, apart from `it`, the lib
unit-test binary and the doctests.

- **Fresh-process tests:** a test that needs a fresh process gets its own
  `[[test]]` target with exactly one `#[test]`, a file comment saying why,
  and an entry in the CI target list (section 5.4). It never re-executes
  the test binary.
- **Expected need after B1: none.** With fixed reserved ids, the "first use
  after exhaustion keeps the shipped statics" tests are moot, and B1
  deletes them with `shipped.rs`. Counter exhaustion is testable in unit
  tests through the existing local-counter seams, `take_next_id(&AtomicU64)`
  and `advance_past(&AtomicU64, id)` (`identifier.rs:209, 224`).
- **The one candidate:** a serde-level test that a payload id at
  `ID_CAP - 1` decodes. It moves the global counter by about 2^63, which
  would break any counter-observing test sharing its process (F-033). If B1
  keeps such a test, it is that one `[[test]]` (suggested name
  `id_cap_decode`).

**The isolation harness (F-038) goes away rather than being hardened.**
Its callers are:

- `identifier.rs:835-853`
- `shipped.rs:162-206` (the rstest `case_N_*` names)
- `op_attribute.rs:466-515`
- `value_domain.rs:1176-1237`
- `diagnostic.rs:551-576`

All of them test first use of shipped statics or exhaustion by payload,
which B1 removes. B1 owns all five callers, including `diagnostic.rs`,
because they test B1's shipped statics. The rule is that whoever removes
the last caller deletes `is_isolated_run`, `assert_isolated_test_passes` and
`ISOLATED_TEST_VARIABLE` from `src/test_support.rs` in the same commit.
Otherwise `dead_code` fails `-D warnings`. So B1 deletes them.

**`interned_equivalence` location:** this depends on B1. The corpus
replays a `clear` op. If B1 makes `InternRegistry::clear` `#[cfg(test)]` or
`pub(crate)` (decision 2), an integration test cannot call it. The replay
then moves into `src/interned.rs`'s unit tests, reading
`include_str!("../tests/golden/interned_cases.json")`, and the nox filter
below uses `--lib`. If `clear` stays callable on a local registry, the
replay lives in `it/interned/equivalence.rs`, reading
`include_str!("../../golden/interned_cases.json")`.

**nextest (B6-D6): not in CI.** The reasons:

- `cargo test` runs every test of a binary in one process on parallel
  threads. That is the stricter check for the shared-process hazards that
  remain (the global counter and the intern registries).
- nextest runs each test in its own process, which would hide exactly those
  hazards.
- nextest does not run doctests.
- The consolidated suite needs no process isolation.

The suite must still pass under `cargo nextest run` (test-style), and
contributors may use it locally.

#### 5.3 Golden corpora (F-036)

- **No new CI step.** Staleness is already checked:
  `tests/test_golden_corpora.py` (in the tree since b3a0a5d, before the
  audit) reruns every `generate_*.py` in a fresh interpreter and compares
  its output with the committed JSON, outside the `provenance` block. It
  runs in every CI `tests` job, on both backends.
- **Why not `git diff --exit-code`:** a regenerate-and-diff step would fail
  on every run. `provenance` records `git_commit` and `python_version`
  (`_golden_support.py:62-68`), which change with every commit and runner.
- **After B1,** only `generate_interned_cases.py` remains.
  `test_golden_directory_has_generators` still holds, and the interned test
  already asserts its shared constants (`defaults_catalogue`) against Rust
  (`interned_equivalence.rs:84-110`).
- **`noxfile.py` in step 6:**

```python
# The integration-test binary the expanded replays run in.
RUST_TEST_TARGET = "it"   # "--lib" instead if B1 moves the interned replay into unit tests


class ExpandedGoldenCorpus(NamedTuple):
    """How to generate and replay one generator's expanded random corpus."""

    options: str
    test_filter: str
    variable: str


EXPANDED_GOLDEN_CORPORA = {
    "generate_interned_cases.py": ExpandedGoldenCorpus(
        options="--seed 7 --random-count 2000 --max-ops 60 --keys a,b,c,d,e",
        test_filter="interned::equivalence::",
        variable="FHY_INTERNED_CORPUS",
    ),
}
```

  The `cargo test` call in `golden_expanded` becomes `cargo test --locked -p
  fhy-core --test it -- --ignored <test_filter>`, without `--features
  testing`. `_EXPANDED_REPLAY_PASSED` (`1 passed;`) stays, and the filter
  selects exactly one ignored test. B1 removes the other two entries when it
  deletes their generators, as the CONTRIBUTING "Freeze the golden corpus"
  rule already requires.
- **Docstrings to update in step 6:**
  - `generate_interned_cases.py`, lines 5-6 and the `DEFAULTS_CATALOGUE`
    comment, which name `rust/fhy-core/tests/interned_equivalence.rs`;
  - the `tests/test_golden_corpora.py` module docstring (B1 already drops
    "the deterministic-identifier scope").

#### 5.4 CI (`.github/workflows/python-package.yml`, job `rust`)

This is the final job, in step 6. New steps are marked `# NEW`.

```yaml
  rust:
    runs-on: "ubuntu-latest"
    timeout-minutes: 20
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy, rustfmt
      - name: Cache Rust Build Artifacts
        uses: Swatinem/rust-cache@v2
      - name: Code Formatting
        run: cargo fmt --all --check
      - name: Code Linting
        run: cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
      - name: Integration Test Targets   # NEW
        # Every integration test shares the `it` binary; a test that needs a
        # fresh process is its own target and is listed here on purpose.
        run: |
          expected="it"
          actual="$(cargo metadata --no-deps --format-version 1 --locked \
            | jq -r '.packages[] | select(.name == "fhy-core") | .targets[]
                     | select(.kind == ["test"]) | .name' | sort | paste -sd' ')"
          if [[ "$actual" != "$expected" ]]; then
            echo "::error::fhy-core test targets are '$actual', expected '$expected'."
            exit 1
          fi
      - name: Unit, Integration & Doc Testing
        run: cargo test --workspace --locked --all-features
      - name: Documentation   # NEW
        env:
          RUSTDOCFLAGS: -D warnings
        run: cargo doc --workspace --no-deps --locked
      - name: Public Paths   # NEW
        # Each public item has exactly one path: no module re-exports another
        # public module's item, and no item name appears under two paths.
        run: |
          doc=target/doc/fhy_core
          if grep -l 'id="reexports"' $(find "$doc" -name index.html); then
            echo "::error::a module re-exports an item that already has a public path."
            exit 1
          fi
          duplicates="$(grep -oE 'href="[^"#]+\.html"' "$doc/all.html" | grep -v '"\.\./' \
            | sed -E 's#^href="(.*/)?[a-z]+\.([A-Za-z0-9_]+)\.html"$#\2#' | sort | uniq -d)"
          if [[ -n "$duplicates" ]]; then
            echo "::error::items with more than one public path: $duplicates"
            exit 1
          fi
      - name: Packaging
        run: cargo package --locked -p fhy-core
      - name: Package Contents   # NEW
        # The include list ships Rust sources, tests and golden JSON only.
        run: |
          unexpected="$(cargo package --list --locked -p fhy-core | grep -Ev \
            '^(Cargo\.toml(\.orig)?|Cargo\.lock|\.cargo_vcs_info\.json|README\.md|LICENSE|src/.+\.rs|tests/.+\.rs|tests/golden/[a-z_]+\.json)$' || true)"
          if [[ -n "$unexpected" ]]; then
            echo "::error::the package ships unexpected files: $unexpected"
            exit 1
          fi
      - name: Testing the Packaged Crate
        # CHANGED: no --features testing; the feature no longer exists.
        run: |
          tar -xzf target/package/fhy-core-*.crate -C "$RUNNER_TEMP"
          cd "$RUNNER_TEMP"/fhy-core-*/
          cargo test --locked
```

I ran both new shell checks ("Public Paths" and "Package Contents") on HEAD
output, and both pass.

- **`rust-msrv`:** the second `check … --all-features` line goes, since no
  feature is left to add. The step comment about `testing` goes with it.
- **Layering:** a CI step for the layering was considered and not added,
  because `grep` cannot see `super::` paths. B6 checks the layering once, in
  step 6, with `grep -rn 'crate::pass' rust/fhy-core/src/expr --include=*.rs
  | grep -v '^rust/fhy-core/src/expr/passes/'` and `grep -rn 'crate::expr'
  rust/fhy-core/src/{pass,tree.rs}`. Both must print nothing.
- **Unchanged:** the `golden-expanded` job itself.

#### 5.5 README and docs (F-040)

- **`rust/fhy-core/README.md`:**
  - Opening: "This crate is the Rust implementation of the `fhy_core`
    Python package. Where a concept is defined in both languages
    (`identifier`, `interned`), the Rust behavior matches Python's and a
    golden corpus checks it. Elsewhere Rust defines the behavior."
  - "Modules": one line per module of section 2.1, the same list as the
    `lib.rs` table.
  - "One copy per process": restricted to the id counter and the intern
    registries.
  - "Using it": `fhy-core = "0.x"` from crates.io, replacing the "not
    published" paragraph and the git dependency.
  - The `testing` bullet goes, and the MSRV line stays.
- **Root `README.md` lines 156-170:** the same changes.
  - Line 158: list the crate's modules instead of "identifiers, interning,
    and the tag types".
  - Line 160: crates.io instead of "not published".
  - Lines 165-170: the `testing` paragraph and the TOML block go (B1 deletes
    the feature; B6 writes the text).
- **Python narrative in rustdoc:** `value_domain.rs:60, 151-154` is B1's
  (value domains). `testing.rs` is deleted by B1. B6 does a final pass in
  step 6: rustdoc outside `identifier` and `interned` mentions Python only
  in a doc line starting "Matches the Python implementation:" (S-4).

### 6. CONTRIBUTING.md "Porting to Rust": replacement text

The intro paragraph (lines 284-289) stays. Each changed or new rule below
replaces the section of the same or former name. The text lands in step 0.2.

**"One extension module per process"** (CHANGED: this narrows the rule to
the state that stays global and records F-005 as deferred):

> All Rust code that uses *FhY* Core's Rust types compiles into a single
> Python extension module. The identifier id counter and each `Interned`
> type's `InternRegistry` are Rust `static`s, which exist once per compiled
> copy of the crate, and PyO3 creates a separate Python type for each
> extension module. A second extension linking the crate would issue ids that
> collide with the first one's, keep registries whose canonical instances
> never match, and fail `isinstance` checks against the first one's classes.
> A downstream *FhY* package that gains Rust code depends on the crate as a
> Rust library and is compiled into one combined extension module; it never
> ships an extension of its own that links the crate. No downstream crate
> links `fhy-core` yet, so `fhy-core-py` does not yet offer the library form
> that such a combined module needs; it gains one before the first
> downstream crate does.

**"Registries are process-global statics"** is replaced by **"Process-global
state is limited to identity"**:

> Exactly two kinds of state are process-global: the identifier id counter
> and each `Interned` type's `InternRegistry`. Both are append-only: an id is
> never reissued, and a canonical value is never replaced or removed.
> Everything else a port keeps between calls, such as the pass registry, run
> statistics or caches, is an owned value that its user creates and passes
> explicitly. Where the Python API needs one shared instance, the binding
> holds it in the extension's module state. A new process-global `static`
> with interior mutability needs the maintainer's agreement and a line in
> this section. Tests never clear a process-global registry; a test that
> needs an empty or controlled registry builds a local one.

**"Decoding checks the payload before its side effects"** is replaced by
**"Serialization is plain serde"**:

> `fhy-core` serializes with `#[derive(Serialize, Deserialize)]` wherever it
> can, in shapes Rust defines. There is no `__type__`/`__data__` envelope in
> the core crate: the binding adds it where Python's serialization framework
> embeds a Rust value in a Python container. Serde impls must work with
> non-self-describing formats as well as JSON, and a type whose impl is
> written by hand has a round-trip test through a binary format. Decoding
> has two side effects, both monotonic: an `Identifier` advances the id
> counter past its id, and a `Canonical<T>` interns its value. A decode that
> fails partway may leave the counter advanced and some canonical values
> registered. The affected types document this; decoding is not ordered to
> prevent it.

**"Replacing a Python class"**: one bullet changes and the rest stay. The
"Freeze the golden corpus" bullet gains a first sentence:

> - Freeze the golden corpus. Golden corpora exist only for concepts defined
>   in both languages, today `identifier` and `interned`. Once a module's
>   Python implementation is deleted, …(rest unchanged)

**"Module paths follow the Python package"** is replaced by **"Module
paths follow Rust layering"**:

> A Rust module's path follows the crate's layering, not the Python package.
> Each public item has exactly one public path, every `pub use` is explicit
> (no globs), and CI rejects an item re-exported under a second path. A
> module depends only on the layers before it:
>
> 1. `identifier`, `interned`
> 2. `diagnostic`, `provenance`, `op_attribute`, `value_domain`
> 3. `tree`
> 4. `expr` (with `expr::pattern` and `expr::builtins`) and `pass`, which do
>    not depend on each other
> 5. `expr::passes`, the passes over expressions, which depends on both
>
> A private module is never named `core`, which shadows the `core` crate. A
> port records its Python module in this table, the one place that maps
> Python paths to Rust ones:
>
> | Python | Rust |
> |---|---|
> | `fhy_core.identifier` | `fhy_core::identifier` |
> | `fhy_core.traits.interned` | `fhy_core::interned` |
> | `fhy_core.diagnostic` | `fhy_core::diagnostic` |
> | `fhy_core.provenance` | `fhy_core::provenance` |
> | `fhy_core.op_attribute` | `fhy_core::op_attribute` |
> | `fhy_core.value_domain` | `fhy_core::value_domain` |
> | `fhy_core.symbolic.symbol_type` | `fhy_core::expr` (`SymbolType`) |
> | `fhy_core.symbolic.expression` (`core`, `errors`, `pprint`, `sort`) | `fhy_core::expr` |
> | `fhy_core.symbolic.expression.builtins` | `fhy_core::expr::builtins` |
> | `fhy_core.symbolic.expression.pattern` (`core`, `rewrite`) | `fhy_core::expr::pattern`; the rule-applier pass is in `fhy_core::expr::passes` |
> | `fhy_core.symbolic.expression.passes` | `fhy_core::expr::passes` |
> | `fhy_core.pass_infrastructure` | `fhy_core::pass`; tree traversal is in `fhy_core::tree` |

**"Errors belong to their module"** (CHANGED: this narrows "the same
message" to the concepts defined in both languages):

> Each module defines the error types for its own operations, one type per
> family of related operations; the crate has no crate-wide error enum. A
> public error is a `#[non_exhaustive]` enum, or a struct with structured
> fields, so callers match variants and fields rather than text. `Display`
> writes one lowercase line with no trailing period and does not repeat the
> text of its `source()`, which returns the underlying cause. `Display` and
> `std::error::Error` are implemented by hand. The binding converts each
> core error it raises through its local `IntoPyErr` trait. For `identifier`
> and `interned`, which are defined in both languages, it raises the Python
> implementation's exception class with the same message. For every other
> module, Rust defines the behavior: the binding raises the exception class
> the replaced Python API documents, with the Rust error's `Display` text.

**New section, "Python parity is limited to dual-defined concepts"**, after
"Errors belong to their module":

> Rust matches the Python implementation's behavior and text only for
> concepts defined in both languages at once, today `identifier` and
> `interned`. Code that exists only to match Python starts its doc comment
> with "Matches the Python implementation:". Everywhere else, Rust
> conventions decide: `true`/`false`, Rust's shortest round-trip float
> formatting, lowercase error messages, and `Display` impls instead of
> Python `repr` emulation. Rustdoc describes Rust behavior and does not
> narrate the Python implementation.

**New section, "Binding crate layout"**:

> `fhy-core-py` declares `fhy_core._rs` with one declarative `#[pymodule]`
> in `lib.rs`. Each core module's bindings live in a file of the same name
> and are exported with `#[pymodule_export]`. The Python namespace of `_rs`
> stays flat, since PyO3 submodules cannot be imported as packages.
> `src/fhy_core/_rs.pyi` is written by hand, and `tests/test_rs_stub.py`
> checks its names and parameters against the built extension.

**New section, "Rust test layout"**:

> `fhy-core`'s integration tests form one binary, `rust/fhy-core/tests/it/`,
> with one module per area mirroring the crate's modules. Shared helpers
> live in `tests/it/support/` at `pub(crate)`, so a helper no test uses is a
> dead-code warning. A test that needs a fresh process, because it moves
> process-global state further than an ordinary test tolerates, is its own
> `[[test]]` target with exactly one `#[test]` and a comment saying why, and
> it is added to the target list the CI `rust` job checks. Nothing
> re-executes a test binary to get a fresh process.

**"Canonical values keep their identity in Python"** stays, with one added
closing sentence:

> …the core crate never holds Python objects. The cache is added with the
> first binding that returns a canonical value.

The test-support sentence in the old module-path rule ("`fhy_core.testing_patches`
becomes the feature-gated `fhy_core::testing`") goes with that rule
(decision 12).

### 7. Error and panic model

- **Binding:** `IdSpaceExhausted` → `RuntimeError("identifier id space
  exhausted")`, unchanged. An id outside B1's range → `OverflowError(<the
  binding's const message>)`, which also covers negative ints and ints of
  2^64 or more that PyO3 used to reject. Binding functions do not panic.
- **CI checks:** each new step fails with an `::error::` line naming the
  offending targets, items or files.
- **No other contract changes:** step 0 changes no error or panic
  contract.

### 8. Non-goals

- **Deferred (decision 5, F-005):** the `rlib` library form of
  `fhy-core-py` and the canonical-object identity cache.
- **Deferred (decision 13):** a crate split (foundation, symbolic, passes).
  Once the layering holds, a later split re-exports from `fhy-core` and
  breaks no one.
- **Not done:** stub generation (PyO3's `experimental-inspect` and
  `maturin generate-stubs` are marked experimental in the 0.29 guide).
- **Superseded by decision 3:** a Python-recorded expression corpus
  (F-036's `generate_expression_cases.py`). The symbolic area is
  Rust-defined.
- **Not in CI:** nextest (section 5.2).
- **Left to other batches:** the content of every moved test (their
  assertions, error-text pinning (F-035), and the F-037 items other than
  the ones listed below).

### 9. Test plan

**Keep:** every test in the 29 files merged into `it` keeps its body. Only
its `use` lines change. Also kept: `tests/test_rs_stub.py`,
`tests/test_golden_corpora.py`, `tests/test_crate_license.py` and
`tests/test_identifier_rust_binding.py` (except the rows below).

**Modify** (none weakens a test):

- **Step 0.3/0.4 imports:** the 20 + 11 test files and the source files listed in
  section 3 change `fhy_core::symbolic::…` and
  `fhy_core::pass_infrastructure::…` to `fhy_core::expr::…` and
  `fhy_core::pass::…`. The doctests in those source files change the same
  way. The acceptance check is that passed and ignored counts are unchanged.
- **Step 0.5 includes:** `#[path] pub mod` becomes `use
  crate::support::… as …`.
- **Step 6 golden file:** `interned_equivalence.rs` moves (5.2); its
  `include_str!` path and `mod common` are replaced.
- **With B1:** in `tests/test_identifier_rust_binding.py:120-133`, the
  `OverflowError` message cases expect the binding's own message instead of
  PyO3's.
- **`noxfile.py`:** `golden_expanded` settings and command (5.3).
  `EXPANDED_GOLDEN_CORPORA` is covered by the session's own "unconfigured
  generator" check.

**Delete:**

- **`tests/common/`** (whole directory, by step 6). Its contents move to
  `it/support/` or into the one remaining user.
- **Dead helpers** that `dead_code` reports after step 0.5 (candidates in
  5.2). They are unused, so deleting them changes no assertion.
- **In `src/test_support.rs`:** `is_isolated_run`,
  `assert_isolated_test_passes` and `ISOLATED_TEST_VARIABLE`, deleted by
  B1 with the last caller (5.2). The eight rstest cases in `shipped.rs` and
  the seven `*_in_isolation` pairs are B1's deletions, justified in B1.

**Cross-cutting F-037 items, and who edits them:**

| Item | File (after 0.5) | Owner | Required outcome |
|---|---|---|---|
| Hand-indented `.with_…` chains at column 0 inside `proptest!` (8 lines; rustfmt does not format macro bodies) | `it/expr/pprint_properties.rs` | B3 (it rewrites these calls for `expr.display`) | Bodies indented as rustfmt would. See B6-D11 |
| `COMPOSED_NAMES`, `NATIVE_FUNCTION_NAMES` and `NATIVE_CONSTANT_NAMES`, re-listed in several tests | `it/expr/builtins_stories.rs:34-57, 378` | B3 (F-024) | Lists derived from the built-in enum or catalogue iterator, and one expected table |
| `build_plus_zero` and `describe_fired`, duplicated | `it/expr/pass_stories.rs`, `it/expr/pattern/rewrite_stories.rs` | B4 | Moved to `support::pattern` |
| `build_file` and `build_named`, duplicated | `it/provenance_stories.rs`, `it/provenance_diagnostic_properties.rs` | B2 | Moved to a `support::provenance` |
| `check_get`, `check_require`, `replay_case` and `lock_replay`, duplicated | `*_equivalence.rs` | B1 | Gone once the tag-type and deterministic replays are deleted |
| About 16 copy-pasted identity pass types | `it/pass/core_stories.rs` | B5 | Generated by one macro (F-031) |
| Isolation harness and `shipped.rs`'s hard-coded `case_N_*` names | `src/test_support.rs`, `src/shipped.rs` and the four other users | spec B6, executed by B1 | Deleted (5.2) |

**Regression test per finding:**

- **F-028:**
  - the crate-level doctest in `lib.rs` (2.3), which reaches `expr` and
    `identifier` by their single paths;
  - the CI "Public Paths" step;
  - a doctest per moved module (`expr`, `expr::pattern`, `expr::builtins`,
    `pass`) using the new path.
  - Also `pattern::matching` exists and no module is named `core`: `find
    rust/fhy-core/src -name core.rs` prints nothing (checked in step 6).
- **F-030:**
  - `tests/test_rs_stub.py::test_stub_names_match_the_built_extension` and
    `::test_stub_function_parameters_match_the_built_extension` pass
    unchanged after the switch to the declarative module;
  - `test_identifier_rust_binding.py`'s `RuntimeError` cases pass through
    `IntoPyErr`.
- **F-034:**
  - the CI "Integration Test Targets" step (expected `it`);
  - a `dead_code` warning for any unused helper, which `-D warnings` makes
    fatal.
- **F-036:** `tests/test_golden_corpora.py::test_committed_corpus_matches_its_generator[generate_interned_cases.py]`
  (existing), plus `nox -s golden_expanded` replaying through
  `--test it`.
- **F-038:** none needed. The harness no longer exists, and the target-list
  check stops a new re-exec harness from coming back as a separate binary.
- **F-040:**
  - the CI "Documentation" step (`RUSTDOCFLAGS=-D warnings`);
  - the "Package Contents" step;
  - `clippy::cargo_common_metadata`;
  - the packaged-crate `cargo test --locked`.

**New story, property or adversarial tests:** none. This batch adds
structure and CI checks, not behavior. The adversarial case for the checks
is manual, once, in step 6. Add a `pub use crate::tree::Tree;` to
`pass/mod.rs` and a `tests/extra.rs`, and confirm that "Public Paths" and
"Integration Test Targets" both fail. Then revert.

### 10. Findings covered, and findings I believe were misjudged

**Covered:**

- F-028 (paths, `mod core`; the split is deferred per decision 13)
- F-030 (declarative module, `IntoPyErr`, borrowed message with B1; cache
  and rlib deferred)
- F-034
- F-036 (CI part)
- F-037 and F-038 (cross-cutting parts; ownership table in 9)
- F-040
- the CONTRIBUTING text for decisions 2, 3, 4 and 13, and the workspace
  lints

**Misjudged or already fixed:**

1. **F-036, "goldens never checked for staleness":** wrong at the audited
   commit. `tests/test_golden_corpora.py` (b3a0a5d) regenerates every corpus
   in every CI `tests` job and compares it outside `provenance`. The
   suggested `git diff --exit-code` would fail on every run, because the
   `provenance` block records the commit and interpreter. The "shared
   constants" suggestion is already met for `interned` (`defaults_catalogue`
   is asserted). The expression-corpus suggestion is superseded by decision
   3. What is left of F-036 is the nox and docstring cleanup in 5.3.
2. **F-030, "hand-maintained stubs; check stubs in CI":** the check exists.
   `tests/test_rs_stub.py` compares names in both directions and parameter
   shapes against the built extension, in CI on the Rust backend. What is
   left is the declarative module, `IntoPyErr` and the borrowed message.
3. **F-034's "four binaries rely on running alone":** the isolation-bound
   binaries are `builtins_scope_stories`, `deterministic_identifiers_constants`,
   `pass_infrastructure_run_count_stories` and `tag_type_equivalence`, which
   clears global registries. `interned_equivalence` is merge-safe, since it
   uses only local registries. It stays separate until step 6 only because
   it shares `common/mod.rs` with two binaries B1 deletes.
4. **F-038 needs no hardened harness.** With B1's reserved ids and the
   existing `&AtomicU64` seams, no remaining test needs a child process. The
   audit's options ("panic when the env var is missing", `harness = false`)
   become unnecessary.
5. **Not in the audit:** CI never runs `cargo doc` with `-D warnings`. The
   baseline's clean docs are not guarded. Added in 5.4.
6. **The HEAD clippy break (B3):** `tests/expression_pass_stories.rs:287` is
   `assert!(outcome.output() == &build_doubling_dag(&a, 64))`, and clippy
   suggests `assert_eq!`. On failure, `assert_eq!` Debug-prints a 64-level
   doubling DAG, which is the F-037 hang (2^64 occurrences). The fix is
   `#[expect(clippy::manual_assert_eq, reason = "Debug of a 64-level doubling DAG does not finish")]`,
   or a comparison with a non-Debug failure message, unless B3's F-003
   `Debug` fix lands first and makes `Debug` DAG-safe.

**Decisions for sign-off:**

- **B6-D1:** `expr::passes`, not `pass`, holds the expression passes.
- **B6-D2:** `NodeHandle` and `NodeIdentity` move to `tree` (B5 to
  confirm).
- **B6-D3:** `pattern/core.rs` is renamed to `pattern/matching.rs`, not
  folded into `mod.rs`.
- **B6-D4:** the path moves, the CONTRIBUTING text and the test
  consolidation land first (steps 0.2 to 0.5). The CONTRIBUTING text
  therefore describes rules before the code fully follows them, for the
  length of B1 to B5.
- **B6-D5:** the `_rs` Python namespace stays flat.
- **B6-D6:** nextest is not added to CI.
- **B6-D7:** no `git diff` golden step (section 10, item 1).
- **B6-D8:** the published crate ships its tests and golden JSON (about
  280 KB), which keeps the packaged-crate test step.
- **B6-D9:** four lints are added (2.6): `exhaustive_enums`,
  `exhaustive_structs`, `self_named_module_files`,
  `cargo_common_metadata`.
- **B6-D10:** fresh-process tests are single-test `[[test]]` targets
  listed in CI. None is expected after B1; the only candidate is an
  `ID_CAP - 1` serde-decode test.
- **B6-D11:** proptest bodies are formatted by hand (the minimum). The
  alternative is to enable proptest 1.11's `attr-macro` feature
  (`#[property_test]`), so that rustfmt formats every property body. That
  is a dev-dependency feature change, so it is the user's call.
- **B6-D12:** where `interned_equivalence` ends up depends on B1's
  visibility for `InternRegistry::clear` (5.2).
- **Keywords and categories** in 2.5 are proposals.

---

## Implementation notes

- **Step 0.1 (R-7):** the `#[expect(clippy::manual_assert_eq, ...)]` sits
  on the test function. On the `assert!` statement itself clippy still
  reports the lint and the expectation goes unfulfilled.
- **Step 0.3 and 0.4 (R-24):** the modules take the `foo.rs` + `foo/`
  layout, so B6 §2.2's `src/expr/mod.rs` and `src/expr/pattern/mod.rs`
  are `src/expr.rs` and `src/expr/pattern.rs`, and `src/pass/mod.rs` is
  `src/pass.rs`. The private `pass_infrastructure/pass.rs` becomes
  `src/pass/compiler_pass.rs` now rather than in B5, because `pass::pass`
  trips `clippy::module_inception` (B5 §2.1 names the file the same way).
- **Step 0.5 (D-18):** `tests/it` uses the same layout: `it/support.rs`,
  `it/interned.rs`, `it/expr.rs`, `it/expr/pattern.rs`, `it/pass.rs` and
  `it/tree.rs` next to their directories, instead of the `mod.rs` files
  that B6 §5.2's target layout shows.
- **Step 0.5, dead helpers:** with every helper `pub(crate)`, `dead_code`
  reports nothing. Each §5.2 candidate is still used inside its own helper
  file, so no helper is deleted.
- **Step 0.5, one-user helpers:** not moved. 21 helpers have exactly one user
  module (for example `support::pattern`'s `build_alternatives`,
  `expect_match` and `match_infallibly`, and `support::tree_ir`'s
  `build_chain`). Step 0.5 was kept to a pure move, and the batch that owns
  each file's content can move a helper when it edits that file. Step 6
  checks that none is left.
- **Step 0.5, type-name assertions:** six tests in
  `it/pass/core_stories.rs` compare `std::any::type_name` output, which
  names the test crate and module path. Their expected strings change from
  `pass_infrastructure_core_stories::…` to `it::pass::core_stories::…` and
  `it::support::pass_ir::…`. No other test text changes.
- **Step 1, F-001 regression (B1 §8.3):** the serde-level test that
  decoding `2^63 - 1` leaves construction working runs in its own binary,
  `tests/id_cap_decode.rs`, with one test, rather than in
  `identifier::tests` with no isolation. Once the counter reaches `2^63`,
  every fresh id is at or above the cap, so no fresh identifier sharing the
  process can round-trip through serde; the lib and `it` binaries are full
  of such round trips. The same binary covers the restore-first order of
  F-032 for the shipped tags and the built-in parameters, so
  `it::identifier_stories::shipped_identifiers_hold_their_reserved_ids`
  does not restore `2^63 - 1` first. For the same reason the postcard
  property `serde_round_trips_through_postcard_for_any_payload_id` decodes
  only already-issued ids, not all of `0..ID_CAP`.
- **Step 1, payload ids under `arbitrary_precision`:** `PayloadId` asks for
  a `u64` as B1 §2.1 says. Until B2 removes `arbitrary_precision`,
  `serde_json` rejects a negative, fractional or oversized number read
  through a `serde_json::Value` with its own "invalid number", not the
  range message. The nested-path identifier tests assert the exact message
  for the paths that read the JSON text (bare text, note kind, op
  attribute, value-domain parent) and only the error position for the
  `serde_json::Value` and expression paths **[weakens, until B2]**;
  `it::diagnostic_stories::note_decode_rejects_malformed_payloads` decodes
  from text. The expression path is kept rather than dropped.
- **Step 1, `IdentifierWire`:** it is `pub(crate)`, not private, so
  `expr::wire` can keep checking a whole expression payload before it
  restores any identifier until B3 replaces that wire format;
  `IdentifierPayload` is gone. `wire.rs`'s `build_node` is now generic over
  the serde error so a restore error needs no new `ExpressionBuildError`
  variant.
- **Step 1, wire structs:** `IdentifierWire`, the tag wire struct and the
  value-domain level carry `#[serde(expecting = ...)]`, so errors say
  "expected an identifier" or "expected a described tag" instead of naming
  a private struct.
- **Step 1, value-domain wire form (R-4):** a list of levels, root first
  and the domain itself last, each `{"name": .., "description": ..}`. An
  empty list is rejected with `invalid_length`. Each level registers as it
  is read, so a conflict at one level leaves the levels before it
  registered.
- **Step 1, shipped defaults:** `DescribedTag::create_shipped` and
  `described_tag::require_shipped` are `pub(crate)`, so each vocabulary
  builds and looks up its defaults without touching `DescribedTag`'s
  fields. The sealing trait lives in `pub(crate) mod sealed` with
  `#[expect(unnameable_types)]`, because the vocabulary impls live in
  `op_attribute` and `diagnostic`, not beside the trait.
- **Step 1, `clear` on an unused registry:** it does nothing and does not
  run `create_defaults`; the first use registers the defaults, which is the
  state a clear restores.
- **Step 1, test deletions not in B1 §8.1:** `tag_type_equivalence` and its
  corpus go in the F-001 commit, because they used the removed public
  `Identifier::restore`. In `it::payload_form_stories` (B2's file) the
  identifier, tag, value-domain and note cases and
  `a_refused_sequence_payload_registers_nothing` go, since derived decoders
  accept the sequence form. The four tag-type stories B1 §8.4 names move to
  `it::tag_type_stories`.
- **Step 1, `decode.rs`:** `DeferredPayload` and `decode/buffered.rs` lost
  their last users and are deleted; `Decode`, `deserialize_via_payload`
  and `deserialize_map_only` remain for `expr::wire` and `provenance`.
- **Step 1, Python:** the pinned ids of the built-in constants in
  `tests/symbolic/expression/test_registry.py` move from 8-11 to
  65,544-65,547 with the counter start. `_PythonIdCounter` takes a
  `next_id` test seam. There is no `_create_reserved_identifier` and no
  `None` entry in `EXPANDED_GOLDEN_CORPORA` (R-3).
- **Step 1, `reserved_identifiers_take_no_id_from_the_counter`:** the two
  anchors are retried until they are one id apart, since a parallel test
  can draw an id between them; a reserved identifier that drew an id would
  keep them at least two apart on every try.

- **Step 2, F-012 not done:** `arbitrary_precision` stays on. Turning it
  off breaks `expr::wire`: big-integer literals as JSON integers, float
  tokens beyond `f64`, and the expression JSON round-trip properties (7
  tests). B3 removes it together with its BigInt-as-decimal-string wire
  format. B3 also moves `serde_json` to `[dev-dependencies]` (`expr::wire`
  is its last `src` user), drops the workspace manifest comment and the
  `lib.rs` `arbitrary_precision` paragraph, restores B1's weakened
  nested-path message checks, and adds B2 §8.3's two F-012 regression
  tests to `it::serde_format_stories`. Those tests fail while the feature
  is on.
- **Step 2, `decode.rs`:** only `deserialize_map_only` and `MapOnly` are
  left, for `expr::wire`. `Decode` and `deserialize_via_payload` are gone,
  and `ExpressionPayload` is now private. B3 deletes the file.
  `it::payload_form_stories` keeps only its two expression cases, since
  map-only decoding is still `expr::wire`'s behavior, instead of being
  deleted as B2 §8.1 says. B3 deletes it with the wire format.
- **Step 2, `python_text.rs`:** the exact decimal normalization,
  `format_normalized_decimal` and the shared positional and scientific
  writers move to `expr/literal/decimal.rs`. `format_float_repr` and
  `format_bool` move to `expr/literal/python_repr.rs`. All are private to
  `literal`, and the tests move unchanged. Literal `Display` and
  `canonical_key` still write the Python text. B3 switches them to Rust
  formatting, deletes `python_repr.rs`, and decides the decimal notation.
- **Step 2, D-16:** the leading `//` is kept. So B2 §8.1's four
  normalization case changes and §8.3's
  `file_provenance_collapses_leading_separators_to_one_root` are void, and
  `file_provenance_keeps_paths_that_normalize_differently_apart` still
  asserts `//a` and `/a` differ.
- **Step 2, F-038:** B1 already deleted the isolation harness, so B2 §5.6
  and its three `test_support` tests are void.
- **Step 2, decode error categories:** where a provenance is expected,
  `serde_json` classifies a JSON value that is neither a string nor a map,
  and a second variant key, as `Category::Syntax`, not `Data`. So
  `assert_decode_rejected` takes the expected category, and those three
  cases (`two_top_level_keys`, `null_caller`, `not_a_map`) expect `Syntax`.
- **Step 2, zero positions:** `Position` decodes through serde's
  `NonZeroU64` (B2 §2.2). A zero line or column therefore gets serde's
  message, and the `zero_line`, `zero_column` and `invalid_position`
  rejection cases check only the category, not `PositionError`'s text as
  B2 §8.1's list of crate-invariant cases implies.
- **Step 2, `diagnostic.rs` unit tests:** B1 had already replaced the nine
  tests B2 §8.2 deletes with characterizations of the derived decode and a
  postcard round trip. They are kept.
- **Step 2, not in B2's scope now:** `it::serde_format_stories` leaves out
  B3's types (`Expression`, sorts, operations) and the big-integer
  decimal-string test. The `vocabulary_stories` change in B2 §8.1 depends
  on B3's `FromStr`. Both are left to B3. The postcard dependency keeps
  B1's `features = ["alloc"]` rather than `use-std`. CONTRIBUTING's
  serialization rule was already rewritten in step 0.2.
- **Step 2, pass call sites (R-9):** `manager.rs` keeps the report text as
  the verification failure's detail, now `report.to_string()`.
  `PassContext::report` keeps its `Option<String>` detail. With a
  `Cow<'static, str>` source, the manager and the validation pipeline pass
  `pass_name.to_owned()` and `validator_name.to_owned()`.

- **Step 3, commit order:** the `skip` hook and the `Validator` trait
  (F-022) land before the `PassError` rework (F-014). With `Nested`, a
  pass run as a validator always records its error diagnostic, so the
  silent-failure path is reachable only through a direct `Validator`;
  landing `Validator` first keeps every commit's tests meaningful.
- **Step 3, `ContextLender`:** B5 §8.2 deletes it with F-008. It is deleted
  one commit later, with `skip`, because the `noop_output` tests called
  the hook directly through it until then.
- **Step 3, `PassContext::report` (R-9):** it records the `Diagnostic` as
  given, source included; it does not overwrite the source with the pass
  name. `report_text(level, message, detail)` keeps its `Option` detail
  (B5 §2.4 "unchanged") and attributes the text to the running pass.
- **Step 3, verification diagnostic (R-9):** its message is the error's
  one-line `Display` and it has no detail; the report is reached through
  `PassErrorKind::Verification`. B5 §8.5's `detail().is_some()` checks
  assert `None` instead.
- **Step 3, registry tests under D-13:** `registry_allows_aliases_without_renaming_the_pass`
  is void (no aliases). Two same-named types from two modules both key
  under `"Fold"`, so the second registration is `NameTaken`
  (`same_named_pass_types_in_two_modules_take_one_name`), and
  `pass_registries_are_independent_values` registers them in two
  registries. `pass_registry_does_not_change_pass_names` becomes
  `pass_registry_keys_a_pass_by_its_own_name`. Registration builds one
  instance to read the name, so `register_pass_does_not_call_the_factory`
  becomes `registry_register_builds_one_instance_to_read_its_name`. The
  factory is expected to build instances that share that name.
- **Step 3, R-6:** there is no `verified` flag in the cache buckets.
  `pass_manager_verifies_each_node_once_per_run` (B5 §8.10, 2 validations)
  becomes `pass_manager_verifies_every_changed_output_even_a_node_seen_before`
  (3 validations).
- **Step 3, merge-only transfer:** `AnalysisCache::invalidate` has no
  caller left and is deleted with its four unit tests; the transfer unit
  tests pin the merge semantics. `get_or_insert_with` is private.
- **Step 3, `AnalysisId::of`:** bounded by `A: Analysis`, like `preserve`
  and `is_preserved` (D-14; B5 had the marker). Test marker types that
  were plain structs implement `Analysis`.
- **Step 3, `RewriteTreeError::Rebuild`:** being `#[non_exhaustive]`, it
  cannot be built outside the crate, so
  `rewrite_tree_error_describes_a_refused_rebuild` gets its error from a
  refused rebuild.
- **Step 3, `PassValidator`:** its `PassError` carries no diagnostics; they
  stay in the validator's context and so in the report, which is why no
  silent-failure diagnostic is added for it. `ValidatorRecord` stores the
  start and end of its slice rather than a `Range`.
- **Step 3, `short_type_name`:** a nominal type (a path with at most one
  trailing generic list) is shortened without allocating. Other names go
  through a scratch buffer and are still returned borrowed when the result
  is one contiguous piece, as for `fn(i32) -> i32`.
- **Step 3, layering:** `tree`'s rustdoc names the pass adapters in plain
  code text, not intra-doc links, so a grep for `crate::pass` in `tree`
  finds nothing.
- **Step 3, B4's F-009 side:** `RuleApplier` still records a firing whose
  rewrite returns the matched node; only the tree no longer counts it as a
  change. `pattern/rewrite_stories.rs` and `pattern/properties.rs` flip
  their two identity-rewrite pins accordingly.

- **Step 4 (B3a), scope and order:** B3 is split. B3a lands the literal
  enum (F-026, F-013), `Expression::display` (F-015), `floor_mod` (F-010),
  `!` (F-024's `Not` part), the n-ary `Logical` node and the bounded `Debug`
  (F-003, D-5, R-7), the per-operation errors and the screen error
  (F-014), and the associated builders (F-029). `BuiltinFunction`,
  `Callee`, `FunctionName`, the `From` conversions, the serde derives and
  `FromStr`, `BooleanScreen`, the flat node-list wire format, floats and
  big integers as strings, the `arbitrary_precision` removal and
  `expr::passes` are B3b's.
- **Step 4 (B3a), decimal notation:** a decimal displays positionally
  (B3 §5.3); `format_normalized_decimal` and the scientific writers go with
  `python_repr.rs`. `Decimal` stores a `BigInt` coefficient, so its unit
  tests compare `coefficient.to_string()` with the old digit strings.
- **Step 4 (B3a), test helper:** B3 §8.1's `build_decimal_literal` always
  builds a `Decimal`, integer texts included (`"5"` is the decimal 5). The
  call sites that meant an integer text parse it with
  `LiteralValue::parse_text` through a local `build_parsed_literal`.
- **Step 4 (B3a), literal patterns:** until B4 decides, a literal pattern
  matches the same variant with an equal raw value: floats by IEEE `==`
  (`0.0` matches `-0.0`, a NaN matches nothing), integers and decimals by
  value. Spellings no longer exist, so the pattern tests' "spelled
  differently" cases move from the rejecting table to the matching one.
- **Step 4 (B3a), interim wire form:** the envelope format stays for B3b
  to replace. A decimal literal is written as a JSON string holding its
  `Display` text, and a string decodes as a `Decimal`, never as an `Int`,
  so round trips keep the variant. A logical node is
  `{"__type__": "logical_expression", "__data__": {"operation": "and" |
  "or", "operands": [...]}}`; fewer than two operands is refused with "a
  logical node needs at least 2 operands, got {n}". `LogicalOperation`'s
  serde goes through `impl_wire_name_traits` (expecting "a logical
  operation name: and or or") until B3b's derive.
- **Step 4 (B3a), `Expression::call`:** it keeps the `&str` name, so it
  stays fallible and returns `FunctionNameError`, which has only `Empty`
  for now; B3b adds `Builtin`, `FunctionName` and `Callee` and makes the
  call infallible. `FunctionNameError` lives in `error.rs` until B3b
  creates `callee.rs`.
- **Step 4 (B3a), `substitute`:** it maps `RebuildError::ChildCount` to
  `unreachable!`, because `rewrite_tree` rebuilds a node from exactly its
  own children and `PiecewiseError` has no count variant. B3 §6 says
  `substitute` maps no error to a panic; this is the one arm that would.
- **Step 4 (B3a), non-exhaustive errors in tests:** `PiecewiseError` is
  `#[non_exhaustive]`, so `expression_piecewise_errors_are_piecewise_errors`
  (B3 §8.2) cannot match it without a wildcard arm from `tests/it`; it
  maps each refusal through a match with a wildcard and asserts the
  variants it gets.
- **Step 4 (B3a), test names:** `BooleanScreen` is B3b's, so the new
  screen tests of B3 §8.2/§8.3 are named after what exists:
  `validate_logical_operands_reports_the_operand_index_of_a_logical_operand`
  and `screen_error_parent_is_none_only_at_a_predicate_root`.
- **Step 4 (B3a), screen message tests:** the tests that pinned the old
  sentence now assert the refusal's operand, parent and position; the new
  text is pinned once, in the position table, plus a predicate-root case
  and a `BooleanPosition` display table.
- **Step 4 (B3a), `Debug` tests:** the text is not pinned. The node tests
  check a deep tree and a 64-level doubling DAG stay under 64 KiB, and the
  `expression_debug_is_bounded` property checks every generated DAG stays
  under 256 KiB. Two small-stack DAG tests in `node_stories` also switch
  from `assert!(a == b)` to `assert_eq!`/`assert_ne!`, which the bounded
  `Debug` makes safe.
- **Step 4 (B3a), patterns over logical nodes:** there is no logical
  pattern shape until B4, so the mirroring pattern property captures a
  logical node whole, as a leaf, and a binary pattern never matches one.
- **Step 4 (B3a), kept for B3b:** the `From<UnaryExpression>` and sibling
  impls and `expression_from_node_struct_rewraps_the_node` stay (its `%`
  becomes `floor_mod`); `FloorMod`'s symbol stays `%`.
