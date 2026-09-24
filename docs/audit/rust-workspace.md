# Audit: Rust workspace (`rust/fhy-core`, `rust/fhy-core-py`)

## Scope

- **In scope:** every `.rs` file in the workspace (about 11.5k source lines and
  29k test lines), both `Cargo.toml` manifests, the workspace lint and
  dependency tables, the golden-corpus mechanism, and the CONTRIBUTING
  "Porting to Rust" rules as they shape the Rust design.
- **Out of scope:** the Python package itself, except as a reference for why
  the Rust looks the way it does.
- **Commit audited:** `e489f41` on `dev-rust`.
- **Brief:** check that the crate architecture is sound and the code is
  idiomatic Rust. The user is willing to change behavior where Python
  faithfulness produces unidiomatic Rust, so every finding records whether it
  is Python-driven and what behavior change the fix implies.

## Current state

The audit reading was done with no working toolchain. The baseline below
was run afterwards at `412e234`, using the rustup toolchain in
`~/.cargo/bin` (stable 1.98).

- **fmt:** clean.
- **clippy** (`--all-targets --all-features -D warnings`): **fails** on one
  lint, `clippy::manual_assert_eq`, at
  `tests/expression_pass_stories.rs:287`. Commit 412e234 added it, and CI's
  stable clippy fails the same way.
- **Stricter clippy** (`-W clippy::nursery`): mostly `const fn` (127) and
  `use_self` (108), plus 15 redundant clones and 4 early-droppable
  temporaries. These are phase-5 style work, not findings.
- **Tests:** 2,422 pass, 0 fail, 17 ignored, across 39 test binaries.
- **Docs** (`RUSTDOCFLAGS=-D warnings`): clean.
- **Unsafe:** none (`unsafe_code = "forbid"`).
- **Dependencies:** lean; the only duplicates are transitive (`getrandom`,
  `syn`).
- **Not installed:** cargo-llvm-cov, cargo-audit, cargo-deny, cargo-mutants,
  cargo-nextest, cargo-semver-checks.
- **Since the audit:** 4db96b9 fixed **F-002** and the DAG re-walk in
  **F-023**, and 412e234 added `RewriteRuleApplier`,
  `ExpressionPrettyFormatter` and `register_expression_passes`.
- **Verification:** each High finding and several Medium ones were checked
  against the code.

## Overall assessment

The line-level Rust is careful and mostly idiomatic:

- private fields with borrowed accessors;
- sealed traits where they matter;
- iterative `Drop`, `Hash` and tree walks with explicit stacks;
- lawful literal equality (NaN and -0.0 handled through a canonical form);
- no `unsafe`, and almost no panics reachable from input;
- a strict lint table and strong tests.

Almost every significant finding comes from one source: reproducing
Python's *runtime model*, not just its behavior. These carry-overs are:

- process-global counters and registries (class variables and module
  singletons);
- deserializers with ordered global side effects;
- a monkeypatch ported as a thread-local hook;
- a CPython text-formatting layer in the core crate;
- a decorator-style pass registry;
- exception-shaped errors;
- dataclass-shaped constructors with `Option` mode arguments.

Several CONTRIBUTING rules *require* these shapes: registries are
process-global statics, module paths follow the Python package, and bindings
raise the same message. Those rules contradict the org's own Rust guidelines
("no global mutable state; pass a context explicitly").

The highest-leverage decision is **F-004: an explicit session/context value
that owns ids and interned vocabularies**. On its own it removes or
simplifies F-001, F-005, F-006, F-007, F-012, F-018 and most of the
test-isolation machinery (F-031, F-032). Make this decision before more
modules are ported under the current rules.

---

## Findings

### Summary

| ID | Title | Sev | Conf | Python-driven |
|---|---|---|---|---|
| F-001 | One crafted payload exhausts the id space; every later `Identifier::new` panics | High | High | yes |
| F-002 | Structural `==`/`Hash`/`free_identifiers` are exponential on shared DAGs | High | High | partly |
| F-003 | Deep right-folded conjunctions cannot round-trip; recursive `Debug`/`Serialize` can abort | High | High | yes |
| F-004 | Process-global mutable state is the foundation of identity | High | High | yes |
| F-005 | "One extension per process" rule cannot be met by downstream Rust | High | Medium | yes |
| F-006 | Pass names and run counters come from a global registry keyed by `type_name` | High | High | yes |
| F-007 | Shipped statics get ids lazily; `identifier` depends on the whole crate via `shipped` | High | Medium | partly |
| F-008 | `walk_tree`/`rewrite_tree` need a `PassContext` outsiders cannot create; IR depends on passes | High | High | partly |
| F-009 | A rewrite returning its own input counts as a change; fixpoint loops never terminate | Medium | High | yes |
| F-010 | `/` and `%` on `Expression` mean Python true-division and floor-modulo | Medium | High | yes |
| F-011 | Deserialize performs global side effects, driving a bespoke decode framework | Medium | High | yes |
| F-012 | Workspace-wide `arbitrary_precision` leaks to dependents; serde impls are JSON-only | Medium | High | yes |
| F-013 | CPython text emulation lives in the core crate and drives `Display` | Medium | Medium | yes |
| F-014 | Errors are Python-exception-shaped: stringly, unmatchable, cause duplicated or dropped | Medium | High | yes |
| F-015 | No `Display` for expressions or reports; multi-line error `Display` | Medium | High | yes |
| F-016 | Analysis cache applies mutable-IR invalidation to immutable handles and pins nodes | Medium | Medium | yes |
| F-017 | Pipelines are `!Send` while IR handles must be `Send + Sync` | Medium | High | no |
| F-018 | The `testing` feature ports a monkeypatch into `Identifier::new` | Medium | Medium | yes |
| F-019 | Pattern captures are string-keyed; constructors use `Option` for "any" | Medium | High | yes |
| F-020 | `Canonical<T>` and `T` have two different equalities; bare `T: Deserialize` escapes interning | Medium | High | yes |
| F-021 | Constructors register globally and return `InternOutcome`; `Option`-mode parameters | Medium | High | yes |
| F-022 | Pass lifecycle hooks: split `should_run`/`noop_output`, validators forced into pass shape | Medium | High | yes |
| F-023 | Screen API copies optional keyword mappings and re-walks shared subtrees | Medium | Medium | yes |
| F-024 | Built-in functions are named by string; `IntoOperand` duplicates `From`/`Into` | Medium | Medium | yes |
| F-025 | Hand-maintained `ALL_*` arrays emulate serde derive; no `FromStr` | Medium | High | yes |
| F-026 | `LiteralValue` emulates Python's `str\|int\|float\|bool` union | Medium | Medium | yes |
| F-027 | Macro emulates mixin inheritance for described tags | Low | Medium | yes |
| F-028 | Module tree copies Python package layout; crate split is blocked | Low | High | yes |
| F-029 | Python module-function API shape (`get_*`, free functions, `list_*`) | Low | High | yes |
| F-030 | Binding crate structure won't scale | Low | Medium | no |
| F-031 | Test isolation depends on hand-maintained rules around global state | Medium | High | yes |
| F-032 | Deterministic-id golden replay passes only because of corpus case order | Medium | High | yes |
| F-033 | Pinned-id headroom scheme can collide under parallel tests | Medium | Medium | yes |
| F-034 | 35 integration-test binaries; `#[path] pub mod` helper sharing hides dead code | Medium | High | no |
| F-035 | Tests pin Python wording, serde's messages, private type names and `type_name` | Medium | High | partly |
| F-036 | Symbolic area has no Python-recorded golden corpus; goldens never checked for staleness | Medium | High | yes |
| F-037 | Test assertions that hang, test nothing, or re-implement the SUT | Low | High | no |
| F-038 | Isolated-child-process harness is fragile and passes vacuously under `--ignored` | Low | High | no |
| F-039 | Performance nits on hot paths | Low | Medium | mixed |
| F-040 | Crate docs and README drifted | Low | High | no |

Totals: 8 High, 24 Medium, 8 Low.

---

### High

#### F-001: One crafted payload exhausts the id space; every later `Identifier::new` panics

- **Severity:** High · **Confidence:** High · **Category:** Correctness / Errors
- **Location:** `rust/fhy-core/src/identifier.rs:74-81` (`new`, no fallible
  form), `:95-102` (`restore`), `:153-156` (`allocate_id` panics), `:282-313`
  (`PayloadId` accepts up to `u64::MAX - 1`)
- **Semver impact of the suggested fix:** breaking
- **Python-driven:** yes. Python restores payload ids verbatim and was
  given the same 2^64 cap (`src/fhy_core/identifier.py:80-131`), so both
  backends share this failure mode; it is not new to the port.

**Issue:** Deserializing `{"id":18446744073709551614,"name_hint":"x"}`
anywhere in a payload sets `NEXT_ID` to `u64::MAX`. From then on, every
`Identifier::new` in the process panics, and panics inside `LazyLock`
initializers poison them. `Identifier::restore` (public) and the Python
binding's `advance_identifier_counter_past` do the same. A test pins the
behavior (`identifier.rs:836-846`).

**Why it matters:** One corrupt or untrusted file permanently breaks every
later compilation in a long-lived process, such as a Python session or a
compile server.

**Suggested fix:** Pick one:
- (a) remap payload ids to fresh ids on decode, keeping identity within the
  payload (this also fixes F-007);
- (b) reject restorable ids above a cap such as 2^63.

Either way, add `Identifier::try_new`. With F-004 (a session), exhaustion
becomes a per-session `Result`.

#### F-002: Structural `==`/`Hash`/`free_identifiers` are exponential on shared DAGs

- **Severity:** High · **Confidence:** High · **Category:** Performance / Correctness
- **Location:** `symbolic/expression/node.rs:161-178` (`is_tree_equal`),
  `:411-421` (`free_identifiers`), `:519-526` (`PartialEq`), `:530-564`
  (`Hash`); `alpha.rs` equivalence
- **Semver impact of the suggested fix:** none
- **Python-driven:** partly. Python uses identity `__eq__`/`__hash__`, which
  is O(1). Rust switched to structural equality but kept plain tree walks.

**Issue:** `Expression` is designed for sharing (`substitute` and
`rewrite_tree` are linear in distinct nodes), but none of these walks keep a
visited set:
- `Hash` pushes every child every time, so hashing even a *single* DAG
  `e_{k+1} = e_k + e_k` visits 2^k nodes.
- `==` skips only pointer-equal pairs, so comparing two separately built
  equal DAGs is also exponential.
- Every `HashMap<Expression, _>` lookup costs O(tree size) plus a Vec
  allocation.

**Why it matters:** DAGs are normal output of substitution, rewriting and
CSE-style builders. At depth about 40 these calls never return.

**Suggested fix:**
- Cache a 64-bit structural hash per node at construction.
- Make `Hash` write the cached value.
- In `==`, reject early on hash mismatch and memoize visited
  `(NodeIdentity, NodeIdentity)` pairs.
- Give `free_identifiers` a visited set.

No observable behavior change.

#### F-003: Deep right-folded conjunctions cannot round-trip; recursive `Debug`/`Serialize` can abort

- **Severity:** High · **Confidence:** High · **Category:** Correctness / API
- **Location:** `symbolic/expression/build.rs:48-67, 377-398` (right-fold);
  `wire.rs:207-214, 375-463` (recursive encode/decode); `node.rs:208-225`
  (documented limits); derived `Debug` at `node.rs:252-301` and
  `error.rs:195`; `pattern/*` `Pattern` derives
- **Semver impact of the suggested fix:** breaking (for an n-ary node)
- **Python-driven:** yes. It ports `_build_right_folded_binary_tree` and
  binary-only `LOGICAL_AND`/`LOGICAL_OR`.

**Issue:**
- **Depth grows with operand count.** `build_logical_and` over n operands
  builds a chain n-1 levels deep.
- **Decode limit.** `serde_json::from_str` refuses beyond about 62 levels
  (pinned at `tests/expression_wire_stories.rs:356`). A conjunction of about
  60 comparisons therefore serializes and then fails to decode.
- **Stack overflows abort.** `Serialize` and derived `Debug` recurse per
  level, so `unwrap()`, `assert_eq!` or `{:?}` on a deep expression or on
  `NonBooleanLogicalOperandError` can overflow the stack. That is a process
  abort, not an `Err`.
- **Tests can hang.** Two tests would hang instead of failing: they
  Debug-print a 2^64-occurrence DAG in a failure arm
  (`tests/expression_tree_stories.rs:383`,
  `tests/pattern_rewrite_stories.rs:713`).

**Why it matters:** Large conjunctions are the most common deep shape in a
constraint IR.

**Suggested fix:**
- Add an n-ary `Logical { op, operands: Box<[Expression]> }` (or
  `All`/`Any`) node, with a new wire type id that Python must mirror.
- Write `Debug` for `Expression` by hand, iteratively or with a depth cap.
- If the wire format is frozen, make decode iterative and expose
  `from_json_str` with `disable_recursion_limit`.

#### F-004: Process-global mutable state is the foundation of identity

- **Severity:** High · **Confidence:** High · **Category:** Foundations / Concurrency / API
- **Location:** `identifier.rs:40` (`static NEXT_ID`); `interned.rs:74-84`
  (`intern_registry() -> &'static`), `:244-246` (public `clear()`);
  `pass_infrastructure/registry.rs:62-73`; `lib.rs:1-11`;
  CONTRIBUTING.md:291-310
- **Semver impact of the suggested fix:** breaking
- **Python-driven:** yes. It mirrors class-level registries, the global
  counter, `InternedMixin._interned_instances` and `clear_interned_registry`.

**Issue:**
- **Globals by construction.** Identity rests on a process-wide id counter
  and per-type `InternRegistry` statics. The `Interned` trait makes a
  per-session registry impossible by construction.
- **A public reset.** `clear()` is public on those statics and has no
  production caller. Any dependent can call it and silently make every other
  holder's handles non-canonical.
- **Unbounded growth.** Registries only grow, so decoding untrusted input
  grows them without bound.
- **Required by CONTRIBUTING.** The "Registries are process-global statics"
  rule requires all of this.

**Why it matters:** It is the root of:
- F-001, F-005, F-006, F-007 and F-018;
- the test-isolation machinery in F-031 to F-033 (registry guards,
  id-counter locks, headroom constants, subprocess re-exec, single-test
  binaries);
- the lack of support for two independent compilations in one process.

**Suggested fix:**
- Add a `Session`/`Context` value that owns the id allocator, the interned
  vocabularies and (optionally) the pass registry. Pass it explicitly to
  constructors, and to decoders via `DeserializeSeed` or an explicit
  `restore(&mut Session)` step (F-011).
- Pre-register shipped defaults when a session is created.
- `Identifier` can become a small `Copy` index into the session's arena.
- The Python binding holds one session in module state, so Python-visible
  behavior is unchanged.
- **Minimum interim step:** gate `InternRegistry::clear` behind
  `cfg(any(test, feature = "testing"))`.

#### F-005: "One extension per process" rule cannot be met by downstream Rust

- **Severity:** High · **Confidence:** Medium · **Category:** Foundations / Dependencies
- **Location:** CONTRIBUTING.md:291-302; `rust/fhy-core-py/Cargo.toml:12-14`
  (`crate-type = ["cdylib"]` only); `src/fhy_core/_backend.py:23`;
  `lib.rs:3-11`
- **Semver impact of the suggested fix:** none for the core crate
- **Python-driven:** indirectly. The rule exists only because of F-004.

**Issue:** The rule says downstream FhY packages with Rust code must compile
into *one combined* extension with fhy-core. Nothing makes that possible:
- the bindings live in a cdylib-only crate that other crates can't link;
- `fhy_core` always imports its own `fhy_core._rs`.

The first downstream wheel that links `fhy-core` therefore gets a second
copy of every static: colliding ids and non-matching canonical instances.

**Suggested fix:** Pick one:
- (a) remove the globals (F-004);
- (b) split the bindings into an rlib with `pub fn register(py, m)` plus a
  thin cdylib, and document how a downstream cdylib provides `fhy_core._rs`;
- (c) export the counter and registries through a `PyCapsule` C-API, the
  numpy pattern.

#### F-006: Pass names and run counters come from a global registry keyed by `type_name`

- **Severity:** High · **Confidence:** High · **Category:** API / Concurrency
- **Location:** `pass_infrastructure/registry.rs:62-73, 89-95, 106-138,
  182-275`; `pass.rs:82-92` (default `name`/`description`), `:417`
  (`record_run`); `error.rs:230-251`
- **Semver impact of the suggested fix:** breaking
- **Python-driven:** yes. It mirrors `CompilerPass._registry`,
  `_run_counts` and `@register_pass`.

**Issue:**
- **Two global locks per run.** Default `CompilerPass::name()` locks a
  global mutex and allocates on every call, and `record_run` locks again
  (three times for `WalkPass`/`RewritePass`).
- **Non-unique keys.** `names_by_type` is keyed by
  `std::any::type_name::<P>()`, which std says is not unique. Counters are
  keyed by the *stripped* display name, so `a::Fold`, `b::Fold`, `Fold<i32>`
  and `Fold<i64>` share a counter.
- **Registration renames passes.** Registering a pass changes the `name()`
  of existing instances, so earlier runs are filed under the old name.
- **Wrong or unchanged names.** Registering one type for two I/O pairs makes
  `create_pass("a")` report the name `"b"`, and re-registration is a no-op
  that doesn't restore the name.
- **Nothing uses it.** Nothing outside tests uses the registry or the
  counters.

**Suggested fix:**
- Default `name()` becomes a pure `Cow<'static, str>` from the stripped
  type name, overridable, with no lookup.
- Registry becomes an owned `PassRegistry` value, keyed by `TypeId`, held by
  the driver or the session.
- Run statistics are returned in `PassManagerResult`.
- If none of this is needed yet, delete the registry and the counters.

#### F-007: Shipped statics get ids lazily; `identifier` depends on the whole crate via `shipped`

- **Severity:** High · **Confidence:** Medium · **Category:** Correctness / Foundations
- **Location:** `described_tag.rs:150-159` (`LazyLock` +
  `Identifier::new_unscoped`); `value_domain.rs:238-269`;
  `identifier.rs:197-200`; `shipped.rs:1-26`
- **Semver impact of the suggested fix:** none (the id values change)
- **Python-driven:** partly. Python creates its constants at import time in
  a fixed order, so their ids are stable. The lazy Rust statics lost that.

**Issue:**
- **Lazy ids aren't portable.** A shipped tag's id depends on how many ids
  the process allocated before first use, so shipped tags serialized in one
  process don't decode as the same tag in another.
- **Silent aliasing.** A restored id can even collide with a *different*
  shipped name in the reader. The payload then decodes as that shipped value
  and the conflicting description is silently dropped (the TODO at
  `interned.rs:409`). The golden tests work around this by normalizing
  shipped ids into "slots" (`tests/tag_type_equivalence.rs:133-155`).
- **A cycle through `shipped`.** To stop F-001-style exhaustion from
  starving the lazy statics, `try_advance_counter_past` calls
  `initialize_shipped_statics()`. That makes the leaf `identifier` module
  depend on `diagnostic`, `op_attribute`, `value_domain` and
  `symbolic::expression::builtins`, which is a cycle.
- **Rules only comments enforce.** The list in `shipped.rs` is maintained by
  hand, and a comment-only rule forbids re-entrant restores (deadlock).
- **Collateral effects.** The Python extension builds every Rust shipped
  static on its first unpickle. F-032 is a direct consequence.

**Suggested fix:**
- Reserve a fixed low id block for shipped identifiers (the counter starts
  at N) and use the same reservation in Python.
- `shipped.rs`, `new_unscoped` and the cycle then disappear.
- The general cross-process aliasing needs F-001(a).

#### F-008: `walk_tree`/`rewrite_tree` need a `PassContext` outsiders cannot create; IR depends on passes

- **Severity:** High · **Confidence:** High · **Category:** API / Foundations
- **Location:** `pass_infrastructure/tree.rs:80-135, 181, 281-289`;
  `context.rs:28` (`new` is `pub(super)`), `:38` (`new_standalone` is
  `pub(crate)`); `symbolic/expression/node.rs:446`; `pattern/rewrite.rs:457`;
  `tests/common/tree_ir.rs:579-586` (`ContextLender`)
- **Semver impact of the suggested fix:** breaking
- **Python-driven:** partly. In Python the visitor *is* the pass.

**Issue:**
- **Unusable outside a pass.** The public generic traversals and every
  `TreeVisitor`/`Rewriter` hook take `&mut PassContext`, which has no public
  constructor. So outside code can call them only from inside a pass hook.
- **Crate-private back door.** `Expression::substitute` and
  `apply_rewrite_rules` use `new_standalone` with a synthetic pass name,
  allocating on every call.
- **Test workaround.** The tests run a fake pass just to borrow a context.
- **Inverted layering.** The IR data layer depends on the pass layer, which
  blocks any crate split (F-028).

**Suggested fix:**
- Move `Tree` and the walkers to a foundation module (e.g. `fhy_core::tree`)
  generic over a caller-chosen context, e.g. `Rewriter<N, C = ()>`.
- `WalkPass`/`RewritePass` pass `PassContext` as `C`.

---

### Medium

#### F-009: A rewrite returning its own input counts as a change; fixpoint loops never terminate

- **Location:** `pass_infrastructure/tree.rs:224-258` (`finish_node`:
  `replacement.or(rebuilt)`); `pattern/rewrite.rs:120-143`, `:297-309`,
  `:433-435`; pinned by `tests/pattern_properties.rs:576-596`,
  `tests/pattern_rewrite_stories.rs:415-431`
- **Confidence:** High · **Python-driven:** yes ("identity is preserved iff
  zero rules fire")

**Issue:** A rewriter or rule that returns `Some(node.clone())`, or a rule
that returns what it matched, is recorded as firing, and every ancestor is
rebuilt. `is_changed()`/`did_change` then report true on every pass. So:
- a `FixpointPassGroup` fails with non-convergence;
- the rustdoc-recommended "repeat until unchanged" loop never ends.

**Suggested fix:** Treat a replacement that is `ptr_eq` to its input as "no
change". Optionally skip rebuilding when every child is `ptr_eq`. Optionally
add a bounded `apply_rewrite_rules_to_fixpoint`. Behavior change: `fired()`
no longer lists identity rewrites.

#### F-010: `/` and `%` on `Expression` mean Python true-division and floor-modulo

- **Location:** `symbolic/expression/build.rs:317-318`; `operation.rs:134-139`
- **Confidence:** High · **Python-driven:** yes (`__truediv__`, `__mod__`)

**Issue:** In Rust, integer `/` truncates and `%` takes the dividend's sign.
Here `x / 4` builds real division and `x % -3` builds floor-modulo. Index
arithmetic written by Rust users silently gets the other semantics.

**Suggested fix:** Remove the `Div`/`Rem` impls and keep named builders
(`true_divide`, `floor_divide`, `floor_modulo`). If the operators must stay,
at least document the semantics on the impls.

#### F-011: Deserialize performs global side effects, driving a bespoke decode framework

- **Location:** `decode.rs:1-136`; `decode/buffered.rs` (whole file);
  `diagnostic.rs:149-172`; `value_domain.rs:169-200`; `wire.rs:1-33`;
  `provenance.rs:500-503, 577`
- **Confidence:** High · **Python-driven:** yes (the module doc says it
  matches Python's deserializer side effects)

**Issue:** Deserializing an `Identifier` advances the global counter, and
deserializing a `Canonical<T>` interns globally.

- **Order of effects.** To reproduce Python's order (check one level,
  restore it, recurse), the crate built a private `Decode` trait,
  `DeferredPayload`, its own `BufferedValue` self-describing value type, and
  a `MapOnly` adapter. A payload rejected at depth N leaves the ids of levels
  0..N-1 restored, and this is documented as the contract.
- **Three buffering strategies.** `Expression` buffers via
  `serde_json::Value` instead, and `Provenance` uses an adjacently tagged
  derive. Each re-implements the `__type__`/`__data__` envelope.
- **Misleading nested errors.** `BufferedValue` stores serde_json's
  arbitrary-precision number token as an ordinary map, so floats and bigints
  inside a deferred level give misleading errors ("invalid type: map").
- **Impure `from_str`.** `serde_json::from_str::<Note>` is impure, so a
  failed untagged or `flatten` attempt or a retry leaves global changes
  behind.

**Suggested fix:**
- Derive plain `Deserialize` on side-effect-free payload structs.
- Add an explicit `payload.restore(&mut Session) -> Result<T, _>`, or use
  `DeserializeSeed`. A rejected payload then restores nothing.
- Delete `Decode`, `DeferredPayload` and `BufferedValue`.
- Centralize the envelope in one `wire` module.

#### F-012: Workspace-wide `arbitrary_precision` leaks to dependents; serde impls are JSON-only

- **Location:** `Cargo.toml:17-23`; `lib.rs:13-21`; `identifier.rs:284-313`
  (`Number::deserialize` → `deserialize_any`); `decode.rs:66-99`
  (`MapOnly`); `wire.rs:142-190, 299-321, 465-470`
- **Confidence:** High · **Python-driven:** yes (unbounded ints as JSON
  integers; Python's "payload must be a dict" rule)

**Issue:**
- **The feature leaks.** Cargo feature unification turns
  `arbitrary_precision` on for every crate in any build that includes
  fhy-core. `Number` then compares by text, and floats inside internally
  tagged, untagged or `flatten` types arrive as maps and fail to deserialize
  in *downstream* code.
- **JSON-only in practice.** The serde impls look format-generic but only
  work with JSON:
  - `Identifier` requires `deserialize_any`;
  - `MapOnly` breaks round trips in struct-as-tuple formats such as bincode
    and postcard;
  - `Expression` buffers every format through `serde_json::Value`;
  - int versus float is decided by re-stringifying the number.

**Suggested fix:**
- Make big-integer JSON opt-in through a feature that only `fhy-core-py`
  enables.
- Decode ids as `u64` and `PayloadNode` with a typed visitor.
- Either document "these impls encode the Python JSON wire format" at crate
  level, or move the wire format behind `to_json`/`from_json`.

#### F-013: CPython text emulation lives in the core crate and drives `Display`

- **Location:** `python_text.rs` (884 lines);
  `symbolic/expression/literal.rs:190-205` (`canonical_key`), `:279-290`
  (`Display`); `pass_infrastructure/registry.rs:77-85`
  (`is_python_whitespace`); `provenance.rs:619-628` (`PurePosixPath`);
  CONTRIBUTING.md:365-371
- **Confidence:** Medium (depends on an open question) · **Python-driven:** yes

**Issue:**
- **Python text in Rust `Display`.** Byte-for-byte `repr(float)`, `Decimal`
  normalization and `True`/`False` feed Rust `Display` (`true` prints as
  `True`, `1e16` as `1e+16`).
- **An unused public key.** `canonical_key() -> String` is public, allocates,
  and has no non-test caller: `Eq` and `Hash` use the private
  `CanonicalForm`.
- **Mixed conventions.** Python-style error text ("Pass name cannot be
  empty.") sits next to Rust-style lowercase messages elsewhere.
- **No cross-check.** No golden corpus checks this text against Python
  (F-036).

**Suggested fix:**
- Core `Display` follows Rust conventions.
- Move `python_text` and the Python message rendering into the binding
  crate, or behind the same opt-in feature as F-012.
- Make `canonical_key` `pub(crate)` or have it return a typed key.
- Behavior change: printed literals change.

#### F-014: Errors are Python-exception-shaped: stringly, unmatchable, cause duplicated or dropped

- **Location:** `pass_infrastructure/error.rs:10-27, 80-121, 164-251`;
  `pass.rs:363-380`; `tree.rs:347-399`; `manager.rs:378-417, 525-554`;
  `symbolic/expression/error.rs:33-73, 145-200`; `pattern/core.rs:543-639`;
  `pattern/rewrite.rs:335-353`
- **Confidence:** High · **Python-driven:** yes (exception classes map to
  `is_*` predicates, and messages are pre-rendered in Python's wording)

**Issue:**
- **Registration errors are stringly.** `PassRegistrationError { message:
  String }` covers five distinct cases, so tests match on text.
- **Kind is unmatchable.** `PassError` exposes its kind only through boolean
  predicates.
- **Cause printed twice.** `PassError`'s `Display` embeds the cause *and*
  `source()` returns it, so chain printers such as anyhow show it twice.
- **Cause printed nowhere.** `RewriteTreeError`'s `Display` drops the cause,
  so `RewritePass` diagnostics say only "rewriting a node failed".
- **Outer diagnostics lost.** Passing through a nested `PassError` (detected
  by downcast) discards the outer pass's diagnostics.
- **Records lost.** A failed pipeline loses all `PipelineRecord`s, and
  non-convergence carries no structure.
- **Broad error types.** `ExpressionBuildError` makes callers match five
  variants where one or two are possible.
- **Erased callback errors.** `CallbackError` erases user error types.
- **Enums can't grow.** No public error or record enum is
  `#[non_exhaustive]`.

**Suggested fix:**
- Use `#[non_exhaustive]` enums with structured fields, including
  `PassError::kind()`.
- Render `Display` from fields without repeating the source.
- Use a `Nested` kind in place of downcast pass-through.
- Attach partial records to the pipeline error.
- Use per-operation error types in the expression builders.
- The binding maps variants to Python classes and messages.

#### F-015: No `Display` for expressions or reports; multi-line error `Display`

- **Location:** `symbolic/expression/pprint.rs:333-343`; `node.rs:205`;
  `error.rs:241-246`; `diagnostic.rs:266-268, 329-403`; `manager.rs:342`
- **Confidence:** High · **Python-driven:** yes (`pformat_expression ->
  str`, `report.format()`, `ValidationFailedError(report.format())`)

**Issue:**
- **Printer only returns `String`.** Expressions can be rendered only
  through `format_expression -> String`.
- **Intermediate allocations.** The screen error's `Display` builds two
  intermediate Strings, and every literal and identifier allocates.
- **Multi-line error text.** `ValidationFailedError`'s `Display` is the whole
  multi-line report, which then gets inlined into other diagnostics' `detail`
  strings.

**Suggested fix:**
- Add `expr.display(opts) -> impl Display` (like `Path::display`) and
  `impl Display for Expression`.
- Add `impl Display for ValidationReport`.
- Make the error's `Display` a short line such as "validation failed with N
  errors".

#### F-016: Analysis cache applies mutable-IR invalidation to immutable handles and pins nodes

- **Location:** `pass_infrastructure/analysis.rs:114-131, 189-233`;
  `manager.rs:272-273, 311-328, 370-371`
- **Confidence:** Medium · **Python-driven:** yes (`AnalysisManager.transfer`
  and `invalidate`, designed for mutable, `id()`-keyed IR)

**Issue:** `NodeHandle` nodes are immutable, so a result cached for a node's
identity can never go stale. Yet:
- `transfer` drops the destination's bucket, including results just computed
  on the output;
- same-handle "changed" outputs lose valid results;
- every analysed node, including rewritten-away subtrees, is pinned by a
  strong clone until the run ends (Python uses weakrefs);
- the verification-report cache can almost never hit.

**Suggested fix:**
- Make preservation merge-only, and never invalidate a node's own results.
- Evict buckets of nodes the current IR no longer reaches, or key the cache
  by a never-reused id.
- Replace the report cache with a `HashSet` of verified identities.

#### F-017: Pipelines are `!Send` while IR handles must be `Send + Sync`

- **Location:** `pass_infrastructure/manager.rs:177, 258`;
  `validation.rs:41`; `analysis.rs:72`
- **Confidence:** High · **Python-driven:** no

**Issue:** `Box<dyn CompilerPass<I> + 'p>` has no auto-trait bounds, so
`PassManager`, `FixpointPassGroup` and `ValidationManager` are `!Send`: they
can't move to another thread or be stored in a `#[pyclass]` without marking
it `unsendable`. Meanwhile `NodeHandle` requires `Send + Sync`, which rules
out `Rc`-based IR for no benefit.

**Suggested fix:** Pick one threading story. Most likely store `+ Send`
passes and keep the IR `Send + Sync`.

#### F-018: The `testing` feature ports a monkeypatch into `Identifier::new`

- **Location:** `identifier.rs:111-143`; `testing.rs` (517 lines);
  `rust/fhy-core/Cargo.toml:15-58` (self dev-dependency, 4
  `required-features`); CI's packaged-crate step
- **Confidence:** Medium · **Python-driven:** yes
  (`deterministic_identifiers_by_name_hint`)

**Issue:**
- **Breaks uniqueness.** Inside a scope, two `Identifier::new("tmp")` calls
  return equal identifiers, which breaks the type's core invariant.
- **Can reach production.** The switch is a Cargo feature and features
  unify, so production builds can pick it up.
- **Production path never tested.** The self dev-dependency means no
  integration or doc test ever exercises the production `next_id` path.
- **Costly for little use.** Only its own four test binaries use it, and
  supporting it costs a feature, a `cfg` fork, a `new_unscoped` path,
  `required-features` and a CI step.

**Suggested fix:**
- Compare graphs modulo identifier renaming (`alpha.rs` already exists;
  `tag_type_equivalence` already uses a slot normalizer).
- Or inject an id source through the session (F-004).
- If kept, rename the feature to `test-util`.

#### F-019: Pattern captures are string-keyed; constructors use `Option` for "any"

- **Location:** `pattern/core.rs:160-306, 397-505, 543-556`
- **Confidence:** High · **Python-driven:** yes (`CapturePattern(name:
  str)`, `value | None = None`, `__post_init__` validation)

**Issue:**
- **Stringly lookups.** `bindings.get("x") -> Option<&Expression>` is a
  linear scan, so every rewrite handles an impossible "unbound" case, and a
  typo compiles.
- **`None` means "any".** `Pattern::literal(None)` and `call(None,
  Some(vec![]))` pack modes into `Option`s.
- **Fallible for harmless input.** `capture`, `alternatives(vec![])` and
  `piecewise` return `Result` only for "never matches" input.
- **Python seams left public.** `match_under` and `try_bind` are public.
- **Clones per attempt.** Bindings are cloned per alternative and per match
  attempt.
- **Two equalities.** Literal patterns and capture unification use different
  literal equality, so `nan - nan → 0` fires.

**Suggested fix:**
- Typed `Capture` handles, with `Index<&Capture>` on bindings.
- Named constructors (`any_literal()`, `literal(v)`, `any_call()`).
- Make construction infallible.
- Keep bindings on a `&mut` trail.
- Make the internal seams `pub(crate)`.

#### F-020: `Canonical<T>` and `T` have two different equalities; bare `T: Deserialize` escapes interning

- **Location:** `interned.rs:348-360, 387`; `value_domain.rs:131-142,
  180-184, 222-235`; `described_tag.rs:130-136`
- **Confidence:** High · **Python-driven:** yes (`is` versus `==`; the
  dataclass `__eq__` over `(name, parent)`)

**Issue:**
- **Two equalities.** Handle `==` is `Arc::ptr_eq`, while value `==`
  compares values, and `ValueDomain`'s compare walks the whole parent chain.
  `is_subdomain_of` is O(depth²).
- **Detached values.** `serde_json::from_str::<ValueDomain>` advances the
  counter and registers parents, but returns a detached, non-canonical value.
  Only the docs prevent downstream `domain: ValueDomain` fields.

**Suggested fix:**
- One equality, by key (the name), for both the handle and the value.
- Bound `Canonical<T>: Deserialize` on the private decode trait, and drop
  the public bare impls.

#### F-021: Constructors register globally and return `InternOutcome`; `Option`-mode parameters

- **Location:** `value_domain.rs:85-91`; `described_tag.rs:47-55`;
  `diagnostic.rs:215-233`; `provenance.rs:152-180, 380-394`;
  `identifier.rs:95-102` (public, panicking `restore(u64, String)`)
- **Confidence:** High · **Python-driven:** yes (`__post_init__`
  self-registration; dataclasses with optional kwargs)

**Issue:**
- **`new` registers globally.** `new` doesn't return `Self`. It registers
  globally, and 93 call sites append `.into_canonical()`.
- **Conflicts accepted silently.** A conflicting parent is accepted
  silently, while decode rejects the same conflict.
- **Too many positional arguments.** `Diagnostic::new` takes four positional
  arguments including `Option<String>`. `Span::try_new` takes four positional
  `Option`s, so swapping start and end compiles.
- **Option-mode argument.** `fuse(…, Option<&str>)` selects a mode.

**Suggested fix:**
- `register_root`/`register_child -> Result<Canonical<Self>, Conflict>`.
- `Diagnostic::error(note, source).with_detail(..)`.
- `Span::from_offsets(Range)`/`from_positions`.
- `fuse` plus `fuse_labelled`.
- `try_restore -> Result`, or make `restore` `pub(crate)`.

#### F-022: Pass lifecycle hooks: split `should_run`/`noop_output`, validators forced into pass shape

- **Location:** `pass_infrastructure/pass.rs:21-31, 104-125, 403-416`;
  `validation.rs:41, 81-105`; `analysis.rs:86-92`; `preserved.rs:34, 137,
  154`
- **Confidence:** High · **Python-driven:** yes (template methods on an ABC;
  validators as pass subclasses)

**Issue:**
- **A runtime error the compiler could catch.** Overriding `should_run`
  without `noop_output` fails at runtime.
- **Boilerplate the manager ignores.** Validators must implement
  `did_change` and `noop_output`, and the manager discards what they return.
- **Cache not shared.** The verifier doesn't use the pipeline's analysis
  cache.
- **Limited analyses.** Analyses must be `Default` and can't depend on each
  other.
- **Silent typos.** `PreservedAnalyses::preserve::<NotAnAnalysis>()`
  compiles and silently does nothing.

**Suggested fix:**
- One `skip(&mut self, ir, cx) -> Result<Option<O>, _>` hook.
- A dedicated `Validator<I>` trait that shares the cache.
- Give `Analysis::run` access to other analyses.
- Add a marker bound on `preserve`.

#### F-023: Screen API copies optional keyword mappings and re-walks shared subtrees

- **Location:** `symbolic/expression/screen.rs:16-20, 45-187, 287-344`
- **Confidence:** Medium · **Python-driven:** yes (`environment: Mapping |
  None = None, *, symbol_types: ...`)

**Issue:**
- **Concrete maps required.** Two public functions take four positional
  parameters, including concrete `&HashMap`s that callers fill with empty
  maps.
- **Exponential on DAGs.** The walk follows the tree, not the DAG, so it is
  exponential on shared subtrees (like F-002).
- **Quadratic on nested piecewise.** Piecewise branches are re-walked per
  enclosing check, which is O(d²).

**Suggested fix:**
- A public builder: `Screen::new(&sorts).with_environment(..)`.
- Lookups as traits or closures.
- A memoized, bottom-up three-valued sort.

#### F-024: Built-in functions are named by string; `IntoOperand` duplicates `From`/`Into`

- **Location:** `symbolic/expression/node.rs:302-305, 481-517`;
  `builtins.rs:69-159, 452-507`; `build.rs:33-174`
- **Confidence:** Medium · **Python-driven:** yes (registry-key strings;
  `_get_expression_from_other(other: Any)`)

**Issue:**
- **Built-ins by string.** `create_call("sigmod", ..)` compiles, and callers
  can't match built-ins exhaustively.
- **A sealed coercion trait.** The sealed `IntoOperand` hides the conversion
  from generic callers.
- **Missing conversions.** There is no `From<i64>`, `From<u64>`,
  `From<usize>` or `From<f64>` for `Expression`, so shape arithmetic needs
  `Expression::from(LiteralValue::from(n))`.

**Suggested fix:**
- A `Copy` `BuiltinFunction` enum, with `Callee::{Builtin, Named}`.
- `From<primitive> for Expression`, with builders taking `impl
  Into<Expression>`. `bool` stays rejected simply by having no impl.

#### F-025: Hand-maintained `ALL_*` arrays emulate serde derive; no `FromStr`

- **Location:** `symbolic/expression/operation.rs:14-37, 105-109, 243-247`;
  `sort.rs:12-17, 59-63`; `symbol_type.rs:11, 50-54`; `wire_name.rs:18-66`
- **Confidence:** High · **Python-driven:** yes (`StrEnum`)

**Issue:** Adding a variant updates the exhaustive `as_str` but not the
lookup array, so the variant serializes but silently fails to deserialize.
The tests iterate their own copies of the arrays, so they can't catch it.

**Suggested fix:** `#[derive(Serialize, Deserialize)] #[serde(rename_all =
"snake_case")]` plus `FromStr` (or `strum`).

#### F-026: `LiteralValue` emulates Python's `str|int|float|bool` union

- **Location:** `symbolic/expression/literal.rs:21-110, 164-206, 279-290`
- **Confidence:** Medium · **Python-driven:** yes

**Issue:** Five stored forms fall into four equality buckets.
`LiteralKind::Int(5) != LiteralKind::IntegerText("05")` even though the
values are equal.

**Suggested fix:** Collapse to `Bool | Int(BigInt) | Float(f64) |
Decimal(Normalized)`, normalizing on parse. This is only acceptable if no
pass needs the original spelling; see Open questions.

---

### Low (architecture and API)

#### F-027: Macro emulates mixin inheritance for described tags

- **Location:** `described_tag.rs:24-188`; `op_attribute.rs:27-88`;
  `diagnostic.rs:31-92`

**Issue:** A 160-line `macro_rules!` generates `OpAttribute` and `NoteKind`.
The generated items are opaque to rustdoc, IDEs and grep.

**Suggested fix:** A generic `DescribedTag<K: TagKind>` with a sealed kind
trait and type aliases.

#### F-028: Module tree copies Python package layout; crate split is blocked

- **Location:** CONTRIBUTING.md:354-363; `lib.rs:23-41`;
  `symbolic/expression/pattern/mod.rs:11` (`mod core` shadows the `core`
  crate and contradicts CONTRIBUTING's own `core.py` rule)

**Issue:**
- **Long paths.** Users write
  `fhy_core::symbolic::expression::pattern::Pattern`.
- **No subset can be pulled out.** The F-007 and F-008 cycles mean no part
  of the single crate can be extracted.

**Suggested fix:**
- Relax the rule to "Rust layering, plus a Python→Rust traceability table".
- Fold `pattern/core.rs` into `mod.rs` now (not breaking).
- After F-004, F-007 and F-008, consider foundation, `fhy-symbolic` and
  `fhy-passes` crates. Don't split while the globals remain.

#### F-029: Python module-function API shape (`get_*`, free functions, `list_*`)

- **Location:** `op_attribute.rs:66-86`; `value_domain.rs:280-289`;
  `diagnostic.rs:75-91`; `pattern/core.rs:648, 661`; `pattern/rewrite.rs:411,
  452`; `builtins.rs:452-503`

**Issue:**
- **`get_` prefixes.** `get_commutative()` and similar go against C-GETTER.
- **Free functions taking the receiver first.** `match_pattern(p, e)`,
  `does_pattern_match` and `apply_rewrite_rule(rule, expr)` sit next to
  `apply_rewrite_rules(expr, rules)`, with inconsistent argument order.
- **Python naming.** `MatchBindings::empty()`/`has()`, and `list_*`
  catalogue functions.

**Suggested fix:**
- `OpAttribute::commutative()`, `pattern.is_match(e)`, `rule.apply(e)`.
- Iterator-returning catalogue functions.

#### F-030: Binding crate structure won't scale

- **Location:** `rust/fhy-core-py/src/lib.rs`; `identifier.rs:15-46`;
  `src/fhy_core/_rs.pyi`

**Issue:**
- **Flat registration.** Everything is added flat to `_rs`.
- **Per-error conversions.** The orphan rule forces a `convert_*` function
  per error type.
- **Hand-maintained stubs.** `_rs.pyi` is written by hand.
- **Unowned identity cache.** The canonical-object identity cache that
  CONTRIBUTING requires has no home yet.
- **Borrowed message text.** Out-of-range ids depend on PyO3's extraction
  message, which the Python side hard-codes.

**Suggested fix:**
- Declarative `#[pymodule] mod`, with a submodule per core module.
- One local `IntoPyErr` trait.
- Check stubs in CI.
- Combine with F-005(b).

### Tests, docs and performance

#### F-031: Test isolation depends on hand-maintained rules around global state

- **Severity:** Medium · **Confidence:** High
- **Location:** `src/test_support.rs:13-136`; 44 `REGISTRY_GUARD.hold()`
  calls in `value_domain.rs`, 22 in `op_attribute.rs`, 6 in `diagnostic.rs`;
  `tests/pass_infrastructure_core_stories.rs:6-10` plus about 16 copy-pasted
  identity pass types; `tests/pass_infrastructure_run_count_stories.rs`;
  `tests/builtins_scope_stories.rs:1-7`

**Issue:** The compiler can't check any of these rules:
- per-module registry guards that don't serialize across modules;
- id-counter locks and far-ahead constants;
- `#[ignore]` bodies re-executed in child processes;
- globally unique pass type names;
- single-test binaries.

All of it follows from F-004 and F-006. Run-count tests mix absolute and
delta assertions and key on short type names.

**Suggested fix:**
- Test `clear()` only on local registries, and delete the
  global-`clear()` unit tests and `RegistryGuard`.
- Generate identity passes with a macro.
- Use deltas everywhere.
- Long term, fix F-004.

#### F-032: Deterministic-id golden replay passes only because of corpus case order

- **Severity:** Medium · **Confidence:** High
- **Location:** `tests/deterministic_identifiers_equivalence.rs:11-23,
  56-93`; `tests/golden/deterministic_identifier_cases.json` (cases 0 and 1)

**Issue:**
- **Wrong doc comment.** The module doc says nothing in the binary touches
  shipped constants. But the first `restore` calls
  `initialize_shipped_statics()`, which allocates about 15 ids.
- **Order hides it.** Today case 0 restores at offset 40, so those ids fit
  below 40. If case 1 (restore at offset 1, then `new c` expecting relative
  id 3) ran first, it would fail.

Adding a shipped constant or reordering the generator's cases breaks the test
with confusing mismatches.

**Suggested fix:** Force the shipped statics once before the first script,
and fix the doc comment. F-007 removes the cause.

#### F-033: Pinned-id headroom scheme can collide under parallel tests

- **Severity:** Medium · **Confidence:** Medium
- **Location:** `src/test_support.rs:13-26`; `src/identifier.rs:863-869`;
  e.g. `src/testing.rs:340-353`, `src/op_attribute.rs:318-324`

**Issue:**
- **Restores move the counter.** `reserve_pinned_id` returns `anchor +
  1_000_000`, on the assumption that nothing allocates a million ids during
  one test. But every test that restores its pinned id jumps the counter
  about a million ahead, and one test jumps it by exactly 1,000,000 without
  holding the counter lock.
- **Consequence.** Fresh ids from other threads can then land in another
  test's pinned window: a latent flake, for example
  `assert_ne!(deserialized, constructed)`.

**Suggested fix:**
- Use `Identifier::new(anchor).id()`, which is already unique, and restore
  already-issued ids, which doesn't move the counter.
- Observe the counter through the local-`AtomicU64` seams that already
  exist.

#### F-034: 35 integration-test binaries; `#[path] pub mod` helper sharing hides dead code

- **Severity:** Medium · **Confidence:** High
- **Location:** `rust/fhy-core/tests/*.rs`; `tests/common/*`

**Issue:**
- **Repeated builds and links.** Every file is its own crate that links
  fhy-core, proptest, rstest and serde_json again, and the shared helpers
  (e.g. the 444-line `common/expression.rs`) are compiled up to 13 times.
- **Hidden dead helpers.** Helpers are `pub` only to silence dead-code
  warnings, so truly dead helpers are never reported, and many "shared"
  helpers have one user.
- **Isolation recorded only in comments.** Four binaries rely on running
  alone in their process, and only file comments say so.

**Suggested fix:**
- Merge the isolation-safe files into `tests/it/main.rs` with one `mod
  support`.
- Keep the four isolated binaries as explicit `[[test]]` entries with a
  comment saying why, or standardize on nextest.

#### F-035: Tests pin Python wording, serde's messages, private type names and `type_name`

- **Severity:** Medium · **Confidence:** High
- **Location:** `tests/provenance_stories.rs:1180-1437` (28 cases, e.g.
  `"expected struct PositionPayload"`); `tests/diagnostic_stories.rs:337-361`;
  `tests/pass_infrastructure_core_stories.rs:494-499, 911-917, 1127-1370`;
  `tests/pass_infrastructure_manager_stories.rs:568-627`;
  `tests/expression_wire_stories.rs:220-221, 355-377`;
  `tests/expression_screen_stories.rs` (full sentence repeated in 5 tests);
  about 30 Python spellings (`"True"`, `"1e+16"`, `"(-1 ** 2)"`,
  `canonical_key` strings) across pprint, literal, screen and builtins tests;
  Debug-text assertions in `identifier.rs`, `interned.rs`, `value_domain.rs`
  and `op_attribute.rs`

**Issue:**
- **Pinned text that isn't the crate's.** Serde and serde_json wording,
  serde_json's recursion limit, private DTO names and unstable `type_name`
  output are all pinned.
- **Python wording pinned without a contract.** Python wording is pinned
  where the bindings don't expose the error, so no contract needs the text.

A serde upgrade, a private rename, a toolchain upgrade or any F-013/F-014
change breaks dozens of expectations.

**Suggested fix:**
- Assert error variants and fields, or `classify()` plus the crate's own
  context.
- Keep one `Display` table per error type.
- Build expected literal spellings from one table.
- Check `canonical_key` by its properties.

#### F-036: Symbolic area has no Python-recorded golden corpus; goldens never checked for staleness

- **Severity:** Medium · **Confidence:** High
- **Location:** `rust/fhy-core/tests/golden/` (only the identifier,
  interned and tag-type corpora); `tests/tag_type_equivalence.rs:41-45,
  299-352`; `.github/workflows/python-package.yml`

**Issue:**
- **No oracle for the symbolic area.** The expression wire JSON, pprint
  text, literal buckets and float repr are specified only by hand-transcribed
  expectations. The one surface that must interoperate byte for byte has no
  Python oracle.
- **Unchecked coupling.** The existing corpora duplicate constants between
  languages by hand, and classify conflicts by message substring.
- **Staleness never checked.** CI never regenerates the corpora to detect
  staleness.

**Suggested fix:**
- Add `generate_expression_cases.py` plus `expression_equivalence.rs`
  covering whatever F-013 keeps as a contract.
- Have generators emit shared constants into the document.
- Add a CI regenerate-and-`git diff --exit-code` step.

#### F-037: Test assertions that hang, test nothing, or re-implement the SUT

- **Confidence:** High
- **Location:**
  - Debug-printing a 2^64-occurrence DAG (see F-003):
    `tests/expression_tree_stories.rs:383`,
    `tests/pattern_rewrite_stories.rs:713`.
  - HashMap-order cases that can't control order:
    `tests/expression_node_stories.rs:1109-1137`.
  - Hash *inequality* asserted as a contract:
    `tests/expression_node_stories.rs:992, 1022`,
    `tests/expression_literal_stories.rs:420`.
  - `render_report` re-implementing `ValidationReport::format`:
    `tests/provenance_diagnostic_properties.rs:373-403`.
  - A symmetry property whose equal branch is almost never hit:
    `tests/expression_properties.rs:196-213`.
  - Bare `is_some()`/`is_err()` where the value is available:
    `tests/pattern_stories.rs` (12 sites), `tests/payload_form_stories.rs:217`.
  - `is_ok()` that may trip the workspace's `assertions_on_result_states`
    lint: `tests/expression_screen_stories.rs:1150`.
  - Duplicated helpers and a 16-name list copied four times:
    `tests/builtins_stories.rs`.
  - rustfmt-skipped misindentation inside `proptest!`:
    `tests/pprint_properties.rs`.

**Suggested fix:** Fix each case as listed in the subagent reports; each is
small.

#### F-038: Isolated-child-process harness is fragile and passes vacuously under `--ignored`

- **Location:** `src/test_support.rs:62-95`; `src/shipped.rs:162-206`
  (hard-codes rstest's `case_N_*` names); `src/diagnostic.rs:550-576`;
  `src/value_domain.rs:1176-1236`

**Issue:**
- **Depends on libtest text.** Success is detected by
  `stdout.contains("1 passed")`.
- **Vacuous passes.** Under `--include-ignored`, the `_in_isolation` bodies
  return early and report a pass.

**Suggested fix:**
- Make the child panic if its environment variable is missing, and check
  the exit status.
- Or use `harness = false` integration tests for "first use in a fresh
  process".

#### F-039: Performance nits on hot paths

- **Location:**
  - A write lock on every `intern`, including decode hits:
    `interned.rs:177-191`.
  - A SipHash insert plus a node clone per rule firing, to serve rare error
    blame: `pattern/rewrite.rs:33-114`.
  - `name()`/`description()` allocate and diagnostics clone the pass name:
    `pass.rs:82-90`, `context.rs:49`.
  - Full diagnostic copies: `pass.rs:378`, `manager.rs:337, 347`.
  - Diagnostics stored twice: `validation.rs:96-102`.
  - SipHash on address-keyed maps: `analysis.rs`.
  - A `String` allocated per enum decode: `wire_name.rs:24`.

**Suggested fix:**
- Read-lock fast path with the `entry` API.
- Use the identity hasher.
- `Cow<'static, str>` for pass names.
- Move instead of copy.

#### F-040: Crate docs and README drifted

- **Location:** `rust/fhy-core/README.md:7-21`; `lib.rs:1-21`;
  `rust/fhy-core/Cargo.toml`; rustdoc in `value_domain.rs:60, 151-154` and
  `testing.rs:16-18`

**Issue:**
- **Incomplete module list.** The README omits `diagnostic`, `provenance`,
  `symbolic` and `pass_infrastructure`.
- **No overview.** `lib.rs` has no module overview.
- **Publish status unclear.** The README says "not published", but the
  manifest lacks `publish = false`, while CI runs `cargo package` machinery.
- **Python narrative in rustdoc.** User-facing rustdoc contains Python
  narrative.

**Suggested fix:**
- Update the README and add a crate overview.
- Decide on publishing.

---

## Decisions (2026-09-23)

1. **Payload ids (open question 1): exact ids, hardened.** Deserialized
   identifiers keep their payload id, so `copy`, `deepcopy`, `pickle` and
   multiprocessing round trips are unchanged. Built-in tags get a fixed
   reserved id block, identically in Python (fixes F-007 and F-032; deletes
   `shipped.rs`). Payload ids above a cap are rejected, and
   `Identifier::try_new` is added (F-001).
2. **Global state (open question 2): hybrid.** The id counter and the intern
   registries stay global and append-only. `InternRegistry::clear` becomes
   test-only, and tests use local registries. The pass registry and run
   counters become owned values or are deleted (F-006; see open question 8).
   There is no full `Session` for now: F-004 is narrowed to this, and F-005
   still needs the binding-crate split.
3. **Python parity (open questions 4 and 5).** Python-identical behavior and
   text are required only for concepts defined in both languages at once:
   today `identifier` and `interned`. Code that exists only to match Python
   is documented as such. Where Rust replaces the Python implementation, Rust
   defines the behavior. This frees F-013, F-014, F-015, F-018, F-026 and
   F-035 to follow Rust conventions. The CONTRIBUTING rules "the binding
   raises the same message" and "module paths follow the Python package" are
   to be narrowed to dual-defined concepts.
4. **Wire format (open question 6): plain serde, no envelope, in the core.**
   The core crate uses plain serde derives with Rust-defined shapes (n-ary
   logic for F-003, derived enum encodings for F-025, no workspace-wide
   `arbitrary_precision` for F-012, no ordered-side-effect decode framework
   for F-011). The `__type__`/`__data__` envelope that Python's serialization
   framework needs to embed Rust objects in not-yet-ported Python containers
   is added in the binding layer, not in the core. No long-lived persisted
   data exists, so no reader for the old format is needed.

5. **Downstream use (open question 3):** nothing links `fhy-core` until the
   port is complete. F-005 is deferred.
6. **Literal spelling (open question 7):** no pass needs the original
   spelling. F-026 may collapse literals to normalized forms.
7. **Pass registry (open question 8): keep it, as an owned `PassRegistry`
   value.** Python's own library registers passes (`verification.py`,
   `pattern/rewrite.py`), and a driver needs to build pipelines by name. The
   Python binding holds one registry in module state. Default pass names no
   longer come from the registry.
8. **Rewrite semantics (open question 10):** a rewrite that returns its own
   input is no change (F-009).
9. **Publishing (open question 11):** `fhy-core` will be published to
   crates.io. Breaking API changes are free until the first release, so the
   batched API cleanup should land before it. F-040 should drop the "not
   published" README line.

10. **Threading (open question 9): passes must be `Send`.** Stored passes
    become `Box<dyn CompilerPass<I, O> + Send + 'p>`, so pipelines are
    `Send`. IR stays `Send + Sync`, and the Python binding wraps pipelines in
    a `Mutex` (F-017).

11. **Operators (F-010):** keep `/` as true division; replace the `%`
    operator with a named `floor_mod()` builder.
12. **Deterministic-identifier scope (F-018):** delete it, together with its
    golden corpus. Tests compare up to identifier renaming or print without
    ids.
13. **Crate layout (F-028):** one `fhy-core` crate with clean internal layers
    (foundation, tree, symbolic, passes) and no dependency cycles. Module
    paths are flattened (e.g. `fhy_core::expr::Pattern`). A later split
    re-exports from `fhy-core`, so it breaks no one.

## Triage (2026-09-24)

**Status:** implemented at `6fdbe68` on `dev-rust`, per
`docs/design/rust-workspace.md`.

- **Fix:** every finding except those below. This includes F-004 narrowed to
  decision 2, and the tag-type golden corpus removed under decision 3.
- **Partly deferred:**
  - F-016: fix merge-only preservation and the verification-report set. Node
    pinning for the length of a run stays, because it prevents address
    reuse.
  - F-022: fix the `skip` hook, the `Validator` trait and the `preserve`
    marker bound. Analyses that depend on other analyses are a new feature,
    so they are deferred.
- **Defer:** F-005 (binding split, per decision 5).
- **Reject:** none.

## Open questions

These decide the direction of most Medium findings. Please resolve them
before the spec.

1. **Cross-process payloads.** Is decoding payloads produced by another
   process (persisted IR, Python↔Rust interop) supported? May decoded ids be
   remapped rather than kept verbatim? This decides F-001(a), F-007 and
   F-011.
2. **Session/context.** Is an explicit session/context acceptable as the
   replacement for global registries (F-004)? Should it be done before more
   modules are ported?
3. **Downstream Rust.** Will a downstream Rust crate link `fhy-core` soon?
   This decides how urgent F-005 is.
4. **Python text parity.** Must Rust produce byte-identical Python text:
   float `repr`, `True`/`False`, `Decimal` `E` notation, report and error
   messages? Or only equivalent JSON on the wire? This decides F-012, F-013,
   F-015 and F-035.
5. **Binding messages.** Does CONTRIBUTING's "binding raises the same
   message" mean exact text or the same exception class?
6. **Wire format freedom.** Is the wire format frozen (binary logical nodes,
   the `__type__`/`__data__` envelope, big ints as JSON integers), or may it
   gain an n-ary logical node and a builtin enum (F-003, F-024)?
7. **Literal spellings.** Does any pass need verbatim literal spellings
   (`"05"`, `"1.50"`)? This decides F-026.
8. **Pass registry.** Are the pass registry and run counters needed at all?
   If not, F-006 is a deletion.
9. **Threading.** Should pipelines be `Send` (e.g. as a `#[pyclass]`)? This
   decides F-017.
10. **Rewrite "changed" semantics.** May a rewrite that returns its input
    count as "no change" (F-009)?
11. **Publishing.** Will `fhy-core` be published to crates.io? This decides
    F-018 and F-040.

## Suggested sequencing (for the spec phase)

1. **Correctness hazards, independent of the big decisions:** F-002 (cached
   hash), F-003 (manual `Debug`; the n-ary node if the wire may change),
   F-009, F-010, F-032, F-033, and the hanging tests in F-037.
2. **The architectural decision:** F-004 (session). F-001, F-005, F-006,
   F-007, F-011, F-018 and F-031 follow from it.
3. **Decouple layers:** F-008 (generic tree walkers), then F-028 (layout and
   crate split).
4. **Python-shaped API cleanup, batched into one breaking release:** F-012 to
   F-015 and F-019 to F-027, F-029, guided by the answers to Open questions 4
   to 7.
5. **Tests:** F-034 (consolidate binaries), F-035 and F-036, after the error
   and text decisions.
