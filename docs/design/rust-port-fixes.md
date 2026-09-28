# Rust-port audit fixes spec

- **Status:** specified 2026-09-27; not started.
- **Base:** `dev-rust` at `d976522`. Every track branches from the commit
  that adds this file.
- **Audit:** `docs/audit/rust-port-2026-09.md`. This spec implements its 47
  findings (F2-001 to F2-047), its triage, and the maintainer's
  "Resolutions" of 2026-09-27, together with the five items those
  resolutions settle that are not numbered findings.
- **Related:** `docs/design/rust-workspace.md`, the first audit's spec, whose
  shape this one mirrors; `docs/design/python-switch.md`, whose slice
  decisions several fixes revise; CONTRIBUTING "Porting to Rust".

This document is the contract for the fix work. Part I is the cross-cutting
design: shared designs, the calls this spec makes where a resolution left
room, the revised decisions, sequencing, tracks, landing order and gates.
Part II holds one requirement per finding. Where Part I and Part II
disagree, Part I governs.

**Path conventions** follow the audit's. Source paths are relative to
`rust/fhy-core/src/`, except:
- paths starting with `fhy-core-py/src/`, which are under `rust/`;
- `tests/it/…` and `tests/<name>.rs`, which are under `rust/fhy-core/`;
- `tests/test_*.py`, `tests/<dir>/…py` and `src/fhy_core/…`, which belong
  to the Python package at the repository root;
- repository-root files (`CONTRIBUTING.md`, `README.md`, `Cargo.toml`,
  `deny.toml`, `noxfile.py`, `pyproject.toml`, `.github/…`).

## Progress checklist

Each track ticks only its own group, in its own worktree, in the commit
that completes the item (or the next commit, with the hash). The groups are
separated by headings and blank lines, so ticks from different tracks never
touch adjacent lines and merge without conflict. The items are listed in
each track's working order. A `[rebase]` line marks where the track rebases
onto `dev-rust` before continuing.

### Track A: `api` (the breaking API batch; lands 1st)

- [x] A0: worktree `port/fix2-api` created; the baseline gates recorded (the worktree is `fix-a-api`, branch `fix/a-api`; see Track A notes): `fd19340`
- [x] R2-023 (F2-023): interrupts outrank a kept exception; no Python after the first error; no cached fallback key: `83c1caa`
- [x] R2-013b (F2-013, constraint `Value`): depth cap of 128 on decode: `68b326d`
- [x] R2-022 (F2-022): reflexive extension defaults; symmetric structural equivalence: `82935f7`
- [x] R2-035 (F2-035): capture renaming restricted to active keys, over distinct identifiers: `9473515`
- [x] R2-025 (F2-025): recording custom-domain and custom-constraint hook tests (Rust and Python): `3f9c59b`
- [x] R2-007 (F2-007): one `BoxError`; `Sync` lookups; symmetric contexts; constructor and conversion conventions; `checked_*` errors; layer-1 `FromStr` error: `0b44aa2`, `450fa2c`, `dcf2022`
- [x] R2-004 (F2-004): `ForeignPart`, one handle and equality convention, fallible hooks, contexts for custom hooks, provided methods for `Option<Result>`: `a5afd95`
- [x] R2-006 (F2-006): `ParamError` split by family: `d03ac7c`
- [x] R2-005b (F2-005, API part): `IntervalProfile` and `Value` `#[non_exhaustive]`; `SimplifyContext` limits: `b6ccb30`
- [x] R2-032a (F2-032, equality part): `PartialEq`/`Eq`/`Hash`/`Display` on constraint and param values; the two type equalities documented; template widths normalized: `fa89655`
- [x] R2-029a (F2-029, `param`, `constraint`, `foreign`): error-text tables and small stories: `dac8ab1`
- [x] Track A status: gates green; counts recorded (Track A notes); landed as `<hash>`: ready on `fix/a-api`, the hash is recorded when it lands

### Track D: `solver` (the SymPy move and build infrastructure; lands 2nd)

- [x] D0: worktree `port/fix2-solver` created; the baseline gates recorded (the worktree is `fix-d-solver`, branch `fix/d-solver`; see the Track D notes; `8e56252`)
- [x] R2-015 (F2-015): control characters in name hints mapped before they reach a solver (`df46944`)
- [x] R2-014 (F2-014): the process backend's timeout bounds the whole call (`a04e029`)
- [x] R2-040 (F2-040): the mixed int/real equality hazard dropped, except for set-constraint residuals, as the maintainer revised it (Track D notes, N-D1; `648ccd9`)
- [x] R2-005a (F2-005, the move): the SymPy backend moves into `fhy-core-py`; the core drops pyo3 and the `sympy` feature (`4683c70`)
- [x] R2-016 (F2-016): negative powers lift as divisions (`1b2fe6a`)
- [x] R2-038a (F2-038, SymPy part): lifting and substitution memoized by object (`40a5822`)
- [x] R2-039 (F2-039): a versioned, hash-checked prelude module (`bc9a9b9`)
- [x] R2-027 (F2-027): non-vacuous solver properties; Boolean and piecewise generators; z3 against the process backend; SymPy stories (`cf6d922`)
- [x] R2-029d (F2-029, `solver`): error-text tables and small stories (`855cff1`)
- [x] R2-009 (F2-009): docs.rs metadata, `doc(cfg)`, the default-feature doc build, per-crate CI steps, doc drift (`1a83801`)
- [x] `[rebase]` onto `dev-rust` after Track A lands (by the maintainer, onto `46b8596`; fixed forward in `ac27b14`)
- [x] Track D status: gates green; counts recorded (Track D notes, "Track D status"); landing on `dev-rust` is the maintainer's

### Track B: `expression` (expressions, the wire and the corpus; lands 3rd)

- [x] B0: worktree `port/fix2-expression` created; the baseline gates recorded (the worktree is `fix-b-expression`, branch `fix/b-expression`; see Track B notes): `4403638`
- [x] R2-N3 (Alternatives): committed choice checked against the pre-S5 matcher; pinned, not changed: `0393b79`
- [x] R2-012 (F2-012): checked lane counts, fallible reservation, per-chunk broadcast slicing: `52e9581`
- [x] R2-013a (F2-013, `Pattern`): iterative drop and budgeted `Debug`: `9900487`
- [x] R2-034 (F2-034): NaN-propagating `max`/`min`/`clamp`/`relu`/`leaky_relu`; `abs(-0.0) = 0.0`: `d77f7ec`
- [x] R2-037 (F2-037): exact-size `Children`; unary `+` passes its operand through; one stored failing node: `9ac5ba7`
- [x] R2-010 (F2-010): bounded node text in errors; lane index; `occurrence_count`; bounded `str`/`repr` in the binding: `489acef`
- [x] R2-026a (F2-026, evaluator part): scalar, array, kernel and chunk tests: `e7614ee`
- [x] R2-047a (F2-047, evaluator part): `NumberAsBoolean` and the dead arms removed: `f1f9d1a`
- [x] R2-029b (F2-029, `expression`): error-text tables and small stories: `40dfb54`
- [x] `[rebase]` onto `dev-rust` after Tracks A and D land (branched from 35519bb, where both have landed)
- [x] R2-011 + R2-036 + R2-001a + R2-046a, one commit (the wire group, J-4): canonical encoding, canonical float and decimal text with the D-7 revision, DAG-linear keys, a `Value` corpus case, one corpus regeneration: `2385e94`
- [x] R2-042 (F2-042): colliding keys grouped by equivalence; system equivalence independent of tie order: `77197df`
- [x] R2-032b (F2-032, order part): `Ord` for `Constraint` from the canonical key: `859a45c`
- [x] R2-N1 (S17 row 2.10): V2 decoding of literal-heavy trees without re-parsing: `eba22e0`
- [x] Track B status: gates green; counts recorded (Track B notes, "Track B status"); landing on `dev-rust` is the maintainer's: ready on `fix/b-expression`

### Track C: `types-param` (types, checking, params and the symbol table; lands 4th)

- [x] C0: worktree `port/fix2-types-param` created; the baseline gates recorded (the worktree is `fix-c-types-param`, branch `fix/c-types-param`, from `dev-rust` at `35519bb`; see the Track C notes)
- [x] R2-017 (F2-017): a negated literal checks as one literal: `f634007`
- [x] R2-019 (F2-019): the body sweep checks the composed built-ins; its test is not vacuous: `8a2862c`
- [x] R2-001b (F2-001, checker part): the checker memoizes shared nodes: `9bf0a97`
- [x] R2-026b (F2-026, checker part): rstests and broadened properties: `2a5bdb1`
- [x] R2-047b (F2-047, checker part): impossible arms backed by a `const` assertion: `609922f`
- [x] `[rebase]` onto `dev-rust` after Track A lands (branched from `35519bb`, after Tracks A and D landed)
- [x] R2-018 (F2-018): `UnificationError::Substitution`, and an occurs check over the binding graph: `50a5641`
- [x] R2-038b (F2-038, shape substitution): memoized, cycle-marked substitution: `4a5e01f`
- [x] R2-020 (F2-020): descendants checked by `add_symbol`; namespaces decoded first; assignments decoded through `restore`: `836875e`
- [x] R2-021 (F2-021): every domain-level procedure enforces its domain's restriction: `c4d62a4`
- [x] R2-038c (F2-038, permutations): in-set candidates instead of `n!` permutations: `ef57f96`
- [x] R2-028 (F2-028): param and constraint decision-rule tests in Rust: `85f9797`
- [x] R2-046b (F2-046, properties): serde round-trip properties for params, types and symbol tables: `98bc2a8`
- [x] R2-029c (F2-029, `types`, `symbol_table`): error-text tables and small stories: `a61d3b6`
- [x] `[rebase]` onto `dev-rust` after Tracks D and B land (rebased onto `03fb9e4` by the maintainer)
- [x] R2-008 (F2-008): one crate-private exact-arithmetic module, and the decimal exponent bound
- [ ] Track C status: gates green; counts recorded; landed as `<hash>`

### Track E: `binding` (the binding and the Python package; lands 5th, last)

- [ ] E0: worktree `port/fix2-binding` created; the baseline gates recorded
- [ ] R2-N5 (xdist stall): reproduce or clear the 99% stall; account for the missing tests
- [ ] R2-N4 (V1 warnings): the 64 V1 `DeprecationWarning`s asserted or filtered; an unmarked one fails
- [ ] R2-N2 (V1 removal): the texts and docs name 0.3.0
- [ ] R2-002 (F2-002): separate advance and read caps for payload ids, in Rust and Python
- [ ] R2-024 (F2-024): `PartiallyOrderedSet` and `Lattice` pickle, copy and deep-copy
- [ ] R2-044 (F2-044): every Python read before a `PyRef`/`PyRefMut` borrow
- [ ] R2-043 (F2-043): `gil_used = true`; the NumPy input contract documented
- [ ] R2-041 (F2-041): diagnostics read through their report
- [ ] R2-030 (F2-030): binding and interface-suite gaps; the stub test checks members
- [ ] `[rebase]` onto `dev-rust` after Track A lands
- [ ] R2-013c (F2-013, binding readers): depth limits in the dict and member readers
- [ ] R2-003 (F2-003): `__traverse__`/`__clear__`, with every Python object in a visible slot
- [ ] `[rebase]` onto `dev-rust` after Tracks D, B and C land
- [ ] R2-045 (F2-045): `Decimal` through `as_tuple`, ints through bytes
- [ ] R2-031 (F2-031): one `ScopedStack` guard for all six thread-local stacks
- [ ] R2-033 (F2-033): binding boilerplate consolidated
- [ ] Track E status: gates green; counts recorded; landed as `<hash>`

### Final verification

- [ ] On `dev-rust` after Track E: every gate of §I.8, the CI jobs replayed, and the audit's probes that can run re-run (see §I.8.4)

## Contents

- Part I: cross-cutting design
  - I.1 Summary
  - I.2 Rules for every track
  - I.3 Shared designs (S-1 to S-5)
  - I.4 Calls this spec makes (J-1 to J-14)
  - I.5 Revised decisions and documents
  - I.6 Dependencies and sequencing
  - I.7 Tracks, ownership and landing order
  - I.8 Gates
  - I.9 Findings resolution
  - I.10 Non-goals
- Part II: requirements R2-001 to R2-047, then R2-N1 to R2-N5
- Implementation notes (one section per track)

---

# Part I: cross-cutting design

## I.1 Summary

After this work:

- **No walk ignores sharing.** Constraint keys and the type checker are
  linear in distinct nodes (R2-001), and so are SymPy lifting, shape
  substitution and error text (R2-038, R2-010). Equal expressions encode to
  equal bytes (R2-011), and keys and the wire share one canonical node table
  (S-1).
- **No input aborts the process.** This covers deep patterns and values,
  deep dicts (R2-013), huge broadcasts (R2-012), NUL name hints (R2-015),
  and payload ids near the cap (R2-002).
- **Extension points have one shape** (R2-004). There is one `ForeignPart`
  supertrait, one handle, and one equality and hash hook pair. Every other
  hook is fallible, the custom hooks receive their context, and provided
  methods replace `Option<Result>`. The binding's side channels shrink to
  the equality and hash hooks.
- **Errors come one type per family** (R2-006, R2-007), with one `BoxError`.
- **The core drops pyo3.** The SymPy backend lives in the binding
  (R2-005a), so `cargo test -p fhy-core` needs no Python.
- **Semantics the maintainer chose:**
  - NaN propagates through the composed built-ins (R2-034);
  - the mixed int/real equality hazard is gone (R2-040);
  - domain-level procedures enforce their own restriction (R2-021);
  - unification errs instead of silently continuing (R2-018);
  - floats have a shorter canonical text (R2-036, the D-7 revision).
- **The binding cooperates with Python's runtime.** Its classes join cyclic
  GC (R2-003), and lattices pickle (R2-024). Re-entrant reads work (R2-044),
  a `KeyboardInterrupt` always wins (R2-023), and the module declares
  `gil_used` (R2-043).
- **The tests assert what they name** (R2-025 to R2-030, R2-046). The V1
  warnings are asserted or filtered (R2-N4), and the xdist stall is
  explained (R2-N5).

The crate is unpublished, so every change is allowed (first spec, §I.3 rule
6). The breaking changes are R2-004 to R2-007, R2-032a/b, R2-041, R2-019,
R2-040 and R2-047a. They all land before the crates.io release, which the
resolutions require.

## I.2 Rules for every track

1. **Test-first for bugs.** For every item whose category is a bug, the
   regression tests are written first and fail against the track's base.
   Then the fix makes them pass. Tests and fix land in one commit, as the
   slices did (first spec §I.6). Test-only items (R2-025 to R2-030, R2-046,
   R2-N3) add tests that pass. If one fails, it is a new finding: record it
   in the track's implementation notes, and fix it under the item or raise
   it with the maintainer.
2. **One commit per item.** The wire group (J-4) is the one exception. The
   subject uses the repository's style, such as `fix(constraint): …`,
   `feat(foreign)!: …` or `test(param): …`. The body names the item and the
   finding, for example "R2-001a (F2-001)". Commit as the configured git
   user. Never pass `--author`, and never add a `Co-Authored-By` line or any
   AI attribution. Never push.
3. **Every commit leaves its track green** on the per-commit gates (§I.8.1).
   The track gates (§I.8.2) run before landing.
4. **Revisions are appended, not rewritten.** A design-doc decision that an
   item revises gets a new bullet directly under it:
   `**Revised (R2-0xx, <date>):** …`. This is S17's practice. It keeps
   history and makes concurrent edits additive.
5. **Documentation moves with the code.** Rustdoc, the READMEs,
   CONTRIBUTING and `src/fhy_core/_rs.pyi` change in the commit that changes
   the behavior they describe. The stub test (`tests/test_rs_stub.py`,
   widened by R2-030) must stay green.
6. **Python-visible changes are recorded.** Each one gets a row in the
   track's implementation notes: the old text or behavior, the new one, and
   the tests updated. A Python test that pinned the old behavior is
   rewritten, not skipped (D-S4-2's rule).
7. **Worktrees.** Each track works in
   `~/Projects/FhY-core-worktrees/fix2-<track>`, on branch
   `port/fix2-<track>`. The rebase approval recorded for `port/*` branches
   covers these branches.
   - Each worktree has its own `.venv`, built with `uv sync`, and never
     shares the main tree's.
   - Each worktree has its own `target/`, and its own copy of
     `target/tooling/gate-env.sh` with `CARGO_TARGET_DIR` pointing into it.
   - `target/` is git-ignored, so the copy is made by hand.
8. **Landing** follows §I.7.3: rebase onto `dev-rust`, re-run the track
   gates, then fast-forward `dev-rust`.
9. **Housekeeping before starting.** The audit left `maturin==1.15.0` in
   the main tree's `.venv`. Remove it once with
   `target/tooling/pyenv/bin/uv pip uninstall --python .venv/bin/python maturin`.
   This is not a track item.

## I.3 Shared designs

### S-1: the canonical node table (R2-011, R2-001a, R2-032b, R2-042)

- **A new crate-private module, `expression/canonical.rs`.** Its
  `CanonicalTable::build(root, Equivalence)` returns the distinct nodes of
  `root`, with children as table indices.
  - **Order:** post-order of first visit, the root last.
  - **Hash-consing:** a node is added only when no equal entry exists.
    Children are indexed before their parent, so two nodes are equal
    exactly when their own data are equal and their child index lists are
    equal. A lookup is therefore a hash of `(data, child indices)` and one
    local comparison. No deep equality or digest confirmation is needed.
  - **Cost:** the walk memoizes shared handles by `NodeIdentity`, as
    `encode_nodes` does today, so it is linear in distinct handles.
- **Two equivalences, because the wire keeps what `==` ignores:**
  - `Equivalence::Wire`, for the encoder. Literal data are equal when their
    wire texts are equal: floats by bits, with every NaN one (the wire
    writes each NaN as `NaN`), and `-0.0` and `0.0` distinct.
  - `Equivalence::Structural`, for keys. Literal data are equal when
    `LiteralValue ==` holds: all NaNs are one, and the zeros are one.
- **The encoder** (R2-011) serializes the `Wire` table. The decoder shares
  every repeated index, as it does today.
- **The equation key** (R2-001a) renders the `Structural` table. `Ord` for
  `Constraint` (R2-032b) and the system order (R2-042) compare those keys.

### S-2: canonical float text (R2-036; the D-7 revision)

- **The writer.** One crate-private function,
  `expression::literal::write_float(f64, &mut fmt::Formatter)`, writes:
  - `NaN`, `inf` and `-inf` for the non-finite values;
  - `0` and `-0` for the zeros;
  - Rust's shortest round-trip positional text (`{}`) when the magnitude
    lies in `[1e-5, 1e16)`;
  - otherwise, Rust's shortest round-trip exponent text (`{:e}`), such as
    `1e300`, `5e-324`, `1.7976931348623157e308`, `1e16` or `9.9e-6`.
- **Its users.** Every float text the core writes goes through it:
  - the expression literal wire form;
  - the constraint member and value wire form (`constraint/wire.rs`);
  - `LiteralValue`'s `Display`, and so `ExpressionDisplay` and every message
    that shows a literal;
  - the literal and member texts of ordering keys.
- **The decoders.** The float decoder parses with `f64::from_str`,
  re-renders with `write_float`, and refuses a text that differs, with
  `invalid float literal {text:?}: not canonical, expected {canonical:?}`.
  The decimal decoder does the same with `Decimal`'s `Display`. The integer
  decoder is already canonical.

### S-3: extension-point traits (R2-004; used by R2-022, R2-023, R2-025, R2-032a, R2-042, R2-003)

- **Foundation items**, in `foreign` (layer 1):
  - `pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;`.
    It replaces `CallbackError`, `OpaqueError`, `CustomError`, `BackendError`,
    `PassFailure` and the inline spellings (R2-007).
  - `pub trait AsAny: Any { fn as_any(&self) -> &dyn Any; }`, with a blanket
    `impl<T: Any> AsAny for T`. The MSRV is 1.85, and trait upcasting needs
    1.86, so the blanket helper stands in.
    - Call it through the trait object, `(*handle).as_any()` or
      `ForeignPart::as_any(&*handle)`, never on the `Arc`. The blanket impl
      also applies to `Arc<dyn …>` itself, which would hand back the `Arc`'s
      `Any`. Each downcast site gets a test.
  - `pub trait ForeignPart: AsAny + fmt::Debug + Send + Sync`:
    - a required `fn type_name(&self) -> Cow<'_, str>`;
    - a provided `fn to_foreign(&self) -> Result<Foreign, ForeignError>`,
      which answers `NoWireForm { type_name }` with the real type name.
- **One handle.** `pub struct Part<T: ?Sized>(Arc<T>)`, in `foreign`, with
  `new`, `from_arc`, `get`, `ptr_eq` and `Clone`. It replaces `Opaque` and
  every bare `Arc<dyn …>` in `Value`, `Constraint::Custom`,
  `ParamDomain::Custom`, `Type::Extension` and `DataType::Extension`.
  - `PartialEq`, `Eq` and `Hash` are implemented per trait object, through a
    crate-private macro: `ptr_eq`, then `eq_part`; the hash is `hash_part`.
- **One equality pair per trait:**
  - `fn eq_part(&self, other: &dyn Trait) -> bool`, where `Trait` is the
    trait itself (`&dyn OpaqueValue`, `&dyn TypeExtension`, …), backs `==`. It defaults to
    identity. It must be an equivalence relation, symmetric included, as
    each trait's rustdoc says.
  - `fn hash_part(&self, state: &mut dyn Hasher)` backs `Hash`. It defaults
    to feeding nothing, which is consistent with any `eq_part`.
  - They replace `is_equal` (on `OpaqueValue`), `eq_extension`/`hash_extension`
    (on the types traits), and `is_structurally_equivalent` on
    `CustomConstraint` and `CustomDomain`.
  - These two stay infallible. So the binding keeps a side channel for these
    two hooks only (the resolution).
- **Every other hook that runs implementor code is fallible**, returning
  `Result<_, BoxError>`. Each module propagates the error through its own
  type (`ConstraintError::Custom`, `UnificationError::Extension`,
  `ParamError::Custom`, and so on). The hooks per trait:

  | Trait | Hook | Today | After |
  |---|---|---|---|
  | `OpaqueValue` | `ordering_key` | `Cow<str>` | `Result<Cow<str>, BoxError>`, called once when a `Member` is built (J-2) |
  | `OpaqueValue` | `order_against` | `Option<Ordering>` | `Result<Option<Ordering>, BoxError>` |
  | `OpaqueValue` | `check_hashable` | `Result<(), OpaqueError>` | `Result<(), BoxError>` |
  | `OpaqueValue` | `is_member_shaped` | `bool` | unchanged: a fact fixed at construction, calling no implementor code |
  | `TypeExtension`, `DataTypeExtension` | `is_structurally_equivalent(&Type)` | `bool`, default `false` | `Result<bool, BoxError>`, default `eq_part` against the other side's extension (R2-022) |
  | `TypeExtension`, `DataTypeExtension` | `bind_template`, `substitute_template`, `unify` | `Option<Result<_, UnificationError>>` | `Result<_, UnificationError>`, provided bodies calling `types::default_bind_template`, `types::default_substitute_template` and `types::default_unify` (new public functions) |
  | `CustomConstraint` | `free_identifiers` | `HashSet` ("reports no identifier") | `Result<HashSet<Identifier>, BoxError>` |
  | `CustomConstraint` | `evaluate(&Bindings)` | `Result<Outcome, CustomError>` | `evaluate(&Bindings, &ConstraintContext<'_>) -> Result<Outcome, BoxError>` |
  | `CustomConstraint` | `ordering_key` | `Cow<str>` | `Result<Cow<str>, BoxError>`, called once in `ConstraintSystem::new` (J-2) |
  | `CustomConstraint` | `is_alpha_equivalent_under` | `bool` | `Result<bool, BoxError>` |
  | `CustomDomain` | `has_feasible_value`, `feasibility_subset`, `union`, `intersection`, `is_value_set_subset` | no context | gain `&ParamContext<'_>` |
  | term traits | `AlphaEquivalence::is_alpha_equivalent_under`, `FreeIdentifiers::free_identifiers` | infallible | each trait gains `type Error: std::error::Error + Send + Sync + 'static` and returns `Result<_, Self::Error>`; `Expression` uses `Infallible` (J-3) |

- **`Bindings::source`** is documented as binding-only: the channel a
  binding's own adapter reads, which core code never reads. It keeps its
  signature.

### S-4: the bounded node text (R2-010)

- **`Expression::occurrence_count(&self) -> u64`** (public). It is
  saturating, linear in distinct nodes and memoized by identity. It is
  documented as the guard for per-occurrence work such as `Display` and
  `walk_tree`.
- **A crate-private `expression::display::Bounded<'a>(&'a Expression, usize)`.**
  It writes the expression's `Display` text but stops after the given
  number of node occurrences, with `…` for the rest. Messages use
  `MESSAGE_NODE_BUDGET = 64`. The binding's `__str__` and `__repr__` use a
  budget of 1,000 occurrences, but only when `occurrence_count()` exceeds
  1,000,000, so ordinary output is unchanged.

### S-5: `ScopedStack` (R2-031, R2-033)

- **Where.** `fhy-core-py/src/scoped.rs` holds
  `pub(crate) struct ScopedStack<T>`, a thread-local `RefCell<Vec<T>>`.
- **The API.** `push(&'static LocalKey<Self>, value) -> ScopedGuard` pops
  in `Drop`, including on unwind. `with_top(|&T| …)` reads the innermost
  frame.
- **Its users.** All six stacks use it:
  - `types/adapter.rs`, `solver/backends.rs`, `constraint/value.rs`
    (today unguarded);
  - `pass/scope.rs`, `pass/context.rs` and
    `expression/pattern/objects.rs` (today each has its own guard).
- **Later.** R2-033's `ObjectTable` is built on it.

## I.4 Calls this spec makes

Each call below settles something the audit or the resolutions left open.
Each is listed in the final report and needs no further decision unless the
maintainer overrides it.

| # | Where the resolution left room | Call |
|---|---|---|
| J-1 | **F2-002.** The fix says "decoding accepts any id below `ID_CAP`", while its behavior change says "payload ids in `[2^62, 2^63)` are rejected" | Two bounds. A payload id below `ADVANCE_CAP = 2^62` decodes and advances the counter. An id in `[2^62, 2^63)` decodes only if this process already issued it (`id < NEXT_ID`), so it needs no advance; otherwise it is rejected. Fresh ids stop at `ID_CAP`. So an id this process issued after a worst-case payload round-trips here, and a foreign id at or above `2^62` is refused. SOL-003's "consider rejecting unassigned reserved ids" is **not adopted**: ids below 65,536 appear in payloads written before D-1, and the first spec accepted them (B1 §8.4) |
| J-2 | **F2-004.** "Fallible hooks except `eq`/`hash`", while `ordering_key` backs a sort | `ordering_key` is fallible and runs once: in `Member::try_from` for an opaque member (already fallible), and in `ConstraintSystem::new`, which becomes `new(..) -> Result<Self, ConstraintError>`. The key is cached in the member or constraint, so `Ord` (R2-032b) and every later sort compare cached keys infallibly. Python sees the same exceptions as today, now raised directly |
| J-3 | **F2-004.** "The term hooks are infallible", but the term traits are generic, not trait objects | `AlphaEquivalence` and `FreeIdentifiers` each gain an associated `Error`. `Expression` uses `Infallible`, and its callers write `let Ok(x) = …;`, which the MSRV (1.85 ≥ 1.82) accepts. `Constraint`, `ConstraintSystem`, `Param` and `ParamAssignment` use their module's error. `Binder::RebuildError` gains `From` bounds for both. The binding's term adapter keeps no side channel, since no term hook is an `eq`/`hash` hook |
| J-4 | **The shared corpus regeneration.** "Share one corpus regeneration" (F2-001, F2-011, F2-036), while every commit must be green | R2-011, R2-036, R2-001a and R2-046a land as **one commit** that regenerates `tests/golden/serialization_cases.json` and the V2 pins once. Each of the four is ticked with that hash. R2-042 and R2-032b join the commit only if they change the corpus |
| J-5 | **F2-011.** "Equal expressions encode to equal bytes" cannot hold for `-0.0` against `0.0` without losing the sign, and it matters (`1 / -0.0`) | The encoder hash-conses by wire identity (S-1 `Wire`). So equal expressions encode alike except where they differ in a zero's sign, and the rustdoc says so. The key uses `Structural`, so equal expressions always key alike |
| J-6 | **F2-036.** The resolution is labeled "(ii)" but also names option (i)'s canonical decoders, and says nothing about `Display` or keys | Both parts. The new text (S-2) is used by the wire, `Display`, keys and messages, and by constraint member floats as well as literals, so the core never writes a float two ways |
| J-7 | **F2-042.** "A per-group order independent of input" does not exist in general for unequal values whose keys collide | The `ordering_key` contract is strengthened to "equal exactly when `eq_part` holds". A collision becomes a contract violation, as a bad `Hash` is. The sort then puts equivalent members of a tie run next to each other, ordering the runs by first appearance. `ConstraintSystem` equivalence compares each tie run as a multiset. So equivalence no longer depends on the input order, and the rustdoc names the residual order |
| J-8 | **F2-032.** "Derive `Ord` from the key's structure" | `Ord` and `PartialOrd` for `Constraint` compare cached keys (R2-032b). They are lawful under J-7's contract. No `Ord` is added to `Param`, `ParamDomain` or `Value`, since nothing sorts them |
| J-9 | **F2-005.** "Add a limits field to `SimplifyContext`", without saying which backends honor it | `SimplifyLimits { timeout: Option<Duration> }` (`#[non_exhaustive]`, with a builder) is added, and `Solver::simplify` passes it. A Python `Simplifier` subclass reads it from its context. `SympySimplifier` documents that it cannot enforce the timeout, since SymPy has no cancellation. Enforcing it would need a subprocess, which is out of scope (§I.10) |
| J-10 | **F2-005.** Where the SymPy stories go once the core has no pyo3 | They move with the backend, as `#[cfg(test)]` modules of `fhy-core-py/src/solver/sympy/`, under `cargo test -p fhy-core-py`, which embeds Python (PyO3 links libpython in a plain cargo build). `with_embedded_python` and `NoInterpreter` go: a binding always runs inside an interpreter. `tests/sympy_unavailable.rs` is deleted, and its missing-SymPy half becomes a Python subprocess test. So the core needs none of `PYO3_PYTHON`, `PYTHONPATH` or the libpython `LD_LIBRARY_PATH`. The workspace still needs all three, for the binding's tests |
| J-11 | **V1 removal in 0.3.0.** The workspace version is 0.2.0, so the next release, 0.3.0, removes V1 | This spec changes the texts and docs (R2-N2). Deleting V1 is a release-blocking task for 0.3.0, following S17 status's "Left for later" list; no track does it. Consequence: unless a 0.2.x release ships first, no published release carries the deprecation warnings or the upgrade path. The maintainer may want a 0.2.x release that carries them |
| J-12 | **R2-N1.** The optimization has no target | Target: `test_deserialize_from_dict[literals]` within 1.10 of the V1-before row (120.29 µs), from the extension built at the track's head. If the target is missed, record the numbers in "S17 benchmarks" for the maintainer |
| J-13 | **F2-043.** "Record the decision", without saying where | A decision paragraph in CONTRIBUTING "One extension module per process", a README note, and a revision bullet under S17 status in `python-switch.md` pointing here |
| J-14 | **Group (d).** "Done in the pre-release breaking batch" | "Batch" means the pre-release window, not one track. R2-007 and R2-032a are in Track A; R2-008, R2-033, R2-037 and R2-047a/b are in the tracks that own their files. All land before the release |

Smaller calls are made inside the items and marked "(call)". Examples:
the `fhy_core::error` module (R2-007), the variant assignment of R2-006,
the exponent bound of R2-008, and the budgets of S-4.

## I.5 Revised decisions and documents

Each revision is appended under the decision it revises (§I.2 rule 4), in
the item's commit.

| Decision or text | Document | Revised by | Revision |
|---|---|---|---|
| §I.1 "payload ids are capped below 2^63"; D-3 (the Python message "a non-negative integer below 2**63"); CONTRIBUTING's `ID_CAP` paragraph | `rust-workspace.md`, CONTRIBUTING | R2-002 | two bounds (J-1); the message names both |
| D-6 and B3 §5.6 ("post-order of first visit") | `rust-workspace.md` | R2-011 | the table holds each distinct node once, hash-consed by wire identity (J-5) |
| D-7 and R-16 (the `{}` float form, lax `f64::from_str` decode) | `rust-workspace.md` | R2-036 | S-2's text; canonical float and decimal decoders |
| D-10 (`CallbackError`) | `rust-workspace.md` | R2-007 | one `BoxError` in `foreign` |
| D-S8-5 and its follow-up ("the screens move unchanged") | `python-switch.md` | R2-040 | hazard 5 dropped; the other four unchanged |
| D-S12-2, D-S12-3, D-S12-13, N-S12-1, S12 status | `python-switch.md` | R2-005a | the backend lives in the binding; the core has no pyo3; the stories are binding tests (J-10) |
| D-S13-6 (pre-order tree keys) | `python-switch.md` | R2-001a | the table of distinct nodes; the prefixes unchanged |
| D-S13-14, D-S13-18, D-S17-22 (the pending-error slot) | `python-switch.md` | R2-023, R2-004 | the slot serves `eq_part`/`hash_part` only; interrupts outrank |
| D-S11-10 (`eq_extension`/`hash_extension`) | `python-switch.md` | R2-004, R2-022 | `eq_part`/`hash_part`; reflexive, symmetric structural equivalence |
| D-S15-11 (replay namespace by namespace) | `python-switch.md` | R2-020 | every namespace first, then every symbol |
| D-S17-9 ("validated types decode through their constructors") | `python-switch.md` | R2-020 | `ParamAssignment` decodes through `restore` |
| D-S17-16, N-S17-3, S17 status ("0.4.0 proposed") | `python-switch.md` | R2-N2 | 0.3.0 (J-11) |
| W-12 (members keyed by the V2 payload's `repr`) | `python-switch.md` | R2-011 | the key no longer depends on sharing |
| D-S6-17 (records index into their report) | `python-switch.md` | R2-041 | read through the report |
| D-S13-14, D-S16-12 (error mapping) | `python-switch.md` | R2-006 | one `IntoPyErr` per new type; the Python classes unchanged |
| S11 divergences (T-2: "pickling as a call" for the types; lattice and poset not listed) | `python-switch.md` | R2-024 | a new row records that the poset and lattice pickle again |
| S17 benchmarks (row 2.10 flagged) | `python-switch.md` | R2-N1 | the after numbers |
| CONTRIBUTING "Porting to Rust" intro, "Rust test layout", "Canonical values keep their identity", and the module table row for SymPy | CONTRIBUTING | R2-005a | no pyo3 in the core; the recipe applies to the binding's tests |
| CONTRIBUTING "CI policy" | CONTRIBUTING | R2-009 | the per-crate steps |
| CONTRIBUTING layer list (layer 1) | CONTRIBUTING | R2-007 | `error` joins layer 1 |
| CONTRIBUTING "Process-global state" (the thread-local stacks and the slot) | CONTRIBUTING | R2-004, R2-031 | the slot serves eq/hash only; one `ScopedStack` guards all six stacks |
| CONTRIBUTING "Binding crate layout" (converters) | CONTRIBUTING | R2-033 | converters that take context are allowed |
| CONTRIBUTING "One extension module per process" | CONTRIBUTING | R2-043 | the `gil_used` decision |
| README rows: Serializable (0.4.0), Solver (the int/float equality refusal) and Expression (the NumPy contract); `rust/fhy-core/README.md` "The `sympy` feature" | README files | R2-N2, R2-040, R2-043, R2-005a | as each item says |

## I.6 Dependencies and sequencing

| Item | Needs | Why |
|---|---|---|
| R2-001a | R2-011 | the key renders S-1's table (sequencing note 1) |
| R2-011, R2-036, R2-001a, R2-046a | each other | one corpus regeneration (J-4; sequencing note 4) |
| R2-042, R2-032b | R2-001a, R2-004 | they order by the new key and group by `eq_part` |
| R2-038b | R2-018 | the memo is valid only on an acyclic environment (sequencing note 2) |
| R2-033 | R2-031 | `ObjectTable` is built on S-5 (sequencing note 3) |
| R2-004 | R2-007 | `BoxError` and the `error` module |
| R2-004 | R2-023, R2-022, R2-025 | these bug fixes and pins land first, so the reshaping keeps their tests |
| R2-006, R2-005b, R2-032a | R2-004 | they build on the new hooks and errors |
| R2-029a | R2-006 | its tables cover the new error types |
| R2-016, R2-038a, R2-039, R2-027 (SymPy parts), R2-029d | R2-005a | they are done in the backend's new home |
| R2-009 | R2-005a | the docs.rs feature list and CI steps no longer name `sympy` |
| R2-018, R2-020, R2-021, R2-028 | Track A landed | they touch `types/unify.rs` and `param/*`, which R2-004 and R2-006 reshape |
| R2-008 | R2-006, R2-036, R2-005a | it rewrites copies in `param/interval.rs`, `expression/literal/`, `solver/smt/lower.rs` and the moved `sympy/lower.rs` |
| R2-045 | R2-008 | it uses R2-008's decimal constructor and exponent bound |
| R2-003, R2-013c | Track A landed | they touch the binding adapters that R2-004 reshapes |
| R2-031, R2-033 | Tracks A to C landed | they sweep every binding file |
| R2-N1 | R2-011 | "on top of F2-011's encoder rewrite" (the resolution) |

**The breaking API batch** is R2-004, R2-005a/b, R2-006, R2-007 and
R2-032a/b, the user's grouping. Its trait and error changes are what the
other tracks build on, so Track A lands first. The SymPy move (R2-005a) is
mechanically independent of the traits, and it owns the infrastructure
files. So it runs in Track D, in parallel, and lands second. Track A touches
nothing under `solver/sympy*` except the one `impl Simplifier` signature
line that R2-007's rename forces, which Track D resolves on its rebase.

## I.7 Tracks, ownership and landing order

### I.7.1 Tracks

| Track | Items | Owns (edits freely) | Touches in other tracks' areas (additively) |
|---|---|---|---|
| **A `api`** | 11 | `foreign.rs`; the new `error.rs`; `constraint/{value,custom,binding,context,error}.rs` and `constraint.rs`; `param/{error,context,custom,assignment,parameter}.rs` and the error types across `param/`; `types/{extension,error,ty,data_type}.rs`; `term/binder.rs`, `term.rs`; `expression/{screen,operation,callee}.rs` (traits, `FromStr`, constructors); `provenance.rs` and `expression/registry/definition.rs` (constructors); `solver/backend.rs`; `pass/compiler_pass.rs`; the binding's `constraint/{value,custom,observer,system,kinds}.rs`, `types/{adapter,dispatch}.rs`, `term/adapter.rs`, `param/{custom,observer,error}.rs`, `wire.rs` (the side channel); `tests/it/{constraint,param,types,term,foreign}_*` | `types/unify.rs` (the hook call sites only); `solver/sympy.rs` (one line); CONTRIBUTING (layer list, global-state paragraph) |
| **D `solver`** | 10 | `solver/{process,screen,z3,error}.rs`, `solver/smt/*`; `solver/sympy*` until it moves, then `fhy-core-py/src/solver/**`; `tests/it/solver/*`; `.github/workflows/*.yml`; the manifests (`Cargo.toml`, `rust/*/Cargo.toml`); `deny.toml`; `noxfile.py` `SOURCES`; `lib.rs` crate docs; `rust/fhy-core/README.md`; CONTRIBUTING "Porting to Rust" intro, "Rust test layout" and "CI policy" | `expression/evaluate.rs:11` (the broken link); root README solver row |
| **B `expression`** | 16 | `expression/{wire,canonical,literal,literal/*,display,node,build,builtins}.rs`; `expression/evaluate/*`; `expression/pattern/*`; `expression/registry/inline.rs`; `tree/walk.rs`; `constraint/{key,system,wire}.rs`; `tests/golden/*` (corpus and generator); `tests/it/{expression,constraint}/*`; the binding's `expression/{node,materialize,payload}.rs`, `expression/evaluate/*`; `tests/symbolic/test_serialization_pins.py` | `solver/error.rs`, `types/checking/error.rs` (`Display` arms of R2-010); the SymPy lowering tests' expected texts (R2-034) |
| **C `types-param`** | 14 | `types/{unify,environment,wire}.rs`, `types/checking/*`; `param/{domain,decide,algebra,interval,value,screen,wire}.rs`; `symbol_table/*`; `expression/literal/exact.rs` (new, R2-008); `tests/it/{types,param,symbol_table}/*`; the binding's `types/checking.rs`, `types/environment.rs`, `param/{domains,functions,objects}.rs` | `solver/smt/lower.rs` and `fhy-core-py/src/solver/sympy/lower.rs` (R2-008's call sites); `expression/literal/decimal.rs` (R2-008) |
| **E `binding`** | 14 | the binding's `lib.rs`, `identifier.rs`, `lattice.rs`, `symbol_table/*`, `pass/*`, `expression/pattern/{rules,objects,walk}.rs`, `expression/registry/*`, `expression/literal.rs`, `param/parameter.rs`, `solver/facade.rs`; the new `scoped.rs`, `python.rs`, `exceptions.rs`; `identifier.rs` (core, R2-002); `pass/validation.rs`, `diagnostic.rs` (core, R2-041); `src/fhy_core/{identifier,serialization,serialization_upgrade}.py`, `_rs.pyi`; `tests/*.py` fixtures and `pyproject.toml`; `tests/test_rs_stub.py`; `tests/id_cap_decode.rs` | every binding file (R2-031, R2-033, R2-003), after its last rebase; CONTRIBUTING (converters, `gil_used`, the ID paragraph) |

**Conflict rules:**
- **Additive edits.** A track that must edit a file another track owns
  keeps the edit additive: a new function, a new variant of a
  `#[non_exhaustive]` enum, one `Display` arm, or a revision bullet
  appended.
- **Who resolves.** The later-landing track resolves the conflict when it
  rebases. It also updates the earlier track's tests where its own change
  alters them, such as B's reworded solver messages in D's R2-029d tables.
- **Shared files.** `CONTRIBUTING.md`, `README.md`, `_rs.pyi` and the two
  design docs are shared. Each item edits only its own paragraph, row or
  decision.

### I.7.2 Expected conflicts, and how to keep them additive

- **A with every track.** A reshapes the extension traits, `BoxError` and
  `ParamError`.
  - Before their `[rebase]` line, other tracks do not add new uses of
    `CustomError`, `OpaqueError`, `CallbackError`, `BackendError`,
    `PassFailure`, `ParamError` variants or the `Option<Result>` hooks.
  - Items that need them are ordered after the rebase (§I.6).
- **A with D:** the one `impl Simplifier` line in `solver/sympy.rs` and the
  `SimplifyContext` limits (R2-005b). D takes A's version and re-applies the
  move.
- **A with B:** `constraint/key.rs` and `constraint/system.rs`.
  - A changes the member-key call and makes `ConstraintSystem::new`
    fallible (J-2).
  - B replaces `write_expression_key` and adds the tie runs. The two touch
    different functions.
- **A with C:** `types/unify.rs`, where A changes the hook call sites and C
  changes `substitute_avoiding`, and the `param/*` error types. C does its
  edits there only after rebasing onto A.
- **A with E:** the binding adapters. E's R2-003 and R2-013c run after the
  rebase onto A.
- **D with B and C:** after D lands, the SymPy code lives under
  `fhy-core-py/src/solver/sympy/`.
  - B's R2-034 updates the SymPy lowering tests' expected texts there.
  - C's R2-008 edits the moved `sympy/lower.rs`.
- **B with C:**
  - B's R2-010 rewords `TypeCheckError::Rule`'s `Display`; C's R2-029c
    tables pin the new text.
  - B's R2-036 changes float text inside `param` and `types` payloads; C's
    serde properties (R2-046b) round-trip through it.
- **E with everyone:** R2-031 and R2-033 are E's last items, done after E
  has rebased onto all four other tracks.
- **The checklist and implementation notes** never conflict, because each
  track edits only its own group and its own notes section.

### I.7.3 Landing order

1. **Track A `api`.** It is the breaking API batch, and the others build on
   its traits and errors.
2. **Track D `solver`.** It rebases onto A. It changes the build and the
   gate recipe (the core stops needing Python), and it moves the SymPy code
   that B and C then edit in place.
3. **Track B `expression`.** It rebases onto A and D, and carries the one
   corpus regeneration.
4. **Track C `types-param`.** It rebases onto A, D and B. R2-008 comes last,
   once every copy it replaces is in its final place.
5. **Track E `binding`.** It rebases onto all four, then does its sweeping
   items.

Each landing: rebase the track onto `dev-rust`, re-run the track gates of
§I.8.2, then fast-forward `dev-rust`. After each landing, every other track
rebases at its next `[rebase]` line, or sooner if its gates need it.

## I.8 Gates

### I.8.1 Per commit

Run with the worktree's copy of `target/tooling/gate-env.sh` sourced, and
the extension rebuilt (`uv sync`) whenever the binding or the Python
package changed:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --locked
cargo test --workspace --locked --all-features
uv run pytest tests            # when the binding or the package changed
```

### I.8.2 Per track, before landing

Everything in §I.8.1, and:

```bash
# Rust
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked
RUSTDOCFLAGS="-D warnings" cargo doc -p fhy-core --no-deps --locked     # default features alone (F2-009)
for f in z3 ndarray; do RUSTDOCFLAGS="-D warnings" cargo doc -p fhy-core --no-deps --locked --features "$f"; done
for f in "" "--features z3" "--features ndarray"; do cargo clippy -p fhy-core --all-targets --locked $f -- -D warnings; done
cargo deny check
cargo +1.85 check --workspace --lib --locked
for f in "" "--features z3" "--features ndarray"; do cargo +1.85 check -p fhy-core --all-targets --locked $f; done
cargo package --locked -p fhy-core && cargo package --list --locked -p fhy-core   # contents as CI's pattern allows
# Python
uv run pytest tests
uv run pytest tests -m "not very_slow"
uv run nox -s property
uv run nox -s lint
uv run nox -s type_check
uv run nox -s tests_minimal
uv run nox -s golden_expanded
```

- **Known failures before R2-009.** Until R2-009 lands, the default-feature
  and single-feature `cargo doc -p fhy-core` runs fail on the known
  `Prepared::evaluate_array` link, and `cargo +1.85 check -p fhy-core
  --all-targets` shows the 6 known `dead_code` warnings. A track that lands
  before R2-009 must show no failure or warning beyond those, compared with
  its base.
- **Known features before R2-005a.** Until R2-005a lands, the feature loops
  also include `sympy`.
- **Recording.** Each track records its counts in its status line, as the
  slices did. The baseline at `d976522` is:
  - Rust: 4,305 tests, or 4,337 with all features;
  - `pytest tests`: 8,281;
  - `-m "not very_slow"`: 8,314;
  - `property`: 282;
  - `tests_minimal`: 6,315 passed, 642 skipped.

### I.8.3 Per-track extras

| Track | Extra gates | Benchmarks (before at the track's base, after at its head, back to back, per CONTRIBUTING "Benchmarks"; a row slower than 10% is fixed or recorded for the maintainer) |
|---|---|---|
| A | `pytest` of the `*_rust_binding.py` suites for constraints, params, types and terms under `-p no:xdist`, to see the side-channel paths in order | `test_constraint.py`, `test_param.py`, `test_types.py`, `test_term.py` |
| D | `cargo test -p fhy-core --all-features` in a shell **without** `PYO3_PYTHON`, `PYTHONPATH` or libpython on `LD_LIBRARY_PATH` (proves J-10); the CI "Integration Test Targets" and "Package Contents" checks replayed locally; `sh` fakes run on Linux | `test_sympy.py`, `test_solver.py` |
| B | `tests/test_golden_corpora.py`; `golden_expanded` (the expanded serialization corpus replays against the new encoder); the audit's DAG probes as stories on a small stack | `test_serialization.py` (row `test_deserialize_from_dict[literals]`, J-12), `test_expression.py`, `test_constraint.py`, `test_evaluate.py` |
| C | the depth-64 doubling-DAG stories run in release mode too (`cargo test --release -p fhy-core --test it types::`) | `test_type_checking.py`, `test_types.py`, `test_param.py`, `test_symbol_table.py` |
| E | the subprocess-marked tests; `pytest -n 0` and `-n 16` runs (R2-N5); a GC leak test per class family (R2-003) | `test_identifier.py`, `test_pass_infrastructure.py`, `test_pattern.py`, `test_symbol_table.py`, `test_types.py` (lattice), `test_expression.py` (literal construction); GC tracking adds an allocation cost, which must be measured |

### I.8.4 Final verification

This runs on `dev-rust` after Track E lands:
- every gate above;
- a CI replay of the `rust`, `rust-msrv`, `deny`, `lint` and `tests`
  (3.10 and 3.14) jobs;
- the audit's probes whose scratch files still exist in `target/audit/`,
  re-run against the final tree, with the result recorded in the Final
  verification line.

## I.9 Findings resolution

| F | Sev | Triage | Resolution | R2 | Track |
|---|---|---|---|---|---|
| F2-001 | High | a | DAG-linear key and memoized checker | 001a, 001b | B, C |
| F2-002 | High | a | two caps (J-1) | 002 | E |
| F2-003 | High | a | `__traverse__`/`__clear__`, visible slots | 003 | E |
| F2-004 | Medium | b (i) | full unification (S-3) | 004 | A |
| F2-005 | Medium | b (i) | SymPy into the binding; non-exhaustive data; limits | 005a, 005b | D, A |
| F2-006 | Medium | b (i) | split by family | 006 | A |
| F2-007 | Medium | d | conventions batch | 007 | A |
| F2-008 | Medium | d | one exact-arithmetic module | 008 | C |
| F2-009 | Medium | a | docs.rs, `doc(cfg)`, per-crate CI, drift | 009 | D |
| F2-010 | Medium | a | bounded node text, lane index, `occurrence_count` | 010 | B |
| F2-011 | Medium | b (a) | canonical encoding (S-1, J-5) | 011 | B |
| F2-012 | Medium | a | checked lanes, `try_reserve`, chunked broadcast | 012 | B |
| F2-013 | Medium | a | iterative pattern drop and `Debug`; depth caps | 013a, 013b, 013c | B, A, E |
| F2-014 | Medium | a | writer thread, bounded waits, process group, `:print-success` | 014 | D |
| F2-015 | Medium | a | control characters mapped | 015 | D |
| F2-016 | Medium | a | negative powers as divisions | 016 | D |
| F2-017 | Medium | a | negated literal as one literal | 017 | C |
| F2-018 | Medium | b (both) | `Substitution` error and a graph occurs check | 018 | C |
| F2-019 | Medium | a | labels that name built-ins; non-vacuous test | 019 | C |
| F2-020 | Medium | a | descendants, build order, `restore` | 020 | C |
| F2-021 | Medium | b (a) | domain procedures enforce the restriction | 021 | C |
| F2-022 | Medium | a | reflexive, symmetric | 022 | A |
| F2-023 | Medium | a | interrupts outrank; stop after the first error | 023 | A |
| F2-024 | Medium | a | `__reduce__`/`__setstate__` | 024 | E |
| F2-025 | Medium | c | custom-hook tests | 025 | A |
| F2-026 | Medium | c | evaluator and checker tests | 026a, 026b | B, C |
| F2-027 | Medium | c | solver property tests | 027 | D |
| F2-028 | Medium | c | decision-rule tests | 028 | C |
| F2-029 | Medium | c | error-text tables | 029a, 029b, 029c, 029d | A, B, C, D |
| F2-030 | Medium | c | binding tests and member stub check | 030 | E |
| F2-031 | Low | a | `ScopedStack` (S-5) | 031 | E |
| F2-032 | Low | d | std traits; `Ord` from the key (J-8) | 032a, 032b | A, B |
| F2-033 | Low | d | binding consolidation | 033 | E |
| F2-034 | Low | b (a) | NaN-propagating bodies | 034 | B |
| F2-035 | Low | a | restricted, deduplicated renaming | 035 | A |
| F2-036 | Low | b (ii) and (i) | canonical decoders and the D-7 revision (J-6) | 036 | B |
| F2-037 | Low | d | exact-size children | 037 | B |
| F2-038 | Low | a | three memo and enumeration fixes | 038a, 038b, 038c | D, C, C |
| F2-039 | Low | a | versioned, hash-checked prelude | 039 | D |
| F2-040 | Low | b (a) | hazard dropped | 040 | D |
| F2-041 | Low | a | read through the report | 041 | E |
| F2-042 | Low | a | tie runs grouped (J-7) | 042 | B |
| F2-043 | Low | b | `gil_used = true`; the NumPy contract | 043 | E |
| F2-044 | Low | a | reads before borrows | 044 | E |
| F2-045 | Low | a | `as_tuple`; bytes | 045 | E |
| F2-046 | Low | c | serde properties; a `Value` corpus case | 046a, 046b | B, C |
| F2-047 | Low | d | dead code removed; guarantees as types | 047a, 047b | B, C |
| S17 row 2.10 | – | settled | optimized on top of R2-011 | N1 | B |
| V1 removal | – | settled | 0.3.0 texts (J-11) | N2 | E |
| `Alternatives` | – | settled | checked; Python did not backtrack; pinned | N3 | B |
| V1 warnings | – | c | asserted or filtered | N4 | E |
| xdist stall | – | c | reproduce or clear | N5 | E |

## I.10 Non-goals

- **The optional parts the fixes offered.** These are left out:
  - F2-010's node budget in `FormatOptions`, its shared-once display style,
    and its occurrence-limited decode entry point;
  - F2-013's explicit-stack matcher (the recursion depth of matching stays
    documented);
  - F2-015's `Symbol::Int` naming of z3 constants (sanitized names already
    remove the NUL).
- **F2-002's reserved-id rejection** (J-1).
- **Enforcing a SymPy timeout** (J-9).
- **Free-threaded support.** F2-043 option (ii): a 3.14t CI job and
  threaded stress tests. Copying NumPy inputs is also left out.
- **Deleting V1** (J-11), which is a 0.3.0 release task.
- **The audit's first-audit carry-overs**, N-2 and N-3, stay deferred.
- **The audit's "Noted, not findings" items stay unchanged:** L-1 to L-6,
  the EXP minor notes, and the `PyOnceLock` monkeypatching note. R2-033 may
  add one CONTRIBUTING sentence on the last.
- **`sign(nan)`** stays `0`. The F2-034 resolution names `max`, `min`,
  `clamp`, `relu`, `leaky_relu` and `abs`; R2-034 pins `sign(nan)` as it is.
- **Style work.** The 408 and 810 nursery and pedantic clippy warnings are
  left alone.
- **A coverage gate.** F2-047 accepts about 93% of regions as the ceiling,
  with no gate added.

---

# Part II: requirements

Each requirement names its track and step. **Change** is the exact change.
**Tests** lists what proves it. For a bug these are written first and fail
at the base. **Behavior change** states what users see. **Revises** names
the decisions and documents it amends (§I.5).

### R2-001a: DAG-linear constraint ordering keys (F2-001, key part)

- **Track:** B, the wire group (J-4). **Needs:** R2-011 (S-1).
- **Change:**
  - **The equation key.** `constraint/key.rs`'s `equation_key` writes
    `equation|`, then S-1's `Structural` table: each distinct node once,
    `;`-separated, as `kind[data](i,j,…)` with children by table index.
    - A literal writes today's `write_literal_key` text, with S-2's floats.
    - An identifier writes its id.
    - A callee writes `builtin:<name>`, or `named:"<name>"` with the name
      escaped by a crate-private quoting helper. The `Debug` text is gone.
  - **Set keys** keep their shape; their float members use S-2's text.
  - **Cost.** `ConstraintSystem::new` still keys each member once, now in
    time and space linear in distinct nodes.
  - **Docs.** The module docs of `constraint/key.rs` and
    `constraint.rs:19-20` describe the table.
- **Tests** (`tests/it/constraint/key_stories.rs`, unless named):
  - `a_doubling_dag_of_depth_64_keys_in_linear_size`: for
    `e_{k+1} = e_k + e_k` at depth 64, the key is shorter than 64 × 48
    bytes, and `ConstraintSystem::new([e_64 == 0])` returns;
  - `keys_are_equal_however_the_expression_is_shared`: `x + x` built from
    one leaf and from two leaves;
  - in `constraint_properties.rs`,
    `keys_are_equal_exactly_when_constraints_are_structurally_equivalent`,
    over generated pairs with random sharing (built through `substitute`);
  - `a_callee_key_is_its_stable_text`: pins `f(x)` and `max(x, 1)`;
  - the existing pins, updated to the new text;
  - in Python (`tests/symbolic/constraint/test_constraint_rust_binding.py`):
    `build_ordering_key()` of `x + 1 == 0` pinned, and a depth-40 doubling
    DAG constraint builds in under a second.
- **Behavior change:**
  - The key text changes, and Python's `build_ordering_key` returns it.
  - Systems whose members' keys reorder get a new canonical order, and so
    new V2 text. The corpus is regenerated in the group commit.
- **Revises:** D-S13-6, second bullet. Its first bullet, the kind prefixes,
  still holds.

### R2-001b: the type checker memoizes shared nodes (F2-001, checker part)

- **Track:** C, step 3.
- **Change:**
  - `types/checking/checker.rs` (the work list at `:480-700`) memoizes the
    result of each shared node (`is_shared()`), keyed by `NodeIdentity`
    when it has no expected type and by `(NodeIdentity, expected)`
    otherwise, as the other walks do.
  - A memoized error is replayed as the same error.
- **Tests:**
  - in `tests/it/types/checking/checker_stories.rs`,
    `a_doubling_dag_of_depth_64_checks` (without the memo it would not
    finish);
  - in `checker_properties.rs`,
    `checking_a_shared_dag_equals_checking_its_unshared_tree`;
  - in Python (`tests/types/checking/`), a depth-40 DAG checks in under a
    second.
- **Behavior change:** none; it is faster.
- **Revises:** none.

### R2-002: separate advance and read caps for payload ids (F2-002)

- **Track:** E, step 4.
- **Change:**
  - **Rust** (`identifier.rs`):
    - Add `pub const ADVANCE_CAP: u64 = 1 << 62`.
    - A payload id below `ADVANCE_CAP` decodes and advances the counter.
    - An id in `[ADVANCE_CAP, ID_CAP)` decodes only when it is below the
      counter, so it was issued here and needs no advance.
    - Any other id is `IdOutOfRange`, whose `Display` names both bounds.
    - `try_allocate_id` returns `IdSpaceExhausted` once the counter reaches
      `ID_CAP`, and `Identifier::new`'s panic doc says so.
    - `try_restore` and `try_advance_counter_past` follow the same rule.
  - **Python** (`src/fhy_core/identifier.py`): the field check mirrors it,
    reading the counter through `_rs`. `_ADVANCE_CAP` is added with the
    "Matches the Rust implementation" line.
  - **Docs.** The module docs, CONTRIBUTING's `ID_CAP` paragraph and D-3's
    message are updated.
- **Tests:**
  - `tests/id_cap_decode.rs` is rewritten (still one test):
    - decode `ADVANCE_CAP - 1`;
    - create an identifier, and round-trip it through JSON and postcard;
    - a foreign `ADVANCE_CAP` is refused;
    - `ID_CAP - 1` is refused unless issued here.
  - `identifier::tests::an_id_issued_here_above_the_advance_cap_round_trips`,
    on a local counter.
  - In Python, `test_deserializing_the_largest_payload_id_leaves_construction_working`
    is updated, `test_an_identifier_created_after_the_largest_payload_round_trips`
    (subprocess) is added, and the pickle case of the audit probe
    (`p25_idcap.py`) passes.
- **Behavior change:** payload ids in `[2^62, 2^63)` that this process did
  not issue are rejected, in Rust and Python alike. Python's message names
  `2**62`.
- **Revises:** `rust-workspace.md` §I.1 and D-3; CONTRIBUTING's `ID_CAP`
  paragraph.

### R2-003: binding classes take part in cyclic GC (F2-003)

- **Track:** E, after the rebase onto A.
- **Change:**
  - **Traversal.** Every pyclass that holds a user object gets
    `__traverse__`, and `__clear__` where it owns the reference. These are
    the classes in F2-003's location list, plus any class R2-005a moved.
    `_rs.SympySimplifier` now holds its SymPy handles in a pyclass, so it
    is traversable too.
  - **Locks.** Behind a `Mutex`, traverse with `try_lock`; never block.
  - **Hidden references.** Every `Py` the binding keeps inside a Rust
    closure or a core trait object moves to one visible slot, an
    `Arc<Mutex<Option<Py<PyAny>>>>`. The owning pyclass traverses and clears
    it, and the closure or adapter borrows from it. This covers the
    `RewriteRule` callbacks, the `Solver`'s backends, and the S-3 adapters
    held in `Part`s.
- **Tests:**
  - `tests/test_gc_cycles.py`, new: one test per class family, modeled on
    `probe_gc_cycles2.py`. It covers a pass keeping its manager, a rewrite
    callback closing over its rule list, a simplifier holding its solver, a
    lattice element pointing at its lattice, a symbol table, a param and a
    registry entry. Each asserts a `weakref` to the cycle dies after
    `gc.collect()`.
  - A plain-Python control cycle.
  - A check that `type(obj).__flags__ & Py_TPFLAGS_HAVE_GC` is set for
    every exported class that holds objects.
- **Behavior change:** unreachable cycles are freed.
- **Revises:** CONTRIBUTING's global-state paragraph (the visible slots).

### R2-004: one shape for the extension points (F2-004)

- **Track:** A, step 7. **Needs:** R2-007.
- **Change:** S-3 in full.
  - **Every extension trait is `: ForeignPart`:** `OpaqueValue`,
    `TypeExtension`, `DataTypeExtension`, `CustomConstraint` and
    `CustomDomain`. Each drops its own `type_name`, `to_foreign` and
    `as_any`. `CustomConstraint` and `CustomDomain` gain `type_name`, so
    `NoWireForm` names the real type.
  - **`Part<T>`** replaces `Opaque` and every bare `Arc<dyn …>` in public
    enums.
  - **`eq_part`/`hash_part`** replace the four equality names and the one
    hash hook.
  - **Fallible hooks**, the `ParamContext` and `ConstraintContext`
    arguments, and provided methods with public default functions, as S-3's
    table lists.
  - **The binding.** Its adapters implement the new signatures.
    - A raised exception becomes the hook's `Err`, boxed as `PyErr`, and
      the entry function unboxes it.
    - The side channels of `types/adapter.rs` (S11's context error) and
      `wire.rs` remain only for `eq_part`/`hash_part`. The term adapter's
      is removed, since no term hook is an equality or hash hook (J-3).
    - `constraint/value.rs`'s `PENDING_ERROR` remains only for `eq_part`
      and `hash_part`.
- **Tests:**
  - **Pins kept.** R2-025's recording hooks keep their assertions under the
    new signatures.
  - **New stories:**
    - `tests/it/constraint/value_stories.rs::a_failing_opaque_key_fails_member_construction`;
    - `tests/it/constraint/system_stories.rs::a_failing_custom_key_fails_system_construction`;
    - `…::a_failing_custom_scope_is_an_error_not_a_closed_constraint`;
    - `tests/it/types/extension_stories.rs::a_failing_extension_equivalence_is_a_unification_error`;
    - `tests/it/foreign_stories.rs::no_wire_form_names_the_custom_type`;
    - `…::as_any_on_a_part_downcasts_to_the_implementation`.
  - **Compile checks:** a `compile_fail` doctest that `ForeignPart` without
    `Send + Sync` is refused, and `assert_impl` checks that every
    `Part<dyn …>` is `Send + Sync`.
  - **Python:** the interface suites for constraints, params, types and
    terms still pass unchanged. That proves the exceptions still surface as
    themselves.
- **Behavior change:**
  - Implementors add `type_name` to the two `Custom*` traits, and drop
    `as_any`.
  - A failing hook is an `Err` from the core operation, not `false`, an
    empty set or an empty key.
  - `ConstraintSystem::new` returns `Result` (J-2).
  - `NoWireForm` names the real type.
  - Python behavior is unchanged, except that a failing custom
    constraint's scope now raises where it used to report no identifier.
- **Revises:**
  - D-S11-10, D-S13-14, D-S13-18 and D-S17-22 (the slot and side channels);
  - CONTRIBUTING "Serialization is plain serde" (`Part` in the Pattern F
    wording) and the global-state paragraph;
  - the rustdoc of `constraint`, `param`, `types` and `term`.

### R2-005a: the SymPy backend moves into `fhy-core-py` (F2-005, the move)

- **Track:** D, step 4.
- **Change:**
  - **The move.** `solver/sympy.rs` and `solver/sympy/*`, the prelude
    included, move to `fhy-core-py/src/solver/sympy/`.
    - The existing binding file `fhy-core-py/src/solver/sympy.rs`, the
      pyclass, becomes that module's root, and the backend's files become
      its submodules.
    - Every item becomes `pub(crate)`.
    - `with_embedded_python` and `SympyUnavailableError::NoInterpreter` are
      deleted (J-10).
    - The imports it uses from `fhy_core` are all public (checked at
      `d976522`).
  - **The core** drops:
    - `solver.rs:90-111`'s `sympy` module and re-exports;
    - the `pyo3` dependency;
    - `[features] sympy`;
    - the `[[test]] sympy_unavailable` target and `tests/sympy_unavailable.rs`;
    - `tests/it/solver/sympy_*.rs` and `tests/it/support/sympy.rs`;
    - `/src/**/*.py` from `include`.
  - **The stories** become `#[cfg(test)]` modules of the binding's
    `solver/sympy/`, with the builders they need copied into a binding test
    support module.
  - **The binding's manifest:** `fhy-core = { …, features = ["ndarray"] }`.
  - **The workspace** `Cargo.toml` pyo3 comment reads "the binding's".
  - **CI** (`.github/workflows/python-package.yml`):
    - the "Install z3 and SymPy" comment says SymPy serves the binding's
      tests;
    - the integration-target list becomes `id_cap_decode it`;
    - the Package Contents pattern drops `src/solver/sympy/prelude\.py`.
  - **`noxfile.py` `SOURCES`** names `rust/fhy-core-py/src/solver/sympy`.
  - **`deny.toml`:** no edit is expected, since pyo3 stays in the graph
    through the binding. `cargo deny check` must pass with no newly unused
    allowance.
  - **The gate recipe** (answering the question in the task):
    - The core no longer needs `PYO3_PYTHON`, `PYTHONPATH` or libpython on
      `LD_LIBRARY_PATH`.
    - `cargo test --workspace` still needs all three, for the binding's
      SymPy tests, and needs the z3 variables for `--all-features`.
    - Update the comment in each worktree's `gate-env.sh` copy. The file is
      untracked. Its variables stay.
  - **CONTRIBUTING:**
    - the "Porting to Rust" intro says the core has no pyo3;
    - "Rust test layout" moves the SymPy paragraph and recipe to the
      binding's tests, and says one fresh-process target remains;
    - "Canonical values keep their identity" drops the SymPy exception;
    - the module-table row reads "the binding (`fhy-core-py`'s
      `solver::sympy`), a `fhy_core::solver::Simplifier`".
  - **README:** `rust/fhy-core/README.md` deletes "The `sympy` feature" and
    the solver bullet's mention. The root README keeps its line 177, which
    is still true.
- **Tests:**
  - the moved stories pass under `cargo test -p fhy-core-py`;
  - `tests/symbolic/expression/passes/test_sympy_rust_binding.py::test_missing_sympy_reports_unavailable`
    (subprocess, `sys.modules["sympy"] = None`) replaces the target's
    missing-SymPy half;
  - D's extra gate: `cargo test -p fhy-core --all-features` with no Python
    environment;
  - `cargo package --list` holds no `.py` file.
- **Behavior change:**
  - Rust: `fhy_core::solver::{SympySimplifier, SympyError, SympyErrorKind,
    SympyPhase, SympyUnavailableError}` and the `sympy` feature are gone.
  - Python: none.
- **Revises:** D-S12-2, D-S12-3, D-S12-13, N-S12-1 and S12 status; D-19's
  package list (no Python file); the CONTRIBUTING and README sections
  above.

### R2-005b: non-exhaustive data and simplification limits (F2-005, API part)

- **Track:** A, step 9.
- **Change:**
  - **`IntervalProfile`** (`param/domain.rs:83-93`) becomes
    `#[non_exhaustive]`, with private fields, `IntervalProfile::new(..)`
    taking a small builder or named enums (call), and getters.
  - **`Value`** (`constraint/value.rs:127`) becomes `#[non_exhaustive]` and
    loses its `#[expect(clippy::exhaustive_enums)]`. The binding's matches
    gain wildcard arms that raise `TypeError`.
  - **Limits.** `solver::SimplifyLimits { timeout: Option<Duration> }`
    (`#[non_exhaustive]`, `Default`, `with_timeout`) is added.
    `SimplifyContext` gets `with_limits` and `limits()`, and
    `Solver::simplify` passes its configured limits.
    - The binding exposes `context.timeout` to Python `Simplifier`
      subclasses, additively.
    - `SympySimplifier`'s rustdoc and docstring say it does not enforce the
      timeout (J-9).
- **Tests:**
  - `tests/it/param/domain_stories.rs::an_interval_profile_is_built_and_read_through_its_api`;
  - `tests/it/solver/solver_stories.rs::simplify_hands_its_limits_to_the_simplifier`,
    with a recording `Simplifier`;
  - in Python, a subclass reads `context.timeout`.
- **Behavior change:**
  - `IntervalProfile { .. }` literals and exhaustive `Value` matches no
    longer compile outside the crate.
  - `Simplifier` implementations may read limits.
- **Revises:** none.

### R2-006: `ParamError` split by family (F2-006)

- **Track:** A, step 8. **Needs:** R2-004.
- **Change:** five `#[non_exhaustive]` enums in `param/error.rs`, each with
  a hand-written `Display` and `source`:

  | Type | Returned by | Variants (from today's) |
  |---|---|---|
  | `DomainError` | the six domain constructors | `EmptyValues`, `NotALeafValue`, `NanValue`, `IncomparableValues`, `DuplicateValues` |
  | `ParamBuildError` | `Param::new`, `with_constraint(s)`, `check_bounds_are_ordered` | `ForbiddenConstraintKind`, `NotABound`, `NativeConstantVariable`, `OutOfScope`, `NaturalBound`, `UnorderedBounds`, `EmptyInterval`, `Constraint(ConstraintError)`, `Custom(BoxError)` |
  | `AssignmentError` | `ParamAssignment::new`, `restore`, the value checks | `Inadmissible`, `ViolatedConstraint`, `UnverifiedConstraint`, `BindingsBindVariable`, `Constraint`, `Custom` |
  | `IntervalError` | `interval.rs`, `Param::checked_*` | `NotAnIntervalOperand`, `UnsupportedOperand`, `NonBoundOperand`, `MalformedBound`, `Build(ParamBuildError)` |
  | `ParamError` | the questions and the set algebra (`decide.rs`, `algebra.rs`, `union`, `intersection`, equivalence) | `KindMismatch`, `EmptyUnion`, `EmptyIntersection`, `DifferentPermutationMembers`, `UnsupportedUnion`, `EmptyParamIntersection`, `Rescope`, `UnexpectedConstraintKind`, `Build(ParamBuildError)`, `Constraint`, `Custom` |

  - **Checking the assignment.** The implementer confirms each variant's
    producers with `rg`, and records any variant that moves elsewhere in
    the notes.
  - **Text.** Messages keep today's text.
  - **The binding** implements `IntoPyErr` for each type, raising exactly
    the Python class the old variant raised.
- **Tests:**
  - every existing `expect_err` pattern is updated to the new type;
  - `tests/it/param/error_stories.rs`, new, holds one test per public
    operation asserting its error type. These are compile-level pins that
    the operation returns only its family;
  - the Python suites pass unchanged.
- **Behavior change:** the Rust error types change per operation. Python is
  unchanged.
- **Revises:** D-S16-12 (the mapping is per type).

### R2-007: API conventions unified (F2-007)

- **Track:** A, step 6.
- **Change**, in sub-steps, each its own commit if large:
  1. **One `BoxError`** (S-3), replacing the five aliases and the inline
     spellings. `types` and `evaluate` stop importing the pattern module.
  2. **A layer-1 `fhy_core::error` module** (call) holds
     `UnknownNameError`, moved from `expression/operation.rs`, and the
     crate-private `FromStr` macro.
     - The macro is applied to `DiagnosticLevel`, `PassHook`, `QueryKind`,
       `Logic` and `DomainKind`. `SympyPhase` moves out with R2-005a.
     - `types/core_data_type.rs`, `types/qualifier.rs` and
       `symbol_table/frame.rs` return it.
     - CONTRIBUTING's layer 1 lists `error`.
  3. **`Sync` supertraits** on `Environment`, `SymbolTypes` and
     `SortLookup`. Static `assert_send_sync` checks cover `SimplifyContext`,
     `QueryContext` and `BooleanScreen`. `IdentifierTypes` and
     `CallTargets` are documented as `!Sync`-friendly (the binding's
     implementations hold `Bound`), and so is `TypeChecker`.
  4. **Symmetric contexts.** `constraint::{Event, Observer}` become
     `ConstraintEvent`/`ConstraintObserver`, beside
     `ParamEvent`/`ParamObserver`. `ParamContext` holds a
     `ConstraintContext` and lends it, instead of rebuilding one per call.
  5. **Constructors and conversions:**
     - the six `try_new`-only types get `new -> Result`;
     - `impl TryFrom<Value> for Member` replaces `try_from_value`;
     - `impl From<LiteralValue> for Value` replaces `from_literal`;
     - the anonymous `bool` arguments of `IntegerDomain::new`,
       `IntervalIntegerDomain::new` and `check_bounds_are_ordered` become
       named two-variant enums (call, recorded);
     - `ParamAssignment::new_unchecked` is renamed `new_unvalidated`.
  6. **One signal for "not an interval operand".** `Param::checked_{add,sub,mul,reverse_sub}`
     return `Err(IntervalError::NotAnIntervalOperand)` instead of
     `Ok(None)`. The binding maps it to `NotImplemented`.
- **Tests:**
  - `tests/it/foundation_stories.rs::every_name_enum_parses_its_own_text`,
    an rstest over the six enums (round trip and unknown-name error);
  - `assert_send_sync` compile checks;
  - `param_stories.rs::checked_add_of_a_non_interval_param_is_an_error`;
  - in Python, `param + param` for non-interval params still returns
    `NotImplemented`, and so raises `TypeError`.
- **Behavior change:**
  - Renames and signatures only.
  - Closures used as `SymbolTypes` must be `Sync`.
- **Revises:** D-10 in `rust-workspace.md`; CONTRIBUTING's layer list.

### R2-008: one exact-arithmetic module (F2-008)

- **Track:** C, last item. **Needs:** R2-006, R2-036 and R2-005a landed.
- **Change:**
  - **The module.** `expression/literal/exact.rs` (crate-private) holds:
    - `Rational { numerator: BigInt, denominator: BigInt }`;
    - `Rational::of_f64`, exact from the IEEE-754 decomposition;
    - `Decimal::to_rational`;
    - `Rational::to_decimal`, where exact;
    - `LiteralValue::exact_cmp`.
  - **Its users.** `param/interval.rs`, `param/value.rs` and
    `solver/smt/lower.rs` call it, and so does the binding's moved
    `solver/sympy/{lower,lift}.rs`. The binding reaches it through a narrow
    public surface, where needed: `Decimal::to_rational_parts` and
    `Decimal::from_parts` (public, documented).
  - **The copies are deleted.** Every out-of-range conversion is checked:
    no `unwrap_or(0)`, `unwrap_or(usize::MAX)` or `expect`.
  - **The exponent bound** (call): `Decimal::MAX_EXPONENT_MAGNITUDE:
    u32 = 10_000`, which the text grammar implies for any accepted literal.
    It is enforced by `Decimal::from_parts`. R2-045 shares it.
- **Tests:**
  - `expression/literal/exact.rs` unit properties: `of_f64` round-trips
    every finite `f64` through `to_f64`; `exact_cmp` agrees with a `BigInt`
    cross-multiplication oracle;
  - `tests/it/param/interval_stories.rs::a_decimal_bound_with_a_huge_exponent_is_refused_not_misordered`,
    through `from_parts`;
  - the existing lowering and interval stories pass unchanged.
- **Behavior change:** none intended. A decimal beyond the bound is refused
  by `from_parts`; the grammar never produced one.
- **Revises:** none.

### R2-009: docs.rs, `doc(cfg)`, the default-feature doc build and per-crate CI (F2-009)

- **Track:** D, step 10. **Needs:** R2-005a.
- **Change:**
  - **docs.rs metadata** in `rust/fhy-core/Cargo.toml`:
    - `[package.metadata.docs.rs] features = ["ndarray"]`, plus `"z3"` if
      a local `DOCS_RS=1 cargo doc --features z3` without libz3 succeeds.
      If not, record why z3 is left out.
    - `rustdoc-args = ["--cfg", "docsrs"]`.
    - `#![cfg_attr(docsrs, feature(doc_cfg))]` in `lib.rs`.
    - `check-cfg = ['cfg(docsrs)']` in the workspace lints.
  - **Gated items.** `#[cfg_attr(docsrs, doc(cfg(feature = "…")))]` on
    every item gated by `z3` or `ndarray`.
  - **The broken link.** `expression/evaluate.rs:11`'s link to
    `Prepared::evaluate_array` becomes code text.
  - **CI**, job `rust`:
    - a default-feature `cargo doc -p fhy-core --no-deps` step;
    - `cargo clippy -p fhy-core --all-targets` with no features and with
      each feature alone;
    - the docs step's comment corrected.
  - **CI**, job `rust-msrv`: `cargo check -p fhy-core --all-targets` with no
    features and with each feature alone.
  - **The 6 `dead_code` warnings** that 1.85 reports are fixed.
  - **Drift:**
    - the `lib.rs` summary lists every module;
    - the serialization claim reads "…or its `Canonical<T>` does";
    - the Cargo `description` names the symbol table, stack and scope;
    - `repository` loses `.git`.
  - **CONTRIBUTING "CI policy"** lists the new steps.
- **Tests:** the per-crate gate loops of §I.8.2 pass without the
  "known failures" exemption. From this item on, no track may use the
  exemption.
- **Behavior change:** none.
- **Revises:** CONTRIBUTING "CI policy".

### R2-010: bounded node text in errors (F2-010)

- **Track:** B, step 6.
- **Change:** S-4.
  - **Bounded text.** These error `Display` arms write
    `Bounded(node, MESSAGE_NODE_BUDGET)` instead of `{node}`:
    - `expression/evaluate/error.rs`'s `BooleanArithmetic`,
      `NumberAsBoolean` (until R2-047a deletes it), `MixedBranches` and
      `Lane`;
    - `solver/error.rs`'s `NonFiniteLiteral`, `Call`, `SortMismatch` and
      `UnsupportedPower`;
    - `types/checking/error.rs`'s `TypeCheckError::Rule`, for both `root`
      and `at`.
    - The node stays in each error's field.
  - **The lane.** `EvaluationError::Lane` gains `lane: Option<usize>`, the
    flat C-order index of the first failed lane, and the message names it.
  - **`occurrence_count`** is added.
  - **The binding.** `Expression.__str__`/`__repr__` fall back to the
    bounded form above 1,000,000 occurrences.
- **Tests:**
  - in `tests/it/expression/evaluate_stories.rs`,
    `a_lane_error_on_a_63_level_doubling_dag_displays_in_bounded_size`
    (< 4 KB);
  - `a_lane_error_names_its_lane`;
  - in `tests/it/solver/smt_lowering_stories.rs`,
    `a_lowering_error_on_a_depth_20_dag_displays_in_bounded_size`;
  - in `tests/it/types/checking/checker_stories.rs`,
    `a_rule_error_on_a_dag_displays_in_bounded_size`;
  - `tests/it/expression/node_stories.rs::occurrence_count_saturates_and_is_linear`
    (depth 70, `u64::MAX`);
  - in Python, `str()` of a decoded 61-node doubling payload returns
    promptly and ends in `…`;
  - the existing message pins in `evaluate_stories.rs` and the Python suite
    are updated.
- **Behavior change:**
  - Long error texts are truncated.
  - `repr`/`str` change only above the threshold.
  - The binding's messages change with the core's.
- **Revises:** none; the display docs gain the budget.

### R2-011: canonical expression encoding (F2-011)

- **Track:** B, the wire group (J-4).
- **Change:** S-1.
  - `expression/wire.rs`'s `encode_nodes` builds the `Wire` table.
  - The `Serialize` rustdoc says equal expressions encode alike except for
    a zero's sign (J-5), and that decoding shares every repeated subtree.
- **Tests:**
  - in `tests/it/expression/wire_stories.rs`,
    `x_plus_x_encodes_alike_from_one_leaf_or_two`;
  - `s_times_s_encodes_alike_however_s_was_built` (the 184- and 302-byte
    pair of `p2`);
  - `a_negative_and_a_positive_zero_stay_apart_on_the_wire`;
  - `a_decoded_expression_shares_every_repeated_subtree`;
  - in `…/properties.rs`, `equal_expressions_encode_to_equal_bytes` over
    generated pairs with random sharing, zero signs excluded;
  - the corpus regenerated, and `tests/it/serialization_golden.rs` replaying
    it byte for byte;
  - the V2 pins in `tests/symbolic/test_serialization_pins.py` updated.
- **Behavior change:** payloads shrink, and decoded values share repeated
  subtrees. The corpus is regenerated.
- **Revises:** D-6 and B3 §5.6 (`rust-workspace.md`); a note under W-12.

### R2-012: large broadcasts are errors, not aborts (F2-012)

- **Track:** B, step 2.
- **Change**, in `expression/evaluate/array.rs`:
  - the lane count uses `checked_mul`, and an overflow is a new
    `EvaluationError::BroadcastTooLarge { shape }`, a `ValueError` in
    Python;
  - the output is reserved with `try_reserve`, and a failure is a new
    `EvaluationError::OutOfMemory { lanes }`, a `MemoryError`;
  - `broadcast_view`'s `unreachable!` becomes that error;
  - a broadcast binding is sliced per chunk instead of materialized whole
    (`ChunkSource::Copied` per chunk).
- **Tests:**
  - in `tests/it/expression/evaluate_array_stories.rs`,
    `a_broadcast_whose_lane_count_overflows_is_an_error`;
  - `a_broadcast_too_large_to_reserve_is_an_error` (shape `(2^20, 1)` ×
    `(1, 2^20)`);
  - `a_broadcast_binding_is_read_per_chunk`, asserting peak allocation
    through a counting allocator, or the chunk source kind if that is
    simpler;
  - in Python (`test_evaluate_rust_binding.py`), `p01`/`p02`'s shapes raise
    `MemoryError` and `ValueError`. These run in a subprocess, so a
    regression cannot kill the suite.
- **Behavior change:** catchable errors instead of aborts, and broadcast
  memory bounded by the chunk.
- **Revises:** none.

### R2-013a: deep patterns drop and print on a small stack (F2-013, `Pattern`)

- **Track:** B, step 3.
- **Change:**
  - `expression/pattern/matching.rs` gets an iterative `Drop` for `Pattern`
    that moves the children of the last handle out, as `Expression`'s does.
  - It gets a hand-written `Debug` with a 1,000-node budget, as
    `Expression`'s has.
  - The rustdoc keeps documenting the matcher's recursion depth (§I.10).
- **Tests:** in `tests/it/expression/pattern/stories.rs`, on a 1 MiB
  thread:
  - `a_200000_level_pattern_drops_on_a_small_stack`;
  - `a_deep_pattern_debug_is_bounded`.
- **Behavior change:** the `Debug` text of deep patterns is truncated.
- **Revises:** none.

### R2-013b: constraint `Value` decoding is depth-capped (F2-013, `Value`)

- **Track:** A, step 2.
- **Change:**
  - `constraint/wire.rs` gets a hand-written `Deserialize` for `ValueRepr`
    that counts nesting and refuses depth above 128, with "value nesting
    exceeds 128 levels".
  - The rustdoc of `Value` states that API-built nesting recurses in drop,
    compare and build, as `Provenance`'s does (R-5's precedent).
- **Tests:** in `tests/it/constraint/serde_stories.rs`:
  - `a_postcard_value_nested_200000_deep_is_refused`;
  - `a_value_nested_128_deep_round_trips`.
- **Behavior change:** absurdly deep payloads are refused.
- **Revises:** none.

### R2-013c: depth limits in the binding's readers (F2-013, binding)

- **Track:** E, after the rebase onto A.
- **Change:**
  - `fhy-core-py/src/wire.rs`'s `read_json_value` refuses depth above 128,
    with a `DeserializationValueError`, and drops iteratively.
  - `fhy-core-py/src/constraint/value.rs`'s `read_member_value` and
    `value_to_python` check depth against `sys.getrecursionlimit()` and
    raise `RecursionError`, as the provenance binding does.
- **Tests:** Python, marked `subprocess`, so a regression cannot crash the
  run:
  - `Expression.deserialize_from_dict` of a 30,000-deep list raises;
  - an `InSetConstraint` over a 20,000-deep tuple raises `RecursionError`
    (`p04`, `p40`, `p26`).
- **Behavior change:** these raise instead of segfaulting.
- **Revises:** none.

### R2-014: the process backend's timeout bounds the call (F2-014)

- **Track:** D, step 2.
- **Change**, in `solver/process.rs`:
  - the script is written from a writer thread;
  - the `Closed` arm and `reap` poll `try_wait` until the deadline, then
    kill. Without a deadline, they allow a 2 s grace after `(exit)`, then
    kill;
  - on Unix, the solver is spawned with `CommandExt::process_group(0)`, and
    the kill signals the group through a `kill -KILL -<pgid>` subprocess.
    The crate forbids `unsafe`, so `libc` is not used;
  - the rustdoc also asks that the program `exec` its solver;
  - `(set-option :print-success false)` is written first.
- **Tests:** in `tests/it/solver/process_stories.rs`, `#[cfg(unix)]`, each
  with a 200 ms timeout and elapsed time under 1.5 s:
  - `a_solver_that_stops_reading_times_out` (C);
  - `a_solver_that_closes_stdout_without_exiting_times_out` (D);
  - `a_wrappers_grandchild_is_killed_with_the_group` (E: the grandchild's
    pid file, then `kill -0` fails);
  - `a_solver_slow_to_exit_after_answering_returns_on_time` (`p30`);
  - `a_solver_that_prints_success_is_answered` (a fake that echoes
    `success`).
- **Behavior change:** these cases answer on time with
  `Unknown { reason: "timeout" }` or their answer. Nothing changes for z3 or
  cvc5 run directly.
- **Revises:** none.

### R2-015: control characters never reach a solver (F2-015)

- **Track:** D, step 1.
- **Change:**
  - `solver/smt/lower.rs`'s `sanitize` maps every `char::is_control`
    character to `_`, as well as `|` and `\`.
  - `solver/z3.rs` names constants by the sanitized text, so no NUL reaches
    `CString`.
- **Tests:**
  - in `smt_lowering_stories.rs`, `a_name_hint_with_control_characters_lowers_to_printable_text`
    (`"a\0b"`, `"a\u{7}b"`);
  - in `z3_stories.rs` (feature `z3`), `a_nul_name_hint_is_answered_not_a_panic`;
  - in Python, `p14`'s hint answers the same on both backends.
- **Behavior change:** symbol text differs only for hints with control
  characters.
- **Revises:** none.

### R2-016: negative powers lift as divisions (F2-016)

- **Track:** D, step 5, in the binding location.
- **Change** in the SymPy lifting (`solver/sympy/lift.rs`, now in the
  binding):
  - `Pow(b, -1)` lifts as `1 / b`;
  - `Pow(b, -k)` lifts as `1 / b ** k`;
  - `Mul(…, Pow(b, -1))` folds into a division node.
- **Tests:**
  - in the binding's SymPy stories, `y_over_x_simplifies_to_a_division`;
  - `simplify_then_evaluate_equals_evaluate_on_integer_grids`, a property
    over `p16`'s generator;
  - in Python (`test_sympy_pass_properties.py`), the same property.
- **Behavior change:** simplified quotients come back as divisions.
- **Revises:** none.

### R2-017: a negated literal checks as one literal (F2-017)

- **Track:** C, step 1.
- **Change:** in `types/checking/checker.rs`, a `Negate(Literal v)` node is
  inferred as the literal `-v` against `expected`, in both the unary
  `Infer` arm and the negate rule.
- **Tests:** in `tests/it/types/checking/checker_stories.rs`, the rstest
  `negated_literals_check_as_one_literal` pins the TYP probe's four rows:
  - `-(5)` against `uint8` is `Err`;
  - the literal `-5` against `uint8` is `Err`;
  - `-(128)` against `int8` is `Ok`;
  - the literal `-128` against `int8` is `Ok`.
  - The Python checker suite gets the same table.
- **Behavior change:** `-(5): uint8` is a literal-range error, and
  `-(128): int8` checks.
- **Revises:** none.

### R2-018: unification errs when a substitution is refused (F2-018)

- **Track:** C, after the rebase onto A.
- **Change:**
  - **The error.** `UnificationError::Substitution(PiecewiseError)` is
    added. `types/unify.rs:571-573` propagates it, so `bind_placeholder` and
    `Type::substitute_template` fail when a substitution is refused.
  - **The occurs check** also walks the binding graph: reachability through
    bound shape variables, iteratively. This is defense in depth.
- **Tests:** in `tests/it/types/unification_stories.rs`, over the TYP
  probe's environment `C := 5, Y := X + 1`:
  - `unifying_through_a_refused_substitution_is_an_error`;
  - `substitute_template_through_a_refused_substitution_is_an_error`;
  - the control `unify(X, Y * 2)` still fails the occurs check;
  - a `types::unify` unit test `the_occurs_check_follows_bound_variables`,
    which calls the check with an unsubstituted expression.
- **Behavior change:** these calls return an error where they now succeed
  with a wrong answer.
- **Revises:** none.

### R2-019: the body sweep checks the composed built-ins (F2-019)

- **Track:** C, step 2.
- **Change:**
  - **Labels.** `types::checking::FunctionLabel { Builtin(BuiltinFunction),
    User(FunctionName) }` is added and implements `Display`.
  - **The sweep** (`body.rs:151-168`) keys failures by label and returns a
    `BodySweep` with the checked labels in catalogue order and the failures.
  - **Signatures.** `FunctionSignature::new` returns an error on a
    parameter/sort length mismatch.
  - **Python.** `check_all_registered_function_bodies()` keeps returning a
    `ValidationReport`, now covering the built-ins.
- **Tests:**
  - in `tests/it/types/checking/body_stories.rs`,
    `the_sweep_checks_every_composed_builtin`: the checked labels equal
    `BuiltinFunction::iter().filter(composed)`, which is 16;
  - a crate-private seam test, `a_broken_builtin_body_is_reported_by_its_label`;
  - `a_signature_with_mismatched_lengths_is_refused`;
  - the vacuous test at `:215-218` is deleted;
  - in Python, the sweep's report is empty and a counting hook sees the
    built-ins.
- **Behavior change:** the sweep checks the built-ins, and its Rust return
  type changes.
- **Revises:** none.

### R2-020: decoders agree with the checked constructors (F2-020)

- **Track:** C, after R2-038b.
- **Change:**
  - **`add_symbol`** (`symbol_table/table.rs`) refuses a symbol that a
    descendant namespace defines, and `violations()` gains
    `Violation::ShadowedSymbol`.
  - **Build order.** `SymbolTableData::build` (`symbol_table/wire.rs`) adds
    every namespace first, in an order where parents precede children, and
    then every symbol.
  - **Assignments.** `param/wire.rs` decodes `ParamAssignment` through
    `restore`, using the context `Param`'s `Deserialize` builds.
- **Tests:**
  - in `tests/it/symbol_table/serde_stories.rs`, the TYP probe's two tables
    round-trip:
    - `a_table_whose_child_was_added_before_its_parent_round_trips`;
    - `a_symbol_added_to_a_child_then_its_parent_is_refused` (it is now
      refused at `add_symbol`);
  - in `table_stories.rs`, `add_symbol_refuses_a_name_a_descendant_defines`;
  - in `tests/it/param/serde_stories.rs`,
    `an_inadmissible_assignment_payload_fails_to_decode`;
  - in Python, the same three through the interface suites.
- **Behavior change:**
  - Some `add_symbol` calls that succeed today are refused.
  - Such table payloads decode.
  - Inadmissible assignment payloads fail.
- **Revises:** D-S15-11 and D-S17-9.

### R2-021: domain-level procedures enforce their own restriction (F2-021)

- **Track:** C, after R2-020.
- **Change:**
  - Every `ParamDomain` procedure folds in the domain's
    `implied_constraints` on each side it takes: `has_feasible_value`,
    `feasibility_subset`, `compute_constraint_implication_subset`, and the
    set algebra.
  - `is_value_admissible` and `is_value_set_subset` respect the restriction.
  - The docs say so.
- **Tests:** in `tests/it/param/decide_stories.rs`, the TYP table, against
  the real solver as the existing z3 stories are gated:
  - `nat.is_value_admissible(-5)` is false;
  - `nat.has_feasible_value(x <= -1)` is `Violated`;
  - `integer.feasibility_subset([], nat, [])` is not `Satisfied`;
  - `integer.is_value_set_subset(nat)` is false;
  - the `Param` path is unchanged;
  - Python interface tests for the same rows.
- **Behavior change:** the domain-level answers become correct.
- **Revises:** none.

### R2-022: extension defaults are reflexive, and equivalence is symmetric (F2-022)

- **Track:** A, step 3.
- **Change:**
  - `is_structurally_equivalent` defaults to the same identity as the
    equality hook (`eq_extension`, `eq_part` after R2-004).
  - `types/unify.rs`'s `(X, Extension(e))` arms ask `e` with the sides
    swapped, so equivalence is symmetric.
  - The rustdoc requires the equality hook to be symmetric.
- **Tests:** in `tests/it/types/extension_stories.rs`:
  - `an_extension_without_overrides_is_equivalent_to_itself`;
  - `…_unifies_with_itself`;
  - `equal_extension_types_bind_as_templates`;
  - `a_numerical_type_against_an_extension_asks_the_extension`;
  - a property: `is_structurally_equivalent` is symmetric over generated
    types with extensions.
- **Behavior change:** an extension with no overrides unifies with itself.
- **Revises:** D-S11-10.

### R2-023: interrupts outrank a kept exception (F2-023)

- **Track:** A, step 1.
- **Change:**
  - `fhy-core-py/src/constraint/value.rs`'s `record_pending_error` replaces
    a kept `Exception` with a later error that is not an `Exception`.
  - `PyOpaqueValue::is_equal`, `order_against` and the custom-constraint
    hooks check for a pending error first, and return the fallback without
    calling Python.
  - A failed lazy `ordering_key` is not cached.
- **Tests:**
  - Python (`test_constraint_rust_binding.py`), from `probe_kbi_pending.py`:
    - with members whose `__eq__` raise `ValueError`, one `__eq__` runs and
      the `ValueError` is raised;
    - with a member whose first `__eq__` raises `KeyboardInterrupt`,
      `KeyboardInterrupt` is raised;
    - a member whose key fails once, then succeeds, orders by its real key
      on the next build.
  - A binding unit test (`#[cfg(test)]`, embedded interpreter) of the
    slot's replace rule.
- **Behavior change:**
  - `KeyboardInterrupt` and `SystemExit` always win.
  - No further member `==` runs after the first exception in a call.
- **Revises:** D-S13-18 (the slot's rule).

### R2-024: posets and lattices pickle again (F2-024)

- **Track:** E, step 5.
- **Change:** `fhy-core-py/src/lattice.rs`'s `PyPartiallyOrderedSet` and
  `PyLattice` get:
  - `__reduce__`, returning `(type(self), (), state)`, where the state is
    the elements in insertion order plus the order pairs;
  - `__setstate__`, which replays `add_element` and `add_order` and
    restores `__dict__` for subclasses, as `SymbolTable` does.
- **Tests:** in `tests/test_lattice_rust_binding.py`, `pickle`, `copy.copy`
  and `copy.deepcopy` round trips of both classes and of a subclass with
  instance state. Insertion order, the order relation, and meet and join
  are preserved.
- **Behavior change:** restores the pre-switch behavior.
- **Revises:** a new row under S11's divergences.

### R2-025: custom-hook tests (F2-025)

- **Track:** A, step 5, before R2-004, so the pins guard the reshaping.
- **Change:** tests only.
- **Tests:**
  - **Rust.** `tests/it/param/custom_stories.rs` gets a recording
    `CustomDomain`, driven on either side through `is_value_set_subset`,
    `check_subset`, `Param::union`, `Param::intersection` and
    `is_structurally_equivalent`. The tests assert which hook ran and what
    it received, and that its error surfaces as the module's custom
    variant.
  - **Python.**
    - `tests/symbolic/param/test_domain_rust_binding.py` does the same with
      call counting.
    - `test_constraint_rust_binding.py` gets a custom constraint compared
      with `is_structurally_equivalent` and `is_alpha_equivalent_under`.
- **Behavior change:** none.
- **Revises:** none.

### R2-026a: evaluator tests (F2-026, evaluator part)

- **Track:** B, step 7.
- **Change:** tests; plus the chunk size becomes a crate-private parameter
  of `evaluate/array.rs`.
- **Tests:**
  - **Generators.** `Positive` and Boolean piecewise branches are added to
    `tree_strategies`, plus rstests for `+3`, `+2.5` and a Boolean piecewise
    in the scalar and array stories.
  - **IEEE edges.** The 13-row IEEE edge table for `real_divmod`, compared
    with `is_same_scalar` or `to_bits()` (from `kernel_probe.rs`), and an
    integer divmod law against an `i128` oracle.
  - **Chunks.** The lane property runs with chunks of 1 to 3 lanes; the
    300,003-lane check comes from `chunk_probe.rs`.
  - **Array bindings.**
    - Boolean and transposed bindings above and below the chunk size.
    - A test `ArrayKernels` that returns the wrong shape.
- **Behavior change:** none.
- **Revises:** none.

### R2-026b: checker tests (F2-026, checker part)

- **Track:** C, step 4.
- **Change:** tests only.
- **Tests:**
  - **Rstests:**
    - `+x` and `!p`;
    - `x_f32 + y_i16` in both orders.
  - **The broadened `checker_properties.rs`:**
    - it draws unsigned and float types, comparisons, connectives, unary
      operators and piecewise;
    - a law: synthesis is symmetric for commutative operations;
    - a law: the checker's result kind agrees with the evaluator's.
- **Behavior change:** none.
- **Revises:** none.

### R2-027: solver properties that cannot pass vacuously (F2-027)

- **Track:** D, step 8.
- **Change:** tests; plus a visible skip.
- **Tests** (`tests/it/solver/solver_properties.rs`):
  - **No vacuous passes.** No answer may be `Unknown(Refused(_))` for the
    safe generators. `GaveUp` is allowed only under a small budget.
  - **Wider generators.** `predicate()` draws Boolean identifiers and
    piecewise leaves, brute-forced over `p ∈ {false, true}`.
  - **Backend agreement.** Under `cfg(feature = "z3")`,
    `z3_agrees_with_the_process_backend` runs when `FHY_SMT_SOLVER` is set.
  - **A visible skip.** Without `FHY_SMT_SOLVER`, the test fails when `CI`
    is set. Otherwise it returns early, printing the reason through
    `eprintln!` under an `#[expect(clippy::print_stderr, reason)]`.
  - **SymPy stories**, in the binding:
    - lifting `Nand` and `Nor`;
    - `Eq(a, b, c)` failing with `Arity`;
    - a Boolean-condition piecewise inside a relational;
    - a hook that fails mid-walk.
  - The mutant check: `Hazard::find` refusing everything makes a property
    fail. This is recorded in the notes, not committed.
- **Behavior change:** none.
- **Revises:** none.

### R2-028: decision-rule tests in Rust (F2-028)

- **Track:** C, after R2-038c.
- **Change:** tests only.
- **Tests:**
  - **The three TYP probes, adopted:**
    - the interval hull with unbounded, natural and exclusive operands;
    - finite and integer intersection;
    - numeric feasibility and subset against z3, gated as
      `system_satisfiability_agrees_with_brute_force` is.
  - **Rstests:**
    - both bound spellings (`x <= 5` and `5 >= x`);
    - a `with_registry` story;
    - categorical subsets;
    - assignment equivalence per `Value` kind, `-0.0` against `0.0` and `1`
      against `True` in tuples included;
    - opaque membership alone, inside a tuple and inside a frozenset.
  - **Properties:**
    - the set-membership property draws opaque members;
    - rebinding an identifier keeps its position;
    - a system with one undecided leaf reports `Undecided`.
- **Behavior change:** none.
- **Revises:** none.

### R2-029a: error-text tables for `param`, `constraint` and `foreign` (F2-029)

- **Track:** A, step 11. **Needs:** R2-006.
- **Change:** tests; and loose matches tightened in the track's test files.
- **Tests:**
  - **One rstest table per error enum** of `param`, `constraint` and
    `foreign`. Each checks every variant's `to_string()` and whether
    `source()` is `Some`, and of which type, by downcast.
  - **`NaturalBound`'s** fields are checked for all 8 combinations.
  - **Tightened tests:**
    - the loose `{ .. }` matches in `param_stories.rs:116, 286` and
      `algebra_stories.rs:403` become full variants;
    - `constraint/equation_stories.rs:166` compares values.
  - **Small stories:** `param/context.rs:162-164`.
- **Behavior change:** none.
- **Revises:** none.

### R2-029b: error-text tables for `expression` (F2-029)

- **Track:** B, step 9. **Needs:** R2-010.
- **Tests:**
  - tables for `expression/evaluate/error.rs`, `expression/error.rs` and
    `expression/registry/error.rs`;
  - `fold_stories.rs:330, 400` and `registry_stories.rs:750, 782`,
    `inline_stories.rs:548` tightened;
  - rewrite blame through a grandchild (`pattern/rewrite.rs:118-126`,
    `:172-174`).

### R2-029c: error-text tables for `types` and `symbol_table` (F2-029)

- **Track:** C, after R2-046b. **Needs:** R2-018 and Track B landed (R2-010's
  `Rule` text).
- **Tests:**
  - tables for `types/error.rs`, `types/checking/error.rs` and
    `symbol_table/error.rs`;
  - `body_stories.rs:80, 178`, `extension_stories.rs:265-266, 353` and
    `unification_stories.rs:424` tightened;
  - table inequivalence in both directions (`table.rs:420, 436`);
  - `types/unify.rs:71, 283-285, 366`;
  - `types/environment.rs:46-56`.

### R2-029d: error-text tables for `solver` (F2-029)

- **Track:** D, step 9. **Needs:** R2-005a.
- **Tests:**
  - tables for `solver/error.rs` and the SymPy errors (binding tests);
  - `process.rs:193-199`, `:308-332` and `:318-321`: an `unknown` whose
    reason line cannot be read;
  - `sympy_simplify_stories.rs:115-124` asserts the simplified form.
- **Note:** B's R2-010 rewords four `LoweringError` texts after D lands;
  B updates these pins on its rebase.

### R2-030: binding and interface-suite gaps; member-level stub checks (F2-030)

- **Track:** E, step 9.
- **Change:** tests; and the two stub entries fixed:
  - `_rs.ValueDomain.from_json` is removed from the base;
  - `rebuild_with_visit_children` moves to the node classes.
- **Tests:**
  - `type_bindings` tested from Rust (`types/environment.rs:111`) and
    Python;
  - a dispatcher handler returning the wrong kind; a `Foreign` payload of
    the wrong kind; a Python extension type through `substitute_template`;
  - each listed `Param` and `Solver` method named in the interface suites,
    with its object identity and `KeyboardInterrupt` behavior;
  - `tests/test_rs_stub.py` compares each class's non-dunder members in
    both directions and their descriptor kinds, with an allowlist for
    PyO3's dunders.
- **Note:** every later item that adds a member updates the stub; the test
  enforces it.

### R2-031: one guard for the thread-local stacks (F2-031)

- **Track:** E, after the last rebase.
- **Change:** S-5, used by all six stacks. The three hand-written guards
  are deleted.
- **Tests:** a binding unit test per stack that panics inside a scope under
  `catch_unwind` and finds the stack empty afterwards.
- **Behavior change:** none, except after a panic.
- **Revises:** CONTRIBUTING's global-state paragraph (one sentence per
  stack becomes one on `ScopedStack`).

### R2-032a: std traits for constraint and param values; template widths (F2-032, equality part)

- **Track:** A, step 10. **Needs:** R2-004.
- **Change:**
  - **Std traits.** `PartialEq`, `Eq` and `Hash` are implemented as
    structural equivalence, with `Part`s going through `eq_part`/`hash_part`.
    `Display` is added. The types: `Constraint`, `EquationConstraint`,
    `SetConstraint`, `ConstraintSystem`, `ParamDomain`, `Param`,
    `ParamAssignment`, `Value` and `Binding`.
  - **Docs.** The `types` module docs explain the two equalities of `Type`,
    `DataType` and `SymbolFrame`: `==` against `is_structurally_equivalent`.
  - **Template widths.** `TemplateDataType` sorts and deduplicates its
    widths, and refuses an empty list with the error its constructor raises
    for an invalid width.
- **Tests:**
  - `assert_eq!` over each type;
  - `HashSet` membership agreeing with equivalence;
  - `template_widths_compare_as_a_set`;
  - `an_empty_width_list_is_refused`;
  - in Python, `TemplateDataType(t, [16, 8]) == TemplateDataType(t, [8, 16])`,
    and an empty list raises.
- **Behavior change:** additive, except the width normalization and the
  empty-list refusal.
- **Revises:** T-2's row notes the normalized widths in `repr`.

### R2-032b: `Ord` for `Constraint` from the canonical key (F2-032, order part)

- **Track:** B, after R2-042.
- **Change:**
  - `impl Ord` and `impl PartialOrd` for `Constraint` compare the cached
    keys (J-8).
  - `ConstraintSystem::new` sorts with them.
  - The rustdoc states the `ordering_key` contract of J-7.
- **Tests:** in `constraint_properties.rs`, `constraint_order_is_total_and_agrees_with_equivalence`,
  over generated built-in constraints, and a `BTreeSet<Constraint>` story.
- **Behavior change:** additive.
- **Revises:** none.

### R2-033: binding boilerplate consolidated (F2-033)

- **Track:** E, last item.
- **Change:**
  - **Imports and exceptions.** `fhy-core-py/src/python.rs` gets an
    `ImportedAttr`/`cached_attr!` helper, replacing the 15 local import
    helpers. `exceptions.rs` gets one constructor per exception class plus
    `unbox_py_err`, merging the duplicate caches of `serialization.rs` and
    `wire.rs`.
  - **Object tables.** One `ObjectTable` is built on S-5, replacing the
    seven node-to-object tables.
  - **The frozen protocol** is generated by a `frozen_protocol!` macro
    (call), replacing the 31 hand-written copies.
  - **Seeds.** One `Seed<T>`, taken once, raises on reuse. It replaces the
    three conventions.
  - **Converters.** CONTRIBUTING "Binding crate layout" allows converters
    that take context, as `fn …_to_python(…, context)`. Context-free
    conversions stay `IntoPyErr`.
- **Tests:**
  - the suite passes unchanged;
  - a Python test that reusing an environment seed raises;
  - the benchmarks of §I.8.3 within 10%.
- **Behavior change:** a reused environment seed raises instead of building
  an empty environment.
- **Revises:** CONTRIBUTING "Binding crate layout".

### R2-034: NaN-propagating composed built-ins (F2-034)

- **Track:** B, step 4.
- **Change** in `expression/builtins.rs:578-606`'s bodies:
  - `max(a, b) = a if (a > b or a != a) else b`;
  - `min(a, b) = a if (a < b or a != a) else b`;
  - `abs(x) = x if x > 0 else -x`.
  - `clamp`, `clamp_symmetric`, `relu` and `leaky_relu` follow through
    `max`/`min`, or their own comparisons, rewritten the same way where
    they compare directly.
  - `sign` is unchanged (§I.10).
- **Tests:** in `tests/it/expression/builtins_stories.rs`, the rstest
  `composed_builtins_propagate_nan`:
  - `p6`'s table, with the new expectations: `max(nan, 1)` and
    `max(1, nan)` are NaN, `relu(nan)` is NaN and `abs(-0.0)` is `0.0`;
  - `sign(nan)` is `0`, pinned;
  - a property that `max` and `min` are commutative over floats with NaN;
  - the Z3 and SymPy lowering tests' expected texts updated;
  - Python's builtin differential tests updated.
- **Behavior change:** as the resolution lists.
- **Revises:** none.

### R2-035: capture renaming restricted and deduplicated (F2-035)

- **Track:** A, step 4.
- **Change:** `term/binder.rs`'s `substitute_avoiding_capture`:
  - restricts the active keys to the binder's free identifiers;
  - renames each distinct capturable bound identifier once, iterating over
    the current binder's identifiers.
- **Tests:** in `tests/it/term/binder_stories.rs`:
  - `a_substitution_that_replaces_nothing_inside_returns_the_same_handle`
    (`p8`), with no fresh id;
  - `a_repeated_binder_is_renamed_once` (`p8b`), where the hook sees only
    identifiers the binder binds;
  - a property that results stay alpha-equivalent to the old algorithm's.
- **Behavior change:** substitutions that replace nothing inside a binder
  return the same handle and allocate no ids.
- **Revises:** none.

### R2-036: canonical float and decimal text, and the D-7 revision (F2-036)

- **Track:** B, the wire group (J-4, J-6).
- **Change:** S-2.
- **Tests:**
  - in `tests/it/expression/literal_stories.rs`, the rstest
    `float_text_is_canonical`: `1e300` → `1e300`, `5e-324` → `5e-324`,
    `f64::MAX`, `1e16`, `1e-5` (`0.00001`), `9.9e-6`, `0`, `-0`, `NaN`,
    `inf`, `-inf`;
  - `non_canonical_float_text_is_refused`: `p7`'s `Infinity`, `+inf`,
    `1e5`, `+1.5`, `.5`, `5.`, `1E-2`, `nan`, `-NaN`, `00.10`;
  - `non_canonical_decimal_text_is_refused`;
  - a property: every `f64` round-trips through the text, and the text is a
    fixed point;
  - constraint member floats pinned in `tests/it/constraint/serde_stories.rs`;
  - the Python `str()` of extreme literals pinned;
  - the corpus regenerated.
- **Behavior change:**
  - Non-canonical payloads are refused; no writer produces them.
  - Extreme floats get short text on the wire, in `Display`, in keys and in
    messages.
- **Revises:** D-7 and R-16 (`rust-workspace.md`); CONTRIBUTING's
  "shortest round-trip float formatting" sentence gains the exponent rule.

### R2-037: exact-size children and pass-through `+` (F2-037)

- **Track:** B, step 5.
- **Change:**
  - `Children` (`expression/node.rs:31-92`) implements `size_hint`,
    `ExactSizeIterator` and `FusedIterator`.
  - The walks use `len()`, and the private `count_children` is deleted.
  - Unary `+` passes its operand's lanes through.
  - `reserve_failures` stores the failing node once.
- **Tests:**
  - `node_stories.rs::children_report_their_exact_length`;
  - an evaluator story that `+x` over an array returns without copying,
    asserting pointer equality of the lane buffer where observable;
  - the existing suites.
- **Behavior change:** none; `ExactSizeIterator` is additive.
- **Revises:** none.

### R2-038a: SymPy lifting and substitution memoized (F2-038, SymPy part)

- **Track:** D, step 6, in the binding location.
- **Change:**
  - `Lifter::lift` and `rebuild_bottom_up` memoize by `id(object)`, keeping
    each object alive in the memo for the call.
  - Lifted results share.
- **Tests:**
  - `lifting_a_depth_16_sin_cos_dag_is_linear`: the S1 probe runs in under
    100 ms in release mode, and the result shares;
  - `test_sympy.py` benchmarks.
- **Behavior change:** SymPy results are shared.
- **Revises:** none.

### R2-038b: shape substitution memoized and cycle-marked (F2-038, shapes)

- **Track:** C, after R2-018.
- **Change:**
  - `types/unify.rs`'s `substitute_avoiding` memoizes each identifier's
    substituted form within one call.
  - Cycles are detected with white, grey and black marking, iteratively.
  - `types/environment.rs`'s `with_*` stop copying the whole map: a
    persistent or `Arc`-shared map, chosen by the implementer and recorded.
- **Tests:**
  - `fibonacci_bindings_substitute_in_linear_time`: n = 64 in release, the
    TYP probe's shape;
  - `a_100000_binding_chain_substitutes_on_a_small_stack`.
- **Behavior change:** none.
- **Revises:** none.

### R2-038c: permutation feasibility without `n!` (F2-038, permutations)

- **Track:** C, after R2-021.
- **Change:** `param/decide.rs:537-593`, `:609-617` and `:662-672`: when an
  in-set constraint is present, enumerate `in_set_candidates` filtered by
  `is_permutation`, not every permutation.
- **Tests:** `a_permutation_param_with_a_singleton_in_set_decides_at_n_10`
  (under 100 ms), and a brute-force agreement property for n ≤ 5.
- **Behavior change:** none.
- **Revises:** none.

### R2-039: a versioned, hash-checked prelude (F2-039)

- **Track:** D, step 7.
- **Change:**
  - The prelude is published as `_fhy_core_sympy_<crate version>_<hash>`,
    where the hash is a 16-hex FNV-1a of `PRELUDE_SOURCE`, computed by a
    `const fn`.
  - The module carries `__fhy_core_prelude__ = <hash>`.
  - On reuse, a module whose attribute differs is refused as
    `SympyUnavailableError::Incompatible`.
- **Tests:**
  - `a_module_under_the_old_fixed_name_is_ignored`: probe S2's floor
    `ROUND` under `_fhy_core_sympy`, and `round(3.5)` still gives 4;
  - `a_module_with_a_mismatched_hash_is_incompatible`.
- **Behavior change:** mismatched preludes fail to load instead of loading
  silently.
- **Revises:** none.

### R2-040: the mixed int/real equality hazard dropped (F2-040)

- **Track:** D, step 3.
- **Change:** `solver/screen.rs` removes hazard 5
  (`Hazard::MixedIntRealEquality`, `:62-66` and `:490-514`). The other four
  hazards, their order and the precedence are unchanged.
- **Tests:**
  - `x_int_equal_to_1_0_is_answered` (probe K): `Yes` at the implication
    that `x = 1` gives;
  - a property that the screen and lowering agree with the evaluator on
    mixed equalities;
  - the Python solver tests that pinned `None` for such questions now pin
    the answer, and the README's Solver row drops "or an int/float
    equality".
- **Behavior change:** more questions are answered instead of refused.
- **Revises:** D-S8-5 (and its follow-up bullet); the README Solver row.
- **Revised (the maintainer, 2026-09-27):** "drop the hazard except for
  set-constraint residuals" (option 1 of the Track D notes, N-D1).
  `Hazard::find` drops hazard 5, and the new public
  `Hazard::find_for_membership` checks the four hazards and then hazard 5;
  `ConstraintSystem`'s three questions screen each set member's expression
  with it (substituted, for `check_satisfiability_with_bindings`) before
  asking, once the solver answers the question's kind, and report a hazard
  as the solver's own refusal is reported. So plain questions and equation
  constraints answer mixed equalities by value, and a set member of the
  other numeric kind than its variable stays `UNDECIDED`. Extra tests: the
  set-residual stories in `constraint/system_stories.rs`, and the two
  false-proof probes in `test_tri_state_feasibility.py`.

### R2-041: diagnostics read through their report (F2-041)

- **Track:** E, step 8.
- **Change:**
  - D-S6-17's representation is kept.
  - `ValidationReport` gains `diagnostics_of(&self, record)` and
    `records()`, an iterator of `(record, diagnostics)` pairs.
  - `ValidatorRecord::diagnostics_in` uses `get(range)` and returns
    `Option<&[Diagnostic]>`. Its doc says it cannot detect a longer,
    unrelated report, and points to `records()`.
  - The binding uses the pairs.
- **Tests:**
  - `a_record_against_a_shorter_report_is_none` (probe P);
  - `records_pair_each_validator_with_its_diagnostics`;
  - the Python validation suite unchanged.
- **Behavior change:** the signature changes, and nothing panics.
- **Revises:** D-S6-17.

### R2-042: colliding keys grouped by equivalence (F2-042)

- **Track:** B, after the wire group. **Needs:** R2-001a, R2-004.
- **Change:** J-7.
  - The `ordering_key` contract on `OpaqueValue` and `CustomConstraint` is
    strengthened.
  - `ConstraintSystem::new` sorts by key, then groups the equivalent
    members of each tie run together, ordering the groups by first
    appearance.
  - `ConstraintSystem`'s alpha and structural equivalence compare each tie
    run as a multiset.
  - `constraint/key.rs:1-9` and `constraint.rs:19-20` say keys are equal
    exactly when constraints are equivalent, for conforming implementations.
- **Tests:**
  - `systems_with_colliding_opaque_keys_are_equivalent_in_either_order`
    (the TYP probe);
  - the system-order property's generator draws colliding opaque keys;
  - `Param` equivalence over such systems.
- **Behavior change:** equivalence no longer depends on the input order;
  the residual order within a tie run is documented.
- **Revises:** none.

### R2-043: `gil_used = true`, and the NumPy contract (F2-043)

- **Track:** E, step 7.
- **Change:**
  - `fhy-core-py/src/lib.rs:33` becomes
    `#[pyo3::pymodule(name = "_rs", gil_used = true)]`.
  - The decision is recorded (J-13): it holds until a free-threaded CI job
    exists.
  - The contract that NumPy inputs must not be mutated during a call is
    documented in `evaluate_expression_with_numpy`'s docstring, the stub
    and the README Expression row.
- **Tests:** in `tests/test_extension.py`,
  `test_the_extension_declares_that_it_uses_the_gil`: skipped unless
  `sysconfig.get_config_var("Py_GIL_DISABLED")`, it asserts
  `sys._is_gil_enabled()` after import.
- **Behavior change:** importing on 3.14t re-enables the GIL, with
  CPython's `RuntimeWarning`.
- **Revises:** CONTRIBUTING "One extension module per process"; S17 status
  (a pointer).

### R2-044: Python reads before borrows (F2-044)

- **Track:** E, step 6.
- **Change:** every Python read happens before a `PyRef`/`PyRefMut` borrow
  is taken:
  - in `symbol_table/table.rs`'s `add_symbol` and `add_namespace`, the
    `Entry` is built and the frame's `name` read first;
  - `lattice.rs`'s `Elements::insert` runs `contains` and `set_item` on a
    cloned `Py<PyDict>`;
  - `iter_stable` computes ranks over a cloned element list.
- **Tests:** `probe_reentrancy.py`'s three cases as interface tests that
  now succeed, plus a mutation inside the callback that sees consistent
  state.
- **Behavior change:** re-entrant reads work.
- **Revises:** none.

### R2-045: big numbers without decimal text (F2-045)

- **Track:** E, after the last rebase. **Needs:** R2-008.
- **Change:**
  - **Decimals.** `fhy-core-py/src/expression/literal.rs`'s `read_decimal`
    reads `Decimal.as_tuple()` and builds through `Decimal::from_parts`,
    refusing exponents beyond `Decimal::MAX_EXPONENT_MAGNITUDE` with a
    `ValueError` naming the bound.
  - **Ints.** `read_big_int` and `big_int_to_python` convert through
    `int.to_bytes`/`int.from_bytes` (signed, little-endian, sized from
    `bit_length`). pyo3's `num-bigint` feature targets num-bigint 0.4,
    which the core does not use.
  - **Evaluation.** `expression/evaluate/literal.rs` follows.
- **Tests:**
  - `Decimal('1e100000000')` is refused in under 10 ms;
  - `p32`'s two calls answer at once;
  - `LiteralExpression(10**5000)` works;
  - `from_json` of a 5,001-digit integer materializes;
  - a property that random ints up to 10^20000 round-trip.
- **Behavior change:** absurdly scaled `Decimal`s are refused at once, and
  ints of any size are valid literals.
- **Revises:** none.

### R2-046a: a `Value` case in the corpus (F2-046, corpus)

- **Track:** B, the wire group (J-4).
- **Change:**
  - `tests/golden/generate_serialization_cases.py` adds a `Value` case, a
    frozenset of tuples.
  - `tests/it/serialization_golden.rs:97`'s replay arm is then exercised.
- **Tests:** the replay.

### R2-046b: serde round-trip properties (F2-046, properties)

- **Track:** C, after R2-028. **Needs:** R2-020.
- **Tests:** proptest strategies that round-trip through JSON and postcard,
  and re-serialize to byte-identical JSON:
  - `ParamDomain`, all six kinds, with nested members of kind bool, tuple
    and frozenset;
  - `Type` and `DataType`, with nested templates and shapes;
  - `SymbolTable`, built from `table_properties.rs`'s operation model.

### R2-047a: dead evaluator code removed (F2-047, evaluator part)

- **Track:** B, step 8.
- **Change:**
  - `EvaluationError::NumberAsBoolean` is deleted, together with its
    mapping in the binding.
  - `arithmetic` (`walk.rs:613-628`) takes integer slices, and the dead
    arms at `:493-494`, `:497`, `:512-513`, `:644` and `:700` go.
  - F2-047's coverage note is taken as §I.10 says.
- **Tests:** the existing suites; a test that `!x`, `all(x, p)` and
  `piecewise(x -> 1, 0)` report `IllTyped`.
- **Behavior change:** a public error variant disappears; this is
  source-breaking only.

### R2-047b: the checker's impossible arms (F2-047, checker part)

- **Track:** C, step 5.
- **Change:**
  - A `const` assertion over `CoreDataType` proves that every integral
    width has a float.
  - `checker.rs:415-420`, `:894-901`, `:1129-1137` and `:1170-1178` then
    use `expect` with that reason.
- **Tests:** the existing suites.
- **Behavior change:** none.

### R2-N1: V2 decoding of literal-heavy trees (S17 row 2.10)

- **Track:** B, last item. **Needs:** R2-011.
- **Change:**
  - The binding builds each literal node's public object directly from the
    core's decoded `LiteralValue`, through S11's seed-based `__new__`,
    without the public constructor's re-parsing. This applies in
    `fhy-core-py/src/expression/{node,materialize,payload}.rs`.
  - Identifier objects are reused per id, as the materializer does.
- **Tests:**
  - the suite;
  - a Python test that a decoded literal of each kind (NaN, big int,
    decimal) equals, hashes and prints as the constructed one;
  - the benchmark target of J-12.
- **Behavior change:** none.
- **Revises:** S17 benchmarks (the after row).

### R2-N2: V1 removal targets 0.3.0

- **Track:** E, step 3.
- **Change:**
  - `src/fhy_core/serialization.py`: `_V1_REMOVAL = "0.3.0"`, and the
    module docstring (`:34-40`) reads "Reading and writing V1 are removed in
    0.3.0".
  - `src/fhy_core/serialization_upgrade.py`'s docstring reads "its reader is
    removed in 0.3.0. Convert stored payloads before then".
  - The README Serializable row reads "until 0.3.0 removes it".
  - `python-switch.md` gets revision bullets under D-S17-16, N-S17-3 and S17
    status (J-11).
- **Tests:** `tests/serialization/test_wire_v1.py` asserts both warnings
  with `match="removed in 0.3.0"`.
- **Behavior change:** warning text only.
- **Revises:** D-S17-16, N-S17-3, S17 status; README.

### R2-N3: `Alternatives` is committed choice, as in Python

- **Track:** B, step 1.
- **Finding.** At `a9ef7b5`, the parent of S5's switch commit `31bdf7e`,
  `src/fhy_core/symbolic/expression/pattern/core.py` behaves as the Rust
  matcher does:
  - `AlternativesPattern.match_under` returns the first alternative that
    matches ("the bindings from the first successful sub-pattern are
    returned");
  - `BinaryExpressionPattern.match_under` matches the right operand under
    that one result, and never retries another alternative.
  - So Python did not backtrack, and the resolution says to fix only if it
    did. **No code change.**
- **Tests:**
  - in `tests/it/expression/pattern/stories.rs`,
    `alternatives_commit_to_the_first_match`:
    `Binary(Add, Alt[Capture(c, _), _], Capture(c))` does not match `1 + 2`,
    and matches `2 + 2`;
  - the same in `tests/symbolic/expression/pattern/test_pattern_rust_binding.py`.
  - The existing rustdoc at `matching.rs:486-491` already documents it.
- **Revises:** none.

### R2-N4: the V1 `DeprecationWarning`s asserted or filtered

- **Track:** E, step 2.
- **Change:**
  - List the 64 warnings with a pass of `pytest tests -W error:"Reading the
    V1":DeprecationWarning -W error:"Writing the V1":DeprecationWarning`.
    The files include `tests/test_provenance.py`, `test_value_domain.py`,
    `tests/types/test_serialization.py` and the `*_rust_binding.py` suites.
  - For each test that reads V1 on purpose, choose one:
    - wrap the read in `pytest.warns(DeprecationWarning, match="V1 wire
      format")`, where the warning is the point;
    - or use `tests/v1.py`'s `reading_v1()`/`writing_v1()`;
    - or add a module `pytestmark = pytest.mark.filterwarnings("ignore:.*V1
      wire format.*:DeprecationWarning")`.
  - `pyproject.toml` gains
    `filterwarnings = ["error:.*V1 wire format.*:DeprecationWarning"]`, so an
    unmarked V1 read fails.
- **Tests:** the suite passes, with no V1 warning in its summary.
- **Revises:** none.

### R2-N5: the xdist stall at 99%

- **Track:** E, step 1, so its result informs every track's Python gate.
- **Procedure:**
  1. **A normal build.** In E's worktree, run `uv sync`, then:
     - `uv run pytest tests` (`-n auto`) to completion;
     - then with `-n 16`;
     - then with `-n 0`.
     Each run uses `-o faulthandler_timeout=900`. Record wall time and
     counts.
  2. **Account for the tests.** Compare
     `pytest tests --collect-only -q` (with the default `-m 'not slow'`)
     with passed, xfailed and skipped. The audit counted 8,224 results
     against 8,281 recorded passes.
  3. **If a stall reproduces:**
     - dump every worker with `py-spy dump`, installed in a scratch venv
       under `target/`, never the project's, and `cat
       /proc/<pid>/task/*/stack`;
     - find the blocking frame. Candidates:
       - a finalizer taking a `Mutex` the GIL holder waits on;
       - a reader thread of `SmtLib2Process` (F2-014);
       - a worker's atexit.
     - Fix it under this item, with a regression test, or record it as a
       finding for the maintainer if it is outside the fix scope.
  4. **If it does not reproduce** on a normal build, record that it is an
     artifact of the instrumented build, most likely the LLVM profile
     runtime's exit-time writes from 16 processes. Also record the coverage
     recipe (`LLVM_PROFILE_FILE` with `%p`, or `-n 0`) in the
     implementation notes.
- **Deliverable:** a paragraph in Track E's notes with the numbers, and the
  fix commit if any.

---

## Implementation notes

Each track appends its notes under its own heading only: deviations,
calls made, Python-visible changes (§I.2 rule 6), counts and benchmark
tables.

### Track A notes

**A0: the worktree and the baseline.** The worktree is
`~/Projects/FhY-core-worktrees/fix-a-api`, on branch `fix/a-api`, from
`dev-rust` at `111df20` (the names differ from §I.2 rule 7's; the
maintainer created them). Its `.venv` is its own (`uv sync --group dev
--group bench`), and its `target/gate-env.sh` points `CARGO_TARGET_DIR` at
`target/gate-cargo`. The baseline at `111df20` matches §I.8.2's:

| Gate | Result at `111df20` |
|---|---|
| `cargo test --workspace` | 4,305 passed, 2 ignored |
| `cargo test --workspace --all-features` | 4,337 passed, 2 ignored |
| fmt; clippy `-D warnings`, both ways and per feature | clean |
| `cargo doc --workspace` `-D warnings` | clean |
| `cargo doc -p fhy-core` alone, `--features z3`, `--features sympy` | the known `Prepared::evaluate_array` link only |
| `cargo deny check`; `cargo package` | clean |
| `cargo +1.85 check`, workspace lib and per feature | the 6 known `dead_code` warnings only |
| `pytest tests` | 8,281 passed, 2 xfailed |
| `pytest tests -m "not very_slow"` | 8,314 passed, 2 xfailed |
| nox `property` | 282 passed |
| nox `tests_minimal` | 6,315 passed, 642 skipped |
| nox `lint`, `type_check`, `golden_expanded` | green |

A clean copy of the base (`git archive 111df20` into `target/base-src`,
with its own `.venv`) serves the baseline's Python gates, the benchmarks'
"before" runs, and the check that each bug's new tests fail at the base.

**R2-023.**
- **Where the pending check sits.** An opaque value's `is_equal` and
  `order_against`, its lazy `ordering_key`, and a Python-defined
  constraint's `free_identifiers`, `is_structurally_equivalent` and
  `is_alpha_equivalent_under` answer their fallback without calling Python
  once an exception is pending. So does a Python-defined domain's
  `is_structurally_equivalent`, the one other hook that answers a fallback
  and keeps its exception. The observers still log: their records are not
  comparisons, and an exception they raise still takes the slot's rule.
- **The lazy key.** No Python path keys one adapter twice: every read of a
  Python value builds a fresh `PyOpaqueValue`, and a stored value's key is
  computed when the member is built. So the "fails once, then succeeds"
  pin is a binding unit test of the key cell (`cached_key`), not a Python
  test. R2-004 then makes the key fallible and computes it once, in
  `Member::try_from` (J-2).
- **Binding unit tests.** `fhy-core-py` gains its first `#[cfg(test)]`
  module (`constraint/value.rs`), which embeds the gate Python through
  `Python::initialize` and needs no `fhy_core` import. It runs under
  `cargo test --workspace`, with the gate environment.
- **Python-visible changes:**

  | Before | After | Tests |
  |---|---|---|
  | after a member's `==` raised, every later member's `==` still ran in that call | none runs; the first exception is raised | `test_no_member_equality_runs_after_the_first_exception` |
  | a `KeyboardInterrupt` or `SystemExit` raised after a kept `Exception` was lost | it replaces the kept `Exception` and is raised | binding unit test `the_first_exception_is_kept_until_an_interrupt_replaces_it` |
  | a `KeyboardInterrupt` from the first member's `==` was raised, and later members' `==` still ran | it is raised, and nothing runs after it | `test_a_keyboard_interrupt_from_the_first_member_is_raised_alone` |

**R2-013b.**
- **Ownership.** `constraint/wire.rs` is Track B's (§I.7.1), but R2-013b
  names it. The edit is additive: a hand-written `Deserialize` for the
  private `ValueRepr` (a seed that counts nesting), and the public
  `wire::MAX_VALUE_DEPTH = 128`. B's R2-036 changes the float text inside
  the same decoder through `float_text`, which the seed still calls.
- **The depth (call).** Depth counts the tuples and sets around a value, so
  128 nested tuples around a scalar decode, and 129 are refused with "value
  nesting exceeds 128 levels". The cap holds for `Value`, `Member` and
  `ValueData`, in every format.
- **Python-visible:** a V2 member or value payload nested more than 128
  deep raises the deserialization error instead of aborting. JSON text
  that deep already hit serde_json's own limit, so only a payload decoded
  from a Python dict reaches the new text; R2-013c's reader limit sits in
  front of it.

**R2-022.**
- **The default** is `eq_extension` against another extension and `false`
  against a built-in part, so it follows an overridden `eq_extension`, as
  S-3's table has it for `eq_part` after R2-004.
- **Symmetry.** `Type::is_structurally_equivalent` and
  `DataType::is_structurally_equivalent` ask a right-hand extension about
  the left side when the left side is built in; two extensions still ask
  the left one, whose rustdoc now requires a symmetric answer. The
  property draws numerical types over primitives, bare extensions and
  aliases, bare and tagged types, and aliases of numerical types. An
  alias of a bare extension is left out: the bare one, knowing nothing of
  aliases, cannot answer symmetrically, which is the implementor's
  contract, not the core's.
- **Rewritten pins.** `an_extension_without_rules_takes_the_default_rules`
  asserted `!first.is_structurally_equivalent(&first)`, and
  `a_numerical_type_over_a_data_type_extension_binds_it_by_the_default_rule`
  expected a type to fail to bind itself. Both now use two distinct bare
  extensions.
- **The binding** keeps Python's dispatcher default: a Python-defined type
  or data type with no `is_structurally_equivalent` handler is equivalent
  to nothing, itself included, as before the switch. So only Rust
  implementors see the reflexive default.
- **Python-visible changes:**

  | Before | After | Tests |
  |---|---|---|
  | `is_structurally_equivalent(built_in, python_defined)` answered `False` without calling Python | the Python-defined operand's handler answers, with the operands swapped | `test_a_built_in_type_against_a_python_defined_one_asks_its_handler` |
- **Fixed forward:** `83c1caa` (R2-023) left one test docstring 89
  characters long, which ruff's E501 refuses; this commit shortens it.

**R2-035.**
- **Hook order kept.** The binder reads its bound identifiers first, and
  its scoped children only when a key is not shadowed, as before, so a
  Python binder's hooks run in the same order; the children's free
  identifiers are then read to keep only the keys that occur. An empty
  substitution returns at once, reading no hook.
- **Renaming once.** The loop walks the binder as renamed so far, by
  position, and renames each distinct capturable identifier once, so the
  hook is never asked about an identifier the binder no longer binds.
- **The "no fresh id" pin** is a recording binder that sees no rename: a
  global id count would race with the other tests' threads.
- **Python-visible changes:**

  | Before | After | Tests |
  |---|---|---|
  | a substitution whose key the binder does not mention, but whose value mentions a bound identifier, renamed the binder and rebuilt it | the binder is returned as itself, and only `get_bound_identifiers` and `get_scoped_children` run | `test_binder_substitution_of_a_key_it_does_not_mention_renames_nothing` |

**R2-025.** Tests only; every new test passed at its first run, so no new
finding.
- **Rust** (`tests/it/param/custom_stories.rs`): a `RecordingDomain` that
  records each hook with what it received. On the left, each set procedure
  reaches its own hook once, with the param's side, the other domain and
  the result's variable. The intersection first asks `interval_profile`,
  since interval operands coerce before intersecting. On the right, a
  built-in left side asks only `symbol_type` (subset) and
  `interval_profile` (intersection). A failing hook surfaces as
  `ParamError::Custom` carrying the implementor's error.
- **Python:** `test_domain_rust_binding.py` drives a counting
  Python-defined domain the same way, including an exception and a
  `KeyboardInterrupt` raised as themselves. `test_constraint_rust_binding.py`
  records a Python constraint's `is_structurally_equivalent` and
  `is_alpha_equivalent_under`, and through a param, that the renaming it
  receives pairs the two variables.

**R2-007**, in three commits: `0b44aa2` (sub-step 1, `BoxError`),
`450fa2c` (sub-step 2, the `error` module), and this one (sub-steps 3 to
6).
- **`BoxError`** is `fhy_core::foreign::BoxError`. The five aliases are
  gone, not deprecated; the crate is unpublished. The rename reaches files
  of other tracks (`solver/{process,z3,error,sympy}.rs`,
  `expression/{evaluate,pattern}/*`, `pass/*`), one type name each.
- **The `error` module (call):** it holds `UnknownNameError` and two
  crate-private macros. `impl_name_text` (Display and a serde-based
  `FromStr`) moved from `expression/operation.rs`; `impl_from_name`, new,
  derives only `FromStr`, from a listed variant set and a text method, for
  enums that keep their own `Display`. `QueryKind` parses its `as_str`
  name, `universal_validity`, not its display words. `Logic`'s invocation
  sits in Track D's `solver/smt.rs`, one additive block.
- **`Sync` lookups.** `Environment`, `SymbolTypes` and `SortLookup` are
  `: Sync`, so the map impls need a `Sync` hasher and the closure impl a
  `Sync` closure. That bound reached `Solver::simplify`, `Prepared::evaluate`
  and `evaluate_array` (Track B's files; a `+ Sync` each). A test lookup
  that counted with a `Cell` counts with an `AtomicUsize`. `IdentifierTypes`
  and `CallTargets` document that they need not be `Sync`.
- **Contexts.** `ConstraintEvent`, `ConstraintObserver` and
  `NoConstraintObserver` (renamed for symmetry with `NoParamObserver`, a
  call). `ParamContext` holds a `ConstraintContext` and lends it through
  the public `constraint_context()`; the crate-private
  `constraint_context_with(observer)` (in Track C's `decide.rs`, a rename
  of the call) adds a forwarding observer.
- **Constructors (call, recorded):** `Sign { Any, NonNegative }`,
  `ZeroInclusion { Included, Excluded }` and `Inclusivity { Inclusive,
  Exclusive }`, each with a `const` constructor from a `bool`
  (`non_negative_if`, `included_if`, `inclusive_if`) for callers that hold
  one. `ZeroInclusion`, not `Zero`, since `num_traits::Zero` is imported
  where the domains are built. `IntegerDomain::new(Sign, ZeroInclusion)`,
  `IntervalIntegerDomain::new(Inclusivity, Sign, ZeroInclusion)` and
  `check_bounds_are_ordered(.., Inclusivity, Inclusivity)`; the getters
  still answer `bool`, and the wire keeps its Boolean fields.
- **`checked_*`.** `NotAnIntervalOperand` is now also the "neither is an
  operand" answer of `checked_{add,sub,mul,reverse_sub}`. The binding maps
  that variant, from those four, to `NotImplemented`. The one in-body
  `NotAnIntervalOperand` (a coerced operand without a profile) cannot
  arise, since coercion yields an interval operand.
- **Python-visible changes:** none. `test_non_interval_params_return_not_implemented_from_each_operator`
  pins the `NotImplemented` answers.

**R2-004.**
- **`Part::new` (call).** One generic `Part::<T>::new(part: impl
  IntoPart<T>)`, where the crate-private `impl_part!` implements the public
  `IntoPart<dyn Trait>` for every implementor of each extension trait. An
  inherent `new` per `Part<dyn Trait>` was tried first: with five of them,
  `Part::new(x)` is ambiguous (E0034) and every caller would have to name
  the trait object.
- **No `Deref` on `Part`.** `part.get()` reaches the part, so
  `part.as_any()` does not compile and cannot return the handle's own
  `Any` (S-3's hazard); `as_any_on_a_part_downcasts_to_the_implementation`
  pins both.
- **The `this` argument (call).** The binding hooks of `TypeExtension` and
  `DataTypeExtension` (`bind_template`, `substitute_template`, `unify`)
  take `this: &Type` (or `&DataType`), the value the extension is the part
  of, so the provided bodies can call `types::default_bind_template`,
  `default_substitute_template` and `default_unify` (and
  `default_bind_data_template`, `default_substitute_data_template`), which
  answer with the very handle; a hook sees only `&self`, not its `Arc`.
- **Fallible structural equivalence.** `Type::is_structurally_equivalent`
  and `DataType::is_structurally_equivalent` return
  `Result<bool, UnificationError>`, the extension's failure as
  `UnificationError::Extension`. That reached, additively in other tracks'
  files, `TypeUnificationEnvironment::is_structurally_equivalent`,
  `VariableFrame`, `FunctionFrame`, `SymbolFrame` and
  `SymbolTable<SymbolFrame>::is_structurally_equivalent` (all now
  `Result<bool, UnificationError>`), and the checker's two index-type
  comparisons, which compare with `==`, the same relation for index types.
- **Other ripples.** `Member::try_from` is fallible on a key
  (`MemberError::OrderingKey`, which drops `MemberError`'s `Clone`,
  `PartialEq` and `Eq`); `Constraint::free_identifiers` and
  `Constraint::ordering_key` return `Result`; `ConstraintSystem::new`
  returns `Result` (J-2), and the param procedures that build systems map
  it to `ParamError::Constraint` (Track C's `screen.rs`, `decide.rs`, a `?`
  each); ordinal sorting takes a fallible comparison;
  `ParamDomain::is_value_set_subset` and `Param::is_value_set_subset` take
  the `ParamContext` the hook needs; `is_mapping_alpha_equivalent_under`
  returns the value comparison's error. A binder over expressions rebuilds
  with `PiecewiseError`, which gains `From<Infallible>`.
- **Equality of custom parts.** `ParamDomain` and `Constraint` equivalence
  ask a custom part only against another custom part, through `==` on the
  `Part` (same part first, then `eq_part`); a custom domain against a
  built-in one is not equivalent, without a call.
- **The binding.** Its adapters implement the new traits, and a raised
  exception becomes the hook's `Err`, boxing the `PyErr`, which the entry
  function's error mapping unboxes. The side channels:
  - `constraint/value.rs`'s slot now serves the hooks behind `==` only
    (`PyOpaqueValue::eq_part`, a Python constraint's and a Python domain's
    `is_structurally_equivalent`);
  - `types/adapter.rs`'s context keeps only `==` and `hash` exceptions;
  - the term adapter's comparisons and scopes return their exceptions.
    Its context still keeps the exception of a binder's
    `get_bound_identifiers` and `get_scoped_children`, which the core
    reads as slices (`Binder::bound_identifiers`/`scoped_children`, not
    fallible under J-3); the next fallible hook raises it. Reading them
    eagerly would remove that, but would change the hook calls the
    Python tests pin (children are read only when the arities match).
  - **Deviation:** `wire.rs`'s slot stays for a part's serialization hook,
    not only for `==`/`hash`: `to_foreign` runs inside serde's
    `Serialize`, whose error carries only text. Recorded under D-S17-22.
  - A failing lazy key is now the hook's error, so the R2-023 unit test of
    a key skipped while an exception is pending is gone; the key cell's
    no-caching test stays.
  - A `TypeError` from an ordinal value's `<` is still chained under the
    order error, now read from `ParamError::Custom`.
- **Python-visible changes:**

  | Before | After | Tests |
  |---|---|---|
  | a Python-defined domain's `is_structurally_equivalent` was asked about a built-in domain on the right of a param's comparison | a built-in domain on the right is not equivalent, without a call | `test_a_python_defined_domain_on_the_right_is_asked_only_its_sort` |
  | serializing a bound opaque value whose key raised fell back to the value's form, the exception left in the slot | the exception is raised | none; no Python test reaches a lazily keyed member |

  The spec's expected change, a raising custom scope, was already raised
  as itself through the slot; `test_a_python_constraint_whose_scope_raises_is_refused_with_its_error`
  pins it.

**R2-006.** The five families, with each variant's producers confirmed by
`rg`. Where the table moved:

| Family | Returned by | Variants |
|---|---|---|
| `DomainError` | the three finite constructors | the table's five, and **`Custom`** (added: R2-004 makes an opaque value's key and order fallible, and they run in `OrdinalDomain::new`) |
| `ParamBuildError` | `Param::new`, `with_constraint(s)`, `with_bound`, `validate_constraint`, `check_bounds_are_ordered`, and `ParamDomain::validate_constraint` and `implied_constraints` | the table's nine |
| `AssignmentError` | `ParamAssignment::new` and `restore`, and the value checks `Param::environment`, `is_value_admissible`, `evaluate_constraints` and `check_value` | the table's six |
| `IntervalError` | `Param::checked_*` and the crate-private interval helpers | the table's five, and **`Custom`** (added: an operand's profile can fail through its custom domain). `EmptyInterval` stays a build error, carried as `Build(ParamBuildError::EmptyInterval)` from the effective interval |
| `ParamError` | the questions (`check_feasibility`, `check_subset`, `is_value_set_subset`, `symbol_type`, the domain's own questions, `compute_constraint_implication_subset`, alpha equivalence), the set algebra | the table's eleven, and **`Domain(DomainError)`** (a union's merged ordinal values can be incomparable) and **`Interval(IntervalError)`** (an intersection coerces its operands) |
| `ConstraintError` | `evaluate_constraints` and `are_all_constraints_satisfied` (moved from `ParamError`: `Constraint` was their only error) | |

- **Text.** Every message is today's; the wrapper variants (`Domain`,
  `Build`, `Interval`) write the wrapped error's text and return its source,
  as `foreign::BuildError` does.
- **Domain questions.** A domain's sort, admissibility and profile fail only
  through a custom domain; a caller of another family unwraps the boxed
  error with the crate-private `ParamError::into_custom`.
- **The binding.** `param/error.rs` flattens the families into a
  crate-private `ParamFailure`, so one mapping raises each old variant's
  Python class and text; each family also implements `IntoPyErr`. The
  `run_with_context` runner is generic over the error type.
- **Tests.** `tests/it/param/error_stories.rs` binds each public
  operation's result to its family's type (a signature that widens fails to
  compile), and the existing `expect_err` patterns name the new types. The
  Python suites pass unchanged.
- **Python-visible changes:** none.

**R2-005b.**

- **`IntervalProfile::new(sign, zero, preferred)`** takes the three named
  enums R2-007 added (`Sign`, `ZeroInclusion`, `Inclusivity`), and
  `with_only_bounds()` marks a profile of an interval-integer domain. It
  stores what it is given, with no normalization, so every profile a
  literal could build can still be built. The getters are
  `is_non_negative`, `is_zero_included`, `is_inclusive_preferred` and
  `is_bounds_only`.
- **`Value`.** The binding's one exhaustive match, `value_to_python`, gains
  the wildcard arm raising `TypeError`; nothing reaches it today.
- **Limits.** The `Solver` holds no limits of its own, so "`Solver::simplify`
  passes its configured limits" is read as `QueryContext`'s `CheckLimits`
  are: the limits ride in the `SimplifyContext` the caller builds, and
  `Solver::simplify` hands that context to the simplifier unchanged.
  `simplify_hands_its_limits_to_the_simplifier` pins that a bounded and
  an unbounded context each reach it.
- **The binding.** `Solver.simplify_expression` gains a keyword
  `timeout_milliseconds`, validated as the logical questions validate it
  (after the capability check), and builds the context's limits from it.
  The module's `simplify_expression` keeps its signature:
  `test_simplify_expression_signature_has_no_timeout_parameter` pins that
  the SymPy-backed seam function has no timeout (python-switch, S8), and
  SymPy cannot enforce one, so a caller that wants to bound a simplifier of
  its own asks a `Solver` holding it. The Python adapter records the limits it
  receives in the thread's simplification frame, and a new
  `SimplifierBase.context` property returns a `fhy_core._rs.SimplifyContext`
  with `timeout` (seconds, a `float`) and `timeout_milliseconds`, both
  `None` when unbounded, which they also are outside a simplification. A
  nested simplification has its own frame, so an inner timeout does not
  leak to the outer hook. `SimplifyContext` is reachable only through the
  property, and not re-exported, so the public paths do not change.
- **Track D's file.** `solver/sympy.rs` gains one additive rustdoc line
  saying the backend does not enforce the timeout; D re-applies it when it
  moves the file. The binding's `SympySimplifier` docstring says the same.
- **Python-visible changes:**

  | Before | After | Tests |
  |---|---|---|
  | no timeout for a simplification | `Solver.simplify_expression(..., timeout_milliseconds=)`, and `Simplifier.context.timeout`/`.timeout_milliseconds` | `test_python_simplifier_reads_the_timeout_from_its_context`, `test_python_simplifier_context_is_unbounded_outside_a_simplification`, `test_nested_simplification_has_its_own_context`, `test_simplification_refuses_a_bad_timeout` |

**R2-032a.**

- **Equality.** Each type's `==` is its `is_structurally_equivalent`, and
  `Hash` feeds what it compares: a custom constraint's or domain's `Part`
  through `eq_part`/`hash_part`, a categorical domain's values as a set, and
  a system's members in canonical order. `SetConstraint` and `Param` check
  the shared pointer first.
- **`Value`** had no structural relation of its own, only the crate-private
  type-strict `are_values_equal`, under which a NaN equals nothing. `Eq`
  must be reflexive, so `Value`'s `==` follows `LiteralValue`'s: a NaN
  equals a NaN, and `-0.0` equals `0.0`. A frozen set equals one holding
  equal values, in any order and with any repeats; its hash feeds the
  sorted, deduplicated hashes of its elements. `are_values_equal` stays
  for the Python-facing comparisons, so `ParamAssignment`'s `==` differs
  from its `is_structurally_equivalent` only for a NaN value, which the
  rustdoc says.
- **`Member` and `MemberSet`** gain `Eq`, `Hash` and `Display`: hashing a
  set constraint or a finite domain needs them. An opaque member hashes its
  ordering key, which equal members share.
- **`Display`** writes for people, as `Expression`'s does, and is not parsed
  back: values as literals write (`true`, `0.5`), a string quoted, `(1,)`,
  `{1, 2}`, a foreign part as `<TypeName>`; `x in {1, 2}`; a system's
  members joined by `and`, or `true`; `positive integer`, `ordinal (1, 2)`,
  `categorical {"a", "b"}`; `x: integer where ...`; `x = 3`.
- **Widths.** `TemplateWidthError` gains a private field and
  `is_empty_list()`. An empty list reads `template data type widths must
  not be empty`, and the zero text is unchanged. The V2 decoder goes through
  `with_widths`, so it normalizes and refuses too. The binding's V1 decoder
  refuses an empty list with `DeserializationValueError`, as it does a zero.
  `unification_stories.rs` loses the empty list's binding case, which can no
  longer be built.
- **Docs.** The `types` module docs have a "Two equalities" section, and
  T-2's row in python-switch is revised.
- **Python-visible changes:**

  | Before | After | Tests |
  |---|---|---|
  | `TemplateDataType(t, [16, 8])` kept the order, and was unequal to `[8, 16]` | the widths are sorted and deduplicated: equal, one hash, `widths` and `repr` show `[8, 16]` | `test_template_widths_compare_as_a_set`, `test_template_data_type_deserialize_sorts_and_deduplicates_widths` |
  | `widths=[]` built a template that bound nothing | `ValueError`, and `DeserializationValueError` from a V1 or V2 payload | `test_an_empty_width_list_is_refused`, `test_template_data_type_refuses_empty_widths` (replacing `test_bind_data_template_empty_widths_rejects_every_concrete_actual`), `test_template_data_type_deserialize_raises_on_empty_widths`, `test_template_data_type_v2_payload_with_empty_widths_is_refused` |

**R2-029a.**

- **Tables.** `tests/it/param/error_text_stories.rs` has one rstest table
  per family (`DomainError`, `ParamBuildError`, `AssignmentError`,
  `IntervalError`, `ParamError`), `tests/it/constraint/error_text_stories.rs`
  one for `ConstraintError` and `MemberError`, and `foreign_stories.rs` one
  for `ForeignError` and `BuildError`. Each case checks `to_string()` and the
  type of `source()` by downcast, through `support/error_text.rs`. A wrapper
  variant has a case with a source and one without. Identifiers are
  restored with fixed ids, so the texts name them exactly.
- **`NaturalBound`.** The table covers the 8 combinations of side, zero
  inclusion and inclusivity, plus a story that `is_negative` changes the
  text only for a zero-including lower bound. The story that drives the
  gate (`natural_gate_refuses_a_bound_literal_the_naturals_do_not_admit`)
  now matches every field of the variant it returns.
- **Tightened.** The spec's line numbers are the audit's; the matches are
  in `param_refuses_a_constraint_outside_its_scope` (the constraint and the
  variable, now comparable through R2-032a, and the text),
  `intersection_refuses_a_set_constraint_on_another_variable` (`from`, `to`,
  `variable` and the whole text), and
  `text_outside_the_literal_grammar_is_refused_with_its_cause` (the
  identifier, the `LiteralTextError` compared with the parser's own, and the
  source downcast to it).
- **`param/context.rs:162-164`**, the default `is_undecidable`, had no
  test: every test observer overrode it. `decide_stories.rs` gains a story
  that the default (for `NoParamObserver` and for an observer that
  overrides only `notify`) counts only a backend's failure as undecidable,
  and one that a context without an observer evaluates past a failing
  simplifier.
- No test found a bug. **Python-visible changes:** none.

**Track A status: the gates at the head.** On the tree of the status
commit (the code of `dac8ab1`):

| Gate | Result | At `111df20` |
|---|---|---|
| `cargo test --workspace` | 4,481 passed, 2 ignored | 4,305 |
| `cargo test --workspace --all-features` | 4,513 passed, 2 ignored | 4,337 |
| fmt; clippy `-D warnings`, both ways and per feature | clean | clean |
| `cargo doc --workspace` `-D warnings`; the public-paths check of CI | clean | clean |
| `cargo doc -p fhy-core` alone, `--features z3`, `--features sympy` | the known `Prepared::evaluate_array` link only (`--features ndarray` clean) | the same |
| `cargo deny check`; `cargo package` | clean | clean |
| `cargo +1.85 check`, workspace lib and per feature | the known `dead_code` warnings only, as many as at the base | the same |
| `pytest tests` | 8,305 passed, 2 xfailed | 8,281 |
| `pytest tests -m "not very_slow"` | 8,338 passed, 2 xfailed | 8,314 |
| nox `property` | 282 passed | 282 |
| nox `tests_minimal` | 6,339 passed, 642 skipped | 6,315 |
| nox `lint`, `type_check`, `golden_expanded` | green | green |
| `pytest -p no:xdist` of the constraint, domain, param, checking, types and term `*_rust_binding.py` suites | 264 passed | |
| attribution grep of the commits and the diff | nothing | |

- **Benchmarks** (`test_constraint.py`, `test_param.py`, `test_types.py`,
  `test_term.py`; medians, the base's `target/base-src` build and the head's
  run back to back, then the rows over 10% rerun twice, alternating head and
  base). Of 144 rows the median moved by -1%. Seven rows were over 10% in
  the first run; five of them, all under 1.1 us, were within noise in the
  reruns (`test_constraint_alpha_equivalence`, both
  `test_param_alpha_equivalence` tables, `test_promote_core_data_types[integer]`,
  `test_variable_symbol_table_frame_construction`,
  `test_symbol_table_structural_equivalence`). **Flagged:**
  `test_term.py::test_binder_substitute[no_capture]` (+27% to +35%, 3.3 us
  to 4.2 us) and `[capture]` (+18% to +30%, 9.5 us to 11.3 us). That is
  R2-035's fix: to return a binder whose children do not mention a key as
  itself, and to rename only for a key that applies, the substitution now
  reads the free identifiers of the binder's children, one more round of
  hooks for a Python-defined binder. Keeping the pinned identity (`result is
  binder`) needs that walk, so it stays; a cheaper test of whether a key
  occurs would need a new hook. The maintainer accepted this cost on
  2026-09-27.
- **An unreproduced hang.** One full `pytest tests` run, right after the
  extension was rebuilt for R2-005b, stopped with every xdist worker idle on
  a futex. It was killed; six full runs since, with a per-test timeout, all
  passed in about 20 s. No test was identified.

### Track D notes

**D0, the worktree and the baseline.** The maintainer created the worktree
as `~/Projects/FhY-core-worktrees/fix-d-solver` on branch `fix/d-solver`,
from `dev-rust` at `111df20`, in place of §I.2 rule 7's
`fix2-solver`/`port/fix2-solver`; the names are the only difference. Its
`.venv` is its own (`uv sync --group dev --group bench`), and its
`target/gate-env.sh` copy points `CARGO_TARGET_DIR` into its own `target/`.
The baseline at `111df20`, which differs from `d976522` only in documents,
matches §I.8.2's: fmt and clippy (both ways) clean; `cargo test
--workspace` 4,305 passed; `pytest tests` 8,281 passed (2 xfailed). The
all-features run was the first to see R2-015's new tests, which failed as
intended (5 lowering cases and the z3 story, the latter with z3's
`CString::new(..).unwrap()` panic at `z3-0.21.1/src/symbol.rs:14`).

**R2-015.** `z3.rs` needed no change: it already names each constant by
`Symbol::name`, the sanitized `<hint>_<id>`, so sanitizing in `lower.rs`
removes the NUL for both backends. The Python test's backends are a fake
that answers `sat` only for a script of whole, printable command lines, the
process backend over the `z3` beside the interpreter when there is one, and
the z3-solver adapter when it is installed; at the base the adapter raised
`Z3Exception` for the NUL and the fake answered `unsat` for every hint.

**R2-014, calls.**
- **All input goes through the writer thread**, not only the script: the
  `(get-info :reason-unknown)` and `(exit)` writes would block the same way
  on a pipe the script filled. The thread owns the pipe and closes it when
  its queue is dropped and drained, or when a write fails.
- **A `success` line before the answer is skipped.** SMT-LIB 2.6 leaves
  open whether a solver whose `:print-success` is on answers the
  `set-option` that turns it off, so the backend tolerates one (or a solver
  that ignores the option). The fake of
  `a_solver_that_prints_success_is_answered` answers `success` to the first
  command and answers `unsat` unless that command was the `set-option`, so
  the test also pins that the option comes first.
- **A new `ProcessError::ClosedOutput` variant** (the enum is
  `#[non_exhaustive]`): without a timeout, a program that closes its output
  and does not exit within the 2 s grace is killed and reported this way;
  with a timeout, it answers `unknown` with `"timeout"` at the deadline, as
  the spec says. A program that closes its output and exits still reports
  `Exited(status)`.
- **Polling.** `try_wait` is polled with a pause doubling from 100 µs to
  10 ms, so a solver that exits at once after `(exit)` costs no fixed
  delay.
- **The group kill** runs `kill -KILL -- -<pgid>` before reaping the
  leader, so the group id cannot have been reused, and then `Child::kill`,
  which also covers a missing `kill` program. A solver whose own process
  group differs (it called `setsid`) is not reached; the rustdoc asks
  wrappers to `exec` their solver.
- **A side effect of the own process group:** a terminal's interrupt no
  longer reaches the solver, which the rustdoc says. The caller still
  bounds the check by its timeout.

**N-D1: R2-040 is held, because dropping the hazard makes the constraint
and param layers report false proofs.** R2-040 was implemented in full
(the variant and the numeric-kind classifier removed; the Rust stories,
the property and 33 Python tests rewritten to pin the answers; all gates
green) and then taken back out before committing, since the resolution's
premise holds for expressions but not for set constraints:
- **The premise.** F2-040 reasons that "the crate's evaluator equates"
  `x_int == 1.0` with `x_int == 1`, so the refusal loses nothing. That is
  true of expressions, and of equation constraints, which evaluate as
  expressions.
- **What it misses.** Membership in a set constraint is type-strict
  (`constraint/value.rs`: a Boolean, an integer and a float never compare
  equal, so `2 ∉ {2.0}`), but `SetConstraint::to_expression` lowers it to
  `x == 2.0`, which the solver decides by value. The hazard was what kept
  the two apart for every solver question, and the param layer relies on
  it: `param/decide.rs` downgrades answers resting on float members only
  over the REAL sort, with this comment's reasoning, and leaves INT to the
  screen.
- **The false proofs, with the hazard dropped** (probes kept in
  `target/logs/probe_nd2.py` and `probe_nd3.py` of this worktree):
  - `create_integer_param_between(2, 2)` with `NotInSetConstraint(v,
    {2.0})`: `is_value_valid(2)` is `True`, but `check_feasibility()` is
    `VIOLATED` and `is_empty()` is `True`. At the base it is `UNDECIDED`.
  - the integer param `x ∉ {2.0}` against the integer param `y != 2`:
    `check_subset` is `SATISFIED`, although `x = 2` is admitted by the
    one and not the other. At the base it is `UNDECIDED`.
  - `InSetConstraint(x, {3.0})` with `x` bound to `y + 1` for an INT `y`:
    `check_satisfiability_with_bindings` is `SATISFIED`, while
    `evaluate_with_bindings({x: 3})` is `VIOLATED`. At the base it is
    `UNDECIDED`, and `test_check_satisfiability_with_bindings_does_not_
    contradict_literal_violation` pins that.
  - The Rust story `param::decide_stories::integer_implication_leaves_a_
    float_member_to_the_solver_s_screen` is the one Rust test that fails;
    its question happens to be answered correctly, but its comment names
    this reliance.
- **Why this track does not decide it.** Every repair is a design choice
  in files other tracks own (`constraint/set.rs`, `constraint/system.rs`,
  `param/decide.rs`), and the choice is the maintainer's:
  1. keep the refusal for set constraints only: the numeric-kind
     classifier stays as a crate-private check, which the constraint
     system applies to the expressions its set members lower to (with
     their bindings substituted), answering `UNDECIDED` as today, while the
     solver itself, and so `fhy_core.symbolic.solver` and equation
     constraints, answer mixed equalities by value. This is the smallest
     change that implements F2-040's intent without a false proof;
  2. lower set members kind-faithfully from the variable's sort (an INT
     variable is never a float member), which works for identifiers but
     not for a binding to an expression of unknown kind, nor for the REAL
     sort, which `param` says conflates kinds;
  3. extend `param/decide.rs`'s downgrades to the INT sort and add the
     same to `ConstraintSystem`, answer by answer;
  4. keep hazard 5 as it is (revise the resolution).
- **Resolved (the maintainer, 2026-09-27): option 1.** Implemented after
  the rebase onto Track A; see "R2-040, as implemented" below. The three
  false proofs are `UNDECIDED` again (the probes re-run), and the Rust
  story above passes unchanged, renamed
  `integer_implication_leaves_a_float_member_to_the_membership_screen`.
- **The work was kept** as `target/r2-040-wip.patch` in this worktree
  (untracked), a diff against `35fae05` whose `rust-port-fixes.md` hunk
  this note supersedes, for whichever option is chosen:
  the screen change, the admission stories, `x_int_equal_to_1_0_is_answered`,
  the z3 story, the property against the evaluator, and the 33 Python
  rewrites. Under option 1 the three set-constraint rewrites in
  `test_constraint_system.py` go back to their base pins, and the param
  story above passes unchanged.

**R2-005a, the move.**
- **Layout.** The pyclass file `fhy-core-py/src/solver/sympy.rs` is the
  module root, as specified; the core's `solver/sympy.rs`, the
  `SympySimplifier` itself, became the submodule `simplifier.rs` beside
  `boolean`, `error`, `lift`, `load`, `lower`, `simplify`, `substitute`
  and `prelude.py`. The stories are `lifting_stories`, `lowering_stories`,
  `simplify_stories` and `properties`, with `test_support` holding the
  embedded interpreter, the SymPy helpers and copies of
  `build_identifier`/`build_literal`. `git mv` keeps each file's history.
- **"Every import is public" was not quite so.** Four uses needed a
  public spelling:
  - `solver::screen::is_native_constant` (`pub(super)` in the core): the
    binding has a two-line copy over `BuiltinConstant::of_identifier` and
    `SortLookup::native_constant_sort`;
  - `tree::BuildIdentityHasher` (`pub(crate)`): the lowering's memo uses
    the standard hasher (the benchmarks below measure it);
  - `num_bigint::Sign` (num-bigint is no dependency of the binding):
    `Signed::is_negative` and `Zero::is_zero` from num-traits, which the
    binding has;
  - `BuiltinConstant` is `#[non_exhaustive]` outside the core, so the
    constant match gained a wildcard arm and the error kind
    `SympyErrorKind::UnsupportedConstant`, unreachable today.
- **The API.** `load` takes `py`, since no call attaches on its own any
  more, and `Simplifier::simplify` attaches with `Python::attach`. The
  `no_run` rustdoc example went with `with_embedded_python`: the binding
  is a `cdylib`, whose docs run no doc-tests, so the doc points at the
  stories.
- **One edit in Track E's `pyproject.toml`:** `[tool.uv] cache-keys` gains
  `rust/fhy-core-py/src/**/*.py`, since the binding compiles the prelude
  in and uv would otherwise reuse a stale extension after a prelude edit
  (true of the core's copy before, and needed by R2-039).
- **Counts.** `cargo test --workspace` goes from 4,315 to 4,313: the 143
  stories now run as `fhy-core-py`'s unit tests, `sympy_unavailable` is
  gone, and so is the one `no_run` doc-test. `pytest tests` gains
  `test_missing_sympy_reports_unavailable` (8,287).
- **The gate recipe.** `cargo test -p fhy-core --all-features` passes in a
  shell whose environment is only `HOME`, a `PATH` of cargo and `/usr/bin`,
  the z3 variables and `FHY_SMT_SOLVER` (3,677 + 120 + 8 + 1 tests), and
  `cargo tree -p fhy-core --all-features` holds no pyo3. The workspace
  still needs `PYO3_PYTHON`, `PYTHONPATH` and libpython on
  `LD_LIBRARY_PATH` for the binding's tests, so the variables of
  `gate-env.sh` stay; its comment in this worktree says so, and the shared
  `target/tooling/gate-env.sh` wants the same comment. CI's `rust` job
  keeps its environment for the same reason.
- **Checks replayed:** the CI target list (`id_cap_decode it`), `cargo
  package` and the Package Contents pattern without the prelude (no `.py`
  in the list), and `cargo deny check` (ok, with the `syn` duplicate
  warning of the base).

**R2-016.**
- **The rule.** A `Pow` whose exponent is a negative SymPy `Integer`
  lifts as `1 / b` or `1 / b ** k`; in a `Mul`, every such factor goes to
  the denominator, so `Mul(2, y, Pow(x, -1), Pow(z, -2))` lifts as
  `(2 * y) / (x * z ** 2)`, each product folded to the right as before. A
  negative rational exponent is left as a power.
- **The property's domain.** p16's generator, with the grid `-3..=3` for
  `x` and `y`. A point is skipped where the original fails or is not
  finite, as p16 skipped, and also where one of its quotients divides by
  zero: there the tree passes through a NaN or an infinity, which
  `sympy.simplify`'s cancellations assume away. The first run without that
  rule found `{(y + y) if x < y / y; -1 otherwise}`, which simplifies to
  `{2 * y if x < 1; -1 otherwise}` and differs at `y = 0`, since `0 / 0`
  is NaN. That is SymPy's generic-value simplification, not the lifting,
  so it is recorded here and not treated as part of F2-016. A tree SymPy
  refuses to lift (the complex infinity of `1 / 0`) is skipped
  (`prop_assume`); floor divisors are 1, 2 or 4 so distributed quotients
  stay exact binary floats.
- **Both properties fail at the base**: the Rust one on `(-1 / x) - 0`,
  simplified to `-1 * x ** -1`, and the Python one likewise.

**R2-038a.**
- **The memos** key by the object's address and hold the object, so no
  address is reused during the call: the lifting's memo maps a SymPy node
  to its expression (a `Remember` task records a node once its parts are
  assembled), and `rebuild_bottom_up`'s maps a node to its result, which
  serves the substitution and the masking of Boolean comparisons alike. In
  the masking walk, a shared comparison now gets one placeholder instead
  of one per occurrence, which `substitute_symbols` puts back the same way.
- **Numbers** (debug build, depth 16): lifting took 5.7 s before and is
  now under the story's 100 ms bound in debug and release; the
  substitution took 3.8 s and now about 0.14 s in debug, which is SymPy's
  own construction of the 49 rebuilt nodes, so its story's bound is 1 s.
  Both results hold the input's 49 distinct nodes.

**R2-039.**
- **The name** is `_fhy_core_sympy_<version>_<hash>` with the version's
  `.`, `-` and `+` written `_` (today `_fhy_core_sympy_0_2_0_<16 hex>`),
  a call on the spec's `_<crate version>_`: a pickle of a lowered
  piecewise or `round` names the prelude module, and pickle imports a
  dotted name as a package path, so `0.2.0` would not load.
- **The hash** is the 64-bit FNV-1a of `PRELUDE_SOURCE`, a `const`
  computed by a `const fn`; its 16-digit text is formatted at run time,
  since `str::from_utf8` is not `const` at the 1.85 MSRV and the crate
  forbids the unchecked form. A published module missing the attribute,
  or holding another hash, is refused with an `ImportError` cause naming
  both, as `SympyUnavailableError::Incompatible`; the check also runs on
  the module `setdefault` returns.
- **Tests.** `a_module_under_the_old_fixed_name_is_ignored` publishes probe
  S2's impostor, whose `ROUND` is `sympy.floor`, under `_fhy_core_sympy`,
  and checks a fresh backend's `round(3.5)`. The spec expected `4`, but
  the real prelude's `round` folds only over an integer, so the call
  stays `round(3.5)`, which the story pins; the impostor gave `3` at the
  base. `a_module_with_a_mismatched_hash_is_incompatible` goes through a
  seam, `Handles::load_with_prelude(py, name)`, under a name of its own,
  since replacing the real module would race the other stories' fresh
  backends; it covers a wrong hash and a missing one. Both old-name
  stories run under `test_support::serialized`, the lock the SymPy
  patches take.
- **Python.** Two tests pinned the module name `_fhy_core_sympy`; they
  now check the versioned name, its hash attribute and that a pickle names
  it.

**R2-027.**
- **"`GaveUp` only under a small budget"** is read as a count: each
  solver-backed property may see at most `GAVE_UP_BUDGET = 4` given-up
  answers over its cases (a static counter per property), and a refused
  answer fails at once. The ground-tree property, which checks scripts
  directly, has the same budget for `unknown`.
- **The generators** add a Boolean identifier `p` (a predicate leaf) and a
  piecewise integer leaf whose condition is `p` or a comparison of two
  leaf terms; the brute force runs over `x, y ∈ [-3, 3]` and `p ∈ {false,
  true}`, and universal validity quantifies over `y` and `p`.
- **The skip.** Without `FHY_SMT_SOLVER` (and without the `z3` feature for
  the three question properties), a property panics when `CI` is set and
  otherwise prints once, "skipping the solver properties: FHY_SMT_SOLVER is
  not set", under the `#[expect(clippy::print_stderr, reason)]`. Checked
  both ways locally. CI's `rust` job sets both variables.
- **The mutant check**, not committed: with `Hazard::find` returning a
  refusal for every expression, the three question properties fail on the
  process backend, and the three and `z3_agrees_with_the_process_backend`
  fail under `--all-features`. The ground property checks scripts
  directly and is unaffected, as it should be.
- **SymPy stories.** The existing `sympy.Nand(a, b)` and `sympy.Nor(a, b)`
  cases never reached the lifting's `Nand`/`Nor` arm, since SymPy builds
  `Not(And(..))` for them; the new story uses `evaluate=False`. The
  others: `Eq(a, b, c)` (built with `Basic.__new__`) refused with `Arity`,
  a Boolean-condition piecewise inside `Lt` lifting to the comparison of
  a piecewise, and a `replace` hook that fails on its second call, which
  stops the walk and fails it with the hook's own error.

**R2-029d.**
- **Tables.** `tests/it/solver/error_stories.rs` holds one rstest table per
  error enum of `solver/error.rs` (`SolveError`, 7 variants;
  `LoweringError`, 10 cases, the four `Call` texts included), each
  checking `to_string()` and `source()` by downcast. `ProcessError`'s
  table (7 variants, `ClosedOutput` from R2-014 included) is in
  `process_stories.rs`, since `Exited(Some(_))` takes the status of a real
  `sh`. The binding's `solver/sympy/error_stories.rs` covers every
  `SympyErrorKind` (19), both `SympyUnavailableError`s and the four
  phases' names.
- **Small stories.** An `unknown` whose solver exits before the reason
  answers `unknown` with an empty reason; one whose reason line is not
  UTF-8 fails with `ProcessError::Io` of kind `InvalidData`, which pins
  that an unreadable line is an error of the talk, not a reason.
  `comparison_between_booleans_is_simplified_on_its_own` asserts the
  simplified form, `(x > -5) && ((x < 1) == true)`, where it asserted
  `is_ok()`.
- **B's R2-010** rewords four `LoweringError` texts; per §I.7.1 B updates
  these pins on its rebase.

**R2-009.**
- **docs.rs features: `["ndarray", "z3"]`.** `DOCS_RS=1 cargo doc -p
  fhy-core --no-deps --features z3,ndarray` succeeds in a shell with no
  libz3, no `z3` executable and no z3 entry for `pkg-config`: z3-sys's
  build script tolerates a failed probe, and rustdoc links nothing. The
  `--cfg docsrs` build needs a nightly rustdoc (`feature(doc_cfg)`); no
  nightly is installed here, so it was checked with `RUSTC_BOOTSTRAP=1` on
  stable, locally only: it builds with `-D warnings`, and `Z3Solver`,
  `ArrayValue` and `Prepared::evaluate_array` carry the "Available on crate
  feature" marking.
- **`doc(cfg)`** is on the gated items themselves (`Z3Solver`,
  `Z3TermError`, `ArrayBinding`, `ArrayValue`, `ArrayKernels`,
  `CoreKernels`, and the `impl Prepared` block of `evaluate_array`); their
  `pub use`s inherit it.
- **The 1.85 `dead_code` warnings** came from helpers used only inside
  `const _: () = { .. }` items, which 1.85's lint does not count. The
  `Send + Sync` assertions of `interned.rs` became a `#[test]`, and the two
  list-order checks became `#[test]`s whose bodies are inline `const`
  blocks, so they still fail at compile time; three tests more (4,378).
  The edits in `tests/it/expression/vocabulary_stories.rs` and
  `tests/it/support/expression.rs` (Track B's) are confined to those
  blocks.
- **CI.** `rust` gains a "Per-Crate Linting" step (clippy on `fhy-core`
  alone, three ways) and builds the default-feature docs of `fhy-core`
  before the workspace's; the docs step's comment now says what each
  build is. `rust-msrv` gains "Build fhy-core's Targets on the MSRV", the
  three `cargo check -p fhy-core --all-targets` runs under `RUSTFLAGS=-D
  warnings`, which needs no libz3 since `check` links nothing. All replayed
  locally, clean.
- **Drift.** The crate summary names every area and gains a "Features"
  section (`solver::Z3Solver` is code text there, since it exists only
  under `z3`); the serialization claim reads "…or its `Canonical<T>`
  does"; the description names the symbol table, stack and scope; the
  workspace `repository` drops `.git`; the README's two "docs.rs builds the
  default features" sentences say docs.rs builds each feature and marks
  its items.

**After R2-009: the SymPy memos hash by address.** The benchmarks below
first showed the lowering 5–8% slower than at the base, from the standard
hasher R2-005a put in place of the core's crate-private
`BuildIdentityHasher`, and the lifting and substitution memos of R2-038a
paying the same. A binding copy of that hasher, `sympy/address_hash.rs`,
now serves all three memos; the lowering and substitution rows are back at
parity, and the lifting keeps about 5% for its memo's bookkeeping, which
R2-038a's linear DAG lifting pays for.

**Benchmarks** (§I.8.3: `benchmarks/test_sympy.py` and `test_solver.py`),
medians in µs, the base at `111df20` and the head (with the address
hasher), each in its own venv built the same way (`uv sync
--no-default-groups --group bench --group test`, CPython 3.11) from a `git
archive` under this worktree's `target/bench/`, run back to back with
`pytest --benchmark-only -n 0`. Other tracks were building on the machine,
so single runs moved by up to 20%; the rows that crossed 10% in some run
were re-measured interleaved, base and head alternating.

| Benchmark | Base | Head | Ratio |
|---|---:|---:|---:|
| `test_check_satisfiability_of_a_conjunction_of_50_bounds` | 891.99 | 934.40 | 1.05 |
| `test_check_satisfiability_of_bounds` | 473.68 | 450.69 | 0.95 |
| `test_check_satisfiability_refused_by_the_screen` | 26.92 | 27.27 | 1.01 |
| `test_constraint_system_check_implication` | 411.59 | 431.55 | 1.05 |
| `test_does_expression_imply_of_bounds` | 406.92 | 425.72 | 1.05 |
| `test_equation_constraint_evaluate_with_bindings` | 10.51 | 10.82 | 1.03 |
| `test_first_simplification_in_a_fresh_interpreter` | 315,105.35 | 344,072.50 | 1.09 (interleaved, 15 pairs: 356.1 ms against 353.0 ms) |
| `test_holds_for_all_free_assignments_with_a_witness` | 903.59 | 968.66 | 1.07 |
| `test_import_fhy_core` | 93,096.63 | 112,112.76 | 1.20 (interleaved, 40 pairs: 112.0 ms against 97.1 ms; minimum 92.5 against 93.0 ms) |
| `test_int_param_intersection_feasibility` | 521.03 | 550.66 | 1.06 |
| `test_lift_from_sympy_of_a_deep_tree` | 67.46 | 69.38 | 1.03 (interleaved, 3 pairs: 1.04 to 1.07) |
| `test_lower_to_smtlib2_of_a_deep_tree` | 75.47 | 65.15 | 0.86 |
| `test_lower_to_sympy_of_a_deep_tree` | 154.15 | 190.03 | 1.23 before the address hasher; interleaved after it, 3 pairs: 0.94 to 1.06 |
| `test_lower_to_z3_of_a_deep_tree` | 366.51 | 400.91 | 1.09 |
| `test_nat_param_is_value_valid` | 11.25 | 11.06 | 0.98 |
| `test_screen_of_a_deep_predicate` | 77.82 | 79.25 | 1.02 |
| `test_simplify_expression_of_a_boolean_comparison` | 8,528.86 | 8,821.41 | 1.03 |
| `test_simplify_expression_of_a_bound_piecewise` | 47.05 | 48.32 | 1.03 |
| `test_simplify_expression_of_a_ground_comparison` | 10.59 | 10.70 | 1.01 |
| `test_simplify_expression_symbolic` | 24.27 | 26.25 | 1.08 |
| `test_substitute_sympy_variables_of_a_deep_tree` | 24.67 | 23.93 | 0.97 (interleaved after the hasher: 0.99 to 1.02) |
| `test_sympy_simplifier_of_a_ground_comparison` | 8.00 | 8.19 | 1.02 |

The table is the second full run, before the address hasher, except where
a row says otherwise; a third full run after it had every row but the
fresh-interpreter one (1.12, 1.01 interleaved) within 1.09. No row is
slower by more than 10% once measured interleaved.

**Where Track D stops.** At the checklist's `[rebase]` line, as the
maintainer asked: the rebase onto Track A, the track gates after it, and
the Track D status line are the maintainer's. The track gates of §I.8.2
and D's extras were run on the head before the rebase (`93fa510`, then
the address-hasher commit re-ran the per-commit gates):
- fmt; clippy `-D warnings` workspace both ways, and `fhy-core` alone
  with no features, `z3` and `ndarray`: clean;
- `cargo test --workspace`: 4,378; `--all-features`: 4,412 (the base:
  4,305 and 4,337);
- `cargo test -p fhy-core --all-features` in a shell with no Python
  environment: 4,233 (3,677 of `it` at R2-005a, now more);
- `cargo doc -D warnings`: the workspace, `fhy-core` with default
  features, and with each feature alone: clean;
- `cargo deny check`: ok; `cargo +1.85 check --workspace --lib` and
  `-p fhy-core --all-targets` three ways: no warning;
- `cargo package` and its list (no `.py`), and the CI target list;
- `pytest tests`: 8,288 (the base 8,281); `-m "not very_slow"`: 8,321
  (8,314); `property`: 283 (282); `tests_minimal`: 6,320 passed, 642
  skipped (6,315 and 642); nox `lint`, `type_check` and `golden_expanded`:
  green;
- the attribution grep over `111df20..HEAD`: no match; every commit is
  the configured user's.

**After the rebase onto Track A** (the maintainer rebased `fix/d-solver`
onto `dev-rust` at `46b8596`; the hashes above are the pre-rebase ones,
and the checklist's are the rebased ones):
- **The resolutions** the maintainer made: `solver/process.rs` keeps
  R2-014's `check` with Track A's `BoxError` (R2-007 removed
  `BackendError`); the moved `sympy/simplifier.rs` keeps `Python::attach`
  and imports `BoxError` from `fhy_core::foreign`; the core README joins
  this track's solver line with Track A's `ConstraintObserver`/
  `ConstraintEvent` renames; `rust-workspace.md` keeps both revision
  bullets.
- **Broken commits, fixed forward.** `a04e029` (R2-014, rebased) alone
  does not compile the `it` tests: its `process_stories.rs` names
  `BackendError`, which `4683c70` (R2-005a, rebased) renames to
  `BoxError`. And the rebased head `f0592b5` compiled neither the core's
  `it` tests nor the binding's: R2-029d's tables called
  `FunctionName::try_new`, which R2-007 renamed `FunctionName::new`, and
  the moved simplifier's imports were unsorted for `cargo fmt`. Both are
  fixed in `ac27b14`, which also points the simplifier's `SimplifyLimits`
  link at `fhy_core::solver`. Track A's rustdoc line that the SymPy
  backend does not enforce the context's timeout is still accurate: the
  moved `simplify` never reads the limits.

**R2-040, as implemented** (the maintainer's option 1):
- **The screen.** `Hazard::find` checks four kinds; the variant
  `MixedIntRealEquality` stays, documented as reported only by
  `Hazard::find_for_membership`, which runs `find` and then the
  numeric-kind classifier, unchanged from the base.
- **The constraint layer.** `ask` in `constraint/system.rs` takes the set
  members' expressions and screens them first; a refusal notifies
  `ConstraintEvent::Refused` exactly as a solver refusal does, so the
  binding's WARNING and the param layer's events are unchanged. Every
  solver question of `param` goes through `ConstraintSystem`, so the
  param probes are covered, the witness-outside exclusion set included.
  The one difference in what is reported: a question whose set member is
  refused by the membership screen and whose equation holds another hazard
  reports the member's hazard, where the base reported the first kind over
  the whole conjunction; both are `UNDECIDED`.
- **Tests, test-first.** At the rebased base, 25 Python pins of the new
  answers failed and the four guards (the three set-residual tests kept
  at their base pins, and the two new false-proof probes) passed; the
  Rust admission stories, `x_int_equal_to_1_0_is_answered`, the z3 story,
  `an_equation_mixing_int_and_float_is_asked_of_the_backend` and the
  property failed to compile or failed. The screen stories that pinned
  the classifier (the old "mixed equality" section and the deep, nested
  and shared walks) now call `Hazard::find_for_membership`, so the
  classifier keeps its coverage, and
  `find_admits_an_equality_of_a_literal_with_an_operand_of_the_other_kind`
  pins `find`'s admission.
- **Python.** The tests of plain questions and equation constraints pin
  the answers (25 cases, which failed at the rebased base) (as in the held patch, less its three set-constraint
  rewrites, which keep their base pins), and two new tests pin the
  false-proof probes as `UNDECIDED`.

**Track D status** (on the final tree, after the rebase and R2-040;
`648ccd9` plus this record):
- fmt; clippy `-D warnings` for the workspace both ways and for `fhy-core`
  alone with no features, `z3` and `ndarray`: clean;
- `cargo test --workspace`: 4,565; `--all-features`: 4,601 (Track A's
  landing counts plus this track's);
- `cargo test -p fhy-core --all-features` in a shell with no Python
  environment: 4,419;
- `cargo doc -D warnings`: the workspace, and `fhy-core` with default
  features and with each feature alone: clean;
- `cargo deny check`: ok; `cargo +1.85 check --workspace --lib` and
  `-p fhy-core --all-targets` three ways: no warning;
- `cargo package`, its list (no `.py`), and the CI target list;
- `pytest tests`: 8,313; `-m "not very_slow"`: 8,346; `property`: 283;
  `tests_minimal`: 6,322 passed, 665 skipped; nox `lint`, `type_check`
  and `golden_expanded`: green;
- the attribution grep over `46b8596..HEAD`: no match; every commit is
  the configured user's.
- **Benchmarks after the rebase and R2-040**, base `46b8596` against the
  head, each built alike under `target/bench/` and run twice back to back
  (`test_sympy.py`, `test_solver.py`, and, since R2-040 screens the
  constraint layer's set members, `test_constraint.py` and
  `test_param.py`; 95 rows, best of two medians): no row slower than
  1.10; the slowest are `test_constraint_repr` (1.10, a path R2-040 does
  not touch) and `test_lift_from_sympy_of_a_deep_tree` (1.06, R2-038a's
  memo, as recorded above).

**Not Track D's.** The maintainer's brief listed "the diagnostics
signature fix"; that is R2-041, which §I.7.1 and the checklist assign to
Track E (`pass/validation.rs`, `diagnostic.rs`), so this track left it.

**Python-visible changes** (§I.2 rule 6):

| Item | Old | New | Tests |
|---|---|---|---|
| R2-015 | a name hint's control characters were written into its quoted SMT-LIB2 symbol (`SmtScript.text`, `convert_expression_to_smtlib2`, the declarations' `symbol`); a NUL made the z3-solver adapter raise `Z3Exception` | each is written as `_` | `test_a_control_character_name_hint_answers_the_same_on_every_backend` (new) |
| R2-039 | the prelude was the module `_fhy_core_sympy`, and any module under that name was trusted | it is `_fhy_core_sympy_0_2_0_<hash>` with `__fhy_core_prelude__`; an impostor under that name raises `SolverBackendUnavailableError` from the first SymPy question; a pickle of a lowered piecewise or `round` names the versioned module, so one written by another version or prelude no longer loads | `test_lowered_round_and_piecewise_pickle_within_the_process`, `test_lowered_piecewise_pickle_loads_where_the_bridge_is_imported` |
| R2-038a | lifting and SymPy substitution walked a SymPy DAG as a tree, in exponential time, and lifted results shared nothing | both are linear in the distinct objects, and a lifted result shares where the SymPy object does | the Rust stories; the Python suites unchanged |
| R2-016 | `simplify_expression(y / x)` returned `y * x ** -1`, which the evaluators refuse at integer points | it returns `y / x`; every power by a negative integer lifts as a division | `test_simplify_then_evaluate_equals_evaluate_on_integer_grids` (new); no existing test pinned the old form |
| R2-040 | an equality of a numeric literal with an operand of the other int/real kind in a plain question or an equation constraint (`x_int == 1.5`, `x_real == 1`, `3.0 == y + 1`, a whole or fractional float or decimal against an int) answered `None`/`UNDECIDED` with a WARNING; `v == 2.0` on an integer param was `UNDECIDED` | each is decided by value, with no warning; `v == 2.0` is feasible and `v == 1.5` empty; a set constraint's member of the other numeric kind than its variable stays `UNDECIDED` with the WARNING; the README Solver and Constraint rows and the `fhy_core.symbolic.solver` and `ConstraintSystem` docstrings say so | the rewritten tests (25 cases) in `test_solver.py`, `test_solver_rust_binding.py`, `test_constraint_system.py` and `test_tri_state_feasibility.py`; 2 new probe tests |
| R2-005a | none in behavior; `SolverBackend.SYMPY`'s backend lives in the extension as before | the same objects, from the binding's own module | `test_missing_sympy_reports_unavailable` (new) |
| R2-014 | `SmtLib2ProcessSolver.check` could outlast its timeout (a solver that stops reading, closes stdout without exiting, exits slowly, or leaves a grandchild), and a solver that printed `success` failed with `SolverBackendError` | the timeout bounds the call; the first line written is `(set-option :print-success false)`; `success` lines before the answer are skipped; the class docstring says so | the Rust stories; the Python process-backend tests unchanged |

### Track B notes

**B0: the worktree and the baseline.** The maintainer created the worktree
as `~/Projects/FhY-core-worktrees/fix-b-expression` on branch
`fix/b-expression`, from `dev-rust` at `35519bb`, where Tracks A and D have
landed, in place of §I.2 rule 7's `fix2-expression`/`port/fix2-expression`;
the names are the only difference. So the checklist's `[rebase]` line holds
from the start, and the track works in its listed order with the wire group
after the pre-rebase items. Its `.venv` is its own (`uv sync --group dev
--group bench`), and its `target/gate-env.sh` copy points
`CARGO_TARGET_DIR` at `target/gate-cargo`. The baseline at `35519bb`
matches the Track D status: `cargo test --workspace` 4,565 passed, 2
ignored; `--all-features` 4,601; fmt and clippy `-D warnings` both ways
clean; `pytest tests` 8,313 passed, 2 xfailed.

**R2-N3.** Checked at `a9ef7b5`: `AlternativesPattern.match_under` returns
the first alternative's bindings, and `BinaryExpressionPattern.match_under`
matches the right operand under that one result, so Python never
backtracked into a later alternative; no code change. The new rstest
`alternatives_commit_to_the_first_match` (and its Python twin in
`test_pattern_rust_binding.py`) pins `Binary(Add, Alt[Capture(c, _), _],
Capture(c))` failing on `1 + 2` and matching `2 + 2`, beside the existing
`pattern_alternatives_commits_to_the_first_match`, which pins the
failing case over `any_literal`. Both passed at their first run.

**R2-012.**
- **The lane count** is checked as `ndarray` checks a shape: the lengths of
  the non-empty axes must multiply to at most `isize::MAX`, even when
  another axis is empty, so no later broadcast of an intermediate value can
  fail. Past that, `EvaluationError::BroadcastTooLarge { shape }` (a
  `ValueError`). `broadcast_view`'s `unreachable!` returns the same error.
- **The output** is reserved with `try_reserve_exact`, and a failure is
  `EvaluationError::OutOfMemory { lanes }` (a `MemoryError`). A lane count
  within one chunk is evaluated whole as before, and needs no reservation.
- **Per-chunk copies.** `ChunkSource::Copied` now holds the binding's
  broadcast view and its C-order iterator, and each chunk copies its next
  lanes into a buffer of one chunk, so a zero-stride binding is never
  materialized whole. The chunks are read in order, so the iterator gives
  each lane once, the same work as the one copy before. The spec's
  `a_broadcast_binding_is_read_per_chunk` asserts the chunk source kind
  and its buffer's size, as the spec allows, so it is a unit test of
  `evaluate/array.rs` (the source is crate-private), beside one that a
  standard-layout binding is sliced in place.
- **Test-first.** At the base, `p01`'s `(2^33, 1) + (1, 2^33)` raised
  `PanicException` and `p02`'s `(2^20, 1) + (1, 2^20)` aborted the
  subprocess with "memory allocation of 8796093022208 bytes failed"; the
  two Rust stories need the new variants.
- **Python-visible changes:**

  | Before | After | Tests |
  |---|---|---|
  | `evaluate_expression_with_numpy` over broadcasts whose lane count wraps raised `PanicException`, and over ones too large to allocate aborted the interpreter | `ValueError: the broadcast shape [..] has more lanes than an array can hold`, and `MemoryError: cannot allocate the N lanes of the result` | `test_a_huge_broadcast_raises_instead_of_aborting` (subprocess, both shapes) |

**R2-013a.**
- **Drop** moves the sub-patterns of a node's last handle onto a work
  list, as `Expression`'s does, leaving `Nothing` in the node.
- **`Debug` (call: its text).** The derived text named every field and
  recursed; the hand-written one is a prefix notation in the style of
  `Expression`'s `Debug`, with the same 1,000-node budget and `..`:
  `Pattern((add (capture "c" (literal 1)) _))`, a kind with any operation
  named by its kind (`(binary _ _)`), any operand list as `*`
  (`(call *)`, `(piecewise * (literal 0))`), `(predicate)`, and
  `(alternatives ...)`. `PatternKind` no longer derives `Debug`. The
  rstest `pattern_debug_writes_each_shape` pins each shape. The matcher's
  recursion stays documented (§I.10).
- **Test-first.** At the base, `a_200000_level_pattern_drops_on_a_small_stack`
  aborted the test binary with a stack overflow on its 1 MiB thread, and so
  did `a_deep_pattern_debug_is_bounded`; the shape pins failed on the
  derived text.
- **Python-visible changes:** none; the binding never shows a pattern's
  `Debug`.

**R2-034.**
- **`max` and `min`** are the spec's bodies, `a if (a > b || a != a) else
  b` and its `<` twin; `clamp`, `clamp_symmetric` and `relu` follow through
  them, and `leaky_relu` (`x if x > 0.0 else x * slope`) already gave NaN
  for a NaN, so it is unchanged, as is `sign` (§I.10).
- **`abs` (deviation).** The spec's `x if x > 0 else -x` gives
  `abs(0.0) = -0.0`, which the base got right (`x >= 0.0`) and the
  resolution does not ask to change. The body is `x if x > 0.0 else 0 -
  x`: the subtraction from the integer zero gives `0.0` for both zeros
  (`0 - 0.0` and `0 - -0.0` are `+0.0`), a NaN for a NaN and `+inf` for
  `-inf`, and keeps an integer operand an integer, as the negation did.
- **Lowering tests.** No Z3, SMT-LIB or SymPy test pinned the text of an
  inlined `max`, `min`, `abs` or `relu`, so none changed; the pins that
  did change are the catalogue's printed bodies and trees, the inlining
  stories, and `registry_properties`' reference evaluator, which gained
  `!=`.
- **Test-first.** At the base, 9 of the 19 cases of
  `composed_builtins_propagate_nan` failed (the p6 table:
  `max(nan, 1) = 1`, `min(nan, 1) = 1`, `relu(nan) = 0`, the clamps,
  `abs(-0.0) = -0.0`), and so did the commutativity property.
- **Python-visible changes:**

  | Before | After | Tests |
  |---|---|---|
  | `max(nan, 1)`, `min(nan, 1)`, `relu(nan)`, `clamp(nan, ...)` evaluated to the other operand (order-dependent); `abs(-0.0)` was `-0.0` | NaN propagates, as `np.maximum`/`np.minimum` do; `abs(-0.0)` is `0.0`; `sign(nan)` stays `0` | `test_max_and_min_propagate_nan_as_numpy_does`, `test_relu_propagates_nan_and_abs_of_negative_zero_is_positive` (new) |
  | the inlined bodies printed `{a if (a > b); b otherwise}`, `{a if (a < b); b otherwise}` and `{x if (x >= 0); (-x) otherwise}` | `{a if ((a > b) \|\| (a != a)); b otherwise}`, the `<` twin, and `{x if (x > 0); (0 - x) otherwise}`; `relu(x)` inlines to `{x if ((x > 0) \|\| (x != x)); 0 otherwise}` | `test_composed_builtin_body_prints_as_pinned`, `test_inliner_reports_a_change_when_it_inlines`, and the `test_builtins.py`/`test_functions_stories.py` shape pins, rewritten |

**R2-037.**
- **`Children`** implements `size_hint`, `ExactSizeIterator` and
  `FusedIterator`, and `Expression::children()` promises both traits in its
  signature (additive). The four walks that counted children by iterating
  (`evaluate/walk.rs`, `evaluate/fold.rs`, `registry/inline.rs`,
  `screen.rs`) use `len()`, and `count_children` is gone:
  `rebuild_with_children` reads `children().len()`. The walks that collect
  children get the exact capacity through `size_hint`.
- **Unary `+`** of a number returns its operand's value, lanes and
  failures, moved when no one else holds it; `unary`'s `+` arm is now
  `unreachable!`. The pointer story observes it through a plugged-in
  kernel: in `exp(+exp(x))`, the outer kernel call receives the inner
  one's result buffer (at the base, a copy `+` made).
- **One stored failing node.** The walk's failure table holds each failing
  node once; id `i` names failure `ALL[(i - 1) % 5]` of node `(i - 1) / 5`.
- **Python-visible changes:** none.

**R2-010.**
- **`Bounded`** (crate-private, `expression/display.rs`, re-exported
  `pub(crate)` from `expression` for `solver` and `types`) writes the
  `Display` text of the first `budget` node occurrences, then `…` once, and
  stops; the nodes already opened stay unclosed, so the text is a prefix
  of the full one. `Debug` keeps its own elision (`..` per node, closed).
  Messages use `MESSAGE_NODE_BUDGET = 64`.
- **The binding needs a public entry** (deviation): `Bounded` is
  crate-private, and the binding is another crate, so
  `Expression::display_bounded(occurrences) -> impl Display` is public and
  documented; it is S-4's `Bounded` under the default options, and adds no
  `FormatOptions` budget (§I.10).
- **The arms.** `EvaluationError`'s `BooleanArithmetic`,
  `NumberAsBoolean`, `MixedBranches` and `Lane`; `LoweringError`'s
  `NonFiniteLiteral`, `Call` (its non-call fallback; a call's arms name only
  the callee), `SortMismatch` and `UnsupportedPower`; and
  `TypeCheckError::Rule`. For the last, `types/checking/error.rs`'s
  `format_expression` itself writes `Bounded` with identifier ids, so the
  root and the sub-expression are bounded, and so are the reasons the
  checker builds with it (`checker.rs`, Track C's, is not touched), which
  the checker formats eagerly and would otherwise also be exponential on a
  DAG.
- **The lane.** `EvaluationError::Lane { lane: Option<usize> }`: `None`
  for a scalar evaluation, and for an array one the flat index in C order
  of the result's shape. The array backend broadcasts the failure ids to
  the shape it computes (the result's, or a chunk's) before searching, and
  a chunk's index is offset by the chunk's start. The text is `integer
  division by zero at lane 2 in (x // y)`; a scalar's is unchanged. The
  lane property now asserts the index is the first lane the scalar
  evaluation fails in.
- **`str`/`repr`.** `Expression.__str__` writes `display_bounded(1000)`
  above 1,000,000 occurrences. `__repr__` already went through the core's
  `Debug`, which has had a 1,000-node budget since the first spec, so it
  needed no change.
- **Other tracks' files**, additively: `solver/error.rs` (the four arms),
  `types/checking/error.rs` (`format_expression`), and one story each in
  `tests/it/solver/smt_lowering_stories.rs` and
  `tests/it/types/checking/checker_stories.rs`. D's R2-029d tables pin
  small nodes, so no pin changed.
- **Python-visible changes:**

  | Before | After | Tests |
  |---|---|---|
  | error messages naming a node wrote its full text, which for a DAG never finished; `str()` of a decoded doubling DAG hung | a node is written up to 64 occurrences, then `…`; `str()` above a million occurrences writes the first thousand, then `…` | `test_str_of_a_decoded_doubling_dag_is_bounded` (new) |
  | a lane failure of `evaluate_expression_with_numpy` read `integer division by zero in (x // y)` | `integer division by zero at lane 2 in (x // y)`, the first failed lane in C order | no Python test pinned the array text |

**R2-026a.** Tests only, and every new test passed at its first run, so no
new finding.
- **Generators.** `tree_strategies` draws `+x` and Boolean piecewise
  nodes. Rstests evaluate `+3`, `+2.5`, `+(-0.0)` and Boolean piecewise
  selections as scalars, and `+x` over integer and real arrays and a
  Boolean piecewise over Boolean arrays.
- **IEEE edges.** `real_floor_division_and_modulo_follow_numpy_at_the_ieee_edges`
  holds the 13 rows of `kernel_probe.rs`, with NumPy 2's `floor_divide`
  and `mod` results, compared by bits (every NaN alike).
  `integer_floor_division_and_modulo_agree_with_an_i128_oracle` checks the
  quotient and remainder against `i128`, and the zero-divisor and
  `i64::MIN // -1` failures.
- **Chunks (call).** The chunk size is a parameter of the crate-private
  `Prepared::evaluate_array_in_chunks`, which `evaluate_array` calls with
  65,536. Since only the crate can reach it, the "lane property with chunks
  of 1 to 3 lanes" is a unit property of `evaluate/array.rs`: over eleven
  trees reaching each node kind, lane failures and the guards that drop
  them, and 1 to 9 random lanes, a chunked evaluation equals the whole one,
  the failed lane's index included. `chunk_probe.rs` is the story
  `a_chunked_evaluation_equals_the_scalar_evaluation_of_every_lane`: its
  300,003 lanes fail at lane 200,003, the third chunk, and every lane
  before it matches the scalar evaluation. It is the suite's slowest story,
  about 20 s in a debug build, from 200,000 scalar evaluations that each
  re-run the Boolean screen.
- **Array bindings.** Boolean and transposed real bindings of 15 and
  120,300 lanes (below and above one chunk), and `MisshapenKernels`, whose
  wrong shape is `EvaluationError::Kernel` with the shape text as its
  source.
- **Python-visible changes:** none.
- **An unreproduced failure.** One `cargo test --workspace --all-features`
  run during this item reported one failed test in one binary, with the
  output piped through `-q`, so the test is not known; six full runs
  since, back to back, passed. Other tracks were building on the machine,
  and the timing-bounded process-backend stories (R2-014) are the likely
  candidate.

**R2-047a.**
- **`EvaluationError::NumberAsBoolean`** is deleted, and so is the
  binding's mapping and its now unused `non_boolean_operand_error`
  helper. The walk's three sites (a number under `!`, in a connective, as
  a piecewise condition) are `unreachable!`, naming the Boolean screen that
  `Prepared::evaluate` and `evaluate_array` run first;
  `a_number_in_a_boolean_position_is_ill_typed` pins `!x`, `all(x, p)` and
  `piecewise(x -> 1, 0)` as `IllTyped`, and passed at the base.
- **`arithmetic`** takes the two integer lane containers; `binary` routes
  integers to it, Booleans to `BooleanArithmetic`, and marks the real case
  `unreachable!`, which `combine` sends to `real_arithmetic`. `unary` keeps
  only its live arms (a Boolean under `-` or `+`, and an integer negation);
  `combine` handles the rest. `logical` loses its unused node argument.
- **Coverage** stays without a gate (§I.10).
- **Python-visible changes:** none; the variant was never produced, so the
  `NonBooleanLogicalOperandError` it mapped to still comes from the
  screen.

**R2-029b.** Tests only; every new test passed at its first run.
- **Tables.** `tests/it/expression/error_text_stories.rs` holds one rstest
  table per error type of the three files: `PiecewiseError`,
  `RebuildError`, `BooleanPosition`'s phrases and
  `NonBooleanLogicalOperandError`; `FunctionDefinitionError` (both
  pluralizations), `ConstantValueError`, `RegistrationError` and
  `InlineError`; `LaneFailure`, `EvaluationError` (each variant, R2-012's
  and R2-010's included, the three `Unbound` near misses, and a lane with
  and without an index) and `FoldError`. Each checks `to_string()` and the
  source's type through `support/error_text.rs`, and a story checks that
  `EvaluationError::Inline` passes the inlining error's own source on.
- **Tightened.** `no_native_calls_fails_every_user_native` matches the
  function and the source's text; `fold_checks_the_arguments_before_the_call_taking_them`
  the function and the NaN; the two registry stories compare the constant
  itself instead of `is_some()`; and the inlining story downcasts its
  source to the `PiecewiseError`.
- **Rewrite blame through a grandchild** is not reachable through
  `apply_rewrite_rules`: its only refusal names a condition that is a
  literal, which a rule returned directly. So the three stories are unit
  tests of `RuleApplier::find_blamed_rule` in `pattern/rewrite.rs`,
  recording replacements by hand: a child rebuilt around a replaced
  grandchild blames the grandchild's rule, through both the refused child
  and the last rewritten child, and a rebuild with no rewritten child
  blames the last firing.
- **Python-visible changes:** none.

**The wire group: R2-011, R2-036, R2-001a and R2-046a** (one commit, J-4).
- **S-1, `expression/canonical.rs`.** `CanonicalTable::build(root,
  Equivalence)` walks post-order on a work list, memoizes shared handles by
  `NodeIdentity`, and hash-conses each node by `(data, child indices)` in
  one `HashMap`, so it is linear in the distinct handles and never
  compares deeply. `Wire` compares floats by bits with one NaN, and
  identifiers by id and name hint (call: the wire writes the hint, so two
  identifiers of one id and different hints, which only `try_restore` can
  make, stay two nodes; the `Serialize` rustdoc names this beside J-5's
  zeros). `Structural` folds the zeros and compares identifiers by id.
- **R2-011.** `encode_nodes` serializes the `Wire` table; the decoder was
  already sharing every repeated index. The property that the round trip
  kept the input's sharing now asserts the canonical sharing: two children
  are one node after the round trip exactly when they encode alike, and
  decoding then encoding is the identity.
- **R2-036 (S-2).** `write_float` (crate-private, `expression/literal.rs`)
  writes every float text: `LiteralValue`'s `Display`, the literal and
  constraint wire forms (`float_text::serialize` replaces
  `serialize_display_text` for floats), the member and value `Display` of
  `constraint/value.rs` (Track A's file, one line each), and the key texts.
  The decoders refuse a float text that is not its value's canonical text,
  and a decimal text that is not its `Display` text, with the spec's
  messages. `Decimal`'s own `Deserialize` is canonical, so every decimal a
  payload holds, a param's bound included, is read canonically; no writer
  in the repository wrote another text. The one existing story that read a
  padded decimal text (`expression_literal_reads_a_text_as_a_normalized_decimal`)
  now pins the refusal, and the float texts of `1e16`, `1e22`, `1e-7` and
  `1.2345678901234568e17` in the display and pprint pins changed.
- **R2-001a.** `equation_key` renders the `Structural` table as
  `kind[data](i,j,…)`, `;`-separated. A literal keeps today's key text with
  S-2's floats, `+ 0.0` folding the zero, so NaN keys as `float:NaN`
  (today's `float:nan` went with the rest of the float text). A callee is
  `builtin:<name>` or `named:"<name>"`, quoted by the crate-private
  `write_quoted` (`"` and `\` backslash-escaped). Member floats of set keys
  use S-2's text. No Rust pin compared an equation key's text, so the new
  stories pin it: `x + x`, `f(x)`, `max(x, 1)`, and floats.
- **R2-046a.** The generator gains `value_frozenset_of_tuples`, a
  frozenset of four tuples over ints, floats (`1e300` and `-0.0`), a
  string, a Boolean and nested tuples, written by `serialize_value` (a
  member value is no `Serializable`) with the class `builtins.frozenset`;
  the Rust replay's `Value` arm now runs. The member canonicalization writes
  `-0.0` inside a set as `0`, which the case pins. In Track E's
  `tests/serialization/test_wire_v2.py`, the class-based replay skips the
  `Value` cases, and a new test replays them through `deserialize_value`
  and `serialize_value` (additive).
- **The regeneration.** One run of the generator: 16 of the committed V2
  texts changed (the extreme floats of the literal and random cases, and
  the tables where an equal subtree now repeats, such as the piecewise
  fixture's two literal `0`s), and the `Value` case was added; the
  expanded corpus (`golden_expanded`, 2,000 random cases) replays. The V2
  pin of the piecewise fixture in `test_serialization_pins.py` changed the
  same way. No other V2 pin held a repeated subtree or an extreme float.
- **R2-042 and R2-032b** did not join the commit; they are next and are
  checked against the corpus there.
- **Test-first.** With the source changes stashed, 29 of the 38 new wire,
  literal and key cases failed at the base (the others are canonical texts
  the base already wrote or refused); the depth-64 key story needs no
  proof, as the base's key is exponential.
- **Other tracks' files**, additively: `constraint/value.rs` (two
  `Display` lines), `tests/serialization/test_wire_v2.py`, and the
  revision bullets of `rust-workspace.md` (D-6, D-7 and R-16, B3 §5.6) and
  `python-switch.md` (W-12, D-S13-6), and CONTRIBUTING's float sentence.
- **Python-visible changes:**

  | Before | After | Tests |
  |---|---|---|
  | V2 wrote a node once per handle, so equal expressions built with different sharing wrote different texts; a decoded value shared only what the payload shared | each distinct node is written once, so equal expressions write the same text, except for a zero's sign; a decoded value shares every repeated subtree | `test_a_fixture_writes_its_golden_v2_text[piecewise_expression]` (repinned), the corpus replay |
  | floats outside `[1e-5, 1e16)` were written positionally in V2, `str()`, `pformat_expression` and messages (`1e300` as 301 digits) | with an exponent: `1e300`, `5e-324`, `1e16`, `1e-7` | `test_pformat_literal_renders_the_core_text` (repinned), `test_str_of_an_extreme_float_literal_is_its_canonical_text` (new) |
  | V2 read any float text Rust parses (`"1e5"`, `"+1.5"`, `"nan"`) and any decimal of the grammar (`"1.50"`) | only the canonical text; another raises `DeserializationValueError` naming the canonical text | the Rust stories; no Python writer produced another text |
  | `build_ordering_key()` of an equation rendered the tree in pre-order, a callee by its `Debug` text | the node table: `equation|identifier[7]();literal[int:1]();binary[add](0,1);...`, `call[builtin:max]`, `call[named:"f"]`; a depth-40 doubling DAG keys at once | `test_an_equation_key_is_its_expressions_node_table`, `test_a_depth_40_doubling_dag_constraint_builds_promptly` (new) |
  | a system's members sorted by the old key text | by the new one, so a system's V2 member order can differ | the corpus replay |

**R2-042.**
- **The system** keeps, beside its members, the length of each tie run (a
  run of members with one key). `new` sorts by key, then orders each run
  by equivalence groups, a group's members together and the groups in the
  order of their first member (J-7's residual order, which the rustdoc
  names). A system of distinct keys, the only kind a conforming
  implementation builds, is unchanged.
- **Equivalence and hashing.** `is_structurally_equivalent`, `==` and
  alpha equivalence require the same run lengths in order, and match each
  run as a multiset (greedily, which is exact for an equivalence; a run of
  one is compared directly). `Hash` feeds each run's length and its
  members' hashes sorted, so it agrees with the multiset equality.
- **Contract text.** `OpaqueValue::ordering_key` and
  `CustomConstraint::ordering_key` (Track A's files, doc lines only) say
  the key is equal exactly when the values are equal, and what a collision
  costs; the module docs of `constraint` and `constraint/key.rs` say keys
  decide equivalence for conforming implementations.
- **Tests.** The TYP probe as `systems_with_colliding_opaque_keys_are_equivalent_in_either_order`
  (structural, `==`, hash, alpha), the grouping order, a `Param` over such
  a system in either order (in `system_stories.rs`, since the param story
  files are Track C's), and the property
  `system_order_and_equivalence_do_not_depend_on_the_input_order`, whose
  members draw colliding opaque keys. The three stories failed at the base.
- **The corpus** is unchanged (the regeneration check passes), so R2-042
  stays out of the wire group's commit, as J-4 allows.
- **Python-visible changes:**

  | Before | After | Tests |
  |---|---|---|
  | two systems (or params) with the same members, two of them unequal members whose keys collide, were unequal when given in the other order | equal, and they hash alike; within a run of colliding keys, equal members sit together | the Rust stories; no Python test built colliding keys |

**R2-032b.**
- **No cached key (deviation).** J-8 has `Ord` compare cached keys, and J-2
  caches a key "in the member or constraint". Track A cached the opaque
  member's key in its `Member`, but a `Constraint` holds no key: its
  `Custom` variant is a bare `Part<dyn CustomConstraint>`, and caching
  there would change that public variant (Track A's type). So `Ord`
  computes the two keys for each comparison: a built-in kind's key is
  linear in its distinct nodes since R2-001a, and a custom one asks its
  hook. A custom constraint whose key fails is outside J-7's contract, and
  `Ord` puts it after every constraint whose key does not fail, equal to
  any other such one, so the order stays total; `ConstraintSystem::new`
  still refuses such a member with its error. `new` sorts by the keys it
  reads once, which is the order `Ord` defines, rather than calling `Ord`,
  which would read each key once per comparison.
- **Tests.** The property `constraint_order_is_total_and_agrees_with_equivalence`
  (equal exactly when `==`, antisymmetric, transitive, `partial_cmp`, and
  the keys' order) over equations and set constraints, a `BTreeSet`
  story, and a story that a custom constraint orders by its key and one
  whose key fails orders last.
- **Python-visible changes:** none; the binding's classes define no order.

**R2-N1.**
- **Where the time went.** At the wire group's head, decoding the
  benchmark's 102-literal tree cost about 270 µs against V1's 110 µs:
  the dict was turned into a `serde_json::Value`, the core decoded an
  `Expression`, and the materializer then rebuilt every node through its
  public class, re-reading each literal from a Python value (a big integer
  from its digits, a decimal through `decimal.Decimal`, whose constructor
  alone is about 1.1 µs).
- **The seed.** `LiteralExpression`'s constructor also accepts a private
  `_LiteralSeed` (not exported, only the binding builds one) holding a core
  literal node: the object keeps that handle, and its `value` is a
  `PyOnceLock` computed on first read (`literal_from_core`). The
  materializer's literal arm uses it too, so every core-built literal
  (substitution, simplification, solver results) skips the parse.
- **The table fast path** (`fhy-core-py/src/expression/table.rs`, new):
  `Expression.deserialize_from_dict` walks a V2 table's Python objects once,
  in table order, building each node's object from its children's objects
  by index, so a shared node is built once and an identifier's Python
  `Identifier` is made once per id. Leaves go through the core's serde
  (`LiteralValue`, `Identifier`, `Callee`), so R2-036's canonical checks
  hold; the structural checks (indices preceding, every node but the root
  referenced, logical arity, piecewise cases and non-Boolean literal
  conditions) are the core decoder's. Any other payload, or a constructor's
  refusal, falls back to the core path, which raises its errors, so the
  errors and the accepted payloads are unchanged; the V1 fast path of
  `payload.rs` is untouched. `from_json` still decodes through the core.
- **The target (J-12).** `test_deserialize_from_dict[v2-literals]`: about
  270 µs at the wire group's head, 225 µs with the seed alone, and 113 µs
  with the table path, against 112 µs for V1 in the same run and J-12's
  132 µs (1.10 × 120.29 µs). The before and after table of §I.8.3 is in the
  Track B status. python-switch's S17 benchmarks gain a revision bullet.
- **Tests.** The suite; `test_a_decoded_literal_is_the_constructed_one`
  over NaN, `-0.0`, `1e300`, big integers, decimals, a Boolean and `0`
  (equality, hash, `str`, `repr`, the value's type and identity across
  reads, pickling and re-serializing); a decoded table shares its repeated
  node and holds the right identifier; and six malformed tables raise the
  core decoder's `DeserializationValueError` text.
- **Python-visible changes:** none in behavior; decoded literals compute
  `value` on first read, and a decoded tree's repeated subtrees are one
  object, as R2-011's decoder already made them one core node.

**Fixed forward.** `d77f7ec` (R2-034) indexed a NumPy result in
`test_relu_propagates_nan_and_abs_of_negative_zero_is_positive` in a way
mypy's NumPy stubs refuse, which nox `type_check` (a track gate, not a
per-commit one) found at the track gates; the test reads the lane through
`tolist()` instead.

**After the benchmarks: a leaner canonical table and literal.** The first
benchmark pass (below) showed the table's cost on small expressions: an
equation key 2.1 times the base's, and a 20-member system 1.7 times. The
table now keeps every node's children in one flat list, looks nodes up by
a `(data, children)` hash chain under the crate's identity hasher instead
of a `HashMap` keyed by owned child vectors, and reserves room for a small
expression; the key is written into a reserved string. A
`LiteralExpression` built by its constructor holds its value in a plain
field, and only a seeded one in the `PyOnceLock`. The key and system rows
came down to 1.26 and 1.19, and the V2 writes of deep trees to 0.92.

**Track B status** (on the final tree, `9800307` plus this record):

| Gate | Result | At `35519bb` |
|---|---|---|
| fmt; clippy `-D warnings`, workspace both ways and `fhy-core` alone with no features, `z3` and `ndarray` | clean | clean |
| `cargo test --workspace` | 4,769 passed, 2 ignored | 4,565 |
| `cargo test --workspace --all-features` | 4,805 passed, 2 ignored | 4,601 |
| `cargo test -p fhy-core --all-features`, with no Python environment | 4,623 passed | |
| `cargo doc -D warnings`: the workspace, `fhy-core` alone, with `z3`, with `ndarray` | clean | |
| `cargo deny check` | ok (the base's `syn` duplicate warning) | |
| `cargo +1.85 check --workspace --lib` and `-p fhy-core --all-targets` three ways | no warning | |
| `cargo package`; its list | ok; no `.py` file | |
| `pytest tests` | 8,350 passed, 2 xfailed | 8,313 |
| `pytest tests -m "not very_slow"` | 8,383 passed, 2 xfailed | 8,346 |
| nox `property` | 283 passed | 283 |
| nox `tests_minimal` | 6,354 passed, 665 skipped | 6,322, 665 |
| nox `lint`, `type_check`, `golden_expanded`; `tests/test_golden_corpora.py` | green | green |
| attribution grep over `35519bb..HEAD` | nothing; every commit is the configured user's | |

**Benchmarks** (§I.8.3: `test_serialization.py`, `test_expression.py`,
`test_constraint.py`, `test_evaluate.py`; 184 rows). The base (`35519bb`)
and the head were each built from a `git archive` under
`target/bench/` (`uv sync --no-default-groups --group bench --group
test`, CPython 3.11), and run with `pytest --benchmark-only -n 0`. Another
user's jobs held the 24-core machine at a load of 30 to 38 throughout, so
single medians moved by up to 2 times, in both directions; the table
compares the least time over three interleaved runs (base, head, base,
...), and the sub-microsecond and flagged rows were re-measured with 200
rounds each. The median row is 1.00.

| Benchmark | Base | Head | Ratio |
|---|---:|---:|---:|
| `test_deserialize_from_dict[v2-literals]` (J-12) | 262.18 µs | 110.64 µs | 0.42 |
| `test_deserialize_from_dict[v1-literals]` | 104.10 µs | 107.93 µs | 1.04 |
| `test_deserialize_from_dict[v2-deep_expression]` | 220.60 µs | 84.47 µs | 0.38 |
| `test_deserialize_from_dict[v2-wide_expression]` | 2.49 ms | 61.07 µs | 0.02 |
| `test_json_round_trip[v2-wide_expression]` | 1.72 ms | 205.01 µs | 0.12 |
| `test_serialize_to_dict[v2-wide_expression]` | 768.24 µs | 141.65 µs | 0.18 |
| `test_serialize_to_dict[v2-deep_expression]` | 65.09 µs | 59.80 µs | 0.92 |
| `test_serialize_to_dict[v2-literals]` | 75.72 µs | 83.07 µs | 1.10 (1.097; 102 literals through the canonical table) |
| **`test_constraint_build_ordering_key[equation]`** | 1.18 µs | 1.52 µs | **1.26 to 1.29** |
| **`test_constraint_system_construction`** | 18.53 µs | 23.32 µs | **1.19 to 1.26** |
| **`test_serialize_to_dict[v2-type]`** | 1.65 µs | 1.84 µs | **1.11 to 1.15** |
| **`test_eq_of_distinct_equal_deep_trees`** | 4.48 µs | 5.02 µs | **1.07 to 1.12** |
| **`test_structural_equivalence_of_deep_trees`** | 4.63 µs | 5.13 µs | **1.06 to 1.12** |
| **`test_set_constraint_construction[4]`** | 1.91 µs | 2.12 µs | **1.11** (one run) |
| `test_literal_expression_construction[int]` | 0.28 µs | 0.28 µs | 1.02 (200 rounds; 1.36 in the three-run pass) |

- **J-12 is met:** the literal-heavy V2 decode is 110.64 µs, against the
  target of 132.3 µs (1.10 × 120.29 µs) and V1's 107.93 µs in the same
  runs.
- **Flagged for the maintainer (above 1.10):**
  - the equation key and a system's construction, which R2-001a makes
    build the canonical table: a key is now linear in distinct nodes (the
    depth-64 DAG keys at once, where the base's key grew fourfold per two
    levels), at about 0.3 µs more for a three-comparison equation;
  - `test_serialize_to_dict[v2-type]`, a type holding two shape
    expressions, whose encoding builds the canonical table (R2-011); the
    larger trees' writes are as fast or faster;
  - the two deep-tree equality rows, about 0.5 µs on 4.5 µs, consistent
    over four runs; the likeliest cause is `Children`'s exact size hint
    (R2-037), which makes each `extend` of the comparison's work list
    reserve, and it was not pursued;
  - `test_set_constraint_construction[4]`, seen once, which R2-042's
    tie-run bookkeeping and the canonical float text may cost.

### Track C notes

**C0: the worktree and the baseline.** The maintainer created the worktree
as `~/Projects/FhY-core-worktrees/fix-c-types-param` on branch
`fix/c-types-param`, from `dev-rust` at `35519bb` (Tracks A and D landed),
in place of §I.2 rule 7's `fix2-types-param`/`port/fix2-types-param`; the
names are the only difference. Its `.venv` is its own (`uv sync --group dev
--group bench`), and its `target/gate-env.sh` points `CARGO_TARGET_DIR` at
`target/gate-cargo`. A clean copy of the base (`git archive 35519bb` into
`target/base-src`, with its own `.venv` and target directory) serves the
baseline and the benchmarks' "before" runs. The baseline is Track D's
status line: `cargo test --workspace` 4,565 and `--all-features` 4,601;
`pytest tests` 8,313.

**R2-017.**
- **Where.** `infer` checks a `Negate` of an integer or float literal as
  the one negated literal against the expected type, so the negate rule
  never sees one; its weak-literal branch, which only that case reached,
  is gone. A Boolean or decimal operand still takes the rule's path and
  its errors.
- **Not widened.** A binary operation still hands the expected type, and
  the weak-literal rescue, only to a bare literal operand:
  `a_literal_nested_below_a_negation_escapes_the_range_check` pins that
  `x_int32 + -(2^200)` synthesizes `int32`, and the spec's change names
  only the negation's own check.
- **Python-visible changes:**

  | Before | After | Tests |
  |---|---|---|
  | `check(-(5), uint8)` returned `uint8`; `check(-(128), int8)` raised "synthesized type int16[] is wider than the expected type int8[]" | the first raises "literal -5 is incompatible with uint8"; the second returns `int8` | `test_negated_literals_check_as_one_literal` |

**R2-019.**
- **The API.** `FunctionLabel { Builtin(BuiltinFunction), User(FunctionName) }`
  (exhaustive, with `From` both ways and `&FunctionName`) names the function
  in `FunctionSignature` and in every `BodyCheckError` variant, whose
  `function` field was a `FunctionName`. `FunctionSignature::new` takes
  `impl Into<FunctionLabel>` and returns `SignatureError::LengthMismatch`
  (new, `#[non_exhaustive]`) on a length mismatch; the signature loses
  `Copy`, since it owns its label. `check_all_function_bodies` returns a
  `BodySweep` (`checked()`, `failures()`, `into_failures()`).
- **A malformed signature in the sweep (call).** The sweep reports it as
  the new `BodyCheckError::Signature` under its label, not a panic; neither
  the catalogue nor a `FunctionDefinition` can produce one today.
- **The seam** is the crate-private `sweep(registry, builtins)`, which
  `check_all_function_bodies` calls with the catalogue's composed bodies;
  the unit test hands it a broken `max`.
- **Python (call).** `_rs.types_check_all_function_bodies` gains an
  optional `on_checked` callable, called with each checked function's name
  after the sweep, which is the "counting hook" of the Python test; the
  public `check_all_registered_function_bodies()` keeps its signature. The
  stub's line changes with it (a shared file).
- **All 16 composed bodies check**, so the report stays empty.
- **Fixed forward:** `f634007` (R2-017) left one Python test line 90
  characters long, which ruff's E501 refuses; this commit formats it.
- **Python-visible changes:**

  | Before | After | Tests |
  |---|---|---|
  | `check_all_registered_function_bodies()` checked no composed built-in | it checks the 16 in catalogue order, then the user functions; a failing built-in gets a diagnostic naming it | `test_the_sweep_checks_every_expression_bodied_builtin` |
  | `_rs.types_check_all_function_bodies()` took no argument | it takes an optional `on_checked` | the same |

**R2-001b.**
- **The memo** maps a shared node's identity to the results inferred for
  it, each with the expected type it was inferred with, compared by `==`
  (the expected types a walk hands down are few: the caller's, `bool` for a
  condition, and none). A `Remember` step under the node's own steps stores
  the result once they finish.
- **Errors (reading the spec).** The walk stops at its first error, so a
  memoized error could never be replayed; the memo keeps results only, and
  `a_shared_ill_typed_node_reports_its_error_once` pins that the error names
  the shared node, as a tree's would.
- **Leaves are not memoized (call).** An identifier or literal costs one
  step, and `test_identifier_lookup_is_called_once_per_occurrence_with_the_callers_objects`
  pins that a shared identifier node reaches the lookup at each occurrence;
  memoizing compound nodes alone keeps that, and keeps the walk linear.
  The binding's and `type_checker.py`'s docs now say a shared compound
  sub-expression is checked, and so looked up, once.
- **Python-visible changes:**

  | Before | After | Tests |
  |---|---|---|
  | a sub-expression shared by several parents was re-checked, and its identifiers and calls looked up, once per path (a depth-40 doubling DAG did not finish) | it is checked once per expected type; the depth-40 DAG checks with two lookups | `test_a_doubling_dag_of_depth_40_checks_in_under_a_second` |

**R2-026b.** Tests only; every new test passed at its first run, so no new
finding.
- **Rstests:** `+x` over `int16`, `uint8`, `float32` and `complex64` keeps
  the operand's type and qualifier; `!p` is `bool` with `p`'s qualifier;
  `x_f32` and `y_i16` in both orders under `+` (a promotion error naming
  the two in the order written), `/` and `//` (`float32` both ways).
- **The broadened properties** draw from eight identifiers (`int8` to
  `int32`, `uint8`, `uint16`, `float32`, `float64`, `bool`). A first
  generator of arbitrary shapes gave well-typed trees in about 1% of cases,
  so the laws would have held vacuously; the generator is type-directed
  instead (integer, float and Boolean trees, with unary operators, the
  arithmetic, comparisons, connectives and piecewise nodes), mixed with
  the arbitrary shapes at one in seven. Counted with a probe, not
  committed: of 256 cases, 233 trees synthesize and 230 evaluate, spread
  over the three kinds, and 182 of the 256 commutative pairs synthesize.
  Both laws also held over 5,000 cases.

**R2-047b.**
- **The assertion.** A `const` block in `checker.rs` walks every core data
  type (`core_data_type.rs`'s `ALL`, now `pub(super)`, a one-word edit) and
  asserts that no integral one is wider than `float64`. So
  `real_float_of_width` returns a `CoreDataType` and `expect`s with that
  reason, and `lift`, `lift_pair` and the integer arm of `division` became
  infallible: the two "no real float core data type found for bit width"
  errors are gone.
- **`primitive_of`** (the audit's `:415-420`) is backed by the walk, not by
  the assertion: every caller reads a type `as_value` admitted, once index
  types are handled, so it is a free function that `expect`s. The audit's
  `:894-901`, the negate rule's non-numeric literal arm, went with R2-017.
- **Behavior change:** none; the removed errors could not be produced.

**R2-018.**
- **The error.** `UnificationError::Substitution(PiecewiseError)` (Track
  A's `types/error.rs`, one variant with its `Display` and `source` arms)
  reads "substituting the existing shape bindings was refused", with the
  refusal as its `source()`. `substitute_expression` returns it, so
  `bind_placeholder`, and through it `unify_expressions` and `Type::unify`,
  and `Type::substitute_template` (numerical shapes and index bounds) fail
  where they went on with the unsubstituted form.
- **The graph occurs check** (`occurs_through_bindings`) walks the free
  identifiers of the unsubstituted expression and of each binding it
  reaches, on a heap stack with a visited set, so it also ends on a cyclic
  environment. `bind_placeholder` refuses when either it or the old check
  on the substituted form finds the placeholder; with substitution now
  exact the two agree, which the unit tests and the stories pin.
- **Python-visible changes:**

  | Before | After | Tests |
  |---|---|---|
  | with `C := 5`, `Y := X + 1`, `unify_expression(X, {Y if C; 0 otherwise})` returned an environment binding `X` in a cycle, and `substitute_template` returned a type still holding `Y` and `C` | both raise `VerificationError` "substituting the existing shape bindings was refused" | `test_a_refused_shape_substitution_is_an_error` |

**R2-038b.**
- **The substitution** keeps its chain on the heap, one frame per binding
  being substituted, marking each shape variable white, grey (on the chain)
  or black (its form known). The definition keeps "a variable already on
  the chain stays", which an environment built by hand may still need,
  since `with_expression_binding` accepts a cycle (the stories
  `substitution_follows_a_chain_of_bindings_and_stops_at_a_cycle` and
  `a_cycle_of_placeholder_bindings_resolves_to_where_it_closes` pin it).
  So a form is remembered only when its computation met no grey variable
  but its own and used no such form: then it is the same along every
  chain. In an acyclic environment every form is remembered, and a call is
  linear in the bindings it reaches; a form that met a cycle is recomputed
  where it is reached again, as before. The property
  `substitution_agrees_with_its_recursive_definition` checks the result
  against the recursive definition over random environments, cycles
  included (5,000 cases run once).
- **The environment (call): a persistent table of shared layers**, in
  `environment.rs`, with no new dependency. A `with_*` adds a one-entry
  layer and merges it into the layer below while that one is no larger, as
  a binary counter carries, so a table holds logarithmically many layers,
  each entry is copied logarithmically often, and a lookup probes each
  layer at most once. The layers are shared and never mutated, so an
  environment stays a value. A persistent map crate (`im`, `imbl`, `rpds`)
  would have added a dependency tree to the core and edits to Track D's
  manifests and `deny.toml`; the layers need neither. The iterators,
  `==`, `Hash`, `Debug` and structural equivalence read the newest value of
  each key. `an_environment_agrees_with_a_map_of_its_bindings` checks every
  step of random `with_*` sequences against a map, and that earlier
  environments are unchanged.
- **Tests:** `fibonacci_bindings_substitute_in_linear_time` (n = 64, run in
  debug and release, under 2 s; it did not finish at the base) and
  `a_100000_binding_chain_substitutes_on_a_small_stack` (building the
  chain's environment alone was quadratic at the base).
- **Behavior change:** none; substitution results are DAGs sharing each
  binding's form.

**R2-020.**
- **`add_symbol`** refuses, after its ancestor check, a symbol that a
  namespace whose chain of parents reaches the target holds, with the new
  `SymbolTableError::SymbolDefinedInDescendant` ("symbol y already defined
  in namespace c, a descendant of namespace p"); only the namespaces
  holding the symbol walk their parents. `violations()` gains
  `Violation::ShadowedSymbol`, reported last, for a symbol an ancestor
  also holds (outside a cycle), which only `insert_namespace` can build
  now. The model of `table_properties.rs` gains the descendant rule.
- **Build order (reading the spec).** `SymbolTableData::build` adds every
  namespace first and then every symbol. The namespaces are added in
  payload order, not reordered parents-first: `add_namespace` checks no
  parent, and payload order keeps the table's namespace order, so a
  decoded table re-encodes byte-identically, which R2-046b's properties
  need.
- **Assignments (call).** `ParamAssignmentData::build` goes through
  `ParamAssignment::restore` under the caller's context. The core's
  `Deserialize` builds its param with a solver without backends, under
  which evaluating any equation fails with `NoCapableBackend`; so it
  checks under a crate-private observer that counts that failure as
  undecided, and `restore` accepts an undecided member. Admissibility and
  set-constraint membership are decided without a solver and refused.
- **Rewritten pins:** `an_assignment_round_trips_without_checking_its_value`
  (a value outside the in-set decoded) is now
  `an_assignment_round_trips_and_its_value_is_checked_on_decode`, and
  `the_nearest_namespace_holding_a_symbol_answers_a_lookup` builds its
  shadowing table with `insert_namespace`, since `add_symbol` refuses it in
  either order.
- **Python.** The binding decodes tables and assignments through its own
  replay and constructors, so of the three interface tests only
  `test_add_symbol_refuses_a_name_a_descendant_defines` failed at the base;
  `test_a_table_whose_child_was_added_before_its_parent_round_trips` and
  `test_an_inadmissible_assignment_payload_fails_to_decode` pass at the
  base too, and pin that Python keeps agreeing.
- **Revises** D-S15-11 and D-S17-9 (bullets appended in
  `python-switch.md`).
- **Python-visible changes:**

  | Before | After | Tests |
  |---|---|---|
  | `SymbolTable.add_symbol(parent, y, ...)` succeeded when a child of `parent` already held `y` | it raises `SymbolTableError` "symbol y::… already defined in namespace child::…, a descendant of namespace parent::…" | `test_add_symbol_refuses_a_name_a_descendant_defines` |

**R2-021.**
- **Admissibility and the value-set subset** read the sign restriction of
  the built-in integer kinds directly: a non-negative domain admits `0` and
  up, a positive one `1` and up, and a numeric domain's set lies in
  another of its sort only when the other's restriction is no stronger.
  Against a custom domain of its sort, a numeric domain's set is a subset
  when each of the custom domain's implied constraints (on a fresh
  variable) is one of its own; any other restriction is not proven and
  answers `false`. A custom domain's own `is_value_admissible` answers for
  its own restriction, as before.
- **The side-taking procedures** (`has_feasible_value`,
  `feasibility_subset`, `compute_constraint_implication_subset`, `union`,
  `intersection`) fold each side's domain's implied constraints into the
  side, skipping one the side already holds, so a param's side, which
  holds them, is unchanged (`the_param_path_holds_the_restriction_once`).
- **Custom domains (call).** A procedure that dispatches to a custom
  domain's own hook hands it the sides as given, since the hook answers
  for its own domain; a built-in procedure that meets a custom domain on
  the other side asks its `implied_constraints` to fold them in. The folds
  happen only on the paths that read the constraints, so a finite domain
  against a custom one still asks the custom domain nothing.
- **Rewritten pins:** `non_negative_integer_domain_admits_a_negative_integer`
  (now `..._refuses_...`), `numeric_value_sets_are_subsets_within_one_sort`
  (the unrestricted interval integers are no subset of the naturals), and
  two custom-hook recordings that gain the `implied_constraints` call. In
  Python, `test_nat_param_is_value_admissible_does_not_gate_on_sign` (now
  `..._gates_on_sign`) and `test_assignment_payload_rejects_only_a_provable_violation`
  (`-1` is now inadmissible, so the violation case uses `x <= 5` with `7`).
- **Python-visible changes:**

  | Before | After | Tests |
  |---|---|---|
  | `IntegerDomain(non_negative=True).is_value_admissible(-5)`, and a natural param's, returned `True` | `False`; a positive domain also refuses `0` | `test_a_natural_domain_admits_only_its_own_values`, `test_nat_param_is_value_admissible_gates_on_sign` |
  | `natural.has_feasible_value((x <= -1,), x)` was `SATISFIED`; `integer.compute_feasibility_subset((), x, natural, (), y)` was `SATISFIED`; `integer.is_value_set_subset(natural)` was `True` | `VIOLATED`, `VIOLATED`, `False` | `test_domain_questions_fold_in_the_domain_s_restriction` |
  | assigning `-1` to a natural param raised "violates constraint" | it raises "is not admissible" | `test_assignment_payload_rejects_only_a_provable_violation` |
  | a numeric procedure with a Python-defined domain on the other side asked it only its sort and values | it also asks its implied constraints | the Rust custom stories |
  | the `IntegerDomain` and `IntervalIntegerDomain` docstrings said `non_negative` does not change admissibility | they say every domain-level procedure respects it | none |

**R2-038c.** `finite_values` takes the side's constraints: with an in-set
constraint present, a permutation domain enumerates `in_set_candidates`
(held by every in-set and by no not-in-set constraint) filtered by
`is_permutation`, and otherwise every permutation as before; feasibility
and the finite subset both go through it. A valid value is a member of
every in-set constraint, so the answers cannot change, which
`permutation_questions_agree_with_brute_force` checks for n up to 5
against every permutation. `a_permutation_param_with_a_singleton_in_set_decides_at_n_10`
decides feasibility, infeasibility and a subset at n = 10 in well under
its 100 ms bound; at the base it did not finish within the five-minute
limit the check ran under. **Behavior change:** none.

**R2-028.** Tests only; every new test passed at its first run, so no new
finding.
- **Where.** The param rules are in a new `tests/it/param/decision_rule_stories.rs`.
  The constraint rules are in a new `tests/it/constraint/decision_rule_stories.rs`,
  with one `mod` line in `tests/it/constraint.rs`: `tests/it/constraint/*`
  is Track B's (§I.7.1), so the edit is a new file rather than changes to
  `constraint_properties.rs`, and the opaque-member membership property is
  a property of its own beside the existing one, not a widening of it.
- **The three probes, adopted** from the TYP scratch crate, rewritten for
  Track A's constructors (`Sign`, `ZeroInclusion`, `Inclusivity`) and for
  `checked_*` returning `Result`: the interval hull of `+`, `-`, `*`,
  reversed `-` and negation over unbounded, natural and exclusive operands
  (500 cases; a spec the builder refuses is rejected, not skipped); finite
  union and intersection, and integer intersection with a natural side;
  and numeric feasibility and subset against the real solver, gated by
  `real_solver()` as the z3 properties are. Counted with a probe, not
  committed: of 300 cases, 264 feasibility and 235 subset answers are
  decided (the audit's "about 85%").
- **Rstests:** `x <= 5` and `5 >= x` (and `x >= 1`, `1 <= x`) bound an
  interval param alike under `checked_neg` and `checked_add`; a context's
  registry makes a registered native constant's identifier known, and
  refused as a param's variable; categorical subsets (order, more
  categories, disjoint, against an ordinal, type-strict `1` against
  `True`); assignment equivalence for every `Value` kind, the zeros equal,
  a NaN equal to nothing but `==` itself, `1` against `True` in a tuple
  and a frozen set; opaque membership alone, colliding keys, inside a
  tuple (by position) and a frozen set (in any order).
- **Properties:** opaque members against a type-strict reference;
  rebinding an identifier keeps its first position with the last value;
  a system with one member on an unbound identifier is undecided unless a
  decided member is violated.

**R2-046b.** Tests only; every new test passed at its first run, so no new
finding. One helper, `support::serde::check_serde_round_trip` (a new
support module), checks JSON and postcard round trips to an equal value
and that the decoded JSON re-encodes to the same text.
- `a_domain_round_trips_through_serde` (`param/serde_stories.rs`): the six
  built-in kinds, the finite ones over members of kind int, bool, str,
  tuple and frozen set, nested two deep, each built through its
  constructor (a set the constructor refuses is filtered out).
- `a_type_round_trips_through_serde` (`types/serde_stories.rs`): numerical
  types over primitives and templates with and without widths, shapes of
  sums and products of literals and shape variables with wildcards, and
  index types; the data type round-trips on its own too.
- `a_table_built_through_the_checked_api_round_trips`
  (`symbol_table/serde_stories.rs`): tables built by random
  `add_namespace`/`remove_namespace`/`add_symbol`/`remove_symbol`
  sequences over `table_properties.rs`'s alphabet, with import, variable
  and function frames; it also checks that the checked API built no
  shadowing (R2-020). Before R2-020, a child added before its parent would
  have failed it, as F2-046 expected.

**R2-029c.** Tests only; every new test passed at its first run, so no new
finding.
- **Done before the rebase onto B (call).** The item "needs" Track B
  landed, for R2-010's reworded `TypeCheckError::Rule` text; B is still
  running, so the tables pin today's `Rule` text in two cases
  (`a_broken_rule_is_framed_by_the_root_and_the_sub_expression` and
  `each_body_check_failure_names_the_function`, whose `IllTyped` and
  `Unsupported` texts embed a `Rule`). They are this track's to update
  when it rebases onto B (§I.7.1, "who resolves"). After the rebase onto
  `03fb9e4` they pass unchanged: B's `Bounded` text differs from the full
  text only past 64 node occurrences, and these expressions are small.
- **Tables** (through Track A's `support/error_text.rs`, `to_string()` and
  the `source()` type by downcast): `types/error_text_stories.rs`, every
  variant of `PromotionError`, `LiteralTypeError` (the float and integer
  `Incompatible` texts both), `TemplateWidthError` (built through
  `with_widths`, its constructors being crate-private) and
  `UnificationError` (each operation's text, the width mismatch of a weak
  type, `Extension` and `Substitution` with their sources);
  `types/checking/error_text_stories.rs`, `TypeCheckError` (a rule at the
  root and below it, an unsupported rule, a deferred unknown call and a
  failing lookup, both with the lookup's error as source),
  `CallTargetError`, `SignatureError` and every `BodyCheckError`, the
  failures produced by `check_function_body` and matched field by field;
  `symbol_table/error_text_stories.rs`, every `SymbolTableError` and the
  text of every `Violation` (which implements no `Error`).
- **Tightened:** `body_stories.rs`'s two `{ .. }` matches (the labels, the
  core type, the sort, the callee's name); `extension_stories.rs`'s width
  mismatch (the template, its widths, the actual type) and the default
  data-type bind (the operation and which side holds the part);
  `unification_stories.rs`'s `is_some()` now compares the bound data type.
- **Small stories:** tables of equal sizes that differ in a namespace or a
  symbol name, both directions; `from_bindings` holds the three tables and
  equals the `with_*` build; an index type's substitution keeps its handle
  unless a bound changes. The default type rule's `TypeMismatch`
  (`unify.rs:366` at the audit) was already pinned by
  `an_extension_without_rules_takes_the_default_rules`.

**After the rebase onto Tracks D and B.** The maintainer rebased
`fix/c-types-param` onto `dev-rust` at `03fb9e4` (Tracks A, D and B
landed); the checklist's hashes are the rebased ones. The resolutions:
`types/checking/error.rs` imports B's `Bounded` beside the checker's
imports without R2-019's `FunctionName`, and `checker_stories.rs` keeps
B's `a_rule_error_on_a_dag_displays_in_bounded_size` followed by this
track's two DAG stories. The per-commit gates pass on the rebased head
(`cargo test --workspace` 4,924, `--all-features` 4,960) with no
fix-forward.

**R2-008.**
- **The module.** `expression/literal/exact.rs` (crate-private, one
  `pub(crate) mod exact;` line in Track B's `literal.rs`, and a
  `pub(crate) use` of `ExactNumber` and `Rational` in `expression.rs`)
  holds `Rational` (always in lowest terms with a positive denominator,
  ordered by cross-multiplication), `Rational::of_f64` (exact from the
  IEEE-754 decomposition, `None` for a NaN or an infinity),
  `Rational::to_f64_exact` (the float equal to the rational, if any,
  subnormals included), `Rational::to_decimal` (where the expansion ends
  and the exponent is within the bound), `Decimal::to_rational`,
  `ExactNumber` (the rationals and the two infinities, in order), and
  `LiteralValue::exact_value`/`exact_cmp` (a Boolean as `0` or `1`, a NaN
  ordered against nothing). Powers of ten and two are computed by squaring
  over a `u64` exponent, so no conversion of an exponent can fail or fall
  back.
- **The copies deleted:** `param/interval.rs`'s `Fraction` and `Extended`
  (`check_bounds_are_ordered` calls `exact_cmp`), `param/value.rs`'s
  `compare_int_with_float` (the ordinal order compares `ExactNumber`s),
  `solver/smt/lower.rs`'s `gcd`, `rationalize_float` and
  `rationalize_decimal`, `expression/literal/decimal.rs`'s `split_float`
  (`to_f64_exact` is now `to_rational().to_f64_exact()`), and in the
  binding `solver/sympy/lower.rs`'s decimal-to-rational and
  `solver/sympy/lift.rs`'s `split_off` and `exact_decimal` (Track D's
  moved files, the call sites the spec names).
- **The public surface, for the binding and Track E's R2-045 (call):**
  - `Decimal::MAX_EXPONENT_MAGNITUDE: u32 = 10_000`;
  - `Decimal::from_parts(coefficient: BigInt, exponent: i64) ->
    Result<Decimal, DecimalPartsError>`: the value `coefficient *
    10^exponent`, normalized as parsing normalizes (trailing zeros move
    into the exponent, zero is `0 * 10^0`), refusing a negative
    coefficient (`DecimalPartsError::NegativeCoefficient`; a decimal is
    non-negative, its sign is a negation around it) and a normalized
    exponent beyond the bound in magnitude
    (`DecimalPartsError::ExponentOutOfRange { exponent }`, the exponent
    given, including one whose normalization would overflow `i64`). For a
    Python `decimal.Decimal`'s `as_tuple()`, R2-045 passes the digits as a
    `BigInt` and the exponent, handling the sign as a negation;
  - `DecimalPartsError`, `#[non_exhaustive]`, exported as
    `fhy_core::expression::DecimalPartsError`;
  - `Decimal::to_rational_parts(&self) -> (BigInt, BigInt)`, lowest terms,
    positive denominator;
  - `Decimal::from_rational_parts(numerator, denominator) ->
    Option<Decimal>`, added beyond the spec's two so the SymPy lifting
    keeps no rational-to-decimal copy: the decimal of the magnitude, or
    `None` for a zero denominator, an expansion that does not end, or an
    exponent beyond the bound.
- **The bound's premise (spec proved wrong).** The spec says the text
  grammar implies the bound for any accepted literal; it does not: the
  grammar has no length limit, so `"0." + 20,000 zeros + "1"` parses with
  exponent `-20,001`. What the grammar implies is that a parsed exponent is
  at most the text's length, so its cost is linear in the input. So
  `FromStr` keeps accepting every text it accepted (no behavior change),
  and the bound holds where parts come from outside the grammar,
  `from_parts`, as the spec places it; `Rational::to_decimal`, and so
  `from_rational_parts`, stay within it too. The exact conversions need no
  bound for correctness, since they never truncate.
- **Tests:** unit properties in `exact.rs` (`of_f64` round-trips every
  finite `f64` bit pattern through `to_f64_exact`; `exact_cmp` agrees with
  a cross-multiplication oracle that decomposes floats through
  `integer_decode` and decimals through their text; a decimal round-trips
  through its rational), rstests of both conversions' edges (the least
  subnormal, `f64::MAX`, values no float equals), and
  `tests/it/param/interval_stories.rs` (new):
  `a_decimal_bound_with_a_huge_exponent_is_refused_not_misordered` through
  `from_parts`, with bounds at the limit ordered exactly against floats and
  a `10^10000 - 1` integer, and the normalization and refusal stories. The
  existing lowering, interval and SymPy stories pass unchanged.
- **Behavior change:** none through the grammar. A decimal beyond the
  bound can only come from `from_parts`, which refuses it. In the SymPy
  lifting, a rational whose decimal would need an exponent beyond 10,000
  now lifts as the quotient `n / d` instead of a decimal literal; no
  Python test reaches one.

### Track E notes

(none yet)
