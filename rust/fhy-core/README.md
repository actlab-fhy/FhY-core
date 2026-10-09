# fhy-core

[![crates.io](https://img.shields.io/crates/v/fhy-core.svg)](https://crates.io/crates/fhy-core)
[![docs.rs](https://img.shields.io/docsrs/fhy-core)](https://docs.rs/fhy-core)

Core IR building blocks for the [*FhY*](https://github.com/actlab-fhy)
compiler: identifiers and interned vocabularies, diagnostics and
provenance, a symbolic expression IR with pattern rewriting, tree
traversals, a compiler-pass framework, a solver with pluggable backends,
constraints, parameters, search spaces, partial orders and lattices, an IR
type system, and a symbol table.

The crate has no Python dependency. It is also the implementation behind
the [`fhy_core`](https://pypi.org/project/fhy_core/) Python package, whose
extension module is built from
[`fhy-core-py`](https://crates.io/crates/fhy-core-py). For `identifier` and
`interned`, which are defined in both languages, Rust matches Python's
behavior and a golden corpus checks it. Everywhere else the Rust behavior
is the definition.

## Usage

```toml
[dependencies]
fhy-core = "0.2"
```

The MSRV is 1.85.

```rust
use std::collections::HashMap;

use fhy_core::expression::Expression;
use fhy_core::expression::evaluate::{Evaluator, Scalar};
use fhy_core::expression::registry::FunctionRegistry;
use fhy_core::identifier::Identifier;

let (s, t) = (Identifier::new("s"), Identifier::new("t"));
let y = Expression::from(t.clone());
let expression: Expression = (3 * Expression::from(s.clone()) + y.clone()) - y;

// Exact affine form, without a solver.
let form = expression.affine_form().expect("affine");
assert_eq!(form.to_string(), "(3 * s)");

let registry = FunctionRegistry::new();
let environment = HashMap::from([(s, Scalar::Int(4)), (t, Scalar::Int(-7))]);
let value = Evaluator::new(&registry)
    .evaluate(&expression, &environment)
    .expect("evaluates");
assert_eq!(value, Scalar::Int(12));
```

## Features

All features are off by default. None raises the MSRV. docs.rs builds with
`z3` and `ndarray` and marks the items each adds.

| Feature | Adds |
| :--- | :--- |
| `z3` | `solver::Z3Solver`, an `SmtSolver` backed by libz3 through the [`z3`](https://crates.io/crates/z3) crate |
| `ndarray` | `Prepared::evaluate_array`: evaluation over [`ndarray`](https://crates.io/crates/ndarray) 0.17 arrays, broadcast as NumPy broadcasts |
| `testing` | `search_space::testing`: conformance checks for `Variable` and `Alternative` implementations. Test-only, not stable API; enable it in `[dev-dependencies]`. |

### Linking z3

`z3-sys` links libz3 4.13.3 or newer, found through `pkg-config` by
default. To choose another method, enable it on your own `z3` dependency;
Cargo's feature unification applies it to this crate's:

- `vendored` builds libz3 from source;
- `gh-release` downloads a z3 release;
- `vcpkg` uses vcpkg.

To link an existing libz3, such as the one in the `z3-solver` Python
wheel, set `Z3_LIBRARY_PATH_OVERRIDE` to its directory. Set
`Z3_SYS_Z3_VERSION` too if neither `pkg-config` nor a `z3` on `PATH`
reports the version. At run time, put the same directory on the library
search path (`LD_LIBRARY_PATH` on Linux).

Without the feature, `solver::SmtLib2Process` drives a z3 executable
(`z3 -in`) or any other SMT-LIB2 solver (`cvc5 --lang=smt2`) over stdin
and stdout.

### Array evaluation

Under `ndarray`, each lane is computed exactly as scalar evaluation of that
lane's bindings would be, and the result is a new array in standard layout.
An `ArrayKernels` implementation can compute native built-ins over whole
arrays; `CoreKernels` uses the crate's per-lane kernels.

## Modules

Listed in layer order. A module depends only on modules in earlier rows.
Each public item has exactly one public path.

| Module | Contents |
| :--- | :--- |
| `identifier` | `Identifier`: a name hint and a process-unique id |
| `interned` | `Interned`, `InternRegistry`, `Canonical`: one canonical value per key |
| `foreign` | `Foreign`, the serialized form of a part defined by another crate (an extension type, a custom constraint or domain, an opaque value, a custom provenance, a search-space variable or alternative): its type id and its payload as text. `Part` holds such a part. |
| `error` | Errors shared across layers, such as `UnknownNameError` for name enums' `FromStr` |
| `described_tag` | `DescribedTag<K>`, an open vocabulary entry named by an `Identifier`, and the sealed `TagKind` |
| `value_domain` | `ValueDomain`: an open, hierarchical classification of the values an operation handles. Defaults: `ValueDomain::data()`, `ValueDomain::address()`. |
| `provenance` | `Position`, `Span`, `Provenance`. `Provenance::Custom` holds a `CustomProvenance` defined elsewhere. |
| `diagnostic` | `Diagnostic`, `Note`, `NoteKind`, `ValidationReport` |
| `op_attribute` | `OpAttribute`: an open tag for compiler operations. Defaults: `commutative()`, `associative()`, `pure()`, `elementwise()`. |
| `tree` | `NodeHandle`, `NodeIdentity`, the `Tree` trait, iterative walks (`walk_tree`) and memoized rewrites (`rewrite_tree`) over any tree-shaped IR |
| `term` | `AlphaRenaming`; the `AlphaEquivalence`, `FreeIdentifiers`, `Term` and `Binder` traits. `Binder` derives alpha equivalence, free identifiers and capture-avoiding substitution for a binding node. `is_mapping_alpha_equivalent_under` compares identifier-keyed maps. |
| `lattice` | `PartiallyOrderedSet` (order queries are bit tests; iteration is topological, ties broken by insertion order) and `Lattice` (meets, joins, missing bounds) |
| `expression` | `Expression`, node kinds, builders, literals (`BigInt`, `Decimal`, `Rational`), sorts, display, `BooleanScreen`, and `AffineForm` (exact affine forms via `Expression::affine_form`) |
| `expression::builtins` | Built-in functions and constants. Each constant has a fixed reserved identifier, so it means the same in every process. |
| `expression::registry` | `FunctionRegistry` of user functions, native functions and native constants; `FunctionRegistry::inline`, linear in the distinct nodes |
| `expression::evaluate` | `Evaluator`: constant folding (`fold`) and evaluation to `Scalar`s (`evaluate`) over `bool`, `i64`, `f64`, with checked integers and per-lane failures that a piecewise or connective discards when it does not need the lane |
| `expression::pattern` | `Pattern`, `Capture`, the `Rule` trait, `RewriteRule`, `apply_rewrite_rules` |
| `pass` | `CompilerPass`, `PassManager`, `FixpointPassGroup`, analyses (named by type or by `AnalysisId::of_identifier`), `Validator`, `PassRegistry`, `VerificationRegistry` |
| `expression::passes` | `RewriteRuleApplier`, `ExpressionPrettyFormatter`, `register_expression_passes` |
| `solver` | `Solver`: satisfiability, implication and validity through an `SmtSolver`, simplification through a `Simplifier`. Questions are type-checked, shapes SMT-LIB2 cannot state soundly are refused (`Hazard::find`), and each question is encoded as one `SmtScript`. Backends: `SmtLib2Process`, `Z3Solver` (feature `z3`), and `GroundSimplifier`, a pure-Rust simplifier over pluggable `SimplificationStrategy` rewrites that folds ground expressions exactly as SymPy would and declines anything else. `GroundWithFallback` chains it with another simplifier. |
| `types` | `CoreDataType` and its promotion orders, `TypeQualifier`, `DataType`, `Type`, template binding, substitution and unification into a `TypeUnificationEnvironment`. `TypeExtension` and `DataTypeExtension` admit types defined elsewhere. |
| `types::checking` | `TypeChecker` (synthesis and checking against `IdentifierTypes` and `CallTargets`), `check_function_body`, `check_all_function_bodies` |
| `constraint` | `EquationConstraint` (a Boolean expression, decided with the solver's simplifier) and `SetConstraint` (type-strict membership in a canonically ordered `MemberSet`), decided under `Bindings` into an `Outcome`. `OpaqueValue` for values only their producer can compare; `ConstraintObserver` explains undecided outcomes. |
| `symbol_table` | `SymbolTable`: namespaces with optional parents, lookup up the parent chain, no redefinition of an ancestor's symbol, and `violations` for missing or cyclic parents. Frames implement `Frame`; `SymbolFrame` is the built-in one (`ImportFrame`, `VariableFrame`, `FunctionFrame`). |
| `param` | `ParamDomain`: integers (`IntegerDomain`, `IntervalIntegerDomain` with interval arithmetic), reals, ordinal and categorical sets, permutations, or a `CustomDomain`. Type-strict admission, feasibility and subset questions, by enumeration or through the solver. `ParamObserver` explains undecided or weakened answers. |
| `stack` | `Stack`: LIFO, `pop`/`peek` return `None` when empty, iterates bottom to top |
| `scope` | `Scope`: lexical frames with shadowing lookup (`define`, `lookup`, `lookup_local`, `with_frame`, which restores depth even on panic). Popping the root frame is a `RootFramePopError`. |
| `search_space` | `Variable` and `Alternative` traits (with `PlainVariable`, `PlainAlternative`), `Choice`, `Space` with `Condition`s and `Forbidden` clauses, `Configuration` and `ConfigurationKey`; sampling, completion, enumeration, `Cardinality`, mutation and crossover; `Recorder`, oracles and `Trace`/`TraceKey` for step-wise search; `Measurement` and `non_dominated` |

### Serialization

Types derive serde in their own shapes; there is no type envelope. A type
that can hold a foreign part (`types`, `constraint`, `param`,
`provenance`, `symbol_table`, `search_space`) has a `wire` submodule whose
data type's `build` takes a `Resolve`r for those parts. The type's own
`Deserialize` refuses foreign parts (`NoForeign`). `search_space::wire`
adds `ResolverRegistry` to compose the resolvers of several crates.

Decoding an `Identifier` advances the id counter past its id, and decoding
a `Canonical<T>` interns its value. A decode that fails partway can leave
both effects behind.

## One copy per process

The identifier counter and each interned type's registry are
process-global statics. Both are append-only: an id is never reissued and
a canonical value is never replaced. A process must therefore hold exactly
one compiled copy of this crate. For Python, that means one extension
module: other *FhY* packages with Rust code build on `fhy-core-py` and
compile into a single combined module instead of linking their own copy.
All other state (pass, verification and function registries) is owned by
the caller.

## License

BSD-3-Clause. See
[LICENSE](https://github.com/actlab-fhy/FhY-core/blob/main/LICENSE).
