# fhy-core

Core IR building blocks for the [FhY](https://github.com/actlab-fhy) compiler, in Rust: identifiers, interned vocabularies, diagnostics and provenance, symbolic expressions with patterns and rewrite rules, tree traversals, a compiler-pass framework, a solver with pluggable backends, constraints, partial orders and lattices, and the IR type system.

This crate is the Rust implementation of the `fhy_core` Python package, which requires it: the package's extension module, built from the `fhy-core-py` binding crate, issues its identifier ids and backs its interned tags, diagnostics, provenance and expressions. Where a concept is defined in both languages (`identifier`, `interned`), the Rust behavior matches Python's and a golden corpus checks it. Elsewhere Rust defines the behavior.

## Modules

Each module depends only on the modules listed before it, except that `tree`, `term` and `lattice` are independent of each other, `expression` and `pass` are independent of each other, `expression::passes` joins them, `solver` and `types` depend on `expression` and not on `pass`, and `constraint` depends on `solver` and not on `pass`. Each public item has exactly one public path.

- `identifier`: `Identifier`, a name hint paired with a process-unique id.
- `interned`: `Interned`, `InternRegistry` and `Canonical`, which keep one canonical value per key.
- `described_tag`: `DescribedTag<K>`, an open vocabulary entry named by an `Identifier`, and the sealed `TagKind` of its vocabularies.
- `diagnostic`: `Diagnostic`, `Note` and its `NoteKind`, and `ValidationReport`.
- `op_attribute`: `OpAttribute`, an open tag attached to compiler operations. The shipped defaults are `OpAttribute::commutative()`, `OpAttribute::associative()`, `OpAttribute::pure()` and `OpAttribute::elementwise()`.
- `value_domain`: `ValueDomain`, an open, hierarchical classification of the values an operation handles. The shipped defaults are `ValueDomain::data()` and `ValueDomain::address()`.
- `provenance`: `Position`, `Span` and `Provenance`, where a value came from.
- `tree`: `NodeHandle` and its `NodeIdentity`, from an `Arc` or from a pointer to a foreign object, the `Tree` trait, and iterative walks (`walk_tree`) and memoized rewrites (`rewrite_tree`) over any tree-shaped IR.
- `term`: `AlphaRenaming`, the correspondence of identifiers between two terms under comparison, and the traits of terms: `AlphaEquivalence`, `FreeIdentifiers`, `Term`, and `Binder`, which derives alpha equivalence, free identifiers and capture-avoiding substitution for a node that binds identifiers; a binder list that repeats an identifier pairs with none. `is_mapping_alpha_equivalent_under` compares identifier-keyed maps.
- `lattice`: `PartiallyOrderedSet`, a partial order over any hashable elements whose order queries are bit tests and whose iteration is topological with insertion order breaking ties, and `Lattice`, its meets, joins and missing bounds.
- `expression`: `Expression`, its node kinds, builders, literals and analyses, and `BooleanScreen`.
  - `expression::builtins`: the catalogue of built-in functions and constants. Each constant has a fixed identifier from the reserved block (`BuiltinConstant::identifier`), so a reference to it means the same in every process.
  - `expression::registry`: the owned `FunctionRegistry` of user functions (`FunctionDefinition`), native functions (`NativeFunction`) and constants (`NativeConstant`), which implements the screen's `SortLookup`, and `FunctionRegistry::inline`, which replaces calls of composed built-ins and user functions by their bodies in time linear in the distinct nodes.
  - `expression::evaluate`: the `Evaluator`, which folds native calls with literal arguments and constant references (`Evaluator::fold`), and evaluates an expression to `Scalar`s over an environment (`Evaluator::evaluate`), in three domains (`bool`, `i64`, `f64`) with checked integers and per-lane failures that a piecewise or a connective discards where it does not need the lane.
  - `expression::pattern`: `Pattern`, `Capture`, the `Rule` trait and `RewriteRule`, and `apply_rewrite_rules`.
  - `expression::passes`: `RewriteRuleApplier`, `ExpressionPrettyFormatter` and `register_expression_passes`.
- `pass`: `CompilerPass`, `PassManager`, `FixpointPassGroup`, analyses, `Validator`s, the owned `PassRegistry`, and the owned `VerificationRegistry`, which builds a pipeline's verifier from the validators registered for the kinds of IR it applies to, looked up along a lineage of kinds (`VerifierId` names a registration by its type or, for one a language binding defines, by an object's address). An analysis is named by its type, or, when no Rust type names it, such as one a language binding defines, by an `Identifier` (`AnalysisId::of_identifier`, `PassContext::analysis_by_id`). `PassContext::with_detached_analyses` lends a hook's code an owned `DetachedAnalyses` handle to the run's cache.
- `solver`: `Solver`, which answers satisfiability, implication and universal-validity questions with an `SmtSolver` backend, and simplification with a `Simplifier` backend. It checks each question's symbol types and Boolean positions, refuses the shapes SMT-LIB2 cannot state in this crate's semantics (`Hazard::find`), and encodes a logical question as one `SmtScript`, lowered to SMT-LIB2 with the crate's semantics. `SmtLib2Process` drives any SMT-LIB2 executable, such as `z3 -in` or `cvc5 --lang=smt2`, over its standard input and output, and, with the `sympy` feature, `SympySimplifier` simplifies with SymPy.
- `constraint`: constraints over identifiers, decided under `Bindings` into an `Outcome`. An `EquationConstraint` is a Boolean expression, decided with the solver's simplifier; a `SetConstraint` decides type-strict membership of one identifier's value in a `MemberSet`, whose members are in one canonical order. A value only its producer can compare is an `Opaque` value behind the `OpaqueValue` trait, and an `Observer` receives the `Event`s that explain undecided outcomes.
- `types`: the IR type system. `CoreDataType` with its promotion orders and the types literals resolve to, `TypeQualifier`, the `DataType`s (primitive, `TemplateDataType`, or an extension) and `Type`s (`NumericalType`, `IndexType`, or an extension), and template binding, substitution and unification into a `TypeUnificationEnvironment`. `TypeExtension` and `DataTypeExtension` let types defined elsewhere take part in each operation.
  - `types::checking`: `TypeChecker`, which synthesizes an expression's type and qualifier or checks it against an expected type over `IdentifierTypes` and `CallTargets` lookups, and `check_function_body` and `check_all_function_bodies`, which hold function bodies to their declared result sorts.
- `symbol_table`: `SymbolTable`, namespaces in insertion order, each naming an optional parent and mapping its symbols to frames of any type implementing `Frame`. A lookup walks up the parents, an inner namespace never redefines an outer symbol, and `violations` reports missing and cyclic parents and frames that name another symbol. `SymbolFrame` is the built-in frame: an `ImportFrame`, a `VariableFrame` or a `FunctionFrame`, with its `FunctionKeyword`. It depends on `types`.

## One copy per process

The identifier id counter and each interned type's registry are process-global statics, and both are append-only: an id is never reissued and a canonical value is never replaced. A process must therefore hold exactly one compiled copy of this crate. Link it into one Python extension module, and compile other FhY packages' Rust code into that same module rather than into a second one. Everything else, including the pass registry, the verification registry and the function registry, is an owned value.

## Using it

```toml
[dependencies]
fhy-core = "0.2"
```

The minimum supported Rust version is 1.85.

## The `z3` feature

`fhy_core::solver::Z3Solver`, a backend that decides scripts with the z3 library through the [`z3`](https://crates.io/crates/z3) crate, is behind the off-by-default `z3` feature:

```toml
[dependencies]
fhy-core = { version = "0.2", features = ["z3"] }
```

The `z3-sys` crate links libz3 4.13.3 or newer. By default it finds a system libz3 through `pkg-config`. To choose another link method, enable it on your own `z3` dependency, which Cargo's feature unification applies to this crate's: `vendored` builds libz3 from source, `gh-release` downloads a z3 release, and `vcpkg` uses vcpkg. To link a libz3 you already have, such as the one the `z3-solver` Python wheel ships, set `Z3_LIBRARY_PATH_OVERRIDE` to its directory, `Z3_SYS_Z3_VERSION` to its version when neither `pkg-config` nor a `z3` executable on `PATH` reports it, and the library search path of your platform (`LD_LIBRARY_PATH` on Linux) to the same directory when running. The feature does not raise the minimum Rust version. docs.rs builds the default features.

Without the feature, `SmtLib2Process` drives a z3 executable (`z3 -in`) or any other SMT-LIB2 solver (`cvc5 --lang=smt2`) over its standard input and output.

## The `ndarray` feature

`Prepared::evaluate_array`, which evaluates an expression over [`ndarray`](https://crates.io/crates/ndarray) arrays bound to its identifiers, broadcast together as NumPy broadcasts, is behind the off-by-default `ndarray` feature:

```toml
[dependencies]
fhy-core = { version = "0.2", features = ["ndarray"] }
```

Every lane is computed as the scalar evaluation of that lane's bindings is, and the result is a new array in the standard layout. An `ArrayKernels` implementation can compute native built-ins over whole arrays instead of the crate's own per-lane kernels; `CoreKernels` uses the crate's. The feature's API names `ndarray` 0.17's types, and it does not raise the minimum Rust version. docs.rs builds the default features.

## The `sympy` feature

`fhy_core::solver::SympySimplifier`, a `Simplifier` that lowers an expression to [SymPy](https://www.sympy.org), simplifies it with `sympy.simplify`, and lifts the result back, is behind the off-by-default `sympy` feature:

```toml
[dependencies]
fhy-core = { version = "0.2", features = ["sympy"] }
```

The feature adds [`pyo3`](https://crates.io/crates/pyo3) 0.29, without its default features and without `auto-initialize`, and pyo3 is part of the feature's public API: `lower`, `lift` and the SymPy-level operations take and return pyo3 types. A build holds one pyo3, since `pyo3-ffi` links `python`, so a crate that also uses pyo3 uses 0.29.

The backend needs a Python interpreter with the `sympy` package at run time, and never starts one on its own. Inside a Python process, such as an extension module, it attaches to the running interpreter. A Rust program starts one with `SympySimplifier::with_embedded_python()`, or `pyo3::Python::initialize()`, before the first simplification; without an interpreter, or without SymPy, each operation fails with `SympyUnavailableError`, and `SympySimplifier::load` tells whether simplification can run. An embedded interpreter is the Python this crate was linked against at build time: pyo3 finds it through `PYO3_PYTHON`, an active virtualenv, or `python3` on `PATH`, and needs its shared libpython. At run time the program finds that libpython on the library search path of its platform (`LD_LIBRARY_PATH` on Linux, when it is not in a system directory), and the embedded interpreter finds SymPy on its default path or on `PYTHONPATH`; it does not read a virtualenv's `pyvenv.cfg`, so a virtualenv's `site-packages` goes on `PYTHONPATH`. The backend loads a small Python prelude of the SymPy classes and hooks only Python code can define, once per interpreter, as the module `_fhy_core_sympy`. The feature does not raise the minimum Rust version. docs.rs builds the default features.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
