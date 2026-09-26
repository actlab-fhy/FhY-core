# fhy-core

Core IR building blocks for the [FhY](https://github.com/actlab-fhy) compiler, in Rust: identifiers, interned vocabularies, diagnostics and provenance, symbolic expressions with patterns and rewrite rules, tree traversals, and a compiler-pass framework.

This crate is the Rust implementation of the `fhy_core` Python package, which requires it: the package's extension module, built from the `fhy-core-py` binding crate, issues its identifier ids and backs its interned tags, diagnostics, provenance and expressions. Where a concept is defined in both languages (`identifier`, `interned`), the Rust behavior matches Python's and a golden corpus checks it. Elsewhere Rust defines the behavior.

## Modules

Each module depends only on the modules listed before it, except that `expression` and `pass` are independent of each other and `expression::passes` joins them. Each public item has exactly one public path.

- `identifier`: `Identifier`, a name hint paired with a process-unique id.
- `interned`: `Interned`, `InternRegistry` and `Canonical`, which keep one canonical value per key.
- `described_tag`: `DescribedTag<K>`, an open vocabulary entry named by an `Identifier`, and the sealed `TagKind` of its vocabularies.
- `diagnostic`: `Diagnostic`, `Note` and its `NoteKind`, and `ValidationReport`.
- `op_attribute`: `OpAttribute`, an open tag attached to compiler operations. The shipped defaults are `OpAttribute::commutative()`, `OpAttribute::associative()`, `OpAttribute::pure()` and `OpAttribute::elementwise()`.
- `value_domain`: `ValueDomain`, an open, hierarchical classification of the values an operation handles. The shipped defaults are `ValueDomain::data()` and `ValueDomain::address()`.
- `provenance`: `Position`, `Span` and `Provenance`, where a value came from.
- `tree`: `NodeHandle` and its `NodeIdentity`, from an `Arc` or from a pointer to a foreign object, the `Tree` trait, and iterative walks (`walk_tree`) and memoized rewrites (`rewrite_tree`) over any tree-shaped IR.
- `expression`: `Expression`, its node kinds, builders, literals and analyses, and `BooleanScreen`.
  - `expression::builtins`: the catalogue of built-in functions and constants. Each constant has a fixed identifier from the reserved block (`BuiltinConstant::identifier`), so a reference to it means the same in every process.
  - `expression::registry`: the owned `FunctionRegistry` of user functions (`FunctionDefinition`), native functions (`NativeFunction`) and constants (`NativeConstant`), which implements the screen's `SortLookup`, and `FunctionRegistry::inline`, which replaces calls of composed built-ins and user functions by their bodies in time linear in the distinct nodes.
  - `expression::pattern`: `Pattern`, `Capture`, the `Rule` trait and `RewriteRule`, and `apply_rewrite_rules`.
  - `expression::passes`: `RewriteRuleApplier`, `ExpressionPrettyFormatter` and `register_expression_passes`.
- `pass`: `CompilerPass`, `PassManager`, `FixpointPassGroup`, analyses, `Validator`s and the owned `PassRegistry`. An analysis is named by its type, or, when no Rust type names it, such as one a language binding defines, by an `Identifier` (`AnalysisId::of_identifier`, `PassContext::analysis_by_id`). `PassContext::with_detached_analyses` lends a hook's code an owned `DetachedAnalyses` handle to the run's cache.

## One copy per process

The identifier id counter and each interned type's registry are process-global statics, and both are append-only: an id is never reissued and a canonical value is never replaced. A process must therefore hold exactly one compiled copy of this crate. Link it into one Python extension module, and compile other FhY packages' Rust code into that same module rather than into a second one. Everything else, including the pass registry and the function registry, is an owned value.

## Using it

```toml
[dependencies]
fhy-core = "0.2"
```

The minimum supported Rust version is 1.85.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
