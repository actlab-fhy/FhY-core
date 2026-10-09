# *FhY* Core

[![PyPI version](https://img.shields.io/pypi/v/fhy_core.svg)](https://pypi.org/project/fhy_core/)
[![Python versions](https://img.shields.io/pypi/pyversions/fhy_core.svg)](https://pypi.org/project/fhy_core/)
[![crates.io](https://img.shields.io/crates/v/fhy-core.svg)](https://crates.io/crates/fhy-core)
[![docs.rs](https://img.shields.io/docsrs/fhy-core)](https://docs.rs/fhy-core)
[![CI](https://github.com/actlab-fhy/FhY-core/actions/workflows/python-package.yml/badge.svg)](https://github.com/actlab-fhy/FhY-core/actions/workflows/python-package.yml)
[![codecov](https://codecov.io/gh/actlab-fhy/FhY-core/branch/main/graph/badge.svg)](https://codecov.io/gh/actlab-fhy/FhY-core)

*FhY* Core is the shared foundation of the *FhY* compiler and its tooling.
It provides identifiers and interned vocabularies, a symbolic expression IR
with pattern rewriting, a solver interface with pluggable backends,
constraints and parameters, search spaces, an IR type system, a symbol
table, a compiler-pass framework, diagnostics, and serialization.

Most of the implementation is in Rust. The Python package binds it and adds
what is specific to Python: the `Serializable` framework, mixins for
Python-defined IR nodes, and hooks that let Python subclasses take part in
Rust-backed algorithms.

| Package | Registry | Contents |
| :--- | :--- | :--- |
| `fhy_core` | [PyPI](https://pypi.org/project/fhy_core/) | Python package. Ships the compiled extension `fhy_core._rs` and requires it. |
| `fhy-core` | [crates.io](https://crates.io/crates/fhy-core) | Pure-Rust library. No PyO3 or Python dependency. |
| `fhy-core-py` | [crates.io](https://crates.io/crates/fhy-core-py) | PyO3 bindings, as a library. For downstream crates that build a combined extension module. |

## Installation

### Python

```bash
pip install fhy_core
```

Wheels include the extension. Building from an sdist compiles it and needs
Rust 1.85 or newer. Python 3.11 through 3.14 are supported.

The solver backends and the NumPy evaluator are optional extras. Each is
imported only when a call needs it.

| Extra | Installs | Enables |
| :--- | :--- | :--- |
| `z3` | `z3-solver` | Satisfiability, implication and validity queries through z3 |
| `sympy` | `sympy` | The SymPy simplifier, and with it validation of params under equation constraints |
| `solvers` | both of the above | |
| `numpy` | `numpy` | `evaluate_expression_with_numpy` |

A call that needs a missing backend raises `SolverBackendUnavailableError`,
which names the extra. Finite domains, set constraints and enumeration work
without any backend. `SmtLib2ProcessSolver` needs no Python package: it
drives any SMT-LIB2 executable, such as `z3 -in` or `cvc5 --lang=smt2`.

### Rust

```toml
[dependencies]
fhy-core = "0.2"
```

The MSRV is 1.85. Two features are off by default:

| Feature | Adds |
| :--- | :--- |
| `z3` | `solver::Z3Solver`, which links libz3 (4.13.3 or newer) through the `z3` crate |
| `ndarray` | `Prepared::evaluate_array`, evaluation over broadcast `ndarray` arrays |

Without `z3`, `solver::SmtLib2Process` drives an external solver over
stdin/stdout. See the [crate README](https://github.com/actlab-fhy/FhY-core/blob/main/rust/fhy-core/README.md) for linking
options.

## API

Most concepts exist in both languages under parallel paths. The Python
module is a binding over the Rust module unless the last column says
otherwise.

| Area | Python | Rust (`fhy_core::`) | Contents |
| :--- | :--- | :--- | :--- |
| Identifiers | `identifier` | `identifier` | `Identifier`: a name hint and a process-unique id |
| Interned tags | `traits.interned`, `op_attribute`, `value_domain` | `interned`, `described_tag`, `op_attribute`, `value_domain` | Hash-consed vocabularies; `OpAttribute` (commutative, pure, ...) and `ValueDomain` (data, address) |
| Errors | `error` | `error`, plus one error type per module | `FhYError` hierarchy and registration of downstream error types |
| Diagnostics | `diagnostic` | `diagnostic` | `Diagnostic`, `Note`, `ValidationReport` |
| Provenance | `provenance` | `provenance` | Source positions and spans; custom provenance kinds |
| Expressions | `symbolic.expression` | `expression` | Expression IR (literals, identifiers, unary, binary, piecewise, calls), sorts, pretty printing, affine forms |
| Functions | `symbolic.expression.registry`, `.builtins` | `expression::registry`, `expression::builtins` | Function registry, native functions and constants, inlining |
| Evaluation | `symbolic.expression.passes` (`evaluate`, `numpy`) | `expression::evaluate` | Constant folding; scalar and array evaluation over `bool`, `i64`, `f64` |
| Patterns | `symbolic.expression.pattern` | `expression::pattern`, `expression::passes` | Pattern algebra, captures, rewrite rules, bottom-up rule application |
| Solver | `symbolic.solver` | `solver` | Satisfiability, implication, validity and simplification over pluggable SMT and simplifier backends; `GroundSimplifier` |
| Constraints | `symbolic.constraint` | `constraint` | Equation and set-membership constraints, constraint systems |
| Parameters | `symbolic.param` | `param` | Real, integer, ordinal, categorical and permutation domains; interval arithmetic, union, intersection |
| Search spaces | `search_space` | `search_space` | Variables, choices, conditions, configurations; oracles, traces, measurements, Pareto filtering |
| Types | `types`, `types.checking` | `types`, `types::checking` | Data types, promotion, unification, expression type checking |
| Symbol table | `symbol_table` | `symbol_table` | Namespaces with parent lookup and symbol frames |
| Passes | `pass_infrastructure` | `pass`, `tree` | Compiler passes, pass and analysis managers, fixpoint groups, validators, verification registry; tree walks and rewrites |
| Terms | `term` | `term` | Alpha equivalence, binders, capture-avoiding substitution, derived equivalence for dataclasses |
| Orders | `lattice`, `utils.poset` | `lattice` | Partially ordered sets and lattices |
| Stack, scope | `utils.stack`, `utils.scope` | `stack`, `scope` | LIFO stack, lexical scope with shadowing. Implemented separately in each language with the same behavior. |
| Serialization | `serialization` | serde on each type; `foreign` | Python: `Serializable` with dict, JSON and binary formats. Rust: plain serde; `Foreign` carries parts defined elsewhere. |
| IR traits | `traits` | — | Python only: `Frozen`, `Equal`, `Orderable`, `HasType`, `HasOperands`, `HasResults`, `Rewritable`, `Visitable`, `Verifiable`, `Canonicalizable` |
| Utilities | `logger`, `utils` | — | Python only: logging setup, dict and numeric helpers |

Python paths are relative to `fhy_core.`. The full Python-to-Rust mapping is
in [CONTRIBUTING.md](https://github.com/actlab-fhy/FhY-core/blob/main/CONTRIBUTING.md#module-map).

### Behavior worth knowing

- Solver queries are three-valued. `True` and `False` are answers; `None`
  means undecided, either because the backend returned unknown or because
  the query contains something the solver cannot lower soundly (a native
  constant, a non-finite literal, a partial operation). The `assert_*`
  variants raise `UndecidableError` instead.
- Values compare type-strictly in constraints and params: `True`, `1` and
  `1.0` are three distinct members.
- Serialization writes the V2 wire format, which is the Rust core's serde
  shape. A Rust-backed value's `to_json()` is byte-identical to what the
  Rust crate writes. The V1 envelope (`{"__type__": .., "__data__": ..}`)
  is deprecated and is removed in 0.3.0. Convert stored payloads with
  `python -m fhy_core.serialization_upgrade old.json > new.json`.
- `evaluate_expression_with_numpy` releases the GIL and reads `float64`
  inputs in place. Do not write to an input array from another thread
  during the call.
- Identifier ids and interned registries are process-global. A process
  must load exactly one compiled copy of the Rust core, so a downstream
  package with Rust code builds on `fhy-core-py` and links into one
  combined extension module. See the
  [`fhy-core-py` README](https://github.com/actlab-fhy/FhY-core/blob/main/rust/fhy-core-py/README.md).

## Examples

### Python

Piecewise expressions:

```python
from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression import IdentifierExpression, piecewise

x = IdentifierExpression(Identifier("x"))
grade = piecewise((x >= 90, 4), (x >= 80, 3), otherwise=0)
```

Multi-variable constraints:

```python
from fhy_core.identifier import Identifier
from fhy_core.symbolic.constraint import EquationConstraint, create_constraint_system
from fhy_core.symbolic.expression import IdentifierExpression
from fhy_core.symbolic.symbol_type import SymbolType

x, y = Identifier("x"), Identifier("y")
system = create_constraint_system(
    EquationConstraint(IdentifierExpression(x) < IdentifierExpression(y))
)

system.is_satisfied_with_bindings({x: 3, y: 5})                      # True
system.check_satisfiability({x: SymbolType.INT, y: SymbolType.INT})  # ConstraintOutcome.SATISFIED (needs z3)
```

Parameter arithmetic and set operations:

```python
from fhy_core.symbolic.param import (
    create_categorical_param,
    create_interval_integer_param_between,
    create_intersection_param,
    create_union_param,
)

scaled = create_interval_integer_param_between(0, 10) * 2  # [0, 20]
precisions = create_union_param(
    create_categorical_param(["fp16", "fp32"]), create_categorical_param(["bf16"])
)
overlap = create_intersection_param(
    create_interval_integer_param_between(0, 10),
    create_interval_integer_param_between(5, 15),
)
```

Solver queries:

```python
from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression import IdentifierExpression
from fhy_core.symbolic.solver import check_expression_satisfiability
from fhy_core.symbolic.symbol_type import SymbolType

z = Identifier("z")
check_expression_satisfiability(IdentifierExpression(z) > 0, {z: SymbolType.INT})  # True
```

### Rust

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

## Building from source

Building needs [uv](https://docs.astral.sh/uv/) and a Rust toolchain
(`rust-toolchain.toml` pins the channel for rustup).

```bash
git clone https://github.com/actlab-fhy/FhY-core.git
cd FhY-core
uv sync                    # dev environment; compiles fhy_core._rs with maturin
uv run pytest              # Python tests
cargo test --workspace     # Rust tests
```

`uv sync --no-default-groups --extra solvers` installs only the runtime
dependencies and the solver backends. `uv sync` installs the package in
editable mode, so Python edits take effect immediately. Rust edits are
picked up by the next `uv sync` or `uv run`, which rebuild the extension.

Development workflow, test lanes, CI and the Rust architecture are covered
in [CONTRIBUTING.md](https://github.com/actlab-fhy/FhY-core/blob/main/CONTRIBUTING.md).

## License

BSD-3-Clause. See [LICENSE](https://github.com/actlab-fhy/FhY-core/blob/main/LICENSE).
