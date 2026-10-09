//! Loading SymPy: the classes and values the backend reads from it, and
//! the prelude module.

use std::ffi::CString;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

use fhy_core::expression::builtins::BuiltinFunction;

use super::error::SympyUnavailableError;

/// The stem of the name the prelude is published under in `sys.modules`.
const PRELUDE_MODULE_STEM: &str = "_fhy_core_sympy";

/// The attribute of the prelude module that holds the hash of its source.
const PRELUDE_HASH_ATTRIBUTE: &str = "__fhy_core_prelude__";

/// The prelude's source.
const PRELUDE_SOURCE: &str = include_str!("prelude.py");

/// The 64-bit FNV-1a hash of the prelude's source.
const PRELUDE_HASH: u64 = fnv1a_64(PRELUDE_SOURCE.as_bytes());

/// Return the 64-bit FNV-1a hash of `bytes`.
const fn fnv1a_64(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET_BASIS;
    let mut index = 0;
    while index < bytes.len() {
        hash ^= bytes[index] as u64;
        hash = hash.wrapping_mul(PRIME);
        index += 1;
    }
    hash
}

/// Return the hash of the prelude's source as 16 hexadecimal digits, the
/// value of the prelude module's `__fhy_core_prelude__`.
fn prelude_hash_text() -> String {
    format!("{PRELUDE_HASH:016x}")
}

/// What the backend reads from SymPy and from its prelude, loaded once per
/// backend.
///
/// The functions the backend calls on a whole expression, `simplify` and
/// `piecewise_fold`, are read from `sympy` at each call instead, so a
/// caller that replaces them on the module reaches the backend.
pub(super) struct Handles {
    pub(super) sympy: Py<PyModule>,
    // The classes the walks dispatch on.
    pub(super) basic: Py<PyAny>,
    pub(super) expr: Py<PyAny>,
    pub(super) boolean: Py<PyAny>,
    pub(super) boolean_function: Py<PyAny>,
    pub(super) symbol: Py<PyAny>,
    pub(super) integer: Py<PyAny>,
    pub(super) float: Py<PyAny>,
    pub(super) rational: Py<PyAny>,
    pub(super) piecewise: Py<PyAny>,
    pub(super) expr_cond_pair: Py<PyAny>,
    pub(super) add: Py<PyAny>,
    pub(super) mul: Py<PyAny>,
    pub(super) modulo: Py<PyAny>,
    pub(super) pow: Py<PyAny>,
    pub(super) relational: Py<PyAny>,
    pub(super) equality: Py<PyAny>,
    pub(super) unequality: Py<PyAny>,
    pub(super) strict_less_than: Py<PyAny>,
    pub(super) less_than: Py<PyAny>,
    pub(super) strict_greater_than: Py<PyAny>,
    pub(super) greater_than: Py<PyAny>,
    pub(super) not: Py<PyAny>,
    pub(super) and: Py<PyAny>,
    pub(super) or: Py<PyAny>,
    pub(super) xor: Py<PyAny>,
    pub(super) nor: Py<PyAny>,
    pub(super) nand: Py<PyAny>,
    pub(super) ite: Py<PyAny>,
    pub(super) implies: Py<PyAny>,
    pub(super) boolean_true: Py<PyAny>,
    pub(super) boolean_false: Py<PyAny>,
    pub(super) complex_infinity: Py<PyAny>,
    pub(super) dummy: Py<PyAny>,
    pub(super) floor: Py<PyAny>,
    pub(super) ceiling: Py<PyAny>,
    pub(super) log: Py<PyAny>,
    pub(super) as_boolean: Py<PyAny>,
    // The values.
    pub(super) true_value: Py<PyAny>,
    pub(super) false_value: Py<PyAny>,
    pub(super) pi: Py<PyAny>,
    pub(super) e: Py<PyAny>,
    pub(super) infinity: Py<PyAny>,
    pub(super) negative_infinity: Py<PyAny>,
    pub(super) nan: Py<PyAny>,
    pub(super) half: Py<PyAny>,
    pub(super) precision_exhausted: Py<PyAny>,
    /// The native built-ins and the SymPy function each lowers through,
    /// except `exp2`, `log2` and `log10`, which lower through `Pow` and
    /// `log`.
    pub(super) natives: Vec<(BuiltinFunction, Py<PyAny>)>,
    /// The SymPy function classes that lift to native calls, in dispatch
    /// order, with the built-in each lifts to.
    pub(super) native_lifts: Vec<(Py<PyAny>, BuiltinFunction)>,
    // The prelude.
    pub(super) parity_opaque_piecewise: Py<PyAny>,
    pub(super) hide_piecewise_parity: Py<PyAny>,
    pub(super) holds_partial_piecewise: Py<PyAny>,
    pub(super) abort_walk: Py<PyAny>,
}

/// Return the attribute at the dotted `path` of `object`.
fn attribute(object: &Bound<'_, PyAny>, path: &str) -> PyResult<Py<PyAny>> {
    let mut current = object.clone();
    for part in path.split('.') {
        current = current.getattr(part)?;
    }
    Ok(current.unbind())
}

/// Return the name the prelude is published under:
/// `_fhy_core_sympy_<crate version>_<hash>`, with the version's `.` and `-`
/// written `_`, since a dotted name would read as a package path when a
/// pickle names one of the prelude's classes.
///
/// So two builds of the backend in one interpreter share a prelude only
/// when their versions and prelude sources are the same.
pub(super) fn prelude_module_name() -> String {
    let version = env!("CARGO_PKG_VERSION").replace(['.', '-', '+'], "_");
    format!("{PRELUDE_MODULE_STEM}_{version}_{}", prelude_hash_text())
}

/// Return the prelude module of this interpreter, published as `name`,
/// running the prelude and publishing it first if no backend has.
///
/// # Errors
///
/// Raises `ImportError` for a module published as `name` whose
/// `__fhy_core_prelude__` is missing or is not the hash of this prelude's
/// source, and the error of running the prelude.
fn prelude<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    let modules = py.import("sys")?.getattr("modules")?;
    let hash = prelude_hash_text();
    let published = if let Some(existing) = modules.cast::<PyDict>()?.get_item(name)? {
        existing
    } else {
        let module = PyModule::new(py, name)?;
        let source = CString::new(PRELUDE_SOURCE)
            .map_err(|_nul| pyo3::exceptions::PyValueError::new_err("the prelude holds a nul"))?;
        py.run(&source, Some(&module.dict()), None)?;
        module.setattr(PRELUDE_HASH_ATTRIBUTE, &hash)?;
        // `setdefault` is atomic under the GIL: if another thread published
        // a prelude while this one ran, every backend uses that one.
        modules.call_method1("setdefault", (name, module))?
    };
    let found = published
        .getattr_opt(PRELUDE_HASH_ATTRIBUTE)?
        .map(|value| value.str().map(|text| text.to_string()))
        .transpose()?;
    if found.as_deref() != Some(hash.as_str()) {
        return Err(pyo3::exceptions::PyImportError::new_err(format!(
            "sys.modules[{name:?}] is not this backend's prelude: its {PRELUDE_HASH_ATTRIBUTE} \
             is {}, not {hash:?}",
            found.map_or_else(|| "missing".to_owned(), |found| format!("{found:?}"))
        )));
    }
    Ok(published)
}

impl Handles {
    /// Import SymPy and the prelude and read what the backend needs.
    pub(super) fn load(py: Python<'_>) -> Result<Self, SympyUnavailableError> {
        Self::load_with_prelude(py, &prelude_module_name())
    }

    /// Import SymPy, and the prelude published as `prelude_name`, and read
    /// what the backend needs.
    pub(super) fn load_with_prelude(
        py: Python<'_>,
        prelude_name: &str,
    ) -> Result<Self, SympyUnavailableError> {
        let sympy = py
            .import("sympy")
            .map_err(SympyUnavailableError::MissingSympy)?;
        Self::read(py, &sympy, prelude_name).map_err(SympyUnavailableError::Incompatible)
    }

    #[expect(clippy::too_many_lines, reason = "one line per handle read")]
    fn read(py: Python<'_>, sympy: &Bound<'_, PyModule>, prelude_name: &str) -> PyResult<Self> {
        py.import("sympy.core.evalf")?;
        py.import("sympy.functions.elementary.piecewise")?;
        py.import("sympy.logic.boolalg")?;
        let module = sympy.as_any();
        let get = |path: &str| attribute(module, path);
        let prelude = prelude(py, prelude_name)?;
        let from_prelude = |name: &str| attribute(&prelude, name);
        let log = get("log")?;
        let exp = get("exp")?;
        let sqrt = get("sqrt")?;
        let floor = get("floor")?;
        let ceiling = get("ceiling")?;
        let round = from_prelude("ROUND")?;
        let natives = vec![
            (BuiltinFunction::Exp, exp.clone_ref(py)),
            (BuiltinFunction::Log, log.clone_ref(py)),
            (BuiltinFunction::Sqrt, sqrt),
            (BuiltinFunction::Sin, get("sin")?),
            (BuiltinFunction::Cos, get("cos")?),
            (BuiltinFunction::Tan, get("tan")?),
            (BuiltinFunction::Arcsin, get("asin")?),
            (BuiltinFunction::Arccos, get("acos")?),
            (BuiltinFunction::Arctan, get("atan")?),
            (BuiltinFunction::Sinh, get("sinh")?),
            (BuiltinFunction::Cosh, get("cosh")?),
            (BuiltinFunction::Tanh, get("tanh")?),
            (BuiltinFunction::Erf, get("erf")?),
            (BuiltinFunction::Round, round.clone_ref(py)),
            (BuiltinFunction::Floor, floor.clone_ref(py)),
            (BuiltinFunction::Ceil, ceiling.clone_ref(py)),
        ];
        let native_lifts = vec![
            (exp.clone_ref(py), BuiltinFunction::Exp),
            (get("sin")?, BuiltinFunction::Sin),
            (get("cos")?, BuiltinFunction::Cos),
            (get("tan")?, BuiltinFunction::Tan),
            (get("asin")?, BuiltinFunction::Arcsin),
            (get("acos")?, BuiltinFunction::Arccos),
            (get("atan")?, BuiltinFunction::Arctan),
            (get("sinh")?, BuiltinFunction::Sinh),
            (get("cosh")?, BuiltinFunction::Cosh),
            (get("tanh")?, BuiltinFunction::Tanh),
            (get("erf")?, BuiltinFunction::Erf),
            (log.clone_ref(py), BuiltinFunction::Log),
            (floor.clone_ref(py), BuiltinFunction::Floor),
            (ceiling.clone_ref(py), BuiltinFunction::Ceil),
            (round, BuiltinFunction::Round),
        ];
        let rational = get("Rational")?;
        let half = rational.bind(py).call1((1, 2))?.unbind();
        Ok(Self {
            sympy: sympy.clone().unbind(),
            basic: get("Basic")?,
            expr: get("Expr")?,
            boolean: get("logic.boolalg.Boolean")?,
            boolean_function: get("logic.boolalg.BooleanFunction")?,
            symbol: get("Symbol")?,
            integer: get("Integer")?,
            float: get("Float")?,
            rational,
            piecewise: get("Piecewise")?,
            expr_cond_pair: get("functions.elementary.piecewise.ExprCondPair")?,
            add: get("Add")?,
            mul: get("Mul")?,
            modulo: get("Mod")?,
            pow: get("Pow")?,
            relational: get("core.relational.Relational")?,
            equality: get("Equality")?,
            unequality: get("Unequality")?,
            strict_less_than: get("StrictLessThan")?,
            less_than: get("LessThan")?,
            strict_greater_than: get("StrictGreaterThan")?,
            greater_than: get("GreaterThan")?,
            not: get("logic.boolalg.Not")?,
            and: get("logic.boolalg.And")?,
            or: get("logic.boolalg.Or")?,
            xor: get("logic.boolalg.Xor")?,
            nor: get("logic.boolalg.Nor")?,
            nand: get("logic.boolalg.Nand")?,
            ite: get("logic.boolalg.ITE")?,
            implies: get("logic.boolalg.Implies")?,
            boolean_true: get("logic.boolalg.BooleanTrue")?,
            boolean_false: get("logic.boolalg.BooleanFalse")?,
            complex_infinity: get("core.numbers.ComplexInfinity")?,
            dummy: get("Dummy")?,
            floor,
            ceiling,
            log,
            as_boolean: get("logic.boolalg.as_Boolean")?,
            true_value: get("true")?,
            false_value: get("false")?,
            pi: get("pi")?,
            e: get("E")?,
            infinity: get("oo")?,
            negative_infinity: get("S.NegativeInfinity")?,
            nan: get("nan")?,
            half,
            precision_exhausted: get("core.evalf.PrecisionExhausted")?,
            natives,
            native_lifts,
            parity_opaque_piecewise: from_prelude("ParityOpaquePiecewise")?,
            hide_piecewise_parity: from_prelude("hide_piecewise_parity")?,
            holds_partial_piecewise: from_prelude("holds_partial_piecewise")?,
            abort_walk: from_prelude("AbortWalk")?,
        })
    }
}
