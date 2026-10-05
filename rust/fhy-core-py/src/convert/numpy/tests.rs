//! The stories of the public `NumPy` conversions, run in an interpreter
//! this test binary embeds; they need Python with `NumPy`.
//!
//! `fhy_core` itself is not importable in the embedded interpreter, so the
//! error mapping is checked on the errors that map to built-in exceptions;
//! the Python suite covers the ones that map to `fhy_core`'s classes.

use std::ffi::CString;
use std::sync::{Mutex, MutexGuard, PoisonError};

use numpy::ndarray::{ArrayD, Axis, CowArray, IxDyn, Slice, arr0, arr1, arr2};
use pyo3::exceptions::{
    PyImportError, PyMemoryError, PyOverflowError, PyRuntimeError, PyTypeError, PyValueError,
    PyZeroDivisionError,
};
use pyo3::types::{PyDict, PyModule};

use fhy_core::expression::Expression;
use fhy_core::expression::builtins::BuiltinFunction;
use fhy_core::expression::evaluate::{
    ArrayBinding, ArrayKernels, ArrayValue, EvaluationError, LaneFailure, Scalar,
};
use fhy_core::foreign::BoxError;

use super::*;

/// What a story needs to run, when `NumPy` cannot be imported.
const RECIPE: &str = "the NumPy conversion stories need Python with NumPy. Build and run them \
     with PYO3_PYTHON naming a Python that has a shared libpython and the numpy package, \
     PYTHONPATH naming that Python's site-packages (an embedded interpreter does not read a \
     virtualenv's pyvenv.cfg), and LD_LIBRARY_PATH naming the directory of its libpython \
     when the loader does not find it; CONTRIBUTING's \"Rust test layout\" records the recipe";

/// Serializes the stories, one of which hides `numpy` from `sys.modules`.
static SERIAL: Mutex<()> = Mutex::new(());

/// Run `body` with the embedded interpreter and the `numpy` module, alone.
fn with_numpy<R>(body: impl for<'py> FnOnce(Python<'py>, &Bound<'py, PyModule>) -> R) -> R {
    let _serial: MutexGuard<'_, ()> = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
    Python::initialize();
    Python::attach(|py| {
        let numpy = require_numpy(py, "story", "pip install numpy")
            .unwrap_or_else(|error| panic!("{RECIPE}. Importing NumPy failed: {error}"));
        body(py, &numpy)
    })
}

/// Return the value of the Python expression `source`, with `np` and `sys`
/// in scope.
fn python<'py>(numpy: &Bound<'py, PyModule>, source: &str) -> Bound<'py, PyAny> {
    let py = numpy.py();
    let scope = PyDict::new(py);
    scope.set_item("np", numpy).expect("set");
    scope
        .set_item("sys", py.import("sys").expect("sys"))
        .expect("set");
    let code = CString::new(source).expect("no nul");
    py.eval(&code, Some(&scope), None)
        .unwrap_or_else(|error| panic!("evaluating {source:?} failed: {error}"))
}

/// Return the conversion of the Python expression `source`, labelled `x`.
fn convert<'py>(numpy: &Bound<'py, PyModule>, source: &str) -> PyResult<NumpyValue<'py>> {
    NumpyValue::from_python(numpy, "x", &python(numpy, source))
}

/// Return the address of the first byte of the `NumPy` array `array`.
fn address(array: &Bound<'_, PyAny>) -> usize {
    array
        .getattr("ctypes")
        .and_then(|ctypes| ctypes.getattr("data"))
        .and_then(|data| data.extract())
        .expect("an array has a data address")
}

/// Return `array`'s dtype name.
fn dtype_name(array: &Bound<'_, PyAny>) -> String {
    array
        .getattr("dtype")
        .and_then(|dtype| dtype.str())
        .map(|name| name.to_string())
        .expect("a dtype name")
}

/// Return the real view of `binding`.
fn reals(binding: ArrayBinding<'_>) -> numpy::ndarray::ArrayViewD<'_, f64> {
    match binding {
        ArrayBinding::Real(view) => view,
        other => panic!("expected reals, got {:?}", other.symbol_type()),
    }
}

mod require {
    use super::*;

    #[test]
    fn numpy_is_returned_when_it_imports() {
        with_numpy(|_py, numpy| {
            assert_eq!(numpy.name().expect("a name").to_string(), "numpy");
        });
    }

    #[test]
    fn a_missing_numpy_raises_import_error_naming_the_entry_point_and_the_cause() {
        with_numpy(|py, numpy| {
            let modules = py
                .import("sys")
                .expect("sys")
                .getattr("modules")
                .expect("modules");
            modules.set_item("numpy", py.None()).expect("hide numpy");
            let raised = require_numpy(py, "the_entry", "pip install the-extra");
            modules.set_item("numpy", numpy).expect("restore numpy");
            let Err(error) = raised else {
                panic!("numpy was hidden");
            };
            assert!(error.is_instance_of::<PyImportError>(py));
            assert_eq!(
                error.value(py).to_string(),
                "NumPy is required for `the_entry`; install it with `pip install the-extra`."
            );
            assert!(error.cause(py).is_some(), "the import's error is the cause");
        });
    }
}

mod from_python {
    use super::*;

    #[test]
    fn python_numbers_are_scalars_of_their_domain() {
        with_numpy(|_py, numpy| {
            let cases = [
                ("True", Scalar::Bool(true)),
                ("False", Scalar::Bool(false)),
                ("7", Scalar::Int(7)),
                ("-(2**63)", Scalar::Int(i64::MIN)),
                ("2.5", Scalar::Real(2.5)),
            ];
            for (source, expected) in cases {
                let value = convert(numpy, source).expect("converts");
                assert_eq!(value.as_scalar(), Some(expected), "{source}");
            }
        });
    }

    #[test]
    fn an_int_outside_the_64_bit_range_is_an_overflow_error_naming_the_binding() {
        with_numpy(|py, numpy| {
            for source in ["2**63", "-(2**63) - 1"] {
                let error = convert(numpy, source).expect_err("out of range");
                assert!(error.is_instance_of::<PyOverflowError>(py), "{source}");
                assert_eq!(
                    error.value(py).to_string(),
                    "the int bound to \"x\" is outside the 64-bit range",
                    "{source}"
                );
            }
        });
    }

    #[test]
    fn the_borrowed_dtypes_are_read_in_place() {
        with_numpy(|_py, numpy| {
            for (source, domain) in [
                ("np.array([True, False, True])", "bool"),
                ("np.array([1, 2, 3], dtype='int64')", "int"),
                ("np.array([1.5, 2.5, 3.5], dtype='float64')", "real"),
            ] {
                let array = python(numpy, source);
                let value = NumpyValue::from_python(numpy, "x", &array).expect("converts");
                assert_eq!(value.as_scalar(), None);
                let binding = value.as_binding();
                let (pointer, kind) = match &binding {
                    ArrayBinding::Bool(view) => (view.as_ptr() as usize, "bool"),
                    ArrayBinding::Int(view) => (view.as_ptr() as usize, "int"),
                    ArrayBinding::Real(view) => (view.as_ptr() as usize, "real"),
                };
                assert_eq!(kind, domain, "{source}");
                assert_eq!(pointer, address(&array), "{source} is borrowed, not copied");
            }
        });
    }

    #[test]
    fn a_strided_borrowed_array_keeps_its_strides() {
        with_numpy(|_py, numpy| {
            let array = python(numpy, "np.arange(6, dtype='float64')[::2]");
            let value = NumpyValue::from_python(numpy, "x", &array).expect("converts");
            let binding = value.as_binding();
            let view = reals(binding);
            assert_eq!(view.as_ptr() as usize, address(&array));
            assert_eq!(view.iter().copied().collect::<Vec<_>>(), [0.0, 2.0, 4.0]);
        });
    }

    #[test]
    fn a_narrower_integer_dtype_is_cast_to_int64_once() {
        with_numpy(|_py, numpy| {
            for dtype in [
                "int8", "int16", "int32", "uint8", "uint16", "uint32", "uint64",
            ] {
                let array = python(numpy, &format!("np.array([1, 2, 3], dtype='{dtype}')"));
                let value = NumpyValue::from_python(numpy, "x", &array).expect("converts");
                let ArrayBinding::Int(view) = value.as_binding() else {
                    panic!("{dtype} is an integer");
                };
                assert_eq!(
                    view.iter().copied().collect::<Vec<_>>(),
                    [1, 2, 3],
                    "{dtype}"
                );
                assert_ne!(view.as_ptr() as usize, address(&array), "{dtype} is cast");
            }
        });
    }

    #[test]
    fn a_narrower_float_dtype_is_cast_to_float64_once() {
        with_numpy(|_py, numpy| {
            for dtype in ["float16", "float32"] {
                let array = python(numpy, &format!("np.array([0.5, 1.5], dtype='{dtype}')"));
                let value = NumpyValue::from_python(numpy, "x", &array).expect("converts");
                let view = reals(value.as_binding());
                assert_eq!(
                    view.iter().copied().collect::<Vec<_>>(),
                    [0.5, 1.5],
                    "{dtype}"
                );
                assert_ne!(view.as_ptr() as usize, address(&array), "{dtype} is cast");
            }
        });
    }

    #[test]
    fn a_list_goes_through_numpy_asarray() {
        with_numpy(|_py, numpy| {
            let value = convert(numpy, "[[1, 2], [3, 4]]").expect("converts");
            let ArrayBinding::Int(view) = value.as_binding() else {
                panic!("a list of ints is integers");
            };
            assert_eq!(view.shape(), [2, 2]);
            assert_eq!(view.iter().copied().collect::<Vec<_>>(), [1, 2, 3, 4]);
        });
    }

    #[test]
    fn a_non_native_byte_order_is_cast_to_native() {
        with_numpy(|_py, numpy| {
            let swapped = "('>' if sys.byteorder == 'little' else '<')";
            let array = python(
                numpy,
                &format!("np.array([1.5, 2.5], dtype={swapped} + 'f8')"),
            );
            assert!(
                !array
                    .getattr("dtype")
                    .expect("dtype")
                    .getattr("isnative")
                    .expect("flag")
                    .extract::<bool>()
                    .expect("bool")
            );
            let value = NumpyValue::from_python(numpy, "x", &array).expect("converts");
            let view = reals(value.as_binding());
            assert_eq!(view.iter().copied().collect::<Vec<_>>(), [1.5, 2.5]);
            assert_ne!(
                view.as_ptr() as usize,
                address(&array),
                "it cannot be borrowed"
            );

            let integers = python(numpy, &format!("np.array([1, 2], dtype={swapped} + 'i8')"));
            let value = NumpyValue::from_python(numpy, "x", &integers).expect("converts");
            let ArrayBinding::Int(view) = value.as_binding() else {
                panic!("integers");
            };
            assert_eq!(view.iter().copied().collect::<Vec<_>>(), [1, 2]);
        });
    }

    #[test]
    fn a_uint64_above_the_signed_range_is_an_overflow_error_and_the_maximum_fits() {
        with_numpy(|py, numpy| {
            let error =
                convert(numpy, "np.array([1, 2**63], dtype='uint64')").expect_err("above i64::MAX");
            assert!(error.is_instance_of::<PyOverflowError>(py));
            assert_eq!(
                error.value(py).to_string(),
                "the uint64 array bound to \"x\" holds a value above the 64-bit signed range"
            );
            let value = convert(numpy, "np.array([2**63 - 1], dtype='uint64')").expect("fits");
            let ArrayBinding::Int(view) = value.as_binding() else {
                panic!("integers");
            };
            assert_eq!(view.iter().copied().collect::<Vec<_>>(), [i64::MAX]);
            convert(numpy, "np.array([], dtype='uint64')").expect("an empty array fits");
        });
    }

    #[test]
    fn a_dtype_that_is_not_boolean_integer_or_real_is_a_type_error_naming_it() {
        with_numpy(|py, numpy| {
            let mut cases = vec![
                ("np.array([1j])", "complex128"),
                ("np.array(['a'])", "<U1"),
                ("np.array([1], dtype='timedelta64[s]')", "timedelta64[s]"),
                ("np.array([None], dtype=object)", "object"),
            ];
            // `longdouble` is a wider type only on some platforms; where it
            // is binary64, it is a real like `float64`.
            let is_wide = python(numpy, "np.dtype(np.longdouble).itemsize > 8")
                .is_truthy()
                .expect("a bool");
            if is_wide {
                cases.push(("np.array([1.0], dtype=np.longdouble)", "float128"));
            }
            for (source, dtype) in cases {
                let error = convert(numpy, source).expect_err(source);
                assert!(error.is_instance_of::<PyTypeError>(py), "{source}");
                assert_eq!(
                    error.value(py).to_string(),
                    format!(
                        "the value bound to \"x\" has dtype {dtype}, which is not boolean, \
                         integer or floating-point of at most 64 bits"
                    ),
                    "{source}"
                );
            }
        });
    }

    #[test]
    fn a_zero_dimensional_array_is_a_scalar_of_its_domain() {
        with_numpy(|_py, numpy| {
            let cases = [
                ("np.array(True)", Scalar::Bool(true)),
                ("np.array(7, dtype='int8')", Scalar::Int(7)),
                ("np.array(3.5)", Scalar::Real(3.5)),
                ("np.array(0.25, dtype='float32')", Scalar::Real(0.25)),
                ("np.float64(2.0)", Scalar::Real(2.0)),
                ("np.int64(-4)", Scalar::Int(-4)),
                ("np.bool_(False)", Scalar::Bool(false)),
            ];
            for (source, expected) in cases {
                let value = convert(numpy, source).expect("converts");
                assert_eq!(value.as_scalar(), Some(expected), "{source}");
                assert_eq!(value.as_binding().shape(), &[] as &[usize], "{source}");
            }
        });
    }

    #[test]
    fn a_one_element_array_is_an_array_not_a_scalar() {
        with_numpy(|_py, numpy| {
            let value = convert(numpy, "np.array([3.5])").expect("converts");
            assert_eq!(value.as_scalar(), None);
            assert_eq!(value.as_binding().shape(), [1]);
        });
    }

    #[test]
    fn a_mutably_borrowed_array_is_a_value_error() {
        with_numpy(|py, numpy| {
            let array = python(numpy, "np.array([1.0, 2.0])");
            let typed = array
                .cast::<numpy::PyArrayDyn<f64>>()
                .expect("an f64 array")
                .clone();
            let _writer = typed.try_readwrite().expect("no other borrow");
            let error = NumpyValue::from_python(numpy, "x", &array).expect_err("borrowed");
            assert!(error.is_instance_of::<PyValueError>(py));
        });
    }
}

mod views {
    use super::*;

    #[test]
    fn a_scalar_binds_as_a_zero_dimensional_view_of_its_domain() {
        with_numpy(|_py, numpy| {
            let boolean = convert(numpy, "True").expect("converts");
            let integer = convert(numpy, "3").expect("converts");
            let real = convert(numpy, "2.5").expect("converts");
            assert!(matches!(boolean.as_binding(), ArrayBinding::Bool(view)
                if view.ndim() == 0 && view.iter().copied().eq([true])));
            assert!(matches!(integer.as_binding(), ArrayBinding::Int(view)
                if view.ndim() == 0 && view.iter().copied().eq([3])));
            assert!(matches!(real.as_binding(), ArrayBinding::Real(view)
                if view.ndim() == 0 && view.iter().copied().eq([2.5])));
        });
    }

    #[test]
    fn to_array_value_copies_arrays_and_wraps_scalars() {
        with_numpy(|_py, numpy| {
            let array = python(numpy, "np.array([[1.0, 2.0], [3.0, 4.0]])");
            let value = NumpyValue::from_python(numpy, "x", &array).expect("converts");
            let owned = value.to_array_value();
            assert_eq!(
                owned,
                ArrayValue::Real(arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn())
            );
            array.call_method1("fill", (9.0,)).expect("fill");
            assert_eq!(
                owned,
                ArrayValue::Real(arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn()),
                "the copy does not follow the array"
            );

            let booleans = convert(numpy, "np.array([True, False])").expect("converts");
            assert_eq!(
                booleans.to_array_value(),
                ArrayValue::Bool(arr1(&[true, false]).into_dyn())
            );
            let scalar = convert(numpy, "5").expect("converts");
            assert_eq!(scalar.to_array_value(), ArrayValue::Int(arr0(5).into_dyn()));
            let zero_dimensional = convert(numpy, "np.array(1.5, dtype='float32')").expect("ok");
            assert_eq!(
                zero_dimensional.to_array_value(),
                ArrayValue::Real(arr0(1.5).into_dyn())
            );
        });
    }
}

mod to_numpy {
    use super::*;

    /// Return the value of `array` as a Python list, through `tolist`.
    fn listed(array: &Bound<'_, PyAny>) -> String {
        array
            .call_method0("tolist")
            .and_then(|list| list.repr())
            .map(|text| text.to_string())
            .expect("a list")
    }

    #[test]
    fn scalars_become_numpy_scalars_of_their_domain() {
        with_numpy(|_py, numpy| {
            for (scalar, dtype, text) in [
                (Scalar::Bool(true), "bool", "True"),
                (Scalar::Int(-3), "int64", "-3"),
                (Scalar::Real(0.5), "float64", "0.5"),
            ] {
                let value = scalar_to_numpy(numpy, scalar).expect("converts");
                assert_eq!(dtype_name(&value), dtype);
                assert_eq!(value.str().expect("text").to_string(), text);
                assert_eq!(
                    value
                        .getattr("ndim")
                        .expect("ndim")
                        .extract::<usize>()
                        .expect("usize"),
                    0
                );
            }
        });
    }

    #[test]
    fn arrays_become_numpy_arrays_of_their_domain() {
        with_numpy(|_py, numpy| {
            let booleans = array_value_to_numpy(
                numpy,
                ArrayValue::Bool(arr2(&[[true, false], [false, true]]).into_dyn()),
            )
            .expect("converts");
            assert_eq!(dtype_name(&booleans), "bool");
            assert_eq!(listed(&booleans), "[[True, False], [False, True]]");

            let integers =
                array_value_to_numpy(numpy, ArrayValue::Int(arr1(&[1, -2, 3]).into_dyn()))
                    .expect("converts");
            assert_eq!(dtype_name(&integers), "int64");
            assert_eq!(listed(&integers), "[1, -2, 3]");

            let reals = array_value_to_numpy(numpy, ArrayValue::Real(arr1(&[0.5, 1.5]).into_dyn()))
                .expect("converts");
            assert_eq!(dtype_name(&reals), "float64");
            assert_eq!(listed(&reals), "[0.5, 1.5]");
        });
    }

    #[test]
    fn an_empty_array_keeps_its_shape() {
        with_numpy(|_py, numpy| {
            let empty = ArrayD::<f64>::zeros(IxDyn(&[0, 3]));
            let value = array_value_to_numpy(numpy, ArrayValue::Real(empty)).expect("converts");
            assert_eq!(
                value
                    .getattr("shape")
                    .expect("shape")
                    .extract::<Vec<usize>>()
                    .expect("shape"),
                [0, 3]
            );
        });
    }

    #[test]
    fn a_zero_dimensional_array_becomes_a_numpy_scalar() {
        with_numpy(|_py, numpy| {
            let value =
                array_value_to_numpy(numpy, ArrayValue::Int(arr0(4).into_dyn())).expect("converts");
            assert_eq!(dtype_name(&value), "int64");
            assert_eq!(
                value.get_type().name().expect("name").to_string(),
                "int64",
                "a scalar, not a 0-d array"
            );
        });
    }

    #[test]
    fn the_array_takes_over_the_buffer() {
        with_numpy(|_py, numpy| {
            let owned = arr1(&[1.0, 2.0, 3.0]).into_dyn();
            let pointer = owned.as_ptr() as usize;
            let value = array_value_to_numpy(numpy, ArrayValue::Real(owned)).expect("converts");
            assert_eq!(address(&value), pointer);
        });
    }

    #[test]
    fn the_three_domains_round_trip_through_numpy() {
        with_numpy(|_py, numpy| {
            let values = [
                ArrayValue::Bool(arr2(&[[true, false, true], [false, false, true]]).into_dyn()),
                ArrayValue::Int(arr2(&[[i64::MIN, -1, 0], [1, 2, i64::MAX]]).into_dyn()),
                ArrayValue::Real(
                    arr2(&[[0.5, -0.0, f64::INFINITY], [1e300, -2.5, 3.0]]).into_dyn(),
                ),
                ArrayValue::Bool(arr0(true).into_dyn()),
                ArrayValue::Int(arr0(-9).into_dyn()),
                ArrayValue::Real(arr0(0.125).into_dyn()),
            ];
            for value in values {
                let python = array_value_to_numpy(numpy, value.clone()).expect("to numpy");
                let back = NumpyValue::from_python(numpy, "x", &python).expect("from numpy");
                assert_eq!(back.to_array_value(), value);
            }
        });
    }
}

mod kernels {
    use super::*;

    /// Return the lanes of `function` over `argument` computed by `kernels`.
    fn native(
        kernels: &NumpyKernels,
        function: BuiltinFunction,
        argument: CowArray<'_, f64, IxDyn>,
    ) -> Vec<f64> {
        kernels
            .native(function, argument)
            .expect("the kernel computes")
            .iter()
            .copied()
            .collect()
    }

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!(
                (a.is_nan() && e.is_nan()) || a.total_cmp(e).is_eq() || (a - e).abs() < 1e-12,
                "{actual:?} != {expected:?}"
            );
        }
    }

    #[test]
    fn the_kernels_handle_the_fourteen_transcendentals_only() {
        with_numpy(|_py, numpy| {
            let kernels = NumpyKernels::new(numpy, &[]);
            let handled = [
                BuiltinFunction::Exp,
                BuiltinFunction::Exp2,
                BuiltinFunction::Log,
                BuiltinFunction::Log2,
                BuiltinFunction::Log10,
                BuiltinFunction::Sin,
                BuiltinFunction::Cos,
                BuiltinFunction::Tan,
                BuiltinFunction::Arcsin,
                BuiltinFunction::Arccos,
                BuiltinFunction::Arctan,
                BuiltinFunction::Sinh,
                BuiltinFunction::Cosh,
                BuiltinFunction::Tanh,
            ];
            for function in handled {
                assert!(kernels.handles(function), "{function:?}");
            }
            for function in [
                BuiltinFunction::Sqrt,
                BuiltinFunction::Round,
                BuiltinFunction::Floor,
                BuiltinFunction::Ceil,
                BuiltinFunction::Erf,
            ] {
                assert!(!kernels.handles(function), "{function:?}");
            }
        });
    }

    #[test]
    fn a_transcendental_is_computed_over_owned_lanes() {
        with_numpy(|_py, numpy| {
            let kernels = NumpyKernels::new(numpy, &[]);
            let lanes = arr1(&[0.0, 1.0, 2.0]).into_dyn();
            assert_close(
                &native(&kernels, BuiltinFunction::Exp, CowArray::from(lanes)),
                &[1.0, 1.0_f64.exp(), 2.0_f64.exp()],
            );
        });
    }

    /// A stand-in for the `numpy` module, run from `source` over the real
    /// one: it has `asarray`, `float64` and an `errstate` that records its
    /// entry and exit in `states`, and `source` defines the ufuncs.
    fn stand_in<'py>(py: Python<'py>, source: &str) -> Bound<'py, PyModule> {
        let prelude = "
import numpy as _numpy

asarray = _numpy.asarray
float64 = _numpy.float64
calls = []
states = []


class _State:
    def __init__(self, settings):
        self.settings = settings

    def __enter__(self):
        states.append(('enter', self.settings))

    def __exit__(self, *arguments):
        states.append(('exit', arguments))
        if exit_message is not None:
            raise RuntimeError(exit_message)


exit_message = None


def errstate(**settings):
    return _State(settings)
";
        let code = CString::new(format!("{prelude}{source}")).expect("no nul");
        PyModule::from_code(py, &code, c"stand_in_numpy.py", c"stand_in_numpy")
            .unwrap_or_else(|error| panic!("the stand-in failed: {error}"))
    }

    /// Return the shape of `array` as a list.
    fn shape_of(array: &Bound<'_, PyAny>) -> Vec<usize> {
        array
            .getattr("shape")
            .and_then(|shape| shape.extract())
            .expect("a shape")
    }

    /// Return the elements of `array`, flattened.
    fn flattened(array: &Bound<'_, PyAny>) -> Vec<f64> {
        array
            .call_method0("ravel")
            .and_then(|flat| flat.call_method0("tolist"))
            .and_then(|list| list.extract())
            .expect("a list of reals")
    }

    #[test]
    fn find_input_returns_the_inputs_own_array_for_its_whole_view() {
        with_numpy(|py, numpy| {
            let array = python(numpy, "np.arange(6, dtype='float64').reshape(2, 3)");
            let input = NumpyValue::from_python(numpy, "x", &array).expect("converts");
            let kernels = NumpyKernels::new(numpy, &[&input]);
            let binding = input.as_binding();
            let view = reals(binding);

            let found = kernels
                .find_input(py, &CowArray::from(view.view()))
                .expect("searches")
                .expect("the whole view is the input");

            assert!(found.is(&array));
        });
    }

    #[test]
    fn find_input_returns_a_view_of_the_inputs_buffer_for_a_chunk() {
        with_numpy(|py, numpy| {
            let array = python(numpy, "np.arange(6, dtype='float64').reshape(2, 3)");
            let input = NumpyValue::from_python(numpy, "x", &array).expect("converts");
            let kernels = NumpyKernels::new(numpy, &[&input]);
            let binding = input.as_binding();
            let view = reals(binding);
            let chunk = view.slice_axis(Axis(0), Slice::from(1..2));

            let found = kernels
                .find_input(py, &CowArray::from(chunk))
                .expect("searches")
                .expect("a chunk is a run of the input");

            assert!(!found.is(&array));
            assert_eq!(shape_of(&found), [1, 3]);
            assert_eq!(flattened(&found), [3.0, 4.0, 5.0]);
            assert_eq!(address(&found), address(&array) + 3 * size_of::<f64>());
        });
    }

    #[test]
    fn find_input_finds_nothing_for_a_strided_or_foreign_view() {
        with_numpy(|py, numpy| {
            let array = python(numpy, "np.arange(6, dtype='float64').reshape(2, 3)");
            let input = NumpyValue::from_python(numpy, "x", &array).expect("converts");
            let kernels = NumpyKernels::new(numpy, &[&input]);
            let binding = input.as_binding();
            let view = reals(binding);

            let strided = view.slice_axis(Axis(1), Slice::new(0, None, 2));
            let found = kernels
                .find_input(py, &CowArray::from(strided))
                .expect("searches");
            assert!(found.is_none(), "a strided view is not a run");

            let other = arr2(&[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]).into_dyn();
            let found = kernels
                .find_input(py, &CowArray::from(other.view()))
                .expect("searches");
            assert!(found.is_none(), "another allocation is not an input");

            let found = kernels
                .find_input(py, &CowArray::from(other))
                .expect("searches");
            assert!(found.is_none(), "an owned argument is not a view");
        });
    }

    #[test]
    fn find_input_gives_a_reshaped_view_its_own_shape() {
        with_numpy(|py, numpy| {
            let array = python(numpy, "np.arange(6, dtype='float64').reshape(2, 3)");
            let input = NumpyValue::from_python(numpy, "x", &array).expect("converts");
            let kernels = NumpyKernels::new(numpy, &[&input]);
            let binding = input.as_binding();
            let view = reals(binding);

            for shape in [vec![3, 2], vec![1, 6], vec![6]] {
                let reshaped = view
                    .view()
                    .into_shape_with_order(IxDyn(&shape))
                    .expect("six lanes");

                let found = kernels
                    .find_input(py, &CowArray::from(reshaped))
                    .expect("searches")
                    .expect("the lanes are the input's");

                assert_eq!(shape_of(&found), shape);
                assert_eq!(flattened(&found), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
                assert_eq!(address(&found), address(&array));
            }
        });
    }

    #[test]
    fn a_transcendental_over_a_view_of_an_input_reuses_its_array() {
        with_numpy(|py, numpy| {
            let array = python(numpy, "np.array([0.0, 0.5, 1.0, 1.5], dtype='float64')");
            let input = NumpyValue::from_python(numpy, "x", &array).expect("converts");
            let kernels = NumpyKernels::new(numpy, &[&input]);
            let binding = input.as_binding();
            let view = reals(binding);
            let whole = native(&kernels, BuiltinFunction::Sin, CowArray::from(view.view()));
            assert_close(&whole, &[0.0, 0.5_f64.sin(), 1.0_f64.sin(), 1.5_f64.sin()]);
            let chunk = view.slice_axis(Axis(0), Slice::from(1..3));
            let part = native(&kernels, BuiltinFunction::Cos, CowArray::from(chunk));
            assert_close(&part, &[0.5_f64.cos(), 1.0_f64.cos()]);

            // The ufunc is handed the input's own array for the whole view,
            // and a view of its buffer for the chunk, never a copy.
            let recorder = stand_in(
                py,
                "
def sin(argument, out=None):
    calls.append(argument)
    return _numpy.sin(argument)
",
            );
            let kernels = NumpyKernels::new(&recorder, &[&input]);
            native(&kernels, BuiltinFunction::Sin, CowArray::from(view.view()));
            let chunk = view.slice_axis(Axis(0), Slice::from(1..3));
            native(&kernels, BuiltinFunction::Sin, CowArray::from(chunk));
            let calls = recorder.getattr("calls").expect("calls");
            let first = calls.get_item(0).expect("the whole view's call");
            let second = calls.get_item(1).expect("the chunk's call");
            assert!(first.is(&array));
            assert_eq!(address(&second), address(&array) + size_of::<f64>());
            assert_eq!(flattened(&second), [0.5, 1.0]);
        });
    }

    #[test]
    fn a_kernels_failure_is_the_pythons_own_exception_and_the_warnings_are_restored() {
        with_numpy(|py, _numpy| {
            let failing = stand_in(
                py,
                "
failure = ValueError('sin failed')


def sin(argument, out=None):
    raise failure
",
            );
            let kernels = NumpyKernels::new(&failing, &[]);
            let lanes = arr1(&[0.0, 1.0]).into_dyn();

            let error = kernels
                .native(BuiltinFunction::Sin, CowArray::from(lanes))
                .expect_err("the ufunc raises");

            let error = crate::util::exceptions::unbox_py_err(error).expect("a Python exception");
            let failure = failing.getattr("failure").expect("the exception");
            assert!(error.value(py).is(&failure));
            let states = failing.getattr("states").expect("states");
            assert_eq!(
                states.repr().expect("repr").to_string(),
                "[('enter', {'all': 'ignore'}), ('exit', (None, None, None))]"
            );
        });
    }

    #[test]
    fn a_failing_exit_wins_over_the_body_with_the_body_as_its_context() {
        with_numpy(|py, _numpy| {
            let failing = stand_in(
                py,
                "
exit_message = 'exit failed'
",
            );

            let after_success = with_floating_point_warnings_silenced(&failing, || Ok(5));
            let after_failure: PyResult<i32> =
                with_floating_point_warnings_silenced(&failing, || {
                    Err(PyValueError::new_err("body failed"))
                });

            let error = after_success.expect_err("the exit's exception replaces the result");
            assert!(error.is_instance_of::<PyRuntimeError>(py));
            assert!(error.context(py).is_none());
            let error = after_failure.expect_err("the exit's exception wins");
            assert_eq!(error.value(py).to_string(), "exit failed");
            let context = error.context(py).expect("the body's exception is chained");
            assert_eq!(context.value(py).to_string(), "body failed");
        });
    }

    #[test]
    fn the_warnings_are_silenced_around_the_body_and_its_result_returned() {
        with_numpy(|py, _numpy| {
            let recorder = stand_in(py, "");

            let result = with_floating_point_warnings_silenced(&recorder, || {
                let states = recorder.getattr("states").expect("states");
                assert_eq!(states.len().expect("len"), 1, "entered before the body");
                Ok(7)
            });

            assert_eq!(result.expect("the body's result"), 7);
            let states = recorder.getattr("states").expect("states");
            assert_eq!(states.len().expect("len"), 2, "exited after the body");
        });
    }

    #[test]
    fn inputs_that_are_not_reals_are_ignored() {
        with_numpy(|_py, numpy| {
            let scalar = convert(numpy, "2.0").expect("converts");
            let integers = convert(numpy, "np.array([1, 2])").expect("converts");
            let kernels = NumpyKernels::new(numpy, &[&scalar, &integers]);
            let lanes = arr1(&[1.0]).into_dyn();
            assert_close(
                &native(&kernels, BuiltinFunction::Tanh, CowArray::from(lanes)),
                &[1.0_f64.tanh()],
            );
        });
    }

    #[test]
    fn numpy_floating_point_warnings_are_silenced() {
        with_numpy(|py, numpy| {
            let warnings = py.import("warnings").expect("warnings");
            let caught = warnings
                .call_method0("catch_warnings")
                .expect("catch_warnings");
            let record = caught.call_method0("__enter__").expect("enter");
            warnings
                .call_method1("simplefilter", ("error",))
                .expect("filter");
            let kernels = NumpyKernels::new(numpy, &[]);
            let lanes = arr1(&[0.0, -1.0]).into_dyn();
            let result = kernels.native(BuiltinFunction::Log, CowArray::from(lanes));
            caught
                .call_method1("__exit__", (py.None(), py.None(), py.None()))
                .expect("exit");
            drop(record);
            let result = result.expect("no warning is raised as an error");
            assert!(result[0].is_infinite() && result[0].is_sign_negative());
            assert!(result[1].is_nan());
        });
    }
}

mod errors {
    use fhy_core::expression::BigInt;

    use super::*;

    /// Return the class name and text of the exception of `error`.
    fn mapped(py: Python<'_>, error: EvaluationError) -> (PyErr, String) {
        let raised = evaluation_error_to_python(py, error);
        let text = raised.value(py).to_string();
        (raised, text)
    }

    #[test]
    fn shapes_and_sizes_are_value_errors_with_the_cores_text() {
        with_numpy(|py, _numpy| {
            let (error, text) = mapped(
                py,
                EvaluationError::Shape {
                    left: vec![2],
                    right: vec![3],
                },
            );
            assert!(error.is_instance_of::<PyValueError>(py));
            assert_eq!(text, "shapes [2] and [3] do not broadcast");
            let (error, _) = mapped(py, EvaluationError::BroadcastTooLarge { shape: vec![4, 4] });
            assert!(error.is_instance_of::<PyValueError>(py));
        });
    }

    #[test]
    fn memory_and_integer_range_errors_keep_their_builtin_classes() {
        with_numpy(|py, _numpy| {
            let (error, text) = mapped(py, EvaluationError::OutOfMemory { lanes: 1 << 40 });
            assert!(error.is_instance_of::<PyMemoryError>(py));
            assert_eq!(
                text,
                "cannot allocate the 1099511627776 lanes of the result"
            );
            let (error, text) = mapped(
                py,
                EvaluationError::IntegerOutOfRange(BigInt::from(i64::MAX) + 1),
            );
            assert!(error.is_instance_of::<PyOverflowError>(py));
            assert_eq!(
                text,
                "integer 9223372036854775808 is outside the 64-bit range"
            );
        });
    }

    #[test]
    fn booleans_used_as_numbers_are_type_errors() {
        with_numpy(|py, _numpy| {
            let node = Expression::literal(1_i64);
            let (error, _) = mapped(py, EvaluationError::BooleanArithmetic(node.clone()));
            assert!(error.is_instance_of::<PyTypeError>(py));
            let (error, _) = mapped(py, EvaluationError::MixedBranches(node));
            assert!(error.is_instance_of::<PyTypeError>(py));
        });
    }

    #[test]
    fn a_failed_lane_maps_by_its_failure() {
        with_numpy(|py, _numpy| {
            let lane = |failure| EvaluationError::Lane {
                failure,
                node: Expression::literal(1_i64),
                lane: Some(3),
            };
            let (error, text) = mapped(py, lane(LaneFailure::IntegerOverflow));
            assert!(error.is_instance_of::<PyOverflowError>(py));
            assert!(text.starts_with("integer overflow at lane 3"), "{text}");
            let (error, _) = mapped(py, lane(LaneFailure::OutOfRangeCast));
            assert!(error.is_instance_of::<PyOverflowError>(py));
            let (error, _) = mapped(py, lane(LaneFailure::DivisionByZero));
            assert!(error.is_instance_of::<PyZeroDivisionError>(py));
            let (error, _) = mapped(py, lane(LaneFailure::NegativeIntegerExponent));
            assert!(error.is_instance_of::<PyValueError>(py));
        });
    }

    #[test]
    fn a_kernels_python_exception_is_raised_unchanged() {
        with_numpy(|py, _numpy| {
            let raised = PyValueError::new_err("the kernel's own");
            let source: BoxError = Box::new(raised);
            let (error, text) = mapped(
                py,
                EvaluationError::Kernel {
                    function: BuiltinFunction::Exp,
                    source,
                },
            );
            assert!(error.is_instance_of::<PyValueError>(py));
            assert_eq!(text, "the kernel's own");
        });
    }
}
