//! `PyO3` classes and functions for [`fhy_core::solver`]: the bases of
//! the Python `SmtSolver` and `Simplifier` ABCs and the adapters that drive
//! a Python backend, the process backend, `SmtScript` and `SatResult`,
//! the `Solver` facade, the default solver the module functions of
//! `fhy_core.symbolic.solver` ask, and the core's SymPy backend.

mod backends;
mod error;
mod facade;
mod state;
mod sympy;
mod values;

pub(crate) use backends::{
    PySimplifierBase, PySimplifyContext, PySmtLib2ProcessSolver, PySmtSolverBase,
};
pub(crate) use error::{is_pass_execution_failure, solve_error_to_py, warn_hazard, warn_unknown};
pub(crate) use facade::{PySolver, read_limits};
pub(crate) use state::{get_default_solver, set_default_solver};
pub(crate) use sympy::PySympySimplifier;
pub(crate) use values::{PySatResult, PySmtScript, read_symbol_types, symbol_type_to_python};
