//! A backend that drives an SMT-LIB2 executable over its standard input
//! and output.

use std::borrow::Cow;
use std::error::Error;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, ExitStatus, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

use super::backend::{BackendError, CheckLimits, SatResult, SmtSolver};
use super::smt::SmtScript;

/// An [`SmtSolver`] that runs an SMT-LIB2 executable for each check, such
/// as `z3 -in` or `cvc5 --lang=smt2`.
///
/// Each check starts the program with its arguments, writes the script and
/// `(check-sat)` to its standard input, and reads the answer from its
/// standard output: `sat`, `unsat`, or `unknown`, after which it asks
/// `(get-info :reason-unknown)` and reads the reason. It then writes
/// `(exit)`. The output is read on a helper thread, so the wait for an
/// answer honors [`CheckLimits::timeout`]: a program that has not answered
/// in time is killed, and the check answers `unknown` with the reason
/// `"timeout"`.
///
/// The program is run as configured; nothing searches for a solver. Its
/// standard error is discarded.
///
/// # Examples
///
/// ```
/// use fhy_core::solver::SmtLib2Process;
///
/// let z3 = SmtLib2Process::new("z3").with_args(["-in"]);
///
/// assert_eq!(z3.program(), "z3");
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmtLib2Process {
    program: OsString,
    args: Vec<OsString>,
}

impl SmtLib2Process {
    /// Return the backend running `program` with no arguments.
    #[must_use]
    pub fn new(program: impl Into<OsString>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
        }
    }

    /// Return this backend running its program with `args`, in place of
    /// the arguments it had.
    #[must_use]
    pub fn with_args<I, A>(self, args: I) -> Self
    where
        I: IntoIterator<Item = A>,
        A: Into<OsString>,
    {
        Self {
            args: args.into_iter().map(Into::into).collect(),
            ..self
        }
    }

    /// Return the program the backend runs.
    #[must_use]
    pub fn program(&self) -> &OsStr {
        &self.program
    }

    /// Return the arguments the program runs with.
    #[must_use]
    pub fn args(&self) -> &[OsString] {
        &self.args
    }
}

impl SmtSolver for SmtLib2Process {
    /// Return the file name of the program, such as `z3` for
    /// `/usr/bin/z3`.
    fn name(&self) -> Cow<'_, str> {
        Path::new(&self.program)
            .file_name()
            .unwrap_or(&self.program)
            .to_string_lossy()
    }

    /// Run the program on `script`, as the type describes.
    ///
    /// # Errors
    ///
    /// Returns a [`ProcessError`], boxed.
    fn check(&self, script: &SmtScript, limits: &CheckLimits) -> Result<SatResult, BackendError> {
        let mut child = Command::new(&self.program)
            .args(&self.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|source| ProcessError::Spawn {
                program: self.program.clone(),
                source,
            })?;
        let session = Session::start(&mut child, limits.timeout());
        let result = session.and_then(|mut session| session.run(script));
        match result {
            Ok(answer) => {
                reap(&mut child);
                Ok(answer)
            }
            Err(Failure::TimedOut) => {
                kill(&mut child);
                Ok(SatResult::Unknown {
                    reason: "timeout".to_owned(),
                })
            }
            Err(Failure::Closed) => {
                let status = child.wait().ok();
                Err(Box::new(ProcessError::Exited(status)))
            }
            Err(Failure::Error(error)) => {
                kill(&mut child);
                Err(Box::new(error))
            }
        }
    }
}

/// Why a session ended without an answer.
enum Failure {
    /// The deadline passed.
    TimedOut,
    /// The program closed its output.
    Closed,
    /// The program failed in another way.
    Error(ProcessError),
}

impl From<ProcessError> for Failure {
    fn from(error: ProcessError) -> Self {
        Self::Error(error)
    }
}

/// The talk with one running program: its input, the lines its output
/// thread reads, and the deadline.
struct Session {
    input: ChildStdin,
    lines: Receiver<io::Result<String>>,
    deadline: Option<Instant>,
}

impl Session {
    /// Take the program's input, and start the thread reading its output.
    fn start(child: &mut Child, timeout: Option<Duration>) -> Result<Self, Failure> {
        let (Some(input), Some(output)) = (child.stdin.take(), child.stdout.take()) else {
            return Err(Failure::Error(ProcessError::Io(io::Error::other(
                "the solver's standard streams are not piped",
            ))));
        };
        let (sender, lines) = mpsc::channel();
        thread::spawn(move || {
            for line in BufReader::new(output).lines() {
                let is_error = line.is_err();
                if sender.send(line).is_err() || is_error {
                    return;
                }
            }
        });
        Ok(Self {
            input,
            lines,
            deadline: timeout.map(|timeout| Instant::now() + timeout),
        })
    }

    /// Check `script`, returning the answer.
    fn run(&mut self, script: &SmtScript) -> Result<SatResult, Failure> {
        self.send(format!("{script}(check-sat)\n").as_bytes());
        let answer = self.read_line()?;
        let result = match answer.as_str() {
            "sat" => SatResult::Sat,
            "unsat" => SatResult::Unsat,
            "unknown" => {
                self.send(b"(get-info :reason-unknown)\n");
                let reason = self.read_line().map(|line| read_reason(&line));
                match reason {
                    Ok(reason) => SatResult::Unknown { reason },
                    Err(Failure::Closed) => SatResult::Unknown {
                        reason: String::new(),
                    },
                    Err(failure) => return Err(failure),
                }
            }
            line if line.starts_with("(error") => {
                return Err(ProcessError::Solver(line.to_owned()).into());
            }
            line => return Err(ProcessError::UnexpectedAnswer(line.to_owned()).into()),
        };
        self.send(b"(exit)\n");
        Ok(result)
    }

    /// Write `text` to the program's input.
    ///
    /// A program that exits early refuses its input. That is no failure of
    /// its own: its output, or its exit, then tells why, so the write's
    /// error is dropped.
    fn send(&mut self, text: &[u8]) {
        if let Err(error) = self.input.write_all(text).and_then(|()| self.input.flush()) {
            drop(error);
        }
    }

    /// Return the next line of output that is not blank.
    fn read_line(&self) -> Result<String, Failure> {
        loop {
            let received = match self.deadline {
                None => self.lines.recv().map_err(|_disconnected| Failure::Closed),
                Some(deadline) => {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    self.lines
                        .recv_timeout(remaining)
                        .map_err(|error| match error {
                            RecvTimeoutError::Timeout => Failure::TimedOut,
                            RecvTimeoutError::Disconnected => Failure::Closed,
                        })
                }
            }?;
            let line = received.map_err(ProcessError::Io)?;
            let line = line.trim();
            if !line.is_empty() {
                return Ok(line.to_owned());
            }
        }
    }
}

/// Return the reason of a `(:reason-unknown ...)` line, unquoted, or the
/// empty string for any other line.
fn read_reason(line: &str) -> String {
    line.strip_prefix("(:reason-unknown")
        .and_then(|rest| rest.strip_suffix(')'))
        .map(str::trim)
        .map(|reason| {
            reason
                .strip_prefix('"')
                .and_then(|reason| reason.strip_suffix('"'))
                .unwrap_or(reason)
                .to_owned()
        })
        .unwrap_or_default()
}

/// Close the program's input and wait for it to exit.
///
/// The check has its answer, so a failure to wait changes nothing.
fn reap(child: &mut Child) {
    drop(child.stdin.take());
    if let Err(error) = child.wait() {
        drop(error);
    }
}

/// Kill the program and wait for it.
///
/// A program that already exited cannot be killed, and the check's outcome
/// does not depend on the wait, so both errors are dropped.
fn kill(child: &mut Child) {
    if let Err(error) = child.kill() {
        drop(error);
    }
    if let Err(error) = child.wait() {
        drop(error);
    }
}

/// A failure of an [`SmtLib2Process`] check.
#[derive(Debug)]
#[non_exhaustive]
pub enum ProcessError {
    /// The program could not be started.
    Spawn {
        /// The program.
        program: OsString,
        /// Why.
        source: io::Error,
    },
    /// Talking to the program failed.
    Io(io::Error),
    /// The program answered with an `(error ...)` line, given whole.
    Solver(String),
    /// The program answered something other than `sat`, `unsat` or
    /// `unknown`, given whole.
    UnexpectedAnswer(String),
    /// The program exited before answering.
    Exited(Option<ExitStatus>),
}

impl fmt::Display for ProcessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spawn { program, .. } => {
                write!(f, "cannot start the solver {:?}", program.to_string_lossy())
            }
            Self::Io(_) => f.write_str("cannot talk to the solver"),
            Self::Solver(line) => write!(f, "the solver reported {line}"),
            Self::UnexpectedAnswer(line) => {
                write!(f, "the solver answered {line:?}, not sat, unsat or unknown")
            }
            Self::Exited(Some(status)) => {
                write!(f, "the solver exited before answering ({status})")
            }
            Self::Exited(None) => f.write_str("the solver exited before answering"),
        }
    }
}

impl Error for ProcessError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Spawn { source, .. } | Self::Io(source) => Some(source),
            Self::Solver(_) | Self::UnexpectedAnswer(_) | Self::Exited(_) => None,
        }
    }
}
