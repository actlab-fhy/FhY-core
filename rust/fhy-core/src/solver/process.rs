//! A backend that drives an SMT-LIB2 executable over its standard input
//! and output.

use std::borrow::Cow;
use std::error::Error;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::thread;
use std::time::{Duration, Instant};

use super::backend::{CheckLimits, SatResult, SmtSolver};
use super::smt::SmtScript;
use crate::foreign::BoxError;

/// An [`SmtSolver`] that runs an SMT-LIB2 executable for each check, such
/// as `z3 -in` or `cvc5 --lang=smt2`.
///
/// Each check starts the program with its arguments, writes
/// `(set-option :print-success false)`, the script and `(check-sat)` to its
/// standard input, and reads the answer from its standard output: `sat`,
/// `unsat`, or `unknown`, after which it asks `(get-info :reason-unknown)`
/// and reads the reason; a program that closes its output before giving
/// one answers `unknown` with the reason
/// `"the solver closed its output before giving a reason"`. A `success`
/// line before the answer, which a solver that starts with
/// `:print-success` on may print for the first command, is skipped. The check then writes `(exit)` and closes the
/// input.
///
/// [`CheckLimits::timeout`] bounds the whole call. The input is written
/// and the output read on helper threads, so neither a program that stops
/// reading nor one that stops writing holds the caller. A program that has
/// not answered in time is killed, and the check answers `unknown` with
/// the reason `"timeout"`. After its answer, a program has until the
/// deadline to exit, or two seconds when there is none, and is then
/// killed. A program that closes its output without answering has two
/// seconds to exit, or until the deadline when that comes first; one that
/// does not is killed, and the check fails, or answers `unknown` with the
/// reason `"timeout"` when the deadline passed first.
///
/// On Unix, the program runs in a process group of its own, and a kill
/// signals the whole group, so a wrapper script's solver dies with it. The
/// group is signalled through the `kill` program, since this crate uses no
/// `unsafe` code. Elsewhere only the program itself is killed, so a
/// wrapper should `exec` its solver (`exec z3 -in`), as it should on Unix
/// too: a solver that is not the program itself outlives it when `kill`
/// cannot be run. The program being in its own group also means a
/// terminal's interrupt does not reach it.
///
/// The program is run as configured; nothing searches for a solver. Its
/// standard error is read as it is written, and the first 4 KiB of it are
/// kept for the error of a program that exits or closes its output before
/// answering.
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

    /// Return the command that starts the program, in a process group of
    /// its own on Unix.
    fn command(&self) -> Command {
        let mut command = Command::new(&self.program);
        command
            .args(&self.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        std::os::unix::process::CommandExt::process_group(&mut command, 0);
        command
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
    fn check(&self, script: &SmtScript, limits: &CheckLimits) -> Result<SatResult, BoxError> {
        let deadline = limits.timeout().map(|timeout| Instant::now() + timeout);
        let mut child = self
            .command()
            .spawn()
            .map_err(|source| ProcessError::Spawn {
                program: self.program.clone(),
                source,
            })?;
        let mut session = match Session::start(&mut child, deadline) {
            Ok(session) => session,
            Err(error) => {
                kill(&mut child);
                return Err(Box::new(error));
            }
        };
        let result = session.run(script);
        // Closing the input after `(exit)` lets a program that reads to the
        // end of its input finish.
        session.close_input();
        match result {
            Ok(answer) => {
                reap(&mut child, deadline);
                Ok(answer)
            }
            Err(Failure::TimedOut) => {
                kill(&mut child);
                Ok(timed_out())
            }
            Err(Failure::Closed) => {
                let grace_end = Instant::now() + EXIT_GRACE;
                let exit_by = deadline.map_or(grace_end, |deadline| deadline.min(grace_end));
                if let Some(status) = wait_until(&mut child, exit_by) {
                    let stderr = session.read_stderr();
                    return Err(Box::new(ProcessError::Exited { status, stderr }));
                }
                kill(&mut child);
                if exit_by < grace_end {
                    Ok(timed_out())
                } else {
                    let stderr = session.read_stderr();
                    Err(Box::new(ProcessError::ClosedOutput { stderr }))
                }
            }
            Err(Failure::Error(error)) => {
                kill(&mut child);
                Err(Box::new(error))
            }
        }
    }
}

/// The time a program has to exit after its last command when the check
/// has no deadline, and at most after closing its output unanswered.
const EXIT_GRACE: Duration = Duration::from_secs(2);

/// The most of a program's standard error that is kept, in bytes.
const STDERR_KEPT: usize = 4096;

/// The time the standard error of a program that exited or was killed has
/// to reach its end.
const STDERR_WAIT: Duration = Duration::from_millis(100);

/// The reason of an `unknown` answer whose program closed its output
/// before giving one.
const NO_REASON: &str = "the solver closed its output before giving a reason";

/// The longest pause between two looks at whether a program has exited.
const LONGEST_POLL: Duration = Duration::from_millis(10);

/// Return the answer of a check whose deadline passed.
fn timed_out() -> SatResult {
    SatResult::Unknown {
        reason: "timeout".to_owned(),
    }
}

/// Return the time by which a program must exit: the check's deadline, or
/// [`EXIT_GRACE`] from now when it has none.
fn exit_deadline(deadline: Option<Instant>) -> Instant {
    deadline.unwrap_or_else(|| Instant::now() + EXIT_GRACE)
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

/// The talk with one running program: the commands its input thread
/// writes, the lines its output thread reads, the start of its standard
/// error, which a third thread reads, and the deadline.
struct Session {
    /// The input thread's queue, or `None` once the input is closed. The
    /// thread closes the program's input when the queue is dropped and
    /// drained, or when a write fails.
    commands: Option<Sender<Vec<u8>>>,
    lines: Receiver<io::Result<String>>,
    /// The kept bytes of the standard error, in the chunks they were read.
    stderr: Receiver<Vec<u8>>,
    deadline: Option<Instant>,
}

impl Session {
    /// Take the program's input, output and standard error, and start the
    /// threads writing the first and reading the others.
    fn start(child: &mut Child, deadline: Option<Instant>) -> Result<Self, ProcessError> {
        let (Some(mut input), Some(output), Some(errors)) =
            (child.stdin.take(), child.stdout.take(), child.stderr.take())
        else {
            return Err(ProcessError::Io(io::Error::other(
                "the solver's standard streams are not piped",
            )));
        };
        let (commands, queued) = mpsc::channel::<Vec<u8>>();
        // A program that exits early, or stops reading and is killed,
        // refuses its input. That is no failure of its own: its output, or
        // its exit, then tells why, so the write's error ends the thread
        // and is dropped.
        thread::spawn(move || {
            for command in queued {
                if let Err(error) = input.write_all(&command).and_then(|()| input.flush()) {
                    drop(error);
                    return;
                }
            }
        });
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
            commands: Some(commands),
            lines,
            stderr: drain_stderr(errors),
            deadline,
        })
    }

    /// Check `script`, returning the answer.
    fn run(&self, script: &SmtScript) -> Result<SatResult, Failure> {
        self.send(format!("(set-option :print-success false)\n{script}(check-sat)\n").into_bytes());
        let answer = self.read_answer()?;
        let result = match answer.as_str() {
            "sat" => SatResult::Sat,
            "unsat" => SatResult::Unsat,
            "unknown" => {
                self.send(b"(get-info :reason-unknown)\n".to_vec());
                let reason = self.read_line().map(|line| read_reason(&line));
                match reason {
                    Ok(reason) => SatResult::Unknown { reason },
                    Err(Failure::Closed) => SatResult::Unknown {
                        reason: NO_REASON.to_owned(),
                    },
                    Err(failure) => return Err(failure),
                }
            }
            line if line.starts_with("(error") => {
                return Err(ProcessError::Solver(line.to_owned()).into());
            }
            line => return Err(ProcessError::UnexpectedAnswer(line.to_owned()).into()),
        };
        self.send(b"(exit)\n".to_vec());
        Ok(result)
    }

    /// Queue `text` for the program's input.
    ///
    /// A closed queue means the input thread stopped because the program
    /// refused its input; its output, or its exit, then tells why, so the
    /// text is dropped.
    fn send(&self, text: Vec<u8>) {
        if let Some(commands) = &self.commands {
            if let Err(refused) = commands.send(text) {
                drop(refused);
            }
        }
    }

    /// Close the program's input once the queued commands are written.
    fn close_input(&mut self) {
        self.commands = None;
    }

    /// Return the kept start of the standard error of the program, which
    /// exited or was killed, lossily decoded, once its end is read or
    /// [`STDERR_WAIT`] has passed.
    fn read_stderr(&self) -> String {
        let by = Instant::now() + STDERR_WAIT;
        let mut kept = Vec::new();
        while let Ok(chunk) = self
            .stderr
            .recv_timeout(by.saturating_duration_since(Instant::now()))
        {
            kept.extend(chunk);
        }
        String::from_utf8_lossy(&kept).into_owned()
    }

    /// Return the first line of output that is neither blank nor
    /// `success`.
    fn read_answer(&self) -> Result<String, Failure> {
        loop {
            let line = self.read_line()?;
            if line != "success" {
                return Ok(line);
            }
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

/// Start the thread reading the standard error `errors` to its end, so
/// the program never blocks on a full pipe, and return the queue of the
/// chunks of its first [`STDERR_KEPT`] bytes.
///
/// A read that fails ends the thread: the standard error only adds detail
/// to another failure, so its error is dropped.
fn drain_stderr(mut errors: impl Read + Send + 'static) -> Receiver<Vec<u8>> {
    let (sender, chunks) = mpsc::channel();
    thread::spawn(move || {
        let mut buffer = [0; 1024];
        let mut kept = 0;
        loop {
            let read = match errors.read(&mut buffer) {
                Ok(0) => return,
                Ok(read) => read,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => {
                    drop(error);
                    return;
                }
            };
            let keep = read.min(STDERR_KEPT - kept);
            if keep > 0 {
                kept += keep;
                // A check that no longer waits has dropped the queue; the
                // rest is still drained.
                if let Err(unwanted) = sender.send(buffer[..keep].to_vec()) {
                    drop(unwanted);
                }
            }
        }
    });
    chunks
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

/// Return the program's exit status once it exits, or `None` if it has
/// not by `deadline` or the look fails.
///
/// The pause between looks doubles from 100 µs up to [`LONGEST_POLL`], so
/// a program that exits at once is seen at once.
fn wait_until(child: &mut Child, deadline: Instant) -> Option<ExitStatus> {
    let mut pause = Duration::from_micros(100);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Some(status),
            Ok(None) => {}
            Err(error) => {
                drop(error);
                return None;
            }
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return None;
        }
        thread::sleep(pause.min(remaining));
        pause = (pause * 2).min(LONGEST_POLL);
    }
}

/// Let the program, which has answered, exit by `deadline`, or within
/// [`EXIT_GRACE`] when there is none, and kill it if it does not.
///
/// The check has its answer, so a failure to wait changes nothing.
fn reap(child: &mut Child, deadline: Option<Instant>) {
    if wait_until(child, exit_deadline(deadline)).is_none() {
        kill(child);
    }
}

/// Kill the program, and on Unix its process group, and wait for it.
///
/// A program that already exited cannot be killed, and the check's outcome
/// does not depend on the wait, so every error is dropped.
fn kill(child: &mut Child) {
    #[cfg(unix)]
    kill_group(child.id());
    if let Err(error) = child.kill() {
        drop(error);
    }
    if let Err(error) = child.wait() {
        drop(error);
    }
}

/// Send `SIGKILL` to the process group `group`, through the `kill`
/// program.
///
/// The group's leader, the program, is not yet reaped when this runs, so
/// the group id cannot have been reused.
#[cfg(unix)]
fn kill_group(group: u32) {
    let signalled = Command::new("kill")
        .args(["-KILL", "--", &format!("-{group}")])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    if let Err(error) = signalled {
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
    Exited {
        /// Its exit status.
        status: ExitStatus,
        /// The first 4 KiB of its standard error, lossily decoded.
        stderr: String,
    },
    /// The program closed its output before answering, and did not exit
    /// within two seconds of closing it, so it was killed. A check whose
    /// deadline comes first answers `unknown` for it instead, at the
    /// deadline.
    ClosedOutput {
        /// The first 4 KiB of its standard error, lossily decoded.
        stderr: String,
    },
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
            Self::Exited { status, stderr } => {
                write!(f, "the solver exited before answering ({status})")?;
                write_stderr(f, stderr)
            }
            Self::ClosedOutput { stderr } => {
                f.write_str("the solver closed its output before answering")?;
                write_stderr(f, stderr)
            }
        }
    }
}

/// Write `stderr`, trimmed and quoted on one line, unless it is blank.
fn write_stderr(f: &mut fmt::Formatter<'_>, stderr: &str) -> fmt::Result {
    let stderr = stderr.trim();
    if stderr.is_empty() {
        return Ok(());
    }
    write!(f, "; its standard error: {stderr:?}")
}

impl Error for ProcessError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Spawn { source, .. } | Self::Io(source) => Some(source),
            Self::Solver(_)
            | Self::UnexpectedAnswer(_)
            | Self::Exited { .. }
            | Self::ClosedOutput { .. } => None,
        }
    }
}
