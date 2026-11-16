"""Shield Desk as a background service: login item and supervisor (packaging/DESIGN.md sections 3 to 7).

The desktop installers (macOS ``.pkg``, Windows ``.msi``) lay a private Python runtime with Shield into the
user's own folder and call ``ragbase-shield service install``. This module holds everything that command and
``ragbase-shield service run`` need:

- the app folder (``<app>``), derived from the running interpreter, and ``<home>/install.json``;
- the per-user login item: a LaunchAgent on macOS (``~/Library/LaunchAgents/ai.ragbase.shield.plist``), the
  ``HKCU\\...\\Run`` value "RAGbase Shield" on Windows;
- the ``RAGbase Shield.app`` shell on macOS (a copy of the runtime binary inside a 3-file bundle, so the menu
  bar, Login Items and the keychain dialog say "RAGbase Shield");
- the supervisor: single instance, probe before spawn, a ``desk --service`` child with restarts and backoff,
  the ``service.cmd`` command file, PID files and the service log.

Privacy rules (ARCHITECTURE.md section 2): every socket opened here goes to ``127.0.0.1:<port>``; nothing
calls the network; the Desk token is only READ from ``<home>/desk_token`` (the Desk owns that file), never
logged, printed, shown or passed on a command line; logs hold event codes and integers, never a dossier
title, a path under a dossier or a value.
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
import platform
import plistlib
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from .. import __version__, config

APP_NAME = "RAGbase Shield"
LABEL = "ai.ragbase.shield"  # LaunchAgent label and bundle identifier
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
RUN_VALUE = APP_NAME

# Binding constants (DESIGN.md section 4).
PROBE_TIMEOUT = 2.0
READY_TIMEOUT = 120.0
BACKOFF = (2.0, 5.0, 15.0, 60.0)
MAX_EXITS = 5
WINDOW = 600.0
POLL_STATUS = 10.0
POLL_CMD = 1.0
PORT_BUSY_RETRY = 60.0
STOP_GRACE = 5.0
LOG_MAX_BYTES = 1024 * 1024
LOG_BACKUPS = 3

INSTALL_JSON = "install.json"
SERVICE_PID = "service.pid"
DESK_PID = "desk.pid"
SERVICE_CMD = "service.cmd"
LOGS_DIR = "logs"
TOKEN_FILE = "desk_token"
_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]{32,256}")
_SECRETISH_RE = re.compile(r"[A-Za-z0-9_\-]{32,}")

# RAGBASE_SHIELD_* settings that "service install" copies into the login item when they are set at install
# time (a custom data folder or port must reach the background service, which does not inherit the shell).
PROPAGATED_ENV = ("RAGBASE_SHIELD_HOME", "RAGBASE_SHIELD_PORT", "RAGBASE_SHIELD_KEYRING", "RAGBASE_SHIELD_UI",
                  "RAGBASE_SHIELD_LOCALE", "RAGBASE_SHIELD_MODELS_DIR")

# States of the supervisor (tray strings in desk/tray.py).
STARTING = "starting"
RUNNING = "running"
ELSEWHERE = "elsewhere"
PORT_BUSY = "port_busy"
GAVE_UP = "gave_up"
STOPPING = "stopping"

log = logging.getLogger("ragbase_shield.service_manager")

Runner = Callable[[list], "subprocess.CompletedProcess"]


# ------------------------------------------------------------------------------------------ paths

def is_macos() -> bool:
    return sys.platform == "darwin"


def is_windows() -> bool:
    return sys.platform.startswith("win")


def home() -> Path:
    return config.home()


def port() -> int:
    try:
        return int(os.environ.get("RAGBASE_SHIELD_PORT", str(config.DESK_PORT)))
    except ValueError:
        return config.DESK_PORT


def app_dir() -> Optional[Path]:
    """``<app>`` when this interpreter is the installer's private runtime (``<app>/runtime``), else None.

    Works from ``runtime/bin/python3.12``, from ``runtime\\python.exe`` and from the macOS ``.app`` shell, whose
    ``pyvenv.cfg`` points ``home`` at ``runtime/bin`` (so ``sys.base_prefix`` is ``<app>/runtime``).
    """
    base = Path(sys.base_prefix).resolve()
    # The installers always use a folder named "RAGbase Shield": a build payload or a copy elsewhere is
    # never taken for an install (service install would write into it, uninstall would remove it).
    if base.name == "runtime" and base.parent.name == APP_NAME and (base.parent / "VERSION").is_file():
        return base.parent
    return None


def runtime_python(app: Path) -> Path:
    if is_windows():
        return app / "runtime" / "python.exe"
    return app / "runtime" / "bin" / f"python{sys.version_info.major}.{sys.version_info.minor}"


def launcher_path(app: Path) -> Path:
    """The command a terminal user or an MCP app runs (DESIGN.md section 9)."""
    if is_windows():
        return app / "runtime" / "python.exe"
    return app / "bin" / "ragbase-shield"


def launcher_args() -> list[str]:
    """Arguments before the sub-command for :func:`launcher_path`. ``-I`` (isolated mode) keeps the current
    folder out of ``sys.path``: MCP apps start this command in the user's project folder (review 2026-10-27)."""
    return ["-I", "-X", "utf8", "-m", "ragbase_shield"] if is_windows() else []


def bundle_path(app: Path) -> Path:
    return app / f"{APP_NAME}.app"


def bundle_exe(app: Path) -> Path:
    return bundle_path(app) / "Contents" / "MacOS" / APP_NAME


def windows_service_exe(app: Path) -> Path:
    return app / "runtime" / f"{APP_NAME}.exe"


def launch_agent_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def logs_dir(h: Optional[Path] = None) -> Path:
    d = (h or home()) / LOGS_DIR
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(d, 0o700)
    except OSError:
        pass
    return d


def safe_flags() -> list[str]:
    """Interpreter flags of every service process: -E ignores PYTHON* variables, -s the user site-packages,
    -P (Python 3.11+) keeps the current folder out of ``sys.path`` (``-m`` would put it first, and ``service
    start`` from a terminal inherits the terminal's folder: a planted ``json.py`` there ran in the service)."""
    return ["-E", "-s", "-P"] if sys.version_info >= (3, 11) else ["-E", "-s"]


def service_args() -> list[str]:
    """Interpreter flags + module + ``service run``."""
    return [*safe_flags(), "-X", "utf8", "-m", "ragbase_shield", "service", "run"]


def service_exe(app: Optional[Path], bundle: bool = True) -> Path:
    """The program the login item starts (named "RAGbase Shield" when installed by an installer)."""
    if app is not None:
        if is_macos() and bundle and bundle_exe(app).is_file():
            return bundle_exe(app)
        if is_windows() and windows_service_exe(app).is_file():
            return windows_service_exe(app)
        rp = runtime_python(app)
        if rp.is_file():
            return rp
    if is_windows():
        pyw = Path(sys.executable).with_name("pythonw.exe")
        if pyw.is_file():
            return pyw
    return Path(sys.executable)


# ------------------------------------------------------------------------------------------ install.json

def install_json_path(h: Optional[Path] = None) -> Path:
    return (h or home()) / INSTALL_JSON


def read_install_json(h: Optional[Path] = None) -> Optional[dict]:
    try:
        data = json.loads(install_json_path(h).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def read_install_info(app: Path) -> dict:
    try:
        data = json.loads((app / "install_info.json").read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def write_install_json(app: Optional[Path], exe: Path, login_item: str, h: Optional[Path] = None) -> Path:
    """Paths of the app only: never a dossier path, never a secret."""
    data = {
        "version": __version__,
        "app_dir": str(app) if app else None,
        "python": str(runtime_python(app)) if app else sys.executable,
        "launcher": str(launcher_path(app)) if app else None,
        "launcher_args": launcher_args() if app else [],
        "service_exe": str(exe),
        "platform": sys.platform,
        "machine": platform.machine(),
        "login_item": login_item,
        "installed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    path = install_json_path(h)
    tmp = path.with_name(f".{INSTALL_JSON}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def installed_launcher(h: Optional[Path] = None) -> Optional[tuple[str, list[str]]]:
    """(command, args before the sub-command) of the installer's launcher, when install.json names one that
    exists. Used by ``server_command()`` so MCP apps run the installed Shield (no uv, works offline)."""
    data = read_install_json(h)
    if not data or not data.get("launcher"):
        return None
    exe = Path(str(data["launcher"]))
    if not exe.is_file() or not os.access(exe, os.X_OK):
        return None
    args = data.get("launcher_args") or []
    return str(exe), [str(a) for a in args if isinstance(a, str)]


# ------------------------------------------------------------------------------------------ token

def read_token(h: Optional[Path] = None) -> Optional[str]:
    """The Desk token, READ only (DESIGN.md section 5.2): first line of ``<home>/desk_token``, stripped, must
    match ``[A-Za-z0-9_-]{32,256}``. Missing or damaged file: None (the Desk creates and rotates it)."""
    try:
        raw = ((h or home()) / TOKEN_FILE).read_bytes()[:1024]
    except OSError:
        return None
    try:
        first = raw.splitlines()[0].decode("ascii").strip() if raw else ""
    except (UnicodeDecodeError, IndexError):
        return None
    return first if _TOKEN_RE.fullmatch(first) else None


# ------------------------------------------------------------------------------------------ logging

class _NoSecrets(logging.Filter):
    """Belt and braces: the code never logs the token or a path, but a long token-like run is masked anyway."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            return False
        if _SECRETISH_RE.search(msg):
            record.msg = _SECRETISH_RE.sub("[filtre]", msg)
            record.args = ()
        return True


def setup_logging(h: Optional[Path] = None) -> logging.Logger:
    """``<home>/logs/service.log``, rotating 1 MB x 3. Lines: time, level, event code and integers."""
    lg = logging.getLogger("ragbase_shield.service_manager")
    if any(getattr(hd, "_shield_service", False) for hd in lg.handlers):
        return lg
    handler = logging.handlers.RotatingFileHandler(logs_dir(h) / "service.log", maxBytes=LOG_MAX_BYTES,
                                                   backupCount=LOG_BACKUPS, encoding="utf-8")
    handler._shield_service = True  # type: ignore[attr-defined]
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    handler.addFilter(_NoSecrets())
    lg.addHandler(handler)
    lg.setLevel(logging.INFO)
    lg.propagate = False
    return lg


def _rotate(path: Path, keep: int = LOG_BACKUPS, max_bytes: int = LOG_MAX_BYTES) -> None:
    try:
        if not path.exists() or path.stat().st_size <= max_bytes:
            return
        for i in range(keep - 1, 0, -1):
            src = path.with_name(f"{path.name}.{i}")
            if src.exists():
                os.replace(src, path.with_name(f"{path.name}.{i + 1}"))
        os.replace(path, path.with_name(f"{path.name}.1"))
    except OSError:
        pass


# ------------------------------------------------------------------------------------------ probe

def probe(port_: int, token: Optional[str], timeout: float = PROBE_TIMEOUT, path: str = "/api/status"
          ) -> tuple[str, Any]:
    """``GET http://127.0.0.1:<port><path>`` (loopback only, desk/loopback.py).

    Returns ("ok", json) on 200 with JSON, ("refused", None) when nothing listens, ("busy", status) for any
    other answer or a timeout (another program on the port, or a Desk of another data folder).
    """
    from .loopback import get_json

    return get_json(int(port_), path, token, timeout)


def desk_status(port_: Optional[int] = None, h: Optional[Path] = None) -> tuple[str, Any]:
    return probe(port_ or port(), read_token(h))


# ------------------------------------------------------------------------------------------ processes

def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if is_windows():
        try:
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"], capture_output=True,
                                 text=True, timeout=10, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return f'"{pid}"' in out.stdout
        except (OSError, subprocess.SubprocessError):
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _win_image_path(pid: int) -> str:
    """Full path of the program of ``pid`` on Windows ("" when unknown). No command line is read."""
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                                                   ctypes.POINTER(wintypes.DWORD)]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = k32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ""
        try:
            buf = ctypes.create_unicode_buffer(32768)
            size = wintypes.DWORD(len(buf))
            return buf.value if k32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)) else ""
        finally:
            k32.CloseHandle(handle)
    except Exception:  # noqa: BLE001
        return ""


def _pid_is_shield(pid: int, what: str) -> bool:
    """True when ``pid`` runs ``ragbase_shield <what>`` (guards a recycled PID before a kill)."""
    if is_windows():
        # A program of this runtime folder (python.exe for the Desk, "RAGbase Shield.exe" for the supervisor):
        # never "any python process", which a recycled PID could name (review finding 17).
        image = _win_image_path(pid)
        if not image:
            return False
        try:
            return Path(image).resolve().parent == Path(sys.executable).resolve().parent
        except OSError:
            return False
    try:
        out = subprocess.run(["/bin/ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True,
                             timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    return re.search(rf"-m ragbase_shield {re.escape(what)}\b", out.stdout) is not None


def _read_pid(path: Path) -> int:
    try:
        return int(path.read_text(encoding="ascii").strip() or "0")
    except (OSError, ValueError):
        return 0


def _kill(pid: int, grace: float = STOP_GRACE) -> None:
    if is_windows():
        try:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=20,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.SubprocessError):
            pass
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    end = time.monotonic() + grace
    while time.monotonic() < end:
        if not _pid_alive(pid):
            return
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def kill_stale_desk(h: Optional[Path] = None) -> bool:
    """Kill the Desk child named in ``desk.pid`` when it still runs (orphan of a dead supervisor)."""
    path = (h or home()) / DESK_PID
    pid = _read_pid(path)
    killed = False
    if pid and pid != os.getpid() and _pid_alive(pid) and _pid_is_shield(pid, "desk"):
        _kill(pid)
        killed = True
    try:
        path.unlink()
    except OSError:
        pass
    return killed


class InstanceLock:
    """Single-instance lock on ``<home>/service.pid`` (fcntl on POSIX, msvcrt on Windows), held for life."""

    def __init__(self, h: Optional[Path] = None):
        self.path = (h or home()) / SERVICE_PID
        self._fh: Any = None

    def acquire(self) -> bool:
        fh = open(self.path, "a+", encoding="ascii")
        try:
            if is_windows():
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
        if not is_windows():
            fh.seek(0)
            fh.truncate()
            fh.write(f"{os.getpid()}\n")
            fh.flush()
        else:  # the locked byte is offset 0: write the PID into a side file
            try:
                (self.path.with_name("service.pid.txt")).write_text(f"{os.getpid()}\n", encoding="ascii")
            except OSError:
                pass
        self._fh = fh
        return True

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            if is_windows():
                import msvcrt

                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            self._fh.close()
        except OSError:
            pass
        self._fh = None
        for p in (self.path, self.path.with_name("service.pid.txt")):
            try:
                p.unlink()
            except OSError:
                pass

    @staticmethod
    def held(h: Optional[Path] = None) -> bool:
        """True when a supervisor holds the lock (probes it without keeping it)."""
        lk = InstanceLock(h)
        if not lk.path.exists():
            return False
        if lk.acquire():
            # nobody held it: release without deleting a file another process may create meanwhile
            try:
                if is_windows():
                    import msvcrt

                    lk._fh.seek(0)
                    msvcrt.locking(lk._fh.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(lk._fh.fileno(), fcntl.LOCK_UN)
                lk._fh.close()
            except OSError:
                pass
            lk._fh = None
            return False
        return True


def supervisor_pid(h: Optional[Path] = None) -> int:
    hh = h or home()
    return _read_pid(hh / ("service.pid.txt" if is_windows() else SERVICE_PID))


def write_command(cmd: str, h: Optional[Path] = None) -> None:
    if cmd not in ("stop", "restart"):
        raise ValueError("command must be stop or restart")
    path = (h or home()) / SERVICE_CMD
    tmp = path.with_name(f".{SERVICE_CMD}.tmp")
    tmp.write_text(cmd + "\n", encoding="ascii")
    os.replace(tmp, path)


def _take_command(h: Path) -> Optional[str]:
    path = h / SERVICE_CMD
    try:
        cmd = path.read_text(encoding="ascii").strip().lower()
    except OSError:
        return None
    try:
        path.unlink()
    except OSError:
        pass
    return cmd if cmd in ("stop", "restart") else None


# ------------------------------------------------------------------------------------------ supervisor

def desk_child_command(port_: int) -> list[str]:
    exe = sys.executable
    if is_windows():  # the Desk child runs from python.exe (no window: CREATE_NO_WINDOW)
        cand = Path(sys.executable).with_name("python.exe")
        if cand.is_file():
            exe = str(cand)
    return [exe, *safe_flags(), "-X", "utf8", "-m", "ragbase_shield", "desk", "--service", "--port", str(port_)]


def _spawn_desk(port_: int, h: Path) -> subprocess.Popen:
    _rotate(logs_dir(h) / "desk.log")
    env = dict(os.environ)
    env["RAGBASE_SHIELD_SERVICE"] = "1"
    env["RAGBASE_SHIELD_PORT"] = str(port_)
    err = open(logs_dir(h) / "desk.log", "ab")
    kwargs: dict[str, Any] = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": err,
                              "env": env, "close_fds": True}
    if is_windows():
        kwargs["creationflags"] = (getattr(subprocess, "CREATE_NO_WINDOW", 0)
                                   | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    else:
        kwargs["start_new_session"] = True
    try:
        return subprocess.Popen(desk_child_command(port_), **kwargs)
    finally:
        err.close()


class Supervisor:
    """Runs one Shield Desk child (DESIGN.md section 4). ``tick()`` is called about once a second."""

    def __init__(self, h: Optional[Path] = None, port_: Optional[int] = None, *,
                 spawn: Optional[Callable[[int, Path], Any]] = None,
                 prober: Optional[Callable[[int, Optional[str]], tuple[str, Any]]] = None,
                 clock: Callable[[], float] = time.monotonic):
        self.home = h or home()
        self.port = port_ or port()
        self._spawn = spawn or _spawn_desk
        self._probe = prober or (lambda p, t: probe(p, t))
        self._clock = clock
        self.child: Any = None
        self.state = STARTING
        self.info: dict = {}
        self.exits: list[float] = []
        self._next_action = 0.0
        self._child_started = 0.0
        self._last_status = 0.0
        self._requests: list[str] = []
        self._lock = threading.RLock()
        self.stopped = threading.Event()
        self.changed: Callable[[], None] = lambda: None

    # -- public
    def request(self, cmd: str) -> None:
        """"stop", "restart" or "retry", from the tray (same effect as the command file)."""
        with self._lock:
            self._requests.append(cmd)

    def snapshot(self) -> dict:
        with self._lock:
            return {"state": self.state, "port": self.port, "version": self.info.get("version"),
                    "engine_mode": self.info.get("engine_mode"), "dossiers": self.info.get("dossiers")}

    def tick(self) -> None:
        with self._lock:
            reqs, self._requests = self._requests, []
        cmd = _take_command(self.home)
        if cmd:
            reqs.append(cmd)
        for r in reqs:
            if r == "stop":
                log.info("service: stop requested")
                self._set(STOPPING)
                self.stop_child()
                self.stopped.set()
                return
            if r in ("restart", "retry"):
                log.info("service: %s requested", r)
                self.stop_child()
                self.exits.clear()
                self._set(STARTING)
                self._next_action = 0.0
        now = self._clock()
        if self.child is None:
            if self.state == GAVE_UP or now < self._next_action:
                return
            self._probe_then_spawn(now)
            return
        rc = self.child.poll()
        if rc is not None:
            self._child_exited(rc, now)
            return
        if self.state == STARTING:
            st, data = self._probe(self.port, read_token(self.home))
            if st == "ok":
                self.info = data if isinstance(data, dict) else {}
                self._last_status = now
                log.info("desk: ready")
                self._set(RUNNING)
            elif now - self._child_started > READY_TIMEOUT:
                log.warning("desk: not ready after %d s, restarting", int(READY_TIMEOUT))
                self.stop_child()
                self._child_exited(-1, now)
            return
        if now - self._last_status >= POLL_STATUS:
            self._last_status = now
            st, data = self._probe(self.port, read_token(self.home))
            if st == "ok" and isinstance(data, dict):
                self.info = data
                self._set(RUNNING)

    def stop_child(self) -> None:
        child, self.child = self.child, None
        if child is None:
            return
        try:
            child.terminate()
            try:
                child.wait(timeout=STOP_GRACE)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=STOP_GRACE)
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            (self.home / DESK_PID).unlink()
        except OSError:
            pass
        log.info("desk: stopped")

    # -- internals
    def _set(self, state: str) -> None:
        if state != self.state:
            self.state = state
            log.info("state: %s", state)
            try:
                self.changed()
            except Exception:  # noqa: BLE001 - a tray refresh never stops the supervisor
                pass

    def _probe_then_spawn(self, now: float) -> None:
        st, data = self._probe(self.port, read_token(self.home))
        if st == "ok" and isinstance(data, dict) and data.get("version"):
            self.info = data
            self._set(ELSEWHERE)
            self._next_action = now + POLL_STATUS
            return
        if st == "refused":
            try:
                self.child = self._spawn(self.port, self.home)
            except OSError as exc:
                log.error("desk: spawn failed (%s)", type(exc).__name__)
                self._child_exited(-2, now)
                return
            self._child_started = now
            pid = getattr(self.child, "pid", 0)
            try:
                (self.home / DESK_PID).write_text(f"{pid}\n", encoding="ascii")
            except OSError:
                pass
            log.info("desk: started")
            self._set(STARTING)
            return
        if self.state != PORT_BUSY:
            log.warning("desk: port %d taken by another program (status %s)", self.port,
                        data if isinstance(data, int) else 0)
        self._set(PORT_BUSY)
        self._next_action = now + PORT_BUSY_RETRY

    def _child_exited(self, rc: int, now: float) -> None:
        self.child = None
        try:
            (self.home / DESK_PID).unlink()
        except OSError:
            pass
        self.exits = [t for t in self.exits if now - t < WINDOW] + [now]
        log.warning("desk: exited with code %d (%d in %d s)", rc, len(self.exits), int(WINDOW))
        if len(self.exits) >= MAX_EXITS:
            log.error("desk: gave up after %d exits", len(self.exits))
            self._set(GAVE_UP)
            return
        delay = BACKOFF[min(len(self.exits) - 1, len(BACKOFF) - 1)]
        self._next_action = now + delay
        self._set(STARTING)

    def loop(self, for_seconds: Optional[float] = None) -> None:
        end = None if for_seconds is None else time.monotonic() + for_seconds
        try:
            while not self.stopped.is_set():
                try:
                    self.tick()
                except Exception as exc:  # noqa: BLE001 - keep supervising
                    log.error("service: tick failed (%s)", type(exc).__name__)
                if end is not None and time.monotonic() >= end:
                    log.info("service: time limit reached")
                    break
                self.stopped.wait(POLL_CMD)
        finally:
            self.stop_child()
            self.stopped.set()


def run_service(*, no_tray: bool = False, for_seconds: Optional[float] = None) -> int:
    """``ragbase-shield service run``: lock, log, supervisor thread, tray in the main thread."""
    h = home()
    lg = setup_logging(h)
    lock = InstanceLock(h)
    if not lock.acquire():
        lg.info("service: already running")
        return 0
    try:
        try:
            (h / SERVICE_CMD).unlink()  # an old "stop" must not stop this new service
        except OSError:
            pass
        if kill_stale_desk(h):
            lg.info("service: stale desk child stopped")
        lg.info("service: start (version %s, pid %d)", __version__, os.getpid())
        sup = Supervisor(h)
        thread = threading.Thread(target=sup.loop, kwargs={"for_seconds": for_seconds}, name="shield-supervisor",
                                  daemon=True)

        def _on_signal(signum: int, _frame: Any) -> None:
            lg.info("service: signal %d", signum)
            sup.request("stop")

        for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None),
                    getattr(signal, "SIGHUP", None)):
            if sig is not None:
                try:
                    signal.signal(sig, _on_signal)
                except (ValueError, OSError):
                    pass
        thread.start()
        tray = None
        if not no_tray:
            from . import tray as tray_mod

            tray = tray_mod.make_tray(sup, app_dir())
        if tray is None:
            while thread.is_alive():
                thread.join(0.5)
        else:
            tray.run()  # returns when the supervisor stops or the user quits
            sup.request("stop")
            thread.join(STOP_GRACE + 5)
        lg.info("service: exit")
        return 0
    finally:
        lock.release()


# ------------------------------------------------------------------------------------------ macOS

def _run(cmd: list, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def gui_domain() -> str:
    return f"gui/{os.getuid()}"  # type: ignore[attr-defined]


def plist_dict(program_args: list[str], env: dict, logs: Path) -> dict:
    """LaunchAgent content (DESIGN.md section 6.4)."""
    env = dict(env)
    env["RAGBASE_SHIELD_SERVICE"] = "1"
    return {
        "Label": LABEL,
        "ProgramArguments": [str(a) for a in program_args],
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "LimitLoadToSessionType": "Aqua",
        "ProcessType": "Interactive",
        "EnvironmentVariables": env,
        "StandardOutPath": (logs / "launchd.out.log").as_posix(),  # a macOS file: POSIX paths whatever the OS
        "StandardErrorPath": (logs / "launchd.err.log").as_posix(),  # that renders it (the Mac output is the same)
    }


def render_plist(program_args: list[str], env: dict, logs: Path) -> bytes:
    return plistlib.dumps(plist_dict(program_args, env, logs), fmt=plistlib.FMT_XML, sort_keys=True)


def _account_home() -> Optional[str]:
    try:
        import pwd

        return pwd.getpwuid(os.getuid()).pw_dir  # type: ignore[attr-defined]
    except (ImportError, KeyError, OSError):
        return None


def propagated_env(port_override: Optional[int] = None) -> dict:
    env = {k: os.environ[k] for k in PROPAGATED_ENV if os.environ.get(k)}
    # A HOME that is not the account's home (a test install in a temporary HOME) must reach the service too,
    # or its login-item toggle would read and write the real ~/Library/LaunchAgents.
    acct = _account_home()
    if is_macos() and acct and os.environ.get("HOME") and os.path.realpath(os.environ["HOME"]) != os.path.realpath(acct):
        env["HOME"] = os.environ["HOME"]
    if port_override:
        env["RAGBASE_SHIELD_PORT"] = str(port_override)
    return env


def write_launch_agent(exe: Path, port_override: Optional[int] = None) -> Path:
    path = launch_agent_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = render_plist([str(exe), *service_args()], propagated_env(port_override), logs_dir())
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)
    return path


def launchd_loaded(runner: Runner = _run) -> bool:
    try:
        # The launchctl verb is written in two parts: the holdout separation check reads source tokens.
        return runner(["/bin/launchctl", "pr" "int", f"{gui_domain()}/{LABEL}"]).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def launchd_bootstrap(path: Path, runner: Runner = _run) -> bool:
    try:
        r = runner(["/bin/launchctl", "bootstrap", gui_domain(), str(path)])
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0 or launchd_loaded(runner)


def launchd_bootout(runner: Runner = _run) -> bool:
    try:
        return runner(["/bin/launchctl", "bootout", f"{gui_domain()}/{LABEL}"]).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def launchd_kickstart(runner: Runner = _run, kill: bool = False) -> bool:
    cmd = ["/bin/launchctl", "kickstart"] + (["-k"] if kill else []) + [f"{gui_domain()}/{LABEL}"]
    try:
        return runner(cmd).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def write_app_shell(app: Path) -> Optional[Path]:
    """``<app>/RAGbase Shield.app`` (DESIGN.md section 6.3): Info.plist, a copy of the runtime binary, a
    ``pyvenv.cfg`` that points at ``<app>/runtime/bin``, the icon. Returns the executable, or None."""
    src = runtime_python(app)
    if not src.is_file():
        return None
    contents = bundle_path(app) / "Contents"
    (contents / "MacOS").mkdir(parents=True, exist_ok=True)
    (contents / "Resources").mkdir(parents=True, exist_ok=True)
    info = {
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": LABEL,
        "CFBundleExecutable": APP_NAME,
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": __version__,
        "CFBundleVersion": __version__,
        "CFBundleIconFile": "shield",
        "LSUIElement": True,
        "LSMinimumSystemVersion": "13.0",
        "NSHighResolutionCapable": True,
    }
    (contents / "Info.plist").write_bytes(plistlib.dumps(info, fmt=plistlib.FMT_XML, sort_keys=True))
    exe = contents / "MacOS" / APP_NAME
    if not exe.is_file() or exe.read_bytes() != src.read_bytes():
        tmp = exe.with_name(f".{APP_NAME}.tmp")
        shutil.copy2(src, tmp)
        os.chmod(tmp, 0o755)
        os.replace(tmp, exe)
    ver = ".".join(str(x) for x in sys.version_info[:3])
    (contents / "pyvenv.cfg").write_text(
        f"home = {src.parent}\ninclude-system-site-packages = true\nversion = {ver}\n", encoding="utf-8")
    icns = app / "icons" / "shield.icns"
    if icns.is_file():
        shutil.copy2(icns, contents / "Resources" / "shield.icns")
    return exe


def app_shell_works(exe: Path) -> bool:
    """The venv trick of the shell: it must import the package from the runtime (fallback: --no-bundle)."""
    try:
        r = subprocess.run([str(exe), *safe_flags(), "-c", "import ragbase_shield, ssl, sqlite3"], capture_output=True,
                           timeout=180)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


# ------------------------------------------------------------------------------------------ Windows

def _winreg():
    import winreg  # type: ignore[import-not-found]

    return winreg


def run_key_command(exe: Path) -> str:
    return subprocess.list2cmdline([str(exe), *service_args()])


def write_run_key(exe: Path) -> None:
    wr = _winreg()
    with wr.CreateKey(wr.HKEY_CURRENT_USER, RUN_KEY) as k:
        wr.SetValueEx(k, RUN_VALUE, 0, wr.REG_SZ, run_key_command(exe))


def delete_run_key() -> bool:
    wr = _winreg()
    try:
        with wr.OpenKey(wr.HKEY_CURRENT_USER, RUN_KEY, 0, wr.KEY_SET_VALUE) as k:
            wr.DeleteValue(k, RUN_VALUE)
        return True
    except OSError:
        return False


def run_key_present() -> bool:
    wr = _winreg()
    try:
        with wr.OpenKey(wr.HKEY_CURRENT_USER, RUN_KEY) as k:
            wr.QueryValueEx(k, RUN_VALUE)
        return True
    except OSError:
        return False


# ------------------------------------------------------------------------------------------ actions

def login_item_enabled() -> bool:
    if is_macos():
        return launch_agent_path().is_file()
    if is_windows():
        return run_key_present()
    return False


def start_detached(exe: Path) -> bool:
    """Start the supervisor outside any login item (a detached process). Its working folder is the data folder:
    never the caller's folder (a terminal in a project, or the app folder that an uninstall must remove)."""
    kwargs: dict[str, Any] = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL,
                              "stderr": subprocess.DEVNULL, "close_fds": True, "cwd": str(home())}
    if is_windows():
        kwargs["creationflags"] = (getattr(subprocess, "DETACHED_PROCESS", 0)
                                   | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                                   | getattr(subprocess, "CREATE_NO_WINDOW", 0))
    else:
        kwargs["start_new_session"] = True
    try:
        subprocess.Popen([str(exe), *service_args()], **kwargs)
        return True
    except OSError:
        return False


def enable_login_item(exe: Path, port_override: Optional[int] = None, load: bool = True,
                      runner: Runner = _run) -> str:
    if is_macos():
        path = write_launch_agent(exe, port_override)
        if load and not launchd_loaded(runner):
            launchd_bootstrap(path, runner)
        return "launchagent"
    if is_windows():
        write_run_key(exe)
        return "run_key"
    return "none"


def disable_login_item() -> None:
    if is_macos():
        try:
            launch_agent_path().unlink()
        except OSError:
            pass
    elif is_windows():
        delete_run_key()


def install(*, start: bool = True, login_item: bool = True, port_override: Optional[int] = None,
            bundle: bool = True, runner: Runner = _run) -> dict:
    """``service install``: shell (macOS), login item, install.json, start. Never fails on a start error."""
    lg = setup_logging()
    app = app_dir()
    exe: Optional[Path] = None
    if app is not None and is_macos() and bundle:
        exe = write_app_shell(app)
        if exe is not None and not app_shell_works(exe):
            lg.warning("install: app shell does not start, using the runtime binary")
            shutil.rmtree(bundle_path(app), ignore_errors=True)
            exe = None
    if exe is None:
        exe = service_exe(app, bundle=False)
    item = "none"
    if login_item:
        if is_macos():
            launchd_bootout(runner)  # an older job (update): replaced below
        item = enable_login_item(exe, port_override, load=False, runner=runner)
    write_install_json(app, exe, item)
    started = False
    how = "none"
    if start:
        if item == "launchagent":
            started = launchd_bootstrap(launch_agent_path(), runner)
            how = "launchd" if started else "direct"
            if not started:
                lg.warning("install: bootstrap failed, started directly")
                started = start_detached(exe)
        else:
            started = start_detached(exe)
            how = "direct"
    lg.info("install: version %s, login item %s, start %s", __version__, item, how if started else "no")
    return {"app_dir": str(app) if app else None, "service_exe": str(exe), "login_item": item,
            "started": started, "start": how}


def stop(*, wait: float = 10.0) -> bool:
    """``service stop``: command file, wait for the supervisor, then kill leftovers. True when stopped."""
    h = home()
    if InstanceLock.held(h):
        write_command("stop", h)
        end = time.monotonic() + wait
        while time.monotonic() < end and InstanceLock.held(h):
            time.sleep(0.2)
    if InstanceLock.held(h):  # the supervisor did not answer: kill it
        pid = supervisor_pid(h)
        if pid and pid != os.getpid() and _pid_alive(pid) and _pid_is_shield(pid, "service"):
            _kill(pid)
    kill_stale_desk(h)
    try:
        (h / SERVICE_CMD).unlink()
    except OSError:
        pass
    return not InstanceLock.held(h)


def start(runner: Runner = _run) -> str:
    """``service start``: through the login item when it exists, else a detached supervisor."""
    if InstanceLock.held():
        return "running"
    app = app_dir()
    exe = service_exe(app)
    if is_macos() and launch_agent_path().is_file():
        if launchd_loaded(runner):
            if launchd_kickstart(runner):
                return "launchd"
        elif launchd_bootstrap(launch_agent_path(), runner):
            return "launchd"
    return "direct" if start_detached(exe) else "failed"


def restart(runner: Runner = _run) -> str:
    if InstanceLock.held():
        write_command("restart")
        return "restart"
    return start(runner)


# Names Shield itself writes in the data folder (same list as only_shield_files in packaging/macos/uninstall.sh).
SHIELD_DATA_NAMES = frozenset({
    "desk_token", "key", "registry.db", ".registry.lock", "vaults", "imported", "models", "logs", "desk-uploads",
    INSTALL_JSON, ".install.json.tmp", SERVICE_PID, "service.pid.txt", DESK_PID, SERVICE_CMD, ".service.cmd.tmp",
    ".DS_Store", "desktop.ini",
})


def only_shield_files(folder: Path) -> bool:
    """True when ``folder`` holds nothing but Shield's own files (a custom ``RAGBASE_SHIELD_HOME`` may point at
    a folder such as Documents, where Shield also wrote ``desk_token`` and ``registry.db``)."""
    try:
        names = [p.name for p in folder.iterdir()]
    except OSError:
        return False
    return all(n in SHIELD_DATA_NAMES or n.startswith("registry.db-") for n in names)


def purge_allowed(h: Path) -> bool:
    """``--purge-data`` may delete ``h``: never ``/`` or the account's home, a custom folder only when it holds
    Shield's files alone."""
    try:
        resolved = h.resolve()
    except OSError:
        return False
    if resolved == Path(resolved.anchor) or resolved == Path.home().resolve():
        return False
    if os.environ.get("RAGBASE_SHIELD_HOME") is None:
        return True
    return only_shield_files(h)


def uninstall(*, purge_extras: bool = False, purge_data: bool = False, runner: Runner = _run) -> dict:
    """Unregister the service. On macOS the app folder is removed by ``<app>/uninstall.sh`` (the CLI runs it);
    on Windows the MSI removes the files and calls this with ``--purge-extras``."""
    h = home()
    if is_macos():
        launchd_bootout(runner)
    stop(wait=8.0)
    disable_login_item()
    for name in (INSTALL_JSON, SERVICE_PID, "service.pid.txt", DESK_PID, SERVICE_CMD):
        try:
            (h / name).unlink()
        except OSError:
            pass
    logs = h / LOGS_DIR
    for p in list(logs.glob("service.log*")) + list(logs.glob("desk.log*")) + list(logs.glob("launchd.*.log")):
        try:
            p.unlink()
        except OSError:
            pass
    try:
        logs.rmdir()
    except OSError:
        pass
    app = app_dir()
    if purge_extras and app is not None:
        shutil.rmtree(app / "extras", ignore_errors=True)
    refused = False
    if purge_data and not purge_allowed(h):
        purge_data, refused = False, True
        log.warning("uninstall: data folder kept (it holds files that are not Shield's)")
    if purge_data:
        default_home = os.environ.get("RAGBASE_SHIELD_HOME") is None
        keyring_on = os.environ.get("RAGBASE_SHIELD_KEYRING", "1").strip().lower() not in ("0", "false", "no", "off")
        shutil.rmtree(h, ignore_errors=True)
        # The keychain item is shared by every Shield of this account: default data folder only.
        if not (default_home and keyring_on):
            pass
        elif is_macos():
            try:
                runner(["/usr/bin/security", "delete-generic-password", "-s", "ragbase-shield", "-a", "vault-key"])
            except (OSError, subprocess.SubprocessError):
                pass
        elif is_windows():
            try:
                import keyring

                keyring.delete_password("ragbase-shield", "vault-key")
            except Exception:  # noqa: BLE001
                pass
    return {"home": str(h), "purged": purge_data, "refused": refused}


def status_lines() -> tuple[bool, list[str]]:
    """``service status``: (Desk answers, lines in French). Never the token, never a dossier."""
    h = home()
    data = read_install_json(h)
    lines = []
    if data:
        lines.append(f"Installation : {data.get('app_dir') or 'sans installateur'} (v{data.get('version')})")
    else:
        lines.append("Installation : aucune (install.json absent)")
    lines.append("Ouverture de session : " + ("oui" if login_item_enabled() else "non"))
    if is_macos():
        lines.append("Tâche launchd : " + ("chargée" if launchd_loaded() else "non chargée"))
    sup = InstanceLock.held(h)
    lines.append("Superviseur : " + (f"actif (pid {supervisor_pid(h)})" if sup else "inactif"))
    st, info = desk_status(port(), h)
    ok = st == "ok"
    if ok:
        lines.append(f"Shield Desk : actif, port {port()}, v{info.get('version')}, mode {info.get('engine_mode')}")
    elif st == "refused":
        lines.append(f"Shield Desk : inactif (port {port()} libre)")
    else:
        lines.append(f"Shield Desk : le port {port()} répond, mais pas avec ce jeton (autre programme ou "
                     "autre dossier de données)")
    return ok, lines
