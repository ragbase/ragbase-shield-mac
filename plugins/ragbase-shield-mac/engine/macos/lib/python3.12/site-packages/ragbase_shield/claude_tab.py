"""RAGbase Shield for the Claude tab of the Claude desktop app (plugin mode).

The plugin carries its own Python (``engine/`` in the plugin folder) and starts this module as its local MCP server:

    ${CLAUDE_PLUGIN_ROOT}/engine/shield -I -B -X utf8 -m ragbase_shield.claude_tab

What differs from ``ragbase-shield mcp``:

- **Locked settings.** The process ignores every environment variable that could move or weaken Shield (data folder,
  key store, detection switches, dossier scope): they are removed before anything else runs. The data folder and the
  key store are fixed per computer. The only way to point Shield at a scratch folder is the test switch
  ``--test-data <folder>`` on the command line, which the plugin's launch command never passes and which Claude
  cannot add (a tool call or a settings file can change the environment, not the arguments of the plugin command).
- **Key store with no fallback.** macOS: the login keychain through Apple's own ``/usr/bin/security`` tool, so the
  item's access list names that stable, Apple-signed program and no keychain window appears after a plugin update.
  Windows: the Credential Manager. A key store failure is an error, never a key file next to the vault.
- **Seven tools, French only.** The six tools of ``ragbase-shield mcp`` plus ``shield_protect_folder``, which takes
  NO path: Shield opens the system's own "choose a folder" window on the lawyer's computer, the lawyer picks the
  client folder there, and the answer holds the dossier code and counts only. A folder name often holds the parties'
  names ("Dossier Martin c. Bernard"): it never goes through Claude.
- **Plain French** in every text that Claude or the lawyer can see.
"""
# No "from __future__ import annotations": the tool signatures below use local Annotated types.

import logging
import os
import re
import subprocess
import sys
import threading
from collections import Counter
from pathlib import Path
from typing import Annotated, Callable, Literal, Optional


def _system_windows_dir() -> str:  # pragma: no cover - Windows
    """The Windows folder (C:\\Windows) from the system, never from an environment variable."""
    try:
        import ctypes

        buf = ctypes.create_unicode_buffer(260)
        if ctypes.windll.kernel32.GetSystemWindowsDirectoryW(buf, 260):
            return buf.value
    except Exception:  # noqa: BLE001
        pass
    return ""


def fill_systemroot(environ=None, windows: Optional[bool] = None,
                    windows_dir: Optional[Callable[[], str]] = None) -> bool:
    """Windows: put SYSTEMROOT back when the app starts Shield with a small environment. Without it Winsock, so
    ``import asyncio``, stops with WinError 10106 before the server answers. Runs before anything imports asyncio.
    Returns True when the variable was set."""
    environ = os.environ if environ is None else environ
    windows = (os.name == "nt") if windows is None else windows
    if not windows or environ.get("SYSTEMROOT"):
        return False
    value = (windows_dir or _system_windows_dir)()
    if not value:
        return False
    environ["SYSTEMROOT"] = value
    return True


fill_systemroot()  # at import time, before the MCP framework (asyncio) is imported

APP_DIR_NAME = "RAGbase Shield"
KEY_SERVICE = "ragbase-shield-plugin"
KEY_ACCOUNT = "vault-key"
#: Biggest folder Shield protects in one go: a client dossier, not a whole archive.
MAX_FOLDER_FILES = 2000
MAX_FOLDER_BYTES = 2 * 1024 ** 3
#: How long the "choose a folder" window waits for the lawyer (seconds).
PICK_TIMEOUT = 290.0
PICK_TITLE = "RAGbase Shield : choisissez le dossier client à protéger"
PICK_BUTTON = "Protéger ce dossier"

log = logging.getLogger(__name__)

#: Variables removed at start (case-insensitive prefixes): Shield's own switches and the model caches.
_SCRUB_PREFIXES = ("RAGBASE_SHIELD_", "HF_", "TRANSFORMERS_", "TOKENIZERS_", "HUGGINGFACE_")


# ================================================================================ fixed places (no environment)

def _posix_home() -> Path:
    import pwd

    return Path(pwd.getpwuid(os.getuid()).pw_dir)


def _known_folder(guid: str) -> Optional[Path]:  # pragma: no cover - Windows
    """A Windows known folder from the shell (never from an environment variable)."""
    try:
        import ctypes
        from ctypes import wintypes

        class GUID(ctypes.Structure):
            _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD),
                        ("Data4", ctypes.c_ubyte * 8)]

        g = GUID()
        ctypes.oledll.ole32.CLSIDFromString(ctypes.c_wchar_p(guid), ctypes.byref(g))
        out = ctypes.c_wchar_p()
        ctypes.oledll.shell32.SHGetKnownFolderPath(ctypes.byref(g), 0, None, ctypes.byref(out))
        try:
            return Path(out.value) if out.value else None
        finally:
            ctypes.windll.ole32.CoTaskMemFree(out)
    except Exception:  # noqa: BLE001
        return None


_FOLDERID = {
    "LocalAppData": "{F1B32785-6FBA-4FCF-9D55-7B8E7F157091}",
    "RoamingAppData": "{3EB685DB-65F9-4CF6-A03A-E3EF65729F3D}",
    "Profile": "{5E6C858F-0E22-4760-9AFE-EA3317B67173}",
    "Desktop": "{B4BFCC3A-DB2C-424C-B029-7FE99A87C641}",
    "Documents": "{FDD39AD0-238F-46AF-ADB4-6C85480369C7}",
    "Downloads": "{374DE290-123F-4565-9164-39C4925E467B}",
    "Windows": "{F38BF404-1D43-42F2-9305-67DE0B28FC23}",
    "ProgramFiles": "{905E63B6-C1BF-494E-B29C-65B732D3D21A}",
    "ProgramFilesX86": "{7C5A40EF-A0FB-4BFC-874A-C0F2E0B9FA8E}",
    "ProgramData": "{62AB5D82-FDC1-4DC3-A9DD-070D1D495D97}",
}


def user_home() -> Path:
    """The user's real home folder, from the system account data (not $HOME or %USERPROFILE%)."""
    if sys.platform == "win32":  # pragma: no cover - Windows
        return _known_folder(_FOLDERID["Profile"]) or Path.home()
    return _posix_home()


def plugin_home() -> Path:
    """Fixed data folder of the plugin: the encrypted vaults and the registry, never the key.

    macOS: ~/Library/Application Support/RAGbase Shield. Windows: %LOCALAPPDATA%\\RAGbase Shield (from the shell,
    not from the variable). Other systems: ~/.local/share/ragbase-shield-plugin. It is NOT the data folder of the
    Shield installers and command line (~/.ragbase-shield), whose key lives in another keychain item."""
    if sys.platform == "win32":  # pragma: no cover - Windows
        base = _known_folder(_FOLDERID["LocalAppData"]) or (user_home() / "AppData" / "Local")
        return base / APP_DIR_NAME
    if sys.platform == "darwin":
        return _posix_home() / "Library" / "Application Support" / APP_DIR_NAME
    return _posix_home() / ".local" / "share" / "ragbase-shield-plugin"


def plugin_root() -> Optional[Path]:
    """The plugin folder (the parent of ``engine/``), found from the running Python, not from the environment."""
    exe = Path(sys.executable).resolve()
    for parent in exe.parents:
        if parent.name == "engine":
            return parent.parent
    return None


# ================================================================================ key stores

class MacKeychainStore:
    """The master key in the macOS login keychain, written and read by Apple's /usr/bin/security tool.

    The item is created by that tool, so its access list names /usr/bin/security (stable path, Apple signature):
    reading it later never shows a keychain window, even after the plugin folder changes with an update. The key
    goes to the tool on its standard input, never on a command line. ``keychain``: a scratch keychain file (tests)."""

    source = "keychain"
    TOOL = "/usr/bin/security"

    def __init__(self, keychain: Optional[str] = None, service: str = KEY_SERVICE, account: str = KEY_ACCOUNT):
        self.keychain = keychain
        self.service = service
        self.account = account

    def _tail(self) -> list[str]:
        return [self.keychain] if self.keychain else []

    def get(self) -> Optional[bytes]:
        r = subprocess.run([self.TOOL, "find-generic-password", "-s", self.service, "-a", self.account, "-w",
                            *self._tail()], capture_output=True, timeout=30, env={"PATH": "/usr/bin:/bin"})
        if r.returncode == 44:  # errSecItemNotFound
            return None
        if r.returncode != 0:
            raise OSError(f"security find-generic-password rc={r.returncode}")
        value = r.stdout.strip()
        return value or None

    def put(self, key: bytes) -> None:
        if not all(c.isalnum() or c in "-_=" for c in key.decode("ascii")):
            raise ValueError("unexpected key characters")
        line = (f"add-generic-password -s {self.service} -a {self.account} -l RAGbase-Shield "
                f"-w {key.decode('ascii')}")
        if self.keychain:
            line += f" {self.keychain}"
        r = subprocess.run([self.TOOL, "-i"], input=(line + "\n").encode("ascii"), capture_output=True,
                           timeout=30, env={"PATH": "/usr/bin:/bin"})
        if r.returncode != 0:
            raise OSError(f"security add-generic-password rc={r.returncode}")


class WindowsCredentialStore:  # pragma: no cover - Windows
    """The master key in the Windows Credential Manager (keyring's WinVault backend, no backend discovery)."""

    source = "credential-manager"

    def __init__(self, service: str = KEY_SERVICE, account: str = KEY_ACCOUNT):
        from keyring.backends.Windows import WinVaultKeyring

        self._kr = WinVaultKeyring()
        self.service = service
        self.account = account

    def get(self) -> Optional[bytes]:
        raw = self._kr.get_password(self.service, self.account)
        return raw.encode("ascii") if raw else None

    def put(self, key: bytes) -> None:
        self._kr.set_password(self.service, self.account, key.decode("ascii"))


class FileKeyStore:
    """Test only: the key in a file of the scratch data folder given with --test-data."""

    source = "test-file"

    def __init__(self, path: Path):
        self.path = Path(path)

    def get(self) -> Optional[bytes]:
        return self.path.read_bytes().strip() if self.path.is_file() else None

    def put(self, key: bytes) -> None:
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(key)


def system_key_store():
    if sys.platform == "darwin":
        return MacKeychainStore()
    if sys.platform == "win32":  # pragma: no cover - Windows
        return WindowsCredentialStore()
    raise OSError("no key store for this system")


# ================================================================================ locked settings

def scrub_environment() -> list[str]:
    """Remove every variable that could move or weaken Shield. Returns the names removed (never their values)."""
    removed = []
    for name in list(os.environ):
        if name.upper().startswith(_SCRUB_PREFIXES):
            removed.append(name)
            os.environ.pop(name, None)
    return removed


def lock_settings(test_data: Optional[Path] = None, key_store=None) -> Path:
    """Fix the data folder, the key store and the detection settings for this process. Returns the data folder.

    ``test_data``: the scratch folder of a test (the data AND a file key go there). Never set by the plugin."""
    from . import service, vault

    scrub_environment()
    if test_data is not None:
        home = Path(test_data).resolve()
        store = key_store or FileKeyStore(home / "test-key")
    else:
        home = plugin_home()
        store = key_store or system_key_store()
        if sys.platform != "win32":
            os.environ["HOME"] = str(_posix_home())
        else:  # pragma: no cover - Windows
            for var, fid in (("USERPROFILE", "Profile"), ("LOCALAPPDATA", "LocalAppData"),
                             ("APPDATA", "RoamingAppData")):
                p = _known_folder(_FOLDERID[fid])
                if p is not None:
                    os.environ[var] = str(p)
    home.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(home, 0o700)
    except OSError:
        pass
    os.environ["RAGBASE_SHIELD_HOME"] = str(home)
    os.environ["RAGBASE_SHIELD_KEYRING"] = "0"  # the keychain library is never used: KEY_STORE below is
    os.environ["RAGBASE_SHIELD_NER"] = "off"  # no model in the plugin: the detection rules only
    os.environ["RAGBASE_SHIELD_LOCALE"] = "fr"
    vault.KEY_STORE = store
    if sys.platform == "darwin":
        try:  # the per-user temporary folder, not the shared /tmp
            import tempfile

            tmp = os.confstr("CS_DARWIN_USER_TEMP_DIR")
            if tmp and Path(tmp).is_dir():
                tempfile.tempdir = tmp
        except (ValueError, OSError):
            pass
    service.reset()
    return home


# ================================================================================ folder guards

class FolderRefused(Exception):
    """The chosen folder cannot be protected. ``kind`` picks the French message."""

    def __init__(self, kind: str):
        super().__init__(kind)
        self.kind = kind


REFUSALS = {
    "too_wide": "Ce dossier est trop large : il contient tout l'ordinateur, un disque entier ou tout l'espace "
                "personnel de l'utilisateur. Rien n'a été protégé. Choisissez le dossier d'un seul client.",
    "user_folder": "Le Bureau, les Documents, les Téléchargements ou un autre grand dossier personnel ne peuvent pas "
                   "être protégés en entier. Rien n'a été protégé. Choisissez le dossier d'un seul client, à "
                   "l'intérieur.",
    "system": "Ce dossier appartient au système de l'ordinateur. Rien n'a été protégé. Choisissez le dossier "
              "d'un client.",
    "app_data": "Ce dossier contient des données de Claude ou de RAGbase Shield. Rien n'a été protégé. Choisissez "
                "le dossier d'un client.",
    "too_big": "Ce dossier contient trop de fichiers (plus de 2 000) ou plus de 2 Go : ce n'est pas le dossier d'un "
               "seul client. Rien n'a été protégé. Choisissez un dossier plus petit.",
    "missing": "Le dossier choisi n'existe plus ou ne peut pas être ouvert. Rien n'a été protégé.",
    "holds_dossier": "Ce dossier contient déjà un dossier client protégé. Rien n'a été protégé. Choisissez le "
                     "dossier d'un seul client.",
}


def _inside(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
        return True
    except ValueError:
        return False


def _norm(p: Path) -> Path:
    try:
        p = p.resolve()
    except (OSError, RuntimeError):
        pass
    if sys.platform in ("win32", "darwin"):  # case-insensitive file systems by default
        return Path(os.path.normcase(str(p)).lower()) if sys.platform == "win32" else Path(str(p).lower())
    return p


def _protected_places(home: Path) -> dict[str, list[Path]]:
    """Folders that are refused (and every folder that holds one of them). Paths only ADD refusals: an environment
    variable can never remove one."""
    shield_data = [plugin_home(), home / ".ragbase-shield", Path(os.environ.get("RAGBASE_SHIELD_HOME") or plugin_home())]
    root = plugin_root()
    claude = [home / ".claude", home / ".claude.json"]
    user: list[Path] = [home / n for n in ("Desktop", "Documents", "Downloads", "Pictures", "Movies", "Music",
                                           "Public", "Videos", "OneDrive", "Dropbox", "Google Drive", "iCloudDrive")]
    system: list[Path] = []
    inside_refused: list[Path] = []  # refused with everything inside them
    if sys.platform == "darwin":
        lib = home / "Library"
        claude += [lib / "Application Support" / "Claude", lib / "Application Support" / "Claude-3p",
                   lib / "Logs" / "Claude", lib / "Containers" / "com.anthropic.claudefordesktop"]
        icloud = lib / "Mobile Documents" / "com~apple~CloudDocs"
        user += [icloud, icloud / "Desktop", icloud / "Documents", icloud / "Downloads", lib / "Mobile Documents",
                 lib / "CloudStorage", lib]
        try:
            for p in (lib / "CloudStorage").iterdir():  # OneDrive, Google Drive, Dropbox roots and their big folders
                user += [p, p / "Desktop", p / "Documents", p / "Bureau", p / "Documents partagés"]
        except OSError:
            pass
        inside_refused += [lib / n for n in ("Application Support", "Containers", "Group Containers", "Preferences",
                                              "Caches", "Logs", "Keychains", "Mail", "Messages", "Safari")]
        system += [Path(p) for p in ("/System", "/Library", "/Applications", "/usr", "/bin", "/sbin", "/etc",
                                     "/private/etc", "/opt", "/cores", "/dev", "/Volumes", "/Users", "/private",
                                     "/var", "/private/var", "/tmp", "/private/tmp", "/Network")]
        inside_refused += [Path(p) for p in ("/System", "/Library", "/Applications", "/usr", "/bin", "/sbin", "/etc",
                                             "/private/etc", "/opt", "/cores", "/dev", "/Network")]
    elif sys.platform == "win32":  # pragma: no cover - Windows
        for fid in ("Desktop", "Documents", "Downloads"):
            p = _known_folder(_FOLDERID[fid])
            if p is not None:
                user.append(p)
        for var in ("OneDrive", "OneDriveCommercial", "OneDriveConsumer"):
            if os.environ.get(var):
                p = Path(os.environ[var])
                user += [p, p / "Desktop", p / "Documents", p / "Bureau"]
        appdata = home / "AppData"
        local = _known_folder(_FOLDERID["LocalAppData"]) or appdata / "Local"
        roaming = _known_folder(_FOLDERID["RoamingAppData"]) or appdata / "Roaming"
        claude += [roaming / "Claude", local / "Claude-Data", local / "AnthropicClaude", local / "Claude"]
        shield_data.append(local / "ragbase-shield")
        inside_refused += [appdata, local, roaming]
        for fid in ("Windows", "ProgramFiles", "ProgramFilesX86", "ProgramData"):
            p = _known_folder(_FOLDERID[fid])
            if p is not None:
                inside_refused.append(p)
        inside_refused += [Path("C:/Windows"), Path("C:/Program Files"), Path("C:/Program Files (x86)"),
                           Path("C:/ProgramData")]
        system += [Path("C:/Users")]
    else:
        inside_refused += [Path(p) for p in ("/etc", "/usr", "/bin", "/sbin", "/boot", "/proc", "/sys", "/dev")]
    data = shield_data + claude + ([root] if root is not None else [])
    return {"app_data": data, "user_folder": user, "system": system, "inside_refused": inside_refused}


def check_folder(folder: os.PathLike | str) -> Path:
    """The resolved folder when Shield may protect it; else :class:`FolderRefused`."""
    try:
        path = Path(folder).resolve(strict=True)
    except (OSError, RuntimeError):
        raise FolderRefused("missing") from None
    if not path.is_dir():
        raise FolderRefused("missing")
    p = _norm(path)
    home = user_home()
    h = _norm(home)
    if p == _norm(Path(path.anchor)) or p.parent == p:
        raise FolderRefused("too_wide")
    if sys.platform == "darwin" and len(p.parts) <= 3 and p.parts[1:2] == ("volumes",):
        raise FolderRefused("too_wide")  # an external disk: /Volumes/<disk>
    if sys.platform == "win32" and str(path).startswith("\\\\") and len(path.parts) <= 1:  # pragma: no cover
        raise FolderRefused("too_wide")  # a network share root
    if p == h or _inside(h, p):
        raise FolderRefused("too_wide")
    places = _protected_places(home)
    for q in places["app_data"]:
        n = _norm(q)
        if p == n or _inside(p, n) or _inside(n, p):
            raise FolderRefused("app_data")
    for q in places["inside_refused"]:
        n = _norm(q)
        if p == n or _inside(p, n):
            raise FolderRefused("system")
    for q in places["system"]:
        if p == _norm(q):
            raise FolderRefused("system")
    for q in places["user_folder"]:
        if p == _norm(q):
            raise FolderRefused("user_folder")
    _check_size(path)
    return path


def _check_size(path: Path) -> None:
    from . import config

    files = 0
    size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != config.OUTPUT_DIR_NAME]
        for f in filenames:
            if f.startswith("."):
                continue
            files += 1
            try:
                size += os.stat(os.path.join(dirpath, f)).st_size
            except OSError:
                pass
            if files > MAX_FOLDER_FILES or size > MAX_FOLDER_BYTES:
                raise FolderRefused("too_big")


# ================================================================================ the "choose a folder" window

class PickCancelled(Exception):
    pass


class PickFailed(Exception):
    def __init__(self, kind: str = "failed"):
        super().__init__(kind)
        self.kind = kind


def _pick_mac(title: str, timeout: float) -> str:
    """The macOS "choose a folder" window (AppleScript ``choose folder``, run by Apple's osascript). It talks to no
    other application, so macOS asks for no automation permission. The title goes as an argument, not as code."""
    script = ["on run argv", "activate",
              "set f to choose folder with prompt (item 1 of argv) default location (path to home folder)",
              "return POSIX path of f", "end run"]
    cmd = ["/usr/bin/osascript"]
    for line in script:
        cmd += ["-e", line]
    cmd.append(title)
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout, env={"PATH": "/usr/bin:/bin"})
    except subprocess.TimeoutExpired:
        raise PickFailed("timeout") from None
    except OSError:
        raise PickFailed("failed") from None
    if r.returncode != 0:
        if b"-128" in r.stderr:  # "User canceled. (-128)"
            raise PickCancelled()
        raise PickFailed("failed")
    out = r.stdout.decode("utf-8", "replace").strip()
    if not out:
        raise PickCancelled()
    return out


def _pick_windows_dialog(title: str, button: str) -> Optional[str]:  # pragma: no cover - Windows
    """The Windows "choose a folder" window (IFileOpenDialog with FOS_PICKFOLDERS), in-process through ctypes: no
    console, no PowerShell. Returns the path, or None when the user closes the window."""
    import ctypes
    from ctypes import POINTER, byref, c_long, c_uint, c_void_p, c_wchar_p, wintypes

    ole32 = ctypes.windll.ole32

    class GUID(ctypes.Structure):
        _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD),
                    ("Data4", ctypes.c_ubyte * 8)]

    def guid(text: str) -> GUID:
        g = GUID()
        if ole32.CLSIDFromString(c_wchar_p(text), byref(g)) != 0:
            raise OSError("CLSIDFromString")
        return g

    def call(obj: c_void_p, index: int, *args, argtypes=()) -> int:
        vtbl = ctypes.cast(obj, POINTER(POINTER(c_void_p))).contents
        proto = ctypes.WINFUNCTYPE(c_long, c_void_p, *argtypes)
        return proto(vtbl[index])(obj, *args) & 0xFFFFFFFF

    ERROR_CANCELLED = 0x800704C7
    FOS = 0x20 | 0x40 | 0x800 | 0x8  # PICKFOLDERS | FORCEFILESYSTEM | PATHMUSTEXIST | NOCHANGEDIR
    SIGDN_FILESYSPATH = 0x80058000
    hr = ole32.CoInitializeEx(None, 0x2 | 0x4) & 0xFFFFFFFF  # apartment threaded, no OLE1 DDE
    if hr not in (0, 1):
        raise OSError("CoInitializeEx")
    dialog = c_void_p()
    item = c_void_p()
    try:
        if ole32.CoCreateInstance(byref(guid("{DC1C5A9C-E88A-4DDE-A5A1-60F82A20AEF7}")), None, 1,
                                  byref(guid("{D57C7288-D4AD-4768-BE02-9D969532D960}")), byref(dialog)) != 0:
            raise OSError("CoCreateInstance")
        opts = c_uint()
        if call(dialog, 10, byref(opts), argtypes=(POINTER(c_uint),)) == 0:  # GetOptions
            call(dialog, 9, c_uint(opts.value | FOS), argtypes=(c_uint,))  # SetOptions
        call(dialog, 17, c_wchar_p(title), argtypes=(c_wchar_p,))  # SetTitle
        call(dialog, 18, c_wchar_p(button), argtypes=(c_wchar_p,))  # SetOkButtonLabel
        owner = ctypes.windll.user32.GetForegroundWindow()
        hr = call(dialog, 3, wintypes.HWND(owner), argtypes=(wintypes.HWND,))  # Show
        if hr == ERROR_CANCELLED:
            return None
        if hr != 0:
            raise OSError("Show")
        if call(dialog, 20, byref(item), argtypes=(POINTER(c_void_p),)) != 0:  # GetResult
            raise OSError("GetResult")
        name = c_wchar_p()
        if call(item, 5, c_uint(SIGDN_FILESYSPATH), byref(name), argtypes=(c_uint, POINTER(c_wchar_p))) != 0:
            raise OSError("GetDisplayName")
        try:
            return name.value
        finally:
            ole32.CoTaskMemFree(name)
    finally:
        for obj in (item, dialog):
            if obj:
                call(obj, 2)  # Release
        ole32.CoUninitialize()


def _pick_windows(title: str, timeout: float) -> str:  # pragma: no cover - Windows
    box: dict = {}

    def run() -> None:
        try:
            box["path"] = _pick_windows_dialog(title, PICK_BUTTON)
        except Exception as exc:  # noqa: BLE001
            box["error"] = type(exc).__name__

    t = threading.Thread(target=run, name="shield-pick-folder", daemon=True)  # its own COM apartment
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise PickFailed("timeout")
    if "error" in box:
        raise PickFailed("failed")
    if not box.get("path"):
        raise PickCancelled()
    return box["path"]


def _system_picker(title: str, timeout: float) -> str:
    if sys.platform == "darwin":
        return _pick_mac(title, timeout)
    if sys.platform == "win32":  # pragma: no cover - Windows
        return _pick_windows(title, timeout)
    raise PickFailed("unsupported")


#: The window that asks the lawyer for a folder. Tests replace it; the plugin's test switch --test-pick sets it to
#: a fixed answer. No tool argument can reach it.
PICKER: Callable[[str, float], str] = _system_picker


# ================================================================================ safe file names

#: Words a file name may show in clear when they start with a capital letter (besides Shield codes and numbers):
#: everyday file words. Names of people and places are not in these lists; a first name never passes.
_FILE_WORDS = frozenset("""
pdf docx doc txt md eml html scan scanne copie version final finale definitif definitive projet brouillon draft
notes note courrier courriers lettre lettres mail email mails piece pieces annexe annexes bordereau conclusions
assignation jugement arret ordonnance requete memoire contrat contrats avenant facture factures devis attestation
attestations releve releves proces verbal pv sous dossier dossiers rg tj cph ca cass synthese recapitulatif
compte rendu cr reponse demande mise demeure relance relances echange echanges plainte constat expertise rapport
convocation citation decision notification signification acte actes statuts bail quittance quittances bulletin
bulletins paie salaire salaires calcul calculs tableau liste chronologie analyse audit fiche fiches modele
""".split())


def _safe_words() -> tuple[frozenset, frozenset]:
    """(words that may stay in clear, first names). Built from the engine's own word lists."""
    global _SAFE_CACHE
    if _SAFE_CACHE is None:
        from .engine import lexicon as lx
        from .engine import names as nm

        stop, legal, common, _fn, afn, months, wl = nm._lists()
        extra: set[str] = set()
        for getter in ("business_words", "trade_nouns", "instrument_terms"):
            try:
                extra |= set(getattr(lx, getter)())
            except Exception:  # noqa: BLE001 - a missing list only makes the filter stricter
                pass
        try:
            for words in lx.lowercase_words().values():
                extra |= set(words)
        except Exception:  # noqa: BLE001
            pass
        known = frozenset(stop | legal | common | months | wl | extra | _FILE_WORDS)
        _SAFE_CACHE = (known, frozenset(afn), frozenset(stop | months | legal | _FILE_WORDS))
    return _SAFE_CACHE[0], _SAFE_CACHE[1]


_SAFE_CACHE: Optional[tuple] = None
_HIDDEN = "\u2026"  # "…"
_LETTERS_RE = re.compile(r"[^\W\d_]+")
_DOTS_RE = re.compile(_HIDDEN + r"(?:[\s\-'’.,&+]*" + _HIDDEN + r")+")


def _code_re():
    from .engine.masker import UNMASK_RE

    return re.compile(r"(?:" + UNMASK_RE.pattern + r"|\[[A-Z]{2,}\]|\b(?:DOC|DOSSIER|OUTPUT)_[0-9]{1,7}\b)")


def safe_file_name(masked: str) -> str:
    """A masked file name made safe to show: Shield codes, numbers, punctuation and everyday words stay; every
    other word that starts with a capital letter, and every first name, becomes "…". A name that the detection
    did not know ("Notes Quillivic.txt") never reaches Claude."""
    known, first = _safe_words()
    from .engine.normalize import fold_word

    strict = _SAFE_CACHE[2]
    out: list[str] = []
    pos = 0

    def words(seg: str) -> str:
        res: list[str] = []
        q = 0
        for m in _LETTERS_RE.finditer(seg):
            res.append(seg[q:m.start()])
            w = m.group()
            f = fold_word(w)
            if w.islower():
                keep = f in known or f not in first
            else:
                keep = len(w) > 1 and f in known and (f not in first or f in strict)
            res.append(w if keep else _HIDDEN)
            q = m.end()
        res.append(seg[q:])
        return "".join(res)

    for m in _code_re().finditer(masked):
        out.append(words(masked[pos:m.start()]))
        out.append(m.group())
        pos = m.end()
    out.append(words(masked[pos:]))
    return _DOTS_RE.sub(_HIDDEN, "".join(out))


def _norm_inside(path: Path, other: Path) -> bool:
    """True when ``path`` is ``other`` or inside it (case rules of the system)."""
    return _inside(_norm(path), _norm(other))


# ================================================================================ French texts

PICK_MESSAGES = {
    "cancelled": "Aucun dossier n'a été choisi : la fenêtre a été fermée. Rien n'a été protégé. Si l'utilisateur "
                 "veut protéger un dossier, il le redemande et choisit le dossier dans la fenêtre.",
    "timeout": "La fenêtre de choix est restée sans réponse trop longtemps. Rien n'a été protégé. L'utilisateur peut "
               "le redemander : une nouvelle fenêtre s'ouvrira.",
    "failed": "La fenêtre de choix du dossier n'a pas pu s'ouvrir sur cet ordinateur. Rien n'a été protégé. "
              "Redémarrer l'application Claude règle souvent le problème.",
    "unsupported": "Cette fonction marche sur Mac et sur Windows uniquement. Rien n'a été protégé.",
}

ERRORS = {
    "unknown_dossier": "Ce code de dossier n'existe pas. Utilisez shield_list_dossiers pour voir les codes.",
    "unknown_document": "Ce code de document n'existe pas dans ce dossier. Utilisez shield_list_documents pour voir "
                        "les codes.",
    "unsupported_format": "Shield ne sait pas lire ce type de fichier (tableur, image...). Demandez à l'utilisateur "
                          "une copie en PDF, Word ou texte.",
    "document_unreadable": "Ce document ne peut pas être lu (fichier abîmé, vide ou protégé par un mot de passe).",
    "document_too_large": "Ce document est trop long pour Shield. Demandez à l'utilisateur de le couper en "
                          "plusieurs fichiers.",
    "vault_error": "Le coffre de Shield ne peut pas être ouvert sur cet ordinateur. Rien n'a été envoyé. "
                   "Redémarrez l'application Claude ; si cela continue, contactez le support RAGbase.",
    "write_failed": "Le fichier n'a pas pu être enregistré dans le dossier « Shield - sorties ». Vérifiez que le "
                    "dossier client existe toujours et qu'il n'est pas en lecture seule.",
    "invalid_argument": "Cette demande n'est pas valide (code mal écrit, texte trop long ou valeur manquante). "
                        "Rien n'a été envoyé.",
    "internal": "Shield a rencontré un problème inattendu. Rien n'a été envoyé. Réessayez ; si cela continue, "
                "redémarrez l'application Claude.",
}

KIND_WORDS = {
    "PERSON": "personne(s)", "ORG": "société(s)", "ADDRESS": "adresse(s)", "EMAIL": "adresse(s) e-mail",
    "PHONE": "numéro(s) de téléphone", "IBAN": "IBAN", "NIR": "numéro(s) de sécurité sociale",
    "SIREN": "SIREN", "SIRET": "SIRET", "VAT": "numéro(s) de TVA", "CARD": "carte(s) bancaire(s)",
    "CASE_REF": "numéro(s) de procédure", "ID_CODE": "numéro(s) d'identité", "URL": "site(s) web",
    "BIRTHDATE": "date(s) de naissance", "USERNAME": "identifiant(s)", "EMPLOYEE_ID": "matricule(s)",
    "ACCOUNT_NUMBER": "numéro(s) de compte", "BIRTHPLACE": "lieu(x) de naissance",
}

TYPE_WORDS = {"pdf": "PDF", "docx": "Word", "txt": "texte", "md": "texte", "eml": "e-mail", "html": "page web",
              "unsupported": "illisible par Shield"}

LIMITS_FR = (
    "Shield remplace les noms et numéros détectés par des codes ; les vrais noms restent sur cet ordinateur. "
    "Ce n'est pas une anonymisation.",
    "La détection n'est jamais parfaite : relisez les passages sensibles.",
    "Un PDF scanné (une image sans texte) ne peut pas être lu.",
    "Un texte tapé ou collé dans la conversation, et un fichier joint à la conversation, ne sont PAS protégés "
    "par Shield : ils partent tels quels.",
    "Un nom seul dans un tableau, sans civilité ni prénom, peut être manqué.",
    "Les numéros de décisions publiées sont aussi remplacés par des codes.",
)

INSTRUCTIONS = """\
RAGbase Shield protège les dossiers clients de l'avocat. Shield remplace sur son ordinateur les noms, adresses,
IBAN et autres données privées par des codes : DOSSIER_1, DOC_3, PERSONNE_A, SOCIETE_B, ADRESSE_1, IBAN_1...
Vous ne voyez que les codes. Les vraies valeurs restent sur l'ordinateur.

Règles :
1. Pour tout travail sur des documents de clients, utilisez uniquement les outils Shield.
2. Pour ajouter un dossier, appelez shield_protect_folder. Une fenêtre s'ouvre sur l'ordinateur de l'avocat et il
   choisit le dossier lui-même. Ne demandez jamais le nom ni l'emplacement du dossier : ils contiennent souvent le
   nom des parties.
3. Ne demandez jamais les vrais noms, adresses ou numéros, et ne les écrivez jamais.
4. Si l'avocat joint un document à la conversation, ou tape ou colle de vrais noms, dites-lui clairement que ce
   texte n'est PAS protégé par Shield : il est parti tel quel. Proposez-lui de protéger le dossier avec Shield.
5. Recopiez les codes à l'identique (PERSONNE_A, jamais « Personne A »). N'inventez pas de code.
6. Enregistrez le document final, écrit avec les codes, avec shield_save_draft. Shield remet les vrais noms sur
   l'ordinateur, dans le dossier « Shield - sorties » du dossier client. Dites à l'avocat de l'ouvrir sur son
   ordinateur.
"""


def _msg(exc: BaseException, scope_code: Optional[str] = None) -> str:
    from .service import ShieldError

    if isinstance(exc, ShieldError):
        if exc.kind == "dossier_scope" and scope_code:
            return (f"Cette conversation travaille sur {scope_code}. Pour passer à un autre dossier, appelez "
                    "shield_protect_folder : l'avocat choisira le dossier dans la fenêtre (un dossier déjà protégé "
                    "garde son code).")
        return ERRORS.get(exc.kind, ERRORS["internal"])
    log.warning("RAGbase Shield (Claude tab): unexpected %s", type(exc).__name__)
    return ERRORS["internal"]


def _kb(n: Optional[int]) -> str:
    return "?" if n is None else f"{max(1, round(n / 1024))} Ko"


def _counts_line(counts: dict) -> str:
    total = sum(counts.values())
    if not total:
        return "Aucun nom ni numéro privé détecté pour l'instant."
    parts = [f"{n} {KIND_WORDS.get(k, 'autre(s)')}" for k, n in sorted(counts.items(), key=lambda kv: -kv[1])]
    return f"{total} élément(s) remplacé(s) par des codes : " + ", ".join(parts) + "."


# ================================================================================ server

def build_server():
    """The MCP server of the plugin: seven tools, French texts, one dossier per conversation at a time."""
    from mcp.server.mcpserver import MCPServer
    from mcp.shared.exceptions import MCPError
    from mcp.types import CallToolResult, TextContent, ToolAnnotations
    from pydantic import Field

    from . import __version__, mcp_server, service
    from .service import DEFAULT_READ_CHARS, MAX_READ_CHARS, MIN_READ_CHARS
    from .vault import UNSUPPORTED_KIND

    scope = mcp_server._Scope()
    lock = threading.Lock()  # one "choose a folder" window at a time

    def error_result(text: str) -> CallToolResult:
        # Returned, not raised: a raised ToolError reaches Claude with an English prefix ("Error executing tool").
        return CallToolResult(content=[TextContent(type="text", text=text)], is_error=True)

    def fail(exc: BaseException) -> CallToolResult:
        return error_result(_msg(exc, scope.pinned))

    def out(text: str, dossier: Optional[str] = None) -> str:
        return mcp_server._out(text, dossier)

    def scoped(dossier: str) -> str:
        scope.check(dossier)
        return dossier

    DossierArg = Annotated[str, Field(description="Code du dossier, par exemple DOSSIER_1.")]
    DocumentArg = Annotated[str, Field(description="Code du document, par exemple DOC_3.")]

    def shield_protect_folder() -> str:
        """Protéger un dossier client avec Shield. Ouvre sur l'ordinateur de l'avocat une fenêtre pour choisir le
        dossier : l'avocat le choisit lui-même. Ne demandez jamais le nom ni l'emplacement du dossier, ils
        contiennent souvent le nom des parties. Répond par le code du dossier (DOSSIER_1) et des nombres. Appelez
        aussi cet outil pour revenir à un dossier déjà protégé : il garde son code."""
        if not lock.acquire(blocking=False):
            return "Une fenêtre de choix est déjà ouverte sur l'ordinateur de l'avocat. Attendez qu'il la ferme."
        try:
            try:
                chosen = PICKER(PICK_TITLE, PICK_TIMEOUT)
            except PickCancelled:
                return PICK_MESSAGES["cancelled"]
            except PickFailed as exc:
                return PICK_MESSAGES.get(exc.kind, PICK_MESSAGES["failed"])
            try:
                folder = check_folder(chosen)
            except FolderRefused as exc:
                return REFUSALS[exc.kind]
            try:
                registered = service.registered_folders()
            except Exception as exc:  # noqa: BLE001
                return fail(exc)
            same = parent = None
            for c, f in registered.items():
                if _norm(Path(f)) == _norm(folder):
                    same = c
                elif _norm_inside(Path(f), folder):
                    return REFUSALS["holds_dossier"]  # one dossier must never hold several client dossiers
                elif _norm_inside(folder, Path(f)):
                    if parent is None or _norm_inside(Path(f), Path(registered[parent])):
                        parent = c  # the deepest protected folder that holds the chosen one
            try:
                if parent is not None and same is None:
                    r = service.scan_dossier(parent)  # a part of a protected dossier: same codes, no second vault
                    r["existing"] = True
                else:
                    r = service.add_dossier(folder)
                if service.register_folder_parties(r["code"]):
                    r.update({k: v for k, v in service.scan_dossier(r["code"]).items() if k != "existing"})
            except Exception as exc:  # noqa: BLE001
                return fail(exc)
            code = r["code"]
            with scope._lock:
                scope.pinned = code  # the lawyer chose this folder himself in the window
            readable = r["documents"] - r["unreadable"]
            if parent is not None and same is None:
                head = (f"Ce dossier fait partie de {code}, déjà protégé. Shield a relu {code} en entier ; la "
                        "conversation travaille sur ce dossier.")
            else:
                head = (f"Ce dossier était déjà protégé : {code}. Shield l'a relu." if r.get("existing")
                        else f"Le dossier choisi est maintenant protégé : {code}.")
            lines = [head,
                     f"{r['documents']} document(s) trouvé(s), {readable} lu(s)"
                     + (f", {r['unreadable']} illisible(s)" if r["unreadable"] else "")
                     + (f", dont {r['needs_ocr']} scanné(s) sans texte" if r["needs_ocr"] else "") + ".",
                     _counts_line(r.get("counts") or {}),
                     "Les vrais noms restent sur cet ordinateur.",
                     f"Suite : shield_list_documents avec {code}."]
            return out("\n".join(lines), code)
        finally:
            lock.release()

    def shield_list_dossiers() -> str:
        """Lister les dossiers protégés par Shield, par code (DOSSIER_1...), avec le nombre de documents. Ne montre
        jamais le nom d'un dossier."""
        try:
            rows = service.list_dossiers()
            pinned = scope.pinned
            if not rows:
                return out("Aucun dossier n'est encore protégé. Appelez shield_protect_folder : l'avocat choisira "
                           "le dossier dans une fenêtre sur son ordinateur.")
            if pinned is not None:
                rows = [r for r in rows if r.get("code") == pinned]
            lines = [f"{len(rows)} dossier(s) protégé(s) :" if pinned is None
                     else f"Cette conversation travaille sur {pinned} :"]
            for r in rows:
                if r.get("error"):
                    lines.append(f"- {r['code']} : coffre illisible")
                else:
                    lines.append(f"- {r['code']} : {r['documents']} document(s), {r['entities']} élément(s) "
                                 "remplacé(s) par des codes")
            if pinned is None:
                lines.append("La conversation se limitera au premier dossier ouvert. Pour un autre dossier : "
                             "shield_protect_folder.")
            lines.append("Suite : shield_list_documents avec le code du dossier.")
            return out("\n".join(lines))
        except Exception as exc:  # noqa: BLE001
            return fail(exc)

    def shield_list_documents(dossier: DossierArg) -> str:
        """Lister les documents d'un dossier protégé : code (DOC_1...), type, taille et nom du fichier. Les noms
        sont remplacés par des codes ou par « … » ; les sous-dossiers sont numérotés (sous-dossier 1, 2...)."""
        try:
            docs = service.list_documents(scoped(dossier))
            folders = service.document_folders(dossier)
            scope.used(dossier)
            code = dossier.strip().upper()
            sub_labels: dict[str, str] = {}  # real sub-folder -> "sous-dossier n" (stable: DOC order)
            for d in docs:
                f = folders.get(d["id"], "")
                if f and f not in sub_labels:
                    sub_labels[f] = f"sous-dossier {len(sub_labels) + 1}"
            if not docs:
                return out(f"{code} : aucun document lisible (Shield lit les PDF, Word, textes, e-mails et pages "
                           "web).", code)
            lines = [f"{code} : {len(docs)} document(s). Code | type | taille | nom (noms remplacés par des codes ou "
                     "par « … », sous-dossiers numérotés)"]
            unsupported = 0
            for d in docs:
                size = _kb(d["size"])
                if d.get("pages"):
                    size += f", {d['pages']} page(s)"
                if d["kind"] == UNSUPPORTED_KIND:
                    unsupported += 1
                name = safe_file_name(d["masked_name"])
                sub = sub_labels.get(folders.get(d["id"], ""))
                lines.append(f"{d['id']} | {TYPE_WORDS.get(d['kind'], d['kind'])} | {size} | "
                             + (f"{sub} / {name}" if sub else name))
            if unsupported:
                lines.append(f"{unsupported} fichier(s) que Shield ne sait pas lire (tableur, image...). Si leur "
                             "contenu compte, demandez à l'utilisateur une copie en PDF, Word ou texte.")
            lines.append("Suite : shield_read_document ou shield_search.")
            return out("\n".join(lines), code)
        except Exception as exc:  # noqa: BLE001
            return fail(exc)

    def shield_read_document(
        dossier: DossierArg,
        document: DocumentArg,
        start: Annotated[int, Field(description="Où commencer, en caractères : 0 au début, puis la valeur « suite » "
                                                "donnée par la réponse précédente.", ge=0)] = 0,
        max_chars: Annotated[int, Field(description=f"Nombre de caractères à lire ({MIN_READ_CHARS} à "
                                                    f"{MAX_READ_CHARS}).", ge=1)] = DEFAULT_READ_CHARS,
    ) -> str:
        """Lire un document d'un dossier protégé. Le texte arrive avec des codes (PERSONNE_A, SOCIETE_B, IBAN_1) à
        la place des vraies valeurs. Un long document arrive en plusieurs parties : rappelez l'outil avec la valeur
        « suite »."""
        try:
            r = service.read_document(scoped(dossier), document, start, max_chars)
            scope.used(dossier)
            code = dossier.strip().upper()
            if r["needs_ocr"]:
                return out(f"{code} / {r['id']} : ce PDF est une image scannée, sans texte. Shield ne peut pas le "
                           "lire. Demandez à l'utilisateur une version texte ou Word.", code)
            end = r["start"] + len(r["text"])
            head = f"[{code} / {r['id']} · caractères {r['start']} à {end} sur {r['total_chars']}"
            if r.get("pages"):
                head += f" · {r['pages']} page(s)"
            head += f" · suite : start={r['next']}]" if r["next"] is not None else " · fin du document]"
            if r.get("warnings"):
                head += "\n[remarque : une partie du document n'a peut-être pas pu être lue]"
            return out(head + "\n" + r["text"], code)
        except Exception as exc:  # noqa: BLE001
            return fail(exc)

    def shield_search(
        dossier: DossierArg,
        query: Annotated[str, Field(description="Mots à chercher. Les codes marchent aussi (PERSONNE_A) : Shield "
                                                "cherche sur l'ordinateur.")],
    ) -> str:
        """Chercher des mots dans les documents d'un dossier protégé. Shield cherche sur l'ordinateur et renvoie
        des extraits avec les codes."""
        try:
            hits = service.search(scoped(dossier), query)
            scope.used(dossier)
            code = dossier.strip().upper()
            if not hits:
                return out(f"{code} : aucun résultat.", code)
            lines = [f"{code} : {len(hits)} résultat(s). Pour lire la suite : shield_read_document avec start."]
            for i, h in enumerate(hits, 1):
                lines.append(f"{i}. {h['document']} (start={h['offset']}) : {h['snippet']}")
            return out("\n".join(lines), code)
        except Exception as exc:  # noqa: BLE001
            return fail(exc)

    def shield_save_draft(
        dossier: DossierArg,
        title: Annotated[str, Field(description="Titre du fichier, avec des codes si besoin.")],
        content: Annotated[str, Field(description="Le document final, écrit avec les codes (titres #, listes, "
                                                  "**gras** et tableaux acceptés).")],
        format: Annotated[Literal["docx", "md"], Field(description="docx (Word, par défaut) ou md (texte).")] = "docx",
    ) -> str:
        """Enregistrer le document final, écrit avec les codes, dans le dossier « Shield - sorties » du dossier
        client. Shield remet les vrais noms sur l'ordinateur uniquement et répond par un code (OUTPUT_1)."""
        try:
            r = service.save_draft_report(scoped(dossier), title, content, format)
            scope.used(dossier)
            code = dossier.strip().upper()
            msg = (f"Enregistré : {r['id']} ({_kb(r['bytes'])}, {'Word' if r['format'] == 'docx' else 'texte'}) "
                   f"dans le dossier « Shield - sorties » de {code}. Les vrais noms y sont remis, sur l'ordinateur "
                   "uniquement. Dites à l'avocat d'ouvrir ce fichier sur son ordinateur.")
            if r["unknown_codes"]:
                msg += ("\nCes codes n'existent pas dans ce dossier et sont restés tels quels : "
                        + ", ".join(r["unknown_codes"]) + ".")
            placeholders = list(r.get("placeholders") or [])
            if placeholders:
                items = ", ".join(f"{p} ×{n}" if n > 1 else p for p, n in Counter(placeholders).items())
                msg += (f"\nAttention : ces marques ne peuvent pas être remplacées par un vrai nom : {items}. "
                        "Réécrivez ces passages avec les codes du dossier, puis enregistrez à nouveau.")
            ids = list(r.get("shield_ids") or [])
            if ids:
                msg += (f"\nCes codes Shield sont restés dans le fichier : {', '.join(ids)}. Remplacez-les par une "
                        "description de la pièce, puis enregistrez à nouveau.")
            return out(msg, code)
        except Exception as exc:  # noqa: BLE001
            return fail(exc)

    def shield_status() -> str:
        """Voir l'état de Shield sur cet ordinateur et ses limites."""
        try:
            s = service.status()
            lines = [f"RAGbase Shield {s['version']} fonctionne sur cet ordinateur, sans connexion à internet. "
                     f"{s['dossiers']} dossier(s) protégé(s)." + (f" Cette conversation travaille sur {scope.pinned}."
                                                                if scope.pinned else ""),
                     "Limites :"]
            lines += [f"- {t}" for t in LIMITS_FR]
            return out("\n".join(lines))
        except Exception as exc:  # noqa: BLE001
            return fail(exc)

    ro = dict(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)
    tools = (
        (shield_protect_folder, "Protéger un dossier",
         dict(read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=False)),
        (shield_list_dossiers, "Lister les dossiers", ro),
        (shield_list_documents, "Lister les documents", ro),
        (shield_read_document, "Lire un document", ro),
        (shield_search, "Chercher dans un dossier", ro),
        (shield_save_draft, "Enregistrer le document final",
         dict(read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False)),
        (shield_status, "État de Shield", ro),
    )
    class _FrenchServer(MCPServer):
        async def call_tool(self, name, arguments, context=None):
            try:
                return await super().call_tool(name, arguments, context)
            except MCPError:
                raise
            except Exception as exc:  # noqa: BLE001 - bad arguments or unknown tool: French text, no detail
                log.info("RAGbase Shield (Claude tab): call refused (%s)", type(exc).__name__)
                return error_result(ERRORS["invalid_argument"])

    server = _FrenchServer(name="RAGbase Shield", title="RAGbase Shield",
                       description="Protège les dossiers clients : les noms restent sur l'ordinateur.",
                       instructions=INSTRUCTIONS, version=__version__, website_url="https://shield.ragbase.ai",
                       log_level="WARNING")
    for fn, title, hints in tools:
        server.add_tool(fn, name=fn.__name__, title=title, annotations=ToolAnnotations(title=title, **hints),
                        structured_output=False)
    server._shield_scope = scope  # tests
    return server


TOOL_NAMES = ("shield_protect_folder", "shield_list_dossiers", "shield_list_documents", "shield_read_document",
              "shield_search", "shield_save_draft", "shield_status")


# ================================================================================ entry point

def _parse(argv: list[str]) -> tuple[Optional[Path], Optional[str]]:
    """Test switches only: --test-data <folder> (scratch data + file key) and --test-pick <folder> (the answer of the
    "choose a folder" window). Anything else stops the start: the plugin passes no argument."""
    test_data = test_pick = None
    it = iter(argv)
    for a in it:
        if a == "--test-data":
            test_data = Path(next(it))
        elif a == "--test-pick":
            test_pick = next(it)
        else:
            raise SystemExit(2)
    if test_pick is not None and test_data is None:
        raise SystemExit(2)  # never point the real vault at a test answer
    return test_data, test_pick


def main(argv: Optional[list[str]] = None) -> int:
    global PICKER
    argv = list(sys.argv[1:] if argv is None else argv)
    test_data, test_pick = _parse(argv)
    if not logging.getLogger().handlers:
        logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    lock_settings(test_data)
    if test_pick is not None:
        answer = test_pick
        PICKER = lambda title, timeout: answer  # noqa: E731
    build_server().run("stdio")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
