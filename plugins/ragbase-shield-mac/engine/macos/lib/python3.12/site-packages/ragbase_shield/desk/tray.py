"""Menu bar (macOS) and tray icon (Windows) of the background Shield Desk (packaging/DESIGN.md section 5).

The menu model, the strings and the actions are shared; a backend only draws:

- ``_RumpsTray`` (macOS): rumps (BSD-3) on pyobjc (MIT), in the Cocoa main thread;
- ``_SystrayTray`` (Windows): infi.systray (BSD-3, ctypes only), in its own window thread;
- ``make_tray()`` returns None when no GUI session or library is available: the supervisor then keeps the
  Desk running without an icon (logged as ``tray: unavailable (<reason>)``).

Token rules (section 5.2): the token is read from ``<home>/desk_token`` just before each use, kept in a local
variable, sent only in the ``X-Shield-Token`` header to 127.0.0.1 and in the ``?t=`` link opened by the system
(NSWorkspace / ShellExecute): never logged, shown, copied or passed to a shell. The tray never imports the
engine and never opens a vault. Dossier titles are client data: they appear only in the local macOS
"Dossiers" submenu (a local UI, ARCHITECTURE.md section 6), never in a log line or a dialog.
"""
from __future__ import annotations

import locale as _locale
import logging
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from . import service_manager as sm

log = logging.getLogger("ragbase_shield.service_manager")

REFRESH = 2.0  # seconds between two menu refreshes (the supervisor polls the Desk every 10 s)

STRINGS = {
    "fr": {
        "status": "Shield Desk : {state}",
        "running": "en marche, port {port}, v{version}, {mode}",
        "starting": "démarrage",
        "gave_up": "arrêté, voir le journal",
        "port_busy": "port {port} occupé par un autre programme",
        "elsewhere": "lancé ailleurs (port {port})",
        "stopping": "arrêt",
        "rules": "règles",
        "rules_ner": "règles + NER",
        "open": "Ouvrir Shield Desk",
        "dossiers": "Dossiers",
        "dossiers_page": "Dossiers (Shield Desk)",
        "no_dossier": "Aucun dossier",
        "data_folder": "Dossier de données Shield",
        "restart": "Redémarrer",
        "login": "Lancer à l'ouverture de session",
        "logs": "Ouvrir le journal",
        "retry": "Réessayer",
        "uninstall": "Désinstaller...",
        "quit": "Quitter",
        "status_item": "Statut...",
        "login_on": "Ouverture de session : oui",
        "login_off": "Ouverture de session : non",
        "uninstall_title": "Désinstaller RAGbase Shield ?",
        "uninstall_text": "Vos dossiers et vos coffres restent dans {home}. Pour tout effacer, supprimez ce dossier "
                          "ensuite (les anciennes réponses ne pourront plus être restaurées).",
        "uninstall_ok": "Désinstaller",
        "cancel": "Annuler",
        "quit_title": "Quitter RAGbase Shield ?",
        "quit_text": "Shield Desk s'arrête jusqu'à votre prochaine ouverture de session. L'extension de navigateur "
                     "et la page Shield Desk ne répondront plus. Continuer ?",
        "quit_ok": "Quitter",
        "not_ready": "Shield Desk ne répond pas encore. Réessayez dans un instant ou ouvrez le journal.",
    },
    "en": {
        "status": "Shield Desk: {state}",
        "running": "running, port {port}, v{version}, {mode}",
        "starting": "starting",
        "gave_up": "stopped, see the log",
        "port_busy": "port {port} used by another program",
        "elsewhere": "started elsewhere (port {port})",
        "stopping": "stopping",
        "rules": "rules",
        "rules_ner": "rules + NER",
        "open": "Open Shield Desk",
        "dossiers": "Dossiers",
        "dossiers_page": "Dossiers (Shield Desk)",
        "no_dossier": "No dossier",
        "data_folder": "Shield data folder",
        "restart": "Restart",
        "login": "Start at login",
        "logs": "Open the log",
        "retry": "Try again",
        "uninstall": "Uninstall...",
        "quit": "Quit",
        "status_item": "Status...",
        "login_on": "Start at login: yes",
        "login_off": "Start at login: no",
        "uninstall_title": "Uninstall RAGbase Shield?",
        "uninstall_text": "Your dossiers and vaults stay in {home}. To erase everything, delete that folder "
                          "afterwards (old answers can then no longer be restored).",
        "uninstall_ok": "Uninstall",
        "cancel": "Cancel",
        "quit_title": "Quit RAGbase Shield?",
        "quit_text": "Shield Desk stops until your next login. The browser extension and the Shield Desk page will "
                     "not answer. Continue?",
        "quit_ok": "Quit",
        "not_ready": "Shield Desk does not answer yet. Try again in a moment or open the log.",
    },
}


def ui_language() -> str:
    """RAGBASE_SHIELD_UI=fr|en, else the system language, else French."""
    v = os.environ.get("RAGBASE_SHIELD_UI", "").strip().lower()
    if v.startswith("en"):
        return "en"
    if v.startswith("fr"):
        return "fr"
    lang = ""
    if sys.platform == "darwin":
        try:
            from Foundation import NSLocale  # type: ignore[import-not-found]

            langs = NSLocale.preferredLanguages()
            lang = str(langs[0]) if langs else ""
        except Exception:  # noqa: BLE001
            lang = ""
    if not lang:
        try:
            lang = _locale.getlocale()[0] or os.environ.get("LANG", "")
        except (ValueError, TypeError):
            lang = os.environ.get("LANG", "")
    return "en" if lang.lower().startswith("en") else "fr"


def t(key: str, **kw: Any) -> str:
    s = STRINGS[ui_language()][key]
    return s.format(**kw) if kw else s


def mode_label(engine_mode: Optional[str]) -> str:
    return t("rules_ner") if engine_mode and "ner" in str(engine_mode) else t("rules")


def status_text(snap: dict) -> str:
    state = snap.get("state")
    port = snap.get("port")
    if state == sm.RUNNING:
        inner = t("running", port=port, version=snap.get("version") or "?", mode=mode_label(snap.get("engine_mode")))
    elif state == sm.PORT_BUSY:
        inner = t("port_busy", port=port)
    elif state == sm.ELSEWHERE:
        inner = t("elsewhere", port=port)
    elif state == sm.GAVE_UP:
        inner = t("gave_up")
    elif state == sm.STOPPING:
        inner = t("stopping")
    else:
        inner = t("starting")
    return t("status", state=inner)


# ------------------------------------------------------------------------------------------ actions

def open_url(url: str) -> bool:
    """Open a URL with the system (never a shell, never an argv holding the token)."""
    if sys.platform == "darwin":
        try:
            from AppKit import NSWorkspace  # type: ignore[import-not-found]
            from Foundation import NSURL  # type: ignore[import-not-found]

            return bool(NSWorkspace.sharedWorkspace().openURL_(NSURL.URLWithString_(url)))
        except Exception as exc:  # noqa: BLE001
            log.warning("tray: open url failed (%s)", type(exc).__name__)
            return False
    if sys.platform.startswith("win"):
        try:
            os.startfile(url)  # type: ignore[attr-defined]  # ShellExecute
            return True
        except OSError as exc:
            log.warning("tray: open url failed (%s)", type(exc).__name__)
            return False
    return False


def open_folder(path: Path) -> bool:
    if not path.exists():
        return False
    if sys.platform == "darwin":
        try:
            from AppKit import NSWorkspace  # type: ignore[import-not-found]
            from Foundation import NSURL  # type: ignore[import-not-found]

            return bool(NSWorkspace.sharedWorkspace().openURL_(NSURL.fileURLWithPath_(str(path))))
        except Exception as exc:  # noqa: BLE001
            log.warning("tray: open folder failed (%s)", type(exc).__name__)
            return False
    if sys.platform.startswith("win"):
        try:
            os.startfile(str(path))  # type: ignore[attr-defined]
            return True
        except OSError:
            return False
    return False


class Controller:
    """What the menu items do. Shared by every backend."""

    def __init__(self, sup: "sm.Supervisor", app: Optional[Path]):
        self.sup = sup
        self.app = app
        self.home = sup.home
        self.dossiers: list[dict] = []
        self._dossiers_at = 0.0

    # -- state
    def snapshot(self) -> dict:
        return self.sup.snapshot()

    def desk_ready(self) -> bool:
        return self.sup.snapshot().get("state") in (sm.RUNNING, sm.ELSEWHERE)

    def refresh_dossiers(self, force: bool = False) -> bool:
        """Re-read GET /api/dossiers (loopback, token header) every 10 s. True when the list changed."""
        now = time.monotonic()
        if not force and now - self._dossiers_at < sm.POLL_STATUS:
            return False
        self._dossiers_at = now
        if not self.desk_ready():
            new: list[dict] = []
        else:
            st, data = sm.probe(self.sup.port, sm.read_token(self.home), path="/api/dossiers")
            new = [{"code": str(r.get("code")), "title": str(r.get("title") or r.get("code")),
                    "folder": str(r.get("folder") or "")}
                   for r in (data if st == "ok" and isinstance(data, list) else []) if isinstance(r, dict)]
        changed = new != self.dossiers
        self.dossiers = new
        return changed

    # -- actions
    def _open_desk_now(self, fragment: str = "") -> bool:
        token = sm.read_token(self.home)
        if not token:
            return False
        return open_url(f"http://127.0.0.1:{self.sup.port}/?t={token}" + (f"#{fragment}" if fragment else ""))

    def open_desk(self, fragment: str = "", notify: Optional[Callable[[str], None]] = None) -> None:
        if self.desk_ready() and self._open_desk_now(fragment):
            return
        if self.sup.snapshot().get("state") == sm.GAVE_UP:
            self.sup.request("retry")

        def wait_then_open() -> None:
            end = time.monotonic() + sm.READY_TIMEOUT
            while time.monotonic() < end:
                if self.desk_ready() and self._open_desk_now(fragment):
                    return
                time.sleep(1.0)
            if notify:
                notify(t("not_ready"))

        threading.Thread(target=wait_then_open, name="shield-open-desk", daemon=True).start()

    def open_dossier(self, folder: str) -> None:
        if folder:
            open_folder(Path(folder))

    def open_home(self) -> None:
        open_folder(self.home)

    def open_logs(self) -> None:
        open_folder(sm.logs_dir(self.home))

    def restart(self) -> None:
        self.sup.request("restart")

    def retry(self) -> None:
        self.sup.request("retry")

    def login_enabled(self) -> bool:
        return sm.login_item_enabled()

    def toggle_login(self) -> bool:
        if sm.login_item_enabled():
            sm.disable_login_item()
            log.info("tray: login item off")
            return False
        sm.enable_login_item(sm.service_exe(self.app))
        log.info("tray: login item on")
        return True

    def uninstall(self) -> None:
        """Run the uninstaller detached (it stops this process); the data folder is always kept."""
        log.info("tray: uninstall")
        try:
            if sys.platform == "darwin" and self.app is not None and (self.app / "uninstall.sh").is_file():
                subprocess.Popen(["/bin/sh", str(self.app / "uninstall.sh"), "--from-tray"],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 start_new_session=True, close_fds=True)
                return
            if sys.platform.startswith("win") and self.app is not None:
                code = str(sm.read_install_info(self.app).get("product_code") or "")
                if re.fullmatch(r"\{[0-9A-Fa-f]{8}(-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}\}", code):
                    flags = (getattr(subprocess, "DETACHED_PROCESS", 0)
                             | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
                    # Started outside the app folder: a working folder inside it would keep it from being removed.
                    subprocess.Popen(["msiexec", "/x", code], creationflags=flags, close_fds=True,
                                     cwd=os.environ.get("SystemRoot") or None)
                    return
        except OSError as exc:
            log.warning("tray: uninstall failed (%s)", type(exc).__name__)
            return
        sm.uninstall()

    def quit(self) -> None:
        log.info("tray: quit")
        self.sup.request("stop")


# ------------------------------------------------------------------------------------------ macOS

class _RumpsTray:
    """Status item in the macOS menu bar (rumps, Cocoa main thread)."""

    def __init__(self, ctl: Controller, rumps: Any):
        self.ctl = ctl
        self.rumps = rumps
        icon = None
        if ctl.app is not None and (ctl.app / "icons" / "shield_template.png").is_file():
            icon = str(ctl.app / "icons" / "shield_template.png")
        self.app = rumps.App(sm.APP_NAME, title=None if icon else "Shield", icon=icon, template=True,
                             quit_button=None)
        self.status_item = rumps.MenuItem(status_text(ctl.snapshot()))
        self.dossiers_menu = rumps.MenuItem(t("dossiers"))
        self.login_item = rumps.MenuItem(t("login"), callback=self._toggle_login)
        self.retry_item = rumps.MenuItem(t("retry"), callback=lambda _s: ctl.retry())
        self.app.menu = [
            self.status_item,
            None,
            rumps.MenuItem(t("open"), callback=lambda _s: ctl.open_desk(notify=self._notify)),
            self.dossiers_menu,
            None,
            rumps.MenuItem(t("restart"), callback=lambda _s: ctl.restart()),
            self.login_item,
            rumps.MenuItem(t("logs"), callback=lambda _s: ctl.open_logs()),
            None,
            rumps.MenuItem(t("uninstall"), callback=self._uninstall),
            rumps.MenuItem(t("quit"), callback=self._quit),
        ]
        self._shown_status = ""
        self._retry_shown = False
        self._reported = False
        self._build_dossiers()
        self.login_item.state = 1 if ctl.login_enabled() else 0
        self.timer = rumps.Timer(self._refresh, REFRESH)

    def _notify(self, text: str) -> None:
        log.info("tray: desk not ready")

    def _build_dossiers(self) -> None:
        menu = self.dossiers_menu
        try:
            menu.clear()
        except Exception:  # noqa: BLE001 - empty submenu
            pass
        seen: set[str] = set()
        for d in self.ctl.dossiers:
            folder = d["folder"]
            title = d["title"] if d["title"] not in seen else f"{d['title']} ({d['code']})"
            seen.add(title)
            menu.add(self.rumps.MenuItem(title, callback=(lambda _s, f=folder: self.ctl.open_dossier(f))))
        if not self.ctl.dossiers:
            menu.add(self.rumps.MenuItem(t("no_dossier")))
        menu.add(None)
        menu.add(self.rumps.MenuItem(t("data_folder"), callback=lambda _s: self.ctl.open_home()))

    def _refresh(self, _timer: Any) -> None:
        if self.ctl.sup.stopped.is_set():
            self.timer.stop()
            self.rumps.quit_application()
            return
        if not self._reported:  # the app's own evidence that its status item exists (support and tests)
            self._reported = True
            try:
                si = self.app._nsapp.nsstatusitem
                log.info("tray: status item visible=%s icon=%s", bool(si.isVisible()),
                         si.button().image() is not None)
            except Exception as exc:  # noqa: BLE001 - rumps internals: informative only
                log.info("tray: status item check skipped (%s)", type(exc).__name__)
        snap = self.ctl.snapshot()
        text = status_text(snap)
        if text != self._shown_status:
            self.status_item.title = text
            self._shown_status = text
            log.info("tray: status line \"%s\"", text)  # state, port, version, mode: no dossier, no token
        gave_up = snap.get("state") == sm.GAVE_UP
        if gave_up and not self._retry_shown:
            self.app.menu.insert_after(self.status_item.title, self.retry_item)
            self._retry_shown = True
        elif not gave_up and self._retry_shown:
            try:
                del self.app.menu[self.retry_item.title]
            except KeyError:
                pass
            self._retry_shown = False
        if self.ctl.refresh_dossiers():
            self._build_dossiers()
        self.login_item.state = 1 if self.ctl.login_enabled() else 0

    def _toggle_login(self, sender: Any) -> None:
        sender.state = 1 if self.ctl.toggle_login() else 0

    def _uninstall(self, _sender: Any) -> None:
        ok = self.rumps.alert(title=t("uninstall_title"), message=t("uninstall_text", home=str(self.ctl.home)),
                              ok=t("uninstall_ok"), cancel=t("cancel"))
        if ok == 1:
            self.ctl.uninstall()

    def _quit(self, _sender: Any) -> None:
        ok = self.rumps.alert(title=t("quit_title"), message=t("quit_text"), ok=t("quit_ok"), cancel=t("cancel"))
        if ok == 1:
            self.ctl.quit()

    def run(self) -> None:
        try:
            from AppKit import NSApplication, NSApplicationActivationPolicyAccessory  # type: ignore

            NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)
        except Exception:  # noqa: BLE001 - the .app shell already has LSUIElement
            pass
        self.timer.start()
        log.info("tray: started (rumps)")
        self.app.run()
        log.info("tray: stopped")


# ------------------------------------------------------------------------------------------ Windows

def _shield_systray_class(base: Any, confirm_quit: Callable[[], bool]) -> Any:
    """infi.systray 0.1.12 with three fixes (Windows build step, 2026-10-27):

    - its own last item "Quit" is in English and quits at once: it gets our label and a confirmation dialog;
    - "TaskbarCreated" (Explorer restarted, or started after us at login) must ADD the icon again: the library
      sends NIM_MODIFY for an icon Explorer no longer knows, so the icon never came back;
    - the library calls the ANSI ("A") functions with ``locale.getpreferredencoding()``, which is UTF-8 under
      ``-X utf8``: French labels came out as mojibake. ``_SystrayTray`` sets the ANSI code page ("mbcs").
    """

    class _ShieldSysTrayIcon(base):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            text, icon, action, item_id = self._menu_options[-1]
            if action == base.QUIT:
                self._menu_options[-1] = (t("quit"), icon, action, item_id)

        def _restart(self, hwnd: Any, msg: Any, wparam: Any, lparam: Any) -> None:
            self._notify_id = None  # next refresh sends NIM_ADD
            self._refresh_icon()

        def _execute_menu_option(self, id: int) -> None:  # noqa: A002 - library signature
            if self._menu_actions_by_id.get(id) == base.QUIT and not confirm_quit():
                return
            super()._execute_menu_option(id)

    return _ShieldSysTrayIcon


class _SystrayTray:
    """Icon in the Windows notification area (infi.systray, own window thread)."""

    def __init__(self, ctl: Controller, systray_cls: Any):
        self.ctl = ctl
        try:  # the library encodes for the ANSI functions; -X utf8 would make that UTF-8 (mojibake)
            from infi.systray import win32_adapter  # type: ignore[import-not-found]

            win32_adapter.LOCALE_ENCODING = "mbcs"
        except Exception:  # noqa: BLE001 - a test double has no adapter
            pass
        icon = str(ctl.app / "icons" / "shield.ico") if ctl.app is not None else None
        menu = ((t("status_item"), None, self._status_box),
                (t("open"), None, lambda _s: ctl.open_desk()),
                (t("dossiers_page"), None, lambda _s: ctl.open_desk(fragment="dossiers")),
                (t("data_folder"), None, lambda _s: ctl.open_home()),
                (t("restart"), None, lambda _s: ctl.restart()),
                (t("login"), None, lambda _s: ctl.toggle_login()),
                (t("logs"), None, lambda _s: ctl.open_logs()),
                (t("retry"), None, lambda _s: ctl.retry()),
                (t("uninstall"), None, self._uninstall))
        cls = _shield_systray_class(systray_cls, self._confirm_quit) if hasattr(systray_cls, "QUIT") else systray_cls
        self.tray = cls(icon, status_text(ctl.snapshot()), menu, on_quit=self._on_quit, default_menu_index=1)

    def _confirm_quit(self) -> bool:
        return self._box(t("quit_text"), t("quit_title"), 0x21) == 1  # MB_OKCANCEL | MB_ICONQUESTION, IDOK

    def _box(self, text: str, title: str, flags: int) -> int:
        import ctypes

        return int(ctypes.windll.user32.MessageBoxW(None, text, title, flags))  # type: ignore[attr-defined]

    def _status_box(self, _s: Any) -> None:
        login = t("login_on") if self.ctl.login_enabled() else t("login_off")
        self._box(status_text(self.ctl.snapshot()) + "\n" + login, sm.APP_NAME, 0x40)

    def _uninstall(self, _s: Any) -> None:
        if self._box(t("uninstall_text", home=str(self.ctl.home)), t("uninstall_title"), 0x21) == 1:
            self.ctl.uninstall()

    def _on_quit(self, _s: Any) -> None:
        if not self.ctl.sup.stopped.is_set():
            self.ctl.quit()

    def run(self) -> None:
        self.tray.start()
        end = time.monotonic() + 5.0
        while getattr(self.tray, "_hwnd", True) is None and time.monotonic() < end:
            time.sleep(0.05)
        if getattr(self.tray, "_hwnd", True) is None:
            log.info("tray: unavailable (window)")  # the supervisor goes on without an icon
        else:
            log.info("tray: started (infi.systray)")
        shown = ""
        try:
            while not self.ctl.sup.stopped.is_set():
                text = status_text(self.ctl.snapshot())
                if text != shown:
                    try:
                        self.tray.update(hover_text=text[:120])
                    except Exception:  # noqa: BLE001
                        pass
                    shown = text
                self.ctl.sup.stopped.wait(REFRESH)
        finally:
            try:
                self.tray.shutdown()
            except Exception:  # noqa: BLE001
                pass
        log.info("tray: stopped")


# ------------------------------------------------------------------------------------------ factory

def windows_session_id() -> Optional[int]:
    """Terminal Services session of this process (0 = services and SSH logons, which have no desktop)."""
    try:
        import ctypes

        sid = ctypes.c_ulong(0)
        k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        if k32.ProcessIdToSessionId(k32.GetCurrentProcessId(), ctypes.byref(sid)):
            return int(sid.value)
    except Exception:  # noqa: BLE001
        pass
    return None


def make_tray(sup: "sm.Supervisor", app: Optional[Path]) -> Optional[Any]:
    """The tray for this platform, or None (logged ``tray: unavailable (<reason>)``)."""
    if os.environ.get("RAGBASE_SHIELD_NO_TRAY", "").strip() in ("1", "true", "yes"):
        log.info("tray: unavailable (disabled)")
        return None
    ctl = Controller(sup, app)
    if sys.platform == "darwin":
        if os.environ.get("SSH_CONNECTION") and not os.environ.get("RAGBASE_SHIELD_SERVICE"):
            log.info("tray: unavailable (no_desktop)")
            return None
        try:
            import rumps  # type: ignore[import-not-found]
        except Exception:  # noqa: BLE001
            log.info("tray: unavailable (no_library)")
            return None
        try:
            return _RumpsTray(ctl, rumps)
        except Exception as exc:  # noqa: BLE001
            log.info("tray: unavailable (%s)", type(exc).__name__)
            return None
    if sys.platform.startswith("win"):
        try:
            from infi.systray import SysTrayIcon  # type: ignore[import-not-found]
        except Exception:  # noqa: BLE001
            log.info("tray: unavailable (no_library)")
            return None
        try:
            if windows_session_id() == 0:  # services and SSH logons: no desktop, an icon there is never seen
                log.info("tray: unavailable (no_desktop)")
                return None
            return _SystrayTray(ctl, SysTrayIcon)
        except Exception as exc:  # noqa: BLE001
            log.info("tray: unavailable (%s)", type(exc).__name__)
            return None
    log.info("tray: unavailable (platform)")
    return None
