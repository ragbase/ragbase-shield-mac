"""System clipboard without extra dependencies: macOS (pbcopy/pbpaste), Windows (PowerShell Get-Clipboard /
Set-Clipboard), Linux (wl-copy/wl-paste, xclip, xsel). Used by ``ragbase-shield mask|restore --clipboard``.

Local only: the text goes to the operating system clipboard, nowhere else. Error messages never contain the
clipboard text.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Optional

__all__ = ["ClipboardError", "backend", "copy", "paste"]

_TIMEOUT = 15


class ClipboardError(Exception):
    """The clipboard is not available or a clipboard command failed (message without any clipboard text)."""


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("LANG", "en_US.UTF-8")
    env["LC_CTYPE"] = "UTF-8" if sys.platform == "darwin" else env.get("LC_CTYPE", "C.UTF-8")
    return env


def _powershell() -> Optional[str]:
    return shutil.which("powershell") or shutil.which("pwsh")


def backend() -> Optional[str]:
    """Name of the clipboard tool that will be used, or None if none is available."""
    if sys.platform == "darwin":
        return "pbcopy" if shutil.which("pbcopy") and shutil.which("pbpaste") else None
    if sys.platform.startswith("win"):
        return "powershell" if _powershell() else None
    if os.environ.get("WAYLAND_DISPLAY") and shutil.which("wl-copy") and shutil.which("wl-paste"):
        return "wl-clipboard"
    if shutil.which("xclip"):
        return "xclip"
    if shutil.which("xsel"):
        return "xsel"
    return None


def _missing() -> ClipboardError:
    if sys.platform.startswith("linux"):
        return ClipboardError("Aucun outil de presse-papiers trouvé : installez wl-clipboard (Wayland), xclip ou "
                              "xsel. (No clipboard tool found: install wl-clipboard, xclip or xsel.)")
    return ClipboardError("Presse-papiers indisponible sur ce système. (Clipboard not available on this system.)")


def _run(cmd: list[str], data: Optional[bytes] = None) -> bytes:
    try:
        proc = subprocess.run(cmd, input=data, capture_output=True, timeout=_TIMEOUT, env=_env(), check=False)
    except (OSError, subprocess.SubprocessError):
        raise ClipboardError("La commande du presse-papiers a échoué. (Clipboard command failed.)") from None
    if proc.returncode != 0:
        raise ClipboardError("La commande du presse-papiers a échoué. (Clipboard command failed.)")
    return proc.stdout


def copy(text: str) -> None:
    """Replace the clipboard content with ``text``."""
    name = backend()
    data = text.encode("utf-8")
    if name == "pbcopy":
        _run(["pbcopy"], data)
    elif name == "powershell":
        script = ("[Console]::InputEncoding = [System.Text.Encoding]::UTF8; "
                  "$t = [Console]::In.ReadToEnd(); Set-Clipboard -Value $t")
        _run([_powershell() or "powershell", "-NoProfile", "-NonInteractive", "-Command", script], data)
    elif name == "wl-clipboard":
        _run(["wl-copy"], data)
    elif name == "xclip":
        _run(["xclip", "-selection", "clipboard", "-in"], data)
    elif name == "xsel":
        _run(["xsel", "--clipboard", "--input"], data)
    else:
        raise _missing()


def paste() -> str:
    """The clipboard content as text ("" when it is empty)."""
    name = backend()
    if name == "pbcopy":
        out = _run(["pbpaste"])
    elif name == "powershell":
        script = ("[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
                  "$t = Get-Clipboard -Raw; if ($t) { [Console]::Out.Write($t) }")
        out = _run([_powershell() or "powershell", "-NoProfile", "-NonInteractive", "-Command", script])
    elif name == "wl-clipboard":
        out = _run(["wl-paste", "--no-newline"])
    elif name == "xclip":
        out = _run(["xclip", "-selection", "clipboard", "-out"])
    elif name == "xsel":
        out = _run(["xsel", "--clipboard", "--output"])
    else:
        raise _missing()
    text = out.decode("utf-8", errors="replace")
    if out.startswith(b"\xef\xbb\xbf"):
        text = text[1:]
    return text.replace("\r\n", "\n") if sys.platform.startswith("win") else text
