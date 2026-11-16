"""Claude Code hook handlers (``ragbase-shield hook <event>``, JSON on stdin, JSON on stdout; ARCHITECTURE 4.6).

Schema: https://code.claude.com/docs/en/hooks (read 2026-10-16).

- ``user-prompt-submit``: a prompt that holds a registered value of any dossier is blocked. The output is
  ``{"decision": "block", "reason": ...}`` plus ``hookSpecificOutput.suppressOriginalPrompt``, the reason
  also goes to stderr and the exit code is 2, so the prompt is blocked even by a Claude Code version that
  does not know one of the JSON fields. The reason names the CODE and the kind, never the value.
- ``pre-tool-use``: Read / Grep / Glob / Edit / Write / LS / LSP, Bash / PowerShell / Monitor commands, the
  Artifact and SendUserFile uploads and the file tools of OTHER MCP servers that touch a registered dossier
  folder, the Shield data folder or Shield Desk are denied (``hookSpecificOutput.permissionDecision = "deny"``,
  exit 2, reason on stderr) and Claude is pointed to ``shield_read_document``. Shell commands that change the
  vault or print real values (``ragbase-shield keep-clear``, ``register``, ``entities``...) are denied too.
  Bash analysis is best effort; PostToolUse is the second net.
- ``post-tool-use``: a tool result that still holds a registered value is replaced by its masked form with
  ``hookSpecificOutput.updatedToolOutput`` (all tools since Claude Code 2.1.121; same shape as the original,
  only strings change, a number that is a registered value becomes its code) and ``updatedMCPToolOutput`` for
  MCP tools. Shield's own MCP tools are skipped: their results already went through the same canary.

No output ever holds a real value. If Shield itself fails, or the check would not finish in time (a very large
prompt or tool result), the hook fails closed: the prompt is blocked, the tool call is denied, or the tool
result is blanked and Claude is stopped.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shlex
import sqlite3
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, TextIO

from . import config

log = logging.getLogger(__name__)

EVENTS = ("user-prompt-submit", "pre-tool-use", "post-tool-use")
_ALIASES = {
    "userpromptsubmit": "user-prompt-submit", "user-prompt-submit": "user-prompt-submit",
    "user_prompt_submit": "user-prompt-submit",
    "pretooluse": "pre-tool-use", "pre-tool-use": "pre-tool-use", "pre_tool_use": "pre-tool-use",
    "posttooluse": "post-tool-use", "post-tool-use": "post-tool-use", "post_tool_use": "post-tool-use",
}

_KIND_FR = {
    "PERSON": "personne", "ORG": "société", "ADDRESS": "adresse", "EMAIL": "e-mail", "PHONE": "téléphone",
    "IBAN": "IBAN", "NIR": "numéro de sécurité sociale", "SIREN": "SIREN", "SIRET": "SIRET", "VAT": "TVA",
    "CARD": "carte bancaire", "CASE_REF": "référence de dossier", "ID_CODE": "identifiant", "URL": "site web",
    "BIRTHDATE": "date de naissance",
}

_FILE_TOOLS = {"Read", "NotebookRead", "Edit", "MultiEdit", "Write", "NotebookEdit", "LSP"}
_TREE_TOOLS = {"Grep", "Glob", "LS"}
_SHELL_TOOLS = {"Bash", "PowerShell", "Monitor"}
# Tools that send a local file somewhere (claude.ai, the user's devices) without the model reading it: PostToolUse
# cannot mask what they upload, so a file inside a dossier is denied before.
_UPLOAD_TOOLS = {"Artifact", "SendUserFile"}
_PATH_KEYS = ("file_path", "notebook_path", "path", "filePath", "uri")
# Shield's own MCP tools (user-level server or plugin server): their results already went through the canary.
_SHIELD_MCP_RE = re.compile(r"^mcp__(?:plugin_ragbase[-_]shield_)?ragbase[-_]shield__shield_[a-z_]+$")
# Hook budgets: past them the check would risk the hook timeout (Claude Code then keeps the raw result).
MAX_PROMPT_CHARS = 1_000_000
MAX_RESULT_CHARS = 2_000_000
TIME_BUDGET_S = 40.0


@dataclass
class HookResult:
    """What the hook prints on stdout (JSON object or nothing), its exit code and its stderr text."""

    output: Optional[dict] = None
    exit_code: int = 0
    stderr: str = ""


def _registry_exists() -> bool:
    return (config.home() / "registry.db").exists()


def _has_dossiers() -> bool:
    """True when the registry holds at least one dossier. Reads the row count only: no key, no vault opened,
    so a prompt or a tool result costs almost nothing when nothing is registered. Any doubt answers True."""
    path = config.home() / "registry.db"
    if not path.exists():
        return False
    try:
        conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5.0)
        try:
            return bool(conn.execute("SELECT EXISTS(SELECT 1 FROM dossiers)").fetchone()[0])
        finally:
            conn.close()
    except sqlite3.Error:
        return True  # cannot tell: do the full check (it fails closed if the registry is broken)


def _service():
    from . import service

    return service.default()


_REGISTRIES: dict[str, Any] = {}


def close_registries() -> None:
    """Close the registries the hooks keep open (``service.reset`` calls it). An open SQLite file cannot be
    deleted on Windows: ``verify`` removes its temporary home, tests remove their temporary folders."""
    for reg in list(_REGISTRIES.values()):
        try:
            reg.close()
        except Exception:  # noqa: BLE001
            pass
    _REGISTRIES.clear()


def _registry():
    """The vault registry alone (the PreToolUse hook needs the folders only, not the detection engine)."""
    home = str(config.home())
    reg = _REGISTRIES.get(home)
    if reg is None:
        from .vault import Registry

        for old in _REGISTRIES.values():
            try:
                old.close()
            except Exception:  # noqa: BLE001
                pass
        _REGISTRIES.clear()
        reg = _REGISTRIES[home] = Registry(home)
    return reg


def _journal(event: str, dossier: Optional[str] = None, **counts: Any) -> None:
    try:
        reg = _registry()
        reg.journal(event, dossier=dossier if dossier and dossier.startswith("DOSSIER_") else None, **counts)
    except Exception as exc:  # noqa: BLE001 - the journal must never break a hook
        log.debug("journal failed: %s", type(exc).__name__)


# ------------------------------------------------------------------------------------------ UserPromptSubmit

def _describe(hits: list[dict]) -> str:
    parts = []
    seen = set()
    for h in hits:
        kind = _KIND_FR.get(h.get("kind", ""), "donnée")
        key = (h.get("code"), h.get("dossier"), kind)
        if key in seen:
            continue
        seen.add(key)
        if h.get("code"):
            parts.append(f"{h['code']} ({kind}, {h['dossier']})")
        else:
            parts.append(f"une valeur enregistrée ({kind}, {h['dossier']})")
    return ", ".join(parts[:12]) + (" ..." if len(parts) > 12 else "")


def user_prompt_submit(payload: dict) -> HookResult:
    """Block a prompt that holds a registered value; the reason names the code to type instead."""
    prompt = payload.get("prompt", "")
    if not isinstance(prompt, str):
        prompt = json.dumps(prompt, ensure_ascii=False)
    if not prompt.strip() or not _has_dossiers():
        return HookResult()
    if len(prompt) > MAX_PROMPT_CHARS:
        reason = (f"RAGbase Shield : ce message est trop long pour être vérifié à temps ({len(prompt):,} caractères, "
                  f"{MAX_PROMPT_CHARS:,} au maximum) : il est bloqué par sécurité. Connectez le document dans Shield "
                  "Desk au lieu de le coller. (Prompt too long to check in time: blocked.)").replace(",", " ")
        return HookResult({"decision": "block", "reason": reason,
                           "hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                                                  "suppressOriginalPrompt": True}}, 2, reason)
    hits = _service().scan_text(prompt)
    if not hits:
        return HookResult()
    codes = [h["code"] for h in hits if h.get("code")]
    reason = ("RAGbase Shield a bloqué ce message : il contient une donnée client enregistrée. "
              f"Écrivez le code à la place : {_describe(hits)}. Le message n'a pas été envoyé. "
              "(RAGbase Shield blocked this message: it contains registered client data. "
              f"Type the code instead{': ' + ', '.join(dict.fromkeys(codes)) if codes else ''}. Nothing was sent.)")
    for d in dict.fromkeys(h["dossier"] for h in hits):
        _journal("hook_prompt_blocked", dossier=d, values=sum(1 for h in hits if h["dossier"] == d))
    out = {"decision": "block", "reason": reason,
           "hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "suppressOriginalPrompt": True}}
    return HookResult(out, 2, reason)


# ------------------------------------------------------------------------------------------ PreToolUse

_CASE_INSENSITIVE = sys.platform == "darwin" or sys.platform.startswith("win")
_GLOB_CHARS = re.compile(r"[*?\[{]")
_RECURSIVE_CMDS = {"rg", "ag", "ack", "find", "fd", "fdfind", "tree", "du", "tar", "rsync", "zip", "7z", "robocopy",
                   "xcopy", "grep-r", "locate", "mdfind", "rgrep"}
_LIST_CMDS = {"ls", "dir", "exa", "eza", "lsd", "gci", "get-childitem", "stat", "file", "vdir"} | _RECURSIVE_CMDS
# "/s" and "//s" (cmd.exe switches; Git Bash needs "//s"), PowerShell "-Recurse" and its short forms ("-rec")
_FLAG_RECURSIVE = re.compile(r"^(?:--recursive|--recurse|-rec(?:u(?:r(?:s(?:e)?)?)?)?|/{1,2}s)$", re.I)
_SHORT_R = re.compile(r"^-[A-Za-z]*[rR][A-Za-z]*$")
_SHORT_R_CMDS = {"grep", "egrep", "fgrep", "zgrep", "ls", "cp", "scp", "chmod", "chown", "rm", "zip", "diff",
                 "select-string", "sls", "wc", "cat", "gci", "get-childitem", "dir", "copy-item", "cpi",
                 "remove-item", "ri"}
# cmd.exe switches that walk a tree: "dir /s", "for /r", "forfiles /s", "where /r", "xcopy /s" (cmd syntax only)
_CMD_RECURSIVE = re.compile(r"^/{1,2}[rs]$", re.I)
_CMD_SWITCH = re.compile(r"^/{1,2}[A-Za-z?](?::.*)?$")  # "/s", "/a:-d", '/c:"a b"' (value after the colon)
_PRIVATE_CLI = re.compile(
    r"(?i)\bragbase[-_]shield\b(?:\s+--?\S+)*\s+(?:list|entities|restore|export|import|journal|desk|keep-clear"
    r"|keep_clear|register|add)\b"
    r"|(?:-m\s+|\bimport\s+|\bfrom\s+)ragbase_shield\b"
    r"|\bragbase_shield\.(?:service|vault|cli|desk|engine)\b")


def _norm(p: str) -> str:
    s = unicodedata.normalize("NFC", p.replace("\\", "/"))
    s = re.sub(r"/+", "/", s)
    if len(s) > 1:
        s = s.rstrip("/")
    return s.casefold() if _CASE_INSENSITIVE else s


_PS_ENV_RE = re.compile(r"\$\{?env:([A-Za-z_][A-Za-z0-9_]*)\}?", re.I)
_HOME_VAR_RE = re.compile(r"\$\{HOME\}|\$HOME(?![A-Za-z0-9_])|%USERPROFILE%|%HOMEPATH%", re.I)


def _env_value(name: str) -> Optional[str]:
    if name in os.environ:
        return os.environ[name]
    low = name.lower()
    return next((v for k, v in os.environ.items() if k.lower() == low), None)


def _expand(path: str) -> str:
    """``~``, ``$VAR`` / ``%VAR%`` (the rules of this machine), PowerShell ``$env:VAR``, and the home forms
    ``$HOME`` / ``%USERPROFILE%`` that the shell knows even when this process has no such variable."""
    p = _PS_ENV_RE.sub(lambda m: _env_value(m.group(1)) or m.group(0), path)
    p = os.path.expanduser(os.path.expandvars(p))
    return _HOME_VAR_RE.sub(lambda m: str(Path.home()), p)


def _canon(path: os.PathLike | str, base: Optional[str] = None) -> str:
    p = _expand(str(path))
    if base and not os.path.isabs(p):
        p = os.path.join(base, p)
    try:
        p = os.path.realpath(p)
    except (OSError, ValueError):
        p = os.path.abspath(p)
    return _norm(p)


def _is_within(child: str, parent: str) -> bool:
    return child == parent or child.startswith(parent.rstrip("/") + "/")


class _PathGuard:
    """Registered dossier folders and the Shield data folder, compared after resolving symlinks."""

    def __init__(self, folders: dict[str, Path], shield_home: Path):
        self.items: list[tuple[str, str, list[str]]] = []
        for code, folder in folders.items():
            self.items.append((code, _canon(folder), self._spellings(folder)))
        self.items.append(("SHIELD_HOME", _canon(shield_home), self._spellings(shield_home)))

    @staticmethod
    def _spellings(folder: Path) -> list[str]:
        """Literal forms a command may use for this folder (absolute, ~, $HOME, escaped spaces)."""
        out = set()
        for raw in {str(folder), os.path.realpath(str(folder))}:
            forms = {raw, raw.replace("\\", "/")}
            home = str(Path.home())
            for f in list(forms):
                if f.startswith(home):
                    rest = f[len(home):]
                    forms |= {"~" + rest, "$HOME" + rest, "${HOME}" + rest, "%USERPROFILE%" + rest,
                              "$env:USERPROFILE" + rest}
            for f in list(forms):
                forms.add(f.replace(" ", "\\ "))
            out |= {_norm(f) for f in forms if len(f) > 3}
        return sorted(out, key=len, reverse=True)

    def inside(self, canon: str) -> Optional[str]:
        best = None
        for code, folder, _sp in self.items:
            if _is_within(canon, folder) and (best is None or len(folder) > len(best[1])):
                best = (code, folder)
        return best[0] if best else None

    def contains(self, canon: str, direct_only: bool = False) -> Optional[str]:
        """Code of a guarded folder below ``canon`` (``canon`` is one of its parents)."""
        for code, folder, _sp in self.items:
            if folder != canon and _is_within(folder, canon):
                if not direct_only or os.path.dirname(folder) == canon:
                    return code
        return None

    def literal(self, command: str) -> Optional[str]:
        low = _norm(command) if _CASE_INSENSITIVE else command.replace("\\", "/")
        low = unicodedata.normalize("NFC", low)
        for code, _folder, spellings in self.items:
            for sp in spellings:
                if sp in low:
                    return code
        return None


def _shell_tokens(command: str, shell: str | bool = "sh") -> list[list[str]]:
    """Split a command line into simple commands (lists of words), best effort. ``shell``: "sh" (POSIX shells,
    Git Bash), "powershell" or "cmd" (``True`` = "powershell", the former flag)."""
    if shell is True or shell in ("powershell", "cmd"):
        return _windows_words(command, ps=shell is True or shell == "powershell")
    try:
        lex = shlex.shlex(command, posix=True, punctuation_chars=";&|()<>")
        lex.whitespace_split = True
        lex.commenters = ""
        words = list(lex)
    except ValueError:
        words = command.split()
    cmds: list[list[str]] = [[]]
    for w in words:
        if w and all(c in ";&|()<>" for c in w):
            cmds.append([])
            continue
        cmds[-1].append(w)
    return [c for c in cmds if c]


def _windows_words(command: str, ps: bool) -> list[list[str]]:
    """cmd.exe (``ps`` False) or PowerShell words, best effort. cmd: "..." quotes, ``^`` escapes, a backslash is
    a plain character. PowerShell: '...' (``''`` inside) and "..." quotes, the backtick escapes (also inside
    "..."), so ``Dossier` Exemple`` is one word. Separators: ``& | ( ) < >``, and ``;`` in PowerShell, where a
    script block ``{ ... }`` opens a new command list: ``{`` at the start of a word and ``}`` at the end of a word
    are separators; inside a word they are plain characters (``a{b}c`` is one word, checked as a glob). The
    opening ``{`` is also kept as a word of the command it closes, so a script block run from a folder above a
    dossier is checked like a glob of that folder (what the block does to each item is not followed)."""
    esc, seps = ("`", ";&|()<>") if ps else ("^", "&|()<>")
    cmds: list[list[str]] = [[]]
    word: list[str] = []
    started = False
    quote = ""
    i, n = 0, len(command)

    def end_word() -> None:
        nonlocal word, started
        if started:
            cmds[-1].append("".join(word))
        word, started = [], False

    while i < n:
        ch = command[i]
        if quote:
            if ch == quote:
                if ps and quote == "'" and command[i + 1:i + 2] == "'":
                    word.append("'")
                    i += 2
                    continue
                quote = ""
            elif ps and quote == '"' and ch == esc and i + 1 < n:
                word.append(command[i + 1])
                i += 1
            else:
                word.append(ch)
        elif ch == esc and i + 1 < n:
            word.append(command[i + 1])
            started = True
            i += 1
        elif ps and ch == "$" and command[i + 1:i + 2] == "{" and "}" in command[i + 2:]:
            close = command.index("}", i + 2)  # ${env:NAME}, ${name}: one variable, not a script block
            word.append(command[i:close + 1])
            started = True
            i = close
        elif ch == '"' or (ps and ch == "'"):
            quote, started = ch, True
        elif ch.isspace():
            end_word()
        elif ch in seps or (ps and ((ch == "{" and not started) or
                                     (ch == "}" and (i + 1 == n or command[i + 1].isspace() or
                                                     command[i + 1] in seps)))):
            end_word()
            if ch == "{":
                cmds[-1].append("{")
            if cmds[-1]:
                cmds.append([])
        else:
            word.append(ch)
            started = True
        i += 1
    end_word()
    return [c for c in cmds if c]


# Git Bash on Windows: a backslash between path parts is a separator for the user ("C:\Users\x"), but POSIX
# parsing drops it ("C:Usersx"). "\\", "\ " and "\" before a quote or a shell character stay escapes.
_BACKSLASH_SEP = re.compile(r"\\\\|\\(?=[\w.%@+,=~-])")
# Git Bash, Cygwin and WSL name drive C: "/c/...", "/cygdrive/c/...", "/mnt/c/..."
_POSIX_DRIVE = re.compile(r"(?<![\w.~$%:/\\-])/(?:cygdrive/|mnt/)?([A-Za-z])(?=/)")
_WINDOWS = os.name == "nt"  # the hook runs on Windows: Bash commands are also read with Windows path rules


def _windows_reading(command: str) -> str:
    """The same Bash command read with the path rules of Git Bash on Windows (``C:\\x\\y`` -> ``C:/x/y``,
    ``/c/x`` -> ``c:/x``). The check runs on both readings: a command denied by either one is denied."""
    s = _BACKSLASH_SEP.sub(lambda m: m.group(0) if len(m.group(0)) == 2 else "/", command)
    return _POSIX_DRIVE.sub(lambda m: m.group(1) + ":", s)


_SH_SHELLS = {"sh", "bash", "zsh", "dash", "ksh", "mksh", "ash", "fish"}
_PS_SHELLS = {"powershell", "pwsh"}
_SH_INLINE = re.compile(r"^-[A-Za-z]*c[A-Za-z]*$")  # bash -c, -lc, -ec...
_PS_INLINE = re.compile(r"^[-/](?:c|co|com|comm|comma|comman|command)$", re.I)
_PS_ENCODED = re.compile(r"^[-/](?:e|ec|en|enc|enco|encod|encode|encoded|encodedc\w*)$", re.I)
_CMD_INLINE = re.compile(r"^/{1,2}([ck])(.*)$", re.I)


def _word_name(word: str) -> str:
    name = os.path.basename(word.replace("\\", "/")).lower()
    return name[:-4] if name.endswith(".exe") else name


def _ps_decode(text: str) -> Optional[str]:
    """A PowerShell -EncodedCommand argument (base64 of UTF-16LE), or None."""
    import base64
    import binascii

    try:
        return base64.b64decode(text + "=" * (-len(text) % 4)).decode("utf-16-le")
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return None


def _nested(words: list[str]) -> list[tuple[str, str]]:
    """Command lines run by a shell named in ``words`` (``bash -c "..."``, ``cmd /c ...``, ``powershell -Command
    ...``, ``-EncodedCommand``, ``eval``, ``Invoke-Expression``), with the syntax to read them."""
    out: list[tuple[str, str]] = []
    for i, w in enumerate(words):
        name = _word_name(w)
        rest = words[i + 1:]
        if name == "eval" and rest:
            out.append((" ".join(rest), "sh"))
        elif name in ("iex", "invoke-expression") and rest:
            out.append((" ".join(a for a in rest if not a.lower().startswith("-command")), "powershell"))
        elif name in _SH_SHELLS:
            for j, a in enumerate(rest):
                if _SH_INLINE.match(a) and j + 1 < len(rest):
                    out.append((rest[j + 1], "sh"))
                    break
                if not a.startswith("-"):
                    break
        elif name == "cmd":
            for j, a in enumerate(rest):
                m = _CMD_INLINE.match(a)
                if m:
                    out.append((" ".join(([m.group(2)] if m.group(2) else []) + rest[j + 1:]), "cmd"))
                    break
                if not a.startswith("/"):
                    break
        elif name in _PS_SHELLS:
            for j, a in enumerate(rest):
                if _PS_INLINE.match(a):
                    out.append((" ".join(rest[j + 1:]), "powershell"))
                    break
                if _PS_ENCODED.match(a) and j + 1 < len(rest):
                    decoded = _ps_decode(rest[j + 1])
                    if decoded:
                        out.append((decoded, "powershell"))
                    break
                if a.lower() in ("-file", "-f", "/file"):
                    break
        else:
            continue
        break
    return out


def _cmd_name(words: list[str]) -> tuple[str, list[str]]:
    """(command name, words from the command word on) after env assignments and wrappers like sudo."""
    i = 0
    while i < len(words) and (re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", words[i]) or words[i] in
                              ("sudo", "env", "command", "builtin", "exec", "nohup", "time", "xargs", "nice")):
        i += 1
    if i >= len(words):
        return "", []
    name = os.path.basename(words[i].replace("\\", "/")).lower()
    if name.endswith(".exe"):
        name = name[:-4]
    return name, words[i:]


def _static_prefix(token: str) -> str:
    m = _GLOB_CHARS.search(token)
    if not m:
        return token
    head = token[:m.start()]
    return head if head.endswith(("/", "\\")) else os.path.dirname(head)


def check_command(command: str, cwd: str, guard: _PathGuard, *, powershell: bool = False,
                  desk_port: int = config.DESK_PORT) -> Optional[str]:
    """Code of the guarded place a shell command touches (DOSSIER_n, SHIELD_HOME, DESK, CLI) or None.

    ``powershell``: the command is PowerShell (the PowerShell tool); otherwise it is a POSIX shell command (Bash,
    Git Bash on Windows). Command lines run by a nested shell (``bash -c``, ``cmd /c``, ``powershell -Command``)
    are checked too, with their own syntax."""
    return _check_shell(command, cwd, guard, "powershell" if powershell else "sh", desk_port, 0)


_MAX_NESTING = 3
_MAX_BASES = 16  # folders a shell may be in after its cd commands; past this the command is denied (fail closed)
_CD_CMDS = {"cd", "pushd", "set-location", "sl", "chdir"}


def _check_shell(command: str, cwd: str, guard: _PathGuard, shell: str, desk_port: int,
                 depth: int) -> Optional[str]:
    if not command or not command.strip():
        return None
    hit = _check_line(command, cwd, guard, shell, desk_port, depth)
    if hit is None and shell == "sh" and _WINDOWS:
        win = _windows_reading(command)
        if win != command:
            hit = _check_line(win, cwd, guard, shell, desk_port, depth)
    return hit


def _check_line(command: str, cwd: str, guard: _PathGuard, shell: str, desk_port: int,
                depth: int) -> Optional[str]:
    if re.search(rf"(?i)(?:127\.0\.0\.1|localhost|\[::1\]|0\.0\.0\.0)\s*:\s*{int(desk_port)}\b", command):
        return "DESK"
    if _PRIVATE_CLI.search(command):
        return "CLI"
    hit = guard.literal(command)
    if hit:
        return hit
    base = _canon(cwd) if cwd else _canon(os.getcwd())
    hit = guard.inside(base)
    if hit:
        return hit
    bases = [base]  # every folder the shell may be in: a cd that fails leaves it where it was
    for words in _shell_tokens(command, shell):
        if shell == "powershell" and len(words) > 2 and words[0].startswith("$") and words[1].lower() == "in":
            words = words[2:]  # foreach ($item in <command>): the command after "in"
        name, rest = _cmd_name(words)
        args = rest[1:]
        if name in _CD_CMDS:
            targets = [a if not a.startswith("-") else _OPTION_VALUE.match(a).group(1) for a in args
                       if not a.startswith("-") or _OPTION_VALUE.match(a)]  # Set-Location -Path:<folder>
            same_drive_only = False
            if shell == "cmd":
                if targets and targets[0].lower() in ("/d", "//d"):  # cmd.exe only: "cd /d <folder>" (in a
                    targets = targets[1:]  # POSIX shell "cd /d x" fails and the shell stays where it was)
                else:
                    same_drive_only = True  # cmd.exe without /d does not change the drive
                if len(targets) > 1:  # cmd.exe: "cd C:\Dossier Exemple" takes the rest of the line, spaces too
                    targets = [" ".join(targets)]
            moved = [_canon(targets[0] if targets else "~", b) for b in bases]
            for b in moved:
                hit = guard.inside(b)
                if hit:
                    return hit
            kept: list[str] = []
            for b, m in zip(bases, moved):
                kept.append(m)
                # the shell may still be in the old place: the cd fails ("cd /no/such ; cat Clients/x"), the
                # folder is made later, the reading differs (several targets; no target = home in bash, no move
                # in cmd), or cmd.exe without /d is given another drive. Every folder it may be in is checked.
                if len(targets) != 1 or not os.path.isdir(m) or \
                        (same_drive_only and os.path.splitdrive(m)[0] != os.path.splitdrive(b)[0]):
                    kept.append(b)
            bases = list(dict.fromkeys(kept))
            if len(bases) > _MAX_BASES:
                return "SHELL"  # fail closed: never drop a folder the shell may be in
            continue
        if depth < _MAX_NESTING:
            for inner, inner_shell in _nested(words):
                for b in bases:
                    hit = _check_shell(inner, b, guard, inner_shell, desk_port, depth + 1)
                    if hit:
                        return hit
        for b in bases:
            hit = _check_words(name, rest, b, guard, shell)
            if hit:
                return hit
    return None


_OPTION_VALUE = re.compile(r"^--?[A-Za-z][\w-]*[=:](.+)$")  # --file=<path>, PowerShell -Path:<path>
# an option value that names a place: "~", ".", "$VAR", "%VAR%", a drive letter, or a separator inside; a plain
# word or a file filter ("--include=*.txt", "--type-add=x:*.y") is not one
_PLACE_VALUE = re.compile(r"^(?:[~.$%]|[A-Za-z]:[\\/]|.*[\\/])")


def _check_words(name: str, rest: list[str], base: str, guard: _PathGuard, shell: str) -> Optional[str]:
    """One simple command (``rest`` = the command word and its arguments) run in ``base``.

    The value of an option (``--exclude-dir=<v>``, PowerShell ``-Path:<v>``, cmd.exe ``/g:<v>``) is checked like
    a path, but it is never the path argument of the command: ``grep -r --exclude=/none <word>`` and
    ``ls --hide=/zz`` still read the current folder, so a recursive or listing command with no positional path
    is checked against the current folder whatever its options hold."""
    args = rest[1:]
    recursive = name in _RECURSIVE_CMDS or any(_FLAG_RECURSIVE.match(a) for a in args) or \
        (name in _SHORT_R_CMDS and any(_SHORT_R.match(a) for a in args)) or \
        (name == "git" and args[:1] and args[0] in ("grep", "ls-files", "archive", "log", "show", "diff")) or \
        (shell == "cmd" and any(_CMD_RECURSIVE.match(a) for a in args))
    listing = name in _LIST_CMDS
    paths = 0
    for pos, a in enumerate(rest):
        positional = pos > 0  # the command word itself is checked as a path too, but it is not an argument
        if pos > 0 and shell == "cmd" and _CMD_SWITCH.match(a):
            a = a.split(":", 1)[1] if ":" in a else ""  # a cmd.exe switch: "/s", "/b" (no value), "/g:<file>"
            positional = False
        elif a.startswith("-") and len(a) > 1 and not a.startswith("-/"):
            m = _OPTION_VALUE.match(a)
            a = m.group(1) if m else ""
            positional = False
        if not a or (pos > 0 and not positional and not _PLACE_VALUE.match(a)):
            continue
        is_glob = bool(_GLOB_CHARS.search(a))
        prefix = _static_prefix(a) if is_glob else a
        canon = _canon(prefix or ".", base)
        hit = guard.inside(canon)
        if hit:
            return hit
        if is_glob or recursive:
            hit = guard.contains(canon)
        elif listing:
            hit = guard.contains(canon, direct_only=True)
        if hit:
            return hit
        if positional and (os.sep in a or "/" in a or a.startswith((".", "~", "$"))):
            paths += 1
    if (recursive or listing) and paths == 0:
        hit = guard.contains(base) if recursive else guard.contains(base, direct_only=True)
        if hit:
            return hit
    return None


def _deny_reason(code: str) -> str:
    if code == "SHIELD_HOME":
        return ("RAGbase Shield : accès refusé aux données de Shield (coffre chiffré et clé). "
                "(RAGbase Shield: access to Shield data is denied.)")
    if code == "DESK":
        return ("RAGbase Shield : accès refusé à Shield Desk, page locale réservée à l'utilisateur. "
                "(RAGbase Shield: Shield Desk is for the user only.)")
    if code == "SHELL":
        return ("RAGbase Shield : commande trop complexe pour être vérifiée (trop de changements de dossier) ; "
                "exécutez-la en plusieurs commandes plus simples. "
                "(RAGbase Shield: this command is too complex to check (too many folder changes); run it as "
                "several simpler commands.)")
    if code == "CLI":
        return ("RAGbase Shield : cette commande affiche des données réelles ; elle est réservée à l'utilisateur, "
                "dans son propre terminal. Utilisez les outils shield_*. "
                "(RAGbase Shield: this command shows real data; it is for the user only. Use the shield_* tools.)")
    return (f"RAGbase Shield : lecture directe refusée dans le dossier client {code}. Utilisez "
            f"shield_list_documents puis shield_read_document (ou shield_search) avec le code {code}. "
            "Ne contournez pas ce blocage. "
            f"(RAGbase Shield: direct access to client dossier {code} is denied. Use shield_list_documents and "
            f"shield_read_document or shield_search with {code}. Do not work around this block.)")


def _guard() -> _PathGuard:
    return _PathGuard(_registry().folders(), config.home())


_PATHLIKE_RE = re.compile(r"^(?:file://|~|/|\.{1,2}[\\/]|[A-Za-z]:[\\/]|\\\\|\$HOME|%USERPROFILE%)")
_RECURSIVE_TOOL_RE = re.compile(r"(?i)search|find|tree|grep|glob|walk|recursive|zip|archive|copy|move|sync")


def _path_strings(obj: Any, out: list[str], depth: int = 0) -> None:
    """String values of a tool input that look like file paths (absolute, ~, ./, file://, C:\\...)."""
    if depth > 6 or len(out) > 200:
        return
    if isinstance(obj, str):
        v = obj.strip()
        if 0 < len(v) < 4096 and "\n" not in v and (_PATHLIKE_RE.match(v) or (("/" in v or "\\" in v)
                                                                            and v.count(" ") <= 3)):
            out.append(_uri_path(v))
    elif isinstance(obj, dict):
        for v in obj.values():
            _path_strings(v, out, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _path_strings(v, out, depth + 1)


def _check_paths(paths: list[str], cwd: str, guard: _PathGuard, recursive: bool) -> Optional[str]:
    for raw in paths:
        canon = _canon(raw, cwd)
        hit = guard.inside(canon) or guard.contains(canon, direct_only=not recursive)
        if hit:
            return hit
    return None


def _unquote(s: str) -> str:
    """Percent-decoding of a file URI path (UTF-8), without urllib."""
    if "%" not in s:
        return s
    out = bytearray()
    i = 0
    raw = s.encode("utf-8")
    while i < len(raw):
        if raw[i] == 0x25 and i + 2 < len(raw) and re.fullmatch(rb"[0-9A-Fa-f]{2}", raw[i + 1:i + 3]):
            out.append(int(raw[i + 1:i + 3], 16))
            i += 3
        else:
            out.append(raw[i])
            i += 1
    return out.decode("utf-8", errors="replace")


def _uri_path(value: str) -> str:
    """``file:///Users/x/a%20b.txt`` -> ``/Users/x/a b.txt`` (``file:///C:/x`` -> ``C:/x``); other text as is."""
    if not value.startswith("file://"):
        return value
    rest = value[7:]
    if re.match(r"^[A-Za-z](?::|%3[Aa])[\\/]", rest):  # file://C:\x or file://C:/x (a Windows path, no host)
        rest = "/" + rest
    if not rest.startswith("/"):  # file://host/path
        rest = "/" + rest.split("/", 1)[1] if "/" in rest else "/"
    rest = rest.split("?", 1)[0].split("#", 1)[0]
    v = _unquote(rest)
    return v[1:] if re.match(r"^/[A-Za-z]:[\\/]", v) else v


def pre_tool_use(payload: dict) -> HookResult:
    """Deny direct reads inside a registered dossier folder (and of Shield's own data)."""
    tool = str(payload.get("tool_name") or "")
    ti = payload.get("tool_input") or {}
    if not isinstance(ti, dict):
        ti = {}
    cwd = str(payload.get("cwd") or os.getcwd())
    is_other_mcp = tool.startswith("mcp__") and not _SHIELD_MCP_RE.match(tool)
    if tool not in _FILE_TOOLS | _TREE_TOOLS | _SHELL_TOOLS | _UPLOAD_TOOLS and not is_other_mcp:
        return HookResult()
    if not _registry_exists():
        guard = _PathGuard({}, config.home())
    else:
        guard = _guard()
    hit: Optional[str] = None
    if tool in _FILE_TOOLS:
        for key in _PATH_KEYS:
            if isinstance(ti.get(key), str) and ti[key]:
                hit = guard.inside(_canon(_uri_path(ti[key]), cwd))
                break
        if not hit and tool == "LSP":
            paths: list[str] = []
            _path_strings(ti, paths)
            hit = _check_paths(paths, cwd, guard, recursive=False)
    elif tool in _UPLOAD_TOOLS or is_other_mcp:
        paths = []
        _path_strings(ti, paths)
        if isinstance(ti.get("files"), dict):  # Artifact "files": {published path: source path}
            _path_strings(list(ti["files"].values()), paths)
        hit = _check_paths(paths, cwd, guard, recursive=bool(_RECURSIVE_TOOL_RE.search(tool)))
    elif tool in _TREE_TOOLS:
        root = ti.get("path") if isinstance(ti.get("path"), str) and ti.get("path") else cwd
        canon = _canon(root, cwd)
        hit = guard.inside(canon) or guard.contains(canon)
        pattern = ti.get("pattern") if tool == "Glob" else ti.get("glob")
        if not hit and isinstance(pattern, str) and (os.path.isabs(pattern) or pattern.startswith("~")):
            canon = _canon(_static_prefix(pattern) or "/", cwd)
            hit = guard.inside(canon) or guard.contains(canon)
    else:
        command = ti.get("command")
        if isinstance(command, str):
            hit = check_command(command, cwd, guard, powershell=tool == "PowerShell")
        elif tool == "Monitor":  # unknown input shape: check every path-like value
            paths = []
            _path_strings(ti, paths)
            hit = _check_paths(paths, cwd, guard, recursive=True)
    if not hit:
        return HookResult()
    reason = _deny_reason(hit)
    _journal("hook_read_denied", dossier=hit)
    out = {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                  "permissionDecisionReason": reason}}
    return HookResult(out, 2, reason)


# ------------------------------------------------------------------------------------------ PostToolUse

def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _number_forms(v: Any) -> list[str]:
    """Text forms of a JSON number to check: "388684722", and with the leading zero a number loses
    ("0612345678" sent as 612345678)."""
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            return []
        if not v.is_integer():
            return [repr(v)]
        v = int(v)
    s = str(abs(v))
    return [s, "0" + s] if len(s) >= 8 else [s]


def _leaves(obj: Any, out: list[str]) -> None:
    """String leaves, dict keys and the text forms of number leaves (booleans and null are skipped)."""
    if isinstance(obj, str):
        out.append(obj)
    elif _is_number(obj):
        out.extend(_number_forms(obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str):
                out.append(k)
            _leaves(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _leaves(v, out)


def _map_leaves(obj: Any, fn: Callable[[str], str], fnum: Optional[Callable[[Any], Any]] = None) -> Any:
    """Same shape as ``obj`` with ``fn`` applied to every string (keys included) and ``fnum`` to numbers."""
    if isinstance(obj, str):
        return fn(obj)
    if _is_number(obj):
        return fnum(obj) if fnum is not None else obj
    if isinstance(obj, dict):
        return {(fn(k) if isinstance(k, str) else k): _map_leaves(v, fn, fnum) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_map_leaves(v, fn, fnum) for v in obj]
    return obj


def _map_strings(obj: Any, fn: Callable[[str], str]) -> Any:
    return _map_leaves(obj, fn)


class _OverBudget(Exception):
    """The check would not finish before the hook timeout: fail closed."""


def post_tool_use(payload: dict) -> HookResult:
    """Replace a tool result that still holds a registered value with its masked form."""
    if "tool_response" in payload:
        resp = payload["tool_response"]
    else:
        resp = payload.get("tool_result")
    tool = str(payload.get("tool_name") or "")
    if resp is None or _SHIELD_MCP_RE.match(tool) or not _has_dossiers():
        return HookResult()
    leaves: list[str] = []
    _leaves(resp, leaves)
    if not any(s.strip() for s in leaves):
        return HookResult()
    if sum(len(s) for s in leaves) > MAX_RESULT_CHARS:
        raise _OverBudget()
    started = time.monotonic()
    svc = _service()
    added: list[str] = []

    def mask(s: str) -> str:
        if not s.strip():
            return s
        if time.monotonic() - started > TIME_BUDGET_S:
            raise _OverBudget()
        new, codes = svc.protect(s)
        for c in codes:
            if c not in added:
                added.append(c)
        return new

    def mask_number(v: Any) -> Any:
        for form in _number_forms(v):
            new, codes = svc.protect(form)
            if new != form:
                for c in codes:
                    if c not in added:
                        added.append(c)
                return codes[0] if len(codes) == 1 and new.strip() == codes[0] else new
        return v

    new_resp = _map_leaves(resp, mask, mask_number)
    if new_resp == resp:
        return HookResult()
    listed = ", ".join(added[:12]) + (" ..." if len(added) > 12 else "")
    context = ("RAGbase Shield a masqué des données client dans ce résultat d'outil"
               + (f" ({listed})" if listed else "") + ". Travaillez avec les codes ; ne cherchez pas les vraies "
               "valeurs. (RAGbase Shield masked client data in this tool result. Work with the codes.)")
    hso: dict[str, Any] = {"hookEventName": "PostToolUse", "updatedToolOutput": new_resp,
                           "additionalContext": context}
    if tool.startswith("mcp__"):
        hso["updatedMCPToolOutput"] = new_resp
    _journal("hook_output_masked", values=len(added))
    out = {"hookSpecificOutput": hso,
           "systemMessage": "RAGbase Shield : données client masquées dans un résultat d'outil"
                            + (f" ({listed})." if listed else ".")}
    return HookResult(out, 0, "")


# ------------------------------------------------------------------------------------------ fail closed

_REMOVED = "[RAGbase Shield : contenu retiré]"
# Field names of tool results (Claude Code built-in tools, MCP content blocks): kept so the shape stays the same.
# Any other key may be a value ({"Kervadec": 3}) and is replaced by "champ_n".
_STRUCTURAL_KEYS = frozenset({
    "stdout", "stderr", "interrupted", "isImage", "returnCodeInterpretation", "noOutputExpected", "type", "text",
    "file", "filePath", "content", "numLines", "startLine", "totalLines", "filenames", "numFiles", "durationMs",
    "truncated", "mode", "matches", "numMatches", "structuredPatch", "oldString", "newString", "originalFile",
    "userModified", "replaceAll", "data", "mimeType", "image", "resource", "uri", "name", "isError", "result",
    "results", "output", "error", "url", "code", "bytes", "status", "statusText", "headers", "title", "path",
    "items", "value", "id", "message", "messages", "lines", "count", "total", "annotations", "structuredContent",
    "meta", "_meta", "blob", "query", "answer", "edits", "patch", "diff", "summary", "description", "size",
})


def _blank(obj: Any) -> Any:
    """Fail-closed copy of a tool result with the SAME shape: every string becomes a removal notice, every
    number 0; booleans and null stay. Known field names stay ("stdout", "filePath"); any other key (it may be a
    value) is replaced by "champ_n"."""
    keys = iter(range(1, 10 ** 6))

    def blank(o: Any) -> Any:
        if isinstance(o, str):
            return _REMOVED
        if _is_number(o):
            return 0
        if isinstance(o, dict):
            return {(k if not isinstance(k, str) or k in _STRUCTURAL_KEYS else f"champ_{next(keys)}"): blank(v)
                    for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [blank(v) for v in o]
        return o

    return blank(obj)


def _fail_closed(event: str, payload: Optional[dict]) -> HookResult:
    if event == "user-prompt-submit":
        reason = ("RAGbase Shield ne peut pas vérifier ce message (erreur interne) : il est bloqué par sécurité. "
                  "Lancez « ragbase-shield doctor ». (RAGbase Shield could not check this prompt: blocked.)")
        return HookResult({"decision": "block", "reason": reason,
                           "hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                                                  "suppressOriginalPrompt": True}}, 2, reason)
    if event == "pre-tool-use":
        reason = ("RAGbase Shield ne peut pas vérifier cet accès (erreur interne) : refusé par sécurité. "
                  "Lancez « ragbase-shield doctor ». (RAGbase Shield could not check this access: denied.)")
        return HookResult({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                                  "permissionDecisionReason": reason}}, 2, reason)
    reason = ("RAGbase Shield ne peut pas vérifier ce résultat d'outil (erreur interne ou résultat trop volumineux) : "
              "il est retiré et la session est arrêtée. Lancez « ragbase-shield doctor ». "
              "(RAGbase Shield could not check this tool result: removed, session stopped.)")
    out: dict[str, Any] = {"continue": False, "stopReason": reason}
    resp = (payload or {}).get("tool_response", (payload or {}).get("tool_result"))
    if resp is not None:
        out["hookSpecificOutput"] = {"hookEventName": "PostToolUse", "updatedToolOutput": _blank(resp)}
        if str((payload or {}).get("tool_name") or "").startswith("mcp__"):
            out["hookSpecificOutput"]["updatedMCPToolOutput"] = _blank(resp)
    return HookResult(out, 0, reason)


_HANDLERS: dict[str, Callable[[dict], HookResult]] = {
    "user-prompt-submit": user_prompt_submit,
    "pre-tool-use": pre_tool_use,
    "post-tool-use": post_tool_use,
}


def event_name(event: str) -> Optional[str]:
    return _ALIASES.get(str(event).strip().lower().replace(" ", ""))


def handle(event: str, payload: Any) -> HookResult:
    """Run one hook. Never raises; a failure inside Shield gives the fail-closed answer for the event."""
    name = event_name(event)
    if name is None:
        return HookResult(None, 1, "RAGbase Shield : événement de hook inconnu (unknown hook event).")
    if not isinstance(payload, dict):
        return _fail_closed(name, None)
    try:
        return _HANDLERS[name](payload)
    except _OverBudget:
        log.warning("RAGbase Shield hook %s: too large to check in time; failing closed", name)
        return _fail_closed(name, payload)
    except Exception as exc:  # noqa: BLE001 - fail closed, log the type only
        log.warning("RAGbase Shield hook %s failed: %s", name, type(exc).__name__)
        return _fail_closed(name, payload)


def main(event: str, stdin: Optional[TextIO] = None, stdout: Optional[TextIO] = None,
         stderr: Optional[TextIO] = None) -> int:
    """Read the hook JSON from stdin, print the answer JSON on stdout, return the exit code."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    try:
        raw = stdin.buffer.read().decode("utf-8", errors="replace") if hasattr(stdin, "buffer") else stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except (ValueError, OSError):
        payload = None
    result = handle(event, payload)
    if result.output is not None:
        stdout.write(json.dumps(result.output, ensure_ascii=True))
        stdout.write("\n")
        stdout.flush()
    if result.stderr:
        stderr.write(result.stderr + "\n")
        stderr.flush()
    return result.exit_code


__all__ = ["EVENTS", "HookResult", "handle", "main", "user_prompt_submit", "pre_tool_use", "post_tool_use",
           "check_command", "event_name"]
