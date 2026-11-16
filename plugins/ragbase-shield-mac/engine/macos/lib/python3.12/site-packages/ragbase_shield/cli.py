"""``ragbase-shield`` command line (contract: ARCHITECTURE.md section 8).

The terminal belongs to the user, so some commands print real values (``list`` titles, ``entities``,
``restore``). When the command runs inside an AI agent's shell (Claude Code sets ``CLAUDECODE``, Codex sets
``CODEX_SANDBOX``), whatever it prints goes to the model: those commands then refuse, or show codes only.
"""
from __future__ import annotations

import argparse
import contextlib
import getpass
import io
import json
import logging
import os
import platform
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from . import __version__, config

log = logging.getLogger(__name__)

APPS = ("claude-desktop", "claude-code", "chatgpt", "codex", "vscode")
SERVER_KEY = "ragbase-shield"
PLUGIN_ID = "ragbase-shield@ragbase"
_AGENT_ENV = ("CLAUDECODE", "CODEX_SANDBOX", "CODEX_SANDBOX_NETWORK_DISABLED", "RAGBASE_SHIELD_AGENT")
_KIND_BY_OPTION = {"person": "PERSON", "org": "ORG"}


# ------------------------------------------------------------------------------------------ helpers

class UsageError(Exception):
    """A user mistake (missing file, unsupported input). Exit code 2, message without client data."""


def agent_shell() -> Optional[str]:
    """Name of the agent environment variable when this command runs inside an AI agent's shell."""
    for name in _AGENT_ENV:
        v = os.environ.get(name, "").strip()
        if v and v.lower() not in ("0", "false", "no"):
            return name
    return None


def _refuse_in_agent(what: str, changes: bool = False) -> int:
    if changes:
        print(f"RAGbase Shield : « {what} » modifie le coffre (ce qui est masqué). Seul l'utilisateur peut le "
              "faire : lancez cette commande vous-même dans votre terminal, ou utilisez Shield Desk. Un assistant IA "
              "ne peut pas changer le masquage. (This command changes the vault: only the user can run it, not an "
              "AI agent.)", file=sys.stderr)
    else:
        print(f"RAGbase Shield : « {what} » affiche des données réelles. Lancez cette commande vous-même dans votre "
              "terminal, pas depuis un assistant IA. (This command shows real data: run it yourself, not from an "
              "AI agent.)", file=sys.stderr)
    return 3


def _utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError, OSError):
            pass


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


def _svc():
    from . import service

    return service.default()


def _read_input(file_arg: Optional[str], use_clipboard: bool) -> str:
    """Text from FILE (any supported document), from the clipboard, or from stdin."""
    from . import documents

    if file_arg and file_arg != "-":
        p = Path(file_arg).expanduser()
        if not p.is_file():
            raise UsageError("Fichier introuvable (file not found).")
        kind = documents.kind_of(p)
        if kind in ("pdf", "docx", "eml", "html"):
            res = documents.read_text(p)
            if res.needs_ocr:
                raise UsageError("PDF sans couche texte : pas d'OCR en v0.1 (no text layer).")
            return res.text
        data = p.read_bytes()
        if b"\x00" in data[:4096] and not data.startswith((b"\xff\xfe", b"\xfe\xff")):
            raise UsageError("Fichier binaire non pris en charge (binary file).")
        return documents.read_text(p).text if kind else data.decode("utf-8", errors="replace")
    if use_clipboard and not file_arg:
        from . import clipboard

        return clipboard.paste()
    return sys.stdin.read()


def _exit(msg: str, code: int) -> int:
    _err(msg)
    return code


_KIND_LABELS = {"PERSON": "Personnes", "ORG": "Sociétés", "ADDRESS": "Adresses", "EMAIL": "E-mails",
                "PHONE": "Téléphones", "IBAN": "IBAN", "NIR": "NIR", "SIREN": "SIREN", "SIRET": "SIRET", "VAT": "TVA",
                "CARD": "Cartes", "CASE_REF": "Références", "ID_CODE": "Identifiants", "URL": "Sites",
                "BIRTHDATE": "Dates de naissance", "USERNAME": "Pseudos", "EMPLOYEE_ID": "Matricules",
                "ACCOUNT_NUMBER": "Comptes", "BIRTHPLACE": "Lieux de naissance"}


def _counts_line(counts: dict) -> str:
    if not counts:
        return "aucun"
    return ", ".join(f"{_KIND_LABELS.get(k, k)} {v}" for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _mask_summary(r: dict) -> str:
    """ "2 codes, 4 remplacements (Sociétés 3, Personnes 1)": unique codes, then mentions replaced."""
    counts = r.get("counts") or {}
    n = sum(int(v) for v in counts.values())
    return f"{len(r.get('found') or [])} code(s), {n} remplacement(s) ({_counts_line(counts)})"


# ------------------------------------------------------------------------------------------ dossier commands

def cmd_add(args: argparse.Namespace) -> int:
    if agent_shell():
        return _refuse_in_agent("add", changes=True)
    r = _svc().add_dossier(args.folder, args.title)
    print(r["code"])
    if r.get("existing"):
        _err(f"{r['code']} : ce dossier était déjà enregistré" + (" ; titre mis à jour." if r.get("title_updated")
                                                                 else "."))
    _err(f"{r['code']} : {r['documents']} document(s) analysé(s), {r['entities']} élément(s) enregistré(s) "
         f"({_counts_line(r.get('counts') or {})}).")
    if r.get("unreadable"):
        _err(f"{r['unreadable']} document(s) illisible(s).")
    if r.get("needs_ocr"):
        _err(f"{r['needs_ocr']} PDF sans couche texte (pas d'OCR en v0.1).")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    private = agent_shell() is None
    rows = _svc().list_dossiers(include_private=private)
    if not rows:
        print("Aucun dossier. Ajoutez-en un : ragbase-shield add <dossier>")
        return 0
    for r in rows:
        line = f"{r['code']}\t{r.get('documents')} doc.\t{r.get('entities')} élém."
        if r.get("error"):
            line += "\tcoffre illisible"
        elif private:
            line += f"\t{r.get('title', '')}\t{r.get('folder', '')}"
        print(line)
    if not private:
        _err("(Codes seulement : commande lancée depuis un assistant IA.)")
    return 0


def cmd_entities(args: argparse.Namespace) -> int:
    if agent_shell():
        return _refuse_in_agent("entities")
    for e in _svc().entities(args.dossier):
        flags = []
        if e["kept_clear"]:
            flags.append("gardé en clair")
        if e.get("source") == "user":
            flags.append("ajouté")
        print(f"{e['code']}\t{e['kind']}\t{e['value']}" + (f"\t[{', '.join(flags)}]" if flags else ""))
    return 0


def cmd_register(args: argparse.Namespace) -> int:
    if agent_shell():  # the code it prints would tell a model whether a guessed name is registered
        return _refuse_in_agent("register", changes=True)
    if args.person:
        kind, value = "PERSON", args.person
    elif args.org:
        kind, value = "ORG", args.org
    else:
        kind, value = args.kind[0], args.kind[1]
    r = _svc().register(args.dossier, kind, value)
    print(r["code"])
    return 0


def cmd_keep_clear(args: argparse.Namespace) -> int:
    if agent_shell():  # a model (or a prompt injection in a document) must never switch masking off
        return _refuse_in_agent("keep-clear", changes=True)
    _svc().keep_clear(args.dossier, args.code, not args.off)
    target = args.code if re.fullmatch(r"[A-Z]+_[A-Z0-9]+", args.code) else "terme"
    print(f"{target} : {'masqué de nouveau' if args.off else 'gardé en clair'}")
    return 0


def cmd_mask(args: argparse.Namespace) -> int:
    text = _read_input(args.file, args.clipboard)
    mode = "synthetic" if getattr(args, "synthetic", False) else getattr(args, "mode", None)
    if mode == "synthetic" and not args.dossier:
        _err("Mode synthétique sans --dossier : les pseudonymes ne sont pas gardés, la restauration ne sera pas "
             "possible (synthetic mode without a dossier cannot be restored).")
    r = _svc().mask_text(text, args.dossier, replacement=mode)
    if r.get("mode") == "synthetic":
        _err("Mode synthétique : pseudonymes réalistes à la place des codes (pseudonymisation, pas anonymisation).")
    if args.clipboard:
        from . import clipboard

        clipboard.copy(r["masked"])
        _err(f"Texte masqué copié dans le presse-papiers : {_mask_summary(r)}.")
    else:
        sys.stdout.write(r["masked"])
        if not r["masked"].endswith("\n"):
            sys.stdout.write("\n")
        _err(f"Texte masqué : {_mask_summary(r)}.")
    if not args.dossier:
        _err("Sans --dossier, les codes ne sont pas gardés : la restauration ne sera pas possible.")
    return 0


def cmd_set(args: argparse.Namespace) -> int:
    """``set DOSSIER_n --replacement codes|synthetic [--language auto|fr|en]`` (v0.2): per-dossier settings."""
    if agent_shell():  # a model must never choose how its own input is masked
        return _refuse_in_agent("set", changes=True)
    changes = {k: v for k, v in (("replacement", args.replacement), ("language", args.language)) if v is not None}
    if not changes:
        raise UsageError("Indiquez --replacement codes|synthetic ou --language auto|fr|en.")
    settings: dict = {}
    for key, value in changes.items():
        settings = _svc().set_dossier_setting(args.dossier, key, None if value == "default" else value)
    labels = {"replacement": "remplacement", "language": "langue"}
    shown = ", ".join(f"{labels.get(k, k)} = {v}" for k, v in sorted(settings.items())) or "réglages par défaut"
    print(f"{args.dossier.upper()} : {shown}")
    if settings.get("replacement") == "synthetic":
        print("Mode synthétique : pseudonymes réalistes, réversibles par le coffre (pseudonymisation, pas anonymisation).")
    return 0


def cmd_restore(args: argparse.Namespace) -> int:
    if agent_shell():
        return _refuse_in_agent("restore")
    text = _read_input(args.file, args.clipboard)
    r = _svc().restore_report(text, args.dossier)
    if args.clipboard:
        from . import clipboard

        clipboard.copy(r["text"])
        _err(f"Texte restauré copié dans le presse-papiers ({r['restored']} code(s) remplacé(s)).")
    else:
        sys.stdout.write(r["text"])
        if not r["text"].endswith("\n"):
            sys.stdout.write("\n")
        _err(f"{r['restored']} code(s) remplacé(s).")
    if r.get("restored_synthetic"):
        _err(f"dont {r['restored_synthetic']} pseudonyme(s) synthétique(s).")
    if r.get("unmatched_synthetic"):
        _err(f"Pseudonymes modifiés par le modèle, non restaurés : {r['unmatched_synthetic']} (à relire).")
    if r["unknown_codes"]:
        _err("Codes inconnus de ce dossier, laissés tels quels : " + ", ".join(r["unknown_codes"]))
    # R1 corrections CL-31 / CL-36: generic marks and Shield ids left in the text are named, never silent.
    if r.get("placeholders"):
        _err("Marques génériques laissées dans le texte (à réécrire) : " + _count_list(r["placeholders"]))
    if r.get("shield_ids"):
        _err("Identifiants Shield sans nom local, laissés tels quels : " + _count_list(r["shield_ids"]))
    return 0


def _count_list(items: list[str]) -> str:
    """"[SOCIETE] x2, DOC_4" : each distinct item once, in first-seen order, with its count when above one."""
    counts: dict[str, int] = {}
    for it in items:
        counts[it] = counts.get(it, 0) + 1
    return ", ".join(k if n == 1 else f"{k} x{n}" for k, n in counts.items())


def cmd_desk(args: argparse.Namespace) -> int:
    import importlib

    try:
        # import_module (not "from .desk import app"): it honours sys.modules, even after an earlier import
        desk_app = importlib.import_module(".desk.app", __package__)
    except ImportError:
        return _exit("Shield Desk n'est pas disponible dans cette installation (desk module missing).", 2)
    serve = getattr(desk_app, "serve", None) or getattr(desk_app, "main", None)
    if serve is None:
        return _exit("Shield Desk : point d'entrée serve() introuvable (desk entry point missing).", 2)
    import inspect

    service_mode = bool(getattr(args, "service", False))  # run by the background service: no browser, no token
    kwargs: dict[str, Any] = {"port": args.port,
                              "open_browser": not args.no_browser and not service_mode and agent_shell() is None,
                              "show_token": not service_mode and agent_shell() is None}
    try:
        params = inspect.signature(serve).parameters
        if not any(p.kind is p.VAR_KEYWORD for p in params.values()):
            kwargs = {k: v for k, v in kwargs.items() if k in params}
    except (TypeError, ValueError):
        pass
    try:
        from . import service as _service

        _service.preload_ner()  # v0.2: the optional model loads while the page opens
    except Exception:  # noqa: BLE001
        pass
    result = serve(**kwargs)
    return int(result or 0)


def cmd_mcp(args: argparse.Namespace) -> int:
    from . import mcp_server
    from . import service as _service

    try:
        _service.preload_ner()  # v0.2: the optional model loads while the server starts
    except Exception:  # noqa: BLE001 - never block the server on the optional layer
        pass
    return int(mcp_server.main() or 0)


def cmd_service(args: argparse.Namespace) -> int:
    """``service``: the background Shield Desk of the desktop installers (packaging/DESIGN.md section 10)."""
    from .desk import service_manager as sm

    act = args.action
    if act in ("install", "uninstall", "enable", "disable") and agent_shell():
        sys.stderr.write(f"RAGbase Shield : « service {act} » modifie cet ordinateur (ouverture de session). Lancez-la "
                         "vous-même dans votre terminal, pas depuis un assistant IA. (This command changes the computer: run "
                         "it yourself, not from an AI agent.)\n")
        return 3
    if args.port is not None and not 1 <= args.port <= 65535:
        raise UsageError("port invalide (1 à 65535).")
    say = (lambda _m: None) if args.quiet else (lambda _m: sys.stdout.write(str(_m) + "\n"))
    if act == "run":
        return sm.run_service(no_tray=args.no_tray, for_seconds=args.for_seconds)
    if act == "install":
        r = sm.install(start=not args.no_start, login_item=not args.no_login_item, port_override=args.port,
                       bundle=not args.no_bundle)
        say("Service Shield Desk installé." + (" Ouverture de session : oui." if r["login_item"] != "none"
                                                else " Ouverture de session : non."))
        if not args.no_start:
            say("Démarrage : " + ("en cours." if r["started"] else "échec, voir le journal (logs/service.log)."))
        return 0
    if act == "uninstall":
        app = sm.app_dir()
        script = app / "uninstall.sh" if app is not None else None
        if sm.is_macos() and script is not None and script.is_file():
            cmd = ["/bin/sh", str(script)] + (["--purge-data"] if args.purge_data else []) + \
                  (["--quiet"] if args.quiet else [])
            return subprocess.run(cmd).returncode
        r = sm.uninstall(purge_extras=args.purge_extras, purge_data=args.purge_data)
        if r["purged"]:
            say("Service retiré. Dossier de données supprimé.")
        elif r.get("refused"):
            say(f"Service retiré. {r['home']} contient d'autres fichiers que ceux de Shield : rien n'y est "
                "supprimé.")
        else:
            say(f"Service retiré. Vos dossiers et vos coffres restent dans {r['home']}. Pour tout effacer, "
                "supprimez ce dossier (les anciennes réponses ne pourront plus être restaurées).")
        return 0
    if act == "status":
        ok, lines = sm.status_lines()
        for line in lines:
            sys.stdout.write(line + "\n")
        return 0 if ok else 1
    if act == "start":
        how = sm.start()
        say({"running": "Shield Desk tourne déjà.", "launchd": "Démarrage demandé (launchd).",
             "direct": "Démarrage demandé.", "failed": "Échec du démarrage."}[how])
        return 1 if how == "failed" else 0
    if act == "stop":
        ok = sm.stop()
        say("Shield Desk arrêté." if ok else "Le service ne répond pas : voir le journal.")
        return 0 if ok else 1
    if act == "restart":
        sm.restart()
        say("Redémarrage demandé.")
        return 0
    if act == "enable":
        sm.enable_login_item(sm.service_exe(sm.app_dir()), args.port)
        say("Ouverture de session : oui.")
        return 0
    sm.disable_login_item()  # disable
    say("Ouverture de session : non (le service en cours continue jusqu'à « Quitter »).")
    return 0


def cmd_ner(args: argparse.Namespace) -> int:
    """``ner status``: packages, cached models, resolved backend. ``ner download`` / ``ner install``: fetch a
    model into the models directory (one explicit user step; refused in an AI agent's shell)."""
    from .engine import ner

    if args.action == "status":
        for status, label, detail in ner.status_lines("fr"):
            print(f"{status:<4}  {label} : {detail}")
        st = ner.status()
        missing = [name for name, v in st["packages"].items() if v is None and name in ("gliner2", "torch")]
        if missing:
            print("INFO  Paquets manquants : " + ", ".join(missing)
                  + " (pip install \"ragbase-shield[ner]\")")
        return 0
    if agent_shell() and not args.dry_run:  # a model must never start a download on its own
        print("RAGbase Shield : « ner download » télécharge un modèle (réseau). Seul l'utilisateur peut le lancer : "
              "tapez cette commande vous-même dans votre terminal, pas depuis un assistant IA. (A download must "
              "be started by the user, not by an AI agent.)", file=sys.stderr)
        return 3
    try:
        ner.download_model(args.model, models_dir_path=Path(args.dir) if args.dir else None,
                           dry_run=args.dry_run, force=args.force)
    except ValueError as exc:
        raise UsageError(str(exc)) from None
    except RuntimeError as exc:
        return _exit(f"Erreur : {exc}", 1)
    return 0


def cmd_hook(args: argparse.Namespace) -> int:
    from . import hooks

    return hooks.main(args.event)


def cmd_export(args: argparse.Namespace) -> int:
    from .vault import MIN_PASSPHRASE_CHARS

    if agent_shell():  # the agent could choose the passphrase and decrypt the file itself
        return _refuse_in_agent("export")

    svc = _svc()
    d = svc.registry.get(args.dossier)
    p1 = getpass.getpass("Phrase de passe (8 caractères minimum) : ")
    if len(p1) < MIN_PASSPHRASE_CHARS:
        return _exit("Phrase de passe trop courte (passphrase too short).", 2)
    if getpass.getpass("Confirmez : ") != p1:
        return _exit("Les deux saisies diffèrent (passphrases differ).", 2)
    target = d.export(Path(args.file).expanduser(), p1)
    print(f"{d.code} exporté : {target}")
    _err("Transmettez la phrase de passe par un autre canal que le fichier.")
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    if agent_shell():
        return _refuse_in_agent("import", changes=True)
    svc = _svc()
    p = getpass.getpass("Phrase de passe : ")
    d = svc.registry.import_(Path(args.file).expanduser(), p, folder=args.folder)
    print(d.code)
    return 0


def cmd_journal(args: argparse.Namespace) -> int:
    for e in _svc().journal(args.dossier):
        counts = json.dumps(e.get("counts") or {}, ensure_ascii=False, sort_keys=True)
        print(f"{e['time']}\t{e['event']}\t{e.get('dossier') or '-'}\t{counts}")
    return 0


# ------------------------------------------------------------------------------------------ install

def _appdata() -> Path:
    return Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))


def _xdg_config() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))


def claude_desktop_config_path() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    if sys.platform.startswith("win"):
        return _appdata() / "Claude" / "claude_desktop_config.json"
    return _xdg_config() / "Claude" / "claude_desktop_config.json"


def vscode_mcp_path() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Code" / "User" / "mcp.json"
    if sys.platform.startswith("win"):
        return _appdata() / "Code" / "User" / "mcp.json"
    return _xdg_config() / "Code" / "User" / "mcp.json"


def codex_config_path() -> Path:
    return Path(os.environ.get("CODEX_HOME") or (Path.home() / ".codex")) / "config.toml"


def _installed_launcher() -> Optional[tuple[str, list[str]]]:
    """The desktop installer's launcher (``<home>/install.json``) when it exists and answers ``--version``."""
    try:
        from .desk import service_manager as sm

        found = sm.installed_launcher()
    except Exception:  # noqa: BLE001 - no installer: the other choices apply
        return None
    if not found:
        return None
    exe, pre = found
    rc, out = _run_cmd([exe, *pre, "--version"], timeout=120)
    return (exe, pre) if rc == 0 and "ragbase-shield" in out else None


def server_command() -> tuple[str, list[str]]:
    """Command that starts the MCP server from a GUI app (absolute path: GUI apps often lack the user PATH)."""
    inst = _installed_launcher()
    if inst:  # the desktop installer's private runtime: no uv, works offline
        return inst[0], [*inst[1], "mcp"]
    uvx = shutil.which("uvx")
    if uvx:
        return uvx, ["ragbase-shield", "mcp"]
    exe = shutil.which("ragbase-shield")
    if exe and "archive-v0" not in exe and f"{os.sep}uv{os.sep}" not in exe:
        return exe, ["mcp"]
    return "uvx", ["ragbase-shield", "mcp"]


def _backup(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    target = path.with_name(f"{path.name}.bak-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    n = 2
    while target.exists():
        target = path.with_name(f"{path.name}.bak-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{n}")
        n += 1
    shutil.copy2(path, target)
    return target


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        if os.name == "posix":
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _load_json(path: Path) -> tuple[Optional[dict], Optional[str]]:
    """(data, error). A missing file gives ({}, None). Never rewrites a file it cannot parse exactly."""
    if not path.exists():
        return {}, None
    raw = path.read_text(encoding="utf-8-sig")
    if not raw.strip():
        return {}, None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None, "Ce fichier n'est pas du JSON strict (commentaires ?) : rien n'a été modifié."
    if not isinstance(data, dict):
        return None, "Ce fichier ne contient pas un objet JSON : rien n'a été modifié."
    return data, None


def _install_json(path: Path, section: str, entry: dict, dry_run: bool, app: str) -> int:
    data, error = _load_json(path)
    snippet = json.dumps({section: {SERVER_KEY: entry}}, indent=2, ensure_ascii=False)
    if data is None:
        _err(f"{app} : {path}\n{error}\nAjoutez ce bloc à la main :\n{snippet}")
        return 1
    servers = data.setdefault(section, {})
    if not isinstance(servers, dict):
        _err(f"{app} : la clé « {section} » n'est pas un objet dans {path} : rien n'a été modifié.")
        return 1
    if servers.get(SERVER_KEY) == entry:
        print(f"{app} : déjà configuré ({path}).")
        return 0
    servers[SERVER_KEY] = entry
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if dry_run:
        print(f"{app} : fichier {path}\nEntrée à ajouter (--dry-run : rien n'est écrit) :\n{snippet}")
        return 0
    backup = _backup(path)
    _write_atomic(path, text)
    print(f"{app} : configuré dans {path}" + (f" (sauvegarde : {backup.name})" if backup else ""))
    return 0


def install_claude_desktop(dry_run: bool) -> int:
    cmd, args = server_command()
    code = _install_json(claude_desktop_config_path(), "mcpServers", {"command": cmd, "args": args}, dry_run,
                         "Claude Desktop")
    if code == 0 and not dry_run:
        print("Redémarrez Claude Desktop. Si l'extension .mcpb est déjà installée, n'utilisez pas les deux.")
    return code


def install_vscode(dry_run: bool) -> int:
    cmd, args = server_command()
    code = _install_json(vscode_mcp_path(), "servers", {"type": "stdio", "command": cmd, "args": args}, dry_run,
                         "VS Code")
    if code == 0 and not dry_run:
        print("Rechargez VS Code. Copilot Business/Enterprise : l'administrateur doit autoriser les serveurs MCP.")
    return code


_TOML_HEADER = re.compile(r"""^\s*\[\s*mcp_servers\s*\.\s*(?:"ragbase-shield"|'ragbase-shield'|ragbase-shield)\s*"""
                          r"""(?:\.[^\]]*)?\]\s*(?:#.*)?$""")
_ANY_HEADER = re.compile(r"^\s*\[")
_TOML_SERVER_ANYWHERE = re.compile(r"""(?m)^[^#\n]*(?:mcp_servers[^\n]*ragbase-shield|["']?ragbase-shield["']?\s*=\s*\{)""")


def _toml_block(cmd: str, args: list[str]) -> str:
    return ("[mcp_servers.ragbase-shield]\n"
            f"command = {json.dumps(cmd)}\n"
            f"args = [{', '.join(json.dumps(a) for a in args)}]\n"
            "startup_timeout_sec = 60\n"
            "tool_timeout_sec = 120\n"
            'default_tools_approval_mode = "writes"\n')


def _toml_load(text: str) -> Optional[dict]:
    try:
        import tomllib  # Python 3.11+
    except ImportError:  # pragma: no cover - Python 3.10: no validation available
        return None
    return tomllib.loads(text)


def _replace_toml_table(text: str, block: str) -> Optional[str]:
    lines = text.splitlines(keepends=True)
    start = next((i for i, l in enumerate(lines) if _TOML_HEADER.match(l)), None)
    if start is None:
        return None
    end = start + 1
    while end < len(lines):
        if _ANY_HEADER.match(lines[end]) and not _TOML_HEADER.match(lines[end]):
            break
        end += 1
    tail = lines[end:]
    return "".join(lines[:start]) + block + ("\n" if tail else "") + "".join(tail)


def install_codex(dry_run: bool, app: str = "Codex") -> int:
    path = codex_config_path()
    cmd, args = server_command()
    block = _toml_block(cmd, args)
    want = {"command": cmd, "args": args}
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    try:
        parsed = _toml_load(text) if text.strip() else {}
    except Exception:  # noqa: BLE001 - tomllib.TOMLDecodeError
        _err(f"{app} : {path} n'est pas un TOML valide : rien n'a été modifié. Ajoutez ce bloc à la main :\n{block}")
        return 1
    current = ((parsed or {}).get("mcp_servers") or {}).get(SERVER_KEY) if parsed is not None else None
    if parsed is None and text.strip() and _TOML_SERVER_ANYWHERE.search(text) \
            and not any(_TOML_HEADER.match(line) for line in text.splitlines()):
        current = {}  # Python 3.10 (no tomllib): the server is defined some other way; never add a second table
    if isinstance(current, dict) and current and all(current.get(k) == v for k, v in want.items()):
        print(f"{app} : déjà configuré ({path}).")
        return 0
    if any(_TOML_HEADER.match(line) for line in text.splitlines()):
        new = _replace_toml_table(text, block)
    elif current is not None:
        new = None  # defined inline or with dotted keys: do not guess
    else:
        sep = "" if not text or text.endswith("\n\n") else ("\n" if text.endswith("\n") else "\n\n")
        new = text + sep + block
    if new is None:
        _err(f"{app} : l'entrée ragbase-shield de {path} n'a pas une forme connue : rien n'a été modifié.\n"
             f"Remplacez-la à la main par :\n{block}")
        return 1
    try:
        check = _toml_load(new)
    except Exception:  # noqa: BLE001
        _err(f"{app} : le résultat ne serait pas un TOML valide : rien n'a été modifié.")
        return 1
    if check is not None:
        got = (check.get("mcp_servers") or {}).get(SERVER_KEY) or {}
        if any(got.get(k) != v for k, v in want.items()):
            _err(f"{app} : vérification impossible du résultat : rien n'a été modifié.")
            return 1
    if dry_run:
        print(f"{app} : fichier {path}\nTable à ajouter (--dry-run : rien n'est écrit) :\n{block}")
        return 0
    backup = _backup(path)
    _write_atomic(path, new)
    print(f"{app} : configuré dans {path}" + (f" (sauvegarde : {backup.name})" if backup else ""))
    print("Redémarrez l'application ChatGPT ou Codex.")
    return 0


def _repo_root() -> Optional[Path]:
    root = Path(__file__).resolve().parents[2]
    return root if (root / ".claude-plugin" / "marketplace.json").is_file() else None


def _run_cmd(cmd: list[str], timeout: int = 180) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, type(exc).__name__
    return p.returncode, (p.stdout + p.stderr).strip()


def install_claude_code(dry_run: bool) -> int:
    claude = shutil.which("claude")
    source = os.environ.get("RAGBASE_SHIELD_MARKETPLACE") or (str(_repo_root()) if _repo_root() else None)
    exe = claude or "claude"
    if source:
        steps = [[exe, "plugin", "marketplace", "add", source],
                 [exe, "plugin", "install", PLUGIN_ID, "--scope", "user"]]
        note = "Plugin complet : serveur MCP, compétence et hooks."
    else:
        cmd, args = server_command()
        steps = [[exe, "mcp", "add", "--scope", "user", SERVER_KEY, "--", cmd, *args]]
        note = ("ATTENTION : serveur MCP seul, SANS les hooks ni la compétence (rien ne bloque un vrai nom tapé "
                "dans Claude Code). Pour la protection complète, installez le plugin dans Claude Code : "
                "/plugin marketplace add <dépôt GitHub de RAGbase Shield> puis /plugin install "
                f"{PLUGIN_ID} ; ou relancez cette commande depuis une copie du dépôt (RAGBASE_SHIELD_MARKETPLACE="
                "<dossier du dépôt>).")
    shown = "\n".join("  " + " ".join(_q(a) for a in s) for s in steps)
    if dry_run:
        print(f"Claude Code : commandes prévues (aucune exécution, --dry-run) :\n{shown}\n{note}")
        return 0
    if not claude:
        _err(f"Claude Code introuvable (commande « claude »). Lancez vous-même :\n{shown}\n{note}")
        return 1
    if not source:
        rc, _out = _run_cmd([claude, "mcp", "get", SERVER_KEY])
        if rc == 0:
            print(f"Claude Code : serveur {SERVER_KEY} déjà configuré.\n{note}")
            return 0
    for s in steps:
        rc, out = _run_cmd(s)
        if rc != 0 and "already" not in out.lower() and "déjà" not in out.lower():
            _err(f"Claude Code : échec de « {' '.join(s[1:4])} » :\n{out[-800:]}")
            return 1
    print(f"Claude Code : installé. {note}\nRedémarrez Claude Code.")
    return 0


def _q(s: str) -> str:
    return s if re.fullmatch(r"[A-Za-z0-9_@%+=:,./-]+", s) else json.dumps(s)


def cmd_install(args: argparse.Namespace) -> int:
    app = args.app
    if app == "claude-desktop":
        return install_claude_desktop(args.dry_run)
    if app == "claude-code":
        return install_claude_code(args.dry_run)
    if app in ("chatgpt", "codex"):
        code = install_codex(args.dry_run, "ChatGPT desktop" if app == "chatgpt" else "Codex")
        if app == "chatgpt" and code == 0:
            print("ChatGPT desktop lit ~/.codex/config.toml (mode Work et Codex). ChatGPT web n'est pas concerné.")
        return code
    if app == "vscode":
        return install_vscode(args.dry_run)
    return _exit("Application inconnue.", 2)


# ------------------------------------------------------------------------------------------ doctor

def _mode(path: Path) -> Optional[int]:
    try:
        return stat.S_IMODE(path.stat().st_mode)
    except OSError:
        return None


def _claude_code_dir() -> Path:
    """Claude Code settings folder: $CLAUDE_CONFIG_DIR when set, else ~/.claude."""
    env = os.environ.get("CLAUDE_CONFIG_DIR")
    return Path(env).expanduser() if env else Path.home() / ".claude"


def _probe_command(cmd: str, args: list[str]) -> tuple[bool, str]:
    """Can the command an app was configured with start RAGbase Shield? Runs it with --version, offline.

    ``uvx ragbase-shield`` works only when the tool is installed (``uv tool install``) or already cached: before
    the PyPI release, or on a machine without network, an app configured that way cannot start the server.
    """
    if not cmd:
        return False, "commande vide"
    exe = cmd if os.path.isabs(cmd) else shutil.which(cmd)
    if not exe:
        return False, "commande introuvable"
    if os.path.basename(exe).lower().startswith("uvx"):
        probe = [exe, "--offline", *[a for a in args if a != "mcp"], "--version"]
    else:
        probe = [exe, *[a for a in args if a != "mcp"], "--version"]
    rc, out = _run_cmd(probe, timeout=60)
    return rc == 0 and "ragbase-shield" in out, (out.splitlines()[-1][:120] if out else f"code {rc}")


def _app_commands() -> list[tuple[str, str, list[str]]]:
    """(app, command, args) of every app configured with the RAGbase Shield server."""
    out: list[tuple[str, str, list[str]]] = []
    for app, path, section in (("Claude Desktop", claude_desktop_config_path(), "mcpServers"),
                               ("VS Code (Copilot)", vscode_mcp_path(), "servers")):
        data, _ = _load_json(path)
        entry = ((data or {}).get(section) or {}).get(SERVER_KEY)
        if isinstance(entry, dict) and isinstance(entry.get("command"), str):
            out.append((app, entry["command"], [str(a) for a in entry.get("args") or []]))
    try:
        parsed = _toml_load(codex_config_path().read_text(encoding="utf-8"))
        entry = ((parsed or {}).get("mcp_servers") or {}).get(SERVER_KEY)
        if isinstance(entry, dict) and isinstance(entry.get("command"), str):
            out.append(("ChatGPT desktop / Codex", entry["command"], [str(a) for a in entry.get("args") or []]))
    except Exception:  # noqa: BLE001 - missing or invalid file: nothing to probe
        pass
    return out


def _app_status() -> list[tuple[str, str, str]]:
    out = []
    data, _ = _load_json(claude_desktop_config_path())
    desktop = bool(data and SERVER_KEY in (data.get("mcpServers") or {}))
    ext_dir = claude_desktop_config_path().parent / "Claude Extensions"
    try:
        ext = ext_dir.is_dir() and any("ragbase-shield" in p.name for p in ext_dir.iterdir())
    except OSError:
        ext = False
    out.append(("OK" if desktop or ext else "INFO", "Claude Desktop",
                "extension .mcpb installée" if ext else ("configuré" if desktop else "non configuré")))
    cc = False
    cc_dir = _claude_code_dir()
    for p in (Path.home() / ".claude.json", cc_dir / ".claude.json", cc_dir / "settings.json"):
        d, _ = _load_json(p)
        if d and (SERVER_KEY in (d.get("mcpServers") or {})
                  or any(k.startswith("ragbase-shield@") for k in (d.get("enabledPlugins") or {}))):
            cc = True
    out.append(("OK" if cc else "INFO", "Claude Code", "configuré" if cc else "non configuré"))
    try:
        text = codex_config_path().read_text(encoding="utf-8")
        codex = any(_TOML_HEADER.match(l) for l in text.splitlines())
    except OSError:
        codex = False
    out.append(("OK" if codex else "INFO", "ChatGPT desktop / Codex", "configuré" if codex else "non configuré"))
    data, _ = _load_json(vscode_mcp_path())
    vs = bool(data and SERVER_KEY in (data.get("servers") or {}))
    out.append(("OK" if vs else "INFO", "VS Code (Copilot)", "configuré" if vs else "non configuré"))
    seen: dict[tuple[str, tuple[str, ...]], tuple[bool, str]] = {}
    for app, cmd, args in _app_commands():
        key = (cmd, tuple(args))
        if key not in seen:
            seen[key] = _probe_command(cmd, args)
        ok, detail = seen[key]
        out.append(("OK", f"{app} : démarrage du serveur", "la commande répond") if ok else
                   ("FAIL", f"{app} : démarrage du serveur",
                    f"« {os.path.basename(cmd)} {' '.join(args)} » ne démarre pas ({detail}). Installez l'outil : "
                    "« uv tool install ragbase-shield » (avant la publication : « uv tool install <dossier source> »), "
                    "puis relancez « ragbase-shield doctor »."))
    return out


def doctor_checks() -> list[tuple[str, str, str]]:
    """[(status, label, detail)] with status OK, INFO, WARN or FAIL. Never shows client data."""
    checks: list[tuple[str, str, str]] = []
    py = sys.version_info
    checks.append(("OK" if py >= (3, 10) else "FAIL", "Python", f"{platform.python_version()} (3.10 minimum)"))
    checks.append(("OK", "RAGbase Shield", __version__))
    from importlib import metadata

    for mod, dist, label, required in (("mcp", "mcp", "mcp", True), ("pypdf", "pypdf", "pypdf", True),
                                       ("docx", "python-docx", "python-docx", True),
                                       ("cryptography", "cryptography", "cryptography", True),
                                       ("keyring", "keyring", "keyring", True),
                                       ("starlette", "starlette", "starlette (Shield Desk)", False),
                                       ("uvicorn", "uvicorn", "uvicorn (Shield Desk)", False)):
        try:
            __import__(mod)
        except Exception:  # noqa: BLE001
            checks.append(("FAIL" if required else "WARN", label, "absent"))
            continue
        try:
            version = metadata.version(dist)
        except Exception:  # noqa: BLE001
            version = "installé"
        checks.append(("OK", label, version))
    home = config.home()
    reg = home / "registry.db"
    keyring_off = os.environ.get("RAGBASE_SHIELD_KEYRING", "1").strip().lower() in ("0", "false", "no", "off")
    if reg.exists():
        try:
            from .vault import Registry

            with Registry(home) as r:
                n = len(r.folders())
                src = r.key_source
            checks.append(("OK", "Clé du coffre", "trousseau du système" if src == "keyring" else "fichier 0600"))
            checks.append(("OK", "Coffre", f"{n} dossier(s)"))
        except Exception as exc:  # noqa: BLE001
            checks.append(("FAIL", "Coffre", f"ne s'ouvre pas ({type(exc).__name__}) : clé absente ou différente"))
    else:
        if keyring_off:
            backend = "fichier 0600 (RAGBASE_SHIELD_KEYRING=0)"
        else:
            try:
                import keyring

                kr = keyring.get_keyring()
                name = type(kr).__name__
                backend = "fichier 0600 (trousseau absent)" if "fail" in name.lower() or "null" in name.lower() \
                    else f"trousseau du système ({name})"
            except Exception:  # noqa: BLE001
                backend = "fichier 0600 (trousseau absent)"
        checks.append(("OK", "Clé du coffre", f"pas encore créée ; sera stockée dans : {backend}"))
    if os.name == "posix":
        for path, want, label in ((home, 0o700, "Dossier de données"), (home / "key", 0o600, "Fichier de clé"),
                                  (reg, 0o600, "Registre"), (home / "vaults", 0o700, "Coffres"),
                                  (home / "desk_token", 0o600, "Jeton Shield Desk")):
            mode = _mode(path)
            if mode is None:
                continue
            ok = not (mode & 0o077)
            checks.append(("OK" if ok else "FAIL", label, f"droits {oct(mode)[2:]}" + ("" if ok else
                           f" (attendu {oct(want)[2:]} : lancez chmod {oct(want)[2:]})")))
    inst = None
    try:
        from .desk import service_manager as sm

        inst = sm.read_install_json()
    except Exception:  # noqa: BLE001
        sm = None  # type: ignore[assignment]
    if inst and sm is not None:
        launcher_ok = sm.installed_launcher() is not None
        checks.append(("OK" if launcher_ok else "WARN", "Installateur",
                       f"{inst.get('app_dir') or 'sans installateur'} (v{inst.get('version')})"
                       + ("" if launcher_ok else " : lanceur introuvable, réinstallez")))
        st, info = sm.desk_status()
        login = "oui" if sm.login_item_enabled() else "non"
        if st == "ok":
            checks.append(("OK", "Service Shield Desk",
                           f"actif, port {sm.port()}, ouverture de session : {login}"))
        else:
            checks.append(("WARN", "Service Shield Desk",
                           f"inactif (port {sm.port()}), ouverture de session : {login} : "
                           "« ragbase-shield service start »"))
    uvx_found = bool(shutil.which("uvx"))
    checks.append(("OK" if uvx_found else ("INFO" if inst else "WARN"), "uv / uvx",
                   "trouvé" if uvx_found else ("absent : les applications utilisent le Shield installé"
                                                if inst else "absent : les intégrations lancent « uvx ragbase-shield »")))
    checks.extend(_app_status())
    try:
        from .engine import ner as _ner

        checks.extend(_ner.status_lines("fr"))  # v0.2: packages, cached models, resolved backend (no load)
    except Exception:  # noqa: BLE001
        checks.append(("INFO", "Détection NER (option)", "non installée (règles seules)"))
    try:
        from . import clipboard

        b = clipboard.backend()
    except Exception:  # noqa: BLE001
        b = None
    checks.append(("OK" if b else "WARN", "Presse-papiers", b or "aucun outil trouvé"))
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        try:
            s.bind(("127.0.0.1", config.DESK_PORT))
            port = "libre"
        except OSError:
            port = "occupé (Shield Desk déjà lancé ?)"
    checks.append(("INFO", f"Port Shield Desk {config.DESK_PORT}", port))
    return checks


def cmd_doctor(args: argparse.Namespace) -> int:
    checks = doctor_checks()
    for status, label, detail in checks:
        print(f"{status:<4}  {label} : {detail}")
    fails = sum(1 for s, _l, _d in checks if s == "FAIL")
    print("Résultat : " + ("tout est prêt." if not fails else f"{fails} problème(s) à corriger."))
    return 1 if fails else 0


# ------------------------------------------------------------------------------------------ verify

def _luhn_digit(body: str) -> str:
    total = 0
    for i, ch in enumerate(reversed(body)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return str((10 - total % 10) % 10)


def _iban_fr(bank: str, branch: str, account: str) -> str:
    rib = 97 - ((89 * int(bank) + 15 * int(branch) + 3 * int(account)) % 97)
    bban = f"{bank}{branch}{account}{rib:02d}"
    num = "".join(str(int(c, 36)) for c in bban + "FR00")
    check = 98 - int(num) % 97
    return f"FR{check:02d}{bban}"


def synthetic_case() -> dict:
    """Invented French client data with valid check digits, for ``verify`` (no real person or company)."""
    siren = "73519264" + _luhn_digit("73519264")
    nic = "0003"
    siret = siren + nic + _luhn_digit(siren + nic)
    vat = f"FR{(12 + 3 * (int(siren) % 97)) % 97:02d}{siren}"
    nir_body = "2850759606123"
    nir = f"{nir_body}{97 - int(nir_body) % 97:02d}"
    iban = _iban_fr("17569", "00342", "40718235096")
    fmt = lambda s, n=4: " ".join(s[i:i + n] for i in range(0, len(s), n))  # noqa: E731
    values = {
        "person1": "Clémence Morvandiau", "person1_up": "MORVANDIAU",
        "person2": "Ugo Ferracci-Daubel", "person3": "Anouk Trivelin",
        "org1": "Lambretti Conseil", "org2": "Ostrevent Négoce",
        "email1": "clemence.morvandiau@exemple-morvandiau.fr", "email2": "a.trivelin@cabinet-trivelin.fr",
        "phone": "06 39 98 41 27", "address1": "14 rue des Glycines, 59300 Valenciennes",
        "address2": "3 allée des Peupliers, 59000 Lille",
        "siren": siren, "siret": siret, "vat": vat, "nir": nir, "iban": iban, "rg": "26/04417",
        "birth": "4 juillet 1985",
    }
    title = "Morvandiau c. Lambretti Conseil SAS"
    assignation = (
        "ASSIGNATION DEVANT LE TRIBUNAL JUDICIAIRE DE VALENCIENNES\n\n"
        "À LA REQUÊTE DE :\n"
        f"Madame Clémence MORVANDIAU, née le {values['birth']} à Douai (59), demeurant {values['address1']}, "
        f"numéro de sécurité sociale {fmt(nir, 3)}, joignable au {values['phone']} et à {values['email1']},\n\n"
        "Ayant pour avocat Maître Anouk TRIVELIN, avocate au barreau de Valenciennes,\n\n"
        "CONTRE :\n"
        f"La société LAMBRETTI CONSEIL SAS, immatriculée au RCS de Lille sous le numéro {fmt(siren, 3)}, "
        f"SIRET {siret}, TVA intracommunautaire {vat}, dont le siège social est situé {values['address2']}, "
        "représentée par M. Ugo FERRACCI-DAUBEL, président,\n\n"
        f"Affaire RG n° {values['rg']}.\n\n"
        "EXPOSÉ DES FAITS\n\n"
        f"Mme Morvandiau a versé un acompte de 12 400 euros sur le compte {fmt(iban)} ouvert au nom de la "
        "société Lambretti Conseil. Aucune prestation n'a été exécutée, en violation de l'article 1217 du "
        "Code civil.\n")
    note = ("# Note interne\n\nRendez-vous avec Mme Morvandiau le 12 mai. M. Ferracci-Daubel conteste la facture "
            "de la société Ostrevent Négoce SARL.\n\n- Pièce 3 : relevé du compte " + iban + "\n")
    eml = (f"From: Anouk Trivelin <{values['email2']}>\nTo: Clémence Morvandiau <{values['email1']}>\n"
           f"Subject: Dossier Morvandiau c. Lambretti Conseil\nDate: Mon, 14 Sep 2026 10:12:00 +0200\n"
           "MIME-Version: 1.0\nContent-Type: text/plain; charset=utf-8\nContent-Transfer-Encoding: 8bit\n\n"
           "Chère Madame Morvandiau,\n\nJe vous confirme l'audience. M. Ferracci-Daubel sera présent.\n\n"
           f"Bien cordialement,\nMaître Anouk Trivelin\n{values['phone']}\n")
    html = ("<html><body><h1>Lambretti Conseil SAS</h1><p>Président : M. Ugo Ferracci-Daubel. "
            f"Contact : {values['phone']}.</p></body></html>\n")
    letter = ("Madame Morvandiau,\n\nNous accusons réception de votre courrier. La société Lambretti Conseil "
              "vous répondra sous huit jours.\n")
    return {"title": title, "values": values, "files": {
        "assignation.txt": assignation, "note.md": note, "message.eml": eml, "fiche.html": html,
        "Courrier Morvandiau.txt": letter}}


def _write_docx_fixture(path: Path, values: dict) -> None:
    import docx

    doc = docx.Document()
    doc.sections[0].header.paragraphs[0].text = "Morvandiau c. Lambretti Conseil"
    doc.add_heading("Contrat de prestation", 1)
    doc.add_paragraph(f"Entre la société LAMBRETTI CONSEIL SAS (SIREN {values['siren']}) et Madame Clémence "
                      f"MORVANDIAU, demeurant {values['address1']}.")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "Prestataire", "Lambretti Conseil SAS"
    table.cell(1, 0).text, table.cell(1, 1).text = "Cliente", "Mme Clémence Morvandiau"
    doc.save(str(path))


class _LogCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.lines.append(record.getMessage())
        except Exception:  # noqa: BLE001
            self.lines.append(str(record.msg))


def _compact(s: str) -> str:
    return re.sub(r"[\s.\-/]", "", s).lower()


class LeakChecker:
    """Finds planted or registered values in a model-facing output (exact, normalised, digits-only)."""

    def __init__(self, values: list[str]):
        from .engine import normalize

        self._norm = normalize
        vals = sorted({" ".join(v.split()) for v in values if v and len(v.strip()) >= 3}, key=len, reverse=True)
        self.exact = vals
        self.normed = [(v, normalize(v)) for v in vals if len(normalize(v)) >= 4]
        self.compact = [(v, _compact(v)) for v in vals if sum(c.isdigit() for c in v) >= 6]

    def find(self, output: Any) -> list[str]:
        text = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False, default=str)
        found = [v for v in self.exact if v in text]
        nt = " " + self._norm(text) + " "
        for v, n in self.normed:
            if v not in found and (" " in n or len(n) >= 6) and re.search(r"(?<![a-z0-9])" + re.escape(n)
                                                                         + r"(?![a-z0-9])", nt):
                found.append(v)
        ct = _compact(text)
        for v, c in self.compact:
            if v not in found and c in ct:
                found.append(v)
        return found


@contextlib.contextmanager
def _temp_env(home: Path) -> Iterator[None]:
    from . import service

    keys = ("RAGBASE_SHIELD_HOME", "RAGBASE_SHIELD_KEYRING")
    old = {k: os.environ.get(k) for k in keys}
    service.reset()
    os.environ["RAGBASE_SHIELD_HOME"] = str(home)
    os.environ["RAGBASE_SHIELD_KEYRING"] = "0"
    try:
        yield
    finally:
        service.reset()
        # The PreToolUse checks keep the temporary registry open: Windows cannot delete an open file, so
        # "verify" ended in "Erreur interne (PermissionError)" before its result line (Windows patch W1,
        # seen again on the installed Shield, 2026-10-27).
        from . import hooks

        for reg in list(hooks._REGISTRIES.values()):
            try:
                reg.close()
            except Exception:  # noqa: BLE001
                pass
        hooks._REGISTRIES.clear()
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _mcp_call(server: Any, name: str, args: dict) -> tuple[bool, str]:
    import asyncio

    from mcp import Client

    async def go() -> tuple[bool, str]:
        async with Client(server) as c:
            r = await c.call_tool(name, args)
            return bool(r.is_error), "\n".join(getattr(x, "text", "") for x in r.content)

    return asyncio.run(go())


def run_verify(report: Callable[[str, str, str], None]) -> dict:
    """Plant synthetic data in a temporary home, run every path, report each one. Returns counts."""
    from . import hooks, service
    from .mcp_server import build_server

    counts = {"CLEAN": 0, "LEAK": 0, "ERROR": 0, "SKIP": 0}

    def rep(status: str, path: str, detail: str = "") -> None:
        counts[status] = counts.get(status, 0) + 1
        report(status, path, detail)

    capture = _LogCapture()
    root = logging.getLogger()
    old_level = root.level
    quiet = [(h, h.level) for h in root.handlers]
    for h, lvl in quiet:
        h.setLevel(max(lvl, logging.WARNING))
    root.addHandler(capture)
    root.setLevel(logging.INFO)
    with tempfile.TemporaryDirectory(prefix="ragbase-shield-verify-") as tmp:
        tmpdir = Path(tmp)
        home, folder = tmpdir / "home", tmpdir / "Dossier Morvandiau"
        folder.mkdir()
        case = synthetic_case()
        for name, text in case["files"].items():
            (folder / name).write_text(text, encoding="utf-8")
        try:
            _write_docx_fixture(folder / "contrat.docx", case["values"])
        except Exception:  # noqa: BLE001
            pass
        with _temp_env(home):
            try:
                svc = service.default()
                info = svc.add_dossier(folder, case["title"])
                code = info["code"]
                d = svc.registry.get(code)
                for kind, value in (("PERSON", "Clémence Morvandiau"), ("ORG", "Lambretti Conseil SAS")):
                    svc.register(code, kind, value)
                values = list(case["values"].values()) + [case["title"], str(folder), folder.name]
                values += [e.value for e in d.all()]
                checker = LeakChecker(values)
            except Exception as exc:  # noqa: BLE001
                rep("ERROR", "setup", type(exc).__name__)
                _restore_logging(root, capture, old_level, quiet)
                return counts

            def check(path: str, output: Any, must_error: Optional[bool] = None, is_error: bool = False) -> None:
                leaks = checker.find(output)
                if leaks:
                    rep("LEAK", path, f"{len(leaks)} valeur(s) visibles")
                elif must_error is not None and must_error != is_error:
                    rep("ERROR", path, "résultat inattendu")
                else:
                    rep("CLEAN", path)

            # --- MCP tools, through the real MCP server (in process)
            server = build_server()
            try:
                for name, args in (("shield_list_dossiers", {}), ("shield_status", {}),
                                   ("shield_list_documents", {"dossier": code})):
                    err, out = _mcp_call(server, name, args)
                    check(f"mcp {name}", out, False, err)
                docs = svc.list_documents(code)
                for doc in docs:
                    start: Optional[int] = 0
                    pages = []
                    while start is not None and len(pages) < 200:
                        err, out = _mcp_call(server, "shield_read_document",
                                             {"dossier": code, "document": doc["id"], "start": start, "max_chars": 1500})
                        pages.append(out)
                        m = re.search(r"suite : start=(\d+)\]", out)
                        start = int(m.group(1)) if m and not err else None
                    check(f"mcp shield_read_document {doc['id']} ({doc['kind']})", "\n".join(pages))
                p1 = next((e.code for e in d.all() if e.kind == "PERSON"), "PERSONNE_A")
                for q, label in ((f"{p1} acompte", "code + mot"), ("acompte", "mot"),
                                 ("Morvandiau", "nom de personne deviné"), ("Lambretti", "nom de société deviné")):
                    err, out = _mcp_call(server, "shield_search", {"dossier": code, "query": q})
                    check(f"mcp shield_search ({label})", out, False, err)
                err, out = _mcp_call(server, "shield_save_draft",
                                     {"dossier": code, "title": f"Conclusions {p1}",
                                      "content": f"# Conclusions\n\n{p1} demande le remboursement.", "format": "docx"})
                check("mcp shield_save_draft", out, False, err)
                out_id = re.search(r"OUTPUT_\d+", out)
                restored_ok = False
                if out_id:
                    from . import documents as docs_mod

                    path = d.get_output(out_id.group())
                    ent = d.get_by_code(p1)
                    restored_ok = bool(path and ent and ent.value in docs_mod.read_text(path).text)
                rep("CLEAN" if restored_ok else "ERROR", "fichier restauré (Shield - sorties)",
                    "" if restored_ok else "noms non restaurés")
                for args in ({"dossier": "DOSSIER_999", "document": "DOC_1"},
                             {"dossier": code, "document": "DOC_9999"},
                             {"dossier": case["title"], "document": "DOC_1"}):
                    err, out = _mcp_call(server, "shield_read_document", args)
                    check("mcp erreur (code inconnu)", out, True, err)
            except Exception as exc:  # noqa: BLE001
                rep("ERROR", "mcp", type(exc).__name__)

            # --- MCP over stdio (a real child process)
            try:
                rep(*_verify_stdio(home, code, checker))
            except Exception as exc:  # noqa: BLE001
                rep("ERROR", "mcp stdio", type(exc).__name__)

            # --- hooks
            def run_hook(event: str, payload: dict) -> tuple[int, str]:
                out_buf, err_buf = io.StringIO(), io.StringIO()
                rc = hooks.main(event, io.StringIO(json.dumps(payload)), out_buf, err_buf)
                return rc, out_buf.getvalue() + err_buf.getvalue()

            try:
                rc, out = run_hook("user-prompt-submit", {"hook_event_name": "UserPromptSubmit", "cwd": tmp,
                                                          "prompt": "Rédige une lettre pour Clémence Morvandiau "
                                                                    "contre Lambretti Conseil."})
                if checker.find(out):
                    rep("LEAK", "hook user-prompt-submit", "valeur répétée dans la réponse")
                elif rc != 2 or '"block"' not in out:
                    rep("LEAK", "hook user-prompt-submit", "le message avec un vrai nom n'est pas bloqué")
                else:
                    rep("CLEAN", "hook user-prompt-submit (message bloqué)")
                rc, out = run_hook("user-prompt-submit", {"prompt": f"Rédige une lettre pour {p1}."})
                rep("CLEAN" if rc == 0 and not out.strip() else "ERROR", "hook user-prompt-submit (codes acceptés)")
                target = folder / "assignation.txt"
                for label, tool, ti in (
                        ("Read", "Read", {"file_path": str(target)}),
                        ("Grep", "Grep", {"pattern": "acompte", "path": str(folder)}),
                        ("Glob", "Glob", {"pattern": "**/*.txt", "path": str(folder)}),
                        ("Bash cat", "Bash", {"command": f'cat "{target}"'}),
                        ("Bash cd", "Bash", {"command": f"cd '{folder.parent}' && grep -r acompte ."}),
                        ("Bash clé", "Bash", {"command": f"cat {home}/key"})):
                    rc, out = run_hook("pre-tool-use", {"tool_name": tool, "tool_input": ti, "cwd": tmp})
                    if checker.find(out):
                        rep("LEAK", f"hook pre-tool-use {label}", "valeur répétée")
                    elif rc != 2 or '"deny"' not in out:
                        rep("LEAK", f"hook pre-tool-use {label}", "accès direct non refusé")
                    else:
                        rep("CLEAN", f"hook pre-tool-use {label} (refusé)")
                rc, out = run_hook("pre-tool-use", {"tool_name": "Bash", "tool_input": {"command": "ls /"},
                                                    "cwd": tmp})
                rep("CLEAN" if rc == 0 else "ERROR", "hook pre-tool-use (commande sans risque acceptée)")
                leak_text = case["files"]["assignation.txt"]
                for label, tool, resp in (
                        ("Bash", "Bash", {"stdout": leak_text, "stderr": "", "interrupted": False, "isImage": False}),
                        ("MCP", "mcp__other__fetch", [{"type": "text", "text": leak_text}]),
                        ("texte", "WebFetch", leak_text)):
                    rc, out = run_hook("post-tool-use", {"tool_name": tool, "tool_input": {}, "tool_response": resp})
                    replaced = '"updatedToolOutput"' in out
                    if checker.find(out) or not replaced:
                        rep("LEAK", f"hook post-tool-use {label}", "valeur restée dans le résultat")
                    else:
                        rep("CLEAN", f"hook post-tool-use {label} (masqué)")
            except Exception as exc:  # noqa: BLE001
                rep("ERROR", "hooks", type(exc).__name__)

            # --- CLI mask path (text a user then pastes)
            try:
                check("cli mask --dossier", svc.mask_text(case["files"]["assignation.txt"], code)["masked"])
                check("cli mask (sans dossier)", svc.mask_text(case["files"]["assignation.txt"])["masked"])
            except Exception as exc:  # noqa: BLE001
                rep("ERROR", "cli mask", type(exc).__name__)

            # --- synthetic replacement mode (v0.2): no real value in the pseudonymised text, exact restore
            try:
                raw_text = case["files"]["assignation.txt"]
                syn = svc.mask_text(raw_text, code, replacement="synthetic")
                check("cli mask --mode synthetic", syn["masked"])
                restored = svc.restore_report(syn["masked"], code)["text"]
                rep("CLEAN" if restored == raw_text else "ERROR", "restauration (mode synthétique)",
                    "" if restored == raw_text else "texte restauré différent")
                again = svc.mask_text(syn["masked"], code)["masked"]
                check("cli mask (réponse synthétique remasquée en codes)", again)
                check("entités (valeurs synthétiques)", [e["synthetic"] for e in svc.entities(code)])
            except Exception as exc:  # noqa: BLE001
                rep("ERROR", "mode synthétique", type(exc).__name__)

            # --- an AI agent's shell can never switch masking off
            try:
                before = svc.read_document(code, "DOC_1")["text"]
                old_agent = os.environ.get("RAGBASE_SHIELD_AGENT")
                os.environ["RAGBASE_SHIELD_AGENT"] = "1"
                try:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        rc = main(["keep-clear", code, p1])
                finally:
                    if old_agent is None:
                        os.environ.pop("RAGBASE_SHIELD_AGENT", None)
                    else:
                        os.environ["RAGBASE_SHIELD_AGENT"] = old_agent
                after = svc.read_document(code, "DOC_1")["text"]
                if checker.find(after):
                    rep("LEAK", "cli keep-clear (terminal d'un agent IA)", "masquage désactivé")
                else:
                    rep("CLEAN" if rc == 3 and after == before else "ERROR", "cli keep-clear (terminal d'un agent IA, refusé)")
            except Exception as exc:  # noqa: BLE001
                rep("ERROR", "cli keep-clear", type(exc).__name__)

            # --- Desk API
            for status, path, detail in _verify_desk(code, folder, case, checker):
                rep(status, path, detail)

            # --- journal and logs
            try:
                check("journal", svc.journal())
            except Exception as exc:  # noqa: BLE001
                rep("ERROR", "journal", type(exc).__name__)
            check("journaux techniques (logs)", "\n".join(capture.lines))
    _restore_logging(root, capture, old_level, quiet)
    return counts


def _restore_logging(root: logging.Logger, capture: logging.Handler, level: int, quiet: list) -> None:
    root.removeHandler(capture)
    root.setLevel(level)
    for h, lvl in quiet:
        h.setLevel(lvl)


def _verify_stdio(home: Path, code: str, checker: LeakChecker) -> tuple[str, str, str]:
    """Start ``python -m ragbase_shield mcp`` as a real child process and call two tools over stdio."""
    import asyncio

    from mcp import Client, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env = dict(os.environ)
    env.update(RAGBASE_SHIELD_HOME=str(home), RAGBASE_SHIELD_KEYRING="0")
    params = StdioServerParameters(command=sys.executable, args=["-m", "ragbase_shield", "mcp"], env=env)

    with tempfile.TemporaryFile("w+", encoding="utf-8") as errlog:
        async def go() -> tuple[set, str]:
            async with Client(stdio_client(params, errlog=errlog)) as c:
                tools = await c.list_tools()
                names = {t.name for t in getattr(tools, "tools", tools)}
                r1 = await c.call_tool("shield_list_documents", {"dossier": code})
                r2 = await c.call_tool("shield_read_document", {"dossier": code, "document": "DOC_1"})
            return names, "\n".join(getattr(x, "text", "") for r in (r1, r2) for x in r.content)

        names, text = asyncio.run(asyncio.wait_for(go(), timeout=90))
        errlog.seek(0)
        child_stderr = errlog.read()
    if checker.find(text) or checker.find(child_stderr):
        return "LEAK", "mcp stdio", "valeur visible"
    if len(names) != 6:
        return "ERROR", "mcp stdio", "liste d'outils inattendue"
    return "CLEAN", "mcp stdio (processus séparé)", ""


def _verify_desk(code: str, folder: Path, case: dict, checker: LeakChecker) -> list[tuple[str, str, str]]:
    try:
        from .desk import app as desk_app
    except ImportError:
        return [("SKIP", "desk", "Shield Desk absent de cette installation")]
    factory = getattr(desk_app, "create_app", None)
    if factory is None:
        return [("SKIP", "desk", "create_app() absent")]
    try:
        from starlette.testclient import TestClient
    except Exception:  # noqa: BLE001
        return [("SKIP", "desk", "starlette.testclient indisponible")]
    out: list[tuple[str, str, str]] = []
    port = config.DESK_PORT
    token = "verify-" + os.urandom(12).hex()
    try:
        app = factory(port=port, token=token)
        base = f"http://127.0.0.1:{port}"
        with TestClient(app, base_url=base) as client:
            r = client.get("/api/dossiers")
            out.append(("CLEAN" if r.status_code in (401, 403) else "LEAK", "desk sans jeton (refusé)",
                        "" if r.status_code in (401, 403) else f"HTTP {r.status_code}"))
            r = client.get("/api/dossiers", headers={"X-Shield-Token": token, "Host": "evil.example"})
            out.append(("CLEAN" if r.status_code == 403 else "LEAK", "desk Host étranger (refusé)",
                        "" if r.status_code == 403 else f"HTTP {r.status_code}"))
            r = client.options("/api/mask", headers={"Origin": "https://evil.example",
                                                     "Access-Control-Request-Method": "POST"})
            allowed = r.headers.get("access-control-allow-origin", "")
            out.append(("LEAK" if allowed in ("*", "https://evil.example") else "CLEAN", "desk CORS site web (refusé)",
                        ""))
            h = {"X-Shield-Token": token}
            ext = {**h, "Origin": "chrome-extension://abcdefghijklmnopabcdefghijklmnop"}
            r = client.get(f"/api/dossiers/{code}/entities", headers=ext)
            out.append(("CLEAN" if r.status_code == 403 and not checker.find(r.text) else "LEAK",
                        "desk extension : entités (refusé)", "" if r.status_code == 403 else f"HTTP {r.status_code}"))
            r = client.get("/api/dossiers", headers=ext)
            out.append(("CLEAN" if r.status_code == 200 and str(folder) not in r.text else "LEAK",
                        "desk extension : dossiers sans chemin", ""))
            r = client.post("/api/mask", json={"text": case["files"]["assignation.txt"], "dossier": code}, headers=h)
            masked = (r.json() or {}).get("masked") if r.status_code == 200 else None
            if masked is None:
                out.append(("ERROR", "desk POST /api/mask", f"HTTP {r.status_code}"))
            else:
                out.append(("LEAK" if checker.find(masked) else "CLEAN", "desk POST /api/mask", ""))
            data = (folder / "assignation.txt").read_bytes()
            r = client.post("/api/files", files={"files": ("assignation.txt", data, "text/plain")},
                            data={"dossier": code}, headers=h)
            if r.status_code != 200:
                out.append(("ERROR", "desk POST /api/files", f"HTTP {r.status_code}"))
            else:
                items = r.json() if isinstance(r.json(), list) else (r.json() or {}).get("files", [])
                shown = [{"masked_name": i.get("masked_name"), "masked_text": i.get("masked_text")} for i in items]
                out.append(("LEAK" if checker.find(shown) else "CLEAN", "desk POST /api/files", ""))
    except Exception as exc:  # noqa: BLE001
        out.append(("ERROR", "desk", type(exc).__name__))
    return out


def cmd_verify(args: argparse.Namespace) -> int:
    print("RAGbase Shield verify : données inventées, dossier temporaire (votre coffre n'est pas touché).")

    def report(status: str, path: str, detail: str) -> None:
        print(f"{status:<5}  {path}" + (f" : {detail}" if detail else ""))

    counts = run_verify(report)
    total = counts["CLEAN"] + counts["LEAK"] + counts["ERROR"]
    if counts["LEAK"]:
        print(f"Résultat : LEAK ({counts['LEAK']} chemin(s) sur {total}). N'utilisez pas Shield et signalez-le "
              "(SECURITY.md).")
        return 1
    if counts["ERROR"]:
        print(f"Résultat : ERREUR ({counts['ERROR']} chemin(s) non vérifié(s) sur {total}).")
        return 2
    print(f"Résultat : CLEAN ({total} chemins testés" + (f", {counts['SKIP']} ignoré(s)" if counts["SKIP"] else "")
          + ").")
    return 0


# ------------------------------------------------------------------------------------------ parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ragbase-shield",
                                description="RAGbase Shield : pseudonymisation locale et réversible des dossiers "
                                            "clients (local, reversible pseudonymisation).")
    p.add_argument("--version", action="version", version=f"ragbase-shield {__version__}")
    sub = p.add_subparsers(dest="command", metavar="<commande>")
    sub.required = True

    s = sub.add_parser("add", help="enregistrer un dossier, afficher DOSSIER_n, l'analyser")
    s.add_argument("folder")
    s.add_argument("--title", default=None, help="titre, par exemple « Dupont c. ACME SAS »")
    s.set_defaults(func=cmd_add)

    s = sub.add_parser("list", help="lister les dossiers (titres réels : terminal local)")
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("entities", help="éléments détectés et enregistrés d'un dossier")
    s.add_argument("dossier", metavar="DOSSIER_n")
    s.set_defaults(func=cmd_entities)

    s = sub.add_parser("register", help="ajouter un nom ou un identifiant")
    s.add_argument("dossier", metavar="DOSSIER_n")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--person", metavar="NAME")
    g.add_argument("--org", metavar="NAME")
    g.add_argument("--kind", nargs=2, metavar=("KIND", "VALUE"))
    s.set_defaults(func=cmd_register)

    s = sub.add_parser("keep-clear", help="garder un élément en clair (ou --off)")
    s.add_argument("dossier", metavar="DOSSIER_n")
    s.add_argument("code", metavar="CODE")
    s.add_argument("--off", action="store_true")
    s.set_defaults(func=cmd_keep_clear)

    s = sub.add_parser("mask", help="masquer un texte (fichier, entrée standard ou presse-papiers)")
    s.add_argument("--dossier", metavar="D", default=None)
    s.add_argument("--clipboard", action="store_true")
    s.add_argument("--mode", choices=("codes", "synthetic"), default=None,
                   help="codes (PERSONNE_A) ou synthetic (pseudonymes réalistes) ; défaut : réglage du dossier")
    s.add_argument("--synthetic", action="store_true", help="raccourci de --mode synthetic")
    s.add_argument("file", nargs="?", metavar="FILE|-")
    s.set_defaults(func=cmd_mask)

    s = sub.add_parser("set", help="régler un dossier : --replacement codes|synthetic, --language auto|fr|en")
    s.add_argument("dossier", metavar="DOSSIER_n")
    s.add_argument("--replacement", choices=("codes", "synthetic", "default"), default=None)
    s.add_argument("--language", choices=("auto", "fr", "en", "default"), default=None)
    s.set_defaults(func=cmd_set)

    s = sub.add_parser("ner", help="modèle NER optionnel (v0.2) : status, download, install")
    s.add_argument("action", choices=("status", "download", "install"))
    s.add_argument("--model", choices=("fastino", "urchade", "all"), default="fastino")
    s.add_argument("--dir", default=None, help="dossier des modèles (défaut : RAGBASE_SHIELD_MODELS_DIR ou <home>/models)")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_ner)

    s = sub.add_parser("restore", help="restaurer les vrais noms (terminal local)")
    s.add_argument("--dossier", metavar="D", required=True)
    s.add_argument("--clipboard", action="store_true")
    s.add_argument("file", nargs="?", metavar="FILE|-")
    s.set_defaults(func=cmd_restore)

    s = sub.add_parser("desk", help="ouvrir Shield Desk (page locale)")
    s.add_argument("--port", type=int, default=config.DESK_PORT)
    s.add_argument("--no-browser", action="store_true")
    s.add_argument("--service", action="store_true",
                   help="mode service d'arrière-plan : n'ouvre pas le navigateur et n'affiche jamais le jeton")
    s.set_defaults(func=cmd_desk)

    s = sub.add_parser("service", help="Shield Desk en arrière-plan (installateurs) : install, uninstall, status, "
                                       "start, stop, restart, enable, disable, run")
    s.add_argument("action", choices=("install", "uninstall", "status", "start", "stop", "restart", "enable",
                                      "disable", "run"))
    s.add_argument("--no-start", action="store_true", help="install : ne pas démarrer le service maintenant")
    s.add_argument("--no-login-item", action="store_true", help="install : pas d'ouverture de session")
    s.add_argument("--no-bundle", action="store_true", help="install (macOS) : sans l'application RAGbase Shield.app")
    s.add_argument("--port", type=int, default=None, help="install : port de Shield Desk (défaut 7878)")
    s.add_argument("--quiet", action="store_true")
    s.add_argument("--purge-extras", action="store_true", help="uninstall : supprimer aussi les paquets NER")
    s.add_argument("--purge-data", action="store_true",
                   help="uninstall : supprimer aussi le dossier de données (coffres, clé) : irréversible")
    s.add_argument("--no-tray", action="store_true", help="run : sans icône (barre des menus ou zone de notification)")
    s.add_argument("--for-seconds", type=float, default=None, help="run : s'arrêter après N secondes (tests)")
    s.set_defaults(func=cmd_service)

    s = sub.add_parser("mcp", help="serveur MCP stdio (utilisé par les applications)")
    s.set_defaults(func=cmd_mcp)

    from .hooks import EVENTS

    s = sub.add_parser("hook", help="hooks Claude Code (JSON sur stdin et stdout)")
    s.add_argument("event", choices=EVENTS)
    s.set_defaults(func=cmd_hook)

    s = sub.add_parser("install", help="brancher Shield dans une application")
    s.add_argument("app", choices=APPS)
    s.add_argument("--dry-run", action="store_true", help="montrer ce qui serait modifié, sans rien écrire")
    s.set_defaults(func=cmd_install)

    s = sub.add_parser("doctor", help="vérifier l'installation, la clé et les intégrations")
    s.set_defaults(func=cmd_doctor)

    s = sub.add_parser("verify", help="tester chaque chemin avec des données inventées : CLEAN ou LEAK")
    s.set_defaults(func=cmd_verify)

    s = sub.add_parser("export", help="exporter un dossier chiffré pour un collègue")
    s.add_argument("dossier", metavar="DOSSIER_n")
    s.add_argument("file", metavar="FILE")
    s.set_defaults(func=cmd_export)

    s = sub.add_parser("import", help="importer un dossier exporté")
    s.add_argument("file", metavar="FILE")
    s.add_argument("--folder", default=None, help="dossier local des pièces (facultatif)")
    s.set_defaults(func=cmd_import)

    s = sub.add_parser("journal", help="journal local (codes et nombres)")
    s.add_argument("dossier", nargs="?", metavar="DOSSIER_n")
    s.set_defaults(func=cmd_journal)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point of ``ragbase-shield`` (reads ``sys.argv`` when ``argv`` is None)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command not in ("mcp", "hook"):
        _utf8_stdio()
    if not logging.getLogger().handlers:
        logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        return int(args.func(args) or 0)
    except UsageError as exc:
        _err(f"Erreur : {exc}")
        return 2
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - never print a traceback that could hold a value
        # Imported here, not at the top: the hook commands must start fast (no engine import).
        from .clipboard import ClipboardError
        from .service import ShieldError
        from .vault import VaultError

        if isinstance(exc, (ShieldError, ClipboardError, VaultError)):
            _err(f"Erreur : {exc}")
        else:
            log.debug("cli failure", exc_info=False)
            _err(f"Erreur interne ({type(exc).__name__}).")
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
