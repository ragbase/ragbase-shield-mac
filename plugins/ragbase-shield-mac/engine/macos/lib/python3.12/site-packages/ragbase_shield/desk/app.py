"""Shield Desk: Starlette app on 127.0.0.1 (ARCHITECTURE.md sections 4.7 and 6).

The Desk is a LOCAL page for the user. It may show real values (entities, restored text), so the whole
surface is locked to this computer and to this install:

- ``serve()`` binds ONLY to 127.0.0.1 (the address is not configurable).
- Every route checks the Host header: ``127.0.0.1:<port>`` or ``localhost:<port>``. Anything else (another
  port, a DNS-rebinding name such as ``evil.example:7878``, ``[::1]``) gets 403.
- Every ``/api/*`` route needs the per-install token in the ``X-Shield-Token`` header. The token lives in
  ``<home>/desk_token`` (0600, created on first run). ``?t=<token>`` is only read by the page itself (it
  stores it and removes it from the address bar); the API never accepts it.
- A request that carries an ``Origin`` must come from the Desk page itself or from a browser extension
  (``chrome-extension://``, ``moz-extension://``); any web origin gets 403. CORS headers are sent only to an
  extension origin, and only with a valid token (the OPTIONS preflight cannot carry the token, so it is
  answered for extension origins only and holds no data).
- Real values (``found[].value`` in ``POST /api/mask``) are returned to the Desk page only, never to an
  extension origin. A browser extension gets the least it needs: ``GET /api/status``, ``GET /api/dossiers``
  (code, title and counts, never the folder path), ``POST /api/mask`` and ``POST /api/restore`` (the restored
  text is shown in the extension's own side panel, never in a provider page). Every other route answers 403
  to an extension origin (entities, keep-clear, journal, files, scan, add and remove a dossier).
- Bodies are capped (JSON 16 MB, uploads 25 MB per file and 100 MB per request). Uploaded files go to a
  private temporary folder inside the Shield data folder, under a fixed name (``upload-N.<ext>``): the
  client file name is never used as a path. They are deleted as soon as they are masked.
- Errors are JSON ``{"error": "<French message>", "code": "<short code>"}`` with codes only: no stack
  trace, no value, no path. Failures are logged locally with their type and code location only.
- Security headers on every response: a strict CSP (``default-src 'self'``, ``frame-ancestors 'none'``...),
  ``Referrer-Policy: no-referrer``, ``X-Content-Type-Options: nosniff``, ``X-Frame-Options: DENY``, and
  ``Cache-Control: no-store`` on ``/api/*``.
"""
from __future__ import annotations

import functools
import hmac
import json
import logging
import os
import re
import secrets
import shutil
import socket
import sys
import tempfile
import threading
import time
import traceback
import unicodedata
import uuid as uuidlib
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers, MutableHeaders, UploadFile
from starlette.exceptions import HTTPException
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.middleware import Middleware
from starlette.requests import ClientDisconnect, Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from .. import __version__, config, documents, service
from ..service import ShieldError

log = logging.getLogger(__name__)

__all__ = [
    "create_app",
    "serve",
    "run",
    "load_or_create_token",
    "token_path",
    "desk_url",
    "TOKEN_FILE",
    "MAX_FILE_BYTES",
    "MAX_FILES_PER_REQUEST",
    "MAX_UPLOAD_BYTES",
    "MAX_JSON_BYTES",
]

STATIC_DIR = Path(__file__).resolve().parent / "static"
TOKEN_FILE = "desk_token"
UPLOAD_DIR = "desk-uploads"
BIND_HOST = "127.0.0.1"  # never configurable: the Desk is reachable from this computer only

MB = 1024 * 1024
MAX_FILE_BYTES = 25 * MB          # per uploaded file
MAX_FILES_PER_REQUEST = 20
MAX_UPLOAD_BYTES = 100 * MB + MB  # whole multipart body (100 MB of files + form overhead)
MAX_JSON_BYTES = 16 * MB          # service.mask_text accepts up to 5 million characters
MAX_OTHER_BYTES = 64 * 1024       # GET, DELETE, scan, page and static routes
MAX_FOLDER_CHARS = 4096
MAX_VALUE_CHARS = 500
MAX_JOURNAL_LIMIT = 5000

_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]{32,256}")
_TOKEN_ARG_RE = re.compile(r"[\x21-\x7e]{16,256}")  # create_app(token=...): printable ASCII, no space
_EXT_ORIGIN_RE = re.compile(r"(?:chrome|moz)-extension://[A-Za-z0-9._\-]{1,128}")
_DOSSIER_RE = re.compile(r"DOSSIER_[0-9]{1,6}")
# What a browser extension may call (least privilege): method, path.
_EXTENSION_ROUTES = {("GET", "/api/status"), ("GET", "/api/dossiers"), ("POST", "/api/mask"),
                     ("POST", "/api/restore")}

CSP = ("default-src 'self'; img-src 'self' data:; script-src 'self'; style-src 'self'; connect-src 'self'; "
       "object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
_SECURITY_HEADERS = (
    ("content-security-policy", CSP),
    ("referrer-policy", "no-referrer"),
    ("x-content-type-options", "nosniff"),
    ("x-frame-options", "DENY"),
    ("cross-origin-opener-policy", "same-origin"),
    ("permissions-policy", "camera=(), microphone=(), geolocation=(), payment=()"),
)

# Status of a ShieldError by its kind (service.ShieldError.kind).
_STATUS_BY_KIND = {
    "invalid_argument": 400,
    "unknown_dossier": 404,
    "unknown_document": 404,
    "document_unreadable": 422,
    "vault_error": 500,
    "write_failed": 500,
    "internal": 500,
}

# Friendlier French messages for document reader codes (documents.DocumentError) in "Fichiers".
_FILE_ERRORS = {
    "pdf_encrypted": "PDF protégé par un mot de passe : retirez la protection, puis réessayez.",
    "unsupported_format": "Format non pris en charge : utilisez PDF, DOCX, TXT, MD, EML ou HTML.",
    "file_not_found": "Fichier introuvable pendant la lecture. Réessayez.",
}

_HTTP_MESSAGES = {
    400: ("Requête invalide.", "bad_request"),
    404: ("Adresse inconnue sur Shield Desk.", "not_found"),
    405: ("Méthode non autorisée pour cette adresse.", "method_not_allowed"),
    413: ("Envoi trop volumineux.", "too_large"),
}


# ------------------------------------------------------------------------------------------ token

def token_path(home: Optional[Path] = None) -> Path:
    """``<home>/desk_token``."""
    return Path(home if home is not None else config.home()) / TOKEN_FILE


def _open_permissions(path: Path) -> bool:
    """True when the token file can be read by other users (POSIX only)."""
    if sys.platform.startswith("win"):
        return False
    try:
        return bool(path.stat().st_mode & 0o077)
    except OSError:
        return False


def _write_token(path: Path, token: str, *, exclusive: bool) -> bool:
    """Write the token with mode 0600. ``exclusive``: only if the file does not exist yet (first run wins)."""
    data = (token + "\n").encode("ascii")
    if exclusive:
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return False
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        return True
    tmp = path.with_name(f".{TOKEN_FILE}.{uuidlib.uuid4().hex}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return True


def load_or_create_token(home: Optional[Path] = None) -> str:
    """The per-install Desk token from ``<home>/desk_token`` (0600), created on first run.

    A missing, empty or damaged file gets a new random token. A file that other users can read gets a new
    token too, because the old one may have been seen (the browser extension must then be paired again).
    """
    path = token_path(home)
    for _attempt in range(3):
        try:
            existing = path.read_text(encoding="ascii").strip()
        except FileNotFoundError:
            existing = None
        except (OSError, UnicodeDecodeError):
            existing = ""
        if existing is None:
            token = secrets.token_urlsafe(32)
            if _write_token(path, token, exclusive=True):
                return token
            continue  # another Desk created it at the same moment: read theirs
        if _TOKEN_RE.fullmatch(existing) and not _open_permissions(path):
            return existing
        if existing and _TOKEN_RE.fullmatch(existing):
            log.warning("RAGbase Shield: the Desk token file was readable by other users; a new token was created")
        else:
            log.warning("RAGbase Shield: the Desk token file was damaged; a new token was created")
        token = secrets.token_urlsafe(32)
        _write_token(path, token, exclusive=False)
        return token
    raise RuntimeError("desk token: could not create the token file")


def desk_url(port: int, token: Optional[str] = None) -> str:
    """``http://127.0.0.1:<port>/`` (with ``?t=<token>`` when given: local browser only, never printed to an agent)."""
    base = f"http://{BIND_HOST}:{port}/"
    return f"{base}?t={token}" if token else base


# ------------------------------------------------------------------------------------------ errors

class _HttpError(Exception):
    """An expected client error with a fixed French message (never holds a value)."""

    def __init__(self, status: int, message: str, code: str):
        super().__init__(code)
        self.status = status
        self.message = message
        self.code = code


class _TooLarge(Exception):
    """The request body went over the limit of its route."""


def _error(status: int, message: str, code: str, headers: Optional[dict] = None) -> JSONResponse:
    return JSONResponse({"error": message, "code": code}, status_code=status, headers=headers)


def _log_failure(op: str, exc: BaseException) -> str:
    """Log a failure locally with its type and code location only (the message may hold a value)."""
    ref = uuidlib.uuid4().hex[:8]
    frames = traceback.extract_tb(exc.__traceback__)[-4:]
    where = " < ".join(f"{Path(f.filename).name}:{f.lineno}:{f.name}" for f in reversed(frames))
    log.warning("RAGbase Shield Desk: %s failed (ref %s): %s at %s", op, ref, type(exc).__name__, where or "?")
    return ref


def _shield_error(exc: ShieldError) -> JSONResponse:
    kind = getattr(exc, "kind", "error") or "error"
    status = _STATUS_BY_KIND.get(kind, 500 if kind in ("internal", "error") else 400)
    message = str(exc)
    if kind == "unknown_dossier":
        message = "Dossier inconnu : il a peut-être été retiré. Rechargez la liste des dossiers."
    return _error(status, message, kind)


Endpoint = Callable[[Request], Awaitable[Response]]


def _endpoint(op: str) -> Callable[[Endpoint], Endpoint]:
    """Wrap an endpoint: every failure becomes a French JSON error with codes only (fail closed)."""

    def deco(fn: Endpoint) -> Endpoint:
        @functools.wraps(fn)
        async def wrapper(request: Request) -> Response:
            try:
                return await fn(request)
            except ShieldError as exc:
                return _shield_error(exc)
            except _HttpError as exc:
                return _error(exc.status, exc.message, exc.code)
            except _TooLarge:
                return _too_large(request.url.path)
            except ClientDisconnect:
                return _error(400, "Envoi interrompu.", "client_disconnect")
            except Exception as exc:  # noqa: BLE001 - never a stack trace or a message to the client
                ref = _log_failure(op, exc)
                return _error(500, f"Erreur interne de Shield Desk (réf. {ref}).", "internal")

        return wrapper

    return deco


def _too_large(path: str) -> JSONResponse:
    if path == "/api/files":
        return _error(413, "Envoi trop volumineux : 25 Mo maximum par fichier et 100 Mo par envoi.", "too_large")
    if path.startswith("/api/"):
        return _error(413, "Texte trop long : 16 Mo maximum par envoi.", "too_large")
    return _error(413, "Requête trop volumineuse.", "too_large")


async def _http_exception(request: Request, exc: Exception) -> Response:
    status = getattr(exc, "status_code", 400)
    message, code = _HTTP_MESSAGES.get(status, ("Requête refusée.", "refused"))
    headers = dict(getattr(exc, "headers", None) or {})
    allow = {k: v for k, v in headers.items() if k.lower() == "allow"}
    return _error(status, message, code, headers=allow or None)


# ------------------------------------------------------------------------------------------ guard

def _body_limit(method: str, path: str) -> int:
    if method == "POST" and path == "/api/files":
        return MAX_UPLOAD_BYTES
    if method in ("POST", "PUT", "PATCH") and path.startswith("/api/"):
        return MAX_JSON_BYTES
    return MAX_OTHER_BYTES


class _Secret:
    """Holds the Desk token; its repr never shows it (Starlette prints middleware arguments in reprs)."""

    __slots__ = ("_value",)

    def __init__(self, value: str):
        self._value = value.encode("utf-8")

    def __repr__(self) -> str:
        return "<hidden>"

    def matches(self, given: bytes) -> bool:
        return hmac.compare_digest(given, self._value)


class _Guard:
    """Outermost ASGI layer: Host, Origin, token, CORS, body size and security headers."""

    def __init__(self, app: Any, *, port: int, secret: _Secret):
        self.app = app
        self.port = port
        self._secret = secret
        self.hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        self.same_origins = {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}

    def __repr__(self) -> str:  # never show the token
        return f"<Shield Desk guard port={self.port}>"

    def _token_ok(self, raw_headers: list) -> Optional[bool]:
        """None: no token header; True/False: valid or not (constant-time compare)."""
        given = None
        for k, v in raw_headers:
            if k.lower() == b"x-shield-token":
                given = v.strip()
                break
        if given is None or not given:
            return None
        return self._secret.matches(given)

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] == "lifespan":
            await self.app(scope, receive, send)
            return
        if scope["type"] != "http":  # no websockets
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
            return

        headers = Headers(scope=scope)
        path: str = scope.get("path", "") or "/"
        method: str = scope.get("method", "GET").upper()
        is_api = path == "/api" or path.startswith("/api/")
        extra: list[tuple[str, str]] = []

        async def reply(response: Response) -> None:
            await response(scope, receive, secured_send)

        async def secured_send(message: dict) -> None:
            if message["type"] == "http.response.start":
                h = MutableHeaders(scope=message)
                for k, v in _SECURITY_HEADERS:
                    if k not in h:
                        h[k] = v
                if is_api:
                    h["cache-control"] = "no-store"
                    h["vary"] = "Origin"
                elif "cache-control" not in h:
                    h["cache-control"] = "no-cache"
                for k, v in extra:
                    h[k] = v
            await send(message)

        # 1. Host check, on every route (stops DNS rebinding and other ports).
        host = headers.get("host", "").strip().lower()
        if host not in self.hosts:
            await reply(_error(403, "Adresse refusée : ouvrez Shield Desk sur http://127.0.0.1:"
                                    f"{self.port}/ depuis cet ordinateur.", "bad_host"))
            return

        # 2. Origin check: the Desk page itself or a browser extension, never a web site.
        origin = headers.get("origin")
        ext_origin = origin if origin is not None and _EXT_ORIGIN_RE.fullmatch(origin) else None
        if origin is not None and ext_origin is None and origin not in self.same_origins:
            await reply(_error(403, "Origine refusée : seule la page Shield Desk et l'extension RAGbase Shield "
                                    "peuvent appeler Shield Desk.", "bad_origin"))
            return
        fetch_site = headers.get("sec-fetch-site", "").lower()
        if is_api and ext_origin is None and fetch_site in ("cross-site", "same-site"):
            await reply(_error(403, "Requête refusée : elle vient d'un autre site.", "bad_origin"))
            return

        # 3. CORS preflight (it never carries the token): answered for extension origins only, no data.
        if method == "OPTIONS" and origin is not None:
            if ext_origin is not None and is_api and headers.get("access-control-request-method"):
                pre = [("access-control-allow-origin", ext_origin),
                       ("access-control-allow-methods", "GET, POST, DELETE, OPTIONS"),
                       ("access-control-allow-headers", "Content-Type, X-Shield-Token"),
                       ("access-control-max-age", "600")]
                if headers.get("access-control-request-private-network", "").lower() == "true":
                    pre.append(("access-control-allow-private-network", "true"))
                extra.extend(pre)
                await reply(Response(status_code=204))
            else:
                await reply(_error(403, "Requête refusée.", "bad_origin"))
            return

        # 4. Token, on every /api/* route (header only: ?t= is for the page, never for the API).
        if is_api:
            ok = self._token_ok(scope.get("headers") or [])
            if ok is None:
                await reply(_error(401, "Le jeton est manquant ou invalide. Ouvrez Shield Desk avec la commande "
                                        "« uvx ragbase-shield desk » (le lien affiché contient le jeton).",
                                   "token_missing"))
                return
            if not ok:
                await reply(_error(403, "Le jeton est manquant ou invalide. Ouvrez Shield Desk avec la commande "
                                        "« uvx ragbase-shield desk » (le lien affiché contient le jeton).",
                                   "token_invalid"))
                return
            if ext_origin is not None:
                if (method, path.rstrip("/") or "/") not in _EXTENSION_ROUTES:
                    await reply(_error(403, "Cette adresse de Shield Desk est réservée à la page Shield Desk : "
                                            "l'extension n'y a pas accès.", "extension_forbidden"))
                    return
                extra.append(("access-control-allow-origin", ext_origin))

        # 5. Body size: Content-Length first, then a counting receive for chunked bodies.
        limit = _body_limit(method, path)
        length = headers.get("content-length")
        if length is not None:
            declared = int(length) if length.strip().isdigit() else -1
            if declared < 0:
                await reply(_error(400, "En-tête Content-Length invalide.", "bad_request"))
                return
            if declared > limit:
                await reply(_too_large(path))
                return
        received = 0

        async def limited_receive() -> dict:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise _TooLarge()
            return message

        started = False

        async def tracking_send(message: dict) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await secured_send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _TooLarge:
            if not started:
                await reply(_too_large(path))
        except Exception as exc:  # noqa: BLE001 - last net: JSON, codes only, no stack trace
            ref = _log_failure("request", exc)
            if not started:
                await reply(_error(500, f"Erreur interne de Shield Desk (réf. {ref}).", "internal"))


# ------------------------------------------------------------------------------------------ request helpers

async def _json_body(request: Request) -> dict:
    ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if ctype != "application/json":
        raise _HttpError(415, "Envoyez les données en JSON (Content-Type: application/json).", "unsupported_media")
    raw = await request.body()
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        raise _HttpError(400, "Corps JSON invalide.", "invalid_json") from None
    if not isinstance(data, dict):
        raise _HttpError(400, "Le corps de la requête doit être un objet JSON.", "invalid_json")
    return data


def _text(data: dict, key: str, *, required: bool, max_chars: Optional[int] = None) -> Optional[str]:
    value = data.get(key)
    if value is None:
        if required:
            raise _HttpError(400, f"Le champ « {key} » est obligatoire.", "missing_field")
        return None
    if not isinstance(value, str):
        raise _HttpError(400, f"Le champ « {key} » doit être un texte.", "invalid_field")
    if max_chars is not None and len(value) > max_chars:
        raise _HttpError(400, f"Le champ « {key} » est trop long.", "invalid_field")
    return value


def _dossier_arg(value: Any, *, required: bool) -> Optional[str]:
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise _HttpError(400, "Choisissez un dossier.", "missing_dossier")
        return None
    if not isinstance(value, str):
        raise _HttpError(400, "Code de dossier invalide : attendu DOSSIER_n.", "invalid_dossier")
    code = value.strip().upper()
    if not _DOSSIER_RE.fullmatch(code):
        raise _HttpError(400, "Code de dossier invalide : attendu DOSSIER_n.", "invalid_dossier")
    return code


def _from_extension(request: Request) -> bool:
    origin = request.headers.get("origin")
    return bool(origin and _EXT_ORIGIN_RE.fullmatch(origin))


def _journal(event: str, dossier: Optional[str] = None, **counts: Any) -> None:
    """Desk-level journal entry (codes and counts only). A journal failure never fails the request."""
    try:
        service.default().registry.journal(event, dossier, **counts)
    except Exception as exc:  # noqa: BLE001
        _log_failure("journal", exc)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _check_folder(raw: Any) -> Path:
    """The dossier folder from POST /api/dossiers, checked before the vault sees it (French errors)."""
    if not isinstance(raw, str) or not raw.strip():
        raise _HttpError(400, "Indiquez le chemin du dossier client.", "missing_field")
    s = raw.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    if not s or len(s) > MAX_FOLDER_CHARS or "\x00" in s:
        raise _HttpError(400, "Chemin de dossier invalide.", "invalid_folder")
    p = Path(s).expanduser()
    if not p.is_absolute():
        raise _HttpError(400, "Indiquez le chemin complet du dossier (il commence par / sur Mac et Linux, "
                              "ou par une lettre comme C:\\ sous Windows).", "invalid_folder")
    try:
        p = p.resolve(strict=True)
    except (OSError, RuntimeError):
        raise _HttpError(400, "Ce dossier n'existe pas sur cet ordinateur.", "folder_not_found") from None
    if not p.is_dir():
        raise _HttpError(400, "Ce chemin n'est pas un dossier.", "not_a_folder")
    shield_home = config.home().resolve()
    try:
        user_home = Path.home().resolve()
    except (OSError, RuntimeError):
        user_home = None
    too_wide = p.parent == p or (user_home is not None and _is_within(user_home, p))
    if too_wide or _is_within(shield_home, p) or _is_within(p, shield_home):
        raise _HttpError(400, "Choisissez le dossier d'un client : pas la racine du disque, pas votre dossier "
                              "personnel, pas le dossier de données de RAGbase Shield.", "folder_too_wide")
    return p


# ------------------------------------------------------------------------------------------ uploads

class _UploadParser(MultiPartParser):
    """Keeps each uploaded file in memory up to the per-file limit (no named temporary file in the system temp
    folder). A bigger file spills to an anonymous temporary file and is then refused."""

    spool_max_size = MAX_FILE_BYTES + 1


def _client_label(filename: Optional[str]) -> str:
    """The user's file name as a label only: base name, no control characters, never used as a path."""
    s = unicodedata.normalize("NFC", str(filename or ""))
    s = re.split(r"[\\/]", s)[-1]
    s = "".join(" " if unicodedata.category(c).startswith("C") else c for c in s)
    s = " ".join(s.split()).strip()
    if s in ("", ".", ".."):
        return "document"
    return s[:255]


def _upload_root() -> Path:
    root = config.home() / UPLOAD_DIR
    root.mkdir(mode=0o700, exist_ok=True)
    try:
        os.chmod(root, 0o700)
    except OSError:
        pass
    return root


def _clean_stale_uploads(max_age_s: float = 3600.0) -> None:
    """Remove upload folders left by a crash (older than one hour)."""
    root = config.home() / UPLOAD_DIR
    if not root.is_dir():
        return
    now = time.time()
    for child in root.iterdir():
        try:
            if child.is_dir() and not child.is_symlink() and now - child.stat().st_mtime > max_age_s:
                shutil.rmtree(child, ignore_errors=True)
        except OSError:
            continue


def _file_error(exc: ShieldError) -> str:
    msg = str(exc)
    m = re.search(r"\(([a-z_]{3,40})(?::[0-9]+)?\)", msg)
    if exc.kind == "document_unreadable" and m and m.group(1) in _FILE_ERRORS:
        return _FILE_ERRORS[m.group(1)]
    if exc.kind == "document_unreadable":
        return "Fichier illisible : il est peut-être abîmé ou protégé."
    return msg


def _upload_size(up: UploadFile) -> int:
    if isinstance(up.size, int):
        return up.size
    fh = up.file
    pos = fh.tell()
    fh.seek(0, os.SEEK_END)
    size = fh.tell()
    fh.seek(pos)
    return size


def _mask_upload(up: UploadFile, index: int, dossier: Optional[str], workdir: Path) -> dict:
    """Mask one uploaded file. Runs in a worker thread. Returns the result item for POST /api/files."""
    label = _client_label(up.filename)
    ext = os.path.splitext(label)[1].lower()
    if ext not in documents.SUPPORTED_EXTENSIONS:
        return {"name": label, "masked_name": "", "masked_text": "", "error": _FILE_ERRORS["unsupported_format"]}
    if _upload_size(up) > MAX_FILE_BYTES:
        return {"name": label, "masked_name": "", "masked_text": "",
                "error": "Fichier trop volumineux : 25 Mo maximum par fichier."}
    target = workdir / f"upload-{index}{ext}"  # fixed name: the client file name is never a path
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as out:
            up.file.seek(0)
            shutil.copyfileobj(up.file, out, MB)
        result = service.mask_file(target, dossier, name=label)
    except ShieldError as exc:
        return {"name": label, "masked_name": "", "masked_text": "", "error": _file_error(exc)}
    finally:
        try:
            target.unlink()
        except OSError:
            pass
    return {"name": label, "masked_name": result.get("masked_name", ""),
            "masked_text": result.get("masked_text", ""), "needs_ocr": bool(result.get("needs_ocr")),
            "warnings": list(result.get("warnings") or [])}


# ------------------------------------------------------------------------------------------ endpoints

async def page(request: Request) -> Response:
    """The Desk page (no data in it: served without the token so a reload works)."""
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html; charset=utf-8")


@_endpoint("status")
async def api_status(request: Request) -> Response:
    st = await run_in_threadpool(service.status)
    return JSONResponse(st)


@_endpoint("dossiers")
async def api_dossiers(request: Request) -> Response:
    if request.method == "GET":
        rows = await run_in_threadpool(service.list_dossiers, True)
        if _from_extension(request):  # the extension shows the titles; it never needs the folder path
            rows = [{k: v for k, v in r.items() if k != "folder"} for r in rows]
        return JSONResponse(rows)
    data = await _json_body(request)
    folder = _check_folder(data.get("folder"))
    title = _text(data, "title", required=False, max_chars=service.MAX_TITLE_CHARS)
    title = " ".join(title.split()) if title else None
    summary = await run_in_threadpool(service.add_dossier, folder, title)
    rows = await run_in_threadpool(service.list_dossiers, True)
    row = next((r for r in rows if r.get("code") == summary.get("code")), {})
    out = dict(summary)
    out.update(title=row.get("title"), folder=row.get("folder"))
    return JSONResponse(out, status_code=201)


@_endpoint("remove_dossier")
async def api_dossier(request: Request) -> Response:
    code = _dossier_arg(request.path_params.get("code"), required=True)
    await run_in_threadpool(service.remove_dossier, code)  # the vault journals "dossier_removed"
    return JSONResponse({"ok": True})


@_endpoint("entities")
async def api_entities(request: Request) -> Response:
    code = _dossier_arg(request.path_params.get("code"), required=True)
    if request.method == "GET":
        rows = await run_in_threadpool(service.entities, code)
        return JSONResponse(rows)
    data = await _json_body(request)
    kind = _text(data, "kind", required=True, max_chars=40)
    value = _text(data, "value", required=True, max_chars=MAX_VALUE_CHARS)
    if not value.strip():
        raise _HttpError(400, "Indiquez le nom ou le terme à ajouter.", "missing_field")
    ent = await run_in_threadpool(service.register, code, kind, value)
    return JSONResponse(ent, status_code=201)


@_endpoint("keep_clear")
async def api_keep_clear(request: Request) -> Response:
    code = _dossier_arg(request.path_params.get("code"), required=True)
    data = await _json_body(request)
    target = data.get("code") if data.get("code") is not None else data.get("term")
    if not isinstance(target, str) or not target.strip() or len(target) > MAX_VALUE_CHARS:
        raise _HttpError(400, "Indiquez le code (PERSONNE_A...) ou le terme à garder en clair.", "missing_field")
    value = data.get("value", True)
    if not isinstance(value, bool):
        raise _HttpError(400, "Le champ « value » doit valoir true ou false.", "invalid_field")
    res = await run_in_threadpool(service.keep_clear, code, target.strip(), value)
    return JSONResponse(res)


@_endpoint("scan")
async def api_scan(request: Request) -> Response:
    code = _dossier_arg(request.path_params.get("code"), required=True)
    res = await run_in_threadpool(service.scan_dossier, code)
    return JSONResponse(res)


@_endpoint("mask")
async def api_mask(request: Request) -> Response:
    data = await _json_body(request)
    text = _text(data, "text", required=True)
    dossier = _dossier_arg(data.get("dossier"), required=False)
    # v0.2: optional "replacement" ("codes" | "synthetic"); default = the dossier setting, else the global default.
    replacement = data.get("replacement", data.get("mode"))
    if replacement is not None and (not isinstance(replacement, str) or replacement.strip().lower()
                                    not in ("codes", "synthetic")):
        raise _HttpError(400, "Le champ « replacement » vaut codes ou synthetic.", "invalid_field")
    replacement = replacement.strip().lower() if isinstance(replacement, str) else None
    # Real values go to the Desk page only (review step), never to a browser extension.
    values = not _from_extension(request)
    res = await run_in_threadpool(service.mask_text, text, dossier, values, replacement)
    if dossier is None:
        _journal("mask", None, spans=sum(int(v) for v in (res.get("counts") or {}).values()))
    return JSONResponse({"masked": res["masked"], "found": res.get("found", []), "counts": res.get("counts", {}),
                         "mode": res.get("mode", "codes")})


@_endpoint("restore")
async def api_restore(request: Request) -> Response:
    data = await _json_body(request)
    text = _text(data, "text", required=True)
    if data.get("dossier") is None or (isinstance(data.get("dossier"), str) and not data["dossier"].strip()):
        raise _HttpError(400, "Choisissez le dossier qui a servi à masquer ce texte.", "missing_dossier")
    dossier = _dossier_arg(data.get("dossier"), required=True)
    res = await run_in_threadpool(service.restore_report, text, dossier)
    return JSONResponse({"text": res["text"], "unknown_codes": list(res.get("unknown_codes") or []),
                         "restored": int(res.get("restored") or 0),
                         "restored_synthetic": int(res.get("restored_synthetic") or 0),
                         "unmatched_synthetic": int(res.get("unmatched_synthetic") or 0)})


@_endpoint("files")
async def api_files(request: Request) -> Response:
    ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if ctype != "multipart/form-data":
        raise _HttpError(415, "Envoyez les fichiers en multipart/form-data (champ « files »).", "unsupported_media")
    parser = _UploadParser(request.headers, request.stream(), max_files=MAX_FILES_PER_REQUEST, max_fields=8,
                           max_part_size=4096)
    try:
        form = await parser.parse()
    except MultiPartException:
        raise _HttpError(400, f"Envoi de fichiers invalide ou trop de fichiers ({MAX_FILES_PER_REQUEST} au maximum "
                              "par envoi).", "bad_upload") from None
    workdir: Optional[Path] = None
    try:
        raw_dossier = form.get("dossier")
        dossier = _dossier_arg(raw_dossier if isinstance(raw_dossier, str) else None, required=False)
        uploads = [v for k, v in form.multi_items() if k in ("files", "file") and isinstance(v, UploadFile)]
        if not uploads:
            raise _HttpError(400, "Aucun fichier reçu.", "no_files")
        if dossier is not None:
            known = {r.get("code") for r in await run_in_threadpool(service.list_dossiers)}
            if dossier not in known:
                raise ShieldError("unknown dossier", "unknown_dossier")
        workdir = Path(tempfile.mkdtemp(prefix="upload-", dir=_upload_root()))
        results = []
        for i, up in enumerate(uploads, start=1):
            results.append(await run_in_threadpool(_mask_upload, up, i, dossier, workdir))
        _journal("files", dossier, files=len(results), errors=sum(1 for r in results if r.get("error")))
        return JSONResponse(results)
    finally:
        await form.close()
        if workdir is not None:
            shutil.rmtree(workdir, ignore_errors=True)


@_endpoint("journal")
async def api_journal(request: Request) -> Response:
    dossier = _dossier_arg(request.query_params.get("dossier"), required=False)
    raw_limit = request.query_params.get("limit")
    limit = 500
    if raw_limit is not None:
        if not raw_limit.isdigit() or not 1 <= int(raw_limit) <= MAX_JOURNAL_LIMIT:
            raise _HttpError(400, f"« limit » doit être un nombre entre 1 et {MAX_JOURNAL_LIMIT}.", "invalid_field")
        limit = int(raw_limit)
    rows = await run_in_threadpool(service.journal, dossier, limit)
    return JSONResponse(rows)


# ------------------------------------------------------------------------------------------ app

def _check_port(port: Any) -> int:
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("port must be an integer between 1 and 65535")
    return port


def create_app(*, port: int = config.DESK_PORT, token: Optional[str] = None) -> Starlette:
    """The Shield Desk Starlette app. ``token`` None reads or creates ``<home>/desk_token``.

    The app checks the Host header against ``127.0.0.1:<port>`` and ``localhost:<port>``, so drive it with
    ``TestClient(app, base_url=f"http://127.0.0.1:{port}")``.
    """
    port = _check_port(port)
    if token is None:
        token = load_or_create_token()
    if not isinstance(token, str) or not _TOKEN_ARG_RE.fullmatch(token):
        raise ValueError("token must be 16 to 256 printable ASCII characters without spaces")
    routes = [
        Route("/", page, methods=["GET"]),
        Mount("/static", app=StaticFiles(directory=str(STATIC_DIR)), name="static"),
        Route("/api/status", api_status, methods=["GET"]),
        Route("/api/dossiers", api_dossiers, methods=["GET", "POST"]),
        Route("/api/dossiers/{code}", api_dossier, methods=["DELETE"]),
        Route("/api/dossiers/{code}/entities", api_entities, methods=["GET", "POST"]),
        Route("/api/dossiers/{code}/keep-clear", api_keep_clear, methods=["POST"]),
        Route("/api/dossiers/{code}/scan", api_scan, methods=["POST"]),
        Route("/api/mask", api_mask, methods=["POST"]),
        Route("/api/restore", api_restore, methods=["POST"]),
        Route("/api/files", api_files, methods=["POST"]),
        Route("/api/journal", api_journal, methods=["GET"]),
    ]
    app = Starlette(debug=False, routes=routes, middleware=[Middleware(_Guard, port=port, secret=_Secret(token))],
                    exception_handlers={HTTPException: _http_exception})
    app.state.port = port
    app.state.version = __version__
    return app


# ------------------------------------------------------------------------------------------ server

def _bind(port: int) -> socket.socket:
    """A listening socket on 127.0.0.1:<port> only. Raises OSError when the port is taken."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if sys.platform.startswith("win"):
            exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
            if exclusive is not None:
                sock.setsockopt(socket.SOL_SOCKET, exclusive, 1)
        else:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((BIND_HOST, port))
        sock.listen(64)
        sock.set_inheritable(True)
    except BaseException:
        sock.close()
        raise
    return sock


def _open_browser_when_ready(server: Any, url: str) -> None:
    for _ in range(200):
        if getattr(server, "started", False):
            break
        time.sleep(0.05)
    try:
        import webbrowser

        webbrowser.open(url, new=2)
    except Exception as exc:  # noqa: BLE001 - the user can still open the address by hand
        _log_failure("open_browser", exc)


def serve(port: int = config.DESK_PORT, open_browser: bool = True, show_token: bool = True) -> int:
    """Run Shield Desk on 127.0.0.1:<port> until Ctrl+C (blocking). Returns an exit code.

    ``show_token=False`` (command run from an AI agent's shell): never print the token or the ``?t=`` link.
    This is the ``ragbase-shield desk`` runtime, so it writes its few status lines to the terminal.
    """
    try:
        port = _check_port(port)
    except ValueError:
        print("Port invalide : choisissez un nombre entre 1 et 65535.", file=sys.stderr)
        return 2
    import uvicorn

    logging.getLogger("pypdf").setLevel(logging.ERROR)  # a damaged PDF is reported in the page, not in the terminal
    token = load_or_create_token()
    app = create_app(port=port, token=token)
    _clean_stale_uploads()
    try:
        sock = _bind(port)
    except OSError:
        if show_token:
            # Shield Desk already runs (same data folder, same token): give the link that works, with the token.
            print(f"Shield Desk est déjà lancé sur le port {port}. Ouvrez ce lien (il contient votre jeton privé, "
                  f"ne le partagez pas) :\n{desk_url(port, token)}", file=sys.stderr)
            if open_browser:
                _open_browser_when_ready(type("_Started", (), {"started": True})(), desk_url(port, token))
        else:
            print(f"Le port {port} est déjà utilisé (Shield Desk est peut-être déjà lancé). Lancez « uvx "
                  "ragbase-shield desk » dans votre propre terminal pour ouvrir la page.", file=sys.stderr)
        return 1
    cfg = uvicorn.Config(app, host=BIND_HOST, port=port, log_level="warning", access_log=False,
                         server_header=False, proxy_headers=False, ws="none", lifespan="off",
                         timeout_keep_alive=5)
    server = uvicorn.Server(cfg)
    if show_token:
        print(f"Shield Desk : {desk_url(port, token)}")
        print("Ce lien contient votre jeton privé : ne le partagez pas.")
    else:
        print(f"Shield Desk : {desk_url(port)}")
        print("Le lien avec le jeton n'est pas affiché ici (terminal d'un agent IA). "
              "Lancez « ragbase-shield desk » dans votre propre terminal pour ouvrir la page.")
    print("Arrêt : Ctrl+C.")
    if open_browser:
        threading.Thread(target=_open_browser_when_ready, args=(server, desk_url(port, token)),
                         name="shield-desk-browser", daemon=True).start()
    try:
        server.run(sockets=[sock])
    except KeyboardInterrupt:
        pass
    finally:
        try:
            sock.close()
        except OSError:
            pass
    print("Shield Desk arrêté.")
    return 0


def run(port: int = config.DESK_PORT, open_browser: bool = True, show_token: bool = True) -> int:
    """Alias of :func:`serve`."""
    return serve(port=port, open_browser=open_browser, show_token=show_token)
