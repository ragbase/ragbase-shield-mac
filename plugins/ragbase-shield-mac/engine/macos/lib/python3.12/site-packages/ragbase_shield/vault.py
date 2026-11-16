"""Dossier registry and encrypted per-dossier entity store (contract: ARCHITECTURE.md section 4.2).

Layout under ``config.home()``:

- ``registry.db``: one row per dossier. Folder path and title are Fernet-encrypted with the master key.
  Each dossier has its own random data key, stored here wrapped (encrypted) by the master key. Removing a
  dossier deletes its wrapped key, so a leftover copy of its vault file can no longer be decrypted.
- ``vaults/<uuid>.db``: one SQLite file per dossier. Real values, normalised keys ("norms", which reveal
  names) and document paths are Fernet-encrypted with the dossier key. Lookups use a keyed HMAC of the norm,
  so no norm is stored in clear.
- ``key``: the master key file (mode 0600), used only when the OS keychain is off or not available.

The master key lives in the OS keychain (service "ragbase-shield", user "vault-key") unless
``RAGBASE_SHIELD_KEYRING=0`` or the keychain fails. A key check token in the registry makes a wrong or
missing key fail closed (VaultError) instead of silently starting a new, empty vault.

Messages of every exception raised here contain codes only (DOSSIER_n, DOC_n, PERSONNE_A), never a real
value, a title or a path.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import math
import os
import re
import sqlite3
import stat
import threading
import time
import unicodedata
import uuid as uuidlib
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from . import config
from .documents import KIND_BY_EXT, SUPPORTED_EXTENSIONS
from .engine.types import CODE_RE, KINDS, Entity, make_code

log = logging.getLogger(__name__)

KEYRING_SERVICE = "ragbase-shield"
KEYRING_USER = "vault-key"
KEY_FILE_NAME = "key"
REGISTRY_FILE_NAME = "registry.db"
VAULTS_DIR_NAME = "vaults"
IMPORTED_DIR_NAME = "imported"
MAX_DOCUMENTS = 20000
#: Vault schema: "2" since v0.2 (synthetic value per entity, id shapes, dossier settings). A "1" vault is migrated
#: in place on open (ALTER TABLE ... ADD COLUMN, CREATE TABLE IF NOT EXISTS): nothing is rewritten or re-encrypted.
SCHEMA_VERSION = "2"
#: Entity sources accepted by the store and the export ("ner" since v0.2).
ENTITY_SOURCES = ("detected", "user", "ner", "directory")  # "directory" (v0.3, S3): seeded from the dossier directory
#: Per-dossier settings (``meta`` table) a user may change, with their allowed values.
DOSSIER_SETTINGS = {"language": ("auto", "fr", "en"), "replacement": ("codes", "synthetic")}

_KEY_CHECK = b"ragbase-shield master key check v1"
_EXPORT_MAGIC = b"RAGBASE-SHIELD-EXPORT 1\n"
_EXPORT_FORMAT = "ragbase-shield-dossier"
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2 ** 15, 8, 1
MIN_PASSPHRASE_CHARS = 8

_DOSSIER_RE = re.compile(r"^DOSSIER_([0-9]{1,6})$")
_DOC_RE = re.compile(r"^DOC_([0-9]{1,7})$")
_OUTPUT_RE = re.compile(r"^OUTPUT_([0-9]{1,7})$")
_EVENT_RE = re.compile(r"^[a-z][a-z0-9_.:-]{0,63}$")
_COUNT_KEY_RE = re.compile(r"^[a-z_][a-z0-9_]{0,63}$")
# Fixed vocabulary that may appear as a journal value besides codes and numbers.
#: Kind of a file Shield lists but cannot read (spreadsheet, image, archive...): R1 correction CL-28.
UNSUPPORTED_KIND = "unsupported"
_JOURNAL_WORDS = frozenset(KINDS) | frozenset(KIND_BY_EXT.values()) | {UNSUPPORTED_KIND}
_SKIP_DIRS = {config.OUTPUT_DIR_NAME, "__MACOSX", "$RECYCLE.BIN", "System Volume Information"}
# Files the operating system writes into folders: never listed, even as unsupported.
_SYSTEM_FILES = frozenset({"thumbs.db", "desktop.ini", "ehthumbs.db", "icon\r"})


class VaultError(Exception):
    """Vault failure. The message holds codes only, never a real value, title or path."""


class UnknownDossier(VaultError, KeyError):
    """No dossier with this code."""

    def __str__(self) -> str:  # KeyError would add quotes
        return Exception.__str__(self)


@dataclass(frozen=True)
class DocRef:
    """A document of a dossier. ``path``, ``name`` and ``rel`` are REAL values: never show them to a model."""

    id: str        # "DOC_7", stable across calls and restarts
    path: Path     # absolute path
    name: str      # file name
    kind: str      # "pdf", "docx", "txt", "md", "eml", "html" ("unsupported": listed, never read)
    size: int      # bytes
    rel: str = ""  # path relative to the dossier folder

    def __repr__(self) -> str:
        return f"DocRef({self.id}, kind={self.kind}, size={self.size})"


# ------------------------------------------------------------------------------------------ normalise

_PUNCT_MAP = str.maketrans({"\u2019": "'", "\u2018": "'", "\u02bc": "'", "\u0153": "oe", "\u0152": "oe",
                            "\u00e6": "ae", "\u00c6": "ae", "\u00a0": " ", "\u202f": " "})


def _local_normalize(value: str) -> str:
    """lower case, no accents, single spaces (fallback when engine.normalize is not available)."""
    s = unicodedata.normalize("NFKD", value.translate(_PUNCT_MAP))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.casefold().split())


def normalize(value: str) -> str:
    """Normalise a value into a lookup key. Uses ``engine.normalize.normalize`` when the engine has it."""
    try:
        from .engine.normalize import normalize as engine_normalize  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - engine module not there yet: use the local normaliser
        engine_normalize = None
    if engine_normalize is not None:
        result = engine_normalize(value)
        if isinstance(result, tuple):  # (text, offsets) style API
            result = result[0]
        if isinstance(result, str):
            return result
    return _local_normalize(value)


# ------------------------------------------------------------------------------------------ keys

def _keyring_enabled() -> bool:
    return os.environ.get("RAGBASE_SHIELD_KEYRING", "1").strip().lower() not in ("0", "false", "no", "off")


def _as_key(raw: Any) -> Optional[bytes]:
    try:
        key = raw.encode("ascii") if isinstance(raw, str) else bytes(raw)
        Fernet(key)
        return key
    except Exception:  # noqa: BLE001
        return None


def _private_file(path: Path) -> None:
    """Create an empty file readable by the owner only (0600), if it does not exist."""
    if not path.exists():
        try:
            os.close(os.open(path, os.O_WRONLY | os.O_CREAT, 0o600))
        except FileExistsError:
            pass
    _tighten(path)


def _tighten(path: Path) -> None:
    if os.name != "posix":
        return
    try:
        if stat.S_IMODE(path.stat().st_mode) & 0o077:
            os.chmod(path, 0o600)
    except OSError:
        pass


def _read_key_file(path: Path) -> Optional[bytes]:
    if not path.exists():
        return None
    if os.name == "posix" and stat.S_IMODE(path.stat().st_mode) & 0o077:
        log.warning("key file permissions were too open; set to 0600")
        os.chmod(path, 0o600)
    key = _as_key(path.read_bytes().strip())
    if key is None:
        raise VaultError("Fichier de clé illisible (key file unreadable)")
    return key


def _create_key_file(path: Path) -> bytes:
    key = Fernet.generate_key()
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        existing = _read_key_file(path)
        assert existing is not None
        return existing
    with os.fdopen(fd, "wb") as fh:
        fh.write(key)
    return key


def _keyring_key(create: bool) -> Optional[bytes]:
    """Master key from the OS keychain; None if the keychain is off, empty or fails."""
    if not _keyring_enabled():
        return None
    try:
        import keyring

        raw = keyring.get_password(KEYRING_SERVICE, KEYRING_USER)
        if raw:
            return _as_key(raw)
        if not create:
            return None
        key = Fernet.generate_key()
        raw = keyring.get_password(KEYRING_SERVICE, KEYRING_USER)  # again, just before writing: never overwrite
        if raw:
            return _as_key(raw)
        keyring.set_password(KEYRING_SERVICE, KEYRING_USER, key.decode("ascii"))
        stored = keyring.get_password(KEYRING_SERVICE, KEYRING_USER)
        # Another process may have written its own key at the same moment: use what the keychain holds.
        return _as_key(stored) if stored else None
    except Exception as exc:  # noqa: BLE001 - any keychain problem: fall back to the key file
        log.info("OS keychain not available (%s); using the key file", type(exc).__name__)
        return None


LOCK_FILE_NAME = ".registry.lock"


@contextmanager
def _creation_lock(home: Path, timeout: float = 30.0) -> Iterator[None]:
    """Exclusive lock (across processes and threads) around the first creation of a registry and its key.

    Without it, two processes opening a fresh home at once (Claude Desktop's server and Shield Desk at first
    install) could each create a different master key: one of them would be lost, and every dossier with it.
    """
    path = home / LOCK_FILE_NAME
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    locked = False
    deadline = time.monotonic() + timeout
    try:
        while not locked:
            try:
                if os.name == "nt":  # pragma: no cover - Windows
                    import msvcrt

                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except OSError:
                if time.monotonic() > deadline:
                    raise VaultError("Registre occupé par un autre programme (registry busy)") from None
                time.sleep(0.05)
        yield
    finally:
        if locked:
            try:
                if os.name == "nt":  # pragma: no cover - Windows
                    import msvcrt

                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(fd)


def _save_key_file(path: Path, key: bytes) -> bool:
    """Keep a copy of the key in use in the 0600 key file (only if there is no key file yet)."""
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return _read_key_file(path) == key
    with os.fdopen(fd, "wb") as fh:
        fh.write(key)
    return True


def _key_matches(key: bytes, check_token: bytes) -> bool:
    try:
        return Fernet(key).decrypt(check_token) == _KEY_CHECK
    except (InvalidToken, ValueError):
        return False


#: Plugin mode (``ragbase_shield.claude_tab``): the ONLY place of the master key. An object with ``source`` (str),
#: ``get() -> Optional[bytes]`` and ``put(key: bytes) -> None``. When it is set, the keychain library and the key
#: file are never used, and a key store failure is a VaultError (no silent fallback to a file next to the vault).
KEY_STORE: Any = None


def _store_master_key(check_token: Optional[bytes]) -> tuple[bytes, str]:
    store = KEY_STORE
    try:
        key = store.get()
    except Exception:  # noqa: BLE001 - never echo the message
        raise VaultError("Clé du coffre illisible (key store error)") from None
    key = _as_key(key) if key is not None else None
    if check_token is not None:
        if key is not None and _key_matches(key, check_token):
            return key, store.source
        raise VaultError("Clé du coffre introuvable ou différente : le coffre ne peut pas être ouvert "
                         "(vault key missing or wrong)")
    if key is None:
        try:
            store.put(Fernet.generate_key())
            key = store.get()
        except Exception:  # noqa: BLE001
            raise VaultError("Clé du coffre impossible à créer (key store error)") from None
        key = _as_key(key) if key is not None else None
        if key is None:
            raise VaultError("Clé du coffre impossible à créer (key store error)")
    return key, store.source


def _master_key(home: Path, check_token: Optional[bytes]) -> tuple[bytes, str]:
    """Find the master key. With an existing registry the key must pass the check token (fail closed)."""
    if KEY_STORE is not None:
        return _store_master_key(check_token)
    key_file = home / KEY_FILE_NAME
    if check_token is not None:
        candidates = [(k, src) for k, src in ((_read_key_file(key_file), "file"), (_keyring_key(False), "keyring"))
                      if k is not None]
        for key, source in candidates:
            if _key_matches(key, check_token):
                return key, source
        raise VaultError("Clé du coffre introuvable ou différente : le coffre ne peut pas être ouvert "
                         "(vault key missing or wrong)")
    key = _keyring_key(True)
    if key is not None:
        return key, "keyring"
    return _read_key_file(key_file) or _create_key_file(key_file), "file"


def _subkey(key: bytes, purpose: bytes) -> bytes:
    raw = base64.urlsafe_b64decode(key)
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b"ragbase-shield " + purpose).derive(raw)


def _passphrase_key(passphrase: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    secret = unicodedata.normalize("NFC", passphrase).encode("utf-8")
    return base64.urlsafe_b64encode(Scrypt(salt=salt, length=32, n=n, r=r, p=p).derive(secret))


# ------------------------------------------------------------------------------------------ sqlite

def _connect(path: Path) -> sqlite3.Connection:
    _private_file(path)
    conn = sqlite3.connect(str(path), timeout=10.0, isolation_level=None, check_same_thread=False)
    try:
        conn.execute("PRAGMA busy_timeout=10000")
        # "PRAGMA journal_mode=WAL" does not wait for the busy timeout: retry while another process creates it.
        for attempt in range(100):
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                break
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() and "busy" not in str(exc).lower() or attempt == 99:
                    raise
                time.sleep(0.05)
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA secure_delete=ON")
    except BaseException:
        # any error (a damaged file raises DatabaseError, not OperationalError): never keep the handle open. An
        # open handle locks the file on Windows, so the vault could not be removed afterwards.
        conn.close()
        raise
    return conn


class _Db:
    """One SQLite connection guarded by a lock; explicit IMMEDIATE write transactions (safe across processes)."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.RLock()
        self.conn = _connect(path)
        self.writes = 0

    @contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        with self.lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield self.conn
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            self.conn.execute("COMMIT")
            self.writes += 1

    def schema(self, script: str) -> None:
        """Run CREATE ... IF NOT EXISTS statements (outside any transaction: executescript commits first)."""
        with self.lock:
            self.conn.executescript(script)

    def query(self, sql: str, args: tuple = ()) -> list[tuple]:
        with self.lock:
            return self.conn.execute(sql, args).fetchall()

    def version(self) -> tuple[int, int]:
        """Changes by other connections (data_version) and by this one (writes)."""
        with self.lock:
            return self.conn.execute("PRAGMA data_version").fetchone()[0], self.writes

    def data_version(self) -> int:
        """Changes committed by OTHER connections since this one opened (cheap)."""
        with self.lock:
            return self.conn.execute("PRAGMA data_version").fetchone()[0]

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        """A read transaction: every query inside sees the same state of the file."""
        with self.lock:
            self.conn.execute("BEGIN")
            try:
                yield self.conn
            finally:
                self.conn.execute("COMMIT")

    def close(self) -> None:
        with self.lock:
            try:
                self.conn.close()
            except sqlite3.Error:
                pass


_REGISTRY_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v BLOB);
CREATE TABLE IF NOT EXISTS dossiers(
  num INTEGER PRIMARY KEY AUTOINCREMENT,
  uuid TEXT NOT NULL UNIQUE,
  folder_mac BLOB NOT NULL,
  folder_enc BLOB NOT NULL,
  title_enc BLOB NOT NULL,
  locale TEXT NOT NULL,
  key_enc BLOB NOT NULL,
  created REAL NOT NULL);
CREATE INDEX IF NOT EXISTS dossiers_folder ON dossiers(folder_mac);
CREATE TABLE IF NOT EXISTS journal(
  id INTEGER PRIMARY KEY AUTOINCREMENT, time REAL NOT NULL, event TEXT NOT NULL,
  dossier TEXT, counts TEXT NOT NULL);
"""

_VAULT_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS entities(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL, idx INTEGER NOT NULL, code TEXT NOT NULL UNIQUE,
  norm_mac BLOB NOT NULL, norm_enc BLOB NOT NULL, value_enc BLOB NOT NULL,
  source TEXT NOT NULL, created REAL NOT NULL,
  synth_enc BLOB, synth_mac BLOB,
  UNIQUE(kind, norm_mac), UNIQUE(kind, idx));
CREATE UNIQUE INDEX IF NOT EXISTS entities_synth ON entities(synth_mac);
CREATE TABLE IF NOT EXISTS shapes(
  id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, template_mac BLOB NOT NULL, template_enc BLOB NOT NULL,
  created REAL NOT NULL, UNIQUE(kind, template_mac));
CREATE TABLE IF NOT EXISTS counters(kind TEXT PRIMARY KEY, next INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS clear_terms(norm_mac BLOB PRIMARY KEY, norm_enc BLOB NOT NULL, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS documents(
  id INTEGER PRIMARY KEY AUTOINCREMENT, path_mac BLOB NOT NULL UNIQUE, path_enc BLOB NOT NULL,
  created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS outputs(id INTEGER PRIMARY KEY AUTOINCREMENT, path_enc BLOB NOT NULL, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS journal(
  id INTEGER PRIMARY KEY AUTOINCREMENT, time REAL NOT NULL, event TEXT NOT NULL, counts TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS evidence(
  key_mac BLOB PRIMARY KEY, key_enc BLOB NOT NULL, counts_enc BLOB NOT NULL, updated REAL NOT NULL);
CREATE TABLE IF NOT EXISTS directory(id INTEGER PRIMARY KEY CHECK(id = 1), data_enc BLOB NOT NULL, updated REAL NOT NULL);
"""
# ``evidence`` and ``directory`` (v0.3, S2+S3) are additive tables created on open by the script above (no schema
# version change): the export-level evidence of what each surface is (counts per folded surface, encrypted) and
# the dossier directory (people from headers, Slack users, authors, signatures; one encrypted JSON document).
# Both keep their own change counters in ``meta`` ("erev", "drev"), separate from the entity revision, so a
# masking that records evidence never invalidates the entity snapshot or the maskers' lexicons.
_EVIDENCE_FIELDS = ("given", "family", "nick", "person_cue", "org", "org_head", "username",
                    "label_org", "label_person")  # v0.4 (T3): cues around a one-word company label
_MAX_EVIDENCE_KEY = 200


# ------------------------------------------------------------------------------------------ journal

def _journal_value(value: Any, nested: bool = False) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("journal: nombre invalide (invalid number)")
        return value
    if isinstance(value, str):
        if CODE_RE.fullmatch(value) or value in _JOURNAL_WORDS:
            return value
        raise ValueError("journal: seuls les codes et les nombres sont acceptés (codes and numbers only)")
    if not nested and isinstance(value, (list, tuple)):
        return [_journal_value(v, True) for v in value]
    if not nested and isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if not isinstance(k, str) or not (CODE_RE.fullmatch(k) or k in _JOURNAL_WORDS):
                raise ValueError("journal: clé invalide (keys must be codes or kinds)")
            out[k] = _journal_value(v, True)
        return out
    raise ValueError("journal: type de valeur refusé (value type refused)")


def _journal_record(event: str, counts: dict[str, Any]) -> tuple[str, str]:
    if not isinstance(event, str) or not _EVENT_RE.match(event):
        raise ValueError("journal: nom d'événement invalide (invalid event name)")
    clean = {}
    for key, value in counts.items():
        if not _COUNT_KEY_RE.match(key):
            raise ValueError("journal: nom de compteur invalide (invalid counter name)")
        clean[key] = _journal_value(value)
    return event, json.dumps(clean, ensure_ascii=True, sort_keys=True)


def _journal_row(t: float, event: str, dossier: Optional[str], counts: str) -> dict:
    return {"time": datetime.fromtimestamp(t).astimezone().isoformat(timespec="seconds"),
            "event": event, "dossier": dossier, "counts": json.loads(counts)}


# ------------------------------------------------------------------------------------------ dossier

@dataclass
class _Snapshot:
    entities: list[Entity] = field(default_factory=list)
    by_code: dict[str, Entity] = field(default_factory=dict)
    by_norm: dict[tuple[str, str], Entity] = field(default_factory=dict)
    clear_macs: frozenset = frozenset()
    synthetic: dict[str, str] = field(default_factory=dict)      # code -> fake value (v0.2)
    by_synth_norm: dict[str, str] = field(default_factory=dict)  # normalised fake value -> code
    shapes: list[tuple[str, str]] = field(default_factory=list)  # (kind, template) id layouts (v0.2)


class Dossier:
    """One client dossier: an encrypted entity store (engine EntityStore protocol) bound to a folder.

    ``title`` and ``folder`` are REAL values: never show them to a model. ``repr()`` shows the code only.
    """

    def __init__(self, *, num: int, uuid: str, folder: Path, title: str, locale: str, data_key: bytes,
                 db_path: Path, created: float = 0.0):
        self.num = num
        self.code = f"DOSSIER_{num}"
        self.uuid = uuid
        self.folder = folder
        self.title = title
        self.locale = locale
        self.created = created
        self._fernet = Fernet(data_key)
        self._mac_key = _subkey(data_key, b"norm index v1")
        self._synth_seed = _subkey(data_key, b"synthetic v1")  # seed of the fake values (v0.2), never stored
        self._snap: Optional[_Snapshot] = None
        self._snap_dv: Optional[int] = None   # data_version when the snapshot was last known to be current
        self._snap_rev: Optional[int] = None  # the vault's "rev" counter the snapshot reflects
        self._snap_local = -1                 # this object's own entity writes the snapshot reflects
        self._local = 0
        self._ev_cache: tuple = (None, None)   # (erev, {key: counts}) (v0.3)
        self._dir_cache: tuple = (None, None)  # (drev, directory dict or None) (v0.3)
        try:
            self._db = _Db(db_path)
        except sqlite3.Error:
            raise VaultError(f"{self.code}: coffre illisible (vault unreadable)") from None
        try:
            self._migrate()
            self._db.schema(_VAULT_SCHEMA)
            if not self._db.query("SELECT v FROM meta WHERE k='uuid'"):
                with self._db.write() as c:
                    for k, v in (("schema", SCHEMA_VERSION), ("uuid", uuid), ("locale", locale), ("rev", "0")):
                        c.execute("INSERT OR IGNORE INTO meta(k, v) VALUES(?, ?)", (k, v))
            elif not self._db.query("SELECT v FROM meta WHERE k='rev'"):  # vault made by an earlier version
                with self._db.write() as c:
                    c.execute("INSERT OR IGNORE INTO meta(k, v) VALUES('rev', '0')")
            stored = self._db.query("SELECT v FROM meta WHERE k='uuid'")[0][0]
        except sqlite3.Error:
            self._db.close()
            raise VaultError(f"{self.code}: coffre illisible (vault unreadable)") from None
        if stored != uuid:
            self._db.close()
            raise VaultError(f"{self.code}: coffre incohérent (vault does not match the registry)")

    def __repr__(self) -> str:
        return f"Dossier({self.code})"

    __str__ = __repr__

    def _migrate(self) -> None:
        """Schema 1 -> 2 (v0.2): add the synthetic columns to ``entities``; the new tables come from the schema
        script. Runs before the schema script so that an existing table gets its columns; idempotent."""
        tables = {r[0] for r in self._db.query("SELECT name FROM sqlite_master WHERE type='table'")}
        if "entities" not in tables:
            return
        cols = {r[1] for r in self._db.query("PRAGMA table_info(entities)")}
        if "synth_enc" in cols and "synth_mac" in cols:
            return
        with self._db.lock:
            for col in ("synth_enc", "synth_mac"):
                if col not in cols:
                    self._db.conn.execute(f"ALTER TABLE entities ADD COLUMN {col} BLOB")
            self._db.conn.execute("INSERT OR REPLACE INTO meta(k, v) VALUES('schema', ?)", (SCHEMA_VERSION,))
        log.info("%s: vault schema upgraded to %s", self.code, SCHEMA_VERSION)

    # crypto helpers
    def _enc(self, text: str) -> bytes:
        return self._fernet.encrypt(text.encode("utf-8"))

    def _dec(self, blob: bytes) -> str:
        try:
            return self._fernet.decrypt(bytes(blob)).decode("utf-8")
        except InvalidToken:
            raise VaultError(f"{self.code}: déchiffrement impossible (decryption failed)") from None

    def _mac(self, text: str) -> bytes:
        return hmac.new(self._mac_key, text.encode("utf-8"), hashlib.sha256).digest()

    @staticmethod
    def _rev_of(c: sqlite3.Connection) -> int:
        row = c.execute("SELECT v FROM meta WHERE k='rev'").fetchone()
        try:
            return int(row[0]) if row else 0
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _bump(c: sqlite3.Connection) -> int:
        """Inside a write transaction: count one change of the entities or of the kept-clear terms."""
        c.execute("INSERT OR IGNORE INTO meta(k, v) VALUES('rev', '0')")
        c.execute("UPDATE meta SET v = CAST(v AS INTEGER) + 1 WHERE k='rev'")
        return Dossier._rev_of(c)

    def _snapshot(self) -> _Snapshot:
        """Decrypted entities and kept-clear terms. Rebuilt only when the entities or the terms changed (by
        this process or another one): journal writes and reads never force a rebuild."""
        dv = self._db.data_version()
        if self._snap is not None and self._snap_local == self._local:
            if self._snap_dv == dv:
                return self._snap
            with self._db.lock:
                rev = self._rev_of(self._db.conn)
            if rev == self._snap_rev:  # another process wrote, but not entities or terms (journal...)
                self._snap_dv = dv
                return self._snap
        snap = _Snapshot()
        with self._db.read() as c:
            rev = self._rev_of(c)
            rows = c.execute("SELECT kind, code, norm_enc, value_enc, source, synth_enc FROM entities ORDER BY id"
                             ).fetchall()
            clear = c.execute("SELECT norm_mac FROM clear_terms").fetchall()
            shapes = c.execute("SELECT kind, template_enc FROM shapes ORDER BY id").fetchall()
        for kind, code, norm_enc, value_enc, source, synth_enc in rows:
            ent = Entity(kind=kind, value=self._dec(value_enc), norm=self._dec(norm_enc), code=code,
                         source=source or "detected")
            snap.entities.append(ent)
            snap.by_code[code] = ent
            snap.by_norm[(kind, ent.norm)] = ent
            if synth_enc:
                fake = self._dec(synth_enc)
                snap.synthetic[code] = fake
                snap.by_synth_norm[normalize(fake)] = code
        snap.clear_macs = frozenset(bytes(r[0]) for r in clear)
        snap.shapes = [(k, self._dec(t)) for k, t in shapes]
        self._snap, self._snap_dv, self._snap_rev, self._snap_local = snap, dv, rev, self._local
        return snap

    def _after_write(self, rev_before: int, rev_after: int, apply=None) -> None:
        """Keep the snapshot current after this object's own write (no full decrypt), or drop it."""
        self._local += 1
        snap = self._snap
        if snap is not None and apply is not None and self._snap_rev == rev_before and \
                self._snap_local == self._local - 1:
            apply(snap)
            self._snap_rev, self._snap_local, self._snap_dv = rev_after, self._local, self._db.data_version()

    @property
    def revision(self) -> tuple[str, int]:
        """Changes whenever the entities or the kept-clear terms change (maskers and caches key on it)."""
        self._snapshot()
        return self.uuid, int(self._snap_rev or 0)

    # ---------------------------------------------------------------- EntityStore protocol
    def all(self) -> list[Entity]:
        """Every entity of the dossier, in creation order."""
        return list(self._snapshot().entities)

    def get_by_norm(self, kind: str, norm: str) -> Optional[Entity]:
        return self._snapshot().by_norm.get((kind, norm))

    def get_by_code(self, code: str) -> Optional[Entity]:
        return self._snapshot().by_code.get(code)

    def add(self, kind: str, value: str, norm: str, source: str = "detected") -> Entity:
        """Return the entity for (kind, norm), creating it with the next code of this kind if needed.

        ``source`` (v0.2): "detected" (rules), "ner" (the optional model); "user" is reserved to :meth:`register`."""
        if source not in ENTITY_SOURCES or source == "user":
            source = "detected"
        return self._add(kind, value, norm, source)

    def is_kept_clear(self, norm: str) -> bool:
        return self._mac(norm) in self._snapshot().clear_macs

    def _add(self, kind: str, value: str, norm: str, source: str) -> Entity:
        if kind not in KINDS:
            raise ValueError("unknown entity kind")
        if not isinstance(value, str) or not value.strip() or not isinstance(norm, str) or not norm:
            raise ValueError("empty entity value")
        found = self.get_by_norm(kind, norm)
        if found is not None:
            return found
        mac = self._mac(norm)
        ent = None
        with self._db.write() as c:
            row = c.execute("SELECT code FROM entities WHERE kind=? AND norm_mac=?", (kind, mac)).fetchone()
            rev0 = rev1 = self._rev_of(c)
            if row is None:  # still new inside the write lock (another process may have added it)
                nxt = c.execute("SELECT next FROM counters WHERE kind=?", (kind,)).fetchone()
                idx = nxt[0] if nxt else 0
                code = make_code(kind, idx, self.locale)
                c.execute("INSERT INTO entities(kind, idx, code, norm_mac, norm_enc, value_enc, source, created)"
                          " VALUES(?,?,?,?,?,?,?,?)",
                          (kind, idx, code, mac, self._enc(norm), self._enc(value), source, time.time()))
                c.execute("INSERT INTO counters(kind, next) VALUES(?, ?) "
                          "ON CONFLICT(kind) DO UPDATE SET next=excluded.next", (kind, idx + 1))
                rev1 = self._bump(c)
                ent = Entity(kind=kind, value=value, norm=norm, code=code, source=source)

        def add_to(snap: _Snapshot) -> None:
            snap.entities.append(ent)
            snap.by_code[ent.code] = ent
            snap.by_norm[(kind, norm)] = ent

        self._after_write(rev0, rev1, add_to if ent is not None else None)
        found = self.get_by_norm(kind, norm)
        if found is None:  # pragma: no cover - would mean the snapshot cache is broken
            raise VaultError(f"{self.code}: entité introuvable après ajout")
        return found

    # ---------------------------------------------------------------- user actions
    def register(self, kind: str, value: str) -> Entity:
        """The user adds a name or identifier. Returns the (possibly existing) entity with its code."""
        if kind not in KINDS:
            raise ValueError("unknown entity kind")
        value = " ".join(str(value).split())
        norm = normalize(value)
        if not norm:
            raise ValueError("empty entity value")
        return self._add(kind, value, norm, "user")

    def keep_clear(self, norm: str, value: bool = True) -> None:
        """Mark a term "garder en clair" (or undo it). Accepts an entity code (PERSONNE_A) or a term."""
        target = str(norm).strip()
        if CODE_RE.fullmatch(target):
            ent = self.get_by_code(target)
            if ent is None:
                raise VaultError(f"{target}: code inconnu dans {self.code} (unknown code)")
            key = ent.norm
        else:
            key = normalize(target)
        if not key:
            raise ValueError("empty term")
        mac = self._mac(key)
        with self._db.write() as c:
            rev0 = self._rev_of(c)
            if value:
                c.execute("INSERT OR IGNORE INTO clear_terms(norm_mac, norm_enc, created) VALUES(?,?,?)",
                          (mac, self._enc(key), time.time()))
            else:
                c.execute("DELETE FROM clear_terms WHERE norm_mac=?", (mac,))
            rev1 = self._bump(c)

        def toggle(snap: _Snapshot) -> None:
            snap.clear_macs = snap.clear_macs | {mac} if value else snap.clear_macs - {mac}

        self._after_write(rev0, rev1, toggle)

    def upgrade(self, code: str, value: str, norm: str) -> Optional[Entity]:
        """Give an existing code a fuller value ("Ouardi" -> "Mehdi Ouardi"). None when the code is unknown
        or the new key belongs to another entity. A kept-clear choice follows the entity."""
        ent = self.get_by_code(code)
        if ent is None or not isinstance(value, str) or not value.strip() or not norm:
            return None
        if self.get_by_norm(ent.kind, norm) is not None:
            return None
        mac, old_mac = self._mac(norm), self._mac(ent.norm)
        kept = self.is_kept_clear(ent.norm)
        with self._db.write() as c:
            if c.execute("SELECT 1 FROM entities WHERE kind=? AND norm_mac=?", (ent.kind, mac)).fetchone():
                return None
            # a fuller sighting confirms an entity the model alone had found (v0.2: containment lifts)
            c.execute("UPDATE entities SET norm_mac=?, norm_enc=?, value_enc=?, "
                      "source=CASE WHEN source='ner' THEN 'detected' ELSE source END WHERE code=? AND norm_mac=?",
                      (mac, self._enc(norm), self._enc(value), code, old_mac))
            if kept:
                c.execute("INSERT OR IGNORE INTO clear_terms(norm_mac, norm_enc, created) VALUES(?,?,?)",
                          (mac, self._enc(norm), time.time()))
            self._bump(c)
        self._after_write(-1, -1)  # rebuild on the next read
        return self.get_by_code(code)

    def confirm(self, code: str) -> Optional[Entity]:
        """v0.3 (S7): an entity the model alone had found is confirmed by the rules or by a second signal: its
        source becomes "detected" (the masker's containment lifts). None when the code is unknown; an entity of
        another source is returned unchanged."""
        ent = self.get_by_code(code)
        if ent is None:
            return None
        if ent.source != "ner":
            return ent
        with self._db.write() as c:
            c.execute("UPDATE entities SET source='detected' WHERE code=? AND source='ner'", (code,))
            self._bump(c)
        self._after_write(-1, -1)  # rebuild on the next read
        return self.get_by_code(code)

    def kept_clear_terms(self) -> list[str]:
        """Normalised terms kept in clear (REAL values: local display only)."""
        return [self._dec(r[0]) for r in self._db.query("SELECT norm_enc FROM clear_terms ORDER BY created")]

    def entity_source(self, code: str) -> Optional[str]:
        """Return "user" (registered by the user), "detected" (added by the engine), "ner" (the model) or None."""
        rows = self._db.query("SELECT source FROM entities WHERE code=?", (code,))
        return rows[0][0] if rows else None

    # ---------------------------------------------------------------- synthetic replacement mode (v0.2, SYNTH)
    def synthetic_seed(self) -> bytes:
        """Secret seed of the fake values: HKDF of the dossier data key (never stored, never exported). An export
        carries the fake values themselves, so a colleague restores the same pseudonyms."""
        return self._synth_seed

    def synthetic(self, code: str) -> Optional[str]:
        """The fake value given to ``code`` in synthetic mode (REAL-looking but invented), or None."""
        return self._snapshot().synthetic.get(code)

    def set_synthetic(self, code: str, value: str) -> None:
        """Store (or replace) the fake value of an entity. Encrypted like the real value; a keyed HMAC of its
        normalised form makes it unique across the dossier."""
        ent = self.get_by_code(code)
        if ent is None:
            raise VaultError(f"{code}: code inconnu dans {self.code} (unknown code)")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("empty synthetic value")
        value = value.strip()
        norm = normalize(value)
        mac = self._mac("synthetic:" + norm)
        with self._db.write() as c:
            rev0 = self._rev_of(c)
            other = c.execute("SELECT code FROM entities WHERE synth_mac=? AND code<>?", (mac, code)).fetchone()
            if other is not None:
                raise VaultError(f"{code}: valeur synthétique déjà prise par {other[0]} (synthetic value taken)")
            c.execute("UPDATE entities SET synth_enc=?, synth_mac=? WHERE code=?", (self._enc(value), mac, code))
            rev1 = self._bump(c)

        def apply(snap: _Snapshot) -> None:
            old = snap.synthetic.get(code)
            if old is not None:
                snap.by_synth_norm.pop(normalize(old), None)
            snap.synthetic[code] = value
            snap.by_synth_norm[norm] = code

        self._after_write(rev0, rev1, apply)

    def get_by_synthetic_norm(self, norm: str) -> Optional[Entity]:
        """The entity whose fake value normalises to ``norm`` (``engine.normalize``), or None."""
        snap = self._snapshot()
        code = snap.by_synth_norm.get(norm)
        return snap.by_code.get(code) if code else None

    # ---------------------------------------------------------------- id shape memory (v0.2, DETECTORS; vault part SYNTH)
    def add_shape(self, kind: str, template: str) -> None:
        """Remember an identifier layout ("EMP-dddddd") learnt from a labelled value of ``kind``."""
        if kind not in KINDS:
            raise ValueError("unknown entity kind")
        template = str(template or "").strip()
        if not template or len(template) > 64:
            raise ValueError("empty or too long shape template")
        mac = self._mac("shape:" + kind + ":" + template)
        with self._db.write() as c:
            rev0 = self._rev_of(c)
            cur = c.execute("INSERT OR IGNORE INTO shapes(kind, template_mac, template_enc, created) VALUES(?,?,?,?)",
                            (kind, mac, self._enc(template), time.time()))
            if cur.rowcount == 0:
                return
            rev1 = self._bump(c)

        def apply(snap: _Snapshot) -> None:
            if (kind, template) not in snap.shapes:
                snap.shapes.append((kind, template))

        self._after_write(rev0, rev1, apply)

    def shapes(self) -> list[tuple[str, str]]:
        """``[(kind, template)]`` of the identifier layouts learnt in this dossier."""
        return list(self._snapshot().shapes)

    # ---------------------------------------------------------------- export evidence and directory (v0.3, S2+S3)
    def _counter(self, name: str) -> int:
        rows = self._db.query("SELECT v FROM meta WHERE k=?", (name,))
        try:
            return int(rows[0][0]) if rows else 0
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _bump_counter(c: sqlite3.Connection, name: str) -> None:
        c.execute("INSERT OR IGNORE INTO meta(k, v) VALUES(?, '0')", (name,))
        c.execute("UPDATE meta SET v = CAST(v AS INTEGER) + 1 WHERE k=?", (name,))

    def evidence(self) -> dict[str, dict[str, int]]:
        """Export-level evidence of the dossier: folded surface -> counts of what the documents showed it to be
        ("given", "family", "person_cue", "org", "org_head", "username", "label_org", "label_person"). REAL
        surfaces: local use only. Cached until :meth:`add_evidence` runs (here or in another process)."""
        erev = self._counter("erev")
        if self._ev_cache[0] == erev and self._ev_cache[1] is not None:
            return self._ev_cache[1]
        out: dict[str, dict[str, int]] = {}
        for key_enc, counts_enc in self._db.query("SELECT key_enc, counts_enc FROM evidence"):
            try:
                counts = json.loads(self._dec(counts_enc))
            except (ValueError, VaultError):
                continue
            if isinstance(counts, dict):
                out[self._dec(key_enc)] = {k: int(v) for k, v in counts.items() if k in _EVIDENCE_FIELDS}
        self._ev_cache = (erev, out)
        return out

    def add_evidence(self, batch: dict[str, dict[str, int]]) -> None:
        """Add the counts of ``batch`` ({surface: {field: n}}) to the stored evidence, in one transaction. The
        entity revision does not change. Unknown fields, empty keys and non-positive counts are ignored."""
        clean: dict[str, dict[str, int]] = {}
        for key, counts in (batch or {}).items():
            if not isinstance(key, str) or not key.strip() or len(key) > _MAX_EVIDENCE_KEY or not isinstance(counts, dict):
                continue
            c = {f: int(n) for f, n in counts.items() if f in _EVIDENCE_FIELDS and isinstance(n, int) and n > 0}
            if c:
                clean[key] = c
        if not clean:
            return
        now = time.time()
        with self._db.write() as c:
            for key, counts in clean.items():
                mac = self._mac("evidence:" + key)
                row = c.execute("SELECT counts_enc FROM evidence WHERE key_mac=?", (mac,)).fetchone()
                merged: dict[str, int] = {}
                if row is not None:
                    try:
                        old = json.loads(self._dec(row[0]))
                        if isinstance(old, dict):
                            merged = {k: int(v) for k, v in old.items() if k in _EVIDENCE_FIELDS}
                    except (ValueError, VaultError):
                        merged = {}
                for f, n in counts.items():
                    merged[f] = merged.get(f, 0) + n
                c.execute("INSERT INTO evidence(key_mac, key_enc, counts_enc, updated) VALUES(?,?,?,?) "
                          "ON CONFLICT(key_mac) DO UPDATE SET counts_enc=excluded.counts_enc, updated=excluded.updated",
                          (mac, self._enc(key), self._enc(json.dumps(merged, sort_keys=True)), now))
            self._bump_counter(c, "erev")
        cached = self._ev_cache[1]
        if cached is not None:  # keep the cache current without a full decrypt
            for key, counts in clean.items():
                slot = cached.setdefault(key, {})
                for f, n in counts.items():
                    slot[f] = slot.get(f, 0) + n
            self._ev_cache = (self._counter("erev"), cached)

    def directory(self) -> Optional[dict]:
        """The dossier directory as the dict of ``engine.directory.Directory.to_dict`` (REAL names: local use
        only), or None when none was built. The same object is returned until :meth:`set_directory` changes it."""
        drev = self._counter("drev")
        if self._dir_cache[0] == drev:
            return self._dir_cache[1]
        rows = self._db.query("SELECT data_enc FROM directory WHERE id=1")
        data = None
        if rows:
            try:
                parsed = json.loads(self._dec(rows[0][0]))
                data = parsed if isinstance(parsed, dict) else None
            except (ValueError, VaultError):
                data = None
        self._dir_cache = (drev, data)
        return data

    def set_directory(self, data: Optional[dict]) -> None:
        """Store the dossier directory (a ``Directory.to_dict()`` dict), or remove it with None."""
        if data is not None and not isinstance(data, dict):
            raise ValueError("directory must be a dict or None")
        with self._db.write() as c:
            if data is None:
                c.execute("DELETE FROM directory WHERE id=1")
            else:
                c.execute("INSERT INTO directory(id, data_enc, updated) VALUES(1, ?, ?) "
                          "ON CONFLICT(id) DO UPDATE SET data_enc=excluded.data_enc, updated=excluded.updated",
                          (self._enc(json.dumps(data, ensure_ascii=False, sort_keys=True)), time.time()))
            self._bump_counter(c, "drev")
        self._dir_cache = (None, None)

    # ---------------------------------------------------------------- dossier settings (v0.2, SYNTH)
    def setting(self, key: str) -> Optional[str]:
        """A per-dossier setting ("language", "replacement"), or None when it was never set."""
        if key not in DOSSIER_SETTINGS:
            raise ValueError("unknown setting")
        rows = self._db.query("SELECT v FROM meta WHERE k=?", ("setting:" + key,))
        value = rows[0][0] if rows else None
        return value if value in DOSSIER_SETTINGS[key] else None

    def set_setting(self, key: str, value: Optional[str]) -> None:
        """Set a per-dossier setting (``None`` removes it). Keys and values are checked against
        :data:`DOSSIER_SETTINGS`: a free string is never written."""
        if key not in DOSSIER_SETTINGS:
            raise ValueError("unknown setting")
        if value is not None and value not in DOSSIER_SETTINGS[key]:
            raise ValueError("invalid setting value")
        with self._db.write() as c:
            if value is None:
                c.execute("DELETE FROM meta WHERE k=?", ("setting:" + key,))
            else:
                c.execute("INSERT OR REPLACE INTO meta(k, v) VALUES(?, ?)", ("setting:" + key, value))

    def settings(self) -> dict[str, str]:
        """Every per-dossier setting that was set."""
        out = {}
        for key in DOSSIER_SETTINGS:
            v = self.setting(key)
            if v is not None:
                out[key] = v
        return out

    # ---------------------------------------------------------------- documents
    def documents(self, include_unsupported: bool = False) -> list[DocRef]:
        """Supported documents in the folder (recursive), with DOC_n ids stable across calls and restarts.

        Skips hidden files and folders, Office temp files (~$x.docx), symlinks that leave the folder and the
        output folder. Ids of deleted files are never reused. ``include_unsupported`` also lists the other
        files (spreadsheets, images...) with kind ``"unsupported"``: they get a DOC_n id but are never read.
        """
        found = self._scan_folder(include_unsupported)
        if not found:
            return []
        known = {bytes(m): i for i, m in self._db.query("SELECT id, path_mac FROM documents")}
        macs = {rel: self._mac(rel) for rel in found}
        new = sorted((rel for rel in found if macs[rel] not in known), key=lambda r: (r.casefold(), r))
        if new:
            with self._db.write() as c:
                for rel in new:
                    c.execute("INSERT OR IGNORE INTO documents(path_mac, path_enc, created) VALUES(?,?,?)",
                              (macs[rel], self._enc(rel), time.time()))
            known = {bytes(m): i for i, m in self._db.query("SELECT id, path_mac FROM documents")}
        refs = []
        for rel, (path, size) in found.items():
            num = known.get(macs[rel])
            if num is not None:
                refs.append(DocRef(id=f"DOC_{num}", path=path, name=path.name, kind=_kind_of(path), size=size,
                                   rel=rel))
        return sorted(refs, key=lambda d: int(d.id[4:]))

    def get_document(self, doc_id: str, include_unsupported: bool = False) -> Optional[DocRef]:
        """The document with this DOC_n id, or None if unknown or no longer in the folder.

        A file Shield cannot read is returned (kind ``"unsupported"``) only with ``include_unsupported``."""
        m = _DOC_RE.match(str(doc_id).strip().upper())
        if not m:
            return None
        rows = self._db.query("SELECT path_enc FROM documents WHERE id=?", (int(m.group(1)),))
        if not rows:
            return None
        rel = self._dec(rows[0][0])
        path = self.folder / rel
        if not path.is_file():
            # The id is keyed on the NFC form of the path; a file system that keeps names as written (NTFS,
            # ext4) may hold the NFD form (a ZIP made on a Mac): find the file by its NFC key.
            hit = (self._scan_folder(True) if include_unsupported else self._scan_folder()).get(rel)
            if hit is not None:
                path = hit[0]
        if not _inside(path, self.folder) or not path.is_file():
            return None
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS and not include_unsupported:
            return None
        return DocRef(id=f"DOC_{int(m.group(1))}", path=path, name=path.name, kind=_kind_of(path),
                      size=path.stat().st_size, rel=rel)

    def _scan_folder(self, include_unsupported: bool = False) -> dict[str, tuple[Path, int]]:
        root = self.folder
        out: dict[str, tuple[Path, int]] = {}
        if not root.is_dir():
            return out
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames
                                 if not _hidden_name(d) and d not in _SKIP_DIRS and not _hidden_attr(Path(dirpath, d)))
            for name in sorted(filenames):
                if _hidden_name(name) or name.casefold() in _SYSTEM_FILES:
                    continue
                if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS and not include_unsupported:
                    continue
                path = Path(dirpath, name)
                try:
                    if path.is_symlink() and not _inside(path.resolve(), root.resolve()):
                        continue
                    if _hidden_attr(path) or not path.is_file():
                        continue
                    size = path.stat().st_size
                except OSError:
                    continue
                rel = unicodedata.normalize("NFC", path.relative_to(root).as_posix())
                out[rel] = (path, size)
                if len(out) >= MAX_DOCUMENTS:
                    log.warning("%s: more than %d documents, the rest is ignored", self.code, MAX_DOCUMENTS)
                    return out
        return out

    # ---------------------------------------------------------------- outputs
    def output_dir(self) -> Path:
        """``<folder>/Shield - sorties``, created on demand."""
        out = self.folder / config.OUTPUT_DIR_NAME
        out.mkdir(parents=True, exist_ok=True)
        return out

    def register_output(self, path: os.PathLike | str) -> str:
        """Record a file written for this dossier. Returns its code OUTPUT_n (stable, never reused)."""
        with self._db.write() as c:
            cur = c.execute("INSERT INTO outputs(path_enc, created) VALUES(?, ?)",
                            (self._enc(str(Path(path))), time.time()))
            return f"OUTPUT_{cur.lastrowid}"

    def get_output(self, output_id: str) -> Optional[Path]:
        m = _OUTPUT_RE.match(str(output_id).strip().upper())
        if not m:
            return None
        rows = self._db.query("SELECT path_enc FROM outputs WHERE id=?", (int(m.group(1)),))
        return Path(self._dec(rows[0][0])) if rows else None

    # ---------------------------------------------------------------- journal
    def journal(self, event: str, **counts: Any) -> None:
        """Log an event with codes and numbers only. Any other string value raises ValueError."""
        event, payload = _journal_record(event, counts)
        with self._db.write() as c:
            c.execute("INSERT INTO journal(time, event, counts) VALUES(?,?,?)", (time.time(), event, payload))

    def journal_entries(self, limit: Optional[int] = None) -> list[dict]:
        """Newest first: [{time, event, dossier, counts}]."""
        return [row for _, row in self._journal_pairs(limit)]

    def _journal_pairs(self, limit: Optional[int]) -> list[tuple[float, dict]]:
        sql = "SELECT time, event, counts FROM journal ORDER BY id DESC"
        args: tuple = ()
        if limit is not None:
            sql += " LIMIT ?"
            args = (int(limit),)
        return [(t, _journal_row(t, e, self.code, c)) for t, e, c in self._db.query(sql, args)]

    # ---------------------------------------------------------------- export
    def export(self, path: os.PathLike | str, passphrase: str) -> Path:
        """Write an encrypted copy of this dossier vault for a colleague (scrypt + Fernet).

        The file holds the entities with their codes, the kept-clear terms and the DOC_n ids, so masked
        text and DOC ids stay valid on the colleague's computer. The journal is not exported.
        """
        if not isinstance(passphrase, str) or len(passphrase) < MIN_PASSPHRASE_CHARS:
            raise ValueError(f"passphrase: {MIN_PASSPHRASE_CHARS} caractères minimum")
        rows = self._db.query("SELECT kind, idx, code, norm_enc, value_enc, source, created, synth_enc FROM entities"
                              " ORDER BY id")
        entities = []
        for k, i, code, n, v, src, t, synth in rows:
            item = {"kind": k, "idx": i, "code": code, "norm": self._dec(n), "value": self._dec(v), "source": src,
                    "created": t}
            if synth:
                item["synthetic"] = self._dec(synth)
            entities.append(item)
        payload = {
            "format": _EXPORT_FORMAT, "version": 2, "code": self.code, "title": self.title,
            "folder": str(self.folder), "locale": self.locale,
            "entities": entities,
            "counters": {k: n for k, n in self._db.query("SELECT kind, next FROM counters")},
            "clear": self.kept_clear_terms(),
            "documents": [{"id": i, "rel": self._dec(p)}
                          for i, p in self._db.query("SELECT id, path_enc FROM documents ORDER BY id")],
            "shapes": [{"kind": k, "template": t} for k, t in self.shapes()],
            "settings": self.settings(),
            # v0.3 (S2+S3), additive: the export evidence and the dossier directory travel with the codes
            "evidence": self.evidence(),
            "directory": self.directory(),
        }
        salt = os.urandom(16)
        header = {"kdf": "scrypt", "n": _SCRYPT_N, "r": _SCRYPT_R, "p": _SCRYPT_P,
                  "salt": base64.b64encode(salt).decode("ascii")}
        key = _passphrase_key(passphrase, salt, _SCRYPT_N, _SCRYPT_R, _SCRYPT_P)
        token = Fernet(key).encrypt(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        target = Path(path)
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(_EXPORT_MAGIC + json.dumps(header).encode("ascii") + b"\n" + token + b"\n")
        self.journal("export", dossier=self.code, entities=len(rows))
        return target

    def _load_import(self, payload: dict) -> None:
        now = time.time()
        self._after_write(-1, -1)
        with self._db.write() as c:
            self._bump(c)
            for e in payload["entities"]:
                synth = e.get("synthetic")
                c.execute("INSERT INTO entities(kind, idx, code, norm_mac, norm_enc, value_enc, source, created,"
                          " synth_enc, synth_mac) VALUES(?,?,?,?,?,?,?,?,?,?)",
                          (e["kind"], e["idx"], e["code"], self._mac(e["norm"]), self._enc(e["norm"]),
                           self._enc(e["value"]), e["source"], float(e.get("created") or now),
                           self._enc(synth) if synth else None,
                           self._mac("synthetic:" + normalize(synth)) if synth else None))
            for sh in payload.get("shapes") or []:
                c.execute("INSERT OR IGNORE INTO shapes(kind, template_mac, template_enc, created) VALUES(?,?,?,?)",
                          (sh["kind"], self._mac("shape:" + sh["kind"] + ":" + sh["template"]),
                           self._enc(sh["template"]), now))
            for key, value in (payload.get("settings") or {}).items():
                c.execute("INSERT OR REPLACE INTO meta(k, v) VALUES(?, ?)", ("setting:" + key, value))
            for key, counts in (payload.get("evidence") or {}).items():
                c.execute("INSERT OR REPLACE INTO evidence(key_mac, key_enc, counts_enc, updated) VALUES(?,?,?,?)",
                          (self._mac("evidence:" + key), self._enc(key),
                           self._enc(json.dumps({f: int(n) for f, n in counts.items()}, sort_keys=True)), now))
            if payload.get("directory"):
                c.execute("INSERT OR REPLACE INTO directory(id, data_enc, updated) VALUES(1, ?, ?)",
                          (self._enc(json.dumps(payload["directory"], ensure_ascii=False, sort_keys=True)), now))
            self._bump_counter(c, "erev")
            self._bump_counter(c, "drev")
            counters = dict(payload["counters"])
            for e in payload["entities"]:
                counters[e["kind"]] = max(counters.get(e["kind"], 0), e["idx"] + 1)
            for kind, nxt in counters.items():
                c.execute("INSERT INTO counters(kind, next) VALUES(?, ?) "
                          "ON CONFLICT(kind) DO UPDATE SET next=excluded.next", (kind, nxt))
            for term in payload["clear"]:
                c.execute("INSERT OR IGNORE INTO clear_terms(norm_mac, norm_enc, created) VALUES(?,?,?)",
                          (self._mac(term), self._enc(term), now))
            for d in payload["documents"]:
                c.execute("INSERT OR IGNORE INTO documents(id, path_mac, path_enc, created) VALUES(?,?,?,?)",
                          (d["id"], self._mac(d["rel"]), self._enc(d["rel"]), now))

    # ---------------------------------------------------------------- lifecycle
    def close(self) -> None:
        self._db.close()
        self._snap = None


def _hidden_name(name: str) -> bool:
    return name.startswith((".", "~")) or name.endswith(("~", ".tmp", ".part", ".crdownload"))


def _kind_of(path: Path) -> str:
    """Reader kind of a file ("pdf", "docx"...), or ``"unsupported"`` when Shield cannot read it."""
    return KIND_BY_EXT.get(path.suffix.lower(), UNSUPPORTED_KIND)


def _hidden_attr(path: Path) -> bool:
    """Windows hidden/system attribute (always False elsewhere)."""
    try:
        attrs = getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0)
    except OSError:
        return False
    return bool(attrs & (getattr(stat, "FILE_ATTRIBUTE_HIDDEN", 2) | getattr(stat, "FILE_ATTRIBUTE_SYSTEM", 4)))


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def _folder_key(folder: Path) -> str:
    return unicodedata.normalize("NFC", str(folder))


# ------------------------------------------------------------------------------------------ registry

class Registry:
    """All dossiers on this computer. Thread-safe; several processes may open it at the same time."""

    def __init__(self, home: Optional[os.PathLike | str] = None):
        if home is None:
            self.home = config.home()
        else:
            self.home = Path(home).expanduser()
            self.home.mkdir(parents=True, exist_ok=True)
            if os.name == "posix":
                try:
                    os.chmod(self.home, 0o700)
                except OSError:
                    pass
        self.vaults_dir = self.home / VAULTS_DIR_NAME
        self.vaults_dir.mkdir(exist_ok=True)
        if os.name == "posix":
            try:
                os.chmod(self.vaults_dir, 0o700)
            except OSError:
                pass
        fresh = not (self.home / REGISTRY_FILE_NAME).exists()
        with _creation_lock(self.home) if fresh else nullcontext():
            try:
                self._db = _Db(self.home / REGISTRY_FILE_NAME)
            except sqlite3.Error:
                raise VaultError("Registre illisible (registry unreadable)") from None
            try:
                key, self.key_source = self._open_key(locked=fresh)
            except BaseException:
                self._db.close()
                raise
        self._fernet = Fernet(key)
        self._mac_key = _subkey(key, b"registry index v1")
        self._lock = threading.RLock()
        self._open: dict[int, Dossier] = {}

    def __repr__(self) -> str:
        return "Registry()"

    def _check_token(self) -> Optional[bytes]:
        rows = self._db.query("SELECT v FROM meta WHERE k='key_check'")
        return bytes(rows[0][0]) if rows else None

    def _open_key(self, locked: bool = False) -> tuple[bytes, str]:
        """Master key that matches this registry. A fresh registry stores a check token for its key.

        The first key is created under the creation lock (see :func:`_creation_lock`), so two processes that
        open a fresh home together end with ONE key. After the check token is stored, the keychain is read
        again: if it no longer holds the key in use, that key is saved to the 0600 key file so it is never lost.
        """
        try:
            self._db.schema(_REGISTRY_SCHEMA)
            check = self._check_token()
        except sqlite3.Error:
            raise VaultError("Registre illisible (registry unreadable)") from None
        if check is not None:
            return _master_key(self.home, check)  # may ask the OS keychain: no DB lock held here
        if not locked:
            with _creation_lock(self.home):
                return self._open_key(locked=True)
        key, source = _master_key(self.home, None)
        with self._db.write() as c:
            c.execute("INSERT OR IGNORE INTO meta(k, v) VALUES('schema', ?)", (SCHEMA_VERSION,))
            c.execute("INSERT OR IGNORE INTO meta(k, v) VALUES('key_check', ?)", (Fernet(key).encrypt(_KEY_CHECK),))
        stored = self._check_token()
        if stored is None or not _key_matches(key, stored):  # another process won the race
            return _master_key(self.home, stored)
        if source == "keyring" and _keyring_key(False) != key:
            if _save_key_file(self.home / KEY_FILE_NAME, key):
                log.warning("the OS keychain changed during the first start; the vault key was saved to the key file")
                source = "file"
        return key, source

    def __enter__(self) -> "Registry":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            for d in self._open.values():
                d.close()
            self._open.clear()
            self._db.close()

    # helpers
    def _enc(self, text: str) -> bytes:
        return self._fernet.encrypt(text.encode("utf-8"))

    def _dec(self, blob: bytes) -> str:
        try:
            return self._fernet.decrypt(bytes(blob)).decode("utf-8")
        except InvalidToken:
            raise VaultError("Registre : déchiffrement impossible (registry decryption failed)") from None

    def _mac(self, text: str) -> bytes:
        return hmac.new(self._mac_key, text.encode("utf-8"), hashlib.sha256).digest()

    def _vault_path(self, dossier_uuid: str) -> Path:
        return self.vaults_dir / f"{dossier_uuid}.db"

    def _open_row(self, row: tuple) -> Dossier:
        num, duuid, folder_enc, title_enc, locale, key_enc, created = row
        d = self._open.get(num)
        if d is not None:
            return d
        path = self._vault_path(duuid)
        if not path.exists():
            raise VaultError(f"DOSSIER_{num}: coffre introuvable (vault file missing)")
        key = self._dec(key_enc).encode("ascii")
        d = Dossier(num=num, uuid=duuid, folder=Path(self._dec(folder_enc)), title=self._dec(title_enc),
                    locale=locale, data_key=key, db_path=path, created=created)
        self._open[num] = d
        return d

    _ROW = "SELECT num, uuid, folder_enc, title_enc, locale, key_enc, created FROM dossiers"

    # public API
    def add_dossier(self, folder: os.PathLike | str, title: Optional[str] = None,
                    on_create: Optional[Callable[[str, str], None]] = None) -> Dossier:
        """Register a folder as a dossier (returns DOSSIER_n). Registering the same folder again returns it.

        v0.4.3: ``on_create(code, uuid)`` runs inside the write transaction of a NEW dossier, before the row is
        committed (the plugin writes its "not read yet" mark there). If it raises, nothing is registered."""
        path = Path(folder).expanduser()
        try:
            path = path.resolve(strict=True)
        except (OSError, RuntimeError):
            raise VaultError("Le dossier n'existe pas (folder not found)") from None
        if not path.is_dir():
            raise VaultError("Ce chemin n'est pas un dossier (not a folder)")
        name = " ".join((title if title is not None else path.name).split()) or path.name
        d, created = self._create(path, name, config.locale(), reuse=True, on_create=on_create)
        if created:
            self.journal("dossier_added", dossier=d.code)
        return d

    def _folder_registered(self, path: Path) -> bool:
        return bool(self._db.query("SELECT 1 FROM dossiers WHERE folder_mac=?", (self._mac(_folder_key(path)),)))

    def _create(self, folder: Path, title: str, locale: str, reuse: bool,
                on_create: Optional[Callable[[str, str], None]] = None) -> tuple[Dossier, bool]:
        """Insert a dossier row and its empty vault. The folder check runs inside the write transaction, so
        two threads or processes cannot register the same folder twice. Returns (dossier, created)."""
        with self._lock:
            duuid = uuidlib.uuid4().hex
            data_key = Fernet.generate_key()
            folder_mac = self._mac(_folder_key(folder))
            with self._db.write() as c:
                row = c.execute("SELECT num FROM dossiers WHERE folder_mac=?", (folder_mac,)).fetchone()
                if row is None:
                    cur = c.execute(
                        "INSERT INTO dossiers(uuid, folder_mac, folder_enc, title_enc, locale, key_enc, created)"
                        " VALUES(?,?,?,?,?,?,?)",
                        (duuid, folder_mac, self._enc(str(folder)), self._enc(title), locale,
                         self._enc(data_key.decode("ascii")), time.time()))
                    num = cur.lastrowid
                    if on_create is not None:
                        on_create(f"DOSSIER_{num}", duuid)  # raises: the transaction is rolled back
            if row is not None:
                if not reuse:
                    raise VaultError("Ce dossier est déjà enregistré (folder already registered)")
                return self._open_row(self._db.query(self._ROW + " WHERE num=?", (row[0],))[0]), False
            try:
                _private_file(self._vault_path(duuid))  # empty vault file, 0600, before SQLite opens it
                return self._open_row(self._db.query(self._ROW + " WHERE num=?", (num,))[0]), True
            except BaseException:
                with self._db.write() as c:
                    c.execute("DELETE FROM dossiers WHERE num=?", (num,))
                _unlink_vault(self._vault_path(duuid))
                raise

    def set_title(self, code: str, title: str) -> None:
        """Change the stored (encrypted) title of a dossier: the user typed a new one."""
        num = self._num(code)
        name = " ".join(str(title or "").split())
        if not name:
            raise ValueError("empty title")
        with self._lock:
            with self._db.write() as c:
                cur = c.execute("UPDATE dossiers SET title_enc=? WHERE num=?", (self._enc(name), num))
                if cur.rowcount == 0:
                    raise UnknownDossier(f"DOSSIER_{num} : dossier inconnu (unknown dossier)")
            if num in self._open:
                self._open[num].title = name

    def is_registered(self, folder: os.PathLike | str) -> bool:
        """True when this folder is already a dossier (registry only, no vault opened)."""
        try:
            path = Path(folder).expanduser().resolve(strict=True)
        except (OSError, RuntimeError):
            return False
        return self._folder_registered(path)

    def list(self) -> list[Dossier]:
        """All dossiers, by number."""
        with self._lock:
            rows = self._db.query(self._ROW + " ORDER BY num")
            alive = {r[0] for r in rows}
            for num in [n for n in self._open if n not in alive]:  # removed by another process
                self._open.pop(num).close()
            return [self._open_row(r) for r in rows]

    @staticmethod
    def _num(code: str) -> int:
        m = _DOSSIER_RE.match(str(code).strip().upper())
        if not m:
            raise UnknownDossier("Code de dossier invalide (invalid dossier code)")
        return int(m.group(1))

    def get(self, code: str) -> Dossier:
        """The dossier "DOSSIER_n". Raises UnknownDossier (or VaultError if its vault cannot be opened)."""
        num = self._num(code)
        with self._lock:
            rows = self._db.query(self._ROW + " WHERE num=?", (num,))
            if not rows:
                stale = self._open.pop(num, None)
                if stale is not None:
                    stale.close()
                raise UnknownDossier(f"DOSSIER_{num} : dossier inconnu (unknown dossier)")
            return self._open_row(rows[0])

    def folders(self) -> dict[str, Path]:
        """{"DOSSIER_n": folder} from the registry only (no vault is opened). REAL paths: local use only.

        Cheap enough for hooks that run on every tool call, and still works when a vault file is damaged.
        """
        with self._lock:
            rows = self._db.query("SELECT num, folder_enc FROM dossiers ORDER BY num")
            return {f"DOSSIER_{num}": Path(self._dec(enc)) for num, enc in rows}

    def dossier_code_for_path(self, path: os.PathLike | str) -> Optional[str]:
        """Code of the dossier whose folder contains this path (deepest folder wins), or None.

        Uses the registry only (no vault is opened), for the PreToolUse hook.
        """
        try:
            target = Path(path).expanduser().resolve()
        except (OSError, RuntimeError):
            return None
        best: Optional[tuple[str, Path]] = None
        for code, folder in self.folders().items():
            if _inside(target, folder) and (best is None or len(folder.parts) > len(best[1].parts)):
                best = (code, folder)
        return best[0] if best else None

    def dossier_for_path(self, path: os.PathLike | str) -> Optional[Dossier]:
        """The dossier whose folder contains this path (deepest folder wins), or None."""
        code = self.dossier_code_for_path(path)
        return self.get(code) if code else None

    def remove(self, code: str) -> None:
        """Delete a dossier: its key and its vault file. Documents in the folder are not touched.

        Works from the registry row alone, so a dossier whose vault file is damaged can still be removed.
        """
        num = self._num(code)
        with self._lock:
            rows = self._db.query("SELECT uuid FROM dossiers WHERE num=?", (num,))
            if not rows:
                raise UnknownDossier(f"DOSSIER_{num} : dossier inconnu (unknown dossier)")
            opened = self._open.pop(num, None)
            if opened is not None:
                opened.close()
            with self._db.write() as c:
                c.execute("DELETE FROM dossiers WHERE num=?", (num,))
            try:
                self._db.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error:
                pass
            _unlink_vault(self._vault_path(rows[0][0]))
        self.journal("dossier_removed", dossier=f"DOSSIER_{num}")

    # journal
    def journal(self, event: str, dossier: Optional[str] = None, **counts: Any) -> None:
        """Registry-level event (dossier added, removed, imported). Codes and numbers only."""
        if dossier is not None and not _DOSSIER_RE.match(dossier):
            raise ValueError("journal: dossier must be a DOSSIER_n code")
        event, payload = _journal_record(event, counts)
        with self._db.write() as c:
            c.execute("INSERT INTO journal(time, event, dossier, counts) VALUES(?,?,?,?)",
                      (time.time(), event, dossier, payload))

    def journal_entries(self, code: Optional[str] = None, limit: Optional[int] = 500) -> list[dict]:
        """Registry and dossier events, newest first: [{time, event, dossier, counts}]."""
        if code is not None:
            d = self.get(code)
            rows = self._db.query("SELECT time, event, dossier, counts FROM journal WHERE dossier=?", (d.code,))
            entries = [(t, _journal_row(t, e, ds, c)) for t, e, ds, c in rows]
            entries += d._journal_pairs(limit)
        else:
            rows = self._db.query("SELECT time, event, dossier, counts FROM journal")
            entries = [(t, _journal_row(t, e, ds, c)) for t, e, ds, c in rows]
            for d in self.list():
                entries += d._journal_pairs(limit)
        entries.sort(key=lambda pair: pair[0], reverse=True)
        out = [x for _, x in entries]
        return out[:limit] if limit is not None else out

    # import
    def import_(self, path: os.PathLike | str, passphrase: str, folder: Optional[os.PathLike | str] = None,
                title: Optional[str] = None) -> Dossier:
        """Import a dossier exported with Dossier.export. Codes are kept; the dossier gets a new DOSSIER_n.

        ``folder``: where the colleague keeps the documents. Default: the original folder if it exists here
        and is not registered yet, else a new empty folder under ``<home>/imported/``.
        """
        payload = _read_export(Path(path), passphrase)
        if folder is not None:
            target = Path(folder).expanduser()
            try:
                target = target.resolve(strict=True)
            except (OSError, RuntimeError):
                raise VaultError("Le dossier n'existe pas (folder not found)") from None
            if not target.is_dir():
                raise VaultError("Ce chemin n'est pas un dossier (not a folder)")
            if self._folder_registered(target):
                raise VaultError("Ce dossier est déjà enregistré (folder already registered)")
        else:
            original = Path(payload["folder"]) if payload.get("folder") else None
            if original is not None and original.is_dir() and not self._folder_registered(original.resolve()):
                target = original.resolve()
            else:
                target = self.home / IMPORTED_DIR_NAME / uuidlib.uuid4().hex
                target.mkdir(parents=True, exist_ok=False)
        name = " ".join((title if title is not None else payload["title"]).split()) or "dossier"
        d, _ = self._create(target, name, payload["locale"], reuse=False)
        try:
            d._load_import(payload)
        except BaseException:
            self.remove(d.code)
            raise
        self.journal("dossier_imported", dossier=d.code, entities=len(payload["entities"]))
        return d


def _unlink_vault(path: Path) -> None:
    for suffix in ("", "-wal", "-shm", "-journal"):
        p = Path(str(path) + suffix)
        try:
            p.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            log.warning("vault file still in use; it stays on disk but its key is deleted")


def _read_export(path: Path, passphrase: str) -> dict:
    try:
        raw = path.read_bytes()
    except OSError:
        raise VaultError("Fichier d'export illisible (export file unreadable)") from None
    if not raw.startswith(_EXPORT_MAGIC):
        raise VaultError("Ce fichier n'est pas un export RAGbase Shield (not a RAGbase Shield export)")
    try:
        header_line, token = raw[len(_EXPORT_MAGIC):].split(b"\n", 1)
        header = json.loads(header_line)
        n, r, p = int(header["n"]), int(header["r"]), int(header["p"])
        salt = base64.b64decode(header["salt"])
        if header.get("kdf") != "scrypt" or not (2 ** 14 <= n <= 2 ** 20 and n & (n - 1) == 0) \
                or not (1 <= r <= 16 and 1 <= p <= 4) or len(salt) < 16:
            raise ValueError
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        raise VaultError("Fichier d'export endommagé (export file damaged)") from None
    try:
        data = Fernet(_passphrase_key(str(passphrase), salt, n, r, p)).decrypt(token.strip())
    except InvalidToken:
        raise VaultError("Phrase de passe incorrecte ou fichier modifié (wrong passphrase or file changed)") from None
    try:
        payload = json.loads(data.decode("utf-8"))
        _validate_payload(payload)
    except (ValueError, KeyError, TypeError):
        raise VaultError("Contenu de l'export invalide (invalid export content)") from None
    return payload


def _validate_payload(payload: dict) -> None:
    """Accepts export versions 1 (v0.1) and 2 (v0.2: ``synthetic`` per entity, ``shapes``, ``settings``)."""
    version = payload.get("version")
    if payload.get("format") != _EXPORT_FORMAT or version not in (1, 2):
        raise ValueError
    locale = payload["locale"]
    if locale not in ("fr", "en") or not isinstance(payload["title"], str):
        raise ValueError
    codes, keys, idxs, synths = set(), set(), set(), set()
    for e in payload["entities"]:
        kind, idx = e["kind"], e["idx"]
        if kind not in KINDS or not isinstance(idx, int) or idx < 0 or e["code"] != make_code(kind, idx, locale):
            raise ValueError
        if not isinstance(e["value"], str) or not e["value"] or not isinstance(e["norm"], str) or not e["norm"]:
            raise ValueError
        if e["source"] not in ENTITY_SOURCES or e["code"] in codes or (kind, e["norm"]) in keys \
                or (kind, idx) in idxs:
            raise ValueError
        created = e.get("created")
        if created is not None and (isinstance(created, bool) or not isinstance(created, (int, float))):
            raise ValueError
        synth = e.get("synthetic")
        if synth is not None:
            if version < 2 or not isinstance(synth, str) or not synth.strip() or normalize(synth) in synths:
                raise ValueError
            synths.add(normalize(synth))
        codes.add(e["code"])
        keys.add((kind, e["norm"]))
        idxs.add((kind, idx))
    if version >= 2:
        for sh in payload.get("shapes") or []:
            if sh["kind"] not in KINDS or not isinstance(sh["template"], str) or not sh["template"].strip() \
                    or len(sh["template"]) > 64:
                raise ValueError
        settings = payload.get("settings") or {}
        if not isinstance(settings, dict):
            raise ValueError
        for key, value in settings.items():
            if key not in DOSSIER_SETTINGS or value not in DOSSIER_SETTINGS[key]:
                raise ValueError
        evidence = payload.get("evidence")
        if evidence is not None:
            if not isinstance(evidence, dict):
                raise ValueError
            for key, counts in evidence.items():
                if not isinstance(key, str) or not key or len(key) > _MAX_EVIDENCE_KEY or not isinstance(counts, dict):
                    raise ValueError
                for f, n in counts.items():
                    if f not in _EVIDENCE_FIELDS or isinstance(n, bool) or not isinstance(n, int) or n < 0:
                        raise ValueError
        directory = payload.get("directory")
        if directory is not None:
            if not isinstance(directory, dict) or not isinstance(directory.get("persons", []), list):
                raise ValueError
            for person in directory.get("persons") or []:
                if not isinstance(person, dict) or not isinstance(person.get("value"), str) or not person["value"]:
                    raise ValueError
    for kind, nxt in payload["counters"].items():
        top = max((i for k, i in idxs if k == kind), default=-1)
        if kind not in KINDS or not isinstance(nxt, int) or nxt <= top:
            raise ValueError
    for term in payload["clear"]:
        if not isinstance(term, str) or not term:
            raise ValueError
    ids = set()
    for d in payload["documents"]:
        rel = d["rel"]
        if not isinstance(d["id"], int) or d["id"] < 1 or d["id"] in ids or not isinstance(rel, str):
            raise ValueError
        parts = Path(rel).parts
        if not rel or Path(rel).is_absolute() or ".." in parts or rel.startswith(("/", "\\")) or ":" in rel[:3]:
            raise ValueError
        ids.add(d["id"])
