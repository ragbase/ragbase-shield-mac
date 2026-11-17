"""v0.4.4 (A): the dossier stays ready between starts: its MASKED texts are kept, encrypted, in the data folder.

The Claude app starts one Shield server per conversation. Without a kept set, every start reads and masks every
document of the dossier again before any document can be read (76 to 112 s for the 800-page test dossier). At the
end of a complete reading, the service writes here, for that dossier, the masked text of each document and the other
derived values it needs to answer (alignment for the search, page counts, states, masked file names, counts), plus a
manifest with three fingerprints. At the next start the dossier is ready WITHOUT reading any file of the lawyer when
the files, the engine and the vault state are the same (service.Service._load_kept). All or nothing per dossier.

Layout, under the data folder (names: codes and random ids only, never a real name or a file name of the lawyer):

    kept/DOSSIER_<n>/m-<20 digits>-<gen>.bin     the manifest of one set (newest name wins)
    kept/DOSSIER_<n>/<gen>.<i>.bin               entry i of that set (one per read document)

``<gen>`` is 8 random hex digits, new for each set. Every file is ``Fernet(dossier data key)`` (the key and the
cipher of the vault) over zlib-compressed bytes. The manifest lists each entry with the SHA-256 of its compressed
bytes. Writes: entries first (temporary file in the same folder, then a rename to a NEW name), the manifest LAST
(same way, under a new name), then the older sets are deleted. A reader only accepts a set through its manifest, so
it gets an old complete set, a new complete set, or nothing (full reading): never a mix. No file is ever renamed
over a file that another process may hold open (Windows), and a failed rename is tried again a few times.

Safety: the kept text is the masked text (masker output and the text after the last check), never the original
text. The search text ("shadow") is the original text with every replaced character changed to ``x`` (or ``*``):
outside the replaced spans it is the text the masked text already holds; it keeps the place and the kind (letter,
space, mark) of each replaced character only, so the search gives byte for byte the same results as on the
original text. It is all encrypted anyway: a kept text can hold a name that the rules did not find.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import sys
import time
import unicodedata
import zlib
from pathlib import Path
from typing import Any, Iterable, Optional

from . import __version__, config, documents

log = logging.getLogger(__name__)

KEPT_DIR_NAME = "kept"
#: Format of the kept files (bump when the layout of a manifest or an entry changes).
FORMAT = 1
#: The "rules fingerprint" constant. BUMP IT whenever the masked output can change for the same files and the same
#: vault state: a rule, a word list, a reader, a limit, the order of the reading. The engine version text is part of
#: the fingerprint too, so a new version never uses a kept set of another one.
RULES_REVISION = "0.4.4-r3"  # r3: the birth place fix (BIRTHPLACE never spreads)
_MANIFEST_RE = re.compile(r"^m-([0-9]{20})-([0-9a-f]{8})\.bin$")
_ENTRY_RE = re.compile(r"^([0-9a-f]{8})\.([0-9]{1,6})\.bin$")
_TMP_SUFFIX = ".tmp"
#: Windows: a rename can fail while an antivirus or another process holds the file; tried this many times.
REPLACE_TRIES = 5
REPLACE_WAIT = 0.1
#: Files that no manifest names (a set whose writer stopped) are deleted after this many seconds.
ORPHAN_SECONDS = 3600.0
ZLIB_LEVEL = 6


# ------------------------------------------------------------------------------------------ fingerprints

def _sha(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:  # noqa: BLE001 - a setting that cannot be read is part of the fingerprint as "?"
        return default if default is not None else "?"


def rules_fingerprint() -> str:
    """(b) Everything outside the vault that can change the masked output of the same files: the engine version, the
    rules revision constant, the reading limits and readers, the settings of the masker and the Python text tables."""
    from . import service

    try:
        from .engine import ner as _ner

        ner_cached = _safe(lambda: bool(_ner.model_cached(config.ner_model())), False)
    except Exception:  # noqa: BLE001
        ner_cached = "?"
    return _sha({
        "format": FORMAT, "engine": __version__, "rules": RULES_REVISION,
        "limits": [service.MAX_DOCUMENT_CHARS, service.MAX_MASK_CHARS, documents.PDF_PLUMBER_MAX_PAGES,
                   documents.PDF_PLUMBER_MAX_PAGE_BYTES, documents.PDF_PLUMBER_MAX_BYTES,
                   documents.PDF_CELL_SEPARATOR, documents.PDF_CELL_GAP_FACTOR, documents.PDF_LINE_TOLERANCE],
        "pdf_tables": _safe(documents._pdf_tables_enabled),
        "settings": [_safe(config.locale), _safe(config.replacement), _safe(config.language), _safe(config.ner_setting),
                     _safe(config.ner_model), _safe(config.mask_legal_form), _safe(config.lowercase_names),
                     _safe(config.own_company_vendor), ner_cached],
        "python": list(sys.version_info[:2]), "unicode": unicodedata.unidata_version,
    })


def lexicon_fingerprint(d: Any, evidence: bool = True) -> str:
    """(c) Everything in the vault state of dossier ``d`` that can change its masked output: its entities (kind,
    code, key, value, source, fake value), the terms kept in clear, the identifier layouts, the directory, the
    settings, the title, the folder, the locale, the id, and (review fix) the "evidence" counts (vault table
    ``evidence``). When in doubt, more is included.

    The evidence decides whether a disputed name is masked at all (masker ``_evidence_pass``), and it can change from
    outside the documents: the Desk, the command line, the browser extension and the hooks mask other texts with the
    dossier's own masker. Measured: the answers of a kept dossier (list, reads, searches, saved draft, status) do not
    add to it, so a kept dossier stays ready after them (test).

    ``evidence=False``: for the guard fingerprint of the other dossiers. The last check reads their values only (the
    canary ``scan`` and the label pass never use evidence), so their evidence cannot change this dossier's output."""
    ents = [[e.kind, e.code, e.norm, e.value, getattr(e, "source", ""), _safe(lambda e=e: d.synthetic(e.code), "")]
            for e in d.all()]
    state = {
        "uuid": d.uuid, "code": d.code, "locale": d.locale, "title": d.title, "folder": str(d.folder),
        "entities": ents,
        "clear": sorted(_safe(d.kept_clear_terms, [])),
        "shapes": [list(x) for x in _safe(d.shapes, [])],
        "directory": _safe(d.directory, {}),
        "settings": _safe(d.settings, {}),
    }
    if evidence:
        state["evidence"] = _safe(d.evidence, {})
    return _sha(state)


def guard_fingerprint(dossiers: Iterable[Any]) -> str:
    """The values of EVERY dossier that the last check uses (a document text is checked against all of them): the
    lexicon fingerprint (without evidence) of each dossier that has entities. A new, changed or removed dossier
    changes it."""
    return _sha(sorted([d.code, d.uuid, lexicon_fingerprint(d, evidence=False)] for d in dossiers if d.all()))


# ------------------------------------------------------------------------------------------ store

def _replace(src: Path, dst: Path) -> None:
    """os.replace with a few tries (Windows: a file held by an antivirus or another process)."""
    for attempt in range(REPLACE_TRIES):
        try:
            os.replace(src, dst)
            return
        except OSError:
            if attempt + 1 >= REPLACE_TRIES:
                raise
            time.sleep(REPLACE_WAIT * (2 ** attempt))


def _unlink(path: Path) -> bool:
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return True
    except OSError:
        return False  # Windows: held open by another process; deleted at a later write or start


class KeptStore:
    """The kept sets of one data folder. Several server processes may use it at the same time."""

    def __init__(self, home: os.PathLike | str):
        self.root = Path(home) / KEPT_DIR_NAME
        self._token = secrets.token_hex(4)

    def folder(self, code: str) -> Path:
        return self.root / code

    def _manifests(self, code: str) -> list[Path]:
        try:
            names = os.listdir(self.folder(code))
        except OSError:
            return []
        return [self.folder(code) / n for n in sorted((n for n in names if _MANIFEST_RE.match(n)), reverse=True)]

    def _write_file(self, folder: Path, name: str, data: bytes) -> Path:
        tmp = folder / f"{name}.{self._token}{_TMP_SUFFIX}"
        try:
            with open(tmp, "wb") as fh:
                fh.write(data)
            _replace(tmp, folder / name)
        except BaseException:
            _unlink(tmp)
            raise
        return folder / name

    # ---------------------------------------------------------------- write
    def write(self, d: Any, manifest: dict, payloads: dict[str, bytes]) -> bool:
        """Write a new set for dossier ``d``: ``payloads`` {key: compressed bytes} become the entries, named in the
        manifest under "entries" {key: [file, sha256]}. Returns False (and leaves no new file) when it cannot be
        written; the reading does not fail for that, the dossier is then read again at its next start."""
        folder = self.folder(d.code)
        gen = secrets.token_hex(4)
        written: list[Path] = []
        try:
            folder.mkdir(parents=True, exist_ok=True)
            entries: dict[str, list[str]] = {}
            for i, (key, payload) in enumerate(payloads.items()):
                name = f"{gen}.{i}.bin"
                written.append(self._write_file(folder, name, d.seal(payload)))
                entries[key] = [name, hashlib.sha256(payload).hexdigest()]
            body = dict(manifest, entries=entries, gen=gen, format=FORMAT)
            raw = zlib.compress(json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8"), ZLIB_LEVEL)
            older = self._manifests(d.code)
            mname = f"m-{time.time_ns():020d}-{gen}.bin"
            written.append(self._write_file(folder, mname, d.seal(raw)))  # LAST: the set exists from now on
        except Exception as exc:  # noqa: BLE001 - never fails the reading
            log.warning("RAGbase Shield: %s: kept set not written (%s)", d.code, type(exc).__name__)
            for p in written:
                _unlink(p)
            for p in self._manifests(d.code):  # the next start must read the dossier again
                _unlink(p)
            return False
        self._drop_sets(d.code, older, keep_gen=gen)
        return True

    def _drop_sets(self, code: str, manifests: list[Path], keep_gen: str) -> None:
        """Delete older manifests, then the entries of their sets (never the files of ``keep_gen``)."""
        folder = self.folder(code)
        gens = set()
        for p in manifests:
            m = _MANIFEST_RE.match(p.name)
            if m and m.group(2) != keep_gen:
                if _unlink(p):
                    gens.add(m.group(2))
        if not gens:
            return
        try:
            names = os.listdir(folder)
        except OSError:
            return
        for n in names:
            m = _ENTRY_RE.match(n)
            if m and m.group(1) in gens:
                _unlink(folder / n)

    # ---------------------------------------------------------------- read
    def load(self, d: Any) -> Optional[tuple[dict, dict[str, bytes]]]:
        """(manifest, {key: compressed bytes}) of the newest complete set of dossier ``d`` whose files all decrypt and
        match their hashes, or None. A missing, damaged or foreign file counts as absent: never an error."""
        for _round in range(2):  # a newer writer may delete the set while it is read: look once more
            manifests = self._manifests(d.code)
            if not manifests:
                return None
            got = self._load_set(d, manifests[0])
            if got is not None:
                return got
        return None

    def _load_set(self, d: Any, mpath: Path) -> Optional[tuple[dict, dict[str, bytes]]]:
        folder = mpath.parent
        try:
            manifest = json.loads(zlib.decompress(d.unseal(mpath.read_bytes())).decode("utf-8"))
            if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
                return None
            gen = _MANIFEST_RE.match(mpath.name).group(2)
            if manifest.get("gen") != gen:
                return None
            payloads: dict[str, bytes] = {}
            for key, (name, digest) in (manifest.get("entries") or {}).items():
                m = _ENTRY_RE.match(str(name))
                if not m or m.group(1) != gen:
                    return None
                payload = d.unseal((folder / name).read_bytes())
                if hashlib.sha256(payload).hexdigest() != digest:
                    return None
                payloads[key] = payload
            return manifest, payloads
        except Exception as exc:  # noqa: BLE001 - absent, cut short, damaged, another dossier's: a full reading
            log.info("RAGbase Shield: %s: kept set not usable (%s)", d.code, type(exc).__name__)
            return None

    # ---------------------------------------------------------------- delete, clean, size
    def delete(self, code: str) -> None:
        """Delete every kept file of a dossier (best effort: a file held open on Windows goes at a later start)."""
        folder = self.folder(code)
        if not folder.is_dir():
            return
        shutil.rmtree(folder, ignore_errors=True)

    def clean(self, registered: dict[str, Any]) -> list[str]:
        """At a start: delete the sets of dossiers that no longer exist, and the files that no manifest names (a
        writer that stopped) once they are older than :data:`ORPHAN_SECONDS`. Returns the codes deleted."""
        gone = []
        try:
            names = os.listdir(self.root)
        except OSError:
            return gone
        now = time.time()
        for code in names:
            folder = self.root / code
            if not folder.is_dir():
                continue
            if code not in registered:
                self.delete(code)
                gone.append(code)
                continue
            try:
                files = os.listdir(folder)
            except OSError:
                continue
            gens = {m.group(2) for m in (_MANIFEST_RE.match(n) for n in files) if m}
            for n in files:
                m = _ENTRY_RE.match(n)
                orphan = (m is not None and m.group(1) not in gens) or n.endswith(_TMP_SUFFIX)
                if orphan:
                    with contextlib.suppress(OSError):
                        if now - (folder / n).stat().st_mtime > ORPHAN_SECONDS:
                            _unlink(folder / n)
        return gone

    def size(self, code: str) -> int:
        """Bytes on disk of the kept files of a dossier."""
        total = 0
        try:
            for n in os.listdir(self.folder(code)):
                with contextlib.suppress(OSError):
                    total += (self.folder(code) / n).stat().st_size
        except OSError:
            pass
        return total


# ------------------------------------------------------------------------------------------ entries

def pack(meta: dict, texts: dict[str, Optional[str]]) -> bytes:
    """One entry: a JSON head (``meta`` and the byte length of each text) and the UTF-8 texts, zlib-compressed."""
    order = [k for k in sorted(texts) if texts[k] is not None]
    blobs = [texts[k].encode("utf-8", "surrogatepass") for k in order]
    head = json.dumps({"meta": meta, "texts": [[k, len(b)] for k, b in zip(order, blobs)]},
                      ensure_ascii=True, sort_keys=True).encode("ascii")
    return zlib.compress(head + b"\n" + b"".join(blobs), ZLIB_LEVEL)


def unpack(payload: bytes) -> tuple[dict, dict[str, str]]:
    raw = zlib.decompress(payload)
    cut = raw.index(b"\n")
    head = json.loads(raw[:cut].decode("ascii"))
    pos = cut + 1
    texts: dict[str, str] = {}
    for key, n in head["texts"]:
        texts[key] = raw[pos:pos + n].decode("utf-8", "surrogatepass")
        pos += n
    if pos != len(raw):
        raise ValueError("entry length")
    return head["meta"], texts
