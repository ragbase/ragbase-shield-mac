"""High-level operations used by the MCP server, the hooks, the CLI and Shield Desk (ARCHITECTURE.md 4.4).

Two kinds of callers:

- **Model-facing** (MCP tools, hook outputs): :func:`list_dossiers`, :func:`list_documents`,
  :func:`read_document`, :func:`search`, :func:`save_draft`, :func:`status`, :func:`scan_text`.
  They return codes and masked text only. Every model-facing string also goes through a last canary:
  a registered value of any dossier that is still present is replaced by its code (or by ``[PERSONNE]``)
  before it leaves this module.
- **Local only** (Desk page, CLI in the user's terminal): :func:`mask_text` with ``include_values``,
  :func:`restore_text`, :func:`entities`, ``list_dossiers(include_private=True)``... These may return real
  values because they never reach a model.

Fail closed: every public function catches all errors and raises :class:`ShieldError`, whose message holds
codes only (DOSSIER_n, DOC_n, PERSONNE_A) and short error codes, never a value, a title, a path or document
text. The original error is logged locally with its type and code location only.
"""
from __future__ import annotations

import bisect
import contextlib
import contextvars
import functools
import logging
import os
import re
import sys
import threading
import time
import traceback
import unicodedata
import uuid as uuidlib
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from . import __version__, config, documents
from . import kept as _kset
from .engine import (
    CODE_RE,
    KINDS,
    UNMASK_RE,
    Entity,
    Masker,
    MemoryStore,
    Span,
    entity_norm,
    fold,
    has_party_separator,
    normalize,
    parse_parties,
    prefix_for,
)
from .engine import ner as _ner
from .engine.masker import ScanPrep, prepare_scan
from .engine.types import PREFIXES, is_weak_value, spreadable
from .vault import DOSSIER_SETTINGS, UNSUPPORTED_KIND, Dossier, Registry, UnknownDossier, VaultError

log = logging.getLogger(__name__)

#: F0 (0.4.3): Shield's own tokens in a text: entity codes (as ``UNMASK_RE`` reads them, also glued), DOC_n,
#: DOSSIER_n, OUTPUT_n and the placeholders "[PERSONNE]", "[IDENT]". The last check never changes them: a hit of
#: the canary that touches one of them is dropped.
SHIELD_TOKEN_RE = re.compile(
    r"(?:" + UNMASK_RE.pattern
    + r"|(?<![A-Za-z0-9])(?:DOC|DOSSIER|OUTPUT)_[0-9]{1,7}(?![0-9])"
    + r"|\[[A-ZÀ-Ý]{2,}(?:[_ -][A-ZÀ-Ý]+)*\])"
)


def shield_token_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of every Shield code, structural token and placeholder of ``text`` (F0, 0.4.3)."""
    return [m.span() for m in SHIELD_TOKEN_RE.finditer(text or "")]


def _touches(spans: list[tuple[int, int]], start: int, end: int) -> bool:
    return any(a < end and start < b for a, b in spans)


__all__ = [
    "ShieldError",
    "Service",
    "default",
    "reset",
    "list_dossiers",
    "list_documents",
    "read_document",
    "search",
    "save_draft",
    "save_draft_report",
    "mask_text",
    "set_dossier_setting",
    "restore_text",
    "restore_report",
    "scan_text",
    "has_registered",
    "status",
    "add_dossier",
    "scan_dossier",
    "remove_dossier",
    "entities",
    "register",
    "keep_clear",
    "journal",
    "mask_file",
    "protect",
    "output_dossier",
    "LIMITS",
]

DOSSIER_RE = re.compile(r"^DOSSIER_([0-9]{1,6})$")
DOC_RE = re.compile(r"^DOC_([0-9]{1,7})$")

DEFAULT_READ_CHARS = 20_000
MIN_READ_CHARS = 500
MAX_READ_CHARS = 50_000
MAX_QUERY_CHARS = 500
DEFAULT_SEARCH_RESULTS = 10
MAX_SEARCH_RESULTS = 50
SNIPPET_RADIUS = 160
MAX_DRAFT_CHARS = 2_000_000
#: v0.4.4 (B1): 6,000,000 characters (was 5,000,000), the same as MAX_DOCUMENT_CHARS: a document of that size is
#: read, masked, listed, searched and readable.
MAX_MASK_CHARS = 6_000_000
#: Biggest document text Shield reads. v0.4.4 (B1): 6,000,000 characters (about 2,300 pages; was 3,000,000). The
#: reading runs in the background, so the size is no longer bound by the time limit of one tool call.
MAX_DOCUMENT_CHARS = 6_000_000
MAX_TITLE_CHARS = 200
_CACHE_DOCS = 64
#: v0.4.3: folder (in the data folder) of the marks "the reading of this dossier is not finished". One small file per
#: dossier, named by its code, holding its random id only (no name, no path). A dossier with a mark is never readable;
#: its reading starts again by itself after a stop.
READING_DIR_NAME = "reading"
_READING_SUFFIX = ".pending"
#: v0.4.3: a reading that fails this many times in one process is not started again by itself (the answer then asks
#: to restart the app), so Claude never waits for ever.
MAX_READING_FAILURES = 3
#: v0.4.3 (fix): the mark holds "<dossier id> <process id> <owner>". The Claude tab starts one server per
#: conversation, so several processes share one data folder: the owner touches its mark every MARK_TOUCH_SECONDS while
#: it reads; a mark untouched for MARK_STALE_SECONDS, or whose process is gone, is free (its reading starts again).
#: A fresh mark of another owner: no content, and no second reading.
MARK_TOUCH_SECONDS = 5.0
MARK_STALE_SECONDS = 60.0
#: v0.4.3 (fix): a reading whose current document (or step) goes past its time budget plus this margin is shown as
#: stopped (the answer asks to restart the app, so Claude stops waiting) and no longer holds back the other dossiers.
STUCK_MARGIN_SECONDS = 60.0
_MARK_WRITE_TRIES = 3
#: v0.4.4 (A): when the first call of a process for a ready dossier finds no usable kept set, the dossier is read
#: again in the background; the call waits up to this many seconds for that reading (a small dossier is then
#: answered at once), else it gives the progress answer.
KEPT_WAIT_SECONDS = 8.0
#: v0.4.4 (B2): the masking (and last check) of one document may take this long: the budget of its reading, and at
#: least 180 s plus 1 s for each 20,000 characters, at most documents.DOCUMENT_TIME_BUDGET_MAX.
MASK_CHARS_PER_SECOND = 20_000
#: v0.4.4 (F4): causes of a document that the last reading could not read, as kept in Service._not_read.
TRANSIENT_CAUSES = ("too_long", "locked", "denied", "io")

#: Honest limits, shown by ``shield_status`` and the Desk (French, then English).
LIMITS: tuple[tuple[str, str], ...] = (
    ("Pseudonymisation locale et réversible : ce n'est pas une anonymisation.",
     "Local, reversible pseudonymisation: this is not anonymisation."),
    ("La détection n'est jamais complète : relisez le texte masqué dans les cas sensibles.",
     "Detection is never complete: review the masked text for sensitive cases."),
    ("Pas d'OCR en v0.1 : un PDF scanné sans couche texte n'est pas lu.",
     "No OCR in v0.1: a scanned PDF without a text layer is not read."),
    ("Un texte collé ou un fichier joint dans la conversation part chez le fournisseur avant Shield.",
     "Text pasted or files attached in the chat reach the provider before Shield can act."),
    ("Un nom seul dans un tableau, sans civilité ni prénom connu, peut être manqué.",
     "A bare name in a table cell, with no title and no known first name, can be missed."),
    ("Les numéros de décisions publiées (pourvoi, ECLI) sont masqués comme ceux du client.",
     "Published case numbers (pourvoi, ECLI) are masked like the client's own."),
)

_CODE_SHAPE = re.compile(r"[A-Z]{2,12}_(?:[A-Z]{1,4}|[0-9]{1,6})")
_SENTINEL_RE = re.compile("[0-9]+")
_WORD_CHARS = re.compile(r"[a-z0-9]")
#: Generic marks the canary writes when it cannot give a code ("[PERSONNE]", "[COMPANY]"...): they cannot be
#: restored, so a saved draft that still holds one is reported (R1 correction CL-31).
_PLACEHOLDER_RE = re.compile(r"\[(?:" + "|".join(sorted({p for fr, en, _l in PREFIXES.values() for p in (fr, en)},
                                                         key=len, reverse=True)) + r")\]")
#: Shield object ids (not entity codes): restore never gives them a value (CL-36).
_SHIELD_ID_RE = re.compile(r"(?<![A-Za-z0-9_])(?:DOSSIER_[0-9]{1,6}|DOC_[0-9]{1,7}|OUTPUT_[0-9]{1,7})(?![A-Za-z0-9_])")
#: Dossier of the model-facing string being guarded (set by the MCP server around its last canary, CL-26).
_OUTPUT_DOSSIER: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("ragbase_shield_output_dossier",
                                                                               default=None)


class ShieldError(Exception):
    """A Shield operation failed. The message holds codes only (never a value, a title or a path).

    ``kind`` is a short machine code: ``invalid_argument``, ``unknown_dossier``, ``unknown_document``,
    ``document_unreadable``, ``vault_error``, ``write_failed``, ``internal``...
    """

    def __init__(self, message: str, kind: str = "error"):
        super().__init__(message)
        self.kind = kind


# ------------------------------------------------------------------------------------------ helpers

def _log_failure(op: str, exc: BaseException) -> str:
    """Log a failure locally with its type and code location only (no message: it may hold a value)."""
    ref = uuidlib.uuid4().hex[:8]
    frames = traceback.extract_tb(exc.__traceback__)[-4:]
    where = " < ".join(f"{Path(f.filename).name}:{f.lineno}:{f.name}" for f in reversed(frames))
    log.warning("RAGbase Shield: %s failed (ref %s): %s at %s", op, ref, type(exc).__name__, where or "?")
    return ref


def _guarded(op: str) -> Callable:
    """Decorator: every error leaves as a ShieldError with a code-only message (fail closed)."""

    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except ShieldError:
                raise
            except UnknownDossier:
                raise ShieldError("Dossier inconnu (unknown dossier). Appelez shield_list_dossiers.",
                                  "unknown_dossier") from None
            except VaultError:
                ref = _log_failure(op, _current_exc())
                raise ShieldError(f"Coffre RAGbase Shield illisible ou clé absente (vault error, ref {ref}). "
                                  "Lancez « ragbase-shield doctor ».", "vault_error") from None
            except documents.DocumentError as exc:
                code = str(exc) if re.fullmatch(r"[a-z_]{3,40}(?::[0-9]{1,6})?", str(exc)) else "unreadable"
                raise ShieldError(f"Document illisible ({code}).", "document_unreadable") from None
            except Exception:  # noqa: BLE001 - fail closed, never echo the message
                ref = _log_failure(op, _current_exc())
                raise ShieldError(f"Erreur interne de RAGbase Shield (internal error, ref {ref}).",
                                  "internal") from None

        return wrapper

    return deco


def _current_exc() -> BaseException:
    import sys

    exc = sys.exc_info()[1]
    assert exc is not None
    return exc


def _too_long(doc_id: str) -> "ShieldError":
    return ShieldError(f"{doc_id} n'a pas été lu : sa lecture prend trop de temps (document_too_long).",
                       "document_too_long")


def _too_large(doc_id: str) -> "ShieldError":
    """v0.4.3 (fix): a document whose text passes MAX_DOCUMENT_CHARS (the 0.4.2 answer, kind ``document_too_large``)."""
    return ShieldError(f"{doc_id} est trop volumineux pour être lu (plus de {MAX_DOCUMENT_CHARS:,} caractères) "
                       "(document_too_large).".replace(",", " "), "document_too_large")


def _write_blocked(code: str) -> "ShieldError":
    """v0.4.4 (F1): the system refused to write the final document (Windows ransomware protection, macOS privacy)."""
    return ShieldError(f"Le système a empêché Shield d'enregistrer le fichier dans le dossier de sortie de {code} "
                       "(write_blocked).", "write_blocked")


def _path_too_long(code: str) -> "ShieldError":
    """v0.4.4 (F3): the folder of the dossier is too long for Windows to hold the final document."""
    return ShieldError(f"Le chemin du dossier client de {code} est trop long pour que Windows y enregistre le "
                       "fichier (output_path_too_long).", "output_path_too_long")


def _locked(doc_id: str) -> "ShieldError":
    """v0.4.4 (F4): a document that another program held open and locked when the dossier was read."""
    return ShieldError(f"{doc_id} est ouvert dans un autre programme qui le bloque (document_locked).",
                       "document_locked")


def _unavailable(doc_id: str) -> "ShieldError":
    """v0.4.4 (review fix): the system could not give the file at that moment (disk, network, a file in the cloud
    only while offline): it can pass, so the document is tried again at the next call."""
    return ShieldError(f"{doc_id} n'a pas pu être ouvert pour le moment (document_unavailable).",
                       "document_unavailable")


def _mask_budget(chars: int, read_budget: float = 0.0) -> float:
    """v0.4.4 (B2): time budget of the masking of one document (seconds)."""
    return min(documents.DOCUMENT_TIME_BUDGET_MAX,
               max(read_budget, documents.DOCUMENT_TIME_BUDGET + chars / MASK_CHARS_PER_SECOND))


def _dossier_code(value: Any) -> str:
    s = str(value if value is not None else "").strip().upper()
    if not DOSSIER_RE.match(s):
        raise ShieldError("Code de dossier invalide : attendu DOSSIER_n (invalid dossier code).",
                          "invalid_argument")
    return s


def _doc_code(value: Any) -> str:
    s = str(value if value is not None else "").strip().upper()
    if not DOC_RE.match(s):
        raise ShieldError("Code de document invalide : attendu DOC_n (invalid document code).",
                          "invalid_argument")
    return s


def _kind_arg(value: Any) -> str:
    s = str(value or "").strip().upper()
    if s not in KINDS:
        raise ShieldError("Type inconnu (unknown kind). Types : " + ", ".join(KINDS), "invalid_argument")
    return s


def _replacement_arg(value: Any) -> Optional[str]:
    """"codes" | "synthetic" | None (default), or ShieldError for anything else (v0.2)."""
    if value is None or value == "":
        return None
    v = str(value).strip().lower()
    if v not in ("codes", "synthetic"):
        raise ShieldError("Mode de remplacement inconnu : codes ou synthetic (unknown replacement mode).",
                          "invalid_argument")
    return v


def _nfc(text: str) -> str:
    """NFC form: decomposed accents (macOS file names, some PDF text) would cut names in two."""
    return unicodedata.normalize("NFC", text) if text and not text.isascii() else text


def _escape_outside(text: str, spans: list[Span]) -> str:
    """The original text with code-like strings escaped the way the masker does (same length)."""
    from .engine.masker import code_like_spans

    starts = [s.start for s in spans]
    out = list(text)
    for a, b, _code in code_like_spans(text):
        i = bisect.bisect_right(starts, b - 1) - 1
        if i >= 0 and spans[i].end > a:
            continue  # inside a replaced span
        for k in range(a, b):
            if out[k] == "_":
                out[k] = "-"
                break
    return "".join(out)


@dataclass
class _Alignment:
    """Offsets of the replaced spans in the original text and in the masked text."""

    starts: list[int] = field(default_factory=list)
    ends: list[int] = field(default_factory=list)
    mstarts: list[int] = field(default_factory=list)
    mends: list[int] = field(default_factory=list)
    codes: list[str] = field(default_factory=list)

    def to_masked(self, pos: int) -> int:
        i = bisect.bisect_right(self.starts, pos) - 1
        if i < 0:
            return pos
        if pos < self.ends[i]:
            return self.mstarts[i]
        return pos - self.ends[i] + self.mends[i]

    def overlaps(self, s: int, e: int) -> bool:
        i = bisect.bisect_right(self.starts, e - 1) - 1
        return i >= 0 and self.ends[i] > s


@dataclass
class _Chain:
    """Alignments applied one after the other: the masker spans, then each pass of the last canary (R1 CL-25).

    Maps an offset of the original text to the text the model sees, and tells whether a slice of the original
    text was replaced by any step."""

    steps: list[_Alignment] = field(default_factory=list)

    def to_masked(self, pos: int) -> int:
        for st in self.steps:
            pos = st.to_masked(pos)
        return pos

    def overlaps(self, s: int, e: int) -> bool:
        for st in self.steps:
            if st.overlaps(s, e):
                return True
            s, e = st.to_masked(s), st.to_masked(e)
        return False


def _edits_alignment(edits: list[tuple[int, int, str]]) -> _Alignment:
    """Alignment of one canary pass from its edits (start, end, replacement), sorted and without overlaps."""
    al = _Alignment()
    delta = 0
    for s, e, rep in edits:
        al.starts.append(s)
        al.ends.append(e)
        al.mstarts.append(s + delta)
        al.mends.append(s + delta + len(rep))
        al.codes.append(rep)
        delta += len(rep) - (e - s)
    return al


def _glued_code(text: str, s: int, e: int, code: str) -> str:
    """The code that replaces ``text[s:e]``, with a space where a glued letter or digit would hide it from
    restore ("2026Brakel" -> "2026 PERSONNE_C")."""
    before = " " if s > 0 and text[s - 1].isalnum() else ""
    nxt = text[e] if e < len(text) else ""
    letters = code.rsplit("_", 1)[-1].isalpha()
    after = " " if nxt and ((letters and nxt.isupper()) or (not letters and nxt.isdigit())) else ""
    return before + code + after


def _align(original: str, masked: str, spans: list[Span], codes: set[str]) -> Optional[_Alignment]:
    """Find where each replaced span sits in the masked text. None if the texts do not line up."""
    spans = sorted(spans, key=lambda s: s.start)
    escaped = _escape_outside(original, spans)
    al = _Alignment()
    delta = 0
    for i, sp in enumerate(spans):
        ms = sp.start + delta
        nxt = spans[i + 1].start if i + 1 < len(spans) else len(original)
        seg = escaped[sp.end:nxt]
        # The masker adds a space next to a code glued to a letter or a digit ("2026 PERSONNE_A").
        lead = 1 if ms < len(masked) and masked[ms] == " " and sp.start > 0 and original[sp.start - 1].isalnum() \
            else 0
        ms += lead
        if ms < len(masked) and masked[ms] == "":
            m = _SENTINEL_RE.match(masked, ms)
            length = len(m.group()) if m and masked.startswith(seg, m.end()) else None
        else:
            m = _CODE_SHAPE.match(masked, ms)
            length = None
            if m:
                cand = m.group()
                for n in range(len(cand), 2, -1):
                    if cand[:n] in codes and masked.startswith(seg, ms + n):
                        if seg or i + 1 == len(spans) or _CODE_SHAPE.match(masked, ms + n) \
                                or masked.startswith("", ms + n):
                            length = n
                            break
        trail = 0
        if length is None and m is not None and masked[ms:ms + 1] != "":
            cand = _CODE_SHAPE.match(masked, ms)
            for n in range(len(cand.group()) if cand else 0, 2, -1):
                if cand.group()[:n] in codes and masked[ms + n:ms + n + 1] == " " and masked.startswith(seg, ms + n + 1):
                    length, trail = n, 1
                    break
        if length is None:
            return None
        al.starts.append(sp.start)
        al.ends.append(sp.end)
        al.mstarts.append(ms)
        al.mends.append(ms + length)
        al.codes.append(masked[ms:ms + length])
        delta += lead + length + trail - (sp.end - sp.start)
    if len(original) + delta != len(masked):
        return None
    return al


def _cut(text: str, start: int, max_chars: int) -> int:
    """End offset of a page: at most ``max_chars`` after ``start``, at a line or word break, never inside a code."""
    end = min(len(text), start + max_chars)
    if end >= len(text):
        return len(text)
    floor = start + int(max_chars * 0.8)
    nl = text.rfind("\n", floor, end)
    if nl > start:
        end = nl + 1
    else:
        sp = text.rfind(" ", floor, end)
        if sp > start:
            end = sp + 1
    for m in UNMASK_RE.finditer(text, max(start, end - 20), min(len(text), end + 20)):
        if m.start() < end < m.end():
            end = m.start() if m.start() > start else m.end()
    return end


def _snap_left(text: str, pos: int, limit: int) -> int:
    """Move ``pos`` left to a word start (at most ``limit`` chars), never inside a code."""
    lo = max(0, pos - limit)
    sp = max(text.rfind(" ", lo, pos), text.rfind("\n", lo, pos))
    pos = sp + 1 if sp >= 0 else pos
    for m in UNMASK_RE.finditer(text, max(0, pos - 20), min(len(text), pos + 20)):
        if m.start() < pos < m.end():
            pos = m.start()
    return pos


def _snap_right(text: str, pos: int, limit: int) -> int:
    hi = min(len(text), pos + limit)
    cands = [i for i in (text.find(" ", pos, hi), text.find("\n", pos, hi)) if i >= 0]
    pos = min(cands) if cands else pos
    for m in UNMASK_RE.finditer(text, max(0, pos - 20), min(len(text), pos + 20)):
        if m.start() < pos < m.end():
            pos = m.end()
    return pos


# ------------------------------------------------------------------------------------------ kept sets (v0.4.4, A)

_ASCII_WORD = frozenset("abcdefghijklmnopqrstuvwxyz0123456789")


def _shadow_char(ch: str) -> str:
    """The character that stands for a replaced character in the search text of a kept entry. The search finds a
    word on the folded text with the rule "no letter or digit [a-z0-9] just before or after": a replaced character
    becomes "x" when its folded form is only such letters or digits, "*" when it has none of them and no space, and
    stays as it is when it is white space, a dropped mark, an accent mark, or a mixed symbol. So the folded search
    text has the same word limits as the original one everywhere."""
    from .engine.normalize import _DROP, _SPACES, _fold_char

    if ch in _SPACES or ch in _DROP:
        return ch
    f = _fold_char(ch)
    if not f:
        return ch  # an accent mark written apart: it belongs to the character before
    if all(c in _ASCII_WORD for c in f):
        return "x"
    if not any(c in _ASCII_WORD or c == " " for c in f):
        return "*"
    return ch


def _replaced_chars(raw_len: int, chain: "_Chain") -> list[tuple[int, int]]:
    """The intervals of the original text that some step of ``chain`` replaced: exactly the characters ``p`` for
    which ``chain.overlaps(p, p + 1)`` is true. (``overlaps(s, e)`` is true when one character of [s, e) is: the
    first step maps a range with no replaced character by one shift, so the union rule holds step after step.) The
    spans of the first step are taken whole; the few edits of the last check are mapped back and checked one
    character at a time."""
    steps = chain.steps
    out: list[tuple[int, int]] = list(zip(steps[0].starts, steps[0].ends)) if steps else []
    for k in range(1, len(steps)):
        for a, b in zip(steps[k].starts, steps[k].ends):
            for st in reversed(steps[:k]):
                a, b = _from_masked(st, a), _from_masked(st, b)
            lo, hi = max(0, min(a, b) - 4), min(raw_len, max(a, b) + 4)
            out.extend((p, p + 1) for p in range(lo, hi) if chain.overlaps(p, p + 1))
    out.sort()
    merged: list[tuple[int, int]] = []
    for a, b in out:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        elif a < b:
            merged.append((a, b))
    return merged


def _from_masked(al: "_Alignment", pos: int) -> int:
    """Approximate original offset of a position of the masked text of one step (outside its spans: exact)."""
    i = bisect.bisect_right(al.mstarts, pos) - 1
    if i < 0:
        return pos
    if pos < al.mends[i]:
        return al.starts[i]
    return pos - al.mends[i] + al.ends[i]


def _search_shadow(raw: str, chain: Optional["_Chain"]) -> Optional[str]:
    """v0.4.4 (A): the search text of a kept entry: ``raw`` with every replaced character changed by
    :func:`_shadow_char`. Same length as ``raw``; equal to it outside the replaced spans (where the masked text holds
    the same words). None when the search does not use the original text (no alignment)."""
    if chain is None or not raw:
        return None
    parts: list[str] = []
    pos = 0
    for a, b in _replaced_chars(len(raw), chain):
        parts.append(raw[pos:a])
        parts.append("".join(_shadow_char(c) for c in raw[a:b]))
        pos = b
    parts.append(raw[pos:])
    shadow = "".join(parts)
    if len(shadow) != len(raw):  # never: one character for one character
        return None
    return shadow


def _al_dump(al: Optional["_Alignment"]) -> Optional[list]:
    return None if al is None else [al.starts, al.ends, al.mstarts, al.mends, al.codes]


def _al_load(data: Optional[list]) -> Optional["_Alignment"]:
    if data is None:
        return None
    starts, ends, mstarts, mends, codes = data
    return _Alignment(list(starts), list(ends), list(mstarts), list(mends), list(codes))


def _steps_sig(chain: Any) -> Optional[tuple]:
    """The edits of the last check in an alignment chain (to tell whether a kept search text still fits)."""
    if chain is None:
        return None
    return tuple((tuple(st.starts), tuple(st.ends), tuple(st.codes)) for st in chain.steps[1:])


class _DryStore:
    """EntityStore wrapper that never writes: new entities get a private sentinel code.

    Used to find which registered entities a text contains without polluting the dossier.
    ``protect`` codes are reported as known, so masking with ``keep_known_codes=True`` keeps them.
    """

    def __init__(self, real: Any, protect: Iterable[str] = ()):
        self.real = real
        self.locale = getattr(real, "locale", "fr")
        self._protect = set(protect)
        self._by_norm: dict[tuple[str, str], Entity] = {}
        self._by_code: dict[str, Entity] = {}

    def all(self) -> list[Entity]:
        return self.real.all()

    def get_by_norm(self, kind: str, norm: str) -> Optional[Entity]:
        return self._by_norm.get((kind, norm)) or self.real.get_by_norm(kind, norm)

    def get_by_code(self, code: str) -> Optional[Entity]:
        ent = self.real.get_by_code(code) or self._by_code.get(code)
        if ent is None and code in self._protect:
            return Entity(kind="PERSON", value=code, norm=code.lower(), code=code)
        return ent

    def add(self, kind: str, value: str, norm: str) -> Entity:
        key = (kind, norm)
        if key in self._by_norm:
            return self._by_norm[key]
        ent = Entity(kind=kind, value=value, norm=norm, code=f"{len(self._by_code) + 1}")
        self._by_norm[key] = ent
        self._by_code[ent.code] = ent
        return ent

    def is_kept_clear(self, norm: str) -> bool:
        return self.real.is_kept_clear(norm)


@dataclass
class _CachedDoc:
    mtime_ns: int
    size: int
    version: tuple[int, int]
    raw: Optional[str]     # the text read from the file; None for a kept entry (v0.4.4: never kept)
    masked: str            # masked text after the last canary of every dossier: what the model reads
    alignment: Any         # _Chain from ``raw`` to ``masked`` (None when the texts do not line up)
    needs_ocr: bool
    pages: Optional[int]
    warnings: tuple[str, ...]
    masked0: str = ""      # masker output before the canary
    align0: Optional[_Alignment] = None
    epoch: tuple = ()      # dossier revisions the canary saw (a change of any dossier guards again)
    #: v0.4.4 (A): kept form: the search text, same length as ``raw``, with every replaced character changed
    #: (see :func:`_search_shadow`). Set when ``raw`` is None.
    shadow: Optional[str] = None
    _folded_raw: Any = None
    _folded_masked: Any = None

    @property
    def search_text(self) -> str:
        """The text the search runs on: the text read from the file, or its kept shadow (same results)."""
        return self.raw if self.raw is not None else (self.shadow or "")

    @property
    def folded_raw(self):
        if self._folded_raw is None:
            self._folded_raw = fold(self.search_text)
        return self._folded_raw

    @property
    def folded_masked(self):
        if self._folded_masked is None:
            self._folded_masked = fold(self.masked)
        return self._folded_masked


class _Stopped(BaseException):
    """Internal: the service closes; a background reading stops between two pages (its mark stays on disk)."""


class _KeptStale(Exception):
    """Internal (v0.4.4): a kept entry whose last check now gives other edits; the file is read again."""


class _Reading:
    """v0.4.3: the background reading of one dossier (codes and counts only).

    ``state``: "waiting" (another dossier is read first), "reading" (text of the documents), "masking" (names
    replaced by codes), "ready", "failed" or "gone" (the dossier was removed)."""

    def __init__(self, code: str, uuid: str):
        self.code = code
        self.uuid = uuid
        self.state = "waiting"
        self.total: Optional[int] = None  # documents to read
        self.done = 0                     # documents read
        self.masked = 0                   # documents whose names are replaced
        self.page = 0                     # page of the document being read
        self.pages = 0                    # pages of the document being read (0: not a PDF, or not started)
        self.summary: Optional[dict] = None
        self.error: Optional[str] = None
        self.started = time.monotonic()
        self.step_started = self.started  # start of the current document or step (v0.4.3 fix: stuck readings)
        self.marked = False               # this reading owns the mark on disk
        self.kept = False                 # v0.4.4: it ended on the kept set (the cheaper re-check), nothing read
        self.finished = threading.Event()
        self.thread: Optional[threading.Thread] = None
        #: v0.4.4 (B2): the time budget of the current document or step, and the reading budget of each document
        self.budget = documents.DOCUMENT_TIME_BUDGET
        self.budgets: dict[str, float] = {}

    def stuck(self, now: Optional[float] = None) -> bool:
        """True when the current document (or step) went past its time budget plus a margin: something inside one
        page or one reader does not end. Only a running reading can be stuck. v0.4.4 (B2): the budget is the one of
        the document being read or masked (a long PDF has more time)."""
        if self.finished.is_set() or self.state not in ("reading", "masking"):
            return False
        now = time.monotonic() if now is None else now
        return now - self.step_started > self.budget + STUCK_MARGIN_SECONDS

    def step(self, budget: Optional[float] = None) -> None:
        self.budget = documents.DOCUMENT_TIME_BUDGET if budget is None else float(budget)
        self.step_started = time.monotonic()

    def view(self) -> dict:
        return {"code": self.code, "state": self.state, "total": self.total, "done": self.done,
                "masked": self.masked, "page": self.page, "pages": self.pages}


def _pid_alive(pid: Optional[int]) -> bool:
    """v0.4.3 (fix): True when the process ``pid`` runs (or when it cannot be told: the age of the mark decides).
    Never sends a signal on Windows (``os.kill`` there would stop the process)."""
    if not pid or pid <= 0 or pid == os.getpid():
        return True
    if sys.platform == "win32":  # pragma: no cover - Windows
        try:
            import ctypes
            from ctypes import wintypes

            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.OpenProcess.restype = wintypes.HANDLE
            k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            k32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
            k32.CloseHandle.argtypes = (wintypes.HANDLE,)
            handle = k32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
            if not handle:
                return ctypes.get_last_error() != 87  # ERROR_INVALID_PARAMETER: no such process
            try:
                code = wintypes.DWORD()
                if not k32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return True
                return code.value == 259  # STILL_ACTIVE
            finally:
                k32.CloseHandle(handle)
        except Exception:  # noqa: BLE001
            return True
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except Exception:  # noqa: BLE001 - PermissionError: it runs under another account
        return True
    return True


# ------------------------------------------------------------------------------------------ service

class Service:
    """Shield operations over one data directory (``config.home()`` by default)."""

    def __init__(self, home: Optional[os.PathLike | str] = None):
        self._home = Path(home) if home is not None else None
        self._lock = threading.RLock()
        self._registry: Optional[Registry] = None
        self._maskers: dict[str, tuple[str, Masker]] = {}
        self._rule_maskers: dict[str, tuple[str, Masker]] = {}
        self._indexes: dict[str, tuple[Any, dict]] = {}
        self._label_indexes: dict[str, tuple[Any, dict]] = {}
        self._docs: OrderedDict[tuple[str, str], _CachedDoc] = OrderedDict()
        self._pages: dict[tuple[str, int, int], Optional[int]] = {}
        # v0.4.3: background readings (plugin). Their own small lock: a status call never waits for a masking.
        self._reads: dict[str, _Reading] = {}
        self._reads_lock = threading.RLock()
        # one reading at a time (one processor core, flat memory); a stuck reading no longer holds the others back
        self._gate = threading.Condition(threading.Lock())
        self._gate_owner: Optional[_Reading] = None
        self._stop = threading.Event()
        #: (dossier, DOC_n) -> (mtime_ns, size, reason): documents not read in their time budget ("too_long")
        self._not_read: dict[tuple[str, str], tuple[int, int, str]] = {}
        self._failures: Counter = Counter()  # failed readings per dossier (this process)
        self._warmed: set[str] = set()  # dossiers whose long PDF files were read once in this process
        #: v0.4.3 (fix): the owner name written in the marks of this service (one per service object)
        self._token = uuidlib.uuid4().hex
        #: v0.4.3 (fix): a background reading masks outside the service lock with its own rules maskers (a masker
        #: keeps per-call state, so one masker object is never used by two threads at once)
        self._tls = threading.local()
        #: v0.4.4 (A): the kept sets. ``_kset_docs``: {dossier: {DOC_n: kept entry}} (not limited like ``_docs``);
        #: ``_kset_labels``: {dossier: (revision, epoch, {DOC_n: masked labels})}; ``_kset_meta``: {dossier:
        #: {DOC_n: pages}}. One kept-set load or write at a time in this process.
        self._kset_docs: dict[str, dict[str, _CachedDoc]] = {}
        self._kset_labels: dict[str, tuple] = {}
        self._kset_lock = threading.RLock()
        self._kset_store_obj: Optional[_kset.KeptStore] = None

    # ---------------------------------------------------------------- plumbing
    @property
    def registry(self) -> Registry:
        reg = self._registry  # v0.4.3: no lock once open, so a status call never waits for a background masking
        if reg is not None:
            return reg
        with self._lock:
            if self._registry is None:
                self._registry = Registry(self._home)
            return self._registry

    def close(self) -> None:
        # v0.4.3: background readings stop between two pages or two documents; their marks stay on disk, so they
        # start again at the next start (a half-read dossier is never readable).
        self._stop.set()
        with self._reads_lock:
            jobs = list(self._reads.values())
        for job in jobs:
            if job.thread is not None and job.thread is not threading.current_thread():
                job.thread.join(timeout=30.0)
        for job in jobs:  # v0.4.3 (fix): our marks stay (not read yet) but are free at once for the next start
            if job.marked and job.state != "ready":
                self._release_mark(job.code)
        with self._lock:
            if self._registry is not None:
                self._registry.close()
            self._registry = None
            self._maskers.clear()
            self._rule_maskers.clear()
            self._indexes.clear()
            self._label_indexes.clear()
            self._docs.clear()
            self._not_read.clear()
            self._kset_docs.clear()
            self._kset_labels.clear()
            self._kset_store_obj = None
        with self._reads_lock:
            self._reads.clear()
            self._failures.clear()
            self._warmed.clear()
        self._stop.clear()

    def _dossier(self, code: Any) -> Dossier:
        return self.registry.get(_dossier_code(code))

    def _masker(self, d: Dossier, replacement: Optional[str] = None) -> Masker:
        """The dossier masker used for documents and pasted text (optional NER when installed).

        ``replacement`` (v0.2): "codes" or "synthetic"; by default the dossier setting, else
        ``config.replacement()``. One masker is cached per (dossier, replacement)."""
        replacement = _replacement_arg(replacement) or self._dossier_replacement(d)
        key = f"{d.code}:{replacement}"
        cached = self._maskers.get(key)
        if cached is not None and cached[0] == d.uuid:
            return cached[1]
        # v0.2: the dossier language setting (else RAGBASE_SHIELD_LANGUAGE) and the NER model setting;
        # v0.4 (P1): the own-company vendor option (RAGBASE_SHIELD_OWN_COMPANY_VENDOR, default off)
        m = Masker(d, locale=d.locale, replacement=replacement, language=self._dossier_language(d),
                   ner_model=config.ner_model(), own_company_vendor=config.own_company_vendor())
        self._maskers[key] = (d.uuid, m)
        return m

    def _drop_maskers(self, code: str) -> None:
        """Forget the cached maskers of a dossier (every replacement mode)."""
        for key in [k for k in self._maskers if k == code or k.startswith(code + ":")]:
            del self._maskers[key]

    @staticmethod
    def _dossier_language(d: Dossier) -> str:
        setting = getattr(d, "setting", None)
        try:
            value = setting("language") if callable(setting) else None
        except Exception:  # noqa: BLE001 - an old store: the global default
            value = None
        return value or config.language()

    @staticmethod
    def _dossier_replacement(d: Dossier) -> str:
        setting = getattr(d, "setting", None)
        try:
            value = setting("replacement") if callable(setting) else None
        except Exception:  # noqa: BLE001 - an old store: the global default
            value = None
        return value or config.replacement()

    def _rules(self, d: Dossier) -> Masker:
        """Rules-only masker of the dossier: canary, hooks, labels (never loads a NER model).

        v0.4.3 (fix): a background reading masks outside the service lock; its thread keeps its own rules maskers,
        so a masker object is never used by two threads at once (same rules, same results)."""
        cache = getattr(self._tls, "rules", None)
        if cache is None:
            cache = self._rule_maskers
        cached = cache.get(d.code)
        if cached is not None and cached[0] == d.uuid:
            return cached[1]
        m = Masker(d, locale=d.locale, use_ner=False, own_company_vendor=config.own_company_vendor())
        cache[d.code] = (d.uuid, m)
        return m

    def _open_dossiers(self) -> list[Dossier]:
        """Every registered dossier, for a model-facing check. Fail closed: if ONE vault cannot be opened, its
        values could no longer be checked, so the whole check fails (ShieldError ``vault_error``)."""
        out = []
        for code in self.registry.folders():
            try:
                out.append(self.registry.get(code))
            except VaultError:
                log.warning("RAGbase Shield: %s cannot be opened; the check fails closed", code)
                raise ShieldError(f"Le coffre de {code} ne s'ouvre pas : vérification impossible, rien n'est "
                                  "renvoyé. Lancez « ragbase-shield doctor » ou retirez ce dossier dans Shield Desk. "
                                  f"(vault of {code} unreadable: failing closed)", "vault_error") from None
        return out

    @staticmethod
    def _version(d: Dossier) -> Any:
        return d.revision

    def _invalidate(self, code: str) -> None:
        for key in [k for k in self._docs if k[0] == code]:
            del self._docs[key]
        self._kset_docs.pop(code, None)
        self._kset_labels.pop(code, None)

    def _cached(self, key: tuple[str, str]) -> Optional[_CachedDoc]:
        """v0.4.4: the cached entry of a document: the last one read in this process, else its kept entry."""
        hit = self._docs.get(key)
        if hit is None:
            hit = self._kset_docs.get(key[0], {}).get(key[1])
        return hit

    def _drop_cached(self, key: tuple[str, str], entry: _CachedDoc) -> None:
        if self._docs.get(key) is entry:
            del self._docs[key]
        kept = self._kset_docs.get(key[0])
        if kept is not None and kept.get(key[1]) is entry:
            del kept[key[1]]

    # ---------------------------------------------------------------- canary
    def _index(self, d: Dossier) -> dict:
        """Lookup tables of the dossier's entities by engine key and by surname (cached per dossier version)."""
        ents = d.all()
        rev = d.revision
        cached = self._indexes.get(d.code)
        if cached is not None and cached[0] == rev:
            return cached[1]
        from .engine.names import first_key, surname_key

        by_key: dict[tuple[str, str], set[str]] = {}
        by_surname: dict[str, set[str]] = {}
        by_first: dict[str, set[str]] = {}
        by_variant: dict[tuple[str, tuple[str, ...]], set[str]] = {}
        for e in ents:
            for key in {e.norm, _safe_norm(e.kind, e.value)}:
                if key:
                    by_key.setdefault((e.kind, key), set()).add(e.code)
            if e.kind == "PERSON":
                sk = surname_key(e.value)
                if sk:
                    by_surname.setdefault(sk, set()).add(e.code)
                for tok in first_key(e.value).split():
                    by_first.setdefault(tok, set()).add(e.code)
            if e.kind in ("PERSON", "ORG"):
                for toks in _variant_keys(e.kind, e.value):
                    by_variant.setdefault((e.kind, toks), set()).add(e.code)
        index = {"key": by_key, "surname": by_surname, "first": by_first, "variant": by_variant}
        self._indexes[d.code] = (rev, index)
        return index

    def _entity_for(self, d: Dossier, kind: str, surface: str) -> Optional[Entity]:
        """Best-effort entity behind a surface form found by the canary (same engine key, or a unique surname).

        A one-word surface that is the surname of one person and the first name of another one is ambiguous:
        None, never the wrong person (R1, with CL-20 and CL-26)."""
        index = self._index(d)
        codes = index["key"].get((kind, _safe_norm(kind, surface)), set())
        if len(codes) != 1 and kind == "PERSON":
            from .engine.names import surname_key

            sk = surname_key(surface)
            codes = set(index["surname"].get(sk, set())) if sk else set()
            if codes and sk and " " not in sk:
                codes |= index["first"].get(sk, set())
        if not codes and kind in ("PERSON", "ORG"):
            # a short form the engine knows (acronym, initials, nickname, head word): its owners, if only one
            from .engine.normalize import phrase_tokens

            codes = index["variant"].get((kind, phrase_tokens(surface, confusables=True)), set())
        if len(codes) == 1:
            return d.get_by_code(next(iter(codes)))
        return None

    def _kept(self, d: Dossier, span: Span) -> bool:
        for key in (normalize(span.value), _safe_norm(span.kind, span.value)):
            if key and d.is_kept_clear(key):
                return True
        ent = self._entity_for(d, span.kind, span.value)
        return bool(ent is not None and d.is_kept_clear(ent.norm))

    def _leftovers(self, d: Dossier, text: str, prep: Optional[ScanPrep] = None) -> list[Span]:
        """Registered values of ``d`` still present in ``text`` (lenient canary, kept-clear terms ignored).

        F0 (0.4.3): the canary reads only the values that may be spread (``engine.types.spreadable``, applied in
        the masker's lexicon, so a weak value such as "4", or since 0.4.4 a place of birth such as "Lyon", is never
        a dictionary hit), a structured hit counts only
        for a value that is not weak, and a hit that touches a Shield code, DOC_n, DOSSIER_n, OUTPUT_n or a
        placeholder ("[IDENT]") is dropped: Shield never masks its own words."""
        if not text or not d.all():
            return []
        tokens = shield_token_spans(text)
        spans = []
        for sp in self._rules(d).scan(text, prepared=prep):
            if _touches(tokens, sp.start, sp.end):
                continue
            # known values come back as "dictionary"; a registered identifier that is also a valid structured
            # id (e-mail, IBAN...) comes back as "regex": keep it only when it is one of the dossier's values
            if sp.source == "dictionary":
                spans.append(sp)
            elif sp.source == "regex":
                # 0.4.4: a structured hit is a detector find in its own context, not a spread value: a place of
                # birth comes back here only where the text says it is one ("né le ... à Lyon"), which is L1 of the
                # birth place fix. The dictionary hits never hold one (``spreadable`` in the masker's lexicon).
                ent = self._entity_for(d, sp.kind, sp.value)
                if ent is not None and not is_weak_value(ent.value):
                    spans.append(sp)
        return [sp for sp in spans if not self._kept(d, sp)]

    def _redact(self, text: str, hits: list[Span], d: Dossier, own: bool,
                mint_into: Optional[Dossier] = None) -> tuple[str, list[tuple[int, int, str]]]:
        """Replace canary hits of dossier ``d``. Returns (text, edits), edits = (start, end, replacement) in the
        input offsets, sorted.

        A hit of the dossier's own entity gets its code (``own``). Otherwise, with ``mint_into`` (the dossier
        the text belongs to), the value becomes a NEW entity of that dossier, so the code can be restored there
        (R1 correction CL-26); without it, a generic placeholder ("[PERSONNE]") that nothing restores."""
        edits: list[tuple[int, int, str]] = []
        last = len(text) + 1
        for s in sorted(hits, key=lambda s: s.start, reverse=True):
            if s.end > last:
                continue
            ent = self._entity_for(d, s.kind, s.value) if own else None
            if ent is None and mint_into is not None:
                # the same name may already be an entity of the text's own dossier: reuse its code
                ent = (None if own else self._entity_for(mint_into, s.kind, s.value)) \
                    or self._mint(mint_into, s.kind, text[s.start:s.end])
            rep = _glued_code(text, s.start, s.end, ent.code) if ent is not None \
                else f"[{prefix_for(s.kind, d.locale)}]"
            edits.append((s.start, s.end, rep))
            last = s.start
        edits.reverse()
        parts: list[str] = []
        pos = 0
        for s, e, rep in edits:
            parts.append(text[pos:s])
            parts.append(rep)
            pos = e
        parts.append(text[pos:])
        return "".join(parts), edits

    def _mint(self, d: Dossier, kind: str, surface: str) -> Optional[Entity]:
        """The entity of ``d`` for a value the canary found (created when new). None if it cannot be made, or
        when the user keeps this term in clear in ``d``: the caller then writes a placeholder."""
        value = " ".join(surface.split())
        norm = _safe_norm(kind, value)
        if not value or not norm or is_weak_value(value) or d.is_kept_clear(norm) or d.is_kept_clear(normalize(value)):
            return None  # F0 (0.4.3): a weak value is never made into an entity of another dossier
        try:
            return d.add(kind, value, norm)
        except Exception:  # noqa: BLE001 - code range exhausted, refused value: keep the placeholder
            log.warning("RAGbase Shield: %s: a canary value could not get a code; placeholder kept", d.code)
            return None

    def _foreign_hit(self, d: Dossier, text: str, sp: Span) -> bool:
        """A canary hit of dossier ``d`` in a text that belongs to ANOTHER dossier: True when it must still be
        replaced (R1 correction CL-26).

        A whole registered value, or a name of several words, always is. A single word that is only PART of a
        registered name of ``d`` is replaced only when it is a strong name word (in no word list) written with a
        capital, or a social handle: a common word, a first name or a lower-case word of the other dossier stays
        as it is in this dossier's text ("la dette financière nette")."""
        if sp.kind not in ("PERSON", "ORG"):
            return True
        word = sp.value.strip()
        if not word or any(c.isspace() for c in word):
            return True
        if self._index(d)["key"].get((sp.kind, _safe_norm(sp.kind, word))):
            return True  # the whole registered value
        if sp.start > 0 and text[sp.start - 1] == "@":
            return True
        if not word[0].isupper():
            return False
        from .engine.normalize import fold_word

        return _strong_name_word(fold_word(word))

    def _guard(self, text: str, primary: Optional[Dossier] = None) -> str:
        """Last canary on a model-facing string: no registered value of any dossier may remain."""
        return self._guard_report(text, primary)[0]

    def _guard_report(self, text: str, primary: Optional[Dossier] = None) -> tuple[str, Counter]:
        """``_guard`` that also counts the values it replaced, per kind."""
        out, _steps, replaced = self._guard_full(text, primary)
        return out, replaced

    def _guard_full(self, text: str, primary: Optional[Dossier] = None
                    ) -> tuple[str, Optional[list[_Alignment]], Counter]:
        """The last canary, with one alignment per dossier pass (None when the NFC form changed the length).

        ``primary`` is the dossier the text belongs to: its own values get their codes, values of other dossiers
        become new entities of ``primary`` (restorable), and single common words of other dossiers stay."""
        replaced: Counter = Counter()
        steps: Optional[list[_Alignment]] = []
        if not text:
            return text, steps, replaced
        nfc = _nfc(text)
        if nfc != text:
            text, steps = nfc, None
        prep: Optional[ScanPrep] = None
        for d in self._open_dossiers():
            if not d.all():
                continue
            if prep is None:
                prep = prepare_scan(text)
            hits = self._leftovers(d, text, prep)
            own = primary is not None and d.code == primary.code
            if hits and primary is not None and not own:
                hits = [h for h in hits if self._foreign_hit(d, text, h)]
            if hits:
                log.warning("RAGbase Shield: %d leftover value(s) of %s replaced in an output", len(hits), d.code)
                text, edits = self._redact(text, hits, d, own=own, mint_into=primary)
                replaced.update(h.kind for h in hits)
                if steps is not None:
                    steps.append(_edits_alignment(edits))
                prep = None  # the text changed: prepare it again for the next dossier
        return text, steps, replaced

    def _has_registered(self, text: str, primary: Optional[Dossier] = None) -> bool:
        """True when ``text`` holds a registered value of any dossier (cheap check, no masking). With
        ``primary``, single common words of other dossiers do not count (same rule as :meth:`_guard`)."""
        if not text:
            return False
        if primary is None:
            primary = self._context_dossier()
        text = _nfc(text)
        prep: Optional[ScanPrep] = None
        for d in self._open_dossiers():
            if not d.all():
                continue
            if prep is None:
                prep = prepare_scan(text)
            hits = self._leftovers(d, text, prep)
            if hits and primary is not None and d.code != primary.code:
                hits = [h for h in hits if self._foreign_hit(d, text, h)]
            if hits:
                return True
        return False

    def _context_dossier(self) -> Optional[Dossier]:
        """The dossier set by :func:`output_dossier` around the current call, if any (and still registered)."""
        code = _OUTPUT_DOSSIER.get()
        if not code or not DOSSIER_RE.match(code):
            return None
        try:
            return self.registry.get(code)
        except UnknownDossier:
            return None

    def _guard_epoch(self) -> tuple:
        """Revisions of every dossier: a cached document is guarded again when one of them changed."""
        return tuple((d.code, d.revision) for d in self._open_dossiers())

    def _guard_many(self, items: list[str], primary: Optional[Dossier] = None) -> list[str]:
        """``_guard`` on many short strings at once (one canary pass per dossier)."""
        if not items:
            return items
        sep = "\n|\n"
        if not any(sep in x for x in items):
            joined = self._guard(sep.join(items), primary)
            parts = joined.split(sep)
            if len(parts) == len(items):
                return parts
        return [self._guard(x, primary) for x in items]

    def _registered_in(self, d: Dossier, text: str, prep: Optional[ScanPrep] = None) -> list[dict]:
        """Registered entities of ``d`` present in ``text``: [{code, kind, dossier}] (code None if unsure)."""
        lenient = self._leftovers(d, text, prep)
        if not lenient:
            return []
        dry = _DryStore(d)
        res = Masker(dry, locale=d.locale, use_ner=False).mask(text)
        real_codes = {e.code for e in d.all()}
        before = Counter(m.group() for m in UNMASK_RE.finditer(text))
        after = Counter(m.group() for m in UNMASK_RE.finditer(res.text))
        out: list[dict] = []
        seen: set[str] = set()
        for m in UNMASK_RE.finditer(res.text):
            code = m.group()
            if code in real_codes and after[code] > before.get(code, 0) and code not in seen:
                ent = d.get_by_code(code)
                if ent is not None and not d.is_kept_clear(ent.norm):
                    seen.add(code)
                    out.append({"code": code, "kind": ent.kind, "dossier": d.code})
        covered = [(s.start, s.end) for s in res.spans]
        for s in lenient:
            if any(a < s.end and s.start < b for a, b in covered):
                continue
            ent = self._entity_for(d, s.kind, s.value)
            code = ent.code if ent is not None else None
            if code is not None and code in seen:
                continue
            if code is not None:
                seen.add(code)
            out.append({"code": code, "kind": s.kind, "dossier": d.code})
        return out

    # ---------------------------------------------------------------- documents
    def _pdf_pages(self, path: Path, mtime_ns: int, size: int) -> Optional[int]:
        key = (str(path), mtime_ns, size)
        if key not in self._pages:
            pages = None
            try:
                from pypdf import PdfReader

                reader = PdfReader(str(path))
                if reader.is_encrypted:
                    reader.decrypt("")
                pages = len(reader.pages)
            except Exception:  # noqa: BLE001 - a page count is optional
                pages = None
            self._pages[key] = pages
        return self._pages[key]

    def _doc(self, d: Dossier, ref: Any, fresh: bool, epoch: Optional[tuple] = None,
             raw: Optional[tuple] = None) -> _CachedDoc:
        """Masked document from the cache, or read + masked now. ``fresh`` re-masks when the dossier changed.

        The cached text is the masked text after the last canary of every dossier (R1 correction CL-25): the
        offsets of ``read_document`` and the alignment used by ``search`` are those of the text the model reads,
        so a raw word the canary hid can never be confirmed by a search hit.

        v0.4.3: ``raw`` is ``(mtime_ns, size, ReadResult)`` read before (background reading); a re-mask of an
        unchanged file uses the text read before instead of reading the file again (same text, same output)."""
        key = (d.code, ref.id)
        st = ref.path.stat()
        hit, raw_text = self._doc_lookup(d, key, st, fresh, epoch, raw)
        if hit is not None:
            return hit
        if raw_text is None:
            raw_text = self._read_file(key, ref.path, st)
        entry = self._doc_make(d, ref, st, raw_text, self._masker(d))
        return self._doc_store(key, entry)

    def _doc_lookup(self, d: Dossier, key: tuple[str, str], st: os.stat_result, fresh: bool,
                    epoch: Optional[tuple] = None, raw: Optional[tuple] = None, guard: bool = True
                    ) -> tuple[Optional[_CachedDoc], Optional[documents.ReadResult]]:
        """(cached entry, None) when the cache answers; else (None, text read before or None). ``guard=False``
        (background reading): a current entry that only needs the canary again is returned with the canary NOT run
        (:meth:`_doc_reguard` runs it outside the service lock)."""
        cached = self._cached(key)
        if cached is not None and cached.mtime_ns == st.st_mtime_ns and cached.size == st.st_size:
            if not fresh or cached.version == self._version(d):
                if guard:
                    try:
                        self._doc_reguard(d, cached, epoch)
                    except _KeptStale:  # v0.4.4: a kept entry the last check now changes: the file is read again
                        self._drop_cached(key, cached)
                        cached = None
                if cached is not None:
                    if key in self._docs:
                        self._docs.move_to_end(key)
                    return cached, None
            elif cached.raw is not None:
                return None, documents.ReadResult(cached.raw, cached.pages, cached.needs_ocr, cached.warnings)
        if raw is not None and raw[0] == st.st_mtime_ns and raw[1] == st.st_size:
            return None, raw[2]
        return None, None

    def _doc_reguard(self, d: Dossier, cached: _CachedDoc, epoch: Optional[tuple] = None) -> None:
        now = epoch if epoch is not None else self._safe_epoch()
        if now is not None and cached.epoch != now:
            current = cached.version == self._version(d)
            before = (cached.masked, _steps_sig(cached.alignment)) if cached.raw is None else None
            self._guard_doc(d, cached)
            if before is not None and (cached.masked, _steps_sig(cached.alignment)) != before:
                # v0.4.4: the search text of a kept entry was made for the edits of the last check; other edits
                # now: never answer from it (the caller reads the file again)
                raise _KeptStale()
            if current:  # only the canary's own new codes changed the dossier
                cached.version = self._version(d)

    def _doc_store(self, key: tuple[str, str], entry: _CachedDoc) -> _CachedDoc:
        self._docs[key] = entry
        self._docs.move_to_end(key)
        while len(self._docs) > _CACHE_DOCS:
            self._docs.popitem(last=False)
        return entry

    def _doc_make(self, d: Dossier, ref: Any, st: os.stat_result, raw: documents.ReadResult, m: Masker) -> _CachedDoc:
        """Mask a document text and run the last canary on it (no cache write). v0.4.3: a background reading calls it
        outside the service lock."""
        key = (d.code, ref.id)
        if len(raw.text) > MAX_DOCUMENT_CHARS:
            self._not_read[key] = (st.st_mtime_ns, st.st_size, "too_large")  # v0.4.3: never read again for nothing
            raise _too_large(ref.id)
        if raw.needs_ocr or not raw.text:
            masked, alignment = "", _Alignment()
        else:
            res = m.mask(raw.text)
            masked = res.text
            if res.mode == "synthetic":  # v0.2: the edits are known exactly, no code to find in the text
                alignment = _edits_alignment(sorted(zip((sp.start for sp in res.spans), (sp.end for sp in res.spans),
                                                        res.replacements)))
            else:
                alignment = _align(raw.text, masked, res.spans, {e.code for e in d.all()})
            if alignment is None:
                log.info("RAGbase Shield: %s %s: offsets not aligned; search uses the masked text", d.code, ref.id)
        entry = _CachedDoc(mtime_ns=st.st_mtime_ns, size=st.st_size, version=self._version(d), raw=raw.text,
                           masked=masked, alignment=None, needs_ocr=bool(raw.needs_ocr),
                           pages=raw.pages, warnings=tuple(raw.warnings), masked0=masked, align0=alignment)
        self._guard_doc(d, entry)
        entry.version = self._version(d)  # the canary may have given new codes to this dossier
        return entry

    def _read_file(self, key: tuple[str, str], path: Path, st: os.stat_result,
                   progress: documents.Progress = None) -> documents.ReadResult:
        """Read a document with its time budget (v0.4.3). A document that was not read in its budget is not read
        again while the file is the same: ShieldError ``document_too_long`` at once."""
        bad = self._not_read.get(key)
        if bad is not None and bad[2] == "io":
            bad = None  # v0.4.4 (review fix): a failure of the disk, the network or a cloud file can pass: try again
        if bad is not None and bad[0] == st.st_mtime_ns and bad[1] == st.st_size:
            # v0.4.4 (F4): a document the last reading could not read is not read alone now (a name found in it must
            # also be hidden in the others): the lawyer protects the folder again
            if bad[2] == "too_large":
                raise _too_large(key[1])
            if bad[2] == "locked":
                raise _locked(key[1])
            if bad[2] in ("denied", "unreadable"):
                raise documents.DocumentError(bad[3] if len(bad) > 3 else "unreadable")
            raise _too_long(key[1])
        try:
            got = documents.read_text(path, progress=progress, max_chars=MAX_DOCUMENT_CHARS)
            if (self._not_read.get(key) or (0, 0, ""))[2] == "io":
                self._not_read.pop(key, None)  # it opens now: no longer marked "(non lu)"
            return got
        except documents.ReadTooLong:
            self._not_read[key] = (st.st_mtime_ns, st.st_size, "too_long")
            raise _too_long(key[1]) from None
        except documents.ReadTooLarge:  # v0.4.3 (fix): over MAX_DOCUMENT_CHARS, stopped as soon as it is known
            self._not_read[key] = (st.st_mtime_ns, st.st_size, "too_large")
            raise _too_large(key[1]) from None
        except documents.ReadLocked:  # v0.4.4 (F4): held open and locked by another program
            self._not_read[key] = (st.st_mtime_ns, st.st_size, "locked")
            raise _locked(key[1]) from None
        except documents.ReadDenied as exc:
            self._not_read[key] = (st.st_mtime_ns, st.st_size, "denied", str(exc))
            raise
        except documents.ReadIO:  # v0.4.4 (review fix): can pass; marked "(non lu)", tried again at the next call
            self._not_read[key] = (st.st_mtime_ns, st.st_size, "io")
            raise _unavailable(key[1]) from None
        except documents.DocumentError as exc:
            self._not_read[key] = (st.st_mtime_ns, st.st_size, "unreadable", str(exc))
            raise

    def _guard_doc(self, d: Dossier, entry: _CachedDoc) -> None:
        """Run the last canary on a whole cached document and chain its edits to the masker alignment.

        When the vault of another dossier cannot be opened, the document stays unguarded in the cache (a scan
        still works); every model-facing read or search then fails closed on the same vault."""
        try:
            guarded, steps, _n = self._guard_full(entry.masked0, d)
            epoch = self._guard_epoch()
        except ShieldError as exc:
            if exc.kind != "vault_error":
                raise
            guarded, steps, epoch = entry.masked0, [], ()
        entry.masked = guarded
        entry.alignment = _Chain([entry.align0] + steps) if entry.align0 is not None and steps is not None else None
        entry._folded_masked = None
        entry.epoch = epoch

    def _safe_epoch(self) -> Optional[tuple]:
        """:meth:`_guard_epoch`, or None when a vault cannot be opened."""
        try:
            return self._guard_epoch()
        except ShieldError as exc:
            if exc.kind != "vault_error":
                raise
            return None

    def _mask_label(self, d: Optional[Dossier], text: str) -> str:
        """Mask a short label (file or folder name): the dossier rules, then a word-by-word pass that also
        catches names glued by hyphens, digits or capitals ("Kervadec-c-Veltrane", "MaelleKervadec",
        "Kervadec2026", "kervadec's notes") for the registered values of every dossier."""
        if not text:
            return text
        spaced = _nfc(text).replace("_", " ")
        if d is not None:
            masked = self._rules(d).mask(spaced).text
        else:
            masked = Masker(MemoryStore(locale=config.locale()), locale=config.locale(), use_ner=False).mask(spaced).text
        return self._label_pass(masked, d)

    def _label_index(self, d: Dossier) -> dict:
        """Name words of the dossier's people and companies, for :meth:`_label_pass` (cached per version)."""
        rev = d.revision
        cached = self._label_indexes.get(d.code)
        if cached is not None and cached[0] == rev:
            return cached[1]
        from .engine import names as nm

        stop, legal, common, _fn, afn, months, wl = nm._lists()
        strong: dict[str, set[str]] = {}
        weak: dict[str, set[str]] = {}
        weak_surname: dict[str, set[str]] = {}
        first: dict[str, set[str]] = {}
        acro: dict[str, set[str]] = {}
        initials: dict[str, set[str]] = {}

        def classify(tok: str, code: str, surname: bool = False) -> None:
            if len(tok) < 3 or tok in stop or tok in legal or tok in months or tok in wl or tok in nm.LEGAL_FORMS \
                    or tok in nm.ORG_GENERIC or tok in nm.TITLE_WORDS or tok in nm.PARTICLES:
                return
            is_weak = tok in common or tok in afn or len(tok) < 4
            (weak if is_weak else strong).setdefault(tok, set()).add(code)
            if is_weak and surname and len(tok) >= 4:
                weak_surname.setdefault(tok, set()).add(code)

        for e in spreadable(d.all()):  # F0 (0.4.3): weak values are never searched for in a label
            if e.kind == "PERSON":
                for tok in nm.surname_key(e.value).split():
                    classify(tok, e.code, surname=True)
                for tok in nm.first_key(e.value).split():
                    if len(tok) >= 3:
                        first.setdefault(tok, set()).add(e.code)
                for ini in _person_initials(e.value):
                    initials.setdefault(ini, set()).add(e.code)
            elif e.kind == "ORG":
                for tok in _safe_norm("ORG", e.value).split():
                    classify(tok, e.code)
                for a in _org_acronyms(e.value):
                    acro.setdefault(a, set()).add(e.code)
        index = {"strong": strong, "weak": weak, "first": first, "weak_surname": weak_surname, "acro": acro,
                 "initials": initials}
        self._label_indexes[d.code] = (rev, index)
        return index

    def _label_pass(self, label: str, primary: Optional[Dossier]) -> str:
        """Replace the words of a label that are registered names of any dossier (own dossier: the code)."""
        toks = _label_tokens(label)
        if not toks:
            return label
        from .engine.normalize import fold_word

        folded = [fold_word(label[a:b]) for a, b in toks]
        hit: list[Optional[tuple[frozenset, Dossier]]] = [None] * len(toks)
        tails: dict[int, int] = {}  # last token of dotted initials -> end of the match (its final dot)
        for d in self._open_dossiers():
            if not d.all():
                continue
            idx = self._label_index(d)
            for i, (a, b) in enumerate(toks):
                if hit[i] is not None:
                    continue
                f = folded[i]
                codes = idx["strong"].get(f)
                if not codes and len(f) >= 4 and label[a:b].isupper():
                    codes = idx["weak"].get(f)
                if codes:
                    codes = {c for c in codes if not self._code_kept(d, c)}
                if codes:
                    hit[i] = (frozenset(codes), d)
            if primary is not None and d.code == primary.code:
                tails.update(self._own_label_forms(label, toks, folded, hit, d, idx))
            for i, f in enumerate(folded):  # a first name next to its own surname: "MaelleKervadec"
                if hit[i] is not None or f not in idx["first"]:
                    continue
                for j in (i - 1, i + 1):
                    if 0 <= j < len(toks) and hit[j] is not None and hit[j][1] is d \
                            and idx["first"][f] & hit[j][0] and _label_glue(label, toks, min(i, j)):
                        hit[i] = (frozenset(idx["first"][f] & hit[j][0]), d)
                        break
        if not any(hit):
            return label
        out: list[str] = []
        pos = 0
        i = 0
        while i < len(toks):
            if hit[i] is None:
                i += 1
                continue
            j = i
            while j + 1 < len(toks) and hit[j + 1] is not None and hit[j + 1] == hit[i] and _label_glue(label, toks, j):
                j += 1
            codes, d = hit[i]
            code = next(iter(codes)) if len(codes) == 1 else None
            ent = d.get_by_code(code) if code else d.get_by_code(next(iter(codes)))
            kind = ent.kind if ent is not None else "PERSON"
            own = primary is not None and d.code == primary.code and code is not None
            rep = code if own else None
            if rep is None and primary is not None and d.code != primary.code:
                # a name of another dossier in this dossier's label: a new, restorable code of this dossier (CL-26)
                minted = self._mint(primary, kind, label[toks[i][0]:toks[j][1]])
                rep = minted.code if minted is not None else None
            out.append(label[pos:toks[i][0]])
            out.append(rep or f"[{prefix_for(kind, d.locale)}]")
            pos = max(toks[j][1], tails.get(j, 0))
            i = j + 1
        out.append(label[pos:])
        return "".join(out)

    def _own_label_forms(self, label: str, toks: list[tuple[int, int]], folded: list[str],
                         hit: list, d: Dossier, idx: dict) -> dict[int, int]:
        """Short forms of the dossier's OWN entities in a label (R1 corrections CL-01, CL-02, CL-05).

        - an upper-case group of 3 to 6 letters that is the acronym of exactly one registered company
          ("Data room XQV", "XQV-310"), or of 3 or 4 letters that is the initials of exactly one registered
          person;
        - dotted initials ("R.T.", "J.-B. W.") of registered persons;
        - a capitalised surname that is also a common word or a first name ("Bulletins Serrurier 2024"), when
          exactly one person of the dossier has it as a surname and no other one as a first name.
        Only names already known in the dossier count: a short form is never guessed from nothing.
        Returns {index of the last token of dotted initials: end of the match}, so the final dot goes too."""
        tails: dict[int, int] = {}
        for i, (a, b) in enumerate(toks):
            if hit[i] is not None:
                continue
            f, surface = folded[i], label[a:b]
            codes: set[str] = set()
            if surface.isalpha() and surface.isupper() and 3 <= len(f) <= 6:
                codes = set(idx["acro"].get(f, ())) | (set(idx["initials"].get(f, ())) if len(f) <= 4 else set())
                if len(codes) != 1:
                    codes = set()
            elif len(f) >= 4 and surface[:1].isupper() and surface[1:].islower():
                owners = idx["weak_surname"].get(f, set())
                if len(owners) == 1 and not (idx["first"].get(f, set()) - owners):
                    codes = set(owners)
            codes = {c for c in codes if not self._code_kept(d, c)}
            if codes:
                hit[i] = (frozenset(codes), d)
        for m in _DOTTED_INITIALS_RE.finditer(label):
            letters = re.sub(r"[^A-Za-z]", "", m.group()).lower()
            if _weak_word(letters):
                continue  # "S.A." is a legal form, not a person
            codes = {c for c in idx["initials"].get(letters, ()) if not self._code_kept(d, c)}
            inside = [i for i, (a, b) in enumerate(toks) if a >= m.start() and b <= m.end()]
            if codes and inside and all(hit[i] is None for i in inside):
                for i in inside:
                    hit[i] = (frozenset(codes), d)
                tails[inside[-1]] = m.end()
        return tails

    @staticmethod
    def _code_kept(d: Dossier, code: str) -> bool:
        ent = d.get_by_code(code)
        return bool(ent is not None and d.is_kept_clear(ent.norm))

    def _clean_label(self, d: Optional[Dossier], label: str, fallback: str) -> str:
        """A masked label that is safe to show or to use as a download name; ``fallback`` when the canary
        still finds a registered value in it."""
        masked = self._guard(self._mask_label(d, label), d)
        with output_dossier(d.code if d is not None else None):
            still = self._has_registered(masked)
        if still:
            return fallback
        return masked

    # ================================================================ model-facing API
    @_guarded("list_dossiers")
    def list_dossiers(self, include_private: bool = False) -> list[dict]:
        """[{code, documents, entities}] (codes only). ``include_private`` adds title and folder (local only).

        v0.4.3: no service lock (the registry and the vaults have their own): it answers during a background
        reading. ``reading`` is the progress of a dossier whose reading is not finished."""
        out = []
        for code in self.registry.folders():
            try:
                reading = self.reading_state(code)
                if reading is not None and not include_private:
                    # being read: the progress only (no folder walk, no vault query: the answer stays fast while a
                    # long document is masked in the background)
                    out.append({"code": code, "documents": reading.get("total"), "entities": None,
                                "reading": reading})
                    continue
                d = self.registry.get(code)
                row: dict[str, Any] = {"code": d.code, "documents": len(d.documents()), "entities": len(d.all())}
                if reading is not None:
                    row["reading"] = reading
                if include_private:
                    row.update(title=d.title, folder=str(d.folder), replacement=self._dossier_replacement(d))
            except VaultError:
                row = {"code": code, "documents": None, "entities": None, "error": "vault_unreadable"}
            out.append(row)
        return out

    @_guarded("list_documents")
    def list_documents(self, dossier_code: str) -> list[dict]:
        """[{id, masked_name, masked_path, kind, size, pages}] with masked file names.

        Files Shield cannot read (spreadsheets, images...) are listed too, with kind ``"unsupported"`` (R1
        correction CL-28): the model can tell the user a file exists and ask for a readable copy."""
        self._require_ready(dossier_code)  # v0.4.3: never a half-read dossier
        with self._lock:
            d = self._dossier(dossier_code)
            refs = d.documents(include_unsupported=True)
            kept = self._labels_now(d)  # v0.4.4: the masked names made at the end of the reading (or kept)
            labels: list[str] = []
            rows = []
            for ref in refs:
                stem, ext = os.path.splitext(ref.name)
                first = len(labels)
                if kept is not None and ref.id in kept:
                    labels.extend(kept[ref.id])
                else:
                    labels.extend(self._ref_labels(d, ref))
                rows.append((ref, ext.lower(), first, len(labels)))
            labels = self._guard_many(labels, d)
            out = []
            for ref, ext, a, b in rows:
                masked_name = labels[b - 1] + ext
                pages = None
                if ref.kind == "pdf":
                    cached = self._cached((d.code, ref.id))
                    pages = cached.pages if cached is not None and cached.pages is not None else \
                        self._pdf_pages(ref.path, ref.path.stat().st_mtime_ns, ref.size)
                bad = self._not_read.get((d.code, ref.id))
                # "too_long", "too_large" (v0.4.3); v0.4.4 (F4): "locked", "denied", "unreadable"
                not_read = bad[2] if bad is not None and bad[1] == ref.size else None
                out.append({"id": ref.id, "masked_name": masked_name,
                            "masked_path": "/".join(labels[a:b - 1] + [masked_name]),
                            "kind": ref.kind, "size": ref.size, "pages": pages, "not_read": not_read})
            d.journal("list_documents", documents=len(out))
            return out

    def _ref_labels(self, d: Dossier, ref: Any) -> list[str]:
        """The masked names (sub-folders, then the file name without extension) of one document, before the last
        check: what shield_list_documents shows."""
        stem, _ext = os.path.splitext(ref.name)
        parent = Path(ref.rel).parent.as_posix() if ref.rel else "."
        parts = [] if parent in ("", ".") else parent.split("/")
        return [self._mask_label(d, p) for p in parts] + [self._mask_label(d, stem)]

    def _labels_now(self, d: Dossier) -> Optional[dict[str, list[str]]]:
        """v0.4.4: the masked names made at the end of the reading of ``d`` (or kept with it), while the dossier and
        the last check are as they were then; None when they changed (the names are masked again, as in 0.4.3)."""
        held = self._kset_labels.get(d.code)
        if held is None:
            return None
        rev, epoch, labels = held
        if rev != self._version(d) or epoch != self._safe_epoch():
            return None
        return labels

    @_guarded("read_document")
    def read_document(self, dossier_code: str, doc_id: str, start: int = 0,
                      max_chars: int = DEFAULT_READ_CHARS) -> dict:
        """One page of a masked document: {id, text, start, next, total_chars, needs_ocr, pages, warnings}."""
        self._require_ready(dossier_code)  # v0.4.3: never a half-read dossier
        with self._lock:
            d = self._dossier(dossier_code)
            doc = _doc_code(doc_id)
            ref = d.get_document(doc)
            if ref is None:
                other = d.get_document(doc, include_unsupported=True)
                if other is not None and other.kind == UNSUPPORTED_KIND:
                    raise ShieldError(f"{doc} : format non pris en charge, Shield ne peut pas lire ce fichier. "
                                      "Demandez à l'utilisateur une copie en PDF, Word ou texte "
                                      "(unsupported_format).", "unsupported_format")
                raise ShieldError(f"{doc} introuvable dans {d.code} (unknown document). "
                                  "Appelez shield_list_documents.", "unknown_document")
            try:
                start = max(0, int(start or 0))
                max_chars = int(max_chars or DEFAULT_READ_CHARS)
            except (TypeError, ValueError):
                raise ShieldError("start et max_chars doivent être des nombres (must be integers).",
                                  "invalid_argument") from None
            max_chars = min(MAX_READ_CHARS, max(MIN_READ_CHARS, max_chars))
            entry = self._doc(d, ref, fresh=start == 0)
            total = len(entry.masked)
            if start >= total:
                page, end = "", total
            else:
                end = _cut(entry.masked, start, max_chars)
                page = entry.masked[start:end]
            d.journal("read_document", document=ref.id, chars=len(page))
            return {"id": ref.id, "text": self._guard(page, d), "start": min(start, total),
                    "next": end if end < total else None, "total_chars": total,
                    "needs_ocr": entry.needs_ocr, "pages": entry.pages, "warnings": list(entry.warnings)}

    @_guarded("search")
    def search(self, dossier_code: str, query: str, limit: int = DEFAULT_SEARCH_RESULTS) -> list[dict]:
        """Search the dossier locally. Codes in the query match every masked mention of that entity; other
        words are searched in the real text, case and accent insensitive. Returns masked snippets:
        [{document, snippet, offset}] where ``offset`` is a ``start`` for ``read_document``."""
        self._require_ready(dossier_code)  # v0.4.3: never a half-read dossier
        with self._lock:
            d = self._dossier(dossier_code)
            q = " ".join(str(query or "").split())
            if not q:
                raise ShieldError("Recherche vide (empty query).", "invalid_argument")
            if len(q) > MAX_QUERY_CHARS:
                raise ShieldError(f"Recherche trop longue (max {MAX_QUERY_CHARS} caractères).", "invalid_argument")
            try:
                limit = min(MAX_SEARCH_RESULTS, max(1, int(limit or DEFAULT_SEARCH_RESULTS)))
            except (TypeError, ValueError):
                limit = DEFAULT_SEARCH_RESULTS
            m = self._masker(d)
            codes = []
            for cm in UNMASK_RE.finditer(q):
                if d.get_by_code(cm.group()) is not None and cm.group() not in codes:
                    codes.append(cm.group())
            plain = UNMASK_RE.sub(" ", q)
            terms = [t for t in dict.fromkeys(fold(plain).text.split()) if len(t) >= 3]
            # A phrase with codes is searched in the MASKED text, where real values never appear: a code next
            # to a guessed name can never confirm the guess.
            phrase = fold(q).text.strip() if codes else fold(plain).text.strip()
            terms = [t for t in terms if t not in _STOPWORDS] or terms
            hits: list[tuple[int, int, str, int, int]] = []  # (-score, doc order, doc id, mstart, mend)
            epoch = self._guard_epoch()
            entries: dict[str, _CachedDoc] = {}
            for order, ref in enumerate(d.documents()):
                try:
                    entry = self._doc(d, ref, fresh=False, epoch=epoch)
                except documents.DocumentError:
                    continue
                except ShieldError as exc:
                    if exc.kind in ("document_too_large", "document_too_long", "document_locked",
                                    "document_unavailable"):
                        continue
                    raise
                entries[ref.id] = entry
                if not entry.masked:
                    continue
                found = self._search_doc(entry, phrase, codes, terms, bool(codes))
                for score, ms, me in _score_hits(found):
                    hits.append((-score, order, ref.id, ms, me))
            hits.sort()
            out: list[dict] = []
            taken: dict[str, list[tuple[int, int]]] = {}
            for neg, _order, doc_id, ms, me in hits:
                if len(out) >= limit:
                    break
                spans = taken.setdefault(doc_id, [])
                if any(a <= ms < b or a < me <= b for a, b in spans):
                    continue
                entry = entries[doc_id]
                a = _snap_left(entry.masked, max(0, ms - SNIPPET_RADIUS), 40)
                b = _snap_right(entry.masked, min(len(entry.masked), me + SNIPPET_RADIUS), 40)
                spans.append((a, b))
                snippet = ("…" if a > 0 else "") + " ".join(entry.masked[a:b].split()) + \
                          ("…" if b < len(entry.masked) else "")
                out.append({"document": doc_id, "snippet": self._guard(snippet, d), "offset": a})
            d.journal("search", results=len(out))
            return out

    @staticmethod
    def _find(folded_text: str, needle: str) -> Iterable[tuple[int, int]]:
        if not needle:
            return []
        pat = re.compile(r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9])")
        return [(mm.start(), mm.end()) for mm in pat.finditer(folded_text)]

    def _search_doc(self, entry: _CachedDoc, phrase: str, codes: list[str], terms: list[str],
                    phrase_has_codes: bool) -> list[tuple[int, str, int, int]]:
        """(weight, item, masked start, masked end) of every match in one document."""
        found: list[tuple[int, str, int, int]] = []
        al = entry.alignment
        if phrase_has_codes:
            fm = entry.folded_masked
            for fs, fe in self._find(fm.text, phrase):
                s, e = fm.span(fs, fe)
                found.append((6, "\0phrase", s, e))
        if al is not None and entry.search_text:
            fr = entry.folded_raw

            def visible(s: int, e: int, needle: str) -> Optional[tuple[int, int]]:
                """Masked offsets of a raw hit, only when the model sees the same words there (CL-25): a word
                hidden by the masker or by the last canary never yields a hit."""
                if al.overlaps(s, e):
                    return None  # a plain word inside a masked value: never confirm a guess
                ms, me = al.to_masked(s), al.to_masked(e)
                if fold(entry.masked[ms:me]).text.strip() != needle:
                    return None
                return ms, me

            for fs, fe in ([] if phrase_has_codes else self._find(fr.text, phrase)):
                pos = visible(*fr.span(fs, fe), phrase)
                if pos is not None:
                    found.append((6, "\0phrase", pos[0], pos[1]))
            for term in terms:
                for fs, fe in self._find(fr.text, term):
                    pos = visible(*fr.span(fs, fe), term)
                    if pos is not None:
                        found.append((1, term, pos[0], pos[1]))
        else:
            fm = entry.folded_masked
            if not phrase_has_codes:
                for fs, fe in self._find(fm.text, phrase):
                    s, e = fm.span(fs, fe)
                    found.append((6, "\0phrase", s, e))
            for term in terms:
                for fs, fe in self._find(fm.text, term):
                    s, e = fm.span(fs, fe)
                    found.append((1, term, s, e))
        for code in codes:
            for mm in re.finditer(r"(?<![A-Za-z0-9])" + re.escape(code) + r"(?![A-Z0-9])", entry.masked):
                found.append((2, code, mm.start(), mm.end()))
        return found

    @_guarded("save_draft")
    def save_draft(self, dossier_code: str, title: str, content: str, fmt: str = "docx") -> dict:
        """Restore the codes locally and write the file into ``<dossier>/Shield - sorties``.

        Returns {id: OUTPUT_n, bytes, format, unknown_codes} only: never the path or the restored text.
        :meth:`save_draft_report` also lists what could not be restored.
        """
        r = self.save_draft_report(dossier_code, title, content, fmt)
        return {k: r[k] for k in ("id", "bytes", "format", "unknown_codes")}

    @_guarded("save_draft")
    def save_draft_report(self, dossier_code: str, title: str, content: str, fmt: str = "docx") -> dict:
        """:meth:`save_draft` that also returns ``placeholders`` (generic marks left in the file, one entry per
        occurrence, "[SOCIETE]") and ``shield_ids`` (DOC_n / DOSSIER_n / OUTPUT_n that got no local name): R1
        corrections CL-31 and CL-36. Codes and generic marks only, never a value or a label."""
        self._require_ready(dossier_code)  # v0.4.3: never a half-read dossier
        with self._lock:
            d = self._dossier(dossier_code)
            fmt = str(fmt or "docx").strip().lower().lstrip(".")
            if fmt not in ("docx", "md"):
                raise ShieldError("Format inconnu : docx ou md (unknown format).", "invalid_argument")
            content = str(content or "")
            if not content.strip():
                raise ShieldError("Contenu vide (empty content).", "invalid_argument")
            if len(content) > MAX_DRAFT_CHARS:
                raise ShieldError("Contenu trop long (content too long).", "invalid_argument")
            title = " ".join(str(title or "").split())[:MAX_TITLE_CHARS] or "Projet"
            m = self._masker(d)
            labels = self._id_labels(d, title + "\n" + content)
            body = self._restore_out(m, content, labels)
            head = self._restore_out(m, title, labels, max_chars=getattr(documents, "MAX_STEM_CHARS", None))
            unknown_all = list(dict.fromkeys(body["unknown_codes"] + head["unknown_codes"]))
            placeholders = body["placeholders"] + head["placeholders"]
            shield_ids = list(dict.fromkeys(body["shield_ids"] + head["shield_ids"]))
            if documents.output_folder_too_long(d.folder / config.OUTPUT_DIR_NAME, fmt):  # v0.4.4 (F3)
                raise _path_too_long(d.code)
            try:
                path = documents.save_output(d.output_dir(), head["text"], body["text"], fmt, d.code)
            except documents.DocumentError as exc:
                if str(exc) == documents.WRITE_BLOCKED:  # v0.4.4 (F1): the system refused the write
                    raise _write_blocked(d.code) from None
                if str(exc) == documents.OUTPUT_PATH_TOO_LONG:
                    raise _path_too_long(d.code) from None
                raise ShieldError(f"Écriture impossible dans le dossier de sortie de {d.code} (write_failed).",
                                  "write_failed") from None
            except OSError as exc:
                if documents.write_refused(exc):  # v0.4.4 (F1): PermissionError (or winerror 5) before OSError
                    raise _write_blocked(d.code) from None
                raise ShieldError(f"Écriture impossible dans le dossier de sortie de {d.code} (write_failed).",
                                  "write_failed") from None
            oid = d.register_output(path)
            size = path.stat().st_size
            d.journal("save_draft", output=oid, format=fmt, bytes=size, unknown_codes=len(unknown_all),
                      placeholders=len(placeholders), shield_ids=len(shield_ids))
            # Codes and generic marks only: the labels used for DOC_n / DOSSIER_n / OUTPUT_n never leave.
            return {"id": oid, "bytes": size, "format": fmt, "unknown_codes": unknown_all,
                    "placeholders": placeholders, "shield_ids": shield_ids}

    def _id_labels(self, d: Dossier, text: str) -> dict[str, str]:
        """Local names of the Shield ids a draft cites (R1 correction CL-36): DOC_n -> file name, the dossier's
        own DOSSIER_n -> its title, OUTPUT_n -> output file name. REAL values: for the restored local file only,
        never returned to a model. Ids of other dossiers and unknown ids get no label."""
        labels: dict[str, str] = {}
        for sid in dict.fromkeys(_SHIELD_ID_RE.findall(text or "")):
            name: Optional[str] = None
            if sid.startswith("DOC_"):
                ref = d.get_document(sid, include_unsupported=True)
                name = ref.name if ref is not None else None
            elif sid.startswith("DOSSIER_"):
                name = d.title if sid == d.code else None
            else:
                out = d.get_output(sid)
                name = out.name if out is not None else None
            if name:
                labels[sid] = name
        return labels

    @staticmethod
    def _restore_out(m: Masker, text: str, id_labels: dict[str, str], max_chars: Optional[int] = None) -> dict:
        """Restore ``text`` for a LOCAL output: {text, unknown_codes, placeholders, shield_ids, restored}.

        Uses ``restore_utils.restore_for_output`` (R1 RESTORE corrections: line breaks, elision, Shield ids...)
        when it is installed; otherwise the masker's restore, with the same report of the generic placeholders
        and Shield ids left in the text (R1 correction CL-31)."""
        helper = _restore_for_output()
        if helper is not None:
            kwargs: dict[str, Any] = {"id_labels": id_labels or None}
            if max_chars:
                kwargs["max_chars"] = max_chars
            rep = helper(text, m, **kwargs)
            return {"text": rep.text, "unknown_codes": list(rep.unknown_codes),
                    "placeholders": list(rep.placeholders), "shield_ids": list(dict.fromkeys(rep.shield_ids)),
                    "restored": int(rep.restored),
                    "restored_synthetic": int(getattr(rep, "restored_synthetic", 0)),
                    "unmatched_synthetic": int(getattr(rep, "unmatched_synthetic", 0))}
        real, unknown = m.unmask_report(text)
        return {"text": real, "unknown_codes": list(unknown), "placeholders": _PLACEHOLDER_RE.findall(text),
                "shield_ids": list(dict.fromkeys(_SHIELD_ID_RE.findall(text))),
                "restored": sum(1 for mm in UNMASK_RE.finditer(text) if m.store.get_by_code(mm.group()) is not None)}

    @_guarded("scan_text")
    def scan_text(self, text: str) -> list[dict]:
        """Canary across ALL dossiers: [{code, kind, dossier}] of the registered values found in ``text``.

        ``code`` is None when the canary found a value (for example an OCR variant) but cannot tell which
        entity it is. Never returns a value.
        """
        with self._lock:
            text = _nfc(str(text or ""))
            if not text:
                return []
            out: list[dict] = []
            prep: Optional[ScanPrep] = None
            for d in self._open_dossiers():
                if not d.all():
                    continue
                if prep is None:
                    prep = prepare_scan(text)
                out.extend(self._registered_in(d, text, prep))
            return out

    @_guarded("has_registered")
    def has_registered(self, text: str) -> bool:
        """True when ``text`` holds a registered value of any dossier (hooks: cheap first check)."""
        with self._lock:
            return self._has_registered(str(text or ""))

    @_guarded("status")
    def status(self) -> dict:
        """{version, engine_mode, locale, language, ner, dossiers, key_storage, ocr, limits, replacement}. Codes
        and counts only (the NER block names packages, models and the models directory, never a dossier).

        v0.4.3: no service lock (it answers while a background reading masks a long document)."""
        mode = Masker(MemoryStore(), locale=config.locale(), ner_model=config.ner_model()).mode
        try:
            st = _ner.status(config.ner_model())
            ner_info = {"setting": st["setting"], "model_setting": st["model_setting"], "backend": st["backend"],
                        "loaded": bool(st["loaded"]), "max_chars": st["max_chars"],
                        "models": {k: bool(m["cached"]) for k, m in st["models"].items()},
                        "models_dir": st["models_dir"]}
        except Exception:  # noqa: BLE001 - status never fails on the optional layer
            ner_info = {"setting": "auto", "backend": None, "loaded": False, "models": {}}
        return {"version": __version__, "engine_mode": mode, "locale": config.locale(),
                "language": config.language(), "ner": ner_info,
                "dossiers": len(self.registry.folders()), "key_storage": self.registry.key_source,
                "ocr": False, "network": False,
                "replacement": config.replacement(), "replacements": ["codes", "synthetic"],
                "own_company_vendor": config.own_company_vendor(),
                "limits": [fr for fr, _en in LIMITS], "limits_en": [en for _fr, en in LIMITS]}

    # ================================================================ hook support
    @_guarded("guard_output")
    def guard_output(self, text: str, dossier_code: Optional[str] = None) -> str:
        """Last canary for any model-facing string: registered values of any dossier become ``[PERSONNE]``...

        With a dossier (argument, or :func:`output_dossier` around the call), the text belongs to that dossier:
        its values get their codes and values of other dossiers new codes of it (R1 correction CL-26)."""
        with self._lock:
            if dossier_code:
                with output_dossier(dossier_code):
                    primary = self._context_dossier()
            else:
                primary = self._context_dossier()
            return self._guard(str(text or ""), primary)

    @_guarded("guard_output")
    def guard_parts(self, parts: list[str], dossier_code: Optional[str] = None) -> list[str]:
        """F0 (0.4.3): the last check on the CONTENT parts of a tool answer (file names, passages, search
        results, titles), one string each, in one canary pass per dossier. The words, codes and numbers that Shield
        writes itself are never passed here. Same rules as :meth:`guard_output`."""
        items = [str(x or "") for x in parts]
        if not any(items):
            return items
        with self._lock:
            if dossier_code:
                with output_dossier(dossier_code):
                    primary = self._context_dossier()
            else:
                primary = self._context_dossier()
            return self._guard_many(items, primary)

    @_guarded("protect")
    def protect(self, text: str) -> tuple[str, list[str]]:
        """Mask every registered value left in ``text`` (any dossier). Returns (text, codes added).

        For outputs that did not come from Shield (other tools' results). The dossier with the most hits is
        masked with its own masker (so names found next to its values are masked too, and added to it);
        codes already in the text are kept. Values of other dossiers, and values the strict masker misses
        (OCR variants), are replaced by the canary.
        """
        with self._lock:
            original = text = _nfc(str(text or ""))
            if not text:
                return text, []
            ranked = []
            prep: Optional[ScanPrep] = None
            for d in self._open_dossiers():
                if not d.all():
                    continue
                if prep is None:
                    prep = prepare_scan(text)
                hits = self._leftovers(d, text, prep)
                if hits:
                    ranked.append((len(hits), d))
            if not ranked:
                return text, []
            ranked.sort(key=lambda x: (-x[0], x[1].num))
            first = ranked[0][1]
            res = self._rules(first).mask(text, keep_known_codes=True)
            text = res.text
            if res.new_entities:
                self._invalidate(first.code)
            prep = None
            for _n, d in ranked:
                if prep is None:
                    prep = prepare_scan(text)
                left = self._leftovers(d, text, prep)
                if left:
                    text = self._redact(text, left, d, own=d.code == first.code)[0]
                    prep = None
            before = Counter(mm.group() for mm in UNMASK_RE.finditer(original))
            added = []
            for mm in UNMASK_RE.finditer(text):
                code = mm.group()
                if code not in added and text.count(code) > before.get(code, 0):
                    added.append(code)
            return text, added

    # ================================================================ local API (Desk, CLI)
    @_guarded("add_dossier")
    def add_dossier(self, folder: os.PathLike | str, title: Optional[str] = None) -> dict:
        """Register a folder, scan its documents and register the parties named in the title ("X c. Y").

        Returns {code, documents, entities, counts, parties, unreadable, needs_ocr}. Codes and counts only.
        """
        with self._lock:
            existed = self.registry.is_registered(folder)
            d = self.registry.add_dossier(folder, title)
            typed = bool(title and title.strip())
            title_changed = False
            if existed and typed and normalize(" ".join(title.split())) != normalize(d.title):
                # The folder was already registered and the user typed a new title: keep it and read its parties.
                self.registry.set_title(d.code, title)
                title_changed = True
            try:
                summary = self._scan(d)
            except BaseException:
                if not existed:  # never leave a half-made dossier behind
                    try:
                        self.registry.remove(d.code)
                    except Exception:  # noqa: BLE001
                        log.warning("RAGbase Shield: %s could not be rolled back after a failed scan", d.code)
                    self._drop_maskers(d.code)
                    self._rule_maskers.pop(d.code, None)
                    self._invalidate(d.code)
                raise
            added = 0
            # A typed title is parsed for parties. The folder-name fallback ("Dossier Ferracci-Daubel") is
            # parsed only when it names opposing parties ("X c. Y"), so a description never becomes a person.
            source = d.title if (typed or has_party_separator(d.title)) else ""
            for kind, name in parse_parties(source):
                if self._party_known(d, kind, name):
                    continue
                try:
                    d.register(kind, name)
                    added += 1
                except ValueError:
                    continue
            if added:
                self._invalidate(d.code)
            d.journal("add_dossier", parties=added, documents=summary["documents"])
            with self._reads_lock:
                self._warmed.add(d.code)  # v0.4.4: read in this process: no second reading at the first call
            summary.update(code=d.code, parties=added, entities=len(d.all()), counts=_kind_counts(d),
                           existing=existed, title_updated=title_changed)
            return summary

    @_guarded("register_folder_parties")
    def register_folder_parties(self, dossier_code: str) -> int:
        """Register the parties that the dossier's SUB-folder names cite ("Martin-Pradel c. Boisgontier"), as
        :meth:`add_dossier` does for the top folder name, so every document and label of the dossier uses the same
        code for them. Only names with opposing parties ("X c. Y") are read. Local API (plugin). Returns the
        number of new entities."""
        with self._lock:
            return self._register_folder_parties(self._dossier(dossier_code))

    def _register_folder_parties(self, d: Dossier) -> int:
        seen: set[str] = set()
        added = 0
        for ref in d.documents(include_unsupported=True):
            parent = Path(ref.rel).parent if ref.rel else Path(".")
            for part in parent.parts:
                if part in seen or part in (".", "") or not has_party_separator(part):
                    continue
                seen.add(part)
                for kind, name in parse_parties(part):
                    if self._party_known(d, kind, name):
                        continue
                    try:
                        d.register(kind, name)
                        added += 1
                    except ValueError:
                        continue
        if added:
            self._invalidate(d.code)
            d.journal("register_folder_parties", parties=added)
        return added

    def _register_title_parties(self, d: Dossier) -> int:
        """The parties named in the dossier title when it names opposing parties ("X c. Y"), as
        :meth:`add_dossier` does for a folder-name title (plugin: the title is always the folder name)."""
        added = 0
        source = d.title if has_party_separator(d.title) else ""
        for kind, name in parse_parties(source):
            if self._party_known(d, kind, name):
                continue
            try:
                d.register(kind, name)
                added += 1
            except ValueError:
                continue
        if added:
            self._invalidate(d.code)
        return added

    @_guarded("document_folders")
    def document_folders(self, dossier_code: str) -> dict[str, str]:
        """{DOC_n: sub-folder of the document, relative to the dossier ("" at the top)}. REAL names: local use only
        (the plugin turns them into neutral labels)."""
        self._require_ready(dossier_code)  # v0.4.3: never a half-read dossier
        with self._lock:
            d = self._dossier(dossier_code)
            out: dict[str, str] = {}
            for ref in d.documents(include_unsupported=True):
                parent = Path(ref.rel).parent.as_posix() if ref.rel else ""
                out[ref.id] = "" if parent in (".", "") else parent
            return out

    @_guarded("registered_folders")
    def registered_folders(self) -> dict[str, Path]:
        """{DOSSIER_n: folder} from the registry. REAL paths: local use only. No service lock (v0.4.3): the
        registry has its own, so the folder window's answer never waits for a background masking."""
        return self.registry.folders()

    def _party_known(self, d: Dossier, kind: str, name: str) -> bool:
        """True when a title party is already one of the dossier's entities, under ANY kind: "Ateliers Brunelac"
        typed as a person is the company the documents found; "Kerdalis" is the company "KERDALIS INDUSTRIES"."""
        for k in ("PERSON", "ORG"):
            if self._same_entity(d, k, name) is not None:
                return True
        other = "ORG" if kind == "PERSON" else "PERSON"
        if self._links_to_known(d, kind, name) or self._links_to_known(d, other, name):
            return True
        words = fold(name).text.split()
        if len(words) == 1:
            for e in d.all():
                if e.kind == "ORG" and _safe_norm("ORG", e.value).split()[:1] == words:
                    return True
        return False

    def _links_to_known(self, d: Dossier, kind: str, name: str) -> bool:
        """True when ``name`` is already covered by one known entity of the same kind (no new code needed)."""
        if not d.all():
            return False
        store = _DryStore(d)
        res = Masker(store, locale=d.locale, use_ner=False).mask(name)
        if _SENTINEL_RE.search(res.text) or not res.spans:
            return False
        kinds = {d.get_by_code(mm.group()).kind for mm in UNMASK_RE.finditer(res.text) if d.get_by_code(mm.group())}
        if kinds != {kind}:
            return False
        rest = UNMASK_RE.sub(" ", res.text)
        return not re.search(r"[^\W\d_]{4,}", rest) or _only_forms(rest)

    def _build_directory(self, d: Dossier) -> int:
        """Build the dossier directory from the folder (e-mail headers, Slack ``users.json``, Word authors,
        signature blocks, nickname declarations), store it in the vault and seed the store with every full name,
        before the documents are masked (v0.3, S3). Never fails a scan: on any error the dossier keeps its
        previous directory. Returns the number of people found."""
        set_directory = getattr(d, "set_directory", None)
        if not callable(set_directory):
            return 0
        try:
            from .engine import directory as dr

            items: list = [ref.path for ref in d.documents()]
            scan_folder = getattr(d, "_scan_folder", None)
            if callable(scan_folder):
                for rel, (path, _size) in scan_folder(True).items():
                    if path.name.lower() == "users.json":
                        items.append(path)
            lang = self._dossier_language(d)
            directory = dr.build(items, lang=lang if lang in ("fr", "en") else "auto")
            set_directory(directory.to_dict() if directory else None)
            if directory:
                directory.seed(d, source="directory")
            return len(directory)
        except Exception as exc:  # noqa: BLE001 - the directory is an aid, never a blocker
            log.warning("RAGbase Shield: %s: directory not built (%s)", d.code, type(exc).__name__)
            return 0

    def _register_pass(self, d: Dossier, raws: Optional[dict] = None) -> None:
        """v0.3 (synthetic leak fix, PLAN 5.1): a dossier in synthetic mode registers every value of every readable
        document in codes mode before a single fake is drawn, so the Synthesizer's collision rules see the whole
        dossier (a fake can never equal a real value that a later document holds). Rules-speed; never blocks."""
        m = self._masker(d, "codes")
        for ref in d.documents():
            known = (raws or {}).get(ref.id)
            if isinstance(known, tuple):
                raw = known[2]
            elif isinstance(known, str):
                continue  # not read (too long, unreadable)
            else:
                try:
                    raw = documents.read_text(ref.path)
                except (documents.DocumentError, OSError):
                    continue
            if raw.needs_ocr or not raw.text or len(raw.text) > MAX_DOCUMENT_CHARS:
                continue
            try:
                m.mask(raw.text)
            except Exception as exc:  # noqa: BLE001 - the pass is a guard; the document is masked again below
                log.warning("RAGbase Shield: %s %s: registration pass failed (%s)", d.code, ref.id, type(exc).__name__)

    def _scan(self, d: Dossier) -> dict:
        persons = self._build_directory(d)
        if self._dossier_replacement(d) == "synthetic":
            self._register_pass(d)
        summary = self._mask_all(d, d.documents())
        summary["directory"] = persons
        return summary

    def _mask_all(self, d: Dossier, refs: list, raws: Optional[dict] = None, job: Optional[_Reading] = None,
                  locked: bool = True, count: bool = True) -> dict:
        """Mask every document of ``refs`` (read before when ``raws`` holds its text). Returns the counts:
        {documents, unreadable, too_long, needs_ocr}. ``locked=False`` (background reading): the service lock is
        taken for one document at a time, so the other calls go on between two documents."""
        n = unreadable = ocr = too_long = 0
        for ref in refs:
            if job is not None and self._stop.is_set():
                raise _Stopped()
            n += 1
            raw = (raws or {}).get(ref.id)
            if raw in (documents.READ_TOO_LONG, documents.READ_TOO_LARGE):
                too_long += 1  # v0.4.3 fix: "non lu(s) car trop long(s)", for both reasons
            elif isinstance(raw, str):
                unreadable += 1
            else:
                if job is not None:  # v0.4.4 (B2): the budget of this document (its reading, and its size)
                    chars = len(raw[2]) if isinstance(raw, tuple) else 0
                    job.step(_mask_budget(chars, job.budgets.get(ref.id, 0.0)))
                try:
                    entry = self._doc(d, ref, fresh=True, raw=raw) if locked else self._doc_unlocked(d, ref, raw)
                    if entry.needs_ocr:
                        ocr += 1
                except (documents.DocumentError, OSError):
                    unreadable += 1
                except ShieldError as exc:
                    if exc.kind in ("document_too_long", "document_too_large"):
                        too_long += 1
                    else:
                        raise
            if job is not None and count:
                job.masked += 1
        return {"documents": n, "unreadable": unreadable, "too_long": too_long, "needs_ocr": ocr}

    def _doc_unlocked(self, d: Dossier, ref: Any, raw: Optional[tuple]) -> _CachedDoc:
        """:meth:`_doc` for a background reading (v0.4.3 fix): the cache is checked and written under the service
        lock, but the masking and the last canary of the document run outside it, so the calls of a ready dossier
        never wait for the masking of a long document. Only this reading uses this dossier meanwhile (its tools give
        the progress answer), and its thread has its own rules maskers."""
        key = (d.code, ref.id)
        with self._lock:
            st = ref.path.stat()
            hit, raw_text = self._doc_lookup(d, key, st, True, None, raw, guard=False)
            m = self._masker(d) if hit is None else None
        if hit is not None:
            try:
                self._doc_reguard(d, hit)
                with self._lock:
                    if key in self._docs:
                        self._docs.move_to_end(key)
                return hit
            except _KeptStale:  # v0.4.4: masked again from the text read before (or from the file)
                with self._lock:
                    self._drop_cached(key, hit)
                    m = self._masker(d)
                if raw is not None and raw[0] == st.st_mtime_ns and raw[1] == st.st_size:
                    raw_text = raw[2]
        if raw_text is None:
            raw_text = self._read_file(key, ref.path, st)
        entry = self._doc_make(d, ref, st, raw_text, m)
        with self._lock:
            return self._doc_store(key, entry)

    @_guarded("scan_dossier")
    def scan_dossier(self, dossier_code: str) -> dict:
        """Re-read the folder and mask every document. {code, documents, entities, counts, unreadable, needs_ocr}."""
        with self._lock:
            d = self._dossier(dossier_code)
            summary = self._scan(d)
            with self._reads_lock:
                self._warmed.add(d.code)
            d.journal("scan", documents=summary["documents"], entities=len(d.all()))
            summary.update(code=d.code, entities=len(d.all()), counts=_kind_counts(d))
            return summary

    # ================================================================ background reading (v0.4.3, plugin)
    def _home_dir(self) -> Path:
        reg = self._registry
        if reg is not None:
            return Path(reg.home)
        return Path(self._home) if self._home is not None else config.home()

    def _mark_path(self, code: str) -> Path:
        return self._home_dir() / READING_DIR_NAME / (code + _READING_SUFFIX)

    def _write_mark(self, code: str, uuid: str) -> bool:
        """Write the mark "<dossier id> <process id> <owner>" (v0.4.3 fix: a few tries, for a file that an
        antivirus or the indexer holds on Windows). Raises OSError when it cannot be written."""
        path = self._mark_path(code)
        tmp = path.with_name(f"{path.name}.{self._token[:8]}.tmp")
        text = f"{uuid} {os.getpid()} {self._token}\n"
        for attempt in range(_MARK_WRITE_TRIES):
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_text(text, encoding="ascii")
                os.replace(tmp, path)
                return True
            except OSError:
                if attempt + 1 >= _MARK_WRITE_TRIES:
                    with contextlib.suppress(OSError):
                        tmp.unlink()
                    raise
                time.sleep(0.2)
        return False

    def _mark_info(self, code: str) -> Optional[dict]:
        """The mark of a dossier on disk: {uuid, pid, owner, age} (age in seconds since its last touch), or None."""
        path = self._mark_path(code)
        try:
            st = path.stat()
            fields = path.read_text(encoding="ascii").split()
        except (OSError, UnicodeDecodeError):
            return None
        if not fields:
            return None
        pid: Optional[int] = None
        if len(fields) > 1 and fields[1].isdigit():
            pid = int(fields[1])
        return {"uuid": fields[0], "pid": pid, "owner": fields[2] if len(fields) > 2 else None,
                "age": max(0.0, time.time() - st.st_mtime)}

    def _disk_mark(self, code: str) -> Optional[str]:
        info = self._mark_info(code)
        return info["uuid"] if info is not None else None

    def _foreign(self, info: Optional[dict]) -> bool:
        """True when a mark belongs to a reading that goes on in ANOTHER service (another conversation): its owner is
        not this service, its process is alive and it was touched less than MARK_STALE_SECONDS ago."""
        if info is None or info.get("owner") == self._token:
            return False
        return info["age"] < MARK_STALE_SECONDS and _pid_alive(info.get("pid"))

    def _touch_mark(self, code: str) -> None:
        with contextlib.suppress(OSError):
            os.utime(self._mark_path(code))  # never creates the file: a removed mark stays removed

    def _release_mark(self, code: str) -> None:
        """Our reading stopped (the service closes): the mark stays (the dossier is not read yet) but is free at
        once for the next start, with no wait for MARK_STALE_SECONDS."""
        info = self._mark_info(code)
        if info is not None and info.get("owner") == self._token:
            with contextlib.suppress(OSError):
                os.utime(self._mark_path(code), (0, 0))

    def _mark_infos(self) -> dict[str, dict]:
        """{DOSSIER_n: mark info} of every mark on disk, read at each call (v0.4.3 fix: another process may write
        or remove a mark at any time; a few small files, read once per answer)."""
        found: dict[str, dict] = {}
        try:
            names = os.listdir(self._home_dir() / READING_DIR_NAME)
        except OSError:
            names = []
        for name in names:
            if name.endswith(_READING_SUFFIX) and DOSSIER_RE.match(name[: -len(_READING_SUFFIX)]):
                code = name[: -len(_READING_SUFFIX)]
                info = self._mark_info(code)
                if info is not None:
                    found[code] = info
        return found

    def _mark_table(self) -> dict[str, str]:
        """{DOSSIER_n: dossier id} of every mark on disk."""
        return {code: info["uuid"] for code, info in self._mark_infos().items()}

    def _clear_mark(self, code: str, uuid: Optional[str] = None) -> None:
        if uuid is not None and self._disk_mark(code) not in (None, uuid):
            return  # a newer reading of a dossier with the same code
        try:
            self._mark_path(code).unlink()
        except OSError:
            pass

    def pending_readings(self) -> dict[str, str]:
        """{DOSSIER_n: dossier id} of every dossier whose reading is not finished (marks on disk). No vault is
        opened."""
        return self._mark_table()

    def reading_state(self, dossier_code: str, _info: Any = False) -> Optional[dict]:
        """None when the dossier is ready to read (no reading going on, no mark). Else {code, state, total, done,
        masked, page, pages}: ``state`` "waiting", "reading", "masking", "failed" or "pending" (a mark on disk and
        no reading in this service). ``elsewhere``: another service (conversation) reads it now. ``gave_up``: Shield
        stops waiting for it (failed too often, or stuck). Codes and counts only; no service lock; one small file."""
        code = str(dossier_code or "").strip().upper()
        with self._reads_lock:
            job = self._reads.get(code)
        if job is not None and not job.finished.is_set() and job.state not in ("ready", "gone"):
            view = job.view()
            if job.stuck():
                view.update(state="failed", gave_up=True, stuck=True)
            return view
        # v0.4.3 fix: always the disk (another process may read this dossier); ``_info``: read just before
        info = self._mark_info(code) if _info is False else _info
        if info is None:
            return None
        if self._foreign(info):
            return {"code": code, "state": "pending", "total": None, "done": 0, "masked": 0, "page": 0, "pages": 0,
                    "elsewhere": True}
        if job is not None and job.state == "failed":
            view = job.view()
            view["gave_up"] = self._failures[code] >= MAX_READING_FAILURES
            return view
        return {"code": code, "state": "pending", "total": None, "done": 0, "masked": 0, "page": 0, "pages": 0}

    def readings(self) -> list[dict]:
        """Every reading known in this process and every mark on disk: the progress of each dossier, or its
        counts once ready ({code, state: "ready", summary}). Codes and counts only; no service lock."""
        with self._reads_lock:
            jobs = dict(self._reads)
        infos = self._mark_infos()  # one look at the disk for the whole answer
        out: dict[str, dict] = {}
        for code, job in jobs.items():
            view = self.reading_state(code, infos.get(code)) if job.state != "gone" else None
            if view is not None:
                out[code] = view
            elif job.state == "ready" and job.summary is not None:
                out[code] = {"code": code, "state": "ready", "summary": dict(job.summary)}
        for code, info in infos.items():
            if code not in out:
                view = self.reading_state(code, info)
                if view is not None:
                    out[code] = view
        return [out[c] for c in sorted(out, key=lambda c: int(DOSSIER_RE.match(c).group(1)))]

    def reading_summary(self, dossier_code: str) -> Optional[dict]:
        """The counts of the last finished reading of the dossier in this process, or None."""
        with self._reads_lock:
            job = self._reads.get(str(dossier_code or "").strip().upper())
        if job is None or job.state != "ready" or job.summary is None:
            return None
        return dict(job.summary)

    def reading_kept(self, dossier_code: str) -> bool:
        """v0.4.4: True when the last reading of the dossier in this process ended on its kept set (the cheaper
        re-check): nothing was read again, the dossier is simply ready (A5)."""
        with self._reads_lock:
            job = self._reads.get(str(dossier_code or "").strip().upper())
        return job is not None and job.state == "ready" and job.kept

    def ready_counts(self, dossier_code: str) -> dict:
        """{documents, unreadable, too_long, needs_ocr, counts} of a ready dossier that another service read (no
        summary here): the documents found and the elements replaced. Codes and counts only."""
        d = self._dossier(dossier_code)
        if self.reading_state(d.code) is None:  # v0.4.4: the counts kept with the dossier (another conversation read it)
            try:
                self._kset_ready(d.code)
            except Exception:  # noqa: BLE001 - the counts below
                pass
            summary = self.reading_summary(d.code)
            if summary is not None:
                return summary
        with self._lock:
            refs = d.documents()
            too_long = sum(1 for ref in refs
                           if (self._not_read.get((d.code, ref.id)) or (0, 0, ""))[2] in ("too_long", "too_large"))
            return {"documents": len(refs), "unreadable": 0, "too_long": too_long, "needs_ocr": 0,
                    "counts": _kind_counts(d)}

    def wait_reading(self, dossier_code: str, timeout: float) -> bool:
        """Wait up to ``timeout`` seconds for the reading of the dossier (in this service, or in another one: its
        mark). True when no reading goes on (it ended, ready or not). False on a time-out, or at once when the
        reading is stuck."""
        code = str(dossier_code or "").strip().upper()
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            with self._reads_lock:
                job = self._reads.get(code)
            if job is not None and not job.finished.is_set():
                if job.stuck():
                    return False
                if job.finished.wait(max(0.0, min(1.0, deadline - time.monotonic()))):
                    continue
            else:
                if not self._foreign(self._mark_info(code)):
                    return True  # no reading goes on (ready, failed here, or free to resume): nothing to wait for
                time.sleep(max(0.0, min(0.25, deadline - time.monotonic())))
            if time.monotonic() >= deadline:
                return False

    def _require_ready(self, dossier_code: Any) -> None:
        """Model-facing dossier calls: a dossier whose reading is not finished gives no content, no name and no
        list (a name found in a late document must also be hidden in the documents read before). A free mark with
        no reading in this service (a stop in the middle) starts the reading again; a mark that another service
        keeps fresh starts nothing (v0.4.3 fix: the disk is checked at every call)."""
        code = _dossier_code(dossier_code)
        state = self.reading_state(code)
        if state is None:
            self._warm(code)
            return
        if state["state"] in ("pending", "failed") and not state.get("elsewhere") and not state.get("stuck"):
            self.resume_readings(code)
        exc = ShieldError(f"{code} n'est pas encore prêt : Shield lit ses documents (dossier_reading).",
                          "dossier_reading")
        exc.dossier = code  # type: ignore[attr-defined]
        raise exc

    def resume_readings(self, only: Optional[str] = None) -> list[str]:
        """Start again the readings that a stop left unfinished: a mark on disk, no reading in this service, and the
        mark is free (ours, untouched for MARK_STALE_SECONDS, or its process is gone). Returns the codes started.
        Never opens a vault itself (the reading does, in its own thread)."""
        started = []
        infos = self._mark_infos() if only is None else {only: self._mark_info(only)}
        for code, info in infos.items():
            if info is None:
                continue
            with self._reads_lock:
                job = self._reads.get(code)
                running = job is not None and not job.finished.is_set()
                gave_up = self._failures[code] >= MAX_READING_FAILURES
            if running or gave_up or self._foreign(info):
                continue
            job = self._launch(code, info["uuid"])
            if job is not None:
                started.append(code)
        return started

    @_guarded("protect_folder")
    def protect_folder(self, folder: os.PathLike | str) -> dict:
        """Plugin: register the folder (or find its dossier) and read it in the background. Returns at once:
        {code, existing, already_reading}. The reading is the one of :meth:`add_dossier` (documents, title
        parties, sub-folder parties); :meth:`reading_state` gives its progress.

        v0.4.3 (fix): the mark of a new dossier is written in the same step as its registry row (a dossier is never
        registered without its mark). A dossier that another conversation reads now is not read a second time."""
        # The registry has its own lock (no service lock: a background masking never delays this answer).
        existed = self.registry.is_registered(folder)
        marked: dict[str, str] = {}

        def mark(code: str, uuid: str) -> None:
            self._write_mark(code, uuid)  # raises after a few tries: nothing is registered
            marked[code] = uuid

        d = self.registry.add_dossier(folder, on_create=mark)
        code, uuid = d.code, d.uuid
        already = self._running(code)
        if not already and code not in marked and self._foreign(self._mark_info(code)):
            return {"code": code, "existing": existed, "already_reading": True}
        if not already and code not in marked and self._kset_ready(code):
            # v0.4.4 (A): same files, same engine, same vault: the dossier is ready without reading any file
            return {"code": code, "existing": existed, "already_reading": False, "kept": True}
        self._launch(code, uuid, premarked=code in marked)
        return {"code": code, "existing": existed, "already_reading": already}

    @_guarded("start_reading")
    def start_reading(self, dossier_code: str) -> dict:
        """Plugin: read a registered dossier again in the background (same codes). {code, already_reading}."""
        d = self._dossier(dossier_code)  # the registry's own lock only
        code, uuid = d.code, d.uuid
        already = self._running(code)
        if not already and self._foreign(self._mark_info(code)):
            return {"code": code, "existing": True, "already_reading": True}
        if not already and self._kset_ready(code):  # v0.4.4 (A)
            return {"code": code, "existing": True, "already_reading": False, "kept": True}
        self._launch(code, uuid)
        return {"code": code, "existing": True, "already_reading": already}

    def _running(self, code: str) -> bool:
        with self._reads_lock:
            job = self._reads.get(code)
            return job is not None and not job.finished.is_set()

    def _warm(self, code: str) -> None:
        """v0.4.4 (A): the first dossier call of this process on a ready dossier. The dossier is ready WITHOUT reading
        any file when its kept set is usable (:meth:`_kset_load`). Otherwise the WHOLE dossier is read again in the
        background, exactly as a first reading (progress answers, mark), and the kept set is replaced at its end; the
        call waits up to KEPT_WAIT_SECONDS for it. Never a mix of kept and fresh texts: a name found in a changed
        document must also be hidden in all the others. Once per dossier and process."""
        with self._reads_lock:
            if code in self._warmed:
                return
        self._open_dossiers()  # fail closed at once (vault_error) when a vault cannot be opened, as every answer does
        with self._kset_lock:
            with self._reads_lock:
                if code in self._warmed:
                    return
            if self._kset_ready(code):
                return
            if self._foreign(self._mark_info(code)):
                job = None  # another conversation started its reading meanwhile
            else:
                try:
                    d = self.registry.get(code)
                except Exception:  # noqa: BLE001 - the call goes on and gives its own answer
                    return
                job = self._launch(code, d.uuid)
        if job is not None and job.finished.wait(KEPT_WAIT_SECONDS) and job.state == "ready":
            return
        exc = ShieldError(f"{code} n'est pas encore prêt : Shield lit ses documents (dossier_reading).",
                          "dossier_reading")
        exc.dossier = code  # type: ignore[attr-defined]
        raise exc

    # ---------------------------------------------------------------- kept sets (v0.4.4, A)
    def _kset_store(self) -> _kset.KeptStore:
        store = self._kset_store_obj
        if store is None:
            store = self._kset_store_obj = _kset.KeptStore(self._home_dir())
        return store

    def _kset_ready(self, code: str) -> bool:
        """Load the kept set of a dossier when it is usable (never raises). True: the dossier is ready."""
        with self._kset_lock:
            if self._mark_info(code) is not None or self._running(code):
                return False  # a reading is not finished: the mark wins
            try:
                return self._kset_load(code)
            except Exception as exc:  # noqa: BLE001 - a kept set is an aid: a full reading instead
                log.warning("RAGbase Shield: %s: kept set not loaded (%s)", code, type(exc).__name__)
                return False

    def _kset_files(self, d: Dossier, refs: list) -> Optional[list]:
        """(a) The files of the folder with their (mtime_ns, size), by DOC_n and keyed hash of the path (no name).
        None when a file or a sub-folder could not be opened (such a dossier is never kept)."""
        if d.skipped.get("long") or d.skipped.get("other"):
            return None
        rows = []
        for ref in refs:
            st = ref.path.stat()
            rows.append([ref.id, d.path_mac(ref.rel), ref.kind, st.st_mtime_ns, st.st_size])
        return sorted(rows)

    def _kset_load(self, code: str) -> bool:
        """A2: the dossier is ready without reading any file if and only if (a) the folder holds the same files with
        the same (mtime_ns, size), (b) the same engine version and rules fingerprint, (c) the same lexicon
        fingerprint of the dossier and the same values in every dossier (the last check), (d) the manifest and every
        entry decrypt and match their hashes, (e) the last reading had no failure that can pass (such a set is never
        written). Any point fails: False (the caller starts a reading of the dossier). When ONLY the values of the
        other dossiers changed, that reading first runs the cheaper re-check (:meth:`_kset_recheck`)."""
        snap = self._kset_decode(code)
        if snap is None or snap["recheck"]:
            return False
        self._kset_install(code, snap)
        return True

    def _kset_decode(self, code: str) -> Optional[dict]:
        """The kept set of a dossier, checked (A2 a to e) and decoded, or None. ``recheck`` is True when every point
        holds except the values of the OTHER dossiers (guard fingerprint): the set may then be used only after the
        last check gave the same texts again (:meth:`_kset_recheck`)."""
        d = self.registry.get(code)
        got = self._kset_store().load(d)
        if got is None:
            return None
        man, payloads = got
        if man.get("dossier") != d.uuid or man.get("code") != d.code:
            return None
        if man.get("engine") != __version__ or man.get("rules") != _kset.rules_fingerprint():
            return None
        all_refs = d.documents(include_unsupported=True)
        if self._kset_files(d, all_refs) != man.get("files"):
            return None
        if man.get("lexicon") != _kset.lexicon_fingerprint(d):
            return None
        recheck = man.get("guard") != _kset.guard_fingerprint(self._open_dossiers())
        refs = {ref.id: ref for ref in all_refs}
        rows = {r[0]: r for r in man["files"]}
        states = man.get("docs") or {}
        rev = self._version(d)
        epoch = None if recheck else self._guard_epoch()  # a re-checked entry is guarded again before any answer
        entries: dict[str, _CachedDoc] = {}
        not_read: dict[tuple[str, str], tuple] = {}
        for ref in d.documents():
            st = states.get(ref.id)
            if not isinstance(st, dict):
                return None
            if st.get("state") == "read":
                payload = payloads.get(ref.id)
                if payload is None:
                    return None
                meta, texts = _kset.unpack(payload)
                if ref.id not in rows or [meta.get("mtime_ns"), meta.get("size")] != rows[ref.id][3:5]:
                    return None
                masked = texts["masked"]
                a0 = _al_load(meta.get("a0"))
                steps = meta.get("steps")
                chain = _Chain([a0] + [_al_load(x) for x in steps]) if a0 is not None and steps is not None else None
                entries[ref.id] = _CachedDoc(
                    mtime_ns=meta["mtime_ns"], size=meta["size"], version=rev if meta.get("current") else ("", -1),
                    raw=None, masked=masked, alignment=chain, needs_ocr=bool(meta.get("needs_ocr")),
                    pages=meta.get("pages"), warnings=tuple(meta.get("warnings") or ()),
                    masked0=texts.get("masked0", masked), align0=a0, epoch=epoch, shadow=texts.get("shadow"))
            elif st.get("state") in ("too_large", "unreadable"):
                row = rows[ref.id]
                cause = (row[3], row[4], st["state"]) + ((st.get("code") or "unreadable",)
                                                         if st["state"] == "unreadable" else ())
                not_read[(d.code, ref.id)] = cause
            else:
                return None
        return {"d": d, "man": man, "payloads": payloads, "refs": refs, "rows": rows, "rev": rev, "epoch": epoch,
                "entries": entries, "not_read": not_read, "recheck": recheck,
                "labels": {k: list(v) for k, v in (man.get("labels") or {}).items()}}

    def _kset_install(self, code: str, snap: dict, job: Optional[_Reading] = None) -> None:
        """Make a decoded kept set the state of the dossier in this process. With ``job`` (the re-check inside a
        reading): that reading takes the summary; its end removes the mark and says "ready"."""
        man, refs, rows = snap["man"], snap["refs"], snap["rows"]
        with self._lock:
            self._invalidate(code)
            self._kset_docs[code] = snap["entries"]
            self._kset_labels[code] = (snap["rev"], snap["epoch"], snap["labels"])
            for k in [k for k in self._not_read if k[0] == code]:
                del self._not_read[k]
            self._not_read.update(snap["not_read"])
            for doc_id, n in (man.get("pages") or {}).items():
                ref = refs.get(doc_id)
                if ref is not None and doc_id in rows:
                    self._pages[(str(ref.path), rows[doc_id][3], rows[doc_id][4])] = n
        summary = dict(man.get("summary") or {})
        if isinstance(summary.get("counts"), list):
            summary["counts"] = {k: v for k, v in summary["counts"]}
        if job is not None:
            job.summary = summary
            job.kept = True
            return
        job = _Reading(code, snap["d"].uuid)
        job.state = "ready"
        job.summary = summary
        job.finished.set()
        with self._reads_lock:
            self._reads[code] = job
            self._warmed.add(code)

    def _kset_recheck(self, d: Dossier, job: _Reading) -> bool:
        """The cheaper path when only ANOTHER dossier changed (a new client folder protected, a name added there).
        The masked texts of this dossier do not depend on the other dossiers; only the last check (their values in
        each text) and the masked file names do. So the last check runs again on the kept masker output of every
        document, and the file names are masked again. When every text, every edit of the last check and every name
        come out exactly as kept, and this dossier did not change meanwhile, the kept set is the state a full
        reading would give: it is used, and written again with the new fingerprints. Any difference: False, and the
        WHOLE dossier is read again (all or nothing). No file of the lawyer is read here."""
        try:
            snap = self._kset_decode(d.code)
        except Exception as exc:  # noqa: BLE001 - a kept set is an aid: a full reading instead
            log.info("RAGbase Shield: %s: kept set not re-checked (%s)", d.code, type(exc).__name__)
            return False
        if snap is None or not snap["recheck"] or snap["d"].uuid != d.uuid:
            return False
        with self._lock:
            rev0, epoch0 = self._version(d), self._safe_epoch()
        if epoch0 is None or rev0 != snap["rev"]:
            return False
        job.state = "masking"
        job.done = job.total or 0
        for entry in snap["entries"].values():
            if self._stop.is_set():
                raise _Stopped()
            job.step(_mask_budget(len(entry.masked0)))
            guarded, steps, _n = self._guard_full(entry.masked0, d)
            chain = _Chain([entry.align0] + steps) if entry.align0 is not None and steps is not None else None
            if guarded != entry.masked or _steps_sig(chain) != _steps_sig(entry.alignment):
                return False
            job.masked += 1
        job.step()
        with self._lock:
            if self._version(d) != rev0 or self._safe_epoch() != epoch0:
                return False  # the last check gave this dossier a new code, or a dossier changed meanwhile
            all_refs = d.documents(include_unsupported=True)
            labels = {ref.id: self._ref_labels(d, ref) for ref in all_refs}
            if labels != snap["labels"] or self._version(d) != rev0:
                return False
            if self._kset_files(d, all_refs) != snap["man"].get("files"):
                return False  # a file changed during the re-check
            for entry in snap["entries"].values():
                entry.epoch = epoch0
            snap["epoch"] = epoch0
            self._kset_install(d.code, snap, job)
            man = {k: v for k, v in snap["man"].items() if k not in ("entries", "gen", "format")}
            man.update(lexicon=_kset.lexicon_fingerprint(d), guard=_kset.guard_fingerprint(self._open_dossiers()))
        self._kset_store().write(d, man, snap["payloads"])  # never fails the reading
        return True

    def _kset_finish(self, d: Dossier, job: _Reading, refs: list, raws: dict, labels: dict, summary: dict) -> None:
        """End of a complete reading (v0.4.4, A1): the entries take their kept form in memory (the text read from the
        file is dropped; the search runs on its shadow), so this process answers exactly as the next one will; then
        the kept set is written, or deleted when the reading had a failure that can pass (A2 e)."""
        code = d.code
        keepable = not any(raw in (documents.READ_TOO_LONG, documents.READ_LOCKED, documents.READ_DENIED,
                                   documents.READ_IO) for raw in raws.values())
        all_refs = d.documents(include_unsupported=True)
        files = self._kset_files(d, all_refs)
        keepable = keepable and files is not None
        stats = {r[0]: (r[3], r[4]) for r in files or []}
        with self._lock:
            epoch = self._safe_epoch()
            rev = self._version(d)
            found: dict[str, _CachedDoc] = {}
            for ref in refs:
                key = (code, ref.id)
                entry = self._cached(key)
                if entry is not None and stats.get(ref.id, (entry.mtime_ns, entry.size)) == (entry.mtime_ns, entry.size):
                    if epoch is not None and entry.epoch != epoch:
                        try:
                            self._doc_reguard(d, entry, epoch)
                        except _KeptStale:
                            self._drop_cached(key, entry)
                            keepable = False
                            continue
                    found[ref.id] = entry
            self._kset_labels[code] = (rev, epoch, labels)
        kept: dict[str, _CachedDoc] = {}
        for doc_id, e in found.items():
            if e.raw is None:
                kept[doc_id] = e
                continue
            kept[doc_id] = _CachedDoc(mtime_ns=e.mtime_ns, size=e.size, version=e.version, raw=None, masked=e.masked,
                                      alignment=e.alignment, needs_ocr=e.needs_ocr, pages=e.pages, warnings=e.warnings,
                                      masked0=e.masked0, align0=e.align0, epoch=e.epoch,
                                      shadow=_search_shadow(e.raw, e.alignment), _folded_masked=e._folded_masked)
        with self._lock:
            for doc_id, e in found.items():
                if self._docs.get((code, doc_id)) is e:
                    del self._docs[(code, doc_id)]
            self._kset_docs[code] = kept
        states: dict[str, dict] = {}
        payloads: dict[str, bytes] = {}
        for ref in refs:
            e = kept.get(ref.id)
            bad = self._not_read.get((code, ref.id))
            if e is not None:
                states[ref.id] = {"state": "read"}
                steps = [_al_dump(st) for st in e.alignment.steps[1:]] if e.alignment is not None else None
                meta = {"mtime_ns": e.mtime_ns, "size": e.size, "needs_ocr": e.needs_ocr, "pages": e.pages,
                        "warnings": list(e.warnings), "a0": _al_dump(e.align0), "steps": steps,
                        "current": e.version == rev}
                payloads[ref.id] = _kset.pack(meta, {"masked": e.masked,
                                                     "masked0": None if e.masked0 == e.masked else e.masked0,
                                                     "shadow": e.shadow})
            elif bad is not None and bad[2] in ("too_large", "unreadable") and stats.get(ref.id) == bad[:2]:
                states[ref.id] = {"state": bad[2], "code": bad[3] if len(bad) > 3 else None}
            else:
                keepable = False  # not read, for a cause that was not kept: read again at the next start
        store = self._kset_store()
        if not keepable:
            store.delete(code)
            return
        pages = {}
        for ref in all_refs:
            if ref.kind == "pdf":
                e = kept.get(ref.id)
                pages[ref.id] = e.pages if e is not None and e.pages is not None else \
                    self._pdf_pages(ref.path, stats[ref.id][0], stats[ref.id][1])
        manifest = {"dossier": d.uuid, "code": code, "engine": __version__, "rules": _kset.rules_fingerprint(),
                    "lexicon": _kset.lexicon_fingerprint(d), "guard": _kset.guard_fingerprint(self._open_dossiers()),
                    "files": files, "docs": states, "labels": labels, "pages": pages,
                    # the counts keep their order (the status lists equal counts in this order); JSON sorts keys
                    "summary": {k: (list(v.items()) if k == "counts" else v) for k, v in summary.items()
                                if k not in ("seconds", "read_seconds")}}
        store.write(d, manifest, payloads)

    @_guarded("clean_kept")
    def clean_kept(self) -> list[str]:
        """v0.4.4 (A3): at a start, delete the kept sets of dossiers that are no longer in the vault (and the files
        of a set whose writer stopped). Returns the codes deleted.

        Review fix: also the set of a dossier whose folder is gone (the firm deleted the client folder at the end of a
        matter): Shield keeps no copy of the content of a folder that no longer exists. A folder on a drive that is
        not plugged in only loses its set: the dossier is read again when the drive comes back."""
        store = self._kset_store()
        registered = self.registry.folders()
        gone = store.clean(registered)
        for code, folder in registered.items():
            if code in gone or not store.folder(code).is_dir():
                continue
            try:
                present = Path(folder).is_dir()
            except OSError:
                present = False
            if not present:
                store.delete(code)
                gone.append(code)
        return gone

    def kept_size(self, dossier_code: str) -> int:
        """Bytes on disk of the kept set of a dossier (measures)."""
        return self._kset_store().size(_dossier_code(dossier_code))

    def skipped_paths(self, dossier_code: str) -> dict[str, int]:
        """v0.4.4 (F2): documents or sub-folders of the dossier folder that could not be opened at the last scan:
        {"long": paths too long for Windows, "other": another cause}. Counts only."""
        d = self._dossier(dossier_code)
        return dict(getattr(d, "skipped", None) or {"long": 0, "other": 0})

    def _launch(self, code: str, uuid: str, mark: bool = True, premarked: bool = False) -> _Reading:
        """Start the reading of a dossier, or return the one going on. The job is known in this service BEFORE the
        mark is written (v0.4.3 fix: the gate holds even when the disk write fails), and the mark is written before
        the thread starts: from now until the end, the dossier is not readable (``mark=False``: a second reading of
        a ready dossier, to fill the cache; ``premarked``: the registry wrote the mark with the new dossier)."""
        with self._reads_lock:
            job = self._reads.get(code)
            if job is not None and not job.finished.is_set():
                return job
            self._warmed.add(code)
            job = _Reading(code, uuid)
            self._reads[code] = job
            if mark:
                if not premarked:
                    try:
                        self._write_mark(code, uuid)
                    except OSError as exc:  # the job above blocks this service; a crash would only lose the resume
                        log.warning("RAGbase Shield: %s: reading mark not written (%s)", code, type(exc).__name__)
                job.marked = True
            job.thread = threading.Thread(target=self._read_job, args=(job,), name=f"shield-reading-{code}",
                                          daemon=True)
            job.thread.start()
            if job.marked:
                threading.Thread(target=self._heartbeat, args=(job,), name=f"shield-mark-{code}", daemon=True).start()
            return job

    def _heartbeat(self, job: _Reading) -> None:
        """Touch the mark of a running reading every MARK_TOUCH_SECONDS, so another service sees that it goes on. A
        stuck reading is no longer touched: its mark becomes free for another start."""
        while not job.finished.wait(MARK_TOUCH_SECONDS):
            if self._stop.is_set():
                return
            if not job.stuck():
                self._touch_mark(job.code)

    def _gate_enter(self, job: _Reading) -> None:
        """One reading at a time in this service. A reading that is stuck (past its budget and margin) no longer
        holds back the next one."""
        with self._gate:
            while True:
                owner = self._gate_owner
                if owner is None or owner.finished.is_set() or owner.stuck():
                    break
                if self._stop.is_set():
                    raise _Stopped()
                self._gate.wait(1.0)
            self._gate_owner = job

    def _gate_leave(self, job: _Reading) -> None:
        with self._gate:
            if self._gate_owner is job:
                self._gate_owner = None
            self._gate.notify_all()

    def _read_raw(self, code: str, ref: Any, job: _Reading):
        """Text of one document for a background reading: (mtime_ns, size, ReadResult), or a short code when it
        cannot be read (``read_too_long``, ``read_too_large``, ``unreadable``). The text of an unchanged cached
        document is reused."""
        def progress(done: int, total: int) -> None:
            job.page, job.pages = done, total
            if done == 0 and total and ref.kind == "pdf":  # v0.4.4 (B2): a long PDF has more time
                job.budget = documents.document_budget(total)
            if self._stop.is_set():
                raise _Stopped()

        try:
            st = ref.path.stat()
            cached = self._docs.get((code, ref.id))
            if cached is not None and cached.raw is not None and cached.mtime_ns == st.st_mtime_ns \
                    and cached.size == st.st_size:
                return (st.st_mtime_ns, st.st_size,
                        documents.ReadResult(cached.raw, cached.pages, cached.needs_ocr, cached.warnings))
            # v0.4.4 (F4): a file that was locked or refused is tried again (the lawyer closed it, then protects the
            # folder again); a too long or unreadable file is not read again while it is the same (as in 0.4.3)
            if (self._not_read.get((code, ref.id)) or (0, 0, ""))[2] in ("locked", "denied", "io"):
                self._not_read.pop((code, ref.id), None)
            raw = self._read_file((code, ref.id), ref.path, st, progress=progress)
            return (st.st_mtime_ns, st.st_size, raw)
        except ShieldError as exc:
            if exc.kind == "document_too_long":
                return documents.READ_TOO_LONG
            if exc.kind == "document_too_large":
                return documents.READ_TOO_LARGE
            if exc.kind == "document_locked":  # v0.4.4 (F4)
                return documents.READ_LOCKED
            if exc.kind == "document_unavailable":  # v0.4.4 (review fix): can pass
                return documents.READ_IO
            return "unreadable"
        except documents.ReadDenied:
            return documents.READ_DENIED
        except OSError:  # v0.4.4 (review fix): the file could not be looked at now (disk, network): can pass
            if (code, ref.id) not in self._not_read:
                self._not_read[(code, ref.id)] = (0, ref.size, "io")
            return documents.READ_IO
        except documents.DocumentError:
            return "unreadable"
        finally:
            job.budgets[ref.id] = job.budget

    def _read_job(self, job: _Reading) -> None:
        """The background reading, in the order of the 0.4.2 one-step reading (same codes, v0.4.3 fix): (1) the text
        of every document, page by page, with no service lock; (2) the directory, then the names replaced by codes in
        DOC order; (3) the parties of the title, then of the sub-folders (a sub-folder party means a second full
        masking, as in 0.4.2); (4) a last pass in DOC order masks again the documents that are out of date (what the
        first reads did in 0.4.2); (5) the mark is removed. The masking of each document runs outside the service lock
        (own rules maskers); the lock is taken only to store it."""
        entered = False
        try:
            self._gate_enter(job)
            entered = True
            if self._stop.is_set():
                raise _Stopped()
            self._tls.rules = {}
            job.state = "reading"
            job.started = time.monotonic()
            job.step()
            with self._lock:
                try:
                    d = self._dossier(job.code)
                except UnknownDossier:
                    d = None
                if d is None or d.uuid != job.uuid:
                    job.state = "gone"
                    self._clear_mark(job.code, job.uuid)
                    return
                refs = d.documents()
            job.total = len(refs)
            # v0.4.4: only the other dossiers changed since the kept set: the cheaper re-check, no file read
            try:
                kept_ok = self._kset_recheck(d, job)
            except _Stopped:
                raise
            except Exception as exc:  # noqa: BLE001 - a kept set is an aid: a full reading instead
                log.info("RAGbase Shield: %s: kept set not re-checked (%s)", d.code, type(exc).__name__)
                kept_ok = False
            if kept_ok:
                self._clear_mark(job.code, job.uuid)  # first the mark, then "ready"
                job.state = "ready"
                return
            job.state = "reading"
            job.done = job.masked = 0
            job.step()
            raws: dict[str, Any] = {}
            for ref in refs:
                if self._stop.is_set():
                    raise _Stopped()
                job.page = job.pages = 0
                job.step()
                raws[ref.id] = self._read_raw(job.code, ref, job)
                job.done += 1
            job.page = job.pages = 0
            job.state = "masking"
            job.step()
            read_seconds = time.monotonic() - job.started
            with self._lock:
                persons = self._build_directory(d)
                if self._dossier_replacement(d) == "synthetic":
                    self._register_pass(d, raws)
            summary = self._mask_all(d, refs, raws, job, locked=False)
            job.step()
            with self._lock:
                added = self._register_title_parties(d)
                folder_added = self._register_folder_parties(d)
                added += folder_added
            if folder_added:  # 0.4.2: register_folder_parties then scan_dossier (directory and every document again)
                job.step()
                with self._lock:
                    persons = self._build_directory(d)
                    if self._dossier_replacement(d) == "synthetic":
                        self._register_pass(d, raws)
                summary = self._mask_all(d, refs, raws, job, locked=False, count=False)
            # v0.4.4: the masked file names, made now (what shield_list_documents shows): a name that only a file name
            # holds is then known before the last pass, so it is hidden in the documents too
            job.step()
            with self._lock:
                labels = {ref.id: self._ref_labels(d, ref) for ref in d.documents(include_unsupported=True)}
                labels_rev = self._version(d)
            # Last pass: only the documents masked before the last new name are masked again (the text read before
            # is used again); the others come back from the cache at once.
            summary = self._mask_all(d, refs, raws, job, locked=False, count=False)
            job.step()
            with self._lock:
                if self._version(d) != labels_rev:  # the last pass found a new name: the names once more
                    labels = {ref.id: self._ref_labels(d, ref) for ref in d.documents(include_unsupported=True)}
                skipped = dict(d.skipped)  # v0.4.4 (F2): files and sub-folders that could not be opened
                d.journal("scan", documents=summary["documents"], entities=len(d.all()))
                summary.update(code=d.code, entities=len(d.all()), counts=_kind_counts(d), directory=persons,
                               parties=added)
            # v0.4.4 (F2): a path too long for Windows is said apart; another failure counts with the unreadable ones
            summary["skipped_long"] = skipped.get("long", 0)
            summary["documents"] += skipped.get("other", 0)
            summary["unreadable"] += skipped.get("other", 0)
            summary["read_seconds"] = round(read_seconds, 1)
            summary["seconds"] = round(time.monotonic() - job.started, 1)
            job.summary = summary
            try:  # v0.4.4 (A1): the kept set, written before the mark is removed; never fails the reading
                self._kset_finish(d, job, refs, raws, labels, summary)
            except Exception as exc:  # noqa: BLE001
                log.warning("RAGbase Shield: %s: kept set not written (%s)", d.code, type(exc).__name__)
                with contextlib.suppress(Exception):
                    self._kset_store().delete(d.code)
            self._clear_mark(job.code, job.uuid)  # first the mark, then "ready": never ready with a mark left
            job.state = "ready"
        except _Stopped:
            job.state = "failed"
            job.error = "stopped"
        except BaseException as exc:  # noqa: BLE001 - the mark stays: the next call starts the reading again
            with self._reads_lock:
                self._failures[job.code] += 1
            job.state = "failed"
            job.error = getattr(exc, "kind", None) or type(exc).__name__
            _log_failure("background reading", exc)
        finally:
            self._tls.rules = None
            job.finished.set()
            if entered:
                self._gate_leave(job)

    @_guarded("remove_dossier")
    def remove_dossier(self, dossier_code: str) -> None:
        with self._lock:
            code = _dossier_code(dossier_code)
            self.registry.remove(code)
            self._drop_maskers(code)
            self._invalidate(code)
            self._clear_mark(code)
            self._kset_store().delete(code)  # v0.4.4 (A3)

    @_guarded("entities")
    def entities(self, dossier_code: str) -> list[dict]:
        """[{code, kind, value, kept_clear, source, synthetic}] with REAL values: local display only.
        ``synthetic`` is the fake value the dossier gave in synthetic mode (None until one was needed)."""
        with self._lock:
            d = self._dossier(dossier_code)
            return [{"code": e.code, "kind": e.kind, "value": e.value, "kept_clear": d.is_kept_clear(e.norm),
                     "source": d.entity_source(e.code), "synthetic": d.synthetic(e.code)} for e in d.all()]

    @_guarded("register")
    def register(self, dossier_code: str, kind: str, value: str) -> dict:
        """The user adds a name or identifier. Returns {code, kind} (and value, for the local caller)."""
        with self._lock:
            d = self._dossier(dossier_code)
            kind = _kind_arg(kind)
            value = " ".join(str(value or "").split())
            if not value:
                raise ShieldError("Valeur vide (empty value).", "invalid_argument")
            ent = self._same_entity(d, kind, value)
            if ent is None:
                ent = d.register(kind, value)
                self._invalidate(d.code)
            d.journal("register", code=ent.code)
            return {"code": ent.code, "kind": ent.kind, "value": ent.value}

    def _same_entity(self, d: Dossier, kind: str, value: str) -> Optional[Entity]:
        """The existing entity with the same engine key ("Jean-Pierre DUPONT" = "Jean-Pierre Dupont",
        "ACME SAS" = "ACME"), so that registering a known name never creates a second code for it."""
        index = self._index(d)
        codes: set[str] = set()
        for key in {_safe_norm(kind, value), normalize(value)}:
            if key:
                codes |= index["key"].get((kind, key), set())
        return d.get_by_code(next(iter(codes))) if len(codes) == 1 else None

    @_guarded("keep_clear")
    def keep_clear(self, dossier_code: str, target: str, value: bool = True) -> dict:
        """Mark an entity code or a term "garder en clair" (``value=False`` masks it again)."""
        with self._lock:
            d = self._dossier(dossier_code)
            target = str(target or "").strip()
            if not target:
                raise ShieldError("Code ou terme vide (empty code or term).", "invalid_argument")
            if CODE_RE.fullmatch(target) and d.get_by_code(target) is None:
                raise ShieldError(f"{target} : code inconnu dans {d.code} (unknown code).", "invalid_argument")
            d.keep_clear(target, bool(value))
            self._invalidate(d.code)
            d.journal("keep_clear", code=target if CODE_RE.fullmatch(target) else None, value=bool(value))
            return {"ok": True}

    @_guarded("mask_text")
    def mask_text(self, text: str, dossier_code: Optional[str] = None, include_values: bool = False,
                  replacement: Optional[str] = None) -> dict:
        """Mask a text for the Desk, the CLI or the browser extension.

        {masked, found: [{code, kind[, value][, synthetic]}], counts: {KIND: mentions}, mode}. Without a dossier a
        temporary in-memory store is used. ``include_values`` adds the real value of each code (local page only).
        ``replacement`` (v0.2): "codes" or "synthetic" (realistic pseudonyms, kept in the dossier so that a
        later restore maps them back); None takes the dossier setting, else ``config.replacement()``.
        """
        with self._lock:
            text = _nfc(str(text or ""))
            if len(text) > MAX_MASK_CHARS:
                raise ShieldError("Texte trop long (text too long).", "invalid_argument")
            replacement = _replacement_arg(replacement)
            if dossier_code:
                d = self._dossier(dossier_code)
                m = self._masker(d, replacement)
                store: Any = d
            else:
                d = None
                store = MemoryStore(locale=config.locale())
                m = Masker(store, locale=config.locale(), replacement=replacement or config.replacement(),
                           language=config.language(), ner_model=config.ner_model(),
                           own_company_vendor=config.own_company_vendor())
            res = m.mask(text)
            # Always the canary of every dossier: without a dossier, a fresh masker cannot know the registered
            # names (a bare surname, a company without its legal form): they become [PERSONNE], [SOCIETE]...
            masked, canary = self._guard_report(res.text, d)
            found: list[dict] = []
            seen: set[str] = set()
            if res.mode == "synthetic":
                for code, rep in zip(res.codes, res.replacements):
                    if code in seen:
                        continue
                    ent = store.get_by_code(code)
                    if ent is None:
                        continue
                    seen.add(code)
                    item = {"code": code, "kind": ent.kind, "synthetic": rep}
                    if include_values:
                        item["value"] = ent.value
                    found.append(item)
            else:
                before = Counter(mm.group() for mm in UNMASK_RE.finditer(text))
                after = Counter(mm.group() for mm in UNMASK_RE.finditer(masked))
                for mm in UNMASK_RE.finditer(masked):
                    code = mm.group()
                    if code in seen or after[code] <= before.get(code, 0):
                        continue
                    ent = store.get_by_code(code)
                    if ent is None:
                        continue
                    seen.add(code)
                    item = {"code": code, "kind": ent.kind}
                    if include_values:
                        item["value"] = ent.value
                    found.append(item)
            counts = dict(Counter(s.kind for s in res.spans) + canary)
            if d is not None:
                if res.new_entities or res.mode == "synthetic":
                    self._invalidate(d.code)
                d.journal("mask", spans=len(res.spans), new_entities=len(res.new_entities),
                          synthetic=res.mode == "synthetic")
            return {"masked": masked, "found": found, "counts": counts, "canary": sum(canary.values()),
                    "mode": res.mode}

    @_guarded("set_dossier_setting")
    def set_dossier_setting(self, dossier_code: str, key: str, value: Optional[str]) -> dict:
        """Change a per-dossier setting (v0.2): ``replacement`` = "codes" | "synthetic", ``language`` = "auto" |
        "fr" | "en". ``None`` removes it. Returns the settings of the dossier. Local callers only (the CLI refuses
        in an AI agent's shell): a model never chooses how its own input is masked."""
        with self._lock:
            d = self._dossier(dossier_code)
            key = str(key or "").strip().lower()
            if key not in DOSSIER_SETTINGS:
                raise ShieldError("Réglage inconnu : language ou replacement (unknown setting).", "invalid_argument")
            if value is not None:
                value = str(value).strip().lower()
                if value not in DOSSIER_SETTINGS[key]:
                    raise ShieldError(f"Valeur refusée pour « {key} » : "
                                      + " | ".join(DOSSIER_SETTINGS[key]) + " (invalid setting value).",
                                      "invalid_argument")
            d.set_setting(key, value)
            self._drop_maskers(d.code)
            self._invalidate(d.code)
            d.journal("set_setting", synthetic=d.setting("replacement") == "synthetic")
            return dict(d.settings())

    @_guarded("restore_text")
    def restore_report(self, text: str, dossier_code: str) -> dict:
        """{text, unknown_codes, restored, placeholders, shield_ids}: codes of the dossier replaced by the real
        values (local only). ``placeholders`` lists the generic marks left ("[SOCIETE]"), ``shield_ids`` the
        DOC_n / DOSSIER_n / OUTPUT_n ids that got no local name (R1 corrections CL-31, CL-36)."""
        with self._lock:
            d = self._dossier(dossier_code)
            text = _nfc(str(text or ""))
            r = self._restore_out(self._masker(d), text, self._id_labels(d, text))
            d.journal("restore", restored=r["restored"], unknown_codes=len(r["unknown_codes"]),
                      placeholders=len(r["placeholders"]), shield_ids=len(r["shield_ids"]),
                      synthetic=r.get("restored_synthetic", 0), unmatched_synthetic=r.get("unmatched_synthetic", 0))
            return {"text": r["text"], "unknown_codes": r["unknown_codes"], "restored": r["restored"],
                    "placeholders": r["placeholders"], "shield_ids": r["shield_ids"],
                    "restored_synthetic": r.get("restored_synthetic", 0),
                    "unmatched_synthetic": r.get("unmatched_synthetic", 0)}

    def restore_text(self, text: str, dossier_code: str) -> str:
        """Restored text (local only)."""
        return self.restore_report(text, dossier_code)["text"]

    @_guarded("mask_file")
    def mask_file(self, path: os.PathLike | str, dossier_code: Optional[str] = None,
                  name: Optional[str] = None) -> dict:
        """Read a file and mask it: {name, masked_name, masked_text, needs_ocr, warnings} (Desk "Fichiers")."""
        with self._lock:
            p = Path(path)
            label = _nfc(name or p.name)
            raw = documents.read_text(p)
            stem, ext = os.path.splitext(label)
            ext = ext.lower() if re.fullmatch(r"\.[A-Za-z0-9]{1,8}", ext or "") else ""
            d = self._dossier(dossier_code) if dossier_code else None
            # The masked name becomes the download name of the masked copy: it must hold no registered value.
            masked_stem = self._clean_label(d, stem, "document")
            masked = "" if raw.needs_ocr else self.mask_text(raw.text, dossier_code)["masked"]
            return {"name": label, "masked_name": masked_stem + ext, "masked_text": masked,
                    "needs_ocr": bool(raw.needs_ocr), "warnings": list(raw.warnings)}

    @_guarded("journal")
    def journal(self, dossier_code: Optional[str] = None, limit: int = 500) -> list[dict]:
        with self._lock:
            code = _dossier_code(dossier_code) if dossier_code else None
            return self.registry.journal_entries(code, limit=limit)

    def folders(self) -> dict[str, Path]:
        """{DOSSIER_n: folder} (REAL paths, local use: hooks)."""
        return self.registry.folders()


def _label_tokens(label: str) -> list[tuple[int, int]]:
    """Letter and digit runs of a short label, split inside glued words: "MaelleKervadec2026" -> "Maelle",
    "Kervadec", "2026"; "Kervadec-c-Veltrane" -> "Kervadec", "c", "Veltrane"; "kervadec's" -> "kervadec", "s"."""
    out: list[tuple[int, int]] = []
    for m in re.finditer(r"[^\W_]+", label):
        a, b = m.start(), m.end()
        cut = a
        for i in range(a + 1, b):
            x, y = label[i - 1], label[i]
            split = (x.islower() and y.isupper()) or (x.isdigit() != y.isdigit())
            if not split and x.isupper() and y.isupper() and i + 1 < b:
                split = label[i + 1].islower()
            if split:
                out.append((cut, i))
                cut = i
        out.append((cut, b))
    return out


#: Dotted initials in a label: "R.T.", "R. T.", "J.-B. W", "J.B.W." (two to four letters).
_DOTTED_INITIALS_RE = re.compile(r"(?<![^\W\d_.])(?:[A-Z]\.[ \-]?){1,3}[A-Z]\.?(?![^\W\d_])")
_ACRONYM_SKIP = frozenset({"et", "and", "of", "the", "de", "du", "des", "d", "la", "le", "les", "l", "en", "sur",
                           "sous", "aux", "au", "a"})


def _org_acronyms(value: str) -> list[str]:
    """Folded acronyms of a registered company: first letters of its significant words, with and without the
    generic company words, 3 to 6 letters ("Brasserie Xavaron Quellec" -> "bxq", "xq" is too short).
    Uses ``names.org_acronyms`` when the engine has it (R1 interface), else this local rule."""
    from .engine import names as nm
    from .engine.normalize import fold_word

    helper = getattr(nm, "org_acronyms", None)
    if callable(helper):
        try:
            return sorted({fold_word(a).replace(".", "") for a in helper(value) if a})
        except Exception:  # noqa: BLE001 - fall back to the local rule
            pass
    words = re.findall(r"[^\W\d_]+", fold_word(nm.org_core(value)))
    sig = [w for w in words if w not in _ACRONYM_SKIP and w not in nm.LEGAL_FORMS and w not in nm.PARTICLES]
    if len(sig) < 2:
        return []
    out = set()
    for group in (sig, [w for w in sig if w not in nm.ORG_GENERIC]):
        if len(group) >= 2:
            out.add("".join(w[0] for w in group))
    return sorted(a for a in out if 3 <= len(a) <= 6 and not _weak_word(a))


def _person_initials(value: str) -> list[str]:
    """Folded initials groups of a registered person with first name and surname ("Jean-Basile Wardec" ->
    "jbw", "jw"). Uses ``names.person_initials`` when the engine has it (R1 interface), else this local rule."""
    from .engine import names as nm
    from .engine.normalize import fold_word

    helper = getattr(nm, "person_initials", None)
    if callable(helper):
        try:
            return sorted({re.sub(r"[^a-z]", "", fold_word(a)) for a in helper(value) if a} - {""})
        except Exception:  # noqa: BLE001 - fall back to the local rule
            pass
    firsts = nm.first_initials(value)
    sur = "".join(w[0] for w in nm.surname_key(value).split() if w)
    if not firsts or not sur:
        return []
    return sorted({firsts + sur, firsts[0] + sur})


def _label_glue(label: str, toks: list[tuple[int, int]], i: int) -> bool:
    """True when tokens ``i`` and ``i + 1`` are glued or joined by one or two separators ("-", "_", ".", " ")."""
    gap = label[toks[i][1]:toks[i + 1][0]]
    return len(gap) <= 2 and all(c in "-_. '’\u2010\u2011" for c in gap)


def _safe_norm(kind: str, value: str) -> str:
    try:
        return entity_norm(kind, value)
    except Exception:  # noqa: BLE001
        return ""


def _restore_for_output() -> Optional[Callable]:
    """``restore_utils.restore_for_output`` (owned by the RESTORE correction) when installed, else None."""
    try:
        from . import restore_utils  # type: ignore[attr-defined]
    except ImportError:
        return None
    helper = getattr(restore_utils, "restore_for_output", None)
    return helper if callable(helper) else None


@contextlib.contextmanager
def output_dossier(code: Optional[str]):
    """Inside this block, :meth:`Service.guard_output` treats the text as belonging to dossier ``code`` (the MCP
    server sets it around the last canary of a tool result): see :meth:`Service._guard_full`."""
    token = _OUTPUT_DOSSIER.set(str(code).strip().upper() if code else None)
    try:
        yield
    finally:
        _OUTPUT_DOSSIER.reset(token)


def _variant_keys(kind: str, value: str) -> list[tuple[str, ...]]:
    """Folded token sequences (OCR look-alikes merged) of the engine variants of a registered name."""
    from .engine import names as nm
    from .engine.normalize import fold_word

    try:
        variants = nm.person_variants(value) if kind == "PERSON" else nm.org_variants(value)
    except Exception:  # noqa: BLE001 - never fail the canary on one stored value
        return []
    return list(dict.fromkeys(tuple(fold_word(t, confusables=True) for t in v.tokens) for v in variants if v.tokens))


def _weak_word(folded: str) -> bool:
    """True for a folded word that is not a distinctive name word: in a word list (function words, legal terms,
    common words, first names, months, whitelist), a legal form, a title, a particle or a generic company word."""
    from .engine import names as nm

    stop, legal, common, _fn, afn, months, wl = nm._lists()
    parts = [p for p in re.split(r"[-'’]", folded) if p] or [folded]
    return all(p in stop or p in legal or p in common or p in afn or p in months or p in wl or p in nm.LEGAL_FORMS
               or p in nm.ORG_GENERIC or p in nm.TITLE_WORDS or p in nm.PARTICLES for p in parts)


def _strong_name_word(folded: str) -> bool:
    """A distinctive name word: 4 letters or more and in no word list ("brakevald", not "financiere")."""
    return len(folded) >= 4 and not _weak_word(folded)


_FORMS = {"sas", "sasu", "sarl", "eurl", "sci", "snc", "sa", "selarl", "selas", "scp", "gie", "ltd", "llc",
          "inc", "gmbh", "bv", "spa", "societe", "monsieur", "madame", "maitre", "mme", "me", "aarpi", "spfpl",
          "selafa", "selca", "scm"}


def _only_forms(rest: str) -> bool:
    words = re.findall(r"[^\W\d_]+", fold(rest).text)
    return all(w in _FORMS or len(w) < 4 for w in words)


def _kind_counts(d: Dossier) -> dict[str, int]:
    return dict(Counter(e.kind for e in d.all()))


_STOPWORDS = frozenset(
    "les des une aux par pour sur dans avec sans sous que qui quoi dont est sont ont etre avoir "
    "ses son sa leur leurs mais donc car cet cette ces nos vos notre votre the and for with from "
    "this that are was were not you your".split())


def _score_hits(found: list[tuple[int, str, int, int]]) -> list[tuple[int, int, int]]:
    """Score each match: its weight plus 2 per other query item found near it (same window)."""
    if not found:
        return []
    found = sorted(found, key=lambda h: h[2])
    starts = [h[2] for h in found]
    out = []
    for weight, item, ms, me in found:
        lo = bisect.bisect_left(starts, ms - 2 * SNIPPET_RADIUS)
        hi = bisect.bisect_right(starts, me + 2 * SNIPPET_RADIUS)
        near = {h[1] for h in found[lo:hi] if h[1] != item}
        out.append((weight + 2 * len(near), ms, me))
    return out


# ------------------------------------------------------------------------------------------ default instance

_DEFAULT: dict[str, Service] = {}
_DEFAULT_LOCK = threading.Lock()


def preload_ner() -> bool:
    """Load the NER model in a background thread so that the first document read does not wait 10 to 15 s
    (v0.2). Returns True when a load was started. Nothing happens when no model is installed or NER is off."""
    model = config.ner_model()
    if not _ner.ner_installed(model):
        return False
    threading.Thread(target=_ner.load_backend, args=(model,), name="shield-ner-preload", daemon=True).start()
    return True


def default() -> Service:
    """The service for the current ``config.home()`` (re-created when RAGBASE_SHIELD_HOME changes)."""
    home = str(config.home())
    with _DEFAULT_LOCK:
        svc = _DEFAULT.get(home)
        if svc is None:
            for old in _DEFAULT.values():
                try:
                    old.close()
                except Exception:  # noqa: BLE001
                    pass
            _DEFAULT.clear()
            svc = Service(home)
            _DEFAULT[home] = svc
        return svc


def reset() -> None:
    """Close the default service (tests, verify), and the registries the hooks keep open when they are loaded
    (an open file cannot be deleted on Windows)."""
    with _DEFAULT_LOCK:
        for svc in _DEFAULT.values():
            try:
                svc.close()
            except Exception:  # noqa: BLE001
                pass
        _DEFAULT.clear()
    hooks = sys.modules.get(__package__ + ".hooks")
    if hooks is not None:
        hooks.close_registries()


def list_dossiers(include_private: bool = False) -> list[dict]:
    return default().list_dossiers(include_private)


def list_documents(dossier_code: str) -> list[dict]:
    return default().list_documents(dossier_code)


def read_document(dossier_code: str, doc_id: str, start: int = 0, max_chars: int = DEFAULT_READ_CHARS) -> dict:
    return default().read_document(dossier_code, doc_id, start, max_chars)


def search(dossier_code: str, query: str, limit: int = DEFAULT_SEARCH_RESULTS) -> list[dict]:
    return default().search(dossier_code, query, limit)


def save_draft(dossier_code: str, title: str, content: str, fmt: str = "docx") -> dict:
    return default().save_draft(dossier_code, title, content, fmt)


def save_draft_report(dossier_code: str, title: str, content: str, fmt: str = "docx") -> dict:
    return default().save_draft_report(dossier_code, title, content, fmt)


def mask_text(text: str, dossier_code: Optional[str] = None, include_values: bool = False,
              replacement: Optional[str] = None) -> dict:
    return default().mask_text(text, dossier_code, include_values, replacement)


def set_dossier_setting(dossier_code: str, key: str, value: Optional[str]) -> dict:
    return default().set_dossier_setting(dossier_code, key, value)


def restore_text(text: str, dossier_code: str) -> str:
    return default().restore_text(text, dossier_code)


def restore_report(text: str, dossier_code: str) -> dict:
    return default().restore_report(text, dossier_code)


def scan_text(text: str) -> list[dict]:
    return default().scan_text(text)


def has_registered(text: str) -> bool:
    return default().has_registered(text)


def status() -> dict:
    return default().status()


def add_dossier(folder: os.PathLike | str, title: Optional[str] = None) -> dict:
    return default().add_dossier(folder, title)


def scan_dossier(dossier_code: str) -> dict:
    return default().scan_dossier(dossier_code)


def register_folder_parties(dossier_code: str) -> int:
    return default().register_folder_parties(dossier_code)


def document_folders(dossier_code: str) -> dict[str, str]:
    return default().document_folders(dossier_code)


def registered_folders() -> dict[str, Path]:
    return default().registered_folders()


def protect_folder(folder: os.PathLike | str) -> dict:
    return default().protect_folder(folder)


def start_reading(dossier_code: str) -> dict:
    return default().start_reading(dossier_code)


def reading_state(dossier_code: str) -> Optional[dict]:
    return default().reading_state(dossier_code)


def readings() -> list[dict]:
    return default().readings()


def reading_summary(dossier_code: str) -> Optional[dict]:
    return default().reading_summary(dossier_code)


def reading_kept(dossier_code: str) -> bool:
    return default().reading_kept(dossier_code)


def wait_reading(dossier_code: str, timeout: float) -> bool:
    return default().wait_reading(dossier_code, timeout)


def ready_counts(dossier_code: str) -> dict:
    return default().ready_counts(dossier_code)


def clean_kept() -> list[str]:
    return default().clean_kept()


def skipped_paths(dossier_code: str) -> dict[str, int]:
    return default().skipped_paths(dossier_code)


def resume_readings() -> list[str]:
    return default().resume_readings()


def remove_dossier(dossier_code: str) -> None:
    default().remove_dossier(dossier_code)


def entities(dossier_code: str) -> list[dict]:
    return default().entities(dossier_code)


def register(dossier_code: str, kind: str, value: str) -> dict:
    return default().register(dossier_code, kind, value)


def guard_parts(parts: list[str], dossier_code: Optional[str] = None) -> list[str]:
    return default().guard_parts(parts, dossier_code)


def keep_clear(dossier_code: str, target: str, value: bool = True) -> dict:
    return default().keep_clear(dossier_code, target, value)


def journal(dossier_code: Optional[str] = None, limit: int = 500) -> list[dict]:
    return default().journal(dossier_code, limit)


def mask_file(path: os.PathLike | str, dossier_code: Optional[str] = None, name: Optional[str] = None) -> dict:
    return default().mask_file(path, dossier_code, name)


def protect(text: str) -> tuple[str, list[str]]:
    return default().protect(text)
