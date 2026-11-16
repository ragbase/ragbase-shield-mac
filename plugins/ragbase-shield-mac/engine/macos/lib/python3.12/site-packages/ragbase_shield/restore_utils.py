"""Restore a model answer for a LOCAL output (a draft file, a restored text shown on this computer).

Contract (R1 corrections, owner RESTORE; used by the service in ``save_draft`` and ``restore_report``)::

    restore_for_output(text, masker, *, id_labels=None, max_chars=None) -> RestoreReport

It does what ``Masker.unmask_report`` does (same codes, same unknown codes) and also:

- line breaks held in old vault values are removed from the restored value ("Jean\\nDUPONT" -> "Jean DUPONT"),
  so a restored value never splits a Markdown table row or a paragraph (CL-30);
- generic placeholders left by the cross-dossier guard ("[SOCIETE]", "[PERSON]"...) are listed, never silent
  (CL-31);
- French elision is adapted to the restored value: "de" + a value that starts with a vowel sound becomes
  "d'", and "d'" + a value that starts with a consonant becomes "de" (CL-32);
- the restored person name reads naturally: "SURNAME Firstname" becomes "Firstname SURNAME", a first name in
  capitals gets its usual case, the surname alone after a birth-name marker ("née"), no doubled word when the
  model already wrote the first name or the generic company word (CL-33);
- ``max_chars`` (for a file name): a text longer than the limit is cut before a restored value, never inside it,
  or at a word boundary (CL-35);
- Shield object ids (DOC_n, DOSSIER_n, OUTPUT_n) are replaced by the local labels of ``id_labels`` and the ones
  left are listed (CL-36);
- synthetic mentions (v0.2 synthetic replacement mode: a fake name, address or number the dossier gave) are
  restored to the real value before the codes, with the same form fixes; a fake name the model altered (one
  edit away from a synthetic name word) is counted in ``unmatched_synthetic``, never guessed.

Privacy: the report fields other than ``text`` hold codes, placeholders and counts only, never a real value or a
label (see :meth:`RestoreReport.public`). The restored ``text`` is for local use only.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from . import documents
from .engine import lexicon as lx
from .engine import names as nm
from .engine.masker import UNMASK_RE, canonical_codes
from .engine.normalize import fold_word
from .engine.types import PREFIXES

__all__ = ["RestoreReport", "restore_for_output", "find_placeholders", "find_shield_ids", "PLACEHOLDER_RE",
           "SHIELD_ID_RE", "FIX_KEYS"]

#: Keys of :attr:`RestoreReport.fixes`.
FIX_KEYS = ("line_break", "name_order", "birth_name", "doubled_word", "elision")

_PLACEHOLDER_NAMES = sorted({p for fr, en, _letters in PREFIXES.values() for p in (fr, en)}, key=len, reverse=True)
#: A generic placeholder of the guard ("[SOCIETE]"), also with Markdown escapes ("\\[SOCIETE\\]").
PLACEHOLDER_RE = re.compile(r"\\?\[(?P<p>" + "|".join(_PLACEHOLDER_NAMES) + r")\\?\]")
#: A Shield object id, also with a Markdown escape ("DOC\\_2") or a wide underscore.
SHIELD_ID_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?P<p>DOSSIER|DOC|OUTPUT)(?:\\_|_|＿)(?P<n>[0-9]{1,7})(?![A-Za-z0-9_])"
)
_ID_KEY_RE = re.compile(r"^(?:DOSSIER|DOC|OUTPUT)_[0-9]{1,7}$")

# Kinds whose restored value is a word that French elision applies to.
_ELISION_KINDS = frozenset({"PERSON", "ORG", "ADDRESS"})
_EM = r"(?P<em>\*{1,3}|_{1,3})?"
# "de PERSONNE_A", "que **SOCIETE_B**": the elidable word right before the value (same line).
_ELIDE_RE = re.compile(r"(?<![^\W\d_])(?P<w>de|le|la|que|jusque|lorsque|puisque)(?P<sp>[ \t\u00a0]+)" + _EM + r"$",
                       re.IGNORECASE)
# "d'PERSONNE_A", "qu’PERSONNE_B": an elided word right before the value.
_UNELIDE_RE = re.compile(r"(?<![^\W\d_])(?P<w>d|qu|jusqu|lorsqu|puisqu)(?P<ap>['’])" + _EM + r"$", re.IGNORECASE)
# A birth-name or usage-name marker right before the value: the surname alone is restored there.
_BIRTH_MARK_RE = re.compile(
    r"(?<![^\W\d_])(?:n[ée]e|n[ée]|nom\s+de\s+naissance|nom\s+de\s+jeune\s+fille|nom\s+d['’]usage)"
    r"[ \t\u00a0]*:?[ \t\u00a0]*(?:\*{1,3}|_{1,3})?$",
    re.IGNORECASE,
)
# The word written right before the value, on the same line.
_PREV_WORD_RE = re.compile(r"([^\W\d_][\w'’\-]*)[ \t\u00a0]+(?:\*{1,3}|_{1,3})?$")
# A line break inside a value; a word cut by a hyphen at a line end ("Kerva-\ndec").
_BREAK_RE = re.compile(r"[ \t\u00a0\u202f]*(?:\r\n|\r|\n)[ \t\u00a0\u202f\r\n]*|\t+| {2,}")
_HYPHEN_BREAK_RE = re.compile(r"(?<=[^\W\d_])-[ \t\u00a0\u202f]*(?:\r\n|\r|\n)[ \t\u00a0\u202f]*(?=[^\W\d_])")


@dataclass
class RestoreReport:
    """Result of :func:`restore_for_output`.

    - ``text``: the restored text (LOCAL use only: it holds real values and labels);
    - ``unknown_codes``: codes that are not in the dossier, and near misses ("personne_a"), as ``unmask_report``;
    - ``placeholders``: generic placeholders left in the text, one entry per occurrence ("[SOCIETE]");
    - ``shield_ids``: DOC_n / DOSSIER_n / OUTPUT_n left in the text, one entry per occurrence;
    - ``ids_replaced``: the ids replaced by a local label (codes only);
    - ``restored``: number of entity codes restored;
    - ``fixes``: counts of the form fixes (keys :data:`FIX_KEYS`);
    - ``truncated``: True when ``max_chars`` cut the text;
    - ``restored_synthetic``: number of synthetic mentions restored (part of ``restored``) (v0.2);
    - ``unmatched_synthetic``: number of words one edit away from a synthetic name word that restore could not map
      back (a fake name the model altered) (v0.2).
    """

    text: str = field(repr=False)
    unknown_codes: list[str] = field(default_factory=list)
    placeholders: list[str] = field(default_factory=list)
    shield_ids: list[str] = field(default_factory=list)
    ids_replaced: list[str] = field(default_factory=list)
    restored: int = 0
    fixes: dict[str, int] = field(default_factory=lambda: dict.fromkeys(FIX_KEYS, 0))
    truncated: bool = False
    restored_synthetic: int = 0
    unmatched_synthetic: int = 0

    def public(self) -> dict[str, Any]:
        """Codes, placeholders and counts only (safe to return to a model): no text, no value, no label."""
        return {"unknown_codes": list(self.unknown_codes), "placeholders": list(self.placeholders),
                "shield_ids": list(self.shield_ids), "ids_replaced": list(self.ids_replaced),
                "restored": self.restored, "fixes": dict(self.fixes), "truncated": self.truncated,
                "restored_synthetic": self.restored_synthetic, "unmatched_synthetic": self.unmatched_synthetic}


def find_placeholders(text: str) -> list[str]:
    """Generic placeholders in ``text``, one entry per occurrence, without Markdown escapes ("[SOCIETE]")."""
    return [f"[{m.group('p')}]" for m in PLACEHOLDER_RE.finditer(text or "")]


def find_shield_ids(text: str) -> list[str]:
    """Shield object ids in ``text``, one entry per occurrence, in their exact form ("DOC_2")."""
    return [f"{m.group('p')}_{m.group('n')}" for m in SHIELD_ID_RE.finditer(text or "")]


# --------------------------------------------------------------------------------------------- values

def _collapse(value: str) -> tuple[str, int]:
    """Remove line breaks, tabs and repeated spaces inside a value. A word cut by a hyphen at a line end is
    joined when the next part goes on in lower case ("Kerva-\\ndec"); "MOREL-\\nDUVAL" keeps its hyphen."""
    if "\n" not in value and "\r" not in value and "\t" not in value and "  " not in value:
        return value, 0

    def hyphen(m: re.Match) -> str:
        nxt = value[m.end():m.end() + 1]
        return "" if nxt.islower() else "-"

    out = _HYPHEN_BREAK_RE.sub(hyphen, value)
    out = _BREAK_RE.sub(" ", out).strip()
    return out, int(out != value)


def _letters(tok: str) -> list[str]:
    return [c for c in tok if c.isalpha()]


def _upper_word(tok: str) -> bool:
    """A name word in capitals ("DUPONT", "LE", "D'ARVOR"), not initials ("J.-P.")."""
    letters = _letters(tok)
    return len(letters) >= 2 and "." not in tok and all(c.isupper() for c in letters) \
        and not any(c.isdigit() for c in tok)


def _cap_word(tok: str) -> bool:
    """A capitalised mixed-case word ("Jean-Pierre", "Maëlle")."""
    return bool(tok) and tok[0].isupper() and any(c.islower() for c in tok) and "." not in tok \
        and not any(c.isdigit() for c in tok)


def _known_first(tok: str) -> bool:
    """A first name of the lexicon ("Jean", "JEAN-PIERRE": every part)."""
    afn = lx.all_first_names()
    low = fold_word(tok).strip("'’")
    if low in afn:
        return True
    parts = [p for p in low.split("-") if p]
    return len(parts) > 1 and all(p in afn for p in parts)


def _title_case(tok: str) -> str:
    """"JEAN-PIERRE" -> "Jean-Pierre"."""
    return "-".join(p[:1].upper() + p[1:].lower() for p in tok.split("-"))


def _natural_order(value: str) -> tuple[str, int]:
    """"DUPONT Jean" -> "Jean DUPONT"; "JEAN DUPONT" -> "Jean DUPONT". Titles stay first.

    Surname-first order is changed only in its unambiguous form: name words in capitals, then capitalised
    words whose first one is a known first name. A value all in capitals is changed only when the split is
    clear: its first names are lexicon first names and no surname word is one.
    """
    toks = value.split(" ")
    i = 0
    while i < len(toks) - 1 and nm.is_title_word(toks[i]):
        i += 1
    head, rest = toks[:i], toks[i:]
    if len(rest) < 2:
        return value, 0
    k = 0
    while k < len(rest) and _upper_word(rest[k]):
        k += 1
    if 1 <= k < len(rest) and len(rest) - k <= 3 and all(_cap_word(t) for t in rest[k:]) \
            and _known_first(rest[k]):
        return " ".join(head + rest[k:] + rest[:k]), 1
    if k == len(rest):
        firsts, surname = nm.split_person(" ".join(rest))
        if firsts and surname and all(_upper_word(t) and _known_first(t) for t in firsts) \
                and not any(_known_first(t) for t in surname):
            new = " ".join(head + [_title_case(t) for t in firsts] + list(surname))
            return new, int(new != value)
    return value, 0


def _surname_only(value: str) -> str:
    firsts, surname = nm.split_person(value)
    return " ".join(surname) if firsts and surname else value


def _fold_plain(word: str) -> str:
    return unicodedata.normalize("NFKD", word).encode("ascii", "ignore").decode("ascii").lower()


def _person_value(value: str, before: str, fixes: dict[str, int]) -> str:
    value, n = _natural_order(value)
    fixes["name_order"] += n
    if _BIRTH_MARK_RE.search(before):
        short = _surname_only(value)
        if short != value:
            fixes["birth_name"] += 1
        return short
    prev = _PREV_WORD_RE.search(before)
    words = value.split(" ")
    if prev and len(words) > 1:
        word = prev.group(1)
        low = fold_word(word)
        if word[:1].isupper() and low not in nm.PARTICLES and low not in nm.TITLE_WORDS \
                and len(_letters(word)) >= 2 and low == fold_word(words[0]):
            fixes["doubled_word"] += 1
            return " ".join(words[1:])
    return value


def _org_value(value: str, before: str, fixes: dict[str, int]) -> str:
    prev = _PREV_WORD_RE.search(before)
    words = value.split(" ")
    if prev and len(words) > 1:
        word = prev.group(1)
        low = fold_word(word)
        if low in nm.ORG_GENERIC and low == fold_word(words[0]):
            fixes["doubled_word"] += 1  # "le Groupe SOCIETE_A" with "Groupe X"
            return " ".join(words[1:])
        if word.isupper() and low in nm.LEGAL_FORMS and fold_word(words[-1]).replace(".", "") == low:
            fixes["doubled_word"] += 1  # "la SAS SOCIETE_A" with "X SAS"
            return " ".join(words[:-1])
    return value


def _sound(value: str) -> Optional[str]:
    """"vowel", "consonant" or None (not a letter, or an "h": aspirated or not is unknown)."""
    if not value or not value[0].isalpha():
        return None
    first = _fold_plain(value[0])[:1]
    if not first:
        return None
    if first in "aeiou":
        return "vowel"
    if first == "y":
        nxt = _fold_plain(value[1])[:1] if len(value) > 1 and value[1].isalpha() else ""
        if not nxt:
            return None
        return "consonant" if nxt in "aeiouy" else "vowel"  # "Yannick" / "Yves"
    if first == "h":
        return None
    return "consonant" if "a" <= first <= "z" else None


def _elide(chunk: str, value: str, prev_char: str, apos: str) -> tuple[str, bool]:
    """Adapt the elidable word at the end of ``chunk`` to the value that follows it."""
    sound = _sound(value)
    if sound is None:
        return chunk, False
    rx = _ELIDE_RE if sound == "vowel" else _UNELIDE_RE
    m = rx.search(chunk)
    if not m:
        return chunk, False
    if m.start() == 0 and prev_char and (prev_char.isalnum() or prev_char in "'’"):
        return chunk, False  # the word is glued to the previous value
    w, em = m.group("w"), m.group("em") or ""
    if sound == "vowel":
        new = w[:-1] + apos + em
    else:
        full = w + ("E" if w.isupper() and len(w) > 1 else "e")
        new = full + " " + em
    return chunk[:m.start()] + new, True


# --------------------------------------------------------------------------------------------- cutting

def _cut(text: str, spans: list[tuple[int, int]], limit: int) -> str:
    """Cut ``text`` to at most ``limit`` characters, before a restored value (never inside it) when a value
    crosses the limit and enough text comes before it, else at a word boundary."""
    if len(text) <= limit:
        return text
    for s, e in spans:
        if s < limit < e and s >= limit // 3:
            return documents.trim_dangling(text[:s]) or text[:s]
    return documents.cut_at_word(text, limit)


# --------------------------------------------------------------------------------------------- main

def _clean_labels(id_labels: Optional[Mapping[str, str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, label in (id_labels or {}).items():
        key = str(key or "").strip()
        label = " ".join(str(label or "").split())
        if _ID_KEY_RE.match(key) and label:
            out[key] = label
    return out


def restore_for_output(text: str, masker, *, id_labels: Optional[Mapping[str, str]] = None,
                       max_chars: Optional[int] = None) -> RestoreReport:
    """Restore ``text`` with the dossier's ``masker`` for a local output and report what is left.

    ``masker`` is the dossier's :class:`~ragbase_shield.engine.masker.Masker` (only ``store.get_by_code``,
    ``unmask_report`` and ``_restore_value`` are used). ``id_labels`` maps DOC_n / DOSSIER_n / OUTPUT_n to the
    local label written in their place (file name, dossier title). ``max_chars`` cuts the restored text for a
    file name.
    """
    text = "" if text is None else str(text)
    report = RestoreReport(text=text)
    if not text:
        return report
    try:
        _plain, unknown = masker.unmask_report(text)
    except Exception:  # noqa: BLE001 - a broken store: report nothing rather than fail the save
        unknown = []
    report.unknown_codes = list(unknown)
    src = canonical_codes(text)
    labels = _clean_labels(id_labels)
    apos = "’" if src.count("’") > src.count("'") else "'"
    restore_value = getattr(masker, "_restore_value", None)

    items: list[tuple[int, int, str, Any]] = []
    # Synthetic mentions first (v0.2): the masker finds the fake surfaces and gives the real text of each one.
    synthetic_items = getattr(masker, "synthetic_restore_items", None)
    if callable(synthetic_items):
        try:
            for s, e, ent, real in synthetic_items(src):
                items.append((s, e, "synthetic", (ent, real)))
        except Exception:  # noqa: BLE001 - a broken synthetic store: the codes are still restored
            pass
        altered = getattr(masker, "synthetic_altered", None)
        if callable(altered):
            try:
                report.unmatched_synthetic = int(altered(src))
            except Exception:  # noqa: BLE001
                report.unmatched_synthetic = 0
    for m in UNMASK_RE.finditer(src):
        try:
            ent = masker.store.get_by_code(m.group())
        except Exception:  # noqa: BLE001
            ent = None
        if ent is not None:
            items.append((m.start(), m.end(), "code", ent))
    for m in SHIELD_ID_RE.finditer(src):
        ident = f"{m.group('p')}_{m.group('n')}"
        if ident in labels:
            items.append((m.start(), m.end(), "id", ident))
    items.sort(key=lambda it: it[0])

    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    out_len = 0
    pos = 0
    fixes = report.fixes
    for s, e, typ, payload in items:
        if s < pos:
            continue
        chunk = src[pos:s]
        before = src[max(0, s - 60):s]
        if typ in ("code", "synthetic"):
            if typ == "code":
                ent = payload
                value = restore_value(ent, src, s, e) if callable(restore_value) else ent.value
            else:
                ent, value = payload
                report.restored_synthetic += 1
            value, n = _collapse(value)
            fixes["line_break"] += n
            if ent.kind == "PERSON":
                value = _person_value(value, before, fixes)
            elif ent.kind == "ORG":
                value = _org_value(value, before, fixes)
            if ent.kind in _ELISION_KINDS:
                prev_char = parts[-1][-1:] if parts else ""  # last character of the previous value
                chunk, changed = _elide(chunk, value, prev_char, apos)
                fixes["elision"] += int(changed)
            report.restored += 1
        else:
            value = labels[payload]
            report.ids_replaced.append(payload)
        parts.append(chunk)
        out_len += len(chunk)
        parts.append(value)
        spans.append((out_len, out_len + len(value)))
        out_len += len(value)
        pos = e
    parts.append(src[pos:])
    out = "".join(parts)
    if max_chars is not None and max_chars > 0 and len(out) > max_chars:
        out = _cut(out, spans, max_chars)
        report.truncated = True
    report.text = out
    report.placeholders = find_placeholders(out)
    report.shield_ids = find_shield_ids(out)
    return report
