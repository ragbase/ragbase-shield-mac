"""Identifier cue rules, in-text id layouts, column identifiers and street-level address edges (S6, 2026-10-20).

Registered from ``detectors.py`` (three lines at its end): ``DETECTORS = DETECTORS + DETECTORS_IDS``. Every rule is
general: label words, value shapes and layouts, never a value, a prefix or a name of a data set. Examples invented.

1. ``labelled_id`` (cue values): a bare label noun of an account or staff identifier ("account", "acct",
   "registration", "plan", "policy", "member", "customer", "employee", "badge", "staff", "payroll"...) followed on
   the same line, within three small words ("number", "no.", "#", "of record", "ending in"...), by an id-shaped
   value: a letter prefix with four or more digits ("ACC-00917"), two to five digit groups with separators and six
   or more digits ("5826-3308"), six to twelve plain digits, or a masked tail ("****-5106"). Phone, SSN, date, time,
   amount, percentage, year, version, ZIP+4 and grouped-number shapes are refused.
2. ``labelled_id`` (keyed lines): a line that starts with such a label (a form or a PDF key-value row) and ends with
   an id-shaped value, the holder's name in between ("Registration  Holder Family Trust  5826-3307").
3. ``layout_id``: once a value is cue-confirmed in a text, every other token of the same layout (same digit group
   lengths, same separators, same letter prefix) in that text is the same kind, and a list of short tails after it
   ("5826-3306, -3307, -3308") gives one id per tail. Plain digit runs and space-separated digit groups never
   propagate (phone, year and amount risk). Letter-prefixed layouts are also learnt dossier-wide by the masker's
   shape memory, which reads the detector name ``labelled_id``.
4. ``table_id``: a pipe-table column whose header holds a label noun, also written with underscores
   ("account_number", "employee_id"); the cell is the value, or ends with the value after a holder name.
5. ``address_street`` (English texts): a US street-level address that the base detector misses when the PDF text
   wraps it over two lines ("4030 Summit\\nBoulevard, Suite 1200") or when the street name holds a street-type word
   ("7120 Grove Street West"); the span keeps the directional suffix and the unit, then extends over the city, state
   and ZIP like the base detector.

6. ``find_grouped_ids`` (English texts, v0.4 T8): an id value whose prefix holds 2 or 3 capital groups
   ("ABC-DE-1234") under a label noun or an id column, every id of a cell that lists several, and the values of
   the same layout in the rest of the text.
7. ``find_member_ids`` (English texts, v0.4 T8): a workspace member id written bare ("U02ABCD7EF") and a profile
   handle after the "LinkedIn" label ("LinkedIn: jane-doe-12a") are usernames.

The identifier kinds are ``EMPLOYEE_ID`` and ``ACCOUNT_NUMBER`` (``types.KINDS``); the benchmark adapters map the
generic ``ID_CODE`` to their ``ID_NUMBER`` alias, which the public scorers accept for both labels.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Callable, Iterable, Iterator, Optional

from . import detectors as det
from .detectors import PRIORITY, Candidate
from .normalize import fold_word

__all__ = ["DETECTORS_IDS", "ID_LABELS", "find_ids", "find_table_ids", "find_street_addresses", "find_grouped_ids",
           "find_member_ids", "grouped_value_at", "id_layout", "id_value_at", "value_refused"]

# ---------------------------------------------------------------------------
# Label words (both languages; folded, lower case)
# ---------------------------------------------------------------------------

_EMP_WORDS = ("employee", "employees", "emp", "staff", "badge", "worker", "personnel", "payroll", "associate",
              "agent", "contractor", "salarie", "salaries", "matricule")
_ACCT_WORDS = ("account", "accounts", "acct", "accts", "a/c", "registration", "registrations", "plan", "policy",
               "member", "membership", "customer", "client", "loyalty", "loan", "card", "portfolio", "ira",
               "401(k)", "403(b)", "annuity", "compte", "comptes", "police", "adherent", "societaire")
ID_LABELS: dict[str, tuple[str, ...]] = {"EMPLOYEE_ID": _EMP_WORDS, "ACCOUNT_NUMBER": _ACCT_WORDS}
_EMP_SET = frozenset(_EMP_WORDS)
_ACCT_SET = frozenset(_ACCT_WORDS)


def _alt(words: Iterable[str]) -> str:
    return "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))


_LABEL_RE = re.compile(r"(?<![\w/])(?:(?P<emp>" + _alt(_EMP_WORDS) + r")|(?P<acct>" + _alt(_ACCT_WORDS) + r"))(?![\w])",
                       re.IGNORECASE)
# words that may sit between the label and its value ("number", "no.", "#", "of record", "ending in", "is"...)
_GAP_TOKEN_RE = re.compile(r"[ \t\u00a0]*(?:[:#=(\-–]|[A-Za-z][A-Za-z.'’]{0,11})")
_LABEL_SUFFIX = r"(?:[ \t\u00a0]+(?:number|no\.?|num\.?|id|#|of[ \t\u00a0]+record|type|holder))?"
# words that may follow a bare label in any case ("Employee ID", "Account Number"); any other capitalised word
# after the label makes it a title or a name ("Account Manager", "Registration Holder Trust"), not a label
_MARKER_WORDS = frozenset({"number", "no", "num", "id", "ref", "of", "record", "ending", "in", "is", "was", "for",
                           "the", "your", "my", "our", "his", "her", "its", "new", "old", "existing", "current",
                           "primary", "joint", "taxable", "type", "status", "holder", "code", "nbr"})
# a bare-noun cue ranks under an explicit label of either kind, the shape memory, a column header and the
# labelled references of the base detectors (ID_CODE); a layout sibling ranks under the cue that taught it
_CUE_PRIORITY = {k: PRIORITY["ID_CODE"] - 1 for k in ("ACCOUNT_NUMBER", "EMPLOYEE_ID")}
_LAYOUT_PRIORITY = {k: v - 1 for k, v in _CUE_PRIORITY.items()}

# ---------------------------------------------------------------------------
# Value shapes
# ---------------------------------------------------------------------------

_V_PREFIXED = r"[A-Z]{1,5}[\-/.]?[ \u00a0]?\d{4,12}(?:[\-/.]\d{1,6})?"
_V_MASKED = r"[*xX•●]{3,}[\-. \u00a0]?\d{3,6}"
_V_GROUPED = r"\d{2,6}(?:[\-. \u00a0]\d{2,6}){1,4}"
_V_PLAIN = r"\d{6,12}"
_VALUE = r"(?:" + "|".join((_V_PREFIXED, _V_MASKED, _V_GROUPED, _V_PLAIN)) + r")"
_VALUE_RE = re.compile(r"(?P<v>" + _VALUE + r")(?![A-Za-z0-9]|[\-/.]\d)")
_VALUE_FULL_RE = re.compile(r"(?P<v>" + _VALUE + r")")

_YEAR_RE = re.compile(r"(?:1[89]|20)\d{2}")
_TIME_RE = re.compile(r"\d{1,2}[hH:]\d{2}(?::\d{2})?")
_NANP_RE = re.compile(r"[2-9]\d{2}[\-. \u00a0]?[2-9]\d{2}[\-. \u00a0]?\d{4}|1[\-. \u00a0]?[2-9]\d{2}[\-. \u00a0]?[2-9]\d{2}[\-. \u00a0]?\d{4}")
_SHORT_PHONE_RE = re.compile(r"\d{3}[\-. \u00a0]\d{4}")
_SSN_RE = re.compile(r"\d{3}[\-. \u00a0]\d{2}[\-. \u00a0]\d{4}")
_ZIP4_RE = re.compile(r"\d{5}-\d{4}")
_YEAR_RANGE_RE = re.compile(r"(?:1[89]|20)\d{2}[\-/.](?:(?:1[89]|20)\d{2}|\d{1,2})")
_VERSION_RE = re.compile(r"\d{1,2}(?:\.\d{1,2}){1,4}")
_CURRENCY_BEFORE_RE = re.compile(r"[$€£¥][ \t\u00a0]*$|(?i:usd|eur|gbp|cad|chf)[ \t\u00a0]*$")
_UNIT_AFTER_RE = re.compile(r"^[ \t\u00a0]*(?:%|‰|px|pt|kg|km|mb|gb|ms|bps|k€|m€|\$|€|£)")


def value_refused(text: str, s: int, e: int, explicit: bool = False) -> bool:
    """A token that looks like an id but is a phone, SSN, date, year or year range, time, version, ZIP+4, amount,
    percentage or grouped number: never an account or staff identifier. A grouped number ("12 345 678") is an id
    only after an explicit marker (``explicit``: "#", ":", "number"...), as in the labelled-reference rule."""
    v = text[s:e]
    if _YEAR_RE.fullmatch(v) or _TIME_RE.fullmatch(v) or det._DATE_SHAPE_RE.fullmatch(v) or _YEAR_RANGE_RE.fullmatch(v):
        return True
    if _NANP_RE.fullmatch(v) or _SHORT_PHONE_RE.fullmatch(v) or _SSN_RE.fullmatch(v) or _ZIP4_RE.fullmatch(v):
        return True
    if _VERSION_RE.fullmatch(v) or (det._GROUPED_NUMBER_RE.fullmatch(v) and not explicit):
        return True
    if "." in v and not any(c.isalpha() for c in v) and max(len(g) for g in re.split(r"\D+", v) if g) < 3:
        return True  # "10.20.30": a version or a section number, not a dotted identifier
    if det._is_amount(text, s, e) or _UNIT_AFTER_RE.match(text[e:e + 6]):
        return True
    if _CURRENCY_BEFORE_RE.search(text[max(0, s - 6):s]):
        return True
    return False


def id_value_at(text: str, pos: int, explicit: bool = False) -> Optional[tuple[int, int]]:
    """(start, end) of an id-shaped value starting at ``pos`` (spaces skipped), or None when it is refused."""
    q = pos
    n = len(text)
    while q < n and text[q] in " \t\u00a0":
        q += 1
    m = _VALUE_RE.match(text, q)
    if not m:
        return None
    s, e = m.start("v"), m.end("v")
    if value_refused(text, s, e, explicit):
        return None
    return s, e


def id_layout(value: str) -> Optional[str]:
    """Layout of an identifier for in-text propagation: digits written ``d``, letters upper case, separators kept
    ("5826-3308" -> "dddd-dddd", "TQ-604183" -> "TQ-dddddd"). None for a plain digit run, a masked value, a value
    under six digits or a layout that has neither a letter prefix nor a "-", "/" or "." separator."""
    v = value.strip()
    if not v or v.isdigit() or v[0] in "*xX•●" or sum(c.isdigit() for c in v) < 6:
        return None
    lay = "".join("d" if c.isdigit() else (c.upper() if c.isalpha() else c) for c in v)
    if "d" not in lay:
        return None
    if v[0].isalpha() or any(ch in lay for ch in "-/."):
        return lay
    return None


@lru_cache(maxsize=256)
def _layout_pattern(layout: str) -> "re.Pattern[str]":
    body = "".join(r"\d" if c == "d" else re.escape(c) for c in layout)
    return re.compile(r"(?<![A-Za-z0-9])" + body + r"(?![A-Za-z0-9]|[\-/.]\d)", re.IGNORECASE)


def _kind_of(group: str) -> str:
    return "EMPLOYEE_ID" if group == "emp" else "ACCOUNT_NUMBER"


# ---------------------------------------------------------------------------
# 1 + 2: cue values and keyed lines, 3: layouts and tails
# ---------------------------------------------------------------------------

_KEYED_RE = re.compile(
    r"(?m)^[ \t\u00a0]*(?i:(?P<emp>" + _alt(_EMP_WORDS) + r")|(?P<acct>" + _alt(_ACCT_WORDS) + r"))(?i:" + _LABEL_SUFFIX + r")"
    + r"[ \t\u00a0]*[:#\-–]?[ \t\u00a0]+(?P<mid>[^\n|]{0,90}?)[ \t\u00a0]*(?P<v>" + _VALUE + r")[ \t\u00a0]*$")
_TAIL_RE_CACHE: dict[int, "re.Pattern[str]"] = {}


def _tail_pattern(n: int) -> "re.Pattern[str]":
    rx = _TAIL_RE_CACHE.get(n)
    if rx is None:
        rx = re.compile(r"[ \t\u00a0]*,?[ \t\u00a0]*(?:(?:and|et|&|to|or)[ \t\u00a0]+)?-(?P<t>\d{%d})(?![A-Za-z0-9\-])" % n)
        _TAIL_RE_CACHE[n] = rx
    return rx


def _cue_values(text: str, value_at: Callable[..., Optional[tuple[int, int]]] = id_value_at) -> Iterator[tuple[int, int, str]]:
    """(start, end, kind) of the values that follow a bare label noun on the same line. ``value_at`` reads the
    value shape (the plain shapes by default; the grouped letter prefixes of :func:`find_grouped_ids`)."""
    n = len(text)
    for lab in _LABEL_RE.finditer(text):
        kind = _kind_of(lab.lastgroup or "acct")
        p = lab.end()
        caps = 0
        explicit = False
        for _ in range(4):
            got = value_at(text, p, explicit)
            if got is not None:
                yield got[0], got[1], kind
                break
            g = _GAP_TOKEN_RE.match(text, p)
            if not g or g.end() == p:
                break
            tok = text[g.start():g.end()].strip()
            low = tok.lower().rstrip(".")
            if tok in (":", "#", "=") or low in ("number", "no", "num", "id", "nbr"):
                explicit = True
            if tok[:1].isupper() and tok[1:2].islower() and low not in _MARKER_WORDS:
                break  # "Account Manager", "Registration Holder Trust": a title or a name, not a label
            if tok[:1].isalpha() and low not in _MARKER_WORDS and not tok.isupper():
                caps += 1
                if caps >= 3:
                    break
            p = g.end()
            if p >= n:
                break


def _keyed_lines(text: str) -> Iterator[tuple[int, int, str]]:
    for m in _KEYED_RE.finditer(text):
        mid = m.group("mid")
        if re.search(r"\d{3,}", mid) or "|" in mid:
            continue
        s, e = m.start("v"), m.end("v")
        if value_refused(text, s, e):
            continue
        yield s, e, _kind_of("emp" if m.group("emp") else "acct")


def find_ids(text: str) -> Iterator[Candidate]:
    """Cue values, keyed lines, then the in-text layouts and tails of the confirmed values."""
    if not any(c.isdigit() for c in text):
        return
    confirmed: dict[tuple[int, int], str] = {}
    for s, e, kind in _cue_values(text):
        confirmed.setdefault((s, e), kind)
    for s, e, kind in _keyed_lines(text):
        confirmed.setdefault((s, e), kind)
    for (s, e), kind in confirmed.items():
        yield Candidate(s, e, kind, True, _CUE_PRIORITY[kind], "labelled_id")
    layouts: dict[str, set[str]] = {}
    for (s, e), kind in confirmed.items():
        lay = id_layout(text[s:e])
        if lay:
            layouts.setdefault(lay, set()).add(kind)
        # a list of short tails after a value: "5826-3306, -3307 and -3308"
        groups = [g for g in re.split(r"\D+", text[s:e]) if g]
        if len(groups) >= 2 and 3 <= len(groups[-1]) <= 6 and not text[s:e].startswith(("*", "x", "X")):
            rx = _tail_pattern(len(groups[-1]))
            p = e
            while True:
                tm = rx.match(text, p)
                if not tm:
                    break
                yield Candidate(tm.start("t"), tm.end("t"), kind, True, _LAYOUT_PRIORITY[kind], "layout_id")
                p = tm.end()
    for lay, kinds in layouts.items():
        if len(kinds) != 1:
            continue  # one layout confirmed as two kinds in one text: nobody propagates
        kind = next(iter(kinds))
        for m in _layout_pattern(lay).finditer(text):
            if (m.start(), m.end()) in confirmed or value_refused(text, m.start(), m.end()):
                continue
            yield Candidate(m.start(), m.end(), kind, True, _LAYOUT_PRIORITY[kind], "layout_id")


# ---------------------------------------------------------------------------
# 4: table columns
# ---------------------------------------------------------------------------

_HEADER_EMP_RE = re.compile(r"(?<![a-z])(?:" + _alt(_EMP_WORDS) + r")(?![a-z])")
_HEADER_ACCT_RE = re.compile(r"(?<![a-z])(?:" + _alt(_ACCT_WORDS) + r")(?![a-z])")
_CELL_TAIL_RE = re.compile(r"^(?P<lead>[^\d|]*?)[ \u00a0]+(?P<v>" + _VALUE + r")[ \t\u00a0]*$")


def _header_kind(labels: Iterable[str]) -> Optional[str]:
    for lab in labels:
        norm = " ".join(re.sub(r"[_\-/]+", " ", lab).split())
        if _HEADER_EMP_RE.search(norm):
            return "EMPLOYEE_ID"
        if _HEADER_ACCT_RE.search(norm):
            return "ACCOUNT_NUMBER"
    return None


def find_table_ids(text: str) -> Iterator[Candidate]:
    if "|" not in text:
        return
    for cells, header in det._table_rows(text):
        for i, (s, e) in enumerate(cells):
            if s >= e:
                continue
            kind = _header_kind(det._cell_labels(cells, header, i, text))
            if kind is None:
                continue
            raw = text[s:e]
            cell = raw.strip(" \t\u00a0")
            cs = s + (len(raw) - len(raw.lstrip(" \t\u00a0")))
            m = _VALUE_FULL_RE.fullmatch(cell)
            if m:
                vs, ve = cs, cs + len(cell)
            else:
                t = _CELL_TAIL_RE.match(cell)
                if not t or any(ch.isdigit() for ch in t.group("lead")):
                    continue
                vs, ve = cs + t.start("v"), cs + t.end("v")
            if value_refused(text, vs, ve):
                continue
            yield Candidate(vs, ve, kind, True, PRIORITY[kind], "table_id")


# ---------------------------------------------------------------------------
# 5: street-level addresses (English)
# ---------------------------------------------------------------------------

_WRAP = r"(?:[ \u00a0]*\n[ \u00a0]*|[ \u00a0]+)"
_ST_WORD = r"(?:[A-Z][\w'’\-]*|\d{1,3}(?:st|nd|rd|th)|of|the|and|de|la|del|du|des)"
_STREET_RE = re.compile(
    r"(?<![\w\-/$#.,])(?P<num>\d{1,6}[A-Za-z]?(?:-\d{1,4})?)[ \u00a0]+"
    r"(?:(?P<dir1>" + det._US_DIRS + r")\.?" + _WRAP + r")?"
    r"(?P<name>(?:" + _ST_WORD + _WRAP + r"){0,4})"
    r"(?P<type>" + det._US_STREET_TYPES + r")\.?(?:[ \u00a0]+(?P<dir2>" + det._US_DIRS + r")\.?)?(?![\w])"
)
_UNIT_RE = re.compile(
    r",?[ \t\u00a0]*(?:\n[ \t\u00a0]*)?(?:(?i:suite|ste|apt|apartment|unit|floor|fl|bldg|building|room|rm|no)\.?"
    r"[ \t\u00a0]*#?|#[ \t\u00a0]*)(?P<id>[\w\-]{1,8})(?![\w])"
)
_ZIP_NEXT_LINE_RE = re.compile(r"[ \t\u00a0]*\n[ \t\u00a0]*(?P<zip>\d{5}(?:-\d{4})?)(?![\w\-])")
_STATE_END_RE = re.compile(r"(?:^|[ \u00a0,])([A-Z]{2})$")


def find_street_addresses(text: str) -> Iterator[Candidate]:
    pr = PRIORITY["ADDRESS"]
    for m in _STREET_RE.finditer(text):
        street = m.group()
        if street.count("\n") > 1 or "\n\n" in street:
            continue
        if not m.group("name").strip() and not m.group("dir1"):
            continue  # "5 Street" alone
        end = m.end()
        units = 0
        for _ in range(2):
            u = _UNIT_RE.match(text, end)
            if not u:
                break
            end = u.end()
            units += 1
        tail_end = det._us_tail(text, end)
        wrapped = "\n" in street
        name_words = len(m.group("name").split())
        if wrapped and not (units or tail_end > end or name_words >= 2):
            continue  # a number at a line end and a street on the next line: needs more than the street
        end = tail_end
        if _STATE_END_RE.search(text[max(0, end - 4):end]):
            z = _ZIP_NEXT_LINE_RE.match(text, end)
            if z and text[max(0, end - 2):end].strip() in det._US_STATE_ABBR:
                end = z.end("zip")
        num = m.group("num")
        if _YEAR_RE.fullmatch(num) and end == m.end() and "\n" not in text[max(0, m.start() - 2):m.start()]:
            continue  # "in 2019 Main Street" needs a tail, as in the base detector
        yield Candidate(m.start(), end, "ADDRESS", True, pr, "address_street")


find_street_addresses.langs = ("en",)  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# 6: identifiers with grouped letter prefixes (English texts; v0.4, T8, K5)
# ---------------------------------------------------------------------------

# "ABC-DE-1234", "AB-CD-EF-0123": a prefix of 2 or 3 capital groups, then the digits. The plain shapes read one
# letter prefix only; such a value is an identifier only under a label noun or a column header of an id kind, and
# a cell of such a column may hold several ids ("ABC-DE-1234; ABC-DE-1235").
_V_GROUPED_PREFIX = r"[A-Z]{1,5}(?:[\-/.][A-Z]{1,5}){1,2}[\-/.]\d{3,12}"
_GROUPED_RE = re.compile(r"(?<![A-Za-z0-9])(?P<v>" + _V_GROUPED_PREFIX + r")(?![A-Za-z0-9]|[\-/.]\d)")
_GROUPED_AT_RE = re.compile(r"(?P<v>" + _V_GROUPED_PREFIX + r")(?![A-Za-z0-9]|[\-/.]\d)")
_LIST_ID_RE = re.compile(r"(?<![A-Za-z0-9])(?P<v>" + _V_GROUPED_PREFIX + r"|" + _V_PREFIXED + r")(?![A-Za-z0-9]|[\-/.]\d)")


def _grouped_layout(value: str) -> Optional[str]:
    """Layout of a grouped-prefix id for in-text propagation ("ABC-DE-1234" -> "ABC-DE-dddd"): the letter groups
    make the layout distinctive, so three digits are enough (the plain layouts need six)."""
    v = value.strip()
    if not _GROUPED_AT_RE.fullmatch(v):
        return None
    return "".join("d" if c.isdigit() else c.upper() for c in v)


def grouped_value_at(text: str, pos: int, explicit: bool = False) -> Optional[tuple[int, int]]:
    """(start, end) of a grouped-prefix id value starting at ``pos`` (spaces skipped), or None."""
    q = pos
    n = len(text)
    while q < n and text[q] in " \t ":
        q += 1
    m = _GROUPED_AT_RE.match(text, q)
    if not m or value_refused(text, m.start("v"), m.end("v"), explicit):
        return None
    return m.start("v"), m.end("v")


def find_grouped_ids(text: str) -> Iterator[Candidate]:
    """Grouped-prefix id values under a label noun (``labelled_id``) or an id column (``table_id``), the other
    ids of a cell that lists several, and the values of the same layout in the rest of the text (``layout_id``)."""
    if "-" not in text and "/" not in text and "." not in text:
        return
    if not any(c.isdigit() for c in text):
        return
    confirmed: dict[tuple[int, int], tuple[str, str]] = {}
    for s, e, kind in _cue_values(text, grouped_value_at):
        confirmed.setdefault((s, e), (kind, "labelled_id"))
    if "|" in text:
        for cells, header in det._table_rows(text):
            for i, (s, e) in enumerate(cells):
                if s >= e:
                    continue
                kind = _header_kind(det._cell_labels(cells, header, i, text))
                if kind is None:
                    continue
                cell = text[s:e]
                found = [m for m in _LIST_ID_RE.finditer(cell)]
                if not found:
                    continue
                grouped = [m for m in found if _GROUPED_AT_RE.fullmatch(m.group("v"))]
                # a plain-prefix value alone in its cell is the base rule's; a list of ids is read here whole
                take = found if len(found) >= 2 else grouped
                for m in take:
                    vs, ve = s + m.start("v"), s + m.end("v")
                    if not value_refused(text, vs, ve):
                        confirmed.setdefault((vs, ve), (kind, "table_id"))
    for (s, e), (kind, detector) in confirmed.items():
        yield Candidate(s, e, kind, True, PRIORITY[kind] if detector == "table_id" else _CUE_PRIORITY[kind], detector)
    layouts: dict[str, set[str]] = {}
    for (s, e), (kind, _detector) in confirmed.items():
        lay = _grouped_layout(text[s:e])
        if lay:
            layouts.setdefault(lay, set()).add(kind)
    for lay, kinds in layouts.items():
        if len(kinds) != 1:
            continue
        kind = next(iter(kinds))
        for m in _layout_pattern(lay).finditer(text):
            if (m.start(), m.end()) in confirmed or value_refused(text, m.start(), m.end()):
                continue
            yield Candidate(m.start(), m.end(), kind, True, _LAYOUT_PRIORITY[kind], "layout_id")


find_grouped_ids.langs = ("en",)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 7: member ids and profile handles (English texts; v0.4, T8, K10)
# ---------------------------------------------------------------------------

# A Slack-style member id written bare ("U02ABCD7EF" in a roster cell or after "Slack"): the shape of a workspace
# member id, a capital U, a zero, then 6 to 15 capitals, digits or underscores with at least one letter. The
# mention form "<@U02ABCD7EF>" is the base detector's.
_MEMBER_ID_RE = re.compile(r"(?<![A-Za-z0-9@/._\-])(?P<v>U0(?=[0-9_]*[A-Z])[A-Z0-9][A-Z0-9_]{5,14})(?![A-Za-z0-9_])")
# A profile handle after the "LinkedIn" label ("LinkedIn: jane-doe-12a", "LinkedIn profile jane-doe"): a lower-case
# token of letters, digits, "-", "_" and "." that holds a hyphen, a digit, a dot or an underscore (a name written
# in plain words after the label is not a handle), after a marker or the profile word.
_LINKEDIN_RE = re.compile(
    r"(?i)(?<![\w/.])linkedin(?:[  ]+(?P<w>profile|handle|id|url|page|username))?[  ]*(?P<m>[:=\-–—|])?"
    r"[  ]*(?-i:(?P<h>[a-z0-9][a-z0-9\-_.]{2,60}))(?![\w@])")


def find_member_ids(text: str) -> Iterator[Candidate]:
    pr = PRIORITY["USERNAME"]
    if "U0" in text:
        for m in _MEMBER_ID_RE.finditer(text):
            yield Candidate(m.start("v"), m.end("v"), "USERNAME", True, pr, "member_id")
    if "linkedin" in text.lower():
        for m in _LINKEDIN_RE.finditer(text):
            s, e = m.start("h"), m.end("h")
            while e > s and text[e - 1] in ".-_":
                e -= 1
            h = text[s:e]
            if len(h) < 3 or not (m.group("m") or m.group("w")):
                continue  # "LinkedIn and Twitter": no marker and no profile word
            if h.isalpha() and not m.group("m"):
                continue  # "LinkedIn profile updated": a plain word after the profile word, no marker
            yield Candidate(s, e, "USERNAME", True, pr, "handle")


find_member_ids.langs = ("en",)  # type: ignore[attr-defined]

DETECTORS_IDS: tuple[Callable[[str], Iterable[Candidate]], ...] = (find_ids, find_table_ids, find_street_addresses,
                                                                     find_grouped_ids, find_member_ids)
