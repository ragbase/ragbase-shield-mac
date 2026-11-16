"""Accent and case folding with an offset map back to the original text.

``fold(text)`` returns a :class:`Folded` object. Its ``text`` is lower case, without accents,
with unified apostrophes (``'``), hyphens (``-``) and quotes (``"``), with every run of white
space (spaces, non-breaking spaces, tabs, line breaks) collapsed to one space, and with
zero-width characters and soft hyphens removed. ``Folded.span(fs, fe)`` maps a folded slice back
to original offsets, so a match found on the folded text can be replaced in the original text.

``PhraseMatcher`` finds token sequences (names, whitelist phrases) in a folded text with a small
gap rule between tokens. It is shared by the whitelist and the dictionary matcher.
"""
from __future__ import annotations

import bisect
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterator, Optional

__all__ = [
    "Folded",
    "fold",
    "fold_word",
    "normalize",
    "tokens",
    "split_tokens",
    "split_windows",
    "PhraseMatcher",
    "TOKEN_RE",
]

_SPACES = frozenset(
    " \t\n\r\f\v\u0085\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008"
    "\u2009\u200a\u2028\u2029\u202f\u205f\u3000"
)
_DROP = frozenset("\u00ad\u200b\u200c\u200d\u200e\u200f\u2060\ufeff")
_APOS = frozenset("'’‘‛ʼʻ`´′＇")
_DASH = frozenset("‐‑‒–—―−⁃﹘﹣－")
_QUOTES = frozenset("«»“”„‟‹›")
_SPECIAL = {"œ": "oe", "Œ": "oe", "æ": "ae", "Æ": "ae", "ß": "ss",
            "ı": "i", "ø": "o", "Ø": "o", "ł": "l", "Ł": "l",
            "đ": "d", "Đ": "d", "ð": "d", "Þ": "th", "þ": "th"}

# OCR / look-alike confusions used by the canary scan only (applied after folding).
_CONFUSABLE = {"0": "o", "1": "l", "i": "l", "|": "l", "!": "l"}

_cache: dict[str, str] = {}


def _fold_char(ch: str) -> str:
    """Fold one character (no white space handling)."""
    hit = _cache.get(ch)
    if hit is not None:
        return hit
    if ch in _APOS:
        out = "'"
    elif ch in _DASH:
        out = "-"
    elif ch in _QUOTES:
        out = '"'
    elif ch in _SPECIAL:
        out = _SPECIAL[ch]
    elif ch < "\x80":
        out = ch.lower()
    else:
        dec = unicodedata.normalize("NFKD", ch)
        out = "".join(c for c in dec if not unicodedata.combining(c)).lower()
        out = "".join(" " if c in _SPACES else c for c in out)
    if len(_cache) < 65536:
        _cache[ch] = out
    return out


@dataclass
class Folded:
    """A folded view of ``original`` with per-character offsets back to it."""

    original: str
    text: str
    starts: list[int] = field(repr=False)
    ends: list[int] = field(repr=False)

    def span(self, fs: int, fe: int) -> tuple[int, int]:
        """Map the folded slice ``[fs, fe)`` to an original slice ``[start, end)``."""
        if fe <= fs:
            pos = self.starts[fs] if fs < len(self.starts) else len(self.original)
            return pos, pos
        return self.starts[fs], self.ends[fe - 1]

    def orig_pos(self, fpos: int) -> int:
        """Original offset of folded position ``fpos`` (end of text allowed)."""
        if fpos >= len(self.starts):
            return len(self.original)
        return self.starts[fpos]


def fold(text: str, *, confusables: bool = False) -> Folded:
    """Fold ``text`` (lower case, no accents, unified punctuation, collapsed white space).

    With ``confusables=True`` the look-alike characters 0/o, 1/l/i/| are merged, which the canary
    scan uses to catch small OCR errors.
    """
    out: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    append_out = out.append
    append_s = starts.append
    append_e = ends.append
    conf = _CONFUSABLE if confusables else None
    for i, ch in enumerate(text):
        if ch in _SPACES:
            if out and out[-1] == " ":
                ends[-1] = i + 1
            else:
                append_out(" ")
                append_s(i)
                append_e(i + 1)
            continue
        if ch in _DROP:
            continue
        f = _fold_char(ch)
        if not f:
            # A combining mark (decomposed text, NFD) folds to nothing: it belongs to the previous character, so
            # a span that ends on that character also covers its accent ("Hélène" written e + U+0301).
            if ends and out[-1] != " ":
                ends[-1] = i + 1
            continue
        if len(f) == 1 and f != " ":
            if conf is not None:
                f = conf.get(f, f)
            append_out(f)
            append_s(i)
            append_e(i + 1)
            continue
        for c in f:
            if c == " ":
                if out and out[-1] == " ":
                    ends[-1] = i + 1
                    continue
            elif conf is not None:
                c = conf.get(c, c)
            append_out(c)
            append_s(i)
            append_e(i + 1)
    return Folded(original=text, text="".join(out), starts=starts, ends=ends)


@lru_cache(maxsize=131072)
def _fold_word_cached(text: str, confusables: bool) -> str:
    return fold(text, confusables=confusables).text.strip()


def fold_word(text: str, *, confusables: bool = False) -> str:
    """Fold a short string without keeping offsets (lexicon entries, keys). Cached."""
    if len(text) > 200:
        return fold(text, confusables=confusables).text.strip()
    return _fold_word_cached(text, confusables)


def normalize(text: str) -> str:
    """Normalised key: lower case, no accents, unified apostrophes/hyphens, single spaces."""
    return fold(text).text.strip()


TOKEN_RE = re.compile(r"[^\W_]+")


def tokens(ftext: str) -> list[tuple[str, int, int]]:
    """Word tokens of a folded text as ``(token, start, end)`` in folded offsets."""
    return [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(ftext)]


# Inner word boundaries in ORIGINAL text: lower -> upper ("KervadecMaelle"), letter <-> digit ("Kervadec2026"),
# and the last capital of a run before a lower-case letter ("DUPONTJean").
_INNER_BOUNDARY_RE = re.compile(
    r"(?<=[a-zß-öø-ÿœ])(?=[A-ZÀ-ÖØ-ÞŒ])|(?<=[^\W\d_])(?=\d)|(?<=\d)(?=[^\W\d_])"
    r"|(?<=[A-ZÀ-ÖØ-ÞŒ])(?=[A-ZÀ-ÖØ-ÞŒ][a-zß-öø-ÿœ])"
)


def _boundaries(folded: Folded) -> list[int]:
    """Folded positions where a glued word starts, from the case and digits of the original text."""
    orig = folded.original
    if not orig:
        return []
    starts = folded.starts
    out: list[int] = []
    for m in _INNER_BOUNDARY_RE.finditer(orig):
        fp = bisect.bisect_left(starts, m.start())
        if fp < len(starts) and starts[fp] == m.start() and fp > 0 and starts[fp - 1] != starts[fp]:
            out.append(fp)
    return out


def split_tokens(folded: Folded, toks: list[tuple[str, int, int]]) -> Optional[list[tuple[str, int, int]]]:
    """Tokens split inside glued words, using the case of the ORIGINAL text: "KervadecMaelle" -> "kervadec",
    "maelle"; "Kervadec2026" -> "kervadec", "2026"; "DUPONTJean" -> "dupont", "jean". Returns None when no token
    had an inner boundary (the caller then has nothing more to match).

    Used as a second view for dictionary matching only, so "MacArthur" still matches as one word too.
    """
    cuts = _boundaries(folded)
    if not cuts:
        return None
    out: list[tuple[str, int, int]] = []
    changed = False
    text = folded.text
    for tok, fs, fe in toks:
        i = bisect.bisect_right(cuts, fs)
        if i >= len(cuts) or cuts[i] >= fe:
            out.append((tok, fs, fe))
            continue
        cut = fs
        while i < len(cuts) and cuts[i] < fe:
            out.append((text[cut:cuts[i]], cut, cuts[i]))
            cut = cuts[i]
            i += 1
        out.append((text[cut:fe], cut, fe))
        changed = True
    return out if changed else None


def split_windows(folded: Folded, toks: list[tuple[str, int, int]], radius: int = 3
                  ) -> list[list[tuple[str, int, int]]]:
    """Small token lists around each glued word, for the second dictionary view: matching only there costs
    little, even on a large text with a few glued words."""
    cuts = _boundaries(folded)
    if not cuts:
        return []
    starts = [t[1] for t in toks]
    glued: list[int] = []
    for c in cuts:
        k = bisect.bisect_right(starts, c) - 1
        if k >= 0 and toks[k][1] < c < toks[k][2] and (not glued or glued[-1] != k):
            glued.append(k)
    if not glued:
        return []
    windows: list[list[tuple[str, int, int]]] = []
    text = folded.text
    prev_hi = -1
    for k in glued:
        if k < prev_hi:
            continue  # this glued word is already inside the previous window
        lo, hi = max(0, k - radius), min(len(toks), k + radius + 1)
        prev_hi = hi
        window: list[tuple[str, int, int]] = []
        for tok, fs, fe in toks[lo:hi]:
            i = bisect.bisect_right(cuts, fs)
            cut = fs
            while i < len(cuts) and cuts[i] < fe:
                window.append((text[cut:cuts[i]], cut, cuts[i]))
                cut = cuts[i]
                i += 1
            window.append((text[cut:fe], cut, fe))
        windows.append(window)
    return windows


def phrase_tokens(text: str, *, confusables: bool = False) -> tuple[str, ...]:
    """Tokens of a phrase, folded (used to register phrases in a :class:`PhraseMatcher`)."""
    return tuple(t for t, _, _ in tokens(fold(text, confusables=confusables).text))


class PhraseMatcher:
    """Find registered token sequences in a folded text.

    Consecutive tokens of a phrase must be separated in the text by a short gap that matches
    ``gap`` (by default spaces, dots, hyphens and apostrophes only, at most 3 characters), so
    "J.-P. Dupont" matches the phrase ("j", "p", "dupont") but "Dupont, Jean" does not match
    ("dupont", "jean").
    """

    def __init__(self, gap: str = r"[ .'\-]{0,3}"):
        self._index: dict[str, list[tuple[tuple[str, ...], object]]] = {}
        self._gap = re.compile(gap)
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, phrase: tuple[str, ...], payload: object) -> None:
        if not phrase:
            return
        self._index.setdefault(phrase[0], []).append((phrase, payload))
        self._size += 1

    def find_all(
        self, ftext: str, toks: list[tuple[str, int, int]], first: Optional[int] = None
    ) -> Iterator[tuple[int, int, object]]:
        """Yield ``(i, j, payload)`` for every phrase matching tokens ``toks[i:j]``."""
        index = self._index
        gap = self._gap.fullmatch
        n_toks = len(toks)
        rng = range(n_toks) if first is None else (first,)
        for i in rng:
            cands = index.get(toks[i][0])
            if not cands:
                continue
            for phrase, payload in cands:
                n = len(phrase)
                if i + n > n_toks:
                    continue
                ok = True
                for k in range(1, n):
                    nxt = toks[i + k]
                    if nxt[0] != phrase[k] or gap(ftext, toks[i + k - 1][2], nxt[1]) is None:
                        ok = False
                        break
                if ok:
                    yield i, i + n, payload
