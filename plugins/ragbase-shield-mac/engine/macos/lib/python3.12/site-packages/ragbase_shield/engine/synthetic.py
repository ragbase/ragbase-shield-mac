"""Synthetic replacement mode: realistic pseudonyms in place of codes (v0.2, design section 4.5).

A :class:`Synthesizer` gives every entity of one store a fake value that is deterministic from a secret seed
and the entity code (``HMAC(seed, code)``), coherent (one fake identity per person, the same fake surname for
the people who share a real surname, a fake e-mail built from the fake person), and shape preserving (case,
number of given names, separators of a phone or an identifier, written form of a date). Nothing of the real
value enters the generator except its kind, the sex of its first name, its surface form and those structural
relations.

Collision rules (v0.3, SYNTH-LEAK). A fake value never collides with the dossier: it never equals a real value
of any kind, never reuses a word of a real value (a fake first name "Nathan" while a real "Nathan" works in the
dossier would hide that person: the masker takes a known fake surface for fake text), never equals another fake
value, never holds a whitelist word, and keeps the number of words of the real name with every replaced word
starting with another letter (an initials-only mention can then never read as the real text). For an address,
an e-mail or a web address every distinctive word of the fake counts, changed or not (street types, unit words
and host words excepted): a fake address never keeps the real street or town. The check runs
against the whole store at every call, not only at the first draw: a fake drawn before a colliding real value
was seen is re-drawn as soon as the Synthesizer is asked for it again (:meth:`Synthesizer.value_for`,
:meth:`Synthesizer.collisions_with_store`, :meth:`Synthesizer.redraw_collisions`). A caller may also
:meth:`Synthesizer.reserve` surfaces it detected but did not register. Fail closed: when no collision-free value
exists, :meth:`Synthesizer.value_for` raises :class:`SyntheticCollision`, and when a rendering would still show
the real text :meth:`Synthesizer.render` raises :class:`SyntheticLeak`; the masker then writes the code. The
messages of these errors carry codes and reasons only, never a value.

Honest wording: this is pseudonymisation with realistic pseudonyms, reversible through the vault, not
anonymisation. The fake identifiers without a reserved fiction range (NIR, SIREN, SIRET, VAT, account numbers,
addresses, phone numbers of other countries) may collide with a real number outside the dossier:
:attr:`Synthesizer.collisions` counts them so that a run can report it.

The masker (``engine/masker.py``) owns the pipeline: it asks :meth:`Synthesizer.value_for` for the fake value,
:meth:`Synthesizer.render` for the part of it that replaces one mention (surname alone, initials, upper case...)
and :meth:`Synthesizer.restore` for the inverse when a model answer comes back.
"""
from __future__ import annotations

import hashlib
import hmac
import re
from collections import Counter
from datetime import date, timedelta
from functools import lru_cache
from typing import Any, Iterable, Optional

from . import detectors as det
from . import lexicon as lx
from . import names as nm
from .normalize import fold_word, normalize
from .types import Entity

__all__ = ["Synthesizer", "SyntheticCollision", "SyntheticError", "SyntheticLeak", "SYNTHETIC_TLD",
           "shift_date_text", "synthetic_lists", "UNVERIFIED_KINDS", "mention_word_fixed",
           "unchanged_mention_word"]

#: RFC 2606 reserved top-level domain: a synthetic e-mail or web address never resolves.
SYNTHETIC_TLD = "example"
#: Kinds whose synthetic value has no reserved fiction range (a collision with a real number is possible).
UNVERIFIED_KINDS = frozenset({"NIR", "SIREN", "SIRET", "VAT", "ADDRESS", "ACCOUNT_NUMBER", "EMPLOYEE_ID",
                              "ID_CODE", "CASE_REF", "USERNAME"})
_MAX_TRIES = 48
_RUN_RE = re.compile(r"[^\W\d_]+")
_ALNUM_RE = re.compile(r"[0-9A-Za-z]")
_MAIL_PROVIDERS = frozenset({"gmail", "googlemail", "outlook", "hotmail", "live", "yahoo", "icloud", "me", "orange",
                             "wanadoo", "free", "sfr", "laposte", "bbox", "protonmail", "proton", "aol", "gmx",
                             "msn", "neuf", "numericable"})
_NAME_KINDS = ("PERSON", "ORG")
#: Kinds whose fake is built from word pools (streets, towns, names, company words): every distinctive word of
#: the fake is checked against the dossier, changed or not (a fake address must not keep the real street).
_WORD_KINDS = ("ADDRESS", "EMAIL", "URL")
#: Words of an address or e-mail that carry nothing of the dossier (never a collision).
_NEUTRAL_WORDS = frozenset({"http", "https", "www", "mailto", "ftp", "com", "net", "org", "html", SYNTHETIC_TLD}
                           | _MAIL_PROVIDERS)
#: Street types, unit words and compass points of both languages: generic, so a fake address may share them
#: with a real one ("Road", "rue"). The English street words of the address detector are added at run time.
_ADDRESS_WORDS = frozenset({
    "rue", "avenue", "boulevard", "place", "allee", "impasse", "chemin", "quai", "route", "cours", "square",
    "passage", "lieu", "dit", "residence", "villa", "cite", "hameau", "parvis", "esplanade", "promenade", "rond",
    "point", "sentier", "faubourg", "montee", "traverse", "clos", "domaine", "lotissement", "chaussee", "venelle",
    "ruelle", "voie", "zone", "industrielle", "artisanale", "commerciale", "activites", "parc", "batiment",
    "escalier", "etage", "appartement", "cedex", "france", "street", "road", "lane", "drive", "close", "crescent",
    "terrace", "gardens", "mews", "grove", "court", "parkway", "highway", "suite", "floor", "unit", "building",
    "apartment", "north", "south", "east", "west", "upper", "lower", "little", "great", "saint", "sainte",
})
_VOWELS = "aeiou"
_CONSONANTS = "bcdfghklmnprstvz"


class SyntheticError(RuntimeError):
    """The Synthesizer could not give a safe value. The message holds codes and reasons only, never a value."""


class SyntheticCollision(SyntheticError):
    """No collision-free fake value exists for an entity (fail closed: the caller writes the code)."""


class SyntheticLeak(SyntheticError):
    """A rendering would still show the real text (fail closed: the caller writes the code)."""


class _Index:
    """What the store holds, folded, for the collision rules (rebuilt when the store or the cache changes)."""

    __slots__ = ("key", "real_norms", "real_tokens", "real_groups", "by_len", "fakes", "fake_norms", "surnames",
                 "real_initials", "real_nicks", "edit_keys", "real_sounds")

    def __init__(self, key: object):
        self.key = key
        self.real_norms: set[str] = set()      # normalised real values (every kind) and reserved surfaces
        self.real_tokens: set[str] = set()     # folded words (2+ letters) of every real value and reserved surface
        self.real_groups: set[str] = set()     # alphanumeric groups (3+ chars) of every real value ("513" of a phone)
        self.by_len: dict[int, list[str]] = {}  # real tokens of 5+ letters by length (one-edit rule)
        self.fakes: dict[str, str] = {}        # code -> fake value
        self.fake_norms: dict[str, str] = {}   # normalised fake value -> code
        self.surnames: dict[str, tuple[str, str]] = {}  # real surname key -> (code, fake surname)
        self.real_initials: set[str] = set()   # one-letter words of every real value and reserved surface (v0.3.1)
        self.real_nicks: set[str] = set()      # nicknames and short forms of the real given names (v0.3.1)
        self.edit_keys: dict[str, list[str]] = {}  # _del_keys -> real tokens of 5+ letters (v0.4 S0, class D)
        self.real_sounds: set[str] = set()     # _sound keys of every real token of 4+ letters (v0.4 S0, class D)


class _Draw:
    """The sets one draw works with: ``hard`` words never used, ``soft`` words (of other fakes) avoided when the
    pool allows, ``shared`` the fake surname of a person with the same real surname (or None); ``ini_hard`` the
    one-letter words of the dossier a fake initial never takes, ``ini_soft`` those of other fakes, avoided when a
    letter is left (v0.3.1)."""

    __slots__ = ("hard", "soft", "shared", "allowed", "ini_hard", "ini_soft")

    def __init__(self, hard: set[str], soft: set[str], shared: Optional[str],
                 ini_hard: Iterable[str] = (), ini_soft: Iterable[str] = ()):
        self.hard, self.soft, self.shared = hard, soft, shared
        self.allowed: set[str] = {fold_word(r) for r, _s, _e in _runs(shared)} if shared else set()
        self.ini_hard: set[str] = set(ini_hard)
        self.ini_soft: set[str] = set(ini_soft)

_MONTHS_FR = ("janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre",
              "novembre", "décembre")
_MONTHS_FR_ABBR = ("janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc.")
_MONTHS_EN = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December")
_MONTHS_EN_ABBR = ("Jan.", "Feb.", "Mar.", "Apr.", "May", "Jun.", "Jul.", "Aug.", "Sep.", "Oct.", "Nov.", "Dec.")
_MONTH_INDEX: dict[str, tuple[int, str]] = {}
# abbreviations first: a name that is both ("mars", "mai", "May") is read as the full name
for _table, _names in (("fr_abbr", _MONTHS_FR_ABBR), ("en_abbr", _MONTHS_EN_ABBR), ("fr", _MONTHS_FR),
                       ("en", _MONTHS_EN)):
    for _i, _n in enumerate(_names):
        _MONTH_INDEX[fold_word(_n).rstrip(".")] = (_i + 1, _table)
_MONTH_INDEX.setdefault("sept", (9, "fr_abbr"))
_MONTH_INDEX.setdefault("fev", (2, "fr_abbr"))
_MONTH_INDEX.setdefault("sep", (9, "en_abbr"))
_MONTH_TABLES = {"fr": _MONTHS_FR, "fr_abbr": _MONTHS_FR_ABBR, "en": _MONTHS_EN, "en_abbr": _MONTHS_EN_ABBR}

_DATE_NUM_RE = re.compile(r"(?<!\d)(\d{1,2})(\s?)([/.\-])(\s?)(\d{1,2})(\s?)([/.\-])(\s?)(\d{2,4})(?!\d)")
_DATE_ISO_RE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_DATE_FR_RE = re.compile(r"(?<!\d)(\d{1,2})(er|ER)?(\s+)([^\W\d_]+)\.?(\s+)(\d{4})(?!\d)")
_DATE_EN_RE = re.compile(r"(?<![^\W\d_])([^\W\d_]+)\.?(\s+)(\d{1,2})(,?\s+)(\d{4})(?!\d)")


# --------------------------------------------------------------------------------------------- random stream

class _Rng:
    """Deterministic stream of numbers from ``HMAC-SHA256(seed, labels, counter)``."""

    def __init__(self, seed: bytes, *labels: str):
        self._key = seed
        self._label = "\x1f".join(labels).encode("utf-8")
        self._n = 0

    def _next(self) -> int:
        h = hmac.new(self._key, self._label + b"\x00" + self._n.to_bytes(4, "big"), hashlib.sha256).digest()
        self._n += 1
        return int.from_bytes(h[:8], "big")

    def below(self, n: int) -> int:
        return self._next() % max(1, n)

    def between(self, lo: int, hi: int) -> int:
        return lo + self.below(hi - lo + 1)

    def pick(self, seq):
        return seq[self.below(len(seq))]

    def digits(self, n: int) -> str:
        return "".join(str(self.below(10)) for _ in range(n))

    def letters(self, n: int, upper: bool = True) -> str:
        out = "".join(chr(65 + self.below(26)) for _ in range(n))
        return out if upper else out.lower()


# --------------------------------------------------------------------------------------------- lists

def _read_pairs(name: str) -> list[tuple[str, str]]:
    out = []
    for line in lx.read_lines(name):
        a, _sep, b = line.partition("\t")
        out.append((a.strip(), b.strip()))
    return out


@lru_cache(maxsize=4)
def synthetic_lists(lang: str) -> dict[str, Any]:
    """Word lists of one language for the generator (``"fr"`` or ``"en"``; a missing file falls back to fr)."""
    lang = "en" if lang == "en" else "fr"

    def lines(stem: str) -> list[str]:
        try:
            return lx.read_lines(f"{stem}_{lang}.txt")
        except (FileNotFoundError, OSError):
            return lx.read_lines(f"{stem}_fr.txt")

    def pairs(stem: str) -> list[tuple[str, str]]:
        try:
            return _read_pairs(f"{stem}_{lang}.txt")
        except (FileNotFoundError, OSError):
            return _read_pairs(f"{stem}_fr.txt")

    firsts = pairs("synthetic_first")
    afn = lx.all_first_names()
    lasts = [w for w in lines("synthetic_last")
             if not nm._lexicon_word(fold_word(w)) and fold_word(w) not in afn and fold_word(w) not in nm.ORG_GENERIC]
    return {
        "first_f": [n for n, s in firsts if s.upper() == "F"],
        "first_m": [n for n, s in firsts if s.upper() == "M"],
        "sex": {fold_word(n): s.upper() for n, s in firsts},
        "last": lasts,
        "org": lines("synthetic_org"),
        "streets": lines("synthetic_streets"),
        "towns": pairs("synthetic_towns"),
    }


@lru_cache(maxsize=1)
def _sex_table() -> dict[str, str]:
    table: dict[str, str] = {}
    for lang in ("en", "fr"):
        table.update(synthetic_lists(lang)["sex"])
    extra = getattr(lx, "first_name_sex", None)
    if callable(extra):
        try:
            for k, v in dict(extra()).items():
                table.setdefault(fold_word(k), str(v).upper()[:1])
        except Exception:  # noqa: BLE001 - optional bundle of another builder
            pass
    return table


@lru_cache(maxsize=1)
def _whitelist_words() -> frozenset[str]:
    return frozenset(p for p in lx.whitelist_phrases() if " " not in p)


@lru_cache(maxsize=1)
def _generic_words() -> frozenset[str]:
    """Words of an address, an e-mail or a web address that identify nothing: scheme and host words, public mail
    providers, street types and unit words (the detector's English street words included)."""
    out = set(_NEUTRAL_WORDS) | set(_ADDRESS_WORDS)
    out.update(fold_word(w) for w in getattr(det, "_EN_STREET_WORDS", ()))
    return frozenset(out)


def _ascii(word: str) -> str:
    return re.sub(r"[^a-z0-9]", "", fold_word(word))


def _core(text: str) -> str:
    """Letters and digits of ``text``, folded: what two mentions share when they read the same."""
    return re.sub(r"[^a-z0-9]", "", fold_word(text))


def ent_is_name(ent: Entity) -> bool:
    return ent.kind in _NAME_KINDS


@lru_cache(maxsize=1)
def _extension_first_names() -> tuple[str, ...]:
    """Public first names outside the fake pools (display case), used only when a pool is exhausted by the
    real names of a large dossier: a realistic name of unknown sex beats a made-up word."""
    out = []
    for w in sorted(lx.all_first_names()):
        if 3 <= len(w) <= 12 and w.isalpha() and not nm._lexicon_word(w) and w not in nm.ORG_GENERIC:
            out.append(w[:1].upper() + w[1:])
    return tuple(out)


def _pronounceable(rng: _Rng, length: int, avoid_initial: str = "") -> str:
    """A made-up word of ``length`` letters (consonant and vowel alternation): the last resort of the generator,
    never a real name but readable, with an initial other than ``avoid_initial``."""
    for _ in range(8):
        start = rng.below(2)
        out = []
        for i in range(length):
            pool = _CONSONANTS if (i + start) % 2 == 0 else _VOWELS
            out.append(pool[rng.below(len(pool))])
        word = "".join(out)
        if not avoid_initial or word[:1] != avoid_initial:
            return word
    first = next(c for c in _CONSONANTS + _VOWELS if c != avoid_initial)
    return first + word[1:]


def _real_word_inside(real_words: Iterable[str], fake: str) -> bool:
    """True when a replaced real word (folded, 4+ letters) still reads inside the letters of ``fake`` joined
    ("vale" in "Springvale", or across two fake words): a reader, and the masker's own leak check, would see it."""
    body = _core(fake)
    return any(len(w) >= 4 and w in body for w in real_words)


def _letter_other_than(rng: _Rng, letter: str) -> str:
    """One capital letter that is not ``letter`` (a fake initial never equals the real one)."""
    for _ in range(8):
        c = rng.letters(1)
        if c.lower() != letter.lower():
            return c
    return "Z" if letter.lower() != "z" else "Q"


def _free_initial(rng: _Rng, letter: str, hard: Iterable[str] = (), soft: Iterable[str] = ()) -> Optional[str]:
    """One capital letter for the fake of a real one-letter word (v0.3.1): never the real ``letter``, never a
    one-letter word of the dossier (``hard``: an initial a text writes), and not the initial of another fake
    (``soft``) while a letter is left. The order of the letters is drawn from ``rng``. None when every letter is
    taken (the caller fails closed)."""
    taken = {h.lower() for h in hard} | {letter.lower()}
    others = {s.lower() for s in soft}
    start, step = rng.below(26), (3, 5, 7, 9, 11)[rng.below(5)]  # each step is prime to 26: every letter comes
    order = [chr(97 + (start + i * step) % 26) for i in range(26)]
    for c in order:
        if c not in taken and c not in others:
            return c.upper()
    for c in order:
        if c not in taken:
            return c.upper()
    return None


#: Mention needs of the masker that name a person by a given-name form (a first name or a nickname alone, a
#: nickname with the family name): a word of such a mention that maps to no word of the real value is written as
#: the fake's first given name (v0.3.1). v0.4 (T2): "dir_slot", a short form in a greeting or sign-off slot.
_GIVEN_NEEDS = frozenset({"first", "salutation", "nick", "mid", "casual", "dir", "dir_slot", "joined", "comma"})


@lru_cache(maxsize=8192)
def _given_forms(first: str) -> frozenset[str]:
    """Nicknames and short forms of a folded given name, from the sources the masker's resolution reads: the
    French and English tables and the prefix forms ("reggie" of "reginald", "millie" of "millicent") (v0.3.1)."""
    from .masker import lx_nicknames  # local import: masker imports this module

    out: set[str] = set()
    try:
        out.update(lx_nicknames(first))
    except Exception:  # noqa: BLE001 - no table: no form
        pass
    try:
        out.update(nm._nick_forms(first))
    except Exception:  # noqa: BLE001
        pass
    out.discard(first)
    return frozenset(f for f in out if f)


def _leading_particle(run: str, surface: str, start: int) -> bool:
    """True for a capitalised particle of two or more letters that opens a name ("Al Brightwater", "Del
    Ostrevane"): it may be a given name (titles before it do not count) (v0.3.1)."""
    f = fold_word(run)
    if len(f) < 2 or not run[:1].isupper() or f not in nm.PARTICLES:
        return False
    for r, _s, _e in _runs(surface[:start]):
        if not nm.is_title_word(r):
            return False
    return True


def _trailing_particle(run: str, surface: str, start: int) -> bool:
    """True for a particle word of two or more letters that ends the name: no word of two or more letters follows
    it in ``surface`` ("Del" alone, or a quoted nickname "Del" cut at the end of a stored value). A particle always
    comes before another name word, so a last word is a given name or a nickname, never a particle (v0.3.1)."""
    f = fold_word(run)
    if len(f) < 2 or f not in nm.PARTICLES or start < 0:
        return False
    return not any(len(r) >= 2 for r, _s, _e in _runs(surface[start + len(run):]))


def _capital_initial(run: str, surface: str, start: int) -> bool:
    """A capital one-letter particle not followed by an apostrophe ("D." of "D. Quillmore", "Y Ostrevane"): an
    initial, never the particle "d'" or "y" (v0.3.1)."""
    return len(run) == 1 and run.isupper() and fold_word(run) in nm.PARTICLES \
        and surface[start + 1:start + 2] not in ("'", "’")


def mention_word_fixed(kind: str, run: str, surface: str = "", start: int = -1, value: str = "") -> bool:
    """True when the word ``run`` (at ``start`` in ``surface``) of a mention may be written unchanged in synthetic
    mode (v0.3.1): a person's title, particle or generational word, a company's legal form or generic word, or the
    "s" of a possessive ("Dupontel's"). Any other word of a mention written unchanged is a leak, whatever its
    length. A capitalised particle that opens the mention ("Al", "Del") is fixed only when the real ``value`` holds
    it as a particle ("De Kestbury" of "Maelle de Kestbury"), never when it is a given name or a nickname."""
    f = fold_word(run)
    if not f:
        return True
    if f == "s" and 0 < start <= len(surface) and surface[start - 1] in "'’":
        return True
    if surface and start >= 0 and _trailing_particle(run, surface, start):
        return False  # a particle never ends a name: "Del" alone is a nickname
    if surface and start >= 0 and _leading_particle(run, surface, start):
        return any(fold_word(r) == f and not _leading_particle(r, value, s) and not _trailing_particle(r, value, s)
                   for r, s, _e in _runs(value))
    if surface and start >= 0 and _capital_initial(run, surface, start):
        return False
    if Synthesizer._person_fixed(run):
        return True
    return kind == "ORG" and Synthesizer._org_fixed(f)


def unchanged_mention_word(kind: str, surface: str, rep: str, value: str = "") -> bool:
    """True when the rendering ``rep`` writes a word of the mention ``surface`` unchanged and that word is not a
    fixed word (:func:`mention_word_fixed`; ``value``: the real value) (v0.3.1): a word of two or more letters
    anywhere in ``rep``, a one-letter word (an initial) at its own place (a fake initial may equal another initial
    of the mention: "B. O." written "I. B.")."""
    s_runs, r_runs = _runs(surface), _runs(rep)
    written = {fold_word(r) for r, _s, _e in r_runs}
    same_shape = len(s_runs) == len(r_runs)
    for i, (run, s, _e) in enumerate(s_runs):
        f = fold_word(run)
        if len(f) == 1 and same_shape:
            unchanged = fold_word(r_runs[i][0]) == f
            if unchanged and any(len(r) >= 2 and fold_word(r)[:1] == f and mention_word_fixed(kind, r, value, vs, value)
                                 for r, vs, _ve in _runs(value)):
                continue  # the initial of a fixed word the fake keeps ("J." of "Jr." in "D. A. F. J.")
        else:
            unchanged = f in written
        if unchanged and not mention_word_fixed(kind, run, surface, s, value):
            return True
    return False


# --------------------------------------------------------------------------------------------- dates

def shift_date_text(text: str, days: int) -> Optional[str]:
    """``text`` with its date moved by ``days``, in the same written form ("14/03/1979", "14.03.79", "1979-03-14",
    "1er mars 1979", "March 14, 1979"). None when no date form is recognised."""
    m = _DATE_ISO_RE.search(text)
    if m:
        d = _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if d is None:
            return None
        n = d + timedelta(days=days)
        return text[:m.start()] + f"{n.year:04d}-{n.month:02d}-{n.day:02d}" + text[m.end():]
    m = _DATE_NUM_RE.search(text)
    if m:
        day, mon, year = int(m.group(1)), int(m.group(5)), int(m.group(9))
        yy = len(m.group(9)) == 2
        d = _safe_date(year + (1900 if year > 30 else 2000) if yy else year, mon, day)
        if d is None:
            return None
        n = d + timedelta(days=days)
        w1, w2 = len(m.group(1)), len(m.group(5))
        ys = f"{n.year % 100:02d}" if yy else f"{n.year:04d}"
        out = f"{n.day:0{w1}d}{m.group(2)}{m.group(3)}{m.group(4)}{n.month:0{w2}d}{m.group(6)}{m.group(7)}{m.group(8)}{ys}"
        return text[:m.start()] + out + text[m.end():]
    m = _DATE_FR_RE.search(text)
    if m and fold_word(m.group(4)).rstrip(".") in _MONTH_INDEX:
        mon, table = _MONTH_INDEX[fold_word(m.group(4)).rstrip(".")]
        d = _safe_date(int(m.group(6)), mon, int(m.group(1)))
        if d is None:
            return None
        n = d + timedelta(days=days)
        name = _month_name(n.month, table, m.group(4), text[m.end(4):m.end(4) + 1] == ".")
        day = f"{n.day:02d}" if m.group(1).startswith("0") else str(n.day)
        if n.day == 1 and m.group(2):
            day = "1" + m.group(2)
        return text[:m.start()] + f"{day}{m.group(3)}{name}{m.group(5)}{n.year:04d}" + text[m.end():]
    m = _DATE_EN_RE.search(text)
    if m and fold_word(m.group(1)).rstrip(".") in _MONTH_INDEX:
        mon, table = _MONTH_INDEX[fold_word(m.group(1)).rstrip(".")]
        d = _safe_date(int(m.group(5)), mon, int(m.group(3)))
        if d is None:
            return None
        n = d + timedelta(days=days)
        name = _month_name(n.month, table, m.group(1), text[m.end(1):m.end(1) + 1] == ".")
        return text[:m.start()] + f"{name}{m.group(2)}{n.day}{m.group(4)}{n.year:04d}" + text[m.end():]
    return None


def _safe_date(y: int, mo: int, d: int) -> Optional[date]:
    try:
        return date(y, mo, d)
    except ValueError:
        return None


def _month_name(month: int, table: str, original: str, dotted: bool) -> str:
    name = _MONTH_TABLES[table][month - 1]
    if table.endswith("abbr") and (dotted or original.endswith(".")) and name.endswith("."):
        name = name[:-1]  # the dot of the text stays in place
    elif table.endswith("abbr"):
        name = name.rstrip(".")
    if original.isupper() and len(original) > 1:
        return name.upper()
    if original[:1].isupper():
        return name[:1].upper() + name[1:]
    return name.lower()


# --------------------------------------------------------------------------------------------- helpers

def _runs(value: str) -> list[tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in _RUN_RE.finditer(value)]


_GROUP_RE = re.compile(r"[0-9A-Za-z]{2,}")


def _groups(value: str) -> set[str]:
    """Alphanumeric groups of 2+ characters of ``value``, folded ("208", "555", "0163" of a phone; "md" of an
    address, "ab", "1042" of a code): what a short fake, or the initials of a fake name, must never equal, or the
    masker would read the fake inside the real value and drop the real value."""
    return {fold_word(m.group()) for m in _GROUP_RE.finditer(fold_word(value))}


def _alpha_parts(text: str) -> set[str]:
    """Folded letter runs of ``text``, whatever their length (the parts of an e-mail local part, initials included)
    (v0.4 S0)."""
    return {fold_word(r) for r, _s, _e in _runs(text)}


def _del_keys(word: str) -> set[str]:
    """``word`` and every form of it with one letter less (v0.4 S0). When two words are one edit apart (one
    substitution, insertion, deletion or swap of two neighbouring letters, at any place) their sets meet; the sets of
    a few pairs two edits apart meet too, so a meeting is only a candidate that :func:`_one_edit` then checks."""
    return {word} | {word[:i] + word[i + 1:] for i in range(len(word))}


def _one_edit(a: str, b: str) -> bool:
    """True when ``a`` and ``b`` differ by exactly one edit at ANY place, the first letter included (one
    substitution, insertion, deletion, or the swap of two neighbouring letters). The masker's ``_dl1`` asks for the
    same first letter; a fake must also never be "Wandrell" for a real "Vandrell" (v0.4 S0, class D)."""
    if a == b:
        return False
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    i = 0
    while i < min(la, lb) and a[i] == b[i]:
        i += 1
    if la == lb:
        return a[i + 1:] == b[i + 1:] or (i + 1 < la and a[i] == b[i + 1] and a[i + 1] == b[i]
                                          and a[i + 2:] == b[i + 2:])
    if la > lb:
        return a[i + 1:] == b[i:]
    return a[i:] == b[i + 1:]


#: Spelling rules that make two words with the same sound read the same (order matters): "Korvath" and
#: "Corvath", "Brynde" and "Brinde", "Phelsar" and "Felsar" (invented words) (v0.4 S0, class D).
_SOUND_RULES = tuple((re.compile(p), r) for p, r in (
    (r"ph", "f"), (r"gh", "g"), (r"ck", "k"), (r"qu", "k"), (r"sch", "sk"), (r"c(?=[eiy])", "s"), (r"[cq]", "k"),
    (r"x", "ks"), (r"z", "s"), (r"y", "i"), (r"(?<=[^aeiou])h", ""), (r"(.)\1+", r"\1")))


def _sound(word: str) -> str:
    """A rough sound key of a word of 4+ letters ("" for a shorter word): two words with the same key sound the
    same (v0.4 S0, class D)."""
    w = re.sub(r"[^a-z]", "", fold_word(word))
    if len(w) < 4:
        return ""
    for rx, rep in _SOUND_RULES:
        w = rx.sub(rep, w)
    if len(w) > 4 and w.endswith("e"):
        w = w[:-1]
    return w


@lru_cache(maxsize=1)
def _public_labels() -> frozenset[str]:
    """Public letter labels of a reference number (folded): the acronyms of the lexicon's stoplist (ECLI, PV, CA...)
    and the court and road labels of the detectors ("RG", "TJ"). An identifier keeps such a label in synthetic mode;
    any other letter code of an identifier (an employer, bank or company code) is replaced (v0.4 S0, class K)."""
    out = set(lx.acronym_stoplist())
    out.update(fold_word(x) for x in getattr(det, "_CAD_NOT_SECTION", ()))
    return frozenset(x for x in out if x)


#: Kinds with a generator of their own. Every other kind (EMPLOYEE_ID, ACCOUNT_NUMBER, ID_CODE, CASE_REF,
#: USERNAME...) gets a fake of the same shape (:func:`_shape`), whose letter codes are replaced (v0.4 S0, class K).
_OWN_GENERATOR_KINDS = frozenset({"PERSON", "ORG", "EMAIL", "PHONE", "IBAN", "CARD", "NIR", "SIREN", "SIRET", "VAT",
                                  "URL", "ADDRESS", "BIRTHDATE"})


def _overlay(surface: str, chars: str) -> Optional[str]:
    """``surface`` with its letters and digits replaced in order by ``chars`` (separators kept). None when the
    counts differ."""
    positions = [m.start() for m in _ALNUM_RE.finditer(surface)]
    if len(positions) != len(chars):
        return None
    out = list(surface)
    for p, c in zip(positions, chars):
        out[p] = c
    return "".join(out)


def _case_like(word: str, fake: str) -> str:
    """``fake`` in the case of ``word`` (upper, capitalised or lower)."""
    letters = [c for c in word if c.isalpha()]
    if len(letters) > 1 and all(c.isupper() for c in letters):
        return fake.upper()
    if word[:1].isupper():
        rest = fake[1:]
        if sum(c.isupper() for c in rest) > 1 and not any(c.islower() for c in rest):
            rest = rest.lower()  # the stored fake is in capitals: "PRÉVOT" -> "Prévot"
        return fake[:1].upper() + rest
    if letters and all(c.islower() for c in letters):
        return fake.lower()
    return fake


def _shape(surface: str, rng: _Rng, keep_prefix: bool = True, prefix: Any = None) -> str:
    """Same length and separator positions, digits and letters randomised (class and case kept). A leading
    alphabetic label of at most five letters before a separator ("RG 24/0012", "ECLI:...") stays. With ``prefix``
    (a function of the label, v0.4 S0 class K) that label is written as ``prefix(label)`` instead, unless it is one
    letter or a public label (:func:`_public_labels`): an employer, bank or company code never stays."""
    out = []
    keep = 0
    head = ""
    if keep_prefix:
        m = re.match(r"([^\W\d_]{1,5})(?=[^\w]|\d)", surface)
        if m and any(c.isdigit() for c in surface[m.end():]):
            keep = m.end()
            label = m.group(1)
            if prefix is not None and len(label) >= 2 and fold_word(label) not in _public_labels():
                head = prefix(label)
                if len(head) != keep:  # never the real label: random letters of the same case instead
                    head = "".join(rng.letters(1, upper=c.isupper()) for c in label)
    for i, ch in enumerate(surface):
        if i < keep:
            out.append(head[i] if head else ch)
        elif ch.isdigit():
            out.append(str(rng.below(10)))
        elif ch.isalpha():
            out.append(rng.letters(1, upper=ch.isupper()))
        else:
            out.append(ch)
    return "".join(out)


def _fr_rib_key(bank: str, branch: str, account: str) -> str:
    conv = "".join(str((ord(c) - 55) % 10) if c.isalpha() else c for c in account.upper())
    if account.isdigit():
        conv = account
    key = 97 - ((89 * int(bank) + 15 * int(branch) + 3 * int(conv)) % 97)
    return f"{key:02d}"


# --------------------------------------------------------------------------------------------- synthesizer

class Synthesizer:
    """Fake values for the entities of one store.

    ``seed``: 32 secret bytes (the vault derives them from the dossier key). ``lang``: language of the fake
    names ("fr" or "en"). ``store``: the entity store; when it has ``synthetic(code)`` / ``set_synthetic(code,
    value)`` the fake values are kept there, else in ``cache`` (a dict the masker owns). ``gender``: "keep" (the
    fake first name has the sex of the real one when the lexicon knows it) or "random".
    """

    def __init__(self, seed: bytes, lang: str = "fr", store: Any = None, *, cache: Optional[dict] = None,
                 gender: str = "keep"):
        if not isinstance(seed, (bytes, bytearray)) or len(seed) < 16:
            raise ValueError("synthetic seed must hold at least 16 bytes")
        self.seed = bytes(seed)
        self.lang = "en" if lang == "en" else "fr"
        self.store = store
        self.cache: dict[str, str] = cache if cache is not None else {}
        self.gender = gender if gender in ("keep", "random") else "keep"
        self.collisions: Counter = Counter()
        #: fakes re-drawn because a real value seen later collided with them, per kind (v0.3).
        self.redrawn: Counter = Counter()
        #: values the Synthesizer refused to give (SyntheticCollision / SyntheticLeak raised), per kind (v0.3).
        self.failed_closed: Counter = Counter()
        self._lists = synthetic_lists(self.lang)
        self._other = synthetic_lists("fr" if self.lang == "en" else "en")
        self._idx: Optional[_Index] = None
        self._dirty = 0          # bumped by _remember: the index key sees cache writes a store cannot report
        self._reserve_n = 0
        self._reserved_norms: set[str] = set()
        self._reserved_tokens: set[str] = set()
        self._reserved_groups: set[str] = set()
        self._reserved_initials: set[str] = set()
        self._memo: dict[str, tuple[str, ...]] = {}
        self._memo1: dict[str, tuple[str, ...]] = {}
        self._memo_n: dict[str, frozenset[str]] = {}
        #: v0.3.1: ``value -> folded given names and nicknames`` the dossier declares for a real person beyond its
        #: own value (the masker plugs its directory in, :meth:`use_nicknames`); None: the tables only.
        self.nickname_source: Optional[Any] = None
        self._nick_rev = 0
        #: v0.4 S0: real tokens already keyed, their one-edit keys (_del_keys -> tokens, 5+ letters) and sound keys
        self._edit_seen: set[str] = set()
        self._edit_keys: dict[str, list[str]] = {}
        self._sound_keys: set[str] = set()

    def use_nicknames(self, source: Any) -> None:
        """Plug the dossier's declared given names and nicknames (``value -> iterable of words``, the masker's
        directory) into the rendering and the collision rules (v0.3.1). The same sources as the masker's
        resolution then name a person: a mention "Reggie" of the real "Reginald ..." is written as the fake's first
        given name, and no fake ever uses a real nickname."""
        if source is not self.nickname_source:
            self.nickname_source = source
            self._memo_n.clear()
            self._nick_rev += 1

    # ------------------------------------------------------------------ store access
    def known(self, code: str) -> Optional[str]:
        """The fake value already given to ``code`` (store or cache), else None."""
        getter = getattr(self.store, "synthetic", None)
        if callable(getter):
            try:
                v = getter(code)
            except Exception:  # noqa: BLE001 - a broken store: no synthetic value
                v = None
            if v:
                return v
        return self.cache.get(code)

    def _remember(self, code: str, value: str) -> None:
        self._dirty += 1
        setter = getattr(self.store, "set_synthetic", None)
        if callable(setter):
            try:
                setter(code, value)
                self.cache.pop(code, None)
                return
            except Exception:  # noqa: BLE001 - keep it in the cache
                pass
        self.cache[code] = value

    def _entities(self) -> list[Entity]:
        try:
            return list(self.store.all()) if self.store is not None else []
        except Exception:  # noqa: BLE001
            return []

    def _store_rev(self) -> object:
        rev = getattr(self.store, "revision", None)
        if rev is None:
            return None
        try:
            return rev() if callable(rev) else rev
        except Exception:  # noqa: BLE001 - a broken store: no cache
            return None

    def _toks(self, value: str) -> tuple[str, ...]:
        """Folded words of 2+ letters of ``value`` (memoised: the index rebuild only folds new values)."""
        out = self._memo.get(value)
        if out is None:
            if len(self._memo) > 100_000:
                self._memo.clear()
            out = tuple(fold_word(r) for r, _s, _e in _runs(value) if len(r) >= 2)
            self._memo[value] = out
        return out

    def _ones(self, value: str) -> tuple[str, ...]:
        """Folded one-letter words of ``value`` (the initials it writes), memoised (v0.3.1)."""
        out = self._memo1.get(value)
        if out is None:
            if len(self._memo1) > 100_000:
                self._memo1.clear()
            out = tuple(fold_word(r) for r, _s, _e in _runs(value) if len(r) == 1)
            self._memo1[value] = out
        return out

    def _declared(self, value: str) -> frozenset[str]:
        """Folded given names and nicknames the dossier declares for the real person ``value`` beyond its own words:
        the nicknames written inside the value ('Reginald "Reggie" Pellworth') and the directory's (v0.3.1)."""
        out: set[str] = set()
        try:
            out.update(fold_word(r) for n in nm.person_nicknames(value) for r, _s, _e in _runs(n))
        except Exception:  # noqa: BLE001 - never fail a rendering on a name helper
            pass
        if self.nickname_source is not None:
            try:
                out.update(fold_word(w) for w in self.nickname_source(value) if isinstance(w, str))
            except Exception:  # noqa: BLE001 - a broken directory: the tables only
                pass
        return frozenset(w for w in out if w)

    def _nicks_of(self, value: str) -> frozenset[str]:
        """Every nickname and short form that names the real person ``value``: the table forms of its given names
        (both languages, prefix forms) and of the declared ones, and the declared words themselves (v0.3.1)."""
        out = self._memo_n.get(value)
        if out is None:
            if len(self._memo_n) > 50_000:
                self._memo_n.clear()
            firsts = {fold_word(r) for w in nm.split_person(value)[0] for r, _s, _e in _runs(w) if len(r) >= 2}
            declared = self._declared(value)
            words: set[str] = set(declared)
            for g in firsts | declared:
                words.update(_given_forms(g))
            out = frozenset(w for w in words - set(self._toks(value)) if len(w) >= 2)
            self._memo_n[value] = out
        return out

    def _index(self) -> _Index:
        """The collision index of the store (every real value of every kind, every fake value, the reserved
        surfaces). Cached on the store revision; a store without a revision is re-read at every call."""
        rev = self._store_rev()
        key = (rev, len(self.cache), self._dirty, self._reserve_n, self._nick_rev)
        if rev is not None and self._idx is not None and self._idx.key == key:
            return self._idx
        idx = _Index(key)
        idx.real_norms.update(self._reserved_norms)
        idx.real_tokens.update(self._reserved_tokens)
        idx.real_groups.update(self._reserved_groups)
        idx.real_initials.update(self._reserved_initials)
        people: list[tuple[Entity, str]] = []
        for e in self._entities():
            idx.real_norms.add(normalize(e.value))
            if e.norm:
                idx.real_norms.add(e.norm)
            idx.real_tokens.update(self._toks(e.value))
            idx.real_groups.update(_groups(e.value))
            idx.real_initials.update(self._ones(e.value))
            if e.kind == "PERSON":
                idx.real_nicks.update(self._nicks_of(e.value))
            fake = self.known(e.code)
            if not fake:
                continue
            idx.fakes[e.code] = fake
            idx.fake_norms.setdefault(normalize(fake), e.code)
            if e.kind == "PERSON":
                people.append((e, fake))
        for code, fake in self.cache.items():  # cache rows of entities a store does not list
            if code not in idx.fakes:
                idx.fakes[code] = fake
                idx.fake_norms.setdefault(normalize(fake), code)
        for t in idx.real_tokens:
            if len(t) >= 5:
                idx.by_len.setdefault(len(t), []).append(t)
        # v0.4 S0 (class D): the one-edit and sound keys grow with the tokens ever seen (a superset of the store's,
        # never smaller: a stricter rule, and no re-hash of the whole store at every rebuild)
        new = idx.real_tokens - self._edit_seen
        if new:
            for t in new:
                if len(t) >= 5:
                    for k in _del_keys(t):
                        self._edit_keys.setdefault(k, []).append(t)
                snd = _sound(t) if len(t) >= 4 else ""
                if snd:
                    self._sound_keys.add(snd)
            self._edit_seen |= new
        idx.edit_keys = self._edit_keys
        idx.real_sounds = self._sound_keys
        for e, fake in people:  # the fake surname of a real surname key: the oldest sharer whose fake is clean
            key_ = nm.surname_key(e.value)
            if not key_ or key_ in idx.surnames:
                continue
            fsur = " ".join(nm.split_person(fake)[1])
            if fsur and not (set(self._toks(fsur)) & idx.real_tokens):
                idx.surnames[key_] = (e.code, fsur)
        self._idx = idx
        return idx

    # ------------------------------------------------------------------ collision rules
    def _collides(self, ent: Entity, fake: str, idx: _Index, allowed: Optional[set[str]] = None) -> Optional[str]:
        """Why ``fake`` is not a safe value for ``ent`` against the store, or None when it is safe."""
        from .masker import entity_norm  # local import: masker imports this module

        allowed = allowed or set()
        nf = normalize(fake)
        if not nf or not fake.strip():
            return "empty"
        if nf == normalize(ent.value) or nf == ent.norm or entity_norm(ent.kind, fake) == ent.norm:
            return "equals the real value"
        if nf in idx.real_norms or entity_norm(ent.kind, fake) in idx.real_norms:
            return "equals a real value of the dossier"
        core = _core(fake)
        if len(core) >= 3 and core in idx.real_groups:
            return "equals a part of a real value"  # a short fake id that reads as a group of a real phone or id
        owner = idx.fake_norms.get(nf)
        if owner is not None and owner != ent.code:
            return "equals another fake value"
        real_core = _core(ent.value)
        if ent.kind != "BIRTHDATE" and len(real_core) >= 4 and _one_edit(core, real_core):
            # v0.4 S0 (class D): a fake phone one digit from the real one, "Wandrell" for a real "Vandrell". A birth
            # date is shifted by design (30 to 400 days) and keeps its own rule
            return "the fake is one edit from the real value"
        changed = self._changed_runs(ent.value, fake) - allowed
        if ent.kind in _NAME_KINDS:
            if changed & idx.real_tokens:
                return "a fake name word is a word of a real value"
            if changed & _whitelist_words():
                return "a fake name word is a whitelist word"
            near = self._near_real_word(changed, idx)
            if near is not None:
                return f"a fake name word {near}"
            if _real_word_inside(self._replaced_real_runs(ent.value, fake) - allowed, fake):
                return "a fake name word holds a real word"
            if any(g in idx.real_groups for g in self._name_groups(ent.kind, fake)):
                return "the initials of the fake read as a part of a real value"  # "MD" of a fake, state of an address
            # v0.3.1: a fake never uses a nickname of a real given name ("Reggie" while a real Reginald is in the
            # dossier), never holds a one-letter word where the real value has a word, and a fake initial is never
            # a one-letter word of the dossier (the masker would read a real "E." as the fake)
            if changed & idx.real_nicks:
                return "a fake name word is a nickname of a real given name"
            initials = self._fake_initials(ent.value, fake)
            if initials is None:
                return "a fake initial where the real name has a word"
            if initials & idx.real_initials:
                return "a fake initial is an initial of the dossier"
            return None
        if ent.kind == "EMAIL":
            # v0.4 S0 (class C): the local part of a fake e-mail never keeps a letter run of the real local part, not
            # even an initial ("t." of "t.marvelle"), and never a real word or nickname of the dossier ("dov" of a
            # real "Dovrena")
            rlocal, fl = ent.value.partition("@")[0], fake.partition("@")[0]
            fparts = _alpha_parts(fl)
            if fparts & _alpha_parts(rlocal):
                return "the fake e-mail keeps a part of the real local part"
            if any(len(p) >= 2 and (p in idx.real_tokens or p in idx.real_nicks) for p in fparts):
                return "a part of the fake local part is a real word or nickname"
        if ent.kind not in _OWN_GENERATOR_KINDS:
            # v0.4 S0 (class K): an identifier never keeps a letter code of the real value (an employer, bank or
            # company code), at its place or elsewhere; public labels (ECLI, RG, PV) excepted
            codes = {fold_word(r) for r, _s, _e in _runs(ent.value) if len(r) >= 2} - _public_labels()
            if codes & {fold_word(r) for r, _s, _e in _runs(fake) if len(r) >= 2}:
                return "the fake keeps a letter code of the real value"
        if ent.kind in _WORD_KINDS:
            # built from word pools: every distinctive word of the fake counts, changed or not (a fake address
            # that keeps the real street, a fake e-mail that keeps a real name part)
            words = {w for w in self._toks(fake) if len(w) >= 4 and w not in _generic_words()}
        else:
            words = {w for w in changed if len(w) >= 4 and w not in _NEUTRAL_WORDS}
        if words & idx.real_tokens:
            return "a word of the fake is a word of a real value"
        # v0.4 S0 (class D): no word one edit from, or sounding like, a word of the real value itself; the local part
        # of an e-mail (made of names) also no word near any real word of the dossier, like a fake name
        near = self._near_own_word(words, self._toks(ent.value))
        if near is None and ent.kind == "EMAIL":
            near = self._near_real_word({w for w in self._toks(fake.partition("@")[0]) if len(w) >= 4}, idx)
        if near is not None:
            return f"a word of the fake {near}"
        return None

    @staticmethod
    def _near_real_word(words: Iterable[str], idx: _Index) -> Optional[str]:
        """Why a word of 4+ letters of ``words`` reads as a real word of the dossier: one edit from a real word of 5+
        letters at any place (the first letter included), or the same sound as a real word of 4+ letters; None
        otherwise (v0.4 S0, class D)."""
        for w in words:
            if len(w) < 4:
                continue
            for k in _del_keys(w):
                if any(_one_edit(w, t) for t in idx.edit_keys.get(k, ())):
                    return "is one edit from a real word"
            snd = _sound(w)
            if snd and snd in idx.real_sounds:
                return "sounds like a real word"
        return None

    @staticmethod
    def _near_own_word(words: Iterable[str], own: Iterable[str]) -> Optional[str]:
        """:meth:`_near_real_word` against the words ``own`` of the real value itself (v0.4 S0, class D)."""
        own = [t for t in own if len(t) >= 4]
        sounds = {_sound(t) for t in own} - {""}
        for w in words:
            if len(w) < 4:
                continue
            if any(len(t) >= 5 and _one_edit(w, t) for t in own):
                return "is one edit from a real word"
            snd = _sound(w)
            if snd and snd in sounds:
                return "sounds like a real word"
        return None

    def _name_groups(self, kind: str, fake: str) -> set[str]:
        """The initials (a person) or acronyms (a company) a reader or the masker's fake-surface matcher can form
        from ``fake``, folded: the first letters of its changeable words, the first and last of them, and what the
        name helpers list."""
        runs = _runs(fake)
        if kind == "PERSON":
            letters = [fold_word(r)[:1] for r, s, _e in runs if not self._person_kept(fake, r, s)]
        else:
            letters = [fold_word(r)[:1] for (r, _s, _e), keep in zip(runs, self._org_fixed_mask(runs)) if not keep]
        out = set()
        if len(letters) >= 2:
            out.add("".join(letters))
            out.add(letters[0] + letters[-1])
        helper = nm.person_initials if kind == "PERSON" else nm.org_acronyms
        try:
            out.update(fold_word(g) for g in helper(fake) if len(g) >= 2)
        except Exception:  # noqa: BLE001 - never fail a draw on a name helper
            pass
        return out

    @staticmethod
    def _initials_differ(real: str, fake: str) -> bool:
        """True when every replaced word of ``fake`` starts with another letter than the real word it replaces
        (an initials-only mention then never renders as the real initials)."""
        a = [fold_word(r) for r, _s, _e in _runs(real)]
        b = [fold_word(r) for r, _s, _e in _runs(fake)]
        if len(a) != len(b):
            return True
        return all(x == y or x[:1] != y[:1] for x, y in zip(a, b))

    @staticmethod
    def _fake_initials(real: str, fake: str) -> Optional[set[str]]:
        """The one-letter words ``fake`` writes in place of a real word (folded), or None when one of them stands
        where the real value has a longer word: a fake holds an initial only where the real value holds one
        (v0.3.1)."""
        a = [fold_word(r) for r, _s, _e in _runs(real)]
        b = [fold_word(r) for r, _s, _e in _runs(fake)]
        if len(a) != len(b):
            ones = {y for y in b if len(y) == 1}
            return None if ones and not any(len(x) == 1 for x in a) else ones - set(a)
        if any(len(y) == 1 and len(x) != 1 for x, y in zip(a, b)):
            return None
        return {y for x, y in zip(a, b) if len(y) == 1 and x != y}

    def _person_kept(self, value: str, run: str, start: int) -> bool:
        """A word of a person's value its fake keeps (a title, a particle, a generational word); a capitalised
        particle that opens the name ("Al Brightwater") may be a given name, and a capital "D" or "Y" not followed by
        an apostrophe is an initial: both are replaced (v0.3.1)."""
        return self._person_fixed(run) and not _leading_particle(run, value, start) \
            and not _capital_initial(run, value, start) and not _trailing_particle(run, value, start)

    def _usable(self, ent: Entity, fake: str, idx: _Index) -> bool:
        """A known fake stays as long as it lines up with the (possibly upgraded) real value and the store that
        grew since the draw still holds no value it collides with."""
        if ent.kind in _NAME_KINDS and self._mapping(ent.value, fake) is None:
            return False  # stale: the entity got a fuller value ("Ouardi" -> "Mehdi Ouardi")
        if self._collides(ent, fake, idx) is not None:
            return False
        if ent.kind in ("EMAIL", "URL") and self._link_stale(ent, fake):
            return False  # the person or company it was built from got another fake value since
        return ent.kind not in _NAME_KINDS or self._initials_differ(ent.value, fake)

    def _link_stale(self, ent: Entity, fake: str) -> bool:
        """True when a fake e-mail or web address no longer carries the current fake of the person (local part)
        or company (host) its real value is linked to."""
        if ent.kind == "EMAIL":
            local, at, host = ent.value.partition("@")
            flocal, fat, fhost = fake.partition("@")
            if not at or not fat:
                return False
            # v0.4 S0: the local part is stale when the person it names would now give other letters
            expected = self._person_local(local, None)
            if expected is not None and _alpha_parts(expected) != _alpha_parts(flocal):
                return True
            return self._part_stale(host, fhost, ("ORG",), 4, all_tokens=False)
        m = re.match(r"^(?:[a-z][a-z0-9+.\-]*://)?(?:www\.)?(?P<host>[^/?#:]+)", ent.value, re.IGNORECASE)
        fm = re.match(r"^(?:[a-z][a-z0-9+.\-]*://)?(?:www\.)?(?P<host>[^/?#:]+)", fake, re.IGNORECASE)
        if not m or not fm:
            return False
        return self._part_stale(m.group("host"), fm.group("host"), ("ORG",), 4, all_tokens=False)

    def _part_stale(self, real_part: str, fake_part: str, kinds: tuple[str, ...], min_len: int,
                    all_tokens: bool) -> bool:
        linked = self._linked(real_part, kinds)
        if linked is None:
            return False
        mapping = self._fake_person_tokens(linked)
        body, fbody = _ascii(real_part), _ascii(fake_part)
        for real, fk in sorted(mapping.items(), key=lambda kv: -len(kv[0])):
            r, f = _ascii(real), _ascii(fk)
            if len(r) < min_len or r not in body or f == r:
                continue
            if f not in fbody:
                return True
            if not all_tokens:
                return False  # the host carries the first changed word only (see _domain)
        return False

    # ------------------------------------------------------------------ public API
    def value_for(self, ent: Entity) -> str:
        """The fake value of ``ent``: generated once, kept in the store or the cache, and re-drawn when a real
        value seen since collides with it. Raises :class:`SyntheticCollision` when no safe value exists."""
        idx = self._index()
        found = self.known(ent.code)
        if found and self._usable(ent, found, idx):
            return found
        value = self._draw(ent, idx, found)
        if found:
            self.redrawn[ent.kind] += 1
        self._remember(ent.code, value)
        return value

    def _draw(self, ent: Entity, idx: _Index, previous: Optional[str]) -> str:
        hard = idx.real_tokens | _whitelist_words() | idx.real_nicks
        soft: set[str] = set()
        ini_soft: set[str] = set()
        for code, fake in idx.fakes.items():
            if code != ent.code:
                soft.update(self._toks(fake))
                ini_soft.update(self._ones(fake))
        shared = None
        if ent.kind == "PERSON":
            own = idx.surnames.get(nm.surname_key(ent.value))
            if own is not None:
                shared = own[1]
            elif previous:  # a stale fake of this very entity: keep its surname when it is still clean
                prev_sur = " ".join(nm.split_person(previous)[1])
                if prev_sur and not (set(self._toks(prev_sur)) & idx.real_tokens):
                    shared = prev_sur
        draw = _Draw(hard, soft, shared, idx.real_initials, ini_soft)
        value = None
        for attempt in range(_MAX_TRIES):
            rng = _Rng(self.seed, ent.code, str(attempt))
            cand = self._generate(ent, rng, draw)
            if not cand or cand == ent.value or self._collides(ent, cand, idx, draw.allowed) is not None:
                continue
            if ent.kind in _NAME_KINDS and not self._initials_differ(ent.value, cand):
                continue
            value = cand
            break
        if value is None:
            value = self._fallback(ent, draw)
        reason = self._collides(ent, value, idx, draw.allowed)
        if reason is None and ent.kind in _NAME_KINDS and (self._mapping(ent.value, value) is None
                                                         or not self._initials_differ(ent.value, value)):
            reason = "no fake name with the shape of the real one"
        if reason is not None:  # canary: never hand back a value that could stand for real text
            self.failed_closed[ent.kind] += 1
            raise SyntheticCollision(f"{ent.code}: {reason}")
        if ent.kind in UNVERIFIED_KINDS:
            self.collisions[ent.kind] += 1
        return value

    def reserve(self, surfaces: Iterable[str]) -> None:
        """Surfaces the caller detected in a text but did not (yet) register: no fake value may equal one of
        them or reuse one of their words. The masker may call it before it renders a text."""
        n = 0
        for s in surfaces:
            if not isinstance(s, str) or not s.strip():
                continue
            self._reserved_norms.add(normalize(s))
            self._reserved_tokens.update(self._toks(s))
            self._reserved_groups.update(_groups(s))
            self._reserved_initials.update(self._ones(s))
            n += 1
        if n:
            self._reserve_n += 1

    def collisions_with_store(self) -> list[tuple[str, str]]:
        """``(code, reason)`` for every known fake value that collides with the store as it is now (a real value
        registered after the draw, a stale shape...). Values never appear in the result."""
        idx = self._index()
        out: list[tuple[str, str]] = []
        for e in self._entities():
            fake = idx.fakes.get(e.code)
            if not fake:
                continue
            reason = self._collides(e, fake, idx)
            if reason is None and ent_is_name(e) and self._mapping(e.value, fake) is None:
                reason = "stale shape"
            if reason is None and ent_is_name(e) and not self._initials_differ(e.value, fake):
                reason = "same initials"
            if reason is not None:
                out.append((e.code, reason))
        return out

    def redraw_collisions(self) -> int:
        """Re-draw every known fake that :meth:`collisions_with_store` lists. Returns the number re-drawn; a
        fake that cannot be re-drawn safely stays and is counted in :attr:`failed_closed` (the caller must then
        stop treating it as a known fake surface)."""
        n = 0
        codes = {code for code, _reason in self.collisions_with_store()}
        if not codes:
            return 0
        for e in self._entities():
            if e.code not in codes:
                continue
            try:
                self.value_for(e)
                n += 1
            except SyntheticError:
                continue
        return n

    def values(self) -> dict[str, str]:
        """``code -> fake value`` for every entity that has one, after every fake that collides with the store as
        it is now was re-drawn. A caller that looks for known fake surfaces in a text (the masker's re-mask and
        restore paths) must take them from here, never from the raw store rows: a row can hold a fake drawn before
        a colliding real value was registered, and that stale surface would hide the real value."""
        self.redraw_collisions()
        return dict(self._index().fakes)

    @staticmethod
    def _changed_runs(real: str, fake: str) -> set[str]:
        """Folded words of ``fake`` that differ from the word of ``real`` at the same place (every word when the
        two values do not line up)."""
        a = [fold_word(r) for r, _s, _e in _runs(real)]
        b = [fold_word(r) for r, _s, _e in _runs(fake)]
        if len(a) != len(b):
            return {w for w in b if len(w) >= 2}
        return {y for x, y in zip(a, b) if x != y and len(y) >= 2}

    @staticmethod
    def _replaced_real_runs(real: str, fake: str) -> set[str]:
        """Folded words of ``real`` that ``fake`` replaces (every word when the two values do not line up)."""
        a = [fold_word(r) for r, _s, _e in _runs(real)]
        b = [fold_word(r) for r, _s, _e in _runs(fake)]
        if len(a) != len(b):
            return {w for w in a if len(w) >= 2}
        return {x for x, y in zip(a, b) if x != y and len(x) >= 2}

    def render(self, ent: Entity, fake: str, surface: str, need: str = "", vtoks: tuple = ()) -> str:
        """The part of ``fake`` that replaces the mention ``surface`` of ``ent`` (surname alone, initials,
        "Surname, Firstname", upper case, another date form...). Raises :class:`SyntheticLeak` when the result
        would still show the real text (canary: the caller then writes the code)."""
        if ent.kind in _NAME_KINDS:
            rep = self._render_name(ent.value, fake, surface, need)
        else:
            rep = self._render_other(ent, ent.value, fake, surface)
        reason = self._leaks(ent, fake, surface, rep)
        if reason is not None:
            self.failed_closed[ent.kind] += 1
            raise SyntheticLeak(f"{ent.code}: {reason}")
        return rep

    def _leaks(self, ent: Entity, fake: str, surface: str, rep: str) -> Optional[str]:
        """Why ``rep`` (the rendering of ``surface``) still shows the real value, or None."""
        core_s, core_r = _core(surface), _core(rep)
        if not core_s:
            return None
        if ent.kind in _NAME_KINDS:
            real_runs = self._replaced_real_runs(ent.value, fake)
            out_runs = {fold_word(r) for r, _s, _e in _runs(rep)}
            if out_runs & real_runs:
                return "a real name word is still written"
            for w in out_runs:
                # v0.4 S0 (class D): one edit at any place (the first letter included), or the same sound
                if len(w) >= 4 and any(len(x) >= 5 and _one_edit(w, x) for x in real_runs):
                    return "a word one edit from a real name word is written"
                if len(w) >= 4 and _sound(w) and any(_sound(x) == _sound(w) for x in real_runs if len(x) >= 4):
                    return "a word that sounds like a real name word is written"
            if _real_word_inside(real_runs, rep):
                return "a word holding a real name word is written"
            # v0.3.1 (cause A): a word of the mention written unchanged is a leak unless it is a fixed word (title,
            # particle, legal form, generic word, the "s" of a possessive), whatever its length and whether or not
            # it is a word of the stored value: "Reggie" of a real "Reginald ..." stays "Reggie" otherwise
            if unchanged_mention_word(ent.kind, surface, rep, ent.value):
                return "a word of the mention is written unchanged"
            if core_r == core_s:
                surf_runs = {fold_word(r) for r, _s, _e in _runs(surface)}
                initials = {x[:1] for x in real_runs}
                if surf_runs & real_runs or any(len(r) == 1 and r in initials for r in surf_runs):
                    return "the mention is written unchanged"
            return None
        if core_r == core_s:
            return "the mention is written unchanged"
        if ent.kind != "BIRTHDATE" and len(core_s) >= 4 and _one_edit(core_r, core_s):
            return "the rendering is one edit from the mention"  # v0.4 S0 (class D)
        real_body = _core(ent.value)
        if len(real_body) >= 4 and real_body in core_r:
            return "the real value is still written"
        if ent.kind == "EMAIL" and "@" in surface and "@" in rep:
            if _alpha_parts(rep.partition("@")[0]) & _alpha_parts(surface.partition("@")[0]):
                return "a part of the real local part is still written"  # v0.4 S0 (class C)
        if ent.kind not in _OWN_GENERATOR_KINDS:
            codes = {fold_word(r) for r, _s, _e in _runs(surface) if len(r) >= 2} - _public_labels()
            if codes & {fold_word(r) for r, _s, _e in _runs(rep) if len(r) >= 2}:
                return "a letter code of the real value is still written"  # v0.4 S0 (class K)
        if ent.kind in _WORD_KINDS:
            real_words = {w for w in self._toks(ent.value) if len(w) >= 4 and w not in _generic_words()}
            if {w for w in self._toks(rep) if len(w) >= 4} & real_words:
                return "a real word of the value is still written"
        if ent.kind == "BIRTHDATE" and det.structured_norm("BIRTHDATE", rep) == det.structured_norm("BIRTHDATE", ent.value):
            return "the real date is still written"
        return None

    def restore(self, ent: Entity, fake: str, surface: str) -> str:
        """The real text for the synthetic mention ``surface`` (inverse of :meth:`render`)."""
        if ent.kind in _NAME_KINDS:
            return self._render_name(fake, ent.value, surface, "", reverse=True)
        return self._render_other(ent, fake, ent.value, surface, reverse=True)

    def date_offset(self, ent: Entity) -> int:
        """Days added to a real birth date (30 to 400, either direction), fixed per entity."""
        rng = _Rng(self.seed, ent.code, "date")
        n = rng.between(30, 400)
        return n if rng.below(2) == 0 else -n

    # ------------------------------------------------------------------ generation
    def _generate(self, ent: Entity, rng: _Rng, draw: _Draw) -> Optional[str]:
        kind, value = ent.kind, ent.value
        if kind == "PERSON":
            return self._person(value, rng, draw)
        if kind == "ORG":
            return self._org(value, rng, draw)
        if kind == "EMAIL":
            return self._email(value, rng, draw)
        if kind == "PHONE":
            return self._phone(value, rng)
        if kind == "IBAN":
            return self._iban(value, rng)
        if kind == "CARD":
            return self._card(value, rng)
        if kind == "NIR":
            return self._nir(value, rng)
        if kind == "SIREN":
            return _overlay(value, self._siren(rng)) or self._siren(rng)
        if kind == "SIRET":
            return _overlay(value, self._siret(rng)) or self._siret(rng)
        if kind == "VAT":
            return self._vat(value, rng)
        if kind == "URL":
            return self._url(value, rng, draw)
        if kind == "ADDRESS":
            return self._address(value, rng)
        if kind == "BIRTHDATE":
            return shift_date_text(value, self.date_offset(ent)) or _shape(value, rng, keep_prefix=False)
        # v0.4 S0 (class K): the leading letter code gets a fake code of the same shape, the same for every
        # identifier of the dossier that carries the same real code
        return _shape(value, rng, prefix=lambda label: self._id_prefix(label, draw.hard))

    def _id_prefix(self, label: str, avoid: set[str]) -> str:
        """The fake of the leading letter code ``label`` of an identifier (v0.4 S0, class K): same length and case,
        drawn from ``HMAC(seed, folded label)`` so that every identifier of the dossier with the same real code gets
        the same fake code; never a letter of the real code at its own place, never a word of the dossier
        (``avoid``) and never a public label."""
        real = fold_word(label)
        rng = _Rng(self.seed, "id-prefix", real)
        for _ in range(64):
            cand = "".join(rng.letters(1, upper=c.isupper()) if c.isalpha() else c for c in label)
            f = fold_word(cand)
            if any(x == y for x, y in zip(f, real)) or f in avoid or f in _public_labels():
                continue
            return cand
        return "".join(_letter_other_than(rng, c) if c.isupper() else _letter_other_than(rng, c).lower()
                       for c in label)

    def _fallback(self, ent: Entity, draw: _Draw) -> str:
        """The last resort when every pool is exhausted: a name keeps its word count, its fixed words (titles,
        particles, legal forms) and its case, with made-up readable words elsewhere; other kinds keep their shape."""
        rng = _Rng(self.seed, ent.code, "fallback")
        if ent.kind not in _NAME_KINDS:
            return _shape(ent.value, rng, keep_prefix=False)
        runs = _runs(ent.value)
        if ent.kind == "ORG":
            fixed = self._org_fixed_mask(runs)
        else:
            fixed = [self._person_kept(ent.value, r, s) for r, s, _e in runs]
        out: list[str] = []
        pos = 0
        for (run, s, e), keep in zip(runs, fixed):
            if keep:
                rep = run
            elif len(run) == 1:
                # v0.3.1: a free initial when one is left; else the draw's final check fails closed
                rep = _free_initial(rng, fold_word(run), draw.ini_hard, draw.ini_soft) \
                    or _letter_other_than(rng, fold_word(run))
            else:
                rep = _case_like(run, _pronounceable(rng, max(5, min(8, len(run))), fold_word(run)[:1]))
            out.append(ent.value[pos:s])
            out.append(rep)
            pos = e
        out.append(ent.value[pos:])
        return "".join(out)

    # ---- persons
    def _sex_of(self, firsts: list[str]) -> str:
        if self.gender == "random":
            return "U"
        for f in firsts:
            for run, _s, _e in _runs(f):
                sex = _sex_table().get(fold_word(run))
                if sex in ("F", "M"):
                    return sex
        return "U"

    def _first_pools(self, sex: str, rng: _Rng) -> list:
        """Pools tried in order for a fake first name: the sex-matched list, then the public first names."""
        if sex == "F":
            first = self._lists["first_f"]
        elif sex == "M":
            first = self._lists["first_m"]
        else:
            first = self._lists["first_f"] if rng.below(2) == 0 else self._lists["first_m"]
        return [first, _extension_first_names()]

    def _last_pools(self) -> list:
        """Fake surnames: the list of the language, then the other language's list, then public first names
        (many surnames are first names) before the made-up fallback."""
        return [self._lists["last"], self._other["last"], _extension_first_names()]

    def _org_pools(self) -> list:
        """Fake company words: the word list, then surnames (companies named after a founder), then the other
        language's word list."""
        return [self._lists["org"], self._lists["last"], self._other["org"], self._other["last"]]

    @staticmethod
    def _pick_word(pools: list, rng: _Rng, draw: _Draw, used: set[str], avoid_initial: str = "") -> Optional[str]:
        """A word of the first pool that still has one: never a hard-forbidden word (real values of the dossier,
        whitelist), never a word used in this value, never a hyphenated word (it would add a token), never a word
        starting with ``avoid_initial``; a word no other fake uses when the pool allows it. None when every
        pool is exhausted (the caller falls back)."""
        for pool in pools:
            ok = []
            for w in pool:
                f = fold_word(w)
                if f in draw.hard or f in used or "-" in f or not f or (avoid_initial and f[0] == avoid_initial):
                    continue
                ok.append((w, f))
            if not ok:
                continue
            fresh = [w for w, f in ok if f not in draw.soft]
            choice = rng.pick(fresh) if fresh else rng.pick([w for w, _f in ok])
            used.add(fold_word(choice))
            return choice
        return None

    @staticmethod
    def _person_fixed(run: str) -> bool:
        f = fold_word(run)
        return (f in nm.PARTICLES or f in ("les", "l") or nm.is_title_word(run) or f in nm.GENERATIONAL
                or f in nm.TITLE_WORDS)

    def _person(self, value: str, rng: _Rng, draw: _Draw) -> Optional[str]:
        firsts, surname = nm.split_person(value)
        first_runs = {fold_word(r) for w in firsts for r, _s, _e in _runs(w)}
        sur_runs = {fold_word(r) for w in surname for r, _s, _e in _runs(w)}
        sex = self._sex_of(firsts)
        first_pools = self._first_pools(sex, rng)
        last_pools = self._last_pools()
        used: set[str] = set()
        shared_runs = [r for r, _s, _e in _runs(draw.shared)] if draw.shared else []
        shared_i = 0
        out: list[str] = []
        pos = 0
        for run, s, e in _runs(value):
            f = fold_word(run)
            lead = _leading_particle(run, value, s)  # "Al Brightwater": a given name, not a particle (v0.3.1)
            if self._person_kept(value, run, s):
                rep = run
            elif len(f) == 1:
                # v0.3.1: a fake initial only where the real value has one, never a one-letter word of the dossier
                rep = _free_initial(rng, f, draw.ini_hard, draw.ini_soft)
            elif f in first_runs or lead:
                rep = self._pick_word(first_pools, rng, draw, used, f[:1])
            elif f in sur_runs and shared_runs:
                rep = shared_runs[min(shared_i, len(shared_runs) - 1)]
                shared_i += 1
            else:
                rep = self._pick_word(last_pools, rng, draw, used, f[:1])
            if rep is None:
                return None
            out.append(value[pos:s])
            out.append(_case_like(run, rep))
            pos = e
        out.append(value[pos:])
        fake = "".join(out)
        return fake if fake != value else None

    # ---- companies
    @staticmethod
    def _org_fixed(f: str) -> bool:
        return (f in nm.LEGAL_FORMS or f in nm.ORG_GENERIC or f in nm.ORG_TRIGGERS or f in nm._ORG_CONNECT
                or f in nm.PARTICLES or f in nm._ARTICLES or f in nm._ORG_SKIP or f in lx.business_words()
                or f in lx.contract_roles() or f in lx.legal_terms() or f in lx.stopwords())

    def _org_fixed_mask(self, runs: list[tuple[str, int, int]]) -> list[bool]:
        fixed = [self._org_fixed(fold_word(r)) for r, _s, _e in runs]
        if fixed and all(fixed):
            # a company made of generic words only ("Garage du Centre"): the last word that is not a legal form or
            # a trigger changes, so that the fake differs from the real
            for i in range(len(runs) - 1, -1, -1):
                f = fold_word(runs[i][0])
                if f not in nm.LEGAL_FORMS and f not in nm.ORG_TRIGGERS and f not in nm._ORG_CONNECT \
                        and f not in nm.PARTICLES and f not in nm._ARTICLES:
                    fixed[i] = False
                    break
        return fixed

    def _org(self, value: str, rng: _Rng, draw: _Draw) -> Optional[str]:
        runs = _runs(value)
        if not runs:
            return None
        fixed = self._org_fixed_mask(runs)
        pools = self._org_pools()
        used: set[str] = set()
        out: list[str] = []
        pos = 0
        for (run, s, e), keep in zip(runs, fixed):
            if keep:
                rep = run
            else:
                word = self._pick_word(pools, rng, draw, used, fold_word(run)[:1])
                if word is None:
                    return None
                rep = _case_like(run, word)
            out.append(value[pos:s])
            out.append(rep)
            pos = e
        out.append(value[pos:])
        fake = "".join(out)
        return fake if fake != value else None

    # ---- e-mail, web
    def _fake_person_tokens(self, ent: Entity) -> dict[str, str]:
        """Folded real name run -> fake run of one registered person or company (its fake value is created)."""
        fake = self.value_for(ent)
        real_runs = [r for r, _s, _e in _runs(ent.value)]
        fake_runs = [r for r, _s, _e in _runs(fake)]
        if len(real_runs) != len(fake_runs):
            return {}
        return {fold_word(r): f for r, f in zip(real_runs, fake_runs)}

    def _linked(self, text: str, kinds: tuple[str, ...]) -> Optional[Entity]:
        """The one registered entity whose distinctive name run (4+ letters) sits in ``text`` (folded, ascii)."""
        body = _ascii(text)
        found: dict[str, Entity] = {}
        for e in self._entities():
            if e.kind not in kinds:
                continue
            for r, _s, _e in _runs(e.value):
                f = _ascii(r)
                if len(f) >= 4 and f in body and not nm._lexicon_word(fold_word(r)) \
                        and fold_word(r) not in nm.ORG_GENERIC and fold_word(r) not in nm.LEGAL_FORMS:
                    found[e.code] = e
                    break
        return next(iter(found.values())) if len(found) == 1 else None

    def _local_part(self, local: str, rng: _Rng, draw: _Draw) -> str:
        """The local part of a fake e-mail (v0.4 S0, class C): the real local part rewritten with the fake of the
        person it names (:meth:`_person_local`), else a neutral one built from fake names in its layout. No letter
        run of the real local part is ever copied, not even an initial or a short form ("dov" of a real "Dovrena")."""
        real_parts = _alpha_parts(local)
        out = self._person_local(local, rng)
        if out is not None:
            return out
        pools_first = [self._lists["first_f"] + self._lists["first_m"], _extension_first_names()]
        parts = re.split(r"[._\-]", local)
        sep = next((c for c in local if c in "._-"), ".")
        digits = re.search(r"\d+$", local)
        lead = fold_word(parts[0])[:1] if parts and parts[0] else ""
        cand = ""
        for _ in range(4):
            used: set[str] = set()
            first = _ascii(self._pick_word(pools_first, rng, draw, used, lead)
                           or _pronounceable(rng, 6, lead))
            last = _ascii(self._pick_word(self._last_pools(), rng, draw, used) or _pronounceable(rng, 7))
            tail = rng.digits(len(digits.group())) if digits else ""
            if len(parts) >= 2:
                head = first[:1] if len(parts[0]) == 1 else first
                cand = f"{head}{sep}{last}{tail}"
            else:
                cand = (first + last if len(local) > 8 else last) + tail
            if not (_alpha_parts(cand) & real_parts):
                break
        return cand

    def _person_local(self, local: str, rng: Optional[_Rng]) -> Optional[str]:
        """The local part ``local`` rewritten with the fake of the one person it names (v0.4 S0, class C), or None.
        Each letter run must read as a sequence of pieces of the real name: a whole word, a short form or nickname
        of a given name (tables, prefix forms, the dossier's declared ones), the first 3+ letters of a word of 5+
        letters, an initial. Each piece is written as the fake word, an initial as the fake initial; digits are
        drawn again (``rng`` None: zeros, for a comparison of the letters only); separators stay. None when no
        person is linked, when a run does not read that way, or when the result would still share a letter run
        with the real local part: the caller then builds a neutral local part."""
        person = self._linked(local, ("PERSON",))
        if person is None:
            return None
        try:
            fake = self.value_for(person)
        except SyntheticError:
            return None
        src = [fold_word(r) for r, _s, _e in _runs(person.value)]
        dst = [r for r, _s, _e in _runs(fake)]
        if not src or len(src) != len(dst):
            return None
        try:
            firsts = {fold_word(r) for w in nm.split_person(person.value)[0] for r, _s, _e in _runs(w)}
        except Exception:  # noqa: BLE001 - never fail a draw on a name helper
            firsts = set()
        declared = self._declared(person.value)
        whole: dict[str, str] = {}
        initial: dict[str, str] = {}
        forms: dict[str, str] = {}
        heads: list[tuple[str, str]] = []
        for f, d in zip(src, dst):
            a, b = _ascii(f), _ascii(d)
            if not a or not b or a == b:
                continue  # a word the fake keeps (title, particle): never written into a local part
            whole.setdefault(a, b)
            initial.setdefault(a[:1], b[:1])
            if len(a) >= 5:
                heads.append((a, b))
            if f in firsts:
                for g in _given_forms(f) | declared:
                    ga = _ascii(g)
                    if len(ga) >= 2:
                        forms.setdefault(ga, b)

        def piece(p: str) -> Optional[str]:
            if p in whole:
                return whole[p]
            if p in forms:
                return forms[p]
            if len(p) >= 3:
                for a, b in heads:
                    if a.startswith(p):
                        return b
            if len(p) == 1:
                return initial.get(p)
            return None

        def read(run: str) -> Optional[str]:
            n = len(run)
            best: list[Optional[list[str]]] = [None] * (n + 1)
            best[0] = []
            for i in range(n):
                if best[i] is None:
                    continue
                for j in range(n, i, -1):
                    rep = piece(run[i:j])
                    if rep is not None and (best[j] is None or len(best[i]) + 1 < len(best[j])):
                        best[j] = best[i] + [rep]
            got = best[n]
            if not got or (n > 3 and all(len(x) == 1 for x in got)):
                return None
            return "".join(got)

        out: list[str] = []
        for m in re.finditer(r"[^\W\d_]+|\d+|.", local):
            t = m.group()
            if t[0].isdigit():
                out.append(rng.digits(len(t)) if rng is not None else "0" * len(t))
            elif t[0].isalpha():
                rep = read(_ascii(t))
                if rep is None:
                    return None
                out.append(_case_like(t, rep))
            else:
                out.append(t)
        result = "".join(out)
        if _alpha_parts(result) & _alpha_parts(local):
            return None
        return result

    def _domain(self, host: str, rng: _Rng, draw: _Draw) -> str:
        labels = host.lower().split(".")
        org = self._linked(host, ("ORG",))
        if org is not None:
            mapping = self._fake_person_tokens(org)
            for real, fake in mapping.items():
                if len(_ascii(real)) >= 4 and any(_ascii(real) in lb for lb in labels) and _ascii(fake) != _ascii(real):
                    return f"{_ascii(fake)}.{SYNTHETIC_TLD}"
        used: set[str] = set()
        word = _ascii(self._pick_word(self._org_pools(), rng, draw, used) or _pronounceable(rng, 7))
        return f"{word}.{SYNTHETIC_TLD}"

    def _email(self, value: str, rng: _Rng, draw: _Draw) -> Optional[str]:
        local, at, host = value.partition("@")
        if not at or not host:
            return _shape(value, rng, keep_prefix=False)
        return f"{self._local_part(local, rng, draw)}@{self._domain(host, rng, draw)}"

    def _url(self, value: str, rng: _Rng, draw: _Draw) -> Optional[str]:
        m = re.match(r"^(?P<scheme>[a-z][a-z0-9+.\-]*://)?(?P<www>www\.)?(?P<host>[^/?#:]+)(?P<rest>.*)$", value,
                     re.IGNORECASE)
        if not m:
            return _shape(value, rng, keep_prefix=False)
        host = self._domain(m.group("host"), rng, draw)
        rest = "".join(str(rng.below(10)) if c.isdigit() else rng.letters(1, upper=c.isupper()) if c.isalpha() else c
                       for c in m.group("rest"))
        return f"{m.group('scheme') or ''}{m.group('www') or ''}{host}{rest}"

    # ---- phones
    def _phone(self, value: str, rng: _Rng) -> Optional[str]:
        digits = re.sub(r"\D", "", value)
        if not digits:
            return None
        stripped = value.strip()
        prefix, national = "", digits
        if stripped.startswith("+"):
            prefix, national = self._split_cc(digits, "")
        elif digits.startswith("00") and len(digits) > 11:
            prefix, national = self._split_cc(digits[2:], "00")
        country = self._country(prefix, national)
        if country == "fr":
            nat = national if prefix == "" else "0" + national
            fake_nat = ("0639" + "98" if nat[:2] in ("06", "07") else "0199" + "00") + rng.digits(4)
            fake_nat = fake_nat[1:] if prefix else fake_nat
        elif country == "us":
            fake_nat = f"{rng.between(2, 9)}{rng.digits(2)}55501{rng.digits(2)}"
        elif country == "uk":
            nat = national if prefix == "" else "0" + national
            fake_nat = ("07700900" if nat[:2] == "07" else "02079460") + rng.digits(3)
            fake_nat = fake_nat[1:] if prefix else fake_nat
        else:
            self.collisions["PHONE"] += 1
            fake_nat = rng.digits(len(national))
        if len(fake_nat) < len(national):
            fake_nat += rng.digits(len(national) - len(fake_nat))
        fake = prefix + fake_nat[:len(national)]
        return _overlay(value, fake) or fake

    @staticmethod
    def _split_cc(digits: str, lead: str) -> tuple[str, str]:
        for n in (1, 2, 3):
            cc = digits[:n]
            if cc in _COUNTRY_CODES:
                return lead + cc, digits[n:]
        return lead + digits[:2], digits[2:]

    @staticmethod
    def _country(prefix: str, national: str) -> str:
        cc = prefix.lstrip("0")
        if cc == "33" or (not cc and len(national) == 10 and national[0] == "0"):
            return "fr"
        if cc == "1" or (not cc and len(national) == 10 and national[0] not in "01"):
            return "us"
        if cc == "44" or (not cc and len(national) == 11 and national[0] == "0"):
            return "uk"
        return "other"

    # ---- identifiers with check digits
    def _iban(self, value: str, rng: _Rng) -> Optional[str]:
        compact = re.sub(r"[^0-9A-Za-z]", "", value).upper()
        if len(compact) < 15:
            return _shape(value, rng, keep_prefix=False)
        country = compact[:2]
        if country == "FR" and len(compact) == 27:
            bank, branch, account = "00000", rng.digits(5), rng.digits(11)
            bban = bank + branch + account + _fr_rib_key(bank, branch, account)
        else:
            body = compact[4:]
            bban = "".join("0" if i < 4 and c.isdigit() else str(rng.below(10)) if c.isdigit()
                           else rng.letters(1) for i, c in enumerate(body))
        fake = country + det.iban_check_digits(country, bban) + bban
        return _overlay(value, fake) or fake

    def _card(self, value: str, rng: _Rng) -> Optional[str]:
        digits = re.sub(r"\D", "", value)
        if len(digits) < 12:
            return _shape(value, rng, keep_prefix=False)
        prefix = {"4": "424242", "5": "555555", "3": "378282", "6": "601111"}.get(digits[0], "424242")
        body = prefix + rng.digits(len(digits) - len(prefix) - 1)
        fake = body + det.luhn_check_digit(body)
        return _overlay(value, fake) or fake

    def _nir(self, value: str, rng: _Rng) -> Optional[str]:
        digits = re.sub(r"\D", "", value)
        if len(digits) == 9:  # a US social security number: SSA advertising range 987-65-4320 to 4329
            fake = "98765432" + str(rng.below(10))
            return _overlay(value, fake) or fake
        if len(digits) not in (13, 15):
            return _shape(value, rng, keep_prefix=False)
        body = (rng.pick("12") + rng.digits(2) + f"{rng.between(1, 12):02d}" + f"{rng.between(1, 95):02d}"
                + rng.digits(3) + rng.digits(3))
        key = f"{97 - int(body) % 97:02d}"
        fake = body + (key if len(digits) == 15 else "")
        return _overlay(value, fake) or fake

    @staticmethod
    def _siren(rng: _Rng) -> str:
        body = rng.digits(8)
        return body + det.luhn_check_digit(body)

    def _siret(self, rng: _Rng) -> str:
        body = self._siren(rng) + rng.digits(4)
        return body + det.luhn_check_digit(body)

    def _vat(self, value: str, rng: _Rng) -> Optional[str]:
        compact = re.sub(r"[^0-9A-Za-z]", "", value).upper()
        if compact.startswith("FR") and len(compact) == 13:
            siren = self._siren(rng)
            fake = f"FR{det.vat_fr_key(siren):02d}{siren}"
            return _overlay(value, fake) or fake
        return _shape(value, rng)

    # ---- addresses
    def _address(self, value: str, rng: _Rng) -> Optional[str]:
        towns = self._lists["towns"]
        streets = self._lists["streets"]
        number = str(rng.between(1, 120))
        street = rng.pick(streets)
        postcode = re.search(r"(?<!\d)(\d{5}(?:-\d{4})?)(?!\d)|\b[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}\b", value)
        if postcode is None:
            return f"{number} {street}"
        before = value[:postcode.start()]
        sep = "\n" if "\n" in before else (", " if "," in before else " ")
        code, town = rng.pick(towns)
        if self.lang == "en" and "," in town:  # US: "City, ST ZIP"
            return f"{number} {street}{sep}{town} {code}"
        if self.lang == "en":
            return f"{number} {street}{sep}{town} {code}"
        return f"{number} {street}{sep}{code} {town}"

    # ------------------------------------------------------------------ rendering
    @staticmethod
    def _mapping(src: str, dst: str) -> Optional[dict[str, str]]:
        a = [r for r, _s, _e in _runs(src)]
        b = [r for r, _s, _e in _runs(dst)]
        if not a or len(a) != len(b):
            return None
        return {fold_word(x): y for x, y in zip(a, b)}

    def _render_name(self, src: str, dst: str, surface: str, need: str, reverse: bool = False) -> str:
        """``surface`` (a mention written with the words of ``src``) rewritten with the words of ``dst``."""
        mapping = self._mapping(src, dst)
        runs = _runs(surface)
        if mapping is None or not runs:
            return _case_like(surface, dst)
        src_runs = [r for r, _s, _e in _runs(src)]
        initials = [fold_word(r)[:1] for r in src_runs]
        dst_runs = [r for r, _s, _e in _runs(dst)]
        given: Optional[str] = None
        declared: frozenset[str] = frozenset()
        if not reverse:
            given = self._given_word(src, src_runs, dst_runs)
            declared = self._declared(src)
        out: list[str] = []
        pos = 0
        used_initial = 0
        for run, s, e in runs:
            f = fold_word(run)
            rep: Optional[str] = mapping.get(f)
            possessive = s > 0 and surface[s - 1] in "'’" and len(f) == 1  # "Dupontel's": not an initial
            if rep is None and not reverse and len(f) >= 2 and f.isalpha():
                # v0.3.1 (cause A): the nicknames and short forms the masker's resolution knows. A form of a word of
                # the real value (French and English tables, prefix forms: "Reggie", "Millie") -> the fake of that word;
                # a nickname the dossier declares (directory, quotes in the value) -> the fake's first given name
                rep = self._form_target(f, src_runs, dst_runs) or (given if f in declared else None)
            if rep is None and len(f) == 1 and not possessive:
                while used_initial < len(initials) and initials[used_initial] != f:
                    used_initial += 1
                if used_initial < len(initials):
                    rep = dst_runs[used_initial][:1]
                    used_initial += 1
            if rep is None and 2 <= len(f) <= 6 and f.isalpha():
                rep = self._acronym_target(f, src, dst) or self._joined_initials(f, src_runs, dst_runs)
            if rep is None and len(f) >= 3:
                rep = self._nick_target(f, src_runs, dst_runs)
            if rep is None and len(runs) == 1 and len(f) >= 5:
                rep = self._typo_target(f, src_runs, dst_runs)
            if rep is None and given and need in _GIVEN_NEEDS and len(f) >= 2 and f.isalpha() \
                    and not mention_word_fixed("PERSON", run, surface, s, src):
                # v0.3.1 (cause A): the masker linked the mention as a given-name form (a first name or nickname
                # alone, a nickname with the family name): a word that maps to nothing is that given name. Any
                # other word that maps to nothing is copied, and the leak check then fails closed (the code)
                rep = given
            if rep is None:
                rep = run
            out.append(surface[pos:s])
            out.append(rep if rep == run else _case_like(run, rep))
            pos = e
        out.append(surface[pos:])
        return "".join(out)

    @staticmethod
    def _form_target(f: str, src_runs: list[str], dst_runs: list[str]) -> Optional[str]:
        """A nickname or short form of a word of the real value, as the masker's resolution reads them (French and
        English tables, prefix forms) -> the fake word in its place (v0.3.1)."""
        for r, d in zip(src_runs, dst_runs):
            fr = fold_word(r)
            if len(fr) >= 3 and len(d) >= 2 and fr != f and f in _given_forms(fr):
                return d
        return None

    def _given_word(self, src: str, src_runs: list[str], dst_runs: list[str]) -> Optional[str]:
        """The fake word in place of the first given name of the real person ``src`` (a capitalised particle that
        opens a name without a given name counts: "Al Brightwater"); None when there is none, or when it is an initial
        (v0.3.1)."""
        try:
            firsts = nm.split_person(src)[0]
        except Exception:  # noqa: BLE001 - never fail a rendering on a name helper
            return None
        target = ""
        if firsts:
            r0 = _runs(firsts[0])
            target = fold_word(r0[0][0]) if r0 else ""
        else:
            for r, s, _e in _runs(src):
                if nm.is_title_word(r):
                    continue
                if _leading_particle(r, src, s):
                    target = fold_word(r)
                break
        if not target:
            return None
        for r, d in zip(src_runs, dst_runs):
            if fold_word(r) == target:
                return d if len(d) >= 2 else None
        return None

    @staticmethod
    def _acronym_target(f: str, src: str, dst: str) -> Optional[str]:
        """The initials group or company acronym of ``src`` that reads ``f`` -> the same group of ``dst``
        ("BVQ" of "Brasseries Velanor de Quimpertin" -> "BOV" of the fake)."""
        for fn in (nm.person_initials, nm.org_acronyms):
            try:
                a, b = fn(src), fn(dst)
            except Exception:  # noqa: BLE001 - never fail a replacement on a name helper
                continue
            if len(a) == len(b):
                for x, y in zip(a, b):
                    if x.lower() == f:
                        return y
        return None

    @staticmethod
    def _joined_initials(f: str, src_runs: list[str], dst_runs: list[str]) -> Optional[str]:
        """"jpd" for "Jean-Pierre Dupont" -> "asv" of the fake; particles and titles may be skipped."""
        letters = [(fold_word(r)[:1], fold_word(r) in nm.PARTICLES or fold_word(r) in ("les", "l")
                    or fold_word(r) in nm.TITLE_WORDS) for r in src_runs]
        for drop in (False, True):
            cand = [(i, c) for i, (c, part) in enumerate(letters) if not (drop and part)]
            if "".join(c for _i, c in cand) == f:
                return "".join(dst_runs[i][:1] for i, _c in cand)
        # a sub-sequence of consecutive runs (first name initials + surname, without the middle names)
        n = len(src_runs)
        for i in range(n):
            for j in range(i + 2, n + 1):
                if "".join(fold_word(r)[:1] for r in src_runs[i:j]) == f:
                    return "".join(r[:1] for r in dst_runs[i:j])
        return None

    @staticmethod
    def _nick_target(f: str, src_runs: list[str], dst_runs: list[str]) -> Optional[str]:
        """A diminutive or a prefix of a source word ("Steph" of "Stéphanie") -> the whole destination word."""
        for r, d in zip(src_runs, dst_runs):
            fr = fold_word(r)
            if len(fr) >= 5 and len(f) >= 4 and fr.startswith(f):
                return d
            if f in lx.nicknames().get(fr, ()):
                return d
        return None

    @staticmethod
    def _typo_target(f: str, src_runs: list[str], dst_runs: list[str]) -> Optional[str]:
        """A one-edit typo of a 6+ letter source word -> its destination word."""
        from .masker import _dl1  # local import: masker imports this module

        found = [d for r, d in zip(src_runs, dst_runs) if len(fold_word(r)) >= 6 and _dl1(fold_word(r), f)]
        return found[0] if len(found) == 1 else None

    def _render_other(self, ent: Entity, src: str, dst: str, surface: str, reverse: bool = False) -> str:
        kind = ent.kind
        if surface == src or normalize(surface) == normalize(src):
            return dst  # the stored form itself: its counterpart, exactly as stored
        if kind == "BIRTHDATE":
            days = self.date_offset(ent)
            shifted = shift_date_text(surface, -days if reverse else days)
            return shifted if shifted is not None else dst
        if kind in ("PHONE", "IBAN", "NIR", "SIREN", "SIRET", "VAT", "CARD", "CASE_REF", "ID_CODE",
                    "EMPLOYEE_ID", "ACCOUNT_NUMBER"):
            chars = "".join(c for c in dst if c.isalnum())
            over = _overlay(surface, chars)
            if over is not None:
                return over
            return dst
        return dst


#: Country calling codes recognised when a phone number starts with "+" or "00" (ITU list, 1 to 3 digits).
_COUNTRY_CODES = frozenset(
    ["1", "7", "20", "27", "30", "31", "32", "33", "34", "36", "39", "40", "41", "43", "44", "45", "46", "47", "48",
     "49", "51", "52", "53", "54", "55", "56", "57", "58", "60", "61", "62", "63", "64", "65", "66", "81", "82", "84",
     "86", "90", "91", "92", "93", "94", "95", "98"]
    + [str(n) for n in range(210, 300)] + [str(n) for n in range(350, 400)] + [str(n) for n in range(420, 430)]
    + [str(n) for n in range(500, 600)] + [str(n) for n in range(670, 700)] + [str(n) for n in range(850, 900)]
    + [str(n) for n in range(960, 1000)]
)
