"""Language detection, per-language lexicon bundles and the English name rules (v0.2, owner LOCALE-EN).

Three things live here (design: docs/V02_DESIGN.md sections 4.1, 4.3, 4.6):

1. :func:`detect_language` decides "fr" or "en" for one text, deterministically, with two small tables of
   function words and the accented letters of French. :func:`casual_text` says whether a text reads as chat
   (lower-case line starts, mention markers). :func:`language_setting` reads ``RAGBASE_SHIELD_LANGUAGE``.
2. :func:`bundle` returns the :class:`Bundle` of a language: every word list the name rules need, loaded once.
   The English bundle comes from the five ``en_*.txt`` files of the lexicon package, which carry sections
   (``# [section]`` lines). The French bundle is assembled from the v0.1 loaders and ``names.py`` constants, so
   that one API serves both languages. The French stop list is never applied to an English text, nor the reverse.
3. :func:`detect_names_en` finds people and companies in English text with rules only (no model): titles, first
   name + surname, "Surname, First", greetings and sign-offs, speaker labels, role words, legal forms, suffix
   words, "X & Y" firms, trigger words and possessives. :func:`protected_intervals_en` gives the intervals an
   English text must keep in clear (courts, agencies, statutes, section references, whitelist phrases).

Everything is standard library plus the sibling engine modules. No network, no model.

Integration contract (what the other owners call):

- ``detect_names_en(text) -> list[names.RuleSpan]`` replaces ``names.detect_names(text)`` when the language is
  "en" (the French rules carry the French stop list and must not run on English text).
- ``protected_intervals_en(text, folded, ftoks)`` replaces ``names.protected_intervals`` for English text.
- ``detectors.DETECTORS_EN`` (section ``# --- en ---`` of ``detectors.py``) must be run by ``detect_candidates``
  when the language is "en" (each function carries ``langs = ("en",)``).
- ``title_before_re(lang)`` / ``salute_before_re(lang)`` give the masker its ``_TITLE_BEFORE_RE`` and
  ``_SALUTE_BEFORE_RE`` for the language; ``dictionary_word(tok, lang)`` its ``_dictionary_word``.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Collection, Iterator, Optional

from . import lexicon as lx
from .detectors import IntervalSet
from .normalize import Folded, PhraseMatcher, fold_word, phrase_tokens

__all__ = [
    "LANGUAGES",
    "detect_language",
    "casual_text",
    "language_setting",
    "Bundle",
    "bundle",
    "EN_FILES",
    "read_sections",
    "is_first_name",
    "first_name_sex",
    "is_stopword",
    "is_vocab_word",
    "is_place",
    "is_acronym",
    "is_title",
    "legal_form",
    "is_name_word",
    "is_org_word",
    "is_distinctive",
    "dictionary_word",
    "detect_names_en",
    "protected_intervals_en",
    "whitelist_matcher_en",
    "own_company_vendor_setting",
    "title_before_re",
    "salute_before_re",
]

LANGUAGES = ("fr", "en")

# ---------------------------------------------------------------------------
# 1. Language detection
# ---------------------------------------------------------------------------

# Function words of each language that are NOT words of the other one ("a", "on", "me", "car", "son", "plus"
# belong to both and carry no weight). Keys are folded (no accent). Weights: 2 for the most frequent words.
_EN_WORDS: dict[str, int] = {
    "the": 2, "and": 2, "of": 2, "to": 2, "is": 2, "that": 2, "with": 2, "for": 2, "you": 2, "this": 2,
    "are": 1, "was": 1, "were": 1, "be": 1, "been": 1, "have": 1, "has": 1, "had": 1, "not": 1, "it": 1,
    "we": 1, "our": 1, "they": 1, "their": 1, "will": 1, "would": 1, "can": 1, "could": 1, "should": 1,
    "please": 1, "thanks": 1, "if": 1, "there": 1, "here": 1, "about": 1, "into": 1, "which": 1, "who": 1,
    "what": 1, "when": 1, "where": 1, "than": 1, "then": 1, "also": 1, "just": 1, "only": 1, "so": 1,
    "yes": 1, "do": 1, "does": 1, "did": 1, "get": 1, "got": 1, "know": 1, "let": 1, "my": 1, "him": 1,
    "his": 1, "her": 1, "she": 1, "he": 1, "them": 1, "up": 1, "out": 1, "some": 1, "more": 1, "most": 1,
    "other": 1, "new": 1, "over": 1, "after": 1, "before": 1, "because": 1, "while": 1, "during": 1,
    "between": 1, "under": 1, "through": 1, "your": 1, "any": 1, "all": 1, "from": 1, "by": 1, "at": 1,
    "in": 1, "an": 0, "or": 0, "on": 0, "a": 0, "as": 0, "but": 0, "me": 0, "us": 0, "no": 0,
    "hi": 1, "hello": 1, "regards": 1, "dear": 1, "email": 1, "meeting": 1, "team": 1, "need": 1,
    "make": 1, "like": 1, "want": 1, "think": 1, "see": 1, "call": 1, "send": 1, "sent": 1, "work": 1,
    "time": 1, "today": 1, "tomorrow": 1, "week": 1, "year": 1, "day": 1,
}
_FR_WORDS: dict[str, int] = {
    "le": 2, "la": 2, "les": 2, "des": 2, "du": 2, "de": 2, "et": 2, "que": 2, "qui": 2, "pour": 2,
    "dans": 2, "est": 2, "une": 2, "un": 1, "au": 1, "aux": 1, "sur": 1, "avec": 1, "par": 1, "pas": 1,
    "ne": 1, "ce": 1, "cette": 1, "ces": 1, "il": 1, "elle": 1, "ils": 1, "elles": 1, "nous": 1, "vous": 1,
    "je": 1, "tu": 1, "se": 1, "sa": 1, "ses": 1, "leur": 1, "leurs": 1, "mais": 1, "ou": 1, "donc": 1,
    "ni": 1, "ete": 1, "etre": 1, "avoir": 1, "fait": 1, "faire": 1, "tres": 1, "bien": 1, "aussi": 1,
    "comme": 1, "tout": 1, "tous": 1, "toute": 1, "toutes": 1, "cet": 1, "ont": 1, "dont": 1, "sans": 1,
    "sous": 1, "vers": 1, "chez": 1, "entre": 1, "apres": 1, "avant": 1, "depuis": 1, "pendant": 1,
    "lors": 1, "selon": 1, "ainsi": 1, "alors": 1, "ici": 1, "oui": 1, "non": 0, "bonjour": 1, "merci": 1,
    "cordialement": 1, "madame": 1, "monsieur": 1, "maitre": 1, "societe": 1, "article": 0, "contrat": 1,
    "dossier": 1, "tribunal": 1, "cour": 1, "jugement": 1, "salarie": 1, "avocat": 1, "sont": 1, "etait": 1,
    "etaient": 1, "sera": 1, "peut": 1, "doit": 1, "cela": 1, "ceci": 1, "celui": 1, "celle": 1, "ceux": 1,
    "notre": 1, "votre": 1, "nos": 1, "vos": 1, "mon": 1, "ton": 1, "mes": 1, "tes": 1, "lui": 1, "eux": 1,
    "y": 1, "en": 1, "ca": 1, "deja": 1, "encore": 1, "jamais": 1, "toujours": 1, "peu": 1, "beaucoup": 1,
    "moins": 1, "aupres": 1, "lorsque": 1, "puisque": 1, "quand": 1, "pourquoi": 1, "comment": 1,
    "demande": 1, "suite": 1, "objet": 1, "piece": 1, "pieces": 1, "jointe": 1, "jour": 1, "mois": 1,
    "annee": 1, "semaine": 1, "aujourd": 1, "demain": 1, "hier": 1, "veuillez": 1, "merci": 1,
}
_LETTER_RE = re.compile(r"[^\W\d_]+")
_FR_ACCENTED = frozenset("éèêëàâäùûüôöîïçœæÉÈÊËÀÂÄÙÛÜÔÖÎÏÇŒÆ")
_MIN_LETTERS = 40


def detect_language(text: str, default: str = "fr") -> str:
    """``"fr"`` or ``"en"`` for one text. Deterministic, standard library only.

    Score of a language = sum of the weights of its function words found in the text, plus 0.5 per accented
    letter for French. A language wins when its score is at least 3 and at least 1.5 times the other score.
    Otherwise, and for a text under 40 letters, ``default`` (which must be "fr" or "en").
    """
    default = "en" if str(default).lower().startswith("en") else "fr"
    if not text:
        return default
    letters = 0
    en = fr = 0.0
    for m in _LETTER_RE.finditer(text):
        w = m.group()
        letters += len(w)
        key = fold_word(w)
        en += _EN_WORDS.get(key, 0)
        fr += _FR_WORDS.get(key, 0)
    if letters < _MIN_LETTERS:
        return default
    fr += 0.5 * sum(1 for ch in text if ch in _FR_ACCENTED)
    if en >= 3 and en >= 1.5 * fr:
        return "en"
    if fr >= 3 and fr >= 1.5 * en:
        return "fr"
    return default


_CHAT_MARKERS = ("<@", "@channel", "@here", "@everyone")


def casual_text(text: str) -> bool:
    """Chat-like text: at least 30 % of the non-empty lines start with a lower-case letter, or the text holds
    chat mention markers (``<@U...>``, ``@channel``). Used for the "casual" lower-case name variants."""
    if not text:
        return False
    if any(mk in text for mk in _CHAT_MARKERS):
        return True
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return False
    lower = sum(1 for ln in lines if ln[0].isalpha() and ln[0].islower())
    return lower >= 0.3 * len(lines)


def language_setting() -> str:
    """``RAGBASE_SHIELD_LANGUAGE``: "auto" (default), "fr" or "en"."""
    v = os.environ.get("RAGBASE_SHIELD_LANGUAGE", "auto").strip().lower()
    if v.startswith("fr"):
        return "fr"
    if v.startswith("en"):
        return "en"
    return "auto"


def own_company_vendor_setting() -> bool:
    """v0.4 (P1, policy option, OFF by default): ``RAGBASE_SHIELD_OWN_COMPANY_VENDOR`` "1", "true", "yes" or "on"
    masks a public tool or vendor of the whitelist ([vendors]) when it is the dossier's own company, that is the
    label of the dossier's main e-mail domain. Default "0": every vendor stays readable, as before."""
    v = os.environ.get("RAGBASE_SHIELD_OWN_COMPANY_VENDOR", "0").strip().lower()
    return v in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# 2. Lexicon bundles
# ---------------------------------------------------------------------------

#: English lexicon files (owner LOCALE-EN) and the sections each one carries.
EN_FILES = {
    "en_first_names.txt": ("census", "supplement", "nicknames"),
    "en_stopwords.txt": ("stopwords", "months_days", "greetings", "titles"),
    "en_common_words.txt": ("common", "vocab", "places", "acronyms"),
    "en_legal_forms.txt": ("forms", "suffix", "generic", "trigger"),
    "en_whitelist.txt": ("whitelist", "vendors", "legal_terms", "roles"),
}

_SECTION_RE = re.compile(r"^#\s*\[([a-z_]+)\]\s*$")


def read_sections(name: str) -> dict[str, list[str]]:
    """Raw entries of a lexicon file grouped by ``# [section]`` header (entries before the first header go to
    ``"default"``). Comments and blank lines are removed; entries are not folded."""
    text = resources.files(lx.__name__).joinpath(name).read_text(encoding="utf-8")
    out: dict[str, list[str]] = {}
    current = "default"
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            m = _SECTION_RE.match(line)
            if m:
                current = m.group(1)
            continue
        out.setdefault(current, []).append(line)
    return out


def _fold_set(entries: list[str]) -> frozenset[str]:
    return frozenset(f for f in (fold_word(e) for e in entries) if f)


def _form_key(surface: str) -> tuple[str, bool]:
    """Key of a legal form: folded, inner dots removed, with a flag for a trailing dot ("L.P." -> ("lp", True))."""
    low = fold_word(surface)
    dotted = low.endswith(".")
    low = low.replace(".", "").replace(" ", "")
    return low, dotted


@dataclass(frozen=True)
class Bundle:
    """Every word list of one language, folded (lower case, no accents)."""

    lang: str
    first_names: frozenset[str]            # strong first-name signal (a person alone, with a cue)
    all_first_names: frozenset[str]        # every first name, the ambiguous ones included
    first_name_sex: dict[str, str]         # folded name -> "F" | "M" | "U"
    nicknames: dict[str, frozenset[str]]   # full first name -> its diminutives that are not prefixes
    stopwords: frozenset[str]
    months_days: frozenset[str]
    greetings: frozenset[str]
    titles: frozenset[str]
    common_words: frozenset[str]           # common words that are also surnames
    vocab: frozenset[str]                  # capitalised common words: never a name by themselves
    places: frozenset[str]
    acronyms: frozenset[str]
    legal_forms: dict[str, str]            # form key -> "strong" | "weak" ("inc", "co", "co." ...)
    legal_forms_spelled: tuple[tuple[str, ...], ...]  # multi-word forms as folded token tuples
    org_suffixes: frozenset[str]
    org_generic: frozenset[str]
    org_triggers: frozenset[str]
    whitelist_phrases: tuple[str, ...]
    legal_terms: frozenset[str]
    roles: frozenset[str]
    #: v0.4 (P1): the public tools and vendors of the whitelist ([vendors] section; a subset of
    #: ``whitelist_phrases``). English only; the French bundle has none.
    vendor_phrases: frozenset[str] = frozenset()

    @property
    def excluded(self) -> frozenset[str]:
        """Words that are never a name word in this language."""
        return frozenset(self.stopwords | self.months_days | self.greetings | self.titles | self.legal_terms
                         | self.roles | frozenset(self.legal_forms) | self.org_triggers)


def _load_en() -> Bundle:
    fn = read_sections("en_first_names.txt")
    sex: dict[str, str] = {}
    for sec in ("census", "supplement", "default"):
        for entry in fn.get(sec, ()):
            name, _, s = entry.partition("\t")
            name = fold_word(name.strip())
            if not name or "=" in entry:
                continue
            s = (s.strip() or "U").upper()[:1]
            sex[name] = s if s in "FMU" else "U"
    nick: dict[str, set[str]] = {}
    for entry in fn.get("nicknames", ()):
        short, sep, full = entry.partition("=")
        short, full = fold_word(short), fold_word(full)
        if sep and short and full:
            nick.setdefault(full, set()).add(short)
            sex.setdefault(short, sex.get(full, "U"))
    sw = read_sections("en_stopwords.txt")
    cw = read_sections("en_common_words.txt")
    lf = read_sections("en_legal_forms.txt")
    wl = read_sections("en_whitelist.txt")
    forms: dict[str, str] = {}
    spelled: list[tuple[str, ...]] = []
    for entry in lf.get("forms", ()):
        form, _, strength = entry.partition("\t")
        form = form.strip()
        if " " in form:
            toks = phrase_tokens(form)
            if toks:
                spelled.append(toks)
            continue
        key, dotted = _form_key(form)
        strength = strength.strip().lower() or "strong"
        forms[key + ("." if dotted else "")] = "weak" if strength == "weak" else "strong"
    stop = _fold_set(sw.get("stopwords", []))
    months = _fold_set(sw.get("months_days", []))
    greet = _fold_set(sw.get("greetings", []))
    titles = _fold_set(sw.get("titles", []))
    common = _fold_set(cw.get("common", []))
    vocab = _fold_set(cw.get("vocab", []))
    places = _fold_set(cw.get("places", []))
    acr = _fold_set(cw.get("acronyms", []))
    suffixes = _fold_set(lf.get("suffix", []))
    generic = _fold_set(lf.get("generic", []))
    triggers = _fold_set(lf.get("trigger", []))
    legal = _fold_set(wl.get("legal_terms", []))
    roles = _fold_set(wl.get("roles", []))
    vendors = _fold_set(wl.get("vendors", []))
    whitelist = tuple(sorted(_fold_set(wl.get("whitelist", [])) | vendors))
    all_first = frozenset(sex)
    ambiguous = stop | months | greet | titles | common | vocab | places | acr | legal | roles | suffixes \
        | generic | frozenset(forms) | triggers
    strong = frozenset(n for n in all_first if n not in ambiguous and len(n) >= 3)
    return Bundle(
        lang="en", first_names=strong, all_first_names=all_first, first_name_sex=sex,
        nicknames={k: frozenset(v) for k, v in nick.items()}, stopwords=stop, months_days=months,
        greetings=greet, titles=titles, common_words=common, vocab=vocab, places=places, acronyms=acr,
        legal_forms=forms, legal_forms_spelled=tuple(spelled), org_suffixes=suffixes, org_generic=generic,
        org_triggers=triggers, whitelist_phrases=whitelist, legal_terms=legal, roles=roles,
        vendor_phrases=vendors,
    )


def _optional_set(name: str) -> frozenset[str]:
    """A folded set from an optional lexicon file (another owner may add it later)."""
    try:
        return frozenset(f for f in (fold_word(x) for x in lx.read_lines(name)) if f)
    except (FileNotFoundError, OSError):
        return frozenset()


def _load_fr() -> Bundle:
    from . import names as nm  # local import: names.py must stay importable without this module

    forms = {k.replace(".", "") + ("." if k.endswith(".") else ""): v for k, v in nm.LEGAL_FORMS.items()}
    return Bundle(
        lang="fr", first_names=lx.first_names(), all_first_names=lx.all_first_names(), first_name_sex={},
        nicknames=lx.nicknames(), stopwords=lx.stopwords(), months_days=lx.months_days(),
        greetings=frozenset(nm._GREETINGS), titles=nm.TITLE_WORDS, common_words=lx.common_words(),
        vocab=_optional_set("vocab_fr.txt"), places=_optional_set("places_fr.txt") | frozenset(nm._PLACE_WORDS),
        acronyms=lx.acronym_stoplist() | lx.instrument_terms(), legal_forms=forms, legal_forms_spelled=(),
        org_suffixes=nm._ORG_SUFFIXES, org_generic=nm.ORG_GENERIC, org_triggers=nm.ORG_TRIGGERS,
        whitelist_phrases=lx.whitelist_phrases(), legal_terms=lx.legal_terms(), roles=nm._ROLE_WORDS,
    )


@lru_cache(maxsize=None)
def _bundle(lang: str) -> Bundle:
    return _load_en() if lang == "en" else _load_fr()


def bundle(lang: str = "fr") -> Bundle:
    """The lexicon bundle of ``lang`` ("fr" or "en"), loaded once per process."""
    return _bundle("en" if str(lang).lower().startswith("en") else "fr")


# ---------------------------------------------------------------------------
# 3. Word predicates (take a surface word and a bundle)
# ---------------------------------------------------------------------------

_ACRONYM_RE = re.compile(r"^[A-Z](?:[A-Z&.]){1,5}$")
_GENERATIONAL = frozenset({"jr", "sr", "ii", "iii", "iv", "junior", "senior"})
_CIVILITY = frozenset({"mr", "mrs", "ms", "mx", "miss", "dr", "prof", "professor", "madam", "madame", "mister",
                       "sir", "dame", "rev", "reverend", "hon", "honorable", "honourable", "esq"})
_PARTICLES_EN = frozenset({"de", "da", "di", "del", "della", "der", "van", "von", "la", "le", "du", "des", "dos",
                           "das", "bin", "ibn", "al", "el", "mac", "mc", "st", "te", "ten", "ter", "y"})
_NAME_CONNECT = frozenset({"&", "and", "of", "the", "for"})
# v0.4 (T3): the word of a firm name after "&" or "and" ("Halberd & Co.", "Velltry and Company")
_FIRM_WORDS = frozenset({"co", "company", "companies"})
# offices written after "Company" ("the Company Secretary"): never part of a firm name
_COMPANY_OFFICES = frozenset({"secretary", "secretaries", "director", "directors", "officer", "officers", "president",
                              "treasurer", "chairman", "chairwoman", "chairperson", "chair", "representative",
                              "representatives", "accountant", "auditor", "auditors", "seal", "stamp"})


def _low(word: str) -> str:
    return fold_word(word).rstrip(".")


def is_first_name(word: str, b: Bundle, *, strong: bool = True) -> bool:
    """``word`` is a first name of the language. ``strong=True``: one that is not also a common word, a place,
    a month or a title (a person on its own with a cue); ``strong=False``: any first name of the lexicon.
    Hyphenated first names count when every part is a first name."""
    low = _low(word)
    pool = b.first_names if strong else b.all_first_names
    if low in pool:
        return True
    if "-" in low:
        parts = [p for p in low.split("-") if p]
        return len(parts) > 1 and all(p in pool for p in parts)
    return False


def first_name_sex(word: str, b: Bundle) -> str:
    """``"F"``, ``"M"`` or ``"U"`` for a first name (``"U"`` when unknown)."""
    return b.first_name_sex.get(_low(word).split("-")[0], "U")


def is_stopword(word: str, b: Bundle) -> bool:
    return _low(word) in b.stopwords


def is_vocab_word(word: str, b: Bundle) -> bool:
    """A common word of the language (vocabulary, stop word, month, legal term): capitalised, it is still not a
    name by itself."""
    low = _low(word)
    return low in b.vocab or low in b.stopwords or low in b.months_days or low in b.legal_terms


def is_place(word: str, b: Bundle) -> bool:
    return _low(word) in b.places


def is_acronym(word: str, b: Bundle) -> bool:
    """An all-capital token of 2 to 6 letters, or a known acronym in any case: never a person."""
    w = word.rstrip(".")
    if _ACRONYM_RE.match(w) and sum(c.isalpha() for c in w) >= 2:
        return True
    return _low(word) in b.acronyms


def is_title(word: str, b: Bundle) -> bool:
    return _low(word) in b.titles


def legal_form(word: str, b: Bundle) -> Optional[str]:
    """``"strong"``, ``"weak"`` or ``None`` for a legal form token ("Inc.", "LLC", "Ltd", "Co.", "S.A.").
    A short weak form (2 letters: "SA", "AG", "LP", "PC", "CO") must be written in capitals or with dots."""
    key, dotted = _form_key(word)
    if not key:
        return None
    strength = b.legal_forms.get(key + ".") if dotted else None
    if strength is None:
        strength = b.legal_forms.get(key)
    if strength is None:
        return None
    core = word.rstrip(".")
    letters = [c for c in core if c.isalpha()]
    if not letters or not letters[0].isupper():
        return None
    if strength == "weak" and len(key) <= 2 and not dotted and not all(c.isupper() for c in letters):
        return None
    return strength


def is_name_word(word: str, b: Bundle, *, kind: str = "PERSON", allow_common: bool = True,
                 allow_place: bool = True) -> bool:
    """A capitalised word that can be part of a name of ``kind``.

    PERSON: not a stop word, month, greeting, title, legal term, role, legal form, company suffix, acronym;
    a vocabulary word only when it is also a surname-like common word (``allow_common``), a place
    (``allow_place``) or a first name. ORG: not a stop word, month, greeting, title, legal form or acronym of
    2 letters; vocabulary and generic words are accepted (the company needs a distinctive word elsewhere).
    """
    core = word.rstrip(".")
    letters = sum(c.isalpha() for c in core)
    if letters < 2 or not core[:1].isupper():
        return False
    low = _low(word)
    if kind == "ORG":
        if legal_form(word, b) or low in b.legal_forms:
            return False
        if low in b.months_days or low in b.greetings or low in _CIVILITY:
            return False
        if low in b.stopwords:
            return (len(low) == 2 and core.isupper()) or low in b.org_generic  # "IT Services", "First National"
        return True
    if low in b.stopwords or low in b.months_days or low in b.greetings or low in b.titles:
        return False
    if legal_form(word, b) or low in b.legal_forms:
        return False
    if low in b.legal_terms or low in b.roles or low in b.org_suffixes or low in b.org_generic:
        return False
    if is_acronym(word, b) and "-" not in core:
        return False
    if low in b.vocab:
        if allow_common and low in b.common_words:
            return True
        if allow_place and low in b.places:
            return True
        return low in b.all_first_names
    if low in b.places and not allow_place:
        return False
    return True


def is_org_word(word: str, b: Bundle) -> bool:
    return word == "&" or is_name_word(word, b, kind="ORG")


def is_distinctive(word: str, b: Bundle) -> bool:
    """A capitalised word that can only be a proper name: not a vocabulary, common, place, generic, legal,
    stop word, not a first name, not an acronym, at least 3 letters."""
    if not is_name_word(word, b, kind="ORG"):
        return False
    low = _low(word)
    core = word.rstrip(".")
    if sum(c.isalpha() for c in core) < 3 or is_acronym(word, b):
        return False
    return not (low in b.vocab or low in b.common_words or low in b.places or low in b.org_generic
                or low in b.org_suffixes or low in b.legal_terms or low in b.roles or low in b.all_first_names
                or low in b.acronyms)


def dictionary_word(tok: str, lang: str = "fr") -> bool:
    """A common, function, legal, calendar, place or generic business word, or a first name (folded token).
    The masker's ``_dictionary_word`` for the language."""
    b = bundle(lang)
    return (tok in b.vocab or tok in b.stopwords or tok in b.months_days or tok in b.legal_terms
            or tok in b.common_words or tok in b.places or tok in b.org_generic or tok in b.all_first_names
            or tok in b.titles or tok in b.greetings or tok in b.roles)


# ---------------------------------------------------------------------------
# 4. English name rules
# ---------------------------------------------------------------------------

_TOK_RE = re.compile(r"[^\W\d_](?:[^\W_]|['’\-](?=[^\W\d_]))*\.?|\d+(?:[:.]\d+)*|&")
_POSS_RE = re.compile(r"['’][sS]$")
_DOT_ABBREV = frozenset({"mr", "mrs", "ms", "mx", "dr", "prof", "jr", "sr", "st", "inc", "corp", "co", "ltd",
                         "llc", "etc", "vs", "no", "e.g", "i.e", "capt", "lt", "sgt", "col", "gen", "rev", "hon",
                         "ave", "blvd", "rd", "pl", "ste", "apt", "dept", "div", "mt", "ft"})
_GAP_RE = re.compile(r"[   \"“”«»]*,?[   \"“”«»]*")
_COLUMN_RE = re.compile(r"\t|[   ]{3,}")
_LINE_RE = re.compile(r"[ \t  ]*,?[ \t  ]*\r?\n[ \t  ]*")
# v0.4 (T8): the cell separator of the product's table readings ("|") starts a cell as a line break starts a line
_SENT_END_RE = re.compile(r"(?:^|[.!?…]|\n|\|)[\s\"“”«»()\-–—•*]*$")
_PLACE_BEFORE = frozenset({"lake", "fort", "mount", "mt", "port", "saint", "st", "san", "santa", "cape", "camp",
                           "point", "los", "las", "el", "la", "new", "north", "south", "east", "west"})
_BUSINESS_NOUNS = frozenset({
    "ceo", "cfo", "cto", "coo", "cmo", "cio", "board", "revenue", "revenues", "results", "earnings", "quarter",
    "q1", "q2", "q3", "q4", "team", "teams", "staff", "employees", "office", "offices", "headquarters", "hq",
    "shares", "stock", "ipo", "merger", "acquisition", "sales", "pipeline", "roadmap", "platform", "product",
    "products", "customers", "clients", "users", "api", "app", "website", "pricing", "contract", "contracts",
    "invoice", "invoices", "account", "policy", "policies", "management", "leadership", "founder", "founders",
    "brand", "logo", "fleet", "factory", "plant", "plants", "warehouse", "stores", "branches", "subsidiary",
    "subsidiaries", "parent", "growth", "margins", "valuation", "investors", "shareholders", "payroll",
    "servers", "data", "systems", "infrastructure", "network", "engineers", "developers", "support",
})
_TRIGGER_AT_LABELS = frozenset({"at", "joined", "joining"})
# Words right after a sentence-initial first name that make it an imperative verb ("Ping me", "Bill them").
_IMPERATIVE_OBJECTS = frozenset({"me", "him", "her", "them", "us", "it", "the", "a", "an", "this", "that", "these",
                                 "those", "my", "your", "our", "his", "their", "its", "everyone", "everybody",
                                 "someone", "anyone", "all", "each", "every", "both", "yourself", "myself"})
_SPEAKER_PREFIX_RE = re.compile(r"[\s\[\]()<>*•\-–—\d:.,]*(?:[AaPp][Mm][\s\]):]*)?")
_ORG_OF_HEADS = frozenset({"bank", "university", "college", "institute", "school", "hospital", "church", "museum",
                           "society", "association", "federation", "academy", "foundation", "order", "house",
                           "brotherhood", "sisters", "friends", "guild", "chamber", "club"})
# v0.4 (T5, FM4): the two-letter postal codes of the US states, the federal district and the territories (USPS,
# public data). A code that is also a short legal form ("CO", "NV") is the state after a place name.
_US_STATE_CODES = frozenset((
    "al ak az ar ca co ct de fl ga hi id il in ia ks ky la me md ma mi mn ms mo mt ne nv nh nj nm ny nc nd oh ok "
    "or pa ri sc sd tn tx ut vt va wa wv wi wy dc pr gu vi as mp").split())
_STATE_AFTER_RE = re.compile(r"[ \t ]*(?:\d{5}(?:-\d{4})?(?!\d)|/[ \t]*[A-Z]{2}(?![A-Za-z]))")
_CODE_GLUE_RE = re.compile(r"-[A-Za-z0-9]*\d")


@dataclass
class _Tok:
    text: str
    start: int
    end: int
    low: str
    cap: bool
    upper: bool
    initial: bool = False
    dot: bool = False
    sent_end: bool = False
    digit: bool = False
    amp: bool = False
    poss: bool = False        # an English possessive "'s" was stripped
    comma: bool = False       # a comma sits between the previous token and this one
    newline: bool = False     # a line break sits between the previous token and this one
    column: bool = False      # a column gap (tab, 3+ spaces) sits before this one
    colon_after: bool = False  # the token is directly followed by ":"
    raw_end: int = 0           # end of the token with its possessive ("'s") included
    memo: Optional[dict] = None  # cached predicate results for this token


def _tokenize(text: str) -> list[_Tok]:
    out: list[_Tok] = []
    for m in _TOK_RE.finditer(text):
        raw, s, e = m.group(), m.start(), m.end()
        if raw == "&":
            out.append(_Tok(raw, s, e, "&", False, False, amp=True, raw_end=e))
            continue
        if raw[0].isdigit():
            out.append(_Tok(raw, s, e, raw, False, False, digit=True, raw_end=e))
            continue
        poss = False
        raw_end = e
        if _POSS_RE.search(raw) and len(raw) > 3 and not (e < len(text) and text[e].isalnum()):
            raw, e, poss = raw[:-2], e - 2, True
        dot = raw.endswith(".")
        base = raw[:-1] if dot else raw
        low = fold_word(base)
        letters = [c for c in base if c.isalpha()]
        cap = base[:1].isupper()
        upper = len(letters) >= 2 and all(c.isupper() for c in letters)
        t = _Tok(raw, s, e, low, cap, upper, poss=poss, raw_end=raw_end)
        if dot:
            if len(letters) == 1 and cap:
                t.initial, t.dot = True, True
            elif low in _DOT_ABBREV or (len(letters) <= 4 and upper and "." in base):
                t.dot = True
            else:
                t.sent_end, t.text, t.end = True, base, e - 1
        if t.end < len(text) and text[t.end] == ":":
            t.colon_after = True
        out.append(t)
    # Merge initials written without spaces: "J.P." -> one initial token.
    merged: list[_Tok] = []
    for t in out:
        if merged and merged[-1].initial and t.initial and text[merged[-1].end:t.start] in ("", "-"):
            prev = merged[-1]
            merged[-1] = _Tok(text[prev.start:t.end], prev.start, t.end, prev.low + t.low, True, False,
                              initial=True, dot=True, raw_end=t.raw_end)
            continue
        merged.append(t)
    for t in merged:
        if t.sent_end:
            t.raw_end = t.end
    for k in range(1, len(merged)):
        gap = text[merged[k - 1].raw_end:merged[k].start]
        merged[k].comma = "," in gap
        merged[k].newline = "\n" in gap or "\r" in gap
        merged[k].column = bool(_COLUMN_RE.search(gap))
    return merged


# v0.4 (T2): a nickname between single quotes or brackets inside a name ("Ottilie 'Otti' Wrexley", "Cornelius
# (Neil) Quenby"): the quote or the bracket is a delimiter, not a run break (double quotes already are, _GAP_RE).
_NICK_PAIRS = {"'": "'’", "‘": "’'", "(": ")", "[": "]"}
_NICK_OPEN_GAP_RE = re.compile(r"[ \t ]+(['‘(\[])")
_NICK_CLOSE_GAP_RE = re.compile(r"(['’)\]])[ \t ]+")


def _nickname_open(text: str, prev: _Tok, t: _Tok) -> bool:
    """``t`` is a one-word nickname between quotes or brackets right after the name word ``prev``: a capitalised
    first name of the lexicon or a short form of ``prev`` ("Otti" of "Ottilie"), closed right after it."""
    m = _NICK_OPEN_GAP_RE.fullmatch(text[prev.raw_end:t.start])
    if m is None or not prev.cap or not t.cap or t.digit or t.amp or t.initial or t.dot or t.poss:
        return False
    close = text[t.end:t.end + 1]
    if not close or close not in _NICK_PAIRS[m.group(1)]:
        return False
    if t.end + 1 < len(text) and text[t.end + 1].isalnum():
        return False
    return is_first_name(t.text, bundle("en"), strong=False) or (len(t.low) >= 2 and prev.low.startswith(t.low))


def _runs(text: str, toks: list[_Tok]) -> list[list[_Tok]]:
    """Runs of tokens separated only by spaces, quotes and commas; a sentence end, a line break, a column gap
    or any other punctuation starts a new run. v0.4 (T2): a nickname between single quotes or brackets after a
    name word stays in the run, and so does the capitalised word after its closing mark."""
    runs: list[list[_Tok]] = []
    cur: list[_Tok] = []
    nick: Optional[_Tok] = None
    for t in toks:
        if cur:
            prev = cur[-1]
            gap = text[prev.raw_end:t.start]
            same = bool(_GAP_RE.fullmatch(gap)) and (bool(gap) or prev.dot or prev.initial) and not _COLUMN_RE.search(gap)
            if not same and not prev.sent_end:
                if _nickname_open(text, prev, t):
                    same, nick = True, t
                    cur.append(t)
                    continue
                if nick is prev and t.cap and _NICK_CLOSE_GAP_RE.fullmatch(gap) and all(
                        is_name_word(w, bundle("en"), allow_common=False, allow_place=False)
                        for w in ((t.text, t.text[:-1]) if t.low.endswith("s") and len(t.low) > 4 else (t.text,))):
                    same = True  # the family name after the nickname, never a common word or its plural ("'Otti' Logistics")
            if prev.sent_end or not same:
                runs.append(cur)
                cur = []
        nick = None
        cur.append(t)
    if cur:
        runs.append(cur)
    return runs


def _starts_sentence(text: str, pos: int) -> bool:
    return bool(_SENT_END_RE.search(text[max(0, pos - 40):pos]))


@lru_cache(maxsize=None)
def _spelled_index(lang: str) -> dict[str, tuple[tuple[str, ...], ...]]:
    idx: dict[str, list[tuple[str, ...]]] = {}
    for phrase in bundle(lang).legal_forms_spelled:
        idx.setdefault(phrase[0], []).append(phrase)
    return {k: tuple(v) for k, v in idx.items()}


class _Rules:
    """One pass of the English rules over a text (holds the bundle and the output list)."""

    def __init__(self, text: str, b: Bundle):
        self.text = text
        self.b = b
        self.out: list[tuple[int, int, str, int, str]] = []

    # --- token predicates -------------------------------------------------------------------------------

    @staticmethod
    def _memo(t: _Tok, key: str, compute):
        memo = t.memo
        if memo is None:
            memo = t.memo = {}
        v = memo.get(key)
        if v is None:
            v = memo[key] = compute()
        return v

    def person_word(self, t: _Tok, *, allow_common: bool = True, allow_place: bool = True) -> bool:
        if t.digit or t.amp or t.initial or not t.cap:
            return False
        key = f"pw{int(allow_common)}{int(allow_place)}"
        return self._memo(t, key, lambda: is_name_word(t.text, self.b, kind="PERSON", allow_common=allow_common,
                                                     allow_place=allow_place))

    def org_word(self, t: _Tok) -> bool:
        if t.digit or t.initial:
            return False
        return t.amp or self._memo(t, "ow", lambda: is_name_word(t.text, self.b, kind="ORG"))

    def distinctive(self, t: _Tok) -> bool:
        return not (t.digit or t.amp or t.initial) and self._memo(t, "di", lambda: is_distinctive(t.text, self.b))

    def first(self, t: _Tok, *, strong: bool = True) -> bool:
        if t.digit or t.amp or t.initial or not t.cap:
            return False
        if t.low in self.b.months_days or t.low in self.b.titles:
            return False
        if strong and t.low in self.b.stopwords:
            return False
        return self._memo(t, "f1" if strong else "f0", lambda: is_first_name(t.text, self.b, strong=strong))

    def title(self, t: _Tok) -> bool:
        if t.digit or t.amp or t.initial or t.low not in self.b.titles:
            return False
        return t.cap or t.dot

    def form(self, t: _Tok) -> Optional[str]:
        if t.digit or t.amp or t.initial:
            return None
        return self._memo(t, "lf", lambda: legal_form(t.text, self.b) or "")  or None

    def form_misread(self, run: list[_Tok], i: int) -> bool:
        """v0.4 (T5, FM4): the legal-form token ``run[i]``, written in capitals with no dot, is something else:
        the prefix of a code ("LTD-78Q4"), a US state code after a place ("Vennick Reach, CO 81099", "Vennick
        Reach CO/UT"), or one code of a list of capital codes ("HRM, LTD and payroll", "HRM, LTD, PAYX"). A form
        written in mixed
        case or with a dot ("Ltd", "Co.") is always a legal form."""
        t = run[i]
        if self.b.lang != "en" or not t.upper or t.dot:
            return False
        after = self.text[t.end:t.end + 16]
        if _CODE_GLUE_RE.match(after):
            return True
        if t.low in _US_STATE_CODES and len(t.low) == 2 and (t.comma or _STATE_AFTER_RE.match(after)):
            return True
        if t.comma and i > 0 and run[i - 1].upper and not self.form(run[i - 1]) and i + 1 < len(run):
            nxt = run[i + 1]
            if not nxt.newline and not nxt.column and not nxt.digit and (
                    (not nxt.cap and not nxt.amp and not nxt.comma) or (nxt.comma and nxt.upper)):
                return True
        return False

    def spelled_form_at(self, run: list[_Tok], i: int) -> int:
        """Length in tokens of a spelled-out legal form starting at ``run[i]``, else 0."""
        for phrase in _spelled_index(self.b.lang).get(run[i].low, ()):
            n = len(phrase)
            if i + n <= len(run) and all(run[i + k].low == phrase[k] for k in range(1, n)):
                return n
        return 0

    def add(self, s: _Tok, e: _Tok, kind: str, prio: int, rule: str) -> None:
        if e.end > s.start:
            self.out.append((s.start, e.end, kind, prio, rule))

    # --- person spans ------------------------------------------------------------------------------------

    def take_person(self, run: list[_Tok], j: int, *, max_words: int = 4, need_surname: bool = True,
                    allow_common: bool = True) -> Optional[int]:
        """End index (exclusive) of a person name starting at ``run[j]``: name words, initials, particles between
        name words, a hyphenated surname, a generational suffix. ``need_surname``: at least one non-initial word."""
        n = len(run)
        k = j
        words = 0
        last_name = -1
        while k < n and k - j < max_words + 2:
            t = run[k]
            if k > j and (t.comma or t.column or t.newline):
                break
            if t.initial and not (self.title(t)):
                k += 1
                continue
            if t.low in _PARTICLES_EN and not t.cap and (
                    k > j or (j > 0 and (self.first(run[j - 1], strong=False) or self.title(run[j - 1])))):
                m = k
                while m < n and run[m].low in _PARTICLES_EN and not run[m].cap and not run[m].comma:
                    m += 1
                if m < n and m > k and not run[m].comma and self.person_word(run[m]):
                    k = m
                    continue
                break
            if k > j and t.low in _GENERATIONAL and (t.cap or t.upper or t.dot):
                k += 1
                last_name = k
                break
            if self.person_word(t, allow_common=allow_common) and not self.title(t):
                words += 1
                k += 1
                last_name = k
                if t.poss:
                    break  # "Thurlowe's report": the possessive ends the name
                continue
            break
        if last_name < 0 or (need_surname and words == 0):
            return None
        return last_name

    def rule_titles(self, run: list[_Tok]) -> None:
        for i, t in enumerate(run):
            if not self.title(t) or i + 1 >= len(run):
                continue
            nxt = run[i + 1]
            if nxt.comma or nxt.newline:
                continue
            end = self.take_person(run, i + 1)
            if end is None and nxt.initial and i + 2 < len(run):
                end = self.take_person(run, i + 2)
            if end is None:
                continue
            self.add(run[i + 1], run[end - 1], "PERSON", 5, "title")

    def rule_first_last(self, run: list[_Tok]) -> None:
        n = len(run)
        sentence_start = _starts_sentence(self.text, run[0].start)
        for i, t in enumerate(run):
            if not self.first(t, strong=False):
                continue
            if i > 0 and (self.title(run[i - 1]) or run[i - 1].low in _PLACE_BEFORE):
                continue
            if i > 0 and not t.comma and run[i - 1].cap and self.person_word(run[i - 1]) \
                    and not self.first(run[i - 1], strong=False) and run[i - 1].low not in self.b.common_words \
                    and not (i == 1 and sentence_start):
                continue  # "Northvale Priya": a capitalised non-name word before it (a heading), left to NER
            if t.low in self.b.stopwords and (i == 0 and sentence_start or t.upper):
                continue  # "Will Smith arrive?": a modal at the sentence start
            strong = self.first(t)
            if t.poss:
                continue  # "Priya's": the possessive rule
            if i + 1 < n and run[i + 1].initial and (i + 2 >= n or not run[i + 2].cap or run[i + 2].comma):
                if strong:
                    self.add(t, run[i + 1], "PERSON", 3, "first_initial")
                continue
            end = self.take_person(run, i + 1)
            if end is None:
                if strong:
                    self.rule_first_alone(run, i)
                continue
            if run[i + 1].comma:
                continue
            sur = [run[k] for k in range(i + 1, end) if self.person_word(run[k])]
            if not sur:
                continue
            if t.upper != sur[-1].upper and t.upper and not sur[-1].upper:
                continue  # "PRIYA Thurlowe": an acronym or a heading word, not the English name order
            if not strong:
                weak_sur = all(s.low in self.b.common_words or s.low in self.b.places or s.low in self.b.vocab
                               for s in sur)
                if weak_sur and sentence_start and i == 0:
                    continue
            self.add(t, run[end - 1], "PERSON", 4, "first_name")

    def rule_first_alone(self, run: list[_Tok], i: int) -> None:
        """A strong first name on its own (chat, lists): a person, low priority. Not after a place word, a
        capitalised word or a possessive org cue, not when the next word is a capitalised vocabulary word that
        reads as a product line ("Priya Analytics" is handled by the company rules)."""
        t = run[i]
        if t.upper or t.poss and False:
            return
        n = len(run)
        if i + 1 < n and not run[i + 1].comma and run[i + 1].cap and not run[i + 1].initial \
                and (run[i + 1].low in self.b.org_suffixes or self.form(run[i + 1])):
            return
        if i + 1 < n and run[i + 1].cap and not run[i + 1].comma and not run[i + 1].newline \
                and not run[i + 1].digit and not self.first(run[i + 1], strong=False) and not run[i + 1].initial:
            return  # "Priya Update", "Grace Period": a capitalised word follows, the pair is not a name
        if _starts_sentence(self.text, t.start):
            # an imperative verb that is also a first name: "Ping me", "Mark the file", "Bill them", "Ping @tomas"
            # (integrator, v0.2: a sentence-initial first name followed by an object pronoun, a determiner or a
            # mention is a verb, not a person)
            nxt = run[i + 1] if i + 1 < n else None
            if nxt is not None and not nxt.cap and not nxt.comma and not nxt.newline and nxt.low in _IMPERATIVE_OBJECTS:
                return
            if re.match(r"[ \t]+<?@", self.text[t.raw_end:t.raw_end + 3]):
                return
        self.add(t, t, "PERSON", 2, "first_alone")

    def rule_surname_first(self, run: list[_Tok]) -> None:
        """"Thurlowe, Priya" (headers, lists, signatures)."""
        n = len(run)
        for i in range(n - 1):
            t, nxt = run[i], run[i + 1]
            if not nxt.comma or nxt.newline or not self.first(nxt):
                continue
            if not self.person_word(t, allow_place=False) or self.first(t) or t.low in self.b.vocab:
                continue
            if i > 0 and not run[i].comma and self.person_word(run[i - 1]) and not self.first(run[i - 1], strong=False):
                continue  # "New York, Priya": part of a longer capitalised phrase
            end = i + 2
            while end < n and not run[end].comma and (run[end].initial or (self.first(run[end], strong=False)
                                                                             and not self.person_word(run[end], allow_common=False))):
                end += 1
            if end < n and run[end].cap and not run[end].comma and not run[end].initial and self.person_word(run[end]):
                continue  # "Thurlowe, Priya Ravikar": not the "Last, First" order (a list of two people)
            self.add(t, run[end - 1], "PERSON", 4, "surname_first")

    def name_after(self, run: list[_Tok], j: int) -> Optional[int]:
        """A person name starting at ``run[j]``: title + name, first + surname, a first name alone (strong, or
        ambiguous when it is followed by punctuation or the end of the line). Returns the end index or None."""
        n = len(run)
        if j >= n:
            return None
        t = run[j]
        if self.title(t):
            end = self.take_person(run, j + 1)
            return end
        if not self.first(t, strong=False):
            return None
        end = self.take_person(run, j + 1)
        if end is not None and not run[j + 1].comma:
            sur = [run[k] for k in range(j + 1, end) if self.person_word(run[k])]
            if sur:
                return end
        if j + 1 < n and run[j + 1].initial:
            return j + 2
        if self.first(t) or j + 1 >= n or run[j + 1].comma or run[j + 1].newline or not run[j + 1].cap \
                or run[j + 1].amp:
            return j + 1
        return None

    def rule_greetings(self, run: list[_Tok], toks: list[_Tok], first_index: int) -> None:
        """"Hi Priya,", "Dear Mr. Thurlowe", "Thanks Marcus", "cc Lee", and the sign-off "Best,\\nPriya"."""
        n = len(run)
        for i, t in enumerate(run):
            if t.low not in self.b.greetings or t.digit:
                continue
            j = i + 1
            if j < n and run[j].low in ("you", "all", "both", "again", "there", "everyone", "team") and not run[j].cap:
                j += 1
            if j < n and run[j].low == "you" and run[j].cap:
                continue  # "Thank You" heading
            if j < n and not run[j].newline:
                self.greeting_names(run, j)
                continue
            # sign-off at the end of a line: the name is on the next line(s)
            if i == n - 1 or run[i + 1].newline:
                k = first_index + i + 1
                if k < len(toks) and toks[k].newline and not toks[k].column:
                    gap = self.text[t.raw_end:toks[k].start]
                    if _LINE_RE.fullmatch(gap):
                        self.signoff_name(toks, k)

    def greeting_names(self, run: list[_Tok], j: int) -> None:
        n = len(run)
        count = 0
        while j < n and count < 4:
            end = self.name_after(run, j)
            if end is None and run[j].cap and run[j].low in self.b.months_days \
                    and run[j].low in self.b.all_first_names \
                    and (j + 1 >= n or run[j + 1].comma or run[j + 1].newline
                         or not run[j + 1].cap and not run[j + 1].digit):
                end = j + 1  # "Hi April," (a month name that is a first name, after a greeting)
            if end is None:
                break
            s = j + 1 if self.title(run[j]) else j
            if s < end:
                self.add(run[s], run[end - 1], "PERSON", 3, "greeting")
            count += 1
            j = end
            if j < n and (run[j].amp or run[j].low == "and") and not run[j].cap:
                j += 1
                continue
            if j < n and run[j].comma and (self.first(run[j], strong=False) or self.title(run[j])):
                continue
            break

    def signoff_name(self, toks: list[_Tok], k: int) -> None:
        """Name on the line after a sign-off word: "Best,\\nPriya Thurlowe", "Thanks,\\nP. Ravikar"."""
        line: list[_Tok] = []
        j = k
        while j < len(toks) and (j == k or not toks[j].newline) and not toks[j].column and len(line) < 5:
            if j > k and (toks[j].comma or not _GAP_RE.fullmatch(self.text[toks[j - 1].raw_end:toks[j].start])):
                break
            line.append(toks[j])
            j += 1
        if not line:
            return
        t = line[0]
        if t.initial and len(line) > 1:
            end = self.take_person(line, 1, max_words=2)
            if end:
                self.add(line[0], line[end - 1], "PERSON", 3, "signoff")
            return
        end = self.name_after(line, 0)
        if end is None:
            return
        s = 1 if self.title(line[0]) else 0
        if s < end and (end == len(line) or not line[end].cap):
            self.add(line[s], line[end - 1], "PERSON", 3, "signoff")

    def rule_speaker(self, run: list[_Tok]) -> None:
        """"Priya: can you ...", "[10:32] Marcus Lee: ..." at the start of a line (chat transcripts)."""
        n = len(run)
        i = 0
        while i < n and (run[i].digit or run[i].low in ("am", "pm")):
            i += 1
        if i >= n:
            return
        t = run[i]
        if not self.first(t, strong=False):
            return
        ls = self.text.rfind("\n", 0, t.start) + 1
        if not _SPEAKER_PREFIX_RE.fullmatch(self.text[ls:t.start]):
            return
        end = self.take_person(run, i + 1)
        if end is not None and not run[i + 1].comma:
            if run[end - 1].colon_after and all(not run[k].comma for k in range(i + 1, end)):
                self.add(t, run[end - 1], "PERSON", 3, "speaker")
            return
        if t.colon_after and self.first(t):
            self.add(t, t, "PERSON", 3, "speaker")

    def rule_roles(self, run: list[_Tok]) -> None:
        """"our manager Jane Doe", "colleague Priya", "attorney Mark Ellis": the role stays in clear."""
        n = len(run)
        for i, t in enumerate(run):
            if t.low not in self.b.roles or t.upper or i + 1 >= n or run[i + 1].comma or run[i + 1].newline:
                continue
            if self.title(t) and t.cap:
                continue  # handled by the title rule ("Judge Thurlowe")
            end = self.name_after(run, i + 1)
            if end is None:
                continue
            s = i + 2 if self.title(run[i + 1]) else i + 1
            if s >= end:
                continue
            if end < n and run[end].cap and not run[end].comma and self.person_word(run[end]):
                continue
            self.add(run[s], run[end - 1], "PERSON", 3, "role")

    def rule_possessive_first(self, run: list[_Tok]) -> None:
        """"Priya's laptop": a strong first name with a possessive is a person."""
        for i, t in enumerate(run):
            if t.poss and self.first(t) and (i == 0 or not self.person_word(run[i - 1]) or run[i - 1].comma):
                self.add(t, t, "PERSON", 3, "possessive_first")

    # --- company spans -----------------------------------------------------------------------------------

    def org_back(self, run: list[_Tok], i: int, *, max_words: int = 6) -> Optional[int]:
        """Start index of a company name whose last word is ``run[i - 1]``."""
        k = i - 1
        words = 0
        while k >= 0 and words < max_words:
            t = run[k]
            if k < i - 1 and (t.column or run[k + 1].column or run[k + 1].newline):
                break
            if k < i - 1 and run[k + 1].comma:
                break
            if t.low in _NAME_CONNECT and not t.cap:
                if t.low == "for" and k == 1 and self.b.lang == "en" and run[0].cap and not run[0].upper \
                        and len(run[0].low) > 4 and run[0].low.endswith("ed"):
                    break  # v0.4 (T5, FM4): "Drafted for Quorvel Inc.": a line-opening participle sits outside the name
                if k - 1 >= 0 and self.org_word(run[k - 1]) and k < i - 1:
                    k -= 1
                    continue
                break
            if t.low in _CIVILITY or not self.org_word(t) or t.low in self.b.roles and not t.upper:
                break
            words += 1
            k -= 1
        s = k + 1
        while s < i and (run[s].low in _NAME_CONNECT and not run[s].cap):
            s += 1
        return s if s < i else None

    def org_meaningful(self, run: list[_Tok], s: int, e: int, *, strong: bool) -> bool:
        toks = [run[k] for k in range(s, e) if not run[k].amp and run[k].low not in _NAME_CONNECT]
        if not toks:
            return False
        if any(self.distinctive(t) for t in toks):
            return True
        heads = [t for t in toks if t.low in self.b.common_words or t.low in self.b.places
                 or t.low in self.b.all_first_names or t.upper]
        if heads and (len(toks) >= 2 or strong):
            return True
        return strong and len(toks) >= 2 and not all(t.low in self.b.legal_terms or t.low in self.b.stopwords
                                                     for t in toks)

    def connector_form(self, run: list[_Tok], i: int) -> bool:
        """v0.4 (T3): ``run[i]`` is the "&" or "and" of a firm name written with "Company" or "Co." after it
        ("Halberd & Co.", "Velltry and Company"): the connector and the word belong to the name."""
        if i == 0 or i + 1 >= len(run):
            return False
        t, nxt = run[i], run[i + 1]
        if not (t.amp or (t.low == "and" and not t.cap)) or t.comma or t.newline or t.column:
            return False
        if nxt.comma or nxt.newline or nxt.column or not nxt.cap or nxt.digit or nxt.poss:
            return False
        if i + 2 < len(run) and not run[i + 2].comma and not run[i + 2].newline \
                and (run[i + 2].low in self.b.roles or run[i + 2].low in _COMPANY_OFFICES):
            return False  # "and Company Secretary": an office, not a firm name
        return nxt.low in _FIRM_WORDS and (nxt.low != "co" or nxt.dot or nxt.upper or i + 2 >= len(run)
                                           or not run[i + 2].cap or run[i + 2].comma)

    def firm_back(self, run: list[_Tok], i: int, *, max_words: int = 4) -> Optional[int]:
        """v0.4 (T3): start index of the name before the "&" or "and" of a firm name (``run[i]``): capitalised
        company words, "&" between two of them; never a lower-case word ("Jane Doe for Halberd & Co." is two
        names), a comma, a line break or a column gap."""
        k = i - 1
        words = 0
        while k >= 0 and words < max_words:
            t = run[k]
            if k < i - 1 and (run[k + 1].comma or run[k + 1].newline or run[k + 1].column):
                break
            if t.amp:
                if k - 1 >= 0 and run[k - 1].cap and self.org_word(run[k - 1]) and k < i - 1:
                    k -= 1
                    continue
                break
            if not t.cap or not self.org_word(t) or t.low in _CIVILITY or (t.low in self.b.roles and not t.upper):
                break
            words += 1
            k -= 1
        s = k + 1
        while s < i and run[s].amp:
            s += 1
        return s if s < i else None

    def rule_legal_forms(self, run: list[_Tok]) -> None:
        n = len(run)
        i = 0
        while i < n:
            t = run[i]
            if self.connector_form(run, i):
                s = self.firm_back(run, i)
                if s is not None and self.org_meaningful(run, s, i, strong=True):
                    self.add(run[s], run[i + 1], "ORG", 6, "legal_form")
                    i += 2
                    continue
            strength = self.form(t)
            width = 1
            if strength and i > 0 and self.form_misread(run, i):
                strength = None  # v0.4 (T5, FM4): a code, a state code or a list of codes, not a legal form
            if not strength:
                width = self.spelled_form_at(run, i)
                strength = "strong" if width else None
            if not strength or i == 0:
                i += 1
                continue
            s = self.org_back(run, i)
            if s is not None and self.org_meaningful(run, s, i, strong=strength == "strong"):
                words = sum(1 for k in range(s, i) if not run[k].amp and run[k].low not in _NAME_CONNECT)
                if strength == "strong" or words >= 2:
                    self.add(run[s], run[i - 1], "ORG", 6, "legal_form")
            i += width

    def rule_suffixes(self, run: list[_Tok]) -> None:
        n = len(run)
        for i, t in enumerate(run):
            if i == 0 or t.low not in self.b.org_suffixes or not t.cap or t.amp or t.low in ("and",):
                continue
            if t.comma or t.newline or t.column:
                continue
            k = i - 1
            words = 0
            while k >= 0 and words < 4:
                w = run[k]
                if run[k + 1].comma or run[k + 1].column or run[k + 1].newline:
                    break
                if w.amp or (w.low == "and" and not w.cap):
                    if k - 1 >= 0 and self.org_word(run[k - 1]) and run[k - 1].cap and not run[k].comma:
                        k -= 1
                        continue
                    break
                if not w.cap or not self.org_word(w) or w.low in _CIVILITY \
                        or (w.low in self.b.stopwords and w.low not in self.b.org_generic):
                    break
                if w.low in self.b.roles and not w.upper:
                    break
                words += 1
                k -= 1
            s = k + 1
            while s < i and not run[s].cap:
                s += 1
            if s >= i:
                continue
            heads = [run[j] for j in range(s, i) if run[j].cap]
            if not any(self.distinctive(h) or h.low in self.b.common_words or h.low in self.b.places
                       or h.low in self.b.all_first_names or h.upper for h in heads):
                continue
            if all(h.low in self.b.vocab and h.low not in self.b.common_words and h.low not in self.b.places
                   and h.low not in self.b.all_first_names and not h.upper for h in heads):
                continue
            self.add(run[s], t, "ORG", 4, "suffix")

    def rule_partnerships(self, run: list[_Tok]) -> None:
        """"Baker & McKenzie", "Smith, Jones & Rowe": surname-like words joined by "&"."""
        n = len(run)
        for i, t in enumerate(run):
            if not t.amp or i == 0 or i + 1 >= n:
                continue
            left, right = run[i - 1], run[i + 1]
            if right.comma or right.newline or t.comma:
                continue
            if not (self.org_word(left) and left.cap and self.org_word(right) and right.cap):
                continue
            if self.first(left) and self.first(right) and left.low != right.low:
                continue
            if left.low in self.b.stopwords or right.low in self.b.stopwords:
                continue
            sides = [left, right]
            if not any(self.distinctive(x) or x.low in self.b.common_words
                       or (x.low in self.b.all_first_names and x.low not in self.b.vocab) for x in sides):
                continue
            if any(x.low in self.b.vocab and x.low not in self.b.common_words for x in sides):
                continue
            s = i - 1
            while s - 1 >= 0 and run[s].comma and self.org_word(run[s - 1]) and run[s - 1].cap \
                    and not self.first(run[s - 1]) and not run[s - 1].low in self.b.vocab:
                s -= 1
            e = i + 1
            while e + 1 < n and not run[e + 1].comma and run[e + 1].cap and run[e + 1].low in self.b.org_suffixes:
                e += 1
            self.add(run[s], run[e], "ORG", 6, "partnership")

    def rule_triggers(self, run: list[_Tok]) -> None:
        """"works at Northvale", "joined Brightwater Capital", "client Lakeshore Medical"."""
        n = len(run)
        for i, t in enumerate(run):
            if t.low not in self.b.org_triggers or t.cap or i + 1 >= n:
                continue
            if run[i + 1].comma or run[i + 1].newline:
                continue
            j = i + 1
            words: list[_Tok] = []
            while j < n and len(words) < 4:
                w = run[j]
                if j > i + 1 and (w.comma or w.newline or w.column):
                    break
                if w.low in _NAME_CONNECT and not w.cap and words and j + 1 < n and run[j + 1].cap:
                    j += 1
                    continue
                if not w.cap or not self.org_word(w) or w.low in _CIVILITY:
                    break
                words.append(w)
                j += 1
            if not words:
                continue
            if j < n and run[j].cap and not run[j].comma and not run[j].newline and self.org_word(run[j]):
                continue  # a longer capitalised phrase: not consumed, left to the other rules
            if self.form(run[j]) if j < n else False:
                continue  # the legal form rule takes it
            head = words[0]
            if self.first(head, strong=False) or head.low in self.b.places and len(words) == 1:
                continue
            if len(words) == 1:
                if not self.distinctive(head) or sum(c.isalpha() for c in head.text) < 4 or head.poss:
                    continue
                if t.low in _TRIGGER_AT_LABELS and j < n and not (run[j].low in self.b.stopwords or run[j].comma
                                                                   or run[j].sent_end) and not run[j].digit:
                    pass
            else:
                if not any(self.distinctive(w) for w in words) and not (
                        any(w.low in self.b.places or w.low in self.b.common_words for w in words)
                        and any(w.low in self.b.org_generic or w.low in self.b.org_suffixes for w in words)):
                    continue
            self.add(words[0], words[-1], "ORG", 5, "trigger")

    def rule_org_of(self, run: list[_Tok]) -> None:
        """"Bank of America", "University of Northvale", "Institute of Applied Optics": a head noun, "of", then
        capitalised words that are not all vocabulary."""
        n = len(run)
        for i, t in enumerate(run):
            if t.low not in _ORG_OF_HEADS or not t.cap or i + 2 >= n or run[i + 1].low != "of" or run[i + 1].cap:
                continue
            j = i + 2
            if run[j].low == "the" and not run[j].cap and j + 1 < n:
                j += 1
            words: list[_Tok] = []
            k = j
            while k < n and len(words) < 4:
                w = run[k]
                if k > j and (w.comma or w.newline or w.column):
                    break
                if w.low in _NAME_CONNECT and not w.cap and words and k + 1 < n and run[k + 1].cap:
                    k += 1
                    continue
                if not w.cap or not self.org_word(w) or w.low in _CIVILITY:
                    break
                words.append(w)
                k += 1
            if not words or all(w.low in self.b.vocab and w.low not in self.b.places
                                and w.low not in self.b.common_words for w in words):
                continue
            s = i
            while s - 1 >= 0 and run[s - 1].cap and not run[s].comma and self.org_word(run[s - 1]) \
                    and run[s - 1].low not in _CIVILITY:
                s -= 1
            self.add(run[s], words[-1], "ORG", 4, "org_of")

    def rule_possessive_orgs(self, run: list[_Tok]) -> None:
        """"Northvale's Q3 results", "Brightwater's CEO": a distinctive word with a possessive before a business
        noun is a company."""
        n = len(run)
        for i, t in enumerate(run):
            if not t.poss or i + 1 >= n or run[i + 1].comma:
                continue
            if run[i + 1].low not in _BUSINESS_NOUNS:
                continue
            if not self.distinctive(t) or self.first(t, strong=False) or t.upper:
                continue
            if i > 0 and run[i - 1].cap and self.org_word(run[i - 1]) and not run[i].comma:
                continue  # part of a longer name ("Brightwater Capital's CEO": the suffix rule has it)
            self.add(t, t, "ORG", 4, "possessive")

    # --- driver ------------------------------------------------------------------------------------------

    def run_all(self) -> list[tuple[int, int, str, int, str]]:
        toks = _tokenize(self.text)
        runs = _runs(self.text, toks)
        first_index = 0
        for run in runs:
            if any(t.cap for t in run):
                self.rule_legal_forms(run)
                self.rule_suffixes(run)
                self.rule_partnerships(run)
                self.rule_triggers(run)
                self.rule_org_of(run)
                self.rule_possessive_orgs(run)
                self.rule_titles(run)
                self.rule_first_last(run)
                self.rule_surname_first(run)
                self.rule_greetings(run, toks, first_index)
                self.rule_speaker(run)
                self.rule_roles(run)
                self.rule_possessive_first(run)
            first_index += len(run)
        return self.out


def _dedupe(spans: list[tuple[int, int, str, int, str]]) -> list[tuple[int, int, str, int, str]]:
    """Drop a span contained in a longer span of the same kind, and a duplicate of an equal span (the higher
    priority stays). Linear sweep: sorted by start, a span is contained in an earlier kept span of its kind
    exactly when that kind's furthest end reaches its own end."""
    spans = sorted(set(spans), key=lambda x: (x[0], -(x[1] - x[0]), -x[3]))
    out: list[tuple[int, int, str, int, str]] = []
    far: dict[str, int] = {}
    for sp in spans:
        if far.get(sp[2], -1) >= sp[1]:
            continue
        out.append(sp)
        far[sp[2]] = sp[1]
    return out


def detect_names_en(text: str, b: Optional[Bundle] = None) -> list:
    """Person and company candidates in English text (unresolved, may overlap), as ``names.RuleSpan``
    objects (``start, end, kind, priority, rule``). Rules only, no model, no network.

    Priorities follow the French rules so that ``Masker._resolve`` ranks them the same way: ``legal_form`` 6,
    ``partnership`` 6, ``title`` 5, ``trigger`` 5, ``first_name`` 4, ``surname_first`` 4, ``suffix`` 4,
    ``possessive`` 4, ``greeting`` / ``signoff`` / ``speaker`` / ``role`` / ``first_initial`` /
    ``possessive_first`` 3, ``first_alone`` 2.
    """
    from .names import RuleSpan  # local import: names.py may import this module one day

    b = b or bundle("en")
    raw = _Rules(text, b).run_all()
    raw = [x for sp in raw for x in (_quote_trim(text, sp, b) if sp[2] == "PERSON" else [sp])]
    return [RuleSpan(s, e, kind, prio, rule) for s, e, kind, prio, rule in _dedupe(raw)]


_QUOTE_CLOSE = {'"': '"', "“": "”", "'": "'’", "‘": "’'", "(": ")", "[": "]"}
_QUOTE_OPEN = {'"': '"', "”": "“", "'": "'‘", "’": "‘'", ")": "(", "]": "["}


def _quote_trim(text: str, sp: tuple[int, int, str, int, str], b: Bundle) -> list[tuple[int, int, str, int, str]]:
    """v0.4 (T2): a person span never cuts through a pair of quotes or brackets. A table cell or a line break can
    end a name inside a quoted nickname ('Ottilie "Otti' before '" | Director', 'Otti" Wrexley' after a line
    break, 'Cosima "Neil' before ' W."'): the span then stops before the opening mark, or starts after the closing
    mark, so that no stored name holds half a nickname. The pair is looked for on the same line, at most 40
    characters away. The quoted word cut off becomes a span of its own when it is one capitalised first name of
    the lexicon (a nickname). Returns the spans left (none, one or two)."""
    s, e = s0, e0 = sp[0], sp[1]
    ls = text.rfind("\n", 0, s) + 1
    le = text.find("\n", e)
    le = len(text) if le < 0 else le

    def word_end(j: int) -> bool:
        return (text[j - 1].isalnum() or text[j - 1] == ".") and (j + 1 >= len(text) or not text[j + 1].isalnum())

    for i in range(s, e - 1):
        ch = text[i]
        if ch in _QUOTE_CLOSE and (i == s or not text[i - 1].isalnum()) and text[i + 1].isalpha():
            closers = _QUOTE_CLOSE[ch]
            if any(text[j] in closers and word_end(j) for j in range(i + 1, e)):
                break  # the pair is inside the span
            if any(text[j] in closers and word_end(j) for j in range(e, min(le, e + 40))):
                e = i
            break
    for j in range(e - 1, s, -1):
        ch = text[j]
        if ch in _QUOTE_OPEN and word_end(j):
            openers = _QUOTE_OPEN[ch]
            if any(text[i] in openers and text[i + 1].isalpha() and (i == 0 or not text[i - 1].isalnum())
                   for i in range(s, j)):
                break
            if any(text[i] in openers and text[i + 1].isalpha() and (i == 0 or not text[i - 1].isalnum())
                   for i in range(max(ls, s - 40), s)):
                s = j + 1
            break
    if (s, e) == (s0, e0):
        return [sp]
    out = []
    for a, z in ((s0, s), (e, e0)):  # the part cut off (only one of the two is not empty)
        word = text[a:z].strip(" \t\r\n\u00a0" + "".join(_QUOTE_CLOSE) + "".join(_QUOTE_OPEN))
        if word and _TOK_RE.fullmatch(word) and word[:1].isupper() and is_first_name(word, b, strong=False):
            k = text.index(word, a)
            out.append((k, k + len(word)) + tuple(sp[2:]))
    while s < e and not text[s].isalnum():
        s += 1
    while e > s and not text[e - 1].isalnum() and text[e - 1] != ".":
        e -= 1
    left = [m.group() for m in _TOK_RE.finditer(text, s, e)]
    if len(left) >= 2 or (len(left) == 1 and left[0][:1].isupper() and is_first_name(left[0], b, strong=False)):
        out.append((s, e) + tuple(sp[2:]))  # one word left is kept only when it is a first name by itself
    return out


# ---------------------------------------------------------------------------
# 5. English protected intervals (never masked)
# ---------------------------------------------------------------------------

_CAPW = r"[A-Z][\w'’\-]*"
# Courts: "Supreme Court", "United States District Court for the Northern District of California", "Court of
# Appeals for the Ninth Circuit", "Employment Tribunal", "Crown Court at Southwark".
_COURT_RE = re.compile(
    r"(?<![\w])(?:(?:" + _CAPW + r" ){0,4}(?:Court|Courts|Tribunal)"
    r"(?: of (?:Appeals?|Session|Justice|Claims|Chancery|Common Pleas|International Trade|Protection))?"
    r"(?: (?:of|for|at|in) (?:the )?(?:" + _CAPW + r"(?: |$)){1,6}?(?:Circuit|District|County|State|Commonwealth|Territory|Division|Borough|" + _CAPW + r"))?"
    r"(?: of (?:the )?" + _CAPW + r"(?: " + _CAPW + r"){0,2})?)(?![\w])"
)
# Public bodies: "Department of Labor", "Office of the Attorney General", "Ministry of Justice", "City of Boston".
_BODY_RE = re.compile(
    r"(?<![\w])(?:Department|Departments|Ministry|Office|Bureau|Board|Commission|Committee|Council|Secretary|"
    r"Secretariat|Agency|Authority|Registry|Register|Institute|Administration|House|Senate|Assembly|Parliament|"
    r"Chamber|City|State|County|Town|Commonwealth|Province|Borough|Township|Village|Republic|Kingdom|Government|"
    r"Embassy|Consulate|Bank) (?:of|for|on) (?:the )?(?:" + _CAPW + r"(?: (?:and|of|the|for|&) )?){1,5}(?![\w])"
)
# "Cook County", "Westminster Borough", "Los Angeles County Sheriff"
_COUNTY_RE = re.compile(r"(?<![\w])" + _CAPW + r"(?: " + _CAPW + r")? (?:County|Parish|Borough|Council)(?![\w])")
# Statutes and codes: "Clean Air Act of 1970", "Employment Rights Act 1996", "California Civil Code".
_ACT_RE = re.compile(
    r"(?<![\w])(?:" + _CAPW + r" ){1,6}(?:Act|Acts|Regulations?|Code|Rules|Directive|Ordinance|Statute|Amendment)"
    r"(?:,? (?:of )?(?:19|20)\d{2})?(?![\w])"
)
# Section references: "Section 230", "§ 1983", "Rule 12(b)(6)", "Title VII", "Article 6(1)(f)".
_SECTION_REF_RE = re.compile(
    r"(?<![\w])(?:Section|Sections|Sec\.|§§?|Article|Articles|Art\.|Rule|Rules|Title|Chapter|Part|Paragraph|"
    r"Schedule|Clause|Regulation)[  ]*(?:\d+[\w.()\-]*|[IVXLC]+\b)(?![\w])"
)


@lru_cache(maxsize=None)
def whitelist_matcher_en() -> PhraseMatcher:
    pm = PhraseMatcher(gap=r"[ .'’\-&,]{0,3}")
    for phrase in bundle("en").whitelist_phrases:
        toks = phrase_tokens(phrase)
        if toks:
            pm.add(toks, phrase)
    return pm


def protected_intervals_en(text: str, folded: Folded, ftoks: list[tuple[str, int, int]], *,
                           release: Collection[str] = ()) -> IntervalSet:
    """Original-text intervals that the English rules and the NER must never mask: whitelist phrases
    (courts, agencies, statutes, vendors), court and public-body patterns, named acts and codes, section
    references. ``release`` (v0.4, P1): folded whitelist phrases that this text's dossier may mask (the vendor that
    is its own company); their mentions are not protected. Default none: the protection is complete."""
    out = IntervalSet()
    spans: list[tuple[int, int]] = []
    released = frozenset(release) if release else frozenset()
    for i, j, phrase in whitelist_matcher_en().find_all(folded.text, ftoks):
        if released and phrase in released:
            continue
        spans.append(folded.span(ftoks[i][1], ftoks[j - 1][2]))
    for rx in (_COURT_RE, _BODY_RE, _COUNTY_RE, _ACT_RE, _SECTION_REF_RE):
        for m in rx.finditer(text):
            spans.append((m.start(), m.end()))
    spans.sort()
    merged: list[list[int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    for s, e in merged:
        out.add(s, e)
    return out


# ---------------------------------------------------------------------------
# 6. Per-language regexes for the masker (title and salutation before a known name)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def title_before_re(lang: str = "fr") -> "re.Pattern[str]":
    """Regex matched against the text right before a name: a title ends there ("Mr. ", "Dr ", "Judge ")."""
    if not str(lang).lower().startswith("en"):
        from .masker import _TITLE_BEFORE_RE  # the v0.1 French regex stays the French one
        return _TITLE_BEFORE_RE
    b = bundle("en")
    short = sorted((t for t in b.titles if len(t) <= 5), key=len, reverse=True)
    long_ = sorted((t for t in b.titles if len(t) > 5), key=len, reverse=True)
    return re.compile(
        r"(?:(?<![\w.])(?:" + "|".join(re.escape(t).capitalize() for t in short) + r"|"
        + "|".join(re.escape(t).upper() for t in short) + r")\.?"
        r"|(?i:(?<![\w.])(?:" + "|".join(re.escape(t) for t in long_) + r")))[ \t  ]+$"
    )


@lru_cache(maxsize=None)
def salute_before_re(lang: str = "fr") -> "re.Pattern[str]":
    """Regex matched against the text right before a name: a greeting on the same line ("Hi ", "Dear ") or a
    sign-off on the previous line ("Best,\\n")."""
    if not str(lang).lower().startswith("en"):
        from .masker import _SALUTE_BEFORE_RE
        return _SALUTE_BEFORE_RE
    return re.compile(
        r"(?i)(?:\b(?:hi|hello|hey|dear|hiya|howdy|greetings|thanks|thank\s+you|cheers|cc|good\s+(?:morning|afternoon|evening))"
        r"\s*,?\s*(?:all|everyone|both|again|there)?\s*,?\s*$"
        r"|(?:best|regards|kind\s+regards|best\s+regards|warm\s+regards|many\s+thanks|thanks(?:\s+again)?|thank\s+you|"
        r"sincerely|yours\s+(?:sincerely|truly|faithfully)|cheers|cordially|respectfully|warmly|talk\s+soon|take\s+care|"
        r"all\s+the\s+best|best\s+wishes)\s*[,.!]?\s*\n\s*$)"
    )
