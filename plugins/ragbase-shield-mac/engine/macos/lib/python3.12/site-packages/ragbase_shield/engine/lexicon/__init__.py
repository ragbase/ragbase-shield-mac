"""Lexicon loader. Data files are plain UTF-8 text, one entry per line, ``#`` starts a comment.

Files:

- ``first_names_fr.txt``: French first names (INSEE Fichier des prénoms, Licence Ouverte 2.0).
- ``whitelist_fr.txt``: courts, institutions, laws, legal terms, months, days (never masked).
- ``stopwords_fr.txt``: function words and sentence-initial words (never a name word).
- ``legal_terms_fr.txt``: roles, document words, legal nouns (never a person's name word).
- ``common_words_fr.txt``: common words, many of them also surnames (ambiguity checks).
- ``contract_roles_fr.txt``: contractual party roles, loaded with the legal terms (R1).
- ``business_words_fr.txt``: generic business adjectives and nouns that start company names (R1).
- ``trade_nouns_fr.txt``: trade head nouns ("Minoterie X") that complete ``names.ORG_TRADES`` (R1).
- ``acronyms_fr.txt``: common business and legal acronyms, never read as a company acronym or initials (R1).
- ``instruments_fr.txt``: financial instrument acronyms, never a first name when written in capitals (R1).
- ``nicknames_fr.txt``: standard diminutives that are not a prefix of the full first name (R1).
- ``verbs_fr.txt``: French verbs, for the lower-case name rules only (v0.4.1): infinitives, models of the third
  group, forms of the irregular verbs. :func:`verb_form` reads a conjugated form from them.
- ``lowercase_fr.txt``: ordinary words for the lower-case name rules only (v0.4.1), in sections.

Every entry is folded with :func:`ragbase_shield.engine.normalize.fold_word` at load time.
"""
from __future__ import annotations

import re
from functools import lru_cache
from importlib import resources

from ..normalize import fold_word

__all__ = [
    "read_lines",
    "read_sections",
    "first_names",
    "whitelist_phrases",
    "stopwords",
    "legal_terms",
    "common_words",
    "months_days",
    "business_words",
    "contract_roles",
    "trade_nouns",
    "acronym_stoplist",
    "instrument_terms",
    "nicknames",
    "verbs",
    "verb_form",
    "lowercase_words",
]

# First names that are also place names or words used capitalised in legal text: they are not
# used as a first-name signal by the rules (they still work inside a registered full name).
_FIRST_NAME_EXCLUDE = frozenset({
    "paris", "france", "europe", "lyon", "nice", "lourdes", "valence", "vienne", "orleans",
    "cannes", "nancy-claude", "loire", "seine", "rhone", "jura", "lorraine", "bretagne",
    "provence", "savoie", "avril", "mai", "juin", "juillet", "aout", "mars", "janvier",
    "fevrier", "septembre", "octobre", "novembre", "decembre", "lundi", "mardi", "mercredi",
    "jeudi", "vendredi", "samedi", "dimanche", "code", "loi", "article", "justice", "cour",
    "conseil", "tribunal", "president", "maitre", "madame", "monsieur", "societe", "sas",
    "sarl", "noel", "toussaint", "paques", "nord", "sud", "est", "ouest", "sainte", "saint",
    "dieu", "ange", "amour", "liberte", "victoire", "fidele", "prudence", "patience",
    "constance", "clemence", "esperance", "providence", "modeste", "juste", "aime", "desire",
    "parfait", "heureux", "merci", "bienvenue", "salut", "joie", "bonheur", "fortune",
    "chance", "destin", "avenir", "horizon", "confiance", "verite", "raison", "sagesse",
})

_MONTHS_DAYS = frozenset({
    "janvier", "fevrier", "mars", "avril", "mai", "juin", "juillet", "aout", "septembre",
    "octobre", "novembre", "decembre", "janv", "fevr", "sept", "oct", "nov", "dec",
    "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche",
})


def read_lines(name: str) -> list[str]:
    """Raw entries of a lexicon file (comments and blank lines removed, not folded)."""
    text = resources.files(__name__).joinpath(name).read_text(encoding="utf-8")
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def _folded_set(name: str) -> frozenset[str]:
    return frozenset(f for f in (fold_word(x) for x in read_lines(name)) if f)


@lru_cache(maxsize=None)
def first_names() -> frozenset[str]:
    """French first names usable as a name signal (folded)."""
    names = _folded_set("first_names_fr.txt")
    return frozenset(n for n in names if n not in _FIRST_NAME_EXCLUDE)


@lru_cache(maxsize=None)
def all_first_names() -> frozenset[str]:
    """Every first name in the lexicon, including the excluded ambiguous ones (folded)."""
    return _folded_set("first_names_fr.txt")


@lru_cache(maxsize=None)
def whitelist_phrases() -> tuple[str, ...]:
    """Whitelist phrases (folded), multi-word and single-word."""
    return tuple(sorted(_folded_set("whitelist_fr.txt")))


@lru_cache(maxsize=None)
def stopwords() -> frozenset[str]:
    return _folded_set("stopwords_fr.txt")


@lru_cache(maxsize=None)
def legal_terms() -> frozenset[str]:
    return _folded_set("legal_terms_fr.txt") | contract_roles()


@lru_cache(maxsize=None)
def contract_roles() -> frozenset[str]:
    """Defined party roles of contracts ("concessionnaire", "franchiseur"), folded; part of the legal terms."""
    return _folded_set("contract_roles_fr.txt")


@lru_cache(maxsize=None)
def common_words() -> frozenset[str]:
    return _folded_set("common_words_fr.txt")


@lru_cache(maxsize=None)
def months_days() -> frozenset[str]:
    return _MONTHS_DAYS


@lru_cache(maxsize=None)
def business_words() -> frozenset[str]:
    """Generic business adjectives and nouns ("financière", "industrielle"), folded."""
    return _folded_set("business_words_fr.txt")


@lru_cache(maxsize=None)
def trade_nouns() -> frozenset[str]:
    """Trade head nouns ("minoterie", "scierie"), folded."""
    return _folded_set("trade_nouns_fr.txt")


@lru_cache(maxsize=None)
def acronym_stoplist() -> frozenset[str]:
    """Common acronyms (folded, lower case): never a company acronym or a person's initials."""
    return _folded_set("acronyms_fr.txt")


@lru_cache(maxsize=None)
def instrument_terms() -> frozenset[str]:
    """Financial instrument acronyms (folded, lower case)."""
    return _folded_set("instruments_fr.txt")


@lru_cache(maxsize=None)
def nicknames() -> dict[str, frozenset[str]]:
    """Full first name (folded) -> its standard diminutives (folded) that are not prefixes."""
    out: dict[str, set[str]] = {}
    for line in read_lines("nicknames_fr.txt"):
        short, sep, full = line.partition("=")
        short, full = fold_word(short), fold_word(full)
        if sep and short and full:
            out.setdefault(full, set()).add(short)
    return {k: frozenset(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# v0.4.1: lists of the lower-case name rules (French). The other rules do not read them.
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(r"^#\s*\[([\w\-]+)\]\s*$")


def read_sections(name: str) -> dict[str, list[str]]:
    """Raw lines of a lexicon file grouped by its ``# [section]`` lines (lines before the first one go to
    ``"default"``). Comments and blank lines are removed; a line may hold several entries; nothing is folded."""
    text = resources.files(__name__).joinpath(name).read_text(encoding="utf-8")
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
                out.setdefault(current, [])
            continue
        out.setdefault(current, []).append(line)
    return out


def _folded_words(lines: list[str]) -> frozenset[str]:
    """Every space-separated entry of ``lines``, folded."""
    return frozenset(f for f in (fold_word(w) for line in lines for w in line.split()) if f)


# Endings of the first group on a folded stem ("contest" of "contester"): present, imperfect, future, conditional,
# present subjunctive, imperative, participles ("é" folds to "e", "ée" to "ee"). No simple past.
_ER_ENDINGS = tuple(sorted(
    "e es ons ez ent ais ait ions iez aient erai eras era erons erez eront erais erait erions eriez eraient "
    "ant ante ants antes ee ees".split(), key=len, reverse=True))
# Endings with a mute "e": before them "y" is written "i" ("emploie") and "l" or "t" may double ("appelle").
_ER_MUTE = frozenset("e es ent erai eras era erons erez eront erais erait erions eriez eraient".split())
# Endings of the second group on a folded stem ("fin" of "finir").
_IR_ENDINGS = tuple(sorted(
    "is it issons issez issent issais issait issions issiez issaient irai iras ira irons irez iront irais irait "
    "irions iriez iraient isse isses issant issante issants issantes i ie ies".split(), key=len, reverse=True))


@lru_cache(maxsize=None)
def verbs() -> dict[str, frozenset[str]]:
    """The French verbs of ``verbs_fr.txt``, folded: ``"er"`` and ``"ir"`` (infinitives of the first and second
    groups), ``"forms"`` (every form of the third-group verbs, from their models, and the listed forms),
    ``"infinitives"`` (every infinitive) and ``"auxiliaries"`` (every form of the auxiliary and modal verbs)."""
    sections = read_sections("verbs_fr.txt")
    er = _folded_words(sections.get("er", []))
    ir = _folded_words(sections.get("ir", []))
    forms: set[str] = set(_folded_words(sections.get("forms", [])))
    infinitives: set[str] = set(er) | set(ir)
    for line in sections.get("models", []):
        name, sep, rest = line.partition(":")
        cut, bar, endings = rest.partition("|")
        if not sep or not bar:
            continue
        cut = fold_word(cut.strip())
        ends = ["" if e == "0" else fold_word(e) for e in endings.split()]
        for inf in _folded_words(sections.get(name.strip(), [])):
            infinitives.add(inf)
            if cut and inf.endswith(cut):
                stem = inf[:len(inf) - len(cut)]
                forms.update(stem + e for e in ends)
    forms.discard("")
    return {"er": er, "ir": ir, "forms": frozenset(forms - infinitives), "infinitives": frozenset(infinitives),
            "auxiliaries": _folded_words(sections.get("auxiliaries", []))}


@lru_cache(maxsize=65536)
def verb_form(low: str) -> str:
    """What the folded word ``low`` is as a French verb: ``"aux"`` (a form of an auxiliary or modal verb), ``"form"``
    (a conjugated form or a participle), ``"inf"`` (an infinitive) or ``""``. A word that is both a form and an
    infinitive is a form. The first and second groups are read from the ending ("conteste", "contestait",
    "mangeons", "emploie", "appelle", "finissent"), the third group from the models of the file."""
    v = verbs()
    if low in v["auxiliaries"]:
        return "aux"
    if low in v["forms"]:
        return "form"
    er, ir = v["er"], v["ir"]
    for end in _ER_ENDINGS:
        if not low.endswith(end):
            continue
        stem = low[:len(low) - len(end)]
        if len(stem) < 2:
            continue
        if stem + "er" in er:
            return "form"
        if end[0] in "ao" and stem.endswith("ge") and stem[:-1] + "er" in er:
            return "form"  # "mangeons", "exigeait"
        if end in _ER_MUTE:
            if stem.endswith("i") and stem[:-1] + "yer" in er:
                return "form"  # "emploie", "paiera"
            if stem[-3:] in ("ell", "ett") and stem[:-1] + "er" in er:
                return "form"  # "appelle", "rejette"
    for end in _IR_ENDINGS:
        if low.endswith(end) and len(low) - len(end) >= 2 and low[:len(low) - len(end)] + "ir" in ir:
            return "form"
    return "inf" if low in v["infinitives"] else ""


@lru_cache(maxsize=None)
def lowercase_words() -> dict[str, frozenset[str]]:
    """The sections of ``lowercase_fr.txt``, folded: ``"function"`` (function words), ``"places"`` (towns,
    countries, regions), ``"jobs"`` (nouns of a trade or a job), ``"persons"`` (nouns that name a person), ``"states"`` (adjectives of a person's state)
    and ``"words"`` (other ordinary words)."""
    sections = read_sections("lowercase_fr.txt")
    return {key: _folded_words(sections.get(key, []))
            for key in ("function", "places", "jobs", "persons", "states", "words")}
