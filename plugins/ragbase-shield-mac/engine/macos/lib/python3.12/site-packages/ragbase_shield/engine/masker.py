"""Masker: replace real names and identifiers by dossier codes, restore them, and scan for leaks.

Pipeline of :meth:`Masker.mask`:

1. structured detectors with check digits (``detectors``),
2. person and company rules (``names``) and the optional NER,
3. whitelist filter (courts with their city, institutions, laws, articles),
4. dictionary of known entities (the store) and of the names found in step 2, with variants
   ("Jean-Pierre Dupont" -> "Dupont", "J.-P. Dupont", "DUPONT Jean-Pierre"...),
5. overlap resolution: structured > dictionary > rule > NER, longer wins; a rule name that
   contains a shorter dictionary name of the same kind wins ("Marie Duvernoy" vs "Duvernoy"),
6. keep-clear filter, code allocation through ``store.add`` (variants share one code),
7. replacement from the end. Real text that already looks like a code is escaped
   (``PERSONNE_A`` -> ``PERSONNE-A``) so that restore never treats it as a code.

No network, no model by default. The optional NER (v0.2: a GLiNER model through ``engine/ner.py``, or the
v0.1 spaCy model) is used only when it is installed; its spans go through ``engine/merge.py`` (whitelist,
rules win on kind, precision guards) and never override structured detectors. The language of a text ("fr" or
"en", ``engine/locale.py``) picks the name rules, the protected intervals and the lexicon bundle.
"""
from __future__ import annotations

import bisect
import logging
import os
import re
import unicodedata
from dataclasses import dataclass, replace as dc_replace
from functools import lru_cache
from typing import Callable, Iterable, Optional

from . import detectors as det
from . import directory as dr
from . import locale as lc
from . import merge as mg
from . import names as nm
from . import ner as nr
from . import synthetic as syn
from .normalize import Folded, PhraseMatcher, fold, fold_word, normalize, phrase_tokens, split_windows, tokens
from .types import CODE_RE, MAX_DIGITS, MAX_LETTERS, PREFIXES, SHAPE_KINDS, Entity, EntityStore, MaskResult, Span

__all__ = ["Masker", "entity_norm", "escape_codes", "canonical_codes", "code_like_spans", "ner_available",
           "ner_installed", "prepare_scan", "ScanPrep", "UNMASK_RE"]

log = logging.getLogger(__name__)

RANK = {"regex": 0, "dictionary": 1, "rule": 2, "ner": 3}
_FALLBACK_RANK = 4  # a candidate that a longer or better name replaced: used only where nothing else is
_NAME_KINDS = ("PERSON", "ORG")
_SLOT_KINDS = ("USERNAME", "EMPLOYEE_ID", "ACCOUNT_NUMBER", "ID_CODE")  # found again in their own slot only (v0.3)
# The affiliation word between a person and the company that follows ("Jane Doe of Acme LLC", "Jean Dupont chez
# Acme"): the company span starts after it (v0.3 integration, ``_evidence_pass``).
_AFFILIATION_RE = re.compile(r"[ \t ]+(?:of|at|from|with|chez|de|d')[ \t ]+", re.IGNORECASE)

_LETTER_PREFIXES = sorted({p for _k, (fr, en, letters) in PREFIXES.items() if letters for p in (fr, en)}, key=len, reverse=True)
_NUMBER_PREFIXES = sorted({p for _k, (fr, en, letters) in PREFIXES.items() if not letters for p in (fr, en)}, key=len, reverse=True)

# Codes in a model answer: exact upper-case codes, also glued to punctuation, digits or a lower-case
# suffix ("PERSONNE_A's", "**SOCIETE_B**", "_RG_1_", "PERSONNE_A2026"). Not "PERSONNE A" and not "personne_a".
UNMASK_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"(?P<lp>" + "|".join(_LETTER_PREFIXES) + r")_(?P<ls>[A-Z]{1,%d})(?![A-Z])" % MAX_LETTERS
    + r"|(?P<np>" + "|".join(_NUMBER_PREFIXES) + r")_(?P<ns>[0-9]{1,%d})(?![0-9])" % MAX_DIGITS
    + r")"
)

_ZW = "\u200b\u200c\u200d\u2060\ufeff\u00ad"
_ALL_PREFIXES = "|".join(_LETTER_PREFIXES + _NUMBER_PREFIXES)
# A code a model wrote with a Markdown escape ("PERSONNE\\_A"), a hidden character ("PERSONNE_\u200bA") or a wide
# underscore ("PERSONNE＿A"). unmask reads them as the code (the Markdown renderer would show "PERSONNE_A").
_MANGLED_CODE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<p>" + _ALL_PREFIXES + r")"
    r"(?:\\_|＿[" + _ZW + r"]*|_[" + _ZW + r"]+|[" + _ZW + r"]+[_＿][" + _ZW + r"]*)"
    r"(?P<s>[A-Z]{1,%d}(?![A-Z0-9])|[0-9]{1,%d}(?![0-9]))" % (MAX_LETTERS, MAX_DIGITS)
)
# Near misses that unmask leaves alone but reports: "personne_a", "Societe_B" (wrong case).
_NEAR_CODE_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?P<p>(?i:" + _ALL_PREFIXES + r"))_(?P<s>(?i:[a-z]{1,%d})|[0-9]{1,%d})(?![A-Za-z0-9_])"
    % (MAX_LETTERS, MAX_DIGITS)
)


@lru_cache(maxsize=65536)
def entity_norm(kind: str, value: str) -> str:
    """Canonical key used by the engine to link spellings of one entity.

    PERSON: folded "first names surname" without titles ("DUPONT Jean-Pierre" -> "jean-pierre dupont").
    ORG: folded name without legal form ("ACME SAS" -> "acme"). Structured kinds: compact canonical
    form (IBAN without spaces, phone as 0XXXXXXXXX, ...).
    """
    if kind == "PERSON":
        return nm.person_norm(_comma_order(value)) or normalize(value)
    if kind == "ORG":
        return nm.org_norm(value) or normalize(value)
    return det.structured_norm(kind, value)


_OBJECT_ID_PREFIXES = ("DOC_", "DOSSIER_", "OUTPUT_")


def _comma_order(value: str) -> str:
    """"Varlamov, Constantine" -> "Constantine Varlamov": one comma between two name parts is the "Last, First" order
    (v0.3, S2), so both written orders key to one entity. Any other value is returned as it is."""
    if value.count(",") != 1:
        return value
    before, _, after = value.partition(",")
    before, after = before.strip(), after.strip()
    if not before or not after or not any(c.isalpha() for c in before) or not any(c.isalpha() for c in after):
        return value
    if any(ch.isdigit() for ch in value) or len(after.split()) > 3 or len(before.split()) > 3:
        return value
    if all(nm.is_title_word(w) for w in before.split()) \
            or fold_word(after.split()[0]).rstrip(".") in nm.GENERATIONAL:
        return value  # "M., Jean" is not a name order; "Varlamov, Jr." keeps its order
    return after + " " + before


def code_like_spans(text: str) -> list[tuple[int, int, str]]:
    """``(start, end, code)`` of every text that CODE_RE or UNMASK_RE reads as a code, without overlaps.

    UNMASK_RE is wider than CODE_RE ("USER_EMAIL_1", "PERSONNE_Ab", "TEL_1er" hold a code for restore), so real
    text must be escaped against both, or restore would put a client value into it. Shield object ids (DOC_n,
    DOSSIER_n, OUTPUT_n) are left out: restore never replaces them, and masking a restored text again keeps them.
    """
    found: dict[int, tuple[int, str]] = {}
    for rx in (CODE_RE, UNMASK_RE):
        for m in rx.finditer(text):
            if m.group().startswith(_OBJECT_ID_PREFIXES):
                continue  # Shield object ids are never restored, so they are never escaped (R1, CL-29)
            prev = found.get(m.start())
            if prev is None or prev[0] < m.end():
                found[m.start()] = (m.end(), m.group())
    out: list[tuple[int, int, str]] = []
    last = -1
    for s in sorted(found):
        e, g = found[s]
        if s < last:
            continue
        out.append((s, e, g))
        last = e
    return out


def _escape_one(code: str) -> str:
    """``PERSONNE_A`` -> ``PERSONNE-A`` (same length: offsets keep lining up)."""
    return code.replace("_", "-", 1)


def escape_codes(text: str) -> str:
    """Escape text that already looks like a Shield code (``PERSONNE_A`` -> ``PERSONNE-A``)."""
    parts: list[str] = []
    pos = 0
    for s, e, code in code_like_spans(text):
        parts.append(text[pos:s])
        parts.append(_escape_one(text[s:e]))
        pos = e
    parts.append(text[pos:])
    return "".join(parts)


def canonical_codes(text: str) -> str:
    """Read mangled codes as codes: ``PERSONNE\\_A``, ``PERSONNE_\u200bA``, ``PERSONNE＿A`` -> ``PERSONNE_A``."""
    if "\\" not in text and "＿" not in text and not any(c in text for c in _ZW):
        return text

    def fix(m: re.Match) -> str:
        p, suffix = m.group("p"), m.group("s")
        letters = p in _LETTER_PREFIXES
        if letters != suffix.isalpha():
            return m.group()
        return f"{p}_{suffix}"

    return _MANGLED_CODE_RE.sub(fix, text)


# ---------------------------------------------------------------------------
# Optional NER (v0.2: engine/ner.py backends; the v0.1 spaCy model is the "spacy" backend there)
# ---------------------------------------------------------------------------


def ner_installed(model: str = "auto") -> bool:
    """True when a NER backend is installed and its model is cached. Cheap: nothing is imported or loaded."""
    return nr.ner_installed(model)


def ner_available(model: str = "auto") -> bool:
    """True when a NER backend loads (the model is loaded on the first call; never required)."""
    return nr.ner_available(model)


# ---------------------------------------------------------------------------
# Internal structures
# ---------------------------------------------------------------------------


@dataclass
class _Pending:
    kind: str
    value: str
    norm: str
    entity: Optional[Entity] = None
    mint: bool = False  # an ambiguous mention: its own entity, never linked by surname (R1, CL-04b, CL-20)
    source: str = "rule"  # "rule" or "ner" (v0.2): a one-word NER entity gets no partial variant until confirmed
    signals: tuple = ()   # v0.3 (S7): the second signals of a model find (``merge.SIGNALS``); empty = unconfirmed


@dataclass
class _Cand:
    start: int
    end: int
    kind: str
    source: str
    priority: int
    entity: Optional[Entity] = None
    pending: Optional[_Pending] = None
    rank: int = 0
    dropped: bool = False
    full_text: str = ""
    surface: Optional[str] = None  # text of the span when it was found in a joined view ("KERVA-\nDEC")
    need: str = ""                 # dictionary candidates: the variant's context need and tokens
    vtoks: tuple = ()
    rule: str = ""                 # rule candidates: the rule that found them ("title", "trigger"...)
    signals: tuple = ()            # NER candidates (v0.3, S7): the second signals the merge attached to the find

    def __post_init__(self) -> None:
        self.rank = RANK[self.source]


class _Known:
    """Known people and companies indexed for linking (surname, first word of the company name)."""

    def __init__(self, ents: Iterable[Entity]):
        self.by_surname: dict[str, list[Entity]] = {}
        self.by_org_head: dict[str, list[Entity]] = {}
        for e in ents:
            self.add(e)

    def add(self, e: Entity) -> None:
        if e.kind == "PERSON":
            key = nm.surname_key(e.value)
            if key:
                self.by_surname.setdefault(key, []).append(e)
        elif e.kind == "ORG":
            toks = entity_norm("ORG", e.value).split()
            if toks:
                self.by_org_head.setdefault(toks[0], []).append(e)

    def replace(self, old: Entity, new: Entity) -> None:
        for table in (self.by_surname, self.by_org_head):
            for lst in table.values():
                for k, e in enumerate(lst):
                    if e.code == old.code:
                        lst[k] = new
        self.add(new)


@dataclass
class _Ref:
    kind: str
    variant: nm.Variant
    entity: Optional[Entity] = None
    pending: Optional[_Pending] = None
    full_text: str = ""


@dataclass
class ScanPrep:
    """Work shared by the canary of every dossier on one text (see :func:`prepare_scan`)."""

    text: str
    structured: list
    folded: Folded
    ftoks: list
    split: Optional[list]


def prepare_scan(text: str) -> ScanPrep:
    """Structured detectors, folding and tokens of ``text``, computed once and reused by ``Masker.scan`` of
    every dossier (the canary of N dossiers no longer folds and scans the same text N times)."""
    folded = fold(text, confusables=True)
    ftoks = tokens(folded.text)
    return ScanPrep(text, det.detect_structured(text), folded, ftoks, split_windows(folded, ftoks))


@dataclass
class _Lexicon:
    """Per store revision: variant references and matchers of the known entities (built once)."""

    refs: list
    owners: dict
    strict: tuple
    lenient: tuple
    join_tokens: frozenset
    handle_tokens: list
    bodies: list
    fuzzy: dict = None             # registered name word (6+ letters) -> entities, for one-edit typos (R1, CL-06)
    fuzzy_first: dict = None       # first letter -> those words
    dir_persons: dict = None       # entity code -> directory Person (v0.3, S3)


_TITLE_BEFORE_RE = re.compile(
    # a bare "M" without a dot is a title only with the name on its line and not after a number ("150 M") (CL-18b)
    r"(?:(?<![\w.])(?:M|MM)\.\s*|(?<![\w.])(?<!\d[ \u00a0])(?:M|MM)[ \t\u00a0\u202f]+"
    r"|(?:(?<![\w.])(?:Mr|Mme|Mmes|Mlle|Mlles|Me|Mes|Dr|Pr|ME|MME|MLLE|DR|PR)\.?"
    r"|(?i:monsieur|madame|mademoiselle|ma[îi]tre|docteur|professeur|messieurs|mesdames|"
    r"[ée]poux|sieur|dame|consorts|veuve|n[ée]e))\s*)$"
)
_SALUTE_BEFORE_RE = re.compile(
    r"(?i)(?:\b(?:bonjour|bonsoir|cher|ch[èe]re|chers|salut|hello|coucou|merci|dear|hi)\s*,?\s*$"
    r"|(?:cordialement|bien\s+(?:à|a)\s+(?:vous|toi)|bien\s+cordialement|salutations(?:\s+distingu[ée]es)?"
    r"|à\s+bient[ôo]t|a\s+bientot|bonne\s+(?:journ[ée]e|soir[ée]e)|sinc[èe]rement|amiti[ée]s|regards|best)"
    r"\s*[,.!]?\s*\n\s*$)"
)
_ORG_BEFORE_RE = re.compile(
    r"(?:(?i:soci[ée]t[ée]|st[ée]|groupe|cabinet|enseigne|entreprise|ets|[ée]tablissements|banque|holding)"
    r"|\b(?:SAS|SASU|SARL|EURL|SCI|SNC|SA|SELARL|SELAS|SCP|GIE|AARPI))\s*(?:«\s*|\"\s*)?$"
)
_ORG_AFTER_RE = re.compile(
    r"^\s*(?:»\s*|\"\s*)?(?:SAS|SASU|SARL|EURL|SCI|SNC|SA|SELARL|SELAS|SCP|GIE|AARPI|Sarl|Ltd|LLC|Inc|GmbH|S\.A\.S\.|S\.A\.R\.L\.|S\.A\.)(?![\w])"
)
_PLACE_BEFORE_RE = re.compile(r"(?i)\b(?:à|a|au|aux|en|de|du|des|d'|d’|sur|vers|près de|dans|via)\s*$")
_NAME_GAP = r"[ .'\-]{0,3}"
_ORG_GAP = r"[ .'\-&+]{0,4}"  # "PERNAUD & VALETTE", "Martin + Fils"
_OTHER_GAP = r"[ .'\-,/@:_()+\"]{0,3}"
# A word broken by a hyphen at the end of a line (PDF text): "KERVA-\nDEC".
_HYPHEN_BREAK_RE = re.compile(r"(?<=[^\W\d_])-[ \t ]*\r?\n[ \t ]*(?=[^\W\d_])")
_HANDLE_RE = re.compile(r"(?<![\w@.])@(?P<h>[A-Za-z0-9_][A-Za-z0-9_.\-]{2,40})")
_PLAIN_TITLE_M = re.compile(r"M\.[ \t  ]")

# R1 context rules of the name-alone variants.
_FIRST_NEEDS = ("first", "salutation", "nick", "mid", "casual", "dir")
# v0.4 (T2): the owners of a slot-only form are counted apart, under this key prefix: a slot form yields to any
# other person's first-name form, and two people with one slot form are never guessed.
_SLOT_OWNER = "\x00slot"
# v0.3 (S2+S3): the rules whose person span carries a local cue (a title, a greeting, a sign-off, a speaker label,
# a role word, a possessive, a surname after a comma); a bare first name alone ("first_alone") carries none.
_PERSON_CUE_RULES = frozenset({"title", "greeting", "signoff", "speaker", "role", "possessive_first", "surname_first",
                               "first_initial", "signed", "title_list"})
# Rules that address a person explicitly: they beat a known company name on the same span ("Hi Sterling,").
_PERSON_ADDRESS_RULES = frozenset({"title", "greeting", "signoff", "speaker"})
# A cue that companies carry too: it beats a company head word only with some first-name evidence.
_WEAK_PERSON_CUE_RULES = frozenset({"possessive_first"})
# A first name alone with nothing around it: the only person rule the export's company evidence overrides.
_BARE_PERSON_RULES = frozenset({"first_alone"})
_STRONG_ORG_RULES_FALLBACK = frozenset({"legal_form", "legal_form_after", "field", "partnership", "party_definition",
                                        "suffix", "trigger", "org_of", "possessive", "trade", "domain", "domain_form",
                                        "acronym", "same_surface"})
# Weak company rules: a span they build only from known person tokens is a list of people, not a company
# ("Devon & Jordan," before a verb).
_WEAK_ORG_RULES = frozenset({"partnership", "trigger", "possessive", "org_of", "suffix", "same_surface", "trade"})
# A label that announces a login name or an identifier value on its line ("Username: jdoe", "Employee ID: 4471").
_SLOT_LABEL_RE = re.compile(
    r"(?i)(?:user[ \u00a0]?name|handle|login|user[ \u00a0]?id|screen[ \u00a0]?name|pseudo(?:nyme)?|identifiant|"
    r"nom[ \u00a0]d['’]utilisateur|employee[ \u00a0]?(?:id|#|no\.?|number)|emp[ \u00a0]?(?:id|#|no\.?)|badge|matricule|"
    r"staff[ \u00a0](?:id|no\.?)|account(?:[ \u00a0](?:number|#|no\.?))?|acct\.?|a/c|member(?:ship)?[ \u00a0](?:id|no\.?|#)|"
    r"customer[ \u00a0]id|client[ \u00a0]id|policy(?:[ \u00a0](?:number|no\.?|#))?|routing|compte|n[°º]|ref\.?|id|#)"
    r"[ \u00a0]*[:#=\-–]?[ \u00a0]*$")
# Words that address or name a person right before a first name in casual text ("ping priya", "cc marcus",
# "told sam"): greetings of both languages, verbs of saying and routing words.
_PERSON_CUE_BEFORE_RE = re.compile(
    r"(?i)(?:\b(?:hi|hey|hello|hiya|dear|thanks|thank[ \u00a0]you|thx|ty|cheers|cc|ping|pinged|ask|asked|tell|told|"
    r"said|says|per|attn|with|w/|bonjour|bonsoir|salut|merci|coucou|cher|ch[èe]re|dis|dit|demande|demand[ée]|"
    r"pr[ée]viens|pr[ée]venir)\b[ \u00a0]*[,:]?[ \u00a0]*)$")
_COMMA_GAP = r" ?, ?|[ \-]"
_ELISION_BEFORE_RE = re.compile(r"(?i)(?:^|[^\w])(?:[dlnjmtsc]|qu|jusqu|lorsqu|puisqu|presqu)['’]$")
_PLACE_ONLY_BEFORE_RE = re.compile(r"(?i)\b(?:à|a|au|aux|en|sur|vers|près de|dans|via)\s*$")
# An all-capital prefix of a hyphenated product or file code ("ABC-1234", "ABC-X-2024") (v0.2, Max class A).
_ACRO_CODE_RE = re.compile(r"(?<![A-Za-z0-9])(?P<a>[A-Z]{2,6})(?=-(?:[A-Z0-9]+-)*[A-Z0-9]*\d[A-Z0-9\-]*(?![A-Za-z0-9]))")
# v0.4 (T2, English only): the reading of a lower-case directory first name written alone in chat. Chat
# abbreviations are never a name. A first name that is also a common word ("hope", "mark") is refused when the
# words around it read it as a noun or a verb: a determiner or a possessive before it ("the mark"), a modal, an
# auxiliary or a subject pronoun before it ("we mark"), a bare verb after it ("will do"), a determiner or an object
# pronoun after it at the start of a clause or after "to" ("mark the file", "to mark it"), a place preposition
# before a place name ("in chelsea").
_CHAT_WORDS = frozenset({"ty", "thx", "tx", "pls", "plz", "lol", "lmk", "idk", "tbh", "imo", "imho", "omg", "brb",
                         "btw", "fyi", "np", "nvm", "rn", "smh", "ttyl", "wfh", "ooo", "eod", "eta", "asap", "afaik",
                         "iirc", "irl", "jk", "ok", "kk", "ya", "yep", "yup", "nah", "hmm", "ugh", "wow", "yay",
                         "tho", "cuz", "gonna", "wanna", "gotta", "dm", "pm", "cc", "re", "fw", "fwd"})
_EN_NOUN_BEFORE_RE = re.compile(
    r"(?i)(?:^|[^\w'’])(?:the|a|an|this|that|these|those|my|your|his|her|its|our|their|each|every|any|some|no|"
    r"another|which|what|whose)[ \t ]+$")
_EN_AUX_BEFORE_RE = re.compile(
    r"(?i)(?:^|[^\w'’])(?:will|would|can|could|may|might|must|shall|should|do|does|did|don['’]t|doesn['’]t|"
    r"didn['’]t|won['’]t|can['’]t|i|we|they|he|she|it|"
    r"(?:i|we|they|he|she|it|you)['’](?:ll|d|ve|m|re))[ \t ]+$")  # v0.4 (T8): a pronoun contracted with an auxiliary
_EN_TO_BEFORE_RE = re.compile(r"(?i)(?:^|[^\w'’])to[ \t ]+$")
_EN_CLAUSE_START_RE = re.compile(
    r"(?i)(?:^|[.!?;:\n(]|(?:^|[^\w'’])(?:and|then|so|but|or|pls|please|also|just))[ \t ]*$")
_EN_BARE_VERB_AFTER_RE = re.compile(
    r"(?i)[ \t ]+(?:do|have|be|go|get|take|send|make|see|need|check|run|ping|call|let|try)\b")
_EN_OBJECT_AFTER_RE = re.compile(
    r"(?i)[ \t ]+(?:the|a|an|this|that|these|those|my|your|his|her|its|our|their|it|them|him|me|us)\b")
_EN_PLACE_BEFORE_RE = re.compile(r"(?i)(?:^|[^\w'’])(?:in|from|near|across|around|into|via|throughout)[ \t ]+$")
# v0.4 (T3): local cues of a company mention for a one-word company of the dossier's e-mail domain that is also
# a person's name ("at Northvale", "non-Northvale", "Northvale's revenue", "NORTHVALE", "Northvale | 12 | ..."):
# the word before it, a prefix glued with a hyphen, and the start or end of a table cell.
_LABEL_ORG_PREV_EN = frozenset({"at", "from", "across", "inside", "outside", "via", "within", "joined", "join", "joining",
                                "left", "leaving", "between", "versus", "vs", "among"})
_LABEL_PREFIX_BEFORE_RE = re.compile(r"(?i)(?:^|[^\w])(?:non|ex|pre|post|anti|pro)-$")
_CELL_BEFORE_RE = re.compile(r"(?:^|\n|\||\t|,)[ \t]*$")
_CELL_AFTER_RE = re.compile(r"[ \t]*(?:$|\r?\n|\||\t|,)")
_NEXT_CELL_WORD_RE = re.compile(r"[ \t]*(?:\||\t|,)[ \t]*([^\W\d_][\w'’\-]*)")
_PREV_CELL_WORD_RE = re.compile(r"([^\W\d_][\w'’\-]*)[ \t]*(?:\||\t|,)[ \t]*$")
# v0.4 (T2, English only): the greeting and sign-off slots of a message, where a capitalised first name that is
# also a common word ("Hope", "Rusty") or a short form of a correspondent's first name ("Lis" of Lisette, a first
# name of two letters) is the person: after a greeting with other names first ("Hi Cosima and Lis"), or alone on
# its line as the sign-off
# (the last line after a blank line, a dash before it, or the full signature on the next line).
_EN_GREETING_LIST_RE = re.compile(
    r"(?:\b(?i:hi|hello|hey|dear|thanks|morning)[ \t]+)"
    r"[A-Z][\w'’\-]*(?:[ \t]*,[ \t]*[A-Z][\w'’\-]*)*[ \t]*,?[ \t]*(?:and|&)[ \t]+$")
# The first line of a message that only addresses people: "Cosima and Lis,", "Hi Cosima, Lis & Otti:".
_EN_ADDRESS_LINE_RE = re.compile(
    r"[ \t]*(?:(?i:hi|hello|hey|dear)[ \t]+)?[A-Z][\w'’\-]*"
    r"(?:[ \t]*(?:,|&|\band\b)[ \t]*[A-Z][\w'’\-]*){0,5}[ \t]*[,:!][ \t]*\r?")
_SIGNOFF_HEAD_RE = re.compile(r"[ \t]*(?:(?P<dash>[-–—~])[ \t]*)?")
_SIGNOFF_TAIL_RE = re.compile(r"[ \t]*[.!]?[ \t]*\r?")
# v0.4 (T5, English only): the precision core of the English rules. The dossier itself tells which words are common
# words: its formal text writes them in lower case in mid-sentence (after a lower-case word, a comma or a semicolon).
_T5_MID_RE = re.compile(r"(?<=[a-z,;] )([A-Za-z][A-Za-z'’\-]{2,})")
_T5_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’&\-]*")
_T5_ONE_WORD_RE = re.compile(r"[a-z][a-z'’\-]*")
# an acronym the text declares in brackets: "(ABC)", "(the \"ABC\")"
_T5_DECLARED_RE = re.compile(r"\([ \t]*(?:the[ \t]+|or[ \t]+|aka[ \t]+)?[\"“'‘]?([A-Z][A-Z&]{1,5})[\"”'’]?[ \t]*\)")
# a legal form in capitals right after the span, then a lower-case word: "Lantern LTD bundle" (a product or plan code)
_T5_FORM_LC_RE = re.compile(r"[ \t ]+[A-Z]{2,4}[ \t ]+[a-z]")
_T5_FORM_INSIDE_LC_RE = re.compile(r"[ \t ][A-Z]{2,4}$")
# person rules that carry a person signal (a title, a greeting, a sign-off, a speaker label, a role word...)
_T5_CUE_RULES = frozenset({"title", "greeting", "signoff", "speaker", "role", "possessive_first", "first_alone",
                           "first_initial", "signed", "title_list", "person_cue"})
# a "First Noun" person: the rules saw a capitalised first name and a word after it, nothing else
_T5_FIRST_NOUN_RULES = frozenset({"first_name", "surname_first"})
# one-word company rules with no evidence of their own ("at Xyz", "Xyz's team", a mention propagated in the text)
_T5_WEAK_ONE_WORD_RULES = frozenset({"trigger", "possessive", "same_surface"})
# company rules with evidence of their own: a sighting there keeps the company (fail closed)
_T5_STRONG_ORG_RULES = frozenset({"legal_form", "legal_form_after", "field", "party_definition", "domain",
                                  "domain_form", "acronym", "partnership"})
# v0.4 (T9, English, NER mode only): the second signals of a model find that are evidence of the export (the rules
# found the name here, a context cue, an e-mail domain of the text); the model's own repeat is not
_T9_EVIDENCE_SIGNALS = frozenset({"rule", "cue", "domain"})
# a reference code of capital-letter groups ("QX-TRV-7", "KLM 2031-04"): a document, ticket or territory code
_T9_CODE_RE = re.compile(r"[A-Z]{2,}[A-Z0-9]*(?:[- ][A-Z0-9]+)+")
# an owner word of an account or a staff number right before a model-found code keeps it ("badge QX-5521",
# "Employee ID: QX-5521")
_T9_OWNER_CUE_RE = re.compile(r"(?i)(?<![\w/])(?:account|acct|a/c|employee|emp|staff|badge|member|membership|policy|"
                              r"card|loan|payroll|personnel|worker|routing|iban)(?:[ \t]*(?:number|no\.?|num|id|#|:))*"
                              r"[ \t#:.]*$")
_T9_ID_KINDS = ("USERNAME", "ACCOUNT_NUMBER", "EMPLOYEE_ID")


def _en_common_word(tok: str) -> bool:
    """A vocabulary, function, calendar, place, legal or title word of English (see ``Masker._common_word``)."""
    b = lc.bundle("en")
    return (tok in b.vocab or tok in b.stopwords or tok in b.months_days or tok in b.greetings
            or tok in b.titles or tok in b.legal_terms or tok in b.roles or tok in b.places
            or tok in b.common_words or tok in b.org_generic)


def _written_as_words(surface: str, kind: str) -> bool:
    """v0.4 (T4): a dictionary match of a known e-mail address with no "@", or of a known host or URL with no dot
    and no slash: its words written as prose, not the address."""
    if kind == "EMAIL":
        return "@" not in surface
    if kind == "URL":
        return "." not in surface and "/" not in surface
    return False


def _possessive_at(text: str, e: int) -> bool:
    """An English possessive right after a name ("Priya's")."""
    return text[e:e + 2] in ("'s", "’s") and (e + 2 >= len(text) or not text[e + 2].isalnum())


def _casual_text_fallback(text: str) -> bool:
    """Chat-like text: a chat marker, or at least 30 % of the non-empty lines start with a lower-case letter
    (the engine's own rule until ``locale.casual_text`` of the LOCALE-EN builder exists)."""
    if "<@" in text or "@channel" in text or "@here" in text:
        return True
    lines = [ln.lstrip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    return sum(1 for ln in lines if ln[0].islower()) * 10 >= len(lines) * 3


@lru_cache(maxsize=4)
def _casual_text(text: str) -> bool:
    try:
        from .locale import casual_text  # type: ignore
    except ImportError:
        return _casual_text_fallback(text)
    try:
        return bool(casual_text(text))
    except Exception:  # noqa: BLE001 - never fail a masking on a heuristic
        return _casual_text_fallback(text)


def _casual_cap_ok(text: str, variant: nm.Variant) -> bool:
    """In chat-like text a lower-case mention of a distinctive registered name word is still that name ("sent it
    to acme"): every significant token has 4+ letters and is not a dictionary word (v0.2)."""
    sig = [t for t in variant.tokens if t not in nm.PARTICLES]
    return bool(sig) and all(len(t) >= 4 and t.isalpha() and not _dictionary_word(t) for t in sig) \
        and _casual_text(text)


def _code_acronyms(value: str) -> set[str]:
    """Upper-case acronyms (2 to 6 letters) of a company name, with and without its generic words, for the
    hyphenated-code rule only (no stoplist: the code context is narrow enough)."""
    words = [w for w in nm._words(nm.org_core(value)) if not w.digit and w.low != "&" and not nm._legal_form(w)]
    named = [w for w in words if not w.prefix and w.low not in nm.PARTICLES and w.low not in nm._ORG_CONNECT]
    out: set[str] = set()
    if len(named) < 2:
        return out
    for ws in (words, named, [w for w in named if w.low not in nm.ORG_GENERIC]):
        letters = "".join(w.low[0] for w in ws if w.low and w.low[0].isalpha())
        if 2 <= len(letters) <= 6:
            out.add(letters.upper())
    return out
_LINE_START_RE = re.compile(r"(?:^|[.!?…:;]|\n)[\s«»\"“”()\-–—•*]*$")
_DETERMINER_BEFORE_RE = re.compile(
    r"(?i)(?:(?:^|[^\w'’])(?:le|la|les|un|une|des|du|au|aux|ce|cet|cette|ces|mon|ma|mes|ton|ta|tes|son|sa|ses|notre"
    r"|nos|votre|vos|leur|leurs)\s+|(?:^|[^\w])l['’])$")
_PARTY_WORDS = r"(?:c/|c\.|C/|C\.|contre|Contre|CONTRE|v\.|vs\.?|VS\.?)"
_PARTY_AFTER_RE = re.compile(r"[ \t\u00a0]+" + _PARTY_WORDS + r"[ \t\u00a0]")
_PARTY_BEFORE_RE = re.compile(r"(?:^|[ \t\u00a0(])" + _PARTY_WORDS + r"[ \t\u00a0]+$")
_DOTTED_RE = re.compile(r"(?:[^\W\d_]\.[ \t\u00a0\-]*)+")
_QUALIFIER_RE = re.compile(r"[ \t\u00a0]+(?:fils|p[èe]re|junior|Junior|senior|Senior|a[îi]n[ée]|cadet|Jr\.?|Sr\.?)(?![\w])")
_SIGN_LINE_RE = re.compile(r"(?i)\b(?:paraphes?|initiales|vis[ée]s?|sign[ée]e?s?|signatures?)\b")
_REF_BEFORE_RE = re.compile(r"(?i)(?:\b[nv]s?\s*/\s*|\b)r[ée]f(?:[ée]rences?)?\b")


@lru_cache(maxsize=65536)
def _strong_tokens(toks: tuple[str, ...]) -> bool:
    """True when a variant is distinctive enough to match even glued to another word ("Atlantide-note")."""
    sig = [t for t in toks if t not in nm.PARTICLES]
    return bool(sig) and sum(len(t) for t in sig) >= 5 and not nm._weak(toks)


class Masker:
    """Mask, restore and scan text against one entity store (one dossier).

    ``use_ner``: ``None`` uses the optional NER when a backend is installed (``engine/ner.py``: a GLiNER model
    downloaded with ``ragbase-shield ner download``, or the v0.1 spaCy model); ``True`` asks for it (warning if
    missing); ``False`` disables it. ``ner_model``: "auto", "fastino", "urchade" or "spacy". ``ner`` lets a
    caller plug another recogniser: a function ``text -> [(start, end, "PERSON"|"ORG"), ...]``. The model is
    loaded on the first :meth:`mask` that needs it, never at construction (hooks build maskers often).
    ``language``: "auto" (detected per text, default French), "fr" or "en": picks the name rules, the protected
    intervals and the lexicon bundle (v0.2). ``own_company_vendor`` (v0.4, P1, policy option): ``True`` masks a
    public tool or vendor of the English whitelist when it is the dossier's own company (the label of the
    dossier's main e-mail domain); ``None`` reads ``RAGBASE_SHIELD_OWN_COMPANY_VENDOR`` (default off).
    """

    def __init__(
        self,
        store: EntityStore,
        *,
        locale: str = "fr",
        language: str = "auto",
        use_ner: Optional[bool] = None,
        ner: Optional[Callable[[str], list[tuple[int, int, str]]]] = None,
        ner_model: str = "auto",
        replacement: str = "codes",
        directory: Optional[dr.Directory] = None,
        own_company_vendor: Optional[bool] = None,
    ):
        self.store = store
        self.locale = locale
        # v0.4 (P1, policy option, OFF by default): the vendor of the whitelist that is the dossier's own company
        # is masked. None = the setting RAGBASE_SHIELD_OWN_COMPANY_VENDOR, read at each mask call.
        self.own_company_vendor = own_company_vendor
        self._vendor_cache: tuple = (None, frozenset())
        # v0.3 (S2+S3): the dossier directory (people from headers, Slack users, document authors, signatures and
        # nickname declarations) and the export-level evidence of what each surface is (given, family, org,
        # username). Both live in the store when it keeps them (the vault); a plain store gets them here.
        self._directory_arg = directory
        self._dir_cache: tuple = (None, None)
        self._dir_seeded: object = None
        self._evidence_local: dict[str, dict[str, int]] = {}
        self._known_cache: tuple = (None, None)
        self._cc: dict = {}  # per-call caches of _detect (keys, scores, roles, evidence)
        self.language = language if language in ("fr", "en") else "auto"
        self._lang = "fr"  # language of the text being processed (set by mask and scan)
        self.ner_model = ner_model or "auto"
        self._backend: Optional[nr.NerBackend] = None
        self._backend_tried = False
        self._ner_skipped = False
        self._watch: list[tuple[int, int, str, str]] = []  # v0.3 (S7): review-step finds of the last mask call
        if replacement not in ("codes", "synthetic"):
            raise ValueError("replacement must be 'codes' or 'synthetic'")
        # v0.2 (SYNTH): "synthetic" writes realistic pseudonyms instead of codes (section 4.5 of the design).
        # The fake values live in the store (vault, MemoryStore) or, for a store without that support, in
        # ``_synth_cache`` of this masker. The canary, the hooks and the label pass build their own maskers in
        # codes mode: synthetic mode never weakens them.
        self.replacement = replacement
        self._synth_cache: dict[str, str] = {}
        self._synthesizer: Optional[syn.Synthesizer] = None
        self._syn_lex: Optional[tuple] = None
        self._syn_key: object = None
        self._ner_fn = ner
        if ner is not None:
            self.use_ner = use_ner is not False
        elif use_ner is False:
            self.use_ner = False
        else:
            installed = ner_installed(self.ner_model)
            if use_ner and not installed:
                log.warning("RAGbase Shield: NER requested but no model is installed; rules only "
                            "(run: ragbase-shield ner download)")
            self.use_ner = installed
        self._lex_key: object = None
        self._lex: Optional[_Lexicon] = None
        self._labels_cache: tuple = (None, frozenset())  # v0.4 (T4): e-mail domain labels the dossier shows
        # v0.4 (T5, English): what this masker saw of the dossier, in memory only. Each distinct text is read
        # once: how its formal text writes each word in mid-sentence ([lower case, capitalised]), the acronyms it
        # declares in brackets, and which rules found each name (kind, norm). Missing knowledge (a new session)
        # only keeps more masks: the precision rules need it to drop a mask (fail closed).
        self._t5_seen: set = set()
        self._t5_mid: dict[str, list[int]] = {}
        self._t5_declared: set[str] = set()
        self._t5_prov: dict[tuple[str, str], set[str]] = {}
        # v0.4 (T8): the caller's word on the text's kind for one mask call (``mask(chat=True)``: a chat message
        # is chat text whatever its lines look like); None = decide from the text itself
        self._chat_hint: Optional[bool] = None
        # v0.4 (T9, English, NER mode only): the names the model found alone in this session, with the evidence
        # signals any sighting carried, and the codes of the store entities that a model find alone created
        self._t9_model: dict[tuple[str, str], set[str]] = {}
        self._t9_made: set[str] = set()
        self._t9_minted: set[tuple[str, str]] = set()

    @property
    def mode(self) -> str:
        """"rules", "rules+ner" (a plugged function) or "rules+ner:<backend>", with "+synthetic" in synthetic mode."""
        if not self.use_ner:
            base = "rules"
        elif self._ner_fn is not None:
            base = "rules+ner"
        else:
            name = self._backend.name if self._backend is not None else nr.backend_name(self.ner_model)
            base = "rules+ner:" + name if name else "rules+ner"
        return base + "+synthetic" if self.replacement == "synthetic" else base

    # ------------------------------------------------------------------
    # Language of a text (v0.2, LOCALE-EN)
    # ------------------------------------------------------------------

    def _language_of(self, text: str) -> str:
        if self.language in ("fr", "en"):
            return self.language
        # a short text (under 40 letters) takes the default: the language of the code labels
        return lc.detect_language(text, default="en" if self.locale == "en" else "fr")

    def _title_re(self) -> "re.Pattern[str]":
        return lc.title_before_re(self._lang) if self._lang == "en" else _TITLE_BEFORE_RE

    def _salute_re(self) -> "re.Pattern[str]":
        return lc.salute_before_re(self._lang) if self._lang == "en" else _SALUTE_BEFORE_RE

    def _dict_word(self, tok: str) -> bool:
        return lc.dictionary_word(tok, self._lang) if self._lang == "en" else _dictionary_word(tok)

    def _ner_backend(self) -> Optional[nr.NerBackend]:
        """The NER backend, loaded on the first call (never at construction). None: rules only."""
        if self._backend is None and not self._backend_tried:
            self._backend_tried = True
            self._backend = nr.load_backend(self.ner_model)
            if self._backend is None:
                self.use_ner = False
        return self._backend

    # ------------------------------------------------------------------
    # Known entities
    # ------------------------------------------------------------------

    def _index(self) -> tuple[list[Entity], dict[tuple[str, str], Entity]]:
        ents = list(self.store.all())
        idx: dict[tuple[str, str], Entity] = {}
        for e in ents:
            idx.setdefault((e.kind, e.norm), e)
            try:
                idx.setdefault((e.kind, entity_norm(e.kind, e.value)), e)
            except Exception:  # pragma: no cover - defensive: never fail on a stored value
                pass
        return ents, idx

    @staticmethod
    def _variants(kind: str, value: str) -> list[nm.Variant]:
        if kind == "PERSON":
            return nm.person_variants(value)
        if kind == "ORG":
            return nm.org_variants(value)
        toks = phrase_tokens(value)
        out = [nm.Variant(toks)] if toks else []
        if kind == "BIRTHDATE":
            out.extend(nm.Variant(t) for t in _date_forms(value) if t != toks)
        return out

    @staticmethod
    def _matchers(refs: Iterable[_Ref], confusables: bool = False) -> tuple[PhraseMatcher, ...]:
        names = PhraseMatcher(gap=_NAME_GAP)
        orgs = PhraseMatcher(gap=_ORG_GAP)
        other = PhraseMatcher(gap=_OTHER_GAP)
        comma = PhraseMatcher(gap=_COMMA_GAP)  # "Surname, Firstname" (R1, CL-05)
        for ref in refs:
            toks = ref.variant.tokens
            if confusables:
                toks = tuple(fold_word(t, confusables=True) for t in toks)
            if ref.variant.need == "comma":
                comma.add(toks, ref)
                continue
            (names if ref.kind == "PERSON" else orgs if ref.kind == "ORG" else other).add(toks, ref)
        return names, orgs, other, comma

    @staticmethod
    def _owners(refs: Iterable[_Ref], into: Optional[dict] = None) -> dict:
        """Who owns each first-name-alone variant (two known people with one first name: never guess)."""
        owners: dict[tuple[str, ...], set] = into if into is not None else {}
        for ref in refs:
            if ref.variant.need in _FIRST_NEEDS:
                who = ref.entity.code if ref.entity is not None else id(ref.pending)
                owners.setdefault(ref.variant.tokens, set()).add(who)
            elif ref.variant.need == "dir_slot":
                who = ref.entity.code if ref.entity is not None else id(ref.pending)
                owners.setdefault((_SLOT_OWNER,) + ref.variant.tokens, set()).add(who)
        return owners

    def _refs_for(self, kind: str, value: str, entity: Optional[Entity] = None,
                  pending: Optional[_Pending] = None) -> list[_Ref]:
        variants = self._variants(kind, value)
        source = entity.source if entity is not None else (pending.source if pending is not None else "")
        confirmed = pending is not None and bool(pending.signals)
        if source == "ner" and not confirmed:
            # v0.2: a one-word entity the model found alone gets no partial and no lower-case variant until a
            # fuller sighting (two words, or the rules) upgrades it: a model mistake stays contained.
            # v0.3 (S7, gated layer): the same containment for every model-only find that no second signal
            # confirms (a confirmed find is stored as a detected entity and gets the full variants), and no
            # dictionary variant at all for a login name or an identifier of the model alone: its mentions are
            # masked where the model finds them, never spread over the dossier.
            if kind in _SLOT_KINDS:
                return []
            variants = [dc_replace(v, case="cap") if v.case == "any" else v
                        for v in variants if not v.partial and v.need in ("", "title", "party")]
        elif kind == "ORG" and self._lang == "en" and len(value.split()) > 1 and not any(ch.isupper() for ch in value):
            # v0.4 (T4): a company of several words seen only in lower case (a chat form of an e-mail domain,
            # "grelmont quay") gets no partial variant made of English common words: no sighting shows such a
            # word as a name ("meadow" of "meadow vane" stays a word; "grelmont" of "grelmont quay" stays a variant)
            variants = [v for v in variants if not (v.partial and all(_en_common_word(t) for t in v.tokens))]
        if kind in _SLOT_KINDS:
            # v0.3 (S2): a login name or an identifier is found again only in its own slot (after "@", after a
            # label, or shaped like an id), never as a word of the prose: the handle "octavia" does not hide the
            # person Octavia, the handle "grant" does not mask every "grant"
            slot = "slot:USERNAME" if kind == "USERNAME" else "slot:ID"
            variants = [dc_replace(v, need=slot) if not v.need else v for v in variants]
        return [_Ref(kind, v, entity, pending, value) for v in variants]

    def _dir_refs(self, e: Entity, person: dr.Person) -> list[_Ref]:
        """Variants a directory person adds to its store entity (v0.3, S3): the usual first name and every
        nickname (declared, display name, e-mail local part) alone, in any case under the ``dir`` context rule,
        and with the family name in both orders."""
        out: list[_Ref] = []
        st = tuple(person.family)
        seen: set[tuple] = set()
        alone = [person.given[0]] if person.given else []
        alone += [n for n in person.nicknames if n not in alone]
        if person.given:
            # the standard diminutives of the usual first name, and its prefixes that are first names themselves
            # ("bernie" of "berniece", never "ster" of "sterling"), as chat writes them in lower case; the "dir"
            # context rule keeps the common-word and short ones under a cue
            g = person.given[0]
            table = set(lx_nicknames(g))
            alone += [n for n in nm._nick_forms(g) if n not in alone and (n in table or _any_first_name(n))]
        existing = {(v.tokens, v.need) for v in self._variants("PERSON", e.value)}
        for tok in alone:
            if len(tok) < 3 or not tok.isalpha() or tok in nm.PARTICLES or nm.is_title_word(tok):
                continue
            key = ((tok,), "dir")
            if key in seen:
                continue
            seen.add(key)
            out.append(_Ref("PERSON", nm.Variant((tok,), need="dir", case="any", partial=True), e, None, e.value))
            if st and tok not in person.given:
                for toks, need in (((tok,) + st, "joined"), (st + (tok,), "joined")):
                    if (toks, "") not in existing and (toks, need) not in seen:
                        seen.add((toks, need))
                        out.append(_Ref("PERSON", nm.Variant(toks, need=need), e, None, e.value))
                if len(st) == 1 and (st + (tok,), "comma") not in existing:
                    out.append(_Ref("PERSON", nm.Variant(st + (tok,), need="comma"), e, None, e.value))
        if person.given:
            # v0.4 (T2): forms found only in a greeting or sign-off slot, capitalised ("dir_slot"): a first name or
            # nickname of two letters, a standard diminutive of the usual first name (from the nickname table,
            # also when it is a common word), and the three-letter short form of a longer usual first name ("Lis"
            # of Lisette) when that form is no common word. English only: the lexicon is cached
            # for every language, so the forms are built from the English lists and only an English text accepts
            # them (:meth:`_dir_slot_ok`)
            g = person.given[0]
            b = lc.bundle("en")
            slot = [(n, True) for n in [g] + list(person.nicknames) if len(n) == 2]
            slot += [(n, True) for n in sorted(lx_nicknames(g))]
            if len(g) >= 5:
                slot.append((g[:3], False))
            have = {k[0] for k in seen}
            for tok, may_be_common in slot:
                if not tok.isalpha() or len(tok) < 2 or (tok,) in have or tok in nm.PARTICLES \
                        or nm.is_title_word(tok) or tok in b.stopwords or tok in b.months_days:
                    continue
                if not may_be_common and _en_common_word(tok):
                    continue
                have.add((tok,))
                out.append(_Ref("PERSON", nm.Variant((tok,), need="dir_slot", case="cap", partial=True), e, None,
                                e.value))
        return out

    def _store_rev(self) -> object:
        rev = getattr(self.store, "revision", None)
        if rev is None:
            return None
        try:
            return rev() if callable(rev) else rev
        except Exception:  # pragma: no cover - a broken store: no cache
            return None

    def _lexicon(self, ents: Optional[list[Entity]] = None) -> _Lexicon:
        """Variant references and matchers of the store's entities, cached until the store changes."""
        rev = self._store_rev()
        if rev is not None and self._lex is not None and self._lex_key == rev:
            return self._lex
        ents = list(self.store.all()) if ents is None else ents
        refs: list[_Ref] = []
        directory = self._directory()
        dir_persons: dict[str, dr.Person] = {}
        for e in ents:
            refs.extend(self._refs_for(e.kind, e.value, entity=e))
            if directory is not None and e.kind == "PERSON":
                person = _dir_person_for(directory, e)
                if person is not None:
                    dir_persons[e.code] = person
                    refs.extend(self._dir_refs(e, person))
        join: set[str] = set()
        handles: list[tuple[str, Entity]] = []
        for code, person in dir_persons.items():
            ent = next((x for x in ents if x.code == code), None)
            for h in person.handles:
                key = re.sub(r"[^a-z0-9]", "", h)
                if ent is not None and len(key) >= 5 and not _dictionary_word(key):
                    handles.append((key, ent))
        bodies: dict[str, list[str]] = {}
        for e in ents:
            if e.kind in _NAME_KINDS:
                join.update(t for t in phrase_tokens(e.value) if len(t) >= 3)
                for tok in _handle_tokens(e):
                    handles.append((tok, e))
            else:
                body = _lenient_body(e.kind, e.value)
                if body:
                    bodies.setdefault(e.kind, []).append(body)
        patterns = []
        for kind, items in bodies.items():
            uniq = sorted(set(items), key=len, reverse=True)
            for k in range(0, len(uniq), 200):
                patterns.append((kind, re.compile(r"(?<![0-9A-Za-z])(?:" + "|".join(uniq[k:k + 200])
                                                  + r")(?![0-9A-Za-z])")))
        fuzzy: dict[str, list[Entity]] = {}
        for e in ents:
            for tok in _fuzzy_tokens(e):
                fuzzy.setdefault(tok, []).append(e)
        fuzzy_first: dict[str, list[str]] = {}
        for tok in fuzzy:
            fuzzy_first.setdefault(tok[0], []).append(tok)
        lex = _Lexicon(refs=refs, owners=self._owners(refs), strict=self._matchers(refs),
                       lenient=self._matchers(refs, confusables=True), join_tokens=frozenset(join),
                       handle_tokens=handles, bodies=patterns, fuzzy=fuzzy, fuzzy_first=fuzzy_first,
                       dir_persons=dir_persons)
        if rev is not None:
            self._lex, self._lex_key = lex, rev
        return lex

    # ------------------------------------------------------------------
    # Dictionary matching
    # ------------------------------------------------------------------

    def _chat_text(self, text: str) -> bool:
        """Chat-like text (v0.4, T8): the caller's word for this mask call (``mask(chat=...)``), else the text's
        own shape (:func:`_casual_text`)."""
        if self._chat_hint is not None:
            return self._chat_hint
        return _casual_text(text)

    def _context_ok(self, text: str, s: int, e: int, need: str) -> bool:
        if not need or need in ("comma", "acro", "inits"):
            return True
        before = text[max(0, s - 40):s]
        if need.startswith("slot:"):
            return _in_slot(text, s, e, need[5:])
        if need == "dir":
            return self._dir_context_ok(text, s, e, before)
        if need == "dir_slot":
            return self._dir_slot_ok(text, s, e, before)
        if need == "joined":
            return not re.search(r"[.!?;:\n]", text[s:e])  # "Tavi. Brennholt" is two sentences, not a name
        if need == "title":
            return bool(self._title_re().search(before))
        if need == "salutation":
            if not self._salute_re().search(before):
                return False
            after = text[e:e + 3]
            return not re.match(r"[ \t]+[A-ZÀ-Þ]", after)
        if need == "org":
            return bool(_ORG_BEFORE_RE.search(before) or _ORG_AFTER_RE.match(text[e:e + 14]))
        if need in ("first", "nick", "mid", "casual"):
            # alone: not glued to another word ("d'Aurore" is fine: an elision, R1 CL-04c), not followed by a
            # capitalised word, never right after a person title ("M. Gaspard" is a surname, R1 CL-20)
            elided = s > 0 and text[s - 1] in "'’" and bool(_ELISION_BEFORE_RE.search(text[max(0, s - 12):s]))
            at_sign = need == "casual" and s > 0 and text[s - 1] == "@"
            if s > 0 and not elided and not at_sign and (text[s - 1] in "-'’" or text[s - 1].isalnum()):
                return False
            if e < len(text) and (text[e] in "-'’" or text[e].isalnum()) and not _possessive_at(text, e):
                return False  # "Priya's" is Priya (v0.2, English possessive)
            if re.match(r"[ \t]+[A-ZÀ-Þ]", text[e:e + 3]) or self._title_re().search(before):
                return False
            if need == "nick" and self._salute_re().search(before):
                return True
            if need == "casual":
                # a registered first name in any case: chat-like text, a possessive, "@priya", "priya," at a
                # line start (v0.2, DETECTORS). v0.3 (S3): in lower case, a first name that is also a common
                # word of the language ("grant", "mark", "pat") needs a person cue, chat or not
                if at_sign or _possessive_at(text, e) \
                        or (text[e:e + 1] == "," and bool(_LINE_START_RE.search(before))):
                    return True
                if not self._chat_text(text):
                    return False
                if text[s:s + 1].islower() and self._common_word(fold_word(text[s:e])):
                    return bool(self._salute_re().search(before)) or bool(_PERSON_CUE_BEFORE_RE.search(before[-30:]))
                return True
            if re.search(r"[A-ZÀ-Þ][\w'’\-]*[ \t]+$", before):
                return False
            if need == "first":
                return elided or not _PLACE_BEFORE_RE.search(before)
            if need == "nick":
                return not _PLACE_ONLY_BEFORE_RE.search(before)
            # "mid": a capitalised word in mid-sentence, not after a determiner ("le Boulanger") (R1, CL-04d, CL-05)
            return not (_LINE_START_RE.search(before) or _DETERMINER_BEFORE_RE.search(before))
        if need == "party":
            if (s > 0 and text[s - 1].isalnum()) or (e < len(text) and text[e].isalnum()):
                return False
            return bool(_PARTY_AFTER_RE.match(text[e:e + 14]) or _PARTY_BEFORE_RE.search(text[max(0, s - 14):s]))
        if need == "mixed_line":
            # inside a phrase ("Requête BOULANGER c/ X", "de BOULANGER est"), not a lone word between punctuation
            # ("Couleur : BLANC ;"), on a line that is not all capitals
            ls = text.rfind("\n", 0, s) + 1
            le = text.find("\n", e)
            le = len(text) if le < 0 else le
            if not any(c.islower() for c in text[ls:le]):
                return False
            return bool(re.search(r"[^\W_][ \t\u00a0]+$", text[ls:s]) or re.match(r"[ \t\u00a0]+[^\W_]", text[e:le]))
        if need == "dotted":
            # "L. P.", "J.-L. P.": every letter has its dot; not part of a longer chain of initials or of "L. P. Name"
            if e >= len(text) or text[e] != "." or not _DOTTED_RE.fullmatch(text[s:e + 1]):
                return False
            if re.search(r"(?:[^\W\d_]\.[ \t\u00a0\-]*|[^\W_])$", text[max(0, s - 4):s]):
                return False
            return not re.match(r"[ \u00a0\-]?(?:[^\W\d_]\.|[A-ZÀ-Þ][a-zß-ÿ]|[A-ZÀ-Þ]{3})", text[e + 1:e + 6])
        if need == "ini_ctx":
            return _initials_context(text, s, e)
        return False

    def _common_word(self, tok: str) -> bool:
        """A vocabulary, function, calendar, place, legal or title word of the language: a first name that is
        also one of these ("grant", "mark", "pat") needs a person cue when written alone (v0.3, S3)."""
        if self._lang == "en":
            return _en_common_word(tok)
        stop, legal, common, _fn, _afn, months, wl = nm._lists()
        return tok in stop or tok in legal or tok in common or tok in months or tok in wl or tok in nm.ORG_GENERIC

    def _generic_head(self, tok: str) -> bool:
        """A head word that names no company by itself: a function word or a generic company word."""
        if len(tok) < 2 or tok in nm.ORG_GENERIC or tok in nm.PARTICLES or tok in nm._ORG_CONNECT:
            return True
        if self._lang == "en":
            b = lc.bundle("en")
            return tok in b.stopwords or tok in b.org_generic or tok in b.org_suffixes or tok in b.legal_forms
        return tok in nm._lists()[0] or tok in nm.LEGAL_FORMS

    def _lexicon_first_name(self, tok: str) -> bool:
        """A first name of the lexicon of the current language (either lexicon for a short text)."""
        if self._lang == "en":
            return tok in lc.bundle("en").all_first_names
        return tok in nm._lists()[4]

    def _dir_context_ok(self, text: str, s: int, e: int, before: str) -> bool:
        """Context rule of a directory first name or nickname written alone (v0.3, S3): capitalised, it is the
        person unless a title precedes it (a surname), a capitalised word follows (a longer name) or a place
        preposition precedes it; a name that is also a common word needs a person cue or a mid-sentence
        position. In lower case it needs chat-like text, and a person cue when it is a common word or short."""
        tok = text[s:e]
        low = fold_word(tok)
        elided = s > 0 and text[s - 1] in "'’" and bool(_ELISION_BEFORE_RE.search(text[max(0, s - 12):s]))
        at_sign = s > 0 and text[s - 1] == "@"
        if s > 0 and not elided and not at_sign and (text[s - 1] in "-'’" or text[s - 1].isalnum()):
            return False
        if e < len(text) and (text[e] in "-'’" or text[e].isalnum()) and not _possessive_at(text, e):
            return False
        if self._title_re().search(before):
            return False
        own = tok[:1].isupper() and self._after_own_full_name(before, low)  # "Marcus Trov Marc": a "goes by" cell
        if re.match(r"[ \t]+[A-ZÀ-Þ][a-zß-ÿ]", text[e:e + 3]) and not own:
            return False
        cue = at_sign or _possessive_at(text, e) or bool(self._salute_re().search(before)) \
            or (text[e:e + 1] in ",:" and bool(_LINE_START_RE.search(before))) \
            or bool(_PERSON_CUE_BEFORE_RE.search(before[-30:]))
        common = self._common_word(low)
        if tok[:1].isupper():
            if self._lang == "en" and (own or self._name_slot(text, s, e, before)):
                return True  # v0.4 (T2): a greeting, sign-off or "goes by" slot ("Hopewell Wrexley (Hope)")
            cap_before = re.search(r"([A-ZÀ-Þ][\w'’\-]*)[ \t]+$", before)
            if cap_before and self._lang == "en" and not common \
                    and fold_word(cap_before.group(1)) in lc.bundle("en").stopwords \
                    and not _EN_NOUN_BEFORE_RE.search(before) and not self._org_head_evidence(low):
                # v0.4 (T2): "When Kes asked": a capitalised function word makes no phrase; a determiner does
                # ("The Wrenfield summit"); a common word ("Hope") and a word the export shows as a company keep
                # the old rule (a common word is the person outside a slot only with a cue)
                cap_before = None
            if cap_before and not cue and not own:
                return False  # "Northvale Priya": part of a capitalised phrase
            if not common:
                return elided or cue or not _PLACE_BEFORE_RE.search(before)
            return cue or not (_LINE_START_RE.search(before) or _DETERMINER_BEFORE_RE.search(before)
                               or _PLACE_BEFORE_RE.search(before))
        if not self._chat_text(text):
            return cue and not common
        if common or len(low) <= 3:
            return cue or (self._lang == "en" and self._chat_form_ok(text, e, low, common, before))
        return True

    def _chat_form_ok(self, text: str, e: int, low: str, common: bool, before: str) -> bool:
        """v0.4 (T2, English): a lower-case directory first name or nickname written alone in chat, with no
        person cue. A form of three letters that is not a common word is the person ("kes", "lis"); a form that
        is also a common word ("hope", "rusty") is the person unless the words around it read it as a function
        word, a noun or a verb. Never a chat abbreviation, and never a word the export shows as a company (a
        domain label of the correspondents, a company or its head word): company evidence wins ("northvale")."""
        if low in _CHAT_WORDS or len(low) < 3 or self._org_head_evidence(low):
            return False
        if not common:
            return True
        b = lc.bundle("en")
        if low in b.stopwords or _EN_NOUN_BEFORE_RE.search(before) or _EN_AUX_BEFORE_RE.search(before):
            return False
        after = text[e:e + 24]
        if _EN_BARE_VERB_AFTER_RE.match(after):
            return False
        if _EN_OBJECT_AFTER_RE.match(after) and (_EN_CLAUSE_START_RE.search(before) or _EN_TO_BEFORE_RE.search(before)):
            return False
        if low in b.places and _EN_PLACE_BEFORE_RE.search(before):
            return False
        return True

    def _name_slot(self, text: str, s: int, e: int, before: str) -> bool:
        """v0.4 (T2, English): the capitalised word at ``text[s:e]`` sits in a greeting or sign-off slot: a
        greeting with other names before it ("Hi Cosima and Lis"), a first line that only addresses people
        ("Cosima and Lis,"), or alone on its line as a message's sign-off,
        after a blank line or a closing word ("Best,"), with a dash before it ("—Hope"), or as the last line of the
        text, or with the full signature on the next line ("Hope" then "Hope Wrenfield", "Lis" then "Lisette
        Merriwen")."""
        if _EN_GREETING_LIST_RE.search(before):
            return True
        ls = text.rfind("\n", 0, s) + 1
        le = text.find("\n", e)
        le = len(text) if le < 0 else le
        if le < len(text) and not text[:ls].strip() and _EN_ADDRESS_LINE_RE.fullmatch(text[ls:le]):
            return True  # the first line of the message only addresses people ("Cosima and Lis,")
        head = _SIGNOFF_HEAD_RE.fullmatch(text[ls:s])
        if head is None or not _SIGNOFF_TAIL_RE.fullmatch(text[e:le]):
            return False
        if head.group("dash"):
            return True
        prev = text[max(0, ls - 60):ls]
        if self._salute_re().search(prev):
            return True  # "Best,\n" on the line before
        if not re.search(r"\n[ \t\r]*\n[ \t\r]*$", prev):
            return False  # the sign-off is a paragraph of its own
        rest = text[le:].lstrip(" \t\r\n")
        if not rest:
            return True  # the last line of the message
        tok = text[s:e]
        return bool(re.match(re.escape(tok) + r"[\w'’\-]*[ \t]+[A-ZÀ-Þ]", rest))  # the full signature follows

    def _dir_slot_ok(self, text: str, s: int, e: int, before: str) -> bool:
        """Context rule of a "dir_slot" form (v0.4, T2, English): capitalised, not glued to another word, no title
        before it, no capitalised word after it, and in a greeting or sign-off slot (:meth:`_name_slot`), at the
        start of a message or line before a comma ("Lis,"), or right after a greeting ("Hi Lis")."""
        if self._lang != "en" or not text[s:s + 1].isupper() or (e - s > 1 and text[s:e].isupper()):
            return False  # lower case or all capitals (an acronym): not a short name
        if s > 0 and (text[s - 1] in "-'’" or text[s - 1].isalnum()):
            return False
        if e < len(text) and (text[e] in "-'’" or text[e].isalnum()) and not _possessive_at(text, e):
            return False
        if self._title_re().search(before):
            return False
        if self._after_own_full_name(before, fold_word(text[s:e])):
            return True  # "goes by": right after the person's own full name ("Hopewell Wrexley (Hope)")
        if re.match(r"[ \t]+[A-ZÀ-Þ][a-zß-ÿ]", text[e:e + 3]):
            return False
        if bool(self._salute_re().search(before)) or (text[e:e + 1] in ",:" and bool(_LINE_START_RE.search(before))):
            return True
        return self._name_slot(text, s, e, before)

    def _casual_cap_ok(self, text: str, variant: nm.Variant) -> bool:
        """In chat-like text a lower-case mention of a distinctive registered name word is still that name ("sent
        it to acme"): every significant token has 4+ letters and is not a common word of the language of the
        text (v0.3, S3: the English bundle for English text; the v0.2 rule used the French lists only)."""
        sig = [t for t in variant.tokens if t not in nm.PARTICLES]
        if not sig or not self._chat_text(text):
            return False
        if self._lang == "en":
            return all(len(t) >= 4 and t.isalpha() and not self._common_word(t) for t in sig)
        return all(len(t) >= 4 and t.isalpha() and not _dictionary_word(t) for t in sig)

    def _case_ok(self, text: str, folded: Folded, ftoks: list[tuple[str, int, int]], i: int, j: int,
                 variant: nm.Variant) -> bool:
        if variant.case == "any" and not variant.initials:
            return True
        for k in range(i, j):
            tok, fs, fe = ftoks[k]
            s, e = folded.span(fs, fe)
            surface = text[s:e]
            if not surface:
                return False
            if variant.initials and len(tok) == 1 and not surface[0].isupper():
                return False
            if variant.case == "upper":
                letters = [c for c in surface if c.isalpha()]
                if letters and not all(c.isupper() for c in letters):
                    return False
            elif variant.case == "cap" and tok not in nm.PARTICLES and not surface[0].isupper():
                if not self._casual_cap_ok(text, variant):
                    return False
        return True

    def _dictionary(self, text: str, folded: Folded, ftoks: list[tuple[str, int, int]],
                    matchers: Iterable[tuple[PhraseMatcher, ...]],
                    owners: Optional[dict], *, lenient: bool = False, split: Optional[list] = None,
                    compounds: Optional[list] = None) -> list[_Cand]:
        """Candidates of known names. ``split`` holds small token windows around glued words (see
        :func:`split_windows`): "KervadecMaelle", "Kervadec2026" are matched there too. ``compounds`` collects the
        hyphen compounds that hold a strong known name ("Maela-Lorvain"): masked whole as a new name (R1, CL-07).

        ``lenient`` (the canary) ignores case, except for a one-word variant that is a dictionary word: it keeps its
        case rule ("financière" in lower case is not the company "Financière Kervalen") (R1, CL-18a)."""
        out: list[_Cand] = []
        seen: set[tuple[int, int, int]] = set()
        views = [ftoks] + list(split or [])
        for toks in views:
            for group in matchers:
                for matcher in group:
                    if not len(matcher):
                        continue
                    for i, j, ref in matcher.find_all(folded.text, toks):
                        v = ref.variant
                        if owners is not None and v.need in _FIRST_NEEDS \
                                and len(owners.get(v.tokens, ())) > 1:
                            continue  # two known people share this first name: never guess
                        if owners is not None and v.need == "dir_slot" and (
                                owners.get(v.tokens) or len(owners.get((_SLOT_OWNER,) + v.tokens, ())) > 1):
                            continue  # v0.4 (T2): a slot form yields to a first name; two owners: never guess
                        s, e = folded.span(toks[i][1], toks[j - 1][2])
                        key = (s, e, id(ref))
                        if key in seen:
                            continue
                        seen.add(key)
                        case_ok = (lenient and not (j - i == 1 and self._dict_word(v.tokens[0]))) \
                            or self._case_ok(text, folded, toks, i, j, v)
                        if v.partial and ref.kind == "ORG" and self._lang == "en" \
                                and self._domain_form_partial(ref.full_text, v.tokens):
                            continue  # v0.4 (T4): not "Pacific" alone for "Pacific Quay" of pacificquay.example
                        if ref.kind in _NAME_KINDS:
                            strong = lenient and v.need == "" and v.case != "upper" and _strong_tokens(v.tokens)
                            if not strong and _inside_compound(text, s, e):
                                # "Dupont" inside "Dupont-Morvanti" is another name: never masked in part; with a
                                # strong known name the whole compound becomes a new name (R1, CL-07)
                                if compounds is not None and case_ok and v.need == "" and v.case != "upper" \
                                        and _strong_tokens(v.tokens) and text[s:s + 1].isupper():
                                    cs, ce = _compound_span(text, s, e)
                                    if (cs, ce) != (s, e) and _name_compound(text[cs:ce], text[s:e]):
                                        compounds.append(_Cand(cs, ce, ref.kind, "dictionary", 2, need="compound"))
                                continue
                        if not case_ok:
                            continue
                        if v.initials and ref.kind == "PERSON" and toks[i][0] == "m" and j - i >= 2 \
                                and _PLAIN_TITLE_M.match(text, s):
                            continue  # "M. Ouardi" is the title + surname, not the initial of "Mehdi"
                        if not (lenient and v.need.startswith("slot:")) and not self._context_ok(text, s, e, v.need):
                            continue  # the canary (lenient) finds a registered login name or id anywhere
                        if v.need == "dotted":
                            e += 1  # "L. P." is replaced with its last dot
                        out.append(_Cand(s, e, ref.kind, "dictionary", 3 if not v.partial else 2,
                                         entity=ref.entity, pending=ref.pending, full_text=ref.full_text,
                                         need=v.need, vtoks=v.tokens))
        return out

    def _handles(self, text: str, sources: list[tuple[str, Optional[Entity], Optional[_Pending]]],
                 orgs: Iterable[tuple[Optional[Entity], Optional[_Pending]]] = ()) -> list[_Cand]:
        """Known names inside tokens that are not names: social handles that hold a known surname or company name
        ("@mkervadec"); the path segments of a URL on a public host ("wikipedia.org/wiki/Jane_Kervadec", v0.2); the
        all-capital prefix of a hyphenated code that is the acronym of exactly one known company ("BVQ-1234",
        v0.2, Max class A; ``orgs`` lists the companies of the store and of the text)."""
        out: list[_Cand] = []

        def hit(body: str):
            for tok, ent, pend in sources:
                if tok in body:
                    return ent, pend
            return None

        if sources and "@" in text:
            for m in _HANDLE_RE.finditer(text):
                body = re.sub(r"[^a-z0-9]", "", fold_word(m.group("h")))
                got = hit(body)
                if got is not None:
                    ent, pend = got
                    kind = ent.kind if ent is not None else pend.kind  # type: ignore[union-attr]
                    out.append(_Cand(m.start("h"), m.end("h"), kind, "dictionary", 2, entity=ent, pending=pend))
        if sources and ("://" in text or "www." in text):
            for m in det._URL_RE.finditer(text):
                e = det._trim_url(text, m.start(), m.end())
                url = text[m.start():e]
                if not det.url_is_public(url):
                    continue  # a private URL is masked whole by the detector
                host_end = re.search(r"^(?:[a-z][a-z0-9+.\-]*://)?[^/?#]*", url, re.IGNORECASE).end()
                for seg in re.finditer(r"[^/?#&=]+", text[m.start() + host_end:e]):
                    got = hit(re.sub(r"[^a-z0-9]", "", fold_word(seg.group())))
                    if got is None:
                        continue
                    ent, pend = got
                    kind = ent.kind if ent is not None else pend.kind  # type: ignore[union-attr]
                    s0 = m.start() + host_end + seg.start()
                    out.append(_Cand(s0, s0 + len(seg.group()), kind, "dictionary", 2, entity=ent, pending=pend))
        if orgs and "-" in text:
            acros: dict[str, dict] = {}
            no_code: set[str] = set()
            for ent, pend in orgs:
                value = ent.value if ent is not None else pend.value  # type: ignore[union-attr]
                who = ent.code if ent is not None else id(pend)
                # v0.4 (T4): a company seen only in lower case, or written as a form of an e-mail domain label (its
                # short form is the label, not initials), gives no initials in a hyphenated code; its initials still
                # count as taken, so they never become another company's alone
                quiet = self._lang == "en" and (not any(ch.isupper() for ch in value) or self._domain_form_value(value))
                for a in _code_acronyms(value):
                    acros.setdefault(a, {})[who] = (ent, pend)
                    if quiet:
                        no_code.add(a)
            singles = {a: next(iter(d.values())) for a, d in acros.items() if len(d) == 1 and a not in no_code}
            if singles:
                for m in _ACRO_CODE_RE.finditer(text):
                    got = singles.get(m.group("a"))
                    if got is not None:
                        ent, pend = got
                        out.append(_Cand(m.start("a"), m.end("a"), "ORG", "dictionary", 2, entity=ent, pending=pend,
                                         need="acro"))
        return out

    # -- id shape memory (v0.2, DETECTORS) -----------------------------------------------------------------

    def _shapes(self) -> tuple:
        """The (kind, template) id layouts of the store, when the store keeps them."""
        fn = getattr(self.store, "shapes", None)
        if not callable(fn):
            return ()
        try:
            return tuple(fn())
        except Exception:  # pragma: no cover - a broken store must not crash the masking
            return ()

    def _learn_shapes(self, text: str, raw: list, final: list) -> None:
        """A labelled EMPLOYEE_ID or ACCOUNT_NUMBER value that is masked teaches the store its layout
        ("EMP-00123" -> "EMP-ddddd"): unlabelled values of that layout are then masked in the whole dossier."""
        add = getattr(self.store, "add_shape", None)
        if not callable(add):
            return
        labelled = {(c.start, c.end): c.kind for c in raw
                    if c.kind in SHAPE_KINDS and c.detector in ("labelled_id", "table_id") and c.valid}
        if not labelled:
            return
        for c in final:
            kind = labelled.get((c.start, c.end))
            if kind is None or c.kind != kind or c.source != "regex":
                continue
            template = det.shape_template(text[c.start:c.end])
            if template:
                try:
                    add(kind, template)
                except Exception as exc:  # noqa: BLE001
                    log.debug("add_shape refused (%s)", type(exc).__name__)

    # ------------------------------------------------------------------
    # Resolution and linking
    # ------------------------------------------------------------------

    def _resolve(self, cands: list[_Cand], text: str = "") -> list[_Cand]:
        # A rule/NER name that strictly contains a dictionary name of the same kind replaces it. A company
        # found by its legal form or a trigger word ("SELARL PERNAUD & VALETTE") also replaces a known person
        # name inside it ("PERNAUD").
        if text:
            self._evidence_pass(text, cands)  # v0.3 (S2): export evidence decides the kind of a disputed surface
        dict_names = sorted((c for c in cands if c.source == "dictionary" and c.kind in _NAME_KINDS
                             and not c.dropped), key=lambda c: c.start)
        starts = [c.start for c in dict_names]
        for r in cands:
            if r.source not in ("rule", "ner") or r.kind not in _NAME_KINDS or r.dropped:
                continue
            lo = bisect.bisect_left(starts, r.start)
            hi = bisect.bisect_right(starts, r.end)
            inside = [d for d in dict_names[lo:hi] if d.end <= r.end]
            same_kind = [d for d in inside if d.kind == r.kind]
            # Equal spans of the other kind (R1, CL-21): a company found by its context (legal form, trigger, trade
            # noun, suffix) wins over a known surname alone; a person after a title wins over a company word.
            # v0.3 (S2): a person found with any local cue (greeting, sign-off, speaker, role, possessive) wins
            # too, and a strong company rule wins over a full person match when the export's evidence says so.
            other = [d for d in inside if d.kind != r.kind and d.start == r.start and d.end == r.end]
            if other and r.source == "rule":
                key = self._key_of(text, r) if text else ""
                p_score, o_score = self._kind_scores(key) if key else (0, 0)
                org_wins = r.kind == "ORG" and r.priority >= 4 and (
                    all(d.priority <= 2 and d.need != "title" for d in other)
                    or (o_score >= p_score and o_score > 0 and not any(d.need == "title" for d in other)))
                org_cue_here = any(c.kind == "ORG" and c.source == "rule" and c.priority >= 4
                                   and c.start == r.start and c.end == r.end for c in cands)
                partial_only = all(d.priority <= 2 for d in other)
                # a possessive on a word the export never shows as a first name does not make a company head
                # word a person ("Northvale's ops team")
                weak_cue_ok = r.rule not in _WEAK_PERSON_CUE_RULES or not key or self._given_evidence(key) > 0 \
                    or (not o_score and self._lexicon_first_name(key))
                person_wins = r.kind == "PERSON" and not org_cue_here and (
                    r.rule in _PERSON_ADDRESS_RULES
                    or (r.rule in _PERSON_CUE_RULES and partial_only and weak_cue_ok)
                    or (r.priority >= 4 and p_score > o_score and not any(d.need == "org" for d in other)))
                if org_wins or person_wins:
                    for d in other:
                        d.rank = _FALLBACK_RANK
                    if not any(d.start == r.start and d.end == r.end for d in same_kind):
                        r.rank = RANK["dictionary"]
            exact_same = [d for d in same_kind if d.start == r.start and d.end == r.end]
            if exact_same:
                r.dropped = True  # a known name covers exactly this span: keep the known entity
                for d in exact_same:  # v0.3 (S2): the known name keeps the weight of the rule that found it
                    d.priority = max(d.priority, r.priority)
                    # v0.3 (S7): the rules (or a signalled model find) see a stored model-only entity: the
                    # dictionary candidate carries the confirmation to ``_apply``
                    got = {"rule"} if r.source == "rule" else set(r.signals)
                    if got:
                        d.signals = tuple(sorted(set(d.signals) | got))
                continue
            inner = [d for d in same_kind if (d.end - d.start) < (r.end - r.start)]
            if r.kind == "ORG" and r.source == "rule" and r.priority >= 5:
                inner += [d for d in inside if d.kind == "PERSON" and (d.end - d.start) < (r.end - r.start)]
            if r.kind == "PERSON" and r.source == "rule" and r.priority >= 4:
                # "Jean Varenne" holds the head word of the firm "Varenne et Maudet": a person (R1, CL-21)
                inner += [d for d in inside if d.kind == "ORG" and (d.end - d.start) < (r.end - r.start)]
            if inner:
                for d in inner:
                    d.rank = _FALLBACK_RANK  # the longer name wins; the known name fills in only if it loses
                r.rank = RANK["dictionary"]
        taken = det.IntervalSet()
        out = []
        for c in sorted((c for c in cands if not c.dropped),
                        key=lambda c: (c.rank, -(c.end - c.start), -c.priority, c.start)):
            if c.end <= c.start or taken.overlaps(c.start, c.end):
                continue
            taken.add(c.start, c.end)
            out.append(c)
        out.sort(key=lambda c: c.start)
        return out

    def _link(self, kind: str, value: str, norm: str, idx: dict[tuple[str, str], Entity],
              known: _Known) -> Optional[Entity]:
        """Existing entity for a new mention: same key, or a safe variant (surname, initials, head word)."""
        ent = idx.get((kind, norm))
        if ent is not None:
            return ent
        try:
            ent = self.store.get_by_norm(kind, norm)
        except Exception:  # pragma: no cover - a broken store must not crash the masking
            ent = None
        if ent is not None:
            return ent
        if kind == "PERSON":
            firsts, surname = nm.split_person(value)
            skey = nm.surname_key(value)
            same = known.by_surname.get(skey, []) if skey else []
            if not same and firsts and len(surname) == 1 and re.fullmatch(r"[^\W\d_]\.?", surname[0]):
                # "Martine G." -> the one known Martine whose surname starts with G
                fkey = nm.first_key(value)
                ini = fold_word(surname[0]).rstrip(".")
                cands = {p.code: p for lst in known.by_surname.values() for p in lst
                         if nm.first_key(p.value) == fkey and nm.surname_key(p.value).startswith(ini)}
                return next(iter(cands.values())) if len(cands) == 1 else None
            if not same and skey and " " not in skey:
                fk = _fuzzy_key(skey, known.by_surname)  # "Pradimomt" -> the one known "Pradimont" (R1, CL-06)
                same = known.by_surname.get(fk, []) if fk else []
            if not same:
                return None
            codes = {p.code for p in same}
            if not firsts:
                return same[0] if len(codes) == 1 else None
            if all(re.fullmatch(r"(?:[^\W\d_]\.?[\-\s]?)+", f) and len(re.sub(r"[^\w]", "", f)) <= 3 for f in firsts):
                ini = nm.first_initials(value)
                comp = [p for p in same if ini and nm.first_initials(p.value).startswith(ini)]
                if len({p.code for p in comp}) == 1:
                    return comp[0]
                return None
            # Same surname and compatible given names, one list a prefix of the other ("Ombeline Talvez" and
            # "Ombeline Maud Talvez"); a compound first name is one name ("Jean" is not "Jean-Marc") (R1, CL-23).
            mine = _first_words(firsts)
            comp = [p for p in same if _compatible_firsts(mine, _first_words(nm.split_person(p.value)[0]))]
            if len({p.code for p in comp}) == 1:
                return comp[0]
            # A full name and ONE known person with the same surname and no first name ("M. Ouardi" seen
            # first, "Mehdi Ouardi" later): the same person, one code.
            if len(codes) == 1 and not nm.split_person(same[0].value)[0]:
                return same[0]
            return None
        if kind == "ORG":
            toks = norm.split()
            if not toks:
                return None
            heads = known.by_org_head.get(toks[0], [])
            comp = [o for o in heads if entity_norm("ORG", o.value).split()[:len(toks)] == toks]
            if len({o.code for o in comp}) == 1:
                return comp[0]
            # v0.2: a one-word company the model alone found ("Northvale") meets a fuller name from the rules
            # ("Northvale Analytics"): the same company, one code (the caller upgrades the value)
            alone = [o for o in heads if o.source == "ner" and len(entity_norm("ORG", o.value).split()) == 1]
            if len(toks) > 1 and len({o.code for o in alone}) == 1 and not comp:
                return alone[0]
        return None

    def _by_norm(self, kind: str, norm: str) -> Optional[Entity]:
        try:
            return self.store.get_by_norm(kind, norm)
        except Exception:  # pragma: no cover - a broken store must not crash the masking
            return None

    def _known_code(self, code: str) -> bool:
        try:
            return self.store.get_by_code(code) is not None
        except Exception:  # pragma: no cover
            return False

    def _kept_clear(self, *norms: str) -> bool:
        for n in norms:
            if n:
                try:
                    if self.store.is_kept_clear(n):
                        return True
                except Exception:  # pragma: no cover
                    continue
        return False

    def _ner_candidates(self, text: str, protected: det.IntervalSet, rules: list[_Cand] = (),
                        regex: list[_Cand] = (), lang: str = "fr", ents: list[Entity] = ()) -> list[_Cand]:
        """Candidates from the optional NER: the plugged function (v0.1 path, legacy guards) or the loaded backend
        (v0.2: ``ner.run_ner`` then ``merge.merge_ner`` with the rule and structured spans of this text)."""
        if not self.use_ner:
            return []
        if self._ner_fn is None:
            return self._backend_candidates(text, protected, rules, regex, lang, ents)
        out = []
        try:
            raw = self._ner_fn(text)
        except Exception as exc:  # NER is optional: never fail the masking because of it
            log.warning("RAGbase Shield: NER failed (%s); rules only", type(exc).__name__)
            return []
        for s, e, kind in raw:
            if kind not in _NAME_KINDS or not 0 <= s < e <= len(text):
                continue
            # trim titles, legal forms and punctuation at the edges
            seg = text[s:e]
            m = re.match(r"(?:(?:M|MM|Mme|Mlle|Me|Dr|Pr)\.?\s+|(?i:monsieur|madame|ma[îi]tre|docteur)\s+)+", seg)
            if m:
                s += m.end()
            seg = text[s:e]
            m = re.search(r"\s+(?:SAS|SASU|SARL|EURL|SA|SCI|SNC|GIE|SELARL|SELAS)\.?\s*$", seg)
            if m and m.start() > 0:
                e = s + m.start()
            while s < e and not text[s].isalnum():
                s += 1
            while e > s and not text[e - 1].isalnum():
                e -= 1
            if e - s < 2 or not text[s].isupper() or protected.overlaps(s, e):
                continue
            words = re.findall(r"[^\W\d_]+", fold_word(text[s:e]))
            stop, legal, common = nm._lists()[0], nm._lists()[1], nm._lists()[2]
            if not words or all(w in stop or w in legal for w in words):
                continue
            if len(words) == 1 and words[0] in common:
                continue
            out.append(_Cand(s, e, kind, "ner", 1))
        return out

    def _backend_candidates(self, text: str, protected: det.IntervalSet, rules: list[_Cand], regex: list[_Cand],
                            lang: str, ents: list[Entity]) -> list[_Cand]:
        self._ner_skipped = False
        backend = self._ner_backend()
        if backend is None:
            return []
        limit = nr.ner_max_chars()
        if limit and len(text) > limit:
            self._ner_skipped = True  # rules only on a very long paste: the result says so
            return []
        try:
            spans = nr.run_ner(text, backend)
        except Exception as exc:  # noqa: BLE001 - NER is optional: never fail the masking because of it
            log.warning("RAGbase Shield: NER backend %s failed (%s); rules only", backend.name, type(exc).__name__)
            return []
        if not spans:
            return []
        acronyms: set[str] = set()
        known_values: list[tuple[str, str]] = []  # v0.3 (S7): the "known" signal = the dossier's own entities
        for e in ents:
            if e.kind == "ORG":
                acronyms.update(nm.org_acronyms(e.value))
            if e.source != "ner":
                known_values.append((e.kind, e.value))
        try:
            merged = mg.merge_ner(text, spans, rules, regex, protected, lang=lang, org_acronyms=acronyms,
                                  known_values=known_values)
        except Exception as exc:  # noqa: BLE001
            log.warning("RAGbase Shield: NER merge failed (%s); rules only", type(exc).__name__)
            return []
        try:  # v0.3 (S7): the model finds the merge never masks on its own, for the review step
            self._watch = [(w.start, w.end, w.kind, w.label)
                           for w in mg.watch_ner(text, spans, rules, protected, lang=lang)]
        except Exception:  # noqa: BLE001 - the watch list is an aid, never a masking failure
            self._watch = []
        out = [_Cand(m.start, m.end, m.kind, "ner", 1, signals=tuple(getattr(m, "signals", ()) or ()))
               for m in merged]
        return self._t9_gate(text, out) if lang == "en" else out

    # ------------------------------------------------------------------
    # Words broken by a hyphen at a line end ("KERVA-\nDEC")
    # ------------------------------------------------------------------

    @staticmethod
    def _dehyphenate(text: str, join_tokens: frozenset) -> Optional[tuple[str, list[int]]]:
        """A view of ``text`` with the "-\\n" of broken words removed, and the offset map back to ``text``.

        A break is joined when the joined word is a known name word, or when the next line goes on in lower
        case ("Kerva-\\ndec"). "MOREL-\\nDUVAL" (two capitals) is a compound name and stays as it is.
        """
        if "\n" not in text or "-" not in text:
            return None
        cuts: list[tuple[int, int]] = []
        for m in _HYPHEN_BREAK_RE.finditer(text):
            left = re.search(r"[^\W\d_]+$", text[max(0, m.start() - 40):m.start()])
            right = re.match(r"[^\W\d_]+", text[m.end():m.end() + 40])
            if not left or not right:
                continue
            joined = fold_word(left.group() + right.group())
            if joined in join_tokens or right.group()[0].islower():
                cuts.append((m.start(), m.end()))
        if not cuts:
            return None
        parts: list[str] = []
        jmap: list[int] = []
        pos = 0
        for s, e in cuts:
            parts.append(text[pos:s])
            jmap.extend(range(pos, s))
            pos = e
        parts.append(text[pos:])
        jmap.extend(range(pos, len(text)))
        jmap.append(len(text))
        return "".join(parts), jmap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mask(self, text: str, *, keep_known_codes: bool = False, chat: Optional[bool] = None) -> MaskResult:
        """Mask ``text``. Returns the masked text, the replaced spans and the new entities.

        Text that already looks like a code is never analysed and is escaped (``PERSONNE_A`` ->
        ``PERSONNE-A``). With ``keep_known_codes=True`` the codes that exist in the store are kept as
        they are, so an already masked text (a model answer) can be masked again safely. ``chat`` (v0.4, T8)
        tells the masker that the text is a chat message (True) or a document (False): the lower-case name forms
        of chat are then accepted or refused on the caller's word; None (default) reads the text's own shape.
        """
        if not text:
            return MaskResult(text=text or "")
        self._chat_hint = chat if chat is None else bool(chat)
        self._watch = []  # v0.3 (S7): the review-step finds of this call
        self._seed_directory()  # v0.3 (S3): every full name of the dossier directory is known before the first mask
        ents, idx = self._index()
        lex = self._lexicon(ents)
        self._lang = self._language_of(text)
        if self._lang == "en":
            self._t5_note(text)  # v0.4 (T5): how the dossier writes its words, the acronyms it declares
        joined = self._dehyphenate(text, lex.join_tokens)
        if joined is None:
            final, known = self._detect(text, ents, idx, lex)
            return self._finish(self._apply(text, final, known, idx, keep_known_codes))
        jtext, jmap = joined
        final, known = self._detect(jtext, ents, idx, lex)
        for c in final:
            c.surface = jtext[c.start:c.end]
            c.start, c.end = jmap[c.start], jmap[c.end - 1] + 1
        return self._finish(self._apply(text, final, known, idx, keep_known_codes))

    def _finish(self, res: MaskResult) -> MaskResult:
        """Language and NER status of this mask call on the result (v0.2)."""
        res.language = self._lang
        res.watch = list(self._watch)  # v0.3 (S7): model finds for the review step, never masked on their own
        if not self.use_ner:
            res.ner = ""
        elif self._ner_fn is not None:
            res.ner = "custom"
        elif self._ner_skipped:
            res.ner = "skipped: text too long"
        else:
            res.ner = self._backend.name if self._backend is not None else ""
        return res

    def _detect(self, text: str, ents: list[Entity], idx: dict[tuple[str, str], Entity],
                lex: _Lexicon) -> tuple[list[_Cand], _Known]:
        lang = self._lang  # set by mask(): pinned, or detected on this text (v0.2, LOCALE-EN)
        self._cc = {"keys": {}, "scores": {}, "roles": {}, "toks": {}}
        folded = fold(text)
        ftoks = tokens(folded.text)
        split = split_windows(folded, ftoks)
        if lang == "en":
            protected = lc.protected_intervals_en(text, folded, ftoks, release=self._vendor_release())
        else:
            protected = nm.protected_intervals(text, folded, ftoks)
        code_like = det.IntervalSet()
        for s, e, _code in code_like_spans(text):
            code_like.add(s, e)

        cands: list[_Cand] = []
        # v0.2: the detectors of the language (``langs``) and the learnt id layouts of the dossier
        raw = det.detect_candidates(text, lang=lang, shapes=self._shapes())
        for c in det._select(c for c in raw if c.valid):
            if code_like.overlaps(c.start, c.end):
                continue
            if c.kind in ("CASE_REF", "ID_CODE") and c.detector not in ("isin", "lei") and protected.overlaps(c.start, c.end):
                continue
            cands.append(_Cand(c.start, c.end, c.kind, "regex", c.priority))
        regex_cands = list(cands)
        rule_cands: list[_Cand] = []
        # v0.3 (S1+S4): one entry point for both languages; the rules of the language then go through the span
        # refinement (legal-form span policy, company span edges, companies named by an e-mail domain of the text
        # or of the dossier directory, declared acronyms, same-surface propagation)
        for r in nm.detect_names(text, lang=lang, protected=protected, domains=self._domains()):
            if protected.overlaps(r.start, r.end) or code_like.overlaps(r.start, r.end):
                continue
            if r.rule == "domain_form" and self._dir_person_surface(text[r.start:r.end]):
                continue  # v0.4 (T4): a person of the dossier named like a domain ("Jane Doe" of janedoe.example)
            rule_cands.append(_Cand(r.start, r.end, r.kind, "rule", r.priority, rule=r.rule))
        cover = nm.legal_form_policy() == "cover"
        if lang == "en":
            rule_cands = self._t5_rules(text, rule_cands, cover)  # v0.4 (T5): readings the dossier refutes
        cands.extend(rule_cands)
        cands.extend(c for c in self._ner_candidates(text, protected, rule_cands, regex_cands, lang, ents)
                     if not code_like.overlaps(c.start, c.end))

        # Known entities first: a rule name that a known name covers exactly is that known entity.
        compounds: list[_Cand] = []
        known_dict = self._dictionary(text, folded, ftoks, [lex.strict], None, split=split, compounds=compounds)
        known_dict.extend(self._fuzzy(text, folded, ftoks, lex))
        exact = {(d.start, d.end, d.kind) for d in known_dict}

        # Names found by the rules become pending entities, so their variants are found too.
        known = _Known(ents)
        pendings: dict[tuple[str, str], _Pending] = {}
        for c in self._resolve([_copy(c) for c in cands], text):
            if c.source not in ("rule", "ner") or (c.start, c.end, c.kind) in exact:
                continue
            value = text[c.start:c.end]
            if cover and c.kind == "ORG":
                # v0.3 (S1, span policy "cover"): the span covers the legal form, the company is stored without it
                value = nm.org_form_parts(value, lang)[1] or value
            norm = entity_norm(c.kind, value)
            if not norm or self._link(c.kind, value, norm, idx, known) is not None:
                continue
            key = (c.kind, norm)
            if key not in pendings:
                pendings[key] = _Pending(c.kind, value, norm, source=c.source, signals=tuple(c.signals))
            elif c.source == "rule":
                pendings[key].source = "rule"  # the rules saw it too: no containment
            elif c.signals:  # v0.3 (S7): a second sighting of the model with a signal confirms the pending find
                pendings[key].signals = tuple(sorted(set(pendings[key].signals) | set(c.signals)))
        self._drop_acronym_pendings(ents, pendings)

        prefs: list[_Ref] = []
        for p in pendings.values():
            # diminutives come from registered people only (R1, CL-03)
            prefs.extend(r for r in self._refs_for(p.kind, p.value, pending=p) if r.variant.need != "nick")
        if prefs:
            known_dict += self._dictionary(text, folded, ftoks, [self._matchers(prefs)], None, split=split,
                                           compounds=compounds)
        if lang == "fr" and nm.lowercase_names():
            # v0.4.1: what a mention typed in lower case of a person typed in lower case may not be (a town, an
            # ordinary word, a court name, the "surname, first name" order across a comma)
            read = [(r.start, r.end) for r in rule_cands if r.kind == "PERSON"]
            known_dict = [d for d in known_dict if not self._lc_mention_is_text(text, d, read)]
        dict_cands = self._disambiguate(text, known_dict, pendings)
        dict_cands.extend(compounds)
        sources = [(tok, e, None) for tok, e in lex.handle_tokens]
        for p in pendings.values():
            sources.extend((tok, None, p) for tok in _handle_tokens(Entity(p.kind, p.value, p.norm, "")))
        orgs = [(e, None) for e in ents if e.kind == "ORG"] + [(None, p) for p in pendings.values() if p.kind == "ORG"]
        dict_cands.extend(self._handles(text, sources, orgs=orgs))
        for d in dict_cands:
            if code_like.overlaps(d.start, d.end):
                d.dropped = True
            elif protected.overlaps(d.start, d.end) and (d.entity is None or normalize(text[d.start:d.end]) != normalize(d.full_text)) \
                    and not (d.entity is not None and d.kind == "BIRTHDATE"):  # "1er mars 1979": a month is clear text
                d.dropped = True
            elif lang == "en" and d.kind in ("EMAIL", "URL") and _written_as_words(text[d.start:d.end], d.kind):
                # v0.4 (T4): the words of a known address or host written as prose ("Hexlo AI" for hexlo.ai,
                # "Intake Desk, Hexlo Co" for intake.desk@hexlo.co) are not that address: the company rules
                # read the company name there
                d.dropped = True
            elif lang == "en" and self._t5_drop(text, d):
                d.dropped = True  # v0.4 (T5): a name the dossier refutes, a common word, an undeclared acronym
        cands.extend(d for d in dict_cands if not d.dropped)
        if lang == "en":
            cands.extend(self._known_org_aliases(text, dict_cands, protected, code_like))  # v0.4 (T3)
        final = self._resolve(cands, text)
        self._learn_shapes(text, raw, final)  # v0.2 (DETECTORS)
        self._tally(text, final)  # v0.3 (S2): what the export shows each surface to be
        return final, known

    def _fuzzy(self, text: str, folded: Folded, ftoks: list[tuple[str, int, int]], lex: _Lexicon) -> list[_Cand]:
        """Capitalised words one edit away from exactly one registered name word ("Pradimomt" for "Pradimont"):
        6+ letters, not a dictionary word, same first letter; they get that entity's code (R1, CL-06)."""
        out: list[_Cand] = []
        if not lex.fuzzy:
            return out
        for tok, fs, fe in ftoks:
            if len(tok) < 6 or tok in lex.fuzzy or tok[0] not in lex.fuzzy_first:
                continue
            s, e = folded.span(fs, fe)
            if not text[s:s + 1].isupper() or not tok.isalpha() or tok in lex.join_tokens or nm._lexicon_word(tok):
                continue
            if _inside_compound(text, s, e):
                continue
            ent = _fuzzy_owner(tok, lex)
            if ent is not None:
                out.append(_Cand(s, e, ent.kind, "dictionary", 2, entity=ent, full_text=ent.value, need="fuzzy"))
        return out

    @staticmethod
    def _drop_acronym_pendings(ents: list[Entity], pendings: dict) -> None:
        """An all-capital company name found by a rule that is the acronym of exactly one other company of the
        text or of the dossier is that company ("BVQ" after "Brasseries Velanor Quimpertin") (R1, CL-01)."""
        orgs = [p for p in pendings.values() if p.kind == "ORG"]
        if not orgs:
            return
        acros: dict[str, set] = {}
        for e in ents:
            if e.kind == "ORG":
                for a in nm.org_acronyms(e.value):
                    acros.setdefault(a.lower(), set()).add(e.code)
        for p in orgs:
            for a in nm.org_acronyms(p.value):
                acros.setdefault(a.lower(), set()).add(id(p))
        for key, p in list(pendings.items()):
            owners = acros.get(p.norm, set()) if p.kind == "ORG" else set()
            if len(owners) == 1 and id(p) not in owners:
                del pendings[key]

    @staticmethod
    def _lc_mention_is_text(text: str, d: _Cand, read: list[tuple[int, int]]) -> bool:
        """v0.4.1: a dictionary candidate that is text, not a mention of a person typed in lower case (see
        :func:`names.lc_mention_is_text`). A person stored with its capitals keeps the logic of 0.4.0."""
        if d.kind != "PERSON" or not d.vtoks or not d.full_text or d.need in ("mint", "fuzzy", "compound"):
            return False
        # a first name alone in any letter case (the chat reading): the guard of the first name with its capital
        s = d.start
        guard = d.need == "casual" and bool(_PLACE_BEFORE_RE.search(text[max(0, s - 40):s])) and not (
            s > 0 and text[s - 1] in "'’" and bool(_ELISION_BEFORE_RE.search(text[max(0, s - 12):s])))
        return nm.lc_mention_is_text(text, s, d.end, d.full_text, d.vtoks, d.need, read, guard)

    def _mint(self, text: str, s: int, e: int, kind: str, pendings: dict) -> Optional[_Cand]:
        """A candidate for an ambiguous mention with its own entity: never the wrong person (R1, CL-04b, CL-20)."""
        value = " ".join(text[s:e].split())
        norm = entity_norm(kind, value)
        if not norm:
            return None
        p = pendings.get((kind, norm))
        if p is None:
            p = _Pending(kind, value, norm)
            pendings[(kind, norm)] = p
        p.mint = True
        return _Cand(s, e, kind, "dictionary", 2, pending=p, full_text=value, need="mint")

    def _disambiguate(self, text: str, cands: list[_Cand], pendings: dict) -> list[_Cand]:
        """One partial name (surname alone, first name alone, initials, acronym, head word) that several known
        entities own (R1, CL-02, CL-04b, CL-20):

        - a diminutive yields to a real first name of another person;
        - initials and acronyms are left as they are;
        - one person split in two codes (same surname, compatible first names): the oldest code;
        - otherwise the mention gets its own entity, restorable, never the code of one of the owners.

        A surname followed by a generational qualifier ("Vellin fils") is always its own person."""
        out: list[_Cand] = []
        groups: dict[tuple[int, int, str], list[_Cand]] = {}
        for d in cands:
            if d.kind in _NAME_KINDS:
                groups.setdefault((d.start, d.end, d.kind), []).append(d)
            else:
                out.append(d)
        for (s, e, kind), grp in groups.items():
            full = [d for d in grp if d.priority > 2]
            if full:
                # A whole-name variant: one owner keeps it ("Ombeline Talvez" of "Ombeline Maud Talvez"); two people
                # whose names both give this form (two middle names) are not guessed.
                if len({_owner(d) for d in full}) == 1:
                    out.extend(full)
                else:
                    m = self._mint(text, s, e, kind, pendings)
                    if m is not None:
                        out.append(m)
                continue
            if kind == "PERSON" and any(d.need in _SURNAME_NEEDS for d in grp):
                q = _QUALIFIER_RE.match(text, e)
                if q:
                    m = self._mint(text, s, q.end(), kind, pendings)
                    if m is not None:
                        out.append(m)
                    continue
            owners = {_owner(d) for d in grp}
            if len(owners) > 1:
                real = [d for d in grp if d.need != "nick"]
                if real and len(real) < len(grp):
                    grp = real
                    owners = {_owner(d) for d in grp}
            if len(owners) <= 1:
                out.extend(grp)
                continue
            if kind == "PERSON" and all(d.need in _SURNAME_NEEDS for d in grp):
                # "Mme Kervadec" among two known Kervadec: the one whose first name has that civility (v0.2,
                # DETECTORS; needs the sex column of the first-name lexicon, else no change)
                civ = nm.title_civility(text, s)
                if civ:
                    keep = [d for d in grp if nm.person_civility(_cand_value(d)) == civ]
                    if keep and len({_owner(d) for d in keep}) == 1:
                        out.extend(keep)
                        continue
            if all(d.need in _INITIAL_NEEDS for d in grp):
                continue
            ents = [d.entity for d in grp]
            if all(x is not None and x.kind == "PERSON" for x in ents) and _one_person(ents):
                oldest = min(ents, key=lambda x: _code_order(x.code))
                out.extend(d for d in grp if d.entity.code == oldest.code)
                continue
            m = self._mint(text, s, e, kind, pendings)
            if m is not None:
                if self._t9_model and self._lang == "en":
                    self._t9_note_mint(m, grp)  # v0.4 (T9): a mention that only names of the model own
                out.append(m)
        return out

    def _apply(self, text: str, final: list[_Cand], known: _Known, idx: dict[tuple[str, str], Entity],
               keep_known_codes: bool = False) -> MaskResult:
        new_entities: list[Entity] = []

        def surface_of(c: _Cand) -> str:
            return c.surface if c.surface is not None else text[c.start:c.end]

        def full_name_first(c: _Cand) -> tuple[int, int]:
            if c.kind == "PERSON":
                value = c.pending.value if c.pending else surface_of(c)
                return (0 if nm.split_person(value)[0] else 1, c.start)
            return (0, c.start)

        # Known synthetic surfaces (a model answer masked again): never a new entity; left as they are in
        # synthetic mode, replaced by their code in codes mode (v0.2, SYNTH).
        # v0.3 (synthetic leak fix): the surfaces found in this text that the store does not hold yet are reserved
        # first, so a fake drawn earlier that equals one of them is re-drawn by the Synthesizer and never taken for
        # a known fake surface that would hide the real value.
        raw_values = self._raw_synthetic_values()
        syn_hits: list[tuple[int, int, Entity, str]] = []
        if raw_values:
            # a candidate that lies inside a known fake form is that fake written back (a pseudonymised answer
            # masked again: "Bosredon" inside "M. Bosredon"): it is not reserved, so the answer stays idempotent;
            # a candidate that reaches beyond every fake form ("Hollis Kestrelford" over the fake first name
            # "Hollis") is real text and is reserved
            raw_hits = self._synthetic_hits(text, raw_values)

            def inside_a_fake(c: _Cand) -> bool:
                return any(s <= c.start and c.end <= e for s, e, _ent, _fake in raw_hits)

            def whole_fake(s: int, e: int, ent: Entity, fake: str) -> bool:
                # a whole fake value Shield gave: a person or a company of two or more distinctive words (any
                # order), or an e-mail or a web address on the reserved domain
                if ent.kind in _NAME_KINDS:
                    words = re.findall(r"[^\W\d_]{2,}", fake)
                    if sum(1 for r in words if not syn.mention_word_fixed(ent.kind, r)) < 2:
                        return False
                    return sorted(fold_word(r) for r in re.findall(r"[^\W\d_]{2,}", text[s:e])) \
                        == sorted(fold_word(r) for r in words)
                return ent.kind in ("EMAIL", "URL") and syn.SYNTHETIC_TLD in fold_word(fake) \
                    and syn._core(text[s:e]) == syn._core(fake)

            # v0.3.1 (synthetic leak, cause B; PLAN 5.1 item 2): known fake surfaces are looked for only in a model
            # answer masked again (the restore path reads them in ``synthetic_restore_items``), never in a source
            # document. An answer is a call with ``keep_known_codes=True``, or a text that writes a whole fake value
            # Shield gave (a source document never holds one: a fake never equals a real value of the dossier). In
            # a source document a short fake form ("E." of a fake "Veronica E.") would otherwise hide real text.
            answer = keep_known_codes or any(whole_fake(*h) for h in raw_hits)
            try:
                # in a source document every candidate the store does not hold yet is real text, even one that
                # reads like a fake form: it is reserved, and a fake that collides with it is re-drawn now
                self.synthesizer().reserve(surface_of(c) for c in final
                                           if c.entity is None and not c.dropped
                                           and not (answer and inside_a_fake(c)))
                if not answer:
                    self.synthesizer().redraw_collisions()
            except Exception as exc:  # noqa: BLE001 - the reservation is a guard, never a masking failure
                log.debug("synthetic reserve skipped (%s)", type(exc).__name__)
            if answer:
                def real_wins(hs: int, he: int) -> bool:
                    # defence in depth: a hit never removes a real candidate longer than itself (or reaching beyond
                    # it), nor a form of a real value of the store that the dictionary found there
                    for c in final:
                        if c.end <= hs or c.start >= he:
                            continue
                        if c.start < hs or c.end > he or (c.entity is not None and c.source == "dictionary"):
                            return True
                    return False

                syn_hits = [h for h in self._synthetic_hits(text, masking=True) if not real_wins(h[0], h[1])]
        syn_taken = det.IntervalSet()
        for hs, he, _hent, _hfake in syn_hits:
            syn_taken.add(hs, he)

        replaced: list[tuple[_Cand, Entity]] = []
        cover = nm.legal_form_policy() == "cover"
        for c in sorted(final, key=full_name_first):
            if syn_hits and syn_taken.overlaps(c.start, c.end):
                continue
            surface = surface_of(c)
            ent = c.entity
            if ent is None and c.pending is not None and c.pending.entity is not None:
                ent = c.pending.entity
            if ent is None:
                value = c.pending.value if c.pending else surface
                if c.pending is None and cover and c.kind == "ORG" and c.source in ("rule", "ner"):
                    # v0.3 (S1, span policy "cover"): the span covers the legal form, the company is stored
                    # without it (a rule span that won over its own core variant carries no pending)
                    value = nm.org_form_parts(value, self._lang)[1] or value
                norm = c.pending.norm if c.pending else entity_norm(c.kind, value)
                if not norm:
                    continue
                if c.pending is not None and c.pending.mint:
                    ent = idx.get((c.kind, norm)) or self._by_norm(c.kind, norm)
                else:
                    ent = self._link(c.kind, value, norm, idx, known)
                    if ent is not None and c.kind == "PERSON" and nm.split_person(value)[0] \
                            and not nm.split_person(ent.value)[0]:
                        ent = self._upgrade(ent, value, norm, idx, known)
                    elif ent is not None and ent.source == "ner" and c.kind in _NAME_KINDS \
                            and len(phrase_tokens(ent.value)) == 1 < len(phrase_tokens(value)):
                        # v0.2: a one-word find of the model alone meets a fuller name ("Northvale" then
                        # "Northvale Analytics"): the fuller value takes the code and the containment ends
                        ent = self._upgrade(ent, value, norm, idx, known)
                if ent is None:
                    if self._kept_clear(norm, normalize(value), normalize(surface)):
                        continue
                    # a value is stored on one line: a PDF line break inside a name is a space (R1, CL-30)
                    # v0.3 (S7): a model find that a second signal confirms is a detected entity (full variants);
                    # an unconfirmed one keeps the source "ner" and stays contained (``_refs_for``)
                    src = "rule"
                    if c.pending is not None and c.pending.source == "ner" and not c.pending.signals:
                        src = "ner"
                    ent = self._store_add(c.kind, " ".join(value.split()), norm, src)
                    if self._t9_model and self._lang == "en" and (
                            (c.pending.source if c.pending is not None else c.source) == "ner"
                            or (c.pending is not None and c.pending.mint and (c.kind, norm) in self._t9_minted)):
                        self._t9_made.add(ent.code)  # v0.4 (T9): an entity a model find alone created
                    if all(ent.code != n.code for n in new_entities) and (ent.kind, ent.norm) not in idx:
                        new_entities.append(ent)
                    idx[(ent.kind, ent.norm)] = ent
                    idx[(ent.kind, norm)] = ent
                    known.add(ent)
            if ent.source == "ner" and (c.source == "rule" or c.signals or (
                    c.pending is not None and (c.pending.source == "rule" or c.pending.signals))):
                # v0.3 (S7): the rules, or a model find with a second signal, see an entity the model alone had
                # found (a rule span, or its dictionary match that ``_resolve`` marked): it is confirmed, the
                # containment lifts (store ``confirm``)
                ent = self._confirm(ent, idx, known)
            if c.pending is not None:
                c.pending.entity = ent
            if self._kept_clear(ent.norm, normalize(surface), entity_norm(ent.kind, surface)):
                continue
            replaced.append((c, ent))

        # Escape real text that looks like a code (for CODE_RE or for UNMASK_RE), outside the replaced spans.
        synthetic = self.replacement == "synthetic"
        taken = det.IntervalSet()
        for c, _e in replaced:
            taken.add(c.start, c.end)
        written: list[tuple[int, int, str, str, str, str]] = []  # (start, end, kind, source, code, replacement)
        for c, e in replaced:
            surface, lead, tail = surface_of(c), "", ""
            if cover and c.kind == "ORG" and c.source in ("rule", "ner"):
                # v0.3 (S1, span policy "cover"): the span covers the legal form, the form stays readable around
                # the code or the pseudonym ("SOCIETE_A SAS" for "ACME SAS", span over "ACME SAS")
                lead, core, tail = nm.org_form_parts(surface, self._lang)
                if (lead or tail) and core.strip():
                    surface = core
            if synthetic:
                rep = self._render_synthetic(e, surface, c.need, c.vtoks)
            else:
                rep = _code_in_place(text, c.start + len(lead), c.end - len(tail), e.code)
            written.append((c.start, c.end, c.kind, c.source, e.code, lead + rep + tail))
        if not synthetic:
            for hs, he, hent, _hfake in syn_hits:
                if taken.overlaps(hs, he):
                    continue
                taken.add(hs, he)
                written.append((hs, he, hent.kind, "dictionary", hent.code, _code_in_place(text, hs, he, hent.code)))
        edits: list[tuple[int, int, str]] = [(s, e, rep) for s, e, _k, _src, _code, rep in written]
        for s, e, code in code_like_spans(text):
            if taken.overlaps(s, e) or (syn_hits and syn_taken.overlaps(s, e)):
                continue
            if keep_known_codes and self._known_code(code):
                continue
            edits.append((s, e, _escape_one(text[s:e])))
        edits.sort()
        parts: list[str] = []
        pos = 0
        for s, e, rep in edits:
            parts.append(text[pos:s])
            parts.append(rep)
            pos = e
        parts.append(text[pos:])
        written.sort(key=lambda w: w[0])
        spans = [Span(s, e, kind, text[s:e], source) for s, e, kind, source, _code, _rep in written]
        return MaskResult(text="".join(parts), spans=spans, new_entities=new_entities,
                          mode="synthetic" if synthetic else "codes",
                          codes=[code for _s, _e, _k, _src, code, _rep in written],
                          replacements=[rep for _s, _e, _k, _src, _code, rep in written])

    # ------------------------------------------------------------------
    # Dossier directory and export evidence (v0.3, S2+S3)
    # ------------------------------------------------------------------

    @property
    def directory(self) -> Optional[dr.Directory]:
        """The dossier directory this masker uses (the one given at construction, else the store's)."""
        return self._directory()

    def _directory(self) -> Optional[dr.Directory]:
        if self._directory_arg is not None:
            return self._directory_arg
        fn = getattr(self.store, "directory", None)
        if not callable(fn):
            return None
        try:
            data = fn()
        except Exception:  # noqa: BLE001 - a broken store: no directory
            return None
        if not data:
            return None
        if isinstance(data, dr.Directory):
            return data
        key = (id(data), len(data.get("persons") or ()) if isinstance(data, dict) else 0)
        if self._dir_cache[0] == key:
            return self._dir_cache[1]
        d = dr.Directory.from_dict(data) if isinstance(data, dict) else None
        self._dir_cache = (key, d)
        return d

    def _domains(self) -> list[str]:
        """E-mail domain labels of the dossier directory (v0.3, S3+S4): the companies a correspondent's domain
        names, handed to the span refinement of the rules. Empty without a directory."""
        d = self._directory()
        if d is None or not d.domains:
            return []
        return [label for label in d.domains if isinstance(label, str) and label]

    def _main_domain_label(self) -> str:
        """v0.4 (P1): the label of the dossier's main e-mail domain: the label with the highest address count in
        the directory, alone at the top (a tie names no main company). "" without a directory."""
        d = self._directory()
        if d is None or not d.domains:
            return ""
        counts = sorted(((int(n), label) for label, n in d.domains.items() if isinstance(label, str) and label),
                        reverse=True)
        if not counts or (len(counts) > 1 and counts[0][0] == counts[1][0]):
            return ""
        return fold_word(counts[0][1])

    def _vendor_release(self) -> frozenset[str]:
        """v0.4 (P1, policy option): the whitelist vendor phrase that the dossier's own company is, released from
        the protected intervals of its English texts, so that the e-mail domain rule masks it like any company.
        Empty when the option is off (the default), without a directory, when the main e-mail domain is not clear,
        or when its label is no vendor of the whitelist. Only that one label; every other vendor stays
        protected."""
        on = self.own_company_vendor
        if on is None:
            on = lc.own_company_vendor_setting()
        if not on:
            return frozenset()
        d = self._directory()
        key = (id(d), self._store_rev())
        if self._vendor_cache[0] == key:
            return self._vendor_cache[1]
        label = self._main_domain_label()
        out = frozenset({label}) if label and label in lc.bundle("en").vendor_phrases else frozenset()
        self._vendor_cache = (key, out)
        return out

    def _seed_directory(self) -> None:
        """Register the full names of the directory in the store, once per directory object."""
        d = self._directory()
        if d is None or self._dir_seeded is d or not callable(getattr(self.store, "add", None)):
            return
        self._dir_seeded = d
        try:
            d.seed(self.store, source="directory")
        except Exception as exc:  # noqa: BLE001 - never fail a masking on the directory
            log.debug("directory seed skipped (%s)", type(exc).__name__)

    def _evidence(self) -> dict[str, dict[str, int]]:
        """Per-surface evidence of the dossier: the store's (vault) or this masker's."""
        cached = self._cc.get("ev")
        if cached is not None:
            return cached
        ev = self._evidence_local
        fn = getattr(self.store, "evidence", None)
        if callable(fn):
            try:
                got = fn()
                if isinstance(got, dict):
                    ev = got
            except Exception:  # noqa: BLE001
                pass
        if self._cc:
            self._cc["ev"] = ev
        return ev

    def _evidence_add(self, batch: dict[str, dict[str, int]]) -> None:
        if not batch:
            return
        fn = getattr(self.store, "add_evidence", None)
        if callable(fn):
            try:
                fn(batch)
                return
            except Exception as exc:  # noqa: BLE001 - keep the evidence in memory for this masker
                log.debug("evidence not stored (%s)", type(exc).__name__)
        for key, counts in batch.items():
            slot = self._evidence_local.setdefault(key, {})
            for field_name, n in counts.items():
                slot[field_name] = slot.get(field_name, 0) + n
        self._cc.pop("ev", None)

    def _kind_scores(self, key: str) -> tuple[int, int]:
        """(person score, organisation score) of a folded surface key from the export's evidence: sightings as
        a given or family token or with a person cue, against sightings as a company (with a legal form, a
        company word, a domain) or as the head word of one; the directory adds its people and domain labels."""
        scores = self._cc.get("scores")
        if scores is not None and key in scores:
            return scores[key]
        out = self._kind_scores_impl(key)
        if scores is not None:
            scores[key] = out
        return out

    def _kind_scores_impl(self, key: str) -> tuple[int, int]:
        evidence = self._evidence()
        ev = evidence.get(key) or {}
        person = ev.get("given", 0) + ev.get("family", 0) + ev.get("person_cue", 0)
        org = ev.get("org", 0) + ev.get("org_head", 0)
        toks = key.split()
        if len(toks) >= 2:
            # a full name seen as a person: its first word as a given name and its last word as a family name
            first, last = evidence.get(toks[0]) or {}, evidence.get(toks[-1]) or {}
            person += min(first.get("given", 0), last.get("family", 0))
        d = self._directory()
        if d is not None and key:
            if " " in key:
                if any(m.role == "full" for m in d.lookup(key)):
                    person += 3
            elif d.roles(key) & {"given", "family", "nick"}:
                person += 2
            if d.is_org_label(key.replace(" ", "")):
                org += 3
        return person, org

    def _key_of(self, text: str, c: _Cand) -> str:
        return " ".join(self._toks_of(text[c.start:c.end], c.kind))

    def _toks_of(self, surface: str, kind: str) -> tuple[str, ...]:
        """Folded tokens of a candidate surface (a company surface without its legal form), cached per call."""
        cache = self._cc.get("toks")
        ck = (surface, kind)
        if cache is not None and ck in cache:
            return cache[ck]
        core = (nm.org_core(surface) or surface) if kind == "ORG" else surface
        toks = phrase_tokens(core)
        if cache is not None:
            cache[ck] = toks
        return toks

    def _person_cue_at(self, text: str, s: int, e: int) -> bool:
        """A local cue that the mention is a person: a title, a greeting, "@", a possessive, a comma after it at
        the start of a line, or a word that addresses someone before it."""
        before = text[max(0, s - 40):s]
        return (s > 0 and text[s - 1] == "@") or _possessive_at(text, e) \
            or bool(self._title_re().search(before)) or bool(self._salute_re().search(before)) \
            or (text[e:e + 1] == "," and bool(_LINE_START_RE.search(before))) \
            or bool(_PERSON_CUE_BEFORE_RE.search(before[-30:]))

    def _person_token(self, tok: str) -> bool:
        d = self._directory()
        if d is not None and d.roles(tok) & {"given", "family", "nick"}:
            return True
        ev = self._evidence().get(tok) or {}
        return ev.get("given", 0) + ev.get("family", 0) > 0

    def _family_token(self, tok: str) -> bool:
        d = self._directory()
        if d is not None and "family" in d.roles(tok):
            return True
        return (self._evidence().get(tok) or {}).get("family", 0) > 0

    def _dir_owners(self, tok: str) -> list:
        """The directory people whose first name, nickname or usual diminutive the folded word is."""
        d = self._directory()
        if d is None or not tok:
            return []
        out = [p for p in d.persons_for(tok) if {"given", "nick"} & {role for q, role in d._token_hits(tok) if q is p}]
        for p in d.persons:
            if p.given and p not in out and (tok in nm._nick_forms(p.given[0]) or tok in lx_nicknames(p.given[0])):
                out.append(p)
        return out

    def _after_own_full_name(self, before: str, tok: str) -> bool:
        """The text before a first name or nickname ends with the family name of its own person ("Marcus Trov
        Marc": a PDF row with a "goes by" cell): the capitalised phrase before it is not a longer name."""
        words = [fold_word(w) for w in dr._WORD_RE.findall(before)]
        if not words:
            return False
        for p in self._dir_owners(tok):
            fam = list(p.family)
            if fam and words[-len(fam):] == fam:
                return True
        return False

    def _org_head_evidence(self, tok: str) -> bool:
        """The export shows the word as a company or the head word of one, or it is a domain label of the
        dossier's correspondents."""
        ev = self._evidence().get(tok) or {}
        if ev.get("org", 0) + ev.get("org_head", 0) > 0:
            return True
        d = self._directory()
        return d is not None and d.is_org_label(tok)

    def _given_evidence(self, key: str) -> int:
        """How often the export shows a one-word surface as a given name (a directory first name or nickname
        counts for two)."""
        n = (self._evidence().get(key) or {}).get("given", 0)
        d = self._directory()
        if d is not None and d.roles(key) & {"given", "nick"}:
            n += 2
        return n

    def _plausible_full_name(self, value: str) -> bool:
        """A stored person value that can give token roles: two words or more, whose last word the dossier does
        not know as a first name or a nickname only ("Marcus Marc", a first name and a nickname that a rule
        joined, is two mentions, not a full name)."""
        toks = phrase_tokens(_comma_order(value))
        if len(toks) < 2:
            return False
        last = toks[-1]
        d = self._directory()
        if d is not None and "family" not in d.roles(last) and self._dir_owners(last):
            return False
        ev = self._evidence().get(last) or {}
        return not (ev.get("given", 0) > 0 and ev.get("family", 0) == 0)

    def _split_joined_mentions(self, text: str, r: _Cand, toks: tuple[str, ...]) -> Optional[str]:
        """A person rule span that runs over two mentions (v0.3, S2): "Marcus Trov Marc" (a known full name, then
        a nickname) is cut back to the full name; the directory variants then find the rest. Returns "cut" or
        None."""
        d = self._directory()
        if d is None:
            return None
        words = list(dr._WORD_RE.finditer(text[r.start:r.end]))
        if len(words) == len(toks):
            for n in range(len(toks) - 1, 1, -1):
                owners = self._dir_owners(toks[n])
                if owners and "family" not in d.roles(toks[n]) \
                        and any(m.role == "full" for m in d.lookup(" ".join(toks[:n]))):
                    r.end = r.start + words[n - 1].end()
                    return "cut"
        # a two-word span of a first name and a nickname ("Marcus Marc") stays one masked span: dropping it
        # would leave both words in clear (each alone sits inside a capitalised phrase); its stored value gives
        # no token roles (:meth:`_plausible_full_name`)
        return None

    def _known_org_aliases(self, text: str, dict_cands: list[_Cand], protected: det.IntervalSet,
                           code_like: det.IntervalSet) -> list[_Cand]:
        """v0.4 (T3, English): the short form a text declares in brackets right after a known company ("Quorin
        Ploverin (QP)") is that company too, as the rules already do for a company they find themselves
        (``names._declared_aliases``); a company mailbox name keeps its declared initials masked."""
        out: list[_Cand] = []
        lw = nm._lang_words("en")
        seen: set[tuple[int, int]] = set()
        for d in dict_cands:
            if d.kind != "ORG" or d.dropped:
                continue
            m = nm._DECLARED_RE.match(text, d.end)
            if m is None:
                continue
            alias = m.group("a").rstrip(".")
            a_s = m.start("a")
            a_e = a_s + len(alias)
            if (a_s, a_e) in seen or protected.overlaps(a_s, a_e) or code_like.overlaps(a_s, a_e):
                continue
            if not nm._abbreviates(alias, bool(m.group("q")), text[d.start:d.end], lw):
                continue
            seen.add((a_s, a_e))
            out.append(_Cand(a_s, a_e, "ORG", "rule", 5, rule="acronym"))
        return out

    def _company_word_en(self, tok: str) -> bool:
        """v0.4 (T3): a folded English word that names a kind of company or a business unit (a generic company
        word, a company suffix or head noun, a legal form, a business noun)."""
        lw = nm._lang_words("en")
        return tok in nm._CUE_NOUNS or lw.head(tok) or lw.generic(tok) or lw.form(tok) is not None \
            or lc.dictionary_word(tok, "en") and not self._lexicon_first_name(tok)

    def _known_labels(self) -> frozenset:
        """v0.4 (T4): the squeezed e-mail domain labels the dossier shows (the directory's correspondents, the
        stored e-mail addresses and hosts), cached until the store changes."""
        d = self._directory()
        key = (self._store_rev(), id(d))
        if key[0] is not None and self._labels_cache[0] == key:
            return self._labels_cache[1]
        out: set[str] = set()
        if d is not None:
            out.update(re.sub(r"[^a-z0-9]", "", k) for k in d.domains)
        for e in self.store.all():
            if e.kind == "EMAIL" and "@" in e.value:
                label = dr.domain_label(e.value)
            elif e.kind == "URL":
                host = re.sub(r"(?i)^[a-z][a-z0-9+.\-]*://", "", e.value).split("/")[0]
                label = dr.domain_label(host) if "." in host else ""
            else:
                continue
            if label:
                out.add(re.sub(r"[^a-z0-9]", "", label))
        labels = frozenset(x for x in out if x)
        self._labels_cache = (key, labels)
        return labels

    def _domain_form_partial(self, value: str, toks: tuple[str, ...]) -> bool:
        """v0.4 (T4): ``toks`` is a partial variant (a head word, leading words) made of English place names only, of
        a company written as a form of an e-mail domain label of the dossier ("Pacific Quay" for
        pacificquay.example). The domain rule finds that company again wherever it is written so; the place name
        alone ("Pacific") is the place, not that company."""
        places = lc.bundle("en").places
        if not toks or not all(t in places for t in toks):
            return False
        return self._domain_form_value(value)

    def _domain_form_value(self, value: str) -> bool:
        """v0.4 (T4): the company ``value`` (two or more words) is written as a form of an e-mail domain label of
        the dossier: its words, joined, spell the label, also with a spelled legal form for its short form."""
        words = list(phrase_tokens(nm.org_core(value) or value)) if value else []
        if len(words) < 2:
            return False
        labels = self._known_labels()
        if not labels:
            return False
        squeezed = re.sub(r"[^a-z0-9]", "", "".join(words))
        if squeezed in labels:
            return True
        short = nm._DF_SPELLED.get(words[-1])
        return short is not None and re.sub(r"[^a-z0-9]", "", "".join(words[:-1])) + short in labels

    def _dir_person_surface(self, surface: str) -> bool:
        """v0.4 (T4): ``surface`` names a person of the dossier directory: the full name of a person, or one word
        that is a first name, a family name, a nickname or a handle of a person. A form of an e-mail domain label
        with this surface stays a person reading (a personal domain, "Jane Doe" of janedoe.example)."""
        d = self._directory()
        if d is None:
            return False
        try:
            return bool(d.lookup(surface))
        except Exception:  # noqa: BLE001 - a lookup failure keeps the company reading (still masked)
            return False

    # ------------------------------------------------------------------
    # v0.4 (T5): precision core of the English rules
    # ------------------------------------------------------------------

    def _t5_note(self, text: str) -> None:
        """Read one English text into the dossier knowledge of the T5 rules, once per distinct text: the acronyms
        it declares in brackets, and, for formal text (not chat), how it writes each word in mid-sentence (in lower
        case: a common word; capitalised: a name). Chat writes names in lower case and gives no word evidence."""
        key = (len(text), hash(text))
        if key in self._t5_seen:
            return
        self._t5_seen.add(key)
        for m in _T5_DECLARED_RE.finditer(text):
            a = m.group(1)
            self._t5_declared.add(a)
            self._t5_declared.add(a.replace("&", ""))
        # the text's own shape decides here, not the caller's chat hint (v0.4 T8): a chat message written as formal
        # text still shows how the export writes its words
        if _casual_text(text) or "<@" in text:
            return
        for m in _T5_MID_RE.finditer(text):
            w = m.group(1)
            if w.islower():
                slot = 0
            elif w[0].isupper() and w[1:].islower():
                slot = 1
            else:
                continue
            counts = self._t5_mid.setdefault(w.lower().replace("’", "'"), [0, 0])
            counts[slot] += 1

    def _t5_name_token(self, low: str) -> bool:
        """A word that names someone or something of the dossier: a first name of the lexicon, a given, family or
        nick name of the directory, an e-mail domain label. Never a common word for the T5 rules."""
        b = lc.bundle("en")
        if low in b.all_first_names or low in self._t5_nicknames():
            return True
        return self._t5_dossier_name(low)

    def _t5_dossier_name(self, low: str) -> bool:
        """A given, family or nick name of the dossier directory, or an e-mail domain label of the dossier."""
        if re.sub(r"[^a-z0-9]", "", low) in self._known_labels():
            return True
        d = self._directory()
        if d is not None:
            try:
                if d.roles(low) & {"given", "family", "nick"}:
                    return True
            except Exception:  # noqa: BLE001 - a directory failure keeps the word a name (fail closed)
                return True
        return False

    @staticmethod
    @lru_cache(maxsize=1)
    def _t5_nicknames() -> frozenset:
        return frozenset(n for v in lc.bundle("en").nicknames.values() for n in v)

    def _t5_common(self, word: str) -> bool:
        """``word`` is a common word of English for the T5 rules: a word of the English vocabulary that is not
        also a surname-like word, or a word the dossier's formal text writes in lower case in mid-sentence (3 times
        or more, and at least 10 % of its mid-sentence uses). A name of the dossier or a first name never is."""
        low = word.lower().replace("’", "'").strip("'-")
        if len(low) < 2:
            return False
        cache = self._cc.setdefault("t5c", {})
        got = cache.get(low)
        if got is not None:
            return got
        out = False
        if not self._t5_name_token(low):
            b = lc.bundle("en")
            lo, ca = self._t5_mid.get(low, (0, 0))
            if (low in b.vocab and low not in b.common_words) or (lo >= 3 and lo * 10 >= lo + ca):
                out = True
            elif "-" in low:
                parts = [p for p in low.split("-") if p]
                out = len(parts) > 1 and all(p in b.stopwords or self._t5_common(p) for p in parts)
        cache[low] = out
        return out

    def _t5_value_key(self, kind: str, value: str) -> str:
        try:
            return entity_norm(kind, value)
        except Exception:  # noqa: BLE001
            return ""

    def _t5_rules(self, text: str, rule_cands: list[_Cand], cover: bool) -> list[_Cand]:
        """Record which rule found each name of this text (the dossier knowledge of the T5 rules), then drop the
        rule spans whose name the dossier refutes (:meth:`_t5_unsupported`); a person reading that opens with the
        dossier's company label becomes that company (FM1, fail closed)."""
        keyed = []
        for c in rule_cands:
            if c.kind not in _NAME_KINDS:
                keyed.append((c, "", ""))
                continue
            value = text[c.start:c.end]
            if cover and c.kind == "ORG":
                value = nm.org_form_parts(value, "en")[1] or value
            key = self._t5_value_key(c.kind, value)
            tag = c.rule
            if c.kind == "ORG" and c.rule == "legal_form" and (
                    _T5_FORM_LC_RE.match(text, c.end)
                    or (_T5_FORM_INSIDE_LC_RE.search(text[c.start:c.end]) and re.match(r"[ \t ]+[a-z]", text[c.end:c.end + 3]))):
                tag = "legal_form~lc"  # "Lantern LTD bundle": the form in capitals before a lower-case word
            if key:
                tags = self._t5_prov.setdefault((c.kind, key), set())
                tags.add(tag)
                if c.kind == "PERSON" and tag not in _T5_CUE_RULES:
                    before = text[max(0, c.start - 40):c.start]
                    if self._title_re().search(before) or self._salute_re().search(before):
                        tags.add("person_cue")  # "Dear Mabel Roster,": the greeting span lost to the longer rule span
            keyed.append((c, value, key))
        out = []
        for c, value, key in keyed:
            why = self._t5_refuted(c.kind, value, key) if key else ""
            if why == "fm1org":
                c.kind, c.rule = "ORG", "evidence"  # a person reading that opens with the dossier's company label
            elif why:
                continue
            out.append(c)
        return out

    def _t5_unsupported(self, kind: str, value: str, norm: str) -> bool:
        """v0.4 (T5, English): the dossier refutes the name ``value`` of ``kind``: every rule that found it in the
        texts this masker read is a reading the T5 rules refuse. A name no rule found in this session, a person of
        the directory and a company seen by a strong company rule keep their masks (fail closed):

        - FM1, a "First Noun" person: only the first-name rules saw it (no title, greeting, sign-off, speaker or
          role), and its last word is a common word or holds a digit ("Mabel Roster", "Mabel K9R"); when its first
          word is an e-mail domain label of the dossier ("Ostrabel Logistics" of ostrabel.example) the rule span is
          read as that company instead (``"fm1org"``: still masked);
        - FM4, a company found only before a legal form in capitals that a lower-case word follows, all of whose
          words are common words ("Lantern LTD bundle": a product or plan code);
        - FM5, a one-word company found only by a trigger word or a possessive, whose word is a common word or a
          place ("at Voucher");
        - FM10, a name of one or two words that is a place, unless a directory person or a domain label holds one
          of its words."""
        return bool(self._t5_refuted(kind, value, norm))

    def _t5_refuted(self, kind: str, value: str, norm: str) -> str:
        """The T5 rule that refutes the name (``"fm1"``, ``"fm1org"``, ``"fm4"``, ``"fm5"``, ``"fm10"``), or ``""``."""
        cache = self._cc.setdefault("t5u", {})
        ck = (kind, value, norm)
        got = cache.get(ck)
        if got is None:
            got = cache[ck] = self._t5_refuted_impl(kind, value, norm)
        return got

    def _t5_refuted_impl(self, kind: str, value: str, norm: str) -> str:
        if kind not in _NAME_KINDS or not value:
            return ""
        tags = self._t5_prov.get((kind, norm))
        if not tags:
            other = self._t5_value_key(kind, value)
            tags = self._t5_prov.get((kind, other)) if other and other != norm else None
        if not tags:
            return ""
        b = lc.bundle("en")
        words = _T5_WORD_RE.findall(value.replace(",", " "))
        if not words:
            return ""
        d = self._directory()
        if kind == "PERSON" and d is not None:
            try:
                if any(m.role == "full" for m in d.lookup(" ".join(phrase_tokens(value)))):
                    return ""
            except Exception:  # noqa: BLE001
                return ""
        lows = [w.lower().replace("’", "'") for w in words]
        strong = kind == "ORG" and self._t5_strong(tags, value)
        # FM10: a place of one or two words (a person cue beyond the bare first-name rule keeps the person)
        if len(words) <= 2 and not strong and not (tags & (_T5_CUE_RULES - {"first_alone"})) \
                and " ".join(phrase_tokens(value)) in b.places and not any(self._t5_dossier_name(w) for w in lows):
            return "fm10"
        if kind == "PERSON":
            if not (tags & _T5_FIRST_NOUN_RULES) or (tags & _T5_CUE_RULES) or len(words) < 2:
                return ""
            last, first = words[-1], lows[0]
            if re.sub(r"[^a-z0-9]", "", first) in self._known_labels():
                return "fm1org"  # "Ostrabel Logistics" of ostrabel.example: the company, still masked (fail closed)
            if not (any(ch.isdigit() for ch in last) or self._t5_common(last)):
                return ""
            p, o = self._kind_scores(" ".join(self._toks_of(value, "PERSON")))
            return "" if o > p else "fm1"  # company evidence: the T3 rule masks it as the company (fail closed)
        if strong:
            return ""
        if "legal_form~lc" in tags:
            return "fm4" if tags <= {"legal_form~lc", "same_surface"} else ""
        if tags <= _T5_WEAK_ONE_WORD_RULES and len(words) == 1:
            return "fm5" if self._t5_common(words[0]) or lows[0] in b.places else ""
        return ""

    def _t5_strong(self, tags: set, value: str) -> bool:
        """A company sighting with evidence of its own: a strong company rule, or a legal form in capitals before a
        lower-case word ("Ostrabel LLC agreed") when the name holds a word outside the common words (FM4)."""
        if tags & _T5_STRONG_ORG_RULES:
            return True
        if "legal_form~lc" not in tags:
            return False
        core = nm.org_form_parts(value, "en")[1] or value
        cw = _T5_WORD_RE.findall(core)
        return not (cw and all(self._t5_common(w) for w in cw))

    def _t5_drop(self, text: str, d: _Cand) -> bool:
        return bool(self._t5_drop_reason(text, d))

    def _t5_drop_reason(self, text: str, d: _Cand) -> str:
        """v0.4 (T5, English): a dictionary mention the T5 rules refuse: a mention of a name the dossier refutes
        (:meth:`_t5_unsupported`); FM7, a lower-case one-word mention that is a common word (a directory first
        name in chat, a company the export shows by its e-mail domain stay masked); FM9, an acronym variant of a
        company that no text of the dossier declares in brackets."""
        if d.kind not in _NAME_KINDS:
            return ""
        ent = d.entity if d.entity is not None else d.pending
        why = self._t5_refuted(d.kind, ent.value, ent.norm) if ent is not None else ""
        if why:
            return why
        surface = text[d.start:d.end]
        low = " ".join(phrase_tokens(surface))
        if len(low) >= 3 and low.count(" ") <= 1 and low in lc.bundle("en").places \
                and d.need not in ("title", "salutation", "dir", "dir_slot", "acro"):
            # FM10: a place written alone names no person and no company of its own ("Mesa" of "Mesa Timber Partners"),
            # as the English rules never read a place first name alone as a person; the directory, a person cue or
            # a company sighting with evidence of its own keep the mask
            tags = self._t5_prov.get((d.kind, ent.norm), set()) if ent is not None else set()
            if not (d.kind == "ORG" and ent is not None and self._t5_strong(tags, ent.value)) and not (tags & (_T5_CUE_RULES - {"first_alone"})) \
                    and not any(self._t5_dossier_name(w) for w in low.split()):
                return "fm10"
        if d.need not in ("dir", "dir_slot") and _T5_ONE_WORD_RE.fullmatch(surface) and self._t5_common(surface):
            tags = self._t5_prov.get((d.kind, ent.norm), ()) if ent is not None else ()
            if not ({"domain", "domain_form"} & set(tags)) and not (ent is not None and self._domain_form_value(ent.value)):
                return "fm7"
        if d.kind == "ORG" and d.need == "acro" and surface not in self._t5_declared \
                and re.sub(r"[^a-z0-9]", "", surface.lower()) not in self._known_labels():
            return "fm9"
        if self._t9_model:
            try:
                return self._t9_drop_reason(text, d)
            except Exception:  # noqa: BLE001 - a failure keeps the mention masked (fail closed)
                return ""
        return ""

    # ------------------------------------------------------------------
    # v0.4 (T9, English, NER mode only): the model-only finds the dossier does not support
    # ------------------------------------------------------------------

    def _t9_key(self, text: str, c: _Cand) -> str:
        value = text[c.start:c.end]
        if c.kind == "ORG" and nm.legal_form_policy() == "cover":
            value = nm.org_form_parts(value, "en")[1] or value
        return self._t5_value_key(c.kind, " ".join(value.split()))

    def _t9_gate(self, text: str, cands: list[_Cand]) -> list[_Cand]:
        """v0.4 (T9, English): the model finds of one text (``merge_ner`` output) that the dossier does not support
        are not masked (:meth:`_t9_refuted`). Every find is recorded first with the evidence it carries (a rule
        of the text, a context cue, an e-mail domain of the text; for a login, account or staff number an owner
        word just before it), so that one sighting with evidence keeps the value for the whole session. A failure
        keeps the model find (fail closed)."""
        keyed = []
        for c in cands:
            key = self._t9_key(text, c) if c.kind in _NAME_KINDS or c.kind in _T9_ID_KINDS else ""
            if key:
                ev = set(c.signals) & _T9_EVIDENCE_SIGNALS
                if c.kind in _T9_ID_KINDS:
                    ev = set(c.signals) & {"known"}
                    if _T9_OWNER_CUE_RE.search(text[max(0, c.start - 40):c.start].rsplit("\n", 1)[-1]):
                        ev.add("cue")
                self._t9_model.setdefault((c.kind, key), set()).update(ev)
            keyed.append((c, key))
        out = []
        for c, key in keyed:
            try:
                why = self._t9_refuted(c.kind, text[c.start:c.end], key) if key else ""
            except Exception:  # noqa: BLE001
                why = ""
            if not why:
                out.append(c)
        return out

    def _t9_supported(self, kind: str, value: str, key: str, word: str = "") -> bool:
        """The dossier supports a value the model found: a sighting of the model with evidence, a rule reading of
        the dossier that the T5 rules do not refute, an entity of the store that no model find alone created (the
        rules, the user, the directory, an earlier session), or a name word of the dossier (directory, e-mail
        domain): the value itself when it is one word, else ``word`` (one word of it written alone)."""
        if self._t9_model.get((kind, key)):
            return True
        if self._t5_prov.get((kind, key)) and not self._t5_refuted(kind, value, key):
            return True
        ent = self.store.get_by_norm(kind, key)
        if ent is not None and ent.code not in self._t9_made and getattr(ent, "source", "") != "ner":
            return True
        if kind not in _NAME_KINDS:
            return False
        words = _T5_WORD_RE.findall(word or value)
        return len(words) == 1 and self._t5_dossier_name(words[0].lower().replace("’", "'"))

    def _t9_refuted(self, kind: str, value: str, key: str) -> str:
        """The T9 rule that refuses a value the model alone found in this session, or ``""`` (a value the dossier
        supports, :meth:`_t9_supported`, is never refused):

        - G3, a login, account or staff number that is a reference code of capital-letter groups joined by a
          hyphen ("QX-TRV-7"): a document, ticket, territory or plan code; the same code with a digit read as a
          company or a person;
        - OM10, a company of one word that is a common word of English (:meth:`_t5_common`: the vocabulary, or the
          dossier's formal text writes it in lower case in mid-sentence);
        - G4, a person of one word that is a common word and no first name.

        A place of one word (G5) and a word in lower case never get here as a model find: the merge drops them
        (``merge._guard_name``); the T5 rules drop a place mention (FM10) and a lower-case common-word mention
        (FM7). A lower-case mention of a model company that is no common word stays masked (in chat it is mostly
        the real company)."""
        if (kind, key) not in self._t9_model or self._t9_supported(kind, value, key):
            return ""
        v = value.strip()
        if "-" in v and _T9_CODE_RE.fullmatch(v) and (kind in _T9_ID_KINDS or any(ch.isdigit() for ch in v)):
            return "g3"
        if kind in _T9_ID_KINDS:
            return ""
        words = _T5_WORD_RE.findall(value.replace(",", " "))
        if len(words) != 1:
            return ""
        return self._t9_word(kind, words[0])

    def _t9_word(self, kind: str, w: str) -> str:
        """OM10 / G4 on one word of a name of the model alone: a common word ("om10" for a company, "g4" for a
        person that is no first name and no nickname), or ``""``."""
        if kind == "ORG":
            return "om10" if self._t5_common(w) else ""
        low = w.lower().replace("’", "'")
        if low in lc.bundle("en").all_first_names or low in self._t5_nicknames():
            return ""
        return "g4" if self._t5_common(w) else ""

    def _t9_model_owner(self, d: _Cand) -> bool:
        """A dictionary mention that belongs to a name a model find alone brought into this session."""
        if d.entity is not None:
            return d.entity.code in self._t9_made
        p = d.pending
        return p is not None and (p.source == "ner" or (p.mint and (d.kind, p.norm) in self._t9_minted))

    def _t9_note_mint(self, m: _Cand, grp: list[_Cand]) -> None:
        """v0.4 (T9): a mention that several names own gets its own entity (``_mint``); when every owner is a name
        of the model alone, so is the new entity."""
        if m.pending is not None and m.kind in _NAME_KINDS and grp and all(self._t9_model_owner(d) for d in grp):
            self._t9_minted.add((m.kind, m.pending.norm))
            self._t9_model.setdefault((m.kind, m.pending.norm), set())

    def _t9_drop_reason(self, text: str, d: _Cand) -> str:
        """v0.4 (T9, English): a mention of a value that a model find alone brought into this session (an entity it
        created, a pending model find, or a mention that only such names own) that T9 refuses
        (:meth:`_t9_refuted`), or a one-word mention of such a name that the dossier does not support and that
        is a common word (OM10 for a company, G4 for a person that is no first name): "Ledger" for "Ledger
        Desk"; or one letter of such a company. The rules' own names, the user's and the directory's entities are
        never touched. (A login, account or staff number of the model exists as an entity only when one of its
        finds was supported, so its mentions need no check here.)"""
        if d.kind not in _NAME_KINDS or not self._t9_model_owner(d):
            return ""
        ent = d.entity if d.entity is not None else d.pending
        key = ent.norm if (d.kind, ent.norm) in self._t9_model else self._t5_value_key(d.kind, ent.value)
        if (d.kind, key) not in self._t9_model:
            return ""
        why = self._t9_refuted(d.kind, ent.value, key)
        if why:
            return why
        surface = text[d.start:d.end]
        if not _T5_WORD_RE.fullmatch(surface) or self._t9_supported(d.kind, ent.value, key, word=surface):
            return ""
        if d.kind == "ORG" and len(surface) == 1:
            return "om10w"  # one letter of a longer company name ("V" of "Z&V") names no company
        why = self._t9_word(d.kind, surface)
        return why + "w" if why else ""

    def _company_label(self, key: str) -> bool:
        """v0.4 (T3): a folded one-word surface that is the e-mail domain label of a company of the dossier (three
        or more addresses of the export at that domain)."""
        d = self._directory()
        if d is None or not key or " " in key or len(key) < 3:
            return False
        return d.domains.get(key, 0) >= 3

    def _label_cue_en(self, text: str, s: int, e: int) -> str:
        """v0.4 (T3, English): what the words around one mention of a company label say it is: "person" (a person
        cue first: "@", a title, a greeting or an addressing word before it, a vocative comma, a greeting or
        sign-off slot, a capitalised name word after it), "org" (written in capitals, a legal form or a company
        noun after it, a possessive before a company noun, a company preposition or a trigger word before it, a
        prefix with a hyphen, alone in a table cell whose neighbour cells hold no family name) or "" (no cue)."""
        before = text[max(0, s - 40):s]
        if (s > 0 and text[s - 1] == "@") or self._title_re().search(before) or self._salute_re().search(before) \
                or _PERSON_CUE_BEFORE_RE.search(before[-30:]) \
                or (text[e:e + 1] == "," and _LINE_START_RE.search(before)) or self._name_slot(text, s, e, before):
            return "person"
        lw = nm._lang_words("en")
        nxt = nm._NEXT_WORD_RE.match(text, e)
        if nxt is not None and "," not in nxt.group("gap"):
            tok = nxt.group("tok")
            low = fold_word(tok.rstrip("."))
            if low in nm._CUE_NOUNS or lw.form(tok) or (tok[:1].isupper() and (lw.head(tok) or lw.generic(tok))):
                return "org"
            if tok[:1].isupper() and (self._family_token(low) or (
                    lc.is_name_word(tok, lc.bundle("en"), allow_common=False) and not self._company_label(low))):
                return "person"  # "Northvale Quist": a longer name
        surface = text[s:e]
        letters = [c for c in surface if c.isalpha()]
        if len(letters) >= 3 and all(c.isupper() for c in letters):
            return "org"
        if nm.legal_form_after(text, e, "en") is not None:
            return "org"
        if text[e:e + 2] in ("'s", "’s"):
            after = nm._NEXT_WORD_RE.match(text, e + 2)
            return "org" if after is not None and fold_word(after.group("tok").rstrip(".")) in nm._CUE_NOUNS else ""
        prev = nm._PREV_WORD_RE.search(text, max(0, s - 24), s)
        if prev is not None and not prev.group("tok")[:1].isupper():
            low = fold_word(prev.group("tok"))
            if low in _LABEL_ORG_PREV_EN or lw.trigger(low):
                return "org"
        if _LABEL_PREFIX_BEFORE_RE.search(before):
            return "org"
        if _CELL_BEFORE_RE.search(before) and _CELL_AFTER_RE.match(text, e):
            nc = _NEXT_CELL_WORD_RE.match(text, e)
            pc = _PREV_CELL_WORD_RE.search(before)
            if (nc is not None and self._family_token(fold_word(nc.group(1)))) \
                    or (pc is not None and self._family_token(fold_word(pc.group(1)))):
                return "person"  # a first name and a family name in two cells of one row
            return "org"
        return ""

    def _label_mentions_en(self, text: str, cands: list[_Cand]) -> None:
        """v0.4 (T3, English): a one-word company of the dossier (the e-mail domain label of three or more
        addresses) that a person reading covers (a first name, a nickname or a family name of a rule or of a known
        name) is decided per mention: a person reading of the word with a company cue and no person cue around it
        (:meth:`_label_cue_en`) becomes the company, and so does a two-word person reading of the word followed
        by a company noun. A name the user keeps in clear is never relabelled (the person reading stays)."""
        d = self._directory()
        if d is None or not d.domains:
            return
        lw = nm._lang_words("en")
        have_org = {(c.start, c.end) for c in cands if c.kind == "ORG" and not c.dropped}
        extra: list[_Cand] = []
        for c in cands:
            if c.kind != "PERSON" or c.dropped or c.source not in ("rule", "dictionary"):
                continue
            if c.source == "rule" and c.rule in _PERSON_ADDRESS_RULES:
                continue
            words = list(dr._WORD_RE.finditer(text[c.start:c.end]))
            if not words or len(words) > 2 or not words[0].group()[:1].isupper():
                continue
            key = fold_word(words[0].group())
            if not self._company_label(key):
                continue
            s, e = c.start + words[0].start(), c.start + words[0].end()
            if len(words) == 2:
                second = words[1].group()
                low2 = fold_word(second)
                if c.source != "rule" or self._family_token(low2) or not (
                        low2 in nm._CUE_NOUNS or lw.head(second) or lw.generic(second)):
                    continue
                before = text[max(0, s - 40):s]
                if self._title_re().search(before) or self._salute_re().search(before) \
                        or _PERSON_CUE_BEFORE_RE.search(before[-30:]):
                    continue
                e = c.start + words[1].end()
            else:
                cue = self._label_cue_en(text, s, e)
                if cue == "":
                    # no cue around this mention: the way the export writes the word elsewhere decides (company
                    # cues at least three times and twice as often as person cues)
                    ev = self._evidence().get(key) or {}
                    lo, lp = ev.get("label_org", 0), ev.get("label_person", 0)
                    cue = "org" if lo >= 3 and lo >= 2 * lp else ""
                if cue != "org":
                    continue
            if self._kept_clear(key, entity_norm("ORG", text[s:e])):
                continue
            c.dropped = True
            if (s, e) not in have_org:
                have_org.add((s, e))
                extra.append(_Cand(s, e, "ORG", "rule", 5, rule="domain"))
        cands.extend(extra)

    def _evidence_pass(self, text: str, cands: list[_Cand]) -> None:
        """Kind decisions from the export's evidence (v0.3, S2), before the span resolution:

        - a bare first name alone (rule ``first_alone``, no cue) on a surface the export shows as a company (a
          domain label, a legal-form or company-word sighting, the head of a company name) is not a person;
        - a "First Last" rule span whose whole surface the export shows as a company more often than as a person
          ("Sterling Freight") is not a person;
        - a weak company rule (an "X & Y" pair, a trigger word, a possessive) whose words are all known person
          tokens, with no company evidence, is a list of people;
        - "Last, First" whose first part is a common word and never a family name is not a person span (the
          first name alone still is);
        - two known entities of different kinds on one span: a local cue decides; without one, the export's
          evidence does, and a one-word surface known as both needs a person cue to be the person;
        - "First Last of Company" (v0.3 integration): a strong company rule span that starts with a full person
          name the rules found, then an affiliation word, keeps the company part only ("Jane Doe of Acme LLC").
        """
        if self._lang == "en":
            self._label_mentions_en(text, cands)  # v0.4 (T3)
        persons = [p for p in cands if p.source == "rule" and p.kind == "PERSON" and p.priority >= 4 and not p.dropped]
        if persons:
            for r in cands:
                if r.source != "rule" or r.kind != "ORG" or r.priority < 4 or r.dropped:
                    continue
                for p in persons:
                    if p.start != r.start or p.end >= r.end:
                        continue
                    m = _AFFILIATION_RE.match(text, p.end)
                    if m and m.end() < r.end and text[m.end()].isupper() \
                            and self._plausible_full_name(text[p.start:p.end]):
                        r.start = m.end()
                        break
        for r in cands:
            if r.source != "rule" or r.kind not in _NAME_KINDS or r.dropped:
                continue
            surface = text[r.start:r.end]
            toks = self._toks_of(surface, r.kind)
            if not toks:
                continue
            key = " ".join(toks)
            if r.kind == "PERSON":
                if len(toks) >= 3:
                    if self._split_joined_mentions(text, r, toks) == "cut":
                        surface = text[r.start:r.end]
                        toks = self._toks_of(surface, r.kind)
                        key = " ".join(toks)
                if len(toks) == 1 and r.rule in _BARE_PERSON_RULES:
                    p, o = self._kind_scores(key)
                    if o > 0 and not self._person_cue_at(text, r.start, r.end):
                        if self._lang != "en":
                            r.dropped = True
                        elif o >= p and not self._kept_clear(entity_norm("ORG", surface)):
                            # v0.4 (T3): company evidence at least as strong as the person evidence: the company,
                            # still masked (fail closed); weaker company evidence, or a company the user keeps in
                            # clear, leaves the person reading
                            r.kind, r.rule = "ORG", "evidence"
                elif len(toks) >= 2 and r.rule == "first_name":
                    p, o = self._kind_scores(key)
                    if o > p and self._lang == "en":
                        # v0.4 (T3): company evidence beats the person reading; the span stays masked, as the
                        # company (fail closed: dropping it could leave the name in clear). A company the user
                        # keeps in clear never takes the mention: the person reading stays.
                        if not self._kept_clear(entity_norm("ORG", surface)):
                            r.kind, r.rule = "ORG", "evidence"
                    elif o > p:
                        r.dropped = True
                    elif len(toks) == 2 and self._dir_owners(toks[0]) and not self._family_token(toks[1]) \
                            and self._org_head_evidence(toks[1]):
                        # "Sarah Northvale": a first name of the dossier, then the head word of a company of
                        # the dossier (a table cell boundary): the person mention is the first name alone
                        words = list(dr._WORD_RE.finditer(surface))
                        if len(words) == 2:
                            r.end = r.start + words[0].end()
                elif r.rule == "surname_first" and "," in surface:
                    fam = phrase_tokens(surface.split(",", 1)[0])
                    giv = phrase_tokens(surface.split(",", 1)[1])
                    if fam and len(fam) == 1 and giv and not self._family_token(fam[0]):
                        if self._common_word(fam[0]):
                            r.dropped = True  # "Perfect, Devon": an interjection, then a first name
                        else:
                            d = self._directory()
                            owners = []
                            if d is not None:
                                owners = [p for p in d.persons_for(giv[0]) if {"given", "nick"} & set(
                                    role for q, role in d._token_hits(giv[0]) if q is p)]
                                owners += [p for p in d.persons if p.given and p not in owners
                                           and giv[0] in nm._nick_forms(p.given[0])]
                            if owners and all(fam[0] not in p.family for p in owners) \
                                    and not self._lexicon_first_name(fam[0]):
                                r.dropped = True  # "Agreed, Devon": the dossier knows Devon with another family name
            elif r.rule in _WEAK_ORG_RULES:
                words = [t for t in toks if t not in nm._ORG_CONNECT and t not in ("and", "of", "the")
                         and t not in nm.LEGAL_FORMS]
                if words and all(self._person_token(t) for t in words) and self._kind_scores(key)[1] == 0:
                    r.dropped = True
        groups: dict[tuple[int, int], list[_Cand]] = {}
        for d in cands:
            if d.source == "dictionary" and d.kind in _NAME_KINDS and not d.dropped:
                groups.setdefault((d.start, d.end), []).append(d)
        for (s, e), grp in groups.items():
            persons = [d for d in grp if d.kind == "PERSON"]
            orgs = [d for d in grp if d.kind == "ORG"]
            if not persons or not orgs:
                continue
            person_cue = any(d.need in ("title", "salutation") for d in persons) or self._person_cue_at(text, s, e)
            org_cue = any(d.need == "org" for d in orgs)
            if person_cue and not org_cue:
                losers = orgs
            elif org_cue and not person_cue:
                losers = persons
            else:
                key = " ".join(phrase_tokens(text[s:e]))
                p, o = self._kind_scores(key)
                if o > 0 and (" " not in key or o >= p):
                    losers = persons
                elif p > o:
                    losers = orgs
                else:
                    losers = []
            for d in losers:
                d.rank = _FALLBACK_RANK

    def _tally(self, text: str, final: list[_Cand]) -> None:
        """Count what this text shows each surface to be: the given and family tokens of the full names, the
        one-word person mentions with a cue, the companies found with a cue (and their head word), the login
        names found in their slot. The counts feed :meth:`_kind_scores` on the next texts of the dossier."""
        batch: dict[str, dict[str, int]] = {}

        def add(key: str, field_name: str) -> None:
            if key:
                slot = batch.setdefault(key, {})
                slot[field_name] = slot.get(field_name, 0) + 1

        strong_org = getattr(nm, "STRONG_ORG_RULES", _STRONG_ORG_RULES_FALLBACK) | _STRONG_ORG_RULES_FALLBACK
        directory = self._directory()
        for c in final:
            if c.dropped or c.end <= c.start:
                continue
            surface = text[c.start:c.end]
            if c.kind == "PERSON":
                value = c.entity.value if c.entity is not None else (c.pending.value if c.pending else surface)
                toks = self._toks_of(surface, "PERSON")
                if len(toks) >= 2:
                    roles = self._roles_of(surface, value, directory)
                    given = [w for w, role in roles if role in ("given", "nick")]
                    known = directory is not None and directory.by_norm(entity_norm("PERSON", value)) is not None
                    # evidence only from a name whose given part is a first name of the lexicon, or a person
                    # of the directory: a rule mistake on a company never counts as person evidence; neither
                    # does a rule span over two mentions ("Marcus Marc")
                    if (known or any(self._lexicon_first_name(fold_word(w)) for w in given)) \
                            and self._plausible_full_name(value):
                        for tok, role in roles:
                            if role in ("given", "family", "nick"):
                                for part in phrase_tokens(tok):
                                    add(part, "given" if role == "nick" else role)
                elif toks:
                    cue = (c.source == "rule" and c.rule in _PERSON_CUE_RULES) or c.need in ("title", "salutation") \
                        or self._person_cue_at(text, c.start, c.end)
                    if cue:
                        add(toks[0], "person_cue")
            elif c.kind == "ORG":
                toks = self._toks_of(surface, "ORG")
                if not toks:
                    continue
                strong = (c.source == "rule" and c.rule in strong_org) or \
                    (c.source == "dictionary" and len(toks) >= 2 and not c.need and c.priority >= 3) or \
                    c.need == "org"
                if strong:
                    add(" ".join(toks), "org")
                    if len(toks) >= 2 and not self._generic_head(toks[0]) and not (
                            self._lang == "en" and directory is not None and len(toks) == 2
                            and directory.roles(toks[0]) & {"given", "nick"} and not self._company_word_en(toks[1])):
                        # v0.4 (T3): a first name of the dossier's people that opens a company named like a person
                        # ("Ilvane Corsetti & Co.": a first name and a name word) is not the company's head word: the
                        # export calls that company by its other word, and the first name alone stays a person's.
                        # A head before a company word keeps the old rule ("Ilvane Logistics")
                        add(toks[0], "org_head")
            elif c.kind == "USERNAME":
                add(" ".join(phrase_tokens(surface)), "username")
            if self._lang == "en" and c.kind in _NAME_KINDS and directory is not None and directory.domains:
                # v0.4 (T3): how the export writes a one-word company of the dossier that is also a person's name:
                # with a company cue or with a person cue (:meth:`_label_cue_en`)
                words = list(dr._WORD_RE.finditer(surface))
                if len(words) == 1 and surface[:1].isupper():
                    key = fold_word(words[0].group())
                    if self._company_label(key):
                        s0 = c.start + words[0].start()
                        cue = self._label_cue_en(text, s0, s0 + len(words[0].group()))
                        if cue:
                            add(key, "label_" + cue)
        self._evidence_add(batch)

    def _roles_of(self, surface: str, value: str, directory: Optional[dr.Directory]) -> list[tuple[str, str]]:
        """Token roles of a person mention, from the directory when it knows the person, else from the stored
        value (``names.surface_roles``, S1) and the comma order."""
        cache = self._cc.get("roles")
        ck = (surface, value)
        if cache is not None and ck in cache:
            return cache[ck]
        out = self._roles_of_impl(surface, value, directory)
        if cache is not None:
            cache[ck] = out
        return out

    def _roles_of_impl(self, surface: str, value: str, directory: Optional[dr.Directory]) -> list[tuple[str, str]]:
        person = None
        if directory is not None:
            norm = entity_norm("PERSON", value)
            person = directory.by_norm(norm)
            if person is None:
                hits = {m.person.norm: m.person for m in directory.lookup(value) if m.role == "full"}
                if len(hits) == 1:
                    person = next(iter(hits.values()))
        if person is None:
            person = _person_from_value(value)
        roles = dr.Directory.person_roles(directory if directory is not None else dr.Directory(), surface, person)
        if "," not in surface:
            fn = getattr(nm, "surface_roles", None)
            if callable(fn):
                try:  # a quoted nickname inside the mention (S1)
                    quoted = {surface[s0:e0]: role for s0, e0, role in fn(surface, value) if role == "nick"}
                    roles = [(w, "nick" if w in quoted else r) for w, r in roles]
                except Exception:  # noqa: BLE001
                    pass
        return roles

    def person_roles(self, surface: str, entity: Optional[Entity] = None) -> list[tuple[str, str]]:
        """``[(word, role)]`` of one written mention of a person, in text order: "given", "family", "nick",
        "initial", "particle", "title" or "other" (v0.3, S2+S3). The roles come from the person the mention
        refers to (``entity``, else the store, else the dossier directory), from the "Last, First" order of the
        mention and from the export's evidence; never from the position alone when something better is known.
        One word that nothing links is "given" unless the dossier knows it only as a family name."""
        words = dr._WORD_RE.findall(surface)
        if not words:
            return []
        directory = self._directory()
        ent = entity
        if ent is None:
            norm = entity_norm("PERSON", surface)
            ent = self._by_norm("PERSON", norm) if norm else None
            if ent is None and norm:
                ents, idx = self._index()
                rev = self._store_rev()
                if rev is None or self._known_cache[0] != rev:
                    self._known_cache = (rev, _Known(ents))
                try:
                    ent = self._link("PERSON", surface, norm, idx, self._known_cache[1])
                except Exception:  # noqa: BLE001
                    ent = None
        if ent is not None and ent.kind == "PERSON":
            roles = self._roles_of(surface, ent.value, directory)
        else:
            person = None
            if directory is not None:
                hits = {m.person.norm: m.person for m in directory.lookup(surface)}
                if len(hits) == 1:
                    person = next(iter(hits.values()))
            roles = dr.Directory.person_roles(directory if directory is not None else dr.Directory(), surface, person)
        if len(words) == 1 and roles and roles[0][1] in ("family", "given"):
            tok = fold_word(words[0])
            known_roles: set[str] = set()
            if directory is not None:
                # the directory (headers, Slack users, authors) is the best witness of a word's role
                known_roles = set(directory.roles(tok)) & {"given", "family", "nick"}
                if not known_roles and self._dir_owners(tok):
                    known_roles = {"nick"}  # the usual diminutive of a directory first name
            if not known_roles and ent is not None and ent.kind == "PERSON" and self._plausible_full_name(ent.value):
                # a name stored alone ("Devon") says nothing about its role: only a full name does
                known_roles = {r for _w, r in self._roles_of(ent.value, ent.value, directory)
                               if fold_word(_w) == tok}
            if not known_roles:
                ev = self._evidence().get(tok) or {}
                if ev.get("given", 0) or ev.get("family", 0):
                    known_roles = {"given"} if ev.get("given", 0) >= ev.get("family", 0) else {"family"}
            if known_roles:
                role = "family" if known_roles == {"family"} else ("nick" if known_roles == {"nick"} else "given")
            elif self._lexicon_first_name(tok) or _any_first_name(tok):
                role = "given"
            else:
                role = roles[0][1] if ent is not None and len(phrase_tokens(ent.value)) == 1 and roles[0][1] == "family" \
                    and not _any_first_name(tok) else "given"
            roles = [(words[0], role)]
        return roles

    def name_roles(self) -> dict[str, set[str]]:
        """Folded name token -> roles ("given", "family", "nick") over the store's people and the directory."""
        out: dict[str, set[str]] = {}
        d = self._directory()
        if d is not None:
            for p in d.persons:
                for t in p.given:
                    out.setdefault(t, set()).add("given")
                for t in p.family:
                    out.setdefault(t, set()).add("family")
                for t in p.nicknames:
                    out.setdefault(t, set()).add("nick")
        for e in self.store.all():
            if e.kind != "PERSON" or not self._plausible_full_name(e.value):
                continue  # a name stored alone has no known role; neither has a rule span over two mentions
            for w, role in self._roles_of(e.value, e.value, d):
                if role in ("given", "family", "nick"):
                    for part in phrase_tokens(w):
                        out.setdefault(part, set()).add(role)
        return out

    # ------------------------------------------------------------------
    # Synthetic replacement mode (v0.2, SYNTH)
    # ------------------------------------------------------------------

    def synthesizer(self) -> syn.Synthesizer:
        """The :class:`~ragbase_shield.engine.synthetic.Synthesizer` of this masker (built on first use).

        The seed comes from the store (``synthetic_seed()``: the vault derives it from the dossier key, a
        MemoryStore draws it once), else it is random for this masker."""
        if self._synthesizer is None:
            seed = None
            getter = getattr(self.store, "synthetic_seed", None)
            if callable(getter):
                try:
                    seed = getter()
                except Exception:  # noqa: BLE001 - a store without a usable seed: a local one
                    seed = None
            if not seed:
                seed = os.urandom(32)
            self._synthesizer = syn.Synthesizer(seed, "en" if self.locale == "en" else "fr", self.store,
                                                cache=self._synth_cache)
        return self._synthesizer

    def _raw_synthetic_values(self) -> dict[str, tuple[Entity, str]]:
        """code -> (entity, fake value) as the store rows and the cache hold them (no collision check)."""
        out: dict[str, tuple[Entity, str]] = {}
        getter = getattr(self.store, "synthetic", None)
        try:
            ents = list(self.store.all())
        except Exception:  # noqa: BLE001
            return out
        for e in ents:
            fake = None
            if callable(getter):
                try:
                    fake = getter(e.code)
                except Exception:  # noqa: BLE001
                    fake = None
            if not fake:
                fake = self._synth_cache.get(e.code)
            if fake:
                out[e.code] = (e, fake)
        return out

    def _synthetic_values(self) -> dict[str, tuple[Entity, str]]:
        """code -> (entity, fake value) for every entity of the store that has a synthetic value.

        v0.3 (synthetic leak fix): the known fake surfaces come from the Synthesizer (``values()``), which first
        re-draws every fake that collides with the store as it is now (a real value registered after the draw, a
        reserved surface of the text being masked). A raw store row could hold a stale fake equal to a real value,
        and that surface would hide the real value in a source text. A Synthesizer failure gives no known fake
        surface at all (fail closed: nothing real is taken for a fake)."""
        raw = self._raw_synthetic_values()
        if not raw:
            return {}
        try:
            fakes = self.synthesizer().values()
        except Exception as exc:  # noqa: BLE001
            log.warning("RAGbase Shield: synthetic values unavailable (%s); no known fake surface", type(exc).__name__)
            return {}
        out: dict[str, tuple[Entity, str]] = {}
        for code, fake in fakes.items():
            pair = raw.get(code)
            if pair is None:
                try:
                    ent = self.store.get_by_code(code)
                except Exception:  # noqa: BLE001
                    ent = None
                if ent is None:
                    continue
                pair = (ent, fake)
            if fake:
                out[code] = (pair[0], fake)
        return out

    def _synthetic_lexicon(self, values: dict[str, tuple[Entity, str]]) -> tuple:
        """(matchers, owners) of the synthetic values, cached until the store or the cache changes."""
        key = (self._store_rev(), tuple(sorted(values)), tuple(v for _e, v in values.values()))
        if self._syn_lex is not None and self._syn_key == key:
            return self._syn_lex
        refs: list[_Ref] = []
        for e, fake in values.values():
            refs.extend(self._refs_for(e.kind, fake, entity=e))
        lex = (self._matchers(refs), self._owners(refs))
        self._syn_lex, self._syn_key = lex, key
        return lex

    def _synthetic_hits(self, text: str, values: Optional[dict[str, tuple[Entity, str]]] = None, *,
                        masking: bool = False) -> list[tuple[int, int, Entity, str]]:
        """``(start, end, entity, fake)`` of the synthetic mentions in ``text`` (full and partial forms, any case),
        longest first, without overlaps. Empty when the store holds no synthetic value. ``values``: the fake values
        to look for (default: the collision-checked values of :meth:`_synthetic_values`). ``masking`` (v0.3.1, the
        masking path; restore keeps every form): a one-letter or initial-only fake form (initials, an acronym) is
        never a hit."""
        if not text:
            return []
        if values is None:
            values = self._synthetic_values()
        if not values:
            return []
        matchers, owners = self._synthetic_lexicon(values)
        folded = fold(text)
        ftoks = tokens(folded.text)
        cands: list[tuple[int, int, Entity, str, nm.Variant]] = []
        for matcher in matchers:
            if not len(matcher):
                continue
            for i, j, ref in matcher.find_all(folded.text, ftoks):
                v = ref.variant
                if v.need == "nick" or (v.need in _FIRST_NEEDS and len(owners.get(v.tokens, ())) > 1):
                    continue
                if masking and (v.need in ("dotted", "inits", "ini_ctx", "acro")
                                or all(len(t) <= 1 for t in v.tokens)):
                    continue  # v0.3.1 (cause B): "E." of a fake "Veronica E." never hides a real "E. Name"
                s, e = folded.span(ftoks[i][1], ftoks[j - 1][2])
                if v.need == "dotted":
                    if e >= len(text) or text[e] != "." or not _DOTTED_RE.fullmatch(text[s:e + 1]):
                        continue
                    e += 1
                if (s > 0 and (text[s - 1].isalnum() or text[s - 1] in "-'’" and text[s - 1] != "'")) \
                        or (e < len(text) and text[e].isalnum()):
                    continue
                if v.initials and not self._case_ok(text, folded, ftoks, i, j, v):
                    continue
                if masking and len(re.sub(r"[\W_]", "", text[s:e])) <= 1:
                    continue
                cands.append((s, e, ref.entity, ref.full_text, v))
        cands.sort(key=lambda c: (-(c[1] - c[0]), c[4].partial, c[0]))
        taken = det.IntervalSet()
        out: list[tuple[int, int, Entity, str]] = []
        for s, e, ent, fake, _v in cands:
            if taken.overlaps(s, e):
                continue
            taken.add(s, e)
            out.append((s, e, ent, fake))
        out.sort(key=lambda c: c[0])
        return out

    def _render_synthetic(self, ent: Entity, surface: str, need: str, vtoks: tuple) -> str:
        """The synthetic text for one mention. Fail closed: when the rendering still holds a distinctive word or
        the digits of the real value, the code is written instead."""
        synth = self.synthesizer()
        if getattr(synth, "nickname_source", False) is None:
            # v0.3.1 (cause A): the renderer and the collision rules read the nicknames of the dossier directory,
            # as the masker's resolution does (``_dir_refs``): "Reggie" of a directory "Reginald ..." is that person
            found: dict[tuple[int, str], tuple[str, ...]] = {}

            def declared(value: str) -> tuple[str, ...]:
                d = self._directory()
                if d is None or not value:
                    return ()
                key = (id(d), value)
                if key not in found:
                    try:
                        person = _dir_person_for(d, Entity("PERSON", value, entity_norm("PERSON", value), ""))
                    except Exception:  # noqa: BLE001 - a broken directory: the tables only
                        person = None
                    found[key] = (tuple(person.given) + tuple(person.nicknames)) if person is not None else ()
                return found[key]

            synth.use_nicknames(declared)
        try:
            current = self.store.get_by_code(ent.code)  # the entity may have been upgraded during this call
            if current is not None and current.kind == ent.kind:
                ent = current
            fake = synth.value_for(ent)
            rep = synth.render(ent, fake, surface, need, vtoks)
        except Exception as exc:  # noqa: BLE001 - never leave the real text: the code is the safe fallback
            log.warning("RAGbase Shield: synthetic rendering failed (%s); code written", type(exc).__name__)
            return ent.code
        if not rep or not rep.strip() or _synthetic_leaks(ent, rep, surface):
            return ent.code
        return rep

    def synthetic_restore_items(self, text: str) -> list[tuple[int, int, Entity, str]]:
        """``(start, end, entity, real text)`` for every synthetic mention of ``text`` (local restore only)."""
        hits = self._synthetic_hits(text)
        if not hits:
            return []
        synth = self.synthesizer()
        out = []
        for s, e, ent, fake in hits:
            try:
                real = synth.restore(ent, fake, text[s:e])
            except Exception:  # noqa: BLE001 - keep the stored value
                real = ent.value
            out.append((s, e, ent, real if real and real.strip() else ent.value))
        return out

    def synthetic_altered(self, text: str) -> int:
        """Capitalised words one edit away from a synthetic name word (6+ letters) that no synthetic mention
        covers: fake names a model altered, which restore cannot map back (reported, never guessed)."""
        values = self._synthetic_values()
        if not values or not text:
            return 0
        fake_words: dict[str, str] = {}
        real_words: set[str] = set()
        for e, fake in values.values():
            if e.kind not in _NAME_KINDS:
                continue
            for w in phrase_tokens(fake):
                if len(w) >= 6 and w.isalpha() and not nm._lexicon_word(w):
                    fake_words[w] = e.code
            real_words.update(phrase_tokens(e.value))
        if not fake_words:
            return 0
        covered = det.IntervalSet()
        for s, e, _ent, _fake in self._synthetic_hits(text):
            covered.add(s, e)
        folded = fold(text)
        n = 0
        for tok, fs, fe in tokens(folded.text):
            if len(tok) < 6 or tok in fake_words or tok in real_words or not tok.isalpha() or nm._lexicon_word(tok):
                continue
            s, e = folded.span(fs, fe)
            if covered.overlaps(s, e) or not text[s:s + 1].isupper():
                continue
            if sum(1 for w in fake_words if abs(len(w) - len(tok)) <= 1 and _dl1(w, tok)) == 1:
                n += 1
        return n

    def _restore_synthetic(self, text: str) -> tuple[str, int]:
        items = self.synthetic_restore_items(text)
        if not items:
            return text, 0
        parts: list[str] = []
        pos = 0
        for s, e, _ent, real in items:
            parts.append(text[pos:s])
            parts.append(real)
            pos = e
        parts.append(text[pos:])
        return "".join(parts), len(items)

    def _upgrade(self, ent: Entity, value: str, norm: str, idx: dict, known: _Known) -> Entity:
        """A surname-only entity ("Ouardi") meets its full name ("Mehdi Ouardi"): keep the code, store the
        fuller value when the store allows it (restore then gives the full name)."""
        upgrade = getattr(self.store, "upgrade", None)
        if not callable(upgrade):
            return ent
        try:
            new = upgrade(ent.code, value, norm)
        except Exception as exc:  # noqa: BLE001 - keep the short form
            log.debug("upgrade refused (%s)", type(exc).__name__)
            return ent
        if new is None:
            return ent
        idx[(new.kind, new.norm)] = new
        idx[(new.kind, norm)] = new
        known.replace(ent, new)
        return new

    def _confirm(self, ent: Entity, idx: dict, known: _Known) -> Entity:
        """v0.3 (S7): the store marks a model-only entity as detected (``confirm(code)``, optional method); a store
        without it keeps the entity as it is (the containment then lifts only with a fuller sighting)."""
        confirm = getattr(self.store, "confirm", None)
        if not callable(confirm):
            return ent
        try:
            new = confirm(ent.code)
        except Exception as exc:  # noqa: BLE001 - keep the entity as it is
            log.debug("confirm refused (%s)", type(exc).__name__)
            return ent
        if new is None or new is ent:
            return ent
        idx[(new.kind, new.norm)] = new
        known.replace(ent, new)
        return new

    # ------------------------------------------------------------------

    def _store_add(self, kind: str, value: str, norm: str, source: str) -> Entity:
        """``store.add`` with the entity source when the store takes it (v0.2: "ner" for a model-only find)."""
        if source == "ner":
            try:
                return self.store.add(kind, value, norm, source="ner")
            except TypeError:
                pass  # a store without the keyword (v0.1 custom store)
        return self.store.add(kind, value, norm)

    def _restore_value(self, ent: Entity, text: str, s: int, e: int) -> str:
        value = ent.value
        words = value.split()
        if len(words) > 1 and ent.kind == "ORG":
            nxt = re.match(r"\s+([^\s,.;:()]+)", text[e:e + 24])
            if nxt and fold_word(nxt.group(1)).replace(".", "") == fold_word(words[-1]).replace(".", "") \
                    and fold_word(words[-1]).replace(".", "") in nm.LEGAL_FORMS:
                value = value[: value.rfind(words[-1])].rstrip()
            prv = re.search(r"([^\s,.;:()]+)\s+$", text[max(0, s - 24):s])
            if prv and fold_word(prv.group(1)).replace(".", "") == fold_word(words[0]).replace(".", "") \
                    and fold_word(words[0]).replace(".", "") in nm.LEGAL_FORMS:
                value = value[len(words[0]):].lstrip()
        if len(words) > 1 and ent.kind == "PERSON" and nm.is_title_word(words[0]):
            before = text[max(0, s - 24):s]
            if _TITLE_BEFORE_RE.search(before) or lc.title_before_re("en").search(before):
                value = value[len(words[0]):].lstrip()
        return value

    def unmask_report(self, text: str) -> tuple[str, list[str]]:
        """Restore codes known to the store; also return the codes that are not in the store.

        Codes a model wrote with a Markdown escape ("PERSONNE\\\\_A"), a hidden character or a wide underscore
        are read as codes. Near misses that stay as they are ("personne_a") are listed with the unknown codes.
        """
        if not text:
            return text or "", []
        text = canonical_codes(text)
        text, _restored = self._restore_synthetic(text)  # synthetic surfaces first (v0.2), then the codes
        unknown: list[str] = []
        parts: list[str] = []
        pos = 0
        exact: set[tuple[int, int]] = set()
        for m in UNMASK_RE.finditer(text):
            exact.add((m.start(), m.end()))
            code = m.group()
            try:
                ent = self.store.get_by_code(code)
            except Exception:  # pragma: no cover
                ent = None
            if ent is None:
                if code not in unknown:
                    unknown.append(code)
                continue
            parts.append(text[pos:m.start()])
            parts.append(self._restore_value(ent, text, m.start(), m.end()))
            pos = m.end()
        parts.append(text[pos:])
        for m in _NEAR_CODE_RE.finditer(text):
            if (m.start(), m.end()) in exact or m.group() in unknown:
                continue
            if m.group("p").isupper() and (m.group("s").isupper() or m.group("s").isdigit()):
                continue  # an exact code (glued forms are handled above)
            unknown.append(m.group())
        return "".join(parts), unknown

    def unmask(self, text: str) -> str:
        """Replace codes known to the store by their real values (local use only)."""
        return self.unmask_report(text)[0]

    # ------------------------------------------------------------------

    def scan(self, text: str, *, prepared: Optional[ScanPrep] = None, chat: Optional[bool] = None) -> list[Span]:
        """Canary: spans of known values (variants, line breaks, small OCR errors, glued words) and of valid
        structured identifiers still present in ``text``. Known values have source "dictionary",
        structured identifiers "regex". ``prepared`` (from :func:`prepare_scan`) shares the folding and the
        structured detectors between the maskers of several dossiers. ``chat`` as in :meth:`mask` (v0.4, T8):
        the hint holds for this call only."""
        if not text:
            return []
        self._chat_hint = chat if chat is None else bool(chat)
        ents = list(self.store.all())
        lex = self._lexicon(ents) if ents else None
        self._lang = self._language_of(text)
        joined = self._dehyphenate(text, lex.join_tokens) if lex is not None else None
        if joined is not None:
            jtext, jmap = joined
            out = []
            for sp in self._scan(jtext, ents, lex, prepare_scan(jtext)):
                s, e = jmap[sp.start], jmap[sp.end - 1] + 1
                out.append(Span(s, e, sp.kind, text[s:e], sp.source))
            return out
        if prepared is None or prepared.text is not text and prepared.text != text:
            prepared = prepare_scan(text)
        return self._scan(text, ents, lex, prepared)

    def _scan(self, text: str, ents: list[Entity], lex: Optional[_Lexicon], prep: ScanPrep) -> list[Span]:
        out: list[Span] = []
        seen: set[tuple[int, int]] = set()
        for s in prep.structured:
            out.append(s)
            seen.add((s.start, s.end))
        if lex is not None:
            cands = self._dictionary(text, prep.folded, prep.ftoks, [lex.lenient], lex.owners, lenient=True,
                                     split=prep.split)
            cands = [c for c in cands if c.need != "nick"]  # a diminutive is a guess: masked, never a canary hit
            cands.extend(self._handles(text, [(tok, e, None) for tok, e in lex.handle_tokens]))
            for c in cands:
                if (c.start, c.end) in seen:
                    continue
                seen.add((c.start, c.end))
                out.append(Span(c.start, c.end, c.kind, text[c.start:c.end], "dictionary"))
            for kind, pat in lex.bodies:
                for m in pat.finditer(text):
                    if m.end() - m.start() < 4 or (m.start(), m.end()) in seen:
                        continue
                    seen.add((m.start(), m.end()))
                    out.append(Span(m.start(), m.end(), kind, m.group(), "dictionary"))
        # Drop spans fully inside a longer span of the same report.
        out.sort(key=lambda s: (s.start, -(s.end - s.start)))
        kept: list[Span] = []
        for s in out:
            if kept and s.start >= kept[-1].start and s.end <= kept[-1].end and (s.start, s.end) != (kept[-1].start, kept[-1].end):
                continue
            kept.append(s)
        return kept


def _synthetic_leaks(ent: Entity, rep: str, surface: str = "") -> bool:
    """True when a synthetic rendering still shows the real value: a distinctive word of a name (4+ letters, not
    a dictionary, generic or legal word) or the alphanumeric body of an identifier (6+ characters).

    v0.3.1 (cause A): for a name, a word of two or more letters of the real value written as a whole word of
    ``rep``, and, with the mention ``surface``, any word of the mention written unchanged whatever its length (an
    initial at its own place: ``synthetic.unchanged_mention_word``), is a leak unless it is a fixed word (title,
    particle, legal form, generic word, the "s" of a possessive): a nickname "Reggie" of a real "Reginald ..."
    written as "Reggie". A number of two or more digits of the real name written again is a leak too (a company
    fake keeps the numbers of the real name: "317 Quorrow" drawn as "317 Brindlemoor")."""
    if ent.kind in _NAME_KINDS:
        real_numbers = set(re.findall(r"\d{2,}", ent.value))
        if real_numbers and real_numbers & set(re.findall(r"\d{2,}", rep)):
            return True
        body = re.sub(r"[^a-z0-9]", "", fold_word(rep))
        for w in phrase_tokens(ent.value):
            if len(w) >= 4 and w not in nm.PARTICLES and w not in nm.LEGAL_FORMS and w not in nm.ORG_GENERIC \
                    and w not in nm.TITLE_WORDS and not nm._lexicon_word(w) and w not in nm.ORG_TRIGGERS \
                    and re.sub(r"[^a-z0-9]", "", w) in body:
                return True
        real_words = {fold_word(m.group()) for m in re.finditer(r"[^\W\d_]{2,}", ent.value)}
        for m in re.finditer(r"[^\W\d_]{2,}", rep):
            if fold_word(m.group()) in real_words \
                    and not syn.mention_word_fixed(ent.kind, m.group(), rep, m.start(), ent.value):
                return True
        return bool(surface) and syn.unchanged_mention_word(ent.kind, surface, rep, ent.value)
    real = re.sub(r"[^0-9A-Za-z]", "", ent.value).lower()
    fake = re.sub(r"[^0-9A-Za-z]", "", rep).lower()
    if len(real) >= 6 and real in fake:
        return True
    return ent.kind == "BIRTHDATE" and det.structured_norm("BIRTHDATE", rep) == det.structured_norm("BIRTHDATE", ent.value)


def _code_in_place(text: str, s: int, e: int, code: str) -> str:
    """The code that replaces ``text[s:e]``, with a space added where a glued neighbour would make it unreadable
    for restore: "2026Kervadec" -> "2026 PERSONNE_A", "KervadecIndustries" -> "PERSONNE_A Industries"."""
    before = " " if s > 0 and text[s - 1].isalnum() else ""
    nxt = text[e] if e < len(text) else ""
    letters = code.rsplit("_", 1)[-1].isalpha()
    after = " " if nxt and ((letters and nxt.isupper()) or (not letters and nxt.isdigit())) else ""
    return before + code + after


def _handle_tokens(e: Entity) -> list[str]:
    """Distinctive pieces of a name that a social handle may hold ("kervadec" in "@mkervadec")."""
    if e.kind == "PERSON":
        words = nm.surname_key(e.value).split()
    elif e.kind == "ORG":
        words = entity_norm("ORG", e.value).split()
    else:
        return []
    stop, legal, common, _fn, afn, months, wl = nm._lists()
    out = []
    joined = "".join(words)
    if len(words) > 1 and len(joined) >= 6:
        out.append(joined)
    for w in words:
        if len(w) >= 5 and w not in common and w not in afn and w not in stop and w not in legal and w not in wl \
                and w not in nm.LEGAL_FORMS and w not in nm.ORG_GENERIC:
            out.append(w)
    return out


def _inside_compound(text: str, s: int, e: int) -> bool:
    """True when ``text[s:e]`` is glued by a hyphen or an apostrophe to another name word ("Dupont-Morvanti").

    Not a compound: an English possessive ("Kervadec's") or a lower-case word after a hyphen ("Atlantide-note").
    """
    n = len(text)
    if e + 1 < n and text[e] in _GLUE and text[e + 1].isalpha():
        if text[e] in "'’" and text[e + 1] in "sS" and (e + 2 >= n or not text[e + 2].isalnum()):
            return False
        if text[e] in _DASHES and not text[e + 1].isupper():
            return False
        return True
    if s >= 2 and text[s - 1] in _DASHES and text[s - 2].isalpha():
        return True
    return False


_DASHES = "-\u2010\u2011\u2012\u2212"
_GLUE = _DASHES + "'’"


def _copy(c: _Cand) -> _Cand:
    n = _Cand(c.start, c.end, c.kind, c.source, c.priority, c.entity, c.pending)
    n.rank = c.rank
    n.surface = c.surface
    n.rule = c.rule
    n.signals = c.signals  # v0.3 (S7): the second signals travel with the copy
    return n


_CONF_CLASS = {"0": "[0Oo]", "1": "[1lIi|]", "o": "[oO0]", "l": "[lL1I|]", "i": "[iI1l|]", "5": "[5Ss]", "s": "[sS5]",
               "8": "[8B]", "b": "[bB8]"}


@lru_cache(maxsize=16384)
def _lenient_body(kind: str, value: str) -> str:
    """Regex body for a structured value with optional separators and OCR look-alikes (0/O, 1/l/I)."""
    base = value if kind in ("EMAIL", "ADDRESS", "BIRTHDATE", "CASE_REF") else det.structured_norm(kind, value)
    chars = [c for c in fold_word(base) if c.isalnum() or c == "@"]
    parts = []
    for c in chars:
        cls = _CONF_CLASS.get(c.lower())
        if cls is None:
            cls = "[" + re.escape(c.lower()) + re.escape(c.upper()) + "]" if c.isalpha() else re.escape(c)
        parts.append(cls)
    return r"[\s.\-/\u00a0]?".join(parts)


# ---------------------------------------------------------------------------
# R1 helpers
# ---------------------------------------------------------------------------


def _initials_context(text: str, s: int, e: int) -> bool:
    """Two capitals ("LP") read as initials only in an initials context: a paraphe or signature line, a list of
    initials ("LP/MD"), a comment author ("[LP]"), a file reference ("N/Réf. 2024/0042/LP", "2025-AB-0442 LP"),
    or alone on a line (an e-mail sign-off) (R1, CL-02)."""
    if (s > 0 and text[s - 1].isalnum()) or (e < len(text) and text[e].isalpha()):
        return False
    ls = text.rfind("\n", 0, s) + 1
    le = text.find("\n", e)
    le = len(text) if le < 0 else le
    before, after = text[ls:s], text[e:le]
    if not before.strip() and not after.strip(" \t.,-"):
        return True
    if _SIGN_LINE_RE.search(text[ls:le]) or _REF_BEFORE_RE.search(before):
        return True
    if re.search(r"[A-Z]{2,4}[ \t]*/[ \t]*$", before) or re.match(r"[ \t]*/[ \t]*[A-Z]{2,4}\b", after):
        return True
    if re.search(r"[\[(][ \t]*$", before) and re.match(r"[ \t]*\d{0,3}[ \t]*[\])\]:]", after):
        return True
    prev = re.search(r"(\S+)[ \t]*$", before)
    if prev and any(c.isdigit() for c in prev.group(1)) and re.search(r"[/\-_]", prev.group(1)):
        return True
    return False


def _compound_span(text: str, s: int, e: int) -> tuple[int, int]:
    """The whole hyphen compound around ``text[s:e]`` ("Maela-Lorvain", "Lorvain-Kerbrat")."""
    n = len(text)

    def letter(c: str) -> bool:
        return c.isalpha() or unicodedata.combining(c) > 0

    while s >= 2 and text[s - 1] in _DASHES and letter(text[s - 2]):
        k = s - 2
        while k > 0 and letter(text[k - 1]):
            k -= 1
        s = k
    while e + 1 < n and text[e] in _DASHES and text[e + 1].isupper():
        k = e + 1
        while k < n and letter(text[k]):
            k += 1
        e = k
    return s, e


_MONTHS_FR = ("janvier", "fevrier", "mars", "avril", "mai", "juin", "juillet", "aout", "septembre", "octobre",
              "novembre", "decembre")
_MONTHS_EN = ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
              "november", "december")


@lru_cache(maxsize=4096)
def _date_forms(value: str) -> tuple[tuple[str, ...], ...]:
    """Token sequences of one date in its usual written forms (14/03/1979, 14.03.79, 1979-03-14, 14 mars 1979,
    1er mars 1979): a known birth date is found whatever form the text uses (R1, CL-15)."""
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", det.structured_norm("BIRTHDATE", value) or "")
    if not m:
        return ()
    y, mo, d = m.group(1), m.group(2), m.group(3)
    if not (1 <= int(mo) <= 12 and 1 <= int(d) <= 31):
        return ()
    yy, mo1, d1 = y[2:], str(int(mo)), str(int(d))
    month = _MONTHS_FR[int(mo) - 1]
    month_en = _MONTHS_EN[int(mo) - 1]
    forms = {(d, mo, y), (d1, mo1, y), (d, mo, yy), (d1, mo1, yy), (y, mo, d), (d, month, y), (d1, month, y),
             (d, month_en, y), (d1, month_en, y), (month_en, d1, y), (month_en, d, y)}  # English forms: v0.2
    if d1 == "1":
        forms.add(("1er", month, y))
    return tuple(sorted(forms))


def _fuzzy_tokens(e: Entity) -> list[str]:
    """Registered name words that a one-edit typo may target: a strong surname word or company word of 6+ letters
    that is not a dictionary word (R1, CL-06)."""
    if e.kind == "PERSON":
        words = nm.surname_key(e.value).split()
    elif e.kind == "ORG":
        words = entity_norm("ORG", e.value).split()
    else:
        return []
    return [w for w in words if len(w) >= 6 and w.isalpha() and not nm._lexicon_word(w)
            and w not in nm.ORG_GENERIC and w not in nm.LEGAL_FORMS and w not in nm.GENERATIONAL]


def _dl1(a: str, b: str) -> bool:
    """True when ``a`` and ``b`` differ by exactly one edit (insertion, deletion, substitution or the swap of two
    neighbouring letters) and start with the same letter."""
    if a == b or not a or not b or a[0] != b[0]:
        return False
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    i = 0
    while i < min(la, lb) and a[i] == b[i]:
        i += 1
    if la == lb:
        if a[i + 1:] == b[i + 1:]:
            return True  # one substitution
        return i + 1 < la and a[i] == b[i + 1] and a[i + 1] == b[i] and a[i + 2:] == b[i + 2:]
    if la > lb:
        return a[i + 1:] == b[i:]
    return a[i:] == b[i + 1:]


def _fuzzy_owner(tok: str, lex: _Lexicon) -> Optional[Entity]:
    """The one registered entity whose name word is one edit away from ``tok`` (None: none or several)."""
    if not lex.fuzzy or len(tok) < 6 or tok in lex.fuzzy:
        return None
    found: dict[str, Entity] = {}
    for t in lex.fuzzy_first.get(tok[0], ()):
        if abs(len(t) - len(tok)) <= 1 and _dl1(t, tok):
            for ent in lex.fuzzy[t]:
                found[ent.code] = ent
    return next(iter(found.values())) if len(found) == 1 else None


def _code_order(code: str) -> tuple[int, str]:
    suffix = code.rsplit("_", 1)[-1]
    return (len(suffix), suffix)


def _fuzzy_key(skey: str, table: dict) -> Optional[str]:
    """The one known surname key one edit away from ``skey`` (6+ letters, not a dictionary word) (R1, CL-06)."""
    if len(skey) < 6 or not skey.isalpha() or nm._lexicon_word(skey):
        return None
    found = [k for k in table if " " not in k and len(k) >= 6 and abs(len(k) - len(skey)) <= 1 and _dl1(k, skey)
             and not nm._lexicon_word(k)]
    return found[0] if len(found) == 1 else None


def _first_words(firsts) -> list[str]:
    """Folded given names, a compound first name kept whole ("Jean-Marc" -> "jean-marc")."""
    return [fold_word(f).replace(".", "") for f in firsts if fold_word(f).replace(".", "")]


def _compatible_firsts(a: list[str], b: list[str]) -> bool:
    """Given names of one person: one list is a prefix of the other ("ombeline" / "ombeline maud")."""
    if not a or not b:
        return False
    k = min(len(a), len(b))
    return a[:k] == b[:k]


_SURNAME_NEEDS = ("", "title", "mid", "party", "mixed_line")
_INITIAL_NEEDS = ("dotted", "ini_ctx", "inits", "acro")


def _owner(c: _Cand) -> object:
    if c.entity is not None:
        return c.entity.code
    return ("pending", id(c.pending))


def _cand_value(c: _Cand) -> str:
    if c.entity is not None:
        return c.entity.value
    return c.pending.value if c.pending is not None else c.full_text


def _one_person(ents: list[Entity]) -> bool:
    """Known entities that are one person split in several codes: one surname, compatible first names."""
    keys = {nm.surname_key(x.value) for x in ents}
    if len(keys) != 1 or not next(iter(keys)):
        return False
    firsts = [_first_words(nm.split_person(x.value)[0]) for x in ents]
    if any(not f for f in firsts):
        return False
    longest = max(firsts, key=len)
    return all(_compatible_firsts(f, longest) for f in firsts)


def _name_compound(compound: str, known: str) -> bool:
    """A hyphen compound that is a name: every other part is a capitalised name word (a first name or a word that is
    not in the dictionaries), not a document word or a number ("Conclusions-Dupont" and "Notes-Dupont-2026" are
    file names, "Maela-Dupont" is a name)."""
    parts = [p for p in re.split("[" + _DASHES + "]", compound) if p]
    others = [p for p in parts if fold_word(p) != fold_word(known)]
    if not others or len(parts) > 4:
        return False
    afn = nm._lists()[4]
    for p in others:
        f = fold_word(p)
        if len(f) < 2 or not p[:1].isupper() or not f.isalpha():
            return False
        word = nm._lexicon_word(f) or f in nm.ORG_GENERIC or (f[-1] in "sx" and nm._lexicon_word(f[:-1]))
        if f not in afn and word:
            return False
    return True


def _dictionary_word(tok: str) -> bool:
    """A common, function, legal, calendar or generic business word, or a first name."""
    return nm._lexicon_word(tok) or tok in nm.ORG_GENERIC


@lru_cache(maxsize=4096)
def lx_nicknames(first: str) -> tuple[str, ...]:
    """The table diminutives of a folded first name, from both languages' lexicons."""
    out: set[str] = set(nm.lx.nicknames().get(first, ()))
    try:
        out |= set(lc.bundle("en").nicknames.get(first, ()))
    except Exception:  # noqa: BLE001 - no English bundle
        pass
    return tuple(sorted(out))


@lru_cache(maxsize=65536)
def _any_first_name(tok: str) -> bool:
    """A first name of either lexicon."""
    if tok in nm._lists()[4]:
        return True
    try:
        return tok in lc.bundle("en").all_first_names
    except Exception:  # noqa: BLE001
        return False


def _in_slot(text: str, s: int, e: int, kind: str) -> bool:
    """A login name or an identifier value sits in its own slot (v0.3, S2): right after "@" (or "<@"), after a
    label on its line ("Username: jdoe", "Employee ID: 4471"), or shaped like an id (a digit or an underscore
    inside; for an identifier, an all-capital code too). A plain word of the prose is never such a value."""
    surface = text[s:e]
    if s > 0 and text[s - 1] == "@":
        return True
    if re.search(r"[\d_]", surface):
        return True
    if kind == "ID" and re.fullmatch(r"[A-Z0-9][A-Z0-9\-./]{4,}", surface) and any(ch.isdigit() for ch in surface):
        return True
    line_start = text.rfind("\n", 0, s) + 1
    before = text[line_start:s]
    return bool(_SLOT_LABEL_RE.search(before[-48:]))


def _person_from_value(value: str) -> dr.Person:
    """A directory Person built from a stored person value (its split, its quoted nicknames); a value stored in
    "Last, First" order is read in that order."""
    firsts, surname = nm.split_person(_comma_order(value))
    nicks: list[str] = []
    fn = getattr(nm, "person_nicknames", None)
    if callable(fn):
        try:
            nicks = list(fn(value))
        except Exception:  # noqa: BLE001
            nicks = []
    return dr.Person(value, tuple(t for w in firsts for t in phrase_tokens(w)),
                     tuple(t for w in surname for t in phrase_tokens(w)),
                     tuple(sorted({t for w in nicks for t in phrase_tokens(w)})))


def _dir_person_for(directory: dr.Directory, e: Entity) -> Optional[dr.Person]:
    """The directory person behind a store entity: the same engine key, else the one person whose full name the
    entity value writes (any order, with or without middle names)."""
    person = directory.by_norm(e.norm)
    if person is None:
        try:
            person = directory.by_norm(entity_norm("PERSON", e.value))
        except Exception:  # noqa: BLE001
            person = None
    if person is not None:
        return person
    hits = {m.person.norm: m.person for m in directory.lookup(e.value) if m.role == "full"}
    return next(iter(hits.values())) if len(hits) == 1 else None
