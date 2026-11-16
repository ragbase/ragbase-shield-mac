"""Merge NER spans into the masker's candidates, with the precision guards (design: V02_DESIGN.md 4.2 and 4.6).

``merge_ner(text, ner_spans, rule_spans, structured, protected, lang=...)`` takes the output of
:func:`ragbase_shield.engine.ner.run_ner` and returns :class:`MergedSpan` items the masker turns into candidates
(``_Cand(s, e, kind, "ner", 1)``). The rules in order:

1. edges: titles, legal forms, leading function words and punctuation are trimmed;
2. a span that overlaps a structured span (checksum identifiers, emails...) or a protected interval (whitelist,
   court + city, law reference, defined case law) is dropped;
3. a span made only of stop words, legal terms, months, titles or NER stop-list entries is dropped; a statutory
   phrase (spelled legal form, "syndicat des copropriétaires", a role phrase) at the start is cut off;
4. rules win on kind: a span equal to a rule span adds nothing; a span that contains a rule span keeps the rule's
   kind and extends it only when every extra word is a name word; any other overlap with a rule span is dropped;
5. NER-only spans pass the guards: capitalised common word, place, acronym, all-capital line, one-word person
   (score and context cue), one-word organisation, generic organisation words, value shape for identifiers; a
   company of two or more vocabulary words ("Bright Path Foods") passes only with a second signal of rule 8, or
   when the model finds it at ``STRONG_REPEAT_MIN`` or more in two chunks (:func:`_vocab_org_ok`);
6. nested spans collapse to the longest; the result is sorted by position;
7. kind policy (gated layer, 2026-10-20): on the masker path (``kinds=None``) a model find may add only the kinds
   of ``MODEL_KINDS`` (people, organisations, addresses, and the identifier kinds USERNAME, EMPLOYEE_ID,
   ACCOUNT_NUMBER); a model "username" needs a handle shape, an id needs a label cue or a letter-prefixed code
   shape; ID_CODE, EMAIL, PHONE and BIRTHDATE finds of the model and the named assets (vessel, building, school)
   go to :func:`watch_ner` for the review step and are never masked on their own;
8. second signals: every kept span carries ``signals``, the independent confirmations of the find (a context
   cue, the rules finding the same name or its surname or head word, the model finding it in two chunks, an
   e-mail domain of the text, a known entity of the dossier). The masker gives store variants to a model-made
   person or organisation only when a signal is present, and none to a model-made handle or identifier (their
   mentions are masked in place; a dotted handle must never match a two-word name elsewhere): one model mistake
   then stays one mention, never a dossier-wide relabel.

:func:`guard_ner_spans` is the same without rule or structured spans (the ``Masker(ner=...)`` hook path), with
every model kind allowed.
Everything is deterministic, standard library only, and the French lists come from the lexicon package; the
English lists come from ``*_en.txt`` files when they exist (LOCALE-EN) and from a small built-in fallback
otherwise, so the French stop list is never applied to an English text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, NamedTuple, Optional, Sequence

from . import lexicon as lx
from . import names as nm
from . import ner as nr
from .detectors import IntervalSet
from .normalize import fold, fold_word, tokens

__all__ = ["MergedSpan", "WatchSpan", "merge_ner", "guard_ner_spans", "watch_ner", "is_name_word", "trim_span",
           "lists_for", "ner_stoplist", "vocab", "spelled_forms", "places", "NAME_KINDS", "VALUE_KINDS",
           "MODEL_KINDS", "ALL_MODEL_KINDS", "SIGNALS"]

NAME_KINDS = ("PERSON", "ORG")
VALUE_KINDS = ("EMAIL", "PHONE", "ADDRESS", "BIRTHDATE", "USERNAME", "EMPLOYEE_ID", "ACCOUNT_NUMBER", "ID_CODE")
# Labels whose value is a name of a thing (letters), not a number: looser value checks.
NAMED_ASSET_LABELS = frozenset({"vessel", "building", "school", "vessel name", "building name"})
#: Kinds a model find may add on its own on the masker path (``merge_ner`` with ``kinds=None``). EMAIL, PHONE
#: and BIRTHDATE belong to the structured detectors; ID_CODE finds of the model (government ids, secrets, named
#: assets) go to the watch list (:func:`watch_ner`), never to a mask without a person's decision.
MODEL_KINDS = frozenset({"PERSON", "ORG", "ADDRESS", "USERNAME", "EMPLOYEE_ID", "ACCOUNT_NUMBER"})
ALL_MODEL_KINDS = frozenset(NAME_KINDS) | frozenset(VALUE_KINDS)
#: The second signals that confirm a model find (``MergedSpan.signals``).
SIGNAL_CUE = "cue"          # a context cue at the mention (title, greeting, role word, "at X", legal form, label)
SIGNAL_RULE = "rule"        # the same value, its surname, its first name or its head word found by the rules here
SIGNAL_REPEAT = "repeat"    # the model found the same value in two different chunks of this text
SIGNAL_DOMAIN = "domain"    # the organisation is a label of an e-mail domain or a URL host of this text
SIGNAL_KNOWN = "known"      # the value is a registered entity of the dossier (rules or user), from ``known_values``
SIGNALS = (SIGNAL_CUE, SIGNAL_RULE, SIGNAL_REPEAT, SIGNAL_DOMAIN, SIGNAL_KNOWN)


class MergedSpan(NamedTuple):
    start: int
    end: int
    kind: str
    label: str
    score: float
    origin: str   # "ner" (NER alone) or "rule+ner" (a rule span the NER extended)
    signals: tuple = ()   # second signals (``SIGNALS``); empty = a model-only find with no confirmation yet


class WatchSpan(NamedTuple):
    """A model find for the review step (:func:`watch_ner`): never masked on its own."""
    start: int
    end: int
    kind: str
    label: str
    score: float


# ---------------------------------------------------------------------------
# Lexicon files owned by the NER builder (optional: an absent file is an empty list)
# ---------------------------------------------------------------------------


def _read_optional(name: str) -> list[str]:
    try:
        return lx.read_lines(name)
    except (FileNotFoundError, OSError):
        return []


def _fold_phrase(line: str) -> str:
    return " ".join(w for w in (fold_word(t) for t in line.split()) if w)


@lru_cache(maxsize=None)
def ner_stoplist(lang: str = "fr") -> tuple[frozenset[str], tuple[str, ...]]:
    """(single words, multi-word phrases), folded: an NER span may never be made of these alone."""
    words: set[str] = set()
    phrases: set[str] = set()
    for line in _read_optional(f"ner_stoplist_{lang}.txt"):
        f = _fold_phrase(line)
        if not f:
            continue
        (phrases if " " in f else words).add(f)
    return frozenset(words), tuple(sorted(phrases, key=len, reverse=True))


@lru_cache(maxsize=None)
def vocab(lang: str = "fr") -> frozenset[str]:
    """Common vocabulary of the language (folded): a capitalised one of these is not a name by itself.

    ``vocab_<lang>.txt`` may hold several words per line (space separated)."""
    out: set[str] = set()
    for line in _read_optional(f"vocab_{lang}.txt"):
        for word in line.split():
            f = fold_word(word)
            if f:
                out.add(f)
    return frozenset(out)


@lru_cache(maxsize=None)
def spelled_forms(lang: str = "fr") -> tuple[tuple[str, ...], frozenset[str]]:
    """(multi-word legal-form phrases, short forms), folded. Never a name; cut from the start of an organisation."""
    phrases: set[str] = set()
    shorts: set[str] = set()
    for line in _read_optional(f"legal_forms_spelled_{lang}.txt"):
        f = _fold_phrase(line)
        if not f:
            continue
        (phrases if " " in f else shorts).add(f)
    return tuple(sorted(phrases, key=len, reverse=True)), frozenset(shorts)


@lru_cache(maxsize=None)
def places(lang: str = "fr") -> frozenset[str]:
    """Place names (folded) from ``places_<lang>.txt`` when LOCALE-EN ships it; empty otherwise."""
    return frozenset(f for f in (fold_word(x) for x in _read_optional(f"places_{lang}.txt")) if f)


# ---------------------------------------------------------------------------
# Language lists
# ---------------------------------------------------------------------------

# Built-in English fallback until LOCALE-EN ships the ``*_en.txt`` bundle (function words, titles, greetings,
# months and days). Written for RAGbase Shield.
_EN_STOP = frozenset("""
the a an and or but nor so yet of to in on at by for with from as into onto upon about over under between among
through during before after above below up down out off again further then once here there when where why how all
any both each few more most other some such no not only own same than too very can will just should now this that
these those it its is are was were be been being have has had having do does did doing i me my we our you your he
him his she her they them their what which who whom whose if because while until although though unless since
re fwd fw cc bcc subject date sent attachment attached please thanks thank hello hi hey dear regards best cheers
sincerely kind warm yours ok okay fyi asap eod cob q1 q2 q3 q4 am pm vs via per etc
""".split())
_EN_TITLES = frozenset({"mr", "mrs", "ms", "mx", "dr", "prof", "sir", "dame", "rev", "hon", "judge", "justice",
                        "capt", "lt", "sgt", "officer", "mister", "miss", "madam", "professor", "doctor"})
_EN_GREETINGS = frozenset({"hi", "hello", "hey", "dear", "thanks", "thank", "regards", "best", "cheers", "sincerely",
                           "yours", "cc", "attn", "attention", "greetings", "morning", "afternoon", "evening"})
_EN_MONTHS = frozenset({"january", "february", "march", "april", "may", "june", "july", "august", "september",
                        "october", "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
                        "sept", "oct", "nov", "dec", "monday", "tuesday", "wednesday", "thursday", "friday",
                        "saturday", "sunday", "mon", "tue", "wed", "thu", "fri", "sat", "sun"})
_EN_LEGAL_FORMS = frozenset({"inc", "incorporated", "llc", "corp", "corporation", "co", "company", "ltd", "limited",
                             "llp", "lp", "plc", "pllc", "pc", "pty", "gmbh", "ag", "nv", "bv", "sa", "srl", "spa"})
_EN_ORG_GENERIC = frozenset({"group", "holdings", "partners", "technologies", "systems", "solutions", "analytics",
                             "labs", "software", "foods", "industries", "capital", "bank", "insurance", "airlines",
                             "motors", "pharmaceuticals", "logistics", "consulting", "media", "networks", "ventures",
                             "trust", "services", "international", "global", "enterprises", "brands", "studios",
                             "health", "energy", "financial", "team", "department", "division", "office", "unit"})
_FR_GREETINGS = frozenset(nm._GREETINGS) | {"cordialement", "amicalement", "sincerement", "respectueusement",
                                             "attn", "cc", "attention"}
_FR_CONNECT = frozenset({"de", "du", "des", "d", "la", "le", "les", "l", "et", "&", "en", "sur", "sous", "aux", "au"})
_EN_CONNECT = frozenset({"of", "the", "and", "&", "for", "at", "in", "on", "de", "la", "le", "du", "des", "d", "l"})
_PARTICLES = frozenset(nm.PARTICLES) | {"of", "the"}
_DEFINED_RE = re.compile(r"(?i)(?:ci-?\s?apr[èe]s|hereinafter|hereafter|d[ée]nomm[ée]e?s?|d[ée]sign[ée]e?s?|"
                         r"appel[ée]e?s?|referred to as|known as)[\s\u00a0]*[:,]?[\s\u00a0]*"
                         r"(?:l[ea]s?[\s\u00a0]+|l['’][\s\u00a0]*)?[«\"“(\[]?[\s\u00a0]*$|[(\[][\s\u00a0«\"“']*$")


@dataclass(frozen=True)
class Lists:
    lang: str
    stop: frozenset
    legal: frozenset
    common: frozenset
    first_names: frozenset
    all_first_names: frozenset
    months: frozenset
    whitelist_single: frozenset
    vocab: frozenset
    places: frozenset
    acronyms: frozenset
    titles: frozenset
    greetings: frozenset
    org_generic: frozenset
    legal_forms: frozenset
    connect: frozenset
    ner_stop_words: frozenset
    ner_stop_phrases: tuple
    spelled_phrases: tuple
    spelled_short: frozenset

    def excluded(self, w: str) -> bool:
        """A folded word that is never a name word."""
        return (w in self.stop or w in self.legal or w in self.months or w in self.titles
                or w in self.whitelist_single or w in self.legal_forms or w in self.spelled_short)

    def alone_excluded(self, w: str) -> bool:
        """A folded word that is never a name by itself (the NER stop list adds document and role nouns that may
        still sit inside a longer name: "Groupe X", "Service X")."""
        return self.excluded(w) or w in self.ner_stop_words

    def vocabulary(self, w: str) -> bool:
        """A folded word that is an ordinary word of the language."""
        return w in self.vocab or w in self.common


def _folded_set(name: str) -> frozenset[str]:
    return frozenset(f for f in (fold_word(x) for x in _read_optional(name)) if f)


@lru_cache(maxsize=None)
def lists_for(lang: str = "fr") -> Lists:
    """The word lists of one language. ``lexicon.bundle(lang)`` is used when LOCALE-EN ships it."""
    b = None
    if lang != "fr":
        try:  # LOCALE-EN: engine/locale.py bundle(lang); an absent or incomplete module must not break the merge
            from . import locale as _locale

            b = _locale.bundle(lang)
        except Exception:  # noqa: BLE001
            b = None

    def field(name: str, default):
        v = getattr(b, name, None) if b is not None else None
        return frozenset(v) if v else default

    if lang == "fr":
        stop, legal, common, fn, afn, months, wl = nm._lists()
        return Lists(
            lang="fr", stop=stop, legal=legal, common=common, first_names=fn, all_first_names=afn, months=months,
            whitelist_single=wl, vocab=vocab("fr"), places=field("places", places("fr")),
            acronyms=lx.acronym_stoplist() | lx.instrument_terms(), titles=frozenset(nm.TITLE_WORDS),
            greetings=_FR_GREETINGS, org_generic=frozenset(nm.ORG_GENERIC) | frozenset(nm.ORG_TRIGGERS),
            legal_forms=frozenset(nm.LEGAL_FORMS), connect=_FR_CONNECT,
            ner_stop_words=ner_stoplist("fr")[0], ner_stop_phrases=ner_stoplist("fr")[1],
            spelled_phrases=spelled_forms("fr")[0], spelled_short=spelled_forms("fr")[1],
        )
    wl_single = frozenset(p for p in field("whitelist_phrases", _folded_set(f"whitelist_{lang}.txt")) if " " not in p)
    legal = field("legal_terms", _folded_set(f"legal_terms_{lang}.txt")) | field("roles", frozenset())
    org_generic = (field("org_generic", _folded_set(f"org_words_{lang}.txt") | _EN_ORG_GENERIC)
                   | field("org_suffixes", frozenset()) | field("org_triggers", frozenset()))
    spelled = spelled_forms(lang)[0]
    b_spelled = getattr(b, "legal_forms_spelled", None) if b is not None else None
    if b_spelled:
        spelled = tuple(sorted(set(spelled) | {" ".join(t) for t in b_spelled if t}, key=len, reverse=True))
    return Lists(
        lang=lang,
        stop=field("stopwords", _folded_set(f"stopwords_{lang}.txt") | _EN_STOP),
        legal=legal,
        common=field("common_words", _folded_set(f"common_words_{lang}.txt")),
        first_names=field("first_names", _folded_set(f"first_names_{lang}.txt")),
        all_first_names=field("all_first_names", _folded_set(f"first_names_{lang}.txt")),
        months=field("months_days", _EN_MONTHS),
        whitelist_single=wl_single, vocab=field("vocab", frozenset()) | vocab(lang),
        places=field("places", frozenset()) | places(lang),
        acronyms=field("acronyms", _folded_set(f"acronyms_{lang}.txt")),
        titles=field("titles", _EN_TITLES) | _EN_TITLES, greetings=field("greetings", _EN_GREETINGS) | _EN_GREETINGS,
        org_generic=org_generic,
        legal_forms=frozenset(k.rstrip(".") for k in field("legal_forms", _EN_LEGAL_FORMS)) | frozenset(nm.LEGAL_FORMS),
        connect=_EN_CONNECT,
        ner_stop_words=ner_stoplist(lang)[0], ner_stop_phrases=ner_stoplist(lang)[1],
        spelled_phrases=spelled, spelled_short=spelled_forms(lang)[1],
    )


# ---------------------------------------------------------------------------
# Words and shapes
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[^\W\d_](?:[^\W_]|['’\-](?=[^\W\d_]))*")
_TOKEN_RE = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*")
_TITLE_LEAD_RE = re.compile(
    r"^(?:(?:M|MM|Mme|Mmes|Mlle|Mlles|Me|Mes|Dr|Pr|Mr|Mrs|Ms|Mx|Prof|Sir|Dame|Rev|Hon|Capt|Lt|Sgt)\.?"
    r"|(?i:monsieur|madame|mademoiselle|messieurs|mesdames|ma[îi]tre|docteur|professeur|professor|doctor|mister|"
    r"judge|justice|officer))(?=[\s ])[\s ]+")
_POSSESSIVE_RE = re.compile(r"['’][sS]$")
_INNER_POSSESSIVE_RE = re.compile(r"['’][sS](?=[\s\u00a0]+[a-z])")
_ACRONYM_RE = re.compile(r"^[A-ZÀ-ÖØ-Þ](?:[A-ZÀ-ÖØ-Þ]|\.(?=[A-ZÀ-ÖØ-Þ])|&(?=[A-ZÀ-ÖØ-Þ])){1,5}\.?$")
# a registry or institution acronym written with dots at the start of a span ("R.C.S. Lyon", "S.I.R.E.N. 123")
_DOTTED_LEAD_RE = re.compile(r"^((?:[A-ZÀ-ÖØ-Þ]\.){2,6})[\s\u00a0]*")
_EDGE_PUNCT = " \t  \r\n\"'«»“”‘’()[]{}<>,;:.!?*_-–—/\\|•"
_YEAR_RE = re.compile(r"^(?:1[89]|20)\d\d$")
_DATE_RE = re.compile(r"^\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}$|^\d{4}[/.\-]\d{1,2}[/.\-]\d{1,2}$")
_AMOUNT_AFTER_RE = re.compile(r"^[\s  ]*(?:€|\$|£|%|eur\b|euros?\b|usd\b|k€|m€)", re.IGNORECASE)
_STREET_WORDS = frozenset({"rue", "avenue", "av", "boulevard", "bd", "chemin", "allee", "impasse", "place", "route",
                           "quai", "cours", "square", "passage", "lotissement", "residence", "voie", "street", "st",
                           "ave", "road", "rd", "drive", "dr", "lane", "ln", "way", "court", "ct", "boulevard",
                           "blvd", "terrace", "circle", "highway", "hwy", "parkway", "box"})
_CUE_WORDS_PERSON = frozenset({"avec", "pour", "chez", "with", "to", "from", "at", "by", "for", "via", "contact",
                               "signe", "signed", "attn", "re", "ref", "c/o", "cc",
                               "dossier", "affaire", "succession", "consorts", "epoux", "epouse", "veuve", "heritiers",
                               "contre", "matter", "estate", "case", "v", "vs", "versus"})
_VERSUS_BEFORE_RE = re.compile(r"(?i)(?:^|\s)(?:c/|c\.|vs\.?|v\.)[\s\u00a0]*$")
_VERSUS_AFTER_RE = re.compile(r"^[\s\u00a0]*(?:c/|c\.|vs\.?|v\.)[\s\u00a0]+[^\W\d_]")
# a quoted nickname after a naming verb: « Lulu », "Tibou"
_NICK_CUE_RE = re.compile(r"(?i)(?:surnomm[ée]e?s?|surnomment|dits?|dites?|alias|appel[ée]e?s?|connue?s? sous le nom de|"
                          r"known as|a\.?k\.?a\.?|nicknamed|called|goes by)[^\n«\"“'‘]{0,20}[«\"“'‘][\s\u00a0]*$")
_CLOSING_QUOTE_RE = re.compile(r"^[\s\u00a0]*[»\"”'’]")
_CUE_WORDS_ORG = frozenset({"chez", "at", "with", "from", "for", "by", "joined", "client", "vendor", "partner",
                            "supplier", "customer", "fournisseur", "prestataire", "partenaire", "societe", "ste",
                            "groupe", "cabinet", "entreprise", "company", "firm", "the", "via", "acquired", "sold"})
# A role noun right before a capitalised word is a context cue for a person ("gérant Dupont", "manager Jane").
_FR_ROLE_CUES = frozenset(nm._ROLE_WORDS) | frozenset({
    "liquidateur", "liquidatrice", "mandataire", "administrateur", "administratrice", "president", "presidente",
    "directeur", "directrice", "gerant", "gerante", "avocat", "avocate", "notaire", "huissier", "expert", "experte",
    "juge", "conseiller", "conseillere", "greffier", "greffiere", "procureur", "substitut", "rapporteur", "temoin",
    "docteur", "professeur", "medecin", "infirmier", "infirmiere", "collaborateur", "collaboratrice", "salarie",
    "salariee", "employe", "employee", "stagiaire", "apprenti", "apprentie", "consultant", "consultante", "client",
    "cliente", "patient", "patiente", "proprietaire", "locataire", "bailleur", "vendeur", "vendeuse", "acquereur",
    "cedant", "cessionnaire", "signataire", "representant", "representante", "delegue", "deleguee", "secretaire",
    "tresorier", "tresoriere", "associe", "associee", "actionnaire", "fondateur", "fondatrice", "dirigeant",
    "dirigeante", "responsable", "chef", "manager", "collegue", "confrere", "consoeur", "ami", "amie", "voisin",
    "voisine", "monsieur", "madame", "mademoiselle",
})
_EN_ROLE_CUES = frozenset({
    "manager", "colleague", "employee", "ceo", "cfo", "cto", "coo", "director", "lawyer", "attorney", "counsel",
    "judge", "doctor", "nurse", "agent", "officer", "partner", "associate", "client", "customer", "user", "witness",
    "plaintiff", "defendant", "claimant", "respondent", "tenant", "landlord", "buyer", "seller", "owner", "founder",
    "president", "chairman", "secretary", "treasurer", "engineer", "developer", "designer", "analyst", "intern",
    "assistant", "coordinator", "consultant", "contractor", "supervisor", "lead", "head", "friend", "neighbour",
    "neighbor", "colleagues", "teammate", "recruiter", "candidate", "applicant", "patient", "driver", "guest",
    "mr", "mrs", "ms", "mx", "dr", "prof",
})
_SIGNOFF_RE = re.compile(r"(?i)^[\s ]*(?:cordialement|bien cordialement|bien à vous|bien a vous|amicalement|"
                         r"sincères salutations|salutations|merci|thanks|thank you|regards|best regards|kind regards|"
                         r"best|cheers|sincerely|yours|many thanks|signé|signe|signed)[\s ,.:!-]*$")


def _words_of(s: str) -> list[str]:
    return _WORD_RE.findall(s)


def _folded_words(s: str) -> list[str]:
    return [f for f in (fold_word(w) for w in _words_of(s)) if f]


def _is_acronym(s: str) -> bool:
    if not _ACRONYM_RE.match(s):
        return False
    letters = sum(c.isalpha() for c in s)
    return 2 <= letters <= 5


def _line_bounds(text: str, s: int, e: int) -> tuple[int, int]:
    ls = text.rfind("\n", 0, s) + 1
    le = text.find("\n", e)
    return ls, (len(text) if le == -1 else le)


def _all_caps_line(line: str) -> bool:
    return any(c.isalpha() for c in line) and not any(c.islower() for c in line)


def is_name_word(word: str, lang: str = "fr", kind: str = "PERSON", *, lists: Optional[Lists] = None) -> bool:
    """A surface word that may extend a rule span of ``kind``: capitalised (or a particle), not a function, legal
    or month word, not an ordinary word of the language unless it is a first name (people) or a generic company
    word (organisations)."""
    L = lists or lists_for(lang)
    if not word:
        return False
    f = fold_word(word)
    if not f:
        return False
    if word[0].islower():
        return f in nm.PARTICLES if kind == "PERSON" else (f in _PARTICLES or f in L.connect)
    if not word[0].isalpha() or L.excluded(f):
        return False
    if sum(c.isalpha() for c in word) < 2:
        return False
    if L.vocabulary(f):
        if kind == "PERSON":
            return f in L.first_names
        return f in L.org_generic
    return f not in L.places


# ---------------------------------------------------------------------------
# Trim
# ---------------------------------------------------------------------------


def _legal_form_here(word: str, L: Lists) -> bool:
    f = fold_word(word).replace(".", "")
    if f not in L.legal_forms and f not in L.spelled_short:
        return False
    base = word.rstrip(".")
    letters = [c for c in base if c.isalpha()]
    return bool(letters) and (all(c.isupper() for c in letters) or base in nm._LEGAL_MIXED
                              or base.replace(".", "") in nm._LEGAL_MIXED or L.lang != "fr")


def trim_span(text: str, s: int, e: int, kind: str, lang: str = "fr", *, lists: Optional[Lists] = None) -> tuple[int, int]:
    """Edges of a span without titles, leading function words, statutory phrases (people and organisations),
    trailing legal forms and punctuation."""
    L = lists or lists_for(lang)
    changed = True
    while changed and s < e:
        changed = False
        while s < e and text[s] in _EDGE_PUNCT:
            s += 1
            changed = True
        while e > s and text[e - 1] in _EDGE_PUNCT:
            e -= 1
            changed = True
        if kind in NAME_KINDS:
            s, e, cut = _cut_statutory_prefix(text, s, e, L)
            if cut:
                changed = True
                continue
        seg = text[s:e]
        m = _TITLE_LEAD_RE.match(seg)
        if m and m.end() < len(seg):
            s += m.end()
            changed = True
            continue
        # a leading function word, company trigger or role noun: "la société X", "the X", "Groupe X",
        # "Distributeur X", "Gérant X" (the rules keep these words in clear too)
        m = _TOKEN_RE.match(seg)
        if m and m.end() < len(seg):
            tok = m.group()
            f = fold_word(tok)
            nxt = _TOKEN_RE.search(seg, m.end())
            next_cap = bool(nxt and nxt.group()[0].isupper())
            if tok[0].islower() and (f in L.stop or (kind == "ORG" and (f in nm.ORG_TRIGGERS or f in L.connect))
                                     or f in L.titles or f in L.greetings):
                s += m.end()
                changed = True
                continue
            if next_cap and ((kind == "ORG" and f in nm.ORG_TRIGGERS)
                             or ((f in L.ner_stop_words or f in L.legal) and f not in L.first_names)):
                s += m.end()
                changed = True
                continue
        # trailing legal form, connector or possessive
        toks = list(_TOKEN_RE.finditer(seg))
        if toks:
            last = toks[-1]
            tok = last.group()
            f = fold_word(tok)
            if last.end() == len(seg) and len(toks) > 1 and (_legal_form_here(tok, L)
                                                              or (f in L.connect and tok[0].islower())):
                e = s + toks[-2].end()
                changed = True
                continue
        pm = _POSSESSIVE_RE.search(seg)
        if pm and pm.start() > 0:
            e = s + pm.start()
            changed = True
            continue
        if kind in NAME_KINDS:
            im = _INNER_POSSESSIVE_RE.search(seg)
            if im and im.start() > 0:
                e = s + im.start()  # "Acme's snacks": the name ends at the possessive
                changed = True
    return s, e


def _cut_statutory_prefix(text: str, s: int, e: int, L: Lists) -> tuple[int, int, bool]:
    """Cut a statutory phrase ("société par actions simplifiée", "syndicat des copropriétaires", "liquidateur
    judiciaire") from the start of a span. Returns the new edges and whether something was cut."""
    seg = text[s:e]
    folded = fold(seg)
    ftoks = tokens(folded.text)
    if not ftoks:
        return s, e, False
    words = [t[0] for t in ftoks]
    cut = False
    for phrases in (L.spelled_phrases, L.ner_stop_phrases):
        for phrase in phrases:
            pw = phrase.split()
            if len(pw) <= len(words) and words[:len(pw)] == pw:
                # the phrase covers the whole span, or sits before its name part
                end_f = ftoks[len(pw) - 1][2]
                _o_s, o_e = folded.span(ftoks[0][1], end_f)
                s = s + o_e
                cut = True
                break
        if cut:
            break
    if cut:
        while s < e and (text[s] in _EDGE_PUNCT):
            s += 1
        if s < e and text[s].islower():
            return s, s, True  # the phrase was the whole name ("syndicat des copropriétaires de l'immeuble")
    return s, e, cut


# ---------------------------------------------------------------------------
# Context cues
# ---------------------------------------------------------------------------


_TITLE_DOT_BEFORE_RE = re.compile(
    r"(?:^|[\s\u00a0(«\"“'])(?:M|MM|Mme|Mmes|Mlle|Mlles|Me|Mes|Dr|Pr|Mr|Mrs|Ms|Mx|Prof|Sir|Dame|Rev|Hon|Capt|Lt|Sgt)"
    r"\.?[\s\u00a0]*$")
_SENTENCE_CUT_RE = re.compile(r"[.!?…][\"'»)\]]*[\s\u00a0]+|\n")


def _cue_before(text: str, s: int, kind: str, L: Lists) -> bool:
    before = text[max(0, s - 40):s]
    if kind == "PERSON" and _TITLE_DOT_BEFORE_RE.search(before):
        return True
    # a cue lives in the same sentence and on the same line as the name
    last_cut = 0
    for m in _SENTENCE_CUT_RE.finditer(before):
        last_cut = m.end()
    before = before[last_cut:]
    if kind == "PERSON" and _NICK_CUE_RE.search(text[max(0, s - 60):s]):
        tok = _TOKEN_RE.match(text, s)
        if tok and _CLOSING_QUOTE_RE.match(text[tok.end():tok.end() + 4]):
            return True
    if _VERSUS_BEFORE_RE.search(before):
        return True
    tail = before.rstrip(" \t  :,;-–—(\"«“'")
    if before.endswith(("@", "<@")) or tail.endswith("@"):
        return True
    found = re.findall(r"[^\W\d_][\w'’\-/.]*", tail)
    if not found:
        return False
    last = [fold_word(x.rstrip(".")) for x in found[-2:]]
    w = last[-1]
    if not w:
        return False
    if kind == "PERSON":
        roles = _FR_ROLE_CUES if L.lang == "fr" else _EN_ROLE_CUES
        if w in L.titles or w in L.greetings or w in _CUE_WORDS_PERSON or w in nm._PLURAL_TITLES or w in roles:
            return True
        # "liquidateur judiciaire X", "avocat associé X": a role word followed by a legal or ordinary adjective
        return len(last) == 2 and last[0] in roles and (w in L.legal or L.vocabulary(w))
    return w in _CUE_WORDS_ORG or w in L.titles


def _cue_after(text: str, e: int, kind: str, L: Lists) -> bool:
    after = text[e:e + 24]
    if _POSSESSIVE_RE.match(after[:2]) or after.startswith(("'s", "’s")):
        return True
    if _VERSUS_AFTER_RE.match(after):
        return True  # "X c/ Y": the parties of a case
    m = re.match(r"[\s ]*,[\s ]*([^\W\d_][\w'’\-]*)", after)
    if m:
        w = fold_word(m.group(1))
        if kind == "PERSON" and (w in L.legal or w in nm._ROLE_WORDS):
            return True  # "Jane, manager" / "Dupont, gérant"
    if kind == "ORG":
        m = _TOKEN_RE.match(after.lstrip("  "))
        if m and _legal_form_here(m.group(), L):
            return True
    return False


def _signature_line(text: str, s: int, e: int) -> bool:
    """The span is alone on its line and the previous non-empty line is a sign-off ("Cordialement,")."""
    ls, le = _line_bounds(text, s, e)
    line = text[ls:le].strip(" \t ,.-–—")
    if line != text[s:e].strip():
        return False
    prev_end = ls - 1
    while prev_end > 0 and text[prev_end - 1] in "\r\n \t":
        prev_end -= 1
    if prev_end <= 0:
        return False
    prev_start = text.rfind("\n", 0, prev_end) + 1
    return bool(_SIGNOFF_RE.match(text[prev_start:prev_end]))


def _defined_term_before(text: str, s: int) -> bool:
    return bool(_DEFINED_RE.search(text[max(0, s - 60):s]))


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def _org_initials_at(text: str, s: int, e: int, L: Lists) -> set[str]:
    """Initials of the organisation at ``text[s:e]``, also with the capitalised trigger word right before it
    ("Société Verrière de X-Y" trimmed to "Verrière de X-Y" still gives SVXY)."""
    out = _org_initials(text[s:e], L)
    before = text[max(0, s - 30):s]
    m = re.search(r"([^\W\d_][\w'’\-]*)[\s\u00a0]+$", before)
    if m and m.group(1)[0].isupper():
        f = fold_word(m.group(1))
        if f in nm.ORG_TRIGGERS or f in nm.ORG_TRADES or f in L.org_generic:
            out |= _org_initials(m.group(1) + " " + text[s:e], L)
    return out


def _org_initials(name: str, L: Lists) -> set[str]:
    """Initials of a multi-word organisation name, with and without a leading trigger word ("Société ...")."""
    parts: list[str] = []
    for w in _words_of(name):
        f = fold_word(w)
        if f in L.connect or f in L.stop:
            continue
        for piece in re.split(r"[-'’]", w):
            if piece and piece[0].isalpha():
                parts.append(piece[0].upper())
    out: set[str] = set()
    if len(parts) >= 2:
        out.add("".join(parts))
        first = fold_word(_words_of(name)[0]) if _words_of(name) else ""
        if first in nm.ORG_TRIGGERS and len(parts) >= 3:
            out.add("".join(parts[1:]))
    return out


def _person_initials(name: str, L: Lists) -> set[str]:
    """Initials of a person ("Paul-Henri Garnaud", "P.-H. G." -> PHG, PG): all parts, and first + last."""
    parts: list[str] = []
    for w in _words_of(name):
        f = fold_word(w)
        if len(w) > 1 and (f in nm.PARTICLES or f in L.titles):
            continue  # a single capital letter is an initial, never a particle
        for piece in re.split(r"[-'’]", w):
            if piece and piece[0].isalpha():
                parts.append(piece[0].upper())
    out: set[str] = set()
    if 2 <= len(parts) <= 4:
        out.add("".join(parts))
        if len(parts) > 2:
            out.add(parts[0] + parts[-1])
    return out


def _first_tokens(name: str, L: Lists) -> set[str]:
    """Folded first-name parts of a multi-word person ("Marie-Hermeline Le Kervran" -> marie, hermeline)."""
    words = _words_of(name)
    if len(words) < 2:
        return set()
    out: set[str] = set()
    for w in words[:-1]:
        f = fold_word(w)
        if len(w) > 1 and (f in L.titles or f in nm.PARTICLES):
            continue
        for piece in f.split("-"):
            if len(piece) >= 3:
                out.add(piece)
        if f in L.all_first_names:
            break
    return out


def _value_ok(kind: str, label: str, value: str, text: str, e: int, L: Lists) -> bool:
    """Shape checks for identifier-like kinds the NER adds on its own."""
    v = value.strip()
    digits = sum(c.isdigit() for c in v)
    letters = sum(c.isalpha() for c in v)
    if kind == "EMAIL":
        return "@" in v[1:] and "." in v.split("@", 1)[-1]
    if kind == "PHONE":
        return digits >= 6 and letters <= 4
    if kind == "ADDRESS":
        words = _folded_words(v)
        return len(v) >= 8 and (digits > 0 or any(w in _STREET_WORDS for w in words))
    if kind == "BIRTHDATE":
        words = _folded_words(v)
        return digits >= 2 and (digits >= 3 or any(w in L.months for w in words)) and not _YEAR_RE.match(v)
    if kind == "USERNAME":
        return 2 <= len(v) <= 40 and not any(c.isspace() for c in v) and not v.isdigit() and "@" not in v[1:]
    if label in NAMED_ASSET_LABELS:
        words = _folded_words(v)
        return letters >= 3 and bool(words) and v[0].isupper() and not all(L.vocabulary(w) or L.excluded(w) for w in words)
    # EMPLOYEE_ID, ACCOUNT_NUMBER, ID_CODE: a code with digits, never a year, a date, an amount or a short number
    if not 4 <= len(v) <= 40 or any(c.isspace() for c in v.strip()) and digits < 6:
        return False
    if _YEAR_RE.match(v) or _DATE_RE.match(v) or _AMOUNT_AFTER_RE.match(text[e:e + 8]):
        return False
    return digits >= 3 or (letters >= 2 and digits >= 2)


def _guard_name(text: str, s: int, e: int, kind: str, score: float, L: Lists, *, org_initials: set[str],
                person_initials: set[str] = frozenset(), first_tokens: set[str] = frozenset(),
                one_word_person_min: float, one_word_org_min: float, lenient_first_names: bool,
                signals: tuple = (), strong_repeat: bool = False) -> bool:
    """The precision guards of an NER-only PERSON or ORG span. True = keep. ``signals``: the second signals of
    the span (:func:`_name_signals`); ``strong_repeat``: the model found this surface in two chunks at
    ``STRONG_REPEAT_MIN`` or more. They let a company made of vocabulary words through (``_vocab_org_ok``)."""
    seg = text[s:e]
    m = _DOTTED_LEAD_RE.match(seg)
    if m:
        lead = fold_word(m.group(1).replace(".", ""))
        if lead in L.acronyms or lead in L.whitelist_single or lead in L.legal_forms:
            return False  # "R.C.S. Lyon": a registry reference, never a company
    words = _words_of(seg)
    if not words:
        return False
    folded = [fold_word(w) for w in words]
    folded = [f for f in folded if f]
    if not folded or all(L.alone_excluded(f) for f in folded):
        return False
    if seg[0].islower() or not (seg[0].isalpha() or (kind == "ORG" and seg[0].isdigit())):
        return False
    ls, le = _line_bounds(text, s, e)
    line = text[ls:le]
    if _all_caps_line(line) and all(L.vocabulary(f) or L.alone_excluded(f) for f in folded):
        return False
    distinctive = [f for f in folded if not (L.vocabulary(f) or L.alone_excluded(f) or f in L.org_generic or f in L.places)]
    if len(words) == 1:
        w, f = words[0], folded[0]
        if _is_acronym(w):
            if f in L.acronyms:
                return False
            letters = re.sub(r"[^A-Za-zÀ-ÖØ-Þ]", "", w)
            if kind == "PERSON":
                # "PHG" for a person of the text ("Paul-Henri Garnaud", "P.-H. G."): initials, never a stray token
                return letters in person_initials and score >= one_word_person_min
            return letters in org_initials or _defined_term_before(text, s)
        if f in L.places:
            return False
        if kind == "PERSON":
            if f in L.first_names and not L.vocabulary(f):
                return True
            if f in L.all_first_names and lenient_first_names and not L.vocabulary(f):
                return True
            # a nickname that is a prefix of a first name of the text ("Herme" for Marie-Hermeline), 4+ letters
            if len(f) >= 4 and not L.vocabulary(f) and any(t != f and t.startswith(f) for t in first_tokens):
                return True
            cue = _cue_before(text, s, kind, L) or _cue_after(text, e, kind, L) or _signature_line(text, s, e)
            if L.vocabulary(f) and not cue:
                return False
            return cue and score >= one_word_person_min
        # ORG
        if f in L.org_generic or L.vocabulary(f):
            return _cue_before(text, s, kind, L) or _cue_after(text, e, kind, L)
        if w.isupper() and len(w) > 5:
            return score >= one_word_org_min or _cue_before(text, s, kind, L) or _cue_after(text, e, kind, L)
        return score >= one_word_org_min or _cue_before(text, s, kind, L) or _cue_after(text, e, kind, L)
    # two words or more
    if not distinctive:
        if kind == "PERSON":
            # "Jean Petit": a first name plus a common-word surname is a person when both are capitalised
            caps = [w for w in words if w[0].isupper()]
            return len(caps) >= 2 and any(f in L.first_names for f in folded) and not any(f in L.places for f in folded)
        return _vocab_org_ok(words, folded, L, signals, score, strong_repeat)
    if kind == "PERSON":
        if "\n" in seg.strip():
            return False  # "Sam\nAcme Corp": a sign-off name and the next line are never one person
        bad = [w for w, f in zip(words, folded)
               if not (len(w) == 1 and w.isupper())  # a single capital letter is an initial
               and (not (w[0].isupper() or f in nm.PARTICLES) or f in L.legal or f in L.ner_stop_words
                    or f in L.titles or (f in L.stop and f not in nm.PARTICLES))]
        if bad:
            return False
        if all(f in L.places or L.vocabulary(f) for f in folded):
            return False
        return True
    # ORG with a distinctive word: ordinary capitalisation checks; the last word is never a lower-case word of
    # the language ("Acme independent", "Acme center-store": a company name and the sentence that follows it)
    lower = [w for w, f in zip(words, folded) if w[0].islower() and f not in L.connect and f not in _PARTICLES]
    if words[-1][0].islower() and fold_word(words[-1]) not in L.legal_forms:
        return False
    return len(lower) <= len(words) // 2


#: A company name made of vocabulary words only ("General Harvest", "Simple Path Foods") is accepted from the
#: model when a second signal confirms it; the model alone confirms it only at this score in two chunks.
STRONG_REPEAT_MIN = 0.85
_VOCAB_ORG_SIGNALS = frozenset({"cue", "rule", "known", "domain"})


def _vocab_org_ok(words: list[str], folded: list[str], L: Lists, signals: tuple, score: float,
                  strong_repeat: bool) -> bool:
    """A model ORG span of two or more words with no distinctive word (brands and units written with ordinary
    words). Kept when every word is capitalised (connectors apart), none is a stop word, a legal term, a title,
    a greeting or a month, and a second signal confirms it: a strong context cue, the rules or the dossier, an
    e-mail domain, or the model itself at ``STRONG_REPEAT_MIN`` or more in two different chunks."""
    if len(words) < 2 or len(words) > 5:
        return False
    for w, f in zip(words, folded):
        if f in L.connect or f in _PARTICLES:
            continue
        if not w[0].isupper() or f in L.stop or f in L.legal or f in L.titles or f in L.greetings \
                or f in L.months or f in L.ner_stop_words:
            return False
    if not any(w[0].isupper() and f not in L.connect and f not in _PARTICLES for w, f in zip(words, folded)):
        return False
    if set(signals) & _VOCAB_ORG_SIGNALS:
        return True
    return strong_repeat and score >= STRONG_REPEAT_MIN


# ---------------------------------------------------------------------------
# Second signals of the gated layer (2026-10-20): what confirms a model find
# ---------------------------------------------------------------------------

_HANDLE_BODY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{1,39}$")
_PUBLIC_MAIL_LABELS = frozenset({"gmail", "googlemail", "outlook", "hotmail", "live", "yahoo", "icloud", "proton",
                                 "protonmail", "aol", "msn", "mail", "email", "orange", "wanadoo", "free", "laposte",
                                 "sfr", "example", "exemple", "test", "localhost", "www", "mailto", "noreply",
                                 "no-reply", "smtp", "imap"})
# A label word of an account or staff identifier within the same line before a model-found value, or "#".
_ID_CUE_RE = re.compile(
    r"(?i)(?:(?<![\w/])(?:account|accounts|acct|a/c|registration|policy|member|membership|customer|client|loyalty|"
    r"loan|card|routing|aba|plan|ira|employee|employees|emp|staff|badge|worker|personnel|payroll|associate|agent|"
    r"contractor|matricule|compte|salari[ée]e?|adh[ée]rent|identifiant|identifier|id|no|number|num|ref|reference)"
    r"\.?(?![\w])[^\n\d]{0,24}|#[ \t ]*)$")
_ID_PREFIX_RE = re.compile(r"^[A-Z]{1,5}[\-/. ]?\d{4,}")


def _value_key(value: str) -> str:
    return re.sub(r"[^0-9a-z]", "", fold_word(value))


def _handle_shape(text: str, s: int, e: int, L: Lists, person_tokens: frozenset = frozenset()) -> bool:
    """A model "username" is a handle when it looks like one: "@" before it, a digit or a separator inside, or a
    lower-case token that is no first name, no word of the language and no name of a known person."""
    v = text[s:e]
    if not _HANDLE_BODY_RE.match(v) or v.isdigit():
        return False
    low = fold_word(v)
    if low in person_tokens:
        return False
    before = text[max(0, s - 2):s]
    if before.endswith("@") and not (len(before) == 2 and before[0].isalnum()):
        return True
    if any(c.isdigit() or c in "._-" for c in v):
        return True
    return (v.islower() and 3 <= len(v) <= 20 and low not in L.all_first_names and not L.vocabulary(low)
            and not L.excluded(low) and low not in L.ner_stop_words)


def _id_cue(text: str, s: int, e: int) -> bool:
    """A label word of an identifier on the same line before a model-found value ("account", "badge", "#"...)."""
    before = text[max(0, s - 48):s].rsplit("\n", 1)[-1]
    return bool(_ID_CUE_RE.search(before))


def _id_shape(text: str, s: int, e: int) -> bool:
    """A letter-prefixed code ("ACC-00917"): the shape alone lets a model identifier through, unconfirmed."""
    return bool(_ID_PREFIX_RE.match(text[s:e]))


def _domain_labels(text: str, struct: Sequence[tuple[int, int, str]]) -> frozenset[str]:
    """Folded labels of the e-mail domains and URL hosts of the text (public mail hosts and the TLD left out)."""
    out: set[str] = set()
    for s, e, k in struct:
        if k not in ("EMAIL", "URL"):
            continue
        v = text[s:e].strip().lower()
        if k == "EMAIL":
            host = v.rsplit("@", 1)[-1]
        else:
            host = re.sub(r"^[a-z][a-z0-9+.\-]*://", "", v).split("/", 1)[0]
        parts = [p for p in host.split(":")[0].split(".") if p]
        for p in (parts[:-1] if len(parts) > 1 else parts):
            f = fold_word(p)
            if len(f) >= 4 and f not in _PUBLIC_MAIL_LABELS and not f.isdigit():
                out.add(f)
    return frozenset(out)


def _chunk_index(offsets: Sequence[int], pos: int) -> int:
    lo, hi = 0, len(offsets)
    while lo < hi:
        mid = (lo + hi) // 2
        if offsets[mid] <= pos:
            lo = mid + 1
        else:
            hi = mid
    return max(0, lo - 1)


class _NameIndex:
    """Surfaces, surnames, first names and head words of the rules' names and of the dossier's known names."""

    def __init__(self, text: str, rules: Sequence[tuple[int, int, str]], known_values: Iterable[tuple[str, str]],
                 L: Lists):
        self.L = L
        self.rule_surf: dict[str, set[str]] = {"PERSON": set(), "ORG": set()}
        self.known_surf: dict[str, set[str]] = {"PERSON": set(), "ORG": set()}
        self.rule_parts: set[str] = set()     # surname and first-name tokens of the rules' full names
        self.known_parts: set[str] = set()
        self.rule_heads: set[str] = set()     # first distinctive word of the rules' organisations
        self.known_heads: set[str] = set()
        tokens_: set[str] = set()
        for s, e, k in rules:
            if k in NAME_KINDS:
                self._add(k, _folded_words(text[s:e]), self.rule_surf, self.rule_parts, self.rule_heads, tokens_)
        for k, v in known_values or ():
            if k in NAME_KINDS and v:
                self._add(k, _folded_words(v), self.known_surf, self.known_parts, self.known_heads, tokens_)
        self.person_tokens = frozenset(tokens_)

    def _add(self, kind: str, words: list[str], surf: dict, parts: set, heads: set, tokens_: set) -> None:
        if not words:
            return
        surf[kind].add(" ".join(words))
        if kind == "PERSON":
            tokens_.update(w for w in words if len(w) >= 3)
            if len(words) >= 2:
                parts.update(w for w in (words[0], words[-1]) if len(w) >= 3 and w not in nm.PARTICLES)
        else:
            h = self.head(words)
            if h:
                heads.add(h)

    def head(self, words: list[str]) -> str:
        L = self.L
        for w in words:
            if len(w) >= 4 and not (L.vocabulary(w) or L.alone_excluded(w) or w in L.org_generic or w in L.places):
                return w
        return ""


# Words that confirm a company before its name (a bare preposition such as "at", "with" or "from" fits a person
# as well as a company and is not a confirmation; the guard still accepts it as a cue for a one-word find).
_STRONG_ORG_CUES = frozenset({"chez", "societe", "ste", "groupe", "cabinet", "entreprise", "company", "firm",
                              "client", "vendor", "partner", "supplier", "customer", "fournisseur", "prestataire",
                              "partenaire", "joined", "acquired", "sold", "employer", "dealer", "carrier",
                              "insurer", "bank", "agency", "startup", "manufacturer", "retailer", "distributor"})


def _strong_cue(text: str, s: int, e: int, kind: str, L: Lists) -> bool:
    """A context cue strong enough to confirm a model find on its own: for a person a title, a greeting, a role
    word, a nickname declaration, an @-mention, a sign-off line or ", role" after it; for a company a legal form
    after it or a company word before it. Bare prepositions ("at X", "with X") confirm nothing."""
    if kind == "PERSON":
        before = text[max(0, s - 40):s]
        if _TITLE_DOT_BEFORE_RE.search(before):
            return True
        if _NICK_CUE_RE.search(text[max(0, s - 60):s]):
            return True
        tail = before.rstrip(" \t\u00a0 :,;-–—(\"«“'")
        if before.endswith(("@", "<@")) or tail.endswith("@"):
            return True
        last_cut = 0
        for m in _SENTENCE_CUT_RE.finditer(before):
            last_cut = m.end()
        found = re.findall(r"[^\W\d_][\w'’\-/.]*", before[last_cut:].rstrip(" \t\u00a0 :,;-–—(\"«“'"))
        w = fold_word(found[-1].rstrip(".")) if found else ""
        roles = _FR_ROLE_CUES if L.lang == "fr" else _EN_ROLE_CUES
        if w and (w in L.titles or w in L.greetings or w in nm._PLURAL_TITLES or w in roles):
            return True
        m = re.match(r"[\s ]*,[\s ]*([^\W\d_][\w'’\-]*)", text[e:e + 24])
        if m and (fold_word(m.group(1)) in L.legal or fold_word(m.group(1)) in nm._ROLE_WORDS):
            return True
        return _signature_line(text, s, e)
    before = text[max(0, s - 40):s]
    last_cut = 0
    for m in _SENTENCE_CUT_RE.finditer(before):
        last_cut = m.end()
    found = re.findall(r"[^\W\d_][\w'’\-/.]*", before[last_cut:].rstrip(" \t\u00a0 :,;-–—(\"«“'"))
    w = fold_word(found[-1].rstrip(".")) if found else ""
    if w and (w in _STRONG_ORG_CUES or w in nm.ORG_TRIGGERS):
        return True
    m = _TOKEN_RE.match(text[e:e + 24].lstrip(" \u00a0"))
    return bool(m and _legal_form_here(m.group(), L))


def _name_signals(text: str, s: int, e: int, kind: str, L: Lists, idx: _NameIndex, domains: frozenset[str],
                  sightings: dict[tuple[str, str], set[int]]) -> tuple[str, ...]:
    """The second signals that confirm a model-only person or organisation span (see the module docstring).
    A span with no letter word (digits, signs) gets no signal (v0.3.1: it made ``matches`` read ``words[0]``)."""
    words = _folded_words(text[s:e])
    if not words:
        return ()
    surf = " ".join(words)
    out: list[str] = []
    if _strong_cue(text, s, e, kind, L):
        out.append(SIGNAL_CUE)
    head = idx.head(words) if kind == "ORG" else ""

    def matches(surfaces: dict, parts: set, heads: set) -> bool:
        if surf in surfaces[kind]:
            return True
        if kind == "PERSON":
            return any(w in parts for w in (words[0], words[-1]) if len(w) >= 3)
        return bool(head) and head in heads

    if matches(idx.rule_surf, idx.rule_parts, idx.rule_heads):
        out.append(SIGNAL_RULE)
    if len(sightings.get((kind, surf), ())) >= 2:
        out.append(SIGNAL_REPEAT)
    if kind == "ORG" and domains and ((head and head in domains) or "".join(words) in domains):
        out.append(SIGNAL_DOMAIN)
    if matches(idx.known_surf, idx.known_parts, idx.known_heads):
        out.append(SIGNAL_KNOWN)
    return tuple(out)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def _as_triples(items: Iterable) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    for it in items or ():
        if isinstance(it, tuple):
            s, e, k = it[0], it[1], it[2]
        else:
            s, e, k = it.start, it.end, it.kind
        if isinstance(s, int) and isinstance(e, int) and e > s:
            out.append((s, e, str(k)))
    out.sort()
    return out


def _overlapping(triples: Sequence[tuple[int, int, str]], s: int, e: int) -> list[tuple[int, int, str]]:
    return [t for t in triples if t[0] < e and t[1] > s]


def merge_ner(text: str, ner_spans: Iterable[tuple[int, int, str, str, float]], rule_spans: Iterable = (),
              structured: Iterable = (), protected: Optional[IntervalSet] = None, *, lang: str = "fr",
              kinds: Optional[Iterable[str]] = None, org_acronyms: Iterable[str] = (),
              one_word_person_min: float = 0.50, one_word_org_min: float = 0.70,
              lenient_first_names: bool = False, known_values: Iterable[tuple[str, str]] = (),
              model_kinds: Optional[Iterable[str]] = None, repeat_min: float = 0.60) -> list[MergedSpan]:
    """Merge the NER output into candidates for the masker (rules 1 to 8 of the module docstring).

    ``ner_spans``: ``run_ner`` output. ``rule_spans`` / ``structured``: the masker's rule and regex candidates
    (anything with ``start``, ``end``, ``kind`` or ``(start, end, kind)`` tuples). ``protected``: the
    ``names.protected_intervals`` of the text, computed here when None. ``org_acronyms``: acronyms of the
    registered organisations (an all-capital ORG span equal to one is accepted). ``kinds``: keep these kinds only
    (an explicit list replaces the policy). ``model_kinds``: the kinds a model find may add on its own when
    ``kinds`` is None (default ``MODEL_KINDS``). ``known_values``: ``(kind, value)`` of the entities the dossier
    already holds from the rules or the user (the "known" signal). ``repeat_min``: score two sightings of one value
    in two different chunks must both reach for the "repeat" signal.
    """
    if not text:
        return []
    L = lists_for(lang)
    allowed = frozenset(model_kinds) if model_kinds is not None else MODEL_KINDS
    want = set(kinds) if kinds is not None else set(allowed)
    rules = _as_triples(rule_spans)
    struct = _as_triples(structured)
    ner_spans = list(ner_spans)
    if protected is None:
        folded = fold(text)
        protected = nm.protected_intervals(text, folded, tokens(folded.text))
    # initials of the multi-word organisations of this text (rules first, then NER): "SBTR" after "Société B... T... R..."
    org_initials: set[str] = {a.upper() for a in org_acronyms if a}
    person_initials: set[str] = set()
    first_tokens: set[str] = set()
    for s, e, k in rules:
        if k == "ORG" and len(_words_of(text[s:e])) >= 2:
            org_initials |= _org_initials_at(text, s, e, L)
        elif k == "PERSON":
            person_initials |= _person_initials(text[s:e], L)
            first_tokens |= _first_tokens(text[s:e], L)
    # second signals of the gated layer: the rules' names, the dossier's names, the e-mail domains, and the
    # model's own repeated sightings (one value in two different chunks)
    idx = _NameIndex(text, rules, known_values, L)
    domains = _domain_labels(text, struct)
    known_vals = {(k, _value_key(v)) for k, v in known_values or () if k not in NAME_KINDS}
    chunk_offsets = [off for off, _c in nr.chunk_text(text)]
    sightings: dict[tuple[str, str], dict[int, float]] = {}   # (kind, surface) -> chunk -> best score
    for s, e, k, _label, score in ner_spans:
        if k not in NAME_KINDS or not (0 <= s < e <= len(text)) or protected.overlaps(s, e):
            continue
        ts, te = trim_span(text, s, e, k, lang, lists=L)
        if te <= ts:
            continue
        if k == "ORG" and len(_words_of(text[ts:te])) >= 2:
            org_initials |= _org_initials_at(text, ts, te, L)
        elif k == "PERSON" and len(_words_of(text[ts:te])) >= 2:
            person_initials |= _person_initials(text[ts:te], L)
            first_tokens |= _first_tokens(text[ts:te], L)
        if score >= repeat_min:
            key = (k, " ".join(_folded_words(text[ts:te])))
            ci = _chunk_index(chunk_offsets, ts)
            per = sightings.setdefault(key, {})
            per[ci] = max(per.get(ci, 0.0), float(score))

    kept: list[MergedSpan] = []
    for s, e, kind, label, score in ner_spans:
        if not (0 <= s < e <= len(text)) or kind not in want:
            continue
        s, e = trim_span(text, s, e, kind, lang, lists=L)
        if e - s < 2:
            continue
        if _overlapping(struct, s, e) or (kind != "BIRTHDATE" and protected.overlaps(s, e)):
            continue  # a date is made of a month word, which the whitelist keeps clear
        hits = _overlapping(rules, s, e)
        if hits:
            exact = [h for h in hits if h[0] == s and h[1] == e]
            if exact:
                continue  # the rule already covers it (rules win on kind)
            inner = [h for h in hits if h[0] >= s and h[1] <= e]
            if len(hits) == 1 and inner and kind in NAME_KINDS and hits[0][2] in NAME_KINDS:
                rs, re_, rk = hits[0]
                extra = _words_of(text[s:rs]) + _words_of(text[re_:e])
                if extra and all(is_name_word(w, lang, rk, lists=L) for w in extra) \
                        and not (rk == "PERSON" and "\n" in text[s:e].strip()):
                    kept.append(MergedSpan(s, e, rk, label, score, "rule+ner", (SIGNAL_RULE,)))
            continue  # a rule span inside, partly overlapping or covering it: the rule stays
        if kind in NAME_KINDS:
            signals = _name_signals(text, s, e, kind, L, idx, domains, sightings)
            strong = sum(1 for v in sightings.get((kind, " ".join(_folded_words(text[s:e]))), {}).values()
                         if v >= STRONG_REPEAT_MIN) >= 2
            if not _guard_name(text, s, e, kind, score, L, org_initials=org_initials,
                               person_initials=person_initials, first_tokens=first_tokens,
                               one_word_person_min=one_word_person_min, one_word_org_min=one_word_org_min,
                               lenient_first_names=lenient_first_names, signals=signals, strong_repeat=strong):
                continue
        elif not _value_ok(kind, label, text[s:e], text, e, L):
            continue
        elif kind == "USERNAME" and not _handle_shape(text, s, e, L, idx.person_tokens):
            continue  # a model "username" that is a plain name is a person's name, never a handle
        elif kind in ("EMPLOYEE_ID", "ACCOUNT_NUMBER") and not (_id_cue(text, s, e) or _id_shape(text, s, e)):
            continue  # an identifier needs a label word before it or a letter-prefixed code shape
        else:
            signals = ()
            if kind in ("USERNAME", "EMPLOYEE_ID", "ACCOUNT_NUMBER") and _id_cue(text, s, e):
                signals += (SIGNAL_CUE,)
            if (kind, _value_key(text[s:e])) in known_vals:
                signals += (SIGNAL_KNOWN,)
        kept.append(MergedSpan(s, e, kind, label, score, "ner", signals))
    # collapse nested and overlapping spans: the longest wins, then the best score
    kept.sort(key=lambda m: (-(m.end - m.start), -m.score, m.start))
    taken = IntervalSet()
    out: list[MergedSpan] = []
    for m in kept:
        if taken.overlaps(m.start, m.end):
            continue
        taken.add(m.start, m.end)
        out.append(m)
    out.sort(key=lambda m: (m.start, m.end))
    return out


def guard_ner_spans(text: str, ner_spans: Iterable[tuple[int, int, str, str, float]], *, lang: str = "fr",
                    kinds: Optional[Iterable[str]] = None, **kw) -> list[MergedSpan]:
    """The NER-only guards without rule or structured spans (the ``Masker(ner=...)`` hook path).

    Every kind the model can name passes here (``ALL_MODEL_KINDS``): the hook already chooses its kinds through
    ``as_hook(kinds=...)``; ``model_kinds=`` in ``kw`` narrows it."""
    kw.setdefault("model_kinds", ALL_MODEL_KINDS)
    return merge_ner(text, list(ner_spans), (), (), None, lang=lang, kinds=kinds, **kw)


def watch_ner(text: str, ner_spans: Iterable[tuple[int, int, str, str, float]], rule_spans: Iterable = (),
              protected: Optional[IntervalSet] = None, *, lang: str = "fr",
              model_kinds: Optional[Iterable[str]] = None) -> list[WatchSpan]:
    """Model finds the merge never masks on its own, for the review step: named assets (vessel, building,
    school: ``ner.WATCH_LABELS``) and every find of a kind outside ``model_kinds`` (default ``MODEL_KINDS``).

    Trimmed like a candidate, never over a protected interval or a rule span, shape-checked like a value. A
    person decides on them (register or keep clear); nothing here is masked without that decision."""
    if not text:
        return []
    L = lists_for(lang)
    allowed = frozenset(model_kinds) if model_kinds is not None else MODEL_KINDS
    rules = _as_triples(rule_spans)
    if protected is None:
        folded = fold(text)
        protected = nm.protected_intervals(text, folded, tokens(folded.text))
    out: list[WatchSpan] = []
    seen: set[tuple[int, int]] = set()
    for s, e, kind, label, score in ner_spans:
        if not (0 <= s < e <= len(text)) or (kind in allowed and label not in nr.WATCH_LABELS):
            continue
        s, e = trim_span(text, s, e, kind, lang, lists=L)
        if e - s < 2 or (s, e) in seen or protected.overlaps(s, e) or _overlapping(rules, s, e):
            continue
        if label in nr.WATCH_LABELS or kind in NAME_KINDS:
            words = _folded_words(text[s:e])
            if not words or not text[s].isupper() or all(L.vocabulary(w) or L.alone_excluded(w) for w in words):
                continue
        elif not _value_ok(kind, label, text[s:e], text, e, L):
            continue
        seen.add((s, e))
        out.append(WatchSpan(s, e, kind, label, float(score)))
    out.sort(key=lambda w: (w.start, w.end))
    return out
