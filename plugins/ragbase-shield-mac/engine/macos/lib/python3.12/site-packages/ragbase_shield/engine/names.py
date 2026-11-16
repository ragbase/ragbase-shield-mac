"""Person and company names: rule detection, variants of known names, dossier titles, whitelist.

Rules (no model):

- titles: ``M.``, ``Mme``, ``Me``, ``Maître``, ``Monsieur``, ``Docteur``... + name words;
- a first name from the INSEE lexicon + a capitalised or upper-case surname, and the
  ``SURNAME Firstname`` order;
- roles: "le salarié DURAND", "les époux MARTIN", "née VASSEUR"; and "NAME, demeurant / né le";
- companies: legal forms before or after the name (``ACME SAS``, ``SCI Les Tilleuls``), trigger words
  ("la société X", "groupe X", "cabinet X"), dossier titles ``X c. Y``.

The legal form and the title are not part of the masked span: "M. PERSONNE_A", "SOCIETE_A SAS".
Every rule span then goes through the whitelist (courts with their city, institutions, laws).

v0.3 (owner S1+S4, spans and organisations), for both languages:

- the legal-form span policy (``legal_form_policy``: "clear", "cover", "mask") and the span refinement pass
  ``refine_rule_spans``: legal form inside the company span, company span edges over head words, companies
  named by an e-mail domain of the text, acronyms the text declares ("Full Name (FN)"), same-surface
  propagation of multi-word company names;
- a quoted or bracketed word inside a person name is a nickname (``person_nicknames``), with its own token
  role (``person_token_roles``, ``surface_roles``) and its own dictionary variants.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Iterable, Optional

from . import lexicon as lx
from .detectors import IntervalSet
from .normalize import Folded, PhraseMatcher, fold, fold_word, phrase_tokens, tokens

__all__ = [
    "detect_names",
    "parse_parties",
    "split_person",
    "person_norm",
    "org_norm",
    "org_core",
    "person_variants",
    "org_variants",
    "Variant",
    "protected_intervals",
    "is_title_word",
    "first_key",
    "LEGAL_FORMS",
    "ORG_GENERIC",
    "ORG_TRADES",
    "ORG_TRIGGERS",
    "TITLE_WORDS",
    "PARTICLES",
    "org_acronyms",
    "person_initials",
    "GENERATIONAL",
    # v0.3 (S1+S4): token roles, nicknames, the legal-form span policy and the span refinement pass
    "person_nicknames",
    "person_token_roles",
    "surface_roles",
    "legal_form_policy",
    "legal_form_after",
    "legal_form_before",
    "org_form_parts",
    "refine_rule_spans",
    "org_candidates_from_domain",
    "domain_org_spans",
    "domain_form_spans",
    "STRONG_ORG_RULES",
]

# ---------------------------------------------------------------------------
# Word lists defined in code (closed lists)
# ---------------------------------------------------------------------------

TITLE_WORDS = frozenset({
    "monsieur", "madame", "mademoiselle", "messieurs", "mesdames", "mesdemoiselles", "maitre",
    "maitres", "docteur", "professeur", "m", "mm", "mr", "mrs", "ms", "mme", "mmes", "mlle",
    "mlles", "me", "mes", "dr", "pr",
})
_TITLE_FULL = frozenset({
    "monsieur", "madame", "mademoiselle", "messieurs", "mesdames", "mesdemoiselles", "maitre",
    "maitres", "docteur", "professeur",
})

PARTICLES = frozenset({
    "de", "du", "des", "d", "van", "von", "der", "den", "ten", "ter", "di", "da", "del", "della",
    "dos", "das", "le", "la", "ben", "bin", "el", "al", "zu", "y",
})
_SURNAME_DROP_PARTICLES = frozenset({"de", "du", "des", "d", "van", "von", "der", "den", "di", "da", "del", "zu"})
_ARTICLES = frozenset({"le", "la", "les", "l"})
_ORG_CONNECT = frozenset({"et", "&", "de", "du", "des", "d", "la", "le", "les", "l", "en", "sur", "sous", "aux", "au"})
# "et" joins two words of one company name only before these words ("Martin et Fils").
_ORG_ET_JOIN = frozenset({"fils", "filles", "associes", "associe", "associees", "freres", "soeurs", "cie",
                          "compagnie", "partenaires", "co", "consorts", "neveux"})

# folded key -> "strong" / "weak". Weak forms are short and ambiguous (SA = "sa", AG = assemblée générale).
LEGAL_FORMS = {
    "sas": "strong", "sasu": "strong", "sarl": "strong", "eurl": "strong", "sci": "strong",
    "snc": "strong", "gie": "strong", "selarl": "strong", "selas": "strong", "selasu": "strong",
    "selafa": "strong", "selca": "strong", "selurl": "strong", "scp": "strong", "sep": "weak",
    "sca": "strong", "scs": "strong", "scm": "strong", "scop": "strong", "scic": "strong", "aarpi": "strong",
    "spfpl": "strong", "selafa.": "strong",
    # foreign and investment fund forms (R1, CL-22): "X Fund II SCSp", "FCPR X"
    "scsp": "strong", "slp": "strong", "fcpr": "strong", "fpci": "strong", "fcpi": "strong", "fcp": "weak",
    "sem": "weak", "spl": "strong", "sicav": "strong", "scpi": "strong", "spa": "strong",
    "srl": "strong", "sarl.": "strong", "ltd": "strong", "limited": "strong", "llc": "strong",
    "llp": "strong", "inc": "strong", "corp": "strong", "gmbh": "strong", "plc": "strong",
    "asbl": "strong", "sprl": "strong", "sa": "weak", "sl": "weak", "ag": "weak", "kg": "weak",
    "bv": "weak", "nv": "weak", "ab": "weak", "oy": "weak", "sc": "weak", "co": "weak",
    # v0.3 (S1+S4): the spelled English forms, so that ``org_core`` strips them from a value stored with its form
    "corporation": "strong", "incorporated": "strong", "company": "weak",
}
# Mixed-case spellings accepted for legal forms (others must be upper case).
_LEGAL_MIXED = frozenset({"Sarl", "Sas", "Sasu", "Eurl", "Sàrl", "SpA", "Ltd", "Limited", "Inc", "Corp",
                          "GmbH", "Plc", "Oy", "Co", "Srl", "Selarl", "Selas", "Sci", "Snc", "Llc", "Llp", "Aarpi",
                          "Spfpl", "Selafa", "Selca", "Scp", "Scm", "SCSp", "Scsp",
                          "Corporation", "Incorporated", "Company"})

ORG_TRIGGERS = frozenset({
    "societe", "ste", "groupe", "cabinet", "etablissements", "ets", "entreprise", "enseigne",
    "association", "fondation", "syndicat", "compagnie", "holding", "mutuelle", "banque",
    "denommee", "denomination",
})
# Lower-case words that may sit between a trigger and the company name.
_ORG_SKIP = frozenset({
    "civile", "immobiliere", "anonyme", "cooperative", "d", "exercice", "liberal", "liberale", "a",
    "responsabilite", "limitee", "par", "actions", "simplifiee", "unipersonnelle", "en", "nom",
    "collectif", "commandite", "de", "droit", "francais", "francaise", "etrangere", "dite",
    "denommee", "ci-apres", "avocats", "avocat", "commerciale", "sociale", "professionnelle",
    "interprofessionnelle", "d'avocats", "sous", "l", "enseigne", "la", "nouvelle", ":",
})
_ROLE_WORDS = frozenset({
    "salarie", "salariee", "gerant", "gerante", "cogerant", "cogerante", "demandeur",
    "demanderesse", "defendeur", "defenderesse", "requerant", "requerante", "appelant", "appelante",
    "intime", "intimee", "employe", "employee", "associe", "associee", "liquidateur", "bailleur",
    "bailleresse", "locataire", "preneur", "preneuse", "acquereur", "vendeur", "vendeuse",
    "creancier", "creanciere", "debiteur", "debitrice", "temoin", "epoux", "epouse", "veuve", "veuf",
    "nee", "sieur", "dame", "consorts", "consort", "client", "cliente", "confrere", "consoeur",
    "collegue", "stagiaire", "apprenti", "apprentie", "mandataire", "beneficiaire", "heritier",
    "heritiere", "defunt", "defunte", "enfant", "fils", "fille", "pere", "mere", "frere", "soeur",
    "conjoint", "conjointe", "concubin", "concubine", "victime", "prevenu", "prevenue", "plaignant",
    "plaignante", "dit", "alias",
})
# Words before a first name that make it a place or an institution, not a person.
_PLACE_WORDS = frozenset({
    "rue", "avenue", "boulevard", "bd", "allee", "impasse", "place", "quai", "square", "chemin",
    "route", "cours", "passage", "pont", "gare", "lycee", "college", "ecole", "hopital", "clinique",
    "stade", "centre", "fondation", "prix", "salle", "espace", "theatre", "musee", "institut",
    "universite", "gymnase", "piscine", "mediatheque", "bibliotheque", "residence", "cite",
    "aeroport", "port", "station", "metro", "promenade", "esplanade", "parc", "jardin", "eglise",
    "cathedrale", "basilique", "saint", "sainte", "st", "ste",
})
# Role words that may be followed by a surname with a particle ("née de Villeroy"); the others read "de" as a genitive.
_ROLE_NAME_PARTICLE = frozenset({"nee", "epouse", "epoux", "veuve", "veuf", "sieur", "dame", "consorts", "consort", "dit",
                                 "alias"})
# Trigger words that define a term: followed by an article they introduce a defined term ("dénommée « la Société »").
_DEFINING = frozenset({"denommee", "denomination"})
_PLURAL_TITLES = frozenset({"mm", "messieurs", "mmes", "mesdames", "mes", "maitres", "mlles", "mesdemoiselles"})
_PLACE_PREPOSITIONS = frozenset({"a", "au", "aux", "en", "de", "du", "des", "sur", "vers", "pres", "dans", "lez", "les"})
# A capitalised company name followed by one of these words is a company ("LERVIGNAC AVOCATS").
_ORG_SUFFIXES = frozenset({
    "avocats", "associes", "partners", "conseil", "conseils", "consulting", "immobilier",
    "logistique", "transports", "holding", "group", "groupe", "finance", "gestion", "patrimoine",
    "industries", "technologies", "services", "solutions", "systemes", "distribution",
    "construction", "constructions", "batiment", "promotion", "investissements", "participations",
    "developpement", "audit", "expertise", "assurances", "courtage", "notaires", "huissiers",
    "laboratoires", "pharma", "energie", "energies", "environnement", "medias", "editions",
    "international", "france", "europe", "capital", "ventures", "invest", "fund", "funds", "trust",
})
_ABBREV = frozenset({
    "m", "mm", "mr", "mrs", "ms", "mme", "mmes", "mlle", "mlles", "me", "mes", "dr", "pr", "st",
    "ste", "ets", "inc", "corp", "ltd", "co", "av", "bd", "etc", "cf", "art", "al", "n", "no",
    "p", "pp", "ch", "civ", "soc", "com", "crim", "cass", "req", "ord", "sect", "vol", "ed",
    "coll", "cie", "sa", "spa", "ref", "tel", "fax", "vs", "v", "c", "s", "dir", "gal", "mal",
})
_ORG_KEYWORDS = frozenset({
    "groupe", "group", "immobilier", "immobiliere", "banque", "assurances", "assurance", "transports",
    "transport", "logistique", "conseil", "conseils", "holding", "industries", "industrie",
    "international", "services", "service", "technologies", "technology", "syndicat", "association",
    "mutuelle", "caisse", "fondation", "compagnie", "construction", "constructions", "batiment",
    "batiments", "distribution", "developpement", "finance", "finances", "gestion",
    "investissements", "patrimoine", "promotion", "participations", "partners", "capital",
    "societe", "ste", "cabinet", "avocats", "laboratoires", "laboratoire", "pharma", "energie",
    "solutions", "systems", "systemes", "consulting", "media", "medias", "editions", "agence",
    "france", "europe", "office", "hotel", "restaurant", "garage", "atelier", "ateliers", "imprimerie",
    "copropriete", "coproprietaires", "residence", "sci", "scp", "selarl", "fils", "freres",
    "associes", "fund", "funds", "trust",
})
# Trade head nouns: "Garage Gravelotte", "TRANSPORTS GRAVELOTTE", "Pharmacie Rouvière". The noun stays in clear
# (like a legal form) and the name after it is the company.
def _with_plurals(words: set) -> frozenset:
    """A trade noun and its plural ("brasserie", "brasseries")."""
    return frozenset(words | {w + "s" for w in words if w[-1:] not in ("s", "x", "z")})


ORG_TRADES = _with_plurals({
    "transports", "transport", "garage", "garages", "boulangerie", "patisserie", "pharmacie", "laboratoire",
    "laboratoires", "imprimerie", "hotel", "clinique", "menuiserie", "restaurant", "agence", "boucherie",
    "charcuterie", "librairie", "quincaillerie", "plomberie", "carrosserie", "traiteur", "brasserie",
    "epicerie", "fromagerie", "ambulances", "taxis", "pressing", "atelier", "ateliers", "maison",
    "menuiseries", "constructions", "chaussures", "auto-ecole", "cabinet",
} | set(lx.trade_nouns()))
# Generic company words: never the distinctive part of a company name on their own.
# Generic business adjectives and nouns ("Financière X", "Industrielle Y") are never a head word either (R1, CL-18d).
ORG_GENERIC = frozenset(_ORG_SUFFIXES | _ORG_KEYWORDS | ORG_TRADES | {"ets", "etablissements", "societe", "ste"}
                        | lx.business_words())
# Professional partnership forms: their name joins partners with "et", "&" or "," (R1, CL-09).
_PARTNERSHIP_FORMS = frozenset({"scp", "selarl", "selas", "selasu", "selafa", "selca", "selurl", "aarpi", "scm",
                                "spfpl", "sel"})
# Generational qualifiers: "Vellin fils" is not "Vellin père" (R1, CL-20).
GENERATIONAL = frozenset({"fils", "pere", "junior", "senior", "aine", "cadet", "jr", "sr"})
_GREETINGS = frozenset({"bonjour", "bonsoir", "cher", "chere", "chers", "cheres", "salut", "merci", "coucou",
                        "hello"})

_SPACE_GAP = re.compile(r"[ \t\u00a0\u202f«»\"“”]*")
# A tab or three spaces in a row separate columns (signature blocks, tables): a name never spans them.
_COLUMN_GAP = re.compile(r"\t|[ \u00a0\u202f]{3,}")
_MARKS = "\u0300-\u036f\u1ab0-\u1aff\u1dc0-\u1dff\u20d0-\u20ff\ufe20-\ufe2f"   # combining accents (NFD text)
_INVISIBLE = "\u00ad\u200b\u200c\u200d\u2060\ufeff"  # soft hyphen, zero-width characters
_HYPHENS = "\\-\u2010\u2011\u2012\u2212"                 # "-", hyphen, non-breaking hyphen, figure dash, minus
_TOKEN_RE = re.compile(r"[^\W\d_][" + _MARKS + r"]*(?:[^\W_]|[" + _MARKS + r"]|['’" + _HYPHENS + r"](?=[^\W\d_])"
                       r"|[" + _INVISIBLE + r"]+(?=[^\W_]))*\.?|\d+|&")
_POSSESSIVE_RE = re.compile(r"['’][sS]$")
_LINE_GAP = re.compile(r"[ \t\u00a0\u202f]*\r?\n[ \t\u00a0\u202f]*")
_PREFIX_RE = re.compile(r"^(?:[dlnjmtsc]|qu|jusqu|lorsqu|puisqu|presqu)['’](?=.)")


@lru_cache(maxsize=None)
def _lists() -> tuple[frozenset, frozenset, frozenset, frozenset, frozenset, frozenset, frozenset]:
    wl_single = frozenset(p for p in lx.whitelist_phrases() if " " not in p)
    return (lx.stopwords(), lx.legal_terms(), lx.common_words(), lx.first_names(),
            lx.all_first_names(), lx.months_days(), wl_single)


# ---------------------------------------------------------------------------
# Word tokens with case information
# ---------------------------------------------------------------------------


@dataclass
class _W:
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
    prefix: bool = False  # "d'", "l'" split from the next word


def _make_word(text: str, start: int, end: int) -> _W:
    raw = text
    dot = raw.endswith(".")
    base = raw[:-1] if dot else raw
    low = fold_word(base).replace(".", "")
    if base.isdigit():
        return _W(raw, start, end, low, False, False, digit=True)
    first = base[:1]
    cap = first.isupper()
    letters = [c for c in base if c.isalpha()]
    upper = len(letters) >= 2 and all(c.isupper() for c in letters)
    w = _W(raw, start, end, low, cap, upper)
    if dot:
        if len(letters) == 1 and cap:
            w.initial = True
            w.dot = True
        elif low in _ABBREV:
            w.dot = True
        else:
            w.sent_end = True
            w.text = base
            w.end = end - 1
    return w


def _words(text: str) -> list[_W]:
    out: list[_W] = []
    for m in _TOKEN_RE.finditer(text):
        raw, s, e = m.group(), m.start(), m.end()
        if _POSSESSIVE_RE.search(raw) and len(raw) > 4 and not (e < len(text) and text[e].isalnum()):
            raw, e = raw[:-2], e - 2  # English possessive: "Kervadec's" is the name "Kervadec"
        pm = _PREFIX_RE.match(raw)
        if pm and len(raw) > pm.end():
            out.append(_W(raw[:pm.end()], s, s + pm.end(), fold_word(raw[:pm.end() - 1]),
                          False, False, prefix=True))
            raw, s = raw[pm.end():], s + pm.end()
        out.append(_make_word(raw, s, e))
    # Merge initials written without spaces: "S.A.S." (legal form) or "J.-P." / "J.P." (initials).
    merged: list[_W] = []
    for w in out:
        if merged:
            prev = merged[-1]
            gap = text[prev.end:w.start]
            if prev.initial and w.initial and gap in ("", "-"):
                joined = text[prev.start:w.end]
                low = prev.low + w.low
                merged[-1] = _W(joined, prev.start, w.end, low, True, False, initial=True, dot=True)
                continue
        merged.append(w)
    for w in merged:
        if w.initial and len(w.low) > 1 and w.low in LEGAL_FORMS:
            w.initial = False
            w.upper = True
    return merged


def _line_join(prev: _W, w: _W, after: list[_W]) -> bool:
    """A name that goes on after a line break: "M.\nDupont", "M. Jean\nMOREL-DUVAL" (PDF text). A signature
    ("Jean-Pierre\nBrivanor Logistique") is not joined: after a first name only an upper-case surname is."""
    if w.digit or not w.cap or w.prefix or _excluded(w.low) or _legal_form(w):
        return False
    if any(x.low in ORG_GENERIC or _legal_form(x) for x in [w] + after):
        return False
    if _title(prev):
        # "... 150 M\nHoraire mensuel": a bare "M" without a dot is a title only when the name is on its line
        return not _bare_m(prev)
    return _is_first_name(prev) and w.upper


def _runs(text: str, words: list[_W]) -> list[list[_W]]:
    runs: list[list[_W]] = []
    cur: list[_W] = []
    for k, w in enumerate(words):
        if cur:
            prev = cur[-1]
            gap = text[prev.end:w.start]
            same_line = bool(_SPACE_GAP.fullmatch(gap) and (gap or prev.prefix or prev.dot)) and not _COLUMN_GAP.search(gap)
            next_line = not same_line and not prev.sent_end and bool(_LINE_GAP.fullmatch(gap)) and \
                _line_join(prev, w, words[k + 1:k + 4])
            if prev.sent_end or not (same_line or next_line):
                runs.append(cur)
                cur = []
        cur.append(w)
    if cur:
        runs.append(cur)
    return runs


# ---------------------------------------------------------------------------
# Word predicates
# ---------------------------------------------------------------------------


def is_title_word(surface: str) -> bool:
    """True for a person title with the right case ("M.", "Mme", "Me", "Maître", "monsieur")."""
    base = surface.rstrip(".")
    low = fold_word(base)
    if low in _TITLE_FULL:
        return True
    if low not in TITLE_WORDS or not base[:1].isupper():
        return False
    if low in ("m", "mm") and base not in ("M", "MM"):
        return False
    return True


def _title(w: _W) -> bool:
    if w.initial and w.low == "m":
        return True  # "M." is read as an initial by the tokenizer
    if w.low not in TITLE_WORDS:
        return False
    return is_title_word(w.text)


def _bare_m(w: _W) -> bool:
    """An undotted "M" or "MM": also a unit or an abbreviation ("150 M", "3 M d'euros")."""
    return w.low in ("m", "mm") and not w.dot and not w.initial


@lru_cache(maxsize=None)
def _instruments() -> frozenset:
    return lx.instrument_terms()


def _instrument(w: _W) -> bool:
    """A financial instrument acronym written in capitals ("ORANE", "BSA"): never a name word (R1, CL-19)."""
    return w.upper and w.low in _instruments()


def _legal_form(w: _W) -> Optional[str]:
    strength = LEGAL_FORMS.get(w.low)
    if strength is None:
        return None
    base = w.text.rstrip(".")
    letters = [c for c in base if c.isalpha()]
    if all(c.isupper() for c in letters) or base in _LEGAL_MIXED or base.replace(".", "") in _LEGAL_MIXED:
        return strength
    return None


def _is_trade(low: str) -> bool:
    """A trade head noun, or a compound of trade nouns ("boulangerie-patisserie")."""
    if low in ORG_TRADES:
        return True
    parts = [p for p in low.split("-") if p]
    return len(parts) > 1 and all(p in ORG_TRADES for p in parts)


def _is_first_name(w: _W) -> bool:
    if not w.cap or w.initial or w.digit or _instrument(w):
        return False
    fn = _lists()[3]
    if w.low in fn:
        return True
    if "-" in w.low:
        parts = [p for p in w.low.split("-") if p]
        return len(parts) > 1 and all(p in fn for p in parts)
    return False


def _excluded(low: str) -> bool:
    stop, legal, _common, _fn, _afn, months, wl = _lists()
    if low in stop or low in legal or low in months or low in wl or low in TITLE_WORDS:
        return True
    if low in LEGAL_FORMS or low in ORG_TRIGGERS:
        return True
    if "-" in low:
        parts = [p for p in low.split("-") if p]
        if parts and all(p in stop or p in legal or p in months for p in parts):
            return True
    return False


def _person_word(w: _W, allow_common: bool = True) -> bool:
    if w.digit or w.initial or w.prefix or not w.cap or _instrument(w):
        return False
    if sum(c.isalpha() for c in w.text) < 2:
        return False
    if _excluded(w.low):
        return False
    if not allow_common and not w.upper and w.low in _lists()[2]:
        return False
    return True


def _org_word(w: _W, allow_article: bool) -> bool:
    if w.digit or w.prefix or w.initial:
        return False
    if not w.cap or sum(c.isalpha() for c in w.text) < 2:
        return False
    if w.low in _ARTICLES:
        return allow_article
    if w.low in _ORG_CONNECT:
        return False
    stop, _legal, _common, _fn, _afn, months, wl = _lists()
    if w.low in stop or w.low in months or w.low in wl or w.low in TITLE_WORDS:
        return False
    if w.low in LEGAL_FORMS and _legal_form(w):
        return False
    if w.low in ("societe", "ste", "denommee", "denomination"):
        return False
    return True


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuleSpan:
    start: int
    end: int
    kind: str
    priority: int
    rule: str


def _company_after(run: list[_W], k: int) -> bool:
    """True when the capitalised words after ``run[k]`` read as a company, not as the rest of a person's name:
    two or more all-capital words ("de LUMIERE FROID SERVICES") or a company word or legal form (R1, CL-22)."""
    upper = 0
    for w in run[k + 1:k + 8]:
        if w.low in PARTICLES and (w.upper or not w.cap):
            continue
        if w.digit or not w.cap:
            break
        if w.low in ORG_GENERIC or _legal_form(w):
            return True
        if w.upper and not w.initial:
            upper += 1
    return upper >= 2


def _take_person(run: list[_W], j: int, max_words: int = 5) -> Optional[tuple[int, int]]:
    """Collect name words from ``run[j]``; return token indices ``(first, last + 1)``."""
    n = len(run)
    idx: list[int] = []
    k = j
    words = 0
    while k < n and words < max_words:
        w = run[k]
        if w.initial and not _title(w):
            idx.append(k)
            k += 1
            continue
        if w.low in ("de", "du", "des", "d") and not w.cap and idx and _company_after(run, k):
            break  # "Kévin RAMEL de LUMIERE FROID SERVICES": the person ends before "de"
        if w.low in PARTICLES and not w.sent_end and (not w.cap or w.upper or w.low in ("le", "la", "de", "du", "des")):
            nxt = run[k + 1] if k + 1 < n else None
            nxt2 = run[k + 2] if k + 2 < n else None
            if nxt is not None and (_person_word(nxt) or (nxt.low in PARTICLES and nxt2 is not None and _person_word(nxt2))):
                idx.append(k)
                k += 1
                continue
            break
        if w.prefix and w.low == "d":
            nxt = run[k + 1] if k + 1 < n else None
            if nxt is not None and _person_word(nxt):
                idx.append(k)
                k += 1
                continue
            break
        if _person_word(w):
            idx.append(k)
            words += 1
            k += 1
            if w.sent_end:
                break
            continue
        break
    while idx and (run[idx[-1]].low in PARTICLES and not _person_word(run[idx[-1]]) or run[idx[-1]].initial or run[idx[-1]].prefix):
        idx.pop()
    if not any(_person_word(run[i]) for i in idx):
        return None
    return idx[0], idx[-1] + 1


def _org_backward(run: list[_W], i: int) -> Optional[tuple[int, int]]:
    k = i - 1
    first_valid = None
    while k >= 0 and i - k <= 7:
        w = run[k]
        if w.sent_end and k != i - 1:
            break
        if _org_word(w, allow_article=k > 0):
            first_valid = k
            k -= 1
            continue
        if w.low in _ORG_CONNECT and first_valid is not None and (not w.cap or w.upper or w.low in ("&",)):
            if w.low == "et" and run[k + 1].low not in _ORG_ET_JOIN:
                break
            k -= 1
            continue
        break
    if first_valid is None or not _org_word(run[i - 1], allow_article=False):
        return None
    while first_valid < i and run[first_valid].low in _ARTICLES and first_valid == 0:
        first_valid += 1
    if first_valid >= i:
        return None
    return first_valid, i


_LEAD_PARTICLES = frozenset({"du", "de", "des", "d", "la", "le", "les", "l"})


@lru_cache(maxsize=1)
def _spelled_phrases() -> tuple[tuple[str, ...], ...]:
    """Word tuples of the spelled-out legal forms (``legal_forms_spelled_fr.txt``, folded), longest first
    (v0.2, integrator: Judilibre class 2.4; for example "société publique locale X", "société coopérative
    d'intérêt collectif X")."""
    try:
        from . import merge as mg  # lazy: merge imports this module

        phrases, _short = mg.spelled_forms("fr")
    except Exception:  # noqa: BLE001 - the lexicon file is optional
        return ()
    out: set[tuple[str, ...]] = set()
    for ph in phrases:
        words = tuple(_singular(w) for w in re.split(r"[ '’\-]+", ph) if w)
        if len(words) >= 2:
            out.add(words)
    return tuple(sorted(out, key=len, reverse=True))


def _singular(word: str) -> str:
    """A folded word without its plural "s" ("habitations" and "habitation" compare equal; "loyers modérés")."""
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def _skip_spelled_form(run: list[_W], j: int) -> int:
    """Index after a spelled-out legal form that starts at ``j`` (any case: "par actions simplifiée",
    "Publique Locale", "coopérative d'intérêt collectif"), else ``j``."""
    phrases = _spelled_phrases()
    if not phrases or j >= len(run):
        return j
    words: list[tuple[str, int]] = []  # (word, index of the token after it)
    k = j
    while k < len(run) and len(words) < 14:
        for piece in re.split(r"['’]", run[k].low):
            if piece:
                words.append((_singular(piece), k + 1))
        k += 1
    flat = [w for w, _ in words]
    for ph in phrases:
        m = len(ph)
        if m <= len(flat) and flat[:m] == list(ph):
            return words[m - 1][1]
    return j


def _geo_complement(run: list[_W], j: int) -> bool:
    """True when the token at ``j`` is a particle ("de", "du", "des", "d'"): after a spelled-out legal form it opens
    a geographic or descriptive complement, not a company name."""
    return j < len(run) and (run[j].low in ("de", "du", "des", "d") or run[j].prefix)


def _org_forward(run: list[_W], j: int) -> Optional[tuple[int, int]]:
    n = len(run)
    k = j
    while k < n and not run[k].cap and (run[k].low in _ARTICLES or run[k].low in ("du", "de", "des", "d")):
        k += 1
    start = k
    # "SCI DU PARC", "SARL DE LA GARE": upper-case particles before an upper-case name belong to the name.
    p = k
    while p < n and p - k < 2 and run[p].upper and run[p].low in _LEAD_PARTICLES:
        p += 1
    if p > k and p < n and run[p].upper and _org_word(run[p], allow_article=False):
        k = p
    last_valid = None
    while k < n and k - start < 7:
        w = run[k]
        if _org_word(w, allow_article=True):
            if w.low not in _ARTICLES:
                last_valid = k
            k += 1
            if w.sent_end:
                break
            continue
        if last_valid is not None and w.low in _ORG_CONNECT and (not w.cap or (w.upper and run[last_valid].upper)):
            if w.low == "et" and (k + 1 >= n or run[k + 1].low not in _ORG_ET_JOIN):
                break
            k += 1
            continue
        break
    if last_valid is None:
        return None
    while last_valid > start and run[last_valid].low in ("avocats", "d'avocats", "avocat"):
        last_valid -= 1  # "SARL PIVOINE D'AVOCATS": the legal descriptor is not part of the name (v0.2)
        if not _org_word(run[last_valid], allow_article=False):
            return None
    return start, last_valid + 1


def _meaningful_org(run: list[_W], s: int, e: int, roles: bool = False) -> bool:
    """A company name holds a word that is not a legal or function word. With ``roles`` (a legal form right next to
    it), a contractual role noun counts: "Concessionnaire SAS" is a company (R1, CL-17)."""
    stop, legal, _common, _fn, _afn, months, wl = _lists()
    words = [w for w in run[s:e] if w.low not in _ORG_CONNECT]
    if not words:
        return False
    cr = lx.contract_roles() if roles else frozenset()
    if len(words) == 1 and words[0].upper and words[0].low in lx.acronym_stoplist():
        return False  # "CABINET JAF": a court function or a legal acronym alone is not a company (v0.2)
    return any((w.low not in legal or w.low in cr) and w.low not in stop for w in words)


def _mask_legal_form() -> bool:
    """RAGBASE_SHIELD_MASK_LEGAL_FORM=1: the legal form is masked with the company name ("ACME SAS" -> SOCIETE_A).
    Default 0: the form stays in clear ("SOCIETE_A SAS"), the readable default (V02_DESIGN section 10, decision 1)."""
    return os.environ.get("RAGBASE_SHIELD_MASK_LEGAL_FORM", "0").strip().lower() in ("1", "true", "yes", "on")


def legal_form_policy() -> str:
    """Where the legal form of a company sits (v0.3, S1 span policy), for both languages:

    - ``"clear"`` (default): the detected span excludes the form, the form stays in clear ("SOCIETE_A SAS");
    - ``"cover"`` (``RAGBASE_SHIELD_LEGAL_FORM_SPAN=cover``): the detected span covers the form, so a consumer of
      the spans (a highlight, a benchmark adapter, a review list) sees "ACME SAS" as one company; the masker keeps
      the form readable in the masked text when it renders with :func:`org_form_parts`, and masks it until then
      (never less than the span);
    - ``"mask"`` (``RAGBASE_SHIELD_MASK_LEGAL_FORM=1``): the span covers the form and the form is masked.
    """
    if _mask_legal_form():
        return "mask"
    v = os.environ.get("RAGBASE_SHIELD_LEGAL_FORM_SPAN", "").strip().lower()
    return "cover" if v in ("cover", "span", "1", "true", "yes", "on") else "clear"


# Civility of a title right before a name: "M" / "F" (v0.2, DETECTORS: shared surnames).
_CIV_BEFORE_RE = re.compile(
    r"(?:(?<![\w.])(?P<m>M|MM|Mr|MR|Messieurs|MESSIEURS|[Mm]onsieur|MONSIEUR|[Ss]ieur|[Vv]euf)\.?"
    r"|(?<![\w.])(?P<f>Mme|Mmes|MME|MMES|Mlle|MLLE|Ms|Mrs|MRS|Madame|MADAME|Mesdames|Mademoiselle|[Dd]ame|[Vv]euve)\.?)"
    r"[ \t\u00a0\u202f]*$"
)


def title_civility(text: str, pos: int) -> str:
    """"M" or "F" when a civility title ("M.", "Mme", "Mr", "Mrs"...) ends right before ``pos``, else ""."""
    m = _CIV_BEFORE_RE.search(text[max(0, pos - 16):pos])
    if not m:
        return ""
    return "M" if m.group("m") else "F"


@lru_cache(maxsize=None)
def _first_name_sex() -> dict[str, str]:
    """Folded first name -> "M" / "F". Source: ``lexicon.first_name_sex()`` when the lexicon provides it
    (LOCALE-EN bundle, v0.2), else the tab-separated sex column of ``first_names_fr.txt`` when present, else empty
    (no civility resolution)."""
    fn = getattr(lx, "first_name_sex", None)
    if callable(fn):
        try:
            return {fold_word(k): str(v).upper()[:1] for k, v in dict(fn()).items() if str(v).upper()[:1] in "MF"}
        except Exception:  # noqa: BLE001 - a lexicon problem never stops the masking
            return {}
    out: dict[str, str] = {}
    try:
        for line in lx.read_lines("first_names_fr.txt"):
            name, sep, sex = line.partition("\t")
            sex = sex.strip().upper()[:1]
            if sep and sex in ("M", "F"):
                out[fold_word(name)] = sex
    except Exception:  # noqa: BLE001
        return {}
    return out


def person_civility(value: str) -> str:
    """"M" or "F" from the usual first name of a person value when its sex is known, else ""."""
    firsts, _surname = split_person(value)
    if not firsts:
        return ""
    first = fold_word(firsts[0]).replace(".", "").split("-")[0]
    return _first_name_sex().get(first, "") if first else ""


_SIGNED_BEFORE_RE = re.compile(r"(?i)\b(?:sign[ée]e?s?|par|de)\s*:\s*$")
_SENTENCE_END_RE = re.compile(r"(?:^|[.!?…]|\n)[\s«»\"“”()\-–—•*]*$")


def _starts_sentence(text: str, pos: int) -> bool:
    """True when ``pos`` starts a sentence or a line (not after a comma, a colon or "et")."""
    return bool(_SENTENCE_END_RE.search(text[max(0, pos - 12):pos]))


_BACKWARD_CTX = re.compile(
    r"^\s*,\s*(?:n[ée]e?s?\b|n[ée]\(e\)|demeurant|domicili[ée]|de nationalit[ée]|exer[çc]ant|agissant|"
    r"ayant (?:son|sa) (?:domicile|résidence)|r[ée]sidant)",
    re.IGNORECASE,
)


def _analyze_run(text: str, run: list[_W], out: list[RuleSpan]) -> None:
    n = len(run)

    def add(s: int, e: int, kind: str, prio: int, rule: str) -> None:
        out.append(RuleSpan(run[s].start, run[e - 1].end, kind, prio, rule))

    # Companies with a legal form before or after the name.
    for i, w in enumerate(run):
        strength = _legal_form(w)
        if not strength:
            continue
        span = _org_backward(run, i) if i > 0 else None
        if span is None and i + 1 < n:
            article_before = i > 0 and run[i - 1].low in ("la", "une", "cette", "l") and not run[i - 1].cap
            if strength == "strong" or article_before:
                span = _org_forward(run, i + 1)
        if span and _meaningful_org(run, *span, roles=True):
            s0, e0 = span
            if legal_form_policy() != "clear":
                s0, e0 = min(s0, i), max(e0, i + 1)  # "cover" or "mask" (v0.3): the form joins the span
            add(s0, e0, "ORG", 6, "legal_form")

    # Companies after a trigger word ("la société X", "groupe X", "cabinet X").
    for i, w in enumerate(run):
        if w.low not in ORG_TRIGGERS and not (w.low[:2] in ("l'", "d'") and w.low[2:] in ORG_TRIGGERS):
            continue  # "société", also glued: "L'association", "D'entreprise"
        if w.low in _DEFINING and i + 1 < n and run[i + 1].low in _ARTICLES:
            continue  # "ci-après dénommée « le Concessionnaire »": a defined term, not a company (R1, CL-17)
        # v0.2: a spelled-out legal form is skipped ("société coopérative d'intérêt collectif X", "société par
        # actions simplifiée X"); a particle right after the form is a geographic complement of a generic
        # description ("... coopérative d'intérêt collectif de la X"), not a company name: nothing is masked.
        j = i + 1
        end = _skip_spelled_form(run, i)
        geo = False
        if end > j:
            j, geo = end, _geo_complement(run, j if False else end)
        while not geo:
            j0 = j
            while j < n and not run[j].cap and (run[j].low in _ORG_SKIP or run[j].prefix):
                j += 1
            end = _skip_spelled_form(run, j)
            if end > j:
                j, geo = end, _geo_complement(run, end)
                continue
            if j == j0:
                break
        if geo:
            continue
        span = _org_forward(run, j)
        if span and _meaningful_org(run, *span):
            add(span[0], span[1], "ORG", 5, "trigger")

    # Companies after a trade noun: "Garage Gravelotte", "TRANSPORTS GRAVELOTTE" (the noun stays in clear). A
    # compound of trade nouns is a trade noun ("Boulangerie-Pâtisserie X"); any capitalised word after it starts the
    # name, a common word included ("Minoterie Bastide") (R1, CL-08).
    for i, w in enumerate(run):
        if i + 1 >= n or not w.cap or not _is_trade(w.low):
            continue
        nxt = run[i + 1]
        if not nxt.cap or nxt.digit or not _org_word(nxt, allow_article=False) or _excluded(nxt.low):
            continue
        if not nxt.upper and nxt.low in ORG_GENERIC:
            continue
        span = _org_forward(run, i + 1)
        if span and span[0] == i + 1 and _meaningful_org(run, *span):
            add(span[0], span[1], "ORG", 4, "trade")

    # Companies named by a suffix word: "LERVIGNAC AVOCATS", "Brivanor Logistique".
    for i, w in enumerate(run):
        if i == 0 or w.low not in _ORG_SUFFIXES or not w.cap:
            continue
        k = i - 1
        while k >= 0 and i - k <= 3 and _person_word(run[k], allow_common=False) and not _title(run[k]) \
                and run[k].low not in _ORG_SUFFIXES:
            k -= 1
        s0 = k + 1
        if k >= 0 and (run[k].prefix or run[k].low in _ARTICLES) and run[k].low != "les":
            continue  # "L'ENSEMBLE IMMOBILIER", "LE GROUPE": a description after an article, not a name (v0.2)
        if s0 < i and run[s0].low.startswith(("l'", "d'", "l’", "d’")):
            continue  # the elision glued to the head word: "L'ENSEMBLE IMMOBILIER" (v0.2)
        if s0 < i and not (k >= 0 and _title(run[k])):
            add(s0, i + 1, "ORG", 4, "suffix")

    # People after a title; plural titles cover "Mmes Garnier et Perrin".
    for i, w in enumerate(run):
        if not _title(w):
            continue
        if _bare_m(w) and i > 0 and run[i - 1].digit:
            continue  # "150 M Horaire": a unit after a number, not a title (R1, CL-18b)
        span = _take_person(run, i + 1)
        if span:
            add(span[0], span[1], "PERSON", 5, "title")
            k = span[1]
            while w.low in _PLURAL_TITLES and k + 1 < n and run[k].low in ("et", ",") and not run[k].cap:
                nxt = _take_person(run, k + 1)
                if not nxt or nxt[0] != k + 1:
                    break
                add(nxt[0], nxt[1], "PERSON", 5, "title")
                k = nxt[1]

    stop, legal, common, _fn, _afn, months, wl = _lists()
    sentence_start = _starts_sentence(text, run[0].start)

    # First name + surname.
    for i, w in enumerate(run):
        if not _is_first_name(w) or _excluded(w.low):
            continue
        if w.upper and i + 1 < n and run[i + 1].cap and not run[i + 1].upper and not run[i + 1].initial:
            continue  # "PRENOM Nom" is not a French name order: an acronym before a name (R1, CL-19)
        if i > 0 and (_title(run[i - 1]) or run[i - 1].low in _PLACE_WORDS):
            continue
        if i + 1 < n and run[i + 1].initial and len(run[i + 1].low) == 1 and not _title(run[i + 1]) \
                and (i + 2 >= n or not run[i + 2].cap):
            add(i, i + 2, "PERSON", 3, "first_initial")  # "Martine G."
            continue
        span = _take_person(run, i + 1)
        if not span or span[0] != i + 1:
            continue
        sur = [run[k] for k in range(span[0], span[1]) if _person_word(run[k])]
        if not sur:
            continue
        ambiguous_first = w.low in common or w.low in legal
        if ambiguous_first and i == 0 and sentence_start and all(s.low in common and not s.upper for s in sur):
            continue  # "Rose Blanche ..." at a sentence start
        add(i, span[1], "PERSON", 4, "first_name")

    # A first name after a greeting: "Bonjour Corinne,", "Chère Inès".
    for i, w in enumerate(run[:-1]):
        nxt = run[i + 1]
        if w.low in _GREETINGS and _is_first_name(nxt) and not _excluded(nxt.low) and \
                (i + 2 >= n or not run[i + 2].cap):
            add(i + 1, i + 2, "PERSON", 3, "greeting")

    # Initials and a surname after "Signé :" / "Par :": "Signé : J. Morel".
    if run[0].initial and not _title(run[0]) and _SIGNED_BEFORE_RE.search(text[max(0, run[0].start - 24):run[0].start]):
        span = _take_person(run, 0, max_words=2)
        if span and span[0] == 0 and span[1] > 1:
            add(span[0], span[1], "PERSON", 3, "signed")

    # SURNAME Firstname ("DUPONT Jean-Pierre").
    for i in range(n - 1):
        w = run[i]
        if not (w.upper and _person_word(w)) or _legal_form(w):
            continue
        nxt = run[i + 1]
        if not (nxt.cap and not nxt.upper and _is_first_name(nxt)):
            continue
        s = i
        while s > 0 and not _title(run[s - 1]) and (
            (run[s - 1].upper and _person_word(run[s - 1]) and not _legal_form(run[s - 1]))
            or (run[s - 1].low in PARTICLES and run[s - 1].upper)
        ):
            s -= 1
        while s < i and run[s].low in PARTICLES and not _person_word(run[s]):
            s += 1
        e = i + 2
        if e < n and run[e].cap and not run[e].upper and _is_first_name(run[e]):
            e += 1
        add(s, e, "PERSON", 4, "surname_first")

    # Role words: "le salarié DURAND", "les époux MARTIN", "née VASSEUR".
    for i, w in enumerate(run):
        if w.low not in _ROLE_WORDS or i + 1 >= n:
            continue
        if run[i + 1].low in PARTICLES and not run[i + 1].cap and w.low not in _ROLE_NAME_PARTICLE:
            continue  # "la gérante de LUMIERE SAS": "de" + a name is a genitive here (R1, CL-22)
        span = _take_person(run, i + 1, max_words=3)
        if not span or span[0] != i + 1:
            continue
        words = [run[k] for k in range(span[0], span[1]) if _person_word(run[k])]
        if len(words) == 1 and not words[0].upper and words[0].low in common:
            continue
        if len(words) > 1 and all(run[k].upper for k in range(span[0], span[1])):
            continue  # an all-capital decision: "LE DEBITEUR SAISI SOUTENAIT ALORS" is prose, not a name (v0.2)
        add(span[0], span[1], "PERSON", 3, "role")

    # "Kofi ABRANTIE, demeurant ..." / "Jean DUPONT, né le ...".
    if _BACKWARD_CTX.match(text[run[-1].end:run[-1].end + 60]):
        k = n - 1
        idx: list[int] = []
        while k >= 0 and len(idx) < 5:
            w = run[k]
            if _title(w) or _legal_form(w):
                break
            if _person_word(w) or w.initial or (w.low in PARTICLES and idx):
                idx.append(k)
                k -= 1
                continue
            break
        while idx and (run[idx[-1]].low in PARTICLES and not _person_word(run[idx[-1]])):
            idx.pop()
        real = [run[i] for i in idx if _person_word(run[i])]
        first_i = min(idx) if idx else 0
        after_place = first_i > 0 and run[first_i - 1].low in _PLACE_PREPOSITIONS
        if real and not after_place and not (len(real) == 1 and not real[0].upper and real[0].low in common):
            add(min(idx), max(idx) + 1, "PERSON", 3, "backward")


_LIST_GAP_RE = re.compile(r"[ \t\u00a0]*(?:,|et|&)[ \t\u00a0]+")


def _list_persons(run: list[_W], pos: int, add) -> int:
    """Persons of ``run`` from ``pos``, joined by "et"; returns the index after the last one taken."""
    while pos < len(run):
        sp = _take_person(run, pos, max_words=3)
        if not sp or sp[0] != pos or _legal_form(run[pos]) or run[pos].low in ORG_TRIGGERS:
            break
        add(sp[0], sp[1])
        pos = sp[1]
        if pos + 1 < len(run) and run[pos].low == "et" and not run[pos].cap:
            pos += 1
            continue
        break
    return pos


def _list_reaches_end(run: list[_W], pos: int, taken: list[tuple[int, int]]) -> bool:
    """True when the persons from ``pos`` (joined by "et") reach the end of ``run``."""
    return _list_persons(run, pos, lambda s, e: taken.append((s, e))) == len(run) and bool(taken)


def _plural_title_lists(text: str, runs: list[list[_W]], out: list[RuleSpan]) -> None:
    """Judges or counsel named in a comma list after one plural title ("Mmes X, Y, Z, conseillers"): every
    capitalised item of the list, up to the lower-case role word, is a person (v0.2, DETECTORS). A comma splits the
    word runs, so the list is walked over the consecutive runs; it stops at a run that holds anything else."""
    for r, run in enumerate(runs):
        heads = [i for i, w in enumerate(run[:-1]) if w.low in _PLURAL_TITLES and _title(w)]
        if not heads:
            continue
        taken: list[tuple[int, int]] = []
        if not _list_reaches_end(run, heads[-1] + 1, taken):
            continue  # the title's own run must end with a name (the in-run rule already covers "Mmes X et Y")
        prev = run
        k = r + 1
        items = 0
        while k < len(runs) and items < 12:
            nxt = runs[k]
            if not _LIST_GAP_RE.fullmatch(text[prev[-1].end:nxt[0].start]):
                break
            spans: list[tuple[int, int]] = []
            end = _list_persons(nxt, 0, lambda s, e: spans.append((s, e)))
            if not spans:
                break
            for s, e in spans:
                out.append(RuleSpan(nxt[s].start, nxt[e - 1].end, "PERSON", 5, "title_list"))
                items += 1
            if end != len(nxt):
                break  # "Z conseillers": the role word ends the list
            prev = nxt
            k += 1


_FIELD_RE = re.compile(
    r"(?im)^[ \t]*(?P<label>d[ée]nomination(?:\s+sociale)?(?:\s+ou\s+raison\s+sociale)?|raison\s+sociale|nom\s+commercial|enseigne|sigle"
    r"|nom(?:\s+de\s+naissance|\s+d['’]usage|\s+marital)?(?:\s*,\s*pr[ée]noms?)?|pr[ée]noms?"
    r"|n[ée]e?\s+sous\s+le\s+nom\s+de)\s*:[ \t]*(?P<value>[^\n;:]{2,80}?)[ \t]*(?:$|;)"
)
_MORE_FIRST_RE = re.compile(r"[ \t]*,[ \t]*([^\W\d_][\w'’\-]*)")
_FIELD_VALUE_RE = re.compile(r"[^\W\d_][\w'’\-. &]*[^\W_.]|[^\W\d_]")


def _field_spans(text: str, out: list[RuleSpan]) -> None:
    """Form fields: "Dénomination : OPTIMALYS SAS", "Nom : DURAND", "Prénoms : Marie Claire"."""
    for m in _FIELD_RE.finditer(text):
        label = fold_word(m.group("label"))
        vs = m.start("value")
        vm = _FIELD_VALUE_RE.match(text, vs, m.end("value"))
        if not vm:
            continue
        value = vm.group()
        is_org = label.startswith(("denomination", "raison", "nom commercial", "enseigne", "sigle"))
        words = _words(value)
        if not words:
            continue
        if is_org:
            k0, k1 = 0, len(words)
            while k0 < k1 and _legal_form(words[k0]) and k1 - k0 > 1:
                k0 += 1
            while k1 > k0 and _legal_form(words[k1 - 1]) and k1 - k0 > 1:
                k1 -= 1
            core = words[k0:k1]
            if core and _meaningful_org(core, 0, len(core)):
                out.append(RuleSpan(vs + core[0].start, vs + core[-1].end, "ORG", 6, "field"))
        else:
            ws = [w for w in words if _person_word(w) or w.low in PARTICLES or w.initial]
            if ws and any(_person_word(w) for w in ws):
                end = vs + ws[-1].end
                if "prenom" in label:
                    # "Nom, prénoms : FRÉMIOT Bertrand, Louis, Marie": the other first names belong to the person
                    fn = _lists()[4]
                    while True:
                        more = _MORE_FIRST_RE.match(text, end)
                        if not more or fold_word(more.group(1)) not in fn:
                            break
                        end = more.end()
                out.append(RuleSpan(vs + ws[0].start, end, "PERSON", 5, "field"))


_FORM_SEP_RE = re.compile(r"[ \t\u00a0]*(?:,|–|—|-|\()[ \t\u00a0]*")
_SPELLED_FORM_RE = re.compile(
    r"(?i)[ \t\u00a0]*(?:,|–|—|-|\()[ \t\u00a0]*(?:une\s+)?(?:"
    r"soci[ée]t[ée]\s+(?:d['’]exercice\s+lib[ée]ral[e]?\s+(?:à|a)\s+(?:responsabilit[ée]\s+limit[ée]e|forme\s+anonyme"
    r"|actions\s+simplifi[ée]e)|par\s+actions\s+simplifi[ée]e(?:\s+unipersonnelle)?"
    r"|(?:à|a)\s+responsabilit[ée]\s+limit[ée]e|civile\s+(?:professionnelle|immobili[èe]re|de\s+moyens)|anonyme"
    r"|en\s+nom\s+collectif)"
    r"|entreprise\s+unipersonnelle\s+(?:à|a)\s+responsabilit[ée]\s+limit[ée]e"
    r"|association\s+d['’]avocats\s+(?:à|a)\s+responsabilit[ée]\s+professionnelle\s+individuelle)\b"
)
_CI_APRES_RE = re.compile(
    r"\(\s*ci-?\s?apr[èe]s(?:\s+(?:d[ée]nomm[ée]e?s?|d[ée]sign[ée]e?s?|appel[ée]e?s?))?\s*[:,]?\s*[«\"“]\s*"
    r"(?P<alias>[^»\"”\n()]{2,40}?)\s*[»\"”]"
    # more names of the same party: "(ci-après « XYZ » ou « la Société »)" (R1, CL-01)
    r"(?:\s*(?:,|ou|et)\s*(?:(?:l[ea]|les)\s+|l['’]\s*)?[«\"“][^»\"”\n()]{1,40}[»\"”])*\s*\)")


def _looks_like_person(run: list[_W], s: int, e: int) -> bool:
    words = run[s:e]
    return bool(words) and (_title(words[0]) or (_is_first_name(words[0]) and len(words) > 1
                                                  and not words[0].upper))


def _separated_forms(text: str, runs: list[list[_W]], out: list[RuleSpan]) -> None:
    """A company name followed by its legal form after a separator: "CASTAGNÈDE & LHOMME – SELAS d'avocats",
    "PERNAUD & VALETTE, société d'exercice libéral à responsabilité limitée", "MOREL & ROUX (SELARL)"."""
    for k, run in enumerate(runs):
        if not run or not run[-1].cap:
            continue
        after = text[run[-1].end:run[-1].end + 120]
        form = False
        if _SPELLED_FORM_RE.match(after):
            form = True
        elif k + 1 < len(runs):
            nxt = runs[k + 1]
            gap = text[run[-1].end:nxt[0].start]
            if _FORM_SEP_RE.fullmatch(gap) and _legal_form(nxt[0]) == "strong" and \
                    (len(nxt) == 1 or not nxt[1].cap):
                form = True
        if not form:
            continue
        span = _org_backward(run, len(run))
        if span and _meaningful_org(run, *span) and not _looks_like_person(run, *span):
            out.append(RuleSpan(run[span[0]].start, run[span[1] - 1].end, "ORG", 6, "legal_form_after"))


def _party_definitions(text: str, runs: list[list[_W]], out: list[RuleSpan]) -> None:
    """Contract party definitions: "Kerdalis (ci-après « Kerdalis »)", "ci-après dénommée « Brivanor »"."""
    if "apr" not in text:
        return
    stop, legal, common, _fn, _afn, months, wl = _lists()
    ends = {run[-1].end: run for run in runs if run}
    for m in _CI_APRES_RE.finditer(text):
        alias = m.group("alias").strip()
        aw = _words(alias)
        if not aw or not aw[0].cap or aw[0].low in _ARTICLES or _title(aw[0]):
            continue
        if all(w.low in common or w.low in legal or w.low in stop or w.low in ORG_GENERIC for w in aw):
            continue
        before = text[:m.start()].rstrip()
        run = ends.get(len(before))
        if run is None:
            continue
        span = _org_backward(run, len(run))
        if not span or not _meaningful_org(run, *span) or _looks_like_person(run, *span):
            continue
        if any(_title(w) for w in run[max(0, span[0] - 2):span[0]]):
            continue
        out.append(RuleSpan(run[span[0]].start, run[span[1] - 1].end, "ORG", 5, "party_definition"))
        a0 = m.start("alias") + (len(m.group("alias")) - len(m.group("alias").lstrip()))
        out.append(RuleSpan(a0, a0 + len(alias), "ORG", 5, "party_definition"))


_PARTNER_GAP_RE = re.compile(r"[ \t\u00a0\u202f]*,?[ \t\u00a0\u202f]*")
_SAME_LINE_GAP_RE = re.compile(r"[ \t\u00a0\u202f]{1,2}")


def _partner_word(w: _W) -> bool:
    if w.digit or w.prefix or w.initial or not w.cap or sum(c.isalpha() for c in w.text) < 2:
        return False
    return not (_excluded(w.low) or w.low in ORG_GENERIC or w.low in _ARTICLES or w.low in _ORG_CONNECT)


def _partnerships(text: str, words: list[_W], out: list[RuleSpan]) -> None:
    """Professional partnerships named after their partners: "SCP Varenne et Maudet", "SELARL Varenne, Maudet &
    Corlay". After the form, up to 4 capitalised names joined by "et", "&" or "," are one company; a lower-case word,
    an article or a trigger word ends it (R1, CL-09)."""
    n = len(words)
    for i, w in enumerate(words):
        if w.low not in _PARTNERSHIP_FORMS or not (_legal_form(w) or w.text.rstrip(".") == "SEL"):
            continue
        names: list[tuple[int, int]] = []
        k = i + 1
        while k < n and len(names) < 4:
            if names:  # a separator: "," (not a token), "et" or "&"
                gap = text[words[k - 1].end:words[k].start]
                if words[k].low in ("et", "&") and not words[k].cap and _SAME_LINE_GAP_RE.fullmatch(gap) \
                        and k + 1 < n:
                    k += 1
                elif not ("," in gap and _PARTNER_GAP_RE.fullmatch(gap)):
                    break
            elif not _SAME_LINE_GAP_RE.fullmatch(text[words[k - 1].end:words[k].start]):
                break
            # one name: an optional capitalised particle ("Le Gall") and one or two capitalised words
            st = k
            if words[k].low in PARTICLES and words[k].cap and k + 1 < n and _partner_word(words[k + 1]):
                k += 1
            if not _partner_word(words[k]):
                break
            k += 1
            if k < n and _partner_word(words[k]) and _SAME_LINE_GAP_RE.fullmatch(text[words[k - 1].end:words[k].start]) \
                    and not words[k - 1].sent_end:
                k += 1
            names.append((st, k))
            if words[k - 1].sent_end:
                break
        if len(names) >= 2:
            out.append(RuleSpan(words[names[0][0]].start, words[names[-1][1] - 1].end, "ORG", 6, "partnership"))


# ---------------------------------------------------------------------------
# Names typed in lower case (v0.4.1, French only)
# ---------------------------------------------------------------------------
# Lawyers type prompts fast, often with no capital: "M. zadou mila", "mme claire dufresne conteste", "mon client
# jean dupont", "la société durand sas". The rules above need a capital on every name word. The rules below find a
# name with no capital from what stands around it: a civility before it (A), a first name of the lexicon at its
# start or at its end (B), a company cue or a legal form around it (C), a field label, a case caption. A word with
# no capital is a name word only when the French word lists do not know it, or know it as a first name or as a
# common word that is also a surname; a name stops at a verb form, a function word or an ordinary word. A rule
# gives a span only when one word of the name, or the civility before it, is in lower case: capitalised text
# keeps the spans of the rules above and gets no other span.


def lowercase_names() -> bool:
    """``RAGBASE_SHIELD_LOWERCASE_NAMES`` (v0.4.1): "0", "false", "no" or "off" turns the lower-case rules off
    (names here, addresses in ``detectors``): the 0.4.0 behaviour. Default on. Read at each call."""
    return os.environ.get("RAGBASE_SHIELD_LOWERCASE_NAMES", "1").strip().lower() not in ("0", "false", "no", "off")


# Words before a noun: after one of them a first name of the lexicon is a common noun ("un jean", "la rose").
_LC_DETERMINERS = frozenset(
    "le la les l un une des du au aux ce cet cette ces mon ma mes ton ta tes son sa ses notre nos votre vos leur "
    "leurs quel quelle quels quelles chaque aucun aucune plusieurs quelques certains certaines tout toute tous "
    "toutes nul nulle tel telle tels telles meme memes autre autres ledit ladite lesdits lesdites dudit audit".split())
# Subject pronouns; the pronouns between a subject and its verb ("il la marie"); and those of them that are no
# determiner, which the search for the verb of a subject may skip ("mme borne ne conteste pas"; never "le", "la",
# "les", "leur": "madame réclame la garde" has no second verb).
_LC_SUBJECTS = frozenset("je j tu il elle on nous vous ils elles qui ca cela ceci".split())
_LC_CLITICS = frozenset("ne n se s me m te t le la l les lui leur nous vous y en".split())
_LC_SKIPPED = frozenset("ne n se s me m te t lui y nous vous".split())
# Prepositions before a person ("à mme porte", "contre m. marche"): the civility after one is no clause subject.
_LC_PREPOSITIONS = frozenset("a de d pour contre avec par chez selon sans envers entre sur vers apres avant depuis "
                             "pendant malgre concernant sauf".split())
# Particles of a name that are no French function word ("karim ben salah", "m. van der berg").
_LC_NAME_PARTICLES = frozenset({"ben", "bin", "el", "al", "van", "von", "der", "den", "di", "da", "del", "della",
                                "ould", "bent", "abd", "abou", "bou",
                                "dos", "das", "ten", "ter", "zu"})
_LC_PLURAL_CIVILITIES = frozenset({"mm", "mmes", "mlles", "messieurs", "mesdames", "mesdemoiselles"})
# Every word that `_lc_civility` may read as a civility (folded): a cheap test before the function.
_LC_CIVILITY_WORDS = frozenset("m mm mme mmes mlle mlles mr dr mrs ms melle melles mdme monsieur madame mademoiselle "
                               "messieurs mesdames mesdemoiselles maitre maitres docteur professeur me pr".split())
# Conjugated forms of the auxiliary and modal verbs: a subject stands before them ("m. gros a écrit").
_LC_FINITE_AUX = frozenset(
    "a ont avait avaient aura auront aurait auraient est sont etait etaient sera seront serait seraient va vont "
    "allait allaient ira iront irait fait font faisait faisaient fera feront ferait peut peuvent pouvait pouvaient "
    "pourra pourront pourrait pourraient veut veulent voulait voulaient voudra voudrait doit doivent devait "
    "devaient devra devront devrait devraient sait savent savait saura faut fallait faudra vaut".split())
# Verb forms that are also prepositions, conjunctions or adverbs: no evidence of a verb.
_LC_NOT_VERBS = frozenset("entre pendant durant suivant concernant moyennant nonobstant maintenant devant excepte "
                          "soit du es as ai tu sue".split())
# Past participles and infinitives of the auxiliaries: they do not follow a subject.
_LC_NOT_FINITE = frozenset("ete eu pu su vu attendu".split())
# Stems of the verbs that take an attribute of their object ("déclare M. Martin démissionnaire", "a nommé Mme
# Bernard cogérante"): a word in lower case after the name of that object is the attribute, no surname.
_LC_ATTRIBUTE_VERBS = ("declar", "nomm", "design", "elu", "elis", "elit", "trouv", "jug", "estim", "consider",
                       "croi", "croy", "cru", "rend", "laiss", "reconn", "proclam", "presum", "reput", "qualifi",
                       "surnomm", "appel", "dit", "disent", "disai", "dira")
# Legal forms that are also words in lower case ("un sas d'entrée", "un spa"): after a company cue or an article only.
_LC_FORM_WORDS = frozenset({"sas", "spa", "inc", "corp", "limited", "corporation", "incorporated", "sca", "scs",
                            "scm", "spl", "slp"})
# Legal forms that the capitalised rules do not list; in lower case they close a company name as the others do.
_LC_EXTRA_FORMS = frozenset("earl gaec scea gfa sccv sarlu".split())
# The last word of a legal form written in full words ("société par actions simplifiée", "société civile
# immobilière", "société à responsabilité limitée").
_LC_SPELLED_FORM_ENDS = frozenset("simplifiee unipersonnelle limitee anonyme immobiliere professionnelle collectif "
                                  "commandite liberal liberale cooperative".split())
# Nouns of time: after a legal form and an article they are no company name ("gérer la sas les mois suivants").
_LC_TIME_WORDS = frozenset("jour jours mois an ans annee annees semaine semaines fois temps matin matins soir soirs "
                           "lendemain veille heure heures minute minutes trimestre semestre".split())
# Articles before a legal form ("la sarl x", "cette sci y").
_LC_FORM_ARTICLES = frozenset({"la", "une", "cette", "ma", "ta", "sa", "notre", "votre", "leur"})
# Endings of ordinary nouns, adjectives and adverbs, and of verb forms, with the shortest word length they apply to:
# a word no list holds that ends so is an ordinary word ("apparemment", "résiliation", "juridique", "contestait",
# "priorise", "fiabilise").
_LC_WORD_ENDINGS = (("phone", 8), ("phones", 9), ("phrene", 9), ("escent", 9), ("escente", 10), ("aire", 11),
                    ("dif", 6), ("dive", 7), ("oque", 7), ("abilite", 9), ("ibilite", 9), ("alite", 8), ("uel", 5), ("uelle", 6), ("ment", 7), ("tion", 6), ("sion", 6), ("xion", 6), ("tions", 7), ("sions", 7), ("isme", 7),
                    ("ique", 7), ("iques", 8), ("able", 7), ("ables", 8), ("ible", 7), ("ibles", 8), ("ance", 7),
                    ("ances", 8), ("ence", 7), ("ences", 8), ("trice", 7), ("euse", 6), ("euses", 7), ("tif", 6),
                    ("sif", 6), ("tive", 6), ("sive", 6), ("tifs", 7), ("tives", 7), ("ee", 5), ("ees", 6),
                    ("iste", 8), ("istes", 9), ("taire", 9), ("taires", 10), ("naire", 9), ("naires", 10),
                    ("graphe", 8))
# Endings of the nouns of a profession or an office that no list holds ("cardiologue", "pédiatre", "opticien",
# "liquidateur", "bailleresse"): a noun that names a person, never a name word.
_LC_PERSON_ENDINGS = (("logue", 7), ("logues", 8), ("iatre", 7), ("iatres", 8), ("pathe", 8), ("peute", 9),
                      ("icien", 8), ("iciens", 9), ("icienne", 10), ("ateur", 8), ("ateurs", 9), ("eresse", 9))
_LC_VERB_ENDINGS = (("ait", 5), ("aient", 6), ("eront", 7), ("erons", 7), ("erez", 7), ("issent", 8),
                    ("issait", 8), ("issant", 8), ("ise", 7), ("isent", 8), ("ifie", 7), ("ifient", 9))
# Prefixes of a verb form that the verb list does not hold with its prefix ("susvisé": "visé").
_LC_VERB_PREFIXES = ("sus", "pre", "re", "co", "sous", "non", "sur", "auto", "inter", "contre")
# Endings of adjectives after a legal form ("la scp notariale", "la sarl familiale"): no company name starts so.
_LC_ADJECTIVE_ENDINGS = ("ale", "elle", "aire", "iste", "iere", "ante", "ente", "oise", "aise", "ienne", "ole")
# Feminine endings and the masculine ending they come from ("ancienne": "ancien").
_LC_FEMININE = (("elle", "el"), ("enne", "en"), ("onne", "on"), ("ette", "et"), ("euse", "eux"), ("euse", "eur"),
                ("trice", "teur"), ("ive", "if"), ("che", "c"), ("gue", "g"))


@lru_cache(maxsize=None)
def _lc_sets() -> dict[str, frozenset]:
    """Word lists of the lower-case rules, folded: the lists of the other rules and the two files written for
    these rules (``verbs_fr.txt`` through ``lexicon.verb_form``, ``lowercase_fr.txt``)."""
    from . import merge as mg  # lazy: merge imports this module

    stop, legal, common, fn, afn, months, wl = _lists()
    low = lx.lowercase_words()
    hard = stop | months | legal | _ROLE_WORDS | TITLE_WORDS | ORG_TRIGGERS | _GREETINGS | wl | low["function"]
    # an acronym of the stoplist or a legal form typed in lower case is never a name word ("la jaf", "ago", "sarl");
    # an acronym that is also a word or a first name keeps the class of the word ("roi", "loi")
    known = mg.vocab("fr") | low["words"] | common | afn
    hard = hard | frozenset(a for a in (fold_word(x) for x in lx.acronym_stoplist()) if a not in known) \
        | frozenset(LEGAL_FORMS) | _LC_EXTRA_FORMS
    # ordinary words: the vocabulary, the words of lowercase_fr.txt, company and place words, and the first names
    # that the strong list leaves out because they are places or words ("lyon", "juste")
    plain = mg.vocab("fr") | low["words"] | ORG_GENERIC | _PLACE_WORDS | mg.ner_stoplist("fr")[0] | _ORG_SKIP \
        | (afn - fn) | (low["places"] - fn)
    # a noun of a trade is a person noun, never a name word, but when it is also a usual surname ("boulanger")
    persons = low["persons"] | (low["jobs"] - common - fn)
    return {"hard": frozenset(hard), "plain": frozenset(plain), "places": low["places"], "persons": persons,
            "states": low["states"], "common": common, "first": fn, "any_first": afn}


def _lc_bases(low: str) -> tuple[str, ...]:
    """``low`` and the singular or masculine words it may be a form of ("anciennes": "ancienne", "ancien")."""
    out = [low]

    def add(x: str) -> None:
        if len(x) >= 3 and x not in out:
            out.append(x)

    if low.endswith("aux"):
        add(low[:-3] + "al")
        add(low[:-3] + "ail")
    if low.endswith(("s", "x")):
        add(low[:-1])
    for b in list(out):
        if b.endswith("e"):
            add(b[:-1])
            for end, masculine in _LC_FEMININE:
                if b.endswith(end):
                    add(b[:len(b) - len(end)] + masculine)
    return tuple(out)


def _lc_shape(low: str) -> str:
    """"v", "n" or "p" for a word that ends like a verb form, like the noun of a profession or like an ordinary
    word, or that is a verb form behind a prefix ("susvisé", "prénommé", "réexaminé"), else ""."""
    n = len(low)
    for end, least in _LC_VERB_ENDINGS:
        if n >= least and low.endswith(end):
            return "v"
    for prefix in _LC_VERB_PREFIXES:
        if low.startswith(prefix) and n - len(prefix) >= 4 and lx.verb_form(low[len(prefix):]) == "form":
            return "v"
    for end, least in _LC_PERSON_ENDINGS:
        if n >= least and low.endswith(end):
            return "n"
    for end, least in _LC_WORD_ENDINGS:
        if n >= least and low.endswith(end):
            return "p"
    for prefix in ("in", "im", "il", "ir", "non"):
        if low.startswith(prefix) and n - len(prefix) >= 5 and _lc_flags(low[len(prefix):]) not in ("", "f"):
            return "p"  # "inexact", "illisible": the negative of a word that a list knows
    return ""


@lru_cache(maxsize=65536)
def _lc_flags(low: str) -> str:
    """Classes of a folded word for the lower-case rules, one letter each: "h" never a name word (a function,
    calendar, legal or role word, a title, a company trigger, a form of an auxiliary verb); "v" a conjugated verb
    form or a participle; "i" an infinitive; "f" a first name of the lexicon; "c" a common word that is also a
    surname; "n" a noun that names a person; "s" an adjective of a person's state; "p" another ordinary word. An
    empty string is a word no French list knows. A compound with a hyphen is read from its parts."""
    sets = _lc_sets()
    if "-" in low and low not in sets["hard"] and low not in sets["plain"] and low not in sets["any_first"]:
        parts = [p for p in low.split("-") if p]
        flags = [_lc_flags(p) for p in parts]
        if len(parts) == 2 and parts[0] in _LC_NAME_PARTICLES and len(parts[1]) >= 3 \
                and (parts[1] in sets["any_first"] or not flags[1]):
            return ""   # "ben-ahmed", "el-kervadec": a name particle and a name
        if len(parts) < 2 or any(len(p) < 2 or "h" in f for p, f in zip(parts, flags)):
            return "h"
        if all(p in sets["first"] for p in parts):
            return "f"  # "jean-pierre", "anne-sophie", "martin-leroy"
        if all(p in sets["any_first"] or f in ("", "c") for p, f in zip(parts, flags)) \
                and any(p in sets["any_first"] or not f for p, f in zip(parts, flags)):
            return ""   # "dupont-durand", "kervadec-talvez": a compound name
        return "p"
    out = ""
    verb = lx.verb_form(low)
    if low in sets["hard"] or verb == "aux":
        out += "h"
    if verb == "form":
        out += "v"
    elif verb == "inf":
        out += "i"
    if low in sets["first"]:
        out += "f"
    bases = (low,) if low in sets["any_first"] else _lc_bases(low)
    if any(b in sets["states"] for b in bases):
        out += "s"
    if any(b in sets["persons"] or (len(b) >= 7 and b.startswith("co") and b[2:] in sets["persons"]) for b in bases):
        out += "n"  # also "co" and a person noun: "cogérante", "codébiteur"
    if any(b in sets["common"] for b in bases):
        out += "c"
    if any(b in sets["plain"] for b in bases):
        out += "p"
    if not out and low not in sets["any_first"]:
        out = _lc_shape(low)
    return out


def _lc_kind(w: _W) -> str:
    """How a word may sit in a name typed in lower case: "first" (a first name), "amb" (a first name that is also
    another word), "oov" (a word no list knows), "common" (a common word that is also a surname), "verb" (a verb
    form), "plain" (an ordinary word) or "no" (never a name word)."""
    if w.digit or w.prefix or w.initial:
        return "no"
    low = w.low
    if sum(c.isalpha() for c in low) < 2 or not low.replace("'", "").replace("-", "").isalpha() \
            or len(set(low)) == 1:
        return "no"  # a digit, a symbol, a placeholder of a template ("m. xxx")
    f = _lc_flags(low)
    if "h" in f:
        return "no"
    if "f" in f:
        return "amb" if len(f) > 1 else "first"
    if "s" in f or "n" in f or (len(low) >= 7 and low.endswith("ment")):
        return "no"  # a state, a person noun, an adverb ("m. totalement opposé")
    if "v" in f:
        return "verb"
    if "c" in f:
        return "common"
    return "plain" if f else "oov"


def _lc_is_verb(w: _W, *, finite: bool = False) -> bool:
    """``w`` is a verb form ("conteste", "a", "vu"); with ``finite``, one that follows its subject (no bare past
    participle of an auxiliary, no present participle). A function word that is also a verb form ("entre",
    "pendant") is no evidence."""
    if w.digit or w.prefix or w.cap or w.text in ("à", "À"):
        return False
    low = w.low
    if low in _LC_NOT_VERBS or (finite and (low in _LC_NOT_FINITE or low.endswith("ant"))):
        return False  # a present participle is often a noun ("le montant", "le restant")
    return low in _LC_FINITE_AUX or lx.verb_form(low) == "form"


def _lc_past_shape(low: str) -> bool:
    """The folded word ends like a verb in the simple past ("quitta", "refusa", "sortit", "naquit", "reçut",
    "vint"). After a civility such a word is a name only before the verb of the name or at the end of the clause."""
    er = lx.verbs()["er"]
    for end in ("erent", "ai", "as", "a"):
        if low.endswith(end) and len(low) - len(end) >= 3 and low[:len(low) - len(end)] + "er" in er:
            return True
    return len(low) >= 5 and low.endswith(("it", "irent", "ut", "urent", "int", "inrent"))


def _lc_neutral_capital(w: _W) -> bool:
    """A capital that does not show that the text has its capitals. A keyboard writes it by itself on a first name
    of the lexicon, on a town, a country or a region of the place list, on a month or a day; and the writer types
    it in a legal form or an acronym in capitals ("CDI", "RG", "TVA": a word of the acronym list, or two or three
    capitals that are no first name). A surname, "Monsieur", "Tribunal", a company name are capitals of correct
    text."""
    low = w.low
    if w.upper:
        if low in LEGAL_FORMS or low in _LC_EXTRA_FORMS:
            return True
        return low.isalpha() and 2 <= len(low) <= 5 and "f" not in _lc_flags(low) and (
            len(low) <= 3 or low in _lc_acronyms())
    if w.initial:
        return False
    if "f" in _lc_flags(low) or low in lx.months_days():
        return True
    places = _lc_sets()["places"]
    if low in places:
        return True
    parts = [p for p in low.split("-") if p]  # "Aix-en-Provence", "Saint-Denis"
    return len(parts) > 1 and any(p in places for p in parts) and all(
        p in places or p in ("saint", "sainte", "sur", "sous", "en", "les", "le", "la", "de", "du", "des", "aux")
        or "f" in _lc_flags(p) for p in parts)


@lru_cache(maxsize=None)
def _lc_acronyms() -> frozenset:
    return frozenset(fold_word(a) for a in lx.acronym_stoplist())


def _lc_first_word_view(w: _W) -> _W:
    """The first word of a sentence typed without capitals, as a word in lower case; an elided article or pronoun
    glued to it is left out ("L'employeur" is read as "employeur")."""
    m = re.match(r"(?i)^(?:l|d|j|n|s|c|m|t|qu)['’](?=\w)", w.text)
    if m and not w.prefix:
        rest = w.text[m.end():]
        return replace(w, text=rest, start=w.start + m.end(), low=fold_word(rest.rstrip(".")).replace(".", ""),
                       cap=False, upper=False, initial=False)
    return replace(w, cap=False, upper=False, initial=False)


def _lc_typed_runs(text: str, runs: list[list[_W]], firsts: Optional[set] = None) -> list[Optional[list[_W]]]:
    """For each run, the run as the lower-case rules read it when its sentence is typed without capitals, else
    None. A sentence (to a full stop or a line break) is typed without capitals when no word has a capital but its
    first word and the words of :func:`_lc_neutral_capital` (first names, places, months, acronyms; a legal form in
    capitals only when the first word is in lower case). In the returned run the first word
    of the sentence counts as a word in lower case ("La sarl x", "Yanis vellin conteste"), but for a civility: its
    capital stays, and the full-word civility rules do not read it. A first name with a capital inside the
    sentence keeps it: "M. Benoit velkor conteste" is read as the capitalised rules read it."""
    res: list[Optional[list[_W]]] = [None] * len(runs)
    group: list[int] = []
    for r, run in enumerate(runs):
        group.append(r)
        nxt = runs[r + 1][0] if r + 1 < len(runs) else None
        if nxt is not None and not run[-1].sent_end and "\n" not in text[run[-1].end:nxt.start]:
            continue
        first = runs[group[0]][0]
        if firsts is not None:
            firsts.add(first.start)
        # a legal form in capitals after a capitalised first word is correct text ("La SCI familiale détient")
        if all(not w.cap or w is first or (_lc_neutral_capital(w) and not (
                w.upper and first.cap and (w.low in LEGAL_FORMS or w.low in _LC_EXTRA_FORMS)))
               for g in group for w in runs[g]):
            for g in group:
                res[g] = [_lc_first_word_view(w) if w is first and w.cap and not w.upper
                          and w.low not in _LC_CIVILITY_WORDS else w for w in runs[g]]
        group = []
    return res


def _lc_finite_third(w: _W) -> bool:
    """A verb form that follows a subject in the third person ("conteste", "a", "doit", "licencient", "continue"):
    no past participle ("reçus", "prévue", "convoquée", "saisie"), no infinitive, no word that is also a state or
    a noun that names a person."""
    if w.digit or w.prefix or w.cap or w.text in ("à", "À"):
        return False
    low = w.low
    if low in _LC_FINITE_AUX:
        return True
    if not _lc_is_verb(w, finite=True) or w.text.rstrip(".").endswith(("é", "ée", "és", "ées")):
        return False
    flags = _lc_flags(low)
    if "s" in flags or "n" in flags:
        return False  # a state or a person noun that is also a verb form ("contente", "absente", "juge")
    if low.endswith(("es", "ons", "ez")):
        return False  # the second person, the first person of the plural, a plural ("graves", "portes")
    if low.endswith(("u", "us", "ue", "i", "is", "ie", "ise", "ert", "erte", "erts")):
        return low.endswith("e") and low[:-1] + "er" in lx.verbs()["er"]  # "continue", "licencie", "utilise"
    return True


def _lc_verb_follows(run: list[_W], k: int, *, strict: bool = False) -> bool:
    """A conjugated verb stands at ``run[k]`` (reflexive pronouns and "ne" before it are skipped): the words before
    it are its subject ("mme parent conteste", "m. borne ne paie plus"). ``strict`` is for a word that is itself a
    verb form ("madame porte plainte", "madame travaille a temps partiel"): the verb after it must then be no noun
    of the lists and no "a" typed with no accent, which may be the preposition."""
    n = len(run)
    j = k
    while j < n and j - k < 3 and run[j].low in _LC_SKIPPED and not run[j].cap:
        j += 1
    if j >= n or not _lc_is_verb(run[j], finite=True):
        return False
    if strict:
        low = run[j].low
        return low != "a" and (low in _LC_FINITE_AUX or _lc_flags(low) == "v")
    return True


def _lc_object_position(run: list[_W], i: int) -> bool:
    """The civility ``run[i]`` follows a preposition or a verb ("à mme porte", "je représente m. gros"): the
    word after it cannot be the verb or the attribute of the civility, it is a name."""
    if i == 0:
        return False
    p = run[i - 1]
    if p.prefix:
        return p.low == "d"
    return p.text in ("à", "À") or p.low in _LC_PREPOSITIONS or _lc_is_verb(p)


def _lc_civility(text: str, run: list[_W], i: int) -> str:
    """The civility ``run[i]`` for the lower-case rule: "strong" (the form has one reading), "weak" (a common noun
    in lower case: "maître", "docteur", "professeur"; or "Pr" at the start of a sentence), "me" ("Me" at the start
    of a sentence, where it may be the pronoun) or "". Lower-case "me", "pr", "mes" and a bare "m" are never one."""
    w = run[i]
    if w.digit or w.prefix:
        return ""
    low = w.low
    base = w.text.rstrip(".")
    prev = run[i - 1] if i > 0 else None
    if low in ("m", "mm"):
        if w.initial:
            return "strong"  # "M."
        if not w.dot or _LC_LIST_LETTER_RE.search(text[max(0, w.start - 200):w.end + 200]) and not text[
                text.rfind("\n", 0, w.start) + 1:w.start].strip():
            return ""  # a bare "m"; the letter "m." of a lettered list ("l. préavis", "m. télétravail")
        after = run[i + 1] if i + 1 < len(run) else None
        if (prev is not None and not prev.cap and prev.low in ("le", "au", "du", "en")) \
                or (after is not None and after.low in _LC_UNIT_WORDS):
            return ""  # a metre: "85 euros le m. linéaire", "en m. cubes", "en mm. linéaires"
        if prev is not None and prev.digit:
            # "12 m." is a length; "le 3 mars 2022 m. vellin a signé" has a name and its verb after "m."
            nxt = run[i + 1] if i + 1 < len(run) else None
            return "strong" if nxt is not None and not nxt.cap and _lc_kind(nxt) in ("first", "oov") \
                and len(nxt.low) >= 3 and _lc_third_follows(run, i + 2) else ""
        return "strong"
    if low == "dr" and prev is not None and (prev.low in _LC_DETERMINERS or prev.low in ("de", "en")
                                             or (prev.prefix and prev.low in ("d", "l"))):
        return ""  # "cours de dr pénal" (droit), "le dr traitant"
    if low in ("mme", "mmes", "mlle", "mlles", "mr", "dr"):
        return "strong"
    if low in ("mrs", "ms"):
        return "strong" if base[:1].isupper() or low == "mrs" else ""
    if low in ("melle", "melles", "mdme") and base[:1].islower():
        return "strong"  # spellings of "mlle" and "mme" in typed text; with a capital the rules above decide
    if low in ("monsieur", "madame", "mademoiselle", "messieurs", "mesdames", "mesdemoiselles"):
        if prev is not None and prev.low in _LC_DETERMINERS:
            return ""  # "ce monsieur", "un monsieur", "ces messieurs": a common noun
        return "strong"
    if low in ("maitre", "maitres", "docteur", "professeur"):
        if base[:1].isupper():
            return "strong"
        if prev is not None and (prev.low in _LC_DETERMINERS or (prev.low == "en" and low.startswith("maitre"))):
            return ""  # "le maître de l'ouvrage", "agir en maître", "le docteur référent", "un professeur émérite"
        return "weak"
    if low == "me" and base in ("Me", "ME"):
        before = text[max(0, w.start - 8):w.start].rstrip()
        opens = before[-1:] in tuple(":«\"“([-–•*>") or bool(re.search(r"(?:^|\n)\s*(?:\d{1,2}[.)]|[a-z]\))$", before))
        return "me" if opens or _starts_sentence(text, w.start) else "strong"  # "Il écrit : « Me ... »"
    if low == "pr" and base in ("Pr", "PR"):
        return "weak" if _starts_sentence(text, w.start) else "strong"
    return ""


def _lc_cap_name_word(w: _W) -> bool:
    """A capitalised word inside a name typed partly in lower case ("m. zadou Mila"): a name word of the rules
    above that the lower-case lists do not read as an ordinary word."""
    return w.cap and _person_word(w) and not _title(w) and _lc_kind(w) in ("first", "amb", "oov", "common")


def _lc_particle_name(run: list[_W], k: int, civility: bool) -> int:
    """End index (exclusive) of a name part that starts with a particle at ``run[k]``, else 0.

    - a name particle ("ben", "el", "van der") then a name word, anywhere in a name: "karim ben salah";
    - only right after a strong civility (``civility``), because after a name these words start the rest of the
      sentence ("imputer à mme x des fautes", "accuser m. x d'escroquerie"): "d'", "n'" or "m'" glued to a word
      no list knows ("mme d'arvignac"); "de", "du", "des" (then "la" or "le") and a word that may be a surname
      ("madame de la tour"). The caller asks for support around such a name (:func:`_lc_particle_support`).
      Never "le" or "la" and a word: after a civility that is an office or the object of the verb ("madame la
      juge", "m. le jex", "accorder à madame la jouissance du logement")."""
    n = len(run)
    w = run[k]
    if w.cap or w.digit:
        return 0
    low = w.low

    def unknown(x: _W) -> bool:
        if x.cap:
            return _lc_cap_name_word(x)
        return _lc_kind(x) == "oov" and len(x.low) >= 3

    def surname(x: _W) -> bool:
        if x.cap:
            return _lc_cap_name_word(x)
        return _lc_kind(x) in ("oov", "first", "amb", "common") and len(x.low) >= 3

    if low in _LC_NAME_PARTICLES and not w.prefix:
        m = k + 1
        if m < n and not run[m].cap and run[m].low in _LC_NAME_PARTICLES:
            m += 1
        return m + 1 if m < n and surname(run[m]) else 0
    if not civility:
        return 0
    if low == "ait" and k + 1 < n and unknown(run[k + 1]) and len(run[k + 1].low) >= 4:
        return k + 2  # "m. ait ouardi" (the caller asks for support: "que monsieur ait raison" is a verb)
    if w.prefix:
        if low in ("d", "n", "m") and k + 1 < n and run[k + 1].start == w.end and unknown(run[k + 1]):
            return k + 2
        return 0
    if low in ("de", "du", "des"):
        m = k + 1
        if low == "de" and m < n and not run[m].cap \
                and (run[m].low in ("la", "le") or (run[m].prefix and run[m].low == "l")):
            m += 1
        return m + 1 if m < n and surname(run[m]) else 0
    return 0


# Civilities written as a full word, and "dr": in correct text they stand alone before a verb, a participle or
# an adjective ("Monsieur quitta le domicile", "le licenciement de Madame notifié le 3 mars"). The rules read them
# only in a sentence typed without capitals, and only when the civility itself is in lower case. The abbreviated
# civilities ("M.", "Mme", "Mlle", "MM.", "Me") are read in every text.
_LC_FULL_CIVILITIES = frozenset("monsieur madame mademoiselle messieurs mesdames mesdemoiselles maitre maitres "
                                "docteur professeur dr pr mr mrs ms melle melles mdme".split())
# Words after "m." or "mm." that make it a unit of length.
_LC_UNIT_WORDS = frozenset("lineaire lineaires cube cubes carre carres courant courants".split())
# A lettered list around the letter "m.": a line that starts with "l." or "n.".
_LC_LIST_LETTER_RE = re.compile(r"(?m)^[ \t]*[ln]\.[ \t]")
# Words that announce a name ("je soussigné paul mercier", "le nommé karim zelmadec").
_LC_NAME_CUES = frozenset("soussigne soussignee soussignes soussignees nomme nommee denomme denommee prenomme "
                          "prenommee appele appelee".split())
# Words before a civility after which the civility and its name are the object: a preposition that is not "à", a
# conjunction, and the stems of verbs whose object is a person ("je représente madame de la tour").
_LC_NAME_LEFT = frozenset("pour contre avec par chez de et ou entre selon sans envers vs concernant".split())
_LC_PERSON_OBJECT_VERBS = ("represent", "defend", "assist", "rencontr", "contact", "appel", "assign", "poursui",
                           "concern", "oppos", "epous", "licenci", "embauch", "recoi", "recu", "recev")


def _lc_particle_support(run: list[_W], i: int, end: int) -> bool:
    """A name that starts with "de", "du", "des" or "d'" right after the civility ``run[i]`` and ends before
    ``run[end]`` has support. The verb of the name follows it ("madame de la tour conteste", "mme de la tour a
    signé"). Or the clause ends there and the civility is no indirect object: it starts the clause, or it follows
    a preposition that is not "à", or a verb whose object is a person ("je représente madame de la tour"). After "à"
    these words are the object of the verb ("le juge a accordé à Madame des délais", "cela cause à monsieur du
    souci", "on a reproché à madame de la négligence")."""
    n = len(run)
    p = run[i - 1] if i > 0 else None
    if p is not None and not p.prefix and (p.text in ("à", "À") or p.low in ("a", "au", "aux")):
        return False
    if end < n:
        if run[end].low == "a" and not run[end].cap:
            if end + 1 < n and (run[end + 1].low in _LC_NOT_FINITE or lx.verb_form(run[end + 1].low) == "form"):
                return True
        else:
            j = end
            while j < n and j - end < 3 and run[j].low in _LC_SKIPPED and not run[j].cap:
                j += 1
            if j < n and _lc_finite_third(run[j]):
                return True
    if p is None:
        return end >= n
    return _lc_object_support(run, i)


def _lc_object_support(run: list[_W], i: int) -> bool:
    """The civility ``run[i]`` and its name are the object of a preposition that is not "à", or of a verb whose
    object is a person ("contre monsieur du roquevert", "je représente mme ly")."""
    if i == 0:
        return False
    p = run[i - 1]
    if p.prefix:
        return p.low == "d"
    return p.low in _LC_NAME_LEFT or (_lc_is_verb(p) and p.low.startswith(_LC_PERSON_OBJECT_VERBS))


def _lc_third_follows(run: list[_W], k: int) -> bool:
    """A verb in the third person stands at ``run[k]`` ("ne", "se" and such before it are skipped): the words
    before it are its subject."""
    n = len(run)
    j = k
    while j < n and j - k < 3 and run[j].low in _LC_SKIPPED and not run[j].cap:
        j += 1
    return j < n and _lc_finite_third(run[j])


def _lc_person_noun_surname(w: _W) -> bool:
    """A noun that names a person and that is also a surname ("cousin", "voisin", "doyen"): no role of a
    procedure, no title, no state, no function word."""
    f = _lc_flags(w.low)
    return "n" in f and "h" not in f and "s" not in f and w.low not in _ROLE_WORDS and len(w.low) >= 4


def _lc_given_particle(run: list[_W], k: int) -> int:
    """End index (exclusive) of a surname with a French particle at ``run[k]``, after a first name inside a name,
    else 0: "de", "du", "des", "d'", "de la", "le" or "la" and a word that may be a surname, before the verb of the
    name, "et", "ou" or the end of the clause ("mme nadia de montclar demande", "m. paul le goff a été licencié").
    After "le" or "la" the word must be one that no list knows; after a bare "de", one that no list knows or a
    first name ("de retour", "de garde" are no names)."""
    n = len(run)
    w = run[k]
    if w.cap or w.digit:
        return 0
    kinds: tuple[str, ...] = ("oov", "first", "amb", "common")  # "de la roche", "du moulin", "d'arvignac"
    if w.prefix:
        if w.low != "d" or k + 1 >= n or run[k + 1].start != w.end:
            return 0
        m = k + 1
    elif w.low in ("de", "du", "des"):
        m = k + 1
        if w.low == "de" and m < n and not run[m].cap \
                and (run[m].low in ("la", "le") or (run[m].prefix and run[m].low == "l")):
            m += 1
        elif w.low == "de":
            kinds = ("oov", "first")  # "de montclar"; never "de retour", "de garde", "de droit"
    elif w.low in ("le", "la", "ait"):
        m, kinds = k + 1, ("oov",)  # "paul le bihannec", "karim ait ouardi"
    else:
        return 0
    if m >= n:
        return 0
    x = run[m]
    if x.cap:
        if not _lc_cap_name_word(x):
            return 0
    elif len(x.low) < 3 or not (_lc_kind(x) in kinds or ("common" in kinds and "c" in _lc_flags(x.low)
                                                         and "h" not in _lc_flags(x.low))):
        return 0  # ("du bois", "de la porte": a word that is also a surname may be a verb form too)
    e = m + 1
    if e >= n or (not run[e].cap and run[e].low in ("et", "ou")) or _lc_third_follows(run, e):
        return e
    return 0


def _lc_name_next(run: list[_W], m: int) -> bool:
    """A name word or a name particle stands at ``run[m]``."""
    if m >= len(run):
        return False
    w = run[m]
    if w.cap:
        return _lc_cap_name_word(w)
    if _lc_kind(w) in ("first", "amb") or (_lc_kind(w) == "oov" and len(w.low) >= 3):
        return True
    return bool(_lc_particle_name(run, m, False)) or bool(_lc_given_particle(run, m))


def _lc_first_word(text: str, run: list[_W], i: int, k: int, civ: str) -> bool:
    """``run[k]`` may be the first word of the name after the civility ``run[i]`` (word classes of
    :func:`_lc_kind`). After a strong civility: a first name, a surname-like common word or a word no list knows;
    a verb form or an ordinary word only where it cannot be the verb or the attribute of the civility. After a weak
    civility: a first name or a word no list knows."""
    n = len(run)
    w = run[k]
    if w.cap:
        return _person_word(w) and not _title(w)  # as in the rules above; the name then needs a lower-case word
    kind = _lc_kind(w)
    last = k + 1 >= n
    if kind == "no" or len(w.low) < 3:
        return False  # a function word, a role, a noun that names a person, a state; a word of two letters
    if kind == "oov" and len(w.low) == 3 and not (_lc_third_follows(run, k + 1) or _lc_name_next(run, k + 1)):
        return False  # three letters: an abbreviation ("m. tns, pas de bulletin"), but before the verb of a name
    if civ == "strong":
        if kind in ("first", "common"):
            return True
        if kind == "oov":
            return True
        if kind == "amb":
            return last or _lc_name_next(run, k + 1) or _lc_verb_follows(run, k + 1)
        return False  # an ordinary word or a verb form is no name, whatever stands before the civility ("le
        #               licenciement de madame notifié le 3 mars", "donne à monsieur acte", "j'ai vu madame crier")
    if kind == "first":
        return True
    if kind == "amb":
        return _lc_name_next(run, k + 1)
    if kind != "oov" or len(w.low) < 3:
        return False
    if civ == "me" and w.low.endswith(("er", "ir", "re", "oir", "ant", "ez")):
        return _lc_verb_follows(run, k + 1)  # "Me résumer ce jugement", "Me traduire la clause 4": the pronoun;
        #                                      "Me garnier plaide" is a name
    return True


def _lc_typed_surname(run: list[_W], i: int, k: int, kind: str) -> bool:
    """``run[k]``, in lower case, goes on the name after the capitalised word before it, the first word after the
    civility ``run[i]``. The capital is the keyboard's (after "M.") and the name was typed in lower case: "M.
    Zadou mila". The word in lower case must be a first name of the lexicon with no other reading: a word that no
    list knows is too often a job, an adverb or a legal word that the lists do not hold ("M. Benoit velkor
    conteste" may be "M. Benoit, soudeur, conteste"). And the verb of the name follows, or the clause ends, but
    not after a verb that takes an attribute ("déclare M. Martin démissionnaire")."""
    w = run[k]
    if kind != "first" or len(w.low) < 3:
        return False
    if k + 1 < len(run):
        return _lc_verb_follows(run, k + 1)
    return not (i > 0 and _lc_is_verb(run[i - 1]) and run[i - 1].low.startswith(_LC_ATTRIBUTE_VERBS))


def _lc_name_after(text: str, run: list[_W], i: int, civ: str) -> Optional[tuple[int, int, bool]]:
    """The name after the civility ``run[i]``: ``(first index, end index, a name word is in lower case)`` or None.
    One to three words (particles not counted); it stops at the end of the run, at a verb form, at a function word
    and at an ordinary word."""
    n = len(run)
    k = i + 1
    if k >= n:
        return None
    idx: list[int] = []      # every token of the name
    names: list[int] = []    # its name words (no particle)
    kinds: list[str] = []
    if civ == "strong" and k + 1 < n and not run[k].cap and re.fullmatch(r"[a-z]-[a-z]", run[k].low) \
            and not run[k + 1].cap and _lc_kind(run[k + 1]) in ("oov", "first") and len(run[k + 1].low) >= 3:
        idx.append(k)  # hyphenated initials before the surname: "m. j-p kervadec"
        k += 1
    end = _lc_particle_name(run, k, civ == "strong")  # "madame de la tour"; never "maître d'apprentissage"
    if end and run[k].low not in _LC_NAME_PARTICLES and not _lc_particle_support(run, i, end):
        return None
    if end:
        idx.extend(range(k, end))
        names.append(end - 1)
        kinds.append("oov")
        k = end
    elif _lc_first_word(text, run, i, k, civ):
        idx.append(k)
        names.append(k)
        kinds.append(_lc_kind(run[k]))
        k += 1
    else:
        return None
    while len(names) < 5 and k < n:
        w = run[k]
        prev = run[names[-1]]
        given = kinds[-1] in ("first", "amb")
        end = _lc_particle_name(run, k, False)
        if not end and given and not prev.cap:
            end = _lc_given_particle(run, k)  # "mme nadia de montclar demande", "m. paul le goff a été licencié"
        if end:
            idx.extend(range(k, end))
            names.append(end - 1)
            kinds.append("oov")
            k = end
            continue
        kind = _lc_kind(w)
        last = k + 1 >= n
        typed = False
        if w.cap:
            ok = _lc_cap_name_word(w) and (kind != "common" or given)
        elif prev.cap:
            # a capital then no capital: the name of the rules above ends here ("M. Martin susvisé"), but for the
            # capital that a keyboard puts on the first word ("M. Zadou mila")
            ok = False
        elif kind == "first":
            ok = True
        elif kind == "amb":
            ok = given or last or _lc_verb_follows(run, k + 1)
        elif kind == "oov":
            # a third word that no list knows: after a first name, or before the verb of the name ("madame nelba
            # thi zadou demande")
            ok = len(w.low) >= 3 and (len(names) == 1 or kinds[0] in ("first", "amb")
                                      or (len(names) == 2 and (last or _lc_third_follows(run, k + 1))))
            if len(w.low) == 2:  # "m. paul ly a signé": a short surname before its verb
                ok = given and len(names) == 1 and _lc_third_follows(run, k + 1)
        elif kind == "common":
            ok = given and (last or _lc_verb_follows(run, k + 1))
        elif kind in ("plain", "verb") or (kind == "no" and _lc_person_noun_surname(w)):
            # after a civility and a first name the next word is the surname, also when it is an ordinary word,
            # if the verb of the name follows it ("m. paul marques a signé", "mme julie cousin conteste")
            ok = given and len(names) == 1 and len(w.low) >= 3 and _lc_third_follows(run, k + 1)
        else:
            ok = False
        if not ok:
            break
        idx.append(k)
        names.append(k)
        kinds.append(kind)
        k += 1
        if typed:
            break  # two words only
    if len(names) > 3 and not (k >= n or _lc_third_follows(run, k) or run[k].low in ("et", "ou")):
        # a name of four or five words needs its verb or the end of the clause; else it stops at three words
        cut = names[2]
        idx = [t for t in idx if t <= cut]
        names = names[:3]
    return idx[0], idx[-1] + 1, any(not run[t].cap for t in names)


def _lc_other_name(run: list[_W], e: int, out: list[RuleSpan], covered: set[int]) -> None:
    """The other name of a person right after the name that ends before ``run[e]``: "mme nadia kervadec épouse
    maudet", "mme pradimont née ouardi". One word in lower case that no list knows, or a first name."""
    n = len(run)
    if e + 1 >= n or run[e].cap or run[e].low not in ("epouse", "nee", "veuve", "dite", "ep", "vve"):
        return
    w = run[e + 1]
    end = _lc_particle_name(run, e + 1, True)  # "née dos reis", "épouse de roquevert"
    if end and not any(t in covered for t in range(e + 1, end)):
        out.append(RuleSpan(w.start, run[end - 1].end, "PERSON", 5, "title"))
        covered.update(range(e + 1, end))
    elif not w.cap and _lc_kind(w) in ("oov", "first") and len(w.low) >= 3 and e + 1 not in covered:
        out.append(RuleSpan(w.start, w.end, "PERSON", 5, "title"))
        covered.add(e + 1)


def _lc_title_support(run: list[_W], s: int, e: int, full: bool) -> bool:
    """The name ``run[s:e]`` after a civility is held. A name with a first name of the lexicon is. A name of words
    that no list knows is held after an abbreviated civility; after a full-word civility ("monsieur", "madame"),
    and when its first word is also a common word or ends like a verb in the simple past ("M. quitta la salle"),
    it needs the verb of the name, "et", "ou" or the end of the clause after it."""
    lows = [w for w in run[s:e] if not w.prefix and w.low not in ("de", "du", "des", "la", "le", "d", "l")]
    if any(not w.cap and "f" in _lc_flags(w.low) and "h" not in _lc_flags(w.low) for w in lows):
        return True
    head = lows[0] if lows else run[s]
    past = not head.cap and _lc_past_shape(head.low)
    if full and past:
        return False  # "monsieur quitta le domicile", "madame refusa de signer"
    if not (full or past or (not head.cap and "c" in _lc_flags(head.low))):
        return True
    n = len(run)
    return e >= n or (not run[e].cap and run[e].low in ("et", "ou")) or _lc_third_follows(run, e)


def _lc_titles(text: str, run: list[_W], out: list[RuleSpan], covered: set[int], typed: bool = True) -> None:
    """Rule A: a civility, then a name typed in lower case ("M. zadou mila", "mme claire dufresne", "maître karim
    ben salah", "madame de la tour", "mme veuve pradimont"). A plural civility takes a second name after "et"."""
    n = len(run)
    for i, w in enumerate(run):
        if w.low not in _LC_CIVILITY_WORDS:
            continue
        civ = _lc_civility(text, run, i)
        if not civ:
            continue
        full = w.low in _LC_FULL_CIVILITIES
        if (full and (not typed or w.cap)) or (civ == "me" and not typed):
            continue  # a full-word civility: only in lower case, in a sentence typed without capitals; "Me" at
            #           the start of a sentence: only in such a sentence
        at = i
        if civ == "strong" and i + 2 < n and run[i + 1].low in ("veuve", "veuf") and not run[i + 1].cap:
            at = i + 1  # "mme veuve pradimont": the name stands after "veuve"
        got = _lc_name_after(text, run, at, civ)
        if (got is None or not got[2]) and civ == "strong" and at == i and not _title(w) \
                and not (_bare_m(w) and i > 0 and run[i - 1].digit):
            # a civility typed in lower case and a name with its capitals ("m. Kervadec", "mme PRADIMONT", "dr
            # Roquevert"): the rules above do not read "m." or "mme" as a title, so the name is taken here, as they
            # take it after "M." or "Mme"
            span = _take_person(run, i + 1)
            k = span[1] if span and span[0] == i + 1 else 0
            while k:
                out.append(RuleSpan(run[span[0]].start, run[span[1] - 1].end, "PERSON", 5, "title"))
                covered.update(range(span[0], span[1]))
                k = span[1]
                span = _take_person(run, k + 1) if w.low in _LC_PLURAL_CIVILITIES and k + 1 < n \
                    and run[k].low in ("et", ",") and not run[k].cap else None
                k = span[1] if span and span[0] == k + 1 else 0
            continue
        while got is not None:
            s, e, lower = got
            if lower and not _lc_title_support(run, s, e, full or civ == "me"):
                break
            if lower:
                out.append(RuleSpan(run[s].start, run[e - 1].end, "PERSON", 5, "title"))
                covered.update(range(s, e))
                _lc_other_name(run, e, out, covered)
            if w.low in _LC_PLURAL_CIVILITIES and e + 1 < n and run[e].low == "et" and not run[e].cap:
                got = _lc_name_after(text, run, e, "strong")
            else:
                break


def _lc_cue_before(run: list[_W], i: int, flags: str) -> bool:
    """What stands before an ambiguous first name (also a common word or a verb form) lets it be a person: the
    start of the clause, a noun that names a person ("mon client"), a preposition, a conjunction or a verb.
    Never an ordinary noun or adjective ("droit romain"), nor a pronoun before a verb form ("il marie")."""
    if i == 0:
        return True
    p = run[i - 1]
    low = p.low
    if "v" in flags and (low in _LC_SUBJECTS or low in _LC_CLITICS):
        return False
    sets = _lc_sets()
    if p.prefix:
        if low != "d" or i < 2:
            return low in ("d", "qu")
        q = run[i - 2]  # "l'épouse d'olivier kervadec" is a person; "une bouteille d'huile d'olive" is a thing
        qf = _lc_flags(q.low)
        return q.low in sets["persons"] or q.low in _ROLE_WORDS or not ("p" in qf or "c" in qf)
    if low in sets["persons"] or low in _ROLE_WORDS:
        return True
    if _lc_is_verb(p) and "p" not in _lc_flags(low) and "c" not in _lc_flags(low):
        return True
    return low in sets["hard"] and low not in _LC_SUBJECTS and low not in _LC_CLITICS and low not in sets["plain"] \
        and low not in lx.legal_terms() and low not in lx.months_days()


def _lc_joins_person(run: list[_W], e: int) -> bool:
    """"et" or "ou" at ``run[e]`` joins another person or a noun phrase: "et emma leroy", "et son mari", "et mme
    x". An adjective or a verb after it belongs to the word before ("lucas grand et fort", "thomas court et
    saute")."""
    if e + 1 >= len(run):
        return True
    x = run[e + 1]
    return x.cap or "f" in _lc_flags(x.low) or x.low in _LC_DETERMINERS or bool(_title(x)) \
        or x.low in ("m", "mme", "mlle", "mr", "dr", "me") or _lc_kind(x) == "oov"


def _lc_name_frame(run: list[_W], i: int, e: int, cue: bool, loose: bool = False) -> bool:
    """The words ``run[i:e]`` stand where a name stands. Before them: the start of the clause, a noun that names a
    person (``cue``), a preposition, a verb, "et" or "ou". After them: the end of the clause, "et" or "ou", a verb
    in the third person ("paul mercier conteste", "mon client paul petit a été licencié"), or a preposition and
    the first name of another person ("oppose paul chevalier à lucas berger"). ``loose``: any preposition after
    them ("zadou mila contre son employeur"), for two words that are a name by themselves."""
    n = len(run)
    if i > 0 and not cue:
        p = run[i - 1]
        if not (p.text in ("à", "À") or p.low in _LC_PREPOSITIONS or (not p.cap and p.low in ("et", "ou"))
                or (p.prefix and p.low == "d") or _lc_is_verb(p) or p.low in _LC_NAME_CUES):
            return False
    if e >= n:
        return True
    if run[e].cap:
        return False
    if run[e].low in ("et", "ou"):
        return loose or _lc_joins_person(run, e)
    if run[e].text in ("à", "À") or run[e].low in _LC_PREPOSITIONS:
        if loose or (e + 1 < n and not run[e + 1].cap and _lc_flags(run[e + 1].low) == "f"):
            return True
    j = e
    while j < n and j - e < 3 and run[j].low in _LC_SKIPPED and not run[j].cap:
        j += 1
    return j < n and _lc_finite_third(run[j])


def _lc_first_names(text: str, run: list[_W], out: list[RuleSpan], covered: set[int]) -> None:
    """Rule B: a first name of the lexicon typed in lower case, then a word that no French list knows ("mon client
    jean dupont", "sophie martin-leroy"). Not after a determiner, "en" or a number ("un jean", "en pierre"), not
    after a place word ("rue jean jaurès"), not after a title (rule A). A first name that is also a common word or
    a verb form needs a cue before it (:func:`_lc_cue_before`)."""
    n = len(run)
    sets = _lc_sets()
    for i, w in enumerate(run):
        if i in covered or i + 1 >= n or w.digit or w.prefix or w.initial or len(w.low) < 3:
            continue
        flags = _lc_flags(w.low)
        if "f" not in flags or "h" in flags:
            continue
        if w.cap and not _is_first_name(w):
            continue
        plain_first = flags == "f"  # no other reading
        prev = run[i - 1] if i > 0 else None
        cue = False
        if prev is not None:
            if prev.digit or ((_title(prev) and prev.cap or _lc_civility(text, run, i - 1))
                              and not (prev.cap and prev.low in _LC_FULL_CIVILITIES)) \
                    or prev.low in _PLACE_WORDS or prev.low in ("en", "arret", "arrets", "jurisprudence") \
                    or prev.low in _LC_DETERMINERS:
                continue  # after a civility the name is rule A's (lower-case "me" is the pronoun, no civility;
                #           "le docteur", "un professeur" are common nouns: the name after them is read here)
            cue = prev.low in sets["persons"] or prev.low in _ROLE_WORDS
        if not plain_first and not _lc_cue_before(run, i, flags):
            continue
        # one more given name ("jean pierre dupont"), then the surname
        k = i + 1
        given = [i]
        if not run[k].cap and "f" in _lc_flags(run[k].low) and "h" not in _lc_flags(run[k].low) \
                and _lc_name_next(run, k + 1) and k + 1 not in covered:
            given.append(k)
            k += 1
        end = _lc_particle_name(run, k, False)
        s = run[k]
        kind = _lc_kind(s)
        last = k + 1 >= n
        # the first name stands where a name starts: the start of the clause, a person noun, a preposition, a verb
        framed = plain_first and len(given) == 1 and _lc_name_frame(run, i, n, cue)
        if not end and framed and not s.cap:
            end = _lc_given_particle(run, k)  # "paul de roquevert conteste", "mon client karim ait ouardi"
        sflags = _lc_flags(s.low) if not s.cap else ""
        if end:
            pass
        elif s.cap:
            if not (_lc_cap_name_word(s) and kind != "common"):
                continue
            end = k + 1
        elif kind == "oov" and len(s.low) >= 3:
            end = k + 1
            # a second surname word ("maria kervadec talvez"), only before a verb or the end of the clause
            if end < n and not run[end].cap and _lc_kind(run[end]) == "oov" and len(run[end].low) >= 3 \
                    and (end + 1 >= n or _lc_verb_follows(run, end + 1)):
                end += 1
        elif kind == "first" and plain_first:
            end = k + 1  # two first names ("jean martin", "paul henri")
        elif kind == "amb" and plain_first and (last or _lc_verb_follows(run, k + 1) or (
                not run[k + 1].cap and run[k + 1].low in ("et", "ou") and _lc_joins_person(run, k + 1))):
            end = k + 1  # "jean pierre conteste", "thomas leroy et emma leroy"
        elif kind == "common" and plain_first and len(given) == 1 and s.low in sets["common"] \
                and _lc_name_frame(run, i, k + 1, cue) and (i > 0 or (k + 1 < n and _lc_third_follows(run, k + 1))):
            end = k + 1  # a surname that is also a common word: "mon client paul petit a été licencié" (the word
            #              itself, no feminine or plural form); alone in its clause the two words are no name
            #              ("habits : jean blanc, chemise rayée")
        elif framed and ((kind == "oov" and len(s.low) == 2 and _lc_third_follows(run, k + 1))
                         or (s.low in sets["common"] and "h" not in sflags and "s" not in sflags
                             and len(s.low) >= 3 and (
                             _lc_verb_follows(run, k + 1, strict=True) if "v" in sflags
                             else _lc_third_follows(run, k + 1)))):
            end = k + 1  # right before its verb: a short surname ("paul ly conteste"), a surname that is also a
            #              title ("mon client paul comte a été licencié") or a verb form ("paul porte conteste";
            #              "paul porte plainte" is a verb and its object)
        else:
            continue
        words = given + [t for t in range(k, end) if not run[t].prefix and run[t].low not in _LC_NAME_PARTICLES
                         and run[t].low not in ("de", "du", "des", "le", "la")]
        if all(run[t].cap for t in words) or any(t in covered for t in range(i, end)):
            continue
        if w.cap and (kind != "first" or s.cap or end != k + 1
                      or not (_lc_verb_follows(run, end) if end < n else cue or _starts_sentence(text, w.start))):
            # a capital on the first name only: the word in lower case must be a first name of the lexicon with no
            # other reading ("Zadou mila", "Paul henri conteste"), and the verb of the name follows, or the clause
            # ends after the first words of a sentence or after a person noun. A word that no list knows is too
            # often a job or an ordinary word.
            continue
        if len(w.low) == 3 and not cue and not (end >= n or _lc_verb_follows(run, end)) \
                and not (prev is not None and (prev.low in _LC_PREPOSITIONS or _lc_is_verb(prev))):
            continue  # a first name of three letters may be an abbreviation ("max", "sam")
        first = i
        if prev is not None and not w.cap and i - 1 not in covered and not prev.cap and not prev.prefix \
                and _lc_kind(prev) == "oov" and len(prev.low) >= 4 and kind == "oov" \
                and _lc_name_frame(run, i - 1, end, i > 1 and (run[i - 2].low in sets["persons"]
                                                               or run[i - 2].low in _ROLE_WORDS)) \
                and not (i > 1 and (run[i - 2].low in _LC_DETERMINERS or run[i - 2].digit)):
            first = i - 1  # the surname first, then two given names: "nelba thi zadou a été licenciée"
        out.append(RuleSpan(run[first].start, run[end - 1].end, "PERSON", 4, "first_name"))
        covered.update(range(first, end))
        _lc_other_name(run, end, out, covered)


def _lc_surname_first(text: str, run: list[_W], out: list[RuleSpan], covered: set[int]) -> None:
    """A name typed surname first, with no civility: a word that no list knows, then a first name of the lexicon
    with no other reading, where a name stands ("zadou mila conteste son licenciement", "mon client s'appelle
    nelba samir", :func:`_lc_name_frame`)."""
    n = len(run)
    sets = _lc_sets()
    for i in range(n - 1):
        w, f = run[i], run[i + 1]
        if f.cap or w.cap or len(f.low) < 3 or len(w.low) < 4 or _lc_flags(f.low) != "f":
            continue
        if i in covered or i + 1 in covered or w.prefix or f.prefix or _lc_kind(w) != "oov":
            continue
        prev = run[i - 1] if i > 0 else None
        if prev is not None and (prev.digit or prev.low in _LC_DETERMINERS or prev.low in _PLACE_WORDS
                                 or _lc_civility(text, run, i - 1) or i - 1 in covered):
            continue
        cue = prev is not None and (prev.low in sets["persons"] or prev.low in _ROLE_WORDS)
        if _lc_name_frame(run, i, i + 2, cue, loose=True):
            out.append(RuleSpan(w.start, f.end, "PERSON", 4, "first_name"))
            covered.update((i, i + 1))


def _lc_lone_first_names(text: str, run: list[_W], out: list[RuleSpan], covered: set[int]) -> None:
    """A noun that names a person, then one first name of the lexicon with no other reading, then the verb of the
    name, "et", "ou" or the end of the clause: "mon client kevin a été licencié", "ma fille emma". After a plural
    noun, the first names that "et" or "ou" join: "les enfants emma et lucas vivent chez madame"."""
    n = len(run)
    sets = _lc_sets()

    def person_noun(x: _W) -> bool:
        low = x.low
        return not x.cap and (low in sets["persons"] or low in _ROLE_WORDS or (
            low.endswith("s") and (low[:-1] in sets["persons"] or low[:-1] in _ROLE_WORDS)))

    def lone(k: int) -> bool:
        x = run[k]
        return k not in covered and not x.cap and not x.prefix and len(x.low) >= 3 and _lc_flags(x.low) == "f" \
            and _lc_name_frame(run, k, k + 1, True)

    for i in range(1, n):
        if i - 1 in covered or not person_noun(run[i - 1]) or not lone(i):
            continue
        k = i
        while True:
            out.append(RuleSpan(run[k].start, run[k].end, "PERSON", 4, "first_name"))
            covered.add(k)
            if k + 2 < n and not run[k + 1].cap and run[k + 1].low in ("et", "ou") and lone(k + 2):
                k += 2
            else:
                break


def _lc_form(w: _W) -> bool:
    """A legal form in any letter case: a form the engine classes "strong" ("sas", "sarl", "selarl"); a weak form
    ("sa", "sc", "co") only as the rules above read it, in capitals."""
    strength = LEGAL_FORMS.get(w.low)
    if w.digit or w.prefix:
        return False
    if strength is None:
        return w.low in _LC_EXTRA_FORMS  # "earl", "gaec": a name in lower case before it is a company name
    return strength == "strong" or _legal_form(w) is not None


def _lc_org_word(w: _W) -> str:
    """A word of a company name typed in lower case: "name" (a first name, a surname-like word or a word no list
    knows), "generic" (a company word or an ordinary noun: "conseil", "transports") or "" (not a name word)."""
    if w.digit or w.prefix or w.initial:
        return ""
    if w.cap:
        return "name" if _org_word(w, allow_article=False) else ""
    low = w.low
    if low in ORG_GENERIC and low not in ORG_TRIGGERS:
        return "generic"
    if 2 <= len(low) <= 8 and low.isalnum() and low[0].isalpha() and any(c.isdigit() for c in low):
        return "code"  # "k2", "b2b": a word with a digit, a name word only between a company cue and a form
    kind = _lc_kind(w)
    if kind in ("first", "amb", "oov", "common"):
        return "name" if len(low) >= 3 else ""
    return "generic" if kind == "plain" else ""


def _lc_headed_company(run: list[_W], i: int, subject: bool) -> Optional[tuple[int, int]]:
    """The company name before the legal form ``run[i]`` when a noun stands before the name: "le garage julien
    sarl", "l'agence zorvane sas", "litige avec transports zelmadec sarl", "mon employeur talvenec eurl". A strong
    legal form closes a company name: the words before it go back to an article, a preposition, a verb, a person
    noun or the start of the clause (there the company needs its verb, ``subject``). The span starts at the first
    name word (a first name or a word that no list knows); the nouns before it stay readable. Before a form that
    is also a word ("sas") the name word must be a first name with no other reading or a word that no list knows.
    Returns token indices or None."""
    w = run[i]
    legal = lx.legal_terms()
    k = i - 1
    while k >= 0 and i - k <= 6 and not run[k].cap and run[k].low not in ORG_TRIGGERS \
            and ((_lc_org_word(run[k]) and "i" not in _lc_flags(run[k].low)) or run[k].low in _ORG_CONNECT
                 or (run[k].low in legal and not run[k].digit and not lx.verb_form(run[k].low))):
        k -= 1  # a legal noun may be the noun before the name ("le dossier kervadec sarl")
    s = k + 1
    while s < i and run[s].low in _ORG_CONNECT:
        s += 1
    lead = s > k + 1  # an article or a preposition stands before the first word
    e = i
    while e > s and run[e - 1].low in _ORG_CONNECT:
        e -= 1
    f = next((t for t in range(s, e) if len(run[t].low) >= 3 and (
        _lc_kind(run[t]) in ("oov", "first", "amb") or _lc_flags(run[t].low) == "c")), None)
    if f is None:
        return None  # ("le contrat de travail sarl": a word of the vocabulary is no proof of a name)
    kind = _lc_kind(run[f])
    if w.low in _LC_FORM_WORDS and not w.upper and kind not in ("oov", "first"):
        return None
    before = run[k] if k >= 0 else None
    person = before is not None and (before.low in _lc_sets()["persons"] or before.low in _ROLE_WORDS)
    if f == s:
        # no noun before the name word: the cases that the rule with no cue does not read ("mon employeur x eurl")
        return (f, e) if person and not lead and kind in ("oov", "first") else None
    if lead or person:
        return f, e
    if before is None:
        return (f, e) if subject else None
    if before.text in ("à", "À") or before.low in _LC_PREPOSITIONS or before.low in _LC_DETERMINERS \
            or _lc_is_verb(before):
        return f, e
    return None


_LC_PARTY_OPEN_RE = re.compile(
    r"(?i)(?:\bc[ \t]?/|/|\b(?:contre|demandeu(?:r|se)|demanderesse|d[ée]fendeu(?:r|se)|d[ée]fenderesse|adversaire|"
    r"partie[ \t]+adverse|clients?|cliente|employeur|bailleur|locataire|soci[ée]t[ée])[ \t]*:)[ \t]*$")


def _lc_party_opening(text: str, run: list[_W]) -> bool:
    """The clause opens a party: it stands right after "c/", "/" or a party label and a colon ("contre :",
    "défendeur :", "client :")."""
    start = run[0].start
    return bool(_LC_PARTY_OPEN_RE.search(text[max(0, start - 24):start]))


def _lc_cued_companies(text: str, run: list[_W], out: list[RuleSpan], covered: set[int]) -> None:
    """A company cue and a name with no legal form, where the shape holds the name: the name, then "&" or "et" and
    a word such as "associés", "fils", "cie" ("le cabinet kervadec & associés"); or a first name of the lexicon
    with no other reading before the verb of the company or the end of the clause ("la société julien a
    licencié"). A cue and one word that no list knows is not enough ("la société éponyme")."""
    n = len(run)
    for t, w in enumerate(run[:-1]):
        if w.cap or w.low not in ORG_TRIGGERS:
            continue
        k = t + 1
        x = run[k]
        if not x.cap and x.low in _ORG_SKIP:
            # a spelled legal form after the cue: "la société par actions simplifiée kerdalis", "la société civile
            # immobilière zorvane"; then one word that no list knows or a first name, at the end of the clause
            # or before the verb of the company
            j = k
            while j < n and not run[j].cap and (run[j].low in _ORG_SKIP or run[j].prefix):
                j += 1
            if j - k >= 2 and run[j - 1].low in _LC_SPELLED_FORM_ENDS and j < n and j not in covered \
                    and not run[j].cap and _lc_kind(run[j]) in ("oov", "first") and len(run[j].low) >= 3 \
                    and (j + 1 >= n or _lc_third_follows(run, j + 1)):
                out.append(RuleSpan(run[j].start, run[j].end, "ORG", 6, "legal_form"))
                covered.add(j)
            continue
        if k in covered or x.cap or len(x.low) < 3 or _lc_kind(x) not in ("oov", "first", "amb", "common"):
            continue
        e = k + 1
        if e < n and not run[e].cap and _lc_kind(run[e]) in ("oov", "first") and len(run[e].low) >= 3 \
                and e + 2 < n and run[e + 1].low in ("et", "&"):
            e += 1  # two names before the join word ("le cabinet kervadec talvez & associés")
        if e + 1 < n and run[e].low in ("et", "&") and not run[e + 1].cap and run[e + 1].low in _ORG_ET_JOIN:
            out.append(RuleSpan(x.start, run[e + 1].end, "ORG", 6, "legal_form"))
            covered.update(range(k, e + 2))
        elif e == k + 1 and _lc_flags(x.low) == "f" and (e >= n or _lc_third_follows(run, e)) \
                and not (e < n and _lc_form(run[e])):
            out.append(RuleSpan(x.start, x.end, "ORG", 6, "legal_form"))
            covered.add(k)


def _lc_companies(text: str, run: list[_W], out: list[RuleSpan]) -> None:
    """Rule C: a company name typed in lower case with a legal form in any letter case. The name stands between a
    company cue and the form ("la société durand sas", "chez dupont conseil sarl"); or before the form with no cue,
    at the start of the clause or after a preposition or a verb, when it starts with a word no list knows or with a
    first name ("contre durand sas"); or after a noun ("le garage julien sarl", :func:`_lc_headed_company`); or
    after an article and the form ("la sarl durand"). The span is the name; the legal form stays outside, as in
    the rules above."""
    n = len(run)
    for i, w in enumerate(run):
        if not _lc_form(w):
            continue
        # the name before the form ("kerdalis et compagnie", "kerdalis & co": the word after "et" or "&" is no cue)
        k = i - 1
        while k >= 0 and i - k <= 5 and (
                (run[k].low in _ORG_ET_JOIN and k > 0 and run[k - 1].low in ("et", "&") and not run[k].cap)
                or ((_lc_org_word(run[k]) or (run[k].low in _ORG_CONNECT and not run[k].cap))
                    and run[k].low not in ORG_TRIGGERS)):
            k -= 1
        s = k + 1
        while s < i and (run[s].low in _ORG_SKIP or run[s].low in _ORG_CONNECT) and not run[s].cap:
            s += 1
        e = i
        while e > s and run[e - 1].low in _ORG_CONNECT and not run[e - 1].cap:
            e -= 1
        words = [run[t] for t in range(s, e) if run[t].low not in _ORG_CONNECT or run[t].cap]
        if k >= 0 and (run[k].low in ORG_TRIGGERS or run[k].low == "chez") and words \
                and any(_lc_org_word(x) in ("name", "code") for x in words) and any(not x.cap for x in words):
            out.append(RuleSpan(run[s].start, run[e - 1].end, "ORG", 6, "legal_form"))
            continue
        # no cue word: the words right before the form, up to a preposition, a verb or the start of the clause
        t = i - 1
        while t >= 0 and i - t <= 4 and not run[t].cap and _lc_org_word(run[t]):
            t -= 1
        # at the start of a clause the company needs its verb, or another party after "et", "ou", "contre", "c/":
        # a note word and a legal form are no company ("compta sci à refaire", "docs sasu reçus ce matin")
        subject = i + 1 < n and (_lc_finite_third(run[i + 1]) or (
            not run[i + 1].cap and ((run[i + 1].low in ("et", "ou", "contre", "c", "vs")
                                     and not (i + 2 < n and _lc_form(run[i + 2]))) or (
                run[i + 1].low in _LC_SKIPPED and i + 2 < n and _lc_finite_third(run[i + 2])))))
        opens = _lc_party_opening(text, run)
        if i - t > 1 and _lc_kind(run[t + 1]) in ("oov", "first") and len(run[t + 1].low) >= 3 and (
                (t < 0 and (subject or opens)) or (t >= 0 and (
                    run[t].text in ("à", "À") or run[t].low in _LC_PREPOSITIONS or _lc_is_verb(run[t])
                    or (not run[t].cap and run[t].low in ("de", "du", "des", "d", "au", "aux", "et", "ou"))))):
            out.append(RuleSpan(run[t + 1].start, run[i - 1].end, "ORG", 6, "legal_form"))
            continue
        headed = _lc_headed_company(run, i, subject or opens)
        if headed:
            out.append(RuleSpan(run[headed[0]].start, run[headed[1] - 1].end, "ORG", 6, "legal_form"))
            continue
        # the name after the form: article, form, name
        if i + 1 >= n or w.sent_end:
            continue
        prev = run[i - 1] if i > 0 else None
        article = prev is not None and not prev.cap \
            and (prev.low in _LC_FORM_ARTICLES or (prev.prefix and prev.low == "l"))
        # the form opens a party after "c/" or after a label ("contre : sarl x", "c/ sas y")
        party = i == 0 and opens
        if not article:
            if not (w.low not in _LC_FORM_WORDS or w.upper or party):
                continue
            if prev is not None and not (prev.low in _LC_PREPOSITIONS or prev.text in ("à", "À")
                                         or (prev.prefix and prev.low == "d")):
                continue
        first = run[i + 1]
        if not first.cap and (first.low in ("le", "la", "les") or (first.prefix and first.low == "l")):
            # a name that starts with an article: "la sci les tilleuls", "sarl les jardins de provence". Nouns
            # after the article, to the end of the clause or to the verb of the company; never a pronoun and a
            # verb ("la sarl les a licenciés") or a time ("gérer la sas les mois suivants")
            e = i + 2
            nouns = 0
            while e < n and e - i <= 6 and not run[e].cap and not _lc_form(run[e]):
                x = run[e]
                if nouns and x.low in _ORG_CONNECT and x.low not in ("et", "&") and e + 1 < n:
                    e += 1
                elif _lc_kind(x) in ("plain", "common", "oov", "first", "amb") and x.low not in _LC_TIME_WORDS \
                        and len(x.low) >= 3:
                    nouns += 1
                    e += 1
                else:
                    break
            while e > i + 2 and run[e - 1].low in _ORG_CONNECT:
                e -= 1
            if nouns and (e >= n or _lc_third_follows(run, e)):
                out.append(RuleSpan(first.start, run[e - 1].end, "ORG", 6, "legal_form"))
            continue
        if first.cap:
            # a form typed in lower case before a capitalised name ("la sarl Dupont Conseil")
            span = None if (_legal_form(w) or w.cap) else _org_forward(run, i + 1)
            if span and span[0] == i + 1 and _meaningful_org(run, *span, roles=True):
                out.append(RuleSpan(run[span[0]].start, run[span[1] - 1].end, "ORG", 6, "legal_form"))
            continue
        g = i + 1  # nouns of a trade before the name: "la sarl boulangerie zorvane"
        while g < n - 1 and g - i <= 2 and not run[g].cap and (run[g].low in ORG_TRADES or _is_trade(run[g].low)):
            g += 1
        name = run[g]
        if name.cap or _lc_kind(name) not in ("oov", "first") or len(name.low) < 3 \
                or (_lc_kind(name) == "oov" and name.low.endswith(_LC_ADJECTIVE_ENDINGS)):
            continue
        e = g + 1
        while e < n and e - i <= 5 and not run[e].cap and not _lc_form(run[e]):
            if run[e].low in ("et", "&") and e + 1 < n and (run[e + 1].low in _ORG_ET_JOIN or (
                    not run[e + 1].cap and _lc_kind(run[e + 1]) == "oov" and len(run[e + 1].low) >= 4)):
                e += 2  # "la sarl kervadec et fils", "la selarl kervadec et talvez"
            elif _lc_org_word(run[e]):
                e += 1
            else:
                break
        out.append(RuleSpan(first.start, run[e - 1].end, "ORG", 6, "legal_form"))


_LC_FIELD_RE = re.compile(
    r"(?im)(?:^|[,;(])[ \t]*(?:(?:[-*•>–]|\d{1,2}[.)]|[a-z]\))[ \t]*)?(?P<label>nom(?:\s+de\s+naissance|\s+d['’]usage|\s+marital|\s+de\s+famille)?|pr[ée]noms?)"
    r"\s*:[ \t]*(?P<value>[^\n,;:]{2,60})")


def _lc_field_pair(text: str, m: "re.Match") -> bool:
    """The field ``m`` stands next to the other name field, and that one holds a first name of the lexicon ("nom :
    zadou, prénom : mila"). A field alone with a word that no list knows may hold a placeholder ("nom : tbc")."""
    for o in _LC_FIELD_RE.finditer(text, max(0, m.start() - 120), min(len(text), m.end() + 120)):
        if o.start() != m.start():
            words = _words(o.group("value").rstrip())
            if words and all(len(w.low) >= 3 and "f" in _lc_flags(w.low) and "h" not in _lc_flags(w.low)
                             for w in words):
                return True
    return False


def _lc_field_spans(text: str, out: list[RuleSpan]) -> None:
    """Form fields typed in lower case, on one line each or after a comma: "nom : zadou", "nom : zadou, prénom :
    mila". The value is one to three words, each a first name of the lexicon with no other reading or a word that
    no list knows: no function word, no ordinary word ("nom : à compléter", "prénom : non renseigné"). Only on a
    line typed without capitals."""
    for m in _LC_FIELD_RE.finditer(text):
        raw = m.group("value")
        vs = m.start("value")
        words = _words(raw.rstrip())
        if not 1 <= len(words) <= 3 or all(w.cap for w in words) or not m.group("label")[:1].islower():
            continue  # a capitalised value is read by the field rule above; a capitalised label is correct text
        line = _words(text[text.rfind("\n", 0, m.start()) + 1:(text.find("\n", m.end()) + 1 or len(text) + 1) - 1])
        if any(w.cap and k and not _lc_neutral_capital(w) for k, w in enumerate(line)):
            continue  # the line has its capitals ("Nom : douteux, Prénom : incomplet")
        kinds = [_lc_kind(replace(w, cap=False, upper=False)) if len(w.low) >= 3 else "no" for w in words]
        if all(k == "first" for k in kinds) or (all(k in ("oov", "first") for k in kinds) and all(
                len(w.low) >= 4 for w in words) and _lc_field_pair(text, m)):
            out.append(RuleSpan(vs + words[0].start, vs + words[-1].end, "PERSON", 5, "field"))


def _lc_captions(text: str, runs: list[list[_W]], out: list[RuleSpan]) -> None:
    """A case caption typed in lower case with bare surnames: "kervadec c/ talvez", "affaire kervadec c/ talvez",
    "dossier martin contre kervadec".
    The word before "c/" and the word after it are parties when each is a first name of the lexicon with no other
    reading or a word that no list knows. The left word stands alone or after a legal noun or a preposition; the
    right word stands alone or where a name stands. A party that another rule has read keeps that reading."""
    legal = lx.legal_terms()

    def taken(w: _W) -> bool:
        return any(sp.start < w.end and w.start < sp.end for sp in out)

    def party(w: _W) -> bool:
        return not w.cap and not w.digit and not w.prefix and len(w.low) >= 4 \
            and _lc_kind(w) in ("oov", "first") and not taken(w)

    def pair(x: _W, y: _W) -> bool:
        # two bare words around "c/" are names when one is a first name of the lexicon, or both are long words
        # that no list knows ("sdc c/ syndic", "copro contre proprio" are no names)
        return "f" in _lc_flags(x.low) or "f" in _lc_flags(y.low) or (len(x.low) >= 6 and len(y.low) >= 6)

    for r in range(len(runs) - 1):
        a, b = runs[r], runs[r + 1]
        c = a[-1]
        if c.low != "c" or text[c.end:c.end + 1] != "/" or text[c.end + 1:b[0].start].strip():
            continue
        left = len(a) >= 2 and party(a[-2]) and (len(a) == 2 or a[-3].low in legal or a[-3].low in _LC_PREPOSITIONS)
        right = party(b[0]) and (len(b) == 1 or _lc_name_frame(b, 0, 1, True))
        other = taken(b[0]) or _lc_form(b[0]) or bool(_lc_civility(text, b, 0))  # the other party is a name too
        if left and (other or (right and pair(a[-2], b[0]))):
            out.append(RuleSpan(a[-2].start, a[-2].end, "PERSON", 4, "first_name"))
        if right and len(a) >= 2 and (taken(a[-2]) or (left and pair(a[-2], b[0]))):
            out.append(RuleSpan(b[0].start, b[0].end, "PERSON", 4, "first_name"))
    # the same with "contre" between two bare names: "dossier martin contre kervadec"
    for run in runs:
        n = len(run)
        for j in range(1, n - 1):
            if run[j].cap or run[j].low != "contre":
                continue
            x, y = run[j - 1], run[j + 1]
            if party(x) and party(y) and pair(x, y) \
                    and (j == 1 or run[j - 2].low in legal or run[j - 2].low in _LC_PREPOSITIONS) \
                    and (j + 2 >= n or _lc_name_frame(run, j + 1, j + 2, True)):
                out.append(RuleSpan(x.start, x.end, "PERSON", 4, "first_name"))
                out.append(RuleSpan(y.start, y.end, "PERSON", 4, "first_name"))


_LC_DOTTED_FORM_RE = re.compile(
    r"(?i)(?<![\w.])(s\.a\.s\.u|s\.a\.r\.l|s\.a\.s|e\.u\.r\.l|s\.c\.i|s\.n\.c|s\.c\.p|s\.e\.l\.a\.r\.l)\.?(?![\w])")


def _lc_company_before(text: str, words: list[_W], form: str, start: int, end: int, out: list[RuleSpan]) -> None:
    """Rule C for a legal form that the tokens do not show as one word after the name (a form with dots, a form in
    brackets): ``words`` stand before the form ``text[start:end]``."""
    found: list[RuleSpan] = []
    _lc_companies(text, words[-7:] + [_W(form, start, end, form, False, False)], found)
    for sp in found:
        if sp.end <= start and not any(o.start == sp.start and o.end == sp.end for o in out):
            out.append(sp)


def _lc_across_clauses(text: str, runs: list[list[_W]], out: list[RuleSpan]) -> None:
    """Names that a dot, a bracket or a comma cuts in two clauses:

    - an initial with a dot between a strong civility and a surname ("m. k. kervadec conteste", "m. j.p. kervadec
      a signé"): the surname is a first name with no other reading or a word that no list knows, before its verb,
      "et", "ou" or the end of the clause;
    - "ep." or "vve" after a name, then the other surname ("mme vellin ep. kervadec conteste");
    - a legal form with dots or in brackets after a company name ("la société kerdalis s.a.s.", "la société
      kerdalis (sas) refuse"), and a name, a comma, then the form and "au capital" ("kerdalis, sas au capital de
      1000 euros")."""

    def taken(w: _W) -> bool:
        return any(sp.start < w.end and w.start < sp.end for sp in out)

    def surname(b: list[_W], k: int) -> bool:
        if k >= len(b):
            return False
        w = b[k]
        return not w.cap and not w.prefix and len(w.low) >= 3 and _lc_kind(w) in ("oov", "first") and not taken(w) \
            and (k + 1 >= len(b) or _lc_third_follows(b, k + 1) or (not b[k + 1].cap and b[k + 1].low in ("et", "ou")))

    for r in range(len(runs) - 1):
        a, b = runs[r], runs[r + 1]
        last = a[-1]
        if last.cap or not last.sent_end or not (len(last.low) == 1 or last.low in ("ep", "vve")) \
                or not re.fullmatch(r"\.[ \u00a0]{0,2}", text[last.end:b[0].start]):
            continue
        if len(a) >= 2 and len(last.low) == 1 and last.low.isalpha() and _lc_civility(text, a, len(a) - 2) == "strong":
            k = 1 if len(b) >= 2 and len(b[0].low) == 1 and b[0].low.isalpha() and not b[0].cap else 0
            if surname(b, k):
                out.append(RuleSpan(last.start, b[k].end, "PERSON", 5, "title"))
        elif last.low in ("ep", "vve") and len(a) >= 3 and taken(a[-2]) and surname(b, 0):
            out.append(RuleSpan(b[0].start, b[0].end, "PERSON", 5, "title"))
    for r in range(1, len(runs)):
        a, b = runs[r - 1], runs[r]
        # the other surname after a comma or a bracket: "madame x, née y, a renoncé", "mme x (née y) demande"
        if len(b) >= 2 and not b[0].cap and b[0].low in ("nee", "epouse", "veuve") and taken(a[-1]) \
                and re.fullmatch(r"[ \u00a0]?[,(][ \u00a0]?", text[a[-1].end:b[0].start]) and not b[1].cap \
                and not taken(b[1]) and len(b[1].low) >= 3 and _lc_kind(b[1]) in ("oov", "first") \
                and (len(b) == 2 or _lc_third_follows(b, 2)):
            out.append(RuleSpan(b[1].start, b[1].end, "PERSON", 5, "title"))
    for m in (_LC_DOTTED_FORM_RE.finditer(text) if "." in text else ()):
        for run in runs:
            if run[0].start <= m.start() <= run[-1].start:
                j = next((t for t, w in enumerate(run) if w.start == m.start()), 0)
                if j:
                    _lc_company_before(text, run[:j], m.group(1).replace(".", "").lower(), m.start(), m.end(), out)
                break
    for r in range(1, len(runs)):
        a, b = runs[r - 1], runs[r]
        w = b[0]
        if not _lc_form(w):
            continue
        gap = text[a[-1].end:w.start]
        if len(b) == 1 and re.fullmatch(r"[ \u00a0]?\(", gap) and text[w.end:w.end + 1] == ")":
            _lc_company_before(text, a, w.low, w.start, w.end, out)
        elif len(b) >= 3 and b[1].low == "au" and b[2].low == "capital" and re.fullmatch(r",[ \u00a0]?", gap):
            k = len(a) - 1
            while k >= 0 and len(a) - k <= 4 and not a[k].cap and _lc_org_word(a[k]) in ("name", "generic", "code"):
                k -= 1
            first = a[k + 1] if k + 1 < len(a) else None
            if first is not None and len(first.low) >= 3 and not taken(first) \
                    and (_lc_kind(first) in ("oov", "first") or _lc_flags(first.low) == "c"):
                out.append(RuleSpan(first.start, a[-1].end, "ORG", 6, "legal_form"))


def _lc_shown_surnames(text: str, runs: list[list[_W]], typed: list[bool], out: list[RuleSpan]) -> None:
    """A surname in lower case that another name of the same text shows.

    Right after a person span, in every text: the word belongs to that name when a name typed all in lower case
    shows it ("mme nadia grenier et m. karim grenier divorcent": nothing after the first surname shows a name, the
    second name shows it). The two persons then have two names and two codes. Without this span the surname alone
    would take the code of the other person, and the restore would write a wrong first name.

    After a first name with the capital of a keyboard, in a sentence typed without capitals ("Yanis kervadec et son
    épouse nadia kervadec", "Mme Nadia kervadec et son mari yanis kervadec"): the two words are one more person."""
    shown, lower = set(), set()
    for sp in out:
        if sp.kind == "PERSON":
            surface = text[sp.start:sp.end]
            parts = surface.split()
            if len(parts) >= 2 and parts[-1].islower():
                if len(parts[-1]) >= 3:
                    shown.add(fold_word(parts[-1]))
                if len(parts[-1]) >= 2 and not any(ch.isupper() for ch in surface):
                    lower.add(fold_word(parts[-1]))  # a span with no capital comes from the lower-case rules only
    if not shown:
        return
    for run, is_typed in zip(runs, typed):
        for i in range(len(run) - 1):
            a, b = run[i], run[i + 1]
            if b.cap or b.digit or b.prefix or b.low not in shown or text[a.end:b.start] not in (" ", "\u00a0") \
                    or any(sp.start <= b.start and b.end <= sp.end for sp in out):
                continue
            before = [k for k, sp in enumerate(out) if sp.kind == "PERSON" and sp.end == a.end] \
                if b.low in lower else []
            if before:
                for k in before:
                    out[k] = replace(out[k], end=b.end)
            elif is_typed and a.cap and not a.upper and _lc_flags(a.low) == "f":
                out.append(RuleSpan(a.start, b.end, "PERSON", 5, "title"))


_LC_TITLE_BEFORE_RE = re.compile(
    r"(?i)(?:(?<![\w.])(?:m|mm|mme|mmes|mlle|mlles|melle|mdme|mr|mrs|me|dr|pr)\.?"
    r"|(?<!\w)(?:monsieur|madame|mademoiselle|ma[îi]tre|docteur|professeur|messieurs|mesdames))[ \u00a0\u202f]+$")
# a court, an administration or a public place named by its town: "le tribunal judiciaire de x", "la cour d'appel
# de x", "le conseil de prud'hommes de x", "la mairie de x"
_LC_COURT_BEFORE_RE = re.compile(
    r"(?i)(?<![\w'’-])(?:tribunal|tribunaux|cour|conseil|juridiction|chambre|greffe|parquet|barreau|pr[ée]fecture"
    r"|sous-pr[ée]fecture|mairie|commune|ville|agglom[ée]ration|m[ée]tropole|acad[ée]mie|rectorat|universit[ée]"
    r"|h[ôo]pital|chu|gare|a[ée]roport|commissariat|gendarmerie|cpam|caf|urssaf|tj|tgi|ta|caa|cph|tc)\b"
    r"[^,.;:!?()\n]{0,48}?(?<![\w'’-])(?:de|du|des|d['’])[ \u00a0\u202f]*$")
# a place cue: a verb or a noun of place and its preposition ("il habite à x", "née à x", "fait à x"), a postcode
_LC_PLACE_BEFORE_RE = re.compile(
    r"(?i)(?:(?<![\w'’-])(?:habit\w*|demeur\w*|r[ée]sid\w*|domicili\w*|vit|vivait|vivent|travaill\w*|n[ée]e?s?"
    r"|situ[ée]e?s?|sise?s?|si[èe]ge|bas[ée]e?s?|install[ée]e?s?|implant[ée]e?s?|mut[ée]e?s?|affect[ée]e?s?|fait"
    r"|audience|rendez-vous|rdv|r[ée]union|d[ée]placement|retour|chantier|agence|magasin|usine|entrep[ôo]t|site"
    r"|bureaux?)[ \u00a0\u202f]+(?:[àa]|au|en|sur|vers|pr[èe]s[ \u00a0\u202f]+de)|(?<!\d)\d{5})[ \u00a0\u202f]+$")


def lc_mention_is_text(text: str, s: int, e: int, value: str, vtoks: tuple, need: str,
                       rule_spans: Iterable[tuple[int, int]] = (), after_place_word: bool = False) -> bool:
    """v0.4.1: True when ``text[s:e]`` must not take the code of the person ``value``. Only for a person whose name
    words are stored in lower case (the lower-case rules found it; a name stored with its capitals keeps the links
    of 0.4.0). For a mention typed in lower case:

    - the "surname, first name" reading across a comma, and a reading that starts or ends inside a name that the
      rules read whole ("hugo kervalen, clara kervalen" are two persons);
    - the surname alone when it is one word that a list knows (a town, a place, an ordinary word, a first name) and
      no civility stands right before it ("m. karim grenier conteste. le grenier est une partie commune");
    - a one-word mention inside the name of a court or after a place cue ("le tribunal judiciaire de x", "il
      habite à x");
    - a first name alone after "de", "à", "en" or another word of place (``after_place_word``): the guard that the
      first name with its capital has in 0.4.0 ("au nom de x").

    For a mention with a capital: the surname alone when it is a town of the place list, or stands inside the
    name of a court or after a place cue, and no civility stands right before it."""
    sig = [t for t in vtoks if t not in PARTICLES]
    if not sig:
        return False
    typed: set = set()
    for w in value.split():
        if not any(ch.isupper() for ch in w):
            typed.update(phrase_tokens(w))
    if not all(t in typed for t in sig):
        return False
    lower = not any(ch.isupper() for ch in text[s:e])
    if lower:
        if need == "comma":
            return True
        for rs, re_ in rule_spans:
            if rs < s < re_ < e or s < rs < e < re_:
                return True
    _firsts, surname = split_person(value)
    st = tuple(t for w in surname for t in phrase_tokens(w))
    vt = tuple(vtoks)
    before = text[max(0, s - 96):s]
    cue = bool(_LC_COURT_BEFORE_RE.search(before) or _LC_PLACE_BEFORE_RE.search(before))
    alone = bool(st) and (vt == st or (len(vt) < len(st) and st[-len(vt):] == vt))
    if not alone:
        return lower and len(vt) == 1 and (cue or after_place_word)
    if len(sig) > 1 or _LC_TITLE_BEFORE_RE.search(before):
        return False
    if not lower:
        return cue or sig[0] in _lc_sets()["places"]
    return cue or bool(_lc_flags(sig[0]))


def _lc_cap_companies(run: list[_W], out: list[RuleSpan], firsts: frozenset = frozenset()) -> None:
    """A company name with its capital next to a legal form typed in lower case: "Brivanor sas m'a licencié", "chez
    Kerdalis sasu", "le garage Talvez sarl", "la sarl Pradimont a résilié". The capitalised rules need the
    form in capitals. Before a form that is also a word ("sas") the name must be a word that no list knows. The
    capital must stand inside the sentence, or the name has two words."""
    n = len(run)
    for i, w in enumerate(run):
        if w.cap or w.digit or w.prefix or LEGAL_FORMS.get(w.low) != "strong":
            continue
        # the name before the form: one to three capitalised words
        k = i - 1
        while k >= 0 and i - k <= 3 and run[k].cap and not run[k].initial and not _title(run[k]) \
                and _org_word(run[k], allow_article=False) and (k > 0 or i - k > 1 or not _lc_flags(run[k].low)
                                                                or "f" in _lc_flags(run[k].low)):
            k -= 1
        if k < i - 1 and (w.low not in _LC_FORM_WORDS or not _lc_flags(run[i - 1].low)) \
                and not (k == i - 2 and run[i - 1].start in firsts):
            # one capitalised word that opens the sentence has the capital of the sentence, no capital of its own
            # ("Honoraires sarl : 2 400 euros"); the rules of a sentence typed without capitals read it
            out.append(RuleSpan(run[k + 1].start, run[i - 1].end, "ORG", 6, "legal_form"))
            continue
        # the name after an article and the form: "la sarl Pradimont"
        prev = run[i - 1] if i > 0 else None
        if prev is not None and (prev.low in _LC_FORM_ARTICLES or (prev.prefix and prev.low == "l")) and i + 1 < n \
                and run[i + 1].cap and w.low not in _LC_FORM_WORDS:
            span = _org_forward(run, i + 1)
            if span and span[0] == i + 1 and _meaningful_org(run, *span, roles=True):
                out.append(RuleSpan(run[span[0]].start, run[span[1] - 1].end, "ORG", 6, "legal_form"))


def _lower_case_names(text: str, runs: list[list[_W]], out: list[RuleSpan]) -> None:
    """The rules for names typed in lower case (v0.4.1): a civility then a name (A); a first name then a surname,
    the surname first, a first name after a person noun (B); a company name with a cue or a legal form (C); form
    fields and case captions."""
    firsts: set = set()  # where the sentences start
    typed = _lc_typed_runs(text, runs, firsts)
    for r, run in enumerate(runs):
        if len(run) < 2:
            continue
        covered: set[int] = set()
        view = typed[r]
        # tier A, every text: an abbreviated civility, then a name in lower case; a capitalised company name next
        # to a legal form in lower case
        _lc_titles(text, view or run, out, covered, view is not None)
        _lc_cap_companies(run, out, firsts)
        if view is None:
            continue
        # tier B, a sentence typed without capitals: the other rules
        _lc_cued_companies(text, view, out, covered)
        _lc_first_names(text, view, out, covered)
        _lc_surname_first(text, view, out, covered)
        _lc_lone_first_names(text, view, out, covered)
        _lc_companies(text, view, out)
    views = [v for v in typed if v is not None]
    if views:
        _lc_field_spans(text, out)
        _lc_captions(text, views, out)
        _lc_across_clauses(text, views, out)
    _lc_shown_surnames(text, [typed[r] or run for r, run in enumerate(runs)], [v is not None for v in typed], out)


def _detect_names_fr(text: str) -> list[RuleSpan]:
    """The French rules of this module (unresolved, may overlap)."""
    out: list[RuleSpan] = []
    words = _words(text)
    runs = _runs(text, words)
    for run in runs:
        if any(w.cap for w in run):
            _analyze_run(text, run, out)
    _plural_title_lists(text, runs, out)  # v0.2 (DETECTORS): "Mmes X, Y, Z, conseillers"
    _separated_forms(text, runs, out)
    _party_definitions(text, runs, out)
    _field_spans(text, out)
    _partnerships(text, words, out)
    if lowercase_names():
        _lower_case_names(text, runs, out)  # v0.4.1: names typed in lower case
    return out


def detect_names(text: str, lang: str = "fr", protected: Optional[IntervalSet] = None,
                 domains: Iterable[str] = ()) -> list[RuleSpan]:
    """Person and company candidates from the rules (unresolved, may overlap).

    ``lang`` picks the rules: the French rules of this module, or ``locale.detect_names_en`` for "en". Both then
    go through :func:`refine_rule_spans` (v0.3): the legal-form span policy, company span edges, companies named by
    an e-mail domain of the text (plus ``domains``, the domains of the dossier when the caller knows them), the
    acronyms the text declares and the other mentions of a multi-word company name. ``protected`` is the
    interval set the rules must never enter (the masker's); without it the refinement computes it when it needs
    it.
    """
    if str(lang).lower().startswith("en"):
        from . import locale as lc

        spans = lc.detect_names_en(text)
    else:
        spans = _detect_names_fr(text)
    return refine_rule_spans(text, spans, lang, protected=protected, domains=domains)


# ---------------------------------------------------------------------------
# Span refinement (v0.3, owner S1+S4): legal form inside the company span, company span edges, companies named by
# an e-mail domain, declared acronyms, same-surface propagation. One pass for both languages: the French word
# lists are this module's, the English ones come from the LOCALE-EN bundle.
# ---------------------------------------------------------------------------

#: Company rules whose span carries strong evidence of its kind: a legal form, a form field, a declared alias or
#: acronym, an e-mail domain of the text. The kind decision of the masker (plan S2) should keep such a span an
#: organisation against a single person sighting of the same words (plan S1 guard).
STRONG_ORG_RULES = frozenset({"legal_form", "legal_form_after", "field", "partnership", "party_definition",
                              "domain", "domain_form", "acronym"})

# The word right after a position on the same line (one or two spaces, or a comma and a space), and the word right
# before one. A line break, a tab, three spaces or a cell separator (" | ") gives no match: a name never spans them.
_NEXT_WORD_RE = re.compile(r"(?P<gap>[   ]{1,2}|,[   ]{1,2})(?P<tok>[^\W\d_][\w'’\-]*\.?|&)")
_PREV_WORD_RE = re.compile(r"(?P<tok>[^\W\d_][\w'’\-]*\.?|&)(?P<gap>[   ]{1,2})$")
_WORD_RE = re.compile(r"[^\W\d_][\w'’\-]*\.?|&")
_CONNECT_WORDS = frozenset({"&", "and", "of", "the", "for", "et", "de", "du", "des", "d", "la", "le", "les", "l"})
# English head nouns that end a company name next to the suffix words of the bundle ("Acme Holding Company",
# "Acme Commercial Banking", "Acme Trust Division"), and the business-unit nouns a company name takes as its
# division, product line or market ("Acme Oncology", "Acme Footwear", "Acme Commercial Banking", "Acme
# Research"). A hand-written list of generic industry vocabulary: it names no company and no benchmark entity.
_EN_HEAD_EXTRA = frozenset({
    "company", "companies", "corporation", "enterprise", "enterprises", "organization", "organisation",
    "division", "divisions", "unit", "units", "department", "banking", "office",
    # health and life sciences
    "oncology", "cardiology", "neurology", "immunology", "dermatology", "diabetes", "vaccines", "biologics",
    "genomics", "diagnostics", "devices", "nutrition", "wellness",
    # consumer goods and retail
    "footwear", "apparel", "sportswear", "eyewear", "running", "basketball", "football", "soccer", "golf",
    "outdoor", "fitness", "beauty", "fragrance", "skincare", "grooming", "snacks", "dairy", "poultry", "meats",
    "seafood", "produce", "grocery", "bakery", "confectionery", "wholesale", "marketplace", "ecommerce",
    "e-commerce", "supercenters", "superstores",
    # finance
    "payments", "treasury", "lending", "wealth", "brokerage", "equities", "commodities", "trading",
    "underwriting", "reinsurance", "annuities", "pensions", "custody", "clearing",
    # industry, energy and transport
    "mobility", "propulsion", "avionics", "satellites", "turbines", "engines", "machinery", "tractors",
    "components", "parts", "supply", "procurement", "operations", "fulfillment", "fulfilment", "delivery",
    "cargo", "express", "renewables", "solar", "wind", "nuclear", "utilities", "materials", "coatings",
    "packaging", "printing", "agronomy", "equipment", "fleet",
    # technology, media and research
    "cloud", "hardware", "cybersecurity", "security", "platforms", "infrastructure", "automation",
    "broadcasting", "gaming", "music", "pictures", "television", "animation", "research", "innovation",
    "worldwide", "americas", "emea", "apac",
})
# Nouns after a possessive that make a capitalised word a company ("Acme's revenue", "Acme's CEO").
_CUE_NOUNS = frozenset({
    "ceo", "cfo", "cto", "coo", "cmo", "cio", "board", "revenue", "revenues", "results", "earnings", "quarter",
    "team", "teams", "staff", "employees", "office", "offices", "headquarters", "hq", "shares", "stock", "ipo",
    "merger", "acquisition", "sales", "pipeline", "roadmap", "platform", "product", "products", "customers",
    "clients", "users", "website", "pricing", "contract", "contracts", "invoice", "invoices", "account", "policy",
    "management", "leadership", "founder", "founders", "brand", "logo", "fleet", "factory", "plant", "warehouse",
    "stores", "branches", "subsidiary", "subsidiaries", "growth", "margins", "valuation", "investors",
    "shareholders", "payroll", "servers", "data", "systems", "network", "engineers", "support", "marketing",
    "finance", "legal", "hr", "it", "chiffre", "siege", "direction", "equipe", "equipes", "clients", "salaries",
    "effectifs", "comptes", "resultats", "actionnaires", "dirigeants", "produits", "marque", "site",
})
# Public mail hosts: their domain names no company.
_PUBLIC_MAIL_HOSTS = frozenset({
    "gmail", "googlemail", "google", "yahoo", "ymail", "rocketmail", "hotmail", "outlook", "live", "msn", "icloud",
    "me", "mac", "aol", "protonmail", "proton", "pm", "gmx", "mail", "email", "zoho", "yandex", "fastmail",
    "hey", "tutanota", "tuta", "orange", "wanadoo", "free", "sfr", "laposte", "bbox", "numericable", "neuf",
    "aliceadsl", "club-internet", "cegetel", "voila", "example", "test", "localhost", "invalid", "qq", "163",
    "126", "sina", "naver", "daum", "web", "t-online", "btinternet", "sky", "virginmedia", "comcast", "verizon",
    "att", "rediffmail", "mailinator", "icloud", "hushmail", "mail-tester", "skynet", "telenet", "bluewin",
    "sunrise", "libero", "virgilio", "seznam", "wp", "o2", "op", "mailbox", "posteo",
})
# Second-level labels of a public suffix ("co.uk", "com.au", "gouv.fr", "asso.fr") and sub-domain prefixes that
# name a mail server, not the company.
_PUBLIC_SECOND_LEVEL = frozenset({"co", "com", "org", "net", "ac", "gov", "gouv", "edu", "ne", "or", "asso",
                                  "nom", "ltd", "plc", "mil", "int", "go", "ed", "sch", "nhs", "police"})
_MAIL_SUBDOMAINS = frozenset({"mail", "email", "smtp", "mx", "www", "webmail", "intranet", "ext", "int",
                              "internal", "corp", "mailer", "noreply", "no-reply", "notifications", "news",
                              "newsletter", "support", "info", "secure", "my", "app", "apps", "go", "em", "e"})


class _LangWords:
    """Word predicates of one language for the refinement pass (French: the lists of this module; English: the
    LOCALE-EN bundle)."""

    def __init__(self, lang: str):
        self.lang = "en" if str(lang).lower().startswith("en") else "fr"
        self.lc = None
        self.b = None
        if self.lang == "en":
            from . import locale as lc

            self.lc = lc
            self.b = lc.bundle("en")

    @staticmethod
    def low(surface: str) -> str:
        return fold_word(surface.rstrip(".")).replace(".", "")

    def form(self, surface: str) -> Optional[str]:
        """"strong", "weak" or None for a legal form written as in the text ("Inc.", "SAS", "Company")."""
        if surface == "&" or not surface:
            return None
        if self.lang == "en":
            return self.lc.legal_form(surface, self.b)
        return _legal_form(_make_word(surface, 0, len(surface)))

    def dotted_form(self, surface: str) -> bool:
        """True when the dot of ``surface`` belongs to the form ("Inc.", "Co.", "S.A.")."""
        if not surface.endswith(".") or self.lang != "en":
            return False
        key = fold_word(surface).replace(".", "")
        return (key + ".") in self.b.legal_forms

    def head(self, surface: str) -> bool:
        """A head or unit noun that may end a company name ("Group", "Partners", "Immobilier")."""
        low = self.low(surface)
        if self.lang == "en":
            return low in self.b.org_suffixes or low in _EN_HEAD_EXTRA
        return low in _ORG_SUFFIXES

    def generic(self, surface: str) -> bool:
        """A generic company word: never the distinctive part of a name ("Group", "Services", "Conseil")."""
        low = self.low(surface)
        if self.lang == "en":
            return low in self.b.org_generic or low in self.b.org_suffixes or low in _EN_HEAD_EXTRA
        return low in ORG_GENERIC

    def excluded(self, surface: str) -> bool:
        """A word that is never a name word (stop word, month, greeting, title, legal term, role)."""
        low = self.low(surface)
        if self.lang == "en":
            b = self.b
            return (low in b.stopwords or low in b.months_days or low in b.greetings or low in b.titles
                    or low in b.legal_terms or low in b.roles)
        return _excluded(low) or low in _ROLE_WORDS

    def title(self, surface: str) -> bool:
        if self.lang == "en":
            return self.lc.is_title(surface, self.b)
        return is_title_word(surface)

    def org_word(self, surface: str) -> bool:
        if surface == "&":
            return True
        if self.lang == "en":
            return self.lc.is_name_word(surface, self.b, kind="ORG")
        return _org_word(_make_word(surface, 0, len(surface)), allow_article=False)

    def dictionary(self, low: str) -> bool:
        """A vocabulary, common, place, first-name, generic, legal or function word: not distinctive on its own."""
        if self.lang == "en":
            b = self.b
            return (low in b.vocab or low in b.common_words or low in b.places or low in b.all_first_names
                    or low in b.org_generic or low in b.org_suffixes or low in b.stopwords or low in b.legal_terms
                    or low in b.acronyms or low in b.roles or low in b.months_days)
        return _lexicon_word(low) or low in ORG_GENERIC or low in _lists()[4] or low in _acronym_stop()

    def trigger(self, low: str) -> bool:
        """A word before a company name that announces it ("at", "joined", "société", "groupe")."""
        if self.lang == "en":
            return low in self.b.org_triggers
        return low in ORG_TRIGGERS or low in ("chez",)


@lru_cache(maxsize=4)
def _lang_words(lang: str) -> _LangWords:
    return _LangWords(lang)


def _strip_dot(tok: str, lw: _LangWords) -> tuple[str, int]:
    """A trailing dot that is not part of the form is a sentence end: the word without it, and 1 when cut."""
    if tok.endswith(".") and not lw.dotted_form(tok):
        return tok[:-1], 1
    return tok, 0


def legal_form_after(text: str, end: int, lang: str = "fr") -> Optional[int]:
    """End of the legal form that follows a company name ending at ``end`` ("Acme Corp", "Acme, Inc.",
    "Acme Pty Ltd", "ACME SAS"), on the same line; None when no form follows (v0.3, span policy "cover")."""
    lw = _lang_words(lang)
    pos, out, n = end, None, 0
    while n < 2:
        m = _NEXT_WORD_RE.match(text, pos)
        if m is None or (n and "," in m.group("gap")):
            break
        tok = m.group("tok")
        if not lw.form(tok):
            break
        tok, cut = _strip_dot(tok, lw)
        if not lw.form(tok):
            break
        te = m.end("tok") - cut
        out, pos, n = te, te, n + 1
        if cut or tok.endswith("."):
            break
    return out


def legal_form_before(text: str, start: int, lang: str = "fr") -> Optional[int]:
    """Start of the legal form written right before a company name starting at ``start`` ("SAS Les Tilleuls",
    "SCI DU PARC"), on the same line; None when none (v0.3, span policy "cover")."""
    lw = _lang_words(lang)
    m = _PREV_WORD_RE.search(text, max(0, start - 24), start)
    if m is None:
        return None
    tok = m.group("tok")
    if tok == "&" or tok.endswith(".") and not lw.dotted_form(tok):
        return None
    return m.start("tok") if lw.form(tok) else None


def org_form_parts(surface: str, lang: str = "fr") -> tuple[str, str, str]:
    """``(leading form, core, trailing form)`` of a company surface, the separators included in the form parts:
    "Acme Corp" -> ("", "Acme", " Corp"), "Acme, Inc." -> ("", "Acme", ", Inc."), "SAS Les Tilleuls" ->
    ("SAS ", "Les Tilleuls", ""). The masker writes ``lead + code + tail`` under the span policy "cover", so the
    form stays readable while the span covers it; a surface made of a form only is returned as its own core."""
    lw = _lang_words(lang)
    words = [(m.group(), m.start(), m.end()) for m in _WORD_RE.finditer(surface)]
    if len(words) < 2:
        return "", surface, ""
    k0, k1 = 0, len(words)
    cut = 0
    while k1 - k0 > 1 and cut < 2 and lw.form(_strip_dot(words[k1 - 1][0], lw)[0]):
        k1 -= 1
        cut += 1
    if k1 - k0 > 1 and lw.form(_strip_dot(words[k0][0], lw)[0]):
        k0 += 1
    cs, ce = words[k0][1], words[k1 - 1][2]
    return surface[:cs], surface[cs:ce], surface[ce:]


def _interval_set(pairs: Iterable[tuple[int, int]]) -> IntervalSet:
    """An IntervalSet of possibly overlapping pairs (merged first: the set expects disjoint intervals)."""
    out = IntervalSet()
    merged: list[list[int]] = []
    for s, e in sorted(pairs):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    for s, e in merged:
        out.add(s, e)
    return out


def _free(text: str, s: int, e: int) -> bool:
    """``text[s:e]`` is a whole word or phrase: not glued to a letter or a digit (an English possessive after
    it is fine)."""
    if s > 0 and (text[s - 1].isalnum() or text[s - 1] in "-'’_@"):
        return False
    if e < len(text) and (text[e].isalnum() or text[e] in "-_") and not (text[e:e + 2] in ("'s", "’s")):
        return False
    return True


def _capitalised_surface(surface: str) -> bool:
    """Every significant word starts with a capital, or the whole surface is in capitals."""
    words = [w for w in _WORD_RE.findall(surface) if w != "&"]
    if not words:
        return False
    return all(w[0].isupper() or fold_word(w) in _CONNECT_WORDS for w in words)


def _extend_org(text: str, s: int, e: int, lw: _LangWords, get_protected, blocked: IntervalSet) -> int:
    """New end of a company span extended over the head words that follow it on the same line ("Acme" ->
    "Acme Commercial Banking", "Brightwater" -> "Brightwater Capital Partners"): capitalised company words up to a
    head noun, at most four words, never over a comma, a line break, a cell separator, a legal form, a protected
    term or another span (plan S4). ``get_protected`` gives the protected intervals when a head word is found."""
    pos, new_e, steps = e, e, 0
    while steps < 4:
        m = _NEXT_WORD_RE.match(text, pos)
        if m is None or "," in m.group("gap"):
            break
        tok, ts, te = m.group("tok"), m.start("tok"), m.end("tok")
        low = lw.low(tok)
        if tok == "&" or (low in _CONNECT_WORDS and not tok[0].isupper()):
            nxt = _NEXT_WORD_RE.match(text, te)
            if nxt is None or "," in nxt.group("gap") or not nxt.group("tok")[0].isupper():
                break
            pos, steps = te, steps + 1
            continue
        if not tok[0].isupper():
            break
        tok, cut = _strip_dot(tok, lw)
        te -= cut
        if lw.excluded(tok) or lw.form(tok) or lw.title(tok) or not lw.org_word(tok):
            break
        if _any_first_name(low):
            # "Acme Corp Kate Ormsby" (a table row without its cell separators): a person starts here
            break
        if blocked.overlaps(ts, te):
            break
        if lw.head(tok):
            if get_protected().overlaps(e, te):
                break
            new_e = te
        pos, steps = te, steps + 1
        if cut:
            break
    return new_e


def _org_cue(text: str, s: int, e: int, lw: _LangWords) -> bool:
    """A context cue that makes a capitalised dictionary word a company: a legal form, a head or generic word
    right after it; a trigger word right before it ("at", "joined", "société"); a possessive before a business
    noun ("Acme's revenue"); the word written in capitals."""
    surface = text[s:e]
    letters = [c for c in surface if c.isalpha()]
    if len(letters) >= 3 and all(c.isupper() for c in letters):
        return True
    if legal_form_after(text, e, lw.lang) is not None:
        return True
    m = _NEXT_WORD_RE.match(text, e)
    if m is not None and "," not in m.group("gap"):
        tok = _strip_dot(m.group("tok"), lw)[0]
        if tok[0].isupper() and (lw.head(tok) or lw.generic(tok)):
            return True
    if text[e:e + 2] in ("'s", "’s"):
        m = _NEXT_WORD_RE.match(text, e + 2)
        if m is not None and lw.low(m.group("tok")) in _CUE_NOUNS:
            return True
    m = _PREV_WORD_RE.search(text, max(0, s - 24), s)
    if m is not None:
        low = lw.low(m.group("tok"))
        if lw.trigger(low):
            return True
    return False


def _person_title_before(text: str, s: int, lw: _LangWords) -> bool:
    """A civility or function title right before ``s`` ("M. Dupont", "Dr Pellworth"): the word is a person."""
    if title_civility(text, s):
        return True
    m = _PREV_WORD_RE.search(text, max(0, s - 24), s)
    return m is not None and lw.title(m.group("tok"))


def _domain_label(domain: str, min_len: int = 3) -> str:
    """The label of an e-mail domain that names its owner ("mail.acme-corp.co.uk" -> "acme-corp"), or "" for a
    public mail host or a label too short to name anything (under ``min_len`` characters; v0.4 T4 reads the
    two-letter labels of the ampersand initials with ``min_len=2``)."""
    labels = [x for x in domain.lower().strip(".").split(".") if x]
    if len(labels) == 1:
        # a bare label, as the dossier directory keeps it ("northvale" for northvale.example): the label itself
        label = labels[0]
        if label in _PUBLIC_MAIL_HOSTS or label in _PUBLIC_SECOND_LEVEL or len(label) < min_len:
            return ""
        return label
    if len(labels) < 2:
        return ""
    tld, rest = labels[-1], labels[:-1]
    if len(rest) >= 2 and rest[-1] in _PUBLIC_SECOND_LEVEL and len(tld) <= 3:
        rest = rest[:-1]
    label = rest[-1]
    if label in _PUBLIC_MAIL_HOSTS or len(label) < min_len:
        return ""
    return label


def org_candidates_from_domain(domain: str, lang: str = "fr") -> list[tuple[tuple[str, ...], bool]]:
    """Token tuples (folded) under which the company that owns the e-mail domain ``domain`` may be written, each
    with a flag: True when the tuple is distinctive (a word that no dictionary holds), False when it is a
    vocabulary, place or first-name word and needs a context cue in the text. "acme-corp.com" gives
    ("acme", "corp"), ("acmecorp",) and ("acme",); "northvale.com" gives ("northvale",); a public mail host gives
    nothing. A bare domain label ("acme-corp", as ``directory.Directory.domains`` keeps it) is accepted as the
    label itself. A helper for the dossier directory (plan S3): the directory owner feeds the domains (or the
    labels) of the export to :func:`detect_names` through ``domains`` (plan S4)."""
    lw = _lang_words(lang)
    label = _domain_label(domain)
    if not label:
        return []
    parts = [p for p in re.split(r"[-_.]+|(?<=\D)(?=\d)|(?<=\d)(?=\D)", label) if p]
    if len(parts) == 1:
        word = parts[0]
        # "northvalelabs": a generic word glued at the end of the label
        for suffix in sorted(_generic_tails(lw.lang), key=len, reverse=True):
            if word.endswith(suffix) and len(word) - len(suffix) >= 4 and not lw.dictionary(word):
                parts = [word[:-len(suffix)], suffix]
                break
    if not parts:
        return []
    out: list[tuple[tuple[str, ...], bool]] = []
    seen: set[tuple[str, ...]] = set()

    def add(toks: tuple[str, ...], distinctive: bool) -> None:
        if toks and toks not in seen and all(toks):
            seen.add(toks)
            out.append((toks, distinctive))

    sig = [p for p in parts if not p.isdigit()]
    distinctive = any(len(p) >= 4 and p.isalpha() and not lw.dictionary(p) for p in sig)
    if len(parts) >= 2:
        add(tuple(parts), distinctive)
        add(("".join(parts),), distinctive)
        core = [p for p in parts if not (lw.generic(p) or lw.form(p.upper()) or p.isdigit())]
        if len(core) == 1 and len(core[0]) >= 4 and core[0].isalpha():
            # the head word alone ("Acme" of acme-corp.com) is weaker evidence: it needs a company cue in the
            # text, so a person named like the company ("Pellstone" of pellstone-avocats.fr) is not taken
            add((core[0],), False)
    else:
        word = parts[0]
        if len(word) >= 4 and word.isalpha():
            add((word,), not lw.dictionary(word))
        elif len(word) >= 3 and word.isalnum():
            add((word,), False)
    return out


@lru_cache(maxsize=4)
def _generic_tails(lang: str) -> frozenset:
    lw = _lang_words(lang)
    if lw.lang == "en":
        return frozenset(x for x in (lw.b.org_suffixes | _EN_HEAD_EXTRA) if len(x) >= 3)
    return frozenset(x for x in ORG_GENERIC if len(x) >= 4)


def _candidate_regex(cands: list[tuple[tuple[str, ...], bool]]) -> "re.Pattern[str]":
    """One case-insensitive pattern for the candidate token tuples, whole words, tokens joined by a short gap."""
    alts = []
    for toks, _distinctive in sorted(cands, key=lambda c: -len("".join(c[0]))):
        alts.append(r"[ .'’\-&]{0,3}".join(re.escape(t) for t in toks))
    return re.compile(r"(?<![\w\-])(?:" + "|".join(alts) + r")(?![\w\-])", re.IGNORECASE)


def domain_org_spans(text: str, domains: Iterable[str], lang: str = "fr", *, protected=None,
                     blocked: Optional[IntervalSet] = None) -> list[RuleSpan]:
    """Company spans of ``text`` named by one of the e-mail ``domains``: every capitalised mention of a
    distinctive domain name, and the mentions of a dictionary-word domain name that carry a company cue
    (:func:`_org_cue`). Never after a person title, never inside ``protected`` (an IntervalSet, or a function
    that returns one when first needed) or ``blocked`` intervals, never glued to another word. Rule ``domain``,
    priority 5 (an equal person dictionary match yields to it)."""
    lw = _lang_words(lang)
    cands: dict[tuple[str, ...], bool] = {}
    for domain in dict.fromkeys(domains):
        for toks, distinctive in org_candidates_from_domain(domain, lang):
            cands[toks] = cands.get(toks, False) or distinctive
    if not cands:
        return []
    rx = _candidate_regex(list(cands.items()))
    out: list[RuleSpan] = []
    for m in rx.finditer(text):
        s, e = m.start(), m.end()
        key = tuple(fold_word(t) for t in re.split(r"[ .'’\-&]+", m.group()) if t)
        distinctive = cands.get(key)
        if distinctive is None:
            distinctive = cands.get(("".join(key),), False)
        if not _free(text, s, e) or not _capitalised_surface(text[s:e]):
            continue
        if blocked is not None and blocked.overlaps(s, e):
            continue
        if protected is not None:
            iv = protected() if callable(protected) else protected
            if iv.overlaps(s, e):
                continue
        if _person_title_before(text, s, lw):
            continue
        if not distinctive and not _org_cue(text, s, e, lw):
            continue
        out.append(RuleSpan(s, e, "ORG", 5, "domain"))
    return out


# v0.4 (T4, OM4): the other written forms of a company named by an e-mail domain (English rules only).
# A word as the squeeze reads it: letters and digits, an apostrophe inside it ("O'Varn"); a final possessive "'s"
# stays outside (the squeeze tries the label with and without it: "Hallow's" for hallows, "Varn's" for varn).
_DF_WORD_RE = re.compile(r"[^\W_](?:[^\W_]|['’](?![sS](?![^\W_]))(?=[^\W_]))*")
# What joins two words of one written name: one or two spaces, a hyphen, an ampersand.
_DF_GAP_RE = re.compile(r"(?:[   ]{1,2}|-|[   ]{0,2}&[   ]{0,2})$")
# A possessive inside a written name, between two of its words.
_DF_POSS_GAP_RE = re.compile(r"['’]s[   ]{1,2}$")
# A spelled legal form at the end of a name and the short form a domain label uses for it ("X Company" for xco).
_DF_SPELLED = {"company": "co", "corporation": "corp", "incorporated": "inc", "limited": "ltd"}
# Ampersand initials of a two-letter to four-letter label ("V&K" for vk.example).
_DF_AMP_RE = re.compile(r"(?<![\w&])([A-Z]{1,2})[   ]?&[   ]?([A-Z]{1,2})(?![\w&])")
# Business abbreviations written with an ampersand: never a company ("R&D", "M&A", "P&L", "Q&A").
_DF_AMP_STOP = frozenset({"rd", "ma", "pl", "qa", "bb", "fb", "tc", "te", "ar", "hs", "om", "do", "eo", "ga",
                          "ld", "ci", "bw", "pc", "ae", "ep", "sa", "ps", "cs"})
# A short prefix joined by a hyphen before a company name ("non-Acme accounts", "ex-Acme staff").
_DF_PREFIX_RE = re.compile(r"(?<![\w\-])(?:non|ex|pre|post|anti)-$", re.IGNORECASE)


def _df_free(text: str, s: int, e: int) -> bool:
    """``text[s:e]`` stands alone: not inside an address, a host, a path, a handle or a file name, not glued to a
    word (a short prefix with a hyphen before it and an English possessive after it are fine)."""
    if s > 0:
        c = text[s - 1]
        if c == "-":
            if not _DF_PREFIX_RE.search(text, max(0, s - 6), s):
                return False
        elif c.isalnum() or c in "@._/\\#'’:+=":
            return False
    if e < len(text):
        c = text[e]
        if text[e:e + 2] in ("'s", "’s"):
            return True
        if c.isalnum() or c in "@_-/\\":
            return False
        if c == "." and e + 1 < len(text) and text[e + 1].isalnum():
            return False  # a host or a file name: "acme.example", "acme.pdf"
    return True


def _df_labels(domains: Iterable[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Squeezed label -> label for the labels of three or more characters, and the same for the two- to
    four-letter labels that ampersand initials may write ("vk")."""
    longs: dict[str, str] = {}
    shorts: dict[str, str] = {}
    for domain in dict.fromkeys(domains):
        label = _domain_label(domain, min_len=2)
        if not label:
            continue
        sq = re.sub(r"[^a-z0-9]", "", fold_word(label))
        if not sq:
            continue
        if len(sq) >= 3:
            longs.setdefault(sq, label)
        if 2 <= len(sq) <= 4 and sq.isalpha():
            shorts.setdefault(sq, label)
    return longs, shorts


def _df_case(words: list[str]) -> str:
    """"cap" when every word of a name starts with a capital (or is a camel-case word, "nVolt"), "lower" when every
    word is in lower case, else "mixed"."""
    sig = [w for w in words if fold_word(w) not in _CONNECT_WORDS]
    if not sig:
        return "mixed"
    if all(w[0].isupper() or (w[0].islower() and any(c.isupper() for c in w[1:])) for w in sig):
        return "cap"
    if all(w.islower() for w in words):
        return "lower"
    return "mixed"


def domain_form_spans(text: str, domains: Iterable[str], lang: str = "en", *, protected=None,
                      others: Iterable[RuleSpan] = (), taken: Iterable[RuleSpan] = ()) -> list[RuleSpan]:
    """v0.4 (T4, OM4): company spans of ``text`` written in a form of an e-mail domain label of the text or of the
    dossier that the v0.3 rule (:func:`domain_org_spans`) does not read. English rules only (French: nothing).

    - a run of two to four capitalised words whose letters, joined, spell the label ("Grelmont Quay" for
      grelmontquay.example), also with a spelled legal form for its short form ("Ostrel Company" for ostrelco);
    - initials joined by an ampersand that spell a two- to four-letter label ("V&K" for vk.example);
    - one camel-case word that is the label ("tZorvel" for tzorvel.example), in any text;
    - in chat-like text (:func:`locale.casual_text`), the label or its squeezed run in lower case ("grelmontquay",
      "grelmont quay"), when the label is no dictionary word, no first name and holds four or more letters (six
      or more for a run of words); outside chat only with a legal form right after it ("tzorvel LLC").

    Never inside an address, a host, a path, a handle or a file name, never after a person title, never inside
    ``protected``, never across a span of another kind (``others``) or a company span already found (``taken``)
    that sticks out of it. Rule ``domain_form``, priority 5."""
    lw = _lang_words(lang)
    if lw.lang != "en":
        return []
    longs, shorts = _df_labels(domains)
    if not longs and not shorts:
        return []
    # spans of the rules already there: one that sticks out of a new span blocks it, one inside it does not (a
    # person reading of the words of "Grelmont Quay", a company span over one of them)
    blockers = [(sp.start, sp.end) for sp in others] + [(sp.start, sp.end) for sp in taken]
    out: list[RuleSpan] = []
    got: list[tuple[int, int]] = []
    chat: list[bool] = []

    def casual() -> bool:
        if not chat:
            try:
                chat.append(bool(lw.lc.casual_text(text)))
            except Exception:  # noqa: BLE001 - never fail a masking on a heuristic
                chat.append(False)
        return chat[0]

    def ok(s: int, e: int) -> bool:
        if not _df_free(text, s, e) or _person_title_before(text, s, lw):
            return False
        if any(a < e and s < b and not (s <= a and b <= e) for a, b in blockers):
            return False
        if any(a < e and s < b for a, b in got):
            return False
        if protected is not None:
            iv = protected() if callable(protected) else protected
            if iv.overlaps(s, e):
                return False
        return True

    def add(s: int, e: int) -> None:
        got.append((s, e))
        out.append(RuleSpan(s, e, "ORG", 5, "domain_form"))

    if longs:
        prefixes = {sq[:k] for sq in longs for k in range(1, len(sq) + 1)}
        words = [(m.group(), m.start(), m.end()) for m in _DF_WORD_RE.finditer(text)
                 if not (m.group() in ("s", "S") and m.start() > 0 and text[m.start() - 1] in "'’")]
        n = len(words)
        cands: list[tuple[int, int, int, str, str]] = []  # (-length, start, end, case, label)
        for i in range(n):
            sq = ""
            for j in range(i, min(n, i + 4)):
                w, ws, we = words[j]
                if j > i:
                    gap = text[words[j - 1][2]:ws]
                    if _DF_POSS_GAP_RE.match(gap):
                        sq += "s"  # "Varnell's Depot" for varnellsdepot.example
                    elif not _DF_GAP_RE.match(gap):
                        break
                wsq = re.sub(r"[^a-z0-9]", "", fold_word(w))
                sq += wsq
                part = [x[0] for x in words[i:j + 1]]
                keys = [sq]
                spelled = _DF_SPELLED.get(wsq)
                if j > i and spelled:
                    keys.append(sq[:len(sq) - len(wsq)] + spelled)
                poss = text[we:we + 2] in ("'s", "’s") and not text[we + 2:we + 3].isalnum()
                for key, end in [(k, we) for k in keys] + ([(sq + "s", we + 2)] if poss else []):
                    label = longs.get(key)
                    if label is None:
                        continue
                    if fold_word(part[0]) in _CONNECT_WORDS or fold_word(part[-1]) in _CONNECT_WORDS:
                        continue
                    cands.append((-(end - words[i][1]), words[i][1], end, _df_case(part), key))
                if sq not in prefixes:
                    break
        casual_label_ok: dict[str, bool] = {}
        for _neg, s, e, case, key in sorted(cands):
            multi = bool(re.search(r"[\s&\-]", text[s:e]))
            if multi:
                if len(key) < 5:
                    continue
                if case != "cap" and not (len(key) >= 6 and casual()):
                    continue
            else:
                surface = text[s:e]
                glued_s = surface.endswith(("'s", "’s"))  # "Hallow's" for hallows.example
                if case == "cap" and surface[0].isupper() and not glued_s:
                    continue  # the capitalised label itself: the v0.3 rule's case
                if not (len(key) >= 4 and key.isalpha()):
                    continue
                if case != "cap" or glued_s:
                    if key not in casual_label_ok:
                        casual_label_ok[key] = not lw.dictionary(key) and not _any_first_name(key)
                    if not casual_label_ok[key]:
                        continue
                    if case != "cap" and not (casual() or legal_form_after(text, e, "en") is not None):
                        continue
            if ok(s, e):
                add(s, e)
    if shorts and "&" in text:
        for m in _DF_AMP_RE.finditer(text):
            key = (m.group(1) + m.group(2)).lower()
            if key in shorts and key not in _DF_AMP_STOP and ok(m.start(), m.end()):
                add(m.start(), m.end())
    return out


_DECLARED_RE = re.compile(
    r"[   ]*\([  ]*"
    r"(?:(?i:hereinafter|hereafter|henceforth|ci-apr[èe]s)"
    r"(?:[  ]+(?i:referred[  ]+to[  ]+as|called|designated|known[  ]+as|d[ée]nomm[ée]e?s?"
    r"|d[ée]sign[ée]e?s?|dite?s?))?[  ]*[:,]?[  ]*)?"
    r"(?:(?i:the|la|le|les)[  ]+|(?i:l)['’])?(?P<q>[«\"“‘']?)[  ]?"
    r"(?P<a>[^\W\d_][\w&.\-]{1,14})[  ]?[»\"”’']?[  ]*\)"
)


def _org_words(surface: str, lw: _LangWords) -> list[str]:
    """Significant words of a company surface, folded: no connector, no legal form."""
    out = []
    for w in _WORD_RE.findall(surface):
        if w == "&":
            continue
        low = lw.low(w)
        if low in _CONNECT_WORDS or lw.form(w) or not low:
            continue
        out.append(low)
    return out


def _abbreviates(alias: str, quoted: bool, surface: str, lw: _LangWords) -> bool:
    """True when ``alias``, declared in brackets after the company ``surface``, abbreviates it: the initials of
    its words (with or without the generic ones), a capital prefix of its first word ("ACME" for "Acme
    Corporation"), or a quoted word of the name."""
    letters = "".join(c for c in alias if c.isalpha())
    low = fold_word(letters)
    if not low or lw.form(alias) or lw.excluded(alias) or lw.title(alias):
        return False
    words = _org_words(surface, lw)
    if not words:
        return False
    upper = letters.isupper() and 2 <= len(letters) <= 8
    if upper:
        if low in _acronym_stop() or low in _instruments():
            return False
        if low == "".join(w[0] for w in words) and len(words) >= 2:
            return True
        sig = [w for w in words if not lw.generic(w)]
        if len(sig) >= 2 and low == "".join(w[0] for w in sig):
            return True
        if len(low) >= 3 and words[0].startswith(low) and not lw.dictionary(words[0]):
            return True
        return False
    if quoted and alias[0].isupper() and len(low) >= 4 and low in words and not lw.dictionary(low):
        return True
    return False


def _declared_aliases(text: str, s: int, e: int, lw: _LangWords, get_protected, blocked: IntervalSet,
                      out: list[RuleSpan]) -> None:
    """"Acme Corporation (ACME)", 'Brightwater Capital Partners ("Brightwater")', "X (hereinafter "Y")": the
    alias the text declares after a company is that company here and wherever the text repeats it (plan S4)."""
    m = _DECLARED_RE.match(text, e)
    if m is None:
        return
    alias = m.group("a").rstrip(".")
    a_s, a_e = m.start("a"), m.start("a") + len(alias)
    if not _abbreviates(alias, bool(m.group("q")), text[s:e], lw):
        return
    protected = get_protected()
    if protected.overlaps(a_s, a_e) or blocked.overlaps(a_s, a_e):
        return
    out.append(RuleSpan(a_s, a_e, "ORG", 5, "acronym"))
    rx = re.compile(r"(?<![\w\-])" + re.escape(alias) + r"(?![\w\-])")
    for hit in rx.finditer(text):
        hs, he = hit.start(), hit.end()
        if (hs, he) == (a_s, a_e) or protected.overlaps(hs, he) or blocked.overlaps(hs, he):
            continue
        if _person_title_before(text, hs, lw):
            continue
        out.append(RuleSpan(hs, he, "ORG", 4, "acronym"))


def _same_surface(text: str, spans: list[RuleSpan], lw: _LangWords, get_protected,
                  blocked: IntervalSet) -> list[RuleSpan]:
    """The other mentions of a company name the rules found once: the same words in the same case, as a whole
    phrase, for a name of two or more significant words, or of one distinctive word found with strong evidence
    (plan S4, same-surface propagation). One scan of the words of the text, indexed by the first word."""
    by_first: dict[str, list[tuple[str, int]]] = {}
    registered: set[str] = set()
    for sp in spans:
        if sp.kind != "ORG":
            continue
        surface = text[sp.start:sp.end]
        if surface in registered or "\n" in surface:
            continue
        words = _org_words(surface, lw)
        if len(words) >= 2:
            if all(lw.dictionary(w) for w in words) and sp.rule not in STRONG_ORG_RULES:
                continue
        elif len(words) == 1:
            if sp.rule not in STRONG_ORG_RULES or len(words[0]) < 4 or lw.dictionary(words[0]):
                continue
        else:
            continue
        first = _WORD_RE.match(surface)
        if first is None:
            continue
        registered.add(surface)
        by_first.setdefault(first.group(), []).append((surface, min(sp.priority, 4)))
    if not by_first:
        return []
    out: list[RuleSpan] = []
    covered = _interval_set((sp.start, sp.end) for sp in spans)
    seen: set[tuple[int, int]] = set()
    for m in _WORD_RE.finditer(text):
        cands = by_first.get(m.group())
        if not cands:
            continue
        s = m.start()
        for surface, prio in cands:
            e = s + len(surface)
            if (s, e) in seen or not text.startswith(surface, s) or not _free(text, s, e):
                continue
            seen.add((s, e))
            if covered.overlaps(s, e) or blocked.overlaps(s, e) or get_protected().overlaps(s, e):
                continue
            if _person_title_before(text, s, lw):
                continue
            out.append(RuleSpan(s, e, "ORG", prio, "same_surface"))
    return out


def refine_rule_spans(text: str, spans: Iterable[RuleSpan], lang: str = "fr", *,
                      protected: Optional[IntervalSet] = None, domains: Iterable[str] = (),
                      folded: Optional[Folded] = None, ftoks: Optional[list] = None) -> list[RuleSpan]:
    """The rule spans of ``text`` refined for both languages (v0.3, plan S1 and S4):

    1. span policy "cover" or "mask": a company span takes the legal form written next to it;
    2. companies named by an e-mail domain of the text (and of ``domains``, the dossier's): rule ``domain``;
    3. a company span extends over the head words that follow it ("Acme Commercial Banking");
    4. an alias or acronym the text declares after a company ("Acme Corporation (ACME)") is that company, here and
       wherever the text repeats it: rule ``acronym``;
    5. the other mentions of a multi-word company name are company spans too: rule ``same_surface``.

    Nothing is extended or added over a line break, a cell separator, a comma, a sentence end, a protected term
    (court, law, institution: ``protected``, computed here from ``folded`` and ``ftoks`` or from the text when a
    candidate needs it) or a span of another kind. Spans of other kinds are returned unchanged; duplicates keep
    the highest priority. The pass works on the original text: no folding, so it costs little on a plain text.
    """
    out = list(spans)
    orgs = [sp for sp in out if sp.kind == "ORG"]
    doms = list(domains)
    if "@" in text:
        from . import detectors as det  # the e-mail addresses of the text name their companies

        doms.extend(det.email_domains(text))
    if not orgs and not doms:
        return out
    lw = _lang_words(lang)
    state: dict = {"protected": protected, "folded": folded, "ftoks": ftoks}

    def get_protected() -> IntervalSet:
        if state["protected"] is None:
            if state["folded"] is None:
                state["folded"] = fold(text)
            if state["ftoks"] is None:
                state["ftoks"] = tokens(state["folded"].text)
            if lw.lang == "en":
                state["protected"] = lw.lc.protected_intervals_en(text, state["folded"], state["ftoks"])
            else:
                state["protected"] = protected_intervals(text, state["folded"], state["ftoks"])
        return state["protected"]

    blocked = _interval_set((sp.start, sp.end) for sp in out if sp.kind != "ORG")
    policy = legal_form_policy()
    # 1. the legal form joins the span
    if orgs and policy != "clear":
        for k, sp in enumerate(out):
            if sp.kind != "ORG":
                continue
            s, e = sp.start, sp.end
            e2 = legal_form_after(text, e, lw.lang)
            if e2 is not None and not blocked.overlaps(e, e2):
                e = e2
            s2 = legal_form_before(text, s, lw.lang)
            if s2 is not None and not blocked.overlaps(s2, s):
                s = s2
            if (s, e) != (sp.start, sp.end):
                out[k] = RuleSpan(s, e, "ORG", sp.priority, sp.rule)
    # 2. companies named by an e-mail domain
    if doms:
        found = domain_org_spans(text, doms, lw.lang, protected=get_protected, blocked=blocked)
        if lw.lang == "en":
            # v0.4 (T4): the squeezed, ampersand, camel-case and chat forms of the same companies
            found.extend(domain_form_spans(text, doms, lw.lang, protected=get_protected,
                                           others=[sp for sp in out if sp.kind != "ORG"],
                                           taken=[sp for sp in out if sp.kind == "ORG"] + found))
        if found:
            if policy != "clear":
                for k, sp in enumerate(found):
                    e2 = legal_form_after(text, sp.end, lw.lang)
                    if e2 is not None and not blocked.overlaps(sp.end, e2):
                        found[k] = RuleSpan(sp.start, e2, "ORG", sp.priority, sp.rule)
            out.extend(found)
    orgs_k = [k for k, sp in enumerate(out) if sp.kind == "ORG"]
    if not orgs_k:
        return out
    # 3. span edges over head words
    for k in orgs_k:
        sp = out[k]
        m = _NEXT_WORD_RE.match(text, sp.end)
        if m is None or "," in m.group("gap") or not m.group("tok")[0].isupper():
            continue
        e2 = _extend_org(text, sp.start, sp.end, lw, get_protected, blocked)
        if e2 > sp.end:
            out[k] = RuleSpan(sp.start, e2, "ORG", sp.priority, sp.rule)
    # 4. declared aliases and acronyms
    added: list[RuleSpan] = []
    if "(" in text:
        for k in orgs_k:
            sp = out[k]
            # the declaration may follow a legal form the span leaves in clear: "X Inc (XI)", "X SAS (ci-après « X »)"
            ends = [sp.end]
            e_form = legal_form_after(text, sp.end, lw.lang)
            if e_form is not None:
                ends.append(e_form)
            for e in ends:
                if text[e:e + 3].lstrip(" \u00a0\u202f").startswith("("):
                    _declared_aliases(text, sp.start, e, lw, get_protected, blocked, added)
    out.extend(added)
    # 5. the other mentions of a multi-word company name
    out.extend(_same_surface(text, out, lw, get_protected, blocked))
    # duplicates: one span per (start, end, kind), the highest priority
    best: dict[tuple[int, int, str], RuleSpan] = {}
    for sp in out:
        key = (sp.start, sp.end, sp.kind)
        if key not in best or sp.priority > best[key].priority:
            best[key] = sp
    return sorted(best.values(), key=lambda x: (x.start, x.end, x.kind))


# ---------------------------------------------------------------------------
# Whitelist protection
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _whitelist_matcher() -> PhraseMatcher:
    pm = PhraseMatcher(gap=r"[ .'\-]{0,3}")
    for phrase in lx.whitelist_phrases():
        toks = phrase_tokens(phrase)
        if toks:
            pm.add(toks, phrase)
    return pm


_COURT_CITY_RE = re.compile(
    r"\b(?:tribunal judiciaire|tribunal de grande instance|tribunal d'instance|tribunal de proximite|"
    r"tribunal de commerce|tribunal des activites economiques|tribunal administratif|tribunal correctionnel|"
    r"tribunal de police|tribunal pour enfants|tribunal paritaire des baux ruraux|cour d'appel|"
    r"cour administrative d'appel|conseil de prud'?hommes|conseil des prud'?hommes|cour d'assises|"
    r"cour criminelle departementale|barreau|ordre des avocats(?: du barreau)?|prefecture|sous-prefecture|"
    r"mairie|ville|commune|chambre des notaires|chambre de commerce et d'industrie|"
    r"chambre des metiers(?: et de l'artisanat)?|universite|academie|rectorat|tj|tgi|cph|caa|ta|tc|ca|"
    r"greffe du tribunal(?: de commerce| judiciaire)?|rcs|registre du commerce et des societes|"
    r"juge de l'execution|juge aux affaires familiales|pole social du tribunal judiciaire)"
    r"\s+(?:de la |de l'|de |d'|du |des )?"  # the particle is absent in headers: "TRIBUNAL JUDICIAIRE SAINT-DENIS"
)
_CITY_RE = re.compile(r"(?:(?:La|Le|Les|LA|LE|LES|L['’])\s?)?[A-ZÀ-ÖØ-ÞŒ][\w'’\-]*")
# A second segment of a court seat: "de la Réunion", "de la Guadeloupe", "de Polynésie française" (v0.2, integrator).
_CITY_SUFFIX_RE = re.compile(r"[ \t\u00a0]+(?:de la |de l['’]|de |du |des |DE LA |DE L['’]|DE |DU |DES )"
                             r"(?:(?:La|Le|Les|LA|LE|LES)[ \t\u00a0])?[A-ZÀ-ÖØ-ÞŒ][\w'’\-]*(?:[ \t\u00a0][Ff]ran[çc]aise)?")
_ARTICLE_REF_RE = re.compile(
    r"(?i)\b(?:articles?|art\.)\s+(?:(?-i:[LRDA])\.?\s?)?\d+(?:[\-.]\d+)*"
    r"(?:\s*(?:,|et|à|&|ou)\s*(?:(?-i:[LRDA])\.?\s?)?\d+(?:[\-.]\d+)*)*(?:\s+(?:bis|ter|quater))?"
    r"|(?<![\w\-])(?-i:[LRD])\.?\s?\d{3,4}(?:-\d+){1,2}(?![\w\-])"
)
_LAW_REF_RE = re.compile(
    r"(?i)\b(?:loi|ordonnance|d[ée]cret|r[èe]glement|directive|d[ée]lib[ée]ration|arr[êe]t[ée])"
    r"\s+(?:\((?:UE|CE|CEE)\)\s+)?(?:n[°o]\.?\s*)?\d{2,4}[-/]\d{1,6}(?:\s+du\s+\d{1,2}\s+\w+\s+\d{4})?"
)


# Leading cases known by a party name ("arrêt Jacques Vabre", "jurisprudence Perruche"): public case law, kept in
# clear only next to a case-law word, so a client with one of these names is still masked elsewhere.
_CASE_LAW_NAMES = ("Société des cafés Jacques Vabre", "Jacques Vabre", "Perruche", "Blanco", "Nicolo", "Manoukian",
                   "Chronopost", "Faurecia", "Desmares", "Jand'heur", "Costedoat", "Teffaine", "Lamoricière",
                   "Sarran", "Danthony", "Ternon", "Bac d'Eloka")
_CASE_LAW_RE = re.compile(r"(?<![\w\-])(?:" + "|".join(re.escape(n).replace("'", "['’]") for n in _CASE_LAW_NAMES)
                          + r")(?![\w\-])")
_CASE_LAW_CTX = re.compile(r"(?i)\b(?:arr[êe]ts?|jurisprudences?|d[ée]cisions?|cass\.?|cassation|conseil\s+d['’][ée]tat"
                           r"|cjue|cedh|tribunal\s+des\s+conflits|solution)\b")


def protected_intervals(text: str, folded: Folded, ftoks: list[tuple[str, int, int]]) -> IntervalSet:
    """Original-text intervals that rules and NER must never mask."""
    out = IntervalSet()
    spans: list[tuple[int, int]] = []
    for i, j, _ in _whitelist_matcher().find_all(folded.text, ftoks):
        spans.append(folded.span(ftoks[i][1], ftoks[j - 1][2]))
    for m in _COURT_CITY_RE.finditer(folded.text):
        s, e = folded.span(m.start(), m.end())
        c = _CITY_RE.match(text, e)
        if c:
            e = c.end()
            sfx = _CITY_SUFFIX_RE.match(text, e)
            if sfx:
                e = sfx.end()
        elif not m.group().rstrip().endswith(("de", "d'", "du", "des", "la", "l'")):
            continue  # no particle and no capitalised seat after the court word: nothing to protect
        spans.append((s, e))
    for rx in (_ARTICLE_REF_RE, _LAW_REF_RE):
        for m in rx.finditer(text):
            spans.append((m.start(), m.end()))
    if _CASE_LAW_CTX.search(text):
        for m in _CASE_LAW_RE.finditer(text):
            window = text[max(0, m.start() - 100):m.end() + 60]
            if _CASE_LAW_CTX.search(window):
                spans.append((m.start(), m.end()))
    # Merge, then add (IntervalSet expects disjoint intervals).
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
# Splitting, norms and variants of known names
# ---------------------------------------------------------------------------


def _name_words(value: str) -> list[_W]:
    return [w for w in _words(value) if not w.digit]


def _strip_titles(words: list[_W]) -> list[_W]:
    while words and (_title(words[0]) or words[0].low in _TITLE_FULL):
        words = words[1:]
    return words


def split_person(value: str) -> tuple[list[str], list[str]]:
    """Split a person's name into (first names, surname words); see :func:`_split_person`. A quoted or bracketed
    nickname inside the name belongs to neither list (:func:`person_nicknames`), so 'Christopher "Chris" Pellworth'
    and "Christopher Pellworth" split, and key, the same way (v0.3)."""
    firsts, _nicks, surname = _split_person(value)
    return list(firsts), list(surname)


def person_nicknames(value: str) -> list[str]:
    """The nicknames written between quotes or brackets inside a person's name: 'Christopher "Chris" Pellworth' ->
    ["Chris"], "Raymond (Ray) Ashbury" -> ["Ray"]. Empty when the name has none (v0.3, S1)."""
    return list(_split_person(value)[1])


def _is_particle_text(word: str) -> bool:
    low = fold_word(word).rstrip("'").replace("'", "")
    return low in PARTICLES or low in ("les", "l")


@lru_cache(maxsize=32768)
def _split_person(value: str) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """(first names, nicknames, surname words) of a person value, cached."""
    firsts, nicks, surname = _split_person_impl(value)
    if firsts and all(_is_particle_text(f) for f in firsts):
        # "Le Goff", "d'Orgeval", "DE LA ROCHÈRE": a particle is never a first name
        surname = list(firsts) + list(surname)
        firsts = []
    return tuple(firsts), tuple(nicks), tuple(surname)


# Quote and bracket pairs that mark a nickname inside a name: 'Christopher "Chris" Pellworth', "Raymond (Ray) Ashbury",
# "Margaret 'Peggy' Hale", «Jean "Jo" Martin». An apostrophe pair must not be an elision ("d'Aurore").
_NICK_PAIRS = {'"': '"', "“": "”", "”": "”", "„": "“", "«": "»", "(": ")", "[": "]", "'": "'’", "‘": "’", "’": "’"}


def _quoted_word(value: str, w: _W) -> bool:
    """True when the word ``w`` of ``value`` is written between a pair of quotes or brackets."""
    s, e = w.start, w.end
    if s == 0 or e >= len(value):
        return False
    close = _NICK_PAIRS.get(value[s - 1])
    if close is None or value[e] not in close:
        return False
    if value[s - 1] in "'‘’" and s >= 2 and value[s - 2].isalpha():
        return False  # "d'Aurore": an elision, not a quote
    return True


def _nick_words(value: str, words: list[_W]) -> list[_W]:
    """The words of a person value written between quotes or brackets that are nicknames: a quoted word
    before the surname ('Christopher "Chris" Pellworth', "Raymond (Ray) Ashbury"), or a quoted last word after a
    full name ('Josephine Pellstone ("Josie")'). A quoted last word after one plain word is the surname in
    brackets ("Sam (Pellstone)"), not a nickname; a quoted word alone is a name."""
    quoted = [w for w in words if _quoted_word(value, w)]
    if not quoted or len(words) <= len(quoted):
        return []
    plain = [w for w in words if not _quoted_word(value, w)]
    out = []
    for w in quoted:
        if w is words[-1] and len(plain) < 2:
            continue
        out.append(w)
    return out


def _comma_order(value: str, words: list[_W]) -> Optional[tuple[list[str], list[str]]]:
    """``(first names, surname words)`` of a value written "Surname, First names" ("Quenlow, Chidera",
    "DUPONT, Jean-Pierre", "Pellstone, R."): one comma between two word groups, the words after it not a
    generational qualifier. None for any other layout (v0.3, S1: the comma decides the roles, not the lexicon)."""
    cuts = [k for k in range(1, len(words)) if "," in value[words[k - 1].end:words[k].start]]
    if len(cuts) != 1:
        return None
    k = cuts[0]
    left, right = words[:k], words[k:]
    if any(w.low in GENERATIONAL for w in right) or all(w.initial for w in left):
        return None
    return [w.text for w in right], [w.text for w in left]


def _split_person_impl(value: str) -> tuple[list[str], list[str], list[str]]:
    """Split a person's name into (first names, nicknames, surname words), surfaces without titles.

    Upper-case words are the surname when the name mixes cases ("Jean-Pierre DUPONT",
    "DUPONT Jean-Pierre"); "Surname, First" puts the surname before the comma whatever the lexicon says
    ("Quenlow, Chidera"); otherwise first names come first unless only the last words are first names
    ("Dupont Jean-Pierre"). A word between quotes or brackets is a nickname: it is set aside before the split,
    so the given and family roles do not shift (v0.3, S1).
    """
    words = _strip_titles(_name_words(value))
    words = [w for w in words if not w.sent_end or w.text]
    nicks = _nick_words(value, words)
    if nicks:
        words = [w for w in words if w not in nicks]
    nick_texts = [w.text for w in nicks]
    comma = _comma_order(value, words)
    if comma is not None:
        return comma[0], nick_texts, comma[1]
    firsts, surname = _split_plain_words(value, words)
    return firsts, nick_texts, surname


def _split_plain_words(value: str, words: list[_W]) -> tuple[list[str], list[str]]:
    """(first names, surname words) of the name words ``words`` (nicknames already removed)."""
    if not words:
        return [], []
    if len(words) == 1:
        return [], [words[0].text]
    if words[-1].low in GENERATIONAL and not words[-1].upper:
        # "Vellin fils", "Jean Vellin père": the qualifier belongs to the surname, so father and son differ (CL-20)
        firsts, surname = _split_plain_words(value, words[:-1])
        return firsts, surname + [words[-1].text]
    fn = _lists()[4]

    def is_fn(w: _W) -> bool:
        if w.initial:
            return True
        if w.low in fn:
            return True
        parts = [p for p in w.low.split("-") if p]
        return len(parts) > 1 and all(p in fn for p in parts)

    upper = [w.upper and not w.initial for w in words]
    mixed = any(upper) and any(w.cap and not w.upper and not w.initial and w.low not in PARTICLES for w in words)
    if mixed:
        sur_idx = set(i for i, u in enumerate(upper) if u)
        # particles right before an upper-case surname word belong to the surname
        for i, w in enumerate(words):
            if w.low in PARTICLES and i + 1 < len(words) and (i + 1) in sur_idx:
                sur_idx.add(i)
        for _ in range(3):
            for i, w in enumerate(words):
                if w.low in PARTICLES and i + 1 < len(words) and (i + 1) in sur_idx:
                    sur_idx.add(i)
        firsts = [w.text for i, w in enumerate(words) if i not in sur_idx]
        surname = [w.text for i, w in enumerate(words) if i in sur_idx]
        if firsts and surname:
            return firsts, surname
    # Same case everywhere.
    part_idx = next((i for i, w in enumerate(words) if i > 0 and w.low in PARTICLES and not w.initial), None)
    if is_fn(words[0]) or words[0].initial:
        k = 1
        while k < len(words) - 1 and (is_fn(words[k]) or words[k].initial) and words[k].low not in PARTICLES:
            k += 1
        # A capitalised word after the lexicon first names belongs to the surname ("Hannah Chidera Quenlow" ->
        # surname "Chidera Quenlow"): a middle-name reading was measured worse (merged table cells make
        # three-word surfaces whose middle word is another person's surname) and the benchmark keys agree.
        if part_idx is not None and part_idx < k:
            k = part_idx
        return [w.text for w in words[:k]], [w.text for w in words[k:]]
    if is_fn(words[-1]) and not is_fn(words[0]):
        k = len(words) - 1
        while k > 1 and is_fn(words[k - 1]) and words[k - 1].low not in PARTICLES:
            k -= 1
        return [w.text for w in words[k:]], [w.text for w in words[:k]]
    if part_idx is not None:
        return [w.text for w in words[:part_idx]], [w.text for w in words[part_idx:]]
    return [w.text for w in words[:-1]], [words[-1].text]


def person_token_roles(value: str) -> list[tuple[int, int, str]]:
    """Role of every word of a person value, as ``(start, end, role)`` in the offsets of ``value``: ``"title"``
    (a civility or function title at the start), ``"given"`` (a first name, a middle name, an initial),
    ``"nick"`` (a quoted or bracketed nickname), ``"family"`` (a surname word, its particles and a generational
    qualifier). The roles come from :func:`split_person`, not from the position of the word, so
    'Christopher "Chris" Pellworth' gives given / nick / family and "DUPONT Jean-Pierre" family / given (v0.3, S1).
    """
    words = [w for w in _name_words(value) if not w.sent_end or w.text]
    firsts, nicks, surname = _split_person(value)
    fi, ni, si = list(firsts), list(nicks), list(surname)
    out: list[tuple[int, int, str]] = []
    lead = True
    for w in words:
        if lead and (_title(w) or w.low in _TITLE_FULL):
            out.append((w.start, w.end, "title"))
            continue
        lead = False
        if ni and w.text == ni[0]:
            out.append((w.start, w.end, "nick"))
            ni.pop(0)
        elif fi and w.text == fi[0]:
            out.append((w.start, w.end, "given"))
            fi.pop(0)
        elif si and w.text == si[0]:
            # a person known by one word ("Josephine", "Pellstone") is stored as a surname; for the roles, a
            # lexicon first name alone is a given name (a bare first name in chat, a greeting)
            role = "given" if not firsts and not nicks and len(surname) == 1 and _any_first_name(w.low) else "family"
            out.append((w.start, w.end, role))
            si.pop(0)
    return out


def _any_first_name(low: str) -> bool:
    """``low`` (folded) is a first name of the French lexicon or of the English bundle (loaded when present)."""
    if low in _lists()[4]:
        return True
    try:
        from . import locale as lc

        return low in lc.bundle("en").all_first_names
    except Exception:  # noqa: BLE001 - the English bundle is optional here
        return False


def _role_key(surface: str) -> str:
    return fold_word(surface).replace(".", "").replace("'", "")


def surface_roles(surface: str, entity_value: Optional[str] = None) -> list[tuple[int, int, str]]:
    """Roles (``"given"``, ``"nick"``, ``"family"``, ``"title"``) of the words of one written mention of a person,
    as ``(start, end, role)`` in the offsets of ``surface``.

    A word between quotes or brackets is a nickname. With ``entity_value`` (the stored name of the person the
    mention refers to), every word takes the role it has in that name, so "Ashbury, Ray" gives family / nick for
    the person "Raymond (Ray) Ashbury" and "Pellworth" alone gives family; a diminutive of the usual first name
    counts as a nickname. A word the entity does not hold takes the role of its position in the mention. For the
    benchmark adapters: a nickname is a given name of its own; contiguous given words form one given span (v0.3).
    """
    words = [w for w in _name_words(surface) if not w.sent_end or w.text]
    if not words:
        return []
    known: dict[str, str] = {}
    if entity_value:
        for s, e, role in person_token_roles(entity_value):
            if role != "title":
                known.setdefault(_role_key(entity_value[s:e]), role)
        for first in split_person(entity_value)[0]:
            for nick in _nick_forms(_role_key(first)):
                known.setdefault(nick, "nick")
    positional = {s: role for s, _e, role in person_token_roles(surface)}
    nicks = _nick_words(surface, _strip_titles(words))
    out: list[tuple[int, int, str]] = []
    for w in words:
        role = positional.get(w.start)
        if role == "title":
            out.append((w.start, w.end, role))
            continue
        if w in nicks:
            role = "nick"
        else:
            role = known.get(_role_key(w.text), role) or "given"
        out.append((w.start, w.end, role))
    return out


def _key(words) -> str:
    """Folded key of name words: dots removed, hyphens read as spaces ("Jean-Pierre" == "Jean Pierre")."""
    parts = []
    for w in words:
        f = fold_word(w).replace(".", "").replace("-", " ")
        parts.extend(x for x in f.split() if x)
    return " ".join(parts)


@lru_cache(maxsize=32768)
def person_norm(value: str) -> str:
    """Canonical key of a person: folded "first names surname", titles removed."""
    firsts, surname = split_person(value)
    return " ".join(x for x in (_key(firsts), _key(surname)) if x)


@lru_cache(maxsize=32768)
def surname_key(value: str) -> str:
    """Folded surname of a person value without particles, used to link "M. Dupont" to "Jean-Pierre Dupont",
    "M. Le Goff" to "Jean-Baptiste Le Goff" and "Me Rochère-Vasseur" to "Hélène de La Rochère-Vasseur"."""
    words = _key(split_person(value)[1]).split()
    core = [w for w in words if w not in PARTICLES and w not in ("les", "l")]
    return " ".join(core or words)


@lru_cache(maxsize=32768)
def first_key(value: str) -> str:
    """Folded first names of a person value ("Jean-Pierre DUPONT" -> "jean pierre"), empty if none."""
    return _key(split_person(value)[0])


@lru_cache(maxsize=32768)
def first_initials(value: str) -> str:
    """Initial letters of the first names ("Jean-Pierre" -> "jp"), empty if none."""
    firsts, _ = split_person(value)
    out = ""
    for f in firsts:
        for part in re.findall(r"[^\W\d_]+", fold_word(f)):
            out += part[0]
    return out


@lru_cache(maxsize=32768)
def org_core(value: str) -> str:
    """Company name without legal forms, trigger words and quotes ("la société ACME SAS" -> "ACME")."""
    words = [w for w in _words(value)]
    while words and (words[0].low in _ARTICLES or words[0].low in ("societe", "ste", "denommee") or words[0].prefix
                     or (_legal_form(words[0]) and len(words) > 1)):
        words = words[1:]
    while words and (_legal_form(words[-1]) and len(words) > 1):
        words = words[:-1]
    if len(words) > 1 and words[-1].low == "and" and len(words) < len(_words(value)):
        words = words[:-1]  # v0.4 (T3): "Velltry and Company" -> "Velltry" (English connector before the form)
    if not words:
        return value.strip()
    return value[words[0].start:words[-1].end].strip(" «»\"“”")


@lru_cache(maxsize=32768)
def org_norm(value: str) -> str:
    """Canonical key of a company: folded core name."""
    return " ".join(re.findall(r"[^\W_]+", fold_word(org_core(value))))


@dataclass(frozen=True)
class Variant:
    """A token sequence that refers to a known entity.

    need: "" (always), "title" (a person title must precede), "salutation" (after "Bonjour",
    "Cher"... or before a signature), "first" (a first name standing alone, not inside a longer
    name, not after a place preposition), "org" (a legal form or "société" must be next to it).
    case: "any", "cap" (each significant word starts with a capital), "upper" (all upper case).
    initials: single-letter tokens must be upper-case letters in the text.
    """

    tokens: tuple[str, ...]
    need: str = ""
    case: str = "any"
    initials: bool = False
    partial: bool = False  # surname alone, initials, first name alone, company head word


def _weak(toks: tuple[str, ...]) -> bool:
    stop, legal, common, _fn, afn, months, wl = _lists()
    sig = [t for t in toks if t not in PARTICLES]
    if not sig or sum(len(t) for t in sig) < 3:
        return True
    return all(t in common or t in afn or t in stop or t in legal or t in months or t in wl for t in sig)


def person_variants(value: str) -> list[Variant]:
    """Every token sequence that refers to the person ``value``."""
    return list(_person_variants(value))


@lru_cache(maxsize=8192)
def _person_variants(value: str) -> tuple[Variant, ...]:
    return tuple(_person_variants_impl(value))


def _distinctive(t: str) -> bool:
    """A name word that is not a common, function, legal or calendar word (usable alone)."""
    stop, legal, common, _fn, _afn, months, wl = _lists()
    return len(t) >= 3 and t not in common and t not in stop and t not in legal and t not in months and t not in wl


def _plain(t: str) -> bool:
    """A name word that may be a common word but is not a function, legal or calendar word."""
    stop, legal, _common, _fn, _afn, months, wl = _lists()
    return len(t) >= 3 and t not in stop and t not in legal and t not in months and t not in wl


def _lexicon_word(t: str) -> bool:
    stop, legal, common, _fn, afn, months, wl = _lists()
    return t in stop or t in legal or t in common or t in afn or t in months or t in wl


def _initial_letters(words, drop_particles: bool) -> list[str]:
    out: list[str] = []
    for w in words:
        f = fold_word(w)
        if drop_particles and (f.rstrip("'") in PARTICLES or f.rstrip("'") in ("les", "l")):
            continue
        for part in re.findall(r"[^\W\d_]+", f):
            out.append(part[0])
    return out


@lru_cache(maxsize=8192)
def _initials_groups(value: str) -> tuple[tuple[str, ...], ...]:
    """Letter groups of a person's initials: usual or all first names, surname with or without particles
    ("Jean-Luc de Pradimont" -> ("j", "l", "p"), ("j", "l", "d", "p")). Two to five letters."""
    firsts, surname = split_person(value)
    if not firsts or not surname:
        return ()
    if any(fold_word(w) in GENERATIONAL for w in surname):
        surname = tuple(w for w in surname if fold_word(w) not in GENERATIONAL)
    out: list[tuple[str, ...]] = []
    for fl in (_initial_letters(firsts[:1], False), _initial_letters(firsts, False)):
        for sl in (_initial_letters(surname, True), _initial_letters(surname, False)):
            g = tuple(fl + sl)
            if 2 <= len(g) <= 5 and g not in out:
                out.append(g)
    return tuple(out)


@lru_cache(maxsize=None)
def _acronym_stop() -> frozenset:
    return lx.acronym_stoplist()


def _plain_group_ok(g: str) -> bool:
    """A group of capitals usable as a person's initials or a company acronym: not a common acronym, a legal form,
    a title or a dictionary word."""
    return g not in _acronym_stop() and g not in LEGAL_FORMS and g not in TITLE_WORDS and not _lexicon_word(g) \
        and g not in ORG_GENERIC and g not in _instruments()


def person_initials(value: str) -> list[str]:
    """Initials groups of a person value without dots, upper case ("Ludovic Pradimont" -> ["LP"]), 2 to 4 letters,
    common acronyms and dictionary words left out. Used for the labels (R1, CL-02)."""
    out = []
    for g in _initials_groups(value):
        j = "".join(g)
        if len(g) <= 4 and _plain_group_ok(j) and j.upper() not in out:
            out.append(j.upper())
    return out


@lru_cache(maxsize=8192)
def _nick_forms(first: str) -> tuple[str, ...]:
    """Diminutives of a first name (folded): prefixes of 4+ letters ("steph" of "stephanie") and the standard
    diminutives of ``nicknames_fr.txt``. A prefix that is itself a first name one or two letters shorter is a
    gender form ("christian" of "christiane"), not a diminutive (R1, CL-03)."""
    if len(first) < 5 or not first.isalpha():
        return tuple(sorted(lx.nicknames().get(first, ())))
    afn = _lists()[4]
    out = set(lx.nicknames().get(first, ()))
    for k in range(4, len(first) - 1):
        p = first[:k]
        if not _distinctive(p) and p not in afn:
            continue
        if p in afn and k >= len(first) - 2:
            continue
        if p in _lists()[2] or p in _lists()[0]:
            continue
        out.add(p)
    return tuple(sorted(out))


def _person_variants_impl(value: str) -> list[Variant]:
    firsts, surname = split_person(value)
    out: dict[tuple, Variant] = {}

    def add(toks: tuple[str, ...], need: str = "", case: str = "any", initials: bool = False,
            partial: bool = False) -> None:
        if not toks:
            return
        key = (toks, need, case)
        if key not in out:
            out[key] = Variant(toks, need, case, initials, partial)

    stop, legal, common, fn, _afn, months, wl = _lists()
    full_words = _strip_titles(_name_words(value))
    full = tuple(t for w in full_words for t in phrase_tokens(w.text))
    st = tuple(t for w in surname for t in phrase_tokens(w))
    if len(full) > 1:
        add(full)
    if firsts and st:
        ft = tuple(t for w in firsts for t in phrase_tokens(w))
        if ft:
            add(ft + st)
            add(st + ft)
            if len(st) == 1:
                add(st + ft, need="comma")  # "Pradimont, Ludovic" (R1, CL-05)
        first_word = tuple(phrase_tokens(firsts[0]))
        if len(firsts) > 1 and first_word:
            add(first_word + st)
            add(st + first_word)
            if len(st) == 1:
                add(st + first_word, need="comma")
        if first_word:
            ini = tuple(p[0] for p in first_word)
            add(ini + st, initials=True, partial=True)
            add(st + ini, initials=True, partial=True)
            if len(ini) > 1:
                add(("".join(ini),) + st, case="cap", initials=True, partial=True)
                add((ini[0],) + st, initials=True, partial=True)
        # First name alone: in a greeting or a signature, or anywhere when it is distinctive (not a common word;
        # the masker also checks it is unique among known people). The usual first name of a person with several
        # given names too (R1, CL-04a); a first name that is a common word only capitalised in mid-sentence (CL-04d).
        alone = [ft] + ([first_word] if len(firsts) > 1 and first_word and first_word != ft else [])
        for toks in alone:
            if not toks or not any(len(t) > 1 for t in toks):
                continue
            add(toks, need="salutation", case="cap", partial=True)
            if all(_distinctive(t) for t in toks):
                add(toks, need="first", case="cap", partial=True)
                if all(len(t) >= 4 and t.isalpha() and t not in common for t in toks):
                    # chat text ("ping priya", "priya's draft", "@priya"): the first name in any case (v0.2)
                    add(toks, need="casual", case="any", partial=True)
            elif all(_plain(t) for t in toks):
                add(toks, need="mid", case="cap", partial=True)
        # Diminutives of the usual first name (R1, CL-03); the masker keeps them for registered people only.
        if len(first_word) == 1:
            for nick in _nick_forms(first_word[0]):
                add((nick,), need="nick", case="cap", partial=True)
        # Initials alone: "L. P." anywhere, "LPD" alone, "LP" in an initials context (R1, CL-02).
        for g in _initials_groups(value):
            add(g, need="dotted", case="upper", initials=True, partial=True)
            j = "".join(g)
            if len(g) <= 4 and _plain_group_ok(j):
                add((j,), need="ini_ctx" if len(g) == 2 else "inits", case="upper", partial=True)
    elif not firsts and len(st) == 1 and st[0] in fn and _plain(st[0]):
        # A person known by a first name only ("son fils Maxence"): the first name alone (R1, CL-04e).
        add(st, need="first" if _distinctive(st[0]) else "mid", case="cap", partial=True)
    # A nickname the name itself declares ('Christopher "Chris" Pellworth', "Raymond (Ray) Ashbury") refers to the
    # person like a first name: with the surname in both orders ("Chris Pellworth", "Ashbury, Ray") and alone under
    # the first-name context rules (v0.3, S1). Declared in the text, it needs no registration to be used.
    for nick in person_nicknames(value):
        nt = tuple(phrase_tokens(nick))
        if not nt or not any(len(t) > 1 for t in nt):
            continue
        if st:
            add(nt + st)
            add(st + nt)
            if len(st) == 1:
                add(st + nt, need="comma")
        add(nt, need="salutation", case="cap", partial=True)
        if all(_distinctive(t) for t in nt):
            add(nt, need="first", case="cap", partial=True)
            if all(len(t) >= 4 and t.isalpha() and t not in common for t in nt):
                add(nt, need="casual", case="any", partial=True)
        elif all(_plain(t) for t in nt):
            add(nt, need="mid", case="cap", partial=True)
    if st:
        weak = _weak(st)
        part = bool(firsts)
        if weak:
            add(st, need="title", case="cap", partial=part)
            sig = [t for t in st if t not in PARTICLES]
            if sig and not any(t in common or t in stop or t in legal or t in months or t in wl for t in sig) \
                    and sum(len(t) for t in sig) >= 4:
                # A surname that is only weak because it is also a first name ("MARTIN", "BENALI")
                # is still matched alone when written in upper case, as in pleadings.
                add(st, case="upper", partial=part)
            elif sig and all(_plain(t) for t in sig) and sum(len(t) for t in sig) >= 4:
                # A common-word surname in capitals inside a phrase of a line that is not all capitals (R1, CL-05)
                add(st, need="mixed_line", case="upper", partial=part)
            if sig and all(_plain(t) for t in sig) and sum(len(t) for t in sig) >= 4:
                # next to a party separator ("Boulanger c/ X") (R1, CL-05)
                add(st, need="party", case="cap", partial=part)
                if any(t in common for t in sig):
                    # a common-word surname capitalised in mid-sentence; a surname that is a first name stays
                    # ambiguous with another person's first name (R1, CL-05)
                    add(st, need="mid", case="cap", partial=part)
        else:
            long_enough = sum(len(t) for t in st if t not in PARTICLES) >= 5
            add(st, case="any" if (len(st) > 1 or long_enough) else "cap", partial=part)
        # "de La Fontaine" -> "La Fontaine"
        k = 0
        while k < len(st) - 1 and st[k] in _SURNAME_DROP_PARTICLES:
            k += 1
        if 0 < k < len(st):
            st2 = st[k:]
            add(st2, need="title" if _weak(st2) else "", case="cap", partial=True)
    return list(out.values())


def org_acronyms(value: str) -> list[str]:
    """Upper-case acronyms of a company value, 3 to 6 letters, from the first letters of its words with and without
    particles and generic words ("Brasseries Velanor de Quimpertin SAS" -> "BVDQ", "BVQ"). Common acronyms and
    dictionary words are left out (R1, CL-01)."""
    return list(_org_acronyms(value))


@lru_cache(maxsize=8192)
def _org_acronyms(value: str) -> tuple[str, ...]:
    words = [w for w in _words(org_core(value)) if not w.digit and w.low != "&" and not _legal_form(w)]
    named = [w for w in words if not w.prefix and w.low not in PARTICLES and w.low not in _ORG_CONNECT]
    if len(named) < 2:
        return ()
    out: list[str] = []
    for ws in (words, named, [w for w in named if w.low not in ORG_GENERIC]):
        letters = "".join(w.low[0] for w in ws if w.low and w.low[0].isalpha())
        if 3 <= len(letters) <= 6 and _plain_group_ok(letters) and letters.upper() not in out:
            out.append(letters.upper())
    return tuple(out)


def org_variants(value: str) -> list[Variant]:
    """Every token sequence that refers to the company ``value``."""
    return list(_org_variants(value))


@lru_cache(maxsize=8192)
def _org_variants(value: str) -> tuple[Variant, ...]:
    return tuple(_org_variants_impl(value))


def _org_variants_impl(value: str) -> list[Variant]:
    out: dict[tuple, Variant] = {}

    def add(toks: tuple[str, ...], need: str = "", case: str = "any", partial: bool = False) -> None:
        if toks and (toks, need, case) not in out:
            out[(toks, need, case)] = Variant(toks, need, case, False, partial)

    full = phrase_tokens(value)
    core = phrase_tokens(org_core(value))
    if len(full) > 1:
        add(full, case="any" if not _weak(full) else "cap")
    # Acronym of the name, as a whole capital word ("BVQ", "BVQ-310", "BVQ2024") (R1, CL-01).
    for acr in _org_acronyms(value):
        add((acr.lower(),), need="acro", case="upper", partial=True)
    # Leading words of a long name ("Lumière Froid" of "Lumière Froid Services") (R1, CL-21).
    for k in range(2, len(core)):
        pre = core[:k]
        if pre[-1] in PARTICLES or pre[-1] in _ORG_CONNECT:
            continue
        sig = [t for t in pre if t not in PARTICLES and t not in _ORG_CONNECT]
        if len(sig) >= 2 and all(len(t) >= 3 for t in sig) and any(t not in ORG_GENERIC and _plain(t) for t in sig):
            add(pre, case="cap", partial=True)
    if core:
        if _weak(core):
            add(core, need="org", case="cap")
            add(core, case="upper")
        else:
            long_enough = len(core) > 1 or len(core[0]) >= 5
            add(core, case="any" if long_enough else "cap")
        if len(core) > 1:
            head = core[:1]
            if head[0] in ORG_GENERIC:
                # "TRANSPORTS GRAVELOTTE", "Garage Gravelotte": the name without the trade noun ("Gravelotte")
                rest = core[1:]
                while rest and rest[0] in PARTICLES:
                    rest = rest[1:]
                if rest and not _weak(rest) and sum(len(t) for t in rest) >= 4:
                    add(rest, case="any" if sum(len(t) for t in rest) >= 5 else "cap", partial=True)
            elif not _weak(head) and len(head[0]) >= 4:
                add(head, case="cap", partial=True)
    return list(out.values())


# ---------------------------------------------------------------------------
# Dossier titles: "X c. Y", "X contre Y", "X v. Y"
# ---------------------------------------------------------------------------

_TITLE_SPLIT_RE = re.compile(r"\s+(?:c\.|C\.|c/|C/|contre|Contre|CONTRE|v\.|V\.|vs\.?|VS\.?|/)\s+")
_PARTY_SPLIT_RE = re.compile(r"\s*[,;]\s*|\s+et\s+(?=(?:M\.|Mme|Mlle|Me|Monsieur|Madame|la soci[ée]t[ée]|Soci[ée]t[ée])\b)")
_PARTY_ET_RE = re.compile(r"\s+et\s+")
# v0.4.3 (fix): "et al." / "et autres" / ", consorts" at the end of a party, as whole words only. The 0.4.0 pattern
# had no word boundary, so the "al" at the end of a name was cut ("Duval" -> "Duv") and the real name stayed clear.
_OTHERS_RE = re.compile(r"(?i)(?:\s*[,&]\s*|\s+(?:et\s+)?|^)\b(?:autres|consorts|cts|al\.?|a\.)(?:\s*$)")


# Words that describe a dossier rather than name a party ("Dossier X", "X - constat", "Contentieux prud'homal X").
_TITLE_DESCRIPTORS = frozenset({
    "dossier", "dossiers", "affaire", "affaires", "aff", "litige", "litiges", "contentieux", "procedure",
    "procedures", "projet", "projets", "client", "clients", "cliente", "clientes", "succession", "successions",
    "divorce", "prud'homal", "prud'homale", "prudhomal", "prudhomale", "prud'hommes", "prudhommes", "appel",
    "constat", "contrat", "contrats", "correspondance", "consultation", "audit", "matter", "case", "re", "ref",
    "reference", "note", "notes", "avis", "cession", "bail", "vente", "liquidation", "indivision", "commercial",
    "commerciale", "social", "sociale", "penal", "penale", "civil", "civile", "famille", "familial", "familiale",
    "sinistre", "recouvrement", "arbitrage", "mediation", "transaction", "acquisition", "fusion",
    "licenciement", "rupture", "expertise", "refere", "pieces", "piece", "archive", "archives", "en", "cours",
    "n", "no", "numero", "nouveau", "nouvelle", "ancien", "ancienne", "brouillon", "copie", "suivi", "travail",
})
# Separators that start a description inside a title: " - ", " – ", " | ", " : ".
_TITLE_DESC_SPLIT_RE = re.compile(r"\s+[-–—|]\s+|\s*[:|]\s+|\s+[-–—]\s*$")
_TITLE_BRACKETS_RE = re.compile(r"[(\[{][^)\]}]*[)\]}]?")


def _is_descriptor(w: _W) -> bool:
    if w.low in ORG_TRIGGERS or w.low in LEGAL_FORMS or w.low in TITLE_WORDS or w.low in _TITLE_FULL:
        return False
    if w.digit or w.low in _TITLE_DESCRIPTORS:
        return True
    _stop, legal, _common, _fn, _afn, months, _wl = _lists()
    # Lower-case legal words or months at the edge of a title ("dossier", "appel") are descriptions.
    return not w.cap and (w.low in legal or w.low in months)


def _strip_descriptors(piece: str) -> str:
    """Drop description words at both ends of a title piece: "Affaire Dupont" -> "Dupont"."""
    words = _words(piece)
    i, j = 0, len(words)
    while i < j and (_is_descriptor(words[i]) or words[i].prefix or words[i].text in ("n°", "N°")):
        i += 1
    while j > i and not words[j - 1].cap and not words[j - 1].upper and _is_descriptor(words[j - 1]):
        j -= 1
    if i >= j:
        return ""
    return piece[words[i].start:words[j - 1].end].strip()


def _title_segments(title: str) -> list[str]:
    """The parts of a title that name the parties: bracketed notes and " - description" tails dropped."""
    t = _TITLE_BRACKETS_RE.sub(" ", title)
    segs = [s.strip() for s in _TITLE_DESC_SPLIT_RE.split(t) if s and s.strip()]
    if not segs:
        return []
    with_sides = [s for s in segs if _TITLE_SPLIT_RE.search(f" {s} ")]
    return with_sides or segs[:1]


def has_party_separator(title: str) -> bool:
    """True when a title names opposing parties ("X c. Y", "X contre Y", "X v. Y", "X / Y")."""
    return bool(title and _TITLE_SPLIT_RE.search(f" {title.strip()} "))


def _classify_party(piece: str) -> Optional[tuple[str, str]]:
    piece = piece.strip(" .;:()[]\"'«»")
    piece = _OTHERS_RE.sub("", piece).strip(" .;:,")
    piece = _strip_descriptors(piece).strip(" .;:,")
    if not piece or fold_word(piece) in ("autres", "consorts", "cts", "al", "a"):
        return None
    words = _words(piece)
    if not words:
        return None
    if _title(words[0]) or words[0].low in _TITLE_FULL:
        rest = _strip_titles(words)
        if not rest:
            return None
        return "PERSON", piece[rest[0].start:].strip()
    lows = [w.low for w in words]
    if any(_legal_form(w) for w in words) or lows[0] in ORG_TRIGGERS or (
        len(lows) > 1 and lows[0] in ("la", "le", "les", "l") and lows[1] in ORG_TRIGGERS
    ):
        k = 0
        while k < len(words) - 1 and (words[k].low in ("la", "le", "les", "l", "societe", "ste") or words[k].prefix):
            k += 1
        return "ORG", piece[words[k].start:].strip()
    if any(l in _ORG_KEYWORDS for l in lows):
        return "ORG", piece
    if len(words) >= 2 and _is_first_name(words[0]):
        return "PERSON", piece
    if len(words) >= 2 and words[0].upper and _is_first_name(words[-1]):
        return "PERSON", piece
    if len(words) == 1:
        w = words[0]
        if w.upper and len(w.low) <= 4:
            return "ORG", piece
        return "PERSON", piece
    if len(words) <= 3:
        return "PERSON", piece
    return "ORG", piece


def parse_parties(title: str) -> list[tuple[str, str]]:
    """Parties named in a dossier title: ``"Dupont c. ACME SAS"`` -> ``[("PERSON", "Dupont"), ("ORG", "ACME SAS")]``."""
    if not title or not title.strip():
        return []
    sides = [s for seg in _title_segments(title.strip()) for s in _TITLE_SPLIT_RE.split(seg)]
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for side in sides:
        pieces = _PARTY_SPLIT_RE.split(side)
        expanded: list[str] = []
        for piece in pieces:
            lows = [w.low for w in _words(piece)]
            has_org = any(l in LEGAL_FORMS or l in ORG_TRIGGERS for l in lows)
            if not has_org and _PARTY_ET_RE.search(piece):
                expanded.extend(_PARTY_ET_RE.split(piece))
            else:
                expanded.append(piece)
        for piece in expanded:
            got = _classify_party(piece)
            if got and (got[0], fold_word(got[1])) not in seen:
                seen.add((got[0], fold_word(got[1])))
                out.append(got)
    return out
