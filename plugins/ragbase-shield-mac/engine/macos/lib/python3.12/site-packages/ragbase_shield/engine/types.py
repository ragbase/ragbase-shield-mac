"""Shared engine types (contract: ARCHITECTURE.md section 4.1)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Protocol

KINDS = (
    "PERSON", "ORG", "ADDRESS", "EMAIL", "PHONE", "IBAN", "NIR", "SIREN", "SIRET",
    "VAT", "CARD", "CASE_REF", "ID_CODE", "URL", "BIRTHDATE",
    # v0.2 (DETECTORS): login names and chat mentions, staff identifiers, account numbers
    "USERNAME", "EMPLOYEE_ID", "ACCOUNT_NUMBER",
    # plugin round 2 (16 Nov 2026): the place of birth next to a birth date ("né le ... à Angers")
    "BIRTHPLACE",
)

# kind -> (fr prefix, en prefix, letter codes?)
PREFIXES = {
    "PERSON": ("PERSONNE", "PERSON", True),
    "ORG": ("SOCIETE", "COMPANY", True),
    "ADDRESS": ("ADRESSE", "ADDRESS", False),
    "EMAIL": ("EMAIL", "EMAIL", False),
    "PHONE": ("TEL", "PHONE", False),
    "IBAN": ("IBAN", "IBAN", False),
    "NIR": ("NIR", "SSN", False),
    "SIREN": ("SIREN", "SIREN", False),
    "SIRET": ("SIRET", "SIRET", False),
    "VAT": ("TVA", "VAT", False),
    "CARD": ("CARTE", "CARD", False),
    "CASE_REF": ("RG", "CASEREF", False),
    "ID_CODE": ("IDENT", "IDENT", False),
    "URL": ("SITE", "SITE", False),
    "BIRTHDATE": ("NAISSANCE", "BIRTHDATE", False),
    # v0.2 (DETECTORS)
    "USERNAME": ("PSEUDO", "USERNAME", False),
    "EMPLOYEE_ID": ("MATRICULE", "EMPLOYEEID", False),
    "ACCOUNT_NUMBER": ("COMPTE", "ACCOUNT", False),
    "BIRTHPLACE": ("LIEU", "PLACE", False),
}

#: Kinds whose labelled values teach the masker an id layout ("EMP-ddddd"): unlabelled values of the same layout
#: are then masked in the whole dossier (v0.2, id shape memory; see ``MemoryStore.add_shape``).
SHAPE_KINDS = ("EMPLOYEE_ID", "ACCOUNT_NUMBER")

_LETTER_PREFIXES = sorted({p for k, (fr, en, letters) in PREFIXES.items() if letters for p in (fr, en)})
_NUMBER_PREFIXES = sorted({p for k, (fr, en, letters) in PREFIXES.items() if not letters for p in (fr, en)})

#: Upper bounds of the code ranges, shared by every code pattern (CODE_RE, UNMASK_RE, the service and the
#: document writers): letter codes PERSONNE_A..PERSONNE_ZZZZ, number codes EMAIL_1..EMAIL_999999.
MAX_LETTERS = 4
MAX_DIGITS = 6
#: Number of letter codes available per (dossier, kind): A..ZZZZ.
MAX_LETTER_CODES = sum(26 ** k for k in range(1, MAX_LETTERS + 1))
MAX_NUMBER_CODES = 10 ** MAX_DIGITS - 1

CODE_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:"
    r"(?:" + "|".join(_LETTER_PREFIXES) + r")_[A-Z]{1,%d}" % MAX_LETTERS
    + r"|(?:" + "|".join(_NUMBER_PREFIXES) + r")_[0-9]{1,%d}" % MAX_DIGITS
    + r"|DOSSIER_[0-9]{1,6}|DOC_[0-9]{1,7}|OUTPUT_[0-9]{1,7}"
    r")(?![A-Za-z0-9_])"
)

#: F0 (0.4.3): a registered value is WEAK when, once spaces and punctuation are removed, it has fewer than
#: ``WEAK_MIN_CHARS`` characters, or only digits and fewer than ``WEAK_MIN_DIGITS`` of them ("4", "13", "35000",
#: "A", "Li", "3M"). A weak value stays masked where its detector found it, but it is never spread: never searched
#: for in other places (dictionary pass of the masker), never used by the last check, never made into an entity of
#: another dossier. The rule reads the value only, so it also holds for the values already in a vault.
WEAK_MIN_CHARS = 3
WEAK_MIN_DIGITS = 6
_WEAK_STRIP_RE = re.compile(r"[\W_]+")


def is_weak_value(value: object) -> bool:
    """True for a registered value too short to be searched for safely (see ``WEAK_MIN_CHARS``)."""
    body = _WEAK_STRIP_RE.sub("", str(value or ""))
    return len(body) < WEAK_MIN_CHARS or (body.isdigit() and len(body) < WEAK_MIN_DIGITS)


def spreadable(ents) -> list:
    """The entities whose values may be searched for outside the place where they were found (F0: the weak values
    are left out). THE one list of known values for the dictionary pass and the last check."""
    return [e for e in ents if not is_weak_value(e.value)]


@dataclass(frozen=True)
class Entity:
    kind: str
    value: str
    norm: str
    code: str
    source: str = "detected"  # "detected", "user", "ner" (v0.2, additive; stores may leave the default)


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    kind: str
    value: str
    source: str  # "regex", "dictionary", "rule", "ner"


@dataclass
class MaskResult:
    text: str
    spans: list[Span] = field(default_factory=list)
    new_entities: list[Entity] = field(default_factory=list)
    # v0.2 (SYNTH, additive): "codes" or "synthetic"; the entity code and the text written for each span (aligned
    # with ``spans``), so that a caller can list what was replaced in synthetic mode without reading codes back.
    mode: str = "codes"
    codes: list[str] = field(default_factory=list)
    replacements: list[str] = field(default_factory=list)
    # v0.2 (integrator, additive): language of the text ("fr" or "en") and the NER backend that ran on it
    # ("" rules only, "gliner2:fastino", "custom", "skipped: text too long").
    language: str = "fr"
    ner: str = ""
    # v0.3 (S7, additive): model finds the gated NER layer never masks on its own (named assets, government ids,
    # secrets): ``(start, end, kind, model label)`` in the offsets of the input text, for a review step where a
    # person decides. Empty in rules mode.
    watch: list = field(default_factory=list)


class EntityStore(Protocol):
    """Entity store of one dossier. Optional methods a store may add (the masker checks with ``getattr``):
    ``upgrade(code, value, norm)``, ``keep_clear(norm, value)``, ``revision``, and since v0.2 the id shape memory
    ``add_shape(kind, template)`` / ``shapes() -> list[(kind, template)]`` (see :class:`MemoryStore`)."""

    def all(self) -> list[Entity]: ...
    def get_by_norm(self, kind: str, norm: str) -> Optional[Entity]: ...
    def get_by_code(self, code: str) -> Optional[Entity]: ...
    def add(self, kind: str, value: str, norm: str) -> Entity: ...
    def is_kept_clear(self, norm: str) -> bool: ...


def prefix_for(kind: str, locale: str = "fr") -> str:
    fr, en, _ = PREFIXES[kind]
    return fr if locale == "fr" else en


def letters(n: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA, ... (bijective base 26)."""
    s = ""
    n += 1
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


class CodeRangeError(ValueError):
    """No code left for this kind in this dossier (the code patterns would no longer recognise it)."""


def make_code(kind: str, index: int, locale: str = "fr") -> str:
    """index is 0-based within (dossier, kind). Raises CodeRangeError past PERSONNE_ZZZZ / EMAIL_999999."""
    fr, en, use_letters = PREFIXES[kind]
    prefix = fr if locale == "fr" else en
    if index < 0 or index >= (MAX_LETTER_CODES if use_letters else MAX_NUMBER_CODES):
        raise CodeRangeError(f"{prefix}: plus de code disponible dans ce dossier (code range exhausted)")
    return f"{prefix}_{letters(index)}" if use_letters else f"{prefix}_{index + 1}"


class MemoryStore:
    """In-memory EntityStore, used by tests and by one-off masking without a dossier."""

    def __init__(self, locale: str = "fr"):
        self.locale = locale
        self._by_code: dict[str, Entity] = {}
        self._by_norm: dict[tuple[str, str], Entity] = {}
        self._counts: dict[str, int] = {}
        self._clear: set[str] = set()
        self._shapes: dict[str, list[str]] = {}  # kind -> id layouts learnt from labelled values (v0.2)
        self.revision = 0  # changes on every add, upgrade and keep-clear (maskers cache their matchers on it)

    def all(self) -> list[Entity]:
        return list(self._by_code.values())

    def get_by_norm(self, kind: str, norm: str) -> Optional[Entity]:
        return self._by_norm.get((kind, norm))

    def get_by_code(self, code: str) -> Optional[Entity]:
        return self._by_code.get(code)

    def add(self, kind: str, value: str, norm: str, source: str = "detected") -> Entity:
        found = self._by_norm.get((kind, norm))
        if found:
            return found
        i = self._counts.get(kind, 0)
        self._counts[kind] = i + 1
        if source not in ("detected", "user", "ner"):
            source = "detected"
        ent = Entity(kind=kind, value=value, norm=norm, code=make_code(kind, i, self.locale), source=source)
        self._by_code[ent.code] = ent
        self._by_norm[(kind, norm)] = ent
        self.revision += 1
        return ent

    def upgrade(self, code: str, value: str, norm: str) -> Optional[Entity]:
        """Give an existing code a fuller value ("Ouardi" -> "Mehdi Ouardi"); None if the key is taken."""
        old = self._by_code.get(code)
        if old is None or (old.kind, norm) in self._by_norm:
            return None
        new = Entity(kind=old.kind, value=value, norm=norm, code=code)
        self._by_code[code] = new
        self._by_norm.pop((old.kind, old.norm), None)
        self._by_norm[(old.kind, norm)] = new
        self.revision += 1
        return new

    def confirm(self, code: str) -> Optional[Entity]:
        """v0.3 (S7): an entity the model alone had found is confirmed by the rules or by a second signal: its
        source becomes "detected" (the containment of ``Masker._refs_for`` lifts). None when the code is unknown."""
        old = self._by_code.get(code)
        if old is None:
            return None
        if old.source != "ner":
            return old
        new = Entity(kind=old.kind, value=old.value, norm=old.norm, code=code, source="detected")
        self._by_code[code] = new
        self._by_norm[(old.kind, old.norm)] = new
        self.revision += 1
        return new

    def is_kept_clear(self, norm: str) -> bool:
        return norm in self._clear

    def keep_clear(self, norm: str, value: bool = True) -> None:
        (self._clear.add if value else self._clear.discard)(norm)
        self.revision += 1

    # ---- synthetic replacement mode (v0.2, SYNTH; optional methods of the EntityStore protocol)
    def synthetic_seed(self) -> bytes:
        """Secret seed of the fake values (random per store; the vault derives it from the dossier key)."""
        if not hasattr(self, "_synthetic_seed"):
            import os

            self._synthetic_seed = os.urandom(32)
        return self._synthetic_seed

    def synthetic(self, code: str) -> Optional[str]:
        """The fake value given to ``code`` in synthetic mode, or None."""
        return getattr(self, "_synthetic", {}).get(code)

    def set_synthetic(self, code: str, value: str) -> None:
        if not hasattr(self, "_synthetic"):
            self._synthetic: dict[str, str] = {}
        if code not in self._by_code or not isinstance(value, str) or not value.strip():
            raise ValueError("unknown code or empty synthetic value")
        self._synthetic[code] = value
        self.revision += 1

    def get_by_synthetic_norm(self, norm: str) -> Optional[Entity]:
        """The entity whose fake value normalises to ``norm`` (``engine.normalize``), or None."""
        from .normalize import normalize

        for code, value in getattr(self, "_synthetic", {}).items():
            if normalize(value) == norm:
                return self._by_code.get(code)
        return None

    # -- id shape memory (v0.2, DETECTORS): "EMP-ddddd" learnt from "Employee ID: EMP-00123" ------------------
    def add_shape(self, kind: str, template: str) -> None:
        """Remember an id layout of ``kind`` (letters kept, digits written ``d``); a known layout is ignored."""
        if kind not in SHAPE_KINDS or not template or "d" not in template:
            return
        lst = self._shapes.setdefault(kind, [])
        if template not in lst:
            lst.append(template)
            self.revision += 1

    def shapes(self) -> list[tuple[str, str]]:
        """Every (kind, template) learnt so far, in insertion order."""
        return [(k, t) for k, lst in self._shapes.items() for t in lst]
