"""Structured identifier detectors: regular expressions plus check digits.

Every detector yields :class:`Candidate` objects. ``valid`` says whether the masker masks the span.
``checked`` is False when the identifier has a check digit and the check fails. A wrong check digit is
masked only when the shape is exact (an IBAN of the exact national length, mostly digits) or when a
label names the identifier on the line (SIREN, SIRET, RCS): such a candidate has ``valid=True,
checked=False``. Other wrong check digits stay ``valid=False`` (amounts, random numbers). Every wrong check
digit is reported by :func:`find_suspects` so the review list can show it; :func:`detect_structured`
lists the identifiers whose check passes (or that have no check digit).

Kinds: EMAIL, PHONE, IBAN, NIR, SIREN, SIRET, VAT, CARD, CASE_REF, ID_CODE, URL, BIRTHDATE, ADDRESS, ORG
(school and care establishment names only); since v0.2 USERNAME (chat mentions, handles, logins), EMPLOYEE_ID and
ACCOUNT_NUMBER (labelled values, table columns, US routing numbers, learnt id layouts). No network access, no
external data. :func:`detect_candidates` takes the language of the text (``lang``: the detectors of another
language are skipped; ``None`` runs every detector) and the id layouts of the dossier (``shapes``).
"""
from __future__ import annotations

import bisect
import ipaddress
import re
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from typing import Callable, Iterable, Iterator, Optional

from . import lexicon as lx
from .normalize import fold_word
from .types import CODE_RE, Span

__all__ = [
    "Candidate",
    "detect_candidates",
    "detect_structured",
    "find_suspects",
    "structured_norm",
    "table_cells",
    "imo_valid",
    "luhn_valid",
    "luhn_check_digit",
    "siren_valid",
    "siret_valid",
    "nir_key",
    "nir_valid",
    "iban_valid",
    "iban_check_digits",
    "vat_fr_key",
    "vat_fr_valid",
    "isin_valid",
    "lei_valid",
    "url_is_public",
    "aba_routing_valid",
    "shape_template",
    "IBAN_LENGTHS",
    "PUBLIC_URL_SUFFIXES",
    "PUBLIC_HANDLES",
    "DETECTORS",
    "IntervalSet",
    "DETECTORS_EN",
    "detect_candidates_en",
    "ssn_valid",
    "itin_valid",
    "nino_valid",
]


@dataclass(frozen=True)
class Candidate:
    start: int
    end: int
    kind: str
    valid: bool
    priority: int
    detector: str
    checked: bool = True  # False: the identifier has a check digit and it fails


# Tie-break between structured detectors on the same text: higher wins.
PRIORITY = {
    "EMAIL": 10, "IBAN": 10, "NIR": 10, "VAT": 10, "SIRET": 9, "ISIN": 9, "LEI": 9,
    "CARD": 8, "SIREN": 7, "BIRTHDATE": 7, "CASE_REF": 6, "PHONE": 6, "URL": 5,
    "ID_CODE": 5, "ADDRESS": 4, "ORG": 3,
    # v0.2: a labelled account number beats a SIREN or phone shape of the same span; an employee id or a username
    # beats the generic labelled reference (ID_CODE - 1)
    "ACCOUNT_NUMBER": 8, "EMPLOYEE_ID": 6, "USERNAME": 6, "BIRTHPLACE": 4,
}

# ---------------------------------------------------------------------------
# Check digits
# ---------------------------------------------------------------------------


def luhn_valid(digits: str) -> bool:
    """Luhn (mod 10) check over a string of digits."""
    if not digits or not digits.isdigit():
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = ord(ch) - 48
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def luhn_check_digit(partial: str) -> str:
    """The digit to append to ``partial`` so that the result passes Luhn."""
    for d in "0123456789":
        if luhn_valid(partial + d):
            return d
    raise ValueError("no Luhn digit")  # pragma: no cover


def siren_valid(value: str) -> bool:
    """SIREN: 9 digits, Luhn."""
    s = re.sub(r"\D", "", value)
    return len(s) == 9 and luhn_valid(s)


def siret_valid(value: str) -> bool:
    """SIRET: 14 digits, Luhn on the SIRET and on its SIREN; La Poste (356000000) uses sum mod 5."""
    s = re.sub(r"\D", "", value)
    if len(s) != 14:
        return False
    if s[:9] == "356000000":
        return sum(int(c) for c in s) % 5 == 0
    return luhn_valid(s) and luhn_valid(s[:9])


def _nir_number(body13: str) -> Optional[int]:
    b = body13.upper()
    if len(b) != 13:
        return None
    if b[5:7] == "2A":
        b = b[:5] + "19" + b[7:]
    elif b[5:7] == "2B":
        b = b[:5] + "18" + b[7:]
    if not b.isdigit():
        return None
    return int(b)


def nir_key(body13: str) -> int:
    """NIR key: 97 - (n mod 97), with Corsica 2A -> 19 and 2B -> 18."""
    n = _nir_number(body13)
    if n is None:
        raise ValueError("NIR body must be 13 characters")
    return 97 - n % 97


def nir_valid(value: str) -> bool:
    """French social security number (NIR): 13 characters + 2-digit key, plausible month."""
    s = re.sub(r"[\s.\-\u00a0\u202f]", "", value).upper()
    if len(s) != 15 or s[0] not in "123478" or not s[13:].isdigit():
        return False
    month = s[3:5]
    if not month.isdigit():
        return False
    m = int(month)
    if not (1 <= m <= 12 or 20 <= m <= 42 or 50 <= m <= 99):
        return False
    n = _nir_number(s[:13])
    if n is None:
        return False
    return 97 - n % 97 == int(s[13:])


# ISO 13616 registry lengths (country -> total IBAN length).
IBAN_LENGTHS = {
    "AD": 24, "AE": 23, "AL": 28, "AT": 20, "AZ": 28, "BA": 20, "BE": 16, "BG": 22, "BH": 22,
    "BI": 27, "BR": 29, "BY": 28, "CH": 21, "CR": 22, "CY": 28, "CZ": 24, "DE": 22, "DJ": 27,
    "DK": 18, "DO": 28, "EE": 20, "EG": 29, "ES": 24, "FI": 18, "FK": 18, "FO": 18, "FR": 27,
    "GB": 22, "GE": 22, "GI": 23, "GL": 18, "GR": 27, "GT": 28, "HN": 28, "HR": 21, "HU": 28,
    "IE": 22, "IL": 23, "IQ": 23, "IS": 26, "IT": 27, "JO": 30, "KW": 30, "KZ": 20, "LB": 28,
    "LC": 32, "LI": 21, "LT": 20, "LU": 20, "LV": 21, "LY": 25, "MC": 27, "MD": 24, "ME": 22,
    "MK": 19, "MN": 20, "MR": 27, "MT": 31, "MU": 30, "NI": 28, "NL": 18, "NO": 15, "OM": 23,
    "PK": 24, "PL": 28, "PS": 29, "PT": 25, "QA": 29, "RO": 24, "RS": 22, "RU": 33, "SA": 24,
    "SC": 31, "SD": 18, "SE": 24, "SI": 19, "SK": 24, "SM": 27, "SO": 23, "ST": 25, "SV": 28,
    "TL": 23, "TN": 24, "TR": 26, "UA": 29, "VA": 22, "VG": 24, "XK": 20, "YE": 30,
    # French overseas territories use the French format.
    "GF": 27, "GP": 27, "MQ": 27, "RE": 27, "PF": 27, "TF": 27, "YT": 27, "NC": 27, "PM": 27,
    "WF": 27, "BL": 27, "MF": 27,
}


def _alnum_to_int_string(s: str) -> str:
    return "".join(str(int(c, 36)) for c in s)


def iban_valid(value: str) -> bool:
    """IBAN: ISO 13616 mod 97 == 1 and the registry length of the country (when known)."""
    s = re.sub(r"[\s\u00a0\u202f\-]", "", value).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", s):
        return False
    expected = IBAN_LENGTHS.get(s[:2])
    if expected is not None and len(s) != expected:
        return False
    return int(_alnum_to_int_string(s[4:] + s[:4])) % 97 == 1


def iban_check_digits(country: str, bban: str) -> str:
    """Compute the two check digits for ``country`` + ``bban`` (used to build test data)."""
    n = int(_alnum_to_int_string((bban + country + "00").upper()))
    return f"{98 - n % 97:02d}"


def vat_fr_key(siren: str) -> int:
    """French VAT key: (12 + 3 * (SIREN mod 97)) mod 97."""
    return (12 + 3 * (int(siren) % 97)) % 97


def vat_fr_valid(value: str) -> bool:
    """French VAT number FR + 2-character key + SIREN. Numeric keys are checked with the formula;
    the newer alphanumeric keys are accepted when the SIREN passes Luhn."""
    s = re.sub(r"[\s\u00a0\u202f.\-]", "", value).upper()
    m = re.fullmatch(r"FR([0-9A-HJ-NP-Z]{2})(\d{9})", s)
    if not m:
        return False
    key, siren = m.groups()
    if key.isdigit():
        return int(key) == vat_fr_key(siren)
    return siren_valid(siren)


def isin_valid(value: str) -> bool:
    """ISIN: 2 letters + 9 alphanumerics + check digit (Luhn on letters expanded to numbers)."""
    s = value.upper()
    if not re.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}\d", s):
        return False
    return luhn_valid(_alnum_to_int_string(s))


def lei_valid(value: str) -> bool:
    """LEI (ISO 17442): 20 alphanumerics, mod 97 == 1."""
    s = value.upper()
    if not re.fullmatch(r"[A-Z0-9]{18}\d{2}", s):
        return False
    return int(_alnum_to_int_string(s)) % 97 == 1


def imo_valid(value: str) -> bool:
    """IMO ship number: 7 digits, the last one is sum(digit * weight 7..2) mod 10."""
    s = re.sub(r"\D", "", value)
    if len(s) != 7:
        return False
    return sum(int(c) * w for c, w in zip(s[:6], range(7, 1, -1))) % 10 == int(s[6])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SP = r"[ \u00a0\u202f]"  # one space-like separator
_SEPC = r"[ .\-\u00a0\u202f]"  # separator inside numbers
_UP = "A-ZÀ-ÖØ-ÞŒŸ"
_LO = "a-zß-öø-ÿœ"


def _before(text: str, start: int, n: int = 60) -> str:
    """Folded window of ``n`` characters before ``start``."""
    return fold_word(text[max(0, start - n):start]) if start else ""


def _before_raw(text: str, start: int, n: int = 60) -> str:
    return text[max(0, start - n):start]


def _scan_at_stems(rx: "re.Pattern[str]", text: str, stems: tuple[str, ...], cased: tuple[str, ...] = ()
                   ) -> Iterator["re.Match[str]"]:
    """``rx.finditer(text)`` for a pattern that starts with one of a few trigger words, run only where a stem of
    those words occurs: a C-speed substring search finds the stems in the lower-cased text (``cased`` stems in the
    text as written), then the pattern is matched at the start of each word holding a stem. Same left-to-right,
    non-overlapping matches as ``finditer`` (falls back to it when lower-casing changes the length)."""
    low = text.lower()
    if len(low) != len(text):
        yield from rx.finditer(text)
        return
    starts: set[int] = set()
    for hay, group in ((low, stems), (text, cased)):
        for stem in group:
            i = hay.find(stem)
            while i >= 0:
                j = i
                while j > 0 and (hay[j - 1].isalnum() or hay[j - 1] in "-/"):
                    j -= 1
                starts.add(j)
                if j != i:
                    starts.add(i)
                i = hay.find(stem, i + 1)
    last = -1
    for p in sorted(starts):
        if p < last:
            continue
        m = rx.match(text, p)
        if m:
            last = m.end() if m.end() > p else p + 1
            yield m


class IntervalSet:
    """Non-overlapping half-open intervals with fast overlap checks."""

    def __init__(self) -> None:
        self._starts: list[int] = []
        self._ends: list[int] = []

    def overlaps(self, s: int, e: int) -> bool:
        i = bisect.bisect_right(self._starts, s)
        if i > 0 and self._ends[i - 1] > s:
            return True
        if i < len(self._starts) and self._starts[i] < e:
            return True
        return False

    def add(self, s: int, e: int) -> None:
        i = bisect.bisect_right(self._starts, s)
        self._starts.insert(i, s)
        self._ends.insert(i, e)


# ---------------------------------------------------------------------------
# EMAIL
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(
    r"(?<![\w.+\-%'])[\w.+\-%']{1,64}@(?:[\w\-]{1,63}\.)+[A-Za-z]{2,24}(?![\w\-])"
)


def _find_email(text: str) -> Iterator[Candidate]:
    for m in _EMAIL_RE.finditer(text):
        s, e = m.start(), m.end()
        local = m.group().split("@", 1)[0]
        while local and local[0] in ".'-":  # leading punctuation is not part of the address
            local = local[1:]
            s += 1
        if not local:
            continue
        yield Candidate(s, e, "EMAIL", True, PRIORITY["EMAIL"], "email")


def email_domains(text: str) -> list[str]:
    """Distinct domains of the e-mail addresses in ``text`` (lower case, in order of first appearance). The
    companies they name are often written in the same text: ``names.refine_rule_spans`` matches them (v0.3)."""
    out: list[str] = []
    seen: set[str] = set()
    for m in _EMAIL_RE.finditer(text):
        domain = m.group().rsplit("@", 1)[1].lower()
        if domain not in seen:
            seen.add(domain)
            out.append(domain)
    return out


# ---------------------------------------------------------------------------
# PHONE
# ---------------------------------------------------------------------------

_PHONE_FR_RE = re.compile(
    r"(?<![\w+])(?<!\d[ .\-])"
    r"(?:(?:\+|00)33" + _SEPC + r"?(?:\(0\)" + _SEPC + r"?)?|0)"
    r"[1-9](?:" + _SEPC + r"?\d{2}){4}"
    r"(?!" + _SEPC + r"?\d)(?!\w)"
)
_PHONE_INTL_RE = re.compile(
    r"(?<![\w+])\+(?!33)[1-9]\d{0,2}(?:" + _SEPC + r"?\(?\d{1,4}\)?){2,6}(?!" + _SEPC + r"?\d)(?!\w)"
)


def _find_phone(text: str) -> Iterator[Candidate]:
    for m in _PHONE_FR_RE.finditer(text):
        yield Candidate(m.start(), m.end(), "PHONE", True, PRIORITY["PHONE"], "phone_fr")
    for m in _PHONE_INTL_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if 8 <= len(digits) <= 15:
            yield Candidate(m.start(), m.end(), "PHONE", True, PRIORITY["PHONE"], "phone_intl")


# ---------------------------------------------------------------------------
# IBAN
# ---------------------------------------------------------------------------

_IBAN_START_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{2}\d{2}(?=[ \u00a0\u202f\-]?[A-Za-z0-9])")
_IBAN_SEPS = " \u00a0\u202f-"


def _find_iban(text: str) -> Iterator[Candidate]:
    n = len(text)
    for m in _IBAN_START_RE.finditer(text):
        s = m.start()
        lower = text[s:s + 2].islower()  # "fr76 3004 ..." typed in lower case: the checksum decides
        if not lower and not text[s:s + 2].isupper():
            continue
        # Walk forward collecting alphanumerics, allowing single separators between them.
        chars: list[tuple[str, int]] = [(text[s + k].upper(), s + k) for k in range(4)]
        p = s + 4

        def ok(c: str) -> bool:
            return c.isdigit() or ("A" <= c <= "Z") or (lower and "a" <= c <= "z")

        while p < n and len(chars) < 34:
            ch = text[p]
            if ch in _IBAN_SEPS and p + 1 < n and ok(text[p + 1]):
                p += 1
                ch = text[p]
            if ok(ch):
                chars.append((ch.upper(), p))
                p += 1
            else:
                break
        country = text[s:s + 2].upper()
        expected = IBAN_LENGTHS.get(country)
        lengths = [expected] if expected else list(range(min(len(chars), 34), 14, -1))
        found = False
        for length in lengths:
            if length is None or len(chars) < length:
                continue
            end = chars[length - 1][1] + 1
            nxt = text[end] if end < n else ""
            if nxt and (nxt.isalnum() or nxt == "_"):
                continue
            value = "".join(c for c, _ in chars[:length])
            if iban_valid(value):
                yield Candidate(s, end, "IBAN", True, PRIORITY["IBAN"], "iban")
                bic = _bic_after_iban(text, end)
                if bic is not None:
                    yield bic
                found = True
                break
            if expected and not lower:
                # Exact national length, mostly digits: an IBAN with a typo, masked all the same (R1, CL-11).
                bban = value[4:]
                shape = sum(c.isdigit() for c in bban) * 2 >= len(bban)
                yield Candidate(s, end, "IBAN", shape, PRIORITY["IBAN"], "iban", checked=False)
                found = True
        if lower:
            continue  # a lower-case look-alike with a wrong checksum is ordinary text, not a suspect
        if not found and expected and len(chars) >= 15:
            # Wrong length for the country: report as suspect only if it looks like an IBAN.
            end = chars[-1][1] + 1
            if len(chars) >= expected - 4:
                yield Candidate(s, end, "IBAN", False, PRIORITY["IBAN"], "iban", checked=False)


# ---------------------------------------------------------------------------
# NIR (French social security number)
# ---------------------------------------------------------------------------

_SEPN = r"[ .\u00a0\u202f]?"
_NIR_RE = re.compile(
    r"(?<![\dA-Za-z])(?<!\d[ .])"
    r"[1-478](?:" + _SEPN + r"\d){4}" + _SEPN + r"(?:2[ABab]|\d" + _SEPN + r"\d)"
    r"(?:" + _SEPN + r"\d){6}(?:[ .\-\u00a0\u202f]?|[ \u00a0]?[/\-][ \u00a0]?|[ \u00a0](?i:cl[ée])[ \u00a0:]{1,2})\d\d"
    r"(?![\dA-Za-z])(?!" + _SEPC + r"\d)"
)


def _find_nir(text: str) -> Iterator[Candidate]:
    for m in _NIR_RE.finditer(text):
        raw = m.group()
        compact = re.sub(r"(?i)[ .\-/:\u00a0\u202f]|cl[ée]", "", raw)
        if len(compact) != 15:
            continue
        month = compact[3:5]
        plausible = month.isdigit() and (1 <= int(month) <= 12 or 20 <= int(month) <= 42 or 50 <= int(month) <= 99)
        if not plausible:
            continue
        yield Candidate(m.start(), m.end(), "NIR", nir_valid(compact), PRIORITY["NIR"], "nir")


# ---------------------------------------------------------------------------
# SIREN, SIRET, VAT
# ---------------------------------------------------------------------------

_SIREN_RE = re.compile(
    r"(?<![\dA-Za-z])(?<!\d[ .\u00a0])"
    r"\d{3}" + _SEPN + r"\d{3}" + _SEPN + r"\d{3}"
    r"(?!" + _SEPN + r"\d)(?![A-Za-z])"
)
_SIRET_RE = re.compile(
    r"(?<![\dA-Za-z])(?<!\d[ .\u00a0])"
    r"(?:\d{3}" + _SEPN + r"\d{3}" + _SEPN + r"\d{3}" + _SEPN + r"\d{5}"
    r"|\d{3}" + _SEPN + r"\d{3}" + _SEPN + r"\d{3}" + _SEPN + r"\d{3}" + _SEPN + r"\d{2})"
    r"(?!" + _SEPN + r"\d)(?![A-Za-z])"
)
_AMOUNT_AFTER_RE = re.compile(
    r"^\s?(?:,\d{1,2}\b)?\s?(?:€|eur\b|euros?\b|k€|m€|\$|usd\b|dollars?\b|chf\b|£|gbp\b|fcfa\b|ht\b|ttc\b|%)",
    re.IGNORECASE,
)
_AMOUNT_BEFORE_RE = re.compile(
    r"(?:€|\$|£|eur|euros|montant|somme|prix|total|capital(?: social)?|valeur|tarif|cout|cours)"
    r"(?:\s+(?:de|d'un|d'une|total|global|brut|net|hors taxes|ttc|ht))*\s*:?\s*$"
)


def _is_amount(text: str, s: int, e: int) -> bool:
    if _AMOUNT_AFTER_RE.match(text[e:e + 12]):
        return True
    return bool(_AMOUNT_BEFORE_RE.search(_before(text, s, 40)))


_EUID_RE = re.compile(r"(?<![A-Za-z0-9])[A-Z]{2}\d{4}\.(?P<siren>\d{9})(?![0-9])")


def _find_siren(text: str) -> Iterator[Candidate]:
    for m in _EUID_RE.finditer(text):  # European unique identifier: country + register code + "." + SIREN
        if luhn_valid(m.group("siren")):
            yield Candidate(m.start("siren"), m.end("siren"), "SIREN", True, PRIORITY["SIREN"] + 1, "euid")
    for m in _SIREN_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if digits == "000000000" or _is_amount(text, m.start(), m.end()):
            continue
        ok = luhn_valid(digits)
        # A wrong Luhn key with a SIREN / SIRET / RCS label on the line is still masked (R1, CL-11).
        mask = ok or _company_label_on_line(text, m.start())
        yield Candidate(m.start(), m.end(), "SIREN", mask, PRIORITY["SIREN"], "siren", checked=ok)


def _find_siret(text: str) -> Iterator[Candidate]:
    for m in _SIRET_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if _is_amount(text, m.start(), m.end()):
            continue
        ok = siret_valid(digits)
        mask = ok or _company_label_on_line(text, m.start())
        yield Candidate(m.start(), m.end(), "SIRET", mask, PRIORITY["SIRET"], "siret", checked=ok)


def _company_label_on_line(text: str, start: int) -> bool:
    """A SIREN, SIRET, RCS or registration label in the 60 characters before ``start``, on the same line; when the
    number starts its line (form layout), the label may end the previous line (v0.2, Max class G)."""
    ls = text.rfind("\n", 0, start) + 1
    before = text[max(ls, start - 60):start]
    if _COMPANY_ID_CTX.search(fold_word(before)):
        return True
    if before.strip() or ls == 0:
        return False
    pls = text.rfind("\n", 0, ls - 1) + 1
    return bool(_COMPANY_ID_CTX.search(fold_word(text[max(pls, ls - 1 - 60):ls - 1])))


_VAT_FR_RE = re.compile(
    r"(?<![A-Za-z0-9])FR" + _SP + r"?[0-9A-HJ-NP-Z]{2}" + _SP + r"?\d{3}" + _SEPN + r"\d{3}" + _SEPN + r"\d{3}(?![ .]?\d)(?![A-Za-z])"
)
_VAT_EU_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:BE" + _SP + r"?0?\d{3}[ .]?\d{3}[ .]?\d{3}|DE" + _SP + r"?\d{9}|LU" + _SP + r"?\d{8}"
    r"|IT" + _SP + r"?\d{11}|ES" + _SP + r"?[A-Z0-9]\d{7}[A-Z0-9]|NL" + _SP + r"?\d{9}B\d{2}|PT" + _SP + r"?\d{9}"
    r"|ATU\d{8}|CHE[-. ]?\d{3}[.\-]?\d{3}[.\-]?\d{3}|IE" + _SP + r"?\d{7}[A-Z]{1,2})(?![A-Za-z0-9])"
)
_VAT_KEYWORDS = re.compile(r"(?:tva|vat|ust|iva|btw|intracommunautaire|intracom)\b[^\n]{0,30}$")


def _find_vat(text: str) -> Iterator[Candidate]:
    for m in _VAT_FR_RE.finditer(text):
        yield Candidate(m.start(), m.end(), "VAT", vat_fr_valid(m.group()), PRIORITY["VAT"], "vat_fr")
    for m in _VAT_EU_RE.finditer(text):
        if not _VAT_KEYWORDS.search(_before(text, m.start(), 50)):
            continue
        raw = re.sub(r"[\s.\-]", "", m.group()).upper()
        valid = True
        if raw.startswith("BE"):
            num = raw[2:].rjust(10, "0")
            valid = num.isdigit() and 97 - int(num[:8]) % 97 == int(num[8:])
        yield Candidate(m.start(), m.end(), "VAT", valid, PRIORITY["VAT"], "vat_eu")


# ---------------------------------------------------------------------------
# CARD
# ---------------------------------------------------------------------------

_CARD_RE = re.compile(
    r"(?<![\dA-Za-z])(?<!\d[ \-])"
    r"(?:\d{4}(?:[ \-\u00a0]?\d{4}){2}[ \-\u00a0]?\d{1,7}"  # 4-4-4-x (13-19 digits)
    r"|\d{4}[ \-\u00a0]?\d{6}[ \-\u00a0]?\d{4,5}"  # 4-6-5 (Amex, Diners)
    r"|\d{13,19})"
    r"(?![ \-]?\d)(?![A-Za-z])"
)


def _find_card(text: str) -> Iterator[Candidate]:
    for m in _CARD_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if not 13 <= len(digits) <= 19 or digits[0] not in "23456":
            continue
        valid = luhn_valid(digits)
        if not valid and not re.fullmatch(r"\d{4}([ \-])\d{4}\1\d{4}\1\d{4}", m.group()):
            continue  # an invalid number is a suspect only in the classic 4-4-4-4 layout
        yield Candidate(m.start(), m.end(), "CARD", valid, PRIORITY["CARD"], "card")


# ---------------------------------------------------------------------------
# CASE_REF: RG, pourvoi, Portalis, ECLI, docket numbers, file references
# ---------------------------------------------------------------------------

_RG_RE = re.compile(
    r"(?:\b[Nn][°ºo]\.?\s*(?:de\s+)?(?:RG|R\.G\.?)|\b(?:RG|R\.G\.?)(?:\s*(?:[Nn][°ºo]\.?|num[ée]ro))?"
    r"|\b[Rr][ée]pertoire\s+[Gg][ée]n[ée]ral(?:\s*[Nn][°ºo]\.?)?)"
    r"\s*:?\s*(?P<num>\d{2}\s?/\s?\d{3,6}|\d{4}\s?/\s?\d{3,6}|(?:19|20)\d{2}\s?[A-Z]\s?\d{4,6})(?![\d/])"
)
# Tribunal de commerce roll numbers ("2024J00817", "2023F01234"): distinctive enough without a keyword.
_RG_TC_RE = re.compile(r"(?<![A-Za-z0-9])(?P<num>(?:19|20)\d{2}[A-Z]\d{5})(?![A-Za-z0-9])")
_POURVOI_NODOT_RE = re.compile(r"(?i:pourvoi)\s+(?:n[°ºo]\.?\s*)?(?P<num>(?:[A-Z]\s)?\d{2}-\d{5})(?![\d.])")
_POURVOI_RE = re.compile(r"(?<![\w\-.])(?P<num>(?:[A-Z]\s)?\d{2}-\d{2}\.\d{3})(?![\d.]\d)")
_POURVOI_CTX = re.compile(
    r"(?:pourvoi|cass\b|cassation|civ\b|civ\.|soc\b|soc\.|com\b|com\.|crim\b|crim\.|ch\.?\s*mixte|ass\.?\s*plen|"
    r"chambre (?:sociale|civile|commerciale|criminelle)|arret|n[°o]\.?\s*$)"
)
_PORTALIS_RE = re.compile(
    r"(?:\bPortalis\s*:?\s*)?(?P<num>(?<![A-Z0-9\-])[A-Z0-9]{4}-[A-Z]-[A-Z0-9]{3}-[A-Z0-9]{4,5}(?![A-Z0-9\-]))"
)
_ECLI_RE = re.compile(r"\bECLI:[A-Z]{2}:[A-Z0-9.]{1,7}:\d{4}:[A-Za-z0-9.]*[A-Za-z0-9]")
_DOCKET_RE = re.compile(r"\b[Nn][°ºo]\.?\s*(?P<num>\d{2,4}\s?/\s?\d{3,6})(?![\d/])")
_DOCKET_CTX = re.compile(
    r"(?:tribunal|cour|conseil|chambre|juridiction|jugement|arret|ordonnance|instance|affaire|"
    r"greffe|role|minute|refere|requete|prud'?hommes|\btj\b|\bcph\b|\bta\b|\bcaa\b|\bca\b|\btc\b|"
    r"procedure|recours|appel|pourvoi|plainte|parquet|enquete|instruction|\bpv\b)"
)
_LAW_REF_BEFORE = re.compile(
    r"(?:reglement|directive|loi|decret|ordonnance n|arrete|deliberation|circulaire|convention)"
    r"\s*(?:\((?:ue|ce|cee)\)\s*)?(?:n[°o]\.?)?\s*$"
)
_ADMIN_REQ_RE = re.compile(r"\b(?:[Rr]equ[êe]te|[Rr]eq\.|[Dd]ossier)\s*[Nn][°ºo]\.?\s*(?P<num>\d{6,8})(?!\d)")
_FILE_REF_RE = re.compile(
    r"\b(?:[Nn]os?\s+r[ée]f(?:[ée]rences?)?\.?|[Vv]os?\s+r[ée]f(?:[ée]rences?)?\.?|[Nn]otre\s+r[ée]f[ée]rence|"
    r"[Vv]otre\s+r[ée]f[ée]rence|[Rr][ée]f\.?\s+dossier|[Dd]ossier\s+n[°ºo]\.?)\s*:?\s*"
    r"(?P<num>[A-Z0-9][A-Za-z0-9]*(?:[/\-.][A-Za-z0-9]+){0,4})"
)


def _find_case_ref(text: str) -> Iterator[Candidate]:
    pr = PRIORITY["CASE_REF"]
    for m in _RG_RE.finditer(text):
        yield Candidate(m.start("num"), m.end("num"), "CASE_REF", True, pr, "rg")
    for m in _RG_TC_RE.finditer(text):
        yield Candidate(m.start("num"), m.end("num"), "CASE_REF", True, pr, "rg_tc")
    for m in _POURVOI_RE.finditer(text):
        if _POURVOI_CTX.search(_before(text, m.start(), 120)):
            yield Candidate(m.start("num"), m.end("num"), "CASE_REF", True, pr, "pourvoi")
    for m in _POURVOI_NODOT_RE.finditer(text):
        yield Candidate(m.start("num"), m.end("num"), "CASE_REF", True, pr, "pourvoi")
    for m in _PORTALIS_RE.finditer(text):
        num = m.group("num")
        has_kw = m.group().lower().startswith("portalis")
        if has_kw or re.fullmatch(r"[A-Z0-9]{4}-[A-Z]-B\d[A-Z]-[A-Z0-9]{4,5}", num):
            yield Candidate(m.start("num"), m.end("num"), "CASE_REF", True, pr, "portalis")
    for m in _ECLI_RE.finditer(text):
        yield Candidate(m.start(), m.end(), "CASE_REF", True, pr, "ecli")
    for m in _DOCKET_RE.finditer(text):
        before = _before(text, m.start(), 90)
        if _LAW_REF_BEFORE.search(before):
            continue
        ctx = before + " " + fold_word(text[m.end():m.end() + 40])
        if _DOCKET_CTX.search(ctx):
            yield Candidate(m.start("num"), m.end("num"), "CASE_REF", True, pr, "docket")
    for m in _ADMIN_REQ_RE.finditer(text):
        yield Candidate(m.start("num"), m.end("num"), "CASE_REF", True, pr, "requete")
    for m in _FILE_REF_RE.finditer(text):
        num = m.group("num")
        if re.fullmatch(r"(?:19|20)\d{2}", num):
            continue  # "dossier n° 2024. Il ...": a year is not a reference (v0.2, Max class E)
        if num.isdigit() and len(num) <= 4 and re.match(r"\.[ \u00a0\n]+[A-ZÀ-Þ]", text[m.end("num"):m.end("num") + 3]):
            continue  # a short number that ends a sentence
        if sum(c.isdigit() for c in num) >= 3 and len(num) >= 4:
            yield Candidate(m.start("num"), m.end("num"), "CASE_REF", True, pr - 1, "file_ref")
    yield from _find_section_case_ref(text)


# Court roll numbers with a section letter and no "n°" (labour courts and others): "F 24/01234", "RG R 23/05678".
# The narrow shape (one capital letter, a space, a 2-digit year, "/", 5 digits) needs no keyword; the looser
# shapes (no space, "-", 4 or 6 digits, or no letter at all) need a court or roll word on the line.
_SECTION_REF_RE = re.compile(
    r"(?<![\w/\-.])(?P<num>(?:(?P<l>[A-Z])(?P<sp>[ \u00a0]?))?\d{2}[ \u00a0]?(?P<sep>[/\-])[ \u00a0]?(?P<n>\d{4,6}))"
    r"(?![\w/]|[\-.,]\d)"
)
# The number part alone: found first (a pattern that starts with a digit fails fast elsewhere), then the full
# pattern is matched from the possible letter before it.
_SECTION_CORE_RE = re.compile(r"\d{2}[ \u00a0]?[/\-][ \u00a0]?\d{4,6}(?![\w/]|[\-.,]\d)")


def _section_ref_matches(text: str) -> Iterator["re.Match[str]"]:
    for c in _SECTION_CORE_RE.finditer(text):
        s = c.start()
        for k in (2, 1, 0):  # "F 24/01234", "F24/01234", "24/01234"
            if s - k < 0:
                continue
            m = _SECTION_REF_RE.match(text, s - k)
            if m and m.end() == c.end():
                yield m
                break


def _find_section_case_ref(text: str) -> Iterator[Candidate]:
    pr = PRIORITY["CASE_REF"]
    for m in _section_ref_matches(text):
        letter, narrow = m.group("l"), False
        if letter and m.group("sp") and m.group("sep") == "/" and len(m.group("n")) == 5:
            narrow = True
        if not narrow:
            if not letter and (m.group("sep") != "/" or len(m.group("n")) != 5):
                continue  # a bare "24-1234" or "24/123456" is too common without a letter
            ls = text.rfind("\n", 0, m.start()) + 1
            le = text.find("\n", m.end())
            le = len(text) if le < 0 else le
            before = fold_word(text[max(ls, m.start() - 80):m.start()])
            if _LAW_REF_BEFORE.search(before):
                continue
            ctx = before + " " + fold_word(text[m.end():min(le, m.end() + 40)])
            if not (_DOCKET_CTX.search(ctx) or re.search(r"\brg\b", ctx)):
                continue
        yield Candidate(m.start("num"), m.end("num"), "CASE_REF", True, pr, "section_ref")


# ---------------------------------------------------------------------------
# ID_CODE: ISIN, LEI, plates, IP addresses, keyword-gated ID numbers
# ---------------------------------------------------------------------------

_ISIN_RE = re.compile(r"(?<![A-Za-z0-9])[A-Z]{2}[A-Z0-9]{9}\d(?![A-Za-z0-9])")
_LEI_RE = re.compile(r"(?<![A-Za-z0-9])[A-Z0-9]{18}\d{2}(?![A-Za-z0-9])")
_PLATE_LETTERS = "A-HJ-NP-TV-Z"
# SIV plate "AB-123-CD"; a department suffix may follow ("AB-123-CD-75", not part of the plate).
_PLATE_RE = re.compile(
    rf"(?<![A-Za-z0-9\-])(?P<p>[{_PLATE_LETTERS}]{{2}})(?P<s1>[- ]?)\d{{3}}(?P<s2>[- ]?)[{_PLATE_LETTERS}]{{2}}"
    rf"(?![A-Za-z0-9])(?!-(?!\d{{2,3}}(?![A-Za-z0-9\-])))"
)
_PLATE_OLD_RE = re.compile(r"(?<![A-Za-z0-9\-])\d{1,4}[ \-][A-Z]{1,3}[ \-](?:\d{2}|2[AB]|97\d)(?![A-Za-z0-9])")
_PLATE_CTX = re.compile(r"(?:immatricul|plaque|vehicule|voiture|camion|scooter|moto|fourgon)[^\n]{0,40}$")
_IPV4_RE = re.compile(r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?![\d.]\d|\d)")
_IPV6_RE = re.compile(r"(?<![0-9A-Fa-f:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![0-9A-Fa-f:])")
_IP_CTX = re.compile(r"(?:\bip\b|adresse ip|ipv4|ipv6|serveur|connexion|log|hote|host)[^\n]{0,40}$")
_NOT_IP_CTX = re.compile(r"(?:article|art\.|section|point|paragraphe|§|version|v\.)\s*$")
_ID_KEYWORD_RE = re.compile(
    r"(?i)(?:passeport|carte\s+nationale\s+d['’]identit[ée]|carte\s+d['’]identit[ée]|(?-i:\bCNI\b)|"
    r"pi[èe]ce\s+d['’]identit[ée]|titre\s+de\s+s[ée]jour|permis\s+de\s+conduire|permis\s+n[°ºo]|num[ée]ro\s+fiscal|"
    r"identifiant\s+fiscal|(?-i:\bSPI\b)|n[°o]\s+fiscal|num[ée]ro\s+d['’]allocataire|"
    r"num[ée]ro\s+(?:de\s+)?client|num[ée]ro\s+de\s+contrat|num[ée]ro\s+de\s+police|"
    r"police\s+d['’]assurance|num[ée]ro\s+(?:d['’])?adh[ée]rent)"
)
_ID_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])(?=[A-Z0-9 ]*\d)[A-Z0-9](?:[A-Z0-9]|[ ](?=[A-Z0-9]{2})){6,17}[A-Z0-9](?![A-Za-z0-9])")


# Kbis and court identifiers that need their keyword: the Kbis "code de vérification" (anyone holding it can
# download the extract), the RCS "numéro de gestion" and the lawyer's "toque" (set RAGBASE_SHIELD_MASK_TOQUE=0
# to keep toque numbers visible).
_VERIF_RE = re.compile(r"(?i)code\s+de\s+v[ée]rification\s*:?\s*(?P<v>(?-i:[A-Za-z0-9]{8,12}))(?![A-Za-z0-9])")
_GESTION_RE = re.compile(r"(?i)(?:num[ée]ro|n[°ºo]\.?)\s+de\s+gestion\s*:?\s*(?P<v>\d{4}\s?[A-Z]\s?\d{3,5})(?![0-9])")
_TOQUE_RE = re.compile(
    r"(?:(?i:\btoque)\s*(?:n[°ºo]\.?\s*)?:?\s*|\(\s*T\.?\s?)(?P<v>[A-Z]?[ \u00a0]?\d{1,5})(?![0-9])"
)


def _mask_toque() -> bool:
    import os

    return os.environ.get("RAGBASE_SHIELD_MASK_TOQUE", "1").strip().lower() not in ("0", "false", "no", "off")


def _find_id_code(text: str) -> Iterator[Candidate]:
    pr = PRIORITY["ID_CODE"]
    for rx in (_VERIF_RE, _GESTION_RE) + ((_TOQUE_RE,) if _mask_toque() else ()):
        for m in rx.finditer(text):
            v = m.group("v")
            if rx is _VERIF_RE and (not any(c.isdigit() for c in v) or not any(c.isalpha() for c in v)):
                continue
            if rx is _TOQUE_RE and re.match(r"\(\s*T\.?\d{1,2}\)", text[m.start():m.end() + 1]):
                continue  # "(T4)": a parenthesised short token (a quarter, a column), not a toque (v0.2, Max class C)
            yield Candidate(m.start("v"), m.end("v"), "ID_CODE", True, pr, "kbis_court")
    for m in _ISIN_RE.finditer(text):
        v = m.group()
        if sum(c.isdigit() for c in v) >= 3:
            yield Candidate(m.start(), m.end(), "ID_CODE", isin_valid(v), PRIORITY["ISIN"], "isin")
    for m in _LEI_RE.finditer(text):
        v = m.group()
        if sum(c.isdigit() for c in v) >= 4 and sum(c.isalpha() for c in v) >= 2:
            yield Candidate(m.start(), m.end(), "ID_CODE", lei_valid(v), PRIORITY["LEI"], "lei")
    for m in _PLATE_RE.finditer(text):
        s1, s2 = m.group("s1"), m.group("s2")
        if m.group("p") == "SS":
            continue
        if (s1 == "-" and s2 == "-") or (s1 and s2 and "-" in (s1, s2)):
            # both separators written, at least one of them a hyphen ("AB-123-CD", "AB 123-CD"): no context needed
            yield Candidate(m.start(), m.end(), "ID_CODE", True, pr, "plate")
        elif _PLATE_CTX.search(_before(text, m.start(), 60)):
            yield Candidate(m.start(), m.end(), "ID_CODE", True, pr, "plate")
    for m in _PLATE_OLD_RE.finditer(text):
        if _PLATE_CTX.search(_before(text, m.start(), 60)):
            yield Candidate(m.start(), m.end(), "ID_CODE", True, pr, "plate_old")
    for m in _IPV4_RE.finditer(text):
        before = _before(text, m.start(), 50)
        if _NOT_IP_CTX.search(before):
            continue
        octets = [int(x) for x in m.group().split(".")]
        if all(o < 10 for o in octets) and not _IP_CTX.search(before):
            continue  # "1.2.3.4" is more likely a section number
        yield Candidate(m.start(), m.end(), "ID_CODE", True, pr, "ipv4")
    for m in _IPV6_RE.finditer(text):
        v = m.group()
        if v.count(":") < 2 or len(v) < 6 or not re.search(r"[0-9A-Fa-f]{3}", v):
            continue
        try:
            ipaddress.IPv6Address(v)
        except ValueError:
            continue
        yield Candidate(m.start(), m.end(), "ID_CODE", True, pr, "ipv6")
    for kw in _ID_KEYWORD_RE.finditer(text):
        for m in _ID_TOKEN_RE.finditer(text, kw.end(), min(len(text), kw.end() + 60)):
            compact = m.group().replace(" ", "")
            if sum(c.isdigit() for c in compact) < 4 or not 8 <= len(compact) <= 18:
                continue
            yield Candidate(m.start(), m.end(), "ID_CODE", True, pr - 1, "id_keyword")
            break
    yield from _find_labelled_refs(text)
    yield from _find_bank_and_postal_ids(text)
    yield from _find_cadastre(text)


# ---------------------------------------------------------------------------
# Labelled references with no check digit (R1, CL-13)
# ---------------------------------------------------------------------------

# A closed list of labels that announce a reference number. Procedure-like labels give CASE_REF, the others
# ID_CODE. "décision", "délibération", "loi" are not labels (public decision and law numbers stay clear).
_REF_LABEL_RE = re.compile(
    r"(?i)(?<![\w/])(?:(?P<case>proc[ée]dure|enqu[êe]te|parquet|plainte|proc[èe]s[- ]verbal|(?-i:PV)|"
    r"[VN][ \u00a0]?/[ \u00a0]?r[ée]f(?:[ée]rences?)?\.?|r[ée]f[ée]rences?|r[ée]f\.?|dossier)"
    r"|(?P<id>matricule|identifiant|allocataire|adh[ée]rent|cotisant|pension|retraite|contrat|"
    r"lettre[ \u00a0]recommand[ée]e|recommand[ée]e?|(?-i:LRAR|LR/AR|LR-AR)|avis[ \u00a0]de[ \u00a0]r[ée]ception|"
    r"suivi|titre[ \u00a0]de[ \u00a0]perception|sinistre|num[ée]ro[ \u00a0]fiscal|avis[ \u00a0]d['’]imposition))"
    r"(?![\w])"
)
# Between the label and the value: number markers, connector words, and (only before a number marker) up to
# three other words ("contrat de prêt n°", "lettre recommandée avec accusé de réception n°").
_GAP_MARK_RE = re.compile(r"(?i)(?:n[°º]s?\.?|nº|no\.|no(?=[ \u00a0]*\d)|num[ée]ro|num\.|#|:|=|\||[\-–](?=[ \u00a0]))")
_GAP_WORD_RE = re.compile(r"[^\W\d_][^\W\d_'’\-]*(?:['’](?=[^\W\d_]))?(?:[\-][^\W\d_]+)*['’]?")
_GAP_CONNECTORS = frozenset({
    "de", "du", "des", "d'", "d’", "la", "le", "l'", "l’", "les", "en", "cours", "interne", "client", "unique",
    "dossier", "suivi", "paiement", "virement", "mandat", "police", "gendarmerie", "preliminaire", "penale",
    "salarie", "agent", "employe", "affilie", "assure", "assuree", "sa", "son", "votre", "notre", "vos", "nos",
})
_GAP_STOP_WORDS = frozenset({
    "loi", "lois", "decret", "ordonnance", "reglement", "directive", "arrete", "article", "articles", "art",
    "decision", "deliberation", "circulaire", "code", "convention", "accord", "annexe", "page", "pages",
    "chapitre", "section", "alinea", "paragraphe", "point", "jugement", "arret",
})
# A value: an optional letter prefix (glued or joined by "-", "/", "."; or 1 to 3 letters and a space), then a
# group that starts with a digit, then more groups joined by "-", "/", "." or a single space before a digit group.
_REF_VALUE_RE = re.compile(
    r"(?:[A-Z]{1,6}[\-/.]?|(?P<pre>[A-Z]{1,3})(?P<psp>[ \u00a0])(?=\d))?\d[A-Za-z0-9]*"
    r"(?:(?:[\-/.]|[ \u00a0](?=[A-Z]{0,3}\d))[A-Za-z0-9]+)*(?![_])"
)
_GROUPED_NUMBER_RE = re.compile(r"\d{1,3}(?:[ \u00a0.]\d{3})+(?:,\d+)?")
_DATE_SHAPE_RE = re.compile(r"\d{1,2}[/.\-]\d{1,2}[/.\-](?:\d{4}|\d{2})|\d{4}[/.\-]\d{1,2}[/.\-]\d{1,2}")
_MONTH_AFTER_RE = re.compile(r"(?i)[ \u00a0]+" + r"(?:janvier|f[ée]vrier|mars|avril|mai|juin|juillet|ao[ûu]t|septembre|"
                             r"octobre|novembre|d[ée]cembre|january|february|march|april|may|june|july|august|"
                             r"october|december)\b")


def _ref_value_at(text: str, pos: int) -> Optional[tuple[int, int, bool]]:
    """(start, end, explicit) of the reference value that follows a label ending at ``pos``, or None."""
    n = len(text)
    limit = min(n, pos + 48)
    p = pos
    explicit = False
    pending_other = 0
    others = 0
    newline_ok = True
    while p < limit:
        q = p
        while q < n and text[q] in " \t\u00a0":
            q += 1
        if q < n and text[q] == "\n" and newline_ok and explicit and text[p - 1:p] in (":", " ", "\t"):
            # a form label "Matricule :" with the value on the next line
            newline_ok = False
            q += 1
            while q < n and text[q] in " \t\u00a0":
                q += 1
        if q >= n:
            return None
        v = _REF_VALUE_RE.match(text, q)
        if v and not pending_other:
            return q, v.end(), explicit
        mk = _GAP_MARK_RE.match(text, q)
        if mk and mk.end() > q:
            explicit = True
            pending_other = 0
            p = mk.end()
            continue
        w = _GAP_WORD_RE.match(text, q)
        if not w or (q == p and p != pos and text[p - 1] not in "'’"):
            return None
        low = fold_word(w.group())
        if low in _GAP_STOP_WORDS:
            return None
        if low.rstrip("'’") in _GAP_CONNECTORS or low in _GAP_CONNECTORS:
            p = w.end()
            continue
        others += 1
        if others > 3 or len(low) > 15:
            return None
        pending_other = 1
        p = w.end()
    return None


_REF_LABEL_STEMS = ("proc", "enqu", "parquet", "plainte", "verbal", "réf", "ref", "dossier", "matricule",
                    "identifiant", "allocataire", "adh", "cotisant", "pension", "retraite", "contrat", "recommand",
                    "lettre", "avis", "suivi", "titre", "sinistre", "num")
_REF_LABEL_CASED = ("PV", "LRAR", "LR/AR", "LR-AR")


def _find_labelled_refs(text: str) -> Iterator[Candidate]:
    for lab in _scan_at_stems(_REF_LABEL_RE, text, _REF_LABEL_STEMS, _REF_LABEL_CASED):
        got = _ref_value_at(text, lab.end())
        if got is None:
            continue
        s, e, explicit = got
        m = _REF_VALUE_RE.match(text, s)
        if m.group("psp") and m.group("pre") in ("RG", "N", "NO", "NR"):
            continue  # "dossier RG 24/01234": the roll number has its own detector
        value = text[s:e]
        # "réf. 4521 12 mars": a trailing day before a month name is not part of the reference
        parts = re.split(r"[ \u00a0]", value)
        if len(parts) > 1 and len(parts[-1]) <= 2 and parts[-1].isdigit() and _MONTH_AFTER_RE.match(text, e):
            e = s + len(value) - len(parts[-1]) - 1
            value = text[s:e]
            parts = parts[:-1]
        # "réf. 4521 2024. Il": a trailing year group before a sentence end is not part of the value (v0.2)
        if len(parts) > 1 and re.fullmatch(r"(?:19|20)\d{2}", parts[-1]) \
                and (re.match(r"\.[ \u00a0\n]+[A-ZÀ-Þ]", text[e:e + 3]) or not text[e:e + 3].strip(". \u00a0\n")):
            e = s + len(value) - len(parts[-1]) - 1
            value = text[s:e]
        digits = sum(c.isdigit() for c in value)
        if digits < 4 or len(value) > 40:
            continue
        if _DATE_SHAPE_RE.fullmatch(value) or re.fullmatch(r"(?:19|20)\d{2}", value) \
                or re.fullmatch(r"\d{1,2}[hH]\d{2}", value):
            continue
        if _GROUPED_NUMBER_RE.fullmatch(value) and not explicit:
            continue
        if _is_amount(text, s, e):
            continue
        kind = "CASE_REF" if lab.group("case") else "ID_CODE"
        yield Candidate(s, e, kind, True, PRIORITY[kind] - 1, "labelled_ref")


# ---------------------------------------------------------------------------
# BIC, MiFIR national client identifier, postal tracking numbers, IMO ship numbers
# ---------------------------------------------------------------------------

_BIC_LABEL_RE = re.compile(
    r"(?<![\w])(?:(?i:code)[ \u00a0]+)?(?:BIC|SWIFT|(?i:bic|swift))(?:[ \u00a0]*/[ \u00a0]*(?:SWIFT|BIC|(?i:swift|bic)))?"
    r"(?:[ \u00a0]+(?i:code))?[ \u00a0]*[:\-]?[ \u00a0]*"
    r"(?P<v>[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?)(?![A-Za-z0-9])"
)
# MiFIR CONCAT: country, date of birth YYYYMMDD, 5 letters of the first name, 5 of the surname ("#" pads).
_CONCAT_RE = re.compile(
    r"(?<![A-Za-z0-9#])[A-Z]{2}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])[A-Z#]{10}(?![A-Za-z0-9#])"
)
_POSTAL_CTX = re.compile(r"(?:recommande|lrar|lr/ar|accuse de reception|avis de reception|\bsuivi\b|colissimo|"
                         r"la poste|\bcolis\b|\benvoi\b)")
_POSTAL_HINTS = ("recommand", "lrar", "lr/ar", "réception", "reception", "suivi", "colis", "la poste", "envoi")
_POSTAL_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:\d[A-Z][ \u00a0]?\d{3}[ \u00a0]?\d{3}[ \u00a0]?\d{4}[ \u00a0]?\d"
    r"|[A-Z]{2}[ \u00a0]?\d{3}[ \u00a0]?\d{3}[ \u00a0]?\d{3}[ \u00a0]?[A-Z]{2})(?![A-Za-z0-9])"
)
_IMO_RE = re.compile(r"(?<![\w])IMO[ \u00a0]*(?:(?i:n[°º])\.?[ \u00a0]*)?:?[ \u00a0]*(?P<v>\d{7})(?![\d])")


def _bic_mode() -> str:
    """RAGBASE_SHIELD_MASK_BIC: "branch" (default: a BIC with a branch code, which locates the client's branch),
    "all" (every labelled BIC, also the 8-character bank code), "0" (none)."""
    import os

    v = os.environ.get("RAGBASE_SHIELD_MASK_BIC", "branch").strip().lower()
    if v in ("0", "false", "no", "off", "none"):
        return "none"
    return "all" if v in ("all", "1", "true", "yes", "on") else "branch"


# Country part of a BIC (ISO 3166): the countries of the IBAN registry.
_BIC_COUNTRIES = frozenset(IBAN_LENGTHS)


def _bic_ok(mode: str, v: str) -> bool:
    if v[4:6] not in _BIC_COUNTRIES:
        return False
    return mode == "all" or (mode == "branch" and len(v) == 11 and not v.endswith("XXX"))


_BIC_AFTER_IBAN_RE = re.compile(
    r"[ \u00a0]*[,;/\-–|(]?[ \u00a0]*(?:\(?(?i:bic|swift)\)?[ \u00a0]*:?[ \u00a0]*)?"
    r"(?P<v>[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?)(?![A-Za-z0-9])"
)


def _bic_after_iban(text: str, end: int) -> Optional[Candidate]:
    """A BIC written right after an IBAN on the same line ("FR76 ... / ABCDFRPPXYZ"), without its own label."""
    mode = _bic_mode()
    if mode == "none":
        return None
    b = _BIC_AFTER_IBAN_RE.match(text, end)
    if not b or "\n" in text[end:b.start("v")]:
        return None
    gap, v = text[end:b.start("v")], b.group("v")
    # an upper-case word after the IBAN is a BIC only with a separator, a BIC label or a digit in it
    if not (gap.strip() or any(c.isdigit() for c in v)):
        return None
    if _bic_ok(mode, v):
        return Candidate(b.start("v"), b.end("v"), "ID_CODE", True, PRIORITY["ID_CODE"], "bic")
    return None


def _find_bank_and_postal_ids(text: str) -> Iterator[Candidate]:
    pr = PRIORITY["ID_CODE"]
    low = text.lower()
    mode = _bic_mode()
    if mode != "none" and ("bic" in low or "swift" in low):
        for m in _scan_at_stems(_BIC_LABEL_RE, text, ("bic", "swift", "code")):
            if _bic_ok(mode, m.group("v")):
                yield Candidate(m.start("v"), m.end("v"), "ID_CODE", True, pr, "bic")
    for m in _CONCAT_RE.finditer(text):
        yield Candidate(m.start(), m.end(), "ID_CODE", True, pr, "mifir_concat")
    # postal tracking numbers: only in the 160 characters after a mail word
    hints = sorted({i for w in _POSTAL_HINTS for i in _all_finds(low, w)})
    seen: set[int] = set()
    for i in hints:
        for m in _POSTAL_RE.finditer(text, i, min(len(text), i + 200)):
            if m.start() >= i + 160 or m.start() in seen:
                continue
            seen.add(m.start())
            if _POSTAL_CTX.search(fold_word(text[max(0, m.start() - 120):m.start()])):
                yield Candidate(m.start(), m.end(), "ID_CODE", True, pr, "postal_tracking")
    for m in (_IMO_RE.finditer(text) if "IMO" in text else ()):
        ok = imo_valid(m.group("v"))
        yield Candidate(m.start("v"), m.end("v"), "ID_CODE", True, pr, "imo", checked=ok)


def _all_finds(hay: str, needle: str) -> Iterator[int]:
    i = hay.find(needle)
    while i >= 0:
        yield i
        i = hay.find(needle, i + 1)


# ---------------------------------------------------------------------------
# Cadastral parcel references (R1, CL-14)
# ---------------------------------------------------------------------------

_CADASTRE_WORD_RE = re.compile(r"(?i)\b(?:cadastr[ée]e?s?|cadastre|cadastrales?|parcelles?)\b")
_CAD_SEC = r"(?:0?[A-Z]{1,2})"
_CAD_NUM = r"(?:(?i:n[°º]s?)\.?[ \u00a0]*)?\d{1,4}"
_CAD_JOIN = r"[ \u00a0]*(?:,|et|&|à)[ \u00a0]*"
_CAD_SECTION_RE = re.compile(
    r"(?<![\w])(?i:section)[ \u00a0]+(?P<v>" + _CAD_SEC + r"[ \u00a0]*,?[ \u00a0]*(?P<no>(?i:n[°º]s?)\.?)?[ \u00a0]*\d{1,4}"
    r"(?:" + _CAD_JOIN + r"(?:" + _CAD_SEC + r"[ \u00a0]?)?" + _CAD_NUM + r")*)(?![\w])"
)
_CAD_ITEMS_RE = re.compile(
    r"(?<![\w\-/])(?P<v>(?P<sec>" + _CAD_SEC + r")[ \u00a0]?" + _CAD_NUM
    + r"(?:" + _CAD_JOIN + r"(?:" + _CAD_SEC + r"[ \u00a0]?)?" + _CAD_NUM + r")*)(?![\w])"
)
_CAD_NOT_SECTION = frozenset({"RN", "RD", "CD", "VC", "CR", "N", "D", "A", "RG", "CA", "TJ", "TC", "TA", "PV", "ZA", "ZI"})


def _sentence_window(text: str, start: int, n: int = 160) -> str:
    """Text before ``start`` in the same sentence (stops at ". ", "; " or a blank line)."""
    w = text[max(0, start - n):start]
    cut = max(w.rfind(". "), w.rfind("; "), w.rfind("\n\n"), w.rfind(".\n"))
    return w[cut + 1:] if cut >= 0 else w


def _find_cadastre(text: str) -> Iterator[Candidate]:
    pr = PRIORITY["ID_CODE"]
    low = text.lower()
    if "cadastr" not in low and "parcelle" not in low and "section" not in low:
        return
    for m in _scan_at_stems(_CAD_SECTION_RE, text, ("section",)):
        before = _sentence_window(text, m.start())
        after = text[m.end():m.end() + 80]
        if m.group("no") or _CADASTRE_WORD_RE.search(before) or _CADASTRE_WORD_RE.search(after.split(".")[0]):
            yield Candidate(m.start("v"), m.end("v"), "ID_CODE", True, pr, "cadastre")
    for w in _scan_at_stems(_CADASTRE_WORD_RE, text, ("cadastr", "parcelle")):
        # the items that follow a cadastre word: "parcelles AB 12 et AB 13", "cadastrée sous ZK 41"
        window_end = min(len(text), w.end() + 60)
        stop = re.search(r"[.;\n]", text[w.end():window_end])
        window_end = w.end() + stop.start() if stop else window_end
        for m in _CAD_ITEMS_RE.finditer(text, w.end(), window_end if stop else min(len(text), window_end + 40)):
            if m.start() >= window_end:
                break
            sec = m.group("sec").lstrip("0")
            direct = re.fullmatch(r"[ \u00a0]*(?:(?i:n[°º]s?)\.?|:|(?i:sous)(?:[ \u00a0]+(?i:les?|la))?(?:[ \u00a0]+(?i:n[°º]s?\.?|num[ée]ros?|r[ée]f[ée]rences?))?)?[ \u00a0]*",
                                  text[w.end():m.start()])
            if sec in _CAD_NOT_SECTION and not direct:
                continue
            yield Candidate(m.start("v"), m.end("v"), "ID_CODE", True, pr, "cadastre")


# ---------------------------------------------------------------------------
# URL
# ---------------------------------------------------------------------------

PUBLIC_URL_SUFFIXES = (
    "gouv.fr", "legifrance.gouv.fr", "courdecassation.fr", "conseil-etat.fr",
    "conseil-constitutionnel.fr", "cnil.fr", "europa.eu", "service-public.fr", "wikipedia.org",
    "wikimedia.org", "wikisource.org", "wiktionary.org", "justice.fr", "cnb.avocat.fr",
    "avocatparis.org", "coe.int", "echr.coe.int", "insee.fr", "urssaf.fr", "inpi.fr",
    "infogreffe.fr", "bodacc.fr", "francetravail.fr", "pole-emploi.fr", "ameli.fr",
    "amf-france.org", "banque-france.fr", "autoritedelaconcurrence.fr", "defenseurdesdroits.fr",
    "assemblee-nationale.fr", "senat.fr", "elysee.fr", "vie-publique.fr", "annuaire-entreprises.data.gouv.fr",
    "data.gouv.fr", "entreprendre.service-public.fr", "impots.gouv.fr", "net-entreprises.fr",
    "w3.org", "ietf.org", "iso.org", "un.org", "oecd.org", "ilo.org", "icj-cij.org",
    "example.com", "example.org", "example.net", "example.fr",
)

_TLDS = (
    "fr|com|net|org|eu|io|be|ch|lu|de|it|es|uk|co|info|biz|pro|legal|law|lawyer|avocat|avocats|"
    "notaires|paris|bzh|corsica|alsace|ai|app|dev|tech|online|site|xyz|us|ca|ma|dz|tn|sn|ci|re|"
    "mc|nl|pt|pl|at|ie|cloud|shop|store|immo|club|digital|consulting|finance|bank|group|eus|fm|tv|"
    "gp|mq|nc|pf|yt|pm|wf|tf|gf|int|edu|gov"
)
_URL_RE = re.compile(r"(?i)(?<![\w@/])(?:https?://|www\.)[^\s<>\"'«»\[\]{}|\\^`]+")
_BARE_DOMAIN_RE = re.compile(
    r"(?i)(?<![\w@./\-:])((?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)+(?:" + _TLDS + r"))(?![\w\-@])"
    r"(?:/[^\s<>\"'«»()\[\]{}]*)?"
)


def _url_host(url: str) -> str:
    u = url.lower()
    u = re.sub(r"^[a-z][a-z0-9+.\-]*://", "", u)
    host = re.split(r"[/:?#]", u, maxsplit=1)[0]
    if host.startswith("www."):
        host = host[4:]
    return host


def url_is_public(url: str) -> bool:
    """True for public institutional sites that stay in clear (legifrance, cnil.fr, europa.eu...)."""
    host = _url_host(url)
    return any(host == suf or host.endswith("." + suf) for suf in PUBLIC_URL_SUFFIXES)


# A profile path on a public host ("/in/", "/u/", "/user/", "/profile/", "/people/", "/@name") locates a person:
# such a URL is masked although its host is public (v0.2).
_PROFILE_PATH_RE = re.compile(r"(?i)^/(?:in|u|user|users|profile|profiles|people|members?|authors?)/[^\s/?#]|^/@[A-Za-z0-9]")


def _personal_path(url: str) -> bool:
    u = re.sub(r"^[a-z][a-z0-9+.\-]*://", "", url.lower())
    m = re.search(r"[/?#]", u)
    return bool(m and u[m.start()] == "/" and _PROFILE_PATH_RE.match(u[m.start():]))


def _trim_url(text: str, s: int, e: int) -> int:
    while e > s and text[e - 1] in ".,;:!?»\"'":
        e -= 1
    while e > s and text[e - 1] == ")" and text[s:e].count("(") < text[s:e].count(")"):
        e -= 1
    return e


def _find_url(text: str) -> Iterator[Candidate]:
    pr = PRIORITY["URL"]
    for m in _URL_RE.finditer(text):
        e = _trim_url(text, m.start(), m.end())
        if e - m.start() < 5:
            continue
        if url_is_public(text[m.start():e]) and not _personal_path(text[m.start():e]):
            continue
        yield Candidate(m.start(), e, "URL", True, pr, "url")
    for m in _BARE_DOMAIN_RE.finditer(text):
        dom = m.group(1)
        tld = dom.rsplit(".", 1)[1]
        if tld != tld.lower():
            continue  # "M.De" or "Ets.Fr": not a domain
        first_label = dom.split(".", 1)[0]
        if len(first_label) < 2:
            continue
        e = _trim_url(text, m.start(), m.end())
        if url_is_public(text[m.start():e]) and not _personal_path(text[m.start():e]):
            continue
        yield Candidate(m.start(), e, "URL", True, pr - 1, "domain")


# ---------------------------------------------------------------------------
# BIRTHDATE
# ---------------------------------------------------------------------------

# Month names of both languages (v0.2: English month names, "dd Month yyyy" and "Month dd, yyyy").
_MONTHS_EN = (
    r"(?:january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan\.?|feb\.?|mar\.?|apr\.?|jun\.?|jul\.?|aug\.?|sept?\.?|oct\.?|nov\.?|dec\.?)"
)
_MONTHS = (
    r"(?:janvier|f[ée]vrier|mars|avril|mai|juin|juillet|ao[ûu]t|septembre|octobre|novembre|"
    r"d[ée]cembre|january|february|march|april|may|june|july|august|december|"
    r"janv\.?|f[ée]vr?\.?|avr\.?|juil\.?|sept\.?|oct\.?|nov\.?|d[ée]c\.?|"
    r"jan\.?|feb\.?|mar\.?|apr\.?|jun\.?|jul\.?|aug\.?|sep\.?)"
)
_DATE_RE = re.compile(
    r"(?i)(?<![\w/.\-])(?:(?:1er|1ᵉʳ|\d{1,2}(?:st|nd|rd|th)?)\s+" + _MONTHS + r"\s+\d{4}"
    r"|" + _MONTHS_EN + r"\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}"
    r"|\d{1,2}\s?[/.\-]\s?\d{1,2}\s?[/.\-]\s?(?:\d{4}|\d{2})"
    r"|(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01]))(?![\w/]|\.\d)"
)
# Birth context right before a date: "né le", "née DURAND le", "Né(e) le :", "Date de naissance :", "DDN",
# "date of birth", "DOB", "born on" (a form label may end with ":" or a table pipe).
_BIRTH_CTX = re.compile(
    r"(?:\b(?:[Nn]é(?:e|s|es)?|NÉ(?:E|S|ES)?|NEE?S?)\b(?:\s*\([eE]\))?|\b[Nn]é\(e\)|"
    r"(?i:date\s+(?:et\s+lieu\s+)?de\s+naissance)|\bDDN\b|(?i:\bnaissance\b)|"
    r"(?i:\bdate\s+(?:and\s+place\s+)?of\s+birth\b)|\bDOB\b|(?i:\bborn(?:\s+on)?\b))"
    r"(?:\s+[A-ZÀ-Þ][\w'\-]*){0,3}"  # maiden name: "née DURAND le ..."
    r"(?:\s+(?:à|a|en|À|A|EN)\s+[^\n;]{1,50}?)?\s*[,:|]?\s*(?:(?:[Ll]e|LE)\b\s*[,:]?\s*)?$"
)


_BIRTH_YEAR_RE = re.compile(r"\b(?:[Nn][ée]e?s?|NÉE?S?)\s+en\s+(?P<y>(?:19|20)\d{2})(?!\d)")


def _find_birthdate(text: str) -> Iterator[Candidate]:
    for m in _DATE_RE.finditer(text):
        if _BIRTH_CTX.search(_before_raw(text, m.start(), 90)):
            yield Candidate(m.start(), m.end(), "BIRTHDATE", True, PRIORITY["BIRTHDATE"], "birthdate")
    for m in _BIRTH_YEAR_RE.finditer(text):
        yield Candidate(m.start("y"), m.end("y"), "BIRTHDATE", True, PRIORITY["BIRTHDATE"], "birth_year")


# ---------------------------------------------------------------------------
# BIRTHPLACE (plugin round 2): the town after a birth date or a birth word. A town is kept in clear elsewhere; next
# to a birth date it identifies the person ("né le 12 mars 1971 à Angers").
# ---------------------------------------------------------------------------

_PLACE_WORD = r"[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿA-ZÀ-ÖØ-Þ'’]*[a-zà-öø-ÿA-ZÀ-ÖØ-Þ]|[A-ZÀ-ÖØ-Þ]"
_PLACE = (
    r"(?P<place>(?:(?:Le|La|Les|LE|LA|LES)\s+|L['’])?(?:" + _PLACE_WORD + r")"
    r"(?:-(?:(?:sur|sous|en|lès|lez|le|la|les|de|du|des|aux?|et|SUR|SOUS|EN|LES|LE|LA|DE|DU|DES)-|[dlDL]['’])?"
    r"(?:" + _PLACE_WORD + r")"
    r"|\s+(?:sur|sous|en|lès|lez)\s+(?:" + _PLACE_WORD + r")){0,4})"
)
# Right after a birth date: "à Angers", "au Mans", ", à Saint-Étienne", "in Leeds".
_AFTER_BIRTHDATE_RE = re.compile(r"\s*,?\s*(?:à|À|au|aux|AU|AUX|in)\s+" + _PLACE)
# "né à Angers", "née DURAND à Brest", "née au Mans", "born in Leeds".
_BORN_AT_RE = re.compile(
    r"(?:\b(?:[Nn]é(?:e|s|es)?|NÉ(?:E|S|ES)?|NEE?S?)\b(?:\s*\([eE]\))?|\b[Nn]é\(e\)|\b[Bb]orn\b)"
    r"(?:\s+[A-ZÀ-Þ][\w'\-]*){0,3}?\s+(?:à|À|au|aux|AU|AUX|in)\s+" + _PLACE)
# "Lieu de naissance : Angers", "Place of birth: Leeds", "Birthplace | Leeds".
_PLACE_LABEL_RE = re.compile(
    r"(?i:\blieu\s+de\s+naissance|\bplace\s+of\s+birth|\bbirth\s*place|\bcommune\s+de\s+naissance)"
    r"\s*[:|]?\s*(?:(?i:à|in)\s+)?" + _PLACE)
_NOT_PLACE = frozenset({"monsieur", "madame", "mademoiselle", "maitre", "me", "mr", "mrs", "ms", "dr", "docteur",
                        "le", "la", "les", "l", "un", "une", "des", "son", "sa", "ses", "leur", "cette", "ce"})


def _place_ok(text: str, s: int, e: int) -> bool:
    words = [fold_word(w) for w in re.findall(r"[^\W\d_]+", text[s:e])]
    if not words:
        return False
    from .lexicon import months_days, stopwords

    head = words[-1] if words[0] in ("le", "la", "les", "l") and len(words) > 1 else words[0]
    return head not in _NOT_PLACE and head not in months_days() and head not in stopwords()


#: Birth words: the place is looked for only near them (a C-speed scan, then small windows).
_BIRTH_WORD_RE = re.compile(r"(?:\b(?:[Nn]é(?:e|s|es)?|NÉ(?:E|S|ES)?|NEE?S?)\b|\b[Nn]é\(e\)|\b[Bb]orn\b|"
                            r"(?i:naissance)|(?i:\bbirth)|\bDDN\b|\bDOB\b)")


def _find_birthplace(text: str) -> Iterator[Candidate]:
    seen: set[tuple[int, int]] = set()

    def emit(m: "Optional[re.Match[str]]") -> Iterator[Candidate]:
        if m is None:
            return
        s, e = m.start("place"), m.end("place")
        if (s, e) not in seen and _place_ok(text, s, e):
            seen.add((s, e))
            yield Candidate(s, e, "BIRTHPLACE", True, PRIORITY["BIRTHPLACE"], "birthplace")

    for h in _BIRTH_WORD_RE.finditer(text):
        a, b = h.start(), h.end()
        yield from emit(_BORN_AT_RE.match(text, a))  # "née DURAND à Brest", "born in Leeds"
        yield from emit(_PLACE_LABEL_RE.search(text, max(0, a - 30), min(len(text), b + 80)))
        y = _BIRTH_YEAR_RE.match(text, a)  # "né en 1958 à Brest"
        if y:
            yield from emit(_AFTER_BIRTHDATE_RE.match(text, y.end("y")))
        for m in _DATE_RE.finditer(text, b, min(len(text), b + 90)):  # "né le 12 mars 1971 à Angers"
            if _BIRTH_CTX.search(_before_raw(text, m.start(), 90)):
                yield from emit(_AFTER_BIRTHDATE_RE.match(text, m.end()))


# ---------------------------------------------------------------------------
# ADDRESS (French postal addresses)
# ---------------------------------------------------------------------------

_VOIE = (
    r"rue|avenue|av\.|bd\.?|boulevard|bld\.?|place|pl\.|all[ée]e|impasse|imp\.|chemin|ch\.|quai|"
    r"route|rte\.?|cours|square|sq\.|passage|lieu[- ]dit|r[ée]sidence|r[ée]s\.|villa|cit[ée]|"
    r"hameau|parvis|esplanade|promenade|rond[- ]point|sentier|faubourg|fbg\.?|faub\.?|mont[ée]e|"
    r"traverse|clos|domaine|lotissement|chauss[ée]e|venelle|ruelle|voie|"
    r"zone\s+(?:industrielle|artisanale|commerciale|d['’]activit[ée]s?)|parc\s+d['’]activit[ée]s?"
)
_VOIE_RE = re.compile(r"(?i)(?<![\w\-])(?:" + _VOIE + r"|(?-i:ZA|ZI|ZAC|ZAE|ZAL))(?![\w\-])")
_STRONG_VOIE = re.compile(
    r"(?i)^(?:rue|avenue|av\.|bd\.?|boulevard|bld\.?|all[ée]e|impasse|imp\.|quai|faubourg|fbg\.?|"
    r"lieu[- ]dit|hameau|r[ée]sidence|lotissement|ZA|ZI|ZAC|ZAE|zone\s+\w+|parc\s+d.*)$"
)
_NUM_BEFORE_RE = re.compile(
    r"(?<![\w,.])(?P<num>\d{1,4}(?:\s?[-–/]\s?\d{1,4})?(?:\s?(?:bis|ter|quater|[A-Da-d])(?![\w]))?)"
    r"\s*,?\s*$"
)
_ADDR_PARTICLES = frozenset({
    "de", "du", "des", "la", "le", "les", "l", "d", "aux", "au", "en", "sur", "sous", "les", "lès",
    "saint", "sainte", "st", "ste",
    # regional and foreign articles inside place names (Breton, Occitan, Catalan, Basque, Italian, Spanish,
    # Dutch, German): taken only between two capitalised words of the name (R1, CL-16)
    "an", "ar", "al", "el", "er", "del", "dels", "della", "delle", "dello", "di", "da", "dal", "das", "dos",
    "do", "van", "von", "der", "den", "ten", "ter", "zum", "zur", "am", "im", "y",
})
# Capitalised words that end a street name ("12 rue des Lilas Monsieur Paul ...").
_STREET_STOP = frozenset({
    "monsieur", "madame", "mademoiselle", "maitre", "m", "mme", "mlle", "me", "dr", "tel", "telephone",
    "email", "courriel", "fax", "siret", "siren", "rcs", "tva", "objet", "ref", "representee", "represente",
    "cedex", "bp", "cs", "france",
})
_ADDR_WORD_RE = re.compile(r"[^\W_][\w'’\-.]*")
# Inside a city name, words are separated by one or two spaces: a wider gap is the next column of the page.
_POSTCODE_RE = re.compile(
    r"(?:F-)?(?P<cp>(?:0[1-9]|[1-8]\d|9[0-8])\d{3})[ \t\u00a0]+"
    r"(?P<city>(?!(?i:cedex)\b)[" + _UP + r"][\w'’\-]*(?:[ \u00a0]{1,2}(?:(?:sur|sous|en|lès|les|de|du|des|la|le|d['’]|aux)[ \u00a0]{1,2})?(?!(?i:cedex)\b)[" + _UP + r"][\w'’\-]*){0,3})"
    r"(?:[ \u00a0]{1,2}(?i:cedex)(?:[ \u00a0]{1,2}\d{1,2})?)?"
)
_COLUMN_GAP_RE = re.compile(r"[ \u00a0]{3,}|[ \u00a0]*\t")
_COMPLEMENT_RE = re.compile(
    r"(?i)(?:b[aâ]t(?:iment)?\.?|esc(?:alier)?\.?|\d{1,2}\s?(?:e|er|[èe]me|ème)\s+[ée]tage|[ée]tage|"
    r"appartement|app?t\.?|porte|lot|r[ée]sidence|immeuble|tour|entr[ée]e|"
    r"(?-i:BP|B\.P\.|CS|TSA)\s?\d{2,6}|bo[iî]te\s+postale\s+\d{1,6}|"
    r"zone\s+(?:industrielle|artisanale|commerciale|d['’]activit[ée]s?)|parc\s+d['’]activit[ée]s?|(?-i:ZA|ZI|ZAC))"
    r"(?![\w])[^\n,;]{0,40}"
)
_BP_RE = re.compile(r"(?<![\w])(?:B\.?P\.?|CS|TSA)[ \u00a0]?\d{2,6}(?![\w])")
_SEG_SEP_RE = re.compile(r"[ \t\u00a0]*(?:[,\r\n–\-][ \t\u00a0]*\r?\n?[ \t\u00a0]*|[ \t\u00a0]+)")


def _address_tail(text: str, pos: int, newline_ok: bool = True) -> int:
    """Extend an address over complements and a postcode + city (up to 3 segments). ``newline_ok=False``: stay
    on the line (an address in the right column of a page: the next line starts with the left column)."""
    end = pos
    for _ in range(4):
        sep = _SEG_SEP_RE.match(text, end)
        if not sep or (not newline_ok and "\n" in sep.group()):
            break
        p = sep.end()
        pc = _POSTCODE_RE.match(text, p)
        if pc:
            return pc.end()
        comp = _COMPLEMENT_RE.match(text, p)
        if comp:
            end = comp.end()
            while end > p and text[end - 1] in " \t":
                end -= 1
            continue
        break
    return end


def _street_name_end(text: str, pos: int, strong: bool) -> Optional[int]:
    """Parse the street name after the street type; return its end or None."""
    p = pos
    n = len(text)
    end = None
    saw_name = False
    count = 0
    while p < n and count < 8:
        m = re.match(r"[ \t\u00a0]+", text[p:p + 4])
        if not m:
            if count == 0:
                return None
            break
        if count and _COLUMN_GAP_RE.match(text, p):
            break  # a column gap (3+ spaces or a tab) ends the street name; a postcode after it is the tail
        q = p + m.end()
        w = _ADDR_WORD_RE.match(text, q)
        if not w:
            break
        word = w.group().rstrip(".") if not re.fullmatch(r"(?:St|Ste|Gal|Mal|Pdt|Dr|Pr)\.", w.group()) else w.group()
        low = fold_word(word)
        first = word[:1]
        if low in _ADDR_PARTICLES:
            p = q + len(word)
            count += 1
            continue
        if first.isdigit():
            if re.fullmatch(r"\d{5}", word):
                break  # a postcode starts the tail
            if not re.fullmatch(r"\d{1,4}(?:er|e)?", word):
                break
        elif not first.isupper():
            break
        elif low in _STREET_STOP:
            break
        else:
            saw_name = True
        p = q + len(word)
        end = p
        count += 1
    if not saw_name or end is None:
        return None
    return end


_CITY_POSTCODE_RE = re.compile(
    r"(?<![\w\-])(?P<city>[" + _UP + r"][\w'’\-]+(?:[ \-](?:sur|sous|en|lès|les|de|du|des|la|le|d['’]|aux|[" + _UP
    + r"][\w'’\-]+)){0,4})\s*\(\s*(?P<cp>(?:0[1-9]|[1-8]\d|9[0-8])\d{3})\s*\)"
)
_CITY_CTX = re.compile(r"(?i)(?:\b(?:à|a|de|d['’]|au|demeurant|domicili[ée]e?s?|r[ée]sidant|si[èe]ge(?:\s+social)?(?:\s+est)?"
                       r"|situ[ée]e?s?|sis(?:e|es)?)\s*(?:à\s*)?)$")


# A capitalised adjective between the number and the street type: "7 Grande Rue", "3 Grand'Rue", "12 Vieille Route".
_STREET_ADJ_RE = re.compile(
    r"(?<![\w'’])(?P<adj>(?i:grande?|petite?|haute?|basse?|vieille|vieux|neuve|nouvelle|belle))(?:[ \u00a0]+|['’])$"
)
# A postcode + town right after an address that is already a code (a masked text masked again).
_ADDR_CODE_RE = re.compile(r"(?<![A-Za-z0-9_])(?:ADRESSE|ADDRESS)_\d{1,6}(?![A-Za-z0-9_])")


def _left_column_gap(text: str, ls: int, start: int) -> bool:
    """The address starts after other text and a column gap on its line (right column of a page)."""
    left = text[ls:start]
    return bool(left.strip()) and bool(re.search(r"(?:[ \u00a0]{3,}|\t)$", left))


def _column_postcode(text: str, start: int, end: int) -> Optional[tuple[int, int]]:
    """Postcode + town on the next line(s) at the column where the street address starts (two-column page
    layout: the street line goes on with another column, so the town is not right after the street)."""
    ls = text.rfind("\n", 0, start) + 1
    col = start - ls
    le = text.find("\n", end)
    if le < 0:
        return None
    if not (_left_column_gap(text, ls, start) or (not text[ls:start].strip() and _COLUMN_GAP_RE.match(text, end))):
        return None  # a one-column street line: the ordinary tail already looked at the next line
    for _ in range(2):  # the next line, or the one after a complement line ("BP 12", "Bâtiment C")
        nls = le + 1
        nle = text.find("\n", nls)
        nle = len(text) if nle < 0 else nle
        line = text[nls:nle]
        j = max(0, col - 2)
        while j < len(line) and j <= col + 2 and line[j] in " \t\u00a0":
            j += 1
        if j > col + 2 or j >= len(line) or (j > 0 and line[j - 1] not in " \t\u00a0" and j != 0):
            return None
        pc = _POSTCODE_RE.match(text, nls + j)
        if pc and pc.end() <= nle:
            return nls + j, pc.end()
        comp = _COMPLEMENT_RE.match(text, nls + j) or _BP_RE.match(text, nls + j)
        if not comp:
            return None
        le = nle
        if le >= len(text):
            return None
    return None


# ---------------------------------------------------------------------------
# v0.4.1: a French postal address typed in lower case ("12 rue des glycines, 69007 lyon")
# ---------------------------------------------------------------------------

def _lowercase_rules() -> bool:
    """The switch of the lower-case rules (``RAGBASE_SHIELD_LOWERCASE_NAMES``, see ``names.lowercase_names``)."""
    import os

    return os.environ.get("RAGBASE_SHIELD_LOWERCASE_NAMES", "1").strip().lower() not in ("0", "false", "no", "off")


# A number and one of these street types is an address. The other street types are also ordinary words ("3 cours
# de droit", "1 place de parking"): they need a home cue before the number or a postcode and a town after the name.
_LC_SURE_VOIE = re.compile(
    r"(?i)^(?:rue|avenue|av\.|bd\.?|boulevard|bld\.?|all[ée]e|impasse|imp\.|quai|faubourg|fbg\.?)$")
_LC_HOME_CUE = re.compile(r"(?i)(?<![\w])(?:habit(?:e|es|ent|ons|ez|ait|aient|ant)|demeur(?:e|ent|ait|aient|ant)|"
                          r"domicili[ée]e?s?|r[ée]sid(?:e|ent|ait|aient|ant)|sise?s?|situ[ée]e?s?|adresse)\s*:?\s*(?:au\s+)?$")
_LC_MONTHS = frozenset("janvier fevrier mars avril mai juin juillet aout septembre octobre novembre decembre".split())
# French particles of a street name typed in lower case; those that may also go on a name that has started.
_LC_ADDR_PARTICLES = frozenset("de du des la le les l d aux au en sur sous saint sainte st ste".split())
_LC_ADDR_LINKS = frozenset("de du des d aux au sur sous saint sainte st ste".split())
# Words that end a street name typed in lower case, beyond the function words: complements, contact words, thanks.
_LC_STREET_STOP = _STREET_STOP | frozenset(
    "bat batiment esc escalier etage appartement appt apt porte lot residence immeuble entree merci svp stp "
    "cordialement bonjour bonsoir salutations".split())
_LC_POSTCODE_RE = re.compile(r"(?:F-)?(?:0[1-9]|[1-8]\d|9[0-8])\d{3}(?![\w])")
_LC_TOWN_WORD_RE = re.compile(r"[^\W\d_][\w'’\-]*")
_LC_TOWN_HEADS = frozenset("le la les saint sainte st ste".split())
_LC_TOWN_LINKS = frozenset("sur sous les en aux de du des d".split())
# A complement typed in lower case: the word and a number or one letter ("appartement 12", "bâtiment c", "3e étage").
# The capitalised rule takes the 40 characters after the word; with no capitals the sentence may go on.
_LC_COMPLEMENT_RE = re.compile(
    r"(?i)(?:\d{1,2}\s?(?:e|er|[èe]me)\s+[ée]tage|(?:b[aâ]t(?:iment)?\.?|esc(?:alier)?\.?|[ée]tage|appartement|app?t\.?|"
    r"porte|lot|immeuble|tour|entr[ée]e)[ \u00a0](?:n°[ \u00a0]?)?(?:\d{1,4}[a-z]?|[a-z])(?![\w])|"
    r"(?-i:BP|B\.P\.|CS|TSA)\s?\d{2,6}(?![\w]))")
_LC_ELISION_RE = re.compile(r"^[ldLD]['’]")
# Words after a number of five digits that are no town ("15000 euros", "35000 exemplaires").
_LC_NOT_TOWNS = frozenset("euros euro eur francs dollars salaries habitants km ha exemplaires actions parts titres "
                          "heures jours mois ans fois personnes pages mots signes caracteres lignes".split())
# A town typed in lower case with its postcode in brackets, after a place cue: "demeurant à villeurbanne (69100)".
# The postcode in brackets is found first (it is rare), then the town is read before it.
_LC_BRACKET_POSTCODE_RE = re.compile(r"\([ \t]*(?:0[1-9]|[1-8]\d|9[0-8])\d{3}[ \t]*\)")
_LC_CITY_BEFORE_RE = re.compile(
    r"(?<![\w\-])(?P<city>[" + _LO + r"][\w'’\-]+(?:[ \-](?:(?:sur|sous|en|lès|les|de|du|des|la|le)[ \-])?["
    + _LO + r"][\w'’\-]+){0,3})[ \t]*$")
_LC_CLAUSE_END_RE = re.compile(r"[ \t\u00a0]*(?:$|[\r\n,;.:!?()«»\"])")


def _lc_stop_word(low: str, name_started: bool = True) -> bool:
    """A function word, a day or a month; after the first word of a name, also a complement or a contact word
    ("rue de la porte dorée" is a street, "12 rue des lilas porte 4" has a complement)."""
    return low in lx.stopwords() or low in lx.months_days() or (name_started and low in _LC_STREET_STOP)


def _lc_unknown_word(low: str) -> bool:
    """No French list knows the folded word (a town, a surname)."""
    from . import names  # the word classes of the lower-case name rules (names imports this module: import late)

    return not names._lc_flags(low)


def _lc_town_word(low: str) -> bool:
    """The folded word may start a town after a postcode: a word of the place list, a word that starts a town name
    ("saint", "le", "la"), a first name, or a word that no list knows ("30000 passages" is a count)."""
    from . import names  # the word classes of the lower-case rules (names imports this module: import late)

    if low in _LC_TOWN_HEADS or low in names._lc_sets()["places"]:
        return True
    flags = names._lc_flags(low)
    return not flags or "f" in flags


def _lc_real_street_name(text: str, start: int, end: int) -> bool:
    """The words after the street type are a street name: a particle starts them ("des lilas", "de la paix"), or
    the first word is a first name, a word that no list knows or a surname-like word ("victor hugo"). An adjective
    or an adverb after the street word is ordinary text ("1 impasse totale", "2 rue piétonne")."""
    from . import names

    words = _ADDR_WORD_RE.findall(text[start:end])
    if not words:
        return False
    low = fold_word(words[0].rstrip("."))
    if low in _LC_ADDR_PARTICLES or low == "notre" or words[0][:1].isdigit() or words[0][:1].isupper():
        return True
    flags = names._lc_flags(low)
    return not flags or "f" in flags or flags == "c"


def _street_name_end_lc(text: str, pos: int, cue: bool = False) -> tuple[Optional[int], bool]:
    """End of a street name typed in lower case after the street type ("des glycines", "victor hugo", "de la
    paix", "du 8 mai 1945"), or None. The name stops at a function word, a day or a month, a complement word, a
    postcode, punctuation, and at the verb of the sentence ("12 rue des lilas conteste son licenciement"): a verb
    form is a word of the name only before a postcode, or at the end of the clause after a home cue (``cue``:
    "habite 12 rue de la vieille forge"). After a capitalised word a word in lower case ends the name, as in the
    capitalised rule. Also returns whether a word of the name is in lower case ("l'Orvalie", "d'Arvignac" are
    capitalised words: the text has its capitals)."""
    n = len(text)
    p, end, names_seen, count, cap_seen, lower_seen = pos, None, 0, 0, False, False
    link = True  # particles may stand here (right after the street type, or after a link word)
    while p < n and count < 9 and names_seen < 5:
        m = re.match(r"[ \u00a0]{1,2}(?![ \u00a0\t])", text[p:p + 3])
        if not m:
            break
        q = p + m.end()
        w = _ADDR_WORD_RE.match(text, q)
        if not w:
            break
        word = w.group().rstrip(".")
        low = fold_word(word)
        if word[:1].isdigit():
            # "rue du 8 mai 1945": a day and a month right after "du"
            dm = re.match(r"\d{1,2}(?:er)?[ \u00a0]([^\W\d_]+)(?:[ \u00a0]\d{4})?(?![\w])", text[q:q + 32])
            if names_seen == 0 and dm and fold_word(dm.group(1)) in _LC_MONTHS \
                    and fold_word(text[pos:q]).strip() == "du":
                p = end = q + dm.end()
                names_seen += 1
                count += 1
                link = False
                lower_seen = True
                continue
            break
        if low in _LC_ADDR_PARTICLES and (link or low in _LC_ADDR_LINKS):
            p = q + len(word)
            count += 1
            link = True
            continue
        if not word[:1].isalpha() or len(low) < 2 or (_lc_stop_word(low, bool(names_seen)) and low != "notre"):
            break  # ("rue notre dame des champs")
        if _LC_ELISION_RE.sub("", word)[:1].isupper():
            cap_seen = True
        elif cap_seen:
            break
        elif names_seen and lx.verb_form(low) in ("form", "aux") and low not in lx.all_first_names() \
                and not word.endswith(("é", "ée", "és", "ées")) \
                and not (cue and _LC_CLAUSE_END_RE.match(text, q + len(word))) \
                and not re.match(r" ?,?[ \u00a0]{1,2}" + _LC_POSTCODE_RE.pattern,
                                 text[q + len(word):q + len(word) + 12]):
            break  # the verb of the sentence
        p = end = q + len(word)
        if low == "notre":  # "rue notre dame des champs"
            dame = re.match(r"[ \u00a0-]dame(?![\w])", text[p:p + 6])
            if not dame:
                return None, False
            p = end = p + dame.end()
        lower_seen = lower_seen or not cap_seen
        names_seen += 1
        count += 1
        link = False
    return (end if names_seen else None), lower_seen


def _postcode_town_lc(text: str, p: int, strict: bool = True) -> Optional[int]:
    """End of a postcode and a town typed in lower case at ``p`` ("69007 lyon", "13100 aix en provence", "92100
    boulogne billancourt", "76600 le havre", "75116 paris cedex 16"), or None."""
    m = _LC_POSTCODE_RE.match(text, p)
    if not m:
        return None

    def word_at(pos: int):
        sp = re.match(r"[ \u00a0]{1,2}(?![ \u00a0\t])", text[pos:pos + 3])
        w = _LC_TOWN_WORD_RE.match(text, pos + sp.end()) if sp else None
        return (w, fold_word(w.group())) if w else (None, "")

    w, low = word_at(m.end())
    if not w or low in _LC_NOT_TOWNS or (strict and not _lc_town_word(low)):
        return None  # "75000 euros", "30000 passages": an amount or a count, no postcode and town (``strict``:
        #              the town is the only proof of the address; after a home cue any town word is read)
    if low in _LC_TOWN_HEADS:
        w2, low2 = word_at(w.end())
        if not w2 or _lc_stop_word(low2):
            return None
        end = w2.end()
    elif _lc_stop_word(low) or len(low) < 2:
        return None
    else:
        end = w.end()
    for _ in range(2):
        w, low = word_at(end)
        if not w:
            break
        if low in _LC_TOWN_LINKS:
            w2, low2 = word_at(w.end())
            if w2 and low2 in ("le", "la", "les") and low in ("sur", "sous"):
                w2, low2 = word_at(w2.end())  # "sur le lez"
            if not w2 or _lc_stop_word(low2):
                break
            if low in ("sur", "sous", "aux") or w.group() == "lès" or _lc_unknown_word(low2) \
                    or _LC_CLAUSE_END_RE.match(text, w2.end()):
                end = w2.end()
                continue
            break
        if low == "cedex":
            end = w.end()
            num = re.match(r"[ \u00a0]{1,2}\d{1,2}(?![\w])", text[end:end + 5])
            end += num.end() if num else 0
            break
        if not _lc_stop_word(low) and len(low) >= 4 and _lc_unknown_word(low) and not w.group()[:1].isupper():
            end = w.end()  # "boulogne billancourt", "clermont ferrand"
            continue
        break
    arr = re.match(r"[ \u00a0]{1,2}\d{1,2}(?:e|er|eme|ème)?(?![\w])", text[end:end + 8])
    if arr and not re.match(r"[ \u00a0]{1,2}[^\W\d_]+", text[end + arr.end():end + arr.end() + 12]):
        end += arr.end()  # "paris 16e" (not "lyon 3 mars")
    return end


def _address_tail_lc(text: str, pos: int, newline_ok: bool = True, strict: bool = True) -> tuple[int, bool, bool]:
    """``_address_tail`` with a town in any letter case: the end, whether a postcode and a town were read, and
    whether the town is in lower case. The postcode and the town may stand on the next line ("12 rue des
    glycines", then "69007 lyon"); a complement stays on the line."""
    end = pos
    for _ in range(4):
        sep = _SEG_SEP_RE.match(text, end)
        if not sep:
            break
        lines = sep.group().count("\n")
        if lines > 1 or (lines and not newline_ok):
            break
        p = sep.end()
        pc = _POSTCODE_RE.match(text, p)
        if pc:
            return pc.end(), True, False
        town = _postcode_town_lc(text, p, strict)
        if town:
            return town, True, True
        comp = None if lines else _LC_COMPLEMENT_RE.match(text, p)
        if comp:
            end = comp.end()
            continue
        break
    return end, False, False


_LC_CAP_WORD_RE = re.compile(r"(?<![\w'’\-])[" + _UP + r"][\w'’\-]*")
_LC_SENTENCE_START_RE = re.compile(r"(?:[.!?…][ \u00a0]+|\n)")


def _lc_typed_sentence(text: str, start: int, end: int) -> bool:
    """The sentence around ``text[start:end]`` is typed without capitals: no word has a capital but its first
    word and the words that ``names._lc_neutral_capital`` reads (first names, places, months, acronyms)."""
    ls = 0
    for m in _LC_SENTENCE_START_RE.finditer(text, max(0, start - 400), start):
        ls = m.end()
    ls = max(ls, start - 400, 0)
    m = _LC_SENTENCE_START_RE.search(text, end)
    le = m.start() if m else min(len(text), end + 400)
    from . import names  # the same test as the name rules (names imports this module: import late)

    first = True
    for m in re.finditer(r"[^\W\d_][\w'’\-]*", text[ls:le]):
        word = m.group()
        if not first and word[:1].isupper():
            w = names._make_word(word, m.start(), m.end())
            if not names._lc_neutral_capital(w):
                return False
        first = False
    return True


def _find_address_lc(text: str, voies: Optional[list] = None) -> Iterator[Candidate]:
    """A street address typed in lower case: "12 rue des glycines, 69007 lyon", "5 avenue victor hugo 75016
    paris"; with no number after a home cue or before a postcode and a town ("il habite rue des glycines",
    "impasse du moulin 44000 nantes"); a town and its postcode in brackets after a place cue ("demeurant à
    villeurbanne (69100)"). The capitalised rule reads the capitalised text; this rule adds the readings it
    refuses (a street name or a town in lower case). The longest candidate wins, so text with capitals keeps its
    reading."""
    pr = PRIORITY["ADDRESS"]
    for b in _LC_BRACKET_POSTCODE_RE.finditer(text):
        w0 = max(0, b.start() - 60)
        m = _LC_CITY_BEFORE_RE.search(text[w0:b.start()])
        if not m:
            continue
        # the cue stands right before the town: the match may start some words before it
        city, cs = m.group("city"), w0 + m.start("city")
        for k in [0] + [j + 1 for j, c in enumerate(city) if c in " -"]:
            if _CITY_CTX.search(text[max(0, cs - 40):cs + k]) and not _lc_stop_word(fold_word(city[k:].split()[0])):
                if _lc_typed_sentence(text, cs + k, b.end()):
                    yield Candidate(cs + k, b.end(), "ADDRESS", True, pr - 1, "city_postcode_lower")
                break
    for m in (voies if voies is not None else _VOIE_RE.finditer(text)):
        vs, ve = m.start(), m.end()
        w0 = max(0, vs - 25)
        num = _NUM_BEFORE_RE.search(text[w0:vs])
        sure = bool(_LC_SURE_VOIE.match(m.group()))
        if not num and not sure:
            continue
        start = w0 + num.start("num") if num else vs
        cue = bool(_LC_HOME_CUE.search(text[max(0, start - 30):start]))
        name_end, lower_name = _street_name_end_lc(text, ve, cue)
        if name_end is None:
            continue
        ls = text.rfind("\n", 0, start) + 1
        end, town, lower_town = _address_tail_lc(text, name_end, newline_ok=not _left_column_gap(text, ls, start),
                                                 strict=not cue)
        if not lower_name and not lower_town:
            continue  # the text has its capitals: the capitalised rule reads it
        if not town and not cue:
            continue  # a postcode and a town, or a home cue: "1 rue complète est fermée" is no address
        if not _lc_real_street_name(text, ve, name_end):
            continue  # "il demeure 1 impasse complète dans les pourparlers"
        if not _lc_typed_sentence(text, start, name_end):
            continue  # the sentence has its capitals ("En 2022 impasse des discussions tarifaires, à Lille")
        yield Candidate(start, end, "ADDRESS", True, pr, "address_lower")


def _find_address(text: str) -> Iterator[Candidate]:
    pr = PRIORITY["ADDRESS"]
    for m in _CITY_POSTCODE_RE.finditer(text):
        # "dont le siège social est à Francheville (69340)", "Demeurant à Lyon (69003)"
        if _CITY_CTX.search(text[max(0, m.start() - 40):m.start()]):
            yield Candidate(m.start(), m.end(), "ADDRESS", True, pr - 1, "city_postcode")
    voies = list(_VOIE_RE.finditer(text))
    for m in voies:
        vs, ve = m.start(), m.end()
        w0 = max(0, vs - 25)
        voie = m.group()
        strong = bool(_STRONG_VOIE.match(voie))
        adj = _STREET_ADJ_RE.search(text[w0:vs])
        if adj and not adj.group("adj")[:1].isupper():
            adj = None
        if adj:
            # "7 Grande Rue": the number is before the adjective, and the street needs no further name
            num = _NUM_BEFORE_RE.search(text[w0:w0 + adj.start("adj")])
            start = w0 + num.start("num") if num else w0 + adj.start("adj")
            if not num:
                continue
        else:
            num = _NUM_BEFORE_RE.search(text[w0:vs])
            start = w0 + num.start("num") if num else vs
        if not num and not strong:
            continue
        if not num and fold_word(voie).startswith("res") and re.match(r"\s+(?:de|d['’])\s*[" + _UP + r"][\w'’\-]*(?:\s|[,.;)]|$)", text[ve:ve + 40]):
            continue  # "titulaire d'un office à la résidence de Lyon": a seat, not a postal address
        name_end = _street_name_end(text, ve, strong)
        if name_end is None:
            if not adj:
                continue
            name_end = ve
        ls = text.rfind("\n", 0, start) + 1
        end = _address_tail(text, name_end, newline_ok=not _left_column_gap(text, ls, start))
        yield Candidate(start, end, "ADDRESS", True, pr, "address")
        if not _POSTCODE_RE.search(text, name_end, end):
            col = _column_postcode(text, start, end)
            if col:
                yield Candidate(col[0], col[1], "ADDRESS", True, pr, "address_column")
    for m in _BP_RE.finditer(text):
        end = _address_tail(text, m.end())
        yield Candidate(m.start(), end, "ADDRESS", True, pr, "bp")
    for m in (_ADDR_CODE_RE.finditer(text) if "ADRESSE_" in text or "ADDRESS_" in text else ()):
        sep = _SEG_SEP_RE.match(text, m.end())
        if sep and sep.end() > m.end():
            pc = _POSTCODE_RE.match(text, sep.end())
            if pc:
                yield Candidate(pc.start(), pc.end(), "ADDRESS", True, pr, "address_code_tail")
    yield from _find_foreign_address(text)
    if _lowercase_rules():
        yield from _find_address_lc(text, voies)


# UK and Canadian addresses: "14 Quarry Lane, Bristol BS7 4QX", "Halifax (Nova Scotia) B3K 2T9".
_EN_CAP = r"[A-Z][\w'’\-]*"
_UK_PC = r"[A-Z]{1,2}\d[A-Z\d]?[ \u00a0]\d[A-Z]{2}"
_CA_PC = r"[A-Z]\d[A-Z][ \u00a0\-]?\d[A-Z]\d"
_EN_REGION = r"(?:[ \u00a0]*\([A-Z][\w'’\- ]{1,30}\)|,[ \u00a0]*[A-Z]{2}(?![\w]))?"
_EN_TAIL_RE = re.compile(
    r"(?:,[ \u00a0]*|[ \u00a0]+|,?[ \t\u00a0]*\r?\n[ \t\u00a0]*)"
    r"(?:" + _EN_CAP + r"(?:[ \u00a0]" + _EN_CAP + r"){0,2}" + _EN_REGION + r",?[ \u00a0]+)?"
    r"(?:" + _UK_PC + "|" + _CA_PC + r")(?![\w])"
)
_EN_STREET_WORDS = ("Street", "Road", "Lane", "Avenue", "Drive", "Close", "Crescent", "Terrace", "Gardens", "Mews",
                    "Boulevard", "Square", "Way", "Grove", "Row", "Court", "Parkway", "Highway", "Rd", "Ave", "Blvd",
                    "Ln")
_EN_PC_RE = re.compile(r"(?<![\w])(?:" + _UK_PC + "|" + _CA_PC + r")(?![\w])")
_EN_CITY_BEFORE_RE = re.compile(
    r"(?<![\w\-])" + _EN_CAP + r"(?:[ \u00a0]" + _EN_CAP + r"){0,2}" + _EN_REGION + r",?[ \u00a0]+$"
)


_EN_PC_CORE_RE = re.compile(r"\d[A-Z][A-Z\d](?![\w])")  # the inward part, "4QX" of "BS7 4QX", "2T9" of "B3K 2T9"


def _en_postcodes(text: str) -> Iterator["re.Match[str]"]:
    last = -1
    for c in _EN_PC_CORE_RE.finditer(text):
        for st in range(max(0, c.start() - 5), c.start()):
            if st < last:
                continue
            m = _EN_PC_RE.match(text, st)
            if m and m.end() == c.end():
                last = m.end()
                yield m
                break


_EN_STREET_BACK_RE = re.compile(r"(?<![\w])\d{1,5}[A-Za-z]?,?[ \u00a0]+(?:" + _EN_CAP + r"[ \u00a0]+){1,4}$")


def _find_foreign_address(text: str) -> Iterator[Candidate]:
    pr = PRIORITY["ADDRESS"]
    n = len(text)
    for word in _EN_STREET_WORDS:
        for i in _all_finds(text, word):
            j = i + len(word)
            if (i and text[i - 1].isalnum()) or (j < n and (text[j].isalnum() or text[j] == "_")):
                continue
            if word in ("Rd", "Ave", "Blvd", "Ln") and j < n and text[j] == ".":
                j += 1
            b = _EN_STREET_BACK_RE.search(text, max(0, i - 90), i)
            if not b:
                continue
            end = j
            t = _EN_TAIL_RE.match(text, end)
            if t:
                end = t.end()
            yield Candidate(b.start(), end, "ADDRESS", True, pr, "address_en")
    for pc in _en_postcodes(text):
        w0 = max(0, pc.start() - 70)
        c = _EN_CITY_BEFORE_RE.search(text[w0:pc.start()])
        if c:
            yield Candidate(w0 + c.start(), pc.end(), "ADDRESS", True, pr - 1, "city_postcode_en")


# ---------------------------------------------------------------------------
# Named assets and places that locate a person: vessels (R1, CL-07), schools and care homes (R1, CL-10)
# ---------------------------------------------------------------------------

# Words that make an establishment name a public body (kept in clear) rather than a local school.
_INSTITUTION_WORDS = frozenset({
    "nationale", "national", "normale", "superieure", "superieur", "polytechnique", "doctorale", "centrale",
    "avocats", "avocat", "barreau", "magistrature", "greffes", "greffe", "formation", "regionale", "europeenne",
    "internationale", "hautes", "etudes", "commerce", "ingenieurs", "droit", "militaire", "navale", "universite",
    "universitaire", "administration", "sante", "police", "gendarmerie", "republique", "nation", "etat",
    "france", "europe", "monde",
})
# Capitalised words after "collège" or a form label that are not names: electoral colleges of a works council,
# short form answers.
_NOT_NAME_WORDS = frozenset({
    "cadres", "cadre", "employes", "employe", "ouvriers", "ouvrier", "techniciens", "maitrise", "agents",
    "salaries", "electoral", "electoraux", "unique", "premier", "deuxieme", "second", "troisieme",
    "oui", "non", "neant", "aucun", "aucune", "sans", "inconnu", "inconnue", "nc", "na", "nr",
})
_NAME_STOP_FIRST = frozenset({
    "m", "mme", "mlle", "me", "monsieur", "madame", "mademoiselle", "maitre", "dr", "docteur", "pr",
    "sas", "sasu", "sarl", "sa", "eurl", "sci", "snc", "selarl", "selas", "scp", "gie",
})
_PNAME_WORD = r"(?:[" + _UP + r"][\w'’]*(?:-[\w'’]+)*)"
_PNAME = (_PNAME_WORD + r"(?:[ \u00a0](?:(?:de|du|des|d['’]|la|le|l['’]|les|en|ar|an|al|el|von|van)[ \u00a0]?)?"
          + _PNAME_WORD + r"){0,3}")
_QUOTED = r"[«\"“][ \u00a0]?[^«»\"“”\n]{2,40}?[ \u00a0]?[»\"”]"
_VESSEL_TRIGGER = (r"(?i:navire(?:[- ]citerne)?|bateau|vedette|catamaran|trimaran|car[- ]ferry|ferry|chalutier|"
                   r"voilier|caboteur|roulier|remorqueur|yacht|p[ée]niche|paquebot|cargo|p[ée]trolier|m[ée]thanier|"
                   r"vraquier|porte[- ]conteneurs|thonier|palangrier|fileyeur|caseyeur|go[ée]lette|gabare|baliseur|"
                   r"dragueur|pilotine|vessel|tanker|trawler)|NGV|M/[VY]|MV")
_VESSEL_RE = re.compile(
    r"(?<![\w\-/])(?P<trig>" + _VESSEL_TRIGGER + r")(?![\w\-])"
    r"(?:[ \u00a0]+(?i:baptis[ée]e?|nomm[ée]e?|d[ée]nomm[ée]e?|appel[ée]e?))?[ \u00a0]*:?[ \u00a0]*"
    r"(?:(?P<q>" + _QUOTED + r")|(?P<name>(?:L['’]|(?:Le|La|Les|LE|LA|LES)[ \u00a0])?" + _PNAME + r"))"
)
_SCHOOL_TRIGGER = (r"(?i:[ée]cole|groupe[ \u00a0]scolaire|coll[èe]ge|lyc[ée]e|micro-cr[èe]che|cr[èe]che|"
                   r"halte-garderie|multi-accueil|jardin[ \u00a0]d['’]enfants|internat|centre[ \u00a0]de[ \u00a0]loisirs|"
                   r"accueil[ \u00a0]de[ \u00a0]loisirs|[ée]tablissement[ \u00a0]scolaire)|IME|ITEP|SESSAD|EHPAD|IEM")
_SCHOOL_QUAL = (r"(?i:maternelle|primaire|[ée]l[ée]mentaire|priv[ée]e?|publique?|professionnel(?:le)?|"
                r"g[ée]n[ée]ral(?:e)?|technique|technologique|agricole|polyvalent|municipal(?:e)?|catholique|bilingue|"
                r"sous[ \u00a0]contrat|et[ \u00a0]coll[èe]ge|et[ \u00a0]lyc[ée]e)")
_SCHOOL_RE = re.compile(
    r"(?<![\w\-])(?P<trig>" + _SCHOOL_TRIGGER + r")(?![\w\-])(?:[ \u00a0]+" + _SCHOOL_QUAL + r"(?![\w\-])){0,3}"
    r"[ \u00a0]*:?[ \u00a0]*(?P<part>(?:de[ \u00a0]la|de[ \u00a0]l['’]|des|du|d['’]|de|l['’]|le|la|les)(?:[ \u00a0]+|(?<=['’])))?"
    r"(?:(?P<q>" + _QUOTED + r")|(?P<name>" + _PNAME + r"))"
)
_TRAILING_PARTICLE_RE = re.compile(r"(?:[ \u00a0]+(?:de|du|des|d['’]|la|le|l['’]|les|en|ar|an|al|el|von|van))+$")


@lru_cache(maxsize=1)
def _plain_words() -> frozenset[str]:
    """Words that never make a name distinctive: stopwords, common and legal words, first names, months."""
    wl = frozenset(p for p in lx.whitelist_phrases() if " " not in p)
    return lx.stopwords() | lx.common_words() | lx.legal_terms() | lx.first_names() | lx.months_days() | wl


def _name_words(name: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", fold_word(name))


def _distinctive(name: str) -> bool:
    """A name with a word that is not an ordinary word (common or legal word, first name, month...): masking it
    alone does not turn an ordinary word into a dictionary entry."""
    plain = _plain_words()
    return any(len(t) >= 4 and t not in plain and t not in _NOT_NAME_WORDS and t not in _ADDR_PARTICLES
               for t in _name_words(name))


def _name_ok(name: str) -> bool:
    words = _name_words(name)
    if not words or words[0] in _NAME_STOP_FIRST or words[0] in lx.months_days():
        return False
    if any(w in _INSTITUTION_WORDS for w in words):
        return False
    return not all(w in _NOT_NAME_WORDS or w in _ADDR_PARTICLES for w in words)


def _named_span(text: str, m: re.Match, q_kind: str) -> Optional[tuple[int, int, str]]:
    """(start, end, kind) for a vessel or school match: the name alone when it is distinctive (the trigger word
    stays in clear), else the trigger word and the name together as one ID_CODE (an ordinary word alone never
    becomes an entity)."""
    if m.group("q"):
        s, e = m.start("q") + 1, m.end("q") - 1  # inside the quotes
        while s < e and text[s] in " \u00a0":
            s += 1
        while e > s and text[e - 1] in " \u00a0":
            e -= 1
        inner = text[s:e]
    else:
        s, e = m.start("name"), m.end("name")
        tp = _TRAILING_PARTICLE_RE.search(text, s, e)
        if tp:
            e = tp.start()
        inner = text[s:e]
    if not inner or not _name_ok(inner):
        return None
    if _distinctive(inner):
        return s, e, q_kind
    return m.start("trig"), e, "ID_CODE"


_VESSEL_STEMS = ("navire", "bateau", "vedette", "catamaran", "trimaran", "ferry", "chalutier", "voilier", "caboteur",
                 "roulier", "remorqueur", "yacht", "péniche", "peniche", "paquebot", "cargo", "pétrolier", "petrolier",
                 "méthanier", "methanier", "vraquier", "conteneurs", "thonier", "palangrier", "fileyeur", "caseyeur",
                 "goélette", "goelette", "gabare", "baliseur", "dragueur", "pilotine", "vessel", "tanker", "trawler")
_VESSEL_CASED = ("NGV", "M/V", "M/Y", "MV")
_SCHOOL_STEMS = ("école", "ecole", "groupe", "collège", "college", "lycée", "lycee", "crèche", "creche", "garderie",
                 "accueil", "jardin", "internat", "centre", "tablissement")
_SCHOOL_CASED = ("IME", "ITEP", "SESSAD", "EHPAD", "IEM")


def _find_named_places(text: str) -> Iterator[Candidate]:
    yield from _find_vessels(text)
    yield from _find_schools(text)


def _find_vessels(text: str) -> Iterator[Candidate]:
    for m in _scan_at_stems(_VESSEL_RE, text, _VESSEL_STEMS, _VESSEL_CASED):
        got = _named_span(text, m, "ID_CODE")
        if got:
            yield Candidate(got[0], got[1], "ID_CODE", True, PRIORITY["ID_CODE"] - 1, "vessel")


def _find_schools(text: str) -> Iterator[Candidate]:
    for m in _scan_at_stems(_SCHOOL_RE, text, _SCHOOL_STEMS, _SCHOOL_CASED):
        part = fold_word(m.group("part") or "").strip()
        if m.group("name") and part in ("de", "d'", "d’") and len(_name_words(m.group("name"))) == 1:
            continue  # "lycée de Kerbrat": a town name, kept in clear like any town
        got = _named_span(text, m, "ORG")
        if got:
            s, e, kind = got
            yield Candidate(s, e, kind, True, PRIORITY[kind] - 1, "school")


# ---------------------------------------------------------------------------
# USERNAME, EMPLOYEE_ID, ACCOUNT_NUMBER (v0.2, DETECTORS): chat mentions, handles, labelled identifiers in both
# languages, US routing numbers, and the id layouts learnt from labelled values (shape memory)
# ---------------------------------------------------------------------------

# Slack-style mention "<@U024BE7LH>" or "<@U024BE7LH|jdoe>": the id is the username, "<@" and ">" stay in clear.
# Teams mentions "<at>Name</at>" carry a name, which the name rules handle.
_MENTION_RE = re.compile(r"<@(?P<id>[A-Za-z0-9][A-Za-z0-9_.\-]{1,})(?:\|[^>\n]{0,80})?>")
# "@handle": 2 to 40 characters of letters, digits, "_", "." and "-", not glued to a word or to another "@" (an
# e-mail address is a longer span and wins), not a domain, not a public mention.
_HANDLE_RE = re.compile(r"(?<![\w@.\-/])@(?P<h>[A-Za-z0-9_][A-Za-z0-9_.\-]{1,39})(?![\w@])")
PUBLIC_HANDLES = frozenset({
    "channel", "here", "everyone", "all", "team", "all-hands", "group", "nobody", "mentions", "mention",
})


def _find_mentions(text: str) -> Iterator[Candidate]:
    if "<@" not in text:
        return
    for m in _MENTION_RE.finditer(text):
        yield Candidate(m.start("id"), m.end("id"), "USERNAME", True, PRIORITY["USERNAME"], "mention")


def _find_handles(text: str) -> Iterator[Candidate]:
    if "@" not in text:
        return
    for m in _HANDLE_RE.finditer(text):
        s, e = m.start("h"), m.end("h")
        while e > s and text[e - 1] in ".-":
            e -= 1  # a sentence dot or a dash after the handle
        body = text[s:e]
        if len(body) < 2 or not any(c.isalpha() for c in body) or body.lower() in PUBLIC_HANDLES:
            continue
        if _BARE_DOMAIN_RE.fullmatch(body) or text[s - 2:s] == "<@":
            continue  # "@acme.example" is a domain; "<@U024...>" is a mention
        yield Candidate(s, e, "USERNAME", True, PRIORITY["USERNAME"], "handle")


# Labels that announce an identifier, mapped to a kind. One list for both languages; the value follows the label
# through the same gap logic as the labelled references (number marks, ":", connector words, the next line after a
# form label). "id" alone (the generic group) needs a value with letters and digits.
_ID_LABEL_RE = re.compile(
    r"(?i)(?<![\w/])(?:"
    r"(?P<emp>employee[ \u00a0]?(?:id|#|number|no\.?|num)|emp[ \u00a0]?(?:id|#|no\.?)|staff[ \u00a0](?:id|number|no\.?)|"
    r"badge(?:[ \u00a0](?:id|number|no\.?))?|personnel[ \u00a0]number|worker[ \u00a0]id|payroll[ \u00a0](?:id|number)|"
    r"matricule|n[°º][ \u00a0]?matricule|num[ée]ro[ \u00a0]de[ \u00a0]salari[ée]|identifiant[ \u00a0]salari[ée]|"
    r"agent[ \u00a0]n[°º])"
    r"|(?P<acct>account[ \u00a0](?:number|#|no\.?|num)|acct\.?(?:[ \u00a0]?(?:#|no\.?|number))?|a/c(?:[ \u00a0]?(?:no\.?|#))?|"
    r"customer[ \u00a0]account|policy[ \u00a0](?:number|#|no\.?)|member(?:ship)?[ \u00a0](?:id|number|no\.?)|"
    r"customer[ \u00a0]id|client[ \u00a0]id|loyalty[ \u00a0]number|routing[ \u00a0](?:number|no\.?|#)|"
    r"(?-i:ABA)(?:[ \u00a0](?:number|no\.?|#))?|num[ée]ro[ \u00a0]de[ \u00a0]compte|compte[ \u00a0]n[°º]|"
    r"n[°º][ \u00a0]de[ \u00a0]compte|compte[ \u00a0]client|n[°º][ \u00a0]adh[ée]rent|n[°º][ \u00a0]de[ \u00a0]police|"
    r"n[°º][ \u00a0]soci[ée]taire)"
    r"|(?P<user>username|user[ \u00a0]name|login|user[ \u00a0]id|handle|screen[ \u00a0]name|"
    r"identifiant[ \u00a0]de[ \u00a0]connexion|nom[ \u00a0]d['’]utilisateur|pseudo(?:nyme)?)"
    r"|(?P<gen>id|identifier|reference[ \u00a0]id|ticket[ \u00a0]id)"
    r")(?![\w])"
)
_ID_LABEL_STEMS = ("employee", "emp", "staff", "badge", "personnel", "worker", "payroll", "matricule", "n°", "nº",
                   "numéro", "numero", "identifiant", "agent", "account", "acct", "a/c", "customer", "policy",
                   "member", "client", "loyalty", "routing", "compte", "username", "user", "login", "handle",
                   "screen", "nom", "pseudo", "id", "reference", "ticket")
_ID_LABEL_CASED = ("ABA",)
_ID_LABEL_KINDS = {"emp": "EMPLOYEE_ID", "acct": "ACCOUNT_NUMBER", "user": "USERNAME", "gen": "ID_CODE"}
_USER_VALUE_RE = re.compile(r"@?(?P<v>[A-Za-z][A-Za-z0-9_.\-]{2,39})(?![\w@])")
_TIME_RE = re.compile(r"\d{1,2}[hH:]\d{2}")


def _id_value_ok(value: str, explicit: bool = True) -> bool:
    """A labelled id value: 4 to 24 characters, at least 3 digits or 2 letters + 2 digits; not a date, a year, a
    time, a grouped number without an explicit marker, nor a Shield code."""
    if not 4 <= len(value) <= 24:
        return False
    digits = sum(c.isdigit() for c in value)
    letters = sum(c.isalpha() for c in value)
    if digits < 3 and not (letters >= 2 and digits >= 2):
        return False
    if _DATE_SHAPE_RE.fullmatch(value) or re.fullmatch(r"(?:19|20)\d{2}", value) or _TIME_RE.fullmatch(value):
        return False
    if _GROUPED_NUMBER_RE.fullmatch(value) and not explicit:
        return False
    return not CODE_RE.fullmatch(value)


def _find_labelled_ids(text: str) -> Iterator[Candidate]:
    n = len(text)
    for lab in _scan_at_stems(_ID_LABEL_RE, text, _ID_LABEL_STEMS, _ID_LABEL_CASED):
        group = lab.lastgroup
        kind = _ID_LABEL_KINDS[group]
        if group == "user":
            q = lab.end()
            while q < n and text[q] in " \t\u00a0":
                q += 1
            mk = re.match(r"(?::|=|-|–)[ \t\u00a0]*", text[q:q + 4])
            if mk:
                q += mk.end()
                if text[q:q + 1] == "\n":  # "Login:" with the value on the next line (form layout)
                    q += 1
                    while q < n and text[q] in " \t\u00a0":
                        q += 1
            m = _USER_VALUE_RE.match(text, q)
            if not m:
                continue
            s, e = m.start("v"), m.end("v")
            while e > s and text[e - 1] in ".-":
                e -= 1
            value = text[s:e]
            plain = fold_word(value)
            if len(value) < 3 or plain in _plain_words() or plain in PUBLIC_HANDLES:
                continue
            if value.isalpha():
                le = text.find("\n", e)
                le = n if le < 0 else le
                if text[e:le].strip(" \t\u00a0|,;."):
                    continue  # "login : see below": a value made of letters only must end its line or cell
            yield Candidate(s, e, kind, True, PRIORITY[kind], "labelled_id")
            continue
        got = _ref_value_at(text, lab.end())
        if got is None:
            continue
        s, e, explicit = got
        value = text[s:e]
        if group == "gen" and not (any(c.isalpha() for c in value) and any(c.isdigit() for c in value)):
            continue
        if not _id_value_ok(value, explicit) or _is_amount(text, s, e) or text[e:e + 1] == "%":
            continue
        yield Candidate(s, e, kind, True, PRIORITY[kind], "labelled_id")


# US routing number (ABA RTN): 9 digits, weights 3-7-1, first two digits in a Federal Reserve range. Masked even
# without a label when a payment word is on the line (a bare 9-digit number is also a SIREN shape).
_ROUTING_RE = re.compile(r"(?<![\dA-Za-z])(?<!\d[ .\-])\d{9}(?![\dA-Za-z])(?![ .\-]?\d)")
_ROUTING_CTX = re.compile(r"(?:routing|\baba\b|\bach\b|\bwire\b|transit|\brtn\b)")
_ROUTING_PREFIXES = frozenset(
    f"{i:02d}" for i in list(range(1, 13)) + list(range(21, 33)) + list(range(61, 73)) + [80]
)


def aba_routing_valid(value: str) -> bool:
    """ABA routing transit number: 9 digits, 3*(d1+d4+d7) + 7*(d2+d5+d8) + (d3+d6+d9) = 0 mod 10, and a Federal
    Reserve prefix (01-12, 21-32, 61-72, 80)."""
    s = re.sub(r"\D", "", value)
    if len(s) != 9 or s[:2] not in _ROUTING_PREFIXES:
        return False
    d = [int(c) for c in s]
    return (3 * (d[0] + d[3] + d[6]) + 7 * (d[1] + d[4] + d[7]) + (d[2] + d[5] + d[8])) % 10 == 0


def _find_routing(text: str) -> Iterator[Candidate]:
    low = text.lower()
    if not _ROUTING_CTX.search(low):
        return
    for m in _ROUTING_RE.finditer(text):
        if not aba_routing_valid(m.group()):
            continue
        ls = text.rfind("\n", 0, m.start()) + 1
        le = text.find("\n", m.end())
        le = len(text) if le < 0 else le
        if _ROUTING_CTX.search(low[max(ls, m.start() - 80):min(le, m.end() + 40)]):
            yield Candidate(m.start(), m.end(), "ACCOUNT_NUMBER", True, PRIORITY["ACCOUNT_NUMBER"], "routing")


# Id shape memory: a labelled EMPLOYEE_ID or ACCOUNT_NUMBER value with a letter prefix ("EMP-00123") or a fixed
# letters-and-digits layout ("A1B2-7788") gives a template (letters upper case, digits "d"); unlabelled values of
# the same layout are then masked in the whole dossier. A pure digit run never becomes a template.
_SHAPE_PREFIX_RE = re.compile(r"(?P<pre>[A-Za-z]{2,5})(?P<sep>[\-/._ ]?)(?P<digits>\d{3,12})")
_SHAPE_CHARS_RE = re.compile(r"[A-Za-z0-9\-/._ ]+")


def shape_template(value: str) -> Optional[str]:
    """Template of an id layout ("EMP-00123" -> "EMP-ddddd"), or None when the value teaches no layout."""
    v = value.strip()
    if not v or v.isdigit() or not _SHAPE_CHARS_RE.fullmatch(v) or not 5 <= len(v) <= 24:
        return None
    m = _SHAPE_PREFIX_RE.fullmatch(v)
    if m:
        return m.group("pre").upper() + m.group("sep") + "d" * len(m.group("digits"))
    if sum(c.isdigit() for c in v) < 3 or not any(c.isalpha() for c in v):
        return None
    return "".join("d" if c.isdigit() else c.upper() for c in v)


@lru_cache(maxsize=256)
def _shape_pattern(template: str) -> "re.Pattern[str]":
    body = "".join(r"\d" if c == "d" else re.escape(c) for c in template)
    return re.compile(r"(?<![A-Za-z0-9])" + body + r"(?![A-Za-z0-9])", re.IGNORECASE)


def _find_shapes(text: str, shapes: Iterable[tuple[str, str]]) -> Iterator[Candidate]:
    for kind, template in shapes:
        if kind not in PRIORITY or not template or "d" not in template:
            continue
        for m in _shape_pattern(template).finditer(text):
            if _is_amount(text, m.start(), m.end()):
                continue
            yield Candidate(m.start(), m.end(), kind, True, PRIORITY[kind], "shape")


# ---------------------------------------------------------------------------
# Tables: pipe-table cells with their column header (R1 interface, used by NAMES too)
# ---------------------------------------------------------------------------

_TABLE_RULE_RE = re.compile(r"^[ \t]*\|?[ \t]*:?-{3,}:?[ \t]*(?:\|[ \t]*:?-{3,}:?[ \t]*)*\|?[ \t]*$")
_PIPE_RE = re.compile(r"(?<!\\)\|")


def _row_cells(text: str, ls: int, le: int) -> Optional[list[tuple[int, int]]]:
    """Trimmed (start, end) of each cell of the row text[ls:le]; None when the row has fewer than 2 cells."""
    pipes = [m.start() for m in _PIPE_RE.finditer(text, ls, le)]
    if not pipes:
        return None
    raw = list(zip([ls] + [p + 1 for p in pipes], pipes + [le]))
    if not text[raw[0][0]:raw[0][1]].strip():
        raw = raw[1:]  # leading "|"
    if raw and not text[raw[-1][0]:raw[-1][1]].strip():
        raw = raw[:-1]  # trailing "|"
    if len(raw) < 2:
        return None
    cells = []
    for s, e in raw:
        while s < e and text[s] in " \t\u00a0":
            s += 1
        while e > s and text[e - 1] in " \t\u00a0\r":
            e -= 1
        cells.append((s, e))
    return cells


def _table_rows(text: str) -> list[tuple[list[tuple[int, int]], list[str]]]:
    """Data rows of the pipe tables of ``text`` with the folded labels of their block's first row."""
    out: list[tuple[list[tuple[int, int]], list[str]]] = []
    if "|" not in text:
        return out
    header: Optional[list[str]] = None
    pos = 0
    n = len(text)
    while pos <= n:
        le = text.find("\n", pos)
        le = n if le < 0 else le
        if "|" in text[pos:le]:
            cells = _row_cells(text, pos, le)
        else:
            cells = None
        if cells is None:
            header = None
        elif _TABLE_RULE_RE.match(text[pos:le]):
            pass
        elif header is None:
            header = [" ".join(fold_word(text[s:e]).split()) for s, e in cells]
        else:
            out.append((cells, header))
        pos = le + 1
    return out


def table_cells(text: str) -> list[tuple[int, int, str]]:
    """(start, end, folded header label) of every non-empty cell of a pipe-table data row ("a | b | c", with or
    without outer pipes). A table is a run of consecutive lines with 2 or more cells; its first row is the header
    (its cells are not returned); a Markdown rule row is skipped. The label is "" past the header's last column."""
    out = []
    for cells, header in _table_rows(text):
        for i, (s, e) in enumerate(cells):
            if s < e:
                out.append((s, e, header[i] if i < len(header) else ""))
    return out


_TH_BIRTH_RE = re.compile(r"naissance|\bddn\b|\bdob\b|\bbirth|^(?:nee?s?|ne\(e\)s?)(?: le)?:?$")
_TH_COMPANY_RE = re.compile(r"\b(?:siret|siren)\b")
_TH_ID_RE = re.compile(
    r"\b(?:matricule|identifiant|references?|ref|dossier|contrat|allocataire|adherent|cotisant|pension|retraite|"
    r"suivi|toque|sinistre|police|immatriculation|plaque|n[°o] (?:de )?(?:dossier|client|contrat|salarie))\b"
)
# v0.2 columns: staff identifiers, account numbers, logins (folded header labels).
_TH_EMPLOYEE_RE = re.compile(
    r"\b(?:matricule|employee(?: id| number| no| #)?|emp id|staff (?:id|number|no)|badge(?: id| number)?|"
    r"personnel number|worker id|payroll(?: id| number)?|n[°o] (?:de )?salarie|identifiant salarie)\b"
)
_TH_ACCOUNT_RE = re.compile(
    r"\b(?:account(?: number| no| #| id)?|acct|a/c|policy (?:number|no)|member(?:ship)? (?:id|number)|"
    r"customer (?:id|account)|client id|loyalty number|routing(?: number)?|aba|(?:n[°o] (?:de )?)?compte(?: client)?|"
    r"n[°o] adherent|n[°o] (?:de )?police|n[°o] societaire)\b"
)
_TH_USERNAME_RE = re.compile(
    r"\b(?:username|user name|login|user id|handle|screen name|identifiant de connexion|nom d'utilisateur|"
    r"pseudo(?:nyme)?)\b"
)
_TH_VESSEL_RE = re.compile(r"\b(?:navires?|bateaux?|vessels?|ships?|chalutiers?|voiliers?|vedettes?|unite navale)\b")
_TH_SCHOOL_RE = re.compile(r"\b(?:ecoles?|etablissement scolaire|colleges?|lycees?|creches?|garderie|scolarite|"
                           r"mode de garde|etablissement d'accueil)\b")
_CELL_DATE_RE = re.compile(_DATE_RE.pattern)


def _cell_labels(cells: list[tuple[int, int]], header: list[str], i: int, text: str) -> list[str]:
    """The column header of cell ``i`` and, in a key-value row ("Date de naissance | 01/02/1990"), the cell on
    its left when that one holds no digit."""
    labels = [header[i]] if i < len(header) else []
    if i >= 1:
        left = text[cells[i - 1][0]:cells[i - 1][1]]
        # a key-value row: two cells, or a left cell written as a form label ("Matricule :")
        if left and len(left) <= 40 and not any(c.isdigit() for c in left) \
                and (len(cells) == 2 or left.rstrip().endswith(":")):
            labels.append(" ".join(fold_word(left).split()).rstrip(" :"))
    return labels


def _find_table_cells(text: str) -> Iterator[Candidate]:
    rows = _table_rows(text)
    for cells, header in rows:
        for i, (s, e) in enumerate(cells):
            if s >= e:
                continue
            labels = _cell_labels(cells, header, i, text)
            if not labels:
                continue
            cell = text[s:e]

            def has(rx: "re.Pattern[str]") -> bool:
                return any(rx.search(lab) for lab in labels)

            if has(_TH_BIRTH_RE):
                for m in _CELL_DATE_RE.finditer(text, s, e):
                    yield Candidate(m.start(), m.end(), "BIRTHDATE", True, PRIORITY["BIRTHDATE"], "table_birthdate")
            if has(_TH_COMPANY_RE):
                for rx, kind, check in ((_SIRET_RE, "SIRET", siret_valid), (_SIREN_RE, "SIREN", siren_valid)):
                    m = rx.fullmatch(text, s, e)
                    if m:
                        yield Candidate(s, e, kind, True, PRIORITY[kind], "table_company_id", checked=check(cell))
                        break
            if has(_TH_ID_RE):
                m = _REF_VALUE_RE.fullmatch(text, s, e)
                if m and sum(c.isdigit() for c in cell) >= 4 and not _DATE_SHAPE_RE.fullmatch(cell) \
                        and not _GROUPED_NUMBER_RE.fullmatch(cell) and not re.fullmatch(r"(?:19|20)\d{2}", cell):
                    yield Candidate(s, e, "ID_CODE", True, PRIORITY["ID_CODE"] - 1, "table_ref")
            for rx_h, kind in ((_TH_EMPLOYEE_RE, "EMPLOYEE_ID"), (_TH_ACCOUNT_RE, "ACCOUNT_NUMBER")):
                if has(rx_h) and _REF_VALUE_RE.fullmatch(text, s, e) and _id_value_ok(cell):
                    yield Candidate(s, e, kind, True, PRIORITY[kind], "table_id")
            if has(_TH_USERNAME_RE):
                um = _USER_VALUE_RE.fullmatch(text, s, e)
                if um and fold_word(cell.lstrip("@")) not in _plain_words():
                    yield Candidate(um.start("v"), um.end("v"), "USERNAME", True, PRIORITY["USERNAME"], "table_id")
            if has(_TH_VESSEL_RE) and not _VESSEL_RE.search(text, s, e) and _cell_name(cell):
                yield Candidate(s, e, "ID_CODE", True, PRIORITY["ID_CODE"] - 1, "table_vessel")
            if has(_TH_SCHOOL_RE) and not _SCHOOL_RE.search(text, s, e) and _cell_name(cell):
                yield Candidate(s, e, "ORG", True, PRIORITY["ORG"] - 1, "table_school")


def _cell_name(cell: str) -> bool:
    """A table cell that is a distinctive proper name (not a number, a date or ordinary words)."""
    if len(cell) > 60 or not cell[:1].isupper() or any(c.isdigit() for c in cell):
        return False
    return _name_ok(cell) and _distinctive(cell)


# ---------------------------------------------------------------------------
# --- en --- US and UK formats (owner: LOCALE-EN, v0.2): NANP and UK phone numbers, US SSN and ITIN, UK NINO,
# US street addresses, PO boxes, "City, ST 12345". Every function here carries ``langs = ("en",)``: the
# registry runs it only on a text detected as English (``detect_candidates(text, lang="en")``, owner DETECTORS;
# ``None`` runs every detector). ``detect_candidates_en`` is the helper used by the tests and the integrator
# until that registry exists. Nothing below this header outside the section belongs to LOCALE-EN.
# ---------------------------------------------------------------------------

_NANP_SEP = r"[ .\- ‑]"
_NANP_RE = re.compile(
    r"(?<![\w+$#])(?<!\d[ .\-/])"
    r"(?:(?:\+?1|001)" + _NANP_SEP + r"?)?"
    r"(?:\(" + _SP + r"?(?P<a1>[2-9]\d{2})" + _SP + r"?\)" + _NANP_SEP + r"?|(?P<a2>[2-9]\d{2})" + _NANP_SEP + r")"
    r"(?P<x>[2-9]\d{2})" + _NANP_SEP + r"?(?P<n>\d{4})"
    r"(?:[ ,]*(?:x|ext\.?|extension|#)[ .]?(?P<ext>\d{1,6}))?"
    r"(?![\w\-]|[ .\-]\d)"
)
_NANP_BARE_RE = re.compile(r"(?<![\w+$#.\-])1?(?P<a>[2-9]\d{2})(?P<x>[2-9]\d{2})(?P<n>\d{4})(?![\w\-])")
_PHONE_LABEL_RE = re.compile(
    r"(?:\b(?:phone|tel|telephone|cell|mobile|fax|call|text|dial|contact|reach|whatsapp|ph|number|num|direct|"
    r"office|home|work|line)\b\.?|#)[^\n]{0,25}$"
)
_UK_PHONE_RE = re.compile(
    r"(?<![\w+])(?<!\d[ .\-])"
    r"(?:\+44[ \- ]?(?:\(0\)[ \- ]?)?|\(0\)[ \- ]?|0)"
    r"(?P<body>(?:1\d{2,3}|2\d|3\d{2}|5\d{2}|7\d{3}|8\d{2})[ \- ]?\d{3,4}[ \- ]?\d{3,4})"
    r"(?![\w\-]|[ .\-]\d)"
)


def _find_phone_en(text: str) -> Iterator[Candidate]:
    """North American (NANP) and UK phone numbers written without a country code ("(415) 555-0134",
    "415-555-0134 x204", "020 7946 0958", "07700 900123"); "+1 ..." and "+44 ..." are the international
    detector's. Ten bare digits are a phone only after a phone label on the line."""
    pr = PRIORITY["PHONE"]
    for m in _NANP_RE.finditer(text):
        yield Candidate(m.start(), m.end(), "PHONE", True, pr, "phone_us")
    for m in _NANP_BARE_RE.finditer(text):
        if _PHONE_LABEL_RE.search(_before(text, m.start(), 40)):
            yield Candidate(m.start(), m.end(), "PHONE", True, pr, "phone_us_bare")
    for m in _UK_PHONE_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group("body"))
        if 9 <= len(digits) <= 10:
            yield Candidate(m.start(), m.end(), "PHONE", True, pr, "phone_uk")


_SSN_RE = re.compile(r"(?<![\w\-])(?P<a>\d{3})(?P<s>[\-‑ ])(?P<g>\d{2})(?P=s)(?P<n>\d{4})(?![\w\-])")
_SSN_BARE_RE = re.compile(r"(?<![\w\-.])(?P<a>\d{3})(?P<g>\d{2})(?P<n>\d{4})(?![\w\-.])")
_SSN_LABEL_RE = re.compile(
    r"(?:\bssn\b|social security|soc\.? ?sec\.?|\bss ?#|\bss no\b|taxpayer id|\btin\b|\bitin\b|tax id)[^\n]{0,30}$"
)
_ITIN_GROUPS = re.compile(r"5\d|6[0-5]|7\d|8[0-8]|9[0-2]|9[4-9]")


def ssn_valid(value: str) -> bool:
    """US Social Security number: 9 digits, area not 000, 666 or 900-999, group not 00, serial not 0000."""
    d = re.sub(r"\D", "", value)
    if len(d) != 9:
        return False
    area, group, serial = d[:3], d[3:5], d[5:]
    return area not in ("000", "666") and area[0] != "9" and group != "00" and serial != "0000"


def itin_valid(value: str) -> bool:
    """US Individual Taxpayer Identification Number: 9xx-(50-65, 70-88, 90-92, 94-99)-xxxx."""
    d = re.sub(r"\D", "", value)
    return len(d) == 9 and d[0] == "9" and bool(_ITIN_GROUPS.fullmatch(d[3:5]))


def _find_ssn(text: str) -> Iterator[Candidate]:
    """SSN (kind NIR, code "SSN_n" in the en locale) and ITIN (kind ID_CODE). The dashed or spaced form is
    enough; nine bare digits need a label ("SSN:", "Social Security number") on the line."""
    for rx, bare in ((_SSN_RE, False), (_SSN_BARE_RE, True)):
        for m in rx.finditer(text):
            v = m.group()
            if bare and not _SSN_LABEL_RE.search(_before(text, m.start(), 50)):
                continue
            if itin_valid(v):
                yield Candidate(m.start(), m.end(), "ID_CODE", True, PRIORITY["ID_CODE"] + 1, "itin")
            elif ssn_valid(v):
                yield Candidate(m.start(), m.end(), "NIR", True, PRIORITY["NIR"], "ssn")


_NINO_RE = re.compile(
    r"(?<![A-Za-z0-9])(?!BG|GB|KN|NK|NT|TN|ZZ)(?P<p>[A-CEGHJ-PR-TW-Z][A-CEGHJ-NPR-TW-Z])(?P<sp>[ \-]?)"
    r"(?P<d1>\d{2})(?P=sp)(?P<d2>\d{2})(?P=sp)(?P<d3>\d{2})(?P=sp)(?P<sfx>[A-D])(?![A-Za-z0-9])"
)
_NINO_LABEL_RE = re.compile(r"(?:national insurance|\bni\b|\bnino\b|n\.i\.)[^\n]{0,30}$")


def nino_valid(value: str) -> bool:
    """UK National Insurance number: two prefix letters (HMRC rules), six digits, suffix A to D."""
    v = re.sub(r"[ \-]", "", value).upper()
    return bool(_NINO_RE.fullmatch(v))


def _find_nino(text: str) -> Iterator[Candidate]:
    for m in _NINO_RE.finditer(text):
        if not m.group("sp") and not _NINO_LABEL_RE.search(_before(text, m.start(), 50)):
            continue  # "AB123456C" without spaces reads as a reference unless labelled
        yield Candidate(m.start(), m.end(), "ID_CODE", True, PRIORITY["ID_CODE"] + 1, "nino")


_US_STATE_NAMES = (
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida",
    "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine",
    "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska",
    "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
    "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas",
    "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming", "District of Columbia",
    "Puerto Rico",
)
_US_STATE_ABBR = frozenset((
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA",
    "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC", "PR",
))
_US_DIRS = r"(?:North|South|East|West|Northeast|Northwest|Southeast|Southwest|N|S|E|W|NE|NW|SE|SW)"
_US_STREET_TYPES = (
    r"(?:Street|St|Avenue|Ave|Av|Boulevard|Blvd|Road|Rd|Drive|Dr|Lane|Ln|Way|Court|Ct|Place|Pl|Parkway|Pkwy|"
    r"Highway|Hwy|Terrace|Ter|Terr|Circle|Cir|Loop|Trail|Trl|Square|Sq|Plaza|Plz|Alley|Aly|Turnpike|Tpke|Pike|"
    r"Run|Path|Row|Crossing|Xing|Bend|Cove|Cv|Point|Pt|Ridge|Rdg|Walk|Grove|Grv|Gardens|Gdns|Close|Crescent|"
    r"Cres|Mews|Broadway|Expressway|Expy|Freeway|Fwy|Causeway|Cswy|Esplanade|Promenade|Commons|Landing|Lndg|"
    r"Manor|Mnr|Meadows|Mdws|Pass|Pathway|Shore|Shores|Summit|Trace|Trce|View|Vista|Valley|Vly|Village|Vlg|"
    r"Heights|Hts|Estates|Ests|Harbor|Hbr|Junction|Jct|Knoll|Mountain|Mtn|Orchard|Overlook|Pines|Plains|"
    r"Rapids|Springs|Spgs|Woods|Hill|Park|Lake)"
)
_US_ADDR_RE = re.compile(
    r"(?<![\w\-/$#.])(?P<num>\d{1,6}[A-Za-z]?(?:-\d{1,4})?)[  ]+"
    r"(?:(?P<dir1>" + _US_DIRS + r")\.?[  ]+)?"
    r"(?P<name>(?:(?:[A-Z][\w'’\-]*|\d{1,3}(?:st|nd|rd|th)|of|the|and|de|la|del|du|des)[  ]+){0,5}?)"
    r"(?P<type>" + _US_STREET_TYPES + r")\.?"
    r"(?:[  ]+(?P<dir2>" + _US_DIRS + r")\.?)?"
    r"(?![\w])"
)
_US_UNIT_RE = re.compile(
    r",?[ \t ]*(?:(?i:suite|ste\.?|apt\.?|apartment|unit|floor|fl\.?|bldg\.?|building|room|rm\.?|no\.?)"
    r"[ \t ]*#?|#[ \t ]*)(?P<id>[\w\-]{1,8})(?![\w])"
)
_US_CITY_RE = re.compile(r"(?P<city>[A-Z][\w'’\-]*(?:[  ][A-Z][\w'’\-]*){0,3})")
_US_STATE_RE = re.compile(r"(?P<state>" + "|".join(sorted(_US_STATE_NAMES, key=len, reverse=True)) + r"|[A-Z]{2})(?![\w])")
_US_ZIP_RE = re.compile(r"(?P<zip>\d{5}(?:-\d{4})?)(?![\w\-])")
_US_TAIL_SEP_RE = re.compile(r",?[ \t ]*(?:\r?\n[ \t ]*)?")
_US_LINE_SEP_RE = re.compile(r",?[ \t ]*(?:\r?\n[ \t ]*)?")
_PO_BOX_RE = re.compile(r"(?i)(?<![\w])(?:P\.?[  ]?O\.?|Post[  ]Office)[  ]?Box[  ]+(?P<n>\d{1,7})(?![\w])")
_CITY_STATE_ZIP_RE = re.compile(
    r"(?<![\w,])(?P<city>[A-Z][\w'’\-]*(?:[  ][A-Z][\w'’\-]*){0,3}),[  ]*(?P<st>[A-Z]{2})[  ]+"
    r"(?P<zip>\d{5}(?:-\d{4})?)(?![\w\-])"
)
_ADDR_LABEL_RE = re.compile(
    r"(?:address|addr|located|residing|resides|lives|living|ship to|mail to|send to|deliver to|office|home|at|"
    r"to|from|hq|headquarters|premises|property)[^\n]{0,20}$"
)
_ADDR_CITY_STOP = frozenset({"the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "by", "with",
                             "is", "was", "are", "were", "be", "he", "she", "it", "we", "they", "you", "i"})


def _state_zip_at(text: str, p: int) -> Optional[int]:
    """End of "IL 62704", "Illinois 62704", "Illinois" or "IL" (an abbreviation needs a ZIP or a line end) at ``p``."""
    st = _US_STATE_RE.match(text, p)
    if not st:
        return None
    name = st.group("state")
    if len(name) == 2 and name not in _US_STATE_ABBR:
        return None
    end = st.end()
    z = re.match(r"[ \t\u00a0,]+", text[end:end + 4])
    if z:
        zm = _US_ZIP_RE.match(text, end + z.end())
        if zm:
            return zm.end()
    if len(name) > 2 or text[end:end + 1] in ("", "\n", "\r", ",", ".", ";"):
        return end
    return None


def _us_tail(text: str, pos: int) -> int:
    """Extend a US street address or PO box over a unit, then "City, ST 12345" (same line or the next line),
    or a state alone on the next line."""
    end = pos
    for _ in range(2):
        u = _US_UNIT_RE.match(text, end)
        if not u:
            break
        end = u.end()
    sep = _US_TAIL_SEP_RE.match(text, end)
    p = sep.end() if sep else end
    newline = "\n" in text[end:p]
    c = _US_CITY_RE.match(text, p)
    if c and fold_word(c.group("city").split()[0]) not in _ADDR_CITY_STOP:
        q = c.end()
        sep2 = re.match(r",?[ \t\u00a0]+", text[q:q + 6])
        if sep2:
            st = _state_zip_at(text, q + sep2.end())
            if st:
                return st  # "Springfield, IL 62704", "New York, New York 10036"
        if not newline and text[q:q + 1] in (",", "\n", "\r", ""):
            return q  # "123 Main St, Springfield," : the city alone, on the same line
    st = _state_zip_at(text, p)  # "9 Elm St.\nIL 62704": a state alone on the next line
    return st if st else end


def _find_address_us(text: str) -> Iterator[Candidate]:
    """US street addresses ("742 Evergreen Terrace, Suite 4, Springfield, IL 62704"), PO boxes and a bare
    "City, ST 12345" line. A year-like house number needs a tail (city or state) or an address label. A bare
    ZIP code is never masked."""
    pr = PRIORITY["ADDRESS"]
    for m in _US_ADDR_RE.finditer(text):
        end = _us_tail(text, m.end())
        num = m.group("num")
        if re.fullmatch(r"(?:19|20)\d{2}", num) and end == m.end() \
                and not _ADDR_LABEL_RE.search(_before(text, m.start(), 30)) \
                and "\n" not in text[max(0, m.start() - 2):m.start()]:
            continue
        if not m.group("name").strip() and not m.group("dir1"):
            continue  # "5 Street" alone
        yield Candidate(m.start(), end, "ADDRESS", True, pr, "address_us")
    for m in _PO_BOX_RE.finditer(text):
        yield Candidate(m.start(), _us_tail(text, m.end()), "ADDRESS", True, pr, "po_box")
    for m in _CITY_STATE_ZIP_RE.finditer(text):
        if m.group("st") in _US_STATE_ABBR and fold_word(m.group("city").split()[0]) not in _ADDR_CITY_STOP:
            yield Candidate(m.start(), m.end(), "ADDRESS", True, pr - 1, "city_state_zip")


DETECTORS_EN: tuple[Callable[[str], Iterable[Candidate]], ...] = (
    _find_phone_en, _find_ssn, _find_nino, _find_address_us,
)
for _f in DETECTORS_EN:
    _f.langs = ("en",)  # type: ignore[attr-defined]


def detect_candidates_en(text: str) -> list[Candidate]:
    """Every candidate for an English text: ``detect_candidates(text, lang="en")`` (kept as a named helper for
    the tests and the runners)."""
    return detect_candidates(text, lang="en")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

DETECTORS: tuple[Callable[[str], Iterable[Candidate]], ...] = (
    _find_email, _find_phone, _find_iban, _find_nir, _find_siret, _find_siren, _find_vat,
    _find_card, _find_case_ref, _find_id_code, _find_url, _find_birthdate, _find_birthplace, _find_address,
    _find_named_places, _find_table_cells,
    # v0.2 (both languages)
    _find_mentions, _find_handles, _find_labelled_ids, _find_routing,
)


@lru_cache(maxsize=1)
def _en_detectors() -> tuple:
    """The detectors of ``detectors_en.py`` (LOCALE-EN builder: US and UK formats, each with ``langs``), when that
    module exists. A detector function may carry a ``langs`` attribute (tuple of language codes); without one it
    runs for every language."""
    found = globals().get("DETECTORS_EN")
    if found:
        return tuple(found)  # the "# --- en ---" section of this module
    try:
        from .detectors_en import DETECTORS_EN as from_module  # type: ignore
    except ImportError:
        return ()
    return tuple(from_module)


def detect_candidates(text: str, lang: Optional[str] = None,
                      shapes: Iterable[tuple[str, str]] = ()) -> list[Candidate]:
    """Every candidate from every structured detector (valid and invalid), unresolved.

    ``lang`` ("fr", "en") skips the detectors whose ``langs`` attribute does not hold it; ``None`` (the canary)
    runs every detector. ``shapes`` are the (kind, template) id layouts learnt in the dossier (see
    :func:`shape_template`): their unlabelled matches are added (v0.2)."""
    out: list[Candidate] = []
    for det in DETECTORS + _en_detectors():
        langs = getattr(det, "langs", None)
        if lang is not None and langs is not None and lang not in langs:
            continue
        out.extend(det(text))
    if shapes:
        out.extend(_find_shapes(text, shapes))
    return out


def _select(cands: Iterable[Candidate]) -> list[Candidate]:
    """Longest first, then priority; drop overlaps."""
    taken = IntervalSet()
    out = []
    for c in sorted(cands, key=lambda c: (-(c.end - c.start), -c.priority, c.start)):
        if c.end <= c.start or taken.overlaps(c.start, c.end):
            continue
        taken.add(c.start, c.end)
        out.append(c)
    out.sort(key=lambda c: c.start)
    return out


def detect_structured(text: str, candidates: Optional[list[Candidate]] = None) -> list[Span]:
    """Valid structured identifiers as non-overlapping spans (source "regex"): the masked candidates whose check
    digit passes or that have none. The masker also masks the exact-shape or labelled identifiers whose check
    digit fails (``valid and not checked``); :func:`find_suspects` lists those."""
    cands = candidates if candidates is not None else detect_candidates(text)
    return [Span(c.start, c.end, c.kind, text[c.start:c.end], "regex")
            for c in _select(c for c in cands if c.valid and c.checked)]


def find_suspects(text: str, candidates: Optional[list[Candidate]] = None) -> list[Span]:
    """Numbers that look like a checksummed identifier but fail the check digit.

    The review list can show them ("suspect"). Most are not masked; an exact-shape IBAN or a SIREN / SIRET
    with its label on the line is masked all the same and still listed here. A suspect that overlaps a valid
    identifier is not reported.
    """
    cands = candidates if candidates is not None else detect_candidates(text)
    valid = IntervalSet()
    for c in _select(c for c in cands if c.valid and c.checked):
        valid.add(c.start, c.end)
    out = []
    for c in _select(c for c in cands if not (c.valid and c.checked)):
        if valid.overlaps(c.start, c.end):
            continue
        if c.kind in ("SIREN", "SIRET") and not _COMPANY_ID_CTX.search(_before(text, c.start, 50)):
            continue  # a bare 9 or 14 digit number is too common to flag without a keyword
        out.append(Span(c.start, c.end, c.kind, text[c.start:c.end], "suspect"))
    return out


_COMPANY_ID_CTX = re.compile(r"(?:siren|siret|\brcs\b|r\.c\.s|immatricul|registre du commerce|nic\b)")


# ---------------------------------------------------------------------------
# Canonical keys for structured kinds
# ---------------------------------------------------------------------------

_MONTH_NUM = {
    "janvier": 1, "janv": 1, "fevrier": 2, "fevr": 2, "fev": 2, "mars": 3, "avril": 4, "avr": 4,
    "mai": 5, "juin": 6, "juillet": 7, "juil": 7, "aout": 8, "septembre": 9, "sept": 9,
    "octobre": 10, "oct": 10, "novembre": 11, "nov": 11, "decembre": 12, "dec": 12,
    # English (v0.2)
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4, "may": 5,
    "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9,
    "october": 10, "november": 11, "december": 12,
}


def _date_key(value: str) -> Optional[str]:
    f = fold_word(value).replace("1er", "1")
    m = re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+)\.?\s+(\d{4})", f)
    try:
        if m:
            d, mo, y = int(m.group(1)), _MONTH_NUM.get(m.group(2)), int(m.group(3))
            if mo:
                return date(y, mo, d).isoformat()
        m = re.fullmatch(r"([a-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", f)  # "March 14, 1979"
        if m:
            mo, d, y = _MONTH_NUM.get(m.group(1)), int(m.group(2)), int(m.group(3))
            if mo:
                return date(y, mo, d).isoformat()
        m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", f)
        if m:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat()
        m = re.fullmatch(r"(\d{1,2})\s?[/.\-]\s?(\d{1,2})\s?[/.\-]\s?(\d{2,4})", f)
        if m:
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if y < 100:
                y += 1900 if y > 30 else 2000
            return date(y, mo, d).isoformat()
    except ValueError:
        return None
    return None


def structured_norm(kind: str, value: str) -> str:
    """Canonical key of a structured value, so different spellings map to one entity."""
    v = value.strip()
    if kind == "EMAIL":
        return v.lower()
    if kind == "PHONE":
        digits = re.sub(r"\D", "", v)
        if v.startswith("+33") or v.startswith("0033"):
            nat = digits[2:] if v.startswith("+") else digits[4:]
            if nat.startswith("0"):
                nat = nat[1:]
            return "0" + nat
        if v.startswith("+"):
            return "+" + digits
        if v.startswith("00"):
            return "+" + digits[2:]
        return digits
    if kind in ("IBAN", "NIR", "SIREN", "SIRET", "VAT", "CARD"):
        return re.sub(r"[^0-9A-Za-z]", "", v).upper()
    if kind == "CASE_REF":
        return re.sub(r"\s+", "", v).upper()
    if kind == "ID_CODE":
        if ":" in v and "." not in v:
            try:
                return str(ipaddress.IPv6Address(v))
            except ValueError:
                pass
        return re.sub(r"[\s\-]", "", v).upper()
    if kind == "USERNAME":
        return v.lstrip("@").lower()
    if kind in ("EMPLOYEE_ID", "ACCOUNT_NUMBER"):
        return re.sub(r"[\s\-./_]", "", v).upper()
    if kind == "URL":
        u = v.lower()
        u = re.sub(r"^[a-z][a-z0-9+.\-]*://", "", u)
        if u.startswith("www."):
            u = u[4:]
        return u.rstrip("/")
    if kind == "BIRTHDATE":
        return _date_key(v) or fold_word(v)
    if kind == "ADDRESS":
        return " ".join(re.findall(r"[^\W_]+", fold_word(v)))
    return fold_word(v)


# S6 (owner: ids, 2026-10-20): account and staff identifier cue rules, in-text id layouts, column ids, street edges.
from .detectors_ids import DETECTORS_IDS as _DETECTORS_IDS  # noqa: E402  (imports this module: keep it last)
DETECTORS = DETECTORS + _DETECTORS_IDS
