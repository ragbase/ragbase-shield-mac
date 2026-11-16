"""Per-dossier directory (v0.3, step S3): the people a dossier declares outside its prose, and the e-mail domain
labels of its correspondents.

A workspace export, a mailbox or a Word file names its people in places the name rules never read: e-mail headers
(From, To, Cc, Bcc, Reply-To), the ``users.json`` of a Slack export (real name, display name, handle, e-mail),
the ``docProps/core.xml`` of a Word document (author, last modified by), the signature block of a message and a
declaration such as 'Constantine "Kit" Varlamov' or 'Constantine Varlamov (goes by Kit)'. Read before the first
text is masked, these give:

- the role of every name token (given, family, nickname), so that "Varlamov, Kit" is read in the right order;
- the full name behind a first name, a nickname or a handle, so that "kit" in a chat line or "@kit.varlamov"
  is the person Constantine Varlamov;
- every real name before the first mention is masked (the store is seeded, :meth:`Directory.seed`);
- the organisation labels of the e-mail domains ("northvale" from ``northvale.example``), as evidence for the
  kind decision of the masker (a word that is both a first name and the company of the dossier).

Everything is deterministic, local and standard library only: no model, no network. The directory holds REAL
values: it lives in the vault (encrypted) or in memory, and never in a model-facing output.

API::

    from ragbase_shield.engine import directory
    d = directory.build([path_to_eml, path_to_users_json, path_to_docx, ("text", body_text)])
    d.lookup("kit")              # [Match(person=<Constantine Varlamov>, role="nick")]
    d.person_roles("Varlamov, Kit")  # [("Varlamov", "family"), ("Kit", "nick")]
    d.seed(store)                # register every full name in an EntityStore
    d.to_dict() / Directory.from_dict(data)
"""
from __future__ import annotations

import email
import email.policy
import email.utils
import json
import logging
import os
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Union
import xml.etree.ElementTree as ET

from . import names as nm
from .normalize import fold_word, phrase_tokens

__all__ = ["Directory", "Person", "Match", "build", "domain_label", "PUBLIC_MAIL_HOSTS", "ROLES"]

log = logging.getLogger(__name__)

#: Token roles of a person name (``Person``, :meth:`Directory.person_roles`).
ROLES = ("given", "family", "nick", "initial", "particle", "title", "other")

#: Mail hosts whose domain label names no organisation (free or personal mail). Written for RAGbase Shield.
PUBLIC_MAIL_HOSTS = frozenset({
    "gmail", "googlemail", "yahoo", "ymail", "rocketmail", "hotmail", "outlook", "live", "msn", "aol", "icloud",
    "me", "mac", "protonmail", "proton", "pm", "gmx", "mail", "email", "yandex", "zoho", "fastmail", "hey",
    "orange", "free", "wanadoo", "laposte", "sfr", "bbox", "neuf", "numericable", "aliceadsl", "club-internet",
    "qq", "163", "126", "sina", "naver", "daum", "web", "t-online", "libero", "virgilio", "tiscali", "bluewin",
    "skynet", "telenet", "ziggo", "kpnmail", "example", "invalid", "localhost", "test",
})
# Generic second-level labels under a two-letter country code ("acme.co.uk", "acme.com.au", "acme.gouv.fr"): the
# organisation label sits one level deeper.
_GENERIC_SECOND_LEVEL = frozenset({"co", "com", "org", "net", "ac", "gov", "gouv", "edu", "ne", "or", "asso", "ltd",
                                   "plc", "me", "nom", "info", "biz", "sch", "mil", "id", "in", "nhs", "police"})
_SOURCE_ORDER = ("header", "slack", "docx", "signature", "declared")

_NAME_TOKEN = r"[A-ZÀ-ÖØ-ÞŒ][\w'’\-]*"
_QUOTE_OPEN = "[\"“'‘]"
_QUOTE_CLOSE = "[\"”'’]"
# 'Constantine "Kit" Varlamov', "Constantine 'Kit' Varlamov", "Constantine (Kit) Varlamov"
_NICK_INSIDE_RE = re.compile(
    r"(?<![\w'’\-])(?P<first>" + _NAME_TOKEN + r")[  ]+(?:" + _QUOTE_OPEN + r"(?P<n1>" + _NAME_TOKEN
    + r")" + _QUOTE_CLOSE + r"|\((?P<n2>" + _NAME_TOKEN + r")\))[  ]+(?P<last>" + _NAME_TOKEN
    + r"(?:[  ]+" + _NAME_TOKEN + r")?)(?![\w'’\-])")
# 'Constantine Varlamov ("Kit")', 'Constantine Varlamov (Kit)', 'Constantine Varlamov (goes by Kit)',
# 'Constantine Varlamov, who goes by Kit', 'Constantine Varlamov, aka Kit', 'Constantine Varlamov (call me Kit)'.
# Two forms only: a bracket that closes right after the nickname ("(Kit)", never "(Kit controller)", which is a
# role or an affiliation), or a declaring keyword after the name. "First Last Nick" with nothing between the words
# is not a declaration.
_NICK_KEYWORD = (r"(?:goes[  ]+by|a\.?k\.?a\.?|also[  ]+known[  ]+as|known[  ]+as|call[  ]+(?:me|him|her)"
                 r"|prefers(?:[  ]+to[  ]+be[  ]+called)?|nicknamed?|dit|dite)")
_NICK_AFTER_RE = re.compile(
    r"(?<![\w'’\-])(?P<first>" + _NAME_TOKEN + r")[  ]+(?P<mid>(?:" + _NAME_TOKEN + r"[  ]+){0,2})"
    r"(?P<last>" + _NAME_TOKEN + r")(?![\w'’\-])"
    r"(?:"
    r"[  ]*\((?:" + _NICK_KEYWORD + r"[  ]+)?(?:" + _QUOTE_OPEN + r")?(?P<nick1>" + _NAME_TOKEN + r")"
    r"(?:" + _QUOTE_CLOSE + r")?(?:[  ]+for[  ]+short)?[  ]*\)"
    r"|"
    r",?[  ]+(?:who[  ]+)?" + _NICK_KEYWORD + r"[  ]+(?:" + _QUOTE_OPEN + r")?(?P<nick2>" + _NAME_TOKEN + r")"
    r"(?:" + _QUOTE_CLOSE + r")?(?:[  ]+for[  ]+short)?"
    r")(?![\w'’\-])")
# A declaring keyword as a whole word (never the "aka" inside a surname).
_NICK_KEYWORDS_RE = re.compile(
    r"(?i)(?<![\w])(?:goes[  ]+by|a\.?k\.?a\.?|known[  ]+as|call[  ]+(?:me|him|her)|prefers|nicknamed?"
    r"|for[  ]+short|dite?)(?![\w])")
_SIGNOFF_RE = re.compile(
    r"(?i)^[ \t]*(?:thanks(?:[ \t]+(?:again|a[ \t]+lot|so[ \t]+much))?|thank[ \t]+you(?:[ \t]+(?:again|both|all))?|"
    r"many[ \t]+thanks|best(?:[ \t]+(?:regards|wishes))?|(?:kind|warm|warmest|kindest)[ \t]+regards|regards|"
    r"cheers|sincerely(?:[ \t]+yours)?|yours(?:[ \t]+(?:sincerely|truly|faithfully))?|all[ \t]+the[ \t]+best|"
    r"take[ \t]+care|talk[ \t]+soon|warmly|respectfully|cordially|cordialement|bien[ \t]+cordialement|"
    r"bien[ \t]+(?:à|a)[ \t]+(?:vous|toi)|sinc[èe]rement|amicalement|amiti[ée]s|(?:à|a)[ \t]+bient[ôo]t|"
    r"bonne[ \t]+(?:journ[ée]e|soir[ée]e)|salutations(?:[ \t]+distingu[ée]es)?|--|—)[ \t]*[,.!]?[ \t]*$")
_NAME_LINE_RE = re.compile(r"^[ \t]*(?P<name>" + _NAME_TOKEN + r"(?:[  ]+(?:" + _NAME_TOKEN
                           + r"|[A-Z]\.|de|van|von|der|du|da|di|la|le))?(?:[  ]+" + _NAME_TOKEN
                           + r"){1,2})[ \t]*(?:,.*)?$")
_LOCAL_SPLIT_RE = re.compile(r"[._\-]+")
_WORD_RE = re.compile(r"[^\W\d_][\w'’\-]*")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Person:
    """One person of the directory. Token fields are folded (lower case, no accents); ``value`` is the display
    form of the fullest name seen."""

    value: str
    given: tuple[str, ...] = ()
    family: tuple[str, ...] = ()
    nicknames: tuple[str, ...] = ()
    handles: tuple[str, ...] = ()
    emails: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()

    @property
    def norm(self) -> str:
        from .masker import entity_norm  # late import: masker imports this module

        return entity_norm("PERSON", self.value)

    @property
    def full(self) -> bool:
        return bool(self.given) and bool(self.family)

    def tokens(self) -> frozenset[str]:
        return frozenset(self.given) | frozenset(self.family) | frozenset(self.nicknames)

    def to_dict(self) -> dict:
        return {"value": self.value, "given": list(self.given), "family": list(self.family),
                "nicknames": list(self.nicknames), "handles": list(self.handles), "emails": list(self.emails),
                "sources": list(self.sources)}

    @classmethod
    def from_dict(cls, d: dict) -> "Person":
        def strs(key: str) -> tuple[str, ...]:
            v = d.get(key) or ()
            return tuple(str(x) for x in v if isinstance(x, str) and x)

        value = d.get("value")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("directory person without a value")
        return cls(value.strip(), strs("given"), strs("family"), strs("nicknames"), strs("handles"),
                   strs("emails"), strs("sources"))


@dataclass(frozen=True)
class Match:
    """A directory hit for a surface: the person and the role the surface plays (``full``, ``given``,
    ``family``, ``nick``, ``handle``, ``email``)."""

    person: Person
    role: str


class _Draft:
    """A person under construction (mutable; merged by key, e-mail or handle)."""

    __slots__ = ("values", "given", "family", "nicknames", "handles", "emails", "sources", "order")

    def __init__(self, order: int) -> None:
        self.values: list[str] = []
        self.given: list[str] = []
        self.family: list[str] = []
        self.nicknames: set[str] = set()
        self.handles: set[str] = set()
        self.emails: set[str] = set()
        self.sources: set[str] = set()
        self.order = order

    def freeze(self) -> Optional[Person]:
        value = max(self.values, key=lambda v: ("," not in v, len(_WORD_RE.findall(v)), -self.values.index(v))) \
            if self.values else ""
        if not value:
            return None
        given = tuple(dict.fromkeys(self.given))
        family = tuple(dict.fromkeys(self.family))
        nicks = tuple(sorted(n for n in self.nicknames if n and n not in given and n not in family))
        return Person(value, given, family, nicks, tuple(sorted(self.handles)), tuple(sorted(self.emails)),
                      tuple(s for s in _SOURCE_ORDER if s in self.sources))


# ---------------------------------------------------------------------------
# Name parsing
# ---------------------------------------------------------------------------


def _clean_display(name: str) -> str:
    name = name.strip().strip("\"'“”‘’").strip()
    name = re.sub(r"\s+", " ", name)
    # a display name that is a login name ("first.last", "f_last") is not a name
    if "@" in name or not any(c.isalpha() for c in name):
        return ""
    if not any(c.isupper() for c in name) and re.fullmatch(r"[\w.\-]+", name) and re.search(r"[._]", name):
        return ""
    return name


def _first_name_ok(tok: str, lang: str) -> bool:
    """``tok`` (folded) is a first name of the language ("auto": either lexicon)."""
    if lang in ("en", "auto"):
        try:
            from . import locale as lc

            if tok in lc.bundle("en").all_first_names:
                return True
        except Exception:  # noqa: BLE001 - no English bundle: the French lexicon alone
            pass
    if lang in ("fr", "auto"):
        return tok in nm._lists()[4]
    return False


def _vocab_word(tok: str, lang: str) -> bool:
    """A common word of the language (never a family name by itself without evidence)."""
    if lang in ("en", "auto"):
        try:
            from . import locale as lc

            b = lc.bundle("en")
            if tok in b.vocab or tok in b.stopwords or tok in b.greetings or tok in b.months_days \
                    or tok in b.legal_terms or tok in b.roles or tok in b.titles:
                return True
        except Exception:  # noqa: BLE001
            pass
    if lang in ("fr", "auto"):
        stop, legal, common, _fn, _afn, months, _wl = nm._lists()
        return tok in stop or tok in legal or tok in months
    return False


def _parts(word: str) -> list[str]:
    """Folded, hyphen-split tokens of one name word ("Okoro-Sand" -> ["okoro", "sand"])."""
    return [t for t in phrase_tokens(word) if t]


def split_name(value: str, lang: str = "auto", local_part: str = "") -> tuple[list[str], list[str]]:
    """(given tokens, family tokens), folded and hyphen-split, of a display name.

    "Last, First Middle" (a comma) is read in that order. A name that mixes an upper-case surname with capitalised
    first names follows :func:`names.split_person`. Otherwise the last word is the family name (with the particle
    before it) when the first word is a first name of the lexicon or the last one is not; a name whose last word
    is a first name and whose first word is not is in surname-first order ("Okoro Wilhelmina"). The local part of an
    e-mail address ("first.last") then corrects the split: the token that ends the local part is the family name."""
    value = _clean_display(value)
    if not value:
        return [], []
    if "," in value:
        before, _, after = value.partition(",")
        fam = [t for w in _WORD_RE.findall(before) for t in _parts(w)]
        giv = [t for w in _WORD_RE.findall(after) for t in _parts(w) if fold_word(w) not in nm.GENERATIONAL]
        if fam and giv and not all(nm.is_title_word(w) for w in before.split()):
            return giv, fam
    words = nm._strip_titles(nm._name_words(value))
    words = [w for w in words if w.text and w.low not in nm.GENERATIONAL and not w.prefix]
    if not words:
        return [], []
    if len(words) == 1:
        return [], _parts(words[0].text)
    mixed = any(w.upper and not w.initial for w in words) and any(w.cap and not w.upper and not w.initial
                                                                   and w.low not in nm.PARTICLES for w in words)
    if mixed:
        firsts, surname = nm.split_person(value)
        giv = [t for w in firsts for t in _parts(w)]
        fam = [t for w in surname for t in _parts(w)]
    else:
        first_ok = _first_name_ok(fold_word(words[0].text), lang) or words[0].initial
        last_ok = _first_name_ok(fold_word(words[-1].text), lang)
        if first_ok or not last_ok or len(words) > 3:
            k = len(words) - 1
            while k > 1 and words[k - 1].low in nm.PARTICLES:
                k -= 1
            giv = [t for w in words[:k] for t in _parts(w.text)]
            fam = [t for w in words[k:] for t in _parts(w.text)]
        else:
            k = 1
            while k < len(words) - 1 and words[k].low in nm.PARTICLES:
                k += 1
            fam = [t for w in words[:k] for t in _parts(w.text)]
            giv = [t for w in words[k:] for t in _parts(w.text)]
    giv = [g for g in giv if g]
    fam = [f for f in fam if f]
    if local_part and giv and fam:
        parts = [p for p in _LOCAL_SPLIT_RE.split(fold_word(local_part)) if p]
        toks = giv + fam
        if len(parts) >= 2 and parts[-1] in toks and parts[0] in toks and parts[-1] != parts[0] \
                and toks.index(parts[0]) < toks.index(parts[-1]):
            k = toks.index(parts[-1])
            if k > 0:
                giv, fam = toks[:k], toks[k:]
    return giv, fam


def domain_label(address_or_host: str) -> str:
    """Organisation label of an e-mail address or host ("northvale" from "jane@mail.northvale.example",
    "acme" from "acme.co.uk"); "" for a public mail host or an invalid value."""
    host = address_or_host.strip().lower()
    if "@" in host:
        host = host.rsplit("@", 1)[1]
    host = host.strip("<>[]() \t.")
    if not host or " " in host or "/" in host:
        return ""
    labels = [p for p in host.split(".") if p]
    if len(labels) < 2:
        return ""
    if len(labels) >= 3 and len(labels[-1]) == 2 and labels[-2] in _GENERIC_SECOND_LEVEL:
        label = labels[-3]
    else:
        label = labels[-2]
    label = fold_word(label)
    if not label or label in PUBLIC_MAIL_HOSTS or not any(c.isalpha() for c in label):
        return ""
    return label


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


DocItem = Union[str, "os.PathLike[str]", tuple]


class _Builder:
    def __init__(self, lang: str) -> None:
        self.lang = lang
        self.drafts: list[_Draft] = []
        self.by_key: dict[tuple, _Draft] = {}
        self.by_email: dict[str, _Draft] = {}
        self.by_handle: dict[str, _Draft] = {}
        self.domains: dict[str, int] = {}
        self.texts: list[str] = []
        # v0.4 (T3): the distinct local parts seen per domain label (a company mailbox needs a shared domain)
        self.locals_by_label: dict[str, set[str]] = {}

    # -- persons -----------------------------------------------------------------------------------------

    def add_person(self, value: str, *, source: str, email_addr: str = "", handle: str = "",
                   given: Optional[list[str]] = None, family: Optional[list[str]] = None,
                   nicknames: Iterable[str] = ()) -> Optional[_Draft]:
        value = _clean_display(value)
        email_addr = email_addr.strip().lower() if email_addr else ""
        local = email_addr.split("@", 1)[0] if "@" in email_addr else ""
        if given is None or family is None:
            given, family = split_name(value, self.lang, local)
        if not value and not email_addr and not handle:
            return None
        key = (tuple(family), given[0] if given else "")
        d = None
        if email_addr and email_addr in self.by_email:
            d = self.by_email[email_addr]
        if d is None and handle and fold_word(handle) in self.by_handle:
            d = self.by_handle[fold_word(handle)]
        if d is None and family:
            d = self.by_key.get(key)
            if d is None:
                # "Octavia Brennholt" and "Octavia Maud Brennholt": one person (compatible given names)
                for cand in self.drafts:
                    if cand.family == list(family) and given and cand.given \
                            and (cand.given[:len(given)] == given or given[:len(cand.given)] == cand.given):
                        d = cand
                        break
                if d is None and given:
                    for cand in self.drafts:
                        if cand.family == list(family) and cand.given and given[0] in cand.nicknames:
                            d = cand
                            break
        if d is None:
            d = _Draft(len(self.drafts))
            self.drafts.append(d)
        if value and value not in d.values:
            d.values.append(value)
        if family and not d.family:
            d.family = list(family)
        if given:
            if not d.given:
                d.given = list(given)
            elif len(given) > len(d.given) and given[:len(d.given)] == d.given:
                d.given = list(given)
            elif given[0] != d.given[0] and family == d.family:
                d.nicknames.add(given[0])
        if email_addr:
            d.emails.add(email_addr)
            self.by_email[email_addr] = d
            self.count_domain(email_addr)
            if local:
                h = fold_word(local)
                if h and h not in d.handles:
                    d.handles.add(h)
                    self.by_handle[h] = d
                self.link_local_part(d, local)
        if handle:
            h = fold_word(handle).lstrip("@")
            if h:
                d.handles.add(h)
                self.by_handle[h] = d
        for n in nicknames:
            n = fold_word(n)
            if n and n not in d.given and n not in d.family:
                d.nicknames.add(n)
        d.sources.add(source)
        if family:
            self.by_key.setdefault(key, d)
        return d

    def link_local_part(self, d: _Draft, local: str) -> None:
        """"rupe" of "rupe.haldane@..." is a nickname of Rupert; "trudy" of "trudy.steinhauer" of Gertrude."""
        parts = [p for p in _LOCAL_SPLIT_RE.split(fold_word(local)) if p and p.isalpha()]
        if not parts or not d.given:
            return
        first = parts[0]
        if len(first) < 2 or first == d.given[0] or first in d.given or first in d.family:
            return
        if first in nm._nick_forms(d.given[0]) or _first_name_ok(first, self.lang) \
                or (len(first) >= 3 and d.given[0].startswith(first)):
            d.nicknames.add(first)

    def count_domain(self, address: str) -> None:
        label = domain_label(address)
        if label:
            self.domains[label] = self.domains.get(label, 0) + 1
            if "@" in address:
                self.locals_by_label.setdefault(label, set()).add(address.split("@", 1)[0].strip().lower())

    def company_mailbox(self, address: str) -> bool:
        """v0.4 (T3): the address is the mailbox of the organisation of its own domain, not a person: its
        local-part words, joined, spell the domain label ("north.vale@northvale.example", "acme@acme.example"),
        and the dossier holds at least two other addresses at that domain (a shared company domain, not a
        personal one)."""
        return _mailbox_spells_label(address) \
            and len(self.locals_by_label.get(domain_label(address), ())) >= 3

    # -- sources -----------------------------------------------------------------------------------------

    def add_eml(self, raw: bytes) -> None:
        msg = email.message_from_bytes(raw, policy=email.policy.default)
        for header in ("From", "Reply-To", "Sender", "To", "Cc", "Bcc", "Resent-From", "Resent-To"):
            try:
                values = msg.get_all(header) or []
            except Exception:  # noqa: BLE001 - an unreadable header
                continue
            for name, addr in email.utils.getaddresses([str(v) for v in values]):
                addr = addr.strip().lower()
                if addr and "@" not in addr:
                    addr = ""
                if not name and not addr:
                    continue
                self.add_person(name, source="header", email_addr=addr)
        try:
            body = msg.get_body(preferencelist=("plain", "html"))
            content = body.get_content() if body is not None else ""
        except Exception:  # noqa: BLE001
            content = ""
        if content and isinstance(content, str):
            self.texts.append(content)

    def add_slack_users(self, data: object) -> None:
        users = data.get("users") if isinstance(data, dict) else data
        if not isinstance(users, list):
            return
        for u in users:
            if not isinstance(u, dict) or u.get("is_bot") or u.get("is_app_user"):
                continue
            prof = u.get("profile") if isinstance(u.get("profile"), dict) else {}
            real = str(u.get("real_name") or prof.get("real_name") or prof.get("real_name_normalized") or "")
            display = str(prof.get("display_name") or prof.get("display_name_normalized") or "")
            handle = str(u.get("name") or "")
            addr = str(prof.get("email") or "").strip().lower()
            if "@" not in addr:
                addr = ""
            first = str(prof.get("first_name") or "")
            last = str(prof.get("last_name") or "")
            value = real or (first + " " + last).strip()
            if not value and not addr and not handle:
                continue
            d = self.add_person(value, source="slack", email_addr=addr, handle=handle)
            if d is None:
                continue
            # the export's own split: the first name is reliable, the "last name" may hold the middle names
            f = fold_word(first)
            if f and d.given and f != d.given[0] and f in d.given:
                d.given = [f] + [g for g in d.given if g != f]
            # the display name: a first name, a nickname or a handle-like token
            disp = _clean_display(display)
            if disp and " " not in disp:
                tok = fold_word(disp)
                if tok and tok not in d.given and tok not in d.family and tok.isalpha():
                    d.nicknames.add(tok)
            elif disp:
                dg, df = split_name(disp, self.lang)
                if dg and df == d.family and dg[0] not in d.given:
                    d.nicknames.add(dg[0])
            # the handle: "octavia.brennholt" confirms the family token and may carry a nickname
            if handle:
                self.link_local_part(d, handle)

    def add_docx(self, payload: bytes) -> None:
        try:
            with zipfile.ZipFile(_bytes_io(payload)) as zf:
                try:
                    core = zf.read("docProps/core.xml")
                except KeyError:
                    core = b""
        except (zipfile.BadZipFile, OSError, ValueError):
            return
        if not core:
            return
        try:
            root = ET.fromstring(core)
        except ET.ParseError:
            return
        for el in root.iter():
            tag = el.tag.rsplit("}", 1)[-1]
            if tag in ("creator", "lastModifiedBy") and el.text:
                for name in re.split(r"\s*[;,/]\s*|\s+and\s+|\s+&\s+", el.text):
                    name = _clean_display(name)
                    if name and len(_WORD_RE.findall(name)) >= 2 and not _looks_like_org(name):
                        self.add_person(name, source="docx")

    def add_text(self, text: str) -> None:
        if text and isinstance(text, str):
            self.texts.append(text)

    # -- text passes (after every structured source) ------------------------------------------------------

    def scan_texts(self) -> None:
        for text in self.texts:
            try:
                self.scan_signatures(text)
                self.scan_declarations(text)
            except Exception as exc:  # noqa: BLE001 - never fail a build on one text
                log.debug("directory: text pass skipped (%s)", type(exc).__name__)

    def scan_signatures(self, text: str) -> None:
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if not _SIGNOFF_RE.match(line):
                continue
            for nxt in lines[i + 1:i + 3]:
                if not nxt.strip():
                    continue
                m = _NAME_LINE_RE.match(nxt)
                if m:
                    name = m.group("name")
                    words = _WORD_RE.findall(name)
                    if 2 <= len(words) <= 4 and not _looks_like_org(name) \
                            and (_first_name_ok(fold_word(words[0]), self.lang)
                                 or self._known_family(fold_word(words[-1]))):
                        self.add_person(name, source="signature")
                break

    def _known_family(self, tok: str) -> bool:
        parts = _parts(tok) or [tok]
        return any(d.family and d.family[-len(parts):] == parts for d in self.drafts)

    def _known_person(self, first: str, last: str) -> bool:
        f, parts = fold_word(first), _parts(last)
        return any(d.family and d.family[-len(parts):] == parts and d.given and (d.given[0] == f or f in d.nicknames)
                   for d in self.drafts) if parts else False

    def scan_declarations(self, text: str) -> None:
        for m in _NICK_INSIDE_RE.finditer(text):
            nick = m.group("n1") or m.group("n2")
            first, last = m.group("first"), m.group("last")
            if _vocab_word(fold_word(last), self.lang):
                continue
            if m.group("n2") and (_vocab_word(fold_word(nick), self.lang) or not self._declaration_ok(first, nick)):
                continue  # "(Nick)" without quotes: must look like a diminutive of the first name
            if _vocab_word(fold_word(first), self.lang) or (m.group("n2") and not (
                    _first_name_ok(fold_word(first), self.lang) or self._known_family(fold_word(last)))):
                continue  # the quotes declare the nickname; the brackets need a first name of the lexicon
            self.add_person(f"{first} {last}", source="declared", nicknames=[nick])
        if not _NICK_KEYWORDS_RE.search(text) and "(" not in text:
            return
        for m in _NICK_AFTER_RE.finditer(text):
            first, mid, last = m.group("first"), m.group("mid") or "", m.group("last")
            nick = m.group("nick1") or m.group("nick2")
            mids = mid.split()
            if mids and _sentence_start(text, m.start()) and _first_name_ok(fold_word(mids[0]), self.lang) \
                    and not self._known_person(first, last) and not any(
                        d.given and d.given[0] == fold_word(first) for d in self.drafts):
                first, mid = mids[0], " ".join(mids[1:]) + (" " if len(mids) > 1 else "")  # "Ping Marcus Vale": a verb
            known = self._known_person(first, last)
            keyword = bool(_NICK_KEYWORDS_RE.search(text[m.end("last"):m.end()]))  # after the name only
            if not (_first_name_ok(fold_word(first), self.lang) or known) or _vocab_word(fold_word(first), self.lang):
                continue
            if _vocab_word(fold_word(nick), self.lang) and not keyword and not _first_name_ok(fold_word(nick), self.lang):
                continue  # "(Final)" is no nickname; "(goes by Daisy)" and "(Daisy)" are
            if not keyword and not self._declaration_ok(first, nick):
                continue
            if not keyword and _vocab_word(fold_word(last), self.lang):
                continue
            full = " ".join((first + " " + mid + " " + last).split())
            self.add_person(full, source="declared", nicknames=[nick])

    def _declaration_ok(self, first: str, nick: str) -> bool:
        """Without a keyword, the nickname must look like one of the first name: a diminutive of its table, a
        prefix, a first name of the lexicon, or the same first two letters."""
        f, n = fold_word(first), fold_word(nick)
        if not f or not n or f == n:
            return False
        if n in nm._nick_forms(f) or f.startswith(n) or n.startswith(f):
            return True
        if _first_name_ok(n, self.lang):
            return True
        return len(n) >= 3 and f[:2] == n[:2]

    # -- result ------------------------------------------------------------------------------------------

    def result(self) -> "Directory":
        persons = []
        orgs: list[str] = []
        for d in sorted(self.drafts, key=lambda x: x.order):
            if any(self.company_mailbox(a) for a in d.emails):
                # v0.4 (T3): a company mailbox is no person; its display name is the organisation's name, and so is
                # its last word when that word is no first name and no common word ("Ploverin" of "Corin
                # Ploverin"): the short form a company named after a person goes by
                for v in d.values:
                    words = _WORD_RE.findall(v)
                    if not v or len(words) > 6:
                        continue
                    if v not in orgs:
                        orgs.append(v)
                    tail = words[-1] if len(words) >= 2 else ""
                    low = fold_word(tail)
                    if tail and tail not in orgs and len(low) >= 3 and not _first_name_ok(low, self.lang) \
                            and not _vocab_word(low, self.lang):
                        orgs.append(tail)
                continue
            p = d.freeze()
            if p is not None:
                persons.append(p)
        return Directory(persons, dict(sorted(self.domains.items())), orgs=orgs)


_SENTENCE_END_RE = re.compile(r"(?:^|[.!?…:;]|\n)[\s\"“”«»()\-–—•*]*$")


def _sentence_start(text: str, pos: int) -> bool:
    return bool(_SENTENCE_END_RE.search(text[max(0, pos - 40):pos]))


def _mailbox_spells_label(address: str) -> bool:
    """The local part of the address, its words joined, equals the organisation label of its domain."""
    if "@" not in address:
        return False
    label = re.sub(r"[^a-z0-9]", "", domain_label(address))
    local = re.sub(r"[^a-z0-9]", "", fold_word(address.split("@", 1)[0].split("+", 1)[0]))
    return len(label) >= 3 and local == label


def _bytes_io(payload: bytes):
    import io

    return io.BytesIO(payload)


def _looks_like_org(name: str) -> bool:
    low = fold_word(name)
    toks = low.split()
    if not toks:
        return True
    if any(t in nm.LEGAL_FORMS or t in nm.ORG_GENERIC for t in toks):
        return True
    try:
        from . import locale as lc

        b = lc.bundle("en")
        if any(lc.legal_form(w, b) for w in name.split()) or any(t in b.org_suffixes or t in b.org_generic for t in toks):
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


def _read_item(item: DocItem) -> tuple[str, object, str]:
    """(kind, data, name) of a document item: a path, or a (kind, data) tuple."""
    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
        return item[0].lower(), item[1], ""
    path = Path(getattr(item, "path", item))
    name = path.name.lower()
    suffix = path.suffix.lower()
    if name == "users.json":
        return "slack_users", path, name
    if suffix == ".eml":
        return "eml", path, name
    if suffix == ".docx":
        return "docx", path, name
    if suffix in (".txt", ".md", ".html", ".htm"):
        return "text_path", path, name
    return "", path, name


def build(docs: Iterable[DocItem], *, lang: str = "auto") -> "Directory":
    """Build the directory of a dossier from its documents.

    ``docs``: an iterable of file paths (``.eml``, ``users.json``, ``.docx``, ``.txt``, ``.md``, ``.html``; other
    files are ignored), objects with a ``path`` attribute (the vault's ``DocRef``), or ``(kind, data)`` pairs for
    callers that hold the payload already: ``("eml", bytes)``, ``("slack_users", list | dict | bytes | str)``,
    ``("docx", bytes)``, ``("text", str)`` (a message body or an extracted document text: signature blocks and
    nickname declarations). Unreadable items are skipped. ``lang``: "en", "fr" or "auto" (both lexicons)."""
    b = _Builder(lang if lang in ("en", "fr") else "auto")
    for item in docs:
        try:
            kind, data, _name = _read_item(item)
            if kind == "eml":
                raw = data if isinstance(data, (bytes, bytearray)) else Path(data).read_bytes()
                b.add_eml(bytes(raw))
            elif kind == "slack_users":
                if isinstance(data, (bytes, bytearray)):
                    data = json.loads(bytes(data).decode("utf-8", errors="replace"))
                elif isinstance(data, str):
                    data = json.loads(data)
                elif not isinstance(data, (list, dict)):
                    data = json.loads(Path(data).read_text(encoding="utf-8", errors="replace"))
                b.add_slack_users(data)
            elif kind == "docx":
                raw = data if isinstance(data, (bytes, bytearray)) else Path(data).read_bytes()
                b.add_docx(bytes(raw))
                if not isinstance(data, (bytes, bytearray)):
                    b.add_text(_document_text(Path(data)))
            elif kind == "text":
                b.add_text(data if isinstance(data, str) else bytes(data).decode("utf-8", errors="replace"))
            elif kind == "text_path":
                b.add_text(_document_text(Path(data)))
        except Exception as exc:  # noqa: BLE001 - one bad document never blocks the dossier
            log.debug("directory: item skipped (%s)", type(exc).__name__)
    b.scan_texts()
    return b.result()


def _document_text(path: Path) -> str:
    try:
        from .. import documents

        return str(documents.read_text(path).text)
    except Exception:  # noqa: BLE001 - unsupported or unreadable: no text pass
        return ""


# ---------------------------------------------------------------------------
# Directory
# ---------------------------------------------------------------------------


@dataclass
class Directory:
    """The people and organisation labels of one dossier (see the module docstring)."""

    persons: list[Person] = field(default_factory=list)
    domains: dict[str, int] = field(default_factory=dict)
    #: v0.4 (T3): display names of the dossier's company mailboxes (the organisation's own name)
    orgs: list[str] = field(default_factory=list)
    _by_token: Optional[dict[str, list[tuple[Person, str]]]] = field(default=None, repr=False, compare=False)
    _by_norm: Optional[dict[str, Person]] = field(default=None, repr=False, compare=False)

    # -- construction ------------------------------------------------------------------------------------

    def __post_init__(self) -> None:
        self.persons = [p for p in self.persons if p.value]
        self.domains = {k: int(v) for k, v in (self.domains or {}).items() if k}
        self.orgs = [o for o in dict.fromkeys(self.orgs or ()) if isinstance(o, str) and o.strip()]

    def __len__(self) -> int:
        return len(self.persons)

    def __bool__(self) -> bool:
        return bool(self.persons or self.domains or self.orgs)

    def to_dict(self) -> dict:
        out = {"version": 1, "persons": [p.to_dict() for p in self.persons], "domains": dict(self.domains)}
        if self.orgs:
            out["orgs"] = list(self.orgs)
        return out

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "Directory":
        if not data or not isinstance(data, dict):
            return cls()
        persons = []
        for item in data.get("persons") or []:
            try:
                persons.append(Person.from_dict(item))
            except (ValueError, TypeError, AttributeError):
                continue
        domains = data.get("domains") or {}
        if not isinstance(domains, dict):
            domains = {}
        orgs = data.get("orgs") or []
        if not isinstance(orgs, list):
            orgs = []
        return cls(persons, {str(k): int(v) for k, v in domains.items() if isinstance(v, (int, float))},
                   orgs=[str(o) for o in orgs if isinstance(o, str)])

    # -- indexes -----------------------------------------------------------------------------------------

    def _index(self) -> dict[str, list[tuple[Person, str]]]:
        if self._by_token is None:
            idx: dict[str, list[tuple[Person, str]]] = {}
            norms: dict[str, Person] = {}
            for p in self.persons:
                for t in p.given:
                    idx.setdefault(t, []).append((p, "given"))
                for t in p.family:
                    idx.setdefault(t, []).append((p, "family"))
                for t in p.nicknames:
                    idx.setdefault(t, []).append((p, "nick"))
                for t in p.handles:
                    idx.setdefault(t, []).append((p, "handle"))
                for t in p.emails:
                    idx.setdefault(t, []).append((p, "email"))
                try:
                    norms.setdefault(p.norm, p)
                except Exception:  # noqa: BLE001
                    pass
            self._by_token = idx
            self._by_norm = norms
        return self._by_token

    def by_norm(self, norm: str) -> Optional[Person]:
        """The person whose full name has this engine key (``entity_norm("PERSON", value)``)."""
        self._index()
        return (self._by_norm or {}).get(norm)

    def _token_hits(self, token: str) -> list[tuple[Person, str]]:
        """(person, role) pairs of one name word; a hyphenated word ("Okoro-Sand") needs every part to play the
        same role for the same person."""
        idx = self._index()
        parts = _parts(token)
        if not parts:
            return []
        if len(parts) == 1:
            return [(p, r) for p, r in idx.get(parts[0], ()) if r != "email"]
        first = {(id(p), r): (p, r) for p, r in idx.get(parts[0], ()) if r != "email"}
        for part in parts[1:]:
            keys = {(id(p), r) for p, r in idx.get(part, ())}
            first = {k: v for k, v in first.items() if k in keys}
        return list(first.values())

    def roles(self, token: str) -> frozenset[str]:
        """Roles a name word plays across the directory ("given", "family", "nick", "handle")."""
        return frozenset(r for _p, r in self._token_hits(token))

    def persons_for(self, token: str, role: Optional[str] = None) -> list[Person]:
        """People for whom the word has ``role`` (any role when None), in directory order."""
        out = []
        for p, r in self._token_hits(token):
            if (role is None or r == role) and p not in out:
                out.append(p)
        return out

    def org_labels(self) -> dict[str, int]:
        """Organisation labels of the correspondents' e-mail domains, with their address counts."""
        return dict(self.domains)

    def is_org_label(self, surface: str) -> bool:
        key = re.sub(r"[^a-z0-9]", "", fold_word(surface))
        return bool(key) and key in self.domains

    # -- lookups -----------------------------------------------------------------------------------------

    def lookup(self, surface: str) -> list[Match]:
        """Directory hits for a surface: a full name (in any order, with or without middle names), a first name,
        a family name, a nickname, a handle (with or without "@") or an e-mail address. Several people may share a
        first name: the caller decides what to do with more than one match."""
        s = surface.strip()
        if not s:
            return []
        idx = self._index()
        low = fold_word(s).lstrip("@")
        if "@" in low:
            return [Match(p, "email") for p, r in idx.get(low, ()) if r == "email"]
        if s.startswith("@") or re.search(r"[._]", low):
            hits = [Match(p, "handle") for p, r in idx.get(low, ()) if r == "handle"]
            if hits or s.startswith("@"):
                return _unique(hits)
        toks = list(phrase_tokens(s))
        if not toks:
            return []
        if len(toks) == 1 or (" " not in s.strip() and "-" in s):
            return _unique([Match(p, r) for p, r in self._token_hits(s.strip())])
        if "," in s:
            fam = [t for t in phrase_tokens(s.split(",", 1)[0])]
            giv = [t for t in phrase_tokens(s.split(",", 1)[1])]
            out = [Match(p, "full") for p in self.persons if p.family and list(p.family) == fam and giv
                   and (giv[0] in p.given or giv[0] in p.nicknames)]
            if out:
                return out
        out = []
        for p in self.persons:
            if not p.family:
                continue
            fam = list(p.family)
            n = len(fam)
            if toks[-n:] == fam:
                rest = toks[:-n]
            elif toks[:n] == fam:
                rest = toks[n:]
            else:
                continue
            rest = [t for t in rest if t not in nm.PARTICLES and not nm.is_title_word(t)]
            if rest and all(t in p.given or t in p.nicknames or (len(t) == 1 and any(g[0] == t for g in p.given))
                            for t in rest):
                out.append(Match(p, "full"))
        return out

    def person_roles(self, surface: str, person: Optional[Person] = None) -> list[tuple[str, str]]:
        """``[(token, role)]`` for the tokens of a name surface, in text order: "given", "family", "nick",
        "initial", "particle", "title" or "other". "Varlamov, Kit" gives [("Varlamov", "family"), ("Kit", "nick")].
        ``person``: the known owner (else the directory's own match, else positional roles with the lexicon)."""
        words = _WORD_RE.findall(surface)
        if not words:
            return []
        if person is None:
            hits = self.lookup(surface)
            if len({m.person.norm for m in hits}) == 1:
                person = hits[0].person
        given = set(person.given) if person else set()
        family = set(person.family) if person else set()
        nicks = set(person.nicknames) if person else set()
        out: list[tuple[str, str]] = []
        comma = "," in surface
        before_comma = surface.split(",", 1)[0] if comma else ""
        n_before = len(_WORD_RE.findall(before_comma)) if comma else 0
        unknown: list[int] = []
        for i, w in enumerate(words):
            t = fold_word(w)
            parts = _parts(w) or [t]
            in_family = all(x in family for x in parts)
            in_given = all(x in given for x in parts)
            in_nick = all(x in nicks for x in parts)
            if nm.is_title_word(w):
                out.append((w, "title"))
            elif len(t) == 1 or re.fullmatch(r"[^\W\d_]\.", w):
                out.append((w, "initial"))
            elif t in nm.PARTICLES and i not in (0, len(words) - 1):
                out.append((w, "particle"))
            elif in_family and not in_given:
                out.append((w, "family"))
            elif in_given and not in_family:
                out.append((w, "given"))
            elif in_nick:
                out.append((w, "nick"))
            elif in_given:
                out.append((w, "given"))
            else:
                out.append((w, ""))
                unknown.append(i)
        if unknown:
            named = [k for k, (_w, r) in enumerate(out) if r not in ("title", "initial", "particle")]
            if comma and n_before:
                for k in unknown:
                    out[k] = (out[k][0], "family" if k < n_before else "given")
            elif person and family and any(r == "family" for _w, r in out):
                for k in unknown:
                    out[k] = (out[k][0], "given")
            else:
                last = named[-1] if named else len(out) - 1
                for k in unknown:
                    out[k] = (out[k][0], "family" if k == last else "given")
        return out

    # -- store -------------------------------------------------------------------------------------------

    def seed(self, store, *, source: str = "directory") -> list:
        """Register every full name of the directory in ``store`` (an ``EntityStore``) so that the first mention
        of a person is masked and its variants are known; v0.4 (T3): every company mailbox name as a company.
        Returns the entities added or found. A store without a ``source`` keyword gets the default source."""
        from .masker import entity_norm  # late import

        out = []
        # v0.4 (T3): the display name of a company mailbox is registered as the company, so its first mention is
        # masked too (it is no person of the directory)
        names = [("PERSON", p.value) for p in self.persons if p.full] + [("ORG", o) for o in self.orgs]
        for kind, raw in names:
            value = " ".join(raw.split())
            norm = entity_norm(kind, value)
            if not norm:
                continue
            try:
                found = store.get_by_norm(kind, norm)
            except Exception:  # noqa: BLE001
                found = None
            if found is None:
                try:
                    found = store.add(kind, value, norm, source=source)
                except TypeError:
                    found = store.add(kind, value, norm)
                except Exception as exc:  # noqa: BLE001 - a refused value never blocks the others
                    log.debug("directory: seed skipped (%s)", type(exc).__name__)
                    continue
            out.append(found)
        return out


def _unique(items: list[Match]) -> list[Match]:
    seen: set[tuple[int, str]] = set()
    out = []
    for m in items:
        key = (id(m.person), m.role)
        if key not in seen:
            seen.add(key)
            out.append(m)
    return out
