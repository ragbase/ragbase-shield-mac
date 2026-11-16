"""RAGbase Shield MCP server (stdio), used by Claude Desktop, Claude Code, Cowork, Codex, ChatGPT desktop
and VS Code (ARCHITECTURE.md 4.5).

Six tools, all built on :mod:`ragbase_shield.service`. Every result holds codes (DOSSIER_n, DOC_n, OUTPUT_n,
PERSONNE_A...) and masked text only. No tool takes raw client text to unmask: the model can never ask for a
real value. Errors come back as short messages with codes only.
"""
from __future__ import annotations

import logging
import os
import re
import sys
import threading
from collections import Counter
from typing import Annotated, Literal, Optional

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from . import __version__, service
from .service import DEFAULT_READ_CHARS, LIMITS, MAX_READ_CHARS, MIN_READ_CHARS, ShieldError
from .vault import UNSUPPORTED_KIND

log = logging.getLogger(__name__)

SERVER_NAME = "RAGbase Shield"

TOOL_NAMES = (
    "shield_list_dossiers",
    "shield_list_documents",
    "shield_read_document",
    "shield_search",
    "shield_save_draft",
    "shield_status",
)

INSTRUCTIONS = """\
RAGbase Shield protège les données des clients de l'utilisateur par une pseudonymisation locale et réversible.
Vous ne voyez que des codes : DOSSIER_1, DOC_3, PERSONNE_A, SOCIETE_B, ADRESSE_1, EMAIL_2, IBAN_1, RG_1...
Les vraies valeurs restent sur l'ordinateur de l'utilisateur.

Règles :
1. Travaillez uniquement avec les outils Shield : shield_list_dossiers, shield_list_documents,
   shield_read_document, shield_search. N'utilisez jamais un autre outil (lecture de fichier, terminal, recherche)
   sur un dossier enregistré.
2. Ne demandez jamais à l'utilisateur les vrais noms, adresses ou numéros. Ne les écrivez jamais, même si vous
   croyez les deviner.
3. Dites à l'utilisateur de ne pas coller de texte client brut ni de joindre de fichier dans la conversation :
   ce contenu part chez le fournisseur avant que Shield puisse agir. Il doit connecter le dossier dans Shield Desk.
4. Recopiez les codes à l'identique (PERSONNE_A, jamais « Personne A »). N'inventez pas de code.
5. Enregistrez le document final, rédigé avec les codes, avec shield_save_draft. Shield remet les vrais noms
   sur l'ordinateur et renvoie seulement un identifiant OUTPUT_n.
6. Dites ensuite à l'utilisateur que le fichier restauré est dans le dossier « Shield - sorties » de son dossier
   client, et qu'il doit l'ouvrir sur son ordinateur.
7. Une session Shield travaille sur un seul dossier client : le premier dossier ouvert. Pour un autre dossier,
   l'utilisateur ouvre une nouvelle session (dans Claude Desktop : quitter puis relancer l'application).

RAGbase Shield protects the user's client data with local, reversible pseudonymisation. You only see codes
(DOSSIER_1, DOC_3, PERSONNE_A, SOCIETE_B, IBAN_1...); the real values stay on the user's computer.
Rules: work only through the Shield tools and never use other file, shell or search tools on a registered
dossier; never ask the user for real names, addresses or numbers, and never type them; tell the user not to
paste raw client text or attach files in the chat (connect the dossier in Shield Desk instead); copy codes
exactly and never invent one; save the final document, written with codes, with shield_save_draft; then tell
the user the restored file is in the "Shield - sorties" folder of the client dossier, to open on their computer.
One Shield session works on one client dossier only (the first one opened); another dossier needs a new session
(in Claude Desktop: quit and restart the app).
"""

_READ_ONLY = dict(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)

DossierArg = Annotated[str, Field(description="Code du dossier, par exemple DOSSIER_1 (dossier code).")]
DocumentArg = Annotated[str, Field(description="Code du document, par exemple DOC_3 (document code).")]


def _kb(n: int | None) -> str:
    if n is None:
        return "?"
    return f"{max(1, round(n / 1024))} Ko"


def _fail(exc: BaseException) -> ToolError:
    """A ToolError with a code-only message (fail closed)."""
    if isinstance(exc, ShieldError):
        return ToolError(str(exc))
    log.warning("RAGbase Shield MCP: unexpected %s", type(exc).__name__)
    return ToolError("Erreur interne de RAGbase Shield (internal error). Rien n'a été renvoyé.")


def _out(text: str, dossier: Optional[str] = None) -> str:
    """Last canary on a whole text before it leaves this process. ``dossier``: the dossier the text belongs to (its
    values keep their codes, values of other dossiers get new codes of it: R1 CL-26).

    F0 (0.4.3): the tools no longer use it on a whole answer: they build the answer with :func:`_answer`, so that
    Shield's own words, codes and counts never go through the last check."""
    with service.output_dossier(dossier):
        return service.default().guard_output(text)


class Content(str):
    """F0 (0.4.3): a part of a tool answer that comes from a document or from a person (file name, folder name,
    passage of text, search result, title, a code the model wrote). Only these parts go through the last check."""

    __slots__ = ()


def _answer(parts: list, dossier: Optional[str] = None) -> str:
    """F0 (0.4.3): a tool answer built from parts. The :class:`Content` parts go through the last check (one pass
    for all of them); every other part is Shield's own frame (fixed words, codes, counts, sizes) and is never
    changed: a registered value of a dossier ("4") can no longer turn "DOSSIER_4" or a count into a code."""
    out = [str(x) for x in parts]
    where = [i for i, x in enumerate(parts) if isinstance(x, Content)]
    if where:
        with service.output_dossier(dossier):
            checked = service.default().guard_parts([out[i] for i in where])
        for i, text in zip(where, checked):
            out[i] = text
    return "".join(out)


def _lines(rows: list[list]) -> list:
    """Parts of several lines (each line a list of parts), joined by line breaks."""
    out: list = []
    for k, row in enumerate(rows):
        if k:
            out.append("\n")
        out.extend(row)
    return out


# ------------------------------------------------------------------------------------------ dossier scope

#: R1 correction CL-27 (product default awaiting confirmation): one MCP server process serves ONE dossier, the
#: first one a tool opens, so a model cannot list and read every client of the home in one session.
PIN_BY_DEFAULT = True
_CODE_RE = re.compile(r"^DOSSIER_[0-9]{1,6}$")
_TRUE = ("1", "true", "yes", "on", "oui")


class _Scope:
    """The dossier this server process is pinned to.

    ``RAGBASE_SHIELD_DOSSIER=DOSSIER_n`` pins it from the start; otherwise the first dossier a tool uses
    successfully is pinned. ``RAGBASE_SHIELD_MULTI_DOSSIER=1`` turns automatic pinning off. Shield Desk and the
    CLI are never pinned (they do not go through this server)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        fixed = os.environ.get("RAGBASE_SHIELD_DOSSIER", "").strip().upper()
        self.pinned: Optional[str] = fixed if _CODE_RE.match(fixed) else None

    @staticmethod
    def automatic() -> bool:
        multi = os.environ.get("RAGBASE_SHIELD_MULTI_DOSSIER", "").strip().lower() in _TRUE
        return PIN_BY_DEFAULT and not multi

    def check(self, dossier: str) -> None:
        """Raise a code-only ShieldError when ``dossier`` is not the pinned dossier."""
        code = str(dossier or "").strip().upper()
        with self._lock:
            pinned = self.pinned
        if pinned is None or not _CODE_RE.match(code) or code == pinned:
            return
        raise ShieldError(f"Cette session RAGbase Shield est limitée à {pinned} : {code} n'est pas accessible ici. "
                          "Pour travailler sur un autre dossier, l'utilisateur ouvre une nouvelle session "
                          "(dans Claude Desktop : quitter puis relancer l'application). "
                          f"(This Shield session is pinned to {pinned}; {code} is refused: dossier_scope.)",
                          "dossier_scope")

    def used(self, dossier: str) -> None:
        """A tool used ``dossier`` successfully: pin it when nothing is pinned yet."""
        code = str(dossier or "").strip().upper()
        if not _CODE_RE.match(code) or not self.automatic():
            return
        with self._lock:
            if self.pinned is None:
                self.pinned = code


_SCOPE = _Scope()


def _scoped(dossier: str) -> str:
    _SCOPE.check(dossier)
    return dossier


def _model_items(r: dict, dossier: Optional[str]) -> tuple[list[str], list[str], list[str]]:
    """F0 (0.4.3): the codes, marks and ids that the model wrote in a draft (``unknown_codes``, ``placeholders``,
    ``shield_ids`` of a save report), each one checked as content; Shield's own words around them are not."""
    groups = [list(r.get("unknown_codes") or []), list(r.get("placeholders") or []), list(r.get("shield_ids") or [])]
    flat = [str(x) for g in groups for x in g]
    if flat:
        with service.output_dossier(dossier):
            flat = service.default().guard_parts(flat)
    out, k = [], 0
    for g in groups:
        out.append(flat[k:k + len(g)])
        k += len(g)
    return out[0], out[1], out[2]


def _placeholder_note(placeholders: list[str], shield_ids: list[str]) -> str:
    """Lines of the save message about what could not be restored (R1 CL-31, CL-36). Generic marks and codes."""
    lines = []
    if placeholders:
        counts = Counter(placeholders)
        items = ", ".join(f"{p} ×{n}" if n > 1 else p for p, n in counts.items())
        lines.append("Attention : marques génériques laissées dans le fichier, elles ne peuvent pas être "
                     f"restaurées : {items}. Réécrivez ces passages avec les codes du dossier (PERSONNE_A, "
                     "SOCIETE_B...) ou avec des mots neutres, puis enregistrez à nouveau avec shield_save_draft. "
                     f"(Placeholders left, not restorable: {items}; rewrite those places and save again.)")
    if shield_ids:
        items = ", ".join(shield_ids)
        lines.append(f"Identifiants Shield laissés tels quels dans le fichier : {items}. Remplacez-les par une "
                     "description de la pièce, puis enregistrez à nouveau. "
                     f"(Shield ids left in the file: {items}.)")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------ tool bodies

def shield_list_dossiers() -> str:
    """List the registered client dossiers by code (DOSSIER_1, DOSSIER_2...) with document and entity counts.
    Never shows real dossier titles. Liste les dossiers clients par code."""
    try:
        rows = service.list_dossiers()
        # F0 (0.4.3): codes, counts and fixed words only: Shield's own frame, never changed by the last check
        if not rows:
            return ("Aucun dossier enregistré. L'utilisateur doit d'abord connecter un dossier dans Shield Desk "
                    "ou avec « ragbase-shield add <dossier> ». (No dossier registered yet.)")
        pinned = _SCOPE.pinned
        if pinned is not None:
            rows = [r for r in rows if r.get("code") == pinned]
            if not rows:
                return (f"Cette session RAGbase Shield est limitée à {pinned}, qui n'existe plus. L'utilisateur "
                        "ouvre une nouvelle session. (Pinned dossier no longer registered.)")
        lines = [f"{len(rows)} dossier(s) RAGbase Shield (codes uniquement) :"]
        for r in rows:
            if r.get("error"):
                lines.append(f"- {r['code']} : coffre illisible (vault unreadable)")
            else:
                lines.append(f"- {r['code']} : {r['documents']} document(s), {r['entities']} élément(s) masqué(s)")
        if pinned is not None:
            lines.append(f"Session limitée à {pinned} (un dossier par session). (Session pinned to {pinned}.)")
        elif _SCOPE.automatic():
            lines.append("Cette session se limitera au premier dossier ouvert (un dossier par session). "
                         "(This session will be pinned to the first dossier it opens.)")
        lines.append("Suite : shield_list_documents avec le code du dossier.")
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        raise _fail(exc) from None


def shield_list_documents(dossier: DossierArg) -> str:
    """List the documents of a dossier by coded id (DOC_n) with masked file names, type and size.
    Liste les pièces d'un dossier (identifiants DOC_n, noms masqués)."""
    try:
        docs = service.list_documents(_scoped(dossier))
        _SCOPE.used(dossier)
        code = dossier.strip().upper()
        if not docs:
            return f"{code} : aucun document lisible (formats : pdf, docx, txt, md, eml, html)."
        lines: list[list] = [[f"{code} : {len(docs)} document(s). Format : id | type | taille | nom masqué"]]
        unsupported = 0
        for d in docs:
            size = _kb(d["size"])
            if d.get("pages"):
                size += f", {d['pages']} page(s)"
            if d["kind"] == UNSUPPORTED_KIND:
                unsupported += 1
            lines.append([f"{d['id']} | {d['kind']} | {size} | ", Content(d["masked_path"])])
        if unsupported:
            lines.append([f"{unsupported} fichier(s) « {UNSUPPORTED_KIND} » (tableur, image...) : Shield ne peut pas "
                          "les lire. Demandez à l'utilisateur une copie en PDF, Word ou texte si leur contenu compte. "
                          "(Unsupported files are listed but cannot be read.)"])
        lines.append(["Suite : shield_read_document(dossier, document) ou shield_search(dossier, query)."])
        return _answer(_lines(lines), code)
    except Exception as exc:  # noqa: BLE001
        raise _fail(exc) from None


def shield_read_document(
    dossier: DossierArg,
    document: DocumentArg,
    start: Annotated[int, Field(description="Position de départ en caractères (0 au début ; utilisez la "
                                            "valeur « suite » renvoyée). Start offset in characters.",
                                ge=0)] = 0,
    max_chars: Annotated[int, Field(description=f"Nombre maximum de caractères ({MIN_READ_CHARS} à "
                                                f"{MAX_READ_CHARS}). Maximum characters per page.",
                                    ge=1)] = DEFAULT_READ_CHARS,
) -> str:
    """Read one document of a dossier, masked on this computer. Returns the text with codes (PERSONNE_A,
    SOCIETE_B, IBAN_1) instead of the real values. Long documents come in pages: call again with the
    returned start value. Lit une pièce masquée, page par page."""
    try:
        r = service.read_document(_scoped(dossier), document, start, max_chars)
        _SCOPE.used(dossier)
        code = dossier.strip().upper()
        if r["needs_ocr"]:
            return (f"{code} / {r['id']} : PDF sans couche texte (document scanné). Pas d'OCR en v0.1 : "
                    "ce document ne peut pas être lu. Demandez à l'utilisateur une version texte. "
                    "(Scanned PDF without a text layer: no OCR in v0.1.)")
        end = r["start"] + len(r["text"])
        head = f"[{code} / {r['id']} · caractères {r['start']}-{end} sur {r['total_chars']}"
        if r.get("pages"):
            head += f" · {r['pages']} page(s) PDF"
        head += f" · suite : start={r['next']}]" if r["next"] is not None else " · fin du document]"
        if r.get("warnings"):
            head += "\n[avertissements : " + ", ".join(r["warnings"]) + "]"  # short codes of the reader
        return _answer([head + "\n", Content(r["text"])], code)
    except Exception as exc:  # noqa: BLE001
        raise _fail(exc) from None


def shield_search(
    dossier: DossierArg,
    query: Annotated[str, Field(description="Mots à chercher. Vous pouvez utiliser des codes (PERSONNE_A) : "
                                            "Shield cherche localement. Words or codes to search for.")],
) -> str:
    """Search the documents of a dossier locally. Codes in the query (PERSONNE_A) match every mention of that
    entity. Returns masked snippets only. Recherche locale, extraits masqués."""
    try:
        hits = service.search(_scoped(dossier), query)
        _SCOPE.used(dossier)
        code = dossier.strip().upper()
        if not hits:
            return f"{code} : aucun résultat. (No result.)"
        lines: list[list] = [[f"{code} : {len(hits)} résultat(s). Pour lire la suite : shield_read_document avec "
                              "start."]]
        for i, h in enumerate(hits, 1):
            lines.append([f"{i}. {h['document']} (start={h['offset']}) : ", Content(h["snippet"])])
        return _answer(_lines(lines), code)
    except Exception as exc:  # noqa: BLE001
        raise _fail(exc) from None


def shield_save_draft(
    dossier: DossierArg,
    title: Annotated[str, Field(description="Titre du fichier, avec des codes si besoin. File title.")],
    content: Annotated[str, Field(description="Document final rédigé avec les codes (Markdown simple : titres #, "
                                              "listes, **gras**, tableaux). Final text written with codes.")],
    format: Annotated[Literal["docx", "md"], Field(description="docx (par défaut) ou md.")] = "docx",
) -> str:
    """Save the final draft (written with codes) into the dossier's 'Shield - sorties' folder. Real names are
    restored on this computer only; the tool returns a coded file id (OUTPUT_n) and the size, never the
    restored text. Enregistre le projet final ; les vrais noms sont remis sur l'ordinateur uniquement."""
    try:
        r = service.save_draft_report(_scoped(dossier), title, content, format)
        _SCOPE.used(dossier)
        code = dossier.strip().upper()
        unknown, placeholders, ids = _model_items(r, code)
        msg = (f"Enregistré : {r['id']} ({_kb(r['bytes'])}, {r['format']}) dans le dossier « Shield - sorties » "
               f"de {code}. Les vrais noms sont restaurés uniquement sur l'ordinateur de l'utilisateur. "
               "Dites à l'utilisateur d'ouvrir ce fichier sur son ordinateur. "
               f"(Saved {r['id']}; the restored file is in the 'Shield - sorties' folder of {code}.)")
        if unknown:
            msg += ("\nCodes inconnus de ce dossier, laissés tels quels (unknown codes, not restored) : "
                    + ", ".join(unknown))
        note = _placeholder_note(placeholders, ids)
        if note:
            msg += "\n" + note
        return msg
    except Exception as exc:  # noqa: BLE001
        raise _fail(exc) from None


def shield_status() -> str:
    """Show the Shield version, the detection mode and its honest limits. Version, mode et limites."""
    try:
        s = service.status()
        lines = [f"RAGbase Shield {s['version']} · détection : {s['engine_mode']} · codes : {s['locale']} · "
                 f"{s['dossiers']} dossier(s) · aucun appel réseau.", "Limites :"]
        lines += [f"- {fr} ({en})" for fr, en in LIMITS]
        return "\n".join(lines)  # F0 (0.4.3): Shield's own frame only
    except Exception as exc:  # noqa: BLE001
        raise _fail(exc) from None


# ------------------------------------------------------------------------------------------ server

_TOOLS = (
    (shield_list_dossiers, "Lister les dossiers", _READ_ONLY),
    (shield_list_documents, "Lister les pièces", _READ_ONLY),
    (shield_read_document, "Lire une pièce masquée", _READ_ONLY),
    (shield_search, "Chercher dans un dossier", _READ_ONLY),
    (shield_save_draft, "Enregistrer le projet final",
     dict(read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False)),
    (shield_status, "Statut de Shield", _READ_ONLY),
)


def build_server() -> MCPServer:
    """The RAGbase Shield MCP server with its six tools. A new server starts with no pinned dossier (CL-27)."""
    global _SCOPE
    _SCOPE = _Scope()
    server = MCPServer(
        name=SERVER_NAME,
        title=SERVER_NAME,
        description="Pseudonymisation locale et réversible des dossiers clients (local, reversible pseudonymisation).",
        instructions=INSTRUCTIONS,
        version=__version__,
        website_url="https://ragbase.ai",
        log_level="WARNING",
    )
    for fn, title, hints in _TOOLS:
        server.add_tool(fn, name=fn.__name__, title=title,
                        annotations=ToolAnnotations(title=title, **hints), structured_output=False)
    return server


def main() -> int:
    """Run the stdio server (``ragbase-shield mcp``). Logs go to stderr, never stdout."""
    if not logging.getLogger().handlers:
        logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    build_server().run("stdio")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
