"""Read and write documents on this computer (contract: ARCHITECTURE.md section 4.3).

Readers: .pdf (pdfplumber up to 40 pages, pypdf beyond: v0.4.3), .docx (python-docx + raw WordprocessingML), .xlsx (stdlib zipfile + SpreadsheetML, v0.4),
.txt/.md, .eml (stdlib email), .html/.htm (stdlib html.parser). Writers: DOCX from simple Markdown, and Markdown.

Privacy rules for this module:
- No network access, no GPL/AGPL/LGPL dependency.
- Errors and warnings are short codes ("pdf_encrypted", "pdf_no_text_layer"). They never contain a file
  name, a path or document text, so callers can show them to a model.
- What a reader skips is never sent anywhere (comment authors, attachment names, field codes), so skipping is
  always the safe choice.
"""
from __future__ import annotations

import codecs
import copy
import logging
import os
import re
import tempfile
import time
import unicodedata
import zipfile
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional
from xml.sax.saxutils import escape as _xml_escape

from . import config

log = logging.getLogger(__name__)
# pypdf logs its own warnings, some with raw bytes of the file ("invalid pdf header: b'...'"): they could put
# client text in a server log. Only Shield's own code-only messages are logged.
logging.getLogger("pypdf").setLevel(logging.ERROR)

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".pdf", ".docx", ".xlsx", ".txt", ".md", ".eml", ".html", ".htm")
KIND_BY_EXT: dict[str, str] = {
    ".pdf": "pdf", ".docx": "docx", ".xlsx": "xlsx", ".txt": "txt", ".md": "md",
    ".eml": "eml", ".html": "html", ".htm": "html",
}
DOCX_DOSSIER_PROPERTY = "RAGbaseShieldDossier"
_DOSSIER_CODE_RE = re.compile(r"^DOSSIER_[0-9]{1,6}$")


class DocumentError(Exception):
    """A document could not be read or written. The message is a short code, never a name or content."""


#: v0.4.3: the code of a document that was not read in its time budget (:data:`DOCUMENT_TIME_BUDGET`).
READ_TOO_LONG = "read_too_long"
#: v0.4.3: time budget of one document, in seconds, checked between the pages of a PDF. Past it the document is not
#: read (like a scanned PDF): the reading of a dossier always ends.
DOCUMENT_TIME_BUDGET = 180.0
#: v0.4.3: a PDF of more pages than this is read with pypdf from the first page. pdfplumber gives the table cells
#: (the published quality numbers were measured with it) but is far too slow on long files with ruled tables, and its
#: memory grows with every page. A PDF of this many pages or fewer is read exactly as in 0.4.2.
PDF_PLUMBER_MAX_PAGES = 40
#: v0.4.3: the pdfplumber reading of a short PDF falls back to pypdf for the whole file past this time (seconds).
PDF_PLUMBER_SECONDS = 20.0
#: v0.4.3 (fix): a short PDF with one page whose drawing instructions pass this size (decoded bytes), or with more
#: than :data:`PDF_PLUMBER_MAX_BYTES` in all, is read with pypdf: pdfplumber can spend minutes on one plan-like page
#: and its time budget is only checked between pages. The repository PDFs have at most 15 KB per page.
PDF_PLUMBER_MAX_PAGE_BYTES = 256 * 1024
PDF_PLUMBER_MAX_BYTES = 4 * 1024 * 1024
#: v0.4.3 (fix): inside one pypdf page, the time budget is checked every this many drawing instructions.
PDF_CHECK_EVERY_OPS = 2_000
#: v0.4.3 (fix): the code of a document whose text passes the ``max_chars`` limit of :func:`read_text`.
READ_TOO_LARGE = "read_too_large"

#: ``progress(done, total)``: pages read so far and pages of the document (PDF only).
Progress = Optional[Callable[[int, int], None]]


class ReadTooLong(DocumentError):
    """The document was not read in its time budget (code :data:`READ_TOO_LONG`)."""

    def __init__(self) -> None:
        super().__init__(READ_TOO_LONG)


class ReadTooLarge(DocumentError):
    """The text of the document passes the ``max_chars`` limit of :func:`read_text` (code :data:`READ_TOO_LARGE`):
    the reading stops there, so no minute is lost on a document Shield will not mask."""

    def __init__(self) -> None:
        super().__init__(READ_TOO_LARGE)


class _PlumberTooSlow(Exception):
    """Internal: the pdfplumber reading went past :data:`PDF_PLUMBER_SECONDS`; pypdf reads the file instead."""


class ReadResult(str):
    """The extracted text, plus reading metadata.

    A ReadResult IS the text (it subclasses ``str``), so ``read_text(p)`` can be used as a plain string.
    It also carries:

    - ``text``: the same text as a plain ``str``
    - ``pages``: number of pages for PDF, ``None`` when the format has no pages
    - ``needs_ocr``: True when a PDF has no text layer (scanned). No OCR in v0.1.
    - ``warnings``: tuple of short codes (no real values)

    ``repr()`` shows only sizes, never the text, so an accidental log line does not leak content.
    """

    pages: Optional[int]
    needs_ocr: bool
    warnings: tuple[str, ...]

    def __new__(cls, text: str = "", pages: Optional[int] = None, needs_ocr: bool = False,
                warnings: Iterable[str] = ()) -> "ReadResult":
        obj = super().__new__(cls, text)
        obj.pages = pages
        obj.needs_ocr = bool(needs_ocr)
        obj.warnings = tuple(warnings)
        return obj

    @property
    def text(self) -> str:
        """The text as a plain ``str``."""
        return str.__str__(self)

    def __repr__(self) -> str:
        return (f"ReadResult(chars={len(self)}, pages={self.pages}, needs_ocr={self.needs_ocr}, "
                f"warnings={list(self.warnings)})")


def kind_of(path: os.PathLike | str) -> Optional[str]:
    """Return the document kind ("pdf", "docx", "xlsx", "txt", "md", "eml", "html") or None if unsupported."""
    return KIND_BY_EXT.get(Path(path).suffix.lower())


# --------------------------------------------------------------------------------------------- reading

def read_text(path: os.PathLike | str, *, progress: Progress = None, budget: Optional[float] = None,
              clock: Callable[[], float] = time.monotonic, max_chars: Optional[int] = None) -> ReadResult:
    """Extract the text of a supported document. Raises DocumentError (code only) on failure.

    Fail closed: a reader error never returns partial raw content.

    v0.4.3: ``progress(done, total)`` is called between the pages of a PDF. ``budget`` (seconds, default
    :data:`DOCUMENT_TIME_BUDGET`) is checked between pages: past it, :class:`ReadTooLong` (code ``read_too_long``).
    ``clock``: the time source (tests).

    v0.4.3 (fix): ``max_chars``: a PDF stops as soon as its pages hold more than this many non-space characters, and
    any document whose text is longer than this raises :class:`ReadTooLarge` (code ``read_too_large``). The text of a
    document under the limit is the same as without it.
    """
    p = Path(path)
    kind = kind_of(p)
    if kind is None:
        raise DocumentError("unsupported_format")
    if not p.is_file():
        raise DocumentError("file_not_found")
    deadline = clock() + (DOCUMENT_TIME_BUDGET if budget is None else float(budget))
    try:
        if kind == "pdf":
            text, pages, needs_ocr, warnings = _read_pdf(p, progress=progress, deadline=deadline, clock=clock,
                                                         max_chars=max_chars)
        else:
            reader = {"docx": _read_docx, "xlsx": _read_xlsx, "txt": _read_plain, "md": _read_plain,
                      "eml": _read_eml, "html": _read_html}[kind]
            text, pages, needs_ocr, warnings = reader(p)
    except DocumentError:
        raise
    except Exception as exc:  # noqa: BLE001 - fail closed with a code, no content in the message
        log.debug("%s reader failed: %s", kind, type(exc).__name__)
        raise DocumentError(f"{kind}_unreadable") from None
    text = _clean_text(text)
    if max_chars is not None and len(text) > max_chars:
        raise ReadTooLarge()
    return ReadResult(text, pages, needs_ocr, warnings)


def _clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", ""))
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip("\n")


def _decode_bytes(data: bytes) -> tuple[str, list[str]]:
    """UTF-8 (with or without BOM), UTF-16 with BOM, else cp1252 with a warning."""
    if data.startswith(codecs.BOM_UTF16_LE) or data.startswith(codecs.BOM_UTF16_BE):
        return data.decode("utf-16", errors="replace"), []
    try:
        return data.decode("utf-8-sig"), []
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace"), ["encoding_fallback_cp1252"]


def _read_plain(path: Path):
    text, warnings = _decode_bytes(path.read_bytes())
    return text, None, False, warnings


# PDF -------------------------------------------------------------------------------------------------
#
# Table cells (v0.3, S1). A PDF has no table structure in its text layer: two cells of one row are two groups of
# glyphs on one line, and a plain text extraction joins them with one space ("Jane Doe Acme Corp 2026"). The
# name rules then read across the cell boundary. When the words of a line are available with their boxes
# (pdfplumber, MIT), a gap wider than ``PDF_CELL_GAP_FACTOR`` times the mean character width of the two words is
# a cell boundary and is written as ``PDF_CELL_SEPARATOR``, which the name rules never cross. The benchmark
# runner that maps text back to glyph boxes calls the same functions, so the product and the measurement agree.

PDF_CELL_SEPARATOR = " | "
PDF_CELL_GAP_FACTOR = 2.0
PDF_LINE_TOLERANCE = 3  # points: words whose tops differ by less belong to one line


def pdf_cell_gap(prev, word, factor: float = PDF_CELL_GAP_FACTOR) -> bool:
    """True when the horizontal gap between two words of one PDF line is a table cell boundary: wider than
    ``factor`` times the mean character width of the two words. ``prev`` and ``word`` are mappings with ``text``,
    ``x0`` and ``x1`` (pdfplumber word dicts)."""
    pw = max(1, len(str(prev["text"])))
    ww = max(1, len(str(word["text"])))
    cw = ((float(prev["x1"]) - float(prev["x0"])) / pw + (float(word["x1"]) - float(word["x0"])) / ww) / 2
    return cw > 0 and (float(word["x0"]) - float(prev["x1"])) > factor * cw


def pdf_word_separator(prev, word, factor: float = PDF_CELL_GAP_FACTOR) -> str:
    """The text written between two words of one PDF line: a space, or :data:`PDF_CELL_SEPARATOR` at a cell gap.
    A word of a ruled-table row (v0.4, T8) carries its own ``sep``: the separator before it is decided by the
    table's cells, not by the gap."""
    sep = word.get("sep") if isinstance(word, dict) else None
    if sep is not None:
        return str(sep)
    return PDF_CELL_SEPARATOR if pdf_cell_gap(prev, word, factor) else " "


def _pdf_tables_enabled() -> bool:
    """``RAGBASE_SHIELD_PDF_TABLES=0`` reads every PDF page line by line (the v0.3 reading; a debugging switch)."""
    return os.environ.get("RAGBASE_SHIELD_PDF_TABLES", "1").strip() != "0"


def pdf_table_rows(page, words: list, *, line_tolerance: float = PDF_LINE_TOLERANCE) -> tuple[list[list], set]:
    """v0.4 (T8, K2): the rows of the ruled tables of a pdfplumber page, each as one list of words.

    A cell that wraps over two or more lines is one cell in the table's own structure (its ruling lines or
    rectangles), but two or three text lines in a line-by-line reading: the address or the identifier it holds is
    cut in the middle. Here every word whose centre sits in a cell belongs to that cell; the cells of a row are
    written left to right, and inside a cell the words keep their reading order (line by line, left to right). The
    first word of every cell after the first carries ``sep = PDF_CELL_SEPARATOR``; the other words ``sep = " "``
    (:func:`pdf_word_separator` reads it), so the row is written as one line of cells. Returns the rows and the ids
    of the words they took. A page without ruled tables gives no rows; a failure of the table finder gives none
    either, so the page is then read line by line (fail safe)."""
    from operator import itemgetter

    from pdfplumber.utils import cluster_objects

    try:
        tables = page.find_tables()
    except Exception:  # noqa: BLE001 - a table finder error never loses a page
        return [], set()
    rows: list[list] = []
    taken: set = set()
    for table in tables:
        for row in getattr(table, "rows", ()):
            cells_words: list[list] = []
            for cell in getattr(row, "cells", ()):
                if not cell:
                    continue  # a merged cell: its words belong to the cell that spans it
                x0, top, x1, bottom = (float(v) for v in cell)
                mine = [w for w in words if id(w) not in taken
                        and x0 - 1 <= (float(w["x0"]) + float(w["x1"])) / 2 <= x1 + 1
                        and top - 1 <= (float(w["top"]) + float(w["bottom"])) / 2 <= bottom + 1]
                if not mine:
                    continue
                ordered: list = []
                for line in cluster_objects(mine, itemgetter("top"), line_tolerance):
                    ordered.extend(sorted(line, key=itemgetter("x0")))
                for w in ordered:
                    taken.add(id(w))
                cells_words.append(ordered)
            if not cells_words:
                continue
            line_out: list = []
            for ci, ws in enumerate(cells_words):
                for wi, w in enumerate(ws):
                    w2 = dict(w)
                    w2["sep"] = PDF_CELL_SEPARATOR if (wi == 0 and ci > 0) else " "
                    line_out.append(w2)
            rows.append(line_out)
    return rows, taken


def pdf_page_lines(page, *, line_tolerance: float = PDF_LINE_TOLERANCE, return_chars: bool = False) -> list[list]:
    """The words of a pdfplumber page grouped by line (top to bottom) and sorted left to right. With
    ``return_chars`` every word carries its ``chars`` (the runner's map back to glyph boxes). The rows of the
    page's ruled tables are one line each, cell by cell (v0.4, T8: :func:`pdf_table_rows`); the other words are
    grouped by their visual line as before."""
    from operator import itemgetter

    from pdfplumber.utils import cluster_objects

    words = page.extract_words(return_chars=return_chars)
    rows, taken = pdf_table_rows(page, words, line_tolerance=line_tolerance) if _pdf_tables_enabled() else ([], set())
    loose = [w for w in words if id(w) not in taken] if taken else words
    lines = [sorted(line, key=itemgetter("x0")) for line in cluster_objects(loose, itemgetter("top"), line_tolerance)]
    if rows:
        lines = sorted(lines + rows, key=lambda line: min(float(w["top"]) for w in line))
    return lines


def pdf_page_text(page, *, line_tolerance: float = PDF_LINE_TOLERANCE, factor: float = PDF_CELL_GAP_FACTOR) -> str:
    """Text of a pdfplumber page: one line per visual line, :data:`PDF_CELL_SEPARATOR` between the cells of a
    table row, a space between the other words."""
    out: list[str] = []
    for line in pdf_page_lines(page, line_tolerance=line_tolerance):
        parts: list[str] = []
        prev = None
        for w in line:
            if prev is not None:
                parts.append(pdf_word_separator(prev, w, factor))
            parts.append(w["text"])
            prev = w
        out.append("".join(parts))
    return "\n".join(out)


def _pdf_page_count(path: Path) -> Optional[int]:
    """Number of pages from pypdf (fast: the page tree only), or None when pypdf cannot open the file (the 0.4.2
    reading then decides, with its own error codes)."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        if reader.is_encrypted and not reader.decrypt(""):
            return None
        return len(reader.pages)
    except Exception:  # noqa: BLE001 - a count is only a choice of reader
        return None


def _pdf_heavy(path: Path) -> bool:
    """v0.4.3 (fix): True when one page of a short PDF holds more than :data:`PDF_PLUMBER_MAX_PAGE_BYTES` of drawing
    instructions (decoded), or the file more than :data:`PDF_PLUMBER_MAX_BYTES` in all: pypdf reads it, with its
    time budget checked inside each page. False when pypdf cannot tell (the 0.4.2 reading then decides)."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        if reader.is_encrypted and not reader.decrypt(""):
            return False
        total = 0
        for index in range(len(reader.pages)):
            page = reader.pages[index]
            try:
                contents = page.get_contents()
                size = len(contents.get_data()) if contents is not None else 0
            finally:
                _free_pypdf_page(page)
            total += size
            if size > PDF_PLUMBER_MAX_PAGE_BYTES or total > PDF_PLUMBER_MAX_BYTES:
                return True
        return False
    except Exception:  # noqa: BLE001 - a size is only a choice of reader
        return False


def _read_pdf(path: Path, progress: Progress = None, deadline: Optional[float] = None,
              clock: Callable[[], float] = time.monotonic, max_chars: Optional[int] = None):
    """v0.4.3: pdfplumber (table cells) for a PDF of :data:`PDF_PLUMBER_MAX_PAGES` pages or fewer, exactly as in
    0.4.2; pypdf from the first page for a longer one, for a short one with a very heavy page (:func:`_pdf_heavy`),
    or when pdfplumber takes more than :data:`PDF_PLUMBER_SECONDS`."""
    try:
        import pdfplumber  # noqa: F401 - optional: word boxes give the table cells (v0.3)
    except ImportError:
        return _read_pdf_pypdf(path, progress, deadline, clock, max_chars)
    pages = _pdf_page_count(path)
    if pages is not None and (pages > PDF_PLUMBER_MAX_PAGES or _pdf_heavy(path)):
        return _read_pdf_pypdf(path, progress, deadline, clock, max_chars)
    try:
        return _read_pdf_plumber(path, progress, deadline, clock, max_chars)
    except _PlumberTooSlow:
        log.info("RAGbase Shield: a PDF took more than %ss with pdfplumber; read with pypdf", PDF_PLUMBER_SECONDS)
        return _read_pdf_pypdf(path, progress, deadline, clock, max_chars)


def _check_time(deadline: Optional[float], clock: Callable[[], float]) -> None:
    if deadline is not None and clock() > deadline:
        raise ReadTooLong()


def _non_space(text: str) -> int:
    """Characters of ``text`` that are not white space: the cleaned text of a document is never shorter (except for
    accents written as two characters), so a running count of them can stop a reading over ``max_chars`` early."""
    return len("".join(text.split()))


def _check_size(seen: int, max_chars: Optional[int]) -> None:
    if max_chars is not None and seen > max_chars:
        raise ReadTooLarge()


def _read_pdf_plumber(path: Path, progress: Progress = None, deadline: Optional[float] = None,
                      clock: Callable[[], float] = time.monotonic, max_chars: Optional[int] = None):
    import pdfplumber

    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    warnings: list[str] = []
    texts: list[str] = []
    empty = 0
    seen = 0
    started = clock()
    try:
        pdf = pdfplumber.open(str(path), password="")  # many PDFs only carry an owner password
    except Exception as exc:  # noqa: BLE001
        # pdfplumber wraps the pdfminer error; the cause names the password problem (never the file or its text)
        names = []
        cause: Optional[BaseException] = exc
        while cause is not None and len(names) < 6:
            names.append(type(cause).__name__)
            cause = cause.__cause__ or cause.__context__
        if any("Password" in n or "Encrypt" in n for n in names):
            raise DocumentError("pdf_encrypted") from None
        raise
    with pdf:
        total = len(pdf.pages)
        if progress is not None:
            progress(0, total)
        for number, page in enumerate(pdf.pages, start=1):
            try:
                page_text = pdf_page_text(page)
            except Exception:  # noqa: BLE001
                page_text = ""
                warnings.append(f"pdf_page_unreadable:{number}")
            finally:
                _close_quietly(page)  # v0.4.3: free the page's objects (memory stays flat)
            if not page_text.strip():
                empty += 1
            texts.append(page_text.strip("\n"))
            if max_chars is not None:
                seen += _non_space(page_text)
                _check_size(seen, max_chars)
            if progress is not None:
                progress(number, total)
            if number < total:
                _check_time(deadline, clock)
                if clock() - started > PDF_PLUMBER_SECONDS:
                    raise _PlumberTooSlow()
    pages = len(texts)
    needs_ocr = pages > 0 and empty == pages
    if needs_ocr:
        warnings.append("pdf_no_text_layer")
    elif empty:
        warnings.append(f"pdf_pages_without_text:{empty}")
    return "\n\n".join(t for t in texts if t.strip()), pages, needs_ocr, warnings


def _close_quietly(page) -> None:
    close = getattr(page, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001 - freeing memory never fails a reading
            pass


def _free_pypdf_page(page) -> None:
    """Drop the decoded content of a pypdf page once its text is read (pypdf keeps it with the file's objects), so
    the memory of a long reading does not grow with the page count."""
    try:
        contents = page.get("/Contents")
        contents = contents.get_object() if contents is not None else None
        items = contents if isinstance(contents, list) else [contents]
        for item in items:
            obj = item.get_object() if item is not None else None
            if obj is not None and getattr(obj, "decoded_self", None) is not None:
                obj.decoded_self = None
    except Exception:  # noqa: BLE001 - freeing memory never fails a reading
        pass


def _read_pdf_pypdf(path: Path, progress: Progress = None, deadline: Optional[float] = None,
                    clock: Callable[[], float] = time.monotonic, max_chars: Optional[int] = None):
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    if reader.is_encrypted:
        try:
            ok = reader.decrypt("")  # many PDFs only carry an owner password
        except Exception:  # noqa: BLE001
            ok = 0
        if not ok:
            raise DocumentError("pdf_encrypted")
    warnings: list[str] = []
    texts: list[str] = []
    empty = 0
    total = len(reader.pages)
    seen = 0
    ops = [0]

    def check(_operator, _operands, _cm, _tm) -> None:
        # v0.4.3 (fix): the time budget inside one page too (a plan-like page can hold millions of instructions)
        ops[0] += 1
        if ops[0] % PDF_CHECK_EVERY_OPS == 0:
            _check_time(deadline, clock)

    visitor = check if deadline is not None else None
    if progress is not None:
        progress(0, total)
    for index in range(total):  # one page at a time: no list of page objects is kept
        number = index + 1
        page = None
        try:
            page = reader.pages[index]
            page_text = (page.extract_text(visitor_operand_before=visitor) if visitor is not None
                         else page.extract_text()) or ""
        except ReadTooLong:
            raise
        except Exception:  # noqa: BLE001
            page_text = ""
            warnings.append(f"pdf_page_unreadable:{number}")
        finally:
            if page is not None:
                _free_pypdf_page(page)
            page = None
        if not page_text.strip():
            empty += 1
        texts.append(page_text.strip("\n"))
        if max_chars is not None:
            seen += _non_space(page_text)
            _check_size(seen, max_chars)
        if progress is not None:
            progress(number, total)
        if number < total:
            _check_time(deadline, clock)
    pages = len(texts)
    needs_ocr = pages > 0 and empty == pages
    if needs_ocr:
        warnings.append("pdf_no_text_layer")
    elif empty:
        warnings.append(f"pdf_pages_without_text:{empty}")
    return "\n\n".join(t for t in texts if t.strip()), pages, needs_ocr, warnings


# DOCX ------------------------------------------------------------------------------------------------

_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_SKIP_TAGS = {"pPr", "rPr", "sectPr", "tblPr", "trPr", "tcPr", "tblGrid", "delText", "instrText",
              "fldData", "footnotePr", "endnotePr", "commentRangeStart", "commentRangeEnd"}
_WRAPPER_TAGS = {"sdt", "sdtContent", "customXml", "smartTag", "ins", "moveTo"}
_SECTION_LABELS = {
    "fr": {"header": "[En-tête]", "footer": "[Pied de page]", "footnotes": "[Notes de bas de page]",
           "endnotes": "[Notes de fin]", "comments": "[Commentaires]", "sheet": "[Feuille]"},
    "en": {"header": "[Header]", "footer": "[Footer]", "footnotes": "[Footnotes]",
           "endnotes": "[Endnotes]", "comments": "[Comments]", "sheet": "[Sheet]"},
}


def _local(tag) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _safe_xml(blob: bytes):
    from lxml import etree

    parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False, huge_tree=False)
    return etree.fromstring(blob, parser=parser)


def _walk(el, out: list[str]) -> None:
    tag = _local(el.tag)
    if not tag or tag in _SKIP_TAGS:
        return
    if tag == "t":
        out.append(el.text or "")
    elif tag == "tab":
        out.append("\t")
    elif tag in ("br", "cr"):
        out.append("\n")
    elif tag == "noBreakHyphen":
        out.append("-")
    elif tag == "AlternateContent":
        # Text boxes are stored twice (mc:Choice and mc:Fallback). Read one branch only.
        branches = [c for c in el if _local(c.tag) == "Choice"] or [c for c in el if _local(c.tag) == "Fallback"]
        if branches:
            for child in branches[0]:
                _walk(child, out)
    elif tag == "tbl":
        _walk_table(el, out)
    elif tag == "p":
        for child in el:
            _walk(child, out)
        out.append("\n")
    else:
        for child in el:
            _walk(child, out)


def _children(el, name: str) -> Iterator:
    """Direct children with this local name, looking through content-control wrappers."""
    for child in el:
        tag = _local(child.tag)
        if tag == name:
            yield child
        elif tag in _WRAPPER_TAGS:
            yield from _children(child, name)


def _walk_table(tbl, out: list[str]) -> None:
    for tr in _children(tbl, "tr"):
        cells = []
        for tc in _children(tr, "tc"):
            buf: list[str] = []
            for child in tc:
                _walk(child, buf)
            cells.append(" ".join("".join(buf).split()))
        if any(cells):
            out.append(" | ".join(cells) + "\n")


def _element_text(el) -> str:
    out: list[str] = []
    _walk(el, out)
    return "".join(out)


def _notes_text(root, note_tag: str) -> str:
    lines = []
    for note in root:
        if _local(note.tag) != note_tag:
            continue
        if note.get(f"{{{_W}}}type") in ("separator", "continuationSeparator", "continuationNotice"):
            continue
        text = _element_text(note).strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def _read_docx(path: Path):
    import docx
    from docx.opc.constants import RELATIONSHIP_TYPE as RT

    document = docx.Document(str(path))
    kinds = {RT.HEADER: "header", RT.FOOTER: "footer", RT.FOOTNOTES: "footnotes",
             RT.ENDNOTES: "endnotes", RT.COMMENTS: "comments"}
    found: dict[str, list[tuple[str, str]]] = {k: [] for k in kinds.values()}
    seen: set[str] = set()
    warnings: list[str] = []
    for rel in document.part.rels.values():
        if rel.is_external or rel.reltype not in kinds:
            continue
        part = rel.target_part
        name = str(part.partname)
        if name in seen:
            continue
        seen.add(name)
        section = kinds[rel.reltype]
        try:
            root = _safe_xml(part.blob)
        except Exception:  # noqa: BLE001
            warnings.append(f"docx_{section}_unreadable")
            continue
        if section == "footnotes":
            text = _notes_text(root, "footnote")
        elif section == "endnotes":
            text = _notes_text(root, "endnote")
        elif section == "comments":
            # Only the comment text. Author names and initials are not read.
            text = "\n".join(t for t in (_element_text(c).strip() for c in root if _local(c.tag) == "comment") if t)
        else:
            text = _element_text(root).strip()
        if text:
            found[section].append((name, text))

    labels = _SECTION_LABELS[config.locale()]

    def block(section: str) -> str:
        texts: list[str] = []
        for _, text in sorted(found[section]):
            if text not in texts:  # first-page / even-page variants often repeat the same text
                texts.append(text)
        return f"{labels[section]}\n" + "\n".join(texts) if texts else ""

    body = _element_text(document.element.body)
    parts = [block("header"), body, block("footer"), block("footnotes"), block("endnotes"), block("comments")]
    return "\n\n".join(p.strip("\n") for p in parts if p.strip()), None, False, warnings


# XLSX (v0.4, T8) ---------------------------------------------------------------------------------------
#
# A workbook is a zip of SpreadsheetML parts; the standard library reads it (no new dependency, local only).
# Each worksheet is written as one line per row with the cells joined by PDF_CELL_SEPARATOR, in column order,
# so the table rules (a column header labels its cells) read a sheet as they read a PDF table. The sheet title
# opens each sheet (it is text of the workbook and is masked like any other). Cell values are read as the file
# stores them: shared and inline strings, a formula's cached result, numbers as written (a date is its serial
# number: no style table is read). Errors and comments are not content and are skipped.

_XLSX_WORKSHEET_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"
_XLSX_CELL_REF_RE = re.compile(r"^([A-Z]{1,3})(\d+)$")


def _xlsx_col_index(ref: Optional[str]) -> int:
    """Column number of a cell reference ("A1" -> 1, "AB7" -> 28); 0 when the reference carries no column."""
    m = _XLSX_CELL_REF_RE.match(ref or "")
    if not m:
        return 0
    n = 0
    for ch in m.group(1):
        n = n * 26 + (ord(ch) - 64)
    return n


def _xlsx_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    try:
        blob = zf.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    out: list[str] = []
    for si in _safe_xml(blob):
        if _local(si.tag) == "si":
            out.append("".join(t.text or "" for t in si.iter() if _local(t.tag) == "t"))
    return out


def _xlsx_sheets(zf: zipfile.ZipFile) -> list[tuple[str, str]]:
    """(title, part name) of the worksheets in workbook order."""
    names = set(zf.namelist())
    rels: dict[str, str] = {}
    try:
        for rel in _safe_xml(zf.read("xl/_rels/workbook.xml.rels")):
            if _local(rel.tag) != "Relationship" or rel.get("Type") != _XLSX_WORKSHEET_REL:
                continue
            target = rel.get("Target") or ""
            rels[rel.get("Id") or ""] = target.lstrip("/") if target.startswith("/") else "xl/" + target
    except KeyError:
        return []
    out: list[tuple[str, str]] = []
    for el in _safe_xml(zf.read("xl/workbook.xml")).iter():
        if _local(el.tag) != "sheet":
            continue
        rid = next((v for k, v in el.attrib.items() if _local(k) == "id"), "")
        part = rels.get(rid)
        if part and part in names:
            out.append((el.get("name") or "", part))
    return out


def _xlsx_cell_text(cell, shared: list[str]) -> str:
    kind = cell.get("t") or ""
    if kind == "inlineStr":
        return "".join(t.text or "" for t in cell.iter() if _local(t.tag) == "t")
    value = next((x for x in cell if _local(x.tag) == "v"), None)
    raw = (value.text or "") if value is not None else ""
    if kind == "s":
        try:
            return shared[int(raw.strip())]
        except (ValueError, IndexError):
            return ""
    if kind == "b":
        return "TRUE" if raw.strip() == "1" else "FALSE"
    if kind == "e":
        return ""  # an error value is not content
    return raw.strip()  # "str" (a formula's text result), "n" or no type (a number as written)


def _read_xlsx(path: Path):
    labels = _SECTION_LABELS[config.locale()]
    warnings: list[str] = []
    blocks: list[str] = []
    with zipfile.ZipFile(str(path)) as zf:
        shared = _xlsx_shared_strings(zf)
        sheets = _xlsx_sheets(zf)
        if not sheets:
            warnings.append("xlsx_no_sheet")
        for title, part in sheets:
            try:
                root = _safe_xml(zf.read(part))
            except Exception:  # noqa: BLE001 - one unreadable sheet never loses the others
                warnings.append("xlsx_sheet_unreadable")
                continue
            rows: list[dict[int, str]] = []
            for row in root.iter():
                if _local(row.tag) != "row":
                    continue
                cells: dict[int, str] = {}
                for cell in row:
                    if _local(cell.tag) != "c":
                        continue
                    text = " ".join(_xlsx_cell_text(cell, shared).split())
                    if text:
                        cells[_xlsx_col_index(cell.get("r")) or (max(cells) + 1 if cells else 1)] = text
                if cells:
                    rows.append(cells)
            if not rows:
                continue
            first = min(min(r) for r in rows)
            lines = [f"{labels['sheet']} {title}".rstrip()]
            for r in rows:
                last = max(r)
                lines.append(PDF_CELL_SEPARATOR.join(r.get(c, "") for c in range(first, last + 1)))
            blocks.append("\n".join(lines))
    return "\n\n".join(blocks), None, False, warnings


# EML -------------------------------------------------------------------------------------------------

_EML_HEADERS = ("From", "To", "Cc", "Subject", "Date")


def _part_content(part) -> str:
    try:
        content = part.get_content()
        if isinstance(content, str):
            return content
    except Exception:  # noqa: BLE001 - unknown charset or broken encoding: decode by hand below
        pass
    payload = part.get_payload(decode=True) or b""
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return _decode_bytes(payload)[0]


def _read_eml(path: Path):
    with open(path, "rb") as fh:
        msg = BytesParser(policy=policy.default).parse(fh)
    warnings: list[str] = []
    lines = []
    for name in _EML_HEADERS:
        try:
            value = msg.get(name)
        except Exception:  # noqa: BLE001
            value = None
            warnings.append(f"eml_header_unreadable:{name.lower()}")
        if value:
            lines.append(f"{name}: {' '.join(str(value).split())}")
    body = ""
    plain = msg.get_body(preferencelist=("plain",))
    if plain is not None:
        body = _part_content(plain)
    else:
        html_part = msg.get_body(preferencelist=("html",))
        if html_part is not None:
            body = html_to_text(_part_content(html_part))
            warnings.append("eml_html_body")
    attachments = sum(1 for _ in msg.iter_attachments()) if msg.is_multipart() else 0
    if attachments:
        warnings.append(f"eml_attachments_not_read:{attachments}")
    return "\n".join(lines) + "\n\n" + body, None, False, warnings


# HTML ------------------------------------------------------------------------------------------------

_SOFT, _PARA = "\x01", "\x02"  # block break (one newline) and paragraph break (blank line), merged at the end


class _HTMLText(HTMLParser):
    PARA = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "title", "table", "ul", "ol", "dl",
            "address", "figure", "form", "fieldset", "hr"}
    BLOCK = {"div", "section", "article", "header", "footer", "main", "aside", "nav", "dt", "dd", "thead",
             "tbody", "tfoot", "tr", "figcaption", "li", "caption"}
    SKIP = {"script", "style", "noscript", "template"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self._skip = 0
        self._pre = 0
        self._cells = 0

    def _at_break(self) -> bool:
        return not self.out or self.out[-1][-1:] in (_SOFT, _PARA, "\n")

    def _break(self, tag: str) -> None:
        self.out.append(_PARA if tag in self.PARA else _SOFT)

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
            return
        if tag == "br":
            self.out.append("\n")
        elif tag == "li":
            self.out.append(_SOFT + "- ")
        elif tag in ("td", "th"):
            if self._cells:
                if self.out:
                    self.out[-1] = self.out[-1].rstrip(" ")
                self.out.append(" | ")
            self._cells += 1
        elif tag in self.PARA or tag in self.BLOCK:
            if tag == "tr":
                self._cells = 0
            self._break(tag)
        if tag == "pre":
            self._pre += 1

    def handle_startendtag(self, tag, attrs):
        if tag == "br":
            self.out.append("\n")
        elif tag == "hr":
            self.out.append(_PARA)

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self._skip = max(0, self._skip - 1)
            return
        if tag in self.PARA or tag in self.BLOCK:
            self._break(tag)
        if tag == "pre":
            self._pre = max(0, self._pre - 1)

    def handle_data(self, data):
        if self._skip or not data:
            return
        if self._pre:
            self.out.append(data)
            return
        text = re.sub(r"\s+", " ", data)
        if self._at_break() or (self.out and self.out[-1].endswith(" | ")):
            text = text.lstrip()
        if text:
            self.out.append(text)


def html_to_text(html: str) -> str:
    """Convert HTML to plain text (scripts and styles removed, blocks on their own lines)."""
    parser = _HTMLText()
    parser.feed(html)
    parser.close()
    raw = "".join(parser.out)
    raw = re.sub(r"[ \t]*[\x01\x02](?:[ \t\n]*[\x01\x02])*",
                 lambda m: "\n\n" if _PARA in m.group(0) else "\n", raw)
    lines = [line.rstrip() for line in raw.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip("\n")


def _read_html(path: Path):
    text, warnings = _decode_bytes(path.read_bytes())
    return html_to_text(text), None, False, warnings


# --------------------------------------------------------------------------------------------- naming

_FORBIDDEN_CHARS = re.compile(r'[<>:"/\\|?*]')
_WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
_EXT_RE = re.compile(r"^\.[A-Za-z0-9]{1,8}$")
MAX_STEM_CHARS = 120
MAX_STEM_BYTES = 200
# Function words that never end a cut file name ("... aux droits de" -> "... aux droits"): French and English
# prepositions, articles, conjunctions and party separators.
_DANGLING_WORDS = frozenset({
    "a", "au", "aux", "avec", "c", "chez", "contre", "d", "dans", "de", "des", "du", "en", "et", "l", "la", "le",
    "les", "ou", "par", "pour", "sans", "sur", "v", "vs",
    "and", "for", "of", "or", "the", "to", "with",
})
_DANGLING_PUNCT = " .,;:-–—/&+(«\"'’"


def _plain_word(word: str) -> str:
    return unicodedata.normalize("NFKD", word).encode("ascii", "ignore").decode("ascii").lower().strip("'’.")


def trim_dangling(text: str) -> str:
    """Remove the punctuation and function words left at the end of a cut name ("Note sur le" -> "Note")."""
    s = text.rstrip(_DANGLING_PUNCT)
    while True:
        head, _sep, last = s.rpartition(" ")
        word = _plain_word(last)
        if not head or not (word in _DANGLING_WORDS or last.endswith(("'", "’")) and len(word) <= 2):
            return s
        s = head.rstrip(_DANGLING_PUNCT)


def cut_at_word(stem: str, limit: int) -> str:
    """At most ``limit`` characters, cut at a word boundary when one is close enough (half the limit)."""
    if len(stem) <= limit:
        return stem
    cut = stem[:limit]
    if not stem[limit].isspace():
        sp = cut.rfind(" ")
        if sp >= limit // 2:
            cut = cut[:sp]
    return trim_dangling(cut) or cut


def safe_filename(name: str, ext: Optional[str] = None, default: str = "document") -> str:
    """Return a file name that is safe on macOS, Windows and Linux.

    - No path separators (no traversal: "../../x" cannot leave the target folder), no control characters,
      no characters Windows refuses, no Windows reserved names, no leading dot (no hidden file).
    - ``ext`` (".docx" or "docx") is forced; without ``ext`` the existing extension is kept.
    - A long name is cut at a word boundary (at most MAX_STEM_CHARS characters and MAX_STEM_BYTES bytes), and a
      function word or punctuation left at the end of the cut is removed. The extension is always kept.
    """
    s = unicodedata.normalize("NFC", str(name))
    s = "".join(" " if unicodedata.category(c).startswith("C") else c for c in s)
    s = _FORBIDDEN_CHARS.sub("-", s)
    s = " ".join(s.split())
    if ext is not None:
        ext = "." + ext.lstrip(".").lower()
        if not _EXT_RE.match(ext):
            raise ValueError("invalid extension")
        stem = s[: -len(ext)] if s.lower().endswith(ext) else s
    else:
        stem, ext = os.path.splitext(s)
        if not _EXT_RE.match(ext or ""):
            stem, ext = s, ""
        ext = ext.lower()
    stem = stem.strip(" .-")
    if stem.split(".")[0].upper() in _WINDOWS_RESERVED:
        stem = "_" + stem
    limit = MAX_STEM_CHARS
    while limit > 1 and len(stem[:limit].encode("utf-8")) > MAX_STEM_BYTES:
        limit -= 1
    stem = cut_at_word(stem, limit)
    stem = stem.rstrip(" .") or default
    return stem + ext


def _candidates(path: Path) -> Iterator[Path]:
    yield path
    for n in range(2, 10000):
        yield path.with_name(f"{path.stem} ({n}){path.suffix}")
    raise DocumentError("no_free_file_name")


def unique_path(path: os.PathLike | str) -> Path:
    """First of "name.ext", "name (2).ext", "name (3).ext"... that does not exist yet."""
    for candidate in _candidates(Path(path)):
        if not candidate.exists():
            return candidate
    raise DocumentError("no_free_file_name")  # pragma: no cover


def _reserve(path: Path, overwrite: bool) -> Path:
    """Pick the target and reserve it atomically (O_EXCL), so two writers never get the same name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        return path
    for candidate in _candidates(path):
        try:
            fd = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            continue
        os.close(fd)
        return candidate
    raise DocumentError("no_free_file_name")  # pragma: no cover


def _write_atomic(target: Path, write, reserved: bool) -> None:
    """Write to a temp file in the same folder, then move it over the target (no half-written file).

    ``reserved``: the target is an empty placeholder made by _reserve; remove it if writing fails.
    """
    fd, tmp = tempfile.mkstemp(prefix=".shield-", suffix=".tmp", dir=str(target.parent))
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, target)
    except BaseException:
        for leftover in ([tmp, target] if reserved else [tmp]):
            try:
                os.unlink(leftover)
            except OSError:
                pass
        raise


# --------------------------------------------------------------------------------------------- writing

def write_md(path: os.PathLike | str, text: str, *, overwrite: bool = False) -> Path:
    """Write UTF-8 Markdown. Returns the path written (a unique name unless overwrite=True)."""
    target = _reserve(Path(path), overwrite)
    data = text.replace("\r\n", "\n").replace("\r", "\n")
    if not data.endswith("\n"):
        data += "\n"

    def write(tmp: str) -> None:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(data)

    try:
        _write_atomic(target, write, reserved=not overwrite)
    except OSError:
        raise DocumentError("write_failed") from None
    return target


def write_docx(path: os.PathLike | str, markdown_text: str, dossier_code: str, *,
               overwrite: bool = False) -> Path:
    """Write a DOCX from simple Markdown (# headings, paragraphs, - and 1. lists, **bold**, *italic*,
    `code`, > quotes, | tables |, ``` fences).

    The only metadata stored is the dossier code, in the custom property ``RAGbaseShieldDossier``, so the
    file can be restored weeks later. Core properties (author, title...) are left empty.
    Returns the path written (a unique name unless overwrite=True).
    """
    if not _DOSSIER_CODE_RE.match(dossier_code or ""):
        raise ValueError("dossier_code must be a dossier code such as DOSSIER_1")
    import docx

    document = docx.Document()
    _render_markdown(document, markdown_text)
    _clean_core_properties(document)
    _set_custom_property(document, DOCX_DOSSIER_PROPERTY, dossier_code)
    target = _reserve(Path(path), overwrite)
    try:
        _write_atomic(target, document.save, reserved=not overwrite)
    except OSError:
        raise DocumentError("write_failed") from None
    return target


def save_output(directory: os.PathLike | str, title: str, text: str, fmt: str = "docx",
                dossier_code: Optional[str] = None) -> Path:
    """Safe name + unique name + write, inside ``directory`` only. fmt: "docx" or "md"."""
    fmt = fmt.lower().lstrip(".")
    if fmt not in ("docx", "md"):
        raise ValueError("fmt must be docx or md")
    base = Path(directory)
    target = base / safe_filename(title, fmt)
    if target.parent.resolve() != base.resolve():  # defence in depth; safe_filename has no separators
        raise DocumentError("unsafe_file_name")
    if fmt == "docx":
        if dossier_code is None:
            raise ValueError("dossier_code is required for docx")
        return write_docx(target, text, dossier_code)
    return write_md(target, text)


def docx_dossier_code(path: os.PathLike | str) -> Optional[str]:
    """Return the dossier code stored by write_docx, or None."""
    try:
        with zipfile.ZipFile(path) as zf:
            blob = zf.read("docProps/custom.xml")
        root = _safe_xml(blob)
    except Exception:  # noqa: BLE001
        return None
    for prop in root:
        if prop.get("name") == DOCX_DOSSIER_PROPERTY:
            value = "".join(prop.itertext()).strip()
            return value if _DOSSIER_CODE_RE.match(value) else None
    return None


_CUSTOM_NS = "http://schemas.openxmlformats.org/officeDocument/2006/custom-properties"
_VT_NS = "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"
_FMTID = "{D5CDD505-2E9C-101B-9397-08002B2CF9AE}"


def _set_custom_property(document, name: str, value: str) -> None:
    from docx.opc.constants import CONTENT_TYPE as CT
    from docx.opc.constants import RELATIONSHIP_TYPE as RT
    from docx.opc.packuri import PackURI
    from docx.opc.part import Part

    xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        f'<Properties xmlns="{_CUSTOM_NS}" xmlns:vt="{_VT_NS}">'
        f'<property fmtid="{_FMTID}" pid="2" name="{_xml_escape(name)}">'
        f"<vt:lpwstr>{_xml_escape(value)}</vt:lpwstr></property></Properties>"
    )
    package = document.part.package
    part = Part(PackURI("/docProps/custom.xml"), CT.OFC_CUSTOM_PROPERTIES, xml.encode("utf-8"), package)
    package.relate_to(part, RT.CUSTOM_PROPERTIES)


def _clean_core_properties(document) -> None:
    cp = document.core_properties
    for attr in ("author", "last_modified_by", "title", "subject", "keywords", "comments", "category",
                 "content_status", "identifier", "language", "version"):
        setattr(cp, attr, "")
    cp.revision = 1
    now = datetime.now(timezone.utc).replace(microsecond=0, tzinfo=None)
    cp.created = now
    cp.modified = now


# Markdown -> DOCX ------------------------------------------------------------------------------------

_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
_HR = re.compile(r"^\s{0,3}([-*_])(?:\s*\1){2,}\s*$")
_BULLET = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_NUMBER = re.compile(r"^(\s*)(\d{1,9})([.)])\s+(.*)$")
_QUOTE = re.compile(r"^\s{0,3}>\s?(.*)$")
_FENCE = re.compile(r"^\s{0,3}(```|~~~)")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
_INLINE = re.compile(
    r"(?P<esc>\\[\\`*_{}\[\]()#+\-.!|>~])"
    r"|(?P<tick>`+)(?P<code>.+?)(?P=tick)"
    r"|\*\*\*(?=\S)(?P<bi1>.+?)(?<=\S)\*\*\*"
    r"|(?<!\w)___(?=\S)(?P<bi2>.+?)(?<=\S)___(?!\w)"
    r"|\*\*(?=\S)(?P<b1>.+?)(?<=\S)\*\*"
    r"|(?<!\w)__(?=\S)(?P<b2>.+?)(?<=\S)__(?!\w)"
    r"|\*(?=[^\s*])(?P<i1>.+?)(?<=[^\s*])\*"
    r"|(?<!\w)_(?=\S)(?P<i2>.+?)(?<=\S)_(?!\w)"   # never inside PERSONNE_A or SOCIETE_B
    r"|\[(?P<ltext>[^\]\n]+)\]\((?P<lurl>[^)\s]+)\)"
)
_MONO_FONT = "Courier New"


def _paragraph(document, style: Optional[str] = None):
    if style:
        try:
            return document.add_paragraph(style=style)
        except KeyError:
            pass
    return document.add_paragraph()


def _run(par, text: str, bold: bool = False, italic: bool = False, code: bool = False) -> None:
    if not text:
        return
    run = par.add_run(text)
    if bold:
        run.bold = True
    if italic:
        run.italic = True
    if code:
        run.font.name = _MONO_FONT


def _add_inline(par, text: str, bold: bool = False, italic: bool = False) -> None:
    pos = 0
    for m in _INLINE.finditer(text):
        if m.start() > pos:
            _run(par, text[pos:m.start()], bold, italic)
        g = m.groupdict()
        if g["esc"]:
            _run(par, g["esc"][1:], bold, italic)
        elif g["code"] is not None:
            _run(par, g["code"], bold, italic, code=True)
        elif g["bi1"] or g["bi2"]:
            _add_inline(par, g["bi1"] or g["bi2"], True, True)
        elif g["b1"] or g["b2"]:
            _add_inline(par, g["b1"] or g["b2"], True, italic)
        elif g["i1"] or g["i2"]:
            _add_inline(par, g["i1"] or g["i2"], bold, True)
        elif g["ltext"]:
            _add_inline(par, g["ltext"], bold, italic)
            if g["lurl"] != g["ltext"]:
                _run(par, f" ({g['lurl']})", bold, italic)
        pos = m.end()
    if pos < len(text):
        _run(par, text[pos:], bold, italic)


def _add_lines(par, lines: list[str]) -> None:
    for i, line in enumerate(lines):
        if i:
            par.add_run().add_break()
        _add_inline(par, line.strip())


def _list_style(base: str, indent: str) -> str:
    level = min(len(indent.expandtabs(4)) // 2, 2)
    return base if level == 0 else f"{base} {level + 1}"


_MAX_START = 32767  # largest list start Word accepts


class _ListNumbers:
    """Word numbering for "1." and "1)" lines: every item shows the number the author wrote.

    An item continues the open list of its level when its number is the next one; any other number starts a new
    Word numbering instance (``w:num`` with a ``startOverride``), so a second list that starts at 1 shows 1 and
    an item written "15." shows 15. Without this, every "List Number" paragraph of the document shares one list
    and Word numbers them 1 to N across the whole document. "1)" lists get their own level text "%1)".
    """

    def __init__(self, document) -> None:
        self.document = document
        self.open: dict[int, list] = {}           # level -> [numId, next number, delimiter]
        self.abstract: dict[tuple[str, str], Optional[int]] = {}  # (style, delimiter) -> abstractNumId

    def _numbering(self):
        return self.document.part.numbering_part.element

    def _abstract_id(self, style: str, delim: str) -> Optional[int]:
        key = (style, delim)
        if key in self.abstract:
            return self.abstract[key]
        result: Optional[int] = None
        try:
            ppr = self.document.styles[style].element.pPr
            num_id = ppr.numPr.numId.val if ppr is not None and ppr.numPr is not None else None
            numbering = self._numbering()
            num = numbering.num_having_numId(num_id) if num_id is not None else None
            result = num.abstractNumId.val if num is not None else None
            if result is not None and delim != ".":
                result = self._copy_abstract(numbering, result, delim)
        except (KeyError, AttributeError, ValueError, TypeError):
            result = None
        self.abstract[key] = result
        return result

    @staticmethod
    def _copy_abstract(numbering, abstract_id: int, delim: str) -> Optional[int]:
        """A copy of an abstract numbering whose level texts end with ``delim`` ("%1)")."""
        tag = f"{{{_W}}}abstractNum"
        attr = f"{{{_W}}}abstractNumId"
        abstracts = [a for a in numbering.iterchildren(tag)]
        source = next((a for a in abstracts if a.get(attr) == str(abstract_id)), None)
        if source is None:
            return None
        new_id = max(int(a.get(attr)) for a in abstracts) + 1
        clone = copy.deepcopy(source)
        clone.set(attr, str(new_id))
        for child in list(clone):
            if _local(child.tag) in ("nsid", "numStyleLink", "styleLink"):
                clone.remove(child)
        for pstyle in list(clone.iter(f"{{{_W}}}pStyle")):  # the style keeps its own list; the copy is used
            pstyle.getparent().remove(pstyle)                # through the paragraph's numbering only
        for text in clone.iter(f"{{{_W}}}lvlText"):
            val = text.get(f"{{{_W}}}val") or ""
            if val.endswith("."):
                text.set(f"{{{_W}}}val", val[:-1] + delim)
        abstracts[-1].addnext(clone)
        return new_id

    def number(self, par, style: str, level: int, n: int, delim: str) -> bool:
        """Give ``par`` the number ``n``. False when Word numbering is not available (the caller writes it)."""
        for deeper in [k for k in self.open if k > level]:
            del self.open[deeper]
        cur = self.open.get(level)
        if cur is not None and cur[1] == n and cur[2] == delim:
            num_id = cur[0]
            cur[1] += 1
        else:
            abstract_id = self._abstract_id(style, delim)
            if abstract_id is None or not 0 <= n <= _MAX_START:
                return False
            num = self._numbering().add_num(abstract_id)
            num.add_lvlOverride(ilvl=0).add_startOverride(n)
            num_id = num.numId
            self.open[level] = [num_id, n + 1, delim]
        num_pr = par._p.get_or_add_pPr().get_or_add_numPr()
        num_pr.get_or_add_ilvl().val = 0
        num_pr.get_or_add_numId().val = num_id
        return True


def _row_closed(line: str) -> bool:
    s = line.rstrip()
    return s.endswith("|") and not s.endswith("\\|")


def _table_row(lines: list[str], i: int, closed: bool) -> tuple[str, int]:
    """The table row that starts at ``lines[i]`` and the index after it.

    A cell value that holds a line break (an old vault value restored as "Jean\\nDUPONT") splits a row in two
    lines. When the table closes its rows with "|", a row left open is joined with the next lines (up to 3) if
    one of them closes it.
    """
    row = lines[i]
    if not closed or _row_closed(row):
        return row, i + 1
    j = i + 1
    while j < len(lines) and j - i <= 3:
        nxt = lines[j]
        if not nxt.strip() or nxt.lstrip().startswith("|"):
            break
        if _row_closed(nxt):
            joined = " ".join([row.rstrip()] + [ln.strip() for ln in lines[i + 1:j + 1]])
            return joined, j + 1
        j += 1
    return row, i + 1


def _split_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    return [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", s)]


def _add_table(document, rows: list[list[str]]) -> None:
    ncols = max(len(r) for r in rows)
    table = document.add_table(rows=len(rows), cols=ncols)
    try:
        table.style = "Table Grid"
    except Exception:  # noqa: BLE001
        pass
    for r, cells in enumerate(rows):
        for c, cell_text in enumerate(cells):
            _add_inline(table.cell(r, c).paragraphs[0], cell_text, bold=(r == 0))


def _render_markdown(document, text: str) -> None:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    buf: list[str] = []
    numbers = _ListNumbers(document)

    def flush() -> None:
        if buf:
            _add_lines(_paragraph(document), buf)
            buf.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        if _FENCE.match(line):
            flush()
            fence = _FENCE.match(line).group(1)
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith(fence):
                code_lines.append(lines[i])
                i += 1
            i += 1  # closing fence
            par = _paragraph(document)
            for j, code_line in enumerate(code_lines):
                if j:
                    par.add_run().add_break()
                _run(par, code_line, code=True)
            continue
        if not line.strip():
            flush()
        elif m := _HEADING.match(line):
            flush()
            _add_inline(document.add_heading(level=len(m.group(1))), m.group(2))
        elif _HR.match(line):
            flush()
            document.add_paragraph()
        elif line.lstrip().startswith("|") and i + 1 < len(lines) and _TABLE_SEP.match(lines[i + 1]):
            flush()
            rows = [_split_row(line)]
            closed = _row_closed(line)
            i += 2
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                row, i = _table_row(lines, i, closed)
                rows.append(_split_row(row))
            _add_table(document, rows)
            continue
        elif m := _BULLET.match(line):
            flush()
            _add_inline(_paragraph(document, _list_style("List Bullet", m.group(1))), m.group(2).strip())
        elif m := _NUMBER.match(line):
            flush()
            style = _list_style("List Number", m.group(1))
            par = _paragraph(document, style)
            content = m.group(4).strip()
            level = min(len(m.group(1).expandtabs(4)) // 2, 2)
            if par.style is None or par.style.name != style or \
                    not numbers.number(par, style, level, int(m.group(2)), m.group(3)):
                content = f"{m.group(2)}{m.group(3)} {content}"  # no Word numbering: keep the number as text
            _add_inline(par, content)
        elif _QUOTE.match(line):
            flush()
            quote = []
            while i < len(lines) and (q := _QUOTE.match(lines[i])) and lines[i].strip():
                quote.append(q.group(1))
                i += 1
            _add_lines(_paragraph(document, "Quote"), quote)
            continue
        else:
            buf.append(line)
        i += 1
    flush()
