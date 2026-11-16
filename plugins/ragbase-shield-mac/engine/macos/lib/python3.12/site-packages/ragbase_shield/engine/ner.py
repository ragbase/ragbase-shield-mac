"""Pluggable NER backends for the v0.2 detection stack (design: docs/V02_DESIGN.md section 4.2).

Backends, in the order ``auto`` tries them:

- ``gliner2:fastino``: ``fastino/gliner2-privacy-filter-PII-multi`` (package ``gliner2``, Apache-2.0, 42 PII labels,
  7 languages incl. French and English);
- ``gliner:urchade``: ``urchade/gliner_multi_pii-v1`` (package ``gliner``, Apache-2.0, free-text labels);
- ``spacy:<model>``: the v0.1 spaCy French model when somebody installed it.

Everything here is local and optional. A model is loaded once per process, on the first call that needs it, never at
import or at ``Masker`` construction. The download is one explicit user step (``download_model``); at run time the
Hugging Face libraries run with ``HF_HUB_OFFLINE=1`` and never touch the network. Absence is graceful: no package or
no cached model gives ``load_backend() is None`` and the engine runs the rules alone (codes only, never raw text).

Contract for the masker (owner: masker; see docs/NER.md section 6)::

    backend = load_backend("auto")                     # None when nothing is installed
    spans = run_ner(text, backend)                     # [(start, end, kind, label, score), ...], exact text offsets
    cands = merge.merge_ner(text, spans, rule_spans, structured, protected, lang=lang)

Or, with the v0.1 plumbing that already exists in ``Masker``::

    Masker(store, ner=as_hook(backend))                # text -> [(start, end, "PERSON"|"ORG")]

No raw text is logged by this module: log lines carry counts and names of backends only.
"""
from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Callable, Iterable, NamedTuple, Optional, Protocol

log = logging.getLogger(__name__)

__all__ = [
    "NerSpan", "NerBackend", "ModelSpec", "MODELS", "KIND_FASTINO", "KIND_URCHADE",
    "models_dir", "ner_setting", "ner_model_setting", "ner_max_chars", "ner_threads", "ner_device",
    "ner_installed", "ner_available", "resolve_model", "backend_name", "load_backend", "unload_backends",
    "chunk_text", "run_ner", "as_hook", "ChunkCache", "WATCH_LABELS", "label_thresholds",
    "snapshot_dir", "model_cached", "download_model", "status", "status_lines",
]

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class NerSpan(NamedTuple):
    """One span inside one chunk: offsets in the chunk, the model's label, its score."""
    start: int
    end: int
    label: str
    score: float


class NerBackend(Protocol):
    name: str                       # "gliner2:fastino", "gliner:urchade", "spacy:fr_core_news_md"
    labels: dict[str, str]          # model label -> Shield kind
    thresholds: dict[str, float]    # model label -> minimum score

    def predict(self, chunks: list[str]) -> list[list[NerSpan]]: ...


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo: str
    package: str                      # importable package that runs it
    licence: str
    size_mb: int                      # approximate download size
    weights: tuple[str, ...]          # one of these files must exist in the snapshot
    config: tuple[str, ...]           # one of these files must exist in the snapshot
    extra_repos: tuple[tuple[str, tuple[str, ...]], ...] = ()   # (repo, allow_patterns) also needed at run time
    ignore_patterns: tuple[str, ...] = ("*.png", "*.jpg", "*.md", "*.gif", ".gitattributes")


MODELS: dict[str, ModelSpec] = {
    "fastino": ModelSpec(
        key="fastino", repo="fastino/gliner2-privacy-filter-PII-multi", package="gliner2", licence="Apache-2.0",
        size_mb=1250, weights=("model.safetensors", "pytorch_model.bin"), config=("config.json",),
    ),
    "urchade": ModelSpec(
        key="urchade", repo="urchade/gliner_multi_pii-v1", package="gliner", licence="Apache-2.0",
        size_mb=1160, weights=("pytorch_model.bin", "model.safetensors"), config=("gliner_config.json",),
        # the encoder tokenizer is loaded by name from the GLiNER config (MIT licence, 4 MB of tokenizer files)
        extra_repos=(("microsoft/mdeberta-v3-base", ("*.json", "spm.model", "*.model", "tokenizer*")),),
    ),
}
_SPACY_MODELS = ("fr_core_news_lg", "fr_core_news_md", "fr_core_news_sm")
_ORDER = ("fastino", "urchade", "spacy")

# Shield kind <- model label. Labels not listed are ignored (cities, dates, card fields: the rules own them).
KIND_FASTINO: dict[str, str] = {
    "person": "PERSON", "full_name": "PERSON", "first_name": "PERSON", "middle_name": "PERSON", "last_name": "PERSON",
    "organization": "ORG", "company": "ORG",
    "email": "EMAIL", "phone_number": "PHONE", "address": "ADDRESS", "street_address": "ADDRESS",
    "date_of_birth": "BIRTHDATE",
    "username": "USERNAME", "account_id": "USERNAME",
    "employee_id": "EMPLOYEE_ID",
    "bank_account": "ACCOUNT_NUMBER", "account_number": "ACCOUNT_NUMBER", "routing_number": "ACCOUNT_NUMBER",
    "sensitive_account_id": "ACCOUNT_NUMBER",
    "government_id": "ID_CODE", "national_id_number": "ID_CODE", "passport_number": "ID_CODE",
    "drivers_license_number": "ID_CODE", "license_number": "ID_CODE", "tax_id": "ID_CODE", "tax_number": "ID_CODE",
    "ip_address": "ID_CODE", "password": "ID_CODE", "secret": "ID_CODE", "api_key": "ID_CODE",
    "access_token": "ID_CODE", "recovery_code": "ID_CODE",
    "vessel": "ID_CODE", "building": "ID_CODE", "school": "ID_CODE",
}
KIND_URCHADE: dict[str, str] = {
    "person": "PERSON",
    "organization": "ORG", "company": "ORG",
    "email": "EMAIL", "email address": "EMAIL",
    "phone number": "PHONE", "mobile phone number": "PHONE", "landline phone number": "PHONE",
    "address": "ADDRESS", "full address": "ADDRESS",
    "date of birth": "BIRTHDATE",
    "username": "USERNAME", "social media handle": "USERNAME",
    "employee id": "EMPLOYEE_ID", "staff number": "EMPLOYEE_ID",
    "bank account number": "ACCOUNT_NUMBER", "account number": "ACCOUNT_NUMBER",
    "passport number": "ID_CODE", "license plate number": "ID_CODE", "ip address": "ID_CODE",
    "identity card number": "ID_CODE", "national id number": "ID_CODE", "driver's license number": "ID_CODE",
    "tax identification number": "ID_CODE",
    "vessel name": "ID_CODE", "building name": "ID_CODE", "school": "ID_CODE",
}
KIND_SPACY: dict[str, str] = {"PER": "PERSON", "PERSON": "PERSON", "ORG": "ORG"}

# Named assets the model may find (vessel, building, school): never a masking candidate on their own. The merge
# never turns an ID_CODE find of the model into a candidate by default (``merge.MODEL_KINDS``); ``merge.watch_ner``
# lists these spans for the review step, where a person decides (gated layer, 2026-10-20).
WATCH_LABELS: frozenset[str] = frozenset({"vessel", "building", "school", "vessel name", "building name"})

# Default minimum scores per Shield kind (the dev-set numbers behind them are in docs/NER.md section 5).
# The merge adds its own rule for one-word people (0.70 without a cue), see merge.py.
_DEFAULT_THRESHOLDS = {"PERSON": 0.40, "ORG": 0.60, "EMAIL": 0.50, "PHONE": 0.50, "ADDRESS": 0.50,
                       "BIRTHDATE": 0.50, "USERNAME": 0.50, "EMPLOYEE_ID": 0.50, "ACCOUNT_NUMBER": 0.50,
                       "ID_CODE": 0.50}
_URCHADE_THRESHOLDS = dict(_DEFAULT_THRESHOLDS, ORG=0.45)
# Minimum scores per model label, on top of the per-kind defaults (gated layer, 2026-10-20). The dev diagnosis of
# the identifier labels: the model's account labels fire on any number of a text, so the two generic account labels
# need a higher score before the merge's cue check even looks at them. A label absent here keeps its kind's value.
_LABEL_THRESHOLDS: dict[str, float] = {"account_id": 0.60, "sensitive_account_id": 0.60}

DEFAULT_KINDS = tuple(sorted(set(KIND_FASTINO.values())))
DEFAULT_CHUNK_CHARS = 1200
DEFAULT_BATCH = 16


def label_thresholds(labels: dict[str, str], per_kind: dict[str, float],
                     per_label: Optional[dict[str, float]] = None) -> dict[str, float]:
    """Model label -> minimum score: the kind's value from ``per_kind``, then the per-label overrides
    (``_LABEL_THRESHOLDS`` by default; a caller's dict replaces them)."""
    out = {label: per_kind.get(kind, 0.5) for label, kind in labels.items()}
    for label, value in (_LABEL_THRESHOLDS if per_label is None else per_label).items():
        if label in out:
            out[label] = float(value)
    return out


def _thresholds_for(labels: dict[str, str], per_kind: dict[str, float]) -> dict[str, float]:
    return label_thresholds(labels, per_kind)


def _select_labels(labels: dict[str, str], kinds: Optional[Iterable[str]]) -> dict[str, str]:
    if kinds is None:
        return dict(labels)
    want = set(kinds)
    return {label: kind for label, kind in labels.items() if kind in want}


# ---------------------------------------------------------------------------
# Settings (environment; config.py may re-export these accessors)
# ---------------------------------------------------------------------------


def ner_setting() -> str:
    """RAGBASE_SHIELD_NER: "auto" (default), "on" or "off"."""
    v = os.environ.get("RAGBASE_SHIELD_NER", "auto").strip().lower()
    if v in ("1", "on", "true", "yes"):
        return "on"
    if v in ("0", "off", "false", "no"):
        return "off"
    return "auto"


def ner_model_setting() -> str:
    """RAGBASE_SHIELD_NER_MODEL: "auto" (default), "fastino", "urchade" or "spacy"."""
    v = os.environ.get("RAGBASE_SHIELD_NER_MODEL", "auto").strip().lower()
    return v if v in ("auto", "fastino", "urchade", "spacy") else "auto"


def ner_max_chars() -> int:
    """RAGBASE_SHIELD_NER_MAX_CHARS: NER runs on pasted text up to this size (default 200000), rules only above."""
    try:
        return max(0, int(os.environ.get("RAGBASE_SHIELD_NER_MAX_CHARS", "200000")))
    except ValueError:
        return 200_000


def ner_threads() -> int:
    """RAGBASE_SHIELD_NER_THREADS: torch threads; 0 (default) keeps the library default, the physical core count.

    Measured on a 8-core / 16-thread Intel Mac: 8 threads 1.3 k chars/s, 16 threads 1.0 k chars/s (hyper-threads
    slow the matrix products down), so the CPU count is not used."""
    try:
        n = int(os.environ.get("RAGBASE_SHIELD_NER_THREADS", "0"))
    except ValueError:
        n = 0
    return n if n > 0 else 0


def ner_device() -> str:
    """RAGBASE_SHIELD_NER_DEVICE: where the model runs. ``auto`` (default) = CUDA when torch sees a CUDA device, else
    the CPU; ``cpu`` or ``cuda`` force it. Apple MPS is not chosen by default (not measured). Always local."""
    v = os.environ.get("RAGBASE_SHIELD_NER_DEVICE", "auto").strip().lower() or "auto"
    return v if v in ("auto", "cpu", "cuda") else "auto"


def _to_device(model):
    """Move a torch model to the device of :func:`ner_device`; a failure keeps the CPU (the model still works)."""
    want = ner_device()
    if want == "cpu":
        return model
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return model.to("cuda")
    except Exception:  # noqa: BLE001 - a device problem is never a masking failure
        log.warning("RAGbase Shield: NER device %s unavailable; CPU", want)
    return model


def _is_oom(exc: BaseException) -> bool:
    """A CUDA out-of-memory error of torch (its own class, or the message of a RuntimeError)."""
    return type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower()


def _free_cuda() -> None:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def models_dir() -> Path:
    """Where ``download_model`` puts the models: RAGBASE_SHIELD_MODELS_DIR, else ``<home>/models``.

    The layout is the Hugging Face hub cache layout (``models--org--name/snapshots/<rev>/``), so the directory can
    also be given to the libraries as their cache. The directory is not created here.
    """
    env = os.environ.get("RAGBASE_SHIELD_MODELS_DIR")
    if env:
        return Path(env).expanduser()
    from .. import config  # light module; home() creates the data directory

    return config.home() / "models"


def _cache_roots() -> list[Path]:
    """Directories searched for a cached snapshot: ours first, then the user's Hugging Face cache (local only)."""
    roots = [models_dir()]
    hub = os.environ.get("HF_HUB_CACHE")
    if hub:
        roots.append(Path(hub).expanduser())
    home = os.environ.get("HF_HOME")
    if home:
        roots.append(Path(home).expanduser() / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    out: list[Path] = []
    for r in roots:
        if r not in out:
            out.append(r)
    return out


def snapshot_dir(spec: ModelSpec, roots: Optional[Iterable[Path]] = None) -> Optional[Path]:
    """The snapshot directory of a cached model (weights + config present), or None. Never loads anything."""
    for root in (list(roots) if roots is not None else _cache_roots()):
        base = root / ("models--" + spec.repo.replace("/", "--"))
        snaps = base / "snapshots"
        candidates: list[Path] = []
        ref = base / "refs" / "main"
        if ref.is_file():
            try:
                rev = ref.read_text(encoding="utf-8").strip()
                if rev:
                    candidates.append(snaps / rev)
            except OSError:
                pass
        if snaps.is_dir():
            try:
                candidates.extend(sorted((p for p in snaps.iterdir() if p.is_dir()),
                                         key=lambda p: p.stat().st_mtime, reverse=True))
            except OSError:
                pass
        candidates.append(root / spec.repo.split("/")[-1])  # a snapshot folder copied by hand
        for c in candidates:
            if _complete(c, spec):
                return c
    return None


def _complete(d: Path, spec: ModelSpec) -> bool:
    try:
        return d.is_dir() and any((d / w).is_file() for w in spec.weights) and any((d / c).is_file() for c in spec.config)
    except OSError:
        return False


def _dir_size_mb(d: Path) -> int:
    total = 0
    try:
        for p in d.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    except OSError:
        pass
    return int(total / (1024 * 1024))


def model_cached(key: str) -> bool:
    spec = MODELS.get(key)
    return spec is not None and snapshot_dir(spec) is not None


def _package_version(name: str) -> Optional[str]:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None
    except Exception:  # noqa: BLE001 - a broken install means "not installed"
        return None


def _importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:  # noqa: BLE001
        return False


def _spacy_model_name() -> Optional[str]:
    if not _importable("spacy"):
        return None
    for name in _SPACY_MODELS:
        if _importable(name):
            return name
    return None


def resolve_model(model: str = "auto") -> Optional[str]:
    """The backend key that ``model`` resolves to without loading: "fastino", "urchade", "spacy" or None.

    A GLiNER backend counts as installed when its package is importable AND its model files are cached.
    """
    model = (model or "auto").lower()
    if model == "auto":
        model = ner_model_setting()
    keys = _ORDER if model == "auto" else (model,)
    for key in keys:
        if key == "spacy":
            if _spacy_model_name():
                return "spacy"
            continue
        spec = MODELS.get(key)
        if spec and _importable(spec.package) and snapshot_dir(spec) is not None:
            return key
    return None


def ner_installed(model: str = "auto") -> bool:
    """True when a backend is installed and its model is cached. Cheap: nothing is imported or loaded."""
    if ner_setting() == "off":
        return False
    return resolve_model(model) is not None


def backend_name(model: str = "auto") -> Optional[str]:
    """Display name of the backend ``model`` resolves to ("gliner2:fastino"...), without loading it; None when
    nothing usable is installed or NER is off."""
    if ner_setting() == "off":
        return None
    key = resolve_model(model)
    if key == "fastino":
        return "gliner2:fastino"
    if key == "urchade":
        return "gliner:urchade"
    if key == "spacy":
        name = _spacy_model_name()
        return f"spacy:{name}" if name else None
    return None


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def _offline_env(cache_root: Optional[Path]) -> None:
    """Local only: set before the libraries are imported. Never overrides a value the user set."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    if cache_root is not None and not os.environ.get("HF_HUB_CACHE") and not os.environ.get("HF_HOME"):
        os.environ["HF_HUB_CACHE"] = str(cache_root)


def _set_threads() -> None:
    n = ner_threads()
    if n <= 0:
        return  # the library default (physical cores) is the fastest setting measured
    try:
        import torch  # type: ignore

        torch.set_num_threads(n)
    except Exception:  # noqa: BLE001 - threads are a tuning, never a failure
        pass


def _clean_span(chunk: str, s: int, e: int, text: Optional[str]) -> Optional[tuple[int, int]]:
    """Offsets of a returned span inside its chunk after the exact-match check and a whitespace trim."""
    if not isinstance(s, int) or not isinstance(e, int) or not (0 <= s < e <= len(chunk)):
        return None
    if text is not None and chunk[s:e] != text:
        return None  # never mask a span that is not in the text at these offsets
    while s < e and chunk[s].isspace():
        s += 1
    while e > s and chunk[e - 1].isspace():
        e -= 1
    return (s, e) if e > s else None


class Gliner2Backend:
    """``fastino/gliner2-privacy-filter-PII-multi`` through the ``gliner2`` package."""

    def __init__(self, model_dir: Path, *, kinds: Optional[Iterable[str]] = None,
                 thresholds: Optional[dict[str, float]] = None, batch_size: int = DEFAULT_BATCH):
        from gliner2 import GLiNER2  # type: ignore  # lazy: never at import of the engine

        self.name = "gliner2:fastino"
        self.labels = _select_labels(KIND_FASTINO, kinds)
        self.thresholds = _thresholds_for(self.labels, dict(_DEFAULT_THRESHOLDS, **(thresholds or {})))
        self.batch_size = batch_size
        self._model = _to_device(GLiNER2.from_pretrained(str(model_dir)))
        self._on_cpu = False
        try:
            self._model.eval()
        except Exception:  # noqa: BLE001
            pass

    def predict(self, chunks: list[str]) -> list[list[NerSpan]]:
        if not chunks:
            return []
        labels = list(self.labels)
        floor = min(self.thresholds.values()) if self.thresholds else 0.5
        results = self._extract(chunks, labels, floor, self.batch_size)
        out: list[list[NerSpan]] = []
        for chunk, res in zip(chunks, results):
            spans: list[NerSpan] = []
            ents = res.get("entities", res) if isinstance(res, dict) else {}
            for label, values in (ents or {}).items():
                if label not in self.labels:
                    continue
                for v in values or []:
                    if not isinstance(v, dict):
                        continue
                    cleaned = _clean_span(chunk, v.get("start"), v.get("end"), v.get("text"))
                    if cleaned is None:
                        continue
                    score = float(v.get("confidence", 1.0) or 0.0)
                    spans.append(NerSpan(cleaned[0], cleaned[1], label, score))
            out.append(spans)
        return out


    def _extract(self, chunks: list[str], labels: list[str], floor: float, batch_size: int):
        """The model call. A CUDA out-of-memory error halves the batch, then moves the model to the CPU: the
        caller never loses the NER pass to a memory spike (the masker would otherwise fall back to the rules)."""
        try:
            return self._model.batch_extract_entities(
                chunks, labels, batch_size=batch_size, threshold=floor, include_confidence=True, include_spans=True,
            )
        except Exception as exc:  # noqa: BLE001
            if not _is_oom(exc):
                raise
            _free_cuda()
            if batch_size > 1:
                return self._extract(chunks, labels, floor, max(1, batch_size // 2))
            if self._on_cpu:
                raise
            log.warning("RAGbase Shield: NER backend %s out of GPU memory; continuing on the CPU", self.name)
            self._model = self._model.to("cpu")
            self._on_cpu = True
            _free_cuda()
            return self._extract(chunks, labels, floor, self.batch_size)


class GlinerBackend:
    """``urchade/gliner_multi_pii-v1`` through the ``gliner`` package (GLiNER v1)."""

    def __init__(self, model_dir: Path, *, cache_root: Optional[Path] = None, kinds: Optional[Iterable[str]] = None,
                 thresholds: Optional[dict[str, float]] = None, batch_size: int = DEFAULT_BATCH):
        from gliner import GLiNER  # type: ignore

        self.name = "gliner:urchade"
        self.labels = _select_labels(KIND_URCHADE, kinds)
        self.thresholds = _thresholds_for(self.labels, dict(_URCHADE_THRESHOLDS, **(thresholds or {})))
        self.batch_size = batch_size
        kw = {"local_files_only": True}
        if cache_root is not None:
            kw["cache_dir"] = str(cache_root)  # the encoder tokenizer is resolved from this cache, offline
        self._model = _to_device(GLiNER.from_pretrained(str(model_dir), **kw))
        try:
            self._model.eval()
        except Exception:  # noqa: BLE001
            pass

    def predict(self, chunks: list[str]) -> list[list[NerSpan]]:
        if not chunks:
            return []
        labels = list(self.labels)
        floor = min(self.thresholds.values()) if self.thresholds else 0.5
        results = self._model.inference(chunks, labels, threshold=floor, flat_ner=True, batch_size=self.batch_size)
        out: list[list[NerSpan]] = []
        for chunk, ents in zip(chunks, results):
            spans: list[NerSpan] = []
            for v in ents or []:
                if not isinstance(v, dict) or v.get("label") not in self.labels:
                    continue
                cleaned = _clean_span(chunk, v.get("start"), v.get("end"), v.get("text"))
                if cleaned is None:
                    continue
                spans.append(NerSpan(cleaned[0], cleaned[1], str(v["label"]), float(v.get("score", 1.0) or 0.0)))
            out.append(spans)
        return out


class SpacyBackend:
    """The v0.1 spaCy French model, behind the same interface (PERSON and ORG only, score 1.0)."""

    def __init__(self, model_name: str, *, kinds: Optional[Iterable[str]] = None):
        import spacy  # type: ignore

        self.name = f"spacy:{model_name}"
        self.labels = _select_labels(KIND_SPACY, kinds)
        self.thresholds = {label: 0.0 for label in self.labels}
        self._nlp = spacy.load(model_name, exclude=["parser", "lemmatizer", "tagger", "morphologizer", "attribute_ruler"])
        self._nlp.max_length = max(getattr(self._nlp, "max_length", 1_000_000), 2_000_000)

    def predict(self, chunks: list[str]) -> list[list[NerSpan]]:
        out: list[list[NerSpan]] = []
        for chunk, doc in zip(chunks, self._nlp.pipe(chunks)):
            spans = []
            for ent in doc.ents:
                if ent.label_ not in self.labels:
                    continue
                cleaned = _clean_span(chunk, ent.start_char, ent.end_char, None)
                if cleaned:
                    spans.append(NerSpan(cleaned[0], cleaned[1], ent.label_, 1.0))
            out.append(spans)
        return out


class CallableBackend:
    """Any ``chunks -> [[(start, end, label, score)]]`` function behind the interface (tests, custom models)."""

    def __init__(self, fn: Callable[[list[str]], list[list[tuple]]], labels: dict[str, str], *,
                 name: str = "custom", thresholds: Optional[dict[str, float]] = None):
        self.name = name
        self.labels = dict(labels)
        self.thresholds = _thresholds_for(self.labels, dict(_DEFAULT_THRESHOLDS, **(thresholds or {})))
        self._fn = fn

    def predict(self, chunks: list[str]) -> list[list[NerSpan]]:
        out: list[list[NerSpan]] = []
        for chunk, ents in zip(chunks, self._fn(list(chunks))):
            spans = []
            for s, e, label, score in ents:
                cleaned = _clean_span(chunk, s, e, None)
                if cleaned and label in self.labels:
                    spans.append(NerSpan(cleaned[0], cleaned[1], label, float(score)))
            out.append(spans)
        return out


_LOCK = threading.RLock()
_BACKENDS: dict[str, Optional[NerBackend]] = {}
_TRIED: set[str] = set()


def load_backend(model: str = "auto", *, kinds: Optional[Iterable[str]] = None) -> Optional[NerBackend]:
    """The loaded backend for ``model`` (thread-safe singleton per key), or None when nothing usable is installed.

    Loads the model on the first call (10 to 15 s for a GLiNER model on a laptop). A load failure is logged once
    without any text and the result is None: the engine goes on with the rules.
    """
    if ner_setting() == "off":
        return None
    key = resolve_model(model)
    if key is None:
        if ner_setting() == "on" and model not in _TRIED:
            _TRIED.add(model)
            log.warning("RAGbase Shield: RAGBASE_SHIELD_NER=on but no NER model is installed; rules only "
                        "(run: ragbase-shield ner download)")
        return None
    cache_key = key if kinds is None else key + ":" + ",".join(sorted(set(kinds)))
    if key != "spacy":
        snap_now = snapshot_dir(MODELS[key])
        cache_key += "@" + str(snap_now)  # a model in another directory is another entry (a failed load is cached too)
    with _LOCK:
        if cache_key in _BACKENDS:
            return _BACKENDS[cache_key]
        backend: Optional[NerBackend] = None
        had_cache = "HF_HUB_CACHE" in os.environ
        try:
            if key == "spacy":
                name = _spacy_model_name()
                backend = SpacyBackend(name, kinds=kinds) if name else None
            else:
                spec = MODELS[key]
                snap = snapshot_dir(spec)
                root = snap.parents[2] if snap is not None and snap.parent.name == "snapshots" else None
                _offline_env(root)
                _set_threads()
                if key == "fastino":
                    backend = Gliner2Backend(snap, kinds=kinds)
                else:
                    backend = GlinerBackend(snap, cache_root=root, kinds=kinds)
            if backend is not None:
                log.info("RAGbase Shield: NER backend %s loaded (%d labels)", backend.name, len(backend.labels))
        except Exception as exc:  # noqa: BLE001 - optional component: never fail the masking
            log.warning("RAGbase Shield: NER backend %s could not be loaded (%s); rules only", key, type(exc).__name__)
            backend = None
        finally:
            if not had_cache:
                os.environ.pop("HF_HUB_CACHE", None)  # the hub cache hint was for this load only (no global change)
        _BACKENDS[cache_key] = backend
        return backend


def ner_available(model: str = "auto") -> bool:
    """True when a backend loads (the model is loaded on the first call; use ``ner_installed`` for a cheap check)."""
    return load_backend(model) is not None


def unload_backends() -> None:
    """Forget the loaded backends (tests; a process that wants the memory back)."""
    with _LOCK:
        _BACKENDS.clear()
        _TRIED.clear()


# ---------------------------------------------------------------------------
# Chunking, batching, exact offsets
# ---------------------------------------------------------------------------

_SENTENCE_END_RE = re.compile(r"[.!?…][\"'»)\]]*\s")


def chunk_text(text: str, max_chars: int = DEFAULT_CHUNK_CHARS) -> list[tuple[int, str]]:
    """Cut ``text`` into ``(offset, chunk)`` pieces of at most ``max_chars`` characters.

    Cuts prefer a blank line, then a line end, then a sentence end, then a space, in the last two thirds of the
    window; otherwise a hard cut. No overlap. The pieces concatenate back to ``text`` exactly.
    """
    if max_chars < 16:
        raise ValueError("max_chars too small")
    out: list[tuple[int, str]] = []
    n = len(text)
    pos = 0
    while pos < n:
        if n - pos <= max_chars:
            out.append((pos, text[pos:]))
            break
        window = text[pos:pos + max_chars]
        floor = max_chars // 3
        cut = -1
        i = window.rfind("\n\n", floor)
        if i != -1:
            cut = i + 2
        if cut == -1:
            i = window.rfind("\n", floor)
            if i != -1:
                cut = i + 1
        if cut == -1:
            last = None
            for m in _SENTENCE_END_RE.finditer(window, floor):
                last = m
            if last is not None:
                cut = last.end()
        if cut == -1:
            i = max(window.rfind(" ", floor), window.rfind("\t", floor))
            if i != -1:
                cut = i + 1
        if cut <= 0:
            cut = max_chars
        out.append((pos, text[pos:pos + cut]))
        pos += cut
    return out


class ChunkCache:
    """Small LRU of NER results keyed by the hash of a chunk (no text is kept, only offsets and labels)."""

    def __init__(self, max_items: int = 4096):
        self.max_items = max_items
        self._items: "OrderedDict[str, list[NerSpan]]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(backend_name: str, labels: Iterable[str], chunk: str) -> str:
        h = hashlib.sha256()
        h.update(backend_name.encode("utf-8"))
        h.update(b"\0")
        h.update(",".join(sorted(labels)).encode("utf-8"))
        h.update(b"\0")
        h.update(chunk.encode("utf-8"))
        return h.hexdigest()

    def get(self, key: str) -> Optional[list[NerSpan]]:
        v = self._items.get(key)
        if v is None:
            self.misses += 1
            return None
        self._items.move_to_end(key)
        self.hits += 1
        return v

    def put(self, key: str, spans: list[NerSpan]) -> None:
        self._items[key] = spans
        self._items.move_to_end(key)
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)

    def __len__(self) -> int:
        return len(self._items)


_SHARED_CACHE = ChunkCache()


def run_ner(text: str, backend: NerBackend, *, kinds: Optional[Iterable[str]] = None,
            max_chars: int = DEFAULT_CHUNK_CHARS, batch_size: int = DEFAULT_BATCH,
            cache: Optional[ChunkCache] = _SHARED_CACHE) -> list[tuple[int, int, str, str, float]]:
    """Run ``backend`` over ``text`` in chunks and batches: ``[(start, end, kind, label, score), ...]``.

    Offsets are in ``text``. Every span was checked against the text (the backend drops a span whose text is not at
    its offsets), scores are kept, a span under the backend's threshold for its label is dropped, and one span with
    two labels keeps the best score. The result is sorted by position.
    """
    if not text or backend is None:
        return []
    want = set(kinds) if kinds is not None else None
    labels = backend.labels
    pieces = chunk_text(text, max_chars)
    todo: list[tuple[int, str, str]] = []   # (index, key, chunk)
    results: list[Optional[list[NerSpan]]] = [None] * len(pieces)
    for i, (_off, chunk) in enumerate(pieces):
        if not chunk.strip():
            results[i] = []
            continue
        key = ChunkCache.key(backend.name, labels, chunk) if cache is not None else ""
        hit = cache.get(key) if cache is not None else None
        if hit is not None:
            results[i] = hit
        else:
            todo.append((i, key, chunk))
    for b in range(0, len(todo), max(1, batch_size)):
        batch = todo[b:b + max(1, batch_size)]
        preds = backend.predict([c for _i, _k, c in batch])
        for (i, key, _chunk), spans in zip(batch, preds):
            spans = list(spans)
            results[i] = spans
            if cache is not None:
                cache.put(key, spans)
    best: dict[tuple[int, int], tuple[int, int, str, str, float]] = {}
    for (off, chunk), spans in zip(pieces, results):
        for sp in spans or ():
            if not (0 <= sp.start < sp.end <= len(chunk)):
                continue
            s, e = off + sp.start, off + sp.end
            if text[s:e] != chunk[sp.start:sp.end]:
                continue
            kind = labels.get(sp.label)
            if kind is None or (want is not None and kind not in want):
                continue
            if sp.score < backend.thresholds.get(sp.label, 0.5):
                continue
            prev = best.get((s, e))
            if prev is None or sp.score > prev[4]:
                best[(s, e)] = (s, e, kind, sp.label, float(sp.score))
    return sorted(best.values())


def as_hook(backend: Optional[NerBackend], *, kinds: Iterable[str] = ("PERSON", "ORG"), lang: str = "fr",
            max_text_chars: Optional[int] = None, cache: Optional[ChunkCache] = _SHARED_CACHE,
            **guard_kw) -> Callable[[str], list[tuple[int, int, str]]]:
    """Adapter for the v0.1 ``Masker(ner=...)`` plumbing: ``text -> [(start, end, kind)]``.

    Runs ``run_ner`` then the NER-only precision guards of ``merge.guard_ner_spans`` (whitelist and protected
    intervals, stop lists, vocabulary, acronyms, all-capital lines, one-word people). Text longer than
    ``max_text_chars`` (default: RAGBASE_SHIELD_NER_MAX_CHARS) gets no NER: the rules alone run on it.
    ``guard_kw`` goes to ``merge.guard_ner_spans`` (``one_word_person_min``, ``one_word_org_min``, ...).
    """
    from . import merge as mg
    from . import names as nm

    want = tuple(kinds)

    def hook(text: str) -> list[tuple[int, int, str]]:
        if backend is None or not text:
            return []
        limit = ner_max_chars() if max_text_chars is None else max_text_chars
        if limit and len(text) > limit:
            return []
        spans = run_ner(text, backend, kinds=want, cache=cache)
        # the rules' view of the text: "rules win on kind", and the acronym rule sees the organisations they found
        rules = nm.detect_names(text)
        kept = mg.merge_ner(text, spans, rules, (), None, lang=lang, kinds=want, **guard_kw)
        return [(m.start, m.end, m.kind) for m in kept]

    return hook


# ---------------------------------------------------------------------------
# Download (one explicit user step) and status
# ---------------------------------------------------------------------------


def download_model(model: str = "fastino", *, models_dir_path: Optional[Path] = None, dry_run: bool = False,
                   force: bool = False, log_fn: Callable[[str], None] = print) -> list[Path]:
    """Download a model into the models directory (Hugging Face hub cache layout) and verify its files.

    ``model``: "fastino", "urchade" or "all". ``dry_run`` prints the plan and returns without any network call.
    A model already present is kept unless ``force``. Needs the ``ner`` extra (``pip install "ragbase-shield[ner]"``).
    Returns the snapshot directories.
    """
    keys = tuple(MODELS) if model == "all" else (model,)
    for k in keys:
        if k not in MODELS:
            raise ValueError(f"unknown model {k!r}; choose fastino, urchade or all")
    target = Path(models_dir_path) if models_dir_path is not None else models_dir()
    out: list[Path] = []
    for k in keys:
        spec = MODELS[k]
        have = snapshot_dir(spec, roots=[target])
        extras = ", ".join(r for r, _p in spec.extra_repos)
        log_fn(f"{spec.repo} ({spec.licence}, about {spec.size_mb} MB{', + ' + extras if extras else ''})"
               f" -> {target}" + (" [already present]" if have else ""))
        if dry_run:
            continue
        if have is None or force:
            if not _importable("huggingface_hub"):
                raise RuntimeError("huggingface_hub is not installed: pip install \"ragbase-shield[ner]\"")
            from huggingface_hub import snapshot_download  # type: ignore  # lazy: the download step only

            offline = os.environ.pop("HF_HUB_OFFLINE", None)  # this one call may use the network
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
            try:
                target.mkdir(parents=True, exist_ok=True)
                snapshot_download(spec.repo, cache_dir=str(target), ignore_patterns=list(spec.ignore_patterns))
                for repo, patterns in spec.extra_repos:
                    snapshot_download(repo, cache_dir=str(target), allow_patterns=list(patterns))
            finally:
                if offline is not None:
                    os.environ["HF_HUB_OFFLINE"] = offline
        snap = snapshot_dir(spec, roots=[target])
        if snap is None:
            raise RuntimeError(f"{spec.repo}: download finished but the model files are incomplete in {target}")
        log_fn(f"  ok: {snap} ({_dir_size_mb(snap)} MB on disk)")
        out.append(snap)
    if not dry_run:
        log_fn("Turn it on: RAGBASE_SHIELD_NER=on" + (f" RAGBASE_SHIELD_NER_MODEL={keys[0]}" if len(keys) == 1 else ""))
    return out


def status(model: str = "auto") -> dict:
    """Packages, cached models and the resolved backend. Nothing is loaded; no path holds client data."""
    packages = {name: _package_version(name) for name in ("gliner2", "gliner", "torch", "transformers", "spacy")}
    models: dict[str, dict] = {}
    for key, spec in MODELS.items():
        snap = snapshot_dir(spec)
        models[key] = {"repo": spec.repo, "package": spec.package, "package_installed": _importable(spec.package),
                       "cached": snap is not None, "path": str(snap) if snap else None,
                       "size_mb": _dir_size_mb(snap) if snap else None, "licence": spec.licence}
    spacy_model = _spacy_model_name()
    models["spacy"] = {"repo": spacy_model, "package": "spacy", "package_installed": _importable("spacy"),
                       "cached": spacy_model is not None, "path": None, "size_mb": None, "licence": "MIT"}
    loaded = [k for k, b in _BACKENDS.items() if b is not None]
    name = backend_name(model)
    return {
        "setting": ner_setting(), "model_setting": ner_model_setting(), "models_dir": str(models_dir()),
        "max_chars": ner_max_chars(), "threads": ner_threads(), "packages": packages, "models": models,
        "backend": name, "mode": f"rules+ner:{name}" if name else "rules", "loaded": loaded,
    }


def status_lines(lang: str = "fr") -> list[tuple[str, str, str]]:
    """Lines for ``ragbase-shield doctor`` and ``ner status``: (level, label, value)."""
    st = status()
    fr = lang == "fr"
    out: list[tuple[str, str, str]] = []
    if st["setting"] == "off":
        out.append(("INFO", "Détection NER (option)" if fr else "NER (optional)",
                    "désactivée (RAGBASE_SHIELD_NER=off)" if fr else "disabled (RAGBASE_SHIELD_NER=off)"))
    elif st["backend"]:
        out.append(("OK", "Détection NER (option)" if fr else "NER (optional)", st["backend"]))
    else:
        level = "WARN" if st["setting"] == "on" else "INFO"
        out.append((level, "Détection NER (option)" if fr else "NER (optional)",
                    "non installée (règles seules) : ragbase-shield ner download" if fr
                    else "not installed (rules only): ragbase-shield ner download"))
    for key, m in st["models"].items():
        if key == "spacy" and not m["package_installed"]:
            continue
        state = ("présent" if fr else "cached") if m["cached"] else ("absent" if fr else "not cached")
        pkg = "" if m["package_installed"] else (" ; paquet manquant" if fr else "; package missing")
        size = f" ({m['size_mb']} MB)" if m.get("size_mb") else ""
        out.append(("INFO", f"  modèle {key}" if fr else f"  model {key}", f"{state}{size}{pkg}"))
    out.append(("INFO", "  dossier des modèles" if fr else "  models directory", st["models_dir"]))
    return out
