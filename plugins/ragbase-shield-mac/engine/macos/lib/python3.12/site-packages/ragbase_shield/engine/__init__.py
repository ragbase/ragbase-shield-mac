"""RAGbase Shield detection engine: local, rules-first pseudonymisation for French legal text.

Public API (contract: ARCHITECTURE.md section 4.1)::

    from ragbase_shield.engine import Masker, MemoryStore
    m = Masker(MemoryStore())
    r = m.mask("M. Jean-Pierre Dupont, IBAN FR76 3000 6000 0112 3456 7890 189")
    r.text            # "M. PERSONNE_A, IBAN IBAN_1"
    m.unmask(r.text)  # restores the real values, locally
    m.scan(text)      # canary: known values and valid identifiers still present

The names below are imported on first use (PEP 562), so ``import ragbase_shield.engine.types`` (the vault, the
PreToolUse hook) does not load the detectors and the lexicons.
"""
from __future__ import annotations

import importlib
from typing import Any

# Eager on purpose (light module): the submodule "normalize" would otherwise shadow the function "normalize".
from .normalize import Folded, fold, normalize

_EXPORTS = {
    "detectors": ("detect_candidates", "detect_structured", "find_suspects", "iban_valid", "isin_valid",
                  "lei_valid", "luhn_valid", "nir_valid", "siren_valid", "siret_valid", "structured_norm",
                  "url_is_public", "vat_fr_valid", "aba_routing_valid", "shape_template", "PUBLIC_HANDLES"),
    "locale": ("detect_language", "casual_text", "bundle", "detect_names_en", "protected_intervals_en"),
    "masker": ("UNMASK_RE", "Masker", "entity_norm", "escape_codes", "canonical_codes", "code_like_spans",
               "ner_available", "ner_installed", "prepare_scan", "ScanPrep"),
    "merge": ("merge_ner", "guard_ner_spans", "is_name_word"),
    "names": ("detect_names", "has_party_separator", "org_core", "parse_parties", "person_norm", "split_person"),
    "ner": ("load_backend", "run_ner", "as_hook", "chunk_text", "download_model", "backend_name"),
    "synthetic": ("Synthesizer", "SYNTHETIC_TLD", "shift_date_text"),
    "types": ("CODE_RE", "KINDS", "PREFIXES", "SHAPE_KINDS", "Entity", "EntityStore", "MaskResult", "MemoryStore",
              "Span", "make_code", "prefix_for", "CodeRangeError"),
}
_WHERE = {name: module for module, names in _EXPORTS.items() for name in names}

__all__ = sorted(set(_WHERE) | {"Folded", "fold", "normalize"})


def __getattr__(name: str) -> Any:
    module = _WHERE.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
