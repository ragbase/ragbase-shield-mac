"""Paths and settings. All data stays on this computer."""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP = "ragbase-shield"


def _windows_home() -> Path:
    """%LOCALAPPDATA%\\ragbase-shield. Never the roaming %APPDATA%: law firms often redirect it to a file
    server, and SQLite (WAL mode) is not safe on a network file system. A folder left in %APPDATA% by an
    earlier version is moved once."""
    local = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")) / APP
    roaming = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming")) / APP
    if not local.exists() and roaming.is_dir() and roaming != local:
        try:
            import shutil

            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(roaming), str(local))
        except OSError:
            return roaming  # could not move it (in use?): keep using the old place rather than lose data
    return local


def home() -> Path:
    """Data directory: $RAGBASE_SHIELD_HOME, else %LOCALAPPDATA%\\ragbase-shield on Windows, else ~/.ragbase-shield."""
    env = os.environ.get("RAGBASE_SHIELD_HOME")
    if env:
        p = Path(env).expanduser()
    elif sys.platform.startswith("win"):
        p = _windows_home()
    else:
        p = Path.home() / f".{APP}"
    p.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(p, 0o700)
    except OSError:
        pass
    return p


def locale() -> str:
    """Code labels: "fr" (PERSONNE_A) by default, "en" (PERSON_A) with RAGBASE_SHIELD_LOCALE=en."""
    v = os.environ.get("RAGBASE_SHIELD_LOCALE", "fr").lower()
    return "en" if v.startswith("en") else "fr"


def replacement() -> str:
    """Default replacement of the masked mentions (v0.2): "codes" (PERSONNE_A) or "synthetic" (realistic
    pseudonyms), from RAGBASE_SHIELD_REPLACEMENT. A dossier setting overrides it; hooks and the canary always
    use codes."""
    v = os.environ.get("RAGBASE_SHIELD_REPLACEMENT", "codes").strip().lower()
    return "synthetic" if v == "synthetic" else "codes"


OUTPUT_DIR_NAME = "Shield - sorties"
DESK_PORT = int(os.environ.get("RAGBASE_SHIELD_PORT", "7878"))


def language() -> str:
    """Text language for the detection rules (v0.2): "auto" (detected per text, default French), "fr" or "en",
    from RAGBASE_SHIELD_LANGUAGE. A dossier setting overrides it."""
    from .engine.locale import language_setting  # light module

    return language_setting()


def ner_setting() -> str:
    """RAGBASE_SHIELD_NER: "auto" (default: a model when one is installed), "on" or "off" (v0.2)."""
    from .engine.ner import ner_setting as _ner_setting

    return _ner_setting()


def ner_model() -> str:
    """RAGBASE_SHIELD_NER_MODEL: "auto" (default), "fastino", "urchade" or "spacy" (v0.2)."""
    from .engine.ner import ner_model_setting

    return ner_model_setting()


def models_dir():
    """Where ``ragbase-shield ner download`` puts the models: RAGBASE_SHIELD_MODELS_DIR, else ``<home>/models``."""
    from .engine.ner import models_dir as _models_dir

    return _models_dir()


def mask_legal_form() -> bool:
    """RAGBASE_SHIELD_MASK_LEGAL_FORM (v0.2, DETECTORS): "1" masks the legal form with the company name
    ("ACME SAS" -> SOCIETE_A); default "0" keeps it in clear ("SOCIETE_A SAS")."""
    return os.environ.get("RAGBASE_SHIELD_MASK_LEGAL_FORM", "0").strip().lower() in ("1", "true", "yes", "on")


def lowercase_names() -> bool:
    """RAGBASE_SHIELD_LOWERCASE_NAMES (v0.4.1): "0" turns off the rules for names and addresses typed in lower case
    in French text ("m. dupont", "mon client jean dupont", "la société durand sas", "12 rue des lilas, 69007 lyon")
    and gives the 0.4.0 detection; default "1". The engine reads the same variable at each mask call
    (``engine.names.lowercase_names``)."""
    return os.environ.get("RAGBASE_SHIELD_LOWERCASE_NAMES", "1").strip().lower() not in ("0", "false", "no", "off")


def own_company_vendor() -> bool:
    """RAGBASE_SHIELD_OWN_COMPANY_VENDOR (v0.4, P1, policy option): "1" masks a public tool or vendor of the English
    whitelist when it is the dossier's own company (the label of the dossier's main e-mail domain); default "0"
    keeps every vendor readable, as before. Whether it becomes the default is an open decision."""
    from .engine.locale import own_company_vendor_setting  # light module

    return own_company_vendor_setting()
