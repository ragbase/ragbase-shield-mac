"""The only client connection in the package: GET on Shield Desk at 127.0.0.1 (packaging/DESIGN.md section 4).

The background service and its menu bar or tray icon ask the local Desk for its status and its dossier list.
The host is the constant ``127.0.0.1`` (never configurable, never a name to resolve): nothing here can reach
another computer. tests/test_local_only.py allows ``http`` in this file only and checks the host.
"""
from __future__ import annotations

import errno
import http.client
import json
from typing import Any, Optional

LOOPBACK = "127.0.0.1"


def get_json(port: int, path: str, token: Optional[str], timeout: float) -> tuple[str, Any]:
    """("ok", json) on 200, ("refused", None) when nothing listens, ("busy", status or None) otherwise."""
    if not path.startswith("/api/"):
        raise ValueError("loopback: /api/ paths only")
    conn = http.client.HTTPConnection(LOOPBACK, int(port), timeout=timeout)
    try:
        try:
            conn.connect()
        except TimeoutError:
            # Windows sends the SYN to a closed loopback port again for about 2 s before it says "refused", so a
            # short probe times out there. A listening socket completes the handshake in the kernel at once: on
            # 127.0.0.1 a handshake that does not complete means nothing listens (found on the Windows VM: the
            # service took a free port for a busy one and never started the Desk).
            return "refused", None
        headers = {"Accept": "application/json"}
        if token:
            headers["X-Shield-Token"] = token
        conn.request("GET", path, headers=headers)
        resp = conn.getresponse()
        body = resp.read(4 * 1024 * 1024)
        if resp.status != 200:
            return "busy", resp.status
        try:
            return "ok", json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return "busy", resp.status
    except ConnectionRefusedError:
        return "refused", None
    except OSError as exc:
        if getattr(exc, "errno", None) in (errno.ECONNREFUSED, getattr(errno, "WSAECONNREFUSED", -1), 10061):
            return "refused", None
        return "busy", None
    except http.client.HTTPException:
        return "busy", None
    finally:
        conn.close()
