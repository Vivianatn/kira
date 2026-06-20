"""Serveur MCP « kira-bridge » — canal de messages partagé Cursor × Claude Code.

But : donner aux deux outils IA un canal commun *live*. Chaque outil lance sa
propre instance de ce serveur (en stdio), mais toutes les instances lisent/écrivent
le MÊME store local (`mcp_bridge/.data/`). Tant que les deux sessions sont
ouvertes, un message posté par l'un est lisible immédiatement par l'autre — sans
passer par git.

⚠️ Limite intrinsèque : ni Cursor ni Claude Code ne tournent en arrière-plan.
Le transport est temps réel, mais chaque agent ne LIT le canal que lorsqu'il est
actif (déclenché par l'humain). Ce pont supprime le détour git, pas le besoin de
lancer l'autre outil.

Outils exposés :
    - post_message(sender, body, to="all")  : poster un message.
    - read_messages(reader, only_unread=True): lire (et marquer lus) les messages.
    - peek_messages(reader, limit=20)        : lire SANS marquer lus.

Lancement (configuré dans .mcp.json et .cursor/mcp.json) :
    D:\\kira\\.python\\python.exe D:\\kira\\mcp_bridge\\server.py
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from filelock import FileLock

# --------------------------------------------------------------------------- #
# Store partagé (local au dépôt, mais NON committé — voir .gitignore)
# --------------------------------------------------------------------------- #
DATA_DIR = Path(__file__).resolve().parent / ".data"
DATA_DIR.mkdir(exist_ok=True)
MSG_FILE = DATA_DIR / "messages.jsonl"
CURSOR_FILE = DATA_DIR / "cursors.json"
LOCK = FileLock(str(DATA_DIR / "bridge.lock"))

KNOWN = {"cursor", "claude-code", "all"}


# --------------------------------------------------------------------------- #
# Logique cœur (testable sans MCP)
# --------------------------------------------------------------------------- #
def _read_all() -> list[dict]:
    if not MSG_FILE.exists():
        return []
    out: list[dict] = []
    for line in MSG_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _load_cursors() -> dict[str, float]:
    if CURSOR_FILE.exists():
        return json.loads(CURSOR_FILE.read_text(encoding="utf-8"))
    return {}


def _save_cursors(cursors: dict[str, float]) -> None:
    CURSOR_FILE.write_text(json.dumps(cursors), encoding="utf-8")


def _normalize(name: str) -> str:
    return (name or "").strip().lower()


def core_post(sender: str, body: str, to: str = "all") -> str:
    sender = _normalize(sender)
    to = _normalize(to) or "all"
    if not body.strip():
        return "Erreur : message vide."
    msg = {
        "id": uuid.uuid4().hex[:8],
        "ts": time.time(),
        "sender": sender,
        "to": to,
        "body": body,
    }
    with LOCK:
        with MSG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(msg, ensure_ascii=False) + "\n")
    return f"Message #{msg['id']} posté ({sender} -> {to})."


def _select(reader: str, msgs: list[dict], since: float) -> list[dict]:
    """Messages adressés à `reader` (ou 'all'), pas envoyés par lui, après `since`."""
    return [
        m
        for m in msgs
        if m["to"] in (reader, "all") and m["sender"] != reader and m["ts"] > since
    ]


def _format(msgs: list[dict]) -> str:
    if not msgs:
        return "(aucun nouveau message)"
    lines = []
    for m in msgs:
        t = time.strftime("%Y-%m-%d %H:%M", time.localtime(m["ts"]))
        lines.append(f"[{t}] {m['sender']} -> {m['to']} (#{m['id']}):\n{m['body']}")
    return "\n\n".join(lines)


def core_read(reader: str, only_unread: bool = True, limit: int = 50) -> str:
    reader = _normalize(reader)
    with LOCK:
        msgs = _read_all()
        cursors = _load_cursors()
        since = cursors.get(reader, 0.0) if only_unread else 0.0
        selected = _select(reader, msgs, since)[-limit:]
        if selected and only_unread:
            cursors[reader] = max(m["ts"] for m in selected)
            _save_cursors(cursors)
    return _format(selected)


def core_peek(reader: str, limit: int = 20) -> str:
    """Lit sans avancer le curseur (ne marque pas comme lu)."""
    reader = _normalize(reader)
    with LOCK:
        msgs = _read_all()
    return _format(_select(reader, msgs, 0.0)[-limit:])


# --------------------------------------------------------------------------- #
# Couche MCP
# --------------------------------------------------------------------------- #
def build_server():
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("kira-bridge")

    @mcp.tool()
    def post_message(sender: str, body: str, to: str = "all") -> str:
        """Poste un message sur le canal partagé Kira.

        sender : qui écrit — 'claude-code' ou 'cursor'.
        body   : le contenu du message.
        to     : destinataire — 'cursor', 'claude-code' ou 'all' (défaut).
        """
        return core_post(sender, body, to)

    @mcp.tool()
    def read_messages(reader: str, only_unread: bool = True, limit: int = 50) -> str:
        """Lit les messages adressés à `reader` ('cursor'|'claude-code').

        Par défaut, ne renvoie que les NON LUS et avance le curseur de lecture
        (les marque comme lus). Mets only_unread=False pour tout relire.
        """
        return core_read(reader, only_unread, limit)

    @mcp.tool()
    def peek_messages(reader: str, limit: int = 20) -> str:
        """Affiche les derniers messages pour `reader` SANS les marquer comme lus."""
        return core_peek(reader, limit)

    return mcp


if __name__ == "__main__":
    build_server().run()
