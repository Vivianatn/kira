"""Tests de la logique cœur du pont MCP (sans lancer le serveur MCP)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    """Recharge le module en redirigeant le store vers un tmp dir isolé."""
    import mcp_bridge.server as server

    importlib.reload(server)
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server, "MSG_FILE", tmp_path / "messages.jsonl")
    monkeypatch.setattr(server, "CURSOR_FILE", tmp_path / "cursors.json")
    from filelock import FileLock

    monkeypatch.setattr(server, "LOCK", FileLock(str(tmp_path / "bridge.lock")))
    return server


def test_post_then_read_unread(bridge):
    bridge.core_post("claude-code", "salut Cursor", to="cursor")
    out = bridge.core_read("cursor")
    assert "salut Cursor" in out
    # Une 2e lecture ne renvoie plus rien (déjà marqué lu).
    assert bridge.core_read("cursor") == "(aucun nouveau message)"


def test_sender_does_not_receive_own_message(bridge):
    bridge.core_post("cursor", "note pour Claude", to="all")
    # L'expéditeur ne reçoit pas son propre message.
    assert bridge.core_read("cursor") == "(aucun nouveau message)"
    # Le destinataire 'all' le voit.
    assert "note pour Claude" in bridge.core_read("claude-code")


def test_peek_does_not_mark_read(bridge):
    bridge.core_post("claude-code", "ping", to="cursor")
    assert "ping" in bridge.core_peek("cursor")
    # peek n'a pas consommé : read le voit encore.
    assert "ping" in bridge.core_read("cursor")


def test_directed_message_not_seen_by_others(bridge):
    bridge.core_post("claude-code", "perso", to="cursor")
    # Adressé à cursor uniquement : claude-code ne le lit pas.
    assert bridge.core_read("claude-code") == "(aucun nouveau message)"


def test_empty_message_rejected(bridge):
    assert "vide" in bridge.core_post("cursor", "   ")


def test_only_unread_false_relit_tout(bridge):
    bridge.core_post("claude-code", "m1", to="cursor")
    bridge.core_read("cursor")  # consomme
    out = bridge.core_read("cursor", only_unread=False)
    assert "m1" in out
