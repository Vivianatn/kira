"""Tests des outils sûrs (files en lecture seule). web n'est pas testé en
réseau ici (dépend d'un service externe)."""

from __future__ import annotations

from kira.tools.files import FilesTool


def test_files_read_within_workspace(security):
    tool = FilesTool(security)
    out = tool.run({"action": "read", "path": "notes.txt"})
    assert "bonjour kira" in out


def test_files_list_workspace(security):
    tool = FilesTool(security)
    out = tool.run({"action": "list", "path": "."})
    assert "notes.txt" in out


def test_files_read_escape_refused(security):
    tool = FilesTool(security)
    out = tool.run({"action": "read", "path": "../policy.yaml"})
    assert out.lower().startswith("refusé")


def test_files_read_missing_file(security):
    tool = FilesTool(security)
    out = tool.run({"action": "read", "path": "absent.txt"})
    assert "introuvable" in out


def test_registry_only_builds_allowed_tools(security):
    from kira.tools import build_registry

    registry = build_registry(security)
    assert set(registry.keys()) == {"web", "files"}
