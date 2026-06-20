"""Tests de l'écriture de fichiers (mode read_write) + confinement et allowlist."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from kira.security import Action, EnforcementLayer
from kira.tools.files import FilesTool


@pytest.fixture
def rw(tmp_path: Path):
    """Projet read_write : workspace + un dossier 'autorisé' supplémentaire."""
    (tmp_path / "workspace").mkdir()
    granted = tmp_path / "granted"
    granted.mkdir()
    forbidden = tmp_path / "forbidden"
    forbidden.mkdir()
    policy = {
        "version": 1,
        "allowed_tools": ["files"],
        "tools": {
            "files": {
                "root": "workspace",
                "mode": "read_write",
                "max_bytes": 100000,
                "writable_paths": [str(granted)],
            }
        },
        "require_human_approval": [{"tool": "files", "action": "delete"}],
    }
    (tmp_path / "policy.yaml").write_text(yaml.safe_dump(policy), encoding="utf-8")
    sec = EnforcementLayer(policy_path=tmp_path / "policy.yaml", project_root=tmp_path)
    return sec, tmp_path, granted, forbidden


def test_write_and_read_in_workspace(rw):
    sec, root, _, _ = rw
    tool = FilesTool(sec)
    assert "OK" in tool.run({"action": "write", "path": "note.txt", "content": "salut"})
    assert (root / "workspace" / "note.txt").read_text(encoding="utf-8") == "salut"
    assert tool.run({"action": "read", "path": "note.txt"}) == "salut"


def test_append(rw):
    sec, root, _, _ = rw
    tool = FilesTool(sec)
    tool.run({"action": "write", "path": "a.txt", "content": "ligne1\n"})
    tool.run({"action": "append", "path": "a.txt", "content": "ligne2\n"})
    assert (root / "workspace" / "a.txt").read_text(encoding="utf-8") == "ligne1\nligne2\n"


def test_mkdir(rw):
    sec, root, _, _ = rw
    tool = FilesTool(sec)
    assert "OK" in tool.run({"action": "mkdir", "path": "sub/dir"})
    assert (root / "workspace" / "sub" / "dir").is_dir()


def test_write_outside_allowed_is_refused(rw):
    sec, _, _, forbidden = rw
    tool = FilesTool(sec)
    out = tool.run({"action": "write", "path": str(forbidden / "x.txt"), "content": "no"})
    assert out.lower().startswith("refusé")
    assert not (forbidden / "x.txt").exists()


def test_write_to_granted_path_works(rw):
    sec, _, granted, _ = rw
    tool = FilesTool(sec)
    out = tool.run({"action": "write", "path": str(granted / "out.txt"), "content": "ok"})
    assert "OK" in out
    assert (granted / "out.txt").read_text(encoding="utf-8") == "ok"


def test_delete_requires_human_approval(rw):
    sec, _, _, _ = rw
    action = Action(tool="files", name="delete", params={"path": "note.txt"})
    assert sec.is_allowed(action) is True
    assert sec.requires_human_approval(action) is True


def test_delete_removes_file(rw):
    sec, root, _, _ = rw
    tool = FilesTool(sec)
    tool.run({"action": "write", "path": "tmp.txt", "content": "x"})
    assert "OK" in tool.run({"action": "delete", "path": "tmp.txt"})
    assert not (root / "workspace" / "tmp.txt").exists()


def test_readonly_policy_blocks_write(tmp_path):
    (tmp_path / "workspace").mkdir()
    policy = {
        "version": 1,
        "allowed_tools": ["files"],
        "tools": {"files": {"root": "workspace", "mode": "read_only"}},
    }
    (tmp_path / "policy.yaml").write_text(yaml.safe_dump(policy), encoding="utf-8")
    sec = EnforcementLayer(policy_path=tmp_path / "policy.yaml", project_root=tmp_path)
    out = FilesTool(sec).run({"action": "write", "path": "x.txt", "content": "no"})
    assert out.lower().startswith("refusé")
