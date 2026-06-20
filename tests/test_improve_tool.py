"""Tests de l'outil self_improve (câblage de SelfImprover dans l'agent)."""

from __future__ import annotations

from pathlib import Path

from kira.improve import SelfImprover
from kira.tools.improve_tool import SelfImproveTool


def _setup(tmp_path: Path, passing: bool = True) -> SelfImproveTool:
    (tmp_path / "kira").mkdir()
    (tmp_path / "kira" / "foo.py").write_text("X = 1\n", encoding="utf-8")
    improver = SelfImprover(
        tmp_path, max_minor_lines=6, test_runner=lambda: (passing, "")
    )
    return SelfImproveTool(improver)


def test_view_returns_content(tmp_path):
    tool = _setup(tmp_path)
    assert tool.run({"action": "view", "path": "kira/foo.py"}) == "X = 1\n"


def test_view_refuses_non_editable(tmp_path):
    tool = _setup(tmp_path)
    out = tool.run({"action": "view", "path": "policy.yaml"})
    assert out.lower().startswith("refusé")


def test_propose_minor_applied(tmp_path):
    tool = _setup(tmp_path, passing=True)
    out = tool.run({"action": "propose", "path": "kira/foo.py", "new_content": "X = 2\n"})
    assert "APPLIQUÉ" in out
    assert (tmp_path / "kira" / "foo.py").read_text(encoding="utf-8") == "X = 2\n"


def test_propose_rejected_when_tests_fail(tmp_path):
    tool = _setup(tmp_path, passing=False)
    out = tool.run({"action": "propose", "path": "kira/foo.py", "new_content": "X = 99\n"})
    assert "REFUSÉ" in out
    assert "ROUGES" in out
    # Inchangé.
    assert (tmp_path / "kira" / "foo.py").read_text(encoding="utf-8") == "X = 1\n"


def test_propose_protected_needs_approval(tmp_path):
    (tmp_path / "kira").mkdir()
    (tmp_path / "kira" / "security.py").write_text("S = 1\n", encoding="utf-8")
    improver = SelfImprover(tmp_path, test_runner=lambda: (True, ""))
    tool = SelfImproveTool(improver)
    out = tool.run({"action": "propose", "path": "kira/security.py", "new_content": "S = 2\n"})
    assert "VALIDATION HUMAINE" in out
    # Le fichier de sécurité reste intact.
    assert (tmp_path / "kira" / "security.py").read_text(encoding="utf-8") == "S = 1\n"


def test_propose_missing_content(tmp_path):
    tool = _setup(tmp_path)
    out = tool.run({"action": "propose", "path": "kira/foo.py"})
    assert "Erreur" in out
