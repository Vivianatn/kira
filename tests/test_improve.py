"""Tests de l'auto-amélioration encadrée. Les tests injectent un faux runner
(on ne lance pas la vraie suite pytest ici)."""

from __future__ import annotations

from pathlib import Path

import pytest

from kira.improve import SelfImprover


def _project(tmp_path: Path) -> Path:
    (tmp_path / "kira").mkdir()
    (tmp_path / "kira" / "foo.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (tmp_path / "kira" / "security.py").write_text("# garde-fou\nX = 1\n", encoding="utf-8")
    return tmp_path


def _improver(root: Path, passing: bool = True, max_minor: int = 6) -> SelfImprover:
    return SelfImprover(
        root, max_minor_lines=max_minor, test_runner=lambda: (passing, "")
    )


def test_minor_change_auto_applied_when_tests_pass(tmp_path):
    root = _project(tmp_path)
    imp = _improver(root, passing=True)
    res = imp.propose("kira/foo.py", "def f():\n    return 2\n")
    assert res.applied is True
    assert res.needs_approval is False
    # Le fichier a bien été modifié.
    assert "return 2" in (root / "kira" / "foo.py").read_text(encoding="utf-8")


def test_change_reverted_when_tests_fail(tmp_path):
    root = _project(tmp_path)
    imp = _improver(root, passing=False)
    res = imp.propose("kira/foo.py", "def f():\n    return 999\n")
    assert res.applied is False
    assert res.tests_passed is False
    # Revenu à l'original (fitness échouée).
    assert "return 1" in (root / "kira" / "foo.py").read_text(encoding="utf-8")


def test_protected_file_needs_approval(tmp_path):
    root = _project(tmp_path)
    imp = _improver(root, passing=True)
    res = imp.propose("kira/security.py", "# garde-fou\nX = 2\n")
    assert res.applied is False
    assert res.needs_approval is True
    # L'original de sécurité est intact.
    assert "X = 1" in (root / "kira" / "security.py").read_text(encoding="utf-8")
    # Une proposition a été écrite pour revue humaine.
    assert res.proposal_path and Path(res.proposal_path).exists()


def test_major_change_needs_approval(tmp_path):
    root = _project(tmp_path)
    imp = _improver(root, passing=True, max_minor=2)
    big = "def f():\n" + "".join(f"    x{i} = {i}\n" for i in range(10)) + "    return 1\n"
    res = imp.propose("kira/foo.py", big)
    assert res.applied is False
    assert res.needs_approval is True
    # Non appliqué : l'original est conservé.
    assert (root / "kira" / "foo.py").read_text(encoding="utf-8") == "def f():\n    return 1\n"


def test_out_of_scope_rejected(tmp_path):
    root = _project(tmp_path)
    (root / "secret.txt").write_text("nope", encoding="utf-8")
    imp = _improver(root)
    res = imp.propose("secret.txt", "hacked")
    assert res.applied is False
    assert res.needs_approval is False
    assert "périmètre" in res.reason


def test_no_change_is_noop(tmp_path):
    root = _project(tmp_path)
    imp = _improver(root)
    res = imp.propose("kira/foo.py", "def f():\n    return 1\n")
    assert res.applied is False
    assert "aucun changement" in res.reason


def test_new_file_minor_applied(tmp_path):
    root = _project(tmp_path)
    imp = _improver(root, passing=True)
    res = imp.propose("kira/bar.py", "VALUE = 42\n")
    assert res.applied is True
    assert (root / "kira" / "bar.py").exists()
