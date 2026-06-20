"""Tests de la couche d'enforcement — le garde-fou doit bloquer ce qu'il faut."""

from __future__ import annotations

import yaml

from kira.security import Action, EnforcementLayer, SecurityError


def test_tool_in_allowlist_is_allowed(security):
    assert security.is_allowed(Action(tool="web", name="search", params={"query": "x"}))
    assert security.is_allowed(
        Action(tool="files", name="read", params={"path": "notes.txt"})
    )


def test_tool_outside_allowlist_is_blocked(security):
    # 'system' (lancement de programme) n'est PAS dans l'allowlist -> refusé.
    action = Action(tool="system", name="run", params={"cmd": "calc.exe"})
    assert security.is_allowed(action) is False
    decision = security.evaluate(action)
    assert decision.allowed is False
    assert "hors allowlist" in decision.reason


def test_check_raises_on_blocked_action(security):
    action = Action(tool="system", name="run", params={"cmd": "rm -rf /"})
    try:
        security.check(action)
    except SecurityError as exc:
        assert "refus" in str(exc).lower()
    else:
        raise AssertionError("check() aurait dû lever SecurityError")


def test_path_escape_is_blocked(security):
    # Tentative d'évasion hors du workspace via '..'.
    action = Action(tool="files", name="read", params={"path": "../policy.yaml"})
    assert security.is_allowed(action) is False


def test_absolute_path_outside_root_is_blocked(security):
    action = Action(tool="files", name="read", params={"path": "C:/Windows/system.ini"})
    assert security.is_allowed(action) is False


def test_write_blocked_in_read_only_mode(security):
    action = Action(tool="files", name="write", params={"path": "notes.txt"})
    decision = security.evaluate(action)
    assert decision.allowed is False
    assert "lecture seule" in decision.reason


def test_human_approval_flagging(project):
    # On enrichit la politique avec un outil sensible générique (pas 'system',
    # qui a désormais ses propres règles fines) et on vérifie le flag d'approbation.
    policy_path = project / "policy.yaml"
    data = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    data["allowed_tools"].append("email")
    data["require_human_approval"] = [{"tool": "email", "action": "send"}]
    policy_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    sec = EnforcementLayer(policy_path=policy_path, project_root=project)
    sensitive = Action(tool="email", name="send", params={})
    assert sec.is_allowed(sensitive) is True
    assert sec.requires_human_approval(sensitive) is True

    # Une autre action du même outil, non listée, ne requiert pas d'aval.
    other = Action(tool="email", name="draft", params={})
    assert sec.requires_human_approval(other) is False


def test_missing_policy_refuses_to_run(tmp_path):
    try:
        EnforcementLayer(policy_path=tmp_path / "nope.yaml")
    except SecurityError:
        pass
    else:
        raise AssertionError("doit refuser de tourner sans politique")
