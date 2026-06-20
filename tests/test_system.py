"""Tests phase 3 — outil système. La sécurité est le cœur du sujet ici."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from kira import sandbox
from kira.agent import Agent
from kira.engine import Engine, EngineResponse, MockBackend, ToolCall
from kira.security import Action, EnforcementLayer
from kira.tools.system import SystemTool

PY = sys.executable  # programme réel et sûr à allowlister pour les tests


def _policy(tmp_path: Path, allowed_commands: list[str]) -> Path:
    # On construit la politique via yaml.safe_dump pour éviter tout souci
    # d'échappement (chemins Windows avec backslashes notamment).
    policy = {
        "version": 1,
        "allowed_tools": ["system"],
        "tools": {
            "system": {
                "allowed_commands": allowed_commands,
                "program_timeout": 30,
                "code_execution": {
                    "enabled": True,
                    "image": "python:3.12-slim",
                    "timeout": 10,
                    "network": "none",
                    "memory": "256m",
                },
            }
        },
        "require_human_approval": [
            {"tool": "system", "action": "run_program"},
            {"tool": "system", "action": "execute_code"},
        ],
        "agent": {"max_steps": 4},
    }
    p = tmp_path / "policy.yaml"
    p.write_text(yaml.safe_dump(policy), encoding="utf-8")
    return p


@pytest.fixture
def security(tmp_path):
    return EnforcementLayer(policy_path=_policy(tmp_path, [PY]), project_root=tmp_path)


# --------------------------------------------------------------------------- #
# Couche de sécurité
# --------------------------------------------------------------------------- #
def test_run_program_not_in_allowlist_is_blocked(security):
    bad = Action(tool="system", name="run_program", params={"command": "rm"})
    assert security.is_allowed(bad) is False
    assert "hors allowlist" in security.evaluate(bad).reason


def test_run_program_in_allowlist_is_allowed_but_needs_approval(security):
    ok = Action(tool="system", name="run_program", params={"command": PY})
    assert security.is_allowed(ok) is True
    assert security.requires_human_approval(ok) is True


def test_execute_code_allowed_but_needs_approval(security):
    act = Action(tool="system", name="execute_code", params={"code": "print(1)"})
    assert security.is_allowed(act) is True
    assert security.requires_human_approval(act) is True


def test_unknown_system_action_blocked(security):
    assert security.is_allowed(Action(tool="system", name="nope")) is False


def test_execute_code_disabled_when_policy_disables(tmp_path):
    pol = tmp_path / "policy.yaml"
    pol.write_text(
        "version: 1\nallowed_tools: [system]\n"
        "tools:\n  system:\n    allowed_commands: []\n"
        "    code_execution:\n      enabled: false\n",
        encoding="utf-8",
    )
    sec = EnforcementLayer(policy_path=pol, project_root=tmp_path)
    act = Action(tool="system", name="execute_code", params={"code": "x"})
    assert sec.is_allowed(act) is False


# --------------------------------------------------------------------------- #
# Outil : run_program (allowlisté)
# --------------------------------------------------------------------------- #
def test_run_program_executes_allowlisted(security):
    tool = SystemTool(security)
    out = tool.run({"action": "run_program", "command": PY, "args": ["-c", "print(42)"]})
    assert "42" in out
    assert "exit=0" in out


def test_run_program_refuses_non_allowlisted(security):
    tool = SystemTool(security)
    out = tool.run({"action": "run_program", "command": "definitely_not_allowed"})
    assert out.lower().startswith("refusé")


# --------------------------------------------------------------------------- #
# Outil : execute_code — FAIL-CLOSED sans Docker
# --------------------------------------------------------------------------- #
def test_execute_code_fail_closed_without_docker(security, monkeypatch):
    # On simule l'absence de Docker : l'exécution DOIT être refusée.
    monkeypatch.setattr(sandbox, "docker_available", lambda: False)
    tool = SystemTool(security)
    out = tool.run({"action": "execute_code", "code": "print('hack')"})
    assert out.lower().startswith("refusé")
    assert "docker" in out.lower()


def test_sandbox_raises_when_docker_absent(monkeypatch):
    monkeypatch.setattr(sandbox, "docker_available", lambda: False)
    with pytest.raises(sandbox.SandboxUnavailable):
        sandbox.run_python_in_sandbox("print(1)")


# --------------------------------------------------------------------------- #
# Agent : la validation humaine est bien appliquée
# --------------------------------------------------------------------------- #
def _engine(tool_call: ToolCall):
    responses = [
        EngineResponse(text="je veux agir", tool_calls=[tool_call]),
        EngineResponse(text="fini", tool_calls=[]),
    ]
    return Engine(MockBackend(responses=responses))


def test_agent_blocks_run_program_when_approval_denied(security):
    call = ToolCall(id="c1", name="system", input={"command": PY, "action": "run_program", "args": ["-c", "print(1)"]})
    agent = Agent(_engine(call), security, approval_handler=lambda n, p: False)
    result = agent.run("lance un programme")
    obs = result.steps[0].actions[0]["observation"]
    assert "validation humaine" in obs


def test_agent_runs_program_when_approved(security):
    call = ToolCall(id="c1", name="system", input={"command": PY, "action": "run_program", "args": ["-c", "print(7)"]})
    agent = Agent(_engine(call), security, approval_handler=lambda n, p: True)
    result = agent.run("lance un programme")
    obs = result.steps[0].actions[0]["observation"]
    assert "7" in obs
