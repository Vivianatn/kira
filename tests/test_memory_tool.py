"""Tests de l'outil mémoire et de l'apprentissage des erreurs."""

from __future__ import annotations

from kira.agent import Agent
from kira.engine import Engine, EngineResponse, MockBackend, ToolCall
from kira.memory import HashEmbedder, Memory
from kira.tools.memory_tool import MemoryTool


def test_memory_tool_remember_and_recall():
    mem = Memory(embedder=HashEmbedder())
    tool = MemoryTool(mem)

    assert tool.run({"action": "remember", "text": "Vivian aime le café noir"}) == "Mémorisé."
    out = tool.run({"action": "recall", "query": "qu'aime Vivian comme café ?"})
    assert "café noir" in out


def test_memory_tool_recall_empty():
    tool = MemoryTool(Memory(embedder=HashEmbedder()))
    assert "aucun souvenir" in tool.run({"action": "recall", "query": "x"})


def test_memory_tool_remember_empty_text():
    tool = MemoryTool(Memory(embedder=HashEmbedder()))
    assert "Erreur" in tool.run({"action": "remember", "text": "   "})


def test_memory_tool_only_registered_when_allowed_and_memory_present(tmp_path):
    import yaml

    from kira.security import EnforcementLayer
    from kira.tools import build_registry

    pol = tmp_path / "policy.yaml"
    pol.write_text(
        yaml.safe_dump({"version": 1, "allowed_tools": ["memory"], "tools": {}}),
        encoding="utf-8",
    )
    sec = EnforcementLayer(policy_path=pol, project_root=tmp_path)

    # 'memory' autorisé + mémoire fournie -> outil présent.
    assert "memory" in build_registry(sec, Memory(embedder=HashEmbedder()))
    # Pas de mémoire fournie -> outil absent (même si autorisé).
    assert "memory" not in build_registry(sec, None)


def test_agent_learns_from_failed_action(security):
    # Le modèle tente un outil hors allowlist -> échec -> leçon mémorisée.
    mem = Memory(embedder=HashEmbedder())
    responses = [
        EngineResponse(
            text="je tente",
            tool_calls=[ToolCall(id="x", name="system", input={"cmd": "rm -rf /"})],
        ),
        EngineResponse(text="ok j'arrête", tool_calls=[]),
    ]
    agent = Agent(Engine(MockBackend(responses=responses)), security, memory=mem)
    agent.run("fais un truc interdit")

    # Une leçon a été enregistrée et est retrouvable.
    hits = mem.recall("system", k=5)
    assert any("Leçon" in h.text for h in hits)
