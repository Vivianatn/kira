"""Tests de la Phase 5 — skills (Kira crée ses outils). Exécution mockée
(sandbox réel non requis)."""

from __future__ import annotations

from kira.sandbox import SandboxResult
from kira.skills import SkillStore
from kira.tools.skill_tool import SkillsTool


def _echo_runner(code: str) -> SandboxResult:
    # Renvoie le code reçu pour vérifier l'injection des args.
    return SandboxResult(ok=True, stdout=code, stderr="", exit_code=0)


def test_propose_writes_pending_not_active(tmp_path):
    store = SkillStore(tmp_path)
    out = store.propose("greet", "dit bonjour", "print('hi')")
    assert "proposé" in out
    # Pas encore actif tant qu'un humain n'a pas validé.
    assert store.get("greet") is None
    assert (tmp_path / "skills" / "pending" / "greet.json").exists()


def test_approve_activates_skill(tmp_path):
    store = SkillStore(tmp_path)
    store.propose("greet", "dit bonjour", "print('hi')")
    assert "activé" in store.approve("greet")
    assert store.get("greet") is not None
    assert [s.name for s in store.list_active()] == ["greet"]


def test_run_injects_args_and_runs_in_sandbox(tmp_path):
    store = SkillStore(tmp_path)
    store.propose("double", "double x", "print(args['x'] * 2)")
    store.approve("double")
    out = store.run("double", {"x": 5}, runner=_echo_runner)
    # Les args sont injectés avant le code, le tout passé au sandbox.
    assert '"x": 5' in out
    assert "args['x'] * 2" in out


def test_invalid_name_rejected(tmp_path):
    store = SkillStore(tmp_path)
    assert "Erreur" in store.propose("Mauvais Nom!", "x", "print(1)")


def test_run_unknown_skill(tmp_path):
    store = SkillStore(tmp_path)
    assert "introuvable" in store.run("inexistant", {}, runner=_echo_runner)


def test_cannot_run_pending_skill(tmp_path):
    # Un skill proposé mais NON activé ne doit pas être exécutable.
    store = SkillStore(tmp_path)
    store.propose("x", "y", "print(1)")
    assert "introuvable" in store.run("x", {}, runner=_echo_runner)


def test_skills_tool_create_and_list(tmp_path):
    store = SkillStore(tmp_path)
    tool = SkillsTool(store)
    assert tool.run({"action": "list"}) == "(aucun skill actif)"
    out = tool.run(
        {"action": "create", "name": "sum2", "description": "somme", "code": "print(args['a']+args['b'])"}
    )
    assert "proposé" in out
    # Création => pending, donc list (actifs) reste vide.
    assert tool.run({"action": "list"}) == "(aucun skill actif)"
