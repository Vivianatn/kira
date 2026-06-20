"""Tests de la boucle ReAct — arrêt à la limite de pas, sécurité, exécution."""

from __future__ import annotations

from kira.agent import Agent
from kira.engine import Engine, EngineResponse, MockBackend, ToolCall


def _engine(responses):
    return Engine(MockBackend(responses=responses), system="test")


def test_stops_at_max_steps(security):
    # Backend qui demande TOUJOURS un appel d'outil -> jamais de réponse finale.
    loop_response = EngineResponse(
        text="je réfléchis encore",
        tool_calls=[ToolCall(id="t", name="files", input={"action": "list", "path": "."})],
    )
    engine = Engine(MockBackend(responses=[loop_response]))  # répété indéfiniment
    agent = Agent(engine, security, max_steps=3)

    result = agent.run("boucle pour toujours")
    assert result.hit_step_limit is True
    assert result.stopped_reason == "max_steps"
    assert len(result.steps) == 3  # exactement max_steps tours


def test_returns_final_answer_without_tools(security):
    engine = _engine([EngineResponse(text="42", tool_calls=[])])
    agent = Agent(engine, security)
    result = agent.run("quelle est la réponse ?")
    assert result.stopped_reason == "final"
    assert result.answer == "42"
    assert len(result.steps) == 1


def test_tool_call_then_final_answer(security):
    # 1er tour : lire un fichier. 2e tour : répondre.
    responses = [
        EngineResponse(
            text="je vais lire le fichier",
            tool_calls=[
                ToolCall(id="c1", name="files", input={"action": "read", "path": "notes.txt"})
            ],
        ),
        EngineResponse(text="le fichier dit bonjour", tool_calls=[]),
    ]
    agent = Agent(_engine(responses), security)
    result = agent.run("lis notes.txt")

    assert result.stopped_reason == "final"
    # L'observation du 1er pas doit contenir le contenu réel du fichier.
    obs = result.steps[0].actions[0]["observation"]
    assert "bonjour kira" in obs


def test_disallowed_tool_is_refused_not_executed(security):
    # Le modèle tente d'appeler un outil hors allowlist : l'agent renvoie
    # une observation "refusé" et ne plante pas.
    responses = [
        EngineResponse(
            text="je tente un truc interdit",
            tool_calls=[ToolCall(id="x", name="system", input={"cmd": "calc"})],
        ),
        EngineResponse(text="ok j'abandonne", tool_calls=[]),
    ]
    agent = Agent(_engine(responses), security)
    result = agent.run("lance calc")
    obs = result.steps[0].actions[0]["observation"]
    assert obs.lower().startswith("refusé")


def test_sensitive_action_requires_approval(project):
    import yaml

    from kira.security import EnforcementLayer

    # On rend l'outil 'files.read' sensible pour le test.
    policy_path = project / "policy.yaml"
    data = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    data["require_human_approval"] = [{"tool": "files", "action": "read"}]
    policy_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    sec = EnforcementLayer(policy_path=policy_path, project_root=project)

    responses = [
        EngineResponse(
            text="lecture sensible",
            tool_calls=[
                ToolCall(id="c1", name="files", input={"action": "read", "path": "notes.txt"})
            ],
        ),
        EngineResponse(text="fini", tool_calls=[]),
    ]

    # Handler qui REFUSE l'approbation -> l'outil ne doit pas s'exécuter.
    agent = Agent(
        _engine(responses), sec, approval_handler=lambda name, params: False
    )
    result = agent.run("lis notes.txt")
    obs = result.steps[0].actions[0]["observation"]
    assert "validation humaine" in obs

    # Handler qui ACCEPTE -> l'outil s'exécute et lit le fichier.
    agent_ok = Agent(
        _engine(
            [
                EngineResponse(
                    text="lecture sensible",
                    tool_calls=[
                        ToolCall(
                            id="c1",
                            name="files",
                            input={"action": "read", "path": "notes.txt"},
                        )
                    ],
                ),
                EngineResponse(text="fini", tool_calls=[]),
            ]
        ),
        sec,
        approval_handler=lambda name, params: True,
    )
    result_ok = agent_ok.run("lis notes.txt")
    assert "bonjour kira" in result_ok.steps[0].actions[0]["observation"]
