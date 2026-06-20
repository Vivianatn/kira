"""Kira — assistant IA agentique (couche locale).

Ce paquet contient le cœur de l'agent :
- engine   : moteur de raisonnement (API LLM, backend interchangeable)
- security : couche d'enforcement (allowlist, validation humaine)
- agent    : boucle ReAct orchestrant moteur + outils
- memory   : mémoire court terme + RAG (phase 4, pas encore implémenté)
- tools    : outils exposés à l'agent (web, files pour l'instant)

La règle d'or : aucune action n'est exécutée sans passer par la couche
`security`. Voir PROJET_KIRA.md (section 8) et policy.yaml.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "Engine",
    "EngineResponse",
    "ToolCall",
    "Agent",
    "AgentResult",
    "EnforcementLayer",
    "Action",
    "SecurityError",
]

from kira.engine import Engine, EngineResponse, ToolCall
from kira.agent import Agent, AgentResult
from kira.security import EnforcementLayer, Action, SecurityError
