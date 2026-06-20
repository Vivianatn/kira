"""Outils de Kira.

Un outil expose :
    - name        : identifiant (doit correspondre à l'allowlist de policy.yaml).
    - description : à quoi il sert (lu par le LLM).
    - schema      : schéma JSON des paramètres d'entrée (format tool-use Anthropic).
    - run(params) : exécute l'outil et renvoie une chaîne (l'« Observation »).
    - to_action(params) : convertit un appel en `Action` pour la couche sécurité.

Phase actuelle : seuls deux outils SÛRS sont fournis : `web` (recherche) et
`files` (lecture seule). Aucun lancement de programme / exécution de code.
"""

from __future__ import annotations

from typing import Any, Protocol

from kira.security import Action, EnforcementLayer


class Tool(Protocol):
    name: str
    description: str
    schema: dict[str, Any]

    def run(self, params: dict[str, Any]) -> str: ...

    def to_action(self, params: dict[str, Any]) -> Action: ...


def build_registry(security: EnforcementLayer) -> dict[str, Tool]:
    """Construit le dict {nom: outil} pour les seuls outils de l'allowlist.

    On n'instancie QUE les outils autorisés par la politique : moindre
    privilège appliqué dès la construction.
    """
    from kira.tools.files import FilesTool
    from kira.tools.system import SystemTool
    from kira.tools.web import WebTool

    available = {
        "files": lambda: FilesTool(security),
        "web": lambda: WebTool(security),
        "system": lambda: SystemTool(security),
    }

    registry: dict[str, Tool] = {}
    for name in security.allowed_tools:
        factory = available.get(name)
        if factory is not None:
            registry[name] = factory()
    return registry


def tool_schemas(registry: dict[str, Tool]) -> list[dict[str, Any]]:
    """Schémas des outils au format attendu par le moteur (tool-use Anthropic)."""
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.schema,
        }
        for tool in registry.values()
    ]


__all__ = ["Tool", "build_registry", "tool_schemas"]
