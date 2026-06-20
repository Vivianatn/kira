"""Outil 'skills' (phase 5) — laisse Kira proposer et utiliser ses propres outils.

Actions :
    - create : proposer un nouvel outil (-> en attente de validation humaine).
    - list   : lister les outils actifs (validés).
    - run    : exécuter un outil actif, dans le sandbox Docker.

Kira ne peut PAS s'auto-activer un outil : l'activation est une opération humaine
(SkillStore.approve, hors de la boucle de l'agent).
"""

from __future__ import annotations

from typing import Any

from kira.security import Action, EnforcementLayer, SecurityError
from kira.skills import SkillStore


class SkillsTool:
    name = "skills"
    description = (
        "Crée et utilise des outils sur-mesure (skills). action='create' avec "
        "'name', 'description', 'code' (Python lisant ses paramètres dans `args` "
        "et imprimant le résultat) pour PROPOSER un outil (validation humaine "
        "ensuite) ; action='list' pour voir les outils actifs ; action='run' avec "
        "'name' (+ 'args' objet) pour exécuter un outil actif en sandbox."
    )
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["create", "list", "run"]},
            "name": {"type": "string", "description": "Nom du skill."},
            "description": {"type": "string", "description": "À quoi sert le skill."},
            "code": {"type": "string", "description": "Code Python du skill."},
            "args": {"type": "object", "description": "Paramètres passés au skill (run)."},
        },
        "required": ["action"],
    }

    def __init__(self, store: SkillStore, security: EnforcementLayer | None = None) -> None:
        self.store = store
        self.security = security

    def to_action(self, params: dict[str, Any]) -> Action:
        return Action(
            tool=self.name,
            name=str(params.get("action", "")),
            params={k: v for k, v in params.items() if k != "action"},
        )

    def run(self, params: dict[str, Any]) -> str:
        if self.security is not None:
            try:
                self.security.check(self.to_action(params))
            except SecurityError as exc:
                return f"Refusé : {exc}"

        action = str(params.get("action", "")).lower()
        if action == "create":
            return self.store.propose(
                str(params.get("name", "")),
                str(params.get("description", "")),
                str(params.get("code", "")),
            )
        if action == "list":
            skills = self.store.list_active()
            if not skills:
                return "(aucun skill actif)"
            return "\n".join(f"- {s.name} : {s.description}" for s in skills)
        if action == "run":
            return self.store.run(str(params.get("name", "")), params.get("args") or {})
        return f"Erreur : action inconnue '{action}'."
