"""Outil mémoire — laisse Kira décider quoi mémoriser et retrouver.

C'est la dimension *agentique* de la mémoire : en plus de la mémoire passive
(l'agent stocke/rappelle automatiquement les échanges et les leçons), Kira peut
**explicitement** retenir un fait important ou aller chercher un souvenir.

Outil SÛR : aucune exécution, juste de la lecture/écriture dans le magasin de
souvenirs local. Deux actions :
    - remember : mémoriser un texte (avec un tag optionnel).
    - recall   : retrouver les souvenirs les plus pertinents pour une requête.
"""

from __future__ import annotations

from typing import Any

from kira.memory import Memory
from kira.security import Action, EnforcementLayer, SecurityError


class MemoryTool:
    name = "memory"
    description = (
        "Mémoire de Kira. action='remember' avec 'text' pour retenir un fait "
        "important (préférence utilisateur, leçon apprise, info utile) ; "
        "action='recall' avec 'query' pour retrouver des souvenirs pertinents."
    )
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["remember", "recall"],
                "description": "Opération mémoire.",
            },
            "text": {
                "type": "string",
                "description": "Le fait à mémoriser (pour remember).",
            },
            "query": {
                "type": "string",
                "description": "Ce qu'on cherche à retrouver (pour recall).",
            },
            "tag": {
                "type": "string",
                "description": "Étiquette optionnelle (ex. 'préférence', 'leçon').",
            },
        },
        "required": ["action"],
    }

    def __init__(self, memory: Memory, security: EnforcementLayer | None = None) -> None:
        self.memory = memory
        self.security = security

    def to_action(self, params: dict[str, Any]) -> Action:
        return Action(
            tool=self.name,
            name=str(params.get("action", "")),
            params={k: v for k, v in params.items() if k != "action"},
        )

    def run(self, params: dict[str, Any]) -> str:
        # Défense en profondeur (si une sécurité est fournie).
        if self.security is not None:
            try:
                self.security.check(self.to_action(params))
            except SecurityError as exc:
                return f"Refusé : {exc}"

        action = str(params.get("action", "")).lower()
        if action == "remember":
            text = str(params.get("text", "")).strip()
            if not text:
                return "Erreur : rien à mémoriser (text vide)."
            tag = params.get("tag")
            meta = {"kind": "note"}
            if tag:
                meta["tag"] = str(tag)
            self.memory.remember(text, **meta)
            return "Mémorisé."
        if action == "recall":
            query = str(params.get("query", "")).strip()
            if not query:
                return "Erreur : requête vide."
            hits = self.memory.recall(query, k=5)
            if not hits:
                return "(aucun souvenir pertinent)"
            return "\n".join(f"- {h.text}" for h in hits)
        return f"Erreur : action inconnue '{action}'."
