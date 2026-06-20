"""Outil 'self_improve' — Kira modifie son propre code, sous garde-fous.

Branche `kira/improve.py` dans la boucle de l'agent. Deux actions :
    - view    : lire un fichier éditable (kira/ ou tests/) pour s'en inspirer.
    - propose : soumettre une nouvelle version d'un fichier.

Le verdict est rendu par `SelfImprover`, pas par cet outil :
    - mineur (peu de lignes, fichier non protégé) + suite pytest verte
      -> appliqué automatiquement ;
    - important / fichier de sécurité protégé / tests rouges
      -> NON appliqué, proposition écrite pour validation humaine.

⚠️ Cet outil NE figure PAS dans `require_human_approval` : c'est volontaire, car
la décision « mineur => automatique » est prise *à l'intérieur* de SelfImprover,
qui garantit déjà la sécurité (tests = fitness, fichiers protégés, rollback).
"""

from __future__ import annotations

from typing import Any

from kira.improve import SelfImprover
from kira.security import Action, EnforcementLayer, SecurityError


class SelfImproveTool:
    name = "self_improve"
    description = (
        "Permet à Kira d'améliorer son propre code (fichiers kira/ ou tests/). "
        "action='view' avec 'path' pour lire un fichier ; action='propose' avec "
        "'path' et 'new_content' (le fichier ENTIER réécrit) pour soumettre une "
        "amélioration. Les petits changements validés par les tests sont appliqués "
        "automatiquement ; les gros ou les fichiers sensibles demandent une "
        "validation humaine. Une modification qui casse les tests est annulée."
    )
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["view", "propose"]},
            "path": {
                "type": "string",
                "description": "Chemin du fichier (ex. 'kira/tools/web.py').",
            },
            "new_content": {
                "type": "string",
                "description": "Contenu COMPLET du fichier réécrit (pour propose).",
            },
        },
        "required": ["action", "path"],
    }

    def __init__(self, improver: SelfImprover, security: EnforcementLayer | None = None) -> None:
        self.improver = improver
        self.security = security

    def to_action(self, params: dict[str, Any]) -> Action:
        return Action(
            tool=self.name,
            name=str(params.get("action", "")),
            params={"path": params.get("path", "")},
        )

    def run(self, params: dict[str, Any]) -> str:
        if self.security is not None:
            try:
                self.security.check(self.to_action(params))
            except SecurityError as exc:
                return f"Refusé : {exc}"

        action = str(params.get("action", "")).lower()
        path = str(params.get("path", ""))

        if action == "view":
            content = self.improver.view(path)
            if content is None:
                return f"Refusé : '{path}' n'est pas un fichier éditable (kira/ ou tests/, .py)."
            return content

        if action == "propose":
            new_content = params.get("new_content")
            if not isinstance(new_content, str) or not new_content.strip():
                return "Erreur : 'new_content' (fichier complet réécrit) manquant."
            res = self.improver.propose(path, new_content)
            verdict = (
                "APPLIQUÉ automatiquement"
                if res.applied
                else ("VALIDATION HUMAINE requise" if res.needs_approval else "REFUSÉ")
            )
            tests = (
                "non lancés"
                if res.tests_passed is None
                else ("verts" if res.tests_passed else "ROUGES")
            )
            out = (
                f"[{verdict}] {path} — {res.reason}. "
                f"Lignes modifiées: {res.diff_lines} ; tests: {tests}."
            )
            if res.proposal_path:
                out += f" Proposition: {res.proposal_path}"
            return out

        return f"Erreur : action inconnue '{action}'."
