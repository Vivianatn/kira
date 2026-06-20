"""Outil fichiers — LECTURE SEULE, confinée au répertoire autorisé.

Toute opération passe par la couche de sécurité (`EnforcementLayer`) :
l'outil n'a aucun pouvoir au-delà de ce que `policy.yaml` autorise. Le
confinement des chemins (anti `../`, anti chemin absolu hors racine) est
vérifié À LA FOIS par la sécurité et ici, par prudence (défense en profondeur).

Opérations exposées :
    - read : lit le contenu d'un fichier texte.
    - list : liste le contenu d'un répertoire.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kira.security import Action, EnforcementLayer, SecurityError


class FilesTool:
    name = "files"
    description = (
        "Lit des fichiers texte (lecture seule) dans le répertoire de travail "
        "autorisé. Utilise action='read' avec un 'path' pour lire un fichier, "
        "ou action='list' pour lister un dossier."
    )
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["read", "list"],
                "description": "Opération à effectuer.",
            },
            "path": {
                "type": "string",
                "description": (
                    "Chemin relatif au répertoire de travail (ex. 'notes.txt'). "
                    "Pour 'list', '.' liste la racine."
                ),
            },
        },
        "required": ["action", "path"],
    }

    def __init__(self, security: EnforcementLayer) -> None:
        self.security = security

    # --- conversion vers le modèle de sécurité ---------------------------- #
    def to_action(self, params: dict[str, Any]) -> Action:
        return Action(
            tool=self.name,
            name=str(params.get("action", "")),
            params={"path": params.get("path", "")},
        )

    # --- exécution -------------------------------------------------------- #
    def run(self, params: dict[str, Any]) -> str:
        action = str(params.get("action", "")).lower()
        # La sécurité a déjà validé en amont (agent), mais on revérifie :
        # défense en profondeur, l'outil ne fait jamais confiance aveuglément.
        # On renvoie une observation de refus plutôt que de lever : un outil
        # doit toujours produire une chaîne (le contrat d'« Observation »).
        try:
            self.security.check(self.to_action(params))
        except SecurityError as exc:
            return f"Refusé : {exc}"

        if action == "read":
            return self._read(params.get("path", ""))
        if action == "list":
            return self._list(params.get("path", "."))
        return f"Erreur : action inconnue '{action}' (attendu: read|list)."

    # --- helpers ---------------------------------------------------------- #
    def _resolve(self, rel_path: str) -> Path:
        root = self.security.files_root()
        candidate = (root / rel_path).resolve()
        # Garde-fou final : on refuse tout ce qui sort de la racine.
        if candidate != root and root not in candidate.parents:
            raise SecurityError(f"chemin '{rel_path}' hors du répertoire autorisé")
        return candidate

    def _read(self, rel_path: str) -> str:
        try:
            path = self._resolve(rel_path)
        except SecurityError as exc:
            return f"Refusé : {exc}"
        if not path.exists():
            return f"Erreur : fichier introuvable '{rel_path}'."
        if not path.is_file():
            return f"Erreur : '{rel_path}' n'est pas un fichier."

        max_bytes = int(self.security.tool_config("files").get("max_bytes", 100_000))
        data = path.read_bytes()
        truncated = len(data) > max_bytes
        text = data[:max_bytes].decode("utf-8", errors="replace")
        if truncated:
            text += f"\n... [tronqué à {max_bytes} octets]"
        return text

    def _list(self, rel_path: str) -> str:
        try:
            path = self._resolve(rel_path)
        except SecurityError as exc:
            return f"Refusé : {exc}"
        if not path.exists():
            return f"Erreur : dossier introuvable '{rel_path}'."
        if not path.is_dir():
            return f"Erreur : '{rel_path}' n'est pas un dossier."

        entries = sorted(
            f"{p.name}/" if p.is_dir() else p.name for p in path.iterdir()
        )
        if not entries:
            return "(dossier vide)"
        return "\n".join(entries)
