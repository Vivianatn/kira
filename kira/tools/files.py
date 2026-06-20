"""Outil fichiers — lecture ET écriture, confinées aux répertoires autorisés.

Toute opération passe par la couche de sécurité (`EnforcementLayer`) :
l'outil n'a aucun pouvoir au-delà de ce que `policy.yaml` autorise.

Confinement (défense en profondeur, vérifié ici ET côté sécurité) : un chemin
doit être DANS le workspace, ou DANS un des `writable_paths` explicitement
autorisés par la politique. Tout le reste est refusé (anti `../`, anti chemin
absolu hors zone).

Opérations :
    - read   : lit un fichier texte.
    - list   : liste un dossier.
    - write  : écrit/écrase un fichier (crée les dossiers parents).
    - append : ajoute à la fin d'un fichier.
    - mkdir  : crée un dossier.
    - delete : supprime un fichier (action SENSIBLE -> validation humaine).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kira.security import Action, EnforcementLayer, SecurityError


class FilesTool:
    name = "files"
    description = (
        "Lit et écrit des fichiers dans le workspace (et les dossiers autorisés). "
        "action='read'/'list' pour lire ; action='write' (écrase) ou 'append' "
        "(ajoute) avec 'path' et 'content' pour écrire ; 'mkdir' pour créer un "
        "dossier ; 'delete' pour supprimer (demande une validation humaine)."
    )
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["read", "list", "write", "append", "mkdir", "delete"],
                "description": "Opération à effectuer.",
            },
            "path": {
                "type": "string",
                "description": (
                    "Chemin relatif au workspace (ex. 'notes.txt') ou absolu si "
                    "dans un dossier autorisé. '.' = racine du workspace."
                ),
            },
            "content": {
                "type": "string",
                "description": "Contenu à écrire (pour write/append).",
            },
        },
        "required": ["action", "path"],
    }

    def __init__(self, security: EnforcementLayer) -> None:
        self.security = security

    def to_action(self, params: dict[str, Any]) -> Action:
        return Action(
            tool=self.name,
            name=str(params.get("action", "")),
            params={"path": params.get("path", "")},
        )

    def run(self, params: dict[str, Any]) -> str:
        action = str(params.get("action", "")).lower()
        # Défense en profondeur : on revérifie auprès de la sécurité.
        try:
            self.security.check(self.to_action(params))
        except SecurityError as exc:
            return f"Refusé : {exc}"

        path = params.get("path", "")
        if action == "read":
            return self._read(path)
        if action == "list":
            return self._list(path or ".")
        if action == "write":
            return self._write(path, str(params.get("content", "")), overwrite=True)
        if action == "append":
            return self._write(path, str(params.get("content", "")), overwrite=False)
        if action == "mkdir":
            return self._mkdir(path)
        if action == "delete":
            return self._delete(path)
        return f"Erreur : action inconnue '{action}'."

    # --- confinement multi-racines ---------------------------------------- #
    def _resolve(self, rel_path: str) -> Path:
        root = self.security.files_root()
        p = Path(rel_path)
        candidate = (p if p.is_absolute() else root / p).resolve()
        roots = self.security.files_roots()
        if not any(candidate == r or r in candidate.parents for r in roots):
            raise SecurityError(f"chemin '{rel_path}' hors des répertoires autorisés")
        return candidate

    # --- lecture ---------------------------------------------------------- #
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
        return "\n".join(entries) if entries else "(dossier vide)"

    # --- écriture --------------------------------------------------------- #
    def _write(self, rel_path: str, content: str, *, overwrite: bool) -> str:
        try:
            path = self._resolve(rel_path)
        except SecurityError as exc:
            return f"Refusé : {exc}"
        if path.is_dir():
            return f"Erreur : '{rel_path}' est un dossier."
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            mode = "w" if overwrite else "a"
            with path.open(mode, encoding="utf-8") as fh:
                fh.write(content)
        except OSError as exc:
            return f"Erreur d'écriture : {exc}"
        verb = "écrit" if overwrite else "complété"
        return f"OK : {verb} {len(content)} caractères dans '{rel_path}'."

    def _mkdir(self, rel_path: str) -> str:
        try:
            path = self._resolve(rel_path)
        except SecurityError as exc:
            return f"Refusé : {exc}"
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return f"Erreur : {exc}"
        return f"OK : dossier '{rel_path}' créé."

    def _delete(self, rel_path: str) -> str:
        try:
            path = self._resolve(rel_path)
        except SecurityError as exc:
            return f"Refusé : {exc}"
        if not path.exists():
            return f"Erreur : '{rel_path}' n'existe pas."
        if path.is_dir():
            return f"Erreur : '{rel_path}' est un dossier (suppression de dossier non permise)."
        try:
            path.unlink()
        except OSError as exc:
            return f"Erreur : {exc}"
        return f"OK : '{rel_path}' supprimé."
