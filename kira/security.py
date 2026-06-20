"""Couche d'enforcement — le garde-fou de Kira.

C'est la pièce la plus importante du projet : AUCUNE action ne doit être
exécutée sans être passée par ici. La couche charge `policy.yaml` et répond
à deux questions :

    - is_allowed(action)              -> l'action est-elle permise par la politique ?
    - requires_human_approval(action) -> faut-il une validation humaine ?

Principes (cf. PROJET_KIRA.md §8bis) :
    - ALLOWLIST uniquement (jamais de denylist contournable).
    - Moindre privilège : par défaut, tout est refusé.
    - Validation humaine (human-in-the-loop) pour les actions sensibles.
    - Journal append-only de toutes les décisions.

Cette couche est volontairement SANS dépendance sur les outils eux-mêmes :
elle raisonne sur des descriptions d'actions, pas sur du code exécutable.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dépendance déclarée dans requirements
    raise ImportError(
        "PyYAML est requis pour charger policy.yaml. "
        "Installe-le : pip install pyyaml"
    ) from exc


class SecurityError(Exception):
    """Levée quand une action est refusée par la politique de sécurité."""


@dataclass(frozen=True)
class Action:
    """Description normalisée d'une action que l'agent veut exécuter.

    tool   : nom de l'outil (ex. "files", "web").
    name   : sous-action / opération (ex. "read", "search"). Optionnel.
    params : paramètres de l'action (ex. {"path": "notes.txt"}).
    """

    tool: str
    name: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        op = f".{self.name}" if self.name else ""
        return f"{self.tool}{op}({self.params})"


# Résultat structuré d'une vérification : pratique pour journaliser / tester.
@dataclass(frozen=True)
class Decision:
    allowed: bool
    needs_approval: bool
    reason: str


class EnforcementLayer:
    """Charge la politique et arbitre chaque action de l'agent."""

    def __init__(
        self,
        policy_path: str | os.PathLike[str] = "policy.yaml",
        *,
        project_root: str | os.PathLike[str] | None = None,
        audit_log: str | os.PathLike[str] | None = None,
    ) -> None:
        self.policy_path = Path(policy_path).resolve()
        # Racine du projet : sert à résoudre les chemins relatifs de la politique
        # (ex. files.root = "workspace"). Par défaut, le dossier du policy.yaml.
        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else self.policy_path.parent
        )
        self.policy = self._load_policy()

        # Journal append-only des décisions (traçabilité). None = désactivé.
        self._audit_log = Path(audit_log).resolve() if audit_log else None

    # ------------------------------------------------------------------ #
    # Chargement de la politique
    # ------------------------------------------------------------------ #
    def _load_policy(self) -> dict[str, Any]:
        if not self.policy_path.exists():
            raise SecurityError(
                f"Fichier de politique introuvable : {self.policy_path}. "
                "Kira refuse de tourner sans politique de sécurité."
            )
        with self.policy_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise SecurityError("policy.yaml doit décrire un mapping (clé: valeur).")
        return data

    def reload(self) -> None:
        """Recharge la politique depuis le disque (utile en dev)."""
        self.policy = self._load_policy()

    # ------------------------------------------------------------------ #
    # Accès pratiques à la configuration
    # ------------------------------------------------------------------ #
    @property
    def allowed_tools(self) -> list[str]:
        return list(self.policy.get("allowed_tools", []))

    @property
    def max_steps(self) -> int:
        return int(self.policy.get("agent", {}).get("max_steps", 6))

    def tool_config(self, tool: str) -> dict[str, Any]:
        return dict(self.policy.get("tools", {}).get(tool, {}))

    def files_root(self) -> Path:
        """Répertoire de travail principal (workspace) — chemin absolu."""
        root = self.tool_config("files").get("root", "workspace")
        root_path = Path(root)
        if not root_path.is_absolute():
            root_path = self.project_root / root_path
        return root_path.resolve()

    def files_roots(self) -> list[Path]:
        """Tous les répertoires autorisés : workspace + writable_paths de la politique.

        `writable_paths` est l'allowlist des « autres fichiers » que l'humain a
        explicitement autorisés. Vide par défaut.
        """
        roots = [self.files_root()]
        for p in self.tool_config("files").get("writable_paths", []) or []:
            pp = Path(p)
            roots.append((pp if pp.is_absolute() else self.project_root / pp).resolve())
        return roots

    # ------------------------------------------------------------------ #
    # Le cœur : décision de sécurité
    # ------------------------------------------------------------------ #
    def is_allowed(self, action: Action) -> bool:
        return self.evaluate(action).allowed

    def requires_human_approval(self, action: Action) -> bool:
        return self.evaluate(action).needs_approval

    def evaluate(self, action: Action) -> Decision:
        """Renvoie une décision détaillée (allowed / needs_approval / reason)."""
        # 1) L'outil est-il dans l'allowlist ?
        if action.tool not in self.allowed_tools:
            return self._record(
                action,
                Decision(False, False, f"outil '{action.tool}' hors allowlist"),
            )

        # 2) Validations spécifiques par outil (ex. confinement des chemins).
        ok, reason = self._tool_specific_check(action)
        if not ok:
            return self._record(action, Decision(False, False, reason))

        # 3) Action sensible nécessitant une validation humaine ?
        needs_approval = self._matches_sensitive(action)
        reason = "ok" if not needs_approval else "validation humaine requise"
        return self._record(action, Decision(True, needs_approval, reason))

    def check(self, action: Action) -> Decision:
        """Comme evaluate() mais LÈVE SecurityError si l'action est refusée.

        À appeler par l'agent juste avant d'exécuter un outil.
        """
        decision = self.evaluate(action)
        if not decision.allowed:
            raise SecurityError(f"Action refusée : {action} — {decision.reason}")
        return decision

    # ------------------------------------------------------------------ #
    # Helpers internes
    # ------------------------------------------------------------------ #
    def _tool_specific_check(self, action: Action) -> tuple[bool, str]:
        """Règles fines propres à un outil. Renvoie (autorisé, raison)."""
        if action.tool == "files":
            return self._check_files(action)
        if action.tool == "system":
            return self._check_system(action)
        # Pas de règle fine pour les autres outils pour l'instant.
        return True, "ok"

    def _check_system(self, action: Action) -> tuple[bool, str]:
        """Phase 3 — outil système. Sécurité d'abord, fail-closed partout.

        - run_program : seul un programme de l'ALLOWLIST (par nom exact) peut être
          lancé. Allowlist vide par défaut => rien n'est lançable.
        - execute_code : autorisé par la politique mais l'exécution réelle exige
          un sandbox Docker (vérifié au runtime par l'outil). Jamais sur l'hôte.
        """
        cfg = self.tool_config("system")

        if action.name == "run_program":
            command = action.params.get("command")
            allowed = cfg.get("allowed_commands", []) or []
            if not command:
                return False, "aucune commande fournie"
            if command not in allowed:
                return False, f"programme '{command}' hors allowlist system.allowed_commands"
            return True, "ok"

        if action.name == "execute_code":
            if not cfg.get("code_execution", {}).get("enabled", False):
                return False, "exécution de code désactivée (system.code_execution.enabled)"
            return True, "ok"

        return False, f"action system inconnue : '{action.name}'"

    def _check_files(self, action: Action) -> tuple[bool, str]:
        cfg = self.tool_config("files")

        # Écriture interdite tant que mode == read_only.
        write_ops = {"write", "append", "delete", "mkdir", "move"}
        if cfg.get("mode", "read_only") == "read_only" and action.name in write_ops:
            return False, f"outil files en lecture seule, '{action.name}' interdit"

        # Confinement : le chemin doit être dans le workspace OU dans un des
        # writable_paths explicitement autorisés. Tout le reste est refusé.
        path_param = action.params.get("path")
        if path_param is not None:
            roots = self.files_roots()
            if not any(self._is_within_root(path_param, r) for r in roots):
                return False, f"chemin '{path_param}' hors des répertoires autorisés"
        return True, "ok"

    @staticmethod
    def _is_within_root(path: str | os.PathLike[str], root: Path) -> bool:
        """True si `path` (résolu) est strictement contenu dans `root`.

        Bloque les évasions par '..' et les chemins absolus hors racine.
        """
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.resolve()
        root = root.resolve()
        return candidate == root or root in candidate.parents

    def _matches_sensitive(self, action: Action) -> bool:
        for rule in self.policy.get("require_human_approval", []) or []:
            if not isinstance(rule, dict):
                continue
            if rule.get("tool") != action.tool:
                continue
            # Si aucune action précisée dans la règle, tout l'outil est sensible.
            rule_action = rule.get("action")
            if rule_action is None or rule_action == action.name:
                return True
        return False

    # ------------------------------------------------------------------ #
    # Journal append-only
    # ------------------------------------------------------------------ #
    def _record(self, action: Action, decision: Decision) -> Decision:
        if self._audit_log is None:
            return decision
        entry = {
            "ts": time.time(),
            "tool": action.tool,
            "name": action.name,
            "params": action.params,
            "allowed": decision.allowed,
            "needs_approval": decision.needs_approval,
            "reason": decision.reason,
        }
        self._audit_log.parent.mkdir(parents=True, exist_ok=True)
        with self._audit_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return decision
