"""Outil système (phase 3) — lancer des programmes & exécuter du code.

⚠️ Outil le plus sensible du projet. Toute action passe par :
    1. la couche de sécurité (allowlist stricte + validation humaine), et
    2. une revalidation défensive ici (défense en profondeur).

Deux actions :
    - run_program  : lance un programme de l'ALLOWLIST (par nom), sans shell
      (pas d'injection), arguments passés en liste. Sur l'hôte, mais limité aux
      seuls programmes que l'humain a explicitement autorisés dans policy.yaml.
    - execute_code : exécute du code Python dans un sandbox Docker éphémère
      (kira/sandbox.py). FAIL-CLOSED : refusé si Docker indisponible. Jamais sur
      l'hôte.

Les deux actions sont marquées « validation humaine requise » dans policy.yaml,
donc l'agent doit obtenir l'aval via le handler d'approbation avant exécution.
"""

from __future__ import annotations

import subprocess
from typing import Any

from kira.sandbox import SandboxUnavailable, run_python_in_sandbox
from kira.security import Action, EnforcementLayer, SecurityError


class SystemTool:
    name = "system"
    description = (
        "Lance un programme autorisé ou exécute du code Python en sandbox. "
        "action='run_program' avec 'command' (+ 'args' liste optionnelle) pour "
        "lancer un programme de l'allowlist ; action='execute_code' avec 'code' "
        "pour exécuter du Python isolé dans Docker. Actions soumises à validation."
    )
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["run_program", "execute_code"],
                "description": "Opération à effectuer.",
            },
            "command": {
                "type": "string",
                "description": "Nom du programme à lancer (pour run_program).",
            },
            "args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Arguments du programme (pour run_program).",
            },
            "code": {
                "type": "string",
                "description": "Code Python à exécuter en sandbox (pour execute_code).",
            },
        },
        "required": ["action"],
    }

    def __init__(self, security: EnforcementLayer) -> None:
        self.security = security

    def to_action(self, params: dict[str, Any]) -> Action:
        name = str(params.get("action", ""))
        relevant = {
            k: v for k, v in params.items() if k in ("command", "args", "code")
        }
        return Action(tool=self.name, name=name, params=relevant)

    def run(self, params: dict[str, Any]) -> str:
        # Revalidation défensive : on ne fait jamais confiance aveuglément.
        try:
            self.security.check(self.to_action(params))
        except SecurityError as exc:
            return f"Refusé : {exc}"

        action = str(params.get("action", ""))
        if action == "run_program":
            return self._run_program(params)
        if action == "execute_code":
            return self._execute_code(params)
        return f"Erreur : action inconnue '{action}'."

    # ------------------------------------------------------------------ #
    def _run_program(self, params: dict[str, Any]) -> str:
        command = str(params.get("command", ""))
        args = params.get("args", []) or []
        if not isinstance(args, list):
            return "Erreur : 'args' doit être une liste."

        cfg = self.security.tool_config("system")
        allowed = cfg.get("allowed_commands", []) or []
        # Double garde : la sécurité a déjà validé, on revérifie l'allowlist.
        if command not in allowed:
            return f"Refusé : programme '{command}' hors allowlist."

        try:
            proc = subprocess.run(
                [command, *[str(a) for a in args]],
                capture_output=True,
                text=True,
                timeout=int(cfg.get("program_timeout", 30)),
                shell=False,  # JAMAIS de shell : pas d'injection de commande.
            )
        except FileNotFoundError:
            return f"Erreur : programme introuvable '{command}'."
        except subprocess.TimeoutExpired:
            return f"Erreur : '{command}' a dépassé le délai."
        except OSError as exc:
            return f"Erreur d'exécution : {exc}"

        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        parts = [f"exit={proc.returncode}"]
        if out:
            parts.append(f"stdout:\n{out}")
        if err:
            parts.append(f"stderr:\n{err}")
        return "\n".join(parts)

    # ------------------------------------------------------------------ #
    def _execute_code(self, params: dict[str, Any]) -> str:
        code = str(params.get("code", ""))
        if not code.strip():
            return "Erreur : aucun code fourni."

        cfg = self.security.tool_config("system").get("code_execution", {})
        try:
            result = run_python_in_sandbox(
                code,
                image=cfg.get("image", "python:3.12-slim"),
                timeout=int(cfg.get("timeout", 10)),
                memory=str(cfg.get("memory", "256m")),
                network=str(cfg.get("network", "none")),
            )
        except SandboxUnavailable as exc:
            return f"Refusé : {exc}"

        if result.timed_out:
            return "Erreur : le code a dépassé le délai dans le sandbox."
        out = (result.stdout or "").strip()
        err = (result.stderr or "").strip()
        parts = [f"exit={result.exit_code}"]
        if out:
            parts.append(f"stdout:\n{out}")
        if err:
            parts.append(f"stderr:\n{err}")
        return "\n".join(parts)
