"""Phase 5 — Kira crée ses propres outils (« skills »).

Un *skill* est un petit programme Python réutilisable que Kira peut **proposer**.
Cycle de vie, sous garde-fous :

    proposé (pending) --[validation humaine]--> actif --[appel]--> exécuté en SANDBOX

- **Proposer** : écrit le skill dans `skills/pending/` (aucune exécution). Sûr.
- **Activer** : opération HUMAINE (`approve`) qui déplace le skill vers `skills/`.
  Kira ne peut pas s'auto-activer un outil.
- **Exécuter** : un skill actif tourne dans le sandbox Docker éphémère (jamais sur
  l'hôte). Même un skill malveillant reste confiné.

Convention d'un skill : du code Python qui lit ses paramètres dans la variable
`args` (un dict) et imprime son résultat sur stdout.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from kira.sandbox import SandboxResult, run_python_in_sandbox

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,40}$")


@dataclass
class Skill:
    name: str
    description: str
    code: str


class SkillStore:
    def __init__(self, root: str | Path) -> None:
        self.dir = Path(root) / "skills"
        self.pending = self.dir / "pending"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pending.mkdir(parents=True, exist_ok=True)

    # --- proposition / activation -------------------------------------- #
    def propose(self, name: str, description: str, code: str) -> str:
        if not _NAME_RE.match(name or ""):
            return "Erreur : nom invalide (minuscules, chiffres, _ ; 2-41 car.)."
        if not code.strip():
            return "Erreur : code vide."
        self._write(self.pending / f"{name}.json", name, description, code)
        return (
            f"Skill '{name}' proposé (en attente de validation humaine). "
            "Un humain doit l'activer avant utilisation."
        )

    def approve(self, name: str) -> str:
        """Active un skill en attente. OPÉRATION HUMAINE (hors agent)."""
        src = self.pending / f"{name}.json"
        if not src.exists():
            return f"Erreur : aucun skill en attente nommé '{name}'."
        src.rename(self.dir / f"{name}.json")
        return f"Skill '{name}' activé."

    # --- lecture -------------------------------------------------------- #
    def get(self, name: str) -> Skill | None:
        path = self.dir / f"{name}.json"
        if not path.exists():
            return None
        d = json.loads(path.read_text(encoding="utf-8"))
        return Skill(d["name"], d.get("description", ""), d["code"])

    def list_active(self) -> list[Skill]:
        out = []
        for p in sorted(self.dir.glob("*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            out.append(Skill(d["name"], d.get("description", ""), d["code"]))
        return out

    # --- exécution (sandbox) ------------------------------------------- #
    def run(
        self,
        name: str,
        args: dict[str, Any] | None = None,
        *,
        runner: Callable[..., SandboxResult] = run_python_in_sandbox,
    ) -> str:
        skill = self.get(name)
        if skill is None:
            return f"Erreur : skill actif '{name}' introuvable (pas activé ?)."
        # On injecte les paramètres puis le code du skill, le tout en sandbox.
        prelude = f"args = {json.dumps(args or {}, ensure_ascii=False)}\n"
        result = runner(prelude + skill.code)
        if result.timed_out:
            return f"Skill '{name}' : timeout dans le sandbox."
        out = (result.stdout or "").strip()
        err = (result.stderr or "").strip()
        parts = [f"exit={result.exit_code}"]
        if out:
            parts.append(out)
        if err:
            parts.append(f"stderr: {err}")
        return "\n".join(parts)

    # --- interne -------------------------------------------------------- #
    @staticmethod
    def _write(path: Path, name: str, description: str, code: str) -> None:
        path.write_text(
            json.dumps(
                {"name": name, "description": description, "code": code},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
