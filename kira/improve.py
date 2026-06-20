"""Auto-amélioration encadrée (niveau C) — Kira modifie son code, sous garde-fous.

Règle du projet : Kira peut s'améliorer, mais JAMAIS au prix de la sécurité.
Deux niveaux, comme demandé :

- **Mineur** (peu de lignes, fichier non protégé) ET **tests verts** -> appliqué
  automatiquement (avec sauvegarde + journal).
- **Important / fichier protégé / tests rouges** -> NON appliqué : la proposition
  est écrite dans `proposals/` pour validation humaine.

Garde-fous non négociables :
1. **Fonction de fitness** : toute modification est validée par la SUITE pytest
   complète. Si un test casse, on REVIENT en arrière. Aucune exception.
2. **Fichiers protégés** : les briques de sécurité ne sont JAMAIS modifiées
   automatiquement (une IA ne doit pas pouvoir affaiblir ses propres garde-fous).
3. **Périmètre** : on ne peut éditer que sous `kira/` et `tests/`. Rien d'autre.
4. **Réversible** : sauvegarde de l'original + journal append-only de chaque action.
"""

from __future__ import annotations

import difflib
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Fichiers que l'auto-amélioration ne modifie JAMAIS toute seule.
PROTECTED = {
    "kira/security.py",
    "kira/sandbox.py",
    "kira/improve.py",
    "kira/agent.py",
    "kira/tools/system.py",
    "policy.yaml",
}

# Seuls ces préfixes de chemin sont éditables.
EDITABLE_ROOTS = ("kira/", "tests/")


@dataclass
class ImprovementResult:
    applied: bool
    needs_approval: bool
    tests_passed: bool | None
    diff_lines: int
    reason: str
    proposal_path: str | None = None


# Un runner de tests renvoie (succès, sortie).
TestRunner = Callable[[], tuple[bool, str]]


def pytest_runner(project_root: Path) -> TestRunner:
    """Runner par défaut : lance la suite pytest avec l'interpréteur courant."""

    def run() -> tuple[bool, str]:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=600,
        )
        return proc.returncode == 0, proc.stdout + proc.stderr

    return run


class SelfImprover:
    def __init__(
        self,
        project_root: str | os.PathLike[str],
        *,
        max_minor_lines: int = 10,
        unrestricted: bool = False,
        test_runner: TestRunner | None = None,
        audit_log: str | os.PathLike[str] | None = None,
    ) -> None:
        self.root = Path(project_root).resolve()
        self.max_minor_lines = max_minor_lines
        # unrestricted : Kira applique toute modification (quelle que soit la
        # taille) tant que les tests passent. SEULE exception non négociable :
        # les fichiers PROTECTED (qui gouvernent ce que Kira peut faire à la
        # machine) demandent toujours une validation humaine.
        self.unrestricted = unrestricted
        self.test_runner = test_runner or pytest_runner(self.root)
        self.audit_log = Path(audit_log).resolve() if audit_log else None

    # ------------------------------------------------------------------ #
    def _rel(self, path: str | os.PathLike[str]) -> str:
        p = (self.root / path).resolve()
        return p.relative_to(self.root).as_posix()

    def _is_editable(self, rel: str) -> bool:
        return rel.startswith(EDITABLE_ROOTS) and rel.endswith(".py")

    @staticmethod
    def _diff_lines(old: str, new: str) -> int:
        diff = difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="")
        return sum(1 for ln in diff if ln and ln[0] in "+-" and ln[:2] not in ("+++", "---"))

    # ------------------------------------------------------------------ #
    def view(self, rel_path: str) -> str | None:
        """Renvoie le contenu actuel d'un fichier éditable, ou None si interdit.

        Sert à Kira pour lire son code avant de proposer une modification.
        """
        try:
            rel = self._rel(rel_path)
        except ValueError:
            return None
        if not self._is_editable(rel):
            return None
        target = self.root / rel
        return target.read_text(encoding="utf-8") if target.exists() else None

    def propose(self, rel_path: str, new_content: str) -> ImprovementResult:
        """Tente d'appliquer une modification, sous tous les garde-fous."""
        try:
            rel = self._rel(rel_path)
        except ValueError:
            return self._log(rel_path, ImprovementResult(
                False, False, None, 0, "chemin hors du projet"))

        # Périmètre éditable strict.
        if not self._is_editable(rel):
            return self._log(rel, ImprovementResult(
                False, False, None, 0,
                f"hors périmètre éditable (autorisé: {EDITABLE_ROOTS}, .py)"))

        target = self.root / rel
        old_content = target.read_text(encoding="utf-8") if target.exists() else ""
        if new_content == old_content:
            return self._log(rel, ImprovementResult(
                False, False, None, 0, "aucun changement"))

        diff_lines = self._diff_lines(old_content, new_content)
        protected = rel in PROTECTED
        # En mode unrestricted, toute modif d'un fichier NON protégé s'applique
        # (peu importe la taille). Les fichiers protégés restent gardés.
        if self.unrestricted:
            auto_ok = not protected
        else:
            auto_ok = (diff_lines <= self.max_minor_lines) and not protected

        # On met la modification en scène (avec sauvegarde) pour lancer les tests.
        backup = target.with_suffix(target.suffix + ".bak")
        existed = target.exists()
        if existed:
            shutil.copy2(target, backup)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(new_content, encoding="utf-8")

        try:
            passed, _output = self.test_runner()
        finally:
            pass

        if not passed:
            # Fonction de fitness échouée : on revient en arrière, toujours.
            self._restore(target, backup, existed)
            return self._log(rel, ImprovementResult(
                False, False, False, diff_lines,
                "tests en échec : modification annulée (fitness)"))

        # Tests verts.
        if auto_ok:
            # Non protégé + tests OK -> appliqué automatiquement (taille libre
            # en mode unrestricted).
            if backup.exists():
                backup.unlink()
            mode = "appliquée (mode illimité, tests OK)" if self.unrestricted else \
                "mineure appliquée automatiquement (tests OK)"
            return self._log(rel, ImprovementResult(
                True, False, True, diff_lines, f"modification {mode}"))

        # Protégé (toujours), ou trop gros hors mode unrestricted : on n'applique
        # PAS, on revient et on propose pour validation humaine.
        self._restore(target, backup, existed)
        proposal = self._write_proposal(rel, new_content)
        reason = (
            "fichier de sécurité protégé"
            if protected
            else f"modification importante (>{self.max_minor_lines} lignes)"
        )
        return self._log(rel, ImprovementResult(
            False, True, True, diff_lines,
            f"{reason} : validation humaine requise (proposition enregistrée)",
            proposal_path=str(proposal)))

    # ------------------------------------------------------------------ #
    @staticmethod
    def _restore(target: Path, backup: Path, existed: bool) -> None:
        if existed and backup.exists():
            shutil.move(str(backup), str(target))
        elif not existed and target.exists():
            target.unlink()  # le fichier n'existait pas avant : on le retire.

    def _write_proposal(self, rel: str, content: str) -> Path:
        prop = self.root / "proposals" / (rel.replace("/", "__") + ".proposed")
        prop.parent.mkdir(parents=True, exist_ok=True)
        prop.write_text(content, encoding="utf-8")
        return prop

    def _log(self, rel: str, result: ImprovementResult) -> ImprovementResult:
        if self.audit_log is not None:
            import json

            entry = {
                "ts": time.time(),
                "path": str(rel),
                "applied": result.applied,
                "needs_approval": result.needs_approval,
                "tests_passed": result.tests_passed,
                "diff_lines": result.diff_lines,
                "reason": result.reason,
            }
            self.audit_log.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return result
