"""Sandbox d'exécution de code — Docker éphémère, jamais sur l'hôte.

Principe de sécurité (cf. PROJET_KIRA.md §8bis) : tout code arbitraire s'exécute
dans un conteneur Docker JETABLE et CONFINÉ :
    - `--rm`            : conteneur supprimé après usage (éphémère).
    - `--network none`  : aucun accès réseau.
    - `--memory`        : plafond mémoire.
    - `--cpus`          : plafond CPU.
    - timeout dur       : on tue le conteneur s'il dépasse.
    - pas de montage de l'hôte : le code ne voit pas le système de fichiers local.

FAIL-CLOSED : si Docker n'est pas disponible, on REFUSE d'exécuter (on ne se
rabat JAMAIS sur l'hôte). C'est la garantie centrale de la phase 3.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


class SandboxUnavailable(RuntimeError):
    """Levée quand aucun sandbox sûr n'est disponible (Docker absent/inactif)."""


@dataclass
class SandboxResult:
    ok: bool
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False


def docker_available() -> bool:
    """True si le client Docker est présent ET le démon répond."""
    if shutil.which("docker") is None:
        return False
    try:
        proc = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return proc.returncode == 0 and bool(proc.stdout.strip())
    except (subprocess.SubprocessError, OSError):
        return False


def run_python_in_sandbox(
    code: str,
    *,
    image: str = "python:3.12-slim",
    timeout: int = 10,
    memory: str = "256m",
    cpus: str = "1.0",
    network: str = "none",
) -> SandboxResult:
    """Exécute `code` Python dans un conteneur Docker éphémère et confiné.

    Lève SandboxUnavailable si Docker n'est pas disponible (fail-closed).
    """
    if not docker_available():
        raise SandboxUnavailable(
            "Docker indisponible : exécution de code refusée. "
            "Le code ne s'exécute JAMAIS sur l'hôte. Installe/démarre Docker Desktop."
        )

    cmd = [
        "docker", "run", "--rm", "--interactive",
        "--network", network,
        "--memory", memory,
        "--cpus", cpus,
        "--pids-limit", "128",
        # Empêche l'escalade de privilèges et durcit le conteneur.
        "--security-opt", "no-new-privileges",
        "--read-only",
        image,
        "python", "-",  # lit le code sur stdin
    ]
    try:
        proc = subprocess.run(
            cmd,
            input=code,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return SandboxResult(
            ok=False, stdout="", stderr="(timeout)", exit_code=124, timed_out=True
        )
    return SandboxResult(
        ok=proc.returncode == 0,
        stdout=proc.stdout,
        stderr=proc.stderr,
        exit_code=proc.returncode,
    )
