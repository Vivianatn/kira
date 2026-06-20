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


def docker_base() -> list[str] | None:
    """Préfixe de commande pour invoquer Docker, ou None si indisponible.

    Trois cas, dans l'ordre :
      1. Override explicite via KIRA_DOCKER (ex. "wsl -d Ubuntu-24.04 -u root docker").
      2. Docker natif sur le PATH Windows -> ["docker"].
      3. Docker dans WSL (cas de cette machine) -> ["wsl","-d",<distro>,"-u","root","docker"].
         Distro réglable via KIRA_DOCKER_WSL_DISTRO (défaut: Ubuntu-24.04).
    """
    import os

    override = os.environ.get("KIRA_DOCKER")
    if override:
        return override.split()
    if shutil.which("docker"):
        return ["docker"]
    if shutil.which("wsl"):
        distro = os.environ.get("KIRA_DOCKER_WSL_DISTRO", "Ubuntu-24.04")
        return ["wsl", "-d", distro, "-u", "root", "docker"]
    return None


def docker_available() -> bool:
    """True si Docker est joignable (client + démon répondent)."""
    base = docker_base()
    if base is None:
        return False
    try:
        proc = subprocess.run(
            [*base, "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=30,
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
    base = docker_base()
    if base is None or not docker_available():
        raise SandboxUnavailable(
            "Docker indisponible : exécution de code refusée. "
            "Le code ne s'exécute JAMAIS sur l'hôte."
        )

    cmd = [
        *base, "run", "--rm", "--interactive",
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
        # Marge pour le démarrage à froid de WSL + du conteneur, en plus du
        # temps de calcul (timeout) du code lui-même.
        proc = subprocess.run(
            cmd,
            input=code,
            capture_output=True,
            text=True,
            timeout=timeout + 30,
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
