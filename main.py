"""Point d'entrée CLI de Kira — un REPL minimal pour dialoguer avec l'agent.

Usage :
    python main.py                 # mode interactif (REPL)
    python main.py "ta question"   # un seul tour puis sortie

Le backend est choisi via .env (KIRA_BACKEND). En "anthropic", il faut
ANTHROPIC_API_KEY. Pour tester la mécanique sans clé ni réseau, utilise le
backend "mock" (KIRA_BACKEND=mock) — il ne raisonne pas, mais la boucle tourne.

Validation humaine : si une action sensible est demandée, on te le demande
ici en console (human-in-the-loop).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# S'assure que la racine du projet est importable, même sous un Python
# « embeddable » (dont le sys.path est verrouillé par un fichier ._pth).
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from kira.agent import Agent
from kira.engine import Engine
from kira.security import EnforcementLayer


def console_approval(tool_name: str, params: dict[str, Any]) -> bool:
    """Demande une confirmation humaine en console pour une action sensible."""
    print(f"\n[VALIDATION REQUISE] Kira veut utiliser '{tool_name}' avec :")
    print(f"    {params}")
    answer = input("Autoriser cette action ? [o/N] ").strip().lower()
    return answer in {"o", "oui", "y", "yes"}


def build_agent() -> Agent:
    security = EnforcementLayer(policy_path="policy.yaml")
    engine = Engine.from_config()
    return Agent(engine, security, approval_handler=console_approval)


def run_once(agent: Agent, prompt: str) -> None:
    result = agent.run(prompt)
    print(f"\nKira > {result.answer}")
    if result.hit_step_limit:
        print("(note : limite de pas atteinte)")


def repl(agent: Agent) -> None:
    print("Kira — assistant agentique (Ctrl-C ou 'quit' pour sortir).")
    while True:
        try:
            prompt = input("\nVous > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nÀ bientôt.")
            return
        if prompt.lower() in {"quit", "exit", "q"}:
            print("À bientôt.")
            return
        if not prompt:
            continue
        run_once(agent, prompt)


def main(argv: list[str]) -> int:
    agent = build_agent()
    if len(argv) > 1:
        run_once(agent, " ".join(argv[1:]))
    else:
        repl(agent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
