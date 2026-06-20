"""Boucle ReAct — l'orchestrateur de Kira.

Cycle : Thought -> Action -> Observation -> ... -> Answer.

À chaque tour :
    1. Le moteur (LLM) réfléchit et, éventuellement, demande des appels d'outils.
    2. CHAQUE appel d'outil passe par la couche de sécurité AVANT exécution.
       - refusé          -> l'observation renvoyée au modèle est « action refusée ».
       - validation requise -> on demande l'aval humain via `approval_handler`.
       - autorisé        -> on exécute l'outil et on renvoie l'observation.
    3. On boucle jusqu'à une réponse finale (aucun appel d'outil) ou jusqu'à
       `max_steps` (anti-emballement).

L'agent ne connaît rien du fournisseur LLM : il travaille avec le format de
messages neutre de `engine.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from kira.engine import Engine, EngineResponse, ToolCall
from kira.security import EnforcementLayer, SecurityError
from kira.tools import Tool, build_registry, tool_schemas

if TYPE_CHECKING:
    from kira.memory import Memory


# Un gestionnaire d'approbation reçoit (tool_name, params) et renvoie True/False.
ApprovalHandler = Callable[[str, dict[str, Any]], bool]


def deny_all_approvals(tool_name: str, params: dict[str, Any]) -> bool:
    """Par défaut, refuser toute action sensible (le plus sûr).

    En usage interactif, on remplacera ce handler par un vrai prompt humain.
    """
    return False


@dataclass
class Step:
    """Trace d'un tour de boucle (pour debug / journal / tests)."""

    index: int
    thought: str = ""
    actions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentResult:
    answer: str
    steps: list[Step]
    stopped_reason: str  # "final" | "max_steps"

    @property
    def hit_step_limit(self) -> bool:
        return self.stopped_reason == "max_steps"


class Agent:
    """Orchestre moteur + outils sous le contrôle de la couche de sécurité."""

    def __init__(
        self,
        engine: Engine,
        security: EnforcementLayer,
        *,
        tools: dict[str, Tool] | None = None,
        max_steps: int | None = None,
        approval_handler: ApprovalHandler | None = None,
        memory: "Memory | None" = None,
        recall_k: int = 3,
    ) -> None:
        self.engine = engine
        self.security = security
        self.tools = tools if tools is not None else build_registry(security)
        self.max_steps = max_steps if max_steps is not None else security.max_steps
        self.approval_handler = approval_handler or deny_all_approvals
        self._schemas = tool_schemas(self.tools)
        # Mémoire optionnelle : court terme (conversation) + long terme (RAG).
        self.memory = memory
        self.recall_k = recall_k

    def run(self, user_input: str) -> AgentResult:
        messages: list[dict[str, Any]] = []
        recalled = ""

        if self.memory is not None:
            # Court terme : on rejoue la conversation précédente comme contexte.
            messages.extend(self.memory.conversation())
            # Long terme (RAG) : souvenirs pertinents pour cette requête.
            recalled = self.memory.recall_as_text(user_input, k=self.recall_k)

        # On augmente le message courant avec les souvenirs rappelés (s'il y en a),
        # sans polluer ce qu'on stockera ensuite (on garde le texte brut à part).
        user_content = user_input
        if recalled:
            user_content = f"[Mémoire]\n{recalled}\n\n[Message]\n{user_input}"
        messages.append({"role": "user", "content": user_content})
        steps: list[Step] = []

        for i in range(self.max_steps):
            response: EngineResponse = self.engine.think(messages, self._schemas)
            step = Step(index=i, thought=response.text)

            # Réponse finale : pas d'appel d'outil -> on s'arrête.
            if response.is_final:
                steps.append(step)
                self._remember(user_input, response.text)
                return AgentResult(
                    answer=response.text, steps=steps, stopped_reason="final"
                )

            # Le modèle veut utiliser des outils : on enregistre son tour...
            messages.append(
                {
                    "role": "assistant",
                    "content": response.text,
                    "tool_calls": response.tool_calls,
                }
            )

            # ... puis on traite chaque appel, sécurité d'abord.
            for call in response.tool_calls:
                observation = self._handle_tool_call(call)
                step.actions.append(
                    {"tool": call.name, "input": call.input, "observation": observation}
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": observation,
                    }
                )

            steps.append(step)

        # Limite de pas atteinte sans réponse finale.
        answer = "(arrêt : limite de pas atteinte sans réponse finale)"
        self._remember(user_input, answer)
        return AgentResult(answer=answer, steps=steps, stopped_reason="max_steps")

    def _remember(self, user_input: str, answer: str) -> None:
        """Persiste l'échange en mémoire (court terme + long terme), si activée."""
        if self.memory is None:
            return
        self.memory.add_turn("user", user_input)
        self.memory.add_turn("assistant", answer)
        # Souvenir long terme de l'échange, récupérable plus tard par RAG.
        self.memory.remember(f"Q: {user_input}\nR: {answer}", kind="exchange")

    # ------------------------------------------------------------------ #
    # Traitement d'un appel d'outil : LE point de contrôle sécurité.
    # ------------------------------------------------------------------ #
    def _handle_tool_call(self, call: ToolCall) -> str:
        tool = self.tools.get(call.name)
        if tool is None:
            # Outil inconnu / non instancié (donc hors allowlist).
            return f"Refusé : outil '{call.name}' indisponible ou non autorisé."

        action = tool.to_action(call.input)

        # 1) La politique autorise-t-elle l'action ?
        try:
            decision = self.security.evaluate(action)
        except SecurityError as exc:
            return f"Refusé : {exc}"
        if not decision.allowed:
            return f"Refusé : {decision.reason}"

        # 2) Validation humaine nécessaire ?
        if decision.needs_approval:
            if not self.approval_handler(call.name, call.input):
                return "Refusé : validation humaine non accordée."

        # 3) Exécution de l'outil.
        try:
            return tool.run(call.input)
        except SecurityError as exc:
            return f"Refusé : {exc}"
        except Exception as exc:  # noqa: BLE001 - observation lisible pour le modèle
            return f"Erreur d'exécution de l'outil '{call.name}' : {exc}"
