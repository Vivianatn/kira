"""Moteur de raisonnement — le « cerveau » de Kira.

Le moteur appelle un LLM pour décider quoi faire. Le backend est
INTERCHANGEABLE (cf. PROJET_KIRA.md §8.1) :

    - "anthropic" : API Claude (défaut, recommandé vu les 2 Go de VRAM).
    - "ollama"    : modèle local via Ollama (lent sur CPU, expérimental).
    - "mock"      : backend déterministe pour les tests (aucune dépendance).

Le moteur expose une seule méthode publique : `think(messages, tools)`.
Il manipule un format de messages NEUTRE (indépendant du fournisseur) ;
chaque backend traduit ce format vers son API.

Format de message interne (liste de dicts) :
    {"role": "user",      "content": "<texte>"}
    {"role": "assistant", "content": "<texte>", "tool_calls": [ToolCall, ...]}
    {"role": "tool",      "tool_call_id": "<id>", "content": "<résultat>"}
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


# --------------------------------------------------------------------------- #
# Chargement minimal du .env (sans dépendance ; python-dotenv si dispo)
# --------------------------------------------------------------------------- #
def load_dotenv(path: str | os.PathLike[str] = ".env") -> None:
    """Charge les variables d'un fichier .env dans os.environ.

    Implémentation autonome (zéro dépendance) ; on ne remplace pas une
    variable déjà présente dans l'environnement.
    """
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


# --------------------------------------------------------------------------- #
# Types d'échange
# --------------------------------------------------------------------------- #
@dataclass
class ToolCall:
    """Demande d'appel d'outil émise par le modèle."""

    id: str
    name: str
    input: dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineResponse:
    """Réponse du moteur pour un tour de boucle.

    text       : texte produit par le modèle (raisonnement / réponse finale).
    tool_calls : outils que le modèle souhaite appeler (vide = réponse finale).
    stop_reason: indication du backend ("end_turn", "tool_use", ...).
    raw        : objet brut du backend (debug).
    """

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    raw: Any = None

    @property
    def is_final(self) -> bool:
        return not self.tool_calls


# --------------------------------------------------------------------------- #
# Interface backend
# --------------------------------------------------------------------------- #
class Backend(Protocol):
    def generate(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> EngineResponse: ...


# --------------------------------------------------------------------------- #
# Backend Anthropic (API Claude) — backend par défaut
# --------------------------------------------------------------------------- #
class AnthropicBackend:
    """Appelle l'API Messages de Claude.

    Nécessite le paquet `anthropic` et la variable ANTHROPIC_API_KEY.
    Import paresseux : on n'impose pas la dépendance aux tests / au backend mock.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        *,
        api_key: str | None = None,
        max_tokens: int = 2048,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None  # initialisé paresseusement

    def _ensure_client(self):
        if self._client is not None:
            return
        if not self._api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY manquante. Renseigne-la dans .env "
                "(voir .env.example)."
            )
        try:
            import anthropic  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Le paquet 'anthropic' est requis pour le backend Claude. "
                "Installe-le : pip install anthropic"
            ) from exc
        self._client = anthropic.Anthropic(api_key=self._api_key)

    def generate(self, system, messages, tools) -> EngineResponse:
        self._ensure_client()
        api_messages = [self._to_anthropic(m) for m in messages]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": api_messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        resp = self._client.messages.create(**kwargs)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, input=dict(block.input))
                )
        return EngineResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=resp.stop_reason,
            raw=resp,
        )

    @staticmethod
    def _to_anthropic(msg: dict[str, Any]) -> dict[str, Any]:
        """Traduit un message interne vers le format de l'API Anthropic."""
        role = msg["role"]
        if role == "user":
            return {"role": "user", "content": msg["content"]}
        if role == "assistant":
            content: list[dict[str, Any]] = []
            if msg.get("content"):
                content.append({"type": "text", "text": msg["content"]})
            for call in msg.get("tool_calls", []):
                content.append(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.input,
                    }
                )
            return {"role": "assistant", "content": content}
        if role == "tool":
            # Les résultats d'outils sont portés par un message 'user' chez Anthropic.
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg["tool_call_id"],
                        "content": msg["content"],
                    }
                ],
            }
        raise ValueError(f"rôle de message inconnu : {role}")


# --------------------------------------------------------------------------- #
# Backend Ollama (modèle local) — expérimental, texte seul pour l'instant
# --------------------------------------------------------------------------- #
class OllamaBackend:
    """Backend local via Ollama (http://localhost:11434).

    Implémentation MINIMALE : génération de texte uniquement, sans tool calling
    structuré (Ollama gère les outils différemment selon les modèles). Suffisant
    pour expérimenter le dialogue hors-ligne. À enrichir en phase ultérieure.
    """

    def __init__(
        self,
        model: str = "llama3.2:3b",
        *,
        host: str | None = None,
    ) -> None:
        self.model = model
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def generate(self, system, messages, tools) -> EngineResponse:
        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})
        for m in messages:
            if m["role"] == "tool":
                api_messages.append(
                    {"role": "tool", "content": str(m.get("content", ""))}
                )
            else:
                api_messages.append(
                    {"role": m["role"], "content": str(m.get("content", ""))}
                )
        payload = json.dumps(
            {"model": self.model, "messages": api_messages, "stream": False}
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data.get("message", {}).get("content", "")
        return EngineResponse(text=text, stop_reason="end_turn", raw=data)


# --------------------------------------------------------------------------- #
# Backend Mock (tests) — déterministe, scriptable, sans réseau
# --------------------------------------------------------------------------- #
class MockBackend:
    """Backend de test : renvoie des réponses pré-scriptées.

    - Si `responses` est fourni : on dépile une EngineResponse par appel.
      Une fois la liste épuisée, on répète la dernière réponse.
    - Sinon, on s'appuie sur `responder(system, messages, tools)`.
    """

    def __init__(
        self,
        responses: list[EngineResponse] | None = None,
        responder=None,
    ) -> None:
        self._responses = list(responses or [])
        self._responder = responder
        self.calls: list[dict[str, Any]] = []  # historique pour les assertions

    def generate(self, system, messages, tools) -> EngineResponse:
        self.calls.append({"system": system, "messages": messages, "tools": tools})
        if self._responder is not None:
            return self._responder(system, messages, tools)
        if not self._responses:
            return EngineResponse(text="(mock) aucune réponse scriptée")
        if len(self._responses) == 1:
            return self._responses[0]
        return self._responses.pop(0)


# --------------------------------------------------------------------------- #
# Le moteur public
# --------------------------------------------------------------------------- #
class Engine:
    """Façade stable au-dessus d'un backend interchangeable.

    Construis-le via la config (env / arguments) avec `Engine.from_config(...)`,
    ou en passant directement un backend (`Engine(backend=...)`).
    """

    DEFAULT_SYSTEM = (
        "Tu es Kira, un assistant agentique. Tu raisonnes étape par étape et "
        "tu utilises les outils mis à ta disposition quand c'est utile. "
        "Quand tu as la réponse finale, donne-la directement sans appeler d'outil."
    )

    def __init__(self, backend: Backend, *, system: str | None = None) -> None:
        self.backend = backend
        self.system = system if system is not None else self.DEFAULT_SYSTEM

    @classmethod
    def from_config(
        cls,
        *,
        backend: str | None = None,
        model: str | None = None,
        system: str | None = None,
        load_env: bool = True,
    ) -> "Engine":
        """Construit le moteur à partir de la config (env par défaut).

        Variables d'environnement reconnues :
            KIRA_BACKEND  : "anthropic" (défaut) | "ollama" | "mock"
            KIRA_MODEL    : identifiant du modèle pour le backend choisi
            ANTHROPIC_API_KEY, OLLAMA_HOST : selon le backend
        """
        if load_env:
            load_dotenv()
        backend_name = (backend or os.environ.get("KIRA_BACKEND", "anthropic")).lower()
        model = model or os.environ.get("KIRA_MODEL")

        if backend_name == "anthropic":
            impl: Backend = (
                AnthropicBackend(model=model) if model else AnthropicBackend()
            )
        elif backend_name == "ollama":
            impl = OllamaBackend(model=model) if model else OllamaBackend()
        elif backend_name == "mock":
            impl = MockBackend()
        else:
            raise ValueError(f"Backend inconnu : {backend_name!r}")
        return cls(impl, system=system)

    def think(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> EngineResponse:
        """Un tour de raisonnement : renvoie texte + éventuels appels d'outils."""
        return self.backend.generate(self.system, messages, tools or [])
