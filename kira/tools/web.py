"""Outil de recherche web — sûr, sans clé API par défaut.

Fournisseur par défaut : DuckDuckGo Instant Answer API (aucune clé requise).
C'est volontairement modeste mais sans dépendance externe ni secret à gérer ;
on pourra brancher un fournisseur plus riche (Tavily, SerpAPI...) plus tard
via la politique, sans changer le reste de l'agent.

Cet outil est en lecture seule (il interroge le web, ne publie rien).
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any

from kira.security import Action, EnforcementLayer, SecurityError


class WebTool:
    name = "web"
    description = (
        "Recherche des informations sur le web et renvoie un court résumé "
        "avec les meilleurs résultats. Fournis une 'query' textuelle."
    )
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "La requête de recherche.",
            },
        },
        "required": ["query"],
    }

    def __init__(self, security: EnforcementLayer) -> None:
        self.security = security

    def to_action(self, params: dict[str, Any]) -> Action:
        return Action(
            tool=self.name,
            name="search",
            params={"query": params.get("query", "")},
        )

    def run(self, params: dict[str, Any]) -> str:
        # Défense en profondeur : on revalide auprès de la sécurité.
        # On renvoie une observation de refus plutôt que de lever.
        try:
            self.security.check(self.to_action(params))
        except SecurityError as exc:
            return f"Refusé : {exc}"

        query = str(params.get("query", "")).strip()
        if not query:
            return "Erreur : requête vide."

        cfg = self.security.tool_config("web")
        provider = str(cfg.get("provider", "duckduckgo")).lower()
        max_results = int(cfg.get("max_results", 5))

        if provider == "duckduckgo":
            return self._search_duckduckgo(query, max_results)
        return f"Erreur : fournisseur de recherche inconnu '{provider}'."

    # --- DuckDuckGo Instant Answer --------------------------------------- #
    def _search_duckduckgo(self, query: str, max_results: int) -> str:
        url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode(
            {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Kira/0.1"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - on remonte une observation lisible
            return f"Erreur réseau lors de la recherche : {exc}"

        lines: list[str] = []

        abstract = data.get("AbstractText") or data.get("Abstract")
        if abstract:
            source = data.get("AbstractSource", "")
            lines.append(f"Résumé ({source}): {abstract}")

        answer = data.get("Answer")
        if answer:
            lines.append(f"Réponse directe : {answer}")

        for topic in self._flatten_topics(data.get("RelatedTopics", [])):
            if len(lines) >= max_results + 2:
                break
            text = topic.get("Text")
            first_url = topic.get("FirstURL")
            if text:
                lines.append(f"- {text}" + (f" ({first_url})" if first_url else ""))

        if not lines:
            return f"Aucun résultat exploitable pour : {query}"
        return "\n".join(lines)

    @staticmethod
    def _flatten_topics(topics: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """RelatedTopics peut contenir des groupes imbriqués ; on aplatit."""
        flat: list[dict[str, Any]] = []
        for t in topics:
            if "Topics" in t and isinstance(t["Topics"], list):
                flat.extend(t["Topics"])
            else:
                flat.append(t)
        return flat
