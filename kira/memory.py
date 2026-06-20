"""Mémoire de Kira (phase 4) — court terme + long terme (RAG).

Deux mémoires complémentaires :

- **Court terme** (`ShortTermMemory`) : un tampon roulant des derniers tours de
  conversation. Borné en taille pour ne pas saturer le contexte.

- **Long terme** (`LongTermMemory`) : un magasin vectoriel. On encode chaque
  souvenir en vecteur (embedding), et `recall(query)` retrouve les souvenirs les
  plus proches par similarité cosinus. C'est la brique RAG (Retrieval-Augmented
  Generation) : on récupère les souvenirs pertinents pour les réinjecter dans le
  prompt.

Embeddings — local d'abord, sans dépendance lourde :
- `OllamaEmbedder` : utilise un modèle d'embedding via Ollama (ex.
  `nomic-embed-text`). Vrais embeddings sémantiques, en local.
- `HashEmbedder` : repli déterministe sans dépendance (sac-de-mots hashé).
  Pas sémantique mais fonctionnel — utilisé pour les tests et le hors-ligne.

Aucune dépendance externe (pas de numpy/chromadb) : le magasin est un simple
fichier JSONL et la similarité cosinus est en Python pur (volumes modestes).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

_WORD = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def cosine(a: list[float], b: list[float]) -> float:
    """Similarité cosinus entre deux vecteurs (0 si l'un est nul)."""
    if len(a) != len(b):
        raise ValueError("vecteurs de tailles différentes")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# --------------------------------------------------------------------------- #
# Embedders
# --------------------------------------------------------------------------- #
class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> list[float]: ...


class HashEmbedder:
    """Embedder déterministe sans dépendance (sac-de-mots hashé, L2-normalisé).

    Repère le recouvrement lexical entre textes : utile pour tests et hors-ligne.
    Déterministe entre processus (hashlib, pas le hash() randomisé de Python).
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _tokenize(text):
            idx = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % self.dim
            vec[idx] += 1.0
        return vec


class OllamaEmbedder:
    """Embeddings via Ollama (`/api/embeddings`). Modèle local, ex. nomic-embed-text."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        *,
        host: str | None = None,
        timeout: int = 60,
    ) -> None:
        self.model = model
        self.host = host or "http://localhost:11434"
        self.timeout = timeout
        self.dim = 0  # découvert au premier appel

    def embed(self, text: str) -> list[float]:
        payload = json.dumps({"model": self.model, "prompt": text}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        emb = data.get("embedding", [])
        self.dim = len(emb)
        return emb


# --------------------------------------------------------------------------- #
# Mémoire court terme
# --------------------------------------------------------------------------- #
@dataclass
class ShortTermMemory:
    """Tampon roulant des derniers tours de conversation (borné)."""

    max_turns: int = 20
    turns: list[dict[str, str]] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})
        # On garde seulement les `max_turns` derniers.
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    def context(self) -> list[dict[str, str]]:
        return list(self.turns)

    def clear(self) -> None:
        self.turns.clear()


# --------------------------------------------------------------------------- #
# Mémoire long terme (magasin vectoriel + RAG)
# --------------------------------------------------------------------------- #
@dataclass
class MemoryItem:
    id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any]
    ts: float

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "ts": self.ts,
        }


@dataclass
class Recall:
    text: str
    score: float
    metadata: dict[str, Any]
    id: str


class LongTermMemory:
    """Magasin vectoriel persistant (JSONL) avec recherche par cosinus."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        *,
        path: str | Path | None = None,
    ) -> None:
        self.embedder = embedder or HashEmbedder()
        self.path = Path(path) if path else None
        self.items: list[MemoryItem] = []
        self._load()

    # --- persistance ---------------------------------------------------- #
    def _load(self) -> None:
        if not self.path or not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            self.items.append(
                MemoryItem(
                    id=d["id"],
                    text=d["text"],
                    embedding=d["embedding"],
                    metadata=d.get("metadata", {}),
                    ts=d.get("ts", 0.0),
                )
            )

    def _append(self, item: MemoryItem) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(item.to_json(), ensure_ascii=False) + "\n")

    # --- API ------------------------------------------------------------ #
    def add(self, text: str, **metadata: Any) -> MemoryItem:
        item = MemoryItem(
            id=uuid.uuid4().hex[:12],
            text=text,
            embedding=self.embedder.embed(text),
            metadata=metadata,
            ts=time.time(),
        )
        self.items.append(item)
        self._append(item)
        return item

    def search(self, query: str, k: int = 3, min_score: float = 0.0) -> list[Recall]:
        if not self.items:
            return []
        qv = self.embedder.embed(query)
        scored = [
            Recall(it.text, cosine(qv, it.embedding), it.metadata, it.id)
            for it in self.items
        ]
        scored = [r for r in scored if r.score > min_score]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]

    def __len__(self) -> int:
        return len(self.items)


# --------------------------------------------------------------------------- #
# Façade combinée
# --------------------------------------------------------------------------- #
class Memory:
    """Réunit mémoire court terme et long terme derrière une seule API."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        *,
        store_path: str | Path | None = None,
        max_turns: int = 20,
    ) -> None:
        self.short = ShortTermMemory(max_turns=max_turns)
        self.long = LongTermMemory(embedder=embedder, path=store_path)

    # court terme
    def add_turn(self, role: str, content: str) -> None:
        self.short.add(role, content)

    def conversation(self) -> list[dict[str, str]]:
        return self.short.context()

    # long terme / RAG
    def remember(self, text: str, **metadata: Any) -> MemoryItem:
        return self.long.add(text, **metadata)

    def recall(self, query: str, k: int = 3, min_score: float = 0.0) -> list[Recall]:
        return self.long.search(query, k=k, min_score=min_score)

    def recall_as_text(self, query: str, k: int = 3) -> str:
        """Souvenirs pertinents formatés pour injection dans un prompt (RAG)."""
        hits = self.recall(query, k=k)
        if not hits:
            return ""
        lines = [f"- {h.text}" for h in hits]
        return "Souvenirs pertinents :\n" + "\n".join(lines)
