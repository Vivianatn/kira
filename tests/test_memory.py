"""Tests de la mémoire (phase 4). On utilise HashEmbedder (offline, déterministe)."""

from __future__ import annotations

from kira.memory import (
    HashEmbedder,
    LongTermMemory,
    Memory,
    ShortTermMemory,
    cosine,
)


# --------------------------------------------------------------------------- #
# Court terme
# --------------------------------------------------------------------------- #
def test_short_term_buffer_trims():
    stm = ShortTermMemory(max_turns=3)
    for i in range(5):
        stm.add("user", f"msg{i}")
    ctx = stm.context()
    assert len(ctx) == 3
    assert ctx[0]["content"] == "msg2"  # les 2 plus anciens sont tombés
    assert ctx[-1]["content"] == "msg4"


def test_short_term_clear():
    stm = ShortTermMemory()
    stm.add("user", "x")
    stm.clear()
    assert stm.context() == []


# --------------------------------------------------------------------------- #
# Embedder & cosinus
# --------------------------------------------------------------------------- #
def test_hash_embedder_deterministic():
    e = HashEmbedder(dim=64)
    assert e.embed("bonjour kira") == e.embed("bonjour kira")
    assert len(e.embed("test")) == 64


def test_cosine_basics():
    assert cosine([1, 0], [1, 0]) == 1.0
    assert cosine([1, 0], [0, 1]) == 0.0
    assert cosine([0, 0], [1, 1]) == 0.0  # vecteur nul -> 0


# --------------------------------------------------------------------------- #
# Long terme / RAG
# --------------------------------------------------------------------------- #
def test_recall_finds_relevant_memory():
    ltm = LongTermMemory(embedder=HashEmbedder())
    ltm.add("Le chat de Vivian s'appelle Minou")
    ltm.add("La capitale de la France est Paris")
    ltm.add("Kira tourne en local avec Ollama")

    hits = ltm.search("comment s'appelle le chat ?", k=1)
    assert hits
    assert "Minou" in hits[0].text


def test_recall_empty_store():
    ltm = LongTermMemory(embedder=HashEmbedder())
    assert ltm.search("rien") == []


def test_metadata_roundtrip():
    ltm = LongTermMemory(embedder=HashEmbedder())
    ltm.add("fait important", source="user", tag="prefs")
    hit = ltm.search("fait important", k=1)[0]
    assert hit.metadata["source"] == "user"
    assert hit.metadata["tag"] == "prefs"


def test_persistence_reload(tmp_path):
    store = tmp_path / "mem" / "store.jsonl"
    ltm = LongTermMemory(embedder=HashEmbedder(), path=store)
    ltm.add("souvenir persistant", source="test")
    assert store.exists()

    # Recharge depuis le disque dans une nouvelle instance.
    ltm2 = LongTermMemory(embedder=HashEmbedder(), path=store)
    assert len(ltm2) == 1
    hit = ltm2.search("souvenir persistant", k=1)[0]
    assert "persistant" in hit.text
    assert hit.metadata["source"] == "test"


# --------------------------------------------------------------------------- #
# Façade combinée
# --------------------------------------------------------------------------- #
def test_memory_facade():
    mem = Memory(embedder=HashEmbedder())
    mem.add_turn("user", "salut")
    mem.add_turn("assistant", "bonjour")
    assert len(mem.conversation()) == 2

    mem.remember("Vivian préfère les réponses courtes", tag="pref")
    txt = mem.recall_as_text("quelles préférences de réponse ?", k=1)
    assert "Vivian" in txt
    assert txt.startswith("Souvenirs pertinents")


def test_recall_as_text_empty():
    mem = Memory(embedder=HashEmbedder())
    assert mem.recall_as_text("rien") == ""
