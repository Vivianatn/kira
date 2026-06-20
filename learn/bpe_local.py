"""
BPE (Byte-Pair Encoding) entraine localement sur un corpus — zero telechargement.

Alternative pedagogique a tiktoken quand le reseau / SSL bloque le vocab GPT-2.
Algorithme : fusion iteratif des paires de bytes les plus frequentes.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


def _byte_tokens(text: str) -> list[bytes]:
    return [bytes([b]) for b in text.encode("utf-8")]


def _pair_counts(tokens: list[list[bytes]]) -> Counter[tuple[bytes, bytes]]:
    counts: Counter[tuple[bytes, bytes]] = Counter()
    for seq in tokens:
        for i in range(len(seq) - 1):
            counts[(seq[i], seq[i + 1])] += 1
    return counts


def _merge_pair(tokens: list[list[bytes]], pair: tuple[bytes, bytes], new: bytes) -> list[list[bytes]]:
    merged: list[list[bytes]] = []
    for seq in tokens:
        i = 0
        out: list[bytes] = []
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                out.append(new)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        merged.append(out)
    return merged


class LocalBPE:
    """BPE entraine sur un seul corpus texte."""

    def __init__(self, merges: list[tuple[bytes, bytes]], vocab: dict[bytes, int]) -> None:
        self.merges = merges
        self.vocab = vocab
        self._byte_to_id = vocab
        self._id_to_bytes = {i: b for b, i in vocab.items()}

    @property
    def n_vocab(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        tokens = _byte_tokens(text)
        for a, b in self.merges:
            new = a + b
            i = 0
            out: list[bytes] = []
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    out.append(new)
                    i += 2
                else:
                    out.append(tokens[i])
                    i += 1
            tokens = out
        return [self._byte_to_id[t] for t in tokens]

    def decode(self, ids: list[int]) -> str:
        chunks = [self._id_to_bytes[i] for i in ids]
        return b"".join(chunks).decode("utf-8", errors="replace")

    def save(self, path: Path) -> None:
        data = {
            "merges": [[a.decode("latin1"), b.decode("latin1")] for a, b in self.merges],
            "vocab": {k.decode("latin1"): v for k, v in self.vocab.items()},
        }
        path.write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> LocalBPE:
        data = json.loads(path.read_text(encoding="utf-8"))
        merges = [
            (a.encode("latin1"), b.encode("latin1")) for a, b in data["merges"]
        ]
        vocab = {k.encode("latin1"): v for k, v in data["vocab"].items()}
        return cls(merges, vocab)


def train_bpe(
    text: str,
    vocab_size: int = 512,
    *,
    min_pair_freq: int = 2,
) -> LocalBPE:
    """Entraine un BPE sur `text` jusqu'a `vocab_size` entrees (bytes + merges)."""
    # Tokenise en lignes pour compter les paires sur des sequences realistes
    lines = re.split(r"\n+", text)
    token_seqs = [_byte_tokens(line) for line in lines if line.strip()]

    vocab: dict[bytes, int] = {}
    for b in range(256):
        vocab[bytes([b])] = b

    merges: list[tuple[bytes, bytes]] = []
    next_id = 256

    while len(vocab) < vocab_size:
        pairs = _pair_counts(token_seqs)
        if not pairs:
            break
        best_pair, freq = pairs.most_common(1)[0]
        if freq < min_pair_freq:
            break
        new_token = best_pair[0] + best_pair[1]
        if new_token in vocab:
            merges.append(best_pair)
            token_seqs = _merge_pair(token_seqs, best_pair, new_token)
            continue
        vocab[new_token] = next_id
        next_id += 1
        merges.append(best_pair)
        token_seqs = _merge_pair(token_seqs, best_pair, new_token)

    return LocalBPE(merges, vocab)


def train_or_load_bpe(corpus_path: Path, cache_path: Path, vocab_size: int = 512) -> LocalBPE:
    if cache_path.exists():
        return LocalBPE.load(cache_path)
    text = corpus_path.read_text(encoding="utf-8")
    bpe = train_bpe(text, vocab_size=vocab_size)
    bpe.save(cache_path)
    return bpe
