"""
Attention causale sur le VRAI mini-GPT entraîne (learn/minigpt.pt).

Montre la matrice d'attention tete 0, couche 0, pour le prompt "Kira"
avec les caracteres en labels — pas des nombres aleatoires.

Usage :
    D:\\kira\\.python\\python.exe learn/demo_attention_real.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from learn.minigpt import MiniGPT, encode

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "minigpt.pt"
PROMPT = "Kira"


def main() -> None:
    if not MODEL_PATH.exists():
        print("Pas de modele : lance d'abord learn/minigpt.py")
        return

    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    stoi, itos = ckpt["stoi"], ckpt["itos"]
    model = MiniGPT(**cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tokens = encode(PROMPT, stoi)
    chars = [itos[t] for t in tokens]
    idx = torch.tensor([tokens], dtype=torch.long)
    T = idx.size(1)
    C = cfg["n_embd"]

    # Embeddings comme dans MiniGPT.forward (sans la pile complete)
    tok_emb = model.token_emb(idx)
    pos_emb = model.pos_emb(torch.arange(T))
    x = model.drop(tok_emb + pos_emb)

    # Premiere couche : pre-norm + attention (couche 0)
    block = model.blocks[0]
    x_norm = block.ln1(x)
    attn = block.attn

    B, _, _ = x_norm.size()
    qkv = attn.c_attn(x_norm)
    q, k, v = qkv.split(C, dim=2)
    n_head = attn.n_head
    head_dim = attn.head_dim
    k = k.view(B, T, n_head, head_dim).transpose(1, 2)
    q = q.view(B, T, n_head, head_dim).transpose(1, 2)

    scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
    mask = attn.bias[:, :, :T, :T]
    scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)

    w0 = weights[0, 0].detach()  # batch 0, tete 0

    print(f"=== Mini-GPT entraine — couche 0, tete 0, prompt « {PROMPT} » ===")
    print(f"Tokens : {chars}\n")

    print("Matrice d'attention (lignes = qui regarde, colonnes = cible) :")
    header = "      " + "  ".join(f"{c:>5}" for c in chars)
    print(header)
    for i, row_char in enumerate(chars):
        row = "  ".join(f"{w0[i, j].item():5.2f}" for j in range(T))
        print(f"  {row_char:>3}  {row}")

    print("\nLecture ligne par ligne :")
    for i, c in enumerate(chars):
        w = w0[i]
        parts = [f"{chars[j]}={w[j].item():.2f}" for j in range(i + 1)]
        print(f"  Position {i} («{c}») regarde : {', '.join(parts)}")

    print("\nMasque causal : chaque ligne ne a que les colonnes <= sa position.")
    print("Le token a la position i ne voit jamais les caracteres suivants.")


if __name__ == "__main__":
    main()
