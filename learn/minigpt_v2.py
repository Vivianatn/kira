"""
Mini-GPT Phase 2 — BPE local, RoPE, RMSNorm, SwiGLU.

Evolue le mini-GPT v1 (learn/minigpt.py) avec les briques des LLM modernes :
  - BPE entraine localement sur input.txt (learn/bpe_local.py, zero reseau)
  - RoPE : encodage positionnel par rotation Q/K
  - RMSNorm : normalisation legere
  - SwiGLU : FFN gated (Llama / Mistral)

Usage :
    python learn/minigpt_v2.py
    python learn/minigpt_v2.py --generate --prompt "Kira"
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from learn.bpe_local import LocalBPE, train_or_load_bpe

# ---------------------------------------------------------------------------
# Hyperparamètres (identiques a v1 — petits, CPU-friendly)
# ---------------------------------------------------------------------------
N_EMBD = 192
N_HEAD = 6
N_LAYER = 6
BLOCK_SIZE = 64           # plus court que v1 — corpus pedagogique petit
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
MAX_ITERS = 2000
EVAL_INTERVAL = 200
EVAL_ITERS = 50
DROPOUT = 0.1
ROPE_THETA = 10000.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / "input.txt"
MODEL_PATH = SCRIPT_DIR / "minigpt_v2.pt"
BPE_CACHE = SCRIPT_DIR / "bpe_vocab.json"
BPE_VOCAB_SIZE = 512


# ---------------------------------------------------------------------------
# Tokenisation BPE locale — sous-mots appris sur le corpus (pas de telechargement)
# ---------------------------------------------------------------------------

def get_bpe() -> LocalBPE:
    return train_or_load_bpe(INPUT_PATH, BPE_CACHE, vocab_size=BPE_VOCAB_SIZE)


def encode_text(bpe: LocalBPE, text: str) -> list[int]:
    return bpe.encode(text)


def decode_tokens(bpe: LocalBPE, tokens: list[int]) -> str:
    return bpe.decode(tokens)


# ---------------------------------------------------------------------------
# RMSNorm — normalise par la racine de la moyenne des carres (pas mean/variance)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(rms + self.eps) * self.weight


# ---------------------------------------------------------------------------
# RoPE — Rotary Position Embedding (rotation de Q et K selon la position)
# ---------------------------------------------------------------------------

def build_rope_cache(
    seq_len: int, head_dim: int, theta: float = ROPE_THETA, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-calcule cos/sin pour chaque position et chaque paire de dimensions."""
    assert head_dim % 2 == 0
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device)
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applique la rotation sur Q et K : (B, n_head, T, head_dim)."""
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


# ---------------------------------------------------------------------------
# Attention causale multi-tetes + RoPE (pas d'embedding positionnel separe)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )
        cos, sin = build_rope_cache(block_size, self.head_dim, ROPE_THETA, "cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        cos = self.rope_cos[:T].to(x.device)
        sin = self.rope_sin[:T].to(x.device)
        q, k = apply_rope(q, k, cos, sin)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


# ---------------------------------------------------------------------------
# SwiGLU FFN — gate * silu(up) puis projection (Llama / Mistral)
# ---------------------------------------------------------------------------

class SwiGLUFeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        hidden = int(8 * n_embd / 3)
        hidden = ((hidden + 63) // 64) * 64  # multiple de 64 (convention Llama)
        self.w_gate = nn.Linear(n_embd, hidden, bias=False)
        self.w_up = nn.Linear(n_embd, hidden, bias=False)
        self.w_down = nn.Linear(hidden, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


# ---------------------------------------------------------------------------
# Bloc Transformer — pre-norm RMSNorm + attention RoPE + SwiGLU
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = RMSNorm(n_embd)
        self.ffn = SwiGLUFeedForward(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Modele GPT v2 — token embedding seul (RoPE gere la position)
# ---------------------------------------------------------------------------

class MiniGPTv2(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embd: int = N_EMBD,
        n_head: int = N_HEAD,
        n_layer: int = N_LAYER,
        block_size: int = BLOCK_SIZE,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[TransformerBlock(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = RMSNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.size()
        assert T <= self.block_size
        x = self.drop(self.token_emb(idx))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx


# ---------------------------------------------------------------------------
# Donnees et entrainement
# ---------------------------------------------------------------------------

def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= block_size + 1:
        raise ValueError(
            f"Sequence trop courte ({len(data)} tokens) pour block_size={block_size}"
        )
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: MiniGPTv2,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    block_size: int,
    batch_size: int,
    eval_iters: int,
    device: str,
) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}
    for split, tensor in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(tensor, block_size, batch_size, device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(
    max_iters: int = MAX_ITERS,
    eval_interval: int = EVAL_INTERVAL,
    eval_iters: int = EVAL_ITERS,
) -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Corpus introuvable : {INPUT_PATH}")

    enc = get_bpe()
    text = INPUT_PATH.read_text(encoding="utf-8")
    tokens = encode_text(enc, text)
    vocab_size = enc.n_vocab

    data = torch.tensor(tokens, dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    print(
        f"Phase 2 | BPE local ({BPE_VOCAB_SIZE} max) | {len(tokens)} tokens | vocab {vocab_size}",
        flush=True,
    )
    print(f"Device : {DEVICE}", flush=True)

    model = MiniGPTv2(vocab_size).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for step in range(max_iters):
        if step % eval_interval == 0 or step == max_iters - 1:
            losses = estimate_loss(
                model, train_data, val_data, BLOCK_SIZE, BATCH_SIZE, eval_iters, DEVICE
            )
            print(
                f"step {step:4d} | train {losses['train']:.4f} | val {losses['val']:.4f}",
                flush=True,
            )
        xb, yb = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    checkpoint = {
        "model_state": model.state_dict(),
        "bpe_cache": str(BPE_CACHE),
        "config": {
            "vocab_size": vocab_size,
            "n_embd": N_EMBD,
            "n_head": N_HEAD,
            "n_layer": N_LAYER,
            "block_size": BLOCK_SIZE,
        },
    }
    torch.save(checkpoint, MODEL_PATH)
    print(f"Modele sauvegarde : {MODEL_PATH}", flush=True)


def run_generate(
    prompt: str = "Kira",
    max_new_tokens: int = 200,
    temperature: float = 0.8,
) -> None:
    if not MODEL_PATH.exists():
        print("Aucun modele v2. Lancez l'entrainement d'abord.")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    bpe_path = Path(checkpoint.get("bpe_cache", BPE_CACHE))
    if not bpe_path.is_absolute():
        bpe_path = SCRIPT_DIR / bpe_path.name
    enc = LocalBPE.load(bpe_path)
    model = MiniGPTv2(**checkpoint["config"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    start = encode_text(enc, prompt)
    if not start:
        start = [0]
    idx = torch.tensor([start], dtype=torch.long, device=DEVICE)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature)
    print(decode_tokens(enc, out[0].tolist()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini-GPT phase 2 (BPE, RoPE, RMSNorm, SwiGLU)")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--prompt", type=str, default="Kira")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-iters", type=int, default=MAX_ITERS)
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    parser.add_argument("--eval-iters", type=int, default=EVAL_ITERS)
    args = parser.parse_args()

    if args.generate:
        run_generate(args.prompt, args.max_tokens, args.temperature)
    else:
        train(args.max_iters, args.eval_interval, args.eval_iters)


if __name__ == "__main__":
    main()
