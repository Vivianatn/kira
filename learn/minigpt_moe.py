"""
Mini-GPT Phase 3 (optionnelle) — Mixture of Experts (MoE) sur la base v2.

Remplace le FFN dense (SwiGLU) par plusieurs « experts » + un routeur :
  - Le routeur choisit top-k experts pour chaque token
  - Capacite totale (tous les experts) >> cout de calcul (top-k actifs)
  - Loss d'equilibrage pour eviter qu'un seul expert soit toujours choisi

Reutilise : BPE local, RoPE, RMSNorm, attention (learn/minigpt_v2.py).

Usage :
    python learn/minigpt_moe.py
    python learn/minigpt_moe.py --generate --prompt "Kira"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from learn.minigpt_v2 import (
    BATCH_SIZE,
    BLOCK_SIZE,
    BPE_CACHE,
    BPE_VOCAB_SIZE,
    DEVICE,
    EVAL_INTERVAL,
    EVAL_ITERS,
    INPUT_PATH,
    LEARNING_RATE,
    MAX_ITERS,
    N_EMBD,
    N_HEAD,
    N_LAYER,
    RMSNorm,
    CausalSelfAttention,
    SwiGLUFeedForward,
    decode_tokens,
    encode_text,
    get_bpe,
    get_batch,
)

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "minigpt_moe.pt"

# MoE — petit pour tourner sur CPU
N_EXPERT = 4
TOP_K = 2
MOE_AUX_COEFF = 0.01  # penalite si les experts sont mal equilibres


# ---------------------------------------------------------------------------
# Couche MoE — routeur + banque d'experts SwiGLU
# ---------------------------------------------------------------------------

class MoELayer(nn.Module):
    """
    Pour chaque token, le routeur (gate) choisit top_k experts.
    Sortie = somme ponderee des experts selectionnes.

    forward renvoie (y, aux_loss) — aux_loss guide l'entrainement vers un usage
    equilibre des experts (sinon le routeur ne prend qu'un expert « favori »).
    """

    def __init__(
        self,
        n_embd: int,
        n_expert: int,
        top_k: int,
        dropout: float,
    ) -> None:
        super().__init__()
        assert 1 <= top_k <= n_expert
        self.n_expert = n_expert
        self.top_k = top_k
        self.gate = nn.Linear(n_embd, n_expert, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLUFeedForward(n_embd, dropout) for _ in range(n_expert)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        logits = self.gate(x)  # (B, T, n_expert)
        router_probs = F.softmax(logits, dim=-1)

        # Loss d'equilibrage (Switch Transformer / load balancing)
        # Encourage une distribution uniforme des tokens vers les experts.
        importance = router_probs.mean(dim=(0, 1))
        aux_loss = self.n_expert * torch.sum(importance * importance)

        x_flat = x.view(-1, C)
        logits_flat = logits.view(-1, self.n_expert)
        weights, indices = torch.topk(logits_flat, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)

        out_flat = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            for e_idx, expert in enumerate(self.experts):
                mask = indices[:, k] == e_idx
                if not mask.any():
                    continue
                out_flat[mask] += weights[mask, k].unsqueeze(-1) * expert(x_flat[mask])

        return out_flat.view(B, T, C), aux_loss


# ---------------------------------------------------------------------------
# Bloc transformer avec MoE au lieu du FFN dense
# ---------------------------------------------------------------------------

class MoETransformerBlock(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float,
        n_expert: int,
        top_k: int,
    ) -> None:
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = RMSNorm(n_embd)
        self.moe = MoELayer(n_embd, n_expert, top_k, dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.ln1(x))
        moe_out, aux = self.moe(self.ln2(x))
        return x + moe_out, aux


# ---------------------------------------------------------------------------
# GPT + MoE
# ---------------------------------------------------------------------------

class MiniGPTMoE(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embd: int = N_EMBD,
        n_head: int = N_HEAD,
        n_layer: int = N_LAYER,
        block_size: int = BLOCK_SIZE,
        dropout: float = 0.1,
        n_expert: int = N_EXPERT,
        top_k: int = TOP_K,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                MoETransformerBlock(
                    n_embd, n_head, block_size, dropout, n_expert, top_k
                )
                for _ in range(n_layer)
            ]
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
        aux_total = torch.tensor(0.0, device=idx.device)
        for block in self.blocks:
            x, aux = block(x)
            aux_total = aux_total + aux
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = ce + MOE_AUX_COEFF * aux_total
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

@torch.no_grad()
def estimate_loss(
    model: MiniGPTMoE,
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
        f"Phase MoE | {N_EXPERT} experts top-{TOP_K} | {len(tokens)} tokens | vocab {vocab_size}",
        flush=True,
    )
    print(f"Device : {DEVICE}", flush=True)

    model = MiniGPTMoE(vocab_size).to(DEVICE)
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
            "n_expert": N_EXPERT,
            "top_k": TOP_K,
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
        print("Aucun modele MoE. Lancez l'entrainement d'abord.")
        return

    from learn.bpe_local import LocalBPE

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    bpe_path = Path(checkpoint.get("bpe_cache", BPE_CACHE))
    if not bpe_path.is_absolute():
        bpe_path = SCRIPT_DIR / bpe_path.name
    enc = LocalBPE.load(bpe_path)
    model = MiniGPTMoE(**checkpoint["config"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    start = encode_text(enc, prompt)
    if not start:
        start = [0]
    idx = torch.tensor([start], dtype=torch.long, device=DEVICE)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature)
    print(decode_tokens(enc, out[0].tolist()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini-GPT MoE (phase optionnelle)")
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
