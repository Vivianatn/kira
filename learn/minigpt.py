"""
Mini-GPT pédagogique — Transformer decoder-only from scratch (PyTorch).

Objectif : comprendre chaque composant d'un LLM (tokenisation, embeddings,
attention causale multi-têtes, blocs transformer, entraînement, génération).

Usage :
    python learn/minigpt.py          # entraîne et sauvegarde learn/minigpt.pt
    python learn/minigpt.py --generate  # charge le modèle et génère du texte
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Hyperparamètres (petits par défaut — tourne sur CPU ou GPU 2 Go)
# ---------------------------------------------------------------------------
N_EMBD = 192          # dimension des embeddings et des vecteurs internes
N_HEAD = 6            # nombre de têtes d'attention (N_EMBD doit être divisible)
N_LAYER = 6           # nombre de blocs transformer empilés
BLOCK_SIZE = 128      # longueur max de séquence (fenêtre de contexte)
BATCH_SIZE = 16       # nombre de séquences par batch
LEARNING_RATE = 3e-4
MAX_ITERS = 2000      # itérations d'entraînement (réduire si trop lent)
EVAL_INTERVAL = 200   # afficher train/val loss tous les N pas
EVAL_ITERS = 50       # batches pour estimer la loss de validation
DROPOUT = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / "input.txt"
MODEL_PATH = SCRIPT_DIR / "minigpt.pt"


# ---------------------------------------------------------------------------
# Tokenisation niveau caractère (zéro dépendance externe)
# Chaque caractère unique du corpus devient un identifiant entier.
# ---------------------------------------------------------------------------

def build_char_tokenizer(text: str) -> tuple[dict[str, int], dict[int, str]]:
    """Construit les tables char → id et id → char depuis le corpus."""
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: dict[str, int]) -> list[int]:
    """Texte → liste d'identifiants entiers."""
    return [stoi[ch] for ch in text]


def decode(tokens: list[int], itos: dict[int, str]) -> str:
    """Liste d'identifiants → texte."""
    return "".join(itos[t] for t in tokens)


# ---------------------------------------------------------------------------
# Self-attention multi-têtes avec masque causal
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    Pour chaque position, calcule à quel point chaque token précédent
    (et lui-même) est pertinent. Le masque causal interdit de voir le futur.

    Démo pas-à-pas : learn/demo_attention.py (matrices T×T, masque, softmax).
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        # Une seule projection linéaire produit Q, K, V (plus efficace)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Masque causal : positions futures masquées avec -inf avant softmax
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T, C) — B sequences en parallele, T tokens, C dimensions d'embedding
        B, T, C = x.size()

        # --- Etape 1 : projections lineaires -> Query, Key, Value ---
        # Une seule matrice W produit [Q|K|V] puis on coupe (efficace en GPU)
        qkv = self.c_attn(x)                    # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=2)           # chacun (B, T, C)

        # --- Etape 2 : multi-tetes — on decoupe C en n_head sous-espaces ---
        # transpose : (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # --- Etape 3 : scores = Q @ K^T / sqrt(d_k) ---
        # att[b,h,i,j] = compatibilite entre position query i et key j
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # --- Etape 4 : masque causal ---
        # self.bias = triangle inferieur ; 0 sur le futur -> -inf avant softmax
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        # --- Etape 5 : softmax -> poids (somme 1 par ligne query) ---
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # --- Etape 6 : melange des Values selon les poids ---
        y = att @ v                           # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # reconcat tetes

        # --- Etape 7 : projection finale + dropout residuel ---
        return self.resid_dropout(self.c_proj(y))


# ---------------------------------------------------------------------------
# Réseau feed-forward (MLP) — « réflexion » après l'attention
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Bloc Transformer : attention + FFN, avec résidu et pre-norm LayerNorm
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    Pre-norm : LayerNorm avant chaque sous-couche, puis connexion résiduelle.
    x → x + Attention(LayerNorm(x)) → x + FFN(LayerNorm(x))
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Modèle GPT complet : embeddings + pile de blocs + tête de sortie
# ---------------------------------------------------------------------------

class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embd: int = N_EMBD,
        n_head: int = N_HEAD,
        n_layer: int = N_LAYER,
        block_size: int = BLOCK_SIZE,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.block_size = block_size

        # Embeddings de tokens : chaque id → vecteur dense
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        # Embeddings positionnels : la position dans la séquence
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[TransformerBlock(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)

        # Tête de sortie : prédit les logits du prochain token (vocab_size classes)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.size()
        assert T <= self.block_size

        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok_emb + pos_emb)
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
        """Génération auto-régressive : prédit un token, l'ajoute, recommence."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx


# ---------------------------------------------------------------------------
# Préparation des données : batches aléatoires de séquences contiguës
# ---------------------------------------------------------------------------

def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Échantillonne des séquences aléatoires et leurs cibles (token suivant)."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: MiniGPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    block_size: int,
    batch_size: int,
    eval_iters: int,
    device: str,
) -> dict[str, float]:
    """Estime la loss moyenne sur train et val."""
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


# ---------------------------------------------------------------------------
# Boucle d'entraînement
# ---------------------------------------------------------------------------

def train(
    max_iters: int = MAX_ITERS,
    eval_interval: int = EVAL_INTERVAL,
    eval_iters: int = EVAL_ITERS,
) -> tuple[MiniGPT, dict[str, int], dict[int, str]]:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Corpus introuvable : {INPUT_PATH}")

    text = INPUT_PATH.read_text(encoding="utf-8")
    stoi, itos = build_char_tokenizer(text)
    vocab_size = len(stoi)

    data = torch.tensor(encode(text, stoi), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    print(f"Corpus : {len(text)} caractères, vocabulaire : {vocab_size} tokens", flush=True)
    print(f"Device : {DEVICE}", flush=True)

    model = MiniGPT(vocab_size).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for step in range(max_iters):
        if step % eval_interval == 0 or step == max_iters - 1:
            losses = estimate_loss(
                model, train_data, val_data, BLOCK_SIZE, BATCH_SIZE, eval_iters, DEVICE
            )
            print(
                f"step {step:4d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}",
                flush=True,
            )

        xb, yb = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    checkpoint = {
        "model_state": model.state_dict(),
        "stoi": stoi,
        "itos": itos,
        "config": {
            "vocab_size": vocab_size,
            "n_embd": N_EMBD,
            "n_head": N_HEAD,
            "n_layer": N_LAYER,
            "block_size": BLOCK_SIZE,
        },
    }
    torch.save(checkpoint, MODEL_PATH)
    print(f"Modèle sauvegardé : {MODEL_PATH}", flush=True)

    return model, stoi, itos


def run_generate(
    prompt: str = "Kira",
    max_new_tokens: int = 200,
    temperature: float = 0.8,
) -> None:
    if not MODEL_PATH.exists():
        print("Aucun modèle trouvé. Lancez d'abord l'entraînement (sans --generate).")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    cfg = checkpoint["config"]
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]

    model = MiniGPT(**cfg).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    start_tokens = [stoi[ch] for ch in prompt if ch in stoi]
    if not start_tokens:
        start_tokens = [0]

    idx = torch.tensor([start_tokens], dtype=torch.long, device=DEVICE)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature)
    generated = decode(out[0].tolist(), itos)
    print(generated)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini-GPT pédagogique")
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Générer du texte avec le modèle sauvegardé",
    )
    parser.add_argument("--prompt", type=str, default="Kira", help="Texte de départ")
    parser.add_argument("--max-tokens", type=int, default=200, help="Tokens à générer")
    parser.add_argument("--temperature", type=float, default=0.8, help="Température")
    parser.add_argument(
        "--max-iters",
        type=int,
        default=MAX_ITERS,
        help="Itérations d'entraînement (défaut: hyperparamètre du fichier)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=EVAL_INTERVAL,
        help="Afficher la loss tous les N pas",
    )
    parser.add_argument(
        "--eval-iters",
        type=int,
        default=EVAL_ITERS,
        help="Batches pour estimer train/val loss",
    )
    args = parser.parse_args()

    if args.generate:
        run_generate(args.prompt, args.max_tokens, args.temperature)
    else:
        train(
            max_iters=args.max_iters,
            eval_interval=args.eval_interval,
            eval_iters=args.eval_iters,
        )


if __name__ == "__main__":
    main()
