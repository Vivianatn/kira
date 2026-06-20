"""
Démo pédagogique — attention multi-têtes + masque causal (mini-GPT).

Objectif : voir les tenseurs et les matrices, pas seulement lire le code.
Lance :
    D:\\kira\\.python\\python.exe learn/demo_attention.py

Relire ensuite CausalSelfAttention dans learn/minigpt.py ligne par ligne.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# Petite séquence fictive : 4 positions, embedding dim = 8, 2 têtes
B, T, C = 1, 4, 8
N_HEAD = 2
HEAD_DIM = C // N_HEAD

# Entrée « x » : 4 tokens, chacun un vecteur de 8 nombres
x = torch.randn(B, T, C)
print("=== Entrée x (1 batch, T=4 positions, C=8 dimensions) ===")
print(x.shape, "\n")

# --- Étape 1 : projections Q, K, V -----------------------------------------
# En pratique : une couche linéaire produit 3*C puis on coupe en trois.
W = torch.randn(C, 3 * C) / math.sqrt(C)
qkv = x @ W
q, k, v = qkv.split(C, dim=2)

q = q.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)  # (B, n_head, T, head_dim)
k = k.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
v = v.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)

print("=== Q, K, V (tête 0 seulement, pour lisibilité) ===")
print("Query  tete0 :", q[0, 0].shape, "  — ce que je cherche a chaque position")
print("Key    tete0 :", k[0, 0].shape, "  — ce que j'offre a chaque position")
print("Value  tete0 :", v[0, 0].shape, "  — contenu a agreger si la cle matche\n")

# --- Étape 2 : scores d'attention (compatibilité query/key) ------------------
# att[b, h, i, j] = à quel point la position i regarde la position j
scores = (q @ k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
scores_head0 = scores[0, 0]

print("=== Scores bruts (tete 0) : matrice T x T — lignes=qui regarde, colonnes=cible ===")
print(scores_head0.round(decimals=2))
print("  -> Chaque LIGNE = une position query qui score toutes les keys\n")

# --- Étape 3 : masque causal -----------------------------------------------
# Sans masque, position 0 pourrait voir position 3 (le futur) → pas auto-régressif.
mask = torch.tril(torch.ones(T, T))
print("=== Masque causal (1 = visible, 0 = futur interdit) ===")
print(mask.int())
print()

scores_masked = scores_head0.masked_fill(mask == 0, float("-inf"))
print("=== Scores APRÈS masque (-inf sur le triangle supérieur) ===")
print(scores_masked.round(decimals=2))
print("  -> Le futur devient -inf ; softmax donnera poids 0 sur ces cases\n")

# --- Étape 4 : softmax → poids d'attention ---------------------------------
weights = F.softmax(scores_masked, dim=-1)
print("=== Poids d'attention (softmax sur chaque ligne, somme = 1) ===")
print(weights.round(decimals=3))
print("  -> Ligne i : distribution sur les positions <= i uniquement\n")

# --- Étape 5 : agrégation des Values ---------------------------------------
out_head0 = weights @ v[0, 0]
print("=== Sortie tête 0 (weights @ V) : nouvelle représentation par position ===")
print(out_head0.shape)
print(out_head0.round(decimals=3))
print()

# --- Étape 6 : pourquoi plusieurs têtes ? ------------------------------------
print("=== Multi-têtes : 6 têtes en parallèle dans minigpt.py ===")
print("Chaque tête apprend un type de relation différent (syntaxe, répétition, etc.).")
print("On concatène les sorties de toutes les têtes puis on repasse par une linéaire (c_proj).")
print()

# --- Lien avec minigpt.py ----------------------------------------------------
print("=== Correspondance avec learn/minigpt.py ===")
print("  self.c_attn(x)           -> projections Q,K,V")
print("  q @ k.transpose(-2,-1)   -> scores (etape 2)")
print("  self.bias + masked_fill    -> masque causal (etape 3)")
print("  F.softmax(att, dim=-1)     -> poids (etape 4)")
print("  att @ v                    -> agregation (etape 5)")
print("  self.c_proj(y)             -> mixage des tetes + projection residuelle")
