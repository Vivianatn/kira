# Projet Kira : Assistant IA agentique — Spécifications & instructions

> Document de référence à placer à la racine du projet. Il décrit le projet, la
> répartition du travail entre **Cursor** et **Claude Code**, et les premières tâches
> concrètes à confier à chacun. Les outils peuvent lire ce fichier pour se cadrer.

---

## 1. Objectif du projet

Construire **Kira**, un **assistant virtuel agentique**, qui tourne sur PC (Windows). Kira
doit, à terme :

- dialoguer (la conversation est **une fonctionnalité parmi d'autres**, pas la finalité) ;
- **agir sur l'ordinateur** : lancer des programmes, lire/écrire des fichiers, exécuter
  du code, chercher sur le web, appeler des API ;
- **mémoriser** ses interactions et s'améliorer dans un cadre sûr.

La théorie complète est documentée dans Notion (architecture transformer, pipeline
d'entraînement, boucle ReAct, MCP, sécurité, auto-amélioration). Ce fichier est le
plan d'action côté code.

---

## 2. Contraintes matérielles (importantes)

- **PC Windows**, GPU **NVIDIA MX350 (2 Go VRAM)**, **16 Go de RAM**.
- Le GPU est trop petit pour entraîner/faire tourner un modèle 7B en local.
- **Décision** : le moteur de raisonnement (le « cerveau ») passe par une **API**.
  Toute la couche agentique tourne en local sur le PC.
- Option secondaire : petit modèle (3B quantifié) sur **CPU via Ollama** pour
  expérimenter en local (lent mais fonctionnel ; fermer les apps lourdes car ~5 Go
  de RAM libres seulement).

---

## 3. Architecture cible (rappel)

```
Utilisateur (texte)
      │
   Perception (parsing entrées, gestion contexte)
      │
   Moteur de raisonnement (LLM via API)  ←→  Mémoire (court terme + RAG long terme)
      │   boucle ReAct : Thought → Action → Observation
   Couche d'enforcement (sécurité : politique + allowlist + validation humaine)
      │
   Outils : web | fichiers | lancer programmes | exécution code (sandbox) | GUI
```

Principe de sécurité non négociable : **la couche d'enforcement est en place AVANT**
toute capacité d'exécution de code ou de lancement de programme.

---

## 4. Répartition du travail : Cursor vs Claude Code

| Type de tâche | Outil conseillé | Pourquoi |
|---|---|---|
| Apprendre la mécanique (mini-GPT) | **Cursor** | Édition ligne à ligne, tu vois et touches chaque rouage |
| Échafaudage multi-fichiers (agent, sécurité, MCP) | **Claude Code** | Autonomie sur tâches multi-fichiers cohérentes |
| Débogage de précision | **Cursor** | Edits inline rapides, contrôle fin |
| Tâches déléguées en autonomie | **Claude Code** | Exécute commandes, itère seul |

### Règles de coexistence
- **Commit Git fréquent** : source de vérité partagée + filet de sécurité (rollback).
- Ne pas lancer les deux outils en écriture sur le **même fichier** au même moment.
- Activer le **sandboxing de Claude Code** (Bubblewrap/Seatbelt, désactivé par défaut)
  avant de lui laisser exécuter du code généré.

---

## 5. Structure de projet proposée

```
kira/
├── PROJET_KIRA.md               # ce fichier
├── .env                        # clés API (jamais commité ; voir .gitignore)
├── .gitignore
├── requirements.txt
├── learn/                      # phase d'apprentissage (Cursor)
│   └── minigpt.py              # mini-GPT from scratch
├── kira/                       # le cœur de Kira (Claude Code)
│   ├── __init__.py
│   ├── engine.py               # moteur : appel API / modèle local (interchangeable)
│   ├── agent.py                # boucle ReAct
│   ├── memory.py               # mémoire court terme + RAG
│   ├── security.py             # couche d'enforcement (politique, allowlist)
│   └── tools/                  # outils
│       ├── __init__.py
│       ├── web.py              # recherche web
│       ├── files.py            # lecture/écriture fichiers
│       └── system.py           # lancer programmes (subprocess + allowlist)
├── policy.yaml                 # politique de sécurité (outils/chemins/commandes autorisés)
└── tests/                      # suite de tests (pytest) = future fonction de fitness
```

---

## 6. Feuille de route en phases

1. **Mini-GPT** (Cursor) — comprendre la mécanique. Tourne sur le PC malgré le petit GPU.
2. **Boucle ReAct + moteur API** (Claude Code) — le vrai Kira commence ici.
3. **Outils système + sécurité** (Claude Code) — lancer programmes, exécuter code en
   sandbox. **Sécurité d'abord.**
4. **Mémoire** (Claude Code) — base vectorielle + RAG (auto-amélioration niveau A).
5. **MCP & extensibilité** (Claude Code) — l'agent crée ses propres outils (niveau B).
6. **Auto-amélioration encadrée** (Claude Code) — seulement si 1-5 solides (niveau C).

---

## 7. PREMIÈRE TÂCHE — pour Cursor

**Objectif : implémenter un mini-GPT pédagogique dans `learn/minigpt.py`.**

Instructions à donner à Cursor :

> Crée un mini-GPT (transformer decoder-only) from scratch en PyTorch, dans
> `learn/minigpt.py`. Objectif pédagogique : je veux comprendre chaque composant.
> Exigences :
> - Tokenisation niveau caractère (aucune dépendance externe hors PyTorch).
> - Lecture d'un fichier `learn/input.txt` comme corpus d'entraînement.
> - Implémente explicitement : embeddings de tokens + positionnels, self-attention
>   multi-têtes avec masque causal, blocs transformer (attention + feed-forward +
>   connexions résiduelles + LayerNorm), une tête de sortie.
> - Boucle d'entraînement avec AdamW et affichage de la perte train/val.
> - Fonction `generate()` auto-régressive.
> - Hyperparamètres en haut du fichier, petits par défaut (doit tourner sur CPU ou
>   GPU 2 Go) : n_embd=192, n_head=6, n_layer=6, block_size=128, batch_size=16.
> - Commente chaque section pour que je comprenne le rôle de chaque rouage.
> - Sauvegarde le modèle entraîné dans `learn/minigpt.pt`.

Ensuite, travaille le fichier ligne à ligne avec Cursor pour comprendre l'attention
et le masque causal — c'est le cœur conceptuel.

---

## 8. PREMIÈRE TÂCHE — pour Claude Code

**Objectif : échafauder le squelette de Kira avec un moteur API et
une boucle ReAct minimale, sécurité incluse dès le départ.**

Instructions à donner à Claude Code :

> Lis `PROJET_KIRA.md` à la racine. Implémente le squelette de l'assistant
> agentique selon la structure de la section 5. Pour cette première itération :
>
> 1. **`kira/engine.py`** — une classe `Engine` avec une méthode `think(messages,
>    tools)` qui appelle une API LLM (lis la clé depuis `.env`). Conçois-la pour que le
>    backend soit **interchangeable** (API maintenant, modèle local Ollama plus tard)
>    via un paramètre de configuration.
> 2. **`kira/security.py`** — une couche d'enforcement qui charge `policy.yaml`
>    et expose `is_allowed(action)` et `requires_human_approval(action)`. Liste
>    blanche d'outils et de commandes. **À implémenter AVANT tout outil système.**
> 3. **`kira/tools/`** — pour l'instant deux outils sûrs uniquement :
>    `web.py` (recherche web) et `files.py` (lecture seule d'un fichier dans un
>    répertoire autorisé par la politique). PAS encore de lancement de programme ni
>    d'exécution de code.
> 4. **`kira/agent.py`** — la boucle ReAct (Thought → Action → Observation →
>    … → Answer) qui orchestre engine + tools, en passant chaque action par la
>    couche de sécurité avant exécution. Limite de pas configurable.
> 5. **`policy.yaml`** — fichier de politique de départ : outils autorisés (`web`,
>    `files`), répertoire de fichiers accessible (un dossier `workspace/` du projet),
>    liste d'actions sensibles vides pour l'instant.
> 6. **`tests/`** — quelques tests pytest : la couche de sécurité bloque bien une
>    action hors politique ; la boucle ReAct s'arrête à la limite de pas.
> 7. **`requirements.txt`**, **`.gitignore`** (ignore `.env`, `*.pt`, `__pycache__`,
>    `workspace/`), et un **`.env.example`** documentant les variables attendues.
>
> Ne mets PAS encore d'exécution de code arbitraire ni de lancement de programme :
> ces capacités viendront en phase 3, une fois la sécurité validée. Commit Git après
> chaque module fonctionnel.

---

## 8 bis. Sécurité — rappels permanents (toutes phases)

- Exécution de code/commandes → **sandbox Docker éphémère**, jamais sur l'hôte directement.
- **Moindre privilège** : l'agent n'a accès qu'aux outils strictement nécessaires.
- **Human-in-the-loop** pour les actions sensibles (suppression, envoi, modif système).
- **Allowlists** (pas de denylists, contournables).
- **Journal append-only** de toutes les actions.
- Activer le sandboxing de Claude Code avant de lui faire exécuter du code généré.

---

## 9. Configuration initiale (à faire une fois)

```bash
# Créer le projet et l'environnement
mkdir kira && cd kira
python -m venv venv
venv\Scripts\activate            # Windows
pip install torch                # pour le mini-GPT (CPU ou CUDA)

# Initialiser Git (indispensable : rollback, source de vérité partagée)
git init
git add .
git commit -m "Initial project scaffold"
```

Créer un fichier `.env` (non commité) avec la clé API du fournisseur choisi pour le moteur.

---

## 10. Ordre recommandé pour démarrer

1. Mettre en place la structure + Git + `.env` (section 9).
2. **Cursor** → mini-GPT (section 7), pour comprendre. En parallèle possible.
3. **Claude Code** → squelette agentique (section 8).
4. Tester la boucle ReAct avec les 2 outils sûrs.
5. Seulement ensuite : phase 3 (outils système + sandbox), avec la sécurité d'abord.

> Rappel : ne jamais activer l'exécution de code ou le lancement de programme avant
> que la couche de sécurité (section 8, point 2) soit en place et testée.
