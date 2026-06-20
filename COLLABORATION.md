# Fiche directrice — Collaboration Cursor × Claude Code

> Document partagé (committé) qui répartit le travail entre les deux outils IA
> du projet Kira et fixe les règles pour éviter les conflits. **À lire avant
> toute session.** Source de vérité = le dépôt git (`main`).

---

## 0. Principe fondateur

Un seul humain pilote **deux outils IA** sur le même dépôt :

- **Cursor** → travail pédagogique, ligne à ligne, sur le mini-GPT (`learn/`).
- **Claude Code** → échafaudage multi-fichiers autonome sur le cœur agentique (`kira/`).

Règle d'or anti-conflit : **chaque dossier a UN seul propriétaire**. On ne lance
jamais les deux outils en écriture sur le même fichier au même moment.

---

## 1. Répartition par dossier (ownership)

| Chemin | Propriétaire | Statut | L'autre peut… |
|---|---|---|---|
| `learn/` (minigpt.py, input.txt, expériences) | **Cursor** | actif | lire seulement |
| `kira/` (engine, agent, security, memory, tools) | **Claude Code** | actif | lire seulement |
| `tests/` | **Claude Code** | actif | lire seulement |
| `policy.yaml` | **Claude Code** | actif | lire seulement |
| `main.py` | **Claude Code** | actif | lire seulement |
| `requirements.txt` | **Claude Code** | partagé* | proposer un ajout |
| `.gitignore`, `.env.example` | **Claude Code** | partagé* | proposer un ajout |
| `PROJET_KIRA.md` | référence | gelé | lire seulement |
| `COLLABORATION.md` (ce fichier) | **Claude Code** | partagé* | proposer un ajout |

\* *Partagé* = modifiable par les deux, mais **uniquement par ajout** (jamais de
réécriture), et toujours juste après un `git pull`, suivi d'un commit immédiat.

> Si une tâche t'oblige à toucher un fichier qui n'est pas à toi : **ne le fais
> pas toi-même**. Note-le, et laisse le propriétaire s'en charger (ou demande à
> l'humain de basculer l'outil).

---

## 2. Protocole git (obligatoire, à chaque session)

Le dépôt est la mémoire partagée entre les deux IA. Discipline :

```bash
# AVANT de commencer à écrire
git pull --rebase            # récupérer le travail de l'autre outil

# APRÈS chaque unité de travail fonctionnelle
git add <tes fichiers>
git commit -m "..."          # commits petits et fréquents
git push
```

Règles :
- **Commits petits et fréquents** : c'est le filet de sécurité (rollback) ET le
  canal de communication entre les deux outils.
- **Pull avant d'écrire, push après** : ne jamais laisser du travail non poussé
  traîner pendant que l'autre outil bosse.
- **Préfixe de commit** pour savoir qui a fait quoi :
  - `learn:` → travail Cursor (ex. `learn: ajoute le masque causal commenté`)
  - `kira:`  → travail Claude Code (ex. `kira: outils système phase 3`)
- En cas de conflit git : c'est qu'on a violé la règle d'ownership. On résout en
  faveur du propriétaire du fichier.

Remote configuré : `git@github.com:Vivianatn/kira.git` (clé SSH en place).

---

## 3. Environnement commun (déjà configuré)

- **Interpréteur Python unique** : `D:\kira\.python\python.exe`
  (package « embeddable » 3.13.14 ; winget et l'installeur MSI échouent sur
  cette machine — voir l'historique). pip y est amorcé.
- Les **deux outils utilisent ce même interpréteur**. Dans Cursor : régler
  l'interpréteur Python du projet sur `D:\kira\.python\python.exe`.
- Lancer quoi que ce soit :
  - Tests : `D:\kira\.python\python.exe -m pytest -q`
  - App   : `D:\kira\.python\python.exe main.py "..."`

---

## 4. Ce que **Claude Code** fait & configure

**Périmètre** : tout le cœur agentique (`kira/`), les tests, la sécurité, l'infra.

**Configuration à sa charge :**
- [x] Squelette agentique (engine, agent, security, tools web/files) — *fait*.
- [x] `policy.yaml` + 18 tests pytest verts — *fait*.
- [ ] `.env` réel : copier `.env.example` → `.env`, renseigner `ANTHROPIC_API_KEY`
      (jamais committé). Permet de tester le backend Anthropic.
- [ ] **Avant la phase 3** (exécution de code / lancement de programmes) :
      mettre en place le **bac à sable Docker éphémère** (jamais d'exécution sur
      l'hôte). Installer Docker Desktop, et n'autoriser l'outil `system` dans
      `policy.yaml` qu'une fois la sandbox testée. **Sécurité d'abord.**
- [ ] Activer le sandboxing de Claude Code lui-même avant de lui faire exécuter
      du code généré (désactivé par défaut).

**Tâches à venir (par phase) :** voir §6.

---

## 5. Ce que **Cursor** fait & configure

**Périmètre** : tout `learn/` (apprentissage de la mécanique des transformers).

**Configuration à sa charge :**
- [ ] Régler l'interpréteur du projet sur `D:\kira\.python\python.exe`.
- [ ] Installer PyTorch dans ce runtime :
      `D:\kira\.python\python.exe -m pip install torch`
      (CPU par défaut ; le GPU MX350 2 Go suffit pour le mini-GPT avec les
      petits hyperparamètres déjà prévus).
- [ ] Vérifier que `learn/minigpt.py` tourne et entraîne sur `learn/input.txt`.

**Tâches Cursor (rappel section 7 du plan) :**
- Comprendre **ligne à ligne** l'attention et le masque causal (le cœur).
- Étendre ensuite (phase 2 conceptuelle) : BPE (tiktoken), RoPE, RMSNorm,
  SwiGLU, puis éventuellement MoE — **tout ça reste dans `learn/`**.

> Cursor ne touche **jamais** à `kira/`. Si le mini-GPT doit un jour alimenter
> Kira, ce sera via une interface définie côté `kira/` par Claude Code.

---

## 6. Feuille de route — qui fait quoi

| Phase | Contenu | Propriétaire |
|---|---|---|
| 1. Mini-GPT | transformer from scratch, pédagogie | **Cursor** (`learn/`) |
| 2. Boucle ReAct + moteur API | squelette agentique | **Claude Code** ✅ *fait* |
| 3. Outils système + sécurité | lancer programmes, exéc. code en **sandbox Docker** | **Claude Code** |
| 4. Mémoire | base vectorielle + RAG | **Claude Code** (`kira/memory.py`) |
| 5. MCP & extensibilité | l'agent crée ses outils | **Claude Code** |
| 6. Auto-amélioration encadrée | seulement si 1–5 solides | **Claude Code** |
| (transverse) Concepts LLM avancés | BPE, RoPE, MoE… | **Cursor** (`learn/`) |

Garde-fou permanent : **aucune exécution de code ni lancement de programme**
n'est activé tant que la couche de sécurité de la phase 3 (sandbox + allowlist
+ validation humaine) n'est pas en place **et testée**.

---

## 7. Démarrage immédiat (prochaine session)

**Cursor :**
1. Interpréteur = `D:\kira\.python\python.exe`, installer `torch`.
2. Lancer `learn/minigpt.py`, travailler l'attention ligne à ligne.
3. `git commit -m "learn: ..."` + `git push` après chaque étape comprise.

**Claude Code :**
1. Créer `.env` (clé API) et valider le backend Anthropic via `main.py`.
2. Préparer la phase 3 : sandbox Docker + outil `system` derrière la sécurité.
3. `git commit -m "kira: ..."` + `git push` après chaque module.

---

*Maj : modifier ce fichier uniquement par ajout, après un `git pull`, puis
commit immédiat. En cas de doute sur « à qui appartient ce fichier », se référer
au tableau §1.*
