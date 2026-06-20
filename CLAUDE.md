# CLAUDE.md — projet Kira

Tu es **Claude Code** sur le projet Kira, en collaboration avec **Cursor**.
Lis `COLLABORATION.md` (fiche directrice complète) en début de session.

## Ton périmètre
- Cœur agentique : `kira/` (engine, agent, security, memory, tools), `tests/`,
  `policy.yaml`, `main.py`, l'infra (`requirements.txt`, `.gitignore`, `.env*`).

## Interdits
- **Ne modifie JAMAIS `learn/`** : c'est le périmètre de Cursor (mini-GPT). Lecture seule.
- N'active aucune exécution de code / lancement de programme tant que la couche
  de sécurité de la phase 3 (sandbox Docker + allowlist + validation humaine)
  n'est pas en place **et testée**. Sécurité d'abord.

## Environnement
- Interpréteur unique : `D:\kira\.python\python.exe` (embeddable ; winget/MSI
  échouent sur cette machine — voir mémoire projet).
- Tests : `D:\kira\.python\python.exe -m pytest -q`
- App   : `D:\kira\.python\python.exe main.py "..."`

## Protocole git
- `git pull --rebase` avant d'écrire, commits petits/fréquents préfixés `kira:`,
  `git push` après chaque module. Remote : `git@github.com:Vivianatn/kira.git`.

## Communication avec Cursor
- **Pont MCP `kira-bridge`** (prioritaire) : outils `post_message` /
  `read_messages` / `peek_messages`. En début de session, fais
  `read_messages(reader="claude-code")` ; pour écrire,
  `post_message(sender="claude-code", body="...", to="cursor")`.
- **Sinon / pour une trace durable** : journal `COLLABORATION.md` §8 (append-only).
- Détails du pont : `mcp_bridge/README.md`.

## État actuel
- Phases 1–2 faites (squelette agentique). Backend **Ollama** (qwen2.5:3b, CPU)
  + **interface web** (`webui/`, port 7860).
- **Phase 3 faite** : outil `system` (run_program allowlisté + execute_code en
  sandbox Docker, `kira/sandbox.py`). INERTE par défaut (allowlist vide, fail-closed
  sans Docker), validation humaine obligatoire.
- **Phase 4 démarrée** : `kira/memory.py` (court terme + long terme RAG, embedders
  Ollama/hash). Pas encore branchée dans l'agent.
- Suite : **45 tests verts**. Prochaines pistes : brancher la mémoire dans l'agent,
  pull `nomic-embed-text` pour du vrai RAG sémantique, installer Docker pour activer
  execute_code.
