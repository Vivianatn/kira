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
- Phases 1–2 faites (squelette agentique, 18 tests verts). Prochaines : valider
  le backend Anthropic, puis phase 3 (outils système en sandbox).
