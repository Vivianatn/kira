# Pont MCP « kira-bridge »

Canal de communication **live** entre **Cursor** et **Claude Code**, via un
serveur MCP partagé. Quand les deux outils sont actifs, ils échangent des
messages instantanément (sans passer par git).

## Comment ça marche

- `mcp_bridge/server.py` est un serveur MCP (stdio) exposant 3 outils.
- Chaque outil IA (Cursor, Claude Code) lance **sa propre instance** du serveur,
  mais toutes pointent vers le **même store local** `mcp_bridge/.data/`
  (gitignoré). C'est ce store partagé qui fait le pont.
- Config : `.mcp.json` (Claude Code) et `.cursor/mcp.json` (Cursor), déjà en place.

## Outils exposés

| Outil | Rôle |
|---|---|
| `post_message(sender, body, to="all")` | poster un message (`sender`/`to` = `claude-code`, `cursor` ou `all`) |
| `read_messages(reader, only_unread=True)` | lire les messages adressés à `reader` (et les marquer lus) |
| `peek_messages(reader)` | lire **sans** marquer comme lus |

## Limite à connaître

Ni Cursor ni Claude Code ne tournent en arrière-plan : chacun ne **lit** le canal
que lorsqu'il est **lancé par l'humain**. Le pont rend le transport instantané,
mais ne remplace pas le démarrage de l'autre outil. Pour une trace persistante
et hors-ligne, le journal `COLLABORATION.md` §8 reste le canal de secours.

## Test

```bash
D:\kira\.python\python.exe -m pytest mcp_bridge/test_bridge.py -q
```

> Activation : les serveurs MCP sont chargés **au démarrage** de chaque outil
> (et peuvent demander une approbation la première fois). Après un `git pull`,
> redémarre Cursor / Claude Code pour que le pont soit disponible.
