"""Interface web de chat pour Kira — serveur léger (bibliothèque standard).

Sert une page de chat et expose un endpoint POST /api/chat qui fait tourner la
boucle ReAct de l'`Agent` et renvoie la réponse + les étapes (outils utilisés).

Choix techniques :
- `http.server` de la stdlib (zéro dépendance lourde type Gradio) : adapté à un
  usage local mono-utilisateur, léger pour une machine à RAM limitée.
- L'agent est construit UNE fois au démarrage et réutilisé.
- Validation humaine : handler par défaut (refuse les actions sensibles) — il n'y
  en a aucune pour l'instant. Une vraie UI d'approbation viendra avec la phase 3.

Lancement :
    D:\\kira\\.python\\python.exe webui/server.py
puis ouvrir http://127.0.0.1:7860
"""

from __future__ import annotations

import json
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Rend `kira` importable (Python embeddable : sys.path verrouillé).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kira.agent import Agent  # noqa: E402
from kira.engine import Engine  # noqa: E402
from kira.security import EnforcementLayer  # noqa: E402

HOST = "127.0.0.1"
PORT = 7860
INDEX_HTML = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")

# Construction unique de l'agent (moteur lu depuis .env : Ollama, Anthropic...).
_security = EnforcementLayer(policy_path=PROJECT_ROOT / "policy.yaml")
_engine = Engine.from_config()
_agent = Agent(_engine, _security)


class KiraHandler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: str, ctype: str = "application/json") -> None:
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 (API imposée par http.server)
        if self.path in ("/", "/index.html"):
            self._send(200, INDEX_HTML, "text/html")
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/chat":
            self._send(404, json.dumps({"error": "not found"}))
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            message = str(payload.get("message", "")).strip()
            if not message:
                self._send(400, json.dumps({"error": "message vide"}))
                return
            result = _agent.run(message)
            steps = [
                {"thought": s.thought, "actions": s.actions} for s in result.steps
            ]
            self._send(
                200,
                json.dumps(
                    {
                        "answer": result.answer,
                        "steps": steps,
                        "stopped": result.stopped_reason,
                    },
                    ensure_ascii=False,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - on renvoie l'erreur au client
            self._send(500, json.dumps({"error": str(exc)}, ensure_ascii=False))

    def log_message(self, *args) -> None:  # silence les logs HTTP par défaut
        pass


def main() -> None:
    httpd = ThreadingHTTPServer((HOST, PORT), KiraHandler)
    url = f"http://{HOST}:{PORT}"
    print(f"Kira — interface web sur {url}")
    print(f"Moteur : backend={_engine.backend.__class__.__name__}")
    print("Ctrl-C pour arrêter.")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt.")
        httpd.server_close()


if __name__ == "__main__":
    main()
