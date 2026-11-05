"""Dependency-free, localhost-only dashboard for reproducible MOLT experiments."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlparse

from .benchmark import DemoConfig, run_demo
from .lab import LabConfig, oracle, run_text_lab
from .text import TASKS, TextSkill, supported

DASHBOARD = Path(__file__).with_name("dashboard.html")


def handler_factory(root: Path):
    class Handler(BaseHTTPRequestHandler):
        server_version = "KiteWatt/1.0"

        def _send(self, status: int, payload: dict, *, html: bool = False):
            data = (payload["html"].encode("utf-8") if html
                    else json.dumps(payload, separators=(",", ":")).encode("utf-8"))
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8" if html else "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; object-src 'none'; base-uri 'none'")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self._send(200, {"html": DASHBOARD.read_text(encoding="utf-8")}, html=True)
            elif path == "/api/status":
                self._send(200, {"name": "KiteWatt", "version": "1.0.0",
                                 "modes": ["text", "neural"], "paid_apis": False,
                                 "notes": "A local research demo, not a production AI optimizer."})
            else:
                self._send(404, {"error": "Not found"})

        def do_POST(self):
            try:
                size = int(self.headers.get("Content-Length", "0") or "0")
            except ValueError:
                self._send(400, {"error": "Invalid content length"})
                return
            if not 0 < size <= 4096:
                self._send(413, {"error": "Request size must be between 1 and 4096 bytes"})
                return
            try:
                data = json.loads(self.rfile.read(size))
                if type(data) is not dict:
                    raise ValueError("Expected an object")
                path = urlparse(self.path).path
                if path == "/api/text":
                    seed = data.get("seed", 7)
                    if type(seed) is not int or not 0 <= seed <= 10_000:
                        raise ValueError("Seed must be an integer from 0 to 10,000")
                    result = run_text_lab(LabConfig(seed=seed, output=str(root / "text")))
                    self._send(200, result)
                elif path == "/api/neural":
                    op = data.get("operator", "max")
                    steps = data.get("steps", 120)
                    if op not in ("add_mod", "max", "xor", "affine_mod") or type(steps) is not int or not 20 <= steps <= 300:
                        raise ValueError("Choose a supported operation and 20–300 training steps")
                    result = run_demo(DemoConfig(operator=op, steps=steps, benchmark_requests=128,
                                                 output_dir=str(root / "neural")))
                    self._send(200, result)
                elif path == "/api/predict":
                    task, value = data.get("task"), data.get("value")
                    if task not in TASKS or type(value) is not str or len(value) > 180:
                        raise ValueError("Unsupported task or input")
                    path = root / "text" / f"{task}.skill.json"
                    report_path = root / "text" / "report.json"
                    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
                    gate = report["tasks"].get(task, {}).get("cost_gate") if report else None
                    deploy = bool(gate and gate.get("deploy"))
                    if deploy and path.exists() and path.stat().st_size < 200_000:
                        skill = TextSkill.from_dict(json.loads(path.read_text(encoding="utf-8")))
                        if supported(task, value):
                            self._send(200, {"route": "compiled", "answer": skill.predict(value),
                                             "operations": list(skill.program.operations)})
                            return
                    if not supported(task, value):
                        raise ValueError("Input outside supported demo domain")
                    self._send(200, {"route": "reference", "answer": oracle(task, value),
                                     "operations": None})
                else:
                    self._send(404, {"error": "Not found"})
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                self._send(400, {"error": str(exc)})
            except Exception:
                # Never leak filesystem paths or stack traces through the demo API.
                self._send(500, {"error": "The experiment failed; inspect the local terminal"})
                raise

    return Handler


def serve(port: int = 8765, output: str = "runs/dashboard") -> None:
    if not 1024 <= port <= 65535:
        raise ValueError("Port must be between 1024 and 65535")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler_factory(root))
    print(f"KiteWatt research dashboard: http://127.0.0.1:{port}")
    print("Local-only. Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
