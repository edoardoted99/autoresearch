"""
Local proxy that forwards Ollama CLI requests to LlamaPass with auth header.

Usage:
  1. python llamapass_proxy.py
  2. In another terminal: OLLAMA_HOST=http://localhost:11435 ollama run qwen3-coder-next:latest
"""

import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.request
import ssl

LLAMAPASS_URL = "https://llamapass.org/ollama"
API_KEY = os.environ.get("API_KEY", "")
LOCAL_PORT = 11435


class ProxyHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        url = f"{LLAMAPASS_URL}{self.path}"
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Api-Key {API_KEY}")

        ctx = ssl.create_default_context()
        try:
            resp = urllib.request.urlopen(req, context=ctx)
            self.send_response(resp.status)
            self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
            self.end_headers()
            while True:
                chunk = resp.read(1)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
            resp.close()
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.end_headers()
            self.wfile.write(e.read())

    def do_GET(self):
        url = f"{LLAMAPASS_URL}{self.path}"
        req = urllib.request.Request(url, method="GET")
        req.add_header("Authorization", f"Api-Key {API_KEY}")

        ctx = ssl.create_default_context()
        try:
            with urllib.request.urlopen(req, context=ctx) as resp:
                resp_body = resp.read()
                self.send_response(resp.status)
                for key, val in resp.getheaders():
                    if key.lower() not in ("transfer-encoding", "connection"):
                        self.send_header(key, val)
                self.end_headers()
                self.wfile.write(resp_body)
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.end_headers()
            self.wfile.write(e.read())

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        print(f"[proxy] {args[0]}")


if __name__ == "__main__":
    if not API_KEY:
        print("ERROR: set API_KEY environment variable")
        raise SystemExit(1)
    server = HTTPServer(("127.0.0.1", LOCAL_PORT), ProxyHandler)
    print(f"Proxy listening on http://localhost:{LOCAL_PORT}")
    print(f"Forwarding to {LLAMAPASS_URL} with auth")
    print(f"\nIn another terminal run:")
    print(f"  OLLAMA_HOST=http://localhost:{LOCAL_PORT} ollama run qwen3-coder-next:latest")
    server.serve_forever()
