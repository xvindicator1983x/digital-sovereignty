#!/usr/bin/env python3
"""
Ollama → OpenAI protocol shim for Home Assistant Assist.
Listens on :11434, forwards to OpenAI-compatible backend (e.g., Unsloth).
Stdlib only — no pip install needed.

Usage: python3 shim.py [--backend http://host:port/v1] [--model MODEL_NAME]
"""

import json
import sys
import argparse
import http.server
import urllib.request


def parse_args():
    parser = argparse.ArgumentParser(description="Ollama → OpenAI protocol shim")
    parser.add_argument("--backend", default="http://127.0.0.1:8888/v1",
                        help="OpenAI-compatible backend URL (default: http://127.0.0.1:8888/v1)")
    parser.add_argument("--model", default="unsloth/Qwen3.5-9B-GGUF",
                        help="Model name to use in requests")
    return parser.parse_args()


class ShimHandler(http.server.BaseHTTPRequestHandler):
    backend_url = None
    model_name = None

    def do_POST(self):
        if self.path == "/api/chat":
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length))

            # Translate Ollama format → OpenAI format
            openai_body = {
                "model": self.model_name,
                "messages": [],
                "stream": False,
            }

            # System prompt
            if body.get("system"):
                openai_body["messages"].append({
                    "role": "system",
                    "content": body["system"]
                })

            # Chat history
            for msg in body.get("messages", []):
                openai_body["messages"].append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })

            # Parameters
            opts = body.get("options", {})
            if "temperature" in opts:
                openai_body["temperature"] = opts["temperature"]
            if "top_p" in opts:
                openai_body["top_p"] = opts["top_p"]
            if "num_predict" in opts:
                openai_body["max_tokens"] = opts["num_predict"]

            # Forward to backend
            req = urllib.request.Request(
                f"{self.backend_url}/chat/completions",
                data=json.dumps(openai_body).encode(),
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            try:
                with urllib.request.urlopen(req) as resp:
                    result = json.loads(resp.read())

                    # Translate back to Ollama format
                    o_response = {
                        "model": body.get("model", self.model_name),
                        "message": {
                            "role": "assistant",
                            "content": result["choices"][0]["message"]["content"]
                        },
                        "done": True,
                        "total_duration": 0,
                        "load_duration": 0,
                        "prompt_eval_count": 0,
                        "eval_count": 0
                    }

                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(o_response).encode())

            except Exception as e:
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                error = {"error": str(e)}
                self.wfile.write(json.dumps(error).encode())

        else:
            # Handle other Ollama endpoints (tags, etc.)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if "/api/tags" in self.path:
                self.wfile.write(json.dumps({"models": []}).encode())
            else:
                self.wfile.write(b"{}")

    def log_message(self, format, *args):
        pass  # Suppress access logs


if __name__ == "__main__":
    args = parse_args()
    ShimHandler.backend_url = args.backend.rstrip("/")
    ShimHandler.model_name = args.model

    server = http.server.HTTPServer(("0.0.0.0", 11434), ShimHandler)
    print("Ollama shim listening on :11434")
    print(f"Forwarding to {args.backend}")
    print(f"Model: {args.model}")
    server.serve_forever()
