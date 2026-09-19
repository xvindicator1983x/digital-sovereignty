# Local Voice Assistant with Home Assistant and Unsloth

**Problem:** Home Assistant's built-in Assist voice assistant uses Ollama's native protocol for local LLM inference. But if you want faster inference (~2x) and better quantization, you should use [Unsloth](https://github.com/unslothai/unsloth). The catch: Unsloth only speaks the OpenAI-compatible API, not Ollama's protocol.

**Solution:** A thin stdlib-only Python shim that translates Ollama protocol → OpenAI protocol, letting HA Assist talk to Unsloth without any modifications to either side.

## Architecture

```
[Microphone] → [HA Assist] → [Ollama Protocol :11434]
                                     ↓
                              [ollama-shim (this guide)]
                                     ↓
                          [OpenAI API :8888/v1]
                                     ↓
                            [Unsloth Studio]
                            [Qwen model loaded]
```

## Why Unsloth over Ollama?

- **~2x faster inference** on the same hardware
- **Dynamic quantization** (per-layer bit allocation) vs Ollama's uniform Q4_K_M — better quality at same size
- **Larger context windows** (262k tokens tested)
- Same underlying weights, better packaging

## The Shim

A single Python file with zero dependencies beyond the standard library. It listens on port 11434 (Ollama's default), translates requests to OpenAI format, forwards them to your Unsloth instance, and translates responses back.

```python
#!/usr/bin/env python3
"""
Ollama → OpenAI protocol shim for Home Assistant Assist.
Listens on :11434, forwards to OpenAI-compatible backend (e.g., Unsloth).
Stdlib only — no pip install needed.
"""

import json
import http.server
import urllib.request
import urllib.parse

BACKEND_URL = "http://127.0.0.1:8888/v1"  # Your Unsloth instance
MODEL_NAME = "unsloth/Qwen3.5-9B-GGUF"   # Model loaded in Unsloth


class ShimHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/api/chat":
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length))

            # Translate Ollama format → OpenAI format
            openai_body = {
                "model": MODEL_NAME,
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
                f"{BACKEND_URL}/chat/completions",
                data=json.dumps(openai_body).encode(),
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            try:
                with urllib.request.urlopen(req) as resp:
                    result = json.loads(resp.read())

                    # Translate back to Ollama format
                    o_response = {
                        "model": body.get("model", MODEL_NAME),
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
    server = http.server.HTTPServer(("0.0.0.0", 11434), ShimHandler)
    print("Ollama shim listening on :11434")
    print(f"Forwarding to {BACKEND_URL}")
    server.serve_forever()
```

## Deployment

### 1. Install the shim

```bash
sudo mkdir -p /opt/ollama-shim
sudo tee /opt/ollama-shim/shim.py > /dev/null << 'EOF'
# (Paste the Python code above)
EOF
sudo chmod +x /opt/ollama-shim/shim.py
```

### 2. Create systemd service

```bash
sudo tee /etc/systemd/system/ollama-shim.service > /dev/null << EOF
[Unit]
Description=Ollama Protocol Shim for Home Assistant Assist
After=network.target unsloth-studio.service
Wants=unsloth-studio.service

[Service]
Type=simple
ExecStart=/usr/bin/python3 /opt/ollama-shim/shim.py
Restart=always
RestartSec=5
User=your_username
Environment="PYTHONUNBUFFERED=1"

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now ollama-shim
```

### 3. Configure Home Assistant

In HA's Assist settings:
- **Conversation agent:** `conversation.ollama_conversation` (the built-in one)
- **Ollama URL:** `http://<your-server-ip>:11434`
- **Model:** Any name (it gets overridden by the shim)

The HA pipeline will send Ollama protocol to port 11434, the shim translates it and forwards to Unsloth at :8888/v1.

## Testing

```bash
# Test the shim directly
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Hello"}]}'

# Should return JSON with assistant response from your Unsloth model
```

## Gotchas

1. **HTTP/1.1 streaming:** HA's httpx client hangs on streaming responses unless `close_connection=True` is set after the body. The shim above uses non-streaming for simplicity.

2. **Tool calling:** If you need tool/function calling, the shim must also translate OpenAI's nested tools format to Ollama's flat format and back. This requires additional handling in the request/response translation.

3. **Authentication:** Unsloth accepts bearer token `ollama` or no header, but rejects arbitrary tokens. The shim sends no auth header by default.

## Benefits

- **Fully local** — no data leaves your network
- **Fast** — Unsloth's optimized inference
- **Private** — no telemetry, no API keys
- **Simple** — one Python file, zero dependencies

This is real digital sovereignty: a voice assistant running on hardware you control, with models you own, talking to services you host. 🔥
