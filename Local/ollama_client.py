import json
import urllib.request

from typing import List

def _sa_call_ollama(model: str, prompt: str, base_urls: List[str], force_json: bool = True, options: dict = None) -> str:
    def _is_gpt_oss(name: str) -> bool:
        n = (name or "").lower()
        return n.startswith("gpt-oss") or (":" in n and n.split(":", 1)[0] == "gpt-oss")

    def _is_llama(name: str) -> bool:
        return (name or "").lower().startswith("llama")

    def _is_json_friendly(name: str) -> bool:
        return (name or "").lower().startswith("deepseek")

    last_err = None
    for base in base_urls:
        try:
            if _is_gpt_oss(model):
                # Usa /api/chat per gpt-oss, con reasoning e JSON mode abilitabile
                body = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "think": "low"
                }
                if force_json:
                    body["format"] = "json"
                if options:
                    body["options"] = options

                data = json.dumps(body).encode("utf-8")
                req = urllib.request.Request(
                    f"{base}/api/chat",
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=180) as resp:
                    out = json.loads(resp.read().decode("utf-8"))

                if "error" in out:
                    raise RuntimeError(str(out["error"]))

                msg = out.get("message") or {}
                content = (msg.get("content") or out.get("response") or "").strip()
                reasoning = (msg.get("reasoning") or msg.get("thinking") or out.get("reasoning") or "").strip()

                if force_json and content:
                    try:
                        obj = json.loads(content)
                        if isinstance(obj, dict) and any(k in obj for k in ("client_selector", "message_compressor", "heterogeneous_data_handler")):
                            if "rationale" not in obj and reasoning:
                                obj["rationale"] = reasoning[:800]
                            return json.dumps(obj, ensure_ascii=False)
                    except Exception:
                        pass

                return content

            else:
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                }
                opts = dict(options or {})
                if _is_llama(model):
                    opts.setdefault("temperature", 0.5)
                    opts.setdefault("top_p", 1.0)
                    opts.setdefault("num_ctx", 4096)  # opzionale ma utile
                    common_stops = ["}\n", "}\r\n", "\n\n##", "\n###", "\n# ", "```"]
                    if isinstance(opts.get("stop"), list):
                        for s in common_stops:
                            if s not in opts["stop"]:
                                opts["stop"].append(s)
                    else:
                        opts.setdefault("stop", common_stops)

                if force_json and (_is_json_friendly(model) or _is_llama(model)):
                    payload["format"] = "json"

                if opts:
                    payload["options"] = opts

                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    f"{base.rstrip('/')}/api/generate",
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=180) as resp:
                    out = json.loads(resp.read().decode("utf-8"))

                if "error" in out:
                    raise RuntimeError(str(out["error"]))
                return (out.get("response") or "").strip()
            
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Ollama unreachable: {last_err}")
