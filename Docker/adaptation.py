import json
import os
import random
import copy
import re
import glob
import time
import urllib.request, urllib.error
from typing import Dict, List, Tuple
from logging import INFO
from flwr.common.logger import log

current_dir = os.getcwd().replace('/adaptation', '')
config_dir = os.path.join(current_dir, 'configuration')
config_file = os.path.join(config_dir, 'config.json')
adaptation_config_file = os.path.join(config_dir, 'config.json')

PATTERNS = [
    "client_selector",
    "message_compressor",
    "heterogeneous_data_handler",
]
USE_RAG = True
AGENT_LOG_FILE = os.environ.get("AGENT_LOG_FILE", os.path.join(os.getcwd(), "logs", "ai_agent_decisions.txt"))

def _append_agent_log(lines):
    p = AGENT_LOG_FILE
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip("\n") + "\n")

class ActivationCriterion:
    def __init__(self, pattern: str, default_enabled: bool, default_params: Dict):
        self.pattern = pattern
        self.default_enabled = default_enabled
        self.default_params = default_params or {}
    def __str__(self) -> str:
        return f"[{self.pattern}] enabled={self.default_enabled}, params={self.default_params}"
    def activate_pattern(self, args: Dict):
        activate = self.default_enabled
        params = self.default_params
        expl = f"{self.pattern}: static policy -> enabled={activate}"
        return activate, params, expl

def get_patterns(adaptation_config: Dict) -> List[str]:
    cfg_patterns = adaptation_config.get("patterns", {})
    if cfg_patterns:
        return list(cfg_patterns.keys())
    return PATTERNS

def get_activation_criteria(adaptation_config: Dict, default_config: Dict) -> List[ActivationCriterion]:
    criteria_list: List[ActivationCriterion] = []
    for pattern_name in get_patterns(adaptation_config):
        pat_cfg = default_config["patterns"].get(pattern_name, {})
        default_enabled = pat_cfg.get("enabled", False)
        default_params = pat_cfg.get("params", {})
        criteria_list.append(ActivationCriterion(pattern=pattern_name, default_enabled=default_enabled, default_params=default_params))
    return criteria_list

def get_model_type(default_config: Dict):
    return default_config["client_details"][0]["model"]

def _sa_latest_round_csv():
    files = glob.glob("**/FLwithAP_performance_metrics_round*.csv", recursive=True)
    if not files:
        return None, None
    def rnum(p):
        m = re.search(r"round(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else -1
    lastf = max(files, key=rnum)
    return rnum(lastf), lastf

def _sa_aggregate_round(df):
    def mean(colnames):
        for c in colnames:
            if c in df.columns:
                try:
                    return df[c].astype(float).mean()
                except Exception:
                    pass
        return None
    return {
        "mean_f1": mean(["F1", "Val F1", "val_f1", "val_f1_mean"]),
        "mean_total_time": mean(["Total Time of FL Round", "Total Time (s)", "TotalTime"]),
        "mean_training_time": mean(["Training Time", "Training (s)", "Training Time (s)"]),
        "mean_comm_time": mean(["Communication Time", "Comm (s)", "server_comm_time"])
    }

def _sa_extract_ap_prev(df):
    ap_prev = {"client_selector": None, "message_compressor": None, "heterogeneous_data_handler": None}
    col = None
    for c in df.columns:
        if "AP List" in c or c.strip().lower() == "ap list":
            col = c; break
    if col is None or df.empty:
        return ap_prev
    val = str(df[col].dropna().iloc[-1]).strip().strip("{}[]() ")
    parts = [x.strip().upper() for x in re.split(r"[;,\s]+", val) if x.strip() and x.strip() not in ["{","}"]]
    if parts and all(p in {"ON","OFF"} for p in parts):
        order = ["client_selector","client_cluster","message_compressor","model_co-versioning_registry","multi-task_model_trainer","heterogeneous_data_handler"]
        for i, name in enumerate(order):
            if name in ap_prev and i < len(parts):
                ap_prev[name] = (parts[i] == "ON")
    return ap_prev

def _sa_build_prompt(mode: str, config, round_idx, agg, ap_prev):
    def _fmt(x, nd=3):
        try:
            return f"{float(x):.{nd}f}"
        except Exception:
            return "?"

    # Basic context pulled from config/agg
    try:
        first = dict((config.get("client_details") or [{}])[0] or {})
    except Exception:
        first = {}
    dataset = first.get("dataset", "") or ""
    model = first.get("model", "") or ""
    clients = int(config.get("clients", 0) or 0)

    mean_f1   = (agg or {}).get("mean_f1")
    mean_tt   = (agg or {}).get("mean_total_time")
    mean_tr   = (agg or {}).get("mean_training_time")
    mean_comm = (agg or {}).get("mean_comm_time")
    comm_share = (float(mean_comm) / max(1e-9, float(mean_tt))) if (mean_comm is not None and mean_tt is not None) else None
    comm_share_txt = "?" if comm_share is None else _fmt(comm_share, 2)

    ap_prev_small = {
        "client_selector": "ON" if (ap_prev or {}).get("client_selector") else "OFF",
        "message_compressor": "ON" if (ap_prev or {}).get("message_compressor") else "OFF",
        "heterogeneous_data_handler": "ON" if (ap_prev or {}).get("heterogeneous_data_handler") else "OFF",
    }

    # 1) Static Context: Role, Task, Output, Guardrails (keep minimal)
    instructions = (
        "Architectural Pattern Decision: Prompt\n\n"
        "## Context\n"
        "Role: You are an expert software architect advising a Federated Learning (FL) system.\n"
        "Task: Read the Context (and, if present, the RAG section) and decide which patterns to enable for the next round.\n"
        "Output: return exactly one JSON object with keys:\n"
        '- "client_selector": "ON" or "OFF"\n'
        '- "message_compressor": "ON" or "OFF"\n'
        '- "heterogeneous_data_handler": "ON" or "OFF"\n'
        '- "selection_value": <int>  # required only if "client_selector"=="ON"\n'
        '- "rationale": short explanation (≈2–3 sentences)\n'
        "Guardrails:\n"
        "- Output only the JSON object, no extra text.\n"
        "- selection_value is a CPU threshold: only clients with CPU > selection_value participate.\n"
        "- Use an integer in [0, max_cpu-1], strictly less than the second-highest CPU value.\n"
        "- If evidence is insufficient, keep previous choices and say why in rationale.\n\n"
    )

    # Runtime context always present
    current_ctx = (
        "### Runtime Context\n"
        f"- Dataset: {dataset}\n"
        f"- Global Model: {model}\n"
        f"- Clients: {clients}\n"
        f"- Round: {round_idx}\n"
        f"- Mean Val F1: {_fmt(mean_f1)}\n"
        f"- Mean Training Time: {_fmt(mean_tr)} s\n"
        f"- Mean Communication Time: {_fmt(mean_comm)} s  (share of total: {comm_share_txt})\n"
        f"- Mean Total Time: {_fmt(mean_tt)} s\n"
        f"- Previous AP: {{\"client_selector\":\"{ap_prev_small['client_selector']}\",\"message_compressor\":\"{ap_prev_small['message_compressor']}\",\"heterogeneous_data_handler\":\"{ap_prev_small['heterogeneous_data_handler']}\"}}\n\n"
    )

    # 2) Optional RAG section: include snapshots from config.json and performance CSV
    rag = ""
    try:
        use_rag = bool(USE_RAG)
    except Exception:
        use_rag = True
    if use_rag:
        # Build a compact snapshot of config.json
        cfg_summary = {}
        try:
            cfg = {}
            try:
                if os.path.exists(config_file):
                    with open(config_file, "r") as f:
                        cfg = json.load(f)
            except Exception:
                cfg = {}
            if not cfg and isinstance(config, dict):
                cfg = config
            cds = cfg.get("client_details") or []
            cpus = [int(c.get("cpu", 0) or 0) for c in cds if isinstance(c, dict)]
            dist = [str((c.get("data_distribution_type") or "")).upper() for c in cds if isinstance(c, dict)]
            counts = {k: sum(1 for d in dist if d == k) for k in set(dist or [])}
            non_iid_ids = [c.get("client_id") for c in cds if str(c.get("data_distribution_type", "")).upper() != "IID"]
            models = sorted({str(c.get("model", "")) for c in cds if isinstance(c, dict)} - {""})
            cfg_summary = {
                "dataset": (cds[0].get("dataset") if cds else cfg.get("dataset", "")) or dataset,
                "clients": int(cfg.get("clients", len(cds) or clients) or clients),
                "cpu_per_client": cpus,
                "data_distribution_counts": counts,
                "non_iid_clients": non_iid_ids,
                "models": models,
            }
        except Exception:
            cfg_summary = {}

        # Get last performance row from CSV
        last_file = None
        try:
            files = glob.glob("**/FLwithAP_performance_metrics_round*.csv", recursive=True)
            if not files:
                files = glob.glob("**/FLwithAP_performance_metrics*.csv", recursive=True)
            def rnum(p):
                m = re.search(r"round(\d+)", os.path.basename(p))
                return int(m.group(1)) if m else -1
            if files:
                last_file = max(files, key=rnum)
                try:
                    import pandas as pd  # lazy import
                    df = pd.read_csv(last_file)
                    last_row = df.tail(1).to_dict(orient="records")[0]
                except Exception:
                    last_row = {}
            else:
                last_row = {}
        except Exception:
            last_row = {}

        def pick(row, keys_list):
            for k in keys_list:
                if k in row:
                    return row[k]
            return None

        f1_last = pick(last_row, ["F1","Val F1","val_f1","val_f1_mean"])
        tr_last = pick(last_row, ["Training Time","Training (s)","Training Time (s)"])
        cm_last = pick(last_row, ["Communication Time","Comm (s)","server_comm_time"])
        tt_last = pick(last_row, ["Total Time of FL Round","Total Time (s)","TotalTime"])

        rag = (
            "## RAG\n"
            "### System Configuration (config.json)\n"
            + json.dumps(cfg_summary, ensure_ascii=False) + "\n\n"
            "### Performance Report (last row of CSV)\n"
            + json.dumps({ "file": last_file, "F1": f1_last, "Training Time (s)": tr_last, "Communication Time (s)": cm_last, "Total Time (s)": tt_last }, ensure_ascii=False)
            + "\n\n"
        )

    # 3) Few-shot examples appended only for few-shot/fine-tuning modes
    if mode != "zero":
        ex1_ctx = (
            "## Example 1\n"
            "### Context\n"
            "- CPUs: [5,5,5,5,5,5,5,5,5,5,2]\n"
            "- Data: all IID\n"
            "- Comm share: 0.30\n"
            "- Previous AP: {\"client_selector\":\"OFF\",\"message_compressor\":\"OFF\",\"heterogeneous_data_handler\":\"OFF\"}\n\n"
            "### Decision\n"
            "{\"client_selector\":\"ON\",\"selection_value\":2,\"message_compressor\":\"OFF\",\"heterogeneous_data_handler\":\"OFF\","
            "\"rationale\":\"One low-spec client (CPU=2) stalls the round; threshold >2 excludes it. All data IID; no HDH. Comm not dominant.\"}\n"
        )
        ex2_ctx = (
            "## Example 2\n"
            "### Context\n"
            "- CPUs: [5,5,5,5]\n"
            "- Data: all IID\n"
            "- Comm share: 0.62\n"
            "- Previous AP: {\"client_selector\":\"OFF\",\"message_compressor\":\"OFF\",\"heterogeneous_data_handler\":\"OFF\"}\n\n"
            "### Decision\n"
            "{\"client_selector\":\"OFF\",\"message_compressor\":\"ON\",\"heterogeneous_data_handler\":\"OFF\","
            "\"rationale\":\"Communication dominates; enable compression. Balanced CPUs, keep selector OFF. No non-IID, HDH OFF.\"}\n"
        )
        header_few = "### Few-Shot Examples\n\n"
        return header_few + instructions + (rag or "") + ex1_ctx + "\n" + ex2_ctx + "\n" + current_ctx + "## Decision\n"

    return instructions + (rag or "") + current_ctx + "## Decision\n"

def _sa_generate_with_retry(model_name: str, mode: str, config, last_round, agg, ap_prev, base_urls: List[str]):
    p = _sa_build_prompt(mode, config, last_round, agg, ap_prev)
    raw = _sa_call_ollama(model_name, p, base_urls, force_json=True, options={"temperature": 0.2, "top_p": 0.9})
    d1, r1, _ = _sa_parse_output(raw)
    return d1, r1

def _sa_mode_from_policy(policy: str) -> str:
    pol = (policy or "").lower()
    if "zero" in pol:
        return "zero"
    if "few" in pol:
        return "few"
    if "fine" in pol:
        return "ft"
    return "few"

def _sa_parse_output(text: str):
    if not text:
        return {"client_selector":"OFF","message_compressor":"OFF","heterogeneous_data_handler":"OFF"}, "", False
    for m in re.finditer(r"\{.*?\}", text, flags=re.S):
        chunk = m.group(0).strip()
        try:
            obj = json.loads(chunk)
            if any(k in obj for k in ("client_selector","message_compressor","heterogeneous_data_handler")):
                decisions = {}
                for k in ("client_selector","message_compressor","heterogeneous_data_handler"):
                    v = str(obj.get(k, "OFF")).strip().upper()
                    decisions[k] = "ON" if v == "ON" else "OFF"

                if "selection_value" in obj:
                    try:
                        decisions["selection_value"] = int(obj["selection_value"])
                    except Exception:
                        pass

                rationale = str(obj.get("rationale", "")).strip()
                return decisions, rationale, bool(rationale)
        except Exception:
            continue
    return {"client_selector":"OFF","message_compressor":"OFF","heterogeneous_data_handler":"OFF"}, "", False

def _sa_call_ollama(model: str, prompt: str, base_urls: List[str], force_json: bool = True, options: dict = None) -> str:
    def _is_gpt_oss(name: str) -> bool:
        n = (name or "").lower()
        return n.startswith("gpt-oss") or (":" in n and n.split(":", 1)[0] == "gpt-oss")

    last_err = None
    for base in base_urls:
        try:
            if _is_gpt_oss(model):
                payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False, "think": "low"}
                if force_json:
                    payload["format"] = "json"
                if options:
                    payload["options"] = options
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(url=f"{base}/api/chat", data=data, headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=60) as resp:
                    out = json.loads(resp.read().decode("utf-8"))
                msg = out.get("message", {}) or {}
                content = msg.get("content", "") or ""
                thinking = msg.get("thinking") or msg.get("reasoning") or ""
                if force_json:
                    try:
                        obj = json.loads(content)
                        if isinstance(obj, dict) and any(k in obj for k in ("client_selector","message_compressor","heterogeneous_data_handler")):
                            if "rationale" not in obj and thinking:
                                obj["rationale"] = str(thinking)[:800]
                            return json.dumps(obj, ensure_ascii=False)
                    except Exception:
                        pass
                return content
            else:
                payload = {"model": model, "prompt": prompt, "stream": False}
                if force_json:
                    payload["format"] = "json"
                if options:
                    payload["options"] = options
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(url=f"{base}/api/generate", data=data, headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=60) as resp:
                    out = json.loads(resp.read().decode("utf-8"))
                if "error" in out:
                    raise RuntimeError(str(out["error"]))
                return out.get("response", "")
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Ollama unreachable: {last_err}")

class AdaptationManager:
    def __init__(self, enabled: bool, default_config: Dict, use_rag=USE_RAG):
        self.name = "AdaptationManager"
        self.use_rag = use_rag
        self.default_config = default_config
        self.policy = str(default_config.get("adaptation", "None")).strip()
        self.total_rounds = int(default_config.get("rounds", 1))
        self.enabled = enabled and (self.policy.lower() != "none")
        self.sa_model = default_config.get("LLM") or "llama3.2:1b"
        self.sa_ollama_urls = [
            default_config.get("ollama_base_url") or "http://host.docker.internal:11434",
            "http://localhost:11434",
        ]
        if self.enabled:
            adaptation_config = json.load(open(adaptation_config_file, "r"))
            self.patterns = get_patterns(adaptation_config)
            pattern_act_criteria = get_activation_criteria(adaptation_config, default_config)
            self.adaptation_criteria: Dict[str, ActivationCriterion] = {c.pattern: c for c in pattern_act_criteria}
            self.model_type = get_model_type(default_config)
            self.cached_config = {
                "patterns": {
                    p: {
                        "enabled": bool(default_config["patterns"].get(p, {}).get("enabled", False)),
                        "params": default_config["patterns"].get(p, {}).get("params", {}),
                    }
                    for p in self.patterns
                }
            }
            self.cached_aggregated_metrics = None
        else:
            self.patterns = PATTERNS[:]
            self.adaptation_criteria = {}
            self.model_type = get_model_type(default_config)
            self.cached_config = {
                "patterns": {
                    p: {"enabled": default_config["patterns"][p]["enabled"]}
                    for p in PATTERNS
                    if p in default_config.get("patterns", {})
                }
            }
            self.cached_aggregated_metrics = None

    def _icon(self, enabled: bool) -> str:
        return "✅" if enabled else "❌"

    def _format_state_array(self, cfg_patterns: Dict[str, Dict]) -> str:
        parts = []
        for p in PATTERNS:
            state = cfg_patterns.get(p, {}).get("enabled", False)
            parts.append(f"{p}={self._icon(state)}")
        return "[" + ", ".join(parts) + "]"

    def update_metrics(self, new_aggregated_metrics: Dict):
        self.cached_aggregated_metrics = new_aggregated_metrics

    def update_config(self, new_config: Dict):
        for pattern in new_config["patterns"]:
            if pattern in self.cached_config["patterns"]:
                if "enabled" in self.cached_config["patterns"][pattern]:
                    self.cached_config["patterns"][pattern]["enabled"] = new_config["patterns"][pattern]["enabled"]
                else:
                    self.cached_config["patterns"][pattern] = {
                        "enabled": new_config["patterns"][pattern]["enabled"],
                        "params": new_config["patterns"][pattern].get("params", {}),
                    }
                if "params" in self.cached_config["patterns"][pattern]:
                    self.cached_config["patterns"][pattern]["params"] = new_config["patterns"][pattern].get("params", {})

    def update_json(self, new_config: Dict):
        with open(config_file, "r") as f:
            config = json.load(f)
        for pattern in new_config["patterns"]:
            if pattern in config.get("patterns", {}):
                config["patterns"][pattern]["enabled"] = new_config["patterns"][pattern]["enabled"]
                if "params" in config["patterns"][pattern] and "params" in new_config["patterns"][pattern]:
                    config["patterns"][pattern]["params"] = new_config["patterns"][pattern]["params"]
        json.dump(config, open(config_file, "w"), indent=4)

    def _infer_current_round(self, metrics_history: Dict) -> int:
        model_hist = metrics_history.get(self.model_type, {})
        round_list = model_hist.get("train_loss", [])
        return len(round_list)

    def _decide_random(self, base_config: Dict, current_round: int) -> Tuple[Dict, List[str]]:
        new_config = copy.deepcopy(base_config)
        logs: List[str] = []
        for pattern in PATTERNS:
            random_state = random.choice([True, False])
            new_config["patterns"][pattern]["enabled"] = random_state
        logs.append(f"[ROUND {current_round}] Random policy executed")
        return new_config, logs

    def _decide_voting(self, base_config: Dict, current_round: int) -> Tuple[Dict, List[str]]:
        logs = [f"[ROUND {current_round}] Voting-based policy not implemented (no changes)"]
        return copy.deepcopy(base_config), logs

    def _decide_role(self, base_config: Dict, current_round: int) -> Tuple[Dict, List[str]]:
        logs = [f"[ROUND {current_round}] Role-based policy not implemented (no changes)"]
        return copy.deepcopy(base_config), logs

    def _decide_debate(self, base_config: Dict, current_round: int) -> Tuple[Dict, List[str]]:
        logs = [f"[ROUND {current_round}] Debate-based policy not implemented (no changes)"]
        return copy.deepcopy(base_config), logs

    def _decide_single_agent(self, base_config: Dict, current_round: int) -> Tuple[Dict, List[str]]:
        logs: List[str] = []
        last_round, last_csv = _sa_latest_round_csv()
        agg, ap_prev = {}, {}
        if last_csv:
            try:
                import pandas as pd
                df = pd.read_csv(last_csv)
                agg = _sa_aggregate_round(df)
                ap_prev = _sa_extract_ap_prev(df)
            except Exception:
                pass
        mode = _sa_mode_from_policy(self.policy)
        if mode == "ft":
            logs.append(f"[Single-Agent] Decision ({self.sa_model}): CS: ·→❌ • MC: ·→❌ • HDH: ·→❌")
            logs.append(f"[Single-Agent] Rationale ({self.sa_model}): Fine-Tuning mode is not implemented")
            return copy.deepcopy(base_config), logs
        try:
            decisions, rationale = _sa_generate_with_retry(self.sa_model, mode, self.default_config, last_round, agg, ap_prev, self.sa_ollama_urls)
        except Exception:
            return copy.deepcopy(base_config), logs
        prev_cs = "✅" if ap_prev.get("client_selector") else ("❌" if ap_prev.get("client_selector") is not None else "·")
        prev_mc = "✅" if ap_prev.get("message_compressor") else ("❌" if ap_prev.get("message_compressor") is not None else "·")
        prev_hdh = "✅" if ap_prev.get("heterogeneous_data_handler") else ("❌" if ap_prev.get("heterogeneous_data_handler") is not None else "·")
        new_cs = "✅" if decisions.get("client_selector") == "ON" else "❌"
        new_mc = "✅" if decisions.get("message_compressor") == "ON" else "❌"
        new_hdh = "✅" if decisions.get("heterogeneous_data_handler") == "ON" else "❌"
        delta = " • ".join([f"CS: {prev_cs}→{new_cs}", f"MC: {prev_mc}→{new_mc}", f"HDH: {prev_hdh}→{new_hdh}"])
        logs.append(f"[Single-Agent] Decision ({self.sa_model}): {delta}")
        logs.append(f"[Single-Agent] Rationale ({self.sa_model}): {rationale if rationale else 'This model does not support rationale generation.'}")
        new_config = copy.deepcopy(base_config)
        for p in PATTERNS:
            if p in new_config.get("patterns", {}):
                new_config["patterns"][p]["enabled"] = (decisions.get(p, "OFF") == "ON")
        cs_enabled = bool(new_config.get("patterns", {}).get("client_selector", {}).get("enabled"))
        if cs_enabled:
            existing = new_config["patterns"]["client_selector"].get("params", {}).get("selection_value")
            sel_val = None
            try:
                sel_val = int(decisions.get("selection_value")) if "selection_value" in decisions else None
            except Exception:
                sel_val = None
            if sel_val is None:
                try:
                    sel_val = int(existing) if existing is not None else 0
                except Exception:
                    sel_val = 0

            new_config["patterns"]["client_selector"]["params"] = {
                "selection_strategy": "Resource-Based",
                "selection_criteria": "CPU",
                "selection_value": sel_val
            }
        return new_config, logs

    def _decide_next_config(self, base_config: Dict, current_round: int) -> Tuple[Dict, List[str]]:
        pol = self.policy.lower().strip()
        if pol in ["single ai-agent", "single ai agent"] or "single" in pol:
            return self._decide_single_agent(base_config, current_round)
        if pol == "random":
            return self._decide_random(base_config, current_round)
        if pol == "voting-based":
            return self._decide_voting(base_config, current_round)
        if pol == "role-based":
            return self._decide_role(base_config, current_round)
        if pol in ["debate-based","debatebased"]:
            return self._decide_debate(base_config, current_round)
        fallback_logs = [f"[ROUND {current_round}] Policy '{self.policy}' not recognized or inactive (no changes)"]
        return copy.deepcopy(base_config), fallback_logs

    def config_next_round(self, metrics_history: Dict, last_round_time: float):
        
        t_agents_start = time.perf_counter()
        if not self.enabled:
            return self.default_config["patterns"]
        current_round = self._infer_current_round(metrics_history)
        if current_round <= 0:
            current_round = 1
        is_last_round = current_round >= self.total_rounds
        if is_last_round:
            log(INFO, f"[ROUND {current_round}] Final round reached ({current_round}/{self.total_rounds}). No adaptation for next round.")
            return self.cached_config["patterns"]

        base_config = copy.deepcopy(self.cached_config)


        next_config, decision_logs = self._decide_next_config(base_config, current_round)
        t_agents_finish = time.perf_counter()
        t_agent =  t_agents_finish - t_agents_start

        for line in decision_logs:
            log(INFO, line)

        _append_agent_log([
            f"\n== ROUND {current_round} ==",
            *decision_logs,
            f"[{self.policy}] PolicyTime: {t_agent:.2f}s",
            ""
        ])

        self.update_metrics(metrics_history)
        self.update_config(next_config)
        self.update_json(next_config)
        return next_config["patterns"]
