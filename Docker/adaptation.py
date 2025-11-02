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

    try:
        first = dict((config.get("client_details") or [{}])[0] or {})
    except Exception:
        first = {}
    dataset = first.get("dataset", "") or ""
    model = first.get("model", "") or ""
    clients = int(config.get("clients", 0) or 0)

    mean_f1   = agg.get("mean_f1")
    mean_tt   = agg.get("mean_total_time")
    mean_tr   = agg.get("mean_training_time")
    mean_comm = agg.get("mean_comm_time")
    comm_share = (float(mean_comm) / max(1e-9, float(mean_tt))) if (mean_comm is not None and mean_tt is not None) else None
    comm_share_txt = "?" if comm_share is None else _fmt(comm_share, 2)

    ap_prev_small = {
        "client_selector": "ON" if ap_prev.get("client_selector") else "OFF",
        "message_compressor": "ON" if ap_prev.get("message_compressor") else "OFF",
        "heterogeneous_data_handler": "ON" if ap_prev.get("heterogeneous_data_handler") else "OFF",
    }

    instructions = (
        "Architectural Pattern Decision Record\n\n"
        "<INSTRUCTIONS>\n"
        "Role: You are an expert software architect advising a Federated Learning (FL) system.\n"
        "Task: Given the ## Context, produce ## Decision only.\n"
        "Output: return exactly one JSON object with the following keys and values:\n"
        '- \"client_selector\": \"ON\" or \"OFF\"\n'
        '- \"message_compressor\": \"ON\" or \"OFF\"\n'
        '- \"heterogeneous_data_handler\": \"ON\" or \"OFF\"\n'
        '- \"rationale\": a short string (>= 100 words)\n'
        '- \"selection_value\": <int>  # required only if \"client_selector\"==\"ON\"\n'
        "\n"
        "Guidance for selection_value:\n"
        "- Hard and mandatory constraint: pick selection_value so that at least two clients remain (i.e., selection_value < second-highest CPU). Otherwise the process will crash.\n"
        "- selection_value is a CPU threshold used by the Client Selector. Only clients with CPU > selection_value participate; others are excluded.\n"
        "- Purpose: remove stragglers (low-spec clients) that slow down the round and keep others idle. This helps reduce round time and variance.\n"
        "- How to choose: prefer a value that keeps the higher-CPU majority while excluding the clear low-CPU group. Example: CPUs [5,5,5,5,2] -> selection_value=3 keeps the four 5-CPU clients and drops the 2-CPU straggler.\n"
        "- Trade-off: try to reduce total round time without significantly hurting validation F1. If F1 is deteriorating, pick a lower threshold.\n"
        "- Bounds: use an integer in [0, max_cpu-1]. Never set selection_value >= max_cpu.\n"
        "- If evidence is insufficient or CPUs look equal, keep CS OFF or keep everyone by using selection_value near (cpu-1) so that CPU > selection_value holds for all.\n"
        "\n"
        "Rules:\n"
        "- Output only the JSON object. Do not add prose before or after.\n"
        "- Prefer stability when validation F1 is improving or stable.\n"
        "- If communication time is a large share of total time (> 0.35), consider \"message_compressor\":\"ON\".\n"
        "- If training time dominates or clients look unstable, consider \"client_selector\":\"ON\".\n"
        "- If data appears heterogeneous (non-IID) or unstable, consider \"heterogeneous_data_handler\":\"ON\".\n"
        "- If evidence is insufficient or contradictory, keep previous choices and explain briefly in \"rationale\".\n"
        "</INSTRUCTIONS>\n\n"
    )

    current_ctx = (
        "## Context\n"
        f"- Dataset: {dataset}\n"
        f"- Global Model: {model}\n"
        f"- Clients: {clients}\n"
        f"- Round: {round_idx}\n"
        f"- Mean Val F1: {_fmt(mean_f1)}\n"
        f"- Mean Training Time: {_fmt(mean_tr)} s\n"
        f"- Mean Communication Time: {_fmt(mean_comm)} s  (share of total: {comm_share_txt})\n"
        f"- Mean Total Time: {_fmt(mean_tt)} s\n"
        f"- Previous AP: {{\"client_selector\":\"{ap_prev_small['client_selector']}\","
        f"\"message_compressor\":\"{ap_prev_small['message_compressor']}\","
        f"\"heterogeneous_data_handler\":\"{ap_prev_small['heterogeneous_data_handler']}\"}}\n\n"
        "## Decision\n"
    )

    if mode == "zero":
        return instructions + current_ctx

    ex1_ctx = (
        "## Context\n"
        f"- Dataset: {dataset or 'CIFAR10'}\n"
        f"- Global Model: {model or 'SmallCNN'}\n"
        f"- Clients: {max(clients, 4) or 4}\n"
        f"- Round: {max(int(round_idx or 1) - 1, 1)}\n"
        f"- Mean Val F1: 0.56\n"
        f"- Mean Training Time: 10.0 s\n"
        f"- Mean Communication Time: 22.0 s  (share of total: 0.69)\n"
        f"- Mean Total Time: 32.0 s\n"
        '- Previous AP: {"client_selector":"OFF","message_compressor":"OFF","heterogeneous_data_handler":"OFF"}\n\n'
        "## Decision\n"
        '{"client_selector":"OFF","message_compressor":"ON","heterogeneous_data_handler":"OFF",'
        '"rationale":"Communication dominates; enable compression. Keep others OFF."}\n'
    )

    ex2_ctx = (
        "## Context\n"
        f"- Dataset: {dataset or 'FashionMNIST'}\n"
        f"- Global Model: {model or 'CNN-16k'}\n"
        f"- Clients: {max(clients, 4) or 4}\n"
        f"- Round: {int(round_idx or 2)}\n"
        f"- Mean Val F1: 0.72\n"
        f"- Mean Training Time: 18.0 s\n"
        f"- Mean Communication Time: 10.0 s  (share of total: 0.36)\n"
        f"- Mean Total Time: 28.0 s\n"
        '- Previous AP: {"client_selector":"OFF","message_compressor":"ON","heterogeneous_data_handler":"OFF"}\n\n'
        "## Decision\n"
        '{"client_selector":"OFF","message_compressor":"ON","heterogeneous_data_handler":"ON",'
        '"rationale":"Signs of heterogeneity; keep compression and enable HDH."}\n'
    )

    header_few = "Architectural Pattern Decision Records — Few-Shot\n\n"
    return header_few + instructions + ex1_ctx + "\n" + ex2_ctx + "\n" + current_ctx


def _sa_build_prompt_strict(config, round_idx, agg, ap_prev):
    core = _sa_build_prompt("zero", config, round_idx, agg, ap_prev)
    return core + ' \nReturn only one JSON object. If you turn on "client_selector", include a top-level integer "selection_value". It is the CPU threshold to filter out stragglers (only clients with CPU > selection_value participate). Choose it to keep the higher-CPU majority and reduce round time without harming validation F1. If unsure, keep previous choices.'

def _sa_generate_with_retry(model_name: str, mode: str, config, last_round, agg, ap_prev, base_urls: List[str]):
    p1 = _sa_build_prompt(mode, config, last_round, agg, ap_prev)
    raw1 = _sa_call_ollama(model_name, p1, base_urls, force_json=True, options={"temperature": 0.2, "top_p": 0.9})
    d1, r1, ok1 = _sa_parse_output(raw1)
    if ok1:
        return d1, r1
    p2 = _sa_build_prompt_strict(config, last_round, agg, ap_prev)
    raw2 = _sa_call_ollama(model_name, p2, base_urls, force_json=True, options={"temperature": 0.2, "top_p": 0.9})
    d2, r2, ok2 = _sa_parse_output(raw2)
    return d2, r2

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
        self.sa_model = default_config.get("LLM") or default_config.get("ollama_model") or "llama3.2:1b"
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

    def describe(self):
        return f"AdaptationManager(policy={self.policy}, total_rounds={self.total_rounds})"

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
            # prova a prendere il valore deciso dal modello (top-level)
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




