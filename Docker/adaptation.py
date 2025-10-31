import json
import os
import random
import copy
import re
import glob
import urllib.request, urllib.error
from typing import Dict, List, Tuple
from logging import INFO
from flwr.common.logger import log

current_dir = os.getcwd().replace('/adaptation', '')
config_dir = os.path.join(current_dir, 'configuration')
config_file = os.path.join(config_dir, 'config.json')
adaptation_config_file = os.path.join(config_dir, 'config.json')

LOG_DIR = os.path.join(current_dir, "logs")
LOG_FILE = os.path.join(LOG_DIR, "ai_agent_decisions.txt")
os.makedirs(LOG_DIR, exist_ok=True)
PATTERNS = [
    "client_selector",
    "message_compressor",
    "heterogeneous_data_handler",
]

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

def _sa_few_shot_block():
    return """### FEW-SHOT

Input:
{"round": 4, "clients": 4, "dataset": "CIFAR10", "mean_f1": 0.56, "mean_total_time": 95.2, "mean_training_time": 28.1, "mean_comm_time": 60.0, "active_patterns_prev": {"client_selector": "OFF", "message_compressor": "OFF", "heterogeneous_data_handler": "OFF"}}
Output:
{"client_selector": "OFF", "message_compressor": "ON", "heterogeneous_data_handler": "OFF", "rationale": "Communication dominates, enable compression. Others OFF."}

Input:
{"round": 7, "clients": 8, "dataset": "FashionMNIST", "mean_f1": 0.61, "mean_total_time": 120.0, "mean_training_time": 85.0, "mean_comm_time": 25.0, "active_patterns_prev": {"client_selector": "OFF", "message_compressor": "ON", "heterogeneous_data_handler": "OFF"}}
Output:
{"client_selector": "ON", "message_compressor": "OFF", "heterogeneous_data_handler": "OFF", "rationale": "Training dominates the round time, enable client selection and disable compression."}

"""

def _sa_build_prompt(mode: str, config, round_idx, agg, ap_prev):
    dataset, model = "", ""
    try:
        first = dict(config.get("client_details", [{}])[0] or {})
        dataset = first.get("dataset", "")
        model = first.get("model", "")
    except Exception:
        pass
    ap_prev_small = {
        "client_selector": "ON" if ap_prev.get("client_selector") else "OFF",
        "message_compressor": "ON" if ap_prev.get("message_compressor") else "OFF",
        "heterogeneous_data_handler": "ON" if ap_prev.get("heterogeneous_data_handler") else "OFF",
    }
    rules = (
        "You are a runtime advisor for a Federated Learning system. "
        "Return ONE JSON object with exactly four keys: "
        "\"client_selector\",\"message_compressor\",\"heterogeneous_data_handler\" (each \"ON\" or \"OFF\"), and \"rationale\" (one short English sentence). "
        "Output JSON only, no extra text. "
        "Guidelines: if communication time dominates versus total, prefer message_compressor=ON; "
        "if training time dominates, prefer client_selector=ON; "
        "if F1 is low or unstable, consider heterogeneous_data_handler=ON.\n\n"
    )
    current_input = {
        "round": round_idx,
        "clients": int(config.get("clients", 0)),
        "dataset": dataset,
        "model": model,
        "mean_f1": (agg or {}).get("mean_f1"),
        "mean_total_time": (agg or {}).get("mean_total_time"),
        "mean_training_time": (agg or {}).get("mean_training_time"),
        "mean_comm_time": (agg or {}).get("mean_comm_time"),
        "active_patterns_prev": ap_prev_small,
    }
    core = "### TASK\n" + rules + "### NOW SOLVE THIS CASE\nInput:\n" + json.dumps(current_input, ensure_ascii=False) + "\nOutput:\n"
    if mode == "few":
        return "### TASK\n" + rules + _sa_few_shot_block() + "### NOW SOLVE THIS CASE\nInput:\n" + json.dumps(current_input, ensure_ascii=False) + "\nOutput:\n"
    return core

def _sa_build_prompt_strict(config, round_idx, agg, ap_prev):
    schema = {
        "client_selector": "ON|OFF",
        "message_compressor": "ON|OFF",
        "heterogeneous_data_handler": "ON|OFF",
        "rationale": "string"
    }
    payload = {
        "round": round_idx,
        "clients": int(config.get("clients", 0)),
        "dataset": (config.get("client_details",[{}])[0] or {}).get("dataset",""),
        "model":   (config.get("client_details",[{}])[0] or {}).get("model",""),
        "mean_f1": (agg or {}).get("mean_f1"),
        "mean_total_time": (agg or {}).get("mean_total_time"),
        "mean_training_time": (agg or {}).get("mean_training_time"),
        "mean_comm_time": (agg or {}).get("mean_comm_time"),
        "active_patterns_prev": {
            "client_selector": "ON" if (ap_prev or {}).get("client_selector") else "OFF",
            "message_compressor": "ON" if (ap_prev or {}).get("message_compressor") else "OFF",
            "heterogeneous_data_handler": "ON" if (ap_prev or {}).get("heterogeneous_data_handler") else "OFF",
        },
    }
    return (
        "Return a single compact JSON object on one line matching this schema and nothing else. "
        "Fields: client_selector,message_compressor,heterogeneous_data_handler,rationale. "
        "Values for the first three must be \"ON\" or \"OFF\". rationale must be a short English sentence.\n"
        "Input:\n" + json.dumps(payload, ensure_ascii=False) + "\n"
        "Schema:\n" + json.dumps(schema, ensure_ascii=False) + "\n"
        "Output:\n"
    )

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
                rationale = str(obj.get("rationale", "")).strip()
                return decisions, rationale, bool(rationale)
        except Exception:
            continue
    return {"client_selector":"OFF","message_compressor":"OFF","heterogeneous_data_handler":"OFF"}, "", False

def _sa_call_ollama(model: str, prompt: str, base_urls: List[str], force_json: bool = True, options: dict = None) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False}
    if force_json:
        payload["format"] = "json"
    if options:
        payload["options"] = options
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    last_err = None
    for base in base_urls:
        try:
            req = urllib.request.Request(url=f"{base}/api/generate", data=data, headers=headers, method="POST")
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
    def __init__(self, enabled: bool, default_config: Dict):
        self.name = "AdaptationManager"
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
        logs.append(f"[Single-Agent] Rationale ({self.sa_model}): {rationale if rationale else 'N/A'}")
        new_config = copy.deepcopy(base_config)
        for p in PATTERNS:
            if p in new_config.get("patterns", {}):
                new_config["patterns"][p]["enabled"] = (decisions.get(p, "OFF") == "ON")
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
        if not self.enabled:
            return self.default_config["patterns"]
        current_round = self._infer_current_round(metrics_history)
        if current_round <= 0:
            current_round = 1
        is_last_round = current_round >= self.total_rounds
        if current_round == 1:
            open(LOG_FILE, "w", encoding="utf-8").close()
        if is_last_round:
            log(INFO, f"[ROUND {current_round}] Final round reached ({current_round}/{self.total_rounds}). No adaptation for next round.")
            return self.cached_config["patterns"]
        base_config = copy.deepcopy(self.cached_config)
        next_config, decision_logs = self._decide_next_config(base_config, current_round)
        for line in decision_logs:
            log(INFO, line)
        if len(decision_logs) >= 2:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"== ROUND {current_round} ==\n\n")
                f.write(decision_logs[-2] + "\n")
                f.write(decision_logs[-1] + "\n\n")
        self.update_metrics(metrics_history)
        self.update_config(next_config)
        self.update_json(next_config)
        return next_config["patterns"]


