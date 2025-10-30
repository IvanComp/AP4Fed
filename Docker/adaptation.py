import random
import json, os, re, glob, copy
from typing import Dict, List, Tuple
import urllib.request
import pandas as pd 
from flwr.common.logger import log
from logging import INFO

current_dir = os.getcwd().replace('/adaptation', '')
config_dir = os.path.join(current_dir, 'configuration')
config_file = os.path.join(config_dir, 'config.json')
adaptation_config_file = os.path.join(config_dir, 'config.json')

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
    """
    Costruisce una ActivationCriterion statica per ogni pattern.
    Lo stato iniziale viene preso da default_config["patterns"].
    """
    criteria_list: List[ActivationCriterion] = []

    for pattern_name in get_patterns(adaptation_config):
        pat_cfg = default_config["patterns"].get(pattern_name, {})
        default_enabled = pat_cfg.get("enabled", False)
        default_params = pat_cfg.get("params", {})
        criteria_list.append(
            ActivationCriterion(
                pattern=pattern_name,
                default_enabled=default_enabled,
                default_params=default_params,
            )
        )

    return criteria_list


def get_model_type(default_config: Dict):
    # Assumiamo stesso modello per tutti i client
    return default_config["client_details"][0]["model"]

SA_PATTERNS = ["client_selector", "message_compressor", "heterogeneous_data_handler"]

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
        "mean_f1":           mean(["F1", "Val F1", "val_f1"]),
        "mean_total_time":   mean(["Total Time of FL Round", "Total Time (s)", "TotalTime"]),
        "mean_training_time":mean(["Training Time", "Training (s)", "Training Time (s)"]),
        "mean_comm_time":    mean(["Communication Time", "Comm (s)", "server_comm_time"])
    }

def _sa_extract_ap_prev(df):
    ap_prev = {p: False for p in [
        "client_selector","client_cluster","message_compressor",
        "model_co-versioning_registry","multi-task_model_trainer","heterogeneous_data_handler"
    ]}
    col = None
    for c in df.columns:
        if "AP List" in c or c.strip().lower() == "ap list":
            col = c; break
    if col is None or df.empty:
        return ap_prev
    val = str(df[col].dropna().iloc[-1]).strip().strip("{}[]() ")
    parts = [x.strip().upper() for x in val.split(",") if x.strip()]
    order = list(ap_prev.keys())
    for i, name in enumerate(order):
        ap_prev[name] = (parts[i] == "ON") if i < len(parts) else False
    return ap_prev

def _sa_few_shot_block():
    return """### FEW-SHOT

Input:
{"round": 4, "clients": 4, "dataset": "CIFAR10", "mean_f1": 0.56, "mean_total_time": 95.2, "mean_training_time": 28.1, "mean_comm_time": 60.0, "active_patterns_prev": {"client_selector": "OFF", "message_compressor": "OFF", "heterogeneous_data_handler": "OFF"}}
Output:
{"client_selector": "OFF", "message_compressor": "ON", "heterogeneous_data_handler": "OFF"}

Input:
{"round": 7, "clients": 8, "dataset": "FashionMNIST", "mean_f1": 0.61, "mean_total_time": 120.0, "mean_training_time": 85.0, "mean_comm_time": 25.0, "active_patterns_prev": {"client_selector": "OFF", "message_compressor": "ON", "heterogeneous_data_handler": "OFF"}}
Output:
{"client_selector": "ON", "message_compressor": "OFF", "heterogeneous_data_handler": "OFF"}

"""

def _sa_build_prompt(config, round_idx, agg, ap_prev):
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
        "Given the latest performance snapshot, pick ON/OFF for exactly three architectural patterns:\n"
        f"{SA_PATTERNS}.\n"
        "Rules:\n"
        "- Output MUST be ONLY one JSON object, no markdown fences, no extra text.\n"
        '- Keys: "client_selector", "message_compressor", "heterogeneous_data_handler".\n'
        '- Values: "ON" or "OFF".\n'
        "Prefer turning ON message_compressor when communication time dominates; "
        "turn ON client_selector when training time dominates; "
        "consider heterogeneous_data_handler when F1 is low or unstable.\n\n"
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

    return "### TASK\n" + rules + _sa_few_shot_block() + "### NOW SOLVE THIS CASE\nInput:\n" + \
           json.dumps(current_input, ensure_ascii=False) + "\nOutput:\n"

def _sa_parse_output(text):
    if not text:
        return {}
    m = re.search(r"\{[\s\S]*\}", text)
    raw = m.group(0) if m else text.strip()
    try:
        obj = json.loads(raw)
    except Exception:
        t = text.lower()
        obj = {
            "client_selector": "on" if ("client_selector" in t and "on" in t) else "off",
            "message_compressor": "on" if ("message_compressor" in t and "on" in t) else "off",
            "heterogeneous_data_handler": "off",
        }
    out = {}
    for k in SA_PATTERNS:
        v = str(obj.get(k, "OFF")).strip().upper()
        out[k] = "ON" if v == "ON" else "OFF"
    return out

def _sa_call_ollama(model, prompt, base_urls):
    body = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    for base in base_urls:
        try:
            req = urllib.request.Request(url=f"{base}/api/generate", data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "")
        except Exception:
            continue
    raise RuntimeError("Ollama non raggiungibile")

class AdaptationManager:
    def __init__(self, enabled: bool, default_config: Dict):
        self.name = "AdaptationManager"
        self.default_config = default_config
        self.policy = str(default_config.get("adaptation", "None")).strip()
        self.full_config = copy.deepcopy(default_config)
        self.sa_model = default_config.get("LLM", "llama3.2:1b")
        self.sa_ollama_urls = [
            default_config.get("ollama_base_url") or "http://host.docker.internal:11434",
            "http://localhost:11434",
        ]
        self.total_rounds = int(default_config.get("rounds", 1))
        self.enabled = enabled and (self.policy.lower() != "none")

        if self.enabled:
            adaptation_config = json.load(open(adaptation_config_file, "r"))
            self.patterns = get_patterns(adaptation_config)
            pattern_act_criteria = get_activation_criteria(adaptation_config, default_config)
            self.adaptation_criteria: Dict[str, ActivationCriterion] = {
                c.pattern: c for c in pattern_act_criteria
            }

            self.model_type = get_model_type(default_config)
            self.cached_config = {
                "patterns": {p: {"enabled": False} for p in PATTERNS}
            }

            self.update_config(default_config)
            self.update_json(default_config)

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
        policy_line = f"Adaptation Policy: {self.policy}"
        return log(INFO, policy_line)

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
                    self.cached_config["patterns"][pattern]["params"] = new_config["patterns"][pattern].get(
                        "params", {}
                    )
            else:
                self.cached_config["patterns"][pattern] = {
                    "enabled": new_config["patterns"][pattern]["enabled"],
                    "params": new_config["patterns"][pattern].get("params", {}),
                }

    def update_json(self, new_config: Dict):
        """
        Scrive i nuovi valori dei pattern dentro configuration/config.json.
        """
        with open(config_file, "r") as f:
            config = json.load(f)

        for pattern in new_config["patterns"]:
            config["patterns"][pattern]["enabled"] = new_config["patterns"][pattern]["enabled"]
            if "params" in new_config["patterns"][pattern]:
                config["patterns"][pattern]["params"] = new_config["patterns"][pattern]["params"]

        json.dump(config, open(config_file, "w"), indent=4)

    def _infer_current_round(self, metrics_history: Dict) -> int:
        """
        Usa metrics_history per capire a che round siamo.
        metrics_history ha questa struttura:
          metrics_history[model_type][metric_name] = [val_round1, val_round2, ...]
        Quindi la lunghezza di una lista di metriche è il numero di round già completati.
        """
        model_hist = metrics_history.get(self.model_type, {})
        round_list = model_hist.get("train_loss", [])
        return len(round_list)

    # ------------------------
    # Strategie di decisione
    # ------------------------

    def _decide_random(self, base_config: Dict, current_round: int) -> Tuple[Dict, List[str]]:
        """
        Strategia Random: per ogni pattern sceglie ON/OFF a caso.
        Ritorna:
          - nuova configurazione
          - log_lines da stampare
        """
        new_config = copy.deepcopy(base_config)
        logs: List[str] = []

        for pattern in PATTERNS:
            random_state = random.choice([True, False])
            current_params = new_config["patterns"].get(pattern, {}).get("params", {})

            new_config["patterns"][pattern]["enabled"] = random_state
            new_config["patterns"][pattern]["params"] = current_params

            logs.append(
                f"[ROUND {current_round}->{current_round + 1}] {pattern} => {self._icon(random_state)}"
            )

        return new_config, logs
    
    def _decide_single_agent(self, base_config, current_round):
        logs = []

        # 1) CSV ultimo round (ok anche se assente)
        last_round, last_csv = _sa_latest_round_csv()
        agg, ap_prev = {}, {}
        if last_csv:
            try:
                import pandas as pd
                df = pd.read_csv(last_csv)
                agg = _sa_aggregate_round(df)
                ap_prev = _sa_extract_ap_prev(df)
                logs.append(f"[Single-Agent] Using last metrics: round {last_round} • {os.path.basename(last_csv)}")
            except Exception as e:
                logs.append(f"[Single-Agent] Warning: cannot read CSV: {e}")

        # 2) Prompt
        prompt = _sa_build_prompt(self.full_config, last_round, agg, ap_prev)
        logs.append(f"[PROMPT → {self.sa_model}]\n{prompt}")

        # 3) Chiamata a Ollama
        try:
            raw = _sa_call_ollama(self.sa_model, prompt, self.sa_ollama_urls)
        except Exception as e:
            logs.append(f"[Single-Agent] ❌ Ollama error: {e}")
            return copy.deepcopy(base_config), logs

        logs.append(f"[RAW ← {self.sa_model}]\n{raw}")

        # 4) Parse + applica alle 3 chiavi
        decisions = _sa_parse_output(raw)
        icons = " • ".join([f"{k}={'✅' if decisions.get(k)=='ON' else '❌'}" for k in ("client_selector","message_compressor","heterogeneous_data_handler")])
        logs.append(f"[Single-Agent] Decisione: {icons}")
        logs.append(f"[Single-Agent] JSON: {json.dumps(decisions, ensure_ascii=False)}")

        new_cfg = copy.deepcopy(base_config)
        for p in ("client_selector","message_compressor","heterogeneous_data_handler"):
            if p in new_cfg.get("patterns", {}):
                new_cfg["patterns"][p]["enabled"] = (decisions.get(p) == "ON")

        logs.append(f"[ROUND {current_round}] Applied: {icons}")
        return new_cfg, logs

    def _decide_voting(self, base_config: Dict, current_round: int) -> Tuple[Dict, List[str]]:
        """
        Placeholder Voting-Based.
        Per ora: non cambia nulla rispetto alla config corrente.
        """
        logs = [
            f"[ROUND {current_round}] Voting-Based policy placeholder (no changes)"
        ]
        return copy.deepcopy(base_config), logs

    def _decide_role(self, base_config: Dict, current_round: int) -> Tuple[Dict, List[str]]:
        """
        Placeholder Role-Based.
        Per ora: non cambia nulla rispetto alla config corrente.
        """
        logs = [
            f"[ROUND {current_round}] Role-Based policy placeholder (no changes)"
        ]
        return copy.deepcopy(base_config), logs

    def _decide_debate(self, base_config: Dict, current_round: int) -> Tuple[Dict, List[str]]:
        """
        Placeholder Debate-Based.
        Per ora: non cambia nulla rispetto alla config corrente.
        """
        logs = [
            f"[ROUND {current_round}] Debate-Based policy placeholder (no changes)"
        ]
        return copy.deepcopy(base_config), logs

    def _decide_next_config(self, base_config: Dict, current_round: int) -> Tuple[Dict, List[str]]:
        pol = self.policy.lower()

        if pol == "random":
            return self._decide_random(base_config, current_round)
        if pol == "single ai-agent":
            return self._decide_single_agent(base_config, current_round)
        if pol == "voting-based":
            return self._decide_voting(base_config, current_round)
        if pol == "role-based":
            return self._decide_role(base_config, current_round)
        if pol == "debate-based" or pol == "debatebased":
            return self._decide_debate(base_config, current_round)

        fallback_logs = [
            f"[ROUND {current_round}] Policy '{self.policy}' not recognized or inactive (no changes)"
        ]
        return copy.deepcopy(base_config), fallback_logs

    # ------------------------
    # Loop principale per decidere la prossima config
    # ------------------------

    def config_next_round(self, metrics_history: Dict, last_round_time: float):

        if not self.enabled:
            return self.default_config["patterns"]

        current_round = self._infer_current_round(metrics_history)
        if current_round <= 0:
            current_round = 1

        is_last_round = current_round >= self.total_rounds

        log(INFO, f"[ROUND {current_round}] {self.name}: Configuring next round")
        log(
            INFO,
            f"[ROUND {current_round}] Active Architectural Patterns: "
            f"{self._format_state_array(self.cached_config['patterns'])}",
        )

        if is_last_round:
            log(
                INFO,
                f"[ROUND {current_round}] Final round reached "
                f"({current_round}/{self.total_rounds}). "
                f"No adaptation for next round."
            )
            return self.cached_config["patterns"]


        base_config = copy.deepcopy(self.cached_config)
        next_config, decision_logs = self._decide_next_config(base_config, current_round)

        # Stampa le decisioni della policy
        for line in decision_logs:
            log(INFO, line)

        # Aggiorna metriche e stato interno e scrivi su disco
        self.update_metrics(metrics_history)
        self.update_config(next_config)
        self.update_json(next_config)

        return next_config["patterns"]
