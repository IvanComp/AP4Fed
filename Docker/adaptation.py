import csv
import json
import os
import random
import copy
import re
import glob
import time
import urllib.request, urllib.error
from typing import Dict, List, Tuple, Optional
from logging import INFO
from flwr.common.logger import log
from agent_policies import AgentPolicyMixin

from adaptation_settings import (
    AGENT_LOG_FILE,
    PERFORMANCE_DIR,
    PATTERNS,
    RATIONALE_CSV_FILE,
    USE_RAG,
    adaptation_config_file,
    config_dir,
    config_file,
    current_dir,
)
from adaptation_criteria import (
    ActivationCriterion,
    get_activation_criteria,
    get_model_type,
    get_patterns,
)
from adaptation_metrics import _sa_aggregate_round, _sa_latest_round_csv
from agent_prompting import (
    _sa_build_prompt,
    _sa_generate_with_retry,
    _sa_mode_from_policy,
    _sa_parse_output,
)
from ollama_client import _sa_call_ollama
from rationale_store import (
    _append_agent_log,
    _extract_rationale_entries,
    _persist_round_rationales,
)

class AdaptationManager(AgentPolicyMixin):
    def __init__(self, enabled: bool, default_config: Dict, use_rag=USE_RAG):
        self.name = "AdaptationManager"
        self.use_rag = use_rag
        self.adaptation_time = None
        self._last_round_client_ids: List[str] = []
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

    def describe(self):
        if not self.enabled:
            return log(INFO, f"{self.name}: adaptation disabled")
        if not getattr(self, "adaptation_criteria", None):
            return log(INFO, f"{self.name}: no explicit activation criteria loaded")
        return log(INFO, "\n".join([str(cr) for cr in self.adaptation_criteria.values()]))

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

    def set_last_round_client_ids(self, client_ids: Optional[List[str]]) -> None:
        deduped: List[str] = []
        seen = set()
        for raw_cid in client_ids or []:
            cid = str(raw_cid).strip()
            if not cid or cid in seen:
                continue
            deduped.append(cid)
            seen.add(cid)
        self._last_round_client_ids = deduped

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

    def _get_client_cpus(self, config: Optional[Dict] = None) -> List[int]:
        cfg = config if isinstance(config, dict) else self.default_config
        if not isinstance(cfg, dict) or not cfg.get("client_details"):
            cfg = self.default_config
        cpus: List[int] = []
        for client in cfg.get("client_details") or []:
            if not isinstance(client, dict):
                continue
            try:
                cpu = int(client.get("cpu", 0) or 0)
            except Exception:
                cpu = 0
            if cpu > 0:
                cpus.append(cpu)
        return cpus

    def _valid_selection_values(self, config: Optional[Dict] = None) -> List[int]:
        cpus = self._get_client_cpus(config)
        if len(cpus) < 2:
            return []
        max_cpu = max(cpus)
        valid: List[int] = []
        for threshold in range(max_cpu):
            active = sum(1 for cpu in cpus if cpu > threshold)
            excluded = sum(1 for cpu in cpus if cpu <= threshold)
            if active >= 2 and excluded >= 1:
                valid.append(threshold)
        return valid

    def _fix_selection_value(self, sel_val, config: Optional[Dict] = None) -> Optional[int]:
        valid = self._valid_selection_values(config)
        if not valid:
            return None
        try:
            candidate = int(sel_val)
        except Exception:
            candidate = None
        if candidate in valid:
            return candidate
        return valid[0]

    def _read_previous_ap_state(self) -> Dict[str, str]:
        prev = {
            "client_selector": "OFF",
            "message_compressor": "OFF",
            "heterogeneous_data_handler": "OFF",
        }
        _, last_csv = _sa_latest_round_csv()
        if not last_csv:
            return prev
        try:
            import pandas as pd

            df = pd.read_csv(last_csv)
            ap_col = next((c for c in df.columns if isinstance(c, str) and c.startswith("AP List")), None)
            if ap_col is None or len(df) == 0:
                return prev

            cell = str(df.iloc[-1][ap_col])
            m_keys = re.search(r"\((?P<inside>.*?)\)", ap_col)
            keys = [k.strip() for k in m_keys.group("inside").split(",")] if m_keys else []
            m_vals = re.search(r"\{(?P<inside>.*?)\}", cell)
            states = [s.strip().upper() for s in m_vals.group("inside").split(",")] if m_vals else []
            mapping = dict(zip(keys, states))

            prev["client_selector"] = mapping.get("client_selector", "OFF")
            prev["message_compressor"] = mapping.get("message_compressor", "OFF")
            prev["heterogeneous_data_handler"] = mapping.get("heterogeneous_data_handler", "OFF")
        except Exception:
            pass
        return prev

    def _build_runtime_config(self) -> Dict:
        runtime_cfg = copy.deepcopy(self.default_config)
        runtime_patterns = runtime_cfg.setdefault("patterns", {})
        cached_patterns = (self.cached_config or {}).get("patterns", {})
        for pattern_name, pattern_cfg in cached_patterns.items():
            runtime_patterns.setdefault(pattern_name, {})
            runtime_patterns[pattern_name]["enabled"] = bool(pattern_cfg.get("enabled", False))
            runtime_patterns[pattern_name]["params"] = copy.deepcopy(pattern_cfg.get("params", {}))
        round_client_ids = set(self._last_round_client_ids or [])
        if round_client_ids:
            filtered_client_details = []
            for client in runtime_cfg.get("client_details") or []:
                if not isinstance(client, dict):
                    continue
                client_id = str(client.get("client_id", "")).strip()
                if client_id in round_client_ids:
                    filtered_client_details.append(client)
            if filtered_client_details:
                runtime_cfg["client_details"] = filtered_client_details
                runtime_cfg["clients"] = len(filtered_client_details)
                runtime_cfg["clients_per_round"] = len(filtered_client_details)
        return runtime_cfg

    def _decide_random(self, base_config: Dict, current_round: int) -> Tuple[Dict, List[str]]:
        new_config = copy.deepcopy(base_config)
        logs: List[str] = []
        for pattern in PATTERNS:
            random_state = random.choice([True, False])
            new_config["patterns"][pattern]["enabled"] = random_state
        logs.append(f"[ROUND {current_round}] Random policy executed")
        return new_config, logs


    def _decide_next_config(self, base_config: Dict, current_round: int) -> Tuple[Dict, List[str]]:
        pol = (self.policy or "").lower().strip()
        if not pol or "none" in pol or "off" in pol:
            return copy.deepcopy(base_config), [f"[ROUND {current_round}] Adaptation disabled ('{self.policy}')"]
        if "single" in pol:
            return self._decide_single_agent(base_config, current_round)
        if "random" in pol:
            return self._decide_random(base_config, current_round)
        if "voting" in pol:
            return self._decide_voting(base_config, current_round)
        if "role" in pol:
            return self._decide_role(base_config, current_round)
        if "debate" in pol:
            return self._decide_debate(base_config, current_round)
        if "expert-driven" in pol:
            return self._decide_expert_driven(base_config, current_round)
        fallback_logs = [f"[ROUND {current_round}] Policy '{self.policy}' not recognized or inactive (no changes)"]
        return copy.deepcopy(base_config), fallback_logs

    def _decide_expert_driven(self, base_config: Dict, current_round: int) -> Tuple[Dict, List[str]]:
        new_config = copy.deepcopy(base_config)
        logs: List[str] = [f"[ROUND {current_round}] Expert-Driven policy executed"]
        
        last_r, last_f = _sa_latest_round_csv()
        if not last_f:
            logs.append("No metrics CSV found, retaining previous config.")
            return new_config, logs
        
        import pandas as pd
        try:
            df = pd.read_csv(last_f)
            
            col_round = next((c for c in df.columns if "round" in str(c).lower().strip()), None)
            
            if col_round and len(df) > 0:
                last_r_val = int(df[col_round].dropna().astype(int).max())
                df_r = df[df[col_round].astype(int) == last_r_val].copy()
                df_prev = df[df[col_round].astype(int) == (last_r_val - 1)].copy()
            else:
                df_r = df.copy()
                df_prev = pd.DataFrame()
                
            agg_r = _sa_aggregate_round(df_r)
            agg_prev = _sa_aggregate_round(df_prev) if not df_prev.empty else {}
            
        except Exception as e:
            logs.append(f"Error reading metrics: {e}")
            return new_config, logs
        
        def get_val(agg_dict, key, default=0.0):
            v = agg_dict.get(key)
            return float(v) if v is not None else default

        f1_r = get_val(agg_r, "mean_f1")
        time_r = get_val(agg_r, "mean_total_time", 1.0)
        jsd_r = get_val(agg_r, "mean_jsd")
        comm_r = get_val(agg_r, "mean_comm_time")

        f1_prev = get_val(agg_prev, "mean_f1")
        time_prev = get_val(agg_prev, "mean_total_time", 1.0)
        jsd_prev = get_val(agg_prev, "mean_jsd")
        comm_prev = get_val(agg_prev, "mean_comm_time")

        rate_r = f1_r / time_r if time_r > 0 else 0
        rate_prev = f1_prev / time_prev if time_prev > 0 else 0
        
        if rate_r < rate_prev:
            new_config["patterns"]["client_selector"]["enabled"] = True
            logs.append(f"Client Selector: ON (Rate {rate_r:.4f} < {rate_prev:.4f})")
        else:
            new_config["patterns"]["client_selector"]["enabled"] = False
            logs.append(f"Client Selector: OFF (Rate {rate_r:.4f} >= {rate_prev:.4f})")

        if jsd_r < jsd_prev and f1_r < f1_prev:
            new_config["patterns"]["heterogeneous_data_handler"]["enabled"] = True
            logs.append(f"Heterogeneous Data Handler: ON (JSD {jsd_r:.3f}<{jsd_prev:.3f} AND F1 {f1_r:.3f}<{f1_prev:.3f})")
        else:
            new_config["patterns"]["heterogeneous_data_handler"]["enabled"] = False
            logs.append(f"Heterogeneous Data Handler: OFF (Condition not met: JSD {jsd_r:.3f} vs {jsd_prev:.3f}, F1 {f1_r:.3f} vs {f1_prev:.3f})")

        if comm_r > comm_prev:
            new_config["patterns"]["message_compressor"]["enabled"] = True
            logs.append(f"Message Compressor: ON (Comm {comm_r:.3f} > {comm_prev:.3f})")
        else:
            new_config["patterns"]["message_compressor"]["enabled"] = False
            logs.append(f"Message Compressor: OFF (Comm {comm_r:.3f} <= {comm_prev:.3f})")

        return new_config, logs

    def config_next_round(self, metrics_history: Dict, last_round_time: float):
        t_agents_start = time.perf_counter()
        self.adaptation_time = 0.0
        if not self.enabled:
            return self.default_config["patterns"]
        current_round = self._infer_current_round(metrics_history)
        if current_round <= 0:
            current_round = 1
        self._last_round_info = current_round
        try:
            _, last_csv = _sa_latest_round_csv()
            if last_csv:
                import pandas as pd
                self._last_round_agg = _sa_aggregate_round(pd.read_csv(last_csv))
            else:
                self._last_round_agg = {}
        except Exception:
            self._last_round_agg = {}
        is_last_round = current_round >= self.total_rounds
        if is_last_round:
            log(INFO, f"[ROUND {current_round}] Final round reached ({current_round}/{self.total_rounds}). No adaptation for next round.")
            return self.cached_config["patterns"]

        base_config = self._build_runtime_config()


        next_config, decision_logs = self._decide_next_config(base_config, current_round)
        t_agents_finish = time.perf_counter()
        t_agent =  t_agents_finish - t_agents_start
        self.adaptation_time = t_agent
        
        for line in decision_logs:
            log(INFO, line)

        try:
            _persist_round_rationales(current_round, self.policy, decision_logs)
        except Exception as exc:
            log(INFO, f"[Rationale Export] ERROR: {exc!r}")

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
