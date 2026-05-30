import copy
import json
import os
from typing import Dict, List, Tuple


class SingleAgentPolicyMixin:
    def _decide_single_agent(self, base_config: Dict, current_round: int) -> Tuple[Dict, List[str]]:
        from adaptation import (
            PATTERNS,
            _append_agent_log,
            _sa_aggregate_round,
            _sa_build_prompt,
            _sa_call_ollama,
            _sa_generate_with_retry,
            _sa_latest_round_csv,
            _sa_mode_from_policy,
            _sa_parse_output,
        )
        import re, json

        logs: List[str] = []
        last_round, last_csv = _sa_latest_round_csv()
        agg = {}
        if last_csv:
            try:
                import pandas as pd
                df = pd.read_csv(last_csv)
                agg = _sa_aggregate_round(df)
            except Exception:
                pass

        ap_prev = self._read_previous_ap_state()

        mode = _sa_mode_from_policy(self.policy)
        try:
            decisions, rationale = _sa_generate_with_retry(
                self.sa_model, mode, base_config, last_round, agg, ap_prev, self.sa_ollama_urls
            )
        except Exception:
            return copy.deepcopy(base_config), logs

        prev_cs  = "✅" if ap_prev.get("client_selector") == "ON" else "❌"
        prev_mc  = "✅" if ap_prev.get("message_compressor") == "ON" else "❌"
        prev_hdh = "✅" if ap_prev.get("heterogeneous_data_handler") == "ON" else "❌"

        new_cs = "✅" if decisions.get("client_selector") == "ON" else "❌"
        new_mc = "✅" if decisions.get("message_compressor") == "ON" else "❌"
        new_hdh = "✅" if decisions.get("heterogeneous_data_handler") == "ON" else "❌"

        delta = " • ".join([f"CS: {prev_cs}→{new_cs}", f"MC: {prev_mc}→{new_mc}", f"HDH: {prev_hdh}→{new_hdh}"])
        logs.append(f"[Single-Agent] Decision ({self.sa_model}): {delta}")

        r_print = (rationale or "").strip()
        if r_print:
            try:
                obj = json.loads(r_print)
                if isinstance(obj, dict) and isinstance(obj.get("rationale"), str):
                    r_print = obj["rationale"].strip()
            except Exception:
                m = re.search(r'"rationale"\s*:\s*"(?P<r>.*?)"', r_print, flags=re.S)
                if m:
                    r_print = m.group("r").strip()

        logs.append(f"[Rationale] {r_print if r_print else 'This model does not support rationale generation.'}")

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
                    sel_val = 3

            sel_val = self._fix_selection_value(sel_val, base_config)
            if sel_val is None:
                new_config["patterns"]["client_selector"]["enabled"] = False
                logs.append("[Single-Agent] CS suggested ON but no safe selection_value exists for the current CPU distribution. CS kept OFF.")
                return new_config, logs

            new_config["patterns"]["client_selector"]["params"] = {
                "selection_strategy": "Resource-Based",
                "selection_criteria": "CPU",
                "selection_value": sel_val
            }

        return new_config, logs
