import copy
import json
import os
from typing import Dict, List, Tuple


class VotingAgentPolicyMixin:
    def _decide_voting(self, base_config: Dict, current_round: int) -> Tuple[Dict, List[str]]:
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
        MODEL, MODE = self.sa_model, "few-shot"
        COORD_MODEL = MODEL
        logs: List[str] = []

        last_round = getattr(self, "_last_round_info", None) or {}
        agg = getattr(self, "_last_round_agg", None) or {}
        try:
            ap_prev = self._read_previous_ap_state()
        except Exception:
            ap_prev = {}

        agent_decisions: List[Dict] = []
        agent_rationales: List[str] = []
        temps = [0.7, 0.8, 0.9]
        for i, t in enumerate(temps, start=1):
            try:
                d_i, r_i = _sa_generate_with_retry(
                    MODEL,
                    MODE,
                    base_config,
                    last_round,
                    agg,
                    ap_prev,
                    self.sa_ollama_urls,
                    options={"temperature": t, "top_p": 0.95, "num_ctx": 8192},
                )
                agent_decisions.append(d_i or {})
                agent_rationales.append((r_i or "").strip())

                prev_cs = ap_prev.get("client_selector", "OFF")
                prev_mc = ap_prev.get("message_compressor", "OFF")
                prev_hh = ap_prev.get("heterogeneous_data_handler", "OFF")
                cs_new = (d_i or {}).get("client_selector", "OFF")
                mc_new = (d_i or {}).get("message_compressor", "OFF")
                hh_new = (d_i or {}).get("heterogeneous_data_handler", "OFF")

                logs.append(
                    f"[Agent {i}] Decision ({MODEL}, temp={t}): CS: {prev_cs}→{cs_new} • MC: {prev_mc}→{mc_new} • HDH: {prev_hh}→{hh_new}"
                )
                if r_i:
                    logs.append(f"[Rationale A{i}] {r_i}")
                logs.append("")
            except Exception as e:
                agent_decisions.append({})
                agent_rationales.append("")
                logs.append(f"[Agent {i}] ERROR: {e!r}")
                logs.append("")

        def _vote(pattern: str) -> int:
            return sum(1 for d in agent_decisions if (d.get(pattern, "OFF") or "OFF").upper() == "ON")

        maj_cs  = "ON" if _vote("client_selector") >= 2 else "OFF"
        maj_mc  = "ON" if _vote("message_compressor") >= 2 else "OFF"
        maj_hdh = "ON" if _vote("heterogeneous_data_handler") >= 2 else "OFF"

        logs.append(f"[Coordinator] Majority: CS: ·→{maj_cs} • MC: ·→{maj_mc} • HDH: ·→{maj_hdh}")
        logs.append("")
        logs.append(f"[Final Decision] CS={maj_cs}; MC={maj_mc}; HDH={maj_hdh}")
        logs.append("")

        try:
            raw = _sa_call_ollama(
                COORD_MODEL,
                _sa_build_prompt(MODE, base_config, last_round, agg, ap_prev)
                + "\n\n### Voting Summary\n"
                + "\n".join([f"- A{i}: {d}" for i, d in enumerate(agent_decisions, start=1)])
                + "\n"
                + f"- Majority: CS={maj_cs}; MC={maj_mc}; HDH={maj_hdh}\n"
                + "Provide a short rationale for the majority outcome. Do NOT restate a final decision line.",
                self.sa_ollama_urls,
                force_json=True,
                options={"temperature": 0.2, "top_p": 0.9, "num_ctx": 8192},
            )
            coord_dec, coord_rat, _ = _sa_parse_output(raw)
            import re as _re

            text = coord_rat or (coord_dec or {}).get("rationale", "") or ""
            if text:
                sanitized_lines = []
                for ln in str(text).splitlines():
                    if _re.match(
                        r"^\s*CS\s*=\s*(ON|OFF)\s*;\s*MC\s*=\s*(ON|OFF)\s*;\s*HDH\s*=\s*(ON|OFF)\s*$",
                        ln,
                        _re.I,
                    ):
                        continue
                    sanitized_lines.append(ln.strip())
                sanitized = " ".join([s for s in sanitized_lines if s]).strip()
                if sanitized:
                    logs.append(f"[Coordinator Rationale] {sanitized}")
                else:
                    logs.append("[Coordinator Rationale] (omitted: only decision-like lines)")
        except Exception as e:
            logs.append(f"[Coordinator] LLM consult ERROR: {e!r}. Majority kept.")
        logs.append("")

        new_cfg = copy.deepcopy(base_config)
        pats = new_cfg.setdefault("patterns", {})
        pats.setdefault("client_selector", {})
        pats.setdefault("message_compressor", {})
        pats.setdefault("heterogeneous_data_handler", {})

        pats["client_selector"]["enabled"] = (maj_cs == "ON")
        pats["message_compressor"]["enabled"] = (maj_mc == "ON")
        pats["heterogeneous_data_handler"]["enabled"] = (maj_hdh == "ON")

        if maj_cs == "ON":
            vals = []
            for d in agent_decisions:
                if (d.get("client_selector") or "").upper() == "ON" and "selection_value" in d:
                    try:
                        vals.append(int(d["selection_value"]))
                    except Exception:
                        pass
            sel_val = max(set(vals), key=vals.count) if vals else None
            sel_val = self._fix_selection_value(sel_val, base_config)
            if sel_val is None:
                pats["client_selector"]["enabled"] = False
                logs.append("[Coordinator] CS requested ON but no safe selection_value exists for the current CPU distribution. CS kept OFF.")
                sel_val = None

            if sel_val is not None:
                base_params = (
                    base_config
                    .get("patterns", {})
                    .get("client_selector", {})
                    .get("params", {})
                )
                strategy = base_params.get("selection_strategy", "Resource-Based")
                criteria = base_params.get("selection_criteria", "CPU")

                pats["client_selector"]["params"] = {
                    "selection_strategy": strategy,
                    "selection_criteria": criteria,
                    "selection_value": sel_val,
                }
        self.update_json(new_cfg)
        self.update_config(new_cfg)

        if str(os.environ.get("AGENT_LOG_TO_FILE", "0")).lower() in ("1", "true", "yes"):
            _append_agent_log(logs)

        return new_cfg, logs
