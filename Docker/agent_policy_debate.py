import copy
import json
import os
from typing import Dict, List, Tuple


class DebateAgentPolicyMixin:
    def _decide_debate(self, base_config: Dict, current_round: int) -> Tuple[Dict, List[str]]:
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
        logs: List[str] = []

        try:
            N_TURNS = int(os.environ.get("DEBATE_TURNS", "5"))
        except Exception:
            N_TURNS = 5

        last_round = getattr(self, "_last_round_info", None) or {}
        agg = getattr(self, "_last_round_agg", None) or {}
        try:
            ap_prev = self._read_previous_ap_state()
        except Exception:
            ap_prev = {}

        temps = [0.7, 0.8, 0.9]

        consensus_reached = False
        final_decisions_last_turn: List[Dict] = []
        debate_history_lines: List[str] = []

        last_cs_consensus = last_mc_consensus = last_hh_consensus = False
        last_cs_majority = last_mc_majority = last_hh_majority = None

        for turn in range(1, N_TURNS + 1):
            agent_decisions: List[Dict] = []
            agent_rationales: List[str] = []

            for i, t in enumerate(temps, start=1):
                try:
                    base_prompt = _sa_build_prompt(MODE, base_config, last_round, agg, ap_prev)

                    history_text = "\n".join(debate_history_lines).strip()
                    if history_text:
                        history_block = (
                            "## Debate history so far\n"
                            + history_text + "\n\n"
                            + f"You are Agent {i} in turn {turn} of the debate. "
                              "Take this history into account, but still output a single JSON object as your final decision.\n\n"
                        )
                        marker = "## Decision"
                        if marker in base_prompt:
                            prompt = base_prompt.replace(marker, history_block + marker, 1)
                        else:
                            prompt = base_prompt + "\n\n" + history_block + "\n## Decision\n"
                    else:
                        prompt = base_prompt

                    options = {"temperature": t, "top_p": 0.95, "num_ctx": 8192}
                    raw = _sa_call_ollama(MODEL, prompt, self.sa_ollama_urls, force_json=True, options=options)
                    d_i, r_i, _ = _sa_parse_output(raw)

                    d_i = d_i or {}
                    r_i = (r_i or "").strip()
                    agent_decisions.append(d_i)
                    agent_rationales.append(r_i)

                    prev_cs = ap_prev.get("client_selector", "OFF")
                    prev_mc = ap_prev.get("message_compressor", "OFF")
                    prev_hh = ap_prev.get("heterogeneous_data_handler", "OFF")
                    cs_new = (d_i.get("client_selector", "OFF") or "OFF").upper()
                    mc_new = (d_i.get("message_compressor", "OFF") or "OFF").upper()
                    hh_new = (d_i.get("heterogeneous_data_handler", "OFF") or "OFF").upper()

                    logs.append(
                        f"[Debate][Turn {turn}] Agent {i} Decision ({MODEL}, temp={t}): "
                        f"CS: {prev_cs}->{cs_new}; MC: {prev_mc}->{mc_new}; HDH: {prev_hh}->{hh_new}"
                    )
                    if r_i:
                        logs.append(f"[Debate][Turn {turn}] Rationale A{i}: {r_i}")
                    logs.append("")

                    hist_line = (
                        f"[Turn {turn} / Agent {i}] CS: {prev_cs}->{cs_new}; "
                        f"MC: {prev_mc}->{mc_new}; HDH: {prev_hh}->{hh_new}"
                    )
                    debate_history_lines.append(hist_line)
                    if r_i:
                        short_rat = r_i if len(r_i) <= 600 else (r_i[:600] + " ... [truncated]")
                        debate_history_lines.append(f"[Turn {turn} / Agent {i} Rationale] {short_rat}")

                except Exception as e:
                    agent_decisions.append({})
                    agent_rationales.append("")
                    logs.append(f"[Debate][Turn {turn}] Agent {i} ERROR: {e!r}")
                    logs.append("")

            def _onoff(d: Dict, key: str) -> str:
                return (d.get(key, "OFF") or "OFF").upper()

            cs_vals = [_onoff(d, "client_selector") for d in agent_decisions]
            mc_vals = [_onoff(d, "message_compressor") for d in agent_decisions]
            hh_vals = [_onoff(d, "heterogeneous_data_handler") for d in agent_decisions]

            def _majority(vals: List[str]):
                on_count = vals.count("ON")
                off_count = vals.count("OFF")
                if on_count > off_count:
                    return True, "ON"
                if off_count > on_count:
                    return True, "OFF"
                return False, None

            cs_consensus, cs_majority = _majority(cs_vals)
            mc_consensus, mc_majority = _majority(mc_vals)
            hh_consensus, hh_majority = _majority(hh_vals)

            final_decisions_last_turn = agent_decisions
            last_cs_consensus, last_mc_consensus, last_hh_consensus = cs_consensus, mc_consensus, hh_consensus
            last_cs_majority, last_mc_majority, last_hh_majority = cs_majority, mc_majority, hh_majority

            if cs_consensus and mc_consensus and hh_consensus:
                consensus_reached = True
                break

        if consensus_reached:
            prev_cs_enabled = bool(
                base_config.get("patterns", {}).get("client_selector", {}).get("enabled")
            )
            prev_mc_enabled = bool(
                base_config.get("patterns", {}).get("message_compressor", {}).get("enabled")
            )
            prev_hh_enabled = bool(
                base_config.get("patterns", {}).get("heterogeneous_data_handler", {}).get("enabled")
            )

            if last_cs_consensus and last_cs_majority is not None:
                cs_final = last_cs_majority
            else:
                cs_final = "ON" if prev_cs_enabled else "OFF"

            if last_mc_consensus and last_mc_majority is not None:
                mc_final = last_mc_majority
            else:
                mc_final = "ON" if prev_mc_enabled else "OFF"

            if last_hh_consensus and last_hh_majority is not None:
                hh_final = last_hh_majority
            else:
                hh_final = "ON" if prev_hh_enabled else "OFF"

            sel_val = None
            if cs_final == "ON":
                if last_cs_consensus and last_cs_majority == "ON":
                    vals: List[int] = []
                    for d in agent_decisions:
                        try:
                            if (d.get("client_selector") or "").upper() == "ON" and "selection_value" in d:
                                vals.append(int(d["selection_value"]))
                        except Exception:
                            pass
                    sel_val = max(set(vals), key=vals.count) if vals else 3
                else:
                    try:
                        sel_val = int(
                            base_config
                            .get("patterns", {})
                            .get("client_selector", {})
                            .get("params", {})
                            .get("selection_value", 3)
                        )
                    except Exception:
                        sel_val = 3

                sel_val = self._fix_selection_value(sel_val, base_config)
                if sel_val is None:
                    cs_final = "OFF"
                    logs.append("[Debate] CS majority reached ON, but no safe selection_value exists for the current CPU distribution. CS kept OFF.")


            new_cfg = copy.deepcopy(base_config)
            pats = new_cfg.setdefault("patterns", {})
            pats.setdefault("client_selector", {})
            pats.setdefault("message_compressor", {})
            pats.setdefault("heterogeneous_data_handler", {})

            pats["client_selector"]["enabled"] = (cs_final == "ON")
            pats["message_compressor"]["enabled"] = (mc_final == "ON")
            pats["heterogeneous_data_handler"]["enabled"] = (hh_final == "ON")

            if cs_final == "ON" and sel_val is not None:
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

            logs.append(
                f"[Debate] Consensus reached at turn {turn}/{N_TURNS}. "
                f"Final decision: CS={cs_final}; MC={mc_final}; HDH={hh_final}."
            )

            self.update_json(new_cfg)
            self.update_config(new_cfg)

            if str(os.environ.get("AGENT_LOG_TO_FILE", "0")).lower() in ("1", "true", "yes"):
                _append_agent_log(logs)

            return new_cfg, logs

        if any(flag for flag in (last_cs_consensus, last_mc_consensus, last_hh_consensus)):
            prev_cs_enabled = bool(
                base_config.get("patterns", {}).get("client_selector", {}).get("enabled")
            )
            prev_mc_enabled = bool(
                base_config.get("patterns", {}).get("message_compressor", {}).get("enabled")
            )
            prev_hh_enabled = bool(
                base_config.get("patterns", {}).get("heterogeneous_data_handler", {}).get("enabled")
            )

            cs_final = last_cs_majority if last_cs_consensus and last_cs_majority is not None else ("ON" if prev_cs_enabled else "OFF")
            mc_final = last_mc_majority if last_mc_consensus and last_mc_majority is not None else ("ON" if prev_mc_enabled else "OFF")
            hh_final = last_hh_majority if last_hh_consensus and last_hh_majority is not None else ("ON" if prev_hh_enabled else "OFF")

            sel_val = None
            if cs_final == "ON":
                vals: List[int] = []
                for d in final_decisions_last_turn:
                    try:
                        if (d.get("client_selector") or "").upper() == "ON" and "selection_value" in d:
                            vals.append(int(d["selection_value"]))
                    except Exception:
                        pass
                fallback_sel = max(set(vals), key=vals.count) if vals else base_config.get("patterns", {}).get("client_selector", {}).get("params", {}).get("selection_value", 2)
                sel_val = self._fix_selection_value(fallback_sel, base_config)
                if sel_val is None:
                    cs_final = "OFF"
                    logs.append("[Debate] Partial-majority CS decision rejected because no safe selection_value exists for the current CPU distribution.")

            new_cfg = copy.deepcopy(base_config)
            pats = new_cfg.setdefault("patterns", {})
            pats.setdefault("client_selector", {})
            pats.setdefault("message_compressor", {})
            pats.setdefault("heterogeneous_data_handler", {})
            pats["client_selector"]["enabled"] = (cs_final == "ON")
            pats["message_compressor"]["enabled"] = (mc_final == "ON")
            pats["heterogeneous_data_handler"]["enabled"] = (hh_final == "ON")

            if cs_final == "ON" and sel_val is not None:
                base_params = (
                    base_config
                    .get("patterns", {})
                    .get("client_selector", {})
                    .get("params", {})
                )
                pats["client_selector"]["params"] = {
                    "selection_strategy": base_params.get("selection_strategy", "Resource-Based"),
                    "selection_criteria": base_params.get("selection_criteria", "CPU"),
                    "selection_value": sel_val,
                }

            logs.append(
                f"[Debate] No full consensus after {N_TURNS} turns, but applying available strict majorities. "
                f"Final decision: CS={cs_final}; MC={mc_final}; HDH={hh_final}."
            )
            self.update_json(new_cfg)
            self.update_config(new_cfg)

            if str(os.environ.get("AGENT_LOG_TO_FILE", "0")).lower() in ("1", "true", "yes"):
                _append_agent_log(logs)

            return new_cfg, logs

        logs.append(
            f"[Debate] No consensus after {N_TURNS} turns. "
            "Keeping previous architectural pattern configuration."
        )
        self.update_json(base_config)
        self.update_config(base_config)

        if str(os.environ.get("AGENT_LOG_TO_FILE", "0")).lower() in ("1", "true", "yes"):
            _append_agent_log(logs)

        return base_config, logs
