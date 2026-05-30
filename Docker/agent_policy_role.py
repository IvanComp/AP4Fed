import copy
import json
import os
from typing import Dict, List, Tuple


class RoleAgentPolicyMixin:
    def _decide_role(self, base_config: Dict, current_round: int) -> Tuple[Dict, List[str]]:
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

        # contesto (coerente con gli altri approcci)
        last_round = getattr(self, "_last_round_info", None) or {}
        agg = getattr(self, "_last_round_agg", None) or {}
        try:
            ap_prev = self._read_previous_ap_state()
        except Exception:
            ap_prev = {}

        # mapping specialisti: etichetta di log, chiave di focus, temperatura
        specs = [
            ("CS",  "client_selector",            1.0),
            ("MC",  "message_compressor",         1.0),
            ("HDH", "heterogeneous_data_handler", 1.0),
        ]

        # raccolta output specialisti (filtrato per ruolo)
        specialist = { "client_selector": {}, "message_compressor": {}, "heterogeneous_data_handler": {} }
        specialist_rat = { "client_selector": "", "message_compressor": "", "heterogeneous_data_handler": "" }

        # util per ON/OFF
        def _onoff(x): return "ON" if str(x).strip().upper() == "ON" else "OFF"

        # valori precedenti per frecce nei log
        prev_cs  = (ap_prev.get("client_selector", "OFF") or "OFF").upper()
        prev_mc  = (ap_prev.get("message_compressor", "OFF") or "OFF").upper()
        prev_hdh = (ap_prev.get("heterogeneous_data_handler", "OFF") or "OFF").upper()

        for label, key_focus, temp in specs:
            try:
                # prompt base "di sempre"
                base_p = _sa_build_prompt(MODE, base_config, last_round, agg, ap_prev)

                # blocco Role Focus (molto rigido)
                role_focus = (
                    "\n\n### Role Focus\n"
                    f"You are the specialist for '{key_focus}'.\n"
                    f"- Decide ONLY '{key_focus}' as ON or OFF.\n"
                    "- For the other patterns, copy their values EXACTLY from PrevAP into your JSON.\n"
                    "- Your 'rationale' MUST discuss ONLY the focused pattern; do NOT ever mention the other patterns.\n"
                    "- If you set 'client_selector' to ON, you MUST derive 'selection_value' from the CPUs of the clients listed in the current System Configuration, as explained in the Guardrails, and choose a threshold that excludes only the slowest clients without killing the round.\n"
                    "- If the focused pattern has been OFF in several recent rounds while the typical conditions for using it are present (for example, high communication time for MC, non-IID clients for HDH, or very high round time with slow clients for CS), treating it as always OFF is a mistake and you should seriously consider turning it ON.\n"
                    "- Before finalizing 'selection_value', simulate its effect over cpu_per_client: DO NOT choose any value that would leave fewer than two clients with CPU > selection_value.\n"
                    "Output STRICTLY one JSON object with EXACTLY these keys:\n"
                    "  'client_selector', 'selection_value' (ONLY if client_selector='ON'), \n"
                    "'message_compressor', 'heterogeneous_data_handler', 'rationale'.\n"
                    "Do not add extra keys. Do not use markdown fences."
                )
                prompt = base_p + role_focus

                raw = _sa_call_ollama(
                    MODEL,
                    prompt,
                    self.sa_ollama_urls,
                    force_json=True,
                    options={"temperature": temp, "top_p": 0.95, "num_ctx": 8192}
                )
                d_i, r_i, _ = _sa_parse_output(raw)
                d_i = d_i or {}
                r_i = (r_i or "").strip()
                cs_val  = prev_cs
                mc_val  = prev_mc
                hdh_val = prev_hdh
                if key_focus == "client_selector":
                    cs_val = _onoff(d_i.get("client_selector", prev_cs))
                elif key_focus == "message_compressor":
                    mc_val = _onoff(d_i.get("message_compressor", prev_mc))
                else:  # HDH
                    hdh_val = _onoff(d_i.get("heterogeneous_data_handler", prev_hdh))

                # costruisci l’oggetto filtrato consegnato al coordinator (owner per pattern)
                out = {
                    "client_selector": cs_val,
                    "message_compressor": mc_val,
                    "heterogeneous_data_handler": hdh_val
                }
                # selection_value solo se focus è CS e CS=ON
                if key_focus == "client_selector" and cs_val == "ON":
                    try:
                        sv = d_i.get("selection_value", None)
                        if sv is not None:
                            out["selection_value"] = int(sv)
                    except Exception:
                        pass

                specialist[key_focus] = out
                specialist_rat[key_focus] = r_i

                # log come nel tuo esempio: solo la riga del pattern di focus
                if key_focus == "client_selector":
                    logs.append(f"[Agent CS] Decision ({MODEL}): CS: ·→{'✅' if cs_val=='ON' else '❌'}")
                    if r_i:
                        logs.append(f"[Rationale CS] {r_i}")
                elif key_focus == "message_compressor":
                    logs.append(f"[Agent MC] Decision ({MODEL}): MC: ·→{'✅' if mc_val=='ON' else '❌'}")
                    if r_i:
                        logs.append(f"[Rationale MC] {r_i}")
                else:
                    logs.append(f"[Agent HDH] Decision ({MODEL}): HDH: ·→{'✅' if hdh_val=='ON' else '❌'}")
                    if r_i:
                        logs.append(f"[Rationale HDH] {r_i}")
                logs.append("")  # spazio tra agenti

            except Exception as e:
                # in caso di errore: mantieni il valore prev per quel ruolo
                if key_focus == "client_selector":
                    specialist[key_focus] = {"client_selector": prev_cs, "message_compressor": prev_mc, "heterogeneous_data_handler": prev_hdh}
                elif key_focus == "message_compressor":
                    specialist[key_focus] = {"client_selector": prev_cs, "message_compressor": prev_mc, "heterogeneous_data_handler": prev_hdh}
                else:
                    specialist[key_focus] = {"client_selector": prev_cs, "message_compressor": prev_mc, "heterogeneous_data_handler": prev_hdh}
                specialist_rat[key_focus] = ""
                logs.append(f"[Agent {label}] ERROR: {e!r}")
                logs.append("")

        # Coordinator deterministico: merge per ruolo (no voting, no LLM)
        cs_final  = specialist["client_selector"]["client_selector"]
        mc_final  = specialist["message_compressor"]["message_compressor"]
        hdh_final = specialist["heterogeneous_data_handler"]["heterogeneous_data_handler"]

        sel_val = None
        if cs_final == "ON":
            v = specialist["client_selector"].get("selection_value", None)
            try:
                if v is not None:
                    sel_val = int(v)
            except Exception:
                sel_val = None
            if sel_val is None:
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
                logs.append("[Coordinator] Role-Merge rejected CS because no safe selection_value exists for the current CPU distribution.")

        # Log coordinator come nel tuo format
        logs.append("[Coordinator] Role-Merge: " +
                    f"CS: ·→{'✅' if cs_final=='ON' else '❌'} • "
                    f"MC: ·→{'✅' if mc_final=='ON' else '❌'} • "
                    f"HDH: ·→{'✅' if hdh_final=='ON' else '❌'}")
        logs.append("")
        logs.append("[Coordinator Verdict]")
        logs.append("Mechanism: Role-Based (3 agents specialized: CS, MC, HDH). Coordinator merges per-role outputs; no voting.")
        logs.append("Roles — CS agent decides Client Selector (and selection_value); MC agent decides Message Compressor; HDH agent decides Heterogeneous Data Handler.")
        logs.append(f"Final decision — CS={'ON' if cs_final=='ON' else 'OFF'}; MC={'ON' if mc_final=='ON' else 'OFF'}; HDH={'ON' if hdh_final=='ON' else 'OFF'}.")
        logs.append("Rationale: each agent focused solely on its assigned pattern; any extra fields from an agent were ignored by design.")
        logs.append("")

        new_cfg = copy.deepcopy(base_config)
        pats = new_cfg.setdefault("patterns", {})
        pats.setdefault("client_selector", {})
        pats.setdefault("message_compressor", {})
        pats.setdefault("heterogeneous_data_handler", {})

        pats["client_selector"]["enabled"] = (cs_final == "ON")
        pats["message_compressor"]["enabled"] = (mc_final == "ON")
        pats["heterogeneous_data_handler"]["enabled"] = (hdh_final == "ON")

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

        self.update_json(new_cfg)
        self.update_config(new_cfg)


        if str(os.environ.get("AGENT_LOG_TO_FILE", "0")).lower() in ("1", "true", "yes"):
            _append_agent_log(logs)

        return new_cfg, logs
