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
PERFORMANCE_DIR = os.environ.get("AP4FED_PERFORMANCE_DIR", os.path.join(os.getcwd(), "performance"))
RATIONALE_CSV_FILE = os.environ.get(
    "AP4FED_RATIONALE_CSV_FILE",
    os.path.join(PERFORMANCE_DIR, "FLwithAP_adaptation_rationales.csv"),
)

def _append_agent_log(lines):
    p = AGENT_LOG_FILE
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip("\n") + "\n")


def _extract_rationale_entries(round_idx: int, policy: str, decision_logs: List[str]) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    patterns = [
        (
            re.compile(r"^\[Rationale\]\s*(?P<text>.*)$", re.S),
            lambda m: {
                "round": str(round_idx),
                "policy": policy,
                "tag": "[Rationale]",
                "turn": "",
                "agent": "",
                "role": "",
                "rationale": m.group("text").strip(),
            },
        ),
        (
            re.compile(r"^\[Rationale A(?P<agent>\d+)\]\s*(?P<text>.*)$", re.S),
            lambda m: {
                "round": str(round_idx),
                "policy": policy,
                "tag": f"[Rationale A{m.group('agent')}]",
                "turn": "",
                "agent": m.group("agent"),
                "role": "",
                "rationale": m.group("text").strip(),
            },
        ),
        (
            re.compile(r"^\[Rationale (?P<role>CS|MC|HDH)\]\s*(?P<text>.*)$", re.S),
            lambda m: {
                "round": str(round_idx),
                "policy": policy,
                "tag": f"[Rationale {m.group('role')}]",
                "turn": "",
                "agent": "",
                "role": m.group("role"),
                "rationale": m.group("text").strip(),
            },
        ),
        (
            re.compile(r"^\[Coordinator Rationale\]\s*(?P<text>.*)$", re.S),
            lambda m: {
                "round": str(round_idx),
                "policy": policy,
                "tag": "[Coordinator Rationale]",
                "turn": "",
                "agent": "",
                "role": "Coordinator",
                "rationale": m.group("text").strip(),
            },
        ),
        (
            re.compile(
                r"^\[Debate\]\[Turn (?P<turn>\d+)\] Rationale A(?P<agent>\d+):\s*(?P<text>.*)$",
                re.S,
            ),
            lambda m: {
                "round": str(round_idx),
                "policy": policy,
                "tag": f"[Debate][Turn {m.group('turn')}] Rationale A{m.group('agent')}",
                "turn": m.group("turn"),
                "agent": m.group("agent"),
                "role": "",
                "rationale": m.group("text").strip(),
            },
        ),
    ]

    for raw_line in decision_logs:
        if raw_line is None:
            continue
        text = str(raw_line).strip()
        if not text:
            continue
        for pattern, factory in patterns:
            match = pattern.match(text)
            if not match:
                continue
            entry = factory(match)
            rationale = entry.get("rationale", "").strip()
            if not rationale or rationale.lower().startswith("(omitted:"):
                break
            entry["rationale"] = rationale
            entries.append(entry)
            break

    return entries


def _persist_round_rationales(round_idx: int, policy: str, decision_logs: List[str]) -> None:
    entries = _extract_rationale_entries(round_idx, policy, decision_logs)
    if not entries:
        return

    os.makedirs(PERFORMANCE_DIR, exist_ok=True)

    fieldnames = ["round", "policy", "tag", "turn", "agent", "role", "rationale"]
    write_header = not os.path.exists(RATIONALE_CSV_FILE)
    with open(RATIONALE_CSV_FILE, "a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(entries)

    round_txt_path = os.path.join(PERFORMANCE_DIR, f"FLwithAP_adaptation_rationales_round{round_idx}.txt")
    with open(round_txt_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(f"Round: {round_idx}\n")
        txt_file.write(f"Policy: {policy}\n\n")
        for entry in entries:
            header_bits = [entry["tag"]]
            if entry["turn"]:
                header_bits.append(f"turn={entry['turn']}")
            if entry["agent"]:
                header_bits.append(f"agent={entry['agent']}")
            if entry["role"]:
                header_bits.append(f"role={entry['role']}")
            txt_file.write(" | ".join(header_bits) + "\n")
            txt_file.write(entry["rationale"] + "\n\n")

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
    # Trova colonne in modo robusto
    def _find(colnames):
        for c in df.columns:
            lc = str(c).strip().lower()
            for pat in colnames:
                if callable(pat):
                    if pat(lc):
                        return c
                else:
                    if pat in lc:
                        return c
        return None

    col_round = _find([lambda s: "round" in s])
    col_cid   = _find([lambda s: ("client" in s and "id" in s) or s.strip() == "client id"])
    col_tr    = _find([lambda s: ("training" in s and "time" in s), "training (s)", "training time (s)", "training time"])
    col_cm    = _find([lambda s: ("comm" in s and "time" in s) or ("communication" in s)])
    col_tt    = _find([lambda s: ("total time of fl round" in s) or ("total" in s and "round" in s)])
    col_f1    = _find([lambda s: "val f1" in s or s == "f1"])
    col_jsd   = _find([lambda s: "jsd" == str(s).strip().lower()])

    # Se la CSV ha più round, prendi l'ultimo; altrimenti usa tutto il df
    dfl = df
    if col_round and col_round in df.columns:
        try:
            last_r = int(df[col_round].dropna().astype(int).max())
            dfl = df[df[col_round].astype(int) == last_r].copy()
        except Exception:
            dfl = df.copy()

    # Righe per-client: dove c'è un Client ID o comunque valori numerici sui tempi
    per_client = dfl.copy()
    if col_cid and col_cid in per_client.columns:
        per_client = per_client[per_client[col_cid].notna()].copy()

    # Serie numeriche pulite
    def _series(dfx, col):
        if not col or col not in dfx.columns:
            return []
        out = []
        for v in dfx[col].tolist():
            try:
                out.append(float(v))
            except Exception:
                pass
        return out

    tr_seq = _series(per_client, col_tr)
    cm_seq = _series(per_client, col_cm)

    # F1 e Total Round Time sono GLOBAL: prendi l'ultimo valore non-NaN nel sottoinsieme del round
    def _last_non_nan(dfx, col):
        if not col or col not in dfx.columns:
            return None
        vals = []
        for v in dfx[col].tolist():
            try:
                vals.append(float(v))
            except Exception:
                pass
        return vals[-1] if vals else None

    f1_last = _last_non_nan(dfl, col_f1)
    tt_last = _last_non_nan(dfl, col_tt)
    jsd_last = _last_non_nan(dfl, col_jsd)

    # Statistiche aggregati per-client (round selezionato)
    def _agg(seq):
        if not seq:
            return {"count": 0, "mean": None, "min": None, "max": None}
        return {
            "count": len(seq),
            "mean": sum(seq) / len(seq),
            "min": min(seq),
            "max": max(seq),
        }

    tr_agg = _agg(tr_seq)
    cm_agg = _agg(cm_seq)

    return {
        "round": int(dfl[col_round].iloc[0]) if col_round and len(dfl) else None,
        "mean_f1": f1_last,                  # GLOBAL Val F1 dell'ultimo round
        "mean_total_time": tt_last,          # GLOBAL Total Time of FL Round dell'ultimo round
        "mean_training_time": tr_agg["mean"],# media per-client del round
        "mean_comm_time": cm_agg["mean"],    # media per-client del round
        "mean_jsd": jsd_last,                # GLOBAL JSD dell'ultimo round
        "training_time_stats": tr_agg,       # include count/min/max
        "comm_time_stats": cm_agg
    }

def _sa_build_prompt(mode: str, config, round_idx, agg, ap_prev):
    def _fmt(x, nd=3):
        try:
            return f"{float(x):.{nd}f}"
        except Exception:
            return "?"
    def _delta(curr, prev, nd=3):
        try:
            c = float(curr)
            p = float(prev)
            diff = c - p
            sign = "+" if diff >= 0 else ""
            return f"{sign}{diff:.{nd}f}"
        except Exception:
            return "?"
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
    mean_jsd  = (agg or {}).get("mean_jsd")
    tr_stats  = (agg or {}).get("training_time_stats") or {}
    cm_stats  = (agg or {}).get("comm_time_stats") or {}

    try:
        round_no = int(round_idx)
    except Exception:
        try:
            round_no = int((agg or {}).get("round"))
        except Exception:
            round_no = None

    def _state(v):
        if isinstance(v, bool):
            return "ON" if v else "OFF"
        return "ON" if str(v).strip().upper() == "ON" else "OFF"

    ap_prev_small = {
        "client_selector": _state((ap_prev or {}).get("client_selector")),
        "message_compressor": _state((ap_prev or {}).get("message_compressor")),
        "heterogeneous_data_handler": _state((ap_prev or {}).get("heterogeneous_data_handler")),
    }

    # 1) Static Context: Role, Task, Guardrails, Output
    instructions = (
        "Architectural Pattern Decision: Prompt\n\n"
        "#Context"
        "\n"
        "## Role: You are an AI agent and expert software architect for Federated Learning systems.\n"
        "At the end of each training round, analyze the current SYSTEM CONFIGURATION and actual system EVALUATION METRICS, then recommend which architectural patterns to turn ON or OFF for the next round to optimize system EVALUATION METRICS.\n"        
        "\n"
        "##Task: Your task is to activate or deactive one or a combination of architectural patterns for the next round of training. The objective is to optimize the system EVALUATION METRICS as much as possible.\n"
        "\n"
        "Ideally, we aim to optimize the following EVALUATION METRICS:\n"
        "1) Maximize Global Model Accuracy or F1 Score, which measures the overall predictive performance of the aggregated model across all clients.\n"
        "2) Minimize Total Round Time, defined as the elapsed time to complete a full federated learning round.\n"
        "3) Minimize Communication Time, representing the duration spent exchanging data between the server and clients during each round.\n"
        "4) Minimize Training Time, referring to the computational time each client requires to perform local model training.\n"
        "\n"
        "Here are the three ARCHITECTURAL PATTERNS that you can consider to (de)activate:\n"
        "1) Client Selector: selects clients that will partecipate to the next round of Federated Learning. The selection is driven by the number of CPUs available by the clients. When active, you must specify 'selection_value': <int>, which is the CPU threshold. Only the clients with CPU > selection_value will be included for the next training round. Critical Note: the value it can take is between 0 and n−1, where n is the number of CPUs present in the client with the largest CPU; otherwise, the system crashes.\n"
        "2) Message Compressor: compresses messages exchanged between the clients and the server.\n"
        "3) Heterogeneous Data Handler: mitigates non-IID effects using class reweighting, augmentation, or distillation.\n"  
        "\n"
        "Architectural Patterns Performance Implications:\n" 
        "Enabling or disabling each architectural pattern has both advantages and disadvantages in terms of performance:\n"
        "- Client Selector (CS): Selects clients with highest number of CPUs to reduce slowdowns caused by clients devices (the Straggler effect). Discarding weaker clients reduces total round time and training time. However, you should keep excluding lower-capacity clients as long as this strategy yields improvements in Val F1 (accuracy); if Val F1 stagnates or decreases, reconsider and gradually re-include them to preserve model diversity. To be activated when there are differences between clients with different CPUs. \n"
        "- Message Compressor (MC): Compresses model updates exchanged between the server and clients to reduce communication time, which is especially beneficial for large models or bandwidth-constrained networks. The downside is the additional computational overhead from compression and decompression, which may outweigh the benefits when dealing with small model sizes. Enable MC when communication time is a clear bottleneck and apply it sparingly, avoiding consecutive rounds.\n"
        "- Heterogeneous Data Handler (HDH): Employs data augmentation to address and mitigate data heterogeneity among non-iid clients, improving global model accuracy and accelerating convergence on non-IID data. The trade-off is an increase in local computation required to generate synthetic data for balancing class distributions, which can add significant overhead. To be activated if non-IID clients are present; consider at most a later follow-up if new non-IID data arrives (New Data).\n"
        "\n"
        "## Guardrails\n"
        "- Do not invent values.\n"
        "- Output only the JSON object plus the rationale, no extra text otherwise the system will fail.\n"
        "- The last line of the 'rationale' must be exactly the signature: CS=<ON|OFF>; MC=<ON|OFF>; HDH=<ON|OFF>. These values must copy the JSON decisions above, in the same order and casing.\n"
        "- Considering the Client Selector, keep at least two clients active in every round: You MUST analyze the CPU values of all clients and choose an integer threshold that is strictly below the CPU values of at least two clients, so that at least two clients remain eligible and the system does not crash.\n"
        "- When 'client_selector' is 'ON', you MUST derive 'selection_value' from the observed CPU values of all clients. A threshold t is SAFE only if at least two clients satisfy CPU > t and at least one client satisfies CPU <= t. Examples: with CPUs {5,5,5,3,3}, safe thresholds are 3 or 4; with CPUs {3,3,3,2,1}, safe thresholds are 1 or 2. Do NOT invent arbitrary thresholds unrelated to the observed CPUs.\n"
        "- Before finalizing 'selection_value', mentally simulate how many clients have CPU > selection_value. If fewer than two clients would remain active, that value is INVALID and MUST NOT be used.\n"
        "- Prefer the SMALLEST safe 'selection_value' that excludes at least one lower-CPU client while keeping at least two clients with CPU > selection_value; if no such value exists, you MUST keep \"client_selector\":\"OFF\".\n"
        "- Do not keep architectural patterns permanently OFF. If a pattern (especially Message Compressor or Heterogeneous Data Handler) has been OFF in recent rounds and the evidence is weak or ambiguous, you should sometimes decide ON to probe its effect, instead of always defaulting to the safest minimal configuration.\n"
        "- In the absence of strong evidence, testing a pattern to verify its improvements is allowed. However, Activating a pattern always introduces overhead. Carefully consider whether it is really necessary. THIS DOES NOT MEAN THAT ARCHITECTURAL PATTERNS SHOULD NEVER BE ACTIVATED.\n"
        "- When some clients are non-IID or validation F1 is clearly worse than in comparable IID runs, you are expected to enable Heterogeneous Data Handler at least once early in the training to test its effect. If HDH has been rarely used and F1 remains poor while round times are acceptable, you should be willing to enable it again in later rounds to reassess its final impact.\n"
        "- If you aggregate metrics over multiple rounds (mean/median), explicitly say which statistic and window you used; otherwise state 'last-round only'.\n"
        "- If a pattern is already active from the previous round and you decide to keep it active, specify that 'it is kept active' and not, for example, 'we activate the pattern'.\n"
        "- Do not rely on unstated assumptions; prioritize the system's current configuration (e.g., number of clients with non-IID data, number of client's CPUs) over past evaluation metrics, which should be used only as secondary context.\n"
        "- When citing Training/Communication Time, always use client-level aggregates (mean and range min–max for the round).\n"
        "\n"
        "## Output\n"
        "Return exactly one JSON object with keys:\n"
        '- "client_selector": "ON" or "OFF"\n'
        '- "message_compressor": "ON" or "OFF"\n'
        '- "heterogeneous_data_handler": "ON" or "OFF"\n'
        '- "selection_value": <int>  # required only if "client_selector"=="ON"; MUST be an integer between 0 and max_cpu-1 (inclusive). Do NOT use values >= max_cpu.\n'
        '- "rationale": A detailed explanation (at least 50 words). The rationale MUST end with the exact signature line: CS=<ON|OFF>; MC=<ON|OFF>; HDH=<ON|OFF>, where ON/OFF equals the JSON decisions above.'
    )

    # 2) RAG: full system config + full metrics history 
    rag = ""
    try:
        use_rag = bool(USE_RAG)
    except Exception:
        use_rag = True

    if use_rag:
        import os, json, glob, re
        cfg_summary = {}
        prev_round_agg = {}
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
            cpus = [int((c or {}).get("cpu", 0) or 0) for c in cds if isinstance(c, dict)]
            rams = [int((c or {}).get("ram", 0) or 0) for c in cds if isinstance(c, dict)]
            dtypes = [str((c or {}).get("data_distribution_type", "")).upper() for c in cds if isinstance(c, dict)]
            delays = [str((c or {}).get("delay_combobox", "")).strip().lower() for c in cds if isinstance(c, dict)]
            dpers = [str((c or {}).get("data_persistence_type", "")).strip() for c in cds if isinstance(c, dict)]
            non_iid_ids = [int((c or {}).get("client_id")) for c in cds if str((c or {}).get("data_distribution_type", "")).upper() != "IID"]
            models = sorted({str((c or {}).get("model", "")).strip() for c in cds if isinstance(c, dict)} - {""})
            dataset_name = (cds[0].get("dataset") if cds else cfg.get("dataset", "")) or dataset
            n_clients = int(cfg.get("clients", len(cds) or clients) or clients)
            max_cpu = max(cpus) if cpus else 0
            sorted_cpus = sorted(cpus, reverse=True)
            second_highest_cpu = sorted_cpus[1] if len(sorted_cpus) >= 2 else max_cpu

            cfg_summary = {
                "dataset": dataset_name,
                "clients": n_clients,
                "cpu_per_client": cpus,
                "ram_per_client": rams,
                "max_cpu": max_cpu,
                "second_highest_cpu": second_highest_cpu,
                "data_distribution_counts": {k: sum(1 for d in dtypes if d == k) for k in set(dtypes or [])},
                "non_iid_clients": non_iid_ids,
                "has_delay_clients": any((d or "") == "yes" for d in delays),
                "models": models,
                "previous_ap": ap_prev_small,
                "data_persistence_per_client": dpers,
                "data_persistence_counts": {k: sum(1 for d in dpers if d == k) for k in set(dpers or [])},
            }
        except Exception:
            cfg_summary = {}

        metrics_digest = {}
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
                    import pandas as pd
                    df = pd.read_csv(last_file)

                    def find_col(cands):
                        for c in df.columns:
                            lc = c.strip().lower()
                            for pat in cands:
                                if isinstance(pat, str):
                                    if pat in lc:
                                        return c
                                else:
                                    if pat(lc):
                                        return c
                        return None

                    col_f1 = find_col(["Val F1"])
                    col_tr = find_col([lambda s: ("training" in s and "time" in s), "training (s)", "training time (s)"])
                    col_cm = find_col([lambda s: ("comm" in s and "time" in s) or ("communication" in s)])
                    col_tt = find_col([lambda s: ("total" in s and "time" in s) or ("total time of fl round" in s) or ("round time" in s)])

                    def to_series(col):
                        if not col or col not in df.columns:
                            return []
                        vals = []
                        for x in df[col].tolist():
                            try:
                                vals.append(float(x))
                            except Exception:
                                pass
                        return vals

                    s_f1 = to_series(col_f1)
                    s_tr = to_series(col_tr)
                    s_cm = to_series(col_cm)
                    s_tt = to_series(col_tt)

                    def stats(seq):
                        if not seq:
                            return {"count": 0}
                        n = len(seq)
                        mean = sum(seq) / n
                        mn = min(seq)
                        mx = max(seq)
                        last = seq[-1]
                        last3_mean = sum(seq[-3:]) / min(3, n)
                        last5_mean = sum(seq[-5:]) / min(5, n)
                        slope = (seq[-1] - seq[0]) / (n - 1) if n >= 2 else 0.0
                        tail = seq[-25:] if n > 25 else seq
                        return {
                            "count": n, "mean": mean, "min": mn, "max": mx, "last": last,
                            "last3_mean": last3_mean, "last5_mean": last5_mean,
                            "trend_slope": slope, "tail": tail
                        }

                    metrics_digest = {
                        "file": last_file,
                        "columns": {"F1": col_f1, "TrainingTime": col_tr, "CommTime": col_cm, "TotalTime": col_tt},
                        "F1": stats(s_f1),
                        "TrainingTime_s": stats(s_tr),
                        "CommunicationTime_s": stats(s_cm),
                        "TotalRoundTime_s": stats(s_tt),
                    }
                except Exception:
                    metrics_digest = {"file": last_file, "error": "failed_to_parse_csv"}
            else:
                metrics_digest = {"file": None, "error": "no_metrics_csv_found"}
        except Exception:
            metrics_digest = {"file": None, "error": "metrics_digest_failed"}

        try:
            import pandas as pd

            def _rnum_for_prev(p):
                m = re.search(r"round(\d+)", os.path.basename(p))
                return int(m.group(1)) if m else -1

            prev_candidates = [
                (r, p)
                for r, p in [(_rnum_for_prev(p), p) for p in glob.glob("**/FLwithAP_performance_metrics_round*.csv", recursive=True)]
                if r >= 0 and round_no is not None and r < round_no
            ]
            if prev_candidates:
                _, prev_path = max(prev_candidates, key=lambda item: item[0])
                prev_round_agg = _sa_aggregate_round(pd.read_csv(prev_path))
        except Exception:
            prev_round_agg = {}

        last_round_snapshot = (
            "### Last-Round Snapshot\n"
            f"- last_completed_round: {round_no if round_no is not None else '?'}\n"
            f"- Val F1: {_fmt(mean_f1, 4)}\n"
            f"- Total Round Time (global): {_fmt(mean_tt, 3)} s\n"
            f"- Training Time mean/min/max: {_fmt(mean_tr, 3)} / {_fmt(tr_stats.get('min'), 3)} / {_fmt(tr_stats.get('max'), 3)} s\n"
            f"- Communication Time mean/min/max: {_fmt(mean_comm, 3)} / {_fmt(cm_stats.get('min'), 3)} / {_fmt(cm_stats.get('max'), 3)} s\n"
            f"- JSD: {_fmt(mean_jsd, 4)}\n"
        )
        if prev_round_agg:
            last_round_snapshot += (
                "### Delta Vs Previous Completed Round\n"
                f"- Val F1 delta: {_delta(mean_f1, prev_round_agg.get('mean_f1'), 4)}\n"
                f"- Total Round Time delta: {_delta(mean_tt, prev_round_agg.get('mean_total_time'), 3)} s\n"
                f"- Training Time mean delta: {_delta(mean_tr, prev_round_agg.get('mean_training_time'), 3)} s\n"
                f"- Communication Time mean delta: {_delta(mean_comm, prev_round_agg.get('mean_comm_time'), 3)} s\n"
                f"- JSD delta: {_delta(mean_jsd, prev_round_agg.get('mean_jsd'), 4)}\n"
            )
        else:
            last_round_snapshot += (
                "### Delta Vs Previous Completed Round\n"
                "- previous_round_metrics: unavailable\n"
            )

        rag = (
            "## RAG\n"
            "### System Configuration (config.json)\n"
            + json.dumps(cfg_summary, ensure_ascii=False) + "\n"
            "### System Configuration — Field Guide\n"
            "- dataset: dataset name.\n"
            "- clients: number of clients.\n"
            "- cpu_per_client: CPUs per client (same order as client_details).\n"
            "- ram_per_client: RAM per client (same order as client_details).\n"
            "- max_cpu: client with the highest amount of CPU across clients.\n"
            "- second_highest_cpu: second-largest CPU (used by CS threshold guardrail).\n"
            "- data_distribution_counts: counts by data_distribution_type (IID / NON-IID).\n"
            "- non_iid_clients: list of client_id having NON-IID data.\n"
            "- has_delay_clients: True if any client has delay_combobox='yes'.\n"
            "- models: distinct model names used by clients.\n"
            "- previous_ap: previous ON/OFF states of {client_selector, message_compressor, heterogeneous_data_handler}.\n"
            "- data_persistence_per_client: for each client, 'New Data' (batched arrivals per round) or 'Same Data' (one-shot, all data at once).\n"
            "- data_persistence_counts: counts of 'New Data' vs 'Same Data'.\n"
            "\n"
            + last_round_snapshot
            + "\n"
            "### Performance Report (full history from CSV)\n"
            "The CSV file contains per-client metrics (Training Time and Communication Time) for each client in each round. You can consider these metrics individually per client or as an average.\n"
            "Note that the 'Val F1' column is the global model accuracy for the entire round, and 'Total Round Time' is the total duration of the round. These two metrics are global, not per-client.\n"
            "Note that column named 'Val F1' refers to the global model accuracy. Do not consider Train F1 column\n."
            + json.dumps(metrics_digest, ensure_ascii=False) + "\n"
            "### Performance Report — Field Guide\n"
            "- columns: mapping of canonical metric names → CSV column names {Val F1, TrainingTime, CommTime, TotalTime}.\n"
            "- Val F1: global model accuracy ('Val F1').\n"
            "- TrainingTime_s: stats over per-client training time (seconds) in the last round.\n"
            "- CommunicationTime_s: stats over per-client communication time (seconds) in the last round.\n"
            "- TotalRoundTime_s: stats over the global Total Round Time (seconds) in the last round.\n"
            "Each stats dict contains: count, mean, min, max, last, last3_mean, last5_mean, trend_slope, tail.\n"
            "\n"
        )
        try:
            import pandas as pd

            def _rnum(p):
                m = re.search(r"round(\d+)", os.path.basename(p))
                return int(m.group(1)) if m else -1

            files = [(_rnum(p), p) for p in glob.glob("**/FLwithAP_performance_metrics_round*.csv", recursive=True)]
            files = sorted([x for x in files if x[0] >= 0], key=lambda x: x[0])
            prev_window = [(r, p) for r, p in files if r < int(round_idx or 0)][-4:]
            recent = []
            for r, path in prev_window:
                try:
                    dfh = pd.read_csv(path)
                    ah = _sa_aggregate_round(dfh)
                    f1h = ah.get("mean_f1")
                    tth = ah.get("mean_total_time")
                    cmh = ah.get("mean_comm_time")
                    csh = (float(cmh) / float(tth)) if (tth and cmh and float(tth) > 0.0) else None
                    recent.append((r, f1h, csh))
                except Exception:
                    pass
            if recent:
                f1_line = "- Recent rounds (F1): " + " | ".join([f"r{r}:{_fmt(f1)}" for (r, f1, _) in recent]) + "\n"
                cs_line = "- Recent rounds (Comm share): " + " | ".join([f"r{r}:{_fmt(cs)}" for (r, _, cs) in recent]) + "\n"
                rag += f1_line + cs_line
        except Exception:
            pass

    # 3) Few-shot examples appended only for few-shot modes
    if mode != "zero":
        ex1_ctx = (
            "## Example 1 — Client Selector (edge-case)\n"
            "### Context\n"
            "- Client Selector can mitigate the Straggler effect\n"
            "### Decision\n"
            "\"client_selector\":\"ON\",\"selection_value\":4\n"
            "\"rationale\":\"low-spec clients (CPU=3) may cause the straggler effect. Activate the Client Selector with selection_value>4 to exclude them and improve both Total Round Time and Training Time; \n"
        )
        ex2_ctx = (
            "## Example 2 — Client Selector (numeric)\n"
            "### Context\n"
            "- 5 clients: 3 High-Spec (CPU=5) + 2 Low-Spec (CPU=3)\n"
            "### Decision\n"
            "\"client_selector\":\"ON\",\"selection_value\":3\n"
            "\"rationale\":\"Set client_selector=ON and selection_value=3 to drop only the Low-Spec clients (CPU=3) while keeping the three 5-CPU clients active; this preserves at least two active clients and reduces Total Round Time without killing the round.\"\n"
        )
        ex3_ctx = (
            "## Example 3 — Message Compressor (edge-case)\n"
            "### Context\n"
            "- Model: large (parameters doubled)\n"
            "### Decision\n"
            "{\"message_compressor\":\"ON\",\n"
            "\"rationale\":\"High communication proportion with a large model favors compression benefits; enable MC to cut communication time.\"}\n"
        )
        ex4_ctx = (
            "## Example 4 — Message Compressor (numeric)\n"
            "### Context\n"
            "- Model sizes: n/2, n, n×2\n"
            "- Observed: n/2 → MC worsens comm time; n → MC improves after round 4; n×2 → MC reduces comm time in all rounds\n"
            "### Decision\n"
            "{\"message_compressor\":\"ON\" (only for nx2)\n"
            "\"rationale\":\"For larger models (n×2) compression consistently reduces communication time across rounds; apply MC in this regime and avoid it for very small models (n/2).\"}\n"
        )
        ex5_ctx = (
            "## Example 5 — Heterogeneous Data Handler (edge-case)\n"
            "### Context\n"
            "- Critical Note: HDH is the most resource-intensive pattern, so handle it carefully. For non-IID clients, you run it once to fix their data. After that, don't re-run it immediately. Control later, if you have batched clients (data_persistence_per_client=New Data) getting new data in later rounds, then consider turning HDH on again a few rounds later.\n"
            "- Clients: mix of IID and non-IID; clear class imbalance on non-IID subset\n"
            "### Decision\n"
            "{\"heterogeneous_data_handler\":\"ON\","
            "\"rationale\":\"Activate HDH to mitigate non-IIDness; improves generalization with local overhead.\"}\n"
        )
        ex6_ctx = (
            "## Example 6 — Heterogeneous Data Handler (numeric)\n"
            "### Context\n"
            "- Consider 4 clients total: 2 IID + 2 non-IID\n"
            "- IID clients, Val F1 (baseline → later): 0.84 → 0.87\n"
            "- non-IID clients, Val F1 (baseline → later): 0.64 → 0.67\n"
            "- After HDH=ON (applied to non-IID clients), Val F1: 0.74 → 0.76\n"
            "### Decision\n"
            "{\"heterogeneous_data_handler\":\"ON\","
            "\"rationale\":\"Enable HDH for the non-IID clients: their F1 moves from 0.64–0.67 to 0.74–0.76, while IID clients are already stable (0.84→0.87).\"}\n"
        )

        header_few = "### Few-Shot Examples\n\n"
        return header_few + instructions + (rag or "") + ex1_ctx + "\n" + ex2_ctx + "\n" + ex3_ctx + "\n" + ex4_ctx + "\n" + ex5_ctx + "\n" + ex6_ctx + "\n" + "## Decision\n"
    
    return instructions + (rag or "") + "## Decision\n"

def _sa_generate_with_retry(model_name: str, mode: str, config, last_round, agg, ap_prev, base_urls: List[str], options: dict = None):
    p = _sa_build_prompt(mode, config, last_round, agg, ap_prev)
    raw = _sa_call_ollama(
        model_name, p, base_urls,
        force_json=True,
        options=(options if options is not None else {"temperature": 1.0, "top_p": 0.9, "num_ctx": 8192})
    )
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
    import re, json

    default = {"client_selector":"OFF","message_compressor":"OFF","heterogeneous_data_handler":"OFF"}

    s = "" if text is None else str(text).strip()
    s = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", s, flags=re.I | re.M).strip()

    obj = None
    last_json = None
    for m in re.finditer(r"\{.*?\}", s, flags=re.S):
        last_json = m.group(0)
    if last_json is not None:
        try:
            obj = json.loads(last_json)
        except Exception:
            obj = None
    if obj is None:
        try:
            obj = json.loads(s)
        except Exception:
            obj = {}

    km = {str(k).lower(): v for k, v in obj.items()} if isinstance(obj, dict) else {}

    def _onoff(v):
        if isinstance(v, bool):
            return "ON" if v else "OFF"
        vs = str(v).strip().upper()
        return "ON" if vs in ("ON","ENABLED","TRUE","1") else "OFF"

    # 1) decisioni iniziali dal JSON (o default se mancanti)
    decisions = {
        "client_selector":            _onoff(km.get("client_selector", default["client_selector"])),
        "message_compressor":         _onoff(km.get("message_compressor", default["message_compressor"])),
        "heterogeneous_data_handler": _onoff(km.get("heterogeneous_data_handler", default["heterogeneous_data_handler"])),
    }
    if "selection_value" in km:
        try:
            decisions["selection_value"] = int(km["selection_value"])
        except Exception:
            pass

    # 2) rationale: prendi dal JSON se c'è, altrimenti prova a ricavarlo dal testo
    rationale = str(km.get("rationale", "") or "").strip()
    if not rationale:
        rationale = (s.replace(last_json, "").strip() if last_json else s) or ""

    # 3) firma: se presente nel rationale o nel testo, diventa la verità che sovrascrive le decisioni
    sig_re = re.compile(r"\bCS\s*=\s*(ON|OFF)\s*;\s*MC\s*=\s*(ON|OFF)\s*;\s*HDH\s*=\s*(ON|OFF)\b", flags=re.I)
    m = sig_re.search(rationale) or sig_re.search(s)
    if m:
        cs_sig, mc_sig, hdh_sig = (m.group(1).upper(), m.group(2).upper(), m.group(3).upper())
        decisions["client_selector"]            = cs_sig
        decisions["message_compressor"]         = mc_sig
        decisions["heterogeneous_data_handler"] = hdh_sig
        if decisions["client_selector"] == "ON" and "selection_value" not in decisions:
            decisions["selection_value"] = 0
        if not sig_re.search(rationale):
            rationale = (rationale + ("\n" if rationale else "") + f"CS={cs_sig}; MC={mc_sig}; HDH={hdh_sig}").strip()
    else:
        cs_sig, mc_sig, hdh_sig = decisions["client_selector"], decisions["message_compressor"], decisions["heterogeneous_data_handler"]
        sig_line = f"CS={cs_sig}; MC={mc_sig}; HDH={hdh_sig}"
        rationale = (rationale + ("\n" if rationale else "") + sig_line).strip()
        if decisions["client_selector"] == "ON" and "selection_value" not in decisions:
            decisions["selection_value"] = 0

    return decisions, rationale, True

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

class AdaptationManager:
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

    def _decide_voting(self, base_config: Dict, current_round: int) -> Tuple[Dict, List[str]]:
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
                logs.append("")  # spazio tra agenti
            except Exception as e:
                agent_decisions.append({})
                agent_rationales.append("")
                logs.append(f"[Agent {i}] ERROR: {e!r}")
                logs.append("")

        # maggioranza (comportamento ufficiale, immutato)
        def _vote(pattern: str) -> int:
            return sum(1 for d in agent_decisions if (d.get(pattern, "OFF") or "OFF").upper() == "ON")

        maj_cs  = "ON" if _vote("client_selector") >= 2 else "OFF"
        maj_mc  = "ON" if _vote("message_compressor") >= 2 else "OFF"
        maj_hdh = "ON" if _vote("heterogeneous_data_handler") >= 2 else "OFF"

        logs.append(f"[Coordinator] Majority: CS: ·→{maj_cs} • MC: ·→{maj_mc} • HDH: ·→{maj_hdh}")
        logs.append("")
        # LOG DETERMINISTICO: ciò che viene davvero applicato
        logs.append(f"[Final Decision] CS={maj_cs}; MC={maj_mc}; HDH={maj_hdh}")
        logs.append("")

        # coordinator LLM consultivo, nessun override del risultato
        try:
            raw = _sa_call_ollama(
                COORD_MODEL,
                # Prompt base + breve riepilogo dei voti/majority
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
            # Sanitize per evitare righe che sembrano un verdetto finale
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

    def _decide_role(self, base_config: Dict, current_round: int) -> Tuple[Dict, List[str]]:
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


    def _decide_debate(self, base_config: Dict, current_round: int) -> Tuple[Dict, List[str]]:
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
    
    def _decide_single_agent(self, base_config: Dict, current_round: int) -> Tuple[Dict, List[str]]:
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
            
            # Trova la colonna dei round
            col_round = next((c for c in df.columns if "round" in str(c).lower().strip()), None)
            
            if col_round and len(df) > 0:
                last_r_val = int(df[col_round].dropna().astype(int).max())
                df_r = df[df[col_round].astype(int) == last_r_val].copy()
                df_prev = df[df[col_round].astype(int) == (last_r_val - 1)].copy()
            else:
                df_r = df.copy()
                df_prev = pd.DataFrame() # Vuoto
                
            agg_r = _sa_aggregate_round(df_r)
            agg_prev = _sa_aggregate_round(df_prev) if not df_prev.empty else {}
            
        except Exception as e:
            logs.append(f"Error reading metrics: {e}")
            return new_config, logs
        
        # Helper per prendere i valori in sicurezza
        def get_val(agg_dict, key, default=0.0):
            v = agg_dict.get(key)
            return float(v) if v is not None else default

        f1_r = get_val(agg_r, "mean_f1")
        time_r = get_val(agg_r, "mean_total_time", 1.0) # Evita div/0
        jsd_r = get_val(agg_r, "mean_jsd")
        comm_r = get_val(agg_r, "mean_comm_time")

        f1_prev = get_val(agg_prev, "mean_f1")
        time_prev = get_val(agg_prev, "mean_total_time", 1.0)
        jsd_prev = get_val(agg_prev, "mean_jsd")
        comm_prev = get_val(agg_prev, "mean_comm_time")

        # Client Selector: (Accuracy_r / Time_r) < (Accuracy_prev / Time_prev)
        rate_r = f1_r / time_r if time_r > 0 else 0
        rate_prev = f1_prev / time_prev if time_prev > 0 else 0
        
        if rate_r < rate_prev:
            new_config["patterns"]["client_selector"]["enabled"] = True
            logs.append(f"Client Selector: ON (Rate {rate_r:.4f} < {rate_prev:.4f})")
        else:
            new_config["patterns"]["client_selector"]["enabled"] = False
            logs.append(f"Client Selector: OFF (Rate {rate_r:.4f} >= {rate_prev:.4f})")

        # Heterogeneous Data Handler: JSD_r < JSD_prev AND F1_r < F1_prev
        if jsd_r < jsd_prev and f1_r < f1_prev:
            new_config["patterns"]["heterogeneous_data_handler"]["enabled"] = True
            logs.append(f"Heterogeneous Data Handler: ON (JSD {jsd_r:.3f}<{jsd_prev:.3f} AND F1 {f1_r:.3f}<{f1_prev:.3f})")
        else:
            new_config["patterns"]["heterogeneous_data_handler"]["enabled"] = False
            logs.append(f"Heterogeneous Data Handler: OFF (Condition not met: JSD {jsd_r:.3f} vs {jsd_prev:.3f}, F1 {f1_r:.3f} vs {f1_prev:.3f})")

        # Message Compressor: Comm_r > Comm_prev
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
