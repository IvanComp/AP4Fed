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
        "training_time_stats": tr_agg,       # include count/min/max
        "comm_time_stats": cm_agg
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

    mean_f1   = (agg or {}).get("mean_f1")
    mean_tt   = (agg or {}).get("mean_total_time")
    mean_tr   = (agg or {}).get("mean_training_time")
    mean_comm = (agg or {}).get("mean_comm_time")

    ap_prev_small = {
        "client_selector": "ON" if (ap_prev or {}).get("client_selector") else "OFF",
        "message_compressor": "ON" if (ap_prev or {}).get("message_compressor") else "OFF",
        "heterogeneous_data_handler": "ON" if (ap_prev or {}).get("heterogeneous_data_handler") else "OFF",
    }

    # 1) Static Context: Role, Task, Output, Guardrails
    instructions = (
        "Architectural Pattern Decision: Prompt\n\n"
        "#Context\n"

        "\n"
        "## Role: You are an AI agent and expert software architect for Federated Learning systems.\n"
        "At the end of each training round, analyze the current SYSTEM CONFIGURATION and actual system EVALUATION METRICS, then recommend which architectural patterns to turn ON or OFF for the next round to optimize system EVALUATION METRICS.\n"        
        
        "\n"
        "##Task: Your task is to activate or deactive one or a combination of architectural patterns for the next round of training. The objective is to optimize the system EVALUATION METRICS as much as possible.\n"
        "Here are the three ARCHITECTURAL PATTERNS that you can consider to (de)activate:\n"
        "1) Client Selector: selects clients that will partecipate to the next round of Federated Learning. The selection is driven by the number of CPUs available by the clients. When active, you must specify 'selection_value': <int>, which is the CPU threshold. Only the clients with CPU > selection_value will be included for the next training round.\n"
        "2) Message Compressor: compresses messages exchanged between the clients and the server.\n"
        "3) Heterogeneous Data Handler: mitigates non-IID effects using class reweighting, augmentation, or distillation.\n"  
        "\n"
        "Ideally, we aim to optimize the following EVALUATION METRICS:\n"
        "1) Maximize Global Model Accuracy, which measures the overall predictive performance of the aggregated model across all clients.\n"
        "2) Minimize Total Round Time, defined as the elapsed time to complete a full federated learning round.\n"
        "3) Minimize Communication Time, representing the duration spent exchanging data between the server and clients during each round.\n"
        "4) Minimize Training Time, referring to the computational time each client requires to perform local model training.\n"
        "\n"
        "Enabling or disabling each architectural pattern has both advantages and disadvantages in terms of performance. Here are the architectural patterns’ performance trade-offs to consider:\n"
        "- Client Selector: Selects clients with higher computational power to reduce slowdowns caused by weaker devices (the Straggler effect). This reduces total round time, training time, and communication time. However, persistently excluding lower-capacity clients may decrease model diversity, harming overall accuracy due to less varied data.\n"
        "- Message Compressor: Compresses model updates exchanged between the server and clients to reduce communication time, which is especially beneficial for large models or bandwidth-constrained networks. The downside is the additional computational overhead from compression and decompression, which may outweigh the benefits when dealing with small model sizes.\n"
        "- Heterogeneous Data Handler: Employs techniques such as data augmentation to address data heterogeneity among clients, improving global model accuracy and accelerating convergence on non-IID data. The trade-off is an increase in local computation required to generate synthetic data for balancing class distributions, which can add significant overhead.\n"
        "\n"
        "## Guardrails\n"
        "- Use only the Context and the RAG section (if present: config snapshot + recent metrics). Do not invent values.\n"
        "- Output only the JSON object, no extra text otherwise the system fails.\n"
        "- The last line of the 'rationale' must be exactly the signature: CS=<ON|OFF>; MC=<ON|OFF>; HDH=<ON|OFF>. These values must copy the JSON decisions above, in the same order and casing.\n"
        "- selection_value is a CPU threshold for the Client Selector pattern: only clients with CPU > selection_value participate in each training round. Therefore, use an integer within [0, max_cpu-1], and ensure it is strictly less than the second-highest CPU value to avoid excluding more clients than necessary. It is crucial to guarantee that at least two clients remain active and are not excluded, since the federated learning process will fail if fewer than two clients participate in a round.\n"
        "- If evidence is insufficient or conflicting, keep the previous choices and say why in the rationale.\n"
        "- Activating a pattern always introduces overhead. Carefully consider whether it is really necessary. This does not mean that they should never be activated.\n"
        "- If you aggregate metrics over multiple rounds (mean/median), explicitly say which statistic and window you used; otherwise state 'last-round only'.\n"
        "- If a pattern is already active from the previous round and you decide to keep it active, specify that 'it is kept active' and not, for example, 'we activate the pattern'.\n"
        "- Never rely on unstated assumptions; prefer measurable values (CPU list, non_iid_clients, last metrics).\n"
        "- F1 always means the GLOBAL MODEL validation F1 (use the column 'Val F1')\n"
        "- When citing Training/Communication Time, always use client-level aggregates (mean and range min–max for the round). Do not quote a single client's time.\n"

        "\n"
        "## Output\n"
        "Return exactly one JSON object with keys:\n"
        '- "client_selector": "ON" or "OFF"\n'
        '- "message_compressor": "ON" or "OFF"\n'
        '- "heterogeneous_data_handler": "ON" or "OFF"\n'
        '- "selection_value": <int>  # required only if "client_selector"=="ON"\n'
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
                "previous_ap": ap_prev_small
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
                    s_share = []
                    for c, t in zip(s_cm, s_tt):
                        try:
                            s_share.append(c / max(1e-9, t))
                        except Exception:
                            s_share.append(None)

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
                        "CommShare_of_Total": stats([x for x in s_share if x is not None])
                    }
                except Exception:
                    metrics_digest = {"file": last_file, "error": "failed_to_parse_csv"}
            else:
                metrics_digest = {"file": None, "error": "no_metrics_csv_found"}
        except Exception:
            metrics_digest = {"file": None, "error": "metrics_digest_failed"}

        rag = (
            "## RAG\n"
            "### System Configuration (config.json)\n"
            + json.dumps(cfg_summary, ensure_ascii=False) + "\n\n"
            "### Performance Report (full history from CSV)\n"
            + json.dumps(metrics_digest, ensure_ascii=False) + "\n\n"
        )
    else:
        rag = (
            "## RAG\n"
            "### System Configuration & Metrics Digest\n"
            "The CSV file contains per-client metrics (Training Time and Communication Time) for each client in each round. You can consider these metrics individually per client or as an average.\n"
            "Note that 'Val F1' is the global model accuracy for the entire round, and 'Total Round Time' is the total duration of the round. These two metrics are global, not per-client.\n"
            "Note that column named 'Val F1' refers to the global model accuracy. Do not consider Train F1 column\n."
            "### System Configuration (runtime snapshot)\n"
             + json.dumps({
                "dataset": dataset,
                "clients": clients,
                "model": model,
                "previous_ap": ap_prev_small
            }, ensure_ascii=False) + "\n\n"
            "### Performance (aggregates)\n"
            + json.dumps({
                "mean_f1": mean_f1, "mean_training_time_s": mean_tr,
                "mean_comm_time_s": mean_comm, "mean_total_time_s": mean_tt
            }, ensure_ascii=False) + "\n\n"
        )

    # 3) Few-shot examples appended only for few-shot modes
    if mode != "zero":
        ex1_ctx = (
            "## Example 1 — Client Selector (edge-case)\n"
            "### Context\n"
            "- CPUs: [5,5,5,3,3]\n"
            "- Data: all IID\n"
            "- Comm share: 0.30\n"
            "- Previous AP: {\"client_selector\":\"OFF\",\"message_compressor\":\"OFF\",\"heterogeneous_data_handler\":\"OFF\"}\n\n"
            "### Decision\n"
            "{\"client_selector\":\"ON\",\"selection_value\":1,\"message_compressor\":\"OFF\",\"heterogeneous_data_handler\":\"OFF\","
            "\"rationale\":\"A single low-spec client (CPU=3) throttles synchronization. Set threshold >3 to exclude it; CS=ON; MC=OFF; HDH=OFF\"}\n"
        )
        ex2_ctx = (
            "## Example 2 — Client Selector (numeric)\n"
            "### Context\n"
            "- 4 clients: 3 High-Spec (CPU=5) + 2 Low-Spec (CPU=3)\n"
            "- Data: IID\n"
            "- Observed: with CS=OFF, Total Round Time ≈ 9× slower than with CS=ON; F1 ≈ 0.57–0.59 in both cases\n"
            "- Previous AP: {\"client_selector\":\"OFF\",\"message_compressor\":\"OFF\",\"heterogeneous_data_handler\":\"OFF\"}\n\n"
            "### Decision\n"
            "{\"client_selector\":\"ON\",\"selection_value\":1,\"message_compressor\":\"OFF\",\"heterogeneous_data_handler\":\"OFF\","
            "\"rationale\":\"Exclude the lowest CPU clients (2) to remove the bottleneck (≈9× Total Round Time reduction reported with same F1). \"}\n"
        )
        ex3_ctx = (
            "## Example 3 — Message Compressor (edge-case)\n"
            "### Context\n"
            "- Model: large (parameters doubled); Comm share: 0.65\n"
            "- CPUs balanced; Data: IID\n"
            "- Previous AP: {\"client_selector\":\"OFF\",\"message_compressor\":\"OFF\",\"heterogeneous_data_handler\":\"OFF\"}\n\n"
            "### Decision\n"
            "{\"client_selector\":\"OFF\",\"message_compressor\":\"ON\",\"heterogeneous_data_handler\":\"OFF\","
            "\"rationale\":\"High communication proportion with a large model favors compression benefits; enable MC to cut comm time.\"}\n"
        )
        ex4_ctx = (
            "## Example 4 — Message Compressor (numeric)\n"
            "### Context\n"
            "- Model sizes: n/2, n, n×2\n"
            "- Observed: n/2 → MC worsens comm time; n → MC improves after round 4; n×2 → MC reduces comm time in all rounds\n"
            "- Previous AP: {\"client_selector\":\"OFF\",\"message_compressor\":\"OFF\",\"heterogeneous_data_handler\":\"OFF\"}\n\n"
            "### Decision\n"
            "{\"client_selector\":\"OFF\",\"message_compressor\":\"ON\",\"heterogeneous_data_handler\":\"OFF\","
            "\"rationale\":\"For larger models (n×2) compression consistently reduces communication time across rounds; apply MC in this regime and avoid it for very small models (n/2).\"}\n"
        )
        ex5_ctx = (
            "## Example 5 — Heterogeneous Data Handler (edge-case)\n"
            "### Context\n"
            "- Clients: mix of IID and non-IID; clear class imbalance on non-IID subset\n"
            "- Previous AP: {\"client_selector\":\"OFF\",\"message_compressor\":\"OFF\",\"heterogeneous_data_handler\":\"OFF\"}\n\n"
            "### Decision\n"
            "{\"client_selector\":\"OFF\",\"message_compressor\":\"OFF\",\"heterogeneous_data_handler\":\"ON\","
            "\"rationale\":\"Activate HDH to mitigate non-IID with augmentation; improves generalization with acceptable local overhead.\"}\n"
        )
        ex6_ctx = (
            "## Example 6 — Heterogeneous Data Handler (numeric)\n"
            "### Context\n"
            "- 8 clients: 4 IID + 4 non-IID → apply GAN-based augmentation to non-IID\n"
            "- Observed: F1 rises (e.g., ≈0.24→≈0.26 by rounds 9–10) and round times stabilize vs non-IID baseline (which shows 700–800 s and higher variance)\n"
            "- Previous AP: {\"client_selector\":\"OFF\",\"message_compressor\":\"OFF\",\"heterogeneous_data_handler\":\"OFF\"}\n\n"
            "### Decision\n"
            "{\"client_selector\":\"OFF\",\"message_compressor\":\"OFF\",\"heterogeneous_data_handler\":\"ON\","
            "\"rationale\":\"Enable HDH to rebalance classes on non-IID clients, improving F1 while keeping round-time variability under control.\"}\n"
        )
        header_few = "### Few-Shot Examples\n\n"
        return header_few + instructions + (rag or "") + ex1_ctx + "\n" + ex2_ctx + "\n" + ex3_ctx + "\n" + ex4_ctx + "\n" + ex5_ctx + "\n" + ex6_ctx + "\n" + "## Decision\n"
    return instructions + (rag or "") + "## Decision\n"

def _sa_generate_with_retry(model_name: str, mode: str, config, last_round, agg, ap_prev, base_urls: List[str]):
    p = _sa_build_prompt(mode, config, last_round, agg, ap_prev)
    raw = _sa_call_ollama(
    model_name, p, base_urls,
    force_json=True,
    options={"temperature": 0.2, "top_p": 0.9, "num_ctx": 8192}
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
        import re, json

        logs: List[str] = []
        last_round, last_csv = _sa_latest_round_csv()
        agg, ap_prev = {}, {}
        if last_csv:
            try:
                import pandas as pd
                df = pd.read_csv(last_csv)
                agg = _sa_aggregate_round(df)

                # === Estrazione ap_prev direttamente da "AP List (...)" ===
                ap_col = next((c for c in df.columns if isinstance(c, str) and c.startswith("AP List")), None)
                if ap_col is not None and len(df) > 0:
                    cell = str(df.iloc[-1][ap_col])

                    # Chiavi dall'header tra parentesi tonde
                    m_keys = re.search(r"\((?P<inside>.*?)\)", ap_col)
                    keys = [k.strip() for k in m_keys.group("inside").split(",")] if m_keys else []

                    # Valori ON/OFF dalla cella tra parentesi graffe
                    m_vals = re.search(r"\{(?P<inside>.*?)\}", cell)
                    states = [s.strip().upper() for s in m_vals.group("inside").split(",")] if m_vals else []

                    mapping = dict(zip(keys, states))
                    ap_prev = {
                        "client_selector": mapping.get("client_selector", "OFF") == "ON",
                        "message_compressor": mapping.get("message_compressor", "OFF") == "ON",
                        "heterogeneous_data_handler": mapping.get("heterogeneous_data_handler", "OFF") == "ON",
                    }
                # === fine estrazione ===
            except Exception:
                pass

        mode = _sa_mode_from_policy(self.policy)
        try:
            decisions, rationale = _sa_generate_with_retry(
                self.sa_model, mode, self.default_config, last_round, agg, ap_prev, self.sa_ollama_urls
            )
        except Exception:
            return copy.deepcopy(base_config), logs

        prev_cs  = "✅" if ap_prev.get("client_selector") else ("❌" if ap_prev.get("client_selector") is not None else "·")
        prev_mc  = "✅" if ap_prev.get("message_compressor") else ("❌" if ap_prev.get("message_compressor") is not None else "·")
        prev_hdh = "✅" if ap_prev.get("heterogeneous_data_handler") else ("❌" if ap_prev.get("heterogeneous_data_handler") is not None else "·")

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
        self.adaptation_time = t_agent
        
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