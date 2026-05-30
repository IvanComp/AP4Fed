import glob
import json
import os
import re
from typing import List

from adaptation_metrics import _sa_aggregate_round
from adaptation_settings import USE_RAG, config_file
from ollama_client import _sa_call_ollama


def _runtime_attr(name, default):
    try:
        import adaptation
        return getattr(adaptation, name, default)
    except Exception:
        return default

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

    rag = ""
    try:
        use_rag = bool(_runtime_attr("USE_RAG", USE_RAG))
    except Exception:
        use_rag = True

    if use_rag:
        import os, json, glob, re
        cfg_summary = {}
        prev_round_agg = {}
        try:
            cfg = {}
            try:
                if os.path.exists(_runtime_attr("config_file", config_file)):
                    with open(_runtime_attr("config_file", config_file), "r") as f:
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
    build_prompt = _runtime_attr("_sa_build_prompt", _sa_build_prompt)
    call_ollama = _runtime_attr("_sa_call_ollama", _sa_call_ollama)
    parse_output = _runtime_attr("_sa_parse_output", _sa_parse_output)
    p = build_prompt(mode, config, last_round, agg, ap_prev)
    raw = call_ollama(
        model_name, p, base_urls,
        force_json=True,
        options=(options if options is not None else {"temperature": 1.0, "top_p": 0.9, "num_ctx": 8192})
    )
    d1, r1, _ = parse_output(raw)
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

    rationale = str(km.get("rationale", "") or "").strip()
    if not rationale:
        rationale = (s.replace(last_json, "").strip() if last_json else s) or ""

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
