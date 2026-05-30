import copy
import json
import os
import random
import time
import urllib.error
import urllib.request
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import (
    FedAdagrad as FlowerFedAdagrad,
    FedAdam as FlowerFedAdam,
    FedAvg as FlowerFedAvg,
    FedAvgM as FlowerFedAvgM,
    FedProx as FlowerFedProx,
    FedYogi as FlowerFedYogi,
    QFedAvg as FlowerQFedAvg,
)


BASELINE_LABELS = {
    "fedavg": "FedAvg",
    "fedavgm": "FedAvgM",
    "fedprox": "FedProx",
    "fedadam": "FedAdam",
    "fedadagrad": "FedAdagrad",
    "fedyogi": "FedYogi",
    "qfedavg": "QFedAvg",
}


def _canonical_baseline(name: Optional[str]) -> str:
    raw = str(name or "").strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "fedavg": "fedavg",
        "fedaverage": "fedavg",
        "fedavgm": "fedavgm",
        "fedavgmomentum": "fedavgm",
        "fedprox": "fedprox",
        "fedadam": "fedadam",
        "fedadagrad": "fedadagrad",
        "fedyogi": "fedyogi",
        "qfedavg": "qfedavg",
        "qffedavg": "qfedavg",
    }
    return aliases.get(raw, "fedavg")


def _safe_float(value):
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


class DynamicAggregationBaselineManager:
    def __init__(self, config: Dict, initial_parameters: Parameters, performance_dir: str):
        agg_cfg = config.get("aggregation_baselines", {}) or {}
        self.enabled = bool(agg_cfg.get("enabled", False))
        self.policy = str(agg_cfg.get("policy", "LLM Dynamic Switch"))
        self.performance_dir = performance_dir
        self.total_rounds = int(config.get("rounds", 1) or 1)
        self.current_baseline = _canonical_baseline(agg_cfg.get("initial_baseline", "fedavg"))
        requested_candidates = agg_cfg.get("candidates") or list(BASELINE_LABELS.keys())
        self.candidates = []
        for candidate in requested_candidates:
            canonical = _canonical_baseline(candidate)
            if canonical not in self.candidates:
                self.candidates.append(canonical)
        if not self.candidates:
            self.candidates = list(BASELINE_LABELS.keys())
        if self.current_baseline not in self.candidates:
            self.current_baseline = self.candidates[0]

        self.llm_model = agg_cfg.get("llm_model") or config.get("LLM") or "llama3.2:3b"
        self.ollama_base_urls = [
            agg_cfg.get("ollama_base_url") or config.get("ollama_base_url"),
            "http://host.docker.internal:11434",
            "http://127.0.0.1:11434",
            "http://localhost:11434",
        ]
        self.ollama_base_urls = [u.rstrip("/") for u in self.ollama_base_urls if u]
        params = agg_cfg.get("params", {}) or {}
        self.fedavgm_momentum = float(params.get("fedavgm_momentum", 0.9))
        self.fedavgm_learning_rate = float(params.get("fedavgm_learning_rate", 1.0))
        self.fedprox_mu = float(params.get("fedprox_mu", 0.01))
        self.fedopt_eta = float(params.get("fedopt_eta", 0.1))
        self.fedopt_eta_l = float(params.get("fedopt_eta_l", 0.1))
        self.fedopt_beta_1 = float(params.get("fedopt_beta_1", 0.9))
        self.fedopt_beta_2 = float(params.get("fedopt_beta_2", 0.99))
        self.fedopt_tau = float(params.get("fedopt_tau", 1e-9))
        self.qfedavg_q = float(params.get("qfedavg_q", 0.2))
        self.qfedavg_learning_rate = float(params.get("qfedavg_learning_rate", 0.1))
        self.history: List[Dict] = []
        self._strategies = self._build_strategies(initial_parameters)

    def _build_strategies(self, initial_parameters: Parameters) -> Dict[str, object]:
        return {
            "fedavg": FlowerFedAvg(
                min_fit_clients=1,
                min_evaluate_clients=1,
                min_available_clients=1,
                initial_parameters=initial_parameters,
                fit_metrics_aggregation_fn=lambda _: {},
                inplace=False,
            ),
            "fedavgm": FlowerFedAvgM(
                min_fit_clients=1,
                min_evaluate_clients=1,
                min_available_clients=1,
                initial_parameters=initial_parameters,
                server_learning_rate=self.fedavgm_learning_rate,
                server_momentum=self.fedavgm_momentum,
                fit_metrics_aggregation_fn=lambda _: {},
            ),
            "fedprox": FlowerFedProx(
                min_fit_clients=1,
                min_evaluate_clients=1,
                min_available_clients=1,
                initial_parameters=initial_parameters,
                fit_metrics_aggregation_fn=lambda _: {},
                proximal_mu=self.fedprox_mu,
            ),
            "fedadam": FlowerFedAdam(
                min_fit_clients=1,
                min_evaluate_clients=1,
                min_available_clients=1,
                initial_parameters=initial_parameters,
                fit_metrics_aggregation_fn=lambda _: {},
                eta=self.fedopt_eta,
                eta_l=self.fedopt_eta_l,
                beta_1=self.fedopt_beta_1,
                beta_2=self.fedopt_beta_2,
                tau=self.fedopt_tau,
            ),
            "fedadagrad": FlowerFedAdagrad(
                min_fit_clients=1,
                min_evaluate_clients=1,
                min_available_clients=1,
                initial_parameters=initial_parameters,
                fit_metrics_aggregation_fn=lambda _: {},
                eta=self.fedopt_eta,
                eta_l=self.fedopt_eta_l,
                tau=self.fedopt_tau,
            ),
            "fedyogi": FlowerFedYogi(
                min_fit_clients=1,
                min_evaluate_clients=1,
                min_available_clients=1,
                initial_parameters=initial_parameters,
                fit_metrics_aggregation_fn=lambda _: {},
                eta=self.fedopt_eta,
                eta_l=self.fedopt_eta_l,
                beta_1=self.fedopt_beta_1,
                beta_2=self.fedopt_beta_2,
                tau=max(self.fedopt_tau, 0.001),
            ),
            "qfedavg": FlowerQFedAvg(
                min_fit_clients=1,
                min_evaluate_clients=1,
                min_available_clients=1,
                initial_parameters=initial_parameters,
                fit_metrics_aggregation_fn=lambda _: {},
                evaluate_fn=lambda _rnd, _weights, _cfg: (1.0, {}),
                q_param=self.qfedavg_q,
                qffl_learning_rate=self.qfedavg_learning_rate,
            ),
        }

    def label(self, baseline: Optional[str] = None) -> str:
        return BASELINE_LABELS.get(_canonical_baseline(baseline or self.current_baseline), "FedAvg")

    def describe(self) -> str:
        if not self.enabled:
            return "Dynamic aggregation baselines disabled"
        candidates = ", ".join(self.label(c) for c in self.candidates)
        return f"Dynamic aggregation baselines enabled: {candidates}; LLM={self.llm_model}; current={self.label()}"

    def fit_config(self) -> Dict[str, float]:
        if self.current_baseline == "fedprox":
            return {"proximal_mu": self.fedprox_mu}
        return {}

    def aggregate(
        self,
        server_round: int,
        results: List[Tuple[Parameters, int, Dict]],
        current_parameters: Parameters,
    ) -> Parameters:
        strategy = self._strategies.get(self.current_baseline) or self._strategies["fedavg"]
        if self.current_baseline in {"fedavgm", "fedadam", "fedadagrad", "fedyogi"}:
            strategy.initial_parameters = current_parameters
        if self.current_baseline == "qfedavg":
            strategy.pre_weights = parameters_to_ndarrays(current_parameters)
        flower_results = [
            (
                None,
                SimpleNamespace(
                    parameters=client_params,
                    num_examples=int(num_examples),
                    metrics=client_metrics or {},
                ),
            )
            for client_params, num_examples, client_metrics in results
        ]
        parameters_aggregated, _ = strategy.aggregate_fit(server_round, flower_results, [])
        if parameters_aggregated is None:
            return current_parameters
        return parameters_aggregated

    def _summarize_latest_round(self, csv_path: str, current_round: int, metrics_history: Dict) -> Dict:
        summary = {
            "round": current_round,
            "current_baseline": self.label(),
            "history": self.history[-5:],
        }
        try:
            df = pd.read_csv(csv_path)
            if "FL Round" in df.columns:
                df = df[pd.to_numeric(df["FL Round"], errors="coerce") == current_round]
            if df.empty:
                return summary
            last = df.iloc[-1]
            for col, key in [
                ("Val F1", "val_f1"),
                ("Val Accuracy", "val_accuracy"),
                ("Total Time of FL Round", "total_round_time"),
                ("JSD", "jsd"),
            ]:
                summary[key] = _safe_float(last.get(col))
            for col, key in [
                ("Training Time", "training_time"),
                ("Communication Time", "communication_time"),
                ("# of CPU", "client_cpu"),
                ("CPU Usage (%)", "cpu_usage"),
                ("RAM Usage (%)", "ram_usage"),
            ]:
                if col in df.columns:
                    values = pd.to_numeric(df[col], errors="coerce").dropna().tolist()
                    if values:
                        summary[f"{key}_mean"] = float(np.mean(values))
                        summary[f"{key}_min"] = float(np.min(values))
                        summary[f"{key}_max"] = float(np.max(values))
            if summary.get("total_round_time") and summary.get("communication_time_mean"):
                summary["communication_share"] = float(summary["communication_time_mean"]) / float(summary["total_round_time"])
            if summary.get("training_time_mean"):
                summary["training_straggler_ratio"] = float(summary.get("training_time_max", 0.0)) / max(
                    float(summary["training_time_mean"]), 1e-9
                )
        except Exception as exc:
            summary["csv_error"] = repr(exc)

        try:
            if metrics_history:
                model_hist = next(iter(metrics_history.values()))
                f1_values = [float(v) for v in model_hist.get("val_f1", []) if v is not None]
                if len(f1_values) >= 2:
                    summary["val_f1_delta"] = f1_values[-1] - f1_values[-2]
                    summary["recent_val_f1"] = f1_values[-5:]
        except Exception:
            pass
        return summary

    def _build_prompt(self, round_summary: Dict) -> str:
        candidate_guide = {
            "fedavg": "Weighted average. Stable default when clients are fairly regular.",
            "fedavgm": "Server momentum. Useful when F1 improvement stalls or convergence is noisy.",
            "fedprox": "Proximal optimization for system/data heterogeneity. Prefer when clients are heterogeneous or non-IID.",
            "fedadam": "Adaptive server optimizer with Adam updates. Useful when server-side optimization can speed convergence.",
            "fedadagrad": "Adaptive server optimizer with Adagrad updates. Useful for noisy or sparse update dynamics.",
            "fedyogi": "Adaptive server optimizer with Yogi updates. Useful for stable adaptive convergence under non-IID updates.",
            "qfedavg": "q-FedAvg optimizes a fairness-aware objective and can improve weak-client performance under non-IID imbalance.",
        }
        candidates = {self.label(c): candidate_guide[c] for c in self.candidates if c in candidate_guide}
        return (
            "You are an expert Federated Learning controller. At the end of each round, choose the aggregation "
            "baseline for the next round.\n"
            "Optimize validation F1/accuracy first, then total round time, communication time, and robustness to "
            "heterogeneous clients. Use only the candidate baselines listed below.\n\n"
            f"Candidate baselines:\n{json.dumps(candidates, indent=2)}\n\n"
            f"Round summary:\n{json.dumps(round_summary, indent=2)}\n\n"
            "Return exactly one JSON object with keys:\n"
            f'- "baseline": one of {json.dumps([self.label(c) for c in self.candidates])}\n'
            '- "rationale": short explanation of the metric evidence used\n'
        )

    def _decide_without_llm(self, summary: Dict) -> Tuple[str, str]:
        policy_l = self.policy.lower()
        if "random" in policy_l:
            selected = random.choice(self.candidates)
            return selected, f"Random policy selected {self.label(selected)} from the configured candidate strategies."
        if "bayesian" in policy_l:
            if summary.get("val_f1_delta") is not None and summary.get("val_f1_delta") < 0:
                selected = "fedyogi" if "fedyogi" in self.candidates else self.current_baseline
                return selected, "Bayesian-optimization placeholder: validation F1 decreased, so a conservative adaptive optimizer is preferred."
            selected = "fedavgm" if "fedavgm" in self.candidates else self.current_baseline
            return selected, "Bayesian-optimization placeholder: no fitted surrogate is available yet, so the exploratory momentum strategy is preferred."
        if "predictor" in policy_l:
            if summary.get("training_straggler_ratio", 1.0) > 1.5:
                selected = "fedprox" if "fedprox" in self.candidates else self.current_baseline
                return selected, "Predictor-based heuristic selected FedProx because client training times suggest heterogeneity."
            if summary.get("val_f1_delta", 0.0) < -0.01:
                selected = "fedadam" if "fedadam" in self.candidates else self.current_baseline
                return selected, "Predictor-based heuristic selected FedAdam because validation F1 degraded and adaptive server updates may improve convergence."
            selected = "fedavg" if "fedavg" in self.candidates else self.current_baseline
            return selected, "Predictor-based heuristic kept FedAvg because no performance signal suggests switching."
        return self.current_baseline, f"Policy '{self.policy}' is not LLM-Based; keeping {self.label()}."

    def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2, "top_p": 0.9, "num_ctx": 4096},
        }
        data = json.dumps(payload).encode("utf-8")
        last_error = None
        for base_url in self.ollama_base_urls:
            try:
                req = urllib.request.Request(
                    f"{base_url}/api/generate",
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    out = json.loads(resp.read().decode("utf-8"))
                if out.get("error"):
                    raise RuntimeError(str(out["error"]))
                return str(out.get("response") or "").strip()
            except (urllib.error.URLError, TimeoutError, RuntimeError, ValueError) as exc:
                last_error = exc
        raise RuntimeError(f"Ollama unreachable: {last_error}")

    def _parse_decision(self, raw: str) -> Tuple[str, str]:
        try:
            obj = json.loads(raw)
        except Exception:
            start = raw.find("{")
            end = raw.rfind("}")
            obj = json.loads(raw[start : end + 1]) if start >= 0 and end > start else {}
        raw_baseline = obj.get("baseline") if isinstance(obj, dict) else None
        baseline = _canonical_baseline(raw_baseline)
        if not raw_baseline or baseline not in self.candidates:
            baseline = self.current_baseline
        rationale = str(obj.get("rationale", "") if isinstance(obj, dict) else "").strip()
        return baseline, rationale

    def decide_next_round(self, current_round: int, metrics_history: Dict, csv_path: str) -> Dict:
        decision = {
            "round": current_round,
            "previous_baseline": self.label(),
            "next_baseline": self.label(),
            "rationale": "Dynamic aggregation disabled.",
            "llm_time": 0.0,
            "llm_model": self.llm_model,
        }
        if not self.enabled:
            self.history.append(copy.deepcopy(decision))
            return decision
        if current_round >= self.total_rounds:
            decision["rationale"] = f"Final round reached ({current_round}/{self.total_rounds}); no next aggregation switch required."
            self.history.append(copy.deepcopy(decision))
            return decision

        summary = self._summarize_latest_round(csv_path, current_round, metrics_history)
        if "llm" not in self.policy.lower():
            selected, rationale = self._decide_without_llm(summary)
            self.current_baseline = selected
            decision["next_baseline"] = self.label(selected)
            decision["rationale"] = rationale
            self.history.append(copy.deepcopy(decision))
            return decision

        prompt = self._build_prompt(summary)
        started = time.perf_counter()
        try:
            raw = self._call_ollama(prompt)
            selected, rationale = self._parse_decision(raw)
            self.current_baseline = selected
            decision["next_baseline"] = self.label(selected)
            decision["rationale"] = rationale or "LLM selected the next aggregation baseline without rationale."
        except Exception as exc:
            decision["rationale"] = f"LLM decision failed; keeping {self.label()}. Error: {exc!r}"
        finally:
            decision["llm_time"] = time.perf_counter() - started

        self.history.append(copy.deepcopy(decision))
        return decision

    def update_latest_csv_decision(self, csv_path: str, current_round: int, decision: Dict) -> None:
        if not os.path.exists(csv_path):
            return
        df = pd.read_csv(csv_path)
        if df.empty:
            return
        for col in ["Aggregation Baseline", "Next Aggregation Baseline", "Aggregation LLM Time (s)", "Aggregation Rationale"]:
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].astype("object")
        mask = pd.to_numeric(df.get("FL Round"), errors="coerce") == current_round if "FL Round" in df.columns else pd.Series([True] * len(df))
        idx = df[mask].tail(1).index
        if len(idx) == 0:
            idx = df.tail(1).index
        df.loc[idx, "Next Aggregation Baseline"] = decision.get("next_baseline", self.label())
        df.loc[idx, "Aggregation LLM Time (s)"] = f"{float(decision.get('llm_time') or 0.0):.2f}"
        df.loc[idx, "Aggregation Rationale"] = decision.get("rationale", "")
        df.to_csv(csv_path, index=False)
