import json
import os
import random
import copy
from logging import INFO
from typing import Dict, List, Tuple
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


class AdaptationManager:
    def __init__(self, enabled: bool, default_config: Dict):
        self.name = "AdaptationManager"

        self.default_config = default_config
        self.policy = str(default_config.get("adaptation", "None")).strip()
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
        if pol == "random":
            return self._decide_random(base_config, current_round)
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
