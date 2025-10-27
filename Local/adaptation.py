import json
import os
from logging import INFO
from typing import Dict, List
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
    Lo stato di riferimento è quello presente in default_config["patterns"].
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
    # FIXME: works only if all clients have the same model type
    return default_config['client_details'][0]['model']


class AdaptationManager:
    def __init__(self, enabled: bool, default_config: Dict):
        self.name = 'AdaptationManager'
        self.enabled = enabled
        self.default_config = default_config

        if self.enabled:
            adaptation_config = json.load(open(adaptation_config_file, 'r'))

            # Pattern sotto controllo adattivo
            self.patterns = get_patterns(adaptation_config)

            # Strategia statica incorporata qui
            pattern_act_criteria = get_activation_criteria(adaptation_config, default_config)
            self.adaptation_criteria: Dict[str, ActivationCriterion] = {
                c.pattern: c for c in pattern_act_criteria
            }

            self.model_type = get_model_type(default_config)

            # Cache interna dello stato corrente dei pattern
            self.cached_config = {
                "patterns": {p: {"enabled": False} for p in PATTERNS}
            }

            # Allineo cache e file con la config iniziale
            self.update_config(default_config)
            self.update_json(default_config)

            self.cached_aggregated_metrics = None

    def describe(self):
        if not self.enabled:
            return log(INFO, f"{self.name}: adaptation disabled")
        return log(
            INFO,
            '\n'.join([str(cr) for cr in self.adaptation_criteria.values()])
        )

    def update_metrics(self, new_aggregated_metrics: Dict):
        self.cached_aggregated_metrics = new_aggregated_metrics

    def update_config(self, new_config: Dict):
        for pattern in new_config["patterns"]:
            if pattern in self.cached_config["patterns"]:
                # aggiorno enabled
                if 'enabled' in self.cached_config["patterns"][pattern]:
                    self.cached_config["patterns"][pattern]['enabled'] = new_config["patterns"][pattern]['enabled']
                else:
                    self.cached_config["patterns"][pattern] = {
                        'enabled': new_config["patterns"][pattern]['enabled'],
                        'params': new_config["patterns"][pattern].get('params', {})
                    }

                # aggiorno params
                if 'params' in self.cached_config["patterns"][pattern]:
                    self.cached_config["patterns"][pattern]['params'] = new_config["patterns"][pattern].get('params', {})
            else:
                self.cached_config["patterns"][pattern] = {
                    'enabled': new_config["patterns"][pattern]['enabled'],
                    'params': new_config["patterns"][pattern].get('params', {})
                }

    def update_json(self, new_config: Dict):
        with open(config_file, 'r') as f:
            config = json.load(f)

        for pattern in new_config['patterns']:
            config['patterns'][pattern]['enabled'] = new_config['patterns'][pattern]['enabled']
            if 'params' in new_config['patterns'][pattern]:
                config['patterns'][pattern]['params'] = new_config['patterns'][pattern]['params']

        json.dump(config, open(config_file, 'w'), indent=4)

    def config_next_round(self, new_aggregated_metrics: Dict, last_round_time: float):
        if not self.enabled:
            return self.default_config["patterns"]

        log(INFO, f"{self.name}: Configuring next round (static policy)...")
        log(INFO, self.default_config["patterns"])
        log(INFO, self.cached_config["patterns"])
        new_config = self.cached_config.copy()

        for pattern in self.patterns:
            if self.default_config["patterns"][pattern]['enabled'] and pattern in self.adaptation_criteria:
                args = {
                    "model_type": self.model_type,
                    "metrics": new_aggregated_metrics,
                    "time": last_round_time,
                }

                activate, params, expl = self.adaptation_criteria[pattern].activate_pattern(args)

                new_config["patterns"][pattern]['enabled'] = activate
                if params is not None:
                    new_config["patterns"][pattern]['params'] = params
                log(INFO, expl)
            else:
                pass

        self.update_metrics(new_aggregated_metrics)
        self.update_config(new_config)
        self.update_json(new_config)
        return new_config["patterns"]