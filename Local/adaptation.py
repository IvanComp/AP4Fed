import json
import os
from logging import INFO
from typing import Dict

from adaptation_utils.strategy import get_patterns, get_activation_criteria, ActivationCriterion
from logger import log

# Path to the 'configuration' directory
current_dir = os.getcwd().replace('/adaptation', '')
config_dir = os.path.join(current_dir, 'configuration')
config_file = os.path.join(config_dir, 'config.json')

adaptation_config_file = os.path.join(config_dir, 'config.json')
PATTERNS = [
    "client_selector",
    "client_cluster",
    "message_compressor",
    "model_co-versioning_registry",
    "multi-task_model_trainer",
    "heterogeneous_data_handler"
]


def get_model_type(default_config):
    # FIXME: works only if all clients have the same model type
    return default_config['client_details'][0]['model']


class AdaptationManager:
    def __init__(self, enabled, default_config):
        self.name = 'AdaptationManager'
        self.enabled = enabled
        self.default_config = default_config

        if self.enabled:
            adaptation_config = json.load(open(adaptation_config_file, 'r'))

            self.patterns = get_patterns(adaptation_config)

            pattern_act_criteria = get_activation_criteria(adaptation_config, default_config)
            self.adaptation_criteria: Dict[str, ActivationCriterion] = {c.pattern: c for c in pattern_act_criteria}

            self.model_type = get_model_type(default_config)

            self.cached_config = {"patterns": {p: {"enabled": False} for p in PATTERNS}}
            self.update_config(default_config)
            self.update_json(default_config)

            self.cached_aggregated_metrics = None

    def describe(self):
        return log(INFO, '\n'.join([str(cr) for cr in self.adaptation_criteria.values()]))

    def update_metrics(self, new_aggregated_metrics):
        self.cached_aggregated_metrics = new_aggregated_metrics

    def update_config(self, new_config):
        for pattern in new_config["patterns"]:
            if pattern in self.cached_config["patterns"]:
                if 'enabled' in self.cached_config["patterns"][pattern]:
                    self.cached_config["patterns"][pattern]['enabled'] = new_config["patterns"][pattern]['enabled']
                else:
                    self.cached_config["patterns"][pattern] = {'enabled': new_config["patterns"][pattern]['enabled'],
                                                               'params': new_config["patterns"][pattern].get('params',
                                                                                                             {})}

                if 'params' in self.cached_config["patterns"][pattern]:
                    self.cached_config["patterns"][pattern]['params'] = new_config["patterns"][pattern].get('params',
                                                                                                            {})
            else:
                self.cached_config["patterns"][pattern] = {'enabled': new_config["patterns"][pattern]['enabled'],
                                                           'params': new_config["patterns"][pattern].get('params', {})}

    def update_json(self, new_config):
        with open(config_file, 'r') as f:
            config = json.load(f)
            for pattern in new_config['patterns']:
                config['patterns'][pattern]['enabled'] = new_config['patterns'][pattern]['enabled']
                if 'params' in new_config['patterns'][pattern]:
                    config['patterns'][pattern]['params'] = new_config['patterns'][pattern]['params']
            json.dump(config, open(config_file, 'w'), indent=4)

    def config_next_round(self, new_aggregated_metrics, last_round_time):
        if self.enabled:
            log(INFO, f"{self.name}: Configuring next round...")
            log(INFO, self.default_config["patterns"])
            log(INFO, self.cached_config["patterns"])
        else:
            return self.default_config["patterns"]

        new_config = self.cached_config.copy()

        for p_i, pattern in enumerate(self.patterns):
            if self.default_config["patterns"][pattern]['enabled'] and pattern in self.adaptation_criteria:
                args = {"model_type": self.model_type, "metrics": new_aggregated_metrics, "time": last_round_time}
                # FIXME: activation criteria may change from pattern to pattern
                activate, args, expl = self.adaptation_criteria[pattern].activate_pattern(args)

                if not activate:
                    new_config["patterns"][pattern]['enabled'] = False
                    if args is not None:
                        new_config["patterns"][pattern]['params'] = args
                    log(INFO, expl)
                else:
                    new_config["patterns"][pattern]['enabled'] = True
                    if args is not None:
                        new_config["patterns"][pattern]['params'] = args
                    log(INFO, expl)

        self.update_metrics(new_aggregated_metrics)
        self.update_config(new_config)
        self.update_json(new_config)

        return new_config["patterns"]
