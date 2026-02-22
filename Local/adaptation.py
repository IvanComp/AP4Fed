import json
import os
import glob
import re
import pandas as pd
from logging import INFO
from typing import Dict, List
from flwr.common.logger import log
# from agent import suggest_next_patterns

current_dir = os.getcwd().replace('/adaptation', '')
config_dir = os.path.join(current_dir, 'configuration')
config_file = os.path.join(config_dir, 'config.json')
adaptation_config_file = os.path.join(config_dir, 'config.json')

def _sa_latest_round_csv(performance_dir):
    files = glob.glob(f"{performance_dir}/FLwithAP_performance_metrics_round*.csv", recursive=True)
    if not files:
        return None, None
    def rnum(p):
        m = re.search(r"round(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else -1
    lastf = max(files, key=rnum)
    return rnum(lastf), lastf

def _sa_aggregate_round(df):
    def _find(colnames):
        for c in df.columns:
            lc = str(c).strip().lower()
            for pat in colnames:
                if callable(pat):
                    if pat(lc): return c
                else:
                    if pat in lc: return c
        return None

    col_round = _find([lambda s: "round" in s])
    col_cid   = _find([lambda s: ("client" in s and "id" in s) or s.strip() == "client id"])
    col_tr    = _find([lambda s: ("training" in s and "time" in s), "training (s)", "training time (s)", "training time"])
    col_cm    = _find([lambda s: ("comm" in s and "time" in s) or ("communication" in s)])
    col_tt    = _find([lambda s: ("total time of fl round" in s) or ("total" in s and "round" in s)])
    col_f1    = _find([lambda s: "val f1" in s or s == "f1"])
    col_jsd   = _find([lambda s: "jsd" == str(s).strip().lower()])

    dfl = df
    if col_round and col_round in df.columns:
        try:
            last_r = int(df[col_round].dropna().astype(int).max())
            dfl = df[df[col_round].astype(int) == last_r].copy()
        except Exception:
            dfl = df.copy()

    per_client = dfl.copy()
    if col_cid and col_cid in per_client.columns:
        per_client = per_client[per_client[col_cid].notna()].copy()

    def _series(dfx, col):
        if not col or col not in dfx.columns: return []
        out = []
        for v in dfx[col].tolist():
            try: out.append(float(v))
            except Exception: pass
        return out

    tr_seq = _series(per_client, col_tr)
    cm_seq = _series(per_client, col_cm)

    def _last_non_nan(dfx, col):
        if not col or col not in dfx.columns: return None
        vals = []
        for v in dfx[col].tolist():
            try: vals.append(float(v))
            except Exception: pass
        return vals[-1] if vals else None

    f1_last = _last_non_nan(dfl, col_f1)
    tt_last = _last_non_nan(dfl, col_tt)
    jsd_last = _last_non_nan(dfl, col_jsd)

    def _agg(seq):
        if not seq: return {"count": 0, "mean": None, "min": None, "max": None}
        return {"count": len(seq), "mean": sum(seq) / len(seq), "min": min(seq), "max": max(seq)}

    tr_agg = _agg(tr_seq)
    cm_agg = _agg(cm_seq)

    return {
        "round": int(dfl[col_round].iloc[0]) if col_round and len(dfl) else None,
        "mean_f1": f1_last,
        "mean_total_time": tt_last,
        "mean_training_time": tr_agg["mean"],
        "mean_comm_time": cm_agg["mean"],
        "mean_jsd": jsd_last,
        "training_time_stats": tr_agg,
        "comm_time_stats": cm_agg
    }

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
            self.policy = adaptation_config.get("adaptation", "None")

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

        # Copia dello stato corrente dei pattern
        new_config = self.cached_config.copy()

        # 1) Logica base/stattica: per ogni pattern che è previsto nel config
        for pattern in self.patterns:
            if self.default_config["patterns"][pattern]["enabled"] and pattern in self.adaptation_criteria:
                args = {
                    "model_type": self.model_type,
                    "metrics": new_aggregated_metrics,
                    "time": last_round_time,
                }

                activate, params, expl = self.adaptation_criteria[pattern].activate_pattern(args)

                new_config["patterns"][pattern]["enabled"] = activate
                if params is not None:
                    new_config["patterns"][pattern]["params"] = params
                log(INFO, expl)
            else:
                # pattern non gestito dalla politica statica
                pass

        # 1.5) Expert-Driven Policy
        pol = getattr(self, "policy", "").lower().strip()
        log(INFO, f"Current Policy is: '{pol}'")
        if "expert-driven" in pol or "expert" in pol:
            performance_dir = os.path.join(current_dir, "performance")
            last_r, last_f = _sa_latest_round_csv(performance_dir)
            if last_f:
                try:
                    df = pd.read_csv(last_f)
                    agg = _sa_aggregate_round(df)
                    
                    f1_last = agg.get("mean_f1")
                    rt_last = agg.get("mean_total_time")
                    rt_comm = agg.get("mean_comm_time")
                    jsd_last = agg.get("mean_jsd")
                    
                    f1_last = float(f1_last) if f1_last is not None else 0.0
                    rt_last = float(rt_last) if rt_last is not None else 0.0
                    rt_comm = float(rt_comm) if rt_comm is not None else 0.0
                    jsd_last = float(jsd_last) if jsd_last is not None else 0.0

                    if rt_last > 0 and (f1_last / rt_last) < 0.005:
                        if "client_selector" in new_config.get("patterns", {}):
                            new_config["patterns"]["client_selector"]["enabled"] = False
                            log(INFO, f"Client Selector: OFF (f1_last={f1_last:.3f} / rt_last={rt_last:.3f} < 0.005)")
                    else:
                        if "client_selector" in new_config.get("patterns", {}):
                            new_config["patterns"]["client_selector"]["enabled"] = True
                            log(INFO, f"Client Selector: ON (f1_last={f1_last:.3f} / rt_last={rt_last:.3f} >= 0.005)")

                    if jsd_last > 0.5:
                        if "heterogeneous_data_handler" in new_config.get("patterns", {}):
                            new_config["patterns"]["heterogeneous_data_handler"]["enabled"] = True
                            log(INFO, f"Heterogeneous Data Handler: ON (jsd={jsd_last:.3f} > 0.5)")
                    else:
                        if "heterogeneous_data_handler" in new_config.get("patterns", {}):
                            new_config["patterns"]["heterogeneous_data_handler"]["enabled"] = False
                            log(INFO, f"Heterogeneous Data Handler: OFF (jsd={jsd_last:.3f} <= 0.5)")

                    if rt_comm > 2.0:
                        if "message_compressor" in new_config.get("patterns", {}):
                            new_config["patterns"]["message_compressor"]["enabled"] = True
                            log(INFO, f"Message Compressor: ON (rt_comm={rt_comm:.3f} > 2.0)")
                    else:
                        if "message_compressor" in new_config.get("patterns", {}):
                            new_config["patterns"]["message_compressor"]["enabled"] = False
                            log(INFO, f"Message Compressor: OFF (rt_comm={rt_comm:.3f} <= 2.0)")
                except Exception as e:
                    log(INFO, f"Expert-Driven layer error -> {e}")

        # 2) LIVELLO AGENTI (AI-Agents)
        #    Qui entra in gioco l'intelligenza multi-agente. Per ora è rule-based,
        #    ma l'entry point è già "suggest_next_patterns" dal file agent.py.
        try:
            perf_csv_path = os.path.join(
                current_dir,
                "performance",
                "FLwithAP_performance_metrics.csv",
            )

            # agent_proposal = suggest_next_patterns(
            #     metrics_history=new_aggregated_metrics,        # è metrics_history nel server
            #     current_patterns=new_config["patterns"],       # stato attuale dei pattern
            #     csv_path=perf_csv_path,                        # log round-by-round
            #     cooperation_mode="debate",                     # "debate" / "voting" / "role"
            #     round_time=last_round_time,                    # durata ultimo round
            # )
            agent_proposal = None

            if agent_proposal:
                log(INFO, f"{self.name}: AI-Agents proposal {agent_proposal}")

                # merge delle proposte degli agenti nel nostro new_config
                for patt, patt_cfg in agent_proposal.items():
                    if patt not in new_config["patterns"]:
                        new_config["patterns"][patt] = {}
                    if "enabled" in patt_cfg:
                        new_config["patterns"][patt]["enabled"] = patt_cfg["enabled"]
                    if "params" in patt_cfg:
                        new_config["patterns"][patt].setdefault("params", {})
                        new_config["patterns"][patt]["params"].update(
                            patt_cfg.get("params", {})
                        )
        except Exception as e:
            log(INFO, f"{self.name}: agent layer error -> {e}")

        # 3) Aggiorno cache interna e scrivo la config aggiornata su disco
        self.update_metrics(new_aggregated_metrics)
        self.update_config(new_config)
        self.update_json(new_config)

        # Ritorno solo la sezione patterns, perché è quello che server.config_patterns() si aspetta
        return new_config["patterns"]
