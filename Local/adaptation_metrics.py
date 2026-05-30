import glob
import os
import re

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
