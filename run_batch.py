import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# --------------------- utilities ---------------------

def load_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def exp_label(p: Path) -> str:
    return p.parent.name if p.name.lower() == "config.json" else p.stem

def stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def clean(p: Path):
    if p.exists():
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)

def sanitize_project(tag: str) -> str:
    tag = tag or ""
    tag = re.sub(r"[^a-zA-Z0-9_.-]+", "-", tag)
    return tag[:50] or f"batch-{stamp()}"

def derive_env_from_cfg(cfg: Dict[str, Any]) -> Dict[str, str]:
    """Deriva env vari best-effort (sicuro anche se mancano)."""
    env: Dict[str, str] = {}
    if "num_rounds" in cfg:
        env["NUM_ROUNDS"] = str(cfg["num_rounds"])
    if "dataset" in cfg:
        env["DATASET_NAME"] = str(cfg["dataset"])
    ap = cfg.get("architectural_patterns") or {}
    env["CLIENT_SELECTOR_ENABLED"] = "1" if ap.get("client_selector") else "0"
    env["MESSAGE_COMPRESSOR_ENABLED"] = "1" if ap.get("message_compressor") else "0"
    env["HETEROGENEOUS_DATA_HANDLER_ENABLED"] = "1" if ap.get("heterogeneous_data_handler") else "0"
    return env

def generate_compose_from_cfg(cfg: Dict[str, Any], template_path: Path, out_path: Path) -> None:
    """Per ora copia il template così com’è."""
    if template_path.resolve() != out_path.resolve():
        shutil.copy2(template_path, out_path)

# --------------------- core ---------------------

def run_one(cfg_path: Path, compose_template: Path, repeat_idx: int, tag: str, keep: bool) -> Path:
    work_dir = compose_template.parent.resolve()
    cfg = load_json(cfg_path)

    (work_dir / "configuration").mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, work_dir / "configuration" / "config.json")

    for d in ("performance", "model_weights", "logs"):
        clean(work_dir / d)
    (work_dir / "logs").mkdir(exist_ok=True)

    dyn_compose = work_dir / f"docker-compose.dynamic.{exp_label(cfg_path)}.yml"
    generate_compose_from_cfg(cfg, compose_template, dyn_compose)

    env = os.environ.copy()
    env.update(derive_env_from_cfg(cfg))
    env["COMPOSE_PROJECT_NAME"] = sanitize_project((tag + "-" if tag else "") + exp_label(cfg_path))

    # down sempre prima
    subprocess.run(
        ["docker", "compose", "-f", str(dyn_compose), "down", "-v", "--remove-orphans"],
        cwd=str(work_dir), check=False
    )

    run_label = f"{stamp()}_{exp_label(cfg_path)}_r{repeat_idx:02d}"
    print(f"[{run_label}] compose={dyn_compose.name}", flush=True)

    dest_csv_dir = cfg_path.parent
    dest_csv_dir.mkdir(parents=True, exist_ok=True)
    dest_csv = dest_csv_dir / f"r{repeat_idx}.csv"

    err: Exception | None = None
    try:
        SERVER_SERVICE = "flwr_server"
        cmd_up = [
            "docker", "compose", "-f", str(dyn_compose), "up",
            "--build", "--remove-orphans",
            "--abort-on-container-exit",
            "--exit-code-from", SERVER_SERVICE,
        ]
        proc = subprocess.run(cmd_up, cwd=str(work_dir), env=env, check=False)
        rc = proc.returncode
        print(f"[{run_label}] exit={rc}", flush=True)
        if rc != 0:
            err = RuntimeError(f"Docker compose exited with code {rc}")

        if err is None:
            perf_dir = work_dir / "performance"
            src_csv = perf_dir / "FLwithAP_performance_metrics.csv"
            if not src_csv.exists():
                candidates = sorted(perf_dir.glob("FLwithAP_performance_metrics*.csv"))
                if candidates:
                    src_csv = candidates[0]
            if src_csv.exists():
                shutil.copy2(src_csv, dest_csv)
                print(f"[{run_label}] saved -> {dest_csv}", flush=True)
            else:
                print(f"[{run_label}] WARNING: performance CSV not found in {perf_dir}", flush=True)
    finally:
        subprocess.run(
            ["docker", "compose", "-f", str(dyn_compose), "down", "-v", "--remove-orphans"],
            cwd=str(work_dir), check=False
        )
        if not keep:
            for d in ("performance", "model_weights", "logs"):
                clean(work_dir / d)

    if err:
        raise err
    return dest_csv_dir

# --------------------- CLI ---------------------

def main():
    ap = argparse.ArgumentParser(description="Run batch FL experiments with Docker Compose.")
    ap.add_argument("configs", nargs="+", help="config.json path(s) or glob(s)")
    ap.add_argument("--compose", required=True, type=Path, help="Path to docker-compose template file")
    ap.add_argument("--repeat", type=int, default=1, help="Repeat each config N times")
    ap.add_argument("--tag", type=str, default="", help="Optional tag for COMPOSE_PROJECT_NAME")
    ap.add_argument("--keep", action="store_true", help="Keep performance/model_weights/logs after each run")
    args = ap.parse_args()

    compose_template = args.compose.resolve()

    cfg_paths: List[Path] = []
    for patt in args.configs:
        if any(ch in patt for ch in "*?[]"):
            for q in sorted(Path().glob(patt)):
                if q.is_file():
                    cfg_paths.append(q.resolve())
        else:
            p = Path(patt).resolve()
            if p.exists():
                cfg_paths.append(p)

    if not cfg_paths:
        print("No config files found.", file=sys.stderr)
        sys.exit(2)

    failures = 0
    for cfg in cfg_paths:
        for i in range(1, args.repeat + 1):
            run_label = f"{exp_label(cfg)} r{i}"
            try:
                run_one(cfg, compose_template, i, args.tag, args.keep)
            except Exception as e:
                failures += 1
                print(f"[ERROR] {run_label}: {e}", file=sys.stderr)
                continue
            time.sleep(1)

    sys.exit(1 if failures else 0)

if __name__ == "__main__":
    main()