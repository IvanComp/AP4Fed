#!/usr/bin/env python3
"""
Batch runner for AP4Fed-style simulations.

Features:
  • Uses your existing docker-compose.yml by default
  • Optional --dynamic compose generation (replicates 'client' per client_details)
  • NEW: --repeat to run each config N times
  • NEW: --seed-field / --seed-start / --seed-step to inject a seed into the config
  • NEW: --set key=value (repeatable) to override JSON fields (dot.path supported)

Usage examples:
  # 6 configs × 10 runs each, seed injected at config['seed']=100..109
  python run_batch.py configs/*.json --compose Docker/docker-compose.yml \
         --repeat 10 --seed-field seed --seed-start 100

  # Apply static overrides (dot path) for all runs
  python run_batch.py configs/*.json --set agent.policy=zero-shot --set objective=accuracy

  # Dynamic compose when needed
  python run_batch.py configs/*.json --dynamic

Notes:
  • Artifacts land in experiments/<timestamp>_<config>_rXX/
  • A manifest.json is written per run with seed/overrides/compose info.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple

BASE_DIR = Path(__file__).resolve().parent

def load_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def dump_json(p: Path, data: Dict[str, Any]):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def safe_name(p: Path) -> str:

def exp_label(p: Path) -> str:
    # If the file is named 'config.json', use its parent folder as the label (e.g., Experiments/zero-shot/config.json -> zero-shot)
    if p.name.lower() == "config.json":
        return safe_name(p.parent)
    return safe_name(p)

    return p.stem.replace(" ", "_")

def stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def copy_tree_if_exists(src: Path, dst: Path):
    if src.exists():
        shutil.copytree(src, dst / src.name)

def clean_dir_if_exists(p: Path):
    if p.exists():
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()

def prepare_config_for_run(cfg_obj: Dict[str, Any], sim_type: str) -> Path:
    """
    Writes the provided cfg_obj into the location expected by the tool:
    - Docker/configuration/config.json  or
    - Local/configuration/config.json
    Returns the destination path.
    """
    dest_dir = BASE_DIR / sim_type / "configuration"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "config.json"
    dump_json(dest, cfg_obj)
    return dest

def set_by_path(d: Dict[str, Any], path: str, value: Any):
    """Set d['a']['b']... using dot-separated path like 'a.b.c'"""
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value

def parse_override(s: str) -> Tuple[str, Any]:
    # Accept int/float/bool/null when possible, else keep string
    if "=" not in s:
        raise ValueError(f"--set expects key=value, got: {s}")
    k, v = s.split("=", 1)
    v = v.strip()
    # Try JSON parsing for the value
    try:
        v_parsed = json.loads(v)
    except json.JSONDecodeError:
        v_parsed = v
    return k.strip(), v_parsed

def build_dynamic_compose(cfg: Dict[str, Any], compose_in: Path, compose_out: Path):
    """
    Builds a docker-compose.dynamic.yml derived from docker-compose.yml,
    replicating the single 'client' service into N clientX services with
    per-client CPU/MEM limits and environment.
    """
    try:
        import yaml  # requires PyYAML
    except ImportError as e:
        raise RuntimeError("PyYAML is required for --dynamic mode (pip install pyyaml)") from e

    compose = yaml.safe_load(compose_in.read_text(encoding="utf-8"))

    services = compose.get("services") or {}
    server = services.get("server")
    template = services.get("client")
    if not server or not template:
        raise RuntimeError("Missing 'server' or 'client' in base compose: cannot build dynamic compose.")

    new_services = {"server": server}
    for d in cfg.get("client_details", []):
        cid = int(d.get("client_id", 1))
        cpu = d.get("cpu", 1)
        ram = d.get("ram", 2)

        svc = json.loads(json.dumps(template))
        svc.pop("image", None)
        svc.pop("container_name", None)
        if "deploy" in svc:
            svc.pop("deploy")

        svc["container_name"] = f"Client{cid}"
        if cpu:
            svc["cpus"] = cpu
        if ram:
            svc["mem_limit"] = f"{ram}g"

        env = svc.setdefault("environment", {})
        env["CLIENT_ID"] = str(cid)
        if "NUM_ROUNDS" not in env and "rounds" in cfg:
            env["NUM_ROUNDS"] = str(cfg.get("rounds"))
        if "NUM_CPUS" not in env:
            env["NUM_CPUS"] = str(cpu)
        if "NUM_RAM" not in env:
            env["NUM_RAM"] = str(ram)

        new_services[f"client{cid}"] = svc

    compose["services"] = new_services
    compose_out.write_text(yaml.safe_dump(compose, sort_keys=False), encoding="utf-8")

def run_one_config(cfg_src_path: Path, cfg_obj: Dict[str, Any], tag: str, repeat_idx: int,
                   num_supernodes: int, keep: bool,
                   compose_file: Path, use_dynamic: bool, compose_project: str,
                   seed_used: Any, overrides_used: Dict[str, Any]) -> Path:

    sim_type = str(cfg_obj.get("simulation_type", "Docker"))
    if sim_type not in ("Docker", "Local"):
        raise ValueError(f"simulation_type must be 'Docker' or 'Local', got: {sim_type}")

    # Place config where the app expects it
    dest_cfg = prepare_config_for_run(cfg_obj, sim_type)

    run_id_base = f"{stamp()}_{exp_label(cfg_src_path)}"
    if tag:
        run_id_base = f"{stamp()}_{tag}_{exp_label(cfg_src_path)}"
    rtag = f"r{repeat_idx:02d}" if repeat_idx is not None else "r00"
    run_id = f"{run_id_base}_{rtag}"

    out_dir = BASE_DIR / "experiments" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if sim_type == "Docker":
        work_dir = BASE_DIR / "Docker"
        base_compose = compose_file if compose_file else (work_dir / "docker-compose.yml")
        compose_to_use = base_compose

        if use_dynamic:
            compose_out = work_dir / "docker-compose.dynamic.yml"
            build_dynamic_compose(cfg_obj, base_compose, compose_out)
            compose_to_use = compose_out

        # Clean previous outputs
        clean_dir_if_exists(work_dir / "performance")
        clean_dir_if_exists(work_dir / "model_weights")
        (work_dir / "logs").mkdir(exist_ok=True)

        cmd = [
            "docker", "compose",
            "-f", str(compose_to_use),
            "up",
            "--build",
            "--remove-orphans",
            "--abort-on-container-exit"
        ]

        print(f"[{run_id}] Starting Docker simulation for {cfg_src_path} using {compose_to_use.name} ...", flush=True)
        env = os.environ.copy()
        if compose_project:
            env["COMPOSE_PROJECT_NAME"] = compose_project
        proc = subprocess.run(cmd, cwd=str(work_dir), env=env)
        print(f"[{run_id}] Docker exited with code {proc.returncode}", flush=True)

        # Save artifacts
        copy_tree_if_exists(work_dir / "performance", out_dir)
        copy_tree_if_exists(work_dir / "model_weights", out_dir)
        logs_dir = work_dir / "logs"
        if logs_dir.exists():
            shutil.copytree(logs_dir, out_dir / "logs", dirs_exist_ok=True)

        if not keep:
            subprocess.run(["docker", "compose", "-f", str(compose_to_use), "down", "-v"], cwd=str(work_dir))

    else:  # Local
        work_dir = BASE_DIR / "Local"
        clean_dir_if_exists(work_dir / "performance")
        clean_dir_if_exists(work_dir / "model_weights")

        cmd = [
            "flower-simulation",
            "--server-app", "server:app",
            "--client-app", "client:app",
            "--num-supernodes", str(num_supernodes),
        ]
        print(f"[{run_id}] Starting Local simulation for {cfg_src_path} ...", flush=True)
        proc = subprocess.run(cmd, cwd=str(work_dir))
        print(f"[{run_id}] Local simulation exited with code {proc.returncode}", flush=True)

        copy_tree_if_exists(work_dir / "performance", out_dir)
        copy_tree_if_exists(work_dir / "model_weights", out_dir)

    manifest = {
        "run_id": run_id,
        "config_src": str(cfg_src_path),
        "config_dest": str(dest_cfg),
        "simulation_type": sim_type,
        "compose_used": str(compose_file) if sim_type == "Docker" else None,
        "dynamic_compose": use_dynamic if sim_type == "Docker" else None,
        "num_supernodes": num_supernodes if sim_type == "Local" else None,
        "seed": seed_used,
        "overrides": overrides_used,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    dump_json(out_dir / "manifest.json", manifest)

    print(f"[{run_id}] Artifacts saved under: {out_dir}", flush=True)
    return out_dir

def main():
    ap = argparse.ArgumentParser(description="Batch runner for AP4Fed-like simulations")
    ap.add_argument("configs", nargs="+", help="List of config.json files (supports shell globs)")
    ap.add_argument("--tag", default="", help="Optional tag prefix for run folders")
    ap.add_argument("--num-supernodes", type=int, default=2, help="Only for Local mode")
    ap.add_argument("--keep", action="store_true", help="Docker only: keep containers running after a run")
    ap.add_argument("--compose", type=str, default="Docker/docker-compose.yml", help="Path to existing compose file (Docker mode)")
    ap.add_argument("--dynamic", action="store_true", help="Generate docker-compose.dynamic.yml from base compose (requires PyYAML)")
    ap.add_argument("--compose-project", type=str, default="", help="COMPOSE_PROJECT_NAME to isolate runs")

    # New repetition / seed / overrides
    ap.add_argument("--repeat", type=int, default=1, help="Number of repetitions per config")
    ap.add_argument("--seed-field", type=str, default="", help="Dot path to seed field to set (e.g., 'seed' or 'trainer.seed')")
    ap.add_argument("--seed-start", type=int, default=0, help="Initial seed value when using --seed-field")
    ap.add_argument("--seed-step", type=int, default=1, help="Increment for seed per repetition")
    ap.add_argument("--set", dest="overrides", action="append", default=[], help="Override JSON field: key=value (repeatable)")

    args = ap.parse_args()

    # Expand globs manually to preserve ordering
    cfg_paths: List[Path] = []
    for patt in args.configs:
        expanded = sorted(Path(".").glob(patt)) if any(ch in patt for ch in "*?[]") else [Path(patt)]
        cfg_paths.extend([Path(p).resolve() for p in expanded])

    if not cfg_paths:
        print("No config files found.", file=sys.stderr)
        sys.exit(2)

    # Parse static overrides once
    static_overrides: Dict[str, Any] = {}
    for ov in args.overrides:
        k, v = parse_override(ov)
        static_overrides[k] = v

    failures = 0
    for cfg_path in cfg_paths:
        base_cfg = load_json(cfg_path)

        for i in range(args.repeat):
            cfg = json.loads(json.dumps(base_cfg))  # deep copy

            # Apply seed if requested
            seed_used = None
            if args.seed_field:
                seed_used = args.seed_start + i * args.seed_step
                set_by_path(cfg, args.seed_field, seed_used)

            # Apply static overrides
            for k, v in static_overrides.items():
                set_by_path(cfg, k, v)

            try:
                run_one_config(
                    cfg_src_path=cfg_path,
                    cfg_obj=cfg,
                    tag=args.tag,
                    repeat_idx=i + 1 if args.repeat > 1 else 0,
                    num_supernodes=args.num_supernodes,
                    keep=args.keep,
                    compose_file=Path(args.compose).resolve(),
                    use_dynamic=args.dynamic,
                    compose_project=args.compose_project,
                    seed_used=seed_used,
                    overrides_used=static_overrides,
                )
            except Exception as e:
                failures += 1
                print(f"[ERROR] {cfg_path} (rep {i+1}/{args.repeat}): {e}", file=sys.stderr)
            time.sleep(2)

    sys.exit(1 if failures else 0)

if __name__ == "__main__":
    main()
