#!/usr/bin/env python3
import argparse, json, os, re, shutil, subprocess, sys, time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
def load_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)
def exp_label(p: Path) -> str:
    return p.parent.name if p.name.lower() == "config.json" else p.stem
def stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")
def copy_dir(src: Path, dst_parent: Path):
    if src.exists():
        shutil.copytree(src, dst_parent / src.name)
def clean(p: Path):
    if p.exists():
        shutil.rmtree(p) if p.is_dir() else p.unlink()
def sanitize_project(name: str) -> str:
    s = re.sub(r"[^a-z0-9_-]", "", (name or "expbatch").lower())
    return s if s and s[0].isalnum() else "expbatch"
def derive_env_from_cfg(cfg: Dict[str, Any]) -> Dict[str,str]:
    envv = {}
    rounds = cfg.get("rounds", 1)
    try: envv["NUM_ROUNDS"] = str(int(rounds))
    except: envv["NUM_ROUNDS"] = "1"
    return envv
def generate_compose_from_cfg(cfg: Dict[str, Any], template_path: Path, out_path: Path):
    # Build a dynamic compose with N clients using client_details
    import yaml, copy as _copy
    tpl = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    services = tpl.get("services", {})
    # keep server and network as-is
    server = services.get("server", {})
    networks = tpl.get("networks", {"flwr_network":{"driver":"bridge"}})
    # find a client template (prefer client1, fallback to any 'client' service)
    client_tpl = services.get("client1") or next((services[k] for k in services if k.startswith("client")), None)
    if client_tpl is None:
        raise RuntimeError("No client template found in compose template")
    # purge existing clients
    for k in list(services.keys()):
        if k.startswith("client"):
            services.pop(k, None)
    # build clients from config
    cds = cfg.get("client_details") or []
    if not isinstance(cds, list) or not cds:
        raise RuntimeError("config.json missing 'client_details' list")
    for i, c in enumerate(cds, start=1):
        ci = _copy.deepcopy(client_tpl)
        ci["container_name"] = f"Client{i}"
        # resources
        cpu = c.get("cpu", c.get("cpus", 1))
        ram = c.get("ram", c.get("memory", 2))
        ci["cpus"] = float(cpu) if "cpus" in ci or True else cpu  # ensure present
        # memory unit
        ram_str = str(ram).lower()
        ci["mem_limit"] = ram_str if ram_str.endswith(("g","m")) else f"{int(float(ram))}g"
        # environment
        env = ci.setdefault("environment", {})
        env["CLIENT_ID"] = str(c.get("client_id", i))
        env["NUM_CPUS"] = str(int(float(cpu))) if cpu is not None else "1"
        env["NUM_RAM"] = ci["mem_limit"]
        services[f"client{i}"] = ci
    # ensure server exists
    services["server"] = server
    tpl["services"] = services
    tpl["networks"] = networks
    out_path.write_text(yaml.safe_dump(tpl, sort_keys=False), encoding="utf-8")
    return out_path
def run_one(cfg_path: Path, compose_template: Path, repeat_idx: int, tag: str, keep: bool) -> Path:
    cfg = load_json(cfg_path)
    work_dir = compose_template.parent.resolve()
    # put config where simulation.py expects it: Docker/configuration/config.json
    (work_dir / "configuration").mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, work_dir / "configuration" / "config.json")
    base = f"{stamp()}_{exp_label(cfg_path)}_r{repeat_idx:02d}"
    if tag: base = f"{stamp()}_{tag}_{exp_label(cfg_path)}_r{repeat_idx:02d}"
    out_dir = cfg_path.parent / base
    out_dir.mkdir(parents=True, exist_ok=True)
    # clean output dirs
    clean(work_dir / "performance"); clean(work_dir / "model_weights"); (work_dir / "logs").mkdir(exist_ok=True)
    # generate per-run compose with N clients from config
    dyn_compose = work_dir / f"docker-compose.dynamic.{exp_label(cfg_path)}.yml"
    generate_compose_from_cfg(cfg, compose_template, dyn_compose)
    # write .env for NUM_ROUNDS like simulation.py environment style
    env = os.environ.copy()
    env.update(derive_env_from_cfg(cfg))
    env["COMPOSE_PROJECT_NAME"] = sanitize_project(tag or exp_label(cfg_path))
    (work_dir / ".env").write_text("\n".join([f"{k}={v}" for k,v in env.items() if k in ("NUM_ROUNDS","NUM_CPUS","NUM_RAM")])+"\n", encoding="utf-8")
    # always down before up to avoid /flwr_server conflicts
    subprocess.run(["docker","compose","-f",str(dyn_compose),"down","-v","--remove-orphans"], cwd=str(work_dir))
    print(f"[{base}] compose={dyn_compose.name}", flush=True)
    proc = subprocess.run(["docker","compose","-f",str(dyn_compose),"up","--build","--remove-orphans","--abort-on-container-exit"], cwd=str(work_dir), env=env)
    print(f"[{base}] exit={proc.returncode}", flush=True)
    # always down after to free names
    subprocess.run(["docker","compose","-f",str(dyn_compose),"down","-v"], cwd=str(work_dir))
    copy_dir(work_dir / "performance", out_dir); copy_dir(work_dir / "model_weights", out_dir)
    if (work_dir / "logs").exists():
        shutil.copytree(work_dir / "logs", out_dir / "logs", dirs_exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps({"run_id": base,"config_src": str(cfg_path),"compose_used": str(dyn_compose)}, indent=2), encoding="utf-8")
    return out_dir
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+", help="Paths/globs to Experiments/*/config.json")
    ap.add_argument("--compose", required=True, help="Path to Docker/docker-compose.dynamic.yml (template)")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--tag", default="dyn")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    compose_template = Path(args.compose).resolve()
    cfg_paths: List[Path] = []
    for patt in args.configs:
        if any(ch in patt for ch in "*?[]"):
            cfg_paths.extend(sorted(Path().glob(patt)))
        else:
            cfg_paths.append(Path(patt))
    cfg_paths = [p.resolve() for p in cfg_paths]
    if not cfg_paths:
        print("No config files found.", file=sys.stderr); sys.exit(2)
    failures = 0
    for cfg in cfg_paths:
        for i in range(1, args.repeat + 1):
            try: run_one(cfg, compose_template, i, args.tag, args.keep)
            except Exception as e:
                failures += 1; print(f"[ERROR] {cfg} r{i}: {e}", file=sys.stderr)
            time.sleep(1)
    sys.exit(1 if failures else 0)
if __name__ == "__main__":
    main()
