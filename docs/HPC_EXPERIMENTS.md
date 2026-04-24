# AP4FED Headless Experiment Setup

This workflow is meant for machines where you only want to run the experiment campaigns and collect the CSV outputs.
Notebook analysis can stay on your local PC after you copy back the results folders.

## 1. Clone the repository

```bash
git clone <YOUR_REPO_URL>
cd AP4Fed
```

## 2. Create the experiment environment

```bash
./scripts/bootstrap_experiment_env.sh
source .venv-experiments/bin/activate
```

If your machine has a specific Python 3.12 binary:

```bash
./scripts/bootstrap_experiment_env.sh --python /path/to/python3.12
```

## 3. Verify the environment

Local-only preflight:

```bash
python scripts/verify_experiment_env.py --mode local
```

Docker preflight:

```bash
python scripts/verify_experiment_env.py --mode docker
```

If you want to validate the LLM endpoint too:

```bash
python scripts/verify_experiment_env.py --mode local --ollama-url http://127.0.0.1:11434
```

## 4. Run the campaigns

5-client setup:

```bash
python build_paper_experiments.py --mode local --rounds 100 --repeat 10
python build_paper_experiments.py --mode docker --rounds 100 --repeat 10
```

500-client, k=5 setup:

```bash
python build_paper_experiments_500clients_k5.py --mode local --rounds 100 --repeat 10
python build_paper_experiments_500clients_k5.py --mode docker --rounds 100 --repeat 10
```

## LLM-based baselines

The Python environment is only the Python side of the stack.
It does **not** install or launch the Ollama daemon for you.

This matters only for the agentic approaches:

- `voting-based`
- `role-based`
- `debate-based`

The non-LLM baselines (`never`, `random`, `expert-driven`) do not need Ollama.

### What the code expects

The runners pass an `ollama_base_url` into the experiment config and the adaptation layer talks to that endpoint over HTTP.
So you need **one** of these two setups on the supercomputer:

1. A local Ollama service running on the same node used to launch the runner
2. A remote Ollama-compatible endpoint reachable from that node

### Typical local-node setup

If the cluster policy allows it:

```bash
ollama serve
ollama pull deepseek-r1:8b
```

Then launch the experiments with:

```bash
python build_paper_experiments.py --mode local --rounds 100 --repeat 10 --ollama-base-url http://127.0.0.1:11434
python build_paper_experiments_500clients_k5.py --mode local --rounds 100 --repeat 10 --ollama-base-url http://127.0.0.1:11434
```

### Docker mode note

In Docker mode the server container must still reach the Ollama endpoint on the host or on the network.
The generated compose now includes a `host.docker.internal:host-gateway` mapping so Linux Docker hosts are handled more reliably.

If Ollama runs on the same host, the default Docker URL is:

```bash
http://host.docker.internal:11434
```

You can still override it explicitly:

```bash
python build_paper_experiments.py --mode docker --rounds 100 --repeat 10 --ollama-base-url http://host.docker.internal:11434
```

### Preflight check

You can probe the endpoint before launching long campaigns:

```bash
python scripts/verify_experiment_env.py --mode local --ollama-url http://127.0.0.1:11434
python scripts/verify_experiment_env.py --mode docker --ollama-url http://host.docker.internal:11434
```

## 5. Copy back only the results you need

Typical folders to bring back to your PC:

- `paper_results_local_100r/`
- `paper_results_docker_100r/`
- `paper_results_local_100r_k5/`
- `paper_results_docker_100r_k5/`
- `Experiments_100r/`
- `Experiments_100r_docker/`
- `Experiments_100r_k5/`
- `Experiments_100r_k5_docker/`

## Notes

- The Docker modes require a working `docker compose` runtime on the execution node.
- On many HPC systems Docker is restricted; in that case only the `--mode local` campaigns are expected to work directly.
- The agentic approaches also need an Ollama-compatible endpoint reachable from the node that launches the experiment runner.
