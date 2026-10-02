# Running the AP4Fed campaign quickly

The campaign contains 600 runs: 10 repetition waves with 60 configurations per
wave. Every launcher below is restartable. A completed run is skipped; the run
that was active at interruption is restarted from FL round 1.

Do not combine measurements produced with different CPU profiles in the same
paper analysis. Choose one of the following solutions for the final campaign.

## Solutions

### 1. Docker VM, 32 cores

Use this when CINECA gives you a virtual machine with Docker Compose. It is the
simplest solution, but one VM executes the 600 runs sequentially.

- server: 2 cores;
- low-spec client: 2 cores;
- high-spec client: 3 cores;
- largest configuration: 30/32 cores.

```bash
./leonardo/run_docker_vm.sh --dry-run
./leonardo/run_docker_vm.sh
```

The first real invocation builds the images and downloads the datasets once.
Containers receive non-overlapping CPU sets. Run the same command again after
an interruption.

### 2. One Leonardo DCGP node, 112 cores

Use this when you want one Slurm job and one result directory. Runs remain
sequential, but each run receives substantially more CPU resources.

- server: 2 cores;
- low-spec client: 7 cores;
- high-spec client: 12 cores;
- largest configuration: 112/112 cores.

```bash
./leonardo/submit_campaign.sh <PROJECT_ACCOUNT>
```

Results are written to `$WORK/AP4Fed-pattern-stress-results`. The production
queue has a finite walltime, so submit the same command again whenever the job
stops before all 600 runs are complete.

### 3. Parallel repetition waves on Leonardo — fastest

This is the recommended solution when the project budget permits multiple DCGP
nodes. Each repetition wave runs on its own 112-core node and writes to an
independent directory, avoiding file collisions.

```bash
# Run all ten waves concurrently
./leonardo/submit_wave_array.sh <PROJECT_ACCOUNT> 10

# Or limit concurrency, for example to three nodes
./leonardo/submit_wave_array.sh <PROJECT_ACCOUNT> 3
```

Each array task executes 60 runs for exactly one seed. Re-submit the same
command after interruptions: completed runs and completed waves are skipped.
The results are stored under
`$WORK/AP4Fed-pattern-stress-parallel/wave_1` through `wave_10`.

When the array has finished, merge and verify it:

```bash
.venv-leonardo/bin/python leonardo/merge_wave_results.py \
  "$WORK/AP4Fed-pattern-stress-parallel"
```

The command succeeds only when all 600 run identifiers are present. It creates:

- `adept_experiments_all_waves.csv`;
- `campaign_verification.json` with the count for every wave.

## Initial setup for Slurm solutions

### A. Build the SIF image on a Linux machine

The Linux machine needs Docker and Singularity or Apptainer:

```bash
./leonardo/build_sif_from_docker.sh
```

The resulting `leonardo/ap4fed.sif` is intentionally excluded from Git because
it is large. Transfer the repository and the SIF image to `$WORK` or `$FAST` on
Leonardo.

### B. Prepare the runner on the Leonardo login node

```bash
./leonardo/prepare_runner.sh
```

PyTorch and Flower stay inside the SIF; the host environment contains only the
small campaign-management dependencies.

### C. Validate without starting experiments

```bash
.venv-leonardo/bin/python run_adept_campaign.py \
  --container-runtime singularity \
  --host-cpus 112 --server-cpus 2 \
  --low-spec-cpus 7 --high-spec-cpus 12 \
  --dry-run
```

The output must report 600 runs, 60 for each seed from 1 to 10.

## Monitoring and recovery

```bash
squeue -u "$USER"
tail -f leonardo/logs/slurm-<JOB_ID>.out
```

Five minutes before a Slurm time limit, the batch shell forwards a termination
signal to the campaign. The current run is marked `interrupted`, containers are
stopped, completed outputs remain valid, and the next submission resumes from
the first incomplete run. A hard node failure is also recoverable: a run is
considered complete only when its `ok` index entry, archived configuration, and
summary CSV are all present.

To stop intentionally, use `scancel <JOB_ID>`. To resume later, submit the same
serial or array command again.

## Files

- `run_docker_vm.sh`: native 32-core Docker VM launcher;
- `ap4fed_campaign.sbatch`: serial 112-core Slurm campaign;
- `ap4fed_wave_array.sbatch`: isolated 112-core repetition wave;
- `submit_campaign.sh`: serial submission wrapper;
- `submit_wave_array.sh`: parallel array submission wrapper;
- `merge_wave_results.py`: final merge and completeness check.
