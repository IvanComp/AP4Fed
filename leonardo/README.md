# Running the AP4Fed campaign quickly

The campaign contains 600 runs: 10 repetition waves with 60 configurations per
wave. Every launcher below is restartable. A completed run is skipped; the run
that was active at interruption is restarted from FL round 1.

## Start here: RCM on Leonardo

Use the two RCM session types as follows:

- **SSH session:** this is the control terminal. Use it to update the repository,
  prepare files, submit jobs, and inspect results.
- **Slurm session:** this is an interactive compute allocation. It is not needed
  for this batch campaign. Close it so it does not reserve resources while idle.

Do not run the campaign directly in the SSH session. Submit it with `sbatch` via
the provided launcher; the jobs then run on DCGP compute nodes and continue even
if RCM is closed.

Open a terminal in the **SSH graphical session**, then run:

```bash
cd /path/to/AP4Fed
git pull
saldo -b --dcgp
ls -lh leonardo/ap4fed.sif
```

Use the project-account name shown by `saldo` as `<PROJECT_ACCOUNT>` below. The
SIF file must exist. It is excluded from Git because it is large. If `ls` says
that it is missing, download the image built by GitHub:

```bash
./leonardo/pull_sif_from_ghcr.sh
```

The GitHub package must be public. On its first creation, open the package page,
choose **Package settings**, then **Change visibility** and select **Public**.
This is required only once.

Prepare and validate everything without starting an experiment:

```bash
chmod +x leonardo/*.sh
./leonardo/prepare_runner.sh
./leonardo/check_setup.sh
```

The last line must be `PRE-FLIGHT PASSED`. Then submit one real wave as a pilot:

```bash
mkdir -p leonardo/logs
sbatch --account=<PROJECT_ACCOUNT> --array=1-1 leonardo/ap4fed_wave_array.sbatch
squeue -u "$USER"
```

`sbatch` prints a job ID. Inspect that job with:

```bash
tail -f leonardo/logs/wave-<JOB_ID>_1.out
```

After wave 1 finishes successfully, submit all waves with maximum concurrency:

```bash
./leonardo/submit_wave_array.sh <PROJECT_ACCOUNT> 10
```

Wave 1 will be detected as complete and skipped. The other nine waves may use
nine DCGP nodes concurrently, subject to the account budget and scheduler. Do
not submit this command while the pilot is still running, or wave 1 could run
twice at the same time.

If ten jobs cannot be scheduled together, use `3` instead of `10`. This changes
only the number of simultaneous nodes, not the 600-run campaign or its results.
After an interruption, submit the same command again; completed runs are skipped.

When `squeue -u "$USER"` no longer shows the campaign, merge and verify it:

```bash
.venv-leonardo/bin/python leonardo/merge_wave_results.py \
  "$WORK/AP4Fed-pattern-stress-parallel"
cat "$WORK/AP4Fed-pattern-stress-parallel/campaign_verification.json"
```

The verification must report 600 total runs and 60 runs for every wave.

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

The job requests one exclusive DCGP node and disables hardware threads. Smaller
configurations intentionally leave some cores unused: changing the cores per
client according to client count would alter the experimental treatment.

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

### A. Obtain the SIF image

The recommended method does not require Docker on the user's computer. The
GitHub Actions workflow `.github/workflows/build-leonardo-image.yml` publishes
the Linux/AMD64 image whenever it or the Docker sources change. On Leonardo:

```bash
./leonardo/pull_sif_from_ghcr.sh
```

Alternatively, build the SIF image on a Linux machine. The Linux machine needs
Docker and Singularity or Apptainer:

```bash
./leonardo/build_sif_from_docker.sh
```

The resulting `leonardo/ap4fed.sif` is intentionally excluded from Git because
it is large. Transfer the repository and the SIF image to `$WORK` or `$FAST` on
Leonardo.

### B. Prepare the runner on the Leonardo login node

```bash
module load python/3.11.7
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
- `check_setup.sh`: Leonardo pre-flight check; it starts no experiments;
- `pull_sif_from_ghcr.sh`: downloads and converts the GitHub image to SIF;
- `ap4fed_campaign.sbatch`: serial 112-core Slurm campaign;
- `ap4fed_wave_array.sbatch`: isolated 112-core repetition wave;
- `submit_campaign.sh`: serial submission wrapper;
- `submit_wave_array.sh`: parallel array submission wrapper;
- `merge_wave_results.py`: final merge and completeness check.
