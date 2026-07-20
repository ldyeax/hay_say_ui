# Native Ubuntu installation

This directory installs Hay Say as native Ubuntu processes. Docker is used only
as an import format for the seven published model runtime images; no Hay Say
container is used after extraction.

## Fresh install

From any working directory:

```bash
sudo /path/to/hay_say_ui/ubuntuserver/install.sh
```

Defaults are user `luna`, install root `/home/luna/hay_say`, and data root
`/mnt/sanic/hay_say` when `/mnt/sanic` is mounted. Otherwise data is placed in
`/home/luna/hay_say/data`. The first run installs Ubuntu dependencies, pulls and
verifies all seven pinned images, extracts only declared source/assets, builds
the UI and per-runtime Python environments with `uv`, enables user lingering,
and starts `hay-say.target`.

Custom paths and account:

```bash
sudo ./ubuntuserver/install.sh \
  --user luna \
  --install-root /home/luna/hay_say \
  --data-root /mnt/sanic/hay_say
```

Preview every system, Docker, and filesystem action without changing anything:

```bash
sudo ./ubuntuserver/install.sh --dry-run
```

## Update

The repository worktree containing `install.sh` is the source of truth. Local
uncommitted changes are deployed too. Existing target `.git` metadata and files
excluded by `.gitignore` are preserved.

```bash
sudo ./ubuntuserver/install.sh --skip-apt --skip-images
```

Re-extract one updated runtime and refresh its environment:

```bash
sudo ./ubuntuserver/install.sh --skip-apt --runtime rvc
```

Image tags are checked against `config/image-digests.tsv`. Update that reviewed
digest intentionally when accepting a new published image. Extraction records
the resolved image ID, repository digest, platform, timestamp, manifest digest,
and copied paths under `<data-root>/provenance/`. The extractor never invokes
Docker prune and removes only temporary containers it creates. Local image
removal requires the explicit `--remove-images` option.

## Service control

The convenience scripts work when invoked as either `root` or `luna`:

```bash
./ubuntuserver/start-hay-say.sh
./ubuntuserver/stop-hay-say.sh
```

Alternatively, run systemd commands directly as the service user (normally
`luna`):

```bash
systemctl --user start hay-say.target
systemctl --user stop hay-say.target
systemctl --user restart hay-say.target
systemctl --user status hay-say.target
journalctl --user -u hay-say-ui.service -f
```

The UI listens on port `6573` and requires HTTP Basic authentication by
default. Retrieve the persistent generated credentials with:

```bash
sudo -iu luna cat /home/luna/.config/hay-say/ui-auth
```

Authentication can be disabled explicitly for a trusted network installation:

```bash
sudo ./ubuntuserver/install.sh --skip-apt --skip-images --no-ui-auth
```

The installer persists this choice as `enabled=0` in `~/.config/hay-say/ui-auth`
and exports `HAY_SAY_UI_AUTH_ENABLED=0` to the services. Later updates preserve
the setting. Run the installer with `--ui-auth` to require credentials again.
Because the UI binds to all interfaces, disabling authentication should only be
used when access is restricted by the host firewall or trusted network.

Redis, the runtime manager, three Celery workers,
and Gunicorn are user services. Model runtimes are started on demand through the
loopback runtime manager at `127.0.0.1:6588`; they are not separate boot-time
services.

Open **Model Runtimes** from the UI toolbar to start, stop, or restart a model
backend and inspect its process, CPU, RAM, GPU-memory, uptime, and job state.
`ready-cold` means the backend is accepting requests but has no confirmed model
resident in memory; `warm-idle` means a model is loaded and waiting for work.
SVC3 also exposes explicit warm and unload operations through the manager API.

Voice-conversion tabs accept multi-pitch output when **Pitch variants** is
enabled. Enter comma-separated semitones such as `-12,0,12` or an
inclusive range such as `-12:12:2`. Preprocessing is shared across the batch and
each pitch is cached as a distinct output. Download names include the character,
architecture, source, and pitch; any pitch result also provides one ZIP download
containing the complete batch.

Generation identity and progress are persisted for the browser session. A page
refresh reattaches to a queued or running request and redraws completed outputs.
The generation **Stop** button cancels only that browser's request and preserves
the model service and its warm replicas. Native runtimes cooperatively discard
cancelled queued work; an inference already inside a non-interruptible Torch
kernel reaches its next safe boundary before returning its worker to the pool.
The admin panel's separate runtime **Stop** action is the explicit way to unload
all workers for that model and reclaim its resources immediately.

**Auto** is the default inference device. It uses a configured GPU when the
driver is live, enough VRAM is free, utilization is below the configured limit,
and Hay Say can reserve a GPU slot immediately. Otherwise it uses a CPU slot.
Explicit **GPU** requests wait for GPU capacity instead of silently changing
devices, while explicit **CPU** requests remain on CPU. Eligible Auto pitch
batches reserve CPU and GPU together, split the first request across both, and
leave both device-specific model caches warm for later requests.

SVC3 extracts HuBERT features once per source segment and keeps a host-sized pool
of isolated VITS replicas, so a pitch range can run across multiple CPU lanes.
Mixed batches leave at least half of their initial variants unclaimed; CPU and
GPU lanes then refill from that shared queue whenever each native request ends.
SVC4's forced-slice path likewise uses isolated persistent model processes,
assembles results in source order, and computes crossfades at the output sample
rate. Set **Slice Workers** to zero for the server default, or choose a smaller
limit for memory-constrained hosts. Newly needed replicas warm in groups of four
instead of serially; GPU slice inference remains single-worker.

CPU BF16 autocast is a backend policy, not a generation option. It defaults on
for SVC4 and StyleTTS2 and off for the other models. Even when configured on,
the runtime enables it only on CPUs advertising both `amx_tile` and
`amx_bf16`; GPU inference remains unaffected. Unsupported or
precision-sensitive operators remain FP32 under PyTorch's autocast policy.
Every model service exposes the running process's resolution under
`/runtime` -> `cpu_bf16`, including the environment variable, default,
requested value, whether AMX tile plus BF16 are available on every reported
processor, and the resulting effective value.

The installer derives an aggressive CPU budget from physical cores, SMT threads,
and RAM. It admits one independent CPU request per four physical cores, budgets
1.5x the logical CPU count across their native math pools, and keeps extra
Celery dispatchers ready so orchestration does not become the bottleneck.
Replica counts use 90% of RAM as their hard backstop. Existing values are
preserved on updates except unsafe SVC3 per-replica thread counts, which are
bounded by the configured host-wide SVC3 thread budget.

SVC3 shares HuBERT features but keeps isolated VITS pitch replicas. SVC4 and
SVC5 keep complete model replicas, budgeted at 3 GiB and 4 GiB per CPU lane
respectively. SVC3 targets 2x logical-CPU oversubscription across its pool.
SVC5 targets 1.6x and warms up to eight newly needed replicas concurrently.
RVC, TalkNet, GPT-SoVITS,
and StyleTTS2 also retain reusable per-character replicas. Every idle model
replica is retained for at least 30 minutes by default. This 60-core, 120-thread,
247 GiB host derives the values below. Tune these entries in
`~/.config/hay-say/environment` when benchmarking a different host:

```text
HAY_SAY_GPU_IDS='0'
HAY_SAY_CPU_CONCURRENCY='20'
HAY_SAY_CPU_INFERENCE_SLOTS='15'
HAY_SAY_GPU_INFERENCE_SLOTS='1'
HAY_SAY_MODEL_CPU_THREADS='12'
HAY_SAY_MODEL_CPU_INTEROP_THREADS='1'
HAY_SAY_AUTO_GPU_MIN_FREE_MIB='4096'
HAY_SAY_AUTO_GPU_MAX_UTILIZATION='95'
HAY_SAY_MIXED_PITCH_MIN_VARIANTS='3'
HAY_SAY_AUTO_CPU_PITCH_VARIANTS='4'
HAY_SAY_SVC3_CPU_PITCH_WORKERS='24'
HAY_SAY_SVC3_CPU_THREAD_BUDGET='240'
HAY_SAY_SVC3_CPU_THREADS='10'
HAY_SAY_SVC4_CPU_SLICE_WORKERS='64'
HAY_SAY_SVC4_CPU_THREADS_PER_WORKER='2'
HAY_SAY_SVC5_CPU_WORKERS='24'
HAY_SAY_SVC5_CPU_THREADS_PER_WORKER='8'
HAY_SAY_SVC5_GPU_WORKERS='1'
HAY_SAY_SVC5_STARTUP_CONCURRENCY='8'
HAY_SAY_RVC_CPU_WORKERS='24'
HAY_SAY_RVC_GPU_WORKERS='1'
HAY_SAY_TALKNET_CPU_WORKERS='12'
HAY_SAY_TALKNET_GPU_WORKERS='1'
HAY_SAY_GPT_SOVITS_CPU_WORKERS='12'
HAY_SAY_GPT_SOVITS_GPU_WORKERS='1'
HAY_SAY_STYLETTS_CPU_WORKERS='12'
HAY_SAY_STYLETTS_GPU_WORKERS='1'
HAY_SAY_TALKNET_CPU_BF16_AUTOCAST='0'
HAY_SAY_SVC3_CPU_BF16_AUTOCAST='0'
HAY_SAY_SVC4_CPU_BF16_AUTOCAST='1'
HAY_SAY_SVC5_CPU_BF16_AUTOCAST='0'
HAY_SAY_RVC_CPU_BF16_AUTOCAST='0'
HAY_SAY_STYLETTS_CPU_BF16_AUTOCAST='1'
HAY_SAY_GPT_SOVITS_CPU_BF16_AUTOCAST='0'
HAY_SAY_MODEL_IDLE_TTL_SECONDS='1800'
HAY_SAY_MAX_BATCH_DOWNLOAD_BYTES='268435456'
```

`HAY_SAY_SVC3_CPU_THREADS` is a per-replica maximum. The installer and runtime
cap it at `floor(HAY_SAY_SVC3_CPU_THREAD_BUDGET /
HAY_SAY_SVC3_CPU_PITCH_WORKERS)`, preventing a large pitch batch from multiplying
one host-wide thread count by every VITS replica. `HAY_SAY_AUTO_CPU_PITCH_VARIANTS`
bounds each native CPU claim. Its default of four limits the non-stealable CPU
tail while still sharing HuBERT preparation and running several VITS replicas.

Modern runtime environments include oneDNN builds that can dispatch BF16 matrix
operations to Intel AMX on supported Xeons. Actual gains remain model-dependent,
so the per-model settings should be changed only after comparing them with FP32.
Restart the affected runtime after changing its environment setting.

## Validation

```bash
sudo -iu luna /home/luna/hay_say/hay_say_ui/ubuntuserver/doctor.sh
```

`doctor.sh` validates commands, data/source links, all virtual environments,
image provenance, SVC3's native overlay, user services, Redis, runtime manager,
host aliases, and free space.

Run end-to-end synthesis against the normal installed Fluttershy voice for all
seven runtimes:

```bash
sudo -u luna bash -c 'set -a; source /home/luna/.config/hay-say/environment; set +a; "$HAY_SAY_UI_VENV/bin/python" "$HAY_SAY_UI/ubuntuserver/smoke-models.py"'
```

The smoke runner starts one backend at a time, checks that its generated audio
is finite and non-silent, removes its temporary cache session, and restores the
backend's original running state. Pass `--runtime rvc` (repeatable) to test only
selected runtimes.

Benchmark all installed Fluttershy models with the same input on CPU FP32, CPU
BF16, and the configured GPU:

```bash
sudo -u luna bash -c 'set -a; source /home/luna/.config/hay-say/environment; set +a; "$HAY_SAY_UI_VENV/bin/python" "$HAY_SAY_UI/ubuntuserver/benchmark-models.py" --json-output /mnt/sanic/hay_say/benchmarks/latest.json --csv-output /mnt/sanic/hay_say/benchmarks/latest.csv'
```

The runner records runtime startup separately from warmups and measured request
latency, validates every output, reports real-time factor, and restores runtimes
that were initially stopped. CPU modes query the already-running model service
and validate its reported effective BF16 state instead of trusting the benchmark
process's environment or sending a request precision override. The JSON and CSV
reports retain that runtime policy alongside each result. Run `cpu-fp32` and
`cpu-bf16` separately after setting the corresponding environment value and
restarting that runtime.

## Safe uninstall

Stop and disable processes first:

```bash
sudo -iu luna systemctl --user disable --now hay-say.target
```

The generated files that can be removed without deleting models, cache, image
sources, or repository history are:

```text
~/.config/systemd/user/hay-say*.service
~/.config/systemd/user/hay-say.target
~/.config/hay-say/
~/.local/lib/hay-say/
```

Then run `systemctl --user daemon-reload`. Do not recursively remove the install
or data roots unless their `models`, `audio_cache`, `runtime-sources`, and
`provenance` contents have been reviewed or backed up. The installer never
performs that deletion and never removes unrelated Docker data.

## Compatibility entrypoints

`setup_root.sh`, `setup_luna.sh`, `start_luna.sh`, `venvs.sh`, and `symlinks.sh`
are small compatibility delegates. New automation should call `install.sh` or
`install-user.sh` directly. There are no `screen` sessions or cwd-dependent
launch commands in the native installation.
