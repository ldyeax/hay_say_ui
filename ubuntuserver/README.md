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

Run these as the service user (normally `luna`):

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

Voice-conversion tabs accept multi-pitch output when **Generate multiple
pitches** is enabled. Enter comma-separated semitones such as `-12,0,12` or an
inclusive range such as `-12:12:2`. Preprocessing is shared across the batch and
each pitch is cached as a distinct output.

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
