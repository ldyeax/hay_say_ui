# Architecture Research

**Domain:** Docker-orchestrated AI voice generation and conversion platform
**Researched:** 2026-02-27
**Confidence:** MEDIUM-HIGH

## Standard Architecture

### System Overview

```text
┌────────────────────────────────────────────────────────────────────────────┐
│                          Interface and Access Layer                        │
├────────────────────────────────────────────────────────────────────────────┤
│  Browser UI  ->  API/Orchestrator  ->  Job API  ->  Reverse Proxy (hosted)│
└───────────────────────────────┬────────────────────────────────────────────┘
                                │ enqueue + status
┌───────────────────────────────▼────────────────────────────────────────────┐
│                          Control Plane (CPU-first)                         │
├────────────────────────────────────────────────────────────────────────────┤
│  Validation + policy  |  Queue broker (Redis)  |  Worker scheduler/routing │
│  Model registry/meta  |  Job state + retries   |  Capacity + warm pool mgr  │
└───────────────────────────────┬────────────────────────────────────────────┘
                                │ dispatch by architecture
┌───────────────────────────────▼────────────────────────────────────────────┐
│                          Data Plane (isolated runtimes)                    │
├────────────────────────────────────────────────────────────────────────────┤
│  Pre/Post pipeline svc | Arch runtime A | Arch runtime B | Arch runtime N  │
│  (shared transforms)   | (GPU/CPU)      | (GPU/CPU)      | (GPU/CPU)        │
└───────────────────────────────┬────────────────────────────────────────────┘
                                │ read/write artifacts
┌───────────────────────────────▼────────────────────────────────────────────┐
│                             Storage and Observability                      │
├────────────────────────────────────────────────────────────────────────────┤
│  Shared volumes: models, audio_cache, outputs, temp                        │
│  Metadata DB (recommended) + Redis broker/result + OTel collector          │
└────────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| API/Orchestrator (`hay_say_ui`) | Request validation, orchestration, job creation, API responses | Python web app + Gunicorn, thin sync endpoints + async dispatch |
| Queue + workers | Decouple request latency from generation time, retries, scheduling | Celery workers on Redis with explicit queues by workload class |
| Shared pre/post processor | Architecture-agnostic transforms and reusable audio pipeline | Dedicated service/module used before and after architecture runtime |
| Architecture runtime services | Model-specific inference and conversion, isolated dependencies | One container per architecture family (`*_server`) |
| Model/asset storage | Character models, staged artifacts, output files | Docker volumes (`models`, `audio_cache`) + optional object store in hosted mode |
| Metadata/state store | Durable job state, auditability, resumability | Recommended: Postgres for job metadata; keep Redis for broker/result |
| Edge/proxy | TLS termination, rate limits, host-header control | NGINX (already used in server deployment) |
| Observability pipeline | Traces/metrics/logs, bottleneck visibility | OpenTelemetry Collector (agent or gateway pattern) |

## Recommended Project Structure

```text
services/
├── gateway/                     # nginx/caddy configs, TLS, rate limits
├── orchestrator/                # HTTP API, validation, job lifecycle
│   ├── api/                     # request handlers
│   ├── domain/                  # job/model/architecture domain logic
│   ├── adapters/                # celery, redis, filesystem, db
│   └── contracts/               # architecture plugin contracts
├── worker/                      # celery app + routing + retry policy
├── runtimes/                    # one folder per architecture runtime
│   ├── so_vits_svc_4/
│   ├── rvc/
│   └── styletts_2/
├── pipeline/                    # shared preprocessing/postprocessing code
├── infra/
│   ├── compose/                 # local/server compose variants
│   ├── observability/           # otel collector config, dashboards
│   └── policies/                # resource limits, queue mappings
└── docs/
    ├── architecture/            # C4-ish diagrams and boundaries
    └── extension-guides/        # add-architecture, add-model-pack guides
```

### Structure Rationale

- **`orchestrator/` as control plane:** keeps workflow logic out of runtime containers and avoids architecture-specific coupling.
- **`runtimes/` as data plane:** each architecture owns its dependency stack and failure domain, preserving isolation.
- **`pipeline/` shared layer:** enforces the project goal that pre/post improvements benefit all architectures.
- **`infra/compose/` split:** local and server overlays can share base services while changing only exposure, scaling, and security defaults.

## Architectural Patterns

### Pattern 1: Control-plane / Data-plane split

**What:** API + queue + state management runs separately from inference runtimes.
**When to use:** Multi-architecture systems with heterogeneous dependencies and variable latency.
**Trade-offs:** Better fault isolation and scaling; slightly more moving parts.

**Example:**
```python
# orchestrator submits normalized jobs; runtime workers only execute
job_id = jobs.create(request_payload)
celery_app.send_task(
    "generate.voice",
    kwargs={"job_id": job_id, "architecture": "SoVitsSvc4"},
    queue="generate.gpu.so_vits_svc_4",
)
```

### Pattern 2: Queue partitioning by workload class

**What:** Separate queues for download, CPU generation, GPU generation, and optionally per-architecture GPU lanes.
**When to use:** Mixed short/long tasks and mixed resource classes.
**Trade-offs:** Higher operational clarity and fairness; more queue config to maintain.

**Example:**
```python
app.conf.task_routes = {
  "tasks.download_*": {"queue": "download.io"},
  "tasks.generate_cpu_*": {"queue": "generate.cpu"},
  "tasks.generate_gpu_so_vits_svc_4": {"queue": "generate.gpu.so_vits_svc_4"},
}
```

### Pattern 3: Profile-gated optional services

**What:** Use Compose profiles to include only needed architecture services.
**When to use:** Large images/model footprints and user-selective installs.
**Trade-offs:** Smaller installs and faster startup; matrix testing becomes broader.

**Example:**
```yaml
services:
  so_vits_svc_4_server:
    image: hydrusbeta/hay_say:so_vits_svc_4_server
    profiles: [so_vits]

  rvc_server:
    image: hydrusbeta/hay_say:rvc_server
    profiles: [rvc]
```

### Pattern 4: Two-stage observability collection

**What:** Run local/host collector as agent, optional central collector as gateway.
**When to use:** Hosted deployments with multiple nodes/regions.
**Trade-offs:** Better telemetry hygiene and routing; extra infra to operate.

## Data Flow

### Request Flow (Generation)

```text
User clicks Generate
    -> API validates inputs + resolves architecture/model
    -> Job record created (queued)
    -> Task enqueued to class-specific queue
    -> Worker claims task based on capacity/policy
    -> Preprocess stage reads raw audio, writes preprocessed artifact
    -> Runtime service invoked (internal network call)
    -> Runtime reads artifact + model, writes generated audio
    -> Postprocess stage writes final artifact
    -> Job state updated to completed/failed
    -> UI polls/subscribes and renders playback/download
```

### State Management

```text
Redis: broker/result transport, short-lived coordination
Metadata store (recommended Postgres): durable job state, attempts, timings
Volumes/object storage: binary audio/model artifacts
```

### Key Data Flows

1. **Inference flow:** `api -> queue -> worker -> runtime -> artifact store -> api response`.
2. **Model lifecycle flow:** `manage models ui -> downloader tasks -> models volume -> runtime discovery`.
3. **Warm pool flow:** `scheduler policy -> keepalive/warm actions -> runtime readiness state -> lower p95 latency`.

## Suggested Build Order (Roadmap Dependencies)

1. **Establish architecture contract and boundaries**
   - Define runtime contract: health, capability manifest, generate endpoint, error schema.
   - Dependency: none; this unblocks every later phase.

2. **Harden control plane (job model + routing policy)**
   - Add explicit job states, retries, timeout policy, queue partitioning.
   - Dependency: contract from step 1.

3. **Isolate shared pre/post pipeline into first-class service/module**
   - Keep architecture services thin; avoid duplicate transforms.
   - Dependency: step 2 (job lifecycle hooks).

4. **Selective install and runtime composition with Compose profiles**
   - Make architecture services opt-in; keep core services always-on.
   - Dependency: step 1 contracts + step 2 queue mapping.

5. **Warm-start and resource-aware scheduling**
   - Add warm pool manager and GPU/CPU admission controls.
   - Dependency: reliable job states and queue partitioning (step 2).

6. **Hosted-mode edge/security hardening**
   - Reverse proxy defaults, strict hostnames, TLS, rate limits.
   - Dependency: stable service map from steps 2-4.

7. **Observability maturity (OTel agent -> gateway as needed)**
   - Instrument orchestrator/workers/runtimes; add bottleneck dashboards.
   - Dependency: stable flow boundaries from all prior steps.

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| 0-1k users (single host / small team) | Docker Compose, Redis broker, per-architecture containers, simple volume storage, basic queue partitioning |
| 1k-100k users (multi-host/self-hosted service) | Separate control plane from runtimes, external Postgres for job metadata, object storage for artifacts, autoscaled worker pools |
| 100k+ users (managed SaaS) | Regional runtime pools, admission control, model shard strategy, centralized OTel gateway, stronger multi-tenant isolation |

### Scaling Priorities

1. **First bottleneck:** cold starts and model load latency; fix with warm pools + architecture-specific queue lanes.
2. **Second bottleneck:** queue fairness and retries under mixed workloads; fix with queue partitioning and explicit retry/idempotency policy.

## Anti-Patterns

### Anti-Pattern 1: Monolithic "UI does everything" process

**What people do:** run web server, all workers, and orchestration policy in one tightly coupled command.
**Why it's wrong:** poor failure isolation, difficult scaling, hard to tune CPU/GPU paths independently.
**Do this instead:** keep API/orchestrator separate from worker pools and runtime services even if all still run under Compose.

### Anti-Pattern 2: Architecture-specific logic leaking into shared pipeline

**What people do:** add per-architecture conditionals inside shared preprocess/postprocess code.
**Why it's wrong:** regressions across architectures and blocked extensibility.
**Do this instead:** put architecture quirks behind runtime adapters and keep shared pipeline contract-driven.

### Anti-Pattern 3: Redis-only durability assumptions

**What people do:** treat broker state as system-of-record for long-running job metadata.
**Why it's wrong:** visibility timeout and broker semantics can cause duplicate execution and weak auditability.
**Do this instead:** use Redis for transport, plus durable metadata storage for lifecycle truth.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| Model sources (HF/Mega/Drive) | Async downloader tasks via dedicated queue | Keep network-heavy tasks isolated from generation workers |
| Redis | Broker/result backend for Celery | Configure visibility timeout/retry policy intentionally for long tasks |
| OTel backend (Prometheus/Tempo/etc.) | OTel collector exports | Start with local agent pattern; add gateway when multi-node |
| Optional cloud object storage | Store final artifacts and large caches | Useful for hosted mode and multi-node scaling |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| UI/API ↔ Queue workers | Task enqueue + status reads | Keep request path fast and non-blocking |
| Workers ↔ Runtime services | Internal HTTP/gRPC + shared artifact references | Runtime services should remain stateless except model cache |
| Runtime services ↔ Model storage | Read-only model access + controlled cache writes | Prevent runtime-specific model format drift from leaking outward |

## Sources

- Docker Compose profiles: https://docs.docker.com/compose/how-tos/profiles/ (official, retrieved 2026-02-27)
- Docker Compose startup order and `depends_on` conditions: https://docs.docker.com/compose/how-tos/startup-order/ (official, retrieved 2026-02-27)
- Compose service reference (`depends_on`, `healthcheck`, `gpus`, `models`): https://docs.docker.com/reference/compose-file/services/ (official, retrieved 2026-02-27)
- Docker Compose GPU support: https://docs.docker.com/compose/how-tos/gpu-support/ (official, retrieved 2026-02-27)
- NVIDIA Container Toolkit install/config: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html (official, retrieved 2026-02-27)
- Celery routing: https://docs.celeryq.dev/en/stable/userguide/routing.html (official, retrieved 2026-02-27)
- Celery optimization/prefetch/memory: https://docs.celeryq.dev/en/stable/userguide/optimizing.html (official, retrieved 2026-02-27)
- Celery Redis caveats (visibility timeout, redelivery): https://docs.celeryq.dev/en/stable/getting-started/backends-and-brokers/redis.html (official, retrieved 2026-02-27)
- OpenTelemetry Collector agent pattern: https://opentelemetry.io/docs/collector/deploy/agent/ (official, modified 2026-01-29)
- OpenTelemetry Collector gateway pattern: https://opentelemetry.io/docs/collector/deploy/gateway/ (official, modified 2026-01-29)
- Project context: `/home/luna/hay_say/hay_say_ui/.planning/PROJECT.md`, `/home/luna/hay_say/hay_say_ui/Readme.md`, `/home/luna/hay_say/hay_say_ui/running as server/Readme.md`, `/home/luna/hay_say/hay_say_ui/docker-compose.yaml`, `/home/luna/hay_say/hay_say_ui/running as server/docker-compose.yaml`

---
*Architecture research for: Hay Say architecture evolution (subsequent milestone)*
*Researched: 2026-02-27*
