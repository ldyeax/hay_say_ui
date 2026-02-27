# Stack Research

**Domain:** Multi-architecture AI voice generation/conversion orchestration platform
**Researched:** 2026-02-27
**Confidence:** MEDIUM-HIGH

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Python | 3.12-3.13 | Runtime for orchestration services and model adapters | PyTorch stable explicitly supports Python 3.10+ (recommended 3.10-3.14), and 3.12/3.13 is the best balance of modern runtime + broad ML wheel compatibility in 2026. |
| FastAPI + Uvicorn | FastAPI 0.133.x, Uvicorn 0.41.x | API/control plane replacing ad-hoc Flask-per-backend patterns over time | FastAPI is now the standard Python API framework for ML services: typed contracts, OpenAPI generation, async I/O, and better long-term maintainability than bespoke Flask endpoints. |
| PyTorch | 2.7.x | Inference runtime across heterogeneous voice model backends | Your platform is model-orchestration-first; PyTorch remains the common denominator for voice architectures and supports CUDA, ROCm, and CPU from one code path. |
| Celery + Redis | Celery 5.6.x, Redis 7.x | Durable async job queue for generation/conversion tasks | Long-running GPU jobs need explicit queueing, retries, routing, and concurrency controls; Celery+Redis is still the most common Python stack for this operational profile. |
| PostgreSQL | 18.x | Persistent system-of-record metadata (jobs, runs, artifacts, users, quotas, audit) | Current PostgreSQL major is 18; using Postgres for control-plane data prevents metadata sprawl in files/volumes and gives transactional guarantees for orchestration state. |
| Object Storage (S3 API) | S3-compatible (AWS S3/MinIO), boto3 1.42.x | Durable storage for models, uploaded audio, intermediate artifacts, and outputs | Shared Docker volumes are fine locally but do not scale operationally; S3-style object storage is the standard boundary between compute services. |
| Docker Compose + BuildKit | Compose v2, Docker Engine 23+ | Local/dev and single-host deployment with selective architecture enablement | Compose profiles directly solve your "install only selected architectures" requirement; BuildKit is default and materially improves image build performance/caching. |
| NVIDIA Container Toolkit | 1.18.x | GPU pass-through for containerized inference services | Official NVIDIA path for Docker/containerd GPU enablement; required if Dockerized architecture containers should use CUDA reliably. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Pydantic | 2.12.x | Strong request/response and config validation | Use for all API schemas and backend adapter contracts to prevent runtime mismatch bugs across architectures. |
| SQLAlchemy + Alembic | SQLAlchemy 2.0.x, Alembic 1.18.x | DB access and schema migration | Use once you centralize job/model metadata in Postgres; avoid hand-written migration scripts. |
| redis-py | 7.2.x | Redis access for queue and cache controls | Use for queue introspection, rate limiting, and warm-model lease tracking. |
| Prometheus client | 0.24.x | Metrics export per service/container | Use in every architecture adapter and orchestrator worker for latency, queue depth, GPU utilization proxy metrics, and failure rate. |
| OpenTelemetry SDK | 1.39.x | Traces/metrics/log correlation across services | Use to debug cross-container generate flows (UI/API -> queue -> backend adapter -> artifact store). |
| Flower | 2.0.x | Celery worker/queue observability | Use in ops/staging/prod to monitor task retries, stuck jobs, and worker health without custom dashboards first. |
| orjson | 3.11.x | Fast JSON serialization | Use for high-frequency API endpoints and queue payload serialization performance. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| uv | Python env + dependency management | Use one lockfile-driven workflow for all services; faster and more reproducible than ad-hoc pip inside long-lived containers. |
| Ruff + MyPy | Linting + type enforcement | Enforce strict contracts between orchestrator and model adapters; catches many integration regressions early. |
| Pytest + pytest-asyncio | Unit/integration tests | Add contract tests for each architecture adapter and queue workflow tests for retries/timeouts. |
| pre-commit | Consistent local quality gates | Run lint, type checks, and lightweight tests before CI to reduce broken container builds. |
| GitHub Actions | CI/CD automation | Standard for build/test/security scanning and image publishing; supports OIDC for cloud auth without long-lived secrets. |
| Prometheus + Grafana | Runtime observability | Default OSS stack for service metrics, queue health, and SLO dashboards for generation latency/failure rate. |

## Installation

```bash
# Core runtime
uv add "fastapi>=0.133,<0.134" "uvicorn>=0.41,<0.42" "pydantic>=2.12,<2.13"
uv add "celery>=5.6,<5.7" "redis>=7.2,<7.3"
uv add "sqlalchemy>=2.0.47,<2.1" "alembic>=1.18,<1.19"
uv add "boto3>=1.42,<1.43" "orjson>=3.11,<3.12"
uv add "prometheus-client>=0.24,<0.25" "opentelemetry-sdk>=1.39,<1.40"

# Worker/ops support
uv add "flower>=2.0,<2.1"

# Dev dependencies
uv add --dev "ruff>=0.15,<0.16" "mypy>=1.19,<1.20" "pytest>=9.0,<9.1" "pytest-asyncio>=1.3,<1.4" "pre-commit>=4.5,<4.6"
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| Celery + Redis | Temporal | Use Temporal only if you need long-running, multi-day, human-in-the-loop workflows with strong workflow history semantics. For Hay Say-style generate/convert jobs, Celery remains simpler and lower overhead. |
| Docker Compose v2 profiles | Kubernetes (v1.35) + Helm | Use Kubernetes once you need multi-host autoscaling, hard multi-tenancy boundaries, and managed rolling deploys. Keep Compose for local and single-node self-hosting. |
| PostgreSQL 18 | MongoDB | Use MongoDB only for truly document-native, schemaless metadata. Job orchestration state transitions are relational and transactional, so Postgres is the safer default. |
| FastAPI control plane | Keep Dash as the only backend surface | Keep Dash-only if product scope stays single-node hobby usage. For long-term platform evolution, split API/control plane from UI. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Python 3.10 baseline for new services | You lose runtime improvements and will hit newer ecosystem deprecations sooner; current PyTorch supports newer Python versions. | Python 3.12-3.13 |
| Monolithic UI process also acting as orchestration control plane | Hard to scale, test, and secure; failures in UI lifecycle can impact job orchestration reliability. | Separate FastAPI control plane + worker services + optional independent web UI |
| Large preloaded model-pack Docker images as primary distribution path | Causes huge pulls/storage waste and slows updates; already identified pain in Hay Say docs. | On-demand model sync to object storage + selective architecture/image pull with Compose profiles |
| Host wildcard defaults in server config | Increases risk of host-header abuse and misrouting in shared-host deployments. | Explicit allowed hosts, TLS termination, and per-environment config |

## Stack Patterns by Variant

**If local/self-hosted single machine (primary Hay Say audience):**
- Use Docker Compose v2 + profiles + NVIDIA toolkit
- Because it keeps setup approachable while still enabling selective architecture install and GPU-aware execution

**If hosted multi-tenant service:**
- Use Kubernetes v1.35 + managed Postgres + object storage + OIDC-based CI deploys
- Because you need horizontal scaling, tenant isolation, and safer secret/auth posture

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| PyTorch 2.7.x | Python 3.10-3.14 | Official install docs explicitly state Python 3.10+ and recommend 3.10-3.14. |
| Celery 5.6.x | Python 3.8-3.13 | Supports modern Python, but Celery docs note Windows is not officially supported for issue handling; design production workers for Linux containers. |
| SQLAlchemy 2.0.x | Alembic 1.18.x | Current stable migration path for typed ORM + schema migration in Python services. |
| Docker Engine 23+ | BuildKit default | BuildKit is default from Engine 23.0+ and should be assumed baseline for build pipeline design. |
| Docker Compose v2 profiles | Compose Deploy GPU reservations | Profiles + GPU device reservations allow architecture-selective + GPU-aware service startup from one compose file. |

## Sources

- Hay Say project docs (`.planning/PROJECT.md`, `Readme.md`, `running as server/Readme.md`) - current architecture, pain points, and deployment constraints. (HIGH)
- https://pytorch.org/get-started/locally/ - PyTorch stable version, Python support, CUDA/ROCm install guidance. (HIGH)
- https://docs.celeryq.dev/en/stable/getting-started/introduction.html - Celery 5.6 stable, Python compatibility, broker model. (HIGH)
- https://docs.docker.com/compose/profiles/ - selective service activation with Compose profiles. (HIGH)
- https://docs.docker.com/compose/how-tos/gpu-support/ - Compose GPU reservation model. (HIGH)
- https://docs.docker.com/build/buildkit/ - BuildKit defaults and capabilities; Engine 23+ default. (HIGH)
- https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html - GPU container runtime standard path and toolkit version line. (HIGH)
- https://www.postgresql.org/docs/current/release.html - current PostgreSQL major release family (18). (HIGH)
- https://docs.sqlalchemy.org/en/20/ - SQLAlchemy 2.0 current line and release date. (HIGH)
- https://kubernetes.io/docs/home/ - current stable docs track (v1.35). (HIGH)
- https://opentelemetry.io/docs/ - OTel standards and collector/instrumentation model. (MEDIUM)
- https://prometheus.io/docs/introduction/overview/ - Prometheus architecture for metrics collection. (HIGH)
- PyPI JSON endpoints for version pinning (fastapi, uvicorn, pydantic, celery, sqlalchemy, alembic, redis, prometheus-client, flower, etc.). (MEDIUM)

---
*Stack research for: Hay Say voice orchestration platform*
*Researched: 2026-02-27*
