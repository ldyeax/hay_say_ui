# Project Research Summary

**Project:** Hay Say
**Domain:** Docker-first multi-architecture voice generation and voice conversion orchestration platform
**Researched:** 2026-02-27
**Confidence:** MEDIUM-HIGH

## Executive Summary

Hay Say should be built as a control-plane-first orchestration product, not a single monolithic UI process. The research consistently points to a split architecture: a typed API/orchestrator handles validation, policy, and job lifecycle; queue workers handle long-running execution; architecture-specific runtime containers stay isolated; and shared pre/post processing remains a first-class cross-architecture layer. This mirrors how mature voice systems handle heterogeneous model stacks while keeping UX consistent.

The recommended approach is to launch with a focused v1: unified TTS + voice conversion UX, selective architecture install, clear preflight validation/errors, and queue/history for non-blocking creator workflows. Under the hood, this means standardizing adapter contracts early, adding queue partitioning by workload class, and persisting durable job metadata (Postgres) instead of relying on Redis semantics alone. Compose profiles and NVIDIA toolkit support remain the right operational baseline for Hay Say's local/self-hosted audience.

The biggest risks are operational, not conceptual: runtime compatibility drift (Python/Torch/CUDA/container matrix), queue backpressure under mixed workloads, unsafe server defaults, and storage lifecycle blowups from models/caches/images. Mitigation should be built into early roadmap phases: a compatibility matrix with startup self-checks, memory-budgeted warm pools and queue admission controls, hardened server profile defaults, and explicit storage retention/prune policies.

## Key Findings

### Recommended Stack

Research in `.planning/research/STACK.md` recommends a modern Python orchestration stack with explicit async job control and durable metadata boundaries. The stack is opinionated toward reliability on heterogeneous hardware and smoother evolution from local Compose deployments to larger hosted setups.

**Core technologies:**
- `Python 3.12-3.13`: orchestration runtime sweet spot between modern runtime gains and broad ML wheel compatibility
- `FastAPI 0.133.x + Uvicorn 0.41.x`: typed API/control plane with stronger maintainability than ad-hoc Flask surfaces
- `PyTorch 2.7.x`: shared inference substrate across CUDA/ROCm/CPU paths
- `Celery 5.6.x + Redis 7.x`: durable async queueing, retry, and routing for long GPU/CPU jobs
- `PostgreSQL 18.x`: system-of-record for job lifecycle, audit, and quotas (Redis remains transport/result)
- `S3-compatible object storage + boto3 1.42.x`: scalable boundary for models/artifacts beyond shared local volumes
- `Docker Compose v2 + BuildKit + NVIDIA Container Toolkit 1.18.x`: selective architecture enablement with GPU pass-through

Critical version constraints to preserve: Python must stay in a PyTorch-supported range, workers should remain Linux-first, and Compose/GPU support should be treated as baseline deployment assumptions.

### Expected Features

Research in `.planning/research/FEATURES.md` confirms that Hay Say's core market expectation is one consistent surface for both generation and conversion, with strong local/self-host ergonomics. Launch scope should stay tight and reliability-focused.

**Must have (table stakes):**
- Unified TTS + voice conversion workflow with architecture-aware options
- Selective architecture/model install and management
- Clear generate preflight validation and actionable field-level errors
- Basic controllability (rate/pitch/style where supported)
- Batch queue visibility plus output history

**Should have (competitive):**
- Warm-start pool with memory-budget scheduler
- Stage-aware compare/listen flow across pipeline stages
- Hardened self-host mode with secure defaults

**Defer (v2+):**
- Cross-architecture preset portability
- Real-time live conversion mode
- Advanced consent/provenance automation

### Architecture Approach

Research in `.planning/research/ARCHITECTURE.md` strongly supports a control-plane/data-plane split with explicit runtime contracts, queue partitioning, and profile-gated runtime services. This pattern aligns with Hay Say's existing multi-container direction while reducing coupling and making future architecture additions safer.

**Major components:**
1. API/orchestrator control plane - validates requests, enforces policy, creates jobs, and coordinates flow
2. Queue/worker layer - executes async workloads with retries, queue lanes, and resource-aware scheduling
3. Runtime service fleet - isolated per-architecture containers handling model-specific inference/conversion
4. Shared pre/post pipeline - architecture-agnostic transforms reused across all runtimes
5. Storage/state/observability - volumes/object storage for artifacts, Postgres for durable metadata, Redis transport, OTel metrics/traces

### Critical Pitfalls

Top risks from `.planning/research/PITFALLS.md` should directly drive phase gates and definition-of-done criteria.

1. **Runtime compatibility drift** - enforce pinned compatibility matrix + startup health/self-checks + matrix CI smoke tests
2. **Queue/backpressure collapse** - partition queues by workload class, tune prefetch/concurrency, and add memory-aware admission controls
3. **Self-host security misconfiguration** - ship hardened server profile defaults (allowed hosts, TLS-first, headers, rate limits)
4. **Storage lifecycle failures** - implement explicit budgets, retention/pruning automation, and selective install/uninstall with size previews
5. **Adapter volatility from upstream ecosystems** - define stable plugin contracts, conformance tests, and lifecycle/EOL policy for architecture adapters

## Implications for Roadmap

Based on combined research, suggested phase structure:

### Phase 1: Platform Baseline and Safe Defaults
**Rationale:** Every later milestone depends on stable contracts and predictable deployment posture.
**Delivers:** Runtime compatibility matrix, architecture adapter contract, startup self-checks, hardened server profile, and generate preflight validation UX baseline.
**Addresses:** Unified workflow readiness, validation/error clarity, secure self-host baseline.
**Avoids:** Compatibility drift, wildcard-host exposure, and opaque UX failure loops.

### Phase 2: Job Orchestration, Queueing, and Storage Lifecycle
**Rationale:** Creator value requires reliable non-blocking jobs before advanced quality/perf features.
**Delivers:** Explicit job states in durable metadata, queue partitioning, retry/timeouts, history surface, storage quotas/retention/pruning, selective install/uninstall flows.
**Uses:** Celery/Redis + Postgres + Compose profiles + object-storage boundary pattern.
**Implements:** Control-plane job model and workload-class routing from architecture research.

### Phase 3: Shared Audio Pipeline and Resource-Aware Performance
**Rationale:** Performance and quality improvements should be cross-architecture and contract-driven.
**Delivers:** First-class shared pre/post pipeline contracts, stage-aware artifacts/compare UX, warm-start pool with memory budget scheduler and eviction policy.
**Addresses:** Basic controllability quality, stage auditability, faster repeated generation.
**Avoids:** Audio contract regressions, OOM from unbounded warm loads, and architecture-specific logic leaking into shared layers.

### Phase 4: Architecture Expansion and Governance Controls
**Rationale:** Once baseline reliability exists, expansion can happen without destabilizing core flow.
**Delivers:** Additional adapter integrations under conformance tests, provenance/licensing metadata gates, policy controls for model import/serving.
**Addresses:** Sustainable architecture growth and safer model ecosystem operations.
**Avoids:** Upstream breakage shocks and trust/legal risk from weak provenance.

### Phase 5: Advanced Experience (Selective v2 Bets)
**Rationale:** High-complexity differentiators should only follow proven core reliability and usage.
**Delivers:** Cross-architecture preset portability and/or real-time conversion lane where validated by usage data.
**Addresses:** Competitive differentiation after PMF signals.
**Avoids:** Premature complexity that diverts from core orchestration value.

### Phase Ordering Rationale

- Contracts, compatibility, and security must precede scale/performance work; otherwise failures are amplified.
- Queue/state/storage reliability should come before warm-start and richer pipeline UX because those features depend on stable lifecycle semantics.
- Governance and advanced features belong after core operational confidence is established.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 3:** Warm-pool memory scheduling and audio quality contracts vary by architecture and hardware profile.
- **Phase 4:** Provenance/licensing policy implementation and adapter lifecycle strategy need legal/operational alignment.
- **Phase 5:** Real-time path feasibility requires latency benchmarking and transport/protocol decisions.

Phases with standard patterns (can usually skip deeper research-phase):
- **Phase 1:** Compatibility matrix, startup checks, host hardening, and validation UX are well-documented patterns.
- **Phase 2:** Queue partitioning, durable job metadata, retries, and retention controls are established implementation patterns.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Strong reliance on official docs (PyTorch, Docker, Celery, PostgreSQL, NVIDIA) with clear compatibility guidance. |
| Features | MEDIUM | Good triangulation across market references, but differentiator prioritization still depends on Hay Say user behavior. |
| Architecture | MEDIUM-HIGH | Patterns are mature and source-backed, but repo-specific migration complexity remains to be validated in implementation. |
| Pitfalls | MEDIUM | Risks are credible and well-mapped, though some severity assumptions should be validated with telemetry after rollout. |

**Overall confidence:** MEDIUM-HIGH

### Gaps to Address

- Benchmark-derived SLO targets are missing: define concrete latency/throughput/memory budgets in Phase 2-3 planning.
- Real-time conversion requirements are under-specified: validate target latency, codec/path tradeoffs, and acceptable quality floor before committing.
- Provenance policy boundaries need product/legal decision points: define what is required for local-only vs public/server deployments.
- Windows-specific storage lifecycle UX needs implementation validation: especially WSL2 VHDX reclaim behavior and user guidance flow.

## Sources

### Primary (HIGH confidence)
- Hay Say docs (`.planning/PROJECT.md`, `Readme.md`, `running as server/Readme.md`) - product constraints, current architecture, and deployment posture
- PyTorch install/support docs - Python/CUDA compatibility baseline
- Docker docs (Compose profiles, GPU support, BuildKit, service/startup references) - deployment/runtime patterns
- Celery docs (routing, optimization, Redis caveats) - queue design and failure semantics
- PostgreSQL official release docs - current major version baseline
- NVIDIA Container Toolkit docs - container GPU runtime requirements
- OpenTelemetry docs - collector deployment patterns (agent/gateway)

### Secondary (MEDIUM confidence)
- Market/competitor docs (ElevenLabs, PlayHT, cloud speech providers) - expectation-setting for feature parity and differentiation
- Open-source ecosystem references (RVC, OpenVoice, Coqui, w-okada) - practical capability and volatility signals
- PyPI version metadata checks - package line pinning support

### Tertiary (LOW confidence)
- None explicitly required; no single-source speculative claims were used as primary decision drivers.

---
*Research completed: 2026-02-27*
*Ready for roadmap: yes*
