# Pitfalls Research

**Domain:** Local/self-hosted AI voice generation and voice conversion platforms
**Researched:** 2026-02-27
**Confidence:** MEDIUM

## Critical Pitfalls

### Pitfall 1: Treating GPU/runtime compatibility as "install-time only"

**What goes wrong:**
Upgrades break inference/training (CUDA mismatch, wrong PyTorch wheel, wrong Python range, missing NVIDIA runtime wiring), often after a seemingly harmless image refresh.

**Why it happens:**
Voice stacks span CUDA drivers, container runtime, PyTorch build, architecture-specific deps, and OS-specific paths. Teams update one layer without testing the full matrix.

**How to avoid:**
Maintain a pinned compatibility matrix per architecture (OS, Python, Torch, CUDA/ROCm, container runtime). Add a startup self-check endpoint that validates `torch.cuda.is_available()`, model file presence, FFmpeg presence, and architecture health before enabling Generate. Gate releases on matrix CI smoke tests.

**Warning signs:**
- Spike in "works on CPU but not GPU" support reports
- Frequent hotfixes to Docker images or compose flags
- New architecture integration requires manual one-off setup docs

**Phase to address:**
Phase 1 - Runtime Baseline and Compatibility Matrix

---

### Pitfall 2: Ignoring queue backpressure for long-running generation jobs

**What goes wrong:**
Queues grow unbounded, short tasks are stuck behind long jobs, RAM/VRAM usage climbs, and latency becomes unpredictable.

**Why it happens:**
Inference jobs are heterogeneous (short previews vs full conversions), but workers are configured as if tasks are uniform. Prefetch and concurrency defaults are left untouched.

**How to avoid:**
Split queues by workload class (interactive preview vs batch/full jobs). Set explicit worker prefetch/concurrency policies per queue. Add queue-length and task-age alerts, plus admission control when VRAM/RAM is near limits. Keep architecture warm pools bounded by memory budgets.

**Warning signs:**
- Queue age keeps rising even when workers are online
- P95/P99 generation latency drifts upward release-over-release
- Worker restarts correlate with memory spikes

**Phase to address:**
Phase 2 - Job Orchestration and Resource Scheduling

---

### Pitfall 3: Missing explicit security hardening in self-hosted/server mode

**What goes wrong:**
Open host-header patterns, weak TLS defaults, broad CORS/security-header gaps, and unsafe management endpoints expose the service to host-header abuse, cache poisoning, and operational compromise.

**Why it happens:**
Teams optimize for local usability and treat server deployment as "same config, bigger machine."

**How to avoid:**
Ship a hardened server profile by default: explicit allowed hosts, TLS-first ingress, rate limits on generation/download endpoints, strict security headers, and disabled model-management actions for public deployments. Add a deployment linter that rejects wildcard host patterns in production.

**Warning signs:**
- Deployment guides require manual edits for security-critical fields
- Public instance runs with wildcard hostnames or plain HTTP
- No environment split between local-dev and internet-exposed mode

**Phase to address:**
Phase 1 - Security Baseline for Self-Hosted Deployments

---

### Pitfall 4: Underestimating storage lifecycle (models, cache, image churn)

**What goes wrong:**
Disk fills unexpectedly from model growth, Docker image churn, cache accumulation, and Windows WSL2 VHDX non-shrinking behavior, causing outages and failed updates.

**Why it happens:**
Teams track only "initial install size" and skip lifecycle policies for cache retention, pruning, and per-architecture footprints.

**How to avoid:**
Implement storage budgets and telemetry per component (models, cache, images, logs). Add automated prune/retention tasks and explicit Windows disk-compaction workflow in-app. Keep selective architecture install/uninstall first-class, with size previews before changes.

**Warning signs:**
- "No space left on device" during updates
- Users repeatedly reinstall to reclaim disk
- Storage growth outpaces new feature adoption

**Phase to address:**
Phase 2 - Storage Governance and Lifecycle Controls

---

### Pitfall 5: Building on archived/upstream-volatile model ecosystems without insulation

**What goes wrong:**
An upstream project is archived or changes formats/tooling; integrations break, docs drift, and Hay Say release velocity stalls.

**Why it happens:**
Architecture adapters are tightly coupled to upstream scripts/config conventions, with little contract testing or migration abstraction.

**How to avoid:**
Define a stable internal adapter contract (capabilities, model metadata schema, required assets, health checks). Add conformance tests for each architecture plugin and version pinning with explicit EOL policy. Track upstream lifecycle status (active/archived/forked) and prefer maintained forks when primary is frozen.

**Warning signs:**
- New architecture support requires touching shared core code
- Frequent regressions when model formats or scripts change
- Documentation references dead/archived upstream paths

**Phase to address:**
Phase 3 - Architecture Plugin Contract and Lifecycle Management

---

### Pitfall 6: Audio pipeline assumptions (sample rate, clip length, loudness) not enforced

**What goes wrong:**
Output quality regresses (pitch artifacts, clipping, discontinuities) or inference fails because input constraints differ by encoder/model.

**Why it happens:**
Pipelines assume "any WAV in, good output out" while real model constraints are strict (clip length, mono/resample, encoder-specific limits).

**How to avoid:**
Codify preprocessing contracts per architecture and encoder (sample rate, channel, max clip length, loudness strategy). Add preflight validation that blocks incompatible jobs with clear remediation steps. Store and display processing provenance for reproducibility.

**Warning signs:**
- Rising support tickets for robotic/electric artifacts
- Same input produces different quality across architectures without explanation
- Manual "try random settings" guidance in docs/support

**Phase to address:**
Phase 2 - Unified Pre/Post Processing Contracts

---

### Pitfall 7: Weak model provenance and licensing guardrails

**What goes wrong:**
Teams accidentally ship questionable models/datasets or permit unsafe usage patterns, creating legal and trust risk.

**Why it happens:**
Model import flows optimize convenience, while provenance metadata, usage restrictions, and review checks are optional.

**How to avoid:**
Require provenance fields on model import (source URL, license, intended use, consent status), and block "unknown" provenance for shared/server deployments. Add policy-based gating in UI/API, plus audit logs for model add/remove/serve events.

**Warning signs:**
- Models in production with no source/license metadata
- Team cannot answer "where did this voice model come from?"
- Public instance allows arbitrary uploads with no review path

**Phase to address:**
Phase 3 - Governance, Provenance, and Policy Controls

---

### Pitfall 8: UX that hides recoverable failures instead of guiding users

**What goes wrong:**
Users see disabled controls or vague errors, retry blindly, and assume generation is broken when inputs are simply incomplete or incompatible.

**Why it happens:**
Validation happens late or only in backend logs; UI state does not expose required-field and compatibility feedback clearly.

**How to avoid:**
Move validation to explicit preflight checks on Generate click (never silently disable as the only signal). Return field-level errors and architecture-specific requirement hints. Add retry-safe error classes and one-click remediation actions.

**Warning signs:**
- High abandonment between upload and first successful generation
- Support requests asking why Generate is disabled/no-op
- Frequent duplicate failures from the same missing prerequisite

**Phase to address:**
Phase 1 - UX Validation and Error Experience Baseline

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Hardcode per-architecture quirks in shared logic | Faster first integration | Shared layer becomes fragile and blocks new adapters | Only for short-lived prototype branches |
| Keep all architectures always installed/running | Simpler mental model | Massive disk/RAM/VRAM waste and poor low-end usability | Never in default user path |
| "Just bump base image" updates | Quick patch velocity | Hidden CUDA/driver/runtime regressions | Only with matrix smoke tests and rollback plan |
| Let users manually clean storage | Reduced engineering effort now | Ongoing support burden and failed updates | Temporary until lifecycle automation ships |

## Integration Gotchas

Common mistakes when connecting to external services.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| GPU runtime (NVIDIA toolkit + Docker) | Installing toolkit but not configuring runtime and restart flow | Enforce post-install runtime configuration check and GPU test container on startup |
| Reverse proxy / DNS / TLS | Keeping wildcard hosts and HTTP defaults from local setup | Use explicit host allowlist, HTTPS redirect, certificate lifecycle checks |
| Model sources (HF/Drive/Mega/manual) | Importing without metadata and integrity checks | Require provenance metadata and checksum/manifest validation |

## Performance Traps

Patterns that work at small scale but fail as usage grows.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Unbounded warm-loading of architectures | OOM kills, degraded host responsiveness | Cap warm pool by memory budget and evict least-used architecture | On mixed workloads and modest GPUs/RAM |
| Single queue for all jobs | Interactive tasks blocked by long conversions | Queue tiering + routing by task duration/class | As soon as concurrent users submit long jobs |
| Cache without retention policy | Fast at first, then disk pressure outages | TTL + size caps + prune schedule + per-user quotas | After repeated sessions and model experimentation |

## Security Mistakes

Domain-specific security issues beyond general web security.

| Mistake | Risk | Prevention |
|---------|------|------------|
| Wildcard host settings in internet-facing mode | Host header abuse, cache poisoning vectors | Explicit host allowlist and deployment-time config linting |
| Exposing model-management endpoints publicly | Unauthorized model deletion/replacement | Disable/manage endpoints in public mode; require authz for admin actions |
| Missing TLS/security headers on server deployment | MITM and browser-side hardening gaps | HTTPS-first config + HSTS + baseline security headers |

## UX Pitfalls

Common user experience mistakes in this domain.

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Disable Generate button with no explanation | Perceived app failure | Allow click + show actionable missing-input errors |
| Architecture-specific constraints hidden from UI | Trial-and-error frustration | Show per-architecture requirements before run |
| Ambiguous quality controls | Inconsistent outputs and distrust | Surface safe presets with clear tradeoff labels (quality/speed/stability) |

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **Selective architecture install:** Verify uninstall actually frees disk and UI hides removed architecture paths
- [ ] **Warm-start performance:** Verify memory budget enforcement and eviction behavior under load
- [ ] **Server mode:** Verify wildcard hosts are removed, TLS is enabled, and security headers are active
- [ ] **Model import:** Verify provenance metadata is mandatory and policy gates block unknown-license models
- [ ] **Generate flow:** Verify preflight catches missing fields and incompatible audio before queue submission

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Runtime compatibility break after update | HIGH | Roll back image set, restore previous matrix lockfile, rerun GPU/FFmpeg health suite |
| Queue backlog meltdown | MEDIUM | Pause low-priority queues, spin up isolated workers, drain with SLA-based routing rules |
| Storage exhaustion outage | MEDIUM | Emergency prune policy, reclaim cache/images, enforce quotas, then backfill telemetry alerts |
| Security misconfiguration in public deploy | HIGH | Rotate credentials, patch ingress config, review access logs, publish incident remediation |

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Runtime compatibility drift | Phase 1 - Runtime Baseline and Compatibility Matrix | CI matrix passes on supported OS/GPU combos + startup self-check green |
| Queue/backpressure collapse | Phase 2 - Job Orchestration and Resource Scheduling | Queue age and P95 latency stay within SLO under load test |
| Self-hosted security misconfig | Phase 1 - Security Baseline for Self-Hosted Deployments | Deployment linter blocks wildcard hosts; HTTPS and header scan pass |
| Storage lifecycle failures | Phase 2 - Storage Governance and Lifecycle Controls | Disk growth stays within budget; prune/retention jobs execute successfully |
| Upstream volatility/archival risk | Phase 3 - Architecture Plugin Contract and Lifecycle Management | Adapter conformance tests pass per architecture version |
| Audio pipeline contract violations | Phase 2 - Unified Pre/Post Processing Contracts | Invalid inputs fail fast with deterministic, field-level messages |
| Provenance/licensing gaps | Phase 3 - Governance, Provenance, and Policy Controls | Every served model has provenance metadata and policy status |
| UX failure opacity | Phase 1 - UX Validation and Error Experience Baseline | First-run success rate and failed-job retry rate improve release-over-release |

## Sources

- Hay Say project context and docs (local): `.planning/PROJECT.md`, `Readme.md`, `running as server/Readme.md` (HIGH for current Hay Say constraints)
- so-vits-svc repository status and README (archived, operational cautions): https://github.com/svc-develop-team/so-vits-svc and https://raw.githubusercontent.com/svc-develop-team/so-vits-svc/4.1-Stable/README.md (HIGH)
- RVC README (dependency and platform complexity): https://raw.githubusercontent.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/main/docs/en/README.en.md (MEDIUM)
- NVIDIA Container Toolkit install/config guidance: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html (HIGH)
- PyTorch install/platform matrix guidance: https://pytorch.org/get-started/locally/ (HIGH)
- Celery optimization and queue/backpressure guidance: https://docs.celeryq.dev/en/stable/userguide/optimizing.html (HIGH)
- Docker prune/storage behavior: https://docs.docker.com/engine/manage-resources/pruning/ (HIGH)
- Django `ALLOWED_HOSTS` host-header protection: https://docs.djangoproject.com/en/stable/ref/settings/#allowed-hosts (HIGH)
- OWASP HTTP header hardening reference: https://cheatsheetseries.owasp.org/cheatsheets/HTTP_Headers_Cheat_Sheet.html (MEDIUM)

---
*Pitfalls research for: Hay Say (voice generation/conversion, local + self-hosted)*
*Researched: 2026-02-27*
