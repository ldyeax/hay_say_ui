# Hay Say

## What This Is

Hay Say is a Docker-first web interface for pony voice generation and voice conversion across multiple AI architectures from one place. It separates shared preprocessing/postprocessing from architecture-specific generation so new models can be integrated quickly without rebuilding the whole UX. The project serves creators and hobbyists who want local or self-hosted voice workflows without dependency-management pain.

## Core Value

A user can reliably generate or convert pony voices from a single, consistent interface without wrestling with per-architecture setup.

## Requirements

### Validated

(None yet - ship to validate)

### Active

- [ ] Users can install only the architectures they need instead of downloading the full stack.
- [ ] Users can run generation workflows with clearer validation feedback when required inputs are missing.
- [ ] Users can get faster repeated generations through architecture warm-start and resource-aware execution.
- [ ] Users can access richer preprocessing and postprocessing controls across supported architectures.
- [ ] Developers can follow clear docs for extending architectures, model packs, and processing features.

### Out of Scope

- Native mobile app clients - web UI remains the primary surface for v1 planning.
- Replacing upstream model repositories - Hay Say orchestrates and integrates, it does not become a model hosting platform.

## Context

The current repository describes an established UI that coordinates multiple architecture-specific backend containers via shared volumes and service calls. Existing docs emphasize installation friction, large storage requirements, cross-platform operation, and deployment patterns (including AWS hosting). The documented roadmap also highlights UX friction around the Generate flow, performance bottlenecks from cold-loading architectures, and demand for better extension/developer documentation.

## Constraints

- **Tech stack**: Dockerized multi-service architecture with Python web services - required to isolate architecture dependencies.
- **Compatibility**: Windows and Linux are primary supported targets - users depend on these environments today.
- **Performance**: Must run acceptably on resource-constrained machines - broad hardware variability is part of the user base.
- **Storage**: Full setup is very large - selective install and disk-efficiency improvements are high priority.
- **Security**: Hosted/server deployments need safer defaults - docs already flag host-header and SSL hardening requirements.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Keep architecture-agnostic preprocessing/postprocessing as a shared layer | New architectures should benefit from common improvements immediately | - Pending |
| Keep Docker-first packaging and distribution | Minimizes dependency conflicts for end users | - Pending |
| Prioritize install/selectivity and performance before broad feature expansion | Current pain points are setup size and runtime responsiveness | - Pending |

---
*Last updated: 2026-02-27 after initialization*
