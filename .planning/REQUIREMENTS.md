# Requirements: Hay Say

**Defined:** 2026-02-27
**Core Value:** A user can reliably generate or convert pony voices from a single, consistent interface without wrestling with per-architecture setup.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Core Workflow

- [ ] **FLOW-01**: User can generate speech from text in at least one supported architecture.
- [ ] **FLOW-02**: User can convert an uploaded source voice to a selected target voice.
- [ ] **FLOW-03**: User can switch architectures from one unified interface without changing tools.
- [ ] **FLOW-04**: User can preview and play generated output in the app.

### Model Management

- [ ] **MODL-01**: User can install only selected architectures instead of downloading all by default.
- [ ] **MODL-02**: User can browse available models/voices for installed architectures.
- [ ] **MODL-03**: User can import a custom model for a supported architecture.
- [ ] **MODL-04**: User can remove an installed model or architecture to reclaim disk space.

### Generation UX

- [ ] **GENX-01**: User can click Generate even when inputs are incomplete and receive clear field-level guidance.
- [ ] **GENX-02**: User sees actionable error messages when generation fails.
- [ ] **GENX-03**: User can control basic generation parameters (for example rate, pitch, style) where supported.

### Jobs and Performance

- [ ] **PERF-01**: User can queue multiple generation jobs and view each job status.
- [ ] **PERF-02**: User can access recent generation history and replay prior outputs.
- [ ] **PERF-03**: User benefits from architecture warm-start behavior that reduces repeated generation latency.
- [ ] **PERF-04**: User is protected from RAM/VRAM overload by resource-aware execution policies.

### Platform and Documentation

- [ ] **PLAT-01**: User can run Hay Say on supported Windows and Linux setups with documented install paths.
- [ ] **PLAT-02**: Operator can deploy a hosted/server variant with documented security-hardening steps.
- [ ] **DOCS-01**: Developer can follow documentation to add a new architecture integration.
- [ ] **DOCS-02**: Developer can follow documentation to add preprocessing/postprocessing capabilities.

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Advanced Experience

- **ADVN-01**: User can run low-latency real-time voice conversion for live scenarios.
- **ADVN-02**: User can apply portable presets across architectures through a translation layer.
- **ADVN-03**: User can use advanced provenance/consent automation for shared voice assets.

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Native mobile app clients | Web-first direction; no mobile app investment in this planning cycle |
| Built-in model marketplace | Pulls product away from orchestration focus and adds moderation/legal overhead |
| Full parameter parity across all architectures | Different model families cannot support exact one-to-one controls |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| FLOW-01 | Phase TBD | Pending |
| FLOW-02 | Phase TBD | Pending |
| FLOW-03 | Phase TBD | Pending |
| FLOW-04 | Phase TBD | Pending |
| MODL-01 | Phase TBD | Pending |
| MODL-02 | Phase TBD | Pending |
| MODL-03 | Phase TBD | Pending |
| MODL-04 | Phase TBD | Pending |
| GENX-01 | Phase TBD | Pending |
| GENX-02 | Phase TBD | Pending |
| GENX-03 | Phase TBD | Pending |
| PERF-01 | Phase TBD | Pending |
| PERF-02 | Phase TBD | Pending |
| PERF-03 | Phase TBD | Pending |
| PERF-04 | Phase TBD | Pending |
| PLAT-01 | Phase TBD | Pending |
| PLAT-02 | Phase TBD | Pending |
| DOCS-01 | Phase TBD | Pending |
| DOCS-02 | Phase TBD | Pending |

**Coverage:**
- v1 requirements: 19 total
- Mapped to phases: 0
- Unmapped: 19

---
*Requirements defined: 2026-02-27*
*Last updated: 2026-02-27 after initial definition*
