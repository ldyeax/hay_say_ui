# Roadmap: Hay Say

## Overview

Hay Say v1 ships a reliable end-to-end voice workflow from install to replay, then hardens throughput and extensibility. Phases are ordered so platform/deployment and selective install come first, core generate/convert flows become usable next, then model controls and queue performance complete creator workflows, and finally extension docs make ongoing architecture growth repeatable.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [ ] **Phase 1: Platform Readiness and Selective Install** - Supported environments run safely and users install only what they need.
- [ ] **Phase 2: Unified Generation and Conversion Flow** - Users complete core TTS and voice conversion from one interface with clear guidance.
- [ ] **Phase 3: Model Lifecycle and Generation Controls** - Users manage model inventory and tune supported generation parameters.
- [ ] **Phase 4: Job Queue, History, and Runtime Performance** - Users run non-blocking workloads with replay and faster repeated runs.
- [ ] **Phase 5: Developer Extension Documentation** - Developers can extend architectures and shared processing with confidence.

## Phase Details

### Phase 1: Platform Readiness and Selective Install
**Goal**: Users can deploy Hay Say on supported platforms with safer defaults and install only selected architecture stacks.
**Depends on**: Nothing (first phase)
**Requirements**: PLAT-01, PLAT-02, MODL-01
**Success Criteria** (what must be TRUE):
  1. User can install and launch Hay Say on documented Windows and Linux paths without manual dependency wrangling.
  2. Operator can follow hosted/server hardening guidance and bring up a deployment with secure baseline settings.
  3. User can choose specific architecture bundles during install instead of pulling every architecture by default.
**Plans**: TBD

### Phase 2: Unified Generation and Conversion Flow
**Goal**: Users can run the primary pony voice workflows from one consistent interface and get clear recovery guidance when requests fail.
**Depends on**: Phase 1
**Requirements**: FLOW-01, FLOW-02, FLOW-03, FLOW-04, GENX-01, GENX-02
**Success Criteria** (what must be TRUE):
  1. User can generate speech from text for at least one installed architecture.
  2. User can upload source audio and convert it to a selected target voice.
  3. User can switch architectures in the same interface without moving to a different tool or page set.
  4. User can submit generation with missing inputs and receive clear field-level guidance for what to fix.
  5. User can preview/play outputs in-app and receives actionable error messages when a run fails.
**Plans**: TBD

### Phase 3: Model Lifecycle and Generation Controls
**Goal**: Users can manage installed voice assets and adjust supported synthesis controls from the main workflow.
**Depends on**: Phase 2
**Requirements**: MODL-02, MODL-03, MODL-04, GENX-03
**Success Criteria** (what must be TRUE):
  1. User can browse available models and voices scoped to installed architectures.
  2. User can import a custom model into a supported architecture and select it for generation.
  3. User can remove installed models or an architecture to reclaim disk space.
  4. User can adjust supported generation controls (for example rate, pitch, style) and see those choices reflected in outputs.
**Plans**: TBD

### Phase 4: Job Queue, History, and Runtime Performance
**Goal**: Users can run multiple jobs reliably, revisit prior outputs, and experience better repeat-generation responsiveness without exhausting system resources.
**Depends on**: Phase 3
**Requirements**: PERF-01, PERF-02, PERF-03, PERF-04
**Success Criteria** (what must be TRUE):
  1. User can queue multiple generation jobs and view per-job state from queued through completion/failure.
  2. User can open recent generation history and replay prior outputs without rerunning jobs.
  3. User sees improved latency on repeated jobs due to architecture warm-start behavior.
  4. User is protected from RAM/VRAM overload through admission or scheduling behavior that avoids runaway failures.
**Plans**: TBD

### Phase 5: Developer Extension Documentation
**Goal**: Developers can extend Hay Say architecture integrations and shared processing features using clear, repeatable documentation.
**Depends on**: Phase 4
**Requirements**: DOCS-01, DOCS-02
**Success Criteria** (what must be TRUE):
  1. Developer can follow docs to add a new architecture integration and run it through the existing workflow.
  2. Developer can follow docs to add preprocessing or postprocessing capabilities used by supported architectures.
  3. Developer can complete extension steps without relying on undocumented tribal knowledge.
**Plans**: TBD

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Platform Readiness and Selective Install | 0/TBD | Not started | - |
| 2. Unified Generation and Conversion Flow | 0/TBD | Not started | - |
| 3. Model Lifecycle and Generation Controls | 0/TBD | Not started | - |
| 4. Job Queue, History, and Runtime Performance | 0/TBD | Not started | - |
| 5. Developer Extension Documentation | 0/TBD | Not started | - |
