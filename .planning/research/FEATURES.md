# Feature Research

**Domain:** Multi-architecture voice generation and voice conversion platform (creators + self-hosters)
**Researched:** 2026-02-27
**Confidence:** MEDIUM

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist. Missing these = product feels incomplete.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Unified TTS + voice conversion in one UI | Major products and open-source tools expose both text-to-speech and speech-to-speech workflows | MEDIUM | Keep architecture-agnostic UX shell with architecture-specific options panel |
| Multi-language generation and voice support | Commercial APIs and open stacks now treat multilingual output as standard | MEDIUM | Must expose language/locale selection and per-voice language compatibility |
| Voice/model library management (browse, import, remove) | Users expect large voice catalogs and personal voice collections | MEDIUM | Includes local model install/uninstall and custom model import pathways |
| Basic controllability of delivery (rate/pitch/style/prosody) | SSML/prosody controls are standard across major speech platforms | MEDIUM | Normalize controls to a shared schema and degrade gracefully per architecture |
| Streaming or low-latency preview path | Real-time/near-real-time playback is widely available in cloud and community tools | HIGH | Split into quick preview path vs high-quality final render path |
| Batch generation + queue visibility | Creator workflows require multiple renders and non-blocking jobs | MEDIUM | Queue status, retries, cancel, and output history are expected UX |
| Clear validation and actionable errors before/after generate | Users expect form-level validation, not silent disabled actions | LOW | Keep Generate enabled; highlight missing required fields + fix suggestions |
| Resource-aware local execution (CPU/GPU selection + fallback) | Self-hosters expect reliability across heterogeneous hardware | HIGH | Detect hardware, choose backend, and surface memory/VRAM pressure warnings |

### Differentiators (Competitive Advantage)

Features that set the product apart. Not required, but valuable.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Architecture warm-start pool with memory budget scheduler | Dramatically reduces repeat latency while preventing RAM/VRAM overload | HIGH | Keep hot models loaded by LRU + per-device memory caps |
| Cross-architecture preset portability | Creators can reuse one preset intent (e.g., "clean narrate") across engines | HIGH | Requires semantic mapping layer from unified controls to engine-specific params |
| Stage-aware audio pipeline (raw -> preprocessed -> generated -> postprocessed compare) | Gives creators auditability and faster iteration on quality | MEDIUM | Hay Say already has stage cache concept; expose diff/listen UX strongly |
| Security-hardened self-host mode (safe host headers, TLS assist, role controls) | Makes community hosting safer without deep DevOps skills | HIGH | Bundle secure defaults and setup checks for server deployments |
| Consent and provenance tooling for voice cloning | Reduces legal/trust risk and supports responsible creator usage | MEDIUM | Store consent metadata and show provenance notices in project/export metadata |
| Optional real-time conversion mode for live use cases | Expands from offline creator workflows to live streaming/gaming workflows | HIGH | Separate low-latency model path from offline high-fidelity path |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem good but create problems.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| "Install everything" by default | Feels easy for new users | Massive disk footprint and slower updates; bad fit for self-hosters | Architecture selector + progressive install with disk estimate |
| Always keep all architectures loaded | Reduces first-run latency | Causes RAM/VRAM thrash and crashes on common hardware | Warm pool with hard memory budget + idle eviction |
| Full parameter parity across every architecture | Users want one-to-one controls | Different model families cannot map perfectly; creates misleading UX | Shared core controls + architecture-specific advanced panel |
| Anonymous cloning with no provenance/consent flow | Frictionless experimentation | High abuse/legal risk and community trust damage | Lightweight consent capture + watermark/provenance metadata options |
| Building a model marketplace inside Hay Say | Looks like growth lever | Pulls product into hosting/business complexity outside core orchestration value | Integrate external model sources and keep Hay Say as orchestration layer |

## Feature Dependencies

```
[Unified model/voice management]
    └──requires──> [Architecture registry + capability metadata]
                          └──requires──> [Per-architecture adapter contracts]

[Batch queue + output history]
    └──requires──> [Job orchestration + cache lifecycle]
                          └──requires──> [Storage quotas + retention policies]

[Warm-start pool]
    └──requires──> [Resource telemetry (RAM/VRAM)]
                          └──requires──> [Scheduler + eviction policy]

[Cross-architecture preset portability]
    └──requires──> [Unified control schema]
                          └──requires──> [Per-engine parameter translators]

[Real-time conversion mode] ──conflicts──> [Maximum-quality offline postprocessing defaults]

[Consent/provenance tooling] ──enhances──> [Custom voice cloning + sharing]
```

### Dependency Notes

- **Unified model/voice management requires architecture registry + capability metadata:** UI cannot present valid choices unless each architecture declares supported tasks, controls, and constraints.
- **Batch queue + output history requires job orchestration + cache lifecycle:** creator workflows break without deterministic job state and artifact retention.
- **Warm-start pool requires resource telemetry:** keeping models warm is only safe if scheduler can enforce memory budgets.
- **Cross-architecture preset portability requires unified control schema:** portability is a translation problem; schema comes before translators.
- **Real-time conversion conflicts with max-quality defaults:** low-latency pipelines must bypass heavy denoise/post chains and some larger models.

## MVP Definition

### Launch With (v1)

Minimum viable product - what's needed to validate the concept.

- [ ] Unified TTS + voice conversion workflow with architecture selection - core user value of one UI for multiple engines
- [ ] Selective architecture/model install and management - solves major storage/setup pain in self-hosting
- [ ] Clear validation + actionable errors in Generate flow - immediate UX quality and lower support burden
- [ ] Basic controllability (rate/pitch/style where supported) - minimum creative control expected in 2026
- [ ] Queue + history for non-blocking generation - required for creator iteration loops

### Add After Validation (v1.x)

Features to add once core is working.

- [ ] Warm-start pool with memory-aware scheduling - add once baseline stability telemetry exists
- [ ] Stage-aware compare/listen UX upgrades - add after reliable cache lifecycle is in place
- [ ] Security-hardened self-host setup assistant - add after core local UX is stable

### Future Consideration (v2+)

Features to defer until product-market fit is established.

- [ ] Cross-architecture preset portability - high leverage, but requires mature abstraction and translation layer
- [ ] Real-time live conversion mode - valuable expansion, but high latency/perf complexity
- [ ] Advanced consent/provenance automation - important for scale and sharing ecosystems

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Unified TTS + voice conversion flow | HIGH | MEDIUM | P1 |
| Selective architecture/model install | HIGH | MEDIUM | P1 |
| Generate validation + error guidance | HIGH | LOW | P1 |
| Queue + history | HIGH | MEDIUM | P1 |
| Warm-start model pool | HIGH | HIGH | P2 |
| Stage compare/listen workflow | MEDIUM | MEDIUM | P2 |
| Security-hardened server mode | MEDIUM | HIGH | P2 |
| Cross-architecture preset portability | HIGH | HIGH | P3 |
| Real-time live conversion mode | MEDIUM | HIGH | P3 |

**Priority key:**
- P1: Must have for launch
- P2: Should have, add when possible
- P3: Nice to have, future consideration

## Competitor Feature Analysis

| Feature | Competitor A | Competitor B | Our Approach |
|---------|--------------|--------------|--------------|
| Voice cloning options | ElevenLabs: instant + professional cloning and large voice library | OpenVoice: open-source instant cloning + style control | Keep local-first cloning/import + optional external model interoperability |
| Low-latency speech path | ElevenLabs Flash (~75ms) and PlayHT websocket/streaming focus | w-okada VC ecosystem emphasizes real-time conversion | Two-lane pipeline: low-latency preview/live + high-quality offline render |
| Voice/style control depth | Azure/AWS expose rich SSML/prosody/style controls | Open-source stacks expose engine-specific tuning | Unified core controls plus explicit advanced per-engine panels |
| Self-hosting posture | Cloud vendors emphasize managed APIs and enterprise controls | Community tools emphasize local control, often weaker safety defaults | Make self-hosting first-class with secure defaults and guided hardening |

## Sources

- Hay Say project context and docs: `/home/luna/hay_say/hay_say_ui/.planning/PROJECT.md`, `/home/luna/hay_say/hay_say_ui/Readme.md`, `/home/luna/hay_say/hay_say_ui/running as server/Readme.md` (HIGH)
- ElevenLabs docs: Text to Speech, Voice Changer, Voices, Private deployment (official docs, accessed 2026-02-27) (HIGH)
  - https://elevenlabs.io/docs/overview/capabilities/text-to-speech
  - https://elevenlabs.io/docs/overview/capabilities/voice-changer
  - https://elevenlabs.io/docs/overview/capabilities/voices
  - https://elevenlabs.io/docs/eleven-api/private-deployment/overview
- PlayHT API docs (Quickstart / feature surface, accessed 2026-02-27) (MEDIUM)
  - https://docs.play.ht/reference/api-getting-started
- Azure Speech SSML voice controls (official docs, updated 2026-01-30) (HIGH)
  - https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-synthesis-markup-voice
- AWS Polly docs: service overview and SSML tag support (official docs, accessed 2026-02-27) (HIGH)
  - https://docs.aws.amazon.com/polly/latest/dg/what-is.html
  - https://docs.aws.amazon.com/polly/latest/dg/supportedtags.html
- Open-source ecosystem references (official repos, accessed 2026-02-27) (MEDIUM)
  - https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
  - https://github.com/w-okada/voice-changer
  - https://github.com/myshell-ai/OpenVoice
  - https://docs.coqui.ai/en/stable/
  - https://docs.fish.audio/

---
*Feature research for: multi-architecture voice generation/conversion*
*Researched: 2026-02-27*
