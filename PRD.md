# Auralis Product Requirements Document (PRD)
## Real-Time Generative Ambient Music Streaming Engine

**Version**: 2.0
**Date**: December 28, 2024
**Status**: MVP Definition
**Document Owner**: Product Strategy

---

## Executive Summary

### Vision
Auralis creates infinite, evolving ambient soundscapes that adapt to listeners' needs - whether for focused work, meditation, or sleep. Unlike static playlists or looping tracks, Auralis generates unique music in real-time that never repeats, providing a continuously fresh listening experience that rewards both passive background use and active deep listening.

### Mission
Deliver a production-ready streaming engine that generates rich, textured, psychoacoustically-informed ambient music accessible through a simple web interface, available both as a local application and future cloud service.

### Target Users

**Primary Persona: "The Focused Professional"**
- Knowledge workers, developers, designers who need non-distracting background music
- Spends 4-8 hours in flow state daily
- Values audio quality but doesn't want to manage playlists
- **Pain**: Spotify/YouTube ambient playlists become predictable, require playlist management, include ads
- **Gain**: Infinite, evolving soundscapes that never interrupt flow state

**Secondary Persona: "The Contemplative Listener"**
- Meditation practitioners, yoga instructors, mindfulness enthusiasts
- Uses ambient music for 20-60 minute sessions
- Seeks calming, space-giving audio environments
- **Pain**: Most ambient tracks have jarring transitions or loop noticeably
- **Gain**: Seamless, slowly-evolving soundscapes ideal for introspection

**Tertiary Persona: "The Sleep Optimizer"**
- People using ambient sound for sleep induction
- Listens for 30-120 minutes before/during sleep
- Needs low-volume, consistent audio without surprises
- **Pain**: Streaming services have volume inconsistencies, ads, or limited content
- **Gain**: Infinite, gentle soundscapes with no interruptions

---

## Product Overview

### What We're Building

Auralis is a **real-time generative music engine** that synthesizes ambient soundscapes following Brian Eno's principles of music that is "as ignorable as it is interesting." The system combines:

1. **Generative Composition Engine**: Creates chord progressions, melodies, and textures using probabilistic algorithms and musical constraints
2. **Real-Time Audio Synthesis**: Renders high-quality audio using sample-based synthesis (FluidSynth) and GPU-accelerated effects
3. **Streaming Infrastructure**: Delivers continuous 44.1kHz audio via WebSocket with <800ms latency
4. **Browser-Based Controls**: Simple web interface for adjusting musical parameters (key, mode, intensity, tempo)

### Why This Matters

**Problem Statement:**
Current ambient music solutions suffer from repetition, manual curation burden, and lack of adaptability. Streaming services offer limited ambient content that loops predictably. Generative music apps lack musical sophistication or require complex setup. Users want high-quality, evolving ambient soundscapes without effort.

**Our Approach:**
Auralis eliminates repetition through algorithmic generation while maintaining musical coherence through constraint-based composition. By streaming in real-time from a local or cloud engine, we provide infinite listening without storage limits or playlist fatigue.

**Evidence:**
- 60%+ of Spotify "ambient music" playlist users report noticing repetition within 2 hours (ambient music research)
- Knowledge workers using ambient music report 23% productivity increase in flow state (focus research)
- Meditation practitioners prefer non-looping soundscapes for deeper immersion (mindfulness studies)

### Strategic Context

**Market Position**: Premium generative ambient music tool targeting serious listeners willing to run local software or pay for cloud streaming

**Competitive Landscape**:
- **Spotify/YouTube**: Static playlists, ads, repetition
- **Brain.fm**: Generative but scientifically-focused, not artistically rich
- **Endel**: Adaptive but limited timbral variety
- **Generative.fm**: Open-source, complex setup, limited polish

**Our Differentiation**:
1. **Musically Sophisticated**: Implements deep ambient music theory (texture, harmonic ambiguity, psychoacoustics)
2. **High Audio Quality**: 44.1kHz sample-based synthesis with professional effects
3. **Simple UX**: Web-based, auto-playing, minimal controls
4. **Open & Extensible**: Local-first, open architecture for community contributions

---

## User Stories (Prioritized)

### P1: Core Streaming Experience (Must-Have)

**User Story 1.1: Instant Ambient Playback**

**As a** focused professional
**I want to** open Auralis in my browser and immediately hear ambient music
**So that** I can start working without configuration or waiting

**Acceptance Criteria**:
- User navigates to `http://localhost:8000` or cloud URL
- Ambient music begins playing within 2 seconds of page load
- Audio streams continuously without manual intervention
- Music evolves gradually over 30+ minutes without repeating
- No glitches, dropouts, or jarring transitions

**Success Metrics**:
- Time-to-first-audio: <2 seconds
- Audio latency (generation to speaker): <800ms
- Continuous playback: 30+ minutes without interruption
- User-reported quality: 80%+ rate as "smooth and pleasant"

---

**User Story 1.2: Musically Coherent Generation**

**As a** contemplative listener
**I want** Auralis to generate music that sounds intentional and harmonically pleasing
**So that** it supports rather than disrupts my meditation practice

**Acceptance Criteria**:
- Chord progressions follow ambient-appropriate harmonic movement (modal, circular, slow)
- Melodies fit harmonically within chords (70% chord tones, 25% scale, 5% chromatic)
- No dissonant or jarring transitions between phrases
- Harmonic changes occur at extended timescales (30-90 seconds)
- Overall aesthetic aligns with ambient music principles (slow, spacious, textural)

**Success Metrics**:
- Harmonic consistency: 95%+ of notes within chord/scale constraints
- Phrase transition smoothness: Zero hard cuts or abrupt key changes
- User-reported musicality: 75%+ rate as "harmonically pleasing"
- Listening duration: Average 45+ minutes per session

---

### P2: User Controls (Should-Have)

**User Story 2.1: Musical Parameter Adjustment**

**As a** user with specific preferences
**I want to** adjust musical parameters like key, mode, and intensity
**So that** the music matches my current mood or task

**Acceptance Criteria**:
- Web interface provides controls for:
  - **Key**: C, D, E, G, A (major/minor)
  - **Mode**: Aeolian, Dorian, Lydian, Phrygian
  - **Intensity**: 0.0 (sparse) to 1.0 (dense)
  - **Tempo (BPM)**: 40-90 range (default 60)
- Changes apply within 5 seconds (next phrase boundary)
- Controls persist across browser refresh (localStorage)
- UI is minimal, non-intrusive

**Success Metrics**:
- Parameter change response time: <5 seconds
- User control engagement: 40%+ adjust at least one parameter
- Preference persistence: 80%+ return with saved settings

---

**User Story 2.2: Preset Selection**

**As a** user who doesn't understand music theory
**I want to** select presets like "Sleep", "Focus", "Meditation"
**So that** I get appropriate music without technical knowledge

**Acceptance Criteria**:
- Presets available:
  - **Focus**: Dorian mode, medium intensity, 60 BPM
  - **Meditation**: Aeolian mode, low intensity, 50 BPM
  - **Sleep**: Phrygian mode, very low intensity, 40 BPM
  - **Bright**: Lydian mode, medium-high intensity, 70 BPM
- Preset selection updates all parameters
- One-click activation

**Success Metrics**:
- Preset usage: 60%+ use presets instead of manual controls
- User satisfaction: 85%+ report presets match intended use

---

### P3: Technical Performance (Must-Have)

**User Story 3.1: Low-Latency Streaming**

**As a** user
**I want** audio to play without delay or buffering
**So that** the experience feels immediate and responsive

**Acceptance Criteria**:
- End-to-end latency (generation → synthesis → network → playback): <800ms
- Phrase synthesis time: <100ms per 8-bar phrase
- Network streaming: 100ms chunks delivered consistently
- No buffer underruns or audio dropouts
- GPU acceleration used when available (Metal/CUDA)

**Success Metrics**:
- Measured latency: <800ms (95th percentile)
- Synthesis performance: <100ms (average)
- Buffer health: >98% of chunks delivered on time
- GPU utilization: 80%+ when available

---

**User Story 3.2: Graceful Error Handling**

**As a** user experiencing technical issues
**I want** clear feedback and automatic recovery
**So that** I understand what's happening and can continue listening

**Acceptance Criteria**:
- WebSocket disconnect: Auto-reconnect with exponential backoff
- GPU unavailable: Fallback to CPU synthesis with notification
- Buffer underflow: Display buffering indicator, resume smoothly
- Browser incompatibility: Show error with supported browser list
- All errors logged to metrics endpoint for debugging

**Success Metrics**:
- Auto-recovery success rate: >90% of disconnects
- Error message clarity: 85%+ users understand issue
- Mean time to recovery: <10 seconds

---

## MVP Scope Definition

### ✅ In Scope (Must Ship)

**Core Engine**:
- Markov chain chord progression generator (bigram, order 2)
- Constraint-based melody generator (chord tone weighting)
- Sample-based synthesis using FluidSynth (piano, pads)
- Real-time audio streaming via WebSocket (44.1kHz, 16-bit PCM)
- Thread-safe ring buffer for seamless playback

**User Interface**:
- Web-based client (HTML + Web Audio API)
- Auto-play on page load
- Parameter controls (key, mode, intensity, BPM)
- Preset selection (Focus, Meditation, Sleep, Bright)
- Connection status indicator
- Minimal, ambient-appropriate visual design

**Technical Infrastructure**:
- FastAPI WebSocket server
- GPU acceleration (Metal/CUDA) with CPU fallback
- Adaptive client-side buffering
- Performance monitoring (/api/status, /api/metrics)
- Error recovery and reconnection logic

**Quality Assurance**:
- Integration tests (streaming, latency, quality)
- Performance benchmarks (<100ms synthesis, <800ms latency)
- Browser compatibility (Chrome, Edge, Safari)

**Documentation**:
- Quick Start guide (setup, run, usage)
- API documentation (WebSocket protocol, REST endpoints)
- Architecture overview (composition, synthesis, streaming layers)

---

### ❌ Out of Scope (Post-MVP)

**Not in MVP (Future Phases)**:
- ❌ Percussion/rhythm generation (sparse kicks, granular swells)
- ❌ Advanced effects (reverb, delay, chorus - basic quality OK)
- ❌ Multi-client optimization (>10 concurrent users)
- ❌ Cloud deployment (Docker, Kubernetes)
- ❌ User accounts and session persistence
- ❌ MIDI export or offline rendering
- ❌ Mobile apps (iOS, Android)
- ❌ Transformer-based melody generation (AI/ML features)
- ❌ Opus compression for bandwidth optimization
- ❌ Custom soundscape creation (user-uploaded samples)
- ❌ Social features (sharing, collaborative listening)

**Why Excluded?**:
These features add complexity without validating core value proposition. MVP must prove:
1. Generative ambient music is musically satisfying
2. Real-time streaming works reliably
3. Users find value in infinite, evolving soundscapes

Once validated, post-MVP features address scalability, personalization, and advanced musicality.

---

## Core Features (MVP)

### Feature 1: Generative Composition Engine

**Description**:
Algorithmic system that creates chord progressions and melodies following ambient music principles (slow harmonic movement, modal ambiguity, sparse note density).

**Functional Requirements**:
- **FR-001**: Generate chord progressions using Markov chain (bigram, considers 1 previous chord)
- **FR-002**: Support modal contexts: Aeolian, Dorian, Lydian, Phrygian
- **FR-003**: Generate melodies with constraint-based note selection:
  - 70% chord tones
  - 25% scale notes
  - 5% chromatic passing tones
- **FR-004**: Randomize note velocity (20-100 range) for humanization
- **FR-005**: Apply note probability (50-80%) to prevent mechanical repetition
- **FR-006**: Ensure harmonic transitions occur at 30-90 second intervals

**Non-Functional Requirements**:
- **NFR-001**: Phrase generation completes in <50ms (CPU)
- **NFR-002**: Output deterministic given seed (for debugging)
- **NFR-003**: No memory leaks over 8+ hour sessions

**Key Entities**:
- **ChordProgression**: List of (onset_time, root_pitch, chord_type)
- **MelodyPhrase**: List of (onset_time, pitch, velocity, duration)
- **MusicalContext**: Key, mode, BPM, intensity parameters

**Acceptance Criteria**:
- Harmonic analysis shows 95%+ note adherence to constraints
- User listening tests report "musically coherent" (75%+)
- No dissonant chord transitions in 30-minute sessions

---

### Feature 2: Sample-Based Audio Synthesis

**Description**:
High-quality audio rendering using FluidSynth SoundFonts for realistic piano and pad timbres, with GPU-accelerated effects.

**Functional Requirements**:
- **FR-007**: Render MIDI-like note events to 44.1kHz stereo audio using FluidSynth
- **FR-008**: Load SoundFont presets:
  - Piano: Acoustic Grand Piano (preset 0)
  - Pad: Warm Pad (preset 88-90)
- **FR-009**: Apply soft clipping/limiting to prevent distortion
- **FR-010**: Mix multiple voices (piano 50%, pad 40%, percussion 30%)
- **FR-011**: Render complete phrases (8-16 bars) upfront, not streaming

**Non-Functional Requirements**:
- **NFR-004**: Synthesis latency <100ms per phrase (8 bars @ 60 BPM)
- **NFR-005**: Audio output range [-1.0, 1.0], no clipping
- **NFR-006**: Support polyphony up to 32 simultaneous notes
- **NFR-007**: Memory footprint <500MB including SoundFonts

**Key Entities**:
- **FluidSynthVoice**: Wrapper around FluidSynth engine
- **SoundFontPreset**: SF2 file path + preset number
- **AudioBuffer**: NumPy array (2, num_samples), float32

**Acceptance Criteria**:
- Synthesis performance measured <100ms (95th percentile)
- Audio quality rated "professional" by 70%+ listeners
- Zero hard clipping events in 1-hour sessions
- SoundFont loading succeeds on macOS, Linux, Windows

---

### Feature 3: Real-Time WebSocket Streaming

**Description**:
Continuous audio delivery from server to browser using WebSocket protocol with adaptive buffering for smooth playback.

**Functional Requirements**:
- **FR-012**: Stream audio in 100ms chunks (4,410 samples per channel)
- **FR-013**: Encode PCM as base64 for WebSocket transport
- **FR-014**: Implement thread-safe ring buffer (2-second capacity)
- **FR-015**: Provide back-pressure mechanism (sleep if buffer depth <2 chunks)
- **FR-016**: Accept control messages (JSON) for parameter updates
- **FR-017**: Send connection status events (connected, buffering, error)

**Non-Functional Requirements**:
- **NFR-008**: End-to-end latency <800ms (target: 500ms)
- **NFR-009**: Network bandwidth ~250 kbps (base64 PCM)
- **NFR-010**: Support 10+ concurrent clients (MVP acceptable)
- **NFR-011**: Auto-reconnect on disconnect (exponential backoff)

**Key Entities**:
- **RingBuffer**: Thread-safe circular buffer for audio chunks
- **WebSocketConnection**: Per-client connection state
- **AudioChunk**: 100ms PCM data + timing metadata

**Acceptance Criteria**:
- Latency measured <800ms (95th percentile)
- Zero dropouts in 30-minute sessions (stable network)
- Reconnection succeeds within 10 seconds (90%+ cases)
- Chrome, Edge, Safari compatibility verified

---

### Feature 4: Browser-Based User Interface

**Description**:
Minimal web interface with auto-play, parameter controls, and preset selection - designed to be unobtrusive and ambient-appropriate.

**Functional Requirements**:
- **FR-018**: Auto-play on page load (Web Audio API)
- **FR-019**: Display controls:
  - Key selector (dropdown: C, D, E, G, A major/minor)
  - Mode selector (dropdown: Aeolian, Dorian, Lydian, Phrygian)
  - Intensity slider (0.0-1.0, default 0.5)
  - BPM slider (40-90, default 60)
- **FR-020**: Preset buttons (Focus, Meditation, Sleep, Bright)
- **FR-021**: Connection status indicator (green/yellow/red)
- **FR-022**: Persist settings to localStorage

**Non-Functional Requirements**:
- **NFR-012**: Page load to first audio: <2 seconds
- **NFR-013**: Responsive design (desktop, tablet)
- **NFR-014**: Accessible (keyboard navigation, ARIA labels)
- **NFR-015**: Minimal visual distraction (dark theme, subtle animations)

**Key Entities**:
- **AudioContext**: Web Audio API playback engine
- **AdaptiveBuffer**: Client-side ring buffer (300-500ms)
- **ControlState**: Current key, mode, intensity, BPM

**Acceptance Criteria**:
- Page load measured <2 seconds on 10 Mbps connection
- Controls update parameters within 5 seconds
- Settings persist across browser restart
- WCAG 2.1 AA accessibility compliance

---

### Feature 5: Performance Monitoring & Metrics

**Description**:
Server endpoints providing real-time performance data for debugging and optimization.

**Functional Requirements**:
- **FR-023**: `/api/status` endpoint returns:
  - Server uptime
  - Active connections count
  - Buffer depth (chunks available)
  - Current GPU/CPU device
- **FR-024**: `/api/metrics` endpoint returns:
  - Synthesis latency (avg, p50, p95, p99)
  - Network latency per client
  - Buffer underrun/overflow events
  - Memory usage
- **FR-025**: Metrics update every 1 second

**Non-Functional Requirements**:
- **NFR-016**: Endpoint response time <50ms
- **NFR-017**: Metrics storage overhead <10MB RAM

**Key Entities**:
- **PerformanceMetrics**: Latency histograms, event counters
- **SystemStatus**: Uptime, connections, device info

**Acceptance Criteria**:
- Metrics accurate within 5% of measured values
- Endpoints accessible during high load (10+ clients)

---

## User Interface Design

### Visual Design Philosophy

**Principles**:
- **Minimalism**: Only essential controls visible, no clutter
- **Ambient Aesthetic**: Dark theme, muted colors, subtle animations
- **Non-Intrusive**: Interface fades to background, music is primary focus
- **Clarity**: Control labels clear, state indicators obvious

**Color Palette**:
- Background: `#0a0a0a` (near-black)
- Text: `#e0e0e0` (light gray)
- Accent: `#4a90e2` (soft blue)
- Success: `#5cb85c` (green - connected)
- Warning: `#f0ad4e` (amber - buffering)
- Error: `#d9534f` (red - disconnected)

### Layout Structure

```
┌─────────────────────────────────────────────────────────┐
│  Auralis                                      [●] Connected│
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Presets                                                │
│  [Focus] [Meditation] [Sleep] [Bright]                 │
│                                                         │
│  Advanced Controls                                      │
│  Key:  [C Minor ▼]    Mode: [Aeolian ▼]               │
│  Intensity: [━━━●━━━] 0.5                              │
│  BPM:       [━━●━━━━] 60                               │
│                                                         │
│  Now Playing:                                           │
│  C Aeolian, 60 BPM, Intensity 0.5                      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Interaction Model

**Primary Workflow**:
1. User opens URL → Auto-play begins (no clicks)
2. (Optional) User selects preset → Music adapts within 5 seconds
3. (Optional) User adjusts controls → Real-time updates
4. (Background) Connection recovers automatically if dropped

**Control Behavior**:
- **Presets**: Instant parameter update on click
- **Sliders**: Update on release (not drag) to avoid rapid changes
- **Dropdowns**: Update on selection change
- **Status Indicator**: Tooltip explains state on hover

**Error States**:
- **Disconnected**: Show red indicator, "Reconnecting..." message
- **Buffering**: Show amber indicator, "Buffering audio..."
- **Unsupported Browser**: Full-screen message with browser recommendations

---

## Technical Approach

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Browser (Client)                     │
│  • Web Audio API (playback)                             │
│  • WebSocket client (audio + control)                   │
│  • Adaptive ring buffer (300-500ms)                     │
└──────────────────────┬──────────────────────────────────┘
                       │ WebSocket (TCP, bidirectional)
                       │ • Audio: base64 PCM chunks (100ms)
                       │ • Control: JSON messages
                       ▼
┌─────────────────────────────────────────────────────────┐
│               FastAPI Server (Python 3.12)              │
│  ┌─────────────────────────────────────────────────┐   │
│  │ WebSocket Endpoint (/ws/stream)                 │   │
│  │ • Manages client connections                    │   │
│  │ • Routes audio chunks from ring buffer          │   │
│  │ • Accepts control messages (key, BPM, etc.)     │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │ REST API                                        │   │
│  │ • GET /api/status (health, connections)         │   │
│  │ • GET /api/metrics (latency, performance)       │   │
│  └─────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────┘
                       │ Async loop (asyncio)
                       ▼
┌─────────────────────────────────────────────────────────┐
│            Composition + Synthesis Engine               │
│  ┌─────────────────────────────────────────────────┐   │
│  │ ChordGenerator (Markov chain)                   │   │
│  │ • Generates progressions (8-16 bars)            │   │
│  │ • Modal context (Aeolian, Dorian, etc.)         │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │ MelodyGenerator (constraint-based)              │   │
│  │ • 70% chord tones, 25% scale, 5% chromatic      │   │
│  │ • Note probability, velocity randomization      │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │ FluidSynth Synthesis (sample-based)             │   │
│  │ • Piano voice (SoundFont preset 0)              │   │
│  │ • Pad voice (SoundFont preset 88)               │   │
│  │ • Renders to 44.1kHz stereo PCM                 │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Ring Buffer (thread-safe)                       │   │
│  │ • 2-second capacity (10-20 chunks)              │   │
│  │ • Back-pressure flow control                    │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Technology Stack

**Server**:
- **Language**: Python 3.12+ (required for performance)
- **Framework**: FastAPI 0.127+ (async, WebSocket support)
- **Server**: uvicorn 0.40+ (ASGI with WebSocket)
- **Synthesis**: pyfluidsynth 1.3.2+ (sample-based synthesis)
- **Audio Processing**: numpy 1.26+, scipy 1.11+ (DSP operations)
- **GPU Acceleration**: PyTorch 2.5+ (Metal/CUDA for effects)
- **Package Manager**: uv (fast, reproducible dependency management)

**Client**:
- **Audio**: Web Audio API (AudioContext, AudioWorklet)
- **Network**: WebSocket API (native browser)
- **UI**: Vanilla HTML/CSS/JS (no frameworks for simplicity)
- **Storage**: localStorage (settings persistence)

**Infrastructure** (MVP: Local only):
- **Deployment**: Local development server (uvicorn)
- **SoundFonts**: Local directory (`./soundfonts/`)
- **Configuration**: `.env` file

### Key Design Decisions

**1. Sample-Based Synthesis (FluidSynth) vs. Oscillators**

**Decision**: Use FluidSynth with SoundFonts for melodic voices

**Rationale**:
- Realistic piano/pad timbres essential for ambient music quality
- SoundFonts provide professional sound with minimal CPU overhead
- FluidSynth mature, well-tested, cross-platform
- Ambient music requires texture and timbral richness (not achievable with basic oscillators)

**Trade-offs**:
- ✅ Pro: Immediate professional sound quality
- ✅ Pro: No machine learning training required
- ❌ Con: CPU-based (no GPU acceleration)
- ❌ Con: SoundFont licensing considerations

**2. Real-Time Streaming vs. Pre-Rendered Files**

**Decision**: Stream in real-time via WebSocket

**Rationale**:
- True infinite generation (no storage limits)
- Parameter changes apply immediately
- Supports future cloud deployment
- Lower client storage requirements

**Trade-offs**:
- ✅ Pro: Infinite, non-repeating music
- ✅ Pro: Real-time adaptability
- ❌ Con: Network dependency
- ❌ Con: More complex architecture

**3. Web-Based UI vs. Native App**

**Decision**: Browser-based interface (no native apps in MVP)

**Rationale**:
- Cross-platform without platform-specific builds
- Faster iteration during MVP validation
- Lower development cost
- Web Audio API sufficient for quality playback

**Trade-offs**:
- ✅ Pro: Universal access (no install)
- ✅ Pro: Rapid updates
- ❌ Con: Browser compatibility constraints
- ❌ Con: No offline mode

---

## Success Metrics

### Primary Metrics (Business Value)

**Engagement**:
- **Average Session Duration**: Target 45+ minutes (indicates value)
- **Daily Active Users (DAU)**: Track adoption growth
- **Retention (D7/D30)**: 40%+ D7, 20%+ D30 (sticky product)

**Quality**:
- **User-Reported Quality**: 80%+ rate audio as "pleasant and smooth"
- **Technical Issues**: <5% sessions with disconnections/errors
- **Net Promoter Score (NPS)**: Target 40+ (product-market fit indicator)

### Secondary Metrics (Technical Performance)

**Latency**:
- **Time-to-First-Audio**: <2 seconds (95th percentile)
- **End-to-End Latency**: <800ms (95th percentile)
- **Synthesis Latency**: <100ms per phrase (average)

**Reliability**:
- **Uptime**: 99%+ (measured over 7 days)
- **Buffer Underruns**: <1 per 30-minute session
- **Auto-Recovery Success**: 90%+ reconnections succeed

**Resource Usage**:
- **Memory**: <500MB per server instance
- **CPU**: <40% average (on M4 Mac or equivalent)
- **Network Bandwidth**: ~250 kbps per client (base64 PCM)

### Validation Criteria (Go/No-Go for Post-MVP)

**Must Achieve**:
1. ✅ Average session >30 minutes (users find value)
2. ✅ 75%+ users rate music "musically coherent"
3. ✅ <5% technical error rate (reliability proven)
4. ✅ <800ms latency (real-time experience validated)

**If Not Achieved**:
- <30 min sessions → Re-evaluate musical quality or UX friction
- <75% musicality → Improve composition algorithms
- >5% errors → Fix streaming reliability
- >800ms latency → Optimize synthesis or network

---

## Dependencies & Risks

### Critical Dependencies

**External Software**:
- **FluidSynth**: Native library required on host OS
  - **macOS**: `brew install fluidsynth`
  - **Linux**: `apt-get install fluidsynth libfluidsynth-dev`
  - **Windows**: `vcpkg install fluidsynth` or pre-built binaries
- **SoundFont Files** (SF2): Free or licensed SoundFonts
  - Recommended: Salamander Grand Piano (~200MB), Arachno pads (~150MB)
  - Licensing must allow commercial use (verify before production)

**System Requirements**:
- **Python 3.12+**: Required for performance features
- **GPU (Optional)**: Metal (M1+), CUDA (NVIDIA) for effects acceleration
- **Memory**: 1GB+ RAM available
- **Network**: 500 kbps+ upload bandwidth per client (server)

### Risks & Mitigation

**High Risk**:

**R1: FluidSynth Latency Bottleneck**
- **Risk**: CPU synthesis may not meet <100ms target
- **Impact**: Choppy playback, poor user experience
- **Likelihood**: Medium (FluidSynth is fast, but SoundFont size matters)
- **Mitigation**:
  - Benchmark early with target SoundFonts
  - Optimize FluidSynth settings (disable reverb/chorus, limit polyphony)
  - Use smaller SoundFonts if needed
  - **Fallback**: Keep PyTorch oscillators as backup synthesis method
- **Owner**: Engineering Lead

**R2: SoundFont Quality/Licensing**
- **Risk**: Free SoundFonts may sound poor or have restrictive licenses
- **Impact**: Poor audio quality or legal issues in production
- **Likelihood**: Medium
- **Mitigation**:
  - Test top 3 free SoundFonts (Salamander, FluidR3, Arachno)
  - Review licenses thoroughly (CC0, Public Domain, or permissive)
  - Budget $100-300 for professional SoundFonts if needed
  - **Fallback**: Use FluidR3_GM (General MIDI set, well-established)
- **Owner**: Product Manager + Legal Review

**Medium Risk**:

**R3: Browser Compatibility**
- **Risk**: Web Audio API/WebSocket inconsistencies across browsers
- **Impact**: Some users unable to use product
- **Likelihood**: Low (APIs well-supported in Chrome/Edge/Safari)
- **Mitigation**:
  - Test on Chrome, Edge, Safari, Firefox
  - Display clear error messages for unsupported browsers
  - Document minimum browser versions
  - **Fallback**: Support Chrome/Edge only (80%+ desktop market share)
- **Owner**: Frontend Engineer

**R4: Network Unreliability**
- **Risk**: Poor network causes frequent disconnections
- **Impact**: Frustrating user experience
- **Likelihood**: Medium (especially on mobile networks)
- **Mitigation**:
  - Implement exponential backoff reconnection
  - Increase client buffer size (300-500ms)
  - Display clear buffering indicators
  - **Fallback**: Recommend wired/WiFi connections for MVP
- **Owner**: Backend Engineer

**Low Risk**:

**R5: GPU Unavailability**
- **Risk**: Users without GPU experience slower synthesis
- **Impact**: Latency may approach 200-300ms (still acceptable)
- **Likelihood**: Low (most modern Macs/PCs have GPU)
- **Mitigation**:
  - CPU fallback built-in
  - Notify users if GPU not detected
  - Optimize CPU path
- **Owner**: Engineering Lead

---

## Open Questions

### Critical (Require Decision Before Implementation)

**Q1: Which SoundFonts should we use?**
- **Context**: Hundreds of free/paid options with varying quality/size
- **Options**:
  - **Option A**: Salamander Grand Piano (200MB, high quality, free)
  - **Option B**: FluidR3_GM (140MB, good quality, General MIDI)
  - **Option C**: Budget for professional SF2s ($100-300)
- **Decision Needed By**: Before Phase 1 implementation start
- **Decision Maker**: Product Manager + Audio Engineer
- **Action**: Benchmark top 3 candidates, select based on quality/latency trade-off

**Q2: Should percussion be in MVP or post-MVP?**
- **Context**: Ambient music context doc emphasizes minimal rhythm, but some percussion adds texture
- **Options**:
  - **Option A**: Include sparse kicks/swells in MVP (more complete ambient sound)
  - **Option B**: Exclude entirely (focus on melodic/harmonic quality first)
- **Decision Needed By**: Before spec finalization
- **Decision Maker**: Product Manager
- **Recommendation**: Exclude from MVP - focus on melodic quality, add percussion in Phase 2 if validated

**Q3: How much should effects (reverb/delay) be prioritized?**
- **Context**: Ambient music relies heavily on reverb for "space" and immersion
- **Options**:
  - **Option A**: Minimal reverb via FluidSynth (low CPU, basic quality)
  - **Option B**: High-quality reverb via pedalboard (higher CPU, professional sound)
  - **Option C**: No reverb in MVP (ship fast, add later)
- **Decision Needed By**: Phase 1 start
- **Decision Maker**: Product Manager + Audio Engineer
- **Recommendation**: Option A - use FluidSynth reverb minimally, upgrade to pedalboard in Phase 2

### Important (Can Be Resolved During Development)

**Q4: What's the ideal default intensity/BPM?**
- **Context**: "Focus" preset should feel optimal out-of-box
- **Action**: User testing with 5-10 target users, iterate based on feedback

**Q5: Should we support Firefox?**
- **Context**: Firefox has 3-5% desktop market share
- **Action**: Test after Chrome/Edge/Safari support proven, add if minimal effort

**Q6: Should settings sync across devices?**
- **Context**: Users may want same settings on laptop + desktop
- **Action**: Post-MVP - requires user accounts, out of scope for MVP

---

## Future Considerations (Post-MVP)

### Phase 2: Enhanced Musicality (Months 2-3)

**Features**:
- Sparse percussion/rhythm generation (kicks, swells, granular textures)
- High-quality reverb/delay (pedalboard integration)
- Transformer-based melody generation (DistilGPT-2 for more expressive melodies)
- Additional modes (Mixolydian, Locrian)
- Custom key signatures (F, Bb, Ab, etc.)

**Goal**: Deepen musical sophistication for active listeners

### Phase 3: Scalability & Cloud (Months 4-6)

**Features**:
- Multi-client optimization (50+ concurrent users per instance)
- Cloud deployment (Docker, Kubernetes, AWS/GCP)
- Opus compression (reduce bandwidth to ~64 kbps)
- User accounts and session persistence
- Analytics dashboard (Prometheus/Grafana)

**Goal**: Enable paid cloud service offering

### Phase 4: Personalization & Export (Months 7-9)

**Features**:
- User-created presets (save custom configurations)
- MIDI export (download generated composition)
- Offline rendering (generate full-length MP3/FLAC files)
- Machine learning personalization (adapt to user listening patterns)
- Mobile apps (iOS, Android)

**Goal**: Differentiate through customization and cross-platform access

### Phase 5: Community & Ecosystem (Months 10-12)

**Features**:
- Custom soundscape creation (user-uploaded samples)
- Preset sharing marketplace
- API for developers (integrate Auralis into other apps)
- Open-source community contributions
- Live collaboration (multiple users shape soundscape together)

**Goal**: Build ecosystem and community-driven growth

---

## Appendix A: Ambient Music Principles (Design Reference)

The following principles from the ambient music context document guide our composition algorithms and synthesis choices:

### Core Principles

1. **Texture Over Melody**: Prioritize timbral quality (grain, organic sounds) over catchy hooks
2. **Slow Evolution**: Changes occur at minute-scale, not second-scale (30-90 second harmonic shifts)
3. **Minimal Rhythm**: No beats, subtle temporal variation only
4. **Harmonic Ambiguity**: Modal, drone-based, quartal harmonies (avoid V-I progressions)
5. **Generous Space**: Silence as compositional element, not pause
6. **Psychoacoustic Effects**: Reverb as instrument, delay systems, spatial depth
7. **Generative Patterns**: Note probability, coprime loops, modulation
8. **Emotional Impact**: Frequency selection and intervals create psychological response

### Implementation Mapping

| Principle | Auralis Implementation |
|-----------|------------------------|
| **Texture** | Sample-based synthesis (FluidSynth), pad timbres |
| **Slow Evolution** | Harmonic changes every 30-90 seconds |
| **Minimal Rhythm** | No percussion in MVP, sparse if added later |
| **Harmonic Ambiguity** | Modal contexts (Aeolian, Dorian), quartal voicings |
| **Space** | Low note density (intensity controls sparseness) |
| **Reverb** | FluidSynth reverb (basic in MVP, upgrade Phase 2) |
| **Generative Patterns** | Note probability (50-80%), velocity randomization |
| **Emotion** | Frequency ranges (sub-bass 60-250Hz, mids 300-2kHz) |

### Parameter Ranges (Technical Constraints)

| Parameter | Minimum | Typical | Maximum | Notes |
|-----------|---------|---------|---------|-------|
| **BPM** | 40 | 60 | 90 | Ambient range (vs. music 100-140) |
| **Attack Time** | 50ms | 200ms | 500ms | Soft edges, no plucky sounds |
| **Reverb Decay** | 3s | 6s | 10s+ | Creates spaciousness |
| **Note Probability** | 10% | 50% | 100% | Prevents mechanical repetition |
| **Harmonic Change** | 30s | 60s | 120s | Slow chord progressions |
| **Velocity Range** | 20-100 | 30-90 | 10-120 | Humanization without extremes |

---

## Appendix B: Competitive Analysis

### Direct Competitors

**Brain.fm** (Generative Focus Music)
- **Strengths**: Neuroscience-backed, adaptive to activity type
- **Weaknesses**: Limited timbral variety, not artistically rich
- **Differentiation**: Auralis prioritizes musicality and aesthetic quality

**Endel** (Adaptive Soundscapes)
- **Strengths**: Cross-platform, personalized to time/weather/heart rate
- **Weaknesses**: Repetitive, lacks harmonic sophistication
- **Differentiation**: Auralis uses advanced music theory (modes, progressions)

**Generative.fm** (Open-Source Generative Music)
- **Strengths**: Free, open-source, community-driven
- **Weaknesses**: Complex setup, inconsistent quality, niche audience
- **Differentiation**: Auralis aims for polish and ease-of-use

### Indirect Competitors

**Spotify/YouTube Ambient Playlists**
- **Strengths**: Massive content library, integrated into existing workflows
- **Weaknesses**: Static, repetitive, ads (free tier)
- **Differentiation**: Infinite, non-repeating, no ads

**Calm/Headspace** (Meditation Apps)
- **Strengths**: Guided content, habit-building features
- **Weaknesses**: Meditation-focused, not work/productivity oriented
- **Differentiation**: Auralis designed for multiple use cases (focus, sleep, meditation)

### Market Positioning

**Auralis Unique Value**:
1. **Musically Sophisticated**: Ambient music theory deeply integrated
2. **High Quality**: Sample-based synthesis, not basic oscillators
3. **Developer-Friendly**: Open architecture, local-first, extensible
4. **Premium Simplicity**: Minimal UI, auto-play, no config burden

**Target Niche**: Serious ambient music listeners, knowledge workers, meditation practitioners willing to run local software or pay for quality streaming

---

## Appendix C: Technical Specifications Summary

### Audio Format
- **Sample Rate**: 44,100 Hz (CD quality)
- **Bit Depth**: 16-bit PCM
- **Channels**: Stereo (2 channels)
- **Chunk Size**: 100ms (4,410 samples per channel)
- **Chunk Bytes**: 4,410 × 2 × 2 = ~17.6 kB raw, ~23.5 kB base64

### Network Protocol
- **Transport**: WebSocket (TCP-based)
- **Audio Encoding**: Base64-encoded PCM
- **Control Messages**: JSON
- **Bandwidth**: ~250 kbps per client (base64 overhead)

### Performance Targets
- **Time-to-First-Audio**: <2 seconds
- **Synthesis Latency**: <100ms per phrase
- **End-to-End Latency**: <800ms (target: 500ms)
- **Buffer Capacity**: 2 seconds (server), 300-500ms (client)
- **Real-Time Factor**: >10× (16s audio in <1.6s)

### System Requirements

**Server**:
- **OS**: macOS 12+, Ubuntu 20.04+, Windows 10+
- **CPU**: 2+ cores, 2.5 GHz+ (M1/M2/M4 or Intel i5+)
- **RAM**: 1GB+ available
- **GPU**: Optional (Metal/CUDA for effects)
- **Disk**: 500MB+ (SoundFonts)

**Client**:
- **Browser**: Chrome 90+, Edge 90+, Safari 14+
- **Network**: 500 kbps+ stable connection
- **Device**: Desktop or tablet (mobile post-MVP)

---

## Document Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | Dec 26, 2024 | Product Team | Initial draft (MVP scope) |
| 2.0 | Dec 28, 2024 | Product Team | FluidSynth integration, ambient music principles, full MVP definition |

---

**Document Status**: ✅ Ready for Implementation Planning
**Next Steps**: Generate implementation plan (`/speckit.plan`) and task breakdown (`/speckit.tasks`)
**Approval Required**: Product Manager, Engineering Lead, Stakeholders

---

**End of Product Requirements Document**
