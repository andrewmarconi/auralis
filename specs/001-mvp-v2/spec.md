# Feature Specification: Auralis MVP v2.0 - Real-Time Generative Ambient Music Streaming Engine

**Feature Branch**: `001-mvp-v2`
**Created**: 2025-12-28
**Status**: Draft
**Input**: User description: "Complete MVP implementation of Auralis real-time generative ambient music streaming engine with FluidSynth synthesis, Markov chord progressions, constraint-based melodies, WebSocket streaming, and browser-based controls"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Instant Ambient Playback (Priority: P1)

A focused professional opens Auralis in their browser and immediately hears evolving ambient music that never repeats, allowing them to enter a flow state for deep work without managing playlists or dealing with interruptions.

**Why this priority**: This is the core value proposition - delivering immediate, infinite ambient music without configuration. Without this, there is no product.

**Independent Test**: Can be fully tested by navigating to the application URL and verifying audio plays within 2 seconds, streams continuously for 30+ minutes without glitches, and exhibits gradual musical evolution without repetition.

**Acceptance Scenarios**:

1. **Given** user navigates to application URL, **When** page loads, **Then** ambient music begins playing within 2 seconds without manual intervention
2. **Given** music is playing, **When** user listens for 30+ minutes, **Then** music evolves gradually with new chord progressions and melodies without repeating patterns
3. **Given** music is streaming, **When** network conditions are stable, **Then** no audio dropouts, glitches, or jarring transitions occur
4. **Given** user opens browser, **When** application starts, **Then** music sounds harmonically coherent with smooth phrase transitions
5. **Given** music is generating, **When** harmonic changes occur, **Then** transitions happen at 30-90 second intervals with modal consistency

---

### User Story 2 - Musical Coherence and Quality (Priority: P1)

A contemplative listener using Auralis for meditation experiences harmonically pleasing, intentional-sounding music that supports rather than disrupts their practice, with no dissonant surprises or mechanical repetition.

**Why this priority**: Musical quality is essential for the ambient music use case. Poor musicality makes the product unusable for meditation, focus, and sleep applications.

**Independent Test**: Can be tested by analyzing generated MIDI/audio for harmonic consistency (95%+ notes within chord/scale constraints), conducting listening tests with target users (75%+ rate as "harmonically pleasing"), and measuring average session duration (45+ minutes indicates engagement).

**Acceptance Scenarios**:

1. **Given** system generates chord progressions, **When** analyzing harmonic movement, **Then** progressions follow ambient-appropriate patterns (modal, circular, slow) without V-I cadences
2. **Given** system generates melodies, **When** analyzing note choices, **Then** 70% are chord tones, 25% are scale notes, 5% are chromatic passing tones
3. **Given** melodies play over chords, **When** listening to phrases, **Then** no dissonant or jarring note clashes occur
4. **Given** music plays continuously, **When** harmonic changes occur, **Then** transitions are smooth without abrupt key changes or hard cuts
5. **Given** user listens actively, **When** evaluating musical quality, **Then** output aligns with ambient music principles (slow, spacious, textural, emotionally supportive)

---

### User Story 3 - Personalized Control (Priority: P2)

A user with specific preferences adjusts musical parameters (key, mode, intensity, tempo) or selects presets (Focus, Meditation, Sleep, Bright) to match their current mood or task, with changes applying smoothly within 5 seconds.

**Why this priority**: Customization enhances user satisfaction but is secondary to core playback. Users can derive value from default settings, but controls enable broader use cases.

**Independent Test**: Can be tested by adjusting each control parameter, verifying changes apply within 5 seconds at phrase boundaries, and confirming settings persist across browser refresh via localStorage.

**Acceptance Scenarios**:

1. **Given** web interface is displayed, **When** user views controls, **Then** key selector (C/D/E/G/A major/minor), mode selector (Aeolian/Dorian/Lydian/Phrygian), intensity slider (0.0-1.0), and BPM slider (40-90) are visible
2. **Given** user adjusts key to G minor, **When** current phrase completes, **Then** next phrase generates in G minor within 5 seconds
3. **Given** user moves intensity slider to 0.8, **When** change is applied, **Then** note density increases noticeably while maintaining musical coherence
4. **Given** user adjusts BPM to 50, **When** change is applied, **Then** harmonic rhythm slows proportionally
5. **Given** user refreshes browser, **When** page reloads, **Then** previously selected settings are restored from localStorage
6. **Given** user clicks "Focus" preset, **When** preset activates, **Then** system sets Dorian mode, medium intensity (0.5), and 60 BPM instantly
7. **Given** user clicks "Sleep" preset, **When** preset activates, **Then** system sets Phrygian mode, very low intensity (0.2), and 40 BPM

---

### User Story 4 - Low-Latency Streaming Performance (Priority: P1)

A user experiences immediate, responsive audio playback with end-to-end latency under 800ms, enabling real-time parameter changes and smooth streaming without buffering delays.

**Why this priority**: Real-time performance is a core technical requirement. High latency breaks the user experience and makes control changes feel sluggish.

**Independent Test**: Can be tested by measuring end-to-end latency from generation to playback (target <800ms, 95th percentile), synthesis performance (<100ms per phrase), and buffer health (>98% of chunks delivered on time).

**Acceptance Scenarios**:

1. **Given** system generates audio phrase, **When** measuring time from generation to speaker output, **Then** end-to-end latency is <800ms (95th percentile)
2. **Given** system synthesizes 8-bar phrase at 60 BPM, **When** measuring synthesis time, **Then** completion occurs in <100ms
3. **Given** system streams audio chunks, **When** monitoring network delivery, **Then** 100ms chunks are delivered consistently with <50ms timing variance
4. **Given** GPU is available (Metal/CUDA), **When** system initializes, **Then** GPU acceleration is used for synthesis
5. **Given** system runs on CPU-only hardware, **When** GPU is unavailable, **Then** system falls back to CPU synthesis with notification to user
6. **Given** buffer is healthy, **When** monitoring playback, **Then** no buffer underruns or audio dropouts occur

---

### User Story 5 - Graceful Error Recovery (Priority: P2)

A user experiencing technical issues (network disconnect, GPU unavailable, buffer underflow) receives clear feedback and sees automatic recovery, allowing them to continue listening without manual intervention.

**Why this priority**: Error handling improves reliability but is secondary to core functionality. Users tolerate occasional issues if recovery is smooth.

**Independent Test**: Can be tested by simulating network disconnects (verify auto-reconnect), disabling GPU (verify CPU fallback), and inducing buffer underflow (verify buffering indicator and smooth resume).

**Acceptance Scenarios**:

1. **Given** WebSocket connection drops, **When** disconnect is detected, **Then** system auto-reconnects with exponential backoff
2. **Given** GPU is unavailable at startup, **When** system initializes, **Then** fallback to CPU synthesis occurs with notification displayed to user
3. **Given** network congestion causes buffer underflow, **When** buffer depletes, **Then** buffering indicator is displayed and playback resumes smoothly when buffer refills
4. **Given** user opens in unsupported browser, **When** application loads, **Then** error message with supported browser list is shown
5. **Given** errors occur during session, **When** logged, **Then** metrics are sent to `/api/metrics` endpoint for debugging
6. **Given** auto-reconnect is attempted, **When** reconnection succeeds, **Then** recovery completes within 10 seconds in 90%+ of cases

---

### Edge Cases

- **What happens when user's network bandwidth drops below 250 kbps?** System should detect slow delivery, display buffering indicator, and attempt to continue playback from buffer. If sustained low bandwidth, user should see warning about connection quality.

- **How does system handle GPU memory exhaustion?** If GPU runs out of memory during synthesis, system should gracefully fall back to CPU synthesis and log error to metrics endpoint.

- **What happens when multiple browser tabs connect simultaneously from same client?** Each tab should maintain independent WebSocket connection and audio stream. System should support 10+ concurrent connections per server instance.

- **How does system handle extremely slow synthesis (>500ms per phrase)?** If synthesis consistently exceeds latency targets, system should log performance warning to metrics, potentially reduce polyphony or complexity, and notify user if degradation is severe.

- **What happens when user adjusts controls rapidly (multiple changes within 1 second)?** System should debounce control changes, applying only the most recent value when phrase boundary is reached, preventing parameter thrashing.

- **How does system handle SoundFont loading failure?** If FluidSynth cannot load SoundFont files at startup, system should fail gracefully with clear error message indicating missing or corrupted SoundFont, preventing silent failure.

- **What happens during long-running sessions (8+ hours)?** System must not leak memory. Memory usage should remain stable (<500MB) over extended sessions, with metrics monitoring tracking memory growth.

## Requirements *(mandatory)*

### Functional Requirements

#### Composition & Generation

- **FR-001**: System MUST generate chord progressions using Markov chain algorithm (bigram, order 2) considering 1 previous chord
- **FR-002**: System MUST support modal contexts: Aeolian, Dorian, Lydian, Phrygian
- **FR-003**: System MUST generate melodies with constraint-based note selection: 70% chord tones, 25% scale notes, 5% chromatic passing tones
- **FR-004**: System MUST randomize note velocity within 20-100 range for humanization
- **FR-005**: System MUST apply note probability (50-80%) to prevent mechanical repetition
- **FR-006**: System MUST ensure harmonic transitions occur at 30-90 second intervals

#### Audio Synthesis

- **FR-007**: System MUST render MIDI-like note events to 44.1kHz stereo audio using FluidSynth sample-based synthesis
- **FR-008**: System MUST load SoundFont presets: Piano (Acoustic Grand Piano, preset 0) and Pad (Warm Pad, preset 88-90)
- **FR-009**: System MUST apply soft clipping/limiting to prevent audio distortion
- **FR-010**: System MUST mix multiple voices with configurable balance (piano 50%, pad 40%, percussion 30% if added)
- **FR-011**: System MUST render complete phrases (8-16 bars) upfront, not streaming sample-by-sample

#### Real-Time Streaming

- **FR-012**: System MUST stream audio in 100ms chunks (4,410 samples per channel at 44.1kHz)
- **FR-013**: System MUST encode PCM as base64 for WebSocket transport
- **FR-014**: System MUST implement thread-safe ring buffer with 2-second capacity (10-20 chunks)
- **FR-015**: System MUST provide back-pressure mechanism, pausing generation if buffer depth <2 chunks
- **FR-016**: System MUST accept control messages via WebSocket in JSON format for parameter updates (key, mode, intensity, BPM)
- **FR-017**: System MUST send connection status events (connected, buffering, error) to clients

#### Browser-Based User Interface

- **FR-018**: System MUST auto-play audio on page load using Web Audio API without user interaction
- **FR-019**: System MUST display controls for: key selector (C/D/E/G/A major/minor), mode selector (Aeolian/Dorian/Lydian/Phrygian), intensity slider (0.0-1.0, default 0.5), BPM slider (40-90, default 60)
- **FR-020**: System MUST provide preset buttons: Focus (Dorian, 0.5 intensity, 60 BPM), Meditation (Aeolian, 0.3 intensity, 50 BPM), Sleep (Phrygian, 0.2 intensity, 40 BPM), Bright (Lydian, 0.6 intensity, 70 BPM)
- **FR-021**: System MUST display connection status indicator (green=connected, yellow=buffering, red=disconnected)
- **FR-022**: System MUST persist user settings to localStorage and restore on page load

#### Performance Monitoring

- **FR-023**: System MUST expose `/api/status` endpoint returning: server uptime, active connections count, buffer depth, current GPU/CPU device
- **FR-024**: System MUST expose `/api/metrics` endpoint returning: synthesis latency (avg, p50, p95, p99), network latency per client, buffer underrun/overflow events, memory usage
- **FR-025**: System MUST update metrics every 1 second

### Non-Functional Requirements

#### Performance

- **NFR-001**: Phrase generation MUST complete in <50ms (CPU)
- **NFR-002**: Synthesis latency MUST be <100ms per phrase (8 bars @ 60 BPM)
- **NFR-003**: End-to-end latency MUST be <800ms (generation → synthesis → network → playback), target 500ms
- **NFR-004**: Audio output MUST remain within range [-1.0, 1.0] with no clipping
- **NFR-005**: System MUST support polyphony up to 32 simultaneous notes

#### Reliability

- **NFR-006**: System MUST NOT leak memory over 8+ hour sessions, maintaining <500MB memory footprint including SoundFonts
- **NFR-007**: Composition output MUST be deterministic given same random seed (for debugging reproducibility)
- **NFR-008**: System MUST support 10+ concurrent WebSocket clients without audio glitches
- **NFR-009**: Auto-reconnect MUST succeed in 90%+ of disconnect cases
- **NFR-010**: Buffer health MUST achieve >98% of chunks delivered on time

#### Compatibility

- **NFR-011**: System MUST support browsers: Chrome 90+, Edge 90+, Safari 14+
- **NFR-012**: System MUST run on: macOS 12+, Ubuntu 20.04+, Windows 10+
- **NFR-013**: Page load to first audio MUST be <2 seconds on 10 Mbps connection
- **NFR-014**: UI MUST be responsive on desktop and tablet (mobile post-MVP)
- **NFR-015**: UI MUST meet WCAG 2.1 AA accessibility standards (keyboard navigation, ARIA labels)

#### Quality

- **NFR-016**: Metrics endpoint response time MUST be <50ms
- **NFR-017**: Metrics storage overhead MUST be <10MB RAM
- **NFR-018**: Harmonic analysis MUST show 95%+ note adherence to constraints (70/25/5% distribution)

### Key Entities *(mandatory)*

#### Composition Layer

- **ChordProgression**: Represents generated harmonic structure as list of (onset_time, root_pitch, chord_type). Onset time is in samples, root pitch is MIDI note number, chord_type is string (e.g., "major", "minor", "sus4"). Generated by Markov chain considering modal context.

- **MelodyPhrase**: Represents generated melodic content as list of (onset_time, pitch, velocity, duration). Onset time in samples, pitch in MIDI, velocity in range 20-100, duration in samples. Generated using constraint-based selection within current chord/scale.

- **MusicalContext**: Represents current generative parameters including key (root MIDI pitch), mode (scale degree pattern), BPM (tempo), intensity (note density multiplier). Updated via WebSocket control messages.

#### Synthesis Layer

- **FluidSynthVoice**: Wrapper around FluidSynth sample-based synthesis engine. Loads SoundFont (.sf2) files and renders note events to PCM audio using specified presets (piano, pad).

- **SoundFontPreset**: Represents loaded SoundFont file path and preset number. Maps instrument names (Piano, Pad) to SF2 file locations and General MIDI preset indices.

- **AudioBuffer**: NumPy array with shape (2, num_samples) in float32 format, representing stereo PCM audio. Generated by FluidSynth, chunked for streaming.

#### Streaming Layer

- **RingBuffer**: Thread-safe circular buffer for audio chunks. Pre-allocated NumPy array with atomic read/write cursors. Capacity of 10-20 chunks (1-2 seconds). Provides back-pressure when depth <2 chunks.

- **WebSocketConnection**: Per-client connection state including socket reference, client ID, buffer health metrics, current parameter settings. Managed by FastAPI WebSocket endpoint.

- **AudioChunk**: 100ms PCM data (4,410 samples × 2 channels × 2 bytes = ~17.6kB raw, ~23.5kB base64-encoded) plus timing metadata (chunk sequence number, generation timestamp).

#### Client Layer

- **AudioContext**: Web Audio API playback engine managing sample rate, buffer size, and AudioWorklet processor. Decodes base64 PCM chunks and queues for playback.

- **AdaptiveBuffer**: Client-side ring buffer targeting 300-500ms latency. Monitors jitter and adjusts buffer size to prevent underruns while minimizing latency.

- **ControlState**: Current user settings including key, mode, intensity, BPM. Persisted to localStorage, sent to server via WebSocket JSON messages.

#### Monitoring Layer

- **PerformanceMetrics**: Latency histograms (synthesis, network, end-to-end), event counters (buffer underrun/overflow, disconnects), memory usage tracking. Exposed via `/api/metrics`.

- **SystemStatus**: Current operational state including server uptime, active connection count, buffer depth, GPU/CPU device in use. Exposed via `/api/status`.

## Success Criteria *(mandatory)*

### Measurable Outcomes

#### Core User Experience

- **SC-001**: Time-to-first-audio is <2 seconds (95th percentile) from page load to audible playback
- **SC-002**: Continuous playback sustains for 30+ minutes without interruption (0 dropouts/glitches in stable network conditions)
- **SC-003**: Music evolves gradually with new harmonic content every 30-90 seconds without perceivable repetition over 60+ minute sessions
- **SC-004**: 80%+ of users rate audio quality as "smooth and pleasant" in listening tests
- **SC-005**: Average session duration is 45+ minutes, indicating sustained engagement

#### Musical Quality

- **SC-006**: Harmonic consistency shows 95%+ of generated notes within chord/scale constraints (70% chord tones, 25% scale, 5% chromatic)
- **SC-007**: Phrase transitions exhibit zero hard cuts or abrupt key changes in 30-minute listening sessions
- **SC-008**: 75%+ of users rate music as "harmonically pleasing" and "musically coherent" in surveys
- **SC-009**: Generated progressions follow ambient-appropriate patterns (modal, circular, slow) with no V-I cadences detected

#### Technical Performance

- **SC-010**: End-to-end latency (generation → synthesis → network → playback) is <800ms (95th percentile), target 500ms
- **SC-011**: Synthesis latency is <100ms per phrase (8 bars @ 60 BPM) on average
- **SC-012**: 95%+ of audio chunks delivered within 50ms variance of target 100ms timing
- **SC-013**: GPU acceleration (when available) reduces synthesis latency by 50% compared to CPU-only mode
- **SC-014**: Buffer health maintains >98% of chunks delivered on time without underruns

#### Reliability

- **SC-015**: System uptime is 99%+ over 7-day measurement period
- **SC-016**: Buffer underruns occur <1 time per 30-minute session on stable network
- **SC-017**: Auto-recovery succeeds in 90%+ of WebSocket disconnect events within 10 seconds
- **SC-018**: Memory usage remains <500MB over 8+ hour continuous operation with <10MB growth
- **SC-019**: System handles 10+ concurrent clients without audio degradation

#### User Control & Customization

- **SC-020**: Parameter changes (key, mode, intensity, BPM) apply within 5 seconds at next phrase boundary
- **SC-021**: 40%+ of users adjust at least one parameter during first session
- **SC-022**: 60%+ of users utilize presets instead of manual controls
- **SC-023**: 85%+ of users report presets match intended use case (Focus/Meditation/Sleep/Bright)
- **SC-024**: 80%+ of users have settings restored correctly on browser refresh (localStorage persistence)

#### Error Handling

- **SC-025**: Auto-reconnect success rate is >90% for WebSocket disconnects
- **SC-026**: CPU fallback activates successfully in 100% of cases when GPU unavailable
- **SC-027**: Error messages are understood by 85%+ of users (clarity rating)
- **SC-028**: Technical issue rate is <5% of sessions (disconnections, glitches, errors)

#### Browser Compatibility

- **SC-029**: Application functions correctly in Chrome 90+, Edge 90+, Safari 14+ (verified via testing)
- **SC-030**: Page load completes in <2 seconds on 10 Mbps connection

## Assumptions *(mandatory)*

### Technical Assumptions

1. **FluidSynth Availability**: FluidSynth native library is installed on host OS (`brew install fluidsynth` on macOS, `apt-get install fluidsynth libfluidsynth-dev` on Linux, or pre-built binaries on Windows). Installation is part of setup instructions, not automated by application.

2. **SoundFont Licensing**: Free SoundFonts (Salamander Grand Piano, FluidR3_GM, or Arachno) are available and licensed for use. Licensing has been verified to allow commercial use or project will use CC0/Public Domain fonts only.

3. **GPU Acceleration**: When GPU is available (Metal on M1/M2/M4, CUDA on NVIDIA), PyTorch automatically detects and utilizes it. CPU-only fallback is acceptable but not optimal for real-time performance.

4. **Network Conditions**: Users have stable network connections with >500 kbps upload bandwidth (server perspective) and <100ms latency. Adaptive buffering handles temporary degradation, but sustained poor conditions may cause dropouts.

5. **Browser Compatibility**: Modern browsers (Chrome 90+, Edge 90+, Safari 14+) support Web Audio API and WebSockets without polyfills. Older browsers are not supported in MVP.

6. **Python Environment**: Python 3.12+ is installed and uv package manager is used for all dependency management. No manual pip or venv usage.

7. **Single-Server Deployment**: MVP targets local development or single-server cloud deployment. Multi-server orchestration (load balancing, session persistence) is out of scope.

8. **Audio Format**: Base64-encoded PCM is acceptable for WebSocket transport despite ~33% overhead. Opus compression is deferred to post-MVP for bandwidth optimization.

### User Assumptions

1. **Default Settings**: Default musical parameters (C Aeolian, 60 BPM, 0.5 intensity) are suitable for majority of users without customization. Presets cover primary use cases (Focus, Meditation, Sleep, Bright).

2. **Auto-Play Acceptance**: Users expect and accept auto-play on page load, consistent with ambient music use case. No user interaction required to start playback.

3. **Desktop/Tablet Primary**: MVP targets desktop and tablet users. Mobile support (iOS, Android) is deferred to post-MVP despite potential user demand.

4. **Session Duration**: Average listening sessions are 30-60 minutes. Shorter sessions (5-10 minutes) are supported but not optimized for.

5. **Musical Knowledge**: Users selecting advanced controls (key, mode) have basic music theory knowledge. Users without this knowledge use presets instead.

### Deployment Assumptions

1. **Local-First MVP**: MVP is deployed locally via `uvicorn` for development/testing. Cloud deployment (Docker, Kubernetes, AWS/GCP) is post-MVP.

2. **SoundFont Storage**: SoundFont files (200-300MB total) are stored in local `./soundfonts/` directory. Remote loading or CDN delivery is not required for MVP.

3. **Monitoring Tools**: `/api/status` and `/api/metrics` endpoints provide sufficient observability for MVP. Integration with Prometheus/Grafana is post-MVP.

4. **TLS/HTTPS**: Local development uses `http://` and unencrypted WebSockets (`ws://`). Production deployment requires TLS (`https://`, `wss://`) but certificate setup is deployment-specific, not application code.

## Constraints *(optional)*

### Technical Constraints

1. **Real-Time Latency Requirement**: All audio processing must maintain <100ms latency. This prohibits blocking I/O, inefficient algorithms, or synchronous operations in the audio pipeline. Asyncio is required for concurrent operations.

2. **Memory Budget**: Total memory footprint must remain <500MB including SoundFonts, ring buffers, and synthesis state. This constrains SoundFont file sizes (prefer optimized SF2s under 200MB each) and limits polyphony to 32 simultaneous notes.

3. **GPU Dependency**: While GPU acceleration (Metal/CUDA) is strongly preferred, system must operate in CPU-only mode for users without GPU. This means CPU fallback must still meet <100ms synthesis target, which constrains synthesis complexity.

4. **WebSocket Protocol**: Real-time audio streaming requires WebSocket bidirectional communication. HTTP/REST is insufficient for continuous audio chunks. Browser must support native WebSocket API.

5. **Base64 Encoding Overhead**: WebSocket transport uses base64 encoding for binary PCM data, adding ~33% bandwidth overhead (~250 kbps vs ~190 kbps raw). This is acceptable for MVP but motivates Opus compression in post-MVP.

6. **Browser Audio API Limitations**: Web Audio API auto-play policies vary by browser. Application relies on user navigation (click on URL) to satisfy auto-play requirements, avoiding need for explicit play button.

### Scope Constraints

1. **No Percussion in MVP**: Percussion/rhythm generation (sparse kicks, granular swells) is explicitly out of scope for MVP. Focus is on melodic/harmonic quality first.

2. **No Advanced Effects in MVP**: High-quality reverb/delay (via pedalboard) is deferred to Phase 2. MVP uses FluidSynth's basic reverb only.

3. **No Multi-Client Optimization**: MVP targets 10+ concurrent clients as acceptable. Scaling to 50+ clients requires optimization (object pooling, shared synthesis buffers) deferred to Phase 3.

4. **No Cloud Deployment in MVP**: Docker/Kubernetes deployment, cloud infrastructure, and related DevOps are out of scope. MVP runs locally via uvicorn.

5. **No User Accounts**: Session persistence, user authentication, and personalization are post-MVP. localStorage provides basic settings persistence per browser.

6. **No Offline Mode**: Application requires active server connection. Offline rendering, MIDI export, and downloadable audio files are post-MVP features.

### Design Constraints

1. **Ambient Music Aesthetic**: Musical output must adhere to ambient music principles (slow evolution, minimal rhythm, harmonic ambiguity, generous space). This constrains composition algorithms to prioritize texture over melody, modal harmony over functional progressions.

2. **Minimal UI**: Interface must be unobtrusive and ambient-appropriate (dark theme, subtle animations, only essential controls visible). Complex configuration screens or visual distractions are prohibited.

3. **Modular Architecture**: System must maintain strict separation between composition, synthesis, streaming, and client layers. Circular dependencies are forbidden. This enables independent testing and future extensibility.

4. **UV-First Development**: All Python operations must use uv package manager exclusively. No pip, venv, or manual environment creation allowed. This ensures reproducible builds.

### Resource Constraints

1. **SoundFont Availability**: Free, high-quality SoundFonts are limited. Project may need to budget $100-300 for professional SF2 files if free options (Salamander, FluidR3, Arachno) prove inadequate in quality or licensing.

2. **Development Timeline**: MVP must be implementable within reasonable timeline without requiring extensive machine learning training, complex DSP research, or custom audio codec development. FluidSynth sample-based synthesis satisfies this constraint.

## Dependencies *(optional)*

### External Software Dependencies

1. **FluidSynth Native Library** (Critical)
   - **Description**: Sample-based synthesis engine (C library)
   - **Installation**: OS-specific (brew/apt-get/vcpkg)
   - **Version**: 2.x or later
   - **Impact**: Blocking - system cannot synthesize audio without it
   - **Mitigation**: Installation documented in README, checked at startup with clear error message

2. **SoundFont Files (.sf2)** (Critical)
   - **Description**: Sample libraries for piano and pad timbres
   - **Source**: Free (Salamander Grand Piano, FluidR3_GM, Arachno) or commercial
   - **Size**: 200-300MB total
   - **Impact**: Blocking - synthesis requires SoundFonts
   - **Mitigation**: Download links in setup instructions, file integrity checks at startup

3. **Python 3.12+** (Critical)
   - **Description**: Runtime environment with performance features
   - **Rationale**: Required for modern async features and PyTorch compatibility
   - **Impact**: Blocking - older Python versions unsupported
   - **Mitigation**: Version check in setup script, documented in requirements

4. **uv Package Manager** (Critical)
   - **Description**: Fast, reproducible Python dependency management
   - **Installation**: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - **Impact**: Blocking - all dependencies installed via uv
   - **Mitigation**: Installation documented, project constitution requires uv-first

### System Requirements Dependencies

1. **GPU (Metal/CUDA)** (Optional but Recommended)
   - **Description**: Hardware acceleration for synthesis
   - **Platforms**: Metal (Apple M1/M2/M4), CUDA (NVIDIA GPUs)
   - **Impact**: Performance - GPU provides 50%+ latency reduction vs CPU
   - **Mitigation**: CPU fallback implemented, GPU detection automatic via PyTorch

2. **Network Bandwidth** (Critical for Streaming)
   - **Description**: 500 kbps+ stable connection per client
   - **Impact**: Blocking - insufficient bandwidth causes dropouts
   - **Mitigation**: Adaptive client buffering, connection quality monitoring, user warnings

3. **Modern Browser** (Critical for Client)
   - **Description**: Chrome 90+, Edge 90+, Safari 14+ with Web Audio API and WebSocket support
   - **Impact**: Blocking - older browsers unsupported
   - **Mitigation**: Browser detection at startup, clear error message with upgrade recommendations

### Python Package Dependencies

1. **FastAPI 0.127+** (Critical)
   - **Purpose**: Async web framework with WebSocket support
   - **Impact**: Blocking - core server infrastructure

2. **uvicorn 0.40+** (Critical)
   - **Purpose**: ASGI server with WebSocket improvements
   - **Impact**: Blocking - server runtime

3. **PyTorch 2.5+** (Critical)
   - **Purpose**: GPU acceleration (Metal/CUDA) for synthesis effects
   - **Impact**: Performance - enables real-time processing
   - **Mitigation**: CPU fallback available

4. **pyfluidsynth 1.3.2+** (Critical)
   - **Purpose**: Python bindings for FluidSynth C library
   - **Impact**: Blocking - synthesis engine interface

5. **numpy 1.26+** (Critical)
   - **Purpose**: Audio buffer operations, DSP processing
   - **Impact**: Blocking - core data structures

### Feature Dependencies

1. **Phase 1 must complete before Phase 2**: Enhanced generation controls (Phase 2) depend on stable core streaming (Phase 1). Cannot build advanced features without reliable foundation.

2. **Performance optimizations before scaling**: Multi-client optimization (Phase 3) requires Phase 2 performance improvements. Premature scaling without optimization causes quality degradation.

3. **Core musical quality gates post-MVP**: Percussion generation, advanced effects, and Transformer-based melodies (Phase 2) require core melodic/harmonic quality to be validated first. No point adding complexity if basics are inadequate.

## Out of Scope *(optional)*

### Explicitly Excluded from MVP

The following features are intentionally excluded from this MVP to maintain focus on core value proposition validation:

#### Post-MVP Features (Future Phases)

1. **Percussion/Rhythm Generation**
   - Sparse kicks, granular swells, ambient percussion events
   - Rationale: Focus on melodic/harmonic quality first; percussion adds complexity without validating core hypothesis

2. **Advanced Audio Effects**
   - High-quality reverb/delay via pedalboard library
   - Chorus, flanger, spatial effects
   - Rationale: FluidSynth's basic reverb sufficient for MVP; advanced effects are polish, not core value

3. **Multi-Client Optimization**
   - Support for 50+ concurrent users per server instance
   - Shared synthesis buffers, object pooling
   - Rationale: MVP targets 10+ clients; scaling addressed after product-market fit proven

4. **Cloud Deployment Infrastructure**
   - Docker containerization
   - Kubernetes orchestration
   - AWS/GCP cloud infrastructure
   - CI/CD pipelines
   - Rationale: Local deployment validates product faster; cloud deployment is distribution, not feature validation

5. **User Accounts & Authentication**
   - User registration, login, session management
   - Profile settings, personalized presets
   - Cross-device synchronization
   - Rationale: localStorage suffices for MVP; accounts add complexity without validating generative music value

6. **MIDI Export & Offline Rendering**
   - Download generated compositions as MIDI files
   - Render full-length audio files (MP3/FLAC) offline
   - Rationale: Real-time streaming is core value; export is secondary utility feature

7. **Mobile Applications**
   - Native iOS app
   - Native Android app
   - Rationale: Web interface proves concept faster; mobile adds development overhead without validation gain

8. **AI/ML Advanced Generation**
   - Transformer-based melody generation (DistilGPT-2)
   - Neural network chord progressions
   - Rationale: Markov chains and constraints validate approach first; ML adds training complexity

9. **Compression & Bandwidth Optimization**
   - Opus audio compression (reduce bandwidth to ~64 kbps)
   - Adaptive bitrate streaming
   - Rationale: Base64 PCM (~250 kbps) acceptable for MVP on modern connections; optimization is polish

10. **Custom Soundscape Creation**
    - User-uploaded samples
    - Sample library management
    - Rationale: Pre-loaded SoundFonts validate synthesis quality; custom samples add content management complexity

11. **Social Features**
    - Preset sharing marketplace
    - Collaborative listening sessions
    - User-generated content
    - Rationale: MVP proves individual listening value first; social features are growth mechanics

#### Why These Are Excluded

These features add implementation complexity without validating the core hypotheses:
1. ✅ Generative ambient music is musically satisfying
2. ✅ Real-time streaming works reliably
3. ✅ Users find value in infinite, evolving soundscapes

Once MVP validates these assumptions, post-MVP phases address scalability, personalization, and advanced musicality.

## Open Questions *(optional)*

### Critical Questions (Require Decision Before Implementation)

**Q1: Which SoundFonts should we use?**

**Context**: The PRD mentions several free SoundFont options (Salamander Grand Piano, FluidR3_GM, Arachno) but doesn't specify which to include in MVP.

**What we need to know**: Which specific SoundFont files should be bundled/recommended, considering quality vs file size trade-offs?

**Suggested Answers**:

| Option | Answer | Implications |
|--------|--------|--------------|
| A      | Salamander Grand Piano (200MB) + Arachno pads (150MB) | Highest quality but 350MB total download; excellent piano timbre, rich pad textures |
| B      | FluidR3_GM (140MB) for both piano and pads | Smaller download, General MIDI compatibility, good quality but less exceptional than Salamander |
| C      | Start with FluidR3_GM, allow user to swap SoundFonts via config | Maximum flexibility, requires documentation for SoundFont management, adds configuration complexity |

**Your choice**: _A - Use Salamander + Arachno for best audio quality. 350MB is reasonable for modern connections, and audio quality is critical for ambient music. Document download links clearly in setup instructions._

---

**Q2: Should we include basic reverb via FluidSynth or defer all effects to post-MVP?**

**Context**: The PRD mentions "FluidSynth reverb (basic in MVP, upgrade Phase 2)" but doesn't definitively include it in FR requirements.

**What we need to know**: Should MVP enable FluidSynth's built-in reverb, or ship completely dry audio?

**Suggested Answers**:

| Option | Answer | Implications |
|--------|--------|--------------|
| A      | Enable FluidSynth reverb with minimal settings (3-4s decay, low CPU) | Adds spaciousness critical for ambient aesthetic; minimal CPU cost; slight latency increase (~10ms); better user experience |
| B      | Ship completely dry audio, no reverb in MVP | Faster to implement, zero reverb-related latency/CPU; audio may sound flat and un-ambient; user experience suffers |
| C      | Make reverb optional via server config flag | Maximum flexibility; adds configuration option; users may not know to enable it |

**Your choice**: _A - Enable minimal FluidSynth reverb. Reverb is essential for ambient music's spacious quality. Use conservative settings (3s decay, 20% wet) to minimize CPU impact. This improves user experience without significant complexity._

---

**Q3: Should percussion be completely excluded, or include minimal ambient textures?**

**Context**: The PRD explicitly excludes "percussion/rhythm generation (sparse kicks, granular swells)" from MVP scope, but some ambient music benefits from subtle textural percussion.

**What we need to know**: Is percussion generation strictly prohibited, or can we include minimal ambient textures if time permits?

**Suggested Answers**:

| Option | Answer | Implications |
|--------|--------|--------------|
| A      | Strictly exclude all percussion - piano and pad voices only | Clearest scope boundary; focuses effort on melodic/harmonic quality; may feel incomplete to users expecting richer textures |
| B      | Include very sparse ambient percussion (1 event per 30-60s, low velocity) | Adds subtle textural interest; increases complexity; risk of scope creep if not carefully constrained |
| C      | Defer decision - implement if time remains after core features validated | Pragmatic approach; preserves focus on P1 priorities; allows flexibility if ahead of schedule |

**Your choice**: _A - Strictly exclude percussion. The PRD is explicit about this exclusion, and adding percussion increases composition complexity. Focus on nailing melodic/harmonic quality first. Percussion can be added in Phase 2 with proper design attention._

## Related Features *(optional)*

### Existing Features This Depends On

This is the foundational MVP feature, so there are no dependencies on other features. However, this feature establishes the architecture and patterns that future features will build upon.

### Features That Will Build on This

1. **Enhanced Generation Controls** (Phase 2 / Future Spec)
   - **Relationship**: Extends musical parameter controls
   - **Dependencies**: Requires stable core streaming and composition engine from this MVP
   - **Example**: Additional modes (Mixolydian, Locrian), custom key signatures, transformer-based melody generation

2. **Performance Optimizations** (Phase 3 / spec 003)
   - **Relationship**: Optimizes synthesis and streaming performance
   - **Dependencies**: Requires working MVP to benchmark and identify bottlenecks
   - **Example**: Object pooling, shared synthesis buffers, GC tuning, multi-client scaling

3. **FluidSynth Integration** (Phase 1 / spec 004)
   - **Relationship**: Replaces PyTorch oscillators with sample-based synthesis
   - **Note**: PRD v2.0 already assumes FluidSynth integration, so this may be concurrent or already completed
   - **Dependencies**: This MVP defines synthesis interface that FluidSynth must satisfy

4. **Cloud Deployment & Scalability** (Phase 3 / Future)
   - **Relationship**: Enables production cloud hosting
   - **Dependencies**: Requires validated MVP with proven reliability metrics
   - **Example**: Docker containers, Kubernetes deployment, AWS/GCP infrastructure, Opus compression

5. **Personalization & Export** (Phase 4 / Future)
   - **Relationship**: Adds user customization and content export
   - **Dependencies**: Requires user accounts (out of MVP scope) and stable synthesis pipeline
   - **Example**: Custom presets, MIDI export, offline rendering, ML-based personalization

### Potential Conflicts or Integration Points

- **FluidSynth Integration (spec 004)**: If spec 004 is concurrent, this spec's synthesis requirements (FR-007 through FR-011) must align with FluidSynth implementation details. Coordination needed on SoundFont loading, preset mapping, and audio buffer format.

- **Performance Optimizations (spec 003)**: If spec 003 exists, optimization strategies must preserve this spec's latency requirements (<100ms synthesis, <800ms end-to-end). Performance improvements cannot break functional requirements.

- **Future Authentication Features**: When user accounts are added post-MVP, localStorage-based settings persistence (FR-022) must migrate to server-side storage without breaking existing user settings.
