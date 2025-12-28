# Feature Specification: FluidSynth Sample-Based Instrument Synthesis

**Feature Branch**: `004-fluidsynth-integration`
**Created**: 2025-12-28
**Status**: Draft
**Input**: User description: "FluidSynth integration for realistic instrument synthesis using sample-based SoundFonts"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Realistic Acoustic Grand Piano Timbres (Priority: P1)

Listeners of the ambient music stream hear authentic Acoustic Grand Piano sounds for melodic phrases instead of synthetic oscillator tones. When melodic phrases play, they sound like actual acoustic piano rather than electronic beeps or simple waveforms.

**Why this priority**: This is the core value proposition - replacing unrealistic oscillator sounds with authentic instrument timbres. Without this, the feature has no purpose.

**Independent Test**: Can be fully tested by streaming ambient music with melodic content and having listeners confirm they hear realistic piano tones. Success is measured by subjective quality improvement over current oscillator-based synthesis.

**Acceptance Scenarios**:

1. **Given** the system is generating ambient music with melodic phrases, **When** a listener connects to the audio stream, **Then** they hear piano notes that sound like an acoustic grand piano, not synthetic oscillator tones
2. **Given** melodic notes at various pitches (low to high register), **When** played, **Then** each pitch range exhibits authentic piano timbre characteristics for that register
3. **Given** a musical phrase with multiple simultaneous notes (chords), **When** the system renders the audio, **Then** all notes blend naturally like a real instrument playing polyphonically
4. **Given** notes at different velocities (soft to loud), **When** rendered, **Then** listeners perceive natural dynamic variation matching real piano behavior

---

### User Story 2 - Rich Polysynth Pad Textures (Priority: P2)

Listeners experience lush polysynth pad sounds that provide atmospheric depth to the music. The pads sound warm and organic using sampled synthesizer pad voices rather than basic oscillators.

**Why this priority**: Pads are essential for ambient music atmosphere, but slightly less critical than getting the melodic piano right. High-quality sampled pads significantly enhance the listening experience over basic oscillators.

**Independent Test**: Can be tested by streaming music with chord progressions and evaluating whether the underlying polysynth pad textures sound rich and atmospheric. Listeners should describe the sound as "warm", "spacious", or "enveloping" rather than "thin" or "electronic".

**Acceptance Scenarios**:

1. **Given** the system is playing chord progressions, **When** polysynth pad voices are active, **Then** listeners hear smooth, continuous synthesizer textures that fill the sonic space
2. **Given** chord transitions in the music, **When** pads move from one chord to another, **Then** the transition sounds smooth and organic without clicks or artifacts
3. **Given** multiple pad voices playing simultaneously, **When** rendered, **Then** the textures blend into a cohesive, non-muddy sound
4. **Given** long sustained pad notes, **When** played over 5+ seconds, **Then** the sampled pad sound maintains its character without becoming static or boring

---

### User Story 3 - Uninterrupted Real-Time Streaming (Priority: P1)

Listeners experience smooth, continuous audio playback without stutters, dropouts, or latency issues. The audio stream maintains real-time performance regardless of whether instruments use sample-based or synthesized voices.

**Why this priority**: Real-time streaming is a core constraint of the existing system. Introducing realistic instruments cannot compromise the streaming performance - this is a critical non-functional requirement elevated to P1 priority.

**Independent Test**: Can be tested by monitoring audio stream latency and checking for dropouts during sustained listening sessions. All timing metrics must remain within established thresholds (<100ms latency, consistent chunk delivery).

**Acceptance Scenarios**:

1. **Given** the system is generating audio with realistic instruments, **When** processing a musical phrase, **Then** total latency from composition to streaming remains under 100 milliseconds
2. **Given** 10+ concurrent listeners are connected, **When** the system renders audio for all streams, **Then** no listener experiences audio dropouts or stuttering
3. **Given** the system transitions from simple to complex musical passages, **When** rendering load increases, **Then** audio streaming maintains consistent timing without buffering pauses
4. **Given** continuous music generation over 30+ minutes, **When** monitoring stream quality, **Then** timing variance between chunks stays within 50ms of target delivery schedule

---

### User Story 4 - Sampled Choir Swells & Hybrid Mixing (Priority: P2)

Listeners hear sampled choir voices (Choir Aahs, Voice Oohs) for ambient swell effects while percussion (kicks) continue using existing synthesis. The system seamlessly blends sampled instruments (piano, pads, choir) with synthesized percussion in the final audio mix.

**Why this priority**: This story ensures all atmospheric elements (swells) benefit from realistic sampled voices, while preserving the existing percussion synthesis that works well. It's P2 because it enhances the swell textures and ensures proper mixing, but isn't as critical as the melodic piano (P1).

**Independent Test**: Can be tested by generating music with percussion, melodic content, pads, and swell events, then verifying that all elements are present in the final audio stream with appropriate volume balance and choir-like swell textures.

**Acceptance Scenarios**:

1. **Given** a musical phrase contains both melodic piano notes and kick drum events, **When** the audio is rendered, **Then** listeners hear both the realistic piano AND the existing synthesized kick sounds in the mix
2. **Given** ambient swell effects are triggered using choir samples, **When** combined with polysynth pad instruments, **Then** listeners hear choir-like voices (Aahs/Oohs) blending cohesively with pads without one masking the other
3. **Given** the system is mixing 4 simultaneous layers (pads, melody, kicks, choir swells), **When** rendered, **Then** the final mix is balanced and no layer is excessively loud or inaudible
4. **Given** the mixing levels are set (e.g., 40% pads, 50% melody, 30% kicks, 20% swells), **When** audio is generated, **Then** the perceived loudness ratios match the intended mix proportions

---

## Clarifications

### Session 2025-12-28

- Q: How should the system respond when it cannot load a required SoundFont file at startup? → A: Fail startup completely - refuse to start server if any SoundFont missing/corrupted
- Q: How does the system decide which instruments to use for each musical role? → A: Always use synth pad (polysynth) for pads, Acoustic Grand for melody, and Choir Aahs and Voice Oohs for swells
- Q: What observability signals (logs, metrics, traces) should the system provide for monitoring FluidSynth synthesis performance and troubleshooting issues? → A: Minimal logging - only critical errors logged, no performance metrics exposed
- Q: What should happen when polyphony exceeds the system's capacity (e.g., 20+ simultaneous notes)? → A: Graceful voice stealing (oldest notes) - when limit reached, stop oldest active note to make room for new note
- Q: How should the system handle SoundFont files that use a different sample rate than the target 44.1kHz? → A: Automatic resampling to 44.1kHz - transparently resample SoundFont audio to match system rate

---

### Edge Cases

- **Missing or corrupted SoundFont files**: System MUST refuse to start if any required SoundFont file (Acoustic Grand Piano, polysynth pads, choir voices) is missing or corrupted. Startup validation checks all configured SoundFont paths and file integrity before initialization.
- **Extremely high polyphony (20+ simultaneous notes)**: System MUST implement voice stealing, stopping the oldest active note when polyphony limit is reached to make room for new notes. This prevents performance degradation and maintains audio quality.
- What occurs if a SoundFont file is too large to load into memory on resource-constrained systems?
- How does the system behave when switching between different instrument presets mid-stream?
- **SoundFont sample rate mismatch**: System MUST automatically resample SoundFont audio (e.g., 48kHz, 22kHz samples) to match the system's 44.1kHz target rate transparently, ensuring compatibility with any quality SoundFont library.
- What happens during very rapid note sequences (e.g., 32nd notes at 180 BPM)?
- How does the system handle notes with very long release tails that extend beyond phrase boundaries?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST render melodic notes using sampled Acoustic Grand Piano, producing timbre indistinguishable from recorded piano to non-expert listeners
- **FR-002**: System MUST render ambient pad chords using sampled polysynth (synthesizer pad) sounds with smooth, sustained textures
- **FR-003**: System MUST render swell effects using sampled choir voices (Choir Aahs and Voice Oohs) for atmospheric vocal textures
- **FR-004**: System MUST support polyphonic playback (multiple simultaneous notes) for all sampled instrument types (piano, pads, choir)
- **FR-005**: System MUST respond to velocity information (soft to loud) when rendering notes, producing natural dynamic variation
- **FR-006**: System MUST continue rendering existing percussion voices (kicks) using current PyTorch synthesis methods
- **FR-007**: System MUST mix sampled instruments (piano, pads, choir) and synthesized percussion (kicks) into a balanced stereo audio stream
- **FR-008**: System MUST maintain audio streaming at 44.1kHz, 16-bit PCM in 100ms chunks
- **FR-009**: System MUST preserve total audio processing latency under 100 milliseconds
- **FR-010**: System MUST handle note events with sample-accurate timing (no rhythmic drift or timing jitter)
- **FR-011**: System MUST generate audio without audible clicks, pops, or artifacts during note transitions
- **FR-012**: System MUST load and use pre-recorded instrument samples from external SoundFont files
- **FR-013**: System MUST use specific General MIDI presets: Acoustic Grand Piano (preset 0) for melody, Pad Polysynth (preset 90) for pads, Choir Aahs (preset 52) and Voice Oohs (preset 53) for swells
- **FR-014**: System MUST render complete musical phrases upfront before streaming (per-phrase rendering strategy)
- **FR-015**: System MUST apply soft clipping or limiting to prevent audio distortion when mixing multiple layers
- **FR-016**: System MUST validate all required SoundFont files (Acoustic Grand, polysynth pads, choir voices) exist and are readable at startup, refusing to start if any are missing or corrupted
- **FR-017**: System MUST implement voice stealing when polyphony exceeds configured limit, stopping the oldest active note to allocate voice for new note event
- **FR-018**: System MUST automatically resample SoundFont audio to 44.1kHz if the SoundFont uses a different native sample rate (e.g., 48kHz, 22kHz), ensuring output always matches system target rate

### Non-Functional Requirements

- **NFR-001**: System MUST log only critical errors related to FluidSynth synthesis (SoundFont loading failures, rendering exceptions, fatal errors)
- **NFR-002**: System MUST NOT expose performance metrics, debug logs, or operational telemetry for FluidSynth synthesis operations

### Key Entities

- **Musical Phrase**: A complete segment of generated ambient music with defined duration, containing chord progressions, melodies, percussion events, and swell effects. Serves as the unit of composition and rendering.
- **Instrument Voice**: A specific instrument timbre responsible for rendering musical notes. Four voice types: Acoustic Grand Piano (melody), Polysynth Pad (chord pads), Choir (swells - Aahs/Oohs), and Kick (synthesized percussion). Each sampled voice has configuration for SoundFont source and General MIDI preset selection.
- **Note Event**: A musical note with properties: onset time (samples), pitch (MIDI number), velocity (0.0-1.0), and duration (seconds). Represents an atomic musical action for sampled instruments.
- **Sample Library**: A collection of pre-recorded instrument sounds (SoundFont file) containing samples at various pitches and velocities. Provides the source material for realistic instrument synthesis. Must include Acoustic Grand Piano, polysynth pad, and choir voice presets.
- **Audio Mix**: The final stereo audio output combining all sampled voices (piano, pads, choir) and synthesized percussion (kicks), balanced by mixing weights and processed with soft clipping.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Subjective listening tests show 80%+ of listeners prefer realistic instrument timbres over original oscillator sounds
- **SC-002**: Audio processing latency remains consistently under 100ms when measured from composition to streaming output
- **SC-003**: System successfully renders 16-second musical phrases in under 1.6 seconds (10× real-time factor)
- **SC-004**: System supports 10+ concurrent listener streams without any audio dropouts or stuttering
- **SC-005**: Timing variance between consecutive 100ms audio chunks stays within ±50ms of target schedule
- **SC-006**: Audio quality metrics show no clipping (samples exceeding ±0.99) after mix-down and soft limiting
- **SC-007**: System handles musical phrases with 15+ simultaneous notes without performance degradation, using voice stealing gracefully when necessary
- **SC-008**: Voice stealing (when polyphony limit exceeded) occurs without audible clicks or artifacts
- **SC-009**: Memory usage per server instance stays under 500MB including loaded sample libraries
- **SC-010**: Polyphonic chords (3-5 notes) render with natural blend, as verified by spectral analysis showing no phase cancellation
- **SC-011**: Note velocity range (0.0 to 1.0) produces perceptibly different dynamics spanning pianissimo to fortissimo

### Qualitative Outcomes

- Listeners describe the music as "realistic", "warm", or "authentic" rather than "synthetic" or "electronic"
- Musicians or audio engineers cannot easily distinguish sampled instruments from real recordings
- The ambient music maintains its characteristic evolving, non-repetitive quality while sounding more professional
- Users report increased emotional engagement or relaxation response compared to oscillator-based version

## Dependencies & Assumptions *(optional)*

### Dependencies

- Existing ambient music composition system (chord generator, melody generator) continues to produce MIDI-like note events
- Existing WebSocket streaming infrastructure remains functional and unchanged
- Existing PyTorch percussion synthesis (kicks) continues to operate for percussion elements
- Access to free or licensed SoundFont (SF2) files containing Acoustic Grand Piano (GM preset 0), Pad Polysynth (GM preset 90), Choir Aahs (GM preset 52), and Voice Oohs (GM preset 53)
- System has sufficient CPU capacity to render sample-based synthesis in real-time

### Assumptions

- Default SoundFont files are assumed to be under 500MB combined to fit within memory constraints
- Sample libraries provide adequate quality (48kHz or 44.1kHz, 16-bit or 24-bit) for ambient music use case
- Users/listeners are accessing the stream via standard web browsers with Web Audio API support
- The target deployment environment has at least 4 CPU cores available for parallel processing
- Sample-based synthesis will be CPU-only (no GPU acceleration required for this feature)
- Existing latency budget (<100ms) is sufficient to accommodate sample playback overhead
- FluidSynth or similar sample-based synthesis engine can be integrated as a dependency
- Cross-platform compatibility (macOS, Linux, Windows) is achievable with appropriate system libraries

### Constraints

- Audio streaming format is fixed at 44.1kHz, 16-bit PCM, stereo
- Total processing latency cannot exceed 100ms (hard real-time constraint)
- WebSocket chunk size remains at 100ms duration (~17.6KB per chunk)
- Changes must not break existing REST API endpoints for control and status
- Implementation must follow UV-first development principle (Python package management)
- Solution must work in development (single-worker) and production (multi-worker) server modes

## Out of Scope

The following are explicitly NOT included in this feature:

- Custom recording or creation of sample libraries (will use existing free/licensed General MIDI SoundFonts)
- GPU acceleration for sample playback (CPU-only approach)
- Real-time pitch shifting or time stretching of samples (samples used at native pitches)
- Advanced effects processing (reverb, delay, chorus) beyond existing soft clipping
- MIDI controller input or interactive instrument playing (system remains generative-only)
- Changing the composition algorithms or musical structure (focus is on synthesis only)
- Migration to alternative synthesis methods (e.g., physical modeling, neural synthesis)
- Dynamic instrument selection or switching (fixed mapping: piano for melody, polysynth for pads, choir for swells, synthesis for kicks)
- Alternative instrument timbres beyond specified GM presets (no harpsichord, organ, strings, etc.)
- Support for non-standard sample formats (SF2/SoundFont only)
- Backwards compatibility with older audio streaming clients (assumes current protocol)

## Notes

This specification focuses on replacing the synthesis layer while preserving the existing composition and streaming architecture. The goal is realistic instrument timbres, not a complete system redesign. Future enhancements (neural synthesis, effects processing, dynamic instrument selection) can build on this foundation but are separate features.

The success of this feature will be measured primarily through subjective listening quality and objective performance metrics (latency, throughput). Both dimensions are equally important - realistic sounds are worthless if they break the real-time streaming requirement.
