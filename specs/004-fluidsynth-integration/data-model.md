# Data Model: FluidSynth Sample-Based Instrument Synthesis

**Feature**: FluidSynth Integration
**Branch**: `004-fluidsynth-integration`
**Source**: Extracted from [spec.md](spec.md) Key Entities section

## Purpose

This document defines the data structures, entities, and their relationships for the FluidSynth integration feature. All entities are derived from functional requirements and user scenarios, focusing on the "what" (data structure) rather than "how" (implementation).

---

## Entity Overview

```
┌─────────────────┐
│ Musical Phrase  │──────────────┬─────────────┬──────────────┐
└─────────────────┘              │             │              │
                                 │             │              │
                         ┌───────▼──────┐ ┌────▼─────┐  ┌────▼──────┐
                         │ Instrument   │ │ Note     │  │ Audio     │
                         │ Voice        │ │ Event    │  │ Mix       │
                         └──────────────┘ └──────────┘  └───────────┘
                                 │
                         ┌───────▼──────────┐
                         │ Sample Library   │
                         │ (SoundFont)      │
                         └──────────────────┘
```

---

## 1. Musical Phrase

**Description**: A complete segment of generated ambient music with defined duration, containing chord progressions, melodies, percussion events, and swell effects. Serves as the unit of composition and rendering.

### Fields

| Field Name | Type | Constraints | Description |
|------------|------|-------------|-------------|
| `phrase_id` | string (UUID) | Required, unique | Unique identifier for this phrase |
| `duration_sec` | float | Required, 4.0-30.0 | Total duration in seconds (typically 8-16 bars) |
| `chord_progression` | list[Chord] | Required, 1-16 chords | Sequence of chord events with timing |
| `melody_notes` | list[NoteEvent] | Optional, 0-100 notes | Melodic note events (piano) |
| `percussion_events` | list[NoteEvent] | Optional, 0-50 events | Kick drum events (PyTorch synthesis) |
| `swell_events` | list[NoteEvent] | Optional, 0-20 events | Choir swell events (Aahs/Oohs) |
| `key` | str | Required, ["C", "D", "E", ...] | Musical key (12 options) |
| `bpm` | int | Required, 60-120 | Tempo in beats per minute |
| `sample_rate` | int | Required, 44100 | Target sample rate for rendering |
| `created_at` | datetime | Required | Timestamp when phrase was composed |

### Validation Rules

- Duration must allow complete rendering within <100ms latency budget (FR-009)
- Total note count (melody + swells) must not exceed polyphony limit (15-20 simultaneous, FR-004, SC-007)
- All note onsets must fall within `[0, duration_sec * sample_rate]` sample range (FR-010)

### Relationships

- **Contains** multiple Instrument Voices (piano, pads, choir, kicks)
- **Generates** Note Events for each voice
- **Produces** Audio Mix as final output

---

## 2. Instrument Voice

**Description**: A specific instrument timbre responsible for rendering musical notes. Four voice types: Acoustic Grand Piano (melody), Polysynth Pad (chord pads), Choir (swells - Aahs/Oohs), and Kick (synthesized percussion). Each sampled voice has configuration for SoundFont source and General MIDI preset selection.

### Voice Types

#### 2.1 Acoustic Grand Piano (Melody Voice)

| Field Name | Type | Constraints | Description |
|------------|------|-------------|-------------|
| `voice_type` | enum | `"piano"` | Voice type identifier |
| `soundfont_file` | string | Required | Path to SoundFont file (e.g., "FluidR3_GM.sf2") |
| `soundfont_id` | int | Required, ≥0 | FluidSynth SoundFont ID (from `sfload`) |
| `gm_preset` | int | 0 (Acoustic Grand Piano) | General MIDI preset number (FR-013) |
| `gm_bank` | int | 0 (default bank) | General MIDI bank number |
| `channel` | int | 0-15 | MIDI channel assignment |
| `polyphony_limit` | int | 15-20 | Maximum simultaneous notes (FR-004, SC-007) |
| `velocity_range` | tuple[float, float] | (0.0, 1.0) | Valid velocity range |

#### 2.2 Pad Polysynth (Chord Pad Voice)

| Field Name | Type | Constraints | Description |
|------------|------|-------------|-------------|
| `voice_type` | enum | `"pad"` | Voice type identifier |
| `soundfont_file` | string | Required | Path to SoundFont file |
| `soundfont_id` | int | Required, ≥0 | FluidSynth SoundFont ID |
| `gm_preset` | int | 90 (Pad Polysynth) | General MIDI preset number (FR-013) |
| `gm_bank` | int | 0 | General MIDI bank number |
| `channel` | int | 0-15 | MIDI channel assignment |
| `polyphony_limit` | int | 15-20 | Maximum simultaneous notes |
| `velocity_range` | tuple[float, float] | (0.0, 1.0) | Valid velocity range |

#### 2.3 Choir Voices (Swell Effects)

| Field Name | Type | Constraints | Description |
|------------|------|-------------|-------------|
| `voice_type` | enum | `"choir_aahs"` or `"voice_oohs"` | Voice type identifier |
| `soundfont_file` | string | Required | Path to SoundFont file |
| `soundfont_id` | int | Required, ≥0 | FluidSynth SoundFont ID |
| `gm_preset` | int | 52 (Choir Aahs) or 53 (Voice Oohs) | GM preset (FR-013) |
| `gm_bank` | int | 0 | General MIDI bank number |
| `channel` | int | 0-15 | MIDI channel assignment |
| `polyphony_limit` | int | 15-20 | Maximum simultaneous notes |
| `velocity_range` | tuple[float, float] | (0.0, 1.0) | Valid velocity range |

#### 2.4 Kick Voice (PyTorch Synthesis - NOT FluidSynth)

| Field Name | Type | Constraints | Description |
|------------|------|-------------|-------------|
| `voice_type` | enum | `"kick"` | Voice type identifier |
| `synthesis_method` | enum | `"pytorch"` | Uses existing PyTorch synthesis (FR-006) |
| `voice_class` | string | `"KickVoice"` | Python class name in synthesis_engine.py |
| `mix_weight` | float | 0.3 (30% mixing level) | Relative mix level (FR-007) |

### Validation Rules

- Each voice type must map to exactly one GM preset (FR-013)
- SoundFont file must exist and be loaded at startup (FR-016)
- Polyphony limit must support 15+ simultaneous notes (FR-004, SC-007)
- Voice stealing must be configured when polyphony limit exceeded (FR-017)

### Relationships

- **Uses** Sample Library (SoundFont) for sound generation
- **Renders** Note Events into audio samples
- **Contributes to** Audio Mix with configured weight

---

## 3. Note Event

**Description**: A musical note with properties: onset time (samples), pitch (MIDI number), velocity (0.0-1.0), and duration (seconds). Represents an atomic musical action for sampled instruments.

### Fields

| Field Name | Type | Constraints | Description |
|------------|------|-------------|-------------|
| `onset_sample` | int | Required, ≥0 | Onset time in samples from phrase start (FR-010) |
| `pitch` | int | Required, 0-127 | MIDI pitch number (60 = Middle C) |
| `velocity` | float | Required, 0.0-1.0 | Note velocity (loudness, 0=silent, 1=fortissimo) (FR-005) |
| `duration_sec` | float | Required, 0.05-30.0 | Note duration in seconds |
| `voice_type` | enum | Required | Which voice renders this note (`"piano"`, `"pad"`, `"choir_aahs"`, `"voice_oohs"`, `"kick"`) |
| `channel` | int | 0-15 | MIDI channel for multi-timbral rendering |

### Derived Fields

| Field Name | Type | Calculation | Description |
|------------|------|-------------|-------------|
| `onset_sec` | float | `onset_sample / sample_rate` | Onset time in seconds |
| `duration_samples` | int | `int(duration_sec * sample_rate)` | Note duration in samples |
| `midi_velocity` | int | `int(velocity * 127)` | MIDI velocity (0-127) for FluidSynth API |

### Validation Rules

- Onset must be sample-accurate (integer sample offset, FR-010)
- Pitch must be valid MIDI number (0-127)
- Velocity must normalize to 0.0-1.0 range (FR-005)
- Duration must be positive and within phrase duration
- Multiple simultaneous notes must not exceed polyphony limit (SC-007)

### State Transitions

```
[Created] ──(trigger)──> [Active/Sustaining]
                              │
                              ├──(note-off)──> [Releasing]
                              │                     │
                              │                     └──(release complete)──> [Inactive]
                              │
                              └──(voice stolen)──> [Fast Release]──> [Inactive]
```

### Relationships

- **Belongs to** Musical Phrase
- **Rendered by** Instrument Voice
- **May trigger** Voice Stealing when polyphony limit exceeded

---

## 4. Sample Library (SoundFont)

**Description**: A collection of pre-recorded instrument sounds (SoundFont file) containing samples at various pitches and velocities. Provides the source material for realistic instrument synthesis. Must include Acoustic Grand Piano, polysynth pad, and choir voice presets.

### Fields

| Field Name | Type | Constraints | Description |
|------------|------|-------------|-------------|
| `file_path` | string (Path) | Required, exists | Absolute path to SoundFont file (e.g., "soundfonts/FluidR3_GM.sf2") |
| `file_size_mb` | float | Required, >100 | File size in megabytes |
| `soundfont_id` | int | Required, ≥0 | FluidSynth internal ID from `sfload()` |
| `format` | string | `"SF2"` (SoundFont 2) | File format (FR-012) |
| `native_sample_rate` | int | 22050, 44100, 48000 | Native sample rate of SF2 samples |
| `num_presets` | int | ≥128 | Number of GM presets available |
| `contains_presets` | list[int] | [0, 52, 53, 90] | Required GM presets (piano, choir, pad) |
| `loaded_at` | datetime | Required | Timestamp when SoundFont was loaded |
| `validated` | bool | Required | Passed startup validation (FR-016) |

### Validation Rules

- File must exist and be readable at startup (FR-016)
- File size must be >100MB to detect corrupted/truncated files (FR-016)
- File must load successfully via `FluidSynth.sfload()` (FR-016)
- Must contain required GM presets: 0, 52, 53, 90 (FR-013)
- Server must refuse to start if validation fails (FR-016)

### Resampling Behavior

- FluidSynth **automatically resamples** samples to target 44.1kHz if native sample rate differs (FR-018)
- No manual resampling required - handled transparently by FluidSynth
- Interpolation quality configured via `synth.interpolation` setting (4th-order recommended)

### Relationships

- **Loaded by** SoundFont Manager at startup
- **Used by** Instrument Voices for sample playback
- **Provides** audio samples for Note Events

---

## 5. Audio Mix

**Description**: The final stereo audio output combining all sampled voices (piano, pads, choir) and synthesized percussion (kicks), balanced by mixing weights and processed with soft clipping.

### Fields

| Field Name | Type | Constraints | Description |
|------------|------|-------------|-------------|
| `mix_id` | string (UUID) | Required, unique | Unique identifier for this mix |
| `phrase_id` | string (UUID) | Required, FK | Reference to source Musical Phrase |
| `duration_samples` | int | Required, >0 | Total duration in samples (44.1kHz) |
| `audio_data` | ndarray | shape=(2, N), dtype=float32 | Stereo audio data, normalized [-1, 1] |
| `mix_weights` | dict[str, float] | Required | Mixing weights per voice type |
| `sample_rate` | int | 44100 | Output sample rate (FR-008) |
| `bit_depth` | int | 16 | Output bit depth (FR-008) |
| `clipped` | bool | Required | Whether soft clipping was applied (FR-015) |
| `peak_level` | float | 0.0-1.0 | Maximum absolute sample value |
| `rms_level` | float | 0.0-1.0 | RMS (average) level |

### Mixing Weights (FR-007)

Default mixing weights from specification:

| Voice Type | Weight | Description |
|------------|--------|-------------|
| `pads` | 0.4 (40%) | Polysynth pad chord textures |
| `melody` | 0.5 (50%) | Acoustic Grand Piano melody |
| `kicks` | 0.3 (30%) | Synthesized percussion (PyTorch) |
| `swells` | 0.2 (20%) | Choir voices (Aahs/Oohs) |

**Note**: Weights sum to 1.4 (>1.0), requiring auto-gain scaling to prevent clipping

### Processing Pipeline

```
[Individual Voice Renders] (stereo)
         │
         ├── Piano (50%)
         ├── Pads (40%)
         ├── Choir (20%)
         └── Kicks (30%, mono→stereo)
         │
         ▼
[Weighted Sum] (weighted mix)
         │
         ▼
[Auto-Gain] (scale if weights > 1.0)
         │
         ▼
[Soft Clipping/Limiting] (prevent distortion, FR-015)
         │
         ▼
[Final Stereo Audio] (44.1kHz, 16-bit, PCM)
```

### Validation Rules

- Audio data must be stereo (2 channels, FR-008)
- Sample rate must be exactly 44.1kHz (FR-008)
- Peak level must not exceed ±0.99 after soft limiting (SC-006)
- No clipping (samples exceeding ±0.99) allowed (SC-006)
- RMS level should be reasonable for ambient music (typically -20dB to -10dB)

### Relationships

- **Produced from** Musical Phrase
- **Combines** outputs from multiple Instrument Voices
- **Outputs to** WebSocket streaming buffer

---

## Entity Relationships Summary

```
Musical Phrase (1)
    │
    ├──> Note Events (0..N)
    │       │
    │       └──> Instrument Voice (piano | pad | choir | kick)
    │               │
    │               └──> Sample Library (SoundFont SF2)
    │
    └──> Audio Mix (1)
            │
            └──> WebSocket Stream Chunks (N)
```

### Cardinality

| Relationship | Cardinality | Notes |
|--------------|-------------|-------|
| Phrase → Note Events | 1:N | One phrase contains many notes |
| Note Event → Instrument Voice | N:1 | Many notes rendered by one voice type |
| Instrument Voice → Sample Library | N:1 | Multiple voices can share one SoundFont |
| Phrase → Audio Mix | 1:1 | One phrase produces one final mix |
| Audio Mix → Stream Chunks | 1:N | Mix divided into 100ms WebSocket chunks |

---

## Data Flow

### Composition → Rendering → Streaming

```
1. [Composition Layer]
   └──> Generate Musical Phrase
        ├── Chord progression (Markov)
        ├── Melody notes (constraint-based)
        ├── Percussion events (sparse)
        └── Swell effects (ambient)

2. [FluidSynth Rendering]
   └──> For each voice type:
        ├── Load SoundFont (startup, cached)
        ├── Select GM preset
        ├── Trigger Note Events (noteon)
        └── Render to stereo audio buffer

3. [PyTorch Rendering]
   └──> Render kick percussion (existing synthesis)

4. [Mixing Layer]
   └──> Weighted sum mix:
        ├── Piano (50%)
        ├── Pads (40%)
        ├── Choir (20%)
        ├── Kicks (30%)
        ├── Auto-gain scaling
        └── Soft clipping/limiting

5. [Audio Mix Output]
   └──> Stereo PCM (44.1kHz, 16-bit)
        └── Split into 100ms chunks
            └── WebSocket streaming
```

---

## Configuration Data

### SoundFont Configuration

```python
# server/soundfont_config.py (conceptual)

SOUNDFONT_CONFIG = {
    "file": "FluidR3_GM.sf2",
    "path": "./soundfonts",
    "voices": {
        "piano": {
            "preset": 0,
            "bank": 0,
            "name": "Acoustic Grand Piano"
        },
        "pad": {
            "preset": 90,
            "bank": 0,
            "name": "Pad Polysynth"
        },
        "choir_aahs": {
            "preset": 52,
            "bank": 0,
            "name": "Choir Aahs"
        },
        "voice_oohs": {
            "preset": 53,
            "bank": 0,
            "name": "Voice Oohs"
        }
    }
}
```

### Mixing Configuration

```python
# server/mixing_config.py (conceptual)

MIXING_CONFIG = {
    "weights": {
        "pads": 0.4,
        "melody": 0.5,
        "kicks": 0.3,
        "swells": 0.2
    },
    "auto_gain_headroom_db": -6.0,  # Leave -6dB headroom
    "soft_clip_threshold": 0.8,
    "soft_clip_knee": 0.1
}
```

### FluidSynth Configuration

```python
# server/fluidsynth_config.py (conceptual)

FLUIDSYNTH_CONFIG = {
    "sample_rate": 44100,
    "polyphony": 20,
    "interpolation": "4thorder",
    "reverb_active": False,
    "chorus_active": False,
    "overflow_important": True,
    "overflow_released": True
}
```

---

## Persistence & State Management

### In-Memory Only (No Database)

This feature does NOT introduce persistent storage:

- **Musical Phrases**: Generated on-the-fly, not persisted
- **Note Events**: Ephemeral, exist only during phrase composition
- **Audio Mix**: Rendered in real-time, streamed immediately, not saved
- **SoundFonts**: Files on disk, loaded into FluidSynth memory at startup

### State Lifecycle

```
Server Startup:
    └──> Load SoundFonts (validate, cache in FluidSynth)

Per Request:
    └──> Generate Phrase
         └──> Render Audio
              └──> Mix
                   └──> Stream Chunks
                        └──> Discard (no persistence)

Server Shutdown:
    └──> Unload SoundFonts
         └──> Cleanup FluidSynth instances
```

---

## Success Criteria Validation

This data model supports all success criteria:

- **SC-001**: Entities support realistic timbres via SoundFont samples
- **SC-002**: Latency tracked via timing metadata in Audio Mix
- **SC-003**: Real-time factor measured from phrase generation to mix completion
- **SC-004**: Concurrent streams handled via independent phrase/mix instances
- **SC-005**: Timing variance tracked in Audio Mix metadata
- **SC-006**: Clipping detection via `peak_level` and `clipped` fields
- **SC-007**: Polyphony limits enforced in Instrument Voice configuration
- **SC-008**: Voice stealing transparency via FluidSynth internal handling
- **SC-009**: Memory usage monitored via SoundFont `file_size_mb` and instance count
- **SC-010**: Polyphonic rendering validated via Note Event relationships
- **SC-011**: Velocity range validation in Note Event schema

---

**Data Model Status**: ✅ COMPLETE
**Ready for Implementation**: YES
**Next Step**: Generate API contracts (if applicable) and quickstart.md testing scenarios
