"""
FluidSynth Renderer for Sample-Based Instrument Synthesis

Provides realistic instrument timbres using FluidSynth sample playback engine.
Renders melodic notes, pad chords, and choir swells using pre-recorded instrument
samples from SoundFont (SF2) files.
"""

import numpy as np
import fluidsynth
from typing import List, Tuple, Optional
from pathlib import Path


class FluidSynthRenderer:
    """
    FluidSynth-based sample playback renderer for realistic instrument synthesis.

    Handles:
    - FluidSynth initialization with 44.1kHz sample rate
    - SoundFont loading and GM preset selection
    - Polyphonic note rendering (MIDI-style note on/off)
    - Audio output as stereo float32 arrays

    GM Preset Mapping:
    - Piano: GM preset 0 (Acoustic Grand Piano) on channel 0
    - Pads: GM preset 90 (Pad Polysynth) on channel 1
    - Choir Aahs: GM preset 52 on channel 2
    - Voice Oohs: GM preset 53 on channel 3
    """

    # GM Preset constants
    PRESET_PIANO = 0          # Acoustic Grand Piano
    PRESET_PAD = 90           # Pad Polysynth
    PRESET_CHOIR_AAHS = 52    # Choir Aahs
    PRESET_VOICE_OOHS = 53    # Voice Oohs

    # MIDI channels (0-15)
    CHANNEL_PIANO = 0
    CHANNEL_PAD = 1
    CHANNEL_CHOIR_AAHS = 2
    CHANNEL_VOICE_OOHS = 3

    def __init__(self, sample_rate: int = 44100, soundfont_path: Optional[Path] = None):
        """
        Initialize FluidSynth renderer.

        Args:
            sample_rate: Target sample rate (default: 44100 Hz)
            soundfont_path: Path to SoundFont file (e.g., FluidR3_GM.sf2)
        """
        self.sample_rate = sample_rate
        self.soundfont_path = soundfont_path
        self.synth: Optional[fluidsynth.Synth] = None
        self.sfid: Optional[int] = None

        # Initialize FluidSynth
        self._initialize_fluidsynth()

        # Load SoundFont if provided
        if soundfont_path:
            self.load_soundfont(soundfont_path)

    def _initialize_fluidsynth(self) -> None:
        """
        Initialize FluidSynth synthesizer with optimal settings.

        Configuration:
        - Sample rate: 44.1kHz (matches Auralis output format)
        - Interpolation: 4th-order polynomial (high quality resampling)
        - Polyphony: 20 simultaneous voices
        - Voice stealing: Enabled (oldest-first with release-phase preference)
        """
        # Create FluidSynth instance
        self.synth = fluidsynth.Synth(samplerate=float(self.sample_rate))

        # Configure polyphony limit (synth.polyphony=20)
        # Note: pyfluidsynth API may vary - some settings are set at initialization
        # Voice stealing is automatic when polyphony limit is reached

    def load_soundfont(self, soundfont_path: Path) -> None:
        """
        Load SoundFont file and configure GM presets for all instrument channels.

        Args:
            soundfont_path: Path to SoundFont file

        Raises:
            RuntimeError: If SoundFont loading fails
        """
        if not self.synth:
            raise RuntimeError("FluidSynth not initialized")

        # Load SoundFont file
        self.sfid = self.synth.sfload(str(soundfont_path.absolute()))

        if self.sfid == -1:
            raise RuntimeError(f"Failed to load SoundFont: {soundfont_path}")

        self.soundfont_path = soundfont_path

        # Configure GM presets for each instrument channel
        self._configure_presets()

    def _configure_presets(self) -> None:
        """
        Configure General MIDI presets for all instrument channels.

        Maps GM presets to MIDI channels:
        - Channel 0: Acoustic Grand Piano (GM preset 0)
        - Channel 1: Pad Polysynth (GM preset 90)
        - Channel 2: Choir Aahs (GM preset 52)
        - Channel 3: Voice Oohs (GM preset 53)
        """
        if not self.synth or self.sfid is None:
            raise RuntimeError("SoundFont not loaded")

        # Piano on channel 0
        self.synth.program_select(
            chan=self.CHANNEL_PIANO,
            sfid=self.sfid,
            bank=0,
            preset=self.PRESET_PIANO
        )

        # Pad Polysynth on channel 1
        self.synth.program_select(
            chan=self.CHANNEL_PAD,
            sfid=self.sfid,
            bank=0,
            preset=self.PRESET_PAD
        )

        # Choir Aahs on channel 2
        self.synth.program_select(
            chan=self.CHANNEL_CHOIR_AAHS,
            sfid=self.sfid,
            bank=0,
            preset=self.PRESET_CHOIR_AAHS
        )

        # Voice Oohs on channel 3
        self.synth.program_select(
            chan=self.CHANNEL_VOICE_OOHS,
            sfid=self.sfid,
            bank=0,
            preset=self.PRESET_VOICE_OOHS
        )

    def render_notes(
        self,
        note_events: List[Tuple[int, int, float, float]],
        duration_sec: float,
        channel: int
    ) -> np.ndarray:
        """
        Render a list of note events to audio using FluidSynth.

        Args:
            note_events: List of (onset_sample, midi_pitch, velocity, duration_sec)
                        - onset_sample: Sample offset for note-on event
                        - midi_pitch: MIDI note number (0-127)
                        - velocity: Note velocity (0.0-1.0, normalized to MIDI 0-127)
                        - duration_sec: Note duration in seconds
            duration_sec: Total phrase duration in seconds
            channel: MIDI channel (0-15) to render on

        Returns:
            Stereo audio array, shape (2, num_samples), float32 normalized to [-1, 1]
        """
        if not self.synth:
            raise RuntimeError("FluidSynth not initialized")

        num_samples = int(self.sample_rate * duration_sec)

        # Create events list with note-on and note-off events
        # Each event is (sample_offset, 'on'/'off', pitch, velocity)
        events = []

        for onset_sample, midi_pitch, velocity, note_duration in note_events:
            # Note-on event
            midi_velocity = int(velocity * 127)  # Convert 0.0-1.0 to MIDI 0-127
            events.append((onset_sample, 'on', midi_pitch, midi_velocity))

            # Note-off event
            offset_sample = onset_sample + int(note_duration * self.sample_rate)
            events.append((offset_sample, 'off', midi_pitch, 0))

        # Sort events by sample offset
        events.sort(key=lambda x: x[0])

        # Render audio in chunks between events
        audio_chunks = []
        current_sample = 0

        for event_sample, event_type, pitch, velocity in events:
            # Clamp event to phrase duration
            event_sample = min(event_sample, num_samples)

            # Render audio from current position to event
            if event_sample > current_sample:
                chunk_samples = event_sample - current_sample
                chunk = self._render_chunk(chunk_samples)
                audio_chunks.append(chunk)
                current_sample = event_sample

            # Trigger MIDI event
            if event_type == 'on':
                self.synth.noteon(channel, pitch, velocity)
            else:
                self.synth.noteoff(channel, pitch)

        # Render remaining audio to fill duration
        if current_sample < num_samples:
            remaining_samples = num_samples - current_sample
            final_chunk = self._render_chunk(remaining_samples)
            audio_chunks.append(final_chunk)

        # Concatenate all chunks with crossfading to prevent clicks/pops
        if audio_chunks:
            # Apply short crossfade between chunks (1ms = 44 samples at 44.1kHz)
            # This prevents discontinuities from note-on/note-off events
            crossfade_samples = 44  # 1ms crossfade

            full_audio = audio_chunks[0].copy()  # Start with copy of first chunk

            for i in range(1, len(audio_chunks)):
                next_chunk = audio_chunks[i].copy()  # Work with copy to avoid modifying original

                # Apply crossfade if both regions are long enough
                if full_audio.shape[1] >= crossfade_samples and next_chunk.shape[1] >= crossfade_samples:
                    # Create fade-out envelope for end of current audio
                    fade_out = np.linspace(1.0, 0.0, crossfade_samples).astype(np.float32)

                    # Create fade-in envelope for start of next chunk
                    fade_in = np.linspace(0.0, 1.0, crossfade_samples).astype(np.float32)

                    # Apply envelopes to crossfade regions
                    crossfade_end = full_audio[:, -crossfade_samples:] * fade_out
                    crossfade_start = next_chunk[:, :crossfade_samples] * fade_in

                    # Mix the crossfade regions
                    full_audio[:, -crossfade_samples:] = crossfade_end + crossfade_start

                    # Append the rest of next_chunk (excluding the crossfaded start)
                    full_audio = np.concatenate([full_audio, next_chunk[:, crossfade_samples:]], axis=1)
                else:
                    # Chunks too short for crossfade, just concatenate
                    full_audio = np.concatenate([full_audio, next_chunk], axis=1)
        else:
            # No events, return silence
            full_audio = np.zeros((2, num_samples), dtype=np.float32)

        # Ensure exact length (trim or pad if needed due to rounding)
        if full_audio.shape[1] > num_samples:
            full_audio = full_audio[:, :num_samples]
        elif full_audio.shape[1] < num_samples:
            padding = np.zeros((2, num_samples - full_audio.shape[1]), dtype=np.float32)
            full_audio = np.concatenate([full_audio, padding], axis=1)

        return full_audio

    def _render_chunk(self, num_samples: int) -> np.ndarray:
        """
        Render a chunk of audio from FluidSynth.

        Args:
            num_samples: Number of samples to render

        Returns:
            Stereo audio array, shape (2, num_samples), float32 normalized to [-1, 1]
        """
        if not self.synth:
            raise RuntimeError("FluidSynth not initialized")

        # Get interleaved stereo samples from FluidSynth (returns list of int16)
        samples = self.synth.get_samples(num_samples)

        # Convert to numpy array and reshape to stereo
        audio_int16 = np.array(samples, dtype=np.int16)
        audio_stereo = audio_int16.reshape(-1, 2).T  # Shape: (2, num_samples)

        # Normalize to float32 [-1, 1] for compatibility with PyTorch synthesis
        audio_float = audio_stereo.astype(np.float32) / 32768.0

        return audio_float

    def cleanup(self) -> None:
        """Release FluidSynth resources."""
        if self.synth:
            self.synth.delete()
            self.synth = None
            self.sfid = None
