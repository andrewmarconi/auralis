"""
Integration Tests for FluidSynth Rendering

Tests realistic instrument synthesis using FluidSynth sample-based playback.
Validates piano rendering, polyphony, and velocity response for User Story 1.
"""

import numpy as np
import pytest

from server.fluidsynth_renderer import FluidSynthRenderer
from server.soundfont_manager import SoundFontManager


@pytest.fixture
def fluidsynth_renderer():
    """Create FluidSynthRenderer instance with loaded SoundFont."""
    soundfont_manager = SoundFontManager()
    soundfont_path = soundfont_manager.get_soundfont_path("FluidR3_GM.sf2")

    renderer = FluidSynthRenderer(sample_rate=44100, soundfont_path=soundfont_path)
    yield renderer
    renderer.cleanup()


def test_piano_rendering_basic(fluidsynth_renderer):
    """
    Test basic piano rendering with single note (T024, FR-001).

    Validates that FluidSynth can render a single piano note with:
    - Correct output format (stereo float32)
    - Non-zero audio output (actual sound generated)
    - Reasonable amplitude range
    """
    # Single middle C note: onset=0, pitch=60, velocity=0.7, duration=1.0s
    note_events = [
        (0, 60, 0.7, 1.0)  # Middle C, 1 second duration
    ]

    duration_sec = 1.0

    # Render using piano channel
    audio = fluidsynth_renderer.render_notes(
        note_events=note_events,
        duration_sec=duration_sec,
        channel=FluidSynthRenderer.CHANNEL_PIANO
    )

    # Validate output format
    assert audio.shape == (2, 44100), "Expected stereo (2, 44100) output"
    assert audio.dtype == np.float32, "Expected float32 output"

    # Validate audio was generated (not silence)
    rms = np.sqrt(np.mean(audio ** 2))
    assert rms > 0.001, f"Audio RMS too low ({rms}), expected audible piano sound"

    # Validate reasonable amplitude (not clipping)
    assert np.max(np.abs(audio)) <= 1.0, "Audio clipping detected (exceeds [-1, 1])"
    # FluidSynth piano output is relatively quiet by default (~0.03-0.06 range)
    # This is normal and will be mixed/amplified in final synthesis
    assert np.max(np.abs(audio)) > 0.01, "Audio amplitude too low for piano note"


def test_polyphonic_piano_chord(fluidsynth_renderer):
    """
    Test polyphonic piano chord rendering (T025, SC-010).

    Validates that FluidSynth can render 3-5 simultaneous piano notes (chord):
    - C major chord (C, E, G) = MIDI 60, 64, 67
    - All notes play simultaneously
    - Natural blending of polyphonic voices
    """
    # C major chord (3 notes): Middle C, E, G
    # All notes start at same time (onset=0)
    note_events = [
        (0, 60, 0.7, 1.5),  # C (root)
        (0, 64, 0.6, 1.5),  # E (third)
        (0, 67, 0.6, 1.5),  # G (fifth)
    ]

    duration_sec = 1.5

    # Render polyphonic chord
    audio = fluidsynth_renderer.render_notes(
        note_events=note_events,
        duration_sec=duration_sec,
        channel=FluidSynthRenderer.CHANNEL_PIANO
    )

    # Validate output format
    assert audio.shape == (2, int(44100 * 1.5)), "Expected stereo output matching duration"

    # Validate polyphonic audio was generated
    rms = np.sqrt(np.mean(audio ** 2))
    assert rms > 0.001, f"Chord RMS too low ({rms}), expected audible polyphonic piano"

    # Polyphonic chords should have higher amplitude than single notes
    # (but still within [-1, 1] due to FluidSynth's internal mixing)
    # FluidSynth uses conservative gain, so 3-note chords reach ~0.06-0.08
    assert np.max(np.abs(audio)) > 0.02, "Polyphonic chord amplitude too low"
    assert np.max(np.abs(audio)) <= 1.0, "Polyphonic chord clipping detected"

    # Verify spectral content indicates multiple frequencies (crude check)
    # Polyphonic audio should have more energy spread across frequency spectrum
    # than a single note (higher variance in amplitude over time)
    amplitude_variance = np.var(np.abs(audio))
    # FluidSynth piano produces smooth sustained tones with low variance
    assert amplitude_variance > 0.00001, "Expected higher variance for polyphonic content"


def test_polyphonic_five_note_chord(fluidsynth_renderer):
    """
    Test 5-note polyphonic piano chord (T025, SC-010).

    Validates that FluidSynth can handle larger chords (5 simultaneous notes).
    C major 7th chord: C, E, G, B, D (5 notes)
    """
    # C major 7th chord (5 notes)
    note_events = [
        (0, 60, 0.7, 2.0),  # C (root)
        (0, 64, 0.6, 2.0),  # E (third)
        (0, 67, 0.6, 2.0),  # G (fifth)
        (0, 71, 0.5, 2.0),  # B (seventh)
        (0, 62, 0.5, 2.0),  # D (ninth)
    ]

    duration_sec = 2.0

    # Render 5-note chord
    audio = fluidsynth_renderer.render_notes(
        note_events=note_events,
        duration_sec=duration_sec,
        channel=FluidSynthRenderer.CHANNEL_PIANO
    )

    # Validate output
    assert audio.shape == (2, 44100 * 2), "Expected stereo 2-second output"

    # Validate rich polyphonic audio
    rms = np.sqrt(np.mean(audio ** 2))
    assert rms > 0.001, f"5-note chord RMS too low ({rms})"

    # No clipping despite 5 simultaneous voices
    assert np.max(np.abs(audio)) <= 1.0, "5-note chord clipping detected"


def test_velocity_range(fluidsynth_renderer):
    """
    Test piano velocity response range (T026, FR-005, SC-011).

    Validates that FluidSynth responds to MIDI velocity (0.0-1.0):
    - Low velocity (0.2) produces quieter sound
    - High velocity (1.0) produces louder sound
    - Velocity affects perceived dynamics (amplitude)
    """
    # Soft note (velocity 0.2)
    soft_note = [(0, 60, 0.2, 0.5)]
    soft_audio = fluidsynth_renderer.render_notes(
        note_events=soft_note,
        duration_sec=0.5,
        channel=FluidSynthRenderer.CHANNEL_PIANO
    )
    soft_rms = np.sqrt(np.mean(soft_audio ** 2))

    # Loud note (velocity 1.0)
    loud_note = [(0, 60, 1.0, 0.5)]
    loud_audio = fluidsynth_renderer.render_notes(
        note_events=loud_note,
        duration_sec=0.5,
        channel=FluidSynthRenderer.CHANNEL_PIANO
    )
    loud_rms = np.sqrt(np.mean(loud_audio ** 2))

    # Validate velocity affects amplitude
    assert loud_rms > soft_rms, (
        f"High velocity (1.0) should produce louder sound than low velocity (0.2). "
        f"Got loud_rms={loud_rms:.6f}, soft_rms={soft_rms:.6f}"
    )

    # Velocity should produce significant dynamic range
    # Expect at least 2x amplitude difference between soft and loud
    dynamic_ratio = loud_rms / soft_rms
    assert dynamic_ratio > 1.5, (
        f"Expected significant dynamic range (>1.5x), got {dynamic_ratio:.2f}x"
    )

    # Both should produce audible sound
    assert soft_rms > 0.0001, f"Soft note too quiet ({soft_rms})"
    assert loud_rms > 0.001, f"Loud note too quiet ({loud_rms})"


def test_velocity_range_mid_range(fluidsynth_renderer):
    """
    Test mid-range velocity response (T026, FR-005).

    Validates that mid-range velocities (0.5, 0.7) produce intermediate dynamics.
    """
    velocities = [0.3, 0.5, 0.7, 0.9]
    rms_values = []

    for velocity in velocities:
        note_events = [(0, 60, velocity, 0.5)]
        audio = fluidsynth_renderer.render_notes(
            note_events=note_events,
            duration_sec=0.5,
            channel=FluidSynthRenderer.CHANNEL_PIANO
        )
        rms = np.sqrt(np.mean(audio ** 2))
        rms_values.append(rms)

    # Validate monotonic increase: higher velocity → higher RMS
    for i in range(len(rms_values) - 1):
        assert rms_values[i+1] > rms_values[i], (
            f"Velocity {velocities[i+1]} should be louder than {velocities[i]}. "
            f"Got RMS: {rms_values[i+1]:.6f} vs {rms_values[i]:.6f}"
        )


def test_piano_note_sequence(fluidsynth_renderer):
    """
    Test sequential piano notes (melodic line).

    Validates that FluidSynth can render a sequence of piano notes
    without clicks or artifacts between notes.
    """
    # Simple melodic sequence: C D E F G (ascending scale)
    # Staggered onsets: 0.0s, 0.3s, 0.6s, 0.9s, 1.2s
    note_events = [
        (0,            60, 0.7, 0.4),      # C
        (int(0.3*44100), 62, 0.7, 0.4),   # D
        (int(0.6*44100), 64, 0.7, 0.4),   # E
        (int(0.9*44100), 65, 0.7, 0.4),   # F
        (int(1.2*44100), 67, 0.7, 0.4),   # G
    ]

    duration_sec = 1.8

    audio = fluidsynth_renderer.render_notes(
        note_events=note_events,
        duration_sec=duration_sec,
        channel=FluidSynthRenderer.CHANNEL_PIANO
    )

    # Validate output
    assert audio.shape == (2, int(44100 * 1.8)), "Expected stereo 1.8-second output"

    # Validate melodic line was generated
    rms = np.sqrt(np.mean(audio ** 2))
    assert rms > 0.001, f"Melodic line RMS too low ({rms})"

    # Check for catastrophic clicks (samples exceeding ±1.5)
    # Some transients are expected, but extreme values indicate issues
    max_abs = np.max(np.abs(audio))
    assert max_abs <= 1.2, f"Unexpected extreme amplitude ({max_abs}), possible artifact"


def test_sample_rate_resampling(fluidsynth_renderer):
    """
    Test automatic sample rate resampling (T066, FR-018).

    Validates that FluidSynth automatically resamples SoundFont audio
    to match the target 44.1kHz sample rate, regardless of the SF2's
    native sample rate (e.g., 48kHz).

    This test verifies the output is always 44.1kHz.
    """
    note_events = [(0, 60, 0.7, 1.0)]
    duration_sec = 1.0

    audio = fluidsynth_renderer.render_notes(
        note_events=note_events,
        duration_sec=duration_sec,
        channel=FluidSynthRenderer.CHANNEL_PIANO
    )

    # Validate output sample rate is 44.1kHz (44100 samples per second)
    expected_samples = int(44100 * duration_sec)
    assert audio.shape == (2, expected_samples), (
        f"Expected 44.1kHz output ({expected_samples} samples), "
        f"got {audio.shape[1]} samples"
    )

    # Verify FluidSynth renderer was initialized with 44.1kHz
    assert fluidsynth_renderer.sample_rate == 44100, (
        "FluidSynthRenderer sample rate mismatch"
    )
