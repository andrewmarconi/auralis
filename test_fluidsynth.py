#!/usr/bin/env python3
"""Simple test to verify FluidSynth basic rendering works."""

import numpy as np
import fluidsynth

# Initialize FluidSynth
fs = fluidsynth.Synth(samplerate=44100)
fs.start()

# Load SoundFont
sfid = fs.sfload("soundfonts/soundfonts/FluidR3_GM.sf2")
fs.program_select(0, sfid, 0, 0)  # Channel 0, Piano

print("Playing C major chord (C-E-G)...")

# Play a simple C major chord
fs.noteon(0, 60, 80)  # C4
fs.noteon(0, 64, 80)  # E4
fs.noteon(0, 67, 80)  # G4

# Render 2 seconds of audio
sample_rate = 44100
duration_sec = 2.0
num_samples = int(sample_rate * duration_sec)

print(f"Rendering {num_samples} samples...")

# Render audio
samples = fs.get_samples(num_samples)
audio = np.array(samples, dtype=np.float32)

print(f"Audio shape: {audio.shape}")
print(f"Audio range: [{audio.min():.4f}, {audio.max():.4f}]")
print(f"Audio mean: {audio.mean():.4f}")
print(f"Audio std: {audio.std():.4f}")

# Check if audio is silence
if np.abs(audio).max() < 0.0001:
    print("WARNING: Audio is silence!")
else:
    print("SUCCESS: Audio contains signal!")

# Turn off notes
fs.noteoff(0, 60)
fs.noteoff(0, 64)
fs.noteoff(0, 67)

# Cleanup
fs.delete()

print("Test complete.")
