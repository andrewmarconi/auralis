# Quickstart Testing Guide: FluidSynth Integration

**Feature**: FluidSynth Sample-Based Instrument Synthesis
**Branch**: `004-fluidsynth-integration`
**Purpose**: Practical testing scenarios to validate FluidSynth integration functionality

## Prerequisites

Before testing, ensure the following setup is complete:

### 1. System Dependencies

```bash
# macOS
brew install fluidsynth

# Ubuntu/Debian
sudo apt-get install fluidsynth libfluidsynth-dev
```

### 2. Python Dependencies

```bash
# Activate environment
source .venv/bin/activate

# Install pyfluidsynth
uv add pyfluidsynth
```

### 3. SoundFont Files

```bash
# Download FluidR3_GM SoundFont
wget http://www.musescore.org/download/fluid-soundfont.tar.gz
tar -xzf fluid-soundfont.tar.gz

# Move to soundfonts directory
mkdir -p soundfonts
mv FluidR3_GM.sf2 soundfonts/

# Verify file size (~142MB)
ls -lh soundfonts/FluidR3_GM.sf2
```

Expected output:
```
-rw-r--r--  1 user  staff   142M  soundfonts/FluidR3_GM.sf2
```

---

## Test Scenario 1: SoundFont Loading & Validation (FR-016)

**Goal**: Verify that the server successfully loads and validates required SoundFont files at startup, and fails fast if files are missing or corrupted.

### Test 1.1: Successful Startup with Valid SoundFont

```bash
# Ensure SoundFont exists
ls soundfonts/FluidR3_GM.sf2

# Start server
uvicorn server.main:app --reload
```

**Expected Output**:
```
INFO: Validated SoundFont: FluidR3_GM.sf2 (142.3MB)
INFO: Loaded SoundFont: soundfonts/FluidR3_GM.sf2 (sfid=1)
INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**Success Criteria**:
- ✅ Server starts without errors
- ✅ Log shows SoundFont validation message
- ✅ Log shows SoundFont load confirmation with `sfid` number
- ✅ No "SoundFont validation failed" error

---

### Test 1.2: Fail-Fast on Missing SoundFont

```bash
# Temporarily move SoundFont file
mv soundfonts/FluidR3_GM.sf2 soundfonts/FluidR3_GM.sf2.bak

# Attempt to start server
uvicorn server.main:app --reload
```

**Expected Output**:
```
ERROR: SoundFont validation failed:
Required SoundFont missing: soundfonts/FluidR3_GM.sf2
Download from: http://www.musescore.org/download/fluid-soundfont.tar.gz
ERROR: Server startup aborted. Fix SoundFont issues and restart.
```

**Expected Behavior**:
- ❌ Server refuses to start
- ❌ Process exits with status code 1
- ✅ Clear error message indicating missing file
- ✅ Download instructions provided

**Cleanup**:
```bash
# Restore SoundFont
mv soundfonts/FluidR3_GM.sf2.bak soundfonts/FluidR3_GM.sf2
```

---

### Test 1.3: Fail-Fast on Corrupted SoundFont

```bash
# Create corrupted SoundFont (truncated file <100MB)
echo "corrupted" > soundfonts/FluidR3_GM.sf2

# Attempt to start server
uvicorn server.main:app --reload
```

**Expected Output**:
```
ERROR: SoundFont validation failed:
SoundFont appears corrupted (too small): soundfonts/FluidR3_GM.sf2
Size: 0.0MB (expected ~142MB)
Re-download the file.
```

**Expected Behavior**:
- ❌ Server refuses to start
- ✅ File size check detects corruption
- ✅ Re-download instructions provided

**Cleanup**:
```bash
# Restore valid SoundFont
wget http://www.musescore.org/download/fluid-soundfont.tar.gz
tar -xzf fluid-soundfont.tar.gz
mv FluidR3_GM.sf2 soundfonts/
```

---

## Test Scenario 2: Realistic Instrument Timbres (FR-001, FR-002, FR-003)

**Goal**: Verify that piano, pad, and choir voices produce realistic, sample-based timbres instead of synthetic oscillator sounds.

### Test 2.1: Acoustic Grand Piano Melody (FR-001)

```bash
# Start server
uvicorn server.main:app --reload

# In another terminal, connect to stream
curl -N -H "Connection: Upgrade" \
     -H "Upgrade: websocket" \
     -H "Sec-WebSocket-Version: 13" \
     -H "Sec-WebSocket-Key: $(openssl rand -base64 16)" \
     http://localhost:8000/stream
```

**Manual Listening Test**:
1. Open browser to `http://localhost:8000` (if UI exists) OR connect via WebSocket client
2. Listen for melodic piano notes in the generated ambient music
3. Compare sound quality to previous oscillator-based version (if available)

**Success Criteria (Subjective)**:
- ✅ Piano notes sound like an **acoustic grand piano**, not a synthesizer
- ✅ Different pitches (low/mid/high register) sound realistic
- ✅ Velocity variations (soft/loud notes) sound natural
- ✅ Sustained notes have natural decay/release
- ✅ No synthetic "beeping" or "oscillator" character

**Acceptance**: 80%+ of listeners prefer this over oscillator version (SC-001)

---

### Test 2.2: Polysynth Pad Textures (FR-002)

**Manual Listening Test**:
1. Listen for chord pad textures (sustained chords in background)
2. Focus on the harmonic richness and warmth

**Success Criteria (Subjective)**:
- ✅ Pads sound **warm**, **spacious**, and **enveloping**
- ✅ Chord transitions are smooth, no clicks or pops
- ✅ Multiple notes blend cohesively (polyphonic rendering)
- ✅ Textures are not "thin" or "harsh"
- ✅ Sustained pads maintain character over 5+ seconds

---

### Test 2.3: Choir Swells (FR-003)

**Manual Listening Test**:
1. Listen for swell effects (occasional ambient vocal textures)
2. Identify choir-like "Aahs" or "Oohs" sounds

**Success Criteria (Subjective)**:
- ✅ Swells sound like **human choir voices**, not oscillators
- ✅ Choir textures blend with pads without muddiness
- ✅ Swells add atmospheric depth to the music
- ✅ No synthetic or robotic character

---

## Test Scenario 3: Real-Time Performance (FR-009, SC-002, SC-003)

**Goal**: Verify that FluidSynth integration maintains <100ms latency and real-time synthesis performance.

### Test 3.1: Latency Measurement

```bash
# Run performance benchmarks
pytest tests/performance/test_real_time_constraints.py -v
```

**Expected Output**:
```
tests/performance/test_real_time_constraints.py::test_synthesis_latency_under_100ms PASSED
tests/performance/test_real_time_constraints.py::test_fluidsynth_rendering_latency PASSED
tests/performance/test_real_time_constraints.py::test_hybrid_synthesis_latency PASSED
```

**Success Criteria**:
- ✅ Total synthesis latency: <100ms (SC-002)
- ✅ FluidSynth rendering: <50ms
- ✅ Hybrid mixing overhead: <10ms
- ✅ Consistent performance across 10+ test runs

---

### Test 3.2: Real-Time Factor Benchmark (SC-003)

```python
# tests/performance/test_fluidsynth_performance.py

import time
from server.synthesis_engine import SynthesisEngine

def test_real_time_factor_10x():
    """Verify 16-second phrase renders in <1.6 seconds (10× real-time)."""
    engine = SynthesisEngine(device='cpu')

    # Generate test phrase (16 seconds)
    chords = generate_test_chords(duration_sec=16)
    melody = generate_test_melody(duration_sec=16)

    # Measure rendering time
    start = time.perf_counter()
    audio = engine.render_phrase(chords, melody, duration_sec=16)
    elapsed = time.perf_counter() - start

    # Verify real-time factor
    real_time_factor = 16.0 / elapsed
    assert real_time_factor >= 10.0, f"Real-time factor {real_time_factor:.1f}× < 10×"
    assert elapsed < 1.6, f"Rendering took {elapsed:.2f}s > 1.6s"
```

**Run Test**:
```bash
pytest tests/performance/test_fluidsynth_performance.py::test_real_time_factor_10x -v
```

**Success Criteria**:
- ✅ 16-second phrase renders in <1.6 seconds (SC-003)
- ✅ Real-time factor ≥10× on target hardware

---

### Test 3.3: Concurrent Stream Load Test (SC-004)

```bash
# Run concurrent stream test
pytest tests/integration/test_websocket_streaming.py::test_concurrent_streams -v
```

**Test Code** (example):
```python
import pytest
import asyncio

@pytest.mark.asyncio
async def test_concurrent_streams():
    """Verify 10+ concurrent streams without dropouts."""
    # Simulate 10 concurrent WebSocket connections
    tasks = [stream_audio_for_duration(duration_sec=30) for _ in range(10)]

    # Run all streams concurrently
    results = await asyncio.gather(*tasks)

    # Verify no dropouts
    for i, result in enumerate(results):
        assert result["dropouts"] == 0, f"Stream {i} had {result['dropouts']} dropouts"
        assert result["avg_latency_ms"] < 100, f"Stream {i} latency {result['avg_latency_ms']}ms"
```

**Success Criteria**:
- ✅ 10+ concurrent streams supported (SC-004)
- ✅ Zero audio dropouts across all streams
- ✅ Average latency <100ms per stream

---

## Test Scenario 4: Polyphonic Playback & Voice Stealing (FR-004, FR-017, SC-007, SC-008)

**Goal**: Verify that the system handles 15+ simultaneous notes with graceful voice stealing when limits are exceeded.

### Test 4.1: Polyphonic Chord Rendering

```python
# tests/integration/test_polyphony_voice_stealing.py

def test_15_simultaneous_notes():
    """Verify system renders 15 simultaneous notes cleanly."""
    engine = SynthesisEngine(device='cpu')

    # Create chord with 15 simultaneous notes (onset=0, all start together)
    notes = [(0, 60 + i, 0.7, 2.0) for i in range(15)]

    # Render
    audio = engine.fluidsynth_renderer.render_notes(notes, duration_samples=88200)

    # Verify output
    assert audio.shape == (2, 88200)  # Stereo, 2 seconds @ 44.1kHz
    assert np.max(np.abs(audio)) > 0.01  # Audio generated
    # No assertion on exact voice count (FluidSynth manages internally)
```

**Run Test**:
```bash
pytest tests/integration/test_polyphony_voice_stealing.py::test_15_simultaneous_notes -v
```

**Success Criteria**:
- ✅ 15 simultaneous notes render without errors (FR-004, SC-007)
- ✅ Audio output is non-zero (notes were rendered)
- ✅ No crashes or exceptions

---

### Test 4.2: Voice Stealing Behavior (>20 notes)

```python
def test_voice_stealing_at_polyphony_limit():
    """Test graceful voice stealing when exceeding 20-voice limit."""
    engine = SynthesisEngine(device='cpu')

    # Generate 25 simultaneous notes (exceeds 20-voice limit)
    notes = [(0, 60 + i, 0.7, 2.0) for i in range(25)]

    # Should render without crashing (FluidSynth steals oldest voices)
    audio = engine.fluidsynth_renderer.render_notes(notes, duration_samples=88200)

    assert audio.shape == (2, 88200)
    assert np.max(np.abs(audio)) > 0.01
    # Verify no exception raised (graceful voice stealing)
```

**Success Criteria**:
- ✅ No crash when exceeding polyphony limit (FR-017)
- ✅ Audio still generated (oldest notes stolen automatically)
- ✅ FluidSynth handles voice stealing internally

---

### Test 4.3: Click-Free Voice Stealing (SC-008)

```python
def test_no_clicks_during_voice_stealing():
    """Verify voice stealing does not produce audible clicks."""
    engine = SynthesisEngine(device='cpu')

    # Generate rapid note sequence with >20 overlapping notes
    notes = [(i * 2205, 60, 0.8, 1.0) for i in range(30)]  # 50ms spacing

    audio = engine.fluidsynth_renderer.render_notes(notes, duration_samples=88200)

    # Detect clicks by analyzing sudden amplitude jumps
    diff = np.abs(np.diff(audio[0]))  # Left channel
    max_jump = np.max(diff)

    # Clicks show as jumps >0.1; smooth voice stealing should keep jumps <0.05
    assert max_jump < 0.05, f"Detected click: max jump {max_jump:.4f}"
```

**Success Criteria**:
- ✅ No clicks detected (max amplitude jump <0.05) (SC-008)
- ✅ Smooth voice transitions during stealing
- ✅ FluidSynth's fast release works correctly

---

## Test Scenario 5: Hybrid Synthesis (FluidSynth + PyTorch) (FR-006, FR-007)

**Goal**: Verify that FluidSynth (piano/pads/choir) and PyTorch (kicks) are correctly mixed in the final output.

### Test 5.1: Hybrid Mix Verification

```python
def test_hybrid_synthesis_mix():
    """Verify FluidSynth and PyTorch sources are both present in mix."""
    engine = SynthesisEngine(device='cpu')

    # Create phrase with:
    # - Piano melody (FluidSynth)
    # - Kick percussion (PyTorch)
    chords = []  # No pads
    melody = [(0, 60, 0.7, 1.0), (22050, 64, 0.7, 1.0)]  # Two piano notes
    kicks = [(0, 0.8), (44100, 0.8)]  # Two kick hits

    audio = engine.render_phrase(chords, melody, kicks, duration_sec=2.0)

    # Analyze audio energy
    # Should have both piano (sustained) and kick (transient) energy
    assert np.max(np.abs(audio)) > 0.3  # Overall audio present
    # Further analysis could decompose frequency content
```

**Success Criteria**:
- ✅ Both FluidSynth and PyTorch sources present in final mix (FR-006)
- ✅ Mix weights applied correctly (40% pads, 50% melody, 30% kicks, FR-007)
- ✅ No source completely missing from output

---

### Test 5.2: Mixing Weights Validation

```python
def test_mixing_weights_applied():
    """Verify mix weights affect relative loudness."""
    engine = SynthesisEngine(device='cpu')

    # Render with default weights
    audio_default = engine.render_phrase(chords, melody, kicks, duration_sec=2.0,
                                          mix_weights={'pads': 0.4, 'melody': 0.5, 'kicks': 0.3})

    # Render with boosted melody
    audio_loud_melody = engine.render_phrase(chords, melody, kicks, duration_sec=2.0,
                                              mix_weights={'pads': 0.2, 'melody': 0.8, 'kicks': 0.1})

    # Melody should be louder in second mix
    melody_rms_default = np.sqrt(np.mean(audio_default ** 2))
    melody_rms_loud = np.sqrt(np.mean(audio_loud_melody ** 2))

    assert melody_rms_loud > melody_rms_default, "Melody not louder with increased weight"
```

---

## Test Scenario 6: Sample Rate Resampling (FR-018)

**Goal**: Verify that SoundFont files with non-44.1kHz sample rates are automatically resampled to 44.1kHz by FluidSynth.

### Test 6.1: Load 48kHz SoundFont

```python
def test_48khz_soundfont_auto_resamples():
    """Verify FluidSynth auto-resamples 48kHz SoundFont to 44.1kHz."""
    # Load SoundFont with 48kHz samples (if available)
    synth = fluidsynth.Synth(samplerate=44100)
    sfid = synth.sfload("soundfonts/salamander_48khz.sf2")  # Example

    # Render audio
    synth.program_select(0, sfid, 0, 0)
    synth.noteon(0, 60, 100)
    audio = synth.get_samples(4410)  # 100ms @ 44.1kHz

    # Verify output is 44.1kHz (4410 samples for 100ms)
    assert len(audio) == 4410 * 2, "Output sample count incorrect (resampling failed?)"
```

**Expected Behavior**:
- ✅ FluidSynth automatically resamples 48kHz samples to 44.1kHz (FR-018)
- ✅ No manual resampling code required
- ✅ Output always matches target sample rate

---

## Test Scenario 7: Clipping Prevention (FR-015, SC-006)

**Goal**: Verify that soft clipping/limiting prevents audio distortion when mixing multiple layers.

### Test 7.1: No Clipping in Dense Mix

```python
def test_no_clipping_in_dense_mix():
    """Verify soft clipping prevents samples from exceeding ±0.99."""
    engine = SynthesisEngine(device='cpu')

    # Create very dense phrase (many loud notes)
    chords = generate_dense_chords(num_chords=16, velocity=0.9)
    melody = generate_dense_melody(num_notes=50, velocity=0.9)
    kicks = generate_dense_kicks(num_kicks=20, velocity=0.9)

    audio = engine.render_phrase(chords, melody, kicks, duration_sec=16.0)

    # Check for clipping
    max_sample = np.max(np.abs(audio))
    assert max_sample < 0.99, f"Clipping detected: max sample {max_sample}"
```

**Success Criteria**:
- ✅ No samples exceed ±0.99 (SC-006)
- ✅ Soft clipping applied correctly (FR-015)
- ✅ Audio quality preserved (no harsh distortion)

---

## Manual Testing Checklist

After automated tests pass, perform manual validation:

### Listening Quality

- [ ] Piano sounds realistic (not synthetic)
- [ ] Pads are warm and spacious
- [ ] Choir swells add atmospheric depth
- [ ] No audible clicks or pops during voice stealing
- [ ] Smooth transitions between chords
- [ ] Dynamic variation (soft/loud notes) sounds natural
- [ ] Overall mix is balanced (no layer too loud/quiet)

### Performance

- [ ] No audio stuttering or dropouts during playback
- [ ] Latency feels responsive (<100ms subjectively)
- [ ] Concurrent streams work smoothly (test with multiple browser tabs)
- [ ] CPU usage reasonable (<50% on target hardware)
- [ ] Memory usage stays under 500MB

### Startup & Configuration

- [ ] Server starts quickly (<5 seconds with SoundFont loading)
- [ ] Server refuses to start if SoundFont missing
- [ ] Error messages are clear and actionable
- [ ] Optional environment variables work (AURALIS_SOUNDFONT_DIR)

---

## Troubleshooting

### Issue: "SoundFont validation failed: File not found"

**Solution**:
```bash
# Download SoundFont
wget http://www.musescore.org/download/fluid-soundfont.tar.gz
tar -xzf fluid-soundfont.tar.gz
mv FluidR3_GM.sf2 soundfonts/
```

### Issue: "FluidSynth failed to load SoundFont"

**Solution**:
- Verify file is not corrupted: `ls -lh soundfonts/FluidR3_GM.sf2` (should be ~142MB)
- Re-download if corrupted
- Check file permissions: `chmod 644 soundfonts/FluidR3_GM.sf2`

### Issue: "Import Error: No module named 'fluidsynth'"

**Solution**:
```bash
# Install system FluidSynth library
brew install fluidsynth  # macOS
# OR
sudo apt-get install libfluidsynth-dev  # Linux

# Install Python bindings
uv add pyfluidsynth
```

### Issue: Audio clicks/pops during playback

**Diagnosis**:
- Check CPU usage (may be overloaded)
- Verify polyphony limit not exceeded excessively
- Test with simpler phrase (fewer notes)

**Solution**:
- Reduce polyphony limit: `synth.setting('synth.polyphony', '16')`
- Use faster interpolation: `synth.setting('synth.interpolation', 'linear')`

---

## Success Criteria Summary

| Test Scenario | Success Criteria | Status |
|---------------|------------------|--------|
| SoundFont Loading | Server starts with valid SF2, fails fast if missing | [ ] |
| Realistic Piano | Listeners prefer sample-based over oscillators (80%+) | [ ] |
| Realistic Pads | Warm, spacious textures | [ ] |
| Realistic Choir | Choir-like swells | [ ] |
| Latency | <100ms total, <50ms FluidSynth rendering | [ ] |
| Real-Time Factor | 16s phrase in <1.6s (10× real-time) | [ ] |
| Concurrent Streams | 10+ streams, zero dropouts | [ ] |
| Polyphony | 15+ simultaneous notes supported | [ ] |
| Voice Stealing | Graceful handling of >20 notes, no clicks | [ ] |
| Hybrid Mix | FluidSynth + PyTorch both present in output | [ ] |
| Resampling | 48kHz SF2 auto-resamples to 44.1kHz | [ ] |
| No Clipping | Max sample <0.99, soft limiting effective | [ ] |

---

**Quickstart Status**: ✅ COMPLETE
**Ready for Implementation**: YES
**Next Step**: Update agent context and begin implementation
