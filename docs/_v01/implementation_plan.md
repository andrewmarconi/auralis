# Phase-by-Phase Implementation Plan

## Overview

A structured 6-phase rollout from MVP to production-ready system. Each phase builds on the previous, allowing early validation and iteration.

---

## Phase 1: MVP Foundation (Weeks 1–2)

**Goal**: End-to-end streaming with basic generation and synthesis

### Deliverables

- [ ] Markov chord generation (bigram chains, hard-coded progressions)
- [ ] Constraint-based melody generator (rule-based, no ML)
- [ ] torchsynth monophonic synth (single voice lead + pad)
- [ ] Ring buffer (server-side)
- [ ] FastAPI WebSocket server with basic streaming
- [ ] Web client with Web Audio API playback
- [ ] End-to-end latency < 800ms

### Tasks

#### Task 1.1: Chord Progression Generator
**Duration**: 2–3 days

```python
# composition/chord_generator.py
import numpy as np

class ChordProgressionGenerator:
    """Markov chain–based chord progression generator."""
    
    def __init__(self):
        # Hard-coded transition matrix for ambient progressions
        # State space: [i, iv, V, VI, III]
        # Transition matrix (bigram): P[current][next]
        self.transition_matrix = np.array([
            [0.1,  0.3,  0.2,  0.3,  0.1],  # From i
            [0.2,  0.1,  0.3,  0.2,  0.2],  # From iv
            [0.1,  0.1,  0.2,  0.4,  0.2],  # From V
            [0.3,  0.2,  0.1,  0.1,  0.3],  # From VI
            [0.2,  0.3,  0.2,  0.2,  0.1],  # From III
        ])
        
        self.chords = ['i', 'iv', 'V', 'VI', 'III']
        self.root_midi = 57  # A3
    
    def generate(self, length_bars: int = 8) -> list:
        """Generate chord progression."""
        progression = []
        current_idx = 0  # Start on i
        
        for _ in range(length_bars):
            chord = self.chords[current_idx]
            progression.append(chord)
            
            # Sample next chord
            next_idx = np.random.choice(
                len(self.chords),
                p=self.transition_matrix[current_idx]
            )
            current_idx = next_idx
        
        return progression

# Test
gen = ChordProgressionGenerator()
print(gen.generate())  # ['i', 'VI', 'III', 'VI', 'i', 'iv', 'V', 'i']
```

**Acceptance Criteria**:
- ✓ Generates 8-bar progressions consistently
- ✓ Feels "ambient" (slower harmonic movement)
- ✓ No crashes on repeated calls

---

#### Task 1.2: Constraint-Based Melody Generator
**Duration**: 3–4 days

```python
# composition/melody_generator.py
import numpy as np

class ConstrainedMelodyGenerator:
    """Generate melodies that fit chord progressions."""
    
    def __init__(self):
        self.scale_intervals = {
            'aeolian': [0, 2, 3, 5, 7, 8, 10],  # A minor scale
        }
        self.default_scale = 'aeolian'
    
    def get_chord_tones(self, root_midi: int, chord_type: str) -> list:
        """Return MIDI notes for a chord."""
        intervals = {
            'i': [0, 3, 7],     # Minor triad
            'iv': [0, 3, 7],    # Minor triad
            'V': [0, 4, 7],     # Major triad
            'VI': [0, 3, 7],    # Minor triad
            'III': [0, 4, 7],   # Major triad
        }
        
        notes = []
        for interval in intervals.get(chord_type, [0, 4, 7]):
            notes.append(root_midi + interval)
        
        return notes
    
    def generate(self, chords: list, root_midi: int = 60, bars: int = 8) -> list:
        """
        Generate melody fitting chord progression.
        
        Args:
            chords: List of chord names ['i', 'VI', ...]
            root_midi: MIDI note of root
            bars: Number of bars
        
        Returns:
            List of (onset_sec, pitch_midi, velocity, duration_sec)
        """
        melody = []
        notes_per_bar = 2  # Sparse: 2 notes per bar on average
        
        for bar_idx, chord in enumerate(chords):
            chord_tones = self.get_chord_tones(root_midi, chord)
            
            # Generate 1–2 notes per bar
            num_notes = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
            
            for note_offset in range(num_notes):
                # Choose pitch (mostly chord tones)
                if np.random.random() < 0.8:
                    pitch = np.random.choice(chord_tones)
                else:
                    # Passing tone (scale tone)
                    scale = self.scale_intervals[self.default_scale]
                    scale_pitches = [root_midi + interval for interval in scale]
                    pitch = np.random.choice(scale_pitches)
                
                # Onset within bar (seconds)
                bar_duration_sec = (60 / 70) * 4  # 70 BPM = ~3.43 sec/bar
                onset_sec = bar_idx * bar_duration_sec + (note_offset * bar_duration_sec / num_notes)
                
                # Duration: long sustains for ambient
                duration_sec = np.random.choice([0.5, 1.0, 1.5, 2.0], p=[0.2, 0.3, 0.3, 0.2])
                
                # Velocity
                velocity = np.random.uniform(0.6, 0.8)
                
                melody.append((onset_sec, pitch, velocity, duration_sec))
        
        return melody

# Test
gen = ConstrainedMelodyGenerator()
chords = ['i', 'VI', 'i', 'VI']
melody = gen.generate(chords, root_midi=60, bars=4)
for m in melody:
    print(f"  {m[1]} (C={m[1] - 60:+d})")  # Show relative to root
```

**Acceptance Criteria**:
- ✓ Melodies fit chord progressions (no clashing notes)
- ✓ Sparse, long-sustained notes (ambient style)
- ✓ 4–8 bar phrases generate in < 100ms

---

#### Task 1.3: Simple Synthesis (torchsynth)
**Duration**: 4–5 days

Build a minimal synth: pad + lead voice.

```python
# synthesis/simple_synth.py
import torch
import numpy as np

class SimpleAmbientSynth:
    """Minimal torchsynth-style synthesizer."""
    
    def __init__(self, sr: int = 44100, device: str = 'mps'):
        self.sr = sr
        self.device = device
    
    def render_chord(self, midi_note: int, duration_sec: float, amp: float = 0.5) -> np.ndarray:
        """Render a single note (pad voice)."""
        num_samples = int(duration_sec * self.sr)
        t = np.arange(num_samples) / self.sr
        
        # Frequency
        freq = 440 * (2 ** ((midi_note - 69) / 12))
        
        # Simple sine oscillator
        phase = 2 * np.pi * freq * t
        osc = np.sin(phase)
        
        # ADSR envelope (slow for pad)
        attack = int(0.5 * self.sr)
        release = int(1.0 * self.sr)
        sustain_start = attack + int(0.5 * self.sr)
        sustain_end = max(sustain_start, num_samples - release)
        
        env = np.ones(num_samples)
        env[:attack] = np.linspace(0, 1, attack)
        env[sustain_end:] = np.linspace(1, 0, release)
        
        return (osc * env * amp).astype(np.float32)
    
    def render_phrase(self, chords: list, melody: list, duration_sec: float) -> np.ndarray:
        """Render full phrase: chords + melody."""
        num_samples = int(duration_sec * self.sr)
        audio = np.zeros((2, num_samples), dtype=np.float32)
        
        # Render chords (simplified: just root note)
        bpm = 70
        bar_duration_sec = (60 / bpm) * 4
        
        for bar_idx, chord in enumerate(chords):
            onset_sample = int(bar_idx * bar_duration_sec * self.sr)
            
            # Map chord to MIDI note
            chord_midi = {'i': 57, 'VI': 62, 'iv': 60, 'V': 64, 'III': 64}
            midi_note = chord_midi.get(chord, 57)
            
            # Render chord tone
            chord_audio = self.render_chord(midi_note, bar_duration_sec, amp=0.3)
            audio[0, onset_sample:onset_sample + len(chord_audio)] += chord_audio[:num_samples - onset_sample]
            audio[1, onset_sample:onset_sample + len(chord_audio)] += chord_audio[:num_samples - onset_sample]
        
        # Render melody
        for onset_sec, midi_note, velocity, duration_sec in melody:
            onset_sample = int(onset_sec * self.sr)
            
            melody_audio = self.render_chord(midi_note, duration_sec, amp=velocity)
            end_sample = min(onset_sample + len(melody_audio), num_samples)
            audio[0, onset_sample:end_sample] += melody_audio[:end_sample - onset_sample]
            audio[1, onset_sample:end_sample] += melody_audio[:end_sample - onset_sample]
        
        # Soft limiter
        audio = np.tanh(audio)
        
        return audio

# Test
synth = SimpleAmbientSynth()
chords = ['i', 'VI', 'i', 'VI']
melody = [(0, 60, 0.7, 1.0), (1.7, 62, 0.7, 2.0)]
audio = synth.render_phrase(chords, melody, duration_sec=13.7)
print(f"Audio shape: {audio.shape}, range: [{audio.min():.2f}, {audio.max():.2f}]")
```

**Acceptance Criteria**:
- ✓ Renders 8-bar phrase in < 5 seconds
- ✓ No NaNs or clipping
- ✓ Stereo output (2 channels)

---

#### Task 1.4: Ring Buffer
**Duration**: 2 days

See `implementation_strategies.md` → Section 2.1

**Acceptance Criteria**:
- ✓ Write/read in separate threads without locks on data
- ✓ Detects buffer over/underflow
- ✓ Can handle continuous writes + reads

---

#### Task 1.5: FastAPI WebSocket Server
**Duration**: 3 days

Minimal server: accept WS connection, stream audio.

```python
# server/main.py
from fastapi import FastAPI, WebSocket
import asyncio
import base64
import numpy as np

app = FastAPI()

# Global state
ring_buffer = None  # Will initialize on startup
synth = None
composition_engine = None

@app.on_event('startup')
async def startup():
    global ring_buffer, synth, composition_engine
    from ring_buffer import RingBuffer
    from synthesis.simple_synth import SimpleAmbientSynth
    from composition.composition import CompositionEngine
    
    ring_buffer = RingBuffer(capacity_sec=2.0)
    synth = SimpleAmbientSynth()
    composition_engine = CompositionEngine()
    
    # Start background synthesis
    asyncio.create_task(synthesis_loop())

async def synthesis_loop():
    """Continuously render phrases."""
    while True:
        # Get next phrase from composition engine
        phrase = await composition_engine.get_next_phrase()
        
        # Render to audio
        audio = synth.render_phrase(
            chords=phrase['chords'],
            melody=phrase['melody'],
            duration_sec=13.7
        )
        
        # Write to ring buffer
        ring_buffer.write(audio)
        
        await asyncio.sleep(0.1)

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            # Read from ring buffer
            chunk_samples = int(0.1 * 44100)  # 100ms
            chunk = ring_buffer.read(chunk_samples)
            
            # Encode
            chunk_int16 = (np.clip(chunk, -1, 1) * 32767).astype(np.int16)
            encoded = base64.b64encode(chunk_int16.tobytes()).decode()
            
            # Send
            await websocket.send_json({
                'type': 'audio',
                'data': encoded,
            })
            
            await asyncio.sleep(0.1)
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
```

**Acceptance Criteria**:
- ✓ Server starts without errors
- ✓ WebSocket accepts connection
- ✓ Sends audio chunks to client every ~100ms

---

#### Task 1.6: Web Client (HTML + JavaScript)
**Duration**: 2–3 days

Simple Web Audio API client.

```html
<!-- client/index.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Ambient Music Streamer</title>
    <script src="audio_client.js"></script>
</head>
<body>
    <h1>Ambient Music Streamer</h1>
    <button id="connectBtn" onclick="startStreaming()">Connect</button>
    <button id="disconnectBtn" onclick="stopStreaming()" disabled>Disconnect</button>
    
    <p>Status: <span id="status">Disconnected</span></p>
    <p>Buffer Depth: <span id="bufferDepth">0</span> ms</p>
    
    <script>
        let client = null;
        
        async function startStreaming() {
            client = new AmbientAudioClient('ws://localhost:8000/ws/stream');
            await client.connect();
            
            document.getElementById('status').textContent = 'Connected';
            document.getElementById('connectBtn').disabled = true;
            document.getElementById('disconnectBtn').disabled = false;
            
            // Update status display
            setInterval(() => {
                const status = client.getStatus();
                document.getElementById('bufferDepth').textContent = 
                    status.buffer_depth_ms.toFixed(0);
            }, 1000);
        }
        
        function stopStreaming() {
            if (client) client.disconnect();
            document.getElementById('status').textContent = 'Disconnected';
            document.getElementById('connectBtn').disabled = false;
            document.getElementById('disconnectBtn').disabled = true;
        }
    </script>
</body>
</html>
```

See `implementation_strategies.md` → Section 3.2 for full JavaScript client code.

**Acceptance Criteria**:
- ✓ Connects to server via WebSocket
- ✓ Plays audio through Web Audio API
- ✓ No browser console errors

---

### Phase 1 Success Criteria

- [ ] Chord generator produces coherent 8-bar progressions
- [ ] Melody generator respects harmonic constraints
- [ ] Synthesis renders 8 bars in < 10 seconds
- [ ] Server accepts WebSocket, streams audio
- [ ] Client connects and plays audio
- **End-to-end latency**: ~500–800ms (acceptable for MVP)

### Phase 1 Testing Checklist

```
□ Unit tests for ChordProgressionGenerator
  □ Test determinism (seed for reproducibility)
  □ Test no invalid state transitions
  
□ Unit tests for ConstrainedMelodyGenerator
  □ Test all returned notes fit chord/scale
  □ Test returns correct number of notes
  
□ Integration test: synthesis + streaming
  □ Render 30-second composition without crashes
  □ Monitor CPU usage (should be < 20% for synthesis)
  □ Verify no audio dropouts over 5-minute stream
  
□ Client tests
  □ Browser console shows no errors
  □ Audio plays without glitches for 1 minute
  □ Disconnect/reconnect works
```

---

## Phase 2: Enhanced Generation & Controls (Weeks 3–4)

**Goal**: Add generation quality, user control, and percussion

### Deliverables

- [ ] Transformer-based melody generator (fine-tuned DistilGPT-2)
- [ ] Percussion/texture generator (rule-based)
- [ ] Control API: key, BPM, intensity
- [ ] Client UI for controls
- [ ] Adaptive client-side buffering
- [ ] Improved error handling + logging

### Key Tasks

#### Task 2.1: Transformer Melody Generator
**Duration**: 4–5 days

Fine-tune DistilGPT-2 on ambient melody MIDI.

```python
# generation/transformer_melody.py
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

class TransformerMelodyGenerator:
    """Fine-tuned GPT-2 for melody generation."""
    
    def __init__(self, model_name: str = 'distilgpt2', device: str = 'mps'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        self.device = device
    
    def generate_melody(self, context: str, num_tokens: int = 32) -> list:
        """
        Generate melody tokens given a context chord progression.
        
        Args:
            context: e.g., "C C C C G G G G"  (chord names)
            num_tokens: Number of tokens to generate
        
        Returns:
            List of MIDI notes
        """
        input_ids = self.tokenizer.encode(context, return_tensors='pt').to(self.device)
        
        output = self.model.generate(
            input_ids,
            max_length=input_ids.shape[1] + num_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            num_beams=3,
        )
        
        generated_text = self.tokenizer.decode(output[0])
        # Parse generated tokens back to MIDI notes
        return self._parse_tokens(generated_text)
    
    def _parse_tokens(self, text: str) -> list:
        """Parse text tokens back to MIDI notes."""
        # TODO: implement parsing
        return []
```

**Note**: Full training pipeline requires ambient MIDI dataset. For MVP, use pre-trained or hard-coded fallback.

**Acceptance Criteria**:
- ✓ Generates 8-bar melodies in < 500ms
- ✓ Melodies fit chord context (validate post-generation)
- ✓ Gracefully falls back to constraint generator on error

---

#### Task 2.2: Percussion Generator
**Duration**: 2–3 days

```python
# composition/percussion_generator.py
import numpy as np

class PercussionGenerator:
    """Generate sparse percussion/texture events."""
    
    def generate(self, duration_bars: int = 8, intensity: float = 0.5) -> list:
        """
        Generate percussion events.
        
        Args:
            duration_bars: Length of percussion pattern
            intensity: 0–1, affects density
        
        Returns:
            List of {'type': 'kick'|'swell'|'bell', 'bar': X, 'velocity': Y, ...}
        """
        events = []
        bpm = 70
        bar_duration_sec = (60 / bpm) * 4
        
        # Sparse kick pattern (every 2–4 bars)
        if intensity > 0.2:
            for bar in range(0, duration_bars, np.random.randint(2, 5)):
                events.append({
                    'type': 'kick',
                    'onset_sample': int(bar * bar_duration_sec * 44100),
                    'velocity': 0.6 + intensity * 0.3,
                })
        
        # Granular swells
        num_swells = int(intensity * 3)
        for _ in range(num_swells):
            bar = np.random.randint(0, duration_bars)
            events.append({
                'type': 'swell',
                'onset_sample': int(bar * bar_duration_sec * 44100),
                'duration_sec': np.random.uniform(2, 4),
                'velocity': intensity * 0.7,
            })
        
        return events
```

**Acceptance Criteria**:
- ✓ Generates without errors
- ✓ Events sorted by onset time
- ✓ Respects intensity parameter

---

#### Task 2.3: Control API
**Duration**: 2 days

Extend FastAPI server with `/api/control` endpoint.

```python
# server/main.py (updated)

@app.post("/api/control")
async def control_generation(params: dict):
    """Update generation parameters."""
    key = params.get('key', 'A')
    bpm = params.get('bpm', 70)
    intensity = params.get('intensity', 0.5)
    
    # Update composition engine
    composition_engine.set_params(key=key, bpm=bpm, intensity=intensity)
    
    return {'status': 'updated'}

@app.get("/api/status")
async def get_status():
    """Get server status."""
    return {
        'buffer_depth_ms': ring_buffer.buffer_depth_ms(),
        'num_clients': 1,  # TODO: track
    }
```

**Acceptance Criteria**:
- ✓ POST /api/control updates generation
- ✓ Parameters persist across phrases
- ✓ Changes audible within 1 phrase (~20 seconds)

---

#### Task 2.4: Client Controls UI
**Duration**: 2 days

Add HTML controls and connect to API.

```html
<!-- client/controls.html -->
<div id="controls">
    <label>Key:
        <select id="keySelect" onchange="updateControl()">
            <option value="A">A minor</option>
            <option value="C">C minor</option>
            <option value="F">F minor</option>
        </select>
    </label>
    
    <label>BPM:
        <input type="range" id="bpmSlider" min="60" max="120" value="70" onchange="updateControl()">
        <span id="bpmValue">70</span>
    </label>
    
    <label>Intensity:
        <input type="range" id="intensitySlider" min="0" max="1" step="0.1" value="0.5" onchange="updateControl()">
        <span id="intensityValue">0.5</span>
    </label>
</div>

<script>
    async function updateControl() {
        const key = document.getElementById('keySelect').value;
        const bpm = document.getElementById('bpmSlider').value;
        const intensity = document.getElementById('intensitySlider').value;
        
        document.getElementById('bpmValue').textContent = bpm;
        document.getElementById('intensityValue').textContent = intensity;
        
        const response = await fetch('http://localhost:8000/api/control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ key, bpm: parseInt(bpm), intensity: parseFloat(intensity) }),
        });
    }
</script>
```

**Acceptance Criteria**:
- ✓ Sliders update control parameters
- ✓ Changes audible in generation
- ✓ No lag or freezing

---

#### Task 2.5: Adaptive Client Buffering
**Duration**: 2–3 days

Implement playback rate adjustment for network jitter.

See `implementation_strategies.md` → Section 3.2 for code.

**Acceptance Criteria**:
- ✓ Client buffer adapts to network latency
- ✓ No audible pitch shift (playback rate stays near 1.0)
- ✓ Works with variable network conditions

---

### Phase 2 Success Criteria

- [ ] Melodies sound more "composed" (Transformer output vs. random sampling)
- [ ] Percussion adds texture without overwhelming
- [ ] User can adjust key, BPM, intensity in real-time
- [ ] Client buffer handles network jitter gracefully
- **End-to-end latency**: ~400–600ms

---

## Phase 3: Performance Optimization (Weeks 5–6)

**Goal**: Ensure scalability and responsiveness on M4 Mac

### Deliverables

- [ ] GPU profiling: measure synthesis bottlenecks
- [ ] Batch rendering for GPU efficiency
- [ ] Opus codec (reduce bandwidth 50× if needed)
- [ ] Effects chain: reverb, delay (GPU-accelerated)
- [ ] Monitoring dashboard: buffer depth, CPU/GPU, latency

### Key Tasks

#### Task 3.1: GPU Profiling & Optimization
**Duration**: 3 days

```python
# profiling/benchmark.py
import time
import torch

def benchmark_synthesis(synth, phrase, num_runs: int = 10):
    """Measure synthesis latency."""
    times = []
    
    for _ in range(num_runs):
        start = time.perf_counter()
        audio = synth.render_phrase(**phrase, duration_sec=22.3)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    print(f"Synthesis: {np.mean(times):.2f}s ± {np.std(times):.2f}s")
    print(f"  Real-time factor: {22.3 / np.mean(times):.0f}×")
    
    # GPU memory usage
    print(f"GPU memory: {torch.mps.current_allocated_memory() / 1e9:.2f} GB")

# Run benchmark
synth = SimpleAmbientSynth(device='mps')
phrase = {
    'chords': ['i', 'VI', 'i', 'VI', 'i', 'VI', 'i', 'VI'],
    'melody': [...],
    'percussion': [...],
}
benchmark_synthesis(synth, phrase)
```

**Target**: > 50× real-time on M4 Mac GPU

---

#### Task 3.2: Opus Codec Integration
**Duration**: 2 days

If bandwidth is constrained:

```python
# streaming/opus_encoder.py
import opuslib

class OpusEncoder:
    """Compress audio with Opus codec."""
    
    def __init__(self, sr: int = 44100, channels: int = 2, bitrate: int = 64):
        self.encoder = opuslib.Encoder(sr, channels, opuslib.APPLICATION_AUDIO)
        self.encoder.bitrate = bitrate * 1000  # kbps
    
    def encode(self, pcm_bytes: bytes) -> bytes:
        """Encode PCM to Opus."""
        return self.encoder.encode(pcm_bytes, 960)  # 960 samples @ 48kHz

# Client-side decoding
# (Use opuslib or browser opus.js)
```

**Savings**: ~5 Mbps → ~256 kbps (20× reduction)

---

#### Task 3.3: Effects Chain
**Duration**: 4 days

Add reverb, delay using GPU-efficient filters.

```python
# synthesis/effects.py
import torch

class ReverbEffect:
    """Simple reverb using delay lines."""
    
    def __init__(self, sr: int = 44100, reverb_time_sec: float = 2.0):
        self.sr = sr
        self.decay_time = int(reverb_time_sec * sr)
    
    def process(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply reverb."""
        # Simple Schroeder reverberator: 4 parallel combs + 2 seriesallpass
        # TODO: implement
        return audio
```

**Acceptance Criteria**:
- ✓ Reverb adds spacious feel without muddiness
- ✓ Doesn't increase latency > 50ms
- ✓ CPU overhead < 5%

---

### Phase 3 Success Criteria

- [ ] Synthesis consistently > 50× real-time on M4
- [ ] Bandwidth reduced to < 500 kbps with Opus (if implemented)
- [ ] Reverb/delay effects audible and musical
- [ ] Monitoring dashboard shows all metrics in real-time

---

## Phase 4: Production Readiness (Weeks 7–8)

**Goal**: Robust, monitored, deployable system

### Deliverables

- [ ] Comprehensive error handling + graceful degradation
- [ ] Structured logging (JSON, ECS format)
- [ ] Health check endpoints
- [ ] Load testing (1–10 concurrent clients)
- [ ] Dockerfile + deployment scripts
- [ ] Documentation: API, architecture, deployment

### Key Tasks

#### Task 4.1: Error Handling & Resilience
**Duration**: 2–3 days

See `system_architecture.md` → Error Handling section for strategies.

```python
# server/resilience.py

class ResilientSynthesisWorker:
    """Synthesis worker with automatic restart on failure."""
    
    def __init__(self, synth, max_retries: int = 3):
        self.synth = synth
        self.max_retries = max_retries
    
    async def render_with_retry(self, phrase, duration_sec):
        for attempt in range(self.max_retries):
            try:
                return await asyncio.to_thread(
                    self.synth.render_phrase,
                    **phrase,
                    duration_sec=duration_sec,
                )
            except Exception as e:
                logger.warning(f"Synthesis attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    # Return silence on final failure
                    logger.error("Rendering silence as fallback")
                    return np.zeros((2, int(duration_sec * 44100)))
```

---

#### Task 4.2: Monitoring & Logging
**Duration**: 2 days

```python
# monitoring/logger.py
import logging
import json
from datetime import datetime

class StructuredLogger:
    """JSON-structured logging for monitoring."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.handler = logging.StreamHandler()
        self.handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(self.handler)
    
    def log_event(self, event_type: str, **kwargs):
        """Log structured event."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event': event_type,
            'data': kwargs,
        }
        self.logger.info(json.dumps(log_entry))

# Usage
logger = StructuredLogger('streaming')
logger.log_event('client_connected', client_id='abc123', buffer_depth_ms=150)
```

---

#### Task 4.3: Load Testing
**Duration**: 2 days

```python
# testing/load_test.py
import asyncio
import aiohttp
import time

async def load_test(url: str, num_clients: int = 5, duration_sec: int = 60):
    """Simulate multiple concurrent clients."""
    
    async def client_stream(client_id):
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(url) as ws:
                start = time.time()
                chunks_received = 0
                
                while time.time() - start < duration_sec:
                    try:
                        msg = await asyncio.wait_for(ws.receive_json(), timeout=1.0)
                        if msg.get('type') == 'audio':
                            chunks_received += 1
                    except asyncio.TimeoutError:
                        print(f"Client {client_id}: timeout")
                
                print(f"Client {client_id}: {chunks_received} chunks in {duration_sec}s")
    
    # Run clients in parallel
    tasks = [client_stream(i) for i in range(num_clients)]
    await asyncio.gather(*tasks)

# Run test
asyncio.run(load_test('ws://localhost:8000/ws/stream', num_clients=5, duration_sec=60))
```

**Target**: ≥ 5 concurrent clients, < 5% packet loss

---

### Phase 4 Success Criteria

- [ ] All exceptions caught and logged
- [ ] Server gracefully handles client disconnect
- [ ] 5 concurrent clients play without dropout
- [ ] Monitoring metrics accessible via API
- [ ] Complete deployment guide written

---

## Phase 5: Advanced Features (Optional, Weeks 9+)

**Goal**: Enhanced user experience and advanced capabilities

### Ideas

- [ ] User authentication + preferences (saved settings)
- [ ] A/B testing UI: compare generation variations
- [ ] Offline mode: pre-render popular compositions
- [ ] Mobile app (React Native)
- [ ] MIDI export: save generated compositions
- [ ] Social sharing: share generated tracks

---

## Phase 6: Production Deployment

**Goal**: Production-grade stability

### Deployment Checklist

- [ ] Cloud platform selected (AWS ECS, Google Cloud Run, etc.)
- [ ] Database for user preferences (PostgreSQL)
- [ ] CDN for client assets (CloudFront, Cloudflare)
- [ ] SSL/TLS certificates (Let's Encrypt)
- [ ] Auto-scaling configured
- [ ] Backup/recovery procedures documented
- [ ] Incident response playbook
- [ ] SLO defined (99.9% uptime, < 500ms latency)

---

## Timeline Summary

| Phase | Duration | Focus | Output |
|-------|----------|-------|--------|
| **1** | Weeks 1–2 | MVP | Streaming works end-to-end |
| **2** | Weeks 3–4 | Quality & Control | Better generation, user controls |
| **3** | Weeks 5–6 | Performance | GPU optimization, effects |
| **4** | Weeks 7–8 | Production | Robustness, monitoring, deployment |
| **5** | Weeks 9+ | Optional | Advanced features |
| **6** | Ongoing | Live | Production deployment & scaling |

**Total MVP-to-Production**: ~8 weeks

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| **GPU synthesis too slow** | Profile early (Week 1); have CPU fallback |
| **Network latency > 500ms** | Client-side buffering; Opus codec option |
| **Generation quality poor** | Start with Markov (simple); add Transformer later |
| **Scalability issues** | Load test at Phase 4; optimize before deployment |
| **Audio dropout on client** | Adaptive playback rate; test with throttled network |

---

## Dependencies & Library Versions

```plaintext
# Python
Python 3.10+
pip install fastapi[websockets]==0.104.1
pip install torch==2.1.0 (PyTorch with Metal support)
pip install numpy==1.24.3
pip install scipy==1.11.0
pip install uvicorn==0.23.0
pip install transformers==4.34.0 (for Transformer melody)
pip install opuslib==3.2.0 (optional, for Opus codec)

# JavaScript/Node.js
Node.js 18+
npm install --save-dev webpack webpack-cli
npm install --save axios (for API calls)
npm install --save opus-codec (browser Opus decoder)

# System
macOS 12.0+
Xcode Command Line Tools
Homebrew
```

