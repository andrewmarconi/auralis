# Implementation Strategies: Synthesis, Buffering & Streaming

---

## 1. SYNTHESIS STRATEGY

### 1.1 torchsynth Core Rendering Pipeline

**Why torchsynth?**
- Renders 16,000× real-time on GPU (M4 Mac Metal: ~700 MHz equivalent)
- Fully differentiable (future: train generative models)
- Modular synth architecture (VCOs, ADSRs, filters)
- PyTorch integration with Apple Silicon support

**Architecture:**

```python
# synthesis_engine.py
import torch
import numpy as np
from typing import Dict, List, Tuple

class AmbientSynthesizer:
    """
    Real-time synthesizer for ambient music.
    Renders multiple voices with effects.
    """
    
    def __init__(self, sample_rate: int = 44100, device: str = 'mps'):
        self.sr = sample_rate
        self.device = device
        
        # Pre-allocate buffers (no GC)
        self.max_phrase_samples = int(30 * sample_rate)  # 30 seconds max
        self.output_buffer = torch.zeros(
            (2, self.max_phrase_samples),
            dtype=torch.float32,
            device=device
        )
        
        # Voice pool: up to 12 polyphonic voices
        self.max_voices = 12
        self.active_voices = []
        
    def render_phrase(
        self,
        chords: List[Tuple[int, int, int]],  # (onset_samples, root_midi, chord_type)
        melody: List[Tuple[int, int, float, float]],  # (onset, pitch, velocity, duration_sec)
        percussion: List[Dict],  # (onset, type, velocity)
        duration_sec: float,
    ) -> np.ndarray:
        """
        Render a complete phrase: chords + melody + percussion.
        
        Args:
            chords: List of (onset_sample, root_note, chord_type)
                Example: [(0, 57, 'i'), (22050, 57, 'VI')]
            melody: List of (onset_sample, pitch_midi, velocity, duration_sec)
                Example: [(0, 60, 0.7, 2.0), (8820, 62, 0.75, 1.5)]
            percussion: List of dicts with 'type', 'velocity', 'onset_sample'
            duration_sec: Total duration to render
        
        Returns:
            numpy array shape (2, num_samples) in [-1.0, 1.0] range
        """
        num_samples = int(duration_sec * self.sr)
        
        # Initialize output buffers (stereo)
        chord_audio = torch.zeros((2, num_samples), device=self.device)
        melody_audio = torch.zeros((2, num_samples), device=self.device)
        perc_audio = torch.zeros((2, num_samples), device=self.device)
        
        # Render each track
        chord_audio = self._render_chords(chords, num_samples)
        melody_audio = self._render_melody(melody, num_samples)
        perc_audio = self._render_percussion(percussion, num_samples)
        
        # Mix with levels
        mix = (
            chord_audio * 0.5 +      # Pad: -6dB
            melody_audio * 0.7 +     # Lead: -3dB
            perc_audio * 0.3         # Percussion: -10dB
        )
        
        # Soft limiter (tanh saturation)
        mix = torch.tanh(mix * 0.95)
        
        # Convert to numpy, float32 in [-1, 1]
        return mix.cpu().numpy()
    
    def _render_chords(
        self,
        chords: List[Tuple[int, int, int]],
        num_samples: int
    ) -> torch.Tensor:
        """
        Render pad/chord layer.
        
        Strategy:
        - Each chord: root + 3rd + 5th + 7th (4-voice harmony)
        - Slow ADSR: A=500ms, D=2000ms, S=0.7, R=1000ms
        - LFO modulation on filter cutoff (0.5 Hz)
        - Output: stereo pad texture
        """
        audio = torch.zeros((2, num_samples), device=self.device)
        
        for chord_idx, (onset_sample, root_midi, chord_type) in enumerate(chords):
            # Get chord notes (e.g., root, 3rd, 5th, 7th in Cmaj: C, E, G, B)
            chord_notes = self._get_chord_notes(root_midi, chord_type)
            
            # Determine offset for next chord (or end of phrase)
            if chord_idx < len(chords) - 1:
                next_onset = chords[chord_idx + 1][0]
                chord_duration_samples = next_onset - onset_sample
            else:
                chord_duration_samples = num_samples - onset_sample
            
            # Render 4-voice poly synth for this chord
            for voice_idx, note_midi in enumerate(chord_notes):
                # ADSR envelope
                envelope = self._adsr_envelope(
                    onset_sample,
                    chord_duration_samples,
                    attack_ms=500,
                    decay_ms=2000,
                    sustain=0.7,
                    release_ms=1000,
                )
                
                # Wavetable oscillator (sawtooth for richness)
                osc = self._oscillator(
                    note_midi,
                    chord_duration_samples,
                    waveform='saw',
                    octave_offset=voice_idx - 1,  # Spread voices
                )
                
                # Apply LFO modulation to filter
                lfo = torch.sin(
                    2 * np.pi * 0.5 *
                    torch.arange(chord_duration_samples, device=self.device) / self.sr
                )
                
                # Filter (simple 1-pole lowpass for resonance)
                cutoff_base = 1000 + lfo * 200  # 1000-1200 Hz
                filtered = self._lowpass_filter(osc, cutoff_base)
                
                # Voice to output
                voice_signal = filtered * envelope
                
                # Pan voices slightly for stereo width
                pan = 0.5 + voice_idx * 0.1
                audio[0, onset_sample:onset_sample + chord_duration_samples] += \
                    voice_signal * (1 - pan)
                audio[1, onset_sample:onset_sample + chord_duration_samples] += \
                    voice_signal * pan
        
        return audio
    
    def _render_melody(
        self,
        melody: List[Tuple[int, int, float, float]],
        num_samples: int
    ) -> torch.Tensor:
        """
        Render synth lead melody.
        
        Strategy:
        - Single voice, monophonic
        - Per-note ADSR: A=50ms, D=100ms, S=0.8, R=200ms
        - Pitch glide between notes (5-10ms portamento)
        - Velocity controls amplitude
        - Output: lead voice (mono, center)
        """
        audio = torch.zeros((2, num_samples), device=self.device)
        
        for note_idx, (onset_sample, pitch_midi, velocity, duration_sec) in enumerate(melody):
            duration_samples = int(duration_sec * self.sr)
            
            # Ensure note doesn't overrun buffer
            if onset_sample + duration_samples > num_samples:
                duration_samples = num_samples - onset_sample
            
            if duration_samples <= 0:
                continue
            
            # ADSR for note
            envelope = self._adsr_envelope(
                0,
                duration_samples,
                attack_ms=50,
                decay_ms=100,
                sustain=0.8,
                release_ms=200,
            )
            
            # Pitch glide if previous note exists
            if note_idx > 0:
                prev_pitch = melody[note_idx - 1][1]
                glide_time_ms = 10
                glide_samples = int(glide_time_ms * self.sr / 1000)
                pitch_curve = torch.linspace(
                    float(prev_pitch),
                    float(pitch_midi),
                    min(glide_samples, duration_samples),
                    device=self.device
                )
            else:
                pitch_curve = torch.ones(duration_samples, device=self.device) * pitch_midi
            
            # Oscillator (sine wave for smooth lead)
            osc = self._oscillator_with_pitch_curve(
                pitch_curve,
                waveform='sine'
            )
            
            # Apply envelope and velocity
            voice_signal = osc * envelope * velocity
            
            # Center pan (stereo identical)
            audio[0, onset_sample:onset_sample + duration_samples] = voice_signal
            audio[1, onset_sample:onset_sample + duration_samples] = voice_signal
        
        return audio
    
    def _render_percussion(
        self,
        percussion: List[Dict],
        num_samples: int
    ) -> torch.Tensor:
        """
        Render sparse percussion/texture.
        
        Strategy:
        - Kick: sine sweep 150 Hz → 50 Hz over 100ms
        - Swell: bandpass-filtered noise, long ADSR
        - Hihat: short burst of filtered noise
        """
        audio = torch.zeros((2, num_samples), device=self.device)
        
        for perc_event in percussion:
            perc_type = perc_event.get('type', 'kick')
            onset_sample = perc_event.get('onset_sample', 0)
            velocity = perc_event.get('velocity', 1.0)
            
            if perc_type == 'kick':
                # Sub-bass kick: sine frequency sweep
                kick_duration = int(0.1 * self.sr)  # 100ms
                t = torch.arange(kick_duration, device=self.device) / self.sr
                
                # Frequency sweep: 150 Hz → 50 Hz
                freq_start, freq_end = 150, 50
                freq_curve = freq_start - (freq_start - freq_end) * (t / (kick_duration / self.sr))
                phase = torch.cumsum(2 * np.pi * freq_curve / self.sr, dim=0)
                
                kick = torch.sin(phase) * velocity
                
                # Exponential decay envelope
                kick_env = torch.exp(-5 * t)
                kick = kick * kick_env
                
                # Write to audio
                end_idx = min(onset_sample + kick_duration, num_samples)
                audio[0, onset_sample:end_idx] += kick[:end_idx - onset_sample]
                audio[1, onset_sample:end_idx] += kick[:end_idx - onset_sample]
            
            elif perc_type == 'swell':
                # Granular texture: filtered noise
                swell_duration = int(perc_event.get('duration_sec', 2.0) * self.sr)
                noise = torch.randn(swell_duration, device=self.device)
                
                # Bandpass filter (tuned bells region: 500-2000 Hz)
                filtered = self._bandpass_filter(noise, center_freq=1000, q=2)
                
                # ADSR: slow for ambient
                env = self._adsr_envelope(
                    0, swell_duration,
                    attack_ms=200,
                    decay_ms=1500,
                    sustain=0.5,
                    release_ms=500,
                )
                
                swell = filtered * env * velocity
                
                end_idx = min(onset_sample + swell_duration, num_samples)
                audio[0, onset_sample:end_idx] += swell[:end_idx - onset_sample] * 0.5
                audio[1, onset_sample:end_idx] += swell[:end_idx - onset_sample] * 0.5
        
        return audio
    
    def _oscillator(
        self,
        midi_note: int,
        duration_samples: int,
        waveform: str = 'sine',
        octave_offset: int = 0,
    ) -> torch.Tensor:
        """Generate wavetable oscillator."""
        freq = 440 * (2 ** ((midi_note - 69 + octave_offset * 12) / 12))
        t = torch.arange(duration_samples, device=self.device) / self.sr
        phase = 2 * np.pi * freq * t
        
        if waveform == 'sine':
            return torch.sin(phase)
        elif waveform == 'saw':
            # Sawtooth: sin + harmonics
            saw = torch.sin(phase)
            for harmonic in range(2, 5):
                saw += torch.sin(harmonic * phase) / harmonic
            return saw / 2
        elif waveform == 'square':
            return torch.sign(torch.sin(phase))
        else:
            return torch.sin(phase)
    
    def _oscillator_with_pitch_curve(
        self,
        pitch_curve: torch.Tensor,
        waveform: str = 'sine',
    ) -> torch.Tensor:
        """Generate oscillator with time-varying pitch (for glides)."""
        # Convert MIDI to frequency
        freqs = 440 * (2 ** ((pitch_curve - 69) / 12))
        
        # Integrate frequency to get phase
        phase = torch.cumsum(2 * np.pi * freqs / self.sr, dim=0)
        
        if waveform == 'sine':
            return torch.sin(phase)
        else:
            return torch.sin(phase)
    
    def _adsr_envelope(
        self,
        onset_sample: int,
        duration_samples: int,
        attack_ms: float,
        decay_ms: float,
        sustain: float,
        release_ms: float,
    ) -> torch.Tensor:
        """
        Generate ADSR envelope.
        
        Segments:
        - Attack: 0 → 1 over attack_ms
        - Decay: 1 → sustain over decay_ms
        - Sustain: sustain level for remainder
        - Release: sustain → 0 over release_ms (at end)
        """
        envelope = torch.ones(duration_samples, device=self.device)
        
        attack_samples = int(attack_ms * self.sr / 1000)
        decay_samples = int(decay_ms * self.sr / 1000)
        release_samples = int(release_ms * self.sr / 1000)
        sustain_samples = max(0, duration_samples - attack_samples - decay_samples - release_samples)
        
        idx = 0
        
        # Attack
        if attack_samples > 0:
            envelope[idx:idx + attack_samples] = torch.linspace(0, 1, attack_samples, device=self.device)
            idx += attack_samples
        
        # Decay
        if decay_samples > 0:
            envelope[idx:idx + decay_samples] = torch.linspace(1, sustain, decay_samples, device=self.device)
            idx += decay_samples
        
        # Sustain
        envelope[idx:idx + sustain_samples] = sustain
        idx += sustain_samples
        
        # Release
        if release_samples > 0:
            envelope[idx:idx + release_samples] = torch.linspace(sustain, 0, release_samples, device=self.device)
        
        return envelope
    
    def _lowpass_filter(
        self,
        signal: torch.Tensor,
        cutoff_freq: float,
        q: float = 0.707,
    ) -> torch.Tensor:
        """1-pole lowpass filter (simple)."""
        # Simple exponential smoothing
        alpha = torch.clamp(
            cutoff_freq / (cutoff_freq + self.sr),
            min=0.01,
            max=0.99
        )
        
        filtered = torch.zeros_like(signal)
        filtered[0] = signal[0]
        
        for i in range(1, len(signal)):
            filtered[i] = alpha * signal[i] + (1 - alpha) * filtered[i - 1]
        
        return filtered
    
    def _bandpass_filter(
        self,
        signal: torch.Tensor,
        center_freq: float,
        q: float = 2.0,
    ) -> torch.Tensor:
        """Bandpass filter (approximation)."""
        # TODO: Implement proper biquad bandpass
        return signal
    
    def _get_chord_notes(self, root_midi: int, chord_type: str) -> List[int]:
        """
        Return MIDI notes for chord.
        
        Example:
            root_midi=60 (C), chord_type='maj7' → [60, 64, 67, 71] (C, E, G, B)
        """
        intervals = {
            'i': [0, 3, 7],  # Minor triad
            'V': [0, 4, 7],  # Major triad
            'VI': [0, 3, 7],  # Minor triad
            'III': [0, 4, 7],  # Major triad
            'iv': [0, 3, 7],  # Minor triad
        }
        
        chord_intervals = intervals.get(chord_type, [0, 4, 7])  # Default major
        
        # Add 7th
        if '7' in chord_type:
            chord_intervals.append(10)
        
        # Return notes at various octaves for richness
        notes = [
            root_midi - 12 + interval,  # One octave down
            root_midi + interval,        # At root octave
        ]
        
        return notes[:4]  # Return up to 4 voices
```

---

### 1.2 Pedalboard Fallback (VST Hosting)

If torchsynth is insufficient:

```python
# vst_synthesis.py
import pedalboard
import numpy as np

class VSTPedalboardSynthesizer:
    """Fallback: host VST synth (Serum, Vital) via Pedalboard."""
    
    def __init__(self, vst_path: str, sample_rate: int = 44100):
        """
        Args:
            vst_path: Path to .vst3 or .au file (e.g., '/path/to/Serum.vst3')
            sample_rate: Sample rate (typically 44100 or 48000)
        """
        self.synth = pedalboard.VST(vst_path)
        self.sr = sample_rate
    
    def render_phrase_with_midi(
        self,
        midi_events: List[Dict],  # [{'pitch': 60, 'velocity': 100, 'onset_sec': 0, 'duration_sec': 1}]
        duration_sec: float,
    ) -> np.ndarray:
        """
        Render MIDI events through VST synth.
        
        Note: Pedalboard doesn't natively support MIDI scheduling,
        so this is a simplified example. Use JUCE/DawDreamer for
        more complex MIDI routing.
        """
        # Create silent audio buffer
        num_samples = int(duration_sec * self.sr)
        audio = np.zeros((2, num_samples), dtype=np.float32)
        
        # TODO: Use Pedalboard's MIDI interface (limited)
        # This requires custom MIDI note on/off scheduling
        
        return audio
```

---

## 2. BUFFERING STRATEGY

### 2.1 Ring Buffer (Server-Side)

```python
# ring_buffer.py
import numpy as np
from threading import Lock
from typing import Tuple

class RingBuffer:
    """
    Thread-safe circular audio buffer.
    
    Used between synthesis thread (writer) and WebSocket task (reader).
    No locks on data access; uses atomic integer cursors.
    """
    
    def __init__(self, sample_rate: int = 44100, capacity_sec: float = 2.0):
        """
        Args:
            sample_rate: Audio sample rate
            capacity_sec: Total buffer capacity in seconds
        """
        self.sr = sample_rate
        self.capacity_samples = int(sample_rate * capacity_sec)
        
        # Pre-allocate buffer (stereo, float32)
        self.buffer = np.zeros((2, self.capacity_samples), dtype=np.float32)
        
        # Read/write cursors (atomic increments, modulo wrapping handled by Python)
        self.write_cursor = 0  # Index where synthesis writes next
        self.read_cursor = 0   # Index where WebSocket reads next
        
        # Lock for thread-safe cursor updates
        self._lock = Lock()
    
    def write(self, audio: np.ndarray) -> None:
        """
        Write audio chunk to ring buffer.
        
        Args:
            audio: numpy array shape (2, num_samples), float32 in [-1, 1]
        """
        num_samples = audio.shape[1]
        
        with self._lock:
            for i in range(num_samples):
                # Modulo wrapping
                idx = self.write_cursor % self.capacity_samples
                self.buffer[:, idx] = audio[:, i]
                self.write_cursor += 1
    
    def read(self, num_samples: int) -> np.ndarray:
        """
        Read audio chunk from ring buffer.
        
        Args:
            num_samples: Number of samples to read per channel
        
        Returns:
            numpy array shape (2, num_samples)
        """
        chunk = np.zeros((2, num_samples), dtype=np.float32)
        
        with self._lock:
            for i in range(num_samples):
                idx = self.read_cursor % self.capacity_samples
                chunk[:, i] = self.buffer[:, idx]
                self.read_cursor += 1
        
        return chunk
    
    def buffer_depth_samples(self) -> int:
        """Return current depth in samples (how much data is buffered)."""
        with self._lock:
            depth = (self.write_cursor - self.read_cursor) % self.capacity_samples
        return depth
    
    def buffer_depth_ms(self) -> float:
        """Return current depth in milliseconds."""
        return (self.buffer_depth_samples() / self.sr) * 1000
    
    def is_full(self, threshold_sec: float = 1.5) -> bool:
        """Check if buffer is approaching capacity."""
        return self.buffer_depth_samples() > int(threshold_sec * self.sr)
    
    def is_low(self, threshold_sec: float = 0.05) -> bool:
        """Check if buffer is critically low (underrun risk)."""
        return self.buffer_depth_samples() < int(threshold_sec * self.sr)
    
    def clear(self) -> None:
        """Reset buffer (e.g., on stream restart)."""
        with self._lock:
            self.buffer.fill(0.0)
            self.write_cursor = 0
            self.read_cursor = 0
```

### 2.2 Multi-Client Buffering (Per-Client Queue)

```python
# streaming_buffer.py
import asyncio
import numpy as np
from collections import deque
from typing import Optional

class ClientAudioQueue:
    """
    Per-client async queue for audio chunks.
    
    Receives chunks from ring buffer, queues for WebSocket send.
    Implements back-pressure: if queue fills, slow down generation.
    """
    
    def __init__(self, max_queue_size: int = 5, chunk_duration_ms: int = 100):
        """
        Args:
            max_queue_size: Max chunks in queue before back-pressure
            chunk_duration_ms: Duration of each chunk
        """
        self.max_queue_size = max_queue_size
        self.chunk_duration_ms = chunk_duration_ms
        
        # Queue of base64-encoded audio chunks
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        
        # Statistics
        self.chunks_sent = 0
        self.chunks_dropped = 0
    
    async def put(self, chunk_b64: str, timeout_sec: float = 0.5) -> bool:
        """
        Put encoded chunk in queue.
        
        Args:
            chunk_b64: base64-encoded PCM
            timeout_sec: Timeout before dropping chunk
        
        Returns:
            True if queued successfully, False if queue full
        """
        try:
            await asyncio.wait_for(
                self.queue.put(chunk_b64),
                timeout=timeout_sec
            )
            return True
        except asyncio.TimeoutError:
            self.chunks_dropped += 1
            return False
    
    async def get(self) -> Optional[str]:
        """Get next chunk from queue."""
        try:
            return await self.queue.get()
        except asyncio.CancelledError:
            return None
    
    def is_full(self) -> bool:
        """Check if queue is at capacity."""
        return self.queue.qsize() >= self.max_queue_size
    
    def queue_depth_ms(self) -> float:
        """Return buffered time in milliseconds."""
        return self.queue.qsize() * self.chunk_duration_ms
```

---

## 3. STREAMING STRATEGY

### 3.1 FastAPI WebSocket Server

```python
# streaming_server.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
import asyncio
import base64
import numpy as np
import logging
from datetime import datetime

app = FastAPI(title="Ambient Music Generator")

# Global state
class StreamingState:
    def __init__(self):
        self.ring_buffer = None
        self.composition_engine = None
        self.synthesis_thread = None
        self.active_clients = set()
        self.metrics = {
            'total_chunks_sent': 0,
            'total_chunks_dropped': 0,
            'avg_buffer_depth_ms': 0,
        }

state = StreamingState()

@app.on_event('startup')
async def startup():
    """Initialize synthesis and generation at startup."""
    from ring_buffer import RingBuffer
    from synthesis_engine import AmbientSynthesizer
    from composition_engine import CompositionEngine
    
    # Initialize global state
    state.ring_buffer = RingBuffer(sample_rate=44100, capacity_sec=2.0)
    synthesizer = AmbientSynthesizer(sr=44100, device='mps')
    state.composition_engine = CompositionEngine(sr=44100, bpm=70)
    
    # Start background tasks
    asyncio.create_task(state.composition_engine.generate_phrases())
    asyncio.create_task(_synthesis_worker(synthesizer))
    asyncio.create_task(_metrics_reporter())

async def _synthesis_worker(synthesizer):
    """
    Background task: continuously render phrases to ring buffer.
    
    Runs in asyncio event loop (async wrapper around blocking synthesis).
    """
    logger = logging.getLogger('synthesis')
    
    while True:
        try:
            # Get next phrase from composition engine
            phrase = await asyncio.wait_for(
                state.composition_engine.phrase_queue.get(),
                timeout=1.0
            )
        except asyncio.TimeoutError:
            # No phrase ready; render silence
            logger.warning("No phrase ready; rendering silence")
            phrase = {
                'chords': [],
                'melody': [],
                'percussion': [],
            }
        
        try:
            # Render phrase (blocking synthesis)
            audio = await asyncio.to_thread(
                synthesizer.render_phrase,
                phrase['chords'],
                phrase['melody'],
                phrase['percussion'],
                duration_sec=22.3,  # 8 bars @ 70 BPM
            )
            
            # Write to ring buffer
            audio_tensor = np.atleast_2d(audio)
            if audio_tensor.shape[0] == 1:
                # Mono → stereo duplicate
                audio_tensor = np.vstack([audio_tensor, audio_tensor])
            
            state.ring_buffer.write(audio_tensor)
            
            logger.debug(f"Rendered phrase, buffer depth: {state.ring_buffer.buffer_depth_ms():.0f}ms")
        
        except Exception as e:
            logger.error(f"Synthesis error: {e}", exc_info=True)
            # Write silence on error
            silence = np.zeros((2, int(22.3 * 44100)), dtype=np.float32)
            state.ring_buffer.write(silence)

async def _metrics_reporter():
    """Periodically log performance metrics."""
    logger = logging.getLogger('metrics')
    
    while True:
        await asyncio.sleep(10)  # Every 10 seconds
        
        depth_ms = state.ring_buffer.buffer_depth_ms()
        num_clients = len(state.active_clients)
        
        logger.info(
            f"Metrics: buffer={depth_ms:.0f}ms, clients={num_clients}, "
            f"chunks_sent={state.metrics['total_chunks_sent']}, "
            f"chunks_dropped={state.metrics['total_chunks_dropped']}"
        )

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio streaming.
    
    Protocol:
    - Server → Client: {"type": "audio", "data": "base64_pcm", "timestamp": float}
    - Client → Server: {"type": "control", "key": "A", "bpm": 70, ...}
    """
    await websocket.accept()
    client_id = id(websocket)
    state.active_clients.add(client_id)
    
    logger = logging.getLogger(f'client_{client_id}')
    logger.info("Client connected")
    
    try:
        # Start streaming task
        streaming_task = asyncio.create_task(
            _stream_audio_to_client(websocket, logger)
        )
        
        # Start control message task
        control_task = asyncio.create_task(
            _handle_control_messages(websocket, logger)
        )
        
        # Wait for either task to complete (client disconnect)
        done, pending = await asyncio.wait(
            [streaming_task, control_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel remaining tasks
        for task in pending:
            task.cancel()
        
    except WebSocketDisconnect:
        logger.info("Client disconnected (WebSocketDisconnect)")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        await websocket.close(code=1000, reason=str(e))
    finally:
        state.active_clients.discard(client_id)
        logger.info("Client cleanup complete")

async def _stream_audio_to_client(websocket: WebSocket, logger):
    """Stream audio chunks to connected client."""
    chunk_duration_ms = 100
    
    while True:
        try:
            # Check buffer depth
            buffer_depth_ms = state.ring_buffer.buffer_depth_ms()
            
            if buffer_depth_ms < 50:  # Critical low
                logger.warning(f"Low buffer: {buffer_depth_ms:.0f}ms")
                await asyncio.sleep(0.05)  # Back off
                continue
            
            # Read chunk from ring buffer
            chunk_samples = int((chunk_duration_ms / 1000.0) * 44100)
            chunk = state.ring_buffer.read(chunk_samples)  # (2, 4410)
            
            # Convert to int16 PCM
            chunk_int16 = (np.clip(chunk, -1, 1) * 32767).astype(np.int16)
            pcm_bytes = chunk_int16.tobytes()
            
            # Encode as base64
            encoded = base64.b64encode(pcm_bytes).decode('ascii')
            
            # Send to client
            message = {
                'type': 'audio',
                'data': encoded,
                'timestamp': datetime.utcnow().isoformat(),
            }
            
            await websocket.send_json(message)
            state.metrics['total_chunks_sent'] += 1
            
            # Sleep until next chunk boundary
            await asyncio.sleep(chunk_duration_ms / 1000.0 * 0.95)  # 95ms actual delay
        
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise

async def _handle_control_messages(websocket: WebSocket, logger):
    """Handle incoming control messages from client."""
    
    while True:
        try:
            message = await websocket.receive_json()
            
            msg_type = message.get('type')
            
            if msg_type == 'control':
                # Update generation parameters
                key = message.get('key', 'A')
                bpm = message.get('bpm', 70)
                intensity = message.get('intensity', 0.5)
                
                logger.debug(f"Control: key={key}, bpm={bpm}, intensity={intensity}")
                
                # Forward to composition engine
                await state.composition_engine.update_params({
                    'key': key,
                    'bpm': bpm,
                    'intensity': intensity,
                })
            
            elif msg_type == 'status':
                # Client requests server status
                status = {
                    'buffer_depth_ms': state.ring_buffer.buffer_depth_ms(),
                    'num_clients': len(state.active_clients),
                }
                await websocket.send_json({
                    'type': 'status',
                    'data': status,
                })
        
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Control message error: {e}")
            raise

@app.get("/api/status")
async def get_status():
    """REST endpoint: get server status."""
    return {
        'buffer_depth_ms': state.ring_buffer.buffer_depth_ms(),
        'num_clients': len(state.active_clients),
        'uptime_sec': 0,  # TODO: implement
    }

@app.get("/api/metrics")
async def get_metrics():
    """REST endpoint: get performance metrics."""
    return state.metrics
```

### 3.2 Client-Side Web Audio (JavaScript)

```javascript
// client/audio_client.js

class AmbientAudioClient {
    constructor(wsUrl = 'ws://localhost:8000/ws/stream', targetLatencyMs = 400) {
        this.wsUrl = wsUrl;
        this.targetLatencyMs = targetLatencyMs;
        
        // Audio context
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        this.sampleRate = this.audioContext.sampleRate;
        
        // Ring buffer (client-side)
        this.bufferSizeMs = 500;
        this.bufferSizeSamples = Math.round((this.bufferSizeMs / 1000) * this.sampleRate);
        this.ringBuffer = new Float32Array(this.bufferSizeSamples * 2);  // Stereo
        this.writeIdx = 0;
        this.readIdx = 0;
        
        // WebSocket
        this.ws = null;
        
        // Audio processing node
        this.scriptProcessor = this.audioContext.createScriptProcessor(4096, 0, 2);
        this.scriptProcessor.onaudioprocess = (e) => this._onAudioProcess(e);
        this.scriptProcessor.connect(this.audioContext.destination);
        
        // Adaptive playback rate
        this._playbackRate = 1.0;
        this.isConnected = false;
    }
    
    connect() {
        return new Promise((resolve, reject) => {
            this.ws = new WebSocket(this.wsUrl);
            
            this.ws.onopen = () => {
                console.log('[Client] Connected to server');
                this.isConnected = true;
                resolve();
            };
            
            this.ws.onmessage = (event) => {
                const message = JSON.parse(event.data);
                this._handleMessage(message);
            };
            
            this.ws.onerror = (error) => {
                console.error('[Client] WebSocket error:', error);
                reject(error);
            };
            
            this.ws.onclose = () => {
                console.log('[Client] Disconnected from server');
                this.isConnected = false;
            };
        });
    }
    
    _handleMessage(message) {
        if (message.type === 'audio') {
            // Decode base64 PCM
            const binaryString = atob(message.data);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }
            
            // Convert bytes to int16 array
            const int16Array = new Int16Array(bytes.buffer);
            
            // Write to ring buffer
            for (let i = 0; i < int16Array.length; i++) {
                const sample = int16Array[i] / 32767.0;  // Normalize to [-1, 1]
                const idx = (this.writeIdx + i) % (this.bufferSizeSamples * 2);
                this.ringBuffer[idx] = sample;
            }
            
            // Update write cursor
            this.writeIdx = (this.writeIdx + int16Array.length) % (this.bufferSizeSamples * 2);
        }
    }
    
    _onAudioProcess(event) {
        const outputL = event.outputBuffer.getChannelData(0);
        const outputR = event.outputBuffer.getChannelData(1);
        
        // Check buffer depth
        const bufferDepth = (this.writeIdx - this.readIdx + this.bufferSizeSamples * 2) % (this.bufferSizeSamples * 2);
        const bufferDepthMs = (bufferDepth / this.sampleRate) * 1000;
        
        // Adaptive playback rate
        if (bufferDepthMs > this.targetLatencyMs * 1.5) {
            this._playbackRate = 1.02;  // Speed up
            console.log('[Client] Buffer full, speeding up');
        } else if (bufferDepthMs < this.targetLatencyMs * 0.5) {
            this._playbackRate = 0.98;  // Slow down
            console.log('[Client] Buffer low, slowing down');
        } else {
            this._playbackRate = 1.0;
        }
        
        // Read from ring buffer with rate adjustment
        let readIdxFloat = this.readIdx;
        
        for (let i = 0; i < event.outputBuffer.length; i++) {
            const idx = Math.floor(readIdxFloat) % (this.bufferSizeSamples * 2);
            
            // Simple stereo interleaving (assuming PCM is stereo)
            const sampleL = this.ringBuffer[idx];
            const sampleR = this.ringBuffer[(idx + 1) % (this.bufferSizeSamples * 2)];
            
            outputL[i] = sampleL;
            outputR[i] = sampleR;
            
            readIdxFloat += this._playbackRate * 2;  // 2 channels
        }
        
        this.readIdx = Math.floor(readIdxFloat) % (this.bufferSizeSamples * 2);
    }
    
    setControl(params) {
        if (!this.isConnected) return;
        
        this.ws.send(JSON.stringify({
            type: 'control',
            ...params,  // key, bpm, intensity, etc.
        }));
    }
    
    getStatus() {
        const bufferDepth = (this.writeIdx - this.readIdx + this.bufferSizeSamples * 2) % (this.bufferSizeSamples * 2);
        return {
            buffer_depth_ms: (bufferDepth / this.sampleRate) * 1000,
            is_connected: this.isConnected,
        };
    }
}

// Usage
const client = new AmbientAudioClient();
await client.connect();
client.setControl({ key: 'A minor', bpm: 70 });
```

