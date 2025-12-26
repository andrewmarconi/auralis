┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT (Browser/App)                     │
│                    Web Audio API + WebSocket                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • AudioContext (sample rate matching)                      │ │
│  │ • ScriptProcessorNode (audio callback)                     │ │
│  │ • Adaptive ring buffer (300-500ms)                         │ │
│  │ • Playback rate adjustment for jitter mitigation           │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  WebSocket Connection (TCP, bidirectional)                      │
│  • Incoming: base64 PCM chunks (100ms)                          │
│  • Outgoing: control events (key, BPM, intensity)               │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ (PCM chunks, ~4kB @ 44.1kHz)
                              │ Latency: ~100ms network
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FASTAPI WEB SERVER                         │
│  (async event loop, uvicorn workers)                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ WebSocket Endpoint: /ws/stream                             │ │
│  │ • Manages client connections (1+ clients)                  │ │
│  │ • Accepts control messages (JSON)                          │ │
│  │ • Forwards client parameters to generation engine          │ │
│  │ • Routes audio chunks to each connected client             │ │
│  │                                                            │ │
│  │ REST API Endpoints:                                        │ │
│  │ • POST /api/control - Update generation parameters         │ │
│  │ • GET /api/status - Server health, buffer depth            │ │
│  │ • GET /api/metrics - Performance monitoring                │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ (Composition events)
                              │ (Audio chunks via queue)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 GENERATION LAYER (Async, Main Loop)             │
│  (asyncio tasks, CPU-bound offloaded to thread pool)            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Composition Engine                                         │ │
│  │ • BPM: 60-120 (default 70)                                 │ │
│  │ • Phrase duration: 8 bars (typical)                        │ │
│  │ • Samples per bar: (60/BPM) * 4 * 44100 samples            │ │
│  │                                                            │ │
│  │ Sub-components:                                            │ │
│  │ ├─ ChordProgressionGenerator (Markov chain)                │ │
│  │ │  • Order: 2 (bigram, considers 1 previous chord)         │ │
│  │ │  • Output: [i, v, VI, III] per 8-bar phrase              │ │
│  │ │                                                          │ │
│  │ ├─ MelodyGenerator (constraint-based)                      │ │
│  │ │  • Input: chord progression, scale, intensity            │ │
│  │ │  • Output: [(pitch, duration, velocity)]                 │ │
│  │ │  • Constraints: chord tones 70%, scale 25%, chromatic 5% │ |
│  │ │                                                          │ │
│  │ └─ PercussionGenerator (rule-based texture)                │ │
│  │    • Sparse kicks (every 2-4 bars)                         │ │
│  │    • Granular swells (probabilistic)                       │ │
│  │    • Output: [{'type': 'kick', 'bar': X, 'velocity': Y}]   | │
│  │                                                            │ │
│  │ Phrase Queue: asyncio.Queue(maxsize=3)                     │ │
│  │ • Holds pre-generated phrases (4-8 bars each)              │ │
│  │ • Consumed by synthesis thread every ~20 seconds           │ │
│  │ • Allows synthesis to run ahead without blocking gen       │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ (Composition data from queue)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              SYNTHESIS LAYER (Blocking, Dedicated Thread)       │
│  (numpy, torch with Metal acceleration)                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Real-Time Audio Rendering                                  │ │
│  │ • Device: GPU (Metal Performance Shaders on M4 Mac)        │ │
│  │ • Engine: torchsynth (primary) or Pedalboard (fallback)    │ │
│  │                                                            │ │
│  │ Synthesis Components:                                      │ │
│  │ ├─ PolyVoiceManager (8-16 voices)                          │ │
│  │ │  • Allocates voices to notes                             │ │
│  │ │  • ADSR envelope per voice                               │ │
│  │ │  • Pitch glide (5-50ms slides)                           │ │
│  │ │                                                          │ │
│  │ ├─ OscillatorBank (VCO-like)                               │ │
│  │ │  • Wavetable synthesis (saw, square, sine)               │ │
│  │ │  • Harmonic content control                              │ │
│  │ │                                                          │ │
│  │ ├─ FilteredNoise (filtered random)                         │ │
│  │ │  • For texture/pad swells                                │ │
│  │ │  • Cutoff modulation via LFO                             │ │
│  │ │                                                          │ │
│  │ └─ EffectsChain                                            │ │
│  │    • Reverb (2-3 seconds, pre-delay 20-50ms)               │ │
│  │    • Delay (250-500ms, 1-2 taps)                           │ │
│  │    • Soft limiter (threshold -6dB, makeup gain)            │ │
│  │                                                            │ │
│  │ Output: numpy float32 array, shape (2, num_samples)        │ │
│  │ • Stereo (L/R channels)                                    │ │
│  │ • Sample range: [-1.0, 1.0]                                │ │
│  │ • Sample rate: 44100 Hz                                    │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ (PCM numpy arrays)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│        AUDIO BUFFERING & ENCODING LAYER (Ring Buffer)           │
│  (thread-safe, zero-copy where possible)                        │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Ring Buffer Management                                     │ │
│  │ • Size: 10-20 chunks (1-2 seconds @ 44.1kHz)               │ │
│  │ • Chunk duration: 100ms (4410 samples stereo)              │ │
│  │ • Bytes per chunk: 4410 * 2 (stereo) * 2 (16-bit) = ~17.6kB| │
│  │ • Allocation: Pre-allocated numpy array (no GC pauses)     │ │
│  │                                                            │ │
│  │ Write Side (synthesis thread):                             │ │
│  │ • Receives PCM from synthesis engine                       │ │
│  │ • Writes to ring buffer via write_cursor                   │ │
│  │ • Updates write_cursor (atomic increment)                  │ │
│  │                                                            │ │
│  │ Read Side (WebSocket task):                                │ │
│  │ • Monitors buffer depth: (write_cursor - read_cursor)      │ │
│  │ • Back-pressure: if buffer < 2 chunks, sleep 10ms          │ │
│  │ • Reads 100ms chunk, encodes to base64                     │ │
│  │ • Updates read_cursor                                      │ │
│  │                                                            │ │
│  │ Encoding:                                                  │ │
│  │ • Convert float32 [-1, 1] → int16 [-32768, 32767]          │ │
│  │ • Byte-order: little-endian (standard)                     │ │
│  │ • base64(pcm_bytes) → ~23.5kB string per chunk             │ │
│  │                                                            │ │
│  │ Back-Pressure & Flow Control:                              │ │
│  │ • If client disconnects, cease writing to its queue        │ │
│  │ • If queue fills (5+ chunks), slow composition gen         │ │
│  │ • Graceful degradation: mute or play silence               │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ (base64 chunks via JSON)
                              │ ~50 Mbps bandwidth @ PCM
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT BUFFER                            │
│  (JavaScript, Web Audio API)                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Adaptive Ring Buffer (client-side)                         │ │
│  │ • Size: 300-500ms (target latency)                         │ │
│  │ • Adjusts playback rate if buffer fills/empties            │ │
│  │ • Sample rate conversion (if needed)                       │ │
│  │                                                            │ │
│  │ Audio Playback:                                            │ │
│  │ • ScriptProcessorNode pulls from ring buffer               │ │
│  │ • Outputs to destination (speaker/headphones)              │ │
│  │ • Callback fires every 4096 samples (~93ms @ 44.1kHz)      │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
