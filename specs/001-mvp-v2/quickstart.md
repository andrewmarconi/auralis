# Quick Start Testing Guide: Auralis MVP v2.0

**Feature**: Real-Time Generative Ambient Music Streaming Engine
**Branch**: `001-mvp-v2`
**Date**: 2025-12-28

## Purpose

This guide provides manual testing scenarios to validate the Auralis MVP v2.0 implementation. Use these scenarios to verify functionality before automated testing and to explore the system's behavior under various conditions.

---

## Prerequisites

### System Requirements
- Python 3.12+ installed (`python --version`)
- FluidSynth 2.x native library installed
  - **macOS**: `brew install fluidsynth`
  - **Linux**: `apt-get install fluidsynth libfluidsynth-dev`
  - **Windows**: Download from https://github.com/FluidSynth/fluidsynth/releases
- Modern web browser (Chrome 90+, Edge 90+, or Safari 14+)
- 1GB+ available RAM
- Network connection (for WebSocket streaming)

### SoundFont Setup
```bash
# Download and install SoundFonts
cd /Users/andrew/Develop/auralis
mkdir -p soundfonts

# Download Salamander Grand Piano (200MB)
# URL: https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html

# Download FluidR3_GM (140MB)
# URL: https://member.keymusician.com/Member/FluidR3_GM/

# Verify files exist
ls -lh soundfonts/
# Expected: Salamander*.sf2, FluidR3_GM.sf2
```

### Environment Configuration
```bash
# Create .env file (copy from .env.example if available)
cat > .env <<EOF
AURALIS_ENV=development
AURALIS_HOST=0.0.0.0
AURALIS_PORT=8000
SOUNDFONT_PIANO=soundfonts/SalamanderGrandPiano.sf2
SOUNDFONT_GM=soundfonts/FluidR3_GM.sf2
AURALIS_LOG_LEVEL=INFO
EOF
```

---

## Scenario 1: Basic Server Startup

**Objective**: Verify server starts successfully and loads SoundFonts.

**Steps**:
1. Start server:
   ```bash
   uv run uvicorn server.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. Observe console output for:
   - FluidSynth initialization log
   - SoundFont loading confirmation
   - Server listening on http://0.0.0.0:8000

**Expected Results**:
- ✅ Server starts without errors
- ✅ Log shows "SoundFont loaded: [path]" for piano and pads
- ✅ No warnings about missing files
- ✅ Server accessible at http://localhost:8000

**Common Issues**:
- ❌ "SoundFont not found" → Verify SoundFont paths in .env
- ❌ "FluidSynth library not found" → Install native library (see Prerequisites)
- ❌ Port already in use → Change AURALIS_PORT in .env

---

## Scenario 2: Health Check Endpoints

**Objective**: Verify REST API endpoints return valid responses.

**Steps**:
1. Check system status:
   ```bash
   curl http://localhost:8000/api/status | jq
   ```

2. Check performance metrics:
   ```bash
   curl http://localhost:8000/api/metrics | jq
   ```

**Expected Results**:
- ✅ `/api/status` returns JSON with fields:
  - `uptime_sec` (number, >0)
  - `active_connections` (0 initially)
  - `buffer_depth` (number, 0-20)
  - `device` ("Metal", "CUDA", or "CPU")
  - `soundfont_loaded` (true)
  - `synthesis_active` (true)

- ✅ `/api/metrics` returns JSON with fields:
  - `synthesis_latency_ms` (object with avg, p50, p95, p99)
  - `memory_usage_mb` (number, <500)
  - `buffer_underruns` (number, initially 0)

**Validation**:
```bash
# Extract key metrics
curl -s http://localhost:8000/api/status | jq '.soundfont_loaded, .device'
# Should return: true, "Metal" (or "CUDA"/"CPU")

curl -s http://localhost:8000/api/metrics | jq '.memory_usage_mb'
# Should return: <500
```

---

## Scenario 3: WebSocket Connection

**Objective**: Verify WebSocket endpoint accepts connections and begins streaming.

**Steps**:
1. Open browser to http://localhost:8000

2. Open browser DevTools (F12) → Network tab → Filter: WS (WebSocket)

3. Observe WebSocket connection at `ws://localhost:8000/ws/stream`

**Expected Results**:
- ✅ WebSocket connection established (Status: 101 Switching Protocols)
- ✅ Messages received every ~100ms
- ✅ Message type: `"audio"` (JSON format)
- ✅ Audio auto-plays within 2 seconds

**Manual Inspection**:
```javascript
// In browser console:
const ws = new WebSocket('ws://localhost:8000/ws/stream');

ws.onopen = () => console.log('Connected');
ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    console.log(`Received chunk #${msg.seq}, timestamp: ${msg.timestamp}`);
};
ws.onerror = (error) => console.error('WebSocket error:', error);
ws.onclose = () => console.log('Disconnected');
```

**Validation**:
- Message count increases steadily (10 messages per second)
- Sequence numbers (`seq`) increment monotonically
- No error events in console

---

## Scenario 4: Audio Playback Quality

**Objective**: Verify audio streams smoothly without glitches or dropouts.

**Steps**:
1. Open http://localhost:8000 in browser
2. Ensure audio auto-plays (unmute if needed)
3. Listen for 5 minutes continuously

**Expected Results**:
- ✅ Audio begins playing within 2 seconds of page load
- ✅ No audible glitches, clicks, or pops
- ✅ No silence gaps or dropouts
- ✅ Smooth harmonic transitions (no abrupt key changes)
- ✅ Reverb creates spacious ambient atmosphere

**Quality Checklist**:
- [ ] Piano timbre sounds realistic (not synthetic)
- [ ] Pad timbre is warm and lush (not harsh)
- [ ] Notes decay naturally (no sudden cutoffs)
- [ ] Reverb tail audible after notes end
- [ ] Stereo field is wide (not mono)

**Common Issues**:
- ❌ Clicking/popping sounds → Check buffer health (see Scenario 6)
- ❌ Flat/dry sound → Verify reverb enabled (check FluidSynth config)
- ❌ Delayed playback → Check end-to-end latency (see Scenario 8)

---

## Scenario 5: Musical Parameter Controls

**Objective**: Verify UI controls update musical generation parameters.

**Steps**:
1. Open http://localhost:8000

2. **Test Key Selection**:
   - Change key dropdown to "G major"
   - Wait 10-30 seconds (next phrase boundary)
   - Verify new chords sound in G major tonality

3. **Test Mode Selection**:
   - Change mode dropdown to "Lydian"
   - Wait 10-30 seconds
   - Verify bright, uplifting sound (characteristic of Lydian)

4. **Test Intensity Slider**:
   - Move intensity slider to 0.8
   - Observe note density increases (more notes per phrase)

5. **Test BPM Slider**:
   - Move BPM slider to 40 (slow)
   - Observe harmonic rhythm slows down

**Expected Results**:
- ✅ Changes apply within 5 seconds (at phrase boundary)
- ✅ Key changes are audibly distinct
- ✅ Mode changes affect harmonic character
- ✅ Intensity slider modulates note density
- ✅ BPM slider changes tempo perceptibly

**Validation**:
- Open browser console and inspect WebSocket messages:
  ```javascript
  // Send control message
  ws.send(JSON.stringify({
      type: 'update_params',
      params: { bpm: 80, intensity: 0.7 },
      timestamp: Date.now() / 1000
  }));
  ```

---

## Scenario 6: Preset Selection

**Objective**: Verify presets apply correct parameter configurations.

**Steps**:
1. Click **Focus** preset button
   - Verify mode changes to Dorian, BPM to 60, intensity to 0.5

2. Click **Meditation** preset button
   - Verify mode changes to Aeolian, BPM to 50, intensity to 0.3

3. Click **Sleep** preset button
   - Verify mode changes to Phrygian, BPM to 40, intensity to 0.2

4. Click **Bright** preset button
   - Verify mode changes to Lydian, BPM to 70, intensity to 0.6

**Expected Results**:
- ✅ Presets apply instantly (no delay)
- ✅ UI controls update to reflect preset values
- ✅ Musical characteristics match preset intent:
  - **Focus**: Balanced, medium tempo, clear but not dense
  - **Meditation**: Slow, sparse, calming
  - **Sleep**: Very slow, minimal, gentle
  - **Bright**: Uplifting, faster, moderately dense

---

## Scenario 7: Settings Persistence (localStorage)

**Objective**: Verify settings persist across browser refresh.

**Steps**:
1. Adjust controls:
   - Key: E minor
   - Mode: Dorian
   - Intensity: 0.7
   - BPM: 70

2. Refresh browser page (F5 or Cmd+R)

3. Observe controls restore to previous values

**Expected Results**:
- ✅ All control values match pre-refresh state
- ✅ Audio resumes with saved settings

**Manual Validation**:
```javascript
// In browser console:
// Check localStorage
localStorage.getItem('auralis_key');       // Should return "64" (E)
localStorage.getItem('auralis_mode');      // Should return "dorian"
localStorage.getItem('auralis_intensity'); // Should return "0.7"
localStorage.getItem('auralis_bpm');       // Should return "70"
```

---

## Scenario 8: Latency Measurement

**Objective**: Verify end-to-end latency meets <800ms target.

**Steps**:
1. Open browser DevTools → Console

2. Run latency measurement script:
   ```javascript
   const latencies = [];

   ws.onmessage = (event) => {
       const msg = JSON.parse(event.data);
       const now = Date.now() / 1000;
       const latency = (now - msg.timestamp) * 1000;  // Convert to ms
       latencies.push(latency);

       if (latencies.length === 100) {
           latencies.sort((a, b) => a - b);
           const p95 = latencies[94];  // 95th percentile
           console.log(`Latency p95: ${p95.toFixed(1)}ms`);
       }
   };
   ```

3. Wait for 100 samples (~10 seconds)

**Expected Results**:
- ✅ P95 latency <800ms (target: 500ms)
- ✅ Median latency <500ms
- ✅ Max latency <1000ms

**Troubleshooting High Latency**:
- Network issues: Check `ping localhost` (should be <1ms)
- Synthesis slow: Check `/api/metrics` synthesis_latency_ms
- Buffer health: Check buffer depth in `/api/status`

---

## Scenario 9: Buffer Health Monitoring

**Objective**: Verify adaptive buffering prevents underruns.

**Steps**:
1. Open http://localhost:8000/debug (if debug UI exists)
   - OR monitor `/api/metrics` endpoint:
   ```bash
   watch -n 1 "curl -s http://localhost:8000/api/metrics | jq '.buffer_underruns, .buffer_overflows'"
   ```

2. Stream audio for 10 minutes

3. Observe buffer underrun/overflow counters

**Expected Results**:
- ✅ Buffer underruns: 0 (on stable network)
- ✅ Buffer overflows: 0
- ✅ Buffer depth stays in 3-15 chunk range

**Simulating Network Jitter** (advanced):
```bash
# macOS: Use Network Link Conditioner
# Linux: Use tc (traffic control)
sudo tc qdisc add dev eth0 root netem delay 50ms 20ms distribution normal

# Stream audio and observe adaptive buffering
# Expected: Client increases buffer size to compensate

# Remove network constraints
sudo tc qdisc del dev eth0 root
```

---

## Scenario 10: Concurrent Client Connections

**Objective**: Verify server handles 10+ concurrent WebSocket clients.

**Steps**:
1. Open 10 browser tabs to http://localhost:8000

2. Verify audio plays in all tabs simultaneously

3. Monitor server metrics:
   ```bash
   curl -s http://localhost:8000/api/status | jq '.active_connections'
   # Should return: 10
   ```

4. Check synthesis latency remains stable:
   ```bash
   curl -s http://localhost:8000/api/metrics | jq '.synthesis_latency_ms.p95'
   # Should remain <100ms
   ```

**Expected Results**:
- ✅ All 10 clients receive audio without glitches
- ✅ Synthesis latency remains <100ms (p95)
- ✅ Memory usage <500MB (check `/api/metrics`)
- ✅ No server errors in console

**Stress Test** (advanced):
- Open 20 tabs (2× target capacity)
- Expected: Audio quality degrades gracefully, no crashes

---

## Scenario 11: Reconnection Behavior

**Objective**: Verify auto-reconnect after network disconnect.

**Steps**:
1. Open http://localhost:8000 and start streaming

2. Simulate disconnect:
   ```bash
   # Stop server (Ctrl+C in uvicorn terminal)
   ```

3. Observe client behavior:
   - Browser console should show "Disconnected" message
   - Client attempts reconnection with exponential backoff

4. Restart server:
   ```bash
   uv run uvicorn server.main:app --reload --host 0.0.0.0 --port 8000
   ```

5. Observe client auto-reconnects within 1-5 seconds

**Expected Results**:
- ✅ Client detects disconnect immediately
- ✅ Exponential backoff visible (1s, 2s, 4s intervals)
- ✅ Reconnection succeeds within 10 seconds of server restart
- ✅ Audio resumes smoothly (no manual refresh needed)

**Manual Reconnection Test**:
```javascript
// In browser console:
ws.close();  // Force disconnect

// Observe automatic reconnection attempts
// Check logs for backoff delays: 1s, 2s, 4s, 8s, ...
```

---

## Scenario 12: Long-Running Session (Memory Stability)

**Objective**: Verify no memory leaks over extended sessions.

**Steps**:
1. Start server and begin streaming

2. Monitor memory usage over time:
   ```bash
   # Record memory every 10 minutes for 1 hour
   for i in {1..6}; do
       curl -s http://localhost:8000/api/metrics | jq '.memory_usage_mb'
       sleep 600  # 10 minutes
   done
   ```

3. Plot memory growth:
   - Initial memory: ~350MB (SoundFonts + baseline)
   - After 1 hour: Should be <370MB (+20MB growth acceptable)
   - After 8 hours: Should be <450MB (target: <500MB)

**Expected Results**:
- ✅ Memory growth <10MB per hour (linear, stable)
- ✅ No sudden memory spikes
- ✅ Total memory <500MB after 8 hours

**Memory Leak Indicators** (troubleshooting):
- ❌ Memory grows >50MB per hour → Investigate with `tracemalloc`
- ❌ Memory spikes on parameter changes → Check buffer pool reuse
- ❌ Memory grows with client count → Check WebSocket cleanup

---

## Scenario 13: Browser Compatibility

**Objective**: Verify functionality across Chrome, Edge, Safari.

**Steps**:
1. **Chrome 90+**:
   - Open http://localhost:8000
   - Verify audio plays, controls work
   - Check DevTools console for errors

2. **Edge 90+**:
   - Repeat Chrome test
   - Verify identical behavior (same Chromium engine)

3. **Safari 14.1+**:
   - Open http://localhost:8000
   - **Safari-specific**: May require user click to start AudioContext
   - Click anywhere on page if audio doesn't auto-play
   - Verify controls work

**Expected Results**:
- ✅ Chrome: Full functionality, no errors
- ✅ Edge: Identical to Chrome
- ✅ Safari: Full functionality (may require click for auto-play)

**Safari Auto-Play Workaround**:
```javascript
// In client code:
document.addEventListener('click', () => {
    if (audioContext.state === 'suspended') {
        audioContext.resume();
    }
}, { once: true });
```

---

## Scenario 14: Error Handling

**Objective**: Verify graceful error handling and user feedback.

**Steps**:
1. **Missing SoundFont**:
   - Rename SoundFont file temporarily
   - Restart server
   - Expected: Server logs error, returns 500 on API calls

2. **GPU Unavailable**:
   - Force CPU-only mode (disable GPU drivers)
   - Expected: Server logs "GPU not available, using CPU"
   - Audio still works (CPU fallback)

3. **Invalid Control Parameters**:
   - Send invalid JSON via WebSocket:
     ```javascript
     ws.send(JSON.stringify({
         type: 'update_params',
         params: { bpm: 999, intensity: 5.0 },  // Invalid
         timestamp: Date.now() / 1000
     }));
     ```
   - Expected: Server ignores invalid values, no crash

**Expected Results**:
- ✅ Clear error messages in logs
- ✅ Server remains stable (no crashes)
- ✅ Client displays user-friendly error (if UI implemented)

---

## Scenario 15: Synthesis Latency Benchmark

**Objective**: Verify synthesis meets <100ms target.

**Steps**:
1. Run synthesis benchmark:
   ```bash
   uv run python -m tests.performance.test_batch_synthesis
   ```

2. Review latency statistics:
   - Mean: Should be <50ms
   - P95: Should be <100ms
   - P99: Should be <150ms

**Expected Results**:
- ✅ Mean synthesis latency <50ms
- ✅ P95 synthesis latency <100ms
- ✅ No synthesis timeouts

**Manual Benchmark** (if automated test unavailable):
```python
import time
# Assuming synth is initialized
start = time.perf_counter()
audio = synth.render_phrase(events, duration_sec=8.0)
end = time.perf_counter()
latency_ms = (end - start) * 1000
print(f"Synthesis latency: {latency_ms:.1f}ms")
# Target: <100ms
```

---

## Troubleshooting Guide

### Audio Not Playing
1. Check browser console for errors
2. Verify WebSocket connected (`ws.readyState === 1`)
3. Ensure SoundFonts loaded (`/api/status` → `soundfont_loaded: true`)
4. Try manual AudioContext resume (Safari)

### Glitchy Audio
1. Check buffer underruns (`/api/metrics` → `buffer_underruns`)
2. Verify network latency (<100ms ping)
3. Check synthesis latency (`/api/metrics` → `synthesis_latency_ms.p95`)
4. Reduce concurrent client count if >10

### High Latency
1. Check `/api/metrics` for bottlenecks:
   - Synthesis slow → Optimize FluidSynth settings
   - Network slow → Check local network
2. Verify GPU acceleration enabled (`/api/status` → `device: "Metal"`)

### Memory Growth
1. Monitor with `tracemalloc` or `/api/metrics`
2. Check GC tuning applied (`gc.get_threshold()`)
3. Verify buffer pool reuse (no new allocations in hot path)

---

## Success Criteria Summary

| Scenario | Metric | Target | Status |
|----------|--------|--------|--------|
| Server Startup | Starts without errors | Yes | ⬜ |
| Health Check | `/api/status` returns valid JSON | Yes | ⬜ |
| WebSocket | Connection established, messages flow | Yes | ⬜ |
| Audio Quality | No glitches/dropouts in 5 min | Yes | ⬜ |
| Parameter Controls | Changes apply within 5s | Yes | ⬜ |
| Presets | All 4 presets work | Yes | ⬜ |
| localStorage | Settings persist on refresh | Yes | ⬜ |
| Latency | P95 <800ms end-to-end | Yes | ⬜ |
| Buffer Health | 0 underruns on stable network | Yes | ⬜ |
| Concurrent Clients | 10+ clients, no degradation | Yes | ⬜ |
| Reconnection | Auto-reconnect within 10s | Yes | ⬜ |
| Memory Stability | <500MB after 8 hours | Yes | ⬜ |
| Browser Compatibility | Chrome, Edge, Safari work | Yes | ⬜ |
| Error Handling | Graceful errors, no crashes | Yes | ⬜ |
| Synthesis Benchmark | P95 <100ms | Yes | ⬜ |

---

## Next Steps

After completing these scenarios:

1. **Pass**: All scenarios meet success criteria → Ready for automated testing
2. **Fail**: Issues found → Document bugs, create tasks for fixes
3. **Performance**: Latency/memory targets missed → Optimization needed

**Automated Testing**:
- Run integration tests: `pytest tests/integration/`
- Run performance benchmarks: `pytest tests/performance/`
- Run memory stability test: `pytest tests/integration/test_memory_stability.py`

**Documentation**:
- Update README with quickstart instructions
- Document known issues
- Create user guide for controls/presets

---

**Testing Status**: ⬜ Not Started | 🟡 In Progress | ✅ Passed | ❌ Failed
