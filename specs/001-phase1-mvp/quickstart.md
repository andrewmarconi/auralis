# Quick Start Guide: Phase 1 MVP - Real-time Ambient Music Streaming

**Version**: 1.0.0  
**Target Audience**: General consumers seeking ambient background music  
**Platform**: Chrome/Edge browsers (last 3 years)

---

## Overview

Auralis Phase 1 MVP provides real-time generated ambient music streaming directly to your web browser. No installation required - just open the web client and start listening to continuously generated ambient music within seconds.

**Features**:
- Real-time ambient music generation using Markov chord progressions
- Constraint-based melody generation with harmonic coherence  
- GPU-accelerated audio synthesis for optimal performance
- WebSocket streaming with <100ms audio chunks
- Automatic playback with minimal setup

---

## System Requirements

### Browser Support
- **Google Chrome**: Version 90+ (last 3 years)
- **Microsoft Edge**: Version 90+ (last 3 years)
- **Required**: Web Audio API support
- **Recommended**: Stable internet connection

### Performance Requirements
- **Network**: Broadband connection (1+ Mbps recommended)
- **Latency**: <300ms round-trip to server for optimal experience
- **Device**: Any modern computer with web browser

---

## Getting Started

### 1. Access the Application
Open your Chrome or Edge browser and navigate to:
```
http://localhost:8000
```

### 2. Start Streaming
The web client will automatically:
1. **Connect** to the WebSocket streaming endpoint
2. **Buffer** initial audio chunks (2-3 chunks ~300ms)
3. **Start** ambient music playback within 2 seconds
4. **Display** current status and buffer depth

### 3. Control Music (Optional)
Use the web interface controls to adjust:
- **Key**: Musical key (A, B, C, D, E, F, G)
- **BPM**: Tempo (40-120 beats per minute)
- **Intensity**: Generation density (0.0-1.0)

Changes apply within ~20 seconds (next phrase generation).

---

## User Interface

### Main Display
```
┌─────────────────────────────────────────┐
│  Auralis - Ambient Music Streamer   │
│                                     │
│  Status: Streaming                   │
│  Buffer: 150ms                     │
│  Phrase: #23                       │
│                                     │
│  ┌─────────────────────────────────┐    │
│  │      Control Panel            │    │
│  │                             │    │
│  │  Key: [A ▼]               │    │
│  │  BPM: [70 ▼]              │    │
│  │  Intensity: [0.5 ▼]        │    │
│  │                             │    │
│  │  [Start] [Stop]            │    │
│  └─────────────────────────────────┘    │
│                                     │
│  Volume: [████████▓▓▓▓▓▓▓▓▓]     │
└─────────────────────────────────────────┘
```

### Control Actions
- **Start/Stop**: Control music playback
- **Key Selection**: Change musical tonality
- **BPM Adjustment**: Modify tempo (40-120)
- **Intensity Slider**: Control generation density
- **Volume Control**: Adjust playback volume

---

## Troubleshooting

### Common Issues

#### No Audio Playback
**Symptoms**: Interface loads but no sound
**Solutions**:
1. Check browser volume and system mute settings
2. Verify Chrome/Edge is updated to latest version
3. Confirm internet connection is stable
4. Refresh page and allow audio permissions

#### Buffering Issues
**Symptoms**: Audio cuts out or stutters
**Solutions**:
1. Check network connection speed
2. Close other browser tabs using bandwidth
3. Verify server is running (`http://localhost:8000/api/health`)
4. Try refreshing the page

#### Control Changes Not Applied
**Symptoms**: Parameter changes don't affect music
**Solutions**:
1. Wait 20 seconds for next phrase generation
2. Verify values are within valid ranges
3. Check browser console for error messages

### Error Messages
- **"GPU Unavailable, falling back to CPU"**: Normal fallback, music continues
- **"Connection lost, attempting to reconnect"**: Network issue, auto-recovery enabled
- **"Browser not supported"**: Update to Chrome/Edge latest version

---

## Advanced Usage

### Musical Characteristics
The generated ambient music features:
- **Chord Progressions**: 8-bar phrases with smooth harmonic movement
- **Melody Style**: Sparse, long-sustained notes (2-3 per bar)
- **Texture**: Monophonic lead + pad sounds for depth
- **Tempo**: Default 70 BPM (typical ambient pace)

### Performance Monitoring
Monitor these metrics for optimal experience:
- **Buffer Depth**: Target 100-200ms for stable playback
- **Latency**: Total delay <800ms for responsive feel
- **GPU Status**: "GPU Accelerated" indicates optimal performance

---

## Technical Details

### Audio Specifications
- **Sample Rate**: 44.1kHz (CD quality)
- **Bit Depth**: 16-bit PCM (standard audio format)
- **Channels**: Mono (converted to stereo by browser)
- **Chunk Size**: 100ms (balance latency/bandwidth)

### Generation Algorithm
- **Chord Progressions**: Markov chain with ambient-optimized transitions
- **Melody Generation**: Constraint-based fitting current harmony
- **Synthesis**: torchsynth with GPU acceleration
- **Real-time Performance**: <5 second phrase generation

---

## Support

### Getting Help
- **Documentation**: Check system status at `/api/status`
- **Health Check**: Verify server operation at `/api/health`
- **Browser Console**: Press F12 for technical error information
- **Performance Issues**: Note buffer depth and latency metrics

### Contact & Feedback
Report issues with:
- Browser version and type
- Network connection details  
- Error messages from browser console
- Steps to reproduce the problem

---

## Keyboard Shortcuts

| Action | Shortcut | Description |
|---------|-----------|-------------|
| Play/Pause | Spacebar | Toggle audio playback |
| Stop | S | Stop streaming completely |
| Refresh | F5 | Restart connection and reload |
| Volume Up | ↑ | Increase volume 10% |
| Volume Down | ↓ | Decrease volume 10% |

---

**Enjoy your continuously generated ambient music experience!** 🎵