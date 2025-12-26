# Quick Start: Enhanced Generation & Controls

**Feature**: 002-enhanced-generation-controls  
**Date**: 2025-12-26  
**Target User**: Music producers and ambient music enthusiasts

## Overview

This guide shows how to use the enhanced music generation controls and preset management features. Start with the basic server setup, then explore the new parameter options.

## Prerequisites

- Auralis server running (`uv run uvicorn server.main:app --reload`)
- Web browser with Web Audio API support
- Basic understanding of music parameters (key, BPM, intensity)

## Getting Started

1. **Open the Web Interface**
   ```
   http://localhost:8000
   ```

2. **Verify Enhanced Controls**
   - Look for new sliders: Melody Complexity, Chord Progression Variety, Harmonic Density
   - These appear alongside the existing Key, BPM, and Intensity controls

3. **Experiment with Parameters**
   - **Melody Complexity**: 0.0 = simple repeating patterns, 1.0 = intricate melodic variations
   - **Chord Progression Variety**: 0.0 = predictable progressions, 1.0 = diverse harmonic exploration
   - **Harmonic Density**: 0.0 = sparse harmonies, 1.0 = rich layered chords

## Using Presets

### Saving a Preset

1. Adjust parameters to your liking
2. Use the API to save:
   ```bash
   curl -X POST http://localhost:8000/api/presets \
        -H "Content-Type: application/json" \
        -d '{"name": "Ambient Dream"}'
   ```

### Loading a Preset

1. Load via API:
   ```bash
   curl -X POST http://localhost:8000/api/presets/Ambient%20Dream/load
   ```
2. Music generation updates immediately with the loaded parameters

### Managing Presets

- **List all presets**:
  ```bash
  curl http://localhost:8000/api/presets
  ```

- **Delete a preset**:
  ```bash
  curl -X DELETE http://localhost:8000/api/presets/Ambient%20Dream
  ```

## Advanced Usage

### Parameter Validation

The system automatically validates parameter combinations:
- Conflicts are highlighted with warnings
- Invalid combinations are adjusted automatically to maintain performance
- Feedback appears in real-time as you adjust controls

### Performance Monitoring

Monitor the impact of enhanced algorithms:
- Check `/api/status` for buffer depth and latency metrics
- Enhanced algorithms may increase CPU/GPU usage but stay within <100ms latency

### API Integration

For programmatic control:

```python
import requests

# Update parameters
params = {
    "key": "C",
    "bpm": 80,
    "intensity": 0.7,
    "melody_complexity": 0.8,
    "chord_progression_variety": 0.6,
    "harmonic_density": 0.4
}
response = requests.post("http://localhost:8000/api/control", json=params)
```

## Troubleshooting

- **No new controls visible**: Ensure you're using the updated client interface
- **Parameter changes not applying**: Check for validation warnings in API responses
- **Audio glitches**: Monitor `/api/status` for buffer issues; reduce complexity if needed
- **High latency**: Enhanced algorithms require more processing; reduce parameters if >100ms

## Next Steps

- Experiment with different parameter combinations to find your preferred styles
- Create multiple presets for different moods or projects
- Monitor performance metrics to optimize for your hardware</content>
<parameter name="filePath">/Users/andrew/Develop/auralis/specs/001-enhanced-generation-controls/quickstart.md