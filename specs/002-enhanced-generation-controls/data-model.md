# Data Model: Enhanced Generation & Controls

**Feature**: 001-enhanced-generation-controls  
**Date**: 2025-12-26  
**Status**: Design Complete

## Overview

This feature introduces in-memory data structures for managing generation presets and parameter history. All data is stored temporarily in memory with no persistence requirements, aligning with the real-time audio streaming architecture.

## Entities

### GenerationPreset

Represents saved combinations of generation parameters for quick access and reuse.

**Fields**:
- `name`: String (required, 1-50 characters) - Human-readable identifier for the preset
- `parameter_values`: Dict (required) - Complete set of generation parameters including melody_complexity, chord_progression_variety, harmonic_density, key, bpm, intensity
- `creation_date`: DateTime (auto-generated) - When the preset was created

**Validation Rules**:
- Name must be unique among active presets
- Parameter values must pass conflict validation
- All numeric parameters within defined ranges (0.0-1.0 for floats, valid musical ranges for others)

**Relationships**: None (standalone entity)

### ParameterHistory

Tracks recent parameter changes for undo/redo functionality and debugging.

**Fields**:
- `changes`: List[Dict] (required) - Sequence of parameter change events
  - Each change dict contains: timestamp, parameter_name, old_value, new_value, source (user/auto)
- `max_entries`: Integer (default: 50) - Maximum history entries to retain

**Validation Rules**:
- Changes list limited to max_entries (FIFO eviction)
- Timestamps must be valid DateTime objects
- Parameter names must match defined parameter schema

**Relationships**: None (standalone entity)

## Data Flow

1. **Preset Creation**: User saves current parameters → GenerationPreset created with validation
2. **Preset Loading**: User selects preset → Parameters applied with conflict checking
3. **Parameter Changes**: All changes logged to ParameterHistory for undo support
4. **Automatic Adjustments**: Overload adjustments logged as source='auto' changes

## Constraints

- All data stored in memory only (no database persistence)
- Memory usage bounded by preset count and history depth
- No cross-session data sharing (resets on server restart)
- Thread-safe access required for concurrent WebSocket connections</content>
<parameter name="filePath">/Users/andrew/Develop/auralis/specs/001-enhanced-generation-controls/data-model.md