# HTTP API Contracts: Enhanced Generation & Controls

**Feature**: 002-enhanced-generation-controls  
**Date**: 2025-12-26  
**Base URL**: `/api`

## Overview

This document extends the Phase 1 HTTP API with additional parameter controls and preset management endpoints. All endpoints maintain backward compatibility with Phase 1 clients.

## Updated Schemas

### SynthesisParameters (Extended)

```json
{
  "key": "string (default: 'A')",
  "bpm": "integer 40-120 (default: 70)",
  "intensity": "float 0.0-1.0 (default: 0.5)",
  "melody_complexity": "float 0.0-1.0 (default: 0.5) - NEW",
  "chord_progression_variety": "float 0.0-1.0 (default: 0.5) - NEW",
  "harmonic_density": "float 0.0-1.0 (default: 0.5) - NEW"
}
```

### GenerationPreset

```json
{
  "name": "string (required, 1-50 chars)",
  "parameter_values": "SynthesisParameters object",
  "creation_date": "datetime (read-only)"
}
```

## Endpoints

### GET /api/status

**Purpose**: Get server status and current parameters (unchanged from Phase 1)

**Response**:
```json
{
  "status": "running",
  "uptime_seconds": 3600,
  "synthesis_engine": {...},
  "buffer_depth_ms": 50.0,
  "active_connections": 2,
  "chunks_generated": 1000,
  "parameters": {
    "key": "A",
    "bpm": 70,
    "intensity": 0.5,
    "melody_complexity": 0.5,
    "chord_progression_variety": 0.5,
    "harmonic_density": 0.5
  }
}
```

### POST /api/control

**Purpose**: Update synthesis parameters (extended with new parameters)

**Request Body**: SynthesisParameters object

**Response**:
```json
{
  "message": "Parameters updated successfully",
  "parameters": {...}
}
```

**Validation**: Parameters validated for conflicts, automatic adjustment if needed

### GET /api/presets

**Purpose**: List available generation presets

**Response**:
```json
{
  "presets": [
    {
      "name": "Ambient Dream",
      "parameter_values": {...},
      "creation_date": "2025-12-26T10:00:00Z"
    }
  ]
}
```

### POST /api/presets

**Purpose**: Save current parameters as a preset

**Request Body**:
```json
{
  "name": "string (required)"
}
```

**Response**:
```json
{
  "message": "Preset saved",
  "preset": {...}
}
```

### DELETE /api/presets/{name}

**Purpose**: Delete a preset

**Response**:
```json
{
  "message": "Preset deleted"
}
```

### POST /api/presets/{name}/load

**Purpose**: Load a preset's parameters

**Response**:
```json
{
  "message": "Preset loaded",
  "parameters": {...}
}
```

## Error Responses

All endpoints return standard HTTP status codes with JSON error details:

```json
{
  "error": "ValidationError",
  "message": "Parameter conflict detected",
  "details": {...}
}
```

## WebSocket Integration

Parameter updates are broadcast to all connected WebSocket clients with the extended parameter set.</content>
<parameter name="filePath">/Users/andrew/Develop/auralis/specs/001-enhanced-generation-controls/contracts/http-api.md