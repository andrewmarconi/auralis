# HTTP API Contract: Control Endpoints

**Version**: 1.0.0  
**Feature**: Phase 1 MVP Ambient Music Streaming  
**Created**: 2024-12-26

---

## Base URL
```
http://localhost:8000/api
```

---

## Control Parameters Endpoint

### Set Generation Parameters
**Method**: `POST /api/control`

**Request Schema**:
```json
{
  "key": "A",
  "bpm": 70,
  "intensity": 0.5
}
```

**Field Descriptions**:
- `key`: Musical key (A, B, C, D, E, F, G with minor mode default)
- `bpm`: Tempo in beats per minute (40-120 range)
- `intensity`: Generation density (0.0-1.0 range)

**Response Schema**:
```json
{
  "status": "updated",
  "timestamp": 1703123456789,
  "parameters": {
    "key": "A",
    "bpm": 70,
    "intensity": 0.5
  }
}
```

**Constraints**:
- Key must be valid musical key (case-insensitive)
- BPM must be within 40-120 range
- Intensity must be between 0.0 and 1.0
- All parameters optional (defaults preserved if not provided)

---

## Status Endpoint

### Get System Status
**Method**: `GET /api/status`

**Response Schema**:
```json
{
  "status": "streaming",
  "timestamp": 1703123456789,
  "metrics": {
    "buffer_depth_ms": 150,
    "current_phrase": 12,
    "synthesis_latency_ms": 45,
    "gpu_accelerated": true
  },
  "parameters": {
    "key": "A",
    "bpm": 70,
    "intensity": 0.5
  }
}
```

**Field Descriptions**:
- `status`: Current system state ("idle", "streaming", "error")
- `timestamp`: Unix timestamp in milliseconds
- `metrics.buffer_depth_ms`: Current audio buffer depth in milliseconds
- `metrics.current_phrase`: Current phrase generation number
- `metrics.synthesis_latency_ms`: Last phrase synthesis time in milliseconds
- `metrics.gpu_accelerated`: Whether GPU acceleration is active

---

## Health Check Endpoint

### System Health
**Method**: `GET /api/health`

**Response Schema**:
```json
{
  "status": "healthy",
  "timestamp": 1703123456789,
  "checks": {
    "synthesis_engine": "ok",
    "ring_buffer": "ok",
    "websocket_server": "ok",
    "gpu_available": true
  }
}
```

**Status Values**:
- `healthy`: All systems operational
- `degraded`: Some components operating with limitations
- `unhealthy`: Critical system failures

---

## Error Responses

### Standard Error Format
```json
{
  "error": {
    "code": "INVALID_PARAMETERS",
    "message": "BPM must be between 40 and 120",
    "timestamp": 1703123456789,
    "details": {
      "field": "bpm",
      "value": 150,
      "constraint": "40-120"
    }
  }
}
```

### Error Codes
- `INVALID_PARAMETERS`: Request validation failed
- `SYNTHESIS_ERROR`: Audio generation failed
- `BUFFER_ERROR`: Ring buffer operational issues
- `GPU_UNAVAILABLE`: GPU acceleration problems
- `INTERNAL_ERROR`: Unexpected server error

---

## Client Implementation Examples

### JavaScript Control API
```javascript
class AuralisControlAPI {
  constructor(baseUrl = 'http://localhost:8000/api') {
    this.baseUrl = baseUrl;
  }
  
  async setParameters(key, bpm, intensity) {
    const response = await fetch(`${this.baseUrl}/control`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        key: key || undefined,
        bpm: bpm || undefined,
        intensity: intensity || undefined
      })
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error.message);
    }
    
    return response.json();
  }
  
  async getStatus() {
    const response = await fetch(`${this.baseUrl}/status`);
    
    if (!response.ok) {
      throw new Error('Failed to get status');
    }
    
    return response.json();
  }
  
  async checkHealth() {
    const response = await fetch(`${this.baseUrl}/health`);
    
    if (!response.ok) {
      throw new Error('Health check failed');
    }
    
    return response.json();
  }
}

// Usage example
const api = new AuralisControlAPI();

// Update parameters
await api.setParameters('C', 80, 0.7);

// Get current status
const status = await api.getStatus();
console.log('Buffer depth:', status.metrics.buffer_depth_ms);
```

---

## Performance Requirements

### Response Times
- **Control Updates**: <200ms response time
- **Status Queries**: <100ms response time
- **Health Checks**: <50ms response time

### Rate Limiting
- No rate limiting for MVP (single user)
- Future: Consider 10 requests/second per client

### Data Validation
- All input validated against JSON schemas
- Invalid parameters rejected with detailed error messages
- SQL injection protection (not applicable for JSON)

---

## Security Considerations

### Input Sanitization
- Key values limited to valid musical notes
- BPM and intensity validated against numeric ranges
- JSON parsing protected against malformed input

### Error Information
- Error messages don't expose internal system details
- Stack traces filtered in production responses
- Rate limiting response headers included when appropriate