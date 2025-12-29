/**
 * Auralis Audio Client (Main Thread)
 *
 * Manages:
 * - AudioContext and AudioWorklet initialization
 * - WebSocket connection to server (/ws/stream)
 * - 4-tier adaptive buffering (Emergency/Low/Healthy/Full)
 * - EMA jitter tracking (α=0.1)
 * - UI updates
 */

class AuralisClient {
    constructor() {
        // Audio components
        this.audioContext = null;
        this.workletNode = null;

        // WebSocket connection
        this.ws = null;
        this.connectionState = 'disconnected'; // disconnected | connecting | connected

        // Adaptive buffering configuration (4 tiers)
        this.bufferTiers = {
            emergency: 300,  // <300ms: Critical underrun risk
            low: 500,        // 300-500ms: Low buffer, increase pre-buffering
            healthy: 700,    // 500-700ms: Optimal range
            full: 1000       // >700ms: High buffer, reduce latency
        };
        this.currentBufferTarget = this.bufferTiers.healthy; // Start at healthy tier
        this.bufferDepth = 0;
        this.bufferCapacity = 50;

        // Jitter tracking (EMA with α=0.1)
        this.jitterEMA = 0.0;
        this.jitterAlpha = 0.1;
        this.lastChunkTimestamp = null;

        // Metrics
        this.chunksReceived = 0;
        this.underruns = 0;
        this.latencyMS = 0;

        // Musical parameters (for localStorage persistence)
        this.currentParameters = {
            key: 60,
            mode: 'aeolian',
            bpm: 70,
            intensity: 0.5
        };

        // Preset definitions (matching server/presets.py)
        this.presets = {
            focus: { key: 62, mode: 'dorian', bpm: 60, intensity: 0.5 },
            meditation: { key: 60, mode: 'aeolian', bpm: 60, intensity: 0.3 },
            sleep: { key: 64, mode: 'phrygian', bpm: 60, intensity: 0.2 },
            bright: { key: 67, mode: 'lydian', bpm: 70, intensity: 0.6 }
        };

        // UI elements
        this.ui = {
            playButton: document.getElementById('playButton'),
            connectionIndicator: document.getElementById('connectionIndicator'),
            connectionStatus: document.getElementById('connectionStatus'),
            audioContextStatus: document.getElementById('audioContextStatus'),
            playbackState: document.getElementById('playbackState'),
            bufferHealth: document.getElementById('bufferHealth'),
            bufferBar: document.getElementById('bufferBar'),
            latencyMetric: document.getElementById('latencyMetric'),
            jitterMetric: document.getElementById('jitterMetric'),
            chunksMetric: document.getElementById('chunksMetric'),
            underrunsMetric: document.getElementById('underrunsMetric'),
            // Musical controls
            keySelect: document.getElementById('keySelect'),
            keyValue: document.getElementById('keyValue'),
            modeSelect: document.getElementById('modeSelect'),
            modeValue: document.getElementById('modeValue'),
            bpmRange: document.getElementById('bpmRange'),
            bpmValue: document.getElementById('bpmValue'),
            intensityRange: document.getElementById('intensityRange'),
            intensityValue: document.getElementById('intensityValue')
        };

        this.setupEventListeners();
        this.loadParametersFromStorage();
    }

    /**
     * Setup UI event listeners.
     */
    setupEventListeners() {
        this.ui.playButton.addEventListener('click', () => this.handlePlayButton());

        // Musical controls
        this.ui.keySelect.addEventListener('change', (e) => this.handleKeyChange(e));
        this.ui.modeSelect.addEventListener('change', (e) => this.handleModeChange(e));
        this.ui.bpmRange.addEventListener('input', (e) => this.handleBPMChange(e));
        this.ui.intensityRange.addEventListener('input', (e) => this.handleIntensityChange(e));

        // Preset buttons
        const presetButtons = document.querySelectorAll('.preset-button');
        presetButtons.forEach(button => {
            button.addEventListener('click', (e) => this.handlePresetClick(e));
        });

        // Update metrics every 500ms
        setInterval(() => this.updateMetrics(), 500);
    }

    /**
     * Handle play button click.
     */
    async handlePlayButton() {
        if (this.connectionState === 'disconnected') {
            await this.start();
        } else if (this.connectionState === 'connected') {
            await this.stop();
        }
    }

    /**
     * Start audio playback.
     */
    async start() {
        try {
            console.log('[Client] Starting Auralis client...');

            // Initialize AudioContext
            await this.initAudioContext();

            // Connect to WebSocket
            await this.connectWebSocket();

            // Update UI
            this.ui.playButton.textContent = 'Stop Listening';
            this.updatePlaybackState('Playing');

        } catch (error) {
            console.error('[Client] Failed to start:', error);
            this.updateConnectionState('disconnected');
            alert(`Failed to start playback: ${error.message}`);
        }
    }

    /**
     * Stop audio playback.
     */
    async stop() {
        console.log('[Client] Stopping Auralis client...');

        // Close WebSocket
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }

        // Suspend AudioContext
        if (this.audioContext && this.audioContext.state === 'running') {
            await this.audioContext.suspend();
        }

        // Update UI
        this.ui.playButton.textContent = 'Start Listening';
        this.updatePlaybackState('Stopped');
        this.updateConnectionState('disconnected');
    }

    /**
     * Initialize AudioContext and load AudioWorklet processor.
     */
    async initAudioContext() {
        console.log('[Client] Initializing AudioContext...');

        // Create AudioContext
        this.audioContext = new AudioContext({ sampleRate: 44100 });

        // Check actual sample rate (browser may ignore requested rate)
        console.log(`[Client] AudioContext sample rate: requested=44100, actual=${this.audioContext.sampleRate}`);
        if (this.audioContext.sampleRate !== 44100) {
            console.warn(`[Client] WARNING: Sample rate mismatch! Server=44100, Browser=${this.audioContext.sampleRate}`);
            console.warn(`[Client] This will cause pitch shift and audio artifacts!`);
        }

        // Update UI
        this.updateAudioContextState(this.audioContext.state);

        // Load AudioWorklet processor
        await this.audioContext.audioWorklet.addModule('/client/audio_worklet_processor.js');

        // Create AudioWorkletNode
        this.workletNode = new AudioWorkletNode(this.audioContext, 'auralis-worklet-processor', {
            numberOfInputs: 0,
            numberOfOutputs: 1,
            outputChannelCount: [2] // Stereo
        });

        // Listen for messages from worklet
        this.workletNode.port.onmessage = (event) => {
            this.handleWorkletMessage(event.data);
        };

        // Connect worklet to destination (speakers)
        this.workletNode.connect(this.audioContext.destination);

        // Resume AudioContext (required for auto-play policies)
        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
            this.updateAudioContextState(this.audioContext.state);
        }

        console.log('[Client] AudioContext initialized');
    }

    /**
     * Handle messages from AudioWorklet processor.
     *
     * @param {Object} message - Message from worklet
     */
    handleWorkletMessage(message) {
        switch (message.type) {
            case 'bufferHealth':
                this.bufferDepth = message.depth;
                this.chunksReceived = message.chunksReceived;
                this.underruns = message.underruns;
                this.updateBufferHealth();

                // Log buffer health periodically (every 20 chunks)
                if (message.chunksReceived % 20 === 0) {
                    console.log(`[Client] Buffer: ${message.depth}/${message.capacity} chunks, buffering: ${message.isBuffering}, underruns: ${message.underruns}`);
                }
                break;

            case 'metrics':
                console.log('[Client] Worklet metrics:', message);
                break;

            default:
                console.warn('[Client] Unknown worklet message:', message.type);
        }
    }

    /**
     * Connect to WebSocket server.
     */
    async connectWebSocket() {
        return new Promise((resolve, reject) => {
            console.log('[Client] Connecting to WebSocket...');
            this.updateConnectionState('connecting');

            // Determine WebSocket URL
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const host = window.location.host;
            const wsUrl = `${protocol}//${host}/ws/stream`;

            this.ws = new WebSocket(wsUrl);
            this.ws.binaryType = 'arraybuffer';

            // Connection opened
            this.ws.onopen = () => {
                console.log('[Client] WebSocket connected');
                this.updateConnectionState('connected');
                resolve();
            };

            // Message received (audio chunk)
            this.ws.onmessage = (event) => {
                this.handleWebSocketMessage(event.data);
            };

            // Connection closed
            this.ws.onclose = () => {
                console.log('[Client] WebSocket disconnected');
                this.updateConnectionState('disconnected');
            };

            // Connection error
            this.ws.onerror = (error) => {
                console.error('[Client] WebSocket error:', error);
                this.updateConnectionState('disconnected');
                reject(new Error('WebSocket connection failed'));
            };
        });
    }

    /**
     * Handle WebSocket message (audio chunk).
     *
     * @param {string} data - JSON message from server
     */
    handleWebSocketMessage(data) {
        try {
            const message = JSON.parse(data);

            if (message.type === 'audio_chunk') {
                // The message IS the chunk (not nested)
                const chunk = message;

                // Track jitter
                this.trackJitter(chunk.timestamp);

                // Calculate latency
                const now = Date.now() / 1000.0;
                this.latencyMS = (now - chunk.timestamp) * 1000.0;

                // Apply adaptive buffering
                this.applyAdaptiveBuffering();

                // Decode base64 in main thread (atob not available in AudioWorklet)
                const decodedChunk = this.decodeChunkMainThread(chunk);

                // Send decoded chunk to worklet
                if (this.chunksReceived <= 5) {
                    console.log(`[Client] Sending decoded chunk ${chunk.seq} to worklet`);
                }
                this.workletNode.port.postMessage({
                    type: 'chunk',
                    chunk: decodedChunk
                });
                this.chunksReceived++;

            } else if (message.type === 'welcome') {
                console.log('[Client] Received welcome message:', message.message);
            } else if (message.type === 'error') {
                console.error('[Client] Server error:', message.message);
            } else {
                console.warn('[Client] Unknown message type:', message.type);
            }

        } catch (error) {
            console.error('[Client] Failed to parse WebSocket message:', error);
        }
    }

    /**
     * Decode base64 PCM chunk in main thread.
     *
     * @param {Object} chunk - Audio chunk with base64 data
     * @returns {Object} Decoded chunk { left: Float32Array, right: Float32Array, seq, timestamp }
     */
    decodeChunkMainThread(chunk) {
        // Decode base64 to ArrayBuffer
        const binaryString = atob(chunk.data);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }

        // Interpret as Int16Array (PCM)
        const int16Data = new Int16Array(bytes.buffer);

        // Deinterleave stereo samples (LRLRLR... → LL...RR...)
        const samplesPerChannel = int16Data.length / 2;
        const leftChannel = new Float32Array(samplesPerChannel);
        const rightChannel = new Float32Array(samplesPerChannel);

        for (let i = 0; i < samplesPerChannel; i++) {
            // Convert int16 [-32768, 32767] to float32 [-1.0, 1.0]
            leftChannel[i] = int16Data[i * 2] / 32768.0;
            rightChannel[i] = int16Data[i * 2 + 1] / 32768.0;
        }

        // DEBUG: Log first chunk stats
        if (chunk.seq % 50 === 0) {
            const leftMin = Math.min(...leftChannel);
            const leftMax = Math.max(...leftChannel);
            const leftRMS = Math.sqrt(leftChannel.reduce((sum, v) => sum + v * v, 0) / leftChannel.length);
            console.log(`[Client] Chunk ${chunk.seq} decoded: ${samplesPerChannel} samples, range [${leftMin.toFixed(4)}, ${leftMax.toFixed(4)}], RMS ${leftRMS.toFixed(4)}`);
        }

        return {
            left: leftChannel,
            right: rightChannel,
            seq: chunk.seq,
            timestamp: chunk.timestamp
        };
    }

    /**
     * Track jitter using EMA (Exponential Moving Average).
     *
     * @param {number} timestamp - Chunk timestamp
     */
    trackJitter(timestamp) {
        if (this.lastChunkTimestamp !== null) {
            const actualInterval = timestamp - this.lastChunkTimestamp;
            const expectedInterval = 0.1; // 100ms chunks
            const jitter = Math.abs(actualInterval - expectedInterval) * 1000.0; // Convert to ms

            // Update EMA: jitterEMA = α * jitter + (1 - α) * jitterEMA
            this.jitterEMA = this.jitterAlpha * jitter + (1 - this.jitterAlpha) * this.jitterEMA;
        }

        this.lastChunkTimestamp = timestamp;
    }

    /**
     * Apply adaptive buffering based on jitter and buffer depth.
     *
     * Tiers:
     * - Emergency (<300ms): Critical underrun risk
     * - Low (300-500ms): Increase pre-buffering
     * - Healthy (500-700ms): Optimal
     * - Full (>700ms): Reduce latency
     */
    applyAdaptiveBuffering() {
        const bufferMS = (this.bufferDepth / this.bufferCapacity) * (this.bufferCapacity * 100); // Approximate buffer in ms

        // Determine tier based on buffer depth and jitter
        let newTier;
        if (bufferMS < this.bufferTiers.emergency || this.jitterEMA > 50) {
            newTier = 'emergency';
            this.currentBufferTarget = this.bufferTiers.emergency;
        } else if (bufferMS < this.bufferTiers.low || this.jitterEMA > 30) {
            newTier = 'low';
            this.currentBufferTarget = this.bufferTiers.low;
        } else if (bufferMS < this.bufferTiers.healthy) {
            newTier = 'healthy';
            this.currentBufferTarget = this.bufferTiers.healthy;
        } else {
            newTier = 'full';
            this.currentBufferTarget = this.bufferTiers.full;
        }

        // Note: Actual buffer management happens in the worklet
        // This tier information could be sent to worklet for adaptive behavior
    }

    /**
     * Update connection state UI.
     *
     * @param {string} state - Connection state
     */
    updateConnectionState(state) {
        this.connectionState = state;

        this.ui.connectionIndicator.className = `status-indicator ${state}`;

        const statusText = {
            disconnected: 'Disconnected',
            connecting: 'Connecting...',
            connected: 'Connected'
        };
        this.ui.connectionStatus.textContent = statusText[state] || state;
    }

    /**
     * Update AudioContext state UI.
     *
     * @param {string} state - AudioContext state
     */
    updateAudioContextState(state) {
        this.ui.audioContextStatus.textContent = state.charAt(0).toUpperCase() + state.slice(1);
    }

    /**
     * Update playback state UI.
     *
     * @param {string} state - Playback state
     */
    updatePlaybackState(state) {
        this.ui.playbackState.textContent = state;
    }

    /**
     * Update buffer health UI.
     */
    updateBufferHealth() {
        const bufferPercentage = (this.bufferDepth / this.bufferCapacity) * 100;
        const bufferMS = this.bufferDepth * 100; // Approximate (100ms per chunk)

        // Determine health status
        let healthStatus;
        if (bufferMS < 300) {
            healthStatus = 'Emergency';
        } else if (bufferMS < 500) {
            healthStatus = 'Low';
        } else if (bufferMS < 700) {
            healthStatus = 'Healthy';
        } else {
            healthStatus = 'Full';
        }

        this.ui.bufferHealth.textContent = `${healthStatus} (${bufferMS.toFixed(0)}ms)`;
        this.ui.bufferBar.style.width = `${bufferPercentage}%`;
    }

    /**
     * Update metrics UI.
     */
    updateMetrics() {
        this.ui.latencyMetric.textContent = this.latencyMS.toFixed(0);
        this.ui.jitterMetric.textContent = this.jitterEMA.toFixed(1);
        this.ui.chunksMetric.textContent = this.chunksReceived;
        this.ui.underrunsMetric.textContent = this.underruns;
    }

    /**
     * Send control message to server.
     *
     * @param {Object} controlMessage - Control parameters
     */
    sendControlMessage(controlMessage) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'control',
                ...controlMessage
            }));
            console.log('[Client] Sent control message:', controlMessage);
        }
    }

    /**
     * Handle key selection change.
     */
    handleKeyChange(event) {
        const keyMIDI = parseInt(event.target.value);
        const keyNames = ['C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B'];
        const keyName = keyNames[keyMIDI % 12];

        this.ui.keyValue.textContent = keyName;
        this.currentParameters.key = keyMIDI;
        this.saveParametersToStorage();
        this.sendControlMessage({ key: keyMIDI });
    }

    /**
     * Handle mode selection change.
     */
    handleModeChange(event) {
        const mode = event.target.value;
        const modeNames = {
            'aeolian': 'Aeolian',
            'dorian': 'Dorian',
            'lydian': 'Lydian',
            'phrygian': 'Phrygian'
        };

        this.ui.modeValue.textContent = modeNames[mode];
        this.currentParameters.mode = mode;
        this.saveParametersToStorage();
        this.sendControlMessage({ mode: mode });
    }

    /**
     * Handle BPM slider change.
     */
    handleBPMChange(event) {
        const bpm = parseInt(event.target.value);
        this.ui.bpmValue.textContent = bpm;
        this.currentParameters.bpm = bpm;
        this.saveParametersToStorage();
        this.sendControlMessage({ bpm: bpm });
    }

    /**
     * Handle intensity slider change.
     */
    handleIntensityChange(event) {
        const intensity = parseFloat(event.target.value);
        this.ui.intensityValue.textContent = intensity.toFixed(1);
        this.currentParameters.intensity = intensity;
        this.saveParametersToStorage();
        this.sendControlMessage({ intensity: intensity });
    }

    /**
     * Handle preset button click.
     */
    handlePresetClick(event) {
        const presetName = event.target.dataset.preset;
        const preset = this.presets[presetName];

        if (!preset) {
            console.error('[Client] Unknown preset:', presetName);
            return;
        }

        console.log('[Client] Loading preset:', presetName, preset);

        // Update UI controls
        this.ui.keySelect.value = preset.key;
        this.ui.modeSelect.value = preset.mode;
        this.ui.bpmRange.value = preset.bpm;
        this.ui.intensityRange.value = preset.intensity;

        // Update display values
        const keyNames = ['C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B'];
        const modeNames = {
            'aeolian': 'Aeolian',
            'dorian': 'Dorian',
            'lydian': 'Lydian',
            'phrygian': 'Phrygian'
        };

        this.ui.keyValue.textContent = keyNames[preset.key % 12];
        this.ui.modeValue.textContent = modeNames[preset.mode];
        this.ui.bpmValue.textContent = preset.bpm;
        this.ui.intensityValue.textContent = preset.intensity.toFixed(1);

        // Update active button state
        document.querySelectorAll('.preset-button').forEach(btn => btn.classList.remove('active'));
        event.target.classList.add('active');

        // Update current parameters and save
        this.currentParameters = { ...preset };
        this.saveParametersToStorage();

        // Send to server
        this.sendControlMessage(preset);
    }

    /**
     * Load parameters from localStorage.
     */
    loadParametersFromStorage() {
        try {
            const stored = localStorage.getItem('auralis-parameters');
            if (stored) {
                const params = JSON.parse(stored);
                console.log('[Client] Loaded parameters from storage:', params);

                // Update UI controls
                this.ui.keySelect.value = params.key || 60;
                this.ui.modeSelect.value = params.mode || 'aeolian';
                this.ui.bpmRange.value = params.bpm || 70;
                this.ui.intensityRange.value = params.intensity || 0.5;

                // Update display values
                const keyNames = ['C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B'];
                const modeNames = {
                    'aeolian': 'Aeolian',
                    'dorian': 'Dorian',
                    'lydian': 'Lydian',
                    'phrygian': 'Phrygian'
                };

                this.ui.keyValue.textContent = keyNames[(params.key || 60) % 12];
                this.ui.modeValue.textContent = modeNames[params.mode || 'aeolian'];
                this.ui.bpmValue.textContent = params.bpm || 70;
                this.ui.intensityValue.textContent = (params.intensity || 0.5).toFixed(1);

                // Update current parameters
                this.currentParameters = { ...params };
            }
        } catch (error) {
            console.error('[Client] Failed to load parameters from storage:', error);
        }
    }

    /**
     * Save parameters to localStorage.
     */
    saveParametersToStorage() {
        try {
            localStorage.setItem('auralis-parameters', JSON.stringify(this.currentParameters));
            console.log('[Client] Saved parameters to storage:', this.currentParameters);
        } catch (error) {
            console.error('[Client] Failed to save parameters to storage:', error);
        }
    }
}

// Initialize client when page loads
window.addEventListener('DOMContentLoaded', () => {
    console.log('[Client] Initializing Auralis client...');
    window.auralisClient = new AuralisClient();
});
