/**
 * Auralis Audio Client - AudioWorklet Version
 *
 * Replaces deprecated ScriptProcessorNode with modern AudioWorklet API.
 * Provides low-latency, glitch-free audio playback.
 *
 * Features (T029):
 * - Adaptive buffer tier management (minimal/normal/stable/defensive)
 * - Jitter-based tier adjustment
 * - Real-time buffer health monitoring
 */

class AuralisAudioClient {
    constructor(wsUrl = 'ws://localhost:8000/ws/stream', initialTier = 'normal') {
        this.wsUrl = wsUrl;

        // Adaptive buffer tiers (T029)
        this.bufferTiers = {
            minimal: { targetMs: 300, description: 'Stable network, lowest latency' },
            normal: { targetMs: 500, description: 'Default for new connections' },
            stable: { targetMs: 800, description: 'Occasional jitter compensation' },
            defensive: { targetMs: 1200, description: 'High jitter protection' },
        };
        this.currentTier = initialTier;
        this.targetLatencyMs = this.bufferTiers[initialTier].targetMs;

        // Audio context
        this.audioContext = null;
        this.workletNode = null;

        // WebSocket
        this.ws = null;
        this.isConnected = false;

        // Metrics
        this.chunksReceived = 0;
        this.chunkErrors = 0;
        this.bufferUnderruns = 0;
        this.currentBufferDepthMs = 0;
        this.currentPlaybackRate = 1.0;

        // Jitter metrics from worklet (T029)
        this.jitterMeanMs = 0;
        this.jitterStdMs = 0;
        this.underrunRate = 0;

        // Tier adjustment tracking
        this.tierAdjustmentInterval = 50; // Adjust every 50 chunks
        this.chunksReceivedSinceLastAdjustment = 0;
    }

    async connect() {
        try {
            // Create audio context (user gesture may be required)
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();

            console.log('[Client] Audio context created:', {
                sampleRate: this.audioContext.sampleRate,
                state: this.audioContext.state,
            });

            // Resume if suspended (Chrome autoplay policy)
            if (this.audioContext.state === 'suspended') {
                await this.audioContext.resume();
            }

            // Load AudioWorklet processor
            await this.audioContext.audioWorklet.addModule('audio_worklet_processor.js');

            // Create worklet node
            this.workletNode = new AudioWorkletNode(
                this.audioContext,
                'auralis-audio-processor',
                {
                    numberOfInputs: 0,
                    numberOfOutputs: 1,
                    outputChannelCount: [2], // Stereo
                }
            );

            // Listen to worklet messages
            this.workletNode.port.onmessage = (event) => {
                this.handleWorkletMessage(event.data);
            };

            // Connect worklet to destination (speakers)
            this.workletNode.connect(this.audioContext.destination);

            console.log('[Client] AudioWorklet connected');

            // Connect WebSocket
            await this.connectWebSocket();

        } catch (error) {
            console.error('[Client] Connection failed:', error);
            throw error;
        }
    }

    async connectWebSocket() {
        return new Promise((resolve, reject) => {
            this.ws = new WebSocket(this.wsUrl);

            this.ws.onopen = () => {
                console.log('[Client] WebSocket connected');
                this.isConnected = true;
                resolve();
            };

            this.ws.onmessage = (event) => {
                this.handleWebSocketMessage(event);
            };

            this.ws.onerror = (error) => {
                console.error('[Client] WebSocket error:', error);
                this.isConnected = false;
                reject(error);
            };

            this.ws.onclose = () => {
                console.log('[Client] WebSocket closed');
                this.isConnected = false;
            };
        });
    }

    handleWebSocketMessage(event) {
        try {
            const message = JSON.parse(event.data);

            if (message.type === 'audio') {
                this.processAudioChunk(message);
            } else if (message.type === 'status') {
                // Server status update
                console.log('[Client] Server status:', message);
            } else if (message.type === 'pong') {
                // Heartbeat response
                const latency = Date.now() - message.client_time;
                console.log(`[Client] Heartbeat latency: ${latency}ms`);
            }
        } catch (error) {
            console.error('[Client] Message parse error:', error);
            this.chunkErrors++;
        }
    }

    processAudioChunk(message) {
        try {
            // Decode base64 PCM
            const binaryString = atob(message.data);
            const bytes = new Uint8Array(binaryString.length);

            for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }

            // Convert to int16 array
            const int16Array = new Int16Array(bytes.buffer);

            // Convert to float32 [-1, 1]
            const float32Array = new Float32Array(int16Array.length);
            for (let i = 0; i < int16Array.length; i++) {
                float32Array[i] = int16Array[i] / 32767.0;
            }

            // Send to AudioWorklet
            this.workletNode.port.postMessage({
                type: 'audio',
                samples: float32Array,
            });

            this.chunksReceived++;

        } catch (error) {
            console.error('[Client] Audio processing error:', error);
            this.chunkErrors++;
        }
    }

    handleWorkletMessage(data) {
        if (data.type === 'status') {
            // Update metrics from worklet
            this.currentBufferDepthMs = data.bufferDepthMs;
            this.currentPlaybackRate = data.playbackRate;
            this.bufferUnderruns = data.underrunCount;

            // Update jitter metrics from worklet (T029)
            if (data.jitterMeanMs !== undefined) {
                this.jitterMeanMs = data.jitterMeanMs;
                this.jitterStdMs = data.jitterStdMs || 0;
                this.underrunRate = data.underrunRate || 0;

                // Check if we should adjust buffer tier
                this.chunksReceivedSinceLastAdjustment++;
                if (this.chunksReceivedSinceLastAdjustment >= this.tierAdjustmentInterval) {
                    this.adjustBufferTier();
                    this.chunksReceivedSinceLastAdjustment = 0;
                }
            }
        } else if (data.type === 'ready') {
            console.log('[Client] AudioWorklet ready');
        }
    }

    adjustBufferTier() {
        /**
         * Adjust buffer tier based on jitter and underrun metrics (T029).
         *
         * Escalation triggers:
         * - Jitter (mean + std) > 50ms
         * - OR underrun rate > 5%
         *
         * De-escalation triggers:
         * - Jitter (mean + std) < 20ms
         * - AND underrun rate < 1%
         */
        const tierOrder = ['minimal', 'normal', 'stable', 'defensive'];
        const currentIndex = tierOrder.indexOf(this.currentTier);

        const jitterTotal = this.jitterMeanMs + this.jitterStdMs;
        const shouldEscalate = jitterTotal > 50 || this.underrunRate > 0.05;
        const shouldDeEscalate = jitterTotal < 20 && this.underrunRate < 0.01;

        let newTier = this.currentTier;

        if (shouldEscalate && currentIndex < tierOrder.length - 1) {
            // Escalate to higher tier
            newTier = tierOrder[currentIndex + 1];
        } else if (shouldDeEscalate && currentIndex > 0) {
            // De-escalate to lower tier
            newTier = tierOrder[currentIndex - 1];
        }

        if (newTier !== this.currentTier) {
            const oldTier = this.currentTier;
            this.currentTier = newTier;
            this.targetLatencyMs = this.bufferTiers[newTier].targetMs;

            console.log(`[Client] Buffer tier adjusted: ${oldTier} → ${newTier} ` +
                        `(jitter=${jitterTotal.toFixed(1)}ms, underruns=${(this.underrunRate * 100).toFixed(1)}%)`);

            // Send updated target latency to worklet
            if (this.workletNode) {
                this.workletNode.port.postMessage({
                    type: 'control',
                    targetLatencyMs: this.targetLatencyMs,
                });
            }
        }
    }

    setControl(params) {
        /**
         * Send control message to server.
         *
         * @param {Object} params - Control parameters
         * @param {string} params.key - Musical key (e.g., "A minor")
         * @param {number} params.bpm - Beats per minute (40-200)
         * @param {number} params.intensity - Intensity (0.0-1.0)
         */
        if (!this.isConnected) {
            console.warn('[Client] Not connected, cannot send control');
            return;
        }

        this.ws.send(JSON.stringify({
            type: 'control',
            timestamp: new Date().toISOString(),
            ...params,
        }));

        console.log('[Client] Sent control update:', params);
    }

    sendHeartbeat() {
        /**
         * Send heartbeat to server.
         */
        if (!this.isConnected) return;

        this.ws.send(JSON.stringify({
            type: 'ping',
            client_time: Date.now(),
        }));
    }

    getStatus() {
        /**
         * Get current client status.
         *
         * @returns {Object} Status object with metrics
         */
        return {
            isConnected: this.isConnected,
            audioContextState: this.audioContext?.state,
            sampleRate: this.audioContext?.sampleRate,
            bufferDepthMs: this.currentBufferDepthMs,
            playbackRate: this.currentPlaybackRate,
            chunksReceived: this.chunksReceived,
            chunkErrors: this.chunkErrors,
            bufferUnderruns: this.bufferUnderruns,
            errorRate: this.chunkErrors / Math.max(this.chunksReceived, 1),
            // Buffer tier information (T029)
            bufferTier: this.currentTier,
            targetLatencyMs: this.targetLatencyMs,
            jitterMeanMs: this.jitterMeanMs,
            jitterStdMs: this.jitterStdMs,
            underrunRate: this.underrunRate,
        };
    }

    disconnect() {
        /**
         * Disconnect from server and cleanup resources.
         */
        console.log('[Client] Disconnecting...');

        // Close WebSocket
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }

        // Disconnect audio graph
        if (this.workletNode) {
            this.workletNode.disconnect();
            this.workletNode = null;
        }

        // Close audio context
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }

        this.isConnected = false;
        console.log('[Client] Disconnected');
    }
}

/**
 * AudioStreamClient - High-level wrapper for index.html integration
 */
export class AudioStreamClient {
    constructor(wsUrl) {
        this.client = new AuralisAudioClient(wsUrl);
        this.statusInterval = null;
        this.heartbeatInterval = null;

        // Callbacks
        this.onStatusUpdate = null;
        this.onError = null;
    }

    async connect() {
        try {
            await this.client.connect();
        } catch (error) {
            if (this.onError) {
                this.onError(error);
            }
            throw error;
        }
    }

    async start() {
        // Start status updates
        this.statusInterval = setInterval(() => {
            const status = this.client.getStatus();
            if (this.onStatusUpdate) {
                this.onStatusUpdate({
                    connected: status.isConnected,
                    bufferDepthMs: status.bufferDepthMs,
                    latencyMs: null, // TODO: Calculate from heartbeat
                    chunksReceived: status.chunksReceived,
                    errorRate: status.errorRate,
                });
            }
        }, 500);

        // Start heartbeat
        this.heartbeatInterval = setInterval(() => {
            this.client.sendHeartbeat();
        }, 30000);
    }

    stop() {
        if (this.statusInterval) {
            clearInterval(this.statusInterval);
            this.statusInterval = null;
        }
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
        this.client.disconnect();
    }

    updateParameters(params) {
        this.client.setControl(params);
    }
}
