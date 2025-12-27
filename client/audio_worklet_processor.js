/**
 * AudioWorklet Processor for Auralis Client
 *
 * Runs in the audio rendering thread for low-latency playback.
 * Reads from ring buffer and outputs to speakers.
 *
 * Features (T030):
 * - Jitter tracking with EMA smoothing
 * - Adaptive buffer tier management
 * - Real-time underrun detection
 */

class AuralisAudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();

        // Ring buffer for audio data (shared with main thread via SharedArrayBuffer)
        // For now, use message passing - SharedArrayBuffer requires COOP/COEP headers
        this.internalBuffer = new Float32Array(44100 * 2); // 2 seconds, stereo interleaved
        this.writeIndex = 0;
        this.readIndex = 0;

        // Playback rate adaptation
        this.targetLatencyMs = 400;
        this.playbackRate = 1.0;

        // Statistics
        this.underrunCount = 0;
        this.frameCount = 0;

        // Jitter tracking (T030)
        this.expectedNextChunkTime = null; // Expected time for next chunk delivery
        this.lastChunkTime = null; // Actual time of last chunk
        this.expectedChunkInterval = 100; // 100ms chunk duration
        this.jitterEMA = 0; // Exponential moving average of jitter (ms)
        this.jitterVarianceEMA = 0; // EMA of jitter variance
        this.jitterAlpha = 0.1; // EMA smoothing factor
        this.chunksReceived = 0;
        this.jitterHistory = []; // Last 50 jitter measurements
        this.maxJitterHistorySize = 50;

        // Listen for audio data from main thread
        this.port.onmessage = (event) => {
            this.handleMessage(event.data);
        };

        // Send ready signal
        this.port.postMessage({ type: 'ready' });
    }

    handleMessage(data) {
        if (data.type === 'audio') {
            // Track jitter (T030)
            const now = currentTime; // AudioWorklet currentTime in seconds
            const nowMs = now * 1000;

            if (this.expectedNextChunkTime !== null) {
                // Calculate jitter (deviation from expected delivery time)
                const jitterMs = Math.abs(nowMs - this.expectedNextChunkTime);

                // Update EMA
                if (this.jitterEMA === 0) {
                    // First measurement
                    this.jitterEMA = jitterMs;
                    this.jitterVarianceEMA = jitterMs * jitterMs;
                } else {
                    // EMA update: new = α * x + (1-α) * old
                    this.jitterEMA = this.jitterAlpha * jitterMs + (1 - this.jitterAlpha) * this.jitterEMA;
                    this.jitterVarianceEMA = this.jitterAlpha * (jitterMs * jitterMs) + (1 - this.jitterAlpha) * this.jitterVarianceEMA;
                }

                // Store in history (rolling window)
                this.jitterHistory.push(jitterMs);
                if (this.jitterHistory.length > this.maxJitterHistorySize) {
                    this.jitterHistory.shift();
                }
            }

            // Set expected time for next chunk
            this.expectedNextChunkTime = nowMs + this.expectedChunkInterval;
            this.lastChunkTime = nowMs;
            this.chunksReceived++;

            // Receive audio chunk from main thread
            // data.samples is Float32Array of stereo samples (LRLRLR...)
            const samples = data.samples;

            // Write to ring buffer (samples are already interleaved LRLR...)
            for (let i = 0; i < samples.length; i++) {
                const bufferIndex = (this.writeIndex + i) % this.internalBuffer.length;
                this.internalBuffer[bufferIndex] = samples[i];
            }

            this.writeIndex = (this.writeIndex + samples.length) % this.internalBuffer.length;
        } else if (data.type === 'control') {
            // Handle control messages (T029)
            if (data.targetLatencyMs !== undefined) {
                this.targetLatencyMs = data.targetLatencyMs;
                console.log(`[AudioWorklet] Target latency updated: ${this.targetLatencyMs}ms`);
            }
        }
    }

    process(inputs, outputs, parameters) {
        const output = outputs[0]; // First output
        const outputL = output[0]; // Left channel
        const outputR = output[1]; // Right channel

        if (!outputL || !outputR) {
            return true; // Keep processor alive
        }

        const framesToRender = outputL.length; // Typically 128 frames

        // Calculate buffer depth
        let bufferDepth = (this.writeIndex - this.readIndex + this.internalBuffer.length) % this.internalBuffer.length;
        const bufferDepthMs = (bufferDepth / 2) / 44.1; // Divide by 2 (stereo), then convert to ms

        // Adaptive playback rate
        if (bufferDepthMs > this.targetLatencyMs * 1.5) {
            // Buffer too full - speed up playback slightly
            this.playbackRate = 1.005;
        } else if (bufferDepthMs < this.targetLatencyMs * 0.5) {
            // Buffer running low - slow down playback
            this.playbackRate = 0.995;
        } else {
            // Normal playback
            this.playbackRate = 1.0;
        }

        // Render audio frames with adaptive rate
        let readIndexFloat = this.readIndex;

        for (let i = 0; i < framesToRender; i++) {
            // Calculate how many samples are available
            const available = (this.writeIndex - Math.floor(readIndexFloat) + this.internalBuffer.length) % this.internalBuffer.length;

            // Check for buffer underrun (need at least 2 samples for stereo)
            if (available < 2) {
                // Underrun - output silence
                outputL[i] = 0;
                outputR[i] = 0;
                this.underrunCount++;
                continue;
            }

            const readPos = Math.floor(readIndexFloat) % this.internalBuffer.length;

            // Read stereo sample (interleaved LRLR...)
            const sampleL = this.internalBuffer[readPos];
            const sampleR = this.internalBuffer[(readPos + 1) % this.internalBuffer.length];

            outputL[i] = sampleL;
            outputR[i] = sampleR;

            // Advance read position by 2 (stereo) * playback rate
            readIndexFloat += 2 * this.playbackRate;
        }

        // Update read index
        this.readIndex = Math.floor(readIndexFloat) % this.internalBuffer.length;

        // Increment frame counter
        this.frameCount += framesToRender;

        // Send status to main thread (throttled)
        if (this.frameCount % 4800 === 0) { // Every ~100ms at 48kHz
            // Calculate jitter std deviation
            const jitterStd = this.getJitterStd();

            this.port.postMessage({
                type: 'status',
                bufferDepthMs: bufferDepthMs,
                playbackRate: this.playbackRate,
                underrunCount: this.underrunCount,
                // Jitter metrics (T030)
                jitterMeanMs: this.jitterEMA,
                jitterStdMs: jitterStd,
                chunksReceived: this.chunksReceived,
                underrunRate: this.underrunCount / Math.max(this.chunksReceived, 1),
            });
        }

        return true; // Keep processor alive
    }

    getJitterStd() {
        /**
         * Calculate jitter standard deviation from EMA variance (T030).
         *
         * Returns:
         *     Standard deviation in milliseconds
         */
        if (this.jitterVarianceEMA <= this.jitterEMA * this.jitterEMA) {
            return 0;
        }
        const variance = this.jitterVarianceEMA - (this.jitterEMA * this.jitterEMA);
        return Math.sqrt(Math.max(0, variance));
    }
}

registerProcessor('auralis-audio-processor', AuralisAudioProcessor);
