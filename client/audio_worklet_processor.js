/**
 * AudioWorklet Processor for Auralis Client
 *
 * Runs in the audio rendering thread for low-latency playback.
 * Reads from ring buffer and outputs to speakers.
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

        // Listen for audio data from main thread
        this.port.onmessage = (event) => {
            this.handleMessage(event.data);
        };

        // Send ready signal
        this.port.postMessage({ type: 'ready' });
    }

    handleMessage(data) {
        if (data.type === 'audio') {
            // Receive audio chunk from main thread
            // data.samples is Float32Array of stereo samples (LRLRLR...)
            const samples = data.samples;

            // Write to ring buffer
            for (let i = 0; i < samples.length; i++) {
                const bufferIndex = (this.writeIndex + i) % this.internalBuffer.length;
                this.internalBuffer[bufferIndex] = samples[i];
            }

            this.writeIndex = (this.writeIndex + samples.length) % this.internalBuffer.length;
        } else if (data.type === 'control') {
            // Handle control messages (future: volume, etc.)
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
            const readPos = Math.floor(readIndexFloat) % this.internalBuffer.length;

            // Check for buffer underrun
            if (readPos === this.writeIndex) {
                // Underrun - output silence
                outputL[i] = 0;
                outputR[i] = 0;
                this.underrunCount++;
            } else {
                // Read stereo sample (interleaved LRLR...)
                const sampleL = this.internalBuffer[readPos];
                const sampleR = this.internalBuffer[(readPos + 1) % this.internalBuffer.length];

                outputL[i] = sampleL;
                outputR[i] = sampleR;

                // Advance read position by 2 (stereo) * playback rate
                readIndexFloat += 2 * this.playbackRate;
            }
        }

        // Update read index
        this.readIndex = Math.floor(readIndexFloat) % this.internalBuffer.length;

        // Send status to main thread (throttled)
        if (currentFrame % 4800 === 0) { // Every ~100ms at 48kHz
            this.port.postMessage({
                type: 'status',
                bufferDepthMs: bufferDepthMs,
                playbackRate: this.playbackRate,
                underrunCount: this.underrunCount,
            });
        }

        return true; // Keep processor alive
    }
}

registerProcessor('auralis-audio-processor', AuralisAudioProcessor);
