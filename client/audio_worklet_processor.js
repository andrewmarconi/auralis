/**
 * AudioWorklet processor for Auralis streaming playback.
 *
 * Runs in the audio rendering thread (separate from main thread).
 * Handles base64 PCM decoding, ring buffer management, and audio output.
 *
 * Performance Critical: All operations must complete within 128 samples @ 44.1kHz (~2.9ms)
 */

class AuralisWorkletProcessor extends AudioWorkletProcessor {
    constructor() {
        super();

        // Ring buffer for audio chunks (stereo float32 samples)
        this.bufferCapacity = 50; // 50 chunks = 5 seconds @ 100ms/chunk
        this.buffer = new Array(this.bufferCapacity).fill(null);
        this.writeIndex = 0;
        this.readIndex = 0;
        this.bufferDepth = 0;

        // Pre-buffering state
        this.isBuffering = true;
        this.bufferTarget = 10; // Wait for 10 chunks (1 second) before starting

        // Chunk playback state
        this.currentChunk = null;
        this.chunkPosition = 0; // Sample position within current chunk

        // Metrics
        this.chunksReceived = 0;
        this.underruns = 0;
        this.lastUnderrunTime = 0;
        this.lastSeq = -1; // Track sequence numbers for gap detection

        // DEBUG: Track process() calls
        this.processCallCount = 0;
        this.totalFramesOutput = 0;

        // Listen for chunks from main thread (reduced logging for performance)
        this.port.onmessage = (event) => {
            this.handleMessage(event.data);
        };

        console.log('[AudioWorklet] AuralisWorkletProcessor initialized');
    }

    /**
     * Handle messages from main thread.
     *
     * @param {Object} message - Message from main thread
     * @param {string} message.type - Message type ('chunk', 'clear', 'getMetrics')
     * @param {Object} message.chunk - Audio chunk data (for 'chunk' type)
     */
    handleMessage(message) {
        switch (message.type) {
            case 'chunk':
                this.addChunk(message.chunk);
                break;

            case 'clear':
                this.clearBuffer();
                break;

            case 'getMetrics':
                this.sendMetrics();
                break;

            default:
                console.warn('[AudioWorklet] Unknown message type:', message.type);
        }
    }

    /**
     * Add audio chunk to ring buffer.
     *
     * @param {Object} chunk - Audio chunk
     * @param {string} chunk.data - Base64-encoded PCM data (int16)
     * @param {number} chunk.seq - Chunk sequence number
     * @param {number} chunk.timestamp - Server timestamp
     */
    addChunk(chunk) {
        // Chunk is already decoded in main thread (has left/right Float32Arrays)

        // Detect gaps in sequence numbers
        if (this.lastSeq >= 0 && chunk.seq !== this.lastSeq + 1) {
            const gap = chunk.seq - this.lastSeq - 1;
            console.warn(`[AudioWorklet] Gap detected! Missing ${gap} chunks (${this.lastSeq} → ${chunk.seq})`);
        }
        this.lastSeq = chunk.seq;

        // Write to ring buffer
        if (this.bufferDepth < this.bufferCapacity) {
            this.buffer[this.writeIndex] = chunk;
            this.writeIndex = (this.writeIndex + 1) % this.bufferCapacity;
            this.bufferDepth++;
            this.chunksReceived++;

            // Check if we've buffered enough to start playback
            if (this.isBuffering && this.bufferDepth >= this.bufferTarget) {
                this.isBuffering = false;
                console.log(`[AudioWorklet] Pre-buffering complete (${this.bufferDepth} chunks), starting playback`);
            }

            // Send buffer health update to main thread
            this.port.postMessage({
                type: 'bufferHealth',
                depth: this.bufferDepth,
                capacity: this.bufferCapacity,
                chunksReceived: this.chunksReceived,
                underruns: this.underruns,
                isBuffering: this.isBuffering
            });
        } else {
            console.warn('[AudioWorklet] Buffer overflow, dropping chunk', chunk.seq);
        }
    }

    /**
     * Decode base64 PCM chunk to stereo Float32Array.
     *
     * @param {Object} chunk - Audio chunk with base64 data
     * @returns {Object} Decoded chunk { left: Float32Array, right: Float32Array, seq, timestamp }
     */
    decodeChunk(chunk) {
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

        return {
            left: leftChannel,
            right: rightChannel,
            seq: chunk.seq,
            timestamp: chunk.timestamp
        };
    }

    /**
     * Clear ring buffer and reset state.
     */
    clearBuffer() {
        this.buffer.fill(null);
        this.writeIndex = 0;
        this.readIndex = 0;
        this.bufferDepth = 0;
        this.currentChunk = null;
        this.chunkPosition = 0;

        console.log('[AudioWorklet] Buffer cleared');
    }

    /**
     * Send metrics to main thread.
     */
    sendMetrics() {
        this.port.postMessage({
            type: 'metrics',
            chunksReceived: this.chunksReceived,
            underruns: this.underruns,
            bufferDepth: this.bufferDepth,
            bufferCapacity: this.bufferCapacity
        });
    }

    /**
     * Process audio (called by Web Audio API at ~128 samples per call).
     *
     * @param {Array<Array<Float32Array>>} inputs - Input audio (unused)
     * @param {Array<Array<Float32Array>>} outputs - Output audio buffers [channels][samples]
     * @param {Object} parameters - Audio parameters (unused)
     * @returns {boolean} True to keep processor alive
     */
    process(inputs, outputs, parameters) {
        const output = outputs[0]; // First output
        const leftOutput = output[0]; // Left channel
        const rightOutput = output[1]; // Right channel
        const framesNeeded = leftOutput.length; // Typically 128 samples

        // If still pre-buffering, output silence
        if (this.isBuffering) {
            leftOutput.fill(0.0);
            rightOutput.fill(0.0);
            return true;
        }

        let framesWritten = 0;

        while (framesWritten < framesNeeded) {
            // If no current chunk, try to read from buffer
            if (!this.currentChunk) {
                if (this.bufferDepth > 0) {
                    this.currentChunk = this.buffer[this.readIndex];
                    this.buffer[this.readIndex] = null;
                    this.readIndex = (this.readIndex + 1) % this.bufferCapacity;
                    this.bufferDepth--;
                    this.chunkPosition = 0;
                } else {
                    // Buffer underrun - output silence
                    const now = Date.now() / 1000.0; // Use timestamp instead of currentTime
                    if (now - this.lastUnderrunTime > 1.0) { // Throttle underrun logging
                        console.warn('[AudioWorklet] Buffer underrun');
                        this.underruns++;
                        this.lastUnderrunTime = now;
                    }

                    // Fill remaining frames with silence
                    for (let i = framesWritten; i < framesNeeded; i++) {
                        leftOutput[i] = 0.0;
                        rightOutput[i] = 0.0;
                    }
                    break;
                }
            }

            // Copy samples from current chunk to output
            if (this.currentChunk) {
                const chunk = this.currentChunk;
                const samplesAvailable = chunk.left.length - this.chunkPosition;
                const samplesToWrite = Math.min(samplesAvailable, framesNeeded - framesWritten);

                for (let i = 0; i < samplesToWrite; i++) {
                    leftOutput[framesWritten + i] = chunk.left[this.chunkPosition + i];
                    rightOutput[framesWritten + i] = chunk.right[this.chunkPosition + i];
                }

                this.chunkPosition += samplesToWrite;
                framesWritten += samplesToWrite;

                // If chunk fully consumed, clear it
                if (this.chunkPosition >= chunk.left.length) {
                    this.currentChunk = null;
                    this.chunkPosition = 0;
                }
            }
        }

        // DEBUG: Track process() calls and output
        this.processCallCount++;
        this.totalFramesOutput += framesWritten;

        // Log every 500 process calls (~1.5 seconds @ 128 samples)
        if (this.processCallCount % 500 === 0) {
            const silentFrames = framesNeeded - framesWritten;
            const percentSilent = (silentFrames / framesNeeded) * 100;
            console.log(
                `[AudioWorklet] Process stats: calls=${this.processCallCount}, ` +
                `totalFrames=${this.totalFramesOutput}, thisCall=${framesWritten}/${framesNeeded}, ` +
                `silent=${percentSilent.toFixed(1)}%, buffer=${this.bufferDepth}`
            );
        }

        // Keep processor alive
        return true;
    }
}

// Register processor
registerProcessor('auralis-worklet-processor', AuralisWorkletProcessor);
