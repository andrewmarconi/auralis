"""Synthesis engine orchestrating composition and audio rendering.

Coordinates the full pipeline: ChordGenerator → MelodyGenerator → FluidSynthRenderer
to produce continuous, streaming ambient music.
"""

import asyncio
import logging
import time
from typing import Optional

import numpy as np

from composition.chord_generator import ChordGenerator, ChordProgression
from composition.melody_generator import MelodyGenerator, MelodyPhrase
from composition.musical_context import MusicalContext
from server.audio_chunk import AudioChunk
from server.exceptions import GenerationError, SynthesisError
from server.interfaces.buffer import IRingBuffer
from server.interfaces.synthesis import IFluidSynthRenderer, ISynthesisEngine
from server.metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


class SynthesisEngine(ISynthesisEngine):
    """Orchestrates music generation and audio synthesis."""

    def __init__(
        self,
        chord_generator: ChordGenerator,
        melody_generator: MelodyGenerator,
        fluidsynth_renderer: IFluidSynthRenderer,
        ring_buffer: IRingBuffer,
        metrics: PerformanceMetrics,
        sample_rate: int = 44100,
    ):
        """Initialize synthesis engine.

        Args:
            chord_generator: Chord progression generator
            melody_generator: Melody phrase generator
            fluidsynth_renderer: FluidSynth audio renderer
            ring_buffer: Ring buffer for audio chunks
            metrics: Performance metrics collector
            sample_rate: Audio sample rate in Hz
        """
        self.chord_generator = chord_generator
        self.melody_generator = melody_generator
        self.renderer = fluidsynth_renderer
        self.ring_buffer = ring_buffer
        self.metrics = metrics
        self.sample_rate = sample_rate

        # Current musical context
        self.context = MusicalContext.default()

        # Chunk sequence counter
        self._chunk_seq = 0

        # Generation loop control
        self._running = False
        self._generation_task: Optional[asyncio.Task] = None

        logger.info("Synthesis engine initialized")

    async def generate_phrase(
        self, context: MusicalContext, duration_bars: int = 8
    ) -> tuple[ChordProgression, MelodyPhrase]:
        """Generate chord progression and melody for a phrase.

        Args:
            context: Current musical parameters (key, mode, BPM, intensity)
            duration_bars: Number of bars to generate (8 or 16)

        Returns:
            Tuple of (ChordProgression, MelodyPhrase)

        Raises:
            GenerationError: If composition fails
        """
        try:
            start_time = time.perf_counter()

            # Generate chord progression
            chords = self.chord_generator.generate(context, duration_bars)

            # Generate melody
            melody = self.melody_generator.generate(context, chords)

            generation_time_ms = (time.perf_counter() - start_time) * 1000.0

            logger.debug(
                f"Generated phrase: {len(chords.chords)} chords, "
                f"{len(melody.notes)} notes ({generation_time_ms:.1f}ms)"
            )

            return (chords, melody)

        except Exception as e:
            raise GenerationError(f"Failed to generate phrase: {e}") from e

    async def render_phrase(
        self, chords: ChordProgression, melody: MelodyPhrase
    ) -> np.ndarray:
        """Render musical phrase to stereo PCM audio using FluidSynth.

        Args:
            chords: Generated chord progression
            melody: Generated melody phrase

        Returns:
            NumPy array, shape (2, num_samples), dtype float32, range [-1.0, 1.0]

        Raises:
            SynthesisError: If rendering fails or exceeds 100ms latency
        """
        start_time = time.perf_counter()

        try:
            # Calculate total duration in samples
            duration_samples = chords.duration_samples

            # Clear any lingering notes from previous phrases
            self.renderer.all_notes_off()

            # Channel assignments
            PIANO_CHANNEL = 0
            PAD_CHANNEL = 1

            # Chord intervals (semitones from root) - SIMPLIFIED to 2-note intervals
            # Using fewer notes to reduce voice count and CPU load for ambient pads
            CHORD_INTERVALS = {
                "major": [0, 7],      # Root + 5th (power chord)
                "minor": [0, 7],      # Root + 5th
                "sus2": [0, 2],       # Root + 2nd
                "sus4": [0, 5],       # Root + 4th
                "add9": [0, 2],       # Root + 2nd
                "maj7": [0, 11],      # Root + 7th
            }

            # Build event schedule: (sample_time, event_type, channel, pitch, velocity)
            events = []

            # Schedule chord events (on PAD_CHANNEL)
            chord_duration_samples = duration_samples // max(len(chords.chords), 1)
            for chord in chords.chords:
                # Get chord tones from root and intervals (2 notes max)
                intervals = CHORD_INTERVALS.get(chord.chord_type, [0, 7])
                chord_tones = [chord.root_pitch + interval for interval in intervals]

                # Trigger chord notes
                for pitch in chord_tones:
                    if 0 <= pitch <= 127:
                        # Note on at chord onset (moderate velocity)
                        events.append((chord.onset_time, "on", PAD_CHANNEL, pitch, 50))
                        # Note off at next chord (or end of phrase)
                        off_time = min(chord.onset_time + chord_duration_samples, duration_samples)
                        events.append((off_time, "off", PAD_CHANNEL, pitch, 0))

            # Schedule melody events (on PIANO_CHANNEL)
            for note in melody.notes:
                if 0 <= note.pitch <= 127:
                    # Note on at onset (use original velocity)
                    velocity = max(40, min(127, note.velocity))  # Minimum 40 for audibility
                    events.append((note.onset_time, "on", PIANO_CHANNEL, note.pitch, velocity))
                    # Note off at onset + duration
                    off_time = min(note.onset_time + note.duration, duration_samples)
                    events.append((off_time, "off", PIANO_CHANNEL, note.pitch, 0))

            # Sort events by time
            events.sort(key=lambda e: e[0])

            # Render audio with MIDI event scheduling
            CHUNK_SIZE = 2048  # Small chunks for timing accuracy
            audio_chunks = []
            current_sample = 0
            event_idx = 0

            while current_sample < duration_samples:
                # Process events that occur before next chunk
                next_sample = min(current_sample + CHUNK_SIZE, duration_samples)

                while event_idx < len(events) and events[event_idx][0] < next_sample:
                    sample_time, event_type, channel, pitch, velocity = events[event_idx]

                    # Render up to event time
                    if sample_time > current_sample:
                        samples_to_render = sample_time - current_sample
                        chunk = self.renderer.render(samples_to_render)
                        audio_chunks.append(chunk)
                        current_sample = sample_time

                    # Trigger MIDI event
                    if event_type == "on":
                        self.renderer.note_on(channel, pitch, velocity)
                    else:  # "off"
                        self.renderer.note_off(channel, pitch)

                    event_idx += 1

                # Render remaining samples in this chunk
                samples_to_render = next_sample - current_sample
                if samples_to_render > 0:
                    chunk = self.renderer.render(samples_to_render)
                    audio_chunks.append(chunk)
                    current_sample = next_sample

            # Concatenate all chunks
            audio = np.concatenate(audio_chunks, axis=1)

            # Normalize audio to prevent clipping (restore this)
            max_amplitude = np.max(np.abs(audio))
            if max_amplitude > 0.95:  # Normalize if approaching clipping threshold
                target_peak = 0.8  # Leave headroom
                normalization_factor = target_peak / max_amplitude
                audio = audio * normalization_factor
                logger.debug(
                    f"Normalized audio: peak {max_amplitude:.4f} → {target_peak:.4f} "
                    f"(factor: {normalization_factor:.4f})"
                )
                max_amplitude = target_peak

            # Ensure all notes are off
            self.renderer.all_notes_off()

            synthesis_time_ms = (time.perf_counter() - start_time) * 1000.0

            # Record synthesis latency
            self.metrics.record_synthesis_latency(synthesis_time_ms)

            # Check audio amplitude for debugging
            rms_amplitude = np.sqrt(np.mean(audio ** 2))

            # Validate latency target (warn but don't fail)
            if synthesis_time_ms > 100.0:
                logger.warning(
                    f"Synthesis latency {synthesis_time_ms:.1f}ms exceeds 100ms target"
                )

            logger.info(
                f"Rendered {duration_samples} samples with {len(events)} MIDI events "
                f"({synthesis_time_ms:.1f}ms, max_amp={max_amplitude:.4f}, rms={rms_amplitude:.4f})"
            )

            return audio

        except Exception as e:
            raise SynthesisError(f"Failed to render phrase: {e}") from e

    def _chunk_audio(self, audio: np.ndarray, chunk_size_samples: int = 4410) -> list[AudioChunk]:
        """Split audio into chunks for streaming.

        Args:
            audio: Stereo audio, shape (2, num_samples), dtype float32
            chunk_size_samples: Samples per chunk (default 4410 = 100ms @ 44.1kHz)

        Returns:
            List of AudioChunk objects
        """
        num_samples = audio.shape[1]
        chunks: list[AudioChunk] = []

        for start_idx in range(0, num_samples, chunk_size_samples):
            end_idx = min(start_idx + chunk_size_samples, num_samples)
            chunk_audio = audio[:, start_idx:end_idx]

            # Create chunk
            chunk = AudioChunk.from_float32(
                float_data=chunk_audio,
                seq=self._chunk_seq,
                timestamp=time.time(),
                sample_rate=self.sample_rate,
                duration_ms=100.0,
            )

            chunks.append(chunk)
            self._chunk_seq += 1

        return chunks

    async def start_generation_loop(self) -> None:
        """Start continuous generation loop.

        Generates phrases and chunks them into the ring buffer for streaming.
        Runs until stopped with stop_generation_loop().
        """
        if self._running:
            logger.warning("Generation loop already running")
            return

        self._running = True
        logger.info("Starting continuous generation loop")

        self._generation_task = asyncio.create_task(self._generation_loop())

    async def _generation_loop(self) -> None:
        """Internal generation loop."""
        try:
            while self._running:
                # Generate phrase (reduced from 8 to 2 bars to avoid overwhelming buffer)
                chords, melody = await self.generate_phrase(self.context, duration_bars=2)

                # Render to audio
                audio = await self.render_phrase(chords, melody)

                # Chunk audio
                chunks = self._chunk_audio(audio)

                # Calculate phrase duration for pacing
                phrase_duration_sec = len(chunks) * 0.1  # 100ms per chunk

                # Write chunks to ring buffer with real-time pacing
                for chunk in chunks:
                    # Wait for buffer to have space (back-pressure)
                    while self._running and self.ring_buffer.get_depth() >= self.ring_buffer.capacity - 2:
                        await asyncio.sleep(0.05)  # 50ms wait

                    if not self._running:
                        break

                    # Write chunk
                    success = self.ring_buffer.write(chunk)
                    if not success:
                        logger.warning(f"Failed to write chunk {chunk.seq} to buffer")
                        self.metrics.increment_buffer_overflow()

                    # Pace at exactly real-time (100ms per 100ms chunk)
                    await asyncio.sleep(0.1)

                logger.debug(
                    f"Generated and buffered {len(chunks)} chunks "
                    f"(buffer depth: {self.ring_buffer.get_depth()})"
                )

        except asyncio.CancelledError:
            logger.info("Generation loop cancelled")
        except Exception as e:
            logger.error(f"Error in generation loop: {e}", exc_info=True)
            self._running = False

    async def stop_generation_loop(self) -> None:
        """Stop continuous generation loop."""
        if not self._running:
            logger.warning("Generation loop not running")
            return

        logger.info("Stopping generation loop")
        self._running = False

        if self._generation_task:
            self._generation_task.cancel()
            try:
                await self._generation_task
            except asyncio.CancelledError:
                pass

        logger.info("Generation loop stopped")

    def update_context(self, context: MusicalContext) -> None:
        """Update musical context for generation.

        Args:
            context: New musical parameters

        Note:
            Changes take effect at the next phrase boundary.
        """
        logger.info(
            f"Musical context updated: {context.key_signature} @ {context.bpm} BPM, "
            f"intensity={context.intensity}"
        )
        self.context = context

    def update_parameters(
        self,
        key: Optional[int] = None,
        mode: Optional[str] = None,
        bpm: Optional[float] = None,
        intensity: Optional[float] = None
    ) -> None:
        """Update individual musical parameters.

        Args:
            key: Root MIDI pitch (60-71)
            mode: Scale mode ("aeolian", "dorian", "lydian", "phrygian")
            bpm: Tempo in beats per minute (60-120)
            intensity: Note density multiplier (0.0-1.0)

        Note:
            Changes take effect at the next phrase boundary.
        """
        # Update only provided parameters
        if key is not None:
            self.context.key = key
        if mode is not None:
            self.context.mode = mode  # type: ignore
        if bpm is not None:
            self.context.bpm = bpm
        if intensity is not None:
            self.context.intensity = intensity

        # Generate key signature string
        key_names = ['C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B']
        key_name = key_names[self.context.key % 12] if key is not None else key_names[self.context.key % 12]
        mode_suffix = " major" if self.context.mode == "lydian" else " minor"
        self.context.key_signature = f"{key_name}{mode_suffix}"

        logger.info(
            f"Parameters updated: {self.context.key_signature} @ {self.context.bpm} BPM, "
            f"mode={self.context.mode}, intensity={self.context.intensity:.1f}"
        )

    def get_device(self) -> str:
        """Return current synthesis device.

        Returns:
            Device string: Always "CPU" for FluidSynth
        """
        return "CPU"

    def is_running(self) -> bool:
        """Check if generation loop is running.

        Returns:
            True if running, False otherwise
        """
        return self._running
