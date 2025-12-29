# Composition Engine Specification

## Overview

The CompositionEngine orchestrates musical generation by coordinating chord progression, melody, and percussion generators. It maintains a phrase queue for continuous streaming without gaps.

---

## 1. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              CompositionEngine (Async Loop)                 │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │    Chord     │  │   Melody     │  │  Percussion     │  │
│  │  Generator   │  │  Generator   │  │   Generator     │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬────────┘  │
│         │                  │                    │           │
│         └──────────────────┼────────────────────┘           │
│                            ▼                                │
│                   ┌────────────────┐                        │
│                   │ Phrase Builder │                        │
│                   └────────┬───────┘                        │
│                            │                                │
│                            ▼                                │
│                   ┌─────────────────┐                       │
│                   │  Phrase Queue   │ (asyncio.Queue)       │
│                   │  Max size: 3    │                       │
│                   └─────────┬───────┘                       │
└────────────────────────────┼────────────────────────────────┘
                             │
                             ▼
                   Synthesis Worker consumes
```

---

## 2. Core Implementation

```python
# auralis/composition/engine.py
import asyncio
import numpy as np
from typing import Dict, List, Optional
from loguru import logger
from pydantic import BaseModel

from auralis.composition.chord_generator import ChordProgressionGenerator
from auralis.composition.melody_generator import ConstrainedMelodyGenerator
from auralis.composition.percussion_generator import PercussionGenerator
from auralis.music.theory import KEY_TO_ROOT_MIDI, bpm_to_bar_duration


class Phrase(BaseModel):
    """A complete musical phrase ready for synthesis."""

    phrase_id: str
    key: str
    bpm: int
    intensity: float
    duration_sec: float

    # Musical content
    chords: List[tuple]  # (onset_sample, root_midi, chord_type)
    melody: List[tuple]  # (onset_sample, pitch_midi, velocity, duration_sec)
    percussion: List[Dict]  # [{type, onset_sample, velocity, ...}]

    class Config:
        arbitrary_types_allowed = True


class GenerationParameters(BaseModel):
    """Current generation parameters."""

    key: str = "A minor"
    bpm: int = 70
    intensity: float = 0.5
    bars_per_phrase: int = 8


class CompositionEngine:
    """
    Manages musical generation and phrase queuing.

    Runs as an async background task, continuously generating
    phrases and queuing them for synthesis.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        phrase_queue_size: int = 3,
        initial_params: Optional[GenerationParameters] = None,
    ):
        self.sr = sample_rate
        self.phrase_queue: asyncio.Queue = asyncio.Queue(maxsize=phrase_queue_size)

        # Current parameters
        self.params = initial_params or GenerationParameters()
        self._params_lock = asyncio.Lock()

        # Generators
        self.chord_gen = ChordProgressionGenerator()
        self.melody_gen = ConstrainedMelodyGenerator()
        self.perc_gen = PercussionGenerator()

        # Statistics
        self.total_phrases_generated = 0
        self._running = False

        logger.info("CompositionEngine initialized")

    async def start(self):
        """Start the background generation loop."""
        if self._running:
            logger.warning("CompositionEngine already running")
            return

        self._running = True
        logger.info("Starting composition generation loop")
        asyncio.create_task(self._generation_loop())

    async def stop(self):
        """Stop the background generation loop."""
        self._running = False
        logger.info("Stopping composition generation loop")

    async def _generation_loop(self):
        """
        Main generation loop - runs continuously.

        Generates phrases and adds them to the queue.
        Blocks when queue is full (backpressure).
        """
        while self._running:
            try:
                # Generate next phrase
                phrase = await self._generate_phrase()

                # Add to queue (blocks if full)
                await self.phrase_queue.put(phrase)

                self.total_phrases_generated += 1
                logger.debug(
                    f"Generated phrase {phrase.phrase_id} "
                    f"(queue depth: {self.phrase_queue.qsize()})"
                )

            except asyncio.CancelledError:
                logger.info("Generation loop cancelled")
                break

            except Exception as e:
                logger.error(f"Error in generation loop: {e}", exc_info=True)
                # Sleep briefly to avoid tight error loop
                await asyncio.sleep(1.0)

    async def _generate_phrase(self) -> Phrase:
        """
        Generate a complete musical phrase.

        This is CPU-bound work, so we run it in a thread pool
        to avoid blocking the async event loop.
        """
        # Capture current params (thread-safe)
        async with self._params_lock:
            params = self.params.copy()

        # Run generation in thread pool (CPU-bound)
        phrase = await asyncio.to_thread(self._generate_phrase_sync, params)

        return phrase

    def _generate_phrase_sync(self, params: GenerationParameters) -> Phrase:
        """
        Synchronous phrase generation (runs in thread pool).

        This is the actual music generation logic.
        """
        import uuid

        phrase_id = f"phrase_{uuid.uuid4().hex[:8]}"

        # Get root MIDI note for key
        root_midi = KEY_TO_ROOT_MIDI[params.key]

        # Calculate phrase duration
        bar_duration_sec = bpm_to_bar_duration(params.bpm)
        phrase_duration_sec = bar_duration_sec * params.bars_per_phrase

        # 1. Generate chord progression
        chord_symbols = self.chord_gen.generate(length_bars=params.bars_per_phrase)

        # Convert to (onset_sample, root_midi, chord_type)
        chords = []
        for bar_idx, chord_symbol in enumerate(chord_symbols):
            onset_sample = int(bar_idx * bar_duration_sec * self.sr)
            chords.append((onset_sample, root_midi, chord_symbol))

        # 2. Generate melody
        melody_events = self.melody_gen.generate(
            chords=chord_symbols,
            root_midi=root_midi,
            bars=params.bars_per_phrase,
            bpm=params.bpm,
            intensity=params.intensity,
        )

        # Convert to (onset_sample, pitch_midi, velocity, duration_sec)
        melody = []
        for onset_sec, pitch_midi, velocity, duration_sec in melody_events:
            onset_sample = int(onset_sec * self.sr)
            melody.append((onset_sample, pitch_midi, velocity, duration_sec))

        # 3. Generate percussion
        percussion_events = self.perc_gen.generate(
            duration_bars=params.bars_per_phrase,
            intensity=params.intensity,
            bpm=params.bpm,
        )

        # Convert onset times to samples
        percussion = []
        for event in percussion_events:
            event_copy = event.copy()
            if "onset_sec" in event_copy:
                event_copy["onset_sample"] = int(event_copy["onset_sec"] * self.sr)
                del event_copy["onset_sec"]
            percussion.append(event_copy)

        # Create phrase object
        phrase = Phrase(
            phrase_id=phrase_id,
            key=params.key,
            bpm=params.bpm,
            intensity=params.intensity,
            duration_sec=phrase_duration_sec,
            chords=chords,
            melody=melody,
            percussion=percussion,
        )

        logger.debug(
            f"Generated {len(chords)} chords, {len(melody)} notes, "
            f"{len(percussion)} percussion events"
        )

        return phrase

    async def update_params(self, new_params: Dict):
        """
        Update generation parameters.

        Changes take effect on the next generated phrase.

        Args:
            new_params: Dict with keys: "key", "bpm", "intensity"
        """
        async with self._params_lock:
            if "key" in new_params:
                self.params.key = new_params["key"]
            if "bpm" in new_params:
                self.params.bpm = new_params["bpm"]
            if "intensity" in new_params:
                self.params.intensity = new_params["intensity"]

            logger.info(f"Updated params: {self.params}")

    async def get_next_phrase(self) -> Phrase:
        """
        Get the next phrase from the queue.

        Blocks until a phrase is available.

        Returns:
            Complete Phrase object ready for synthesis
        """
        phrase = await self.phrase_queue.get()
        return phrase

    def get_current_params(self) -> GenerationParameters:
        """Get current generation parameters (snapshot)."""
        return self.params.copy()
```

---

## 3. Generator Specifications

### 3.1 Chord Progression Generator

```python
# auralis/composition/chord_generator.py
import numpy as np
from typing import List
from auralis.music.theory import AMBIENT_MINOR_TRANSITIONS


class ChordProgressionGenerator:
    """Markov chain-based chord progression generator."""

    def __init__(self, seed: Optional[int] = None):
        """
        Args:
            seed: Random seed for reproducibility (optional)
        """
        self.transition_matrix = AMBIENT_MINOR_TRANSITIONS
        self.chord_symbols = ["i", "ii", "III", "iv", "v", "VI", "VII"]

        if seed is not None:
            np.random.seed(seed)

    def generate(self, length_bars: int = 8, start_chord: str = "i") -> List[str]:
        """
        Generate chord progression.

        Args:
            length_bars: Number of bars (one chord per bar)
            start_chord: Starting chord (default: "i" = tonic)

        Returns:
            List of chord symbols, e.g., ['i', 'VI', 'III', 'iv', ...]
        """
        progression = []

        # Find start index
        try:
            current_idx = self.chord_symbols.index(start_chord)
        except ValueError:
            current_idx = 0  # Default to i

        for _ in range(length_bars):
            chord = self.chord_symbols[current_idx]
            progression.append(chord)

            # Sample next chord from transition probabilities
            next_idx = np.random.choice(
                len(self.chord_symbols), p=self.transition_matrix[current_idx]
            )

            current_idx = next_idx

        return progression
```

### 3.2 Melody Generator

```python
# auralis/composition/melody_generator.py
import numpy as np
from typing import List, Tuple
from auralis.music.theory import SCALE_INTERVALS, CHORD_INTERVALS, get_scale_notes


class ConstrainedMelodyGenerator:
    """Generate melodies that fit chord progressions."""

    def __init__(self, scale_type: str = "aeolian"):
        self.scale_type = scale_type
        self.scale_intervals = SCALE_INTERVALS[scale_type]

    def generate(
        self,
        chords: List[str],
        root_midi: int,
        bars: int,
        bpm: int,
        intensity: float = 0.5,
    ) -> List[Tuple[float, int, float, float]]:
        """
        Generate melody constrained to chord/scale.

        Args:
            chords: Chord progression (one per bar)
            root_midi: Root MIDI note
            bars: Number of bars
            bpm: Tempo
            intensity: 0.0 (sparse) to 1.0 (dense)

        Returns:
            List of (onset_sec, pitch_midi, velocity, duration_sec)
        """
        from auralis.music.theory import bpm_to_bar_duration

        bar_duration_sec = bpm_to_bar_duration(bpm)
        melody = []

        # Get scale notes for passing tones
        scale_notes = get_scale_notes(root_midi, self.scale_type)

        # Notes per bar increases with intensity
        notes_per_bar_range = (1, int(4 * intensity) + 1)

        for bar_idx, chord_symbol in enumerate(chords):
            # Get chord tones for this bar
            chord_tones = self._get_chord_tones(root_midi, chord_symbol)

            # Number of notes in this bar
            num_notes = np.random.randint(*notes_per_bar_range)

            for note_offset in range(num_notes):
                # Onset time within bar
                note_pos = note_offset / num_notes if num_notes > 0 else 0
                onset_sec = (bar_idx + note_pos) * bar_duration_sec

                # Choose pitch
                if np.random.random() < 0.75:  # 75% chord tones
                    pitch_midi = np.random.choice(chord_tones)
                else:  # 25% scale tones
                    pitch_midi = np.random.choice(scale_notes)

                # Duration (longer for ambient)
                duration_sec = np.random.choice(
                    [0.5, 1.0, 1.5, 2.0, 3.0],
                    p=[0.1, 0.2, 0.3, 0.3, 0.1],
                )

                # Velocity (softer for ambient)
                velocity = np.random.uniform(0.5, 0.8)

                melody.append((onset_sec, pitch_midi, velocity, duration_sec))

        return melody

    def _get_chord_tones(self, root_midi: int, chord_symbol: str) -> List[int]:
        """Get MIDI notes for a chord across 2 octaves."""
        intervals = CHORD_INTERVALS.get(chord_symbol, [0, 3, 7])

        notes = []
        for octave in range(-1, 2):  # 3 octaves
            for interval in intervals:
                note = root_midi + octave * 12 + interval
                if 36 <= note <= 96:  # Keep in reasonable range
                    notes.append(note)

        return notes
```

### 3.3 Percussion Generator

```python
# auralis/composition/percussion_generator.py
import numpy as np
from typing import List, Dict


class PercussionGenerator:
    """Generate sparse percussion/texture events."""

    def generate(
        self, duration_bars: int, intensity: float, bpm: int
    ) -> List[Dict]:
        """
        Generate percussion events.

        Args:
            duration_bars: Length of phrase in bars
            intensity: 0.0 to 1.0 (density control)
            bpm: Tempo (for timing calculations)

        Returns:
            List of percussion event dicts
        """
        from auralis.music.theory import bpm_to_bar_duration

        bar_duration_sec = bpm_to_bar_duration(bpm)
        events = []

        # Sparse kick pattern (every 2-4 bars)
        if intensity > 0.2:
            for bar in range(0, duration_bars, np.random.randint(2, 5)):
                onset_sec = bar * bar_duration_sec
                events.append(
                    {
                        "type": "kick",
                        "onset_sec": onset_sec,
                        "velocity": 0.6 + intensity * 0.3,
                    }
                )

        # Granular swells (probabilistic)
        num_swells = int(intensity * 3)
        for _ in range(num_swells):
            bar = np.random.randint(0, duration_bars)
            onset_sec = bar * bar_duration_sec + np.random.uniform(0, bar_duration_sec)

            events.append(
                {
                    "type": "swell",
                    "onset_sec": onset_sec,
                    "duration_sec": np.random.uniform(2.0, 4.0),
                    "velocity": intensity * 0.7,
                }
            )

        # Sort by onset time
        events.sort(key=lambda e: e["onset_sec"])

        return events
```

---

## 4. Thread Safety & Async Design

### 4.1 Thread Safety Considerations

1. **Parameter Updates**:
   - Use `asyncio.Lock` for params mutation
   - Copy params before passing to thread pool

2. **Queue Access**:
   - `asyncio.Queue` is thread-safe
   - Put/get operations can be called from any thread

3. **Generator State**:
   - Generators are stateless (use numpy RNG per call)
   - No shared mutable state between calls

### 4.2 Backpressure Mechanism

```
Queue full (3 phrases buffered)
        ↓
phrase_queue.put() blocks
        ↓
Generation loop pauses
        ↓
Synthesis consumes phrase
        ↓
Queue has space
        ↓
Generation resumes
```

This prevents unbounded memory growth if synthesis is slow.

---

## 5. Integration with Server

```python
# server/main.py
from fastapi import FastAPI
from auralis.composition.engine import CompositionEngine, GenerationParameters

app = FastAPI()

# Global state
composition_engine: CompositionEngine = None


@app.on_event("startup")
async def startup():
    global composition_engine

    # Initialize with default params
    initial_params = GenerationParameters(
        key="A minor",
        bpm=70,
        intensity=0.5,
    )

    composition_engine = CompositionEngine(
        sample_rate=44100,
        phrase_queue_size=3,
        initial_params=initial_params,
    )

    # Start background generation
    await composition_engine.start()


@app.on_event("shutdown")
async def shutdown():
    if composition_engine:
        await composition_engine.stop()
```

---

## 6. Testing Strategy

```python
# tests/test_composition_engine.py
import pytest
import asyncio
from auralis.composition.engine import CompositionEngine, GenerationParameters


@pytest.mark.asyncio
async def test_phrase_generation():
    """Test that engine generates valid phrases."""
    engine = CompositionEngine(sample_rate=44100)
    await engine.start()

    # Get a phrase
    phrase = await asyncio.wait_for(engine.get_next_phrase(), timeout=5.0)

    assert phrase.phrase_id is not None
    assert len(phrase.chords) > 0
    assert len(phrase.melody) >= 0  # Melody can be empty
    assert phrase.duration_sec > 0

    await engine.stop()


@pytest.mark.asyncio
async def test_parameter_updates():
    """Test that parameter updates work correctly."""
    engine = CompositionEngine()
    await engine.start()

    # Update params
    await engine.update_params({"key": "C minor", "bpm": 80})

    # Generate phrase with new params
    phrase = await engine.get_next_phrase()

    assert phrase.key == "C minor"
    assert phrase.bpm == 80

    await engine.stop()
```

---

This composition engine design ensures continuous, gap-free music generation with clean separation of concerns and robust async handling.
