# Research & Technical Decisions

**Feature**: Enhanced Generation & Controls  
**Date**: 2025-12-26  
**Status**: Completed - All clarifications resolved

## Decision: Second-order Markov chains for chord progressions

**Context**: Need enhanced Markov chain algorithms for more sophisticated chord progressions beyond Phase 1's basic implementation.

**Rationale**: Second-order chains consider the previous two chords for transitions, providing better musical coherence while increasing variety. Maintains real-time performance with O(1) lookups using pre-computed transition matrices. Aligns with music theory principles of harmonic progression.

**Alternatives Considered**:
- First-order Markov: Simpler but produces more repetitive patterns, reducing perceived variety
- Third-order or higher: Increases complexity exponentially, risking performance degradation
- Recurrent Neural Networks: Better long-term coherence but 10-100x slower, violates real-time constraints

**Implementation Notes**: Use numpy arrays for transition probabilities, GPU acceleration for matrix operations.

## Decision: Normalized Shannon entropy for chord progression variety

**Context**: Need quantitative measure for "harmonic variety" in success criteria SC-002.

**Rationale**: Shannon entropy provides a mathematically sound measure of information content in chord transitions. Normalized to 0-1 scale for easy comparison. Can be computed incrementally during generation without significant performance impact.

**Alternatives Considered**:
- Simple unique chord count: Easy to compute but doesn't account for transition probabilities or frequency
- Kullback-Leibler divergence: More sophisticated but computationally expensive for real-time use
- Musical complexity heuristics: Domain-specific but subjective and harder to validate

**Implementation Notes**: Calculate entropy = -Σ(p_i * log₂(p_i)) for transition probabilities, normalize by log₂(n) where n is number of possible transitions.

## Decision: Rule-based parameter validation with conflict matrices

**Context**: Need to validate parameter combinations and provide user feedback for conflicts.

**Rationale**: Deterministic rule-based approach using pre-defined conflict matrices is fast (O(1) lookup), predictable, and easy to maintain. Covers common musical conflicts like high complexity + low intensity.

**Alternatives Considered**:
- Machine learning classification: Overkill for known musical relationships, requires training data
- No validation: Risks poor user experience with invalid combinations
- Runtime simulation: Too slow for real-time validation

**Implementation Notes**: Use 2D numpy arrays for conflict scores, threshold-based warnings.

## Decision: Linear parameter scaling for overload prevention

**Context**: Need automatic parameter adjustment when combinations cause computational overload.

**Rationale**: Linear interpolation towards safe defaults maintains musical relationships while ensuring performance. Predictable behavior helps users understand system responses.

**Alternatives Considered**:
- Exponential scaling: Too aggressive, can cause abrupt changes
- Random parameter adjustment: Unpredictable, may create worse combinations
- Hard limits: Abrupt rejection without graceful degradation

**Implementation Notes**: Scale parameters by factor f = min(1.0, target_load / current_load), applied per parameter type.</content>
<parameter name="filePath">/Users/andrew/Develop/auralis/specs/001-enhanced-generation-controls/research.md