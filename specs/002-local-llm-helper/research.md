# Research: Local LLM Helper for GraphMERT

**Date**: 2026-03-13  
**Feature**: Local LLM Helper for GraphMERT

---

## Decision 1: LLM Model Selection

**Decision**: TinyLlama 1.1B

**Rationale**:
- Smallest viable model (~700MB RAM footprint) suitable for 8GB laptop
- Fastest inference among quality options for CPU-only
- GGUF format support for llama.cpp
- Good enough quality for entity/relation extraction tasks (70% recall target)

**Alternatives Considered**:
| Model | Size | RAM | Rejected Because |
|-------|------|-----|------------------|
| Llama-2-7B | 7B | ~16GB | Exceeds laptop RAM |
| Phi-2 | 2.7B | ~3GB | Good but larger than TinyLlama |
| Qwen2-0.5B | 0.5B | ~600MB | Too small, quality insufficient |

---

## Decision 2: Inference Engine

**Decision**: LlamaCpp.jl (Julia wrapper around llama.cpp)

**Rationale**:
- Only production-ready Julia inference option
- Actively maintained (v0.5.0, Feb 2025)
- Supports GGUF format (required for TinyLlama)
- No external runtime dependencies (uses llama_cpp_jll.jl binary)
- Cross-platform: Linux, macOS, x86_64 and ARM

**Alternatives Considered**:
| Option | Status | Rejected Because |
|--------|--------|------------------|
| Pure Julia (Llama2.jl) | Experimental | Educational only, not production-ready |
| llama.cpp via HTTP | Possible | Adds latency, more complex |
| llama.cpp direct | Possible | Less idiomatic Julia |

---

## Decision 3: Architecture Pattern

**Decision**: Pluggable backend with shared interface

**Rationale**:
- Minimal changes to existing extraction pipeline
- Users can swap between local and cloud backends
- Preserves existing `HelperLLMClient` interface

**Implementation**:
- Create `LocalLLMClient` with same method signatures as `HelperLLMClient`
- Add `use_local::Bool` flag to `ProcessingOptions`
- Domain providers remain unchanged

---

## Technical Details

### Dependencies
- `LlamaCpp.jl` - Julia wrapper
- `llama_cpp_jll.jl` - Binary artifacts (auto-installed)
- Model file: TinyLlama-1.1B-Chat.Q4_0.gguf (or similar quantized)

### Memory Budget
- Model: ~700MB
- Inference buffer: ~500MB
- System overhead: ~500MB
- **Total**: ~1.7GB (within 8GB laptop constraint)

### API Compatibility
The local backend must implement:
- `discover_entities(client, text, domain)` → Vector{String}
- `match_relations(client, entities, text)` → Dict
- `form_tail_from_tokens(client, tokens, text)` → Vector{String}

---

## Conclusion

All key technical decisions resolved during clarification phase. No further research needed.

**Next Phase**: Proceed to data model design and contracts.
