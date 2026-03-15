# Research: Local LLM Helper for GraphMERT

**Date**: 2026-03-13  
**Feature**: Local LLM Helper for GraphMERT
**Updated**: 2026-03-15 (llama-cpp / GGUF; Ollama removed)

---

## Decision 1: LLM Model Selection

**Decision**: lfm2.5-thinking:latest (recommended)

**Rationale**:
- Best entity extraction quality among tested models
- Excellent classification accuracy for people, places, organizations
- Good reasoning capabilities for relation extraction
- ~4GB model size (within 8GB laptop constraint)

**Alternatives Tested**:
| Model | Entities Found | Classification | Status |
|-------|---------------|----------------|--------|
| lfm2.5-thinking | 6 | Excellent | ✅ Recommended |
| ministral-3 | 4 | Good (misclassifies) | ⚠️ Alternative |
| qwen3:0.6b | 1 | Poor (concatenates) | ❌ Rejected |

---

## Decision 2: Inference Engine

**Decision**: LlamaCpp.jl (llama-cpp) with GGUF models

**Rationale**:
- Single in-process stack; no separate server (Ollama) required
- GGUF is the standard format for local models
- Better suited to embedding in a Julia pipeline; llama-cpp is the supported local path

**Ollama**: Removed from the project in favour of llama-cpp.

---

## Decision 3: Architecture Pattern

**Decision**: Pluggable backend with shared interface

**Rationale**:
- Minimal changes to existing extraction pipeline
- Users can swap between local and cloud backends
- Preserves existing `HelperLLMClient` interface

**Implementation**:
- `LocalLLMClient` with same method signatures as `HelperLLMClient`
- `use_local::Bool` and `local_config::LocalLLMConfig` in `ProcessingOptions`
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
