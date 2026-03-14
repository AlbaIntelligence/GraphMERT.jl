# Feature Specification: Local LLM Helper for GraphMERT

**Feature Branch**: `002-local-llm-helper`  
**Created**: 2026-03-13  
**Status**: Draft  
**Input**: User description: "We are going to work on adding a small LLM that will run locally (laptop - CPU only) to be the helper LLM. We will use Julia as much as possible to do that."

## Clarifications

### Session 2026-03-13

- Q: Which open-source LLM model should be used for the local helper LLM? → A: TinyLlama (1.1B params, ~700MB RAM, fastest inference for CPU-only)
- Q: Which inference engine should be used? → A: LlamaCpp.jl (Julia wrapper around llama.cpp, production-ready, supports GGUF format)

## User Scenarios & Testing

### User Story 1 - Enable Offline Knowledge Graph Extraction (Priority: P1)

A researcher or data scientist wants to extract knowledge graphs from documents without requiring internet connectivity or external API services.

**Why this priority**: Offline capability is fundamental to the feature - without it, users cannot work in secure environments or without reliable internet.

**Independent Test**: Can be tested by disconnecting all network access and successfully extracting entities and relations from sample Wikipedia articles.

**Acceptance Scenarios**:

1. **Given** a laptop with no internet connection, **When** the user runs entity extraction on a Wikipedia article, **Then** the system extracts entities using the local LLM without errors.
2. **Given** a user with GraphMERT.jl installed, **When** they configure the system to use the local helper LLM, **Then** all extraction stages (head discovery, relation matching, tail formation) work without external API calls.

---

### User Story 2 - Reduce Extraction Costs (Priority: P2)

A user wants to avoid paying for external API calls (such as OpenAI) when extracting knowledge graphs from large document collections.

**Why this priority**: Cost reduction enables broader adoption for academic and personal projects with limited budgets.

**Independent Test**: Can be verified by processing 100 Wikipedia articles and confirming zero external API calls were made.

**Acceptance Scenarios**:

1. **Given** a batch of 100 documents to process, **When** extraction completes, **Then** no external API services were contacted and no usage costs were incurred.
2. **Given** a user monitoring their system, **When** extraction runs, **Then** all model inference occurs locally on the CPU.

---

### User Story 3 - Maintain Extraction Quality (Priority: P3)

A researcher needs the local LLM to produce entity and relation extraction quality comparable to external APIs for knowledge graph construction.

**Why this priority**: If quality degrades significantly, the feature loses practical value; the local LLM must be "good enough" for real use cases.

**Independent Test**: Can be tested by comparing extraction results from local vs. external LLM on the same set of documents and measuring entity/relation overlap.

**Acceptance Scenarios**:

1. **Given** the same Wikipedia article processed with local and external LLM, **When** results are compared, **Then** entity recall is at least 70% of the external LLM baseline.
2. **Given** a diverse set of 50 Wikipedia articles, **When** entity and relation extraction runs, **Then** the system identifies at least 80% of the entities found by a reference external API.

---

### Edge Cases

- What happens when the local model runs out of memory on very long documents?
- How does the system behave when the local LLM produces malformed output?
- What fallback mechanism exists if local inference fails mid-extraction?

## Requirements

### Functional Requirements

- **FR-001**: System MUST provide a local LLM interface that can be used as a drop-in replacement for external API-based helper LLMs.
- **FR-002**: System MUST perform all LLM inference (entity extraction, relation matching, tail formation) on CPU without GPU acceleration.
- **FR-003**: System MUST complete entity extraction on a 5,000-word Wikipedia article in under 5 minutes on a standard laptop.
- **FR-004**: Users MUST be able to configure which local model to use without code changes.
- **FR-005**: System MUST gracefully handle local LLM failures with informative error messages.
- **FR-006**: System MUST support model sizes suitable for consumer laptops (under 4GB RAM for model loading).
- **FR-007**: Extracted entities and relations MUST match the same output format as the existing domain provider interface.

### Key Entities

- **LocalLLMConfig**: Configuration for the local LLM including model path, context size, and inference parameters
- **ExtractionResult**: Structured output containing extracted entities and relations with confidence scores
- **ModelRegistry**: Catalog of supported local LLM models with their capabilities and requirements
- **TinyLlamaModel**: Specific implementation using TinyLlama 1.1B parameters (~700MB RAM footprint)
- **LlamaCppBackend**: Inference engine using LlamaCpp.jl wrapper around llama.cpp

## Success Criteria

### Measurable Outcomes

- **SC-001**: Users can extract knowledge graphs from documents without any network connectivity (fully offline operation verified)
- **SC-002**: Processing a standard Wikipedia article completes in under 5 minutes on laptop hardware
- **SC-003**: Entity extraction achieves at least 70% recall compared to external API baseline on the same documents
- **SC-004**: No external API services are contacted during extraction (zero network requests to LLM providers)
- **SC-005**: System runs on consumer laptop hardware with 8GB RAM and 4-core CPU

## Assumptions

- User has a standard laptop (8GB RAM, 4+ CPU cores, no GPU)
- The local LLM will use TinyLlama 1.1B model (~700MB RAM footprint)
- The local LLM will use LlamaCpp.jl (Julia wrapper around llama.cpp) for inference
- Model files will be downloaded once and stored locally in GGUF format
- The implementation will leverage the existing DomainProvider interface for extraction
