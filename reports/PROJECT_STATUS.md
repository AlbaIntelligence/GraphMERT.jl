# Project Status — March 2026

> **Required reading for all agents.** This document provides a comprehensive snapshot of the GraphMERT.jl project status. It supersedes older summaries and should be consulted before starting any new task.

---

## 1. Architecture Overview

**Model roles (do not confuse):** The **RoBERTa encoder** (paper methodology) is the compact transformer + H-GAT loaded via `load_model(path)` or **`load_model()`** (no args). The **default encoder path** is `~/.cache/llama-cpp/models/encoders/roberta-base` (override with `GRAPHMERT_ENCODER_ROOT`). Models in `~/.cache/llama-cpp/models` (e.g. GGUF files) are **helper LLMs** for entity/relation extraction only; they are **not** RoBERTa. See `GraphMERT/docs/src/getting_started/gguf_models.md` and `reports/REFERENCE_SOURCES_AND_ENCODER.md`. For **encoder alternatives**, see `GraphMERT/docs/src/getting_started/encoder_alternatives.md`.

GraphMERT.jl implements the GraphMERT algorithm for extracting knowledge graphs from unstructured text:

| Component | Status | Location |
|-----------|--------|----------|
| **RoBERTa encoder** | ✅ Implemented | `src/architectures/roberta.jl` |
| **H-GAT** | ✅ Implemented | `src/architectures/hgat.jl` |
| **Leafy chain graphs** | ✅ Implemented | `src/graphs/leafy_chain.jl` |
| **MLM training** | ✅ Implemented | `src/training/mlm.jl` |
| **MNM training** | ✅ Implemented | `src/training/mnm.jl` (Real gradient flow + single-pass optimization) |
| **Seed KG injection** | ✅ Implemented | `src/training/seed_injection.jl` (Ontology-driven injection) |
| **Entity extraction** | ✅ Wired | Wired to LLM with regex fallback; sub-span matching fixed |
| **Relation extraction** | ⚠️ Partial | LLM-based extraction wired; Wikipedia domain needs testing |
| **Model persistence** | ✅ Implemented | JLD2-based weights + config + optimizer state |
| **Evaluation (FActScore, Validity, GraphRAG)** | ✅ Implemented | FActScore (linear), ValidityScore, GraphRAG pipeline |
| **Reliability pipeline** | ✅ Implemented | Provenance, validate_kg, clean_kg, evaluate_factscore(kg, reference) |

---

## 2. What's Implemented

### 2.1 Core Models
- **RoBERTa encoder**: Full implementation with embeddings, attention, feed-forward
- **H-GAT**: Heterogeneous Graph Attention Network with multi-head attention
- **Leafy chain graph**: Document representation with root tokens (text) and leaf tokens (semantic triples)

### 2.2 Training
- **MLM (Masked Language Model)**: Span masking, boundary loss, metrics
- **MNM (Masked Node Modeling)**: Mask selection, application, loss computation
- **Joint training**: MLM + MNM combined loss step
- **Seed injection**: Framework for injecting seed knowledge into training

### 2.3 Domain System
- **DomainProvider interface**: `src/domains/interface.jl`
- **Domain registry**: `src/domains/registry.jl`
- **Biomedical domain**: Full implementation with UMLS integration stubs
- **Wikipedia domain**: Implementation with Wikidata integration stubs

### 2.4 Extraction Pipeline
- **5-stage extraction**: Head discovery → relation matching → tail prediction → tail formation → filtering
- **API entry point**: `extract_knowledge_graph(text, model; options)`

---

## 3. Known Issues & Stubbed Code

### 3.1 MNM Training (Implemented)
- **Location**: `src/training/mnm.jl`
- **Status**: `train_joint_mlm_mnm_step` computes real gradients using Zygote. Single-forward-pass optimization implemented.
- **Remaining Work**: Large-scale distributed training (future optimization).

### 3.2 LLM-Based Entity/Relation Extraction (Wired)
- **Location**: `src/llm/helper.jl` has `discover_entities()` function
- **Status**: Biomedical domain wired to LLM. Sub-span matching fixed.
- **Impact**: Supports paper's hybrid extraction methodology.
- **Remaining**: Wikipedia domain testing.

### 3.3 Model Persistence (Implemented)
- **Location**: `models/persistence.jl`
- **Status**: Full JLD2 persistence for weights, config, and optimizer state. Round-trip verified.
- **Impact**: Training resumption and model serving fully supported.

### 3.4 Training Pipeline (Verified)
- **Location**: `src/training/pipeline.jl`
- **Status**: `test_real_training_loop.jl` passes. Metrics logging and validation loop implemented.
- **Impact**: Core training loop is reliable.

### 3.5 Seed Injection (Implemented)
- **Location**: `src/training/seed_injection.jl`
- **Status**: `OntologySource` abstraction supports multi-domain injection (UMLS, Wikidata).
- **Impact**: Enables domain-adaptive pre-training.

---

## 4. High-Priority Tasks

| Priority | Task | Status |
|----------|------|--------|
| **P0** | Wire LLM-based entity extraction (biomedical) | ✅ Done |
| **P1** | Complete training pipeline with proper MNM batching | ✅ Done |
| **P1** | Fix `max_position_embeddings` (512->1024) | ✅ Done |
| **P2** | Extend seed injection for large corpora | ✅ Done |
| **P2** | Complete evaluation module wiring | ✅ Done |
| **P0** | Fix MNM forward pass | ✅ Done |
| **P1** | Wire model persistence | ✅ Done |

---

## 5. Testing Status

| Test Suite | Status |
|------------|--------|
| Unit tests (`test/unit/`) | ✅ All pass |
| Integration tests (`test/integration/`) | ✅ All pass |
| Domain tests | Biomedical ✅; Wikipedia 🟡 (needs config) |
| Wikipedia KG testing | Test infrastructure created; needs LLM config |

**Note**: Tests currently use regex fallback for extraction. Must configure `use_helper_llm=true` and provide OpenAI API key.

---

## 6. Documentation

| Document | Status |
|----------|--------|
| `AGENTS.md` | This file |
| `reports/PROJECT_STATUS.md` | **This document** |
| `reports/CODE_REVIEW.md` | Detailed code review |
| `reports/REFERENCE_SOURCES_AND_ENCODER.md` | Contextual info, notebook, Python clone; RoBERTa vs other encoders |
| `original_paper/expanded_rewrite/` | Spec documents (canonical) |

**Reference sources:** `Contextual_information.md` (root), `original_paper/graphMERT.ipynb`, and `original_paper/GraphMert/` (cloned Python repo) are summarized in `reports/REFERENCE_SOURCES_AND_ENCODER.md`. The spec requires a **RoBERTa** encoder; extraction should use it once model persistence is wired.

---

## 7. Getting Started

For new tasks:

1. Read this status document first
2. Check `original_paper/expanded_rewrite/` for spec details
3. Look at `reports/CODE_REVIEW.md` for code-specific issues
4. Check `reports/IMPLEMENTATION_FILES.md` for file locations

---

## 8. Reliability pipeline (encoder-in-path and status)

- **Provenance:** Extraction populates `ProvenanceRecord` per triple when `enable_provenance_tracking=true`; `get_provenance(kg, relation_or_index)` returns source document/segment.
- **Validation:** `validate_kg(kg, domain)` returns `ValidityReport` (score, valid_count, total_triples); graceful degradation when domain/ontology missing.
- **Factuality:** `evaluate_factscore(kg, reference::KnowledgeGraph)` returns `FactualityScore` when reference is provided.
- **Cleaning:** `clean_kg(kg; policy)` returns new KnowledgeGraph; policy: min_confidence, require_provenance, contradiction_handling.
- **Encoder-in-path:** Extraction uses the model encoder (RoBERTa + H-GAT) when `model isa GraphMERTModel`; `load_model(path)` returns full model (config-based; full weight load from checkpoint is post-MVP).
- **Iterative seed:** Path documented: clean KG → export → use as seed path for training/extraction; see quickstart and `specs/003-align-contextual-description/quickstart.md`.

---

## 9. Recent Changes (March 2026)

- Created comprehensive project review: `reports/PROJECT_REVIEW_2026-03-10.md`
- Created Wikipedia KG testing infrastructure: `GraphMERT/test/wikipedia/`
- Identified regex fallback issue in entity extraction
- Identified MNM forward pass stub
- Implemented reliability pipeline (provenance, validate_kg, clean_kg, evaluate_factscore with reference); updated REFERENCE_SOURCES_AND_ENCODER.md with narrative mapping

---

*Last updated: 2026-03-15*
