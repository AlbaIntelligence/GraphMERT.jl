# Document 12: Implementation Mapping
## Spec-to-Code Cross-Reference

**Status**: 🟢 **Complete Analysis**
**Priority**: P2 (Helpful for developers)
**Purpose**: Map specifications to existing code

---

## Overview

This document provides a **complete mapping** between specification documents and existing Julia code, enabling developers to:

1. Find where each algorithm is implemented
2. Identify what's complete vs. placeholder
3. Locate code that needs work
4. Understand implementation quality

**Format**: Spec Section → Code Location → Status → Line Numbers

---

## Part 1: Core Architectures

### Document 03: RoBERTa Encoder

| Spec Section             | Code File                  | Status     | Lines   | Notes                             |
| ------------------------ | -------------------------- | ---------- | ------- | --------------------------------- |
| **RoBERTaConfig**        | `architectures/roberta.jl` | ✅ Complete | 16-60   | Fully implemented with validation |
| **RoBERTaEmbeddings**    | `architectures/roberta.jl` | ✅ Complete | 66-87   | Word + position + type embeddings |
| **RoBERTaSelfAttention** | `architectures/roberta.jl` | ✅ Complete | 89-115  | Multi-head self-attention         |
| **RoBERTaSelfOutput**    | `architectures/roberta.jl` | ✅ Complete | 117-134 | Residual + LayerNorm              |
| **RoBERTaAttention**     | `architectures/roberta.jl` | ✅ Complete | 136-151 | Complete attention block          |
| **RoBERTaIntermediate**  | `architectures/roberta.jl` | ✅ Complete | 153-168 | Feed-forward first layer          |
| **RoBERTaOutput**        | `architectures/roberta.jl` | ✅ Complete | 170-187 | Feed-forward second layer         |
| **RoBERTaLayer**         | `architectures/roberta.jl` | ✅ Complete | 189-206 | Full transformer layer            |
| **RoBERTaEncoder**       | `architectures/roberta.jl` | ✅ Complete | 208-225 | Stack of 12 layers                |
| **RoBERTaModel**         | `architectures/roberta.jl` | ✅ Complete | 227-245 | Complete model                    |
| **Forward Pass**         | `architectures/roberta.jl` | ✅ Complete | 247-372 | All forward functions             |
| **Utility Functions**    | `architectures/roberta.jl` | ✅ Complete | 374-444 | Helpers and I/O                   |

**Overall**: ✅ **100% Complete** (444/444 lines functional)

---

### Document 04: H-GAT Component

| Spec Section                | Code File               | Status     | Lines   | Notes                            |
| --------------------------- | ----------------------- | ---------- | ------- | -------------------------------- |
| **HGATConfig**              | `architectures/hgat.jl` | ✅ Complete | 16-54   | Configuration with validation    |
| **HGATAttention**           | `architectures/hgat.jl` | ✅ Complete | 60-87   | Graph-aware multi-head attention |
| **HGATFeedForward**         | `architectures/hgat.jl` | ✅ Complete | 89-110  | Standard FFN                     |
| **HGATLayer**               | `architectures/hgat.jl` | ✅ Complete | 112-136 | Single H-GAT layer               |
| **HGATModel**               | `architectures/hgat.jl` | ✅ Complete | 138-156 | Complete H-GAT stack             |
| **Forward Pass**            | `architectures/hgat.jl` | ✅ Complete | 158-266 | All forward functions            |
| **Graph Construction**      | `architectures/hgat.jl` | ✅ Complete | 268-325 | Adjacency matrix builders        |
| **Attention Visualization** | `architectures/hgat.jl` | ✅ Complete | 327-378 | Extract attention weights        |
| **Model Utilities**         | `architectures/hgat.jl` | ✅ Complete | 380-437 | Helpers and I/O                  |

**Overall**: ✅ **100% Complete** (437/437 lines functional)

---

## Part 2: Graph Structures

### Document 02: Leafy Chain Graphs

| Spec Section                 | Code File               | Status    | Lines | Notes             |
| ---------------------------- | ----------------------- | --------- | ----- | ----------------- |
| **ChainGraphNode**           | `graphs/leafy_chain.jl` | 🔴 Missing | N/A   | Needs 50 lines    |
| **ChainGraphConfig**         | `graphs/leafy_chain.jl` | 🔴 Missing | N/A   | Needs 30 lines    |
| **LeafyChainGraph**          | `graphs/leafy_chain.jl` | 🟡 Stub    | 19-29 | Basic struct only |
| **create_empty_chain_graph** | `graphs/leafy_chain.jl` | 🔴 Missing | N/A   | Needs ~50 lines   |
| **build_adjacency_matrix**   | `graphs/leafy_chain.jl` | 🔴 Missing | N/A   | Needs ~80 lines   |
| **floyd_warshall**           | `graphs/leafy_chain.jl` | 🔴 Missing | N/A   | Needs ~30 lines   |
| **inject_triple!**           | `graphs/leafy_chain.jl` | 🔴 Missing | N/A   | Needs ~60 lines   |
| **graph_to_sequence**        | `graphs/leafy_chain.jl` | 🔴 Missing | N/A   | Needs ~40 lines   |
| **create_attention_mask**    | `graphs/leafy_chain.jl` | 🔴 Missing | N/A   | Needs ~30 lines   |
| **All helper functions**     | `graphs/leafy_chain.jl` | 🔴 Missing | N/A   | Needs ~120 lines  |

**Overall**: 🔴 **7% Complete** (30/500 lines, stubs only)

**Priority**: P0 - BLOCKS EVERYTHING

---

### Document 05: Attention Mechanisms

| Spec Section                        | Code File               | Status     | Lines | Notes                            |
| ----------------------------------- | ----------------------- | ---------- | ----- | -------------------------------- |
| **Shortest Paths (Floyd-Warshall)** | N/A                     | 🔴 Missing  | N/A   | Can reuse from leafy_chain.jl    |
| **GELU activation**                 | `architectures/hgat.jl` | ✅ Complete | N/A   | Using Flux.gelu                  |
| **Decay mask computation**          | N/A                     | 🔴 Missing  | N/A   | Needs ~40 lines                  |
| **LearnableDecayMask**              | N/A                     | 🔴 Missing  | N/A   | Needs ~50 lines                  |
| **apply_attention_decay**           | N/A                     | 🔴 Missing  | N/A   | Needs ~30 lines                  |
| **Integration with RoBERTa**        | N/A                     | 🔴 Missing  | N/A   | Needs modification to roberta.jl |
| **Caching mechanism**               | N/A                     | 🔴 Missing  | N/A   | Needs ~50 lines                  |

**Overall**: 🟡 **10% Complete** (GELU only, ~20/200 lines)

**Priority**: P1 - Improves quality

---

## Part 3: Training Objectives

### Document 06: MLM Training

| Spec Section                 | Code File         | Status     | Lines   | Notes                  |
| ---------------------------- | ----------------- | ---------- | ------- | ---------------------- |
| **MLMConfig**                | `training/mlm.jl` | ✅ Complete | 17-50   | Configuration          |
| **MLMBatch**                 | `training/mlm.jl` | ✅ Complete | 52-73   | Batch structure        |
| **create_span_masks**        | `training/mlm.jl` | ✅ Complete | 79-128  | Span masking algorithm |
| **find_valid_positions**     | `training/mlm.jl` | ✅ Complete | 130-147 | Filter special tokens  |
| **apply_masks**              | `training/mlm.jl` | ✅ Complete | 149-178 | 80/10/10 strategy      |
| **calculate_mlm_loss**       | `training/mlm.jl` | ✅ Complete | 184-209 | Cross-entropy loss     |
| **calculate_boundary_loss**  | `training/mlm.jl` | ✅ Complete | 211-242 | SpanBERT boundary loss |
| **calculate_total_mlm_loss** | `training/mlm.jl` | ✅ Complete | 244-262 | Combined loss          |
| **create_mlm_batch**         | `training/mlm.jl` | ✅ Complete | 268-289 | Batch creation         |
| **train_mlm_step**           | `training/mlm.jl` | ✅ Complete | 291-311 | Training step          |
| **evaluate_mlm**             | `training/mlm.jl` | ✅ Complete | 313-348 | Evaluation             |
| **calculate_mlm_metrics**    | `training/mlm.jl` | ✅ Complete | 354-395 | Metrics                |
| **Utility Functions**        | `training/mlm.jl` | ✅ Complete | 397-436 | Helpers                |

**Overall**: ✅ **100% Complete** (436/436 lines functional)

---

### Document 07: MNM Training

| Spec Section                   | Code File         | Status    | Lines | Notes            |
| ------------------------------ | ----------------- | --------- | ----- | ---------------- |
| **MNMConfig**                  | `training/mnm.jl` | 🟡 Stub    | 13-28 | Config only      |
| **select_leaves_to_mask**      | `training/mnm.jl` | 🔴 Missing | N/A   | Needs ~50 lines  |
| **apply_mnm_masks**            | `training/mnm.jl` | 🔴 Missing | N/A   | Needs ~40 lines  |
| **calculate_mnm_loss**         | `training/mnm.jl` | 🔴 Missing | N/A   | Needs ~60 lines  |
| **train_joint_mlm_mnm_step**   | `training/mnm.jl` | 🔴 Missing | N/A   | Needs ~80 lines  |
| **Relation embedding dropout** | `training/mnm.jl` | 🔴 Missing | N/A   | Needs ~30 lines  |
| **Gradient flow validation**   | `training/mnm.jl` | 🔴 Missing | N/A   | Needs ~40 lines  |
| **All helper functions**       | `training/mnm.jl` | 🔴 Missing | N/A   | Needs ~100 lines |

**Overall**: 🔴 **8% Complete** (30/400 lines, config only)

**Priority**: P0 - BLOCKS TRAINING

---

## Part 4: Data Preparation

### Document 08: Seed KG Injection

| Spec Section                     | Code File                    | Status    | Lines | Notes            |
| -------------------------------- | ---------------------------- | --------- | ----- | ---------------- |
| **SapBERT integration**          | `training/seed_injection.jl` | 🔴 Missing | N/A   | Needs ~100 lines |
| **Entity linking (2-phase)**     | `training/seed_injection.jl` | 🔴 Missing | N/A   | Needs ~150 lines |
| **UMLS triple retrieval**        | `training/seed_injection.jl` | 🔴 Missing | N/A   | Needs ~100 lines |
| **Contextual selection**         | `training/seed_injection.jl` | 🔴 Missing | N/A   | Needs ~120 lines |
| **Injection algorithm**          | `training/seed_injection.jl` | 🔴 Missing | N/A   | Needs ~150 lines |
| **Score+diversity optimization** | `training/seed_injection.jl` | 🔴 Missing | N/A   | Needs ~100 lines |
| **Helper LLM integration**       | `training/seed_injection.jl` | 🔴 Missing | N/A   | Needs ~80 lines  |
| **All data structures**          | `training/seed_injection.jl` | 🔴 Missing | N/A   | Needs ~100 lines |

**Overall**: 🔴 **2% Complete** (19/800 lines, placeholder only)

**Priority**: P0 - BLOCKS TRAINING DATA

---

## Part 5: Extraction Pipeline

### Document 09: Triple Extraction

| Spec Section                | Code File      | Status    | Lines | Notes                       |
| --------------------------- | -------------- | --------- | ----- | --------------------------- |
| **Head discovery (LLM)**    | Multiple files | 🟡 Partial | N/A   | Some in `api/extraction.jl` |
| **Relation matching (LLM)** | Multiple files | 🟡 Partial | N/A   | Some in `api/extraction.jl` |
| **Tail prediction**         | Multiple files | 🟡 Partial | N/A   | Some in `api/extraction.jl` |
| **Tail formation (LLM)**    | Multiple files | 🔴 Missing | N/A   | Needs ~100 lines            |
| **Similarity filtering**    | Multiple files | 🔴 Missing | N/A   | Needs ~80 lines             |
| **Deduplication**           | Multiple files | 🔴 Missing | N/A   | Needs ~60 lines             |
| **Provenance tracking**     | Multiple files | 🟡 Partial | N/A   | Some in types.jl            |
| **End-to-end pipeline**     | Multiple files | 🔴 Missing | N/A   | Needs ~120 lines            |

**Overall**: 🟡 **25% Complete** (~150/600 lines, scattered)

**Priority**: P0 - BLOCKS OUTPUT

---

## Part 6: Evaluation

### Document 10: Evaluation Metrics

| Spec Section                   | Code File                 | Status    | Lines | Notes                |
| ------------------------------ | ------------------------- | --------- | ----- | -------------------- |
| **FActScore* calculation**     | `evaluation/factscore.jl` | 🟡 Stub    | N/A   | Basic structure only |
| **ValidityScore calculation**  | `evaluation/validity.jl`  | 🟡 Stub    | N/A   | Basic structure only |
| **GraphRAG evaluation**        | `evaluation/graphrag.jl`  | 🟡 Stub    | N/A   | Basic structure only |
| **Triple-to-claim conversion** | N/A                       | 🔴 Missing | N/A   | Needs ~50 lines      |
| **LLM verification**           | N/A                       | 🔴 Missing | N/A   | Needs ~100 lines     |
| **UMLS ontology integration**  | `biomedical/umls.jl`      | 🟡 Partial | N/A   | Some functions exist |
| **Triple retrieval for QA**    | N/A                       | 🔴 Missing | N/A   | Needs ~80 lines      |
| **Answer generation**          | N/A                       | 🔴 Missing | N/A   | Needs ~60 lines      |

**Overall**: 🟡 **15% Complete** (~60/400 lines)

**Priority**: P2 - Needed for validation

---

## Part 7: Core Types

### Document 11: Data Structures

| Spec Section                   | Code File         | Status     | Lines   | Notes                      |
| ------------------------------ | ----------------- | ---------- | ------- | -------------------------- |
| **BiomedicalEntity**           | `types.jl`        | ✅ Complete | 1-20    | Fully implemented          |
| **BiomedicalRelation**         | `types.jl`        | ✅ Complete | 21-40   | Fully implemented          |
| **KnowledgeGraph**             | `types.jl`        | ✅ Complete | 41-60   | Fully implemented          |
| **GraphMERTConfig**            | `types.jl`        | ✅ Complete | 61-120  | Fully implemented          |
| **ProcessingOptions**          | `types.jl`        | ✅ Complete | 121-150 | Fully implemented          |
| **GraphMERTModel**             | `types.jl`        | ✅ Complete | 151-180 | Fully implemented          |
| **FActScore**                  | `types.jl`        | ✅ Complete | 181-200 | Fully implemented          |
| **ValidityScore**              | `types.jl`        | ✅ Complete | 201-220 | Fully implemented          |
| **GraphRAG**                   | `types.jl`        | ✅ Complete | 221-240 | Fully implemented          |
| **ChainGraphNode**             | N/A               | 🔴 Missing  | N/A     | Needs addition to types.jl |
| **LeafyChainGraph (complete)** | N/A               | 🔴 Missing  | N/A     | Needs expansion            |
| **MNMConfig (complete)**       | `training/mnm.jl` | 🟡 Partial  | 13-28   | Basic only                 |
| **SeedInjectionConfig**        | N/A               | 🔴 Missing  | N/A     | Needs ~40 lines            |
| **ExtractionConfig**           | N/A               | 🔴 Missing  | N/A     | Needs ~40 lines            |
| **Training batch types**       | N/A               | 🔴 Missing  | N/A     | Needs ~60 lines            |

**Overall**: 🟡 **70% Complete** (240/340 lines)

**Priority**: P1 - Needed for all implementation

---

## Part 8: Supporting Infrastructure

### Helper LLM Integration

| Component            | Code File       | Status    | Lines | Notes            |
| -------------------- | --------------- | --------- | ----- | ---------------- |
| **LLM API client**   | `llm/helper.jl` | 🔴 Missing | N/A   | Needs ~100 lines |
| **Prompt templates** | `llm/helper.jl` | 🔴 Missing | N/A   | Needs ~80 lines  |
| **Response parsing** | `llm/helper.jl` | 🔴 Missing | N/A   | Needs ~50 lines  |
| **Caching layer**    | `llm/helper.jl` | 🔴 Missing | N/A   | Needs ~70 lines  |

**Overall**: 🔴 **0% Complete** (0/300 lines)

**Priority**: P1 - Needed for extraction

---

### UMLS Integration

| Component            | Code File            | Status    | Lines | Notes                |
| -------------------- | -------------------- | --------- | ----- | -------------------- |
| **REST API client**  | `biomedical/umls.jl` | 🟡 Partial | N/A   | Some functions exist |
| **CUI lookup**       | `biomedical/umls.jl` | 🟡 Partial | N/A   | Some functions exist |
| **Triple retrieval** | `biomedical/umls.jl` | 🔴 Missing | N/A   | Needs ~100 lines     |
| **Local caching**    | `biomedical/umls.jl` | 🔴 Missing | N/A   | Needs ~80 lines      |
| **Error handling**   | `biomedical/umls.jl` | 🔴 Missing | N/A   | Needs ~60 lines      |

**Overall**: 🟡 **30% Complete** (~120/400 lines)

**Priority**: P1 - Needed for injection

---

### Training Pipeline

| Component                     | Code File              | Status    | Lines | Notes            |
| ----------------------------- | ---------------------- | --------- | ----- | ---------------- |
| **Data loading**              | `training/pipeline.jl` | 🔴 Missing | N/A   | Needs ~80 lines  |
| **Batch creation**            | `training/pipeline.jl` | 🔴 Missing | N/A   | Needs ~60 lines  |
| **Training loop**             | `training/pipeline.jl` | 🔴 Missing | N/A   | Needs ~100 lines |
| **Checkpoint saving/loading** | `training/pipeline.jl` | 🔴 Missing | N/A   | Needs ~60 lines  |

**Overall**: 🔴 **0% Complete** (0/300 lines)

**Priority**: P1 - Needed for training

---

## Part 9: Overall Status Summary

### By Component Category

| Category              | Complete Lines | Total Lines | % Complete | Status |
| --------------------- | -------------- | ----------- | ---------- | ------ |
| **RoBERTa**           | 444            | 444         | 100%       | ✅      |
| **H-GAT**             | 437            | 437         | 100%       | ✅      |
| **MLM Training**      | 436            | 436         | 100%       | ✅      |
| **Core Types**        | 240            | 340         | 71%        | 🟡      |
| **Leafy Chain Graph** | 30             | 500         | 6%         | 🔴      |
| **MNM Training**      | 30             | 400         | 8%         | 🔴      |
| **Seed Injection**    | 19             | 800         | 2%         | 🔴      |
| **Triple Extraction** | 150            | 600         | 25%        | 🟡      |
| **Attention Decay**   | 20             | 200         | 10%        | 🔴      |
| **Training Pipeline** | 0              | 300         | 0%         | 🔴      |
| **Helper LLM**        | 0              | 300         | 0%         | 🔴      |
| **UMLS Integration**  | 120            | 400         | 30%        | 🟡      |
| **Evaluation**        | 60             | 400         | 15%        | 🟡      |
| **TOTAL**             | **1,986**      | **5,117**   | **39%**    | 🟡      |

### By Priority

**P0 - Critical (BLOCKS SYSTEM)**:
- Leafy Chain Graph: 6% ← **URGENT**
- MNM Training: 8% ← **URGENT**
- Seed KG Injection: 2% ← **URGENT**
- Triple Extraction: 25% ← **URGENT**

**P1 - High (IMPROVES SYSTEM)**:
- Attention Decay: 10%
- Training Pipeline: 0%
- Helper LLM: 0%
- UMLS Integration: 30%
- Core Types: 71%

**P2 - Medium (VALIDATION)**:
- Evaluation Metrics: 15%

---

## Part 10: Implementation Roadmap Cross-Reference

### Week 1-2: Foundation (from Roadmap)

**Tasks**:
1. ✅ Complete specifications → **DONE** (15/15 documents)
2. 🔴 Extend types → Map to `types.jl` lines 241-340
3. 🔴 Leafy Chain Graph → Map to `graphs/leafy_chain.jl` (500 lines needed)

### Week 3-4: Training Preparation

**Tasks**:
1. 🔴 Seed KG Injection → Map to `training/seed_injection.jl` (800 lines needed)

### Week 5-6: Training Implementation

**Tasks**:
1. 🔴 MNM Training → Map to `training/mnm.jl` (400 lines needed)
2. 🔴 Training Pipeline → Map to `training/pipeline.jl` (300 lines needed)

### Week 7-8: Extraction

**Tasks**:
1. 🔴 Helper LLM → Map to `llm/helper.jl` (300 lines needed)
2. 🔴 Triple Extraction → Complete `api/extraction.jl` + others (450 lines needed)

### Week 9-10: Enhancement

**Tasks**:
1. 🔴 Attention Decay → Add to `architectures/` (200 lines needed)
2. 🔴 Evaluation → Complete `evaluation/*.jl` (340 lines needed)

---

## Part 11: Code Quality Assessment

### High Quality (Well-Implemented)

✅ **RoBERTa Encoder** (`architectures/roberta.jl`):
- Clean architecture
- Comprehensive forward passes
- Good documentation
- Validation checks
- **Quality**: 9/10

✅ **H-GAT Component** (`architectures/hgat.jl`):
- Graph-aware attention
- Multi-head architecture
- Visualization tools
- **Quality**: 9/10

✅ **MLM Training** (`training/mlm.jl`):
- Span masking algorithm
- Boundary loss
- Complete metrics
- **Quality**: 8/10

### Medium Quality (Partial)

🟡 **Core Types** (`types.jl`):
- Good structure
- Missing some types
- **Quality**: 7/10

🟡 **UMLS Integration** (`biomedical/umls.jl`):
- Basic functions exist
- Needs completion
- **Quality**: 5/10

### Low Quality (Stubs/Placeholders)

🔴 **Leafy Chain Graph** (`graphs/leafy_chain.jl`):
- Only basic struct
- **Quality**: 1/10

🔴 **MNM Training** (`training/mnm.jl`):
- Only config
- **Quality**: 1/10

🔴 **Seed Injection** (`training/seed_injection.jl`):
- Single placeholder
- **Quality**: 0/10

---

## Part 12: Testing Coverage

### Well-Tested Components

✅ RoBERTa: Unit tests exist
✅ H-GAT: Unit tests exist
✅ MLM: Unit tests exist

### No Tests

🔴 Leafy Chain Graph
🔴 MNM Training
🔴 Seed Injection
🔴 Triple Extraction
🔴 Evaluation Metrics

**Test Coverage**: ~25% (only existing complete components)

---

## Summary

**Current State**: 39% complete (1,986 / 5,117 lines)

**Strengths**:
- ✅ Excellent RoBERTa implementation
- ✅ Excellent H-GAT implementation
- ✅ Excellent MLM implementation
- ✅ Good type foundations

**Critical Gaps** (P0):
- 🔴 Leafy Chain Graph (6% complete)
- 🔴 MNM Training (8% complete)
- 🔴 Seed KG Injection (2% complete)
- 🔴 Triple Extraction (25% complete)

**Estimated Work**: ~3,131 lines of new code needed

**Time to Working System**: 4-6 weeks (implementing P0 only)

**Recommendation**: Follow implementation roadmap, starting with Leafy Chain Graph

---

**Related Documents**:
- → [Doc 13: Gaps Analysis](13-gaps-analysis.md) - Detailed gap analysis
- → [Implementation Roadmap](00-IMPLEMENTATION-ROADMAP.md) - Week-by-week plan
- → [Status](STATUS.md) - Current progress
