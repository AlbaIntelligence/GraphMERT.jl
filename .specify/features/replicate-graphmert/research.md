# GraphMERT Research and Technical Decisions

## Overview

This document consolidates research findings, technical decisions, and alternatives considered for the GraphMERT implementation. All decisions are backed by the comprehensive technical specification (15 documents, ~6,500 lines) located in `original_paper/expanded_rewrite/`.

**Status**: ✅ Complete
**Last Updated**: 2025-01-20

---

## Core Architectural Decisions

### Decision 1: Leafy Chain Graph Structure

**Chosen**: Fixed-size regular graph structure (128 roots, 7 leaves per root)

**Rationale**:
- Enables efficient batch processing with fixed tensor sizes
- Simplifies graph encoding (no dynamic structures needed)
- Allows precomputation of shortest paths and attention masks
- Matches GPU memory constraints efficiently (1024 tokens total)
- Paper-specified design backed by experimental validation

**Alternatives Considered**:
1. **Dynamic graph sizes** - Rejected: Variable sizes complicate batching and increase memory overhead
2. **More leaves per root (14 or 21)** - Rejected: Increases memory without proportional quality gains (per paper ablation)
3. **Sparse graph encoding** - Rejected: Complicates implementation without efficiency gains for regular structure

**Implementation Reference**: Doc 02 (Leafy Chain Graphs, 480 lines)

---

### Decision 2: RoBERTa-Based Architecture with H-GAT

**Chosen**: RoBERTa encoder (80M parameters) + Hierarchical Graph Attention Network

**Rationale**:
- RoBERTa provides strong pre-trained biomedical language understanding
- 80M parameters enable laptop deployment while maintaining quality
- H-GAT enables relation-aware semantic fusion
- Paper demonstrates this architecture achieves 69.8% FActScore

**Alternatives Considered**:
1. **BERT-based** - Rejected: RoBERTa's improved pre-training yields better performance
2. **GPT-style autoregressive** - Rejected: Encoder-only better for extraction tasks
3. **Larger models (330M+)** - Rejected: Violates laptop deployment constraint
4. **Standard attention only** - Rejected: H-GAT's relation-awareness improves semantic quality

**Implementation Reference**:
- Doc 03 (RoBERTa Encoder, 350 lines) - ✅ **Complete** (444 lines code)
- Doc 04 (H-GAT Component, 400 lines) - ✅ **Complete** (437 lines code)

---

### Decision 3: Joint MLM+MNM Training

**Chosen**: Combined Masked Language Modeling + Masked Node Modeling with μ=1.0 weighting

**Rationale**:
- MLM trains syntactic understanding (text patterns)
- MNM trains semantic understanding (KG structure)
- Joint training enables vocabulary transfer between spaces
- Equal weighting (μ=1.0) balances both objectives
- Gradient flow through H-GAT updates relation embeddings

**Alternatives Considered**:
1. **MLM only** - Rejected: Cannot learn semantic space representations
2. **MNM only** - Rejected: Lacks syntactic grounding in text
3. **Sequential training (MLM then MNM)** - Rejected: Joint training yields better transfer
4. **Different weighting (μ ≠ 1.0)** - Rejected: Paper ablation shows μ=1.0 is optimal

**Mathematical Formulation**:
```
L(θ) = L_MLM(θ) + μ·L_MNM(θ)
     = -Σ_{t∈M_x} log p(x_t | G_{∖M_x∪M_g})
       - μ·Σ_{ℓ∈M_g} log p(g_ℓ | G_{∖M_x∪M_g})
```

**Implementation Reference**:
- Doc 06 (MLM Training, 450 lines) - ✅ **Complete** (436 lines code)
- Doc 07 (MNM Training, 420 lines) - 🔴 **Missing** (30 lines stub)

---

### Decision 4: Seed KG Injection Algorithm

**Chosen**: Multi-stage injection with diversity maximization (Paper Algorithm 1)

**Rationale**:
- Stage 1-2: Entity linking via SapBERT embeddings + string matching
- Stage 3: Contextual triple selection from UMLS
- Stage 4: Injection algorithm balancing score and relation diversity
- Prevents relation imbalance in training data
- Enables vocabulary transfer from seed KG to text domain

**Alternatives Considered**:
1. **Random injection** - Rejected: No semantic grounding
2. **Highest score only** - Rejected: Creates relation imbalance
3. **Fixed number per entity** - Rejected: Doesn't adapt to context
4. **No injection (text only)** - Rejected: Cannot learn semantic space

**Algorithm Components**:
1. **Entity Linking**: SapBERT → ANN retrieval → Jaccard filtering (threshold=0.5)
2. **Triple Selection**: UMLS query → Top-40 per entity → Contextual ranking
3. **Injection**: Score buckets → Relation buckets → Iterative selection
4. **Constraints**: One injection per head entity, diversity maximization

**Implementation Reference**: Doc 08 (Seed KG Injection, 500 lines) - 🔴 **Missing** (19 lines stub)

---

### Decision 5: Triple Extraction Pipeline

**Chosen**: 5-stage pipeline with helper LLM integration

**Rationale**:
- Stage 1 (Head Discovery): LLM extracts entities from text
- Stage 2 (Relation Matching): LLM assigns relations to entity pairs
- Stage 3 (Tail Prediction): GraphMERT predicts tail tokens (top-k=20)
- Stage 4 (Tail Formation): LLM combines tokens into coherent tails
- Stage 5 (Filtering): Similarity check (β threshold) + deduplication
- Balances neural prediction with LLM grammaticality checking

**Alternatives Considered**:
1. **End-to-end neural** - Rejected: Produces ungrammatical tails
2. **LLM-only** - Rejected: Doesn't leverage trained GraphMERT model
3. **Rule-based combination** - Rejected: Less flexible than LLM
4. **No filtering** - Rejected: Produces low-quality/hallucinated triples

**Helper LLM Choice**:
- Paper uses: **Qwen3-32B with "thinking mode"**
- Alternative: GPT-4, local models (Llama 3, etc.)
- Requirements: Entity discovery, relation matching, grammatical tail formation

**Implementation Reference**: Doc 09 (Triple Extraction, 470 lines) - 🔴 **Missing** (scattered)

---

## Data Structure Decisions

### Decision 6: Julia Type System Design

**Chosen**: Strong typing with abstract types and parametric polymorphism

**Rationale**:
- Leverages Julia's multiple dispatch for clean interfaces
- Type safety prevents runtime errors
- Enables compiler optimizations
- Self-documenting code structure
- Follows Julia best practices

**Key Types Defined**:

```julia
# Core Graph Types
struct ChainGraphNode{T}
    node_type::Symbol  # :root or :leaf
    token_id::Int
    position::Int
    parent_root::Union{Int, Nothing}
end

struct LeafyChainGraph
    root_nodes::Vector{ChainGraphNode{:root}}
    leaf_nodes::Vector{Vector{ChainGraphNode{:leaf}}}
    adjacency_matrix::Matrix{Bool}
    shortest_paths::Matrix{Int}
    config::ChainGraphConfig
end

# Training Types
struct MNMConfig
    vocab_size::Int
    hidden_size::Int
    num_leaves::Int
    mask_probability::Float64
    relation_dropout::Float64
end

# Extraction Types
struct BiomedicalEntity
    text::String
    cui::Union{String, Nothing}
    type::String
    position::TextPosition
    confidence::Float64
end

struct BiomedicalRelation
    relation_type::String
    confidence::Float64
    evidence::String
end

struct KnowledgeGraph
    entities::Vector{BiomedicalEntity}
    relations::Vector{BiomedicalRelation}
    triples::Vector{Tuple{Int, Int, Int}}
    metadata::Dict{String, Any}
end
```

**Implementation Reference**: Doc 11 (Data Structures, 450 lines)

---

### Decision 7: Graph Encoding Strategy

**Chosen**: Sequential encoding with attention decay masks

**Rationale**:
- Maps 2D graph to 1D sequence for transformer processing
- Encoding order: [r1, l1_1, l1_2, ..., l1_7, r2, l2_1, ..., r128, l128_7]
- Attention decay based on graph distance (not sequence distance)
- Learned decay mask using GELU activation
- Preserves graph structure while enabling efficient attention

**Encoding Algorithm**:

```
Sequence positions:
- Root r_i: position i-1 (0-indexed)
- Leaf l_{i,j}: position (i-1) + j

Attention decay:
f(d) = λ^GELU(√d - p)
where d = shortest_path(node_i, node_j)
      λ = 0.6 (decay parameter)
      p = learnable threshold parameter
```

**Alternatives Considered**:
1. **No decay (full attention)** - Rejected: Doesn't respect graph structure
2. **Hard masking** - Rejected: Prevents distant node interactions
3. **Linear decay** - Rejected: Exponential decay matches graph properties better

**Implementation Reference**: Doc 05 (Attention Mechanisms, 350 lines)

---

## Biomedical Domain Decisions

### Decision 8: UMLS Integration

**Chosen**: UMLS REST API with local caching

**Rationale**:
- UMLS provides comprehensive biomedical ontology (SNOMED CT, Gene Ontology)
- CUI codes enable entity linking and validation
- REST API accessible with authentication
- Local caching reduces API calls and costs
- Paper demonstrates UMLS integration improves quality

**API Design**:
- Endpoint: `https://uts-ws.nlm.nih.gov/rest`
- Authentication: API key from UTS account
- Rate limiting: 100 requests/minute
- Caching: Local SQLite database for CUI lookups
- Fallback: Local entity recognition when API unavailable

**Alternatives Considered**:
1. **Local UMLS download** - Rejected: Large dataset (hundreds of GB), complex setup
2. **No ontology** - Rejected: Reduces semantic quality
3. **Different ontology (ChEBI, HPO)** - Rejected: UMLS more comprehensive
4. **No caching** - Rejected: Excessive API calls, slow performance

**Implementation Reference**: Doc 08 (Seed KG Injection, section on entity linking)

---

### Decision 9: SapBERT for Entity Linking

**Chosen**: SapBERT embeddings with ANN indexing for candidate retrieval

**Rationale**:
- SapBERT trained on biomedical entity linking
- Dense embeddings enable semantic similarity search
- ANN (Approximate Nearest Neighbor) for efficient retrieval
- Two-phase: embedding similarity → string matching
- Paper demonstrates this approach achieves high linking accuracy

**Pipeline**:
1. **Candidate Retrieval**: SapBERT embeddings + ANN index → Top-10 candidates
2. **String Matching**: Character 3-grams + Jaccard similarity (threshold=0.5)
3. **Validation**: Context check + confidence scoring

**Alternatives Considered**:
1. **Exact string matching only** - Rejected: Misses synonyms and variations
2. **LLM-based linking** - Rejected: Too slow, expensive
3. **BioSyn** - Rejected: SapBERT performs better on UMLS
4. **No entity linking** - Rejected: Cannot access seed KG

**Implementation Reference**: Doc 08 (Seed KG Injection, Stage 1-2)

---

## Performance Decisions

### Decision 10: Model Size (80M Parameters)

**Chosen**: 80M parameter GraphMERT model

**Rationale**:
- Fits in laptop GPU memory (4GB VRAM)
- Enables local deployment for researchers
- Achieves strong performance (69.8% FActScore)
- Faster inference than larger models
- Paper demonstrates 80M sufficient for biomedical KG construction

**Configuration**:
- Hidden size: 512
- Number of layers: 12
- Attention heads: 8
- Vocabulary size: 30,522 (BioMedBERT tokenizer)
- Feed-forward size: 2048

**Alternatives Considered**:
1. **Smaller (20M)** - Rejected: Insufficient capacity for biomedical domain
2. **Larger (330M)** - Rejected: Violates laptop deployment requirement
3. **Different architecture** - Rejected: RoBERTa well-validated for this size

**Implementation Reference**: Doc 03 (RoBERTa Encoder, configuration section)

---

### Decision 11: Batch Processing and Memory Management

**Chosen**: Fixed batch size with gradient accumulation

**Rationale**:
- Batch size: 32 (fits in 16GB RAM)
- Gradient accumulation: 4 steps (effective batch size 128)
- Sequence length: 1024 tokens (fixed)
- Mixed precision training: FP16 for efficiency
- Memory profiling and optimization continuous

**Memory Budget**:
- Model parameters: ~320MB (80M × 4 bytes)
- Activations: ~2GB (batch × sequence × hidden)
- Gradients: ~320MB
- Optimizer states: ~640MB (Adam)
- Total: ~3.3GB per GPU

**Alternatives Considered**:
1. **Larger batches** - Rejected: Exceeds memory limits
2. **Smaller sequences** - Rejected: Reduces quality, violates paper design
3. **No accumulation** - Rejected: Small effective batch size hurts training
4. **Dynamic batching** - Rejected: Complicates fixed graph structure

**Implementation Reference**: Doc 06-07 (Training sections)

---

## Evaluation Decisions

### Decision 12: Evaluation Metrics

**Chosen**: FActScore*, ValidityScore, GraphRAG (as per paper)

**Rationale**:
- **FActScore***: Combines factuality + validity checking
- **ValidityScore**: Ontological alignment (UMLS validation)
- **GraphRAG**: Graph-level utility (QA accuracy)
- All three validated in paper, enable comprehensive KG quality assessment
- Targets: FActScore 69.8%, ValidityScore 68.8%

**FActScore* Formula**:
```
f(τ) = 𝟙[τ is supported by C(τ)]
FActScore*(G) = E_{τ∈G}[f(τ)]

where C(τ) = context sentences containing head/tail entities
```

**ValidityScore Formula**:
```
ValidityScore = count(yes) / total_triples
Using LLM judge: "Is this triple semantically valid?"
```

**GraphRAG Evaluation**:
```
Accuracy = correct_answers / total_questions
Using local search: entity retrieval → context → LLM QA
```

**Alternatives Considered**:
1. **Only FActScore** - Rejected: Doesn't capture ontological validity
2. **Only ValidityScore** - Rejected: Doesn't capture factual grounding
3. **Human evaluation only** - Rejected: Not scalable, expensive
4. **Automatic metrics (BLEU, etc.)** - Rejected: Doesn't capture semantic quality

**Implementation Reference**: Doc 10 (Evaluation Metrics, 400 lines)

---

## Technology Stack Decisions

### Decision 13: Julia ML Ecosystem

**Chosen**: Flux.jl, Transformers.jl, LightGraphs.jl

**Rationale**:
- **Flux.jl**: Flexible, idiomatic Julia ML framework
- **Transformers.jl**: Standard transformer implementations
- **LightGraphs.jl**: Efficient graph algorithms (Floyd-Warshall)
- **DataFrames.jl**: Data manipulation
- **HTTP.jl + JSON.jl**: API integration

**Package Versions**:
```toml
[deps]
Flux = "0.14"
Transformers = "0.3"
LightGraphs = "1.3"
MetaGraphs = "0.8"
DataFrames = "1.6"
HTTP = "1.10"
JSON3 = "1.14"
```

**Alternatives Considered**:
1. **Python (PyTorch)** - Rejected: Want idiomatic Julia implementation
2. **Knet.jl** - Rejected: Less maintained than Flux
3. **Different graph library** - Rejected: LightGraphs well-established
4. **Custom graph implementation** - Rejected: Reinventing the wheel

**Implementation Reference**: Project.toml, README.md

---

## Testing and Validation Decisions

### Decision 14: Testing Strategy

**Chosen**: Multi-level testing (unit, integration, end-to-end, scientific)

**Rationale**:
- **Unit tests**: Each component isolated, >90% coverage target
- **Integration tests**: Component interactions, small datasets
- **End-to-end**: Full pipeline on diabetes dataset
- **Scientific validation**: Paper result replication
- Constitution requirement: >80% coverage

**Test Structure**:
```
test/
├── unit/
│   ├── test_leafy_chain.jl
│   ├── test_mnm.jl
│   ├── test_seed_injection.jl
│   └── test_extraction.jl
├── integration/
│   ├── test_training_pipeline.jl
│   └── test_extraction_pipeline.jl
├── scientific/
│   ├── test_diabetes_replication.jl
│   └── test_evaluation_metrics.jl
└── runtests.jl
```

**Alternatives Considered**:
1. **Unit tests only** - Rejected: Doesn't validate integration
2. **End-to-end only** - Rejected: Slow, hard to debug
3. **No scientific validation** - Rejected: Cannot verify paper replication
4. **Lower coverage target** - Rejected: Violates constitution

**Implementation Reference**: test/ directory structure

---

### Decision 15: Scientific Reproducibility

**Chosen**: Deterministic training with documented seeds and environment

**Rationale**:
- Fixed random seeds for all RNG operations
- Pinned dependency versions (Manifest.toml)
- Containerized build (Nix flake)
- Complete hyperparameter documentation
- Paper comparison benchmarks

**Reproducibility Checklist**:
- [ ] Random seed: 42 (consistent across runs)
- [ ] Julia version: 1.10+ (documented)
- [ ] Package versions: Manifest.toml (pinned)
- [ ] Hardware specs: Documented (CPU, RAM, GPU)
- [ ] Dataset version: Diabetes dataset checksum
- [ ] Hyperparameters: Complete specification
- [ ] Evaluation protocol: Exact paper methodology

**Alternatives Considered**:
1. **Unpinned dependencies** - Rejected: Breaks reproducibility
2. **No seed control** - Rejected: Non-deterministic results
3. **No environment specification** - Rejected: Cannot reproduce
4. **Docker only** - Rejected: Nix provides better scientific computing support

**Implementation Reference**: Constitution, flake.nix, README.md

---

## Implementation Risks and Mitigations

### High-Risk Components

**1. Seed KG Injection Algorithm** (Risk: 9/10)
- **Risk**: Most complex component, external dependencies, novel algorithm
- **Mitigation**:
  - Study paper Appendix B in detail
  - Implement in stages (entity linking → triple selection → injection)
  - Create small test cases with known outputs
  - Mock external APIs initially
  - Validate each stage independently

**2. MNM Training Objective** (Risk: 8/10)
- **Risk**: Novel objective, gradient flow critical, integration complexity
- **Mitigation**:
  - Start with simple test cases
  - Validate gradients explicitly
  - Compare with MLM training patterns
  - Monitor relation embedding updates
  - Test on toy datasets first

**3. External Dependencies** (Risk: 7/10)
- **Risk**: SapBERT, UMLS, LLM APIs may be unavailable or rate-limited
- **Mitigation**:
  - Implement local caching aggressively
  - Provide fallback mechanisms
  - Mock interfaces for testing
  - Document API requirements clearly
  - Consider offline alternatives

---

## Summary of Key Decisions

| Decision        | Chosen Approach                     | Status                  | Priority |
| --------------- | ----------------------------------- | ----------------------- | -------- |
| Graph Structure | Fixed 128×7 leafy chain             | ✅ Spec complete         | P0       |
| Architecture    | RoBERTa (80M) + H-GAT               | ✅ Implemented           | P0       |
| Training        | Joint MLM+MNM (μ=1.0)               | 🟡 MLM done, MNM missing | P0       |
| Seed Injection  | Multi-stage with diversity          | 🔴 Not implemented       | P0       |
| Extraction      | 5-stage with helper LLM             | 🔴 Not implemented       | P0       |
| Types           | Strong Julia typing                 | 🟡 Partial               | P0       |
| UMLS            | REST API + caching                  | 🟡 Partial               | P1       |
| Entity Linking  | SapBERT + Jaccard                   | 🔴 Not implemented       | P1       |
| Evaluation      | FActScore*, ValidityScore, GraphRAG | 🟡 Partial               | P2       |
| Testing         | Multi-level strategy                | 🟡 Partial               | P2       |

---

## Research References

### Primary Sources
1. **GraphMERT Paper**: arXiv:2510.09580 (original paper)
2. **Comprehensive Spec**: 15 documents, ~6,500 lines (original_paper/expanded_rewrite/)
3. **Existing Implementation**: ~3,500 lines Julia code (GraphMERT/src/)

### Secondary Sources
1. **RoBERTa**: Liu et al. 2019 (pre-training improvements)
2. **SpanBERT**: Joshi et al. 2020 (span masking, boundary loss)
3. **GAT**: Veličković et al. 2018 (graph attention networks)
4. **SapBERT**: Liu et al. 2021 (biomedical entity linking)
5. **UMLS**: Bodenreider 2004 (unified medical language system)
6. **FActScore**: Min et al. 2023 (factuality evaluation)

### Implementation Guides
- **Leafy Chain Graphs**: Doc 02 (480 lines)
- **MNM Training**: Doc 07 (420 lines)
- **Seed KG Injection**: Doc 08 (500 lines)
- **Triple Extraction**: Doc 09 (470 lines)
- **Gap Analysis**: Doc 13 (670 lines)
- **Implementation Roadmap**: Doc 00-ROADMAP (770 lines)

---

## Next Steps

With research complete, proceed to:
1. **Data Model Design** (data-model.md)
2. **API Contracts** (contracts/)
3. **Implementation** (following roadmap)

**Status**: Research phase ✅ **COMPLETE** - Ready for implementation planning
