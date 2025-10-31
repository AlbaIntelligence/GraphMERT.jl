# GraphMERT.jl Implementation Specification

## Source of Truth: Original Paper Analysis

This specification is derived from the complete GraphMERT paper analysis in `/original_paper/expanded_rewrite/`, which provides detailed algorithms, data structures, and implementation guidance for all components.

## System Overview

GraphMERT is a biomedical knowledge graph construction system that combines:
- **RoBERTa Encoder** (80M parameters): Contextual text embeddings
- **H-GAT Component**: Hierarchical Graph Attention for relations
- **Dual Training Objectives**: MLM (syntactic) + MNM (semantic)
- **Seed KG Injection**: Vocabulary transfer from knowledge graphs
- **5-Stage Triple Extraction**: End-to-end KG generation

## Core Components Specification

### 1. Leafy Chain Graph Structure
**Source**: `02-leafy-chain-graphs.md` (921 lines)

#### Data Structure
```julia
struct ChainGraphNode
    id::Int
    node_type::Symbol  # :root or :leaf
    text::String       # token text or KG entity
    position::Int      # sequential position
    attributes::Dict{Symbol,Any}
end

struct LeafyChainGraph
    config::ChainGraphConfig
    nodes::Vector{ChainGraphNode}
    adjacency_matrix::SparseMatrixCSC{Float32,Int}
    injected_mask::Matrix{Bool}  # tracks injected triples
    distance_matrix::Matrix{Float32}  # Floyd-Warshall results
end

struct ChainGraphConfig
    num_roots::Int           # 128 (fixed for efficiency)
    num_leaves_per_root::Int # 7 (matches paper)
    max_sequence_length::Int # 512
    injection_ratio::Float64 # 0.2
end
```

#### Key Algorithms
- **Graph Construction**: Fixed structure with roots (text) and leaves (KG slots)
- **Adjacency Matrix**: Sparse representation for efficiency
- **Floyd-Warshall**: Shortest path computation for attention decay
- **Triple Injection**: Map KG triples to leaf positions
- **Sequential Encoding**: Convert graph to token sequence for RoBERTa

#### Performance Requirements
- Graph construction: < 10ms
- Floyd-Warshall: O(n³) but n=1024, acceptable
- Memory: < 50MB per graph

### 2. RoBERTa Encoder Architecture
**Source**: `03-roberta-encoder.md` (620 lines)

#### Modified Architecture (GraphMERT-specific)
```julia
# Standard RoBERTa-base: 125M parameters
# GraphMERT RoBERTa: 80M parameters (36% reduction)

struct GraphMERTRoBERTaConfig
    vocab_size::Int = 30522          # BioMedBERT vocabulary
    hidden_size::Int = 512            # Reduced from 768
    num_attention_heads::Int = 8      # Reduced from 12
    num_hidden_layers::Int = 12       # Same
    intermediate_size::Int = 2048     # Proportional reduction
    max_position_embeddings::Int = 512
    layer_norm_eps::Float64 = 1e-12
    hidden_dropout_prob::Float64 = 0.1
    attention_probs_dropout_prob::Float64 = 0.1
end
```

#### Key Modifications for GraphMERT
1. **Smaller Hidden Size**: 768 → 512 (efficiency)
2. **Reduced Attention Heads**: 12 → 8 (matches hidden size)
3. **Proportional FFN**: 3072 → 2048
4. **BioMedBERT Vocab**: 30,522 tokens vs 50,265
5. **Attention Decay Integration**: Spatial decay mask applied

#### Forward Pass Integration
```julia
function (model::GraphMERTModel)(input_ids, attention_mask, position_ids, token_type_ids, leafy_graph)
    # 1. RoBERTa encoding
    roberta_output, pooled = model.roberta(input_ids, attention_mask, position_ids, token_type_ids)

    # 2. Create spatial decay mask from leafy graph
    decay_mask = create_attention_decay_mask(leafy_graph.distance_matrix, model.config)

    # 3. Apply decay to attention (integrated into H-GAT)
    hgat_output = model.hgat(roberta_output, leafy_graph.adjacency_matrix, decay_mask)

    # 4. Entity/relation classification
    entity_logits = model.entity_classifier(hgat_output)
    relation_logits = compute_relation_logits(hgat_output, leafy_graph, model.relation_classifier)

    return entity_logits, relation_logits, hgat_output
end
```

### 3. H-GAT (Hierarchical Graph Attention)
**Source**: `04-hgat-component.md` (580 lines)

#### Architecture
```julia
struct HGATConfig
    input_dim::Int = 512
    hidden_dim::Int = 256
    num_heads::Int = 8
    num_layers::Int = 2
    dropout_rate::Float64 = 0.1
    attention_dropout_rate::Float64 = 0.1
    layer_norm_eps::Float64 = 1e-12
    use_residual::Bool = true
    use_layer_norm::Bool = true
end

struct HGATLayer
    attention::MultiHeadAttention
    feed_forward::FeedForward
    layer_norm1::LayerNorm
    layer_norm2::LayerNorm
    dropout::Dropout
end
```

#### Key Features
- **Relation-Aware Attention**: Different attention for different relation types
- **Hierarchical Structure**: Processes root-leaf relationships
- **Residual Connections**: Maintains gradient flow
- **Relation Embeddings**: Learned embeddings for 50+ relation types

### 4. Dual Training Objectives
**Source**: `06-training-mlm.md` (640 lines) + `07-training-mnm.md` (823 lines)

#### MLM (Masked Language Modeling) - Syntactic
```julia
struct MLMConfig
    vocab_size::Int = 30522
    hidden_size::Int = 512
    max_length::Int = 512
    mask_probability::Float64 = 0.15      # 15% of tokens
    span_length::Int = 7                  # Matches leaf size
    boundary_loss_weight::Float64 = 1.0   # Equal weight
    temperature::Float64 = 1.0
end

# Span masking strategy (SpanBERT-inspired)
function create_span_masks(input_ids, config)
    # 1. Select spans of length 7
    # 2. Apply 80/10/10 masking strategy
    # 3. Calculate boundary loss
end

L_MLM = L_main + λ_boundary * L_boundary
```

#### MNM (Masked Node Modeling) - Semantic
```julia
struct MNMConfig
    vocab_size::Int = 30522
    hidden_size::Int = 512
    num_leaves::Int = 7
    mask_probability::Float64 = 0.15
    relation_dropout::Float64 = 0.3
    loss_weight::Float64 = 1.0
end

# Key difference: Masks ENTIRE leaf groups
function select_leaves_to_mask(graph, mnm_probability)
    # For each selected root, mask ALL 7 leaves
    # Ensures gradients flow through complete triples
end

L_MNM = -∑_{ℓ∈M_g} log P_θ(g_ℓ | G_{∖M_g∪M_x})
```

#### Joint Training Loss
```julia
L_total = L_MLM + μ * L_MNM  # μ = 1.0 (equal weighting)
```

### 5. Seed KG Injection Pipeline
**Source**: `08-seed-kg-injection.md` (698 lines)

#### 4-Stage Pipeline
```julia
struct SeedInjectionConfig
    entity_linking_threshold::Float64 = 0.5
    top_k_candidates::Int = 10
    top_n_triples_per_entity::Int = 40
    alpha_score_threshold::Float64 = 0.7
    score_bucket_size::Int = 10
    relation_bucket_size::Int = 5
    injection_ratio::Float64 = 0.2
    max_triples_per_sequence::Int = 10
end
```

#### Stages
1. **Entity Linking**: SapBERT embeddings + Jaccard similarity
2. **Triple Retrieval**: UMLS API for entity triples
3. **Contextual Selection**: Embedding-based relevance
4. **Injection Algorithm**: Score + diversity bucketing

#### Algorithm Complexity
- **SapBERT**: O(n * d) where n=entities, d=embedding_dim
- **Bucketing**: O(t log t) where t=triples
- **Injection**: O(n * t) where n=nodes, t=triples

### 6. Triple Extraction Pipeline
**Source**: `09-triple-extraction.md` (909 lines)

#### 5-Stage Pipeline
```julia
struct TripleExtractionConfig
    llm_model::String = "gpt-3.5-turbo"
    confidence_threshold::Float64 = 0.7
    max_candidates::Int = 20
    similarity_threshold::Float64 = 0.8
    batch_size::Int = 32
end
```

#### Stages
1. **Head Discovery**: LLM extracts entity mentions
2. **Relation Matching**: LLM pairs entities with relations
3. **Tail Prediction**: GraphMERT predicts tail tokens
4. **Tail Formation**: LLM forms coherent tail entities
5. **Filtering**: Similarity filtering + deduplication

### 7. Attention Mechanisms
**Source**: `05-attention-mechanisms.md` (598 lines)

#### Spatial Decay Attention
```julia
struct SpatialAttentionConfig
    max_distance::Int = 512
    decay_lambda::Float64 = 0.6
    decay_p_init::Float64 = 1.0
    use_distance_bias::Bool = true
    distance_bias_weight::Float64 = 0.1
end

# Exponential decay based on graph distance
function create_attention_decay_mask(distance_matrix, config)
    decay_mask = exp.(-config.decay_lambda * distance_matrix)
    return decay_mask
end
```

#### Integration with RoBERTa
- Applied before self-attention computation
- Modifies attention scores: `attention = attention * decay_mask`
- Ensures local coherence in leafy chain structure

### 8. Data Structures
**Source**: `11-data-structures.md` (600 lines)

#### Core Types
```julia
# Knowledge Graph Types
struct KnowledgeEntity
    id::String
    text::String
    label::String
    confidence::Float64
    position::TextPosition
    attributes::Dict{String,Any}
    created_at::DateTime
end

struct KnowledgeRelation
    head::String
    tail::String
    relation_type::String
    confidence::Float64
    attributes::Dict{String,Any}
    created_at::DateTime
end

struct KnowledgeGraph
    entities::Vector{KnowledgeEntity}
    relations::Vector{KnowledgeRelation}
    metadata::Dict{String,Any}
    created_at::DateTime
end

# Training Types
struct SemanticTriple
    head::String
    head_cui::Union{String,Nothing}
    relation::String
    tail::String
    tail_tokens::Vector{Int}
    score::Float64
    source::String
end

struct MLMBatch
    input_ids::Matrix{Int}
    attention_mask::Matrix{Int}
    labels::Matrix{Int}
    masked_positions::Vector{Int}
    span_boundaries::Vector{Tuple{Int,Int}}
end

struct MNMBatch
    graph_sequence::Matrix{Int}
    attention_mask::Array{Bool,3}
    masked_leaf_spans::Vector{Vector{Tuple{Int,Int}}}
    original_leaf_tokens::Vector{Vector{Int}}
    relation_ids::Matrix{Int}
end
```

### 9. Evaluation Metrics
**Source**: `10-evaluation-metrics.md` (550 lines)

#### FActScore* (Primary Metric)
```julia
# FActScore*(G) = E_{τ∈G}[f(τ)]
# where f(τ) = 1[τ is supported by C(τ)]
# and C(τ) is context sentences containing head and tail

struct FActScoreResult
    triple_scores::Vector{Bool}
    triple_contexts::Vector{String}
    factscore::Float64
    supported_triples::Int
    total_triples::Int
    metadata::Dict{String,Any}
end
```

#### ValidityScore (Secondary Metric)
- Ontological consistency checking
- Relation type validation
- Entity type compatibility

#### GraphRAG (Retrieval-Augmented Generation)
- Integration with LLMs for evaluation
- Context retrieval from KG
- Answer generation quality

## Performance Targets

### Training Performance
- **Throughput**: 5,000 tokens/second on laptop
- **Memory**: < 4GB for datasets up to 124.7M tokens
- **Training Time**: ~90 GPU hours (4× H100 equivalent)

### Evaluation Targets
- **FActScore**: ≥ 69.8% (paper baseline)
- **ValidityScore**: ≥ 68.8% (paper baseline)
- **Inference Speed**: > 4,000 tokens/second

### Resource Requirements
- **Model Size**: 80M parameters
- **Memory Peak**: ~500MB during training
- **Disk Space**: ~320MB (model) + training data

## Integration Points

### Component Dependencies
```
RoBERTa ← H-GAT ← MNM Training
    ↑         ↑         ↑
    └── Seed Injection ←─┘
              ↑
        Triple Extraction
```

### Data Flow
1. **Training Data**: Raw text → Seed KG Injection → Leafy Chain Graphs
2. **Training**: MLM + MNM objectives on graph sequences
3. **Inference**: Text → Triple Extraction → Knowledge Graph

### External Dependencies
- **SapBERT**: Entity linking (can be mocked for development)
- **UMLS API**: Knowledge graph triples (can be cached)
- **LLM API**: Head discovery, relation matching, tail formation
- **BioMedBERT Vocab**: Biomedical tokenization

## Testing Strategy

### Unit Testing
- All algorithms have mathematical verification
- Edge cases covered in specifications
- Performance benchmarks included

### Integration Testing
- Component interfaces validated
- Data flow between modules tested
- End-to-end pipelines verified

### Performance Testing
- Memory usage monitoring
- Throughput measurement
- Convergence validation

## Implementation Priorities

### Phase 1: Foundation (Week 1-2)
1. Extend type system from Doc 11
2. Implement Leafy Chain Graph (Doc 02)

### Phase 2: Training Preparation (Week 3-4)
3. Implement Seed KG Injection (Doc 08)

### Phase 3: Training Implementation (Week 5-6)
4. Implement MNM Training (Doc 07)
5. Complete Training Pipeline integration

### Phase 4: Extraction Implementation (Week 7-8)
6. Implement Helper LLM Integration
7. Implement Triple Extraction (Doc 09)

### Phase 5: Enhancement & Validation (Week 9-10)
8. Add Attention Mechanisms (Doc 05)
9. Implement Evaluation Metrics (Doc 10)

## Success Criteria

### Functional Completeness
- ✅ Training pipeline works end-to-end
- ✅ Triple extraction produces valid KGs
- ✅ Evaluation metrics match paper baselines

### Code Quality
- ✅ 90% test coverage
- ✅ Comprehensive documentation
- ✅ Performance meets targets

### Research Reproducibility
- ✅ Results match paper within 5%
- ✅ Ablation studies possible
- ✅ Hyperparameter tuning supported

---

*This specification is derived from the complete GraphMERT paper analysis in `/original_paper/expanded_rewrite/`. All algorithms, data structures, and performance requirements are specified in detail across 15 comprehensive documents totaling ~9,800 lines.*
