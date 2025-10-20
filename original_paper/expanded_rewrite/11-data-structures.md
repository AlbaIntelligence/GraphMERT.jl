# Document 11: Data Structures and Types
## Complete Julia Type Definitions

**Status**: 🟡 **Needs Expansion from 272 lines**
**Priority**: P1 (High - needed for all implementation)
**Paper Reference**: Throughout
**Existing Code**: `GraphMERT/src/types.jl` (272 lines, partial)

---

## Overview

This document provides **complete Julia type definitions** for all GraphMERT components. These types form the foundation for all implementation work.

**Organization**:
1. Core types (already in `types.jl`)
2. Graph structures (from Doc 02)
3. Training configurations (from Docs 06-07)
4. Injection and extraction (from Docs 08-09)
5. Evaluation types (from Doc 10)

---

## Part 1: Core Types (Existing in types.jl)

### Text Position

```julia
"""
    TextPosition

Position of text in a document.
"""
struct TextPosition
    start::Int
    stop::Int
    line::Int
    column::Int
end
```

### Biomedical Entities

```julia
"""
    BiomedicalEntity

Represents a biomedical entity in the knowledge graph.
"""
struct BiomedicalEntity
    id::String
    text::String
    label::String
    confidence::Float64
    position::TextPosition
    attributes::Dict{String,Any}
    created_at::DateTime

    function BiomedicalEntity(
        id::String, text::String, label::String,
        confidence::Float64, position::TextPosition,
        attributes::Dict{String,Any}=Dict{String,Any}(),
        created_at::DateTime=now()
    )
        new(id, text, label, confidence, position, attributes, created_at)
    end
end
```

### Biomedical Relations

```julia
"""
    BiomedicalRelation

Relation between two biomedical entities.
"""
struct BiomedicalRelation
    head::String
    tail::String
    relation_type::String
    confidence::Float64
    attributes::Dict{String,Any}
    created_at::DateTime

    function BiomedicalRelation(
        head::String, tail::String, relation_type::String,
        confidence::Float64,
        attributes::Dict{String,Any}=Dict{String,Any}(),
        created_at::DateTime=now()
    )
        new(head, tail, relation_type, confidence, attributes, created_at)
    end
end
```

### Knowledge Graph

```julia
"""
    KnowledgeGraph

Knowledge graph with entities and relations.
"""
struct KnowledgeGraph
    entities::Vector{BiomedicalEntity}
    relations::Vector{BiomedicalRelation}
    metadata::Dict{String,Any}
    created_at::DateTime

    function KnowledgeGraph(
        entities::Vector{BiomedicalEntity},
        relations::Vector{BiomedicalRelation},
        metadata::Dict{String,Any}=Dict{String,Any}(),
        created_at::DateTime=now()
    )
        new(entities, relations, metadata, created_at)
    end
end
```

---

## Part 2: Graph Structures (From Doc 02)

### Chain Graph Node

```julia
"""
    ChainGraphNode

Single node in a leafy chain graph.
"""
struct ChainGraphNode
    # Identification
    id::Int
    node_type::Symbol  # :root or :leaf

    # Position
    root_index::Int
    leaf_index::Union{Int,Nothing}

    # Token information
    token_id::Int
    token_text::String
    is_padding::Bool

    # Semantic information (leaves only)
    relation::Union{Symbol,Nothing}
    head_text::Union{String,Nothing}

    # Embedding (filled during forward pass)
    embedding::Union{Vector{Float32},Nothing}
end
```

### Chain Graph Config

```julia
"""
    ChainGraphConfig

Configuration for leafy chain graph construction.
"""
struct ChainGraphConfig
    num_roots::Int
    num_leaves_per_root::Int
    max_sequence_length::Int
    pad_token_id::Int
    vocab_size::Int
end

function default_chain_graph_config()
    ChainGraphConfig(128, 7, 1024, 0, 30522)
end
```

### Leafy Chain Graph

```julia
"""
    LeafyChainGraph

Complete leafy chain graph for GraphMERT training/extraction.
"""
struct LeafyChainGraph
    # Graph structure
    nodes::Vector{ChainGraphNode}
    adjacency_matrix::SparseMatrixCSC{Float32}
    shortest_paths::Matrix{Int}

    # Root information (syntactic)
    root_tokens::Vector{Int}
    root_texts::Vector{String}

    # Leaf information (semantic)
    leaf_tokens::Matrix{Int}              # 128×7
    leaf_relations::Matrix{Union{Symbol,Nothing}}
    injected_mask::Matrix{Bool}

    # Metadata
    source_sequence_id::String
    sequence_length::Int
    num_injections::Int

    # Config
    config::ChainGraphConfig
end
```

---

## Part 3: Model Configuration

### GraphMERT Configuration

```julia
"""
    GraphMERTConfig

Complete configuration for GraphMERT model.
"""
struct GraphMERTConfig
    # Model architecture
    model_path::String
    vocab_size::Int
    hidden_size::Int
    num_attention_heads::Int
    num_hidden_layers::Int
    max_position_embeddings::Int
    type_vocab_size::Int

    # Training parameters
    initializer_range::Float64
    layer_norm_eps::Float64
    dropout::Float64
    attention_dropout::Float64

    # Generation parameters
    use_cache::Bool
    pad_token_id::Int
    bos_token_id::Int
    eos_token_id::Int
    max_length::Int
    temperature::Float64
    top_k::Int
    top_p::Float64
    repetition_penalty::Float64
    length_penalty::Float64

    # Output control
    output_attentions::Bool
    output_hidden_states::Bool
    return_dict::Bool
end

function default_graphmert_config()
    GraphMERTConfig(
        model_path = "",
        vocab_size = 30522,
        hidden_size = 512,
        num_attention_heads = 8,
        num_hidden_layers = 12,
        max_position_embeddings = 1024,
        type_vocab_size = 2,
        initializer_range = 0.02,
        layer_norm_eps = 1e-12,
        dropout = 0.1,
        attention_dropout = 0.1,
        use_cache = true,
        pad_token_id = 0,
        bos_token_id = 1,
        eos_token_id = 2,
        max_length = 1024,
        temperature = 1.0,
        top_k = 20,
        top_p = 1.0,
        repetition_penalty = 1.0,
        length_penalty = 1.0,
        output_attentions = false,
        output_hidden_states = false,
        return_dict = true
    )
end
```

### Processing Options

```julia
"""
    ProcessingOptions

Options for text processing and KG extraction.
"""
struct ProcessingOptions
    max_length::Int
    batch_size::Int
    use_umls::Bool
    use_helper_llm::Bool
    confidence_threshold::Float64
    entity_types::Vector{String}
    relation_types::Vector{String}
    cache_enabled::Bool
    parallel_processing::Bool
    verbose::Bool
end

function default_processing_options()
    ProcessingOptions(
        max_length = 512,
        batch_size = 32,
        use_umls = true,
        use_helper_llm = true,
        confidence_threshold = 0.5,
        entity_types = String[],
        relation_types = String[],
        cache_enabled = true,
        parallel_processing = false,
        verbose = false
    )
end
```

---

## Part 4: Training Configurations

### MLM Configuration

```julia
"""
    MLMConfig

Configuration for Masked Language Modeling.
"""
struct MLMConfig
    vocab_size::Int
    hidden_size::Int
    max_length::Int
    mask_probability::Float64
    span_length::Int
    boundary_loss_weight::Float64
    temperature::Float64
    mask_token_id::Int
end

function default_mlm_config()
    MLMConfig(
        vocab_size = 30522,
        hidden_size = 512,
        max_length = 1024,
        mask_probability = 0.15,
        span_length = 7,
        boundary_loss_weight = 1.0,
        temperature = 1.0,
        mask_token_id = 103
    )
end
```

### MNM Configuration

```julia
"""
    MNMConfig

Configuration for Masked Node Modeling.
"""
struct MNMConfig
    vocab_size::Int
    hidden_size::Int
    max_length::Int
    mask_probability::Float64
    mask_token_id::Int
    relation_embedding_dropout::Float64
    num_relations::Int
    temperature::Float64
    loss_weight::Float64  # μ parameter
end

function default_mnm_config()
    MNMConfig(
        vocab_size = 30522,
        hidden_size = 512,
        max_length = 1024,
        mask_probability = 0.15,
        mask_token_id = 103,
        relation_embedding_dropout = 0.3,
        num_relations = 28,
        temperature = 1.0,
        loss_weight = 1.0
    )
end
```

### Training Configuration

```julia
"""
    TrainingConfig

Complete training hyperparameters.
"""
struct TrainingConfig
    # Optimization
    learning_rate::Float64
    warmup_steps::Int
    max_steps::Int
    batch_size::Int
    gradient_accumulation_steps::Int

    # Regularization
    weight_decay::Float64
    dropout::Float64
    attention_dropout::Float64
    relation_dropout::Float64

    # Learning rate schedule
    lr_scheduler::Symbol  # :cosine, :linear, :constant
    min_lr::Float64

    # Graph parameters
    lambda::Float64  # Attention decay base

    # Checkpointing
    save_steps::Int
    eval_steps::Int
    logging_steps::Int

    # Hardware
    device::Symbol  # :cpu, :cuda
    precision::Symbol  # :fp32, :fp16, :bf16
    num_gpus::Int
end

function default_training_config()
    TrainingConfig(
        learning_rate = 4e-4,
        warmup_steps = 500,
        max_steps = 25 * 1000,  # 25 epochs × ~1000 steps
        batch_size = 32,
        gradient_accumulation_steps = 4,
        weight_decay = 0.01,
        dropout = 0.1,
        attention_dropout = 0.1,
        relation_dropout = 0.3,
        lr_scheduler = :cosine,
        min_lr = 1e-5,
        lambda = 0.6,
        save_steps = 1000,
        eval_steps = 500,
        logging_steps = 100,
        device = :cuda,
        precision = :bf16,
        num_gpus = 4
    )
end
```

---

## Part 5: Seed KG Injection Types

### Entity Mention

```julia
"""
    EntityMention

Entity mention discovered in text.
"""
struct EntityMention
    text::String
    start_pos::Int
    end_pos::Int
    confidence::Float64
    cui::Union{String,Nothing}  # UMLS CUI if linked
    semantic_types::Vector{String}
end
```

### UMLS Triple

```julia
"""
    UMLSTriple

Triple from UMLS knowledge graph.
"""
struct UMLSTriple
    head_cui::String
    head_name::String
    relation::Symbol
    tail_cui::String
    tail_name::String
    source_vocabulary::String  # SNOMED_CT, GO, etc.
end
```

### Matched Triple

```julia
"""
    MatchedTriple

Triple matched to a specific sequence.
"""
struct MatchedTriple
    sequence_id::String
    sequence_text::String
    head::String
    head_cui::String
    relation::Symbol
    tail::String
    tail_cui::String
    similarity_score::Float64
    embedding_similarity::Float64
    string_similarity::Float64
end
```

### Seed Injection Configuration

```julia
"""
    SeedInjectionConfig

Configuration for seed KG injection pipeline.
"""
struct SeedInjectionConfig
    # Entity linking
    sapbert_model_path::String
    umls_index_path::String
    string_sim_threshold::Float64
    top_k_candidates::Int

    # Triple selection
    umls_kg_path::String
    allowed_relations::Set{Symbol}
    top_k_per_entity::Int
    embedding_model::String  # "gemini-text-embedding-004"

    # Injection algorithm
    alpha::Float64  # Similarity threshold
    score_bucket_size::Float64
    relation_bucket_size::Int

    # Helper LLM
    helper_llm_model::String
    helper_llm_api_key::Union{String,Nothing}
    helper_llm_endpoint::Union{String,Nothing}
end

function default_seed_injection_config()
    SeedInjectionConfig(
        sapbert_model_path = "path/to/sapbert",
        umls_index_path = "path/to/umls_index",
        string_sim_threshold = 0.5,
        top_k_candidates = 10,
        umls_kg_path = "path/to/umls_kg",
        allowed_relations = Set([:isa, :associated_with, :cause_of]),
        top_k_per_entity = 40,
        embedding_model = "gemini-text-embedding-004",
        alpha = 0.55,
        score_bucket_size = 0.05,
        relation_bucket_size = 20,
        helper_llm_model = "qwen3-32b",
        helper_llm_api_key = nothing,
        helper_llm_endpoint = nothing
    )
end
```

---

## Part 6: Triple Extraction Types

### Extraction Triple

```julia
"""
    ExtractionTriple

Triple extracted from trained model with provenance.
"""
struct ExtractionTriple
    head::String
    relation::Symbol
    tail::String
    similarity::Float64
    confidence::Float64
    source_sequence::String
    source_sequence_id::String
    predicted_tokens::Vector{String}
    token_probabilities::Vector{Float64}
    extracted_at::DateTime
end
```

### Extraction Configuration

```julia
"""
    ExtractionConfig

Configuration for knowledge graph extraction.
"""
struct ExtractionConfig
    # Helper LLM
    helper_llm_model::String
    helper_llm_api_key::Union{String,Nothing}
    domain::String

    # Relations
    available_relations::Vector{Symbol}

    # Prediction
    top_k_tokens::Int

    # Filtering
    embedding_model::String
    beta::Float64  # Similarity threshold
    min_confidence::Float64

    # Provenance
    track_provenance::Bool
    include_probabilities::Bool

    # Caching
    cache_llm_responses::Bool
    cache_dir::String
end

function default_extraction_config()
    ExtractionConfig(
        helper_llm_model = "qwen3-32b",
        helper_llm_api_key = nothing,
        domain = "diabetes",
        available_relations = Symbol[],  # Load from seed KG
        top_k_tokens = 20,
        embedding_model = "gemini-text-embedding-004",
        beta = 0.67,
        min_confidence = 0.5,
        track_provenance = true,
        include_probabilities = true,
        cache_llm_responses = true,
        cache_dir = "cache/llm_responses"
    )
end
```

---

## Part 7: Evaluation Types

### FActScore

```julia
"""
    FActScore

FActScore evaluation metric.
"""
struct FActScore
    score::Float64
    precision::Float64
    recall::Float64
    f1::Float64
    total_facts::Int
    correct_facts::Int
    incorrect_facts::Int

    function FActScore(
        score::Float64,
        precision::Float64,
        recall::Float64,
        f1::Float64,
        total_facts::Int,
        correct_facts::Int,
        incorrect_facts::Int
    )
        new(score, precision, recall, f1, total_facts, correct_facts, incorrect_facts)
    end
end
```

### ValidityScore

```julia
"""
    ValidityScore

ValidityScore evaluation metric.
"""
struct ValidityScore
    score::Float64
    valid_relations::Int
    total_relations::Int
    invalid_relations::Int
    yes_count::Int
    maybe_count::Int
    no_count::Int

    function ValidityScore(
        score::Float64,
        valid_relations::Int,
        total_relations::Int,
        invalid_relations::Int,
        yes_count::Int,
        maybe_count::Int,
        no_count::Int
    )
        new(score, valid_relations, total_relations, invalid_relations,
            yes_count, maybe_count, no_count)
    end
end
```

### GraphRAG Score

```julia
"""
    GraphRAGScore

GraphRAG evaluation metric.
"""
struct GraphRAGScore
    score::Float64
    retrieval_accuracy::Float64
    generation_quality::Float64
    overall_performance::Float64
    num_queries::Int
    correct_answers::Int
end
```

---

## Part 8: Batch Structures

### MLM Batch

```julia
"""
    MLMBatch

Batch data for MLM training.
"""
struct MLMBatch
    input_ids::Matrix{Int}           # [batch_size, seq_len]
    attention_mask::Matrix{Int}
    labels::Matrix{Int}
    masked_positions::Vector{Int}
    span_boundaries::Vector{Tuple{Int,Int}}
end
```

### MNM Batch

```julia
"""
    MNMBatch

Batch data for MNM training.
"""
struct MNMBatch
    input_ids::Matrix{Int}
    attention_mask::Matrix{Int}
    mlm_labels::Matrix{Int}
    mnm_labels::Array{Int,3}  # [batch_size, num_roots, num_leaves]
    masked_leaves::Vector{Tuple{Int,Int}}  # (root_idx, leaf_idx) pairs
    relations::Matrix{Union{Symbol,Nothing}}
end
```

### Joint Training Batch

```julia
"""
    JointTrainingBatch

Combined MLM+MNM batch.
"""
struct JointTrainingBatch
    graphs::Vector{LeafyChainGraph}
    input_ids::Matrix{Int}
    attention_mask::Matrix{Int}
    mlm_labels::Matrix{Int}
    mnm_labels::Array{Int,3}
    mlm_spans::Vector{Vector{Tuple{Int,Int}}}
    mnm_leaves::Vector{Vector{Tuple{Int,Int}}}
end
```

---

## Part 9: Model Structures

### GraphMERT Model

```julia
"""
    GraphMERTModel

Main GraphMERT model structure.
"""
mutable struct GraphMERTModel
    config::GraphMERTConfig
    roberta::Any  # RoBERTa encoder
    hgat::Any  # H-GAT component
    classifier::Any  # Output projection
    is_training::Bool

    function GraphMERTModel(config::GraphMERTConfig)
        new(config, nothing, nothing, nothing, false)
    end
end
```

---

## Part 10: Helper Types

### LLM Client

```julia
"""
    LLMClient

Client for helper LLM interaction.
"""
struct LLMClient
    model_name::String
    api_key::Union{String,Nothing}
    endpoint::Union{String,Nothing}
    temperature::Float64
    max_tokens::Int
    top_p::Float64
    top_k::Int
    thinking_mode::Bool
end

function create_llm_client(
    model_name::String;
    api_key=nothing,
    endpoint=nothing,
    temperature=0.6,
    max_tokens=8192,
    thinking_mode=true
)
    LLMClient(
        model_name,
        api_key,
        endpoint,
        temperature,
        max_tokens,
        0.95,
        20,
        thinking_mode
    )
end
```

### Embedding Model

```julia
"""
    EmbeddingModel

Embedding model for similarity calculations.
"""
struct EmbeddingModel
    model_name::String
    dimension::Int
    api_key::Union{String,Nothing}
    cache::Dict{String,Vector{Float32}}
end
```

---

## Part 11: Utility Types

### Training Progress

```julia
"""
    TrainingProgress

Track training progress and metrics.
"""
mutable struct TrainingProgress
    epoch::Int
    step::Int
    mlm_loss::Float64
    mnm_loss::Float64
    total_loss::Float64
    learning_rate::Float64
    throughput::Float64  # samples/sec
    elapsed_time::Float64
    best_loss::Float64
    patience_counter::Int
end
```

### Checkpoint

```julia
"""
    Checkpoint

Model checkpoint with metadata.
"""
struct Checkpoint
    model_state::Dict{String,Any}
    optimizer_state::Dict{String,Any}
    training_progress::TrainingProgress
    config::GraphMERTConfig
    timestamp::DateTime
    step::Int
    epoch::Int
end
```

---

## Type Relationships

### Inheritance Hierarchy

```
Abstract Types:
  - No abstract types used (Julia favor composition)

Concrete Types:
  - All types are concrete structs
  - Immutable by default (except Model and Progress)
  - Composition over inheritance
```

### Key Relationships

```
GraphMERTModel
  ├─ config: GraphMERTConfig
  ├─ roberta: RoBERTa
  └─ hgat: H-GAT

LeafyChainGraph
  ├─ nodes: Vector{ChainGraphNode}
  ├─ config: ChainGraphConfig
  └─ [used in] JointTrainingBatch

KnowledgeGraph
  ├─ entities: Vector{BiomedicalEntity}
  └─ relations: Vector{BiomedicalRelation}

ExtractionTriple → KnowledgeGraph (after aggregation)
```

---

## Usage Examples

### Creating a Training Batch

```julia
# Create graphs
graphs = [create_empty_chain_graph(...) for _ in 1:32]

# Create batch
batch = JointTrainingBatch(
    graphs = graphs,
    input_ids = stack([graph_to_sequence(g) for g in graphs]),
    attention_mask = stack([create_attention_mask(g) for g in graphs]),
    mlm_labels = create_mlm_labels(graphs),
    mnm_labels = create_mnm_labels(graphs),
    mlm_spans = [find_mlm_spans(g) for g in graphs],
    mnm_leaves = [find_mnm_leaves(g) for g in graphs]
)
```

### Creating Extraction Config

```julia
config = ExtractionConfig(
    helper_llm_model = "qwen3-32b",
    domain = "diabetes",
    available_relations = [:isa, :associated_with, :cause_of],
    top_k_tokens = 20,
    beta = 0.67,
    track_provenance = true
)
```

---

## Implementation Checklist

- [ ] Add missing types from this document to `types.jl`
- [ ] Create `graph_types.jl` for chain graph structures
- [ ] Create `training_types.jl` for training configurations
- [ ] Create `extraction_types.jl` for extraction structures
- [ ] Create `evaluation_types.jl` for metrics
- [ ] Add constructor validation
- [ ] Add display/show methods
- [ ] Add serialization/deserialization
- [ ] Write comprehensive tests
- [ ] Document all fields

---

**Related Documents**:
- → [Doc 02: Leafy Chain Graphs](02-leafy-chain-graphs.md)
- → [Doc 06-07: Training](06-training-mlm.md)
- → [Doc 08: Seed Injection](08-seed-kg-injection.md)
- → [Doc 09: Extraction](09-triple-extraction.md)
- → [Doc 10: Evaluation](10-evaluation-metrics.md)
