"""
Core type definitions for GraphMERT.jl

This module defines the fundamental data structures used throughout
the GraphMERT implementation.
"""

using Dates

# ============================================================================
# New Types for GraphMERT Enhancement
# ============================================================================

# ============================================================================
# Text Position
# ============================================================================

"""
    TextPosition

Represents the position of text in a document.

$(FIELDS)
"""
struct TextPosition
  start::Int
  stop::Int
  line::Int
  column::Int
end

# ============================================================================
# Knowledge Entities
# ============================================================================

"""
    KnowledgeEntity

Represents a knowledge entity in the knowledge graph.

$(FIELDS)
"""
struct KnowledgeEntity
  id::String
  text::String
  label::String
  confidence::Float64
  position::TextPosition
  attributes::Dict{String,Any}
  created_at::DateTime

  function KnowledgeEntity(
    id::String,
    text::String,
    label::String,
    confidence::Float64,
    position::TextPosition,
    attributes::Dict{String,Any}=Dict{String,Any}(),
    created_at::DateTime=now(),
  )
    new(id, text, label, confidence, position, attributes, created_at)
  end
end

"""
    KnowledgeRelation

Represents a relation between two knowledge entities.

$(FIELDS)
"""
struct KnowledgeRelation
  head::String
  tail::String
  relation_type::String
  confidence::Float64
  attributes::Dict{String,Any}
  created_at::DateTime

  function KnowledgeRelation(
    head::String,
    tail::String,
    relation_type::String,
    confidence::Float64,
    attributes::Dict{String,Any}=Dict{String,Any}(),
    created_at::DateTime=now(),
  )
    new(head, tail, relation_type, confidence, attributes, created_at)
  end
end

# ============================================================================
# Knowledge Graph
# ============================================================================

"""
    KnowledgeGraph

Represents a knowledge graph with entities and relations.

$(FIELDS)
"""
struct KnowledgeGraph
  entities::Vector{KnowledgeEntity}
  relations::Vector{KnowledgeRelation}
  metadata::Dict{String,Any}
  created_at::DateTime

  function KnowledgeGraph(
    entities::Vector{KnowledgeEntity},
    relations::Vector{KnowledgeRelation},
    metadata::Dict{String,Any}=Dict{String,Any}(),
    created_at::DateTime=now(),
  )
    new(entities, relations, metadata, created_at)
  end
end

# ============================================================================
# Model Configuration
# ============================================================================

"""
    GraphMERTConfig

Configuration for GraphMERT model.

$(FIELDS)
"""
struct GraphMERTConfig
  model_path::String
  vocab_size::Int
  hidden_size::Int
  num_attention_heads::Int
  num_hidden_layers::Int
  max_position_embeddings::Int
  type_vocab_size::Int
  initializer_range::Float64
  layer_norm_eps::Float64
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
  output_attentions::Bool
  output_hidden_states::Bool
  return_dict::Bool

  function GraphMERTConfig(;
    model_path::String="",
    vocab_size::Int=50265,
    hidden_size::Int=768,
    num_attention_heads::Int=12,
    num_hidden_layers::Int=12,
    max_position_embeddings::Int=512,
    type_vocab_size::Int=2,
    initializer_range::Float64=0.02,
    layer_norm_eps::Float64=1e-12,
    use_cache::Bool=true,
    pad_token_id::Int=1,
    bos_token_id::Int=0,
    eos_token_id::Int=2,
    max_length::Int=512,
    temperature::Float64=1.0,
    top_k::Int=50,
    top_p::Float64=1.0,
    repetition_penalty::Float64=1.0,
    length_penalty::Float64=1.0,
    output_attentions::Bool=false,
    output_hidden_states::Bool=false,
    return_dict::Bool=true,
  )
    new(
      model_path,
      vocab_size,
      hidden_size,
      num_attention_heads,
      num_hidden_layers,
      max_position_embeddings,
      type_vocab_size,
      initializer_range,
      layer_norm_eps,
      use_cache,
      pad_token_id,
      bos_token_id,
      eos_token_id,
      max_length,
      temperature,
      top_k,
      top_p,
      repetition_penalty,
      length_penalty,
      output_attentions,
      output_hidden_states,
      return_dict,
    )
  end
end

"""
    ProcessingOptions

Options for text processing and knowledge graph extraction.

$(FIELDS)
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

  function ProcessingOptions(;
    max_length::Int=512,
    batch_size::Int=32,
    use_umls::Bool=true,
    use_helper_llm::Bool=true,
    confidence_threshold::Float64=0.5,
    entity_types::Vector{String}=String[],
    relation_types::Vector{String}=String[],
    cache_enabled::Bool=true,
    parallel_processing::Bool=false,
    verbose::Bool=false,
  )
    new(
      max_length,
      batch_size,
      use_umls,
      use_helper_llm,
      confidence_threshold,
      entity_types,
      relation_types,
      cache_enabled,
      parallel_processing,
      verbose,
    )
  end
end

# ============================================================================
# Model Structures
# ============================================================================

"""
    GraphMERTModel

Main GraphMERT model structure.

$(FIELDS)
"""
mutable struct GraphMERTModel
  config::GraphMERTConfig
  roberta::Any
  hgat::Any
  classifier::Any
  is_training::Bool

  function GraphMERTModel(config::GraphMERTConfig)
    new(config, nothing, nothing, nothing, false)
  end
end

# ============================================================================
# Evaluation Metrics
# ============================================================================

"""
    FActScore

FActScore evaluation metric.

$(FIELDS)
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
    incorrect_facts::Int,
  )
    new(score, precision, recall, f1, total_facts, correct_facts, incorrect_facts)
  end
end

"""
    ValidityScore

ValidityScore evaluation metric.

$(FIELDS)
"""
struct ValidityScore
  score::Float64
  valid_relations::Int
  total_relations::Int
  invalid_relations::Int

  function ValidityScore(
    score::Float64,
    valid_relations::Int,
    total_relations::Int,
    invalid_relations::Int,
  )
    new(score, valid_relations, total_relations, invalid_relations)
  end
end

"""
    GraphRAG

GraphRAG evaluation metric.

$(FIELDS)
"""
struct GraphRAG
  score::Float64
  retrieval_accuracy::Float64
  generation_quality::Float64
  overall_performance::Float64

  function GraphRAG(
    score::Float64,
    retrieval_accuracy::Float64,
    generation_quality::Float64,
    overall_performance::Float64,
  )
    new(score, retrieval_accuracy, generation_quality, overall_performance)
  end
end

# ============================================================================
# Enhanced Types for GraphMERT Enhancement (from data-model.md)
# ============================================================================

"""
    ChainGraphNode

Represents a node in the leafy chain graph structure.

$(FIELDS)
"""
struct ChainGraphNode
  node_type::Symbol  # :root or :leaf
  token_id::Int
  position::Int
  parent_root::Union{Int,Nothing}
  is_padding::Bool

  function ChainGraphNode(
    node_type::Symbol,
    token_id::Int,
    position::Int,
    parent_root::Union{Int,Nothing}=nothing,
    is_padding::Bool=false,
  )
    @assert node_type ∈ [:root, :leaf] "node_type must be :root or :leaf"
    @assert token_id ≥ 0 "token_id must be non-negative"
    if node_type == :leaf
      @assert parent_root !== nothing "leaf nodes must have parent_root"
    else
      @assert parent_root === nothing "root nodes cannot have parent_root"
    end
    new(node_type, token_id, position, parent_root, is_padding)
  end
end

"""
    ChainGraphConfig

Configuration parameters for leafy chain graph construction.

$(FIELDS)
"""
struct ChainGraphConfig
  num_roots::Int
  num_leaves_per_root::Int
  vocab_size::Int
  pad_token_id::Int
  mask_token_id::Int
  precompute_shortest_paths::Bool

  function ChainGraphConfig(
    num_roots::Int=128,
    num_leaves_per_root::Int=7,
    vocab_size::Int=30522,
    pad_token_id::Int=1,
    mask_token_id::Int=103,
    precompute_shortest_paths::Bool=true,
  )
    @assert num_roots > 0 "num_roots must be positive"
    @assert num_leaves_per_root > 0 "num_leaves_per_root must be positive"
    @assert vocab_size > 0 "vocab_size must be positive"
    new(
      num_roots,
      num_leaves_per_root,
      vocab_size,
      pad_token_id,
      mask_token_id,
      precompute_shortest_paths,
    )
  end
end

"""
    LeafyChainGraph

Complete leafy chain graph structure combining roots (syntactic) and leaves (semantic).

$(FIELDS)
"""
struct LeafyChainGraph
  root_nodes::Vector{ChainGraphNode}
  leaf_nodes::Vector{Vector{ChainGraphNode}}
  adjacency_matrix::Matrix{Bool}
  shortest_paths::Union{Matrix{Int},Nothing}
  config::ChainGraphConfig
  metadata::Dict{String,Any}

  function LeafyChainGraph(
    root_nodes::Vector{ChainGraphNode},
    leaf_nodes::Vector{Vector{ChainGraphNode}},
    adjacency_matrix::Matrix{Bool},
    config::ChainGraphConfig;
    shortest_paths::Union{Matrix{Int},Nothing}=nothing,
    metadata::Dict{String,Any}=Dict{String,Any}(),
  )

    # Validation
    @assert length(root_nodes) == config.num_roots "root_nodes length must match config.num_roots"
    @assert length(leaf_nodes) == config.num_roots "leaf_nodes length must match config.num_roots"
    @assert all(length(leaves) == config.num_leaves_per_root for leaves in leaf_nodes) "all leaf groups must have config.num_leaves_per_root nodes"

    total_nodes = config.num_roots * (1 + config.num_leaves_per_root)
    @assert size(adjacency_matrix) == (total_nodes, total_nodes) "adjacency_matrix size must be (total_nodes, total_nodes)"

    if config.precompute_shortest_paths && shortest_paths !== nothing
      @assert size(shortest_paths) == (total_nodes, total_nodes) "shortest_paths size must match adjacency_matrix"
    end

    new(root_nodes, leaf_nodes, adjacency_matrix, shortest_paths, config, metadata)
  end
end

"""
    MNMConfig

Configuration for Masked Node Modeling training objective.

$(FIELDS)
"""
struct MNMConfig
  vocab_size::Int
  hidden_size::Int
  num_leaves::Int
  mask_probability::Float64
  relation_dropout::Float64
  loss_weight::Float64
  mask_entire_leaf_span::Bool
  mask_token_id::Int

  function MNMConfig(
    vocab_size::Int=30522,
    hidden_size::Int=512,
    num_leaves::Int=7,
    mask_probability::Float64=0.15,
    relation_dropout::Float64=0.3,
    loss_weight::Float64=1.0,
    mask_entire_leaf_span::Bool=true,
    mask_token_id::Int=103,
  )
    @assert vocab_size > 0 "vocab_size must be positive"
    @assert hidden_size > 0 "hidden_size must be positive"
    @assert num_leaves > 0 "num_leaves must be positive"
    @assert 0 < mask_probability < 1 "mask_probability must be between 0 and 1"
    @assert 0 ≤ relation_dropout ≤ 1 "relation_dropout must be between 0 and 1"
    @assert loss_weight > 0 "loss_weight must be positive"
    @assert mask_token_id ≥ 0 "mask_token_id must be non-negative"
    new(
      vocab_size,
      hidden_size,
      num_leaves,
      mask_probability,
      relation_dropout,
      loss_weight,
      mask_entire_leaf_span,
      mask_token_id,
    )
  end
end

"""
    SemanticTriple

Represents a knowledge graph triple for seed injection.

$(FIELDS)
"""
struct SemanticTriple
  head::String
  head_cui::Union{String,Nothing}
  relation::String
  tail::String
  tail_tokens::Vector{Int}
  score::Float64
  source::String

  function SemanticTriple(
    head::String,
    head_cui::Union{String,Nothing},
    relation::String,
    tail::String,
    tail_tokens::Vector{Int},
    score::Float64,
    source::String,
  )
    @assert !isempty(head) "head cannot be empty"
    @assert !isempty(relation) "relation cannot be empty"
    @assert !isempty(tail) "tail cannot be empty"
    @assert !isempty(tail_tokens) "tail_tokens cannot be empty"
    @assert 0 ≤ score ≤ 1 "score must be between 0 and 1"
    @assert !isempty(source) "source cannot be empty"
    new(head, head_cui, relation, tail, tail_tokens, score, source)
  end
end

"""
    SeedInjectionConfig

Configuration for seed KG injection algorithm.

$(FIELDS)
"""
struct SeedInjectionConfig
  entity_linking_threshold::Float64
  top_k_candidates::Int
  top_n_triples_per_entity::Int
  alpha_score_threshold::Float64
  score_bucket_size::Int
  relation_bucket_size::Int
  injection_ratio::Float64
  max_triples_per_sequence::Int

  function SeedInjectionConfig(
    entity_linking_threshold::Float64=0.5,
    top_k_candidates::Int=10,
    top_n_triples_per_entity::Int=40,
    alpha_score_threshold::Float64=0.7,
    score_bucket_size::Int=10,
    relation_bucket_size::Int=5,
    injection_ratio::Float64=0.2,
    max_triples_per_sequence::Int=10,
  )
    @assert 0 < entity_linking_threshold < 1 "entity_linking_threshold must be between 0 and 1"
    @assert top_k_candidates > 0 "top_k_candidates must be positive"
    @assert top_n_triples_per_entity > 0 "top_n_triples_per_entity must be positive"
    @assert 0 < alpha_score_threshold < 1 "alpha_score_threshold must be between 0 and 1"
    @assert score_bucket_size > 0 "score_bucket_size must be positive"
    @assert relation_bucket_size > 0 "relation_bucket_size must be positive"
    @assert 0 < injection_ratio < 1 "injection_ratio must be between 0 and 1"
    @assert max_triples_per_sequence > 0 "max_triples_per_sequence must be positive"
    new(
      entity_linking_threshold,
      top_k_candidates,
      top_n_triples_per_entity,
      alpha_score_threshold,
      score_bucket_size,
      relation_bucket_size,
      injection_ratio,
      max_triples_per_sequence,
    )
  end
end

"""
    LLMRequest

Request to helper LLM.

$(FIELDS)
"""
struct LLMRequest
  prompt::String
  context::String
  task_type::Symbol
  temperature::Float64
  max_tokens::Int

  function LLMRequest(
    prompt::String,
    context::String,
    task_type::Symbol,
    temperature::Float64=0.0,
    max_tokens::Int=500,
  )
    @assert !isempty(prompt) "prompt cannot be empty"
    @assert task_type ∈ [:entity_discovery, :relation_matching, :tail_formation] "invalid task_type"
    @assert 0 ≤ temperature ≤ 2 "temperature must be between 0 and 2"
    @assert max_tokens > 0 "max_tokens must be positive"
    new(prompt, context, task_type, temperature, max_tokens)
  end
end

"""
    LLMResponse

Response from helper LLM.

$(FIELDS)
"""
struct LLMResponse
  raw_response::String
  parsed_result::Any
  success::Bool
  error_message::Union{String,Nothing}
  metadata::Dict{String,Any}

  function LLMResponse(
    raw_response::String,
    parsed_result::Any,
    success::Bool;
    error_message::Union{String,Nothing}=nothing,
    metadata::Dict{String,Any}=Dict{String,Any}(),
  )
    @assert !isempty(raw_response) "raw_response cannot be empty"
    new(raw_response, parsed_result, success, error_message, metadata)
  end
end

"""
    EntityLinkingResult

Result of entity linking to UMLS.

$(FIELDS)
"""
struct EntityLinkingResult
  entity_text::String
  cui::String
  preferred_name::String
  semantic_types::Vector{String}
  similarity_score::Float64
  source::String

  function EntityLinkingResult(
    entity_text::String,
    cui::String,
    preferred_name::String,
    semantic_types::Vector{String},
    similarity_score::Float64,
    source::String,
  )
    @assert !isempty(entity_text) "entity_text cannot be empty"
    @assert !isempty(cui) "cui cannot be empty"
    @assert !isempty(preferred_name) "preferred_name cannot be empty"
    @assert 0 ≤ similarity_score ≤ 1 "similarity_score must be between 0 and 1"
    @assert !isempty(source) "source cannot be empty"
    new(entity_text, cui, preferred_name, semantic_types, similarity_score, source)
  end
end

"""
    MNMBatch

Batch data for MNM training.

$(FIELDS)
"""
struct MNMBatch
  graph_sequence::Matrix{Int}
  attention_mask::Array{Bool,3}
  masked_leaf_spans::Vector{Vector{Tuple{Int,Int}}}
  original_leaf_tokens::Vector{Vector{Int}}
  relation_ids::Matrix{Int}

  function MNMBatch(
    graph_sequence::Matrix{Int},
    attention_mask::Array{Bool,3},
    masked_leaf_spans::Vector{Vector{Tuple{Int,Int}}},
    original_leaf_tokens::Vector{Vector{Int}},
    relation_ids::Matrix{Int},
  )
    @assert size(graph_sequence, 1) == size(attention_mask, 1) "batch dimensions must match"
    @assert size(graph_sequence, 2) ==
            size(attention_mask, 2) ==
            size(attention_mask, 3) "sequence length must match"
    @assert length(masked_leaf_spans) == length(original_leaf_tokens) "masked_leaf_spans and original_leaf_tokens must have same length"
    @assert size(relation_ids, 1) == size(graph_sequence, 1) "relation_ids batch dimension must match graph_sequence"
    new(
      graph_sequence,
      attention_mask,
      masked_leaf_spans,
      original_leaf_tokens,
      relation_ids,
    )
  end
end
