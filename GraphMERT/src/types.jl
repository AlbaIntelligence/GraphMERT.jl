"""
Core type definitions for GraphMERT.jl

This module defines the fundamental data structures used throughout
the GraphMERT implementation.
"""

using Dates
using SparseArrays
using DocStringExtensions

# ============================================================================
# New Types for GraphMERT Enhancement
# ============================================================================

# ============================================================================
# Text Position
# ============================================================================

"""
    TextPosition

Represents the position of text in a document.



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

Knowledge graph containing entities and relations extracted from text.


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
# Generic Knowledge Graph Entities
# ============================================================================

"""
    Entity

Represents a generic knowledge graph entity with domain-specific attributes.


"""
struct Entity
  id::String
  text::String
  label::String
  entity_type::String  # Domain-specific entity type (e.g., "DISEASE", "PERSON", "CONCEPT")
  attributes::Dict{String,Any}  # Domain-specific attributes (cui, semantic_types, etc.)
  position::TextPosition
  confidence::Float64
  provenance::String  # Source text or document

  function Entity(
    id::String,
    text::String,
    label::String,
    entity_type::String = "UNKNOWN",
    attributes::Dict{String,Any} = Dict{String,Any}(),
    position::TextPosition = TextPosition(0, 0, 0, 0),
    confidence::Float64 = 0.5,
    provenance::String = "",
  )
    @assert !isempty(id) "Entity ID cannot be empty"
    @assert !isempty(text) "Entity text cannot be empty"
    @assert 0.0 <= confidence <= 1.0 "Confidence must be between 0.0 and 1.0"
    new(id, text, label, entity_type, attributes, position, confidence, provenance)
  end
end

"""
    Relation

Represents a generic relation between entities.


"""
struct Relation
  id::String
  head::String  # Head entity ID
  tail::String  # Tail entity ID
  relation_type::String
  confidence::Float64
  provenance::String  # Source text or document
  evidence::String  # Supporting evidence text
  attributes::Dict{String,Any}  # Domain-specific relation attributes

  function Relation(
    head::String,
    tail::String,
    relation_type::String,
    confidence::Float64,
    provenance::String = "",
    evidence::String = "",
    attributes::Dict{String,Any} = Dict{String,Any}(),
    id::String = "",
  )
    @assert !isempty(head) "Head entity ID cannot be empty"
    @assert !isempty(tail) "Tail entity ID cannot be empty"
    @assert !isempty(relation_type) "Relation type cannot be empty"
    @assert 0.0 <= confidence <= 1.0 "Confidence must be between 0.0 and 1.0"

    if isempty(id)
      id = "$(head)_$(relation_type)_$(tail)"
    end

    new(id, head, tail, relation_type, confidence, provenance, evidence, attributes)
  end
end

# ============================================================================
# Domain-Specific Entity Specializations
# ============================================================================

"""
    BiomedicalEntity

Specialization of Entity for biomedical domain with UMLS mappings.


"""
struct BiomedicalEntity
  entity::Entity  # Base generic entity

  function BiomedicalEntity(
    id::String,
    text::String,
    label::String,
    cui::Union{String,Nothing} = nothing,
    semantic_types::Vector{String} = String[],
    position::TextPosition = TextPosition(0, 0, 0, 0),
    confidence::Float64 = 0.5,
    provenance::String = "",
  )
    # Create attributes dict with biomedical-specific fields
    attributes = Dict{String,Any}()
    if cui !== nothing
      attributes["cui"] = cui
    end
    if !isempty(semantic_types)
      attributes["semantic_types"] = semantic_types
    end

    entity = Entity(id, text, label, label, attributes, position, confidence, provenance)
    new(entity)
  end
end

# Convenience accessors for biomedical entities
Base.getproperty(be::BiomedicalEntity, prop::Symbol) = begin
  if prop === :id
    be.entity.id
  elseif prop === :text
    be.entity.text
  elseif prop === :label
    be.entity.label
  elseif prop === :cui
    get(be.entity.attributes, "cui", nothing)
  elseif prop === :semantic_types
    get(be.entity.attributes, "semantic_types", String[])
  elseif prop === :position
    be.entity.position
  elseif prop === :confidence
    be.entity.confidence
  elseif prop === :provenance
    be.entity.provenance
  else
    getfield(be, prop)
  end
end

"""
    BiomedicalRelation

Specialization of Relation for biomedical domain.


"""
struct BiomedicalRelation
  relation::Relation  # Base generic relation

  function BiomedicalRelation(
    head::String,
    tail::String,
    relation_type::String,
    confidence::Float64,
    provenance::String = "",
    evidence::String = "",
    id::String = "",
  )
    relation = Relation(head, tail, relation_type, confidence, provenance, evidence, Dict{String,Any}(), id)
    new(relation)
  end
end

# Convenience accessors for biomedical relations
Base.getproperty(br::BiomedicalRelation, prop::Symbol) = begin
  if prop === :id
    br.relation.id
  elseif prop === :head
    br.relation.head
  elseif prop === :tail
    br.relation.tail
  elseif prop === :relation_type
    br.relation.relation_type
  elseif prop === :confidence
    br.relation.confidence
  elseif prop === :provenance
    br.relation.provenance
  elseif prop === :evidence
    br.relation.evidence
  else
    getfield(br, prop)
  end
end

# ============================================================================
# Knowledge Graph
# ============================================================================

# ============================================================================
# Domain-Specific Entity Specializations
# ============================================================================

# ============================================================================
# Entity Type System
# ============================================================================

struct EntityTypeRegistry
  types::Dict{String, Dict{String, Any}}

  function EntityTypeRegistry()
    new(Dict{String, Dict{String, Any}}())
  end
end

"""
    register_entity_type!(registry::EntityTypeRegistry, type_name::String, attributes::Dict{String, Any})

Register a new entity type in the registry.
"""
function register_entity_type!(registry::EntityTypeRegistry, type_name::String, attributes::Dict{String, Any} = Dict{String, Any}())
  registry.types[type_name] = attributes
end

"""
    get_entity_type_info(registry::EntityTypeRegistry, type_name::String)

Get information about an entity type.
"""
function get_entity_type_info(registry::EntityTypeRegistry, type_name::String)
  get(registry.types, type_name, Dict{String, Any}())
end

# Global registry for entity types
const ENTITY_TYPE_REGISTRY = EntityTypeRegistry()

"""
    EntityType

Abstract type for entity type representations.
"""
abstract type EntityType end

"""
    GenericEntityType <: EntityType

Generic entity type that can represent any domain-specific type.
"""
struct GenericEntityType <: EntityType
  name::String
  attributes::Dict{String, Any}
end

# ============================================================================
# Model Configuration
# ============================================================================

"""
    GraphMERTConfig

Configuration for GraphMERT model.


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
  # Domain configuration
  entity_types::Vector{String}
  relation_types::Vector{String}
  domain::String

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
    # Domain configuration
    entity_types::Vector{String}=String[],
    relation_types::Vector{String}=String[],
    domain::String="general",
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
      entity_types,
      relation_types,
      domain,
    )
  end
end

"""
    ProcessingOptions

Options for text processing and knowledge graph extraction.


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


# ============================================================================
# Evaluation Metrics
# ============================================================================

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
    incorrect_facts::Int,
  )
    new(score, precision, recall, f1, total_facts, correct_facts, incorrect_facts)
  end
end

"""
    ValidityScore

ValidityScore evaluation metric.


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

Represents a single node in the leafy chain graph.


"""
struct ChainGraphNode
    # Node identification
    id::Int                      # Global node ID (0-1023)
    node_type::Symbol            # :root or :leaf

    # Position in structure
    root_index::Int              # Which root this belongs to (0-127)
    leaf_index::Union{Int,Nothing}  # If leaf, which position (0-6); nothing if root

    # Token information
    token_id::Int                # Vocabulary token ID
    token_text::String           # Original text (for debugging)
    is_padding::Bool             # Whether this is a <pad> token

    # Semantic information (only for leaves)
    relation::Union{Symbol,Nothing}  # e.g., :isa, :associated_with
    head_text::Union{String,Nothing} # Text of the head entity

    # Embedding info (filled during forward pass)
    embedding::Union{Vector{Float32},Nothing}

    function ChainGraphNode(;
        id::Int,
        node_type::Symbol,
        root_index::Int,
        leaf_index::Union{Int,Nothing} = nothing,
        token_id::Int,
        token_text::String,
        is_padding::Bool = false,
        relation::Union{Symbol,Nothing} = nothing,
        head_text::Union{String,Nothing} = nothing,
        embedding::Union{Vector{Float32},Nothing} = nothing,
    )
        new(id, node_type, root_index, leaf_index, token_id, token_text, is_padding, relation, head_text, embedding)
    end
end

"""
    ChainGraphConfig

Configuration for leafy chain graph construction.


"""
struct ChainGraphConfig
    num_roots::Int                    # Fixed: 128
    num_leaves_per_root::Int          # Fixed: 7
    max_sequence_length::Int          # Fixed: 1024
    pad_token_id::Int                 # Usually 0 or 1
    vocab_size::Int                   # e.g., 30522 for BioMedBERT

    function ChainGraphConfig(;
        num_roots::Int = 128,
        num_leaves_per_root::Int = 7,
        max_sequence_length::Int = 1024,
        pad_token_id::Int = 0,
        vocab_size::Int = 30522,
    )
        new(num_roots, num_leaves_per_root, max_sequence_length, pad_token_id, vocab_size)
    end
end

# Default configuration matching paper
function default_chain_graph_config()
    return ChainGraphConfig(
        num_roots = 128,
        num_leaves_per_root = 7,
        max_sequence_length = 1024,
        pad_token_id = 0,
        vocab_size = 30522
    )
end

"""
    LeafyChainGraph

Complete leafy chain graph structure for GraphMERT.


"""
mutable struct LeafyChainGraph
    # Graph structure
    nodes::Vector{ChainGraphNode}     # All 1024 nodes
    adjacency_matrix::SparseMatrixCSC{Float32}  # 1024×1024 adjacency
    shortest_paths::Matrix{Int}       # 1024×1024 shortest path distances

    # Root information (syntactic space)
    root_tokens::Vector{Int}          # 128 token IDs
    root_texts::Vector{String}        # Original text tokens

    # Leaf information (semantic space)
    leaf_tokens::Matrix{Int}          # 128×7 matrix of token IDs
    leaf_relations::Matrix{Union{Symbol,Nothing}}  # 128×7 relations
    injected_mask::Matrix{Bool}       # 128×7 which leaves have injections

    # Metadata
    source_sequence_id::String        # Identifier for source text
    sequence_length::Int              # Original text length (≤128)
    num_injections::Int               # Count of non-padding leaves

    # Configuration
    config::ChainGraphConfig

    function LeafyChainGraph(;
        nodes::Vector{ChainGraphNode},
        adjacency_matrix::SparseMatrixCSC{Float32},
        shortest_paths::Matrix{Int},
        root_tokens::Vector{Int},
        root_texts::Vector{String},
        leaf_tokens::Matrix{Int},
        leaf_relations::Matrix{Union{Symbol,Nothing}},
        injected_mask::Matrix{Bool},
        source_sequence_id::String = "",
        sequence_length::Int,
        num_injections::Int,
        config::ChainGraphConfig,
    )
        new(nodes, adjacency_matrix, shortest_paths, root_tokens, root_texts,
            leaf_tokens, leaf_relations, injected_mask, source_sequence_id,
            sequence_length, num_injections, config)
    end
end

"""
    MNMConfig

Configuration for Masked Node Modeling training objective.


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
    default_mnm_config()

Create default MNM configuration for GraphMERT training.

$(TYPEDSIGNATURES)
"""
function default_mnm_config()
    return MNMConfig(
        vocab_size = 30522,      # BioMedBERT vocabulary
        hidden_size = 512,       # Hidden dimension
        num_leaves = 7,          # Number of leaves per root
        mask_probability = 0.15, # Masking probability
        relation_dropout = 0.3,  # Relation dropout rate
        loss_weight = 1.0,       # Loss weighting factor
        mask_entire_leaf_span = true, # Always mask entire spans
        mask_token_id = 103,     # [MASK] token ID
    )
end

"""
    SemanticTriple

Represents a knowledge graph triple for seed injection.


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
