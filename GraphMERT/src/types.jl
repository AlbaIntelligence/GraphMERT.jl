"""
Core data structures for GraphMERT.jl

This module defines the fundamental data types used throughout the GraphMERT
implementation, including knowledge graphs, entities, relations, and metadata.
"""

using Dates
using LinearAlgebra
using SparseArrays

# ============================================================================
# Core Entity Types
# ============================================================================

"""
    EntityPosition

Represents the position of an entity within the original text.
"""
struct EntityPosition
  start::Int
  stop::Int
  line::Int
  column::Int
end

"""
    BiomedicalEntity

Represents an extracted biomedical entity with its properties and confidence.
"""
struct BiomedicalEntity
  id::String
  text::String
  label::String
  confidence::Float64
  position::EntityPosition
  attributes::Dict{String,Any}
  created_at::DateTime

  function BiomedicalEntity(id::String, text::String, label::String,
    confidence::Float64, position::EntityPosition,
    attributes::Dict{String,Any}=Dict{String,Any}())
    @assert 0.0 <= confidence <= 1.0 "Confidence must be between 0.0 and 1.0"
    @assert !isempty(id) "Entity ID cannot be empty"
    @assert !isempty(text) "Entity text cannot be empty"
    @assert !isempty(label) "Entity label cannot be empty"

    new(id, text, label, confidence, position, attributes, now())
  end
end

"""
    BiomedicalRelation

Represents an extracted biomedical relation between two entities.
"""
struct BiomedicalRelation
  head::String
  tail::String
  relation_type::String
  confidence::Float64
  attributes::Dict{String,Any}
  created_at::DateTime

  function BiomedicalRelation(head::String, tail::String, relation_type::String,
    confidence::Float64, attributes::Dict{String,Any}=Dict{String,Any}())
    @assert 0.0 <= confidence <= 1.0 "Confidence must be between 0.0 and 1.0"
    @assert !isempty(head) "Head entity cannot be empty"
    @assert !isempty(tail) "Tail entity cannot be empty"
    @assert !isempty(relation_type) "Relation type cannot be empty"

    new(head, tail, relation_type, confidence, attributes, now())
  end
end

# ============================================================================
# Graph Metadata
# ============================================================================

"""
    GraphMetadata

Contains metadata and statistics about a knowledge graph.
"""
struct GraphMetadata
  total_entities::Int
  total_relations::Int
  entity_types::Dict{String,Int}
  relation_types::Dict{String,Int}
  average_confidence::Float64
  processing_time::Float64
  model_version::String
  created_at::DateTime

  function GraphMetadata(entities::Vector{BiomedicalEntity},
    relations::Vector{BiomedicalRelation},
    processing_time::Float64, model_version::String)
    entity_types = Dict{String,Int}()
    relation_types = Dict{String,Int}()

    for entity in entities
      entity_types[entity.label] = get(entity_types, entity.label, 0) + 1
    end

    for relation in relations
      relation_types[relation.relation_type] = get(relation_types, relation.relation_type, 0) + 1
    end

    avg_conf = if !isempty(entities) && !isempty(relations)
      (sum(e.confidence for e in entities) + sum(r.confidence for r in relations)) /
      (length(entities) + length(relations))
    else
      0.0
    end

    new(length(entities), length(relations), entity_types, relation_types,
      avg_conf, processing_time, model_version, now())
  end
end

"""
    GraphMERTModelInfo

Information about the GraphMERT model used for processing.
"""
struct GraphMERTModelInfo
  model_name::String
  model_version::String
  architecture::String
  parameters::Int
  training_data::String
  created_at::DateTime

  function GraphMERTModelInfo(model_name::String, model_version::String,
    architecture::String, parameters::Int, training_data::String)
    new(model_name, model_version, architecture, parameters, training_data, now())
  end
end

# ============================================================================
# Knowledge Graph
# ============================================================================

"""
    KnowledgeGraph

Main output structure containing the complete knowledge graph.
"""
struct KnowledgeGraph
  entities::Vector{BiomedicalEntity}
  relations::Vector{BiomedicalRelation}
  metadata::GraphMetadata
  confidence_threshold::Float64
  created_at::DateTime
  model_info::GraphMERTModelInfo
  umls_mappings::Dict{String,String}
  fact_score::Float64
  validity_score::Float64

  function KnowledgeGraph(entities::Vector{BiomedicalEntity},
    relations::Vector{BiomedicalRelation},
    confidence_threshold::Float64,
    model_info::GraphMERTModelInfo,
    umls_mappings::Dict{String,String}=Dict{String,String}(),
    fact_score::Float64=0.0,
    validity_score::Float64=0.0)
    @assert 0.0 <= confidence_threshold <= 1.0 "Confidence threshold must be between 0.0 and 1.0"
    @assert 0.0 <= fact_score <= 1.0 "FActScore must be between 0.0 and 1.0"
    @assert 0.0 <= validity_score <= 1.0 "ValidityScore must be between 0.0 and 1.0"

    # Validate that all relations reference valid entities
    entity_ids = Set(e.id for e in entities)
    for relation in relations
      @assert relation.head in entity_ids "Head entity $(relation.head) not found in entities"
      @assert relation.tail in entity_ids "Tail entity $(relation.tail) not found in entities"
    end

    processing_time = 0.0  # Will be set during processing
    metadata = GraphMetadata(entities, relations, processing_time, model_info.model_version)

    new(entities, relations, metadata, confidence_threshold, now(),
      model_info, umls_mappings, fact_score, validity_score)
  end
end

# ============================================================================
# GraphMERT Architecture Types
# ============================================================================

"""
    LeafyChainGraph

Represents a leafy chain graph structure for text representation.
"""
struct LeafyChainGraph
  root_nodes::Vector{Int}
  leaf_nodes::Vector{Int}
  edges::Vector{Tuple{Int,Int}}
  node_features::Dict{Int,Vector{Float64}}
  edge_weights::Dict{Tuple{Int,Int},Float64}

  function LeafyChainGraph(root_nodes::Vector{Int}, leaf_nodes::Vector{Int},
    edges::Vector{Tuple{Int,Int}},
    node_features::Dict{Int,Vector{Float64}}=Dict{Int,Vector{Float64}}(),
    edge_weights::Dict{Tuple{Int,Int},Float64}=Dict{Tuple{Int,Int},Float64}())
    new(root_nodes, leaf_nodes, edges, node_features, edge_weights)
  end
end

"""
    H_GAT

Hierarchical Graph Attention Network component.
"""
struct H_GAT
  num_heads::Int
  hidden_dim::Int
  attention_weights::Matrix{Float64}
  layer_norm::Bool

  function H_GAT(num_heads::Int, hidden_dim::Int, attention_weights::Matrix{Float64}, layer_norm::Bool=true)
    @assert num_heads > 0 "Number of attention heads must be positive"
    @assert hidden_dim > 0 "Hidden dimension must be positive"
    @assert size(attention_weights, 1) == num_heads "Attention weights must match number of heads"

    new(num_heads, hidden_dim, attention_weights, layer_norm)
  end
end

"""
    SeedKG

Seed knowledge graph for training data preparation.
"""
struct SeedKG
  entities::Vector{BiomedicalEntity}
  relations::Vector{BiomedicalRelation}
  source::String
  confidence::Float64

  function SeedKG(entities::Vector{BiomedicalEntity}, relations::Vector{BiomedicalRelation},
    source::String, confidence::Float64)
    @assert 0.0 <= confidence <= 1.0 "Confidence must be between 0.0 and 1.0"
    new(entities, relations, source, confidence)
  end
end

"""
    UMLSIntegration

UMLS integration configuration and mappings.
"""
struct UMLSIntegration
  enabled::Bool
  api_key::String
  mappings::Dict{String,String}
  confidence_threshold::Float64

  function UMLSIntegration(enabled::Bool, api_key::String="",
    mappings::Dict{String,String}=Dict{String,String}(),
    confidence_threshold::Float64=0.8)
    @assert 0.0 <= confidence_threshold <= 1.0 "Confidence threshold must be between 0.0 and 1.0"
    new(enabled, api_key, mappings, confidence_threshold)
  end
end

"""
    MLM_MNM_Training

Configuration for MLM and MNM training objectives.
"""
struct MLM_MNM_Training
  mlm_probability::Float64
  mnm_probability::Float64
  span_length::Int
  boundary_loss_weight::Float64

  function MLM_MNM_Training(mlm_probability::Float64=0.15, mnm_probability::Float64=0.15,
    span_length::Int=3, boundary_loss_weight::Float64=1.0)
    @assert 0.0 <= mlm_probability <= 1.0 "MLM probability must be between 0.0 and 1.0"
    @assert 0.0 <= mnm_probability <= 1.0 "MNM probability must be between 0.0 and 1.0"
    @assert span_length > 0 "Span length must be positive"
    @assert boundary_loss_weight >= 0.0 "Boundary loss weight must be non-negative"

    new(mlm_probability, mnm_probability, span_length, boundary_loss_weight)
  end
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    validate_entity(entity::BiomedicalEntity)

Validate that an entity meets GraphMERT requirements.
"""
function validate_entity(entity::BiomedicalEntity)
  return !isempty(entity.id) &&
         !isempty(entity.text) &&
         !isempty(entity.label) &&
         0.0 <= entity.confidence <= 1.0
end

"""
    validate_relation(relation::BiomedicalRelation)

Validate that a relation meets GraphMERT requirements.
"""
function validate_relation(relation::BiomedicalRelation)
  return !isempty(relation.head) &&
         !isempty(relation.tail) &&
         !isempty(relation.relation_type) &&
         0.0 <= relation.confidence <= 1.0
end

"""
    validate_knowledge_graph(graph::KnowledgeGraph)

Validate that a knowledge graph meets GraphMERT requirements.
"""
function validate_knowledge_graph(graph::KnowledgeGraph)
  # Check basic structure
  if isempty(graph.entities)
    return false
  end

  # Check confidence threshold
  if !(0.0 <= graph.confidence_threshold <= 1.0)
    return false
  end

  # Check that all relations reference valid entities
  entity_ids = Set(e.id for e in graph.entities)
  for relation in graph.relations
    if !(relation.head in entity_ids) || !(relation.tail in entity_ids)
      return false
    end
  end

  return true
end

"""
    get_entity_by_id(graph::KnowledgeGraph, id::String)

Get an entity by its ID.
"""
function get_entity_by_id(graph::KnowledgeGraph, id::String)
  for entity in graph.entities
    if entity.id == id
      return entity
    end
  end
  return nothing
end

"""
    get_relations_by_entity(graph::KnowledgeGraph, entity_id::String)

Get all relations involving a specific entity.
"""
function get_relations_by_entity(graph::KnowledgeGraph, entity_id::String)
  relations = Vector{BiomedicalRelation}()
  for relation in graph.relations
    if relation.head == entity_id || relation.tail == entity_id
      push!(relations, relation)
    end
  end
  return relations
end

"""
    filter_by_confidence(graph::KnowledgeGraph, threshold::Float64)

Filter graph elements by confidence threshold.
"""
function filter_by_confidence(graph::KnowledgeGraph, threshold::Float64)
  filtered_entities = filter(e -> e.confidence >= threshold, graph.entities)
  filtered_relations = filter(r -> r.confidence >= threshold, graph.relations)

  return KnowledgeGraph(filtered_entities, filtered_relations, threshold,
    graph.model_info, graph.umls_mappings,
    graph.fact_score, graph.validity_score)
end
