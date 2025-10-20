"""
Core type definitions for GraphMERT.jl

This module defines the fundamental data structures used throughout
the GraphMERT implementation.
"""

using Dates

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
# Biomedical Entities
# ============================================================================

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

    function BiomedicalEntity(id::String, text::String, label::String, 
                            confidence::Float64, position::TextPosition,
                            attributes::Dict{String,Any}=Dict{String,Any}(),
                            created_at::DateTime=now())
        new(id, text, label, confidence, position, attributes, created_at)
    end
end

"""
    BiomedicalRelation

Represents a relation between two biomedical entities.
"""
struct BiomedicalRelation
    head::String
    tail::String
    relation_type::String
    confidence::Float64
    attributes::Dict{String,Any}
    created_at::DateTime

    function BiomedicalRelation(head::String, tail::String, relation_type::String,
                              confidence::Float64, attributes::Dict{String,Any}=Dict{String,Any}(),
                              created_at::DateTime=now())
        new(head, tail, relation_type, confidence, attributes, created_at)
    end
end

# ============================================================================
# Knowledge Graph
# ============================================================================

"""
    KnowledgeGraph

Represents a knowledge graph with entities and relations.
"""
struct KnowledgeGraph
    entities::Vector{BiomedicalEntity}
    relations::Vector{BiomedicalRelation}
    metadata::Dict{String,Any}
    created_at::DateTime

    function KnowledgeGraph(entities::Vector{BiomedicalEntity},
                          relations::Vector{BiomedicalRelation},
                          metadata::Dict{String,Any}=Dict{String,Any}(),
                          created_at::DateTime=now())
        new(entities, relations, metadata, created_at)
    end
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
        return_dict::Bool=true
    )
        new(
            model_path, vocab_size, hidden_size, num_attention_heads, num_hidden_layers,
            max_position_embeddings, type_vocab_size, initializer_range, layer_norm_eps,
            use_cache, pad_token_id, bos_token_id, eos_token_id, max_length,
            temperature, top_k, top_p, repetition_penalty, length_penalty,
            output_attentions, output_hidden_states, return_dict
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
        verbose::Bool=false
    )
        new(max_length, batch_size, use_umls, use_helper_llm, confidence_threshold,
            entity_types, relation_types, cache_enabled, parallel_processing, verbose)
    end
end

# ============================================================================
# Model Structures
# ============================================================================

"""
    GraphMERTModel

Main GraphMERT model structure.
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
"""
struct FActScore
    score::Float64
    precision::Float64
    recall::Float64
    f1::Float64
    total_facts::Int
    correct_facts::Int
    incorrect_facts::Int

    function FActScore(score::Float64, precision::Float64, recall::Float64, f1::Float64,
                      total_facts::Int, correct_facts::Int, incorrect_facts::Int)
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

    function ValidityScore(score::Float64, valid_relations::Int, total_relations::Int, invalid_relations::Int)
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

    function GraphRAG(score::Float64, retrieval_accuracy::Float64, generation_quality::Float64, overall_performance::Float64)
        new(score, retrieval_accuracy, generation_quality, overall_performance)
    end
end
