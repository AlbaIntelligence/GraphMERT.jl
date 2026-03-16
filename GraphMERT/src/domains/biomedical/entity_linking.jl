"""
Entity Linking for Biomedical Domain using SapBERT and other methods.

This module provides an abstraction for entity linking, primarily focusing on
mapping text mentions to UMLS CUIs using SapBERT embeddings and approximate nearest neighbor search.
"""

using LinearAlgebra
using Statistics

"""
    AbstractEntityLinker

Abstract base type for entity linkers.
"""
abstract type AbstractEntityLinker end

"""
    link_entity(linker::AbstractEntityLinker, text::String; top_k::Int=1)

Link an entity mention to concepts (CUIs).
Returns a list of tuples: `(cui::String, score::Float64, name::String)`.
"""
function link_entity(linker::AbstractEntityLinker, text::String; top_k::Int=1)
    error("link_entity not implemented for $(typeof(linker))")
end

# ============================================================================
# Mock Entity Linker
# ============================================================================

"""
    MockEntityLinker <: AbstractEntityLinker

Mock linker returning deterministic or random results for testing.
"""
struct MockEntityLinker <: AbstractEntityLinker
    mappings::Dict{String, String} # text -> CUI
    
    function MockEntityLinker()
        mappings = Dict(
            "diabetes" => "C0011849",
            "metformin" => "C0025598",
            "covid-19" => "C5203670",
            "cancer" => "C0006826"
        )
        new(mappings)
    end
end

function link_entity(linker::MockEntityLinker, text::String; top_k::Int=1)
    lowercase_text = lowercase(text)
    
    # Check exact match in mappings
    if haskey(linker.mappings, lowercase_text)
        cui = linker.mappings[lowercase_text]
        return [(cui, 1.0, text)]
    end
    
    # Fallback: deterministic CUI generation
    # Using hash to generate a stable CUI-like string
    h = hash(lowercase_text) % 1000000
    cui = "C" * lpad(string(abs(h)), 7, '0')
    
    return [(cui, 0.8, text)]
end

# ============================================================================
# SapBERT Entity Linker (Stub)
# ============================================================================

"""
    SapBERTLinker <: AbstractEntityLinker

Entity linker using SapBERT embeddings and vector search.
Requires a running SapBERT model service or loaded ONNX model.
"""
struct SapBERTLinker <: AbstractEntityLinker
    model_path::String
    index_path::String
    # In a real implementation, this would hold the model and index objects
    
    function SapBERTLinker(model_path::String, index_path::String)
        # Verify paths existence if we were loading real models
        # if !isfile(model_path) || !isfile(index_path)
        #     error("SapBERT model or index not found")
        # end
        new(model_path, index_path)
    end
end

function link_entity(linker::SapBERTLinker, text::String; top_k::Int=1)
    # TODO: Implement real SapBERT inference and ANN search
    # For now, this is a placeholder that warns and returns empty
    @warn "SapBERT linking not fully implemented yet. Using mock behavior."
    
    # Mock behavior for now to unblock
    h = hash(lowercase(text)) % 1000000
    cui = "C" * lpad(string(abs(h)), 7, '0')
    return [(cui, 0.5, text)]
end
