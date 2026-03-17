"""
Entity Linking for Biomedical Domain using SapBERT and other methods.

This module provides an abstraction for entity linking, primarily focusing on
mapping text mentions to UMLS CUIs using SapBERT embeddings and approximate nearest neighbor search.
"""

using LinearAlgebra
using Statistics
using Random

"""
    calculate_character_3gram_jaccard(s1::String, s2::String)

Calculate the Jaccard similarity between character 3-grams of two strings.
Returns a score between 0.0 and 1.0.
"""
function calculate_character_3gram_jaccard(s1::String, s2::String)
    # Normalize strings
    s1 = lowercase(strip(s1))
    s2 = lowercase(strip(s2))
    
    if length(s1) < 3 || length(s2) < 3
        # Fallback for very short strings: simple character overlap or equality
        return s1 == s2 ? 1.0 : 0.0
    end
    
    # Generate 3-grams
    ngrams1 = Set{String}()
    for i in 1:(length(s1)-2)
        push!(ngrams1, s1[i:i+2])
    end
    
    ngrams2 = Set{String}()
    for i in 1:(length(s2)-2)
        push!(ngrams2, s2[i:i+2])
    end
    
    # Calculate Jaccard: |Intersection| / |Union|
    intersection_size = length(intersect(ngrams1, ngrams2))
    union_size = length(union(ngrams1, ngrams2))
    
    return union_size > 0 ? intersection_size / union_size : 0.0
end

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

Entity linker using SapBERT embeddings and vector search, with optional Jaccard reranking.
Requires a running SapBERT model service or loaded ONNX model.
"""
struct SapBERTLinker <: AbstractEntityLinker
    model_path::String
    index_path::String
    reranking_weight::Float64
    top_n_candidates::Int
    
    # In-memory index for small-scale or fallback use
    # If loaded from index_path, this would be populated
    embeddings::Union{Matrix{Float32}, Nothing} # (dim, num_entities)
    cui_list::Union{Vector{String}, Nothing}    # (num_entities,)
    name_list::Union{Vector{String}, Nothing}   # (num_entities,) - Optional
    
    function SapBERTLinker(model_path::String, index_path::String; 
                          reranking_weight::Float64=0.3, top_n_candidates::Int=20,
                          embeddings::Union{Matrix{Float32}, Nothing}=nothing,
                          cui_list::Union{Vector{String}, Nothing}=nothing,
                          name_list::Union{Vector{String}, Nothing}=nothing)
        # Verify paths existence if we were loading real models
        # if !isfile(model_path) || !isfile(index_path)
        #     error("SapBERT model or index not found")
        # end
        new(model_path, index_path, reranking_weight, top_n_candidates, embeddings, cui_list, name_list)
    end
end

"""
    encode_text(linker::SapBERTLinker, text::String)

Encode text into an embedding vector.
Currently a stub/mock until ONNX model integration.
"""
function encode_text(linker::SapBERTLinker, text::String)
    # TODO: Implement real SapBERT encoding via ONNX
    # For now, return a random vector (deterministic based on text hash)
    rng = MersenneTwister(hash(text))
    dim = linker.embeddings !== nothing ? size(linker.embeddings, 1) : 768
    v = rand(rng, Float32, dim)
    return v ./ norm(v) # Normalize
end

"""
    search_candidates(linker::SapBERTLinker, query_vec::Vector{Float32}; top_k::Int=20)

Search for nearest neighbors in the embedding index using cosine similarity.
Returns list of (cui, score, name).
"""
function search_candidates(linker::SapBERTLinker, query_vec::Vector{Float32}; top_k::Int=20)
    if linker.embeddings === nothing || linker.cui_list === nothing
        return Vector{Tuple{String, Float64, String}}()
    end
    
    # Compute cosine similarity: query · embeddings (assuming normalized)
    # query_vec is (dim,), embeddings is (dim, N)
    scores = query_vec' * linker.embeddings # (1, N)
    scores = vec(scores) # (N,)
    
    # Get top-k indices
    # TODO: Use partialsortperm for efficiency
    perm = sortperm(scores, rev=true)
    top_indices = perm[1:min(top_k, length(perm))]
    
    results = Vector{Tuple{String, Float64, String}}()
    for idx in top_indices
        cui = linker.cui_list[idx]
        score = scores[idx]
        name = linker.name_list !== nothing ? linker.name_list[idx] : "Unknown"
        push!(results, (cui, score, name))
    end
    
    return results
end

function link_entity(linker::SapBERTLinker, text::String; top_k::Int=1)
    # 1. Retrieve candidates
    candidates = Vector{Tuple{String, Float64, String}}()
    
    if linker.embeddings !== nothing && linker.cui_list !== nothing
        # Use simple in-memory index search
        query_vec = encode_text(linker, text)
        candidates = search_candidates(linker, query_vec, top_k=linker.top_n_candidates)
    else
        # Fallback: Mock candidates (simulate retrieval)
        # Candidate 1: Main match (mocked via hash to be deterministic)
        h = hash(lowercase(text)) % 1000000
        main_cui = "C" * lpad(string(abs(h)), 7, '0')
        push!(candidates, (main_cui, 0.85, text)) 
        
        # Candidate 2: Similar string but different concept
        push!(candidates, ("C" * lpad(string(abs(h)+1), 7, '0'), 0.75, text * " type 2"))
        
        # Candidate 3: Distant string match
        push!(candidates, ("C" * lpad(string(abs(h)+2), 7, '0'), 0.70, "metabolic syndrome"))
    end

    # 2. Rerank with Character 3-gram Jaccard
    reranked_candidates = Vector{Tuple{String, Float64, String}}()
    
    for (cui, bert_score, candidate_name) in candidates
        jaccard_score = calculate_character_3gram_jaccard(text, candidate_name)
        
        # Combined score: (1 - α) * BERT + α * Jaccard
        alpha = linker.reranking_weight
        final_score = (1 - alpha) * bert_score + alpha * jaccard_score
        
        push!(reranked_candidates, (cui, final_score, candidate_name))
    end
    
    # 3. Sort and select top-k
    sort!(reranked_candidates, by = x -> x[2], rev = true)
    
    return reranked_candidates[1:min(top_k, length(reranked_candidates))]
end
