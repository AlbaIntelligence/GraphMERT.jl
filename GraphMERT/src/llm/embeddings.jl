"""
LLM-based text embeddings integration.

This module provides an abstraction for fetching text embeddings from various providers
(e.g., Google Gemini, OpenAI) and computing similarities.
"""

using HTTP
using JSON3
using LinearAlgebra
using Statistics
using Random

"""
    AbstractEmbeddingClient

Abstract base type for embedding clients.
"""
abstract type AbstractEmbeddingClient end

"""
    embed(client::AbstractEmbeddingClient, text::String)::Vector{Float32}

Compute the embedding vector for a given text.
"""
function embed(client::AbstractEmbeddingClient, text::String)::Vector{Float32}
    error("embed not implemented for $(typeof(client))")
end

"""
    embed_batch(client::AbstractEmbeddingClient, texts::Vector{String})::Vector{Vector{Float32}}

Compute embeddings for a batch of texts.
"""
function embed_batch(client::AbstractEmbeddingClient, texts::Vector{String})::Vector{Vector{Float32}}
    # Default fallback: sequential calls
    return [embed(client, text) for text in texts]
end

"""
    cosine_similarity(a::AbstractVector, b::AbstractVector)::Float32

Compute cosine similarity between two vectors.
"""
function cosine_similarity(a::AbstractVector, b::AbstractVector)::Float32
    dot_prod = dot(a, b)
    norm_a = norm(a)
    norm_b = norm(b)
    
    if norm_a == 0 || norm_b == 0
        return 0.0f0
    end
    
    return Float32(dot_prod / (norm_a * norm_b))
end

# ============================================================================
# Gemini Embedding Client
# ============================================================================

"""
    GeminiEmbeddingClient <: AbstractEmbeddingClient

Google Gemini text embedding client.
"""
struct GeminiEmbeddingClient <: AbstractEmbeddingClient
    api_key::String
    model::String
    request_fn::Function
end

"""
    create_gemini_embedding_client(api_key::String; model::String="embedding-001", request_fn::Function=HTTP.post)

Create a Gemini embedding client.
"""
function create_gemini_embedding_client(
    api_key::String;
    model::String = "embedding-001",
    request_fn::Function = HTTP.post,
)
    return GeminiEmbeddingClient(api_key, model, request_fn)
end

function embed(client::GeminiEmbeddingClient, text::String)::Vector{Float32}
    # https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key=API_KEY
    url = "https://generativelanguage.googleapis.com/v1beta/models/$(client.model):embedContent?key=$(client.api_key)"
    
    body = Dict(
        "content" => Dict("parts" => [Dict("text" => text)])
    )
    
    try
        response = client.request_fn(url, ["Content-Type" => "application/json"], JSON3.write(body))
        
        if response.status == 200
            data = JSON3.read(String(response.body))
            if haskey(data, "embedding") && haskey(data["embedding"], "values")
                return Float32.(data["embedding"]["values"])
            end
        end
        @warn "Gemini embedding failed: Empty or invalid response" status=response.status
        return Float32[]
    catch e
        @warn "Gemini embedding request failed: $e"
        return Float32[]
    end
end

# ============================================================================
# Mock Embedding Client
# ============================================================================

"""
    MockEmbeddingClient <: AbstractEmbeddingClient

Mock embedding client returning random or deterministic vectors.
"""
struct MockEmbeddingClient <: AbstractEmbeddingClient
    dimension::Int
    deterministic::Bool
    seed_offset::Int
end

"""
    create_mock_embedding_client(dimension::Int=768; deterministic::Bool=true)

Create a mock embedding client.
"""
function create_mock_embedding_client(dimension::Int = 768; deterministic::Bool = true)
    return MockEmbeddingClient(dimension, deterministic, 42)
end

function embed(client::MockEmbeddingClient, text::String)::Vector{Float32}
    if client.deterministic
        # Deterministic but pseudo-random based on text hash
        # Use simple seeded RNG
        seed = hash(text) + client.seed_offset
        rng = Random.MersenneTwister(seed)
        vec = rand(rng, Float32, client.dimension)
        # Normalize
        return vec ./ norm(vec)
    else
        vec = rand(Float32, client.dimension)
        return vec ./ norm(vec)
    end
end
