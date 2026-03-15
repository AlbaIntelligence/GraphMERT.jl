"""
Local LLM backend for entity discovery and relation matching.

This module provides a local LLM client using LlamaCpp.jl for running
GGUF-format language models locally. It offers the same functionality
as the helper LLM but runs entirely on-device without external API calls.
"""

module LocalLLM

using JSON3
using LlamaCpp

# =============================================================================
# Type Definitions (must come before functions that use them)
# =============================================================================

"""
    LocalLLMConfig(; model_path, context_length=2048, threads=4, temperature=0.7, max_tokens=512, n_gpu_layers=0)

Configuration struct for local LLM inference using GGUF models.

# Arguments
- `model_path::String`: Path to GGUF model file (required)
- `context_length::Int`: Context window size (default: 2048)
- `threads::Int`: CPU threads for inference (default: 4)
- `temperature::Float64`: Sampling temperature (default: 0.7)
- `max_tokens::Int`: Max tokens to generate (default: 512)
- `n_gpu_layers::Int`: GPU layers (default: 0, CPU only)

# Validation
- `model_path` must point to an existing file

# Examples
```julia
config = LocalLLMConfig(model_path="path/to/model.gguf")
config = LocalLLMConfig(
    model_path="path/to/model.gguf",
    context_length=4096,
    threads=8,
    temperature=0.5
)
```
"""
struct LocalLLMConfig
    model_path::String
    context_length::Int
    threads::Int
    temperature::Float64
    max_tokens::Int
    n_gpu_layers::Int

    function LocalLLMConfig(;
        model_path::String,
        context_length::Int=2048,
        threads::Int=4,
        temperature::Float64=0.7,
        max_tokens::Int=512,
        n_gpu_layers::Int=0
    )
        if !isfile(model_path)
            throw(ArgumentError("model_path must be an existing file: $model_path"))
        end
        if context_length <= 0 || context_length > 8192
            throw(ArgumentError("context_length must be between 1 and 8192"))
        end
        if threads < 1
            throw(ArgumentError("threads must be at least 1"))
        end
        if temperature < 0.0 || temperature > 2.0
            throw(ArgumentError("temperature must be between 0.0 and 2.0"))
        end
        if max_tokens < 1 || max_tokens > 4096
            throw(ArgumentError("max_tokens must be between 1 and 4096"))
        end
        if n_gpu_layers < 0
            throw(ArgumentError("n_gpu_layers must be non-negative"))
        end
        new(model_path, context_length, threads, temperature, max_tokens, n_gpu_layers)
    end
end

"""
    LocalModelMetadata

Metadata about available local models. This struct stores information about
GGUF models that can be loaded by the local LLM client, including the display
name, filename, parameter count, quantization level, estimated RAM usage,
and context window size.

# Fields
- `name::String`: Model display name
- `filename::String`: GGUF filename
- `params::Int`: Parameter count
- `quantization::String`: Quantization level (Q4_0, Q5_1, etc.)
- `ram_estimate::Int`: Estimated RAM in MB
- `context_length::Int`: Context window
"""
struct LocalModelMetadata
    name::String
    filename::String
    params::Int
    quantization::String
    ram_estimate::Int
    context_length::Int
end

function LocalModelMetadata(;
    name::String = "",
    filename::String = "",
    params::Int = 0,
    quantization::String = "",
    ram_estimate::Int = 0,
    context_length::Int = 2048
)
    return LocalModelMetadata(name, filename, params, quantization, ram_estimate, context_length)
end

"""
    LocalLLMClient

Client for local LLM inference using LlamaCpp.jl. Provides on-device entity
discovery and relation matching without external API calls.

# Fields
- `config::LocalLLMConfig`: Configuration for the local LLM
- `model`: The LlamaCpp.jl model handle (Any type, set to nothing when unloaded)
- `is_loaded::Bool`: Whether the model is currently loaded in memory
"""
mutable struct LocalLLMClient
    config::LocalLLMConfig
    model::Any
    is_loaded::Bool
end

function LocalLLMClient(config::LocalLLMConfig)
    return LocalLLMClient(config, nothing, false)
end

# =============================================================================
# Core Functions
# =============================================================================

"""
    load_local_model(config::LocalLLMConfig)::LocalLLMClient

Load a GGUF model using LlamaCpp.jl and return a configured LocalLLMClient.

# Arguments
- `config::LocalLLMConfig`: Configuration containing model_path and inference parameters

# Returns
- `LocalLLMClient`: Client with loaded model, ready for inference

# Error Cases
- `SystemError("Model file not found: \\\$path")`: Model file does not exist
- `ArgumentError("Invalid GGUF model file")`: File is not a valid GGUF model
- `OutOfMemoryError("Insufficient memory to load model")`: Not enough memory to load model
"""
function load_local_model(config::LocalLLMConfig)::LocalLLMClient
    path = config.model_path

    if !isfile(path)
        throw(SystemError("Model file not found: $path"))
    end

    try
        # Use LlamaContext from LlamaCpp.jl for model loading
        llm = LlamaCpp.LlamaContext(;
            path=path,
            n_ctx=config.context_length,
            n_threads=config.threads,
            n_gpu_layers=config.n_gpu_layers
        )
        return LocalLLMClient(config, llm, true)
    catch e
        if isa(e, OutOfMemoryError)
            rethrow(OutOfMemoryError("Insufficient memory to load model"))
        elseif isa(e, ArgumentError) && occursin("model", lowercase(sprint(show, e)))
            throw(ArgumentError("Invalid GGUF model file"))
        elseif isa(e, ErrorException) || isa(e, ArgumentError)
            throw(ArgumentError("Invalid GGUF model file"))
        else
            rethrow(e)
        end
    end
end

# =============================================================================
# Entity and Relation Extraction
# =============================================================================

"""
    discover_entities(client::LocalLLMClient, text::String, domain::DomainProvider)::Vector{String}

Discover entities in text using local LlamaCpp model with domain-specific prompts.

# Arguments
- `client::LocalLLMClient`: Local LLM client instance
- `text::String`: Text to extract entities from
- `domain::DomainProvider`: Domain provider for prompt generation

# Returns
- `Vector{String}`: Extracted entity names

# Error Cases
- `ErrorException("Model not loaded")`: Model is not loaded in memory

# Contract Rules
1. Empty text: return empty vector
2. Model not loaded: throw error
3. Return unique entity strings
"""
function discover_entities(client::LocalLLMClient, text::String, domain::Any)::Vector{String}
    if !client.is_loaded || client.model === nothing
        throw(ErrorException("Model not loaded"))
    end

    if isempty(strip(text))
        return String[]
    end

    context = Dict{String, Any}("text" => text, "task_type" => :entity_discovery)
    prompt = try
        GraphMERT.create_prompt(domain, :entity_discovery, context)
    catch e
        @warn "Domain prompt creation failed: $e, using fallback prompt"
        _create_entity_discovery_prompt(text)
    end

    response = _generate(client, prompt)

    if !isempty(response)
        return _parse_entity_response(response)
    else
        return String[]
    end
end

function _create_entity_discovery_prompt(text::String)
    return """
You are an expert tasked with extracting entities from text.

Extract all entities (people, places, organizations, concepts, etc.) from the following text.

Return only the entity names, one per line, in the format:
ENTITY_NAME

Rules:
- Extract proper entities only
- Include people, places, organizations, concepts
- Do not include generic words
- Return entities in their original form as they appear in text
- List each entity on a separate line

Text: $text

Extracted Entities:
"""
end

function _parse_entity_response(response::String)
    entities = String[]
    for line in split(response, '\n')
        line = strip(line)
        if !isempty(line) &&
           !startswith(line, "#") &&
           !startswith(line, "Extracted") &&
           !startswith(line, "Entities")
            clean_line = replace(line, r"^\d+\.?\s*" => "")
            clean_line = replace(clean_line, r"[-*_]" => "")
            clean_line = strip(clean_line)
            if !isempty(clean_line) && length(clean_line) > 2
                push!(entities, clean_line)
            end
        end
    end
    return unique(entities)
end

"""
    match_relations(client::LocalLLMClient, entities::Vector{String}, text::String)::Dict{String, Dict{String, String}}

Match relations between entities using local LlamaCpp model inference.

# Arguments
- `client::LocalLLMClient`: Local LLM client instance
- `entities::Vector{String}`: Entity names to match relations between
- `text::String`: Original text for context

# Returns
- `Dict{String, Dict{String, String}}`: Relations in format entity => (relation => target_entity)

# Error Cases
- `ErrorException("Model not loaded")`: Model is not loaded in memory
- Throws on inference errors (wrapped)

# Contract Rules
- Empty entities: return empty dict
- Model not loaded: throw error
"""
function match_relations(client::LocalLLMClient, entities::Vector{String}, text::String)::Dict{String, Dict{String, String}}
    if isempty(entities)
        return Dict{String, Dict{String, String}}()
    end

    if !client.is_loaded || client.model === nothing
        throw(ErrorException("Model not loaded"))
    end

    prompt = _create_relation_prompt(entities, text)

    try
        response = _generate(client, prompt)
        return _parse_relation_response(response)
    catch e
        @warn "Relation matching failed: $e"
        return Dict{String, Dict{String, String}}()
    end
end

function _create_relation_prompt(entities::Vector{String}, text::String)
    entities_str = join(entities, "\n- ")
    return """
You are an expert tasked with finding relationships between entities.

Given the following entities extracted from text, determine what relationships exist between them based on the text content.

Entities:
- $entities_str

Text: $text

For each pair of entities that are related in the text, return in format:
ENTITY1 -> RELATION -> ENTITY2

Where RELATION describes the relationship type.

Rules:
- Only include relationships that are explicitly or implicitly stated in the text
- Use appropriate relationship terms
- Each relationship should be on a separate line
- If no clear relationship exists, return nothing for that pair

Relationships:
"""
end

function _generate(client::LocalLLMClient, prompt::String)
    stream = LlamaCpp.generate(client.model, prompt)
    full_response = ""
    for token in stream
        full_response *= token
    end
    return full_response
end

"""
    generate(client::LocalLLMClient, prompt::String)::String

Run the local LLM on a prompt and return the full response. Used by domain modules
for relation matching, tail formation, or other custom prompts.
"""
function generate(client::LocalLLMClient, prompt::String)::String
    if !client.is_loaded || client.model === nothing
        throw(ErrorException("Model not loaded"))
    end
    return _generate(client, prompt)
end

function _parse_relation_response(response::String)
    relations = Dict{String,Dict{String,String}}()

    for line in split(response, '\n')
        line = strip(line)
        if !isempty(line) &&
           !startswith(line, "#") &&
           !startswith(line, "Relationships") &&
           occursin("->", line)
            parts = split(line, "->")
            if length(parts) >= 3
                entity1 = strip(strip(parts[1]))
                relation = strip(strip(parts[2]))
                entity2 = strip(strip(join(parts[3:end], "->")))

                relation = replace(relation, r"[^\w\s]" => "")
                relation = uppercase(replace(relation, r"\s+" => "_"))

                if !isempty(entity1) && !isempty(relation) && !isempty(entity2)
                    relations[entity1] = Dict("relation" => relation, "entity2" => entity2)
                end
            end
        end
    end

    return relations
end

"""
    form_tail_from_tokens(client::LocalLLMClient, tokens::Vector{Tuple{Int, Float64}}, text::String)::Vector{String}

Form coherent tail entities from predicted tokens using local LlamaCpp model inference.

# Arguments
- `client::LocalLLMClient`: Local LLM client instance
- `tokens::Vector{Tuple{Int, Float64}}`: Top-k predicted tokens with probabilities
- `text::String`: Original text for context

# Returns
- `Vector{String}`: Formed tail entity names

# Error Cases
- `ErrorException("Model not loaded")`: Model is not loaded in memory

# Contract Rules
- Empty tokens: return empty vector
- Model not loaded: throw error
- Output: unique strings only
"""
function form_tail_from_tokens(client::LocalLLMClient, tokens::Vector{Tuple{Int,Float64}}, text::String)::Vector{String}
    if isempty(tokens)
        return String[]
    end

    if !client.is_loaded || client.model === nothing
        throw(ErrorException("Model not loaded"))
    end

    prompt = _create_tail_formation_prompt(tokens, text)

    try
        response = _generate(client, prompt)
        return _parse_tail_formation_response(response)
    catch e
        @warn "Tail formation failed: $e"
        return String[]
    end
end

function _create_tail_formation_prompt(tokens::Vector{Tuple{Int,Float64}}, text::String)
    token_list = join(["$(token[1]) (prob: $(round(token[2], digits=3)))" for token in tokens], "\n")
    return """
You are an expert tasked with forming coherent entity names from predicted tokens.

Given the following predicted tokens with their probabilities, form coherent entity names that would logically complete the relationship in the text.

Predicted tokens:
$token_list

Original text context: $text

Create 3-5 coherent entity names that could be the tail entity in a relationship. Each entity should:
- Be a valid term
- Be consistent with the context
- Use appropriate terminology
- Be 1-3 words long

Return each entity on a separate line:

Formed entities:
"""
end

function _parse_tail_formation_response(response::String)
    entities = String[]
    for line in split(response, '\n')
        line = strip(line)
        if !isempty(line) && !startswith(line, "#") && !startswith(line, "Formed")
            clean_line = replace(line, r"^\d+\.?\s*" => "")
            clean_line = replace(clean_line, r"[-*_]" => "")
            clean_line = strip(clean_line)
            if !isempty(clean_line) && length(clean_line) > 2
                push!(entities, clean_line)
            end
        end
    end
    return unique(entities)
end

# =============================================================================
# Exports
# =============================================================================

export LocalLLMConfig, LocalLLMClient, LocalModelMetadata, load_local_model, match_relations, discover_entities, form_tail_from_tokens, generate

end  # module LocalLLM
