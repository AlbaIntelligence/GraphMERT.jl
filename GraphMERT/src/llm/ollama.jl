"""
Ollama LLM backend for GraphMERT.jl

This module provides an Ollama client for running GGUF-format language models locally.
It uses Ollama's HTTP API to interact with the local Ollama server.

# Recommended Models

For entity extraction, the recommended model is `lfm2.5-thinking:latest`:
- Excellent entity classification accuracy
- Correctly identifies people, places, organizations
- Good reasoning capabilities for relation extraction

Other tested models:
- `ministral-3:latest` - Good but may misclassify some entities
- `qwen3:0.6b` - Not recommended (poor entity extraction)

# Usage

```julia
using GraphMERT
using GraphMERT.OllamaClient

# Create client with recommended model
config = OllamaConfig(model="lfm2.5-thinking:latest")
client = OllamaLLMClient(config)

# Or use with ProcessingOptions
options = ProcessingOptions(
    domain = "wikipedia",
    use_ollama = true,
    ollama_config = config,
)

# Extract entities
entities = extract_entities(domain, text, options)
```
"""

module OllamaClient

using HTTP
using JSON3

# =============================================================================
# Type Definitions
# =============================================================================

"""
    OllamaConfig(; model="lfm2.5-thinking:latest", base_url="http://localhost:11434", timeout=120)

Configuration for Ollama client.

# Arguments
- `model::String`: Model name to use (default: "lfm2.5-thinking:latest")
- `base_url::String`: Ollama server URL (default: "http://localhost:11434")
- `timeout::Int`: Request timeout in seconds (default: 120)

# Example

```julia
config = OllamaConfig(
    model="lfm2.5-thinking:latest",
    base_url="http://localhost:11434",
    timeout=180,
)
```
"""
struct OllamaConfig
    model::String
    base_url::String
    timeout::Int

    function OllamaConfig(;
        model::String="lfm2.5-thinking:latest",
        base_url::String="http://localhost:11434",
        timeout::Int=120
    )
        if timeout < 1
            throw(ArgumentError("timeout must be at least 1"))
        end
        new(model, base_url, timeout)
    end
end

"""
    OllamaLLMClient

Client for interacting with Ollama server.
"""
mutable struct OllamaLLMClient
    config::OllamaConfig
    process::Union{Base.Process, Nothing}
    is_running::Bool

    function OllamaLLMClient(config::OllamaConfig)
        new(config, nothing, false)
    end
end

# =============================================================================
# Server Control
# =============================================================================

"""
    start_server(client::OllamaLLMClient; ollama_path::String="ollama")

Start the Ollama server as a background process.
"""
function start_server(client::OllamaLLMClient; ollama_path::String="ollama")
    if client.is_running
        @warn "Ollama server already running"
        return client
    end

    # Start ollama serve in background
    try
        # Use pipeline to redirect output
        process = run(`$ollama_path serve`; wait=false)
        client.process = process
        
        # Wait for server to be ready
        for i in 1:30
            if is_server_ready(client.config.base_url)
                client.is_running = true
                @info "Ollama server started successfully"
                return client
            end
            sleep(1)
        end
        
        @warn "Ollama server may not be ready yet"
        client.is_running = true
        return client
    catch e
        @error "Failed to start Ollama server: $e"
        rethrow(e)
    end
end

"""
    stop_server(client::OllamaLLMClient)

Stop the Ollama server.
"""
function stop_server(client::OllamaLLMClient)
    if !client.is_running || client.process === nothing
        @warn "Ollama server not running"
        return
    end

    try
        # Kill the process
        kill(client.process)
        wait(client.process)
        client.is_running = false
        client.process = nothing
        @info "Ollama server stopped"
    catch e
        @warn "Error stopping Ollama server: $e"
    end
end

"""
    is_server_ready(url::String)::Bool

Check if Ollama server is ready.
"""
function is_server_ready(url::String)::Bool
    try
        response = HTTP.get("$url/api/tags"; timeout=5)
        return response.status == 200
    catch
        return false
    end
end

"""
    is_available(url::String)::Bool

Check if Ollama server is available at the given URL.
"""
function is_available(url::String="http://localhost:11434")::Bool
    return is_server_ready(url)
end

# =============================================================================
# Generation
# =============================================================================

"""
    generate(client::OllamaLLMClient, prompt::String; temperature::Float64=0.7, max_tokens::Int=512)

Generate text using Ollama.
"""
function generate(client::OllamaLLMClient, prompt::String; temperature::Float64=0.7, max_tokens::Int=512)
    url = "$(client.config.base_url)/api/generate"
    
    body = Dict(
        "model" => client.config.model,
        "prompt" => prompt,
        "temperature" => temperature,
        "max_tokens" => max_tokens,
        "stream" => false
    )
    
    headers = ["Content-Type" => "application/json"]
    
    try
        response = HTTP.post(url, headers, JSON3.write(body); timeout=client.config.timeout)
        if response.status != 200
            error("Ollama API error: $(response.status)")
        end
        
        result = JSON3.read(response.body, Dict{String, Any})
        return result["response"]
    catch e
        @error "Generation failed: $e"
        rethrow(e)
    end
end

"""
    chat(client::OllamaLLMClient, messages::Vector{Dict{String, String}}; temperature::Float64=0.7)

Chat with Ollama using a message history.
"""
function chat(client::OllamaLLMClient, messages::Vector{Dict{String, String}}; temperature::Float64=0.7)
    url = "$(client.config.base_url)/api/chat"
    
    body = Dict(
        "model" => client.config.model,
        "messages" => messages,
        "temperature" => temperature,
        "stream" => false
    )
    
    headers = ["Content-Type" => "application/json"]
    
    try
        response = HTTP.post(url, headers, JSON3.write(body); timeout=client.config.timeout)
        if response.status != 200
            error("Ollama API error: $(response.status)")
        end
        
        result = JSON3.read(response.body, Dict{String, Any})
        return result["message"]["content"]
    catch e
        @error "Chat failed: $e"
        rethrow(e)
    end
end

# =============================================================================
# Entity/Relation Extraction Helpers
# =============================================================================

"""
    discover_entities(client::OllamaLLMClient, text::String, domain::String)::Vector{String}

Discover entities in text using Ollama.
"""
function discover_entities(client::OllamaLLMClient, text::String, domain::String="wikipedia")::Vector{String}
    prompt = """
You are an expert entity extraction system.

Extract all entities (people, places, organizations, concepts) from the following $domain text.

Return only the entity names, one per line. Do not include any other text.

Text:
$text

Entities:
"""
    
    response = generate(client, prompt; temperature=0.3, max_tokens=512)
    
    entities = String[]
    for line in split(response, '\n')
        line = strip(line)
        if !isempty(line) && !startswith(line, "#")
            push!(entities, line)
        end
    end
    
    return unique(entities)
end

"""
    match_relations(client::OllamaLLMClient, entities::Vector{String}, text::String)::Dict{String, Dict{String, String}}

Match relations between entities using Ollama.
"""
function match_relations(client::OllamaLLMClient, entities::Vector{String}, text::String)::Dict{String, Dict{String, String}}
    if length(entities) < 2
        return Dict{String, Dict{String, String}}()
    end
    
    entities_str = join(entities, ", ")
    
    prompt = """
You are an expert relation extraction system.

Given the following text and entities, identify relations between pairs of entities.

Entities: $entities_str

Text:
$text

Return relations in the format:
HEAD_ENTITY|RELATION_TYPE|TAIL_ENTITY

One per line. Use relation types like: WORKS_AT, LOCATED_IN, MARRIED_TO, PARENT_OF, RULES, etc.
If no clear relation exists, skip that pair.
"""
    
    response = generate(client, prompt; temperature=0.3, max_tokens=512)
    
    relations = Dict{String, Dict{String, String}}()
    
    for line in split(response, '\n')
        line = strip(line)
        parts = split(line, "|")
        if length(parts) == 3
            head, rel_type, tail = strip.(parts)
            if head in entities && tail in entities
                relations["$head|$rel_type|$tail"] = Dict(
                    "head" => head,
                    "relation" => rel_type,
                    "tail" => tail
                )
            end
        end
    end
    
    return relations
end

"""
    form_tail_from_tokens(client::OllamaLLMClient, tokens::Vector{String}, context::String)::Vector{String}

Form coherent tails from tokens using Ollama.
"""
function form_tail_from_tokens(client::OllamaLLMClient, tokens::Vector{String}, context::String)::Vector{String}
    if isempty(tokens)
        return String[]
    end
    
    tokens_str = join(tokens, ", ")
    
    prompt = """
You are an expert at forming coherent entity names from partial tokens.

Given these tokens: $tokens_str

And this context:
$context

Form the most likely complete entity (person, place, organization, etc.) that these tokens represent.
Return just the entity name, nothing else.
"""
    
    response = strip(generate(client, prompt; temperature=0.3, max_tokens=64))
    
    return [response]
end

# =============================================================================
# Convenience Constructors
# =============================================================================

"""
    OllamaLLMClient(; model="lfm2.5-thinking:latest", base_url="http://localhost:11434")

Create an Ollama client with default settings.

# Example

```julia
# Use default model (lfm2.5-thinking:latest)
client = OllamaLLMClient()

# Or specify a different model
client = OllamaLLMClient(model="ministral-3:latest")
```
"""
function OllamaLLMClient(; model="lfm2.5-thinking:latest", base_url="http://localhost:11434")
    config = OllamaConfig(model=model, base_url=base_url)
    return OllamaLLMClient(config)
end

# Export
export OllamaConfig, OllamaLLMClient
export start_server, stop_server, is_available
export generate, chat
export discover_entities, match_relations, form_tail_from_tokens

end  # module
