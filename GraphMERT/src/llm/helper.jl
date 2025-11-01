"""
Helper LLM integration for entity discovery and relation matching.

This module provides a complete LLM API client for enhancing GraphMERT's
entity discovery and relation matching capabilities with modern language models.
Domain-specific prompts are generated using domain providers.
"""

using HTTP
using JSON3
using Dates

"""
    HelperLLMConfig

Configuration for helper LLM client.
"""
struct HelperLLMConfig
    api_key::String
    base_url::String
    model::String
    timeout::Int
    max_retries::Int
    temperature::Float64
    max_tokens::Int
    rate_limit::Int  # tokens per minute
end

"""
    HelperLLMResponse

Response from helper LLM.
"""
struct HelperLLMResponse
    success::Bool
    content::String
    error::Union{String,Nothing}
    usage::Dict{String,Any}
    http_status::Union{Int,Nothing}
end

"""
    HelperLLMCache

Local cache for LLM responses with TTL.
"""
mutable struct HelperLLMCache
    responses::Dict{String,Tuple{HelperLLMResponse,DateTime}}
    max_size::Int
    ttl_seconds::Int

    function HelperLLMCache(max_size::Int = 1000, ttl_seconds::Int = 3600)
        new(Dict{String,Tuple{HelperLLMResponse,DateTime}}(), max_size, ttl_seconds)
    end
end

"""
    HelperLLMClient

Helper LLM client with rate limiting and caching.
"""
mutable struct HelperLLMClient
    config::HelperLLMConfig
    cache::HelperLLMCache
    last_request_time::Float64
    request_count::Int
    token_count::Int
    rate_limit_window_start::Float64
end

"""
    create_helper_llm_client(api_key::String; kwargs...)

Create a new helper LLM client with OpenAI API integration.

# Arguments
- `api_key::String`: OpenAI API key
- `base_url::String`: API base URL (default: "https://api.openai.com/v1")
- `model::String`: Model to use (default: "gpt-3.5-turbo")
- `timeout::Int`: Request timeout in seconds (default: 30)
- `max_retries::Int`: Maximum retry attempts (default: 3)
- `temperature::Float64`: Sampling temperature (default: 0.7)
- `max_tokens::Int`: Maximum response tokens (default: 1000)
- `rate_limit::Int`: Token rate limit per minute (default: 10000)

# Returns
- `HelperLLMClient`: Configured LLM client

# Example
```julia
client = create_helper_llm_client(ENV["OPENAI_API_KEY"])
entities = discover_entities(client, "Diabetes is a chronic condition.")
println("Found entities: ", entities)
```
"""
function create_helper_llm_client(
    api_key::String;
    base_url::String = "https://api.openai.com/v1",
    model::String = "gpt-3.5-turbo",
    timeout::Int = 30,
    max_retries::Int = 3,
    temperature::Float64 = 0.7,
    max_tokens::Int = 1000,
    rate_limit::Int = 10000,
)
    config = HelperLLMConfig(
        api_key,
        base_url,
        model,
        timeout,
        max_retries,
        temperature,
        max_tokens,
        rate_limit,
    )
    cache = HelperLLMCache(1000, 3600)
    return HelperLLMClient(config, cache, 0.0, 0, 0, time())
end

"""
    _make_llm_request(client::HelperLLMClient, messages::Vector{Dict{String, String}})

Make an authenticated HTTP request to OpenAI API with rate limiting and retry logic.

# Arguments
- `client::HelperLLMClient`: LLM client instance
- `messages::Vector{Dict{String, String}}`: Chat messages

# Returns
- `Tuple{Union{Dict, Nothing}, Union{String, Nothing}, Union{Int, Nothing}}`: (response_data, error_message, status_code)
"""
function _make_llm_request(client::HelperLLMClient, messages::Vector{Dict{String,String}})
    # Rate limiting check (tokens per minute)
    current_time = time()
    if current_time - client.rate_limit_window_start < 60.0  # 1 minute window
        if client.token_count >= client.config.rate_limit
            sleep_time = 60.0 - (current_time - client.rate_limit_window_start)
            @info "Token rate limit reached, sleeping for $sleep_time seconds"
            sleep(sleep_time)
            client.rate_limit_window_start = time()
            client.token_count = 0
        end
    else
        client.rate_limit_window_start = current_time
        client.token_count = 0
    end

    client.request_count += 1
    client.last_request_time = current_time

    # Prepare request
    url = "$(client.config.base_url)/chat/completions"
    headers = [
        "Authorization" => "Bearer $(client.config.api_key)",
        "Content-Type" => "application/json",
    ]

    # Estimate token count (rough approximation)
    estimated_tokens = sum(length(msg["content"]) for msg in messages) ÷ 4
    client.token_count += estimated_tokens

    request_body = Dict(
        "model" => client.config.model,
        "messages" => messages,
        "temperature" => client.config.temperature,
        "max_tokens" => client.config.max_tokens,
    )

    # Make request with retry logic
    for attempt = 1:client.config.max_retries
        try
            response = HTTP.post(
                url,
                headers,
                JSON3.write(request_body);
                timeout = client.config.timeout,
            )

            if response.status == 200
                data = JSON3.read(String(response.body))
                return data, nothing, response.status
            elseif response.status == 429  # Rate limited
                if attempt < client.config.max_retries
                    sleep_time = 2^attempt  # Exponential backoff
                    @warn "Rate limited, retrying in $sleep_time seconds (attempt $attempt/$(client.config.max_retries))"
                    sleep(sleep_time)
                    continue
                else
                    return nothing, "Rate limit exceeded", response.status
                end
            else
                error_msg = "HTTP $(response.status): $(String(response.body))"
                return nothing, error_msg, response.status
            end
        catch e
            if attempt < client.config.max_retries
                sleep_time = 2^attempt
                @warn "Request failed, retrying in $sleep_time seconds (attempt $attempt/$(client.config.max_retries)): $e"
                sleep(sleep_time)
                continue
            else
                return nothing, "Request failed: $e", nothing
            end
        end
    end

    return nothing, "Max retries exceeded", nothing
end

"""
    make_llm_request(client::HelperLLMClient, prompt::String)

Make a request to the helper LLM with a simple text prompt.

# Arguments
- `client::HelperLLMClient`: LLM client instance
- `prompt::String`: Text prompt

# Returns
- `HelperLLMResponse`: LLM response with content and metadata
"""
function make_llm_request(client::HelperLLMClient, prompt::String)
    messages = [Dict("role" => "user", "content" => prompt)]

    data, error_msg, status = _make_llm_request(client, messages)

    if data !== nothing && haskey(data, "choices") && !isempty(data["choices"])
        content = data["choices"][1]["message"]["content"]
        usage = get(data, "usage", Dict{String,Any}())

        return HelperLLMResponse(true, content, nothing, usage, status)
    else
        return HelperLLMResponse(false, "", error_msg, Dict{String,Any}(), status)
    end
end

"""
    discover_entities(client::HelperLLMClient, text::String, domain::Any; use_cache::Bool=true)

Discover entities in text using LLM with domain-specific prompts.

# Arguments
- `client::HelperLLMClient`: LLM client instance
- `text::String`: Text to analyze
- `domain::DomainProvider`: Domain provider for prompt generation
- `use_cache::Bool`: Whether to use cached responses

# Returns
- `Vector{String}`: Discovered entity names
"""
function discover_entities(
  client::HelperLLMClient,
  text::String,
  domain::Any;
  use_cache::Bool = true,
)
  # Check cache first
  cache_key = "entities:$(hash(text)):$(hash(string(typeof(domain))))"
  if use_cache && haskey(client.cache.responses, cache_key)
    cached_response, cache_time = client.cache.responses[cache_key]
    if (now() - cache_time).value / 1000 < client.cache.ttl_seconds
      return parse_entity_response(cached_response.content)
    end
  end

  # Create entity discovery prompt using domain provider
  context = Dict{String, Any}("text" => text, "task_type" => :entity_discovery)
  prompt = try
    GraphMERT.create_prompt(domain, :entity_discovery, context)
  catch e
    @warn "Domain prompt creation failed: $e, using fallback prompt"
    create_fallback_entity_discovery_prompt(text)
  end

  # Make LLM request
  response = make_llm_request(client, prompt)

  if response.success
    entities = parse_entity_response(response.content)

    # Cache response
    if use_cache
      client.cache.responses[cache_key] = (response, now())
    end

    return entities
  else
    @warn "Entity discovery failed: $(response.error)"
    return String[]  # Return empty on failure
  end
end

"""
    match_relations(client::HelperLLMClient, entities::Vector{String}, text::String; use_cache::Bool=true)

Match relations between entities using LLM.

# Arguments
- `client::HelperLLMClient`: LLM client instance
- `entities::Vector{String}`: Entity names
- `text::String`: Original text for context
- `use_cache::Bool`: Whether to use cached responses

# Returns
- `Dict{String, Dict{String, String}}`: Relations in format entity1 => (relation => entity2)
"""
function match_relations(
    client::HelperLLMClient,
    entities::Vector{String},
    text::String;
    use_cache::Bool = true,
)
    # Check cache first
    entities_key = join(sort(entities), "|")
    cache_key = "relations:$(hash(entities_key)):$(hash(text))"
    if use_cache && haskey(client.cache.responses, cache_key)
        cached_response, cache_time = client.cache.responses[cache_key]
        if (now() - cache_time).value / 1000 < client.cache.ttl_seconds
            return parse_relation_response(cached_response.content)
        end
    end

    # Create relation matching prompt
    prompt = create_relation_matching_prompt(entities, text)

    # Make LLM request
    response = make_llm_request(client, prompt)

    if response.success
        relations = parse_relation_response(response.content)

        # Cache response
        if use_cache
            client.cache.responses[cache_key] = (response, now())
        end

        return relations
    else
        @warn "Relation matching failed: $(response.error)"
        return Dict{String,Dict{String,String}}()  # Return empty on failure
    end
end

"""
    create_entity_discovery_prompt(text::String)

DEPRECATED: Use domain.create_prompt(domain, :entity_discovery, context) instead.

Create a structured prompt for entity discovery (fallback).
"""
function create_entity_discovery_prompt(text::String)
  return create_fallback_entity_discovery_prompt(text)
end

"""
    create_fallback_entity_discovery_prompt(text::String)

Fallback entity discovery prompt when domain provider is not available.
"""
function create_fallback_entity_discovery_prompt(text::String)
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

"""
    create_relation_matching_prompt(entities::Vector{String}, text::String)

DEPRECATED: Use domain.create_prompt(domain, :relation_matching, context) instead.

Create a structured prompt for relation matching (fallback).
"""
function create_relation_matching_prompt(entities::Vector{String}, text::String)
  return create_fallback_relation_matching_prompt(entities, text)
end

"""
    create_fallback_relation_matching_prompt(entities::Vector{String}, text::String)

Fallback relation matching prompt when domain provider is not available.
"""
function create_fallback_relation_matching_prompt(entities::Vector{String}, text::String)
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

"""
    create_tail_formation_prompt(tokens::Vector{Tuple{Int, Float64}}, text::String, domain::Any)

Create a prompt for forming coherent tail entities from predicted tokens using domain provider.

# Arguments
- `tokens::Vector{Tuple{Int, Float64}}`: Top-k predicted tokens with probabilities
- `text::String`: Original text for context
- `domain::DomainProvider`: Domain provider for prompt generation

# Returns
- `String`: Formatted prompt for LLM
"""
function create_tail_formation_prompt(tokens::Vector{Tuple{Int,Float64}}, text::String, domain::Any)
  context = Dict{String, Any}("tokens" => tokens, "text" => text, "task_type" => :tail_formation)
  return try
    GraphMERT.create_prompt(domain, :tail_formation, context)
  catch e
    @warn "Domain prompt creation failed: $e, using fallback prompt"
    create_fallback_tail_formation_prompt(tokens, text)
  end
end

"""
    create_fallback_tail_formation_prompt(tokens::Vector{Tuple{Int, Float64}}, text::String)

Fallback tail formation prompt when domain provider is not available.
"""
function create_fallback_tail_formation_prompt(tokens::Vector{Tuple{Int,Float64}}, text::String)
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

"""
    parse_entity_response(response::String)

Parse LLM response for entity discovery.

# Arguments
- `response::String`: LLM response text

# Returns
- `Vector{String}`: Extracted entity names
"""
function parse_entity_response(response::String)
    entities = String[]
    for line in split(response, '\n')
        line = strip(line)
        # Skip empty lines, comments, and headers
        if !isempty(line) &&
           !startswith(line, "#") &&
           !startswith(line, "Extracted") &&
           !startswith(line, "Entities")
            # Clean up common artifacts
            clean_line = replace(line, r"^\d+\.?\s*" => "")  # Remove numbering
            clean_line = replace(clean_line, r"[-*_]" => "")  # Remove bullets
            clean_line = strip(clean_line)
            if !isempty(clean_line) && length(clean_line) > 2
                push!(entities, clean_line)
            end
        end
    end
    return unique(entities)  # Remove duplicates
end

"""
    parse_relation_response(response::String)

Parse LLM response for relation matching.

# Arguments
- `response::String`: LLM response text

# Returns
- `Dict{String, Dict{String, String}}`: Relations in format entity1 => (relation => entity2)
"""
function parse_relation_response(response::String)
    relations = Dict{String,Dict{String,String}}()

    for line in split(response, '\n')
        line = strip(line)
        if !isempty(line) &&
           !startswith(line, "#") &&
           !startswith(line, "Relationships") &&
           occursin("->", line)
            # Parse format: entity1 -> relation -> entity2
            parts = split(line, "->")
            if length(parts) >= 3
                entity1 = strip(strip(parts[1]))
                relation = strip(strip(parts[2]))
                entity2 = strip(strip(join(parts[3:end], "->")))  # Handle multi-word relations

                # Clean up relation
                relation = replace(relation, r"[^\w\s]" => "")  # Remove punctuation
                relation = uppercase(replace(relation, r"\s+" => "_"))  # Normalize

                if !isempty(entity1) && !isempty(relation) && !isempty(entity2)
                    relations[entity1] = Dict("relation" => relation, "entity2" => entity2)
                end
            end
        end
    end

    return relations
end

"""
    parse_tail_formation_response(response::String)

Parse LLM response for tail entity formation.

# Arguments
- `response::String`: LLM response text

# Returns
- `Vector{String}`: Formed tail entity names
"""
function parse_tail_formation_response(response::String)
    entities = String[]
    for line in split(response, '\n')
        line = strip(line)
        if !isempty(line) && !startswith(line, "#") && !startswith(line, "Formed")
            # Clean up common artifacts
            clean_line = replace(line, r"^\d+\.?\s*" => "")  # Remove numbering
            clean_line = replace(clean_line, r"[-*_]" => "")  # Remove bullets
            clean_line = strip(clean_line)
            if !isempty(clean_line) && length(clean_line) > 2
                push!(entities, clean_line)
            end
        end
    end
    return unique(entities)
end

"""
    discover_entities_batch(client::HelperLLMClient, texts::Vector{String})

Discover entities in multiple texts using LLM.

# Arguments
- `client::HelperLLMClient`: LLM client instance
- `texts::Vector{String}`: Texts to analyze

# Returns
- `Vector{Vector{String}}`: List of entity lists, one per text
"""
function discover_entities_batch(client::HelperLLMClient, texts::Vector{String})
    results = Vector{Vector{String}}()

    for text in texts
        entities = discover_entities(client, text)
        push!(results, entities)

        # Rate limiting - don't overwhelm the API
        sleep(0.1)
    end

    return results
end

"""
    match_relations_batch(client::HelperLLMClient, entity_lists::Vector{Vector{String}}, texts::Vector{String})

Match relations in multiple texts using LLM.

# Arguments
- `client::HelperLLMClient`: LLM client instance
- `entity_lists::Vector{Vector{String}}`: Entity lists for each text
- `texts::Vector{String}`: Original texts

# Returns
- `Vector{Dict{String, Dict{String, String}}}`: List of relation dictionaries
"""
function match_relations_batch(
    client::HelperLLMClient,
    entity_lists::Vector{Vector{String}},
    texts::Vector{String},
)
    results = Vector{Dict{String,Dict{String,String}}}()

    for (entities, text) in zip(entity_lists, texts)
        if !isempty(entities)
            relations = match_relations(client, entities, text)
            push!(results, relations)
        else
            push!(results, Dict{String,Dict{String,String}}())
        end

        # Rate limiting - don't overwhelm the API
        sleep(0.1)
    end

    return results
end

"""
    form_tail_from_tokens(tokens::Vector{Tuple{Int, Float64}}, text::String,
                        client::HelperLLMClient; use_cache::Bool=true)

Form coherent tail entities from predicted tokens using LLM.

# Arguments
- `tokens::Vector{Tuple{Int, Float64}}`: Top-k predicted tokens
- `text::String`: Original text for context
- `client::HelperLLMClient`: LLM client instance
- `use_cache::Bool`: Whether to use cached responses

# Returns
- `Vector{String}`: Formed tail entity names
"""
function form_tail_from_tokens(
    tokens::Vector{Tuple{Int,Float64}},
    text::String,
    client::HelperLLMClient;
    use_cache::Bool = true,
)
    # Check cache first
    tokens_key = join([string(t[1]) for t in tokens], "|")
    cache_key = "tails:$(hash(tokens_key)):$(hash(text))"
    if use_cache && haskey(client.cache.responses, cache_key)
        cached_response, cache_time = client.cache.responses[cache_key]
        if (now() - cache_time).value / 1000 < client.cache.ttl_seconds
            return parse_tail_formation_response(cached_response.content)
        end
    end

    # Create tail formation prompt
    prompt = create_tail_formation_prompt(tokens, text)

    # Make LLM request
    response = make_llm_request(client, prompt)

    if response.success
        tails = parse_tail_formation_response(response.content)

        # Cache response
        if use_cache
            client.cache.responses[cache_key] = (response, now())
        end

        return tails
    else
        @warn "Tail formation failed: $(response.error)"
        return String[]  # Return empty on failure
    end
end

# These fallback functions have been moved to domain providers
# Use domain-specific fallback methods instead
