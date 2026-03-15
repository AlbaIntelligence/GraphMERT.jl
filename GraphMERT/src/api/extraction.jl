"""
Knowledge Graph Extraction API for GraphMERT.jl

This module implements the complete knowledge graph extraction pipeline
from text using a trained GraphMERT model with domain-specific providers.

The extraction pipeline follows the 5-stage process from the paper:
1. Head Discovery: Extract entities from text using domain provider
2. Relation Matching: Match entities to relations using domain provider
3. Tail Prediction: Use GraphMERT to predict tail tokens
4. Tail Formation: Combine tokens into coherent tails
5. Filtering: Apply similarity and deduplication filters

All entity and relation extraction uses domain providers for domain-specific logic.
"""

# Types will be available from main module
# using ..Types: KnowledgeGraph, BiomedicalEntity, BiomedicalRelation, TextPosition

# using DocStringExtensions  # Temporarily disabled

# Import LLM client types from main module for backend selection
using ...GraphMERT: HelperLLMClient, HelperLLMConfig, create_helper_llm_client
using ...GraphMERT: LocalLLMClient, LocalLLMConfig, load_model as load_local_model

# These functions have been moved to domain providers
# Use domain.extract_entities() and domain.calculate_entity_confidence() instead


"""
    _create_llm_client(options::ProcessingOptions)

Create an LLM client based on processing options.

# Arguments
- `options::ProcessingOptions`: Processing options containing use_local/local_config

# Returns
- `Union{LocalLLMClient, HelperLLMClient, Nothing}`: LLM client instance, or nothing if no client needed
"""
function _create_llm_client(options::GraphMERT.ProcessingOptions)
    if options.use_local
        if options.local_config === nothing
            throw(ArgumentError("local_config must be provided when use_local is true"))
        end
        return load_local_model(options.local_config)
    else
        return nothing
    end
end


"""
    match_relations_for_entities(entities::Vector{Entity}, text::String, domain::Any, options::ProcessingOptions; llm_client=nothing)

Stage 2: Relation Matching - Match entities to relations using domain provider.
When the domain supports it, `llm_client` is passed for LLM-based relation extraction.

# Returns
- `Vector{Relation}`: Extracted relations
"""
function match_relations_for_entities(
  entities::Vector{GraphMERT.Entity},
  text::String,
  domain::Any,
  options::GraphMERT.ProcessingOptions = GraphMERT.default_processing_options();
  llm_client = nothing,
)
  if domain === nothing
    return _fallback_cooccurrence_relations(entities, text, options)
  end
  try
    relations = Base.invokelatest(
      GraphMERT.extract_relations,
      domain,
      entities,
      text,
      options;
      llm_client = llm_client,
    )
    return relations
  catch e
    @warn "Domain relation extraction failed: $e, falling back to sentence-level co-occurrence relations."
    return _fallback_cooccurrence_relations(entities, text, options)
  end
end

function _fallback_cooccurrence_relations(
  entities::Vector{GraphMERT.Entity},
  text::String,
  options::GraphMERT.ProcessingOptions,
)::Vector{GraphMERT.Relation}
  rels = GraphMERT.Relation[]
  length(entities) < 2 && return rels
  seen = Set{Tuple{String,String,String}}()
  sentences = split(text, r"[.!?]\s*")
  for sentence in sentences
    sentence = strip(sentence)
    isempty(sentence) && continue
    sentence_lower = lowercase(sentence)
    in_sentence = [e for e in entities if occursin(lowercase(e.text), sentence_lower)]
    for i in 1:length(in_sentence)
      for j in (i + 1):length(in_sentence)
        head = in_sentence[i]
        tail = in_sentence[j]
        key = (head.id, "ASSOCIATED_WITH", tail.id)
        key in seen && continue
        push!(seen, key)
        relation = "ASSOCIATED_WITH"
        if occursin(r"treat|treated with", sentence_lower)
          relation = "TREATS"
        elseif occursin(r"cause|causes", sentence_lower)
          relation = "CAUSES"
        end
        push!(
          rels,
          GraphMERT.Relation(
            head.id,
            tail.id,
            relation,
            0.5,
            options.domain,
            String(sentence),
            "",
            Dict{String,Any}(),
            "",
          ),
        )
      end
    end
  end
  return rels
end

# These functions have been moved to domain providers
# Use domain.validate_relation() and domain.calculate_relation_confidence() instead

"""
    predict_tail_tokens(model, head_entity::Entity,
                       relation::String, text::String, top_k::Int=20)

Stage 3: Tail Prediction - Use GraphMERT (or a compatible mock model) to predict tail tokens.

The `model` argument is treated as any callable that accepts `(input_ids, attention_mask)`
and returns a 3D logits tensor of shape `[batch, seq_len, vocab_size]`.
"""
function predict_tail_tokens(
  model,
  head_entity::Any,
  relation::String,
  text::String,
  top_k::Int=20,
)
  # Create a simple graph for prediction (would be more sophisticated)
  graph = GraphMERT.create_leafy_chain_from_text(text)

  # Encode head entity and relation into the graph
  # This is a simplified implementation - full version would be more complex

  # Forward pass through model
  input_ids = GraphMERT.graph_to_sequence(graph)
  attention_mask = GraphMERT.create_attention_mask(graph)

  # Get model predictions (simplified)
  logits = model(reshape(input_ids, 1, :), reshape(attention_mask, 1, :))

  # Extract top-k predictions for tail positions
  # This is a placeholder - full implementation would be much more sophisticated
  vocab_size = size(logits, 3)
  tail_probs = rand(Float32, vocab_size)  # Placeholder probabilities

  # Get top-k tokens
  top_indices = sortperm(tail_probs, rev=true)[1:min(top_k, vocab_size)]
  top_tokens = [(idx, Float64(tail_probs[idx])) for idx in top_indices]

  return top_tokens
end

"""
    form_tail_from_tokens(tokens::Vector{Tuple{Int, Float64}}, text::String,
                        llm_client::Union{LLMClient, Nothing}=nothing)

Stage 4: Tail Formation - Combine tokens into coherent tails.

"""
function form_tail_from_tokens(
  tokens::Vector{Tuple{Int,Float64}},
  text::String,
  llm_client::Union{Any,Nothing}=nothing,
)
  possible_tails = Vector{String}()

  # Simple approach: combine tokens into entity-like strings
  # In practice, would use LLM to form grammatically correct entities

  # Take top tokens and form combinations
  top_token_ids = [token[1] for token in tokens[1:min(5, length(tokens))]]

  # Simple entity formation (would be more sophisticated)
  for token_id in top_token_ids
    # Convert token ID to string representation (simplified)
    entity_str = "entity_$(token_id)"
    push!(possible_tails, entity_str)
  end

  return possible_tails
end

"""
    filter_and_deduplicate_triples(triples::Vector{<:Tuple},
                                  text::String, β_threshold::Float64=0.8)

Stage 5: Filtering - Apply similarity and deduplication filters.

Accepts any head type `T` that has a `text` field (e.g., `Entity`, `BiomedicalEntity`).
"""
function filter_and_deduplicate_triples(
  triples::Vector{<:Tuple},
  text::String,
  β_threshold::Float64=0.8,
)
  filtered_triples = Tuple[]

  # Remove duplicates
  seen = Set{String}()
  for (head, relation, tail, confidence) in triples
    head_text = getfield(head, :text)
    triple_key = "$(head_text)_$(relation)_$(tail)"
    if !(triple_key in seen)
      push!(seen, triple_key)
      push!(filtered_triples, (head, relation, tail, confidence))
    end
  end

  # Filter by similarity threshold
  final_triples = Tuple[]
  for (head, relation, tail, confidence) in filtered_triples
    similarity = calculate_tail_similarity(tail, text)
    if similarity ≥ β_threshold
      push!(final_triples, (head, relation, tail, confidence * similarity))
    end
  end

  return final_triples
end

"""
    calculate_tail_similarity(tail::String, text::String)

Simple similarity heuristic between a candidate tail string and the source text.
Used for filtering triples when full semantic similarity is not available.
"""
function calculate_tail_similarity(tail::String, text::String)
  tail_norm = lowercase(strip(tail))
  text_norm = lowercase(strip(text))
  isempty(tail_norm) && return 0.0

  # Token-overlap based Jaccard similarity
  tail_tokens = Set(split(tail_norm))
  text_tokens = Set(split(text_norm))
  inter = length(intersect(tail_tokens, text_tokens))
  union_sz = max(length(union(tail_tokens, text_tokens)), 1)
  return inter / union_sz
end

"""
    extract_knowledge_graph(text::String, model::GraphMERTModel;
                           options::ProcessingOptions=ProcessingOptions())::KnowledgeGraph

Main knowledge graph extraction function.

Extracts structured knowledge from text using a trained GraphMERT model and domain provider.

# Arguments
- `text::String`: Input text to extract from
- `model::GraphMERTModel`: Trained GraphMERT model
- `options::ProcessingOptions`: Processing options (must include domain field)

# Returns
- `KnowledgeGraph`: Extracted knowledge graph with entities and relations
"""
function extract_knowledge_graph(
  text::String,
  model;
  options::GraphMERT.ProcessingOptions=GraphMERT.default_processing_options(),
)::GraphMERT.KnowledgeGraph
  # Empty text: return empty KG with no phantom provenance (edge case per spec)
  if isempty(text)
    return GraphMERT.KnowledgeGraph(
      GraphMERT.KnowledgeEntity[],
      GraphMERT.KnowledgeRelation[],
      Dict("extraction_time" => string(now()), "domain" => options.domain, "num_entities" => 0, "num_relations" => 0, "empty_corpus" => true),
    )
  end
  if length(text) > options.max_length
    throw(ArgumentError("Input text length $(length(text)) exceeds max_length=$(options.max_length)"))
  end

  # Full encoder path (tail prediction) runs only for GraphMERTModel; other model types use fallback entity/relation path only
  @info "Starting knowledge graph extraction from text of length $(length(text)) with domain: $(options.domain)"

  # Create LLM client based on options (local or cloud)
  llm_client = _create_llm_client(options)
  if llm_client !== nothing
    backend_type = isa(llm_client, LocalLLMClient) ? "LocalLLMClient" : "HelperLLMClient"
    @info "Using $backend_type for LLM operations"
  end

  # Get domain provider from registry (may be missing during lightweight/API tests)
  domain_provider = GraphMERT.get_domain(options.domain)
  if domain_provider === nothing
    available_domains = GraphMERT.list_domains()
    if isempty(available_domains)
      @warn "Domain '$(options.domain)' is not registered and no domains are registered; using heuristic fallback for extraction."
    else
      @warn "Domain '$(options.domain)' is not registered. Available domains: $(join(available_domains, ", ")); using heuristic fallback for extraction."
    end
  end

  # Stage 1: Head Discovery using domain provider
  entities = discover_head_entities(text, domain_provider, options, llm_client)
  @info "Discovered $(length(entities)) entities using domain: $(options.domain)"

  # Stage 2: Relation Matching using domain provider (pass llm_client when domain supports it)
  relations = match_relations_for_entities(entities, text, domain_provider, options; llm_client = llm_client)
  @info "Extracted $(length(relations)) relations using domain: $(options.domain)"

  # Stages 3–5 (optional): When a full GraphMERT model is provided, run tail prediction, tail formation, and filtering
  if model !== nothing && !isempty(relations) && model isa GraphMERTModel
    triples = _build_triples_from_relations(entities, relations)
    top_k = options.top_k_predictions
    β = options.similarity_threshold
    id_to_entity = Dict(e.id => e for e in entities)
    for i in eachindex(triples)
      he, rel_type, _tail_text, conf = triples[i]
      tok_tuples = predict_tail_tokens(model, he, rel_type, text, top_k)
      possible_tails = form_tail_from_tokens(tok_tuples, text, llm_client)
      if !isempty(possible_tails)
        triples[i] = (he, rel_type, possible_tails[1], conf)
      end
    end
    filtered = filter_and_deduplicate_triples(triples, text, β)
    doc_id = "doc_$(hash(text) % UInt)"
    entities, relations = _triples_to_entities_relations(entities, filtered, options.domain, options.enable_provenance_tracking, doc_id)
  end

  # Create knowledge graph
  llm_backend = llm_client === nothing ? "none" : (isa(llm_client, LocalLLMClient) ? "local" : "cloud")
  return GraphMERT.KnowledgeGraph(
    entities,
    relations,
    Dict(
      "extraction_time" => string(now()),
      "model_version" => "GraphMERT-v0.1",
      "domain" => options.domain,
      "num_entities" => length(entities),
      "num_relations" => length(relations),
      "num_triples" => length(relations),
      "source_text" => text,
      "llm_backend" => llm_backend,
    ),
  )
end

# ============================================================================
# Triples helpers (stages 3–5)
# ============================================================================

function _build_triples_from_relations(entities::Vector, relations::Vector)
  id_to_entity = Dict(e.id => e for e in entities)
  triples = Tuple{Any,String,String,Float64}[]
  for r in relations
    he = get(id_to_entity, r.head, nothing)
    te = get(id_to_entity, r.tail, nothing)
    he === nothing && continue
    tail_text = te !== nothing ? te.text : r.tail
    push!(triples, (he, r.relation_type, tail_text, r.confidence))
  end
  return triples
end

function _triples_to_entities_relations(
  entities::Vector,
  filtered_triples::Vector{<:Tuple},
  domain::String,
  enable_provenance_tracking::Bool=false,
  document_id::String="doc_1",
)
  out_entities = copy(entities)
  entity_by_text = Dict(e.text => e for e in out_entities)
  out_relations = GraphMERT.Relation[]
  for (seg_idx, (head_entity, rel_type, tail_text, conf)) in enumerate(filtered_triples)
    tail_entity = get(entity_by_text, tail_text, nothing)
    if tail_entity === nothing
      tail_id = "tail_$(hash(tail_text))"
      tail_entity = GraphMERT.Entity(
        tail_id, tail_text, tail_text, "UNKNOWN", domain,
        Dict{String,Any}(), GraphMERT.TextPosition(1, 1, 1, 1), conf, "",
      )
      push!(out_entities, tail_entity)
      entity_by_text[tail_text] = tail_entity
    end
    attrs = Dict{String,Any}()
    if enable_provenance_tracking
      attrs["provenance_record"] = GraphMERT.ProvenanceRecord(document_id, seg_idx; context=nothing)
    end
    prov_str = enable_provenance_tracking ? "$(document_id)#$(seg_idx)" : ""
    push!(out_relations, GraphMERT.Relation(
      head_entity.id, tail_entity.id, rel_type, conf, domain, prov_str, "", attrs,
    ))
  end
  return out_entities, out_relations
end

# ============================================================================
# Head Discovery Helpers
# ============================================================================

"""
    discover_head_entities(text::String, domain::DomainProvider,
                           options::ProcessingOptions=default_processing_options(),
                           llm_client::Any=nothing)

Stage 1: Head Discovery – extract entities from text using a domain provider.
If llm_client is provided and options.use_local is true, uses LLM-based extraction.

Returns a `Vector{Entity}` produced by the domain's `extract_entities` implementation.
"""
function discover_head_entities(
  text::String,
  domain::Any,
  options::GraphMERT.ProcessingOptions = GraphMERT.default_processing_options(),
  llm_client::Any=nothing,
)
  # If llm_client is provided and options.use_local is true, use LLM-based extraction
  if llm_client !== nothing && options.use_local
    try
      # Pass llm_client to domain for LLM-based entity discovery
      return Base.invokelatest(GraphMERT.extract_entities, domain, text, options, llm_client)
    catch e
      @warn "LLM entity extraction failed: $e, falling back to domain extraction."
    end
  end
  
  # Default: use domain's extract_entities
  try
    # Use invokelatest so concrete domain methods (e.g. BiomedicalDomain) are selected after load
    return Base.invokelatest(GraphMERT.extract_entities, domain, text, options)
  catch e
    @warn "Domain entity extraction failed: $e. Falling back to simple entity recognition."
    # Fallback: use simple pattern-based recognition to create generic entities
    fallback_texts = GraphMERT.fallback_entity_recognition(text)
    entities = Vector{GraphMERT.Entity}()
    for (i, etext) in enumerate(fallback_texts)
      push!(
        entities,
        GraphMERT.Entity(
          "fallback_entity_$i",
          etext,
          etext,
          "UNKNOWN",
          options.domain,
          Dict{String,Any}(),
          GraphMERT.TextPosition(1, lastindex(etext), 1, 1),
          0.5,
          text,
        ),
      )
    end
    return entities
  end
end

"""
    discover_head_entities(text::String)

Convenience overload that uses the current default domain from the registry and
default processing options. Returns a `Vector{Entity}`.
"""
function discover_head_entities(text::String)
  opts = GraphMERT.default_processing_options()

  # Determine default domain name; if none, try to initialize defaults for backward compatibility
  domain_name = GraphMERT.get_default_domain_name()
  if domain_name === nothing
    try
      GraphMERT.initialize_default_domains()
    catch e
      @warn "Failed to initialize default domains: $e"
    end
    domain_name = GraphMERT.get_default_domain_name()
    if domain_name === nothing
      @warn "No default domain set; discover_head_entities(text) returning empty list."
      return Vector{GraphMERT.Entity}()
    end
  end

  if !GraphMERT.has_domain(domain_name)
    @warn "Default domain name '$(domain_name)' is not registered; discover_head_entities(text) returning empty list."
    return Vector{GraphMERT.Entity}()
  end

  domain = GraphMERT.get_domain(domain_name)
  if domain === nothing
    @warn "Default domain provider for '$(domain_name)' not found; discover_head_entities(text) returning empty list."
    return Vector{GraphMERT.Entity}()
  end

  # Ensure options.domain matches the default domain
  opts = GraphMERT.ProcessingOptions(
    max_length = opts.max_length,
    batch_size = opts.batch_size,
    use_umls = opts.use_umls,
    use_helper_llm = opts.use_helper_llm,
    confidence_threshold = opts.confidence_threshold,
    entity_types = opts.entity_types,
    relation_types = opts.relation_types,
    cache_enabled = opts.cache_enabled,
    parallel_processing = opts.parallel_processing,
    verbose = opts.verbose,
    domain = domain_name,
  )

  return discover_head_entities(text, domain, opts)
end

# These enhanced functions have been replaced by domain provider-based extraction
# Use discover_head_entities(domain, options) and match_relations_for_entities(domain, options) instead

"""
    classify_entity_type(token::String)

Classify the entity type of a biomedical token using simple heuristics.
In a full implementation, this would use the trained entity classifier.
"""
function classify_entity_type(token::String)
  token_lower = lowercase(token)

  # Simple rule-based classification
  if occursin(r"diabet|insulin|glucos", token_lower)
    return "DISEASE"
  elseif occursin(r"metformin|drug|medic", token_lower)
    return "DRUG"
  elseif occursin(r"protein|gene|dna", token_lower)
    return "PROTEIN"
  elseif occursin(r"heart|cardio|blood", token_lower)
    return "ANATOMY"
  elseif occursin(r"patient|clinical", token_lower)
    return "PROCEDURE"
  else
    return "BIOMARKER"  # Default biomedical entity type
  end
end

"""
    deduplicate_entities(entities::Vector{Entity})

Remove duplicate entities based on overlapping positions and similar text.
"""
function deduplicate_entities(entities::Vector{GraphMERT.Entity})
  if isempty(entities)
    return entities
  end

  # Sort by position
  sorted_entities = sort(entities, by=e -> e.position.start)

  deduplicated = Vector{GraphMERT.Entity}()
  for entity in sorted_entities
    # Check if this entity overlaps significantly with already included entities
    overlap = false
    for existing in deduplicated
      # Simple overlap check: if positions overlap by more than 50%
      overlap_start = max(entity.position.start, existing.position.start)
      overlap_end = min(entity.position.stop, existing.position.stop)
      if overlap_end > overlap_start
        overlap_length = overlap_end - overlap_start + 1
        entity_length = entity.position.stop - entity.position.start + 1
        if overlap_length / entity_length > 0.5
          overlap = true
          break
        end
      end
    end

    if !overlap
      push!(deduplicated, entity)
    end
  end

  return deduplicated
end

# Export functions for external use
export discover_head_entities,
  match_relations_for_entities,
  predict_tail_tokens,
  form_tail_from_tokens,
  filter_and_deduplicate_triples,
  extract_knowledge_graph,
  deduplicate_entities

# Deprecated: extract_biomedical_terms - use domain provider instead
# Deprecated: discover_head_entities_enhanced - use discover_head_entities with domain provider
# Deprecated: match_relations_for_entities_enhanced - use match_relations_for_entities with domain provider
# Deprecated: classify_entity_type - use domain provider's extract_entities instead
