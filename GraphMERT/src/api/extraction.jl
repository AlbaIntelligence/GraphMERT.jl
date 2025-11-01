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

# These functions have been moved to domain providers
# Use domain.extract_entities() and domain.calculate_entity_confidence() instead


"""
    match_relations_for_entities(entities::Vector{Entity}, text::String, domain::Any, options::ProcessingOptions)

Stage 2: Relation Matching - Match entities to relations using domain provider.

# Arguments
- `entities::Vector{Entity}`: Extracted entities
- `text::String`: Input text
- `domain::DomainProvider`: Domain provider instance
- `options::ProcessingOptions`: Processing options

# Returns
- `Vector{Relation}`: Extracted relations
"""
function match_relations_for_entities(
  entities::Vector{GraphMERT.Entity},
  text::String,
  domain::Any,
  options::GraphMERT.ProcessingOptions = GraphMERT.default_processing_options(),
)
  # Use domain provider for relation extraction
  try
    relations = GraphMERT.extract_relations(domain, entities, text, options)
    return relations
  catch e
    @warn "Domain relation extraction failed: $e, falling back to simple extraction"
    # Fallback: create empty relations list
    return Vector{GraphMERT.Relation}()
  end
end

# These functions have been moved to domain providers
# Use domain.validate_relation() and domain.calculate_relation_confidence() instead

"""
    predict_tail_tokens(model::GraphMERTModel, head_entity::Entity,
                       relation::String, text::String, top_k::Int=20)

Stage 3: Tail Prediction - Use GraphMERT to predict tail tokens.

"""
function predict_tail_tokens(
  model::GraphMERT.GraphMERTModel,
  head_entity::GraphMERT.Entity,
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
    filter_and_deduplicate_triples(triples::Vector{Tuple{Entity, String, String, Float64}},
                                  text::String, β_threshold::Float64=0.8)

Stage 5: Filtering - Apply similarity and deduplication filters.

"""
function filter_and_deduplicate_triples(
  triples::Vector{Tuple{GraphMERT.Entity,String,String,Float64}},
  text::String,
  β_threshold::Float64=0.8,
)
  filtered_triples = Vector{Tuple{GraphMERT.Entity,String,String,Float64}}()

  # Remove duplicates
  seen = Set{String}()
  for (head, relation, tail, confidence) in triples
    triple_key = "$(head.text)_$relation_$(tail)"
    if !(triple_key in seen)
      push!(seen, triple_key)
      push!(filtered_triples, (head, relation, tail, confidence))
    end
  end

  # Filter by similarity threshold
  final_triples = Vector{Tuple{GraphMERT.Entity,String,String,Float64}}()
  for (head, relation, tail, confidence) in filtered_triples
    similarity = calculate_tail_similarity(tail, text)
    if similarity ≥ β_threshold
      push!(final_triples, (head, relation, tail, confidence * similarity))
    end
  end

  return final_triples
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
  model::GraphMERT.GraphMERTModel;
  options::GraphMERT.ProcessingOptions=GraphMERT.default_processing_options(),
)::GraphMERT.KnowledgeGraph
  @info "Starting knowledge graph extraction from text of length $(length(text)) with domain: $(options.domain)"

  # Get domain provider from registry
  domain_provider = GraphMERT.get_domain(options.domain)
  if domain_provider === nothing
    @error "Domain '$(options.domain)' not found. Available domains: $(GraphMERT.list_domains())"
    error("Domain '$(options.domain)' not registered")
  end

  # Stage 1: Head Discovery using domain provider
  entities = discover_head_entities(text, domain_provider, options)
  @info "Discovered $(length(entities)) entities using domain: $(options.domain)"

  # Stage 2: Relation Matching using domain provider
  relations = match_relations_for_entities(entities, text, domain_provider, options)
  @info "Extracted $(length(relations)) relations using domain: $(options.domain)"

  # Create knowledge graph
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
    ),
  )
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
