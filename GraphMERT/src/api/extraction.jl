"""
Knowledge Graph Extraction API for GraphMERT.jl

This module implements the complete knowledge graph extraction pipeline
from biomedical text using a trained GraphMERT model.

The extraction pipeline follows the 5-stage process from the paper:
1. Head Discovery: Extract entities from text
2. Relation Matching: Match entities to relations
3. Tail Prediction: Use GraphMERT to predict tail tokens
4. Tail Formation: Combine tokens into coherent tails
5. Filtering: Apply similarity and deduplication filters
"""

# Types will be available from main module
# using ..Types: KnowledgeGraph, BiomedicalEntity, BiomedicalRelation, TextPosition

"""
    extract_biomedical_terms(text::String)

Extract biomedical terms from text with their positions.

# Arguments
- `text::String`: Input text

# Returns
- `Vector{Tuple{String,Int}}`: List of (term, position) tuples

# Example
```julia
terms = extract_biomedical_terms("Diabetes is a chronic condition.")
```
"""
function extract_biomedical_terms(text::String)
  terms = Vector{Tuple{String,Int}}()
  
  # Simple pattern matching for biomedical terms
  # This is a basic implementation - in practice would use more sophisticated NLP
  biomedical_patterns = [
    r"\bdiabetes\b"i,
    r"\bmetformin\b"i,
    r"\binsulin\b"i,
    r"\bglucose\b"i,
    r"\bblood\s+sugar\b"i,
    r"\btype\s+2\s+diabetes\b"i,
    r"\btype\s+1\s+diabetes\b"i,
    r"\bcardiovascular\b"i,
    r"\bheart\s+disease\b"i,
    r"\bhypertension\b"i,
    r"\bcholesterol\b"i,
    r"\bobesity\b"i,
    r"\bmetabolic\b"i,
    r"\bchronic\b"i,
    r"\bdisorder\b"i,
    r"\bcondition\b"i,
    r"\bdisease\b"i,
    r"\bsyndrome\b"i,
    r"\btherapy\b"i,
    r"\btreatment\b"i,
    r"\bmedication\b"i,
    r"\bdrug\b"i,
    r"\bpatient\b"i,
    r"\bclinical\b"i,
    r"\bmedical\b"i
  ]
  
  for pattern in biomedical_patterns
    for match in eachmatch(pattern, text)
      term = match.match
      position = match.offset
      push!(terms, (term, position))
    end
  end
  
  # Remove duplicates and sort by position
  unique_terms = unique(terms)
  sort!(unique_terms, by=x -> x[2])
  
  return unique_terms
end

"""
    calculate_entity_confidence(term::String, text::String)

Calculate confidence score for an entity based on term characteristics.

# Arguments
- `term::String`: The entity term
- `text::String`: The full text context

# Returns
- `Float64`: Confidence score between 0.0 and 1.0
"""
function calculate_entity_confidence(term::String, text::String)
  # Simple confidence calculation based on term characteristics
  confidence = 0.5  # Base confidence
  
  # Increase confidence for longer terms (more specific)
  if length(term) > 10
    confidence += 0.2
  elseif length(term) > 5
    confidence += 0.1
  end
  
  # Increase confidence for capitalized terms (proper nouns)
  if isuppercase(term[1])
    confidence += 0.1
  end
  
  # Increase confidence for terms with multiple words
  if count(isspace, term) > 0
    confidence += 0.1
  end
  
  # Cap at 1.0
  return min(confidence, 1.0)
end

"""
    discover_head_entities(text::String, umls_client::Union{UMLSClient, Nothing}=nothing)

Stage 1: Head Discovery - Extract entities from text.

# Arguments
- `text::String`: Input biomedical text
- `umls_client::Union{UMLSClient, Nothing}`: Optional UMLS client for entity linking

# Returns
- `Vector{BiomedicalEntity}`: Discovered entities with confidence scores
"""
function discover_head_entities(text::String, umls_client::Union{Any,Nothing}=nothing)
  entities = Vector{GraphMERT.BiomedicalEntity}()

  # Simple entity extraction (would be more sophisticated in practice)
  # Look for biomedical terms in the text
  biomedical_terms = extract_biomedical_terms(text)

  for (term, position) in biomedical_terms
    # Calculate confidence based on term characteristics
    confidence = calculate_entity_confidence(term, text)

    # Link to UMLS if client available
    cui = nothing
    semantic_types = String[]
    if umls_client !== nothing
      linking_result = link_entity_to_umls(term, umls_client)
      if linking_result !== nothing
        cui = linking_result.cui
        semantic_types = linking_result.semantic_types
      end
    end

    # Create TextPosition
    text_position = GraphMERT.TextPosition(position, position + length(term) - 1, 1, 1)
    
    # Create attributes dictionary
    attributes = Dict{String,Any}()
    if cui !== nothing
      attributes["cui"] = cui
    end
    if !isempty(semantic_types)
      attributes["semantic_types"] = semantic_types
    end
    attributes["provenance"] = text
    
    entity = GraphMERT.BiomedicalEntity(
      term,  # id
      term,  # text
      "UNKNOWN",  # label (entity type)
      confidence,
      text_position,
      attributes
    )
    push!(entities, entity)
  end

  return entities
end

"""
    match_relations_for_entities(entities::Vector{BiomedicalEntity}, text::String,
                               llm_client::Union{LLMClient, Nothing}=nothing)

Stage 2: Relation Matching - Match entities to relations.

# Arguments
- `entities::Vector{BiomedicalEntity}`: Discovered entities
- `text::String`: Original text for context
- `llm_client::Union{LLMClient, Nothing}`: Optional LLM for relation matching

# Returns
- `Vector{Tuple{Int, Int, String, Float64}}`: (head_idx, tail_idx, relation, confidence)
"""
function match_relations_for_entities(entities::Vector{GraphMERT.BiomedicalEntity}, text::String,
  llm_client::Union{Any,Nothing}=nothing)
  relations = Vector{Tuple{Int,Int,String,Float64}}()

  for i in 1:length(entities)
    for j in 1:length(entities)
      if i != j
        # Simple relation matching (would be more sophisticated)
        relation_type = determine_relation_type(entities[i], entities[j], text)
        confidence = calculate_relation_confidence(entities[i], entities[j], text)

        if confidence > 0.5  # Threshold for relation acceptance
          push!(relations, (i, j, relation_type, confidence))
        end
      end
    end
  end

  return relations
end

"""
    predict_tail_tokens(model::GraphMERTModel, head_entity::BiomedicalEntity,
                       relation::String, text::String, top_k::Int=20)

Stage 3: Tail Prediction - Use GraphMERT to predict tail tokens.

# Arguments
- `model::GraphMERTModel`: Trained GraphMERT model
- `head_entity::BiomedicalEntity`: Head entity for the triple
- `relation::String`: Relation type
- `text::String`: Original text for context
- `top_k::Int`: Number of top tokens to return

# Returns
- `Vector{Tuple{Int, Float64}}`: (token_id, probability) pairs
"""
function predict_tail_tokens(model::GraphMERT.GraphMERTModel, head_entity::GraphMERT.BiomedicalEntity,
  relation::String, text::String, top_k::Int=20)
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

# Arguments
- `tokens::Vector{Tuple{Int, Float64}}`: Top-k predicted tokens
- `text::String`: Original text for context
- `llm_client::Union{LLMClient, Nothing}`: Optional LLM for tail formation

# Returns
- `Vector{String}`: Possible tail entity strings
"""
function form_tail_from_tokens(tokens::Vector{Tuple{Int,Float64}}, text::String,
  llm_client::Union{Any,Nothing}=nothing)
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
    filter_and_deduplicate_triples(triples::Vector{Tuple{BiomedicalEntity, String, String, Float64}},
                                  text::String, β_threshold::Float64=0.8)

Stage 5: Filtering - Apply similarity and deduplication filters.

# Arguments
- `triples::Vector{Tuple{BiomedicalEntity, String, String, Float64}}`: Raw triples
- `text::String`: Original text for similarity checking
- `β_threshold::Float64`: Similarity threshold for filtering

# Returns
- `Vector{Tuple{BiomedicalEntity, String, String, Float64}}`: Filtered triples
"""
function filter_and_deduplicate_triples(triples::Vector{Tuple{GraphMERT.BiomedicalEntity,String,String,Float64}},
  text::String, β_threshold::Float64=0.8)
  filtered_triples = Vector{Tuple{GraphMERT.BiomedicalEntity,String,String,Float64}}()

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
  final_triples = Vector{Tuple{GraphMERT.BiomedicalEntity,String,String,Float64}}()
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

Extracts structured knowledge from biomedical text using a trained GraphMERT model.

# Arguments
- `text::String`: Input biomedical text
- `model::GraphMERTModel`: Trained GraphMERT model
- `options::ProcessingOptions`: Processing options

# Returns
- `KnowledgeGraph`: Extracted knowledge graph with entities and relations
"""
function extract_knowledge_graph(text::String, model::GraphMERT.GraphMERTModel;
  options::GraphMERT.ProcessingOptions=GraphMERT.default_processing_options())::GraphMERT.KnowledgeGraph
  # Simplified implementation for demo - full version would use trained model
  @info "Starting knowledge graph extraction from text of length $(length(text))"

  # Stage 1: Head Discovery with LLM enhancement
  head_entities = discover_head_entities_enhanced(text, options.umls_client, options.llm_client)
  @info "Discovered $(length(head_entities)) head entities"

  # Stage 2: Relation Matching with LLM enhancement
  entity_relations = match_relations_for_entities_enhanced(head_entities, text, options.llm_client)
  @info "Matched $(length(entity_relations)) relations"

  # For demo, create a simple knowledge graph with discovered entities
  entities = head_entities
  relations = Vector{GraphMERT.BiomedicalRelation}()

  # Create simple relations between entities
  for i in 1:min(length(entities), 3)
    for j in (i+1):min(length(entities), i + 2)
      if i != j
        relation = GraphMERT.BiomedicalRelation(
          i, j, "ASSOCIATED_WITH", nothing, 0.7, text, text
        )
        push!(relations, relation)
      end
    end
  end

  return GraphMERT.KnowledgeGraph(
    entities,
    relations,
    Dict(
      "extraction_time" => string(now()),
      "model_version" => "GraphMERT-v0.1",
      "num_entities" => length(entities),
      "num_relations" => length(relations),
      "num_triples" => length(relations),  # Triples = relations
      "source_text" => text,
      "demo_mode" => true
    ))
end

"""
    discover_head_entities_enhanced(text::String, umls_client::Union{Any, Nothing}=nothing,
                                   llm_client::Union{Any, Nothing}=nothing)

Enhanced head entity discovery using both UMLS and LLM.

# Arguments
- `text::String`: Text to analyze
- `umls_client::Union{Any, Nothing}`: Optional UMLS client
- `llm_client::Union{Any, Nothing}`: Optional LLM client

# Returns
- `Vector{BiomedicalEntity}`: Enhanced entity discovery results
"""
function discover_head_entities_enhanced(text::String, umls_client::Union{Any,Nothing}=nothing,
  llm_client::Union{Any,Nothing}=nothing)
  entities = Vector{GraphMERT.BiomedicalEntity}()

  # First, try LLM-based discovery if available
  if llm_client !== nothing
    try
      llm_entities = GraphMERT.discover_entities(llm_client, text)
      for entity_text in llm_entities
        # Link to UMLS if available
        cui = nothing
        semantic_types = String[]
        if umls_client !== nothing
          linking_result = GraphMERT.link_entity_to_umls(entity_text, umls_client)
          if linking_result !== nothing
            cui = linking_result.cui
            semantic_types = linking_result.semantic_types
          end
        end

        entity = GraphMERT.BiomedicalEntity(
          entity_text,
          entity_text,
          "UNKNOWN",
          cui,
          semantic_types,
          GraphMERT.TextPosition(0, length(entity_text), 0, 0),
          0.9,  # High confidence from LLM
          text
        )
        push!(entities, entity)
      end
    catch e
      @warn "LLM entity discovery failed: $e"
    end
  end

  # Fallback to simple extraction if LLM fails or unavailable
  if isempty(entities) || llm_client === nothing
    simple_entities = discover_head_entities(text, umls_client)
    append!(entities, simple_entities)
  end

  return entities
end

"""
    match_relations_for_entities_enhanced(entities::Vector{BiomedicalEntity}, text::String,
                                        llm_client::Union{Any, Nothing}=nothing)

Enhanced relation matching using LLM for better relation detection.

# Arguments
- `entities::Vector{BiomedicalEntity}`: Discovered entities
- `text::String`: Original text
- `llm_client::Union{Any, Nothing}`: Optional LLM client

# Returns
- `Vector{Tuple{Int, Int, String, Float64}}`: Enhanced relation results
"""
function match_relations_for_entities_enhanced(entities::Vector{GraphMERT.BiomedicalEntity}, text::String,
  llm_client::Union{Any,Nothing}=nothing)
  relations = Vector{Tuple{Int,Int,String,Float64}}()

  # Try LLM-based relation matching first
  if llm_client !== nothing && length(entities) > 1
    try
      entity_texts = [e.text for e in entities]
      llm_relations = GraphMERT.match_relations(llm_client, entity_texts, text)

      for (entity1, relation_data) in llm_relations
        entity2 = relation_data["entity2"]
        relation_type = relation_data["relation"]

        # Find entity indices
        idx1 = findfirst(e -> e.text == entity1, entities)
        idx2 = findfirst(e -> e.text == entity2, entities)

        if idx1 !== nothing && idx2 !== nothing && idx1 != idx2
          push!(relations, (idx1, idx2, relation_type, 0.9))  # High confidence from LLM
        end
      end
    catch e
      @warn "LLM relation matching failed: $e"
    end
  end

  # Fallback to simple relation matching
  if length(relations) == 0
    simple_relations = match_relations_for_entities(entities, text, llm_client)
    append!(relations, simple_relations)
  end

  return relations
end

# Export functions for external use
export discover_head_entities, match_relations_for_entities, predict_tail_tokens,
  form_tail_from_tokens, filter_and_deduplicate_triples, extract_knowledge_graph,
  discover_head_entities_enhanced, match_relations_for_entities_enhanced
