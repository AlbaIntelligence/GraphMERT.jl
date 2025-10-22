"""
ValidityScore evaluation metric for GraphMERT.jl

This module implements the ValidityScore metric as described in the GraphMERT paper.
ValidityScore evaluates the ontological validity of extracted triples by checking
whether they align with established biomedical knowledge bases.

The metric is defined as:
ValidityScore = count(yes) / total_triples

where "yes" indicates that a triple is ontologically valid according to
established biomedical knowledge.
"""

# Types are defined in the main GraphMERT module
# using ..LLM: make_llm_request, HelperLLMClient
# using ..Biomedical: link_entity_to_umls
using Distributions: Normal, quantile

"""
    ValidityScoreResult

Result of ValidityScore evaluation containing detailed metrics.
"""
struct ValidityScoreResult
  triple_validity::Vector{Symbol}
  justifications::Vector{String}
  validity_score::Float64
  valid_triples::Int
  total_triples::Int
  metadata::Dict{String,Any}
end

# Use filter_triples_by_confidence from factscore.jl

"""
    evaluate_validity(kg::KnowledgeGraph;
                     llm_client::Union{HelperLLMClient, Nothing}=nothing,
                     umls_client::Union{UMLSClient, Nothing}=nothing,
                     confidence_threshold::Float64=0.5)

Calculate ValidityScore for a knowledge graph.

ValidityScore evaluates whether extracted triples are ontologically valid
by checking alignment with established biomedical knowledge.

# Arguments
- `kg::KnowledgeGraph`: Knowledge graph to evaluate
- `llm_client::Union{HelperLLMClient, Nothing}`: Optional LLM client for validation
- `umls_client::Union{UMLSClient, Nothing}`: Optional UMLS client for ontology checking
- `confidence_threshold::Float64`: Minimum confidence for triple inclusion

# Returns
- `ValidityScoreResult`: Detailed evaluation results

# Mathematical Definition
ValidityScore = count(yes) / total_triples

where "yes" indicates ontological validity of the triple.
"""
function evaluate_validity(kg::GraphMERT.KnowledgeGraph;
  llm_client::Union{HelperLLMClient,Nothing}=nothing,
  umls_client::Union{UMLSClient,Nothing}=nothing,
  confidence_threshold::Float64=0.5)
  @info "Starting ValidityScore evaluation for $(length(kg.entities)) entities and $(length(kg.relations)) relations"

  # Filter triples by confidence threshold
  high_confidence_triples = filter_triples_by_confidence(kg, confidence_threshold)

  @info "Evaluating $(length(high_confidence_triples)) high-confidence triples"

  # Evaluate each triple
  triple_validity = Vector{Symbol}()
  justifications = Vector{String}()

  for (head_idx, rel_idx, tail_idx) in high_confidence_triples
    head_entity = kg.entities[head_idx]
    tail_entity = kg.entities[tail_idx]
    relation = kg.relations[rel_idx]

    # Evaluate ontological validity
    is_valid, justification = evaluate_triple_validity(
      head_entity, relation, tail_entity, llm_client, umls_client
    )

    push!(triple_validity, is_valid)
    push!(justifications, justification)
  end

  # Calculate overall ValidityScore
  total_triples = length(triple_validity)
  valid_triples = count(v -> v == :yes, triple_validity)
  validity_score = total_triples > 0 ? valid_triples / total_triples : 0.0

  @info "ValidityScore = $(round(validity_score, digits=4)) ($valid_triples/$total_triples valid)"

  return ValidityScoreResult(
    triple_validity,
    justifications,
    validity_score,
    valid_triples,
    total_triples,
    Dict(
      "confidence_threshold" => confidence_threshold,
      "valid_count" => valid_triples,
      "maybe_count" => count(v -> v == :maybe, triple_validity),
      "no_count" => count(v -> v == :no, triple_validity)
    )
  )
end

"""
    evaluate_triple_validity(head_entity::BiomedicalEntity, relation::BiomedicalRelation,
                            tail_entity::BiomedicalEntity,
                            llm_client::Union{HelperLLMClient, Nothing},
                            umls_client::Union{UMLSClient, Nothing})

Evaluate the ontological validity of a triple.

# Arguments
- `head_entity::BiomedicalEntity`: Head entity
- `relation::BiomedicalRelation`: Relation
- `tail_entity::BiomedicalEntity`: Tail entity
- `llm_client::Union{HelperLLMClient, Nothing}`: Optional LLM client
- `umls_client::Union{UMLSClient, Nothing}`: Optional UMLS client

# Returns
- `Tuple{Symbol, String}`: (validity, justification)
"""
function evaluate_triple_validity(head_entity::BiomedicalEntity, relation::BiomedicalRelation,
  tail_entity::BiomedicalEntity,
  llm_client::Union{HelperLLMClient,Nothing},
  umls_client::Union{UMLSClient,Nothing})
  # Try LLM-based evaluation first if available
  if llm_client !== nothing
    validity, justification = evaluate_triple_with_llm(
      head_entity, relation, tail_entity, llm_client
    )
    if validity !== :unknown
      return validity, justification
    end
  end

  # Fallback to UMLS-based evaluation if available
  if umls_client !== nothing
    validity, justification = evaluate_triple_with_umls(
      head_entity, relation, tail_entity, umls_client
    )
    if validity !== :unknown
      return validity, justification
    end
  end

  # Final fallback to heuristic evaluation
  return evaluate_triple_heuristic_validity(head_entity, relation, tail_entity)
end

"""
    evaluate_triple_with_llm(head_entity::BiomedicalEntity, relation::BiomedicalRelation,
                            tail_entity::BiomedicalEntity, llm_client::HelperLLMClient)

Evaluate triple validity using LLM.

# Arguments
- `head_entity::BiomedicalEntity`: Head entity
- `relation::BiomedicalRelation`: Relation
- `tail_entity::BiomedicalEntity`: Tail entity
- `llm_client::HelperLLMClient`: LLM client

# Returns
- `Tuple{Symbol, String}`: (validity, justification)
"""
function evaluate_triple_with_llm(head_entity::BiomedicalEntity, relation::BiomedicalRelation,
  tail_entity::BiomedicalEntity, llm_client::HelperLLMClient)
  prompt = """
  Evaluate if the following biomedical relationship is ontologically valid:

  Relationship: $(head_entity.text) --[$(relation.relation_type)]--> $(tail_entity.text)

  Is this relationship semantically valid in biomedical knowledge?
  Consider:
  - Are the entity types compatible with the relation?
  - Does this relationship make sense in medical context?
  - Are there any contradictions with established knowledge?

  Answer YES, MAYBE, or NO with a brief justification:
  """

  try
    response = make_llm_request(llm_client, prompt)
    if response.success
      answer = strip(lowercase(response.content))
      if startswith(answer, "yes")
        return :yes, "LLM validation: semantically valid"
      elseif startswith(answer, "no")
        return :no, "LLM validation: semantically invalid"
      else
        return :maybe, "LLM validation: uncertain validity"
      end
    end
  catch e
    @warn "LLM validity evaluation failed: $e"
  end

  return :unknown, "LLM evaluation failed"
end

"""
    evaluate_triple_with_umls(head_entity::BiomedicalEntity, relation::BiomedicalRelation,
                             tail_entity::BiomedicalEntity, umls_client::UMLSClient)

Evaluate triple validity using UMLS ontology.

# Arguments
- `head_entity::BiomedicalEntity`: Head entity
- `relation::BiomedicalRelation`: Relation
- `tail_entity::BiomedicalEntity`: Tail entity
- `umls_client::UMLSClient`: UMLS client

# Returns
- `Tuple{Symbol, String}`: (validity, justification)
"""
function evaluate_triple_with_umls(head_entity::BiomedicalEntity, relation::BiomedicalRelation,
  tail_entity::BiomedicalEntity, umls_client::UMLSClient)
  # Check if entities are linked to UMLS
  head_cui = head_entity.cui
  tail_cui = tail_entity.cui

  if head_cui === nothing || tail_cui === nothing
    return :unknown, "Entities not linked to UMLS"
  end

  # Get semantic types
  head_types = get_entity_semantic_types(umls_client, head_cui)
  tail_types = get_entity_semantic_types(umls_client, tail_cui)

  # Check ontological compatibility
  is_valid = check_ontological_compatibility(head_types, relation.relation_type, tail_types)

  if is_valid
    return :yes, "UMLS validation: ontologically compatible"
  else
    return :no, "UMLS validation: ontologically incompatible"
  end
end

"""
    check_ontological_compatibility(head_types::Vector{String}, relation::String,
                                   tail_types::Vector{String})

Check if entity types are compatible with the relation.

# Arguments
- `head_types::Vector{String}`: Head entity semantic types
- `relation::String`: Relation type
- `tail_types::Vector{String}`: Tail entity semantic types

# Returns
- `Bool`: True if compatible, false otherwise
"""
function check_ontological_compatibility(head_types::Vector{String}, relation::String,
  tail_types::Vector{String})
  # Simple compatibility rules (would be more sophisticated in practice)
  relation_compatibility = Dict(
    "TREATS" => (["Disease", "Pharmacologic Substance"], ["Disease"]),
    "CAUSES" => (["Disease", "Finding"], ["Disease", "Finding"]),
    "ASSOCIATED_WITH" => (["*"], ["*"]),  # Very permissive
    "INDICATES" => (["Finding", "Laboratory Procedure"], ["Disease", "Finding"]),
    "PREVENTS" => (["Pharmacologic Substance", "Therapeutic Procedure"], ["Disease"])
  )

  compatible = get(relation_compatibility, relation, ([], []))
  head_compatible, tail_compatible = compatible

  head_ok = "*" in head_compatible || any(t in head_compatible for t in head_types)
  tail_ok = "*" in tail_compatible || any(t in tail_compatible for t in tail_types)

  return head_ok && tail_ok
end

"""
    evaluate_triple_heuristic_validity(head_entity::BiomedicalEntity, relation::BiomedicalRelation,
                                      tail_entity::BiomedicalEntity)

Simple heuristic evaluation of triple validity.

# Arguments
- `head_entity::BiomedicalEntity`: Head entity
- `relation::BiomedicalRelation`: Relation
- `tail_entity::BiomedicalEntity`: Tail entity

# Returns
- `Tuple{Symbol, String}`: (validity, justification)
"""
function evaluate_triple_heuristic_validity(head_entity::BiomedicalEntity, relation::BiomedicalRelation,
  tail_entity::BiomedicalEntity)
  # Simple heuristic based on entity and relation types
  head_type = lowercase(head_entity.label)
  tail_type = lowercase(tail_entity.label)
  relation_type = lowercase(relation.relation_type)

  # Basic validity rules
  if relation_type == "treats" && head_type == "drug" && tail_type == "disease"
    return :yes, "Heuristic: drug treats disease"
  elseif relation_type == "causes" && head_type == "disease" && tail_type == "disease"
    return :yes, "Heuristic: disease causes disease"
  elseif relation_type == "associated_with"  # Very permissive
    return :maybe, "Heuristic: general association"
  else
    return :no, "Heuristic: incompatible types for relation"
  end
end

"""
    calculate_validity_confidence_interval(validity_score::Float64, n::Int, confidence_level::Float64=0.95)

Calculate confidence interval for ValidityScore using Wilson score interval.

# Arguments
- `validity_score::Float64`: Observed ValidityScore
- `n::Int`: Number of triples evaluated
- `confidence_level::Float64`: Confidence level (default: 0.95)

# Returns
- `Tuple{Float64, Float64}`: (lower_bound, upper_bound)
"""
function calculate_validity_confidence_interval(validity_score::Float64, n::Int, confidence_level::Float64=0.95)
  if n == 0
    return (0.0, 0.0)
  end

  # Wilson score interval for binomial proportion
  z = quantile(Normal(), (1 + confidence_level) / 2)
  p = validity_score
  n = Float64(n)

  center = (p + z^2 / (2 * n)) / (1 + z^2 / n)
  margin = z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / (1 + z^2 / n)

  lower = max(0.0, center - margin)
  upper = min(1.0, center + margin)

  return (lower, upper)
end
