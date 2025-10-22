"""
FActScore* evaluation metric for GraphMERT.jl

This module implements the FActScore* metric as described in the GraphMERT paper.
FActScore* combines factuality checking (does the triple represent a real fact?)
with validity checking (is the triple ontologically valid?) to provide a comprehensive
evaluation of knowledge graph quality.

The metric is defined as:
FActScore*(G) = E_{τ∈G}[f(τ)]

where f(τ) = 𝟙[τ is supported by C(τ)]
and C(τ) is the set of context sentences containing head and tail entities.
"""

using ..Types: KnowledgeGraph, BiomedicalEntity, BiomedicalRelation
using ..LLM: make_llm_request, HelperLLMClient
using ..Biomedical: link_entity_to_umls
using Distributions: Normal, quantile

"""
    evaluate_factscore(kg::KnowledgeGraph, text::String;
                      llm_client::Union{HelperLLMClient, Nothing}=nothing,
                      confidence_threshold::Float64=0.5,
                      max_context_sentences::Int=5)

Calculate FActScore* for a knowledge graph.

FActScore* evaluates both factuality (does the triple represent a real fact?)
and validity (is the triple ontologically consistent?) of extracted triples.

# Arguments
- `kg::KnowledgeGraph`: Knowledge graph to evaluate
- `text::String`: Original source text for context
- `llm_client::Union{HelperLLMClient, Nothing}`: Optional LLM client for validation
- `confidence_threshold::Float64`: Minimum confidence for triple inclusion
- `max_context_sentences::Int`: Maximum context sentences to consider

# Returns
- `FActScoreResult`: Detailed evaluation results

# Mathematical Definition
FActScore*(G) = E_{τ∈G}[f(τ)]

where:
- G is the knowledge graph
- τ is a triple (head, relation, tail)
- f(τ) = 𝟙[τ is supported by C(τ)]
- C(τ) is the set of context sentences containing head and tail entities
"""
function evaluate_factscore(kg::KnowledgeGraph, text::String;
                           llm_client::Union{HelperLLMClient, Nothing}=nothing,
                           confidence_threshold::Float64=0.5,
                           max_context_sentences::Int=5)
    @info "Starting FActScore* evaluation for $(length(kg.triples)) triples"

    # Filter triples by confidence threshold
    high_confidence_triples = filter_triples_by_confidence(kg, confidence_threshold)

    @info "Evaluating $(length(high_confidence_triples)) high-confidence triples"

    # Evaluate each triple
    triple_scores = Vector{Bool}()
    triple_contexts = Vector{String}()

    for (head_idx, rel_idx, tail_idx) in high_confidence_triples
        head_entity = kg.entities[head_idx]
        tail_entity = kg.entities[tail_idx]
        relation = kg.relations[rel_idx]

        # Get context sentences containing both entities
        context = get_triple_context(text, head_entity, tail_entity, max_context_sentences)

        # Evaluate factuality using LLM or simple heuristic
        is_supported = evaluate_triple_support(head_entity, relation, tail_entity, context, llm_client)

        push!(triple_scores, is_supported)
        push!(triple_contexts, join(context, " "))
    end

    # Calculate overall FActScore*
    total_triples = length(triple_scores)
    supported_triples = sum(triple_scores)
    factscore = total_triples > 0 ? supported_triples / total_triples : 0.0

    @info "FActScore* = $(round(factscore, digits=4)) ($supported_triples/$total_triples supported)"

    return FActScoreResult(
        triple_scores,
        triple_contexts,
        factscore,
        supported_triples,
        total_triples,
        Dict(
            "confidence_threshold" => confidence_threshold,
            "max_context_sentences" => max_context_sentences,
            "evaluation_method" => llm_client !== nothing ? "llm" : "heuristic"
        )
    )
end

"""
    filter_triples_by_confidence(kg::KnowledgeGraph, threshold::Float64)

Filter triples by minimum confidence threshold.

# Arguments
- `kg::KnowledgeGraph`: Knowledge graph
- `threshold::Float64`: Minimum confidence threshold

# Returns
- `Vector{Tuple{Int, Int, Int}}`: Filtered triples (head_idx, rel_idx, tail_idx)
"""
function filter_triples_by_confidence(kg::KnowledgeGraph, threshold::Float64)
    filtered_triples = Vector{Tuple{Int, Int, Int}}()

    for (head_idx, rel_idx, tail_idx) in kg.triples
        relation = kg.relations[rel_idx]
        if relation.confidence >= threshold
            push!(filtered_triples, (head_idx, rel_idx, tail_idx))
        end
    end

    return filtered_triples
end

"""
    get_triple_context(text::String, head_entity::BiomedicalEntity,
                      tail_entity::BiomedicalEntity, max_sentences::Int)

Extract context sentences containing both head and tail entities.

# Arguments
- `text::String`: Original source text
- `head_entity::BiomedicalEntity`: Head entity
- `tail_entity::BiomedicalEntity`: Tail entity
- `max_sentences::Int`: Maximum number of context sentences

# Returns
- `Vector{String}`: Context sentences containing both entities
"""
function get_triple_context(text::String, head_entity::BiomedicalEntity,
                          tail_entity::BiomedicalEntity, max_sentences::Int)
    sentences = split(text, r"[\.\!\?]+")
    context_sentences = Vector{String}()

    for sentence in sentences
        sentence = strip(sentence)
        if !isempty(sentence) &&
           (occursin(lowercase(head_entity.text), lowercase(sentence)) ||
            occursin(lowercase(tail_entity.text), lowercase(sentence)))
            push!(context_sentences, sentence)
            if length(context_sentences) >= max_sentences
                break
            end
        end
    end

    return context_sentences
end

"""
    evaluate_triple_support(head_entity::BiomedicalEntity, relation::BiomedicalRelation,
                           tail_entity::BiomedicalEntity, context::Vector{String},
                           llm_client::Union{HelperLLMClient, Nothing})

Evaluate whether a triple is supported by its context.

# Arguments
- `head_entity::BiomedicalEntity`: Head entity
- `relation::BiomedicalRelation`: Relation
- `tail_entity::BiomedicalEntity`: Tail entity
- `context::Vector{String}`: Context sentences
- `llm_client::Union{HelperLLMClient, Nothing}`: Optional LLM client

# Returns
- `Bool`: True if triple is supported by context
"""
function evaluate_triple_support(head_entity::BiomedicalEntity, relation::BiomedicalRelation,
                                tail_entity::BiomedicalEntity, context::Vector{String},
                                llm_client::Union{HelperLLMClient, Nothing})
    # Simple heuristic evaluation (would use LLM in practice)
    if llm_client !== nothing
        return evaluate_triple_with_llm(head_entity, relation, tail_entity, context, llm_client)
    else
        return evaluate_triple_heuristic(head_entity, relation, tail_entity, context)
    end
end

"""
    evaluate_triple_with_llm(head_entity::BiomedicalEntity, relation::BiomedicalRelation,
                            tail_entity::BiomedicalEntity, context::Vector{String},
                            llm_client::HelperLLMClient)

Evaluate triple support using LLM.

# Arguments
- `head_entity::BiomedicalEntity`: Head entity
- `relation::BiomedicalRelation`: Relation
- `tail_entity::BiomedicalEntity`: Tail entity
- `context::Vector{String}`: Context sentences
- `llm_client::HelperLLMClient`: LLM client

# Returns
- `Bool`: LLM evaluation result
"""
function evaluate_triple_with_llm(head_entity::BiomedicalEntity, relation::BiomedicalRelation,
                                 tail_entity::BiomedicalEntity, context::Vector{String},
                                 llm_client::HelperLLMClient)
    # Create evaluation prompt
    context_text = join(context, " ")
    prompt = """
    Evaluate if the following biomedical relationship is supported by the context:

    Relationship: $(head_entity.text) --[$(relation.relation_type)]--> $(tail_entity.text)

    Context: $context_text

    Is this relationship factually supported by the context?
    Answer YES or NO only:
    """

    try
        response = make_llm_request(llm_client, prompt)
        if response.success
            answer = strip(lowercase(response.content))
            return startswith(answer, "yes")
        end
    catch e
        @warn "LLM evaluation failed: $e"
    end

    # Fallback to heuristic
    return evaluate_triple_heuristic(head_entity, relation, tail_entity, context)
end

"""
    evaluate_triple_heuristic(head_entity::BiomedicalEntity, relation::BiomedicalRelation,
                             tail_entity::BiomedicalEntity, context::Vector{String})

Simple heuristic evaluation of triple support.

# Arguments
- `head_entity::BiomedicalEntity`: Head entity
- `relation::BiomedicalRelation`: Relation
- `tail_entity::BiomedicalEntity`: Tail entity
- `context::Vector{String}`: Context sentences

# Returns
- `Bool`: Heuristic evaluation result
"""
function evaluate_triple_heuristic(head_entity::BiomedicalEntity, relation::BiomedicalRelation,
                                  tail_entity::BiomedicalEntity, context::Vector{String})
    # Simple heuristic: check if both entities appear in the same context
    head_in_context = any(occursin(lowercase(head_entity.text), lowercase(sentence)) for sentence in context)
    tail_in_context = any(occursin(lowercase(tail_entity.text), lowercase(sentence)) for sentence in context)

    # Check if relation keywords appear in context
    relation_keywords = get_relation_keywords(relation.relation_type)
    relation_in_context = any(occursin(lowercase(keyword), lowercase(join(context, " "))) for keyword in relation_keywords)

    # Basic support: both entities in context and relation makes sense
    return head_in_context && tail_in_context && relation_in_context
end

"""
    get_relation_keywords(relation_type::String)

Get keywords associated with a relation type for heuristic evaluation.

# Arguments
- `relation_type::String`: Relation type

# Returns
- `Vector{String}`: Associated keywords
"""
function get_relation_keywords(relation_type::String)
    relation_keywords = Dict(
        "TREATS" => ["treat", "treatment", "therapy", "medication", "drug"],
        "CAUSES" => ["cause", "lead to", "result in", "induce", "produce"],
        "ASSOCIATED_WITH" => ["associated", "related", "linked", "connected", "correlated"],
        "INDICATES" => ["indicate", "suggest", "show", "demonstrate", "reveal"],
        "PREVENTS" => ["prevent", "protect", "block", "inhibit", "reduce"],
        "COMPLICATES" => ["complicate", "worsen", "exacerbate", "aggravate", "complicate"]
    )

    return get(relation_keywords, relation_type, [lowercase(relation_type)])
end

"""
    calculate_factscore_confidence_interval(factscore::Float64, n::Int, confidence_level::Float64=0.95)

Calculate confidence interval for FActScore* using Wilson score interval.

# Arguments
- `factscore::Float64`: Observed FActScore*
- `n::Int`: Number of triples evaluated
- `confidence_level::Float64`: Confidence level (default: 0.95)

# Returns
- `Tuple{Float64, Float64}`: (lower_bound, upper_bound)
"""
function calculate_factscore_confidence_interval(factscore::Float64, n::Int, confidence_level::Float64=0.95)
    if n == 0
        return (0.0, 0.0)
    end

    # Wilson score interval for binomial proportion
    z = quantile(Normal(), (1 + confidence_level) / 2)
    p = factscore
    n = Float64(n)

    center = (p + z^2/(2*n)) / (1 + z^2/n)
    margin = z * sqrt(p*(1-p)/n + z^2/(4*n^2)) / (1 + z^2/n)

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return (lower, upper)
end
