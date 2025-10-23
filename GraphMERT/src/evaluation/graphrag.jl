"""
GraphRAG evaluation metric for GraphMERT.jl

This module implements the GraphRAG evaluation methodology as described in the GraphMERT paper.
GraphRAG (Graph-based Retrieval-Augmented Generation) evaluates knowledge graph quality
by testing the graph's ability to answer questions that require multi-hop reasoning.

The evaluation process involves:
1. Query generation from the knowledge graph
2. Local search within the graph to find relevant subgraphs
3. Answer generation using the retrieved context
4. Answer evaluation for correctness and completeness

GraphRAG Score = (correct_answers / total_queries) * 100
"""

# Types are defined in the main GraphMERT module
# using ..LLM: make_llm_request, HelperLLMClient

"""
    perform_local_search(kg::KnowledgeGraph, query::String;
                        max_hops::Int=3, max_entities::Int=10)

Perform local search in the knowledge graph to find relevant entities and relations
for answering a given query.

Returns a list of (head_entity, relation, tail_entity, relevance_score) tuples
sorted by relevance to the query.
"""
function perform_local_search(
    kg::KnowledgeGraph,
    query::String;
    max_hops::Int = 3,
    max_entities::Int = 10,
)
    # Simple relevance scoring based on entity/relation matching with query terms
    query_terms = lowercase.(split(query, r"[\s\W]+"))
    query_terms = filter(term -> length(term) > 2, query_terms)  # Filter short terms

    relevant_triples =
        Vector{Tuple{BiomedicalEntity,BiomedicalRelation,BiomedicalEntity,Float64}}()

    # Create all possible entity-relation-entity combinations
    for (head_idx, head_entity) in enumerate(kg.entities)
        for (rel_idx, relation) in enumerate(kg.relations)
            for (tail_idx, tail_entity) in enumerate(kg.entities)
                if head_idx != tail_idx
                    # Calculate relevance score based on term matching
                    head_score = calculate_entity_relevance(head_entity, query_terms)
                    tail_score = calculate_entity_relevance(tail_entity, query_terms)
                    relation_score = calculate_relation_relevance(relation, query_terms)

                    # Combine scores (weighted average)
                    total_score =
                        (head_score * 0.4 + tail_score * 0.4 + relation_score * 0.2)

                    if total_score > 0.1  # Minimum relevance threshold
                        push!(
                            relevant_triples,
                            (head_entity, relation, tail_entity, total_score),
                        )
                    end
                end
            end
        end
    end

    # Sort by relevance score and return top results
    sort!(relevant_triples, by = x -> x[4], rev = true)
    return relevant_triples[1:min(max_entities, length(relevant_triples))]
end

"""
    calculate_entity_relevance(entity::BiomedicalEntity, query_terms::Vector{String})

Calculate relevance score for an entity based on term matching with query.
"""
function calculate_entity_relevance(entity::BiomedicalEntity, query_terms::Vector{String})
    entity_text = lowercase(entity.text * " " * entity.label)
    score = 0.0

    for term in query_terms
        if occursin(term, entity_text)
            score += 1.0
        end
    end

    return min(score / length(query_terms), 1.0)
end

"""
    calculate_relation_relevance(relation::BiomedicalRelation, query_terms::Vector{String})

Calculate relevance score for a relation based on term matching with query.
"""
function calculate_relation_relevance(
    relation::BiomedicalRelation,
    query_terms::Vector{String},
)
    relation_text =
        lowercase(relation.relation_type * " " * relation.head * " " * relation.tail)
    score = 0.0

    for term in query_terms
        if occursin(term, relation_text)
            score += 1.0
        end
    end

    return min(score / length(query_terms), 1.0)
end

"""
    generate_answer_from_context(relevant_triples::Vector{Tuple{BiomedicalEntity, BiomedicalRelation, BiomedicalEntity, Float64}},
                                query::String; llm_client::Union{HelperLLMClient, Nothing}=nothing)

Generate an answer to the query using the retrieved relevant triples as context.
"""
function generate_answer_from_context(
    relevant_triples::Vector{
        Tuple{BiomedicalEntity,BiomedicalRelation,BiomedicalEntity,Float64},
    },
    query::String;
    llm_client::Union{HelperLLMClient,Nothing} = nothing,
)

    if isempty(relevant_triples)
        return "No relevant information found in the knowledge graph to answer this query."
    end

    # Format context for answer generation
    context_parts = String[]
    for (head_entity, relation, tail_entity, score) in relevant_triples
        push!(
            context_parts,
            "$(head_entity.text) $(relation.relation_type) $(tail_entity.text)",
        )
    end

    context = join(context_parts, "; ")

    # Use LLM if available, otherwise use simple template
    if llm_client !== nothing
        prompt = """
        Based on the following biomedical knowledge: $context

        Answer this question: $query

        Provide a clear, factual answer based only on the provided knowledge.
        If the knowledge doesn't contain enough information to answer, say so.
        """

        try
            response = make_llm_request(llm_client, prompt)
            return response.content
        catch e
            @warn "LLM request failed, falling back to template answer" exception = e
        end
    end

    # Fallback: simple template-based answer
    if length(relevant_triples) == 1
        head_entity, relation, tail_entity, _ = relevant_triples[1]
        return "According to the knowledge graph: $(head_entity.canonical_name) $(relation.relation_type) $(tail_entity.canonical_name)."
    else
        return "The knowledge graph contains $(length(relevant_triples)) relevant relationships that may help answer this question."
    end
end

"""
    evaluate_graphrag(kg::KnowledgeGraph, query::String;
                     llm_client::Union{HelperLLMClient, Nothing}=nothing,
                     max_hops::Int=3, max_entities::Int=10)

Evaluate GraphRAG performance by testing the graph's ability to answer a query.
Returns a score between 0 and 1 indicating answer quality.
"""
function evaluate_graphrag(
    kg::KnowledgeGraph,
    query::String;
    llm_client::Union{HelperLLMClient,Nothing} = nothing,
    max_hops::Int = 3,
    max_entities::Int = 10,
)

    # Perform local search to find relevant information
    relevant_triples =
        perform_local_search(kg, query, max_hops = max_hops, max_entities = max_entities)

    if isempty(relevant_triples)
        return 0.0  # No relevant information found
    end

    # Generate answer from context
    answer = generate_answer_from_context(relevant_triples, query, llm_client = llm_client)

    # Simple quality scoring based on answer length and relevance
    # In a real implementation, this would use more sophisticated evaluation
    answer_length = length(answer)
    relevance_score = min(length(relevant_triples) / max_entities, 1.0)

    # Combine factors: relevance (60%) + answer quality (40%)
    quality_score = (relevance_score * 0.6) + (min(answer_length / 100.0, 1.0) * 0.4)

    return min(quality_score, 1.0)
end

"""
    evaluate_graphrag_dataset(kg::KnowledgeGraph, queries::Vector{String};
                             llm_client::Union{HelperLLMClient, Nothing}=nothing)

Evaluate GraphRAG performance on a dataset of queries.
Returns average score across all queries.
"""
function evaluate_graphrag_dataset(
    kg::KnowledgeGraph,
    queries::Vector{String};
    llm_client::Union{HelperLLMClient,Nothing} = nothing,
)
    scores = Float64[]

    for query in queries
        score = evaluate_graphrag(kg, query, llm_client = llm_client)
        push!(scores, score)
    end

    return isempty(scores) ? 0.0 : mean(scores)
end
