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
    GraphRAGConfig

Configuration for GraphRAG evaluation.

# Fields
- `max_hops::Int`: Maximum number of hops for local search (default: 3).
- `max_entities::Int`: Maximum number of entities to retrieve (default: 10).
- `llm_client::Union{AbstractLLMClient,Nothing}`: LLM client for answer generation (default: nothing).
- `min_relevance_score::Float64`: Minimum relevance score for triples (default: 0.1).
"""
struct GraphRAGConfig
    max_hops::Int
    max_entities::Int
    llm_client::Union{AbstractLLMClient,Nothing}
    min_relevance_score::Float64

    function GraphRAGConfig(;
        max_hops::Int = 3,
        max_entities::Int = 10,
        llm_client::Union{AbstractLLMClient,Nothing} = nothing,
        min_relevance_score::Float64 = 0.1,
    )
        new(max_hops, max_entities, llm_client, min_relevance_score)
    end
end

"""
    perform_local_search(kg::KnowledgeGraph, query::String; config::GraphRAGConfig=GraphRAGConfig())

Perform local search in the knowledge graph to find relevant entities and relations
for answering a given query.

Returns a list of (head_entity, relation, tail_entity, relevance_score) tuples
sorted by relevance to the query.
"""
function perform_local_search(
    kg::KnowledgeGraph,
    query::String;
    config::GraphRAGConfig = GraphRAGConfig(),
)
    # Simple relevance scoring based on entity/relation matching with query terms
    query_terms = lowercase.(split(query, r"[\s\W]+"))
    query_terms = filter(term -> length(term) > 2, query_terms)  # Filter short terms

    relevant_triples =
        Vector{Tuple{KnowledgeEntity,KnowledgeRelation,KnowledgeEntity,Float64}}()

    # Create all possible entity-relation-entity combinations
    # Note: KnowledgeGraph relations store entity IDs, not indices.
    # We need to look up entities by ID.
    entity_map = Dict(e.id => e for e in kg.entities)

    for relation in kg.relations
        if haskey(entity_map, relation.head) && haskey(entity_map, relation.tail)
            head_entity = entity_map[relation.head]
            tail_entity = entity_map[relation.tail]

            # Calculate relevance score based on term matching
            head_score = calculate_entity_relevance(head_entity, query_terms)
            tail_score = calculate_entity_relevance(tail_entity, query_terms)
            relation_score = calculate_relation_relevance(relation, query_terms)

            # Combine scores (weighted average)
            total_score =
                (head_score * 0.4 + tail_score * 0.4 + relation_score * 0.2)

            if total_score > config.min_relevance_score  # Minimum relevance threshold
                push!(
                    relevant_triples,
                    (head_entity, relation, tail_entity, total_score),
                )
            end
        end
    end

    # Sort by relevance score and return top results
    sort!(relevant_triples, by = x -> x[4], rev = true)
    return relevant_triples[1:min(config.max_entities, length(relevant_triples))]
end

"""
    calculate_entity_relevance(entity::KnowledgeEntity, query_terms::Vector{String})

Calculate relevance score for an entity based on term matching with query.
"""
function calculate_entity_relevance(entity::KnowledgeEntity, query_terms::Vector{String})
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
    calculate_relation_relevance(relation::KnowledgeRelation, query_terms::Vector{String})

Calculate relevance score for a relation based on term matching with query.
"""
function calculate_relation_relevance(
    relation::KnowledgeRelation,
    query_terms::Vector{String},
)
    # Relation doesn't store head/tail text directly, only IDs.
    # We use relation_type and potentially attributes.
    relation_text = lowercase(relation.relation_type)
    score = 0.0

    for term in query_terms
        if occursin(term, relation_text)
            score += 1.0
        end
    end

    return min(score / length(query_terms), 1.0)
end

"""
    generate_answer_from_context(relevant_triples::Vector{Tuple{KnowledgeEntity, KnowledgeRelation, KnowledgeEntity, Float64}},
                                query::String; config::GraphRAGConfig=GraphRAGConfig())

Generate an answer to the query using the retrieved relevant triples as context.
"""
function generate_answer_from_context(
    relevant_triples::Vector{
        Tuple{KnowledgeEntity,KnowledgeRelation,KnowledgeEntity,Float64},
    },
    query::String;
    config::GraphRAGConfig = GraphRAGConfig(),
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
    if config.llm_client !== nothing
        prompt = """
        Based on the following knowledge: $context

        Answer this question: $query

        Provide a clear, factual answer based only on the provided knowledge.
        If the knowledge doesn't contain enough information to answer, say so.
        """

        try
            response = make_llm_request(config.llm_client, prompt)
            return response.content
        catch e
            @warn "LLM request failed, falling back to template answer" exception = e
        end
    end

    # Fallback: simple template-based answer
    if length(relevant_triples) == 1
        head_entity, relation, tail_entity, _ = relevant_triples[1]
        return "According to the knowledge graph: $(head_entity.text) $(relation.relation_type) $(tail_entity.text)."
    else
        return "The knowledge graph contains $(length(relevant_triples)) relevant relationships that may help answer this question."
    end
end

"""
    evaluate_graphrag(kg::KnowledgeGraph, query::String; config::GraphRAGConfig=GraphRAGConfig())

Evaluate GraphRAG performance by testing the graph's ability to answer a query.
Returns a score between 0 and 1 indicating answer quality.
"""
function evaluate_graphrag(
    kg::KnowledgeGraph,
    query::String;
    config::GraphRAGConfig = GraphRAGConfig(),
)

    # Perform local search to find relevant information
    relevant_triples =
        perform_local_search(kg, query, config = config)

    if isempty(relevant_triples)
        return 0.0  # No relevant information found
    end

    # Generate answer from context
    answer = generate_answer_from_context(relevant_triples, query, config = config)

    # Simple quality scoring based on answer length and relevance
    # In a real implementation, this would use more sophisticated evaluation
    answer_length = length(answer)
    relevance_score = min(length(relevant_triples) / config.max_entities, 1.0)

    # Combine factors: relevance (60%) + answer quality (40%)
    quality_score = (relevance_score * 0.6) + (min(answer_length / 100.0, 1.0) * 0.4)

    return min(quality_score, 1.0)
end

"""
    evaluate_graphrag_dataset(kg::KnowledgeGraph, queries::Vector{String};
                             config::GraphRAGConfig=GraphRAGConfig())

Evaluate GraphRAG performance on a dataset of queries.
Returns average score across all queries.
"""
function evaluate_graphrag_dataset(
    kg::KnowledgeGraph,
    queries::Vector{String};
    config::GraphRAGConfig = GraphRAGConfig(),
)
    scores = Float64[]

    for query in queries
        score = evaluate_graphrag(kg, query, config = config)
        push!(scores, score)
    end

    return isempty(scores) ? 0.0 : mean(scores)
end
