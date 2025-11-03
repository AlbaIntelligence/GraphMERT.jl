"""
Apply filtering to knowledge graph based on criteria.
"""
function apply_filter(kg::KnowledgeGraph, filter_func)
    filter_func(kg)
end

"""
    filter_by_confidence(kg::KnowledgeGraph, min_conf::Float64)

Filter knowledge graph to include only entities and relations above minimum confidence.
"""
function filter_by_confidence(kg::KnowledgeGraph, min_conf::Float64)
    KnowledgeGraph(
        filter(e -> e.confidence >= min_conf, kg.entities),
        filter(r -> r.confidence >= min_conf, kg.relations),
        kg.metadata,
        kg.created_at
    )
end

"""
    filter_by_entity_type(kg::KnowledgeGraph, entity_types::Vector{String})

Filter knowledge graph to include only specified entity types.
"""
function filter_by_entity_type(kg::KnowledgeGraph, entity_types::Vector{String})
    KnowledgeGraph(
        filter(e -> e.entity_type in entity_types, kg.entities),
        kg.relations,  # Keep all relations (could filter based on connected entities)
        kg.metadata,
        kg.created_at
    )
end
