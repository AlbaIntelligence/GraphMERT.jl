"""
Graph conversion utilities for GraphMERT visualization.

Provides functions to convert KnowledgeGraph objects to Graphs.jl/MetaGraphs.jl format
for visualization and analysis.
"""

using ..GraphMERT: KnowledgeGraph, KnowledgeEntity, KnowledgeRelation
using ..GraphMERT: Entity, Relation
using Graphs
using MetaGraphs

"""
    kg_to_graphs_format(kg::KnowledgeGraph; validate::Bool=true)

Convert a KnowledgeGraph to Graphs.jl/MetaGraphs.jl format.

This function creates a MetaGraph where:
- Nodes represent entities with metadata
- Edges represent relations with metadata
- All entity and relation attributes are preserved

# Arguments
- `kg::KnowledgeGraph`: Knowledge graph to convert
- `validate::Bool=true`: Whether to validate graph structure integrity

# Returns
- `MetaGraph`: Graph in MetaGraphs.jl format with entity/relation metadata

# Examples
```julia
using GraphMERT

# Create a simple knowledge graph
entities = [
    KnowledgeEntity("e1", "Alice", "PERSON", 0.9, TextPosition(0,5,1,1)),
    KnowledgeEntity("e2", "Bob", "PERSON", 0.8, TextPosition(10,13,1,11))
]
relations = [
    KnowledgeRelation("e1", "e2", "knows", 0.7)
]
kg = KnowledgeGraph(entities, relations)

# Convert to MetaGraph format
g = kg_to_graphs_format(kg)
```
"""
function kg_to_graphs_format(kg::KnowledgeGraph; validate::Bool = true)
    # Create entity ID to index mapping
    entity_map = Dict{String, Int}()
    for (i, entity) in enumerate(kg.entities)
        entity_map[entity.id] = i
    end

    # Validate that all relations reference valid entities
    if validate
        for relation in kg.relations
            if !(relation.head in keys(entity_map))
                error("Relation references unknown head entity: $(relation.head)")
            end
            if !(relation.tail in keys(entity_map))
                error("Relation references unknown tail entity: $(relation.tail)")
            end
            if relation.head == relation.tail
                error("Self-referential relation not allowed: $(relation.head) -> $(relation.tail)")
            end
        end
    end

    # Create MetaGraph with number of entities as nodes
    n_entities = length(kg.entities)
    mg = MetaGraph(n_entities)

    # Add entity metadata to nodes
    for (i, entity) in enumerate(kg.entities)
        # Convert entity to dictionary for metadata (using Symbol keys for MetaGraphs)
        entity_meta = Dict{Symbol, Any}(
            :id => entity.id,
            :text => entity.text,
            :label => entity.label,
            :confidence => entity.confidence,
            :position => entity.position,
            :attributes => entity.attributes,
            :created_at => entity.created_at,
            :entity_type => get_entity_type(entity),
        )

        # Set node metadata
        set_props!(mg, i, entity_meta)
    end

    # Add relations as edges
    for relation in kg.relations
        # Convert relation to dictionary for metadata (using Symbol keys for MetaGraphs)
        relation_meta = Dict{Symbol, Any}(
            :head_id => relation.head,
            :tail_id => relation.tail,
            :relation_type => relation.relation_type,
            :confidence => relation.confidence,
            :attributes => relation.attributes,
            :created_at => relation.created_at,
        )

        # Add edge with metadata
        head_idx = entity_map[relation.head]
        tail_idx = entity_map[relation.tail]

        add_edge!(mg, head_idx, tail_idx)
        set_props!(mg, Edge(head_idx, tail_idx), relation_meta)
    end

    return mg
end

"""
    get_entity_type(entity::KnowledgeEntity)

Extract entity type from KnowledgeEntity.
For backward compatibility with KnowledgeEntity.
"""
function get_entity_type(entity::KnowledgeEntity)
    # KnowledgeEntity doesn't have entity_type field, use label as fallback
    return get(entity.attributes, "entity_type", entity.label)
end

"""
    validate_graph_structure(kg::KnowledgeGraph)

Validate the structural integrity of a KnowledgeGraph.

Checks for:
- Valid entity IDs
- Valid relation references
- No self-referential relations
- Connected components (warning if disconnected)

# Returns
- `Bool`: true if valid, throws error if invalid
"""
function validate_graph_structure(kg::KnowledgeGraph)
    # Check entity IDs are unique and non-empty
    entity_ids = Set{String}()
    for entity in kg.entities
        if isempty(entity.id)
            error("Entity with empty ID found")
        end
        if entity.id in entity_ids
            error("Duplicate entity ID: $(entity.id)")
        end
        push!(entity_ids, entity.id)
    end

    # Check relation references
    for relation in kg.relations
        if !(relation.head in entity_ids)
            error("Relation references unknown head entity: $(relation.head)")
        end
        if !(relation.tail in entity_ids)
            error("Relation references unknown tail entity: $(relation.tail)")
        end
        if relation.head == relation.tail
            error("Self-referential relation found: $(relation.head)")
        end
    end

    return true
end

"""
    get_graph_statistics(kg::KnowledgeGraph)

Get basic statistics about a knowledge graph.

# Returns
- `Dict{String, Any}`: Statistics including node/edge counts, connectivity, etc.
"""
function get_graph_statistics(kg::KnowledgeGraph)
    stats = Dict{String, Any}()

    # Basic counts
    stats["num_entities"] = length(kg.entities)
    stats["num_relations"] = length(kg.relations)

    # Entity types
    entity_types = Dict{String, Int}()
    for entity in kg.entities
        etype = get_entity_type(entity)
        entity_types[etype] = get(entity_types, etype, 0) + 1
    end
    stats["entity_types"] = entity_types

    # Relation types
    relation_types = Dict{String, Int}()
    for relation in kg.relations
        rtype = relation.relation_type
        relation_types[rtype] = get(relation_types, rtype, 0) + 1
    end
    stats["relation_types"] = relation_types

    # Connectivity analysis
    if !isempty(kg.entities)
        mg = kg_to_graphs_format(kg, validate = false)
        components = connected_components(mg)
        stats["num_components"] = length(components)
        stats["is_connected"] = length(components) == 1
        stats["average_degree"] = mean(degree(mg))
    end

    return stats
end

"""
    create_subgraph(kg::KnowledgeGraph, entity_ids::Vector{String})

Create a subgraph containing only specified entities and their relations.

# Arguments
- `kg::KnowledgeGraph`: Original knowledge graph
- `entity_ids::Vector{String}`: Entity IDs to include in subgraph

# Returns
- `KnowledgeGraph`: Subgraph with selected entities and connecting relations
"""
function create_subgraph(kg::KnowledgeGraph, entity_ids::Vector{String})
    entity_id_set = Set(entity_ids)

    # Filter entities
    filtered_entities = filter(e -> e.id in entity_id_set, kg.entities)

    # Filter relations that connect selected entities
    filtered_relations = filter(r ->
            r.head in entity_id_set && r.tail in entity_id_set, kg.relations)

    return KnowledgeGraph(filtered_entities, filtered_relations, kg.metadata, kg.created_at)
end
