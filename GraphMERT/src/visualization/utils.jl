"""
Utility functions for GraphMERT visualization.

Provides filtering, simplification, and other helper functions for knowledge graph visualization.
"""

using ..GraphMERT: KnowledgeGraph, KnowledgeEntity, KnowledgeRelation
using Graphs
using MetaGraphs

"""
    filter_by_confidence(kg::KnowledgeGraph, min_confidence::Float64)

Filter knowledge graph to include only entities and relations above minimum confidence.

# Arguments
- `kg::KnowledgeGraph`: Original knowledge graph
- `min_confidence::Float64`: Minimum confidence threshold (0.0 to 1.0)

# Returns
- `KnowledgeGraph`: Filtered knowledge graph

# Examples
```julia
filtered_kg = filter_by_confidence(kg, 0.8)
```
"""
function filter_by_confidence(kg::KnowledgeGraph, min_confidence::Float64)
    @assert 0.0 ≤ min_confidence ≤ 1.0 "min_confidence must be between 0.0 and 1.0"

    filtered_entities = filter(e -> e.confidence >= min_confidence, kg.entities)
    filtered_relations = filter(r -> r.confidence >= min_confidence, kg.relations)

    return KnowledgeGraph(filtered_entities, filtered_relations, kg.metadata, kg.created_at)
end

"""
    filter_by_entity_type(kg::KnowledgeGraph, entity_types::Vector{String})

Filter knowledge graph to include only specified entity types.

# Arguments
- `kg::KnowledgeGraph`: Original knowledge graph
- `entity_types::Vector{String}`: Entity types to include

# Returns
- `KnowledgeGraph`: Filtered knowledge graph with only specified entity types

# Examples
```julia
biomedical_kg = filter_by_entity_type(kg, ["DISEASE", "DRUG", "GENE"])
```
"""
function filter_by_entity_type(kg::KnowledgeGraph, entity_types::Vector{String})
    type_set = Set(entity_types)

    # Get entity IDs that match the types
    entity_ids = Set{String}()
    for entity in kg.entities
        entity_type = get_entity_type_for_filtering(entity)
        if entity_type in type_set
            push!(entity_ids, entity.id)
        end
    end

    # Filter entities
    filtered_entities = filter(e -> e.id in entity_ids, kg.entities)

    # Filter relations that connect selected entities
    filtered_relations = filter(r -> r.head in entity_ids && r.tail in entity_ids, kg.relations)

    return KnowledgeGraph(filtered_entities, filtered_relations, kg.metadata, kg.created_at)
end

"""
    filter_by_relation_type(kg::KnowledgeGraph, relation_types::Vector{String})

Filter knowledge graph to include only specified relation types.

# Arguments
- `kg::KnowledgeGraph`: Original knowledge graph
- `relation_types::Vector{String}`: Relation types to include

# Returns
- `KnowledgeGraph`: Filtered knowledge graph with only specified relation types

# Examples
```julia
treatment_kg = filter_by_relation_type(kg, ["TREATS", "CAUSES"])
```
"""
function filter_by_relation_type(kg::KnowledgeGraph, relation_types::Vector{String})
    type_set = Set(relation_types)

    filtered_relations = filter(r -> r.relation_type in type_set, kg.relations)

    # Get entity IDs that are involved in the selected relations
    entity_ids = Set{String}()
    for relation in filtered_relations
        push!(entity_ids, relation.head)
        push!(entity_ids, relation.tail)
    end

    # Filter entities
    filtered_entities = filter(e -> e.id in entity_ids, kg.entities)

    return KnowledgeGraph(filtered_entities, filtered_relations, kg.metadata, kg.created_at)
end

"""
    simplify_graph(kg::KnowledgeGraph;
                   max_nodes::Union{Nothing,Int}=nothing,
                   max_edges::Union{Nothing,Int}=nothing,
                   min_confidence::Float64=0.0)

Simplify a knowledge graph by limiting size and filtering by confidence.

# Arguments
- `kg::KnowledgeGraph`: Original knowledge graph
- `max_nodes::Union{Nothing,Int}`: Maximum number of nodes (entities)
- `max_edges::Union{Nothing,Int}`: Maximum number of edges (relations)
- `min_confidence::Float64`: Minimum confidence threshold

# Returns
- `KnowledgeGraph`: Simplified knowledge graph

# Examples
```julia
simple_kg = simplify_graph(kg, max_nodes=50, max_edges=100, min_confidence=0.5)
```
"""
function simplify_graph(kg::KnowledgeGraph;
                        max_nodes::Union{Nothing,Int}=nothing,
                        max_edges::Union{Nothing,Int}=nothing,
                        min_confidence::Float64=0.0)

    # Start with confidence filtering
    filtered_kg = filter_by_confidence(kg, min_confidence)

    # Limit nodes if specified
    if max_nodes !== nothing && length(filtered_kg.entities) > max_nodes
        # Sort entities by confidence and take top N
        sorted_entities = sort(filtered_kg.entities, by=e->e.confidence, rev=true)
        selected_entities = sorted_entities[1:max_nodes]

        entity_ids = Set(e.id for e in selected_entities)

        # Filter relations to only those connecting selected entities
        selected_relations = filter(r -> r.head in entity_ids && r.tail in entity_ids, filtered_kg.relations)

        filtered_kg = KnowledgeGraph(selected_entities, selected_relations, filtered_kg.metadata, filtered_kg.created_at)
    end

    # Limit edges if specified
    if max_edges !== nothing && length(filtered_kg.relations) > max_edges
        # Sort relations by confidence and take top N
        sorted_relations = sort(filtered_kg.relations, by=r->r.confidence, rev=true)
        selected_relations = sorted_relations[1:max_edges]

        # Get all entity IDs involved in selected relations
        entity_ids = Set{String}()
        for relation in selected_relations
            push!(entity_ids, relation.head)
            push!(entity_ids, relation.tail)
        end

        # Filter entities to only those involved in selected relations
        selected_entities = filter(e -> e.id in entity_ids, filtered_kg.entities)

        filtered_kg = KnowledgeGraph(selected_entities, selected_relations, filtered_kg.metadata, filtered_kg.created_at)
    end

    return filtered_kg
end

"""
    cluster_entities(kg::KnowledgeGraph, method::Symbol=:entity_type)

Cluster entities in a knowledge graph for visualization simplification.

# Arguments
- `kg::KnowledgeGraph`: Knowledge graph to cluster
- `method::Symbol`: Clustering method (:entity_type, :confidence_bins, :connected_components)

# Returns
- `Dict{String, Vector{KnowledgeEntity}}`: Clusters of entities

# Examples
```julia
clusters = cluster_entities(kg, :entity_type)
```
"""
function cluster_entities(kg::KnowledgeGraph, method::Symbol=:entity_type)
    clusters = Dict{String, Vector{KnowledgeEntity}}()

    if method == :entity_type
        for entity in kg.entities
            entity_type = get_entity_type_for_filtering(entity)
            if !haskey(clusters, entity_type)
                clusters[entity_type] = Vector{KnowledgeEntity}()
            end
            push!(clusters[entity_type], entity)
        end
    elseif method == :confidence_bins
        for entity in kg.entities
            # Create confidence bins
            conf_bin = floor(entity.confidence * 10) / 10  # 0.1 intervals
            bin_key = "conf_$(conf_bin)"

            if !haskey(clusters, bin_key)
                clusters[bin_key] = Vector{KnowledgeEntity}()
            end
            push!(clusters[bin_key], entity)
        end
    elseif method == :connected_components
        # Convert to graph and find components
        mg = kg_to_graphs_format(kg, validate=false)
        components = connected_components(mg)

        for (i, component) in enumerate(components)
            cluster_key = "component_$i"
            clusters[cluster_key] = Vector{KnowledgeEntity}()

            for node_idx in component
                entity_id = mg[node_idx][:id]
                entity = find_entity_by_id(kg, entity_id)
                if entity !== nothing
                    push!(clusters[cluster_key], entity)
                end
            end
        end
    else
        error("Unknown clustering method: $method")
    end

    return clusters
end

"""
    create_visualization_summary(kg::KnowledgeGraph)

Create a summary of knowledge graph statistics for visualization planning.

# Arguments
- `kg::KnowledgeGraph`: Knowledge graph to analyze

# Returns
- `Dict{String, Any}`: Summary statistics

# Examples
```julia
stats = create_visualization_summary(kg)
println("Graph has \$(stats["num_entities"]) entities and \$(stats["num_relations"]) relations")
```
"""
function create_visualization_summary(kg::KnowledgeGraph)
    stats = Dict{String, Any}()

    # Basic counts
    stats["num_entities"] = length(kg.entities)
    stats["num_relations"] = length(kg.relations)

    # Entity type distribution
    entity_types = Dict{String, Int}()
    for entity in kg.entities
        etype = get_entity_type_for_filtering(entity)
        entity_types[etype] = get(entity_types, etype, 0) + 1
    end
    stats["entity_types"] = entity_types

    # Relation type distribution
    relation_types = Dict{String, Int}()
    for relation in kg.relations
        rtype = relation.relation_type
        relation_types[rtype] = get(relation_types, rtype, 0) + 1
    end
    stats["relation_types"] = relation_types

    # Confidence statistics
    if !isempty(kg.entities)
        entity_confidences = [e.confidence for e in kg.entities]
        stats["entity_confidence_mean"] = mean(entity_confidences)
        stats["entity_confidence_std"] = std(entity_confidences)
        stats["entity_confidence_min"] = minimum(entity_confidences)
        stats["entity_confidence_max"] = maximum(entity_confidences)
    end

    if !isempty(kg.relations)
        relation_confidences = [r.confidence for r in kg.relations]
        stats["relation_confidence_mean"] = mean(relation_confidences)
        stats["relation_confidence_std"] = std(relation_confidences)
        stats["relation_confidence_min"] = minimum(relation_confidences)
        stats["relation_confidence_max"] = maximum(relation_confidences)
    end

    # Graph structure analysis
    mg = kg_to_graphs_format(kg, validate=false)
    stats["is_connected"] = is_connected(mg)
    stats["num_components"] = length(connected_components(mg))
    stats["average_degree"] = mean([degree(mg, v) for v in vertices(mg)])

    # Visualization recommendations
    stats["recommended_layout"] = get_recommended_layout(stats)
    stats["recommended_simplification"] = get_recommended_simplification(stats)

    return stats
end

# Helper functions

function get_entity_type_for_filtering(entity::KnowledgeEntity)
    # Try to get entity_type from attributes, fallback to label
    return get(entity.attributes, "entity_type", entity.label)
end

function find_entity_by_id(kg::KnowledgeGraph, entity_id::String)
    for entity in kg.entities
        if entity.id == entity_id
            return entity
        end
    end
    return nothing
end

function get_recommended_layout(stats::Dict{String, Any})
    num_entities = stats["num_entities"]
    is_connected = stats["is_connected"]
    num_components = stats["num_components"]

    if num_entities < 10
        return :circular
    elseif num_entities < 50 && is_connected
        return :spring
    elseif num_components > 1
        return :shell
    else
        return :stress
    end
end

function get_recommended_simplification(stats::Dict{String, Any})
    num_entities = stats["num_entities"]
    num_relations = stats["num_relations"]

    if num_entities > 1000 || num_relations > 2000
        return "High simplification recommended: limit to 500 nodes, filter by confidence > 0.7"
    elseif num_entities > 500 || num_relations > 1000
        return "Medium simplification recommended: limit to 200 nodes, filter by confidence > 0.5"
    elseif num_entities > 100 || num_relations > 200
        return "Light simplification recommended: filter by confidence > 0.3"
    else
        return "No simplification needed"
    end
end

"""
    validate_visualization_input(kg::KnowledgeGraph)

Validate that a knowledge graph is suitable for visualization.

# Arguments
- `kg::KnowledgeGraph`: Knowledge graph to validate

# Returns
- `Bool`: true if valid for visualization

# Throws
- Error if graph is not suitable for visualization
"""
function validate_visualization_input(kg::KnowledgeGraph)
    if isempty(kg.entities)
        error("Knowledge graph has no entities to visualize")
    end

    if length(kg.entities) > 10000
        @warn "Very large graph ($(length(kg.entities)) entities). Consider simplification for better performance."
    end

    # Check for orphaned relations
    entity_ids = Set(e.id for e in kg.entities)
    orphaned_relations = 0

    for relation in kg.relations
        if !(relation.head in entity_ids) || !(relation.tail in entity_ids)
            orphaned_relations += 1
        end
    end

    if orphaned_relations > 0
        @warn "$orphaned_relations relations reference non-existent entities and will be ignored"
    end

    return true
end
