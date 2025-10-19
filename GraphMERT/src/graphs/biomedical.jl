"""
Biomedical knowledge graph for GraphMERT.jl

This module implements biomedical knowledge graph construction with UMLS mappings
as specified in the GraphMERT paper for biomedical knowledge graph construction.
"""

using LightGraphs
using MetaGraphs
using SparseArrays
using LinearAlgebra
using Dates

# ============================================================================
# Biomedical Knowledge Graph Types
# ============================================================================

"""
    BiomedicalKnowledgeGraph

Specialized knowledge graph for biomedical entities and relations.
"""
struct BiomedicalKnowledgeGraph
    entities::Vector{BiomedicalEntity}
    relations::Vector{BiomedicalRelation}
    umls_mappings::Dict{String,String}
    semantic_types::Dict{String,Vector{String}}
    confidence_scores::Dict{String,Float64}
    created_at::DateTime
    metadata::Dict{String,Any}

    function BiomedicalKnowledgeGraph(entities::Vector{BiomedicalEntity},
        relations::Vector{BiomedicalRelation},
        umls_mappings::Dict{String,String}=Dict{String,String}(),
        semantic_types::Dict{String,Vector{String}}=Dict{String,Vector{String}(),
        confidence_scores::Dict{String,Float64}=Dict{String,Float64}(),
        metadata::Dict{String,Any}=Dict{String,Any}())
        
        new(entities, relations, umls_mappings, semantic_types, confidence_scores, now(), metadata)
    end
end

"""
    BiomedicalGraphNode

Node in the biomedical knowledge graph.
"""
struct BiomedicalGraphNode
    id::String
    entity::BiomedicalEntity
    umls_cui::String
    semantic_types::Vector{String}
    confidence::Float64
    position::Tuple{Int,Int}  # (x, y) coordinates for visualization

    function BiomedicalGraphNode(id::String, entity::BiomedicalEntity,
        umls_cui::String="", semantic_types::Vector{String}=String[],
        confidence::Float64=0.0, position::Tuple{Int,Int}=(0, 0))
        
        new(id, entity, umls_cui, semantic_types, confidence, position)
    end
end

"""
    BiomedicalGraphEdge

Edge in the biomedical knowledge graph.
"""
struct BiomedicalGraphEdge
    id::String
    source_node_id::String
    target_node_id::String
    relation::BiomedicalRelation
    weight::Float64
    confidence::Float64

    function BiomedicalGraphEdge(id::String, source_node_id::String, target_node_id::String,
        relation::BiomedicalRelation, weight::Float64=1.0, confidence::Float64=0.0)
        
        new(id, source_node_id, target_node_id, relation, weight, confidence)
    end
end

# ============================================================================
# Graph Construction
# ============================================================================

"""
    create_biomedical_graph(entities::Vector{BiomedicalEntity}, relations::Vector{BiomedicalRelation})

Create a biomedical knowledge graph from entities and relations.
"""
function create_biomedical_graph(entities::Vector{BiomedicalEntity}, relations::Vector{BiomedicalRelation})
    # Create entity ID mapping
    entity_id_map = Dict{String,Int}()
    for (i, entity) in enumerate(entities)
        entity_id_map[entity.id] = i
    end
    
    # Create graph
    num_nodes = length(entities)
    graph = MetaGraph(num_nodes)
    
    # Add nodes
    for (i, entity) in enumerate(entities)
        set_prop!(graph, i, :entity, entity)
        set_prop!(graph, i, :entity_id, entity.id)
        set_prop!(graph, i, :entity_type, entity.label)
        set_prop!(graph, i, :confidence, entity.confidence)
    end
    
    # Add edges
    for relation in relations
        source_idx = get(entity_id_map, relation.head, 0)
        target_idx = get(entity_id_map, relation.tail, 0)
        
        if source_idx > 0 && target_idx > 0
            add_edge!(graph, source_idx, target_idx)
            set_prop!(graph, source_idx, target_idx, :relation, relation)
            set_prop!(graph, source_idx, target_idx, :relation_type, relation.relation_type)
            set_prop!(graph, source_idx, target_idx, :confidence, relation.confidence)
        end
    end
    
    return graph
end

"""
    build_biomedical_graph(entities::Vector{BiomedicalEntity}, relations::Vector{BiomedicalRelation}; umls_client=nothing)

Build a complete biomedical knowledge graph with UMLS integration.
"""
function build_biomedical_graph(entities::Vector{BiomedicalEntity}, relations::Vector{BiomedicalRelation}; umls_client=nothing)
    # Initialize mappings
    umls_mappings = Dict{String,String}()
    semantic_types = Dict{String,Vector{String}}()
    confidence_scores = Dict{String,Float64}()
    
    # Process entities with UMLS integration
    if umls_client !== nothing
        for entity in entities
            try
                # Get UMLS CUI
                cui = get_entity_cui(umls_client, entity.text)
                if cui !== nothing
                    umls_mappings[entity.id] = cui
                    
                    # Get semantic types
                    sem_types = get_entity_semantic_types(umls_client, cui)
                    if !isempty(sem_types)
                        semantic_types[entity.id] = sem_types
                    end
                    
                    # Calculate confidence based on UMLS match
                    confidence_scores[entity.id] = min(entity.confidence + 0.2, 1.0)
                else
                    confidence_scores[entity.id] = entity.confidence
                end
            catch e
                @warn "UMLS processing failed for entity $(entity.id): $e"
                confidence_scores[entity.id] = entity.confidence
            end
        end
    else
        # Use original confidence scores
        for entity in entities
            confidence_scores[entity.id] = entity.confidence
        end
    end
    
    # Process relations
    for relation in relations
        relation_id = "$(relation.head)-$(relation.tail)-$(relation.relation_type)"
        confidence_scores[relation_id] = relation.confidence
    end
    
    # Create metadata
    metadata = Dict{String,Any}(
        "total_entities" => length(entities),
        "total_relations" => length(relations),
        "umls_entities" => length(umls_mappings),
        "entity_types" => count_entity_types(entities),
        "relation_types" => count_relation_types(relations),
        "average_confidence" => calculate_average_confidence(confidence_scores)
    )
    
    return BiomedicalKnowledgeGraph(entities, relations, umls_mappings, semantic_types, confidence_scores, metadata)
end

# ============================================================================
# Graph Analysis
# ============================================================================

"""
    analyze_biomedical_graph(graph::BiomedicalKnowledgeGraph)

Analyze the biomedical knowledge graph and return statistics.
"""
function analyze_biomedical_graph(graph::BiomedicalKnowledgeGraph)
    stats = Dict{String,Any}()
    
    # Basic statistics
    stats["total_entities"] = length(graph.entities)
    stats["total_relations"] = length(graph.relations)
    stats["umls_mapped_entities"] = length(graph.umls_mappings)
    
    # Entity type distribution
    entity_types = Dict{String,Int}()
    for entity in graph.entities
        entity_types[entity.label] = get(entity_types, entity.label, 0) + 1
    end
    stats["entity_types"] = entity_types
    
    # Relation type distribution
    relation_types = Dict{String,Int}()
    for relation in graph.relations
        relation_types[relation.relation_type] = get(relation_types, relation.relation_type, 0) + 1
    end
    stats["relation_types"] = relation_types
    
    # Confidence statistics
    entity_confidences = [e.confidence for e in graph.entities]
    relation_confidences = [r.confidence for r in graph.relations]
    
    stats["avg_entity_confidence"] = mean(entity_confidences)
    stats["avg_relation_confidence"] = mean(relation_confidences)
    stats["min_entity_confidence"] = minimum(entity_confidences)
    stats["max_entity_confidence"] = maximum(entity_confidences)
    stats["min_relation_confidence"] = minimum(relation_confidences)
    stats["max_relation_confidence"] = maximum(relation_confidences)
    
    # UMLS integration statistics
    if !isempty(graph.umls_mappings)
        stats["umls_coverage"] = length(graph.umls_mappings) / length(graph.entities)
    else
        stats["umls_coverage"] = 0.0
    end
    
    return stats
end

"""
    find_connected_components(graph::BiomedicalKnowledgeGraph)

Find connected components in the biomedical knowledge graph.
"""
function find_connected_components(graph::BiomedicalKnowledgeGraph)
    # Create a simple graph for connected components analysis
    num_entities = length(graph.entities)
    simple_graph = SimpleGraph(num_entities)
    
    # Add edges
    entity_id_map = Dict{String,Int}()
    for (i, entity) in enumerate(graph.entities)
        entity_id_map[entity.id] = i
    end
    
    for relation in graph.relations
        source_idx = get(entity_id_map, relation.head, 0)
        target_idx = get(entity_id_map, relation.tail, 0)
        
        if source_idx > 0 && target_idx > 0
            add_edge!(simple_graph, source_idx, target_idx)
        end
    end
    
    # Find connected components
    components = connected_components(simple_graph)
    
    return components
end

"""
    calculate_graph_metrics(graph::BiomedicalKnowledgeGraph)

Calculate various graph metrics for the biomedical knowledge graph.
"""
function calculate_graph_metrics(graph::BiomedicalKnowledgeGraph)
    metrics = Dict{String,Any}()
    
    # Basic metrics
    num_entities = length(graph.entities)
    num_relations = length(graph.relations)
    
    metrics["num_entities"] = num_entities
    metrics["num_relations"] = num_relations
    metrics["density"] = num_relations / (num_entities * (num_entities - 1) / 2)
    
    # Create adjacency matrix
    entity_id_map = Dict{String,Int}()
    for (i, entity) in enumerate(graph.entities)
        entity_id_map[entity.id] = i
    end
    
    adjacency_matrix = zeros(Int, num_entities, num_entities)
    for relation in graph.relations
        source_idx = get(entity_id_map, relation.head, 0)
        target_idx = get(entity_id_map, relation.tail, 0)
        
        if source_idx > 0 && target_idx > 0
            adjacency_matrix[source_idx, target_idx] = 1
        end
    end
    
    # Calculate metrics
    metrics["num_edges"] = sum(adjacency_matrix)
    metrics["num_self_loops"] = sum(diag(adjacency_matrix))
    metrics["num_reciprocal_edges"] = sum(adjacency_matrix .* adjacency_matrix')
    
    # Degree statistics
    out_degrees = sum(adjacency_matrix, dims=2)[:]
    in_degrees = sum(adjacency_matrix, dims=1)[:]
    
    metrics["avg_out_degree"] = mean(out_degrees)
    metrics["avg_in_degree"] = mean(in_degrees)
    metrics["max_out_degree"] = maximum(out_degrees)
    metrics["max_in_degree"] = maximum(in_degrees)
    
    return metrics
end

# ============================================================================
# Graph Filtering
# ============================================================================

"""
    filter_by_confidence(graph::BiomedicalKnowledgeGraph, threshold::Float64)

Filter the graph by confidence threshold.
"""
function filter_by_confidence(graph::BiomedicalKnowledgeGraph, threshold::Float64)
    # Filter entities
    filtered_entities = filter(e -> e.confidence >= threshold, graph.entities)
    
    # Filter relations
    filtered_relations = filter(r -> r.confidence >= threshold, graph.relations)
    
    # Update mappings and types
    filtered_umls_mappings = Dict{String,String}()
    filtered_semantic_types = Dict{String,Vector{String}}()
    filtered_confidence_scores = Dict{String,Float64}()
    
    for entity in filtered_entities
        if haskey(graph.umls_mappings, entity.id)
            filtered_umls_mappings[entity.id] = graph.umls_mappings[entity.id]
        end
        if haskey(graph.semantic_types, entity.id)
            filtered_semantic_types[entity.id] = graph.semantic_types[entity.id]
        end
        if haskey(graph.confidence_scores, entity.id)
            filtered_confidence_scores[entity.id] = graph.confidence_scores[entity.id]
        end
    end
    
    # Update metadata
    filtered_metadata = copy(graph.metadata)
    filtered_metadata["total_entities"] = length(filtered_entities)
    filtered_metadata["total_relations"] = length(filtered_relations)
    filtered_metadata["confidence_threshold"] = threshold
    
    return BiomedicalKnowledgeGraph(filtered_entities, filtered_relations, filtered_umls_mappings,
        filtered_semantic_types, filtered_confidence_scores, filtered_metadata)
end

"""
    filter_by_entity_type(graph::BiomedicalKnowledgeGraph, entity_types::Vector{String})

Filter the graph by entity types.
"""
function filter_by_entity_type(graph::BiomedicalKnowledgeGraph, entity_types::Vector{String})
    # Filter entities
    filtered_entities = filter(e -> e.label in entity_types, graph.entities)
    
    # Get entity IDs
    filtered_entity_ids = Set(e.id for e in filtered_entities)
    
    # Filter relations
    filtered_relations = filter(r -> r.head in filtered_entity_ids && r.tail in filtered_entity_ids, graph.relations)
    
    # Update mappings and types
    filtered_umls_mappings = Dict{String,String}()
    filtered_semantic_types = Dict{String,Vector{String}}()
    filtered_confidence_scores = Dict{String,Float64}()
    
    for entity in filtered_entities
        if haskey(graph.umls_mappings, entity.id)
            filtered_umls_mappings[entity.id] = graph.umls_mappings[entity.id]
        end
        if haskey(graph.semantic_types, entity.id)
            filtered_semantic_types[entity.id] = graph.semantic_types[entity.id]
        end
        if haskey(graph.confidence_scores, entity.id)
            filtered_confidence_scores[entity.id] = graph.confidence_scores[entity.id]
        end
    end
    
    # Update metadata
    filtered_metadata = copy(graph.metadata)
    filtered_metadata["total_entities"] = length(filtered_entities)
    filtered_metadata["total_relations"] = length(filtered_relations)
    filtered_metadata["filtered_entity_types"] = entity_types
    
    return BiomedicalKnowledgeGraph(filtered_entities, filtered_relations, filtered_umls_mappings,
        filtered_semantic_types, filtered_confidence_scores, filtered_metadata)
end

# ============================================================================
# Graph Export
# ============================================================================

"""
    export_to_json(graph::BiomedicalKnowledgeGraph)

Export the biomedical knowledge graph to JSON format.
"""
function export_to_json(graph::BiomedicalKnowledgeGraph)
    export_data = Dict{String,Any}()
    
    # Basic information
    export_data["metadata"] = graph.metadata
    export_data["created_at"] = string(graph.created_at)
    
    # Entities
    export_data["entities"] = [
        Dict(
            "id" => e.id,
            "text" => e.text,
            "label" => e.label,
            "confidence" => e.confidence,
            "position" => Dict(
                "start" => e.position.start,
                "stop" => e.position.stop,
                "line" => e.position.line,
                "column" => e.position.column
            ),
            "attributes" => e.attributes,
            "created_at" => string(e.created_at)
        ) for e in graph.entities
    ]
    
    # Relations
    export_data["relations"] = [
        Dict(
            "head" => r.head,
            "tail" => r.tail,
            "relation_type" => r.relation_type,
            "confidence" => r.confidence,
            "attributes" => r.attributes,
            "created_at" => string(r.created_at)
        ) for r in graph.relations
    ]
    
    # UMLS mappings
    export_data["umls_mappings"] = graph.umls_mappings
    
    # Semantic types
    export_data["semantic_types"] = graph.semantic_types
    
    # Confidence scores
    export_data["confidence_scores"] = graph.confidence_scores
    
    return export_data
end

"""
    export_to_graphml(graph::BiomedicalKnowledgeGraph)

Export the biomedical knowledge graph to GraphML format.
"""
function export_to_graphml(graph::BiomedicalKnowledgeGraph)
    # This would require a GraphML library
    # For now, return a placeholder
    return "GraphML export not yet implemented"
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    count_entity_types(entities::Vector{BiomedicalEntity})

Count entities by type.
"""
function count_entity_types(entities::Vector{BiomedicalEntity})
    counts = Dict{String,Int}()
    for entity in entities
        counts[entity.label] = get(counts, entity.label, 0) + 1
    end
    return counts
end

"""
    count_relation_types(relations::Vector{BiomedicalRelation})

Count relations by type.
"""
function count_relation_types(relations::Vector{BiomedicalRelation})
    counts = Dict{String,Int}()
    for relation in relations
        counts[relation.relation_type] = get(counts, relation.relation_type, 0) + 1
    end
    return counts
end

"""
    calculate_average_confidence(confidence_scores::Dict{String,Float64})

Calculate average confidence score.
"""
function calculate_average_confidence(confidence_scores::Dict{String,Float64})
    if isempty(confidence_scores)
        return 0.0
    end
    
    return sum(values(confidence_scores)) / length(confidence_scores)
end

"""
    get_entity_by_id(graph::BiomedicalKnowledgeGraph, entity_id::String)

Get an entity by its ID.
"""
function get_entity_by_id(graph::BiomedicalKnowledgeGraph, entity_id::String)
    for entity in graph.entities
        if entity.id == entity_id
            return entity
        end
    end
    return nothing
end

"""
    get_relations_by_entity(graph::BiomedicalKnowledgeGraph, entity_id::String)

Get all relations involving a specific entity.
"""
function get_relations_by_entity(graph::BiomedicalKnowledgeGraph, entity_id::String)
    relations = Vector{BiomedicalRelation}()
    for relation in graph.relations
        if relation.head == entity_id || relation.tail == entity_id
            push!(relations, relation)
        end
    end
    return relations
end

"""
    get_umls_cui(graph::BiomedicalKnowledgeGraph, entity_id::String)

Get the UMLS CUI for an entity.
"""
function get_umls_cui(graph::BiomedicalKnowledgeGraph, entity_id::String)
    return get(graph.umls_mappings, entity_id, "")
end

"""
    get_semantic_types(graph::BiomedicalKnowledgeGraph, entity_id::String)

Get the semantic types for an entity.
"""
function get_semantic_types(graph::BiomedicalKnowledgeGraph, entity_id::String)
    return get(graph.semantic_types, entity_id, String[])
end
