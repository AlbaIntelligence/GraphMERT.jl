"""
Wikipedia Domain Provider for GraphMERT.jl

This module implements the DomainProvider interface for the Wikipedia/general knowledge domain,
providing Wikipedia-specific entity extraction, relation extraction, validation,
confidence calculation, Wikidata linking (optional), and LLM prompt generation.
"""

using Dates
using Logging

# Import domain interface (these will be available from the main module)
# Note: When this module is included from GraphMERT.jl, these will already be loaded

# Import Wikipedia submodules
include("entities.jl")
include("relations.jl")
include("prompts.jl")
include("wikidata.jl")  # Wikidata integration

"""
    WikipediaDomain

Domain provider for Wikipedia/general knowledge graph extraction.
"""
mutable struct WikipediaDomain <: DomainProvider
    config::DomainConfig
    entity_types::Dict{String, Dict{String, Any}}
    relation_types::Dict{String, Dict{String, Any}}
    wikidata_client::Union{Any, Nothing}  # WikidataClient when available
    
    function WikipediaDomain(wikidata_client::Union{Any, Nothing} = nothing)
        # Initialize entity types
        entity_types = Dict{String, Dict{String, Any}}(
            "PERSON" => Dict("domain" => "wikipedia", "category" => "entity"),
            "ORGANIZATION" => Dict("domain" => "wikipedia", "category" => "entity"),
            "LOCATION" => Dict("domain" => "wikipedia", "category" => "place"),
            "CONCEPT" => Dict("domain" => "wikipedia", "category" => "abstract"),
            "EVENT" => Dict("domain" => "wikipedia", "category" => "temporal"),
            "TECHNOLOGY" => Dict("domain" => "wikipedia", "category" => "artifact"),
            "ARTWORK" => Dict("domain" => "wikipedia", "category" => "artifact"),
            "PERIOD" => Dict("domain" => "wikipedia", "category" => "temporal"),
            "THEORY" => Dict("domain" => "wikipedia", "category" => "abstract"),
            "METHOD" => Dict("domain" => "wikipedia", "category" => "process"),
            "INSTITUTION" => Dict("domain" => "wikipedia", "category" => "organization"),
            "COUNTRY" => Dict("domain" => "wikipedia", "category" => "place"),
        )
        
        # Initialize relation types
        relation_types = Dict{String, Dict{String, Any}}(
            "CREATED_BY" => Dict("domain" => "wikipedia", "category" => "creation"),
            "WORKED_AT" => Dict("domain" => "wikipedia", "category" => "occupation"),
            "BORN_IN" => Dict("domain" => "wikipedia", "category" => "biographical"),
            "DIED_IN" => Dict("domain" => "wikipedia", "category" => "biographical"),
            "FOUNDED" => Dict("domain" => "wikipedia", "category" => "creation"),
            "LED" => Dict("domain" => "wikipedia", "category" => "leadership"),
            "INFLUENCED" => Dict("domain" => "wikipedia", "category" => "influence"),
            "DEVELOPED" => Dict("domain" => "wikipedia", "category" => "creation"),
            "INVENTED" => Dict("domain" => "wikipedia", "category" => "creation"),
            "DISCOVERED" => Dict("domain" => "wikipedia", "category" => "discovery"),
            "WROTE" => Dict("domain" => "wikipedia", "category" => "creation"),
            "PAINTED" => Dict("domain" => "wikipedia", "category" => "creation"),
            "COMPOSED" => Dict("domain" => "wikipedia", "category" => "creation"),
            "DIRECTED" => Dict("domain" => "wikipedia", "category" => "creation"),
            "ACTED_IN" => Dict("domain" => "wikipedia", "category" => "participation"),
            "OCCURRED_IN" => Dict("domain" => "wikipedia", "category" => "temporal"),
            "HAPPENED_DURING" => Dict("domain" => "wikipedia", "category" => "temporal"),
            "PART_OF_EVENT" => Dict("domain" => "wikipedia", "category" => "composition"),
            "RELATED_TO" => Dict("domain" => "wikipedia", "category" => "association"),
            "SIMILAR_TO" => Dict("domain" => "wikipedia", "category" => "similarity"),
            "OPPOSITE_OF" => Dict("domain" => "wikipedia", "category" => "opposition"),
            "PRECEDED_BY" => Dict("domain" => "wikipedia", "category" => "temporal"),
            "FOLLOWED_BY" => Dict("domain" => "wikipedia", "category" => "temporal"),
        )
        
        config = DomainConfig(
            "wikipedia";
            entity_types=collect(keys(entity_types)),
            relation_types=collect(keys(relation_types)),
            validation_rules=Dict{String, Any}(),
            extraction_patterns=Dict{String, Any}(),
            confidence_strategies=Dict{String, Any}(),
        )
        
        new(config, entity_types, relation_types, wikidata_client)
    end
end

# ============================================================================
# Required DomainProvider Methods
# ============================================================================

"""
    register_entity_types(domain::WikipediaDomain)

Register Wikipedia entity types.
"""
function register_entity_types(domain::WikipediaDomain)
    return domain.entity_types
end

"""
    register_relation_types(domain::WikipediaDomain)

Register Wikipedia relation types.
"""
function register_relation_types(domain::WikipediaDomain)
    return domain.relation_types
end

"""
    extract_entities(domain::WikipediaDomain, text::String, config::ProcessingOptions)

Extract Wikipedia entities from text.
"""
function extract_entities(domain::WikipediaDomain, text::String, config::Any)
    # Delegate to Wikipedia entities module
    return extract_wikipedia_entities(text, config, domain)
end

"""
    extract_relations(domain::WikipediaDomain, entities::Vector{Entity}, text::String, config::ProcessingOptions)

Extract Wikipedia relations between entities.
"""
function extract_relations(domain::WikipediaDomain, entities::Vector{Any}, text::String, config::Any)
    # Delegate to Wikipedia relations module
    return extract_wikipedia_relations(entities, text, config, domain)
end

"""
    validate_entity(domain::WikipediaDomain, entity_text::String, entity_type::String, context::Dict)

Validate a Wikipedia entity.
"""
function validate_entity(domain::WikipediaDomain, entity_text::String, entity_type::String, context::Dict{String, Any} = Dict{String, Any}())
    return validate_wikipedia_entity(entity_text, entity_type, context)
end

"""
    validate_relation(domain::WikipediaDomain, head::String, relation_type::String, tail::String, context::Dict)

Validate a Wikipedia relation.
"""
function validate_relation(domain::WikipediaDomain, head::String, relation_type::String, tail::String, context::Dict{String, Any} = Dict{String, Any}())
    return validate_wikipedia_relation(head, relation_type, tail, context)
end

"""
    calculate_entity_confidence(domain::WikipediaDomain, entity_text::String, entity_type::String, context::Dict)

Calculate confidence for a Wikipedia entity.
"""
function calculate_entity_confidence(domain::WikipediaDomain, entity_text::String, entity_type::String, context::Dict{String, Any} = Dict{String, Any}())
    return calculate_wikipedia_entity_confidence(entity_text, entity_type, context)
end

"""
    calculate_relation_confidence(domain::WikipediaDomain, head::String, relation_type::String, tail::String, context::Dict)

Calculate confidence for a Wikipedia relation.
"""
function calculate_relation_confidence(domain::WikipediaDomain, head::String, relation_type::String, tail::String, context::Dict{String, Any} = Dict{String, Any}())
    return calculate_wikipedia_relation_confidence(head, relation_type, tail, context)
end

# ============================================================================
# Optional DomainProvider Methods
# ============================================================================

"""
    link_entity(domain::WikipediaDomain, entity_text::String, config::Any)

Link entity to Wikidata using Wikipedia domain's Wikidata integration (if available).

# Arguments
- `domain::WikipediaDomain`: Domain provider instance
- `entity_text::String`: Entity text to link
- `config::Any`: Configuration

# Returns
- `Union{Dict, Nothing}`: Dict with :candidates or :candidate key, or nothing if not found
"""
function link_entity(domain::WikipediaDomain, entity_text::String, config::Any)
    if domain.wikidata_client === nothing
        return nothing
    end
    
    # Delegate to Wikidata linking
    linking_result = link_entity_to_wikidata(entity_text, domain.wikidata_client)
    
    # Convert to domain interface format
    if linking_result === nothing || !isa(linking_result, Dict)
        return nothing
    end
    
    # If linking_result is a Dict with "results" key, convert to EntityLinkingResult format
    # Otherwise assume it's already in the right format
    if haskey(linking_result, "results") && isa(linking_result["results"], Vector)
        candidates = []
        for result in linking_result["results"]
            if isa(result, Dict)
                push!(candidates, Dict(
                    :kb_id => get(result, :qid, get(result, "qid", "")),
                    :name => get(result, :label, get(result, "label", entity_text)),
                    :types => get(result, :types, get(result, "types", String[])),
                    :score => get(result, :score, get(result, "score", 0.0)),
                    :source => "Wikidata",
                ))
            end
        end
        return Dict(:candidates => candidates)
    end
    
    # Return in the format expected by seed_injection.jl
    # Format: Dict with :candidates or :candidate key
    if haskey(linking_result, :candidates) && isa(linking_result[:candidates], Vector)
        return linking_result
    elseif haskey(linking_result, :candidate) && isa(linking_result[:candidate], Dict)
        return linking_result
    end
    
    return nothing
end

"""
    create_seed_triples(domain::WikipediaDomain, entity_text::String, config::Any)

Create seed KG triples from Wikidata for a Wikipedia entity (if available).

This function queries Wikidata for relations involving the entity and converts them
to SemanticTriple format for seed injection.

# Arguments
- `domain::WikipediaDomain`: Domain provider instance
- `entity_text::String`: Entity text or QID (if entity_text is a QID, it will be used directly)
- `config::Any`: Configuration (can be SeedInjectionConfig or ProcessingOptions)

# Returns
- `Vector{Any}`: Vector of SemanticTriple objects or Dicts that can be converted to SemanticTriple
"""
function create_seed_triples(domain::WikipediaDomain, entity_text::String, config::Any)
    if domain.wikidata_client === nothing
        return Vector{Any}()
    end
    
    # Get QID from entity text (or use entity_text directly if it's already a QID)
    # First try to link the entity to get its QID
    linking_result = link_entity(domain, entity_text, config)
    
    qid = nothing
    entity_name = entity_text
    
    if linking_result !== nothing && isa(linking_result, Dict)
        if haskey(linking_result, :candidate) && isa(linking_result[:candidate], Dict)
            qid = get(linking_result[:candidate], :kb_id, nothing)
            entity_name = get(linking_result[:candidate], :name, entity_text)
        elseif haskey(linking_result, :candidates) && isa(linking_result[:candidates], Vector) && !isempty(linking_result[:candidates])
            first_candidate = linking_result[:candidates][1]
            if isa(first_candidate, Dict)
                qid = get(first_candidate, :kb_id, nothing)
                entity_name = get(first_candidate, :name, entity_text)
            end
        end
    end
    
    # If we couldn't get a QID, try using entity_text as QID directly (in case it's already a QID)
    if qid === nothing && startswith(entity_text, "Q") && length(entity_text) > 1 && all(c -> isdigit(c), entity_text[2:end])
        # Looks like a QID (format: Q123456)
        qid = entity_text
    end
    
    if qid === nothing
        return Vector{Any}()
    end
    
    # Query Wikidata for relations
    relations_response = get_wikidata_relations(qid, domain.wikidata_client)
    
    if !relations_response.success || !haskey(relations_response.data, "result")
        return Vector{Any}()
    end
    
    relations_data = relations_response.data["result"]
    relations_list = get(relations_data, "relations", Vector{Any}())
    
    if isempty(relations_list)
        return Vector{Any}()
    end
    
    # Convert Wikidata relations to SemanticTriple format
    triples = Vector{Any}()
    
    # Get SemanticTriple type from main module
    SemanticTripleType = Main.GraphMERT.SemanticTriple
    
    for relation in relations_list
        if !isa(relation, Dict)
            continue
        end
        
        # Extract relation information
        property_id = get(relation, "property", "")
        target_qid = get(relation, "targetQID", "")
        target_value = get(relation, "targetValue", "")
        
        if isempty(property_id) || (isempty(target_qid) && isempty(target_value))
            continue
        end
        
        # Map Wikidata property to our relation type
        mapped_relation = map_wikidata_property_to_relation_type(property_id)
        
        # Get target entity name
        if !isempty(target_qid)
            target_entity_name = get_wikidata_label(target_qid, domain.wikidata_client)
            if isempty(target_entity_name)
                target_entity_name = target_qid  # Fallback to QID if label not found
            end
        else
            target_entity_name = target_value
        end
        
        # Create token IDs for tail entity (simplified - would tokenize properly)
        # For now, create placeholder tokens
        tail_tokens = [hash(target_entity_name) % 10000]  # Placeholder token IDs
        
        # Calculate confidence score (simplified)
        confidence = 0.7  # Base confidence for Wikidata relations
        
        # Create SemanticTriple
        # Note: head_cui field will contain QID for Wikidata
        try
            triple = SemanticTripleType(
                entity_name,      # head
                qid,              # head_cui (contains QID for Wikidata)
                mapped_relation,  # relation
                target_entity_name,  # tail
                tail_tokens,      # tail_tokens
                confidence,        # score
                "Wikidata",       # source
            )
            push!(triples, triple)
        catch e
            # If SemanticTriple construction fails, create Dict format instead
            push!(triples, Dict(
                :head => entity_name,
                :head_kb_id => qid,
                :relation => mapped_relation,
                :tail => target_entity_name,
                :tail_tokens => tail_tokens,
                :score => confidence,
                :source => "Wikidata",
            ))
        end
    end
    
    return triples
end

"""
    create_prompt(domain::WikipediaDomain, task_type::Symbol, context::Dict)

Generate LLM prompt for Wikipedia domain tasks.
"""
function create_prompt(domain::WikipediaDomain, task_type::Symbol, context::Dict{String, Any})
    return create_wikipedia_prompt(task_type, context)
end

"""
    get_domain_name(domain::WikipediaDomain)

Get the name of this domain.
"""
function get_domain_name(domain::WikipediaDomain)
    return "wikipedia"
end

"""
    get_domain_config(domain::WikipediaDomain)

Get the configuration for this domain.
"""
function get_domain_config(domain::WikipediaDomain)
    return domain.config
end

"""
    create_evaluation_metrics(domain::WikipediaDomain, kg::KnowledgeGraph)

Create domain-specific evaluation metrics for a Wikipedia/general knowledge graph.

This function computes Wikipedia-specific metrics including:
- Wikidata entity linking coverage
- Entity type distribution
- Relation type distribution
- Wikidata validation scores (if Wikidata client is available)
- Entity linking quality metrics

# Arguments
- `domain::WikipediaDomain`: Domain provider instance
- `kg::KnowledgeGraph`: Knowledge graph to evaluate

# Returns
- `Dict{String, Any}`: Dictionary containing domain-specific metrics
"""
function create_evaluation_metrics(domain::WikipediaDomain, kg::Any)
    metrics = Dict{String, Any}()
    
    # Basic KG statistics
    total_entities = length(kg.entities)
    total_relations = length(kg.relations)
    
    metrics["total_entities"] = total_entities
    metrics["total_relations"] = total_relations
    
    # Entity type distribution
    entity_type_counts = Dict{String, Int}()
    for entity in kg.entities
        entity_type = get(entity.attributes, "entity_type", entity.label)
        entity_type_counts[entity_type] = get(entity_type_counts, entity_type, 0) + 1
    end
    metrics["entity_type_distribution"] = entity_type_counts
    
    # Relation type distribution
    relation_type_counts = Dict{String, Int}()
    for relation in kg.relations
        relation_type_counts[relation.relation_type] = get(relation_type_counts, relation.relation_type, 0) + 1
    end
    metrics["relation_type_distribution"] = relation_type_counts
    
    # Wikidata entity linking metrics
    wikidata_linked_count = 0
    wikidata_qids = Set{String}()
    entity_types = Set{String}()
    
    for entity in kg.entities
        # Check for QID (Wikidata ID)
        qid = get(entity.attributes, "qid", get(entity.attributes, "wikidata_id", nothing))
        if qid !== nothing && !isempty(string(qid))
            wikidata_linked_count += 1
            push!(wikidata_qids, string(qid))
        end
        
        # Collect entity types
        entity_type = get(entity.attributes, "entity_type", entity.label)
        if !isempty(entity_type)
            push!(entity_types, entity_type)
        end
    end
    
    metrics["wikidata_linking_coverage"] = total_entities > 0 ? wikidata_linked_count / total_entities : 0.0
    metrics["wikidata_linked_entities"] = wikidata_linked_count
    metrics["unique_wikidata_qids"] = length(wikidata_qids)
    metrics["unique_entity_types"] = length(entity_types)
    metrics["entity_types_list"] = collect(entity_types)
    
    # Average confidence scores
    if total_entities > 0
        avg_entity_confidence = sum(e.confidence for e in kg.entities) / total_entities
        metrics["average_entity_confidence"] = avg_entity_confidence
    else
        metrics["average_entity_confidence"] = 0.0
    end
    
    if total_relations > 0
        avg_relation_confidence = sum(r.confidence for r in kg.relations) / total_relations
        metrics["average_relation_confidence"] = avg_relation_confidence
    else
        metrics["average_relation_confidence"] = 0.0
    end
    
    # Wikidata validation score (if Wikidata client is available)
    if domain.wikidata_client !== nothing && total_relations > 0
        validated_count = 0
        for relation in kg.relations
            # Find head and tail entities
            head_entity = nothing
            tail_entity = nothing
            
            for entity in kg.entities
                if entity.id == relation.head
                    head_entity = entity
                elseif entity.id == relation.tail
                    tail_entity = entity
                end
            end
            
            if head_entity !== nothing && tail_entity !== nothing
                # Check if both are linked to Wikidata
                head_qid = get(head_entity.attributes, "qid", get(head_entity.attributes, "wikidata_id", nothing))
                tail_qid = get(tail_entity.attributes, "qid", get(tail_entity.attributes, "wikidata_id", nothing))
                
                if head_qid !== nothing && tail_qid !== nothing
                    # Validate relation using domain validation
                    context = Dict{String, Any}(
                        "head_qid" => head_qid,
                        "tail_qid" => tail_qid,
                    )
                    if validate_relation(domain, head_entity.text, relation.relation_type, tail_entity.text, context)
                        validated_count += 1
                    end
                end
            end
        end
        metrics["wikidata_validated_relations"] = validated_count
        metrics["wikidata_validation_rate"] = validated_count / total_relations
    else
        metrics["wikidata_validated_relations"] = 0
        metrics["wikidata_validation_rate"] = 0.0
    end
    
    # Domain-specific entity type metrics
    domain_entity_types = ["PERSON", "ORGANIZATION", "LOCATION", "CONCEPT", "EVENT", "TECHNOLOGY"]
    for entity_type in domain_entity_types
        count = get(entity_type_counts, entity_type, 0)
        metrics["entity_type_$(lowercase(entity_type))_count"] = count
        if total_entities > 0
            metrics["entity_type_$(lowercase(entity_type))_percentage"] = count / total_entities
        else
            metrics["entity_type_$(lowercase(entity_type))_percentage"] = 0.0
        end
    end
    
    # Domain-specific relation type metrics
    domain_relation_types = ["CREATED_BY", "BORN_IN", "WORKED_AT", "FOUNDED", "RELATED_TO"]
    for relation_type in domain_relation_types
        count = get(relation_type_counts, relation_type, 0)
        metrics["relation_type_$(lowercase(relation_type))_count"] = count
        if total_relations > 0
            metrics["relation_type_$(lowercase(relation_type))_percentage"] = count / total_relations
        else
            metrics["relation_type_$(lowercase(relation_type))_percentage"] = 0.0
        end
    end
    
    # Graph connectivity metrics
    if total_entities > 0 && total_relations > 0
        # Average relations per entity
        metrics["average_relations_per_entity"] = total_relations / total_entities
        
        # Entities with relations
        entities_with_relations = Set{String}()
        for relation in kg.relations
            push!(entities_with_relations, relation.head)
            push!(entities_with_relations, relation.tail)
        end
        metrics["entities_with_relations"] = length(entities_with_relations)
        metrics["entities_with_relations_percentage"] = length(entities_with_relations) / total_entities
    else
        metrics["average_relations_per_entity"] = 0.0
        metrics["entities_with_relations"] = 0
        metrics["entities_with_relations_percentage"] = 0.0
    end
    
    # Entity linking quality metrics
    if total_entities > 0
        # Entities with high confidence linking
        high_confidence_linked = 0
        for entity in kg.entities
            linking_score = get(entity.attributes, "linking_score", get(entity.attributes, "wikidata_score", 0.0))
            if linking_score > 0.7  # High confidence threshold
                high_confidence_linked += 1
            end
        end
        metrics["high_confidence_linked_entities"] = high_confidence_linked
        metrics["high_confidence_linking_rate"] = high_confidence_linked / total_entities
    else
        metrics["high_confidence_linked_entities"] = 0
        metrics["high_confidence_linking_rate"] = 0.0
    end
    
    # Add timestamp
    metrics["evaluation_timestamp"] = now()
    metrics["domain"] = "wikipedia"
    
    return metrics
end

# Export
export WikipediaDomain
