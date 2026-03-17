"""
Ontology Sources for Seed KG Injection.

This module implements the abstraction for retrieving triples from different ontology sources
(UMLS, Wikidata) in a domain-agnostic way, facilitating multi-domain seed injection.
"""

# retrieve_triples is already exported by GraphMERT (from umls.jl), we will add methods to it.

"""
    get_allowed_relations(source::OntologySource)

Get the list of allowed relation types for this ontology source.
Returns `nothing` if all relations are allowed.
"""
function get_allowed_relations(source::OntologySource)
    return nothing
end

"""
    retrieve_triples(source::OntologySource, domain::DomainProvider, entity_text::String)

Retrieve triples from the specified ontology source for a given entity.
Delegates to the appropriate client available in the domain or creates a temporary one.

# Arguments
- `source::OntologySource`: The ontology source configuration
- `domain::DomainProvider`: The domain provider (context)
- `entity_text::String`: The entity text or ID to query for

# Returns
- `Vector{SemanticTriple}`: Retrieved triples
"""
function retrieve_triples(source::OntologySource, domain::Any, entity_text::String)
    @warn "retrieve_triples not implemented for $(typeof(source)) and $(typeof(domain))"
    return Vector{SemanticTriple}()
end

# ============================================================================
# UMLS Implementation
# ============================================================================

function retrieve_triples(source::UMLSOntologySource, domain::Any, entity_text::String)
    # Check if domain has a UMLS client
    if !hasproperty(domain, :umls_client) || domain.umls_client === nothing
        # In the future, we could try to create a temporary client if credentials are in ENV
        @warn "UMLS source requested but domain $(typeof(domain)) does not have a UMLS client."
        return Vector{SemanticTriple}()
    end
    
    # Reuse the logic that was previously in BiomedicalDomain.create_seed_triples
    # We need to access internal helper functions from BiomedicalDomain, but they might not be exported.
    # Ideally, we should refactor the core logic out of domain.jl or call a helper in domain.jl
    
    # Since we are in GraphMERT module (conceptually), we can access the same things.
    # However, create_seed_triples in domain.jl did a lot of work (linking, getting CUI, mapping relations).
    
    # Let's try to reuse the existing create_seed_triples if the source matches the domain default
    # But that creates a cycle if create_seed_triples calls us.
    
    # So we must duplicate or move the logic. Moving is better.
    # But for now, to ensure stability, I will duplicate/adapt the logic here.
    
    client = domain.umls_client
    
    # 1. Get CUI
    # We assume domain has link_entity method
    # We need a config for link_entity. We can use a default one or extract from domain.
    # create_seed_triples received a config. We don't have it here directly in the signature
    # unless we add it. But the signature in G2.1 didn't specify config.
    # However, retrieve_triples is called by create_seed_triples which HAS config.
    # Maybe we should pass config to retrieve_triples?
    # For now, let's use a default config for linking.
    
    # Actually, entity_text passed here might already be a CUI if the caller handled linking.
    # In seed_injection.jl: seed_triples = create_seed_triples(domain, entity_kb_id, config)
    # It passes `entity_kb_id`.
    
    cui = nothing
    if length(entity_text) == 8 && all(c -> isdigit(c) || isuppercase(c), entity_text)
        cui = entity_text
    end
    
    # If not a CUI, we might need to link it. 
    # But if called from seed_injection.jl, it IS a CUI/ID.
    # Let's assume it is a CUI if it looks like one.
    
    if cui === nothing
         @warn "retrieve_triples(UMLS) expects a CUI, got: $entity_text. Auto-linking not fully supported here yet."
         return Vector{SemanticTriple}()
    end
    
    # 2. Query UMLS
    relations_response = get_relations(client, cui)
    
    if !relations_response.success || !haskey(relations_response.data, "result")
        return Vector{SemanticTriple}()
    end
    
    relations_data = relations_response.data["result"]
    relations_list = get(relations_data, "relations", Vector{Any}())
    
    if isempty(relations_list)
        return Vector{SemanticTriple}()
    end
    
    triples = Vector{SemanticTriple}()
    
    for relation in relations_list
        if !isa(relation, Dict)
            continue
        end
        
        target_cui = get(relation, "targetCUI", "")
        relation_name = get(relation, "relationName", "")
        
        if isempty(target_cui) || isempty(relation_name)
            continue
        end
        
        # Map relation
        # We need access to map_umls_relation_to_biomedical_type
        # It is defined in biomedical/domain.jl. 
        # If it's not exported, we need to find another way or copy it.
        # It is just a function. We can copy it here or make it shared.
        # For now, I'll include a copy of the mapping function helper here to keep this file self-contained.
        mapped_relation = _map_umls_relation(relation_name)
        
        target_entity_name = target_cui # Placeholder
        tail_tokens = [hash(target_entity_name) % 10000] 
        confidence = 0.7
        
        try
            triple = SemanticTriple(
                entity_text,      # head (using passed text/CUI)
                cui,              # head_cui
                mapped_relation,  # relation
                target_entity_name, # tail
                tail_tokens,      # tail_tokens
                confidence,       # score
                "UMLS",           # source
            )
            push!(triples, triple)
        catch e
            # Ignore malformed triples
        end
    end
    
    return triples
end

function _map_umls_relation(umls_relation::String)
    relation_lower = lowercase(umls_relation)
    if occursin("treats", relation_lower) || occursin("therapeutic", relation_lower); return "TREATS"
    elseif occursin("causes", relation_lower) || occursin("etiology", relation_lower); return "CAUSES"
    elseif occursin("associated", relation_lower) || occursin("related", relation_lower); return "ASSOCIATED_WITH"
    elseif occursin("prevents", relation_lower) || occursin("prophylaxis", relation_lower); return "PREVENTS"
    elseif occursin("inhibits", relation_lower) || occursin("suppresses", relation_lower); return "INHIBITS"
    elseif occursin("activates", relation_lower) || occursin("stimulates", relation_lower); return "ACTIVATES"
    elseif occursin("binds", relation_lower) || occursin("interacts", relation_lower); return "BINDS_TO"
    elseif occursin("location", relation_lower) || occursin("site", relation_lower); return "LOCATED_IN"
    elseif occursin("part of", relation_lower) || occursin("component", relation_lower); return "PART_OF"
    elseif occursin("diagnoses", relation_lower) || occursin("indicates", relation_lower); return "INDICATES"
    elseif occursin("manifestation", relation_lower); return "MANIFESTS_AS"
    else; return "ASSOCIATED_WITH" # Default
    end
end

# ============================================================================
# Wikidata Implementation
# ============================================================================

function retrieve_triples(source::WikidataOntologySource, domain::Any, entity_text::String)
    # Check if domain has a Wikidata client
    client = nothing
    if hasproperty(domain, :wikidata_client) && domain.wikidata_client !== nothing
        client = domain.wikidata_client
    else
        # TODO: Create ephemeral Wikidata client if generic domain?
        # For now, require the client in the domain
        @warn "Wikidata source requested but domain $(typeof(domain)) does not have a Wikidata client."
        return Vector{SemanticTriple}()
    end

    qid = entity_text # Assumed to be QID if passed from seed_injection logic
    if !startswith(qid, "Q")
         @warn "retrieve_triples(Wikidata) expects a QID, got: $qid"
         return Vector{SemanticTriple}()
    end

    # Use client to get relations
    # We need to access get_wikidata_relations from wikipedia/wikidata.jl
    # Assuming it is available in Main.GraphMERT scope
    
    # We need to handle the case where functions are not exported.
    # Ideally they should be shared.
    # For now, I will assume get_wikidata_relations is available or accessible.
    
    # Wait, create_seed_triples in wikipedia/domain.jl calls get_wikidata_relations.
    # If I am replacing that logic, I need to call the same lower-level functions.
    
    # Let's assume get_wikidata_relations and get_wikidata_label are available.
    # If not, I might need to move them to a shared utility or duplicate logic (undesirable).
    # Since they are in `wikipedia/wikidata.jl` and that file is included in `WikipediaDomain`,
    # they are likely defined in `GraphMERT` module scope if included in `GraphMERT.jl`?
    # Yes, `include("domains/wikipedia.jl")` -> `include("wikipedia/domain.jl")` -> `include("wikidata.jl")`.
    # So `get_wikidata_relations` is in `GraphMERT` module.
    
    relations_response = get_wikidata_relations(qid, client)
    
    if !relations_response.success || !haskey(relations_response.data, "result")
        return Vector{SemanticTriple}()
    end
    
    relations_data = relations_response.data["result"]
    relations_list = get(relations_data, "relations", Vector{Any}())
    
    if isempty(relations_list)
        return Vector{SemanticTriple}()
    end
    
    triples = Vector{SemanticTriple}()
    SemanticTripleType = SemanticTriple
    
    for relation in relations_list
        if !isa(relation, Dict)
            continue
        end
        
        property_id = get(relation, "property", "")
        target_qid = get(relation, "targetQID", "")
        target_value = get(relation, "targetValue", "")
        
        if isempty(property_id) || (isempty(target_qid) && isempty(target_value))
            continue
        end
        
        mapped_relation = _map_wikidata_property(property_id)
        
        target_entity_name = ""
        if !isempty(target_qid)
            target_entity_name = get_wikidata_label(target_qid, client)
            if isempty(target_entity_name)
                target_entity_name = target_qid
            end
        else
            target_entity_name = target_value
        end
        
        tail_tokens = [hash(target_entity_name) % 10000]
        confidence = 0.7
        
        try
            triple = SemanticTripleType(
                entity_text,      # head (QID)
                qid,              # head_cui
                mapped_relation,  # relation
                target_entity_name, # tail
                tail_tokens,      # tail_tokens
                confidence,       # score
                "Wikidata",       # source
            )
            push!(triples, triple)
        catch e
            # Ignore
        end
    end
    
    return triples
end

function _map_wikidata_property(property_id::String)
    # Simplified mapping
    pid = property_id
    if pid == "P40" || pid == "P22" || pid == "P25" || pid == "P26"; return "FAMILY_RELATION"
    elseif pid == "P19" || pid == "P20"; return "LOCATED_IN"
    elseif pid == "P50" || pid == "P57" || pid == "P86"; return "CREATED_BY"
    elseif pid == "P106"; return "WORKED_AT"
    else; return "RELATED_TO"
    end
end
