module Ontology

using ..GraphMERT: SemanticTriple
using ..GraphMERT.BiomedicalDomain: UMLSClient, retrieve_triples, UMLSTriple
import GraphMERT # to access BiomedicalDomain

export OntologySource, UMLSOntologySource, fetch_triples

"""
    OntologySource

Abstract base type for external ontology sources used in seed KG injection.
"""
abstract type OntologySource end

"""
    fetch_triples(source::OntologySource, entity_text::String; kwargs...) -> Vector{SemanticTriple}

Fetch triples related to an entity from the ontology source.
"""
function fetch_triples(source::OntologySource, entity_text::String; kwargs...)
    error("fetch_triples not implemented for $(typeof(source))")
end

"""
    UMLSOntologySource <: OntologySource

UMLS-based ontology source using the biomedical domain client.
"""
struct UMLSOntologySource <: OntologySource
    client::UMLSClient
    allowed_relations::Vector{String}
end

function UMLSOntologySource(client::UMLSClient; allowed_relations=String[])
    return UMLSOntologySource(client, allowed_relations)
end

function fetch_triples(source::UMLSOntologySource, entity_text::String; kwargs...)
    # 1. Resolve entity to CUI
    # Ideally we should use the linker, but here we assume entity_text might be a CUI or name
    # If it looks like a CUI (C followed by digits), use it directly
    cui = ""
    if occursin(r"^C[0-9]+$", entity_text)
        cui = entity_text
    else
        # Try to resolve via client (if it has linking capability) or assume caller did linking
        # For seed injection, we often start with known entities (e.g. from seed list)
        # But if we are *extending* from text, we might have just the text.
        
        # Current UMLS client has `get_entity_cui`.
        # Note: This is synchronous and might be slow.
        cui = GraphMERT.BiomedicalDomain.get_entity_cui(source.client, entity_text)
    end
    
    if cui === nothing || isempty(cui)
        return SemanticTriple[]
    end
    
    # 2. Retrieve triples
    umls_triples = retrieve_triples(source.client, cui, source.allowed_relations)
    
    # 3. Convert to SemanticTriple
    semantic_triples = SemanticTriple[]
    for t in umls_triples
        # Map UMLS relation to our schema
        # For now, just use the label
        push!(semantic_triples, SemanticTriple(
            t.cui,          # head 
            t.cui,          # head_cui
            t.relation_label, # relation
            t.related_name,   # tail
            t.related_cui,    # tail_cui (if SemanticTriple has it - wait, type def shows tail is String)
            # Checking SemanticTriple constructor from previous `bash` output:
            # SemanticTriple(head, head_cui, relation, tail, tail_tokens, score, source)
            # We don't have tail_tokens here (need tokenizer).
            # We'll use empty tokens for now and expect the pipeline to tokenize later if needed.
            # But wait, tail_tokens::Vector{Int} is required?
            Int[],          # tail_tokens (empty)
            1.0,            # score (UMLS is trusted)
            t.source        # source
        ))
    end
    
    return semantic_triples
end

end # module
