"""
Seed KG Injection Algorithm for GraphMERT.jl

This module implements the seed KG injection algorithm as specified in the GraphMERT paper.
The algorithm injects relevant knowledge graph triples into training data to enable
vocabulary transfer from semantic space to syntactic space.

Algorithm Overview:
1. Entity linking: Map text entities to knowledge base concepts using domain-specific linking
2. Triple selection: Retrieve relevant triples from seed KG based on entities
3. Injection algorithm: Select diverse, high-quality triples for injection
4. Validation: Ensure semantic consistency with source text

This module is domain-agnostic and delegates domain-specific operations to DomainProvider instances.
"""

# Types will be available from main module after types.jl is included
# Domain provider functions (link_entity, create_seed_triples, extract_entities, get_domain_name, get_default_domain)
# are available from the domains module included in GraphMERT.jl

# Global caches
const ENTITY_LINKING_CACHE = Dict{String, Vector{EntityLinkingResult}}()
const TRIPLE_CACHE = Dict{String, Vector{SemanticTriple}}()
const CACHE_MAX_SIZE = 10000

"""
    link_entity_sapbert(entity_text::String, config::SeedInjectionConfig, domain::DomainProvider)

Link entity to knowledge base using domain-specific entity linking.

This function delegates to the domain provider's `link_entity` method and converts
the result to `EntityLinkingResult` format for backward compatibility.

# Arguments
- `entity_text::String`: Entity mention from text
- `config::SeedInjectionConfig`: Injection configuration
- `domain::DomainProvider`: Domain provider instance for domain-specific linking

# Returns
- `Vector{EntityLinkingResult}`: Ranked list of potential knowledge base matches
"""
function link_entity_sapbert(
    entity_text::String,
    config::SeedInjectionConfig,
    domain::Any,  # DomainProvider
)
    # Check cache first (cache key includes domain to avoid cross-domain conflicts)
    cache_key = "$(get_domain_name(domain)):$entity_text"
    if haskey(ENTITY_LINKING_CACHE, cache_key)
        return ENTITY_LINKING_CACHE[cache_key]
    end

    results = Vector{EntityLinkingResult}()

    # Delegate to domain provider's link_entity method
    linking_result = link_entity(domain, entity_text, config)
    
    if linking_result === nothing || !isa(linking_result, Dict)
        # Domain doesn't support entity linking or returned invalid format
        # Return empty results
        return results
    end

    # Convert domain provider's Dict result to EntityLinkingResult format
    # Domain providers should return a Dict with keys:
    # - :candidates (Vector of Dicts with :kb_id, :name, :types, :score, :source)
    #   or :candidate (single Dict) for single result
    if haskey(linking_result, :candidates) && isa(linking_result[:candidates], Vector)
        for candidate in linking_result[:candidates]
            if isa(candidate, Dict)
                kb_id = get(candidate, :kb_id, "")  # CUI for biomedical, QID for Wikidata, etc.
                name = get(candidate, :name, entity_text)
                types = get(candidate, :types, String[])
                score = get(candidate, :score, 0.0)
                source = get(candidate, :source, get_domain_name(domain))
                
                # Filter by threshold
                if score ≥ config.entity_linking_threshold && !isempty(kb_id)
                    push!(
                        results,
                        EntityLinkingResult(
                            entity_text,
                            kb_id,
                            name,
                            types,
                            score,
                            source,
                        ),
                    )
                end
            end
        end
    elseif haskey(linking_result, :candidate) && isa(linking_result[:candidate], Dict)
        # Single candidate result
        candidate = linking_result[:candidate]
        kb_id = get(candidate, :kb_id, "")
        name = get(candidate, :name, entity_text)
        types = get(candidate, :types, String[])
        score = get(candidate, :score, 0.0)
        source = get(candidate, :source, get_domain_name(domain))
        
        if score ≥ config.entity_linking_threshold && !isempty(kb_id)
            push!(
                results,
                EntityLinkingResult(
                    entity_text,
                    kb_id,
                    name,
                    types,
                    score,
                    source,
                ),
            )
        end
    end

    # Sort by similarity score (highest first)
    sort!(results, by = r -> r.similarity_score, rev = true)

    # Return top-k candidates
    final_results = results[1:min(config.top_k_candidates, length(results))]

    # Cache the results
    if length(ENTITY_LINKING_CACHE) < CACHE_MAX_SIZE
        ENTITY_LINKING_CACHE[cache_key] = final_results
    end

    return final_results
end

# Backward compatibility: version without domain parameter (uses default domain)
function link_entity_sapbert(entity_text::String, config::SeedInjectionConfig)
    Base.depwarn(
        "link_entity_sapbert(entity_text, config) without domain parameter is deprecated. " *
        "Please pass a domain provider explicitly: link_entity_sapbert(entity_text, config, domain). " *
        "Load domain with: include(\"GraphMERT/src/domains/biomedical.jl\"); bio = load_biomedical_domain(). " *
        "See MIGRATION_GUIDE.md for details.",
        :link_entity_sapbert
    )
    domain = get_default_domain()
    if domain === nothing
        error("No default domain set. Please register a domain or pass domain explicitly.")
    end
    return link_entity_sapbert(entity_text, config, domain)
end

"""
    link_entities_batch(entities::Vector{String}, config::SeedInjectionConfig, domain::DomainProvider)

Batch version of entity linking for improved efficiency.
"""
function link_entities_batch(
    entities::Vector{String},
    config::SeedInjectionConfig,
    domain::Any,  # DomainProvider
)
    results = Vector{Vector{EntityLinkingResult}}()

    for entity in entities
        push!(results, link_entity_sapbert(entity, config, domain))
    end

    return results
end

# Backward compatibility: version without domain parameter
function link_entities_batch(entities::Vector{String}, config::SeedInjectionConfig)
    Base.depwarn(
        "link_entities_batch(entities, config) without domain parameter is deprecated. " *
        "Please pass a domain provider explicitly: link_entities_batch(entities, config, domain). " *
        "See MIGRATION_GUIDE.md for details.",
        :link_entities_batch
    )
    domain = get_default_domain()
    if domain === nothing
        error("No default domain set. Please register a domain or pass domain explicitly.")
    end
    return link_entities_batch(entities, config, domain)
end

"""
    select_triples_for_entity(entity_kb_id::String, config::SeedInjectionConfig, domain::DomainProvider)

Select relevant triples from seed KG for a given entity using domain-specific knowledge base.

# Arguments
- `entity_kb_id::String`: Knowledge base ID of the entity (e.g., CUI for biomedical, QID for Wikidata)
- `config::SeedInjectionConfig`: Injection configuration
- `domain::DomainProvider`: Domain provider instance for domain-specific triple retrieval

# Returns
- `Vector{SemanticTriple}`: Top-n triples involving this entity
"""
function select_triples_for_entity(
    entity_kb_id::String,
    config::SeedInjectionConfig,
    domain::Any,  # DomainProvider
)
    # Check cache first (cache key includes domain to avoid cross-domain conflicts)
    cache_key = "$(get_domain_name(domain)):$entity_kb_id"
    if haskey(TRIPLE_CACHE, cache_key)
        return TRIPLE_CACHE[cache_key]
    end

    triples = Vector{SemanticTriple}()

    # Delegate to domain provider's create_seed_triples method
    # Note: We need to get entity text from kb_id, but create_seed_triples takes entity_text
    # For now, we'll use the kb_id as a fallback. Domains should handle this appropriately.
    seed_triples = create_seed_triples(domain, entity_kb_id, config)
    
    if !isempty(seed_triples) && isa(seed_triples, Vector)
        # Filter triples where this entity is the head
        for triple in seed_triples
            if isa(triple, SemanticTriple)
                if triple.head_cui === entity_kb_id || triple.head == entity_kb_id
                    push!(triples, triple)
                end
            elseif isa(triple, Dict)
                # Convert Dict format to SemanticTriple if needed
                head = get(triple, :head, "")
                head_kb_id = get(triple, :head_kb_id, nothing)
                relation = get(triple, :relation, "")
                tail = get(triple, :tail, "")
                tail_tokens = get(triple, :tail_tokens, Int[])
                score = get(triple, :score, 0.0)
                source = get(triple, :source, get_domain_name(domain))
                
                if (head_kb_id === entity_kb_id || head == entity_kb_id) && !isempty(head) && !isempty(relation) && !isempty(tail)
                    push!(
                        triples,
                        SemanticTriple(
                            head,
                            head_kb_id,
                            relation,
                            tail,
                            tail_tokens,
                            score,
                            source,
                        ),
                    )
                end
            end
        end
    end

    # Sort by score and return top-n
    sort!(triples, by = t -> t.score, rev = true)
    final_triples = triples[1:min(config.top_n_triples_per_entity, length(triples))]

    # Cache the results
    if length(TRIPLE_CACHE) < CACHE_MAX_SIZE
        TRIPLE_CACHE[cache_key] = final_triples
    end

    return final_triples
end

# Backward compatibility: version without domain parameter
function select_triples_for_entity(entity_kb_id::String, config::SeedInjectionConfig)
    Base.depwarn(
        "select_triples_for_entity(entity_kb_id, config) without domain parameter is deprecated. " *
        "Please pass a domain provider explicitly: select_triples_for_entity(entity_kb_id, config, domain). " *
        "See MIGRATION_GUIDE.md for details.",
        :select_triples_for_entity
    )
    domain = get_default_domain()
    if domain === nothing
        error("No default domain set. Please register a domain or pass domain explicitly.")
    end
    return select_triples_for_entity(entity_kb_id, config, domain)
end

"""
    inject_seed_kg(sequences::Vector{String}, seed_kg::Vector{SemanticTriple},
                   config::SeedInjectionConfig, domain::DomainProvider)

Main seed KG injection algorithm (Algorithm 1 from paper).

Injects seed knowledge graph triples into training sequences to enable
vocabulary transfer and semantic grounding.

# Arguments
- `sequences::Vector{String}`: Training text sequences
- `seed_kg::Vector{SemanticTriple}`: Seed knowledge graph triples
- `config::SeedInjectionConfig`: Injection configuration
- `domain::DomainProvider`: Domain provider instance for domain-specific entity extraction and linking

# Returns
- `Vector{Tuple{String, Vector{SemanticTriple}}}`: Sequences with injected triples
"""
function inject_seed_kg(
    sequences::Vector{String},
    seed_kg::Vector{SemanticTriple},
    config::SeedInjectionConfig,
    domain::Any,  # DomainProvider
)
    injected_sequences = Vector{Tuple{String,Vector{SemanticTriple}}}()

    # For demo purposes, inject into a fixed percentage of sequences
    num_to_inject = round(Int, config.injection_ratio * length(sequences))

    # Use domain provider's extract_entities method for entity extraction
    # Create a temporary ProcessingOptions for entity extraction
    options = ProcessingOptions(domain=get_domain_name(domain))

    for (i, sequence) in enumerate(sequences)
        # Extract entities using domain provider
        entities = extract_entities(domain, sequence, options)
        
        # Extract entity text strings for linking
        entity_texts = [e.text for e in entities]

        if !isempty(entity_texts) && i ≤ num_to_inject
            # Link entities to knowledge base
            linked_entities = Vector{EntityLinkingResult}()
            for entity_text in entity_texts
                linked = link_entity_sapbert(entity_text, config, domain)
                append!(linked_entities, linked)
            end

            # Select triples for injection
            selected_triples =
                select_triples_for_injection(linked_entities, seed_kg, config)

            # Limit to max_triples_per_sequence
            if length(selected_triples) > config.max_triples_per_sequence
                selected_triples = selected_triples[1:config.max_triples_per_sequence]
            end

            push!(injected_sequences, (sequence, selected_triples))
        else
            push!(injected_sequences, (sequence, Vector{SemanticTriple}()))
        end
    end

    return injected_sequences
end

# Backward compatibility: version without domain parameter
function inject_seed_kg(
    sequences::Vector{String},
    seed_kg::Vector{SemanticTriple},
    config::SeedInjectionConfig,
)
    Base.depwarn(
        "inject_seed_kg(sequences, seed_kg, config) without domain parameter is deprecated. " *
        "Please pass a domain provider explicitly: inject_seed_kg(sequences, seed_kg, config, domain). " *
        "See MIGRATION_GUIDE.md for details.",
        :inject_seed_kg
    )
    domain = get_default_domain()
    if domain === nothing
        error("No default domain set. Please register a domain or pass domain explicitly.")
    end
    return inject_seed_kg(sequences, seed_kg, config, domain)
end

"""
    select_triples_for_injection(linked_entities::Vector{EntityLinkingResult},
                                seed_kg::Vector{SemanticTriple},
                                config::SeedInjectionConfig)

Select diverse, high-quality triples for injection using the paper's algorithm.

# Arguments
- `linked_entities::Vector{EntityLinkingResult}`: Linked entities from text
- `seed_kg::Vector{SemanticTriple}`: Available seed triples
- `config::SeedInjectionConfig`: Injection configuration

# Returns
- `Vector{SemanticTriple}`: Selected triples for injection
"""
function select_triples_for_injection(
    linked_entities::Vector{EntityLinkingResult},
    seed_kg::Vector{SemanticTriple},
    config::SeedInjectionConfig,
)
    selected_triples = Vector{SemanticTriple}()

    # Filter triples by entity knowledge base IDs (e.g., CUI for biomedical, QID for Wikidata)
    relevant_triples = filter(t -> t.head_cui in [e.cui for e in linked_entities], seed_kg)

    # Filter by score threshold
    relevant_triples = filter(t -> t.score ≥ config.alpha_score_threshold, relevant_triples)

    if isempty(relevant_triples)
        return selected_triples
    end

    # Algorithm 1: Score bucketing + Relation diversity
    # Step 1: Make triples unique (keep highest score)
    unique_triples = Dict{String,SemanticTriple}()
    for triple in relevant_triples
        key = "$(triple.head_cui)_$(triple.relation)_$(triple.tail)"
        if !haskey(unique_triples, key) || triple.score > unique_triples[key].score
            unique_triples[key] = triple
        end
    end

    # Step 2: Bucket by score
    score_buckets =
        bucket_by_score(collect(values(unique_triples)), config.score_bucket_size)

    # Step 3: Within score buckets, bucket by relation frequency
    for (bucket_idx, bucket) in enumerate(score_buckets)
        if !isempty(bucket)
            relation_buckets =
                bucket_by_relation_frequency(bucket, config.relation_bucket_size)

            # Step 4: Select highest-scoring from rarest relations
            for relation_bucket in relation_buckets
                if !isempty(relation_bucket)
                    # Select the highest scoring triple from this relation bucket
                    best_triple = argmax(t -> t.score, relation_bucket)
                    push!(selected_triples, best_triple)

                    # Constraint: one injection per head entity
                    if length(selected_triples) >= length(linked_entities)
                        break
                    end
                end
            end
        end

        if length(selected_triples) >= config.max_triples_per_sequence
            break
        end
    end

    return selected_triples
end

"""
    bucket_by_score(triples::Vector{SemanticTriple}, num_buckets::Int)

Bucket triples by their similarity scores.

# Arguments
- `triples::Vector{SemanticTriple}`: Triples to bucket
- `num_buckets::Int`: Number of buckets

# Returns
- `Vector{Vector{SemanticTriple}}`: List of buckets (highest scores first)
"""
function bucket_by_score(triples::Vector{SemanticTriple}, num_buckets::Int)
    if isempty(triples)
        return [Vector{SemanticTriple}() for _ = 1:num_buckets]
    end

    # Sort by score (highest first)
    sorted_triples = sort(triples, by = t -> t.score, rev = true)

    # Calculate bucket size
    bucket_size = max(1, length(sorted_triples) ÷ num_buckets)

    # Create buckets
    buckets = Vector{Vector{SemanticTriple}}()
    for i = 1:num_buckets
        start_idx = (i-1) * bucket_size + 1
        end_idx = min(i * bucket_size, length(sorted_triples))
        if start_idx ≤ length(sorted_triples)
            push!(buckets, sorted_triples[start_idx:end_idx])
        else
            push!(buckets, Vector{SemanticTriple}())
        end
    end

    return buckets
end

"""
    bucket_by_relation_frequency(triples::Vector{SemanticTriple}, num_buckets::Int)

Bucket triples by relation frequency within a score bucket.

# Arguments
- `triples::Vector{SemanticTriple}`: Triples to bucket
- `num_buckets::Int`: Number of buckets

# Returns
- `Vector{Vector{SemanticTriple}}`: Relation frequency buckets (rarest first)
"""
function bucket_by_relation_frequency(triples::Vector{SemanticTriple}, num_buckets::Int)
    if isempty(triples)
        return [Vector{SemanticTriple}() for _ = 1:num_buckets]
    end

    # Count relation frequencies
    relation_counts = Dict{String,Int}()
    for triple in triples
        relation_counts[triple.relation] = get(relation_counts, triple.relation, 0) + 1
    end

    # Sort relations by frequency (lowest first for diversity)
    sorted_relations = sort(collect(keys(relation_counts)), by = r -> relation_counts[r])

    # Create buckets
    buckets = Vector{Vector{SemanticTriple}}()
    for i = 1:num_buckets
        bucket_relations = sorted_relations[1:min(i, length(sorted_relations))]
        bucket_triples = filter(t -> t.relation in bucket_relations, triples)
        push!(buckets, bucket_triples)
    end

    return buckets
end

"""
    validate_injected_triples(sequence::String, injected_triples::Vector{SemanticTriple})

Validate that injected triples are semantically consistent with source text.

# Arguments
- `sequence::String`: Original text sequence
- `injected_triples::Vector{SemanticTriple}`: Triples to validate

# Returns
- `Dict{SemanticTriple, Bool}`: Validation results for each triple
"""
function validate_injected_triples(
    sequence::String,
    injected_triples::Vector{SemanticTriple},
)
    validation_results = Dict{SemanticTriple,Bool}()

    for triple in injected_triples
        # Simple validation: check if head entity appears in text
        # In full implementation, would use more sophisticated validation
        head_in_text = occursin(lowercase(triple.head), lowercase(sequence))
        tail_in_text = occursin(lowercase(triple.tail), lowercase(sequence))

        # Accept if at least one entity is in text (basic consistency check)
        is_valid = head_in_text || tail_in_text
        validation_results[triple] = is_valid
    end

    return validation_results
end

# Helper functions (would be implemented in biomedical and text modules)
function get_entity_name_from_cui(cui::String)
    # Mock implementation - would query UMLS
    cui_to_name = Dict(
        "C0011849" => "Diabetes Mellitus",
        "C0025598" => "Metformin",
        "C0032961" => "Pregnancy",
        "C0008976" => "Clopidogrel",
    )
    return get(cui_to_name, cui, "Unknown Entity")
end

function tokenize_entity_name(name::String)
    # Simple tokenization - would use proper tokenizer
    return Int[hash(c) % 30522 for c in name]
end

# Export functions for external use
export link_entity_sapbert,
    link_entities_batch,
    select_triples_for_entity,
    inject_seed_kg,
    select_triples_for_injection,
    bucket_by_score,
    bucket_by_relation_frequency,
    validate_injected_triples
