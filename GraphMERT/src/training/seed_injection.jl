"""
Seed KG Injection Algorithm for GraphMERT.jl

This module implements the seed KG injection algorithm as specified in the GraphMERT paper.
The algorithm injects relevant knowledge graph triples into training data to enable
vocabulary transfer from semantic space to syntactic space.

Algorithm Overview:
1. Entity linking: Map text entities to UMLS concepts using SapBERT embeddings
2. Triple selection: Retrieve relevant triples from seed KG based on entities
3. Injection algorithm: Select diverse, high-quality triples for injection
4. Validation: Ensure semantic consistency with source text
"""

# Types will be available from main module after types.jl is included

"""
    link_entity_sapbert(entity_text::String, config::SeedInjectionConfig)

Link entity to UMLS using SapBERT embeddings and string matching.

Stage 1: SapBERT embedding-based candidate retrieval
Stage 2: Character 3-gram Jaccard similarity filtering

# Arguments
- `entity_text::String`: Entity mention from text
- `config::SeedInjectionConfig`: Injection configuration

# Returns
- `Vector{EntityLinkingResult}`: Ranked list of potential UMLS matches
"""
function link_entity_sapbert(entity_text::String, config::SeedInjectionConfig)
    results = Vector{EntityLinkingResult}()

    # Stage 1: SapBERT embedding-based retrieval (simplified implementation)
    # In full implementation, would use SapBERT embeddings + ANN search
    # For demo, we'll simulate with string similarity

    # Stage 2: Character 3-gram string matching with Jaccard similarity
    entity_3grams = Set([entity_text[i:(i+2)] for i = 1:(length(entity_text)-2)])

    # Simulate UMLS lookup (would be actual UMLS API calls)
    candidate_concepts = [
        ("C0011849", "Diabetes Mellitus", ["Disease", "Endocrine System Disease"]),
        ("C0025598", "Metformin", ["Pharmacologic Substance", "Organic Chemical"]),
        ("C0032961", "Pregnancy", ["Physiologic Function"]),
        ("C0008976", "Clopidogrel", ["Pharmacologic Substance"]),
    ]

    for (cui, preferred_name, semantic_types) in candidate_concepts
        # Calculate Jaccard similarity
        concept_3grams = Set([preferred_name[i:(i+2)] for i = 1:(length(preferred_name)-2)])
        intersection = length(entity_3grams ∩ concept_3grams)
        union_size = length(entity_3grams ∪ concept_3grams)

        if union_size > 0
            jaccard_sim = intersection / union_size
        else
            jaccard_sim = 0.0
        end

        # Filter by threshold
        if jaccard_sim ≥ config.entity_linking_threshold
            push!(
                results,
                EntityLinkingResult(
                    entity_text,
                    cui,
                    preferred_name,
                    semantic_types,
                    jaccard_sim,
                    "jaccard_similarity",
                ),
            )
        end
    end

    # Sort by similarity score (highest first)
    sort!(results, by = r -> r.similarity_score, rev = true)

    # Return top-k candidates
    return results[1:min(config.top_k_candidates, length(results))]
end

"""
    select_triples_for_entity(entity_cui::String, config::SeedInjectionConfig)

Select relevant triples from seed KG for a given entity.

# Arguments
- `entity_cui::String`: UMLS CUI of the entity
- `config::SeedInjectionConfig`: Injection configuration

# Returns
- `Vector{SemanticTriple}`: Top-n triples involving this entity
"""
function select_triples_for_entity(entity_cui::String, config::SeedInjectionConfig)
    triples = Vector{SemanticTriple}()

    # Simulate UMLS triple retrieval (would be actual UMLS API calls)
    # In practice, would query UMLS for triples involving this CUI
    candidate_triples = [
        ("C0011849", "treats", "C0025598", 0.95),  # diabetes treats metformin
        ("C0011849", "complicates", "C0032961", 0.87),  # diabetes complicates pregnancy
        ("C0025598", "inhibits", "C0011849", 0.92),  # metformin inhibits diabetes
        ("C0025598", "metabolized_by", "C0032961", 0.78),  # metformin metabolized by pregnancy
    ]

    for (head_cui, relation, tail_cui, score) in candidate_triples
        if head_cui == entity_cui
            # Get tail entity name (would be UMLS lookup)
            tail_name = get_entity_name_from_cui(tail_cui)
            triple = SemanticTriple(
                entity_cui,
                head_cui,
                relation,
                tail_name,
                tokenize_entity_name(tail_name),
                score,
                "UMLS",
            )
            push!(triples, triple)
        end
    end

    # Sort by score and return top-n
    sort!(triples, by = t -> t.score, rev = true)
    return triples[1:min(config.top_n_triples_per_entity, length(triples))]
end

"""
    inject_seed_kg(sequences::Vector{String}, seed_kg::Vector{SemanticTriple},
                   config::SeedInjectionConfig)

Main seed KG injection algorithm (Algorithm 1 from paper).

Injects seed knowledge graph triples into training sequences to enable
vocabulary transfer and semantic grounding.

# Arguments
- `sequences::Vector{String}`: Training text sequences
- `seed_kg::Vector{SemanticTriple}`: Seed knowledge graph triples
- `config::SeedInjectionConfig`: Injection configuration

# Returns
- `Vector{Tuple{String, Vector{SemanticTriple}}}`: Sequences with injected triples
"""
function inject_seed_kg(
    sequences::Vector{String},
    seed_kg::Vector{SemanticTriple},
    config::SeedInjectionConfig,
)
    injected_sequences = Vector{Tuple{String,Vector{SemanticTriple}}}()

    # For demo purposes, inject into a fixed percentage of sequences
    num_to_inject = round(Int, config.injection_ratio * length(sequences))

    # Simple entity extraction for demo (would be more sophisticated)
    function extract_entities(text::String)
        entities = String[]
        if occursin("diabetes", lowercase(text))
            push!(entities, "diabetes")
        end
        if occursin("metformin", lowercase(text))
            push!(entities, "metformin")
        end
        if occursin("pregnancy", lowercase(text))
            push!(entities, "pregnancy")
        end
        return entities
    end

    for (i, sequence) in enumerate(sequences)
        entities = extract_entities(sequence)

        if !isempty(entities) && i ≤ num_to_inject
            # Link entities to UMLS
            linked_entities = Vector{EntityLinkingResult}()
            for entity in entities
                linked = link_entity_sapbert(entity, config)
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

    # Filter triples by entity CUIs
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
    select_triples_for_entity,
    inject_seed_kg,
    select_triples_for_injection,
    bucket_by_score,
    bucket_by_relation_frequency,
    validate_injected_triples
