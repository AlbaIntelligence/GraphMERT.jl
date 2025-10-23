# Document 09: Triple Extraction Pipeline

## Knowledge Graph Generation from Trained GraphMERT

**Status**: 🔴 **CRITICAL - Scattered partial implementations**
**Priority**: P0 (BLOCKING KG output)
**Paper Reference**: Section 4.4, Figures 7-8
**Existing Code**: Various files, needs consolidation

---

## Overview

The **Triple Extraction Pipeline** converts trained GraphMERT models into explicit knowledge graphs. Unlike training (which uses seed KG), extraction operates on **unseen text** to predict novel triples.

### Key Principle

**Syntactic-to-Semantic Conversion**: GraphMERT learned to map syntactic context (text) to semantic tokens (KG nodes) during training. At extraction time, it predicts semantic tokens using only syntactic context.

### Pipeline Stages

```
Text Corpus
    ↓
[1] Head Discovery (Helper LLM)
    ↓ entities
[2] Relation Matching (Helper LLM)
    ↓ (entity, relation) pairs
[3] Tail Prediction (GraphMERT)
    ↓ top-k tokens
[4] Tail Formation (Helper LLM)
    ↓ complete tails
[5] Filtering & Validation
    ↓
Knowledge Graph
```

---

## Stage 1: Head Discovery

### Purpose

Identify biomedical entities in text that will serve as triple heads.

### Method: Helper LLM with Few-Shot Prompting

```julia
"""
    discover_heads(
        sequence::String,
        helper_llm::LLMClient,
        domain::String = "diabetes"
    )

Discover entity mentions in text using helper LLM.
"""
function discover_heads(
    sequence::String,
    helper_llm::LLMClient,
    domain::String = "diabetes"
)::Vector{EntityMention}

    prompt = """
    You are a biomedical entity extraction system. Extract all medical entities
    relevant to $domain and its comorbidities from the following text.

    Return ONLY entity names, one per line, in the exact form they appear.

    Examples:
    Text: "Diabetes mellitus is associated with cardiovascular disease."
    Entities:
    - Diabetes mellitus
    - cardiovascular disease

    Text: "Metformin improves glycemic control in type 2 diabetes patients."
    Entities:
    - Metformin
    - glycemic control
    - type 2 diabetes

    Now extract from:
    Text: "$sequence"
    Entities:
    """

    response = call_llm(helper_llm, prompt)

    # Parse response
    entities = parse_entity_list(response)

    # Validate against source text
    validated = validate_entities_in_text(entities, sequence)

    return validated
end

"""
Parse LLM response into entity mentions.
"""
function parse_entity_list(response::String)::Vector{String}
    lines = split(response, '\n')
    entities = String[]

    for line in lines
        # Remove bullets, numbering, etc.
        cleaned = strip(replace(line, r"^[-•*0-9.)\s]+" => ""))

        if !isempty(cleaned)
            push!(entities, cleaned)
        end
    end

    return entities
end

"""
Validate that entities actually appear in source text.
"""
function validate_entities_in_text(
    entities::Vector{String},
    text::String
)::Vector{EntityMention}

    validated = EntityMention[]
    text_lower = lowercase(text)

    for entity in entities
        entity_lower = lowercase(entity)

        # Check if entity appears in text (fuzzy match allowed)
        if contains(text_lower, entity_lower) ||
           levenshtein_distance(entity_lower, text_lower) < 3

            # Find position
            start_pos = findfirst(entity_lower, text_lower)

            push!(validated, EntityMention(
                text = entity,
                start_pos = start_pos,
                end_pos = start_pos + length(entity) - 1,
                confidence = 1.0
            ))
        end
    end

    return validated
end

struct EntityMention
    text::String
    start_pos::Int
    end_pos::Int
    confidence::Float64
end
```

**Paper Uses**: Qwen3-32B with "thinking mode" enabled

---

## Stage 2: Relation Matching

### Purpose

For each discovered entity (head), determine which relations from the seed KG are applicable.

### Method: Helper LLM with Relation List

```julia
"""
    match_relations(
        entity::String,
        sequence::String,
        available_relations::Vector{Symbol},
        helper_llm::LLMClient
    )

Determine which relations apply to an entity in context.
"""
function match_relations(
    entity::String,
    sequence::String,
    available_relations::Vector{Symbol},
    helper_llm::LLMClient
)::Vector{Symbol}

    # Format relations for prompt
    relation_list = join(["- $rel" for rel in available_relations], "\n")

    prompt = """
    Given a medical entity in context, select ALL relations that make sense.

    Entity: "$entity"
    Context: "$sequence"

    Available relations:
    $relation_list

    Return ONLY the relation names that are appropriate for this entity in this context,
    one per line. Consider:
    - Biological relationships
    - Clinical associations
    - Anatomical connections
    - Causal relationships

    Relations for "$entity":
    """

    response = call_llm(helper_llm, prompt)

    # Parse relation names
    matched = parse_relation_list(response, available_relations)

    return matched
end

"""
Parse LLM response into relation symbols.
"""
function parse_relation_list(
    response::String,
    valid_relations::Vector{Symbol}
)::Vector{Symbol}

    lines = split(response, '\n')
    relations = Symbol[]

    for line in lines
        cleaned = strip(lowercase(replace(line, r"^[-•*0-9.)\s]+" => "")))

        # Try to match to valid relations
        for rel in valid_relations
            if contains(cleaned, string(rel)) ||
               string(rel) == cleaned
                push!(relations, rel)
                break
            end
        end
    end

    return unique(relations)
end
```

---

## Stage 3: Tail Prediction (GraphMERT)

### Purpose

Use trained GraphMERT to predict tail tokens for each (head, relation) pair.

### Method: Masked Leaf Prediction

```julia
"""
    predict_tail_tokens(
        model::GraphMERTModel,
        sequence::String,
        head_entity::String,
        relation::Symbol,
        k::Int = 20
    )

Predict top-k tail tokens for a triple using GraphMERT.
"""
function predict_tail_tokens(
    model::GraphMERTModel,
    sequence::String,
    head_entity::String,
    relation::Symbol,
    k::Int = 20
)::Vector{Tuple{String, Float32}}  # Returns (token, probability) pairs

    # 1. Create leafy chain graph from sequence
    graph = create_empty_chain_graph_from_text(sequence)

    # 2. Find head position in roots
    head_position = find_head_in_roots(graph, head_entity)

    if head_position === nothing
        @warn "Head entity '$head_entity' not found in sequence"
        return []
    end

    # 3. Create masked leaf for this head+relation
    masked_graph = create_masked_leaf_graph(
        graph,
        head_position,
        relation
    )

    # 4. Forward pass through GraphMERT
    logits = forward_pass(model, masked_graph)  # [1024, vocab_size]

    # 5. Extract logits at masked leaf position
    leaf_position = get_leaf_position(head_position, 0)  # First leaf
    leaf_logits = logits[leaf_position, :]

    # 6. Get top-k predictions
    probs = softmax(leaf_logits)
    top_k_indices = partialsortperm(probs, 1:k, rev=true)

    # 7. Convert to tokens
    tokenizer = model.tokenizer
    predictions = [(decode_token(tokenizer, idx), probs[idx])
                   for idx in top_k_indices]

    return predictions
end

"""
Create graph with masked leaf for prediction.
"""
function create_masked_leaf_graph(
    graph::LeafyChainGraph,
    head_position::Int,
    relation::Symbol
)::LeafyChainGraph

    masked = deepcopy(graph)

    # Set first leaf to [MASK], keep rest as <pad>
    masked.leaf_tokens[head_position + 1, 1] = MASK_TOKEN_ID
    masked.leaf_relations[head_position + 1, :] .= relation
    masked.injected_mask[head_position + 1, 1] = true

    return masked
end
```

**Key Points**:

- Predict at **one leaf position** (first leaf of the head's leaves)
- Use **top-k tokens** (k=20 as per paper) as building blocks
- H-GAT applies relation embedding during forward pass
- Model outputs vocabulary distribution over semantic space

---

## Stage 4: Tail Formation (Helper LLM)

### Purpose

Combine individual predicted tokens into coherent, grammatically correct tail phrases.

### Method: Constrained Generation

```julia
"""
    form_tail_from_tokens(
        head::String,
        relation::Symbol,
        predicted_tokens::Vector{Tuple{String, Float32}},
        sequence::String,
        helper_llm::LLMClient
    )

Form complete tail phrases from predicted tokens using helper LLM.
"""
function form_tail_from_tokens(
    head::String,
    relation::Symbol,
    predicted_tokens::Vector{Tuple{String, Float32}},
    sequence::String,
    helper_llm::LLMClient
)::Vector{String}

    # Format token list
    token_list = join([token for (token, _) in predicted_tokens], ", ")

    prompt = """
    You are completing a medical knowledge graph triple.

    Triple to complete:
    Head: "$head"
    Relation: $relation
    Tail: ???

    Context: "$sequence"

    Available tokens (ONLY use these): $token_list

    Form grammatically correct, medically meaningful tail phrases using ONLY the provided tokens.
    You may combine multiple tokens. Return 0-3 best tail completions, one per line.

    Requirements:
    - Use ONLY tokens from the list above
    - Must be grammatically correct
    - Must be medically meaningful
    - Must fit the relation type
    - Each tail on separate line

    Tails:
    """

    response = call_llm(helper_llm, prompt)

    # Parse tails
    tails = parse_tail_list(response)

    # Validate: only tokens from predicted set allowed
    validated = filter_hallucinated_tails(tails, predicted_tokens)

    return validated
end

"""
Filter tails that contain tokens not in predicted set (hallucinations).
"""
function filter_hallucinated_tails(
    tails::Vector{String},
    allowed_tokens::Vector{Tuple{String, Float32}}
)::Vector{String}

    allowed_set = Set([token for (token, _) in allowed_tokens])
    valid_tails = String[]

    for tail in tails
        # Tokenize tail
        tail_tokens = split(lowercase(tail), r"[\s\-_]")

        # Check all tokens are allowed
        all_valid = all(token in allowed_set for token in tail_tokens)

        if all_valid
            push!(valid_tails, tail)
        else
            @debug "Filtered hallucinated tail: $tail"
        end
    end

    return valid_tails
end
```

**Paper Insight**: "The laborious (grammatical) part of the work is carried out with help from an LLM and the essential (triple extraction) part done with GraphMERT."

### Why LLM Needed?

**Encoder-only challenge**: GraphMERT predicts individual tokens independently. For coherent spans:

- Each token should be conditioned on other tokens
- But encoder predicts all masked positions independently
- Semantic space has limited training examples (~28k vs 124M)

**LLM role**: Combines tokens into grammatically coherent phrases while constrained to predicted vocabulary.

---

## Stage 5: Filtering & Validation

### 5A: Similarity Check

**Purpose**: Ensure extracted triples are grounded in source text.

```julia
"""
    filter_by_similarity(
        triples::Vector{Triple},
        sequence::String,
        embedding_model::GeminiEmbedding,
        β::Float64 = 0.67
    )

Filter triples by similarity to source sequence.
"""
function filter_by_similarity(
    triples::Vector{Triple},
    sequence::String,
    embedding_model::GeminiEmbedding,
    β::Float64 = 0.67
)::Vector{Triple}

    # Encode sequence
    seq_embedding = encode_gemini(embedding_model, sequence)

    filtered = Triple[]

    for triple in triples
        # Linearize triple
        triple_text = "$(triple.head) $(triple.relation) $(triple.tail)"

        # Encode triple
        triple_embedding = encode_gemini(embedding_model, triple_text)

        # Compute similarity
        similarity = cosine_similarity(seq_embedding, triple_embedding)

        if similarity >= β
            push!(filtered, Triple(
                triple.head,
                triple.relation,
                triple.tail,
                similarity,
                sequence  # Store provenance
            ))
        end
    end

    return filtered
end
```

**Paper Values**:

- β = 0.67 (found via grid search)
- Higher β → fewer, more text-specific triples
- Lower β → more, more general triples

**Trade-off**:

- High β: Explicit facts from text
- Low β: Cross-document knowledge (model learned patterns)

### 5B: Deduplication

```julia
"""
Remove duplicate triples, keeping highest similarity.
"""
function deduplicate_triples(
    triples::Vector{Triple}
)::Vector{Triple}

    # Group by (head, relation, tail)
    groups = Dict{Tuple{String, Symbol, String}, Vector{Triple}}()

    for triple in triples
        key = (triple.head, triple.relation, triple.tail)
        if !haskey(groups, key)
            groups[key] = Triple[]
        end
        push!(groups[key], triple)
    end

    # Keep highest similarity from each group
    unique_triples = Triple[]

    for (key, group) in groups
        best = group[argmax([t.similarity for t in group])]
        push!(unique_triples, best)
    end

    return unique_triples
end
```

### 5C: Provenance Tracking

```julia
"""
Triple with provenance.
"""
struct Triple
    head::String
    relation::Symbol
    tail::String
    similarity::Float64
    source_sequence::String
    source_sequence_id::String
    extracted_at::DateTime
end
```

**Paper Feature**: "Each triple is directly traceable to its originating sequence."

---

## Complete Extraction Pipeline

### End-to-End Function

```julia
"""
    extract_knowledge_graph(
        corpus::Vector{String},
        model::GraphMERTModel,
        config::ExtractionConfig
    )

Complete pipeline: text → KG extraction.
"""
function extract_knowledge_graph(
    corpus::Vector{String},
    model::GraphMERTModel,
    config::ExtractionConfig
)::KnowledgeGraph

    all_triples = Triple[]

    for (seq_id, sequence) in enumerate(corpus)
        @info "Processing sequence $seq_id/$(length(corpus))"

        # Stage 1: Discover heads
        heads = discover_heads(sequence, config.helper_llm, config.domain)

        for head in heads
            # Stage 2: Match relations
            relations = match_relations(
                head.text, sequence, config.available_relations, config.helper_llm
            )

            for relation in relations
                # Stage 3: Predict tail tokens
                predicted_tokens = predict_tail_tokens(
                    model, sequence, head.text, relation, config.top_k
                )

                if isempty(predicted_tokens)
                    continue
                end

                # Stage 4: Form complete tails
                tails = form_tail_from_tokens(
                    head.text, relation, predicted_tokens,
                    sequence, config.helper_llm
                )

                # Create triples
                for tail in tails
                    triple = Triple(
                        head = head.text,
                        relation = relation,
                        tail = tail,
                        similarity = 0.0,  # Will be computed in filtering
                        source_sequence = sequence,
                        source_sequence_id = string(seq_id),
                        extracted_at = now()
                    )

                    push!(all_triples, triple)
                end
            end
        end
    end

    # Stage 5: Filter and validate
    @info "Filtering $(length(all_triples)) candidate triples"

    filtered = filter_by_similarity(
        all_triples, corpus, config.embedding_model, config.beta
    )

    deduplicated = deduplicate_triples(filtered)

    @info "Final KG: $(length(deduplicated)) unique triples"

    # Create knowledge graph
    kg = KnowledgeGraph(
        triples = deduplicated,
        num_entities = length(unique([t.head for t in deduplicated] ∪
                                     [t.tail for t in deduplicated])),
        num_relations = length(unique([t.relation for t in deduplicated])),
        source_corpus_size = length(corpus),
        extracted_at = now(),
        model_config = model.config,
        extraction_config = config
    )

    return kg
end
```

---

## Configuration

```julia
"""
Configuration for KG extraction.
"""
struct ExtractionConfig
    # Helper LLM
    helper_llm::LLMClient
    domain::String

    # Relations
    available_relations::Vector{Symbol}

    # Prediction
    top_k::Int                        # Top-k tokens (20)

    # Filtering
    embedding_model::GeminiEmbedding
    beta::Float64                     # Similarity threshold (0.67)

    # Provenance
    track_provenance::Bool
end

function default_extraction_config()
    return ExtractionConfig(
        helper_llm = create_llm_client("qwen3-32b"),
        domain = "diabetes",
        available_relations = load_seed_kg_relations(),
        top_k = 20,
        embedding_model = GeminiEmbedding("text-embedding-004"),
        beta = 0.67,
        track_provenance = true
    )
end
```

---

## Worked Example

**Input Sequence**:

```
"Metformin is a first-line medication for type 2 diabetes."
```

**Stage 1: Head Discovery**

```
Discovered heads:
- "Metformin"
- "type 2 diabetes"
```

**Stage 2: Relation Matching**

```
Metformin:
- :plays_role
- :has_disposition
- :treats (not in seed KG, skipped)

Type 2 diabetes:
- :isa
- :associated_with
- :has_pathological_process
```

**Stage 3: Tail Prediction (for Metformin, :plays_role)**

```
Top-20 tokens:
therapeutic, role, agent, drug, medication, treatment,
pharmacological, antidiabetic, hypoglycemic, ...
```

**Stage 4: Tail Formation**

```
LLM combinations:
1. "therapeutic role" ✓
2. "pharmacological agent" ✓
3. "antidiabetic medication" ✓
```

**Stage 5: Filtering**

```
Similarities:
1. (Metformin, plays_role, therapeutic role) - 0.78 ✓ (> 0.67)
2. (Metformin, plays_role, pharmacological agent) - 0.65 ✗ (< 0.67)
3. (Metformin, plays_role, antidiabetic medication) - 0.82 ✓
```

**Final Triples**:

```
- (Metformin, plays_role, therapeutic role) - sim: 0.78
- (Metformin, plays_role, antidiabetic medication) - sim: 0.82
- (type 2 diabetes, isa, disease) - sim: 0.85
- ...
```

---

## Performance Optimization

### Batching

```julia
"""
Batch extraction for efficiency.
"""
function extract_kg_batched(
    corpus::Vector{String},
    model::GraphMERTModel,
    config::ExtractionConfig,
    batch_size::Int = 16
)

    kg_parts = []

    for batch_start in 1:batch_size:length(corpus)
        batch_end = min(batch_start + batch_size - 1, length(corpus))
        batch = corpus[batch_start:batch_end]

        # Process batch in parallel
        batch_kg = extract_knowledge_graph(batch, model, config)
        push!(kg_parts, batch_kg)
    end

    # Merge all KGs
    return merge_knowledge_graphs(kg_parts)
end
```

### Caching

```julia
"""
Cache helper LLM responses to reduce API costs.
"""
mutable struct CachedLLMClient
    client::LLMClient
    cache::Dict{String, String}
    cache_file::String
end

function call_llm_cached(client::CachedLLMClient, prompt::String)::String
    if haskey(client.cache, prompt)
        return client.cache[prompt]
    end

    response = call_llm(client.client, prompt)
    client.cache[prompt] = response

    # Persist cache
    save_cache(client.cache_file, client.cache)

    return response
end
```

---

## Validation

### Output Quality Metrics

```julia
"""
Compute extraction statistics.
"""
function compute_extraction_stats(kg::KnowledgeGraph)
    return Dict(
        "num_triples" => length(kg.triples),
        "num_entities" => kg.num_entities,
        "num_relations" => kg.num_relations,
        "triples_per_relation" => count_triples_per_relation(kg),
        "avg_similarity" => mean([t.similarity for t in kg.triples]),
        "median_similarity" => median([t.similarity for t in kg.triples]),
        "similarity_distribution" => histogram([t.similarity for t in kg.triples])
    )
end
```

---

## Testing

```julia
@testset "Triple Extraction Pipeline" begin

    @testset "Head Discovery" begin
        text = "Diabetes mellitus is a metabolic disorder."
        heads = discover_heads(text, llm, "diabetes")
        @test "diabetes mellitus" in [h.text for h in heads]
    end

    @testset "Tail Prediction" begin
        model = load_trained_model()
        tokens = predict_tail_tokens(model, text, "diabetes", :isa, 20)
        @test length(tokens) == 20
        @test all(p -> 0 <= p <= 1, [prob for (_, prob) in tokens])
    end

    @testset "End-to-End" begin
        corpus = load_test_corpus()
        kg = extract_knowledge_graph(corpus, model, config)
        @test length(kg.triples) > 0
        @test all(t -> t.similarity >= config.beta, kg.triples)
    end
end
```

---

## Implementation Checklist

- [ ] Implement head discovery with LLM
- [ ] Implement relation matching
- [ ] Implement tail token prediction
- [ ] Implement tail formation with LLM
- [ ] Implement similarity filtering
- [ ] Implement deduplication
- [ ] Implement provenance tracking
- [ ] Add caching for LLM calls
- [ ] Add batching for efficiency
- [ ] Write comprehensive tests
- [ ] Validate on diabetes dataset

---

**Related Documents**:

- → [Doc 02: Leafy Chain Graphs](02-leafy-chain-graphs.md)
- → [Doc 07: MNM Training](07-training-mnm.md)
- → [Doc 08: Seed Injection](08-seed-kg-injection.md)
- → [Doc 10: Evaluation](10-evaluation-metrics.md)
