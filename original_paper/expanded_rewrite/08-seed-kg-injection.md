# Document 08: Seed KG Injection Algorithm

## Training Data Preparation with External Knowledge

**Status**: 🔴 **CRITICAL - Only 19 lines placeholder**
**Priority**: P0 (BLOCKING training)
**Paper Reference**: Section 4.3, Appendix B (Algorithm 1)
**Existing Code**: `GraphMERT/src/training/seed_injection.jl` (19 lines, stub)

---

## Overview

The **Seed KG Injection Algorithm** prepares training data by selecting relevant triples from an external knowledge graph (UMLS) and injecting them into leafy chain graphs. This creates the semantic space for MNM training while maintaining diversity and relevance.

### Purpose

**Input**:

- Text corpus (syntactic space)
- External KG (e.g., UMLS with millions of triples)
- Target: ~100-1000 examples per relation

**Output**:

- Seed KG (~28k triples for diabetes dataset)
- Leafy chain graphs with injected triples
- Balanced relation distribution

### Key Challenges

1. **Scale Mismatch**: UMLS has millions of triples, but can only inject ~30k
2. **Relevance**: Triples must be contextually relevant to text
3. **Diversity**: Must balance all relations equally
4. **Quality**: High similarity scores can dominate with trivial triples

---

## Four-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Entity Linking                                     │
│   Text → Discover Entities → Match to UMLS CUIs             │
└────────────────┬────────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Triple Retrieval                                   │
│   CUIs → Retrieve UMLS Triples → Filter by Relations        │
└────────────────┬────────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Contextual Selection                               │
│   Triples → Compute Similarity to Text → Top-40 per Entity  │
└────────────────┬────────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 4: Injection Algorithm                                │
│   Matched Triples → Maximize Score+Diversity → Inject       │
└─────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Entity Linking

### Purpose

Match biomedical entities in text to standardized UMLS Concept Unique Identifiers (CUIs).

### Two-Phase Approach

#### Phase 1A: Embedding-Based Candidate Retrieval

**Model**: SapBERT (biomedical BERT pre-trained on UMLS)

```julia
"""
    retrieve_umls_candidates_embedding(
        entity_text::String,
        sapbert_model::SapBERT,
        umls_index::ANNIndex,
        k::Int = 10
    )

Retrieve top-k UMLS candidates using embedding similarity.
"""
function retrieve_umls_candidates_embedding(
    entity_text::String,
    sapbert_model::SapBERT,
    umls_index::ANNIndex,  # Approximate Nearest Neighbor index
    k::Int = 10
)::Vector{Tuple{String, Float32}}  # Returns (CUI, similarity) pairs

    # Encode entity with SapBERT
    entity_embedding = encode_sapbert(sapbert_model, entity_text)

    # Search UMLS index for nearest neighbors
    neighbors, distances = search_ann(umls_index, entity_embedding, k)

    # Convert distances to similarities (cosine)
    similarities = 1.0f0 .- distances

    # Get CUIs for neighbors
    candidates = [(umls_index.cuis[idx], sim)
                  for (idx, sim) in zip(neighbors, similarities)]

    return candidates
end
```

**Precomputation**:

- Encode all UMLS concepts with SapBERT (one-time cost)
- Build ANN index (e.g., using FAISS or Annoy)
- Store CUI → embedding mapping

#### Phase 1B: String Similarity Filtering

**Method**: Character 3-gram Jaccard similarity

```julia
"""
    filter_candidates_string_similarity(
        entity_text::String,
        candidates::Vector{Tuple{String, Float32}},
        umls_names::Dict{String, String},
        threshold::Float64 = 0.5
    )

Filter candidates using character 3-gram Jaccard similarity.
"""
function filter_candidates_string_similarity(
    entity_text::String,
    candidates::Vector{Tuple{String, Float32}},
    umls_names::Dict{String, String},  # CUI → preferred name
    threshold::Float64 = 0.5
)::Vector{Tuple{String, Float32, Float32}}  # (CUI, emb_sim, string_sim)

    # Extract character 3-grams
    source_3grams = extract_char_3grams(entity_text)

    filtered = Tuple{String, Float32, Float32}[]

    for (cui, emb_sim) in candidates
        candidate_name = umls_names[cui]
        candidate_3grams = extract_char_3grams(candidate_name)

        # Compute Jaccard similarity
        jaccard = jaccard_similarity(source_3grams, candidate_3grams)

        if jaccard >= threshold
            push!(filtered, (cui, emb_sim, jaccard))
        end
    end

    return filtered
end

"""
Extract character 3-grams from text.
"""
function extract_char_3grams(text::String)::Set{String}
    normalized = lowercase(strip(text))
    grams = Set{String}()

    for i in 1:(length(normalized) - 2)
        push!(grams, normalized[i:i+2])
    end

    return grams
end

"""
Compute Jaccard similarity between two sets.
"""
function jaccard_similarity(set1::Set{T}, set2::Set{T})::Float64 where T
    intersection_size = length(intersect(set1, set2))
    union_size = length(union(set1, set2))

    return union_size > 0 ? intersection_size / union_size : 0.0
end
```

**Paper Values**:

- k = 10 candidates from embedding search
- Jaccard threshold = 0.5

---

## Stage 2: Triple Retrieval

### Retrieve Triples from UMLS

For each linked CUI, retrieve all triples where it appears as the head entity.

```julia
"""
    retrieve_umls_triples(
        cui::String,
        umls_kg::UMLSKnowledgeGraph,
        allowed_relations::Set{Symbol}
    )

Retrieve triples for a CUI, filtering by allowed relations.
"""
function retrieve_umls_triples(
    cui::String,
    umls_kg::UMLSKnowledgeGraph,
    allowed_relations::Set{Symbol}
)::Vector{Triple}

    # Query UMLS for triples with this head
    all_triples = query_umls_by_head(umls_kg, cui)

    # Filter by allowed relations
    filtered_triples = filter(all_triples) do triple
        triple.relation in allowed_relations
    end

    return filtered_triples
end
```

**Relation Filtering**:
Paper excludes low-value relations (see Table A1 in Appendix):

- Mapping relations (e.g., `mapped_from`, `mapped_to`)
- Overly specific (e.g., `has_laterality` - almost always "side")
- Redundant (e.g., `has_associated_finding` - tail = subset of head)

**Included Relations** (28 for diabetes):
`:isa`, `:associated_with`, `:cause_of`, `:finding_site_of`, `:has_component`, `:has_disposition`, `:has_method`, `:has_part`, `:has_pathological_process`, `:is_modification_of`, `:part_of`, `:plays_role`, etc.

---

## Stage 3: Contextual Triple Selection

### Semantic Similarity Ranking

```julia
"""
    select_contextual_triples(
        sequence::String,
        entity_triples::Vector{Triple},
        embedding_model::GeminiEmbedding,  # Paper uses Gemini text-embedding-004
        top_k::Int = 40
    )

Select most contextually relevant triples for a sequence.
"""
function select_contextual_triples(
    sequence::String,
    entity_triples::Vector{Triple},
    embedding_model::GeminiEmbedding,
    top_k::Int = 40
)::Vector{Tuple{Triple, Float32}}  # (triple, similarity)

    # Encode sequence
    sequence_embedding = encode_gemini(embedding_model, sequence)

    # Compute similarity for each triple
    triple_scores = map(entity_triples) do triple
        # Linearize triple to text
        triple_text = "$(triple.head) $(triple.relation) $(triple.tail)"

        # Encode triple
        triple_embedding = encode_gemini(embedding_model, triple_text)

        # Cosine similarity
        similarity = cosine_similarity(sequence_embedding, triple_embedding)

        (triple, similarity)
    end

    # Sort by similarity and take top-k
    sort!(triple_scores, by=x->x[2], rev=true)

    return triple_scores[1:min(top_k, length(triple_scores))]
end
```

**Paper Details**:

- Embedding model: Gemini text-embedding-004
- Top-40 triples per entity (balances quality and diversity)
- Cosine similarity metric

---

## Stage 4: Injection Algorithm

### Problem Statement

**Given**: For each sequence, multiple entities with matched triples
**Goal**: Select **one triple per entity** that:

1. Maximizes relevance (similarity score)
2. Maximizes diversity (balance relation types)

### Algorithm: Maximize Score then Diversity

**From Paper Appendix B, Algorithm 1**:

```julia
"""
    injection_algorithm(
        matched_triples::DataFrame,  # Columns: sequence_id, head, relation, tail, score
        α::Float64,                   # Similarity threshold (e.g., 0.55)
        score_bucket_size::Float64,   # Paper uses 0.01 (Algorithm 1)
        relation_bucket_size::Int     # Paper uses 100 (Algorithm 1)
    )

Inject triples using score+diversity optimization.

DataFrame schema:
- sequence_id: Text sequence identifier
- head: Entity text (matched to tokens)
- relation: Relation type (Symbol)
- tail: Tail entity text
- score: Similarity score (0-1)
"""
function injection_algorithm(
    matched_triples::DataFrame,
    α::Float64,
    score_bucket_size::Float64,
    relation_bucket_size::Int
)::DataFrame

    # ========== Preprocessing ==========

    # Step 1: Filter by similarity threshold
    df = filter(row -> row.score >= α, matched_triples)

    # Step 2: Make triples unique (keep highest score per triple)
    df = combine(groupby(df, [:head, :relation, :tail])) do group
        # For duplicate triples across sequences, keep highest score
        group[argmax(group.score), :]
    end

    # ========== Bucketing ==========

    # Step 3: Create score buckets
    # IMPORTANT: Paper uses (max_s - score) so higher scores get LOWER bucket IDs
    # This ensures sorting by ascending score_bucket prioritizes high scores
    max_s = maximum(df.score)
    df.score_bucket = floor.(Int, (max_s .- df.score) ./ score_bucket_size)

    # Step 4: Compute relation frequencies
    relation_counts = combine(groupby(df, :relation), nrow => :count)
    df = leftjoin(df, relation_counts, on=:relation)

    # Step 5: Create relation buckets
    df.relation_bucket = floor.(Int, df.count ./ relation_bucket_size)

    # ========== Selection ==========

    # Step 6: Sort by score bucket (ascending), then relation bucket (ascending), then score (descending)
    # Paper Algorithm 1: Sort by ascending (score_bucket, relation_bucket) and descending score
    # Since higher scores have lower score_bucket IDs (from max_s - score), ascending sort prioritizes high scores
    # Then within same buckets, prefer rarer relations (lower relation_bucket = rarer)
    # Finally, within same buckets, prefer highest score
    sort!(df, [:score_bucket, :relation_bucket, :score], rev=[false, false, true])

    # Step 7: Select one triple per head (greedy)
    selected = DataFrame[]
    seen_heads = Set{String}()

    for row in eachrow(df)
        head_id = "$(row.sequence_id)_$(row.head)"

        if !(head_id in seen_heads)
            push!(selected, row)
            push!(seen_heads, head_id)
        end
    end

    return vcat(selected...)
end
```

### Visual Example

**Before Injection Algorithm**:

```
Entity "diabetes":
  Triple 1: (diabetes, isa, disease)           score=0.82, freq=5000
  Triple 2: (diabetes, isa, syndrome)          score=0.80, freq=5000
  Triple 3: (diabetes, associated_with, obesity) score=0.75, freq=100
  Triple 4: (diabetes, cause_of, neuropathy)   score=0.70, freq=200
```

**After Bucketing** (using paper's formula: `(max_s - score) / bucket_size`):

Assume max_s = 0.85 for this example:

```
Score Buckets (size 0.01, paper default):
  Triple 1 (score=0.82): bucket = floor((0.85-0.82)/0.01) = floor(3.0) = 3
  Triple 2 (score=0.80): bucket = floor((0.85-0.80)/0.01) = floor(5.0) = 5
  Triple 3 (score=0.75): bucket = floor((0.85-0.75)/0.01) = floor(10.0) = 10
  Triple 4 (score=0.70): bucket = floor((0.85-0.70)/0.01) = floor(15.0) = 15

Note: Lower bucket ID = higher score (correct for ascending sort)

Relation Buckets (size 100, paper default):
  :isa (5000 triples): bucket = floor(5000/100) = 50
  :associated_with (100 triples): bucket = floor(100/100) = 1
  :cause_of (200 triples): bucket = floor(200/100) = 2
```

**Selection Order** (score_bucket ASC, relation_bucket ASC, score DESC):

1. Bucket (3, 50): Triple 1 - highest score in highest score bucket
2. Bucket (5, 50): Triple 2 - second highest score
3. Bucket (10, 1): Triple 3 - rarer relation (bucket 1 vs 50)
4. Bucket (15, 2): Triple 4

**Result**: With paper's algorithm, Triple 1 is selected first (highest score), but if Triple 3 had score 0.82 (bucket 3):
- Bucket (3, 1): Triple 3 ← SELECTED (rarer relation in same score bucket)
- Bucket (3, 50): Triple 1

The algorithm correctly balances score and relation diversity!

---

## Complete Integration

### Full Pipeline

```julia
"""
    prepare_training_data_with_seed_kg(
        corpus::Vector{String},
        umls_kg::UMLSKnowledgeGraph,
        config::SeedInjectionConfig
    )

Complete pipeline: text → entity linking → injection → chain graphs.
"""
function prepare_training_data_with_seed_kg(
    corpus::Vector{String},
    umls_kg::UMLSKnowledgeGraph,
    config::SeedInjectionConfig
)::Tuple{Vector{LeafyChainGraph}, DataFrame}

    # Stage 0: Entity Discovery (using Helper LLM)
    entity_mentions = discover_entities_llm(corpus, config.helper_llm)

    # Create dataframe to track all matches
    all_matches = DataFrame(
        sequence_id = String[],
        sequence_text = String[],
        head = String[],
        cui = String[],
        relation = Symbol[],
        tail = String[],
        score = Float64[]
    )

    # Process each sequence
    for (seq_id, sequence) in enumerate(corpus)
        # Get entities for this sequence
        entities = filter(e -> e.sequence_id == seq_id, entity_mentions)

        for entity in entities
            # Stage 1: Entity Linking
            candidates = retrieve_umls_candidates_embedding(
                entity.text, config.sapbert, config.umls_index
            )
            linked = filter_candidates_string_similarity(
                entity.text, candidates, umls_kg.names, 0.5
            )

            for (cui, emb_sim, str_sim) in linked
                # Stage 2: Triple Retrieval
                triples = retrieve_umls_triples(
                    cui, umls_kg, config.allowed_relations
                )

                # Stage 3: Contextual Selection
                contextual = select_contextual_triples(
                    sequence, triples, config.embedding_model, 40
                )

                # Add to matches
                for (triple, sim_score) in contextual
                    push!(all_matches, (
                        sequence_id = string(seq_id),
                        sequence_text = sequence,
                        head = entity.text,
                        cui = cui,
                        relation = triple.relation,
                        tail = triple.tail,
                        score = sim_score
                    ))
                end
            end
        end
    end

    # Stage 4: Injection Algorithm
    seed_kg = injection_algorithm(
        all_matches,
        config.alpha,
        config.score_bucket_size,
        config.relation_bucket_size
    )

    # Build leafy chain graphs with injections
    graphs = build_graphs_with_injections(corpus, seed_kg, config)

    return graphs, seed_kg
end
```

### Configuration

```julia
"""
Configuration for seed KG injection.
"""
struct SeedInjectionConfig
    # Entity linking
    sapbert::SapBERT
    umls_index::ANNIndex
    string_sim_threshold::Float64

    # Triple retrieval
    umls_kg::UMLSKnowledgeGraph
    allowed_relations::Set{Symbol}

    # Contextual selection
    embedding_model::GeminiEmbedding
    top_k_per_entity::Int

    # Injection algorithm
    alpha::Float64                    # Similarity threshold (e.g., 0.55)
    score_bucket_size::Float64        # e.g., 0.05
    relation_bucket_size::Int         # e.g., 20

    # Helper LLM for entity discovery
    helper_llm::LLMClient
end

function default_seed_injection_config()
    return SeedInjectionConfig(
        sapbert = load_sapbert(),
        umls_index = build_umls_index(),
        string_sim_threshold = 0.5,
        umls_kg = load_umls_kg(),
        allowed_relations = load_allowed_relations(),
        embedding_model = GeminiEmbedding("text-embedding-004"),
        top_k_per_entity = 40,
        alpha = 0.55,
        score_bucket_size = 0.05,
        relation_bucket_size = 20,
        helper_llm = create_llm_client("qwen3-32b")
    )
end
```

---

## Worked Example: Diabetes Dataset

**Input Sequence**:

```
"Diabetes mellitus is a chronic metabolic disorder characterized by hyperglycemia."
```

**Stage 1: Entity Linking**

```
Discovered entities: ["diabetes mellitus", "hyperglycemia"]

diabetes mellitus →
  Embedding candidates (top 3):
    C0011849 (Diabetes Mellitus) - emb_sim: 0.95
    C0011860 (Diabetes Mellitus, Non-Insulin-Dependent) - emb_sim: 0.88
    C0011854 (Diabetes Mellitus, Insulin-Dependent) - emb_sim: 0.87
  String filtering (Jaccard > 0.5):
    C0011849 ✓ (jaccard: 1.0)

hyperglycemia →
  C0020456 (Hyperglycemia) - emb_sim: 0.98, jaccard: 1.0
```

**Stage 2: Triple Retrieval**

```
C0011849 triples (filtered): 150 triples
C0020456 triples (filtered): 80 triples
```

**Stage 3: Contextual Selection**

```
For C0011849, top-3 contextual matches:
  1. (diabetes mellitus, isa, disease) - sim: 0.82
  2. (diabetes mellitus, isa, metabolic disorder) - sim: 0.80
  3. (diabetes mellitus, has_finding_site, pancreas) - sim: 0.65
  ...

For C0020456, top-3:
  1. (hyperglycemia, finding_site_of, blood) - sim: 0.75
  2. (hyperglycemia, associated_with, diabetes) - sim: 0.73
  ...
```

**Stage 4: Injection**

```
After bucketing and selection:
  diabetes mellitus → (diabetes mellitus, isa, disease) ← Selected
  hyperglycemia → (hyperglycemia, finding_site_of, blood) ← Selected
```

**Final Graph**:

```
Roots: [diabetes, mellitus, is, a, chronic, metabolic, disorder, ..., hyperglycemia]
Leaves:
  Root 0 "diabetes" → [disease, <pad>, <pad>, ...] (relation: isa)
  Root 9 "hyperglycemia" → [blood, <pad>, <pad>, ...] (relation: finding_site_of)
```

---

## Statistics and Validation

### Paper Results (Diabetes Dataset)

**Input**:

- 350k abstracts (124.7M tokens)
- UMLS (SNOMED CT + Gene Ontology)
- α = 0.55

**Output**:

- 28,533 seed triples
- 28 unique relations
- Mean similarity: 0.613
- Median similarity: 0.605
- Max similarity: 0.848

**Relation Distribution**:
Most frequent: `:isa`, `:associated_with`, `:cause_of`
Rarest: `:has_method`, `:is_modification_of`

---

## Testing

```julia
@testset "Seed KG Injection" begin

    @testset "Entity Linking" begin
        text = "diabetes mellitus"
        candidates = retrieve_umls_candidates_embedding(text, sapbert, index, 10)
        @test length(candidates) == 10
        @test all(s -> 0 <= s <= 1, [sim for (_, sim) in candidates])

        filtered = filter_candidates_string_similarity(text, candidates, names, 0.5)
        @test length(filtered) <= length(candidates)
    end

    @testset "Injection Algorithm" begin
        df = create_test_matched_triples()
        seed = injection_algorithm(df, 0.55, 0.05, 20)

        # One triple per head
        @test length(seed) <= length(unique(df.head))

        # All above threshold
        @test all(seed.score .>= 0.55)
    end
end
```

---

## Implementation Checklist

- [ ] Implement SapBERT integration
- [ ] Build ANN index for UMLS
- [ ] Implement character 3-gram extraction
- [ ] Implement Jaccard similarity
- [ ] Implement UMLS triple retrieval
- [ ] Implement Gemini embedding integration
- [ ] Implement contextual selection
- [ ] Implement injection algorithm with bucketing
- [ ] Integrate with leafy chain graph construction
- [ ] Write comprehensive tests
- [ ] Validate on diabetes dataset

---

**Related Documents**:

- → [Doc 02: Leafy Chain Graphs](02-leafy-chain-graphs.md)
- → [Doc 07: MNM Training](07-training-mnm.md)
- → [Doc 09: Triple Extraction](09-triple-extraction.md)
