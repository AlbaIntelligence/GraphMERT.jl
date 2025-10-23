# Document 10: Evaluation Metrics
## Measuring Knowledge Graph Quality

**Status**: 🟡 **Partial Implementation**
**Priority**: P1 (High - needed for validation)
**Paper Reference**: Section 5.2, Table 3
**Existing Code**: `evaluation/*.jl` (partial stubs)

---

## Overview

GraphMERT uses three complementary metrics to evaluate extracted knowledge graphs:

1. **FActScore*** - Measures **factual accuracy** against source text
2. **ValidityScore** - Measures **ontological correctness** of relations
3. **GraphRAG** - Measures **downstream utility** for question answering

**Key Insight**: Traditional NLP metrics (BLEU, ROUGE) inadequate for KG evaluation

---

## Part 1: FActScore* (Factuality)

### What is FActScore*?

**Original FActScore**: Measures factuality of generated text by decomposing into atomic facts

**FActScore***: Adapted for knowledge graphs

**Purpose**: Answer "Does this triple accurately reflect the source text?"

### Algorithm

```
For each extracted triple (h, r, t):
    1. Retrieve source text passage
    2. Convert triple to natural language claim:
       "The relation r holds between h and t"
    3. Use LLM to verify claim against source
    4. Score: Supported / Not Supported / Contradicted
```

### Mathematical Definition

$$\text{FActScore*} = \frac{\text{# Supported Triples}}{\text{# Total Triples}}$$

**Range**: [0, 1]
**Higher is better**
**Paper Target**: **69.8%** (diabetes domain)

---

### Implementation

```julia
"""
    calculate_factscore_star(
        triples::Vector{ExtractionTriple},
        source_texts::Dict{String,String},
        llm_client::LLMClient
    )

Calculate FActScore* for extracted triples.
"""
function calculate_factscore_star(
    triples::Vector{ExtractionTriple},
    source_texts::Dict{String,String},
    llm_client::LLMClient
)
    supported = 0
    contradicted = 0
    not_supported = 0

    for triple in triples
        # 1. Get source text
        source_text = source_texts[triple.source_sequence_id]

        # 2. Convert triple to claim
        claim = format_triple_as_claim(triple)
        # Example: "Diabetes mellitus is a disease"

        # 3. Verify with LLM
        verification = verify_claim_with_llm(claim, source_text, llm_client)

        if verification == :supported
            supported += 1
        elseif verification == :contradicted
            contradicted += 1
        else
            not_supported += 1
        end
    end

    total = length(triples)
    score = total > 0 ? supported / total : 0.0

    return FActScore(
        score = score,
        precision = score,  # Same as score for FActScore*
        recall = supported / (supported + not_supported),
        f1 = 2 * score * recall / (score + recall),
        total_facts = total,
        correct_facts = supported,
        incorrect_facts = contradicted + not_supported
    )
end
```

### LLM Verification Prompt

```julia
function verify_claim_with_llm(
    claim::String,
    source_text::String,
    llm_client::LLMClient
)
    prompt = """
    Source Text:
    \"\"\"
    $source_text
    \"\"\"

    Claim to Verify:
    \"\"\"
    $claim
    \"\"\"

    Question: Is this claim supported by the source text?

    Answer with ONLY one of:
    - SUPPORTED: The claim is clearly stated or implied in the source
    - CONTRADICTED: The claim contradicts information in the source
    - NOT_SUPPORTED: The claim is not mentioned in the source

    Answer:
    """

    response = llm_client(prompt)

    if contains(response, "SUPPORTED")
        return :supported
    elseif contains(response, "CONTRADICTED")
        return :contradicted
    else
        return :not_supported
    end
end
```

### Format Triple as Claim

```julia
function format_triple_as_claim(triple::ExtractionTriple)
    relation_text = relation_to_text(triple.relation)
    return "$(triple.head) $(relation_text) $(triple.tail)"
end

function relation_to_text(relation::Symbol)
    # Convert relation symbols to natural language
    relation_map = Dict(
        :isa => "is a",
        :associated_with => "is associated with",
        :cause_of => "is a cause of",
        :has_symptom => "has symptom",
        :treats => "treats",
        :prevents => "prevents",
        # ... 28 relations total
    )

    return get(relation_map, relation, string(relation))
end
```

---

### Example Evaluation

```
Triple: (Diabetes mellitus, isa, disease)

Source Text: "Diabetes mellitus is a metabolic disorder characterized
by high blood sugar levels."

Claim: "Diabetes mellitus is a disease"

LLM Verification:
→ SUPPORTED (source says "metabolic disorder" which implies disease)

FActScore*: +1
```

```
Triple: (Metformin, treats, cancer)

Source Text: "Metformin is a first-line medication for type 2 diabetes."

Claim: "Metformin treats cancer"

LLM Verification:
→ NOT_SUPPORTED (source doesn't mention cancer)

FActScore*: +0
```

---

## Part 2: ValidityScore (Ontological Correctness)

### What is ValidityScore?

**Purpose**: Measure whether triples use relations correctly according to domain ontology

**Example Problem**:
- ✅ Good: `(diabetes, isa, disease)`
- ❌ Bad: `(diabetes, treats, patient)` ← Wrong relation type!

### Algorithm

```
For each extracted triple (h, r, t):
    1. Check relation r definition in ontology (UMLS)
    2. Verify h matches expected head type
    3. Verify t matches expected tail type
    4. LLM judges: "Is this relation usage valid?"
    5. Score: Yes / Maybe / No
```

### Mathematical Definition

$$\text{ValidityScore} = \frac{\text{# Yes} + 0.5 \cdot \text{# Maybe}}{\text{# Total Triples}}$$

**Range**: [0, 1]
**Higher is better**
**Paper Target**: **68.8%** (diabetes domain)

---

### Implementation

```julia
"""
    calculate_validity_score(
        triples::Vector{ExtractionTriple},
        umls_ontology::UMLSOntology,
        llm_client::LLMClient
    )

Calculate ValidityScore for extracted triples.
"""
function calculate_validity_score(
    triples::Vector{ExtractionTriple},
    umls_ontology::UMLSOntology,
    llm_client::LLMClient
)
    yes_count = 0
    maybe_count = 0
    no_count = 0

    for triple in triples
        # 1. Get relation definition from UMLS
        relation_def = get_relation_definition(umls_ontology, triple.relation)

        # 2. Check semantic types
        head_type = get_semantic_type(umls_ontology, triple.head)
        tail_type = get_semantic_type(umls_ontology, triple.tail)

        # 3. Verify with LLM
        validity = verify_relation_validity_with_llm(
            triple, relation_def, head_type, tail_type, llm_client
        )

        if validity == :yes
            yes_count += 1
        elseif validity == :maybe
            maybe_count += 1
        else
            no_count += 1
        end
    end

    total = length(triples)
    score = total > 0 ? (yes_count + 0.5 * maybe_count) / total : 0.0

    return ValidityScore(
        score = score,
        valid_relations = yes_count,
        total_relations = total,
        invalid_relations = no_count,
        yes_count = yes_count,
        maybe_count = maybe_count,
        no_count = no_count
    )
end
```

### LLM Verification Prompt

```julia
function verify_relation_validity_with_llm(
    triple::ExtractionTriple,
    relation_def::String,
    head_type::String,
    tail_type::String,
    llm_client::LLMClient
)
    prompt = """
    Relation: $(triple.relation)
    Definition: $relation_def

    Expected:
    - Head type: $head_type
    - Tail type: $tail_type

    Actual Triple:
    - Head: $(triple.head)
    - Tail: $(triple.tail)

    Question: Is this usage of the relation ontologically correct?

    Answer with ONLY one of:
    - YES: The relation is used correctly
    - MAYBE: The relation might be correct, but it's ambiguous
    - NO: The relation is used incorrectly

    Answer:
    """

    response = llm_client(prompt)

    if contains(response, "YES")
        return :yes
    elseif contains(response, "MAYBE")
        return :maybe
    else
        return :no
    end
end
```

---

### Example Evaluation

```
Triple: (Diabetes mellitus, isa, disease)

Relation Definition: "isa" links a specific concept to its parent category

Expected Types:
- Head: Disorder/Disease
- Tail: General Category

Actual:
- Head: "Diabetes mellitus" (Disorder) ✓
- Tail: "disease" (General Category) ✓

LLM Judgment: YES

ValidityScore: +1.0
```

```
Triple: (Insulin, prevents, diabetes)

Relation Definition: "prevents" indicates a preventive relationship

Expected Types:
- Head: Pharmacological Substance / Procedure
- Tail: Disease/Disorder

Actual:
- Head: "Insulin" (Hormone/Drug) ✓
- Tail: "diabetes" (Disease) ✓

LLM Judgment: MAYBE (insulin manages, doesn't prevent)

ValidityScore: +0.5
```

```
Triple: (Blood sugar, treats, patient)

Relation Definition: "treats" indicates therapeutic relationship

Expected Types:
- Head: Pharmacological Substance / Procedure
- Tail: Disease/Disorder

Actual:
- Head: "Blood sugar" (Laboratory Test) ✗
- Tail: "patient" (Person) ✗

LLM Judgment: NO (wrong types, wrong relation)

ValidityScore: +0
```

---

## Part 3: GraphRAG (Downstream Utility)

### What is GraphRAG?

**Purpose**: Measure how useful the KG is for **question answering** (QA)

**Insight**: A good KG should help answer questions about its domain

### Algorithm

```
For each test question:
    1. Retrieve relevant triples from extracted KG
    2. Use LLM to generate answer from triples
    3. Compare answer to ground truth
    4. Score: Correct / Incorrect
```

### Mathematical Definition

$$\text{GraphRAG Score} = \frac{\text{# Correct Answers}}{\text{# Total Questions}}$$

**Range**: [0, 1]
**Higher is better**
**Paper Result**: Improves QA accuracy significantly vs. text-only retrieval

---

### Implementation

```julia
"""
    calculate_graphrag_score(
        kg::KnowledgeGraph,
        test_questions::Vector{Tuple{String,String}},  # (question, answer)
        llm_client::LLMClient
    )

Calculate GraphRAG score using KG-augmented QA.
"""
function calculate_graphrag_score(
    kg::KnowledgeGraph,
    test_questions::Vector{Tuple{String,String}},
    llm_client::LLMClient
)
    correct = 0

    for (question, ground_truth) in test_questions
        # 1. Retrieve relevant triples
        relevant_triples = retrieve_triples_for_question(kg, question)

        # 2. Format as context
        context = format_triples_as_context(relevant_triples)

        # 3. Generate answer using LLM + KG
        generated_answer = generate_answer_with_kg(question, context, llm_client)

        # 4. Check correctness
        is_correct = verify_answer(generated_answer, ground_truth, llm_client)

        if is_correct
            correct += 1
        end
    end

    total = length(test_questions)
    score = total > 0 ? correct / total : 0.0

    return GraphRAGScore(
        score = score,
        retrieval_accuracy = 1.0,  # Simplified
        generation_quality = score,
        overall_performance = score,
        num_queries = total,
        correct_answers = correct
    )
end
```

### Triple Retrieval

```julia
function retrieve_triples_for_question(
    kg::KnowledgeGraph,
    question::String;
    top_k::Int=10
)
    # 1. Extract entities from question
    question_entities = extract_entities(question)

    # 2. Find triples mentioning those entities
    relevant = ExtractionTriple[]

    for triple in kg.relations
        if any(e -> contains(triple.head, e) || contains(triple.tail, e),
               question_entities)
            push!(relevant, triple)
        end
    end

    # 3. Rank by similarity to question (using embeddings)
    ranked = rank_by_similarity(relevant, question)

    return ranked[1:min(top_k, length(ranked))]
end
```

### Answer Generation

```julia
function generate_answer_with_kg(
    question::String,
    kg_context::String,
    llm_client::LLMClient
)
    prompt = """
    Knowledge Graph Context:
    \"\"\"
    $kg_context
    \"\"\"

    Question: $question

    Instructions: Answer the question using ONLY the information provided
    in the knowledge graph context above. If the answer is not in the
    context, say "I don't know."

    Answer:
    """

    return llm_client(prompt)
end
```

---

### Example Evaluation

```
Question: "What are the symptoms of diabetes?"

Retrieved Triples:
1. (diabetes, has_symptom, increased thirst)
2. (diabetes, has_symptom, frequent urination)
3. (diabetes, has_symptom, fatigue)
4. (diabetes, isa, metabolic disorder)

Context: "diabetes has_symptom increased thirst. diabetes has_symptom
frequent urination. diabetes has_symptom fatigue. diabetes isa metabolic disorder."

Generated Answer: "The symptoms of diabetes include increased thirst,
frequent urination, and fatigue."

Ground Truth: "Diabetes symptoms include increased thirst, frequent urination,
extreme hunger, unexplained weight loss, and fatigue."

Verification: CORRECT (covers main symptoms, even if not exhaustive)

GraphRAG Score: +1
```

---

## Part 4: Comparative Baselines

### GraphMERT vs. Baselines (Paper Results)

**Diabetes Domain**:

| Method               | FActScore* ↑ | ValidityScore ↑ |
| -------------------- | ------------ | --------------- |
| **LLM-only** (GPT-4) | 40.2%        | 43.0%           |
| **Text+LLM**         | 55.6%        | 58.2%           |
| **GraphMERT**        | **69.8%**    | **68.8%**       |

**Improvement**:
- **+29.6%** FActScore* over LLM-only
- **+25.8%** ValidityScore over LLM-only
- **+14.2%** FActScore* over text-based extraction

---

## Part 5: Implementation Checklist

### Data Structures

```julia
struct FActScore
    score::Float64
    precision::Float64
    recall::Float64
    f1::Float64
    total_facts::Int
    correct_facts::Int
    incorrect_facts::Int
end

struct ValidityScore
    score::Float64
    valid_relations::Int
    total_relations::Int
    invalid_relations::Int
    yes_count::Int
    maybe_count::Int
    no_count::Int
end

struct GraphRAGScore
    score::Float64
    retrieval_accuracy::Float64
    generation_quality::Float64
    overall_performance::Float64
    num_queries::Int
    correct_answers::Int
end
```

### Required Components

- [ ] FActScore* calculation
  - [ ] Triple-to-claim conversion
  - [ ] LLM verification
  - [ ] Score aggregation

- [ ] ValidityScore calculation
  - [ ] UMLS ontology integration
  - [ ] Semantic type checking
  - [ ] LLM validity verification

- [ ] GraphRAG evaluation
  - [ ] Triple retrieval for questions
  - [ ] Answer generation
  - [ ] Answer verification

- [ ] LLM client integration
  - [ ] API calls with caching
  - [ ] Prompt templates
  - [ ] Response parsing

- [ ] Test question datasets
  - [ ] Diabetes QA pairs
  - [ ] Ground truth annotations
  - [ ] Domain coverage

---

## Part 6: Testing Strategy

### Unit Tests

```julia
@testset "Evaluation Metrics" begin
    @testset "FActScore*" begin
        triples = [
            ExtractionTriple("diabetes", :isa, "disease", 0.95, ...),
            ExtractionTriple("insulin", :treats, "diabetes", 0.90, ...)
        ]

        sources = Dict(
            "seq1" => "Diabetes is a disease.",
            "seq2" => "Insulin treats diabetes."
        )

        score = calculate_factscore_star(triples, sources, llm_client)
        @test 0.0 <= score.score <= 1.0
        @test score.total_facts == 2
    end

    @testset "ValidityScore" begin
        triples = [...]
        score = calculate_validity_score(triples, umls, llm_client)
        @test 0.0 <= score.score <= 1.0
    end

    @testset "GraphRAG" begin
        kg = KnowledgeGraph(entities, relations, metadata)
        questions = [
            ("What is diabetes?", "A metabolic disorder"),
            ("What treats diabetes?", "Insulin")
        ]

        score = calculate_graphrag_score(kg, questions, llm_client)
        @test 0.0 <= score.score <= 1.0
    end
end
```

---

## Part 7: Optimization Strategies

### Reduce LLM Calls

**Problem**: Evaluating 100k triples = 100k LLM calls = expensive!

**Solutions**:

1. **Caching**:
```julia
cache = Dict{Tuple{String,String}, Symbol}()  # (claim, source) → verdict

function verify_claim_with_cache(claim, source, llm_client, cache)
    key = (claim, source)
    if haskey(cache, key)
        return cache[key]
    end

    verdict = verify_claim_with_llm(claim, source, llm_client)
    cache[key] = verdict
    return verdict
end
```

2. **Sampling**:
```julia
# Evaluate random sample instead of all triples
sample_size = 1000
sampled_triples = sample(all_triples, sample_size)
score = calculate_factscore_star(sampled_triples, sources, llm_client)
```

3. **Batching**:
```julia
# Verify multiple claims in one LLM call
function batch_verify_claims(claims, sources, llm_client)
    prompt = """
    Verify each of the following claims:

    1. Claim: "..." Source: "..."
    2. Claim: "..." Source: "..."
    ...

    Answer with: 1. SUPPORTED, 2. NOT_SUPPORTED, ...
    """
    # ... parse response
end
```

---

## Summary

**Evaluation Metrics for GraphMERT**:

🎯 **Three Complementary Metrics**:
1. **FActScore***: Factual accuracy (69.8%)
2. **ValidityScore**: Ontological correctness (68.8%)
3. **GraphRAG**: Downstream utility

🔬 **Why These Metrics?**:
- FActScore* → Checks "Is this triple true?"
- ValidityScore → Checks "Is this relation used correctly?"
- GraphRAG → Checks "Is this KG useful?"

📊 **Paper Benchmarks**:
- GraphMERT beats LLM-only by **~29%**
- GraphMERT beats text-extraction by **~14%**

**Implementation Priority**: P1 - Needed to validate system against paper

---

**Related Documents**:
- → [Doc 01: Architecture Overview](01-architecture-overview.md)
- → [Doc 09: Triple Extraction](09-triple-extraction.md) - What's being evaluated
- → [Implementation Roadmap](00-IMPLEMENTATION-ROADMAP.md) - When to implement
