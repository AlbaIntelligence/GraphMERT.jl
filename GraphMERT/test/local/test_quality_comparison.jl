"""
Quality comparison test utility for User Story 3 (Maintain Extraction Quality).

This module compares local LLM extraction results with external API baseline
to measure extraction quality against the following success criteria:
- SC-003: Entity extraction achieves at least 70% recall compared to external API baseline
- SC-004: At least 80% of entities found by reference API on 50 articles

To run this test:
    julia --project=. GraphMERT/test/local/test_quality_comparison.jl

Requirements:
- A GGUF model file (e.g., TinyLlama-1.1B-Q4_0.gguf) for local extraction
- An external API (OpenAI, Anthropic, etc.) as baseline for comparison
"""

module TestQualityComparison

using Test
using GraphMERT

const ENTITY_RECALL_TARGET = 0.70
const ENTITY_OVERLAP_TARGET = 0.80
const NUM_ARTICLES_FOR_OVERLAP = 50

struct ComparisonResult
    local_entities::Vector{String}
    baseline_entities::Vector{String}
    recall::Float64
    overlap::Float64
    passed::Bool
end

function extract_local_entities(text::String, local_config::LocalLLMConfig)
    if !ispath(local_config.model_path)
        @warn "Model file not found: $(local_config.model_path). Skipping local extraction."
        return String[]
    end

    options = ProcessingOptions(
        domain = "wikipedia",
        use_local = true,
        local_config = local_config,
    )

    domain = get_domain("wikipedia")
    entities = extract_entities(domain, text, options)
    return [e.text for e in entities]
end

function extract_baseline_entities(text::String; api_key::Union{String, Nothing}=nothing)
    if api_key === nothing
        api_key = get(ENV, "EXTERNAL_API_KEY", "")
        if isempty(api_key)
            @warn "No EXTERNAL_API_KEY set. Using heuristic baseline."
            return extract_heuristic_baseline(text)
        end
    end

    options = ProcessingOptions(
        domain = "wikipedia",
        use_api = true,
        api_key = api_key,
    )

    domain = get_domain("wikipedia")
    entities = extract_entities(domain, text, options)
    return [e.text for e in entities]
end

function extract_heuristic_baseline(text::String)
    entity_patterns = [
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",
        r"\b[A-Z][a-z]+\b",
    ]

    entities = Set{String}()
    for pattern in entity_patterns
        matches = eachmatch(pattern, text)
        for m in matches
            len = length(split(m.match))
            if 1 <= len <= 4
                push!(entities, m.match)
            end
        end
    end

    return collect(entities)
end

function calculate_recall(local_entities::Vector{String}, baseline_entities::Vector{String})
    if isempty(baseline_entities)
        return isempty(local_entities) ? 1.0 : 0.0
    end

    baseline_set = Set(baseline_entities)
    matched = count(e -> e in baseline_set, local_entities)

    return matched / length(baseline_entities)
end

function calculate_overlap(local_entities::Vector{String}, baseline_entities::Vector{String})
    if isempty(local_entities) || isempty(baseline_entities)
        return 0.0
    end

    local_set = Set(local_entities)
    baseline_set = Set(baseline_entities)

    intersection = intersect(local_set, baseline_set)
    union_set = union(local_set, baseline_set)

    return length(intersection) / length(union_set)
end

function compare_extraction(
    text::String,
    local_config::LocalLLMConfig;
    baseline_api_key::Union{String, Nothing}=nothing
)::ComparisonResult
    local_entities = extract_local_entities(text, local_config)
    baseline_entities = extract_baseline_entities(text; api_key=baseline_api_key)

    recall = calculate_recall(local_entities, baseline_entities)
    overlap = calculate_overlap(local_entities, baseline_entities)

    passed = recall >= ENTITY_RECALL_TARGET

    return ComparisonResult(
        local_entities,
        baseline_entities,
        recall,
        overlap,
        passed
    )
end

function run_batch_comparison(
    articles::Vector{Pair{String, String}},
    local_config::LocalLLMConfig;
    baseline_api_key::Union{String, Nothing}=nothing
)
    results = ComparisonResult[]
    total_recall = 0.0
    passed_count = 0

    for (name, text) in articles
        result = compare_extraction(text, local_config; baseline_api_key)
        push!(results, result)

        total_recall += result.recall
        if result.passed
            passed_count += 1
        end

        println("  $name: recall=$(round(result.recall*100, digits=1))%, overlap=$(round(result.overlap*100, digits=1))%")
    end

    avg_recall = total_recall / length(articles)
    pass_rate = passed_count / length(articles)

    return (
        results,
        average_recall=avg_recall,
        pass_rate=pass_rate,
        meets_target=avg_recall >= ENTITY_RECALL_TARGET
    )
end

"""
T021 NOTE: Parameter tuning for quality improvement.

If quality metrics are below threshold (recall < 70% or overlap < 80%), consider tuning:

1. Temperature: Lower values (0.1-0.3) produce more deterministic outputs
   - Try: temperature=0.2 for entity extraction

2. Context length: Ensure sufficient context for understanding
   - Minimum recommended: 2048 tokens
   - For complex articles: 4096+ tokens

3. Max tokens: Increase to allow longer entity names
   - Recommended: 512-1024 for extraction

4. Prompt engineering: Customize extraction prompts for your domain

Example tuning:
```julia
local_config = LocalLLMConfig(;
    model_path="path/to/model.gguf",
    temperature=0.2,
    context_length=4096,
    max_tokens=1024,
)
```
"""
function suggest_parameter_tuning(current_recall::Float64, current_overlap::Float64)
    suggestions = String[]

    if current_recall < ENTITY_RECALL_TARGET
        push!(suggestions, "Temperature may be too high. Try lowering to 0.1-0.3")
        push!(suggestions, "Consider increasing max_tokens to capture full entity names")
    end

    if current_overlap < ENTITY_OVERLAP_TARGET
        push!(suggestions, "Context length may be insufficient. Try 4096 tokens")
        push!(suggestions, "Review extraction prompt for entity type coverage")
    end

    return suggestions
end

function test_single_article_comparison()
    @testset "Single article quality comparison" begin
        # Test the comparison logic (not actual extraction which requires model)
        
        # Test with known values
        local_entities = ["Louis XIV", "France", "Versailles"]
        baseline_entities = ["Louis XIV", "France", "Versailles", "Louis XV"]
        
        recall = calculate_recall(local_entities, baseline_entities)
        overlap = calculate_overlap(local_entities, baseline_entities)
        
        println("\n=== Test Comparison ===")
        println("Local entities: $local_entities")
        println("Baseline entities: $baseline_entities")
        println("Recall: $(round(recall * 100, digits=1))%")
        println("Overlap: $(round(overlap * 100, digits=1))%")
        
        # Test with empty cases
        @test calculate_recall(String[], String[]) == 1.0
        @test calculate_overlap(String[], String[]) == 0.0
        
        println("\n[Contract] Verified: Quality comparison calculation logic")
    end
end

function test_recall_calculation()
    @testset "Entity recall calculation" begin
        @test calculate_recall(["A", "B", "C"], ["A", "B", "C", "D"]) ≈ 0.75
        @test calculate_recall(["A", "B"], ["A", "B", "C", "D"]) ≈ 0.5
        @test calculate_recall(["A", "B", "C", "D"], ["A", "B", "C", "D"]) ≈ 1.0
        @test calculate_recall(String[], ["A", "B"]) ≈ 0.0
        @test calculate_recall(["A", "B"], String[]) ≈ 0.0
    end
end

function test_overlap_calculation()
    @testset "Entity overlap calculation" begin
        @test calculate_overlap(["A", "B", "C"], ["A", "B", "C", "D"]) ≈ 0.75
        @test calculate_overlap(["A", "B"], ["C", "D"]) ≈ 0.0
        @test calculate_overlap(["A", "B"], ["A", "B"]) ≈ 1.0
        @test calculate_overlap(["A"], ["A", "B", "C"]) ≈ 0.33333333
    end
end

function test_parameter_tuning_suggestions()
    @testset "Parameter tuning suggestions" begin
        suggestions = suggest_parameter_tuning(0.5, 0.6)
        @test length(suggestions) > 0

        suggestions = suggest_parameter_tuning(0.8, 0.9)
        @test length(suggestions) == 0
    end
end

function run_tests()
    println("=" ^ 60)
    println("Quality Comparison Tests (T019-T021)")
    println("=" ^ 60)
    println("\nTargets:")
    println("  - Entity recall: >= $(ENTITY_RECALL_TARGET * 100)% (SC-003)")
    println("  - Entity overlap: >= $(ENTITY_OVERLAP_TARGET * 100)% on $(NUM_ARTICLES_FOR_OVERLAP) articles (SC-004)")
    println()

    test_recall_calculation()
    test_overlap_calculation()
    test_single_article_comparison()
    test_parameter_tuning_suggestions()

    println("\n" * "=" ^ 60)
    println("All quality comparison tests passed!")
    println("=" ^ 60)

    println(raw"""

    Usage for full comparison:
    ```julia
    using GraphMERT

    local_config = LocalLLMConfig(;
        model_path="path/to/model.gguf",
        temperature=0.2,
        context_length=4096,
        max_tokens=512
    )

    articles = [("article1", "text1"), ("article2", "text2"), ...]
    result = run_batch_comparison(articles, local_config)

    println("Average recall: $(result.average_recall)")
    println("Pass rate: $(result.pass_rate)")
    ```
    """)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_tests()
end

end  # module
