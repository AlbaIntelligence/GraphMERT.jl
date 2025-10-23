"""
ValidityScore evaluation metric for GraphMERT.jl

This module implements the ValidityScore metric as described in the GraphMERT paper.
ValidityScore evaluates the ontological validity of extracted triples by checking
whether they align with established biomedical knowledge bases.

The metric is defined as:
ValidityScore = count(yes) / total_triples

where "yes" indicates that a triple is ontologically valid according to
established biomedical knowledge.
"""

# Types are defined in the main GraphMERT module
# using ..LLM: make_llm_request, HelperLLMClient
# using ..Biomedical: link_entity_to_umls
using Distributions: Normal, quantile, TDist, chi2cdf
using Statistics: mean, std, var

"""
    ValidityScoreResult

Result of ValidityScore evaluation containing detailed metrics.
"""
struct ValidityScoreResult
    triple_validity::Vector{Symbol}
    justifications::Vector{String}
    validity_score::Float64
    valid_triples::Int
    total_triples::Int
    metadata::Dict{String,Any}
end

# Use filter_triples_by_confidence from factscore.jl

"""
    evaluate_validity(kg::KnowledgeGraph;
                     llm_client::Union{HelperLLMClient, Nothing}=nothing,
                     umls_client::Union{UMLSClient, Nothing}=nothing,
                     confidence_threshold::Float64=0.5)

Calculate ValidityScore for a knowledge graph.

ValidityScore evaluates whether extracted triples are ontologically valid
by checking alignment with established biomedical knowledge.

# Arguments
- `kg::KnowledgeGraph`: Knowledge graph to evaluate
- `llm_client::Union{HelperLLMClient, Nothing}`: Optional LLM client for validation
- `umls_client::Union{UMLSClient, Nothing}`: Optional UMLS client for ontology checking
- `confidence_threshold::Float64`: Minimum confidence for triple inclusion

# Returns
- `ValidityScoreResult`: Detailed evaluation results

# Mathematical Definition
ValidityScore = count(yes) / total_triples

where "yes" indicates ontological validity of the triple.
"""
function evaluate_validity(
    kg::GraphMERT.KnowledgeGraph;
    llm_client::Union{HelperLLMClient,Nothing} = nothing,
    umls_client::Union{UMLSClient,Nothing} = nothing,
    confidence_threshold::Float64 = 0.5,
)
    @info "Starting ValidityScore evaluation for $(length(kg.entities)) entities and $(length(kg.relations)) relations"

    # Filter triples by confidence threshold
    high_confidence_triples = filter_triples_by_confidence(kg, confidence_threshold)

    @info "Evaluating $(length(high_confidence_triples)) high-confidence triples"

    # Evaluate each triple
    triple_validity = Vector{Symbol}()
    justifications = Vector{String}()

    for (head_idx, rel_idx, tail_idx) in high_confidence_triples
        head_entity = kg.entities[head_idx]
        tail_entity = kg.entities[tail_idx]
        relation = kg.relations[rel_idx]

        # Evaluate ontological validity
        is_valid, justification = evaluate_triple_validity(
            head_entity,
            relation,
            tail_entity,
            llm_client,
            umls_client,
        )

        push!(triple_validity, is_valid)
        push!(justifications, justification)
    end

    # Calculate overall ValidityScore
    total_triples = length(triple_validity)
    valid_triples = count(v -> v == :yes, triple_validity)
    validity_score = total_triples > 0 ? valid_triples / total_triples : 0.0

    @info "ValidityScore = $(round(validity_score, digits=4)) ($valid_triples/$total_triples valid)"

    return ValidityScoreResult(
        triple_validity,
        justifications,
        validity_score,
        valid_triples,
        total_triples,
        Dict(
            "confidence_threshold" => confidence_threshold,
            "valid_count" => valid_triples,
            "maybe_count" => count(v -> v == :maybe, triple_validity),
            "no_count" => count(v -> v == :no, triple_validity),
        ),
    )
end

"""
    evaluate_triple_validity(head_entity::BiomedicalEntity, relation::BiomedicalRelation,
                            tail_entity::BiomedicalEntity,
                            llm_client::Union{HelperLLMClient, Nothing},
                            umls_client::Union{UMLSClient, Nothing})

Evaluate the ontological validity of a triple.

# Arguments
- `head_entity::BiomedicalEntity`: Head entity
- `relation::BiomedicalRelation`: Relation
- `tail_entity::BiomedicalEntity`: Tail entity
- `llm_client::Union{HelperLLMClient, Nothing}`: Optional LLM client
- `umls_client::Union{UMLSClient, Nothing}`: Optional UMLS client

# Returns
- `Tuple{Symbol, String}`: (validity, justification)
"""
function evaluate_triple_validity(
    head_entity::BiomedicalEntity,
    relation::BiomedicalRelation,
    tail_entity::BiomedicalEntity,
    llm_client::Union{HelperLLMClient,Nothing},
    umls_client::Union{UMLSClient,Nothing},
)
    # Try LLM-based evaluation first if available
    if llm_client !== nothing
        validity, justification =
            evaluate_triple_with_llm(head_entity, relation, tail_entity, llm_client)
        if validity !== :unknown
            return validity, justification
        end
    end

    # Fallback to UMLS-based evaluation if available
    if umls_client !== nothing
        validity, justification =
            evaluate_triple_with_umls(head_entity, relation, tail_entity, umls_client)
        if validity !== :unknown
            return validity, justification
        end
    end

    # Final fallback to heuristic evaluation
    return evaluate_triple_heuristic_validity(head_entity, relation, tail_entity)
end

"""
    evaluate_triple_with_llm(head_entity::BiomedicalEntity, relation::BiomedicalRelation,
                            tail_entity::BiomedicalEntity, llm_client::HelperLLMClient)

Evaluate triple validity using LLM.

# Arguments
- `head_entity::BiomedicalEntity`: Head entity
- `relation::BiomedicalRelation`: Relation
- `tail_entity::BiomedicalEntity`: Tail entity
- `llm_client::HelperLLMClient`: LLM client

# Returns
- `Tuple{Symbol, String}`: (validity, justification)
"""
function evaluate_triple_with_llm(
    head_entity::BiomedicalEntity,
    relation::BiomedicalRelation,
    tail_entity::BiomedicalEntity,
    llm_client::HelperLLMClient,
)
    prompt = """
    Evaluate if the following biomedical relationship is ontologically valid:

    Relationship: $(head_entity.text) --[$(relation.relation_type)]--> $(tail_entity.text)

    Is this relationship semantically valid in biomedical knowledge?
    Consider:
    - Are the entity types compatible with the relation?
    - Does this relationship make sense in medical context?
    - Are there any contradictions with established knowledge?

    Answer YES, MAYBE, or NO with a brief justification:
    """

    try
        response = make_llm_request(llm_client, prompt)
        if response.success
            answer = strip(lowercase(response.content))
            if startswith(answer, "yes")
                return :yes, "LLM validation: semantically valid"
            elseif startswith(answer, "no")
                return :no, "LLM validation: semantically invalid"
            else
                return :maybe, "LLM validation: uncertain validity"
            end
        end
    catch e
        @warn "LLM validity evaluation failed: $e"
    end

    return :unknown, "LLM evaluation failed"
end

"""
    evaluate_triple_with_umls(head_entity::BiomedicalEntity, relation::BiomedicalRelation,
                             tail_entity::BiomedicalEntity, umls_client::UMLSClient)

Evaluate triple validity using UMLS ontology.

# Arguments
- `head_entity::BiomedicalEntity`: Head entity
- `relation::BiomedicalRelation`: Relation
- `tail_entity::BiomedicalEntity`: Tail entity
- `umls_client::UMLSClient`: UMLS client

# Returns
- `Tuple{Symbol, String}`: (validity, justification)
"""
function evaluate_triple_with_umls(
    head_entity::BiomedicalEntity,
    relation::BiomedicalRelation,
    tail_entity::BiomedicalEntity,
    umls_client::UMLSClient,
)
    # Check if entities are linked to UMLS
    head_cui = head_entity.cui
    tail_cui = tail_entity.cui

    if head_cui === nothing || tail_cui === nothing
        return :unknown, "Entities not linked to UMLS"
    end

    # Get semantic types
    head_types = get_entity_semantic_types(umls_client, head_cui)
    tail_types = get_entity_semantic_types(umls_client, tail_cui)

    # Check ontological compatibility
    is_valid =
        check_ontological_compatibility(head_types, relation.relation_type, tail_types)

    if is_valid
        return :yes, "UMLS validation: ontologically compatible"
    else
        return :no, "UMLS validation: ontologically incompatible"
    end
end

"""
    check_ontological_compatibility(head_types::Vector{String}, relation::String,
                                   tail_types::Vector{String})

Check if entity types are compatible with the relation.

# Arguments
- `head_types::Vector{String}`: Head entity semantic types
- `relation::String`: Relation type
- `tail_types::Vector{String}`: Tail entity semantic types

# Returns
- `Bool`: True if compatible, false otherwise
"""
function check_ontological_compatibility(
    head_types::Vector{String},
    relation::String,
    tail_types::Vector{String},
)
    # Simple compatibility rules (would be more sophisticated in practice)
    relation_compatibility = Dict(
        "TREATS" => (["Disease", "Pharmacologic Substance"], ["Disease"]),
        "CAUSES" => (["Disease", "Finding"], ["Disease", "Finding"]),
        "ASSOCIATED_WITH" => (["*"], ["*"]),  # Very permissive
        "INDICATES" => (["Finding", "Laboratory Procedure"], ["Disease", "Finding"]),
        "PREVENTS" =>
            (["Pharmacologic Substance", "Therapeutic Procedure"], ["Disease"]),
    )

    compatible = get(relation_compatibility, relation, ([], []))
    head_compatible, tail_compatible = compatible

    head_ok = "*" in head_compatible || any(t in head_compatible for t in head_types)
    tail_ok = "*" in tail_compatible || any(t in tail_compatible for t in tail_types)

    return head_ok && tail_ok
end

"""
    evaluate_triple_heuristic_validity(head_entity::BiomedicalEntity, relation::BiomedicalRelation,
                                      tail_entity::BiomedicalEntity)

Simple heuristic evaluation of triple validity.

# Arguments
- `head_entity::BiomedicalEntity`: Head entity
- `relation::BiomedicalRelation`: Relation
- `tail_entity::BiomedicalEntity`: Tail entity

# Returns
- `Tuple{Symbol, String}`: (validity, justification)
"""
function evaluate_triple_heuristic_validity(
    head_entity::BiomedicalEntity,
    relation::BiomedicalRelation,
    tail_entity::BiomedicalEntity,
)
    # Simple heuristic based on entity and relation types
    head_type = lowercase(head_entity.label)
    tail_type = lowercase(tail_entity.label)
    relation_type = lowercase(relation.relation_type)

    # Basic validity rules
    if relation_type == "treats" && head_type == "drug" && tail_type == "disease"
        return :yes, "Heuristic: drug treats disease"
    elseif relation_type == "causes" && head_type == "disease" && tail_type == "disease"
        return :yes, "Heuristic: disease causes disease"
    elseif relation_type == "associated_with"  # Very permissive
        return :maybe, "Heuristic: general association"
    else
        return :no, "Heuristic: incompatible types for relation"
    end
end

"""
    calculate_validity_confidence_interval(validity_score::Float64, n::Int, confidence_level::Float64=0.95)

Calculate confidence interval for ValidityScore using Wilson score interval.

# Arguments
- `validity_score::Float64`: Observed ValidityScore
- `n::Int`: Number of triples evaluated
- `confidence_level::Float64`: Confidence level (default: 0.95)

# Returns
- `Tuple{Float64, Float64}`: (lower_bound, upper_bound)
"""
function calculate_validity_confidence_interval(
    validity_score::Float64,
    n::Int,
    confidence_level::Float64 = 0.95,
)
    if n == 0
        return (0.0, 0.0)
    end

    # Wilson score interval for binomial proportion
    z = quantile(Normal(), (1 + confidence_level) / 2)
    p = validity_score
    n = Float64(n)

    center = (p + z^2 / (2 * n)) / (1 + z^2 / n)
    margin = z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / (1 + z^2 / n)

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return (lower, upper)
end

"""
    StatisticalSignificanceResult

Result of statistical significance testing.
"""
struct StatisticalSignificanceResult
    p_value::Float64
    significant::Bool
    confidence_interval::Tuple{Float64,Float64}
    effect_size::Float64
    test_statistic::Float64
    degrees_of_freedom::Int
    test_type::String
    metadata::Dict{String,Any}
end

"""
    perform_statistical_significance_test(validity_scores::Vector{Float64};
                                       alpha::Float64=0.05,
                                       test_type::String="t_test") -> StatisticalSignificanceResult

Perform statistical significance testing on validity scores.

# Arguments
- `validity_scores::Vector{Float64}`: Vector of validity scores to test
- `alpha::Float64`: Significance level (default: 0.05)
- `test_type::String`: Type of statistical test ("t_test", "chi_square", "wilcoxon")

# Returns
- `StatisticalSignificanceResult`: Statistical test results
"""
function perform_statistical_significance_test(
    validity_scores::Vector{Float64};
    alpha::Float64 = 0.05,
    test_type::String = "t_test",
)

    if isempty(validity_scores)
        return StatisticalSignificanceResult(
            1.0,
            false,
            (0.0, 0.0),
            0.0,
            0.0,
            0,
            test_type,
            Dict(),
        )
    end

    n = length(validity_scores)
    mean_score = mean(validity_scores)
    std_score = std(validity_scores)

    if test_type == "t_test"
        # One-sample t-test against null hypothesis (score = 0.5)
        null_hypothesis = 0.5
        t_statistic = (mean_score - null_hypothesis) / (std_score / sqrt(n))
        df = n - 1
        p_value = 2 * (1 - cdf(TDist(df), abs(t_statistic)))

        # Calculate confidence interval
        t_critical = quantile(TDist(df), 1 - alpha / 2)
        margin_error = t_critical * (std_score / sqrt(n))
        ci_lower = mean_score - margin_error
        ci_upper = mean_score + margin_error

        # Effect size (Cohen's d)
        effect_size = (mean_score - null_hypothesis) / std_score

    elseif test_type == "chi_square"
        # Chi-square test for goodness of fit
        expected_valid = n * 0.5  # Expected 50% valid
        observed_valid = sum(validity_scores .> 0.5)
        observed_invalid = n - observed_valid

        chi2_statistic =
            ((observed_valid - expected_valid)^2 / expected_valid) +
            ((observed_invalid - expected_valid)^2 / expected_valid)
        df = 1
        p_value = 1 - chi2cdf(chi2_statistic, df)

        # Confidence interval for proportion
        p_hat = observed_valid / n
        z_critical = quantile(Normal(), 1 - alpha / 2)
        margin_error = z_critical * sqrt(p_hat * (1 - p_hat) / n)
        ci_lower = p_hat - margin_error
        ci_upper = p_hat + margin_error

        effect_size = (p_hat - 0.5) / sqrt(0.5 * 0.5)
        t_statistic = chi2_statistic

    else
        # Default to t-test
        null_hypothesis = 0.5
        t_statistic = (mean_score - null_hypothesis) / (std_score / sqrt(n))
        df = n - 1
        p_value = 2 * (1 - cdf(TDist(df), abs(t_statistic)))

        t_critical = quantile(TDist(df), 1 - alpha / 2)
        margin_error = t_critical * (std_score / sqrt(n))
        ci_lower = mean_score - margin_error
        ci_upper = mean_score + margin_error
        effect_size = (mean_score - null_hypothesis) / std_score
    end

    significant = p_value < alpha

    return StatisticalSignificanceResult(
        p_value,
        significant,
        (ci_lower, ci_upper),
        effect_size,
        t_statistic,
        df,
        test_type,
        Dict("sample_size" => n, "mean_score" => mean_score, "std_score" => std_score),
    )
end

"""
    compare_validity_scores(kg1_scores::Vector{Float64}, kg2_scores::Vector{Float64};
                          alpha::Float64=0.05) -> StatisticalSignificanceResult

Compare validity scores between two knowledge graphs using two-sample t-test.

# Arguments
- `kg1_scores::Vector{Float64}`: Validity scores from first knowledge graph
- `kg2_scores::Vector{Float64}`: Validity scores from second knowledge graph
- `alpha::Float64`: Significance level (default: 0.05)

# Returns
- `StatisticalSignificanceResult`: Comparison test results
"""
function compare_validity_scores(
    kg1_scores::Vector{Float64},
    kg2_scores::Vector{Float64};
    alpha::Float64 = 0.05,
)

    if isempty(kg1_scores) || isempty(kg2_scores)
        return StatisticalSignificanceResult(
            1.0,
            false,
            (0.0, 0.0),
            0.0,
            0.0,
            0,
            "two_sample_t_test",
            Dict(),
        )
    end

    n1, n2 = length(kg1_scores), length(kg2_scores)
    mean1, mean2 = mean(kg1_scores), mean(kg2_scores)
    std1, std2 = std(kg1_scores), std(kg2_scores)

    # Two-sample t-test
    pooled_std = sqrt(((n1 - 1) * std1^2 + (n2 - 1) * std2^2) / (n1 + n2 - 2))
    t_statistic = (mean1 - mean2) / (pooled_std * sqrt(1 / n1 + 1 / n2))
    df = n1 + n2 - 2

    p_value = 2 * (1 - cdf(TDist(df), abs(t_statistic)))

    # Confidence interval for difference
    t_critical = quantile(TDist(df), 1 - alpha / 2)
    margin_error = t_critical * pooled_std * sqrt(1 / n1 + 1 / n2)
    diff = mean1 - mean2
    ci_lower = diff - margin_error
    ci_upper = diff + margin_error

    # Effect size (Cohen's d)
    effect_size = (mean1 - mean2) / pooled_std

    significant = p_value < alpha

    return StatisticalSignificanceResult(
        p_value,
        significant,
        (ci_lower, ci_upper),
        effect_size,
        t_statistic,
        df,
        "two_sample_t_test",
        Dict("kg1_size" => n1, "kg2_size" => n2, "mean1" => mean1, "mean2" => mean2),
    )
end

"""
    evaluate_validity_with_statistics(kg::KnowledgeGraph;
                                   llm_client::Union{HelperLLMClient, Nothing}=nothing,
                                   umls_client::Union{UMLSClient, Nothing}=nothing,
                                   confidence_threshold::Float64=0.5,
                                   alpha::Float64=0.05) -> Tuple{ValidityScoreResult, StatisticalSignificanceResult}

Evaluate validity with statistical significance testing.

# Arguments
- `kg::KnowledgeGraph`: Knowledge graph to evaluate
- `llm_client::Union{HelperLLMClient, Nothing}`: Optional LLM client for validation
- `umls_client::Union{UMLSClient, Nothing}`: Optional UMLS client for ontology checking
- `confidence_threshold::Float64`: Minimum confidence threshold for triples
- `alpha::Float64`: Significance level for statistical testing

# Returns
- `Tuple{ValidityScoreResult, StatisticalSignificanceResult}`: Validity and statistical results
"""
function evaluate_validity_with_statistics(
    kg::KnowledgeGraph;
    llm_client::Union{HelperLLMClient,Nothing} = nothing,
    umls_client::Union{UMLSClient,Nothing} = nothing,
    confidence_threshold::Float64 = 0.5,
    alpha::Float64 = 0.05,
)

    # Get basic validity results
    validity_result = evaluate_validity(
        kg,
        llm_client = llm_client,
        umls_client = umls_client,
        confidence_threshold = confidence_threshold,
    )

    # Convert validity symbols to numeric scores for statistical testing
    validity_scores = Float64[]
    for validity in validity_result.triple_validity
        if validity == :yes
            push!(validity_scores, 1.0)
        elseif validity == :no
            push!(validity_scores, 0.0)
        else  # :maybe
            push!(validity_scores, 0.5)
        end
    end

    # Perform statistical significance testing
    stats_result = perform_statistical_significance_test(validity_scores, alpha = alpha)

    return validity_result, stats_result
end

# Export functions
export evaluate_validity,
    evaluate_triple_validity,
    evaluate_triple_heuristic_validity,
    StatisticalSignificanceResult,
    perform_statistical_significance_test,
    compare_validity_scores,
    evaluate_validity_with_statistics
