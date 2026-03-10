"""
Quality metrics computation helpers for Wikipedia knowledge graph testing.

Provides functions to calculate precision, recall, F1, and other quality metrics
for evaluating extracted knowledge graphs against reference facts.
"""

using Statistics

"""
Quality metrics for knowledge graph evaluation
"""
struct QualityMetrics
    entity_precision::Float64
    entity_recall::Float64
    entity_f1::Float64
    relation_precision::Float64
    relation_recall::Float64
    relation_f1::Float64
    fact_capture_rate::Float64
    confidence_auc::Float64
end

"""
Calculate entity precision

Compares extracted entities against expected entities.
"""
function calculate_entity_precision(extracted::Vector{Tuple{String, String, Float64}}, 
                                    expected::Vector{Tuple{String, String, Float64}})
    if isempty(extracted)
        return isempty(expected) ? 1.0 : 0.0
    end
    
    if isempty(expected)
        return 0.0
    end
    
    matches = 0
    for ext in extracted
        ext_name, ext_type, _ = ext
        for exp in expected
            exp_name, exp_type, _ = exp
            # Match on both name and type
            if ext_name == exp_name && ext_type == exp_type
                matches += 1
                break
            end
        end
    end
    
    return matches / length(extracted)
end

"""
Calculate entity recall
"""
function calculate_entity_recall(extracted::Vector{Tuple{String, String, Float64}}, 
                                 expected::Vector{Tuple{String, String, Float64}})
    if isempty(extracted)
        return isempty(expected) ? 1.0 : 0.0
    end
    
    if isempty(expected)
        return 1.0  # Nothing to recall
    end
    
    matches = 0
    for exp in expected
        exp_name, exp_type, _ = exp
        for ext in extracted
            ext_name, ext_type, _ = ext
            if ext_name == exp_name && ext_type == exp_type
                matches += 1
                break
            end
        end
    end
    
    return matches / length(expected)
end

"""
Calculate relation precision
"""
function calculate_relation_precision(extracted::Vector{Tuple{String, String, String, Float64}},
                                       expected::Vector{Tuple{String, String, String, Float64}})
    if isempty(extracted)
        return isempty(expected) ? 1.0 : 0.0
    end
    
    if isempty(expected)
        return 0.0
    end
    
    matches = 0
    for ext in extracted
        ext_head, ext_rel, ext_tail, _ = ext
        for exp in expected
            exp_head, exp_rel, exp_tail, _ = exp
            # Match on head, relation, and tail
            if ext_head == exp_head && ext_rel == exp_rel && ext_tail == exp_tail
                matches += 1
                break
            end
        end
    end
    
    return matches / length(extracted)
end

"""
Calculate relation recall
"""
function calculate_relation_recall(extracted::Vector{Tuple{String, String, String, Float64}},
                                   expected::Vector{Tuple{String, String, String, Float64}})
    if isempty(extracted)
        return isempty(expected) ? 1.0 : 0.0
    end
    
    if isempty(expected)
        return 1.0
    end
    
    matches = 0
    for exp in expected
        exp_head, exp_rel, exp_tail, _ = exp
        for ext in extracted
            ext_head, ext_rel, ext_tail, _ = ext
            if ext_head == exp_head && ext_rel == exp_rel && ext_tail == exp_tail
                matches += 1
                break
            end
        end
    end
    
    return matches / length(expected)
end

"""
Calculate fact capture rate

Measures what percentage of known reference facts are captured in the extracted graph.
"""
function calculate_fact_capture_rate(extracted_relations::Vector{Tuple{String, String, String, Float64}},
                                      reference_facts::Vector{Tuple{String, String, String, Bool}})
    if isempty(reference_facts)
        return 1.0
    end
    
    captured = 0
    for ref in reference_facts
        ref_head, ref_pred, ref_obj = ref[1], ref[2], ref[3]
        for ext in extracted_relations
            ext_head, ext_rel, ext_tail = ext[1], ext[2], ext[3]
            if ext_head == ref_head && ext_rel == ref_pred && ext_tail == ref_obj
                captured += 1
                break
            end
        end
    end
    
    return captured / length(reference_facts)
end

"""
Calculate confidence AUC

Measures how well confidence scores correlate with correctness.
This is a simplified version - in practice, you'd use ROC analysis.
"""
function calculate_confidence_auc(extracted::Vector{Tuple{String, String, Float64}},
                                  expected::Vector{Tuple{String, String, Float64}})
    if isempty(extracted) || isempty(expected)
        return 0.5  # Neutral
    end
    
    # Simple approximation: compare average confidence of correct vs incorrect
    correct_confidences = Float64[]
    incorrect_confidences = Float64[]
    
    expected_set = Set([(exp[1], exp[2]) for exp in expected])
    
    for ext in extracted
        if (ext[1], ext[2]) in expected_set
            push!(correct_confidences, ext[3])
        else
            push!(incorrect_confidences, ext[3])
        end
    end
    
    if isempty(correct_confidences) || isempty(incorrect_confidences)
        return 0.5
    end
    
    # If correct items have higher average confidence, AUC > 0.5
    avg_correct = mean(correct_confidences)
    avg_incorrect = mean(incorrect_confidences)
    
    # Simple heuristic: map to 0-1 range
    auc = avg_correct / (avg_correct + avg_incorrect + eps(Float64))
    return clamp(auc, 0.0, 1.0)
end

"""
Calculate all quality metrics

Returns a QualityMetrics struct with all computed metrics.
"""
function calculate_quality_metrics(extracted_entities::Vector{Tuple{String, String, Float64}},
                                   expected_entities::Vector{Tuple{String, String, Float64}},
                                   extracted_relations::Vector{Tuple{String, String, String, Float64}},
                                   expected_relations::Vector{Tuple{String, String, String, Float64}},
                                   reference_facts::Vector{Tuple{String, String, String, Bool}})
    
    entity_precision = calculate_entity_precision(extracted_entities, expected_entities)
    entity_recall = calculate_entity_recall(extracted_entities, expected_entities)
    entity_f1 = entity_precision + entity_recall > 0 ? 
                2 * entity_precision * entity_recall / (entity_precision + entity_recall) : 0.0
    
    relation_precision = calculate_relation_precision(extracted_relations, expected_relations)
    relation_recall = calculate_relation_recall(extracted_relations, expected_relations)
    relation_f1 = relation_precision + relation_recall > 0 ?
                  2 * relation_precision * relation_recall / (relation_precision + relation_recall) : 0.0
    
    fact_capture = calculate_fact_capture_rate(extracted_relations, reference_facts)
    confidence_auc = calculate_confidence_auc(extracted_entities, expected_entities)
    
    return QualityMetrics(
        entity_precision,
        entity_recall,
        entity_f1,
        relation_precision,
        relation_recall,
        relation_f1,
        fact_capture,
        confidence_auc
    )
end

"""
Print quality metrics in a readable format
"""
function print_metrics(metrics::QualityMetrics)
    println("=== Quality Metrics ===")
    println("Entity Precision:  $(round(metrics.entity_precision * 100, digits=1))%")
    println("Entity Recall:    $(round(metrics.entity_recall * 100, digits=1))%")
    println("Entity F1:        $(round(metrics.entity_f1 * 100, digits=1))%")
    println()
    println("Relation Precision:  $(round(metrics.relation_precision * 100, digits=1))%")
    println("Relation Recall:    $(round(metrics.relation_recall * 100, digits=1))%")
    println("Relation F1:        $(round(metrics.relation_f1 * 100, digits=1))%")
    println()
    println("Fact Capture Rate: $(round(metrics.fact_capture_rate * 100, digits=1))%")
    println("Confidence AUC:    $(round(metrics.confidence_auc, digits=3))")
end
