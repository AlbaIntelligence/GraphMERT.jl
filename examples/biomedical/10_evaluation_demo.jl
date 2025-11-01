"""
Evaluation Metrics Demo

This example demonstrates the GraphMERT evaluation metrics for assessing
knowledge graph quality, including FActScore*, ValidityScore, and GraphRAG.

Key concepts demonstrated:
1. FActScore* for factuality evaluation
2. ValidityScore for ontological validity
3. GraphRAG for graph-level utility assessment
4. Statistical significance testing
5. Integration with paper results validation
"""

using GraphMERT

println("=== Evaluation Metrics Demo ===")

# Load and register biomedical domain
println("\n📋 Loading biomedical domain...")
include("../../GraphMERT/src/domains/biomedical.jl")
bio_domain = load_biomedical_domain()
register_domain!("biomedical", bio_domain)
set_default_domain("biomedical")
println("   ✅ Biomedical domain loaded and registered")
println()

# 1. Create sample knowledge graph
println("\n1. Creating sample knowledge graph...")
text = """
Diabetes mellitus is a chronic metabolic disorder characterized by
elevated blood glucose levels. Metformin is commonly used to treat
type 2 diabetes. Insulin resistance is a key feature of type 2 diabetes.
Cardiovascular disease is a major complication of diabetes.
"""

# Extract knowledge graph (simplified demo)
try
    options = ProcessingOptions(domain="biomedical")
    kg = extract_knowledge_graph(text, model; options=options)
    println("Sample KG: $(length(kg.entities)) entities, $(length(kg.relations)) relations")
catch e
    println("⚠️  Model not available - using placeholder KG")
    println("   In production, load a model with: model = GraphMERT.load_model(\"path/to/model\")")
    # Create placeholder KG for demo
    kg = nothing
end

# 2. Evaluate FActScore*
println("\n2. Evaluating FActScore*...")
try
    if kg === nothing
        error("Knowledge graph not available")
    end
    factscore_result = evaluate_factscore(kg, text; domain_name="biomedical", include_domain_metrics=true)
    println("FActScore*: $(round(factscore_result.overall_score, digits=4))")
    println("Supported triples: $(factscore_result.num_supported)/$(factscore_result.num_total)")

    # Calculate confidence interval
    if factscore_result.num_total > 0
        ci_lower, ci_upper = calculate_factscore_confidence_interval(
            factscore_result.overall_score, factscore_result.num_total
        )
        println("95% CI: [$(round(ci_lower, digits=4)), $(round(ci_upper, digits=4))]")
    end

catch e
    println("FActScore* evaluation: $(e)")
    println("This is expected in demo mode without full LLM integration")
end

# 3. Evaluate ValidityScore
println("\n3. Evaluating ValidityScore...")
try
    validity_result = evaluate_validity(kg)
    println("ValidityScore: $(round(validity_result.overall_score, digits=4))")
    println("Valid triples: $(validity_result.num_yes)/$(validity_result.num_total)")

catch e
    println("ValidityScore evaluation: $(e)")
    println("This is expected in demo mode without full UMLS integration")
end

# 4. Evaluate GraphRAG
println("\n4. Evaluating GraphRAG...")
try
    if kg === nothing
        error("Knowledge graph not available")
    end
    # Sample questions for GraphRAG evaluation
    questions = [
        "What treats diabetes?",
        "What are symptoms of diabetes?",
        "What causes cardiovascular disease?"
    ]

    ground_truth = [
        "Metformin",
        "Elevated blood glucose",
        "Diabetes"
    ]

    graphrag_result = evaluate_graphrag(kg, questions, ground_truth)
    println("GraphRAG Accuracy: $(round(graphrag_result.accuracy, digits=4))")
    println("Questions answered correctly: $(sum(graphrag_result.correct_flags))")

catch e
    println("GraphRAG evaluation: $(e)")
    println("This is expected in demo mode without full model integration")
end

# 5. Paper comparison
println("\n5. Paper comparison:")
println("  Paper FActScore*: 69.8% (target: 66.3-73.3%)")
println("  Paper ValidityScore: 68.8% (target: 65.4-72.2%)")
println("  Current demo results:")
println("    FActScore*: $(get(factscore_result, :overall_score, 0.0))")
println("    ValidityScore: $(get(validity_result, :overall_score, 0.0))")
println("    GraphRAG: $(get(graphrag_result, :accuracy, 0.0))")
println("  Note: Full validation requires complete evaluation pipeline")

# 6. Statistical analysis
println("\n6. Statistical analysis:")
try
    if factscore_result !== nothing && factscore_result.num_total > 0
    # Compare with paper target
    paper_target = 0.698
    observed_score = factscore_result.overall_score
    difference = abs(observed_score - paper_target)
    within_tolerance = difference <= 0.05  # 5% tolerance as per requirements

    println("  FActScore* analysis:")
    println("    Observed: $(round(observed_score, digits=4))")
    println("    Target: $(round(paper_target, digits=4))")
    println("    Difference: $(round(difference, digits=4))")
    println("    Within 5% tolerance: $within_tolerance")

    if within_tolerance
        println("    ✅ Meets paper requirements (REQ-011)")
    else
        println("    ❌ Does not meet paper requirements")
    end
end

# 7. Evaluation summary
println("\n7. Evaluation summary:")
println("  Evaluation metrics provide comprehensive KG quality assessment:")
println("  • FActScore*: Factuality evaluation with context checking")
println("  • ValidityScore: Ontological validity against UMLS")
println("  • GraphRAG: Graph-level utility for question answering")
println("  • Statistical significance testing with confidence intervals")

println("\n✅ Evaluation metrics demo complete!")

println("\nNext steps for complete implementation:")
println("• Implement LLM-based evaluation for FActScore*")
println("• Add UMLS-based validity checking")
println("• Create GraphRAG question answering system")
println("• Validate against full diabetes dataset (350k abstracts)")
println("• Achieve paper targets: FActScore* 69.8%, ValidityScore 68.8%")
