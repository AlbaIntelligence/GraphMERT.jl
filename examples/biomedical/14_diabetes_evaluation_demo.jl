"""
Diabetes Knowledge Graph Evaluation Demo

This example demonstrates comprehensive evaluation of diabetes knowledge graphs using:
1. ICD-Bench benchmark integration
2. MedMCQA benchmark evaluation
3. Statistical significance testing (p < 0.05)
4. FActScore* and ValidityScore metrics
5. GraphRAG evaluation methodology

The demo shows how to evaluate biomedical knowledge graphs against
standard benchmarks and reproduce paper results.
"""

using GraphMERT
using GraphMERT: run_diabetes_benchmark_evaluation, evaluate_factscore,
  evaluate_validity_with_statistics, evaluate_graphrag,
  ICDBenchResult, MedMCQAResult, StatisticalSignificanceResult

println("=== Diabetes Knowledge Graph Evaluation Demo ===")

# 1. Create sample diabetes knowledge graph
println("\n1. Creating sample diabetes knowledge graph...")

# Create sample entities
entities = [
  GraphMERT.Entity("entity_1", "diabetes mellitus", "diabetes mellitus", "DISEASE", "biomedical",
    Dict("cui" => "C0011849", "icd_code" => "E11", "semantic_type" => "Disease"),
    GraphMERT.TextPosition(1, 20, 1, 1),
    0.95, ""),
  GraphMERT.Entity("entity_2", "metformin", "metformin", "DRUG", "biomedical",
    Dict("cui" => "C0025595", "drug_class" => "biguanide", "semantic_type" => "Drug"),
    GraphMERT.TextPosition(25, 35, 1, 1),
    0.92, ""),
  GraphMERT.Entity("entity_3", "insulin", "insulin", "DRUG", "biomedical",
    Dict("cui" => "C0021641", "drug_class" => "hormone", "semantic_type" => "Drug"),
    GraphMERT.TextPosition(40, 47, 1, 1),
    0.98, ""),
  GraphMERT.Entity("entity_4", "glucose", "glucose", "CHEMICAL", "biomedical",
    Dict("cui" => "C0017725", "chemical_type" => "sugar", "semantic_type" => "Chemical"),
    GraphMERT.TextPosition(50, 57, 1, 1),
    0.89, "")
]

# Create sample relations
relations = [
  GraphMERT.Relation("entity_1", "entity_2", "TREATS", "biomedical", 0.88,
    "", "", Dict("evidence_level" => "high", "source" => "clinical_trial")),
  GraphMERT.Relation("entity_1", "entity_3", "TREATS", "biomedical", 0.94,
    "", "", Dict("evidence_level" => "high", "source" => "clinical_guideline")),
  GraphMERT.Relation("entity_1", "entity_4", "AFFECTS", "biomedical", 0.91,
    "", "", Dict("evidence_level" => "high", "source" => "pathophysiology")),
  GraphMERT.Relation("entity_2", "entity_4", "REDUCES", "biomedical", 0.87,
    "", "", Dict("evidence_level" => "high", "source" => "mechanism"))
]

# Create knowledge graph
kg = GraphMERT.KnowledgeGraph(entities, relations,
  Dict("domain" => "diabetes", "source" => "demo", "version" => "1.0"), now())

println("✓ Created knowledge graph with $(length(kg.entities)) entities and $(length(kg.relations)) relations")

# 2. Run ICD-Bench evaluation
println("\n2. Running ICD-Bench evaluation...")

# Create mock ICD-Bench data
icd_data = DataFrame(
  icd_code=["E11", "E10", "E12", "E13", "E14"],
  description=["Type 2 diabetes", "Type 1 diabetes", "Other diabetes", "Unspecified diabetes", "Gestational diabetes"],
  severity=["moderate", "severe", "mild", "moderate", "mild"]
)

# Save mock data
mkdir("data", exist_ok=true)
CSV.write("data/icd_bench.csv", icd_data)

# Run ICD-Bench evaluation
icd_results = GraphMERT.evaluate_icd_benchmark(kg, icd_data)
println("✓ ICD-Bench Results:")
println("  • Accuracy: $(round(icd_results.accuracy, digits=3))")
println("  • Precision: $(round(icd_results.precision, digits=3))")
println("  • Recall: $(round(icd_results.recall, digits=3))")
println("  • F1-Score: $(round(icd_results.f1_score, digits=3))")
println("  • Coverage: $(round(icd_results.coverage, digits=3))")

# 3. Run MedMCQA evaluation
println("\n3. Running MedMCQA evaluation...")

# Create mock MedMCQA data
medmcqa_data = DataFrame(
  question=[
    "What is the first-line treatment for type 2 diabetes?",
    "Which drug class does metformin belong to?",
    "What is the primary mechanism of insulin?",
    "How does glucose affect diabetes?"
  ],
  answer=[
    "metformin",
    "biguanide",
    "glucose regulation",
    "elevated levels cause complications"
  ],
  question_type=["treatment", "classification", "mechanism", "pathophysiology"]
)

# Save mock data
CSV.write("data/medmcqa.csv", medmcqa_data)

# Run MedMCQA evaluation
medmcqa_results = GraphMERT.evaluate_medmcqa(kg, medmcqa_data)
println("✓ MedMCQA Results:")
println("  • Overall Accuracy: $(round(medmcqa_results.accuracy, digits=3))")
println("  • Reasoning Accuracy: $(round(medmcqa_results.reasoning_accuracy, digits=3))")
println("  • Knowledge Accuracy: $(round(medmcqa_results.knowledge_accuracy, digits=3))")
println("  • Correct Answers: $(medmcqa_results.correct_answers)/$(medmcqa_results.total_questions)")

# 4. Run comprehensive benchmark evaluation
println("\n4. Running comprehensive benchmark evaluation...")

benchmark_results = GraphMERT.run_diabetes_benchmark_evaluation(kg,
  icd_data_path="data/icd_bench.csv",
  medmcqa_data_path="data/medmcqa.csv")

println("✓ Comprehensive Benchmark Results:")
println("  • Overall Accuracy: $(round(benchmark_results["overall_accuracy"], digits=3))")
println("  • ICD-Bench F1: $(round(benchmark_results["icd_bench"].f1_score, digits=3))")
println("  • MedMCQA Accuracy: $(round(benchmark_results["medmcqa"].accuracy, digits=3))")

# 5. Run FActScore* evaluation
println("\n5. Running FActScore* evaluation...")

factscore_results = GraphMERT.evaluate_factscore(kg, confidence_threshold=0.5)
println("✓ FActScore* Results:")
println("  • FActScore: $(round(factscore_results.factscore, digits=3))")
println("  • Supported Triples: $(factscore_results.supported_triples)/$(factscore_results.total_triples)")
println("  • Triple Scores: $(length(factscore_results.triple_scores)) evaluated")

# 6. Run ValidityScore with statistical testing
println("\n6. Running ValidityScore with statistical significance testing...")

validity_result, stats_result = GraphMERT.evaluate_validity_with_statistics(kg, alpha=0.05; domain_name="biomedical", include_domain_metrics=true)
println("✓ ValidityScore Results:")
println("  • Validity Score: $(round(validity_result.validity_score, digits=3))")
println("  • Valid Triples: $(validity_result.valid_triples)/$(validity_result.total_triples)")

println("✓ Statistical Significance Results:")
println("  • P-value: $(round(stats_result.p_value, digits=4))")
println("  • Significant: $(stats_result.significant ? "Yes" : "No")")
println("  • Effect Size: $(round(stats_result.effect_size, digits=3))")
println("  • Confidence Interval: ($(round(stats_result.confidence_interval[1], digits=3)), $(round(stats_result.confidence_interval[2], digits=3)))")

# 7. Run GraphRAG evaluation
println("\n7. Running GraphRAG evaluation...")

# Create sample questions for GraphRAG
questions = [
  "What drugs are used to treat diabetes?",
  "How does metformin work?",
  "What is the relationship between glucose and diabetes?"
]

graphrag_results = GraphMERT.evaluate_graphrag(kg, questions)
println("✓ GraphRAG Results:")
println("  • Questions Answered: $(length(graphrag_results))")
for (i, result) in enumerate(graphrag_results)
  println("  • Question $i: $(round(result["relevance_score"], digits=3)) relevance")
end

# 8. Compare with paper results
println("\n8. Comparing with paper results...")

paper_factscore = 0.698  # 69.8%
paper_validity = 0.688   # 68.8%

our_factscore = factscore_results.factscore
our_validity = validity_result.validity_score

factscore_diff = abs(our_factscore - paper_factscore)
validity_diff = abs(our_validity - paper_validity)

println("✓ Paper Comparison:")
println("  • Paper FActScore: $(round(paper_factscore, digits=3))")
println("  • Our FActScore: $(round(our_factscore, digits=3))")
println("  • Difference: $(round(factscore_diff, digits=3))")

println("  • Paper ValidityScore: $(round(paper_validity, digits=3))")
println("  • Our ValidityScore: $(round(our_validity, digits=3))")
println("  • Difference: $(round(validity_diff, digits=3))")

# 9. Summary
println("\n✅ Diabetes Knowledge Graph Evaluation Complete!")
println("\nKey Achievements:")
println("• ICD-Bench integration: $(round(icd_results.f1_score, digits=3)) F1-score")
println("• MedMCQA evaluation: $(round(medmcqa_results.accuracy, digits=3)) accuracy")
println("• FActScore* metric: $(round(factscore_results.factscore, digits=3))")
println("• ValidityScore metric: $(round(validity_result.validity_score, digits=3))")
println("• Statistical significance: $(stats_result.significant ? "Significant" : "Not significant") (p=$(round(stats_result.p_value, digits=4)))")
println("• GraphRAG evaluation: $(length(graphrag_results)) questions answered")
println("• Paper reproduction: $(round((1 - factscore_diff) * 100, digits=1))% FActScore match, $(round((1 - validity_diff) * 100, digits=1))% ValidityScore match")

println("\nNext steps for full evaluation:")
println("• Load real ICD-Bench and MedMCQA datasets")
println("• Train on full diabetes corpus (350k abstracts)")
println("• Validate against paper's exact methodology")
println("• Run statistical significance tests on full results")

# Clean up
rm("data/icd_bench.csv", force=true)
rm("data/medmcqa.csv", force=true)
rmdir("data", force=true)
println("\n✓ Cleanup completed")
