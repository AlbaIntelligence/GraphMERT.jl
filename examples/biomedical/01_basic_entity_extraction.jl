"""
Basic Entity Extraction Demo

This example demonstrates the first stage of the GraphMERT knowledge graph
extraction pipeline: discovering biomedical entities from text.

Key concepts demonstrated:
1. Entity extraction using simple heuristics
2. Confidence scoring based on term characteristics
3. Entity position tracking in source text
4. Basic biomedical term identification
"""

using GraphMERT

println("=== Basic Entity Extraction Demo ===")

# 1. Sample biomedical text
text = """
Diabetes mellitus is a chronic metabolic disorder characterized by
elevated blood glucose levels. Metformin is commonly used to treat
type 2 diabetes. Insulin resistance is a key feature of type 2 diabetes.
Cardiovascular disease is a major complication of diabetes.
"""

println("Sample text:")
println(text)
println()

# 2. Extract biomedical entities
println("2. Extracting biomedical entities...")
entities = discover_head_entities(text)
println("Found $(length(entities)) biomedical entities:")

for entity in entities
  println("  Entity: $(entity.text)")
  println("    Type: $(entity.entity_type)")
  println("    CUI: $(entity.cui)")
  println("    Confidence: $(round(entity.confidence, digits=3))")
  println("    Position: $(entity.position.start_char)-$(entity.position.end_char)")
  println()
end

# 3. Demonstrate confidence scoring
println("3. Entity confidence analysis:")
for entity in entities
  confidence_factors = analyze_confidence_factors(entity.text, text)
  println("  '$(entity.text)' confidence: $(round(entity.confidence, digits=3))")
  println("    Factors: $confidence_factors")
end

# 4. Show entity statistics
println("4. Entity extraction statistics:")
total_chars = length(text)
entity_chars = sum(length(e.text) for e in entities)
coverage = entity_chars / total_chars * 100

println("  Total characters: $total_chars")
println("  Entity characters: $entity_chars")
println("  Coverage: $(round(coverage, digits=1))%")
println("  Average confidence: $(round(mean(e.confidence for e in entities), digits=3))")

# 5. Integration with knowledge graph construction
println("\n5. Integration with knowledge graph construction:")
println("These entities form the foundation for:")
println("• Relation extraction between entity pairs")
println("• Knowledge graph triple formation")
println("• Semantic relationship modeling")
println("• Biomedical ontology alignment")

println("\n✅ Basic entity extraction demo complete!")
println("\nNext steps:")
println("• Relation matching between entity pairs")
println("• Tail entity prediction using GraphMERT model")
println("• Triple formation and filtering")
println("• Complete knowledge graph construction")

function analyze_confidence_factors(entity_text::String, full_text::String)
  factors = String[]

  # Length factor
  if length(entity_text) > 5
    push!(factors, "long_term")
  end

  # Capitalization factor
  if isuppercase(entity_text[1])
    push!(factors, "capitalized")
  end

  # Frequency factor
  occurrences = length(findall(entity_text, full_text))
  if occurrences > 1
    push!(factors, "frequent_$(occurrences)x")
  end

  return join(factors, ", ")
end