"""
Relation Extraction Demo

This example demonstrates the second stage of the GraphMERT knowledge graph
extraction pipeline: matching entities to relations.

Key concepts demonstrated:
1. Entity pair analysis for potential relations
2. Relation type determination based on text context
3. Confidence scoring for relation extraction
4. Relation filtering based on thresholds
"""

using GraphMERT

println("=== Relation Extraction Demo ===")

# 1. Sample biomedical text with entities
text = """
Diabetes mellitus is a chronic metabolic disorder characterized by
elevated blood glucose levels. Metformin is commonly used to treat
type 2 diabetes. Insulin resistance is a key feature of type 2 diabetes.
Cardiovascular disease is a major complication of diabetes.
"""

println("Sample text:")
println(text)
println()

# 2. Extract entities first
println("2. Extracting entities...")
entities = discover_head_entities(text)
println("Found $(length(entities)) entities:")

for (i, entity) in enumerate(entities)
  println("  $i. $(entity.text) ($(entity.entity_type), conf: $(round(entity.confidence, digits=3)))")
end
println()

# 3. Find relations between entities
println("3. Finding relations between entities...")
relations = match_relations_for_entities(entities, text)
println("Found $(length(relations)) potential relations:")

for (head_idx, tail_idx, relation_type, confidence) in relations
  head_entity = entities[head_idx]
  tail_entity = entities[tail_idx]
  println("  $(head_entity.text) --[$(relation_type)]--> $(tail_entity.text) (conf: $(round(confidence, digits=3)))")
end
println()

# 4. Analyze relation types
println("4. Relation type analysis:")
relation_counts = Dict{String,Int}()

for (_, _, relation_type, _) in relations
  relation_counts[relation_type] = get(relation_counts, relation_type, 0) + 1
end

for (relation_type, count) in sort(collect(relation_counts), by=x -> x[2], rev=true)
  println("  $relation_type: $count relations")
end
println()

# 5. Relation confidence analysis
println("5. Relation confidence analysis:")
high_confidence = filter(r -> r[4] > 0.8, relations)
medium_confidence = filter(r -> 0.6 ≤ r[4] ≤ 0.8, relations)
low_confidence = filter(r -> r[4] < 0.6, relations)

println("  High confidence (>0.8): $(length(high_confidence)) relations")
println("  Medium confidence (0.6-0.8): $(length(medium_confidence)) relations")
println("  Low confidence (<0.6): $(length(low_confidence)) relations")

if !isempty(high_confidence)
  println("  High confidence relations:")
  for (head_idx, tail_idx, relation_type, confidence) in high_confidence
    head_entity = entities[head_idx]
    tail_entity = entities[tail_idx]
    println("    $(head_entity.text) --[$(relation_type)]--> $(tail_entity.text)")
  end
end

# 6. Integration with knowledge graph construction
println("\n6. Integration with knowledge graph construction:")
println("These relations form the foundation for:")
println("• Knowledge graph triple formation")
println("• Semantic relationship modeling")
println("• Graph structure creation")
println("• Biomedical knowledge representation")

println("\n✅ Relation extraction demo complete!")
println("\nNext steps:")
println("• Tail entity prediction using GraphMERT model")
println("• Triple formation and filtering")
println("• Complete knowledge graph construction")
println("• Performance evaluation against ground truth")