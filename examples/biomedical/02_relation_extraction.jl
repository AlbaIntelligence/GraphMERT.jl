"""
Relation Extraction Demo - Domain System Version

This example demonstrates the second stage of the GraphMERT knowledge graph
extraction pipeline: matching entities to relations using the domain system.

Key concepts demonstrated:
1. Loading and registering the biomedical domain
2. Entity extraction using domain providers
3. Relation extraction between entities using domain providers
4. Relation type analysis and confidence scoring
5. Relation filtering based on thresholds
"""

using GraphMERT
using Statistics: mean

println("=== Relation Extraction Demo (Domain System) ===\n")

# Load and register the biomedical domain
println("1. Loading biomedical domain...")
include("../../GraphMERT/src/domains/biomedical.jl")

bio_domain = load_biomedical_domain()
register_domain!("biomedical", bio_domain)
set_default_domain("biomedical")

println("   ✅ Biomedical domain loaded and registered\n")

# Sample biomedical text
text = """
Diabetes mellitus is a chronic metabolic disorder characterized by
elevated blood glucose levels. Metformin is commonly used to treat
type 2 diabetes. Insulin resistance is a key feature of type 2 diabetes.
Cardiovascular disease is a major complication of diabetes.
"""

println("2. Sample text:")
println(text)
println()

# Extract entities using domain system
println("3. Extracting entities using domain system...")
options = ProcessingOptions(domain="biomedical")
entities = extract_entities(bio_domain, text, options)

println("   Found $(length(entities)) entities:\n")

for (i, entity) in enumerate(entities)
  println("   $i. $(entity.text) ($(entity.entity_type), conf: $(round(entity.confidence, digits=3)))")
end
println()

# Extract relations using domain system
println("4. Extracting relations using domain system...")
relations = extract_relations(bio_domain, entities, text, options)

println("   Found $(length(relations)) potential relations:\n")

for relation in relations
  # Find head and tail entities by ID
  head_entity = nothing
  tail_entity = nothing
  
  for entity in entities
    if entity.id == relation.head
      head_entity = entity
    elseif entity.id == relation.tail
      tail_entity = entity
    end
  end
  
  if head_entity !== nothing && tail_entity !== nothing
    println("   $(head_entity.text) --[$(relation.relation_type)]--> $(tail_entity.text) (conf: $(round(relation.confidence, digits=3)))")
  end
end
println()

# Analyze relation types
println("5. Relation type analysis:")
relation_counts = Dict{String,Int}()

for relation in relations
  relation_counts[relation.relation_type] = get(relation_counts, relation.relation_type, 0) + 1
end

for (relation_type, count) in sort(collect(relation_counts), by=x -> x[2], rev=true)
  println("   $relation_type: $count relations")
end
println()

# Relation confidence analysis
println("6. Relation confidence analysis:")
high_confidence = filter(r -> r.confidence > 0.8, relations)
medium_confidence = filter(r -> 0.6 ≤ r.confidence ≤ 0.8, relations)
low_confidence = filter(r -> r.confidence < 0.6, relations)

println("   High confidence (>0.8): $(length(high_confidence)) relations")
println("   Medium confidence (0.6-0.8): $(length(medium_confidence)) relations")
println("   Low confidence (<0.6): $(length(low_confidence)) relations")

if !isempty(high_confidence)
  println("\n   High confidence relations:")
  for relation in high_confidence
    head_entity = nothing
    tail_entity = nothing
    
    for entity in entities
      if entity.id == relation.head
        head_entity = entity
      elseif entity.id == relation.tail
        tail_entity = entity
      end
    end
    
    if head_entity !== nothing && tail_entity !== nothing
      println("     $(head_entity.text) --[$(relation.relation_type)]--> $(tail_entity.text)")
    end
  end
end

# Show relation statistics
println("\n7. Relation extraction statistics:")
if !isempty(relations)
  println("   Total relations: $(length(relations))")
  println("   Average confidence: $(round(mean(r.confidence for r in relations), digits=3))")
  println("   Min confidence: $(round(minimum(r.confidence for r in relations), digits=3))")
  println("   Max confidence: $(round(maximum(r.confidence for r in relations), digits=3))")
else
  println("   No relations extracted")
end

# Integration with knowledge graph construction
println("\n8. Integration with knowledge graph construction:")
println("   These relations form the foundation for:")
println("   • Knowledge graph triple formation")
println("   • Semantic relationship modeling")
println("   • Graph structure creation")
println("   • Biomedical knowledge representation")

println("\n✅ Relation extraction demo complete!")
println("\nNext steps:")
println("• Tail entity prediction using GraphMERT model")
println("• Triple formation and filtering")
println("• Complete knowledge graph construction")
println("• Performance evaluation against ground truth")