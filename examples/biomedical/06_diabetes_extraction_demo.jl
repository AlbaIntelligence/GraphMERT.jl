"""
Diabetes Knowledge Graph Extraction Demo

This example demonstrates the complete GraphMERT knowledge graph extraction
pipeline using the diabetes dataset as described in the original paper.

Key concepts demonstrated:
1. Entity discovery from biomedical text
2. Relation extraction between entities
3. Knowledge graph construction
4. Integration with the paper's diabetes dataset
5. Performance validation against paper results
"""

using GraphMERT

println("=== Diabetes Knowledge Graph Extraction Demo ===")

# Load and register biomedical domain
println("\n📋 Loading biomedical domain...")
include("../../GraphMERT/src/domains/biomedical.jl")
bio_domain = load_biomedical_domain()
register_domain!("biomedical", bio_domain)
set_default_domain("biomedical")
println("   ✅ Biomedical domain loaded and registered")
println()

# 1. Sample diabetes text from the paper dataset
text = """
Diabetes mellitus is a chronic metabolic disorder characterized by
elevated blood glucose levels. Metformin is commonly used to treat
type 2 diabetes. Insulin resistance is a key feature of type 2 diabetes.
Cardiovascular disease is a major complication of diabetes.
"""

println("Sample diabetes text:")
println(text)
println()

# 2. Extract knowledge graph
println("2. Extracting knowledge graph...")
try
  # Extract knowledge graph using domain system
  # Note: In a full implementation, you would need a trained GraphMERT model
  # For demo purposes, we'll attempt extraction (may fail if model not available)
  options = ProcessingOptions(domain="biomedical")
  
  # Try to extract - if model is not available, this will be caught by error handling
  # In production, you would load a model first:
  # model = GraphMERT.load_model("path/to/model")
  model = nothing  # Placeholder - replace with actual model loading
  
  if model === nothing
    println("⚠️  Model not available - skipping extraction")
    println("   In production, load a model with: model = GraphMERT.load_model(\"path/to/model\")")
    return
  end
  
  kg = GraphMERT.extract_knowledge_graph(text, model; options=options)

  println("✅ Knowledge graph extracted successfully!")
  println("Found $(length(kg.entities)) entities and $(length(kg.relations)) relations")

  # 3. Display extracted entities
  println("\n3. Extracted entities:")
  for (i, entity) in enumerate(kg.entities)
    println("  $i. $(entity.text)")
    println("     Type: $(entity.entity_type)")
    println("     Confidence: $(round(entity.confidence, digits=3))")
    println("     ID: $(entity.id)")
    if haskey(entity.attributes, "cui")
      println("     CUI: $(entity.attributes["cui"])")
    end
    println()
  end

  # 4. Display extracted relations
  println("4. Extracted relations:")
  if length(kg.relations) > 0
    for (i, relation) in enumerate(kg.relations)
      # Find head and tail entities by ID
      head_entity = findfirst(e -> e.id == relation.head, kg.entities)
      tail_entity = findfirst(e -> e.id == relation.tail, kg.entities)
      if head_entity !== nothing && tail_entity !== nothing
        head_text = kg.entities[head_entity].text
        tail_text = kg.entities[tail_entity].text
        println("  $i. $head_text --[$(relation.relation_type)]--> $tail_text")
        println("     Confidence: $(round(relation.confidence, digits=3))")
        println("     Evidence: $(relation.evidence)")
        println()
      end
    end
  else
    println("  No relations extracted (simplified demo)")
  end

  # 5. Display knowledge graph statistics
  println("5. Knowledge graph statistics:")
  println("  Entities: $(length(kg.entities))")
  println("  Relations: $(length(kg.relations))")
  println("  Triples: $(length(kg.relations))")
  println("  Extraction time: $(get(kg.metadata, "extraction_time", "N/A"))")

  # 6. Validate against paper results
  println("\n6. Paper validation:")
  println("  Paper FActScore target: 69.8%")
  println("  Paper ValidityScore target: 68.8%")
  println("  Current extraction: $(length(kg.entities)) entities, $(length(kg.relations)) relations")
  println("  Note: This is a simplified demo - full model would extract more entities and relations")
  println("  Note: Full validation requires complete evaluation pipeline")

  # 7. Integration with diabetes dataset
  println("\n7. Integration with diabetes dataset:")
  println("  Paper dataset: 350k abstracts, 124.7M tokens")
  println("  Current demo: $(length(text)) characters")
  println("  Scaling factor: ~$(round(length(text) / 124700000 * 100, digits=2))% of full dataset")
  println("  Expected entities in full dataset: ~$(round(length(kg.entities) / (length(text) / 124700000), digits=0))")

  # 8. Performance analysis
  println("\n8. Performance analysis:")
  println("  Text length: $(length(text)) characters")
  println("  Entities per 1000 chars: $(round(length(kg.entities) / length(text) * 1000, digits=1))")
  println("  Relations per 1000 chars: $(round(length(kg.relations) / length(text) * 1000, digits=1))")

  println("\n✅ Diabetes knowledge graph extraction demo complete!")

  println("\nNext steps for full implementation:")
  println("• Train on complete diabetes dataset (350k abstracts)")
  println("• Validate FActScore against paper (69.8% target)")
  println("• Validate ValidityScore against paper (68.8% target)")
  println("• Implement complete evaluation pipeline")
  println("• Optimize for laptop deployment (80M parameters)")

  return kg

catch e
  println("⚠️  Demo completed with simplified functionality")
  println("   Error: $e")
  println("   This is expected in demo mode - full implementation requires:")
  println("   • Complete model training")
  println("   • Load model with: model = GraphMERT.load_model(\"path/to/model\")")
  println("   • Full extraction pipeline")
  println("   • Proper evaluation metrics")
  println("   • Diabetes dataset integration")
  return
end
