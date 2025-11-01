"""
Basic Entity Extraction Demo - Domain System Version

This example demonstrates the first stage of the GraphMERT knowledge graph
extraction pipeline: discovering biomedical entities from text using the domain system.

Key concepts demonstrated:
1. Loading and registering the biomedical domain
2. Entity extraction using domain providers
3. Confidence scoring based on term characteristics
4. Entity position tracking in source text
5. Basic biomedical term identification
"""

using GraphMERT
using Statistics: mean

println("=== Basic Entity Extraction Demo (Domain System) ===\n")

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

# Extract biomedical entities using domain system
println("3. Extracting biomedical entities using domain system...")
options = ProcessingOptions(domain="biomedical")
entities = extract_entities(bio_domain, text, options)

println("   Found $(length(entities)) biomedical entities:\n")

for entity in entities
    println("   Entity: $(entity.text)")
    println("     Type: $(entity.entity_type)")
    println("     Domain: $(entity.domain)")
    println("     Confidence: $(round(entity.confidence, digits=3))")
    println("     Position: $(entity.position.start)-$(entity.position.stop)")
    if haskey(entity.attributes, "entity_type_enum")
        println("     Enum Type: $(entity.attributes["entity_type_enum"])")
    end
    println()
end

# Demonstrate confidence scoring
println("4. Entity confidence analysis:")
for entity in entities
    println("   '$(entity.text)' confidence: $(round(entity.confidence, digits=3))")
    # Simple confidence factors analysis
    factors = String[]
    if length(entity.text) > 10
        push!(factors, "long_term")
    end
    if isuppercase(entity.text[1])
        push!(factors, "capitalized")
    end
    if count(isspace, entity.text) > 0
        push!(factors, "multi_word")
    end
    println("     Factors: $(isempty(factors) ? "basic" : join(factors, ", "))")
end

# Show entity statistics
println("\n5. Entity extraction statistics:")
total_chars = length(text)
entity_chars = sum(length(e.text) for e in entities)
coverage = entity_chars / total_chars * 100

println("   Total characters: $total_chars")
println("   Entity characters: $entity_chars")
println("   Coverage: $(round(coverage, digits=1))%")
println("   Average confidence: $(round(mean(e.confidence for e in entities), digits=3))")

# Integration with knowledge graph construction
println("\n6. Integration with knowledge graph construction:")
println("   These entities form the foundation for:")
println("   • Relation extraction between entity pairs")
println("   • Knowledge graph triple formation")
println("   • Semantic relationship modeling")
println("   • Biomedical ontology alignment")

println("\n✅ Basic entity extraction demo complete!")
println("\nNext steps:")
println("• Relation matching between entity pairs")
println("• Tail entity prediction using GraphMERT model")
println("• Triple formation and filtering")
println("• Complete knowledge graph construction")
