"""
Domain Switching Example
========================

This example demonstrates how to use multiple domains in the same session,
showing the flexibility of the GraphMERT domain system. We'll extract
knowledge graphs from both biomedical and Wikipedia-style text using
their respective domain providers.

Key concepts demonstrated:
1. Loading multiple domains simultaneously
2. Registering multiple domains
3. Switching between domains based on content type
4. Comparing extraction results across domains
"""

using GraphMERT
using Statistics: mean

println("="^60)
println("GraphMERT Domain Switching Example")
println("="^60)

# Load both domains
println("\n1. Loading domains...")
include("../../GraphMERT/src/domains/biomedical.jl")
include("../../GraphMERT/src/domains/wikipedia.jl")

bio_domain = load_biomedical_domain()
wiki_domain = load_wikipedia_domain()

register_domain!("biomedical", bio_domain)
register_domain!("wikipedia", wiki_domain)

println("   ✅ Biomedical domain registered")
println("   ✅ Wikipedia domain registered")
println("   Available domains: $(list_domains())")

# Sample texts from different domains
biomedical_text = """
Diabetes mellitus is a chronic metabolic disorder characterized by
elevated blood glucose levels. Metformin is commonly used to treat
type 2 diabetes. Insulin resistance is a key feature of type 2 diabetes.
Cardiovascular disease is a major complication of diabetes.
"""

wikipedia_text = """
The Renaissance was a period of cultural, artistic, and intellectual
rebirth in Europe from the 14th to 17th centuries. Artists like
Leonardo da Vinci and Michelangelo created masterpieces that continue
to influence art today. The period also saw advances in science,
literature, and philosophy.
"""

# Extract using biomedical domain
println("\n2. Extracting from biomedical text using biomedical domain...")
bio_options = ProcessingOptions(domain="biomedical")
bio_entities = extract_entities(bio_domain, biomedical_text, bio_options)
bio_relations = extract_relations(bio_domain, bio_entities, biomedical_text, bio_options)

println("   Found $(length(bio_entities)) entities and $(length(bio_relations)) relations")
println("   Entity types: $(unique([e.entity_type for e in bio_entities]))")
println("   Average confidence: $(round(mean(e.confidence for e in bio_entities), digits=3))")

# Extract using Wikipedia domain
println("\n3. Extracting from Wikipedia text using Wikipedia domain...")
wiki_options = ProcessingOptions(domain="wikipedia")
wiki_entities = extract_entities(wiki_domain, wikipedia_text, wiki_options)
wiki_relations = extract_relations(wiki_domain, wiki_entities, wikipedia_text, wiki_options)

println("   Found $(length(wiki_entities)) entities and $(length(wiki_relations)) relations")
println("   Entity types: $(unique([e.entity_type for e in wiki_entities]))")
println("   Average confidence: $(round(mean(e.confidence for e in wiki_entities), digits=3))")

# Compare extraction capabilities
println("\n4. Comparing domain capabilities...")
println("   Biomedical domain:")
println("     • Optimized for medical/biomedical terminology")
println("     • Supports UMLS integration")
println("     • Entity types: DISEASE, DRUG, PROTEIN, GENE, etc.")
println("     • Relation types: TREATS, CAUSES, INHIBITS, etc.")

println("\n   Wikipedia domain:")
println("     • Optimized for general knowledge")
println("     • Entity types: PERSON, ORGANIZATION, LOCATION, CONCEPT, etc.")
println("     • Relation types: CREATED_BY, BORN_IN, WORKED_AT, etc.")

# Demonstrate domain switching with same text
println("\n5. Testing domain switching with same text...")
test_text = "Artificial Intelligence is transforming healthcare and medical diagnosis."

println("   Text: \"$test_text\"\n")

# Try with biomedical domain
println("   Using biomedical domain:")
bio_test_entities = extract_entities(bio_domain, test_text, bio_options)
println("     Found $(length(bio_test_entities)) entities")
for entity in bio_test_entities
    println("       • $(entity.text) ($(entity.entity_type))")
end

# Try with Wikipedia domain
println("\n   Using Wikipedia domain:")
wiki_test_entities = extract_entities(wiki_domain, test_text, wiki_options)
println("     Found $(length(wiki_test_entities)) entities")
for entity in wiki_test_entities
    println("       • $(entity.text) ($(entity.entity_type))")
end

println("\n   Note: Different domains extract different entity types from the same text!")

# Demonstrate default domain behavior
println("\n6. Demonstrating default domain behavior...")
set_default_domain("biomedical")
println("   Default domain set to: $(get_default_domain())")

# Create knowledge graphs using domain system
println("\n7. Creating knowledge graphs using domain system...")

# Biomedical knowledge graph
bio_model = create_graphmert_model(GraphMERTConfig())
bio_graph = extract_knowledge_graph(biomedical_text, bio_model; options=bio_options)
println("   Biomedical KG: $(length(bio_graph.entities)) entities, $(length(bio_graph.relations)) relations")

# Wikipedia knowledge graph
wiki_graph = extract_knowledge_graph(wikipedia_text, bio_model; options=wiki_options)
println("   Wikipedia KG: $(length(wiki_graph.entities)) entities, $(length(wiki_graph.relations)) relations")

# Summary
println("\n" * "="^60)
println("✅ Domain Switching Example Complete!")
println("\nKey Takeaways:")
println("• Multiple domains can be loaded and registered simultaneously")
println("• Domain switching is as simple as changing ProcessingOptions(domain=\"...\")")
println("• Each domain is optimized for its specific domain (biomedical vs. general knowledge)")
println("• The same text can produce different results depending on the active domain")
println("• Domain system enables GraphMERT to work across multiple application domains")
println("="^60)
