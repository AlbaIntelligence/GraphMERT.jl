"""
Example 3: Knowledge Graph Construction - Domain System Version
===============================================================

This example demonstrates the construction of a biomedical knowledge graph
using the entities and relations extracted with the domain system, following
the progression of the original GraphMERT paper.

Based on: GraphMERT paper - Section 3.3 Knowledge Graph Construction
"""

using GraphMERT
using Statistics: mean

println("="^60)
println("GraphMERT Example 3: Knowledge Graph Construction (Domain System)")
println("="^60)

# Load and register the biomedical domain
println("\n1. Loading biomedical domain...")
include("../../GraphMERT/src/domains/biomedical.jl")

bio_domain = load_biomedical_domain()
register_domain!("biomedical", bio_domain)
set_default_domain("biomedical")

println("   ✅ Biomedical domain loaded and registered\n")

# Sample biomedical texts for knowledge graph construction
texts = [
    """
    Alzheimer's disease is a neurodegenerative disorder characterized by
    progressive cognitive decline and memory loss. The disease is associated
    with the accumulation of beta-amyloid plaques and tau protein tangles
    in the brain.
    """,
    """
    Donepezil is a cholinesterase inhibitor that treats Alzheimer's disease
    by preventing the breakdown of acetylcholine in the brain. The drug
    binds to acetylcholinesterase enzyme and inhibits its activity.
    """,
    """
    Memantine is an NMDA receptor antagonist used to treat moderate to
    severe Alzheimer's disease. It works by blocking excessive glutamate
    activity in the brain, which can damage nerve cells.
    """,
    """
    Acetylcholine is a neurotransmitter that plays a crucial role in
    memory and learning. It is synthesized by choline acetyltransferase
    and broken down by acetylcholinesterase.
    """,
]

println("📄 Processing $(length(texts)) biomedical texts...\n")

# Step 1: Extract entities from all texts using domain system
println("🔍 Step 1: Extracting entities from all texts...")
all_entities = Vector{GraphMERT.Entity}()
options = ProcessingOptions(domain="biomedical")

for (i, text) in enumerate(texts)
    println("  Processing text $i/$(length(texts))...")
    entities = extract_entities(bio_domain, text, options)
    append!(all_entities, entities)
end

println("  Total entities extracted: $(length(all_entities))")

# Remove duplicates and merge entities
println("\n🔄 Step 2: Merging and deduplicating entities...")
entity_map = Dict{String, GraphMERT.Entity}()

for entity in all_entities
    normalized_text = lowercase(strip(entity.text))
    if !haskey(entity_map, normalized_text) || entity.confidence > entity_map[normalized_text].confidence
        entity_map[normalized_text] = entity
    end
end

unique_entities = collect(values(entity_map))
println("  Original entities: $(length(all_entities))")
println("  Unique entities: $(length(unique_entities))")

# Step 3: Extract relations using domain system
println("\n🔗 Step 3: Extracting relations...")
all_relations = Vector{GraphMERT.Relation}()

for (i, text) in enumerate(texts)
    println("  Processing text $i/$(length(texts))...")
    
    # Extract entities from this text
    text_entities = extract_entities(bio_domain, text, options)
    
    # Extract relations between entities in this text
    if !isempty(text_entities)
        relations = extract_relations(bio_domain, text_entities, text, options)
        append!(all_relations, relations)
    end
end

println("  Found $(length(all_relations)) relations")

# Step 4: Create knowledge graph
println("\n🕸️  Step 4: Creating knowledge graph...")

# Create entity ID mapping for relations
entity_id_map = Dict{String, String}()
for entity in unique_entities
    entity_id_map[entity.text] = entity.id
end

# Filter relations to only include those with valid entity IDs
valid_relations = Vector{GraphMERT.Relation}()
for relation in all_relations
    if haskey(entity_id_map, relation.head) && haskey(entity_id_map, relation.tail)
        # Update relation to use entity IDs
        updated_relation = GraphMERT.Relation(
            entity_id_map[relation.head],
            entity_id_map[relation.tail],
            relation.relation_type,
            relation.confidence,
            relation.attributes,
            relation.created_at,
        )
        push!(valid_relations, updated_relation)
    end
end

# Build the knowledge graph
kg = KnowledgeGraph(
    unique_entities,
    valid_relations,
    Dict(
        "source" => "knowledge_graph_construction_demo",
        "domain" => "biomedical",
        "num_texts" => length(texts),
        "extraction_time" => string(now()),
    ),
)

println("  Knowledge graph created successfully!")
println("  • Entities: $(length(kg.entities))")
println("  • Relations: $(length(kg.relations))")

# Step 5: Analyze the knowledge graph
println("\n📊 Step 5: Analyzing knowledge graph...")

# Basic statistics
println("\n📈 Knowledge Graph Statistics:")
println("  Total entities: $(length(kg.entities))")
println("  Total relations: $(length(kg.relations))")

# Entity type distribution
entity_type_counts = Dict{String, Int}()
for entity in kg.entities
    entity_type = get(entity.attributes, "entity_type", entity.entity_type)
    entity_type_counts[entity_type] = get(entity_type_counts, entity_type, 0) + 1
end

println("\n🏷️  Entity Type Distribution:")
for (type_name, count) in sort(collect(entity_type_counts), by = x->x[2], rev = true)
    println("  $type_name: $count")
end

# Relation type distribution
relation_type_counts = Dict{String, Int}()
for relation in kg.relations
    relation_type_counts[relation.relation_type] = get(relation_type_counts, relation.relation_type, 0) + 1
end

println("\n🔗 Relation Type Distribution:")
for (rel_type, count) in sort(collect(relation_type_counts), by = x->x[2], rev = true)
    println("  $rel_type: $count")
end

# Confidence statistics
if !isempty(kg.entities)
    entity_confidences = [e.confidence for e in kg.entities]
    println("\n📊 Entity Confidence Statistics:")
    println("  Average: $(round(mean(entity_confidences), digits=3))")
    println("  Min: $(round(minimum(entity_confidences), digits=3))")
    println("  Max: $(round(maximum(entity_confidences), digits=3))")
end

if !isempty(kg.relations)
    relation_confidences = [r.confidence for r in kg.relations]
    println("\n📊 Relation Confidence Statistics:")
    println("  Average: $(round(mean(relation_confidences), digits=3))")
    println("  Min: $(round(minimum(relation_confidences), digits=3))")
    println("  Max: $(round(maximum(relation_confidences), digits=3))")
end

# Show sample entities and relations
println("\n📋 Sample Entities:")
for (i, entity) in enumerate(kg.entities[1:min(5, length(kg.entities))])
    println("  $i. $(entity.text) ($(entity.entity_type), conf: $(round(entity.confidence, digits=3)))")
end

println("\n📋 Sample Relations:")
for (i, relation) in enumerate(kg.relations[1:min(5, length(kg.relations))])
    # Find entity texts
    head_text = ""
    tail_text = ""
    for entity in kg.entities
        if entity.id == relation.head
            head_text = entity.text
        elseif entity.id == relation.tail
            tail_text = entity.text
        end
    end
    if !isempty(head_text) && !isempty(tail_text)
        println("  $i. $head_text --[$(relation.relation_type)]--> $tail_text (conf: $(round(relation.confidence, digits=3)))")
    end
end

# Step 6: Domain-specific metrics
println("\n📊 Step 6: Domain-specific evaluation metrics...")
try
    domain_metrics = create_evaluation_metrics(bio_domain, kg)
    println("  UMLS linking coverage: $(round(get(domain_metrics, "umls_linking_coverage", 0.0) * 100, digits=1))%")
    println("  Average entity confidence: $(round(get(domain_metrics, "average_entity_confidence", 0.0), digits=3))")
    println("  Average relation confidence: $(round(get(domain_metrics, "average_relation_confidence", 0.0), digits=3))")
catch e
    println("  ⚠️  Domain metrics not available: $e")
end

println("\n" * "="^60)
println("✅ Example 3 completed successfully!")
println("Next: Run 04_seed_injection_demo.jl or other examples")
println("="^60)
