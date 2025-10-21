"""
Seed KG Injection Algorithm Demo

This example demonstrates the seed KG injection algorithm used in GraphMERT.
The algorithm injects relevant knowledge graph triples into training data to enable
vocabulary transfer from semantic space to syntactic space.

Key concepts demonstrated:
1. Entity linking using SapBERT-style similarity matching
2. Triple selection from seed knowledge graph
3. Injection algorithm with score and relation diversity bucketing
4. Semantic consistency validation
5. Integration with the leafy chain graph structure
"""

using GraphMERT
using GraphMERT: SeedInjectionConfig, SemanticTriple, EntityLinkingResult, link_entity_sapbert, select_triples_for_entity,
  inject_seed_kg, select_triples_for_injection, bucket_by_score, bucket_by_relation_frequency,
  validate_injected_triples

println("=== Seed KG Injection Demo ===")

# 1. Create injection configuration
println("\n1. Creating injection configuration...")
config = SeedInjectionConfig(
  0.5,     # entity_linking_threshold: Jaccard similarity threshold
  10,      # top_k_candidates: Number of candidates from SapBERT
  40,      # top_n_triples_per_entity: Number of triples per entity
  0.7,     # alpha_score_threshold: Minimum similarity score
  10,      # score_bucket_size: Number of buckets by score
  5,       # relation_bucket_size: Number of buckets by relation frequency
  0.2,     # injection_ratio: Percentage of sequences to inject
  10       # max_triples_per_sequence: Maximum triples injected per sequence
)
println("Config: threshold=$(config.entity_linking_threshold), top_k=$(config.top_k_candidates)")

# 2. Create seed knowledge graph
println("\n2. Creating seed knowledge graph...")
seed_kg = [
  SemanticTriple("diabetes", "C0011849", "treats", "metformin", [2156, 23421], 0.95, "UMLS"),
  SemanticTriple("diabetes", "C0011849", "complicates", "pregnancy", [11234, 15678], 0.87, "UMLS"),
  SemanticTriple("metformin", "C0025598", "inhibits", "diabetes", [23421, 2156], 0.92, "UMLS"),
  SemanticTriple("metformin", "C0025598", "metabolized_by", "liver", [23421, 18901], 0.78, "UMLS"),
  SemanticTriple("diabetes", "C0011849", "increases_risk", "cardiovascular_disease", [2156, 26789], 0.89, "UMLS"),
]
println("Created seed KG with $(length(seed_kg)) triples")

# 3. Demonstrate entity linking
println("\n3. Demonstrating entity linking...")
test_entities = ["diabetes", "metformin", "pregnancy", "cardiovascular"]

for entity in test_entities
  linked = link_entity_sapbert(entity, config)
  println("Entity '$entity' linked to:")
  for result in linked
    println("  - CUI: $(result.cui), Name: $(result.preferred_name), Score: $(round(result.similarity_score, digits=3))")
  end
  println()
end

# 4. Demonstrate triple selection
println("\n4. Demonstrating triple selection...")
cui = "C0011849"  # Diabetes Mellitus
selected_triples = select_triples_for_entity(cui, config)
println("Selected $(length(selected_triples)) triples for entity $cui:")
for triple in selected_triples
  println("  - $(triple.head) --[$(triple.relation)]--> $(triple.tail) (score: $(round(triple.score, digits=3)))")
end

# 5. Demonstrate injection algorithm
println("\n5. Demonstrating injection algorithm...")
sequences = [
  "Diabetes mellitus is a chronic metabolic disorder characterized by elevated blood glucose levels.",
  "Metformin is commonly prescribed for the treatment of type 2 diabetes.",
  "Pregnancy can be complicated by gestational diabetes.",
  "Cardiovascular disease is a major complication of diabetes.",
  "The liver plays a key role in glucose metabolism."
]

# Mock entity extraction (would be implemented in text module)
function mock_extract_entities(text::String)
  # Simple entity extraction for demo
  entities = String[]
  if occursin("diabetes", lowercase(text))
    push!(entities, "diabetes")
  end
  if occursin("metformin", lowercase(text))
    push!(entities, "metformin")
  end
  if occursin("pregnancy", lowercase(text))
    push!(entities, "pregnancy")
  end
  if occursin("cardiovascular", lowercase(text))
    push!(entities, "cardiovascular")
  end
  return entities
end

# Inject seed KG into sequences
injected_sequences = Vector{Tuple{String,Vector{SemanticTriple}}}()
for sequence in sequences
  entities = mock_extract_entities(sequence)
  if !isempty(entities)
    # Link entities
    linked_entities = Vector{GraphMERT.EntityLinkingResult}()
    for entity in entities
      linked = link_entity_sapbert(entity, config)
      append!(linked_entities, linked)
    end

    # Select triples
    selected_triples = select_triples_for_injection(linked_entities, seed_kg, config)

    # Limit to max_triples_per_sequence
    if length(selected_triples) > config.max_triples_per_sequence
      selected_triples = selected_triples[1:config.max_triples_per_sequence]
    end

    push!(injected_sequences, (sequence, selected_triples))
  else
    push!(injected_sequences, (sequence, Vector{SemanticTriple}()))
  end
end

println("Injection results:")
for (i, (sequence, triples)) in enumerate(injected_sequences)
  println("Sequence $i: $(length(triples)) triples injected")
  for triple in triples
    println("  - $(triple.head) --[$(triple.relation)]--> $(triple.tail)")
  end
end

# 6. Demonstrate bucketing algorithms
println("\n6. Demonstrating bucketing algorithms...")

# Create test triples with different scores and relations
test_triples = [
  SemanticTriple("entity1", "C001", "treats", "entity2", [100], 0.95, "test"),
  SemanticTriple("entity1", "C001", "causes", "entity3", [101], 0.90, "test"),
  SemanticTriple("entity1", "C001", "prevents", "entity4", [102], 0.85, "test"),
  SemanticTriple("entity2", "C002", "inhibits", "entity1", [103], 0.88, "test"),
  SemanticTriple("entity2", "C002", "treats", "entity5", [104], 0.82, "test"),
]

# Score bucketing
score_buckets = bucket_by_score(test_triples, 3)
println("Score buckets:")
for (i, bucket) in enumerate(score_buckets)
  println("  Bucket $i: $(length(bucket)) triples")
  for triple in bucket
    println("    - $(triple.relation) (score: $(round(triple.score, digits=3)))")
  end
end

# Relation frequency bucketing
relation_buckets = bucket_by_relation_frequency(test_triples, 3)
println("\nRelation frequency buckets:")
for (i, bucket) in enumerate(relation_buckets)
  println("  Bucket $i: $(length(bucket)) triples")
  relations = Set(t.relation for t in bucket)
  println("    Relations: $relations")
end

# 7. Demonstrate validation
println("\n7. Demonstrating validation...")
test_sequence = "Diabetes mellitus is a chronic metabolic disorder."
injected_triples = [
  SemanticTriple("diabetes", "C0011849", "treats", "metformin", [2156], 0.95, "UMLS"),
  SemanticTriple("diabetes", "C0011849", "unrelated", "something", [999], 0.85, "UMLS")
]

validation_results = validate_injected_triples(test_sequence, injected_triples)
println("Validation results:")
for (triple, is_valid) in validation_results
  println("  - $(triple.head) --[$(triple.relation)]--> $(triple.tail): $(is_valid ? "VALID" : "INVALID")")
end

# 8. Integration with leafy chain graph
println("\n8. Integration with leafy chain graph...")

# Create a graph and inject a triple
graph = create_empty_chain_graph()
test_triple = SemanticTriple("diabetes", "C0011849", "treats", "metformin", [2156, 23421], 0.95, "UMLS")

success = inject_triple!(graph, test_triple, 1)
println("Triple injection into graph: $success")

if success
  # Convert to sequence to show integration
  sequence = graph_to_sequence(graph)
  println("Graph sequence length: $(length(sequence))")
  println("First 10 tokens: $(sequence[1:10])")
  println("Injected tokens at positions 2-3: $(sequence[2:3])")
end

println("\n✅ Seed KG Injection demo complete!")
println("\nAlgorithm demonstrates:")
println("• Entity linking with SapBERT-style similarity matching")
println("• Triple selection with score and relation diversity")
println("• Injection algorithm with semantic consistency")
println("• Integration with leafy chain graph structure")
println("• Validation to ensure semantic coherence")
