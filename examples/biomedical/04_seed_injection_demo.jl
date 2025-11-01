"""
Seed KG Injection Demo for GraphMERT.jl

This demo showcases the complete seed knowledge graph injection pipeline,
demonstrating how biomedical knowledge is injected into training sequences
to enable vocabulary transfer and semantic grounding.

Features demonstrated:
- Entity linking with SapBERT embeddings
- Character 3-gram Jaccard similarity filtering
- Contextual triple selection (top-40)
- Score and relation bucketing for diversity
- Injection algorithm (Paper Appendix B)
- Validation and consistency checks
"""

using GraphMERT
using GraphMERT: SeedInjectionConfig, SemanticTriple, EntityLinkingResult,
  link_entity_sapbert, select_triples_for_entity, inject_seed_kg,
  select_triples_for_injection, validate_injected_triples,
  get_entity_name_from_cui, bucket_by_score, bucket_by_relation_frequency

function main()
  println("🧬 GraphMERT Seed KG Injection Demo")
  println("="^50)
  
  # Load and register biomedical domain
  println("\n📋 Loading biomedical domain...")
  include("../../GraphMERT/src/domains/biomedical.jl")
  bio_domain = load_biomedical_domain()
  register_domain!("biomedical", bio_domain)
  set_default_domain("biomedical")
  println("   ✅ Biomedical domain loaded and registered")
  println()

  # Create configuration
  config = SeedInjectionConfig(
    0.3,  # entity_linking_threshold
    10,   # top_k_candidates
    40,   # top_n_triples_per_entity
    0.6,  # alpha_score_threshold
    5,    # score_bucket_size
    3,    # relation_bucket_size
    0.5,  # injection_ratio
    8     # max_triples_per_sequence
  )

  println("📋 Configuration:")
  println("  • Entity linking threshold: $(config.entity_linking_threshold)")
  println("  • Top-K candidates: $(config.top_k_candidates)")
  println("  • Top-N triples per entity: $(config.top_n_triples_per_entity)")
  println("  • Alpha score threshold: $(config.alpha_score_threshold)")
  println("  • Injection ratio: $(config.injection_ratio)")
  println("  • Max triples per sequence: $(config.max_triples_per_sequence)")
  println()

  # Create comprehensive seed knowledge graph
  seed_kg = create_sample_seed_kg()
  println("🗂️  Seed Knowledge Graph:")
  println("  • Total triples: $(length(seed_kg))")
  println("  • Relations: $(length(unique([t.relation for t in seed_kg])))")
  println("  • Entities: $(length(unique([t.head for t in seed_kg])))")
  println()

  # Create diverse training sequences
  sequences = create_sample_sequences()
  println("📝 Training Sequences:")
  for (i, seq) in enumerate(sequences)
    println("  $i. $(seq)")
  end
  println()

  # Demonstrate entity linking
  println("🔗 Entity Linking Demo:")
  demo_entity_linking(config, bio_domain)
  println()

  # Demonstrate triple selection
  println("🎯 Triple Selection Demo:")
  demo_triple_selection(config, seed_kg, bio_domain)
  println()

  # Demonstrate full injection pipeline
  println("💉 Full Injection Pipeline:")
  injected_sequences = inject_seed_kg(sequences, seed_kg, config, bio_domain)

  for (i, (sequence, injected_triples)) in enumerate(injected_sequences)
    println("  Sequence $i: \"$(sequence)\"")
    println("    • Injected triples: $(length(injected_triples))")

    if !isempty(injected_triples)
      # Validate injected triples
      validation_results = validate_injected_triples(sequence, injected_triples)
      valid_count = sum(values(validation_results))
      println("    • Valid triples: $valid_count/$(length(injected_triples))")

      # Show sample injected triples
      println("    • Sample triples:")
      for (j, triple) in enumerate(injected_triples[1:min(3, length(injected_triples))])
        validity = validation_results[triple] ? "✓" : "✗"
        println("      $j. $validity $(triple.head) --[$(triple.relation)]--> $(triple.tail) (score: $(triple.score))")
      end
    end
    println()
  end

  # Demonstrate bucketing algorithms
  println("🪣 Bucketing Algorithms Demo:")
  demo_bucketing_algorithms(seed_kg, config)
  println()

  # Summary statistics
  println("📊 Summary Statistics:")
  total_injected = sum(length(triples) for (_, triples) in injected_sequences)
  total_valid = sum(sum(values(validate_injected_triples(seq, triples)))
                    for (seq, triples) in injected_sequences)

  println("  • Total sequences processed: $(length(sequences))")
  println("  • Sequences with injections: $(count(!isempty, [triples for (_, triples) in injected_sequences]))")
  println("  • Total triples injected: $total_injected")
  println("  • Valid triples: $total_valid")
  println("  • Validation rate: $(total_injected > 0 ? round(total_valid/total_injected*100, digits=1) : 0)%")
  println()

  println("✅ Seed KG Injection Demo completed successfully!")
end

function create_sample_seed_kg()
  """Create a comprehensive sample seed knowledge graph."""
  return [
    # Diabetes-related triples
    SemanticTriple("diabetes", "C0011849", "treats", "metformin", [100, 150], 0.95, "UMLS"),
    SemanticTriple("diabetes", "C0011849", "causes", "complications", [200, 250], 0.88, "UMLS"),
    SemanticTriple("diabetes", "C0011849", "prevents", "exercise", [300, 350], 0.82, "UMLS"),
    SemanticTriple("diabetes", "C0011849", "monitored_by", "glucose", [400, 450], 0.90, "UMLS"),
    SemanticTriple("diabetes", "C0011849", "complicates", "pregnancy", [500, 550], 0.85, "UMLS"),

    # Metformin-related triples
    SemanticTriple("metformin", "C0025598", "treats", "diabetes", [600, 650], 0.92, "UMLS"),
    SemanticTriple("metformin", "C0025598", "inhibits", "glucose", [700, 750], 0.87, "UMLS"),
    SemanticTriple("metformin", "C0025598", "metabolized_by", "liver", [800, 850], 0.83, "UMLS"),
    SemanticTriple("metformin", "C0025598", "causes", "nausea", [900, 950], 0.75, "UMLS"),

    # Pregnancy-related triples
    SemanticTriple("pregnancy", "C0032961", "complicates", "diabetes", [1000, 1050], 0.89, "UMLS"),
    SemanticTriple("pregnancy", "C0032961", "requires", "monitoring", [1100, 1150], 0.91, "UMLS"),
    SemanticTriple("pregnancy", "C0032961", "increases", "risk", [1200, 1250], 0.86, "UMLS"),
    SemanticTriple("pregnancy", "C0032961", "affects", "metabolism", [1300, 1350], 0.84, "UMLS"),

    # Hypertension-related triples
    SemanticTriple("hypertension", "C0020538", "causes", "stroke", [1400, 1450], 0.93, "UMLS"),
    SemanticTriple("hypertension", "C0020538", "treats", "medication", [1500, 1550], 0.88, "UMLS"),
    SemanticTriple("hypertension", "C0020538", "monitored_by", "pressure", [1600, 1650], 0.90, "UMLS"),
    SemanticTriple("hypertension", "C0020538", "prevents", "diet", [1700, 1750], 0.81, "UMLS"),

    # Additional diverse triples
    SemanticTriple("glucose", "C0017725", "measured_by", "test", [1800, 1850], 0.94, "UMLS"),
    SemanticTriple("liver", "C0023884", "metabolizes", "drugs", [1900, 1950], 0.89, "UMLS"),
    SemanticTriple("stroke", "C0038454", "causes", "paralysis", [2000, 2050], 0.87, "UMLS"),
  ]
end

function create_sample_sequences()
  """Create diverse training sequences for injection."""
  return [
    "The patient has diabetes and takes metformin for treatment.",
    "Pregnancy complications require careful monitoring of glucose levels.",
    "Hypertension is a major risk factor for stroke in elderly patients.",
    "Metformin is commonly prescribed for type 2 diabetes management.",
    "Diabetic patients should monitor their blood glucose regularly.",
    "Pregnant women with diabetes need specialized care.",
    "High blood pressure can lead to serious cardiovascular events.",
    "Regular exercise helps prevent diabetes complications.",
    "This is a non-medical sequence with no relevant entities.",
    "The weather is nice today and the sun is shining brightly."
  ]
end

function demo_entity_linking(config)
  """Demonstrate entity linking capabilities."""
  test_entities = ["diabetes", "metformin", "pregnancy", "hypertension", "glucose"]

  for entity in test_entities
    results = link_entity_sapbert(entity, config)
    println("  Entity: '$entity'")
    println("    • Candidates found: $(length(results))")

    if !isempty(results)
      best = results[1]
      println("    • Best match: '$(best.preferred_name)' (CUI: $(best.cui))")
      println("    • Similarity score: $(round(best.similarity_score, digits=3))")
      println("    • Semantic types: $(join(best.semantic_types, ", "))")
    end
    println()
  end
end

function demo_triple_selection(config, seed_kg, domain)
  """Demonstrate triple selection for specific entities."""
  test_cuis = ["C0011849", "C0025598", "C0032961"]  # diabetes, metformin, pregnancy

  for cui in test_cuis
    triples = select_triples_for_entity(cui, config, domain)
    entity_name = get_entity_name_from_cui(cui)

    println("  Entity: '$entity_name' (CUI: $cui)")
    println("    • Available triples: $(length(triples))")

    if !isempty(triples)
      println("    • Sample triples:")
      for (i, triple) in enumerate(triples[1:min(3, length(triples))])
        println("      $i. $(triple.head) --[$(triple.relation)]--> $(triple.tail) (score: $(triple.score))")
      end
    end
    println()
  end
end

function demo_bucketing_algorithms(seed_kg, config)
  """Demonstrate score and relation bucketing algorithms."""
  # Test score bucketing
  println("  Score Bucketing:")
  score_buckets = bucket_by_score(seed_kg, config.score_bucket_size)
  println("    • Number of buckets: $(length(score_buckets))")
  for (i, bucket) in enumerate(score_buckets)
    if !isempty(bucket)
      avg_score = sum(t.score for t in bucket) / length(bucket)
      println("    • Bucket $i: $(length(bucket)) triples, avg score: $(round(avg_score, digits=3))")
    end
  end
  println()

  # Test relation frequency bucketing
  println("  Relation Frequency Bucketing:")
  relation_buckets = bucket_by_relation_frequency(seed_kg, config.relation_bucket_size)
  println("    • Number of buckets: $(length(relation_buckets))")
  for (i, bucket) in enumerate(relation_buckets)
    if !isempty(bucket)
      relations = unique([t.relation for t in bucket])
      println("    • Bucket $i: $(length(bucket)) triples, relations: $(join(relations, ", "))")
    end
  end
end

# Run the demo
if abspath(PROGRAM_FILE) == @__FILE__
  main()
end