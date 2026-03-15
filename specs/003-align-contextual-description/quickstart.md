# Quickstart: Reliability Pipeline (003-align-contextual-description)

**Purpose**: Get started with the reliability pipeline—provenance, validation, factuality, cleaning, and iterative seed.  
**Last Updated**: 2026-03-15  
**Feature**: [spec.md](spec.md)

---

## Prerequisites

- GraphMERT package loaded (`using GraphMERT`)
- A domain registered (e.g., biomedical or Wikipedia) when using ontology validation
- For factuality: reference or ground-truth data (optional)
- For encoder-in-path: a full model checkpoint (RoBERTa + H-GAT) loaded via `load_model`

---

## 1. Extract a KG with provenance

Enable provenance tracking so every triple is traceable to a source span (document + segment/sentence).

```julia
using GraphMERT

model = load_model("path/to/full_model")   # Full model with encoder
opts = ProcessingOptions(domain = "biomedical", enable_provenance_tracking = true)
kg = extract_knowledge_graph("Your domain text here.", model; options = opts)

# Each relation in kg has provenance
for (i, rel) in enumerate(kg.relations)
    prov = get_provenance(kg, i)   # or relation-based access
    println("Triple: $(rel.head) --$(rel.relation_type)--> $(rel.tail)")
    println("  Source: doc=$(prov.document_id) segment=$(prov.segment_id)")
end
```

---

## 2. Validate against ontology (ValidityScore)

Check how well the KG conforms to the seed ontology and get a ValidityScore.

```julia
# Domain name (string); validate_kg looks up domain and returns ValidityReport
report = validate_kg(kg, "biomedical")   # or evaluate_validity(kg, "biomedical")

println("ValidityScore: $(report.score)")
println("Valid: $(report.valid_count) / $(report.total_triples)")
```

If the ontology is missing or incomplete, validation is skipped or relaxed and the report indicates that.

---

## 3. Factuality evaluation (FActScore)

When you have reference or ground-truth triples, compute FActScore.

```julia
reference = load_reference_triples("path/to/gold_triples")   # or in-memory
result = evaluate_factscore(kg, reference)
println("FActScore: $(result.score)")
```

Without reference data, this step is not run; the system operates normally but does not produce a factuality score.

---

## 4. KG cleaning

Remove or rectify unsupported, low-confidence, or contradicted triples.

```julia
policy = CleaningPolicy(min_confidence = 0.6, require_provenance = true)
cleaned_kg = clean_kg(kg; policy = policy)
println("Before: $(length(kg.relations)) triples, after: $(length(cleaned_kg.relations))")
```

Use `cleaned_kg` for downstream tasks or as augmented seed (below).

---

## 5. Iterative seed re-use

Use the cleaned (or curated) KG as seed for the next run.

```julia
# Export cleaned KG and use as seed in next extraction or training
export_knowledge_graph(cleaned_kg, "json"; filepath = "seed_kg.json")

# Configure training or extraction to use this as augmented seed
# (exact API depends on implementation: e.g., seed_path option or seed_injection API)
# Then run extraction or training again with the augmented seed.
```

At least one such path will be documented in the main docs once implemented.

---

## 6. Full pipeline (extract → validate → clean → re-use)

```julia
model = load_model("path/to/full_model")
opts = ProcessingOptions(domain = "biomedical", enable_provenance_tracking = true)
kg = extract_knowledge_graph(corpus_text, model; options = opts)

report = validate_kg(kg, get_domain("biomedical"))
cleaned_kg = clean_kg(kg; policy = CleaningPolicy(min_confidence = 0.6))

# Optional: evaluate factuality if reference exists
# result = evaluate_factscore(cleaned_kg, reference)

# Re-use cleaned KG as seed for next run (see step 5)
```

---

## Where to read more

- **Reliability narrative and mapping to Contextual_information.md**: `reports/REFERENCE_SOURCES_AND_ENCODER.md`
- **Project status and capabilities**: `reports/PROJECT_STATUS.md`
- **API contracts for this feature**: [contracts/01-reliability-api.md](contracts/01-reliability-api.md)
- **Data model**: [data-model.md](data-model.md)
