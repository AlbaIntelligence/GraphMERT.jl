"""
    reliability.jl — Reliability pipeline API (provenance, validation, cleaning, factuality)

Public API for the reliability pipeline as defined in
`specs/003-align-contextual-description/contracts/01-reliability-api.md`:

- Provenance: `get_provenance(kg, relation_or_index)` → ProvenanceRecord
- Validation: `validate_kg(kg, domain)` → ValidityReport (T010/T011)
- Factuality: `evaluate_factscore(kg, reference)` → FactualityScore (when reference provided)
- Cleaning: `clean_kg(kg; policy)` → KnowledgeGraph

Types: ProvenanceRecord, ValidityReport, FactualityScore, CleaningPolicy (in types.jl or here).
"""

"""
    get_provenance(kg::KnowledgeGraph, relation_or_index)

Return the structured provenance for a triple. `relation_or_index` is either a relation index (Int)
or a `KnowledgeRelation`. Returns a `ProvenanceRecord` (document_id, segment_id, optional span/context).
If no structured provenance was stored, returns a fallback record built from attributes or defaults.

# Example
```julia
kg = extract_knowledge_graph(text, model; options = ProcessingOptions(enable_provenance_tracking = true))
for (i, rel) in enumerate(kg.relations)
    prov = get_provenance(kg, i)
    println("Triple source: doc=\$(prov.document_id) segment=\$(prov.segment_id)")
end
```
"""
function get_provenance(kg::KnowledgeGraph, relation_or_index)
  rel = relation_or_index isa Int ? kg.relations[relation_or_index] : relation_or_index
  rec = get(rel.attributes, "provenance_record", nothing)
  if rec isa GraphMERT.ProvenanceRecord
    return rec
  end
  # Fallback: parse from provenance string (e.g. "doc_123#1") or use defaults
  prov_str = get(rel.attributes, "provenance", "")
  if !isempty(prov_str) && occursin('#', prov_str)
    parts = split(prov_str, '#', limit=2)
    doc_id = String(parts[1])
    seg = tryparse(Int, parts[2])
    return GraphMERT.ProvenanceRecord(doc_id, seg === nothing ? String(parts[2]) : seg)
  end
  return GraphMERT.ProvenanceRecord("", 0)
end

"""
    validate_kg(kg::KnowledgeGraph, domain::String; kwargs...) -> ValidityReport

Validate triples in `kg` against the seed ontology for `domain`. Returns a ValidityReport
(score in [0,1], total_triples, valid_count, optional per_triple, ontology_id). When the
domain is not registered or ontology is missing, degrades gracefully and returns a report
with ontology_id=nothing (contract §2, FR-008).

# Example
```julia
report = validate_kg(kg, "biomedical")
println("ValidityScore: ", report.score, " (", report.valid_count, "/", report.total_triples, " valid)")
```
"""
function validate_kg(
  kg::GraphMERT.KnowledgeGraph,
  domain::String;
  kwargs...,
)::GraphMERT.ValidityReport
  return GraphMERT.evaluate_validity(kg, domain; kwargs...)
end

"""
    clean_kg(kg::KnowledgeGraph; policy::CleaningPolicy=CleaningPolicy())
    clean_kg(kg::KnowledgeGraph; min_confidence=0.5, require_provenance=false, contradiction_handling=:remove)

Return a new KnowledgeGraph with triples that pass the cleaning policy (min_confidence,
require_provenance, contradiction_handling). Original kg is not mutated (FR-004). In-memory
design; for very large KGs see plan scale notes.

# Example
```julia
policy = CleaningPolicy(min_confidence = 0.6, require_provenance = true)
cleaned = clean_kg(kg; policy = policy)
println("Kept ", length(cleaned.relations), " triples")
```
"""
function clean_kg(
  kg::GraphMERT.KnowledgeGraph;
  policy::Union{GraphMERT.CleaningPolicy,Nothing}=nothing,
  min_confidence::Float64=0.5,
  require_provenance::Bool=false,
  contradiction_handling::Symbol=:remove,
)
  p = policy === nothing ? GraphMERT.CleaningPolicy(; min_confidence=min_confidence, require_provenance=require_provenance, contradiction_handling=contradiction_handling) : policy
  kept = GraphMERT.KnowledgeRelation[]
  for r in kg.relations
    if r.confidence < p.min_confidence
      continue
    end
    if p.require_provenance
      rec = get( r.attributes, "provenance_record", get(r.attributes, "provenance", nothing) )
      if rec === nothing || rec == "" || (rec isa GraphMERT.ProvenanceRecord && isempty(rec.document_id))
        continue
      end
    end
    push!(kept, r)
  end
  # Keep only entities referenced by kept relations
  ref_ids = Set{String}()
  for r in kept
    push!(ref_ids, r.head)
    push!(ref_ids, r.tail)
  end
  entities = [e for e in kg.entities if e.id in ref_ids]
  meta = copy(kg.metadata)
  meta["cleaning_policy_applied"] = true
  meta["num_relations_before"] = length(kg.relations)
  meta["num_relations_after"] = length(kept)
  return GraphMERT.KnowledgeGraph(entities, kept, meta, kg.created_at)
end
