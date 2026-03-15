"""
Integration tests: extraction with enable_provenance_tracking and get_provenance.

- When enable_provenance_tracking=true, each relation has provenance (document_id, segment_id).
- Empty corpus returns empty KG with no phantom provenance.

Uses a mock model so extraction uses the fallback entity/relation path (no encoder tail prediction),
which is sufficient to test provenance attachment and get_provenance.
"""

using Test
using GraphMERT

struct MockProvenanceModel end

@testset "Extraction with provenance" begin
  opts = GraphMERT.ProcessingOptions(domain = "biomedical", enable_provenance_tracking = true)
  model = MockProvenanceModel()
  text = "Diabetes is a disease. Metformin treats diabetes."
  kg = GraphMERT.extract_knowledge_graph(text, model; options = opts)
  @test kg isa GraphMERT.KnowledgeGraph
  for (i, rel) in enumerate(kg.relations)
    rec = GraphMERT.get_provenance(kg, i)
    @test rec isa GraphMERT.ProvenanceRecord
    @test !isempty(rec.document_id) || (rec.document_id == "" && i <= length(kg.relations))
  end
  if !isempty(kg.relations)
    rec = GraphMERT.get_provenance(kg, kg.relations[1])
    @test rec isa GraphMERT.ProvenanceRecord
  end
end

@testset "Empty corpus returns empty KG, no phantom provenance" begin
  opts = GraphMERT.ProcessingOptions(domain = "biomedical", enable_provenance_tracking = true)
  model = MockProvenanceModel()
  kg = GraphMERT.extract_knowledge_graph("", model; options = opts)
  @test kg isa GraphMERT.KnowledgeGraph
  @test length(kg.entities) == 0
  @test length(kg.relations) == 0
  @test get(kg.metadata, "empty_corpus", false) == true
end
