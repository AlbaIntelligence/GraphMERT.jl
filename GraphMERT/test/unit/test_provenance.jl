"""
Unit tests for provenance (ProvenanceRecord, get_provenance).

Covers: ProvenanceRecord construction, get_provenance(kg, index), get_provenance(kg, relation),
and fallback when no structured provenance is stored.
"""

using Test
using Dates
using GraphMERT

@testset "ProvenanceRecord" begin
  r = GraphMERT.ProvenanceRecord("doc_1", 1)
  @test r.document_id == "doc_1"
  @test r.segment_id == 1
  @test r.span_start === nothing
  @test r.span_end === nothing

  r2 = GraphMERT.ProvenanceRecord("doc_2", "sentence_3"; context="Some snippet.")
  @test r2.document_id == "doc_2"
  @test r2.segment_id == "sentence_3"
  @test r2.context == "Some snippet."
end

@testset "get_provenance" begin
  # Build a minimal KG with one relation that has provenance_record in attributes
  ke = GraphMERT.KnowledgeEntity("e1", "A", "A", 0.9, GraphMERT.TextPosition(1, 1, 1, 1), Dict{String,Any}(), now())
  kr = GraphMERT.KnowledgeRelation("e1", "e2", "REL", 0.8, Dict{String,Any}("provenance_record" => GraphMERT.ProvenanceRecord("doc_1", 1)), now())
  kg = GraphMERT.KnowledgeGraph([ke, ke], [kr], Dict{String,Any}(), now())

  rec = get_provenance(kg, 1)
  @test rec isa GraphMERT.ProvenanceRecord
  @test rec.document_id == "doc_1"
  @test rec.segment_id == 1

  rec2 = get_provenance(kg, kr)
  @test rec2.document_id == "doc_1"

  # Relation without provenance_record: fallback from provenance string or default
  kr_no_rec = GraphMERT.KnowledgeRelation("e1", "e2", "REL", 0.8, Dict{String,Any}("provenance" => "docX#2"), now())
  kg2 = GraphMERT.KnowledgeGraph([ke, ke], [kr_no_rec], Dict{String,Any}(), now())
  rec3 = get_provenance(kg2, 1)
  @test rec3.document_id == "docX"
  @test rec3.segment_id == 2
end
