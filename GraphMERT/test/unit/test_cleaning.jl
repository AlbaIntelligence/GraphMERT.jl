"""
Unit tests for clean_kg and CleaningPolicy (US3).

Covers: min_confidence filter, require_provenance filter, policy struct, output is new KG.
"""

using Test
using Dates
using GraphMERT

function _kg_with_relations(; with_provenance::Bool=false)
  entities = [
    GraphMERT.KnowledgeEntity("e1", "A", "A", 0.9, GraphMERT.TextPosition(1, 1, 1, 1), Dict{String,Any}(), now()),
    GraphMERT.KnowledgeEntity("e2", "B", "B", 0.9, GraphMERT.TextPosition(1, 1, 1, 1), Dict{String,Any}(), now()),
  ]
  attrs = with_provenance ? Dict{String,Any}("provenance_record" => GraphMERT.ProvenanceRecord("doc1", 1)) : Dict{String,Any}()
  relations = [
    GraphMERT.KnowledgeRelation("e1", "e2", "R", 0.9, copy(attrs), now()),
    GraphMERT.KnowledgeRelation("e1", "e2", "S", 0.3, copy(attrs), now()),
    GraphMERT.KnowledgeRelation("e1", "e2", "T", 0.6, copy(attrs), now()),
  ]
  GraphMERT.KnowledgeGraph(entities, relations, Dict{String,Any}(), now())
end

@testset "CleaningPolicy" begin
  p = GraphMERT.CleaningPolicy(; min_confidence=0.5, require_provenance=false)
  @test p.min_confidence == 0.5
  @test p.require_provenance == false
  p2 = GraphMERT.CleaningPolicy(; require_provenance=true, contradiction_handling=:flag)
  @test p2.contradiction_handling == :flag
end

@testset "clean_kg min_confidence" begin
  kg = _kg_with_relations(with_provenance=false)
  cleaned = GraphMERT.clean_kg(kg; min_confidence=0.5)
  @test cleaned isa GraphMERT.KnowledgeGraph
  @test length(cleaned.relations) == 2
  @test all(r.confidence >= 0.5 for r in cleaned.relations)
  @test kg.relations !== cleaned.relations
  @test get(cleaned.metadata, "cleaning_policy_applied", false) == true
end

@testset "clean_kg require_provenance" begin
  kg_no_prov = _kg_with_relations(with_provenance=false)
  cleaned = GraphMERT.clean_kg(kg_no_prov; min_confidence=0.0, require_provenance=true)
  @test length(cleaned.relations) == 0
  kg_with_prov = _kg_with_relations(with_provenance=true)
  cleaned2 = GraphMERT.clean_kg(kg_with_prov; min_confidence=0.0, require_provenance=true)
  @test length(cleaned2.relations) == 3
end

@testset "clean_kg with policy struct" begin
  kg = _kg_with_relations(with_provenance=false)
  policy = GraphMERT.CleaningPolicy(; min_confidence=0.6, require_provenance=false)
  cleaned = GraphMERT.clean_kg(kg; policy=policy)
  @test length(cleaned.relations) == 2
end
