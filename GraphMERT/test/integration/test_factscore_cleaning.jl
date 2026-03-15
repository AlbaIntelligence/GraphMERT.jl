"""
Integration tests for evaluate_factscore(kg, reference) and clean_kg (US3).

- evaluate_factscore(kg, reference) returns FactualityScore when reference provided.
- clean_kg(kg; policy) returns new KG with reduced triples; metadata has before/after counts.
"""

using Test
using Dates
using GraphMERT

function _small_kg(; n_rels=2)
  entities = [
    GraphMERT.KnowledgeEntity("e$i", "E$i", "E$i", 0.9, GraphMERT.TextPosition(1, 1, 1, 1), Dict{String,Any}(), now())
    for i in 1:2
  ]
  relations = [
    GraphMERT.KnowledgeRelation("e1", "e2", "REL", 0.8 + i * 0.05, Dict{String,Any}(), now())
    for i in 1:n_rels
  ]
  GraphMERT.KnowledgeGraph(entities, relations, Dict{String,Any}(), now())
end

@testset "evaluate_factscore with reference" begin
  kg = _small_kg(n_rels=3)
  reference = _small_kg(n_rels=2)
  score = GraphMERT.evaluate_factscore(kg, reference; reference_id="ref1")
  @test score isa GraphMERT.FactualityScore
  @test score.total_triples == 3
  @test 0 <= score.correct_count <= 3
  @test score.reference_id == "ref1"
  @test 0.0 <= score.score <= 1.0
end

@testset "evaluate_factscore empty reference" begin
  kg = _small_kg(n_rels=2)
  ref_empty = GraphMERT.KnowledgeGraph(
    GraphMERT.KnowledgeEntity[],
    GraphMERT.KnowledgeRelation[],
    Dict{String,Any}(),
    now(),
  )
  score = GraphMERT.evaluate_factscore(kg, ref_empty)
  @test score.correct_count == 0
  @test score.total_triples == 2
  @test score.score == 0.0
end

@testset "clean_kg integration" begin
  kg = _small_kg(n_rels=4)
  cleaned = GraphMERT.clean_kg(kg; min_confidence=0.85, require_provenance=false)
  @test cleaned isa GraphMERT.KnowledgeGraph
  @test length(cleaned.relations) <= length(kg.relations)
  @test get(cleaned.metadata, "num_relations_before", 0) == 4
  @test get(cleaned.metadata, "num_relations_after", -1) == length(cleaned.relations)
end
